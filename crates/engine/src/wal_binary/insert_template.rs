//! Sealed v1 binary INSERT templates for the fixed-width INSERT-001 carrier.
//!
//! A template is created only from `PreparedInsertBatch`; it contains neither an allocator claim
//! nor a publish capability. Binding a checked proposed range consumes the template and patches
//! the payload plus its canonical operation body together, so no caller can mix either artifact
//! with separately supplied row ids, counts, or high-water metadata.

use super::*;
use crate::engine_canonical_operation::{
    ENGINE_OPERATION_CODEC_RESOLVED_BINARY, ENGINE_OPERATION_MAGIC,
};

/// A checked, side-effect-free proposal for the row identities one INSERT will later consume.
///
/// `allocator_high_water` is exclusive: binding `(first, count)` encodes exactly
/// `first..allocator_high_water`. Zero and the reserved maximum row identity are never valid
/// proposed row ids.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ProposedRowIdRange {
    first: u64,
    count: u32,
    allocator_high_water: u64,
}

impl ProposedRowIdRange {
    pub(crate) fn new(first: u64, count: u32) -> Result<Self, EngineError> {
        if first == 0 || count == 0 {
            return Err(EngineError::Durability(
                "proposed INSERT row-id range must be nonzero".to_string(),
            ));
        }
        let allocator_high_water = first.checked_add(u64::from(count)).ok_or_else(|| {
            EngineError::Durability("proposed INSERT row-id range overflows".to_string())
        })?;
        Ok(Self {
            first,
            count,
            allocator_high_water,
        })
    }

    fn row_id_at(&self, offset: usize) -> Result<u64, EngineError> {
        let offset = u64::try_from(offset).map_err(|_| {
            EngineError::Durability("proposed INSERT row-id offset overflows".to_string())
        })?;
        self.first.checked_add(offset).ok_or_else(|| {
            EngineError::Durability("proposed INSERT row-id range overflows".to_string())
        })
    }

    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    pub(crate) fn first(&self) -> u64 {
        self.first
    }

    pub(crate) fn allocator_high_water(&self) -> u64 {
        self.allocator_high_water
    }

    /// Materialize the exact identities sealed by this checked proposal without mutating the
    /// allocator. The typed residency plan owns this boxed proof through device apply, while the
    /// original proposal remains move-only inside the canonical envelope for exact allocation.
    pub(crate) fn exact_row_ids(&self) -> Result<Box<[u64]>, EngineError> {
        (0..self.count as usize)
            .map(|offset| self.row_id_at(offset))
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    }
}

/// An off-lock v1 INSERT payload and resolved-binary operation-body template with zero id slots.
///
/// The row-id offsets are private and both byte sequences are owned by this one token. It cannot
/// be assembled from raw bytes or metadata; `PreparedInsertBatch` is the only construction input.
pub(crate) struct PreparedBinaryInsertTemplate {
    payload: Arc<[u8]>,
    payload_row_id_offsets: Box<[usize]>,
    operation_body: Vec<u8>,
    operation_row_id_offsets: Box<[usize]>,
    count: u32,
}

impl PreparedBinaryInsertTemplate {
    pub(crate) fn from_sealed_batch(
        batch: &crate::prepared_insert_batch::PreparedInsertBatch,
    ) -> Result<Self, EngineError> {
        let count = batch.binary_insert_template_row_count();
        let table = batch.binary_insert_template_table_name();
        if count == 0 || table.len() > u16::MAX as usize {
            return Err(EngineError::Durability(
                "sealed INSERT batch cannot form a v1 binary template".to_string(),
            ));
        }
        let count_usize = count as usize;
        let mut payload = Vec::with_capacity(64 + count_usize * 48);
        payload.push(WAL_BINARY_TAG);
        payload.push(WAL_BINARY_VERSION);
        payload.push(OP_INSERT);
        payload.extend_from_slice(&(table.len() as u16).to_le_bytes());
        payload.extend_from_slice(table.as_bytes());
        payload.extend_from_slice(&count.to_le_bytes());
        let mut payload_row_id_offsets = Vec::with_capacity(count_usize);
        for row in 0..count_usize {
            payload_row_id_offsets.push(payload.len());
            payload.extend_from_slice(&0_u64.to_le_bytes());
            let row_length_offset = payload.len();
            payload.extend_from_slice(&0_u32.to_le_bytes());
            let row_start = payload.len();
            batch.append_binary_insert_template_row(row, &mut payload)?;
            let row_length = u32::try_from(payload.len() - row_start).map_err(|_| {
                EngineError::Durability(
                    "sealed INSERT row exceeds v1 binary WAL length".to_string(),
                )
            })?;
            payload[row_length_offset..row_start].copy_from_slice(&row_length.to_le_bytes());
        }
        let payload: Arc<[u8]> = Arc::from(payload);
        let payload_len = u64::try_from(payload.len()).map_err(|_| {
            EngineError::Durability("sealed INSERT payload length overflows".to_string())
        })?;
        let operation_prefix = ENGINE_OPERATION_MAGIC.len() + 1 + 3 + std::mem::size_of::<u64>();
        let mut operation_body = Vec::with_capacity(operation_prefix + payload.len());
        operation_body.extend_from_slice(ENGINE_OPERATION_MAGIC);
        operation_body.push(ENGINE_OPERATION_CODEC_RESOLVED_BINARY);
        operation_body.extend_from_slice(&[0; 3]);
        operation_body.extend_from_slice(&payload_len.to_le_bytes());
        operation_body.extend_from_slice(&payload);
        let operation_row_id_offsets = payload_row_id_offsets
            .iter()
            .map(|offset| {
                operation_prefix.checked_add(*offset).ok_or_else(|| {
                    EngineError::Durability("sealed INSERT operation offset overflows".to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            payload,
            payload_row_id_offsets: payload_row_id_offsets.into_boxed_slice(),
            operation_body,
            operation_row_id_offsets: operation_row_id_offsets.into_boxed_slice(),
            count,
        })
    }

    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Consume this zero-id template, patching the payload and its operation body in lock-step.
    pub(crate) fn bind(
        mut self,
        proposed: ProposedRowIdRange,
    ) -> Result<BoundBinaryInsert, EngineError> {
        if proposed.count != self.count
            || self.payload_row_id_offsets.len() != self.count as usize
            || self.operation_row_id_offsets.len() != self.count as usize
        {
            return Err(EngineError::Durability(
                "proposed INSERT row-id count does not match sealed template".to_string(),
            ));
        }
        // Validate every write and both zero placeholders before changing either owned buffer.
        // A rejected proposal therefore cannot leave a partially bound token behind.
        for (offset, operation_offset) in self
            .payload_row_id_offsets
            .iter()
            .zip(self.operation_row_id_offsets.iter())
        {
            let payload_slot = self
                .payload
                .get(*offset..offset.saturating_add(8))
                .ok_or_else(|| {
                    EngineError::Durability(
                        "sealed INSERT payload row-id offset is invalid".to_string(),
                    )
                })?;
            let operation_slot = self
                .operation_body
                .get(*operation_offset..operation_offset.saturating_add(8))
                .ok_or_else(|| {
                    EngineError::Durability(
                        "sealed INSERT operation-body row-id offset is invalid".to_string(),
                    )
                })?;
            if payload_slot != [0; 8] || operation_slot != [0; 8] {
                return Err(EngineError::Durability(
                    "sealed INSERT template has already been bound".to_string(),
                ));
            }
        }
        let payload = Arc::get_mut(&mut self.payload).ok_or_else(|| {
            EngineError::Durability(
                "sealed INSERT payload unexpectedly has an external owner before bind".to_string(),
            )
        })?;
        for (index, offset) in self.payload_row_id_offsets.iter().enumerate() {
            let row_id = proposed.row_id_at(index)?;
            payload[*offset..*offset + 8].copy_from_slice(&row_id.to_le_bytes());
        }
        for (index, offset) in self.operation_row_id_offsets.iter().enumerate() {
            let row_id = proposed.row_id_at(index)?;
            self.operation_body[*offset..*offset + 8].copy_from_slice(&row_id.to_le_bytes());
        }
        Ok(BoundBinaryInsert {
            payload: self.payload,
            operation_body: self.operation_body,
            proposed,
        })
    }
}

/// A bound v1 INSERT whose proposal bytes and canonical operation body carry the same row ids.
///
/// The token is consumed by `SealedCanonicalOperation`; callers may cheaply clone only the exact
/// proposal payload needed by the replication proposal before doing so.
pub(crate) struct BoundBinaryInsert {
    payload: Arc<[u8]>,
    operation_body: Vec<u8>,
    /// The exact move-only proposal that patched both owned byte artifacts. Later allocator
    /// application receives this original proof; it is never reconstructed from scalar metadata.
    proposed: ProposedRowIdRange,
}

impl BoundBinaryInsert {
    pub(crate) fn proposal_payload(&self) -> Arc<[u8]> {
        Arc::clone(&self.payload)
    }

    pub(crate) fn into_operation_body_and_proposed_range(self) -> (Vec<u8>, ProposedRowIdRange) {
        (self.operation_body, self.proposed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sealed_accounts_batch(
        sql: &str,
    ) -> (
        Engine,
        Command,
        crate::prepared_insert_batch::PreparedInsertBatch,
    ) {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        let command = parse_command(sql).expect("canonical accounts INSERT parses");
        let catalog = engine.catalog_snapshot();
        let delta = engine
            .prepare_dml(
                &command,
                engine.dml_read_snapshot(engine.committed_seq()),
                InsertPrepareValidation::WaveOffLock,
            )
            .expect("authoritative off-lock prepare accepts the workload");
        let batch = crate::prepared_insert_batch::try_prepare_fixed_insert_batch(
            &command,
            &delta,
            &catalog,
            catalog.commit_seq,
            None,
        )
        .expect("fixed-width accounts INSERT is eligible");
        (engine, command, batch)
    }

    fn accounts_workload(rows: usize) -> String {
        let mut sql = String::from("INSERT INTO accounts VALUES ");
        for row in 0..rows {
            if row != 0 {
                sql.push(',');
            }
            sql.push_str(&format!("({row}, {})", row * 10));
        }
        sql
    }

    #[test]
    fn exact_1000_row_accounts_template_binds_only_nonzero_proposed_ids() {
        let (_engine, _command, batch) = sealed_accounts_batch(&accounts_workload(1_000));
        let template = batch.binary_insert_template().unwrap();
        assert_eq!(template.count(), 1_000);
        assert!(ProposedRowIdRange::new(0, 1_000).is_err());
        let bound = template
            .bind(ProposedRowIdRange::new(41, 1_000).unwrap())
            .unwrap();
        let payload = bound.proposal_payload();
        let decoded = decode_binary_insert(&payload).unwrap();
        assert_eq!(decoded.table, "accounts");
        assert_eq!(decoded.rows.len(), 1_000);
        assert_eq!(decoded.rows[0], (41, "i:0|i:0".to_string()));
        assert_eq!(decoded.rows[999], (1_040, "i:999|i:9990".to_string()));
    }

    #[test]
    fn bound_payload_matches_legacy_encoder_for_signed_i32_values_and_ids() {
        let sql = "INSERT INTO accounts VALUES \
            (-2147483648,2147483647),(-1,0),(2147483647,-42)";
        let (_engine, command, batch) = sealed_accounts_batch(sql);
        let Command::Insert(insert) = &command else {
            unreachable!("test SQL is an INSERT");
        };
        let first = 8_000_001_u64;
        let ids_rows = insert
            .rows
            .iter()
            .enumerate()
            .map(|(offset, row)| (first + offset as u64, row.as_slice()))
            .collect::<Vec<_>>();
        let expected = try_encode_binary_insert("accounts", &ids_rows).unwrap();
        let bound = batch
            .binary_insert_template()
            .unwrap()
            .bind(ProposedRowIdRange::new(first, 3).unwrap())
            .unwrap();
        let payload = bound.proposal_payload();
        assert_eq!(payload.as_ref(), expected.as_slice());
        assert_eq!(
            decode_binary_insert(&payload).unwrap().rows,
            vec![
                (first, "i:-2147483648|i:2147483647".to_string()),
                (first + 1, "i:-1|i:0".to_string()),
                (first + 2, "i:2147483647|i:-42".to_string()),
            ]
        );
    }

    #[test]
    fn bound_operation_is_byte_identical_to_one_decode_canonical_path_and_recovers() {
        let sql = "INSERT INTO accounts VALUES (-7,70),(0,-1),(9,90)";
        let (engine, _command, batch) = sealed_accounts_batch(sql);
        let first = engine.read_state.mvcc.current_row_id();
        let bound = batch
            .binary_insert_template()
            .unwrap()
            .bind(ProposedRowIdRange::new(first, 3).unwrap())
            .unwrap();
        let proposal = bound.proposal_payload();
        let request_digest = gpu_db_wal::canonical_request_digest(sql.as_bytes());
        let commit_seq = engine.committed_seq() + 1;
        let scope = crate::engine_canonical_operation::LiveBinaryDecodeScope::begin();
        let prepared_bound = {
            let mut commit = engine.commit_state();
            Engine::canonical_wal_record_with_commit_bound_insert(
                &mut commit,
                2,
                commit_seq,
                0,
                bound,
                request_digest,
            )
            .unwrap()
        };
        assert_eq!(scope.count(), 0);
        let (bound_record, proposed_range) = prepared_bound.into_record_and_proposed_range();
        assert_eq!(proposed_range.first, first);
        assert_eq!(proposed_range.count, 3);
        assert_eq!(proposed_range.allocator_high_water, first + 3);
        let legacy_record = {
            let mut commit = engine.commit_state();
            Engine::canonical_wal_record_with_commit_outcome(
                &mut commit,
                2,
                commit_seq,
                0,
                &proposal,
                request_digest,
                gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
                3,
            )
            .unwrap()
        };
        assert_eq!(scope.count(), 1);
        assert_eq!(
            bound_record.as_wal_record().payload,
            legacy_record.as_wal_record().payload
        );
        let envelope =
            gpu_db_wal::decode_canonical_record_payload(&bound_record.as_wal_record().payload)
                .unwrap()
                .unwrap();
        assert_eq!(envelope.outcome.affected_rows, 3);
        assert_eq!(envelope.header.allocator_high_water, first + 3);
        let operation = &envelope.fragments[0].body;
        assert_eq!(&operation[..8], ENGINE_OPERATION_MAGIC);
        assert_eq!(operation[8], ENGINE_OPERATION_CODEC_RESOLVED_BINARY);
        assert_eq!(&operation[20..], proposal.as_ref());

        let mut records = engine.durable_wal_records();
        records.push(bound_record.into_wal_record());
        let recovered = Engine::recover_from_durable_wal(&records).unwrap();
        assert_eq!(
            recovered
                .execute_relational_select_text("SELECT id, balance FROM accounts ORDER BY id")
                .unwrap()
                .rows,
            vec![
                vec![SqlValue::Int4(-7), SqlValue::Int4(70)],
                vec![SqlValue::Int4(0), SqlValue::Int4(-1)],
                vec![SqlValue::Int4(9), SqlValue::Int4(90)],
            ]
        );
    }

    #[test]
    fn range_overflow_and_template_count_mismatch_reject_before_binding() {
        let (_engine, _command, batch) =
            sealed_accounts_batch("INSERT INTO accounts VALUES (1,10),(2,20),(3,30)");
        assert!(ProposedRowIdRange::new(u64::MAX, 1).is_err());
        assert!(batch
            .binary_insert_template()
            .unwrap()
            .bind(ProposedRowIdRange::new(1, 2).unwrap())
            .is_err());
    }
}
