//! One-decode sealing of the live canonical WAL operation intrinsic fields.
//!
//! Recovery deliberately reconstructs and validates the same fields from durable bytes.  This
//! module only removes repeated decoding while a live operation is being framed under the commit
//! boundary; it is not a recovery cache or a second durable authority.

use super::*;

pub(super) const ENGINE_OPERATION_MAGIC: &[u8; 8] = b"GPUDBOP1";
pub(super) const ENGINE_OPERATION_CODEC_RESOLVED_BINARY: u8 = 2;
pub(super) const ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2: u8 = 4;

/// Exact fixed-width prefix carried by every canonical engine operation body.  Planning-only
/// accounting uses this owner rather than duplicating the codec layout.
const ENGINE_OPERATION_BODY_PREFIX_BYTES: usize =
    ENGINE_OPERATION_MAGIC.len() + std::mem::size_of::<u8>() + 3 + std::mem::size_of::<u64>();

pub(crate) fn canonical_operation_body_len(payload_len: usize) -> Result<usize, EngineError> {
    let _ = u64::try_from(payload_len).map_err(|_| {
        EngineError::Durability("canonical WAL operation length overflow".to_string())
    })?;
    ENGINE_OPERATION_BODY_PREFIX_BYTES
        .checked_add(payload_len)
        .ok_or_else(|| {
            EngineError::Durability("canonical WAL operation length overflow".to_string())
        })
}

enum AffectedRowsDefault {
    Known(u64),
    ExactOutcomeRequired(&'static str),
}

/// Intrinsic canonical fields derived from exactly one decoded live payload.
///
/// Fields stay private so no caller can mix metadata sourced from another payload into the
/// canonical envelope.  The caller owns only the externally observed outcome count for routes
/// whose affected rows are determined by applied GPU work.
pub(super) struct SealedCanonicalOperation {
    body: Vec<u8>,
    kind: gpu_db_wal::CanonicalFragmentKind,
    table_block_count: u32,
    allocator_high_water: u64,
    affected_rows_default: AffectedRowsDefault,
}

impl SealedCanonicalOperation {
    pub(super) fn from_live_payload(payload: &[u8], txn_id: TxnId) -> Result<Self, EngineError> {
        if payload.first() == Some(&WAL_BINARY_TAG) {
            let record = decode_live_binary_record(payload)?;
            return Self::from_binary_payload(payload, &record, txn_id);
        }
        Self::from_typed_command(payload)
    }

    /// Seal a just-built transaction record without sending its newly encoded bytes through the
    /// historical binary decoder. The live terminal still validates the exact metadata that the
    /// decoder would derive; recovery remains the sole reader of the encoded transaction bytes.
    pub(super) fn from_live_binary_transaction(
        payload: &[u8],
        record: &crate::wal_binary::BinaryTransactionRecord,
        txn_id: TxnId,
    ) -> Result<Self, EngineError> {
        if record.sequence_value_references.iter().any(|reference| {
            reference.parent_txn_id != txn_id || reference.transition_txn_id == txn_id
        }) {
            return Err(EngineError::Durability(
                "sequence-reference parent does not match canonical transaction identity"
                    .to_string(),
            ));
        }
        let allocator_high_water = record
            .mutations
            .iter()
            .map(|mutation| match mutation {
                BinaryTransactionMutation::Insert { row_id, .. }
                | BinaryTransactionMutation::Update { row_id, .. }
                | BinaryTransactionMutation::Delete { row_id, .. } => {
                    row_id.checked_add(1).ok_or_else(|| {
                        EngineError::Durability(
                            "canonical WAL references the reserved maximum row identity"
                                .to_string(),
                        )
                    })
                }
            })
            .try_fold(record.allocator_high_water, |high, next| {
                next.map(|next| high.max(next))
            })?;
        let kind = if !record.catalog_commands.is_empty() {
            gpu_db_wal::CanonicalFragmentKind::CatalogMutation
        } else if !record.table_resets.is_empty() {
            gpu_db_wal::CanonicalFragmentKind::TableReset
        } else {
            gpu_db_wal::CanonicalFragmentKind::RowMutation
        };
        let ordered = !record.operation_order.is_empty();
        let tables = record
            .catalog_commands
            .iter()
            .filter_map(|operation| match &operation.command {
                Command::CreateTable(create) if ordered => Some(create.table.as_str()),
                _ => None,
            })
            .chain(
                record
                    .operation_order
                    .iter()
                    .filter_map(|operation| ordered.then(|| operation.table()).flatten()),
            )
            .chain(record.table_resets.iter().map(|reset| reset.table.as_str()))
            .chain(record.mutations.iter().map(|mutation| match mutation {
                BinaryTransactionMutation::Insert { table, .. }
                | BinaryTransactionMutation::Update { table, .. }
                | BinaryTransactionMutation::Delete { table, .. } => table.as_str(),
            }))
            .collect::<BTreeSet<_>>();
        let table_block_count = u32::try_from(tables.len()).map_err(|_| {
            EngineError::Durability(
                "canonical transaction table-block count exceeds u32".to_string(),
            )
        })?;
        let affected_rows = u64::try_from(record.mutations.len()).map_err(|_| {
            EngineError::Durability(
                "transaction mutation count exceeds canonical affected-row framing".to_string(),
            )
        })?;
        Ok(Self {
            body: encode_operation_body(ENGINE_OPERATION_CODEC_RESOLVED_BINARY, payload)?,
            kind,
            table_block_count,
            allocator_high_water,
            affected_rows_default: AffectedRowsDefault::Known(affected_rows),
        })
    }

    fn from_binary_payload(
        payload: &[u8],
        record: &BinaryWalRecord,
        txn_id: TxnId,
    ) -> Result<Self, EngineError> {
        validate_binary_transaction_id(record, txn_id)?;
        let kind = binary_fragment_kind(record);
        let affected_rows_default = binary_affected_rows_default(record);
        Ok(Self {
            body: encode_operation_body(ENGINE_OPERATION_CODEC_RESOLVED_BINARY, payload)?,
            kind,
            table_block_count: binary_table_block_count(record, kind)?,
            allocator_high_water: binary_allocator_high_water(record)?,
            affected_rows_default,
        })
    }

    fn from_typed_command(payload: &[u8]) -> Result<Self, EngineError> {
        let command = Engine::decode_engine_command(payload)?.ok_or_else(|| {
            EngineError::Durability(
                "canonical WAL operation is neither a resolved binary record nor a typed command"
                    .to_string(),
            )
        })?;
        let kind = command_fragment_kind(&command);
        let canonical_payload = serde_json::to_vec(&command).map_err(|error| {
            EngineError::Durability(format!(
                "canonical WAL typed command encode failed: {error}"
            ))
        })?;
        let affected_rows_default = match command {
            Command::Insert(insert) => AffectedRowsDefault::Known(insert.rows.len() as u64),
            Command::Delete(_) | Command::Update(_) => AffectedRowsDefault::ExactOutcomeRequired(
                "unresolved text UPDATE/DELETE requires an exact outcome marker",
            ),
            _ => AffectedRowsDefault::Known(0),
        };
        Ok(Self {
            body: encode_operation_body(
                ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
                &canonical_payload,
            )?,
            kind,
            table_block_count: u32::from(matches!(
                kind,
                gpu_db_wal::CanonicalFragmentKind::RowMutation
                    | gpu_db_wal::CanonicalFragmentKind::TableReset
                    | gpu_db_wal::CanonicalFragmentKind::TableRewrite
            )),
            allocator_high_water: 0,
            affected_rows_default,
        })
    }

    pub(super) fn kind(&self) -> gpu_db_wal::CanonicalFragmentKind {
        self.kind
    }

    pub(super) fn table_block_count(&self) -> u32 {
        self.table_block_count
    }

    pub(super) fn allocator_high_water(&self) -> u64 {
        self.allocator_high_water
    }

    pub(super) fn operation_digest(&self) -> gpu_db_wal::CanonicalDigest {
        gpu_db_wal::canonical_request_digest(&self.body)
    }

    pub(super) fn affected_rows_or_default(
        &self,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        outcome_rows: Option<u64>,
    ) -> Result<u64, EngineError> {
        match (outcome_rows, &self.affected_rows_default) {
            (Some(rows), AffectedRowsDefault::Known(expected))
                if matches!(
                    outcome_kind,
                    gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
                        | gpu_db_wal::CanonicalOutcomeKind::CommitNoOp
                ) && rows != *expected =>
            {
                Err(EngineError::Durability(
                    "canonical WAL outcome affected-row count conflicts with sealed operation"
                        .to_string(),
                ))
            }
            (Some(rows), _) => Ok(rows),
            (None, AffectedRowsDefault::Known(rows)) => Ok(*rows),
            (None, AffectedRowsDefault::ExactOutcomeRequired(message)) => {
                Err(EngineError::Durability((*message).to_string()))
            }
        }
    }

    pub(super) fn into_fragment(self) -> gpu_db_wal::CanonicalFragment {
        gpu_db_wal::CanonicalFragment {
            kind: self.kind,
            body: self.body,
        }
    }
}

fn encode_operation_body(codec: u8, payload: &[u8]) -> Result<Vec<u8>, EngineError> {
    let len = u64::try_from(payload.len()).map_err(|_| {
        EngineError::Durability("canonical WAL operation length overflow".to_string())
    })?;
    let expected_len = canonical_operation_body_len(payload.len())?;
    let mut body = Vec::with_capacity(expected_len);
    body.extend_from_slice(ENGINE_OPERATION_MAGIC);
    body.push(codec);
    body.extend_from_slice(&[0; 3]);
    body.extend_from_slice(&len.to_le_bytes());
    body.extend_from_slice(payload);
    debug_assert_eq!(body.len(), expected_len);
    Ok(body)
}

fn binary_fragment_kind(record: &BinaryWalRecord) -> gpu_db_wal::CanonicalFragmentKind {
    match record {
        BinaryWalRecord::Transaction(record) if !record.catalog_commands.is_empty() => {
            gpu_db_wal::CanonicalFragmentKind::CatalogMutation
        }
        BinaryWalRecord::Transaction(record) if !record.table_resets.is_empty() => {
            gpu_db_wal::CanonicalFragmentKind::TableReset
        }
        BinaryWalRecord::SequenceValueTransition(_) => {
            gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition
        }
        _ => gpu_db_wal::CanonicalFragmentKind::RowMutation,
    }
}

fn command_fragment_kind(command: &Command) -> gpu_db_wal::CanonicalFragmentKind {
    match command {
        Command::TruncateTable(_) => gpu_db_wal::CanonicalFragmentKind::TableReset,
        Command::RefreshMaterializedView(_) => gpu_db_wal::CanonicalFragmentKind::TableRewrite,
        Command::SequenceNextVal(_) | Command::SequenceSetVal(_) => {
            gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition
        }
        Command::SetKv { .. }
        | Command::DeleteKv { .. }
        | Command::Insert(_)
        | Command::Delete(_)
        | Command::Update(_) => gpu_db_wal::CanonicalFragmentKind::RowMutation,
        _ => gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
    }
}

fn binary_allocator_high_water(record: &BinaryWalRecord) -> Result<u64, EngineError> {
    let referenced_next = |row_id: u64| {
        row_id.checked_add(1).ok_or_else(|| {
            EngineError::Durability(
                "canonical WAL references the reserved maximum row identity".to_string(),
            )
        })
    };
    match record {
        BinaryWalRecord::Insert(record) => record
            .rows
            .iter()
            .map(|(row_id, _)| referenced_next(*row_id))
            .try_fold(0, |high, next| next.map(|next| high.max(next))),
        BinaryWalRecord::DeleteByKey(_) => Ok(0),
        BinaryWalRecord::UpdateByKey(record) => referenced_next(record.new_row_id),
        BinaryWalRecord::Transaction(record) => record
            .mutations
            .iter()
            .map(|mutation| match mutation {
                BinaryTransactionMutation::Insert { row_id, .. }
                | BinaryTransactionMutation::Update { row_id, .. }
                | BinaryTransactionMutation::Delete { row_id, .. } => referenced_next(*row_id),
            })
            .try_fold(record.allocator_high_water, |high, next| {
                next.map(|next| high.max(next))
            }),
        BinaryWalRecord::SequenceValueTransition(_) => Ok(0),
    }
}

fn binary_affected_rows_default(record: &BinaryWalRecord) -> AffectedRowsDefault {
    match record {
        BinaryWalRecord::Insert(record) => AffectedRowsDefault::Known(record.rows.len() as u64),
        BinaryWalRecord::Transaction(record) => {
            AffectedRowsDefault::Known(record.mutations.len() as u64)
        }
        BinaryWalRecord::SequenceValueTransition(_) => AffectedRowsDefault::Known(0),
        BinaryWalRecord::DeleteByKey(_) | BinaryWalRecord::UpdateByKey(_) => {
            AffectedRowsDefault::ExactOutcomeRequired(
                "unresolved by-key WAL requires an exact GPU outcome marker",
            )
        }
    }
}

fn binary_table_block_count(
    record: &BinaryWalRecord,
    operation_kind: gpu_db_wal::CanonicalFragmentKind,
) -> Result<u32, EngineError> {
    let BinaryWalRecord::Transaction(record) = record else {
        return Ok(u32::from(matches!(
            operation_kind,
            gpu_db_wal::CanonicalFragmentKind::RowMutation
                | gpu_db_wal::CanonicalFragmentKind::TableReset
                | gpu_db_wal::CanonicalFragmentKind::TableRewrite
        )));
    };
    // Opcodes 4--9 predate the ordered statement vector and their acknowledged canonical
    // envelopes counted only reset/mutation output tables. Preserve that exact header
    // interpretation for upgrade replay. Opcodes 10/11 always decode a non-empty operation
    // order and additionally cover catalog-only/private tables.
    let ordered = !record.operation_order.is_empty();
    let tables = record
        .catalog_commands
        .iter()
        .filter_map(|operation| match &operation.command {
            Command::CreateTable(create) if ordered => Some(create.table.as_str()),
            _ => None,
        })
        .chain(record.operation_order.iter().filter_map(|operation| {
            if ordered {
                operation.table()
            } else {
                None
            }
        }))
        .chain(record.table_resets.iter().map(|reset| reset.table.as_str()))
        .chain(record.mutations.iter().map(|mutation| match mutation {
            BinaryTransactionMutation::Insert { table, .. }
            | BinaryTransactionMutation::Update { table, .. }
            | BinaryTransactionMutation::Delete { table, .. } => table.as_str(),
        }))
        .collect::<BTreeSet<_>>();
    u32::try_from(tables.len()).map_err(|_| {
        EngineError::Durability("canonical transaction table-block count exceeds u32".to_string())
    })
}

fn validate_binary_transaction_id(
    record: &BinaryWalRecord,
    outer_txn_id: TxnId,
) -> Result<(), EngineError> {
    match record {
        BinaryWalRecord::SequenceValueTransition(record) if record.transition_txn_id != outer_txn_id => {
            Err(EngineError::Durability(format!(
                "sequence transition identity {} does not match canonical transaction identity {outer_txn_id}",
                record.transition_txn_id
            )))
        }
        BinaryWalRecord::Transaction(record)
            if record.sequence_value_references.iter().any(|reference| {
                reference.parent_txn_id != outer_txn_id || reference.transition_txn_id == outer_txn_id
            }) =>
        {
            Err(EngineError::Durability(
                "sequence-reference parent does not match canonical transaction identity".to_string(),
            ))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
thread_local! {
    static LIVE_BINARY_DECODE_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) struct LiveBinaryDecodeScope {
    prior: u64,
}

#[cfg(test)]
impl LiveBinaryDecodeScope {
    pub(super) fn begin() -> Self {
        let prior = LIVE_BINARY_DECODE_COUNT.with(|count| {
            let prior = count.get();
            count.set(0);
            prior
        });
        Self { prior }
    }

    pub(super) fn count(&self) -> u64 {
        LIVE_BINARY_DECODE_COUNT.with(std::cell::Cell::get)
    }
}

#[cfg(test)]
impl Drop for LiveBinaryDecodeScope {
    fn drop(&mut self) {
        LIVE_BINARY_DECODE_COUNT.with(|count| count.set(self.prior));
    }
}

fn decode_live_binary_record(payload: &[u8]) -> Result<BinaryWalRecord, EngineError> {
    #[cfg(test)]
    LIVE_BINARY_DECODE_COUNT.with(|count| count.set(count.get() + 1));
    decode_binary_record(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_operation_body_length_helper_matches_real_encoder() {
        for payload in [
            Vec::new(),
            vec![0xa5],
            vec![0x5a; 4_096],
            vec![0; 16 * 1024 * 1024 - 20],
        ] {
            let encoded = encode_operation_body(ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &payload)
                .expect("test payload fits the operation codec");
            assert_eq!(
                canonical_operation_body_len(payload.len()).unwrap(),
                encoded.len()
            );
        }
        assert!(canonical_operation_body_len(usize::MAX).is_err());
    }

    fn identity() -> gpu_db_wal::CanonicalIdentity {
        gpu_db_wal::CanonicalIdentity {
            database_id: [0x11; 16],
            cluster_id: [0x22; 16],
            timeline_id: [0x33; 16],
            format_epoch: 1,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn legacy_binary_reference(
        payload: &[u8],
        identity: gpu_db_wal::CanonicalIdentity,
        catalog_epoch: u64,
        catalog_digest: gpu_db_wal::CanonicalDigest,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        request_digest: gpu_db_wal::CanonicalDigest,
        affected_rows: u64,
        table_block_count: u32,
        allocator_high_water: u64,
    ) -> gpu_db_wal::PreparedCanonicalWalRecord {
        let mut operation_body = Vec::with_capacity(20 + payload.len());
        operation_body.extend_from_slice(b"GPUDBOP1");
        operation_body.push(2);
        operation_body.extend_from_slice(&[0; 3]);
        operation_body.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        operation_body.extend_from_slice(payload);
        let operation = gpu_db_wal::CanonicalFragment {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: operation_body,
        };
        let mut status = Vec::with_capacity(100);
        status.extend_from_slice(b"GPUDBSTATUS1");
        status.extend_from_slice(&identity.database_id);
        status.extend_from_slice(&identity.timeline_id);
        status.extend_from_slice(&txn_id.to_le_bytes());
        status.extend_from_slice(&request_digest);
        status.push(1);
        status.extend_from_slice(&[0; 7]);
        status.extend_from_slice(&u64::MAX.to_le_bytes());
        let header = gpu_db_wal::CanonicalPreApplyHeader {
            identity,
            leader_epoch: 1,
            commit_seq,
            stable_transaction_id: txn_id,
            request_digest,
            isolation: gpu_db_wal::CanonicalIsolation::ReadCommitted,
            flags: u32::from(gpu_db_wal::CanonicalFragmentKind::RowMutation as u16),
            catalog_before_epoch: catalog_epoch,
            catalog_after_epoch: catalog_epoch,
            catalog_before_digest: catalog_digest,
            catalog_after_digest: catalog_digest,
            operation_count: 2,
            table_block_count,
            allocator_high_water,
        };
        let outcome = gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            affected_rows,
            sqlstate: None,
            constraint_id: 0,
            target_digest: gpu_db_wal::canonical_request_digest(&operation.body),
            returning_digest: [0; 32],
        };
        gpu_db_wal::encode_canonical_envelope(
            gpu_db_wal::CanonicalPhysicalRange {
                log_epoch: 1,
                lane_id,
                segment_id: commit_seq,
                first_frame_ordinal: 0,
            },
            &header,
            &[
                operation,
                gpu_db_wal::CanonicalFragment {
                    kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
                    body: status,
                },
            ],
            &outcome,
        )
        .unwrap()
        .into_prepared_record(txn_id)
        .unwrap()
    }

    #[test]
    fn live_binary_seal_is_one_decode_and_byte_identical_to_legacy_reference() {
        let row_values = [vec![SqlValue::Int4(7)], vec![SqlValue::Int4(11)]];
        let rows = [
            (7, row_values[0].as_slice()),
            (11, row_values[1].as_slice()),
        ];
        let insert: Arc<[u8]> =
            Arc::from(encode_historical_binary_insert_fixture("sealed_insert", &rows).unwrap());
        let identity = identity();
        let catalog_digest = [0x44; 32];
        let request_digest = [0x55; 32];
        let scope = LiveBinaryDecodeScope::begin();
        let actual = Engine::canonical_wal_record_with_boundary_and_outcome(
            identity,
            9,
            catalog_digest,
            91,
            17,
            3,
            &insert,
            request_digest,
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            2,
        )
        .unwrap();
        assert_eq!(scope.count(), 1);
        let expected = legacy_binary_reference(
            &insert,
            identity,
            9,
            catalog_digest,
            91,
            17,
            3,
            request_digest,
            2,
            1,
            12,
        );
        assert_eq!(actual.as_wal_record(), expected.as_wal_record());

        let envelope = gpu_db_wal::decode_canonical_record_payload(&actual.as_wal_record().payload)
            .unwrap()
            .unwrap();
        assert_eq!(envelope.header.request_digest, request_digest);
        assert_ne!(
            request_digest,
            gpu_db_wal::canonical_request_digest(&insert)
        );
        assert_eq!(envelope.header.table_block_count, 1);
        assert_eq!(envelope.header.allocator_high_water, 12);
        assert_eq!(
            envelope.fragments[0].kind,
            gpu_db_wal::CanonicalFragmentKind::RowMutation
        );
        // Durable replay decodes the envelope payload anew; it does not receive this live seal.
        assert_eq!(
            Engine::decode_engine_operation(&envelope.fragments[0].body)
                .unwrap()
                .as_ref(),
            insert.as_ref()
        );
    }

    #[test]
    fn transaction_table_metadata_is_sealed_once_and_matches_legacy_reference() {
        let transaction = BinaryTransactionRecord {
            catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
            allocator_high_water: 3,
            catalog_commands: Vec::new(),
            created_table_identities: BTreeMap::new(),
            created_table_index_identities: BTreeMap::new(),
            catalog_output: None,
            view_operations: Vec::new(),
            view_lifecycle_operations: Vec::new(),
            index_lifecycle_operations: Vec::new(),
            sequence_lifecycle_operations: Vec::new(),
            sequence_reset_operations: Vec::new(),
            sequence_advances_by_oid: BTreeMap::new(),
            operation_order: Vec::new(),
            statement_digests: Vec::new(),
            sequence_input_oids: BTreeMap::new(),
            sequence_value_references: Vec::new(),
            table_resets: Vec::new(),
            sequence_advances: BTreeMap::new(),
            table_identities: BTreeMap::new(),
            mutations: vec![
                BinaryTransactionMutation::Insert {
                    table: "sealed_a".to_string(),
                    row_id: 21,
                    row_encoded: "i:7".to_string(),
                },
                BinaryTransactionMutation::Delete {
                    table: "sealed_b".to_string(),
                    row_id: 30,
                    old_row_encoded: "i:11".to_string(),
                },
            ],
        };
        let payload: Arc<[u8]> = Arc::from(try_encode_binary_transaction(&transaction).unwrap());
        let identity = identity();
        let scope = LiveBinaryDecodeScope::begin();
        let actual = Engine::canonical_wal_record_with_boundary_and_outcome(
            identity,
            4,
            [0x66; 32],
            92,
            18,
            4,
            &payload,
            [0x77; 32],
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            2,
        )
        .unwrap();
        assert_eq!(scope.count(), 1);
        let expected = legacy_binary_reference(
            &payload, identity, 4, [0x66; 32], 92, 18, 4, [0x77; 32], 2, 2, 31,
        );
        assert_eq!(actual.as_wal_record(), expected.as_wal_record());
    }

    #[test]
    fn live_binary_seal_rejects_identity_mismatch_outcome_mismatch_and_malformed_payload() {
        let operation = BinarySequenceValueOperation::NextVal;
        let parent_request_digest = [0x81; 32];
        let transition = BinarySequenceValueTransitionRecord {
            transition_txn_id: 101,
            parent_txn_id: 100,
            parent_autocommit: false,
            statement_ordinal: 2,
            expression_ordinal: 3,
            parent_request_digest,
            input_digest: sequence_value_input_digest(SequenceValueInput {
                parent_txn_id: 100,
                parent_autocommit: false,
                statement_ordinal: 2,
                expression_ordinal: 3,
                parent_request_digest,
                source_name: "sealed_sequence",
                operation,
                set_value: None,
            }),
            sequence_oid: 41,
            source_name: "sealed_sequence".to_string(),
            effective_name: "sealed_sequence".to_string(),
            published_name: "sealed_sequence".to_string(),
            base_catalog_generation: 7,
            prior_last_value: 9,
            prior_is_called: true,
            new_last_value: 10,
            new_is_called: true,
            returned_value: 10,
            private_descriptor_digest: None,
            operation,
        };
        let sequence: Arc<[u8]> = Arc::from(encode_sequence_value_transition(&transition).unwrap());
        let scope = LiveBinaryDecodeScope::begin();
        let error = Engine::canonical_wal_record_with_boundary_and_outcome(
            identity(),
            0,
            [0; 32],
            102,
            1,
            0,
            &sequence,
            [0x82; 32],
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("sequence transition identity"));
        assert_eq!(scope.count(), 1);

        let values = [SqlValue::Int4(1)];
        let insert: Arc<[u8]> = Arc::from(
            encode_historical_binary_insert_fixture("sealed_outcome", &[(1, values.as_slice())])
                .unwrap(),
        );
        let error = Engine::canonical_wal_record_with_boundary_and_outcome(
            identity(),
            0,
            [0; 32],
            103,
            1,
            0,
            &insert,
            [0x83; 32],
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            0,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("conflicts with sealed operation"));

        let malformed: Arc<[u8]> = Arc::from(&[WAL_BINARY_TAG][..]);
        let scope = LiveBinaryDecodeScope::begin();
        assert!(
            Engine::canonical_wal_record_with_boundary_and_request_digest(
                identity(),
                0,
                [0; 32],
                104,
                1,
                0,
                &malformed,
                [0x84; 32],
            )
            .is_err()
        );
        assert_eq!(scope.count(), 1);
    }
}
