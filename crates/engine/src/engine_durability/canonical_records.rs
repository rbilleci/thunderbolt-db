//! Canonical live-WAL framing and deterministic pre-WAL outcome selection.
//!
//! This private leaf owns only construction of one canonical record from an already prepared
//! operation. Recovery validation and replay remain in the parent durability controller.

use super::*;

impl Engine {
    pub(crate) fn canonical_wal_record(
        commit: &mut CommitState,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
    ) -> Result<gpu_db_wal::PreparedCanonicalWalRecord, EngineError> {
        Self::canonical_wal_record_with_commit_request_digest(
            commit,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            gpu_db_wal::canonical_request_digest(payload),
        )
    }

    /// Frame a just-built resolved transaction record without decoding its live binary payload.
    /// The caller still owns the source record, so this is a production-only handoff; historical
    /// WAL bytes continue to enter through the decoder during recovery.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn canonical_wal_record_from_live_transaction(
        commit: &mut CommitState,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        transaction: &crate::wal_binary::BinaryTransactionRecord,
        isolation: gpu_db_wal::CanonicalIsolation,
        request_digest: gpu_db_wal::CanonicalDigest,
        affected_rows: u64,
    ) -> Result<gpu_db_wal::PreparedCanonicalWalRecord, EngineError> {
        let (catalog_epoch, catalog_digest) = Self::canonical_catalog_boundary(
            commit.canonical_identity,
            commit.wal.canonical_catalog_tail()?,
        )?;
        let operation =
            SealedCanonicalOperation::from_live_binary_transaction(payload, transaction, txn_id)?;
        Self::canonical_wal_record_from_sealed_operation(
            commit.canonical_identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            operation,
            request_digest,
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            Some(affected_rows),
            isolation,
        )
    }

    pub(crate) fn canonical_affected_rows(payload: &[u8]) -> Result<u64, EngineError> {
        if payload.first() == Some(&WAL_BINARY_TAG) {
            match decode_binary_record(payload)? {
                crate::wal_binary::BinaryWalRecord::Insert(record) => Ok(record.rows.len() as u64),
                crate::wal_binary::BinaryWalRecord::Transaction(record) => {
                    Ok(record.mutations.len() as u64)
                }
                crate::wal_binary::BinaryWalRecord::SequenceValueTransition(_) => Ok(0),
                crate::wal_binary::BinaryWalRecord::DeleteByKey(_)
                | crate::wal_binary::BinaryWalRecord::UpdateByKey(_) => {
                    Err(EngineError::Durability(
                        "unresolved by-key WAL requires an exact GPU outcome marker".to_string(),
                    ))
                }
            }
        } else {
            match Self::decode_engine_command(payload)?.ok_or_else(|| {
                EngineError::Durability("canonical WAL command has no typed operation".to_string())
            })? {
                Command::Insert(insert) => Ok(insert.rows.len() as u64),
                Command::Delete(_) | Command::Update(_) => Err(EngineError::Durability(
                    "unresolved text UPDATE/DELETE requires an exact outcome marker".to_string(),
                )),
                _ => Ok(0),
            }
        }
    }

    pub(super) fn canonical_table_block_count(
        payload: &[u8],
        operation_kind: gpu_db_wal::CanonicalFragmentKind,
    ) -> Result<u32, EngineError> {
        if payload.first() == Some(&WAL_BINARY_TAG) {
            if let crate::wal_binary::BinaryWalRecord::Transaction(record) =
                decode_binary_record(payload)?
            {
                // Opcodes 4--9 predate the ordered statement vector and their acknowledged
                // canonical envelopes counted only reset/mutation output tables. Preserve that
                // exact header interpretation for upgrade replay. Opcodes 10/11 always decode a
                // non-empty operation order and additionally cover catalog-only/private tables.
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
                        crate::wal_binary::BinaryTransactionMutation::Insert { table, .. }
                        | crate::wal_binary::BinaryTransactionMutation::Update { table, .. }
                        | crate::wal_binary::BinaryTransactionMutation::Delete { table, .. } => {
                            table.as_str()
                        }
                    }))
                    .collect::<BTreeSet<_>>();
                return u32::try_from(tables.len()).map_err(|_| {
                    EngineError::Durability(
                        "canonical transaction table-block count exceeds u32".to_string(),
                    )
                });
            }
        }
        Ok(u32::from(matches!(
            operation_kind,
            gpu_db_wal::CanonicalFragmentKind::RowMutation
                | gpu_db_wal::CanonicalFragmentKind::TableReset
                | gpu_db_wal::CanonicalFragmentKind::TableRewrite
        )))
    }

    pub(crate) fn canonical_wal_record_with_commit_request_digest(
        commit: &mut CommitState,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<gpu_db_wal::PreparedCanonicalWalRecord, EngineError> {
        let (catalog_epoch, catalog_digest) = Self::canonical_catalog_boundary(
            commit.canonical_identity,
            commit.wal.canonical_catalog_tail()?,
        )?;
        Self::canonical_wal_record_with_boundary_and_request_digest(
            commit.canonical_identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            request_digest,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn canonical_wal_record_with_boundary_and_request_digest(
        identity: gpu_db_wal::CanonicalIdentity,
        catalog_epoch: u64,
        catalog_digest: gpu_db_wal::CanonicalDigest,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<gpu_db_wal::PreparedCanonicalWalRecord, EngineError> {
        Self::canonical_wal_record_with_boundary_and_optional_outcome_isolation(
            identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            request_digest,
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            None,
            gpu_db_wal::CanonicalIsolation::ReadCommitted,
        )
    }

    /// Resolve the exact deterministic outcome for the serialized path while its commit lock is
    /// held and before sequence/WAL assignment. The returned delta is intentionally discarded:
    /// apply re-runs against the same protected committed boundary, and recovery compares that
    /// result with this durable marker. A route that cannot be resolved here is refused pre-WAL.
    pub(crate) fn canonical_serialized_outcome(
        &self,
        payload: &[u8],
        commit_seq: Index,
    ) -> Result<(gpu_db_wal::CanonicalOutcomeKind, u64), EngineError> {
        let rows = if payload.first() == Some(&WAL_BINARY_TAG) {
            match decode_binary_record(payload)? {
                crate::wal_binary::BinaryWalRecord::Insert(record) => record.rows.len() as u64,
                crate::wal_binary::BinaryWalRecord::Transaction(record) => {
                    record.mutations.len() as u64
                }
                crate::wal_binary::BinaryWalRecord::SequenceValueTransition(_) => 0,
                crate::wal_binary::BinaryWalRecord::DeleteByKey(_)
                | crate::wal_binary::BinaryWalRecord::UpdateByKey(_) => {
                    return Err(EngineError::Durability(
                        "serialized by-key WAL requires an applied GPU outcome".to_string(),
                    ));
                }
            }
        } else {
            let Some(command) = Self::decode_engine_command(payload)? else {
                return Err(EngineError::Durability(
                    "serialized canonical operation is not decodable".to_string(),
                ));
            };
            let snapshot = DmlReadSnapshot {
                commit_seq,
                next_row_id: self.read_state.mvcc.current_row_id(),
            };
            match command {
                Command::Insert(insert) => self
                    .prepare_insert(&insert, snapshot, None)?
                    .rows_affected(),
                Command::Delete(delete) => self.prepare_delete(&delete, snapshot)?.rows_affected(),
                Command::Update(update) => self.prepare_update(&update, snapshot)?.rows_affected(),
                _ => 0,
            }
        };
        Ok((
            if rows == 0
                && matches!(
                    Self::decode_engine_command(payload)?,
                    Some(Command::Insert(_) | Command::Delete(_) | Command::Update(_))
                )
            {
                gpu_db_wal::CanonicalOutcomeKind::CommitNoOp
            } else {
                gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
            },
            rows,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn canonical_wal_record_with_commit_outcome(
        commit: &mut CommitState,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        affected_rows: u64,
    ) -> Result<gpu_db_wal::PreparedCanonicalWalRecord, EngineError> {
        let (catalog_epoch, catalog_digest) = Self::canonical_catalog_boundary(
            commit.canonical_identity,
            commit.wal.canonical_catalog_tail()?,
        )?;
        Self::canonical_wal_record_with_boundary_and_outcome(
            commit.canonical_identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            request_digest,
            outcome_kind,
            affected_rows,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn canonical_wal_record_with_boundary_and_outcome(
        identity: gpu_db_wal::CanonicalIdentity,
        catalog_epoch: u64,
        catalog_digest: gpu_db_wal::CanonicalDigest,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        affected_rows: u64,
    ) -> Result<gpu_db_wal::PreparedCanonicalWalRecord, EngineError> {
        Self::canonical_wal_record_with_boundary_and_outcome_isolation(
            identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            request_digest,
            outcome_kind,
            affected_rows,
            gpu_db_wal::CanonicalIsolation::ReadCommitted,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn canonical_wal_record_with_boundary_and_outcome_isolation(
        identity: gpu_db_wal::CanonicalIdentity,
        catalog_epoch: u64,
        catalog_digest: gpu_db_wal::CanonicalDigest,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        affected_rows: u64,
        isolation: gpu_db_wal::CanonicalIsolation,
    ) -> Result<gpu_db_wal::PreparedCanonicalWalRecord, EngineError> {
        Self::canonical_wal_record_with_boundary_and_optional_outcome_isolation(
            identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            request_digest,
            outcome_kind,
            Some(affected_rows),
            isolation,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn canonical_wal_record_with_boundary_and_optional_outcome_isolation(
        identity: gpu_db_wal::CanonicalIdentity,
        catalog_epoch: u64,
        catalog_digest: gpu_db_wal::CanonicalDigest,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        outcome_rows: Option<u64>,
        isolation: gpu_db_wal::CanonicalIsolation,
    ) -> Result<gpu_db_wal::PreparedCanonicalWalRecord, EngineError> {
        if outcome_kind == gpu_db_wal::CanonicalOutcomeKind::AbortError {
            return Err(EngineError::Durability(
                "committed engine WAL cannot be encoded with an abort outcome".to_string(),
            ));
        }
        let operation = SealedCanonicalOperation::from_live_payload(payload, txn_id)?;
        Self::canonical_wal_record_from_sealed_operation(
            identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            operation,
            request_digest,
            outcome_kind,
            outcome_rows,
            isolation,
        )
    }
}
