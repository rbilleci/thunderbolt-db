//! Shared sealed-operation canonical envelope construction.
//!
//! This module owns no durable state itself. It keeps the common framing path bounded while the
//! parent module remains the recovery/lineage owner.

use super::*;

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn canonical_wal_record_from_sealed_operation(
        identity: gpu_db_wal::CanonicalIdentity,
        catalog_epoch: u64,
        catalog_digest: gpu_db_wal::CanonicalDigest,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        operation: SealedCanonicalOperation,
        request_digest: gpu_db_wal::CanonicalDigest,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        outcome_rows: Option<u64>,
        isolation: gpu_db_wal::CanonicalIsolation,
    ) -> Result<gpu_db_wal::PreparedCanonicalWalRecord, EngineError> {
        Self::canonical_wal_record_from_sealed_operation_with_encoding(
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

    #[allow(clippy::too_many_arguments)]
    fn canonical_wal_record_from_sealed_operation_with_encoding(
        identity: gpu_db_wal::CanonicalIdentity,
        catalog_epoch: u64,
        catalog_digest: gpu_db_wal::CanonicalDigest,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        operation: SealedCanonicalOperation,
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
        let operation_digest = operation.operation_digest();
        let operation_kind = operation.kind();
        let table_block_count = operation.table_block_count();
        let allocator_high_water = operation.allocator_high_water();
        let affected_rows = operation.affected_rows_or_default(outcome_kind, outcome_rows)?;
        let operation_fragment = operation.into_fragment();
        let status_fragment = gpu_db_wal::CanonicalFragment {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: Self::encode_transaction_claim_status(identity, txn_id, request_digest),
        };
        let catalog_after_epoch =
            if operation_kind == gpu_db_wal::CanonicalFragmentKind::CatalogMutation {
                catalog_epoch.checked_add(1).ok_or_else(|| {
                    EngineError::Durability("canonical catalog epoch overflow".to_string())
                })?
            } else {
                catalog_epoch
            };
        let catalog_after_digest = Self::canonical_catalog_transition(
            catalog_digest,
            operation_kind,
            &operation_fragment.body,
        );
        let header = gpu_db_wal::CanonicalPreApplyHeader {
            identity,
            leader_epoch: 1,
            commit_seq,
            stable_transaction_id: txn_id,
            request_digest,
            isolation,
            flags: u32::from(operation_kind as u16),
            catalog_before_epoch: catalog_epoch,
            catalog_after_epoch,
            catalog_before_digest: catalog_digest,
            catalog_after_digest,
            operation_count: 2,
            table_block_count,
            allocator_high_water,
        };
        let outcome = gpu_db_wal::CanonicalOutcome {
            kind: outcome_kind,
            affected_rows,
            sqlstate: None,
            constraint_id: 0,
            target_digest: operation_digest,
            returning_digest: [0; 32],
        };
        let physical = gpu_db_wal::CanonicalPhysicalRange {
            log_epoch: 1,
            lane_id,
            segment_id: commit_seq,
            first_frame_ordinal: 0,
        };
        let fragments = [operation_fragment, status_fragment];
        gpu_db_wal::encode_canonical_envelope(physical, &header, &fragments, &outcome)?
            .into_prepared_record(txn_id)
    }
}
