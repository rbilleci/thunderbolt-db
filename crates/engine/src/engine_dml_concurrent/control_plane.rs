//! Typed canonical control-plane capacity ownership.
//!
//! This module receives the prebuilt exact canonical record owner and reserves the four live
//! collection owners that the typed serial canonical path mutates: local replication entries,
//! canonical WAL records, terminal transaction status, and commit timestamps. Conflict-ledger
//! recording remains in the shared committed-transaction publisher; this reservation does not
//! mint a separate typed ledger claim. It does not own bounded serial/FUA group arenas, wave
//! containers, publication, or result responses.

use super::CommitState;
use crate::{DurableTransactionOutcome, DurableTransactionStatus, EngineError, Index, TxnId};
use gpu_db_replication::LocalReplicatorProposalReservation;
use gpu_db_types::CommitToken;
use gpu_db_wal::{CanonicalDigest, PreparedCanonicalWalRecord, WalTypedExactAppendReservation};
use std::collections::hash_map::Entry;
#[cfg(test)]
use std::sync::Arc;

/// All actual collection capacity credits needed by one typed canonical operation.
///
/// It is created only from the mutable commit owner, has no public constructor, and each credit
/// is moved into its exact mutation method. Dropping an unconsumed bundle is harmless: capacity
/// remains available, but no logical frontier has changed.
#[must_use = "a typed canonical control-plane reservation must be consumed or abandoned before WAL"]
pub(crate) struct TypedCanonicalControlPlaneReservation {
    expected_owner_id: u64,
    txn_id: TxnId,
    expected_commit_seq: Index,
    replication: Option<LocalReplicatorProposalReservation>,
    wal: Option<WalTypedExactAppendReservation>,
    status: Option<TransactionStatusSlotReservation>,
    timestamp: Option<CommitTimestampSlotReservation>,
    status_inserted: bool,
}

struct TransactionStatusSlotReservation {
    expected_len: usize,
    expected_capacity: usize,
    expected_generation: u64,
}

struct CommitTimestampSlotReservation {
    expected_len: usize,
    expected_capacity: usize,
    expected_generation: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// The full `Before*` names are stable sabotage-boundary labels surfaced in failure diagnostics.
#[allow(clippy::enum_variant_names)]
pub(super) enum TypedControlPlaneAcquisitionFault {
    BeforeReplication,
    BeforeWal,
    BeforeStatus,
    BeforeTimestamp,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// Keep the same explicit boundary vocabulary for mutation-side sabotage diagnostics.
#[allow(clippy::enum_variant_names)]
pub(super) enum TypedControlPlaneMutationFault {
    BeforeCanonicalAppend,
    BeforeTransactionStatus,
    BeforeCommitTimestamp,
}

#[cfg(test)]
impl TypedControlPlaneMutationFault {
    const fn code(self) -> u8 {
        match self {
            Self::BeforeCanonicalAppend => 1,
            Self::BeforeTransactionStatus => 2,
            Self::BeforeCommitTimestamp => 3,
        }
    }
}

#[cfg(test)]
impl TypedControlPlaneAcquisitionFault {
    const fn code(self) -> u8 {
        match self {
            Self::BeforeReplication => 1,
            Self::BeforeWal => 2,
            Self::BeforeStatus => 3,
            Self::BeforeTimestamp => 4,
        }
    }
}

#[cfg(test)]
thread_local! {
    static TYPED_CONTROL_PLANE_ACQUISITION_FAULT: std::cell::Cell<u8> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
thread_local! {
    static TYPED_CONTROL_PLANE_MUTATION_FAULT: std::cell::Cell<u8> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
pub(super) fn fail_typed_control_plane_acquisition_at(boundary: TypedControlPlaneAcquisitionFault) {
    TYPED_CONTROL_PLANE_ACQUISITION_FAULT.with(|fault| fault.set(boundary.code()));
}

#[cfg(test)]
pub(super) fn fail_typed_control_plane_mutation_at(boundary: TypedControlPlaneMutationFault) {
    TYPED_CONTROL_PLANE_MUTATION_FAULT.with(|fault| fault.set(boundary.code()));
}

#[cfg(test)]
fn injected_acquisition_failure(
    boundary: TypedControlPlaneAcquisitionFault,
) -> Result<(), EngineError> {
    let injected = TYPED_CONTROL_PLANE_ACQUISITION_FAULT.with(|fault| {
        let injected = fault.get() == boundary.code();
        if injected {
            fault.set(0);
        }
        injected
    });
    if injected {
        return Err(EngineError::ProposalFailed(format!(
            "injected typed control-plane reservation failure at {boundary:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
fn injected_mutation_failure(boundary: TypedControlPlaneMutationFault) -> Result<(), EngineError> {
    let injected = TYPED_CONTROL_PLANE_MUTATION_FAULT.with(|fault| {
        let injected = fault.get() == boundary.code();
        if injected {
            fault.set(0);
        }
        injected
    });
    if injected {
        return Err(EngineError::Durability(format!(
            "injected typed control-plane mutation failure at {boundary:?}"
        )));
    }
    Ok(())
}

#[cfg(not(test))]
fn injected_acquisition_failure(_boundary: ()) -> Result<(), EngineError> {
    Ok(())
}

#[cfg(not(test))]
fn injected_mutation_failure(_boundary: ()) -> Result<(), EngineError> {
    Ok(())
}

#[cfg(test)]
macro_rules! fault_boundary {
    ($boundary:ident) => {
        injected_acquisition_failure(TypedControlPlaneAcquisitionFault::$boundary)?
    };
}

#[cfg(test)]
macro_rules! mutation_fault_boundary {
    ($boundary:ident) => {
        injected_mutation_failure(TypedControlPlaneMutationFault::$boundary)?
    };
}

#[cfg(not(test))]
macro_rules! mutation_fault_boundary {
    ($boundary:ident) => {
        injected_mutation_failure(())?
    };
}

#[cfg(not(test))]
macro_rules! fault_boundary {
    ($boundary:ident) => {
        injected_acquisition_failure(())?
    };
}

impl CommitState {
    /// Move the prebuilt exact canonical record owner into a reservation for every live host
    /// control-plane collection touched by the typed claim. The caller owns `commit_mutex`
    /// through `&mut self`; failure occurs before proposal, logical WAL append, status/timestamp
    /// insertion, ledger mutation, or device apply. It excludes only bounded group arenas,
    /// publication, result, and wave owners.
    pub(crate) fn reserve_typed_canonical_control_plane(
        &mut self,
        txn_id: TxnId,
        expected_commit_seq: Index,
        prepared: PreparedCanonicalWalRecord,
    ) -> Result<TypedCanonicalControlPlaneReservation, EngineError> {
        if self.repl.peek_next_index() != expected_commit_seq {
            return Err(EngineError::ProposalFailed(format!(
                "typed canonical reservation expected commit sequence {expected_commit_seq}, found {}",
                self.repl.peek_next_index()
            )));
        }
        if self.transaction_status.contains_key(&txn_id)
            || self.wal_commit_timestamps_micros.contains_key(&txn_id)
        {
            return Err(EngineError::Durability(format!(
                "typed canonical reservation refuses already-recorded transaction {txn_id}"
            )));
        }

        fault_boundary!(BeforeReplication);
        let replication = self.repl.reserve_next_proposal()?;
        fault_boundary!(BeforeWal);
        let wal = self.wal.reserve_typed_exact_append(prepared)?;

        fault_boundary!(BeforeStatus);
        self.transaction_status.try_reserve(1).map_err(|_| {
            EngineError::Durability(
                "unable to reserve transaction-status capacity before typed WAL".to_string(),
            )
        })?;
        let status = TransactionStatusSlotReservation {
            expected_len: self.transaction_status.len(),
            expected_capacity: self.transaction_status.capacity(),
            expected_generation: self.transaction_status_reservation_generation,
        };

        fault_boundary!(BeforeTimestamp);
        self.wal_commit_timestamps_micros
            .try_reserve(1)
            .map_err(|_| {
                EngineError::Durability(
                    "unable to reserve commit-timestamp capacity before typed WAL".to_string(),
                )
            })?;
        let timestamp = CommitTimestampSlotReservation {
            expected_len: self.wal_commit_timestamps_micros.len(),
            expected_capacity: self.wal_commit_timestamps_micros.capacity(),
            expected_generation: self.wal_commit_timestamp_reservation_generation,
        };

        Ok(TypedCanonicalControlPlaneReservation {
            expected_owner_id: self.control_plane_reservation_owner_id,
            txn_id,
            expected_commit_seq,
            replication: Some(replication),
            wal: Some(wal),
            status: Some(status),
            timestamp: Some(timestamp),
            status_inserted: false,
        })
    }
}

impl TypedCanonicalControlPlaneReservation {
    fn validate_owner_and_sequence(&self, commit: &CommitState) -> Result<(), EngineError> {
        if commit.control_plane_reservation_owner_id != self.expected_owner_id {
            return Err(EngineError::Durability(
                "typed canonical control-plane reservation belongs to another commit state"
                    .to_string(),
            ));
        }
        if commit.repl.peek_next_index() != self.expected_commit_seq {
            return Err(EngineError::ProposalFailed(format!(
                "typed canonical control-plane sequence drifted before consumption: expected {}, found {}",
                self.expected_commit_seq,
                commit.repl.peek_next_index()
            )));
        }
        Ok(())
    }

    pub(crate) fn propose(&mut self, commit: &mut CommitState) -> Result<CommitToken, EngineError> {
        self.validate_owner_and_sequence(commit)?;
        let payload = self
            .wal
            .as_ref()
            .ok_or_else(|| {
                EngineError::ProposalFailed(
                    "typed canonical WAL owner was consumed before replication proposal"
                        .to_string(),
                )
            })?
            .replication_payload()?;
        let reservation = self.replication.take().ok_or_else(|| {
            EngineError::ProposalFailed(
                "typed canonical replication credit was already consumed".to_string(),
            )
        })?;
        let token = commit.repl.propose_reserved(reservation, payload)?;
        if token.index != self.expected_commit_seq {
            return Err(EngineError::ProposalFailed(format!(
                "typed canonical reserved proposal assigned {}, expected {}",
                token.index, self.expected_commit_seq
            )));
        }
        Ok(token)
    }

    pub(crate) fn append_canonical(&mut self, commit: &mut CommitState) -> Result<(), EngineError> {
        if commit.control_plane_reservation_owner_id != self.expected_owner_id {
            return Err(EngineError::Durability(
                "typed canonical WAL credit belongs to another commit state".to_string(),
            ));
        }
        mutation_fault_boundary!(BeforeCanonicalAppend);
        let reservation = self.wal.as_mut().ok_or_else(|| {
            EngineError::Durability("typed canonical WAL owner was already consumed".to_string())
        })?;
        // Any unexpected state drift here occurs after replication proposal.  The canonical
        // owner treats it as fail-stop; only the injected boundary above returns normally.
        commit
            .wal
            .append_typed_exact_tentative(reservation)
            .unwrap_or_else(|error| {
                panic!(
                    "commit-path invariant violation: typed exact WAL append drifted after replication proposal: {error}"
                )
            });
        Ok(())
    }

    /// Restore the exact tentative record and all its WAL frontiers before the final claim.
    /// The replication frontier is restored by the canonical owner immediately alongside this.
    pub(crate) fn rollback_tentative_wal_after_pre_durable_failure(
        &mut self,
        commit: &mut CommitState,
    ) {
        let reservation = self
            .wal
            .as_mut()
            .expect("commit-path invariant violation: typed WAL owner disappeared before rollback");
        commit.wal.rollback_typed_exact_append(reservation).expect(
            "commit-path invariant violation: typed exact WAL rollback drifted after proposal",
        );
    }

    /// The final rollbackable step is complete once the timestamp is recorded.  This makes the
    /// exact tail flushable and intentionally consumes the only generic rollback carrier.
    pub(crate) fn claim_tentative_wal_after_final_rollbackable_step(
        &mut self,
        commit: &mut CommitState,
    ) -> usize {
        let reservation = self
            .wal
            .take()
            .expect("commit-path invariant violation: typed WAL owner disappeared before claim");
        commit
            .wal
            .claim_typed_exact_append(reservation)
            .expect("commit-path invariant violation: typed exact WAL claim drifted after proposal")
    }

    pub(crate) fn record_transaction_status(
        &mut self,
        commit: &mut CommitState,
        request_digest: CanonicalDigest,
        affected_rows: u64,
    ) -> Result<(), EngineError> {
        if commit.control_plane_reservation_owner_id != self.expected_owner_id {
            return Err(EngineError::Durability(
                "typed canonical status credit belongs to another commit state".to_string(),
            ));
        }
        mutation_fault_boundary!(BeforeTransactionStatus);
        let status_slot = self.status.as_ref().ok_or_else(|| {
            EngineError::Durability(
                "typed canonical status credit was already consumed".to_string(),
            )
        })?;
        if commit.transaction_status.len() != status_slot.expected_len
            || commit.transaction_status.capacity() != status_slot.expected_capacity
            || commit.transaction_status_reservation_generation != status_slot.expected_generation
            || commit.transaction_status.contains_key(&self.txn_id)
        {
            return Err(EngineError::Durability(
                "typed canonical transaction-status reservation drifted before consumption"
                    .to_string(),
            ));
        }
        let capacity_before = commit.transaction_status.capacity();
        let status = DurableTransactionStatus {
            request_digest,
            outcome: DurableTransactionOutcome::Committed {
                commit_seq: self.expected_commit_seq,
                affected_rows,
            },
        };
        let status_slot = self.status.take().expect("status credit checked above");
        match commit.transaction_status.entry(self.txn_id) {
            Entry::Vacant(entry) => {
                entry.insert(status);
            }
            Entry::Occupied(_) => {
                self.status = Some(status_slot);
                return Err(EngineError::Durability(
                    "typed canonical transaction status became occupied before insertion"
                        .to_string(),
                ));
            }
        }
        assert_eq!(
            commit.transaction_status.capacity(),
            capacity_before,
            "reserved typed transaction-status insert unexpectedly grew its HashMap"
        );
        commit.invalidate_transaction_status_reservations();
        self.status_inserted = true;
        Ok(())
    }

    pub(crate) fn record_commit_timestamp(
        &mut self,
        commit: &mut CommitState,
        timestamp_micros: u64,
    ) -> Result<(), EngineError> {
        if commit.control_plane_reservation_owner_id != self.expected_owner_id {
            return Err(EngineError::Durability(
                "typed canonical timestamp credit belongs to another commit state".to_string(),
            ));
        }
        mutation_fault_boundary!(BeforeCommitTimestamp);
        let timestamp_slot = self.timestamp.as_ref().ok_or_else(|| {
            EngineError::Durability(
                "typed canonical timestamp credit was already consumed".to_string(),
            )
        })?;
        if commit.wal_commit_timestamps_micros.len() != timestamp_slot.expected_len
            || commit.wal_commit_timestamps_micros.capacity() != timestamp_slot.expected_capacity
            || commit.wal_commit_timestamp_reservation_generation
                != timestamp_slot.expected_generation
            || commit
                .wal_commit_timestamps_micros
                .contains_key(&self.txn_id)
        {
            return Err(EngineError::Durability(
                "typed canonical commit-timestamp reservation drifted before consumption"
                    .to_string(),
            ));
        }
        let capacity_before = commit.wal_commit_timestamps_micros.capacity();
        self.timestamp
            .take()
            .expect("timestamp credit checked above");
        commit
            .wal_commit_timestamps_micros
            .insert(self.txn_id, timestamp_micros);
        assert_eq!(
            commit.wal_commit_timestamps_micros.capacity(),
            capacity_before,
            "reserved typed commit-timestamp insert unexpectedly grew its HashMap"
        );
        commit.max_commit_timestamp_micros =
            commit.max_commit_timestamp_micros.max(timestamp_micros);
        commit.invalidate_commit_timestamp_reservations();
        Ok(())
    }

    /// Undo only the exact status inserted by this bundle when a later pre-durable owner fails.
    /// This is deliberately narrower than a generic status delete: retry identity is durable
    /// authority once retained, so an unexpected key/value mismatch is a fail-stop invariant
    /// violation rather than permission to erase somebody else's terminal claim.
    pub(crate) fn rollback_inserted_status_after_pre_durable_failure(
        &mut self,
        commit: &mut CommitState,
        request_digest: CanonicalDigest,
        affected_rows: u64,
    ) {
        if !self.status_inserted {
            return;
        }
        assert_eq!(
            commit.control_plane_reservation_owner_id, self.expected_owner_id,
            "commit-path invariant violation: typed status rollback crossed commit-state owners"
        );
        let expected = DurableTransactionStatus {
            request_digest,
            outcome: DurableTransactionOutcome::Committed {
                commit_seq: self.expected_commit_seq,
                affected_rows,
            },
        };
        assert_eq!(
            commit.transaction_status.get(&self.txn_id),
            Some(&expected),
            "commit-path invariant violation: typed timestamp failure found a different terminal status"
        );
        let removed = commit.transaction_status.remove(&self.txn_id);
        assert_eq!(
            removed,
            Some(expected),
            "commit-path invariant violation: typed status rollback lost its exact terminal status"
        );
        commit.invalidate_transaction_status_reservations();
        self.status_inserted = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;

    #[derive(Debug, Eq, PartialEq)]
    struct LogicalFrontiers {
        repl_entries: usize,
        repl_next: Index,
        wal_records: usize,
        statuses: usize,
        timestamps: usize,
        timestamp_max: u64,
        ledger_entries: usize,
    }

    fn frontiers(commit: &CommitState) -> LogicalFrontiers {
        LogicalFrontiers {
            repl_entries: commit.repl.retained_entry_count(),
            repl_next: commit.repl.peek_next_index(),
            wal_records: commit.wal.len(),
            statuses: commit.transaction_status.len(),
            timestamps: commit.wal_commit_timestamps_micros.len(),
            timestamp_max: commit.max_commit_timestamp_micros,
            ledger_entries: commit.ledger.len(),
        }
    }

    fn prepared_record(txn_id: TxnId, commit_seq: Index) -> PreparedCanonicalWalRecord {
        let identity = gpu_db_wal::CanonicalIdentity {
            database_id: [1; 16],
            cluster_id: [2; 16],
            timeline_id: [3; 16],
            format_epoch: 1,
        };
        let operation = gpu_db_wal::CanonicalFragment {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: vec![5],
        };
        let status = gpu_db_wal::CanonicalFragment {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: vec![6],
        };
        let header = gpu_db_wal::CanonicalPreApplyHeader {
            identity,
            leader_epoch: 1,
            commit_seq,
            stable_transaction_id: txn_id,
            request_digest: [4; 32],
            isolation: gpu_db_wal::CanonicalIsolation::ReadCommitted,
            flags: u32::from(gpu_db_wal::CanonicalFragmentKind::RowMutation as u16),
            catalog_before_epoch: 0,
            catalog_after_epoch: 0,
            catalog_before_digest: [7; 32],
            catalog_after_digest: [7; 32],
            operation_count: 2,
            table_block_count: 0,
            allocator_high_water: 0,
        };
        let outcome = gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            affected_rows: 1,
            sqlstate: None,
            constraint_id: 0,
            target_digest: [8; 32],
            returning_digest: [0; 32],
        };
        gpu_db_wal::prepare_exact_canonical_wal_record(
            txn_id,
            gpu_db_wal::CanonicalPhysicalRange {
                log_epoch: 1,
                lane_id: 0,
                segment_id: commit_seq,
                first_frame_ordinal: 0,
            },
            header,
            &[operation, status],
            outcome,
        )
        .expect("prepare exact test canonical record")
    }

    #[test]
    fn acquisition_faults_and_drop_leave_all_four_logical_owners_unchanged() {
        for fault in [
            TypedControlPlaneAcquisitionFault::BeforeReplication,
            TypedControlPlaneAcquisitionFault::BeforeWal,
            TypedControlPlaneAcquisitionFault::BeforeStatus,
            TypedControlPlaneAcquisitionFault::BeforeTimestamp,
        ] {
            let engine = Engine::new_local();
            let mut commit = engine.commit_state();
            let before = frontiers(&commit);
            fail_typed_control_plane_acquisition_at(fault);
            let error =
                match commit.reserve_typed_canonical_control_plane(7, 1, prepared_record(7, 1)) {
                    Ok(_) => {
                        panic!("injected acquisition boundary must fail before logical mutation")
                    }
                    Err(error) => error,
                };
            assert!(error.to_string().contains("injected typed control-plane"));
            assert_eq!(frontiers(&commit), before, "fault {fault:?}");
        }

        let engine = Engine::new_local();
        let mut commit = engine.commit_state();
        let before = frontiers(&commit);
        let reservation = commit
            .reserve_typed_canonical_control_plane(7, 1, prepared_record(7, 1))
            .expect("reserve all four owners");
        drop(reservation);
        assert_eq!(frontiers(&commit), before);
    }

    #[test]
    fn successful_bundle_consumes_each_owner_without_status_or_timestamp_growth() {
        let engine = Engine::new_local();
        let mut commit = engine.commit_state();
        let mut reservation = commit
            .reserve_typed_canonical_control_plane(7, 1, prepared_record(7, 1))
            .expect("reserve all four owners");
        let status_capacity = commit.transaction_status.capacity();
        let timestamp_capacity = commit.wal_commit_timestamps_micros.capacity();

        let proposal_payload = reservation
            .wal
            .as_ref()
            .expect("preproposal typed WAL owner")
            .replication_payload()
            .expect("preproposal packed payload");
        let _token = reservation
            .propose(&mut commit)
            .expect("consume replication credit");
        let replicated_payload = &commit
            .repl
            .drain_committed_from(0)
            .next()
            .expect("local proposal commits one entry")
            .payload;
        assert!(
            Arc::ptr_eq(&proposal_payload, replicated_payload),
            "replication must retain the exact packed payload Arc"
        );
        reservation
            .append_canonical(&mut commit)
            .expect("consume WAL credit");
        assert!(
            Arc::ptr_eq(
                &proposal_payload,
                &commit.wal.last_record().expect("typed WAL record").payload
            ),
            "WAL logical record must retain the exact replication payload Arc"
        );
        reservation
            .record_transaction_status(&mut commit, [9; 32], 1)
            .expect("consume status credit");
        reservation
            .record_commit_timestamp(&mut commit, 12)
            .expect("consume timestamp credit");
        assert_eq!(
            reservation.claim_tentative_wal_after_final_rollbackable_step(&mut commit),
            1
        );

        assert_eq!(frontiers(&commit).repl_entries, 1);
        assert_eq!(frontiers(&commit).wal_records, 1);
        assert_eq!(frontiers(&commit).statuses, 1);
        assert_eq!(frontiers(&commit).timestamps, 1);
        assert_eq!(commit.transaction_status.capacity(), status_capacity);
        assert_eq!(
            commit.wal_commit_timestamps_micros.capacity(),
            timestamp_capacity
        );
    }

    #[test]
    fn bundle_rejects_cross_engine_and_status_state_drift_before_consumption() {
        let first = Engine::new_local();
        let mut first_commit = first.commit_state();
        let mut reservation = first_commit
            .reserve_typed_canonical_control_plane(7, 1, prepared_record(7, 1))
            .expect("reserve first owner");
        let second = Engine::new_local();
        let mut second_commit = second.commit_state();
        let before = frontiers(&second_commit);
        let error = reservation
            .propose(&mut second_commit)
            .expect_err("another commit state must reject this bundle");
        assert!(error.to_string().contains("another commit state"));
        assert_eq!(frontiers(&second_commit), before);
        drop(second_commit);

        first_commit
            .record_transaction_status_digest_outcome(99, [3; 32], 99, 1)
            .expect("introduce status drift");
        let error = reservation
            .record_transaction_status(&mut first_commit, [9; 32], 1)
            .expect_err("status generation drift must reject before insertion");
        assert!(error.to_string().contains("reservation drifted"));
        assert!(!first_commit.transaction_status.contains_key(&7));
    }

    #[test]
    fn checkpoint_timestamp_prune_invalidates_an_outstanding_timestamp_credit() {
        let engine = Engine::new_local();
        let mut commit = engine.commit_state();
        commit.record_commit_timestamp(99, 11);
        let mut reservation = commit
            .reserve_typed_canonical_control_plane(7, 1, prepared_record(7, 1))
            .expect("reserve timestamp credit before checkpoint pruning");
        let checkpointed = std::collections::BTreeSet::from([99]);

        commit.prune_checkpointed_commit_timestamps(&checkpointed);

        let error = reservation
            .record_commit_timestamp(&mut commit, 12)
            .expect_err("checkpoint pruning must stale the earlier timestamp credit");
        assert!(error.to_string().contains("reservation drifted"));
        assert!(!commit.wal_commit_timestamps_micros.contains_key(&7));
    }

    #[test]
    fn later_typed_control_plane_failures_restore_all_four_logical_frontiers() {
        let engine = Engine::new_local();
        let mut commit = engine.commit_state();
        let before = frontiers(&commit);
        let mut reservation = commit
            .reserve_typed_canonical_control_plane(7, 1, prepared_record(7, 1))
            .expect("reserve all four owners");
        reservation
            .propose(&mut commit)
            .expect("consume replication credit");
        fail_typed_control_plane_mutation_at(TypedControlPlaneMutationFault::BeforeCanonicalAppend);
        let error = reservation
            .append_canonical(&mut commit)
            .expect_err("injected canonical append failure");
        assert!(error.to_string().contains("BeforeCanonicalAppend"));
        commit.repl.rollback_unapplied_from(1);
        assert_eq!(frontiers(&commit), before);

        let engine = Engine::new_local();
        let mut commit = engine.commit_state();
        let before = frontiers(&commit);
        let mut reservation = commit
            .reserve_typed_canonical_control_plane(7, 1, prepared_record(7, 1))
            .expect("reserve all four owners");
        reservation
            .propose(&mut commit)
            .expect("consume replication credit");
        reservation
            .append_canonical(&mut commit)
            .expect("consume WAL credit");
        fail_typed_control_plane_mutation_at(
            TypedControlPlaneMutationFault::BeforeTransactionStatus,
        );
        let error = reservation
            .record_transaction_status(&mut commit, [9; 32], 1)
            .expect_err("injected status failure");
        assert!(error.to_string().contains("BeforeTransactionStatus"));
        reservation.rollback_tentative_wal_after_pre_durable_failure(&mut commit);
        commit.repl.rollback_unapplied_from(1);
        assert_eq!(frontiers(&commit), before);

        let engine = Engine::new_local();
        let mut commit = engine.commit_state();
        let before = frontiers(&commit);
        let mut reservation = commit
            .reserve_typed_canonical_control_plane(7, 1, prepared_record(7, 1))
            .expect("reserve all four owners");
        reservation
            .propose(&mut commit)
            .expect("consume replication credit");
        reservation
            .append_canonical(&mut commit)
            .expect("consume WAL credit");
        reservation
            .record_transaction_status(&mut commit, [9; 32], 1)
            .expect("consume status credit");
        fail_typed_control_plane_mutation_at(TypedControlPlaneMutationFault::BeforeCommitTimestamp);
        let error = reservation
            .record_commit_timestamp(&mut commit, 12)
            .expect_err("injected timestamp failure");
        assert!(error.to_string().contains("BeforeCommitTimestamp"));
        reservation.rollback_inserted_status_after_pre_durable_failure(&mut commit, [9; 32], 1);
        reservation.rollback_tentative_wal_after_pre_durable_failure(&mut commit);
        commit.repl.rollback_unapplied_from(1);
        assert_eq!(frontiers(&commit), before);
    }

    #[test]
    fn reservation_owns_exact_record_and_excludes_group_publication_and_result_owners() {
        let source = include_str!("control_plane.rs");
        let archive = include_str!("../engine_wal_archive.rs");
        assert!(source.contains("prebuilt exact canonical record owner"));
        assert!(source.contains("bounded serial/FUA group arenas"));
        assert!(source.contains("publication, or result responses"));
        assert!(archive.contains("prune_checkpointed_commit_timestamps"));
        assert!(!archive.contains(".wal_commit_timestamps_micros\n            .retain("));
    }
}
