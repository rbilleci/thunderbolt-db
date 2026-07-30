//! The serial wave's one canonical proposal/status authority.
//!
//! Both the legacy re-resolved delta and INSERT-001's sealed fixed-width candidate enter this
//! owner only after their distinct pre-WAL preparation. It owns canonical WAL/status buffering,
//! but no device apply, physical group durability, or publication; the caller applies its selected
//! mutation before group durability and publication/ack.

use super::{CommitState, Engine, ExecuteError, Index, WriteSet};
use gpu_db_replication::LogReplicator;
use std::sync::Arc;
use std::time::Duration;

pub(super) enum WaveCanonicalOperation {
    Resolved {
        wal_payload: Arc<[u8]>,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        affected_rows: u64,
    },
    TypedInsert {
        bound: crate::wal_binary::BoundBinaryInsert,
        expected_commit_seq: Index,
        reserved_ledger_delta: crate::write_path::ReservedLedgerDelta,
    },
}

/// The ordinary result of a resolved operation.  Resolved operations carry no post-WAL typed
/// apply capability.
pub(super) struct ResolvedWaveCanonicalCommit {
    pub(super) commit_seq: Index,
    pub(super) wal_position: usize,
}

/// The one result shape returned by the serial canonical owner.
///
/// A typed INSERT cannot be observed as loose `(commit_seq, proposed_range)` scalars: its
/// post-WAL residency/allocator authority remains linear until the typed apply consumes it.
pub(super) enum WaveCanonicalCommit {
    Resolved(ResolvedWaveCanonicalCommit),
    TypedInsert(ClaimedTypedInsertAuthority),
}

impl WaveCanonicalCommit {
    pub(super) fn into_resolved(self) -> ResolvedWaveCanonicalCommit {
        match self {
            Self::Resolved(commit) => commit,
            Self::TypedInsert(_) => panic!(
                "commit-path invariant violation: typed INSERT canonical result escaped its claimed post-WAL apply boundary"
            ),
        }
    }
}

/// An opaque post-WAL permit accepted by the residency-facing device plan.
///
/// Only [`ClaimedTypedInsertAuthority`] mints this token.  Its fields and constructor remain
/// private to the canonical owner, so a plan cannot be applied with an independently selected
/// commit stamp.
#[must_use = "a typed INSERT post-WAL permit must be consumed by DeviceInsertPlan apply"]
pub(crate) struct TypedInsertPostWalApplyPermit {
    commit_seq: Index,
    _issued_by_claimed_typed_insert_authority: (),
    consumed: bool,
}

impl TypedInsertPostWalApplyPermit {
    pub(crate) fn into_append_created_by(
        mut self,
    ) -> crate::engine_residency::AppendCreatedBy<'static> {
        self.consumed = true;
        crate::engine_residency::AppendCreatedBy::InsertUniform(self.commit_seq)
    }
}

impl Drop for TypedInsertPostWalApplyPermit {
    fn drop(&mut self) {
        if !self.consumed && !std::thread::panicking() {
            panic!(
                "commit-path invariant violation: typed INSERT post-WAL permit dropped without device apply; restart recovery required"
            );
        }
    }
}

/// Move-only authority issued only once the serial canonical owner has proposed, appended, and
/// recorded every authoritative commit fact.  Dropping an unconsumed claim on a normal path is a
/// fail-stop invariant violation: the active wave guard then wedges rather than serving a torn
/// post-WAL state.
#[must_use = "a claimed typed INSERT must be consumed by TypedInsertApply after WAL"]
pub(super) struct ClaimedTypedInsertAuthority {
    commit_seq: Index,
    wal_position: usize,
    txn_id: u64,
    request_digest: gpu_db_wal::CanonicalDigest,
    proposed_range: Option<crate::wal_binary::ProposedRowIdRange>,
    ledger_receipt: Option<crate::write_path::LedgerClaimReceipt>,
    consumed: bool,
}

pub(super) struct ClaimedTypedInsertApply {
    pub(super) commit_seq: Index,
    pub(super) wal_position: usize,
    pub(super) proposed_range: crate::wal_binary::ProposedRowIdRange,
    pub(super) residency_permit: TypedInsertPostWalApplyPermit,
    pub(super) ledger_receipt: crate::write_path::LedgerClaimReceipt,
}

impl ClaimedTypedInsertAuthority {
    fn new(
        commit_seq: Index,
        wal_position: usize,
        txn_id: u64,
        request_digest: gpu_db_wal::CanonicalDigest,
        proposed_range: crate::wal_binary::ProposedRowIdRange,
        ledger_receipt: crate::write_path::LedgerClaimReceipt,
    ) -> Self {
        Self {
            commit_seq,
            wal_position,
            txn_id,
            request_digest,
            proposed_range: Some(proposed_range),
            ledger_receipt: Some(ledger_receipt),
            consumed: false,
        }
    }

    pub(super) fn into_apply_authority(
        mut self,
        expected_commit_seq: Index,
        expected_txn_id: u64,
        expected_request_digest: gpu_db_wal::CanonicalDigest,
    ) -> ClaimedTypedInsertApply {
        assert_eq!(
            self.commit_seq, expected_commit_seq,
            "commit-path invariant violation: typed INSERT post-WAL claim commit sequence drifted"
        );
        assert_eq!(
            self.txn_id, expected_txn_id,
            "commit-path invariant violation: typed INSERT post-WAL claim transaction drifted"
        );
        assert_eq!(
            self.request_digest, expected_request_digest,
            "commit-path invariant violation: typed INSERT post-WAL claim request digest drifted"
        );
        let proposed_range = self.proposed_range.take().expect(
            "commit-path invariant violation: typed INSERT post-WAL claim lost its original row-id range",
        );
        let ledger_receipt = self.ledger_receipt.take().expect(
            "commit-path invariant violation: typed INSERT post-WAL claim lost its stable ledger receipt",
        );
        self.consumed = true;
        ClaimedTypedInsertApply {
            commit_seq: self.commit_seq,
            wal_position: self.wal_position,
            proposed_range,
            residency_permit: TypedInsertPostWalApplyPermit {
                commit_seq: self.commit_seq,
                _issued_by_claimed_typed_insert_authority: (),
                consumed: false,
            },
            ledger_receipt,
        }
    }
}

impl Drop for ClaimedTypedInsertAuthority {
    fn drop(&mut self) {
        if !self.consumed && !std::thread::panicking() {
            panic!(
                "commit-path invariant violation: typed INSERT post-WAL claim dropped without device apply; restart recovery required"
            );
        }
    }
}

/// This remains deliberately test-only so physical-plan unit tests can exercise the fatal
/// post-WAL publisher seam without adding a production permit constructor or a second WAL path.
#[cfg(test)]
pub(crate) fn issue_test_only_typed_insert_post_wal_apply_permit(
    commit_seq: Index,
) -> TypedInsertPostWalApplyPermit {
    TypedInsertPostWalApplyPermit {
        commit_seq,
        _issued_by_claimed_typed_insert_authority: (),
        consumed: false,
    }
}

pub(super) enum WaveCanonicalFailure {
    PreDurable(ExecuteError),
}

impl Engine {
    /// The sole serial-wave proposal -> canonical record -> append -> status buffer -> ledger
    /// sequence. It returns before device apply, physical group durability, and publication/ack;
    /// `TypedInsert` is only a sealed-operation input, not another WAL/status authority.
    #[allow(clippy::too_many_arguments)] // one canonical transaction boundary, not a public API
    pub(super) fn append_canonical_wave_operation(
        &self,
        commit: &mut CommitState,
        txn_id: u64,
        expected_commit_seq: Index,
        wall_clock: u64,
        request_digest: gpu_db_wal::CanonicalDigest,
        write_set: &WriteSet,
        operation: WaveCanonicalOperation,
    ) -> Result<WaveCanonicalCommit, WaveCanonicalFailure> {
        if let WaveCanonicalOperation::TypedInsert {
            expected_commit_seq: bound_expected_commit_seq,
            ..
        } = &operation
        {
            assert_eq!(
                *bound_expected_commit_seq, expected_commit_seq,
                "commit-path invariant violation: fixed INSERT canonical append sequence drifted from its commit-gate proof"
            );
        }
        assert_eq!(
            commit.repl.peek_next_index(),
            expected_commit_seq,
            "commit-path invariant violation: canonical append no longer owns the fixed INSERT commit sequence"
        );
        // Typed INSERT constructs its one exact packed/outer record before replication proposal.
        // The resulting move-only owner is the only source of both proposal payload and WAL
        // bytes; the compatibility path remains unchanged until it independently migrates.
        let (resolved_operation, typed_operation) = match operation {
            WaveCanonicalOperation::Resolved {
                wal_payload,
                outcome_kind,
                affected_rows,
            } => (Some((wal_payload, outcome_kind, affected_rows)), None),
            WaveCanonicalOperation::TypedInsert {
                bound,
                reserved_ledger_delta,
                ..
            } => match Self::canonical_wal_record_with_commit_bound_insert(
                commit,
                txn_id,
                expected_commit_seq,
                0,
                bound,
                request_digest,
            ) {
                Ok(prepared) => {
                    let (record, proposed_range) = prepared.into_record_and_proposed_range();
                    (None, Some((record, proposed_range, reserved_ledger_delta)))
                }
                Err(error) => {
                    return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                        error,
                    )));
                }
            },
        };
        let mut typed_control_plane = match typed_operation {
            Some((record, proposed_range, reserved_ledger_delta)) => Some((
                commit
                    .reserve_typed_canonical_control_plane(txn_id, expected_commit_seq, record)
                    .map_err(|error| {
                        WaveCanonicalFailure::PreDurable(ExecuteError::Engine(error))
                    })?,
                proposed_range,
                reserved_ledger_delta,
            )),
            None => None,
        };
        let wal_len_before = commit.wal.len();
        let token = match typed_control_plane.as_mut() {
            Some((control_plane, ..)) => control_plane.propose(commit),
            None => {
                let (wal_payload, ..) = resolved_operation
                    .as_ref()
                    .expect("typed and resolved canonical operation ownership diverged");
                commit.repl.propose(Arc::clone(wal_payload))
            }
        }
        .map_err(|error| WaveCanonicalFailure::PreDurable(ExecuteError::Engine(error)))?;
        assert_eq!(
            token.index, expected_commit_seq,
            "the sequencer is the single proposer: the proposed index must equal the peek"
        );

        let affected_rows = match typed_control_plane.as_ref() {
            Some((_, proposed_range, _)) => u64::from(proposed_range.count()),
            None => {
                resolved_operation
                    .as_ref()
                    .expect("resolved canonical operation missing after proposal")
                    .2
            }
        };
        if let Some((control_plane, ..)) = typed_control_plane.as_mut() {
            if let Err(error) = control_plane.append_canonical(commit) {
                // The reserved WAL append is fallible before it mutates `records`; the proposal
                // already consumed its replication credit, so restore that frontier before
                // returning this typed pre-durable failure.
                commit.repl.rollback_unapplied_from(expected_commit_seq);
                return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                    error,
                )));
            }
        } else {
            let (wal_payload, outcome_kind, affected_rows) = resolved_operation
                .as_ref()
                .expect("resolved canonical operation missing before WAL append");
            let record = match Self::canonical_wal_record_with_commit_outcome(
                commit,
                txn_id,
                token.index,
                0,
                wal_payload,
                request_digest,
                *outcome_kind,
                *affected_rows,
            ) {
                Ok(record) => record,
                Err(error) => {
                    commit.repl.rollback_unapplied_from(token.index);
                    return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                        error,
                    )));
                }
            };
            commit.wal.append_canonical(record);
        }
        if let Err(error) = commit.repl.wait_committed(token, Duration::from_millis(0)) {
            if let Some((control_plane, ..)) = typed_control_plane.as_mut() {
                control_plane.rollback_tentative_wal_after_pre_durable_failure(commit);
            } else {
                commit.wal.truncate(wal_len_before);
            }
            commit.repl.rollback_unapplied_from(expected_commit_seq);
            return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                error,
            )));
        }
        let status_result = match typed_control_plane.as_mut() {
            Some((control_plane, ..)) => {
                debug_assert_eq!(token.index, expected_commit_seq);
                control_plane.record_transaction_status(commit, request_digest, affected_rows)
            }
            None => commit.record_transaction_status_digest_outcome(
                txn_id,
                request_digest,
                token.index,
                affected_rows,
            ),
        };
        if let Err(error) = status_result {
            if let Some((control_plane, ..)) = typed_control_plane.as_mut() {
                control_plane.rollback_tentative_wal_after_pre_durable_failure(commit);
            } else {
                commit.wal.truncate(wal_len_before);
            }
            commit.repl.rollback_unapplied_from(expected_commit_seq);
            return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                error,
            )));
        }
        let timestamp_micros = wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
        let timestamp_result = match typed_control_plane.as_mut() {
            Some((control_plane, ..)) => {
                control_plane.record_commit_timestamp(commit, timestamp_micros)
            }
            None => {
                commit.record_commit_timestamp(txn_id, timestamp_micros);
                Ok(())
            }
        };
        if let Err(error) = timestamp_result {
            if let Some((control_plane, ..)) = typed_control_plane.as_mut() {
                control_plane.rollback_inserted_status_after_pre_durable_failure(
                    commit,
                    request_digest,
                    affected_rows,
                );
                control_plane.rollback_tentative_wal_after_pre_durable_failure(commit);
            } else {
                commit.wal.truncate(wal_len_before);
            }
            commit.repl.rollback_unapplied_from(expected_commit_seq);
            return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                error,
            )));
        }
        Ok(match typed_control_plane {
            Some((mut control_plane, proposed_range, reserved_ledger_delta)) => {
                let wal_position =
                    control_plane.claim_tentative_wal_after_final_rollbackable_step(commit);
                let ledger_receipt = commit
                    .ledger
                    .claim_typed_delta(reserved_ledger_delta, expected_commit_seq);
                WaveCanonicalCommit::TypedInsert(ClaimedTypedInsertAuthority::new(
                    expected_commit_seq,
                    wal_position,
                    txn_id,
                    request_digest,
                    proposed_range,
                    ledger_receipt,
                ))
            }
            None => {
                commit.ledger.record(write_set, expected_commit_seq);
                WaveCanonicalCommit::Resolved(ResolvedWaveCanonicalCommit {
                    commit_seq: expected_commit_seq,
                    wal_position: commit.wal.len(),
                })
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_post_wal_claim_panics_if_normally_dropped_without_apply() {
        let result = std::panic::catch_unwind(|| {
            drop(ClaimedTypedInsertAuthority::new(
                7,
                11,
                13,
                [17; 32],
                crate::wal_binary::ProposedRowIdRange::new(23, 2).unwrap(),
                crate::write_path::issue_test_only_visible_ledger_claim_receipt(),
            ));
        });
        assert!(result.is_err());
    }

    #[test]
    fn typed_post_wal_claim_binds_and_consumes_the_original_range() {
        let claim = ClaimedTypedInsertAuthority::new(
            7,
            11,
            13,
            [17; 32],
            crate::wal_binary::ProposedRowIdRange::new(23, 2).unwrap(),
            crate::write_path::issue_test_only_visible_ledger_claim_receipt(),
        );
        let apply = claim.into_apply_authority(7, 13, [17; 32]);
        assert_eq!(apply.commit_seq, 7);
        assert_eq!(apply.wal_position, 11);
        assert_eq!(apply.proposed_range.first(), 23);
        assert_eq!(apply.proposed_range.count(), 2);
        assert!(matches!(
            apply.residency_permit.into_append_created_by(),
            crate::engine_residency::AppendCreatedBy::InsertUniform(7)
        ));
    }

    #[test]
    fn typed_post_wal_permit_panics_if_normally_dropped_without_device_apply() {
        let result = std::panic::catch_unwind(|| {
            drop(issue_test_only_typed_insert_post_wal_apply_permit(7));
        });
        assert!(result.is_err());
    }
}
