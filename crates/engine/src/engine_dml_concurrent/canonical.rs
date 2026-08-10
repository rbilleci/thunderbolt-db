//! The serial UPDATE/DELETE wave's canonical proposal/status authority.
//!
//! INSERT is intercepted before wave admission and commits through the codec-5 transaction
//! terminal. This owner therefore accepts only an already resolved UPDATE/DELETE payload and
//! returns before apply, physical group durability, and publication/ack.

use super::{CommitState, Engine, ExecuteError, Index, WriteSet};
use gpu_db_replication::LogReplicator;
use std::sync::Arc;
use std::time::Duration;

pub(super) struct ResolvedWaveCanonicalCommit {
    pub(super) commit_seq: Index,
    pub(super) wal_position: usize,
}

/// Opaque post-WAL capability accepted by the shared GPU INSERT publisher.
///
/// The codec-5 transaction terminal mints this only after installing the exact canonical
/// WAL/status outcome. Tests may mint it solely to exercise fatal post-WAL physical seams.
#[must_use = "a typed INSERT post-WAL permit must be consumed by DeviceInsertPlan apply"]
pub(crate) struct TypedInsertPostWalApplyPermit {
    commit_seq: Index,
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

pub(crate) fn issue_transaction_terminal_typed_insert_apply_permit(
    commit_seq: Index,
) -> TypedInsertPostWalApplyPermit {
    TypedInsertPostWalApplyPermit {
        commit_seq,
        consumed: false,
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

#[cfg(test)]
pub(crate) fn issue_test_only_typed_insert_post_wal_apply_permit(
    commit_seq: Index,
) -> TypedInsertPostWalApplyPermit {
    TypedInsertPostWalApplyPermit {
        commit_seq,
        consumed: false,
    }
}

pub(super) enum WaveCanonicalFailure {
    PreDurable(ExecuteError),
}

impl Engine {
    /// Propose, append, and record one resolved UPDATE/DELETE outcome under the serial wave.
    #[allow(clippy::too_many_arguments)] // one canonical transaction boundary, not a public API
    pub(super) fn append_canonical_wave_operation(
        &self,
        commit: &mut CommitState,
        txn_id: u64,
        expected_commit_seq: Index,
        wall_clock: u64,
        request_digest: gpu_db_wal::CanonicalDigest,
        write_set: &WriteSet,
        wal_payload: Arc<[u8]>,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        affected_rows: u64,
    ) -> Result<ResolvedWaveCanonicalCommit, WaveCanonicalFailure> {
        assert_eq!(
            commit.repl.peek_next_index(),
            expected_commit_seq,
            "commit-path invariant violation: canonical append lost the serial wave sequence"
        );
        let wal_len_before = commit.wal.len();
        let token = commit
            .repl
            .propose(Arc::clone(&wal_payload))
            .map_err(|error| WaveCanonicalFailure::PreDurable(ExecuteError::Engine(error)))?;
        assert_eq!(
            token.index, expected_commit_seq,
            "the sequencer is the single proposer: the proposed index must equal the peek"
        );
        let record = match Self::canonical_wal_record_with_commit_outcome(
            commit,
            txn_id,
            token.index,
            0,
            &wal_payload,
            request_digest,
            outcome_kind,
            affected_rows,
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
        if let Err(error) = commit.repl.wait_committed(token, Duration::from_millis(0)) {
            commit.wal.truncate(wal_len_before);
            commit.repl.rollback_unapplied_from(expected_commit_seq);
            return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                error,
            )));
        }
        if let Err(error) = commit.record_transaction_status_digest_outcome(
            txn_id,
            request_digest,
            expected_commit_seq,
            affected_rows,
        ) {
            commit.wal.truncate(wal_len_before);
            commit.repl.rollback_unapplied_from(expected_commit_seq);
            return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                error,
            )));
        }
        let timestamp_micros = wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
        commit.record_commit_timestamp(txn_id, timestamp_micros);
        commit.ledger.record(write_set, expected_commit_seq);
        Ok(ResolvedWaveCanonicalCommit {
            commit_seq: expected_commit_seq,
            wal_position: commit.wal.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_post_wal_permit_panics_if_normally_dropped_without_device_apply() {
        let result = std::panic::catch_unwind(|| {
            drop(issue_test_only_typed_insert_post_wal_apply_permit(7));
        });
        assert!(result.is_err());
    }
}
