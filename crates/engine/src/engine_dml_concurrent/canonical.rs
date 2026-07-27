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
    FixedInsert {
        proposal_payload: Arc<[u8]>,
        bound: crate::wal_binary::BoundBinaryInsert,
    },
}

pub(super) struct WaveCanonicalCommit {
    pub(super) commit_seq: Index,
    pub(super) wal_position: usize,
    pub(super) proposed_range: Option<crate::wal_binary::ProposedRowIdRange>,
}

pub(super) enum WaveCanonicalFailure {
    PreDurable(ExecuteError),
}

impl Engine {
    /// The sole serial-wave proposal -> canonical record -> append -> status buffer -> ledger
    /// sequence. It returns before device apply, physical group durability, and publication/ack;
    /// `FixedInsert` is only a sealed-operation input, not another WAL/status authority.
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
        let proposal_payload = match &operation {
            WaveCanonicalOperation::Resolved { wal_payload, .. } => Arc::clone(wal_payload),
            WaveCanonicalOperation::FixedInsert {
                proposal_payload, ..
            } => Arc::clone(proposal_payload),
        };
        let wal_len_before = commit.wal.len();
        let token = commit
            .repl
            .propose(proposal_payload)
            .map_err(|error| WaveCanonicalFailure::PreDurable(ExecuteError::Engine(error)))?;
        debug_assert_eq!(
            token.index, expected_commit_seq,
            "the sequencer is the single proposer: the proposed index must equal the peek"
        );

        let (record, proposed_range, affected_rows) = match operation {
            WaveCanonicalOperation::Resolved {
                wal_payload,
                outcome_kind,
                affected_rows,
            } => match Self::canonical_wal_record_with_commit_outcome(
                commit,
                txn_id,
                token.index,
                0,
                &wal_payload,
                request_digest,
                outcome_kind,
                affected_rows,
            ) {
                Ok(record) => (record, None, affected_rows),
                Err(error) => {
                    commit.repl.rollback_unapplied_from(token.index);
                    return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                        error,
                    )));
                }
            },
            WaveCanonicalOperation::FixedInsert { bound, .. } => {
                match Self::canonical_wal_record_with_commit_bound_insert(
                    commit,
                    txn_id,
                    token.index,
                    0,
                    bound,
                    request_digest,
                ) {
                    Ok(prepared) => {
                        let (record, proposed_range) = prepared.into_record_and_proposed_range();
                        let affected_rows = u64::from(proposed_range.count());
                        (record, Some(proposed_range), affected_rows)
                    }
                    Err(error) => {
                        commit.repl.rollback_unapplied_from(token.index);
                        return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                            error,
                        )));
                    }
                }
            }
        };
        commit.wal.append_canonical(record);
        let wal_position = commit.wal.len();
        if let Err(error) = commit.repl.wait_committed(token, Duration::from_millis(0)) {
            commit.repl.rollback_unapplied_from(expected_commit_seq);
            commit.wal.truncate(wal_len_before);
            return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                error,
            )));
        }
        if let Err(error) = commit.record_transaction_status_digest_outcome(
            txn_id,
            request_digest,
            token.index,
            affected_rows,
        ) {
            commit.repl.rollback_unapplied_from(expected_commit_seq);
            commit.wal.truncate(wal_len_before);
            return Err(WaveCanonicalFailure::PreDurable(ExecuteError::Engine(
                error,
            )));
        }
        let timestamp_micros = wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
        commit.record_commit_timestamp(txn_id, timestamp_micros);
        commit.ledger.record(write_set, expected_commit_seq);
        Ok(WaveCanonicalCommit {
            commit_seq: expected_commit_seq,
            wal_position,
            proposed_range,
        })
    }
}
