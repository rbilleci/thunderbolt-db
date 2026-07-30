use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term};
use std::sync::atomic::{AtomicU64, Ordering};

use super::{LogReplicator, RecoveryProgressGap, ReplicationProgress, ReplicationStatusSnapshot};

static NEXT_PROPOSAL_RESERVATION_OWNER_ID: AtomicU64 = AtomicU64::new(1);

fn next_proposal_reservation_owner_id() -> u64 {
    let owner_id = NEXT_PROPOSAL_RESERVATION_OWNER_ID.fetch_add(1, Ordering::Relaxed);
    assert_ne!(
        owner_id, 0,
        "LocalReplicator proposal reservation owner identifiers exhausted"
    );
    owner_id
}

#[derive(Debug)]
pub struct LocalReplicator {
    term: Term,
    next_index: Index,
    commit_index: Index,
    applied_index: Index,
    applied_term: Term,
    role: Role,
    entries: Vec<LogEntry>,
    /// Bumped after every mutation that can make an outstanding reserved proposal stale.  A
    /// reservation carries this value in addition to the exact frontier/length binding, so a
    /// caller cannot keep a Vec slot across an intervening log transition and replay it later.
    proposal_reservation_generation: u64,
    /// Stable, process-local identity carried by every reservation so equal frontiers from two
    /// independent replicators cannot consume one another's capacity token.
    proposal_reservation_owner_id: u64,
    snapshot_id: u64,
}

/// One fallibly acquired, exact slot in [`LocalReplicator::entries`] for the next leader
/// proposal.
///
/// The fields are deliberately private and this type is neither `Clone` nor `Copy`: only a
/// [`LocalReplicator`] can issue it and only the same unmodified replicator can consume it through
/// [`LocalReplicator::propose_reserved`].  Reserving capacity is not a log mutation; dropping an
/// unused reservation therefore has no logical effect.
#[must_use = "a reserved replication proposal slot must be consumed or deliberately abandoned before WAL"]
pub struct LocalReplicatorProposalReservation {
    expected_owner_id: u64,
    expected_next_index: Index,
    expected_entries_len: usize,
    expected_entries_capacity: usize,
    expected_term: Term,
    expected_commit_index: Index,
    expected_generation: u64,
}

impl LocalReplicator {
    pub fn leader() -> Self {
        Self {
            term: 1,
            next_index: 1,
            commit_index: 0,
            applied_index: 0,
            applied_term: 0,
            role: Role::Leader,
            entries: Vec::new(),
            proposal_reservation_generation: 0,
            proposal_reservation_owner_id: next_proposal_reservation_owner_id(),
            snapshot_id: 0,
        }
    }

    fn invalidate_proposal_reservations(&mut self) {
        self.proposal_reservation_generation = self
            .proposal_reservation_generation
            .checked_add(1)
            .expect("LocalReplicator proposal reservation generation exhausted");
    }

    pub fn become_follower(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Follower;
        self.invalidate_proposal_reservations();
    }

    pub fn become_leader(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Leader;
        self.invalidate_proposal_reservations();
    }

    pub fn become_candidate(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Candidate;
        self.invalidate_proposal_reservations();
    }

    /// The `Index` the NEXT [`LogReplicator::propose`] will assign, WITHOUT consuming it. The
    /// concurrent commit path peeks this (under the single-proposer commit_mutex) to re-resolve a
    /// transaction's delta at its would-be `commit_seq` BEFORE proposing, so a re-validation failure
    /// can abort without ever consuming an index (no commit-seq hole, nothing durable).
    pub fn peek_next_index(&self) -> Index {
        self.next_index
    }

    /// O(1) lookup of the retained log entry at `index`. The log is contiguous in `index` (prefix-
    /// compacted by `install_snapshot`, suffix-trimmed by `rollback_unapplied_from`), so the position
    /// is `index - entries[0].index`. `None` if `index` was prefix-compacted away or is past the tail.
    pub(super) fn entry_at(&self, index: Index) -> Option<&LogEntry> {
        let first = self.entries.first()?.index;
        let pos = index.checked_sub(first)? as usize;
        let entry = self.entries.get(pos)?;
        debug_assert_eq!(
            entry.index, index,
            "LocalReplicator log must stay contiguous in index"
        );
        Some(entry)
    }

    /// Committed entries with index in `(start_exclusive, commit_index]`, in order. O(k) in the
    /// number yielded, NOT O(entries): the contiguous log lets the window map directly to a slice
    /// range, so the commit hot path no longer scans the unbounded entries vec. Yields exactly the
    /// same set as the former `entries.iter().filter(index > start && index <= commit_index)`.
    pub fn drain_committed_from(&self, start_exclusive: Index) -> impl Iterator<Item = &LogEntry> {
        let (start_pos, end_pos) = match self.entries.first().map(|e| e.index) {
            Some(first) if self.commit_index >= first => {
                let lo = start_exclusive.saturating_add(1).max(first);
                let start = (lo - first) as usize;
                let end = ((self.commit_index - first) as usize + 1).min(self.entries.len());
                (start.min(end), end)
            }
            _ => (0, 0),
        };
        self.entries[start_pos..end_pos].iter()
    }

    pub fn retained_entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn committed_but_unapplied_count(&self) -> usize {
        self.commit_index.saturating_sub(self.applied_index) as usize
    }

    pub fn has_committed_entries_pending_apply(&self) -> bool {
        self.commit_index > self.applied_index
    }

    pub fn mark_applied(&mut self, idx: Index) {
        let bounded = idx.min(self.commit_index);
        if bounded <= self.applied_index {
            return;
        }

        self.applied_index = bounded;
        // O(1) via the contiguous-log index->position map (was an O(n) `entries.iter().find`).
        if let Some(term) = self.entry_at(bounded).map(|entry| entry.term) {
            self.applied_term = term;
        }
        self.invalidate_proposal_reservations();
    }

    /// W1b: drop the applied prefix (see [`LogReplicator::compact_applied_prefix`]). Keeps the
    /// log contiguous-from-first (the `entry_at` invariant): only a PREFIX is removed.
    pub fn compact_applied_prefix_inner(&mut self) {
        let applied = self.applied_index;
        self.entries.retain(|e| e.index > applied);
        self.invalidate_proposal_reservations();
    }

    /// E2.5b — batch propose for the wave sequencer: append `payloads` as consecutive leader
    /// entries and advance the commit index ONCE. Returns the FIRST assigned index; the batch
    /// covers `[first, first + n)`. Semantically identical to `propose` called in a loop (the
    /// single-node leader commits immediately; all-or-nothing on the leadership check, and the
    /// caller aborts the whole wave on error exactly as a first-item `propose` failure would).
    /// Exists because the wave commit cut is the write path's serial section and per-item
    /// propose/wait/mark round-trips were measured as pure bookkeeping overhead there.
    pub fn propose_batch<I>(&mut self, payloads: I) -> Result<Index, EngineError>
    where
        I: IntoIterator<Item = std::sync::Arc<[u8]>>,
    {
        if self.role != Role::Leader {
            return Err(EngineError::NotLeader);
        }
        let first = self.next_index;
        let payloads: Vec<_> = payloads.into_iter().collect();
        let count = u64::try_from(payloads.len()).map_err(|_| {
            EngineError::ProposalFailed("commit sequence batch length overflow".to_string())
        })?;
        if count > 0 {
            let last = first.checked_add(count - 1).ok_or_else(|| {
                EngineError::ProposalFailed("commit sequence space exhausted".to_string())
            })?;
            if last == u64::MAX {
                return Err(EngineError::ProposalFailed(
                    "commit sequence space exhausted".to_string(),
                ));
            }
        }
        self.entries.reserve(payloads.len());
        for payload in payloads {
            let idx = self.next_index;
            self.next_index = self.next_index.checked_add(1).ok_or_else(|| {
                EngineError::ProposalFailed("commit sequence space exhausted".to_string())
            })?;
            self.entries.push(LogEntry {
                term: self.term,
                index: idx,
                payload,
            });
        }
        if self.next_index > first {
            self.commit_index = self.next_index - 1;
        }
        if self.next_index != first {
            self.invalidate_proposal_reservations();
        }
        Ok(first)
    }

    /// Fallibly reserve the exact `entries` slot that the next typed canonical proposal will use.
    /// This must be acquired before the caller crosses its WAL claim boundary.  The resulting
    /// token is bound to the exact leader term, sequence frontier, log length/capacity, and
    /// mutation generation; any intervening local-replication mutation rejects consumption.
    pub fn reserve_next_proposal(
        &mut self,
    ) -> Result<LocalReplicatorProposalReservation, EngineError> {
        if self.role != Role::Leader {
            return Err(EngineError::NotLeader);
        }
        if self.next_index == u64::MAX {
            return Err(EngineError::ProposalFailed(
                "commit sequence space exhausted".to_string(),
            ));
        }
        self.entries.try_reserve_exact(1).map_err(|_| {
            EngineError::ProposalFailed(
                "unable to reserve the next local replication entry before WAL".to_string(),
            )
        })?;
        Ok(LocalReplicatorProposalReservation {
            expected_owner_id: self.proposal_reservation_owner_id,
            expected_next_index: self.next_index,
            expected_entries_len: self.entries.len(),
            expected_entries_capacity: self.entries.capacity(),
            expected_term: self.term,
            expected_commit_index: self.commit_index,
            expected_generation: self.proposal_reservation_generation,
        })
    }

    /// Consume one [`Self::reserve_next_proposal`] token without letting `entries.push` grow its
    /// backing allocation.  This is intentionally separate from [`LogReplicator::propose`]:
    /// historical and resolved callers retain their legacy path, while typed canonical callers
    /// cannot accidentally lose their pre-WAL capacity proof.
    pub fn propose_reserved(
        &mut self,
        reservation: LocalReplicatorProposalReservation,
        payload: std::sync::Arc<[u8]>,
    ) -> Result<CommitToken, EngineError> {
        if self.role != Role::Leader
            || self.proposal_reservation_owner_id != reservation.expected_owner_id
            || self.next_index != reservation.expected_next_index
            || self.entries.len() != reservation.expected_entries_len
            || self.entries.capacity() != reservation.expected_entries_capacity
            || self.term != reservation.expected_term
            || self.commit_index != reservation.expected_commit_index
            || self.proposal_reservation_generation != reservation.expected_generation
        {
            return Err(EngineError::ProposalFailed(
                "reserved local replication proposal state drifted before consumption".to_string(),
            ));
        }
        debug_assert!(self.entries.len() < self.entries.capacity());
        let capacity_before = self.entries.capacity();
        let idx = self.next_index;
        self.next_index = self
            .next_index
            .checked_add(1)
            .expect("reserved maximum commit sequence was refused");
        self.entries.push(LogEntry {
            term: self.term,
            index: idx,
            payload,
        });
        assert_eq!(
            self.entries.capacity(),
            capacity_before,
            "reserved local replication proposal unexpectedly grew entries"
        );
        self.commit_index = idx;
        self.invalidate_proposal_reservations();
        Ok(CommitToken { index: idx })
    }

    pub fn rollback_unapplied_from(&mut self, index_inclusive: Index) {
        if index_inclusive <= self.applied_index {
            return;
        }

        self.entries.retain(|e| e.index < index_inclusive);
        self.commit_index = self
            .entries
            .last()
            .map(|e| e.index)
            .unwrap_or(self.applied_index);
        self.next_index = self.commit_index.saturating_add(1);
        self.invalidate_proposal_reservations();
    }

    pub fn export_snapshot_meta(&mut self) -> SnapshotMeta {
        self.snapshot_id += 1;
        self.invalidate_proposal_reservations();
        self.snapshot_meta()
    }

    pub fn install_snapshot(&mut self, meta: SnapshotMeta) {
        let current = self.snapshot_meta();
        let advances_frontier = meta.last_included_index > current.last_included_index;
        let same_frontier_same_term = meta.last_included_index == current.last_included_index
            && meta.last_included_term == current.last_included_term;
        let regresses_term_on_advanced_frontier =
            advances_frontier && meta.last_included_term < current.last_included_term;
        if regresses_term_on_advanced_frontier || (!advances_frontier && !same_frontier_same_term) {
            return;
        }

        self.term = self.term.max(meta.last_included_term);
        self.commit_index = self.commit_index.max(meta.last_included_index);
        if meta.last_included_index > self.applied_index {
            self.applied_index = meta.last_included_index;
            self.applied_term = meta.last_included_term;
        }
        self.snapshot_id = meta.snapshot_id;
        self.entries.retain(|e| e.index > meta.last_included_index);

        let tail_index = self
            .entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.commit_index);
        self.next_index = tail_index.saturating_add(1);
        self.invalidate_proposal_reservations();
    }

    pub fn progress(&self) -> ReplicationProgress {
        let progress = ReplicationProgress {
            role: self.role,
            term: self.term,
            commit_index: self.commit_index,
            applied_index: self.applied_index,
            next_index: self.next_index,
            snapshot: self.snapshot_meta(),
            committed_but_unapplied_count: self.committed_but_unapplied_count(),
            has_committed_entries_pending_apply: self.has_committed_entries_pending_apply(),
            uncommitted_entry_count: 0,
            has_uncommitted_entries: false,
        };
        progress
            .validate()
            .expect("local replication progress invariants should hold");
        progress
    }

    pub fn status_snapshot(&self) -> ReplicationStatusSnapshot {
        let live = self.progress();
        ReplicationStatusSnapshot::new(
            live.clone(),
            live,
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 0,
                uncommitted_entry_gap: 0,
            },
        )
        .expect("local replication status snapshot invariants should hold")
    }
}

impl LogReplicator for LocalReplicator {
    fn compact_applied_prefix(&mut self) {
        self.compact_applied_prefix_inner();
    }

    fn propose(&mut self, payload: std::sync::Arc<[u8]>) -> Result<CommitToken, EngineError> {
        if self.role != Role::Leader {
            return Err(EngineError::NotLeader);
        }
        if self.next_index == u64::MAX {
            return Err(EngineError::ProposalFailed(
                "commit sequence space exhausted".to_string(),
            ));
        }

        let idx = self.next_index;
        self.next_index = self
            .next_index
            .checked_add(1)
            .expect("reserved maximum commit sequence was refused");

        let entry = LogEntry {
            term: self.term,
            index: idx,
            payload,
        };

        self.entries.push(entry);
        self.commit_index = idx;
        self.invalidate_proposal_reservations();

        Ok(CommitToken { index: idx })
    }

    fn wait_committed(
        &self,
        token: CommitToken,
        _timeout: std::time::Duration,
    ) -> Result<Index, EngineError> {
        if self.commit_index >= token.index {
            Ok(token.index)
        } else {
            Err(EngineError::ProposalFailed(format!(
                "token {} is not committed yet (commit_index={})",
                token.index, self.commit_index
            )))
        }
    }

    fn role(&self) -> Role {
        self.role
    }

    fn current_term(&self) -> Term {
        self.term
    }

    fn commit_index(&self) -> Index {
        self.commit_index
    }

    fn applied_index(&self) -> Index {
        self.applied_index
    }

    fn snapshot_meta(&self) -> SnapshotMeta {
        SnapshotMeta {
            last_included_index: self.applied_index,
            last_included_term: self.applied_term,
            snapshot_id: self.snapshot_id,
        }
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    #[test]
    fn last_valid_commit_is_checkpointable_but_reserved_sentinel_is_not_claimed() {
        let mut repl = LocalReplicator::leader();
        repl.install_snapshot(SnapshotMeta {
            last_included_index: u64::MAX - 1,
            last_included_term: 1,
            snapshot_id: 1,
        });
        let before = repl.progress();
        let error = repl.propose(std::sync::Arc::from(&b"x"[..])).unwrap_err();
        assert!(error.to_string().contains("sequence space exhausted"));
        assert_eq!(repl.progress(), before);
        let error = repl
            .propose_batch([std::sync::Arc::from(&b"x"[..])])
            .unwrap_err();
        assert!(error.to_string().contains("sequence space exhausted"));
        assert_eq!(repl.progress(), before);
    }

    #[test]
    fn reserved_proposal_consumes_the_exact_slot_without_growing_entries() {
        let mut repl = LocalReplicator::leader();
        let reservation = repl.reserve_next_proposal().expect("reserve one entry");
        let capacity = repl.entries.capacity();
        let token = repl
            .propose_reserved(reservation, std::sync::Arc::from(&b"typed"[..]))
            .expect("consume reserved entry");

        assert_eq!(token.index, 1);
        assert_eq!(repl.entries.len(), 1);
        assert_eq!(repl.entries.capacity(), capacity);
        assert_eq!(repl.commit_index, 1);
    }

    #[test]
    fn reserved_proposal_rejects_generation_and_cross_owner_drift() {
        let mut repl = LocalReplicator::leader();
        let reservation = repl.reserve_next_proposal().expect("reserve one entry");
        repl.export_snapshot_meta();
        let before = repl.progress();
        let error = repl
            .propose_reserved(reservation, std::sync::Arc::from(&b"typed"[..]))
            .expect_err("intervening mutation must reject the token");
        assert!(error.to_string().contains("state drifted"));
        assert_eq!(repl.progress(), before);

        let mut first = LocalReplicator::leader();
        let reservation = first.reserve_next_proposal().expect("reserve first owner");
        let mut second = LocalReplicator::leader();
        let before = second.progress();
        let error = second
            .propose_reserved(reservation, std::sync::Arc::from(&b"typed"[..]))
            .expect_err("another replicator must reject this token");
        assert!(error.to_string().contains("state drifted"));
        assert_eq!(second.progress(), before);
    }
}
