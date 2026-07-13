use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term};

use super::{LogReplicator, RecoveryProgressGap, ReplicationProgress, ReplicationStatusSnapshot};

#[derive(Debug)]
pub struct LocalReplicator {
    term: Term,
    next_index: Index,
    commit_index: Index,
    applied_index: Index,
    applied_term: Term,
    role: Role,
    entries: Vec<LogEntry>,
    snapshot_id: u64,
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
            snapshot_id: 0,
        }
    }

    pub fn become_follower(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Follower;
    }

    pub fn become_leader(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Leader;
    }

    pub fn become_candidate(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Candidate;
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
    }

    /// W1b: drop the applied prefix (see [`LogReplicator::compact_applied_prefix`]). Keeps the
    /// log contiguous-from-first (the `entry_at` invariant): only a PREFIX is removed.
    pub fn compact_applied_prefix_inner(&mut self) {
        let applied = self.applied_index;
        self.entries.retain(|e| e.index > applied);
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
        let payloads = payloads.into_iter();
        self.entries.reserve(payloads.size_hint().0);
        for payload in payloads {
            let idx = self.next_index;
            self.next_index += 1;
            self.entries.push(LogEntry {
                term: self.term,
                index: idx,
                payload,
            });
        }
        if self.next_index > first {
            self.commit_index = self.next_index - 1;
        }
        Ok(first)
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
        self.next_index = self.commit_index + 1;
    }

    pub fn export_snapshot_meta(&mut self) -> SnapshotMeta {
        self.snapshot_id += 1;
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
        self.next_index = tail_index + 1;
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

        let idx = self.next_index;
        self.next_index += 1;

        let entry = LogEntry {
            term: self.term,
            index: idx,
            payload,
        };

        self.entries.push(entry);
        self.commit_index = idx;

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
