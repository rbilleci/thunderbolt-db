use std::collections::{BTreeMap, BTreeSet};

use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryState {
    pub term: Term,
    pub snapshot: SnapshotMeta,
    pub committed_entries: Vec<LogEntry>,
    pub applied_index: Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationProgress {
    pub role: Role,
    pub term: Term,
    pub commit_index: Index,
    pub applied_index: Index,
    pub next_index: Index,
    pub snapshot: SnapshotMeta,
    pub committed_but_unapplied_count: usize,
    pub has_committed_entries_pending_apply: bool,
    pub uncommitted_entry_count: usize,
    pub has_uncommitted_entries: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReplicationProgressInvariantError {
    #[error("applied_index {applied_index} exceeds commit_index {commit_index}")]
    AppliedExceedsCommit {
        applied_index: Index,
        commit_index: Index,
    },
    #[error("next_index {next_index} is behind commit boundary {commit_index}")]
    NextIndexBehindCommit {
        next_index: Index,
        commit_index: Index,
    },
    #[error(
        "committed_but_unapplied_count {committed_but_unapplied_count} does not match commit/apply gap {expected}"
    )]
    PendingApplyCountMismatch {
        committed_but_unapplied_count: usize,
        expected: usize,
    },
    #[error(
        "has_committed_entries_pending_apply {has_pending} does not match commit/apply gap {expected}"
    )]
    PendingApplyFlagMismatch { has_pending: bool, expected: bool },
    #[error(
        "has_uncommitted_entries {has_uncommitted} does not match uncommitted_entry_count {uncommitted_entry_count}"
    )]
    UncommittedFlagMismatch {
        has_uncommitted: bool,
        uncommitted_entry_count: usize,
    },
    #[error("snapshot last_included_index {snapshot_index} exceeds applied_index {applied_index}")]
    SnapshotAheadOfApplied {
        snapshot_index: Index,
        applied_index: Index,
    },
}

impl ReplicationProgress {
    pub fn apply_gap(&self) -> usize {
        self.commit_index.saturating_sub(self.applied_index) as usize
    }

    pub fn is_caught_up(&self) -> bool {
        self.apply_gap() == 0 && !self.has_uncommitted_entries
    }

    pub fn validate(&self) -> Result<(), ReplicationProgressInvariantError> {
        if self.applied_index > self.commit_index {
            return Err(ReplicationProgressInvariantError::AppliedExceedsCommit {
                applied_index: self.applied_index,
                commit_index: self.commit_index,
            });
        }
        if self.next_index < self.commit_index + 1 {
            return Err(ReplicationProgressInvariantError::NextIndexBehindCommit {
                next_index: self.next_index,
                commit_index: self.commit_index,
            });
        }

        let expected_pending = self.apply_gap();
        if self.committed_but_unapplied_count != expected_pending {
            return Err(
                ReplicationProgressInvariantError::PendingApplyCountMismatch {
                    committed_but_unapplied_count: self.committed_but_unapplied_count,
                    expected: expected_pending,
                },
            );
        }

        let expected_has_pending = expected_pending > 0;
        if self.has_committed_entries_pending_apply != expected_has_pending {
            return Err(
                ReplicationProgressInvariantError::PendingApplyFlagMismatch {
                    has_pending: self.has_committed_entries_pending_apply,
                    expected: expected_has_pending,
                },
            );
        }

        let expected_has_uncommitted = self.uncommitted_entry_count > 0;
        if self.has_uncommitted_entries != expected_has_uncommitted {
            return Err(ReplicationProgressInvariantError::UncommittedFlagMismatch {
                has_uncommitted: self.has_uncommitted_entries,
                uncommitted_entry_count: self.uncommitted_entry_count,
            });
        }

        if self.snapshot.last_included_index > self.applied_index {
            return Err(ReplicationProgressInvariantError::SnapshotAheadOfApplied {
                snapshot_index: self.snapshot.last_included_index,
                applied_index: self.applied_index,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecoveryInvariantError {
    #[error("applied_index {applied_index} is behind snapshot boundary {snapshot_index}")]
    AppliedBehindSnapshot {
        applied_index: Index,
        snapshot_index: Index,
    },
    #[error("recovery entries must be contiguous from {expected_index} but saw {actual_index}")]
    NonContiguousEntries {
        expected_index: Index,
        actual_index: Index,
    },
    #[error("recovery entry term {entry_term} exceeds local term {local_term} at index {index}")]
    EntryTermExceedsLocal {
        entry_term: Term,
        local_term: Term,
        index: Index,
    },
    #[error("applied_index {applied_index} exceeds commit boundary {commit_index}")]
    AppliedExceedsCommit {
        applied_index: Index,
        commit_index: Index,
    },
}

impl RecoveryState {
    pub fn commit_index(&self) -> Index {
        self.committed_entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.snapshot.last_included_index)
    }

    pub fn next_index(&self) -> Index {
        self.commit_index() + 1
    }

    pub fn committed_but_unapplied_count(&self) -> usize {
        self.commit_index().saturating_sub(self.applied_index) as usize
    }

    pub fn has_committed_entries_pending_apply(&self) -> bool {
        self.applied_index < self.commit_index()
    }

    pub fn apply_gap(&self) -> usize {
        self.commit_index().saturating_sub(self.applied_index) as usize
    }

    pub fn is_caught_up(&self) -> bool {
        self.apply_gap() == 0
    }

    pub fn progress_as_follower(&self) -> Result<ReplicationProgress, RecoveryInvariantError> {
        self.validate()?;
        let progress = ReplicationProgress {
            role: Role::Follower,
            term: self.term.max(self.snapshot.last_included_term),
            commit_index: self.commit_index(),
            applied_index: self.applied_index,
            next_index: self.next_index(),
            snapshot: self.snapshot.clone(),
            committed_but_unapplied_count: self.committed_but_unapplied_count(),
            has_committed_entries_pending_apply: self.has_committed_entries_pending_apply(),
            uncommitted_entry_count: 0,
            has_uncommitted_entries: false,
        };
        progress.validate().expect(
            "validated recovery state should always map to valid follower replication progress",
        );
        Ok(progress)
    }

    pub fn validate(&self) -> Result<(), RecoveryInvariantError> {
        if self.applied_index < self.snapshot.last_included_index {
            return Err(RecoveryInvariantError::AppliedBehindSnapshot {
                applied_index: self.applied_index,
                snapshot_index: self.snapshot.last_included_index,
            });
        }

        let mut expected_index = self.snapshot.last_included_index + 1;
        for entry in &self.committed_entries {
            if entry.index != expected_index {
                return Err(RecoveryInvariantError::NonContiguousEntries {
                    expected_index,
                    actual_index: entry.index,
                });
            }
            if entry.term > self.term {
                return Err(RecoveryInvariantError::EntryTermExceedsLocal {
                    entry_term: entry.term,
                    local_term: self.term,
                    index: entry.index,
                });
            }
            expected_index += 1;
        }

        let commit_index = self.commit_index();
        if self.applied_index > commit_index {
            return Err(RecoveryInvariantError::AppliedExceedsCommit {
                applied_index: self.applied_index,
                commit_index,
            });
        }

        Ok(())
    }
}

pub trait LogReplicator {
    fn propose(&mut self, payload: Vec<u8>) -> Result<CommitToken, EngineError>;
    fn wait_committed(
        &self,
        token: CommitToken,
        timeout: std::time::Duration,
    ) -> Result<Index, EngineError>;
    fn role(&self) -> Role;
    fn current_term(&self) -> Term;
    fn commit_index(&self) -> Index;
    fn applied_index(&self) -> Index;
    fn snapshot_meta(&self) -> SnapshotMeta;
}

pub trait ReplicatedStateMachine {
    fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError>;
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

    pub fn drain_committed_from(&self, start_exclusive: Index) -> impl Iterator<Item = &LogEntry> {
        self.entries
            .iter()
            .filter(move |e| e.index > start_exclusive && e.index <= self.commit_index)
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
        if let Some(entry) = self.entries.iter().find(|entry| entry.index == bounded) {
            self.applied_term = entry.term;
        }
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
        self.term = self.term.max(meta.last_included_term);
        self.commit_index = self.commit_index.max(meta.last_included_index);
        if meta.last_included_index > self.applied_index {
            self.applied_index = meta.last_included_index;
            self.applied_term = meta.last_included_term;
        }
        self.snapshot_id = self.snapshot_id.max(meta.snapshot_id);
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
}

#[derive(Debug)]
pub struct RaftReplicator {
    term: Term,
    next_index: Index,
    commit_index: Index,
    applied_index: Index,
    applied_term: Term,
    role: Role,
    entries: Vec<LogEntry>,
    snapshot_id: u64,
    compacted_index: Index,
    compacted_term: Term,
    voters: usize,
    quorum: usize,
    ack_counts: BTreeMap<Index, BTreeSet<u64>>,
}

impl RaftReplicator {
    pub fn new(voters: usize) -> Self {
        assert!(voters >= 1, "raft requires at least one voter");
        let quorum = (voters / 2) + 1;
        Self {
            term: 1,
            next_index: 1,
            commit_index: 0,
            applied_index: 0,
            applied_term: 0,
            role: Role::Follower,
            entries: Vec::new(),
            snapshot_id: 0,
            compacted_index: 0,
            compacted_term: 0,
            voters,
            quorum,
            ack_counts: BTreeMap::new(),
        }
    }

    pub fn single_node_leader() -> Self {
        let mut s = Self::new(1);
        s.role = Role::Leader;
        s
    }

    pub fn resume_as_follower(voters: usize, recovery: RecoveryState) -> Result<Self, EngineError> {
        recovery
            .validate()
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;

        let commit_index = recovery.commit_index();
        let next_index = recovery.next_index();

        let mut replicator = Self::new(voters);
        replicator.term = recovery.term.max(recovery.snapshot.last_included_term);
        replicator.role = Role::Follower;
        replicator.commit_index = commit_index;
        replicator.applied_index = recovery.applied_index;
        replicator.applied_term = if recovery.applied_index == recovery.snapshot.last_included_index
        {
            recovery.snapshot.last_included_term
        } else {
            recovery
                .committed_entries
                .iter()
                .find(|entry| entry.index == recovery.applied_index)
                .map(|entry| entry.term)
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "missing applied entry {} in recovery state",
                        recovery.applied_index
                    ))
                })?
        };
        replicator.snapshot_id = recovery.snapshot.snapshot_id;
        replicator.compacted_index = recovery.snapshot.last_included_index;
        replicator.compacted_term = recovery.snapshot.last_included_term;
        replicator.entries = recovery.committed_entries;
        replicator.next_index = next_index;
        Ok(replicator)
    }

    pub fn become_follower(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Follower;
        self.entries.retain(|e| e.index <= self.commit_index);
        self.next_index = self.commit_index + 1;
        self.ack_counts.clear();
    }

    pub fn become_leader(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Leader;
        self.entries.retain(|e| e.index <= self.commit_index);
        self.next_index = self.commit_index + 1;
        self.ack_counts.clear();
    }

    pub fn become_candidate(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Candidate;
        self.entries.retain(|e| e.index <= self.commit_index);
        self.next_index = self.commit_index + 1;
        self.ack_counts.clear();
    }

    pub fn voter_count(&self) -> usize {
        self.voters
    }

    pub fn quorum_size(&self) -> usize {
        self.quorum
    }

    pub fn drain_committed_from(&self, start_exclusive: Index) -> impl Iterator<Item = &LogEntry> {
        self.entries
            .iter()
            .filter(move |e| e.index > start_exclusive && e.index <= self.commit_index)
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

    pub fn uncommitted_entry_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.index > self.commit_index)
            .count()
    }

    pub fn has_uncommitted_entries(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.index > self.commit_index)
    }

    pub fn mark_applied(&mut self, idx: Index) {
        let bounded = idx.min(self.commit_index);
        if bounded <= self.applied_index {
            return;
        }

        self.applied_index = bounded;
        if let Some(entry) = self.entries.iter().find(|entry| entry.index == bounded) {
            self.applied_term = entry.term;
        }
    }

    pub fn truncate_uncommitted_from(&mut self, index_inclusive: Index) {
        if index_inclusive <= self.commit_index {
            return;
        }

        self.entries.retain(|e| e.index < index_inclusive);
        self.ack_counts.retain(|idx, _| *idx < index_inclusive);

        let tail_index = self
            .entries
            .last()
            .map(|e| e.index)
            .unwrap_or(self.commit_index);
        self.next_index = tail_index + 1;
    }

    pub fn register_follower_ack(&mut self, index: Index, follower_id: u64) {
        if self.role != Role::Leader || index == 0 || index >= self.next_index || follower_id == 0 {
            return;
        }

        let Some(acks) = self.ack_counts.get_mut(&index) else {
            return;
        };
        acks.insert(follower_id);

        while self.commit_index + 1 < self.next_index {
            let next = self.commit_index + 1;
            let Some(acks) = self.ack_counts.get(&next) else {
                break;
            };
            if acks.len() >= self.quorum {
                self.commit_index = next;
            } else {
                break;
            }
        }

        self.ack_counts.retain(|idx, _| *idx > self.commit_index);
    }

    pub fn export_snapshot_meta(&mut self) -> SnapshotMeta {
        self.snapshot_id += 1;
        self.snapshot_meta()
    }

    pub fn recovery_state(&self) -> RecoveryState {
        let snapshot = self.snapshot_meta();
        RecoveryState {
            term: self.term,
            snapshot: snapshot.clone(),
            committed_entries: self
                .entries
                .iter()
                .filter(|entry| {
                    entry.index > snapshot.last_included_index && entry.index <= self.commit_index
                })
                .cloned()
                .collect(),
            applied_index: self.applied_index,
        }
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
            uncommitted_entry_count: self.uncommitted_entry_count(),
            has_uncommitted_entries: self.has_uncommitted_entries(),
        };
        progress
            .validate()
            .expect("raft replication progress invariants should hold");
        progress
    }

    pub fn append_entries_from_leader(
        &mut self,
        leader_term: Term,
        prev_log_index: Index,
        prev_log_term: Term,
        entries: Vec<LogEntry>,
        leader_commit: Index,
    ) -> Result<(), EngineError> {
        if self.role == Role::Leader {
            return Err(EngineError::ProposalFailed(
                "leader cannot accept follower append path".to_string(),
            ));
        }

        if leader_term < self.term {
            return Err(EngineError::ProposalFailed(format!(
                "stale leader term {} (local term {})",
                leader_term, self.term
            )));
        }

        self.term = leader_term;
        self.role = Role::Follower;

        if prev_log_index < self.compacted_index {
            return Err(EngineError::ProposalFailed(format!(
                "prev_log_index={} is behind compacted boundary {}",
                prev_log_index, self.compacted_index
            )));
        }

        if prev_log_index > 0 {
            let Some(local_prev_term) = self.term_at(prev_log_index) else {
                return Err(EngineError::ProposalFailed(format!(
                    "missing prev_log_index={} for append",
                    prev_log_index
                )));
            };

            if local_prev_term != prev_log_term {
                return Err(EngineError::ProposalFailed(format!(
                    "prev_log_term mismatch at index {}: local={}, remote={}",
                    prev_log_index, local_prev_term, prev_log_term
                )));
            }
        }

        let mut expected_index = prev_log_index + 1;
        for entry in &entries {
            if entry.term > leader_term {
                return Err(EngineError::ProposalFailed(format!(
                    "entry term {} exceeds leader term {} at index {}",
                    entry.term, leader_term, entry.index
                )));
            }

            if entry.index != expected_index {
                return Err(EngineError::ProposalFailed(format!(
                    "append entries must be contiguous from {} but saw {}",
                    prev_log_index + 1,
                    entry.index
                )));
            }
            expected_index += 1;
        }

        for incoming in entries {
            if let Some(existing) = self.entries.iter().find(|e| e.index == incoming.index) {
                if existing.term == incoming.term {
                    if existing.payload != incoming.payload {
                        return Err(EngineError::ProposalFailed(format!(
                            "payload mismatch at index {} term {}",
                            incoming.index, incoming.term
                        )));
                    }
                    continue;
                }

                if incoming.index <= self.commit_index {
                    return Err(EngineError::ProposalFailed(format!(
                        "refusing to overwrite committed index {}",
                        incoming.index
                    )));
                }

                self.truncate_uncommitted_from(incoming.index);
            }

            if self
                .entries
                .iter()
                .all(|entry| entry.index != incoming.index)
            {
                self.entries.push(incoming);
            }
        }

        self.entries.sort_by_key(|entry| entry.index);

        let last_local_index = self
            .entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.commit_index);
        let target_commit = leader_commit.min(last_local_index);
        if target_commit > self.commit_index {
            self.commit_index = target_commit;
        }
        self.next_index = last_local_index + 1;

        Ok(())
    }

    pub fn install_snapshot(&mut self, meta: SnapshotMeta) {
        self.term = self.term.max(meta.last_included_term);
        self.commit_index = self.commit_index.max(meta.last_included_index);
        if meta.last_included_index > self.compacted_index {
            self.compacted_index = meta.last_included_index;
            self.compacted_term = meta.last_included_term;
        }
        if meta.last_included_index > self.applied_index {
            self.applied_index = meta.last_included_index;
            self.applied_term = meta.last_included_term;
        }
        self.snapshot_id = self.snapshot_id.max(meta.snapshot_id);
        self.entries.retain(|e| e.index > meta.last_included_index);
        self.ack_counts
            .retain(|idx, _| *idx > meta.last_included_index);

        let tail_index = self
            .entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.commit_index);
        self.next_index = tail_index + 1;
    }

    fn term_at(&self, index: Index) -> Option<Term> {
        if index == 0 {
            return Some(0);
        }

        if index == self.compacted_index {
            return Some(self.compacted_term);
        }

        if index < self.compacted_index {
            return None;
        }

        if let Some(entry) = self.entries.iter().find(|entry| entry.index == index) {
            return Some(entry.term);
        }

        if index == self.applied_index {
            return Some(self.applied_term);
        }

        None
    }
}

impl LogReplicator for LocalReplicator {
    fn propose(&mut self, payload: Vec<u8>) -> Result<CommitToken, EngineError> {
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

impl LogReplicator for RaftReplicator {
    fn propose(&mut self, payload: Vec<u8>) -> Result<CommitToken, EngineError> {
        if self.role != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        let idx = self.next_index;
        self.next_index += 1;

        self.entries.push(LogEntry {
            term: self.term,
            index: idx,
            payload,
        });

        // Leader has an implicit self-ack represented by voter id 0.
        let mut acks = BTreeSet::new();
        acks.insert(0);
        self.ack_counts.insert(idx, acks);
        if self.quorum == 1 {
            self.commit_index = idx;
        }

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
mod tests {
    use super::*;

    #[test]
    fn commit_index_monotonic() {
        let mut r = LocalReplicator::leader();
        let a = r.propose(vec![1]).unwrap();
        let b = r.propose(vec![2]).unwrap();

        assert!(b.index > a.index);
        assert_eq!(r.commit_index(), b.index);
        assert!(r.applied_index() <= r.commit_index());
    }

    #[test]
    fn follower_rejects_writes() {
        let mut r = LocalReplicator::leader();
        r.become_follower(2);
        let err = r.propose(vec![1]).unwrap_err();
        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(r.current_term(), 2);
    }

    #[test]
    fn leader_accepts_after_promotion() {
        let mut r = LocalReplicator::leader();
        r.become_follower(2);
        r.become_leader(3);
        let tok = r.propose(vec![42]).unwrap();
        assert_eq!(tok.index, 1);
        assert_eq!(r.current_term(), 3);
    }

    #[test]
    fn candidate_rejects_writes() {
        let mut r = LocalReplicator::leader();
        r.become_candidate(2);

        let err = r.propose(vec![1]).unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(r.current_term(), 2);
        assert_eq!(r.role(), Role::Candidate);
    }

    #[test]
    fn rollback_unapplied_removes_tail_and_resets_indices() {
        let mut r = LocalReplicator::leader();
        let _ = r.propose(vec![1]).unwrap();
        let t2 = r.propose(vec![2]).unwrap();
        assert_eq!(r.commit_index(), t2.index);

        r.rollback_unapplied_from(t2.index);

        assert_eq!(r.commit_index(), 1);
        let t3 = r.propose(vec![3]).unwrap();
        assert_eq!(t3.index, 2);
    }

    #[test]
    fn snapshot_meta_tracks_applied_index() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1]).unwrap();
        r.mark_applied(t1.index);

        let meta = r.export_snapshot_meta();

        assert_eq!(meta.last_included_index, t1.index);
        assert_eq!(meta.last_included_term, r.current_term());
        assert_eq!(meta.snapshot_id, 1);
    }

    #[test]
    fn snapshot_meta_preserves_last_applied_term_across_term_bumps() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1]).unwrap();
        r.mark_applied(t1.index);

        r.become_follower(5);
        let meta = r.snapshot_meta();

        assert_eq!(r.current_term(), 5);
        assert_eq!(meta.last_included_index, t1.index);
        assert_eq!(meta.last_included_term, 1);
    }

    #[test]
    fn install_snapshot_advances_log_watermarks() {
        let mut r = LocalReplicator::leader();
        let _ = r.propose(vec![1]).unwrap();
        let t2 = r.propose(vec![2]).unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: t2.index,
            last_included_term: 2,
            snapshot_id: 9,
        });

        assert_eq!(r.commit_index(), t2.index);
        assert_eq!(r.applied_index(), t2.index);
        assert_eq!(r.current_term(), 2);
        assert_eq!(r.snapshot_meta().snapshot_id, 9);

        let t3 = r.propose(vec![3]).unwrap();
        assert_eq!(t3.index, t2.index + 1);
    }

    #[test]
    fn install_older_snapshot_is_ignored_for_commit_and_apply_indices() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1]).unwrap();
        r.mark_applied(t1.index);

        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index.saturating_sub(1),
            last_included_term: 1,
            snapshot_id: 10,
        });

        assert_eq!(r.commit_index(), t1.index);
        assert_eq!(r.applied_index(), t1.index);
        assert_eq!(r.snapshot_meta().snapshot_id, 10);
    }

    #[test]
    fn local_install_snapshot_preserves_next_index_from_uncompacted_tail() {
        let mut r = LocalReplicator::leader();
        let _t1 = r.propose(vec![1]).unwrap();
        let t2 = r.propose(vec![2]).unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: t2.index - 1,
            last_included_term: 1,
            snapshot_id: 11,
        });

        let next = r.propose(vec![3]).unwrap();
        assert_eq!(next.index, t2.index + 1);
    }

    #[test]
    fn mark_applied_does_not_exceed_commit_index() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1]).unwrap();

        r.mark_applied(t1.index + 10);

        assert_eq!(r.applied_index(), t1.index);
    }

    #[test]
    fn local_replicator_pending_apply_helpers_track_committed_tail() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1]).unwrap();
        let _t2 = r.propose(vec![2]).unwrap();

        assert_eq!(r.retained_entry_count(), 2);
        assert!(r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 2);

        r.mark_applied(t1.index);
        assert!(r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 1);

        r.mark_applied(r.commit_index());
        assert!(!r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 0);
    }

    #[test]
    fn local_progress_snapshot_matches_commit_and_apply_state() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1]).unwrap();
        let _t2 = r.propose(vec![2]).unwrap();
        r.mark_applied(t1.index);

        let progress = r.progress();

        assert_eq!(progress.role, Role::Leader);
        assert_eq!(progress.commit_index, 2);
        assert_eq!(progress.applied_index, 1);
        assert_eq!(progress.next_index, 3);
        assert_eq!(progress.committed_but_unapplied_count, 1);
        assert!(progress.has_committed_entries_pending_apply);
        assert_eq!(progress.uncommitted_entry_count, 0);
        assert!(!progress.has_uncommitted_entries);
        assert_eq!(progress.apply_gap(), 1);
        assert!(!progress.is_caught_up());
        progress.validate().unwrap();
    }

    #[test]
    fn raft_replicator_rejects_proposal_when_not_leader() {
        let mut r = RaftReplicator::new(3);
        let err = r.propose(vec![1]).unwrap_err();
        assert!(matches!(err, EngineError::NotLeader));
    }

    #[test]
    fn raft_candidate_rejects_proposal_and_drops_uncommitted_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        let _uncommitted = r.propose(vec![2]).unwrap();

        r.become_candidate(2);
        let err = r.propose(vec![3]).unwrap_err();
        assert!(matches!(err, EngineError::NotLeader));

        r.become_leader(3);
        let tok = r.propose(vec![4]).unwrap();
        assert_eq!(tok.index, t1.index + 1);
    }

    #[test]
    fn raft_replicator_commits_after_quorum_acks() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(2);

        let t1 = r.propose(vec![1]).unwrap();
        assert_eq!(r.commit_index(), 0, "self-ack is not quorum for 3 voters");

        r.register_follower_ack(t1.index, 1);

        assert_eq!(r.commit_index(), t1.index);
        assert_eq!(r.quorum_size(), 2);
        assert_eq!(r.voter_count(), 3);
    }

    #[test]
    fn raft_replicator_commit_index_advances_in_order() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![10]).unwrap();
        let t2 = r.propose(vec![20]).unwrap();

        r.register_follower_ack(t2.index, 2);
        assert_eq!(r.commit_index(), 0, "cannot skip index 1");

        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t2.index);
    }

    #[test]
    fn raft_replicator_entry_state_helpers_track_committed_and_uncommitted_work() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![10]).unwrap();
        let t2 = r.propose(vec![20]).unwrap();

        assert_eq!(r.retained_entry_count(), 2);
        assert!(r.has_uncommitted_entries());
        assert_eq!(r.uncommitted_entry_count(), 2);
        assert!(!r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 0);

        r.register_follower_ack(t1.index, 1);
        assert!(r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 1);
        assert!(r.has_uncommitted_entries());
        assert_eq!(r.uncommitted_entry_count(), 1);

        r.mark_applied(t1.index);
        assert!(!r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 0);

        r.register_follower_ack(t2.index, 1);
        assert!(r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 1);
        assert!(!r.has_uncommitted_entries());
        assert_eq!(r.uncommitted_entry_count(), 0);
    }

    #[test]
    fn raft_follower_lagging_apply_delay_exposes_pending_apply_until_catch_up() {
        let mut leader = RaftReplicator::new(3);
        leader.become_leader(3);
        let t1 = leader.propose(vec![10]).unwrap();
        let t2 = leader.propose(vec![20]).unwrap();
        leader.register_follower_ack(t1.index, 1);
        leader.register_follower_ack(t2.index, 1);

        let mut follower = RaftReplicator::new(3);
        follower.become_follower(3);
        follower
            .append_entries_from_leader(
                3,
                0,
                0,
                vec![
                    LogEntry {
                        term: 3,
                        index: t1.index,
                        payload: vec![10],
                    },
                    LogEntry {
                        term: 3,
                        index: t2.index,
                        payload: vec![20],
                    },
                ],
                t2.index,
            )
            .unwrap();

        assert_eq!(follower.commit_index(), t2.index);
        assert_eq!(follower.applied_index(), 0);
        assert!(follower.has_committed_entries_pending_apply());
        assert_eq!(follower.committed_but_unapplied_count(), 2);

        follower.mark_applied(t1.index);
        assert_eq!(follower.committed_but_unapplied_count(), 1);
        assert!(follower.has_committed_entries_pending_apply());

        follower.mark_applied(t2.index);
        assert_eq!(follower.applied_index(), t2.index);
        assert!(!follower.has_committed_entries_pending_apply());
        assert_eq!(follower.committed_but_unapplied_count(), 0);
    }

    #[test]
    fn raft_recovery_state_resumes_pending_apply_without_rewinding_commit() {
        let mut leader = RaftReplicator::new(3);
        leader.become_leader(4);
        let t1 = leader.propose(vec![1]).unwrap();
        let t2 = leader.propose(vec![2]).unwrap();
        leader.register_follower_ack(t1.index, 1);
        leader.register_follower_ack(t2.index, 1);
        leader.mark_applied(t1.index);
        let snapshot = leader.export_snapshot_meta();

        let resumed = RaftReplicator::resume_as_follower(3, leader.recovery_state()).unwrap();

        assert_eq!(resumed.role(), Role::Follower);
        assert_eq!(resumed.current_term(), 4);
        assert_eq!(resumed.commit_index(), t2.index);
        assert_eq!(resumed.applied_index(), t1.index);
        assert_eq!(resumed.snapshot_meta(), snapshot);
        assert!(resumed.has_committed_entries_pending_apply());
        assert_eq!(resumed.committed_but_unapplied_count(), 1);
        assert_eq!(resumed.next_index, t2.index + 1);
    }

    #[test]
    fn raft_progress_snapshot_tracks_uncommitted_and_pending_apply_work() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![10]).unwrap();
        let _t2 = r.propose(vec![20]).unwrap();
        r.register_follower_ack(t1.index, 1);

        let progress = r.progress();

        assert_eq!(progress.role, Role::Leader);
        assert_eq!(progress.commit_index, t1.index);
        assert_eq!(progress.applied_index, 0);
        assert_eq!(progress.next_index, 3);
        assert_eq!(progress.committed_but_unapplied_count, 1);
        assert!(progress.has_committed_entries_pending_apply);
        assert_eq!(progress.uncommitted_entry_count, 1);
        assert!(progress.has_uncommitted_entries);
        assert_eq!(progress.apply_gap(), 1);
        assert!(!progress.is_caught_up());
        progress.validate().unwrap();
    }

    #[test]
    fn raft_progress_snapshot_stays_consistent_across_snapshot_boundary_append() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 3,
            snapshot_id: 1,
        });

        r.append_entries_from_leader(
            4,
            5,
            3,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![6],
            }],
            6,
        )
        .unwrap();

        let progress = r.progress();

        assert_eq!(progress.snapshot.last_included_index, 5);
        assert_eq!(progress.commit_index, 6);
        assert_eq!(progress.applied_index, 5);
        assert_eq!(progress.next_index, 7);
        assert_eq!(progress.committed_but_unapplied_count, 1);
        assert!(progress.has_committed_entries_pending_apply);
        assert_eq!(progress.uncommitted_entry_count, 0);
        assert!(!progress.has_uncommitted_entries);
        assert_eq!(progress.apply_gap(), 1);
        assert!(!progress.is_caught_up());
        progress.validate().unwrap();
    }

    #[test]
    fn replication_progress_reports_caught_up_only_when_apply_and_tail_are_clear() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(5);
        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);

        let before_apply = r.progress();
        assert_eq!(before_apply.apply_gap(), 1);
        assert!(!before_apply.is_caught_up());

        r.mark_applied(t1.index);
        let after_apply = r.progress();
        assert_eq!(after_apply.apply_gap(), 0);
        assert!(after_apply.is_caught_up());
    }

    #[test]
    fn recovery_state_helpers_report_commit_apply_boundaries() {
        let state = RecoveryState {
            term: 4,
            snapshot: SnapshotMeta {
                last_included_index: 3,
                last_included_term: 4,
                snapshot_id: 7,
            },
            committed_entries: vec![LogEntry {
                term: 4,
                index: 4,
                payload: vec![1],
            }],
            applied_index: 3,
        };

        assert_eq!(state.commit_index(), 4);
        assert_eq!(state.next_index(), 5);
        assert!(state.has_committed_entries_pending_apply());
        assert_eq!(state.committed_but_unapplied_count(), 1);
        assert_eq!(state.apply_gap(), 1);
        assert!(!state.is_caught_up());
        state.validate().unwrap();

        let progress = state.progress_as_follower().unwrap();
        assert_eq!(progress.role, Role::Follower);
        assert_eq!(progress.term, 4);
        assert_eq!(progress.commit_index, 4);
        assert_eq!(progress.applied_index, 3);
        assert_eq!(progress.next_index, 5);
        assert_eq!(progress.apply_gap(), 1);
        assert!(!progress.is_caught_up());
    }

    #[test]
    fn recovery_state_reports_caught_up_when_apply_reaches_commit_boundary() {
        let state = RecoveryState {
            term: 4,
            snapshot: SnapshotMeta {
                last_included_index: 3,
                last_included_term: 4,
                snapshot_id: 7,
            },
            committed_entries: vec![LogEntry {
                term: 4,
                index: 4,
                payload: vec![1],
            }],
            applied_index: 4,
        };

        assert_eq!(state.apply_gap(), 0);
        assert!(state.is_caught_up());
        assert!(state.progress_as_follower().unwrap().is_caught_up());
    }

    #[test]
    fn snapshot_only_recovery_state_maps_to_caught_up_progress() {
        let state = RecoveryState {
            term: 6,
            snapshot: SnapshotMeta {
                last_included_index: 9,
                last_included_term: 5,
                snapshot_id: 11,
            },
            committed_entries: vec![],
            applied_index: 9,
        };

        assert_eq!(state.commit_index(), 9);
        assert_eq!(state.next_index(), 10);
        assert_eq!(state.committed_but_unapplied_count(), 0);
        assert_eq!(state.apply_gap(), 0);
        assert!(state.is_caught_up());

        let progress = state.progress_as_follower().unwrap();
        assert_eq!(progress.role, Role::Follower);
        assert_eq!(progress.term, 6);
        assert_eq!(progress.commit_index, 9);
        assert_eq!(progress.applied_index, 9);
        assert_eq!(progress.next_index, 10);
        assert_eq!(progress.uncommitted_entry_count, 0);
        assert!(progress.is_caught_up());
    }

    #[test]
    fn recovery_state_validation_rejects_applied_index_past_commit_boundary() {
        let err = RecoveryState {
            term: 4,
            snapshot: SnapshotMeta {
                last_included_index: 3,
                last_included_term: 4,
                snapshot_id: 7,
            },
            committed_entries: vec![LogEntry {
                term: 4,
                index: 4,
                payload: vec![1],
            }],
            applied_index: 5,
        }
        .validate()
        .unwrap_err();

        assert_eq!(
            err,
            RecoveryInvariantError::AppliedExceedsCommit {
                applied_index: 5,
                commit_index: 4,
            }
        );
    }

    #[test]
    fn resumed_follower_progress_matches_recovery_projection() {
        let mut leader = RaftReplicator::new(3);
        leader.become_leader(4);
        let t1 = leader.propose(vec![1]).unwrap();
        let t2 = leader.propose(vec![2]).unwrap();
        leader.register_follower_ack(t1.index, 1);
        leader.register_follower_ack(t2.index, 1);
        leader.mark_applied(t1.index);

        let recovery = leader.recovery_state();
        let projected = recovery.progress_as_follower().unwrap();
        let resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();

        assert_eq!(resumed.progress(), projected);
    }

    #[test]
    fn resumed_follower_from_snapshot_only_recovery_matches_projection() {
        let recovery = RecoveryState {
            term: 6,
            snapshot: SnapshotMeta {
                last_included_index: 9,
                last_included_term: 5,
                snapshot_id: 11,
            },
            committed_entries: vec![],
            applied_index: 9,
        };

        let projected = recovery.progress_as_follower().unwrap();
        let resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();

        assert_eq!(resumed.progress(), projected);
        assert!(resumed.progress().is_caught_up());
    }

    #[test]
    fn replication_progress_validation_rejects_inconsistent_uncommitted_flag() {
        let err = ReplicationProgress {
            role: Role::Follower,
            term: 4,
            commit_index: 5,
            applied_index: 5,
            next_index: 6,
            snapshot: SnapshotMeta {
                last_included_index: 5,
                last_included_term: 4,
                snapshot_id: 1,
            },
            committed_but_unapplied_count: 0,
            has_committed_entries_pending_apply: false,
            uncommitted_entry_count: 1,
            has_uncommitted_entries: false,
        }
        .validate()
        .unwrap_err();

        assert_eq!(
            err,
            ReplicationProgressInvariantError::UncommittedFlagMismatch {
                has_uncommitted: false,
                uncommitted_entry_count: 1,
            }
        );
    }

    #[test]
    fn raft_resume_rejects_non_contiguous_recovery_entries() {
        let err = RaftReplicator::resume_as_follower(
            3,
            RecoveryState {
                term: 5,
                snapshot: SnapshotMeta {
                    last_included_index: 3,
                    last_included_term: 4,
                    snapshot_id: 8,
                },
                committed_entries: vec![LogEntry {
                    term: 5,
                    index: 5,
                    payload: vec![9],
                }],
                applied_index: 3,
            },
        )
        .unwrap_err();

        assert!(matches!(err, EngineError::ApplyFailed(_)));
    }

    #[test]
    fn raft_single_node_leader_commits_immediately() {
        let mut r = RaftReplicator::single_node_leader();
        let tok = r.propose(vec![7]).unwrap();
        assert_eq!(r.commit_index(), tok.index);
    }

    #[test]
    fn raft_follower_acks_are_ignored_when_not_leader() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1]).unwrap();

        r.become_follower(2);
        r.register_follower_ack(t1.index, 1);

        assert_eq!(r.commit_index(), 0);
    }

    #[test]
    fn raft_rejects_ack_for_unknown_index() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let _ = r.propose(vec![1]).unwrap();

        r.register_follower_ack(2, 1);

        assert_eq!(r.commit_index(), 0);
    }

    #[test]
    fn raft_ignores_reserved_self_ack_follower_id() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 0);

        assert_eq!(
            r.commit_index(),
            0,
            "follower id 0 is reserved for the leader self-ack"
        );
    }

    #[test]
    fn raft_duplicate_follower_ack_does_not_count_twice() {
        let mut r = RaftReplicator::new(5);
        r.become_leader(1);

        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.register_follower_ack(t1.index, 2);
        assert_eq!(r.commit_index(), t1.index);

        let t2 = r.propose(vec![2]).unwrap();
        r.register_follower_ack(t2.index, 1);
        r.register_follower_ack(t2.index, 1);

        assert_eq!(
            r.commit_index(),
            t1.index,
            "same follower should not be able to satisfy quorum twice"
        );

        r.register_follower_ack(t2.index, 2);
        assert_eq!(r.commit_index(), t2.index);
    }

    #[test]
    fn raft_prunes_ack_tracking_for_committed_entries() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1]).unwrap();
        let t2 = r.propose(vec![2]).unwrap();

        assert!(r.ack_counts.contains_key(&t1.index));
        assert!(r.ack_counts.contains_key(&t2.index));

        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);
        assert!(!r.ack_counts.contains_key(&t1.index));
        assert!(r.ack_counts.contains_key(&t2.index));

        r.register_follower_ack(t2.index, 1);
        assert_eq!(r.commit_index(), t2.index);
        assert!(r.ack_counts.is_empty());
    }

    #[test]
    fn raft_role_change_discards_uncommitted_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        let _t2_uncommitted = r.propose(vec![2]).unwrap();
        assert_eq!(r.commit_index(), t1.index);

        r.become_follower(2);
        r.become_leader(3);

        let t2_new_epoch = r.propose(vec![3]).unwrap();
        assert_eq!(t2_new_epoch.index, t1.index + 1);

        r.register_follower_ack(t2_new_epoch.index, 1);
        assert_eq!(r.commit_index(), t2_new_epoch.index);
    }

    #[test]
    fn raft_truncate_uncommitted_from_drops_tail_and_resets_next_index() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        let t2 = r.propose(vec![2]).unwrap();
        let t3 = r.propose(vec![3]).unwrap();
        assert!(r.ack_counts.contains_key(&t2.index));
        assert!(r.ack_counts.contains_key(&t3.index));

        r.truncate_uncommitted_from(t2.index);

        assert_eq!(r.commit_index(), t1.index);
        assert!(r.entries.iter().all(|entry| entry.index <= t1.index));
        assert!(r.ack_counts.is_empty());

        let replacement = r.propose(vec![9]).unwrap();
        assert_eq!(replacement.index, t1.index + 1);
    }

    #[test]
    fn raft_truncate_uncommitted_from_ignores_committed_boundary() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        r.truncate_uncommitted_from(t1.index);

        assert_eq!(r.commit_index(), t1.index);
        assert_eq!(r.next_index, t1.index + 1);
    }

    #[test]
    fn local_wait_committed_rejects_uncommitted_token() {
        let mut r = LocalReplicator::leader();
        let token = r.propose(vec![1]).unwrap();
        r.rollback_unapplied_from(token.index);

        let err = r
            .wait_committed(token, std::time::Duration::from_millis(1))
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));
    }

    #[test]
    fn raft_wait_committed_resolves_after_quorum_commit() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(2);

        let token = r.propose(vec![1]).unwrap();
        let pending = r.wait_committed(token, std::time::Duration::from_millis(1));
        assert!(matches!(pending, Err(EngineError::ProposalFailed(_))));

        r.register_follower_ack(token.index, 1);
        let committed = r
            .wait_committed(token, std::time::Duration::from_millis(1))
            .unwrap();
        assert_eq!(committed, token.index);
    }

    #[test]
    fn raft_install_snapshot_prunes_ack_tracking_and_uncompacted_entries() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(4);

        let t1 = r.propose(vec![1]).unwrap();
        let t2 = r.propose(vec![2]).unwrap();
        let t3 = r.propose(vec![3]).unwrap();

        r.register_follower_ack(t1.index, 1);
        r.register_follower_ack(t2.index, 1);
        assert_eq!(r.commit_index(), t2.index);
        assert!(r.ack_counts.contains_key(&t3.index));

        r.install_snapshot(SnapshotMeta {
            last_included_index: t2.index,
            last_included_term: 5,
            snapshot_id: 42,
        });

        assert_eq!(r.current_term(), 5);
        assert_eq!(r.commit_index(), t2.index);
        assert_eq!(r.applied_index(), t2.index);
        assert_eq!(r.snapshot_meta().snapshot_id, 42);
        assert!(!r.ack_counts.contains_key(&t1.index));
        assert!(!r.ack_counts.contains_key(&t2.index));
        assert!(r.ack_counts.contains_key(&t3.index));
        assert!(r.entries.iter().all(|entry| entry.index > t2.index));
    }

    #[test]
    fn raft_progress_snapshot_advances_consistently_after_snapshot_install() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(4);

        let t1 = r.propose(vec![1]).unwrap();
        let t2 = r.propose(vec![2]).unwrap();

        r.register_follower_ack(t1.index, 1);
        r.register_follower_ack(t2.index, 1);
        r.mark_applied(t1.index);

        let before = r.progress();
        assert_eq!(before.commit_index, t2.index);
        assert_eq!(before.applied_index, t1.index);
        assert_eq!(before.apply_gap(), 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: t2.index,
            last_included_term: 5,
            snapshot_id: 42,
        });

        let after = r.progress();
        assert_eq!(after.commit_index, t2.index);
        assert_eq!(after.applied_index, t2.index);
        assert_eq!(after.snapshot.snapshot_id, 42);
        assert_eq!(after.snapshot.last_included_index, t2.index);
        assert_eq!(after.apply_gap(), 0);
        assert!(after.is_caught_up());
        after.validate().unwrap();
    }

    #[test]
    fn raft_install_snapshot_preserves_next_index_from_uncompacted_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![1]).unwrap();
        let _t2 = r.propose(vec![2]).unwrap();

        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index,
            last_included_term: 3,
            snapshot_id: 7,
        });

        let replacement = r.propose(vec![9]).unwrap();
        assert_eq!(replacement.index, 3);
    }

    #[test]
    fn raft_progress_snapshot_preserves_uncommitted_tail_after_snapshot_install() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![1]).unwrap();
        let t2 = r.propose(vec![2]).unwrap();

        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index,
            last_included_term: 3,
            snapshot_id: 7,
        });

        let progress = r.progress();
        assert_eq!(progress.commit_index, t1.index);
        assert_eq!(progress.applied_index, t1.index);
        assert_eq!(progress.next_index, t2.index + 1);
        assert_eq!(progress.uncommitted_entry_count, 1);
        assert!(progress.has_uncommitted_entries);
        assert!(!progress.is_caught_up());
        progress.validate().unwrap();
    }

    #[test]
    fn raft_snapshot_meta_preserves_last_applied_term_across_term_bumps() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(2);

        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.mark_applied(t1.index);

        r.become_follower(7);
        let meta = r.snapshot_meta();

        assert_eq!(r.current_term(), 7);
        assert_eq!(meta.last_included_index, t1.index);
        assert_eq!(meta.last_included_term, 2);
    }

    #[test]
    fn raft_follower_append_entries_truncates_conflicting_uncommitted_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        let _t2_old = r.propose(vec![2]).unwrap();
        r.become_follower(2);

        r.append_entries_from_leader(
            2,
            t1.index,
            1,
            vec![LogEntry {
                term: 2,
                index: t1.index + 1,
                payload: vec![9],
            }],
            t1.index + 1,
        )
        .unwrap();

        assert_eq!(r.commit_index(), t1.index + 1);
        assert_eq!(r.next_index, t1.index + 2);
        assert!(r
            .entries
            .iter()
            .any(|entry| entry.index == t1.index + 1 && entry.term == 2));
    }

    #[test]
    fn raft_follower_append_entries_empty_heartbeat_can_advance_commit_index() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        r.become_follower(2);
        let t2_index = t1.index + 1;
        r.append_entries_from_leader(
            2,
            t1.index,
            1,
            vec![LogEntry {
                term: 2,
                index: t2_index,
                payload: vec![2],
            }],
            t1.index,
        )
        .unwrap();
        assert_eq!(r.commit_index(), t1.index);

        r.append_entries_from_leader(2, t2_index, 2, vec![], t2_index)
            .unwrap();

        assert_eq!(r.commit_index(), t2_index);
        assert_eq!(r.next_index, t2_index + 1);
    }

    #[test]
    fn raft_progress_snapshot_tracks_heartbeat_commit_advancement() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let t2_index = t1.index + 1;
        r.append_entries_from_leader(
            2,
            t1.index,
            1,
            vec![LogEntry {
                term: 2,
                index: t2_index,
                payload: vec![2],
            }],
            t1.index,
        )
        .unwrap();

        let before_heartbeat = r.progress();
        assert_eq!(before_heartbeat.commit_index, t1.index);
        assert_eq!(before_heartbeat.applied_index, 0);
        assert_eq!(before_heartbeat.next_index, t2_index + 1);
        assert_eq!(before_heartbeat.uncommitted_entry_count, 1);
        assert!(before_heartbeat.has_uncommitted_entries);
        assert_eq!(before_heartbeat.apply_gap(), t1.index as usize);

        r.append_entries_from_leader(2, t2_index, 2, vec![], t2_index)
            .unwrap();

        let after_heartbeat = r.progress();
        assert_eq!(after_heartbeat.commit_index, t2_index);
        assert_eq!(after_heartbeat.applied_index, 0);
        assert_eq!(after_heartbeat.next_index, t2_index + 1);
        assert_eq!(after_heartbeat.uncommitted_entry_count, 0);
        assert!(!after_heartbeat.has_uncommitted_entries);
        assert_eq!(after_heartbeat.apply_gap(), t2_index as usize);
        assert!(!after_heartbeat.is_caught_up());
        after_heartbeat.validate().unwrap();
    }

    #[test]
    fn raft_follower_append_entries_bumps_local_term_from_leader_term() {
        let mut r = RaftReplicator::new(3);
        r.become_candidate(3);

        r.append_entries_from_leader(4, 0, 0, vec![], 0).unwrap();

        assert_eq!(r.current_term(), 4);
        assert_eq!(r.role(), Role::Follower);
    }

    #[test]
    fn raft_follower_append_entries_rejects_stale_leader_term() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        let err = r
            .append_entries_from_leader(4, 0, 0, vec![], 0)
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.current_term(), 5);
    }

    #[test]
    fn raft_follower_append_entries_rejects_stale_term_without_state_mutation() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(5);
        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(5);

        let role_before = r.role();
        let term_before = r.current_term();
        let commit_before = r.commit_index();
        let next_before = r.next_index;
        let entries_before = r.entries.clone();

        let err = r
            .append_entries_from_leader(
                4,
                t1.index,
                5,
                vec![LogEntry {
                    term: 4,
                    index: t1.index + 1,
                    payload: vec![9],
                }],
                t1.index + 1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.role(), role_before);
        assert_eq!(r.current_term(), term_before);
        assert_eq!(r.commit_index(), commit_before);
        assert_eq!(r.next_index, next_before);
        assert_eq!(r.entries, entries_before);
    }

    #[test]
    fn raft_progress_snapshot_is_stable_across_stale_append_rejection() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(5);
        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(5);

        let before = r.progress();

        let err = r
            .append_entries_from_leader(
                4,
                t1.index,
                5,
                vec![LogEntry {
                    term: 4,
                    index: t1.index + 1,
                    payload: vec![9],
                }],
                t1.index + 1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.progress(), before);
    }

    #[test]
    fn raft_follower_append_entries_rejects_entries_with_term_ahead_of_leader() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        let err = r
            .append_entries_from_leader(
                5,
                0,
                0,
                vec![LogEntry {
                    term: 6,
                    index: 1,
                    payload: vec![1],
                }],
                1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.current_term(), 5);
        assert_eq!(r.commit_index(), 0);
        assert_eq!(r.next_index, 1);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_leader_rejects_follower_append_path_without_state_mutation() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(6);

        let committed = r.propose(vec![1]).unwrap();
        r.register_follower_ack(committed.index, 1);

        let role_before = r.role();
        let term_before = r.current_term();
        let commit_before = r.commit_index();
        let next_before = r.next_index;

        let err = r
            .append_entries_from_leader(
                7,
                committed.index,
                term_before,
                vec![LogEntry {
                    term: 7,
                    index: committed.index + 1,
                    payload: vec![9],
                }],
                committed.index + 1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.role(), role_before);
        assert_eq!(r.current_term(), term_before);
        assert_eq!(r.commit_index(), commit_before);
        assert_eq!(r.next_index, next_before);
        assert!(r.entries.iter().all(|entry| entry.term != 7));
    }

    #[test]
    fn raft_follower_append_entries_rejects_prev_term_mismatch() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let err = r
            .append_entries_from_leader(
                2,
                t1.index,
                999,
                vec![LogEntry {
                    term: 2,
                    index: t1.index + 1,
                    payload: vec![2],
                }],
                t1.index + 1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), t1.index);
    }

    #[test]
    fn raft_follower_append_entries_rejects_non_contiguous_batches() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let err = r
            .append_entries_from_leader(
                2,
                t1.index,
                1,
                vec![
                    LogEntry {
                        term: 2,
                        index: t1.index + 1,
                        payload: vec![2],
                    },
                    LogEntry {
                        term: 2,
                        index: t1.index + 3,
                        payload: vec![3],
                    },
                ],
                t1.index + 3,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), t1.index);
        assert_eq!(r.next_index, t1.index + 1);
        assert!(r.entries.iter().all(|entry| entry.index <= t1.index));
    }

    #[test]
    fn raft_progress_snapshot_is_stable_across_non_contiguous_append_rejection() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let before = r.progress();

        let err = r
            .append_entries_from_leader(
                2,
                t1.index,
                1,
                vec![
                    LogEntry {
                        term: 2,
                        index: t1.index + 1,
                        payload: vec![2],
                    },
                    LogEntry {
                        term: 2,
                        index: t1.index + 3,
                        payload: vec![3],
                    },
                ],
                t1.index + 3,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.progress(), before);
    }

    #[test]
    fn raft_follower_append_entries_rejects_first_entry_that_skips_prev_index() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let err = r
            .append_entries_from_leader(
                2,
                t1.index,
                1,
                vec![LogEntry {
                    term: 2,
                    index: t1.index + 2,
                    payload: vec![2],
                }],
                t1.index + 2,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), t1.index);
        assert_eq!(r.next_index, t1.index + 1);
    }

    #[test]
    fn raft_follower_append_entries_does_not_overwrite_committed_entries() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let err = r
            .append_entries_from_leader(
                2,
                0,
                0,
                vec![LogEntry {
                    term: 2,
                    index: t1.index,
                    payload: vec![9],
                }],
                t1.index,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        let committed = r
            .entries
            .iter()
            .find(|entry| entry.index == t1.index)
            .unwrap();
        assert_eq!(committed.term, 1);
        assert_eq!(committed.payload, vec![1]);
    }

    #[test]
    fn raft_follower_append_entries_rejects_payload_mismatch_for_same_index_and_term() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(3);

        r.append_entries_from_leader(
            3,
            0,
            0,
            vec![LogEntry {
                term: 3,
                index: 1,
                payload: vec![1],
            }],
            1,
        )
        .unwrap();

        let commit_before = r.commit_index();
        let next_before = r.next_index;

        let err = r
            .append_entries_from_leader(
                3,
                0,
                0,
                vec![LogEntry {
                    term: 3,
                    index: 1,
                    payload: vec![9],
                }],
                1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), commit_before);
        assert_eq!(r.next_index, next_before);
        let preserved = r.entries.iter().find(|entry| entry.index == 1).unwrap();
        assert_eq!(preserved.term, 3);
        assert_eq!(preserved.payload, vec![1]);
    }

    #[test]
    fn raft_follower_append_entries_caps_commit_index_to_local_log_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(2);

        r.append_entries_from_leader(
            2,
            0,
            0,
            vec![LogEntry {
                term: 2,
                index: 1,
                payload: vec![1],
            }],
            99,
        )
        .unwrap();

        assert_eq!(r.commit_index(), 1);
        assert_eq!(r.next_index, 2);
    }

    #[test]
    fn raft_follower_append_entries_missing_prev_index_is_rejected_without_state_mutation() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        let role_before = r.role();
        let term_before = r.current_term();
        let commit_before = r.commit_index();
        let next_before = r.next_index;

        let err = r
            .append_entries_from_leader(
                5,
                10,
                5,
                vec![LogEntry {
                    term: 5,
                    index: 11,
                    payload: vec![7],
                }],
                11,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.role(), role_before);
        assert_eq!(r.current_term(), term_before);
        assert_eq!(r.commit_index(), commit_before);
        assert_eq!(r.next_index, next_before);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_candidate_append_rejection_still_updates_term_and_role_for_newer_leader() {
        let mut r = RaftReplicator::new(3);
        r.become_candidate(5);

        let err = r
            .append_entries_from_leader(
                6,
                10,
                5,
                vec![LogEntry {
                    term: 6,
                    index: 11,
                    payload: vec![7],
                }],
                11,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.role(), Role::Follower);
        assert_eq!(r.current_term(), 6);
        assert_eq!(r.commit_index(), 0);
        assert_eq!(r.next_index, 1);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_follower_append_entries_accepts_prev_index_at_snapshot_boundary_after_apply_advances() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(3);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 2,
            snapshot_id: 1,
        });

        r.append_entries_from_leader(
            3,
            5,
            2,
            vec![LogEntry {
                term: 3,
                index: 6,
                payload: vec![6],
            }],
            6,
        )
        .unwrap();
        r.mark_applied(6);

        r.append_entries_from_leader(
            3,
            5,
            2,
            vec![LogEntry {
                term: 3,
                index: 6,
                payload: vec![6],
            }],
            6,
        )
        .unwrap();

        assert_eq!(r.commit_index(), 6);
        assert_eq!(r.next_index, 7);
    }

    #[test]
    fn raft_progress_snapshot_remains_consistent_across_append_at_snapshot_boundary() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(3);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 2,
            snapshot_id: 1,
        });

        let before = r.progress();
        assert_eq!(before.commit_index, 5);
        assert_eq!(before.applied_index, 5);
        assert_eq!(before.next_index, 6);
        assert!(before.is_caught_up());

        r.append_entries_from_leader(
            3,
            5,
            2,
            vec![LogEntry {
                term: 3,
                index: 6,
                payload: vec![6],
            }],
            6,
        )
        .unwrap();

        let after_append = r.progress();
        assert_eq!(after_append.commit_index, 6);
        assert_eq!(after_append.applied_index, 5);
        assert_eq!(after_append.next_index, 7);
        assert_eq!(after_append.apply_gap(), 1);
        assert!(!after_append.is_caught_up());

        r.mark_applied(6);
        let after_apply = r.progress();
        assert_eq!(after_apply.commit_index, 6);
        assert_eq!(after_apply.applied_index, 6);
        assert_eq!(after_apply.next_index, 7);
        assert_eq!(after_apply.apply_gap(), 0);
        assert!(after_apply.is_caught_up());

        r.append_entries_from_leader(
            3,
            5,
            2,
            vec![LogEntry {
                term: 3,
                index: 6,
                payload: vec![6],
            }],
            6,
        )
        .unwrap();

        let after_repeat = r.progress();
        assert_eq!(after_repeat, after_apply);
    }

    #[test]
    fn raft_follower_append_entries_rejects_snapshot_boundary_term_mismatch() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 3,
            snapshot_id: 1,
        });

        let err = r
            .append_entries_from_leader(
                4,
                5,
                2,
                vec![LogEntry {
                    term: 4,
                    index: 6,
                    payload: vec![6],
                }],
                6,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), 5);
        assert_eq!(r.applied_index(), 5);
        assert_eq!(r.next_index, 6);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_progress_snapshot_is_stable_across_snapshot_boundary_term_mismatch() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 3,
            snapshot_id: 1,
        });

        let before = r.progress();

        let err = r
            .append_entries_from_leader(
                4,
                5,
                2,
                vec![LogEntry {
                    term: 4,
                    index: 6,
                    payload: vec![6],
                }],
                6,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.progress(), before);
    }

    #[test]
    fn raft_follower_append_entries_rejects_prev_index_behind_snapshot_boundary() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 3,
            snapshot_id: 1,
        });

        let err = r
            .append_entries_from_leader(
                4,
                0,
                0,
                vec![LogEntry {
                    term: 4,
                    index: 1,
                    payload: vec![1],
                }],
                1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), 5);
        assert_eq!(r.applied_index(), 5);
        assert_eq!(r.next_index, 6);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_progress_snapshot_is_stable_across_prev_index_behind_snapshot_boundary() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 3,
            snapshot_id: 1,
        });

        let before = r.progress();

        let err = r
            .append_entries_from_leader(
                4,
                0,
                0,
                vec![LogEntry {
                    term: 4,
                    index: 1,
                    payload: vec![1],
                }],
                1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.progress(), before);
    }

    #[test]
    fn raft_install_snapshot_does_not_rewrite_compacted_term_for_same_index() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 2,
            snapshot_id: 3,
        });

        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![9],
            }],
            6,
        )
        .unwrap();

        assert_eq!(r.commit_index(), 6);
        assert_eq!(r.next_index, 7);
    }
}
