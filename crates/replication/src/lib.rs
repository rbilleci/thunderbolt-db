use std::collections::{BTreeMap, BTreeSet};

use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term};

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
        self.next_index = self.commit_index + 1;
        self.snapshot_id = self.snapshot_id.max(meta.snapshot_id);
        self.entries.retain(|e| e.index > meta.last_included_index);
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

    pub fn append_entries_from_leader(
        &mut self,
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
        if meta.last_included_index > self.applied_index {
            self.applied_index = meta.last_included_index;
            self.applied_term = meta.last_included_term;
        }
        self.next_index = self.commit_index + 1;
        self.snapshot_id = self.snapshot_id.max(meta.snapshot_id);
        self.entries.retain(|e| e.index > meta.last_included_index);
        self.ack_counts
            .retain(|idx, _| *idx > meta.last_included_index);
    }

    fn term_at(&self, index: Index) -> Option<Term> {
        if index == 0 {
            return Some(0);
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
    fn mark_applied_does_not_exceed_commit_index() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1]).unwrap();

        r.mark_applied(t1.index + 10);

        assert_eq!(r.applied_index(), t1.index);
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
    fn raft_follower_append_entries_rejects_prev_term_mismatch() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let err = r
            .append_entries_from_leader(
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
    fn raft_follower_append_entries_rejects_first_entry_that_skips_prev_index() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1]).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let err = r
            .append_entries_from_leader(
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
}
