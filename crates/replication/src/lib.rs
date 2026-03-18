use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term};

pub trait LogReplicator {
    fn propose(&mut self, payload: Vec<u8>) -> Result<CommitToken, EngineError>;
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

    pub fn drain_committed_from(&self, start_exclusive: Index) -> impl Iterator<Item = &LogEntry> {
        self.entries
            .iter()
            .filter(move |e| e.index > start_exclusive && e.index <= self.commit_index)
    }

    pub fn mark_applied(&mut self, idx: Index) {
        self.applied_index = self.applied_index.max(idx);
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
        self.applied_index = self.applied_index.max(meta.last_included_index);
        self.next_index = self.commit_index + 1;
        self.snapshot_id = self.snapshot_id.max(meta.snapshot_id);
        self.entries.retain(|e| e.index > meta.last_included_index);
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
            last_included_term: self.term,
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
}
