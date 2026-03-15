use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, Term};

pub trait LogReplicator {
    fn propose(&mut self, payload: Vec<u8>) -> Result<CommitToken, EngineError>;
    fn role(&self) -> Role;
    fn current_term(&self) -> Term;
    fn commit_index(&self) -> Index;
    fn applied_index(&self) -> Index;
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
}
