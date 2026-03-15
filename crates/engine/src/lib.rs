use gpu_db_replication::{LocalReplicator, LogReplicator, ReplicatedStateMachine};
use gpu_db_types::{CommitToken, EngineError, Index, LogEntry};
use gpu_db_wal::{WalBuffer, WalRecord};

#[derive(Debug, Default)]
pub struct KvStateMachine {
    pub applied: Vec<Vec<u8>>,
}

impl ReplicatedStateMachine for KvStateMachine {
    fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError> {
        self.applied.push(entry.payload.clone());
        Ok(())
    }
}

pub struct Engine {
    repl: LocalReplicator,
    wal: WalBuffer,
    sm: KvStateMachine,
    visible_up_to: Index,
}

impl Engine {
    pub fn new_local() -> Self {
        Self {
            repl: LocalReplicator::leader(),
            wal: WalBuffer::default(),
            sm: KvStateMachine::default(),
            visible_up_to: 0,
        }
    }

    pub fn commit_mutation(&mut self, txn_id: u64, payload: Vec<u8>) -> Result<CommitToken, EngineError> {
        // 1) append WAL intent
        self.wal.append(WalRecord {
            txn_id,
            payload: payload.clone(),
        });

        // 2) propose replicated log entry
        let token = self.repl.propose(payload)?;

        // 3) durable flush gate before visibility
        self.wal.flush_all();

        // 4) apply committed entries up to token
        let to_apply: Vec<LogEntry> = self
            .repl
            .drain_committed_from(self.repl.applied_index())
            .cloned()
            .collect();

        for e in &to_apply {
            self.sm.apply(e)?;
            self.repl.mark_applied(e.index);
        }

        // 5) advance visibility only after WAL flush + apply
        self.visible_up_to = self.visible_up_to.max(token.index);

        Ok(token)
    }

    pub fn visible_up_to(&self) -> Index {
        self.visible_up_to
    }

    pub fn applied_len(&self) -> usize {
        self.sm.applied.len()
    }

    pub fn wal_flushed_count(&self) -> usize {
        self.wal.flushed_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wal_before_visibility_holds() {
        let mut e = Engine::new_local();
        let t = e.commit_mutation(1, b"set a=1".to_vec()).unwrap();
        assert!(e.wal_flushed_count() >= 1);
        assert!(e.visible_up_to() >= t.index);
        assert!(e.applied_len() >= 1);
    }

    #[test]
    fn commit_indices_monotonic() {
        let mut e = Engine::new_local();
        let a = e.commit_mutation(1, b"a".to_vec()).unwrap();
        let b = e.commit_mutation(2, b"b".to_vec()).unwrap();
        assert!(b.index > a.index);
        assert!(e.visible_up_to() >= b.index);
    }
}
