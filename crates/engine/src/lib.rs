use std::collections::BTreeMap;

use gpu_db_metrics::{FallbackReason, RuntimeMetrics};
use gpu_db_protocol::{parse_command, Command, ParseError};
use gpu_db_replication::{LocalReplicator, LogReplicator, ReplicatedStateMachine};
use gpu_db_types::{CommitToken, EngineError, Index, LogEntry};
use gpu_db_wal::{WalBuffer, WalRecord};

#[derive(Debug, Default)]
pub struct KvStateMachine {
    pub applied: Vec<Vec<u8>>,
    pub kv: BTreeMap<String, String>,
}

impl ReplicatedStateMachine for KvStateMachine {
    fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError> {
        self.applied.push(entry.payload.clone());
        if let Ok(s) = std::str::from_utf8(&entry.payload) {
            if let Ok(Command::SetKv { key, value }) = parse_command(s) {
                self.kv.insert(key, value);
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecuteError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Engine(#[from] EngineError),
}

pub struct Engine {
    repl: LocalReplicator,
    wal: WalBuffer,
    sm: KvStateMachine,
    visible_up_to: Index,
    metrics: RuntimeMetrics,
}

impl Engine {
    pub fn new_local() -> Self {
        Self {
            repl: LocalReplicator::leader(),
            wal: WalBuffer::default(),
            sm: KvStateMachine::default(),
            visible_up_to: 0,
            metrics: RuntimeMetrics::default(),
        }
    }

    pub fn commit_mutation(&mut self, txn_id: u64, payload: Vec<u8>) -> Result<CommitToken, EngineError> {
        self.wal.append(WalRecord {
            txn_id,
            payload: payload.clone(),
        });

        let token = self.repl.propose(payload)?;

        self.wal.flush_all();

        let to_apply: Vec<LogEntry> = self
            .repl
            .drain_committed_from(self.repl.applied_index())
            .cloned()
            .collect();

        for e in &to_apply {
            self.sm.apply(e)?;
            self.repl.mark_applied(e.index);
        }

        self.visible_up_to = self.visible_up_to.max(token.index);
        self.metrics.inc_commit();

        Ok(token)
    }

    pub fn execute_text(&mut self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;

        match cmd {
            Command::SetKv { .. } => {
                self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
            }
            Command::Begin | Command::Commit | Command::Rollback => {
                // pre-NVIDIA bootstrap: control commands parsed and accepted.
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
        }

        Ok(())
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

    pub fn get(&self, key: &str) -> Option<&str> {
        self.sm.kv.get(key).map(|s| s.as_str())
    }

    pub fn metrics(&self) -> &RuntimeMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wal_before_visibility_holds() {
        let mut e = Engine::new_local();
        let t = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        assert!(e.wal_flushed_count() >= 1);
        assert!(e.visible_up_to() >= t.index);
        assert!(e.applied_len() >= 1);
    }

    #[test]
    fn commit_indices_monotonic() {
        let mut e = Engine::new_local();
        let a = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        let b = e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();
        assert!(b.index > a.index);
        assert!(e.visible_up_to() >= b.index);
    }

    #[test]
    fn execute_set_updates_state_machine() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        assert_eq!(e.get("balance"), Some("100"));
        assert_eq!(e.metrics().commits_total, 1);
    }
}
