use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use gpu_db_batching::{DualTriggerBatcher, FlushReason};
use gpu_db_metrics::{BatchFlushReason, FallbackReason, RuntimeMetrics};
use gpu_db_protocol::{parse_command, Command, ParseError};
use gpu_db_replication::{LocalReplicator, LogReplicator, ReplicatedStateMachine};
use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, Term};
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
            if let Ok(cmd) = parse_command(s) {
                match cmd {
                    Command::SetKv { key, value } => {
                        self.kv.insert(key, value);
                    }
                    Command::DeleteKv { key } => {
                        self.kv.remove(&key);
                    }
                    Command::Begin | Command::Commit | Command::Rollback | Command::Flush => {}
                }
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

#[derive(Debug, Clone)]
struct PendingMutation {
    txn_id: u64,
    payload: Vec<u8>,
}

pub struct Engine {
    repl: LocalReplicator,
    wal: WalBuffer,
    sm: KvStateMachine,
    visible_up_to: Index,
    metrics: RuntimeMetrics,
    batcher: DualTriggerBatcher<PendingMutation>,
}

impl Engine {
    pub fn new_local() -> Self {
        Self {
            repl: LocalReplicator::leader(),
            wal: WalBuffer::default(),
            sm: KvStateMachine::default(),
            visible_up_to: 0,
            metrics: RuntimeMetrics::default(),
            batcher: DualTriggerBatcher::new(64, Duration::from_millis(1)),
        }
    }

    pub fn with_batching(max_items: usize, max_wait: Duration) -> Self {
        let mut s = Self::new_local();
        s.batcher = DualTriggerBatcher::new(max_items, max_wait);
        s
    }

    pub fn simulate_next_wal_flush_failure(&mut self) {
        self.wal.fail_next_flush();
    }

    pub fn become_follower(&mut self, term: Term) {
        self.repl.become_follower(term);
    }

    pub fn become_leader(&mut self, term: Term) {
        self.repl.become_leader(term);
    }

    pub fn commit_mutation(
        &mut self,
        txn_id: u64,
        payload: Vec<u8>,
    ) -> Result<CommitToken, EngineError> {
        if self.repl.role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        self.wal.append(WalRecord {
            txn_id,
            payload: payload.clone(),
        });

        let token = self.repl.propose(payload)?;
        if let Err(err) = self.wal.flush_all() {
            self.repl.rollback_unapplied_from(token.index);
            return Err(err);
        }

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

    pub fn enqueue_set_text(
        &mut self,
        txn_id: u64,
        text: &str,
        now: Instant,
    ) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;
        match cmd {
            Command::SetKv { .. } | Command::DeleteKv { .. } => {
                let maybe_batch = self.batcher.enqueue(
                    PendingMutation {
                        txn_id,
                        payload: text.as_bytes().to_vec(),
                    },
                    now,
                );
                if let Some(batch) = maybe_batch {
                    self.apply_batch(batch.reason, batch.items.into_iter().map(|i| i.item))?;
                }
            }
            Command::Flush => {
                self.flush_admin()?;
            }
            Command::Begin | Command::Commit | Command::Rollback => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
        }
        Ok(())
    }

    pub fn tick_batching(&mut self, now: Instant) -> Result<(), EngineError> {
        if let Some(batch) = self.batcher.maybe_flush_due_to_time(now) {
            self.apply_batch(batch.reason, batch.items.into_iter().map(|i| i.item))?;
        }
        Ok(())
    }

    pub fn flush_admin(&mut self) -> Result<(), EngineError> {
        if let Some(batch) = self.batcher.flush_admin() {
            self.apply_batch(batch.reason, batch.items.into_iter().map(|i| i.item))?;
        }
        Ok(())
    }

    fn apply_batch<I>(&mut self, reason: FlushReason, items: I) -> Result<(), EngineError>
    where
        I: Iterator<Item = PendingMutation>,
    {
        let metric_reason = match reason {
            FlushReason::Count => BatchFlushReason::Count,
            FlushReason::Time => BatchFlushReason::Time,
            FlushReason::Admin => BatchFlushReason::Admin,
        };
        for p in items {
            self.commit_mutation(p.txn_id, p.payload)?;
        }
        self.metrics.inc_batch_flush(metric_reason);
        Ok(())
    }

    pub fn execute_text(&mut self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;

        match cmd {
            Command::SetKv { .. } | Command::DeleteKv { .. } => {
                self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
            }
            Command::Flush => {
                self.flush_admin()?;
            }
            Command::Begin | Command::Commit | Command::Rollback => {
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

    pub fn pending_batch_len(&self) -> usize {
        self.batcher.len()
    }

    pub fn has_pending_batch(&self) -> bool {
        !self.batcher.is_empty()
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

    #[test]
    fn execute_del_removes_existing_key() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        e.execute_text(2, "DEL balance").unwrap();

        assert_eq!(e.get("balance"), None);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn batching_flushes_on_count_and_updates_metric() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.enqueue_set_text(2, "SET b=2", t0).unwrap();
        assert_eq!(e.get("a"), Some("1"));
        assert_eq!(e.get("b"), Some("2"));
        assert_eq!(e.metrics().batch_flush_count, 1);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 1);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn batching_flushes_on_time() {
        let mut e = Engine::with_batching(10, Duration::from_millis(2));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=7", t0).unwrap();
        assert!(e.has_pending_batch());
        assert_eq!(e.pending_batch_len(), 1);
        e.tick_batching(t0 + Duration::from_millis(3)).unwrap();
        assert_eq!(e.get("a"), Some("7"));
        assert!(!e.has_pending_batch());
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flush_count, 1);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Time), 1);
    }

    #[test]
    fn admin_flush_tracks_reason() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=9", t0).unwrap();
        assert_eq!(e.pending_batch_len(), 1);
        e.flush_admin().unwrap();

        assert_eq!(e.get("a"), Some("9"));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
    }

    #[test]
    fn batching_can_apply_set_then_del_in_order() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.enqueue_set_text(2, "DEL a", t0).unwrap();

        assert_eq!(e.get("a"), None);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 1);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn flush_command_drains_pending_batch() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=5", t0).unwrap();
        e.execute_text(2, "FLUSH").unwrap();

        assert_eq!(e.get("a"), Some("5"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
    }

    #[test]
    fn wal_flush_failure_prevents_visibility_advance() {
        let mut e = Engine::new_local();
        e.simulate_next_wal_flush_failure();
        let res = e.commit_mutation(1, b"SET a=1".to_vec());
        assert!(matches!(res, Err(EngineError::Durability(_))));
        assert_eq!(e.visible_up_to(), 0);
    }

    #[test]
    fn wal_flush_failure_does_not_leak_into_later_successful_commit() {
        let mut e = Engine::new_local();
        e.simulate_next_wal_flush_failure();
        let _ = e.commit_mutation(1, b"SET a=1".to_vec());

        e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();

        assert_eq!(e.get("a"), None);
        assert_eq!(e.get("b"), Some("2"));
        assert_eq!(e.applied_len(), 1);
    }

    #[test]
    fn follower_rejects_commit_without_visibility_or_wal_flush() {
        let mut e = Engine::new_local();
        e.become_follower(2);

        let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(e.visible_up_to(), 0);
        assert_eq!(e.wal_flushed_count(), 0);
        assert_eq!(e.applied_len(), 0);
    }

    #[test]
    fn failed_batch_flush_does_not_increment_flush_metrics() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_follower(2);

        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        let err = e.enqueue_set_text(2, "SET b=2", t0).unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn execute_text_transaction_controls_count_as_not_gpu_eligible_fallbacks() {
        let mut e = Engine::new_local();

        e.execute_text(1, "BEGIN").unwrap();
        e.execute_text(1, "COMMIT").unwrap();
        e.execute_text(1, "ROLLBACK").unwrap();

        assert_eq!(e.metrics().fallback_total, 3);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 3);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn enqueue_transaction_controls_count_as_not_gpu_eligible_fallbacks() {
        let mut e = Engine::with_batching(2, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "BEGIN", t0).unwrap();
        e.enqueue_set_text(1, "COMMIT", t0).unwrap();
        e.enqueue_set_text(1, "ROLLBACK", t0).unwrap();

        assert_eq!(e.metrics().fallback_total, 3);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 3);
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 0);
    }
}
