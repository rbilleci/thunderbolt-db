use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use gpu_db_batching::{BatchItem, DualTriggerBatcher, FlushReason};
use gpu_db_metrics::{BatchFlushReason, FallbackReason, RuntimeMetrics};
use gpu_db_protocol::{parse_command, Command, ParseError};
use gpu_db_replication::{LocalReplicator, LogReplicator, ReplicatedStateMachine};
use gpu_db_txn::{TxnError, TxnManager};
use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term};
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
                    Command::Begin
                    | Command::Commit
                    | Command::Rollback
                    | Command::Flush
                    | Command::GetKv { .. } => {}
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
    #[error(transparent)]
    Txn(#[from] TxnError),
}

#[derive(Debug, Clone)]
struct PendingMutation {
    txn_id: u64,
    payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplicationWatermarks {
    pub role: Role,
    pub term: Term,
    pub commit_index: Index,
    pub applied_index: Index,
    pub visible_index: Index,
    pub snapshot_id: u64,
    pub wal_flushed_count: usize,
    pub wal_buffered_count: usize,
    pub wal_unflushed_count: usize,
}

pub struct Engine {
    repl: LocalReplicator,
    wal: WalBuffer,
    sm: KvStateMachine,
    txn_manager: TxnManager,
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
            txn_manager: TxnManager::default(),
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

    pub fn become_candidate(&mut self, term: Term) {
        self.repl.become_candidate(term);
    }

    pub fn commit_mutation(
        &mut self,
        txn_id: u64,
        payload: Vec<u8>,
    ) -> Result<CommitToken, EngineError> {
        if self.repl.role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        let wal_len_before = self.wal.len();
        self.wal.append(WalRecord {
            txn_id,
            payload: payload.clone(),
        });

        let token = match self.repl.propose(payload) {
            Ok(token) => token,
            Err(err) => {
                self.wal.truncate(wal_len_before);
                return Err(err);
            }
        };
        if let Err(err) = self.wal.flush_all() {
            self.repl.rollback_unapplied_from(token.index);
            self.wal.truncate(wal_len_before);
            return Err(err);
        }

        self.repl.wait_committed(token, Duration::from_millis(0))?;

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
                if self.repl.role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }

                let maybe_batch = self.batcher.enqueue(
                    PendingMutation {
                        txn_id,
                        payload: text.as_bytes().to_vec(),
                    },
                    now,
                );
                if let Some(batch) = maybe_batch {
                    self.apply_batch(batch.reason, batch.items.into_iter(), now)?;
                }
            }
            Command::Flush => {
                self.flush_admin()?;
            }
            Command::Begin => {
                self.txn_manager.begin_with_id(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit => {
                self.txn_manager.commit(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback => {
                self.txn_manager.rollback(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { .. } => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
        }
        Ok(())
    }

    pub fn tick_batching(&mut self, now: Instant) -> Result<(), EngineError> {
        if self.repl.role() != Role::Leader {
            if self.has_pending_batch() {
                return Err(EngineError::NotLeader);
            }
            return Ok(());
        }

        if let Some(batch) = self.batcher.maybe_flush_due_to_time(now) {
            self.apply_batch(batch.reason, batch.items.into_iter(), now)?;
        }
        Ok(())
    }

    pub fn flush_admin(&mut self) -> Result<(), EngineError> {
        if self.repl.role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        if let Some(batch) = self.batcher.flush_admin() {
            self.apply_batch(batch.reason, batch.items.into_iter(), Instant::now())?;
        }
        Ok(())
    }

    fn apply_batch<I>(
        &mut self,
        reason: FlushReason,
        items: I,
        flushed_at: Instant,
    ) -> Result<(), EngineError>
    where
        I: Iterator<Item = BatchItem<PendingMutation>>,
    {
        let metric_reason = match reason {
            FlushReason::Count => BatchFlushReason::Count,
            FlushReason::Time => BatchFlushReason::Time,
            FlushReason::Admin => BatchFlushReason::Admin,
        };

        let mut remaining = items.peekable();
        while let Some(p) = remaining.next() {
            let wait = flushed_at
                .saturating_duration_since(p.enqueued_at)
                .as_millis() as u64;
            let txn_id = p.item.txn_id;
            let payload = p.item.payload.clone();

            // In no-GPU bootstrap mode, batched mutations represent the simulated
            // GPU-eligible write path. Track transfer and kernel timing envelopes
            // so telemetry contracts are stable before CUDA is wired in.
            self.metrics.observe_h2d_bytes(payload.len() as u64);
            let simulated_kernel_ms = ((payload.len() as u64) / 1024).max(1);
            self.metrics.observe_kernel_exec_ms(simulated_kernel_ms);

            if let Err(err) = self.commit_mutation(txn_id, payload) {
                let tail: Vec<_> = std::iter::once(p).chain(remaining).collect();
                self.batcher.requeue_front(tail);
                return Err(err);
            }

            self.metrics.observe_batch_wait_ms(wait);
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
            Command::Begin => {
                self.txn_manager.begin_with_id(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit => {
                self.txn_manager.commit(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback => {
                self.txn_manager.rollback(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { .. } => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
        }

        Ok(())
    }

    pub fn execute_read_text(&mut self, text: &str) -> Result<Option<&str>, ExecuteError> {
        let cmd = parse_command(text)?;

        match cmd {
            Command::GetKv { key } => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
                Ok(self.get(&key))
            }
            _ => Ok(None),
        }
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

    pub fn wal_buffered_count(&self) -> usize {
        self.wal.len()
    }

    pub fn wal_unflushed_count(&self) -> usize {
        self.wal.unflushed_count()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.sm.kv.get(key).map(|s| s.as_str())
    }

    pub fn active_txn_count(&self) -> usize {
        self.txn_manager.active_count()
    }

    pub fn replication_watermarks(&self) -> ReplicationWatermarks {
        ReplicationWatermarks {
            role: self.repl.role(),
            term: self.repl.current_term(),
            commit_index: self.repl.commit_index(),
            applied_index: self.repl.applied_index(),
            visible_index: self.visible_up_to,
            snapshot_id: self.repl.snapshot_meta().snapshot_id,
            wal_flushed_count: self.wal.flushed_count(),
            wal_buffered_count: self.wal.len(),
            wal_unflushed_count: self.wal.unflushed_count(),
        }
    }

    pub fn export_snapshot_meta(&mut self) -> SnapshotMeta {
        self.repl.export_snapshot_meta()
    }

    pub fn install_snapshot(&mut self, meta: SnapshotMeta) {
        let last_included_index = meta.last_included_index;
        self.repl.install_snapshot(meta);
        self.visible_up_to = self.visible_up_to.max(last_included_index);
    }

    pub fn snapshot_meta(&self) -> SnapshotMeta {
        self.repl.snapshot_meta()
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

    pub fn pending_batch_oldest_age(&self, now: Instant) -> Option<Duration> {
        self.batcher
            .first_enqueued_at()
            .map(|head| now.saturating_duration_since(head))
    }

    pub fn pending_batch_time_until_deadline(&self, now: Instant) -> Option<Duration> {
        self.batcher.time_until_flush_deadline(now)
    }

    pub fn batching_config(&self) -> (usize, Duration) {
        (self.batcher.max_items(), self.batcher.max_wait())
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
    fn execute_delete_alias_removes_existing_key() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        e.execute_text(2, "DELETE balance").unwrap();

        assert_eq!(e.get("balance"), None);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn execute_read_text_get_returns_current_value_without_committing() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();

        let value = e.execute_read_text("GET balance").unwrap();
        assert_eq!(value, Some("100"));
        assert_eq!(e.metrics().commits_total, 1);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 1);
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);
        assert_eq!(
            e.metrics().last_fallback_reason(),
            Some(FallbackReason::NotGpuEligible)
        );
    }

    #[test]
    fn execute_read_text_get_missing_key_does_not_track_d2h_bytes() {
        let mut e = Engine::new_local();
        let value = e.execute_read_text("GET absent").unwrap();

        assert_eq!(value, None);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
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
        assert_eq!(
            e.metrics().last_batch_flush_reason(),
            Some(BatchFlushReason::Count)
        );
        assert_eq!(e.metrics().batch_wait_samples, 2);
        assert_eq!(e.metrics().batch_wait_total_ms, 0);
        assert_eq!(e.metrics().last_batch_wait_ms(), Some(0));
        assert_eq!(
            e.metrics().h2d_bytes_total,
            "SET a=1".len() as u64 + "SET b=2".len() as u64
        );
        assert_eq!(e.metrics().kernel_exec_samples, 2);
        assert_eq!(e.metrics().kernel_exec_total_ms, 2);
        assert_eq!(e.metrics().last_kernel_exec_ms(), Some(1));
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
        assert_eq!(e.metrics().batch_wait_samples, 1);
        assert_eq!(e.metrics().batch_wait_total_ms, 3);
        assert_eq!(e.metrics().last_batch_wait_ms(), Some(3));
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
    fn admin_flush_without_pending_queue_is_noop() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();

        e.flush_admin().unwrap();

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.pending_batch_oldest_age(t0), None);
        assert_eq!(e.pending_batch_time_until_deadline(t0), None);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 0);
        assert_eq!(e.metrics().last_batch_flush_reason(), None);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn batching_config_reflects_engine_settings() {
        let e = Engine::with_batching(7, Duration::from_millis(42));
        assert_eq!(e.batching_config(), (7, Duration::from_millis(42)));
    }

    #[test]
    fn pending_batch_deadline_counts_down_and_clears_after_flush() {
        let mut e = Engine::with_batching(10, Duration::from_millis(10));
        let t0 = Instant::now();

        assert_eq!(e.pending_batch_time_until_deadline(t0), None);

        e.enqueue_set_text(1, "SET a=9", t0).unwrap();
        assert_eq!(
            e.pending_batch_time_until_deadline(t0 + Duration::from_millis(4)),
            Some(Duration::from_millis(6))
        );
        assert_eq!(
            e.pending_batch_time_until_deadline(t0 + Duration::from_millis(12)),
            Some(Duration::ZERO)
        );

        e.flush_admin().unwrap();
        assert_eq!(
            e.pending_batch_time_until_deadline(t0 + Duration::from_millis(13)),
            None
        );
    }

    #[test]
    fn pending_batch_oldest_age_tracks_then_clears_after_flush() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=9", t0).unwrap();

        let age = e
            .pending_batch_oldest_age(t0 + Duration::from_millis(5))
            .expect("pending batch age should exist");
        assert!(age >= Duration::from_millis(5));

        e.flush_admin().unwrap();
        assert_eq!(
            e.pending_batch_oldest_age(t0 + Duration::from_millis(6)),
            None
        );
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
    fn wal_flush_failure_discards_unflushed_record_from_buffer() {
        let mut e = Engine::new_local();
        e.simulate_next_wal_flush_failure();

        let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();

        assert!(matches!(err, EngineError::Durability(_)));
        assert_eq!(e.wal_flushed_count(), 0);
        assert_eq!(e.wal_buffered_count(), 0);
        assert_eq!(e.wal_unflushed_count(), 0);
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
    fn follower_rejects_batched_enqueue_without_mutating_queue_or_metrics() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_follower(2);

        let t0 = Instant::now();
        let err = e.enqueue_set_text(1, "SET a=1", t0).unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 0);
        assert_eq!(e.metrics().last_batch_flush_reason(), None);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn candidate_rejects_commit_and_batched_enqueue() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_candidate(2);

        let commit_err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();
        assert!(matches!(commit_err, EngineError::NotLeader));

        let enqueue_err = e
            .enqueue_set_text(1, "SET a=1", Instant::now())
            .unwrap_err();
        assert!(matches!(
            enqueue_err,
            ExecuteError::Engine(EngineError::NotLeader)
        ));

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 0);
        assert_eq!(e.visible_up_to(), 0);
    }

    #[test]
    fn failed_admin_flush_does_not_increment_flush_metrics_or_drop_pending_queue() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(2);

        let err = e.flush_admin().unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 0);
        assert_eq!(e.metrics().last_batch_flush_reason(), None);
        assert_eq!(e.metrics().commits_total, 0);
        assert_eq!(e.pending_batch_len(), 1);
    }

    #[test]
    fn batch_flush_wal_failure_requeues_items_for_retry() {
        let mut e = Engine::with_batching(2, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.simulate_next_wal_flush_failure();
        let err = e
            .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
            .unwrap_err();

        assert!(matches!(
            err,
            ExecuteError::Engine(EngineError::Durability(_))
        ));
        assert_eq!(e.pending_batch_len(), 2);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_wait_samples, 0);
        assert_eq!(e.metrics().commits_total, 0);

        e.flush_admin().unwrap();
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.get("a"), Some("1"));
        assert_eq!(e.get("b"), Some("2"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn failed_time_flush_does_not_drop_pending_queue() {
        let mut e = Engine::with_batching(10, Duration::from_millis(2));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(2);

        let err = e.tick_batching(t0 + Duration::from_millis(3)).unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(e.pending_batch_len(), 1);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn follower_tick_without_pending_batch_is_noop() {
        let mut e = Engine::with_batching(10, Duration::from_millis(2));
        e.become_follower(2);

        e.tick_batching(Instant::now()).unwrap();

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn pending_batch_can_be_flushed_after_follower_is_promoted_back_to_leader() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(2);

        let tick_err = e.tick_batching(t0 + Duration::from_secs(1)).unwrap_err();
        assert!(matches!(tick_err, EngineError::NotLeader));
        assert_eq!(e.pending_batch_len(), 1);
        assert_eq!(e.get("a"), None);

        e.become_leader(3);
        e.flush_admin().unwrap();

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.get("a"), Some("1"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
        assert_eq!(e.metrics().commits_total, 1);
    }

    #[test]
    fn execute_text_non_mutations_count_as_not_gpu_eligible_fallbacks() {
        let mut e = Engine::new_local();

        e.execute_text(1, "BEGIN").unwrap();
        e.execute_text(1, "COMMIT").unwrap();
        e.execute_text(2, "BEGIN").unwrap();
        e.execute_text(2, "ROLLBACK").unwrap();
        e.execute_text(3, "GET missing").unwrap();

        assert_eq!(e.active_txn_count(), 0);
        assert_eq!(e.metrics().fallback_total, 5);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 5);
        assert_eq!(
            e.metrics().last_fallback_reason(),
            Some(FallbackReason::NotGpuEligible)
        );
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn enqueue_non_mutations_count_as_not_gpu_eligible_fallbacks() {
        let mut e = Engine::with_batching(2, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "BEGIN", t0).unwrap();
        e.enqueue_set_text(1, "COMMIT", t0).unwrap();
        e.enqueue_set_text(2, "BEGIN", t0).unwrap();
        e.enqueue_set_text(2, "ROLLBACK", t0).unwrap();
        e.enqueue_set_text(3, "GET missing", t0).unwrap();

        assert_eq!(e.active_txn_count(), 0);
        assert_eq!(e.metrics().fallback_total, 5);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 5);
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn commit_and_rollback_require_active_transaction_context() {
        let mut e = Engine::new_local();

        let commit_err = e.execute_text(10, "COMMIT").unwrap_err();
        assert!(matches!(
            commit_err,
            ExecuteError::Txn(TxnError::NotFound(10))
        ));

        let rollback_err = e.execute_text(11, "ROLLBACK").unwrap_err();
        assert!(matches!(
            rollback_err,
            ExecuteError::Txn(TxnError::NotFound(11))
        ));

        e.execute_text(12, "BEGIN").unwrap();
        let duplicate_begin_err = e.execute_text(12, "BEGIN").unwrap_err();
        assert!(matches!(
            duplicate_begin_err,
            ExecuteError::Txn(TxnError::NotActive(12))
        ));

        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.active_txn_count(), 1);
    }

    #[test]
    fn replication_watermarks_track_commit_apply_visibility_and_durability() {
        let mut e = Engine::new_local();

        let before = e.replication_watermarks();
        assert_eq!(before.role, Role::Leader);
        assert_eq!(before.commit_index, 0);
        assert_eq!(before.applied_index, 0);
        assert_eq!(before.visible_index, 0);
        assert_eq!(before.snapshot_id, 0);
        assert_eq!(before.wal_flushed_count, 0);
        assert_eq!(before.wal_buffered_count, 0);
        assert_eq!(before.wal_unflushed_count, 0);

        let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        let after = e.replication_watermarks();

        assert_eq!(after.role, Role::Leader);
        assert!(after.term >= before.term);
        assert_eq!(after.commit_index, token.index);
        assert_eq!(after.applied_index, token.index);
        assert_eq!(after.visible_index, token.index);
        assert_eq!(after.snapshot_id, 0);
        assert!(after.wal_flushed_count >= 1);
        assert_eq!(after.wal_buffered_count, e.wal_buffered_count());
        assert_eq!(after.wal_unflushed_count, e.wal_unflushed_count());
    }

    #[test]
    fn replication_watermarks_do_not_advance_on_rejected_follower_commit() {
        let mut e = Engine::new_local();
        e.become_follower(2);

        let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();
        assert!(matches!(err, EngineError::NotLeader));

        let marks = e.replication_watermarks();
        assert_eq!(marks.role, Role::Follower);
        assert_eq!(marks.term, 2);
        assert_eq!(marks.commit_index, 0);
        assert_eq!(marks.applied_index, 0);
        assert_eq!(marks.visible_index, 0);
        assert_eq!(marks.wal_flushed_count, 0);
        assert_eq!(marks.wal_buffered_count, 0);
        assert_eq!(marks.wal_unflushed_count, 0);
    }

    #[test]
    fn replication_watermarks_include_buffered_wal_records() {
        let mut e = Engine::new_local();

        e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();

        let marks = e.replication_watermarks();
        assert_eq!(marks.wal_buffered_count, 2);
        assert_eq!(marks.wal_flushed_count, 2);
        assert_eq!(marks.wal_unflushed_count, 0);
    }

    #[test]
    fn snapshot_export_tracks_last_applied_index() {
        let mut e = Engine::new_local();
        let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();

        let exported = e.export_snapshot_meta();
        let current = e.snapshot_meta();

        assert_eq!(exported.last_included_index, token.index);
        assert_eq!(current.last_included_index, token.index);
        assert_eq!(exported.snapshot_id, 1);
        assert_eq!(current.snapshot_id, 1);
    }

    #[test]
    fn install_snapshot_advances_visible_and_replication_watermarks() {
        let mut e = Engine::new_local();
        e.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 3,
            snapshot_id: 11,
        });

        let marks = e.replication_watermarks();
        assert_eq!(marks.term, 3);
        assert_eq!(marks.commit_index, 7);
        assert_eq!(marks.applied_index, 7);
        assert_eq!(marks.visible_index, 7);
        assert_eq!(marks.snapshot_id, 11);

        let next = e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();
        assert_eq!(next.index, 8);
        assert_eq!(e.get("b"), Some("2"));
    }

    #[test]
    fn installing_older_snapshot_does_not_rewind_visibility_or_watermarks() {
        let mut e = Engine::new_local();
        let committed = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();

        e.install_snapshot(SnapshotMeta {
            last_included_index: committed.index.saturating_sub(1),
            last_included_term: 1,
            snapshot_id: 99,
        });

        let marks = e.replication_watermarks();
        assert_eq!(marks.commit_index, committed.index);
        assert_eq!(marks.applied_index, committed.index);
        assert_eq!(marks.visible_index, committed.index);
        assert_eq!(e.visible_up_to(), committed.index);
        assert_eq!(e.get("a"), Some("1"));
    }
}
