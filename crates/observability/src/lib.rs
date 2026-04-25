use std::collections::BTreeMap;

use gpu_db_execution::GpuRuntimeSnapshot;
use gpu_db_metrics::{GpuParityIssue, RuntimeMetricsSnapshot};
use gpu_db_types::{Index, Role, TxnId};

const BACKLOG_BLOCKER_WAL: u8 = 1 << 0;
const BACKLOG_BLOCKER_PENDING_BATCH: u8 = 1 << 1;
const BACKLOG_BLOCKER_ACTIVE_TXN: u8 = 1 << 2;
const BACKLOG_BLOCKER_COMMIT_APPLY_GAP: u8 = 1 << 3;
const BACKLOG_BLOCKER_APPLY_VISIBLE_GAP: u8 = 1 << 4;
const KNOWN_BACKLOG_BLOCKER_MASK: u8 = BACKLOG_BLOCKER_WAL
    | BACKLOG_BLOCKER_PENDING_BATCH
    | BACKLOG_BLOCKER_ACTIVE_TXN
    | BACKLOG_BLOCKER_COMMIT_APPLY_GAP
    | BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;
const BACKLOG_BLOCKER_LABELS: [(u8, &str); 5] = [
    (BACKLOG_BLOCKER_WAL, "wal"),
    (BACKLOG_BLOCKER_PENDING_BATCH, "pending_batch"),
    (BACKLOG_BLOCKER_ACTIVE_TXN, "active_txn"),
    (BACKLOG_BLOCKER_COMMIT_APPLY_GAP, "commit_apply_gap"),
    (BACKLOG_BLOCKER_APPLY_VISIBLE_GAP, "apply_visible_gap"),
];

#[derive(Debug, Clone, PartialEq)]
pub struct ReplicationLagSnapshot {
    pub commit_index: Index,
    pub applied_index: Index,
    pub visible_index: Index,
    pub commit_apply_gap: Index,
    pub apply_visible_gap: Index,
}

impl ReplicationLagSnapshot {
    pub fn max_gap(&self) -> Index {
        self.commit_apply_gap.max(self.apply_visible_gap)
    }

    pub fn has_gap(&self) -> bool {
        self.commit_apply_gap > 0 || self.apply_visible_gap > 0
    }

    pub fn is_caught_up(&self) -> bool {
        !self.has_gap()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineTelemetrySnapshot {
    pub role: Role,
    pub replication_lag: ReplicationLagSnapshot,
    pub runtime_metrics: RuntimeMetricsSnapshot,
    pub snapshot_id: u64,
    pub wal_flushed_count: usize,
    pub wal_last_durable_txn_id: Option<TxnId>,
    pub wal_buffered_count: usize,
    pub wal_unflushed_count: usize,
    pub pending_batch_len: usize,
    pub pending_batch_cap: usize,
    pub active_txn_count: usize,
    pub backlog_blocker_count: u8,
    pub backlog_blocker_mask: u8,
    pub mutation_admission_saturated: bool,
    pub quiescent_for_failover: bool,
    pub follower_promotion_ready: bool,
    pub gpu_parity_fallbacks: BTreeMap<GpuParityIssue, u64>,
    pub gpu_runtime: GpuRuntimeSnapshot,
}

impl EngineTelemetrySnapshot {
    pub const fn known_backlog_blocker_mask() -> u8 {
        KNOWN_BACKLOG_BLOCKER_MASK
    }

    pub const fn sanitize_backlog_blocker_mask(mask: u8) -> u8 {
        mask & KNOWN_BACKLOG_BLOCKER_MASK
    }

    pub fn gpu_parity_fallback_total(&self) -> u64 {
        self.gpu_parity_fallbacks.values().copied().sum()
    }

    pub fn has_gpu_runtime_pressure(&self) -> bool {
        self.gpu_runtime.has_pressure()
    }

    pub fn blocked_gpu_ids(&self) -> Vec<u16> {
        self.gpu_runtime.blocked_gpu_ids()
    }

    pub fn has_gpu_parity_fallbacks(&self) -> bool {
        !self.gpu_parity_fallbacks.is_empty()
    }

    pub fn gpu_parity_fallback_count_for(&self, issue: &GpuParityIssue) -> u64 {
        self.gpu_parity_fallbacks.get(issue).copied().unwrap_or(0)
    }

    pub fn hottest_gpu_parity_fallback(&self) -> Option<(GpuParityIssue, u64)> {
        self.gpu_parity_fallbacks
            .iter()
            .max_by_key(|(issue, count)| (*count, *issue))
            .map(|(issue, count)| (*issue, *count))
    }

    pub fn has_backlog_blockers(&self) -> bool {
        Self::sanitize_backlog_blocker_mask(self.backlog_blocker_mask) != 0
    }

    pub fn backlog_blocker_labels(&self) -> Vec<&'static str> {
        BACKLOG_BLOCKER_LABELS
            .iter()
            .filter(|(bit, _)| self.backlog_blocker_mask & bit != 0)
            .map(|(_, label)| *label)
            .collect()
    }

    pub fn backlog_blocker_label_count(&self) -> u8 {
        Self::sanitize_backlog_blocker_mask(self.backlog_blocker_mask).count_ones() as u8
    }

    pub fn backlog_blocker_delimited_labels(&self, delimiter: &str) -> String {
        self.backlog_blocker_labels().join(delimiter)
    }

    pub fn unknown_backlog_blocker_mask(&self) -> u8 {
        self.backlog_blocker_mask & !KNOWN_BACKLOG_BLOCKER_MASK
    }

    pub fn has_unknown_backlog_blockers(&self) -> bool {
        self.unknown_backlog_blocker_mask() != 0
    }

    pub fn unknown_backlog_blocker_count(&self) -> u8 {
        self.unknown_backlog_blocker_mask().count_ones() as u8
    }

    pub fn pending_batch_remaining_capacity(&self) -> usize {
        self.pending_batch_cap
            .saturating_sub(self.pending_batch_len)
    }

    pub fn has_buffered_wal(&self) -> bool {
        self.wal_buffered_count > 0
    }

    pub fn total_backlog_items(&self) -> usize {
        self.wal_unflushed_count + self.pending_batch_len + self.active_txn_count
    }

    pub fn is_write_path_quiescent(&self) -> bool {
        self.total_backlog_items() == 0
    }

    pub fn is_fully_caught_up(&self) -> bool {
        self.replication_lag.is_caught_up()
            && self.total_backlog_items() == 0
            && !self.has_backlog_blockers()
    }
}

pub trait TelemetrySink {
    fn publish(&mut self, snapshot: &EngineTelemetrySnapshot);
}

#[derive(Debug, Default)]
pub struct InMemoryTelemetrySink {
    snapshots: Vec<EngineTelemetrySnapshot>,
}

impl InMemoryTelemetrySink {
    pub fn snapshots(&self) -> &[EngineTelemetrySnapshot] {
        &self.snapshots
    }

    pub fn latest(&self) -> Option<&EngineTelemetrySnapshot> {
        self.snapshots.last()
    }
}

impl TelemetrySink for InMemoryTelemetrySink {
    fn publish(&mut self, snapshot: &EngineTelemetrySnapshot) {
        self.snapshots.push(snapshot.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_snapshot() -> EngineTelemetrySnapshot {
        let metrics = RuntimeMetricsSnapshot {
            commits_total: 2,
            batch_flush_count: 1,
            fallback_total: 0,
            batch_wait_samples: 1,
            batch_wait_total_ms: 3,
            h2d_bytes_total: 64,
            d2h_bytes_total: 16,
            kernel_exec_samples: 1,
            kernel_exec_total_ms: 1,
            kernel_occupancy_samples: 1,
            kernel_occupancy_total_permyriad: 8_500,
            pending_batch_peak: 2,
            fallback_by_reason: Default::default(),
            batch_flush_by_reason: Default::default(),
            last_fallback_reason: None,
            last_batch_flush_reason: None,
            last_batch_wait_ms: Some(3),
            last_kernel_exec_ms: Some(1),
            last_kernel_occupancy_permyriad: Some(8_500),
            last_pending_batch_len: Some(0),
        };
        EngineTelemetrySnapshot {
            role: Role::Leader,
            replication_lag: ReplicationLagSnapshot {
                commit_index: 12,
                applied_index: 12,
                visible_index: 12,
                commit_apply_gap: 0,
                apply_visible_gap: 0,
            },
            runtime_metrics: metrics,
            snapshot_id: 7,
            wal_flushed_count: 12,
            wal_last_durable_txn_id: Some(42),
            wal_buffered_count: 12,
            wal_unflushed_count: 0,
            pending_batch_len: 0,
            pending_batch_cap: 64,
            active_txn_count: 0,
            backlog_blocker_count: 0,
            backlog_blocker_mask: 0,
            mutation_admission_saturated: false,
            quiescent_for_failover: true,
            follower_promotion_ready: false,
            gpu_parity_fallbacks: BTreeMap::new(),
            gpu_runtime: GpuRuntimeSnapshot::default(),
        }
    }

    #[test]
    fn in_memory_sink_records_snapshots_in_order() {
        let snapshot = empty_snapshot();

        let mut sink = InMemoryTelemetrySink::default();
        assert!(sink.latest().is_none());

        sink.publish(&snapshot);
        sink.publish(&snapshot);

        assert_eq!(sink.snapshots().len(), 2);
        assert_eq!(sink.snapshots()[0], sink.snapshots()[1]);
        assert_eq!(sink.snapshots()[0].gpu_parity_fallback_total(), 0);
        assert!(!sink.snapshots()[0].has_gpu_parity_fallbacks());
        assert_eq!(sink.latest(), Some(&snapshot));
    }

    #[test]
    fn parity_fallback_helpers_report_counts_and_hottest_issue() {
        let mut snapshot = empty_snapshot();
        let issue_120 = GpuParityIssue {
            id: "GPU-120",
            owner: "runtime",
            milestone: "m0-bootstrap",
        };
        let issue_121 = GpuParityIssue {
            id: "GPU-121",
            owner: "runtime",
            milestone: "m0-bootstrap",
        };
        snapshot.gpu_parity_fallbacks.insert(issue_120, 2);
        snapshot.gpu_parity_fallbacks.insert(issue_121, 3);

        assert!(snapshot.has_gpu_parity_fallbacks());
        assert_eq!(snapshot.gpu_parity_fallback_total(), 5);
        assert_eq!(snapshot.gpu_parity_fallback_count_for(&issue_120), 2);
        assert_eq!(snapshot.gpu_parity_fallback_count_for(&issue_121), 3);
        assert_eq!(snapshot.hottest_gpu_parity_fallback(), Some((issue_121, 3)));
    }

    #[test]
    fn gpu_runtime_helpers_report_pressure_and_blocked_ids() {
        let mut snapshot = empty_snapshot();
        assert!(!snapshot.has_gpu_runtime_pressure());
        assert!(snapshot.blocked_gpu_ids().is_empty());

        snapshot.gpu_runtime = GpuRuntimeSnapshot {
            unavailable_gpu_ids: vec![0, 2],
            memory_pressured_gpu_ids: vec![2, 4],
            saturated: true,
        };

        assert!(snapshot.has_gpu_runtime_pressure());
        assert_eq!(snapshot.blocked_gpu_ids(), vec![0, 2, 4]);
    }

    #[test]
    fn telemetry_helpers_report_write_path_readiness() {
        let mut snapshot = empty_snapshot();
        assert_eq!(
            EngineTelemetrySnapshot::known_backlog_blocker_mask(),
            KNOWN_BACKLOG_BLOCKER_MASK
        );
        assert!(!snapshot.has_backlog_blockers());
        assert_eq!(snapshot.backlog_blocker_label_count(), 0);
        assert_eq!(
            snapshot.backlog_blocker_labels(),
            Vec::<&'static str>::new()
        );
        assert_eq!(snapshot.unknown_backlog_blocker_mask(), 0);
        assert!(!snapshot.has_unknown_backlog_blockers());
        assert_eq!(snapshot.unknown_backlog_blocker_count(), 0);
        assert_eq!(snapshot.backlog_blocker_delimited_labels(","), "");
        assert_eq!(snapshot.snapshot_id, 7);
        assert_eq!(snapshot.wal_flushed_count, 12);
        assert_eq!(snapshot.wal_last_durable_txn_id, Some(42));
        assert!(snapshot.has_buffered_wal());
        assert_eq!(snapshot.pending_batch_remaining_capacity(), 64);
        assert_eq!(snapshot.total_backlog_items(), 0);
        assert!(snapshot.is_write_path_quiescent());
        assert!(snapshot.is_fully_caught_up());
        assert!(snapshot.quiescent_for_failover);
        assert!(!snapshot.follower_promotion_ready);

        snapshot.wal_buffered_count = 0;
        snapshot.wal_unflushed_count = 2;
        snapshot.pending_batch_len = 5;
        snapshot.active_txn_count = 1;
        snapshot.backlog_blocker_count = 3;
        snapshot.backlog_blocker_mask = 0b0_0111;
        snapshot.quiescent_for_failover = false;

        assert!(snapshot.has_backlog_blockers());
        assert_eq!(snapshot.backlog_blocker_label_count(), 3);
        assert_eq!(
            snapshot.backlog_blocker_labels(),
            vec!["wal", "pending_batch", "active_txn"]
        );
        assert_eq!(snapshot.unknown_backlog_blocker_mask(), 0);
        assert!(!snapshot.has_unknown_backlog_blockers());
        assert_eq!(snapshot.unknown_backlog_blocker_count(), 0);
        assert_eq!(
            snapshot.backlog_blocker_delimited_labels(" | "),
            "wal | pending_batch | active_txn"
        );
        assert!(!snapshot.has_buffered_wal());
        assert_eq!(snapshot.pending_batch_remaining_capacity(), 59);
        assert_eq!(snapshot.total_backlog_items(), 8);
        assert!(!snapshot.is_write_path_quiescent());
        assert!(!snapshot.is_fully_caught_up());
        assert!(!snapshot.quiescent_for_failover);
    }

    #[test]
    fn telemetry_backlog_helpers_ignore_unknown_bits() {
        let mut snapshot = empty_snapshot();
        snapshot.backlog_blocker_mask =
            BACKLOG_BLOCKER_WAL | BACKLOG_BLOCKER_APPLY_VISIBLE_GAP | (1 << 7);
        snapshot.backlog_blocker_count = 3;

        assert!(snapshot.has_backlog_blockers());
        assert_eq!(snapshot.backlog_blocker_label_count(), 2);
        assert_eq!(
            EngineTelemetrySnapshot::sanitize_backlog_blocker_mask(snapshot.backlog_blocker_mask),
            BACKLOG_BLOCKER_WAL | BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
        assert_eq!(snapshot.unknown_backlog_blocker_mask(), 1 << 7);
        assert!(snapshot.has_unknown_backlog_blockers());
        assert_eq!(snapshot.unknown_backlog_blocker_count(), 1);
        assert_eq!(
            snapshot.backlog_blocker_labels(),
            vec!["wal", "apply_visible_gap"]
        );
        assert_eq!(
            snapshot.backlog_blocker_delimited_labels(";"),
            "wal;apply_visible_gap"
        );
        assert!(!snapshot.is_fully_caught_up());
    }

    #[test]
    fn replication_lag_helpers_report_catchup_state() {
        let caught_up = ReplicationLagSnapshot {
            commit_index: 9,
            applied_index: 9,
            visible_index: 9,
            commit_apply_gap: 0,
            apply_visible_gap: 0,
        };
        assert_eq!(caught_up.max_gap(), 0);
        assert!(!caught_up.has_gap());
        assert!(caught_up.is_caught_up());

        let lagging = ReplicationLagSnapshot {
            commit_index: 12,
            applied_index: 10,
            visible_index: 8,
            commit_apply_gap: 2,
            apply_visible_gap: 2,
        };
        assert_eq!(lagging.max_gap(), 2);
        assert!(lagging.has_gap());
        assert!(!lagging.is_caught_up());
    }
}
