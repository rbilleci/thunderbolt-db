use std::collections::BTreeMap;

use gpu_db_execution::GpuRuntimeSnapshot;
use gpu_db_metrics::{GpuParityIssue, RuntimeMetricsSnapshot};
use gpu_db_types::{Index, Role};

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
    pub gpu_parity_fallbacks: BTreeMap<GpuParityIssue, u64>,
    pub gpu_runtime: GpuRuntimeSnapshot,
}

impl EngineTelemetrySnapshot {
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
