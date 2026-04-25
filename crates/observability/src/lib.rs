use std::collections::BTreeMap;

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

#[derive(Debug, Clone, PartialEq)]
pub struct EngineTelemetrySnapshot {
    pub role: Role,
    pub replication_lag: ReplicationLagSnapshot,
    pub runtime_metrics: RuntimeMetricsSnapshot,
    pub gpu_parity_fallbacks: BTreeMap<GpuParityIssue, u64>,
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
}

impl TelemetrySink for InMemoryTelemetrySink {
    fn publish(&mut self, snapshot: &EngineTelemetrySnapshot) {
        self.snapshots.push(snapshot.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_sink_records_snapshots_in_order() {
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
        let snapshot = EngineTelemetrySnapshot {
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
        };

        let mut sink = InMemoryTelemetrySink::default();
        sink.publish(&snapshot);
        sink.publish(&snapshot);

        assert_eq!(sink.snapshots().len(), 2);
        assert_eq!(sink.snapshots()[0], sink.snapshots()[1]);
    }
}
