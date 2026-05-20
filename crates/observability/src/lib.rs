use std::collections::BTreeMap;

use gpu_db_execution::{CudaDeviceMemoryProof, GpuRuntimeSnapshot};
use gpu_db_metrics::{FallbackReason, GpuParityIssue, RuntimeMetricsSnapshot};
use gpu_db_types::{Index, Role, Term, TxnId};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotStatus {
    pub snapshot_id: u64,
    pub last_included_index: Index,
    pub last_included_term: Term,
    pub visible_index: Index,
}

impl SnapshotStatus {
    pub fn served_frontier(&self) -> Index {
        self.visible_index.max(self.last_included_index)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActiveFallbackReason {
    GpuUnavailable { gpu_ids: Vec<u16> },
    GpuMemoryPressure { gpu_ids: Vec<u16> },
    GpuQueueSaturated,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FallbackStatus {
    pub last_reason: Option<FallbackReason>,
    pub gpu_parity_fallbacks: BTreeMap<GpuParityIssue, u64>,
    pub active_reasons: Vec<ActiveFallbackReason>,
    pub gpu_runtime: GpuRuntimeSnapshot,
}

impl FallbackStatus {
    pub fn gpu_parity_fallback_total(&self) -> u64 {
        self.gpu_parity_fallbacks.values().copied().sum()
    }

    pub fn has_gpu_parity_fallbacks(&self) -> bool {
        !self.gpu_parity_fallbacks.is_empty()
    }

    pub fn is_actively_degraded(&self) -> bool {
        !self.active_reasons.is_empty()
    }

    pub fn active_reason_labels(&self) -> Vec<&'static str> {
        self.active_reasons
            .iter()
            .map(|reason| match reason {
                ActiveFallbackReason::GpuUnavailable { .. } => "gpu_unavailable",
                ActiveFallbackReason::GpuMemoryPressure { .. } => "gpu_memory_pressure",
                ActiveFallbackReason::GpuQueueSaturated => "gpu_queue_saturated",
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadinessStatus {
    pub pending_batch_len: usize,
    pub pending_batch_cap: usize,
    pub active_txn_count: usize,
    pub wal_unflushed_count: usize,
    pub backlog_blocker_count: u8,
    pub backlog_blocker_mask: u8,
    pub mutation_admission_saturated: bool,
    pub quiescent_for_failover: bool,
    pub follower_promotion_ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidencyTableStatus {
    pub schema: String,
    pub table: String,
    pub gpu_id: u16,
    pub cache_state: String,
    pub row_count: usize,
    pub column_count: usize,
    pub resident_bytes: u64,
    pub valid_through_index: Index,
    pub valid: bool,
    pub invalidated_by_txn_id: Option<TxnId>,
    pub invalidated_at_index: Option<Index>,
    pub invalidated_by_memory_pressure: bool,
    pub memory_pressure_active: bool,
    pub admission_budget_bytes: Option<u64>,
    pub resident_bytes_after_admission: u64,
    pub evicted_tables_on_admission: Vec<String>,
    pub last_decision_accepted: Option<bool>,
    pub last_decision_reason: Option<String>,
    pub last_decision_current_bytes_before: Option<u64>,
    pub last_decision_current_bytes_after: Option<u64>,
    pub device_memory_proof: Option<CudaDeviceMemoryProof>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidentRouteDecisionStatus {
    pub table: String,
    pub gpu_id: Option<u16>,
    pub accepted: bool,
    pub reason: String,
    pub query_shape: String,
    pub cache_state: String,
    pub valid: bool,
    pub has_retained_device_memory: bool,
    pub estimated_rows: usize,
    pub resident_bytes: u64,
    pub budget_bytes: Option<u64>,
    pub refresh_resident_bytes: Option<u64>,
    pub h2d_bytes_if_resident: u64,
    pub h2d_bytes_if_cold: u64,
    pub d2h_rows_estimate: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RelationalResidencyStatus {
    pub tables: Vec<RelationalResidencyTableStatus>,
    pub latest_route_decisions: Vec<RelationalResidentRouteDecisionStatus>,
    pub resident_bytes_by_gpu: BTreeMap<u16, u64>,
    pub budget_bytes_by_gpu: BTreeMap<u16, u64>,
}

impl RelationalResidencyStatus {
    pub fn snapshot_count(&self) -> usize {
        self.tables.len()
    }

    pub fn valid_snapshot_count(&self) -> usize {
        self.tables.iter().filter(|table| table.valid).count()
    }

    pub fn invalid_snapshot_count(&self) -> usize {
        self.snapshot_count()
            .saturating_sub(self.valid_snapshot_count())
    }

    pub fn memory_pressured_snapshot_count(&self) -> usize {
        self.tables
            .iter()
            .filter(|table| table.memory_pressure_active || table.invalidated_by_memory_pressure)
            .count()
    }

    pub fn total_resident_bytes(&self) -> u64 {
        self.resident_bytes_by_gpu.values().copied().sum()
    }

    pub fn table(&self, table: &str) -> Option<&RelationalResidencyTableStatus> {
        self.tables.iter().find(|status| status.table == table)
    }

    pub fn latest_route_decision(
        &self,
        table: &str,
    ) -> Option<&RelationalResidentRouteDecisionStatus> {
        self.latest_route_decisions
            .iter()
            .find(|decision| decision.table == table)
    }

    pub fn has_resident_tables(&self) -> bool {
        !self.tables.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineStatusSnapshot {
    pub role: Role,
    pub term: Term,
    pub snapshot: SnapshotStatus,
    pub replication_lag: ReplicationLagSnapshot,
    pub readiness: ReadinessStatus,
    pub fallback: FallbackStatus,
    pub relational_residency: RelationalResidencyStatus,
    pub runtime_metrics: RuntimeMetricsSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineStatusInvariantError {
    #[error("applied index {applied_index} exceeds commit index {commit_index}")]
    AppliedExceedsCommit {
        commit_index: Index,
        applied_index: Index,
    },
    #[error("visible index {visible_index} exceeds applied index {applied_index}")]
    VisibleExceedsApplied {
        applied_index: Index,
        visible_index: Index,
    },
    #[error(
        "snapshot last_included_index {last_included_index} exceeds visible frontier {visible_index}"
    )]
    SnapshotExceedsVisible {
        last_included_index: Index,
        visible_index: Index,
    },
    #[error(
        "commit/apply gap {commit_apply_gap} does not match commit {commit_index} and applied {applied_index}"
    )]
    CommitApplyGapMismatch {
        commit_index: Index,
        applied_index: Index,
        commit_apply_gap: Index,
    },
    #[error(
        "apply/visible gap {apply_visible_gap} does not match applied {applied_index} and visible {visible_index}"
    )]
    ApplyVisibleGapMismatch {
        applied_index: Index,
        visible_index: Index,
        apply_visible_gap: Index,
    },
    #[error(
        "backlog blocker count {backlog_blocker_count} does not match backlog mask {backlog_blocker_mask:#010b}"
    )]
    BacklogBlockerCountMismatch {
        backlog_blocker_count: u8,
        backlog_blocker_mask: u8,
    },
    #[error(
        "mutation admission saturated={mutation_admission_saturated} is inconsistent with pending queue {pending_batch_len}/{pending_batch_cap}"
    )]
    MutationAdmissionMismatch {
        pending_batch_len: usize,
        pending_batch_cap: usize,
        mutation_admission_saturated: bool,
    },
    #[error("leader-only quiescent_for_failover flag set while role is {role:?}")]
    QuiescentRoleMismatch { role: Role },
    #[error("follower_promotion_ready flag set while role is {role:?}")]
    PromotionReadyRoleMismatch { role: Role },
}

impl EngineStatusSnapshot {
    pub fn new(
        role: Role,
        term: Term,
        snapshot: SnapshotStatus,
        replication_lag: ReplicationLagSnapshot,
        readiness: ReadinessStatus,
        fallback: FallbackStatus,
        runtime_metrics: RuntimeMetricsSnapshot,
    ) -> Result<Self, EngineStatusInvariantError> {
        let snapshot = Self {
            role,
            term,
            snapshot,
            replication_lag,
            readiness,
            fallback,
            relational_residency: RelationalResidencyStatus::default(),
            runtime_metrics,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> Result<(), EngineStatusInvariantError> {
        let lag = &self.replication_lag;
        let readiness = &self.readiness;
        if lag.applied_index > lag.commit_index {
            return Err(EngineStatusInvariantError::AppliedExceedsCommit {
                commit_index: lag.commit_index,
                applied_index: lag.applied_index,
            });
        }
        if self.snapshot.visible_index > lag.applied_index {
            return Err(EngineStatusInvariantError::VisibleExceedsApplied {
                applied_index: lag.applied_index,
                visible_index: self.snapshot.visible_index,
            });
        }
        if self.snapshot.last_included_index > self.snapshot.visible_index {
            return Err(EngineStatusInvariantError::SnapshotExceedsVisible {
                last_included_index: self.snapshot.last_included_index,
                visible_index: self.snapshot.visible_index,
            });
        }
        if lag.commit_apply_gap != lag.commit_index.saturating_sub(lag.applied_index) {
            return Err(EngineStatusInvariantError::CommitApplyGapMismatch {
                commit_index: lag.commit_index,
                applied_index: lag.applied_index,
                commit_apply_gap: lag.commit_apply_gap,
            });
        }
        if lag.apply_visible_gap
            != lag
                .applied_index
                .saturating_sub(self.snapshot.visible_index)
        {
            return Err(EngineStatusInvariantError::ApplyVisibleGapMismatch {
                applied_index: lag.applied_index,
                visible_index: self.snapshot.visible_index,
                apply_visible_gap: lag.apply_visible_gap,
            });
        }

        let sanitized_mask = Self::known_backlog_blocker_mask() & readiness.backlog_blocker_mask;
        let expected_blocker_count = sanitized_mask.count_ones() as u8;
        if readiness.backlog_blocker_count != expected_blocker_count {
            return Err(EngineStatusInvariantError::BacklogBlockerCountMismatch {
                backlog_blocker_count: readiness.backlog_blocker_count,
                backlog_blocker_mask: readiness.backlog_blocker_mask,
            });
        }

        let expected_mutation_admission_saturated =
            readiness.pending_batch_len >= readiness.pending_batch_cap;
        if readiness.mutation_admission_saturated != expected_mutation_admission_saturated {
            return Err(EngineStatusInvariantError::MutationAdmissionMismatch {
                pending_batch_len: readiness.pending_batch_len,
                pending_batch_cap: readiness.pending_batch_cap,
                mutation_admission_saturated: readiness.mutation_admission_saturated,
            });
        }

        if readiness.quiescent_for_failover && self.role != Role::Leader {
            return Err(EngineStatusInvariantError::QuiescentRoleMismatch { role: self.role });
        }
        if readiness.follower_promotion_ready && self.role != Role::Follower {
            return Err(EngineStatusInvariantError::PromotionReadyRoleMismatch { role: self.role });
        }

        Ok(())
    }

    pub const fn known_backlog_blocker_mask() -> u8 {
        KNOWN_BACKLOG_BLOCKER_MASK
    }

    pub fn served_snapshot_frontier(&self) -> Index {
        self.snapshot.served_frontier()
    }

    pub fn latest_fallback_reason(&self) -> Option<FallbackReason> {
        self.fallback.last_reason
    }

    pub fn why_routed_to_fallback_labels(&self) -> Vec<&'static str> {
        self.fallback.active_reason_labels()
    }

    pub fn backlog_blocker_labels(&self) -> Vec<&'static str> {
        BACKLOG_BLOCKER_LABELS
            .iter()
            .filter(|(bit, _)| self.readiness.backlog_blocker_mask & bit != 0)
            .map(|(_, label)| *label)
            .collect()
    }

    pub fn replication_distance(&self) -> Index {
        self.replication_lag.max_gap()
    }

    pub fn resident_table_count(&self) -> usize {
        self.relational_residency.snapshot_count()
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
    pub relational_residency: RelationalResidencyStatus,
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

    pub fn pending_batch_utilization_permyriad(&self) -> u16 {
        if self.pending_batch_cap == 0 {
            return 0;
        }
        let len = self.pending_batch_len.min(self.pending_batch_cap) as u128;
        let cap = self.pending_batch_cap as u128;
        ((len * 10_000) / cap) as u16
    }

    pub fn pending_batch_remaining_capacity_permyriad(&self) -> u16 {
        10_000_u16.saturating_sub(self.pending_batch_utilization_permyriad())
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

    pub fn resident_table_count(&self) -> usize {
        self.relational_residency.snapshot_count()
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
            relational_residency: RelationalResidencyStatus::default(),
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
        assert_eq!(snapshot.pending_batch_utilization_permyriad(), 0);
        assert_eq!(
            snapshot.pending_batch_remaining_capacity_permyriad(),
            10_000
        );
        assert_eq!(snapshot.total_backlog_items(), 0);
        assert!(snapshot.is_write_path_quiescent());
        assert!(snapshot.is_fully_caught_up());
        assert_eq!(snapshot.resident_table_count(), 0);
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
        assert_eq!(snapshot.pending_batch_utilization_permyriad(), 781);
        assert_eq!(snapshot.pending_batch_remaining_capacity_permyriad(), 9_219);
        assert_eq!(snapshot.total_backlog_items(), 8);
        assert!(!snapshot.is_write_path_quiescent());
        assert!(!snapshot.is_fully_caught_up());
        assert!(!snapshot.quiescent_for_failover);
    }

    #[test]
    fn residency_status_helpers_summarize_tables_and_budgets() {
        let status = RelationalResidencyStatus {
            tables: vec![
                RelationalResidencyTableStatus {
                    schema: "public".to_string(),
                    table: "events".to_string(),
                    gpu_id: 0,
                    cache_state: "Valid".to_string(),
                    row_count: 2,
                    column_count: 2,
                    resident_bytes: 128,
                    valid_through_index: 4,
                    valid: true,
                    invalidated_by_txn_id: None,
                    invalidated_at_index: None,
                    invalidated_by_memory_pressure: false,
                    memory_pressure_active: false,
                    admission_budget_bytes: Some(256),
                    resident_bytes_after_admission: 128,
                    evicted_tables_on_admission: Vec::new(),
                    last_decision_accepted: Some(true),
                    last_decision_reason: Some("admitted".to_string()),
                    last_decision_current_bytes_before: Some(0),
                    last_decision_current_bytes_after: Some(128),
                    device_memory_proof: None,
                },
                RelationalResidencyTableStatus {
                    schema: "public".to_string(),
                    table: "stale_events".to_string(),
                    gpu_id: 0,
                    cache_state: "Invalidated".to_string(),
                    row_count: 1,
                    column_count: 2,
                    resident_bytes: 64,
                    valid_through_index: 2,
                    valid: false,
                    invalidated_by_txn_id: Some(5),
                    invalidated_at_index: Some(5),
                    invalidated_by_memory_pressure: true,
                    memory_pressure_active: true,
                    admission_budget_bytes: Some(256),
                    resident_bytes_after_admission: 192,
                    evicted_tables_on_admission: vec!["old_events".to_string()],
                    last_decision_accepted: Some(false),
                    last_decision_reason: Some("gpu memory pressure".to_string()),
                    last_decision_current_bytes_before: Some(192),
                    last_decision_current_bytes_after: Some(192),
                    device_memory_proof: None,
                },
            ],
            latest_route_decisions: vec![RelationalResidentRouteDecisionStatus {
                table: "events".to_string(),
                gpu_id: Some(0),
                accepted: true,
                reason: "resident route accepted".to_string(),
                query_shape: "count_all".to_string(),
                cache_state: "Valid".to_string(),
                valid: true,
                has_retained_device_memory: true,
                estimated_rows: 2,
                resident_bytes: 128,
                budget_bytes: Some(256),
                refresh_resident_bytes: None,
                h2d_bytes_if_resident: 0,
                h2d_bytes_if_cold: 128,
                d2h_rows_estimate: 1,
            }],
            resident_bytes_by_gpu: BTreeMap::from([(0, 192)]),
            budget_bytes_by_gpu: BTreeMap::from([(0, 256)]),
        };

        assert!(status.has_resident_tables());
        assert_eq!(status.snapshot_count(), 2);
        assert_eq!(status.valid_snapshot_count(), 1);
        assert_eq!(status.invalid_snapshot_count(), 1);
        assert_eq!(status.memory_pressured_snapshot_count(), 1);
        assert_eq!(status.total_resident_bytes(), 192);
        assert_eq!(status.table("events").unwrap().row_count, 2);
        assert!(status.latest_route_decision("events").unwrap().accepted);
        assert!(status.table("missing").is_none());
        assert!(status.latest_route_decision("missing").is_none());
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
        assert_eq!(snapshot.pending_batch_utilization_permyriad(), 0);
        assert_eq!(
            snapshot.pending_batch_remaining_capacity_permyriad(),
            10_000
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
