use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use gpu_db_execution::GpuFallbackReason;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FallbackReason {
    NotGpuEligible,
    GpuMvccReadParityGap,
    GpuUnavailable,
    GpuQueueSaturated,
    GpuMemoryPressure,
}

impl From<GpuFallbackReason> for FallbackReason {
    fn from(value: GpuFallbackReason) -> Self {
        match value {
            GpuFallbackReason::Unavailable => Self::GpuUnavailable,
            GpuFallbackReason::QueueSaturated => Self::GpuQueueSaturated,
            GpuFallbackReason::MemoryPressure => Self::GpuMemoryPressure,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GpuParityIssue {
    pub id: &'static str,
    pub owner: &'static str,
    pub milestone: &'static str,
}

impl FallbackReason {
    pub fn gpu_parity_issue(self) -> Option<GpuParityIssue> {
        match self {
            Self::NotGpuEligible => None,
            Self::GpuMvccReadParityGap => Some(GpuParityIssue {
                id: "GPU-123",
                owner: "execution",
                milestone: "m0-bootstrap",
            }),
            Self::GpuUnavailable => Some(GpuParityIssue {
                id: "GPU-120",
                owner: "runtime",
                milestone: "m0-bootstrap",
            }),
            Self::GpuQueueSaturated => Some(GpuParityIssue {
                id: "GPU-121",
                owner: "runtime",
                milestone: "m0-bootstrap",
            }),
            Self::GpuMemoryPressure => Some(GpuParityIssue {
                id: "GPU-122",
                owner: "runtime",
                milestone: "m0-bootstrap",
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchFlushReason {
    Count,
    Time,
    Admin,
}

/// Runtime counters as lock-free atomics plus a small mutex-guarded `tail` for the
/// per-reason maps and "last observed" values, so the whole struct updates through
/// `&self` — the prerequisite for concurrent readers on the engine read path
/// (P1-M3 step 3). Counter *totals* saturate (preserving the previous
/// `saturating_add` semantics); per-counter atomicity is sufficient because no metric
/// requires cross-field consistency. Read a coherent view via [`RuntimeMetrics::snapshot`].
#[derive(Debug, Default)]
pub struct RuntimeMetrics {
    commits_total: AtomicU64,
    batch_flush_count: AtomicU64,
    fallback_total: AtomicU64,
    batch_wait_samples: AtomicU64,
    batch_wait_total_ms: AtomicU64,
    h2d_bytes_total: AtomicU64,
    d2h_bytes_total: AtomicU64,
    kernel_exec_samples: AtomicU64,
    kernel_exec_total_ms: AtomicU64,
    kernel_event_timing_samples: AtomicU64,
    kernel_event_elapsed_total_us: AtomicU64,
    kernel_occupancy_samples: AtomicU64,
    kernel_occupancy_total_permyriad: AtomicU64,
    pending_batch_peak: AtomicUsize,
    tail: Mutex<RuntimeMetricsTail>,
}

/// Fields that are not single integers (per-reason maps and latest-observation
/// values), guarded by one mutex so `RuntimeMetrics` updates stay `&self`.
#[derive(Debug, Default)]
struct RuntimeMetricsTail {
    fallback_by_reason: BTreeMap<FallbackReason, u64>,
    batch_flush_by_reason: BTreeMap<BatchFlushReason, u64>,
    last_fallback_reason: Option<FallbackReason>,
    last_batch_flush_reason: Option<BatchFlushReason>,
    last_batch_wait_ms: Option<u64>,
    last_kernel_exec_ms: Option<u64>,
    last_kernel_event_elapsed_us: Option<u64>,
    last_kernel_occupancy_permyriad: Option<u16>,
    last_pending_batch_len: Option<usize>,
}

/// Saturating add on an atomic counter (a CAS loop), preserving the previous
/// `saturating_add` overflow behavior under `&self`.
fn saturating_fetch_add(counter: &AtomicU64, value: u64) {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_add(value);
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => current = actual,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeMetricsSnapshot {
    pub commits_total: u64,
    pub batch_flush_count: u64,
    pub fallback_total: u64,
    pub batch_wait_samples: u64,
    pub batch_wait_total_ms: u64,
    pub h2d_bytes_total: u64,
    pub d2h_bytes_total: u64,
    pub kernel_exec_samples: u64,
    pub kernel_exec_total_ms: u64,
    pub kernel_event_timing_samples: u64,
    pub kernel_event_elapsed_total_us: u64,
    pub kernel_occupancy_samples: u64,
    pub kernel_occupancy_total_permyriad: u64,
    pub pending_batch_peak: usize,
    pub fallback_by_reason: BTreeMap<FallbackReason, u64>,
    pub batch_flush_by_reason: BTreeMap<BatchFlushReason, u64>,
    pub last_fallback_reason: Option<FallbackReason>,
    pub last_batch_flush_reason: Option<BatchFlushReason>,
    pub last_batch_wait_ms: Option<u64>,
    pub last_kernel_exec_ms: Option<u64>,
    pub last_kernel_event_elapsed_us: Option<u64>,
    pub last_kernel_occupancy_permyriad: Option<u16>,
    pub last_pending_batch_len: Option<usize>,
}

impl RuntimeMetrics {
    /// Lock the tail, recovering from poison (one panicking observer must not wedge
    /// metrics for every reader/writer).
    fn tail(&self) -> std::sync::MutexGuard<'_, RuntimeMetricsTail> {
        self.tail
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn snapshot(&self) -> RuntimeMetricsSnapshot {
        let tail = self.tail();
        RuntimeMetricsSnapshot {
            commits_total: self.commits_total.load(Ordering::Relaxed),
            batch_flush_count: self.batch_flush_count.load(Ordering::Relaxed),
            fallback_total: self.fallback_total.load(Ordering::Relaxed),
            batch_wait_samples: self.batch_wait_samples.load(Ordering::Relaxed),
            batch_wait_total_ms: self.batch_wait_total_ms.load(Ordering::Relaxed),
            h2d_bytes_total: self.h2d_bytes_total.load(Ordering::Relaxed),
            d2h_bytes_total: self.d2h_bytes_total.load(Ordering::Relaxed),
            kernel_exec_samples: self.kernel_exec_samples.load(Ordering::Relaxed),
            kernel_exec_total_ms: self.kernel_exec_total_ms.load(Ordering::Relaxed),
            kernel_event_timing_samples: self.kernel_event_timing_samples.load(Ordering::Relaxed),
            kernel_event_elapsed_total_us: self
                .kernel_event_elapsed_total_us
                .load(Ordering::Relaxed),
            kernel_occupancy_samples: self.kernel_occupancy_samples.load(Ordering::Relaxed),
            kernel_occupancy_total_permyriad: self
                .kernel_occupancy_total_permyriad
                .load(Ordering::Relaxed),
            pending_batch_peak: self.pending_batch_peak.load(Ordering::Relaxed),
            fallback_by_reason: tail.fallback_by_reason.clone(),
            batch_flush_by_reason: tail.batch_flush_by_reason.clone(),
            last_fallback_reason: tail.last_fallback_reason,
            last_batch_flush_reason: tail.last_batch_flush_reason,
            last_batch_wait_ms: tail.last_batch_wait_ms,
            last_kernel_exec_ms: tail.last_kernel_exec_ms,
            last_kernel_event_elapsed_us: tail.last_kernel_event_elapsed_us,
            last_kernel_occupancy_permyriad: tail.last_kernel_occupancy_permyriad,
            last_pending_batch_len: tail.last_pending_batch_len,
        }
    }

    pub fn inc_commit(&self) {
        self.commits_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe_batch_wait_ms(&self, wait_ms: u64) {
        self.batch_wait_samples.fetch_add(1, Ordering::Relaxed);
        saturating_fetch_add(&self.batch_wait_total_ms, wait_ms);
        self.tail().last_batch_wait_ms = Some(wait_ms);
    }

    pub fn inc_batch_flush(&self, reason: BatchFlushReason) {
        self.batch_flush_count.fetch_add(1, Ordering::Relaxed);
        let mut tail = self.tail();
        *tail.batch_flush_by_reason.entry(reason).or_insert(0) += 1;
        tail.last_batch_flush_reason = Some(reason);
    }

    pub fn inc_fallback(&self, reason: FallbackReason) {
        self.fallback_total.fetch_add(1, Ordering::Relaxed);
        let mut tail = self.tail();
        *tail.fallback_by_reason.entry(reason).or_insert(0) += 1;
        tail.last_fallback_reason = Some(reason);
    }

    pub fn inc_gpu_fallback(&self, reason: GpuFallbackReason) {
        self.inc_fallback(reason.into());
    }

    pub fn observe_h2d_bytes(&self, bytes: u64) {
        saturating_fetch_add(&self.h2d_bytes_total, bytes);
    }

    pub fn observe_d2h_bytes(&self, bytes: u64) {
        saturating_fetch_add(&self.d2h_bytes_total, bytes);
    }

    pub fn observe_kernel_exec_ms(&self, exec_ms: u64) {
        self.kernel_exec_samples.fetch_add(1, Ordering::Relaxed);
        saturating_fetch_add(&self.kernel_exec_total_ms, exec_ms);
        self.tail().last_kernel_exec_ms = Some(exec_ms);
    }

    pub fn observe_kernel_event_elapsed_us(&self, elapsed_us: u64) {
        self.kernel_event_timing_samples
            .fetch_add(1, Ordering::Relaxed);
        saturating_fetch_add(&self.kernel_event_elapsed_total_us, elapsed_us);
        self.tail().last_kernel_event_elapsed_us = Some(elapsed_us);
    }

    pub fn observe_kernel_occupancy_permyriad(&self, occupancy_permyriad: u16) {
        self.kernel_occupancy_samples
            .fetch_add(1, Ordering::Relaxed);
        saturating_fetch_add(
            &self.kernel_occupancy_total_permyriad,
            u64::from(occupancy_permyriad),
        );
        self.tail().last_kernel_occupancy_permyriad = Some(occupancy_permyriad);
    }

    pub fn observe_pending_batch_len(&self, len: usize) {
        self.pending_batch_peak.fetch_max(len, Ordering::Relaxed);
        self.tail().last_pending_batch_len = Some(len);
    }

    pub fn fallback_for(&self, reason: FallbackReason) -> u64 {
        self.tail()
            .fallback_by_reason
            .get(&reason)
            .copied()
            .unwrap_or(0)
    }

    pub fn batch_flushes_for(&self, reason: BatchFlushReason) -> u64 {
        self.tail()
            .batch_flush_by_reason
            .get(&reason)
            .copied()
            .unwrap_or(0)
    }

    pub fn fallback_counts_by_gpu_parity_issue(&self) -> BTreeMap<GpuParityIssue, u64> {
        let tail = self.tail();
        let mut counts = BTreeMap::new();
        for (reason, count) in &tail.fallback_by_reason {
            if let Some(issue) = reason.gpu_parity_issue() {
                *counts.entry(issue).or_insert(0) += *count;
            }
        }
        counts
    }

    pub fn last_fallback_reason(&self) -> Option<FallbackReason> {
        self.tail().last_fallback_reason
    }

    pub fn last_batch_flush_reason(&self) -> Option<BatchFlushReason> {
        self.tail().last_batch_flush_reason
    }

    pub fn last_batch_wait_ms(&self) -> Option<u64> {
        self.tail().last_batch_wait_ms
    }

    pub fn last_kernel_exec_ms(&self) -> Option<u64> {
        self.tail().last_kernel_exec_ms
    }

    pub fn last_kernel_event_elapsed_us(&self) -> Option<u64> {
        self.tail().last_kernel_event_elapsed_us
    }

    pub fn last_kernel_occupancy_permyriad(&self) -> Option<u16> {
        self.tail().last_kernel_occupancy_permyriad
    }

    pub fn last_pending_batch_len(&self) -> Option<usize> {
        self.tail().last_pending_batch_len
    }

    pub fn avg_batch_wait_ms(&self) -> Option<f64> {
        let samples = self.batch_wait_samples.load(Ordering::Relaxed);
        if samples == 0 {
            return None;
        }
        Some(self.batch_wait_total_ms.load(Ordering::Relaxed) as f64 / samples as f64)
    }

    pub fn avg_kernel_exec_ms(&self) -> Option<f64> {
        let samples = self.kernel_exec_samples.load(Ordering::Relaxed);
        if samples == 0 {
            return None;
        }
        Some(self.kernel_exec_total_ms.load(Ordering::Relaxed) as f64 / samples as f64)
    }

    pub fn avg_kernel_event_elapsed_us(&self) -> Option<f64> {
        let samples = self.kernel_event_timing_samples.load(Ordering::Relaxed);
        if samples == 0 {
            return None;
        }
        Some(self.kernel_event_elapsed_total_us.load(Ordering::Relaxed) as f64 / samples as f64)
    }

    pub fn avg_kernel_occupancy_permyriad(&self) -> Option<f64> {
        let samples = self.kernel_occupancy_samples.load(Ordering::Relaxed);
        if samples == 0 {
            return None;
        }
        Some(
            self.kernel_occupancy_total_permyriad
                .load(Ordering::Relaxed) as f64
                / samples as f64,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_reason_counts() {
        let m = RuntimeMetrics::default();
        assert_eq!(m.last_fallback_reason(), None);

        m.inc_fallback(FallbackReason::GpuUnavailable);
        m.inc_fallback(FallbackReason::GpuUnavailable);
        m.inc_fallback(FallbackReason::NotGpuEligible);

        assert_eq!(m.snapshot().fallback_total, 3);
        assert_eq!(m.fallback_for(FallbackReason::GpuUnavailable), 2);
        assert_eq!(m.fallback_for(FallbackReason::NotGpuEligible), 1);
        assert_eq!(
            m.last_fallback_reason(),
            Some(FallbackReason::NotGpuEligible)
        );
    }

    #[test]
    fn gpu_fallback_runtime_reasons_map_into_metric_fallback_reasons() {
        let m = RuntimeMetrics::default();

        m.inc_gpu_fallback(GpuFallbackReason::Unavailable);
        m.inc_gpu_fallback(GpuFallbackReason::QueueSaturated);
        m.inc_gpu_fallback(GpuFallbackReason::MemoryPressure);

        assert_eq!(m.snapshot().fallback_total, 3);
        assert_eq!(m.fallback_for(FallbackReason::GpuUnavailable), 1);
        assert_eq!(m.fallback_for(FallbackReason::GpuQueueSaturated), 1);
        assert_eq!(m.fallback_for(FallbackReason::GpuMemoryPressure), 1);
        assert_eq!(
            m.last_fallback_reason(),
            Some(FallbackReason::GpuMemoryPressure)
        );
    }

    #[test]
    fn batch_flush_reason_counts() {
        let m = RuntimeMetrics::default();
        assert_eq!(m.last_batch_flush_reason(), None);

        m.inc_batch_flush(BatchFlushReason::Count);
        m.inc_batch_flush(BatchFlushReason::Time);
        m.inc_batch_flush(BatchFlushReason::Time);

        assert_eq!(m.snapshot().batch_flush_count, 3);
        assert_eq!(m.batch_flushes_for(BatchFlushReason::Count), 1);
        assert_eq!(m.batch_flushes_for(BatchFlushReason::Time), 2);
        assert_eq!(m.batch_flushes_for(BatchFlushReason::Admin), 0);
        assert_eq!(m.last_batch_flush_reason(), Some(BatchFlushReason::Time));
    }

    #[test]
    fn batch_wait_observations_track_totals_and_latest() {
        let m = RuntimeMetrics::default();
        assert_eq!(m.last_batch_wait_ms(), None);
        assert_eq!(m.avg_batch_wait_ms(), None);

        m.observe_batch_wait_ms(4);
        m.observe_batch_wait_ms(7);

        assert_eq!(m.snapshot().batch_wait_samples, 2);
        assert_eq!(m.snapshot().batch_wait_total_ms, 11);
        assert_eq!(m.last_batch_wait_ms(), Some(7));
        assert_eq!(m.avg_batch_wait_ms(), Some(5.5));
    }

    #[test]
    fn transfer_byte_counters_accumulate_with_saturation() {
        let m = RuntimeMetrics::default();

        m.observe_h2d_bytes(128);
        m.observe_d2h_bytes(64);

        assert_eq!(m.snapshot().h2d_bytes_total, 128);
        assert_eq!(m.snapshot().d2h_bytes_total, 64);

        m.observe_h2d_bytes(u64::MAX);
        m.observe_d2h_bytes(u64::MAX);

        assert_eq!(m.snapshot().h2d_bytes_total, u64::MAX);
        assert_eq!(m.snapshot().d2h_bytes_total, u64::MAX);
    }

    #[test]
    fn kernel_exec_observations_track_totals_latest_and_average() {
        let m = RuntimeMetrics::default();
        assert_eq!(m.last_kernel_exec_ms(), None);
        assert_eq!(m.avg_kernel_exec_ms(), None);

        m.observe_kernel_exec_ms(9);
        m.observe_kernel_exec_ms(3);

        assert_eq!(m.snapshot().kernel_exec_samples, 2);
        assert_eq!(m.snapshot().kernel_exec_total_ms, 12);
        assert_eq!(m.last_kernel_exec_ms(), Some(3));
        assert_eq!(m.avg_kernel_exec_ms(), Some(6.0));
    }

    #[test]
    fn kernel_event_observations_track_totals_latest_and_average() {
        let m = RuntimeMetrics::default();
        assert_eq!(m.last_kernel_event_elapsed_us(), None);
        assert_eq!(m.avg_kernel_event_elapsed_us(), None);

        m.observe_kernel_event_elapsed_us(150);
        m.observe_kernel_event_elapsed_us(50);

        assert_eq!(m.snapshot().kernel_event_timing_samples, 2);
        assert_eq!(m.snapshot().kernel_event_elapsed_total_us, 200);
        assert_eq!(m.last_kernel_event_elapsed_us(), Some(50));
        assert_eq!(m.avg_kernel_event_elapsed_us(), Some(100.0));
    }

    #[test]
    fn pending_batch_depth_tracks_latest_and_peak() {
        let m = RuntimeMetrics::default();
        assert_eq!(m.last_pending_batch_len(), None);
        assert_eq!(m.snapshot().pending_batch_peak, 0);

        m.observe_pending_batch_len(1);
        m.observe_pending_batch_len(3);
        m.observe_pending_batch_len(2);
        m.observe_pending_batch_len(0);

        assert_eq!(m.snapshot().pending_batch_peak, 3);
        assert_eq!(m.last_pending_batch_len(), Some(0));
    }

    #[test]
    fn kernel_occupancy_observations_track_totals_latest_and_average() {
        let m = RuntimeMetrics::default();
        assert_eq!(m.last_kernel_occupancy_permyriad(), None);
        assert_eq!(m.avg_kernel_occupancy_permyriad(), None);

        m.observe_kernel_occupancy_permyriad(6_250);
        m.observe_kernel_occupancy_permyriad(7_500);

        assert_eq!(m.snapshot().kernel_occupancy_samples, 2);
        assert_eq!(m.snapshot().kernel_occupancy_total_permyriad, 13_750);
        assert_eq!(m.last_kernel_occupancy_permyriad(), Some(7_500));
        assert_eq!(m.avg_kernel_occupancy_permyriad(), Some(6_875.0));
    }

    #[test]
    fn gpu_fallback_reasons_have_tracked_parity_issues() {
        assert_eq!(FallbackReason::NotGpuEligible.gpu_parity_issue(), None);

        let unavailable = FallbackReason::GpuUnavailable
            .gpu_parity_issue()
            .expect("gpu unavailable should map to a parity issue");
        assert_eq!(unavailable.id, "GPU-120");
        assert_eq!(unavailable.owner, "runtime");
        assert_eq!(unavailable.milestone, "m0-bootstrap");

        let queue = FallbackReason::GpuQueueSaturated
            .gpu_parity_issue()
            .expect("gpu queue saturation should map to a parity issue");
        assert_eq!(queue.id, "GPU-121");

        let memory = FallbackReason::GpuMemoryPressure
            .gpu_parity_issue()
            .expect("gpu memory pressure should map to a parity issue");
        assert_eq!(memory.id, "GPU-122");
    }

    #[test]
    fn fallback_counts_can_be_aggregated_by_parity_issue() {
        let m = RuntimeMetrics::default();
        m.inc_fallback(FallbackReason::NotGpuEligible);
        m.inc_fallback(FallbackReason::GpuUnavailable);
        m.inc_fallback(FallbackReason::GpuUnavailable);
        m.inc_fallback(FallbackReason::GpuQueueSaturated);

        let by_issue = m.fallback_counts_by_gpu_parity_issue();
        assert_eq!(by_issue.len(), 2);

        assert_eq!(
            by_issue.get(&GpuParityIssue {
                id: "GPU-120",
                owner: "runtime",
                milestone: "m0-bootstrap"
            }),
            Some(&2)
        );
        assert_eq!(
            by_issue.get(&GpuParityIssue {
                id: "GPU-121",
                owner: "runtime",
                milestone: "m0-bootstrap"
            }),
            Some(&1)
        );
    }

    #[test]
    fn snapshot_captures_counters_and_last_observations() {
        let m = RuntimeMetrics::default();
        m.inc_commit();
        m.inc_batch_flush(BatchFlushReason::Admin);
        m.inc_gpu_fallback(GpuFallbackReason::QueueSaturated);
        m.observe_batch_wait_ms(8);
        m.observe_h2d_bytes(512);
        m.observe_d2h_bytes(64);
        m.observe_kernel_exec_ms(3);
        m.observe_kernel_occupancy_permyriad(8_750);
        m.observe_pending_batch_len(5);

        let snapshot = m.snapshot();

        assert_eq!(snapshot.commits_total, 1);
        assert_eq!(snapshot.batch_flush_count, 1);
        assert_eq!(snapshot.fallback_total, 1);
        assert_eq!(snapshot.batch_wait_samples, 1);
        assert_eq!(snapshot.batch_wait_total_ms, 8);
        assert_eq!(snapshot.h2d_bytes_total, 512);
        assert_eq!(snapshot.d2h_bytes_total, 64);
        assert_eq!(snapshot.kernel_exec_samples, 1);
        assert_eq!(snapshot.kernel_exec_total_ms, 3);
        assert_eq!(snapshot.kernel_occupancy_samples, 1);
        assert_eq!(snapshot.kernel_occupancy_total_permyriad, 8_750);
        assert_eq!(snapshot.pending_batch_peak, 5);
        assert_eq!(
            snapshot
                .fallback_by_reason
                .get(&FallbackReason::GpuQueueSaturated),
            Some(&1)
        );
        assert_eq!(
            snapshot.batch_flush_by_reason.get(&BatchFlushReason::Admin),
            Some(&1)
        );
        assert_eq!(
            snapshot.last_fallback_reason,
            Some(FallbackReason::GpuQueueSaturated)
        );
        assert_eq!(
            snapshot.last_batch_flush_reason,
            Some(BatchFlushReason::Admin)
        );
        assert_eq!(snapshot.last_batch_wait_ms, Some(8));
        assert_eq!(snapshot.last_kernel_exec_ms, Some(3));
        assert_eq!(snapshot.last_kernel_occupancy_permyriad, Some(8_750));
        assert_eq!(snapshot.last_pending_batch_len, Some(5));
    }
}
