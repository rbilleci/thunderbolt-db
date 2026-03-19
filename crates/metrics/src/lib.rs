use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FallbackReason {
    NotGpuEligible,
    GpuUnavailable,
    GpuQueueSaturated,
    GpuMemoryPressure,
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

#[derive(Debug, Default)]
pub struct RuntimeMetrics {
    pub commits_total: u64,
    pub batch_flush_count: u64,
    pub fallback_total: u64,
    pub batch_wait_samples: u64,
    pub batch_wait_total_ms: u64,
    pub h2d_bytes_total: u64,
    pub d2h_bytes_total: u64,
    pub kernel_exec_samples: u64,
    pub kernel_exec_total_ms: u64,
    pub pending_batch_peak: usize,
    fallback_by_reason: BTreeMap<FallbackReason, u64>,
    batch_flush_by_reason: BTreeMap<BatchFlushReason, u64>,
    last_fallback_reason: Option<FallbackReason>,
    last_batch_flush_reason: Option<BatchFlushReason>,
    last_batch_wait_ms: Option<u64>,
    last_kernel_exec_ms: Option<u64>,
    last_pending_batch_len: Option<usize>,
}

impl RuntimeMetrics {
    pub fn inc_commit(&mut self) {
        self.commits_total += 1;
    }

    pub fn observe_batch_wait_ms(&mut self, wait_ms: u64) {
        self.batch_wait_samples += 1;
        self.batch_wait_total_ms = self.batch_wait_total_ms.saturating_add(wait_ms);
        self.last_batch_wait_ms = Some(wait_ms);
    }

    pub fn inc_batch_flush(&mut self, reason: BatchFlushReason) {
        self.batch_flush_count += 1;
        *self.batch_flush_by_reason.entry(reason).or_insert(0) += 1;
        self.last_batch_flush_reason = Some(reason);
    }

    pub fn inc_fallback(&mut self, reason: FallbackReason) {
        self.fallback_total += 1;
        *self.fallback_by_reason.entry(reason).or_insert(0) += 1;
        self.last_fallback_reason = Some(reason);
    }

    pub fn observe_h2d_bytes(&mut self, bytes: u64) {
        self.h2d_bytes_total = self.h2d_bytes_total.saturating_add(bytes);
    }

    pub fn observe_d2h_bytes(&mut self, bytes: u64) {
        self.d2h_bytes_total = self.d2h_bytes_total.saturating_add(bytes);
    }

    pub fn observe_kernel_exec_ms(&mut self, exec_ms: u64) {
        self.kernel_exec_samples += 1;
        self.kernel_exec_total_ms = self.kernel_exec_total_ms.saturating_add(exec_ms);
        self.last_kernel_exec_ms = Some(exec_ms);
    }

    pub fn observe_pending_batch_len(&mut self, len: usize) {
        self.pending_batch_peak = self.pending_batch_peak.max(len);
        self.last_pending_batch_len = Some(len);
    }

    pub fn fallback_for(&self, reason: FallbackReason) -> u64 {
        self.fallback_by_reason.get(&reason).copied().unwrap_or(0)
    }

    pub fn batch_flushes_for(&self, reason: BatchFlushReason) -> u64 {
        self.batch_flush_by_reason
            .get(&reason)
            .copied()
            .unwrap_or(0)
    }

    pub fn fallback_counts_by_gpu_parity_issue(&self) -> BTreeMap<GpuParityIssue, u64> {
        let mut counts = BTreeMap::new();
        for (reason, count) in &self.fallback_by_reason {
            if let Some(issue) = reason.gpu_parity_issue() {
                *counts.entry(issue).or_insert(0) += *count;
            }
        }
        counts
    }

    pub fn last_fallback_reason(&self) -> Option<FallbackReason> {
        self.last_fallback_reason
    }

    pub fn last_batch_flush_reason(&self) -> Option<BatchFlushReason> {
        self.last_batch_flush_reason
    }

    pub fn last_batch_wait_ms(&self) -> Option<u64> {
        self.last_batch_wait_ms
    }

    pub fn last_kernel_exec_ms(&self) -> Option<u64> {
        self.last_kernel_exec_ms
    }

    pub fn last_pending_batch_len(&self) -> Option<usize> {
        self.last_pending_batch_len
    }

    pub fn avg_batch_wait_ms(&self) -> Option<f64> {
        if self.batch_wait_samples == 0 {
            return None;
        }
        Some(self.batch_wait_total_ms as f64 / self.batch_wait_samples as f64)
    }

    pub fn avg_kernel_exec_ms(&self) -> Option<f64> {
        if self.kernel_exec_samples == 0 {
            return None;
        }
        Some(self.kernel_exec_total_ms as f64 / self.kernel_exec_samples as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_reason_counts() {
        let mut m = RuntimeMetrics::default();
        assert_eq!(m.last_fallback_reason(), None);

        m.inc_fallback(FallbackReason::GpuUnavailable);
        m.inc_fallback(FallbackReason::GpuUnavailable);
        m.inc_fallback(FallbackReason::NotGpuEligible);

        assert_eq!(m.fallback_total, 3);
        assert_eq!(m.fallback_for(FallbackReason::GpuUnavailable), 2);
        assert_eq!(m.fallback_for(FallbackReason::NotGpuEligible), 1);
        assert_eq!(
            m.last_fallback_reason(),
            Some(FallbackReason::NotGpuEligible)
        );
    }

    #[test]
    fn batch_flush_reason_counts() {
        let mut m = RuntimeMetrics::default();
        assert_eq!(m.last_batch_flush_reason(), None);

        m.inc_batch_flush(BatchFlushReason::Count);
        m.inc_batch_flush(BatchFlushReason::Time);
        m.inc_batch_flush(BatchFlushReason::Time);

        assert_eq!(m.batch_flush_count, 3);
        assert_eq!(m.batch_flushes_for(BatchFlushReason::Count), 1);
        assert_eq!(m.batch_flushes_for(BatchFlushReason::Time), 2);
        assert_eq!(m.batch_flushes_for(BatchFlushReason::Admin), 0);
        assert_eq!(m.last_batch_flush_reason(), Some(BatchFlushReason::Time));
    }

    #[test]
    fn batch_wait_observations_track_totals_and_latest() {
        let mut m = RuntimeMetrics::default();
        assert_eq!(m.last_batch_wait_ms(), None);
        assert_eq!(m.avg_batch_wait_ms(), None);

        m.observe_batch_wait_ms(4);
        m.observe_batch_wait_ms(7);

        assert_eq!(m.batch_wait_samples, 2);
        assert_eq!(m.batch_wait_total_ms, 11);
        assert_eq!(m.last_batch_wait_ms(), Some(7));
        assert_eq!(m.avg_batch_wait_ms(), Some(5.5));
    }

    #[test]
    fn transfer_byte_counters_accumulate_with_saturation() {
        let mut m = RuntimeMetrics::default();

        m.observe_h2d_bytes(128);
        m.observe_d2h_bytes(64);

        assert_eq!(m.h2d_bytes_total, 128);
        assert_eq!(m.d2h_bytes_total, 64);

        m.observe_h2d_bytes(u64::MAX);
        m.observe_d2h_bytes(u64::MAX);

        assert_eq!(m.h2d_bytes_total, u64::MAX);
        assert_eq!(m.d2h_bytes_total, u64::MAX);
    }

    #[test]
    fn kernel_exec_observations_track_totals_latest_and_average() {
        let mut m = RuntimeMetrics::default();
        assert_eq!(m.last_kernel_exec_ms(), None);
        assert_eq!(m.avg_kernel_exec_ms(), None);

        m.observe_kernel_exec_ms(9);
        m.observe_kernel_exec_ms(3);

        assert_eq!(m.kernel_exec_samples, 2);
        assert_eq!(m.kernel_exec_total_ms, 12);
        assert_eq!(m.last_kernel_exec_ms(), Some(3));
        assert_eq!(m.avg_kernel_exec_ms(), Some(6.0));
    }

    #[test]
    fn pending_batch_depth_tracks_latest_and_peak() {
        let mut m = RuntimeMetrics::default();
        assert_eq!(m.last_pending_batch_len(), None);
        assert_eq!(m.pending_batch_peak, 0);

        m.observe_pending_batch_len(1);
        m.observe_pending_batch_len(3);
        m.observe_pending_batch_len(2);
        m.observe_pending_batch_len(0);

        assert_eq!(m.pending_batch_peak, 3);
        assert_eq!(m.last_pending_batch_len(), Some(0));
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
        let mut m = RuntimeMetrics::default();
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
}
