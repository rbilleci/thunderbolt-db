use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FallbackReason {
    NotGpuEligible,
    GpuUnavailable,
    GpuQueueSaturated,
    GpuMemoryPressure,
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
    fallback_by_reason: BTreeMap<FallbackReason, u64>,
    batch_flush_by_reason: BTreeMap<BatchFlushReason, u64>,
    last_fallback_reason: Option<FallbackReason>,
    last_batch_flush_reason: Option<BatchFlushReason>,
    last_batch_wait_ms: Option<u64>,
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

    pub fn fallback_for(&self, reason: FallbackReason) -> u64 {
        self.fallback_by_reason.get(&reason).copied().unwrap_or(0)
    }

    pub fn batch_flushes_for(&self, reason: BatchFlushReason) -> u64 {
        self.batch_flush_by_reason
            .get(&reason)
            .copied()
            .unwrap_or(0)
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

        m.observe_batch_wait_ms(4);
        m.observe_batch_wait_ms(7);

        assert_eq!(m.batch_wait_samples, 2);
        assert_eq!(m.batch_wait_total_ms, 11);
        assert_eq!(m.last_batch_wait_ms(), Some(7));
    }
}
