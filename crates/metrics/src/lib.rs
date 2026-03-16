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
    fallback_by_reason: BTreeMap<FallbackReason, u64>,
    batch_flush_by_reason: BTreeMap<BatchFlushReason, u64>,
}

impl RuntimeMetrics {
    pub fn inc_commit(&mut self) {
        self.commits_total += 1;
    }

    pub fn inc_batch_flush(&mut self, reason: BatchFlushReason) {
        self.batch_flush_count += 1;
        *self.batch_flush_by_reason.entry(reason).or_insert(0) += 1;
    }

    pub fn inc_fallback(&mut self, reason: FallbackReason) {
        self.fallback_total += 1;
        *self.fallback_by_reason.entry(reason).or_insert(0) += 1;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_reason_counts() {
        let mut m = RuntimeMetrics::default();
        m.inc_fallback(FallbackReason::GpuUnavailable);
        m.inc_fallback(FallbackReason::GpuUnavailable);
        m.inc_fallback(FallbackReason::NotGpuEligible);
        assert_eq!(m.fallback_total, 3);
        assert_eq!(m.fallback_for(FallbackReason::GpuUnavailable), 2);
        assert_eq!(m.fallback_for(FallbackReason::NotGpuEligible), 1);
    }

    #[test]
    fn batch_flush_reason_counts() {
        let mut m = RuntimeMetrics::default();
        m.inc_batch_flush(BatchFlushReason::Count);
        m.inc_batch_flush(BatchFlushReason::Time);
        m.inc_batch_flush(BatchFlushReason::Time);

        assert_eq!(m.batch_flush_count, 3);
        assert_eq!(m.batch_flushes_for(BatchFlushReason::Count), 1);
        assert_eq!(m.batch_flushes_for(BatchFlushReason::Time), 2);
        assert_eq!(m.batch_flushes_for(BatchFlushReason::Admin), 0);
    }
}
