use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FallbackReason {
    NotGpuEligible,
    GpuUnavailable,
    GpuQueueSaturated,
    GpuMemoryPressure,
}

#[derive(Debug, Default)]
pub struct RuntimeMetrics {
    pub commits_total: u64,
    pub batch_flush_count: u64,
    pub fallback_total: u64,
    fallback_by_reason: BTreeMap<FallbackReason, u64>,
}

impl RuntimeMetrics {
    pub fn inc_commit(&mut self) {
        self.commits_total += 1;
    }

    pub fn inc_batch_flush(&mut self) {
        self.batch_flush_count += 1;
    }

    pub fn inc_fallback(&mut self, reason: FallbackReason) {
        self.fallback_total += 1;
        *self.fallback_by_reason.entry(reason).or_insert(0) += 1;
    }

    pub fn fallback_for(&self, reason: FallbackReason) -> u64 {
        self.fallback_by_reason.get(&reason).copied().unwrap_or(0)
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
}
