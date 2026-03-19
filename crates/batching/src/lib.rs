use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushReason {
    Count,
    Time,
    Admin,
}

#[derive(Debug, Clone)]
pub struct BatchItem<T> {
    pub item: T,
    pub enqueued_at: Instant,
}

#[derive(Debug)]
pub struct Batch<T> {
    pub items: Vec<BatchItem<T>>,
    pub reason: FlushReason,
}

#[derive(Debug)]
pub struct DualTriggerBatcher<T> {
    max_items: usize,
    max_wait: Duration,
    queue: Vec<BatchItem<T>>,
    first_enqueued_at: Option<Instant>,
}

impl<T> DualTriggerBatcher<T> {
    pub fn new(max_items: usize, max_wait: Duration) -> Self {
        assert!(max_items > 0, "max_items must be > 0");
        Self {
            max_items,
            max_wait,
            queue: Vec::new(),
            first_enqueued_at: None,
        }
    }

    pub fn enqueue(&mut self, item: T, now: Instant) -> Option<Batch<T>> {
        if self.first_enqueued_at.is_none() {
            self.first_enqueued_at = Some(now);
        }
        self.queue.push(BatchItem {
            item,
            enqueued_at: now,
        });

        if self.queue.len() >= self.max_items {
            return self.flush(FlushReason::Count);
        }

        None
    }

    pub fn maybe_flush_due_to_time(&mut self, now: Instant) -> Option<Batch<T>> {
        let first = self.first_enqueued_at?;
        if now.saturating_duration_since(first) >= self.max_wait && !self.queue.is_empty() {
            return self.flush(FlushReason::Time);
        }
        None
    }

    pub fn flush_admin(&mut self) -> Option<Batch<T>> {
        self.flush(FlushReason::Admin)
    }

    pub fn requeue_front(&mut self, mut items: Vec<BatchItem<T>>) {
        if items.is_empty() {
            return;
        }

        let existing = std::mem::take(&mut self.queue);
        items.extend(existing);
        self.first_enqueued_at = items.first().map(|i| i.enqueued_at);
        self.queue = items;
    }

    fn flush(&mut self, reason: FlushReason) -> Option<Batch<T>> {
        if self.queue.is_empty() {
            return None;
        }
        let items = std::mem::take(&mut self.queue);
        self.first_enqueued_at = None;
        Some(Batch { items, reason })
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn first_enqueued_at(&self) -> Option<Instant> {
        self.first_enqueued_at
    }

    pub fn max_items(&self) -> usize {
        self.max_items
    }

    pub fn max_wait(&self) -> Duration {
        self.max_wait
    }

    pub fn time_until_flush_deadline(&self, now: Instant) -> Option<Duration> {
        let first = self.first_enqueued_at?;
        let elapsed = now.saturating_duration_since(first);
        Some(self.max_wait.saturating_sub(elapsed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flushes_on_count() {
        let mut b = DualTriggerBatcher::new(2, Duration::from_millis(100));
        let t0 = Instant::now();
        assert!(b.enqueue(1u8, t0).is_none());
        let batch = b.enqueue(2u8, t0).expect("must flush on count");
        assert_eq!(batch.items.len(), 2);
        assert_eq!(batch.reason, FlushReason::Count);
    }

    #[test]
    fn flushes_on_time() {
        let mut b = DualTriggerBatcher::new(10, Duration::from_millis(10));
        let t0 = Instant::now();
        assert!(b.enqueue(1u8, t0).is_none());
        let t1 = t0 + Duration::from_millis(11);
        let batch = b.maybe_flush_due_to_time(t1).expect("must flush on time");
        assert_eq!(batch.items.len(), 1);
        assert_eq!(batch.reason, FlushReason::Time);
    }

    #[test]
    fn first_enqueued_at_tracks_head_and_resets_after_flush() {
        let mut b = DualTriggerBatcher::new(2, Duration::from_millis(100));
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_millis(1);

        assert_eq!(b.first_enqueued_at(), None);
        assert!(b.enqueue(1u8, t0).is_none());
        assert_eq!(b.first_enqueued_at(), Some(t0));

        let _ = b.enqueue(2u8, t1).expect("must flush on count");
        assert_eq!(b.first_enqueued_at(), None);
    }

    #[test]
    fn requeue_front_prepends_items_and_tracks_oldest_enqueue_time() {
        let mut b = DualTriggerBatcher::new(10, Duration::from_secs(1));
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_millis(1);
        let t2 = t0 + Duration::from_millis(2);

        assert!(b.enqueue(3u8, t2).is_none());
        b.requeue_front(vec![
            BatchItem {
                item: 1u8,
                enqueued_at: t0,
            },
            BatchItem {
                item: 2u8,
                enqueued_at: t1,
            },
        ]);

        assert_eq!(b.first_enqueued_at(), Some(t0));
        let batch = b.flush_admin().expect("batch should flush");
        let items: Vec<u8> = batch.items.into_iter().map(|it| it.item).collect();
        assert_eq!(items, vec![1, 2, 3]);
    }

    #[test]
    fn config_accessors_report_dual_trigger_thresholds() {
        let b = DualTriggerBatcher::<u8>::new(7, Duration::from_millis(42));
        assert_eq!(b.max_items(), 7);
        assert_eq!(b.max_wait(), Duration::from_millis(42));
    }

    #[test]
    fn time_until_flush_deadline_counts_down_and_saturates_at_zero() {
        let mut b = DualTriggerBatcher::new(10, Duration::from_millis(10));
        let t0 = Instant::now();
        assert_eq!(b.time_until_flush_deadline(t0), None);

        assert!(b.enqueue(1u8, t0).is_none());
        assert_eq!(
            b.time_until_flush_deadline(t0),
            Some(Duration::from_millis(10))
        );
        assert_eq!(
            b.time_until_flush_deadline(t0 + Duration::from_millis(4)),
            Some(Duration::from_millis(6))
        );
        assert_eq!(
            b.time_until_flush_deadline(t0 + Duration::from_millis(12)),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn maybe_flush_due_to_time_does_not_panic_when_clock_moves_backwards() {
        let mut b = DualTriggerBatcher::new(10, Duration::from_millis(10));
        let t0 = Instant::now();
        assert!(b.enqueue(1u8, t0).is_none());

        // Use an earlier instant to simulate non-monotonic caller-provided timing.
        let earlier = t0.checked_sub(Duration::from_millis(1)).unwrap_or(t0);
        assert!(b.maybe_flush_due_to_time(earlier).is_none());
        assert_eq!(b.len(), 1);
    }
}
