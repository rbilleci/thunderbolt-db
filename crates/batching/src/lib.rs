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
        if now.duration_since(first) >= self.max_wait && !self.queue.is_empty() {
            return self.flush(FlushReason::Time);
        }
        None
    }

    pub fn flush_admin(&mut self) -> Option<Batch<T>> {
        self.flush(FlushReason::Admin)
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
}
