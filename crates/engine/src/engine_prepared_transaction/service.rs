//! Prepared-transaction foreground service admission.
//!
//! W1/T8/T32 are latency envelopes, not caller-selected labels. After exact route/resource proof
//! derives one of those classes, this controller reserves a class-specific FIFO slot, operation
//! population, and byte population before BEGIN. A timed-out waiter is rejected pre-effect. The
//! permit remains charged through durable/apply/publication completion. General transactions keep
//! their PostgreSQL-compatible unbounded operation-count surface and cannot consume these reserved
//! low-latency populations.

use super::*;

const FAST_CLASS_COUNT: usize = 3;

#[derive(Debug)]
pub(crate) struct PreparedTransactionServiceController {
    state: Mutex<PreparedServiceState>,
    cv: std::sync::Condvar,
    rejected: [AtomicU64; FAST_CLASS_COUNT],
}

#[derive(Debug, Default)]
struct PreparedServiceState {
    lanes: [PreparedServiceLane; FAST_CLASS_COUNT],
}

#[derive(Debug, Default)]
struct PreparedServiceLane {
    next_ticket: u64,
    serving_ticket: u64,
    cancelled: BTreeSet<u64>,
    transactions: u32,
    operations: u32,
    bytes: u64,
}

#[derive(Clone, Copy)]
struct PreparedServiceProfile {
    max_transactions: u32,
    max_operations: u32,
    max_bytes: u64,
    queue_budget: Duration,
}

#[derive(Debug)]
pub(super) struct PreparedTransactionServicePermit<'a> {
    controller: &'a PreparedTransactionServiceController,
    lane: usize,
    operations: u32,
    bytes: u64,
}

impl Default for PreparedTransactionServiceController {
    fn default() -> Self {
        Self {
            state: Mutex::new(PreparedServiceState::default()),
            cv: std::sync::Condvar::new(),
            rejected: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl PreparedTransactionServiceController {
    pub(super) fn admit(
        &self,
        class: TransactionClass,
        resources: TransactionResources,
    ) -> Result<Option<PreparedTransactionServicePermit<'_>>, ExecuteError> {
        let Some((lane_index, profile)) = service_profile(class) else {
            return Ok(None);
        };
        let bytes = resources
            .post_image_and_wal_bytes
            .checked_add(resources.result_bytes)
            .ok_or_else(|| {
                ExecuteError::ResourceExhausted(format!(
                    "{class:?} prepared transaction byte reservation overflowed"
                ))
            })?;
        if resources.operations > profile.max_operations || bytes > profile.max_bytes {
            return Err(ExecuteError::ResourceExhausted(format!(
                "{class:?} prepared transaction exceeds its foreground stage-credit capacity"
            )));
        }

        let started = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let ticket = state.lanes[lane_index].next_ticket;
        state.lanes[lane_index].next_ticket = ticket.saturating_add(1);
        loop {
            let lane = &state.lanes[lane_index];
            let has_turn = ticket == lane.serving_ticket;
            let has_credits = lane.transactions < profile.max_transactions
                && lane.operations.saturating_add(resources.operations) <= profile.max_operations
                && lane.bytes.saturating_add(bytes) <= profile.max_bytes;
            if has_turn && has_credits {
                let lane = &mut state.lanes[lane_index];
                lane.transactions += 1;
                lane.operations += resources.operations;
                lane.bytes += bytes;
                advance_serving_ticket(lane);
                self.cv.notify_all();
                return Ok(Some(PreparedTransactionServicePermit {
                    controller: self,
                    lane: lane_index,
                    operations: resources.operations,
                    bytes,
                }));
            }

            let elapsed = started.elapsed();
            let Some(remaining) = profile.queue_budget.checked_sub(elapsed) else {
                cancel_ticket(&mut state.lanes[lane_index], ticket);
                self.rejected[lane_index].fetch_add(1, AtomicOrdering::Relaxed);
                self.cv.notify_all();
                return Err(ExecuteError::ResourceExhausted(format!(
                    "{class:?} prepared transaction could not reserve foreground credits within {} microseconds",
                    profile.queue_budget.as_micros()
                )));
            };
            let waited = self
                .cv
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = waited.0;
            if waited.1.timed_out() {
                cancel_ticket(&mut state.lanes[lane_index], ticket);
                self.rejected[lane_index].fetch_add(1, AtomicOrdering::Relaxed);
                self.cv.notify_all();
                return Err(ExecuteError::ResourceExhausted(format!(
                    "{class:?} prepared transaction could not reserve foreground credits within {} microseconds",
                    profile.queue_budget.as_micros()
                )));
            }
        }
    }
}

impl Drop for PreparedTransactionServicePermit<'_> {
    fn drop(&mut self) {
        let mut state = self
            .controller
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let lane = &mut state.lanes[self.lane];
        lane.transactions = lane
            .transactions
            .checked_sub(1)
            .expect("prepared service transaction credits remain paired");
        lane.operations = lane
            .operations
            .checked_sub(self.operations)
            .expect("prepared service operation credits remain paired");
        lane.bytes = lane
            .bytes
            .checked_sub(self.bytes)
            .expect("prepared service byte credits remain paired");
        drop(state);
        self.controller.cv.notify_all();
    }
}

fn service_profile(class: TransactionClass) -> Option<(usize, PreparedServiceProfile)> {
    // Queue time receives at most one quarter of the class p99 envelope, leaving a conservative
    // three-quarter downstream budget. BENCH-001 still qualifies the measured end-to-end p50,
    // p99, and p99.9; these constants are hard stage bounds, not performance claims.
    match class {
        TransactionClass::W1 => Some((
            0,
            PreparedServiceProfile {
                max_transactions: 4_096,
                max_operations: 4_096,
                max_bytes: 4 * 1024 * 1024,
                queue_budget: Duration::from_micros(375),
            },
        )),
        TransactionClass::T8 => Some((
            1,
            PreparedServiceProfile {
                max_transactions: 512,
                max_operations: 4_096,
                max_bytes: 8 * 1024 * 1024,
                queue_budget: Duration::from_micros(750),
            },
        )),
        TransactionClass::T32 => Some((
            2,
            PreparedServiceProfile {
                max_transactions: 128,
                max_operations: 4_096,
                max_bytes: 8 * 1024 * 1024,
                queue_budget: Duration::from_micros(1_500),
            },
        )),
        TransactionClass::General => None,
    }
}

fn cancel_ticket(lane: &mut PreparedServiceLane, ticket: u64) {
    if ticket == lane.serving_ticket {
        advance_serving_ticket(lane);
    } else {
        lane.cancelled.insert(ticket);
    }
}

fn advance_serving_ticket(lane: &mut PreparedServiceLane) {
    lane.serving_ticket = lane.serving_ticket.saturating_add(1);
    while lane.cancelled.remove(&lane.serving_ticket) {
        lane.serving_ticket = lane.serving_ticket.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resources(operations: u32, bytes: u64) -> TransactionResources {
        TransactionResources {
            operations,
            mutations: operations.min(1),
            post_image_and_wal_bytes: bytes,
            maintained_index_fanout: 0,
            touched_tables: 1,
            cold_accesses: 0,
            result_bytes: 0,
        }
    }

    #[test]
    fn general_work_does_not_consume_reserved_latency_class_credits() {
        let controller = PreparedTransactionServiceController::default();
        assert!(controller
            .admit(TransactionClass::General, resources(100_000, u64::MAX))
            .unwrap()
            .is_none());
        let state = controller.state.lock().unwrap();
        assert!(state.lanes.iter().all(|lane| lane.transactions == 0));
    }

    #[test]
    fn permit_charges_and_releases_exact_class_population() {
        let controller = PreparedTransactionServiceController::default();
        let permit = controller
            .admit(TransactionClass::T8, resources(8, 4_096))
            .unwrap()
            .unwrap();
        {
            let state = controller.state.lock().unwrap();
            assert_eq!(state.lanes[1].transactions, 1);
            assert_eq!(state.lanes[1].operations, 8);
            assert_eq!(state.lanes[1].bytes, 4_096);
            assert_eq!(state.lanes[0].transactions, 0);
            assert_eq!(state.lanes[2].transactions, 0);
        }
        drop(permit);
        let state = controller.state.lock().unwrap();
        assert_eq!(state.lanes[1].transactions, 0);
        assert_eq!(state.lanes[1].operations, 0);
        assert_eq!(state.lanes[1].bytes, 0);
    }

    #[test]
    fn oversized_fast_work_rejects_without_consuming_a_ticket() {
        let controller = PreparedTransactionServiceController::default();
        let error = controller
            .admit(TransactionClass::W1, resources(1, 4 * 1024 * 1024 + 1))
            .expect_err("oversized reservation must fail");
        assert!(matches!(error, ExecuteError::ResourceExhausted(_)));
        let state = controller.state.lock().unwrap();
        assert_eq!(state.lanes[0].next_ticket, 0);
        assert_eq!(state.lanes[0].transactions, 0);
    }

    #[test]
    fn saturated_class_times_out_and_advances_its_fifo_without_leaking_credits() {
        let controller = PreparedTransactionServiceController::default();
        {
            let mut state = controller.state.lock().unwrap();
            state.lanes[0].transactions = 4_096;
            state.lanes[0].operations = 4_096;
        }
        let error = controller
            .admit(TransactionClass::W1, resources(1, 128))
            .expect_err("saturated W1 admission must honor its queue deadline");
        assert!(matches!(error, ExecuteError::ResourceExhausted(_)));
        assert_eq!(controller.rejected[0].load(AtomicOrdering::Relaxed), 1);
        {
            let mut state = controller.state.lock().unwrap();
            assert_eq!(state.lanes[0].next_ticket, 1);
            assert_eq!(state.lanes[0].serving_ticket, 1);
            state.lanes[0].transactions = 0;
            state.lanes[0].operations = 0;
        }
        let permit = controller
            .admit(TransactionClass::W1, resources(1, 128))
            .unwrap()
            .unwrap();
        drop(permit);
        let state = controller.state.lock().unwrap();
        assert_eq!(state.lanes[0].transactions, 0);
        assert_eq!(state.lanes[0].operations, 0);
        assert_eq!(state.lanes[0].bytes, 0);
    }
}
