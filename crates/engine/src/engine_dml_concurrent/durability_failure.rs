//! Post-WAL durability-failure ownership for the concurrent commit path.
//!
//! The serial WAL supplies a copyable [`DurabilityFault`].  Once a typed INSERT has crossed WAL,
//! this module carries that exact value through every queue, tail, and waiter handoff.  The fixed
//! drain deliberately performs no formatting or collection: allocation is incompatible with a
//! failure path that must settle already-reserved work before it wakes another committer.

#[cfg(test)]
use super::state::wave_tail_failure_publish_hook;
use super::{AtomicOrdering, Engine, EngineError, ExecuteError};
use gpu_db_types::DurabilityFault;

/// The sole sticky failure carrier for the classic wave queue and group coordinator.
///
/// Compatibility callers retain their established text diagnostics.  The fixed serial branch is
/// copy-only so a concrete WAL fault cannot be replaced by a reconstructed generic error.
#[derive(Debug, Clone)]
pub(crate) enum CommitPathFailure {
    Fixed(DurabilityFault),
    Compatibility(String),
}

impl CommitPathFailure {
    pub(super) fn compatibility(message: String) -> Self {
        Self::Compatibility(message)
    }

    pub(super) fn fixed_fault(&self) -> Option<DurabilityFault> {
        match self {
            Self::Fixed(fault) => Some(*fault),
            Self::Compatibility(_) => None,
        }
    }

    pub(super) fn outcome_error(&self) -> ExecuteError {
        match self {
            Self::Fixed(fault) => ExecuteError::IndeterminateDurability(*fault),
            Self::Compatibility(message) => ExecuteError::Engine(EngineError::Durability(format!(
                "the concurrent commit path is wedged pending restart recovery: {message}"
            ))),
        }
    }

    pub(super) fn engine_error(&self) -> EngineError {
        match self {
            Self::Fixed(fault) => EngineError::DurabilityFault(*fault),
            Self::Compatibility(message) => EngineError::Durability(format!(
                "group-commit durability failed; the commit path is wedged pending restart \
                 recovery: {message}"
            )),
        }
    }
}

/// Preserve a concrete post-WAL durability result at the execution API boundary.
pub(super) fn execute_error_from_engine(error: EngineError) -> ExecuteError {
    ExecuteError::from_post_wal_engine(error)
}

impl Engine {
    /// Compatibility drain retained for pre-existing generic failure paths.  It intentionally
    /// retains their allocation-backed diagnostics; fixed serial faults use the sibling method.
    pub(crate) fn fail_all_pending_commit_work(&self, reason: &str) {
        let failure = CommitPathFailure::compatibility(reason.to_string());
        let stranded = {
            let mut queue = self.lock_commit_wave_queue();
            queue.wedged.get_or_insert_with(|| failure.clone());
            queue.sequencer_active = false;
            queue.items.drain(..).collect::<Vec<_>>()
        };
        for item in stranded {
            item.set_outcome(Err(failure.outcome_error()));
        }
        let pending_tails = {
            let mut tails = self
                .commit_wave
                .pending_tails
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            tails.drain(..).collect::<Vec<_>>()
        };
        let pending_tail_count = pending_tails.len() as u64;
        drop(pending_tails);
        if pending_tail_count != 0 {
            self.commit_wave
                .tails_finished
                .fetch_add(pending_tail_count, AtomicOrdering::Release);
        }

        if let Some(lanes) = &self.intent_lanes {
            let mut intents = Vec::new();
            for queue in &lanes.queues {
                intents.extend(
                    queue
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .drain(..),
                );
            }
            intents.extend(
                lanes
                    .resize_hold
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .drain(..),
            );
            for request in lanes
                .validate_queue
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .drain(..)
            {
                *request
                    .slot
                    .result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(None);
                request.slot.done.store(true, AtomicOrdering::Release);
            }
            for item in intents {
                item.set_outcome(Err(failure.outcome_error()));
            }
        }
        self.commit_wave.cv.notify_all();
    }

    /// Settle every outstanding post-WAL owner before notifying waiters.  The exact WAL fault is
    /// copied unchanged to every recipient before the final condition-variable notification.
    pub(super) fn fail_all_pending_commit_work_fixed(
        &self,
        fault: DurabilityFault,
    ) -> DurabilityFault {
        // BEGIN FIXED SERIAL DURABILITY DRAIN CODE
        let fault = self.group_flush.fixed_poison.install(fault);
        self.commit_path_wedged.store(true, AtomicOrdering::Release);

        let stranded = {
            let mut queue = self.lock_commit_wave_queue();
            let selected = match queue
                .wedged
                .as_ref()
                .and_then(CommitPathFailure::fixed_fault)
            {
                Some(existing) => existing,
                None => {
                    queue.wedged = Some(CommitPathFailure::Fixed(fault));
                    fault
                }
            };
            queue.sequencer_active = false;
            (selected, std::mem::take(&mut queue.items))
        };
        for item in stranded.1 {
            item.set_outcome(Err(ExecuteError::IndeterminateDurability(stranded.0)));
        }

        let pending_tails = {
            let mut tails = self
                .commit_wave
                .pending_tails
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut *tails)
        };
        let pending_tail_count = pending_tails.len() as u64;
        for mut tail in pending_tails {
            tail.set_fixed_durability_failure(stranded.0);
        }
        if pending_tail_count != 0 {
            self.commit_wave
                .tails_finished
                .fetch_add(pending_tail_count, AtomicOrdering::Release);
        }

        if let Some(lanes) = &self.intent_lanes {
            for queue in &lanes.queues {
                let intents = {
                    let mut queue = queue
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    std::mem::take(&mut *queue)
                };
                for item in intents {
                    item.set_outcome(Err(ExecuteError::IndeterminateDurability(stranded.0)));
                }
            }
            let resize_hold = {
                let mut resize_hold = lanes
                    .resize_hold
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                std::mem::take(&mut *resize_hold)
            };
            for item in resize_hold {
                item.set_outcome(Err(ExecuteError::IndeterminateDurability(stranded.0)));
            }
            let requests = {
                let mut requests = lanes
                    .validate_queue
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                std::mem::take(&mut *requests)
            };
            for request in requests {
                *request
                    .slot
                    .result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(None);
                request.slot.done.store(true, AtomicOrdering::Release);
            }
        }
        self.commit_wave.cv.notify_all();
        // END FIXED SERIAL DURABILITY DRAIN CODE
        stranded.0
    }
}

/// Unwind-safe accounting for one claimed wave tail.  Exact failures settle the entire queue
/// first; compatibility failures retain the established `wedge_commit_path` behavior.
pub(super) struct TailCompletion<'a> {
    engine: &'a Engine,
    clean: bool,
    fixed_fault: Option<DurabilityFault>,
}

impl<'a> TailCompletion<'a> {
    pub(super) fn new(engine: &'a Engine) -> Self {
        Self {
            engine,
            clean: false,
            fixed_fault: None,
        }
    }

    pub(super) fn mark_clean(&mut self) {
        self.clean = true;
    }

    pub(super) fn mark_fixed_failure(&mut self, fault: DurabilityFault) {
        self.fixed_fault = Some(fault);
    }
}

impl Drop for TailCompletion<'_> {
    fn drop(&mut self) {
        let wedge = !self.clean;
        if wedge {
            if let Some(fault) = self.fixed_fault {
                self.engine.fail_all_pending_commit_work_fixed(fault);
            } else {
                self.engine.wedge_commit_path();
            }
        }
        #[cfg(test)]
        if wedge {
            let hook = wave_tail_failure_publish_hook()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            if let Some((engine, reached, resume)) = hook {
                if engine == self.engine as *const Engine as usize {
                    reached.wait();
                    resume.wait();
                }
            }
        }
        self.engine
            .commit_wave
            .tails_finished
            .fetch_add(1, AtomicOrdering::Release);
        let queue = self.engine.lock_commit_wave_queue();
        self.engine.commit_wave.cv.notify_all();
        drop(queue);
    }
}

#[cfg(test)]
fn injected_fixed_group_completion_failure(
) -> &'static std::sync::Mutex<Option<(usize, DurabilityFault)>> {
    static INJECTED: std::sync::OnceLock<std::sync::Mutex<Option<(usize, DurabilityFault)>>> =
        std::sync::OnceLock::new();
    INJECTED.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(test)]
pub(super) fn inject_fixed_group_completion_failure_for_test(
    engine: &Engine,
    result: Result<usize, EngineError>,
) -> Result<usize, EngineError> {
    let injected = injected_fixed_group_completion_failure();
    let mut injected = injected
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match injected.take() {
        Some((target, fault)) if target == engine as *const Engine as usize => {
            Err(EngineError::DurabilityFault(fault))
        }
        Some(other) => {
            *injected = Some(other);
            result
        }
        None => result,
    }
}

#[cfg(test)]
pub(crate) fn set_fixed_group_completion_failure_for_test(engine: &Engine, fault: DurabilityFault) {
    let injected = injected_fixed_group_completion_failure();
    *injected
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
        Some((engine as *const Engine as usize, fault));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_dml_concurrent::{CanonicalRequest, CommitWaveTail};
    use crate::engine_residency::assert_no_thread_allocations;
    use crate::{parse_command, WriteSet};
    use gpu_db_types::{DurabilityBackend, DurabilityStage};
    use std::sync::Arc;

    fn fixed_fault(stage: DurabilityStage, sequence: u64) -> DurabilityFault {
        DurabilityFault::new(DurabilityBackend::SerialWal, stage, Some(5), 91, sequence)
    }

    fn pending_item(
        engine: &Engine,
        txn_id: u64,
    ) -> (
        super::super::CommitWaveItem,
        super::super::CommitWaveOutcome,
    ) {
        let text = "INSERT INTO fixed_durability_fixture VALUES (1)";
        let item = engine.make_covered_insert_wave_item(
            txn_id,
            parse_command(text).expect("fixture insert parses"),
            CanonicalRequest::from_text(engine, text),
            WriteSet::default(),
            engine.committed_seq(),
            engine.catalog_snapshot().commit_seq,
            None,
            None,
        );
        let outcome = Arc::clone(&item.outcome);
        (item, outcome)
    }

    fn assert_fixed_outcome(result: Result<u64, ExecuteError>, fault: DurabilityFault) {
        assert!(matches!(
            result,
            Err(ExecuteError::IndeterminateDurability(observed)) if observed == fault
        ));
    }

    #[test]
    fn fixed_group_completion_drains_queue_tail_and_waiter_with_one_fault() {
        for (sequence, stage) in [
            DurabilityStage::PositionalWrite,
            DurabilityStage::SyncData,
            DurabilityStage::Abandoned,
        ]
        .into_iter()
        .enumerate()
        {
            let engine = Engine::new_local_test_engine();
            let fault = fixed_fault(stage, sequence as u64 + 1);
            let (queued, queued_outcome) = pending_item(&engine, 10 + sequence as u64);
            let (tail_item, tail_outcome) = pending_item(&engine, 100 + sequence as u64);
            {
                let mut queue = engine.lock_commit_wave_queue();
                queue.sequencer_active = true;
                queue.items.push_back(queued);
            }
            let tail = CommitWaveTail {
                batch: vec![tail_item],
                committed: Vec::new(),
                typed_ledger_receipts: Vec::new(),
                last_position: 1,
                armed: true,
                durability_fault: None,
            };

            set_fixed_group_completion_failure_for_test(&engine, fault);
            std::thread::scope(|scope| {
                let waiter = scope.spawn(|| engine.await_commit_wave_outcome(&queued_outcome));
                std::thread::yield_now();
                engine.finish_wave_tail(tail);
                assert_fixed_outcome(waiter.join().expect("queue waiter does not panic"), fault);
            });
            assert_fixed_outcome(
                tail_outcome
                    .take_if_done()
                    .expect("tail owner receives fixed outcome"),
                fault,
            );
            assert_eq!(engine.group_flush.fixed_poison.snapshot(), Some(fault));
            assert!(matches!(
                engine
                    .ensure_commit_path_available()
                    .expect_err("fixed failure wedges service"),
                EngineError::DurabilityFault(observed) if observed == fault
            ));
        }
    }

    #[test]
    fn fixed_drain_is_allocation_free_and_first_fault_wins() {
        let engine = Engine::new_local_test_engine();
        let first = fixed_fault(DurabilityStage::PositionalWrite, 7);
        let second = fixed_fault(DurabilityStage::SyncData, 8);
        let (item, outcome) = pending_item(&engine, 7);
        engine.lock_commit_wave_queue().items.push_back(item);

        assert_eq!(
            assert_no_thread_allocations(|| engine.fail_all_pending_commit_work_fixed(first)),
            first
        );
        assert_fixed_outcome(
            outcome
                .take_if_done()
                .expect("fixed drain settles queued item"),
            first,
        );
        assert_eq!(engine.fail_all_pending_commit_work_fixed(second), first);
        assert_eq!(engine.group_flush.fixed_poison.snapshot(), Some(first));
    }

    #[test]
    fn fixed_drain_code_has_no_build_or_formatting_escape_hatch() {
        let source = include_str!("durability_failure.rs");
        let (_, fixed) = source
            .split_once("BEGIN FIXED SERIAL DURABILITY DRAIN CODE")
            .expect("fixed drain start marker");
        let (fixed, _) = fixed
            .split_once("END FIXED SERIAL DURABILITY DRAIN CODE")
            .expect("fixed drain end marker");
        for forbidden in ["format!(", ".to_string()", "collect::<", "Vec::new"] {
            assert!(
                !fixed.contains(forbidden),
                "fixed serial drain must not allocate or rebuild: {forbidden}"
            );
        }
    }
}
