//! The single logical publication join for live commits.
//!
//! Physical preparation, WAL flushing, and GPU apply may complete out of order. None of those
//! strategies may advance reader visibility directly: they report only the exact commit indices
//! whose durability and apply work are complete. This owner advances the published boundary over
//! the contiguous ready prefix and performs the sole release-store observed by readers.

use super::{CommitState, Engine, EngineError, Index};
use std::collections::BTreeSet;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Condvar, Mutex};

#[cfg(test)]
type CommitPrelockHook = (
    usize,
    std::sync::Arc<std::sync::Barrier>,
    std::sync::Arc<std::sync::Barrier>,
);

#[cfg(test)]
fn commit_prelock_hook() -> &'static Mutex<Option<CommitPrelockHook>> {
    static HOOK: std::sync::OnceLock<Mutex<Option<CommitPrelockHook>>> = std::sync::OnceLock::new();
    HOOK.get_or_init(|| Mutex::new(None))
}

#[derive(Debug)]
pub(crate) struct CommitPublicationCoordinator {
    state: Mutex<CommitPublicationState>,
    tail_cv: Condvar,
    tails_started: std::sync::atomic::AtomicU64,
    tails_finished: std::sync::atomic::AtomicU64,
}

/// Payload bytes of one publication-join entry. The slot pool, not this scalar, bounds BTree
/// node and allocator overhead; no per-transaction atomic publication-root object exists yet.
pub(crate) const fn commit_publication_join_entry_payload_bytes() -> usize {
    std::mem::size_of::<Index>()
}

#[derive(Debug)]
struct CommitPublicationState {
    /// First commit index not yet publication-covered. Commit indices begin at one; the reader
    /// watermark is the inclusive `visible_next - 1` value.
    visible_next: Index,
    /// Individually durable-and-applied indices stranded above a lower unfinished index.
    ready: BTreeSet<Index>,
}

impl Default for CommitPublicationCoordinator {
    fn default() -> Self {
        Self {
            state: Mutex::new(CommitPublicationState {
                visible_next: 1,
                ready: BTreeSet::new(),
            }),
            tail_cv: Condvar::new(),
            tails_started: std::sync::atomic::AtomicU64::new(0),
            tails_finished: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl Engine {
    #[cfg(test)]
    pub(crate) fn set_commit_prelock_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *commit_prelock_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    #[cfg(test)]
    pub(crate) fn run_commit_prelock_hook(&self) {
        let hook = {
            let mut hook = commit_prelock_hook()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            hook.as_ref()
                .is_some_and(|(owner, _, _)| *owner == self as *const Self as usize)
                .then(|| hook.take())
                .flatten()
        };
        if let Some((_, reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
    }

    /// Acquire the global claim/apply lock only after every earlier classic wave has completed its
    /// off-lock durability/publication tail, and re-prove that condition after acquiring the lock.
    /// Serialized, COPY, explicit, and lane-activation claimants use this one gap-closing entrance.
    pub(crate) fn commit_state_after_wave_quiescence(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, CommitState>, EngineError> {
        loop {
            if !self.wait_wave_tail_quiescence() {
                return Err(self.commit_path_unavailable_error());
            }
            self.wait_publication_tail_quiescence()?;
            let commit = self.commit_state();
            let applied = self.commit_wave.tails_applied.load(AtomicOrdering::Acquire);
            let finished = self
                .commit_wave
                .tails_finished
                .load(AtomicOrdering::Acquire);
            let publication_started = self
                .commit_publication
                .tails_started
                .load(AtomicOrdering::Acquire);
            let publication_finished = self
                .commit_publication
                .tails_finished
                .load(AtomicOrdering::Acquire);
            if applied == finished && publication_started == publication_finished {
                self.ensure_commit_path_available()?;
                return Ok(commit);
            }
            drop(commit);
            self.ensure_commit_path_available()?;
        }
    }

    /// Register an apply-complete durability/publication tail while the canonical commit lock is
    /// still held. Quiescent claimants recheck this counter under that same lock, closing the
    /// apply-to-off-lock-durability handoff gap for every physical strategy.
    pub(crate) fn register_publication_tail(&self) {
        self.commit_publication
            .tails_started
            .fetch_add(1, AtomicOrdering::Release);
    }

    /// Complete a registered tail after durability, contiguous publication, and outcome
    /// resolution (or after wedging it fail-closed). Wake both generic quiescence waiters and the
    /// inherited classic-wave publication wait loop.
    pub(crate) fn finish_publication_tail(&self) {
        self.commit_publication
            .tails_finished
            .fetch_add(1, AtomicOrdering::Release);
        self.commit_publication.tail_cv.notify_all();
        self.commit_wave.cv.notify_all();
    }

    fn wait_publication_tail_quiescence(&self) -> Result<(), EngineError> {
        let mut state = self
            .commit_publication
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            let started = self
                .commit_publication
                .tails_started
                .load(AtomicOrdering::Acquire);
            let finished = self
                .commit_publication
                .tails_finished
                .load(AtomicOrdering::Acquire);
            if started == finished {
                return Ok(());
            }
            self.ensure_commit_path_available()?;
            state = self
                .commit_publication
                .tail_cv
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Report exact durable-and-applied commit indices and publish only their new contiguous
    /// prefix. Duplicate reports below the prefix are idempotent; a future ready index remains
    /// hidden until every lower index is also reported.
    pub(crate) fn publish_ready_indices(
        &self,
        indices: impl IntoIterator<Item = Index>,
    ) -> Result<Index, EngineError> {
        let mut state = self
            .commit_publication
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for index in indices {
            if index == 0 || index == u64::MAX {
                return Err(EngineError::Durability(format!(
                    "commit index {index} is outside the publishable sequence domain"
                )));
            }
            if index >= state.visible_next {
                state.ready.insert(index);
            }
        }
        loop {
            let next = state.visible_next;
            if !state.ready.remove(&next) {
                break;
            }
            state.visible_next = state.visible_next.checked_add(1).ok_or_else(|| {
                EngineError::Durability("publication visible-next overflow".to_string())
            })?;
        }
        let visible = state.visible_next - 1;
        self.read_state
            .committed_seq
            .store(visible, AtomicOrdering::Release);
        Ok(visible)
    }

    /// A non-pipelined claimant may acknowledge only if the exact terminal index it just reported
    /// is already inside the contiguous published prefix. Such paths quiesce classic tails before
    /// claiming, so a gap here is an invariant failure after durable apply and wedges service.
    pub(crate) fn require_publication_coverage(
        &self,
        visible: Index,
        terminal: Index,
    ) -> Result<(), EngineError> {
        if visible >= terminal {
            return Ok(());
        }
        self.wedge_commit_path();
        Err(EngineError::Durability(format!(
            "durable applied commit {terminal} is not publication-covered (visible through {visible}); restart recovery is required"
        )))
    }

    #[cfg(test)]
    pub(crate) fn publish_ready_index(&self, index: Index) -> Result<Index, EngineError> {
        self.publish_ready_indices(std::iter::once(index))
    }

    /// Snapshot installation is a quiescent authority replacement, not live commit completion.
    /// Reset the join to the installed prefix and publish that inclusive boundary atomically.
    pub(crate) fn install_publication_snapshot(&mut self, last_included: Index) {
        let state = self
            .commit_publication
            .state
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.visible_next = last_included.saturating_add(1);
        state.ready.clear();
        self.read_state
            .committed_seq
            .store(last_included, AtomicOrdering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn publication_join_entry_payload_size_is_owned_by_its_index_value() {
        assert_eq!(
            commit_publication_join_entry_payload_bytes(),
            std::mem::size_of::<Index>()
        );
    }

    fn release_simulated_wave_tail(engine: &Engine) {
        engine
            .commit_wave
            .tails_finished
            .store(1, Ordering::Release);
        engine.notify_simulated_wave_tail_change();
    }

    #[test]
    fn publication_join_never_skips_an_unfinished_lower_index() {
        let engine = Engine::new_local();
        assert_eq!(engine.publish_ready_index(2).unwrap(), 0);
        assert_eq!(engine.committed_seq(), 0);
        assert_eq!(engine.publish_ready_index(1).unwrap(), 2);
        assert_eq!(engine.committed_seq(), 2);
    }

    #[test]
    fn duplicate_ready_reports_are_idempotent() {
        let engine = Engine::new_local();
        assert_eq!(engine.publish_ready_index(1).unwrap(), 1);
        assert_eq!(engine.publish_ready_index(1).unwrap(), 1);
        assert_eq!(engine.committed_seq(), 1);
    }

    #[test]
    fn serialized_claim_waits_for_an_applied_classic_tail() {
        let engine = Arc::new(Engine::new_local());
        engine.commit_wave.tails_applied.store(1, Ordering::Release);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let writer = {
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || {
                let result =
                    engine.commit_mutation_at(1, Arc::from(&b"SET publication_gate=open"[..]), 1);
                done_tx.send(()).unwrap();
                result
            })
        };
        assert!(done_rx.recv_timeout(Duration::from_millis(50)).is_err());
        release_simulated_wave_tail(&engine);
        writer.join().unwrap().unwrap();
        assert_eq!(engine.committed_seq(), 1);
    }

    #[test]
    fn serialized_claim_waits_for_a_registered_canonical_tail() {
        let engine = Arc::new(Engine::new_local());
        engine.register_publication_tail();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let writer = {
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || {
                let result =
                    engine.commit_mutation_at(1, Arc::from(&b"SET canonical_gate=open"[..]), 1);
                done_tx.send(()).unwrap();
                result
            })
        };
        assert!(done_rx.recv_timeout(Duration::from_millis(50)).is_err());
        engine.finish_publication_tail();
        writer.join().unwrap().unwrap();
        assert_eq!(engine.committed_seq(), 1);
    }
}
