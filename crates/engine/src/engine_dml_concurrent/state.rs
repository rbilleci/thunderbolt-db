//! Commit-wave and lane item ownership shared by the concurrent DML subpaths.

use super::{
    AtomicOrdering, AtomicU64, Command, Engine, ExecuteError, Index, Mutex, RelationalSelectResult,
    SqlValue, WriteSet,
};
use std::sync::Arc;

#[cfg(test)]
type WaveTailTestHook = (
    usize,
    std::sync::Arc<std::sync::Barrier>,
    std::sync::Arc<std::sync::Barrier>,
);

#[cfg(test)]
pub(super) fn wave_tail_handoff_hook() -> &'static Mutex<Option<WaveTailTestHook>> {
    static HOOK: std::sync::OnceLock<Mutex<Option<WaveTailTestHook>>> = std::sync::OnceLock::new();
    HOOK.get_or_init(|| Mutex::new(None))
}

#[cfg(test)]
pub(super) fn wave_tail_failure_publish_hook() -> &'static Mutex<Option<WaveTailTestHook>> {
    static HOOK: std::sync::OnceLock<Mutex<Option<WaveTailTestHook>>> = std::sync::OnceLock::new();
    HOOK.get_or_init(|| Mutex::new(None))
}

/// One enqueued concurrent commit: everything the sequencer needs to conflict-check, re-resolve,
/// append, apply, and publish it — plus the shared slot its owner blocks on.
pub(crate) struct CommitWaveItem {
    pub(super) txn_id: u64,
    pub(super) cmd: Command,
    pub(super) payload: Arc<[u8]>,
    pub(super) write_set: WriteSet,
    pub(super) read_snapshot: Index,
    /// The catalog generation the OFF-LOCK prepare validated against.
    /// The sequencer grants the device-covered re-resolve skip ONLY while the live catalog
    /// still carries this stamp — a constraint-adding DDL (ADD UNIQUE/CHECK) committing
    /// between snapshot and wave is absent from the prepared key projection, so the skip would
    /// silently bypass the new constraint; any DDL bumps the stamp and forces the
    /// always-correct Full re-validation instead.
    pub(super) prepared_catalog_seq: Index,
    /// Catalog generation revalidated for a protocol-neutral prepared execution. Unlike
    /// `prepared_catalog_seq` (the off-lock optimizer stamp), this is a correctness precondition:
    /// a mismatch must fail before WAL/apply rather than re-resolve under a changed row type.
    pub(super) expected_catalog_version:
        Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
    /// DELTA-REUSE (B): the OFF-LOCK-prepared insert delta, carried forward for reuse-eligible
    /// items (elided, FK-free, no nextval). The under-lock re-resolve then only RE-KEYS it at the
    /// wave's `next_row_id` (`rekey_offlock_insert_delta`) instead of re-running the full
    /// coerce+validate+write-set rebuild — the coerced values / write-set are input-deterministic,
    /// so they are identical to a fresh re-prepare while the catalog generation still matches
    /// (`prepared_catalog_seq`; a DDL bump forces the Full path, dropping the reuse). `None` = the
    /// item takes the normal `prepare_dml` re-resolve.
    pub(super) offlock_delta: Option<crate::write_path::WriteDelta>,
    /// E2.2(b) — the PRE-ENCODED W5a binary WAL record, built OFF the sequencer at intent-build
    /// time as a pure function of `(route, params)` with a PLACEHOLDER row id, plus the fixed byte
    /// offset of that row id. Present only for single-row covered-INSERT intents. The sequencer
    /// patches the 8-byte row id at `offset` with the wave-assigned id (no String row-key parse, no
    /// per-item `encode_relational_row` + `try_encode_binary_insert`) and uses the result verbatim
    /// as the reuse-eligible delta's WAL payload. `None` = the classic per-item encode path.
    pub(super) binary_wal_template: Option<(Arc<[u8]>, u32)>,
    /// Shared stable-OID lease owned by the queued work through terminal apply/cancel. A caller
    /// ticket may be dropped independently; the mutation item remains the reset-exclusion owner.
    pub(crate) table_access: Option<Arc<crate::table_access::TableAccessLease>>,
    pub(super) outcome: CommitWaveOutcome,
}

pub(crate) type CommitWaveOutcome = Arc<CommitWaveDone>;

/// A wave item's completion slot: the payload behind a mutex, the `done` flag an ATOMIC so
/// waiters can SPIN on completion (a few µs) instead of paying a futex sleep+wake round-trip
/// per commit — the wakeup latency, not the mutex, dominated the first wave measurement.
///
/// U1: the Ok payload is ROWS AFFECTED (INSERT intents = 1; classic wave items = the applied
/// delta's exact row count; 0-row lane DELETEs complete with Ok(0) at the pre-claim filter) —
/// the engine's first rows-affected surface, introduced with the lane DELETE intents.
#[derive(Default)]
pub(crate) struct CommitWaveDone {
    pub(super) done: std::sync::atomic::AtomicBool,
    result: Mutex<Option<Result<u64, ExecuteError>>>,
    returning: Mutex<Option<RelationalSelectResult>>,
}

impl CommitWaveDone {
    pub(crate) fn is_done(&self) -> bool {
        self.done.load(AtomicOrdering::Acquire)
    }

    pub(crate) fn take_if_done(&self) -> Option<Result<u64, ExecuteError>> {
        if !self.done.load(AtomicOrdering::Acquire) {
            return None;
        }
        self.result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }

    pub(crate) fn set_returning(&self, returning: Option<RelationalSelectResult>) {
        *self
            .returning
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = returning;
    }

    pub(crate) fn take_returning(&self) -> Option<RelationalSelectResult> {
        self.returning
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

/// U1/U2: the lane-op kind. INSERT is the E2 flagship; DELETE + UPDATE are the Tier-1 mutation
/// ops — covered by-PK, target resolved by the coalesced device VISIBLE-LOCATE at APPLY (WAL-first:
/// the locate moved off the pump critical path). An UPDATE rides the delete's tombstone plus an
/// insert's append: tombstone-OLD + append-NEW with the old version's GPU-returned stable entity
/// identity. The append is CONDITIONAL on the old-version locate. The v1 WAL record still burns a
/// legacy row-id reservation for replay-format/high-water compatibility, including on a 0-row update.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneOpKind {
    Insert,
    Delete,
    Update,
}

/// E2.5b-2 LEAN LANE ITEM: everything the lane pump needs, ~112B + the value
/// row — vs the ~500B CommitWaveItem plus its AST/delta/text attachments. The
/// pump's host passes were the measured final wall (~6.5ms/lane cycle of cold
/// cache traffic at ~925-item waves); this struct is the fix.
pub(crate) struct LaneIntent {
    /// U1/U2: which op this intent performs (Insert rides every existing path
    /// unchanged; Delete adds the visible-locate + tombstone arms; Update adds a
    /// conditional new-version append on top of the delete's locate + tombstone).
    pub(crate) op: LaneOpKind,
    pub(crate) txn_id: u64,
    pub(crate) slot: crate::write_path::IntUniqueSlotKey,
    pub(crate) read_snapshot: Index,
    pub(crate) prepared_catalog_seq: Index,
    pub(crate) filter_idx: u32,
    pub(crate) row_id_offset: u32,
    pub(crate) table: Arc<str>,
    pub(crate) template: Arc<[u8]>,
    pub(crate) values: Vec<SqlValue>,
    pub(crate) outcome: CommitWaveOutcome,
    /// Shared table/dependency lease retained by the lane item until its terminal outcome.
    pub(crate) table_access: Option<Arc<crate::table_access::TableAccessLease>>,
    /// Stable request identity plus the one shared admission-to-WAL reservation registry used by
    /// every queued write strategy. Durable terminal status belongs to the canonical
    /// `CommitState` transaction index.
    pub(crate) request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) transaction_claims:
        Option<Arc<Mutex<std::collections::HashMap<u64, gpu_db_wal::CanonicalDigest>>>>,
    /// Live-population decrement handle (see `IntentLaneState::outstanding`);
    /// None outside lanes mode.
    pub(crate) outstanding: Option<Arc<std::sync::atomic::AtomicU64>>,
    /// U1: the rows-affected count a settled Ok reports for INSERT intents (always 1).
    pub(crate) rows_affected: u64,
    /// U1/U2 WAL-first: a DELETE's (and UPDATE's) rows-affected is resolved at APPLY (the locate
    /// moved off the pump critical path), so the outcome comes from this shared cell the apply
    /// writes (0 or 1). `None` for inserts — they use `rows_affected`. Shared with the delete's
    /// `LaneTombstone.rows_affected` / the update's `LaneUpdate.rows_affected`; completion reads it
    /// after canonical device apply succeeds.
    pub(crate) rows_affected_cell: Option<Arc<std::sync::atomic::AtomicU64>>,
}

impl LaneIntent {
    /// The rows-affected an Ok outcome reports: a delete reads its apply-resolved cell; an
    /// insert (no cell) uses the fixed `rows_affected` (1). Read at completion, after apply.
    pub(crate) fn resolved_rows_affected(&self) -> u64 {
        match &self.rows_affected_cell {
            Some(cell) => cell.load(AtomicOrdering::Acquire),
            None => self.rows_affected,
        }
    }

    pub(crate) fn set_outcome(&self, result: Result<u64, ExecuteError>) {
        let mut outcome = self
            .outcome
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.outcome.done.load(AtomicOrdering::Acquire) {
            return;
        }
        if let Some(claims) = &self.transaction_claims {
            let mut claims = claims
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if claims
                .get(&self.txn_id)
                .is_some_and(|claim| *claim == self.request_digest)
            {
                claims.remove(&self.txn_id);
            }
        }
        *outcome = Some(result);
        self.outcome.done.store(true, AtomicOrdering::Release);
        // Population bookkeeping: set_outcome is the single completion choke
        // point, so submit/settle pairing is exact by construction.
        if let Some(outstanding) = &self.outstanding {
            outstanding.fetch_sub(1, AtomicOrdering::Relaxed);
        }
    }
}

/// A fresh, pending completion slot (shared by the lean lane path).
pub(crate) fn new_pending_outcome() -> CommitWaveOutcome {
    Arc::new(CommitWaveDone {
        done: std::sync::atomic::AtomicBool::new(false),
        result: Mutex::new(None),
        returning: Mutex::new(None),
    })
}

impl CommitWaveItem {
    pub(crate) fn set_outcome(&self, result: Result<u64, ExecuteError>) {
        let mut outcome = self
            .outcome
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.outcome.done.load(AtomicOrdering::Acquire) {
            return;
        }
        *outcome = Some(result);
        self.outcome.done.store(true, AtomicOrdering::Release);
    }
}

/// The deterministic commit-wave state (ledger #6): the arrival-ordered item queue, the
/// single-sequencer election flag, and the sticky wedge. The condvar doubles as the completion
/// signal for waiters, the promotion signal for the next sequencer, and (W2) the tail-finished
/// signal for the depth-1 durability pipeline.
pub(crate) struct CommitWaveState {
    pub(super) queue: Mutex<CommitWaveQueue>,
    pub(crate) cv: std::sync::Condvar,
    /// W2/W2b — the DURABILITY PIPELINE (depth [`WAVE_TAIL_PIPELINE_DEPTH`]): sequenced waves
    /// whose group-fsync wait, `committed_seq` publish, and outcome acks have NOT yet run. The
    /// sequencer pushes tails here and immediately drains/sequences the NEXT wave under the
    /// commit_mutex; tails are FINISHED (fsync-wait → publish → acks) by whichever threads claim
    /// them — the waves' own blocked waiters (they are spinning on their outcomes anyway) or the
    /// sequencer as the fallback claimer at the capacity gate. Finish order is UNCONSTRAINED:
    /// a tail's durability wait covers all earlier WAL positions (prefix frontier) and the
    /// publication joins exact ready indices into one contiguous prefix, so concurrent out-of-order
    /// finishing is safe without exposing a gap. This is what lets
    /// wave N+1's serial sequencing overlap wave N's fdatasync, and lets consecutive waves'
    /// records coalesce into SHARED fsyncs via the WAL's group-flush protocol.
    /// Liveness: a pending tail always has ≥1 live claimer — its members' outcomes are unset
    /// until it finishes, so they are by definition still in the waiter loops (which probe this
    /// deque), and the sequencer try-claims before ever blocking on the capacity gate.
    pub(super) pending_tails: Mutex<std::collections::VecDeque<CommitWaveTail>>,
    /// Tails handed to the pipeline slot / tails fully finished. `handed == finished` ⇔ the
    /// pipeline is empty (the depth-1 gate the sequencer enforces before handing a new tail).
    pub(super) tails_handed: AtomicU64,
    /// Registered under the commit lock as soon as a wave has installed state. This closes the
    /// apply-to-deque handoff gap that `tails_handed` cannot witness.
    pub(crate) tails_applied: AtomicU64,
    pub(crate) tails_finished: AtomicU64,
}

impl Default for CommitWaveState {
    fn default() -> Self {
        Self {
            queue: Mutex::new(CommitWaveQueue::default()),
            cv: std::sync::Condvar::new(),
            pending_tails: Mutex::new(std::collections::VecDeque::new()),
            tails_handed: AtomicU64::new(0),
            tails_applied: AtomicU64::new(0),
            tails_finished: AtomicU64::new(0),
        }
    }
}

impl Engine {
    /// Passive explicit-transaction barrier: wait until every classic wave registered under the
    /// commit lock has finished durability/visibility publication. Unlike the sequencer capacity
    /// gate, this never claims another client's tail:
    /// an injected fsync failure must reach COMMIT as the sticky fail-stop error, not as an
    /// unrelated tail-finisher panic on the transaction thread.
    pub(crate) fn wait_wave_tail_quiescence(&self) -> bool {
        loop {
            let applied = self.commit_wave.tails_applied.load(AtomicOrdering::Acquire);
            let finished = self
                .commit_wave
                .tails_finished
                .load(AtomicOrdering::Acquire);
            if applied == finished {
                return true;
            }
            let queue = self.lock_commit_wave_queue();
            if queue.wedged.is_some() {
                return false;
            }
            let applied = self.commit_wave.tails_applied.load(AtomicOrdering::Acquire);
            let finished = self
                .commit_wave
                .tails_finished
                .load(AtomicOrdering::Acquire);
            if applied == finished {
                return true;
            }
            let _queue = self
                .commit_wave
                .cv
                .wait(queue)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

/// W2 — one sequenced-but-not-yet-durable wave: everything needed to finish it OFF the
/// sequencer's critical path. The deltas are applied and the WAL records appended (that is what
/// lets the next wave's re-resolves see them); nothing is client-visible until `finish` runs the
/// group-durability wait and publishes `committed_seq` (WAL-before-visibility per tail; the
/// the publication join holds out-of-order completion behind gaps). `armed` keeps the
/// wedge-don't-strand policy:
/// a tail dropped unfinished (claimer panic, pipeline abandonment) fails every still-unset
/// outcome and wedges the queue, exactly like `CommitWaveBatchGuard` does for the in-section
/// half of the wave.
pub(super) struct CommitWaveTail {
    pub(super) batch: Vec<CommitWaveItem>,
    /// `(batch position, commit_seq, rows_affected)` for every item that reached the
    /// durable-commit point, in wave order (aborted items' outcomes were already set in-section).
    /// `rows_affected` is the applied delta's exact row count — the Ok payload of the ack (U1).
    pub(super) committed: Vec<(usize, Index, u64)>,
    pub(super) last_position: usize,
    pub(super) armed: bool,
}

impl Drop for CommitWaveTail {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // The durability half of the wave died before acking (the group-fsync-failure panic
        // path, or claimer death): fail every still-unset member outcome. Queue wedging + the
        // finished-counter bump need `&Engine` and are handled by `finish_wave_tail`'s
        // unwind-safe completion guard.
        for item in &self.batch {
            if !item.outcome.done.load(AtomicOrdering::Acquire) {
                item.set_outcome(Err(ExecuteError::Indeterminate(
                    "the concurrent commit path is wedged pending restart recovery: the \
                     commit-wave durability tail died after sequence/WAL assignment"
                        .to_string(),
                )));
            }
        }
    }
}

#[derive(Default)]
pub(super) struct CommitWaveQueue {
    pub(super) items: std::collections::VecDeque<CommitWaveItem>,
    pub(super) sequencer_active: bool,
    /// Sticky: a wave failed after its deltas were applied (durability failure mid-wave). No
    /// further concurrent commits may run until restart recovery.
    pub(super) wedged: Option<String>,
}
