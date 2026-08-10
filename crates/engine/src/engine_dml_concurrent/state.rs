//! Commit-wave and lane item ownership shared by the concurrent DML subpaths.

use super::CommitPathFailure;
use super::{
    AtomicOrdering, AtomicU64, Command, Engine, EngineError, ExecuteError, Index, Mutex,
    RelationalSelectResult, SqlValue, WriteSet,
};
use gpu_db_types::DurabilityFault;
use std::sync::Arc;

/// One immutable request allocation and the canonical identity derived from its exact bytes.
///
/// Admission creates this before it takes a table-access lease.  The bytes cannot be replaced or
/// mutably borrowed after construction, so queued retry, WAL, and status users can carry the
/// same digest without independently hashing the same logical request.  There is deliberately no
/// from-parts constructor: an identity always originates from this exact `Arc` allocation.
pub(crate) struct CanonicalRequest {
    payload: Arc<[u8]>,
    digest: gpu_db_wal::CanonicalDigest,
}

impl CanonicalRequest {
    /// Seal one request identity before table access. With `probe-timing` this is also the sole
    /// accounting seam for bytes passed to canonical digest derivation; the feature-off hot path
    /// remains the allocation and digest it already required.
    pub(crate) fn from_text(engine: &Engine, text: &str) -> Self {
        let payload: Arc<[u8]> = Arc::from(text.as_bytes());
        let digest = gpu_db_wal::canonical_request_digest(payload.as_ref());
        #[cfg(feature = "probe-timing")]
        engine.record_insert_probe_raw_request_digest_derivation(
            u64::try_from(payload.len()).expect("request payload length exceeds u64"),
        );
        #[cfg(not(feature = "probe-timing"))]
        let _ = engine;
        Self { payload, digest }
    }

    pub(crate) fn digest(&self) -> gpu_db_wal::CanonicalDigest {
        self.digest
    }

    pub(super) fn payload_arc(&self) -> Arc<[u8]> {
        Arc::clone(&self.payload)
    }

    #[cfg(test)]
    fn same_origin(&self, payload: &Arc<[u8]>) -> bool {
        Arc::ptr_eq(&self.payload, payload)
    }
}

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

/// One authoritative off-lock UPDATE/DELETE preparation carried into a commit wave.
/// INSERT is intercepted into the codec-5 transaction overlay before this type is constructed.
pub(super) struct OfflockPreparedDml {
    delta: crate::write_path::WriteDelta,
    read_snapshot: Index,
}

impl OfflockPreparedDml {
    /// The generic wave owns UPDATE and DELETE only.  INSERT must retain its consumed typed
    /// batch through the canonical cut; accepting an INSERT `WriteDelta` here would recreate a
    /// second live authority.
    pub(super) fn legacy_update_or_delete(
        delta: crate::write_path::WriteDelta,
        read_snapshot: Index,
    ) -> Result<Self, EngineError> {
        if matches!(
            &delta.mutation,
            crate::write_path::PreparedMutation::Insert { .. }
        ) {
            return Err(EngineError::ApplyFailed(
                "legacy off-lock carrier rejects INSERT; use the typed INSERT authority"
                    .to_string(),
            ));
        }
        Ok(Self {
            delta,
            read_snapshot,
        })
    }

    pub(super) fn legacy_delta(&self) -> Option<&crate::write_path::WriteDelta> {
        Some(&self.delta)
    }

    /// The wave's conflict footprint is always owned by the same off-lock carrier that supplied
    /// it. The item retains a clone for the canonical owner, never an independently derived set.
    pub(super) fn write_set(&self) -> &WriteSet {
        &self.delta.write_set
    }

    /// Snapshot identity travels with either carrier variant; a wave-item mismatch is a
    /// programming error at admission, while a fixed preflight still rechecks it defensively.
    pub(super) fn read_snapshot(&self) -> Index {
        self.read_snapshot
    }
}

#[cfg(test)]
mod offlock_prepared_tests {
    use super::*;

    #[test]
    fn commit_wave_done_payload_size_is_owned_by_its_value_layout() {
        assert_eq!(
            commit_wave_done_payload_bytes(),
            std::mem::size_of::<CommitWaveDone>()
        );
    }

    #[test]
    fn production_legacy_carrier_rejects_insert_deltas() {
        let insert = crate::write_path::WriteDelta {
            write_set: WriteSet::default(),
            read_snapshot: 0,
            catalog_dependencies: std::collections::BTreeMap::new(),
            foreign_key_dependencies: std::collections::BTreeSet::new(),
            rows_consumed: 1,
            mutation: crate::write_path::PreparedMutation::Insert {
                table: "forbidden_legacy_insert".to_string(),
                inserted_rows: Vec::new(),
                seq_advances: std::collections::BTreeMap::new(),
            },
        };
        let error = match OfflockPreparedDml::legacy_update_or_delete(insert, 0) {
            Ok(_) => panic!("the generic wave must not carry an INSERT delta"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("rejects INSERT"));
    }

    #[test]
    fn canonical_request_hashes_once_at_admission_and_waves_reuse_the_sealed_identity() {
        let state = include_str!("state.rs")
            .split("\n#[cfg(test)]\nmod offlock_prepared_tests")
            .next()
            .expect("production state precedes its tests");
        assert_eq!(state.matches("canonical_request_digest(").count(), 1);
        assert!(state.contains("Arc::ptr_eq(&self.payload, payload)"));
        assert!(!state.contains("from_parts"));

        let request = include_str!("request.rs");
        assert!(request.contains("CanonicalRequest::from_text(self, text)"));
        assert!(!request.contains("canonical_request_digest("));

        let wave = include_str!("wave.rs");
        assert!(!wave.contains("canonical_request_digest(&item."));
        assert!(!wave.contains("canonical_request_digest(&batch"));

        let intent = include_str!("../engine_dml_intent.rs");
        assert!(intent.contains("execute_parsed_dml_concurrent_with_result("));
        assert!(!intent.contains("canonical_request_digest(logical_request.as_bytes())"));
    }

    #[test]
    fn covered_wave_item_keeps_the_pre_lease_request_allocation_and_digest() {
        let engine = Engine::new_local();
        let text = "INSERT INTO accounts VALUES (7, 70)";
        let request = CanonicalRequest::from_text(&engine, text);
        let digest = request.digest();
        let witness = request.payload_arc();
        let item = engine.make_test_commit_wave_item(
            2,
            crate::parse_command(text).unwrap(),
            request,
            WriteSet::default(),
            engine.committed_seq(),
        );

        assert_eq!(item.request.digest(), digest);
        assert!(item.request.same_origin(&witness));
        assert_eq!(item.request.payload_arc().as_ref(), text.as_bytes());
    }

    #[cfg(feature = "probe-timing")]
    #[test]
    fn request_identity_probe_counts_only_the_sealed_dml_payload_bytes() {
        let engine = Engine::new_local();
        let before_create = engine.insert_probe_snapshot();
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        let after_create = engine.insert_probe_snapshot();
        let create_delta = after_create.delta_since(before_create);
        assert_eq!(create_delta.raw_request_digest_derivations, 0);
        assert_eq!(create_delta.raw_request_digest_derivation_bytes, 0);

        let sql = "INSERT INTO accounts VALUES (7, 70)";
        engine.execute_dml_concurrent(2, sql).unwrap();
        let delta = engine.insert_probe_snapshot().delta_since(after_create);
        assert_eq!(delta.raw_request_digest_derivations, 1);
        assert_eq!(
            delta.raw_request_digest_derivation_bytes,
            u64::try_from(sql.len()).unwrap()
        );
    }
}

/// One enqueued concurrent commit: everything the sequencer needs to conflict-check, re-resolve,
/// append, apply, and publish it — plus the shared slot its owner blocks on.
pub(crate) struct CommitWaveItem {
    pub(super) txn_id: u64,
    pub(super) cmd: Command,
    /// The single sealed request identity carried from admission through the canonical owner.
    pub(super) request: CanonicalRequest,
    pub(super) write_set: WriteSet,
    pub(super) read_snapshot: Index,
    /// Catalog generation revalidated for a protocol-neutral prepared execution. A mismatch must
    /// fail before WAL/apply rather than re-resolve under a changed row type.
    pub(super) expected_catalog_version:
        Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
    /// Off-lock UPDATE/DELETE delta used for conflict projection. The wave re-resolves at its
    /// serial commit sequence before WAL/apply.
    pub(super) offlock_prepared: Option<OfflockPreparedDml>,
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

/// Payload bytes of one completion value. Capacity slots, rather than this helper, account for
/// the enclosing `Arc` and allocator overhead.
pub(crate) const fn commit_wave_done_payload_bytes() -> usize {
    std::mem::size_of::<CommitWaveDone>()
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

    pub(crate) fn set_outcome(&self, result: Result<u64, ExecuteError>) {
        let mut outcome = self
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.done.load(AtomicOrdering::Acquire) {
            return;
        }
        *outcome = Some(result);
        self.done.store(true, AtomicOrdering::Release);
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
    Delete,
    Update,
}

/// E2.5b-2 LEAN LANE ITEM: everything the lane pump needs, ~112B + the value
/// row — vs the ~500B CommitWaveItem plus its AST/delta/text attachments. The
/// pump's host passes were the measured final wall (~6.5ms/lane cycle of cold
/// cache traffic at ~925-item waves); this struct is the fix.
pub(crate) struct LaneIntent {
    /// U1/U2: DELETE uses the visible-locate + tombstone arm; UPDATE adds a conditional
    /// new-version append on top of the locate + tombstone.
    pub(crate) op: LaneOpKind,
    pub(crate) txn_id: u64,
    pub(crate) slot: crate::write_path::IntUniqueSlotKey,
    pub(crate) read_snapshot: Index,
    pub(crate) prepared_catalog_seq: Index,
    pub(crate) filter_idx: u32,
    pub(crate) row_id_offset: u32,
    pub(crate) table: Arc<str>,
    /// Stable catalog identity carried through every lean-lane writer so its
    /// post-canonical table-root entry never falls back to a table name.
    pub(crate) table_oid: u32,
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
    /// Exact pre-WAL target cardinality reproduced by device apply.
    pub(crate) rows_affected: u64,
    /// U1/U2 WAL-first: a DELETE's (and UPDATE's) rows-affected is resolved at APPLY (the locate
    /// moved off the pump critical path), so the outcome comes from this shared cell the apply
    /// writes (0 or 1). Shared with the delete's `LaneTombstone.rows_affected` / the update's
    /// `LaneUpdate.rows_affected`; completion reads it after canonical device apply succeeds.
    pub(crate) rows_affected_cell: Option<Arc<std::sync::atomic::AtomicU64>>,
}

impl LaneIntent {
    /// Read the apply-resolved rows-affected result after device publication.
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
        self.outcome.set_outcome(result);
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
    /// A structured serial-WAL failure selected by the owning completion handoff.  Its `Copy`
    /// payload lets tail drop settle every member without constructing a string.
    pub(super) durability_fault: Option<DurabilityFault>,
}

impl CommitWaveTail {
    pub(super) fn set_fixed_durability_failure(&mut self, fault: DurabilityFault) {
        self.durability_fault = Some(fault);
    }
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
        if let Some(fault) = self.durability_fault {
            for item in &self.batch {
                if !item.outcome.done.load(AtomicOrdering::Acquire) {
                    item.set_outcome(Err(ExecuteError::IndeterminateDurability(fault)));
                }
            }
            return;
        }
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
    pub(super) wedged: Option<CommitPathFailure>,
}
