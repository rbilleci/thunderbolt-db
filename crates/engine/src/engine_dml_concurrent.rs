//! Concurrent DML path + top-level text execution (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for the snapshot-isolation
//! autocommit write path (is_concurrent_dml, register/deregister_active_snapshot,
//! execute_dml_concurrent[_instrumented], prepare_dml, dml_mutated_tables,
//! commit_dml_concurrent), the execute_text / execute_text_at_timestamp_micros /
//! execute_read_text entry points, and the read-pin helper.

use super::*;

/// Upper bound on items per wave drain (keeps a single wave's worst-case commit latency bounded;
/// under saturation the NEXT wave picks the rest up immediately).
const COMMIT_WAVE_MAX_DEFAULT: usize = 1024;

/// E2.2(d) — the wave-size / pipeline-depth knobs are env-overridable for the latency-knee sweep
/// (`GPU_DB_COMMIT_WAVE_MAX`, `GPU_DB_WAVE_TAIL_PIPELINE_DEPTH`), read once. The defaults are the
/// production values; the sweep finds the throughput/latency knee once (a)-(c) reshape the loop.
fn commit_wave_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GPU_DB_COMMIT_WAVE_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(COMMIT_WAVE_MAX_DEFAULT)
    })
}

fn wave_tail_pipeline_depth() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GPU_DB_WAVE_TAIL_PIPELINE_DEPTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(WAVE_TAIL_PIPELINE_DEPTH_DEFAULT)
    })
}

/// E2.4a — VARIANT 1 (shared-WAL sharded sequencing): the number of PARALLEL shard workers the
/// sequencer fans a homogeneous covered-INSERT-intent wave out to. `1` (default) = the E2.3 serial
/// sequencer, byte-identical. `N>1` moves the per-item conflict check/record (per-shard private
/// integer ledger for intent-only slots; same-PK → same shard → single-winner 23505 preserved),
/// the value clone, and the WAL-record clone OFF the ordered critical section into N workers hashing
/// on the row's unique-slot key; the ordered WAL append + global commit-seq claim + device-append
/// buffer stay under ONE thin serial cut (the ~0.3-0.4us/item floor).
fn intent_sequencer_shards() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GPU_DB_INTENT_SEQUENCER_SHARDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1)
    })
}

/// E2.4a — minimum wave size before the sharded fan-out is worth its `std::thread::scope`
/// fork/join. Below this the serial sequencer runs (tiny waves are latency-bound, not
/// throughput-bound, and the thread hand-off would dominate).
const SHARD_MIN_WAVE: usize = 64;

/// E2.4a — the per-table batched device-append accumulator shape (rows / row-ids / (position,seq) /
/// per-row birth stamps). Shared by the serial and sharded sequencer paths so the wave-batched
/// open-shard append (one HtoD per table per wave) stays a single code path.
type WavePendingAppends = BTreeMap<
    String,
    (
        Vec<Vec<SqlValue>>,
        Vec<u64>,
        // (batch position, commit_seq, rows_affected) per buffered item (U1: the flush's
        // committed entries carry the item's exact applied row count through to the ack).
        Vec<(usize, Index, u64)>,
        Vec<Index>,
    ),
>;

/// E2.4a — one shard worker's verdict for a wave position: either a retryable/duplicate abort
/// (outcome set verbatim in the serial cut) or a Commit carrying the cloned row image and the
/// cloned W5a WAL record (row id still the encode-time placeholder; the serial cut patches it with
/// the wave-assigned id). Built entirely off the commit lock by [`Engine::shard_prepare_intents`].
enum ShardVerdict {
    Commit {
        table: String,
        values: Vec<SqlValue>,
        wal_record: Vec<u8>,
        wal_offset: usize,
    },
    Abort(ExecuteError),
}

/// E2.4a — the deterministic shard of a unique slot: a fibonacci-hash mix of the packed slot id and
/// the i32 value, folded to `[0, shards)`. Same `(slot_id, value)` → same shard, which is what keeps
/// two writers of the same unique slot on the same worker (single-winner conflict detection).
fn shard_index(slot_id: u64, value: i32, shards: usize) -> usize {
    let mixed = slot_id.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (value as u32 as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    (mixed % shards as u64) as usize
}

/// W2b — max sequenced-but-unpublished wave tails outstanding in the durability pipeline. Depth 1
/// forced one fsync per wave (the sequencer stalled at the gate for ~the full fdatasync — a
/// measured 14% durable-arm REGRESSION); at depth N consecutive waves' records coalesce into
/// SHARED fsyncs (the group covers everything appended while the previous flush was in flight),
/// so the fsync count per commit drops toward 1/N. Correctness does not depend on finish order:
/// a tail's durability wait covers all EARLIER WAL positions (prefix durability) and
/// `publish_committed_seq` is a CAS-max, so concurrent claimers may finish tails out of order.
/// The bound caps applied-but-unpublished state at N waves (restart recovery replays the durable
/// prefix; nothing unpublished was ever acked).
const WAVE_TAIL_PIPELINE_DEPTH_DEFAULT: u64 = 8;

/// Commit-wave telemetry (waves sequenced / items committed through waves / total sequencing
/// nanos). W2 CHANGED [2]'s meaning: the timer now stops when `sequence_commit_wave` returns —
/// the group-durability tail (fsync-wait + publish + acks) is PIPELINED off the sequencer and
/// is NOT included (pre-W2 numbers included it; comparisons across d6d10f8e are
/// apples-to-oranges). Three relaxed adds PER WAVE — not per item — so it stays on permanently;
/// the phase-D SLO benchmark reads it to report wave amortization (`items/waves`) alongside TPS.
pub static WAVE_STATS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// FUSE-recon device-phase timing (nanos summed across the run): [0]=wave-batch LOCATE (validate),
/// [1]=device APPEND (HtoD chunks), [2]=device INDEX-INSERT (atom.cas kernel). Relaxed adds gated on
/// `GPU_DB_BENCH_DEVPHASE=1` so they never touch the steady-state hot path; the SLO benchmark reads
/// them to attribute the per-wave device round-trip cost and pick the fusion target.
pub static WAVE_DEVICE_STATS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// True when `GPU_DB_BENCH_DEVPHASE=1` — enables the [`WAVE_DEVICE_STATS`] per-phase timing.
pub(crate) fn wave_device_phase_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("GPU_DB_BENCH_DEVPHASE").is_ok_and(|v| v == "1"))
}

/// HOST-sequencer per-item phase timing (nanos summed): the serial work under the commit_mutex,
/// which the driver-call probe showed is the PEAK-throughput wall (not device round-trips). Indices:
/// [0]=wave_batch_validate (design-B batched unique, whole call incl. its device locate),
/// [1]=SI conflict check, [2]=under-lock re-resolve (prepare_dml), [3]=sequence (WAL append +
/// propose + wait_committed), [4]=ledger record, [5]=apply (apply_delta / fast-run / append flush),
/// [6]=residency invalidate. Gated on `GPU_DB_BENCH_HOSTPHASE=1` so it never touches steady state.
pub static WAVE_HOST_STATS: [std::sync::atomic::AtomicU64; 7] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// True when `GPU_DB_BENCH_HOSTPHASE=1` — enables the [`WAVE_HOST_STATS`] per-phase timing.
pub(crate) fn wave_host_phase_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("GPU_DB_BENCH_HOSTPHASE").is_ok_and(|v| v == "1"))
}

/// One enqueued concurrent commit: everything the sequencer needs to conflict-check, re-resolve,
/// append, apply, and publish it — plus the shared slot its owner blocks on.
pub(crate) struct CommitWaveItem {
    txn_id: u64,
    cmd: Command,
    payload: std::sync::Arc<[u8]>,
    write_set: WriteSet,
    read_snapshot: Index,
    residency_tables: BTreeSet<String>,
    /// Ledger #18 audit fix: the catalog generation the OFF-LOCK prepare validated against.
    /// The sequencer grants the ledger-covered re-resolve skip ONLY while the live catalog
    /// still carries this stamp — a constraint-adding DDL (ADD UNIQUE/CHECK) committing
    /// between the snapshot and the wave records NOTHING in the conflict ledger, so the skip
    /// would silently bypass the new constraint; any DDL bumps the stamp and forces the
    /// always-correct Full re-validation instead.
    prepared_catalog_seq: Index,
    /// DELTA-REUSE (B): the OFF-LOCK-prepared insert delta, carried forward for reuse-eligible
    /// items (elided, FK-free, no nextval). The under-lock re-resolve then only RE-KEYS it at the
    /// wave's `next_row_id` (`rekey_offlock_insert_delta`) instead of re-running the full
    /// coerce+validate+write-set rebuild — the coerced values / write-set are input-deterministic,
    /// so they are identical to a fresh re-prepare while the catalog generation still matches
    /// (`prepared_catalog_seq`; a DDL bump forces the Full path, dropping the reuse). `None` = the
    /// item takes the normal `prepare_dml` re-resolve.
    offlock_delta: Option<crate::write_path::WriteDelta>,
    /// E2.2(b) — the PRE-ENCODED W5a binary WAL record, built OFF the sequencer at intent-build
    /// time as a pure function of `(route, params)` with a PLACEHOLDER row id, plus the fixed byte
    /// offset of that row id. Present only for single-row covered-INSERT intents. The sequencer
    /// patches the 8-byte row id at `offset` with the wave-assigned id (no String row-key parse, no
    /// per-item `encode_relational_row` + `try_encode_binary_insert`) and uses the result verbatim
    /// as the reuse-eligible delta's WAL payload. `None` = the classic per-item encode path.
    binary_wal_template: Option<(std::sync::Arc<[u8]>, u32)>,
    outcome: CommitWaveOutcome,
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
    done: std::sync::atomic::AtomicBool,
    result: Mutex<Option<Result<u64, ExecuteError>>>,
}

impl CommitWaveDone {
    pub(crate) fn take_if_done(&self) -> Option<Result<u64, ExecuteError>> {
        if !self.done.load(AtomicOrdering::Acquire) {
            return None;
        }
        self.result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

/// E2.5b-2 LEAN LANE ITEM: everything the lane pump needs, ~112B + the value
/// row — vs the ~500B CommitWaveItem plus its AST/delta/text attachments. The
/// pump's host passes were the measured final wall (~6.5ms/lane cycle of cold
/// cache traffic at ~925-item waves); this struct is the fix.
pub(crate) struct LaneIntent {
    pub(crate) txn_id: u64,
    pub(crate) slot: crate::write_path::IntUniqueSlotKey,
    pub(crate) read_snapshot: Index,
    pub(crate) prepared_catalog_seq: Index,
    pub(crate) filter_idx: u32,
    pub(crate) row_id_offset: u32,
    pub(crate) table: std::sync::Arc<str>,
    pub(crate) template: std::sync::Arc<[u8]>,
    pub(crate) values: Vec<SqlValue>,
    pub(crate) outcome: CommitWaveOutcome,
    /// Live-population decrement handle (see `IntentLaneState::outstanding`);
    /// None outside lanes mode.
    pub(crate) outstanding: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
    /// PostgreSQL `synchronous_commit` model: true (default) acks at the
    /// STRICT gate (durable AND applied); false acks at the APPLIED cut
    /// (async commit — bounded loss on power failure, consistency preserved).
    pub(crate) synchronous: bool,
    /// U1: the rows-affected count a settled Ok reports (INSERT intents = 1;
    /// DELETE intents that reach the wave located exactly one live row).
    pub(crate) rows_affected: u64,
}

impl LaneIntent {
    pub(crate) fn set_outcome(&self, result: Result<u64, ExecuteError>) {
        *self
            .outcome
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(result);
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
    })
}

impl CommitWaveItem {
    fn set_outcome(&self, result: Result<u64, ExecuteError>) {
        *self
            .outcome
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(result);
        self.outcome.done.store(true, AtomicOrdering::Release);
    }
}

/// The deterministic commit-wave state (ledger #6): the arrival-ordered item queue, the
/// single-sequencer election flag, and the sticky wedge. The condvar doubles as the completion
/// signal for waiters, the promotion signal for the next sequencer, and (W2) the tail-finished
/// signal for the depth-1 durability pipeline.
pub(crate) struct CommitWaveState {
    pub(crate) queue: Mutex<CommitWaveQueue>,
    pub(crate) cv: std::sync::Condvar,
    /// W2/W2b — the DURABILITY PIPELINE (depth [`WAVE_TAIL_PIPELINE_DEPTH`]): sequenced waves
    /// whose group-fsync wait, `committed_seq` publish, and outcome acks have NOT yet run. The
    /// sequencer pushes tails here and immediately drains/sequences the NEXT wave under the
    /// commit_mutex; tails are FINISHED (fsync-wait → publish → acks) by whichever threads claim
    /// them — the waves' own blocked waiters (they are spinning on their outcomes anyway) or the
    /// sequencer as the fallback claimer at the capacity gate. Finish order is UNCONSTRAINED:
    /// a tail's durability wait covers all earlier WAL positions (prefix frontier) and the
    /// publish is a CAS-max, so concurrent out-of-order finishing is safe. This is what lets
    /// wave N+1's serial sequencing overlap wave N's fdatasync, and lets consecutive waves'
    /// records coalesce into SHARED fsyncs via the WAL's group-flush protocol.
    /// Liveness: a pending tail always has ≥1 live claimer — its members' outcomes are unset
    /// until it finishes, so they are by definition still in the waiter loops (which probe this
    /// deque), and the sequencer try-claims before ever blocking on the capacity gate.
    pub(crate) pending_tails: Mutex<std::collections::VecDeque<CommitWaveTail>>,
    /// Tails handed to the pipeline slot / tails fully finished. `handed == finished` ⇔ the
    /// pipeline is empty (the depth-1 gate the sequencer enforces before handing a new tail).
    pub(crate) tails_handed: AtomicU64,
    pub(crate) tails_finished: AtomicU64,
}

impl Default for CommitWaveState {
    fn default() -> Self {
        Self {
            queue: Mutex::new(CommitWaveQueue::default()),
            cv: std::sync::Condvar::new(),
            pending_tails: Mutex::new(std::collections::VecDeque::new()),
            tails_handed: AtomicU64::new(0),
            tails_finished: AtomicU64::new(0),
        }
    }
}

/// W2 — one sequenced-but-not-yet-durable wave: everything needed to finish it OFF the
/// sequencer's critical path. The deltas are applied and the WAL records appended (that is what
/// lets the next wave's re-resolves see them); nothing is client-visible until `finish` runs the
/// group-durability wait and publishes `committed_seq` (WAL-before-visibility per tail; the
/// CAS-max publish makes out-of-order tail completion safe). `armed` keeps the wedge-don't-strand policy:
/// a tail dropped unfinished (claimer panic, pipeline abandonment) fails every still-unset
/// outcome and wedges the queue, exactly like `CommitWaveBatchGuard` does for the in-section
/// half of the wave.
pub(crate) struct CommitWaveTail {
    batch: Vec<CommitWaveItem>,
    /// `(batch position, commit_seq, appended, rows_affected)` for every item that reached the
    /// durable-commit point, in wave order (aborted items' outcomes were already set in-section).
    /// `rows_affected` is the applied delta's exact row count — the Ok payload of the ack (U1).
    committed: Vec<(usize, Index, bool, u64)>,
    last_seq: Index,
    last_position: usize,
    armed: bool,
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
                item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                    "the concurrent commit path is wedged pending restart recovery: the \
                     commit-wave durability tail died before completing"
                        .to_string(),
                ))));
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct CommitWaveQueue {
    items: std::collections::VecDeque<CommitWaveItem>,
    sequencer_active: bool,
    /// Sticky: a wave failed after its deltas were applied (durability failure mid-wave). No
    /// further concurrent commits may run until restart recovery.
    wedged: Option<String>,
}

/// Fails every still-unset outcome in a wave batch if the sequencer dies mid-wave (the
/// apply-invariant panic path), and wedges the queue so waiters and future committers error out
/// instead of hanging. Forgotten (`std::mem::forget`) on the successful path.
struct CommitWaveBatchGuard<'a> {
    engine: &'a Engine,
    items: &'a [CommitWaveItem],
}

impl Drop for CommitWaveBatchGuard<'_> {
    fn drop(&mut self) {
        let mut queue = self.engine.lock_commit_wave_queue();
        let reason = queue
            .wedged
            .clone()
            .unwrap_or_else(|| "commit-wave sequencer died mid-wave".to_string());
        queue.wedged = Some(reason.clone());
        queue.sequencer_active = false;
        // Fail everything still queued too — no sequencer will ever run it.
        let stranded: Vec<CommitWaveItem> = queue.items.drain(..).collect();
        drop(queue);
        for item in self.items.iter().chain(stranded.iter()) {
            if !item.outcome.done.load(AtomicOrdering::Acquire) {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(format!(
                    "the concurrent commit path is wedged pending restart recovery: {reason}"
                )))));
            }
        }
        self.engine.commit_wave.cv.notify_all();
    }
}

impl Engine {
    /// Whether `text` is a DML statement (`INSERT`/`UPDATE`/`DELETE` on an existing base table whose
    /// columns carry no `nextval` sequence default) that the **concurrent** commit path can execute
    /// via off-lock prepare + the short commit critical section (write-half MVCC, Stage 4). Anything
    /// else — DDL, KV, sequence-default INSERTs, transaction control, parse errors, unknown tables —
    /// returns `false` and the caller routes it through the SERIALIZED `execute_text` under the
    /// catalog latch. Conservative by construction: it never returns `true` for a statement the
    /// concurrent path can't faithfully execute (a wrong "yes" only ever means a serialized fallback,
    /// never a wrong result — but here a wrong "yes" would mis-route, so the checks are exact).
    pub fn is_concurrent_dml(&self, text: &str) -> bool {
        let Ok(cmd) = parse_command(text) else {
            return false;
        };
        let table_name = match &cmd {
            Command::Insert(insert) => &insert.table,
            Command::Update(update) => &update.table,
            Command::Delete(delete) => &delete.table,
            _ => return false,
        };
        // Lock-free concurrent-DML classify (Stage 2 — blocker #1): probe the pinned catalog snapshot.
        let catalog = self.catalog_snapshot();
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return false;
        };
        // INSERTs that evaluate a `nextval` column default mutate sequence state, which is not
        // interior-mutable; route those through the serialized path (which applies the advance under
        // `&mut self`). UPDATE/DELETE never touch sequences, so they are always eligible.
        if matches!(cmd, Command::Insert(_)) {
            let touches_sequence_default = table.columns.iter().any(|column| {
                matches!(column.default, Some(ColumnDefault::SequenceNextVal { .. }))
            });
            if touches_sequence_default {
                return false;
            }
        }
        true
    }

    /// Register a transaction's read snapshot (its `read_snapshot` `commit_seq`) for the
    /// oldest-active GC/ledger-prune boundary, returning a guard that deregisters on drop (write-half
    /// MVCC, Stage 4). Done off-lock at prepare-begin so taking a snapshot never serializes on the
    /// commit_mutex.
    pub(crate) fn register_active_snapshot(&self, snapshot: Index) -> ActiveSnapshotGuard<'_> {
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .register(snapshot);
        ActiveSnapshotGuard {
            engine: self,
            snapshot,
        }
    }

    pub(crate) fn deregister_active_snapshot(&self, snapshot: Index) {
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .deregister(snapshot);
    }

    /// Execute one autocommit DML statement (`INSERT`/`UPDATE`/`DELETE`) on the CONCURRENT commit
    /// path under Snapshot Isolation (write-half MVCC, Stage 4 — the concurrency flip):
    ///
    /// 1. **Begin (off-lock):** pin a read snapshot `S = committed_seq` and register it.
    /// 2. **Prepare (off-lock, no commit_mutex):** parse, constraint-preflight against `S`, and
    ///    compute the conflict write-set (`prepare_*` at `S`). Many writers run this concurrently,
    ///    and concurrently with lock-free readers.
    /// 3. **Commit (short critical section under the commit_mutex):** validate the write-set against
    ///    the recent-commits ledger (overlap since `S` ⇒ retryable [`ExecuteError::Serialization`],
    ///    first-committer-wins) → assign `commit_seq` (the commit `Index`) → WAL append + group-commit
    ///    fsync → install the delta RE-RESOLVED at `commit_seq` (so the live apply is byte-identical
    ///    to a WAL replay) + publish the table generation → record the write-set in the ledger → bump
    ///    `committed_seq` LAST (release-store: the publish point).
    /// 4. **Abort/retry:** a conflict (or any prepare error) publishes nothing and is returned; a
    ///    serialization conflict is retryable with a fresh snapshot.
    ///
    /// `&self`: the whole path runs without an engine write lock, so writers overlap on prepare and
    /// serialize only briefly on the commit_mutex, and a writer never blocks a reader.
    pub fn execute_dml_concurrent(&self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        self.intent_lanes_write_guard()
            .map_err(ExecuteError::Engine)?;
        self.execute_dml_concurrent_instrumented(txn_id, text, || {})
    }

    /// [`Engine::execute_dml_concurrent`] with a hook invoked AFTER the off-lock snapshot capture +
    /// prepare but BEFORE the commit critical section. The concurrency-correctness suite uses this to
    /// rendezvous two writers at a barrier between snapshot and commit, deterministically forcing the
    /// SI write-write conflict window (both read the same snapshot, then both try to commit) — the
    /// lost-update exit criterion. The production entry point passes an empty hook, so this is a
    /// zero-overhead extraction of the real path, not a separate code path.
    pub fn execute_dml_concurrent_instrumented(
        &self,
        txn_id: u64,
        text: &str,
        on_prepared: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        // (1) Begin: pin + register the read snapshot for the off-lock prepare.
        let read_snapshot = self.committed_seq();
        let _snapshot_guard = self.register_active_snapshot(read_snapshot);

        // (2) Prepare OFF-LOCK at the read snapshot: validate constraints + compute the conflict
        // write-set. (The delta itself is recomputed at commit_seq under the lock so the live apply
        // matches a WAL replay; this off-lock pass is the expensive validation + the write-set.)
        // P2 (write-path assessment): the separate `preflight_unique_index_constraints` pass this
        // path used to run first was a full duplicate of the validation `prepare_dml` performs —
        // same validator fns, same error messages — but at a WORSE visibility boundary (the facade
        // txn id rather than the pinned read snapshot), and it cost an extra O(table)
        // materialization per statement. The prepare below is the authoritative off-lock
        // validation; the serialized `execute_text` path keeps its preflight, where it is the gate
        // that stops a constraint-violating statement from ever reaching the WAL.
        // RETIREMENT A4e GAP-1 guard: ELIDED (device-authoritative) tables take the SERIALIZED
        // path — its commit arm carries the elision lifecycle hooks (elide-entry, rehydrate-on-
        // unhandled). The concurrent arm has none yet: an unhandled concurrent commit would
        // invalidate + re-admit from the EMPTY host store. Concurrent-native elision hooks are
        // the ledgered follow-up (they are what the SLO target ultimately needs).
        if self.host_install_elision_enabled()
            && !matches!(cmd, Command::Insert(_))
            && Self::dml_mutated_tables(&cmd)
                .iter()
                .any(|table| self.table_install_elided(table))
        {
            drop(_snapshot_guard);
            return self.execute_text(txn_id, text).map(|_| ());
        }
        // Ledger #18 audit fix: capture the catalog generation BEFORE the prepare — the
        // prepare's own catalog bind is at least this fresh, so a stamp match at re-resolve
        // proves no constraint-adding DDL landed since the off-lock validation (a capture
        // AFTER prepare could miss a DDL slipping between the bind and the capture).
        let prepared_catalog_seq = self.catalog_snapshot().commit_seq;
        let snapshot = self.dml_read_snapshot(read_snapshot);
        let prepared = self.prepare_dml(&cmd, snapshot, InsertPrepareValidation::Full)?;
        let residency_tables = Self::dml_mutated_tables(&cmd);

        // The snapshot is now pinned and prepare is done; the commit critical section has not started.
        // (Tests barrier here to align two writers' snapshots before their commits race.)
        on_prepared();

        // DELTA-REUSE (B): carry the whole delta forward for reuse-eligible (elided, no-nextval)
        // inserts so the under-lock re-resolve only re-keys it; every other item keeps just its
        // write-set (the conflict path reads it BEFORE the re-resolve, so it is always needed).
        let write_set = prepared.write_set.clone();
        let offlock_delta = Self::reresolve_reuse_eligible(&prepared).then_some(prepared);

        // (3) Commit: enqueue into the deterministic commit WAVE (ledger #6) and wait for the
        // sequencer to durably commit + publish it (or abort it with a retryable conflict).
        self.commit_dml_concurrent(
            txn_id,
            cmd,
            text,
            write_set,
            read_snapshot,
            residency_tables,
            prepared_catalog_seq,
            offlock_delta,
        )
    }

    /// Off-lock prepare dispatch: run the pure `prepare_*` for a DML command against `snapshot`.
    /// `insert_validation` = `Full` off-lock (the authoritative validation);
    /// `ReResolveLedgerCovered` only from the sequencer's under-lock re-resolve (ledger #18 —
    /// the coverage proof lives on [`InsertPrepareValidation`]).
    fn prepare_dml(
        &self,
        cmd: &Command,
        snapshot: DmlReadSnapshot,
        insert_validation: InsertPrepareValidation,
    ) -> Result<WriteDelta, ExecuteError> {
        let delta = match cmd {
            Command::Insert(insert) => {
                self.prepare_insert(insert, snapshot, None, insert_validation)
            }
            Command::Update(update) => self.prepare_update(update, snapshot),
            Command::Delete(delete) => self.prepare_delete(delete, snapshot),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "execute_dml_concurrent received a non-DML command".to_string(),
                )))
            }
        }?;
        Ok(delta)
    }

    /// DELTA-REUSE (B): is this off-lock-prepared delta safe to REUSE at the under-lock re-resolve
    /// (re-key only) instead of re-preparing? An INSERT with (a) NO nextval advances (those route
    /// through the serialized path) and (b) an EMPTY value-index. An empty value-index is the exact
    /// ground truth that `prepare_insert` skipped it because the table was ELIDED at prepare (a
    /// non-elided insert of >=1 row into a >=1-column table always yields >=1 entry) — and an elided
    /// table is FK-free + CHECK-free by `table_elision_eligible`, so the re-resolve owes no
    /// constraint re-validation beyond the ledger. The generation gate (ReResolveLedgerCovered) is
    /// enforced at reuse time, so a post-prepare DDL (e.g. ADD FK) still forces the Full path.
    fn reresolve_reuse_eligible(delta: &crate::write_path::WriteDelta) -> bool {
        matches!(
            &delta.mutation,
            crate::write_path::PreparedMutation::Insert { seq_advances, value_index_entries, .. }
                if seq_advances.is_empty() && value_index_entries.is_empty()
        )
    }

    /// DELTA-REUSE (B): rebuild a reuse-eligible off-lock INSERT delta at `snapshot`'s
    /// `next_row_id`, recomputing ONLY the per-row keys (the sole `next_row_id`-dependent output).
    /// The coerced VALUES, the (value-derived) `write_set`, `rows_consumed`, and the empty
    /// value-index / seq-advances are input-deterministic, so under a matched catalog generation
    /// this equals a fresh `prepare_insert` re-resolve — minus the re-coerce + not-null + write-set
    /// rebuild. (value-index stays empty; the apply recomputes it if the table de-elided —
    /// `value_index_entries_for_deferred_apply`.)
    fn rekey_offlock_insert_delta(
        delta: &crate::write_path::WriteDelta,
        snapshot: DmlReadSnapshot,
    ) -> crate::write_path::WriteDelta {
        let crate::write_path::PreparedMutation::Insert {
            table,
            inserted_rows,
            value_index_entries,
            seq_advances,
        } = &delta.mutation
        else {
            unreachable!("reresolve_reuse_eligible gates this to inserts");
        };
        let rekeyed: Vec<(String, Vec<SqlValue>)> = inserted_rows
            .iter()
            .enumerate()
            .map(|(offset, (_stale_key, values))| {
                (
                    relational_row_key(table, snapshot.next_row_id + offset as u64),
                    values.clone(),
                )
            })
            .collect();
        crate::write_path::WriteDelta {
            write_set: delta.write_set.clone(),
            rows_consumed: delta.rows_consumed,
            mutation: crate::write_path::PreparedMutation::Insert {
                table: table.clone(),
                inserted_rows: rekeyed,
                value_index_entries: value_index_entries.clone(),
                seq_advances: seq_advances.clone(),
            },
        }
    }

    /// The set of tables a DML command mutates (for per-table residency invalidation on commit).
    fn dml_mutated_tables(cmd: &Command) -> BTreeSet<String> {
        let mut tables = BTreeSet::new();
        match cmd {
            Command::Insert(insert) => {
                tables.insert(insert.table.clone());
            }
            Command::Update(update) => {
                tables.insert(update.table.clone());
            }
            Command::Delete(delete) => {
                tables.insert(delete.table.clone());
            }
            _ => {}
        }
        tables
    }

    /// Enqueue one prepared concurrent DML commit into the deterministic commit WAVE and block
    /// until a sequencer completes it (ledger #6 / ADR-009 host spine — "the order IS the log").
    ///
    /// Committers no longer take the commit_mutex per statement. They push a fully-prepared wave
    /// item and either (a) become the SEQUENCER — the one thread that drains everything queued as
    /// a WAVE and commits it under a single commit_mutex hold — or (b) wait on the shared condvar
    /// until a sequencer completes their item. Any waiter can be PROMOTED to sequencer when the
    /// active one steps down (after its own item completes), so queued items are never
    /// leaderless. Wave order = queue arrival order = WAL/log order = apply order; within a wave
    /// there are NO ordering aborts (each item's re-resolve sees every earlier item's applied
    /// delta) — aborts remain only for genuine SI conflicts against state committed after an
    /// item's read snapshot, and for constraint phantoms at its commit seq (both retryable,
    /// pre-durable, side-effect-free).
    ///
    /// vs. the previous per-commit critical section: one mutex hand-off + one group-fsync wait +
    /// one publish PER WAVE instead of per commit, and the apply loop runs back-to-back on one
    /// core (the sequencer) instead of bouncing the MVCC structures across every writer's cache.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_dml_concurrent(
        &self,
        txn_id: u64,
        cmd: Command,
        text: &str,
        write_set: WriteSet,
        read_snapshot: Index,
        residency_tables: BTreeSet<String>,
        prepared_catalog_seq: Index,
        offlock_delta: Option<crate::write_path::WriteDelta>,
    ) -> Result<(), ExecuteError> {
        let item = CommitWaveItem {
            txn_id,
            cmd,
            payload: std::sync::Arc::from(text.as_bytes()),
            prepared_catalog_seq,
            offlock_delta,
            binary_wal_template: None,
            write_set,
            read_snapshot,
            residency_tables,
            outcome: Arc::new(CommitWaveDone::default()),
        };
        let outcome = self.enqueue_commit_wave_item(item)?;
        // Blocking client: become the sequencer or spin/park on our own outcome, running the
        // pipeline's pending tails as a fallback claimer (the classic per-statement blocking arm).
        // (U1: the classic blocking APIs keep their `()` signature — rows-affected surfaces via
        // the intent path; the count is dropped here, not fabricated.)
        if let Some(result) = self.pump_as_sequencer_if_idle(&outcome) {
            return result.map(|_| ());
        }
        self.await_commit_wave_outcome(&outcome).map(|_| ())
    }

    /// E2.2(c) — build a covered-INSERT wave item (the intent fast path's per-statement item),
    /// carrying the reuse delta, integer conflict slots (already in `write_set`), and the
    /// pre-encoded W5a binary template. Kept in this module so [`CommitWaveItem`]'s fields stay
    /// private; the intent module supplies the pure-function inputs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn make_covered_insert_wave_item(
        &self,
        txn_id: u64,
        cmd: Command,
        text: &str,
        write_set: WriteSet,
        read_snapshot: Index,
        residency_tables: BTreeSet<String>,
        prepared_catalog_seq: Index,
        offlock_delta: Option<crate::write_path::WriteDelta>,
        binary_wal_template: Option<(std::sync::Arc<[u8]>, u32)>,
    ) -> CommitWaveItem {
        CommitWaveItem {
            txn_id,
            cmd,
            payload: std::sync::Arc::from(text.as_bytes()),
            prepared_catalog_seq,
            offlock_delta,
            binary_wal_template,
            write_set,
            read_snapshot,
            residency_tables,
            outcome: Arc::new(CommitWaveDone::default()),
        }
    }

    /// E2.2(c) — enqueue a built item and BLOCK until it commits (the intent path's blocking arm,
    /// identical wait machinery to [`Engine::commit_dml_concurrent`]).
    pub(crate) fn commit_wave_item_blocking(
        &self,
        item: CommitWaveItem,
    ) -> Result<(), ExecuteError> {
        // Lanes-mode seq-collision guard: the classic sequencer must not run
        // concurrently with activated lanes (oracle vs repl seqs). Blocking
        // classic-shaped items are refused post-activation like all classic
        // writes; the lean submit path is the lanes-mode entry.
        if let Some(lanes) = &self.intent_lanes {
            if lanes.activated.load(std::sync::atomic::Ordering::Acquire) {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "intent lanes are ACTIVE: blocking classic-path commits are refused \
                     (submit_covered_insert_intent is the lanes-mode write entry)"
                        .to_string(),
                )));
            }
        }
        let outcome = self.enqueue_commit_wave_item(item)?;
        if let Some(result) = self.pump_as_sequencer_if_idle(&outcome) {
            return result.map(|_| ());
        }
        self.await_commit_wave_outcome(&outcome).map(|_| ())
    }

    /// E2.2(c) — enqueue a built item WITHOUT blocking, returning its completion slot (the
    /// driver-multiplexed submit path). The caller advances the pipeline via
    /// [`Engine::drive_commit_wave`] and observes completion via [`CommitWaveOutcome::take_if_done`].
    pub(crate) fn submit_commit_wave_item(
        &self,
        item: CommitWaveItem,
    ) -> Result<CommitWaveOutcome, ExecuteError> {
        // E2.5b-2 lean path: lanes-mode intents enter via build_lane_intent in
        // submit_covered_insert_intent (LaneIntent queues); classic-shaped
        // items always take the classic queue (pre-activation only).
        self.enqueue_commit_wave_item(item)
    }

    /// Enqueue one already-built wave item (non-blocking). Returns its completion slot, or the
    /// wedge error if the concurrent path is wedged pending recovery. The single shared push point
    /// for the blocking commit arm and the E2.2(c) async submit path.
    fn enqueue_commit_wave_item(
        &self,
        item: CommitWaveItem,
    ) -> Result<CommitWaveOutcome, ExecuteError> {
        let outcome = Arc::clone(&item.outcome);
        let mut queue = self.lock_commit_wave_queue();
        if let Some(reason) = &queue.wedged {
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "the concurrent commit path is wedged pending restart recovery: {reason}"
            ))));
        }
        queue.items.push_back(item);
        Ok(outcome)
    }

    /// E2.2(c) — the driver-multiplexed pipeline pump: advance the commit wave by ONE unit of work
    /// without blocking on any particular outcome. If no sequencer is active and work is queued,
    /// promote this thread to sequence a drain; otherwise claim a pending durability tail. Returns
    /// whether it did work. N event-loop drivers call this in their poll loops so M logical clients
    /// share a few OS threads with no per-commit park/wake — the disruptor ingress the mandate
    /// calls for. `take_if_done` on a submitted ticket observes the result.
    pub fn drive_commit_wave(&self) -> bool {
        // E2.5b-2: in lanes mode a pump call advances one lane's pipeline (round-robin);
        // the classic wave machinery below still services pre-activation traffic.
        if let Some(lanes) = &self.intent_lanes {
            let lanes = std::sync::Arc::clone(lanes);
            self.maybe_resize_lanes(&lanes);
            let lane = (lanes
                .pump_cursor
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                % lanes.lane_count as u64) as usize;
            if self.drive_intent_lane(lane) {
                return true;
            }
        }
        // Prefer claiming a pending tail (cheap, unblocks acks) before taking sequencer duty.
        if self.try_finish_pending_wave_tail() {
            return true;
        }
        let promote = {
            let mut queue = match self.commit_wave.queue.try_lock() {
                Ok(queue) => queue,
                Err(_) => return false,
            };
            if !queue.sequencer_active && !queue.items.is_empty() && queue.wedged.is_none() {
                queue.sequencer_active = true;
                true
            } else {
                false
            }
        };
        if promote {
            // Drive one full drain cycle. `own_outcome` is a never-completing sentinel: the
            // sequencer steps down on an empty queue (or when this dummy is "done", which it never
            // is), so this returns after draining everything currently queued.
            let sentinel: CommitWaveOutcome = Arc::new(CommitWaveDone::default());
            self.run_commit_wave_sequencer(&sentinel);
            return true;
        }
        false
    }

    /// If no sequencer is running, promote THIS thread to drain+sequence waves until `own_outcome`
    /// completes, then return its result. Returns `None` if a sequencer is already active (the
    /// caller should wait) or if the promotion ran but our outcome is not yet set (a pipelined
    /// tail we did not claim — fall through to the waiter loop).
    fn pump_as_sequencer_if_idle(
        &self,
        own_outcome: &CommitWaveOutcome,
    ) -> Option<Result<u64, ExecuteError>> {
        let promote = {
            let mut queue = self.lock_commit_wave_queue();
            if !queue.sequencer_active {
                queue.sequencer_active = true;
                true
            } else {
                false
            }
        };
        if promote {
            self.run_commit_wave_sequencer(own_outcome);
            return own_outcome.take_if_done();
        }
        None
    }

    /// Block until `own_outcome` completes: spin (claiming pending tails + promoting a leaderless
    /// queue), then fall back to the condvar. The tail of the classic blocking commit path.
    fn await_commit_wave_outcome(
        &self,
        own_outcome: &CommitWaveOutcome,
    ) -> Result<u64, ExecuteError> {
        let outcome: &CommitWaveOutcome = own_outcome;
        loop {
            // Spin first: under load a wave completes within tens of µs, far cheaper to poll than
            // to pay a futex sleep + wake per commit. The periodic promotion probe keeps queued
            // items from ever being leaderless (the previous sequencer may have stepped down
            // between our enqueue and our first probe). W2: the same probe CLAIMS the pipeline's
            // pending durability tail — a waiter whose own wave was pipelined is the natural
            // thread to run its fsync-wait + publish + acks (including its own).
            for spin in 0..4096_u32 {
                if let Some(result) = outcome.take_if_done() {
                    return result;
                }
                if spin % 64 == 63 {
                    self.try_finish_pending_wave_tail();
                    if let Some(result) = outcome.take_if_done() {
                        return result;
                    }
                    if let Ok(mut queue) = self.commit_wave.queue.try_lock() {
                        if !queue.sequencer_active && !queue.items.is_empty() {
                            queue.sequencer_active = true;
                            drop(queue);
                            self.run_commit_wave_sequencer(outcome);
                        }
                    }
                }
                std::hint::spin_loop();
            }
            // Fall back to the condvar (arrival lulls / oversubscribed cores).
            let mut queue = self.lock_commit_wave_queue();
            loop {
                if let Some(result) = outcome.take_if_done() {
                    return result;
                }
                // W2: claim the pending tail before sleeping — the sequencer's hand-off
                // notify_all wakes this loop, and the claim may complete our own outcome.
                drop(queue);
                if self.try_finish_pending_wave_tail() {
                    queue = self.lock_commit_wave_queue();
                    continue;
                }
                queue = self.lock_commit_wave_queue();
                if let Some(result) = outcome.take_if_done() {
                    return result;
                }
                if !queue.sequencer_active {
                    if queue.items.is_empty() {
                        break;
                    }
                    queue.sequencer_active = true;
                    drop(queue);
                    self.run_commit_wave_sequencer(outcome);
                    break;
                }
                queue = self
                    .commit_wave
                    .cv
                    .wait(queue)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
    }

    fn lock_commit_wave_queue(&self) -> std::sync::MutexGuard<'_, CommitWaveQueue> {
        self.commit_wave
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The promoted sequencer: drain-and-commit WAVES until this thread's own item is done (then
    /// hand leadership off) or the queue is empty. Wave size self-tunes to the arrival rate —
    /// everything queued while the previous wave was committing forms the next wave (bounded by
    /// `COMMIT_WAVE_MAX` per drain to keep worst-case wave latency bounded).
    ///
    /// W2/W2b — the durability pipeline: a sequenced wave's tail (fsync-wait → publish → acks)
    /// is handed to `pending_tails` instead of running inline, so this loop's next iteration
    /// sequences wave N+1 while claimers (the waves' own blocked waiters, or this thread at the
    /// capacity gate) run the fdatasyncs. Consecutive waves' records coalesce into shared
    /// fsyncs via the WAL's group-flush protocol (the sequencer keeps appending while a flush
    /// is in flight). The capacity gate bounds applied-but-unpublished state at
    /// [`WAVE_TAIL_PIPELINE_DEPTH`] waves; post-publish auto_admit re-admissions stay HERE
    /// (behind a clean full-drain barrier) so a stale re-admission can never interleave with a
    /// later wave's apply (the internal form of ledger #26).
    fn run_commit_wave_sequencer(&self, own_outcome: &CommitWaveOutcome) {
        loop {
            let batch: Vec<CommitWaveItem> = {
                let mut queue = self.lock_commit_wave_queue();
                let n = queue.items.len().min(commit_wave_max());
                if n == 0 {
                    queue.sequencer_active = false;
                    drop(queue);
                    // Liveness on the empty-queue step-down: never leave pipelined tails
                    // behind with no sequencer (their member waiters would still claim them,
                    // but finishing inline here is strictly sooner and keeps the single-writer
                    // case's latency identical to the pre-pipeline path). Tails already claimed
                    // by an in-flight waiter finish on that waiter.
                    while self.try_finish_pending_wave_tail() {}
                    // W1b: idle moment — the cheap auto-checkpoint bound probe (see the wave
                    // counter probe above for the sustained-load path).
                    self.maybe_auto_checkpoint_wal();
                    self.commit_wave.cv.notify_all();
                    return;
                }
                queue.items.drain(..n).collect()
            };
            let wave_started = std::time::Instant::now();
            let wave_len = batch.len() as u64;
            let tail = self.sequence_commit_wave(batch);
            let waves = WAVE_STATS[0].fetch_add(1, AtomicOrdering::Relaxed);
            // W1b: periodic auto-checkpoint probe OFF the critical section (the rotation takes
            // the commit_mutex itself). Every 256 waves keeps the under-bound check (one mutex
            // lock + field read) off the per-wave path; sustained load can never starve
            // rotation, and the empty-queue step-down below covers bursty loads.
            if waves % 256 == 255 {
                self.maybe_auto_checkpoint_wal();
            }
            WAVE_STATS[1].fetch_add(wave_len, AtomicOrdering::Relaxed);
            WAVE_STATS[2].fetch_add(
                wave_started.elapsed().as_nanos() as u64,
                AtomicOrdering::Relaxed,
            );
            if let Some((tail, admit_tables)) = tail {
                // W2b capacity gate: at most WAVE_TAIL_PIPELINE_DEPTH tails outstanding —
                // consecutive waves' records coalesce into shared fsyncs while the bound caps
                // applied-but-unpublished state. Claim tails ourselves when the pipe is full
                // (under load the shared fsync already covered them and the finishes are
                // instant).
                self.wait_wave_tail_capacity(wave_tail_pipeline_depth() - 1);
                {
                    let mut tails = self
                        .commit_wave
                        .pending_tails
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    tails.push_back(tail);
                }
                self.commit_wave
                    .tails_handed
                    .fetch_add(1, AtomicOrdering::Release);
                {
                    // Wake spinners/sleepers so a member waiter can claim the tail while we
                    // drain + sequence the next wave.
                    let _queue = self.lock_commit_wave_queue();
                    self.commit_wave.cv.notify_all();
                }
                if !admit_tables.is_empty() {
                    // Rare (auto_admit ON + a committed item whose device append declined):
                    // the re-admission must observe THIS wave's publish and must not overlap a
                    // later wave's apply — drain the WHOLE pipeline, then re-admit, then
                    // continue. AUDIT d6d10f8e E: on a WEDGED early-return the drain is NOT
                    // complete (an in-flight tail may still legitimately publish after the
                    // wedge via the durable-frontier fast path) — admitting then would install
                    // a fresh valid snapshot gathered BELOW that late publish, serving stale
                    // reads in the wedged-but-still-readable state. Skip the admit; residency
                    // stays invalidated, which is always correct.
                    if self.wait_wave_tail_capacity(0) {
                        self.auto_admit_resident_tables(&admit_tables);
                    }
                }
            } else {
                let _queue = self.lock_commit_wave_queue();
                self.commit_wave.cv.notify_all();
            }
            if own_outcome.done.load(AtomicOrdering::Acquire) {
                // Step down so this client thread can return; a waiter with a still-queued item
                // (or the next arrival) promotes itself. A pending tail may remain — its member
                // waiters claim it (their outcomes are unset, so they are still in the waiter
                // loops by definition).
                let mut queue = self.lock_commit_wave_queue();
                queue.sequencer_active = false;
                drop(queue);
                self.commit_wave.cv.notify_all();
                return;
            }
        }
    }

    /// W2/W2b — the sequencer's pipeline gate: block until at most `max_outstanding` handed
    /// tails remain unfinished, claiming pending ones ourselves first (the common single-writer
    /// / idle-waiter case finishes them inline; under load member waiters usually already did
    /// during our sequencing). `0` = full drain (the admit barrier). Returns `true` when the
    /// capacity condition genuinely holds; `false` on the WEDGED early-return, where in-flight
    /// tails may still complete (and even publish, via the durable-frontier fast path) after
    /// this returns — callers needing a REAL barrier (the admit arm) must treat `false` as
    /// "barrier not established".
    fn wait_wave_tail_capacity(&self, max_outstanding: u64) -> bool {
        loop {
            let handed = self.commit_wave.tails_handed.load(AtomicOrdering::Acquire);
            let finished = self
                .commit_wave
                .tails_finished
                .load(AtomicOrdering::Acquire);
            if handed.saturating_sub(finished) <= max_outstanding {
                return true;
            }
            if self.try_finish_pending_wave_tail() {
                continue;
            }
            // Claimers are mid-flight (took the tails, still fsyncing): wait for their signal.
            let queue = self.lock_commit_wave_queue();
            if queue.wedged.is_some() {
                // Defensive: a wedged path never completes tails normally; the completion
                // guard's counter bump is the primary unblock, this is the belt-and-braces.
                return false;
            }
            let finished = self
                .commit_wave
                .tails_finished
                .load(AtomicOrdering::Acquire);
            let handed = self.commit_wave.tails_handed.load(AtomicOrdering::Acquire);
            if handed.saturating_sub(finished) <= max_outstanding {
                return true;
            }
            let _queue = self
                .commit_wave
                .cv
                .wait(queue)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Commit one WAVE: the per-item (3a)-(3e) steps of the old per-commit critical section, run
    /// back-to-back under ONE commit_mutex hold in wave order. W2: the durability tail (group
    /// fsync wait + `committed_seq` publish + acks) is RETURNED as a [`CommitWaveTail`] (plus the
    /// sequencer-owned post-publish admit set) instead of running inline, so the caller can
    /// pipeline it against the next wave's sequencing. `None` = every item aborted pre-durable
    /// (outcomes already set). Every item's outcome slot is set exactly once; the
    /// `CommitWaveBatchGuard` fails any still-unset outcome (and wedges the queue) if this
    /// thread panics mid-wave (e.g. the apply-invariant panic, which also poisons the
    /// commit_mutex — the established wedge-don't-serve-torn-state policy); once the tail is
    /// built, its `armed` Drop carries that responsibility.
    fn sequence_commit_wave(
        &self,
        batch: Vec<CommitWaveItem>,
    ) -> Option<(CommitWaveTail, BTreeSet<String>)> {
        // AUDIT f80f2350 FINDING A/B: the entire wave (conflict-check, re-resolve, flush,
        // apply, publish) runs inside the commit critical section — flag it so any lock-aware
        // rehydrate seam reached from the re-resolve's validator ladder
        // (`rehydrate_elided_serialized`) takes its DIRECT branch instead of re-locking the
        // commit_mutex this thread already holds.
        self.skip_leader_check_during_internal_read(|engine| {
            engine.sequence_commit_wave_inner(batch)
        })
    }

    // (helper is a free fn below the impl)

    /// M1 design B: WAVE-TIME batched PK-unique validation. For every eligible SINGLE-ROW INSERT
    /// whose off-lock unique check was DEFERRED (`insert_unique_wave_batchable`, re-derived here
    /// under the catalog-generation gate), batch the whole wave's PK needles per (table, column)
    /// into ONE device locate (`wave_batch_locate_hit_counts`): count==0 -> no visible dup, pass;
    /// count>0 -> the authoritative per-item `visible_row_with_value` (a tombstoned/invisible
    /// slot passes there). Returns the byte-identical 23505 message per violating item position.
    ///
    /// CATALOG DRIFT (a constraint-adding DDL committed since the item's prepare, gen mismatch):
    /// the item MIGHT have been deferred but its eligibility can't be re-derived, so full-validate
    /// it now (redundant if it wasn't deferred, safe either way — DDL mid-wave is rare). SAME-wave
    /// dups are caught by the unique-slot conflict ledger (#18), NOT here; this catches
    /// ALREADY-COMMITTED dups. A locate DECLINE / non-batchable item -> per-item full validation.
    fn wave_batch_validate_unique(
        &self,
        batch: &[CommitWaveItem],
    ) -> std::collections::BTreeMap<usize, String> {
        let mut violations: std::collections::BTreeMap<usize, String> =
            std::collections::BTreeMap::new();
        if !self.device_write_locate_wave_batch_enabled() {
            return violations;
        }
        let catalog = self.catalog_snapshot();
        // PERF (this fn runs SERIALLY on the sequencer, so per-item host work must be tiny — the
        // off-lock path it replaced ran 32-way parallel): NO per-item String allocs. Distinct
        // tables are cached once (a wave is usually one table); groups key on the DISTINCT-table
        // INDEX + filter_idx (integers); the 23505 index name is looked up only on the rare
        // violation. All table refs borrow the pinned `catalog`.
        struct TableCtx<'c> {
            name: &'c str,
            table: &'c RelationalTable,
            // (filter_idx, unique-index ordinal) for each strictly-i32 unique index; empty = not
            // eligible (its inserts take the full-validate fallback / were validated off-lock).
            unique_cols: Vec<(usize, usize)>,
        }
        let mut tables: Vec<TableCtx> = Vec::new();
        // group key = (distinct-table index, filter_idx) -> (needles, positions).
        let mut group_keys: Vec<(usize, usize)> = Vec::new();
        let mut group_needles: Vec<Vec<i32>> = Vec::new();
        let mut group_positions: Vec<Vec<usize>> = Vec::new();
        let mut full_validate: Vec<usize> = Vec::new();

        for (pos, item) in batch.iter().enumerate() {
            let Command::Insert(insert) = &item.cmd else {
                continue;
            };
            if insert.rows.len() != 1 {
                continue;
            }
            // Locate (or bind + cache) this insert's distinct-table context.
            let tctx_idx = match tables.iter().position(|t| t.name == insert.table) {
                Some(i) => i,
                None => {
                    let Some(table) = catalog.relational_catalog.get(&insert.table) else {
                        continue;
                    };
                    let eligible = self.insert_unique_wave_batchable(&catalog, table);
                    let unique_cols = if eligible {
                        table
                            .indexes
                            .iter()
                            .enumerate()
                            .filter(|(_, index)| index.unique)
                            .filter_map(|(ord, index)| {
                                table
                                    .columns
                                    .iter()
                                    .position(|c| c.name == index.column)
                                    .map(|fi| (fi, ord))
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    tables.push(TableCtx {
                        name: &insert.table,
                        table,
                        unique_cols,
                    });
                    tables.len() - 1
                }
            };
            let gen_matches = catalog.commit_seq == item.prepared_catalog_seq;
            if !gen_matches {
                // Catalog drift: might have deferred off-lock -> full-validate to be safe.
                full_validate.push(pos);
                continue;
            }
            if tables[tctx_idx].unique_cols.is_empty() {
                continue; // not eligible -> off-lock validated it
            }
            // Eligible + gen-matched -> it was deferred. Bind each unique needle (no alloc).
            let table = tables[tctx_idx].table;
            let mut bound_all = true;
            let cols = tables[tctx_idx].unique_cols.clone();
            for (filter_idx, _ord) in &cols {
                let Some((_, needle)) = insert_i32_unique_needle_at(insert, table, *filter_idx)
                else {
                    bound_all = false;
                    break;
                };
                // Find/create the (tctx_idx, filter_idx) group.
                let gk = (tctx_idx, *filter_idx);
                let gi = match group_keys.iter().position(|k| *k == gk) {
                    Some(i) => i,
                    None => {
                        group_keys.push(gk);
                        group_needles.push(Vec::new());
                        group_positions.push(Vec::new());
                        group_keys.len() - 1
                    }
                };
                group_needles[gi].push(needle);
                group_positions[gi].push(pos);
            }
            if !bound_all {
                full_validate.push(pos);
            }
        }
        // Batched device locate per group; count==0 passes, count>0 authoritative-checks.
        for (gi, &(tctx_idx, filter_idx)) in group_keys.iter().enumerate() {
            let table = tables[tctx_idx].table;
            let locate_started = wave_device_phase_timing_enabled().then(Instant::now);
            let locate = self.wave_batch_locate_hit_counts(table, filter_idx, &group_needles[gi]);
            if let Some(started) = locate_started {
                WAVE_DEVICE_STATS[0]
                    .fetch_add(started.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
            }
            match locate {
                Some(counts) => {
                    for ((&pos, &needle), &count) in group_positions[gi]
                        .iter()
                        .zip(group_needles[gi].iter())
                        .zip(counts.iter())
                    {
                        if count == 0 {
                            continue; // the common case: no physical slot holds the key
                        }
                        // Rare: >0 hits -> authoritative visibility+value check.
                        let visibility = crate::StorageVisibility {
                            read_txn_id: batch[pos].read_snapshot,
                        };
                        if self
                            .visible_row_with_value(
                                table,
                                visibility,
                                filter_idx,
                                &SqlValue::Int4(needle),
                                None,
                            )
                            .unwrap_or(false)
                        {
                            // Look up the index name ONLY now (rare) for the byte-identical 23505.
                            let index_name = table
                                .indexes
                                .iter()
                                .find(|idx| {
                                    idx.unique
                                        && table.columns.iter().position(|c| c.name == idx.column)
                                            == Some(filter_idx)
                                })
                                .map(|idx| idx.name.as_str())
                                .unwrap_or("");
                            violations.entry(pos).or_insert_with(|| {
                                format!(
                                    "duplicate key value violates unique index \"{index_name}\""
                                )
                            });
                        }
                    }
                }
                None => full_validate.extend(group_positions[gi].iter().copied()),
            }
        }
        // Full validation for drifted / declined / unbindable inserts (rare).
        for pos in full_validate {
            if violations.contains_key(&pos) {
                continue;
            }
            let Command::Insert(insert) = &batch[pos].cmd else {
                continue;
            };
            if catalog.relational_catalog.get(&insert.table).is_none() {
                continue;
            }
            let snapshot = self.dml_read_snapshot(batch[pos].read_snapshot);
            if let Err(err) =
                self.prepare_insert(insert, snapshot, None, InsertPrepareValidation::Full)
            {
                violations.insert(pos, err.to_string());
            }
        }
        violations
    }

    fn sequence_commit_wave_inner(
        &self,
        batch: Vec<CommitWaveItem>,
    ) -> Option<(CommitWaveTail, BTreeSet<String>)> {
        // E2.4a VARIANT 1 — shared-WAL sharded sequencing. When the whole wave is homogeneous
        // covered-INSERT intents (the flagship OLTP shape), fan the expensive per-item prep
        // (conflict check/record, value + WAL-record clones) out to N parallel shard workers and
        // keep only the ordered WAL append + global commit-seq claim + device-append buffer under a
        // thin serial cut. A mixed wave (any non-intent / classic item) keeps the fully-serial path
        // below, byte-identical: the sharded conflict verdict is computed from a shared-ledger
        // SNAPSHOT + a per-shard private dedup set, which is only equivalent to the serial
        // record-as-you-go ledger when no in-wave classic write can slip a same-slot record between
        // the snapshot and the serial cut (a homogeneous-intent wave has none).
        let shards = intent_sequencer_shards();
        if shards > 1 && batch.len() >= SHARD_MIN_WAVE && self.auto_admit_on_commit_enabled() {
            let wave_catalog_seq = self.catalog_snapshot().commit_seq;
            if self.device_write_locate_wave_batch_enabled()
                && batch
                    .iter()
                    .all(|item| self.item_sharded_intent_eligible(item, wave_catalog_seq))
            {
                return self.sequence_commit_wave_sharded(batch, shards);
            }
        }
        let guard = CommitWaveBatchGuard {
            engine: self,
            items: &batch,
        };
        let wall_clock = current_timestamp_micros();
        let mut wave_tail: Option<(Index, usize)> = None;
        let mut committed: Vec<(usize, Index, bool, u64)> = Vec::with_capacity(batch.len());
        // Homogeneous fast-INSERT run accumulator (ADR-009's homogeneous-wave shape): consecutive
        // constraint-free INSERTs are installed together by ONE `with_table_mut` per table at run
        // flush. Their re-resolves never read table rows (constraint-free) and their row keys come
        // from the VIRTUAL row-id cursor below, so deferring the install is invisible; any
        // non-fast item flushes the run first so its re-resolve sees every prior wave write.
        let mut fast_run: Vec<(usize, Index, String, WriteDelta)> = Vec::new();
        let auto_admit = self.auto_admit_on_commit_enabled();
        let mut fast_table_cache: BTreeMap<String, bool> = BTreeMap::new();
        // The virtual row-id cursor: fast-run deltas are prepared against this cursor (their
        // installs — which advance the real allocator — are deferred to the run flush).
        let mut next_row_id = self.read_state.mvcc.current_row_id();

        let mut commit = self.commit_state();
        let flush_fast_run =
            |commit: &mut CommitState,
             fast_run: &mut Vec<(usize, Index, String, WriteDelta)>,
             committed: &mut Vec<(usize, Index, bool, u64)>| {
                if fast_run.is_empty() {
                    return;
                }
                let _ = commit; // the commit_mutex guard is held by the caller for the whole wave
                let mut by_table: BTreeMap<String, Vec<(WriteDelta, Index)>> = BTreeMap::new();
                let mut run_meta: Vec<(usize, Index, String, u64)> =
                    Vec::with_capacity(fast_run.len());
                for (position, seq, table, delta) in fast_run.drain(..) {
                    run_meta.push((position, seq, table.clone(), delta.rows_affected()));
                    by_table.entry(table).or_default().push((delta, seq));
                }
                for (table, deltas) in by_table {
                    self.apply_insert_deltas_batched(&table, deltas)
                        .unwrap_or_else(|err| {
                            panic!(
                                "commit-path invariant violation: batched wave apply on {table} \
                             failed after re-validation succeeded: {err}"
                            )
                        });
                }
                for (position, seq, _table, rows) in run_meta {
                    commit.repl.mark_applied(seq);
                    self.invalidate_relational_residency_tables_concurrent(
                        &batch[position].residency_tables,
                        batch[position].txn_id,
                        seq,
                    );
                    committed.push((position, seq, false, rows));
                }
            };
        // A4e OPTIMIZATION: wave-BATCHED residency appends. The measured elision residual was
        // the PER-ITEM device append (~27us/item = several small HtoD copies + bookkeeping per
        // single-row INSERT). Consecutive INSERT items buffer here per table and flush as ONE
        // `try_append_resident_int4_open_shard` call per (table, flush) — same rows, same order,
        // ~items/wave fewer launch sets. Flush points: before any NON-insert item's processing
        // (its device locate must see prior rows), and at the wave tail before publish (the
        // residency-before-publish invariant is per WAVE, not per item — rows become reader-
        // visible only at the tail publish either way). D3 (ADR-013 pre1, LANDED): each buffered
        // row carries its own birth stamp (`InsertPerRow`) since the batch spans multiple commit
        // seqs; the stamps + the hwm publish with the row_count bump at flush.
        let mut pending_appends: WavePendingAppends = BTreeMap::new();
        // E2.4a — the pending-append flush is a shared method (`flush_wave_pending_appends`) so the
        // serial and sharded sequencer paths keep ONE wave-batched open-shard append code path.
        let flush_appends =
            |pending: &mut WavePendingAppends, committed: &mut Vec<(usize, Index, bool, u64)>| {
                self.flush_wave_pending_appends(pending, committed, &batch);
            };
        // M1 design B: WAVE-TIME BATCHED PK-UNIQUE VALIDATION. Eligible INSERTs deferred their
        // unique check off-lock (`prepare_insert`); validate the whole wave here with ONE device
        // locate per (table, key-column) (the amortization win). Returns the item positions that
        // are unique violations -> aborted in the loop below with the byte-identical 23505.
        let hostphase = wave_host_phase_timing_enabled();
        let wave_validate_started = hostphase.then(Instant::now);
        let wave_unique_violations = self.wave_batch_validate_unique(&batch);
        if let Some(started) = wave_validate_started {
            WAVE_HOST_STATS[0]
                .fetch_add(started.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }
        // E2.3 — the catalog generation is CONSTANT for the whole wave: DDL is the only publisher
        // and it commits under the very commit_mutex this sequencer holds, so no generation bump can
        // interleave a wave's items. Load the snapshot ONCE here instead of per item (the old
        // per-item `catalog_snapshot()` was an ArcSwap load + Arc clone on every commit — the
        // generation gate at re-resolve, the fast-run eligibility probe, and the intent fast-lane
        // gate all read it). `wave_catalog_seq` is the ledger-#18 stamp every item compares against.
        let wave_catalog = self.catalog_snapshot();
        let wave_catalog_seq = wave_catalog.commit_seq;
        // HOST-phase probe: `_hp` timestamps the running phase boundary; `hp!(k)` charges the elapsed
        // time since the last boundary to WAVE_HOST_STATS[k] and resets. Reset at each item's top.
        let mut _hp = hostphase.then(Instant::now);
        macro_rules! hp {
            ($k:expr) => {
                if let Some(ref mut t) = _hp {
                    let now = Instant::now();
                    WAVE_HOST_STATS[$k].fetch_add(
                        now.duration_since(*t).as_nanos() as u64,
                        AtomicOrdering::Relaxed,
                    );
                    *t = now;
                }
            };
        }
        // Index-based (not `iter().enumerate()`): the intent fast lane and the general path both
        // reach `batch[position]` while the `flush_*` closures also borrow `batch` — an index keeps
        // those borrows disjoint per statement without threading an iterator through the closures.
        #[allow(clippy::needless_range_loop)]
        for position in 0..batch.len() {
            if let Some(ref mut t) = _hp {
                *t = Instant::now();
            }
            // M1 design B: a deferred INSERT whose PK value already exists (wave-batch verdict)
            // aborts here — the same 23505 the off-lock validation would have raised.
            if let Some(err) = wave_unique_violations.get(&position) {
                batch[position].set_outcome(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    err.clone(),
                ))));
                continue;
            }
            // Batched-append ORDER: a non-INSERT item's re-resolve (device locate) and its
            // tombstone paths must observe every prior row of this wave — flush first.
            if !matches!(batch[position].cmd, Command::Insert(_)) {
                flush_appends(&mut pending_appends, &mut committed);
            }
            // (3a) SI first-committer-wins: any key in the write-set committed after this item's
            // read snapshot aborts it (retryable). Earlier items in THIS wave recorded into the
            // ledger below, so intra-wave conflicts are caught here exactly like cross-wave ones.
            if commit
                .ledger
                .conflicts(&batch[position].write_set, batch[position].read_snapshot)
            {
                let read_snapshot = batch[position].read_snapshot;
                batch[position].set_outcome(Err(ExecuteError::Serialization(format!(
                    "write-write conflict on a key committed after read snapshot {read_snapshot}"
                ))));
                continue;
            }
            hp!(1);

            // E2.3 — INTENT INTEGER FAST LANE. A single-row covered-INSERT intent (pre-encoded
            // binary WAL template + reuse-eligible off-lock delta + catalog generation unchanged
            // since prepare + table still elided + auto-admit on) owes NO String row key: its row id
            // is the wave's integer `next_row_id`, its W5a record is patched in place, and its values
            // flow straight into the batched device append. This collapses the general path's
            // `rekey_offlock_insert_delta` (row-key `format!` + write-set/value clones) AND the
            // `insert_append` value-clone + String→u64 parse — the two top host buckets (reresolve,
            // apply) for the flagship shape — into one value clone + one WAL patch. Any drift (gen
            // bump, de-elision, auto-admit off, non-intent item) falls through to the always-correct
            // general path below, byte-identical to before.
            let intent_fast = auto_admit
                && wave_catalog_seq == batch[position].prepared_catalog_seq
                && batch[position].binary_wal_template.is_some()
                && matches!(&batch[position].offlock_delta, Some(d)
                    if Self::reresolve_reuse_eligible(d)
                        && matches!(&d.mutation,
                            crate::write_path::PreparedMutation::Insert { inserted_rows, .. }
                                if inserted_rows.len() == 1))
                && match &batch[position].cmd {
                    Command::Insert(insert) => self.table_install_elided(&insert.table),
                    _ => false,
                };
            if intent_fast {
                // Land any earlier deferred fast-run installs first so seq order == apply order
                // (intents are never themselves fast-run — they hold a unique slot — but a mixed
                // wave may have buffered plain inserts ahead of this one).
                flush_fast_run(&mut commit, &mut fast_run, &mut committed);
                let commit_seq = commit.repl.peek_next_index();
                // The row id is the wave's integer cursor — IDENTICAL to what the general path's
                // `rekey` would `format!` into `rel/{table}/{row_id:020}` and then parse back out.
                let row_id = next_row_id;
                // WAL: patch the pre-encoded W5a record's 8-byte row id at its fixed offset (no
                // key parse, no `encode_relational_row`, no `try_encode_binary_insert`).
                let wal_payload: std::sync::Arc<[u8]> = {
                    let (template, offset) = batch[position]
                        .binary_wal_template
                        .as_ref()
                        .expect("intent_fast requires a binary WAL template");
                    let off = *offset as usize;
                    // ONE copy: clone the template straight into the Arc allocation and patch the
                    // row id in place (the fresh Arc is unique). `to_vec()` + `Arc::from(vec)`
                    // was two full copies of every WAL record on the serial cut.
                    let mut payload: std::sync::Arc<[u8]> = std::sync::Arc::from(&template[..]);
                    std::sync::Arc::get_mut(&mut payload).expect("freshly created Arc is unique")
                        [off..off + 8]
                        .copy_from_slice(&row_id.to_le_bytes());
                    payload
                };
                let wal_len_before = commit.wal.len();
                commit.wal.append(WalRecord {
                    txn_id: batch[position].txn_id,
                    payload: wal_payload.clone(),
                });
                let wal_position = commit.wal.len();
                let token = match commit.repl.propose(wal_payload) {
                    Ok(token) => token,
                    Err(err) => {
                        commit.wal.truncate(wal_len_before);
                        batch[position].set_outcome(Err(ExecuteError::Engine(err)));
                        continue;
                    }
                };
                debug_assert_eq!(
                    token.index, commit_seq,
                    "the sequencer is the single proposer: the proposed index must equal the peek"
                );
                if let Err(err) = commit.repl.wait_committed(token, Duration::from_millis(0)) {
                    commit.repl.rollback_unapplied_from(commit_seq);
                    commit.wal.truncate(wal_len_before);
                    batch[position].set_outcome(Err(ExecuteError::Engine(err)));
                    continue;
                }
                let timestamp_micros =
                    wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
                commit.record_commit_timestamp(batch[position].txn_id, timestamp_micros);
                hp!(3);
                commit.ledger.record(&batch[position].write_set, commit_seq);
                hp!(4);
                // Elided apply == advance the row-id allocator (host store skipped) + the elision
                // counter, exactly `apply_delta`'s elided-insert branch for one row. Clone the row
                // image + table out of the carried delta straight into the batched append (one value
                // clone total, vs the general path's two + the String round-trip).
                let (table, values) = {
                    let delta = batch[position]
                        .offlock_delta
                        .as_ref()
                        .expect("intent_fast requires an off-lock delta");
                    let crate::write_path::PreparedMutation::Insert {
                        table,
                        inserted_rows,
                        ..
                    } = &delta.mutation
                    else {
                        unreachable!("intent_fast gates to single-row inserts");
                    };
                    (table.clone(), inserted_rows[0].1.clone())
                };
                self.read_state.mvcc.advance_row_id(1);
                self.read_state
                    .residency
                    .host_install_elisions
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                commit.repl.mark_applied(commit_seq);
                next_row_id = self.read_state.mvcc.current_row_id();
                let entry = pending_appends.entry(table).or_default();
                entry.3.push(commit_seq);
                entry.0.push(values);
                entry.1.push(row_id);
                // Intent fast-lane items are single-row covered INSERTs by eligibility.
                entry.2.push((position, commit_seq, 1));
                wave_tail = Some((commit_seq, wal_position));
                hp!(5);
                continue;
            }
            let item = &batch[position];

            // (3b) Re-resolve at the peeked commit seq — sees every PRIOR wave item's applied
            // delta (they are installed already), so wave order is the only order there is. A
            // failure is a legal concurrent interleaving (constraint phantom): retryable,
            // pre-durable, side-effect-free.
            let commit_seq = commit.repl.peek_next_index();
            let install_snapshot = DmlReadSnapshot {
                commit_seq,
                next_row_id,
            };
            // AUDIT f80f2350 FINDING D: capture elided-ness BEFORE the re-resolve — a
            // constrained INSERT's validator ladder can REHYDRATE on a device-probe decline
            // (de-eliding the table mid-wave), and the rehydration gather reads only the
            // DEVICE + the pre-elision store, so this wave's still-BUFFERED appends are
            // invisible to it (they exist nowhere until the tail flush, whose unhandled
            // recovery is elided-gated and would now skip). Detected below, repaired with the
            // flush's own upsert convention.
            let insert_table = match &item.cmd {
                Command::Insert(insert) => Some(insert.table.clone()),
                _ => None,
            };
            let was_elided = insert_table
                .as_deref()
                .is_some_and(|table| self.table_install_elided(table));
            // Ledger #18: FK-free INSERT re-resolves skip the redundant unique/CHECK pass —
            // the conflicts() check above IS the commit-time guard (coverage proof on
            // InsertPrepareValidation) — but ONLY while the catalog generation still matches
            // the off-lock prepare's (audit fix): a constraint-adding DDL committed since S
            // records nothing in the ledger and the item's write_set lacks slots for the new
            // index, so the skip would silently bypass it. Any DDL bumps the stamp -> Full
            // (always correct; DDL is rare so the hot path keeps the skip).
            let insert_validation = if wave_catalog_seq == item.prepared_catalog_seq {
                InsertPrepareValidation::ReResolveLedgerCovered
            } else {
                InsertPrepareValidation::Full
            };
            // DELTA-REUSE (B): a reuse-eligible elided insert whose catalog generation still
            // matches (ReResolveLedgerCovered) owes no re-validation — RE-KEY the off-lock delta
            // at the wave's `next_row_id` instead of re-coercing + rebuilding it. A generation
            // drift (Full) or a non-eligible item falls through to the authoritative re-prepare.
            let prepared = match &item.offlock_delta {
                Some(delta)
                    if insert_validation == InsertPrepareValidation::ReResolveLedgerCovered =>
                {
                    Ok(Self::rekey_offlock_insert_delta(delta, install_snapshot))
                }
                _ => self.prepare_dml(&item.cmd, install_snapshot, insert_validation),
            };
            if let Some(table_name) = insert_table.as_deref() {
                if was_elided && !self.table_install_elided(table_name) {
                    // Mid-re-resolve de-elision: reconcile the buffered same-table rows into
                    // the freshly rehydrated store (repair runs even when the re-resolve
                    // errored — the de-elision happened and the earlier items' hole exists
                    // regardless). The tail flush still appends them to the device.
                    if let Some((rows, row_ids, items_meta, _stamps)) =
                        pending_appends.get(table_name)
                    {
                        if !rows.is_empty() {
                            let first_seq = items_meta
                                .first()
                                .map(|(_, seq, _)| *seq)
                                .unwrap_or_default();
                            let last_seq = items_meta
                                .last()
                                .map(|(_, seq, _)| *seq)
                                .unwrap_or_default();
                            let upserts: std::collections::BTreeMap<u64, Vec<SqlValue>> =
                                row_ids.iter().copied().zip(rows.iter().cloned()).collect();
                            let catalog_table = self
                                .relational_catalog_table(table_name)
                                .expect("a just-rehydrated table is in the catalog");
                            self.rehydrate_elided_table(
                                &catalog_table,
                                first_seq.saturating_sub(1),
                                &upserts,
                                &Default::default(),
                                last_seq,
                            )
                            .unwrap_or_else(|err| {
                                panic!(
                                    "commit-path invariant violation: mid-wave de-elision \
                                     repair on {table_name} failed: {err}"
                                )
                            });
                        }
                    }
                }
            }
            let delta = match prepared {
                Ok(delta) => delta,
                Err(err) => {
                    item.set_outcome(Err(match err {
                        ExecuteError::Serialization(_) => err,
                        other => ExecuteError::Serialization(format!(
                            "re-resolve at commit_seq {commit_seq} failed on a concurrent \
                             interleaving (retryable): {other}"
                        )),
                    }));
                    continue;
                }
            };
            hp!(2);

            // (3c) Assign the seq for real: WAL append + propose (the sequencer is the single
            // proposer under the commit_mutex). The fsync is deferred to the wave tail.
            // W5a: covered inserts (the delta-reuse class — elided, FK/CHECK-free, no sequence
            // defaults, ledger-covered uniqueness) log the RESOLVED BINARY record instead of the
            // SQL text: replay becomes decode+install (no parse, no re-resolve), the record
            // carries the ORIGINAL row ids, and checkpoints shrink. Everything else keeps the
            // SQL-text payload unchanged.
            let wal_payload: std::sync::Arc<[u8]> = if self.binary_wal_records_enabled()
                && Self::reresolve_reuse_eligible(&delta)
            {
                let crate::write_path::PreparedMutation::Insert {
                    table,
                    inserted_rows,
                    ..
                } = &delta.mutation
                else {
                    unreachable!("reuse-eligible is insert-shaped");
                };
                // E2.2(b): a single-row covered-INSERT intent carries its W5a record PRE-ENCODED
                // (built off the sequencer at intent-build time). The only wave-time-dependent
                // field is the row id, at a fixed offset — patch it in place instead of parsing the
                // row key + re-encoding the row image. The reuse re-key assigns the single row's id
                // as `next_row_id + 0`, i.e. this item's `install_snapshot.next_row_id`.
                match &item.binary_wal_template {
                    Some((template, offset)) if inserted_rows.len() == 1 => {
                        let row_id = install_snapshot.next_row_id;
                        debug_assert_eq!(
                            crate::engine_residency::parse_relational_row_id(
                                &inserted_rows[0].0,
                                &relational_key_prefix(table),
                            ),
                            Some(row_id),
                            "pre-encoded intent row id must equal the re-keyed delta's row id"
                        );
                        let mut bytes = template.to_vec();
                        let off = *offset as usize;
                        bytes[off..off + 8].copy_from_slice(&row_id.to_le_bytes());
                        bytes.into()
                    }
                    _ => {
                        let prefix = relational_key_prefix(table);
                        let id_rows: Vec<(u64, &[SqlValue])> = inserted_rows
                            .iter()
                            .map(|(key, values)| {
                                (
                                    crate::engine_residency::parse_relational_row_id(key, &prefix)
                                        .expect("re-keyed insert rows carry canonical row keys"),
                                    values.as_slice(),
                                )
                            })
                            .collect();
                        match try_encode_binary_insert(table, &id_rows) {
                            Some(payload) => payload.into(),
                            // Width-exceeding shape (unrealistic; audit 21eddaa7 C): keep the text.
                            None => item.payload.clone(),
                        }
                    }
                }
            } else {
                item.payload.clone()
            };
            let wal_len_before = commit.wal.len();
            commit.wal.append(WalRecord {
                txn_id: item.txn_id,
                payload: wal_payload.clone(),
            });
            let wal_position = commit.wal.len();
            let token = match commit.repl.propose(wal_payload) {
                Ok(token) => token,
                Err(err) => {
                    commit.wal.truncate(wal_len_before);
                    item.set_outcome(Err(ExecuteError::Engine(err)));
                    continue;
                }
            };
            debug_assert_eq!(
                token.index, commit_seq,
                "the sequencer is the single proposer: the proposed index must equal the peek"
            );
            if let Err(err) = commit.repl.wait_committed(token, Duration::from_millis(0)) {
                commit.repl.rollback_unapplied_from(commit_seq);
                commit.wal.truncate(wal_len_before);
                item.set_outcome(Err(ExecuteError::Engine(err)));
                continue;
            }
            let timestamp_micros =
                wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
            commit.record_commit_timestamp(item.txn_id, timestamp_micros);
            hp!(3);

            // (3e) Record the write-set for future conflict detection (also read by LATER items
            // in this same wave — the intra-wave conflict path above).
            commit.ledger.record(&item.write_set, commit_seq);
            hp!(4);

            // Fast-run eligibility: a plain INSERT into a table with no unique index, no CHECK,
            // and no FK (its re-resolve read no rows and claimed no unique slots), with auto-admit
            // off (the in-place resident append is a per-item protocol). Everything else is a
            // SLOW item: flush the pending run first so this item's apply-order matches seq order
            // and later re-resolves see it.
            let fast_table = matches!(item.cmd, Command::Insert(_))
                && !auto_admit
                && delta.write_set.unique_slots.is_empty()
                && delta.write_set.unique_slots_i32.is_empty()
                && match &delta.mutation {
                    crate::write_path::PreparedMutation::Insert { table, .. } => {
                        *fast_table_cache.entry(table.clone()).or_insert_with(|| {
                            wave_catalog.relational_catalog.get(table).is_some_and(|t| {
                                !t.indexes.iter().any(|index| index.unique)
                                    && t.check_constraints.is_empty()
                                    && t.foreign_keys.is_empty()
                            })
                        })
                    }
                    _ => false,
                };
            if fast_table {
                let crate::write_path::PreparedMutation::Insert { table, .. } = &delta.mutation
                else {
                    unreachable!("fast_table guarantees an insert delta");
                };
                next_row_id += delta.rows_consumed;
                fast_run.push((position, commit_seq, table.clone(), delta));
                wave_tail = Some((commit_seq, wal_position));
                continue;
            }
            flush_fast_run(&mut commit, &mut fast_run, &mut committed);

            // (3d) SLOW item: install the re-validated delta now (apply-before-durable, D3b). A
            // failure here is a true invariant violation — PANIC, poisoning the commit_mutex; the
            // batch guard fails the wave's remaining outcomes and wedges the queue.
            // RETIREMENT A1: carry the inserted rows' host identities (parsed from their keys) so
            // the residency append can stamp the row-identity region.
            let insert_append: Option<(String, Vec<Vec<SqlValue>>, Vec<u64>)> =
                match &delta.mutation {
                    crate::write_path::PreparedMutation::Insert {
                        table,
                        inserted_rows,
                        ..
                    } => {
                        let prefix = relational_key_prefix(table);
                        Some((
                            table.clone(),
                            inserted_rows
                                .iter()
                                .map(|(_key, values)| values.clone())
                                .collect(),
                            inserted_rows
                                .iter()
                                .map(|(key, _)| {
                                    crate::engine_residency::parse_relational_row_id(key, &prefix)
                                        .unwrap_or(u64::MAX)
                                })
                                .collect(),
                        ))
                    }
                    _ => None,
                };
            let item_rows = delta.rows_affected();
            self.apply_delta(delta, commit_seq, None)
                .unwrap_or_else(|err| {
                    panic!(
                        "commit-path invariant violation: apply at commit_seq {commit_seq} failed \
                     after re-validation at this seq succeeded: {err}"
                    )
                });
            commit.repl.mark_applied(commit_seq);
            next_row_id = self.read_state.mvcc.current_row_id();
            hp!(5);

            // Residency, before publish: INSERT rows BUFFER into the wave-batched append (the
            // flush handles elide-entry / rehydrate / invalidate per table); everything else
            // invalidates conservatively as before.
            wave_tail = Some((commit_seq, wal_position));
            match insert_append {
                Some((table, rows, row_ids)) if self.auto_admit_on_commit_enabled() => {
                    let entry = pending_appends.entry(table).or_default();
                    // D3: one birth stamp per row of THIS item (the flush spans commit seqs).
                    entry
                        .3
                        .extend(std::iter::repeat(commit_seq).take(rows.len()));
                    entry.0.extend(rows);
                    entry.1.extend(row_ids);
                    entry.2.push((position, commit_seq, item_rows));
                }
                _ => {
                    self.invalidate_relational_residency_tables_concurrent(
                        &item.residency_tables,
                        item.txn_id,
                        commit_seq,
                    );
                    committed.push((position, commit_seq, false, item_rows));
                }
            }
            hp!(6);
        }
        flush_appends(&mut pending_appends, &mut committed);
        flush_fast_run(&mut commit, &mut fast_run, &mut committed);
        // Prune the ledger once per wave (was per commit) below the oldest active snapshot.
        if let Some((last_seq, _)) = wave_tail {
            let prune_boundary = self
                .active_snapshots
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .oldest()
                .map(|oldest| oldest.saturating_sub(1))
                .unwrap_or(last_seq);
            commit.ledger.prune_below(prune_boundary);
        }
        // === leave the commit critical section BEFORE the fsync (D3b group commit) ===
        drop(commit);

        let Some((last_seq, last_position)) = wave_tail else {
            // Every item aborted pre-durable; outcomes are already set.
            std::mem::forget(guard);
            return None;
        };

        // W2: the durability tail (fsync-wait → publish → acks) no longer runs on the
        // sequencer's critical path — it is handed back as a `CommitWaveTail` for the depth-1
        // pipeline, so the NEXT wave's sequencing overlaps THIS wave's fdatasync. The batch
        // guard's responsibility transfers to the tail's own `armed` Drop.
        // The post-publish auto_admit re-admissions stay a SEQUENCER duty (running them from an
        // arbitrary claimer thread could interleave a stale re-admission with the next wave's
        // apply — the internal form of ledger #26): collect the tables here; the sequencer waits
        // for this tail and runs them before draining the next wave (rare — only !appended
        // committed items with auto_admit ON).
        let mut admit_tables: BTreeSet<String> = BTreeSet::new();
        if self.auto_admit_on_commit_enabled() {
            for (position, _seq, appended, _rows) in &committed {
                if !appended {
                    admit_tables.extend(batch[*position].residency_tables.iter().cloned());
                }
            }
        }
        std::mem::forget(guard);
        Some((
            CommitWaveTail {
                batch,
                committed,
                last_seq,
                last_position,
                armed: true,
            },
            admit_tables,
        ))
    }

    /// A4e / E2.4a — flush the wave-batched device open-shard append: ONE
    /// `try_append_resident_int4_open_shard` per (table, flush) with per-row birth stamps (the
    /// batch spans commit seqs), the lazy elide-entry on first successful append, and the
    /// rehydrate-on-unhandled / invalidate fallback when the device declines. Extracted from the
    /// serial sequencer's inner closure so the sharded sequencer shares the exact same append path.
    fn flush_wave_pending_appends(
        &self,
        pending: &mut WavePendingAppends,
        committed: &mut Vec<(usize, Index, bool, u64)>,
        batch: &[CommitWaveItem],
    ) {
        if pending.is_empty() {
            return;
        }
        for (table, (rows, row_ids, items, stamps)) in std::mem::take(pending) {
            // D3 (ADR-013 pre1): the batched flush spans MULTIPLE commit seqs — each row
            // carries its own birth stamp (the per-row slice the D3-COMPOSE note called for).
            let appended = self.auto_admit_on_commit_enabled()
                && self.try_append_resident_int4_open_shard(
                    &table,
                    &rows,
                    crate::engine_residency::AppendCreatedBy::InsertPerRow(&stamps),
                    Some(&row_ids),
                );
            if appended {
                // A4e elide-entry (audit B1 eligibility), once per flushed table.
                if self.host_install_elision_enabled() && !self.table_install_elided(&table) {
                    let snapshot = self.catalog_snapshot();
                    if self.table_elision_eligible(&snapshot, &table) {
                        self.set_table_install_elided(&table, true);
                    }
                }
            } else {
                // A4e rehydrate-on-unhandled: the batch's rows were never installed (elided
                // apply skip) NOR appended — they ride the rehydration as upserts over the
                // gather at the batch's first seq - 1 (device state is complete through it:
                // flushes happen in seq order).
                if self.table_install_elided(&table) {
                    let first_seq = items.first().map(|(_, seq, _)| *seq).unwrap_or_default();
                    let last_seq = items.last().map(|(_, seq, _)| *seq).unwrap_or_default();
                    let upserts: std::collections::BTreeMap<u64, Vec<SqlValue>> =
                        row_ids.iter().copied().zip(rows.iter().cloned()).collect();
                    let catalog_table = self
                        .relational_catalog_table(&table)
                        .expect("an elided table is in the catalog");
                    self.rehydrate_elided_table(
                        &catalog_table,
                        first_seq.saturating_sub(1),
                        &upserts,
                        &Default::default(),
                        last_seq,
                    )
                    .unwrap_or_else(|err| {
                        panic!(
                            "commit-path invariant violation: elided rehydration for the \
                             batched append on {table} failed: {err}"
                        )
                    });
                }
                for (position, seq, _rows) in &items {
                    self.invalidate_relational_residency_tables_concurrent(
                        &batch[*position].residency_tables,
                        batch[*position].txn_id,
                        *seq,
                    );
                }
            }
            for (position, seq, rows) in items {
                committed.push((position, seq, appended, rows));
            }
        }
    }

    /// E2.4a — is `item` a covered-INSERT intent the SHARDED sequencer can fan out? The exact serial
    /// `intent_fast` gate (catalog generation unchanged since prepare, pre-encoded binary WAL
    /// template, reuse-eligible single-row off-lock delta, table still elided) PLUS a single
    /// integer unique slot and no other conflict dimension. The single-slot restriction is what
    /// makes "hash the unique slot → shard" a CORRECT same-conflict-slot-same-worker partition: a
    /// row with two unique columns could collide with a different row on its SECOND column while
    /// hashing to a different shard, so those (and every non-intent item) stay on the serial path.
    fn item_sharded_intent_eligible(&self, item: &CommitWaveItem, wave_catalog_seq: Index) -> bool {
        wave_catalog_seq == item.prepared_catalog_seq
            && item.binary_wal_template.is_some()
            && item.write_set.unique_slots_i32.len() == 1
            && item.write_set.unique_slots.is_empty()
            && item.write_set.rows.is_empty()
            && matches!(&item.offlock_delta, Some(d)
                if Self::reresolve_reuse_eligible(d)
                    && matches!(&d.mutation,
                        crate::write_path::PreparedMutation::Insert { inserted_rows, .. }
                            if inserted_rows.len() == 1))
            && match &item.cmd {
                Command::Insert(insert) => self.table_install_elided(&insert.table),
                _ => false,
            }
    }

    /// E2.4a VARIANT 1 — sequence a HOMOGENEOUS covered-INSERT-intent wave with N parallel shard
    /// workers over ONE ordered WAL + ONE global commit-seq.
    ///
    /// Stage 1 (device, coordinator): the wave-batched PK-unique locate — one device call, the
    /// committed-dup 23505 verdicts. Stage 2 (N parallel workers, NO commit lock): each worker
    /// owns the wave positions whose unique slot hashes to its shard and, in wave-position order,
    /// runs the SI conflict check against a shared-ledger SNAPSHOT + a per-shard PRIVATE dedup set
    /// (same-slot → same shard, so the lowest-position writer wins — byte-identical to the serial
    /// record-as-you-go single-winner), then clones the row image + WAL record. Stage 3 (thin
    /// serial cut, commit lock): walk the wave in order, and for each committing item claim the
    /// next commit-seq + integer row id, patch the WAL record's row id, append + propose, record the
    /// commit timestamp + the write-set into the SHARED ledger (classic-path interop), advance the
    /// elided row-id allocator, and buffer the row into the per-table device append. The device
    /// append flushes ONCE per table at the tail (one HtoD/wave); the durability tail is returned
    /// for the W2 pipeline exactly as the serial path.
    fn sequence_commit_wave_sharded(
        &self,
        batch: Vec<CommitWaveItem>,
        shards: usize,
    ) -> Option<(CommitWaveTail, BTreeSet<String>)> {
        let guard = CommitWaveBatchGuard {
            engine: self,
            items: &batch,
        };
        let wall_clock = current_timestamp_micros();
        let n = batch.len();

        // Stage 1 — the wave-batched device PK-unique locate (committed-dup 23505 verdicts). One
        // device call, off the commit lock, identical to the serial path.
        let hostphase = wave_host_phase_timing_enabled();
        let wave_validate_started = hostphase.then(Instant::now);
        let wave_unique_violations = self.wave_batch_validate_unique(&batch);
        if let Some(started) = wave_validate_started {
            WAVE_HOST_STATS[0]
                .fetch_add(started.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }

        let mut commit = self.commit_state();

        // Stage 2 — parallel shard prep against the shared-ledger snapshot (`&commit.ledger` is
        // borrowed immutably by the scoped workers; the coordinator resumes mutable use after join).
        let shard_started = hostphase.then(Instant::now);
        let mut verdicts =
            self.shard_prepare_intents(&batch, &commit.ledger, &wave_unique_violations, shards);
        if let Some(started) = shard_started {
            // Charge the parallel prep to the conflict bucket (it subsumes conflict + reresolve).
            WAVE_HOST_STATS[1]
                .fetch_add(started.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }

        // Stage 3 — the thin serial cut, E2.5b BATCHED: one commit-seq block claim, one
        // repl propose_batch, one row-id block, one applied mark, one timestamp for the whole
        // wave. The per-item body is reduced to the WAL push, the shared-ledger record, and the
        // device-buffer push — the E2.4a measurement showed the per-item repl round-trips,
        // BTreeMap timestamp insert, and allocator atomics WERE the ordered cut (~1.3us/item).
        // Aborts never consume a commit seq (same as the serial path's peek-before-propose), and
        // a propose_batch failure aborts the WHOLE wave — identical semantics to a first-item
        // propose failure, since the single-node leader either accepts all or is not leader.
        let mut wave_tail: Option<(Index, usize)> = None;
        let mut committed: Vec<(usize, Index, bool, u64)> = Vec::with_capacity(n);
        let mut pending_appends: WavePendingAppends = BTreeMap::new();
        let cut_started = hostphase.then(Instant::now);
        // Pass 1 — settle aborts, collect winners (position + payload parts) in wave order.
        let mut winners: Vec<(usize, String, Vec<SqlValue>, Vec<u8>, usize)> =
            Vec::with_capacity(n);
        for position in 0..n {
            match verdicts[position]
                .take()
                .expect("every position has a verdict")
            {
                ShardVerdict::Abort(err) => {
                    batch[position].set_outcome(Err(err));
                }
                ShardVerdict::Commit {
                    table,
                    values,
                    wal_record,
                    wal_offset,
                } => winners.push((position, table, values, wal_record, wal_offset)),
            }
        }
        if !winners.is_empty() {
            let k = winners.len() as u64;
            let first_seq = commit.repl.peek_next_index();
            let row_id_base = self.read_state.mvcc.current_row_id();
            let wal_len_before = commit.wal.len();
            let mut payloads: Vec<std::sync::Arc<[u8]>> = Vec::with_capacity(winners.len());
            for (offset, (position, _table, _values, wal_record, wal_offset)) in
                winners.iter_mut().enumerate()
            {
                // Patch the pre-encoded W5a record's 8-byte row id (the only wave-time field).
                let row_id = row_id_base + offset as u64;
                wal_record[*wal_offset..*wal_offset + 8].copy_from_slice(&row_id.to_le_bytes());
                let wal_payload: std::sync::Arc<[u8]> =
                    std::sync::Arc::from(std::mem::take(wal_record));
                commit.wal.append(WalRecord {
                    txn_id: batch[*position].txn_id,
                    payload: wal_payload.clone(),
                });
                payloads.push(wal_payload);
            }
            match commit.repl.propose_batch(payloads) {
                Ok(proposed_first) => {
                    debug_assert_eq!(
                        proposed_first, first_seq,
                        "the sequencer is the single proposer: the batch must start at the peek"
                    );
                    let last_seq = first_seq + k - 1;
                    // Per-winner UNIQUE timestamps (audit F1): base + offset reproduces the serial
                    // path's strictly-increasing per-txn stamps (the max-guard chain), keeping
                    // PITR-to-timestamp unambiguous at wave boundaries. `record_commit_timestamp`
                    // bumps the running max per call, so later waves stay monotonic.
                    let base_timestamp_micros =
                        wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
                    for (offset, (position, table, values, _record, _off)) in
                        winners.into_iter().enumerate()
                    {
                        let commit_seq = first_seq + offset as u64;
                        commit.record_commit_timestamp(
                            batch[position].txn_id,
                            base_timestamp_micros + offset as u64,
                        );
                        // Classic-path interop: record the write-set into the SHARED ledger (in
                        // wave order). Same-slot dups were already resolved by the workers
                        // (single-winner), so recording every winner is conflict-free.
                        commit.ledger.record(&batch[position].write_set, commit_seq);
                        let entry = pending_appends.entry(table).or_default();
                        entry.3.push(commit_seq);
                        entry.0.push(values);
                        entry.1.push(row_id_base + offset as u64);
                        // Sharded winners are single-row covered INSERTs by eligibility.
                        entry.2.push((position, commit_seq, 1));
                    }
                    // Elided apply, batched: advance the row-id allocator + elision counter by the
                    // whole wave (host store skipped) and mark the block applied once.
                    self.read_state.mvcc.advance_row_id(k);
                    self.read_state
                        .residency
                        .host_install_elisions
                        .fetch_add(k, std::sync::atomic::Ordering::Relaxed);
                    commit.repl.mark_applied(last_seq);
                    wave_tail = Some((last_seq, commit.wal.len()));
                }
                Err(err) => {
                    // Whole-wave abort: nothing proposed, nothing durable, no seq consumed.
                    commit.wal.truncate(wal_len_before);
                    let message = format!("wave propose failed: {err}");
                    for (position, _table, _values, _record, _offset) in winners.into_iter() {
                        batch[position].set_outcome(Err(ExecuteError::Engine(
                            EngineError::ProposalFailed(message.clone()),
                        )));
                    }
                }
            }
        }
        self.flush_wave_pending_appends(&mut pending_appends, &mut committed, &batch);
        if let Some(started) = cut_started {
            // Charge the ordered serial cut (WAL append + commit-seq + shared-ledger record +
            // device buffer) to the `sequence` bucket — raw nanos; the bench divides by items.
            WAVE_HOST_STATS[3]
                .fetch_add(started.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }
        // Prune the ledger once per wave below the oldest active snapshot (same as serial).
        if let Some((last_seq, _)) = wave_tail {
            let prune_boundary = self
                .active_snapshots
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .oldest()
                .map(|oldest| oldest.saturating_sub(1))
                .unwrap_or(last_seq);
            commit.ledger.prune_below(prune_boundary);
        }
        // === leave the commit critical section BEFORE the fsync (D3b group commit) ===
        drop(commit);

        let Some((last_seq, last_position)) = wave_tail else {
            // Every item aborted pre-durable; outcomes are already set.
            std::mem::forget(guard);
            return None;
        };

        // Post-publish auto_admit re-admissions stay a SEQUENCER duty (identical to serial).
        let mut admit_tables: BTreeSet<String> = BTreeSet::new();
        if self.auto_admit_on_commit_enabled() {
            for (position, _seq, appended, _rows) in &committed {
                if !appended {
                    admit_tables.extend(batch[*position].residency_tables.iter().cloned());
                }
            }
        }
        std::mem::forget(guard);
        Some((
            CommitWaveTail {
                batch,
                committed,
                last_seq,
                last_position,
                armed: true,
            },
            admit_tables,
        ))
    }

    /// E2.4a — the parallel shard-prep pass (Stage 2 of [`Engine::sequence_commit_wave_sharded`]).
    /// Partitions the wave's positions across `shards` workers by hashing each row's single unique
    /// slot (so same-slot rows land in the same worker) and, per worker, computes a per-position
    /// [`ShardVerdict`] in wave-position order: 23505 for a committed-dup (device-locate verdict),
    /// a retryable serialization abort for a shared-ledger conflict OR an intra-wave same-slot
    /// duplicate (the per-shard private dedup set — lowest position wins), else a Commit carrying
    /// the cloned row image + the cloned (still-placeholder-row-id) WAL record. `shared_ledger` is
    /// read-only for the whole pass (the coordinator holds the commit lock and does not mutate it
    /// until after join), so the workers see a consistent snapshot of all pre-wave commits.
    fn shard_prepare_intents(
        &self,
        batch: &[CommitWaveItem],
        shared_ledger: &crate::write_path::RecentCommitsLedger,
        unique_violations: &std::collections::BTreeMap<usize, String>,
        shards: usize,
    ) -> Vec<Option<ShardVerdict>> {
        let n = batch.len();
        // Assign each position to a shard by its single unique slot (same slot → same shard).
        let shard_of: Vec<usize> = (0..n)
            .map(|pos| {
                let (slot_id, value) = batch[pos].write_set.unique_slots_i32[0];
                shard_index(slot_id, value, shards)
            })
            .collect();
        let shard_of = &shard_of;
        let mut results: Vec<Option<ShardVerdict>> = (0..n).map(|_| None).collect();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..shards)
                .map(|s| {
                    scope.spawn(move || {
                        let mut private: std::collections::HashSet<crate::write_path::IntUniqueSlotKey> =
                            std::collections::HashSet::new();
                        let mut out: Vec<(usize, ShardVerdict)> = Vec::new();
                        for pos in 0..n {
                            if shard_of[pos] != s {
                                continue;
                            }
                            let item = &batch[pos];
                            let verdict = if let Some(msg) = unique_violations.get(&pos) {
                                // Committed-dup: the same 23505 the serial device-locate verdict raises.
                                ShardVerdict::Abort(ExecuteError::Engine(EngineError::ApplyFailed(
                                    msg.clone(),
                                )))
                            } else if shared_ledger
                                .conflicts(&item.write_set, item.read_snapshot)
                            {
                                let rs = item.read_snapshot;
                                ShardVerdict::Abort(ExecuteError::Serialization(format!(
                                    "write-write conflict on a key committed after read snapshot {rs}"
                                )))
                            } else {
                                let slot = item.write_set.unique_slots_i32[0];
                                if !private.insert(slot) {
                                    // Intra-wave same-slot duplicate: the later position loses,
                                    // exactly the serial integer-ledger first-committer-wins verdict.
                                    let rs = item.read_snapshot;
                                    ShardVerdict::Abort(ExecuteError::Serialization(format!(
                                        "write-write conflict on a key committed after read snapshot {rs}"
                                    )))
                                } else {
                                    let (template, offset) = item
                                        .binary_wal_template
                                        .as_ref()
                                        .expect("sharded eligibility requires a binary WAL template");
                                    let delta = item
                                        .offlock_delta
                                        .as_ref()
                                        .expect("sharded eligibility requires an off-lock delta");
                                    let crate::write_path::PreparedMutation::Insert {
                                        table,
                                        inserted_rows,
                                        ..
                                    } = &delta.mutation
                                    else {
                                        unreachable!("sharded eligibility gates to single-row inserts");
                                    };
                                    ShardVerdict::Commit {
                                        table: table.clone(),
                                        values: inserted_rows[0].1.clone(),
                                        wal_record: template.to_vec(),
                                        wal_offset: *offset as usize,
                                    }
                                }
                            };
                            out.push((pos, verdict));
                        }
                        out
                    })
                })
                .collect();
            for handle in handles {
                for (pos, verdict) in handle.join().expect("shard worker panicked") {
                    results[pos] = Some(verdict);
                }
            }
        });
        results
    }

    /// W2 — finish one pipelined wave tail: ONE group-durability wait covering every record the
    /// wave appended, then ONE `committed_seq` publish at the wave's last seq (CAS max), then the
    /// outcome acks. WAL-before-visibility holds wave-wide: nothing in the wave is visible until
    /// its highest record is fsync-covered. Runs on WHICHEVER thread claimed the tail (a member
    /// waiter or the sequencer); requires no locks beyond what `wait_group_durable` takes
    /// internally, and never the commit_mutex the sequencer may be holding for the next wave.
    fn finish_wave_tail(&self, mut tail: CommitWaveTail) {
        // AUDIT d6d10f8e D (documented decision): on an fsync FAILURE the flusher arm panics on
        // the CLAIMING thread — which may be a client whose OWN statement is already durably
        // acked (it claimed a DIFFERENT wave's tail). That client observes a panic for a
        // committed statement: the classic group-commit ack ambiguity, bounded to the fail-stop
        // fsync-failure world where the whole path wedges anyway. Accepted: the pre-W2 shape
        // had the same ambiguity on the sequencer's client thread, and any retry-after-restart
        // discipline must already tolerate acked-but-uncertain outcomes.
        // Unwind-safe completion accounting: the finished-counter bump + wakeups MUST fire even
        // when the group-fsync-failure arm PANICS below (wait_group_durable's flusher arm panics
        // holding the commit_mutex — the wedge-don't-serve-torn-state policy). Without it the
        // next sequencer parks forever at the depth gate (handed > finished, empty slot). On any
        // unclean exit the guard also WEDGES the queue and fails everything still queued —
        // parity with `CommitWaveBatchGuard` for the durability half of the wave (the tail's own
        // Drop fails its member outcomes).
        struct TailCompletion<'a> {
            engine: &'a Engine,
            clean: bool,
        }
        impl Drop for TailCompletion<'_> {
            fn drop(&mut self) {
                if !self.clean {
                    let mut queue = self.engine.lock_commit_wave_queue();
                    let reason = queue.wedged.clone().unwrap_or_else(|| {
                        "the commit-wave durability tail died before completing".to_string()
                    });
                    queue.wedged = Some(reason.clone());
                    queue.sequencer_active = false;
                    let stranded: Vec<CommitWaveItem> = queue.items.drain(..).collect();
                    drop(queue);
                    for item in &stranded {
                        if !item.outcome.done.load(AtomicOrdering::Acquire) {
                            item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                                format!(
                                    "the concurrent commit path is wedged pending restart \
                                     recovery: {reason}"
                                ),
                            ))));
                        }
                    }
                }
                self.engine
                    .commit_wave
                    .tails_finished
                    .fetch_add(1, AtomicOrdering::Release);
                let _queue = self.engine.lock_commit_wave_queue();
                self.engine.commit_wave.cv.notify_all();
            }
        }
        let mut completion = TailCompletion {
            engine: self,
            clean: false,
        };
        if let Err(err) = self.wait_group_durable(tail.last_position) {
            // The wave's deltas are applied-but-unpublishable; the flush protocol has recorded
            // its sticky failure (and the flusher wedge-panicked if we were the flusher — in
            // that case this line is unreachable and the completion guard runs on unwind). Fail
            // the wave's outcomes and wedge the queue.
            let mut queue = self.lock_commit_wave_queue();
            queue.wedged.get_or_insert_with(|| err.to_string());
            drop(queue);
            drop(tail); // armed: fails every still-unset member outcome with the wedge error
            return; // completion guard (clean=false) wedges idempotently + counts + notifies
        }
        self.publish_committed_seq(tail.last_seq);
        for (position, _seq, _appended, rows) in &tail.committed {
            self.metrics.inc_commit();
            tail.batch[*position].set_outcome(Ok(*rows));
        }
        tail.armed = false;
        completion.clean = true;
    }

    /// W2 — claim and finish the pending pipeline tail if one is waiting. Called from the waiter
    /// loops' probes (the tail's own members are the natural claimers — they are blocked on its
    /// outcomes) and by the sequencer as the fallback claimer at the depth gate. Returns whether
    /// a tail was finished.
    pub(crate) fn try_finish_pending_wave_tail(&self) -> bool {
        // AUDIT d6d10f8e F: every waiter probes this every 64 spin iterations — pre-check the
        // counters (two atomic loads) so the common nothing-pending case never touches the
        // shared mutex.
        {
            let handed = self.commit_wave.tails_handed.load(AtomicOrdering::Acquire);
            let finished = self
                .commit_wave
                .tails_finished
                .load(AtomicOrdering::Acquire);
            if handed <= finished {
                return false;
            }
        }
        let taken = {
            let mut tails = self
                .commit_wave
                .pending_tails
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            tails.pop_front()
        };
        match taken {
            Some(tail) => {
                self.finish_wave_tail(tail);
                true
            }
            None => false,
        }
    }

    /// D3b — the group-commit flush protocol. Blocks until the WAL's durable frontier covers
    /// `wal_position` (this committer's appended record). The first committer to arrive while no
    /// flush is in flight becomes the FLUSHER: one `flush_all` (one fsync) covers every record
    /// appended up to that moment — its own and every waiter's — after which all covered waiters
    /// proceed. Committers arriving mid-flush wait and (if the in-flight fsync started before
    /// their append) run/join the NEXT flush, so an fsync is never wasted on a stale frontier.
    ///
    /// Failure semantics (deliberate, documented): the caller's delta is ALREADY APPLIED (that is
    /// what lets later committers validate against it while this fsync is in flight), so a failed
    /// group fsync cannot be rolled back into a clean per-statement abort — the applied-but-
    /// unpublishable deltas must never become visible. The flusher therefore records a STICKY
    /// failure (every current and future concurrent committer errors out) and PANICS while
    /// holding the commit_mutex, poisoning it — the façade's poison-on-panic policy refuses
    /// further service and a restart recovers from the durable WAL prefix (the un-fsynced records
    /// were never acknowledged nor visible). This mirrors the concurrent path's existing
    /// panic-on-post-durable-apply-failure wedge philosophy, and matches the D1 append-only
    /// writer's fail-closed poisoning of the segment backing on real I/O errors. The SERIALIZED
    /// commit path (`commit_mutation_at`) keeps its inline flush and clean per-statement abort.
    /// E2.5b-2 stage 3b — one pump iteration for intent lane `lane`: the N-lane
    /// parallel ordered cut. Single-writer per lane (try_lock guard); the ONLY
    /// shared-state touch is one brief commit lock per wave (global seq-block
    /// claim + timestamp merge). Everything else — device validate, private
    /// ledger, lane WAL append, device apply — runs lane-parallel. Outcomes
    /// settle exclusively behind the visible cut (durable AND applied AND
    /// published), so an ack can never precede any lower seq's durability.
    pub(crate) fn drive_intent_lane(&self, lane: usize) -> bool {
        let Some(lanes) = self.intent_lanes.as_ref().map(std::sync::Arc::clone) else {
            return false;
        };
        let Ok(_pump_guard) = lanes.pump_guards[lane].try_lock() else {
            return false; // another pump owns this lane right now
        };
        // settle matured waves first: acks lead each iteration
        let settle_started = Instant::now();
        let progressed = self.settle_intent_lane(&lanes, lane);
        lanes.stat_settle_ns.fetch_add(
            settle_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        // WORKLOAD-ADAPTIVE wave formation (the conveyor laws, population-
        // scaled): the ship target and the age deadline follow the LIVE
        // population, with the configured MIN_WAVE/GROUP_US acting as the
        // high-load CAPS. At 512 outstanding the target is ~a lane's arrival
        // share and the deadline tens of microseconds (latency regime); at
        // 60k+ outstanding both clamp to the configured batching values
        // (throughput regime). One configuration serves both ends.
        let wave_max = crate::engine_intent_lanes::intent_lane_wave_max();
        let min_wave = crate::engine_intent_lanes::intent_lane_min_wave();
        let outstanding = lanes.outstanding.load(std::sync::atomic::Ordering::Relaxed) as usize;
        let ship_target = (outstanding
            / (lanes.lane_count * crate::engine_intent_lanes::intent_lane_ship_div()))
        .clamp(1, min_wave);
        let group_window = std::time::Duration::from_micros(
            ((outstanding / lanes.lane_count) as u64)
                .min(crate::engine_intent_lanes::intent_lane_group_us()),
        );
        let drain_started = Instant::now();
        let batch: Vec<LaneIntent> = {
            let mut queue = lanes.queues[lane]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if queue.is_empty() {
                Vec::new()
            } else {
                let ship = if queue.len() < ship_target {
                    let mut since = lanes.pending_since[lane]
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let started = since.get_or_insert_with(std::time::Instant::now);
                    let expired = started.elapsed() >= group_window;
                    if expired {
                        *since = None;
                    }
                    expired // else: let the wave fill
                } else {
                    *lanes.pending_since[lane]
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
                    true
                };
                if ship {
                    let n = queue.len().min(wave_max);
                    queue.drain(..n).collect()
                } else {
                    Vec::new()
                }
            }
        };
        if batch.is_empty() {
            // Idle or wave still forming: keep the apply coalescer moving so
            // queued waves complete and their cuts advance.
            let applied = self.drive_apply_queue_once(&lanes);
            if applied {
                self.settle_intent_lane(&lanes, lane);
            }
            return progressed || applied;
        }
        lanes.stat_drain_ns.fetch_add(
            drain_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        // device validate (committed-dup 23505 verdicts), off-lock and LEAN:
        // needles come straight from the intents' integer slots (no AST walk);
        // the locate goes through the cross-lane coalescer; count>0 hits get
        // the same authoritative visibility recheck as the classic path.
        let stat_start = Instant::now();
        let violations = self.lane_validate_unique(&batch);
        lanes.stat_validate_ns.fetch_add(
            stat_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        lanes.stat_waves.fetch_add(1, AtomicOrdering::Relaxed);
        lanes
            .stat_items
            .fetch_add(batch.len() as u64, AtomicOrdering::Relaxed);

        // private-ledger conflicts + intra-wave same-slot dedup (lowest position wins).
        // PASS-FUSION: integer slots are extracted here once (winner_slots) so the
        // post-claim ledger record never re-walks the fat items.
        let conflict_started = Instant::now();
        let mut winners: Vec<LaneIntent> = Vec::with_capacity(batch.len());
        let mut winner_slots: Vec<crate::write_path::IntUniqueSlotKey> =
            Vec::with_capacity(batch.len());
        {
            let ledger = lanes.ledgers[lane]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut wave_slots: std::collections::HashSet<crate::write_path::IntUniqueSlotKey> =
                std::collections::HashSet::with_capacity(batch.len());
            for (position, item) in batch.into_iter().enumerate() {
                if let Some(err) = violations.get(&position) {
                    item.set_outcome(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        err.clone(),
                    ))));
                    continue;
                }
                if ledger.conflicts_int_slot(item.slot, item.read_snapshot) {
                    let read_snapshot = item.read_snapshot;
                    item.set_outcome(Err(ExecuteError::Serialization(format!(
                        "write-write conflict on a key committed after read snapshot {read_snapshot}"
                    ))));
                    continue;
                }
                if !wave_slots.insert(item.slot) {
                    item.set_outcome(Err(ExecuteError::Serialization(
                        "intra-wave duplicate key: an earlier same-wave insert holds this unique slot"
                            .to_string(),
                    )));
                    continue;
                }
                winner_slots.push(item.slot);
                winners.push(item);
            }
        }
        lanes.stat_conflict_ns.fetch_add(
            conflict_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        let k = winners.len() as u64;
        if k == 0 {
            return true;
        }

        // LAZY WAL BACKING (E2.5c-3 default flip): resolve/create the lane set BEFORE any seq
        // is claimed — a creation failure (ENOSPC/EDQUOT during the per-lane prewrite) here
        // fails the wave cleanly; after the claim it would HOLE the cross-lane cut (claimed
        // seqs that can never become durable stall every later ack).
        let wal_lanes = match lanes.wal() {
            Ok(wal) => wal,
            Err(err) => {
                let message = format!("intent lane WAL unavailable: {err}");
                for item in winners {
                    item.set_outcome(Err(ExecuteError::Engine(EngineError::ProposalFailed(
                        message.clone(),
                    ))));
                }
                return true;
            }
        };

        // row-id block (atomic claim — safe under concurrent lanes) + W5a patches
        let patch_started = Instant::now();
        let row_id_base = self.read_state.mvcc.claim_row_id_block(k);
        // FUSED patch+envelope pass: the frame payload is assembled in the same
        // loop that patches each record (bytes are cache-warm), replacing the
        // separate encode_record_into pass that was measured at ~1.4us/record
        // of cold Arc re-walks (1.4ms of a 1000-record wave).
        let mut frame_payload: Vec<u8> = Vec::with_capacity(winners.len() * 24 + 4096);
        // Per-record end offsets: sub-frame publishing splits the payload on
        // record boundaries (see `intent_lane_subframes`).
        let mut record_ends: Vec<usize> = Vec::with_capacity(winners.len());
        for (offset, item) in winners.iter().enumerate() {
            let off = item.row_id_offset as usize;
            let row_id = row_id_base + offset as u64;
            let mut payload: std::sync::Arc<[u8]> = std::sync::Arc::from(&item.template[..]);
            std::sync::Arc::get_mut(&mut payload).expect("freshly created Arc is unique")
                [off..off + 8]
                .copy_from_slice(&row_id.to_le_bytes());
            gpu_db_wal::encode_wal_record_parts_into(&mut frame_payload, item.txn_id, &payload);
            record_ends.push(frame_payload.len());
        }
        lanes.stat_patch_ns.fetch_add(
            patch_started.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        // THE CLAIM, LOCK-FREE: after activation, a seq block is one fetch_add
        // on the lanes oracle (the per-wave commit-lock claim measured 77%
        // lock-wait at 8 lanes). The FIRST wave seeds the oracle + timestamp
        // reservation under the commit lock, then flips the activation latch;
        // v1 contract: classic writes are refused after activation and the
        // repl log intentionally does not carry lane payloads (single-node;
        // recovery reads the lane logs' explicit seqs — Raft integration is
        // an E2.5c+ concern).
        let stat_start = Instant::now();
        let first_seq = if lanes.activated.load(std::sync::atomic::Ordering::Acquire) {
            lanes
                .seq_oracle
                .fetch_add(k, std::sync::atomic::Ordering::AcqRel)
        } else {
            // AUDIT F1 (CRITICAL): double-checked activation UNDER the commit
            // lock. peek_next_index does not advance, so two lanes' first
            // waves racing the old check-then-act would both seed the oracle
            // at the same base and claim DUPLICATE global seqs — acked
            // commits then fail recovery (overlapping lane claims). Exactly
            // one seeder exists now (latch set LAST, under the lock); the
            // race loser re-reads the seeded oracle.
            let commit = self.commit_state();
            if lanes.activated.load(std::sync::atomic::Ordering::Acquire) {
                drop(commit);
                lanes
                    .seq_oracle
                    .fetch_add(k, std::sync::atomic::Ordering::AcqRel)
            } else {
                let first = commit.repl.peek_next_index();
                lanes
                    .seq_oracle
                    .store(first + k, std::sync::atomic::Ordering::Release);
                lanes
                    .base_seq
                    .store(first, std::sync::atomic::Ordering::Release);
                lanes
                    .activated
                    .store(true, std::sync::atomic::Ordering::Release);
                first
            }
        };
        lanes.stat_claim_ns.fetch_add(
            stat_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        let base = lanes.base_seq.load(std::sync::atomic::Ordering::Acquire);
        let local_first = first_seq - base;

        // record winners into the lane's private ledger at their global seqs —
        // from the slots extracted in the conflict pass (no fat-item re-walk)
        {
            let mut ledger = lanes.ledgers[lane]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for (offset, slot) in winner_slots.iter().enumerate() {
                ledger.record_int_slot(*slot, first_seq + offset as u64);
            }
        }

        // OFF-LOCK: durable lane append (envelope already fused into the patch
        // pass above; stat_encode retired into the claim-adjacent patch time).
        // SUB-FRAME SPLITTING (see `intent_lane_subframes`): the wave publishes
        // as N contiguous-seq frames so the same traffic generates N in-flight
        // FUA fences — pushing the drive into its fast mode at low load. The
        // ack waits on the contiguous cut over all N (pipelined), and recovery
        // semantics are unchanged (per-frame intervals, same merge math).
        let stat_start = Instant::now();
        let configured_subframes = crate::engine_intent_lanes::intent_lane_subframes();
        let subframes = if configured_subframes == 0 {
            // AUTO: split only in the low-depth regime — a mostly-idle fence
            // pool means the drive is out of its bimodal fast mode and two
            // pipelined frames beat one slow one. A busy pool (high load)
            // publishes single frames.
            let free = wal_lanes.free_fence_slots(lane).unwrap_or(0);
            if free * 4 >= lanes.fence_lanes * 3 {
                2
            } else {
                1
            }
        } else {
            configured_subframes
        }
        .min(k as usize)
        .max(1);
        let mut append_error: Option<String> = None;
        if subframes == 1 {
            if let Err(err) = wal_lanes.append_encoded(
                lane,
                local_first,
                local_first + k,
                k as u32,
                &frame_payload,
            ) {
                append_error = Some(format!("lane WAL append failed: {err}"));
            }
        } else {
            let per = k as usize / subframes;
            let rem = k as usize % subframes;
            let mut rec_start = 0usize;
            let mut byte_start = 0usize;
            let mut seq = local_first;
            for chunk_idx in 0..subframes {
                let take = per + usize::from(chunk_idx < rem);
                if take == 0 {
                    continue;
                }
                let rec_end = rec_start + take;
                let byte_end = record_ends[rec_end - 1];
                if let Err(err) = wal_lanes.append_encoded(
                    lane,
                    seq,
                    seq + take as u64,
                    take as u32,
                    &frame_payload[byte_start..byte_end],
                ) {
                    append_error = Some(format!("lane WAL append failed: {err}"));
                    break;
                }
                seq += take as u64;
                rec_start = rec_end;
                byte_start = byte_end;
            }
        }
        if let Some(message) = append_error {
            for item in winners {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::ProposalFailed(
                    message.clone(),
                ))));
            }
            return true;
        }
        lanes.stat_publish_ns.fetch_add(
            stat_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        // device apply via the APPLY COALESCER: push this wave's prepared rows;
        // whoever wins the device lock becomes the leader and applies EVERY
        // pending request in ONE merged per-table append pass (fixed-per-pass
        // device cost amortizes across lanes; the leader lock preserves the
        // PK-index extension chain exactly like the old exclusive section).
        let apply_slot = std::sync::Arc::new(crate::engine_intent_lanes::ApplySlot {
            done: std::sync::atomic::AtomicBool::new(false),
            failed: std::sync::atomic::AtomicBool::new(false),
        });
        {
            let mut rows = Vec::with_capacity(winners.len());
            let mut stamps = Vec::with_capacity(winners.len());
            let mut txn_ids = Vec::with_capacity(winners.len());
            let mut row_ids = Vec::with_capacity(winners.len());
            let table_name = winners
                .first()
                .map(|item| item.table.to_string())
                .unwrap_or_default();
            for (offset, item) in winners.iter_mut().enumerate() {
                rows.push(std::mem::take(&mut item.values));
                stamps.push(first_seq + offset as u64);
                txn_ids.push(item.txn_id);
                row_ids.push(row_id_base + offset as u64);
            }
            lanes
                .apply_queue
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(crate::engine_intent_lanes::ApplyRequest {
                    table: table_name,
                    rows,
                    row_ids,
                    stamps,
                    txn_ids,
                    slot: std::sync::Arc::clone(&apply_slot),
                });
        }
        // NO-REAP PIPELINE (disruptor staging): the wave's apply request is
        // queued and its settlement entry goes straight into the settle queue
        // — the pump NEVER waits on device apply. Settlement is gated on the
        // visible cut, which only the apply LEADER advances (at completion),
        // so a settled-Ok still implies durable AND applied; the failed flag
        // covers the failure path. Same-slot safety holds without the device
        // index seeing this wave: the lane ledger recorded the winners at
        // claim and PK-hash routing pins a PK to one lane.
        // Partition by commit mode (pg `synchronous_commit`): async winners
        // ack at the APPLIED cut, strict winners at the visible (durable AND
        // applied) cut. Same wave, same WAL frames, same apply — only the
        // ack gate differs.
        let (async_winners, winners): (Vec<LaneIntent>, Vec<LaneIntent>) =
            winners.into_iter().partition(|item| !item.synchronous);
        lanes.settle[lane]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push_back(crate::engine_intent_lanes::LaneSettle {
                end_seq: local_first + k,
                winners,
                async_winners,
                async_settled: false,
                apply_slot,
                published_at: std::time::Instant::now(),
            });
        // Opportunistic non-blocking leader pass keeps the apply queue moving
        // (nobody blocks on it anymore).
        self.drive_apply_queue_once(&lanes);
        self.settle_intent_lane(&lanes, lane);
        true
    }

    /// WORKLOAD-ADAPTIVE ACTIVE-LANE RESIZE (slice 3): pick the routing-subset
    /// size from the live population and, when it changes, pass through the
    /// DRAIN BARRIER — divert new submits to the hold queue, pump every lane
    /// until nothing is in flight, flip `active_lanes`, then re-route the held
    /// intents through the new epoch. Safety: at the barrier every prior
    /// commit precedes every post-flip snapshot, so the device validate alone
    /// catches old duplicates (lane-ledger continuity across epochs is not
    /// required). Fail-open: a drain that cannot complete (wedge) aborts the
    /// resize and keeps the old epoch.
    fn maybe_resize_lanes(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
    ) {
        const UP_AT: u64 = 4096;
        const DOWN_AT: u64 = 1024;
        const DWELL: std::time::Duration = std::time::Duration::from_millis(200);
        /// A down-flip drains everything in flight, so a momentary dip at
        /// high load must not trigger one: the population has to stay low
        /// for this long, continuously, first.
        const DOWN_STREAK: std::time::Duration = std::time::Duration::from_millis(500);
        const DRAIN_LIMIT: std::time::Duration = std::time::Duration::from_secs(5);
        let low = lanes.lane_count.min(4);
        if lanes.lane_count <= low {
            return; // nothing to adapt between
        }
        let active = lanes
            .active_lanes
            .load(std::sync::atomic::Ordering::Acquire);
        let outstanding = lanes.outstanding.load(std::sync::atomic::Ordering::Relaxed);
        let target = if outstanding >= UP_AT {
            lanes.lane_count
        } else if outstanding <= DOWN_AT {
            low
        } else {
            active // hysteresis band: hold
        };
        if target >= active {
            // Not shrinking: clear any low streak (population recovered).
            if outstanding > DOWN_AT {
                if let Ok(mut since) = lanes.resize_low_since.try_lock() {
                    *since = None;
                }
            }
            if target == active {
                return;
            }
            // Up-flips proceed immediately (throughput emergency).
        } else {
            // Down-flip: require a SUSTAINED low population first.
            let Ok(mut since) = lanes.resize_low_since.try_lock() else {
                return;
            };
            match *since {
                None => {
                    *since = Some(std::time::Instant::now());
                    return;
                }
                Some(started) if started.elapsed() < DOWN_STREAK => return,
                Some(_) => {}
            }
        }
        let Ok(mut leader) = lanes.resize_leader.try_lock() else {
            return; // a resize is already in progress
        };
        if leader.is_some_and(|last| last.elapsed() < DWELL) {
            return; // dwell: no flapping
        }
        // BARRIER: divert new submits, drain everything in flight. SeqCst on
        // the holding store pairs with the SeqCst increment-then-check in
        // `submit_lane_intent` (Dekker): a submit that observed holding=false
        // has its `outstanding` increment ordered before our drain reads, so
        // the drain below cannot miss it (AUDIT: the uncounted-straggler
        // TOCTOU admitted a same-PK intent into the old epoch after the last
        // drain observation).
        lanes
            .resize_holding
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let drain_started = Instant::now();
        let mut drained = true;
        // The barrier must ALSO wait for the VISIBLE cut to cover every
        // claimed seq: async-commit winners leave `outstanding` at the
        // APPLIED cut, but the post-flip snapshot-refresh safety argument
        // needs their commits VISIBLE (covered by committed_seq) — an
        // applied-but-not-yet-durable row is invisible to the authoritative
        // recheck and would reopen the duplicate-key hole the merge audit
        // closed.
        let claimed_frontier = |lanes: &crate::engine_intent_lanes::IntentLaneState| -> u64 {
            if !lanes.activated.load(std::sync::atomic::Ordering::Acquire) {
                return 0;
            }
            let base = lanes.base_seq.load(std::sync::atomic::Ordering::Acquire);
            lanes
                .seq_oracle
                .load(std::sync::atomic::Ordering::Acquire)
                .saturating_sub(base)
        };
        // AUDIT (minor): after a WAL-append failure the claimed frontier
        // contains seqs that can never become durable — skip the frontier
        // wait once the WAL is poisoned (the engine is wedging loudly via the
        // settle drain anyway) so resize keeps failing OPEN in 5s, not
        // permanently spinning.
        let wal_poisoned = |lanes: &crate::engine_intent_lanes::IntentLaneState| -> bool {
            lanes.wal_peek().is_some_and(|wal| wal.is_poisoned())
        };
        while lanes.outstanding.load(std::sync::atomic::Ordering::SeqCst) > 0
            || (!wal_poisoned(lanes) && lanes.visible_local_cut() < claimed_frontier(lanes))
        {
            for lane in 0..lanes.lane_count {
                self.drive_intent_lane(lane);
            }
            if drain_started.elapsed() > DRAIN_LIMIT {
                drained = false; // wedge: fail-open, keep the old epoch
                break;
            }
        }
        if drained {
            lanes
                .active_lanes
                .store(target, std::sync::atomic::Ordering::Release);
            // AUDIT (async-commit slice, MUST-FIX): the loop above waits for
            // the VISIBLE cut to cover the claimed frontier, but committed_seq
            // is only published inside settle — a fence completing between the
            // last settle and the loop exit leaves committed_seq BEHIND the
            // frontier, and the re-route's refreshed snapshots (which read
            // committed_seq) would miss a just-fenced async commit: the
            // duplicate-key hole again. Publish the covering cut HERE, before
            // any held intent re-routes.
            self.publish_committed_seq(lanes.visible_global_cut());
            lanes
                .stat_resizes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            lanes.stat_resize_ns.fetch_add(
                drain_started.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        lanes
            .resize_holding
            .store(false, std::sync::atomic::Ordering::SeqCst);
        // Re-route the held intents through the (possibly new) epoch, with a
        // POST-BARRIER read snapshot. AUDIT (CRITICAL): a held intent carries
        // the snapshot it captured at submit — PRE-flip. The barrier's safety
        // argument ("every prior commit precedes every post-flip snapshot, so
        // the device validate alone catches old duplicates") holds only for
        // post-flip snapshots: with the stale one, a same-PK commit that
        // settled during the drain is invisible to the authoritative recheck
        // (committed after the stale snapshot) AND unknown to the new lane's
        // ledger (it was recorded in the old lane) — a silent duplicate-key
        // admission. `committed_seq()` here covers every drained commit by
        // construction (the drain waited for settle, which publishes before
        // outcomes). The ticket's registered snapshot hold keeps the OLD
        // value — a conservative GC boundary, harmless. For covered INSERTs a
        // fresher snapshot is strictly safer: it can only turn an admission
        // into a duplicate-key/serialization rejection, never the reverse.
        self.rescue_held_intents(lanes);
        *leader = Some(Instant::now());
    }

    /// Lane-ingress with the resize-barrier Dekker protocol: count the intent
    /// into the live population FIRST (SeqCst), THEN check the barrier. If the
    /// barrier is up, back the count out and divert to the hold queue; the
    /// resize leader's `holding=true (SeqCst)` -> `outstanding` drain reads
    /// pair with this increment -> check, so every intent is either counted
    /// (and drained before the flip) or diverted (and re-routed after it with
    /// a refreshed snapshot). The SINGLE lane-ingress point.
    pub(crate) fn submit_lane_intent(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
        mut intent: LaneIntent,
    ) {
        lanes
            .outstanding
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if lanes
            .resize_holding
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            lanes
                .outstanding
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            lanes
                .resize_hold
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(intent);
            // STRAND GUARD: if the barrier released between our check and the
            // push, the leader's hold-queue take may already be done and
            // nothing would ever route this intent (until the next resize).
            // Re-check AFTER the push: holding still true means the current
            // leader's take (which happens after its holding=false store) is
            // still ahead of us and will collect the item; holding false is
            // ambiguous, so self-rescue — take whatever is held and route it
            // with a fresh post-barrier snapshot (same rule as the leader's
            // re-inject; double-takes are safe, mem::take is atomic and each
            // taker routes only what it got).
            if !lanes
                .resize_holding
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                self.rescue_held_intents(lanes);
            }
            return;
        }
        intent.outstanding = Some(std::sync::Arc::clone(&lanes.outstanding));
        let lane = lanes.lane_for_pk(intent.slot.1);
        lanes.queues[lane]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push_back(intent);
    }

    /// Drain the resize hold queue outside an active barrier and route the
    /// items with a fresh read snapshot (see the CRITICAL-audit note in
    /// `maybe_resize_lanes`: held intents must never carry a pre-flip
    /// snapshot into the new epoch).
    fn rescue_held_intents(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
    ) {
        let held: Vec<LaneIntent> = std::mem::take(
            &mut *lanes
                .resize_hold
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        if held.is_empty() {
            return;
        }
        let refreshed_snapshot = self.committed_seq();
        for mut intent in held {
            intent.read_snapshot = refreshed_snapshot;
            self.submit_lane_intent(lanes, intent);
        }
    }

    /// ONE opportunistic apply-leader pass: if the device lock is free and the
    /// apply coalescing queue is non-empty, drain it and run the merged apply.
    /// Non-blocking — a busy lock or an empty queue returns immediately.
    /// Returns whether a merged apply ran.
    fn drive_apply_queue_once(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
    ) -> bool {
        let stat_start = Instant::now();
        {
            let Ok(_leader) = lanes.device_apply_lock.try_lock() else {
                return false;
            };
            let batch: Vec<crate::engine_intent_lanes::ApplyRequest> = {
                let mut queue = lanes
                    .apply_queue
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                std::mem::take(&mut *queue)
            };
            if batch.is_empty() {
                return false;
            }
            lanes
                .stat_apply_launches
                .fetch_add(1, AtomicOrdering::Relaxed);
            lanes
                .stat_apply_requests
                .fetch_add(batch.len() as u64, AtomicOrdering::Relaxed);
            let leader_started = Instant::now();
            let mut batch = batch;
            // AUDIT F2: a leader panic (rehydrate invariant, catalog expect)
            // must not strand waiters spinning on `done` forever nor poison
            // the leader lock into a permanent livelock. Catch, fail every
            // drained request loudly, and resume (the panic is re-raised
            // after waiters are released so the invariant violation still
            // surfaces).
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.lane_apply_merged(&mut batch)
            }));
            lanes.stat_apply_leader_ns.fetch_add(
                leader_started.elapsed().as_nanos() as u64,
                AtomicOrdering::Relaxed,
            );
            let failed = outcome.is_err();
            if failed {
                // A failed merged apply permanently HOLES the applied cut (its
                // seqs never apply), so later waves would wait forever behind
                // it — poison the lanes so settle drains everything loudly.
                lanes
                    .apply_poisoned
                    .store(true, std::sync::atomic::Ordering::Release);
            }
            let base = lanes.base_seq.load(std::sync::atomic::Ordering::Acquire);
            let mut applied_rows = 0u64;
            for request in &batch {
                if failed {
                    request
                        .slot
                        .failed
                        .store(true, std::sync::atomic::Ordering::Release);
                } else if let (Some(first), len) = (request.stamps.first(), request.stamps.len()) {
                    // Advance the applied cut HERE, at apply completion — the
                    // cut is GLOBAL-gating (every lane's acks wait on it);
                    // deferring it to the owning pump measurably inflated
                    // every ack (depth-2 v1: p50 21ms -> 28ms, sustained -10%).
                    // AUDIT (minor): `done` is stored BEFORE the cut advance
                    // so "cut covers the wave" always implies "its slot is
                    // done" — the settle-side debug_assert's precondition.
                    request
                        .slot
                        .done
                        .store(true, std::sync::atomic::Ordering::Release);
                    let local = first - base;
                    lanes.record_applied(local, local + len as u64);
                    applied_rows += len as u64;
                    continue;
                }
                request
                    .slot
                    .done
                    .store(true, std::sync::atomic::Ordering::Release);
            }
            self.read_state
                .residency
                .host_install_elisions
                .fetch_add(applied_rows, std::sync::atomic::Ordering::Relaxed);
            if let Err(panic) = outcome {
                std::panic::resume_unwind(panic);
            }
        }
        lanes.stat_apply_ns.fetch_add(
            stat_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );
        true
    }

    /// LEAN device validate for lane intents: needles straight from the
    /// integer slots, locate through the cross-lane coalescer, count>0 hits
    /// re-checked authoritatively at the item's read snapshot (same semantics
    /// as wave_batch_validate_unique's covered-insert arm). Returns 23505
    /// messages by batch position. Catalog drift (DDL between build and pump,
    /// pre-activation-window only) aborts the item retryably.
    fn lane_validate_unique(
        &self,
        batch: &[LaneIntent],
    ) -> std::collections::BTreeMap<usize, String> {
        let mut violations = std::collections::BTreeMap::new();
        if batch.is_empty() {
            return violations;
        }
        let catalog = self.catalog_snapshot();
        // group needles per (table, filter_idx); usually exactly one group
        let mut group_keys: Vec<(&str, u32)> = Vec::new();
        let mut group_needles: Vec<Vec<i32>> = Vec::new();
        let mut group_positions: Vec<Vec<usize>> = Vec::new();
        for (position, item) in batch.iter().enumerate() {
            if catalog.commit_seq != item.prepared_catalog_seq {
                violations.insert(
                    position,
                    "catalog drift between intent build and lane wave (retry)".to_string(),
                );
                continue;
            }
            let key = (&*item.table, item.filter_idx);
            let group = match group_keys.iter().position(|k| *k == key) {
                Some(index) => index,
                None => {
                    group_keys.push(key);
                    group_needles.push(Vec::new());
                    group_positions.push(Vec::new());
                    group_keys.len() - 1
                }
            };
            group_needles[group].push(item.slot.1);
            group_positions[group].push(position);
        }
        for (group, &(table_name, filter_idx)) in group_keys.iter().enumerate() {
            let Some(table) = catalog.relational_catalog.get(table_name) else {
                for &position in &group_positions[group] {
                    violations.insert(position, "table dropped".to_string());
                }
                continue;
            };
            let locate = self.wave_batch_locate_hit_counts(
                table,
                filter_idx as usize,
                &group_needles[group],
            );
            let Some(counts) = locate else {
                // decline -> authoritative per-needle recheck (rare)
                for (&position, &needle) in group_positions[group]
                    .iter()
                    .zip(group_needles[group].iter())
                {
                    self.lane_authoritative_dup_check(
                        table,
                        filter_idx as usize,
                        needle,
                        batch[position].read_snapshot,
                        position,
                        &mut violations,
                    );
                }
                continue;
            };
            for ((&position, &needle), &count) in group_positions[group]
                .iter()
                .zip(group_needles[group].iter())
                .zip(counts.iter())
            {
                if count == 0 {
                    continue;
                }
                self.lane_authoritative_dup_check(
                    table,
                    filter_idx as usize,
                    needle,
                    batch[position].read_snapshot,
                    position,
                    &mut violations,
                );
            }
        }
        violations
    }

    fn lane_authoritative_dup_check(
        &self,
        table: &RelationalTable,
        filter_idx: usize,
        needle: i32,
        read_snapshot: Index,
        position: usize,
        violations: &mut std::collections::BTreeMap<usize, String>,
    ) {
        let visibility = crate::StorageVisibility {
            read_txn_id: read_snapshot,
        };
        let value = SqlValue::Int4(needle);
        match self.visible_row_with_value(table, visibility, filter_idx, &value, None) {
            Ok(true) => {
                let index_name = table
                    .columns
                    .get(filter_idx)
                    .map(|column| format!("{}_{}_key", table.name, column.name))
                    .unwrap_or_else(|| format!("{}_key", table.name));
                violations.insert(
                    position,
                    format!("duplicate key value violates unique index \"{index_name}\""),
                );
            }
            Ok(false) => {}
            Err(err) => {
                violations.insert(position, format!("unique validation failed: {err}"));
            }
        }
    }

    /// APPLY LEADER body: merge every pending lane request per table and run
    /// ONE open-shard append pass (rows + per-row created_by stamps + row ids).
    /// The leader lock serializes appends, so the PK-index extension chain
    /// (entry.row_count == base) is preserved exactly as under the old
    /// exclusive section — just batched across lanes. The non-appended
    /// fallback mirrors flush_wave_pending_appends' rehydrate/invalidate arm
    /// using only request-carried data (no CommitWaveItem).
    fn lane_apply_merged(&self, batch: &mut [crate::engine_intent_lanes::ApplyRequest]) {
        use std::collections::BTreeMap;
        // group request indexes per table (usually exactly one table)
        let mut tables: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, request) in batch.iter().enumerate() {
            tables.entry(request.table.clone()).or_default().push(index);
        }
        for (table, requests) in tables {
            let table = table.as_str();
            let total: usize = requests.iter().map(|&i| batch[i].rows.len()).sum();
            let mut rows: Vec<Vec<SqlValue>> = Vec::with_capacity(total);
            let mut row_ids: Vec<u64> = Vec::with_capacity(total);
            let mut stamps: Vec<Index> = Vec::with_capacity(total);
            for &i in &requests {
                // MOVE the row vectors (pointer moves) — the leader was cloning
                // every merged row's SqlValues, ~1900 heap allocs per wave.
                rows.append(&mut batch[i].rows);
                row_ids.extend_from_slice(&batch[i].row_ids);
                stamps.extend_from_slice(&batch[i].stamps);
            }
            let appended = self.auto_admit_on_commit_enabled()
                && self.try_append_resident_int4_open_shard(
                    table,
                    &rows,
                    crate::engine_residency::AppendCreatedBy::InsertPerRow(&stamps),
                    Some(&row_ids),
                );
            if appended {
                if self.host_install_elision_enabled() && !self.table_install_elided(table) {
                    let snapshot = self.catalog_snapshot();
                    if self.table_elision_eligible(&snapshot, table) {
                        self.set_table_install_elided(table, true);
                    }
                }
                continue;
            }
            // Fallback (rare on the lanes path — intents gate on elided,
            // auto-admit tables): rehydrate the merged batch as upserts and
            // invalidate per txn, mirroring flush_wave_pending_appends.
            if self.table_install_elided(table) {
                let first_seq = stamps.first().copied().unwrap_or_default();
                let last_seq = stamps.last().copied().unwrap_or_default();
                let upserts: BTreeMap<u64, Vec<SqlValue>> =
                    row_ids.iter().copied().zip(rows.iter().cloned()).collect();
                let catalog_table = self
                    .relational_catalog_table(table)
                    .expect("an elided table is in the catalog");
                self.rehydrate_elided_table(
                    &catalog_table,
                    first_seq.saturating_sub(1),
                    &upserts,
                    &Default::default(),
                    last_seq,
                )
                .unwrap_or_else(|err| {
                    panic!(
                        "commit-path invariant violation: elided rehydration for the \
                         merged lane append on {table} failed: {err}"
                    )
                });
            }
            let residency: std::collections::BTreeSet<String> =
                std::iter::once(table.to_string()).collect();
            for &i in &requests {
                for (offset, txn_id) in batch[i].txn_ids.iter().enumerate() {
                    self.invalidate_relational_residency_tables_concurrent(
                        &residency,
                        *txn_id,
                        batch[i].stamps[offset],
                    );
                }
            }
        }
    }

    /// Settle every lane wave whose end seq the visible cut covers (durable AND
    /// applied), publishing `committed_seq` to the global cut first so a polled
    /// Ok is never observable before the commit is readable.
    fn settle_intent_lane(
        &self,
        lanes: &std::sync::Arc<crate::engine_intent_lanes::IntentLaneState>,
        lane: usize,
    ) -> bool {
        // AUDIT F3 (+ apply poison): an ASYNC durability failure (fence pool
        // poison after the append returned Ok) OR a failed merged apply
        // permanently stalls the cut; without this check the stalled waves
        // would hang their clients forever instead of wedging loudly like the
        // classic path. The probes are lock-free flags; the mutex-walking
        // reason fetch (N poison locks) is paid only on an actual wedge.
        let wal_poisoned = lanes.wal_peek().is_some_and(|wal| wal.is_poisoned());
        if wal_poisoned
            || lanes
                .apply_poisoned
                .load(std::sync::atomic::Ordering::Acquire)
        {
            let reason = if wal_poisoned {
                let inner = lanes
                    .wal_peek()
                    .and_then(|wal| wal.poison_reason())
                    .unwrap_or_else(|| "lane wedged (reason pending)".to_string());
                format!("intent lane WAL poisoned: {inner}")
            } else {
                "intent lane apply leader failed; cut permanently holed".to_string()
            };
            let mut queue = lanes.settle[lane]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut settled = false;
            while let Some(entry) = queue.pop_front() {
                for item in entry.winners.into_iter().chain(entry.async_winners) {
                    item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                        reason.clone(),
                    ))));
                }
                settled = true;
            }
            return settled;
        }
        let local_cut = lanes.visible_local_cut();
        let global_cut = lanes.visible_global_cut();
        if global_cut > 0 {
            self.publish_committed_seq(global_cut);
        }
        let mut settled = false;
        let mut queue = lanes.settle[lane]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // ASYNC COMMIT tier (pg `synchronous_commit = off`): winners that
        // opted out of the durability wait ack as soon as the APPLIED cut
        // covers their wave — the WAL fence keeps running behind them.
        // Visibility (`publish_committed_seq` above) stays gated on the
        // STRICT cut, so readers never observe a row a power failure could
        // revoke; the async writer's own read-back lags by <= ~one fence
        // (documented deviation from pg, which exposes async commits
        // immediately).
        let applied_cut = lanes
            .applied_mirror
            .load(std::sync::atomic::Ordering::Acquire);
        for entry in queue.iter_mut() {
            if entry.end_seq > applied_cut {
                break;
            }
            if !entry.async_settled {
                entry.async_settled = true;
                for item in entry.async_winners.drain(..) {
                    let rows = item.rows_affected;
                    item.set_outcome(Ok(rows));
                }
                settled = true;
            }
        }
        while queue
            .front()
            .is_some_and(|entry| entry.end_seq <= local_cut)
        {
            let entry = queue.pop_front().expect("front checked");
            debug_assert!(
                entry
                    .apply_slot
                    .done
                    .load(std::sync::atomic::Ordering::Acquire),
                "cut covered a wave whose apply slot is not done"
            );
            lanes.stat_acklag_ns.fetch_add(
                entry.published_at.elapsed().as_nanos() as u64,
                AtomicOrdering::Relaxed,
            );
            lanes
                .stat_settled_waves
                .fetch_add(1, AtomicOrdering::Relaxed);
            for item in entry.winners.into_iter().chain(entry.async_winners) {
                let rows = item.rows_affected;
                item.set_outcome(Ok(rows));
            }
            settled = true;
        }
        settled
    }

    fn wait_group_durable(&self, wal_position: usize) -> Result<(), EngineError> {
        // E1 step 2 — FUA fence-pool backend: no single-flusher election. Every committer whose
        // record isn't yet covered runs its own `begin_group_flush` + `job.commit()` concurrently;
        // the WAL's ticket gate keeps frames ordered and the fence pool pipelines durability.
        if self.group_flush.concurrent_durability {
            return self.wait_group_durable_concurrent(wal_position);
        }
        loop {
            if self
                .group_flush
                .durable_records
                .load(AtomicOrdering::Acquire)
                >= wal_position
            {
                return Ok(());
            }
            let mut coord = self
                .group_flush
                .coord
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(msg) = &coord.failed {
                return Err(EngineError::Durability(format!(
                    "group-commit durability failed; the commit path is wedged pending restart \
                     recovery: {msg}"
                )));
            }
            // Re-check under the coordination lock (a flusher may have finished in between).
            if self
                .group_flush
                .durable_records
                .load(AtomicOrdering::Acquire)
                >= wal_position
            {
                return Ok(());
            }
            if coord.flusher_active {
                // A flush is in flight; wait for its result and re-evaluate.
                let _coord = self
                    .group_flush
                    .cv
                    .wait(coord)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                continue;
            }
            // Become the flusher. Lock order: coord is RELEASED before the commit_mutex is taken
            // (a flusher marks itself active first so no second fsync stacks up behind it).
            coord.flusher_active = true;
            drop(coord);
            // Two-phase group flush: snapshot the unflushed tail under a BRIEF commit_mutex hold,
            // then run the write + fsync with NO lock held — this is what lets other committers
            // validate/append/apply (and queue into the NEXT group) while this group's disk IO is
            // in flight. Completion takes only the WAL core's own lock, never the commit_mutex.
            let begun = {
                let mut commit = self.commit_state();
                commit.wal.begin_group_flush()
            };
            let flush_result = match begun {
                Ok(gpu_db_wal::WalGroupFlushBegin::Clean { flushed_records }) => {
                    Ok(flushed_records)
                }
                Ok(gpu_db_wal::WalGroupFlushBegin::Job(job)) => job.commit(),
                Err(err) => Err(err),
            };
            let mut coord = self
                .group_flush
                .coord
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            coord.flusher_active = false;
            match flush_result {
                Ok(flushed_records) => {
                    self.group_flush
                        .durable_records
                        .store(flushed_records, AtomicOrdering::Release);
                    self.group_flush.cv.notify_all();
                    // Loop: our own record was appended before this flush began, so the frontier
                    // now covers it (or a concurrent truncate shrank nothing below it — see the
                    // rollback paths, which only ever drop UNFLUSHED records they own).
                }
                Err(err) => {
                    coord.failed = Some(err.to_string());
                    self.group_flush.cv.notify_all();
                    drop(coord);
                    // Wedge-don't-serve-torn-state: poison the commit_mutex so the façade refuses
                    // further statements (see the doc comment above for why no clean abort exists).
                    let _commit = self.commit_state();
                    panic!(
                        "group-commit fsync failed after member deltas were applied: {err} — \
                         wedging the commit path; restart recovery replays the durable WAL prefix"
                    );
                }
            }
        }
    }

    /// E1 step 2 — the CONCURRENT-durability variant of [`Engine::wait_group_durable`], taken when
    /// the WAL's FUA fence-pool backend is active (`concurrent_durability`). There is NO
    /// single-flusher election: every committer whose record isn't yet covered snapshots + publishes
    /// its own group frame under a BRIEF commit_mutex hold (the WAL's ticket gate assigns frame
    /// order there) and then waits for the fence pool's contiguous durable cut OFF-LOCK, so MANY
    /// groups are durable in flight at once. A committer whose tail another thread already published
    /// (so `begin_group_flush` reports `Clean`) does NOT re-snapshot in a tight loop — it POLLS the
    /// shared `durable_records` mirror (spin-then-yield, no per-commit wakeup) until the owning
    /// thread's fence completes and advances it. Visibility still gates on the durable frontier
    /// exactly as the serial path; a fence/publish/roll failure wedges fail-closed identically.
    fn wait_group_durable_concurrent(&self, wal_position: usize) -> Result<(), EngineError> {
        loop {
            if self
                .group_flush
                .durable_records
                .load(AtomicOrdering::Acquire)
                >= wal_position
            {
                return Ok(());
            }
            if self.group_flush.wedged.load(AtomicOrdering::Acquire) {
                return Err(self.group_flush_wedged_error());
            }
            // FENCE-POOL PACING (E1 step 3 — the engine-seam anti-convoy law): snapshot + publish
            // our unflushed tail under a BRIEF commit_mutex hold (frame order is assigned there),
            // but ONLY if the fence pool has a free lane. While every lane is busy we do NOT begin
            // — a fresh begin here would frame the FEW records accumulated since the last publish
            // (the E1.2 negative: 61-record serial-election groups collapse to ~28), doubling the
            // durable-op count and starving the pool. Instead we release the lock and POLL the
            // durable mirror: whoever begins when a lane frees sweeps the WHOLE accumulated tail
            // (ours included) into ONE larger frame. This recreates serial-election batching but
            // with up to `lanes` groups pipelined instead of one. `fua_free_fence_slots` is `None`
            // on the serial backend (never reached here) → treat as "a lane is free".
            let begun = {
                let mut commit = self.commit_state();
                if commit.wal.fua_free_fence_slots().unwrap_or(1) == 0 {
                    None // all lanes busy → accumulate + poll (do not ship a tiny frame)
                } else {
                    Some(commit.wal.begin_group_flush())
                }
            };
            match begun {
                None => {
                    // Pacing back-off: another begin will cover us once a lane frees. Poll the
                    // durable mirror off-lock; on budget-elapse re-loop to re-check the pacing gate.
                    if self.poll_concurrent_durable_mirror(wal_position)? {
                        return Ok(());
                    }
                }
                Some(Ok(gpu_db_wal::WalGroupFlushBegin::Clean { flushed_records })) => {
                    // Another committer already published our tail and owns the in-flight frame that
                    // covers us; it will advance `durable_records` when its fence completes. Refresh
                    // the mirror with the snapshot's durable cut, then POLL (no wakeup) until either
                    // the frontier covers us or the owner wedges.
                    self.group_flush
                        .durable_records
                        .fetch_max(flushed_records, AtomicOrdering::AcqRel);
                    if self.poll_concurrent_durable_mirror(wal_position)? {
                        return Ok(());
                    }
                }
                Some(Ok(gpu_db_wal::WalGroupFlushBegin::Job(job))) => match job.commit() {
                    Ok(flushed_records) => {
                        self.group_flush
                            .durable_records
                            .fetch_max(flushed_records, AtomicOrdering::AcqRel);
                        // Loop: our own record was appended before this frame was published, so the
                        // frontier now covers it.
                    }
                    Err(err) => return Err(self.wedge_group_flush(err)),
                },
                Some(Err(err)) => return Err(self.wedge_group_flush(err)),
            }
        }
    }

    /// Spin-then-yield poll of the concurrent durable mirror (no per-commit wakeup), bounded by a
    /// yield budget. Returns `Ok(true)` when the frontier covers `wal_position`, `Ok(false)` when
    /// the budget elapses (the caller re-loops to re-check the pacing gate / a freed fence lane —
    /// this also surfaces a poisoned backend that died mid-flight without setting `wedged`, far
    /// beyond one fence latency), and `Err` when the path is wedged. Pure spin at low contention
    /// keeps the ack near one fence latency; the yield fallback avoids burning a core when the pool
    /// is genuinely backed up.
    fn poll_concurrent_durable_mirror(&self, wal_position: usize) -> Result<bool, EngineError> {
        const SPIN_BEFORE_YIELD: u32 = 256;
        const POLL_YIELD_BUDGET: u32 = 1 << 16;
        let mut spins: u32 = 0;
        let mut yields: u32 = 0;
        loop {
            if self
                .group_flush
                .durable_records
                .load(AtomicOrdering::Acquire)
                >= wal_position
            {
                return Ok(true);
            }
            if self.group_flush.wedged.load(AtomicOrdering::Acquire) {
                return Err(self.group_flush_wedged_error());
            }
            if spins < SPIN_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
                yields += 1;
                if yields >= POLL_YIELD_BUDGET {
                    return Ok(false);
                }
            }
        }
    }

    /// E1 step 2 — wedge the concurrent group-flush path fail-closed after a fence/publish/roll
    /// failure whose group's member deltas were already applied (they can never be published, and
    /// nothing later may publish over them). Records the sticky failure under the coordination lock,
    /// flips the lock-free `wedged` mirror so POLLING waiters bail, and PANICS while holding the
    /// commit_mutex — poisoning it so the façade refuses further service (restart recovery replays
    /// the durable WAL prefix; the un-fsynced records were never acknowledged nor visible). This is
    /// the same wedge-don't-serve-torn-state policy the serial path's flusher applies.
    fn wedge_group_flush(&self, err: EngineError) -> EngineError {
        {
            let mut coord = self
                .group_flush
                .coord
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if coord.failed.is_none() {
                coord.failed = Some(err.to_string());
            }
        }
        self.group_flush.wedged.store(true, AtomicOrdering::Release);
        let _commit = self.commit_state();
        panic!(
            "group-commit FUA durability failed after member deltas were applied: {err} — \
             wedging the commit path; restart recovery replays the durable WAL prefix"
        );
    }

    /// The sticky-failure error a concurrent waiter returns once the path is wedged.
    fn group_flush_wedged_error(&self) -> EngineError {
        let coord = self
            .group_flush
            .coord
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let msg = coord.failed.clone().unwrap_or_else(|| "wedged".to_string());
        EngineError::Durability(format!(
            "group-commit durability failed; the commit path is wedged pending restart \
             recovery: {msg}"
        ))
    }

    pub fn execute_text(&self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        let timestamp_micros = self.next_commit_timestamp_micros();
        self.execute_text_at_timestamp_micros(txn_id, text, timestamp_micros)
    }

    pub fn execute_text_at_timestamp_micros(
        &self,
        txn_id: u64,
        text: &str,
        timestamp_micros: u64,
    ) -> Result<(), ExecuteError> {
        let result = self.execute_text_at_timestamp_micros_inner(txn_id, text, timestamp_micros);
        // VACUUM #5: run a commit-parked auto-vacuum now — the commit lock + catalog latch are
        // released, so `vacuum_table` can take them fresh (the in-commit trigger would deadlock).
        let parked = self
            .read_state
            .residency
            .pending_auto_vacuum
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(table) = parked {
            // Audit F1: the statement is already durable + published — a MAINTENANCE failure
            // must not turn its result into an error (a client retry would double-apply). The
            // scan still serves dead slots correctly; the churn counter was NOT reset, so the
            // trigger re-arms on the table's next tombstone. Count it and move on.
            if self.vacuum_table(&table).is_err() {
                self.read_state
                    .residency
                    .auto_vacuum_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        // W1b: the serialized path's maintenance point — commit lock + catalog latch are
        // released here, so the rotation can take them fresh (cheap under the size bound).
        self.maybe_auto_checkpoint_wal();
        result
    }

    fn execute_text_at_timestamp_micros_inner(
        &self,
        txn_id: u64,
        text: &str,
        timestamp_micros: u64,
    ) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;
        // RETIREMENT A4e (audit B1-DDL/B2): any NON-DML command rehydrates every elided table
        // FIRST (under the commit lock via the serialized helper) — DDL preflights/validators
        // (ADD UNIQUE/PK/FK, CREATE INDEX) read the host store via visible_relational_rows, and a
        // stale prefix would validate a constraint over data that violates it. DDL is rare and
        // the elided set is tiny; the blunt sweep is the safe shape.
        if self.host_install_elision_enabled()
            && !matches!(
                cmd,
                Command::Insert(_) | Command::Update(_) | Command::Delete(_) | Command::Select(_)
            )
        {
            let elided: Vec<String> = self
                .read_state
                .residency
                .elided_tables
                .load()
                .iter()
                .cloned()
                .collect();
            for table_name in elided {
                self.rehydrate_elided_serialized(&table_name)
                    .map_err(ExecuteError::Engine)?;
            }
        }

        match cmd {
            Command::SetKv { .. }
            | Command::DeleteKv { .. }
            | Command::CreateSchema(_)
            | Command::DropSchema(_)
            | Command::CreateDatabase(_)
            | Command::DropDatabase(_)
            | Command::RenameDatabase(_)
            | Command::CreateTablespace(_)
            | Command::DropTablespace(_)
            | Command::RenameTablespace(_)
            | Command::CreateTable(_)
            | Command::AddPrimaryKey(_)
            | Command::AddUniqueConstraint(_)
            | Command::AddCheckConstraint(_)
            | Command::AddForeignKey(_)
            | Command::AddColumn(_)
            | Command::RenameTable(_)
            | Command::RenameColumn(_)
            | Command::RenameConstraint(_)
            | Command::DropColumn(_)
            | Command::DropConstraint(_)
            | Command::CreateIndex(_)
            | Command::RenameIndex(_)
            | Command::CreateView(_)
            | Command::RenameView(_)
            | Command::CreateMaterializedView(_)
            | Command::RefreshMaterializedView(_)
            | Command::RenameMaterializedView(_)
            | Command::CreateFunction(_)
            | Command::RenameFunction(_)
            | Command::DropFunction(_)
            | Command::CreateSequence(_)
            | Command::CreateDomain(_)
            | Command::SequenceNextVal(_)
            | Command::SequenceSetVal(_)
            | Command::RenameSequence(_)
            | Command::DropTable(_)
            | Command::TruncateTable(_)
            | Command::DropIndex(_)
            | Command::DropView(_)
            | Command::DropMaterializedView(_)
            | Command::DropSequence(_)
            | Command::DropDomain(_)
            | Command::GrantTable(_)
            | Command::RevokeTable(_)
            | Command::GrantSchema(_)
            | Command::RevokeSchema(_)
            | Command::GrantDatabase(_)
            | Command::RevokeDatabase(_)
            | Command::GrantTablespace(_)
            | Command::RevokeTablespace(_)
            | Command::GrantFunction(_)
            | Command::RevokeFunction(_)
            | Command::CreatePublication(_)
            | Command::DropPublication(_)
            | Command::CreateSubscription(_)
            | Command::DropSubscription(_)
            | Command::CreateRole(_)
            | Command::DropRole(_)
            | Command::RenameRole(_)
            | Command::GrantDefaultTablePrivileges(_)
            | Command::RevokeDefaultTablePrivileges(_)
            | Command::AlterColumnDefault(_)
            | Command::CommentOn(_)
            | Command::Insert(_)
            | Command::Delete(_)
            | Command::Update(_) => {
                self.preflight_unique_index_constraints(&cmd, txn_id)?;
                match self.route_command(&cmd) {
                    RouteDecision::Gpu(_) | RouteDecision::Cpu => {
                        self.commit_mutation_at(
                            txn_id,
                            std::sync::Arc::from(text.as_bytes()),
                            timestamp_micros,
                        )?;
                    }
                    RouteDecision::CpuFallback { reason, .. } => {
                        self.metrics.inc_gpu_fallback(reason);
                        self.commit_mutation_at(
                            txn_id,
                            std::sync::Arc::from(text.as_bytes()),
                            timestamp_micros,
                        )?;
                    }
                }
            }
            Command::Flush => {
                self.flush_admin()?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::ResetAll | Command::SetRole { .. } => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::CreateExtension(create) => {
                validate_bootstrap_create_extension(&create).map_err(ExecuteError::Engine)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::DropExtension(drop) => {
                validate_bootstrap_drop_extension(&drop).map_err(ExecuteError::Engine)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Begin => {
                self.commit_state().txn_manager.begin_with_id(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                self.commit_state().txn_manager.commit(txn_id)?;
                if chain {
                    self.commit_state().txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                self.commit_state().txn_manager.rollback(txn_id)?;
                if chain {
                    self.commit_state().txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { key } => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.commit_state().sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
            }
            Command::Select(_) | Command::SelectFunction(_) | Command::SequenceCurrVal(_) => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
        }

        Ok(())
    }

    pub fn execute_read_text(&mut self, text: &str) -> Result<Option<String>, ExecuteError> {
        let cmd = parse_command(text)?;

        match cmd {
            Command::GetKv { key } => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }

                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.commit_state_mut().sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
                Ok(self.get(&key))
            }
            Command::Begin => Err(ExecuteError::NonReadCommand("BEGIN")),
            Command::Commit { .. } => Err(ExecuteError::NonReadCommand("COMMIT")),
            Command::Rollback { .. } => Err(ExecuteError::NonReadCommand("ROLLBACK")),
            Command::Flush => Err(ExecuteError::NonReadCommand("FLUSH")),
            Command::ResetAll => Err(ExecuteError::NonReadCommand("RESET ALL")),
            Command::SetRole { .. } => Err(ExecuteError::NonReadCommand("SET ROLE")),
            Command::SetKv { .. } => Err(ExecuteError::NonReadCommand("SET")),
            Command::DeleteKv { .. } => Err(ExecuteError::NonReadCommand("DEL/DELETE")),
            Command::CreateSchema(_) => Err(ExecuteError::NonReadCommand("CREATE SCHEMA")),
            Command::DropSchema(_) => Err(ExecuteError::NonReadCommand("DROP SCHEMA")),
            Command::CreateDatabase(_) => Err(ExecuteError::NonReadCommand("CREATE DATABASE")),
            Command::DropDatabase(_) => Err(ExecuteError::NonReadCommand("DROP DATABASE")),
            Command::RenameDatabase(_) => Err(ExecuteError::NonReadCommand("ALTER DATABASE")),
            Command::CreateTablespace(_) => Err(ExecuteError::NonReadCommand("CREATE TABLESPACE")),
            Command::DropTablespace(_) => Err(ExecuteError::NonReadCommand("DROP TABLESPACE")),
            Command::RenameTablespace(_) => Err(ExecuteError::NonReadCommand("ALTER TABLESPACE")),
            Command::CreateTable(_) => Err(ExecuteError::NonReadCommand("CREATE TABLE")),
            Command::AddPrimaryKey(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddUniqueConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddCheckConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddForeignKey(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddColumn(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::RenameTable(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::RenameColumn(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::RenameConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::DropColumn(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::DropConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::CreateIndex(_) => Err(ExecuteError::NonReadCommand("CREATE INDEX")),
            Command::RenameIndex(_) => Err(ExecuteError::NonReadCommand("ALTER INDEX")),
            Command::CreateView(_) => Err(ExecuteError::NonReadCommand("CREATE VIEW")),
            Command::RenameView(_) => Err(ExecuteError::NonReadCommand("ALTER VIEW")),
            Command::CreateMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("CREATE MATERIALIZED VIEW"))
            }
            Command::CreateExtension(_) => Err(ExecuteError::NonReadCommand("CREATE EXTENSION")),
            Command::DropExtension(_) => Err(ExecuteError::NonReadCommand("DROP EXTENSION")),
            Command::RefreshMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("REFRESH MATERIALIZED VIEW"))
            }
            Command::RenameMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("ALTER MATERIALIZED VIEW"))
            }
            Command::CreateFunction(_) => Err(ExecuteError::NonReadCommand("CREATE FUNCTION")),
            Command::RenameFunction(_) => Err(ExecuteError::NonReadCommand("ALTER FUNCTION")),
            Command::DropFunction(_) => Err(ExecuteError::NonReadCommand("DROP FUNCTION")),
            Command::CreateSequence(_) => Err(ExecuteError::NonReadCommand("CREATE SEQUENCE")),
            Command::CreateDomain(_) => Err(ExecuteError::NonReadCommand("CREATE DOMAIN")),
            Command::SequenceNextVal(_) => Err(ExecuteError::NonReadCommand("SELECT nextval")),
            Command::SequenceSetVal(_) => Err(ExecuteError::NonReadCommand("SELECT setval")),
            Command::RenameSequence(_) => Err(ExecuteError::NonReadCommand("ALTER SEQUENCE")),
            Command::DropTable(_) => Err(ExecuteError::NonReadCommand("DROP TABLE")),
            Command::TruncateTable(_) => Err(ExecuteError::NonReadCommand("TRUNCATE TABLE")),
            Command::DropIndex(_) => Err(ExecuteError::NonReadCommand("DROP INDEX")),
            Command::DropView(_) => Err(ExecuteError::NonReadCommand("DROP VIEW")),
            Command::DropMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("DROP MATERIALIZED VIEW"))
            }
            Command::DropSequence(_) => Err(ExecuteError::NonReadCommand("DROP SEQUENCE")),
            Command::DropDomain(_) => Err(ExecuteError::NonReadCommand("DROP DOMAIN")),
            Command::GrantTable(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeTable(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantSchema(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeSchema(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantDatabase(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeDatabase(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantTablespace(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeTablespace(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantFunction(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeFunction(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::CreatePublication(_) => {
                Err(ExecuteError::NonReadCommand("CREATE PUBLICATION"))
            }
            Command::DropPublication(_) => Err(ExecuteError::NonReadCommand("DROP PUBLICATION")),
            Command::CreateSubscription(_) => {
                Err(ExecuteError::NonReadCommand("CREATE SUBSCRIPTION"))
            }
            Command::DropSubscription(_) => Err(ExecuteError::NonReadCommand("DROP SUBSCRIPTION")),
            Command::CreateRole(_) => Err(ExecuteError::NonReadCommand("CREATE ROLE")),
            Command::DropRole(_) => Err(ExecuteError::NonReadCommand("DROP ROLE")),
            Command::RenameRole(_) => Err(ExecuteError::NonReadCommand("ALTER ROLE")),
            Command::GrantDefaultTablePrivileges(_) => {
                Err(ExecuteError::NonReadCommand("ALTER DEFAULT PRIVILEGES"))
            }
            Command::RevokeDefaultTablePrivileges(_) => {
                Err(ExecuteError::NonReadCommand("ALTER DEFAULT PRIVILEGES"))
            }
            Command::AlterColumnDefault(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::CommentOn(_) => Err(ExecuteError::NonReadCommand("COMMENT")),
            Command::Insert(_) => Err(ExecuteError::NonReadCommand("INSERT")),
            Command::Delete(_) => Err(ExecuteError::NonReadCommand("DELETE")),
            Command::Update(_) => Err(ExecuteError::NonReadCommand("UPDATE")),
            Command::Select(_) | Command::SelectFunction(_) | Command::SequenceCurrVal(_) => {
                Err(ExecuteError::NonReadCommand("SELECT"))
            }
        }
    }

    /// Pin one statement-stable relational read snapshot at the ALREADY-CHOSEN boundary `s` (PART B
    /// catalog↔data co-pinning). The boundary `s` was loaded ONCE per statement (by
    /// [`Engine::bind_relational_select_for_execution`], which also selected the catalog as-of `s`), so
    /// the data this pins and the catalog the statement bound against are the SAME generation — a
    /// concurrent shape-changing DDL can never split the reader's (catalog, data) pair. Pins ONE
    /// generation of `table` (its rows + value-index together) at `s`.
    pub(crate) fn pin_relational_read_at(&self, table: &str, s: Index) -> RelationalReadPin {
        RelationalReadPin {
            visibility: StorageVisibility { read_txn_id: s },
            table_rows: self.read_state.mvcc.table_rows(table),
        }
    }
}

/// M1 design B (perf): bind the needle by catalog COLUMN INDEX (no column-name search — the
/// caller cached the filter_idx). Same coercion + strict `i32_section_needle` encode.
fn insert_i32_unique_needle_at(
    insert: &Insert,
    table: &RelationalTable,
    filter_idx: usize,
) -> Option<(usize, i32)> {
    let unique_column = &table.columns.get(filter_idx)?.name;
    let column_ty = table.columns[filter_idx].ty;
    let row = insert.rows.first()?;
    // Where does this column's value sit in the insert row? Explicit column list -> its index;
    // empty column list -> catalog order (== filter_idx).
    let source_pos = if insert.columns.is_empty() {
        filter_idx
    } else {
        insert.columns.iter().position(|c| c == unique_column)?
    };
    let raw = row.get(source_pos)?.clone();
    let coerced = coerce_filter_literal(raw, column_ty);
    let needle = crate::engine_residency::i32_section_needle(column_ty, &coerced)?;
    Some((filter_idx, needle))
}
