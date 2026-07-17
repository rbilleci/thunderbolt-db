//! Concurrent DML path + top-level text execution (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for the snapshot-isolation
//! autocommit write path (is_concurrent_dml, register/deregister_active_snapshot,
//! execute_dml_concurrent[_instrumented], prepare_dml, dml_mutated_tables,
//! commit_dml_concurrent), the execute_text / execute_text_at_timestamp_micros /
//! execute_read_text entry points, and the read-pin helper.

use super::*;

mod lane;
mod lane_apply;
mod state;
mod wave;

pub(crate) use state::{
    new_pending_outcome, CommitWaveItem, CommitWaveOutcome, CommitWaveState, LaneIntent, LaneOpKind,
};
#[cfg(test)]
use state::{wave_tail_failure_publish_hook, wave_tail_handoff_hook};
use state::{CommitWaveDone, CommitWaveQueue, CommitWaveTail, TailMaintenanceCompletion};

/// Upper bound on items per wave drain (keeps a single wave's worst-case commit latency bounded;
/// under saturation the NEXT wave picks the rest up immediately).
const COMMIT_WAVE_MAX_DEFAULT: usize = 1024;

/// Parallel vectors returned by one device launch form a single verdict. Every component must
/// cover exactly the submitted positions: accepting a short vector makes `zip` fail open, while
/// accepting a long vector hides an ABI/launch mismatch that can misassociate later results.
#[inline]
fn exact_device_verdict_cardinality(expected: usize, component_lengths: &[usize]) -> bool {
    component_lengths.iter().all(|length| *length == expected)
}

fn is_device_prepare_verdict_unavailable(error: &ExecuteError) -> bool {
    matches!(
        error,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message.starts_with("device DML verdict unavailable")
                || message.starts_with("device constraint verdict unavailable")
                || message.starts_with("device tuple-constraint verdict unavailable")
    )
}

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

impl Engine {
    /// Central fail-stop drain. The sticky engine flag is stored before this method runs, so any
    /// racing submit/pump observes the gate even if its item is between local pipeline stages.
    pub(crate) fn fail_all_pending_commit_work(&self, reason: &str) {
        let error = || {
            ExecuteError::Engine(EngineError::Durability(format!(
                "commit path is wedged pending restart recovery: {reason}"
            )))
        };

        let stranded = {
            let mut queue = self.lock_commit_wave_queue();
            queue.wedged.get_or_insert_with(|| reason.to_string());
            queue.sequencer_active = false;
            queue.items.drain(..).collect::<Vec<_>>()
        };
        for item in stranded {
            item.set_outcome(Err(error()));
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
        drop(pending_tails); // armed Drop fails every tail member
        if pending_tail_count != 0 {
            self.commit_wave
                .tails_finished
                .fetch_add(pending_tail_count, AtomicOrdering::Release);
        }

        if let Some(lanes) = &self.intent_lanes {
            lanes.apply_poisoned.store(true, AtomicOrdering::Release);
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
            // Lane publishers install an ApplyRequest and its LaneSettle while holding these in
            // this same order. Holding both sets across the drain prevents an active pump from
            // publishing one half after the other half was already swept.
            let mut apply_queue = lanes
                .apply_queue
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut settle_queues = lanes
                .settle
                .iter()
                .map(|queue| {
                    queue
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                })
                .collect::<Vec<_>>();
            for queue in &mut settle_queues {
                for entry in queue.drain(..) {
                    intents.extend(entry.winners);
                    intents.extend(entry.async_winners);
                }
            }
            for request in apply_queue.drain(..) {
                request.slot.failed.store(true, AtomicOrdering::Release);
                request.slot.done.store(true, AtomicOrdering::Release);
                let mut inflight = lanes.inflight_slots[request.lane]
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                for slot in request.unique_slots {
                    inflight.remove(&slot);
                }
            }
            drop(settle_queues);
            drop(apply_queue);
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
                item.set_outcome(Err(error()));
            }
        }
        self.commit_wave.cv.notify_all();
    }

    #[cfg(test)]
    pub(crate) fn set_wave_tail_handoff_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *wave_tail_handoff_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    #[cfg(test)]
    pub(crate) fn set_wave_tail_failure_publish_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *wave_tail_failure_publish_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

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

    /// Register an autocommit statement's scalar read boundary for the oldest-active
    /// GC/ledger-prune boundary, returning a guard that deregisters on drop (write-half MVCC,
    /// Stage 4). Explicit transactions use the generation-owned keyed registration below.
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

    /// Capture the exact catalog, MVCC table generations, and resident GPU resource descriptors at
    /// `boundary`. The caller holds `commit`, so no DML/DDL publisher can advance the logical state
    /// while the bundle is assembled. The descriptor publisher lock makes the single-buffer and
    /// shard map loads one residency publication observation; both descriptor kinds own the device
    /// allocations they describe. Cached device indexes are lifetime-pinned as optional accelerators
    /// and remain subject to their existing buffer-identity validation before execution.
    pub(crate) fn capture_transaction_snapshot(&self, boundary: Index) -> Arc<TransactionSnapshot> {
        let catalog = self.read_state.catalog_as_of(boundary);
        let table_versions = self.read_state.mvcc.capture_table_versions();
        let (
            resident_snapshots,
            resident_shards,
            elided_tables,
            chunk_authoritative_tables,
            streaming_cold_chunks,
        ) = {
            let _publish = self
                .read_state
                .residency
                .descriptor_publish_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                self.read_state.residency.snapshots.load_full(),
                self.read_state.residency.shards.load_full(),
                self.read_state.residency.elided_tables.load_full(),
                self.read_state
                    .residency
                    .chunk_authoritative_tables
                    .load_full(),
                self.read_state.residency.streaming_cold_chunks.load_full(),
            )
        };
        let mut resident_index_resources = self
            .read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .filter_map(|index| index.index_memory.as_ref().cloned())
            .collect::<Vec<_>>();
        resident_index_resources.extend(
            self.read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .values()
                .filter_map(|index| index.device_index.as_ref().cloned()),
        );
        resident_index_resources.extend(
            self.read_state
                .residency
                .chunk_key_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .values()
                .map(|index| Arc::clone(&index.device)),
        );
        resident_index_resources.extend(
            self.read_state
                .residency
                .chunk_key_bloom
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .values()
                .map(|bloom| Arc::clone(&bloom.device)),
        );
        Arc::new(TransactionSnapshot {
            boundary,
            next_row_id: self.read_state.mvcc.current_row_id(),
            catalog,
            table_versions,
            resident_snapshots,
            resident_shards: Arc::clone(&resident_shards),
            elided_tables,
            chunk_authoritative_tables,
            delta: std::sync::Mutex::new(TransactionDeltaState {
                generation: 0,
                resident_shards: Arc::clone(&resident_shards),
                streaming_cold_chunks: Arc::clone(&streaming_cold_chunks),
                deltas: Vec::new(),
                write_set: WriteSet::default(),
                next_row_id: self.read_state.mvcc.current_row_id(),
                sequence_state: BTreeMap::new(),
                private_gpu_bytes_by_gpu: BTreeMap::new(),
            }),
            statement_lock: std::sync::Mutex::new(()),
            private_gpu_account: Arc::clone(&self.transaction_private_gpu_bytes),
            _resident_index_resources: resident_index_resources,
        })
    }

    /// Open an explicit transaction and pin its one lifetime generation bundle: visibility boundary,
    /// catalog, table versions, and resident GPU resource descriptors. Its scalar boundary remains
    /// in the same space used by conflict-ledger pruning and MVCC GC. Lock order is `commit` ->
    /// residency publishers/caches -> `active_snapshots`.
    pub(crate) fn begin_transaction_context(&self, txn_id: TxnId) -> Result<(), TxnError> {
        let mut commit = self.commit_state();
        let snapshot = self.capture_transaction_snapshot(self.committed_seq());
        let txn = commit.txn_manager.begin_with_id(txn_id)?;
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .register_transaction(txn.id, snapshot);
        Ok(())
    }

    /// Finish an explicit transaction, releasing its lifetime snapshot. `AND CHAIN` creates the
    /// successor while the same locks are held and registers a fresh boundary for that new identity,
    /// so there is no un-fenced gap between the two transaction contexts.
    pub(crate) fn finish_transaction_context(
        &self,
        txn_id: TxnId,
        committed: bool,
        chain: bool,
    ) -> Result<Option<TxnId>, TxnError> {
        let mut commit = self.commit_state();
        if committed {
            commit.txn_manager.commit(txn_id)?;
        } else {
            commit.txn_manager.rollback(txn_id)?;
        }

        let successor = if chain {
            let next = commit.txn_manager.begin()?;
            Some((
                next.id,
                self.capture_transaction_snapshot(self.committed_seq()),
            ))
        } else {
            None
        };
        let mut active = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed = active.deregister_transaction(txn_id);
        debug_assert!(
            removed.is_some(),
            "active transaction {txn_id} had no lifetime snapshot"
        );
        let successor_id = successor.as_ref().map(|(next_id, _)| *next_id);
        if let Some((next_id, next_snapshot)) = successor {
            active.register_transaction(next_id, next_snapshot);
        }
        drop(active);
        self.gc_transaction_created_by_regions();
        Ok(successor_id)
    }

    /// Execute one autocommit DML statement (`INSERT`/`UPDATE`/`DELETE`) on the CONCURRENT commit
    /// path under Snapshot Isolation (write-half MVCC, Stage 4 — the concurrency flip):
    ///
    /// 1. **Begin (off-lock):** pin a read snapshot `S = committed_seq` and register it.
    /// 2. **Prepare (off-lock, no commit_mutex):** parse, constraint-preflight against `S`, and
    ///    compute the conflict write-set (`prepare_*` at `S`). Many writers run this concurrently,
    ///    and concurrently with lock-free readers.
    /// 3. **Commit (short critical section under the commit_mutex):** validate row identities
    ///    against the recent-commits map and unique keys against device version history plus
    ///    wave-local arbitration (overlap since `S` ⇒ retryable [`ExecuteError::Serialization`],
    ///    first-committer-wins) → assign `commit_seq` (the commit `Index`) → WAL append + group-commit
    ///    fsync → install the delta RE-RESOLVED at `commit_seq` (so the live apply is byte-identical
    ///    to a WAL replay) + publish the table generation → record row identities → bump
    ///    `committed_seq` LAST (release-store: the publish point).
    /// 4. **Abort/retry:** a conflict (or any prepare error) publishes nothing and is returned; a
    ///    serialization conflict is retryable with a fresh snapshot.
    ///
    /// `&self`: the whole path runs without an engine write lock, so writers overlap on prepare and
    /// serialize only briefly on the commit_mutex, and a writer never blocks a reader.
    pub fn execute_dml_concurrent(&self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        self.intent_lanes_write_guard()
            .map_err(ExecuteError::Engine)?;
        if self.transaction_snapshot_handle(txn_id).is_some() {
            return self.execute_dml_in_transaction(txn_id, text);
        }
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
        if self.is_commit_path_poisoned() {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "commit path is wedged; restart recovery required".to_string(),
            )));
        }
        if self.transaction_snapshot_handle(txn_id).is_some() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "instrumented concurrent DML is autocommit-only; an active transaction must use the serialized private-generation entry"
                    .to_string(),
            )));
        }
        let cmd = parse_command(text)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        // A table's very first write may precede any data-triggered admission. Build its
        // empty/current device generation under the same publication cut before the off-lock
        // prepare; thereafter every authoritative DML and constraint verdict remains device-current.
        self.ensure_dml_device_generation(&cmd)?;
        // (1) Begin: an explicit transaction reuses its lifetime generation and registry hold;
        // an autocommit statement captures and temporarily registers the newest scalar boundary.
        // The retained scope is entered ONLY around prepare below. Letting it leak into the wave
        // sequencer's commit-time re-resolve would make a publisher mutate against old resources.
        let transaction_snapshot = self.transaction_snapshot_handle(txn_id);
        let read_snapshot = transaction_snapshot
            .as_ref()
            .map_or_else(|| self.committed_seq(), |snapshot| snapshot.boundary);
        let _snapshot_guard = transaction_snapshot
            .is_none()
            .then(|| self.register_active_snapshot(read_snapshot));

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
        if transaction_snapshot.is_none()
            && self.host_install_elision_enabled()
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
        let prepared_catalog_seq = transaction_snapshot.as_ref().map_or_else(
            || self.catalog_snapshot().commit_seq,
            |snapshot| snapshot.catalog.commit_seq,
        );
        let snapshot = transaction_snapshot.as_ref().map_or_else(
            || self.dml_read_snapshot(read_snapshot),
            |generation| DmlReadSnapshot {
                commit_seq: generation.boundary,
                next_row_id: generation.next_row_id,
            },
        );
        let prepared = {
            const GENERATION_RETRIES: usize = 64;
            let mut attempts = 0;
            loop {
                let result = if let Some(generation) = transaction_snapshot.as_ref() {
                    let _scope = self.enter_transaction_read(Arc::clone(generation));
                    self.prepare_dml(&cmd, snapshot, InsertPrepareValidation::Full)
                } else {
                    self.prepare_dml(&cmd, snapshot, InsertPrepareValidation::Full)
                };
                match result {
                    Ok(prepared) => break prepared,
                    Err(error)
                        if attempts < GENERATION_RETRIES
                            && is_device_prepare_verdict_unavailable(&error) =>
                    {
                        attempts += 1;
                        // A classic commit can replace/invalidate a generation between the
                        // pre-prepare admission and this off-lock probe. Re-enter the same
                        // device-only admission barrier and retry at the already-pinned snapshot;
                        // unsupported predicates still exhaust the bound and fail loud.
                        self.ensure_dml_device_generation(&cmd)?;
                        std::thread::yield_now();
                    }
                    Err(error) => return Err(error),
                }
            }
        };
        let residency_tables = Self::dml_mutated_tables(&cmd);

        // The snapshot is now pinned and prepare is done; the commit critical section has not started.
        // (Tests barrier here to align two writers' snapshots before their commits race.)
        on_prepared();

        // Carry the off-lock delta forward for device unique-history validation. Reuse-eligible
        // inserts may additionally re-key this delta under the catalog-generation gate; every
        // other shape still takes the normal authoritative re-prepare after conflict validation.
        let write_set = prepared.write_set.clone();
        let offlock_delta = Some(prepared);

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
    /// `ReResolveDeviceCovered` only from the sequencer's under-lock re-resolve (device history +
    /// wave-local arbitration —
    /// the coverage proof lives on [`InsertPrepareValidation`]).
    pub(crate) fn prepare_dml(
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
    /// constraint re-validation beyond that device verdict. The generation gate
    /// (`ReResolveDeviceCovered`) is
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
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let outcome = Arc::clone(&item.outcome);
        let mut queue = self.lock_commit_wave_queue();
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
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
        if self.ensure_commit_path_available().is_err() {
            return false;
        }
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
            if let Err(error) = self.ensure_commit_path_available() {
                let message = error.to_string();
                for item in batch {
                    item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                        message.clone(),
                    ))));
                }
                let mut queue = self.lock_commit_wave_queue();
                queue.sequencer_active = false;
                drop(queue);
                self.commit_wave.cv.notify_all();
                return;
            }
            let wave_started = std::time::Instant::now();
            let wave_len = batch.len() as u64;
            let tail = self.sequence_commit_wave(batch);
            #[cfg(test)]
            if tail.is_some() {
                let hook = wave_tail_handoff_hook()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
                if let Some((engine, reached, resume)) = hook {
                    if engine == self as *const Self as usize {
                        reached.wait();
                        resume.wait();
                    }
                }
            }
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
                // Publish the maintenance obligation before the tail can be claimed and finished.
                // A transaction observing the finished count must therefore also observe either
                // the completed re-admission or this pending count.
                let maintenance =
                    (!admit_tables.is_empty()).then(|| TailMaintenanceCompletion::new(self));
                let mut tail = Some(tail);
                let handed = {
                    let mut tails = self
                        .commit_wave
                        .pending_tails
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if self.is_commit_path_poisoned() {
                        false
                    } else {
                        tails.push_back(tail.take().expect("tail is handed once"));
                        true
                    }
                };
                if !handed {
                    drop(tail); // armed Drop fails every member outcome
                    self.commit_wave
                        .tails_finished
                        .fetch_add(1, AtomicOrdering::Release);
                    let mut queue = self.lock_commit_wave_queue();
                    queue.sequencer_active = false;
                    drop(queue);
                    self.commit_wave.cv.notify_all();
                    return;
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
                drop(maintenance);
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
    pub(crate) fn wait_wave_tail_capacity(&self, max_outstanding: u64) -> bool {
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
                let wedge = !self.clean;
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
                if wedge {
                    // Publish the sticky engine-wide fail-stop BEFORE the release-store that tells
                    // barrier waiters this tail is finished. A COMMIT that observes the finished
                    // counter through Acquire must therefore also observe the wedge before it can
                    // claim identities or append WAL.
                    self.engine.wedge_commit_path();
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
        if self.is_commit_path_poisoned() {
            drop(tail); // armed: fail outcomes; a wedged service publishes no later visibility
            return;
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
        if self.is_commit_path_poisoned() {
            return false;
        }
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
        if self.is_commit_path_poisoned() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "commit path is wedged; restart recovery required".to_string(),
            )));
        }
        let cmd = parse_command(text)?;
        if self.transaction_snapshot_handle(txn_id).is_some() {
            if matches!(
                &cmd,
                Command::Insert(_) | Command::Update(_) | Command::Delete(_)
            ) {
                return self.execute_dml_in_transaction(txn_id, text);
            }
            if !matches!(
                &cmd,
                Command::Begin | Command::Commit { .. } | Command::Rollback { .. }
            ) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "command is not supported inside an active transaction; it was not executed"
                        .to_string(),
                )));
            }
        }
        // RETIREMENT A4e (audit B1-DDL/B2): any NON-DML command rehydrates every elided table
        // FIRST (under the commit lock via the serialized helper) — DDL preflights/validators
        // (ADD UNIQUE/PK/FK, CREATE INDEX) read the host store via visible_relational_rows, and a
        // stale prefix would validate a constraint over data that violates it. DDL is rare and
        // the elided set is tiny; the blunt sweep is the safe shape.
        let representation_neutral = matches!(
            &cmd,
            Command::Insert(_)
                | Command::Update(_)
                | Command::Delete(_)
                | Command::Select(_)
                | Command::Begin
                | Command::Commit { .. }
                | Command::Rollback { .. }
        );
        if self.host_install_elision_enabled() && !representation_neutral {
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
        // P4-2b (S-E.P4, design review H2): the SAME sweep for CHUNK-AUTHORITATIVE tables — a
        // DDL preflight (ADD PK/UNIQUE/CHECK/FK) reading visible_relational_rows against a
        // FROZEN store would validate vacuously; de-authoritize every class table first.
        if !representation_neutral {
            let class_tables: Vec<String> = self
                .read_state
                .residency
                .chunk_authoritative_tables
                .load()
                .keys()
                .cloned()
                .collect();
            for table_name in class_tables {
                self.deauthoritize_chunk_table(&table_name, false)
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
                self.begin_transaction_context(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                let snapshot = self
                    .transaction_snapshot_handle(txn_id)
                    .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
                let _statement = snapshot
                    .statement_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                self.ensure_commit_path_available()
                    .map_err(ExecuteError::Engine)?;
                if snapshot.transaction_delta_is_empty() {
                    self.finish_transaction_context(txn_id, true, chain)?;
                } else {
                    self.commit_transaction_delta(txn_id, chain, timestamp_micros)?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                let snapshot = self
                    .transaction_snapshot_handle(txn_id)
                    .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
                let _statement = snapshot
                    .statement_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                self.ensure_commit_path_available()
                    .map_err(ExecuteError::Engine)?;
                self.finish_transaction_context(txn_id, false, chain)?;
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
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
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
            table_rows: self.read_table_rows_at(table, s),
        }
    }
}

#[cfg(test)]
mod device_verdict_cardinality_tests {
    use super::exact_device_verdict_cardinality;

    #[test]
    fn exact_cardinality_rejects_short_and_long_device_components() {
        assert!(exact_device_verdict_cardinality(3, &[3, 3, 3]));
        assert!(!exact_device_verdict_cardinality(3, &[2, 3, 3]));
        assert!(!exact_device_verdict_cardinality(3, &[3, 4, 3]));
    }
}
