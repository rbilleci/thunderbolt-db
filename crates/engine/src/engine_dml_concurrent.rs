//! Concurrent DML path + top-level text execution (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for the snapshot-isolation
//! autocommit write path (is_concurrent_dml, register/deregister_active_snapshot,
//! execute_dml_concurrent[_instrumented], prepare_dml, dml_mutated_tables,
//! commit_dml_concurrent), the execute_text / execute_text_at_timestamp_micros /
//! execute_read_text entry points, and the read-pin helper.

use super::*;

/// Upper bound on items per wave drain (keeps a single wave's worst-case commit latency bounded;
/// under saturation the NEXT wave picks the rest up immediately).
const COMMIT_WAVE_MAX: usize = 1024;

/// Commit-wave telemetry (waves sequenced / items committed through waves / total sequencing
/// nanos including each wave's group-durability tail). Three relaxed adds PER WAVE — not per
/// item — so it stays on permanently; the phase-D SLO benchmark reads it to report wave
/// amortization (`items/waves`) alongside TPS.
pub static WAVE_STATS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// One enqueued concurrent commit: everything the sequencer needs to conflict-check, re-resolve,
/// append, apply, and publish it — plus the shared slot its owner blocks on.
pub(crate) struct CommitWaveItem {
    txn_id: u64,
    cmd: Command,
    payload: Vec<u8>,
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
    outcome: CommitWaveOutcome,
}

pub(crate) type CommitWaveOutcome = Arc<CommitWaveDone>;

/// A wave item's completion slot: the payload behind a mutex, the `done` flag an ATOMIC so
/// waiters can SPIN on completion (a few µs) instead of paying a futex sleep+wake round-trip
/// per commit — the wakeup latency, not the mutex, dominated the first wave measurement.
#[derive(Default)]
pub(crate) struct CommitWaveDone {
    done: std::sync::atomic::AtomicBool,
    result: Mutex<Option<Result<(), ExecuteError>>>,
}

impl CommitWaveDone {
    fn take_if_done(&self) -> Option<Result<(), ExecuteError>> {
        if !self.done.load(AtomicOrdering::Acquire) {
            return None;
        }
        self.result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

impl CommitWaveItem {
    fn set_outcome(&self, result: Result<(), ExecuteError>) {
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
/// signal for waiters and the promotion signal for the next sequencer.
pub(crate) struct CommitWaveState {
    pub(crate) queue: Mutex<CommitWaveQueue>,
    pub(crate) cv: std::sync::Condvar,
}

impl Default for CommitWaveState {
    fn default() -> Self {
        Self {
            queue: Mutex::new(CommitWaveQueue::default()),
            cv: std::sync::Condvar::new(),
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

        // (3) Commit: enqueue into the deterministic commit WAVE (ledger #6) and wait for the
        // sequencer to durably commit + publish it (or abort it with a retryable conflict).
        self.commit_dml_concurrent(
            txn_id,
            cmd,
            text,
            prepared.write_set,
            read_snapshot,
            residency_tables,
            prepared_catalog_seq,
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
    fn commit_dml_concurrent(
        &self,
        txn_id: u64,
        cmd: Command,
        text: &str,
        write_set: WriteSet,
        read_snapshot: Index,
        residency_tables: BTreeSet<String>,
        prepared_catalog_seq: Index,
    ) -> Result<(), ExecuteError> {
        let outcome: CommitWaveOutcome = Arc::new(CommitWaveDone::default());
        let mut queue = self.lock_commit_wave_queue();
        if let Some(reason) = &queue.wedged {
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "the concurrent commit path is wedged pending restart recovery: {reason}"
            ))));
        }
        queue.items.push_back(CommitWaveItem {
            txn_id,
            cmd,
            payload: text.as_bytes().to_vec(),
            prepared_catalog_seq,
            write_set,
            read_snapshot,
            residency_tables,
            outcome: Arc::clone(&outcome),
        });
        if !queue.sequencer_active {
            queue.sequencer_active = true;
            drop(queue);
            self.run_commit_wave_sequencer(&outcome);
            if let Some(result) = outcome.take_if_done() {
                return result;
            }
        } else {
            drop(queue);
        }
        loop {
            // Spin first: under load a wave completes within tens of µs, far cheaper to poll than
            // to pay a futex sleep + wake per commit. The periodic promotion probe keeps queued
            // items from ever being leaderless (the previous sequencer may have stepped down
            // between our enqueue and our first probe).
            for spin in 0..4096_u32 {
                if let Some(result) = outcome.take_if_done() {
                    return result;
                }
                if spin % 64 == 63 {
                    if let Ok(mut queue) = self.commit_wave.queue.try_lock() {
                        if !queue.sequencer_active && !queue.items.is_empty() {
                            queue.sequencer_active = true;
                            drop(queue);
                            self.run_commit_wave_sequencer(&outcome);
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
                if !queue.sequencer_active {
                    if queue.items.is_empty() {
                        break;
                    }
                    queue.sequencer_active = true;
                    drop(queue);
                    self.run_commit_wave_sequencer(&outcome);
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
    fn run_commit_wave_sequencer(&self, own_outcome: &CommitWaveOutcome) {
        loop {
            let batch: Vec<CommitWaveItem> = {
                let mut queue = self.lock_commit_wave_queue();
                let n = queue.items.len().min(COMMIT_WAVE_MAX);
                if n == 0 {
                    queue.sequencer_active = false;
                    drop(queue);
                    self.commit_wave.cv.notify_all();
                    return;
                }
                queue.items.drain(..n).collect()
            };
            let wave_started = std::time::Instant::now();
            let wave_len = batch.len() as u64;
            self.sequence_commit_wave(batch);
            WAVE_STATS[0].fetch_add(1, AtomicOrdering::Relaxed);
            WAVE_STATS[1].fetch_add(wave_len, AtomicOrdering::Relaxed);
            WAVE_STATS[2].fetch_add(
                wave_started.elapsed().as_nanos() as u64,
                AtomicOrdering::Relaxed,
            );
            {
                let _queue = self.lock_commit_wave_queue();
                self.commit_wave.cv.notify_all();
            }
            if own_outcome.done.load(AtomicOrdering::Acquire) {
                // Step down so this client thread can return; a waiter with a still-queued item
                // (or the next arrival) promotes itself.
                let mut queue = self.lock_commit_wave_queue();
                queue.sequencer_active = false;
                drop(queue);
                self.commit_wave.cv.notify_all();
                return;
            }
        }
    }

    /// Commit one WAVE: the per-item (3a)-(3e) steps of the old per-commit critical section, run
    /// back-to-back under ONE commit_mutex hold in wave order, then one group-durability wait +
    /// one `committed_seq` publish for the whole wave. Every item's outcome slot is set exactly
    /// once; the `CommitWaveBatchGuard` fails any still-unset outcome (and wedges the queue) if
    /// this thread panics mid-wave (e.g. the apply-invariant panic, which also poisons the
    /// commit_mutex — the established wedge-don't-serve-torn-state policy).
    fn sequence_commit_wave(&self, batch: Vec<CommitWaveItem>) {
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
        // Groups: (table_name, filter_idx) -> [(item_position, needle, index_name)].
        let mut groups: std::collections::BTreeMap<(String, usize), Vec<(usize, i32, String)>> =
            std::collections::BTreeMap::new();
        // Full-validate fallback for drifted / non-batchable inserts.
        let mut full_validate: Vec<usize> = Vec::new();
        for (pos, item) in batch.iter().enumerate() {
            let Command::Insert(insert) = &item.cmd else {
                continue;
            };
            if insert.rows.len() != 1 {
                continue; // multi-row was validated off-lock (in-statement dup)
            }
            let Some(table) = catalog.relational_catalog.get(&insert.table) else {
                continue; // a dropped table -> its own apply errors
            };
            let gen_matches = catalog.commit_seq == item.prepared_catalog_seq;
            if gen_matches && self.insert_unique_wave_batchable(&catalog, table) {
                // Re-derived eligible (== off-lock eligible) -> it was deferred. Extract each
                // unique index's needle from the single insert row.
                for index in table.indexes.iter().filter(|index| index.unique) {
                    let Some((filter_idx, needle)) =
                        insert_i32_unique_needle(insert, table, &index.column)
                    else {
                        // Can't bind the needle (odd shape) -> full-validate to be safe.
                        full_validate.push(pos);
                        break;
                    };
                    groups
                        .entry((table.name.clone(), filter_idx))
                        .or_default()
                        .push((pos, needle, index.name.clone()));
                }
            } else if !gen_matches {
                // Catalog drift: the item might have deferred off-lock -> full-validate now.
                full_validate.push(pos);
            }
            // gen matches + not eligible -> off-lock validated it, nothing to do.
        }
        // Batched device locate per group; count==0 passes, count>0 authoritative-checks.
        for ((table_name, filter_idx), members) in &groups {
            let Some(table) = catalog.relational_catalog.get(table_name) else {
                continue;
            };
            let needles: Vec<i32> = members.iter().map(|(_, n, _)| *n).collect();
            match self.wave_batch_locate_hit_counts(table, *filter_idx, &needles) {
                Some(counts) => {
                    for ((pos, needle, index_name), &count) in members.iter().zip(counts.iter()) {
                        if count == 0 {
                            continue; // no physical slot holds the key -> no dup
                        }
                        // >0 hits: authoritative visibility+value check at the item's snapshot.
                        let visibility = crate::StorageVisibility {
                            read_txn_id: batch[*pos].read_snapshot,
                        };
                        let table_rows = self.read_state.mvcc.table_rows(table_name);
                        let dup = self
                            .visible_row_with_value(
                                table,
                                visibility,
                                *filter_idx,
                                &SqlValue::Int4(*needle),
                                None,
                            )
                            .unwrap_or(false);
                        let _ = table_rows;
                        if dup {
                            violations.entry(*pos).or_insert_with(|| {
                                format!(
                                    "duplicate key value violates unique index \"{index_name}\""
                                )
                            });
                        }
                    }
                }
                None => {
                    // Locate declined -> full-validate each member (safe fallback).
                    full_validate.extend(members.iter().map(|(pos, _, _)| *pos));
                }
            }
        }
        // Full validation for drifted / declined / unbindable inserts.
        for pos in full_validate {
            if violations.contains_key(&pos) {
                continue;
            }
            let Command::Insert(insert) = &batch[pos].cmd else {
                continue;
            };
            let Some(table) = catalog.relational_catalog.get(&insert.table) else {
                continue;
            };
            let snapshot = self.dml_read_snapshot(batch[pos].read_snapshot);
            // A pure re-validation via prepare_insert (Full) — any Err is the constraint verdict.
            if let Err(err) = self.prepare_insert(insert, snapshot, None, InsertPrepareValidation::Full)
            {
                violations.insert(pos, err.to_string());
            }
        }
        violations
    }

    fn sequence_commit_wave_inner(&self, batch: Vec<CommitWaveItem>) {
        let guard = CommitWaveBatchGuard {
            engine: self,
            items: &batch,
        };
        let wall_clock = current_timestamp_micros();
        let mut wave_tail: Option<(Index, usize)> = None;
        let mut committed: Vec<(usize, Index, bool)> = Vec::with_capacity(batch.len());
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
             committed: &mut Vec<(usize, Index, bool)>| {
                if fast_run.is_empty() {
                    return;
                }
                let _ = commit; // the commit_mutex guard is held by the caller for the whole wave
                let mut by_table: BTreeMap<String, Vec<(WriteDelta, Index)>> = BTreeMap::new();
                let mut run_meta: Vec<(usize, Index, String)> = Vec::with_capacity(fast_run.len());
                for (position, seq, table, delta) in fast_run.drain(..) {
                    run_meta.push((position, seq, table.clone()));
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
                for (position, seq, _table) in run_meta {
                    commit.repl.mark_applied(seq);
                    self.invalidate_relational_residency_tables_concurrent(
                        &batch[position].residency_tables,
                        batch[position].txn_id,
                        seq,
                    );
                    committed.push((position, seq, false));
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
        let mut pending_appends: BTreeMap<
            String,
            (
                Vec<Vec<SqlValue>>,
                Vec<u64>,
                Vec<(usize, Index)>,
                Vec<Index>,
            ),
        > = BTreeMap::new();
        let flush_appends =
            |pending: &mut BTreeMap<
                String,
                (
                    Vec<Vec<SqlValue>>,
                    Vec<u64>,
                    Vec<(usize, Index)>,
                    Vec<Index>,
                ),
            >,
             committed: &mut Vec<(usize, Index, bool)>| {
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
                        if self.host_install_elision_enabled() && !self.table_install_elided(&table)
                        {
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
                            let first_seq = items.first().map(|(_, seq)| *seq).unwrap_or_default();
                            let last_seq = items.last().map(|(_, seq)| *seq).unwrap_or_default();
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
                        for (position, seq) in &items {
                            self.invalidate_relational_residency_tables_concurrent(
                                &batch[*position].residency_tables,
                                batch[*position].txn_id,
                                *seq,
                            );
                        }
                    }
                    for (position, seq) in items {
                        committed.push((position, seq, appended));
                    }
                }
            };
        // M1 design B: WAVE-TIME BATCHED PK-UNIQUE VALIDATION. Eligible INSERTs deferred their
        // unique check off-lock (`prepare_insert`); validate the whole wave here with ONE device
        // locate per (table, key-column) (the amortization win). Returns the item positions that
        // are unique violations -> aborted in the loop below with the byte-identical 23505.
        let wave_unique_violations = self.wave_batch_validate_unique(&batch);
        for (position, item) in batch.iter().enumerate() {
            // M1 design B: a deferred INSERT whose PK value already exists (wave-batch verdict)
            // aborts here — the same 23505 the off-lock validation would have raised.
            if let Some(err) = wave_unique_violations.get(&position) {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::ApplyFailed(err.clone()))));
                continue;
            }
            // Batched-append ORDER: a non-INSERT item's re-resolve (device locate) and its
            // tombstone paths must observe every prior row of this wave — flush first.
            if !matches!(item.cmd, Command::Insert(_)) {
                flush_appends(&mut pending_appends, &mut committed);
            }
            // (3a) SI first-committer-wins: any key in the write-set committed after this item's
            // read snapshot aborts it (retryable). Earlier items in THIS wave recorded into the
            // ledger below, so intra-wave conflicts are caught here exactly like cross-wave ones.
            if commit.ledger.conflicts(&item.write_set, item.read_snapshot) {
                item.set_outcome(Err(ExecuteError::Serialization(format!(
                    "write-write conflict on a key committed after read snapshot {}",
                    item.read_snapshot
                ))));
                continue;
            }

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
            let insert_validation =
                if self.catalog_snapshot().commit_seq == item.prepared_catalog_seq {
                    InsertPrepareValidation::ReResolveLedgerCovered
                } else {
                    InsertPrepareValidation::Full
                };
            let prepared = self.prepare_dml(&item.cmd, install_snapshot, insert_validation);
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
                            let first_seq =
                                items_meta.first().map(|(_, seq)| *seq).unwrap_or_default();
                            let last_seq =
                                items_meta.last().map(|(_, seq)| *seq).unwrap_or_default();
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

            // (3c) Assign the seq for real: WAL append + propose (the sequencer is the single
            // proposer under the commit_mutex). The fsync is deferred to the wave tail.
            let wal_len_before = commit.wal.len();
            commit.wal.append(WalRecord {
                txn_id: item.txn_id,
                payload: item.payload.clone(),
            });
            let wal_position = commit.wal.len();
            let token = match commit.repl.propose(item.payload.clone()) {
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

            // (3e) Record the write-set for future conflict detection (also read by LATER items
            // in this same wave — the intra-wave conflict path above).
            commit.ledger.record(&item.write_set, commit_seq);

            // Fast-run eligibility: a plain INSERT into a table with no unique index, no CHECK,
            // and no FK (its re-resolve read no rows and claimed no unique slots), with auto-admit
            // off (the in-place resident append is a per-item protocol). Everything else is a
            // SLOW item: flush the pending run first so this item's apply-order matches seq order
            // and later re-resolves see it.
            let fast_table = matches!(item.cmd, Command::Insert(_))
                && !auto_admit
                && delta.write_set.unique_slots.is_empty()
                && match &delta.mutation {
                    crate::write_path::PreparedMutation::Insert { table, .. } => {
                        *fast_table_cache.entry(table.clone()).or_insert_with(|| {
                            self.catalog_snapshot()
                                .relational_catalog
                                .get(table)
                                .is_some_and(|t| {
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
            self.apply_delta(delta, commit_seq, None)
                .unwrap_or_else(|err| {
                    panic!(
                        "commit-path invariant violation: apply at commit_seq {commit_seq} failed \
                     after re-validation at this seq succeeded: {err}"
                    )
                });
            commit.repl.mark_applied(commit_seq);
            next_row_id = self.read_state.mvcc.current_row_id();

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
                    entry.2.push((position, commit_seq));
                }
                _ => {
                    self.invalidate_relational_residency_tables_concurrent(
                        &item.residency_tables,
                        item.txn_id,
                        commit_seq,
                    );
                    committed.push((position, commit_seq, false));
                }
            }
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
            return;
        };

        // Wave tail: ONE group-durability wait covering every record this wave appended, then ONE
        // publish at the wave's last seq (CAS max). WAL-before-visibility holds wave-wide: nothing
        // in this wave is visible until its highest record is fsync-covered.
        if let Err(err) = self.wait_group_durable(last_position) {
            // The wave's deltas are applied-but-unpublishable; the flush protocol has recorded its
            // sticky failure (and the flusher wedge-panicked if we were the flusher — in that case
            // this line is unreachable). Fail the wave's outcomes and wedge the queue.
            let mut queue = self.lock_commit_wave_queue();
            queue.wedged = Some(err.to_string());
            drop(queue);
            drop(guard); // fails every still-unset outcome with the wedge error
            return;
        }
        self.publish_committed_seq(last_seq);

        for (position, _seq, appended) in &committed {
            let item = &batch[*position];
            if self.auto_admit_on_commit_enabled() && !appended {
                self.auto_admit_resident_tables(&item.residency_tables);
            }
            self.metrics.inc_commit();
            item.set_outcome(Ok(()));
        }
        std::mem::forget(guard);
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
    fn wait_group_durable(&self, wal_position: usize) -> Result<(), EngineError> {
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
                            text.as_bytes().to_vec(),
                            timestamp_micros,
                        )?;
                    }
                    RouteDecision::CpuFallback { reason, .. } => {
                        self.metrics.inc_gpu_fallback(reason);
                        self.commit_mutation_at(
                            txn_id,
                            text.as_bytes().to_vec(),
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

/// M1 design B: bind an INSERT's value for a UNIQUE-index column to its i32-section needle +
/// the column's catalog position. Handles the insert's own column list (explicit or catalog
/// order) and coerces the raw literal (Text->Date, Int4->Int2) as the DML binder does, then
/// encodes via `i32_section_needle` (strict variant agreement). `None` (odd shape / non-i32
/// section / NULL / uncoercible) -> the caller full-validates that item instead of batching.
/// SINGLE-ROW only (the caller gates on `insert.rows.len() == 1`).
fn insert_i32_unique_needle(
    insert: &Insert,
    table: &RelationalTable,
    unique_column: &str,
) -> Option<(usize, i32)> {
    let filter_idx = table.columns.iter().position(|c| c.name == *unique_column)?;
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
