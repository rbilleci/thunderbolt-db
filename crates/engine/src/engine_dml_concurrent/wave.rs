use super::{
    coerce_filter_literal, current_timestamp_micros, relational_key_prefix,
    try_encode_binary_insert, wave_device_phase_timing_enabled, wave_host_phase_timing_enabled,
    Command, CommitState, CommitWaveItem, CommitWaveTail, DmlReadSnapshot, Engine, EngineError,
    ExecuteError, Index, Insert, InsertPrepareValidation, LogReplicator, RelationalIndex,
    RelationalTable, SqlValue, WalRecord, WriteDelta, WAVE_DEVICE_STATS, WAVE_HOST_STATS,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::{Duration, Instant};

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

/// One wave winner awaiting the ordered commit cut: batch position, table, applied values,
/// pre-encoded WAL record, and the row-id patch offset within that record.
type CommitWaveWinner = (usize, String, Vec<SqlValue>, Vec<u8>, usize);

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
    pub(super) fn sequence_commit_wave(
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
            // COMPOUND KEYS: the ORDINALS (into `table.indexes`) of the unique indexes to validate —
            // each keyed on i32-section column(s). Empty = not eligible (full-validate fallback).
            unique_indexes: Vec<usize>,
        }
        let mut tables: Vec<TableCtx> = Vec::new();
        // group key = (distinct-table index, probe key_id, index ordinal) -> (needles, positions).
        let mut group_keys: Vec<(usize, usize, usize)> = Vec::new();
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
                    let unique_indexes: Vec<usize> = if eligible {
                        table
                            .indexes
                            .iter()
                            .enumerate()
                            .filter(|(_, index)| index.unique)
                            .map(|(ord, _)| ord)
                            .collect()
                    } else {
                        Vec::new()
                    };
                    tables.push(TableCtx {
                        name: &insert.table,
                        table,
                        unique_indexes,
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
            if tables[tctx_idx].unique_indexes.is_empty() {
                continue; // not eligible -> off-lock validated it
            }
            // Eligible + gen-matched -> it was deferred. Bind each unique index's probe needle
            // (single-column raw key, or the compound tuple fingerprint).
            let table = tables[tctx_idx].table;
            let mut bound_all = true;
            let ords = tables[tctx_idx].unique_indexes.clone();
            for ord in &ords {
                let index = &table.indexes[*ord];
                let Some((key_id, needle)) = insert_index_probe_needle(insert, table, index, *ord)
                else {
                    bound_all = false;
                    break;
                };
                // Find/create the (tctx_idx, key_id, ord) group.
                let gk = (tctx_idx, key_id, *ord);
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
        for (gi, &(tctx_idx, key_id, ord)) in group_keys.iter().enumerate() {
            let table = tables[tctx_idx].table;
            let index = &table.indexes[ord];
            let compound = crate::engine_residency::index_is_compound(index);
            let locate_started = wave_device_phase_timing_enabled().then(Instant::now);
            let locate = self.wave_batch_locate_hit_counts(table, key_id, &group_needles[gi]);
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
                            continue; // the common case: no physical slot holds the key/fingerprint
                        }
                        // Rare: >0 hits -> authoritative visibility+value check. For a COMPOUND key
                        // the recheck compares the FULL key tuple (a fingerprint hit alone is not a
                        // duplicate), restoring exactness to the surrogate probe.
                        let visibility = crate::StorageVisibility {
                            read_txn_id: batch[pos].read_snapshot,
                        };
                        let is_dup = if compound {
                            let Command::Insert(insert) = &batch[pos].cmd else {
                                continue;
                            };
                            match insert_index_key_tuple(insert, table, index) {
                                Some(tuple) => self
                                    .visible_row_with_tuple(
                                        table,
                                        visibility,
                                        key_id,
                                        Some(needle),
                                        &tuple,
                                        None,
                                    )
                                    .unwrap_or(false),
                                None => {
                                    // Unbindable tuple (e.g. NULL key column) -> full host validate.
                                    full_validate.push(pos);
                                    continue;
                                }
                            }
                        } else {
                            self.visible_row_with_value(
                                table,
                                visibility,
                                key_id,
                                &SqlValue::Int4(needle),
                                None,
                            )
                            .unwrap_or(false)
                        };
                        if is_dup {
                            let index_name = index.name.as_str();
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
            if !catalog.relational_catalog.contains_key(&insert.table) {
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
                    entry.3.extend(std::iter::repeat_n(commit_seq, rows.len()));
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
        let mut winners: Vec<CommitWaveWinner> = Vec::with_capacity(n);
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

/// COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): the coerced value of `index`'s key column `name` in this
/// insert's single row — the shared per-column bind used by both the probe needle (folded to words) and
/// the recheck tuple. `None` if the column is missing from the row. Returns `(catalog_column_idx,
/// coerced_value)`.
fn insert_key_column_bind(
    insert: &Insert,
    table: &RelationalTable,
    name: &str,
) -> Option<(usize, SqlValue)> {
    let filter_idx = table.columns.iter().position(|c| c.name == *name)?;
    let column_ty = table.columns[filter_idx].ty;
    let row = insert.rows.first()?;
    let source_pos = if insert.columns.is_empty() {
        filter_idx
    } else {
        insert.columns.iter().position(|c| c == name)?
    };
    // Coerce the raw literal to the column TYPE the way the INSERT apply does (Text -> Uuid via
    // parse_uuid; Numeric rescaled to the column's scale) so the folded WORDS match the stored b128
    // section bytes exactly — `coerce_filter_literal` leaves a uuid/numeric literal as Text/unscaled.
    let coerced =
        crate::rel_exec_helpers::coerce_insert_value(row.get(source_pos)?.clone(), column_ty, name)
            .ok()?;
    Some((filter_idx, coerced))
}

/// COMPOUND KEYS: bind the DEVICE-PROBE `(key_id, needle)` for `index` against this insert's row.
/// Single-column -> `(col_idx, raw i32)`; compound -> `(FLAG | ord, fingerprint)` folded over every key
/// column's i32 WORDS (`sql_value_key_words` — i64 keys contribute 2 words). `None` if any key column
/// can't bind / is an unsupported key value (the caller falls to full host validation).
fn insert_index_probe_needle(
    insert: &Insert,
    table: &RelationalTable,
    index: &RelationalIndex,
    ord: usize,
) -> Option<(usize, i32)> {
    if crate::engine_residency::index_is_compound(index) {
        let mut words: Vec<i32> = Vec::with_capacity(index.key_columns.len());
        for name in &index.key_columns {
            let (col_idx, value) = insert_key_column_bind(insert, table, name)?;
            words.extend(crate::engine_residency::sql_value_key_words(
                table.columns[col_idx].ty,
                &value,
            )?);
        }
        let key_id = crate::engine_residency::index_probe_key_id(table, index, ord)?;
        Some((
            key_id,
            crate::engine_residency::compound_key_fingerprint(&words),
        ))
    } else {
        insert_i32_unique_needle_at(insert, table, ord_first_column_idx(table, index)?)
    }
}

/// COMPOUND KEYS: the key TUPLE `(catalog_column_idx, coerced_value)` for `index`'s recheck. `None`
/// if any key column can't bind (parallel to `insert_index_probe_needle`).
fn insert_index_key_tuple(
    insert: &Insert,
    table: &RelationalTable,
    index: &RelationalIndex,
) -> Option<Vec<(usize, SqlValue)>> {
    index
        .key_columns
        .iter()
        .map(|name| insert_key_column_bind(insert, table, name))
        .collect()
}

/// The catalog index of a single-column index's key column (`index.column`).
fn ord_first_column_idx(table: &RelationalTable, index: &RelationalIndex) -> Option<usize> {
    table.columns.iter().position(|c| c.name == index.column)
}
