use super::{
    coerce_filter_literal, current_timestamp_micros, exact_device_verdict_cardinality,
    relational_key_prefix, try_encode_binary_insert, wave_device_phase_timing_enabled,
    wave_host_phase_timing_enabled, CatalogSnapshot, Command, CommitWaveItem, CommitWaveTail,
    DmlReadSnapshot, Engine, EngineError, ExecuteError, Index, Insert, InsertPrepareValidation,
    LogReplicator, RelationalIndex, RelationalTable, SqlValue, WAVE_DEVICE_STATS, WAVE_HOST_STATS,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::{Duration, Instant};

#[cfg(test)]
type ShardedPostValidationHook = (
    usize,
    std::sync::Arc<std::sync::Barrier>,
    std::sync::Arc<std::sync::Barrier>,
);

#[cfg(test)]
type SerialPreCommitLockHook = (
    usize,
    std::sync::Arc<std::sync::Barrier>,
    std::sync::Arc<std::sync::Barrier>,
);

#[cfg(test)]
fn serial_pre_commit_lock_hook() -> &'static std::sync::Mutex<Option<SerialPreCommitLockHook>> {
    static HOOK: std::sync::OnceLock<std::sync::Mutex<Option<SerialPreCommitLockHook>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(test)]
fn sharded_post_validation_hook() -> &'static std::sync::Mutex<Option<ShardedPostValidationHook>> {
    static HOOK: std::sync::OnceLock<std::sync::Mutex<Option<ShardedPostValidationHook>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

/// E2.4a — VARIANT 1 (shared-WAL sharded sequencing): the number of PARALLEL shard workers the
/// sequencer fans a homogeneous covered-INSERT-intent wave out to. `1` (default) = the E2.3 serial
/// sequencer. `N>1` moves per-item conflict arbitration (device history plus per-shard private
/// same-wave sets; same-PK → same shard → single-winner 23505 preserved),
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
/// the wave-assigned id). Built by parallel workers while the coordinator retains the serialized
/// publication boundary; the workers do not touch the guarded commit state.
enum ShardVerdict {
    Commit {
        table: String,
        values: Vec<SqlValue>,
        wal_record: Vec<u8>,
        wal_offset: usize,
    },
    Abort(ExecuteError),
}

/// Exact outcome of the wave-batched device uniqueness pass. `violations` are authoritative
/// duplicate-key verdicts. `declined` contains positions for which the device could not return a
/// complete verdict; those positions must abort retryably before WAL rather than being interpreted
/// as misses or delegated to a production host relational source.
#[derive(Default)]
struct WaveUniqueVerdicts {
    violations: BTreeMap<usize, String>,
    declined: BTreeSet<usize>,
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
                item.set_outcome(Err(ExecuteError::Indeterminate(format!(
                    "the concurrent commit path is wedged pending restart recovery: {reason}"
                ))));
            }
        }
        self.engine.commit_wave.cv.notify_all();
        self.engine.wedge_commit_path();
    }
}

impl Engine {
    #[cfg(test)]
    pub(crate) fn set_serial_pre_commit_lock_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *serial_pre_commit_lock_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    #[cfg(test)]
    pub(crate) fn set_sharded_post_validation_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *sharded_post_validation_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    /// Commit one WAVE: the per-item (3a)-(3e) steps of the old per-commit critical section, run
    /// back-to-back under ONE commit_mutex hold in wave order. W2: the durability tail (group
    /// fsync wait + `committed_seq` publish + acks) is RETURNED as a [`CommitWaveTail`] instead of
    /// running inline, so the caller can
    /// pipeline it against the next wave's sequencing. `None` = every item aborted pre-durable
    /// (outcomes already set). Every item's outcome slot is set exactly once; the
    /// `CommitWaveBatchGuard` fails any still-unset outcome (and wedges the queue) if this
    /// thread panics mid-wave (e.g. the apply-invariant panic, which also poisons the
    /// commit_mutex — the established wedge-don't-serve-torn-state policy); once the tail is
    /// built, its `armed` Drop carries that responsibility.
    pub(super) fn sequence_commit_wave(
        &self,
        batch: Vec<CommitWaveItem>,
    ) -> Option<CommitWaveTail> {
        // The entire wave (conflict-check, device re-resolve, flush, apply, publish) runs inside
        // one commit critical section. The internal-read marker preserves the existing lock
        // discipline for catalog/materialized-view work without enabling a DML repair fallback.
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
    /// dups are caught by wave-local slot arbitration, NOT here; this catches
    /// ALREADY-COMMITTED dups. A device locate/recheck decline aborts retryably; an unbindable item
    /// takes the typed per-item full validator.
    fn wave_batch_validate_unique(&self, batch: &[CommitWaveItem]) -> WaveUniqueVerdicts {
        let mut verdicts = WaveUniqueVerdicts::default();
        if !self.device_write_locate_wave_batch_enabled() {
            return verdicts;
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
            let fingerprint_backed = crate::engine_residency::index_uses_fingerprint(table, index);
            let locate_started = wave_device_phase_timing_enabled().then(Instant::now);
            let locate = self.wave_batch_locate_hit_counts(table, key_id, &group_needles[gi]);
            if let Some(started) = locate_started {
                WAVE_DEVICE_STATS[0]
                    .fetch_add(started.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
            }
            match locate {
                Some(counts) => {
                    let expected = group_positions[gi].len();
                    if !exact_device_verdict_cardinality(
                        expected,
                        &[group_needles[gi].len(), counts.len()],
                    ) {
                        verdicts
                            .declined
                            .extend(group_positions[gi].iter().copied());
                        continue;
                    }
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
                        let is_dup = if fingerprint_backed {
                            let Command::Insert(insert) = &batch[pos].cmd else {
                                continue;
                            };
                            match insert_index_key_tuple(insert, table, index) {
                                Some(tuple) => match self.visible_row_with_tuple(
                                    table,
                                    visibility,
                                    key_id,
                                    Some(needle),
                                    &tuple,
                                    None,
                                ) {
                                    Ok(is_dup) => is_dup,
                                    Err(_) => {
                                        verdicts.declined.insert(pos);
                                        continue;
                                    }
                                },
                                None => {
                                    // An unbindable tuple (for example a NULL key column) takes the
                                    // full typed validator. On a device-authoritative relation that
                                    // validator remains device-native; an actual device decline is
                                    // represented separately above and never becomes `false`.
                                    full_validate.push(pos);
                                    continue;
                                }
                            }
                        } else {
                            match self.visible_row_with_value(
                                table,
                                visibility,
                                key_id,
                                &SqlValue::Int4(needle),
                                None,
                            ) {
                                Ok(is_dup) => is_dup,
                                Err(_) => {
                                    verdicts.declined.insert(pos);
                                    continue;
                                }
                            }
                        };
                        if is_dup {
                            let index_name = index.name.as_str();
                            verdicts.violations.entry(pos).or_insert_with(|| {
                                format!(
                                    "duplicate key value violates unique index \"{index_name}\""
                                )
                            });
                        }
                    }
                }
                None => verdicts
                    .declined
                    .extend(group_positions[gi].iter().copied()),
            }
        }
        // Full validation for drifted / declined / unbindable inserts (rare).
        for pos in full_validate {
            if verdicts.violations.contains_key(&pos) || verdicts.declined.contains(&pos) {
                continue;
            }
            let Command::Insert(insert) = &batch[pos].cmd else {
                continue;
            };
            if !catalog.relational_catalog.contains_key(&insert.table) {
                continue;
            }
            let snapshot = self.dml_read_snapshot(batch[pos].read_snapshot);
            if let Err(err) = self.prepare_insert(
                insert,
                snapshot,
                None,
                InsertPrepareValidation::WaveFallbackFull,
            ) {
                verdicts.violations.insert(pos, err.to_string());
            }
        }
        verdicts
    }

    fn sequence_commit_wave_inner(&self, batch: Vec<CommitWaveItem>) -> Option<CommitWaveTail> {
        // E2.4a VARIANT 1 — shared-WAL sharded sequencing. When the whole wave is homogeneous
        // covered-INSERT intents (the flagship OLTP shape), fan the expensive per-item prep
        // (conflict check/record, value + WAL-record clones) out to N parallel shard workers and
        // keep only the ordered WAL append + global commit-seq claim + device-append buffer under a
        // thin serial cut. A mixed wave (any non-intent / classic item) keeps the fully-serial path
        // below. The sharded conflict verdict combines device history with a per-shard private
        // dedup set; a homogeneous-intent wave guarantees no classic item can interleave between
        // that verdict and the ordered cut.
        let shards = intent_sequencer_shards();
        if shards > 1 && batch.len() >= SHARD_MIN_WAVE {
            let wave_catalog = self.catalog_snapshot();
            if self.device_write_locate_wave_batch_enabled()
                && batch
                    .iter()
                    .all(|item| self.item_sharded_intent_eligible(item, &wave_catalog))
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
        let mut committed: Vec<(usize, Index, u64)> = Vec::with_capacity(batch.len());

        #[cfg(test)]
        {
            let hook = {
                let mut hook = serial_pre_commit_lock_hook()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                hook.as_ref()
                    .is_some_and(|(engine, _, _)| *engine == self as *const Self as usize)
                    .then(|| hook.take())
                    .flatten()
            };
            if let Some((_, reached, resume)) = hook {
                reached.wait();
                resume.wait();
            }
        }

        let mut commit = self.commit_state();
        // The virtual row-id cursor assigns device-native INSERT identities in wave order. Read it
        // only after taking the canonical commit mutex: explicit transactions, classic waves,
        // sharded waves, and optimized lanes all advance the same allocator under this cut.
        let mut next_row_id = self.read_state.mvcc.current_row_id();
        if let Err(error) = self.ensure_commit_path_available() {
            let message = error.to_string();
            for item in &batch {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                    message.clone(),
                ))));
            }
            std::mem::forget(guard);
            return None;
        }
        if let Err(error) = self.legacy_lane_history_write_guard() {
            let message = error.to_string();
            for item in &batch {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                    message.clone(),
                ))));
            }
            std::mem::forget(guard);
            return None;
        }
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
        let flush_appends = |pending: &mut WavePendingAppends,
                             committed: &mut Vec<(usize, Index, u64)>| {
            self.flush_wave_pending_appends(pending, committed);
        };
        // M1 design B: WAVE-TIME BATCHED PK-UNIQUE VALIDATION. Eligible INSERTs deferred their
        // unique check off-lock (`prepare_insert`); validate the whole wave here with ONE device
        // locate per (table, key-column) (the amortization win). Returns the item positions that
        // are unique violations -> aborted in the loop below with the byte-identical 23505.
        let hostphase = wave_host_phase_timing_enabled();
        let wave_validate_started = hostphase.then(Instant::now);
        let wave_unique_verdicts = self.wave_batch_validate_unique(&batch);
        if let Some(started) = wave_validate_started {
            WAVE_HOST_STATS[0]
                .fetch_add(started.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }
        // E2.3 — the catalog generation is CONSTANT for the whole wave: DDL is the only publisher
        // and it commits under the very commit_mutex this sequencer holds, so no generation bump can
        // interleave a wave's items. Load the snapshot ONCE here instead of per item (the old
        // per-item `catalog_snapshot()` was an ArcSwap load + Arc clone on every commit — the
        // generation gate at re-resolve, the fast-run eligibility probe, and the intent fast-lane
        // gate all read it). `wave_catalog_seq` is the schema stamp every item compares against.
        let wave_catalog = self.catalog_snapshot();
        let wave_catalog_seq = wave_catalog.commit_seq;
        // Unique keys of successful earlier members whose device append/tombstone may still be
        // buffered until this wave's flush. This set is wave-bounded; committed history lives in
        // the resident version stamps queried below, not in the CPU commit ledger.
        let mut wave_unique_slots: std::collections::HashSet<
            crate::write_path::UniqueIndexSlotKey,
        > = std::collections::HashSet::new();
        let mut wave_unique_slots_i32: std::collections::HashSet<
            crate::write_path::IntUniqueSlotKey,
        > = std::collections::HashSet::new();
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
            let request_digest = gpu_db_wal::canonical_request_digest(&batch[position].payload);
            match commit
                .resolve_transaction_retry_digest_outcome(batch[position].txn_id, request_digest)
            {
                Ok(Some((token, affected_rows))) if self.committed_seq() >= token.index => {
                    batch[position].set_outcome(Ok(affected_rows));
                    continue;
                }
                Ok(Some(_)) => {
                    batch[position].set_outcome(Err(ExecuteError::Indeterminate(format!(
                        "transaction id {} is committed but not yet publication-covered",
                        batch[position].txn_id
                    ))));
                    continue;
                }
                Err(error) => {
                    batch[position].set_outcome(Err(ExecuteError::Engine(error)));
                    continue;
                }
                Ok(None) => {}
            }
            match self.resolve_pending_transaction_claim(batch[position].txn_id, request_digest) {
                Ok(true) => {
                    batch[position].set_outcome(Err(ExecuteError::Indeterminate(format!(
                        "transaction id {} is pending in canonical mutation admission",
                        batch[position].txn_id
                    ))));
                    continue;
                }
                Err(error) => {
                    batch[position].set_outcome(Err(ExecuteError::Engine(error)));
                    continue;
                }
                Ok(false) => {}
            }
            if let Some(expected) = batch[position].expected_catalog_version {
                if let Err(error) =
                    crate::engine_mutation_admission::validate_prepared_catalog_version(
                        expected,
                        wave_catalog_seq,
                    )
                {
                    batch[position].set_outcome(Err(error));
                    continue;
                }
            }
            if let Some(state) = commit.txn_manager.state(batch[position].txn_id) {
                batch[position].set_outcome(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    format!(
                        "transaction id {} is already owned by transaction state {state:?}",
                        batch[position].txn_id
                    ),
                ))));
                continue;
            }
            // M1 design B: a deferred INSERT whose PK value already exists (wave-batch verdict)
            // aborts here — the same 23505 the off-lock validation would have raised.
            if let Some(err) = wave_unique_verdicts.violations.get(&position) {
                batch[position].set_outcome(Err(ExecuteError::Engine(
                    EngineError::UniqueViolation(err.clone()),
                )));
                continue;
            }
            if wave_unique_verdicts.declined.contains(&position) {
                batch[position].set_outcome(Err(ExecuteError::Serialization(
                    "device uniqueness validation declined before commit (retryable)".to_string(),
                )));
                continue;
            }
            // Batched-append ORDER: a non-INSERT item's re-resolve (device locate) and its
            // tombstone paths must observe every prior row of this wave — flush first.
            if !matches!(batch[position].cmd, Command::Insert(_)) {
                flush_appends(&mut pending_appends, &mut committed);
            }
            // (3a) SI first-committer-wins. Row identity conflicts retain the bounded
            // oldest-active CPU map. Unique conflicts come from exact device version history,
            // plus the wave-local bridge for earlier buffered members not yet published.
            let item = &batch[position];
            let row_conflict = commit
                .ledger
                .conflicts_rows(&item.write_set, item.read_snapshot);
            let has_unique = !item.write_set.unique_slots.is_empty()
                || !item.write_set.unique_slots_i32.is_empty();
            let wave_unique_conflict = item
                .write_set
                .unique_slots
                .iter()
                .any(|slot| wave_unique_slots.contains(slot))
                || item
                    .write_set
                    .unique_slots_i32
                    .iter()
                    .any(|slot| wave_unique_slots_i32.contains(slot));
            let device_unique_conflict =
                if !has_unique || wave_unique_conflict {
                    false
                } else {
                    match item.offlock_delta.as_ref().and_then(|delta| {
                        self.device_unique_write_conflicts(delta, item.read_snapshot)
                    }) {
                        Some(conflict) => conflict,
                        None => {
                            // Host-neutral specification fixtures keep their parity ledger without
                            // claiming execution. Production has no host authority: a missing device
                            // history verdict fails closed. Test builds follow that same law once a
                            // table is device-authoritative, so an actual-GPU acceptance target cannot
                            // pass via the cfg(test) parity map.
                            #[cfg(test)]
                            {
                                let device_authoritative = match &item.cmd {
                                    Command::Insert(insert) => Some(insert.table.as_str()),
                                    Command::Update(update) => Some(update.table.as_str()),
                                    Command::Delete(delete) => Some(delete.table.as_str()),
                                    _ => None,
                                }
                                .is_some_and(|table| {
                                    self.table_device_authoritative(table)
                                        || self.table_chunk_authoritative(table).is_some()
                                });
                                device_authoritative
                                    || commit
                                        .ledger
                                        .conflicts_unique(&item.write_set, item.read_snapshot)
                            }
                            #[cfg(not(test))]
                            {
                                true
                            }
                        }
                    }
                };
            if row_conflict || wave_unique_conflict || device_unique_conflict {
                let read_snapshot = batch[position].read_snapshot;
                batch[position].set_outcome(Err(ExecuteError::Serialization(format!(
                    "write-write conflict on a key committed after read snapshot {read_snapshot}"
                ))));
                continue;
            }
            hp!(1);

            // E2.3 — INTENT INTEGER FAST LANE. A single-row covered-INSERT intent (pre-encoded
            // binary WAL template + reuse-eligible off-lock delta + catalog generation unchanged
            // since prepare + table still device-authoritative) owes NO String row key: its row id
            // is the wave's integer `next_row_id`, its W5a record is patched in place, and its values
            // flow straight into the batched device append. This collapses the general path's
            // `rekey_offlock_insert_delta` (row-key `format!` + write-set/value clones) AND the
            // `insert_append` value-clone + String→u64 parse — the two top host buckets (reresolve,
            // apply) for the flagship shape — into one value clone + one WAL patch. Any drift (gen
            // bump or non-intent item) falls through to the general device path below.
            let intent_fast = wave_catalog_seq == batch[position].prepared_catalog_seq
                && batch[position].binary_wal_template.is_some()
                && matches!(&batch[position].offlock_delta, Some(d)
                    if Self::reresolve_reuse_eligible(d, &wave_catalog)
                        && matches!(&d.mutation,
                            crate::write_path::PreparedMutation::Insert { inserted_rows, .. }
                                if inserted_rows.len() == 1))
                && match &batch[position].cmd {
                    Command::Insert(insert) => self.table_device_authoritative(&insert.table),
                    _ => false,
                };
            if intent_fast {
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
                let token = match commit.repl.propose(wal_payload.clone()) {
                    Ok(token) => token,
                    Err(err) => {
                        batch[position].set_outcome(Err(ExecuteError::Engine(err)));
                        continue;
                    }
                };
                let record = match Self::canonical_wal_record_with_commit_request_digest(
                    &commit,
                    batch[position].txn_id,
                    token.index,
                    0,
                    &wal_payload,
                    gpu_db_wal::canonical_request_digest(&batch[position].payload),
                ) {
                    Ok(record) => record,
                    Err(err) => {
                        commit.repl.rollback_unapplied_from(token.index);
                        batch[position].set_outcome(Err(ExecuteError::Engine(err)));
                        continue;
                    }
                };
                commit.wal.append(record);
                let wal_position = commit.wal.len();
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
                if let Err(error) = commit.record_transaction_status_digest_outcome(
                    batch[position].txn_id,
                    gpu_db_wal::canonical_request_digest(&batch[position].payload),
                    token.index,
                    1,
                ) {
                    commit.repl.rollback_unapplied_from(commit_seq);
                    commit.wal.truncate(wal_len_before);
                    batch[position].set_outcome(Err(ExecuteError::Engine(error)));
                    continue;
                }
                let timestamp_micros =
                    wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
                commit.record_commit_timestamp(batch[position].txn_id, timestamp_micros);
                hp!(3);
                commit.ledger.record(&batch[position].write_set, commit_seq);
                wave_unique_slots.extend(batch[position].write_set.unique_slots.iter().cloned());
                wave_unique_slots_i32
                    .extend(batch[position].write_set.unique_slots_i32.iter().copied());
                hp!(4);
                // Device-authoritative apply advances the durable row-id allocator and authority
                // counter, exactly `apply_delta`'s insert branch for one row. Clone the row
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
                    .device_authoritative_commits
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
            // Ledger #18: FK-free INSERT re-resolves skip the redundant unique/CHECK pass —
            // the conflicts() check above IS the commit-time guard (coverage proof on
            // InsertPrepareValidation) — but ONLY while the catalog generation still matches
            // the off-lock prepare's (audit fix): a constraint-adding DDL committed since S
            // is absent from the prepared key projection and the item's write_set lacks slots for the new
            // index, so the skip would silently bypass it. Any DDL bumps the stamp -> Full
            // (always correct; DDL is rare so the hot path keeps the skip).
            let insert_validation = if wave_catalog_seq == item.prepared_catalog_seq {
                InsertPrepareValidation::ReResolveDeviceCovered
            } else {
                InsertPrepareValidation::Full
            };
            // DELTA-REUSE (B): a reuse-eligible elided insert whose catalog generation still
            // matches (`ReResolveDeviceCovered`) owes no re-validation — RE-KEY the off-lock delta
            // at the wave's `next_row_id` instead of re-coercing + rebuilding it. A generation
            // drift (Full) or a non-eligible item falls through to the authoritative re-prepare.
            let prepared = match &item.offlock_delta {
                Some(delta)
                    if insert_validation == InsertPrepareValidation::ReResolveDeviceCovered
                        && Self::reresolve_reuse_eligible(delta, &wave_catalog) =>
                {
                    Ok(Self::rekey_offlock_insert_delta(delta, install_snapshot))
                }
                _ => self.prepare_dml(&item.cmd, install_snapshot, insert_validation),
            };
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
            let item_rows = delta.rows_affected();
            let returning = match self.project_dml_returning(&item.cmd, &delta, commit_seq) {
                Ok(returning) => returning,
                Err(error) => {
                    item.set_outcome(Err(error));
                    continue;
                }
            };
            item.outcome.set_returning(returning);
            hp!(2);

            // (3c) Assign the seq for real: WAL append + propose (the sequencer is the single
            // proposer under the commit_mutex). The fsync is deferred to the wave tail.
            // W5a: covered inserts (the delta-reuse class — elided, FK/CHECK-free, no sequence
            // defaults, device-history-covered uniqueness) log the RESOLVED BINARY record instead of the
            // SQL text: replay becomes decode+install (no parse, no re-resolve), the record
            // carries the ORIGINAL row ids, and checkpoints shrink. Everything else keeps the
            // SQL-text payload unchanged.
            let wal_payload: std::sync::Arc<[u8]> = if self.binary_wal_records_enabled()
                && Self::reresolve_reuse_eligible(&delta, &wave_catalog)
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
            let token = match commit.repl.propose(wal_payload.clone()) {
                Ok(token) => token,
                Err(err) => {
                    item.set_outcome(Err(ExecuteError::Engine(err)));
                    continue;
                }
            };
            let record = match Self::canonical_wal_record_with_commit_outcome(
                &commit,
                item.txn_id,
                token.index,
                0,
                &wal_payload,
                gpu_db_wal::canonical_request_digest(&item.payload),
                if item_rows == 0 {
                    gpu_db_wal::CanonicalOutcomeKind::CommitNoOp
                } else {
                    gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
                },
                item_rows,
            ) {
                Ok(record) => record,
                Err(err) => {
                    commit.repl.rollback_unapplied_from(token.index);
                    item.set_outcome(Err(ExecuteError::Engine(err)));
                    continue;
                }
            };
            commit.wal.append(record);
            let wal_position = commit.wal.len();
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
            if let Err(error) = commit.record_transaction_status_digest_outcome(
                item.txn_id,
                gpu_db_wal::canonical_request_digest(&item.payload),
                token.index,
                item_rows,
            ) {
                commit.repl.rollback_unapplied_from(commit_seq);
                commit.wal.truncate(wal_len_before);
                item.set_outcome(Err(ExecuteError::Engine(error)));
                continue;
            }
            let timestamp_micros =
                wall_clock.max(commit.max_commit_timestamp_micros.saturating_add(1));
            commit.record_commit_timestamp(item.txn_id, timestamp_micros);
            hp!(3);

            // (3e) Record the write-set for future conflict detection (also read by LATER items
            // in this same wave — the intra-wave conflict path above).
            commit.ledger.record(&item.write_set, commit_seq);
            wave_unique_slots.extend(item.write_set.unique_slots.iter().cloned());
            wave_unique_slots_i32.extend(item.write_set.unique_slots_i32.iter().copied());
            hp!(4);

            // (3d) Install the re-validated delta now (apply-before-durable, D3b). A
            // failure here is a true invariant violation — PANIC, poisoning the commit_mutex; the
            // batch guard fails the wave's remaining outcomes and wedges the queue.
            // RETIREMENT A1: carry the inserted rows' host identities (parsed from their keys) so
            // the residency append can stamp the row-identity region.
            enum DeviceMaintenance {
                Insert(String, Vec<Vec<SqlValue>>, Vec<u64>),
                Delete(String, Vec<Vec<SqlValue>>, Vec<u64>),
                Update(String, Vec<Vec<SqlValue>>, Vec<Vec<SqlValue>>, Vec<u64>),
            }
            let device_maintenance = match &delta.mutation {
                crate::write_path::PreparedMutation::Insert {
                    table,
                    inserted_rows,
                    ..
                } => {
                    let prefix = relational_key_prefix(table);
                    DeviceMaintenance::Insert(
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
                    )
                }
                crate::write_path::PreparedMutation::Delete {
                    table,
                    deleted_rows,
                    ..
                } => {
                    let prefix = relational_key_prefix(table);
                    DeviceMaintenance::Delete(
                        table.clone(),
                        deleted_rows.clone(),
                        delta
                            .write_set
                            .rows
                            .iter()
                            .map(|key| {
                                crate::engine_residency::parse_relational_row_id(
                                    &key.row_key,
                                    &prefix,
                                )
                                .unwrap_or(u64::MAX)
                            })
                            .collect(),
                    )
                }
                crate::write_path::PreparedMutation::Update {
                    table,
                    installs,
                    updated_old_rows,
                    ..
                } => {
                    let prefix = relational_key_prefix(table);
                    DeviceMaintenance::Update(
                        table.clone(),
                        updated_old_rows.clone(),
                        installs
                            .iter()
                            .map(|(_, _, values)| values.clone())
                            .collect(),
                        installs
                            .iter()
                            .map(|(_, key, _)| {
                                crate::engine_residency::parse_relational_row_id(key, &prefix)
                                    .unwrap_or(u64::MAX)
                            })
                            .collect(),
                    )
                }
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
            hp!(5);

            // Residency, before publish: INSERT rows buffer into the wave-batched append; UPDATE
            // and DELETE maintain the same authoritative generation in place. A device decline
            // invalidates and schedules an exact re-admission before the next wave.
            wave_tail = Some((commit_seq, wal_position));
            match device_maintenance {
                DeviceMaintenance::Insert(table, rows, row_ids) => {
                    let entry = pending_appends.entry(table).or_default();
                    // D3: one birth stamp per row of THIS item (the flush spans commit seqs).
                    entry.3.extend(std::iter::repeat_n(commit_seq, rows.len()));
                    entry.0.extend(rows);
                    entry.1.extend(row_ids);
                    entry.2.push((position, commit_seq, item_rows));
                }
                DeviceMaintenance::Delete(table, rows, row_ids) => {
                    let handled =
                        wave_catalog
                            .relational_catalog
                            .get(&table)
                            .is_some_and(|table| {
                                self.try_tombstone_rows_by_identity(
                                    table, &rows, &row_ids, commit_seq,
                                )
                            });
                    if !handled {
                        panic!(
                            "commit-path invariant violation: durable wave DELETE for relation \"{table}\" \
                             declined device publication — refusing acknowledgement; WAL replay is required"
                        );
                    }
                    committed.push((position, commit_seq, item_rows));
                }
                DeviceMaintenance::Update(table, old_rows, new_rows, row_ids) => {
                    let handled =
                        wave_catalog
                            .relational_catalog
                            .get(&table)
                            .is_some_and(|table| {
                                self.try_update_resident_table(
                                    table,
                                    &old_rows,
                                    &new_rows,
                                    commit_seq,
                                    Some(&row_ids),
                                )
                            });
                    if !handled {
                        panic!(
                            "commit-path invariant violation: durable wave UPDATE for relation \"{table}\" \
                             declined device publication — refusing acknowledgement; WAL replay is required"
                        );
                    }
                    committed.push((position, commit_seq, item_rows));
                }
            }
            hp!(6);
        }
        flush_appends(&mut pending_appends, &mut committed);
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
        // Register the applied-but-unpublished tail while the commit lock still excludes explicit
        // transaction validation. The sequencer publishes it into the deque after returning; the
        // applied counter closes that handoff gap for the explicit-transaction settle proof.
        if wave_tail.is_some() {
            self.commit_wave
                .tails_applied
                .fetch_add(1, AtomicOrdering::Release);
        }
        // === leave the commit critical section BEFORE the fsync (D3b group commit) ===
        drop(commit);

        let Some((_last_seq, last_position)) = wave_tail else {
            // Every item aborted pre-durable; outcomes are already set.
            std::mem::forget(guard);
            return None;
        };

        // W2: the durability tail (fsync-wait → publish → acks) no longer runs on the
        // sequencer's critical path — it is handed back as a `CommitWaveTail` for the depth-1
        // pipeline, so the NEXT wave's sequencing overlaps THIS wave's fdatasync. The batch
        // guard's responsibility transfers to the tail's own `armed` Drop.
        std::mem::forget(guard);
        Some(CommitWaveTail {
            batch,
            committed,
            last_position,
            armed: true,
        })
    }

    /// A4e / E2.4a — flush the wave-batched device open-shard append: ONE
    /// `try_append_resident_int4_open_shard` per (table, flush) with per-row birth stamps (the
    /// batch spans commit seqs), the lazy device-authority entry on first successful append, and the
    /// fail-closed publication contract when the device declines. Extracted from the serial
    /// sequencer's inner closure so the sharded sequencer shares the exact same append path.
    fn flush_wave_pending_appends(
        &self,
        pending: &mut WavePendingAppends,
        committed: &mut Vec<(usize, Index, u64)>,
    ) {
        if pending.is_empty() {
            return;
        }
        for (table, (rows, row_ids, items, stamps)) in std::mem::take(pending) {
            // D3 (ADR-013 pre1): the batched flush spans MULTIPLE commit seqs — each row
            // carries its own birth stamp (the per-row slice the D3-COMPOSE note called for).
            let appended = self.try_append_resident_int4_open_shard(
                &table,
                &rows,
                crate::engine_residency::AppendCreatedBy::InsertPerRow(&stamps),
                Some(&row_ids),
            );
            if appended {
                // Establish device authority once per successfully flushed eligible table.
                if !self.table_device_authoritative(&table) {
                    let snapshot = self.catalog_snapshot();
                    if self.table_device_authority_eligible(&snapshot, &table) {
                        self.set_table_device_authoritative(&table, true);
                    }
                }
            } else {
                panic!(
                    "commit-path invariant violation: durable wave DML for relation \"{table}\" \
                     declined device publication — refusing acknowledgement; WAL replay is required"
                );
            }
            for (position, seq, rows) in items {
                committed.push((position, seq, rows));
            }
        }
    }

    /// E2.4a — is `item` a covered-INSERT intent the SHARDED sequencer can fan out? The exact serial
    /// `intent_fast` gate (catalog generation unchanged since prepare, pre-encoded binary WAL
    /// template, reuse-eligible single-row off-lock delta, table still device-authoritative) PLUS a single
    /// integer unique slot and no other conflict dimension. The single-slot restriction is what
    /// makes "hash the unique slot → shard" a CORRECT same-conflict-slot-same-worker partition: a
    /// row with two unique columns could collide with a different row on its SECOND column while
    /// hashing to a different shard, so those (and every non-intent item) stay on the serial path.
    fn item_sharded_intent_eligible(
        &self,
        item: &CommitWaveItem,
        wave_catalog: &CatalogSnapshot,
    ) -> bool {
        wave_catalog.commit_seq == item.prepared_catalog_seq
            && item.binary_wal_template.is_some()
            && item.write_set.unique_slots_i32.len() == 1
            && item.write_set.unique_slots.is_empty()
            && item.write_set.rows.is_empty()
            && matches!(&item.offlock_delta, Some(d)
                if Self::reresolve_reuse_eligible(d, wave_catalog)
                    && matches!(&d.mutation,
                        crate::write_path::PreparedMutation::Insert { inserted_rows, .. }
                            if inserted_rows.len() == 1))
            && match &item.cmd {
                Command::Insert(insert) => self.table_device_authoritative(&insert.table),
                _ => false,
            }
    }

    /// Batched device-history verdict for the homogeneous single-i32-key sharded intent shape.
    /// The visible-locate kernel returns the latest physical version stamp independently of each
    /// item's visibility snapshot, so a claim followed by a delete/key-away still conflicts.
    fn sharded_intent_history_conflicts(
        &self,
        batch: &[CommitWaveItem],
    ) -> std::collections::BTreeSet<usize> {
        type Group = (Vec<i32>, Vec<u64>, Vec<usize>);
        let catalog = self.catalog_snapshot();
        let mut conflicts = std::collections::BTreeSet::new();
        let mut groups = BTreeMap::<(String, usize), Group>::new();
        for (position, item) in batch.iter().enumerate() {
            let Command::Insert(insert) = &item.cmd else {
                conflicts.insert(position);
                continue;
            };
            let Some(table) = catalog.relational_catalog.get(&insert.table) else {
                conflicts.insert(position);
                continue;
            };
            let slot_id = item.write_set.unique_slots_i32[0].0;
            let column_id = slot_id as u32;
            let Some(column_idx) = table
                .columns
                .iter()
                .position(|column| column.id == column_id)
            else {
                conflicts.insert(position);
                continue;
            };
            let group = groups
                .entry((insert.table.clone(), column_idx))
                .or_default();
            group.0.push(item.write_set.unique_slots_i32[0].1);
            group.1.push(item.read_snapshot);
            group.2.push(position);
        }
        for ((table_name, column_idx), (needles, snapshots, positions)) in groups {
            let Some(table) = catalog.relational_catalog.get(&table_name) else {
                conflicts.extend(positions);
                continue;
            };
            let Some(locate) =
                self.wave_batch_visible_locate(table, column_idx, &needles, &snapshots)
            else {
                conflicts.extend(positions);
                continue;
            };
            let expected = positions.len();
            if !exact_device_verdict_cardinality(
                expected,
                &[
                    needles.len(),
                    snapshots.len(),
                    locate.counts.len(),
                    locate.shard_ids.len(),
                    locate.slots.len(),
                    locate.row_ids.len(),
                    locate.latest_write.len(),
                ],
            ) {
                conflicts.extend(positions);
                continue;
            }
            for ((position, latest), snapshot) in positions
                .iter()
                .copied()
                .zip(locate.latest_write.iter().copied())
                .zip(snapshots.iter().copied())
            {
                if latest > snapshot {
                    conflicts.insert(position);
                }
            }
        }
        conflicts
    }

    /// E2.4a VARIANT 1 — sequence a HOMOGENEOUS covered-INSERT-intent wave with N parallel shard
    /// workers over ONE ordered WAL + ONE global commit-seq.
    ///
    /// Stage 1 (device, coordinator): the wave-batched PK-unique locate — one device call, the
    /// committed-dup 23505 verdicts. Stage 2 (N parallel workers under the coordinator's retained
    /// publication boundary): each worker owns the wave positions whose unique slot hashes to its
    /// shard and, in wave-position order, consumes the batched device-history verdict plus a
    /// per-shard PRIVATE dedup set
    /// (same-slot → same shard, so the lowest-position writer wins — byte-identical to the serial
    /// record-as-you-go single-winner), then clones the row image + WAL record. Stage 3 (thin
    /// serial cut, commit lock): walk the wave in order, and for each committing item claim the
    /// next commit-seq + integer row id, patch the WAL record's row id, append + propose, record the
    /// commit timestamp + row-identity write-set record, advance the
    /// elided row-id allocator, and buffer the row into the per-table device append. The device
    /// append flushes ONCE per table at the tail (one HtoD/wave); the durability tail is returned
    /// for the W2 pipeline exactly as the serial path.
    fn sequence_commit_wave_sharded(
        &self,
        batch: Vec<CommitWaveItem>,
        shards: usize,
    ) -> Option<CommitWaveTail> {
        let guard = CommitWaveBatchGuard {
            engine: self,
            items: &batch,
        };
        let wall_clock = current_timestamp_micros();
        let n = batch.len();

        // Acquire the publication boundary BEFORE validation. Serialized/classic writers and DDL
        // publish under this same lock, so neither a row/version nor a catalog generation can change
        // between the device verdict and this wave's ordered WAL cut.
        let mut commit = self.commit_state();
        if let Err(error) = self.ensure_commit_path_available() {
            let message = error.to_string();
            for item in &batch {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                    message.clone(),
                ))));
            }
            std::mem::forget(guard);
            return None;
        }
        if let Err(error) = self.legacy_lane_history_write_guard() {
            let message = error.to_string();
            for item in &batch {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                    message.clone(),
                ))));
            }
            std::mem::forget(guard);
            return None;
        }

        // Stage 1 — the wave-batched device PK-unique locate (committed-dup 23505 verdicts), now
        // inside the serialized publication boundary. Holding the lock through worker preparation
        // is intentionally conservative: the verdict cannot become stale before Stage 3.
        let hostphase = wave_host_phase_timing_enabled();
        let wave_validate_started = hostphase.then(Instant::now);
        let wave_unique_verdicts = self.wave_batch_validate_unique(&batch);
        let mut wave_history_conflicts = self.sharded_intent_history_conflicts(&batch);
        wave_history_conflicts.extend(wave_unique_verdicts.declined.iter().copied());
        let serialized_catalog_seq = self.catalog_snapshot().commit_seq;
        wave_history_conflicts.extend(
            batch
                .iter()
                .enumerate()
                .filter(|(_, item)| item.prepared_catalog_seq != serialized_catalog_seq)
                .map(|(position, _)| position),
        );
        // The durable map is populated only after winner sequencing below, so arbitrate transaction
        // identities within this candidate wave as well. Lowest position owns a new identity;
        // later exact duplicates remain pending on that owner, while payload mismatch fails loud.
        let mut wave_transaction_claims = BTreeMap::new();
        for (position, item) in batch.iter().enumerate() {
            let request_digest = gpu_db_wal::canonical_request_digest(&item.payload);
            match commit.resolve_transaction_retry_digest_outcome(item.txn_id, request_digest) {
                Ok(Some((token, affected_rows))) if self.committed_seq() >= token.index => {
                    item.set_outcome(Ok(affected_rows));
                    wave_history_conflicts.insert(position);
                }
                Ok(Some(_)) => {
                    item.set_outcome(Err(ExecuteError::Indeterminate(format!(
                        "transaction id {} is committed but not yet publication-covered",
                        item.txn_id
                    ))));
                    wave_history_conflicts.insert(position);
                }
                Err(error) => {
                    item.set_outcome(Err(ExecuteError::Engine(error)));
                    wave_history_conflicts.insert(position);
                }
                Ok(None) => {
                    if let Some(expected) = item.expected_catalog_version {
                        if let Err(error) =
                            crate::engine_mutation_admission::validate_prepared_catalog_version(
                                expected,
                                serialized_catalog_seq,
                            )
                        {
                            item.set_outcome(Err(error));
                            wave_history_conflicts.insert(position);
                            continue;
                        }
                    }
                    match self.resolve_pending_transaction_claim(item.txn_id, request_digest) {
                        Ok(true) => {
                            item.set_outcome(Err(ExecuteError::Indeterminate(format!(
                                "transaction id {} is pending in canonical mutation admission",
                                item.txn_id
                            ))));
                            wave_history_conflicts.insert(position);
                            continue;
                        }
                        Err(error) => {
                            item.set_outcome(Err(ExecuteError::Engine(error)));
                            wave_history_conflicts.insert(position);
                            continue;
                        }
                        Ok(false) => {}
                    }
                    if let Some(state) = commit.txn_manager.state(item.txn_id) {
                        item.set_outcome(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            format!(
                                "transaction id {} is already owned by transaction state {state:?}",
                                item.txn_id
                            ),
                        ))));
                        wave_history_conflicts.insert(position);
                        continue;
                    }
                    match wave_transaction_claims.entry(item.txn_id) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(request_digest);
                        }
                        std::collections::btree_map::Entry::Occupied(entry)
                            if *entry.get() == request_digest =>
                        {
                            item.set_outcome(Err(ExecuteError::Indeterminate(format!(
                                "transaction id {} is already pending earlier in this commit wave",
                                item.txn_id
                            ))));
                            wave_history_conflicts.insert(position);
                        }
                        std::collections::btree_map::Entry::Occupied(_) => {
                            item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                                format!(
                                    "transaction id {} is already claimed earlier in this commit wave by a different request",
                                    item.txn_id
                                ),
                            ))));
                            wave_history_conflicts.insert(position);
                        }
                    }
                }
            }
        }
        #[cfg(test)]
        {
            let hook = {
                let mut hook = sharded_post_validation_hook()
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
        if let Some(started) = wave_validate_started {
            WAVE_HOST_STATS[0]
                .fetch_add(started.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }
        // Stage 2 — parallel shard prep against the device-history verdict plus per-shard private
        // same-wave dedup. The coordinator resumes the ordered commit cut after join.
        let shard_started = hostphase.then(Instant::now);
        let mut verdicts = self.shard_prepare_intents(
            &batch,
            &wave_unique_verdicts.violations,
            &wave_history_conflicts,
            shards,
        );
        if let Some(started) = shard_started {
            // Charge the parallel prep to the conflict bucket (it subsumes conflict + reresolve).
            WAVE_HOST_STATS[1]
                .fetch_add(started.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }

        // Stage 3 — the thin serial cut, E2.5b BATCHED: one commit-seq block claim, one
        // repl propose_batch, one row-id block, one applied mark, one timestamp for the whole
        // wave. The per-item body is reduced to the WAL push, row-identity record, and the
        // device-buffer push — the E2.4a measurement showed the per-item repl round-trips,
        // BTreeMap timestamp insert, and allocator atomics WERE the ordered cut (~1.3us/item).
        // Aborts never consume a commit seq (same as the serial path's peek-before-propose), and
        // a propose_batch failure aborts the WHOLE wave — identical semantics to a first-item
        // propose failure, since the single-node leader either accepts all or is not leader.
        let mut wave_tail: Option<(Index, usize)> = None;
        let mut committed: Vec<(usize, Index, u64)> = Vec::with_capacity(n);
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
                let canonical = match Self::canonical_wal_record_with_commit_request_digest(
                    &commit,
                    batch[*position].txn_id,
                    first_seq + offset as u64,
                    0,
                    &wal_payload,
                    gpu_db_wal::canonical_request_digest(&batch[*position].payload),
                ) {
                    Ok(record) => record,
                    Err(err) => {
                        commit.wal.truncate(wal_len_before);
                        for (position, _table, _values, _record, _offset) in winners.into_iter() {
                            batch[position].set_outcome(Err(ExecuteError::Engine(
                                EngineError::Durability(format!(
                                    "canonical wave WAL encode failed: {err}"
                                )),
                            )));
                        }
                        return None;
                    }
                };
                commit.wal.append(canonical);
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
                        if commit
                            .record_transaction_status_digest_outcome(
                                batch[position].txn_id,
                                gpu_db_wal::canonical_request_digest(&batch[position].payload),
                                commit_seq,
                                1,
                            )
                            .is_err()
                        {
                            return None;
                        }
                        commit.record_commit_timestamp(
                            batch[position].txn_id,
                            base_timestamp_micros + offset as u64,
                        );
                        // Record row identities in wave order. Unique-slot history is already
                        // resolved by device history + worker-local single-winner arbitration.
                        commit.ledger.record(&batch[position].write_set, commit_seq);
                        let entry = pending_appends.entry(table).or_default();
                        entry.3.push(commit_seq);
                        entry.0.push(values);
                        entry.1.push(row_id_base + offset as u64);
                        // Sharded winners are single-row covered INSERTs by eligibility.
                        entry.2.push((position, commit_seq, 1));
                    }
                    // Device-authoritative batched apply: advance the row-id allocator and authority
                    // counter by the whole wave, then mark the block applied once.
                    self.read_state.mvcc.advance_row_id(k);
                    self.read_state
                        .residency
                        .device_authoritative_commits
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
        self.flush_wave_pending_appends(&mut pending_appends, &mut committed);
        if let Some(started) = cut_started {
            // Charge the ordered serial cut (WAL append + commit-seq + row record +
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
        // Register under the commit lock for the same apply-to-deque handoff proof as the serial
        // sequencer. `tails_finished` catches up only after durability and visibility publication.
        if wave_tail.is_some() {
            self.commit_wave
                .tails_applied
                .fetch_add(1, AtomicOrdering::Release);
        }
        // === leave the commit critical section BEFORE the fsync (D3b group commit) ===
        drop(commit);

        let Some((_last_seq, last_position)) = wave_tail else {
            // Every item aborted pre-durable; outcomes are already set.
            std::mem::forget(guard);
            return None;
        };

        std::mem::forget(guard);
        Some(CommitWaveTail {
            batch,
            committed,
            last_position,
            armed: true,
        })
    }

    /// E2.4a — the parallel shard-prep pass (Stage 2 of [`Engine::sequence_commit_wave_sharded`]).
    /// Partitions the wave's positions across `shards` workers by hashing each row's single unique
    /// slot (so same-slot rows land in the same worker) and, per worker, computes a per-position
    /// [`ShardVerdict`] in wave-position order: 23505 for a committed-dup (device-locate verdict),
    /// a retryable serialization abort for a device-history conflict OR an intra-wave same-slot
    /// duplicate (the per-shard private dedup set — lowest position wins), else a Commit carrying
    /// the cloned row image + the cloned (still-placeholder-row-id) WAL record.
    fn shard_prepare_intents(
        &self,
        batch: &[CommitWaveItem],
        unique_violations: &std::collections::BTreeMap<usize, String>,
        history_conflicts: &std::collections::BTreeSet<usize>,
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
                                ShardVerdict::Abort(ExecuteError::Engine(EngineError::UniqueViolation(
                                    msg.clone(),
                                )))
                            } else if history_conflicts.contains(&pos) {
                                let rs = item.read_snapshot;
                                ShardVerdict::Abort(ExecuteError::Serialization(format!(
                                    "write-write conflict on a key committed after read snapshot {rs}"
                                )))
                            } else {
                                let slot = item.write_set.unique_slots_i32[0];
                                if !private.insert(slot) {
                                    // Intra-wave same-slot duplicate: the later position loses,
                                    // exactly the serial wave-local first-committer-wins verdict.
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

/// Bind the DEVICE-PROBE `(key_id, needle)` for `index` against this insert's row. Raw single-i32
/// indexes use `(col_idx, value)`; compound and single wider/text indexes use `(FLAG | ord,
/// fingerprint)` over the canonical typed words. `None` declines to full validation.
fn insert_index_probe_needle(
    insert: &Insert,
    table: &RelationalTable,
    index: &RelationalIndex,
    ord: usize,
) -> Option<(usize, i32)> {
    if crate::engine_residency::index_uses_fingerprint(table, index) {
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
