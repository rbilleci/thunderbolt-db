//! GPU residency management + resident-route planning (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for populating/admitting
//! resident snapshots (incl. on-GPU), the benchmark chunk/shard installs,
//! resident device-memory + bytes accounting, retained-read snapshot handles,
//! warmup/maintenance policy execution, and the resident-route planners
//! (plan_relational_resident_route + sharded variant) + residency status.

use super::*;

/// Snapshot construction, admission, and publication ownership.
mod admission;
/// Resident append, rollover, sparse-version stamping, and fused-apply ownership.
mod mutation;
/// Typed payload/key encoding and open-shard append construction.
mod payload;
/// Residency feature policy, elision eligibility, and telemetry ownership.
mod policy;
/// Commit auto-admission, transient relations, and benchmark installation ownership.
mod transient;

pub(crate) use payload::{
    AppendCreatedBy, COMPOUND_KEY_ID_FLAG, CREATED_BY_VISIBLE_FILL_BYTE,
    DELETED_BY_LIVE_FILL_BYTE, ROW_ID_UNSTAMPED_FILL_BYTE, UnifiedResidentSnapshotParts,
    build_relational_device_payload, build_relational_device_payload_with_capacity,
    compound_index_row_fingerprint, compound_key_fingerprint, compound_key_type_supported,
    compound_unique_slot_id, compute_open_shard_int4_append_chunks,
    i32_section_needle, index_all_key_columns_foldable, index_is_compound,
    index_key_column_positions, index_probe_key_id, key_column_width_words,
    parse_relational_row_id, probe_key_id_positions, sql_value_as_int4,
    sql_value_from_i32_section, sql_value_from_i64_section, sql_value_key_words,
};

#[cfg(test)]
// STRUCT-001 keeps this parent-positioned include as a source-reconstruction boundary. Moving the
// 6,819-line test owner after production items would obscure exact extraction history for no runtime gain.
#[allow(clippy::items_after_test_module)]
mod capacity_payload_tests {
    use super::*;

    fn int4_cols() -> (Vec<String>, Vec<SqlType>) {
        (
            vec!["id".to_string(), "balance".to_string()],
            vec![SqlType::Int4, SqlType::Int4],
        )
    }
    fn int4_rows(n: i32) -> Vec<Vec<SqlValue>> {
        (0..n)
            .map(|i| vec![SqlValue::Int4(i), SqlValue::Int4(i * 10)])
            .collect()
    }

    include!("tests/residency_payload.rs");

    include!("tests/residency_shard_baseline.rs");

    /// Read a shard's ON-DEMAND `deleted_by` region back from device (DtoH), first `count` slots. Returns
    /// `None` when the shard has NO region (delete-free). u64 reconstructed from i32 LE (lo, hi) pairs.
    fn read_shard_deleted_by_region(
        e: &Engine,
        table: &str,
        shard_id: u32,
        count: usize,
    ) -> Option<Vec<u64>> {
        let region = e
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table.to_string(), shard_id))?;
        let halves = region
            .read_resident_i32_column(0, count * 2)
            .expect("read deleted_by region");
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let lo = halves[2 * i] as u32 as u64;
            let hi = halves[2 * i + 1] as u32 as u64;
            out.push((hi << 32) | lo);
        }
        Some(out)
    }

    /// Test helper: does ANY shard of `table` currently hold a LIVE `deleted_by` region (cell present AND
    /// `Some`)? False after either `invalidate_table` (publishes `None`, device buffer freed, cell kept) or
    /// `remove_table` (cell dropped). Use to prove a region was RELEASED. Reads the published cell map
    /// directly (in-crate).
    fn table_has_any_deleted_by_cell(e: &Engine, table: &str) -> bool {
        e.read_state
            .residency
            .shard_deleted_by_memory
            .cells
            .load()
            .iter()
            .any(|((cell_table, _), cell)| cell_table == table && cell.load().get().is_some())
    }

    /// Test helper: does ANY cell KEY for `table` still exist (regardless of `Some`/`None`)? Distinguishes
    /// `invalidate_table` (key KEPT as a `None` tombstone) from `remove_table` (key ERASED). Use to prove
    /// DROP fully removes the entry -- invalidate alone would leak a dangling `None` key per dropped table.
    fn table_has_any_deleted_by_key(e: &Engine, table: &str) -> bool {
        e.read_state
            .residency
            .shard_deleted_by_memory
            .cells
            .load()
            .keys()
            .any(|(cell_table, _)| cell_table == table)
    }

    /// SV6 test helper: does ANY shard of `table` hold a LIVE `created_by` region? Mirrors
    /// `table_has_any_deleted_by_cell` — the presence proof that the UPDATE-append STAMP path ran (a
    /// re-admit fallback rebuilds all-live with NO region), and the release proof for the lifecycle gates.
    fn table_has_any_created_by_cell(e: &Engine, table: &str) -> bool {
        e.read_state
            .residency
            .shard_created_by_memory
            .cells
            .load()
            .iter()
            .any(|((cell_table, _), cell)| cell_table == table && cell.load().get().is_some())
    }

    /// SV6 test helper: does ANY `created_by` cell KEY for `table` still exist? Mirrors
    /// `table_has_any_deleted_by_key` (DROP must erase keys, not just publish `None`).
    fn table_has_any_created_by_key(e: &Engine, table: &str) -> bool {
        e.read_state
            .residency
            .shard_created_by_memory
            .cells
            .load()
            .keys()
            .any(|(cell_table, _)| cell_table == table)
    }

    include!("tests/residency_sparse_visibility.rs");

    include!("tests/residency_update_visibility.rs");

    include!("tests/residency_pk_index.rs");

    include!("tests/residency_route_parity.rs");

    include!("tests/residency_elision_core.rs");

    include!("tests/residency_type_coverage.rs");

    include!("tests/residency_device_locate.rs");

    include!("tests/residency_wide_type_reads.rs");

    include!("tests/residency_elision_waves.rs");

    include!("tests/residency_maintenance_materialization.rs");

    include!("tests/residency_identity_validation.rs");

    include!("tests/residency_sharded_point_reads.rs");

    include!("tests/residency_capacity_budget.rs");
}

impl Engine {
    /// RETIREMENT A4e (audit B3): rehydrate an elided table FROM AN OFF-COMMIT-LOCK context
    /// (the CPU-shape read seam, the execute_text DDL entry). `rehydrate_elided_table` mutates the
    /// host store via COW `with_table_mut` — safe ONLY under the commit lock (writers + other
    /// rehydrators serialize there; a lost-update would leave the table DE-ELIDED WITH A STALE
    /// STORE = permanent wrong reads). Mid-commit internal reads (matview refresh) already HOLD
    /// the lock — detected via the same thread-local that suppresses their leader check — so they
    /// rehydrate directly (a second acquisition would self-deadlock). The elided-ness RE-CHECK
    /// under the lock closes the race with a rehydrator that won the lock first.
    /// VACUUM #5 (A5 gate): REBUILD a churned table's residency DENSE + ALL-LIVE — reclaims
    /// tombstoned slots and stale duplicate physical keys (an SV5/A4b update-append leaves the
    /// old version's slot holding the key, which dup-declines the per-shard PK index until a
    /// rebuild changes the buffer ptr — the monotone decline clears BY DESIGN on a new
    /// generation). Composition of audited pieces: an ELIDED table first REHYDRATES (the A4c
    /// device gather is the truth; the host store is a stale prefix), then the standard
    /// invalidate + re-admit rebuilds dense from the now-complete store; a non-elided table's
    /// store is already complete, so it skips straight to the rebuild. The table RE-ENTERS
    /// elision on its next handled commit (the normal entry path) — vacuum does not special-case
    /// it. Runs under the COMMIT LOCK (the same discipline as `rehydrate_elided_serialized`; the
    /// mid-commit-read detection makes an auto-trigger from inside a commit safe). The churn
    /// counter resets so the auto-trigger re-arms.
    ///
    /// V2 (ledgered): KEY-CLUSTERED rebuild (feed the builder rows sorted by PK so zone maps
    /// tighten under update scatter) — needs the slot-order-decoupled builder.
    pub fn vacuum_table(&self, table_name: &str) -> Result<(), EngineError> {
        if self.mvcc_read_skips_leader_check() {
            return self.vacuum_table_locked(table_name);
        }
        let _commit_guard = self.commit_state();
        self.vacuum_table_locked(table_name)
    }

    /// The vacuum core for callers ALREADY under the commit lock (the auto-trigger fires inside
    /// the serialized commit arm; a second acquisition would self-deadlock).
    pub(crate) fn vacuum_table_locked(&self, table_name: &str) -> Result<(), EngineError> {
        let run = |engine: &Self| -> Result<(), EngineError> {
            let Some(table) = engine.relational_catalog_table(table_name) else {
                return Ok(()); // no such table: vacuum is a no-op, not an error
            };
            if engine.table_install_elided(table_name) {
                let seq = engine.committed_seq();
                engine.rehydrate_elided_table(
                    &table,
                    seq,
                    &Default::default(),
                    &Default::default(),
                    seq,
                )?;
            }
            let seq = engine.committed_seq();
            let tables: std::collections::BTreeSet<String> =
                std::iter::once(table_name.to_string()).collect();
            engine.invalidate_relational_residency_tables_concurrent(&tables, seq, seq);
            if engine.auto_admit_on_commit_enabled() {
                engine.auto_admit_resident_tables(&tables);
            }
            engine.reset_tombstone_churn(table_name);
            Ok(())
        };
        run(self)
    }

    /// VACUUM #5: enable/disable the churn-triggered AUTO vacuum (default OFF — the A/B lever;
    /// `vacuum_table` stays callable either way).
    pub fn set_auto_vacuum_enabled(&self, on: bool) {
        self.auto_vacuum_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// VACUUM #5 (audit F1): deferred auto-vacuums that failed (telemetry; the trigger re-arms).
    pub fn auto_vacuum_failures(&self) -> u64 {
        self.read_state
            .residency
            .auto_vacuum_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn auto_vacuum_enabled(&self) -> bool {
        self.auto_vacuum_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// VACUUM #5: bump the per-table churn counter by `stamps` tombstones (serialized path only)
    /// and return the new value.
    pub(crate) fn add_tombstone_churn(&self, table: &str, stamps: u64) -> u64 {
        let cur = self.read_state.residency.resident_tombstone_churn.load();
        let mut next = (**cur).clone();
        let counter = next.entry(table.to_string()).or_insert(0);
        *counter += stamps;
        let value = *counter;
        self.read_state
            .residency
            .resident_tombstone_churn
            .store(std::sync::Arc::new(next));
        value
    }

    pub(crate) fn reset_tombstone_churn(&self, table: &str) {
        let cur = self.read_state.residency.resident_tombstone_churn.load();
        if !cur.contains_key(table) {
            return;
        }
        let mut next = (**cur).clone();
        next.remove(table);
        self.read_state
            .residency
            .resident_tombstone_churn
            .store(std::sync::Arc::new(next));
    }

    /// VACUUM #5: the churn threshold — max(1024, table's resident rows / 8). Above it the
    /// auto-trigger rebuilds (dead slots ≥ ~12% bloat scans and keep the PK index dup-declined).
    pub(crate) fn tombstone_churn(&self, table: &str) -> u64 {
        self.read_state
            .residency
            .resident_tombstone_churn
            .load()
            .get(table)
            .copied()
            .unwrap_or(0)
    }

    /// Test lever: force the auto-vacuum threshold (0 = the size-derived default).
    pub fn set_tombstone_churn_threshold_override(&self, threshold: u64) {
        self.tombstone_churn_threshold_override
            .store(threshold, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn tombstone_churn_threshold(&self, table: &str) -> u64 {
        let forced = self
            .tombstone_churn_threshold_override
            .load(std::sync::atomic::Ordering::Relaxed);
        if forced != 0 {
            return forced;
        }
        let rows: usize = self
            .read_state
            .residency
            .shards
            .load()
            .get(table)
            .map(|shards| shards.iter().map(|shard| shard.row_count).sum())
            .unwrap_or(0);
        (rows as u64 / 8).max(1024)
    }

    pub(crate) fn rehydrate_elided_serialized(&self, table_name: &str) -> Result<(), EngineError> {
        let rehydrate = |engine: &Self| -> Result<(), EngineError> {
            if !engine.table_install_elided(table_name) {
                return Ok(()); // another rehydrator won the race
            }
            // PUBLISHED-SNAPSHOT catalog read, NEVER `relational_catalog_table` (audit f80f2350
            // FINDING A, second cycle): that accessor takes the CATALOG LATCH, and this seam is
            // reachable from the DDL apply loop which already HOLDS it (the internal-read flag
            // wrap) — the re-acquire self-deadlocked (gdb-verified: apply_and_publish held
            // commit_mutex + latch, this closure blocked in ddl_catalog()). The published
            // snapshot is layout-correct here: any column-shape-changing DDL rehydrates via the
            // pre-commit execute_text sweep, so the mid-apply seam only fires on the re-elision
            // race, where the layout is unchanged (the in-vacuum catalog-latch lesson, again).
            let Some(table) = engine
                .catalog_snapshot()
                .relational_catalog
                .get(table_name)
                .cloned()
            else {
                return Ok(());
            };
            let seq = engine.committed_seq();
            engine.rehydrate_elided_table(
                &table,
                seq,
                &Default::default(),
                &Default::default(),
                seq,
            )
        };
        if self.mvcc_read_skips_leader_check() {
            // Mid-commit internal read: the commit lock is already held by THIS thread.
            return rehydrate(self);
        }
        let _commit_guard = self.commit_state();
        rehydrate(self)
    }

    /// RETIREMENT A4e: REHYDRATE an elided table — the STICKY DE-ELISION transition. The A4c
    /// gather (at `read_txn`, the last seq whose state the device fully holds) repopulates the
    /// host tuple store + value indexes THROUGH the normal install path (clearing the stale
    /// pre-elision prefix first), then the table LEAVES the elided set. Callers: a DML
    /// prepare/probe whose device resolve declines on an elided table (then the host path
    /// proceeds, always correct), and the commit arm's !handled fallback (then the re-admit
    /// rebuilds from the now-complete store). `extra_rows` carries an in-flight commit's rows
    /// (the mutation the device could NOT absorb — e.g. a NULL append) that the gather at
    /// `read_txn = C-1` cannot see. Returns Err when the gather declines — for an elided table
    /// that is a broken invariant (elision eligibility ⊆ gather eligibility), and failing LOUDLY
    /// beats a silently incomplete store.
    /// U1: resolve elided rows' identities BY int4 KEY against the device gather at `read_txn`
    /// — the rare lane-delete fallback's removal set (the tombstones' (shard, slot) targets are
    /// exactly what a declined/stale device generation can no longer be trusted for; the KEY is
    /// generation-independent). `keys` are `(column_index, value)`; a key with no visible match
    /// at `read_txn` resolves to nothing (its delete was against a row this gather cannot see —
    /// impossible for a wave-located 1-row target, but the resolve is total rather than lossy).
    /// Returns `(row-id removals for the rehydrate, the KEY VALUES that matched a visible row)`
    /// — the matched-key set lets the WAL-first delete fallback set each delete's rows-affected
    /// (1 if its key matched, else 0).
    pub(crate) fn resolve_elided_row_ids_by_int4_key(
        &self,
        table: &RelationalTable,
        read_txn: u64,
        keys: &[(usize, i32)],
    ) -> Result<
        (
            std::collections::BTreeSet<u64>,
            std::collections::HashSet<i32>,
        ),
        EngineError,
    > {
        if keys.is_empty() {
            return Ok(Default::default());
        }
        let gathered = self
            .gather_resident_table_rows_from_device(table, read_txn)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "tombstone key-resolution gather declined for elided table \"{}\"",
                    table.name
                ))
            })?;
        let mut removals = std::collections::BTreeSet::new();
        let mut matched_keys = std::collections::HashSet::new();
        for (row_id, values) in &gathered {
            for &(column, key) in keys {
                if values.get(column) == Some(&SqlValue::Int4(key)) {
                    removals.insert(*row_id);
                    matched_keys.insert(key);
                }
            }
        }
        Ok((removals, matched_keys))
    }

    pub(crate) fn rehydrate_elided_table(
        &self,
        table: &RelationalTable,
        read_txn: u64,
        upserts: &std::collections::BTreeMap<u64, Vec<SqlValue>>,
        removals: &std::collections::BTreeSet<u64>,
        commit_seq: u64,
    ) -> Result<(), EngineError> {
        let gathered = self
            .gather_resident_table_rows_from_device(table, read_txn)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "rehydration gather declined for elided table \"{}\" — device-authoritative \
                     invariant broken (WAL replay is the recovery path)",
                    table.name
                ))
            })?;
        let prefix = relational_key_prefix(&table.name);
        // The gather is the device truth at read_txn; the in-flight commit's delta (the mutation
        // the device could NOT absorb) applies ON TOP: upserts overwrite by identity (an UPDATE
        // keeps its row_id — the new image wins), removals drop (a DELETE the apply skipped).
        let mut merged: std::collections::BTreeMap<u64, Vec<SqlValue>> =
            gathered.into_iter().collect();
        for (row_id, row) in upserts {
            merged.insert(*row_id, row.clone());
        }
        for row_id in removals {
            merged.remove(row_id);
        }
        let install: Vec<(u64, Vec<SqlValue>)> = merged.into_iter().collect();
        let visibility = crate::StorageVisibility {
            read_txn_id: read_txn,
        };
        self.read_state.mvcc.with_table_mut(&table.name, |data| {
            // RECONCILE, not clear+reinsert: the stale pre-elision prefix rows update in place
            // (same key -> tuple_update), gathered-only keys insert, host-only keys (deleted
            // during the elided era) tombstone. The per-table value_index rebuilds wholesale.
            let mut stale: std::collections::BTreeMap<String, u64> = Default::default();
            {
                let mut cursor = data
                    .rows
                    .seq_scan_open(visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                while let Some(tuple) = cursor.next() {
                    if tuple.key.starts_with(&prefix) {
                        stale.insert(tuple.key.clone(), tuple.tuple_id);
                    }
                }
            }
            for (row_id, row) in &install {
                let key = relational_row_key(&table.name, *row_id);
                if let Some(tuple_id) = stale.remove(&key) {
                    data.rows
                        .tuple_update(tuple_id, encode_relational_row(row), commit_seq)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                } else {
                    let tuple_id = self.read_state.mvcc.reserve_tuple_id();
                    data.rows
                        .tuple_insert_reserved_key_with_id(
                            tuple_id,
                            gpu_db_storage::NewTuple {
                                key: key.clone(),
                                value: encode_relational_row(row),
                            },
                            commit_seq,
                        )
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
            }
            for (_key, tuple_id) in stale {
                data.rows
                    .tuple_delete(tuple_id, commit_seq)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            }
            data.value_index.clear();
            for (row_id, row) in &install {
                let key = relational_row_key(&table.name, *row_id);
                for (idx, column) in table.columns.iter().enumerate() {
                    let slot_key = crate::resident_storage::ColumnValueKey {
                        column: column.name.clone(),
                        value: relational_index_value(&row[idx]),
                    };
                    let mut slot = data.value_index.get(&slot_key).cloned().unwrap_or_default();
                    slot.push_back(key.clone());
                    data.value_index.insert(slot_key, slot);
                }
            }
            Ok::<(), EngineError>(())
        })?;
        self.set_table_install_elided(&table.name, false);
        Ok(())
    }

    /// RETIREMENT A4c: gather a shard-resident table's VISIBLE rows + identities ENTIRELY FROM
    /// THE DEVICE — the rebuild source that replaces the host store for re-admits and for the
    /// eligibility de-elision transition once A4e stops installing host rows. Per shard: one bulk
    /// DtoH per int4 column + the row_id/deleted_by/created_by regions, then the host-side
    /// SV3b/SV6 visibility filter (`created_by <= read_txn < deleted_by`) — an amortized-once
    /// control-plane readback (the DATA SOURCE is the device generation, not host tuples); the
    /// device-to-device recompaction that avoids the round-trip is the ledgered follow-up.
    /// Returns rows in (shard, slot) order with their identities. `None` = DECLINE (caller must
    /// use the host store): invalid/mismatched shard, null-bearing shard (raw i32 would alias
    /// NULL as 0), non-strictly-Int4 table (Date/Int2 would mistype — the A4a F1 discipline), a
    /// missing identity region, an UNSTAMPED live slot (identity hole), or a device-read failure.
    /// Same born-visible contract as A4a, PLUS snapshot freshness (audit A4c F2): callers must
    /// run on the SERIALIZED commit path with `read_txn` >= every INSERT-appended slot's commit
    /// AND the loaded shard snapshot already reflecting every commit <= `read_txn` (re-admit
    /// callers pass the invalidating commit's seq or newer). Completeness rests on the pinned
    /// `row_count` bounding born-visible slots and on seq monotonicity making any concurrent
    /// commit's mutations (seq > read_txn) correctly invisible to the sequential region reads.
    pub(crate) fn gather_resident_table_rows_from_device(
        &self,
        table: &RelationalTable,
        read_txn: u64,
    ) -> Option<Vec<(u64, Vec<SqlValue>)>> {
        // TYPE-COVERAGE track 2 (stage iii): every FIXED-WIDTH-section type gathers with its
        // catalog-derived variant (i32 via one u32/slot; i64 via two — the 4-mod-8 discipline).
        // TYPE-COVERAGE #14 (bool/numeric/uuid): these also gather here — this is the DEVICE->HOST
        // rehydration a read shape the on-device routes can't serve falls back to (a filtered bool/
        // numeric projection, an ORDER BY on a bool key). Without it an elided table with such a column
        // would hard-error on those shapes. Bool = 1 bit/row bitmap; Numeric/Uuid = the 16-byte b128
        // section (numeric = i128 mantissa LE at the catalog scale; uuid = the raw 16 bytes); Text = the
        // offsets section + bytes blob. Every elision-eligible type now rehydrates (nothing declined by type).
        if table.columns.iter().any(|column| {
            !matches!(
                column.ty,
                gpu_db_sql::SqlType::Int4
                    | gpu_db_sql::SqlType::Date
                    | gpu_db_sql::SqlType::Int2
                    | gpu_db_sql::SqlType::Int8
                    | gpu_db_sql::SqlType::Timestamp
                    | gpu_db_sql::SqlType::Bool
                    | gpu_db_sql::SqlType::Numeric { .. }
                    | gpu_db_sql::SqlType::Uuid
                    | gpu_db_sql::SqlType::Text
            )
        }) {
            return None;
        }
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut out: Vec<(u64, Vec<SqlValue>)> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            // D4 (ADR-013 pre2): buffer + identity + version regions all ride the loaded descriptor
            // — the gather's freshness seam (audit A4c F2) now holds by construction instead of by
            // four separate map loads racing a republish.
            let device_memory = shard.device_memory.clone()?;
            let row_id_region = shard.row_id_region.clone()?;
            let rows = shard.row_count;
            // ADR-006 (NULL coverage): per CATALOG column, the shard's validity bitmap words (ceil(rows/32)
            // u32, read as i32; bit `slot` = 1 => valid, 0 => NULL) if the column carries one, else `None`
            // (no NULLs => all-valid). A sparse map: only null-bearing columns have a layout. This lets the
            // gather (the rehydrate source) materialize a null-bearing shard instead of declining +
            // hard-erroring, so a null-carrying elided table's rehydrate is CORRECT (was: unrecoverable).
            let null_bitmaps: Vec<Option<Vec<i32>>> = {
                let mut per_col = Vec::with_capacity(table.columns.len());
                for column in &table.columns {
                    match shard
                        .resident_device_null_columns
                        .iter()
                        .find(|layout| layout.name == column.name)
                    {
                        Some(layout) => per_col.push(Some(
                            device_memory
                                .read_resident_i32_column(
                                    layout.bitmap_byte_offset,
                                    rows.div_ceil(32),
                                )
                                .ok()?,
                        )),
                        None => per_col.push(None),
                    }
                }
                per_col
            };
            // Bulk DtoH: identities (2 i32 halves LE per slot), then each column's live prefix.
            let id_halves = row_id_region.read_resident_i32_column(0, rows * 2).ok()?;
            let deleted = match &shard.deleted_by_region {
                Some(region) => Some(region.read_resident_i32_column(0, rows * 2).ok()?),
                None => None,
            };
            let created = match &shard.created_by_region {
                Some(region) => Some(region.read_resident_i32_column(0, rows * 2).ok()?),
                None => None,
            };
            // Per-column raw reads: i32 sections one u32/slot, i64 sections two u32/slot (the
            // halves pair below). The enum keeps slot addressing uniform for the typing zip.
            enum GatheredColumn {
                I32(Vec<i32>),
                I64(Vec<i32>),
                // TYPE-COVERAGE #14 (bool): the raw bitmap words (ceil(rows/32) u32, read as i32); bit
                // `slot` = word[slot/32] >> (slot%32) & 1.
                Bool(Vec<i32>),
                // TYPE-COVERAGE #14 (numeric/uuid): the b128 section as FOUR i32 words/slot (16 LE
                // bytes/row). Reassembled to i128 per row: numeric = the mantissa (at the catalog
                // scale); uuid = the raw 16 bytes.
                B128(Vec<i32>),
                // TYPE-COVERAGE #14 (text): the offsets (row+1 u64) + the bytes blob; row `slot` =
                // blob[offsets[slot]..offsets[slot+1]].
                Text(Vec<u64>, Vec<u8>),
            }
            let mut columns: Vec<GatheredColumn> = Vec::with_capacity(table.columns.len());
            for idx in 0..table.columns.len() {
                match table.columns[idx].ty {
                    gpu_db_sql::SqlType::Int8 | gpu_db_sql::SqlType::Timestamp => {
                        let base = crate::relational_model::resident_device_int8_column_offset(
                            &descriptor,
                            table,
                            idx,
                        )
                        .ok()?;
                        columns.push(GatheredColumn::I64(
                            device_memory
                                .read_resident_i32_column(base, rows * 2)
                                .ok()?,
                        ));
                    }
                    gpu_db_sql::SqlType::Bool => {
                        let base = crate::relational_model::resident_device_bool_column_offset(
                            &descriptor,
                            table,
                            idx,
                        )
                        .ok()?;
                        // ceil(rows/32) words cover the live prefix (the shard bitmap is
                        // capacity-strided; the words past `rows` map to headroom -> unread).
                        columns.push(GatheredColumn::Bool(
                            device_memory
                                .read_resident_i32_column(base, rows.div_ceil(32))
                                .ok()?,
                        ));
                    }
                    gpu_db_sql::SqlType::Numeric { .. } | gpu_db_sql::SqlType::Uuid => {
                        let base = crate::relational_model::resident_device_numeric_column_offset(
                            &descriptor,
                            table,
                            idx,
                        )
                        .ok()?;
                        // FOUR i32 words per row (16 bytes), read as the live prefix.
                        columns.push(GatheredColumn::B128(
                            device_memory
                                .read_resident_i32_column(base, rows * 4)
                                .ok()?,
                        ));
                    }
                    gpu_db_sql::SqlType::Text => {
                        let layout = crate::relational_model::resident_device_text_column_layout(
                            &descriptor,
                            table,
                            idx,
                        )
                        .ok()?;
                        // The offsets section is (rows+1) u64; the blob is `bytes_len` raw bytes.
                        let offsets = device_memory
                            .read_resident_u64_column(layout.offsets_byte_offset, rows + 1)
                            .ok()?;
                        let blob = device_memory
                            .read_resident_bytes(
                                layout.bytes_byte_offset,
                                layout.bytes_len as usize,
                            )
                            .ok()?;
                        columns.push(GatheredColumn::Text(offsets, blob));
                    }
                    _ => {
                        let base = crate::relational_model::resident_device_int4_column_offset(
                            &descriptor,
                            table,
                            idx,
                        )
                        .ok()?;
                        columns.push(GatheredColumn::I32(
                            device_memory.read_resident_i32_column(base, rows).ok()?,
                        ));
                    }
                }
            }
            let u64_at = |halves: &[i32], slot: usize| -> u64 {
                (halves[slot * 2] as u32 as u64) | ((halves[slot * 2 + 1] as u32 as u64) << 32)
            };
            for slot in 0..rows {
                let deleted_by = deleted.as_ref().map_or(u64::MAX, |h| u64_at(h, slot));
                let created_by = created.as_ref().map_or(0, |h| u64_at(h, slot));
                if !(created_by <= read_txn && read_txn < deleted_by) {
                    continue; // not visible at this snapshot (tombstoned / future version)
                }
                let row_id = u64_at(&id_halves, slot);
                if row_id == u64::MAX {
                    return None; // an UNSTAMPED live slot: identity hole -> host source
                }
                let row: Vec<SqlValue> = columns
                    .iter()
                    .zip(table.columns.iter())
                    .enumerate()
                    .map(|(col_idx, (column, catalog_column))| {
                        // ADR-006 (NULL coverage): a column with a validity bitmap whose bit for this slot is
                        // 0 is NULL — short-circuit the decode (the raw section holds a don't-care placeholder
                        // 0 / empty, exactly what the payload builder wrote). Columns without a bitmap are
                        // all-valid. Mirrors the on-device M3 read path's NULL semantics.
                        if let Some(words) = &null_bitmaps[col_idx] {
                            if (words[slot / 32] as u32 >> (slot % 32)) & 1 == 0 {
                                return Some(SqlValue::Null);
                            }
                        }
                        match column {
                            GatheredColumn::I32(vals) => {
                                sql_value_from_i32_section(catalog_column.ty, vals[slot])
                            }
                            GatheredColumn::I64(halves) => {
                                let lo = halves[slot * 2] as u32 as u64;
                                let hi = halves[slot * 2 + 1] as u32 as u64;
                                sql_value_from_i64_section(
                                    catalog_column.ty,
                                    (lo | (hi << 32)) as i64,
                                )
                            }
                            GatheredColumn::Bool(words) => {
                                let bit = (words[slot / 32] as u32 >> (slot % 32)) & 1;
                                Some(SqlValue::Bool(bit == 1))
                            }
                            GatheredColumn::B128(words) => {
                                // Reassemble the 16 LE bytes (4 u32 words) for this slot.
                                let mut bytes = [0u8; 16];
                                for w in 0..4 {
                                    bytes[w * 4..w * 4 + 4]
                                        .copy_from_slice(&words[slot * 4 + w].to_le_bytes());
                                }
                                match catalog_column.ty {
                                    gpu_db_sql::SqlType::Numeric { scale, .. } => {
                                        // numeric = i128 mantissa LE, at the column's declared scale
                                        // (byte-identical to the on-device projection's Decimal128::new).
                                        Some(SqlValue::Numeric(gpu_db_sql::Decimal128::new(
                                            i128::from_le_bytes(bytes),
                                            scale,
                                        )))
                                    }
                                    // uuid = the raw 16 bytes (storage wrote them verbatim).
                                    gpu_db_sql::SqlType::Uuid => Some(SqlValue::Uuid(bytes)),
                                    _ => None,
                                }
                            }
                            GatheredColumn::Text(offsets, blob) => {
                                // row `slot` = blob[offsets[slot]..offsets[slot+1]] as UTF-8.
                                let start = offsets[slot] as usize;
                                let end = offsets[slot + 1] as usize;
                                blob.get(start..end)
                                    .and_then(|b| std::str::from_utf8(b).ok())
                                    .map(|s| SqlValue::Text(s.to_string()))
                            }
                        }
                    })
                    .collect::<Option<Vec<SqlValue>>>()?;
                out.push((row_id, row));
            }
        }
        Some(out)
    }

    /// Synthesize a single-store-shaped [`RelationalResidencySnapshot`] DESCRIPTOR for ONE shard
    /// (S10c slice 1). A shard's SoA payload is self-contained (`count_header_byte_offset == 0`,
    /// sized by `shard.row_count`), so a descriptor whose `row_count == shard.row_count` plus the
    /// shard's `resident_device_{int4,text}_columns` makes the SINGLE-store offset helpers address the
    /// shard buffer BYTE-IDENTICALLY — letting the general resident-Expr executor serve one shard
    /// slice when handed it via `ResidentExecSource`. Mirrors the benchmark snapshot constructor (the
    /// per-table install path) field-for-field; the fields the offset helpers DON'T read (generation,
    /// stats, int8/numeric/bool/null columns, refresh cost, admission accounting) take inert defaults.
    /// The identity guard (`schema`/`table` == catalog) and `is_valid()` are satisfied for a valid
    /// shard, so the executor's per-source identity/validity prechecks pass.
    pub(crate) fn resident_snapshot_for_shard(
        &self,
        shard: &RelationalResidentShard,
        table: &RelationalTable,
    ) -> RelationalResidencySnapshot {
        RelationalResidencySnapshot {
            gpu_id: shard.gpu_id,
            schema: shard.schema.clone(),
            table: shard.table.clone(),
            generation: 0,
            // CRITICAL: the shard's own row count sizes the SoA the single-store offset helpers
            // read, so they address THIS shard's buffer (not the whole table). S-d2: an OPEN shard is
            // capacity-padded (headroom for appends), so the column STRIDE is `shard.capacity` while the
            // live row count is `shard.row_count` — exactly the single buffer's capacity/row_count split.
            row_count: shard.row_count,
            capacity: shard.capacity,
            column_count: table.columns.len(),
            resident_bytes: shard.resident_bytes,
            resident_device_int4_columns: shard.resident_device_int4_columns.clone(),
            resident_device_int4_column_stats: Vec::new(),
            // TYPE-COVERAGE track 2 slice 2: the shard's i64 section labels ride the synthesized
            // descriptor so the shared offset helpers address it (layout == single-buffer).
            resident_device_int8_columns: shard.resident_device_int8_columns.clone(),
            // TYPE-COVERAGE #14 (numeric): the shard's b128 (Numeric/Uuid) section labels ride the
            // descriptor (layout == single-buffer, so the shared 16-byte offset helper addresses it).
            resident_device_numeric_columns: shard.resident_device_numeric_columns.clone(),
            // TYPE-COVERAGE #14 (bool): the shard's per-column bool bitmaps (offsets relative to the
            // shard's buffer, which this descriptor addresses) so the executor reads bool on-device.
            resident_device_bool_columns: shard.resident_device_bool_columns.clone(),
            resident_device_text_columns: shard.resident_device_text_columns.clone(),
            // M3-for-shards: carry the shard's own per-column NULL bitmaps (offsets are relative to the
            // shard's buffer, which this descriptor addresses). Empty for the NULL-free majority.
            resident_device_null_columns: shard.resident_device_null_columns.clone(),
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: shard.invalidated_by_txn_id,
            invalidated_at_index: shard.invalidated_at_index,
            invalidated_by_memory_pressure: shard.invalidated_by_memory_pressure,
            memory_pressure_active: shard.memory_pressure_active,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: shard.device_memory_proof.clone(),
        }
    }

    /// S10c slice 2a: synthesize the single-store-shaped DESCRIPTOR for the ONE UNIFIED int4-only
    /// buffer recompacted from all of a table's shards. Like [`Self::resident_snapshot_for_shard`]
    /// but sized by the WHOLE table (`row_count == total_row_count`) so the single-store offset helpers
    /// address the unified SoA byte-identically. `int4_columns` is the shards' OWN (uniform)
    /// `resident_device_int4_columns` -- i.e. the list the unified buffer was physically recompacted from,
    /// NOT a catalog re-derivation. Labelling the descriptor with the actual buffer layout keeps the
    /// offset helper's per-read name-check load-bearing (a read of a column whose name does not sit at the
    /// labelled int4 ordinal errors instead of silently returning another column's bytes) -- audit F1.
    /// Text is deferred in this slice, so `resident_device_text_columns` is empty. The
    /// `device_memory_proof` is the unified buffer's freshly-built proof.
    pub(crate) fn resident_snapshot_for_unified(
        &self,
        table: &RelationalTable,
        parts: UnifiedResidentSnapshotParts,
    ) -> RelationalResidencySnapshot {
        let UnifiedResidentSnapshotParts {
            total_row_count,
            gpu_id,
            resident_bytes,
            proof,
            int4_columns,
            int8_columns,
            numeric_columns,
            bool_columns,
            text_columns,
            null_columns,
        } = parts;
        let resident_device_int4_columns = int4_columns;
        RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 0,
            row_count: total_row_count,
            capacity: total_row_count,
            column_count: table.columns.len(),
            resident_bytes,
            resident_device_int4_columns,
            resident_device_int4_column_stats: Vec::new(),
            resident_device_int8_columns: int8_columns,
            resident_device_numeric_columns: numeric_columns,
            resident_device_bool_columns: bool_columns,
            resident_device_text_columns: text_columns,
            resident_device_null_columns: null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: Some(proof),
        }
    }

    fn validate_benchmark_resident_chunk_columns(
        table: &RelationalTable,
        int4_columns: &[String],
        int4_stats: &[ResidentDeviceInt4ColumnStats],
        text_columns: &[ResidentDeviceTextColumnLayout],
    ) -> Result<(), ExecuteError> {
        let expected_int4 = table
            .columns
            .iter()
            .filter(|column| column.ty == SqlType::Int4)
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        if int4_columns != expected_int4.as_slice() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk int4 column layout {:?} does not match catalog int4 columns {:?}",
                int4_columns, expected_int4
            ))));
        }
        let actual_int4_stats = int4_stats
            .iter()
            .map(|stats| stats.name.clone())
            .collect::<Vec<_>>();
        if actual_int4_stats != expected_int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk int4 stats layout {:?} does not match catalog int4 columns {:?}",
                actual_int4_stats, expected_int4
            ))));
        }
        if let Some(stats) = int4_stats.iter().find(|stats| stats.min > stats.max) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk int4 stats for column \"{}\" have min greater than max",
                stats.name
            ))));
        }
        let expected_text = table
            .columns
            .iter()
            .filter(|column| column.ty == SqlType::Text)
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let actual_text = text_columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        if actual_text != expected_text {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk text column layout {:?} does not match catalog text columns {:?}",
                actual_text, expected_text
            ))));
        }
        Ok(())
    }

    fn visible_relational_row_count(&self, table: &str) -> Result<usize, ExecuteError> {
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };
        let prefix = relational_key_prefix(table);
        let table_rows = self.read_state.mvcc.table_rows(table);
        let mut cursor = table_rows.store().seq_scan_open(visibility)?;
        let mut row_count = 0usize;
        while let Some(tuple) = cursor.next() {
            if tuple.key.starts_with(&prefix) {
                row_count = row_count.checked_add(1).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "visible relational row count overflowed".to_string(),
                    ))
                })?;
            }
        }
        Ok(row_count)
    }

    fn relational_resident_bytes_for_gpu_excluding(&self, gpu_id: u16, table: &str) -> u64 {
        let snapshot_bytes: u64 = self
            .read_state
            .residency
            .snapshots
            .load()
            .iter()
            .filter(|(name, entry)| name.as_str() != table && entry.descriptor.gpu_id == gpu_id)
            .map(|(_name, entry)| {
                entry
                    .descriptor
                    .device_memory_proof
                    .as_ref()
                    .map_or(0, |proof| proof.allocated_bytes)
            })
            .sum();
        let shard_bytes: u64 = self
            .read_state
            .residency
            .shards
            .load()
            .iter()
            .filter(|(name, _shards)| name.as_str() != table)
            .flat_map(|(_name, shards)| shards)
            .filter(|shard| shard.gpu_id == gpu_id)
            .map(|shard| {
                let regions = [
                    shard.deleted_by_region.as_ref(),
                    shard.created_by_region.as_ref(),
                    shard.row_id_region.as_ref(),
                ]
                .into_iter()
                .flatten()
                .map(|region| region.metadata().allocated_bytes)
                .sum::<u64>();
                shard.allocated_bytes.saturating_add(regions)
            })
            .sum();
        let single_indexes = self
            .read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|(name, _)| name.as_str() != table)
            .filter_map(|(_, index)| index.index_memory.as_ref())
            .filter(|memory| memory.metadata().gpu_id == gpu_id)
            .map(|memory| memory.metadata().allocated_bytes)
            .sum::<u64>();
        let shard_indexes = self
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|((name, _, _), _)| name.as_str() != table)
            .filter_map(|(_, index)| index.device_index.as_ref())
            .filter(|memory| memory.metadata().gpu_id == gpu_id)
            .map(|memory| memory.metadata().allocated_bytes)
            .sum::<u64>();
        snapshot_bytes
            .saturating_add(shard_bytes)
            .saturating_add(single_indexes)
            .saturating_add(shard_indexes)
    }

    /// Actual retained allocation bytes attributable to one table on one GPU. This is the exact
    /// inverse unit used by two-phase admission: candidate selection subtracts these bytes from the
    /// same payload/region/index categories counted by `relational_resident_bytes_for_gpu_excluding`,
    /// so the chosen prefix is known to fit before any descriptor is retired.
    fn relational_resident_table_bytes_for_gpu(&self, table: &str, gpu_id: u16) -> u64 {
        let snapshot_bytes = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .filter(|entry| entry.descriptor.gpu_id == gpu_id)
            .and_then(|entry| entry.descriptor.device_memory_proof.as_ref())
            .map_or(0, |proof| proof.allocated_bytes);
        let shard_bytes = self
            .read_state
            .residency
            .shards
            .load()
            .get(table)
            .into_iter()
            .flatten()
            .filter(|shard| shard.gpu_id == gpu_id)
            .map(|shard| {
                let regions = [
                    shard.deleted_by_region.as_ref(),
                    shard.created_by_region.as_ref(),
                    shard.row_id_region.as_ref(),
                ]
                .into_iter()
                .flatten()
                .map(|region| region.metadata().allocated_bytes)
                .sum::<u64>();
                shard.allocated_bytes.saturating_add(regions)
            })
            .sum::<u64>();
        let single_index_bytes = self
            .read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(table)
            .and_then(|index| index.index_memory.as_ref())
            .filter(|memory| memory.metadata().gpu_id == gpu_id)
            .map_or(0, |memory| memory.metadata().allocated_bytes);
        let shard_index_bytes = self
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|((name, _, _), _)| name == table)
            .filter_map(|(_, index)| index.device_index.as_ref())
            .filter(|memory| memory.metadata().gpu_id == gpu_id)
            .map(|memory| memory.metadata().allocated_bytes)
            .sum::<u64>();
        snapshot_bytes
            .saturating_add(shard_bytes)
            .saturating_add(single_index_bytes)
            .saturating_add(shard_index_bytes)
    }

    pub fn relational_residency_snapshot(
        &self,
        table: &str,
    ) -> Option<RelationalResidencySnapshot> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| {
                let mut snapshot = (*entry.descriptor).clone();
                snapshot.memory_pressure_active = self
                    .router
                    .runtime()
                    .snapshot()
                    .memory_pressured_gpu_ids
                    .contains(&snapshot.gpu_id);
                snapshot
            })
    }

    pub fn relational_retained_snapshot_handle(
        &self,
        table: &str,
    ) -> Option<RelationalRetainedSnapshotHandle> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| {
                let snapshot = &entry.descriptor;
                let memory_pressure_active = self
                    .router
                    .runtime()
                    .snapshot()
                    .memory_pressured_gpu_ids
                    .contains(&snapshot.gpu_id);
                RelationalRetainedSnapshotHandle {
                    schema: snapshot.schema.clone(),
                    table: snapshot.table.clone(),
                    gpu_id: snapshot.gpu_id,
                    generation: snapshot.generation,
                    row_count: snapshot.row_count,
                    column_count: snapshot.column_count,
                    resident_bytes: snapshot.resident_bytes,
                    valid_through_index: snapshot.valid_through_index,
                    valid: snapshot.invalidated_by_txn_id.is_none()
                        && snapshot.invalidated_at_index.is_none()
                        && !snapshot.invalidated_by_memory_pressure
                        && !memory_pressure_active,
                    has_retained_device_memory: self
                        .read_state
                        .residency
                        .device_memory
                        .contains_key(table),
                    resident_device_int4_columns: snapshot.resident_device_int4_columns.clone(),
                    resident_device_text_columns: snapshot.resident_device_text_columns.clone(),
                    resident_device_null_columns: snapshot.resident_device_null_columns.clone(),
                }
            })
    }

    pub fn relational_retained_device_read_view(
        &self,
        table: &str,
    ) -> Option<CudaResidentDeviceMemoryReadView> {
        let handle = self.relational_retained_snapshot_handle(table)?;
        if !handle.valid || !handle.has_retained_device_memory {
            return None;
        }
        self.read_state
            .residency
            .device_memory
            .get(table)
            .map(|device_memory| device_memory.read_view())
    }

    /// Pin the resident snapshot metadata for `table` as an OWNED clone (Stage 3 — blocker #2). The
    /// resident-route consumers used to hold a `&` borrow of the snapshot map across the kernel launch;
    /// now the map is published behind `ArcSwap`, so this loads the published generation and clones the
    /// table's entry out. The clone is owned (no map/guard borrow held across the submission), and the
    /// consumers only read scalar fields + column layouts off it before submitting — so an owned clone
    /// is a drop-in for the former borrow with no lifetime entanglement. Cloning a single snapshot's
    /// metadata once per resident-route statement is negligible against the GPU kernel it precedes.
    /// The lightweight, Arc-shared GPU/catalog descriptor for a resident table. Readers clone the
    /// `Arc` -- a refcount bump, never row data.
    pub(crate) fn relational_residency_snapshot_ref(
        &self,
        table: &str,
    ) -> Option<Arc<RelationalResidencySnapshot>> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone())
    }

    /// The residency entry from one atomic descriptor-map load.
    pub(crate) fn relational_residency_entry(
        &self,
        table: &str,
    ) -> Option<RelationalResidencyEntry> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .cloned()
    }

    pub fn warm_relational_residency_with_policy(
        &mut self,
        policy: RelationalResidencyWarmupPolicy,
    ) -> RelationalResidencyWarmupReport {
        let policy_sets_gpu = policy.gpu_id.is_some();
        let gpu_id = policy
            .gpu_id
            .unwrap_or_else(|| self.planner.default_gpu_id());
        let policy_sets_budget = policy.budget_bytes.is_some();
        if let Some(budget_bytes) = policy.budget_bytes {
            self.set_relational_residency_budget_bytes(gpu_id, budget_bytes);
        }
        let budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let requested_tables = if policy.tables.is_empty() {
            self.ddl_catalog_mut()
                .relational_catalog
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        } else {
            policy.tables.clone()
        };
        let mut selected_tables = requested_tables.clone();
        selected_tables.sort();
        selected_tables.dedup();
        if let Some(max_table_count) = policy.max_table_count {
            selected_tables.truncate(max_table_count);
        }

        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let mut entries = Vec::new();
        for table in selected_tables {
            if memory_pressure_active {
                entries.push(RelationalResidencyWarmupEntry {
                    table,
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: format!("GPU {gpu_id} is memory pressured"),
                    resident_bytes: 0,
                    evicted_tables: Vec::new(),
                    route_decision: None,
                });
                continue;
            }
            if !self
                .ddl_catalog_mut()
                .relational_catalog
                .contains_key(&table)
            {
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: "only supported public base tables can be warmed".to_string(),
                    resident_bytes: 0,
                    evicted_tables: Vec::new(),
                    route_decision: None,
                });
                continue;
            }

            let existing = self.relational_residency_snapshot(&table);
            let existing_valid = existing
                .as_ref()
                .is_some_and(|snapshot| snapshot.is_valid());
            let existing_retained = self.read_state.residency.device_memory.contains_key(&table);
            if existing_valid && existing_retained && !policy_sets_budget && !policy_sets_gpu {
                let route_decision = self.warmup_route_readiness_decision(&table);
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::AlreadyResident,
                    reason: "resident snapshot is already valid and retained".to_string(),
                    resident_bytes: existing
                        .as_ref()
                        .map(|snapshot| snapshot.resident_bytes)
                        .unwrap_or(0),
                    evicted_tables: Vec::new(),
                    route_decision,
                });
                continue;
            }
            if existing.is_some() && !policy.refresh_invalidated && !existing_valid {
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: "resident snapshot is invalidated and refresh is disabled".to_string(),
                    resident_bytes: existing
                        .as_ref()
                        .map(|snapshot| snapshot.resident_bytes)
                        .unwrap_or(0),
                    evicted_tables: Vec::new(),
                    route_decision: self.warmup_route_readiness_decision(&table),
                });
                continue;
            }

            let refreshing = existing.is_some();
            match self.populate_relational_residency_snapshot_on_gpu(&table, gpu_id) {
                Ok(snapshot) => {
                    let route_decision = self.warmup_route_readiness_decision(&table);
                    entries.push(RelationalResidencyWarmupEntry {
                        table: table.clone(),
                        action: if refreshing {
                            RelationalResidencyWarmupAction::Refreshed
                        } else {
                            RelationalResidencyWarmupAction::Warmed
                        },
                        reason: self
                            .ddl_catalog_mut()
                            .relational_resident_cache
                            .last_decision(&table)
                            .map(|decision| decision.reason.clone())
                            .unwrap_or_else(|| "resident snapshot warmed".to_string()),
                        resident_bytes: snapshot.resident_bytes,
                        evicted_tables: snapshot.evicted_tables_on_admission,
                        route_decision,
                    });
                }
                Err(err) => {
                    entries.push(RelationalResidencyWarmupEntry {
                        table: table.clone(),
                        action: RelationalResidencyWarmupAction::Error,
                        reason: err.to_string(),
                        resident_bytes: 0,
                        evicted_tables: Vec::new(),
                        route_decision: self.warmup_route_readiness_decision(&table),
                    });
                }
            }
        }

        RelationalResidencyWarmupReport {
            gpu_id,
            budget_bytes,
            requested_tables,
            entries,
        }
    }

    pub fn maintain_relational_residency_with_policy(
        &mut self,
        policy: RelationalResidencyMaintenancePolicy,
    ) -> RelationalResidencyMaintenanceReport {
        let warmup = self.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
            gpu_id: policy.gpu_id,
            tables: policy.tables,
            max_table_count: policy.max_table_count,
            budget_bytes: policy.budget_bytes,
            refresh_invalidated: policy.refresh_invalidated,
        });
        let mut warmed_count = 0;
        let mut refreshed_count = 0;
        let mut already_resident_count = 0;
        let mut skipped_count = 0;
        let mut error_count = 0;
        let mut route_ready_tables = Vec::new();
        let mut route_blockers = Vec::new();

        for entry in &warmup.entries {
            match entry.action {
                RelationalResidencyWarmupAction::Warmed => warmed_count += 1,
                RelationalResidencyWarmupAction::Refreshed => refreshed_count += 1,
                RelationalResidencyWarmupAction::AlreadyResident => already_resident_count += 1,
                RelationalResidencyWarmupAction::Skipped => skipped_count += 1,
                RelationalResidencyWarmupAction::Error => error_count += 1,
            }

            match entry.route_decision.as_ref() {
                Some(route) if route.accepted => route_ready_tables.push(entry.table.clone()),
                Some(route) => route_blockers.push(RelationalResidencyMaintenanceBlocker {
                    table: entry.table.clone(),
                    reason: if matches!(
                        entry.action,
                        RelationalResidencyWarmupAction::Skipped
                            | RelationalResidencyWarmupAction::Error
                    ) {
                        entry.reason.clone()
                    } else {
                        route.reason.clone()
                    },
                }),
                None => route_blockers.push(RelationalResidencyMaintenanceBlocker {
                    table: entry.table.clone(),
                    reason: entry.reason.clone(),
                }),
            }
        }

        let entry_count = warmup.entries.len();
        RelationalResidencyMaintenanceReport {
            gpu_id: warmup.gpu_id,
            budget_bytes: warmup.budget_bytes,
            requested_tables: warmup.requested_tables,
            entry_count,
            warmed_count,
            refreshed_count,
            already_resident_count,
            skipped_count,
            error_count,
            route_ready_count: route_ready_tables.len(),
            route_blocked_count: route_blockers.len(),
            route_ready_tables,
            route_blockers,
            entries: warmup.entries,
        }
    }

    fn warmup_route_readiness_decision(
        &mut self,
        table: &str,
    ) -> Option<RelationalResidentRouteDecisionStatus> {
        let sql = format!("SELECT COUNT(*) FROM {table}");
        let Ok(Command::Select(select)) = parse_command(&sql) else {
            return None;
        };
        Some(self.plan_relational_resident_route(&select))
    }

    fn relational_snapshot_cache_state(
        snapshot: &RelationalResidencySnapshot,
        memory_pressure_active: bool,
    ) -> &'static str {
        if memory_pressure_active || snapshot.invalidated_by_memory_pressure {
            "InvalidatedByMemoryPressure"
        } else if snapshot.invalidated_by_txn_id.is_some()
            || snapshot.invalidated_at_index.is_some()
        {
            "Invalidated"
        } else {
            "Valid"
        }
    }

    fn resident_route_reject(
        table: &str,
        reason: impl Into<String>,
        query_shape: impl Into<String>,
    ) -> RelationalResidentRouteDecisionStatus {
        RelationalResidentRouteDecisionStatus {
            table: table.to_string(),
            gpu_id: None,
            snapshot_generation: None,
            shard_count: 0,
            accepted: false,
            reason: reason.into(),
            query_shape: query_shape.into(),
            cache_state: "Absent".to_string(),
            valid: false,
            has_retained_device_memory: false,
            estimated_rows: 0,
            resident_bytes: 0,
            budget_bytes: None,
            refresh_resident_bytes: None,
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: 0,
            d2h_bytes_estimate: 0,
            d2h_rows_estimate: 0,
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        }
    }

    pub fn plan_relational_resident_route(
        &self,
        select: &Select,
    ) -> RelationalResidentRouteDecisionStatus {
        let decision = self.plan_relational_resident_route_inner(select);
        self.read_state
            .route_telemetry
            .record_route_decision(decision.clone());
        decision
    }

    fn plan_relational_resident_route_inner(
        &self,
        select: &Select,
    ) -> RelationalResidentRouteDecisionStatus {
        // Lock-free read path (Stage 2 — blocker #1): pin the catalog snapshot for the relation-kind
        // check (the subsequent table bind pins its own; both are immutable published snapshots).
        let catalog = self.catalog_snapshot();
        if catalog.relational_views.contains_key(&select.table)
            || catalog
                .relational_materialized_views
                .contains_key(&select.table)
        {
            return Self::resident_route_reject(
                &select.table,
                "resident routing currently supports only public base tables",
                "unsupported_relation_kind",
            );
        }

        let (table, bound, _copin_s) = match self.bind_relational_select_for_execution(select) {
            Ok(bound) => bound,
            Err(err) => {
                return Self::resident_route_reject(
                    &select.table,
                    format!("unsupported select shape: {err}"),
                    "unsupported_select",
                );
            }
        };

        // Stage 3 — blocker #2: pin the published resident shard + snapshot maps for the rest of
        // the planning decision (the shard slice is passed by reference into the sharded-route
        // planner, and the snapshot is read field-by-field below — both must outlive those uses, so the
        // guards are bound here and held to the end of the function).
        let shards_guard = self.read_state.residency.shards.load();
        let snapshots_guard = self.read_state.residency.snapshots.load();

        let query_shape = match resident_route_query_shape(select, &table, &bound) {
            Some(shape) => shape,
            None => {
                if let Some(shards) = shards_guard.get(&table.name) {
                    if let Some(shape) = sharded_resident_route_query_shape(select, &table, &bound)
                    {
                        return self
                            .plan_relational_sharded_resident_route(select, &table, shape, shards);
                    }
                }
                return Self::resident_route_reject(
                    &table.name,
                    "resident routing has no retained-kernel proof for this SELECT shape",
                    "unsupported_select",
                );
            }
        };

        if let Some(shards) = shards_guard.get(&table.name) {
            return self.plan_relational_sharded_resident_route(
                select,
                &table,
                query_shape,
                shards,
            );
        }

        let Some(entry) = snapshots_guard.get(&table.name) else {
            return Self::resident_route_reject(
                &table.name,
                "relation has no resident snapshot",
                query_shape,
            );
        };
        let snapshot = &entry.descriptor;
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&snapshot.gpu_id);
        let cache_state = Self::relational_snapshot_cache_state(snapshot, memory_pressure_active);
        let valid = snapshot.invalidated_by_txn_id.is_none()
            && snapshot.invalidated_at_index.is_none()
            && !snapshot.invalidated_by_memory_pressure
            && !memory_pressure_active;
        let has_retained_device_memory = self
            .read_state
            .residency
            .device_memory
            .contains_key(&table.name);
        let d2h_bytes_estimate = resident_route_d2h_bytes_estimate(select, &query_shape, snapshot);
        let mut decision = RelationalResidentRouteDecisionStatus {
            table: table.name.clone(),
            gpu_id: Some(snapshot.gpu_id),
            snapshot_generation: Some(snapshot.generation),
            shard_count: 1,
            accepted: false,
            reason: String::new(),
            query_shape,
            cache_state: cache_state.to_string(),
            valid,
            has_retained_device_memory,
            estimated_rows: snapshot.row_count,
            resident_bytes: snapshot.resident_bytes,
            budget_bytes: snapshot.admission_budget_bytes,
            refresh_resident_bytes: snapshot
                .last_refresh_cost
                .as_ref()
                .map(|cost| cost.refreshed_resident_bytes),
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: snapshot.resident_bytes,
            d2h_bytes_estimate,
            d2h_rows_estimate: resident_route_d2h_rows_estimate(select, snapshot.row_count),
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        };

        if snapshot.schema != table.schema || snapshot.table != table.name {
            decision.reason =
                "resident snapshot no longer matches catalog table identity".to_string();
        } else if !valid {
            decision.reason = format!("resident snapshot is {cache_state}");
        } else if !has_retained_device_memory {
            decision.reason = "resident snapshot has no retained device memory".to_string();
        } else {
            decision.accepted = true;
            decision.reason = "resident route accepted".to_string();
        }
        decision
    }

    fn plan_relational_sharded_resident_route(
        &self,
        select: &Select,
        table: &RelationalTable,
        query_shape: String,
        shards: &[RelationalResidentShard],
    ) -> RelationalResidentRouteDecisionStatus {
        let total_rows = shards.iter().map(|shard| shard.row_count).sum::<usize>();
        let total_resident_bytes = shards.iter().map(|shard| shard.resident_bytes).sum::<u64>();
        let gpu_id = shards.first().map(|shard| shard.gpu_id);
        let sharded_query_shape = if query_shape == "count_all" {
            "sharded_count_all".to_string()
        } else if query_shape == "int4_equality_projection" {
            "sharded_int4_equality_projection".to_string()
        } else if query_shape == "int4_equality_multi_column_projection" {
            "sharded_int4_equality_multi_column_projection".to_string()
        } else if matches!(
            query_shape.as_str(),
            "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
        ) {
            query_shape
        } else if matches!(
            query_shape.as_str(),
            "int4_equality_count"
                | "int4_range_count"
                | "int4_between_scalar_aggregate"
                | "int4_projection"
                | "int4_projection_all"
                | "int4_composite_equality_multi_column_projection"
        ) {
            // THE FLIP audit F1: these filtered/range int4 shapes had NO sharded mapping, so the
            // now-default sharded layout demoted them to the CPU host scan (GPU-served pre-flip).
            // The `sharded_` prefix routes them to the sharded BRIDGE (their unprefixed names
            // dispatch to the single-buffer enumerated kernels), whose general executor evaluates
            // the predicate + projection/aggregate on-device over the unified (or zero-copy
            // single-shard) source.
            format!("sharded_{query_shape}")
        } else if query_shape == "int4_filter_group_count" {
            // THE FLIP (burn-in): an OR-of-int4-equalities COUNT fell to the CPU engine on a sharded
            // table (no sharded mapping — the SUM cliff's sibling). The shape keeps its single-buffer
            // name: the dispatch arm routes it to the grouped bridge, whose `src: None` now resolves
            // the sharded unified source inside `execute_resident_expr_select_with_binding`.
            query_shape
        } else if query_shape == "int4_scalar_aggregate" {
            // FLIP slice (measured): an UNFILTERED scalar aggregate (bare SUM/AVG/MIN/MAX) had NO sharded
            // mapping, so it fell through the dispatch to the CPU engine's host scan — MEASURED p50
            // 496,554us vs the bridge-served sharded COUNT's 460us at 524k rows (~1000x, a charter
            // violation in the hot path). The bridge's COUNT-precheck + general run computes scalar
            // aggregates on the unified device buffer, so route it there.
            "sharded_int4_scalar_aggregate".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Avg { .. })
        {
            "sharded_int4_filtered_avg".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Min { .. })
        {
            "sharded_int4_filtered_min".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Max { .. })
        {
            "sharded_int4_filtered_max".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate" {
            // THE FLIP audit F1 (residue): the filtered aggregates NOT covered by the tuned
            // avg/min/max mappings above (a filtered SUM) route to the sharded bridge's general
            // executor instead of falling to the CPU host scan.
            "sharded_int4_filtered_scalar_aggregate".to_string()
        } else if query_shape == "int4_distinct_projection" {
            "sharded_int4_distinct_projection".to_string()
        } else if query_shape == "int4_filtered_distinct_projection" {
            "sharded_int4_filtered_distinct_projection".to_string()
        } else if query_shape == "int4_grouped_aggregate" {
            "sharded_int4_grouped_aggregate".to_string()
        } else if query_shape == "int4_filtered_grouped_aggregate" {
            "sharded_int4_filtered_grouped_aggregate".to_string()
        } else if query_shape == "int4_ordered_projection" {
            "sharded_int4_ordered_projection".to_string()
        } else {
            query_shape
        };
        let d2h_bytes_estimate = if matches!(
            sharded_query_shape.as_str(),
            "sharded_count_all"
                | "sharded_int4_equality_projection"
                | "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
        ) {
            shards
                .len()
                .checked_mul(std::mem::size_of::<u64>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(u64::MAX)
        } else if sharded_query_shape == "sharded_int4_equality_multi_column_projection" {
            let SelectProjection::Columns(columns) = &select.projection else {
                return Self::resident_route_reject(
                    &table.name,
                    "sharded resident routing has no retained-kernel proof for this SELECT shape",
                    sharded_query_shape,
                );
            };
            u64::try_from(total_rows)
                .unwrap_or(u64::MAX)
                .saturating_mul(
                    u64::try_from(columns.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(std::mem::size_of::<i32>() as u64)
                        .saturating_add(std::mem::size_of::<u64>() as u64),
                )
                .saturating_add(
                    u64::try_from(shards.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(std::mem::size_of::<u64>() as u64),
                )
        } else {
            0
        };
        let mut decision = RelationalResidentRouteDecisionStatus {
            table: table.name.clone(),
            gpu_id,
            snapshot_generation: None,
            shard_count: shards.len(),
            accepted: false,
            reason: String::new(),
            query_shape: sharded_query_shape,
            cache_state: "Valid".to_string(),
            valid: true,
            has_retained_device_memory: false,
            estimated_rows: total_rows,
            resident_bytes: total_resident_bytes,
            budget_bytes: gpu_id.and_then(|gpu_id| self.relational_residency_budget_bytes(gpu_id)),
            refresh_resident_bytes: None,
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: total_resident_bytes,
            d2h_bytes_estimate,
            d2h_rows_estimate: resident_route_d2h_rows_estimate(select, total_rows),
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        };

        if !matches!(
            decision.query_shape.as_str(),
            "sharded_count_all"
                | "sharded_int4_scalar_aggregate"
                | "int4_filter_group_count"
                | "sharded_int4_equality_count"
                | "sharded_int4_range_count"
                | "sharded_int4_filtered_scalar_aggregate"
                | "sharded_int4_between_scalar_aggregate"
                | "sharded_int4_projection"
                | "sharded_int4_projection_all"
                | "sharded_int4_composite_equality_multi_column_projection"
                | "sharded_int4_equality_projection"
                | "sharded_int4_equality_multi_column_projection"
                | "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
                | "sharded_int4_distinct_projection"
                | "sharded_int4_filtered_distinct_projection"
                | "sharded_int4_grouped_aggregate"
                | "sharded_int4_filtered_grouped_aggregate"
                | "sharded_int4_ordered_projection"
        ) {
            decision.cache_state = "Absent".to_string();
            decision.valid = false;
            decision.reason =
                "sharded resident routing currently supports only unfiltered COUNT(*), same-column int4 equality projection, int4 equality multi-column projection, int4 equality SUM, int4 BETWEEN AVG, int4 filtered AVG, int4 filtered MIN, int4 filtered MAX, int4 [filtered] DISTINCT projection, int4 [filtered] grouped aggregate, and int4 ordered projection"
                    .to_string();
            return decision;
        }
        let mut required_int4_columns = BTreeSet::new();
        // R-ver: the UNFILTERED projection (`sharded_int4_projection_all`) needs exactly its
        // projected columns resident — no filter columns (there is no WHERE by shape definition).
        // It shares the multi-column projection's extraction (the filter loops are no-ops here).
        if decision.query_shape == "sharded_int4_equality_multi_column_projection"
            || decision.query_shape == "sharded_int4_projection_all"
        {
            match &select.projection {
                SelectProjection::Columns(columns) => {
                    for column in columns {
                        required_int4_columns.insert(column.clone());
                    }
                }
                // R-ver: `SELECT * FROM t` needs EVERY column resident (the classifier already
                // proved all columns are int4 for the `sharded_int4_projection_all` shape).
                SelectProjection::All => {
                    for column in &table.columns {
                        required_int4_columns.insert(column.name.clone());
                    }
                }
                _ => {
                    decision.cache_state = "Absent".to_string();
                    decision.valid = false;
                    decision.reason =
                        "sharded resident routing requires projected columns".to_string();
                    return decision;
                }
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_scalar_aggregate" {
            // FLIP slice: an UNFILTERED scalar aggregate (bare SUM/AVG/MIN/MAX over an int4 column —
            // the single-buffer `int4_scalar_aggregate` shape, remapped). Only the aggregate column is
            // required; there are no filters by shape definition.
            let (SelectProjection::Sum { column }
            | SelectProjection::Avg { column }
            | SelectProjection::Min { column }
            | SelectProjection::Max { column }) = &select.projection
            else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires SUM/AVG/MIN/MAX(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
        } else if decision.query_shape == "sharded_int4_equality_sum" {
            let SelectProjection::Sum { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires SUM(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_between_avg" | "sharded_int4_filtered_avg"
        ) {
            let SelectProjection::Avg { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires AVG(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_filtered_min" {
            let SelectProjection::Min { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires MIN(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_filtered_max" {
            let SelectProjection::Max { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires MAX(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_grouped_aggregate" | "sharded_int4_filtered_grouped_aggregate"
        ) {
            // Grouped int4 aggregate (S10c slice 2b): the per-shard layout check must cover both the
            // GROUP BY key column AND the aggregated value column, plus every filter column. Mirror the
            // route classifier's group/value extraction (resident_route.rs `resident_route_query_shape`):
            // GroupedCount groups by `column` and counts it; the other grouped projections carry an
            // explicit group/value column pair.
            match &select.projection {
                SelectProjection::GroupedCount { column } => {
                    required_int4_columns.insert(column.clone());
                }
                SelectProjection::GroupedSum {
                    group_column,
                    sum_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(sum_column.clone());
                }
                SelectProjection::GroupedAvg {
                    group_column,
                    avg_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(avg_column.clone());
                }
                SelectProjection::GroupedMin {
                    group_column,
                    min_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(min_column.clone());
                }
                SelectProjection::GroupedMax {
                    group_column,
                    max_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(max_column.clone());
                }
                _ => {
                    decision.cache_state = "Absent".to_string();
                    decision.valid = false;
                    decision.reason =
                        "sharded resident routing requires a grouped int4 aggregate projection"
                            .to_string();
                    return decision;
                }
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_distinct_projection"
                | "sharded_int4_filtered_distinct_projection"
                | "sharded_int4_ordered_projection"
        ) {
            // Single-column DISTINCT / ordered int4 projection (S10c slice 2b): the per-shard layout
            // check must cover the single projected/distinct column plus every filter column. The route
            // classifier accepts only a single projected column for these shapes.
            let SelectProjection::Columns(columns) = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires a single projected int4 column".to_string();
                return decision;
            };
            for column in columns {
                required_int4_columns.insert(column.clone());
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        }
        if shards.is_empty() {
            decision.cache_state = "Absent".to_string();
            decision.valid = false;
            decision.reason = "relation has no resident shards".to_string();
            return decision;
        }

        let mut has_all_device_memory = true;
        for shard in shards {
            let memory_pressure_active = self
                .router
                .runtime()
                .snapshot()
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            let valid = shard.is_valid(memory_pressure_active);
            decision.valid &= valid;
            if memory_pressure_active || shard.invalidated_by_memory_pressure {
                decision.cache_state = "InvalidatedByMemoryPressure".to_string();
            } else if shard.invalidated_by_txn_id.is_some() || shard.invalidated_at_index.is_some()
            {
                decision.cache_state = "Invalidated".to_string();
            }
            if shard.schema != table.schema || shard.table != table.name {
                decision.reason =
                    "resident shard no longer matches catalog table identity".to_string();
                return decision;
            }
            // D4: the planner's device check reads the loaded descriptor (advisory — execution
            // re-validates from its own snapshot).
            if shard.device_memory.is_none() {
                has_all_device_memory = false;
            }
            // TYPE-COVERAGE #14: a required column must sit in SOME device section the general
            // executor + recompaction gather serve — int4 / int8 / b128 (Numeric/Uuid) / bool bitmap.
            // (Filter columns are still int4 by shape definition; only PROJECTED columns can be
            // i64/b128/bool.) Text required columns are never classified here.
            if !required_int4_columns.is_empty()
                && required_int4_columns.iter().any(|column| {
                    !shard.resident_device_int4_columns.contains(column)
                        && !shard.resident_device_int8_columns.contains(column)
                        && !shard.resident_device_numeric_columns.contains(column)
                        && !shard
                            .resident_device_bool_columns
                            .iter()
                            .any(|b| &b.name == column)
                        && !shard
                            .resident_device_text_columns
                            .iter()
                            .any(|t| &t.name == column)
                })
            {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                // (Message kept as "int4 projection layout" for the p8 route tests; the check now
                // also accepts i64/b128 sections — a truly-missing column still rejects here.)
                decision.reason = format!(
                    "resident shard {} lacks required int4 projection layout",
                    shard.shard_id
                );
                return decision;
            }
        }
        decision.has_retained_device_memory = has_all_device_memory;
        if !decision.valid {
            decision.reason = format!("resident shard set is {}", decision.cache_state);
        } else if !has_all_device_memory {
            decision.reason = "resident shard set has missing retained device memory".to_string();
        } else {
            decision.accepted = true;
            decision.reason = "sharded resident route accepted".to_string();
        }
        decision
    }

    pub(crate) fn relational_residency_status(&self) -> RelationalResidencyStatus {
        // Stage 3 — blocker #2: iterate a pinned snapshot generation (the per-table `last_decision` it
        // joins to still lives on the resident cache and is read via `&self` inside the closure).
        let snapshots_guard = self.read_state.residency.snapshots.load();
        let mut tables = snapshots_guard
            .values()
            .map(|entry| {
                let snapshot = &entry.descriptor;
                let memory_pressure_active = self
                    .router
                    .runtime()
                    .snapshot()
                    .memory_pressured_gpu_ids
                    .contains(&snapshot.gpu_id);
                let last_decision = self
                    .ddl_catalog()
                    .relational_resident_cache
                    .last_decision(&snapshot.table)
                    .cloned();
                let last_decision = last_decision.as_ref();
                let cache_state =
                    Self::relational_snapshot_cache_state(snapshot, memory_pressure_active);
                RelationalResidencyTableStatus {
                    schema: snapshot.schema.clone(),
                    table: snapshot.table.clone(),
                    gpu_id: snapshot.gpu_id,
                    snapshot_generation: snapshot.generation,
                    cache_state: cache_state.to_string(),
                    row_count: snapshot.row_count,
                    column_count: snapshot.column_count,
                    resident_bytes: snapshot.resident_bytes,
                    valid_through_index: snapshot.valid_through_index,
                    valid: snapshot.invalidated_by_txn_id.is_none()
                        && snapshot.invalidated_at_index.is_none()
                        && !snapshot.invalidated_by_memory_pressure
                        && !memory_pressure_active,
                    invalidated_by_txn_id: snapshot.invalidated_by_txn_id,
                    invalidated_at_index: snapshot.invalidated_at_index,
                    invalidated_by_memory_pressure: snapshot.invalidated_by_memory_pressure,
                    memory_pressure_active,
                    admission_budget_bytes: snapshot.admission_budget_bytes,
                    resident_bytes_after_admission: snapshot.resident_bytes_after_admission,
                    evicted_tables_on_admission: snapshot.evicted_tables_on_admission.clone(),
                    last_decision_accepted: last_decision.map(|decision| decision.accepted),
                    last_decision_reason: last_decision.map(|decision| decision.reason.clone()),
                    last_decision_current_bytes_before: last_decision
                        .map(|decision| decision.current_bytes_before),
                    last_decision_current_bytes_after: last_decision
                        .map(|decision| decision.current_bytes_after),
                    device_memory_proof: snapshot.device_memory_proof.clone(),
                }
            })
            .collect::<Vec<_>>();
        tables.sort_by(|left, right| {
            left.gpu_id
                .cmp(&right.gpu_id)
                .then_with(|| left.schema.cmp(&right.schema))
                .then_with(|| left.table.cmp(&right.table))
        });

        let mut resident_bytes_by_gpu = BTreeMap::new();
        for table in &tables {
            *resident_bytes_by_gpu.entry(table.gpu_id).or_insert(0) += table.resident_bytes;
        }

        RelationalResidencyStatus {
            tables,
            latest_route_decisions: self
                .read_state
                .route_telemetry
                .route_decisions()
                .values()
                .cloned()
                .collect(),
            resident_bytes_by_gpu,
            budget_bytes_by_gpu: self
                .ddl_catalog()
                .relational_resident_cache
                .budget_bytes_by_gpu
                .clone(),
        }
    }
}
