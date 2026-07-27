//! Vacuum, serialized rehydration, and device-gather ownership.

use super::*;

impl Engine {
    /// RETIRE-002 bridge for an authoritative hot table that no longer fits its configured
    /// residency budget. Reverse-gather the exact device generation into the temporary repair
    /// store, retire the hot allocation, and let the existing streaming scan-build publish
    /// device-format cold chunks. The caller must subsequently enter chunk authority after a
    /// complete cold capture; no relational predicate or result is evaluated on the host here.
    pub(crate) fn transition_oversized_device_table_to_streaming_repair(
        &self,
        table_name: &str,
    ) -> Result<bool, EngineError> {
        let gpu_id = self.planner.default_gpu_id();
        let Some(budget) = self.relational_residency_budget_bytes(gpu_id) else {
            return Ok(false);
        };
        if budget == 0 {
            return Ok(false);
        }
        self.transition_device_table_to_streaming_repair_above(table_name, budget)
    }

    /// Query-working-set variant of the oversized transition. Operators that need substantial
    /// scratch may set a lower resident-input ceiling than the table-wide residency budget.
    pub(crate) fn transition_device_table_to_streaming_repair_above(
        &self,
        table_name: &str,
        resident_input_ceiling: u64,
    ) -> Result<bool, EngineError> {
        if !self.table_device_authoritative(table_name) {
            return Ok(false);
        }
        let gpu_id = self.planner.default_gpu_id();
        // Compare the exact retained allocation charge—not logical live-row bytes. Open-shard
        // capacity, MVCC/identity regions, and device indexes consume real VRAM and must trigger
        // the transition even when the compact live image alone would appear to fit.
        if self.relational_resident_table_bytes_for_gpu(table_name, gpu_id)
            <= resident_input_ceiling
        {
            return Ok(false);
        }

        let _commit_guard = self.commit_state();
        if !self.table_device_authoritative(table_name) {
            return Ok(false);
        }
        let boundary = self.committed_seq();
        let Some(table) = self
            .catalog_snapshot()
            .relational_catalog
            .get(table_name)
            .cloned()
        else {
            return Ok(false);
        };
        self.rehydrate_elided_table(
            &table,
            boundary,
            &Default::default(),
            &Default::default(),
            boundary,
        )?;
        self.invalidate_relational_residency_table(table_name, boundary, boundary);
        self.read_state.residency.with_snapshots_mut(|snapshots| {
            snapshots.remove(table_name);
        });
        self.read_state
            .residency
            .with_shards_mut_for_table(table_name, |shards| {
                shards.remove(table_name);
            });
        Ok(true)
    }

    /// R3-003 bounded device-version GC. A created_by sidecar is redundant once the oldest active
    /// read boundary is at or beyond that shard's stamp high-water: every possible reader sees all
    /// its versions as born-visible, and new transactions begin no earlier. Republish current
    /// descriptors without the sidecar and release the write-side owner; captured generations keep
    /// their own Arc until their transaction deregisters. Deleted_by cannot be dropped in place
    /// (that would resurrect dead slots) and remains owned by thresholded dense VACUUM.
    pub(crate) fn gc_transaction_created_by_regions(&self) -> usize {
        // A pre-WAL fixed INSERT plan may bind an existing created_by Arc across canonical
        // WAL/status buffering and device apply before physical group durability. Every normal
        // publisher takes this same gate, so GC must join it rather than rely on each caller to
        // retain the commit mutex. The lane leader already owns the gate; all other callers acquire
        // it here before touching either descriptor or write-side sidecar ownership.
        let apply_leader =
            crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(std::cell::Cell::get);
        let _device_apply = if apply_leader {
            None
        } else {
            Some(
                self.read_state
                    .residency
                    .mutation_gate
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            )
        };
        let safe_boundary = self
            .active_snapshots_oldest()
            .unwrap_or_else(|| self.committed_seq());
        let removable = self
            .read_state
            .residency
            .shards
            .load()
            .iter()
            .flat_map(|(table, shards)| {
                shards.iter().filter_map(|shard| {
                    (shard.created_by_region.is_some() && shard.max_created_by <= safe_boundary)
                        .then_some((table.clone(), shard.shard_id))
                })
            })
            .collect::<Vec<_>>();
        if removable.is_empty() {
            return 0;
        }
        let removable_set = removable.iter().cloned().collect::<BTreeSet<_>>();
        self.read_state.residency.with_shards_mut(|shards| {
            for (table, table_shards) in shards {
                for shard in table_shards {
                    if removable_set.contains(&(table.clone(), shard.shard_id)) {
                        shard.created_by_region = None;
                        shard.max_created_by = 0;
                    }
                }
            }
        });
        for (table, shard_id) in &removable {
            self.read_state
                .residency
                .shard_created_by_memory
                .invalidate_shard(table, *shard_id);
        }
        removable.len()
    }

    /// RETIRE-002 repair entry for an off-commit-lock caller. It reverse-gathers the authoritative
    /// device generation into the temporary DDL/vacuum/recovery representation under the commit
    /// lock. Normal DML and production reads never call this as a fallback. Mid-commit repair work
    /// already holds the lock, so the thread-local internal-read marker selects the direct arm and
    /// avoids a non-reentrant acquisition.
    /// VACUUM #5 (A5 gate): REBUILD a churned table's residency DENSE + ALL-LIVE — reclaims
    /// tombstoned slots and stale duplicate physical keys (an SV5/A4b update-append leaves the
    /// old version's slot holding the key, which dup-declines the per-shard PK index until a
    /// rebuild changes the buffer ptr — the monotone decline clears BY DESIGN on a new
    /// generation). Composition of audited pieces: an authoritative table reverse-gathers its
    /// device truth across the explicit repair boundary, then invalidate + rebuild publishes a
    /// dense replacement. Runs under the COMMIT LOCK (the same discipline as
    /// `rehydrate_elided_serialized`; the
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
            let current = engine.committed_seq();
            // Dense VACUUM reconstructs only the current live image. An older retained writer
            // still needs post-BEGIN claim/release versions for commit-time conflict detection,
            // so it fences the rebuild exactly like an old reader fences MVCC reclamation. Equal
            // boundaries are safe: no history newer than that snapshot exists under this lock.
            if engine
                .active_snapshots_oldest()
                .is_some_and(|oldest| oldest < current)
            {
                return Ok(());
            }
            if engine.table_device_authoritative(table_name) {
                engine.rehydrate_elided_table(
                    &table,
                    current,
                    &Default::default(),
                    &Default::default(),
                    current,
                )?;
            }
            let tables: std::collections::BTreeSet<String> =
                std::iter::once(table_name.to_string()).collect();
            engine.invalidate_relational_residency_tables_concurrent(&tables, current, current);
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
            if !engine.table_device_authoritative(table_name) {
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

    /// RETIRE-002 reverse-gather core. It reconstructs the temporary host repair representation
    /// from one device generation and then clears device authority. `upserts`/`removals` are kept
    /// for representation-changing repair callers; normal DML must publish in place or fail stop
    /// before acknowledgement. A gather decline is an invariant failure, never permission to use
    /// stale host tuples.
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
        self.set_table_device_authoritative(&table.name, false);
        Ok(())
    }

    /// RETIRE-002: gather a shard-resident table's VISIBLE rows + identities ENTIRELY FROM
    /// THE DEVICE for an explicit DDL/vacuum/recovery repair transition. Per shard: one bulk
    /// DtoH per int4 column + the row_id/deleted_by/created_by regions, then the host-side
    /// SV3b/SV6 visibility filter (`created_by <= read_txn < deleted_by`) — an amortized-once
    /// control-plane readback (the DATA SOURCE is the device generation, not host tuples); the
    /// device-to-device recompaction that avoids the round-trip is the ledgered follow-up.
    /// Returns rows in (shard, slot) order with their identities. `None` means the repair cannot
    /// prove a complete device source and must fail loudly; it is never a host-fallback signal.
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
        // TYPE-COVERAGE #14 (bool/numeric/uuid): these also gather in this RETIRE-002 repair-only
        // path. Normal SELECT execution never rehydrates for an unsupported device shape; it fails
        // loudly instead. Bool = 1 bit/row bitmap; Numeric/Uuid = the 16-byte b128
        // section (numeric = i128 mantissa LE at the catalog scale; uuid = the raw 16 bytes); Text = the
        // offsets section + bytes blob. This repair path covers every elision-eligible type.
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
                    return None; // an UNSTAMPED live slot: identity hole makes repair incomplete -> fail loudly
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
}
