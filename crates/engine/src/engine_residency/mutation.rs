//! Resident append, rollover, sparse-version stamping, and fused-apply ownership.

use super::*;

impl Engine {
    /// Slice 1b-ii: append an INSERT's APPLIED rows IN PLACE into the table's resident OPEN shard's
    /// capacity headroom, instead of a full re-admit. `new_rows` MUST be the post-coercion/post-default
    /// applied images (the WriteDelta's `PreparedMutation::Insert.inserted_rows`), in catalog order, so
    /// the appended bytes match what a full rebuild would store. Returns `true` iff it appended +
    /// republished; `false` (the caller MUST fall back to invalidate + re-admit) when the table is not
    /// purely-int4-resident, is invalidated, lacks headroom, or the device append fails. MUST run BEFORE
    /// `committed_seq` is published, so a reader at the new commit observes the advanced `row_count`
    /// (visibility ordering — same placement as the invalidation it replaces).
    pub(crate) fn try_append_resident_int4_open_shard(
        &self,
        table: &str,
        new_rows: &[Vec<SqlValue>],
        // RETIREMENT A1: `row_ids` = the appended rows' host identities (parsed from the commit's
        // write-set keys; an UPDATE append passes the ORIGINAL row's id). `None` = unknown (the
        // benchmark/synthetic paths): existing regions stay sentinel at those slots and no region
        // is created on rollover — identity-unknown, the device resolve declines.
        // D3 (ADR-013 pre1, STAMP-ALL-APPENDS): every append carries its birth commit seq(s) — the
        // sharded (default) layout stamps `created_by` for INSERT and UPDATE alike, so a reader
        // pinned at `s < commit_seq` no longer sees a decided-but-unpublished append (the former
        // "born-visible" premature-insert anomaly). See [`AppendCreatedBy`] for the variants
        // (uniform / per-row / update-new-version) and the single-buffer kill-switch scoping.
        created_by: AppendCreatedBy<'_>,
        row_ids: Option<&[u64]>,
    ) -> bool {
        if new_rows.is_empty() {
            return false;
        }
        debug_assert!(
            row_ids.is_none_or(|ids| ids.len() == new_rows.len()),
            "row_ids must parallel new_rows"
        );
        // S-d2b: a SHARD-resident table (the segmented layout, default-OFF flag) appends to its OPEN shard's
        // headroom instead of the single buffer. (Admission publishes a table to shards XOR snapshots, so
        // the two paths never overlap for one table.) ADR-006 (NULL coverage): the shard path now handles a
        // NULL in an appended row by rolling a DENSE shard whose validity bitmaps the payload builder
        // constructs (like TEXT), so a NULL insert STAYS ELIDED instead of de-eliding via re-admit.
        if self
            .read_state
            .residency
            .shards
            .load()
            .get(table)
            .is_some_and(|shards| !shards.is_empty())
        {
            return self.try_append_to_resident_open_shard(table, new_rows, created_by, row_ids);
        }
        // SINGLE-BUFFER path only: a NULL in an appended row would need a validity bitmap, but the single
        // buffer this path appends into is bitmap-free by construction (the `purely_int4` eligibility below
        // requires `resident_device_null_columns.is_empty()`) and this append writes a NULL int4 as a
        // placeholder 0 WITHOUT a bitmap. The device aggregate / DISTINCT / GROUP BY routes derive NULL-ness
        // solely from the bitmap, so an appended NULL would read as a phantom 0. Decline -> the caller
        // re-admits, which BUILDS the correct bitmap. (The SHARD path above now maintains bitmaps on the
        // rollover, so it no longer declines here; only the single-buffer kill-switch layout does.)
        if new_rows
            .iter()
            .any(|row| row.iter().any(|v| matches!(v, SqlValue::Null)))
        {
            return false;
        }
        // SV6 defensive: an UPDATE-appended NEW VERSION must be stamped + hidden from older readers,
        // and the single unified buffer carries no per-row version regions — decline and let the
        // caller re-admit (always correct). Unreachable today: the SV5 UPDATE route requires shard
        // residency. An INSERT append proceeds UNSTAMPED here: the single-buffer layout is the
        // kill-switch configuration outside the ADR-013/A5 gate (no region machinery); its
        // born-visible INSERT semantics are documented pre-D3 behavior.
        if matches!(created_by, AppendCreatedBy::UpdateNewVersion(_)) {
            return false;
        }
        let Some(valid_through_index) = created_by
            .stamps_for(new_rows.len())
            .and_then(|stamps| stamps.into_iter().max())
        else {
            return false;
        };
        let (capacity, row_start, column_count) = {
            let snapshots = self.read_state.residency.snapshots.load();
            let Some(entry) = snapshots.get(table) else {
                return false;
            };
            let s = &entry.descriptor;
            // Purely int4-resident: every column rides the i32 section (no other typed sections), so the
            // int4 append op covers the whole row. (Date/Int2 ride i32 too — handled by the op.)
            let purely_int4 = s.resident_device_int8_columns.is_empty()
                && s.resident_device_numeric_columns.is_empty()
                && s.resident_device_bool_columns.is_empty()
                && s.resident_device_text_columns.is_empty()
                && s.resident_device_null_columns.is_empty()
                && s.column_count == s.resident_device_int4_columns.len();
            if !s.is_valid() || !purely_int4 {
                return false;
            }
            match s.row_count.checked_add(new_rows.len()) {
                Some(end) if end <= s.capacity => (s.capacity, s.row_count, s.column_count),
                _ => return false, // no headroom (or overflow) -> caller re-admits (with fresh headroom)
            }
        };
        let Some(device_memory) = self.read_state.residency.device_memory.get(table) else {
            return false;
        };
        // The append op reads each value's SqlValue variant (Int4/Date/Int2) for encoding; the column
        // TYPES only gate eligibility + count, and a purely-int4 table is all-i32-section by definition.
        let column_types = vec![SqlType::Int4; column_count];
        let chunks = match compute_open_shard_int4_append_chunks(
            &column_types,
            capacity,
            row_start,
            new_rows,
        ) {
            Ok(chunks) => chunks,
            Err(_) => return false,
        };
        if device_memory.append_owned_chunks(chunks).is_err() {
            // A partial/failed append leaves bytes only in the (still-invisible) headroom beyond
            // row_count; returning false makes the caller invalidate + re-admit, discarding them.
            return false;
        }
        let k = new_rows.len();
        let appended_bytes = (k * column_count * std::mem::size_of::<i32>()) as u64;
        self.read_state.residency.with_snapshots_mut(|snapshots| {
            if let Some(entry) = snapshots.get_mut(table) {
                let desc = std::sync::Arc::make_mut(&mut entry.descriptor);
                desc.generation = desc.generation.saturating_add(1);
                desc.row_count += k;
                desc.resident_bytes = desc.resident_bytes.saturating_add(appended_bytes);
                desc.valid_through_index = desc.valid_through_index.max(valid_through_index);
            }
        });
        // Slice 1b-ii (audit Finding A): the wave/lpb GPU index cache (engine_retained_read.rs) validates a
        // cached entry by (column, resident_device_ptr) ONLY — it is BLIND to generation/row_count. An
        // in-place append keeps the SAME device_ptr, so a cached index built over [0, old_row_count) would
        // be a stale HIT that reports the just-appended keys as not-found (a lost-from-reads committed
        // INSERT). Drop the table's entry so the next probe rebuilds over the new row_count. This runs
        // before the caller's publish_committed_seq, so a reader that observes the new committed_seq can
        // never bind the stale index. (In-flight probes pinned the prior index's own Arc; removing the
        // map entry only prevents NEW binds — the buffer frees once no submission holds it.)
        self.read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(table);
        self.read_state
            .residency
            .open_shard_append_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// S-d2b: append a committed INSERT's applied rows IN PLACE into the resident table's OPEN shard's
    /// headroom (the last shard in `residency.shards`) — the shard-path analog of the single-buffer append.
    /// Empty + NULL-bearing rows are already rejected by the caller. Returns false (caller invalidates +
    /// re-admits) when the open shard isn't int4-appendable, is invalid, or has no headroom (seal + a fresh
    /// open shard on overflow is S-d2c), or the device append fails. Shard tables read via device
    /// recompaction and have no single-buffer `wave_index` to drop. The device PK-index cache is
    /// `(ptr,row_count)`-validated, so an in-place append makes the next probe rebuild or extend it.
    fn try_append_to_resident_open_shard(
        &self,
        table: &str,
        new_rows: &[Vec<SqlValue>],
        created_by: AppendCreatedBy<'_>,
        row_ids: Option<&[u64]>,
    ) -> bool {
        // D3: materialize one birth stamp per appended row (validated len) — the in-place branch
        // stamps them into the open shard's created_by region and the rollover branch bakes them
        // into the new shard's region; both bump the descriptor's max_created_by high-water.
        let Some(stamps) = created_by.stamps_for(new_rows.len()) else {
            return false;
        };
        let stamps_max = stamps.iter().copied().max().unwrap_or(0);
        let pressured_gpus = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .clone();
        let k = new_rows.len();
        // ADR-006 (NULL coverage): does this appended BATCH carry any NULL? A null-bearing batch cannot
        // append in place (the in-place chunk encoder has no validity-bitmap channel); it rolls a DENSE
        // shard whose bitmaps the payload builder constructs — exactly like a text column.
        let batch_has_null = new_rows
            .iter()
            .any(|row| row.iter().any(|v| matches!(v, SqlValue::Null)));
        // Read the OPEN (last) shard's state once.
        let (
            shard_id,
            capacity,
            row_count,
            row_start,
            shard_int4_names,
            shard_int8_names,
            shard_numeric_names,
            shard_bool_layouts,
            shard_text_layouts,
            shard_null_layouts,
            gpu_id,
            schema,
            max_shard_id,
        ) = {
            let shards = self.read_state.residency.shards.load();
            let Some(table_shards) = shards.get(table) else {
                return false;
            };
            let Some(open) = table_shards.last() else {
                return false;
            };
            if !open.int4_appendable || !open.is_valid(pressured_gpus.contains(&open.gpu_id)) {
                return false;
            }
            (
                open.shard_id,
                open.capacity,
                open.row_count,
                open.row_start,
                open.resident_device_int4_columns.clone(),
                open.resident_device_int8_columns.clone(),
                open.resident_device_numeric_columns.clone(),
                open.resident_device_bool_columns.clone(),
                open.resident_device_text_columns.clone(),
                open.resident_device_null_columns.clone(),
                open.gpu_id,
                open.schema.clone(),
                table_shards.iter().map(|s| s.shard_id).max().unwrap_or(0),
            )
        };
        // TYPE-COVERAGE track 2 slice 2 stage (ii): CATALOG-ordered names/types drive the
        // section-aware chunk encoder + the rollover payload (mixed i32/i64 sections —
        // catalog order != section ordinal). Defensive arity guard: the shard's section
        // lists must cover the catalog exactly, else decline to the re-admit oracle.
        let Some(catalog_table) = self
            .catalog_snapshot()
            .relational_catalog
            .get(table)
            .cloned()
        else {
            return false;
        };
        let column_names: Vec<String> = catalog_table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_types: Vec<SqlType> = catalog_table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect();
        let column_count = column_types.len();
        if shard_int4_names.len()
            + shard_int8_names.len()
            + shard_numeric_names.len()
            + shard_bool_layouts.len()
            + shard_text_layouts.len()
            != column_count
        {
            return false;
        }
        // TYPE-COVERAGE #14 (text): a text-bearing shard is DENSE (no headroom) -> it never appends in
        // place; force the ROLLOVER branch (the in-place path's chunk encoder rejects text anyway).
        let has_text = !shard_text_layouts.is_empty();
        // ADR-006 (NULL coverage): a NULL-bearing shard is likewise DENSE (rolled with validity bitmaps), so
        // it never appends in place; and a null-carrying BATCH must roll a fresh bitmap-bearing dense shard
        // (the in-place chunk encoder has no validity-bitmap channel — it would write a NULL as a phantom 0).
        let has_null_shard = !shard_null_layouts.is_empty();
        let num_i32_cols = shard_int4_names.len();
        let num_i64_cols = shard_int8_names.len();
        let num_numeric_cols = shard_numeric_names.len();
        let num_bool_cols = shard_bool_layouts.len();

        // FITS the open shard's headroom -> append IN PLACE (1b-ii on the shard path). Text / null-bearing
        // shards never qualify (dense: row_count == capacity), and a null-carrying batch is excluded so it
        // takes the bitmap-building rollover; gate explicitly so the intent is clear.
        if !has_text
            && !has_null_shard
            && !batch_has_null
            && row_count.checked_add(k).is_some_and(|end| end <= capacity)
        {
            let Some(shard_device_memory) = self
                .read_state
                .residency
                .shard_device_memory
                .get(&(table.to_string(), shard_id))
            else {
                return false;
            };
            // The append position within THIS shard's buffer is its LOCAL row_count (rows [0, row_count)
            // are live; the new rows go at [row_count, row_count+k)), NOT the shard's global `row_start`.
            let chunks = match compute_open_shard_int4_append_chunks(
                &column_types,
                capacity,
                row_count,
                new_rows,
            ) {
                Ok(chunks) => chunks,
                Err(_) => return false,
            };
            // Column values in catalog order (used by the fused pass, the host PK-index cache
            // extension, and the device index maintenance below). NULL-free by the appendable
            // guard, so `sql_value_as_int4` yields exactly the bytes the chunks encode for the
            // i32 columns.
            let column_values: Vec<Vec<i32>> = (0..column_count)
                .map(|c| {
                    new_rows
                        .iter()
                        .map(|row| sql_value_as_int4(&row[c]))
                        .collect()
                })
                .collect();
            // E2.5c 2M+ push (b): the FUSED merged-apply device pass — column scatter +
            // created_by/row-id stamps + PK index insert in ONE staging HtoD + ONE launch
            // (replacing the ~8 driver calls of the unfused chain below). Int4-only shards
            // (the covered-INSERT shape); ineligible falls through to the unfused sequence,
            // byte-identical to before the flag.
            let fused = if self.fused_apply_enabled()
                && num_i64_cols == 0
                && num_numeric_cols == 0
                && num_bool_cols == 0
            {
                let append_started =
                    crate::engine_dml_concurrent::wave_device_phase_timing_enabled()
                        .then(std::time::Instant::now);
                // Per-COLUMN offsets only: the encoder's FINAL chunk is the device
                // row-count header (offset 0), which the fused submit publishes after fencing the kernel.
                let chunk_offsets: Vec<u64> = chunks
                    .iter()
                    .take(column_count)
                    .map(|chunk| chunk.byte_offset)
                    .collect();
                let outcome = self.try_fused_apply_in_place(
                    table,
                    shard_id,
                    &shard_device_memory,
                    &chunk_offsets,
                    &column_values,
                    row_count,
                    capacity,
                    gpu_id,
                    &stamps,
                    row_ids,
                );
                if let Some(started) = append_started {
                    crate::engine_dml_concurrent::WAVE_DEVICE_STATS[1].fetch_add(
                        started.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                match outcome {
                    Some(true) => true,
                    // Device failure mid-pass: bytes live only in invisible headroom beyond
                    // row_count; the caller invalidates + re-admits (same contract as the
                    // unfused arm's partial-failure rule).
                    Some(false) => return false,
                    None => false, // not eligible -> unfused sequence
                }
            } else {
                false
            };
            if !fused {
                // `deleted_by` needs no write on append — the headroom was pre-filled with the live sentinel at
                // admission, so appended rows are born live. SV6: an UPDATE-appended NEW VERSION additionally
                // stamps `created_by = commit_seq` (below); a plain INSERT append stays unstamped (born-visible).
                let append_started =
                    crate::engine_dml_concurrent::wave_device_phase_timing_enabled()
                        .then(std::time::Instant::now);
                let append_result = shard_device_memory.append_owned_chunks(chunks);
                if let Some(started) = append_started {
                    crate::engine_dml_concurrent::WAVE_DEVICE_STATS[1].fetch_add(
                        started.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                if append_result.is_err() {
                    // Partial/failed append leaves bytes only in invisible headroom beyond row_count;
                    // returning false makes the caller invalidate + re-admit, discarding them.
                    return false;
                }
                // TYPE-COVERAGE #14 (bool): the chunk encoder emits NO bytes for bool columns (a bitmap
                // is not a capacity-strided fixed-width chunk), so set the k appended rows' value bits
                // here via the device atomicOr op — writing into the pre-zeroed bitmap headroom at the
                // shard's LOCAL row_count. Same before-the-`row_count`-bump ordering as the version
                // stamps: the slots are still invisible headroom, so a torn (bits written, count not
                // bumped) state is unreadable, and a failure -> false -> re-admit (rebuild is truthful).
                for layout in &shard_bool_layouts {
                    let Some(col_idx) = column_names.iter().position(|n| n == &layout.name) else {
                        return false; // shard/catalog bool label mismatch -> decline to the oracle
                    };
                    let values: Vec<u8> = new_rows
                        .iter()
                        .map(|row| match row[col_idx] {
                            SqlValue::Bool(true) => 1u8,
                            // false / NULL leave the bit 0 (NULL-free by the appendable guard anyway;
                            // the validity bitmap, absent here, would decide a real NULL).
                            _ => 0u8,
                        })
                        .collect();
                    if shard_device_memory
                        .set_bool_bitmap_range(layout.bitmap_byte_offset, row_count as u32, &values)
                        .is_err()
                    {
                        return false;
                    }
                }
                // SV6 ORDER (load-bearing): stamp created_by BEFORE the `row_count` bump below publishes the
                // appended slots. The slots are still invisible headroom here, so a torn state (values + stamps
                // written, count not bumped) is unreadable; stamping AFTER the bump would let a reader bound to
                // an older snapshot observe the new version born-visible (created_by = fill 0) — exactly the
                // SV5 P2 double-read window this gate closes. A stamp failure -> false -> the caller re-admits
                // (the re-admit purge releases any partial region; rebuild-all-live is always correct).
                if !self.stamp_created_by_resident_shard_slots(
                    table, shard_id, row_count, capacity, gpu_id, &stamps,
                ) {
                    return false;
                }
                // RETIREMENT A1: stamp the appended slots' host identities (get-or-skip: a region-less
                // benchmark lineage skips; an identity-bearing shard gets exact stamps). Same
                // before-the-bump ordering as the version stamps.
                if let Some(ids) = row_ids {
                    if !self.stamp_row_id_resident_shard_slots(table, shard_id, row_count, ids) {
                        return false;
                    }
                }
            }
            let appended_bytes = (k
                * (num_i32_cols * std::mem::size_of::<i32>()
                    + num_i64_cols * std::mem::size_of::<i64>()
                    + num_numeric_cols * 16)) as u64;
            // S-d3: extend the open shard's zone map (min/max per int4 column) to cover the appended
            // rows. The stats vector is INT4-ORDINAL-aligned, so iterate only the i32-section catalog
            // columns, in order (stage ii: i64 columns carry no zone map — they simply never prune).
            let new_min_max: Vec<(i32, i32)> = (0..column_count)
                .filter(|&c| {
                    matches!(
                        column_types[c],
                        SqlType::Int4 | SqlType::Date | SqlType::Int2
                    )
                })
                .map(|c| {
                    new_rows.iter().fold((i32::MAX, i32::MIN), |(lo, hi), row| {
                        let v = sql_value_as_int4(&row[c]);
                        (lo.min(v), hi.max(v))
                    })
                })
                .collect();
            // M1 (ledger #24): incrementally maintain the DEVICE PK index too (the index_insert
            // kernel), so the wave-batched device locate never triggers the O(rows) rebuild.
            // Only fires when a device index is cached; no-op otherwise.
            if !fused {
                let idx_started = crate::engine_dml_concurrent::wave_device_phase_timing_enabled()
                    .then(std::time::Instant::now);
                self.extend_shard_pk_device_index_on_append(
                    table,
                    shard_id,
                    shard_device_memory.device_ptr(),
                    row_count,
                    &column_values,
                    new_rows,
                );
                if let Some(started) = idx_started {
                    crate::engine_dml_concurrent::WAVE_DEVICE_STATS[2].fetch_add(
                        started.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            }
            self.read_state.residency.with_shards_mut(|shards| {
                if let Some(table_shards) = shards.get_mut(table) {
                    if let Some(open) = table_shards.last_mut() {
                        open.row_count += k;
                        // D3: the high-water publishes WITH the row_count that exposes the slots —
                        // a reader at s >= hwm treats the shard as effectively version-free.
                        open.max_created_by = open.max_created_by.max(stamps_max);
                        open.resident_bytes = open.resident_bytes.saturating_add(appended_bytes);
                        for (stat, (lo, hi)) in open
                            .resident_device_int4_column_stats
                            .iter_mut()
                            .zip(new_min_max.iter())
                        {
                            stat.min = stat.min.min(*lo);
                            stat.max = stat.max.max(*hi);
                        }
                    }
                }
            });
            self.read_state
                .residency
                .open_shard_append_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }

        // S-d2c ROLLOVER: the open shard is full -> SEAL it (leave it in place, immutable) and build + install
        // a NEW open shard holding the k rows (capacity = the target, so it grows to the target before the
        // next rollover). O(rows appended), NOT the O(table) re-admit -> this is what removes the 536M cap.
        // TYPE-COVERAGE #14 (text): a text column has NO capacity-strided headroom (the builder rejects
        // capacity > row_count for text), so a text-bearing rollover shard is DENSE (capacity == k) — it
        // never appends in place; the NEXT commit rolls another dense shard. Fixed-width/bool rollovers
        // keep the growth headroom.
        let has_text = column_types.iter().any(|ty| matches!(ty, SqlType::Text));
        // ADR-006 (NULL coverage): a null-carrying batch rolls a DENSE shard (capacity == k, like text) so
        // the validity bitmaps are exact for the live rows and the shard never appends in place afterward
        // (which would need in-place bitmap maintenance). A fixed-width null-FREE batch keeps growth headroom.
        let new_capacity = if has_text || batch_has_null {
            k
        } else {
            self.shard_size_target()
                .max(k.saturating_mul(2).next_power_of_two())
        };
        let (device_payload, text_layouts, bool_layouts, int4_stats, null_layouts) =
            match build_relational_device_payload_with_capacity(
                &column_names,
                &column_types,
                new_rows,
                new_capacity,
            ) {
                // Keep the columnar payload + the int4 zone-map stats (min/max over the k rows) for pruning
                // (S-d3). TYPE-COVERAGE #14: the bool bitmap AND text (offsets+blob) layouts (offsets into
                // THIS payload) carry. ADR-006 (NULL coverage): the VALIDITY BITMAP layouts (`null_cols`,
                // offsets into THIS payload) now ALSO carry, so a null-carrying rollover STAYS ELIDED with
                // correct NULL-ness (was discarded, forcing a de-elide via re-admit). Null-free batch -> empty.
                Ok((payload, text_cols, bool_cols, stats, _b128, null_cols)) => {
                    (payload, text_cols, bool_cols, stats, null_cols)
                }
                Err(_) => return false,
            };
        // R-1 / S-F: rollover is an admission event too. Account the new payload plus its
        // mandatory created_by region and optional row-identity region before allocating any of
        // them. If it would cross the configured/default GPU budget, decline atomically; the
        // caller invalidates this table and the normal admission path may evict an older table or
        // leave this relation to the bounded streaming executor. The commit remains durable.
        let rollover_bytes = (device_payload.len() as u64)
            .saturating_add((new_capacity as u64).saturating_mul(8))
            .saturating_add(if row_ids.is_some() {
                (new_capacity as u64).saturating_mul(8)
            } else {
                0
            });
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(rollover_bytes)
                    > budget
            })
        {
            self.read_state
                .residency
                .rollover_budget_declines
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        // SV1/SV2: the rolled shard carries NO version metadata in its payload — `created_by` is gone and
        // `deleted_by` is on-demand (allocated in `shard_deleted_by_memory` on the shard's first delete).
        let Some(new_device_memory) = self
            .relational_residency_device_memory(gpu_id, &device_payload)
            .map(Arc::new)
        else {
            return false;
        };
        let new_shard_id = max_shard_id.saturating_add(1);
        let pressured = pressured_gpus.contains(&gpu_id);
        // D4 (ADR-013 pre2): build the regions BEFORE the descriptor literal so their Arcs ride the
        // published descriptor — the map inserts below keep the same Arcs for write-side bookkeeping.
        // SV6: a version-stamped (UPDATE-appended) rollover's created_by region must be observable
        // with the shard itself; carrying it IN the descriptor makes that atomic by construction.
        // D3: every rolled shard's first `k` slots carry their birth stamps; the headroom keeps the
        // born-visible fill (0) and later appends stamp into it. The region costs 8B/slot on the
        // OPEN shard lineage only (bulk-admitted shards stay region-free, hwm 0); reclaiming sealed
        // shards' regions once hwm falls below every active reader is VACUUM's job (ledger #5).
        let rolled_created_by_region = {
            let mut created_payload =
                vec![CREATED_BY_VISIBLE_FILL_BYTE; new_capacity * std::mem::size_of::<u64>()];
            for (slot, stamp) in stamps.iter().enumerate() {
                created_payload[slot * 8..slot * 8 + 8].copy_from_slice(&stamp.to_le_bytes());
            }
            let Some(created_region) =
                self.relational_residency_device_memory(gpu_id, &created_payload)
            else {
                return false;
            };
            Some(Arc::new(created_region))
        };
        let rolled_row_id_region = if let Some(ids) = &row_ids {
            let mut payload =
                vec![ROW_ID_UNSTAMPED_FILL_BYTE; new_capacity * std::mem::size_of::<u64>()];
            for (slot, row_id) in ids.iter().enumerate() {
                payload[slot * 8..slot * 8 + 8].copy_from_slice(&row_id.to_le_bytes());
            }
            self.relational_residency_device_memory(gpu_id, &payload)
                .map(Arc::new)
        } else {
            None
        };
        if row_ids.is_some() && rolled_row_id_region.is_none() {
            return false;
        }
        let rollover_allocated_bytes = new_device_memory
            .metadata()
            .allocated_bytes
            .saturating_add(
                rolled_created_by_region
                    .as_ref()
                    .map_or(0, |region| region.metadata().allocated_bytes),
            )
            .saturating_add(
                rolled_row_id_region
                    .as_ref()
                    .map_or(0, |region| region.metadata().allocated_bytes),
            );
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(rollover_allocated_bytes)
                    > budget
            })
        {
            self.read_state
                .residency
                .rollover_budget_declines
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        let new_shard = RelationalResidentShard {
            shard_id: new_shard_id,
            row_start: row_start.saturating_add(row_count),
            row_count: k,
            // A rollover is additive: it does not replace or compact any older shard history.
            history_floor_index: 0,
            capacity: new_capacity,
            int4_appendable: true,
            resident_device_int4_column_stats: int4_stats,
            // Audit NOTE adopted: count i64 columns at 8 bytes + b128 (Numeric/Uuid) columns at
            // 16 bytes (was a telemetry undercount vs the admit path; allocated_bytes was always
            // correct).
            resident_bytes: (8
                + k * (num_i32_cols * std::mem::size_of::<i32>()
                    + num_i64_cols * std::mem::size_of::<i64>()
                    + num_numeric_cols * 16)
                + bool_layouts.len() * k.div_ceil(32) * 4
                // ADR-006 (NULL coverage): each validity bitmap is ceil(k/32) u32 words.
                + null_layouts.len() * k.div_ceil(32) * 4
                // TYPE-COVERAGE #14 (text): the offsets section ((k+1)*8) + the bytes blob per column.
                + text_layouts
                    .iter()
                    .map(|t| (k + 1) * 8 + t.bytes_len as usize)
                    .sum::<usize>()) as u64,
            allocated_bytes: new_device_memory.metadata().allocated_bytes,
            count_header_byte_offset: 0,
            resident_device_int4_columns: shard_int4_names.clone(),
            resident_device_int8_columns: shard_int8_names.clone(),
            // TYPE-COVERAGE #14 (numeric): the b128 (Numeric/Uuid) section rides the rollover payload.
            resident_device_numeric_columns: shard_numeric_names.clone(),
            // TYPE-COVERAGE #14 (bool): the bool bitmaps ride the rollover payload (offsets from the builder).
            resident_device_bool_columns: bool_layouts,
            // TYPE-COVERAGE #14 (text): the DENSE text (offsets+blob) layouts ride the rollover payload.
            resident_device_text_columns: text_layouts,
            // ADR-006 (NULL coverage): the validity-bitmap layouts (offsets into THIS payload) ride the
            // rollover, so a null-carrying rolled shard reads NULL-correctly on-device + rehydrate. Empty
            // for a null-free batch (the common case) -> byte-identical to before.
            resident_device_null_columns: null_layouts,
            gpu_id,
            schema,
            table: table.to_string(),
            device_memory_proof: Some(new_device_memory.metadata().clone()),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: pressured,
            memory_pressure_active: pressured,
            // D4: the descriptor IS the one-load snapshot — buffer + regions ride it. First `k`
            // created_by slots = `commit_seq`; the headroom keeps the born-visible fill for now
            // (D3 stamps later appends into it via stamp_created_by_resident_shard_slots).
            device_memory: Some(Arc::clone(&new_device_memory)),
            deleted_by_region: None,
            created_by_region: rolled_created_by_region.clone(),
            row_id_region: rolled_row_id_region.clone(),
            max_created_by: stamps_max,
        };
        // Write-side bookkeeping mirrors of the SAME Arcs (alloc/stamp/purge choreography unchanged);
        // readers take them from the published descriptor above.
        if let Some(created_region) = rolled_created_by_region {
            self.read_state
                .residency
                .shard_created_by_memory
                .insert_shard(table, new_shard_id, created_region);
        }
        if let Some(region) = rolled_row_id_region {
            self.read_state
                .residency
                .shard_row_id_memory
                .insert_shard(table, new_shard_id, region);
        }
        // Publish the new shard's device memory BEFORE its metadata, so a reader that observes the new shard
        // in the shards list always finds its device memory (the recompaction loads the list then the memory).
        self.read_state.residency.shard_device_memory.insert_shard(
            table,
            new_shard_id,
            new_device_memory,
        );
        self.read_state.residency.with_shards_mut(|shards| {
            if let Some(table_shards) = shards.get_mut(table) {
                table_shards.push(new_shard);
            }
        });
        self.read_state
            .residency
            .open_shard_append_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Slice A2 (incremental DELETE): stamp `deleted_by[slot] = commit_seq` for each of `slots` in a
    /// resident shard's tombstone section, via ONE targeted device write (`append_owned_chunks`, which
    /// bounds-checks each chunk against the allocation). This is an OUT-OF-LINE tombstone: it writes only
    /// the `deleted_by` metadata word, NEVER the row's column bytes — so a lock-free, predicate-free reader
    /// can never observe a torn row (review Finding 3), and the change is a single aligned u64 store.
    ///
    /// Returns `false` (caller must fall back to invalidate + re-admit) if the shard is missing / any slot is
    /// out of `[0, row_count)` / the region allocation or device write fails. `slots` are LOCAL indices.
    ///
    /// WIRED into the DELETE commit path by SV4b (slot-finding via the pruned-shard predicate).
    /// **SV4 PREREQUISITES (audit-flagged):**
    ///  1. **Lifecycle/leak — DONE (SV4 prereq #1):** `shard_deleted_by_memory` cleanup is now wired at every
    ///     site the resident buffer it annotates is retired. Two categories, distinguished by whether an
    ///     invalidate precedes the retire:
    ///       - `invalidate_table` (device buffer freed, cell kept) on the three invalidate paths (serialized
    ///         `invalidate_relational_residency_table` + concurrent-commit + memory-pressure variants).
    ///       - full `remove_table` (keys erased) on the paths that retire a buffer WITHOUT a preceding
    ///         invalidate: the single-buffer AND sharded re-admit branches in `populate_..._snapshot_inner`
    ///         (a warmup/refresh has no invalidate), the BUDGET-EVICTION path (`RelationalResidentCache::
    ///         remove_table`, evicting a different table during admission), and `apply_drop_table` (a DROPped
    ///         table is gone for good — stronger than `shard_device_memory`, which leaves `None` cells on DROP).
    ///         Gates (all sabotage-verified non-vacuous): `shard_deleted_by_region_released_on_invalidate_and_drop`
    ///         (invalidate + DROP), `shard_deleted_by_region_released_on_warmup_readmit` (sharded re-admit with no
    ///         preceding invalidate), and `resident_cache_remove_table_releases_deleted_by_region` (the eviction-
    ///         cleanup method contract, currently defensive). This keeps a re-admit (rebuilt all-live from the host
    ///         store) from inheriting a stale tombstone region and stops evicted/dropped tables leaking regions.
    ///  2. **Concurrency:** hold the COMMIT LOCK across the get-or-allocate below, else two concurrent
    ///     first-deletes to the same shard both allocate + the losing region's `Arc` leaks (writes still land
    ///     safely; only the buffer leaks). SV4 runs this under the serialized commit lock, which is the fix.
    pub(crate) fn tombstone_resident_shard_slots(
        &self,
        table: &str,
        shard_id: u32,
        slots: &[u32],
        commit_seq: Index,
    ) -> bool {
        let stamped: Vec<(u32, Index)> = slots.iter().map(|&slot| (slot, commit_seq)).collect();
        self.tombstone_resident_shard_slots_stamped(table, shard_id, &stamped)
    }

    /// U1: the per-slot-stamp generalization of [`Self::tombstone_resident_shard_slots`] — a
    /// merged lane apply batch spans commit seqs, so each tombstone carries its OWN stamp
    /// (exactly like the append path's `InsertPerRow` birth stamps). Same on-demand region
    /// allocation, same bounds/decline contract.
    pub(crate) fn tombstone_resident_shard_slots_stamped(
        &self,
        table: &str,
        shard_id: u32,
        slots: &[(u32, Index)],
    ) -> bool {
        if slots.is_empty() {
            return true;
        }
        // Read the shard's shape once. The commit lock (when wired) makes this + the region allocation atomic;
        // even without it, `append_owned_chunks` re-bounds-checks every chunk vs the region's allocated_bytes,
        // so a torn read can only produce a rejected write (-> `false` -> re-admit), never an OOB.
        let (capacity, row_count, gpu_id) = {
            let shards = self.read_state.residency.shards.load();
            let Some(table_shards) = shards.get(table) else {
                return false;
            };
            let Some(shard) = table_shards.iter().find(|s| s.shard_id == shard_id) else {
                return false;
            };
            (shard.capacity, shard.row_count, shard.gpu_id)
        };
        // Bounds: every slot must be a live row of THIS shard (never headroom / out of range).
        if slots.iter().any(|&(slot, _)| (slot as usize) >= row_count) {
            return false;
        }
        // SV2: get-or-allocate the shard's ON-DEMAND `deleted_by` region (a separate `capacity`-sized u64
        // device buffer born all-live). A delete-free shard has NO entry -> the FIRST delete allocates it, so
        // the un-versioned majority pays zero. The region is `capacity` (not `row_count`) u64s so later
        // in-place appends into the open shard's headroom are already live without extending it.
        let width = std::mem::size_of::<u64>() as u64;
        let region = match self
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table.to_string(), shard_id))
        {
            Some(region) => region,
            None => {
                let _budget_allocation = self
                    .read_state
                    .residency
                    .budget_allocation_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                // Re-check after joining the allocation transaction: another first-delete may
                // have published the region while this caller waited.
                if let Some(region) = self
                    .read_state
                    .residency
                    .shard_deleted_by_memory
                    .get(&(table.to_string(), shard_id))
                {
                    region
                } else {
                    // Born all-live: every u64 is the large positive signed sentinel produced by
                    // byte-fill 0x7F. The signed visibility comparison therefore keeps it live.
                    let live_payload =
                        vec![DELETED_BY_LIVE_FILL_BYTE; capacity * std::mem::size_of::<u64>()];
                    if self
                        .relational_residency_budget_bytes(gpu_id)
                        .is_some_and(|budget| {
                            self.relational_resident_bytes_for_gpu(gpu_id)
                                .saturating_add(live_payload.len() as u64)
                                > budget
                        })
                    {
                        return false;
                    }
                    let Some(region) =
                        self.relational_residency_device_memory(gpu_id, &live_payload)
                    else {
                        return false;
                    };
                    let region = Arc::new(region);
                    if self
                        .relational_residency_budget_bytes(gpu_id)
                        .is_some_and(|budget| {
                            self.relational_resident_bytes_for_gpu(gpu_id)
                                .saturating_add(region.metadata().allocated_bytes)
                                > budget
                        })
                    {
                        return false;
                    }
                    self.read_state
                        .residency
                        .shard_deleted_by_memory
                        .insert_shard(table, shard_id, Arc::clone(&region));
                    // D4: republish the descriptor with the same region before releasing the
                    // allocation transaction; readers obtain resources from one shard snapshot.
                    self.read_state.residency.with_shards_mut(|shards| {
                        if let Some(table_shards) = shards.get_mut(table) {
                            if let Some(shard) =
                                table_shards.iter_mut().find(|s| s.shard_id == shard_id)
                            {
                                shard.deleted_by_region = Some(Arc::clone(&region));
                            }
                        }
                    });
                    region
                }
            }
        };
        // U1 perf lever B: ONE scatter launch (2 HtoDs + 1 kernel) instead of N per-slot HtoD
        // chunks — the measured device-apply cost (~468us/wave at ~75 tombstones). The region is
        // JUST deleted_by (0-based u64s): slot `s`'s stamp is at byte `s*8`, which the scatter
        // kernel computes from the slot index directly. `width` is unused on this path now.
        let _ = width;
        let slot_ids: Vec<u32> = slots.iter().map(|&(slot, _)| slot).collect();
        let stamps: Vec<u64> = slots.iter().map(|&(_, stamp)| stamp).collect();
        region.scatter_u64_slots(&slot_ids, &stamps).is_ok()
    }

    /// RETIREMENT A1: stamp the row-identity region for `k` just-appended contiguous slots
    /// `[first_slot, first_slot+k)` with the rows' `row_id`s. GET-OR-SKIP (not get-or-allocate):
    /// a shard WITHOUT a region (benchmark/synthetic install — no host identity exists) skips
    /// silently, keeping the absent-region = identity-unknown contract; a shard WITH one (admission
    /// or rollover created it, sentinel-filled headroom) gets exact stamps. Runs BEFORE the
    /// row_count bump (the slots are invisible headroom), same ordering as the version stamps.
    /// Get-or-allocate a shard's ON-DEMAND `created_by` region (factored from the stamp path
    /// so the FUSED apply pass shares the exact allocate + descriptor-republish semantics; see
    /// `stamp_created_by_resident_shard_slots` for the SV6/D4 contract).
    fn get_or_alloc_created_by_region(
        &self,
        table: &str,
        shard_id: u32,
        capacity: usize,
        gpu_id: u16,
    ) -> Option<Arc<CudaResidentDeviceMemory>> {
        if let Some(region) = self
            .read_state
            .residency
            .shard_created_by_memory
            .get(&(table.to_string(), shard_id))
        {
            return Some(region);
        }
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(region) = self
            .read_state
            .residency
            .shard_created_by_memory
            .get(&(table.to_string(), shard_id))
        {
            return Some(region);
        }
        let payload = vec![CREATED_BY_VISIBLE_FILL_BYTE; capacity * std::mem::size_of::<u64>()];
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(payload.len() as u64)
                    > budget
            })
        {
            return None;
        }
        let region = Arc::new(self.relational_residency_device_memory(gpu_id, &payload)?);
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(region.metadata().allocated_bytes)
                    > budget
            })
        {
            return None;
        }
        self.read_state
            .residency
            .shard_created_by_memory
            .insert_shard(table, shard_id, Arc::clone(&region));
        // D4 (ADR-013 pre2): REPUBLISH the descriptor with the new region (see the
        // deleted_by twin above). Born all-visible (fill 0), so a reader observing the
        // republished descriptor mid-commit is unchanged until the stamps + row_count land.
        self.read_state.residency.with_shards_mut(|shards| {
            if let Some(table_shards) = shards.get_mut(table) {
                if let Some(shard) = table_shards.iter_mut().find(|s| s.shard_id == shard_id) {
                    shard.created_by_region = Some(Arc::clone(&region));
                }
            }
        });
        Some(region)
    }

    /// E2.5c 2M+ push (b): the FUSED merged-apply device pass — column scatter + created_by /
    /// row-id stamps + PK hash-index insert in one staging HtoD + one launch + one decline DtoH,
    /// followed by the ordered device-header HtoD. The decline read completes every kernel block
    /// before that header publishes, preserving the SV6 stamp-before-publish order. Returns:
    /// - `None`  -> not eligible; the caller runs the unfused sequence (byte-identical);
    /// - `Some(true)`  -> the pass covered append + stamps + index maintenance;
    /// - `Some(false)` -> device failure mid-pass; bytes live only in invisible headroom, the
    ///   caller must NOT publish and must invalidate + re-admit (the unfused contract).
    #[allow(clippy::too_many_arguments)]
    fn try_fused_apply_in_place(
        &self,
        table: &str,
        shard_id: u32,
        shard_device_memory: &Arc<CudaResidentDeviceMemory>,
        chunk_offsets: &[u64],
        column_values: &[Vec<i32>],
        row_count: usize,
        capacity: usize,
        gpu_id: u16,
        stamps: &[Index],
        row_ids: Option<&[u64]>,
    ) -> Option<bool> {
        let k = stamps.len();
        if k == 0
            || chunk_offsets.len() != column_values.len()
            || column_values.iter().any(|col| col.len() != k)
            || row_count.saturating_add(k) > capacity
        {
            return None;
        }
        let base_row_u32 = u32::try_from(row_count).ok()?;
        // created_by region (get-or-allocate — same semantics as the unfused stamp path).
        let created_by_region =
            self.get_or_alloc_created_by_region(table, shard_id, capacity, gpu_id)?;
        // Row-id region: get-or-skip, exactly like `stamp_row_id_resident_shard_slots` (a
        // region-less lineage stamps nothing).
        let row_ids_arg = row_ids.and_then(|ids| {
            if ids.len() != k {
                return None;
            }
            self.read_state
                .residency
                .shard_row_id_memory
                .get(&(table.to_string(), shard_id))
                .map(|region| {
                    (
                        ids,
                        gpu_db_execution::CudaWriteDestination {
                            memory: Arc::clone(&region),
                            byte_offset: (row_count as u64) * std::mem::size_of::<u64>() as u64,
                        },
                    )
                })
        });
        if row_ids.is_some() && row_ids_arg.is_none() {
            // ids provided but no region (or arity drift): only the no-region case is a
            // legitimate skip; arity drift declines to the unfused path's own guards.
            if row_ids.is_some_and(|ids| ids.len() != k) {
                return None;
            }
        }
        // PK device-index snapshot: mirror `extend_shard_pk_device_index_on_append`'s basis
        // validation + load rule for the (at most one) column with a LIVE cached device index.
        // More than one live entry -> not eligible (the fused kernel inserts into one index).
        let mut index_arg: Option<gpu_db_execution::CudaWriteIndex> = None;
        let mut index_col: Option<usize> = None;
        {
            let new_count = row_count + k;
            let device_ptr = shard_device_memory.device_ptr();
            let mut live: Vec<(usize, Arc<CudaResidentDeviceMemory>, u32, u32)> = Vec::new();
            {
                let cache = self
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                for col_idx in 0..column_values.len() {
                    let key = (table.to_string(), shard_id, col_idx);
                    let Some(entry) = cache.get(&key) else {
                        continue;
                    };
                    if entry.resident_device_ptr != device_ptr || entry.row_count != row_count {
                        continue; // stale basis -> the prober path converges it (unfused rule)
                    }
                    let Some(index) = entry.device_index.clone() else {
                        continue; // DECLINED is monotone under appends
                    };
                    live.push((col_idx, index, entry.table_mask, entry.hash_shift));
                }
            }
            match live.len() {
                0 => {}
                1 => {
                    let (col_idx, index, table_mask, hash_shift) = live.pop().expect("len 1");
                    let table_size = (table_mask as u64) + 1;
                    if (new_count as u64).saturating_mul(2) > table_size {
                        // Past the builder's load rule -> drop so the next probe rebuilds at
                        // the grown size (unfused rule), then run WITHOUT index maintenance.
                        self.read_state
                            .residency
                            .shard_pk_device_index
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .remove(&(table.to_string(), shard_id, col_idx));
                    } else {
                        index_arg = Some(gpu_db_execution::CudaWriteIndex {
                            memory: index,
                            table_mask,
                            hash_shift,
                            key_column: col_idx as u32,
                        });
                        index_col = Some(col_idx);
                    }
                }
                _ => return None, // multi-index shard: the unfused per-column loop handles it
            }
        }
        // Flatten col-major values + owned per-column destinations.
        let values: Vec<i32> = column_values.iter().flatten().copied().collect();
        let columns: Vec<gpu_db_execution::CudaWriteDestination> = chunk_offsets
            .iter()
            .map(|&byte_offset| gpu_db_execution::CudaWriteDestination {
                memory: Arc::clone(shard_device_memory),
                byte_offset,
            })
            .collect();
        let request = gpu_db_execution::FusedApplyRequest {
            columns: &columns,
            values: &values,
            stamps,
            created_by: gpu_db_execution::CudaWriteDestination {
                memory: created_by_region,
                byte_offset: (row_count as u64) * std::mem::size_of::<u64>() as u64,
            },
            row_ids: row_ids_arg,
            index: index_arg,
            base_row: base_row_u32,
            // The device row-count header word (the unfused path's FINAL append chunk).
            header: gpu_db_execution::CudaWriteDestination {
                memory: Arc::clone(shard_device_memory),
                byte_offset: 0,
            },
        };
        match shard_device_memory.submit_i32_fused_apply(&request) {
            Ok(dup) => {
                if let Some(col_idx) = index_col {
                    // Post-launch entry update, mirroring the unfused path: advance the basis,
                    // or DECLINE monotonically on a dup/overflow verdict.
                    let mut cache = self
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if let Some(entry) = cache.get_mut(&(table.to_string(), shard_id, col_idx)) {
                        if entry.resident_device_ptr == shard_device_memory.device_ptr()
                            && entry.row_count == row_count
                        {
                            if dup {
                                entry.device_index = None;
                            }
                            entry.row_count = row_count + k;
                        }
                    }
                }
                self.read_state
                    .residency
                    .fused_apply_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(true)
            }
            Err(_) => {
                if let Some(col_idx) = index_col {
                    // A failed launch may have partially mutated the index -> drop the entry
                    // (rebuild on next probe); never a wrong index. Same as the unfused path.
                    self.read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .remove(&(table.to_string(), shard_id, col_idx));
                }
                Some(false)
            }
        }
    }

    fn stamp_row_id_resident_shard_slots(
        &self,
        table: &str,
        shard_id: u32,
        first_slot: usize,
        row_ids: &[u64],
    ) -> bool {
        if row_ids.is_empty() {
            return true;
        }
        let Some(region) = self
            .read_state
            .residency
            .shard_row_id_memory
            .get(&(table.to_string(), shard_id))
        else {
            return true; // no identity region on this shard lineage: nothing to keep consistent
        };
        let mut bytes = Vec::with_capacity(row_ids.len() * 8);
        for row_id in row_ids {
            bytes.extend_from_slice(&row_id.to_le_bytes());
        }
        region
            .append_owned_chunks(vec![CudaOwnedDeviceMemoryChunk {
                byte_offset: (first_slot as u64) * 8,
                bytes,
            }])
            .is_ok()
    }

    /// SV6 (the SV5 `created_by` flip-gate): stamp `created_by[slot] = commit_seq` for the `k` just-appended
    /// CONTIGUOUS slots `[first_slot, first_slot + k)` of a resident shard, get-or-allocating the shard's
    /// ON-DEMAND `created_by` region — a `capacity`-sized i64 device buffer born all-visible
    /// ([`CREATED_BY_VISIBLE_FILL_BYTE`] = 0x00: `0 <= read_txn_id` for every snapshot) — on its first
    /// stamped append, so un-versioned shards pay zero (the same sparse-versioning property as
    /// `deleted_by`). The caller MUST invoke this BEFORE the shard's `row_count` bump publishes the slots
    /// (they are invisible headroom here — see the append path's ORDER comment) and runs under the commit
    /// lock, making the get-or-allocate atomic (SV2 prereq #2). Returns `false` (caller falls back to
    /// invalidate + re-admit; the re-admit purge releases any partial region) on any allocation or device
    /// write failure. ONE contiguous chunk write (`k * 8` bytes at `first_slot * 8`), bounds-checked by
    /// `append_owned_chunks` against the region's allocation.
    #[allow(clippy::too_many_arguments)] // mirrors the shard-shape tuple its caller already destructured
    pub(super) fn stamp_created_by_resident_shard_slots(
        &self,
        table: &str,
        shard_id: u32,
        first_slot: usize,
        capacity: usize,
        gpu_id: u16,
        // D3: one birth stamp per appended slot (the wave-batched flush spans commit seqs).
        stamps: &[Index],
    ) -> bool {
        let k = stamps.len();
        if k == 0 {
            return true;
        }
        // Bounds: the stamped slots must lie inside the region (capacity slots). `append_owned_chunks`
        // re-checks against the real allocation, so a torn shape read can only reject, never write OOB.
        if first_slot.saturating_add(k) > capacity {
            return false;
        }
        let Some(region) = self.get_or_alloc_created_by_region(table, shard_id, capacity, gpu_id)
        else {
            return false;
        };
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(stamps));
        for stamp in stamps {
            bytes.extend_from_slice(&stamp.to_le_bytes());
        }
        region
            .append_owned_chunks(vec![CudaOwnedDeviceMemoryChunk {
                byte_offset: (first_slot as u64) * std::mem::size_of::<u64>() as u64,
                bytes,
            }])
            .is_ok()
    }
}
