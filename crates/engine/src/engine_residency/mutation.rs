//! Resident append, rollover, sparse-version stamping, and fused-apply ownership.

use super::append_source::ResidentAppendSource;
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
        let apply_leader =
            crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(std::cell::Cell::get);
        let _apply = if apply_leader {
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
            return self.try_append_to_resident_open_shard(
                table,
                ResidentAppendSource::Rows(new_rows),
                created_by,
                row_ids,
            );
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
        // before the caller's publication join, so a reader that observes the new committed_seq can
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

    /// Size every new device allocation the canonical explicit-transaction append will retain.
    /// The caller runs under the serialized commit boundary and reserves this geometry before WAL;
    /// [`Self::try_append_to_resident_open_shard`] then performs the same one-batch append. Existing
    /// payload/index headroom costs zero, while first-use version sidecars and rollover generations
    /// are charged in full.
    pub(crate) fn transaction_resident_append_allocation_bytes(
        &self,
        table: &RelationalTable,
        new_rows: &[Vec<SqlValue>],
        final_named_indexes_required: bool,
    ) -> Result<(u16, u64), ExecuteError> {
        if new_rows.is_empty() {
            return Ok((self.planner.default_gpu_id(), 0));
        }
        let pressured_gpus = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .clone();
        let (
            capacity,
            row_count,
            gpu_id,
            int4_appendable,
            valid,
            int4_columns,
            int8_columns,
            numeric_columns,
            bool_columns,
            text_columns,
            null_columns,
            created_by_present,
            row_ids_present,
        ) = {
            let shards = self.read_state.residency.shards.load();
            let open = shards
                .get(&table.name)
                .and_then(|table_shards| table_shards.last())
                .ok_or_else(|| {
                    ExecuteError::Serialization(format!(
                        "relation \"{}\" lost its open resident shard before transaction publication",
                        table.name
                    ))
                })?;
            (
                open.capacity,
                open.row_count,
                open.gpu_id,
                open.int4_appendable,
                open.is_valid(pressured_gpus.contains(&open.gpu_id)),
                open.resident_device_int4_columns.len(),
                open.resident_device_int8_columns.len(),
                open.resident_device_numeric_columns.len(),
                open.resident_device_bool_columns.len(),
                open.resident_device_text_columns.len(),
                open.resident_device_null_columns.len(),
                open.created_by_region.is_some(),
                open.row_id_region.is_some(),
            )
        };
        if !int4_appendable || !valid {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{}\" cannot retain its open GPU shard for transaction publication",
                table.name
            )));
        }
        if int4_columns + int8_columns + numeric_columns + bool_columns + text_columns
            != table.columns.len()
        {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{}\" resident layout changed before transaction publication",
                table.name
            )));
        }
        let k = new_rows.len();
        let batch_has_null = new_rows
            .iter()
            .any(|row| row.iter().any(|value| matches!(value, SqlValue::Null)));
        if text_columns == 0
            && null_columns == 0
            && !batch_has_null
            && row_count.checked_add(k).is_some_and(|end| end <= capacity)
        {
            if !row_ids_present {
                return Err(ExecuteError::Serialization(format!(
                    "relation \"{}\" lost its resident entity-identity region before transaction publication",
                    table.name
                )));
            }
            let bytes = if created_by_present {
                0
            } else {
                (capacity as u64).checked_mul(8).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction created-by reservation overflowed".to_string(),
                    ))
                })?
            };
            return Ok((gpu_id, bytes));
        }

        let names = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let types = table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect::<Vec<_>>();
        let has_text = types.iter().any(|ty| matches!(ty, SqlType::Text));
        let named_indexes_required =
            final_named_indexes_required || self.relational_named_index_publication_required(table);
        if !has_text && !batch_has_null {
            let desired_capacity =
                super::rollover::ResidentRolloverPlan::fixed_width_desired_capacity(
                    k,
                    Some(self.shard_size_target()),
                )
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction fixed-width rollover capacity overflowed".to_string(),
                    ))
                })?;
            // This is the pre-WAL half of the same sealed planner used by the global rollover
            // publisher. The reservation itself is charged under `budget_allocation_lock` by
            // `TransactionGpuReservation::reserve`; the later apply deliberately replans under
            // that lock rather than smuggling a stale device/shape token through the canonical
            // WAL envelope. Any generation/GPU/layout drift therefore fails closed at apply.
            let current_bytes = self.relational_resident_bytes_for_gpu(gpu_id);
            let remaining_budget = self
                .relational_residency_budget_bytes(gpu_id)
                .map(|budget| budget.saturating_sub(current_bytes));
            let plan = super::rollover::ResidentRolloverPlan::fixed_width_null_free(
                table,
                &types,
                k,
                desired_capacity,
                true,
                named_indexes_required,
                remaining_budget,
            )?
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" cannot fit a dense fixed-width rollover before WAL",
                    table.name
                )))
            })?;
            return Ok((gpu_id, plan.total_allocation_bytes()));
        }

        // Text and NULL-carrying generations stay dense. Their variable-length/validity sections
        // have no capacity-strided append contract, so budget pressure must not invent `2*k`
        // headroom for this fallback.
        let new_capacity = k;
        let (payload, ..) =
            build_relational_device_payload_with_capacity(&names, &types, new_rows, new_capacity)
                .map_err(|error| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction publication payload sizing failed for relation \"{}\": {error}",
                    table.name
                )))
            })?;
        let named_index_bytes = if named_indexes_required {
            estimated_named_index_bytes_for_shard(table, k, new_capacity).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has unsupported mandatory index allocation geometry",
                    table.name
                )))
            })?
        } else {
            0
        };
        let bytes = (payload.len() as u64)
            .checked_add((new_capacity as u64).checked_mul(16).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "transaction rollover sidecar reservation overflowed".to_string(),
                ))
            })?)
            .and_then(|bytes| bytes.checked_add(named_index_bytes))
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "transaction rollover reservation overflowed".to_string(),
                ))
            })?;
        Ok((gpu_id, bytes))
    }

    /// S-d2b: append a committed INSERT's applied rows IN PLACE into the resident table's OPEN shard's
    /// headroom (the last shard in `residency.shards`) — the shard-path analog of the single-buffer append.
    /// Empty + NULL-bearing rows are already rejected by the caller. Returns false (caller invalidates +
    /// re-admits) when the open shard isn't int4-appendable, is invalid, or has no headroom (seal + a fresh
    /// open shard on overflow is S-d2c), or the device append fails. Shard tables read via device
    /// recompaction and have no single-buffer `wave_index` to drop. The device PK-index cache is
    /// `(ptr,row_count)`-validated, so an in-place append makes the next probe rebuild or extend it.
    pub(super) fn try_append_to_resident_open_shard(
        &self,
        table: &str,
        mut source: ResidentAppendSource<'_, '_>,
        created_by: AppendCreatedBy<'_>,
        row_ids: Option<&[u64]>,
    ) -> bool {
        // D3: materialize one birth stamp per appended row (validated len) — the in-place branch
        // stamps them into the open shard's created_by region and the rollover branch bakes them
        // into the new shard's region; both bump the descriptor's max_created_by high-water.
        let Some(stamps) = created_by.stamps_for(source.row_count()) else {
            return false;
        };
        let stamps_max = stamps.iter().copied().max().unwrap_or(0);
        let pressured_gpus = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .clone();
        let k = source.row_count();
        // ADR-006 (NULL coverage): does this appended BATCH carry any NULL? A null-bearing batch cannot
        // append in place (the in-place chunk encoder has no validity-bitmap channel); it rolls a DENSE
        // shard whose bitmaps the payload builder constructs — exactly like a text column.
        let batch_has_null = source.has_null();
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
        ) = {
            let shards = self.read_state.residency.shards.load();
            let Some(table_shards) = shards.get(table) else {
                return false;
            };
            let Some(open) = table_shards.last() else {
                return false;
            };
            let pressured = pressured_gpus.contains(&open.gpu_id);
            if !open.int4_appendable
                || !open.is_valid(pressured)
                || !source.matches_open_descriptor(open, pressured)
            {
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
            )
        };
        // TYPE-COVERAGE track 2 slice 2 stage (ii): CATALOG-ordered names/types drive the
        // section-aware chunk encoder + the rollover payload (mixed i32/i64 sections —
        // catalog order != section ordinal). Defensive arity guard: the shard's section
        // lists must cover the catalog exactly, else decline to the re-admit oracle.
        let catalog = self.catalog_snapshot();
        let Some(catalog_table) = catalog.relational_catalog.get(table).cloned() else {
            return false;
        };
        if !source.matches_current_catalog(&catalog_table, catalog.commit_seq) {
            return false;
        }
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
        let preheld_budget_reservation = source.holds_budget_reservation();

        // Only fixed-width, NULL-free batches may append into the open shard's headroom.
        if !has_text
            && !has_null_shard
            && !batch_has_null
            && !source.requires_dense_rollover()
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
            // Append at the shard-local `row_count`, not its global `row_start`.
            let chunks = match source.append_chunks(&column_types, capacity, row_count) {
                Ok(chunks) => chunks,
                Err(_) => return false,
            };
            // Typed plans install/use the exact pre-WAL created-by Arc; legacy rows retain sparse allocation.
            let typed_created_by_region = match &mut source {
                ResidentAppendSource::Rows(_) => None,
                ResidentAppendSource::DevicePlan(plan) => {
                    match plan.take_in_place_created_by(capacity, row_count) {
                        Some(super::fixed_insert::PreparedInPlaceCreatedBy::Existing(region)) => {
                            Some(region)
                        }
                        Some(super::fixed_insert::PreparedInPlaceCreatedBy::Reserved(pending)) => {
                            let Some(region) = pending.into_region(capacity) else {
                                return false;
                            };
                            match self.get_or_alloc_created_by_region(
                                table,
                                shard_id,
                                capacity,
                                gpu_id,
                                preheld_budget_reservation,
                                Some(region),
                            ) {
                                Some(region) => Some(region),
                                None => return false,
                            }
                        }
                        None => return false,
                    }
                }
            };
            // The fused i32-only pass combines scatter, sidecar stamps, and index maintenance.
            let fused = if self.fused_apply_enabled()
                && num_i64_cols == 0
                && num_numeric_cols == 0
                && num_bool_cols == 0
            {
                // The fused scatter remains an internal all-i32 operator. Mixed fixed-width
                // plans intentionally skip it and use the same sealed chunks below.
                let column_values = source.i32_columns(column_count);
                let append_started =
                    crate::engine_dml_concurrent::wave_device_phase_timing_enabled()
                        .then(std::time::Instant::now);
                // Per-COLUMN offsets only: the encoder's FINAL chunk is the device
                // row-count header (offset 0), which the fused submit publishes after fencing the kernel.
                let Some(chunk_offsets) = chunks.first_offsets(column_count) else {
                    return false;
                };
                let outcome = column_values.as_ref().and_then(|column_values| {
                    self.try_fused_apply_in_place(
                        table,
                        &catalog_table,
                        shard_id,
                        &shard_device_memory,
                        chunk_offsets.as_slice(),
                        column_values,
                        row_count,
                        capacity,
                        gpu_id,
                        &stamps,
                        row_ids,
                        preheld_budget_reservation,
                        typed_created_by_region.clone(),
                    )
                });
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
                // Keep the count header private until values, BoolBits, and identity sidecars
                // are complete.  The sealed encoder always emits it last, but bool uses a
                // separate bitmap operator and therefore must run before this final publish.
                let (header, chunks) = match chunks.split_final_header() {
                    Some((header, payload)) if header.byte_offset == 0 => (header, payload),
                    _ => return false,
                };
                // `deleted_by` needs no write on append — the headroom was pre-filled with the live sentinel at
                // admission, so appended rows are born live. SV6: an UPDATE-appended NEW VERSION additionally
                // stamps `created_by = commit_seq` (below); a plain INSERT append stays unstamped (born-visible).
                let append_started =
                    crate::engine_dml_concurrent::wave_device_phase_timing_enabled()
                        .then(std::time::Instant::now);
                let append_result = chunks.append_to(&shard_device_memory);
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
                match &mut source {
                    ResidentAppendSource::Rows(rows) => {
                        for layout in &shard_bool_layouts {
                            let Some(col_idx) = column_names.iter().position(|n| n == &layout.name)
                            else {
                                return false;
                            };
                            let values: Vec<u8> = rows
                                .iter()
                                .map(|row| u8::from(matches!(row[col_idx], SqlValue::Bool(true))))
                                .collect();
                            if shard_device_memory
                                .set_bool_bitmap_range(
                                    layout.bitmap_byte_offset,
                                    row_count as u32,
                                    &values,
                                )
                                .is_err()
                            {
                                return false;
                            }
                        }
                    }
                    ResidentAppendSource::DevicePlan(plan) => {
                        let Some(uploads) = plan.take_bool_uploads(&shard_bool_layouts) else {
                            return false;
                        };
                        for (layout, upload) in shard_bool_layouts.iter().zip(uploads) {
                            if upload.name.as_ref() != layout.name
                                || shard_device_memory
                                    .set_bool_bitmap_range(
                                        layout.bitmap_byte_offset,
                                        row_count as u32,
                                        &upload.values,
                                    )
                                    .is_err()
                            {
                                return false;
                            }
                        }
                    }
                }
                // SV6 ORDER (load-bearing): stamp created_by BEFORE the `row_count` bump below publishes the
                // appended slots. The slots are still invisible headroom here, so a torn state (values + stamps
                // written, count not bumped) is unreadable; stamping AFTER the bump would let a reader bound to
                // an older snapshot observe the new version born-visible (created_by = fill 0) — exactly the
                // SV5 P2 double-read window this gate closes. A stamp failure -> false -> the caller re-admits
                // (the re-admit purge releases any partial region; rebuild-all-live is always correct).
                if !self.stamp_created_by_resident_shard_slots(
                    table,
                    shard_id,
                    row_count,
                    capacity,
                    gpu_id,
                    &stamps,
                    preheld_budget_reservation,
                    typed_created_by_region,
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
                if shard_device_memory
                    .append_owned_chunks(std::iter::once(header))
                    .is_err()
                {
                    return false;
                }
            }
            let appended_bytes = (k
                * (num_i32_cols * std::mem::size_of::<i32>()
                    + num_i64_cols * std::mem::size_of::<i64>()
                    + num_numeric_cols * 16)) as u64;
            // S-d3: extend the open shard's zone map (min/max per int4 column) to cover the appended
            // rows. The stats vector is INT4-ORDINAL-aligned, so iterate only the i32-section catalog
            // columns, in order (stage ii: i64 columns carry no zone map — they simply never prune).
            let Some(new_min_max) = source.int4_min_max(&column_types) else {
                return false;
            };
            // M1 (ledger #24): incrementally maintain the DEVICE PK index too (the index_insert
            // kernel), so the wave-batched device locate never triggers the O(rows) rebuild.
            // Only fires when a device index is cached; no-op otherwise.
            if !fused {
                let idx_started = crate::engine_dml_concurrent::wave_device_phase_timing_enabled()
                    .then(std::time::Instant::now);
                if let Some(rows) = source.rows() {
                    let Some(column_values) = source.i32_columns(column_count) else {
                        return false;
                    };
                    let Some(column_values) = column_values.rows() else {
                        return false;
                    };
                    if !self.extend_shard_pk_device_index_on_append(
                        table,
                        shard_id,
                        shard_device_memory.device_ptr(),
                        row_count,
                        column_values,
                        rows,
                    ) {
                        return false;
                    }
                } else {
                    // A sealed fixed-width source is admitted only for a no-index catalog shape.
                    // Retire any impossible stale index basis before descriptor publication.
                    if !catalog_table.indexes.is_empty() {
                        return false;
                    }
                    self.read_state
                        .residency
                        .purge_shard_pk_index_for_table(table);
                }
                if let Some(started) = idx_started {
                    crate::engine_dml_concurrent::WAVE_DEVICE_STATS[2].fetch_add(
                        started.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            } else {
                // The fused pass owns the raw primary-key insert. PRODUCT-002 named compound
                // secondaries still require their fingerprint candidates before row-count publication.
                let idx_started = crate::engine_dml_concurrent::wave_device_phase_timing_enabled()
                    .then(std::time::Instant::now);
                if let Some(rows) = source.rows() {
                    if !self.extend_shard_fingerprint_device_indexes_on_append(
                        table,
                        shard_id,
                        shard_device_memory.device_ptr(),
                        row_count,
                        rows,
                    ) {
                        return false;
                    }
                    if !self.named_indexes_cover_after_fused_append(
                        table,
                        shard_id,
                        shard_device_memory.device_ptr(),
                        row_count,
                        row_count + k,
                    ) {
                        return false;
                    }
                } else if !catalog_table.indexes.is_empty() {
                    return false;
                } else {
                    self.read_state
                        .residency
                        .purge_shard_pk_index_for_table(table);
                }
                if let Some(started) = idx_started {
                    crate::engine_dml_concurrent::WAVE_DEVICE_STATS[2].fetch_add(
                        started.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            }
            self.read_state
                .residency
                .with_shards_mut_for_table(table, |shards| {
                    if let Some(table_shards) = shards.get_mut(table) {
                        if let Some(open) = table_shards.last_mut() {
                            open.row_count += k;
                            // D3: the high-water publishes WITH the row_count that exposes the slots —
                            // a reader at s >= hwm treats the shard as effectively version-free.
                            open.max_created_by = open.max_created_by.max(stamps_max);
                            open.resident_bytes =
                                open.resident_bytes.saturating_add(appended_bytes);
                            for (stat, (lo, hi)) in open
                                .resident_device_int4_column_stats
                                .iter_mut()
                                .zip(new_min_max.as_slice().iter())
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

        // S-d2c ROLLOVER: the open shard is full -> seal it in place and build a new private
        // generation. NULL-free fixed-width batches use the fit-aware plan below. A typed dense
        // plan already owns its exact nullable/text generation; legacy row input retains the
        // established dense builder because those layouts cannot append into headroom.
        let has_text = column_types.iter().any(|ty| matches!(ty, SqlType::Text));
        let preallocated_dense_plan = matches!(
            &source,
            ResidentAppendSource::DevicePlan(plan) if plan.dense_rollover_payload_len().is_some()
        );
        let preallocated_fixed_plan = matches!(
            &source,
            ResidentAppendSource::DevicePlan(plan) if plan.fixed_rollover_payload_len().is_some()
        );
        let named_indexes_required =
            self.relational_named_index_publication_required(&catalog_table);
        // The allocation lock makes the exact retained/pinned-generation scan and capacity choice
        // one transaction. It is intentionally held through the private construction: otherwise a
        // concurrent allocation could turn a correctly planned generation into an over-budget one
        // between the scan and CUDA allocation.
        let _budget_allocation = match &source {
            ResidentAppendSource::Rows(_) => Some(
                self.read_state
                    .residency
                    .budget_allocation_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ),
            ResidentAppendSource::DevicePlan(plan) => {
                // The move-only pre-WAL plan retains this mutex through WAL. Re-locking would
                // deadlock; accepting a plan without it would reopen the budget race it seals.
                if !plan.holds_budget_reservation() {
                    return false;
                }
                None
            }
        };
        // The move-only plan crosses WAL with its descriptor identity. Re-observe that exact
        // open tail while its allocation transaction remains held; a drift is fatal to the
        // post-WAL caller, never permission to choose fresh geometry or fall back.
        let open_still_matches = self
            .read_state
            .residency
            .shards
            .load()
            .get(table)
            .and_then(|table_shards| table_shards.last())
            .is_some_and(|open| {
                open.shard_id == shard_id
                    && open.capacity == capacity
                    && open.row_count == row_count
                    && open.row_start == row_start
                    && open.gpu_id == gpu_id
                    && open.schema == schema
                    && open.resident_device_int4_columns == shard_int4_names
                    && open.resident_device_int8_columns == shard_int8_names
                    && open.resident_device_numeric_columns == shard_numeric_names
                    && open.resident_device_bool_columns == shard_bool_layouts
                    && open.resident_device_text_columns == shard_text_layouts
                    && open.resident_device_null_columns == shard_null_layouts
            });
        if !open_still_matches {
            return false;
        }
        let (current_resident_bytes, budget_scan_entries, remaining_budget) =
            if preallocated_dense_plan || preallocated_fixed_plan {
                (None, None, None)
            } else {
                let (resident_bytes, scan_entries) =
                    self.relational_resident_bytes_and_entries_for_gpu(gpu_id);
                (
                    Some(resident_bytes),
                    Some(scan_entries),
                    self.relational_residency_budget_bytes(gpu_id)
                        .map(|budget| budget.saturating_sub(resident_bytes)),
                )
            };
        let mut sealed_rollover_coordinates = None;
        let (
            new_capacity,
            new_device_memory,
            rolled_created_by_region,
            rolled_row_id_region,
            int4_stats,
            bool_layouts,
            text_layouts,
            null_layouts,
            rollover_allocated_bytes,
            rollover_probe,
        ) = if preallocated_dense_plan {
            // The dense device plan owns three unpublished allocations made before WAL. Do not
            // re-encode rows, re-evaluate budget, or allocate here: post-WAL work is limited to
            // writing the already-reserved buffers, stamping sidecars, publishing the count
            // header last, and letting this mutation owner publish the descriptor below.
            let ResidentAppendSource::DevicePlan(prepared) = &mut source else {
                return false;
            };
            let Some(dense) = prepared.take_dense_rollover() else {
                return false;
            };
            sealed_rollover_coordinates = Some((dense.new_shard_id, dense.new_row_start));
            let budget_scan_entries = dense.budget_scan_entries;
            let pending = dense.pending;
            if named_indexes_required
                || row_ids.is_some() != pending.row_id_region.is_some()
                || pending.device_memory.metadata().allocated_bytes < pending.payload_bytes
                || stamps.len() != k
                || k.checked_mul(std::mem::size_of::<u64>()) != Some(pending.created_by_bytes)
                || pending.created_by_region.metadata().allocated_bytes
                    < pending.created_by_bytes_u64
                || pending.row_id_region.as_ref().is_some_and(|region| {
                    region.metadata().allocated_bytes < pending.created_by_bytes_u64
                })
            {
                return false;
            }
            let descriptor = pending.payload.into_descriptor_parts();
            let mut created_payload = vec![CREATED_BY_VISIBLE_FILL_BYTE; pending.created_by_bytes];
            for (slot, stamp) in stamps.iter().enumerate() {
                created_payload[slot * 8..slot * 8 + 8].copy_from_slice(&stamp.to_le_bytes());
            }
            if pending
                .created_by_region
                .append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: created_payload,
                }))
                .is_err()
            {
                return false;
            }
            // Header last makes a partial post-WAL upload private and unreadable: the allocation
            // has no descriptor until the publication below, and its row count remains zero until
            // every typed section and sidecar stamp has landed.
            if pending
                .device_memory
                .append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: descriptor.final_count_header.to_vec(),
                }))
                .is_err()
            {
                return false;
            }
            (
                k,
                pending.device_memory,
                Some(pending.created_by_region),
                pending.row_id_region,
                descriptor.int4_stats,
                descriptor.bool_layouts,
                descriptor.text_layouts,
                descriptor.null_layouts,
                pending.allocation_bytes,
                Some((
                    0,
                    budget_scan_entries,
                    pending.sidecar_fill_bytes,
                    pending.live_h2d_bytes,
                    pending.persistent_allocation_count,
                )),
            )
        } else if preallocated_fixed_plan {
            // The fixed plan preallocated/uploaded every immutable section; apply only stamps and publishes.
            let ResidentAppendSource::DevicePlan(prepared) = &mut source else {
                return false;
            };
            let Some(fixed) = prepared.take_fixed_rollover() else {
                return false;
            };
            sealed_rollover_coordinates = Some((fixed.new_shard_id, fixed.new_row_start));
            let pending = match fixed.pending.finish_post_wal(&stamps) {
                Ok(pending) => pending,
                Err(_) => return false,
            };
            if row_ids.is_some() != pending.row_id_region.is_some()
                || stamps.len().checked_mul(std::mem::size_of::<u64>())
                    != Some(pending.created_by_stamp_bytes)
                || pending.created_by_region.metadata().allocated_bytes < pending.created_by_bytes
            {
                return false;
            }
            (
                fixed.capacity,
                pending.device_memory,
                Some(pending.created_by_region),
                pending.row_id_region,
                Vec::from(pending.int4_stats),
                Vec::from(pending.bool_layouts),
                Vec::new(),
                Vec::new(),
                pending.allocation_bytes,
                Some((
                    fixed.capacity_fit_evaluations,
                    fixed.budget_scan_entries,
                    pending.sidecar_fill_bytes,
                    pending.live_h2d_bytes,
                    pending.persistent_allocation_count,
                )),
            )
        } else if !has_text && !batch_has_null {
            let plan = match &source {
                ResidentAppendSource::Rows(_) => {
                    let desired_capacity =
                        match super::rollover::ResidentRolloverPlan::fixed_width_desired_capacity(
                            k,
                            Some(self.shard_size_target()),
                        ) {
                            Some(desired) => desired,
                            None => return false,
                        };
                    match super::rollover::ResidentRolloverPlan::fixed_width_null_free(
                        &catalog_table,
                        &column_types,
                        k,
                        desired_capacity,
                        row_ids.is_some(),
                        named_indexes_required,
                        remaining_budget,
                    ) {
                        Ok(Some(plan)) => plan,
                        Ok(None) | Err(_) => {
                            self.read_state
                                .residency
                                .rollover_budget_declines
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            return false;
                        }
                    }
                }
                // A typed fixed-width rollover is consumed by the preallocated arm above.
                // Reaching this legacy builder would mean the plan identity drifted after WAL,
                // which is a failed physical apply, not authority to allocate a replacement.
                ResidentAppendSource::DevicePlan(_) => return false,
            };
            let pending_result = match &mut source {
                ResidentAppendSource::Rows(rows) => super::rollover::PendingResidentShard::build(
                    self,
                    gpu_id,
                    &catalog_table,
                    &column_types,
                    rows,
                    &stamps,
                    row_ids,
                    &plan,
                ),
                ResidentAppendSource::DevicePlan(_) => return false,
            };
            let pending = match pending_result {
                Ok(pending) => pending,
                Err(_) => return false,
            };
            debug_assert_eq!(
                pending
                    .device_memory
                    .metadata()
                    .allocated_bytes
                    .saturating_add(pending.created_by_region.metadata().allocated_bytes)
                    .saturating_add(
                        pending
                            .row_id_region
                            .as_ref()
                            .map_or(0, |region| region.metadata().allocated_bytes),
                    ),
                plan.allocation_bytes_before_indexes(),
                "fixed-width rollover allocation must match its sealed plan"
            );
            (
                plan.capacity(),
                pending.device_memory,
                Some(pending.created_by_region),
                pending.row_id_region,
                pending.int4_stats,
                pending.bool_layouts,
                Vec::new(),
                Vec::new(),
                plan.allocation_bytes_before_indexes(),
                Some((
                    plan.capacity_scan_entries(),
                    budget_scan_entries.unwrap_or(0),
                    pending.sidecar_fill_bytes,
                    pending.live_h2d_bytes,
                    pending.persistent_allocation_count,
                )),
            )
        } else {
            // Legacy row-input dense construction: a NULL-bearing shard owns exact validity
            // bitmaps for its k live rows; a text shard owns exact offsets and bytes. Neither has
            // capacity-strided append headroom. Typed dense plans took the preallocated arm above.
            let Some(rows) = source.rows() else {
                return false;
            };
            let new_capacity = k;
            let (device_payload, text_layouts, bool_layouts, int4_stats, null_layouts) =
                match build_relational_device_payload_with_capacity(
                    &column_names,
                    &column_types,
                    rows,
                    new_capacity,
                ) {
                    Ok((payload, text_cols, bool_cols, stats, _b128, null_cols)) => {
                        (payload, text_cols, bool_cols, stats, null_cols)
                    }
                    Err(_) => return false,
                };
            let named_index_bytes = if named_indexes_required {
                let Some(bytes) =
                    estimated_named_index_bytes_for_shard(&catalog_table, k, new_capacity)
                else {
                    return false;
                };
                bytes
            } else {
                0
            };
            let rollover_bytes = (device_payload.len() as u64)
                .saturating_add((new_capacity as u64).saturating_mul(8))
                .saturating_add(if row_ids.is_some() {
                    (new_capacity as u64).saturating_mul(8)
                } else {
                    0
                })
                .saturating_add(named_index_bytes);
            let dense_live_h2d_bytes = (device_payload.len() as u64)
                .saturating_add((new_capacity as u64).saturating_mul(8))
                .saturating_add(if row_ids.is_some() {
                    (new_capacity as u64).saturating_mul(8)
                } else {
                    0
                });
            if remaining_budget.is_some_and(|remaining| rollover_bytes > remaining) {
                self.read_state
                    .residency
                    .rollover_budget_declines
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return false;
            }
            let Some(new_device_memory) = self
                .relational_residency_device_memory(gpu_id, &device_payload)
                .map(Arc::new)
            else {
                return false;
            };
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
            let rolled_row_id_region = if let Some(ids) = row_ids {
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
            let allocated_bytes = new_device_memory
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
            (
                new_capacity,
                new_device_memory,
                rolled_created_by_region,
                rolled_row_id_region,
                int4_stats,
                bool_layouts,
                text_layouts,
                null_layouts,
                allocated_bytes,
                Some((
                    0,
                    budget_scan_entries.unwrap_or(0),
                    0,
                    dense_live_h2d_bytes,
                    2 + u64::from(row_ids.is_some()),
                )),
            )
        };
        if let Some(current_resident_bytes) = current_resident_bytes {
            if self
                .relational_residency_budget_bytes(gpu_id)
                .is_some_and(|budget| {
                    current_resident_bytes.saturating_add(rollover_allocated_bytes) > budget
                })
            {
                self.read_state
                    .residency
                    .rollover_budget_declines
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return false;
            }
        }
        // Dense plans sealed these coordinates before their CUDA allocations. Legacy paths retain
        // their historical checked derivation here because their geometry is built in this owner.
        let (new_shard_id, new_row_start) = match sealed_rollover_coordinates {
            Some(coordinates) => coordinates,
            None => match shard_id
                .checked_add(1)
                .zip(row_start.checked_add(row_count))
            {
                Some(coordinates) => coordinates,
                None => return false,
            },
        };
        let pressured = pressured_gpus.contains(&gpu_id);
        // D4: every resource above was constructed before this descriptor. The private pending
        // owner writes the fixed-width payload and sidecars before the final header; both the
        // sealed typed-dense allocation and legacy dense construction retain the same
        // no-public-owner-before-descriptor property.
        let new_shard = RelationalResidentShard {
            shard_id: new_shard_id,
            row_start: new_row_start,
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
            point_route_generation: Arc::new(()),
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
        // The descriptor is the first public owner: it already carries every payload/sidecar Arc.
        // Do not publish the legacy write-side mirrors while a pending generation is still being
        // built, because a failed private build must be released solely by its local owner.
        self.read_state
            .residency
            .with_shards_mut_for_table(table, |shards| {
                if let Some(table_shards) = shards.get_mut(table) {
                    table_shards.push(new_shard);
                }
            });
        // Write-side bookkeeping mirrors of the SAME Arcs (alloc/stamp/purge choreography unchanged);
        // readers take resources from the one-load published descriptor above.
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
        self.read_state.residency.shard_device_memory.insert_shard(
            table,
            new_shard_id,
            new_device_memory,
        );
        // PRODUCT-002: a rollover is a new resident generation, so every declared index must cover
        // the new shard before the commit can publish visibility. Existing shards are cache hits;
        // only this k-row shard builds. The surrounding admission budget guard remains the single
        // allocation transaction, and a failure returns false so the caller invalidates/re-admits.
        if named_indexes_required {
            let current_shards = self.read_state.residency.shards.load_full();
            let Some(table_shards) = current_shards.get(table) else {
                return false;
            };
            let incremental = self
                .read_state
                .residency
                .named_index_coverage_complete
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(table)
                .is_some_and(|(oid, indexes)| {
                    *oid == catalog_table.oid && indexes == &catalog_table.indexes
                });
            let publish_shards = if incremental {
                let Some(new_shard) = table_shards.last() else {
                    return false;
                };
                std::slice::from_ref(new_shard)
            } else {
                table_shards.as_slice()
            };
            #[cfg(feature = "probe-timing")]
            let publish_shard_visits = publish_shards.len() as u64;
            if self
                .publish_relational_resident_indexes_for_generation(
                    &catalog_table,
                    publish_shards,
                    self.committed_seq(),
                    true,
                    true,
                    !incremental,
                )
                .is_err()
            {
                self.read_state
                    .residency
                    .purge_shard_pk_index_for_table(table);
                let boundary = self.committed_seq();
                self.read_state
                    .residency
                    .flag_table_descriptors_invalidated(table, boundary, boundary);
                return false;
            }
            #[cfg(feature = "probe-timing")]
            self.record_insert_probe_named_index_shard_visits(publish_shard_visits);
        }
        #[cfg(feature = "probe-timing")]
        if let Some((
            capacity_fit_evaluations,
            budget_scans,
            sidecar_fills,
            live_h2d,
            persistent_allocations,
        )) = rollover_probe
        {
            let current_shards = self
                .read_state
                .residency
                .shards
                .load()
                .get(table)
                .map_or(0, |table_shards| table_shards.len() as u64);
            self.record_insert_probe_rollover_geometry(
                crate::engine_insert_probe::InsertProbeRolloverGeometry {
                    capacity_rows: new_capacity as u64,
                    current_shards,
                    persistent_allocations,
                    capacity_fit_evaluations,
                    budget_scans,
                    sidecar_fills,
                    live_h2d,
                },
            );
        }
        #[cfg(not(feature = "probe-timing"))]
        let _ = rollover_probe;
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
    /// Returns `false` if the shard is missing, any slot is out of `[0, row_count)`, or the region
    /// allocation/device write fails. A normal durable DML caller fails stop; it never treats false
    /// as permission to use a host representation. `slots` are local indices.
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
    ///         cleanup method contract, currently defensive). This prevents a fresh explicit-repair
    ///         generation from inheriting a stale tombstone region and stops leaks.
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
                    self.read_state
                        .residency
                        .with_shards_mut_for_table(table, |shards| {
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

    /// Mutation's sole created-by publisher: use a sealed pre-WAL Arc or lazily allocate legacy rows.
    fn get_or_alloc_created_by_region(
        &self,
        table: &str,
        shard_id: u32,
        capacity: usize,
        gpu_id: u16,
        budget_reservation_held: bool,
        preallocated_region: Option<Arc<CudaResidentDeviceMemory>>,
    ) -> Option<Arc<CudaResidentDeviceMemory>> {
        let sealed_region = preallocated_region.is_some();
        let existing = self
            .read_state
            .residency
            .shard_created_by_memory
            .get(&(table.to_string(), shard_id));
        if sealed_region && existing.is_some() {
            return None;
        }
        if let Some(region) = existing {
            return Some(region);
        }
        let _budget_allocation = (!sealed_region && !budget_reservation_held).then(|| {
            self.read_state
                .residency
                .budget_allocation_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        let existing = self
            .read_state
            .residency
            .shard_created_by_memory
            .get(&(table.to_string(), shard_id));
        if sealed_region && existing.is_some() {
            return None;
        }
        if let Some(region) = existing {
            return Some(region);
        }
        let region = match preallocated_region {
            Some(region) => region,
            None => {
                let payload = vec![
                    CREATED_BY_VISIBLE_FILL_BYTE;
                    capacity.checked_mul(std::mem::size_of::<u64>())?
                ];
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
                region
            }
        };
        self.read_state
            .residency
            .shard_created_by_memory
            .insert_shard(table, shard_id, Arc::clone(&region));
        // D4 (ADR-013 pre2): REPUBLISH the descriptor with the new region (see the
        // deleted_by twin above). Born all-visible (fill 0), so a reader observing the
        // republished descriptor mid-commit is unchanged until the stamps + row_count land.
        self.read_state
            .residency
            .with_shards_mut_for_table(table, |shards| {
                if let Some(table_shards) = shards.get_mut(table) {
                    if let Some(shard) = table_shards.iter_mut().find(|s| s.shard_id == shard_id) {
                        shard.created_by_region = Some(Arc::clone(&region));
                    }
                }
            });
        Some(region)
    }

    /// Fused i32 scatter/stamp/index apply. `None` declines; `Some(false)` requires re-admission.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_fused_apply_in_place(
        &self,
        table: &str,
        table_relation: &RelationalTable,
        shard_id: u32,
        shard_device_memory: &Arc<CudaResidentDeviceMemory>,
        chunk_offsets: &[u64],
        column_values: &super::append_source::ResidentAppendI32Columns<'_>,
        row_count: usize,
        capacity: usize,
        gpu_id: u16,
        stamps: &[Index],
        row_ids: Option<&[u64]>,
        budget_reservation_held: bool,
        sealed_created_by_region: Option<Arc<CudaResidentDeviceMemory>>,
    ) -> Option<bool> {
        let k = stamps.len();
        if k == 0
            || chunk_offsets.len() != column_values.len()
            || !column_values.all_i32_len(k)
            || row_count.saturating_add(k) > capacity
        {
            return None;
        }
        let base_row_u32 = u32::try_from(row_count).ok()?;
        // Typed plans provide the exact Arc sealed before WAL; only the legacy row path may
        // preserve the historical sparse get-or-allocate behavior.
        let created_by_region = match sealed_created_by_region {
            Some(region) => region,
            None => self.get_or_alloc_created_by_region(
                table,
                shard_id,
                capacity,
                gpu_id,
                budget_reservation_held,
                None,
            )?,
        };
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
        let mut index_owner: Option<Arc<CudaResidentDeviceMemory>> = None;
        let mut index_publication: Option<(
            Arc<std::sync::atomic::AtomicUsize>,
            Arc<std::sync::atomic::AtomicBool>,
        )> = None;
        {
            let new_count = row_count + k;
            let device_ptr = shard_device_memory.device_ptr();
            type FusedIndexBasis = (
                usize,
                Arc<CudaResidentDeviceMemory>,
                u32,
                u32,
                Arc<std::sync::atomic::AtomicUsize>,
                Arc<std::sync::atomic::AtomicBool>,
            );
            let mut live: Vec<FusedIndexBasis> = Vec::new();
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
                    live.push((
                        col_idx,
                        index,
                        entry.table_mask,
                        entry.hash_shift,
                        Arc::clone(&entry.published_row_count),
                        Arc::clone(&entry.published_has_postings),
                    ));
                }
            }
            match live.len() {
                0 => {}
                1 => {
                    let (
                        col_idx,
                        index,
                        table_mask,
                        hash_shift,
                        published_rows,
                        published_postings,
                    ) = live.pop().expect("len 1");
                    let table_size = (table_mask as u64) + 1;
                    if (new_count as u64).saturating_mul(2) > table_size {
                        // Past the builder's load rule -> drop so the next probe rebuilds at
                        // the grown size (unfused rule), then run WITHOUT index maintenance.
                        self.read_state
                            .residency
                            .purge_shard_pk_index_for_table(table);
                    } else {
                        index_publication = Some((published_rows, published_postings));
                        index_owner = Some(Arc::clone(&index));
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
        let values = column_values.flatten_i32()?;
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
        let _index_mutation = if index_col.is_some() {
            let Some(guard) = self
                .read_state
                .residency
                .begin_point_index_mutation_for_table(&self.read_state, table_relation)
            else {
                self.read_state
                    .residency
                    .purge_shard_pk_index_for_table(table);
                return Some(false);
            };
            Some(guard)
        } else {
            None
        };
        let apply_result = shard_device_memory.submit_i32_fused_apply_status(&request);
        #[cfg(test)]
        if matches!(&apply_result, Ok(status) if !status.declined) {
            self.read_state
                .residency
                .run_shard_pk_index_append_post_launch_hook();
        }
        match apply_result {
            Ok(status) => {
                if let Some(col_idx) = index_col {
                    if status.declined {
                        // The fused insert overflowed its bounded probe. Retire prepared routes before the
                        // unusable index leaves the accounted map; a later probe rebuilds safely.
                        self.read_state
                            .residency
                            .purge_shard_pk_index_for_table(table);
                    } else {
                        // The fused kernel already mutated the allocation. Publish through the
                        // basis-owned Arcs before consulting the replaceable cache entry, so a
                        // concurrent purge cannot strand retained prepared pins on stale metadata.
                        if let Some((published_rows, published_postings)) = &index_publication {
                            if status.created_posting {
                                published_postings
                                    .store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            published_rows
                                .store(row_count + k, std::sync::atomic::Ordering::Release);
                        }
                        // Post-launch entry update, mirroring the unfused path: advance the basis,
                        // while preserving the same allocation/accounting owner.
                        let mut cache = self
                            .read_state
                            .residency
                            .shard_pk_device_index
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        if let Some(entry) = cache.get_mut(&(table.to_string(), shard_id, col_idx))
                        {
                            if entry.resident_device_ptr == shard_device_memory.device_ptr()
                                && entry.row_count == row_count
                                && entry.device_index.as_ref().is_some_and(|current| {
                                    index_owner
                                        .as_ref()
                                        .is_some_and(|launched| Arc::ptr_eq(current, launched))
                                })
                            {
                                entry.has_postings |= status.created_posting;
                                entry.row_count = row_count + k;
                            }
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
                if index_col.is_some() {
                    // A failed launch may have partially mutated the index -> drop the entry
                    // (rebuild on next probe); never a wrong index. Same as the unfused path.
                    self.read_state
                        .residency
                        .purge_shard_pk_index_for_table(table);
                }
                Some(false)
            }
        }
    }

    pub(super) fn stamp_row_id_resident_shard_slots(
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

    /// Stamp created_by before publishing row_count; legacy rows may allocate its sparse sidecar here.
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
        budget_reservation_held: bool,
        sealed_created_by_region: Option<Arc<CudaResidentDeviceMemory>>,
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
        let region = match sealed_created_by_region {
            Some(region) => region,
            None => match self.get_or_alloc_created_by_region(
                table,
                shard_id,
                capacity,
                gpu_id,
                budget_reservation_held,
                None,
            ) {
                Some(region) => region,
                None => return false,
            },
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
