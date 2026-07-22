//! Snapshot construction, admission, and publication ownership.

use super::*;

impl Engine {
    /// Return a descriptive snapshot for the currently published device-authoritative generation.
    /// A sharded table deliberately has no entry in the single-buffer snapshot map, so aggregate
    /// its shard descriptors for warmup callers without copying or republishing device data.
    fn current_device_authoritative_snapshot(
        &self,
        cat: &DdlCatalogState,
        table: &str,
    ) -> Option<RelationalResidencySnapshot> {
        if let Some(entry) = self.relational_residency_entry(table) {
            let mut snapshot = (*entry.descriptor).clone();
            let pressured = self
                .router
                .runtime()
                .snapshot()
                .memory_pressured_gpu_ids
                .contains(&snapshot.gpu_id);
            snapshot.memory_pressure_active = pressured;
            if snapshot.is_valid() && entry.device_memory.is_some() && !pressured {
                return Some(snapshot);
            }
            return None;
        }

        let catalog_table = cat.relational_catalog.get(table)?;
        let pressured = self.router.runtime().snapshot().memory_pressured_gpu_ids;
        let shards = self.read_residency_shards();
        let table_shards = shards.get(table)?;
        let first = table_shards.first()?;
        if table_shards.iter().any(|shard| {
            shard.device_memory.is_none()
                || !shard.is_valid(pressured.contains(&shard.gpu_id))
                || shard.schema != catalog_table.schema
                || shard.table != catalog_table.name
        }) {
            return None;
        }
        let mut snapshot = self.resident_snapshot_for_shard(first, catalog_table);
        snapshot.row_count = table_shards.iter().map(|shard| shard.row_count).sum();
        snapshot.capacity = table_shards.iter().map(|shard| shard.capacity).sum();
        snapshot.resident_bytes = table_shards.iter().map(|shard| shard.resident_bytes).sum();
        snapshot.valid_through_index = self.read_snapshot_boundary();
        Some(snapshot)
    }

    pub fn populate_relational_residency_snapshot(
        &mut self,
        table: &str,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let gpu_id = self.planner.default_gpu_id();
        self.populate_relational_residency_snapshot_on_gpu(table, gpu_id)
    }

    pub(crate) fn populate_relational_residency_snapshot_inner(
        &self,
        cat: &mut DdlCatalogState,
        table: &str,
        gpu_id: u16,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        self.populate_relational_residency_snapshot_inner_with_boundary(cat, table, gpu_id, None)
    }

    /// Build the same authoritative generation while allowing a typed table-root replacement to
    /// stamp its actual publication index. Ordinary admission derives the boundary from the
    /// working catalog; row-only transactions retain the table's catalog generation, so a reset
    /// must supply its newer non-MVCC root boundary explicitly.
    pub(crate) fn populate_relational_residency_snapshot_inner_with_boundary(
        &self,
        cat: &mut DdlCatalogState,
        table: &str,
        gpu_id: u16,
        residency_boundary_override: Option<Index>,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let apply_leader =
            crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(std::cell::Cell::get);
        let _apply = if apply_leader {
            None
        } else {
            self.intent_lanes.as_ref().map(|lanes| {
                lanes
                    .device_apply_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            })
        };
        // R3-004: normal DML no longer keeps a host tuple-store shadow. An explicit warmup of an
        // already authoritative table therefore means "retain the current device generation", not
        // "scan the host store and overwrite it". If that generation is unavailable, fail closed;
        // only the explicit RETIRE-002 repair boundary may reverse-gather and clear elision first.
        if self.table_device_authoritative(table) {
            return self
                .current_device_authoritative_snapshot(cat, table)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "device-authoritative relation \"{table}\" has no live generation"
                    )))
                });
        }
        let previous_snapshot = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone());
        let catalog_table = cat
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();
        let named_index_publication_table = self
            .relational_named_index_publication_required(&catalog_table)
            .then(|| catalog_table.clone());
        // During a grouped committed-entry apply, `with_apply_catalog` exposes the evolving working
        // catalog at `entry.index - 1` while the public `committed_seq` intentionally remains at the
        // pre-batch boundary. Build the device generation at that working boundary so an earlier
        // entry in the same durable batch is visible to the next DML entry. Storage rejects boundary
        // zero even for an empty relation, so use one only for the physical empty scan while retaining
        // the exact catalog boundary in the published descriptor below.
        let residency_boundary =
            residency_boundary_override.unwrap_or_else(|| self.catalog_snapshot().commit_seq);
        let visibility = StorageVisibility {
            read_txn_id: residency_boundary.max(1),
        };
        let prefix = relational_key_prefix(table);
        let mut row_count = 0usize;
        let mut resident_bytes = 0u64;
        let mut resident_rows = Vec::new();
        let mut resident_row_ids: Vec<u64> = Vec::new();
        let mut resident_created_by: Vec<Index> = Vec::new();
        let mut raw_device_tail = Vec::new();
        {
            let table_rows = self.read_state.mvcc.table_rows(table);
            let mut cursor = table_rows.store().seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                raw_device_tail.extend_from_slice(tuple.key.as_bytes());
                raw_device_tail.extend_from_slice(tuple.value.as_bytes());
                let decoded = decode_relational_row(&tuple.value, &catalog_table.columns)?;
                // RETIREMENT A1: the row's host identity, parsed from its key (sentinel on any
                // malformed key — identity unknown is safe, wrong identity is not). Collected only
                // when the sharded branch (the sole consumer) is reachable (audit finding 3).
                resident_row_ids
                    .push(parse_relational_row_id(&tuple.key, &prefix).unwrap_or(u64::MAX));
                resident_created_by.push(tuple.created_by);
                row_count += 1;
                resident_bytes = resident_bytes
                    .saturating_add(tuple.key.len() as u64)
                    .saturating_add(
                        decoded
                            .iter()
                            .map(relational_resident_value_bytes)
                            .sum::<u64>(),
                    );
                resident_rows.push(decoded);
            }
        }
        // int4 AND date columns share the i32 section: a `date` is physically an i32 (days since
        // 2000-01-01), so it rides the int4 residency layout + the i32 compare kernels (the type
        // matrix, doc 19). The catalog type distinguishes them for lowering/projection.
        // int4, date AND int2 share the i32 section: a `date` is i32 days and a `smallint` widens to
        // i32, so both ride the int4 residency layout + compare path (the type matrix, doc 19).
        let resident_device_int4_columns = catalog_table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        // int8 AND timestamp columns share the i64 section: a `timestamp` is i64 microseconds, so it
        // rides the int8 residency layout + the i64 compare kernels (the type matrix, doc 19).
        let resident_device_int8_columns = catalog_table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_numeric_columns = catalog_table
            .columns
            .iter()
            // numeric AND uuid share the 16-byte section: a `uuid` is 16 raw bytes (a byte-wise
            // compare kernel reads them; numeric stores its i128 mantissa). The type matrix, doc 19.
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
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
        // Slice 1b-ii: a PURELY-int4 table is laid down as an OPEN shard with capacity headroom (~2x
        // rows, power-of-two) so committed INSERTs append in place (amortized O(1)/row) instead of
        // re-uploading the whole table every commit. R3-004 deliberately includes an EMPTY table:
        // its first write must append to an already-authoritative device generation, not bootstrap a
        // host tuple store and re-admit from it after the commit.
        let purely_int4 = row_count < (1usize << 29)
            && !column_types.is_empty()
            && column_types
                .iter()
                .all(|ty| matches!(ty, SqlType::Int4 | SqlType::Date | SqlType::Int2));
        // TYPE-COVERAGE track 2 slice 2 (i64 sections, default-OFF flag): a FIXED-WIDTH-SECTION
        // table (every column i32- or i64-section) shard-admits like the purely-i32 shape — the
        // payload builder already lays the i64 section after the i32 ones, capacity-strided, so
        // the same headroom/rollover story applies. Everything below that branches on
        // `purely_int4 || fixed_width_sections` treats both shapes identically EXCEPT
        // appendability, which stage (ii) widens (int8-bearing shards decline appends -> writes
        // re-admit until then).
        let fixed_width_sections = !purely_int4
            && self.shard_int8_section_enabled()
            && row_count < (1usize << 29)
            && !column_types.is_empty()
            && column_types.iter().all(|ty| {
                matches!(
                    ty,
                    SqlType::Int4
                        | SqlType::Date
                        | SqlType::Int2
                        | SqlType::Int8
                        | SqlType::Timestamp
                        // TYPE-COVERAGE #14 (numeric): the b128 (16-byte) section rides the payload
                        // after the i64 sections; Numeric + Uuid share it (same fixed width).
                        | SqlType::Numeric { .. }
                        | SqlType::Uuid
                        // TYPE-COVERAGE #14 (bool): the 1-bit/row bitmap section rides the payload
                        // after the b128 sections; the incremental append sets its bits on-device.
                        | SqlType::Bool
                )
            });
        // TYPE-COVERAGE #14/R3-004: every supported relation with a TEXT column shard-admits as a
        // DENSE shard (capacity == row_count — text has no capacity-strided headroom). Writes roll a
        // fresh dense shard, which keeps the mutation path device-native independently of indexes.
        let text_sectioned = !purely_int4
            && !fixed_width_sections
            && self.shard_int8_section_enabled()
            && row_count < (1usize << 29)
            && !column_types.is_empty()
            && column_types.iter().any(|ty| matches!(ty, SqlType::Text))
            && column_types.iter().all(|ty| {
                matches!(
                    ty,
                    SqlType::Int4
                        | SqlType::Date
                        | SqlType::Int2
                        | SqlType::Int8
                        | SqlType::Timestamp
                        | SqlType::Numeric { .. }
                        | SqlType::Uuid
                        | SqlType::Bool
                        | SqlType::Text
                )
            });
        // A PURELY-int4 table is laid down with capacity HEADROOM (~2x rows, power-of-two) so committed
        // INSERTs append in place (1b-ii). S-d2: the sharded read is now capacity-aware (the recompaction
        // gather + `resident_snapshot_for_shard` stride by `shard.capacity`), so the OPEN shard gets the
        // same headroom as the single buffer. Other shapes (and huge/empty tables) stay dense.
        let capacity = if row_count == 0 {
            // An empty authoritative generation needs only its descriptor/header. Giving it the
            // normal 262k-row open-shard floor consumes megabytes before the first write and can
            // make a correctly configured small STRATA budget impossible to establish.
            0
        } else if purely_int4 || fixed_width_sections {
            let doubled = row_count.saturating_mul(2).next_power_of_two();
            // S-d2c: on the shard path, CAP the open shard at the target size (`row_count` if it already
            // exceeds it — a large admit is one dense shard) so it seals + rolls over at the target rather
            // than growing unbounded. The single buffer is uncapped (its cap is the 536M guard above).
            if self.shard_residency_enabled() {
                // E2.5b-2: FIRST-CAPACITY FLOOR (restores the archived W1b lesson).
                // Without it the open shard is born at ~2x the warm-up rows and
                // RE-ADMITS geometrically as it fills — each re-admission is a
                // full device gather+re-upload (hundreds of ms at multi-M rows),
                // measured as the periodic stalls capping sustained lane TPS.
                let floor: usize = if self.relational_residency_budget_bytes(gpu_id).is_some() {
                    // A configured STRATA budget is an explicit bound. Do not reserve the
                    // throughput-oriented 262k-row growth floor inside a small bounded working set.
                    1
                } else {
                    std::env::var("GPU_DB_OPEN_SHARD_FLOOR_ROWS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(262_144)
                };
                doubled
                    .max(floor.next_power_of_two())
                    .min(self.shard_size_target())
                    .max(row_count)
            } else {
                doubled
            }
        } else {
            row_count
        };
        let (
            mut device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload_with_capacity(
            &column_names,
            &column_types,
            &resident_rows,
            capacity,
        )?;
        // The dead MVCC tail rides only the dense (sealed) payload; an OPEN shard (capacity > row_count)
        // omits it (no kernel reads it) so the section headroom an append writes into stays clean.
        if capacity == row_count {
            device_payload.extend_from_slice(&raw_device_tail);
        }
        // SV1/SV2 (sparse-versioning): NO version metadata rides the shard payload. `created_by` is gone
        // (SV1 — returns as a zone map + boundary under SI), and `deleted_by` is now ON-DEMAND: a shard is
        // born delete-free with NO tombstone region; its `deleted_by` region (a separate device buffer in
        // `shard_deleted_by_memory`) is allocated on the shard's FIRST delete. So a delete-free / cold shard
        // pays ZERO version overhead (the HyPer property). See docs/proposals/sparse-mvcc-version-metadata.md.

        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let row_id_payload = Some({
            let mut payload =
                vec![ROW_ID_UNSTAMPED_FILL_BYTE; capacity * std::mem::size_of::<u64>()];
            for (slot, row_id) in resident_row_ids.iter().enumerate() {
                payload[slot * 8..slot * 8 + 8].copy_from_slice(&row_id.to_le_bytes());
            }
            payload
        });
        // R3-003 device conflict state: a re-admission normally collapses live rows to an all-
        // visible generation. While an older transaction exists, preserve creation stamps newer
        // than its boundary so a commit-time device verdict can still distinguish "same value"
        // from "written since BEGIN" after a rebuild. No active old boundary (or no newer row)
        // means no sidecar and zero steady-state sparse-version overhead.
        let oldest_active = self.active_snapshots_oldest();
        let created_by_payload = oldest_active
            .filter(|oldest| resident_created_by.iter().any(|created| created > oldest))
            .map(|_| {
                let mut payload =
                    vec![CREATED_BY_VISIBLE_FILL_BYTE; capacity * std::mem::size_of::<u64>()];
                for (slot, created_by) in resident_created_by.iter().enumerate() {
                    payload[slot * 8..slot * 8 + 8].copy_from_slice(&created_by.to_le_bytes());
                }
                payload
            });

        // S-F/R-1: allocate the complete mandatory replacement set BEFORE selecting or removing
        // an evictee. Allocation failure therefore leaves every published resident generation
        // untouched. The same lock is used by lazy index publication, making the budget preflight
        // and the final descriptor/index publication one serialized accounting transaction.
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut device_memory = self.relational_residency_device_memory(gpu_id, &device_payload);
        #[cfg(not(test))]
        if device_memory.is_none() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{table}\" GPU {gpu_id} residency allocation failed before admission"
            ))));
        }
        let admitted_row_id_region = if device_memory.is_some() {
            row_id_payload.as_ref().and_then(|payload| {
                self.relational_residency_device_memory(gpu_id, payload)
                    .map(Arc::new)
            })
        } else {
            None
        };
        let admitted_created_by_region = if device_memory.is_some() {
            created_by_payload.as_ref().and_then(|payload| {
                self.relational_residency_device_memory(gpu_id, payload)
                    .map(Arc::new)
            })
        } else {
            None
        };
        #[cfg(not(test))]
        if row_id_payload
            .as_ref()
            .is_some_and(|payload| !payload.is_empty())
            && admitted_row_id_region.is_none()
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{table}\" GPU {gpu_id} mandatory row-identity allocation failed before admission"
            ))));
        }
        #[cfg(not(test))]
        if created_by_payload.is_some() && admitted_created_by_region.is_none() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{table}\" GPU {gpu_id} transaction conflict-stamp allocation failed before admission"
            ))));
        }
        let allocated_payload_bytes = device_memory
            .as_ref()
            .map_or(device_payload.len() as u64, |memory| {
                memory.metadata().allocated_bytes
            });
        let allocated_row_id_bytes = admitted_row_id_region
            .as_ref()
            .map_or(0, |memory| memory.metadata().allocated_bytes);
        let allocated_created_by_bytes = admitted_created_by_region
            .as_ref()
            .map_or(0, |memory| memory.metadata().allocated_bytes);
        let named_index_bytes = if named_index_publication_table.is_some() {
            estimated_named_index_bytes_for_shard(&catalog_table, row_count, capacity).ok_or_else(
                || {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{table}\" named-index allocation geometry is unsupported"
                    )))
                },
            )?
        } else {
            0
        };
        let admitted_allocated_bytes = allocated_payload_bytes
            .saturating_add(allocated_row_id_bytes)
            .saturating_add(allocated_created_by_bytes)
            .saturating_add(named_index_bytes);
        let admission_budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let (evicted_tables_on_admission, resident_bytes_after_admission) = self
            .admit_relational_residency_snapshot_inner(
                cat,
                table,
                gpu_id,
                admitted_allocated_bytes,
            )?;
        let device_memory_proof = device_memory
            .as_ref()
            .map(|device_memory| device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            // headroom for an open shard (== row_count for the dense/sealed path); Slice 1b-ii.
            capacity,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: residency_boundary,
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: memory_pressure_active,
            memory_pressure_active,
            last_refresh_cost: previous_snapshot.as_ref().map(|previous| {
                RelationalResidencyRefreshCost {
                    previous_row_count: previous.row_count,
                    refreshed_row_count: row_count,
                    row_delta: row_count as i128 - previous.row_count as i128,
                    previous_resident_bytes: previous.resident_bytes,
                    refreshed_resident_bytes: resident_bytes,
                    resident_byte_delta: resident_bytes as i128 - previous.resident_bytes as i128,
                    refreshed_from_index: previous.valid_through_index,
                    refreshed_through_index: residency_boundary,
                    invalidated_by_txn_id: previous.invalidated_by_txn_id,
                    invalidated_at_index: previous.invalidated_at_index,
                    invalidated_by_memory_pressure: previous.invalidated_by_memory_pressure,
                }
            }),
            admission_budget_bytes,
            resident_bytes_after_admission,
            evicted_tables_on_admission,
            device_memory_proof,
        };
        let read_state = Arc::clone(&self.read_state);
        // Billions-of-rows segmented layout (S-d1; DEFAULT ON since THE FLIP): admit as a SEGMENTED shard
        // list — routed through the sharded resident read path, instead of the single capacity-padded
        // unified buffer (which caps at ~536M rows and re-admits O(table)). The single dense shard reuses
        // the SAME columnar payload + layout the single buffer uses (header at offset 0, dense columns), so
        // the (already tested) sharded read path reads it identically. Requires GPU device memory; without
        // it (no GPU) admission fails through the GPU-required boundary.
        //
        // THE FLIP scopes sharded admission to PURELY-int4-section tables (int4/int2/date — `purely_int4`
        // above): the shard read stack (unified exec source, index routes, dense kernels) is int4-only
        // today, so sharding a MIXED-type table would DEMOTE its text/int8/numeric shapes from the proven
        // single-buffer GPU paths to unsupported-route failures — the opposite of the flip's goal
        // (caught by the single-buffer text-probe suite). Mixed-type tables keep the single-buffer
        // layout until shards carry every section (type-coverage ledger item).
        let shard_device_memory = if self.shard_residency_enabled()
            && (purely_int4 || fixed_width_sections || text_sectioned)
        {
            device_memory.take()
        } else {
            None
        };
        if let Some(device_memory) = shard_device_memory {
            // Audit (S-d1) fix: this re-admit makes the SHARD representation authoritative — clear any prior
            // single-buffer cell for the table so a runtime flag flip (OFF->ON) cannot leave a stale
            // snapshot/device_memory shadowing the shards. Idempotent (a no-op when none exists).
            read_state.residency.with_snapshots_mut(|snapshots| {
                snapshots.remove(table);
            });
            read_state.residency.device_memory.remove(table);
            // SV4 prereq #1 (lifecycle): this SHARDED re-admit installs a FRESH all-live shard 0, but a
            // warmup/refresh (`populate_relational_residency_snapshot_on_gpu`) reaches here with NO preceding
            // invalidate -- so erase any stale `deleted_by` regions for the table (keyed by the reused
            // shard_id) or the fresh shard would inherit them (SV4 wrong-results). Symmetric to the
            // single-buffer path below. INERT until SV4 (no region exists today).
            read_state
                .residency
                .shard_deleted_by_memory
                .remove_table(table);
            // SV6: erase stale `created_by` regions symmetrically -- a fresh all-live shard 0 inheriting a
            // stale stamp region would wrongly HIDE rebuilt rows from older-snapshot readers.
            read_state
                .residency
                .shard_created_by_memory
                .remove_table(table);
            if let Some(region) = &admitted_created_by_region {
                read_state.residency.shard_created_by_memory.insert_shard(
                    table,
                    0,
                    Arc::clone(region),
                );
            }
            // RETIREMENT A1: replace the row-identity regions with this rebuild's already-allocated
            // capacity-sized region. Allocation happened before budget eviction, so a failure could
            // not strand the resident set in a partially-evicted state.
            read_state.residency.shard_row_id_memory.remove_table(table);
            if let Some(region) = &admitted_row_id_region {
                read_state
                    .residency
                    .shard_row_id_memory
                    .insert_shard(table, 0, Arc::clone(region));
            }
            // Sub-slice 3b: this sharded re-admit replaces the table's shards -> purge stale cached indexes.
            read_state.residency.purge_shard_pk_index_for_table(table);
            let dm = Arc::new(device_memory);
            let shard = RelationalResidentShard {
                shard_id: 0,
                row_start: 0,
                row_count,
                // This dense rebuild materializes only the current live image. Physical versions
                // removed at/before this boundary cannot affect a writer at or above it; older
                // writers must decline a history miss.
                history_floor_index: snapshot.valid_through_index,
                // S-d2: the OPEN shard carries headroom (capacity > row_count for int4); the recompaction
                // gather + offset helpers stride by this capacity. (Dead MVCC tail omitted when padded.)
                capacity,
                // S-d2b/A4e: append-eligible iff purely int4 (no text / int8 / numeric / bool /
                // NULL sections). NOT gated on headroom: a DENSE purely-int4 open shard (S-d2c's
                // "large admit is one dense shard") must reach the append fn's ROLLOVER branch —
                // the in-place branch checks headroom itself. Gating headroom here made every
                // bulk-admitted lineage decline appends outright -> O(table) re-admit per commit.
                // TYPE-COVERAGE #14: `int4_appendable` = "participates in the append/rollover machinery"
                // (NOT necessarily in-place). Fixed-width (i32/i64/b128) + bool append IN PLACE into
                // headroom; TEXT tables are admitted DENSE (capacity == row_count) so they never fit
                // in place -> every commit ROLLS OVER a fresh dense text shard (the rollover-only model).
                // NULL-bearing shards are DENSE but still participate: the append path observes
                // their validity layout and rolls a fresh dense bitmap-bearing shard. Marking them
                // non-appendable here made that safe rollover branch unreachable and forced every
                // nullable commit through O(table) invalidate/re-admit.
                int4_appendable: snapshot.column_count
                    == snapshot.resident_device_int4_columns.len()
                        + snapshot.resident_device_int8_columns.len()
                        + snapshot.resident_device_numeric_columns.len()
                        + snapshot.resident_device_bool_columns.len()
                        + snapshot.resident_device_text_columns.len(),
                // S-d3: the zone map (min/max per int4 column) for shard pruning.
                resident_device_int4_column_stats: snapshot
                    .resident_device_int4_column_stats
                    .clone(),
                resident_bytes,
                allocated_bytes: allocated_payload_bytes,
                count_header_byte_offset: 0,
                resident_device_int4_columns: snapshot.resident_device_int4_columns.clone(),
                // TYPE-COVERAGE track 2 slice 2: the i64 section rides the same payload.
                resident_device_int8_columns: snapshot.resident_device_int8_columns.clone(),
                // TYPE-COVERAGE #14 (numeric): the b128 (Numeric/Uuid) section rides the same payload.
                resident_device_numeric_columns: snapshot.resident_device_numeric_columns.clone(),
                // TYPE-COVERAGE #14 (bool): the bool bitmaps ride the same single-buffer payload;
                // offsets are relative to it (which this shard's device buffer is), so they carry directly.
                resident_device_bool_columns: snapshot.resident_device_bool_columns.clone(),
                resident_device_text_columns: snapshot.resident_device_text_columns.clone(),
                // M3-for-shards: carry the payload's per-column NULL validity bitmaps so the sharded scan's
                // recompaction can rebuild them into the unified buffer. Nullable append rollover
                // builds the same layouts for its fresh dense shard; synthetic benchmark shards remain
                // NULL-free.
                resident_device_null_columns: snapshot.resident_device_null_columns.clone(),
                gpu_id,
                schema: snapshot.schema.clone(),
                table: snapshot.table.clone(),
                point_route_generation: Arc::new(()),
                device_memory_proof: snapshot.device_memory_proof.clone(),
                invalidated_by_txn_id: None,
                invalidated_at_index: None,
                invalidated_by_memory_pressure: memory_pressure_active,
                memory_pressure_active,
                // D4/R3-003: resources ride the descriptor. Re-admission is normally all-live;
                // while an older transaction exists, the compact created_by sidecar preserves
                // newer conflict/visibility stamps across the rebuild.
                device_memory: Some(Arc::clone(&dm)),
                deleted_by_region: None,
                created_by_region: admitted_created_by_region,
                row_id_region: admitted_row_id_region,
                max_created_by: resident_created_by.iter().copied().max().unwrap_or(0),
            };
            let mut shard_memory = BTreeMap::new();
            shard_memory.insert(0_u32, dm);
            cat.relational_resident_cache.install_shards(
                catalog_table.name.clone(),
                vec![shard],
                shard_memory,
                &read_state.residency,
            );
            // PRODUCT-002: once a catalog shape has declared named indexes mandatory, an ordinary
            // repair/re-admission cannot publish a replacement payload without republishing the full
            // device-index set in the same budget transaction. On a CUDA/semantic failure, mark the
            // replacement invalid before returning the explicit error; no reader may bind a partial set.
            if let Some(named_index_table) = named_index_publication_table
                .as_ref()
                .filter(|_| row_count != 0)
            {
                let current_shards = read_state.residency.shards.load_full();
                let table_shards = current_shards.get(table).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident index re-publication lost relation \"{table}\""
                    )))
                })?;
                if let Err(error) = self.publish_relational_resident_indexes_for_generation(
                    named_index_table,
                    table_shards,
                    residency_boundary,
                    true,
                    true,
                    true,
                ) {
                    read_state.residency.purge_shard_pk_index_for_table(table);
                    read_state.residency.flag_table_descriptors_invalidated(
                        table,
                        residency_boundary,
                        residency_boundary,
                    );
                    return Err(error);
                }
            }
            return Ok(snapshot);
        }
        // Audit (S-d1) fix: the single-buffer path is authoritative here — clear any prior SHARD cell for
        // the table so a flag flip (ON->OFF) cannot leave a stale shard shadowing the fresh snapshot (the
        // read route checks shards FIRST, so a still-valid stale shard would serve wrong rows). Idempotent.
        read_state
            .residency
            .with_shards_mut_for_table(table, |shards| {
                shards.remove(table);
            });
        read_state.residency.shard_device_memory.remove_table(table);
        // SV4 prereq #1 (lifecycle): the single-buffer path replaces the table's shards, so clear any stale
        // `deleted_by` regions -- a flag flip / re-admit must not leave a tombstone region shadowing the fresh
        // all-live buffer (wrong-results guard). INERT until SV4 (no region exists today).
        read_state
            .residency
            .shard_deleted_by_memory
            .remove_table(table);
        // SV6: clear stale `created_by` regions symmetrically (same wrong-results guard).
        read_state
            .residency
            .shard_created_by_memory
            .remove_table(table);
        if let Some(region) = &admitted_created_by_region {
            read_state
                .residency
                .shard_created_by_memory
                .insert_shard(table, 0, Arc::clone(region));
        }
        // R3-004: a single-buffer relation keeps the same device row-identity sidecar contract as
        // shard 0. This is write metadata only; the established single-buffer read facade is unchanged.
        read_state.residency.shard_row_id_memory.remove_table(table);
        if let Some(region) = &admitted_row_id_region {
            read_state
                .residency
                .shard_row_id_memory
                .insert_shard(table, 0, Arc::clone(region));
        }
        // Sub-slice 3b: the single-buffer path replaces the table's shards -> purge stale cached indexes.
        read_state.residency.purge_shard_pk_index_for_table(table);
        cat.relational_resident_cache.install_snapshot(
            catalog_table.name,
            snapshot.clone(),
            device_memory,
            &read_state.residency,
        );
        Ok(snapshot)
    }

    fn admit_relational_residency_snapshot_inner(
        &self,
        cat: &mut DdlCatalogState,
        table: &str,
        gpu_id: u16,
        resident_bytes: u64,
    ) -> Result<(Vec<String>, u64), ExecuteError> {
        let Some(budget_bytes) = self.relational_residency_budget_bytes(gpu_id) else {
            let resident_bytes_after_admission = self
                .relational_resident_bytes_for_gpu_excluding(gpu_id, table)
                .saturating_add(resident_bytes);
            cat.relational_resident_cache
                .record_decision(RelationalResidentCacheDecision {
                    table: table.to_string(),
                    gpu_id,
                    accepted: true,
                    reason: "admitted without budget limit".to_string(),
                    resident_bytes,
                    budget_bytes: None,
                    current_bytes_before: resident_bytes_after_admission
                        .saturating_sub(resident_bytes),
                    current_bytes_after: resident_bytes_after_admission,
                    evicted_tables: Vec::new(),
                });
            return Ok((Vec::new(), resident_bytes_after_admission));
        };
        if resident_bytes > budget_bytes {
            let current_bytes = self.relational_resident_bytes_for_gpu(gpu_id);
            cat.relational_resident_cache
                .record_decision(RelationalResidentCacheDecision {
                    table: table.to_string(),
                    gpu_id,
                    accepted: false,
                    reason: "resident snapshot exceeds GPU budget".to_string(),
                    resident_bytes,
                    budget_bytes: Some(budget_bytes),
                    current_bytes_before: current_bytes,
                    current_bytes_after: current_bytes,
                    evicted_tables: Vec::new(),
                });
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{table}\" resident snapshot requires {resident_bytes} bytes, exceeding GPU {gpu_id} residency budget {budget_bytes} bytes"
            ))));
        }

        let current_bytes = self.relational_resident_bytes_for_gpu_excluding(gpu_id, table);
        let current_bytes_before = current_bytes;
        if current_bytes.saturating_add(resident_bytes) <= budget_bytes {
            cat.relational_resident_cache
                .record_decision(RelationalResidentCacheDecision {
                    table: table.to_string(),
                    gpu_id,
                    accepted: true,
                    reason: "admitted within budget".to_string(),
                    resident_bytes,
                    budget_bytes: Some(budget_bytes),
                    current_bytes_before,
                    current_bytes_after: current_bytes + resident_bytes,
                    evicted_tables: Vec::new(),
                });
            return Ok((Vec::new(), current_bytes + resident_bytes));
        }

        // An active transaction owns exact resident Arc generations. Evicting their global map
        // entries would leave the still-live allocations outside global accounting because private
        // COW accounting deliberately excludes base pointers. Pinned tables are therefore not
        // evictable; this preserves both snapshot correctness and the hard physical GPU budget.
        let pinned_tables = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .transactions
            .values()
            .flat_map(|snapshot| {
                snapshot
                    .resident_snapshots
                    .keys()
                    .chain(snapshot.resident_shards.keys())
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect::<BTreeSet<_>>();
        let mut candidates: BTreeMap<String, u64> = self
            .read_state
            .residency
            .snapshots
            .load()
            .iter()
            .filter(|(name, entry)| {
                name.as_str() != table
                    && entry.descriptor.gpu_id == gpu_id
                    && !pinned_tables.contains(name.as_str())
            })
            .map(|(name, entry)| (name.clone(), entry.descriptor.valid_through_index))
            .collect();
        for (name, shards) in self.read_state.residency.shards.load().iter() {
            if name == table
                || pinned_tables.contains(name.as_str())
                || !shards.iter().any(|shard| shard.gpu_id == gpu_id)
            {
                continue;
            }
            let age = shards
                .iter()
                .filter(|shard| shard.gpu_id == gpu_id)
                .map(|shard| shard.max_created_by)
                .max()
                .unwrap_or(0);
            candidates.entry(name.clone()).or_insert(age);
        }
        let mut candidates: Vec<(u64, String)> = candidates
            .into_iter()
            .map(|(name, age)| (age, name))
            .collect();
        candidates.sort();
        // Select the complete deterministic eviction prefix without mutating the published
        // resident set. `resident_bytes` has already been allocated successfully by the caller.
        // If the prefix cannot make enough room, the operation declines with zero evictions.
        let mut projected_bytes = current_bytes;
        let mut evicted_tables = Vec::new();
        for (_age, map_key) in candidates {
            if projected_bytes.saturating_add(resident_bytes) <= budget_bytes {
                break;
            }
            // Retired compound point plans may still be pinned by an in-flight execution or a
            // previously loaded route snapshot. They remain charged until their last owner
            // drains, so table eviction cannot promise those bytes as immediately reclaimable.
            let candidate_bytes = self
                .relational_resident_table_bytes_for_gpu(&map_key, gpu_id)
                .saturating_sub(self.live_compound_point_route_bytes_for_table(gpu_id, &map_key));
            if candidate_bytes == 0 {
                continue;
            }
            projected_bytes = projected_bytes.saturating_sub(candidate_bytes);
            evicted_tables.push(map_key);
        }
        if projected_bytes.saturating_add(resident_bytes) > budget_bytes {
            cat.relational_resident_cache
                .record_decision(RelationalResidentCacheDecision {
                    table: table.to_string(),
                    gpu_id,
                    accepted: false,
                    reason: "insufficient evictable GPU residency budget".to_string(),
                    resident_bytes,
                    budget_bytes: Some(budget_bytes),
                    current_bytes_before,
                    current_bytes_after: current_bytes,
                    evicted_tables: Vec::new(),
                });
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{table}\" cannot be admitted within GPU {gpu_id} residency budget {budget_bytes} bytes from the available eviction set"
            ))));
        }

        // No fallible work remains after this point: the replacement buffers are retained and the
        // full eviction prefix is known to fit. Publish the retirements, then the replacement.
        let read_state = Arc::clone(&self.read_state);
        for map_key in &evicted_tables {
            cat.relational_resident_cache.remove_table(
                map_key,
                &read_state.residency,
                &read_state.route_telemetry,
            );
        }

        cat.relational_resident_cache
            .record_decision(RelationalResidentCacheDecision {
                table: table.to_string(),
                gpu_id,
                accepted: true,
                reason: if evicted_tables.is_empty() {
                    "admitted within budget".to_string()
                } else {
                    "admitted after deterministic eviction".to_string()
                },
                resident_bytes,
                budget_bytes: Some(budget_bytes),
                current_bytes_before,
                current_bytes_after: projected_bytes + resident_bytes,
                evicted_tables: evicted_tables.clone(),
            });
        Ok((evicted_tables, projected_bytes + resident_bytes))
    }

    /// `&mut self` entry for the operator warm path: acquire the catalog latch, then run the
    /// `&self`+held-guard producer (the STRATA S-B seam — also reachable from the `&self` commit path).
    pub(super) fn populate_relational_residency_snapshot_on_gpu(
        &mut self,
        table: &str,
        gpu_id: u16,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let mut guard = self.ddl_catalog();
        self.populate_relational_residency_snapshot_inner(&mut guard, table, gpu_id)
    }

    /// `&self` residency admission for lazy optimized-route preparation. Same latch + producer as
    /// the operator warm path.
    pub(crate) fn populate_relational_residency_snapshot_shared(
        &self,
        table: &str,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let gpu_id = self.planner.default_gpu_id();
        let mut guard = self.ddl_catalog();
        self.populate_relational_residency_snapshot_inner(&mut guard, table, gpu_id)
    }

    /// Catalog-latched entry for the benchmark residency installers. `&self` lets the caller keep
    /// the shared allocation transaction held through allocate → admit → publish.
    pub(super) fn admit_relational_residency_snapshot(
        &self,
        table: &str,
        gpu_id: u16,
        resident_bytes: u64,
    ) -> Result<(Vec<String>, u64), ExecuteError> {
        let mut guard = self.ddl_catalog();
        self.admit_relational_residency_snapshot_inner(&mut guard, table, gpu_id, resident_bytes)
    }
}
