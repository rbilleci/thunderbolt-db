//! GPU residency management + resident-route planning (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for populating/admitting
//! resident snapshots (incl. on-GPU), the benchmark chunk/shard installs,
//! resident device-memory + bytes accounting, retained-read snapshot handles,
//! warmup/maintenance policy execution, and the resident-route planners
//! (plan_relational_resident_route + sharded variant) + residency status.

use super::*;

/// Build the GPU device payload for typed columns + their row values -- the columnar
/// `[8-byte row_count header][int4/date/int2 i32][int8/timestamp i64][numeric/uuid 16B][bool bitmap]
/// [text 8-aligned offsets + bytes]` layout (type-grouped, catalog order within each type; varlen text
/// offsets 8-aligned per the CUDA-716 lesson). Returns the payload (row_count header filled, NO MVCC
/// tail) + the text/bool column layouts + per-int4-column min/max -- the SAME bytes/offsets the
/// resident-table builder produces, so a non-table caller (e.g. the grouped-sort) can build a
/// resident-like buffer without re-implementing the byte mappings. `column_names` / `column_types` /
/// each row in `rows` are parallel by column index.
#[allow(clippy::type_complexity)]
pub(crate) fn build_relational_device_payload(
    column_names: &[String],
    column_types: &[SqlType],
    rows: &[Vec<SqlValue>],
) -> Result<
    (
        Vec<u8>,
        Vec<ResidentDeviceTextColumnLayout>,
        Vec<ResidentDeviceBoolColumnLayout>,
        Vec<ResidentDeviceInt4ColumnStats>,
        // (column name, byte-offset) of each numeric/uuid 16-byte section -- so a non-table caller
        // (the GPU grouped-sort) can address them without recomputing the layout by formula.
        Vec<(String, u64)>,
        // Per-column NULL validity bitmaps (M3 — doc 21), one per column that contains a NULL.
        Vec<ResidentDeviceNullBitmapLayout>,
    ),
    ExecuteError,
> {
    let row_count = rows.len();
    let mut device_payload = vec![0u8; std::mem::size_of::<u64>()];
    let mut resident_device_text_columns = Vec::new();
    let mut resident_device_bool_columns = Vec::new();
    let mut resident_device_int4_column_stats = Vec::new();
    let mut resident_device_b128_columns: Vec<(String, u64)> = Vec::new();
    let mut resident_device_null_columns: Vec<ResidentDeviceNullBitmapLayout> = Vec::new();

    // int4 / date / int2 share the i32 section (a date is i32 days; a smallint widens to i32).
    for col_idx in column_types
        .iter()
        .enumerate()
        .filter(|&(_i, ty)| matches!(ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
        .map(|(i, _)| i)
    {
        let mut min = i32::MAX;
        let mut max = i32::MIN;
        for row in rows {
            // A NULL writes a don't-care 0 placeholder (the validity bitmap marks the row; the kernels
            // skip it) and is EXCLUDED from min/max so it can't pull the stats toward 0.
            let is_null = matches!(row[col_idx], SqlValue::Null);
            let value: i32 = match row[col_idx] {
                SqlValue::Int4(value) | SqlValue::Date(value) => value,
                SqlValue::Int2(value) => i32::from(value),
                SqlValue::Null => 0,
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot int4/date/int2 payload encountered a non-i32 value"
                            .to_string(),
                    )));
                }
            };
            if !is_null {
                min = min.min(value);
                max = max.max(value);
            }
            device_payload.extend_from_slice(&value.to_le_bytes());
        }
        resident_device_int4_column_stats.push(ResidentDeviceInt4ColumnStats {
            name: column_names[col_idx].clone(),
            min,
            max,
        });
    }
    // int8 / timestamp share the i64 section (a timestamp is i64 microseconds).
    for col_idx in column_types
        .iter()
        .enumerate()
        .filter(|&(_i, ty)| matches!(ty, SqlType::Int8 | SqlType::Timestamp))
        .map(|(i, _)| i)
    {
        for row in rows {
            // NULL → a don't-care 0 placeholder (the validity bitmap marks the row).
            let value: i64 = match row[col_idx] {
                SqlValue::Int8(value) | SqlValue::Timestamp(value) => value,
                SqlValue::Null => 0,
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot int8/timestamp payload encountered a non-i64 value"
                            .to_string(),
                    )));
                }
            };
            device_payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    // numeric / uuid share the 16-byte section (numeric = i128 mantissa LE; uuid = raw 16 bytes).
    for col_idx in column_types
        .iter()
        .enumerate()
        .filter(|&(_i, ty)| matches!(ty, SqlType::Numeric { .. } | SqlType::Uuid))
        .map(|(i, _)| i)
    {
        let section_byte_offset = device_payload.len() as u64;
        for row in rows {
            match &row[col_idx] {
                SqlValue::Numeric(value) => {
                    device_payload.extend_from_slice(&value.mantissa.to_le_bytes());
                }
                SqlValue::Uuid(bytes) => device_payload.extend_from_slice(bytes),
                // NULL → 16 don't-care zero bytes (the validity bitmap marks the row).
                SqlValue::Null => device_payload.extend_from_slice(&[0u8; 16]),
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot numeric/uuid payload encountered a wrong-typed value"
                            .to_string(),
                    )));
                }
            }
        }
        resident_device_b128_columns.push((column_names[col_idx].clone(), section_byte_offset));
    }
    // bool -> a 1-bit-per-row bitmap (ceil(row_count/32) LE u32 words, bit i = row i, LSB-first).
    for col_idx in column_types
        .iter()
        .enumerate()
        .filter(|&(_i, ty)| matches!(ty, SqlType::Bool))
        .map(|(i, _)| i)
    {
        let bitmap_byte_offset = device_payload.len() as u64;
        let mut words = vec![0u32; row_count.div_ceil(32)];
        for (i, row) in rows.iter().enumerate() {
            match row[col_idx] {
                // NULL leaves the value bit 0 (don't-care; the validity bitmap marks the row).
                SqlValue::Bool(true) => words[i / 32] |= 1u32 << (i % 32),
                SqlValue::Bool(false) | SqlValue::Null => {}
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot bool payload encountered a non-bool value".to_string(),
                    )));
                }
            }
        }
        for word in &words {
            device_payload.extend_from_slice(&word.to_le_bytes());
        }
        resident_device_bool_columns.push(ResidentDeviceBoolColumnLayout {
            name: column_names[col_idx].clone(),
            bitmap_byte_offset,
        });
    }
    // NULL validity bitmaps (M3 — doc 21): ONE 1-bit-per-row bitmap (1 = valid/present, 0 = NULL,
    // LSB-first u32 words like bool) per column that actually contains a NULL. A no-NULL column emits
    // nothing — absence ⇒ all-valid — so existing non-null payloads stay byte-identical. Placed after
    // the bool section (every preceding section is a multiple of 4 bytes ⇒ this section start is
    // 4-aligned, so the u32 words load safely) and before text (text records its own offset, so it just
    // starts later). Iterates ALL columns in catalog order — a NULL can appear in any type, its value
    // riding the don't-care placeholder its own typed section wrote above.
    for (col_idx, name) in column_names.iter().enumerate() {
        if !rows.iter().any(|row| matches!(row[col_idx], SqlValue::Null)) {
            continue;
        }
        let bitmap_byte_offset = device_payload.len() as u64;
        let mut words = vec![0u32; row_count.div_ceil(32)];
        for (i, row) in rows.iter().enumerate() {
            if !matches!(row[col_idx], SqlValue::Null) {
                words[i / 32] |= 1u32 << (i % 32); // 1 = valid/present
            }
        }
        for word in &words {
            device_payload.extend_from_slice(&word.to_le_bytes());
        }
        resident_device_null_columns.push(ResidentDeviceNullBitmapLayout {
            name: name.clone(),
            bitmap_byte_offset,
        });
    }
    // text -> an 8-ALIGNED offsets section (n+1 i64 LE; read as 2x ld.u32 -> 716-safe) + a bytes blob.
    for col_idx in column_types
        .iter()
        .enumerate()
        .filter(|&(_i, ty)| matches!(ty, SqlType::Text))
        .map(|(i, _)| i)
    {
        while !device_payload.len().is_multiple_of(8) {
            device_payload.push(0);
        }
        let offsets_byte_offset = device_payload.len() as u64;
        let mut text_offsets = Vec::with_capacity(row_count + 1);
        let mut text_bytes = Vec::new();
        text_offsets.push(0_u64);
        for row in rows {
            match &row[col_idx] {
                SqlValue::Text(value) => text_bytes.extend_from_slice(value.as_bytes()),
                // NULL → an empty (zero-length) placeholder span; the validity bitmap marks the row.
                SqlValue::Null => {}
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot text payload encountered non-text value".to_string(),
                    )));
                }
            }
            text_offsets.push(text_bytes.len() as u64);
        }
        for offset in &text_offsets {
            device_payload.extend_from_slice(&offset.to_le_bytes());
        }
        let bytes_byte_offset = device_payload.len() as u64;
        device_payload.extend_from_slice(&text_bytes);
        resident_device_text_columns.push(ResidentDeviceTextColumnLayout {
            name: column_names[col_idx].clone(),
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len: text_bytes.len() as u64,
        });
    }
    device_payload[..std::mem::size_of::<u64>()]
        .copy_from_slice(&(row_count as u64).to_le_bytes());
    Ok((
        device_payload,
        resident_device_text_columns,
        resident_device_bool_columns,
        resident_device_int4_column_stats,
        resident_device_b128_columns,
        resident_device_null_columns,
    ))
}

impl Engine {
    pub fn populate_relational_residency_snapshot(
        &mut self,
        table: &str,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let gpu_id = self.planner.default_gpu_id();
        self.populate_relational_residency_snapshot_on_gpu(table, gpu_id)
    }

    fn populate_relational_residency_snapshot_inner(
        &self,
        cat: &mut DdlCatalogState,
        table: &str,
        gpu_id: u16,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
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
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };
        let prefix = relational_key_prefix(table);
        let mut row_count = 0usize;
        let mut resident_bytes = 0u64;
        let mut resident_rows = Vec::new();
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
        let column_types: Vec<SqlType> =
            catalog_table.columns.iter().map(|column| column.ty).collect();
        let (
            mut device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload(&column_names, &column_types, &resident_rows)?;
        // The MVCC tuple bytes (key + value per row) ride after the columnar sections (unchanged).
        device_payload.extend_from_slice(&raw_device_tail);

        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let admission_budget_bytes = cat.relational_resident_cache.budget_bytes_by_gpu.get(&gpu_id).copied();
        let (evicted_tables_on_admission, resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot_inner(cat, table, gpu_id, resident_bytes)?;
        let device_memory = self.relational_residency_device_memory(gpu_id, &device_payload);
        let device_memory_proof = device_memory
            .as_ref()
            .map(|device_memory| device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: self.committed_seq(),
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
                    refreshed_through_index: self.committed_seq(),
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
        cat
            .relational_resident_cache
            .install_snapshot(
                catalog_table.name,
                snapshot.clone(),
                resident_rows,
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
        let Some(budget_bytes) = cat.relational_resident_cache.budget_bytes_by_gpu.get(&gpu_id).copied() else {
            let resident_bytes_after_admission = self
                .relational_resident_bytes_for_gpu_excluding(gpu_id, table)
                .saturating_add(resident_bytes);
            cat
                .relational_resident_cache
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
            cat
                .relational_resident_cache
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

        let mut current_bytes = self.relational_resident_bytes_for_gpu_excluding(gpu_id, table);
        let current_bytes_before = current_bytes;
        let mut evicted_tables = Vec::new();
        if current_bytes.saturating_add(resident_bytes) <= budget_bytes {
            cat
                .relational_resident_cache
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
            return Ok((evicted_tables, current_bytes + resident_bytes));
        }

        let mut candidates = self
            .read_state
            .residency
            .snapshots
            .load()
            .iter()
            .filter(|(name, entry)| name.as_str() != table && entry.descriptor.gpu_id == gpu_id)
            .map(|(name, entry)| {
                (
                    entry.descriptor.valid_through_index,
                    entry.descriptor.table.clone(),
                    name.clone(),
                    entry.descriptor.resident_bytes,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort();
        for (_valid_through_index, _snapshot_table, map_key, bytes) in candidates {
            if current_bytes.saturating_add(resident_bytes) <= budget_bytes {
                break;
            }
            let read_state = Arc::clone(&self.read_state);
            cat
                .relational_resident_cache
                .remove_table(&map_key, &read_state.residency, &read_state.route_telemetry);
            current_bytes = current_bytes.saturating_sub(bytes);
            evicted_tables.push(map_key);
        }

        cat
            .relational_resident_cache
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
                current_bytes_after: current_bytes + resident_bytes,
                evicted_tables: evicted_tables.clone(),
            });
        Ok((evicted_tables, current_bytes + resident_bytes))
    }

    /// `&mut self` entry for the operator warm path: acquire the catalog latch, then run the
    /// `&self`+held-guard producer (the STRATA S-B seam — also reachable from the `&self` commit path).
    fn populate_relational_residency_snapshot_on_gpu(
        &mut self,
        table: &str,
        gpu_id: u16,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let mut guard = self.ddl_catalog();
        self.populate_relational_residency_snapshot_inner(&mut guard, table, gpu_id)
    }

    /// `&mut self` entry for the benchmark residency installers.
    fn admit_relational_residency_snapshot(
        &mut self,
        table: &str,
        gpu_id: u16,
        resident_bytes: u64,
    ) -> Result<(Vec<String>, u64), ExecuteError> {
        let mut guard = self.ddl_catalog();
        self.admit_relational_residency_snapshot_inner(&mut guard, table, gpu_id, resident_bytes)
    }

    /// STRATA S-B: enable/disable automatic GPU-residency admission on commit. Default OFF — turning it
    /// on makes a committed table GPU-resident so subsequent reads take the GPU-native route instead of
    /// the host path. `&self` (an interior-mutable flag the commit path reads).
    pub fn set_auto_admit_on_commit(&self, on: bool) {
        self.auto_admit_on_commit
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn auto_admit_on_commit_enabled(&self) -> bool {
        self.auto_admit_on_commit
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// ADR-009 R1: enable/disable the GPU index-probe point-lookup route (default OFF). When on, a
    /// resident int4 unique-key equality batch probes a cached GPU hash index instead of full-scanning;
    /// non-unique columns / un-buildable indexes transparently fall back to the scan. `&self` (an
    /// interior-mutable flag the read path reads).
    pub fn set_wave_engine_enabled(&self, on: bool) {
        self.wave_engine_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn wave_engine_enabled(&self) -> bool {
        self.wave_engine_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// ADR-009 R2.2b: enable/disable the PERSISTENT wave-kernel point-lookup route (default OFF). Only
    /// takes effect when `wave_engine_enabled` is ALSO on — it selects the persistent `WaveReadEngine`
    /// over the launch-per-batch R1 index probe for resident int4 unique-key equality batches, falling
    /// back to lpb on any wave error / harvest timeout / oversize batch. This is the R2.2b-3 A/B lever
    /// (wave vs lpb). `&self` (an interior-mutable flag the read path reads).
    pub fn set_wave_persistent_engine_enabled(&self, on: bool) {
        self.wave_persistent_engine_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn wave_persistent_engine_enabled(&self) -> bool {
        self.wave_persistent_engine_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// ADR-009 R2.2b: total batches served by the persistent wave route (one per successful
    /// `WaveReadEngine::submit`) vs falling through to lpb/scan. Wave-hit-vs-fallback telemetry for the
    /// R2.2b-3 A/B; also the test hook proving the ROUTE (not just the engine) produced the rows.
    pub fn wave_route_hits(&self) -> u64 {
        self.read_state
            .residency
            .wave_route_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// STRATA S-B: commit-triggered, best-effort GPU-residency admission for the tables a commit
    /// mutated. Runs AFTER `publish_committed_seq` (so it snapshots the new generation) while the
    /// commit_mutex is held; it can NEVER fail the commit — over-budget / memory-pressure / GPU-absent /
    /// dropped-table simply leaves the table non-resident (reads fall back to the host path). N=1
    /// unified buffer per table (single-GPU); shard/spill is S-C/S-E.
    pub(crate) fn auto_admit_resident_tables(&self, tables: &std::collections::BTreeSet<String>) {
        if tables.is_empty() {
            return;
        }
        let gpu_id = self.planner.default_gpu_id();
        let mut guard = self.ddl_catalog();
        let cat = &mut *guard;
        for table in tables {
            let _ = self.populate_relational_residency_snapshot_inner(cat, table, gpu_id);
        }
    }

    fn relational_residency_device_memory(
        &self,
        gpu_id: u16,
        payload: &[u8],
    ) -> Option<CudaResidentDeviceMemory> {
        let runtime = self.cuda_driver_probe_runtime();
        runtime.retain_device_memory_copy(gpu_id, payload).ok()
    }

    /// Build a TRANSIENT resident-like relation from already-materialized host `rows` -- a `RelationalTable`
    /// descriptor + an uploaded device payload that the GPU join path consumes EXACTLY like a published
    /// resident table (`lower_resident_predicate`, `project_*_rows_from_payload`, `hash_join_inner_i64`),
    /// but WITHOUT publishing/admitting/evicting anything (the descriptor + device memory live only for the
    /// caller's query). This is the M5 J5 bridge for a SYNTHESIZED `pg_catalog`/`information_schema`
    /// relation, which has no residency snapshot: synthesize its rows -> this helper -> the existing int4
    /// inner join over the transient payload. Charter: the catalog join runs on the SAME GPU kernels as a
    /// user-table join (no CPU relational join; only the host-rows gather crosses to the host, as for a
    /// resident table). `&self`: the upload only needs `cuda_driver_probe_runtime` (also `&self`).
    ///
    /// Mirrors `populate_relational_residency_snapshot_on_gpu`'s payload + descriptor build (the column
    /// lists feed `build_relational_device_payload`, whose offsets the descriptor's resident-column lists
    /// index), but SKIPS the MVCC tuple tail (the join reads columnar sections + host rows, never the tail)
    /// and the admission machinery. A 0-row relation is fine: the payload is still a non-empty 8-byte
    /// row-count header (the upload's empty-payload guard never trips), and the inner join then yields an
    /// empty result via the empty-survivor / empty-key short-circuits (an empty side is the join's identity).
    pub(crate) fn build_transient_relation_residency(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, CudaResidentDeviceMemory), ExecuteError> {
        let gpu_id = self.planner.default_gpu_id();
        let column_names: Vec<String> =
            table.columns.iter().map(|column| column.name.clone()).collect();
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        // The resident-column lists name (in catalog order) the columns living in each type-grouped
        // payload section; the descriptor's offset helpers index `build_relational_device_payload`'s
        // sections via these lists, so they MUST use the SAME type filters as the resident builder.
        let resident_device_int4_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_int8_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_numeric_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let (
            device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload(&column_names, &column_types, rows)?;
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_copy(gpu_id, &device_payload)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 1,
            row_count: rows.len(),
            column_count: table.columns.len(),
            resident_bytes: device_payload.len() as u64,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: Some(device_memory.metadata().clone()),
        };
        Ok((snapshot, device_memory))
    }

    pub fn install_benchmark_relational_residency_chunks(
        &mut self,
        install: BenchmarkRelationalResidencyChunkInstall<'_>,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let table = install.table;
        let gpu_id = install.gpu_id;
        let row_count = install.row_count;
        let resident_bytes = install.resident_bytes;
        let allocated_bytes = install.allocated_bytes;
        let chunks = install.chunks;
        if row_count == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one generated row"
                    .to_string(),
            )));
        }
        if chunks.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one retained chunk"
                    .to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();

        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }
        Self::validate_benchmark_resident_chunk_columns(
            &catalog_table,
            &install.resident_device_int4_columns,
            &install.resident_device_int4_column_stats,
            &install.resident_device_text_columns,
        )?;
        let copied_bytes = chunks
            .iter()
            .try_fold(0_u64, |total, chunk| {
                let len = u64::try_from(chunk.bytes.len()).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident chunk length exceeds u64".to_string(),
                    ))
                })?;
                let end = chunk.byte_offset.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident chunk offset overflowed".to_string(),
                    ))
                })?;
                if end > allocated_bytes {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident chunk ending at byte {end} exceeds allocation {allocated_bytes}"
                    ))));
                }
                total.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident copied byte count overflowed".to_string(),
                    ))
                })
            })?;
        if copied_bytes == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission copied no bytes".to_string(),
            )));
        }

        let previous_snapshot = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone());
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let admission_budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let (evicted_tables_on_admission, resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, gpu_id, resident_bytes)?;
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_chunks(gpu_id, allocated_bytes, chunks)
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident chunk admission failed CUDA retained upload: {err}"
                )))
            })?;
        let device_memory_proof = Some(device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns: install.resident_device_int4_columns,
            resident_device_int4_column_stats: install.resident_device_int4_column_stats,
            // Benchmark install path: int8 device retention is not wired here yet (doc 19 — the
            // general executor reads int8 only from the standard populate path).
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: install.resident_device_text_columns,
            resident_device_null_columns: Vec::new(),
            valid_through_index: self.committed_seq(),
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
                    refreshed_through_index: self.committed_seq(),
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
        self.ddl_catalog_mut()
            .relational_resident_cache
            .install_snapshot(
                catalog_table.name,
                snapshot.clone(),
                Vec::new(), // benchmark install path: no host-row materialization
                Some(device_memory),
                &read_state.residency,
            );
        Ok(snapshot)
    }

    pub fn install_benchmark_relational_residency_owned_chunks<I>(
        &mut self,
        install: BenchmarkRelationalResidencyOwnedChunkInstall<'_, I>,
    ) -> Result<RelationalResidencySnapshot, ExecuteError>
    where
        I: IntoIterator<Item = CudaOwnedDeviceMemoryChunk>,
    {
        let table = install.table;
        let gpu_id = install.gpu_id;
        let row_count = install.row_count;
        let resident_bytes = install.resident_bytes;
        let allocated_bytes = install.allocated_bytes;
        if row_count == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one generated row"
                    .to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();

        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }
        Self::validate_benchmark_resident_chunk_columns(
            &catalog_table,
            &install.resident_device_int4_columns,
            &install.resident_device_int4_column_stats,
            &install.resident_device_text_columns,
        )?;

        let previous_snapshot = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone());
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let admission_budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let (evicted_tables_on_admission, resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, gpu_id, resident_bytes)?;
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_owned_chunks(gpu_id, allocated_bytes, install.chunks)
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident chunk admission failed CUDA retained upload: {err}"
                )))
            })?;
        let device_memory_proof = Some(device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns: install.resident_device_int4_columns,
            resident_device_int4_column_stats: install.resident_device_int4_column_stats,
            // Benchmark install path: int8 device retention is not wired here yet (doc 19 — the
            // general executor reads int8 only from the standard populate path).
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: install.resident_device_text_columns,
            resident_device_null_columns: Vec::new(),
            valid_through_index: self.committed_seq(),
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
                    refreshed_through_index: self.committed_seq(),
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
        self.ddl_catalog_mut()
            .relational_resident_cache
            .install_snapshot(
                catalog_table.name,
                snapshot.clone(),
                Vec::new(), // benchmark install path: no host-row materialization
                Some(device_memory),
                &read_state.residency,
            );
        Ok(snapshot)
    }

    pub fn install_benchmark_relational_residency_owned_shards(
        &mut self,
        install: BenchmarkRelationalResidencyOwnedShardInstall<'_>,
    ) -> Result<(), ExecuteError> {
        let table = install.table;
        if install.shards.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident shard admission requires at least one shard"
                    .to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();
        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident shard admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }

        let total_resident_bytes =
            install
                .shards
                .iter()
                .try_fold(0_u64, |total, shard| {
                    if shard.row_count == 0 {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "benchmark resident shard {} has no rows",
                            shard.shard_id
                        ))));
                    }
                    if shard.chunks.is_empty() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "benchmark resident shard {} has no retained chunks",
                            shard.shard_id
                        ))));
                    }
                    total.checked_add(shard.resident_bytes).ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "benchmark resident shard byte count overflowed".to_string(),
                        ))
                    })
                })?;
        let (_evicted_tables_on_admission, _resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, install.gpu_id, total_resident_bytes)?;

        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&install.gpu_id);
        let runtime = self.cuda_driver_probe_runtime();
        let mut shards = Vec::new();
        let mut device_memory = BTreeMap::new();
        for shard in install.shards {
            let copied_bytes = shard.chunks.iter().try_fold(0_u64, |total, chunk| {
                let len = u64::try_from(chunk.bytes.len()).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard chunk length exceeds u64".to_string(),
                    ))
                })?;
                let end = chunk.byte_offset.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard chunk offset overflowed".to_string(),
                    ))
                })?;
                if end > shard.allocated_bytes {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident shard {} chunk ending at byte {end} exceeds allocation {}",
                        shard.shard_id, shard.allocated_bytes
                    ))));
                }
                total.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard copied byte count overflowed".to_string(),
                    ))
                })
            })?;
            if copied_bytes == 0 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident shard {} copied no bytes",
                    shard.shard_id
                ))));
            }
            let retained = runtime
                .retain_device_memory_owned_chunks(
                    install.gpu_id,
                    shard.allocated_bytes,
                    shard.chunks,
                )
                .map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident shard admission failed CUDA retained upload: {err}"
                    )))
                })?;
            let device_memory_proof = Some(retained.metadata().clone());
            shards.push(RelationalResidentShard {
                shard_id: shard.shard_id,
                row_start: shard.row_start,
                row_count: shard.row_count,
                resident_bytes: shard.resident_bytes,
                allocated_bytes: shard.allocated_bytes,
                count_header_byte_offset: 0,
                resident_device_int4_columns: shard.resident_device_int4_columns,
                resident_device_text_columns: shard.resident_device_text_columns,
                gpu_id: install.gpu_id,
                schema: catalog_table.schema.clone(),
                table: catalog_table.name.clone(),
                device_memory_proof,
                invalidated_by_txn_id: None,
                invalidated_at_index: None,
                invalidated_by_memory_pressure: memory_pressure_active,
                memory_pressure_active,
            });
            device_memory.insert(shard.shard_id, retained);
        }
        shards.sort_by_key(|shard| (shard.row_start, shard.shard_id));
        let read_state = Arc::clone(&self.read_state);
        self.ddl_catalog_mut()
            .relational_resident_cache
            .install_shards(
                catalog_table.name,
                shards,
                device_memory,
                &read_state.residency,
            );
        Ok(())
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
            // read, so they address THIS shard's buffer (not the whole table).
            row_count: shard.row_count,
            column_count: table.columns.len(),
            resident_bytes: shard.resident_bytes,
            resident_device_int4_columns: shard.resident_device_int4_columns.clone(),
            resident_device_int4_column_stats: Vec::new(),
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: shard.resident_device_text_columns.clone(),
            resident_device_null_columns: Vec::new(),
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
        total_row_count: usize,
        gpu_id: u16,
        resident_bytes: u64,
        proof: CudaDeviceMemoryProof,
        int4_columns: Vec<String>,
    ) -> RelationalResidencySnapshot {
        let resident_device_int4_columns = int4_columns;
        RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 0,
            row_count: total_row_count,
            column_count: table.columns.len(),
            resident_bytes,
            resident_device_int4_columns,
            resident_device_int4_column_stats: Vec::new(),
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: Vec::new(),
            resident_device_null_columns: Vec::new(),
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
            .map(|(_name, entry)| entry.descriptor.resident_bytes)
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
            .map(|shard| shard.resident_bytes)
            .sum();
        snapshot_bytes.saturating_add(shard_bytes)
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
    /// The lightweight, Arc-shared GPU/catalog DESCRIPTOR for a resident table (no host rows). Readers
    /// clone the `Arc` -- a refcount bump, never the row data. (Was an owned deep-clone that copied the
    /// table's host rows on every general-executor query; the split moved those to `host_rows`.)
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

    /// The WHOLE residency entry (descriptor + host rows) from ONE atomic `load()`, so a reader that
    /// needs BOTH halves sees a single consistent generation. Use this instead of calling
    /// `relational_residency_snapshot_ref` + `relational_residency_host_rows` separately -- two
    /// `load()`s could straddle a concurrent publish and pair a descriptor with mismatched rows.
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
                    if let Some(shape) =
                        sharded_resident_route_query_shape(select, &table, &bound)
                    {
                        return self.plan_relational_sharded_resident_route(
                            select, &table, shape, shards,
                        );
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
        let total_rows = shards
            .iter()
            .map(|shard| shard.row_count)
            .sum::<usize>();
        let total_resident_bytes = shards
            .iter()
            .map(|shard| shard.resident_bytes)
            .sum::<u64>();
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
        if decision.query_shape == "sharded_int4_equality_multi_column_projection" {
            let SelectProjection::Columns(columns) = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires projected columns".to_string();
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
        } else if decision.query_shape == "sharded_int4_equality_sum" {
            let SelectProjection::Sum { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires SUM(int4_column)".to_string();
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
                decision.reason =
                    "sharded resident routing requires AVG(int4_column)".to_string();
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
                decision.reason =
                    "sharded resident routing requires MIN(int4_column)".to_string();
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
                decision.reason =
                    "sharded resident routing requires MAX(int4_column)".to_string();
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
            } else if shard.invalidated_by_txn_id.is_some()
                || shard.invalidated_at_index.is_some()
            {
                decision.cache_state = "Invalidated".to_string();
            }
            if shard.schema != table.schema || shard.table != table.name {
                decision.reason =
                    "resident shard no longer matches catalog table identity".to_string();
                return decision;
            }
            if !self
                .read_state
                .residency
                .shard_device_memory
                .contains_key(&(table.name.clone(), shard.shard_id))
            {
                has_all_device_memory = false;
            }
            if !required_int4_columns.is_empty()
                && required_int4_columns
                    .iter()
                    .any(|column| !shard.resident_device_int4_columns.contains(column))
            {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
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
            decision.reason =
                "resident shard set has missing retained device memory".to_string();
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
