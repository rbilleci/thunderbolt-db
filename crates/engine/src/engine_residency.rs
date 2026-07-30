//! GPU residency management + resident-route planning (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for populating/admitting
//! resident snapshots (incl. on-GPU), the benchmark chunk/shard installs,
//! resident device-memory + bytes accounting, retained-read snapshot handles,
//! warmup/maintenance policy execution, and the resident-route planners
//! (plan_relational_resident_route + sharded variant) + residency status.

use super::*;

/// One resident index directory has two slots for every row that can appear in the current shard
/// generation. This keeps linear probing at or below 50% load for the full open-shard lifetime,
/// while its posting-link suffix still reserves exactly one link per physical row.
///
/// The CUDA directory ABI uses a 30-bit slot count and 31-bit physical row ids. Returning `None`
/// makes an optional route decline before allocation when either bound cannot be represented.
pub(crate) fn resident_shard_index_table_size(
    live_rows: u64,
    physical_capacity: u64,
) -> Option<u64> {
    if live_rows == 0 {
        return None;
    }
    let horizon = live_rows.max(physical_capacity);
    let table_size = horizon.checked_mul(2)?.checked_next_power_of_two()?;
    (table_size <= (1_u64 << 30)).then_some(table_size)
}

/// Exact retained bytes for every distinct named-index key id on one shard. The sizing mirrors
/// `ensure_shard_pk_device_index`: a directory at the full physical append horizon plus one posting
/// link per physical row. Shared key ids are charged once because publication reuses their allocation.
pub(crate) fn estimated_named_index_key_bytes_for_shard(
    row_count: usize,
    capacity: usize,
) -> Option<u64> {
    if row_count == 0 {
        return Some(0);
    }
    let table_size = resident_shard_index_table_size(row_count as u64, capacity as u64)?;
    gpu_db_execution::resident_index_allocated_bytes(
        (table_size - 1) as u32,
        capacity.max(row_count) as u64,
    )
}

pub(crate) fn estimated_named_index_bytes_for_shard(
    table: &RelationalTable,
    row_count: usize,
    capacity: usize,
) -> Option<u64> {
    let mut key_ids = std::collections::BTreeSet::new();
    for (ordinal, index) in table.indexes.iter().enumerate() {
        if !index_all_key_columns_foldable(table, index) {
            return None;
        }
        key_ids.insert(index_probe_key_id(table, index, ordinal)?);
    }
    if key_ids.is_empty() || row_count == 0 {
        return Some(0);
    }
    let bytes = estimated_named_index_key_bytes_for_shard(row_count, capacity)?;
    bytes.checked_mul(key_ids.len() as u64)
}

/// Snapshot construction, admission, and publication ownership.
mod admission;
/// Logical input forms for the one mutation-owned resident append publisher.
mod append_source;
/// Typed INSERT device-plan compilation and physical residency adaptation.
mod fixed_insert;
/// In-place physical reservation ingredients for the unreachable indexed handoff.
pub(crate) mod index_delta;
/// Zero-CUDA generation/cache preview consumed by the indexed in-place reservation.
mod index_delta_preview;
#[cfg(test)]
mod index_delta_preview_tests;
/// Fixed-rollover physical reservation ingredients for the unreachable indexed handoff.
pub(crate) mod index_rollover;
/// One sealed forecast-to-materialization boundary for indexed physical preparation.
mod indexed_forecast;
/// Private physical resources for the still-unreachable indexed INSERT handoff.
///
/// This compiles with production so its move-only lifetime and abandonment behavior cannot
/// silently diverge.  It is not a live strategy selector, WAL carrier, or publisher.
mod indexed_reservation;
/// Vacuum, serialized rehydration, and device-gather ownership.
mod maintenance;
/// Resident append, rollover, sparse-version stamping, and fused-apply ownership.
mod mutation;
/// Typed payload/key encoding and open-shard append construction.
mod payload;
/// Residency feature policy, elision eligibility, and telemetry ownership.
mod policy;
/// Prepared, branch-neutral indexed table publication capability.  Its constructors remain
/// test-only until WRITE-001 opens the one live `DeviceInsertPlan` handoff.
mod prepared_table_index_manifest;
#[cfg(test)]
pub(crate) use prepared_table_index_manifest::allocation_test_support::assert_no_thread_allocations;
/// Fit-aware fixed-width rollover planning and private device construction ownership.
mod rollover;
/// Residency warmup, route planning, and status ownership.
mod routes;
/// Commit auto-admission, transient relations, and benchmark installation ownership.
mod transient;

#[allow(unused_imports)] // some sealed apply errors are asserted only by focused tests
pub(crate) use fixed_insert::{
    DeviceInsertPlan, DeviceInsertPlanApplyError, DeviceInsertPlanPrepareError, DeviceInsertRowIds,
};
pub(crate) use payload::{
    build_relational_device_payload, build_relational_device_payload_with_capacity,
    compound_index_row_fingerprint, compound_key_fingerprint, compound_key_type_supported,
    compound_unique_slot_id, compute_open_shard_int4_append_chunks, i32_section_needle,
    index_all_key_columns_foldable, index_is_compound, index_key_column_positions,
    index_probe_key_id, index_uses_fingerprint, key_column_width_words, parse_relational_row_id,
    probe_key_id_positions, sql_value_as_int4, sql_value_from_i32_section,
    sql_value_from_i64_section, sql_value_key_words, AppendCreatedBy, UnifiedResidentSnapshotParts,
    COMPOUND_KEY_ID_FLAG, CREATED_BY_VISIBLE_FILL_BYTE, DELETED_BY_LIVE_FILL_BYTE,
    ROW_ID_UNSTAMPED_FILL_BYTE,
};

#[cfg(test)]
// STRUCT-001 keeps this parent-positioned include as a source-reconstruction boundary. Moving the
// 6,819-line test owner after production items would obscure exact extraction history for no runtime gain.
#[allow(clippy::items_after_test_module)]
mod capacity_payload_tests {
    use super::*;
    use crate::tests::{
        install_test_single_buffer_residency, invalidate_test_relational_residency,
        repair_test_relational_host_copy,
    };

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

    include!("tests/residency_compound_point_reads.rs");

    include!("tests/residency_named_index_publication.rs");

    include!("tests/residency_capacity_budget.rs");
}

impl Engine {
    // Shared facade seam: both transient admission and maintenance reset the serialized churn
    // signal. Keeping this exact helper here avoids a maintenance <-> transient child cycle.
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
            valid_through_index: self.read_snapshot_boundary(),
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

    pub(crate) fn live_compound_point_route_bytes_for_gpu(&self, gpu_id: u16) -> u64 {
        self.live_compound_point_route_bytes_and_entries_for_gpu(gpu_id)
            .0
    }

    /// Compound-route accounting with the exact map entries inspected. The byte total remains
    /// unchanged; the entry count feeds build-only rollover diagnostics.
    pub(crate) fn live_compound_point_route_bytes_and_entries_for_gpu(
        &self,
        gpu_id: u16,
    ) -> (u64, u64) {
        let routes = self
            .read_state
            .residency
            .live_compound_point_route_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entries = routes.len() as u64;
        let bytes = routes
            .iter()
            .filter(|((charged_gpu, _), _)| *charged_gpu == gpu_id)
            .map(|(_, bytes)| *bytes)
            .sum();
        (bytes, entries)
    }

    pub(crate) fn live_compound_point_route_bytes_for_table(
        &self,
        gpu_id: u16,
        table: &str,
    ) -> u64 {
        self.read_state
            .residency
            .live_compound_point_route_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&(gpu_id, table.to_string()))
            .copied()
            .unwrap_or(0)
    }

    fn relational_resident_bytes_for_gpu_excluding(&self, gpu_id: u16, table: &str) -> u64 {
        let retained_gpu = self
            .transaction_retained_gpu_allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut included_allocation_identities = BTreeSet::new();
        let snapshots = self.read_state.residency.snapshots.load();
        included_allocation_identities.extend(
            snapshots
                .iter()
                .filter(|(name, _)| name.as_str() != table)
                .filter_map(|(_, entry)| entry.device_memory.as_ref())
                .map(|memory| (memory.metadata().gpu_id, memory.device_ptr())),
        );
        let snapshot_bytes: u64 = snapshots
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
        let snapshot_sidecar_bytes = [
            &self.read_state.residency.shard_deleted_by_memory,
            &self.read_state.residency.shard_created_by_memory,
            &self.read_state.residency.shard_row_id_memory,
        ]
        .into_iter()
        .map(|sidecars| {
            sidecars.retained_bytes_matching(gpu_id, |name| {
                name != table
                    && snapshots
                        .get(name)
                        .is_some_and(|entry| entry.descriptor.gpu_id == gpu_id)
            })
        })
        .sum::<u64>();
        let shards = self.read_state.residency.shards.load();
        included_allocation_identities.extend(
            shards
                .iter()
                .filter(|(name, _)| name.as_str() != table)
                .flat_map(|(_, shards)| shards)
                .flat_map(|shard| {
                    [
                        shard.device_memory.as_ref(),
                        shard.deleted_by_region.as_ref(),
                        shard.created_by_region.as_ref(),
                        shard.row_id_region.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    .map(|memory| (memory.metadata().gpu_id, memory.device_ptr()))
                }),
        );
        let shard_bytes: u64 = shards
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
        let single_indexes = {
            let cache = self
                .read_state
                .residency
                .wave_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let memories = cache
                .iter()
                .filter(|(name, _)| name.as_str() != table)
                .filter_map(|(_, index)| index.index_memory.as_ref())
                .collect::<Vec<_>>();
            included_allocation_identities.extend(
                memories
                    .iter()
                    .map(|memory| (memory.metadata().gpu_id, memory.device_ptr())),
            );
            memories
                .into_iter()
                .filter(|memory| memory.metadata().gpu_id == gpu_id)
                .map(|memory| memory.metadata().allocated_bytes)
                .sum::<u64>()
        };
        let shard_indexes = {
            let cache = self
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let memories = cache
                .iter()
                .filter(|((name, _, _), _)| name.as_str() != table)
                .filter_map(|(_, index)| index.device_index.as_ref())
                .collect::<Vec<_>>();
            included_allocation_identities.extend(
                memories
                    .iter()
                    .map(|memory| (memory.metadata().gpu_id, memory.device_ptr())),
            );
            memories
                .into_iter()
                .filter(|memory| memory.metadata().gpu_id == gpu_id)
                .map(|memory| memory.metadata().allocated_bytes)
                .sum::<u64>()
        };
        let chunk_indexes = {
            let cache = self
                .read_state
                .residency
                .chunk_key_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let memories = cache
                .iter()
                .filter(|((name, _, _), _)| name.as_str() != table)
                .map(|(_, index)| &index.device)
                .collect::<Vec<_>>();
            included_allocation_identities.extend(
                memories
                    .iter()
                    .map(|memory| (memory.metadata().gpu_id, memory.device_ptr())),
            );
            memories
                .into_iter()
                .filter(|memory| memory.metadata().gpu_id == gpu_id)
                .map(|memory| memory.metadata().allocated_bytes)
                .sum::<u64>()
        };
        let chunk_blooms = {
            let cache = self
                .read_state
                .residency
                .chunk_key_bloom
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let memories = cache
                .iter()
                .filter(|((name, _, _), _)| name.as_str() != table)
                .map(|(_, bloom)| &bloom.device)
                .collect::<Vec<_>>();
            included_allocation_identities.extend(
                memories
                    .iter()
                    .map(|memory| (memory.metadata().gpu_id, memory.device_ptr())),
            );
            memories
                .into_iter()
                .filter(|memory| memory.metadata().gpu_id == gpu_id)
                .map(|memory| memory.metadata().allocated_bytes)
                .sum::<u64>()
        };
        let retained_non_reclaimable = retained_gpu
            .iter()
            .filter(|((device, ptr), _)| {
                *device == gpu_id && !included_allocation_identities.contains(&(*device, *ptr))
            })
            .map(|(_, (bytes, _owners))| *bytes)
            .sum::<u64>();
        drop(retained_gpu);
        let route_descriptors = self
            .read_state
            .residency
            .sharded_point_route_descriptor_bytes_for_gpu_excluding(gpu_id, table);
        // Compound directories are free only after every retired/in-flight plan owner drains, so
        // none of their live charge is treated as immediately reclaimable by table replacement.
        let live_compound_routes = self.live_compound_point_route_bytes_for_gpu(gpu_id);
        // Transaction-private device generations are not attributable to an evictable global
        // table. Keep their retained charge in every "excluding table" admission projection.
        let private_bytes = self
            .transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&gpu_id)
            .copied()
            .unwrap_or(0);
        snapshot_bytes
            .saturating_add(snapshot_sidecar_bytes)
            .saturating_add(shard_bytes)
            .saturating_add(single_indexes)
            .saturating_add(shard_indexes)
            .saturating_add(chunk_indexes)
            .saturating_add(chunk_blooms)
            .saturating_add(retained_non_reclaimable)
            .saturating_add(route_descriptors)
            .saturating_add(live_compound_routes)
            .saturating_add(private_bytes)
            .saturating_sub(Self::active_transaction_commit_gpu_credit(gpu_id))
    }

    /// Actual retained allocation bytes attributable to one table on one GPU. Two-phase admission
    /// subtracts the immediately reclaimable portion of these bytes from the same categories counted
    /// by `relational_resident_bytes_for_gpu_excluding`; live compound-route charges are deliberately
    /// excluded from that subtraction until their final retired/in-flight owner drains.
    fn relational_resident_table_bytes_for_gpu(&self, table: &str, gpu_id: u16) -> u64 {
        let (payload_and_regions, indexes) =
            self.relational_resident_table_byte_components_for_gpu(table, gpu_id);
        payload_and_regions.saturating_add(indexes)
    }

    fn relational_resident_table_byte_components_for_gpu(
        &self,
        table: &str,
        gpu_id: u16,
    ) -> (u64, u64) {
        let snapshot_bytes = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .filter(|entry| entry.descriptor.gpu_id == gpu_id)
            .and_then(|entry| entry.descriptor.device_memory_proof.as_ref())
            .map_or(0, |proof| proof.allocated_bytes);
        let snapshot_sidecar_bytes = if snapshot_bytes == 0 {
            0
        } else {
            [
                &self.read_state.residency.shard_deleted_by_memory,
                &self.read_state.residency.shard_created_by_memory,
                &self.read_state.residency.shard_row_id_memory,
            ]
            .into_iter()
            .map(|sidecars| sidecars.retained_bytes_matching(gpu_id, |name| name == table))
            .sum::<u64>()
        };
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
        let route_descriptor_bytes = self
            .read_state
            .residency
            .sharded_point_route_descriptor_bytes_for_table_gpu(table, gpu_id);
        let compound_route_bytes = self.live_compound_point_route_bytes_for_table(gpu_id, table);
        (
            snapshot_bytes
                .saturating_add(snapshot_sidecar_bytes)
                .saturating_add(shard_bytes),
            single_index_bytes
                .saturating_add(shard_index_bytes)
                .saturating_add(route_descriptor_bytes)
                .saturating_add(compound_route_bytes),
        )
    }

    /// Build-only footprint instrumentation for ADR/benchmark probes. Returns exact retained
    /// `(payload_and_mvcc_region_bytes, device_index_bytes)` for `table` on `gpu_id`; it is absent
    /// from normal builds so analysis code cannot become a product API accidentally.
    #[cfg(feature = "probe-timing")]
    pub fn probe_relational_resident_table_byte_components(
        &self,
        table: &str,
        gpu_id: u16,
    ) -> (u64, u64) {
        self.relational_resident_table_byte_components_for_gpu(table, gpu_id)
    }

    pub fn relational_residency_snapshot(
        &self,
        table: &str,
    ) -> Option<RelationalResidencySnapshot> {
        self.read_residency_snapshots().get(table).map(|entry| {
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
        if self.ensure_commit_path_available().is_err() {
            return None;
        }
        let entry = self.read_residency_snapshots().get(table).cloned()?;
        Some(self.relational_retained_snapshot_handle_from_entry(&entry))
    }

    fn relational_retained_snapshot_handle_from_entry(
        &self,
        entry: &RelationalResidencyEntry,
    ) -> RelationalRetainedSnapshotHandle {
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
            has_retained_device_memory: entry.device_memory.is_some()
                && snapshot
                    .device_memory_proof
                    .as_ref()
                    .is_some_and(|proof| proof.retained),
            resident_device_int4_columns: snapshot.resident_device_int4_columns.clone(),
            resident_device_text_columns: snapshot.resident_device_text_columns.clone(),
            resident_device_null_columns: snapshot.resident_device_null_columns.clone(),
        }
    }

    pub fn relational_retained_device_read_view(
        &self,
        table: &str,
    ) -> Option<RelationalRetainedDeviceReadView> {
        let table_access = self.acquire_autocommit_table_access(table).ok()?;
        self.ensure_commit_path_available().ok()?;
        // One immutable entry owns both descriptor and allocation. The legacy side map is
        // write-side lifecycle bookkeeping and may already contain generation B while the entry
        // map still publishes A; read execution must never combine those independent loads.
        let entry = self.read_residency_snapshots().get(table).cloned()?;
        let handle = self.relational_retained_snapshot_handle_from_entry(&entry);
        if !handle.valid || !handle.has_retained_device_memory {
            return None;
        }
        let device_memory = entry.device_memory.as_ref()?;
        Some(RelationalRetainedDeviceReadView::new(
            handle,
            device_memory.read_view(),
            table_access,
        ))
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
        self.read_residency_snapshots()
            .get(table)
            .map(|entry| entry.descriptor.clone())
    }

    /// The residency entry from one atomic descriptor-map load.
    pub(crate) fn relational_residency_entry(
        &self,
        table: &str,
    ) -> Option<RelationalResidencyEntry> {
        self.read_residency_snapshots().get(table).cloned()
    }
}
