//! Sharded resident route selection and result framing. Unified-source construction, index
//! primitives, and downstream relational executors remain with their established owners.

use super::execution_source::{ResidentExecSource, ResidentVisibility};
use super::normalization::resident_predicate_from_bound_filters;
use super::shard_pruning::shard_point_lookup_int4_eq;
use crate::rel_exec_helpers::bind_relational_select;
use crate::relational_model::{
    resident_device_int4_column_offset, RelationalSelectResult, RelationalTable,
};
use crate::resident_route::BoundRelationalSelect;
use crate::{Engine, ExecuteError};
use gpu_db_execution::{CudaResidentDeviceMemory, DeviceTarget};
use gpu_db_sql::{Select, SelectProjection, SqlType, SqlValue};
use gpu_db_types::{EngineError, Index};
use std::sync::Arc;

impl Engine {
    /// S10c slice 2a: `&Select`->general BRIDGE for the MULTI-PARTITION resident shapes (single-GPU,
    /// int4-only). RECOMPACTS the table's shard buffers into ONE unified int4-only
    /// `CudaResidentDeviceMemory` via `build_sharded_unified_exec_source` (device-to-device copies — the
    /// host stays fully out; only the 8-byte row-count header crosses HtoD), then runs the SAME on-device
    /// general resident-Expr executor ONCE over the unified buffer.
    ///
    /// All-empty handling: the general SUM/MIN/MAX/AVG HARD-ERROR on an empty filtered set (NULL-on-empty
    /// is an unfinished M3 feature for the general path). So we first run a `COUNT(*)` over the unified
    /// buffer; if it is 0 AND the projection is an aggregate, we return the PG-correct empty value WITHOUT
    /// the hard error: SUM/AVG/MIN/MAX -> `SqlValue::Null` (an aggregate of no rows is NULL), COUNT(*) ->
    /// `Int8(0)`. (Was the legacy empty-text/zero sentinel; PG-correctness wins -- `sql-spec-over-cpu-parity`.)
    /// Otherwise the executor runs ONCE over the unified buffer with the real
    /// projection and its result is returned directly (it handles COUNT/SUM/MIN/MAX/AVG/projection
    /// on-device). Text columns are DEFERRED in this slice (the unified buffer is int4-only).
    pub(crate) fn execute_resident_sharded_via_general(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, mut bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        // Sub-slice 3b: detect a single top-level int4 `Eq` POINT-lookup shape from the (still-populated)
        // bound BEFORE the filters are cleared below — the precondition for the cross-shard PK-index route.
        // Mirrors the retained-read (lpb) route's shape gate: exactly one equality group of one, `Eq`, an
        // int4 filter column + an int4 needle. `None` = not a point lookup -> the scan path runs unchanged.
        let point_lookup_eq: Option<(usize, i32)> = shard_point_lookup_int4_eq(&bound, &table);
        // Rebuild the WHERE predicate from the bound filters, then clear them so the executor filters
        // SOLELY via the predicate (the SQL->Expr contract), exactly as the grouped bridge does.
        let predicate = resident_predicate_from_bound_filters(&bound)?;
        bound.filter = None;
        bound.filters.clear();
        bound.filter_groups.clear();
        // Sub-slice 3b: try the CROSS-SHARD PK-INDEX point-lookup route (flag-gated, DEFAULT OFF). On a hit
        // it uses the cached hash+bloom `locate` to jump straight to the (shard, slot) and gather ONLY that
        // row (a few tiny DtoH reads), skipping the zone-map scan + recompaction below; on ANY shape or
        // soundness guard it returns None and we fall through to the scan (byte-identical). Placed before the
        // zone-map prune so it also wins when zone maps DEGRADE under UPDATE key-scatter (membership pruning,
        // scalability-ledger #4/#8) — locate finds the exact shard even when [min,max] can't exclude any.
        // M3-for-shards: the point-index route gathers RAW i32 slots (no validity bitmap), so it would read a
        // NULL-stored-0 as 0 while the scan below is now NULL-aware -> SKIP it for a null-bearing table so the
        // NULL-aware scan serves it. (ADR-006: a null-bearing table can now be MULTI-shard — a NULL insert
        // rolls a dense shard — so this may forgo the many-shard point-index win for such tables; correctness
        // first.) The `..._null_blind_matches_scan` differential is the tripwire that this decline keeps route == scan.
        // (The null check runs on its own lightweight shards load; the route re-validates internally against
        // its own generation-consistent capture, and the unified gather below is NULL-aware regardless.)
        if let Some((filter_idx, needle)) = point_lookup_eq {
            let any_shard_has_nulls = self
                .read_state
                .residency
                .shards
                .load()
                .get(&table.name)
                .is_some_and(|shards| {
                    shards
                        .iter()
                        .any(|s| !s.resident_device_null_columns.is_empty())
                });
            if !any_shard_has_nulls {
                if let Some(result) = self.try_shard_index_point_route(
                    select, &table, &bound, filter_idx, needle, copin_s,
                ) {
                    return Ok(result);
                }
            }
        }
        // FLIP slice — METADATA COUNT fast path (measured: the unpredicated sharded COUNT(*) paid the
        // full recompaction DtoD, p50 ~430-530us at 524k rows vs single-buffer's 19us). An unpredicated
        // COUNT(*) over ALL-version-free shards is exactly `sum(shard.row_count)` — descriptor metadata
        // the route planner already reads (control plane; no row data touched, no kernel, no copy). Any
        // version region (a tombstone could hide rows / a stamp could hide appended versions) or any
        // predicate falls through to the device path unchanged.
        if matches!(select.projection, SelectProjection::CountAll) && predicate.is_none() {
            let shards_guard = self.read_state.residency.shards.load();
            if let Some(shards) = shards_guard.get(&table.name) {
                // D4: read the version-freeness from the SAME loaded descriptors being summed —
                // the metadata COUNT can no longer pair an old shard list with freshly-purged maps.
                // D3 hwm gate: created_by-only shards whose stamps are all <= the reader's boundary
                // count every row (effectively version-free); a reader pinned inside an append
                // window (copin_s < hwm) falls through to the gated device path.
                // W0c (audit B1): ALSO require every shard VALID — this executor-side load can be
                // NEWER than the accepted route plan's (a concurrent commit flags + publishes in
                // between), and a flagged shard's row_count excludes the host-installed rows the
                // reader's pinned boundary includes. An invalid shard falls through to the gated
                // device path, whose source_for declines and the statement re-serves from the CPU.
                let runtime_snapshot = self.router.runtime().snapshot();
                let version_free = shards.iter().all(|shard| {
                    shard.is_valid(
                        runtime_snapshot
                            .memory_pressured_gpu_ids
                            .contains(&shard.gpu_id),
                    ) && shard.deleted_by_region.is_none()
                        && (shard.created_by_region.is_none() || copin_s >= shard.max_created_by)
                });
                if version_free && !shards.is_empty() {
                    let total: i64 = shards.iter().map(|s| s.row_count as i64).sum();
                    let gpu_id = shards[0].gpu_id;
                    drop(shards_guard);
                    let (_query, access_path) =
                        self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
                    return Ok(RelationalSelectResult {
                        columns: Arc::new(bound.selected_columns.clone()),
                        rows: vec![vec![SqlValue::Int8(total)]].into(),
                        planned_target: DeviceTarget::Gpu(gpu_id),
                        executed_target: DeviceTarget::Gpu(gpu_id),
                        fallback_reason: None,
                        access_path: Arc::new(access_path),
                    });
                }
            }
        }
        // SLICE B: the shard load + zone-map prune + int4/version/null-bitmap recompaction live in
        // `build_sharded_unified_exec_source`, SHARED with the SQL->Expr PG path so IS NULL and every
        // other general-executor shape run over sharded tables through the SAME on-device execution.
        let unified =
            self.build_sharded_unified_exec_source(&table, predicate.as_ref(), copin_s)?;
        let gpu_id = unified.gpu_id;
        let visibility = unified.visibility;
        let unified_src = unified.src;

        // Run one (already-bound) select against an injected source via the general executor. `vis` is the
        // SV3b/SV6 MVCC visibility descriptor for the unified buffer (`Some` when any surviving shard is
        // versioned, else `None`) -- forwarded so the on-device predicate ANDs the visibility bound(s)
        // (`deleted_by > read_txn_id`, `created_by <= read_txn_id`) and hides invisible versions. BOTH the
        // COUNT precheck and the real run pass the SAME `vis` so the count and the projection agree on
        // which rows are visible.
        let run = |select_ref: &Select,
                   bound_for_select: BoundRelationalSelect,
                   src: &ResidentExecSource,
                   vis: Option<ResidentVisibility>|
         -> Result<RelationalSelectResult, ExecuteError> {
            self.execute_resident_expr_select_with_binding(
                select_ref,
                &table,
                Some(src),
                bound_for_select,
                copin_s,
                predicate.as_ref(),
                vis,
                &[],
                &[],
                None,
                &[],
            )
        };

        // The access path is shape/table-level metadata (it does not depend on the shard data), so
        // compute it once from the (filter-cleared) bound exactly as a probe did (engine_resident_probe.rs
        // ~846). bound was filter-cleared above; the sharded shapes carry no ORDER BY / LIMIT.
        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;

        let finalize = |rows: Vec<Vec<SqlValue>>| -> RelationalSelectResult {
            RelationalSelectResult {
                columns: Arc::new(bound.selected_columns.clone()),
                rows: rows.into(),
                planned_target: DeviceTarget::Gpu(gpu_id),
                executed_target: DeviceTarget::Gpu(gpu_id),
                fallback_reason: None,
                access_path: Arc::new(access_path.clone()),
            }
        };

        // S10c slice 2b: DISTINCT / GROUP BY / ORDER-BY-projection are CORRECT over the unified buffer (it
        // holds the WHOLE table), so route each to the grouped/distinct sub-bridge with the unified source
        // injected. These sub-bridges re-bind + re-derive the WHERE predicate INTERNALLY from `select`, so
        // they need only `select` + `Some(&unified_src)` (the outer filter-cleared `bound` is unused by
        // them). Order matters: DISTINCT carries no `group_by` but synthesizes one internally, so it must be
        // checked first; a grouped select may ALSO carry ORDER BY and must take the grouped path. The plain
        // scalar/projection shapes fall through to the COUNT-precheck + single run below (unchanged).
        //
        // R-ver PART 2: the reshaping sub-bridges are visibility-correct — they route back into
        // `execute_resident_expr_select_with_binding`, whose survivor `indices` fold in the SV3b/SV6
        // visibility BEFORE group/sort/dedup (verified: every grouped/distinct/ordered kernel reads
        // only `indices`/`indices_u64`). So thread the versioned buffer's `visibility` through to
        // them instead of refusing. (`visibility` is `None` for a version-free buffer -> byte-identical.)
        if select.distinct {
            return self.execute_resident_distinct_via_general(
                select,
                Some(&unified_src),
                visibility,
            );
        }
        if select.group_by.is_some() {
            return self.execute_resident_grouped_via_general(
                select,
                Some(&unified_src),
                visibility,
            );
        }
        if !select.order_by.is_empty() {
            return self.execute_resident_grouped_via_general(
                select,
                Some(&unified_src),
                visibility,
            );
        }

        // All-empty handling (pins byte-identicality with slice 1): the general SUM/MIN/MAX/AVG hard-error
        // on an empty filtered set, so first run a COUNT(*) over the unified buffer; if it is 0 AND the
        // projection is an aggregate, return the SAME placeholder slice 1 did.
        let count_select = {
            let mut s = select.clone();
            s.projection = SelectProjection::CountAll;
            s
        };
        let count_bound = bind_relational_select(&table, &count_select)?;
        let matched = {
            let result = run(&count_select, count_bound, &unified_src, visibility)?;
            match result.rows.iter().next().and_then(|row| row.first()) {
                Some(SqlValue::Int8(n)) => *n,
                other => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sharded resident COUNT(*) precheck returned an unexpected value: {other:?}"
                    ))));
                }
            }
        };
        if matched == 0 {
            // PG: SUM/AVG/MIN/MAX over zero rows is NULL (never 0 or an empty-text sentinel); only
            // COUNT(*) is 0. The old non-NULL placeholders were legacy CPU-engine parity (interim debt,
            // ADR-006); PG-correctness wins (see the `sql-spec-over-cpu-parity` working agreement).
            let placeholder = match &select.projection {
                SelectProjection::Min { .. }
                | SelectProjection::Max { .. }
                | SelectProjection::Avg { .. }
                | SelectProjection::Sum { .. } => Some(SqlValue::Null),
                SelectProjection::CountAll => Some(SqlValue::Int8(0)),
                _ => None,
            };
            if let Some(cell) = placeholder {
                return Ok(finalize(vec![vec![cell]]));
            }
        }

        // Run the executor ONCE over the unified buffer with the real projection — it handles
        // COUNT / SUM / MIN / MAX / AVG / projection on-device — and return its result directly.
        run(select, bound.clone(), &unified_src, visibility)
    }

    /// Sub-slice 3b: the CROSS-SHARD PK-INDEX point-lookup route for the sharded read path. For a
    /// shard-resident int4 UNIQUE-key equality POINT lookup with a PLAIN int4-column projection, use the
    /// cached hash+bloom `locate` to jump straight to the located `(shard, slot)` and gather ONLY that row
    /// (a few tiny DtoH reads) instead of scanning + recompacting the whole (zone-map-pruned) shard — the
    /// per-shard scan is ~half of a point-lookup's latency at the 4M production shard size (MEASURED: point
    /// p50 131k=343us -> 4M=709us at avg-gathered 1.00, so the rise is the one gathered shard's scan).
    ///
    /// Returns `Some(result)` — BYTE-IDENTICAL to the scan (`columns` / `access_path` / target computed the
    /// SAME way as the scan's `finalize`) — when it served the read, or `None` to FALL BACK to the existing
    /// scan + recompaction (the caller runs it unchanged) on ANY shape or soundness guard: never a wrong
    /// result. Guards (each -> None -> scan): the flag is OFF; the projection is not a plain `All`/`Columns`
    /// of int4-only columns, or carries DISTINCT / GROUP BY / ORDER BY / LIMIT / OFFSET / HAVING; `locate`
    /// declines (duplicate / oversize / invalid shard); more than one hit (a cross-shard duplicate -> the
    /// scan applies its multi-row + ordering); or the located shard is gone / invalid / raced (`slot >=
    /// row_count`).
    ///
    /// NULLs (M3-for-shards): the sharded SCAN is now NULL-AWARE (its recompaction rebuilds each column's
    /// validity bitmap into the unified buffer + labels the unified descriptor), but this route gathers RAW i32
    /// slots with no validity channel. So the CALLER SKIPS this route entirely for a null-bearing table (any
    /// surviving shard with a non-empty `resident_device_null_columns`) -> the NULL-aware scan serves it. null-
    /// bearing is single-shard by construction, so the skip never costs the many-shard route. The
    /// `deleted_by[slot] > read_txn_id` visibility gate mirrors the scan's SV3b filter — a tombstoned row
    /// materializes ZERO rows. (`cross_shard_pk_index_route_declines_on_null_bearing` is the tripwire that this
    /// decline keeps route == the NULL-aware scan.)
    fn try_shard_index_point_route(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        filter_idx: usize,
        needle: i32,
        copin_s: Index,
    ) -> Option<RelationalSelectResult> {
        if !self.shard_index_probe_enabled() {
            return None;
        }
        // Plain int4-column projection ONLY: `All`/`Columns` (both resolve to `selected_indexes`), no
        // aggregate / DISTINCT / GROUP BY / ORDER BY / LIMIT / OFFSET / HAVING — anything else is the scan's.
        if !matches!(
            select.projection,
            SelectProjection::All | SelectProjection::Columns(_)
        ) || select.distinct
            || select.group_by.is_some()
            || !select.order_by.is_empty()
            || select.limit.is_some()
            || select.offset.is_some()
            || !select.having_groups.is_empty()
        {
            return None;
        }
        if bound.selected_indexes.is_empty() {
            return None;
        }
        for &idx in &bound.selected_indexes {
            if table.columns.get(idx).map(|c| c.ty) != Some(SqlType::Int4) {
                return None;
            }
        }
        // Locate the (shard_id, slot) via the cached hash+bloom index. `None` = the index declined (a
        // duplicate / oversize / invalidated shard) -> scan. `Some(hits)`: 0 hits = key absent everywhere
        // (0 rows), 1 hit = the row, >1 = a cross-shard duplicate -> scan (the scan returns every match).
        let hits = self.locate_resident_pk_via_shard_index_detailed(table, filter_idx, needle)?;
        if hits.len() > 1 {
            return None;
        }
        // gpu_id is a result LABEL (shape metadata, not shard data), so a lightweight lock-free load is fine
        // — the DATA read below uses the generation-consistent handles captured inside the hit, which is what
        // closes the concurrent TOCTOU (a stale gpu_id label on the same single GPU is harmless).
        let gpu_id = self
            .read_state
            .residency
            .shards
            .load()
            .get(&table.name)?
            .first()?
            .gpu_id;
        // `access_path` + `columns` are shape/table-level metadata (independent of the shard data), computed
        // EXACTLY as the scan's `finalize` does (from the filter-cleared `bound`) so the result is
        // byte-identical to the scan's. The transient read pin is dropped immediately (RAII).
        let (_query, access_path) = self
            .relational_select_mvcc_query_pinned(select, table, bound, copin_s)
            .ok()?;

        let rows: Vec<Vec<SqlValue>> = if let Some(hit) = hits.first() {
            // GENERATION-CONSISTENT materialization (audit fix). The slot, the capacity-stride
            // (`hit.descriptor`), the int4 buffer (`hit.device_memory`, PINNED by the Arc), and the
            // `deleted_by` region were ALL captured in the SAME `shards.load()` snapshot inside
            // `locate...detailed`. Reading the slot out of `hit`'s pinned buffer (NOT a fresh
            // `shard_device_memory.get`) closes the concurrent TOCTOU: reads are lock-free and straddle
            // commits, so a concurrent DELETE re-admit can republish a shard_id's buffer with COMPACTED /
            // reordered slots; resolving the slot against one generation and re-`get`-ing the buffer (a
            // second independent `ArcSwap` load) could read the slot out of a DIFFERENT generation -> a wrong
            // row. Holding the exact buffer the slot indexes into makes that impossible. (locate already ran
            // the identity / is_valid / memory-pressure prechecks before capturing the hit.)
            let slot = hit.slot as u64;
            if hit.slot as usize >= hit.descriptor.row_count {
                return None; // defensive: slot past the captured live region
            }
            // NULL handling: the sharded read path is uniformly NULL-BLIND (NULL stored as 0). Its
            // recompaction is int4-only with NO validity-bitmap segment, and BOTH descriptors it builds --
            // `resident_snapshot_for_shard` AND the scan's `resident_snapshot_for_unified` -- carry
            // `resident_device_null_columns: Vec::new()`, so the scan reads the raw i32 (a NULL reads back as
            // 0, and `col = 0` MATCHES a NULL-stored-0 row). This raw-i32 slot gather is therefore
            // BYTE-IDENTICAL to the scan on NULLs BY CONSTRUCTION (the `..._null_blind_matches_scan`
            // differential proves it; it is the tripwire when M3 recompacts bitmaps through the sharded path).
            //
            // SV3b visibility gate: a VERSIONED shard's `deleted_by[slot]` (dense i64 at byte `slot*8`, NO
            // header, little-endian) HIDES the row when `deleted_by <= read_txn_id`; an un-versioned shard (no
            // region) is all-live. `copin_s as i64` is the read snapshot, exactly the scan's `vis`. The region
            // is the SAME-generation handle captured in the hit.
            let read_i64_at_slot = |region: &Arc<CudaResidentDeviceMemory>| -> Option<i64> {
                let halves = region.read_resident_i32_column(slot * 8, 2).ok()?;
                Some(
                    ((*halves.first()? as u32 as u64) | ((*halves.get(1)? as u32 as u64) << 32))
                        as i64,
                )
            };
            let visible = match &hit.deleted_by {
                Some(region) => read_i64_at_slot(region)? > copin_s as i64,
                None => true,
            }
            // SV6 lower bound: a `created_by`-versioned shard's `created_by[slot]` HIDES the row when it
            // exceeds the read snapshot (an UPDATE-appended version whose commit this reader must not see —
            // the double-read gate), mirroring the scan's `created_by <= read_txn_id` conjunct. An
            // un-stamped shard (no region) is born-visible.
            && match &hit.created_by {
                Some(region) => read_i64_at_slot(region)? <= copin_s as i64,
                None => true,
            };
            if visible {
                // Materialize the projected row: one tiny DtoH per projected int4 column at its
                // capacity-strided slot byte (`col_base + slot*4`) in the PINNED buffer. Column order =
                // `selected_indexes` = the scan's projection order, so the row is byte-identical to the scan's.
                let mut row = Vec::with_capacity(bound.selected_indexes.len());
                for &idx in &bound.selected_indexes {
                    let col_base =
                        resident_device_int4_column_offset(&hit.descriptor, table, idx).ok()?;
                    let v = hit
                        .device_memory
                        .read_resident_i32_column(col_base + slot * 4, 1)
                        .ok()?;
                    row.push(SqlValue::Int4(*v.first()?));
                }
                vec![row]
            } else {
                Vec::new()
            }
        } else {
            // All shards Miss -> the key is absent -> zero rows (the scan returns the same empty set).
            Vec::new()
        };

        self.read_state
            .residency
            .shard_index_route_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns.clone()),
            rows: rows.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(access_path),
        })
    }
}
