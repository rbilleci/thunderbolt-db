//! STRATA S-E.1 — the streaming out-of-core executor (ADR-012 / PLAN S-E).
//!
//! A query whose working set exceeds the GPU byte budget must still run ON THE DEVICE: the charter
//! forbids the host from being an execution tier, and ADR-006 deletes the CPU relational engine. The
//! streaming executor closes the over-VRAM gap for the SIMPLEST foldable operator class — the SCALAR
//! REDUCTIONS (`COUNT(*)` / `SUM` / `MIN` / `MAX`) — by folding over the table in bounded chunks:
//!
//!   admit chunk -> push the reduction down (device kernel) -> combine the partial -> evict chunk -> next
//!
//! Only ONE chunk's working set is resident at a time, so a single GPU serves a relation far larger than
//! its VRAM. Host RAM is the cold STORAGE tier (the MVCC store the chunk bytes are staged from); the GPU
//! is the SOLE execution tier — the WHERE filter and the reduction both run in a device kernel. The host
//! does only what the charter permits: MVCC visibility resolution (control plane), the staging upload
//! (`build + upload the next device generation`), and the associative partial COMBINE (COUNT = sum of
//! counts, SUM = checked sum, MIN/MAX = the extreme) — never the reduction itself.
//!
//! This reuses the whole predicate + aggregate executor: each chunk is a transient
//! [`ResidentExecSource`] (`build_transient_relation_residency`) run through
//! `execute_resident_expr_select_with_binding`, exactly as a resident shard slice is. The only new logic
//! is the cross-chunk combine. Any executor error on a chunk defers to the authoritative CPU pinned path
//! (streaming only ever ADDS on-device reach — it can never return a wrong answer), mirroring the
//! general-executor read fallback (engine_select_exec.rs). Activation gates on a CONFIGURED per-GPU
//! residency budget (the operator's VRAM-management signal); with no budget there is no notion of
//! "over-VRAM" and the read stays on the interim host path — so default behavior is byte-identical.
//!
//! **S-E.2 (here too): streaming filter/project.** A plain `All`/`Columns` projection folds by CONCAT
//! (the ARCHITECTURE §13 projection combine): each chunk's device-filtered + device-gathered survivors
//! append to the result, with LIMIT/OFFSET applied as cross-chunk windowing of the survivor stream (the
//! executor's own "LIMIT/OFFSET as control-plane WINDOWING" precedent) and a satisfied LIMIT stopping the
//! scan early — the table tail is never staged.
//!
//! Follow-ons (see PLAN S-E): GROUP BY / DISTINCT with a persistent device accumulator (S-E.3), ORDER BY
//! via k-way run merge (S-E.4), and copy/compute overlap (S-E.5). AVG is deferred (it needs the
//! (sum, count) pair combined, not the divided per-chunk average).

use super::*;

use crate::engine_expr::{
    grouped_projection_to_aggregates, resident_predicate_from_bound_filters, ResidentExecSource,
    ResidentExpr,
};
use crate::rel_exec_helpers::{
    bind_relational_select, catalog_relation_table, compare_sql_values, decode_relational_row,
    relational_key_prefix,
};
use std::sync::atomic::Ordering;

/// The DEVICE payload bytes one row contributes to a transient chunk. Unlike the logical
/// `relational_resident_value_bytes` (which is 0 for `NULL` and 0 for empty text), this counts the FIXED
/// typed slot the columnar device layout allocates for EVERY row — a `NULL` still occupies its 4/8/16-byte
/// column slot — so the out-of-core chunk-size bound holds even for null-heavy tables (audit Finding 1).
/// Text adds its actual blob length plus one offset entry. The match is exhaustive over `SqlType` so a
/// future scalar type is a compile error here (not a silent mis-estimate).
/// The 8-byte header + per-column null bitmaps are per-chunk and small vs. the typed widths (the budget/2
/// chunk target absorbs them), so they are intentionally omitted.
fn chunk_row_device_bytes(row: &[SqlValue], column_types: &[SqlType]) -> u64 {
    column_types
        .iter()
        .zip(row.iter())
        .map(|(ty, value)| match ty {
            SqlType::Int2 | SqlType::Int4 | SqlType::Date => 4,
            SqlType::Int8 | SqlType::Timestamp => 8,
            SqlType::Numeric { .. } | SqlType::Uuid => 16,
            SqlType::Bool => 1,
            SqlType::Text => {
                4 + match value {
                    SqlValue::Text(text) => text.len() as u64,
                    _ => 0,
                }
            }
        })
        .sum()
}

/// The foldable scalar reduction a streaming fold serves. AVG is intentionally absent (deferred).
#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamAgg {
    Count,
    Sum,
    Min,
    Max,
}

/// The running SUM partial. The executor returns `Int8` for `SUM(int4)` and `Numeric` for
/// `SUM(int8)`/`SUM(numeric)`; a fold sees ONE column type, so the kind is fixed after the first
/// non-null contribution (a mismatch => defer to CPU, never a wrong combine).
enum SumPartial {
    Empty,
    Int(i128),
    Dec(Decimal128),
}

/// The cross-chunk accumulator: the host-side (control-plane) COMBINE of the per-chunk device partials.
enum StreamAccum {
    Count(i128),
    Sum(SumPartial),
    Extreme { is_max: bool, current: Option<SqlValue> },
}

/// A chunk's device reduction either combined cleanly, or hit a shape the streaming path should hand back
/// to the authoritative CPU pinned path (`Defer`), or a genuine SQL error to surface (`Hard`).
enum ChunkOutcome {
    Ok,
    Defer,
    Hard(ExecuteError),
}

impl StreamAccum {
    fn new(agg: StreamAgg) -> Self {
        match agg {
            StreamAgg::Count => StreamAccum::Count(0),
            StreamAgg::Sum => StreamAccum::Sum(SumPartial::Empty),
            StreamAgg::Min => StreamAccum::Extreme {
                is_max: false,
                current: None,
            },
            StreamAgg::Max => StreamAccum::Extreme {
                is_max: true,
                current: None,
            },
        }
    }

    /// Fold one chunk's device COUNT(*) (always available — the empty-filtered-set guard).
    fn add_count(&mut self, chunk_count: i64) {
        if let StreamAccum::Count(total) = self {
            *total += i128::from(chunk_count);
        }
    }

    /// Fold one chunk's device SUM/MIN/MAX value into the running partial. A `Null` value (an all-NULL
    /// filtered chunk) contributes nothing (PG). Returns `Defer` on an unexpected variant.
    fn combine_value(&mut self, value: SqlValue) -> ChunkOutcome {
        match self {
            StreamAccum::Count(_) => ChunkOutcome::Ok,
            StreamAccum::Sum(partial) => {
                match value {
                    SqlValue::Null => ChunkOutcome::Ok,
                    // SUM(int4) -> Int8 partials: accumulate as i128 (the running total can exceed i64;
                    // the final narrow to bigint is overflow-checked in `finalize`).
                    SqlValue::Int8(n) => match partial {
                        SumPartial::Empty => {
                            *partial = SumPartial::Int(i128::from(n));
                            ChunkOutcome::Ok
                        }
                        SumPartial::Int(acc) => {
                            *acc += i128::from(n);
                            ChunkOutcome::Ok
                        }
                        SumPartial::Dec(_) => ChunkOutcome::Defer,
                    },
                    // SUM(int8)/SUM(numeric) -> Numeric partials at the column scale: checked scale-aligned
                    // add (a genuine numeric overflow surfaces as a hard error, matching PG + the executor).
                    SqlValue::Numeric(d) => match partial {
                        SumPartial::Empty => {
                            *partial = SumPartial::Dec(d);
                            ChunkOutcome::Ok
                        }
                        SumPartial::Dec(acc) => match acc.checked_add(d) {
                            Ok(sum) => {
                                *acc = sum;
                                ChunkOutcome::Ok
                            }
                            Err(_) => ChunkOutcome::Hard(ExecuteError::Engine(
                                EngineError::ApplyFailed("numeric field overflow".to_string()),
                            )),
                        },
                        SumPartial::Int(_) => ChunkOutcome::Defer,
                    },
                    _ => ChunkOutcome::Defer,
                }
            }
            StreamAccum::Extreme { is_max, current } => {
                if matches!(value, SqlValue::Null) {
                    return ChunkOutcome::Ok;
                }
                match current {
                    None => *current = Some(value),
                    Some(existing) => {
                        let ordering = compare_sql_values(&value, existing);
                        let take = if *is_max {
                            ordering == std::cmp::Ordering::Greater
                        } else {
                            ordering == std::cmp::Ordering::Less
                        };
                        if take {
                            *current = Some(value);
                        }
                    }
                }
                ChunkOutcome::Ok
            }
        }
    }

    /// The combined scalar: COUNT -> Int8; SUM -> the partial narrowed to its PG type (NULL if no
    /// non-null contribution); MIN/MAX -> the extreme (NULL over an empty/all-null set).
    fn finalize(self) -> Result<SqlValue, ExecuteError> {
        match self {
            StreamAccum::Count(total) => i64::try_from(total).map(SqlValue::Int8).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "bigint out of range in COUNT(*)".to_string(),
                ))
            }),
            StreamAccum::Sum(SumPartial::Empty) => Ok(SqlValue::Null),
            StreamAccum::Sum(SumPartial::Int(total)) => {
                i64::try_from(total).map(SqlValue::Int8).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "bigint out of range".to_string(),
                    ))
                })
            }
            StreamAccum::Sum(SumPartial::Dec(sum)) => Ok(SqlValue::Numeric(sum)),
            StreamAccum::Extreme { current, .. } => Ok(current.unwrap_or(SqlValue::Null)),
        }
    }
}

/// A device reduction error that is a genuine arithmetic OVERFLOW (matched on the executor's stable PG
/// overflow phrases). Such an error must surface, not defer to the CPU path (audit Finding 2).
fn is_overflow_error(err: &ExecuteError) -> bool {
    let message = err.to_string();
    message.contains("overflow") || message.contains("out of range")
}

/// The streaming operator class serving a SELECT: a scalar reduction fold (S-E.1), a filter/project
/// concat fold (S-E.2), or a grouped / distinct two-level fold (S-E.3).
enum StreamShape {
    Reduction(StreamAgg),
    Projection,
    /// `GROUP BY g` with Count/Sum/Min/Max aggregates — the normalized [`SelectProjection::
    /// GroupedAggregates`] form. `distinct_key_only` = the shape is a synthesized `SELECT DISTINCT col`
    /// (a GROUP BY col + COUNT(*) whose count column is dropped from the result, exactly the distinct
    /// bridge's synthesis).
    Grouped {
        normalized: SelectProjection,
        distinct_key_only: bool,
    },
}

/// Classify a SELECT as a streamable shape, or `None` for the non-foldable classes (ORDER BY — S-E.4;
/// HAVING; grouped AVG / COUNT(DISTINCT) — not associatively decomposable from per-chunk partials). A
/// SCALAR reduction (COUNT(*)/SUM/MIN/MAX, no LIMIT/OFFSET — PG applies LIMIT to the one-row aggregate
/// result, a shape not worth streaming) folds by combine; a plain `All`/`Columns` projection folds by
/// CONCAT, with LIMIT/OFFSET as cross-chunk windowing (LIMIT without ORDER BY is any-N-rows per SQL, so
/// early-exit + scan-order windowing is a valid instance); GROUP BY / DISTINCT fold TWO-LEVEL (per-chunk
/// device partials -> concat -> one final device merge pass).
fn streaming_shape(select: &Select) -> Option<StreamShape> {
    if !select.order_by.is_empty() || !select.having_groups.is_empty() {
        return None;
    }
    // S-E.3 DISTINCT: single-column `SELECT DISTINCT col` == `SELECT col, COUNT(*) GROUP BY col` with
    // the count dropped (the distinct bridge's own synthesis) — so it rides the grouped fold.
    if select.distinct {
        if select.group_by.is_some() || select.limit.is_some() || select.offset.is_some() {
            return None;
        }
        let SelectProjection::Columns(columns) = &select.projection else {
            return None;
        };
        let [column] = columns.as_slice() else {
            return None;
        };
        return Some(StreamShape::Grouped {
            normalized: SelectProjection::GroupedAggregates {
                group_column: column.clone(),
                aggregates: vec![GroupedAggregate {
                    kind: GroupedAggKind::Count,
                    value_column: None,
                }],
            },
            distinct_key_only: true,
        });
    }
    // S-E.3 GROUP BY: normalize the legacy 1-aggregate forms to GroupedAggregates (as the grouped
    // bridge does) and accept only associatively-decomposable kinds: COUNT merges as SUM(count),
    // SUM as SUM(sum), MIN as MIN(min), MAX as MAX(max). AVG needs the (sum,count) pair and
    // COUNT(DISTINCT) is not decomposable from per-chunk distinct counts — both decline to CPU.
    if let Some(group_column) = &select.group_by {
        if select.limit.is_some() || select.offset.is_some() {
            return None;
        }
        let normalized = match grouped_projection_to_aggregates(&select.projection) {
            Some(normalized) => normalized,
            None => match &select.projection {
                SelectProjection::GroupedAggregates { .. } => select.projection.clone(),
                _ => return None,
            },
        };
        let SelectProjection::GroupedAggregates {
            group_column: normalized_key,
            aggregates,
        } = &normalized
        else {
            return None;
        };
        if normalized_key != group_column {
            return None;
        }
        if !aggregates.iter().all(|aggregate| {
            matches!(
                aggregate.kind,
                GroupedAggKind::Count
                    | GroupedAggKind::Sum
                    | GroupedAggKind::Min
                    | GroupedAggKind::Max
            )
        }) {
            return None;
        }
        return Some(StreamShape::Grouped {
            normalized,
            distinct_key_only: false,
        });
    }
    match &select.projection {
        SelectProjection::All | SelectProjection::Columns(_) => Some(StreamShape::Projection),
        _ if select.limit.is_some() || select.offset.is_some() => None,
        SelectProjection::CountAll => Some(StreamShape::Reduction(StreamAgg::Count)),
        SelectProjection::Sum { .. } => Some(StreamShape::Reduction(StreamAgg::Sum)),
        SelectProjection::Min { .. } => Some(StreamShape::Reduction(StreamAgg::Min)),
        SelectProjection::Max { .. } => Some(StreamShape::Reduction(StreamAgg::Max)),
        _ => None,
    }
}

impl Engine {
    /// STRATA S-E.1/S-E.2: try to serve a SELECT OUT-OF-CORE via a streaming fold — a scalar reduction
    /// (combine partials) or a filter/project (concat + windowing). Returns `Some(result)` when the
    /// streaming path handled the read (`Ok`) or must surface a genuine SQL error (`Err`); `None` to fall
    /// through to the caller's path (the CPU pinned read). It NEVER returns a wrong answer: any shape the
    /// device cannot express defers to the authoritative CPU path.
    pub(crate) fn try_streaming_select(
        &self,
        select: &Select,
    ) -> Option<Result<RelationalSelectResult, ExecuteError>> {
        let shape = streaming_shape(select)?;
        let gpu_id = self.planner.default_gpu_id();
        // Activation gate: a per-GPU residency budget must be configured (the operator's VRAM-management
        // signal). With no budget there is no notion of "over-VRAM" -> stay on the interim host path
        // (byte-identical default behavior).
        let budget = self.relational_residency_budget_bytes(gpu_id)?;
        if budget == 0 {
            return None;
        }
        // Never stream an ELIDED table: its host MVCC store is intentionally stale (device-authoritative
        // writes), so a seq-scan would read the wrong data. Its device residency is served upstream.
        if self.table_install_elided(&select.table) {
            return None;
        }
        // Bind + lower the WHERE to a device predicate exactly as the sharded bridge does. A bind failure
        // or an un-lowerable predicate falls through to the host path (never a wrong answer).
        let (table, mut bound, copin_s) = self.bind_relational_select_for_execution(select).ok()?;
        let predicate = resident_predicate_from_bound_filters(&bound).ok()?;
        // The executor filters SOLELY via the predicate (the SQL->Expr contract) — clear the bound filters.
        bound.filter = None;
        bound.filters.clear();
        bound.filter_groups.clear();

        Some(match shape {
            StreamShape::Reduction(agg) => self.run_streaming_reduction_fold(
                select,
                &table,
                &bound,
                predicate.as_ref(),
                copin_s,
                agg,
                gpu_id,
                budget,
            ),
            StreamShape::Projection => self.run_streaming_projection_fold(
                select,
                &table,
                &bound,
                predicate.as_ref(),
                copin_s,
                gpu_id,
                budget,
            ),
            StreamShape::Grouped {
                normalized,
                distinct_key_only,
            } => self.run_streaming_grouped_fold(
                select,
                &table,
                normalized,
                distinct_key_only,
                &bound,
                predicate.as_ref(),
                copin_s,
                gpu_id,
                budget,
            ),
        })
    }

    /// The fold driver: scan the table's MVCC-visible rows at the pinned boundary, accumulate them into
    /// byte-bounded chunks, and reduce+combine each chunk on the device. Peak device residency stays at
    /// one chunk (<= the budget). Any executor error on a chunk defers the WHOLE query to the CPU path.
    #[allow(clippy::too_many_arguments)]
    fn run_streaming_reduction_fold(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        agg: StreamAgg,
        gpu_id: u16,
        budget: u64,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Chunk to HALF the budget so the transient device payload + the executor's scratch/output buffers
        // stay within the budget together (the out-of-core invariant the peak-bytes gauge proves).
        let chunk_target_bytes = (budget / 2).max(1);
        let count_select = {
            let mut s = select.clone();
            s.projection = SelectProjection::CountAll;
            s
        };
        let count_bound = bind_relational_select(table, &count_select)?;

        let mut accum = StreamAccum::new(agg);
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        let mut chunks_run: u64 = 0;

        let prefix = relational_key_prefix(&select.table);
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        {
            let table_rows = self.read_state.mvcc.table_rows(&select.table);
            let mut cursor = table_rows.store().seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let decoded = decode_relational_row(&tuple.value, &table.columns)?;
                // Size by the DEVICE payload bytes this row occupies (a NULL / empty-text cell still
                // takes its fixed typed slot on the device), NOT the logical value bytes — else a
                // null-heavy table never flushes and the whole table uploads as one chunk (the
                // out-of-core bound would break; audit Finding 1).
                chunk_bytes =
                    chunk_bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
                chunk_rows.push(decoded);
                if chunk_bytes >= chunk_target_bytes {
                    match self.reduce_streaming_chunk(
                        select,
                        table,
                        bound,
                        predicate,
                        copin_s,
                        agg,
                        gpu_id,
                        &chunk_rows,
                        &count_select,
                        &count_bound,
                        &mut accum,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                            return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    chunks_run += 1;
                    chunk_rows.clear();
                    chunk_bytes = 0;
                }
            }
        }
        // The final (partial) chunk. When the table is EMPTY (or the tail cleared exactly), still run one
        // chunk so the aggregate gets its PG empty-set semantics (COUNT -> 0, SUM/MIN/MAX -> NULL).
        if !chunk_rows.is_empty() || chunks_run == 0 {
            match self.reduce_streaming_chunk(
                select,
                table,
                bound,
                predicate,
                copin_s,
                agg,
                gpu_id,
                &chunk_rows,
                &count_select,
                &count_bound,
                &mut accum,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            chunks_run += 1;
        }

        let value = accum.finalize()?;
        // Non-vacuity + out-of-core telemetry: the fold fired on the GPU, over `chunks_run` bounded chunks.
        self.read_state
            .residency
            .streaming_fold_hits
            .fetch_add(1, Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_fold_chunks
            .fetch_add(chunks_run, Ordering::Relaxed);

        Ok(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns.clone()),
            rows: vec![vec![value]].into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        })
    }

    /// STRATA S-E.2 — the filter/project fold: scan the visible rows into byte-bounded chunks; per chunk,
    /// run the projection (predicate + column gather ON THE DEVICE, with a device-side LIMIT bounding the
    /// gather to the rows still needed) and CONCAT the returned rows — the ARCHITECTURE §13 projection
    /// combine. LIMIT/OFFSET are cross-chunk WINDOWING of the concatenated survivor stream: the same
    /// control-plane index slicing the executor itself performs on its survivor vector ("LIMIT/OFFSET as
    /// control-plane WINDOWING", engine_expr.rs) — LIMIT without ORDER BY is any-N-rows per SQL, so
    /// scan-order windowing is a valid instance. A satisfied LIMIT stops the scan EARLY: the tail of the
    /// table is never even staged (the out-of-core win compounds).
    #[allow(clippy::too_many_arguments)]
    fn run_streaming_projection_fold(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        gpu_id: u16,
        budget: u64,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let chunk_target_bytes = (budget / 2).max(1);
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        // The per-chunk select: the SAME projection + WHERE, OFFSET dropped (applied cross-chunk — a
        // chunk cannot know the prior chunks' survivor count at bind time) and LIMIT re-derived per
        // chunk from the rows still needed (skip-span rows must also be gathered; they are dropped at
        // the cross-chunk window below — a bounded cost of OFFSET over a stream).
        let mut chunk_select = select.clone();
        chunk_select.offset = None;
        chunk_select.limit = None;

        let mut remaining_skip: usize = select.offset.unwrap_or(0);
        let mut remaining_take: Option<usize> = select.limit;
        let mut rows_out: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        let mut chunks_run: u64 = 0;

        let prefix = relational_key_prefix(&select.table);
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        {
            let table_rows = self.read_state.mvcc.table_rows(&select.table);
            let mut cursor = table_rows.store().seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                // LIMIT satisfied -> STOP the scan: no further row is staged, decoded, or uploaded.
                if remaining_take == Some(0) {
                    break;
                }
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let decoded = decode_relational_row(&tuple.value, &table.columns)?;
                chunk_bytes =
                    chunk_bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
                chunk_rows.push(decoded);
                if chunk_bytes >= chunk_target_bytes {
                    match self.project_streaming_chunk(
                        &mut chunk_select,
                        table,
                        predicate,
                        copin_s,
                        &chunk_rows,
                        &mut remaining_skip,
                        &mut remaining_take,
                        &mut rows_out,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                            return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    chunks_run += 1;
                    chunk_rows.clear();
                    chunk_bytes = 0;
                }
            }
        }
        // The final (partial) chunk — skipped when the LIMIT already filled (rows staged before the
        // early-exit tripped would be dropped by the window anyway; don't upload them).
        if !chunk_rows.is_empty() && remaining_take != Some(0) {
            match self.project_streaming_chunk(
                &mut chunk_select,
                table,
                predicate,
                copin_s,
                &chunk_rows,
                &mut remaining_skip,
                &mut remaining_take,
                &mut rows_out,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            chunks_run += 1;
        }

        self.read_state
            .residency
            .streaming_fold_hits
            .fetch_add(1, Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_fold_chunks
            .fetch_add(chunks_run, Ordering::Relaxed);

        Ok(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns.clone()),
            rows: rows_out.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        })
    }

    /// Upload one chunk as a transient resident source, run the projection on the device (bounded by a
    /// device-side LIMIT of skip+take), then CONCAT the survivors into `rows_out` through the cross-chunk
    /// OFFSET/LIMIT window. The transient source drops at the end of the call (one chunk resident).
    #[allow(clippy::too_many_arguments)]
    fn project_streaming_chunk(
        &self,
        chunk_select: &mut Select,
        table: &RelationalTable,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        chunk_rows: &[Vec<SqlValue>],
        remaining_skip: &mut usize,
        remaining_take: &mut Option<usize>,
        rows_out: &mut Vec<Vec<SqlValue>>,
    ) -> ChunkOutcome {
        // Device gather bound: the skip-span rows must still be gathered (dropped at the window below),
        // so the device LIMIT is skip + take. No LIMIT -> unbounded (every survivor gathers).
        chunk_select.limit = remaining_take.map(|take| remaining_skip.saturating_add(take));
        let chunk_bound = match bind_relational_select(table, chunk_select) {
            Ok(chunk_bound) => chunk_bound,
            Err(_) => return ChunkOutcome::Defer,
        };
        let (descriptor, device_memory) =
            match self.build_transient_relation_residency(table, chunk_rows) {
                Ok(pair) => pair,
                Err(_) => return ChunkOutcome::Defer,
            };
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(descriptor.resident_bytes, Ordering::Relaxed);
        let src = ResidentExecSource {
            descriptor: Arc::new(descriptor),
            device_memory: Arc::new(device_memory),
            row_count: chunk_rows.len() as u64,
        };
        let result = match self.execute_resident_expr_select_with_binding(
            chunk_select,
            table,
            Some(&src),
            chunk_bound,
            copin_s,
            predicate,
            None,
            &[],
            &[],
            None,
            &[],
        ) {
            Ok(result) => result,
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        let mut rows = result.rows.into_boxed();
        // Cross-chunk OFFSET: drop this chunk's survivors that fall inside the remaining skip span.
        if *remaining_skip > 0 {
            let dropped = (*remaining_skip).min(rows.len());
            rows.drain(..dropped);
            *remaining_skip -= dropped;
        }
        // Cross-chunk LIMIT: keep only the rows still needed (the scan early-exits once this hits 0).
        if let Some(take) = remaining_take {
            let kept = rows.len().min(*take);
            rows.truncate(kept);
            *take -= kept;
        }
        rows_out.append(&mut rows);
        ChunkOutcome::Ok
    }

    /// STRATA S-E.3 — the GROUP BY / DISTINCT two-level fold. LEVEL 1: each byte-bounded chunk runs the
    /// (normalized) grouped aggregate ON THE DEVICE, producing partial group rows `(key, agg_1..agg_N)`.
    /// LEVEL 2: the partials CONCAT (control-plane, the S-E.2 combine) into an accumulator that is itself
    /// a synthesized relation, and ONE final device grouped pass MERGES them — COUNT folds as SUM(count),
    /// SUM as SUM(sum), MIN as MIN(min), MAX as MAX(max) — so the host never groups or aggregates; it only
    /// stages partials and re-types merged cells (the same control-plane narrowing class as the scalar
    /// fold's finalize). If the accumulator outgrows the chunk budget mid-scan it is COMPACTED by the same
    /// device merge (the "persistent accumulator" realized as periodic re-merge); if even the compacted
    /// (true-cardinality) partials exceed the budget, the query defers to the CPU path — honest coverage.
    /// DISTINCT rides this fold via the distinct bridge's own synthesis (GROUP BY col + COUNT dropped).
    #[allow(clippy::too_many_arguments)]
    fn run_streaming_grouped_fold(
        &self,
        select: &Select,
        table: &RelationalTable,
        normalized: SelectProjection,
        distinct_key_only: bool,
        bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        gpu_id: u16,
        budget: u64,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let chunk_target_bytes = (budget / 2).max(1);
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let SelectProjection::GroupedAggregates {
            group_column,
            aggregates,
        } = &normalized
        else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        // The per-chunk grouped select: the normalized projection over the ORIGINAL table (WHERE rides
        // the predicate; ORDER BY/LIMIT/HAVING are absent by the classifier).
        let mut grouped_select = select.clone();
        grouped_select.distinct = false;
        grouped_select.group_by = Some(group_column.clone());
        grouped_select.projection = normalized.clone();
        let Ok(grouped_bound) = bind_relational_select(table, &grouped_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        // The synthesized PARTIALS relation: `(key, __p0..__pN)` with each partial column typed by its
        // aggregate — Count -> Int8; Sum(int2/int4) -> Int8, Sum(int8) -> Numeric(38,0), Sum(numeric) ->
        // the column's numeric type; Min/Max -> the value column's own type. An unsupported combination
        // (e.g. SUM over text) declines to the CPU path, which raises the proper SQL error.
        let key_type = match table
            .columns
            .iter()
            .find(|column| &column.name == group_column)
        {
            Some(column) => column.ty,
            None => return self.execute_relational_select_cpu_pinned(select),
        };
        let mut partial_columns: Vec<(String, SqlType)> =
            vec![(group_column.clone(), key_type)];
        // Guard the reserved partial names (a user column literally named `__pN` would collide).
        if group_column.starts_with("__p") {
            return self.execute_relational_select_cpu_pinned(select);
        }
        for (i, aggregate) in aggregates.iter().enumerate() {
            let partial_type = match aggregate.kind {
                GroupedAggKind::Count => Some(SqlType::Int8),
                GroupedAggKind::Sum | GroupedAggKind::Min | GroupedAggKind::Max => {
                    let value_type = aggregate.value_column.as_ref().and_then(|name| {
                        table
                            .columns
                            .iter()
                            .find(|column| &column.name == name)
                            .map(|column| column.ty)
                    });
                    match (aggregate.kind, value_type) {
                        (GroupedAggKind::Sum, Some(SqlType::Int2 | SqlType::Int4)) => {
                            Some(SqlType::Int8)
                        }
                        (GroupedAggKind::Sum, Some(SqlType::Int8)) => Some(SqlType::Numeric {
                            precision: 38,
                            scale: 0,
                        }),
                        (GroupedAggKind::Sum, Some(numeric @ SqlType::Numeric { .. })) => {
                            Some(numeric)
                        }
                        (GroupedAggKind::Min | GroupedAggKind::Max, Some(value_type)) => {
                            Some(value_type)
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            match partial_type {
                Some(ty) => partial_columns.push((format!("__p{i}"), ty)),
                None => return self.execute_relational_select_cpu_pinned(select),
            }
        }
        let partial_types: Vec<SqlType> = partial_columns.iter().map(|(_, ty)| *ty).collect();
        let partial_column_refs: Vec<(&str, SqlType)> = partial_columns
            .iter()
            .map(|(name, ty)| (name.as_str(), *ty))
            .collect();
        let partials_table =
            catalog_relation_table(&table.schema, "__stream_partials", &partial_column_refs);

        // The MERGE select over the partials relation: GROUP BY key with the fold aggregate per column.
        let merge_select = Select {
            table: partials_table.name.clone(),
            distinct: false,
            projection: SelectProjection::GroupedAggregates {
                group_column: group_column.clone(),
                aggregates: aggregates
                    .iter()
                    .enumerate()
                    .map(|(i, aggregate)| GroupedAggregate {
                        kind: match aggregate.kind {
                            // COUNT partials fold by SUMMING; SUM partials by SUMMING.
                            GroupedAggKind::Count | GroupedAggKind::Sum => GroupedAggKind::Sum,
                            other => other,
                        },
                        value_column: Some(format!("__p{i}")),
                    })
                    .collect(),
            },
            group_by: Some(group_column.clone()),
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        let Ok(merge_bound) = bind_relational_select(&partials_table, &merge_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };
        // Which merged columns re-type back to Int8 (the merge SUM widens Int8 partials to Numeric(_,0);
        // COUNT and SUM-over-int2/int4 are Int8-typed partials — their merged cells narrow back).
        let narrow_to_int8: Vec<bool> = aggregates
            .iter()
            .zip(partial_types.iter().skip(1))
            .map(|(aggregate, partial_ty)| {
                matches!(
                    aggregate.kind,
                    GroupedAggKind::Count | GroupedAggKind::Sum
                ) && matches!(partial_ty, SqlType::Int8)
            })
            .collect();

        let mut partials_acc: Vec<Vec<SqlValue>> = Vec::new();
        let mut partials_bytes: u64 = 0;
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        let mut chunks_run: u64 = 0;

        let prefix = relational_key_prefix(&select.table);
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        {
            let table_rows = self.read_state.mvcc.table_rows(&select.table);
            let mut cursor = table_rows.store().seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let decoded = decode_relational_row(&tuple.value, &table.columns)?;
                chunk_bytes =
                    chunk_bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
                chunk_rows.push(decoded);
                if chunk_bytes >= chunk_target_bytes {
                    match self.grouped_streaming_chunk(
                        &grouped_select,
                        table,
                        &grouped_bound,
                        predicate,
                        copin_s,
                        group_column,
                        &chunk_rows,
                        &partial_types,
                        &mut partials_acc,
                        &mut partials_bytes,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                            return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    chunks_run += 1;
                    chunk_rows.clear();
                    chunk_bytes = 0;
                    // COMPACTION: the accumulator outgrew the chunk budget — device-merge it down to
                    // one row per true group. If even the compacted form exceeds the budget, the group
                    // cardinality itself is over-budget: defer (S-E.4+ may spill; v1 is honest).
                    if partials_bytes >= chunk_target_bytes {
                        match self.merge_streaming_partials(
                            &merge_select,
                            &partials_table,
                            &merge_bound,
                            copin_s,
                            group_column,
                            &narrow_to_int8,
                            &mut partials_acc,
                            &mut partials_bytes,
                            &partial_types,
                            budget,
                        ) {
                            ChunkOutcome::Ok => {}
                            ChunkOutcome::Defer => {
                                return self.execute_relational_select_cpu_pinned(select)
                            }
                            ChunkOutcome::Hard(err) => return Err(err),
                        }
                        if partials_bytes >= chunk_target_bytes {
                            return self.execute_relational_select_cpu_pinned(select);
                        }
                    }
                }
            }
        }
        if !chunk_rows.is_empty() {
            match self.grouped_streaming_chunk(
                &grouped_select,
                table,
                &grouped_bound,
                predicate,
                copin_s,
                group_column,
                &chunk_rows,
                &partial_types,
                &mut partials_acc,
                &mut partials_bytes,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            chunks_run += 1;
        }

        // FINAL MERGE: fold duplicate keys across chunks into the one true group table. An empty
        // accumulator (empty table / all rows filtered) is PG's empty grouped result: ZERO rows.
        if !partials_acc.is_empty() {
            match self.merge_streaming_partials(
                &merge_select,
                &partials_table,
                &merge_bound,
                copin_s,
                group_column,
                &narrow_to_int8,
                &mut partials_acc,
                &mut partials_bytes,
                &partial_types,
                budget,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
        }
        let mut rows_out = std::mem::take(&mut partials_acc);
        // DISTINCT: drop the synthesized COUNT column — the bare distinct keys.
        if distinct_key_only {
            for row in &mut rows_out {
                row.truncate(1);
            }
        }

        self.read_state
            .residency
            .streaming_fold_hits
            .fetch_add(1, Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_fold_chunks
            .fetch_add(chunks_run, Ordering::Relaxed);

        // Columns: DISTINCT keeps the outer (single-column) binding; grouped uses the NORMALIZED
        // binding (byte-identical to the grouped bridge, which also binds the normalized form).
        let columns = if distinct_key_only {
            bound.selected_columns.clone()
        } else {
            grouped_bound.selected_columns.clone()
        };
        Ok(RelationalSelectResult {
            columns: Arc::new(columns),
            rows: rows_out.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        })
    }

    /// LEVEL 1 of the grouped fold: upload one chunk, run the grouped aggregate ON THE DEVICE, and append
    /// its partial group rows to the accumulator (concat — the control-plane combine).
    #[allow(clippy::too_many_arguments)]
    fn grouped_streaming_chunk(
        &self,
        grouped_select: &Select,
        table: &RelationalTable,
        grouped_bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        group_column: &str,
        chunk_rows: &[Vec<SqlValue>],
        partial_types: &[SqlType],
        partials_acc: &mut Vec<Vec<SqlValue>>,
        partials_bytes: &mut u64,
    ) -> ChunkOutcome {
        let (descriptor, device_memory) =
            match self.build_transient_relation_residency(table, chunk_rows) {
                Ok(pair) => pair,
                Err(_) => return ChunkOutcome::Defer,
            };
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(descriptor.resident_bytes, Ordering::Relaxed);
        let src = ResidentExecSource {
            descriptor: Arc::new(descriptor),
            device_memory: Arc::new(device_memory),
            row_count: chunk_rows.len() as u64,
        };
        let group_key_columns = [group_column.to_string()];
        let result = match self.execute_resident_expr_select_with_binding(
            grouped_select,
            table,
            Some(&src),
            grouped_bound.clone(),
            copin_s,
            predicate,
            None,
            &[],
            &[],
            None,
            &group_key_columns,
        ) {
            Ok(result) => result,
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        for row in result.rows.into_boxed() {
            *partials_bytes =
                partials_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
            partials_acc.push(row);
        }
        ChunkOutcome::Ok
    }

    /// LEVEL 2 of the grouped fold: upload the accumulated partials as a transient relation and run ONE
    /// device grouped pass that MERGES duplicate keys (SUM/SUM/MIN/MAX per column), then re-type the
    /// merged cells back to the canonical partial types (the merge SUM widens Int8 partials to
    /// Numeric(_,0); narrowing back is the same control-plane cast as the scalar fold's finalize — a
    /// narrow overflow defers, matching PG's own running-accumulation error surface). Replaces the
    /// accumulator in place. Serves BOTH the mid-scan compaction and the final merge.
    ///
    /// BUDGET GATE (S-E.3 audit Finding 1): a partial row can be WIDER than its source rows (a 4-byte
    /// key + an 8-byte count = 12B partials from 4B rows), so a near-unique-key chunk can inflate the
    /// accumulator past the budget before the over-cardinality defer triggers. The merge must never be
    /// the thing that busts the budget it exists to honor — defer WITHOUT uploading when the accumulator
    /// exceeds it (the CPU path serves the query; the peak-bytes gauge invariant stays <= budget).
    #[allow(clippy::too_many_arguments)]
    fn merge_streaming_partials(
        &self,
        merge_select: &Select,
        partials_table: &RelationalTable,
        merge_bound: &BoundRelationalSelect,
        copin_s: Index,
        group_column: &str,
        narrow_to_int8: &[bool],
        partials_acc: &mut Vec<Vec<SqlValue>>,
        partials_bytes: &mut u64,
        partial_types: &[SqlType],
        budget: u64,
    ) -> ChunkOutcome {
        if *partials_bytes > budget {
            return ChunkOutcome::Defer;
        }
        let (descriptor, device_memory) =
            match self.build_transient_relation_residency(partials_table, partials_acc) {
                Ok(pair) => pair,
                Err(_) => return ChunkOutcome::Defer,
            };
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(descriptor.resident_bytes, Ordering::Relaxed);
        let src = ResidentExecSource {
            descriptor: Arc::new(descriptor),
            device_memory: Arc::new(device_memory),
            row_count: partials_acc.len() as u64,
        };
        let group_key_columns = [group_column.to_string()];
        let result = match self.execute_resident_expr_select_with_binding(
            merge_select,
            partials_table,
            Some(&src),
            merge_bound.clone(),
            copin_s,
            None,
            None,
            &[],
            &[],
            None,
            &group_key_columns,
        ) {
            Ok(result) => result,
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        let mut merged: Vec<Vec<SqlValue>> = Vec::new();
        let mut merged_bytes: u64 = 0;
        for mut row in result.rows.into_boxed() {
            // Re-type: merged column i+1 narrows Numeric(_,0) -> Int8 where the partial is Int8-typed.
            for (agg_idx, narrow) in narrow_to_int8.iter().enumerate() {
                if !narrow {
                    continue;
                }
                let cell = &mut row[agg_idx + 1];
                match cell {
                    SqlValue::Numeric(d) if d.scale == 0 => match i64::try_from(d.mantissa) {
                        Ok(narrowed) => *cell = SqlValue::Int8(narrowed),
                        Err(_) => return ChunkOutcome::Defer,
                    },
                    SqlValue::Null | SqlValue::Int8(_) => {}
                    _ => return ChunkOutcome::Defer,
                }
            }
            merged_bytes =
                merged_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
            merged.push(row);
        }
        *partials_acc = merged;
        *partials_bytes = merged_bytes;
        ChunkOutcome::Ok
    }

    /// Upload one chunk as a transient resident source, reduce it on the device, and fold the partial in.
    /// One upload; a COUNT(*) launch (the empty-filtered-set guard + the COUNT value); and, for
    /// SUM/MIN/MAX with survivors, the reduction launch. The transient source drops (device memory freed)
    /// at the end of this call — so only this chunk is ever resident.
    #[allow(clippy::too_many_arguments)]
    fn reduce_streaming_chunk(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        agg: StreamAgg,
        gpu_id: u16,
        chunk_rows: &[Vec<SqlValue>],
        count_select: &Select,
        count_bound: &BoundRelationalSelect,
        accum: &mut StreamAccum,
    ) -> ChunkOutcome {
        let (descriptor, device_memory) =
            match self.build_transient_relation_residency(table, chunk_rows) {
                Ok(pair) => pair,
                Err(_) => return ChunkOutcome::Defer,
            };
        // The out-of-core proof: the ACTUAL transient device bytes for this chunk (fetch_max monotonic).
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(descriptor.resident_bytes, Ordering::Relaxed);
        let src = ResidentExecSource {
            descriptor: Arc::new(descriptor),
            device_memory: Arc::new(device_memory),
            row_count: chunk_rows.len() as u64,
        };
        let _ = gpu_id;

        // COUNT(*) over the chunk (predicate applied on-device): both the COUNT value AND the empty-set
        // guard for SUM/MIN/MAX (the general reduction hard-errors over an empty filtered set).
        let chunk_count = match self.execute_resident_expr_select_with_binding(
            count_select,
            table,
            Some(&src),
            count_bound.clone(),
            copin_s,
            predicate,
            None,
            &[],
            &[],
            None,
            &[],
        ) {
            Ok(result) => match result.rows.iter().next().and_then(|row| row.first()) {
                Some(SqlValue::Int8(n)) => *n,
                _ => return ChunkOutcome::Defer,
            },
            Err(_) => return ChunkOutcome::Defer,
        };

        if agg == StreamAgg::Count {
            accum.add_count(chunk_count);
            return ChunkOutcome::Ok;
        }
        if chunk_count == 0 {
            // No surviving row in this chunk -> nothing to reduce (avoids the empty-set hard-error).
            return ChunkOutcome::Ok;
        }
        let value = match self.execute_resident_expr_select_with_binding(
            select,
            table,
            Some(&src),
            bound.clone(),
            copin_s,
            predicate,
            None,
            &[],
            &[],
            None,
            &[],
        ) {
            Ok(result) => match result.rows.iter().next().and_then(|row| row.first()) {
                Some(cell) => cell.clone(),
                None => return ChunkOutcome::Defer,
            },
            // A genuine arithmetic OVERFLOW must SURFACE (PG errors on it); the CPU path cannot compute a
            // wider-type reduction anyway, so deferring would mask it with a misleading message and make
            // per-chunk overflow disagree with cross-chunk overflow (audit Finding 2). Any other error
            // (an un-expressible shape) defers to the authoritative CPU path.
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        accum.combine_value(value)
    }

    /// STRATA S-E.1 telemetry: streaming folds served (non-vacuity — an over-VRAM aggregate stayed on the
    /// GPU), total chunks reduced, and the peak single-chunk device bytes (the out-of-core proof).
    pub fn streaming_fold_hits(&self) -> u64 {
        self.read_state
            .residency
            .streaming_fold_hits
            .load(Ordering::Relaxed)
    }

    pub fn streaming_fold_chunks(&self) -> u64 {
        self.read_state
            .residency
            .streaming_fold_chunks
            .load(Ordering::Relaxed)
    }

    pub fn streaming_fold_peak_chunk_bytes(&self) -> u64 {
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .load(Ordering::Relaxed)
    }
}
