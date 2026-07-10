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
//! Follow-ons (see PLAN S-E): streaming filter/project + LIMIT (S-E.2), GROUP BY / DISTINCT with a
//! persistent device accumulator (S-E.3), ORDER BY via k-way run merge (S-E.4), and copy/compute overlap
//! (S-E.5). AVG is deferred (it needs the (sum, count) pair combined, not the divided per-chunk average).

use super::*;

use crate::engine_expr::{resident_predicate_from_bound_filters, ResidentExecSource, ResidentExpr};
use crate::rel_exec_helpers::{
    bind_relational_select, compare_sql_values, decode_relational_row, relational_key_prefix,
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

/// Classify a SELECT as a foldable SCALAR reduction, or `None` if any non-foldable shape is present
/// (GROUP BY / DISTINCT / ORDER BY / LIMIT / OFFSET / HAVING, or a non-{COUNT*,SUM,MIN,MAX} projection).
fn streaming_reduction_agg(select: &Select) -> Option<StreamAgg> {
    if select.distinct
        || select.group_by.is_some()
        || !select.order_by.is_empty()
        || select.limit.is_some()
        || select.offset.is_some()
        || !select.having_groups.is_empty()
    {
        return None;
    }
    match &select.projection {
        SelectProjection::CountAll => Some(StreamAgg::Count),
        SelectProjection::Sum { .. } => Some(StreamAgg::Sum),
        SelectProjection::Min { .. } => Some(StreamAgg::Min),
        SelectProjection::Max { .. } => Some(StreamAgg::Max),
        _ => None,
    }
}

impl Engine {
    /// STRATA S-E.1: try to serve a scalar reduction OUT-OF-CORE via the streaming fold. Returns
    /// `Some(result)` when the streaming path handled the read (`Ok`) or must surface a genuine SQL error
    /// (`Err`); `None` to fall through to the caller's path (the CPU pinned read). It NEVER returns a wrong
    /// answer: any shape the device cannot express defers to the authoritative CPU path.
    pub(crate) fn try_streaming_scalar_reduction(
        &self,
        select: &Select,
    ) -> Option<Result<RelationalSelectResult, ExecuteError>> {
        let agg = streaming_reduction_agg(select)?;
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

        Some(self.run_streaming_reduction_fold(select, &table, &bound, predicate.as_ref(), copin_s, agg, gpu_id, budget))
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
