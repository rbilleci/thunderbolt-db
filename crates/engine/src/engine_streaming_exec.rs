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
    bind_relational_select, catalog_relation_table, decode_relational_row, relational_key_prefix,
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

/// A chunk's device reduction either combined cleanly, or hit a shape the streaming path should hand back
/// to the authoritative CPU pinned path (`Defer`), or a genuine SQL error to surface (`Hard`).
enum ChunkOutcome {
    Ok,
    Defer,
    Hard(ExecuteError),
}

/// STRATA S-E.5: a chunk whose transient device upload is IN FLIGHT on a private copy stream. The fold
/// stages chunk N, then computes chunk N-1 — so N's PCIe DMA overlaps N-1's kernels (separate streams)
/// and the host's staging of N+1 (MEASURED: the fold is host-staging-bound at ~68%, upload ~9%, compute
/// ~1%; the lookahead hides the upload+compute slices under the host build — the host build itself is
/// the interim MVCC-store decode, ADR-006 deletion debt retired by the raw-shard-bytes cold tier, S-E.6).
/// Residency invariant: at most TWO chunks are transiently resident (one uploading + one computing),
/// each <= the chunk target = budget/2, so the total stays <= the budget.
struct StagedChunk {
    snapshot: RelationalResidencySnapshot,
    pending: gpu_db_execution::PendingCudaResidentDeviceCopy,
    row_count: u64,
}

impl StagedChunk {
    /// Block until the upload completes and wrap the chunk as an executor source.
    fn ready(self) -> Result<ResidentExecSource, ()> {
        let device_memory = self.pending.wait().map_err(|_| ())?;
        Ok(ResidentExecSource {
            descriptor: Arc::new(self.snapshot),
            device_memory: Arc::new(device_memory),
            row_count: self.row_count,
        })
    }
}

/// STRATA S-E.6 — the streaming COLD TIER: one table's DEVICE-FORMAT chunk payloads, cached in host
/// RAM after a fold's first (MVCC-scan) build and REPLAYED byte-for-byte by later streaming reads.
/// The replay skips the per-row MVCC decode + columnar payload assembly (MEASURED at ~68% of fold
/// wall-clock — the ADR-006 interim-store staging cost) — a cold read is upload + device compute.
/// Validity: `generation` pins the tuple-store generation-payload Arc the build scanned; a HIT
/// requires pointer equality with the table's CURRENT generation (every write COW-publishes a fresh
/// Arc, so equality proves not one write has touched the table — and reads pin the monotonic
/// `committed_seq`, so the later reader's visible set is identical). Holding the Arc makes the check
/// ABA-safe and costs only the structural-sharing delta. `chunk_target_bytes` must also match (a
/// budget change re-chunks).
pub(crate) struct ColdTableChunks {
    generation: Arc<crate::resident_storage::TableVersionData>,
    /// 6c-1 (the ALTER guard): the catalog column layout the payloads were built with. A patch
    /// REUSES cached chunk bytes, so a shape-changing DDL (which also republishes the store) must
    /// evict rather than patch — layout inequality forces the evict arm.
    column_signature: Vec<(String, SqlType)>,
    /// The build's pinned boundary, PROVEN SETTLED at install (under the commit lock,
    /// `committed_seq == build_copin_s` with the generation unchanged — so the generation contains
    /// NO stamp above it). A hit additionally requires `reader_copin_s >= build_copin_s`: every
    /// stamp <= build <= reader makes the visible set boundary-invariant, closing the audit-F1
    /// publish-before-seq-bump window (a reader pinned BELOW a stamp baked into the replay).
    build_copin_s: Index,
    chunk_target_bytes: u64,
    total_payload_bytes: u64,
    /// S-E.6b: this table's payloads live in the unlinked spill file (counts against the DISK cap,
    /// not the RAM cap).
    spilled: bool,
    chunks: Vec<ColdChunk>,
}

/// One cached chunk: the exact device payload bytes + the descriptor template the build produced.
pub(crate) struct ColdChunk {
    payload: ColdPayload,
    snapshot: RelationalResidencySnapshot,
    row_count: u64,
    /// 6c-1: the INCLUSIVE TupleId range this chunk's rows were scanned from (scan order IS
    /// TupleId order — the S-E.2 determinism fact). `(1, 0)` = the empty chunk (no rows). A write's
    /// changed TupleIds map to dirty chunks through these ranges; untouched ranges REUSE their
    /// bytes verbatim (chain identity: an untouched range's version chains are pointer-identical
    /// across the COW generations, so its visible set at any settled boundary is unchanged).
    tuple_range: (u64, u64),
}

/// S-E.6b: where a cached chunk's payload bytes live — host RAM below the spill threshold, or an
/// UNLINKED spill file above it (created then `remove_file`d with the handle kept open: the OS
/// reclaims the space on the last close — crash-safe, zero litter, no startup sweep; positional
/// `read_exact_at` reads are seek-free and thread-safe). The spill directory honors `TMPDIR`
/// (production points it at NVMe; the test profile already pins it to `target/tmp`).
enum ColdPayload {
    Ram(Arc<Vec<u8>>),
    Spilled {
        file: Arc<std::fs::File>,
        offset: u64,
        len: usize,
    },
}

impl ColdPayload {
    /// Materialize the payload bytes for a replay upload. A spill-read failure is an `Err` the
    /// caller turns into a MISS/defer — never a wrong answer.
    fn read(&self) -> Result<std::borrow::Cow<'_, [u8]>, ()> {
        match self {
            ColdPayload::Ram(bytes) => Ok(std::borrow::Cow::Borrowed(bytes)),
            ColdPayload::Spilled { file, offset, len } => {
                use std::os::unix::fs::FileExt;
                let mut bytes = vec![0u8; *len];
                file.read_exact_at(&mut bytes, *offset).map_err(|_| ())?;
                Ok(std::borrow::Cow::Owned(bytes))
            }
        }
    }
}

/// Accumulates a fold's scan-built chunks for install (discarded on any defer/error/early-exit —
/// only a COMPLETE scan installs). S-E.6b: once the captured bytes cross the SPILL threshold the
/// builder opens an unlinked spill file, retro-writes the RAM chunks captured so far, and
/// write-throughs every later chunk — so an over-RAM table's capture never holds its payloads in
/// host memory (the very tables streaming exists for are the ones that must spill).
struct ColdCacheBuilder {
    generation: Arc<crate::resident_storage::TableVersionData>,
    build_copin_s: Index,
    chunk_target_bytes: u64,
    total_payload_bytes: u64,
    column_signature: Vec<(String, SqlType)>,
    chunks: Vec<ColdChunk>,
    /// The open spill file + its append offset once the threshold tripped (`None` = all-RAM).
    spill: Option<(Arc<std::fs::File>, u64)>,
    /// A spill IO error poisons the capture (the fold keeps running; the install is skipped).
    poisoned: bool,
}

impl ColdCacheBuilder {
    /// Append one captured chunk, spilling at the threshold. On any IO error the builder poisons
    /// itself (no install) — the fold's own compute path is unaffected.
    fn push(
        &mut self,
        payload: Vec<u8>,
        snapshot: RelationalResidencySnapshot,
        row_count: u64,
        tuple_range: (u64, u64),
    ) {
        use std::io::Write;
        if self.poisoned {
            return;
        }
        self.total_payload_bytes = self.total_payload_bytes.saturating_add(payload.len() as u64);
        if self.spill.is_none() && self.total_payload_bytes > streaming_cold_spill_threshold() {
            // Threshold crossed: open the unlinked spill file and retro-write the RAM prefix.
            let Ok(file) = unlinked_spill_file() else {
                self.poisoned = true;
                return;
            };
            let mut writer: &std::fs::File = file.as_ref();
            let mut offset = 0u64;
            for chunk in &mut self.chunks {
                let ColdPayload::Ram(bytes) = &chunk.payload else {
                    self.poisoned = true;
                    return;
                };
                if writer.write_all(bytes).is_err() {
                    self.poisoned = true;
                    return;
                }
                let len = bytes.len();
                chunk.payload = ColdPayload::Spilled {
                    file: Arc::clone(&file),
                    offset,
                    len,
                };
                offset += len as u64;
            }
            self.spill = Some((file, offset));
        }
        let cold_payload = match &mut self.spill {
            None => ColdPayload::Ram(Arc::new(payload)),
            Some((file, offset)) => {
                use std::io::Write;
                let mut writer = file.as_ref();
                if writer.write_all(&payload).is_err() {
                    self.poisoned = true;
                    return;
                }
                let chunk_offset = *offset;
                *offset += payload.len() as u64;
                ColdPayload::Spilled {
                    file: Arc::clone(file),
                    offset: chunk_offset,
                    len: payload.len(),
                }
            }
        };
        self.chunks.push(ColdChunk {
            payload: cold_payload,
            snapshot,
            row_count,
            tuple_range,
        });
    }
}

/// Create an UNLINKED spill file in the `TMPDIR`-honoring temp directory: created, then
/// `remove_file`d while the handle stays open (unix) — the OS reclaims the bytes on the last
/// close, so a crash leaks nothing and no startup sweep exists to forget.
fn unlinked_spill_file() -> Result<Arc<std::fs::File>, ()> {
    let dir = std::env::temp_dir();
    // pid + monotonic seq + wall-nanos: the name exists only for the create+unlink instant, but a
    // predictable name on a SHARED temp dir would let a local nuisance pre-create it (create_new
    // fails -> poison -> CPU fallback; O_EXCL already blocks anything worse — audit LOW).
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let name = format!(
        "gpu-db-stream-spill-{}-{}-{nanos}",
        std::process::id(),
        SPILL_FILE_SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let path = dir.join(name);
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|_| ())?;
    std::fs::remove_file(&path).map_err(|_| ())?;
    Ok(Arc::new(file))
}

/// Uniquifies spill file names within the process (the path exists only for the create+unlink
/// instant, but two concurrent builds must not collide in it).
static SPILL_FILE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// S-E.6b: the RAM->spill threshold for a single table's capture. A `#[cfg(test)]` override lets
/// the GPU tests force spilling on small tables (NOT product config — the no-flag mandate).
fn streaming_cold_spill_threshold() -> u64 {
    #[cfg(test)]
    {
        let forced = STREAMING_COLD_SPILL_THRESHOLD_TEST.load(Ordering::Relaxed);
        if forced != 0 {
            return forced;
        }
    }
    STREAMING_COLD_SPILL_THRESHOLD_BYTES
}

/// Captures above this stay out of host RAM (spilled). Engine-internal, not config.
const STREAMING_COLD_SPILL_THRESHOLD_BYTES: u64 = 256 * 1024 * 1024;

#[cfg(test)]
pub(crate) static STREAMING_COLD_SPILL_THRESHOLD_TEST: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

impl std::fmt::Debug for ColdTableChunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColdTableChunks")
            .field("chunks", &self.chunks.len())
            .field("total_payload_bytes", &self.total_payload_bytes)
            .field("chunk_target_bytes", &self.chunk_target_bytes)
            .finish_non_exhaustive()
    }
}

/// The cold tier's global host-RAM cap. NOT product config (the no-flag mandate): a generous
/// engine-internal bound; exceeding it clears the map (crude, correct — the next reads rebuild).
/// The cache is INTERIM double-residency next to the MVCC tuple store it shadows; both retire with
/// ADR-006 (the sealed shard bytes become the only cold representation).
const STREAMING_COLD_CAP_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// S-E.6b: the spilled-class cap (unlinked NVMe/temp files). Engine-internal, not config.
const STREAMING_COLD_DISK_CAP_BYTES: u64 = 128 * 1024 * 1024 * 1024;

/// 6c-0 (charter-drift ruling): the SCALAR combine runs ON THE DEVICE. Per-chunk partials collect as
/// rows of a synthesized one-column relation and ONE final device aggregate pass folds them — the
/// S-E.3 synthesized-relation merge shape (`catalog_relation_table` + injected src). The former host
/// accumulator (StreamAccum: Decimal128 adds, compare_sql_values extremes) is DELETED in this merge.
/// The partial column's type + the fold aggregate per original shape (the S-E.3 typing rules):
/// COUNT -> Int8 partials folded by SUM; SUM(int2/4) -> Int8 by SUM; SUM(int8) -> Numeric{38,0} by
/// SUM; SUM(numeric{p,s}) -> same numeric by SUM; MIN/MAX -> the value type by MIN/MAX. Returns None
/// when the value type is not device-reducible (the per-chunk pass would have deferred anyway).
fn scalar_partial_plan(
    table: &RelationalTable,
    select: &Select,
    agg: StreamAgg,
) -> Option<(SqlType, SelectProjection)> {
    let value_type = |column: &String| {
        table
            .columns
            .iter()
            .find(|c| &c.name == column)
            .map(|c| c.ty)
    };
    let partial = "__p0".to_string();
    match (agg, &select.projection) {
        (StreamAgg::Count, _) => Some((SqlType::Int8, SelectProjection::Sum { column: partial })),
        (StreamAgg::Sum, SelectProjection::Sum { column }) => match value_type(column)? {
            SqlType::Int2 | SqlType::Int4 => {
                Some((SqlType::Int8, SelectProjection::Sum { column: partial }))
            }
            SqlType::Int8 => Some((
                SqlType::Numeric {
                    precision: 38,
                    scale: 0,
                },
                SelectProjection::Sum { column: partial },
            )),
            numeric @ SqlType::Numeric { .. } => {
                Some((numeric, SelectProjection::Sum { column: partial }))
            }
            _ => None,
        },
        (StreamAgg::Min, SelectProjection::Min { column }) => {
            Some((value_type(column)?, SelectProjection::Min { column: partial }))
        }
        (StreamAgg::Max, SelectProjection::Max { column }) => {
            Some((value_type(column)?, SelectProjection::Max { column: partial }))
        }
        _ => None,
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
    /// S-E.4: a single-key `ORDER BY` projection. `top_n` = the `[OFFSET,OFFSET+LIMIT)` window bound
    /// (`Some` = a top-N stream: each chunk's device-sorted local top-(skip+take) is its only possible
    /// contribution to the global window; `None` = unbounded — the whole survivor set must fit the
    /// budget for the final device sort, else defer).
    Ordered { top_n: Option<usize> },
}

/// Classify a SELECT as a streamable shape, or `None` for the non-foldable classes (ORDER BY — S-E.4;
/// HAVING; grouped AVG / COUNT(DISTINCT) — not associatively decomposable from per-chunk partials). A
/// SCALAR reduction (COUNT(*)/SUM/MIN/MAX, no LIMIT/OFFSET — PG applies LIMIT to the one-row aggregate
/// result, a shape not worth streaming) folds by combine; a plain `All`/`Columns` projection folds by
/// CONCAT, with LIMIT/OFFSET as cross-chunk windowing (LIMIT without ORDER BY is any-N-rows per SQL, so
/// early-exit + scan-order windowing is a valid instance); GROUP BY / DISTINCT fold TWO-LEVEL (per-chunk
/// device partials -> concat -> one final device merge pass).
fn streaming_shape(select: &Select) -> Option<StreamShape> {
    if !select.having_groups.is_empty() {
        return None;
    }
    // S-E.4 ORDER BY: a single-key ordered `All`/`Columns` projection streams — the sort key must ride
    // the partials (be among the projected columns) so the FINAL device sort can re-order them. Ordered
    // DISTINCT / grouped / multi-key stay declined (multi-key is hard-rejected upstream anyway).
    if !select.order_by.is_empty() {
        if select.distinct || select.group_by.is_some() || select.order_by.len() != 1 {
            return None;
        }
        let key = &select.order_by[0].column;
        // An ORDER BY EXPRESSION parses to the empty-string sentinel column (the expression rides a
        // separate order_by_exprs vector this path never receives) — decline it up front instead of
        // burning chunk uploads before the executor's "column does not exist" defer (audit LOW; the
        // grouped bridge guards the same sentinel).
        if key.is_empty() {
            return None;
        }
        let key_projected = match &select.projection {
            SelectProjection::All => true,
            SelectProjection::Columns(columns) => columns.contains(key),
            _ => return None,
        };
        if !key_projected {
            return None;
        }
        // The window bound: OFFSET-without-LIMIT has no top-N bound -> treat as unbounded.
        let top_n = select
            .limit
            .map(|limit| select.offset.unwrap_or(0).saturating_add(limit));
        return Some(StreamShape::Ordered { top_n });
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
            StreamShape::Ordered { top_n } => self.run_streaming_ordered_fold(
                select,
                &table,
                &bound,
                predicate.as_ref(),
                copin_s,
                top_n,
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
        // 6c-0: the device-combine plan (partial column type + the fold aggregate). A value type the
        // device cannot reduce declines here (the per-chunk pass would defer on it anyway).
        let Some((partial_type, fold_projection)) = scalar_partial_plan(table, select, agg) else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        let mut partials: Vec<Vec<SqlValue>> = Vec::new();
        let mut total_matched: i64 = 0;
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        // 6c-1: the current chunk's inclusive TupleId range ((1,0) = empty sentinel).
        let mut chunk_range: (u64, u64) = (1, 0);
        let mut chunks_run: u64 = 0;
        // S-E.5: the ONE-chunk lookahead — the staged chunk's upload is in flight while the previous
        // chunk computes and the next chunk's rows stage on the host.
        let mut staged: Option<StagedChunk> = None;
        // S-E.6: cold-tier replay (no MVCC decode) on a hit; a miss scans + CAPTURES for next time.
        let cold = self.load_streaming_cold(&select.table, table, chunk_target_bytes, copin_s);
        let mut capture: Option<ColdCacheBuilder> = None;

        let prefix = relational_key_prefix(&select.table);
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        if let Some(cold) = &cold {
            for chunk in &cold.chunks {
                let Ok(next) = self.stage_cold_chunk(chunk) else {
                    // A failed replay (e.g. a bad spill read) evicts the entry so the next read
                    // rebuilds instead of defer-thrashing (audit LOW).
                    self.evict_streaming_cold(&select.table);
                    return self.execute_relational_select_cpu_pinned(select);
                };
                if let Some(prev) = staged.replace(next) {
                    let Ok(src) = prev.ready() else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    match self.reduce_streaming_chunk(
                        select,
                        table,
                        bound,
                        predicate,
                        copin_s,
                        agg,
                        &src,
                        &count_select,
                        &count_bound,
                        &mut partials,
                        &mut total_matched,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                        return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    chunks_run += 1;
                }
            }
        } else {
            let table_rows = self.read_state.mvcc.table_rows(&select.table);
            capture = Some(ColdCacheBuilder {
                generation: table_rows.generation_payload(),
                build_copin_s: copin_s,
                column_signature: table
                    .columns
                    .iter()
                    .map(|c| (c.name.clone(), c.ty))
                    .collect(),
                chunk_target_bytes,
                total_payload_bytes: 0,
                chunks: Vec::new(),
                spill: None,
                poisoned: false,
            });
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
                chunk_range = if chunk_rows.is_empty() {
                    (tuple.tuple_id, tuple.tuple_id)
                } else {
                    (chunk_range.0, tuple.tuple_id)
                };
                chunk_rows.push(decoded);
                if chunk_bytes >= chunk_target_bytes {
                    // S-E.5 lookahead: stage this chunk (its upload overlaps), compute the PREVIOUS.
                    let Ok(next) = self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture) else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    if let Some(prev) = staged.replace(next) {
                        let Ok(src) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.reduce_streaming_chunk(
                            select,
                            table,
                            bound,
                            predicate,
                            copin_s,
                            agg,
                            &src,
                            &count_select,
                            &count_bound,
                            &mut partials,
                            &mut total_matched,
                        ) {
                            ChunkOutcome::Ok => {}
                            ChunkOutcome::Defer => {
                                return self.execute_relational_select_cpu_pinned(select)
                            }
                            ChunkOutcome::Hard(err) => return Err(err),
                        }
                        chunks_run += 1;
                    }
                    chunk_rows.clear();
                    chunk_bytes = 0;
                    chunk_range = (1, 0);
                }
            }
        }
        // The final (partial) chunk. When the table is EMPTY (or the tail cleared exactly), still run one
        // chunk so the aggregate gets its PG empty-set semantics (COUNT -> 0, SUM/MIN/MAX -> NULL). Then
        // DRAIN the pipeline (the last staged chunk still needs its compute).
        if !chunk_rows.is_empty() || (chunks_run == 0 && staged.is_none()) {
            let Ok(next) = self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture) else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            if let Some(prev) = staged.replace(next) {
                let Ok(src) = prev.ready() else {
                    return self.execute_relational_select_cpu_pinned(select);
                };
                match self.reduce_streaming_chunk(
                    select,
                    table,
                    bound,
                    predicate,
                    copin_s,
                    agg,
                    &src,
                    &count_select,
                    &count_bound,
                    &mut partials,
                    &mut total_matched,
                ) {
                    ChunkOutcome::Ok => {}
                    ChunkOutcome::Defer => {
                        return self.execute_relational_select_cpu_pinned(select)
                    }
                    ChunkOutcome::Hard(err) => return Err(err),
                }
                chunks_run += 1;
            }
        }
        if let Some(prev) = staged.take() {
            let Ok(src) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.reduce_streaming_chunk(
                select,
                table,
                bound,
                predicate,
                copin_s,
                agg,
                &src,
                &count_select,
                &count_bound,
                &mut partials,
                &mut total_matched,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            chunks_run += 1;
        }

        // 6c-0: the CROSS-CHUNK COMBINE runs ON THE DEVICE — one aggregate pass over the collected
        // partials as a synthesized one-column relation. Zero matched rows emits PG's empty-set
        // result (COUNT->0, else NULL) from CARDINALITY bookkeeping alone — charter basis: kernel
        // orchestration (the host may decide WHETHER to launch from row counts, and the device SUM
        // fold over an empty partial set could not represent COUNT's 0 anyway); no data VALUE is
        // read or combined on the host here (audit 6c-0 LOW: justified from charter text, not
        // precedent, per the charter-drift ruling).
        let value = if total_matched == 0 {
            match agg {
                StreamAgg::Count => SqlValue::Int8(0),
                _ => SqlValue::Null,
            }
        } else {
            match self.combine_scalar_partials_on_device(
                select,
                table,
                partial_type,
                &fold_projection,
                copin_s,
                &partials,
                budget,
            ) {
                Ok(value) => value,
                // An all-NULL partial set (every matched row NULL in every chunk) hard-errors the
                // device scalar path; the CPU path serves it — honest edge coverage, never wrong.
                Err(()) => return self.execute_relational_select_cpu_pinned(select),
            }
        };
        // 6c-0 readback boundary: the device SUM over Int8 partials returns Numeric(_,0) (PG's SUM
        // ladder); COUNT / SUM(int4) present as bigint on the wire — ONE checked narrow of the single
        // result cell at materialization (the charter's readback carve-out; a narrow overflow is PG's
        // own "bigint out of range").
        let value = match (&value, agg, matches!(partial_type, SqlType::Int8)) {
            (SqlValue::Numeric(d), StreamAgg::Count | StreamAgg::Sum, true) if d.scale == 0 => {
                match i64::try_from(d.mantissa) {
                    Ok(narrowed) => SqlValue::Int8(narrowed),
                    Err(_) => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "bigint out of range".to_string(),
                        )))
                    }
                }
            }
            _ => value,
        };
        // S-E.6: a COMPLETE scan installs its captured chunks for byte-replay by later reads.
        if let Some(builder) = capture {
            self.install_streaming_cold(&select.table, builder);
        }
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
        // 6c-0: the WINDOW BOUND = offset+limit. Per chunk the DEVICE limit is this constant (a
        // chunk's first `bound` survivors are its only possible global-window contribution); the
        // cross-chunk window itself runs as ONE final device pass. The fold's early-exit is pure
        // cardinality flow-control (`collected >= bound`), never value-based windowing on the host.
        let window_bound: Option<usize> = select
            .limit
            .map(|limit| select.offset.unwrap_or(0).saturating_add(limit));
        let mut chunk_select = select.clone();
        chunk_select.offset = None;
        chunk_select.limit = window_bound;
        let Ok(chunk_bound) = bind_relational_select(table, &chunk_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        let mut rows_out: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        // 6c-1: the current chunk's inclusive TupleId range ((1,0) = empty sentinel).
        let mut chunk_range: (u64, u64) = (1, 0);
        let mut chunks_run: u64 = 0;
        // S-E.5 lookahead — for the UNBOUNDED scan only: with a LIMIT the early-exit decision needs
        // THIS chunk's contribution before scanning further, so limited queries compute eagerly
        // (semantics identical to pre-pipeline, including "the tail is never staged").
        let mut staged: Option<StagedChunk> = None;
        // S-E.6: cold-tier replay on a hit; a miss scans + captures (discarded on a LIMIT early-exit
        // — only a COMPLETE scan installs).
        let cold = self.load_streaming_cold(&select.table, table, chunk_target_bytes, copin_s);
        let mut capture: Option<ColdCacheBuilder> = None;

        let prefix = relational_key_prefix(&select.table);
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        if let Some(cold) = &cold {
            for chunk in &cold.chunks {
                if window_bound.is_some_and(|bound| rows_out.len() >= bound) {
                    break;
                }
                let Ok(next) = self.stage_cold_chunk(chunk) else {
                    // A failed replay (e.g. a bad spill read) evicts the entry so the next read
                    // rebuilds instead of defer-thrashing (audit LOW).
                    self.evict_streaming_cold(&select.table);
                    return self.execute_relational_select_cpu_pinned(select);
                };
                let to_compute = if window_bound.is_some() {
                    Some(next)
                } else {
                    staged.replace(next)
                };
                if let Some(prev) = to_compute {
                    let Ok(src) = prev.ready() else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    match self.project_streaming_chunk(
                        &chunk_select,
                        table,
                        &chunk_bound,
                        predicate,
                        copin_s,
                        &src,
                        &mut rows_out,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                            return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    chunks_run += 1;
                }
            }
        } else {
            let table_rows = self.read_state.mvcc.table_rows(&select.table);
            capture = Some(ColdCacheBuilder {
                generation: table_rows.generation_payload(),
                build_copin_s: copin_s,
                column_signature: table
                    .columns
                    .iter()
                    .map(|c| (c.name.clone(), c.ty))
                    .collect(),
                chunk_target_bytes,
                total_payload_bytes: 0,
                chunks: Vec::new(),
                spill: None,
                poisoned: false,
            });
            let mut cursor = table_rows.store().seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                // LIMIT satisfied -> STOP the scan: no further row is staged, decoded, or uploaded.
                // The capture is INCOMPLETE at an early exit — discard it (never install a partial set).
                if window_bound.is_some_and(|bound| rows_out.len() >= bound) {
                    capture = None;
                    break;
                }
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let decoded = decode_relational_row(&tuple.value, &table.columns)?;
                chunk_bytes =
                    chunk_bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
                chunk_range = if chunk_rows.is_empty() {
                    (tuple.tuple_id, tuple.tuple_id)
                } else {
                    (chunk_range.0, tuple.tuple_id)
                };
                chunk_rows.push(decoded);
                if chunk_bytes >= chunk_target_bytes {
                    let Ok(next) = self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
                    else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    let to_compute = if window_bound.is_some() {
                        Some(next) // limited: compute eagerly (early-exit fidelity)
                    } else {
                        staged.replace(next) // unbounded: pipeline one chunk ahead
                    };
                    if let Some(prev) = to_compute {
                        let Ok(src) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.project_streaming_chunk(
                            &chunk_select,
                            table,
                            &chunk_bound,
                            predicate,
                            copin_s,
                            &src,
                            &mut rows_out,
                        ) {
                            ChunkOutcome::Ok => {}
                            ChunkOutcome::Defer => {
                                return self.execute_relational_select_cpu_pinned(select)
                            }
                            ChunkOutcome::Hard(err) => return Err(err),
                        }
                        chunks_run += 1;
                    }
                    chunk_rows.clear();
                    chunk_bytes = 0;
                    chunk_range = (1, 0);
                }
            }
        }
        // The final (partial) chunk — skipped when the LIMIT already filled (rows staged before the
        // early-exit tripped would be dropped by the window anyway; don't upload them). Then DRAIN the
        // unbounded pipeline (the last staged chunk still needs its compute).
        if !chunk_rows.is_empty() && window_bound.is_none_or(|bound| rows_out.len() < bound) {
            let Ok(next) = self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture) else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            let to_compute = if window_bound.is_some() {
                Some(next)
            } else {
                staged.replace(next)
            };
            if let Some(prev) = to_compute {
                let Ok(src) = prev.ready() else {
                    return self.execute_relational_select_cpu_pinned(select);
                };
                match self.project_streaming_chunk(
                    &chunk_select,
                    table,
                    &chunk_bound,
                    predicate,
                    copin_s,
                    &src,
                    &mut rows_out,
                ) {
                    ChunkOutcome::Ok => {}
                    ChunkOutcome::Defer => {
                        return self.execute_relational_select_cpu_pinned(select)
                    }
                    ChunkOutcome::Hard(err) => return Err(err),
                }
                chunks_run += 1;
            }
        }
        if let Some(prev) = staged.take() {
            let Ok(src) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.project_streaming_chunk(
                &chunk_select,
                table,
                &chunk_bound,
                predicate,
                copin_s,
                &src,
                &mut rows_out,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            chunks_run += 1;
        }

        // 6c-0: the cross-chunk OFFSET/LIMIT window — ONE device pass over the collected survivors
        // (the executor's own window path); without a window the concat IS the result. A window-pass
        // decline (e.g. the collected set over budget) defers to the CPU path — never a wrong window.
        if (select.limit.is_some() || select.offset.is_some())
            && self
                .window_streaming_rows(select, bound, copin_s, &mut rows_out, budget)
                .is_err()
        {
            return self.execute_relational_select_cpu_pinned(select);
        }
        // S-E.6: a COMPLETE scan installs its captured chunks (None after a LIMIT early-exit).
        if let Some(builder) = capture {
            self.install_streaming_cold(&select.table, builder);
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
    /// device-side LIMIT of the window bound — a chunk's first `offset+limit` survivors are its only
    /// possible contribution to the global window), and CONCAT the survivors. 6c-0 (charter-drift
    /// ruling): the former host drain/truncate windowing is DELETED — the cross-chunk window runs as
    /// ONE final device pass (`window_streaming_rows`); the fold's early-exit is pure CARDINALITY
    /// flow-control on the collected count.
    #[allow(clippy::too_many_arguments)]
    fn project_streaming_chunk(
        &self,
        chunk_select: &Select,
        table: &RelationalTable,
        chunk_bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        src: &ResidentExecSource,
        rows_out: &mut Vec<Vec<SqlValue>>,
    ) -> ChunkOutcome {
        let result = match self.execute_resident_expr_select_with_binding(
            chunk_select,
            table,
            Some(src),
            chunk_bound.clone(),
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
        rows_out.append(&mut result.rows.into_boxed());
        ChunkOutcome::Ok
    }

    /// 6c-0: the cross-chunk OFFSET/LIMIT window as ONE device pass — the collected survivors upload
    /// as a synthesized relation and the executor applies `[OFFSET, OFFSET+LIMIT)` on its own window
    /// path (`sort_streaming_runs` with an EMPTY ORDER BY — the S-E.4 machinery minus the sort).
    fn window_streaming_rows(
        &self,
        select: &Select,
        bound: &BoundRelationalSelect,
        copin_s: Index,
        rows_out: &mut Vec<Vec<SqlValue>>,
        budget: u64,
    ) -> Result<(), ()> {
        let window_column_refs: Vec<(&str, SqlType)> = bound
            .selected_columns
            .iter()
            .map(|column| (column.name.as_str(), column.ty))
            .collect();
        let partial_types: Vec<SqlType> = window_column_refs.iter().map(|(_, ty)| *ty).collect();
        let window_table =
            catalog_relation_table(&select.table, "__stream_window", &window_column_refs);
        let window_select = Select {
            table: window_table.name.clone(),
            distinct: false,
            projection: SelectProjection::Columns(
                window_column_refs
                    .iter()
                    .map(|(name, _)| (*name).to_string())
                    .collect(),
            ),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: select.limit,
            offset: select.offset,
        };
        let window_bound = bind_relational_select(&window_table, &window_select).map_err(|_| ())?;
        let mut rows_bytes: u64 = rows_out
            .iter()
            .map(|row| chunk_row_device_bytes(row, &partial_types))
            .sum();
        match self.sort_streaming_runs(
            &window_select,
            &window_table,
            &window_bound,
            copin_s,
            &partial_types,
            rows_out,
            &mut rows_bytes,
            budget,
        ) {
            ChunkOutcome::Ok => Ok(()),
            _ => Err(()),
        }
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
                // 6c-0(c) RE-LANDED: Count/Sum-int32 partials are DECLARED Numeric{38,0} so every
                // merge round's device SUM output matches the column type directly — the per-round
                // host narrow loop is DELETED; the single PG-type narrow happens once at result
                // materialization (the readback carve-out). (The first landing exposed the
                // masked-pass2 phantom-group kernel bug, now fixed + regression-gated.)
                GroupedAggKind::Count => Some(SqlType::Numeric {
                    precision: 38,
                    scale: 0,
                }),
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
                            Some(SqlType::Numeric {
                                precision: 38,
                                scale: 0,
                            })
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
        // 6c-0(c): which agg columns are PG-bigint results carried as Numeric{38,0} partials — their
        // chunk cells (the executor emits Int8 for COUNT / SUM(int2/4)) WRAP to Numeric at the staging
        // encode, and the merged cells NARROW back to Int8 exactly once, at result materialization
        // (the readback carve-out; a narrow overflow defers to the CPU path). TYPE-CONSISTENT by
        // construction: keyed off the DECLARED partial column type.
        let bigint_as_numeric: Vec<bool> = aggregates
            .iter()
            .zip(partial_types.iter().skip(1))
            .map(|(aggregate, partial_ty)| {
                matches!(
                    aggregate.kind,
                    GroupedAggKind::Count | GroupedAggKind::Sum
                ) && matches!(partial_ty, SqlType::Numeric { .. })
                    && aggregate.value_column.as_ref().is_none_or(|name| {
                        table
                            .columns
                            .iter()
                            .find(|c| &c.name == name)
                            .is_some_and(|c| matches!(c.ty, SqlType::Int2 | SqlType::Int4))
                    })
            })
            .collect();

        let mut partials_acc: Vec<Vec<SqlValue>> = Vec::new();
        let mut partials_bytes: u64 = 0;
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        // 6c-1: the current chunk's inclusive TupleId range ((1,0) = empty sentinel).
        let mut chunk_range: (u64, u64) = (1, 0);
        let mut chunks_run: u64 = 0;
        // S-E.5: the one-chunk lookahead (drained BEFORE any accumulator merge so a chunk upload never
        // rides alongside the merge upload — the residency invariant stays two-chunks-max).
        let mut staged: Option<StagedChunk> = None;
        // S-E.6: cold-tier replay on a hit; a miss scans + captures for later reads.
        let cold = self.load_streaming_cold(&select.table, table, chunk_target_bytes, copin_s);
        let mut capture: Option<ColdCacheBuilder> = None;

        let prefix = relational_key_prefix(&select.table);
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        if let Some(cold) = &cold {
            for chunk in &cold.chunks {
                let Ok(next) = self.stage_cold_chunk(chunk) else {
                    // A failed replay (e.g. a bad spill read) evicts the entry so the next read
                    // rebuilds instead of defer-thrashing (audit LOW).
                    self.evict_streaming_cold(&select.table);
                    return self.execute_relational_select_cpu_pinned(select);
                };
                if let Some(prev) = staged.replace(next) {
                    let Ok(src) = prev.ready() else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    match self.grouped_streaming_chunk(
                        &grouped_select,
                        table,
                        &grouped_bound,
                        predicate,
                        copin_s,
                        group_column,
                        &src,
                        &partial_types,
                        &bigint_as_numeric,
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
                }
                // The loop's compaction, with the same drain-first discipline.
                if partials_bytes >= chunk_target_bytes {
                    if let Some(prev) = staged.take() {
                        let Ok(src) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.grouped_streaming_chunk(
                            &grouped_select,
                            table,
                            &grouped_bound,
                            predicate,
                            copin_s,
                            group_column,
                            &src,
                            &partial_types,
                            &bigint_as_numeric,
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
                    }
                }
                if partials_bytes >= chunk_target_bytes {
                    match self.merge_streaming_partials(
                        &merge_select,
                        &partials_table,
                        &merge_bound,
                        copin_s,
                        group_column,
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
        } else {
            let table_rows = self.read_state.mvcc.table_rows(&select.table);
            capture = Some(ColdCacheBuilder {
                generation: table_rows.generation_payload(),
                build_copin_s: copin_s,
                column_signature: table
                    .columns
                    .iter()
                    .map(|c| (c.name.clone(), c.ty))
                    .collect(),
                chunk_target_bytes,
                total_payload_bytes: 0,
                chunks: Vec::new(),
                spill: None,
                poisoned: false,
            });
            let mut cursor = table_rows.store().seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let decoded = decode_relational_row(&tuple.value, &table.columns)?;
                chunk_bytes =
                    chunk_bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
                chunk_range = if chunk_rows.is_empty() {
                    (tuple.tuple_id, tuple.tuple_id)
                } else {
                    (chunk_range.0, tuple.tuple_id)
                };
                chunk_rows.push(decoded);
                if chunk_bytes >= chunk_target_bytes {
                    let Ok(next) = self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
                    else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    if let Some(prev) = staged.replace(next) {
                        let Ok(src) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.grouped_streaming_chunk(
                            &grouped_select,
                            table,
                            &grouped_bound,
                            predicate,
                            copin_s,
                            group_column,
                            &src,
                            &partial_types,
                            &bigint_as_numeric,
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
                    }
                    chunk_rows.clear();
                    chunk_bytes = 0;
                    chunk_range = (1, 0);
                    // COMPACTION: the accumulator outgrew the chunk budget — device-merge it down to
                    // one row per true group. If even the compacted form exceeds the budget, the group
                    // cardinality itself is over-budget: defer (S-E.4+ may spill; v1 is honest).
                    // Drain the in-flight chunk FIRST so the merge upload never overlaps a chunk upload.
                    if partials_bytes >= chunk_target_bytes {
                        if let Some(prev) = staged.take() {
                            let Ok(src) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.grouped_streaming_chunk(
                            &grouped_select,
                            table,
                            &grouped_bound,
                            predicate,
                            copin_s,
                            group_column,
                            &src,
                            &partial_types,
                            &bigint_as_numeric,
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
                        }
                    }
                    if partials_bytes >= chunk_target_bytes {
                        match self.merge_streaming_partials(
                            &merge_select,
                            &partials_table,
                            &merge_bound,
                            copin_s,
                            group_column,
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
            let Ok(next) = self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture) else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            if let Some(prev) = staged.replace(next) {
                let Ok(src) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.grouped_streaming_chunk(
                &grouped_select,
                table,
                &grouped_bound,
                predicate,
                copin_s,
                group_column,
                &src,
                &partial_types,
                &bigint_as_numeric,
                &mut partials_acc,
                &mut partials_bytes,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            chunks_run += 1;
            }
        }
        // Pipeline-lag compaction: the tail-block compute lands one whole chunk's partials AFTER the
        // loop's last compaction check, so re-check here — else the accumulator can reach the drain
        // compute already over-target and overshoot the merge budget gate (defer where the pre-pipeline
        // fold compacted and succeeded).
        if partials_bytes >= chunk_target_bytes {
            match self.merge_streaming_partials(
                &merge_select,
                &partials_table,
                &merge_bound,
                copin_s,
                group_column,
                &mut partials_acc,
                &mut partials_bytes,
                &partial_types,
                budget,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            if partials_bytes >= chunk_target_bytes {
                return self.execute_relational_select_cpu_pinned(select);
            }
        }
        // DRAIN the pipeline before the final merge (no chunk upload alongside the merge upload).
        if let Some(prev) = staged.take() {
            let Ok(src) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.grouped_streaming_chunk(
                &grouped_select,
                table,
                &grouped_bound,
                predicate,
                copin_s,
                group_column,
                &src,
                &partial_types,
                &bigint_as_numeric,
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
        // S-E.6: a COMPLETE scan installs its captured chunks for byte-replay by later reads.
        if let Some(builder) = capture {
            self.install_streaming_cold(&select.table, builder);
        }
        let mut rows_out = std::mem::take(&mut partials_acc);
        // 6c-0(c) readback boundary: the PG-bigint aggregates (COUNT / SUM(int2/4)) rode as
        // Numeric{38,0} partials; narrow each merged cell to Int8 exactly ONCE, at result
        // materialization (a narrow overflow defers to the authoritative CPU path — PG's own
        // "bigint out of range" surface).
        if !distinct_key_only {
            for row in &mut rows_out {
                for (agg_idx, narrow) in bigint_as_numeric.iter().enumerate() {
                    if !narrow {
                        continue;
                    }
                    let cell = &mut row[agg_idx + 1];
                    match cell {
                        SqlValue::Numeric(d) if d.scale == 0 => match i64::try_from(d.mantissa) {
                            Ok(narrowed) => *cell = SqlValue::Int8(narrowed),
                            Err(_) => return self.execute_relational_select_cpu_pinned(select),
                        },
                        SqlValue::Null | SqlValue::Int8(_) => {}
                        _ => return self.execute_relational_select_cpu_pinned(select),
                    }
                }
            }
        }
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
        src: &ResidentExecSource,
        partial_types: &[SqlType],
        bigint_as_numeric: &[bool],
        partials_acc: &mut Vec<Vec<SqlValue>>,
        partials_bytes: &mut u64,
    ) -> ChunkOutcome {
        let group_key_columns = [group_column.to_string()];
        let result = match self.execute_resident_expr_select_with_binding(
            grouped_select,
            table,
            Some(src),
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
        for mut row in result.rows.into_boxed() {
            // 6c-0(c) staging encode: the executor emits Int8 for COUNT/SUM(int2/4); the partials
            // relation declares those columns Numeric{38,0} — wrap losslessly (i64 -> i128 mantissa)
            // so every merge round is type-stable with NO per-round narrow.
            for (agg_idx, wrap) in bigint_as_numeric.iter().enumerate() {
                if !wrap {
                    continue;
                }
                if let SqlValue::Int8(n) = row[agg_idx + 1] {
                    row[agg_idx + 1] = SqlValue::Numeric(Decimal128::new(i128::from(n), 0));
                }
            }
            *partials_bytes =
                partials_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
            partials_acc.push(row);
        }
        ChunkOutcome::Ok
    }

    /// STRATA S-E.4 — the ORDER BY fold. THE SORT IS ALWAYS ON THE DEVICE (the charter forbids a host
    /// k-way merge): TOP-N (`ORDER BY k LIMIT n [OFFSET m]`) runs each chunk's projection through the
    /// device sort + device window — a chunk's local top-(m+n) is its ONLY possible contribution to the
    /// global window — concats the runs (control plane), COMPACTS the accumulator by device re-sort +
    /// re-window whenever it outgrows the chunk target, and finishes with ONE device sort + the REAL
    /// window over the synthesized runs relation. UNBOUNDED ORDER BY skips the per-chunk sort (plain
    /// device filter/project per chunk — a final re-sort makes chunk runs pointless) and defers honestly
    /// mid-scan if the survivor set outgrows the budget (its final device sort could not fit).
    #[allow(clippy::too_many_arguments)]
    fn run_streaming_ordered_fold(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        top_n: Option<usize>,
        gpu_id: u16,
        budget: u64,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let chunk_target_bytes = (budget / 2).max(1);
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        // The per-chunk select: top-N keeps the ORDER BY + a device LIMIT of the window bound (the run
        // truncation); unbounded strips the ORDER BY (plain filter/project — sorted once at the end).
        let mut chunk_select = select.clone();
        chunk_select.offset = None;
        match top_n {
            Some(bound_n) => chunk_select.limit = Some(bound_n),
            None => {
                chunk_select.order_by = Vec::new();
                chunk_select.limit = None;
            }
        }
        let Ok(chunk_bound) = bind_relational_select(table, &chunk_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        // The synthesized RUNS relation: the projected columns by name/type (the classifier guarantees
        // the sort key is among them), consumed by the FINAL device sort + window pass.
        let runs_column_refs: Vec<(&str, SqlType)> = bound
            .selected_columns
            .iter()
            .map(|column| (column.name.as_str(), column.ty))
            .collect();
        let partial_types: Vec<SqlType> = runs_column_refs.iter().map(|(_, ty)| *ty).collect();
        let runs_table = catalog_relation_table(&table.schema, "__stream_runs", &runs_column_refs);
        let final_select = Select {
            table: runs_table.name.clone(),
            distinct: false,
            projection: SelectProjection::Columns(
                runs_column_refs
                    .iter()
                    .map(|(name, _)| (*name).to_string())
                    .collect(),
            ),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: select.order_by.clone(),
            limit: select.limit,
            offset: select.offset,
        };
        let Ok(final_bound) = bind_relational_select(&runs_table, &final_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };
        // The COMPACTION select: same device sort but windowed to the top-N bound only (OFFSET stays 0 —
        // the real window slices once, at the end).
        let compact_select = Select {
            limit: top_n,
            offset: None,
            ..final_select.clone()
        };
        let Ok(compact_bound) = bind_relational_select(&runs_table, &compact_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        let mut runs_acc: Vec<Vec<SqlValue>> = Vec::new();
        let mut runs_bytes: u64 = 0;
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        // 6c-1: the current chunk's inclusive TupleId range ((1,0) = empty sentinel).
        let mut chunk_range: (u64, u64) = (1, 0);
        let mut chunks_run: u64 = 0;
        // S-E.5: the one-chunk lookahead (drained BEFORE any accumulator sort upload).
        let mut staged: Option<StagedChunk> = None;
        // S-E.6: cold-tier replay on a hit; a miss scans + captures for later reads.
        let cold = self.load_streaming_cold(&select.table, table, chunk_target_bytes, copin_s);
        let mut capture: Option<ColdCacheBuilder> = None;

        let prefix = relational_key_prefix(&select.table);
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        if let Some(cold) = &cold {
            for chunk in &cold.chunks {
                let Ok(next) = self.stage_cold_chunk(chunk) else {
                    // A failed replay (e.g. a bad spill read) evicts the entry so the next read
                    // rebuilds instead of defer-thrashing (audit LOW).
                    self.evict_streaming_cold(&select.table);
                    return self.execute_relational_select_cpu_pinned(select);
                };
                if let Some(prev) = staged.replace(next) {
                    let Ok(src) = prev.ready() else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    match self.ordered_streaming_chunk(
                        &chunk_select,
                        table,
                        &chunk_bound,
                        predicate,
                        copin_s,
                        &src,
                        &partial_types,
                        &mut runs_acc,
                        &mut runs_bytes,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                            return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    chunks_run += 1;
                }
                // The loop's compaction/defer checks, drain-first (as the scan loop).
                if runs_bytes >= chunk_target_bytes {
                    if let Some(prev) = staged.take() {
                        let Ok(src) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.ordered_streaming_chunk(
                            &chunk_select,
                            table,
                            &chunk_bound,
                            predicate,
                            copin_s,
                            &src,
                            &partial_types,
                            &mut runs_acc,
                            &mut runs_bytes,
                        ) {
                            ChunkOutcome::Ok => {}
                            ChunkOutcome::Defer => {
                                return self.execute_relational_select_cpu_pinned(select)
                            }
                            ChunkOutcome::Hard(err) => return Err(err),
                        }
                        chunks_run += 1;
                    }
                }
                if runs_bytes >= chunk_target_bytes {
                    match top_n {
                        Some(_) => {
                            match self.sort_streaming_runs(
                                &compact_select,
                                &runs_table,
                                &compact_bound,
                                copin_s,
                                &partial_types,
                                &mut runs_acc,
                                &mut runs_bytes,
                                budget,
                            ) {
                                ChunkOutcome::Ok => {}
                                ChunkOutcome::Defer => {
                                    return self.execute_relational_select_cpu_pinned(select)
                                }
                                ChunkOutcome::Hard(err) => return Err(err),
                            }
                            if runs_bytes > budget {
                                return self.execute_relational_select_cpu_pinned(select);
                            }
                        }
                        None => {
                            if runs_bytes > budget {
                                return self.execute_relational_select_cpu_pinned(select);
                            }
                        }
                    }
                }
            }
        } else {
            let table_rows = self.read_state.mvcc.table_rows(&select.table);
            capture = Some(ColdCacheBuilder {
                generation: table_rows.generation_payload(),
                build_copin_s: copin_s,
                column_signature: table
                    .columns
                    .iter()
                    .map(|c| (c.name.clone(), c.ty))
                    .collect(),
                chunk_target_bytes,
                total_payload_bytes: 0,
                chunks: Vec::new(),
                spill: None,
                poisoned: false,
            });
            let mut cursor = table_rows.store().seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let decoded = decode_relational_row(&tuple.value, &table.columns)?;
                chunk_bytes =
                    chunk_bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
                chunk_range = if chunk_rows.is_empty() {
                    (tuple.tuple_id, tuple.tuple_id)
                } else {
                    (chunk_range.0, tuple.tuple_id)
                };
                chunk_rows.push(decoded);
                if chunk_bytes >= chunk_target_bytes {
                    let Ok(next) = self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
                    else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    if let Some(prev) = staged.replace(next) {
                        let Ok(src) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.ordered_streaming_chunk(
                            &chunk_select,
                            table,
                            &chunk_bound,
                            predicate,
                            copin_s,
                            &src,
                            &partial_types,
                            &mut runs_acc,
                            &mut runs_bytes,
                        ) {
                            ChunkOutcome::Ok => {}
                            ChunkOutcome::Defer => {
                                return self.execute_relational_select_cpu_pinned(select)
                            }
                            ChunkOutcome::Hard(err) => return Err(err),
                        }
                        chunks_run += 1;
                    }
                    chunk_rows.clear();
                    chunk_bytes = 0;
                    chunk_range = (1, 0);
                    // Drain the in-flight chunk FIRST so an accumulator sort upload never overlaps it.
                    if runs_bytes >= chunk_target_bytes {
                        if let Some(prev) = staged.take() {
                            let Ok(src) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.ordered_streaming_chunk(
                            &chunk_select,
                            table,
                            &chunk_bound,
                            predicate,
                            copin_s,
                            &src,
                            &partial_types,
                            &mut runs_acc,
                            &mut runs_bytes,
                        ) {
                            ChunkOutcome::Ok => {}
                            ChunkOutcome::Defer => {
                                return self.execute_relational_select_cpu_pinned(select)
                            }
                            ChunkOutcome::Hard(err) => return Err(err),
                        }
                        chunks_run += 1;
                        }
                    }
                    if runs_bytes >= chunk_target_bytes {
                        match top_n {
                            // TOP-N compaction: device re-sort + truncate to the window bound.
                            Some(_) => {
                                match self.sort_streaming_runs(
                                    &compact_select,
                                    &runs_table,
                                    &compact_bound,
                                    copin_s,
                                    &partial_types,
                                    &mut runs_acc,
                                    &mut runs_bytes,
                                    budget,
                                ) {
                                    ChunkOutcome::Ok => {}
                                    ChunkOutcome::Defer => {
                                        return self
                                            .execute_relational_select_cpu_pinned(select)
                                    }
                                    ChunkOutcome::Hard(err) => return Err(err),
                                }
                                // A window bound too large to compact under the budget cannot final-
                                // sort either: defer (mirrors the grouped over-cardinality defer).
                                if runs_bytes > budget {
                                    return self.execute_relational_select_cpu_pinned(select);
                                }
                            }
                            // UNBOUNDED: the survivor set itself outgrew the budget — its final
                            // device sort cannot fit. Defer honestly (the CPU path serves it).
                            None => {
                                if runs_bytes > budget {
                                    return self.execute_relational_select_cpu_pinned(select);
                                }
                            }
                        }
                    }
                }
            }
        }
        if !chunk_rows.is_empty() {
            let Ok(next) = self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture) else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            if let Some(prev) = staged.replace(next) {
                let Ok(src) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.ordered_streaming_chunk(
                &chunk_select,
                table,
                &chunk_bound,
                predicate,
                copin_s,
                &src,
                &partial_types,
                &mut runs_acc,
                &mut runs_bytes,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            chunks_run += 1;
            }
        }
        // Pipeline-lag re-check (mirrors the grouped fold): the tail-block compute lands one chunk's
        // runs after the loop's last check — re-compact (top-N) / re-gate (unbounded) before draining.
        if runs_bytes >= chunk_target_bytes {
            match top_n {
                Some(_) => {
                    match self.sort_streaming_runs(
                        &compact_select,
                        &runs_table,
                        &compact_bound,
                        copin_s,
                        &partial_types,
                        &mut runs_acc,
                        &mut runs_bytes,
                        budget,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                            return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    if runs_bytes > budget {
                        return self.execute_relational_select_cpu_pinned(select);
                    }
                }
                None => {
                    if runs_bytes > budget {
                        return self.execute_relational_select_cpu_pinned(select);
                    }
                }
            }
        }
        // DRAIN the pipeline before the final sort (no chunk upload alongside the sort upload).
        if let Some(prev) = staged.take() {
            let Ok(src) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.ordered_streaming_chunk(
                &chunk_select,
                table,
                &chunk_bound,
                predicate,
                copin_s,
                &src,
                &partial_types,
                &mut runs_acc,
                &mut runs_bytes,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            chunks_run += 1;
        }

        // FINAL: one device sort + the REAL [OFFSET, OFFSET+LIMIT) window over the accumulated runs.
        // An empty accumulator (empty table / all filtered) is the empty ordered result.
        if !runs_acc.is_empty() {
            match self.sort_streaming_runs(
                &final_select,
                &runs_table,
                &final_bound,
                copin_s,
                &partial_types,
                &mut runs_acc,
                &mut runs_bytes,
                budget,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
        }
        // S-E.6: a COMPLETE scan installs its captured chunks for byte-replay by later reads.
        if let Some(builder) = capture {
            self.install_streaming_cold(&select.table, builder);
        }
        let rows_out = std::mem::take(&mut runs_acc);

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

    /// One ordered-fold chunk: upload, run the per-chunk select on the device (top-N = sort + window to
    /// the bound; unbounded = plain filter/project), and CONCAT the resulting run into the accumulator.
    #[allow(clippy::too_many_arguments)]
    fn ordered_streaming_chunk(
        &self,
        chunk_select: &Select,
        table: &RelationalTable,
        chunk_bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        src: &ResidentExecSource,
        partial_types: &[SqlType],
        runs_acc: &mut Vec<Vec<SqlValue>>,
        runs_bytes: &mut u64,
    ) -> ChunkOutcome {
        let order_by_exprs: Vec<Option<ResidentExpr>> = vec![None; chunk_select.order_by.len()];
        let order_by_nulls_first: Vec<Option<bool>> = vec![None; chunk_select.order_by.len()];
        let result = match self.execute_resident_expr_select_with_binding(
            chunk_select,
            table,
            Some(src),
            chunk_bound.clone(),
            copin_s,
            predicate,
            None,
            &order_by_exprs,
            &order_by_nulls_first,
            None,
            &[],
        ) {
            Ok(result) => result,
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        for row in result.rows.into_boxed() {
            *runs_bytes = runs_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
            runs_acc.push(row);
        }
        ChunkOutcome::Ok
    }

    /// Device-sort (+ window) the accumulated runs as a transient synthesized relation, replacing the
    /// accumulator with the sorted/windowed rows. Serves BOTH the top-N compaction and the final pass.
    /// BUDGET GATE (the S-E.3 lesson): defer WITHOUT uploading when the accumulator exceeds the budget.
    #[allow(clippy::too_many_arguments)]
    fn sort_streaming_runs(
        &self,
        sort_select: &Select,
        runs_table: &RelationalTable,
        sort_bound: &BoundRelationalSelect,
        copin_s: Index,
        partial_types: &[SqlType],
        runs_acc: &mut Vec<Vec<SqlValue>>,
        runs_bytes: &mut u64,
        budget: u64,
    ) -> ChunkOutcome {
        if *runs_bytes > budget {
            return ChunkOutcome::Defer;
        }
        let (descriptor, device_memory) =
            match self.build_transient_relation_residency(runs_table, runs_acc) {
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
            row_count: runs_acc.len() as u64,
        };
        let order_by_exprs: Vec<Option<ResidentExpr>> = vec![None; sort_select.order_by.len()];
        let order_by_nulls_first: Vec<Option<bool>> = vec![None; sort_select.order_by.len()];
        let result = match self.execute_resident_expr_select_with_binding(
            sort_select,
            runs_table,
            Some(&src),
            sort_bound.clone(),
            copin_s,
            None,
            None,
            &order_by_exprs,
            &order_by_nulls_first,
            None,
            &[],
        ) {
            Ok(result) => result,
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        let mut sorted: Vec<Vec<SqlValue>> = Vec::new();
        let mut sorted_bytes: u64 = 0;
        for row in result.rows.into_boxed() {
            sorted_bytes = sorted_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
            sorted.push(row);
        }
        *runs_acc = sorted;
        *runs_bytes = sorted_bytes;
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
        // 6c-0(c): NO per-round re-typing — the merge output's cell types ARE the declared partial
        // column types (Numeric{38,0} for the PG-bigint aggregates); rounds compose with no host
        // work. The single PG-type narrow happens at result materialization.
        for row in result.rows.into_boxed() {
            merged_bytes =
                merged_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
            merged.push(row);
        }
        *partials_acc = merged;
        *partials_bytes = merged_bytes;
        ChunkOutcome::Ok
    }

    /// STRATA S-E.5: stage one chunk — build its transient payload (host) and enqueue the upload on a
    /// private copy stream (async when the driver supports it). The caller computes the PREVIOUSLY
    /// staged chunk next, so this upload overlaps that compute and the subsequent host staging.
    fn stage_streaming_chunk(
        &self,
        table: &RelationalTable,
        chunk_rows: &[Vec<SqlValue>],
        chunk_range: (u64, u64),
        capture: &mut Option<ColdCacheBuilder>,
    ) -> Result<StagedChunk, ()> {
        let (snapshot, pending, payload) = self
            .build_transient_relation_residency_async(table, chunk_rows)
            .map_err(|_| ())?;
        // The out-of-core proof: the ACTUAL transient device bytes for this chunk (fetch_max monotonic).
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(snapshot.resident_bytes, Ordering::Relaxed);
        // S-E.6: capture the built payload bytes for the cold tier (the upload already staged them
        // into pinned memory; keeping the Vec is zero extra copies). Above the spill threshold the
        // builder streams them to the unlinked spill file instead of holding RAM (S-E.6b).
        if let Some(builder) = capture {
            builder.push(payload, snapshot.clone(), chunk_rows.len() as u64, chunk_range);
        }
        Ok(StagedChunk {
            snapshot,
            pending,
            row_count: chunk_rows.len() as u64,
        })
    }

    /// S-E.6b (audit LOW): evict a table's cold entry after a replay failure — a bad spill file
    /// (disk fault) would otherwise defer-thrash every future streaming read on the table; dropping
    /// the entry lets the next read rebuild it.
    fn evict_streaming_cold(&self, table_name: &str) {
        let residency = &self.read_state.residency;
        let _publish = residency
            .streaming_cold_lock
            .lock()
            .expect("streaming cold-tier lock poisoned");
        let mut map = std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
        if map.remove(table_name).is_some() {
            residency.streaming_cold_chunks.store(Arc::new(map));
        }
    }

    /// S-E.6: stage one COLD chunk — re-upload the cached device payload bytes (async copy stream),
    /// with a fresh proof stamped onto the cached descriptor template. No decode, no assembly.
    fn stage_cold_chunk(&self, chunk: &ColdChunk) -> Result<StagedChunk, ()> {
        let runtime = self.cuda_driver_probe_runtime();
        // RAM chunks borrow; spilled chunks positional-read from the unlinked file (an IO error is
        // a defer, never a wrong answer).
        let payload = chunk.payload.read()?;
        let pending = runtime
            .retain_device_memory_copy_async(chunk.snapshot.gpu_id, &payload)
            .map_err(|_| ())?;
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(chunk.snapshot.resident_bytes, Ordering::Relaxed);
        let mut snapshot = chunk.snapshot.clone();
        snapshot.device_memory_proof = Some(pending.metadata().clone());
        Ok(StagedChunk {
            snapshot,
            pending,
            row_count: chunk.row_count,
        })
    }

    /// 6c-1: rebuild the visible rows of ONE effective TupleId range into cold chunks (payload +
    /// descriptor, NO upload — replays stamp a fresh proof). Splits at the chunk byte target. The
    /// decode/build here is the SAME staging the scan path performs, bounded to the dirty range —
    /// the O(delta) win (charter: the staging upload carve-out; the registered scan-build debt
    /// shrinks from O(table)/write to O(delta)/write).
    #[allow(clippy::too_many_arguments)]
    fn build_cold_chunks_for_range(
        &self,
        table: &RelationalTable,
        store: &crate::resident_storage::TableVersionData,
        copin_s: Index,
        eff_lo: u64,
        eff_hi: u64,
        chunk_target_bytes: u64,
        // F1 (6c-1 audit): rebuilt chunks accumulate through this SPILL-AWARE builder — a big
        // tail / wide dirty range streams to the unlinked spill file above the threshold instead
        // of materializing all payloads in host RAM (the same out-of-core bound the scan-build
        // has). The builder's chunks stay in ascending range order across calls.
        builder: &mut ColdCacheBuilder,
    ) -> Result<(), ()> {
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        let versions = store
            .rows
            .visible_versions_in_range(visibility, eff_lo, eff_hi)
            .map_err(|_| ())?;
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let mut rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut bytes: u64 = 0;
        let mut range: Option<(u64, u64)> = None;
        let prefix = relational_key_prefix(&table.name);
        let flush = |rows: &mut Vec<Vec<SqlValue>>,
                         range: &mut Option<(u64, u64)>,
                         builder: &mut ColdCacheBuilder|
         -> Result<(), ()> {
            if rows.is_empty() {
                return Ok(());
            }
            let (snapshot, payload) = self.build_cold_payload(table, rows)?;
            builder.push(
                payload,
                snapshot,
                rows.len() as u64,
                range.take().expect("non-empty chunk has a range"),
            );
            rows.clear();
            Ok(())
        };
        for version in versions {
            if !version.key.starts_with(&prefix) {
                continue;
            }
            let decoded = decode_relational_row(&version.value, &table.columns).map_err(|_| ())?;
            bytes = bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
            range = Some(match range {
                None => (version.tuple_id, version.tuple_id),
                Some((lo, _)) => (lo, version.tuple_id),
            });
            rows.push(decoded);
            if bytes >= chunk_target_bytes {
                flush(&mut rows, &mut range, builder)?;
                bytes = 0;
            }
        }
        flush(&mut rows, &mut range, builder)?;
        if builder.poisoned {
            return Err(());
        }
        Ok(())
    }

    /// The payload + descriptor for a cold chunk WITHOUT uploading (proof = None; stage_cold_chunk
    /// stamps a fresh proof per replay). Mirrors `build_transient_relation_residency`'s descriptor.
    fn build_cold_payload(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, Vec<u8>), ()> {
        let (snapshot, pending, payload) = self
            .build_transient_relation_residency_async(table, rows)
            .map_err(|_| ())?;
        // The transient upload is a byproduct here (the builder uploads); drop it un-used — wait()
        // completes the DMA so the pinned buffer recycles safely. (A payload-only builder split is
        // a follow-up; correctness first.)
        let _ = pending.wait().map_err(|_| ())?;
        Ok((snapshot, payload))
    }

    /// 6c-1 — CHUNK-GRANULAR DELTA PATCHING (deletes the whole-table invalidation): a stale cold
    /// entry (generation mismatch = a write happened) is PATCHED, not discarded. The changed
    /// TupleIds come from the O(delta) COW-chain diff (`changed_tuple_ids` — untouched subtrees are
    /// pointer-equal); each maps to its chunk through the EFFECTIVE range tiling (chunk i owns
    /// (prev.hi, hi]; ids beyond the last chunk are the TAIL — the rollover pattern). Untouched
    /// chunks REUSE their bytes verbatim (chain identity + the old entry's settled boundary make
    /// their visible sets boundary-invariant); dirty ranges + the tail REBUILD at the patching
    /// reader's boundary. The patched entry re-installs under the SAME settled-boundary commit-lock
    /// proof as a fresh build (S-E.6a). Returns the landed entry, or None (caller evicts + scans).
    fn patch_streaming_cold(
        &self,
        table_name: &str,
        table: &RelationalTable,
        stale: &Arc<ColdTableChunks>,
        current: &Arc<crate::resident_storage::TableVersionData>,
        copin_s: Index,
        chunk_target_bytes: u64,
    ) -> Option<Arc<ColdTableChunks>> {
        // The ALTER guard: a shape-changing DDL republished the store too — cached payload layouts
        // would be reused with the WRONG column shape. Signature inequality -> evict.
        let signature: Vec<(String, SqlType)> = table
            .columns
            .iter()
            .map(|c| (c.name.clone(), c.ty))
            .collect();
        if signature != stale.column_signature
            || stale.chunk_target_bytes != chunk_target_bytes
        {
            return None;
        }
        let changed = stale.generation.rows.changed_tuple_ids(&current.rows);
        // Map changed ids to dirty chunks via the effective tiling; ids past the last hi = tail.
        let mut dirty = vec![false; stale.chunks.len()];
        let his: Vec<u64> = stale.chunks.iter().map(|c| c.tuple_range.1).collect();
        let last_hi = his.last().copied().unwrap_or(0);
        let mut tail_dirty = stale.chunks.is_empty();
        for id in &changed {
            if *id > last_hi {
                tail_dirty = true;
                continue;
            }
            let idx = his.partition_point(|hi| *hi < *id);
            dirty[idx] = true;
        }
        // F2 (6c-1 audit — fragmentation cap): when the tail grows, COALESCE a trailing RUNT chunk
        // (under half the target) into the tail rebuild — insert/read ping-pong would otherwise
        // accrete one tiny chunk per write, degrading every later replay. Each patch absorbs the
        // runt, so at most one lives at any time.
        if tail_dirty && !stale.chunks.is_empty() {
            let last = stale.chunks.len() - 1;
            let last_bytes = match &stale.chunks[last].payload {
                ColdPayload::Ram(bytes) => bytes.len() as u64,
                ColdPayload::Spilled { len, .. } => *len as u64,
            };
            if last_bytes < chunk_target_bytes / 2 {
                dirty[last] = true;
            }
        }
        // Rebuild dirty ranges through ONE spill-aware builder (F1: rebuilt payloads stream to the
        // spill file above the threshold — never unbounded host RAM), then MERGE with the reused
        // chunks by ascending range (both sequences are ascending; control-plane assembly).
        let mut rebuild = ColdCacheBuilder {
            generation: Arc::clone(current),
            build_copin_s: copin_s,
            chunk_target_bytes,
            total_payload_bytes: 0,
            column_signature: signature.clone(),
            chunks: Vec::new(),
            spill: None,
            poisoned: false,
        };
        let mut reused: Vec<ColdChunk> = Vec::new();
        let mut eff_lo: u64 = 0;
        for (i, chunk) in stale.chunks.iter().enumerate() {
            let eff_hi = chunk.tuple_range.1;
            // A dirty chunk's range REBUILDS; when the runt-coalesce marked the LAST chunk dirty,
            // extend its rebuild into the tail in one scan (eff_hi = MAX below handles it).
            let rebuild_hi = if dirty[i] && i == stale.chunks.len() - 1 && tail_dirty {
                u64::MAX
            } else {
                eff_hi
            };
            if dirty[i] {
                self.build_cold_chunks_for_range(
                    table,
                    current,
                    copin_s,
                    eff_lo,
                    rebuild_hi,
                    chunk_target_bytes,
                    &mut rebuild,
                )
                .ok()?;
                self.read_state
                    .residency
                    .streaming_cold_chunks_rebuilt
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                reused.push(ColdChunk {
                    payload: match &chunk.payload {
                        ColdPayload::Ram(bytes) => ColdPayload::Ram(Arc::clone(bytes)),
                        ColdPayload::Spilled { file, offset, len } => ColdPayload::Spilled {
                            file: Arc::clone(file),
                            offset: *offset,
                            len: *len,
                        },
                    },
                    snapshot: chunk.snapshot.clone(),
                    row_count: chunk.row_count,
                    tuple_range: chunk.tuple_range,
                });
            }
            eff_lo = eff_hi.saturating_add(1);
        }
        // The tail (unless the runt-coalesce already extended the last rebuild through MAX).
        let tail_absorbed =
            tail_dirty && !stale.chunks.is_empty() && dirty[stale.chunks.len() - 1];
        if tail_dirty && !tail_absorbed {
            self.build_cold_chunks_for_range(
                table,
                current,
                copin_s,
                last_hi.saturating_add(1),
                u64::MAX,
                chunk_target_bytes,
                &mut rebuild,
            )
            .ok()?;
        }
        // Merge reused + rebuilt by ascending range start (both already ascending).
        let mut chunks: Vec<ColdChunk> = Vec::with_capacity(reused.len() + rebuild.chunks.len());
        {
            let mut a = reused.into_iter().peekable();
            let mut b = rebuild.chunks.into_iter().peekable();
            loop {
                match (a.peek(), b.peek()) {
                    (Some(x), Some(y)) => {
                        if x.tuple_range.0 <= y.tuple_range.0 {
                            chunks.push(a.next().expect("peeked"));
                        } else {
                            chunks.push(b.next().expect("peeked"));
                        }
                    }
                    (Some(_), None) => chunks.push(a.next().expect("peeked")),
                    (None, Some(_)) => chunks.push(b.next().expect("peeked")),
                    (None, None) => break,
                }
            }
        }
        let total_payload_bytes: u64 = chunks
            .iter()
            .map(|c| match &c.payload {
                ColdPayload::Ram(bytes) => bytes.len() as u64,
                ColdPayload::Spilled { len, .. } => *len as u64,
            })
            .sum();
        let builder = ColdCacheBuilder {
            generation: Arc::clone(current),
            build_copin_s: copin_s,
            chunk_target_bytes,
            total_payload_bytes,
            column_signature: signature,
            chunks,
            spill: None,
            poisoned: false,
        };
        if !self.install_streaming_cold_inner(table_name, builder, true) {
            return None;
        }
        self.read_state
            .residency
            .streaming_cold_patches
            .fetch_add(1, Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
    }

    /// S-E.6: the table's valid cold-tier chunks, or `None` (miss -> the caller scans + captures).
    /// A hit requires the SAME tuple-store generation (pointer equality — see [`ColdTableChunks`]),
    /// the same chunk target, AND `copin_s >= build_copin_s` (the boundary-invariance condition:
    /// the install proved no stamp exceeds the build boundary, so every boundary at-or-above it
    /// sees the identical set — audit F1). A GENERATION-mismatched entry is EVICTED here (audit
    /// F3: a stale entry would otherwise pin the superseded TableVersionData until the next
    /// install). `streaming_cold_hits` counts validity-passed ATTEMPTS (the fold may still defer
    /// on a later chunk — audit F4).
    fn load_streaming_cold(
        &self,
        table_name: &str,
        table: &RelationalTable,
        chunk_target_bytes: u64,
        copin_s: Index,
    ) -> Option<Arc<ColdTableChunks>> {
        let cold = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()?;
        let current = self.read_state.mvcc.table_rows(table_name).generation_payload();
        if !Arc::ptr_eq(&cold.generation, &current) {
            // 6c-1: the table was written — PATCH the entry (rebuild only the dirty chunks + tail,
            // O(delta)) instead of discarding it. A patch that cannot apply (ALTER'd shape, install
            // race, IO error) falls through to the evict arm; the next read scans + rebuilds.
            if let Some(patched) = self.patch_streaming_cold(
                table_name,
                table,
                &cold,
                &current,
                copin_s,
                chunk_target_bytes,
            ) {
                self.read_state
                    .residency
                    .streaming_cold_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Some(patched);
            }
            // The table was written: drop the stale entry (and its pinned old generation) now.
            let residency = &self.read_state.residency;
            let _publish = residency
                .streaming_cold_lock
                .lock()
                .expect("streaming cold-tier lock poisoned");
            let mut map =
                std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
            // Re-check under the lock (a concurrent rebuild may have installed a FRESH entry).
            if let Some(entry) = map.get(table_name) {
                if !Arc::ptr_eq(&entry.generation, &current) {
                    map.remove(table_name);
                    residency.streaming_cold_chunks.store(Arc::new(map));
                }
            }
            return None;
        }
        if cold.chunk_target_bytes != chunk_target_bytes || copin_s < cold.build_copin_s {
            return None;
        }
        self.read_state
            .residency
            .streaming_cold_hits
            .fetch_add(1, Ordering::Relaxed);
        Some(cold)
    }

    /// S-E.6: install a completed scan's captured chunks — under the COMMIT LOCK, which makes the
    /// settled-boundary proof AIRTIGHT (audit F1): inside the lock no committer can sit mid
    /// publish->seq-bump (both stores live in the commit critical section), so "generation
    /// unchanged AND `committed_seq() == build_copin_s`" proves the generation contains NO stamp
    /// above the build boundary — the captured set is then boundary-invariant for every reader at
    /// or above it. Any commit since the bind (even to another table) discards the install
    /// (conservative; caches build in the read-mostly phases they exist for). A mid-commit
    /// internal read skips installing entirely (the lock is already held by this thread — the
    /// `rehydrate_elided_serialized` pattern). CAP policy (audit F2): an entry alone over the cap
    /// never installs (rebuild-then-clear thrash); a combined breach evicts the OTHER entries.
    fn install_streaming_cold(&self, table_name: &str, builder: ColdCacheBuilder) -> bool {
        self.install_streaming_cold_inner(table_name, builder, false)
    }

    /// `is_patch` keeps the BUILD counter honest (a patch re-install is not a fresh build — audit
    /// 6c-1 F3); everything else is identical.
    fn install_streaming_cold_inner(
        &self,
        table_name: &str,
        builder: ColdCacheBuilder,
        is_patch: bool,
    ) -> bool {
        // A spill IO error poisoned the capture: the chunk list is incomplete — never install it.
        if builder.poisoned {
            return false;
        }
        // 6c-1: a PATCHED entry can mix reused Spilled chunks with rebuilt Ram ones — class by the
        // chunks themselves, not the builder's own spill stream.
        let spilled = builder.spill.is_some()
            || builder
                .chunks
                .iter()
                .any(|c| matches!(c.payload, ColdPayload::Spilled { .. }));
        let class_cap = if spilled {
            STREAMING_COLD_DISK_CAP_BYTES
        } else {
            STREAMING_COLD_CAP_BYTES
        };
        if builder.total_payload_bytes > class_cap {
            return false;
        }
        if self.mvcc_read_skips_leader_check() {
            return false;
        }
        let _commit_guard = self.commit_state();
        let current = self.read_state.mvcc.table_rows(table_name).generation_payload();
        if !Arc::ptr_eq(&builder.generation, &current)
            || self.committed_seq() != builder.build_copin_s
        {
            return false;
        }
        let entry = Arc::new(ColdTableChunks {
            generation: builder.generation,
            column_signature: builder.column_signature,
            build_copin_s: builder.build_copin_s,
            chunk_target_bytes: builder.chunk_target_bytes,
            total_payload_bytes: builder.total_payload_bytes,
            spilled,
            chunks: builder.chunks,
        });
        let residency = &self.read_state.residency;
        let _publish = residency
            .streaming_cold_lock
            .lock()
            .expect("streaming cold-tier lock poisoned");
        let mut map = std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
        map.insert(table_name.to_string(), entry);
        // Per-class caps (RAM vs spilled/DISK): a breach evicts the OTHER entries of that class.
        let class_total: u64 = map
            .values()
            .filter(|c| c.spilled == spilled)
            .map(|c| c.total_payload_bytes)
            .sum();
        if class_total > class_cap {
            let kept = map.remove(table_name).expect("just inserted");
            map.retain(|_, c| c.spilled != spilled);
            map.insert(table_name.to_string(), kept);
        }
        if !is_patch {
            residency
                .streaming_cold_builds
                .fetch_add(1, Ordering::Relaxed);
        }
        if spilled {
            residency
                .streaming_cold_spills
                .fetch_add(1, Ordering::Relaxed);
        }
        residency.streaming_cold_chunks.store(Arc::new(map));
        true
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
        src: &ResidentExecSource,
        count_select: &Select,
        count_bound: &BoundRelationalSelect,
        partials: &mut Vec<Vec<SqlValue>>,
        total_matched: &mut i64,
    ) -> ChunkOutcome {
        // COUNT(*) over the chunk (predicate applied on-device): both the COUNT value AND the empty-set
        // guard for SUM/MIN/MAX (the general reduction hard-errors over an empty filtered set).
        let chunk_count = match self.execute_resident_expr_select_with_binding(
            count_select,
            table,
            Some(src),
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

        *total_matched = total_matched.saturating_add(chunk_count);
        if agg == StreamAgg::Count {
            // 6c-0: the chunk's COUNT becomes an Int8 partial row; the device SUM pass folds them.
            partials.push(vec![SqlValue::Int8(chunk_count)]);
            return ChunkOutcome::Ok;
        }
        if chunk_count == 0 {
            // No surviving row in this chunk -> nothing to reduce (avoids the empty-set hard-error).
            return ChunkOutcome::Ok;
        }
        let value = match self.execute_resident_expr_select_with_binding(
            select,
            table,
            Some(src),
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
        // 6c-0: the chunk's device partial (Null for an all-NULL matched set — the final pass's M3
        // validity conjunct skips it in-kernel) collects as a one-column row; NO host combine.
        partials.push(vec![value]);
        ChunkOutcome::Ok
    }

    /// 6c-0: the final SCALAR combine — ONE device aggregate pass over the collected partials
    /// uploaded as a synthesized one-column relation (the S-E.3 merge shape; the same pre-upload
    /// budget gate). Err(()) = the caller defers to the CPU path.
    #[allow(clippy::too_many_arguments)]
    fn combine_scalar_partials_on_device(
        &self,
        select: &Select,
        table: &RelationalTable,
        partial_type: SqlType,
        fold_projection: &SelectProjection,
        copin_s: Index,
        partials: &[Vec<SqlValue>],
        budget: u64,
    ) -> Result<SqlValue, ()> {
        let _ = (select, table);
        let partials_bytes: u64 = partials
            .iter()
            .map(|row| chunk_row_device_bytes(row, &[partial_type]))
            .sum();
        if partials_bytes > budget {
            return Err(());
        }
        let scalar_table =
            catalog_relation_table("public", "__stream_scalar", &[("__p0", partial_type)]);
        let fold_select = Select {
            table: scalar_table.name.clone(),
            distinct: false,
            projection: fold_projection.clone(),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        let fold_bound = bind_relational_select(&scalar_table, &fold_select).map_err(|_| ())?;
        let (descriptor, device_memory) = self
            .build_transient_relation_residency(&scalar_table, partials)
            .map_err(|_| ())?;
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(descriptor.resident_bytes, Ordering::Relaxed);
        let src = ResidentExecSource {
            descriptor: Arc::new(descriptor),
            device_memory: Arc::new(device_memory),
            row_count: partials.len() as u64,
        };
        let result = self
            .execute_resident_expr_select_with_binding(
                &fold_select,
                &scalar_table,
                Some(&src),
                fold_bound,
                copin_s,
                None,
                None,
                &[],
                &[],
                None,
                &[],
            )
            .map_err(|_| ())?;
        match result.rows.iter().next().and_then(|row| row.first()) {
            Some(cell) => Ok(cell.clone()),
            None => Err(()),
        }
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

    /// STRATA S-E.6 telemetry: streaming reads served from the COLD TIER (device-format byte
    /// replay — no MVCC decode, no payload assembly) and cold-tier builds installed.
    pub fn streaming_cold_hits(&self) -> u64 {
        self.read_state
            .residency
            .streaming_cold_hits
            .load(Ordering::Relaxed)
    }

    pub fn streaming_cold_builds(&self) -> u64 {
        self.read_state
            .residency
            .streaming_cold_builds
            .load(Ordering::Relaxed)
    }

    /// 6c-1 telemetry: cold entries PATCHED after a write (O(delta) maintenance, not O(table)).
    pub fn streaming_cold_patches(&self) -> u64 {
        self.read_state
            .residency
            .streaming_cold_patches
            .load(Ordering::Relaxed)
    }

    /// 6c-1 telemetry: dirty chunks rebuilt across all patches.
    pub fn streaming_cold_chunks_rebuilt(&self) -> u64 {
        self.read_state
            .residency
            .streaming_cold_chunks_rebuilt
            .load(Ordering::Relaxed)
    }

    /// S-E.6b telemetry: cold-tier installs whose payloads live in the unlinked spill file.
    pub fn streaming_cold_spills(&self) -> u64 {
        self.read_state
            .residency
            .streaming_cold_spills
            .load(Ordering::Relaxed)
    }
}
