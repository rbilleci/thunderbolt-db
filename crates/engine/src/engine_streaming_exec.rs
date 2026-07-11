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
    relational_row_key,
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
    /// P2: the tombstone mask of a sidecar-bearing COLD chunk (`deleted_by > read_txn_id`, applied
    /// IN-KERNEL by the mask VM through the sanctioned src=Some + vis=Some seam). `None` for
    /// scan-built and sidecar-free chunks — byte-identical pre-P2 behavior.
    visibility: Option<crate::engine_expr::ResidentVisibility>,
}

impl StagedChunk {
    /// Block until the upload completes and wrap the chunk as an executor source. Returns the
    /// chunk's tombstone mask (if any) beside the source — the caller threads it into the
    /// per-chunk execute.
    fn ready(
        self,
    ) -> Result<
        (
            ResidentExecSource,
            Option<crate::engine_expr::ResidentVisibility>,
        ),
        (),
    > {
        let device_memory = self.pending.wait().map_err(|_| ())?;
        Ok((
            ResidentExecSource {
                descriptor: Arc::new(self.snapshot),
                device_memory: Arc::new(device_memory),
                row_count: self.row_count,
            },
            self.visibility,
        ))
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
    pub(crate) chunks: Vec<ColdChunk>,
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
    /// P2 (sealed-shards-primary): the boundary the PAYLOAD's row set reflects — set at scan /
    /// rebuild / restore, PRESERVED by stamp-patches and reuse (unlike the entry's install
    /// boundary, which advances with every patch). A deleted TupleId's SLOT is its rank among the
    /// ids visible at THIS boundary within the chunk's range (scan order IS TupleId order), so the
    /// rank walk must anchor here, never at the entry boundary.
    payload_copin_s: Index,
    /// P2 (SV2 sparse versioning): the on-demand `deleted_by` tombstone SIDECAR — dense i64/slot,
    /// `0x7F`-live fill (a large POSITIVE signed i64: the mask compare is the SIGNED s64 kernel),
    /// COW-stamped host bytes. ABSENT for a delete-free chunk (it pays nothing — the SV2/HyPer
    /// property). At replay the sidecar is appended to the device buffer after the 8-aligned
    /// payload and applied IN-KERNEL via `ResidentVisibility { deleted_by_offset }` (`deleted_by >
    /// read_txn_id`). `created_by` is NEVER materialized for chunks — the payload boundary IS the
    /// D3 high-water mark (every payload row born-visible at it). NO version metadata rides the
    /// row payload itself (SV1/SV2, settled).
    deleted_by: Option<Arc<Vec<u8>>>,
}

/// P2: the SV2 live-fill byte for a cold chunk's `deleted_by` sidecar (mirrors the shard regions'
/// `DELETED_BY_LIVE_FILL_BYTE` — see engine_residency.rs).
const COLD_DELETED_BY_LIVE_FILL_BYTE: u8 = 0x7F;

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
        self.total_payload_bytes = self
            .total_payload_bytes
            .saturating_add(payload.len() as u64);
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
            // A freshly built payload reflects the builder's boundary and has no tombstones.
            payload_copin_s: self.build_copin_s,
            deleted_by: None,
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

/// TEST-ONLY (the no-flag-compliant internal-policy override, the spill-threshold pattern):
/// disable chunk-class ENTRY so the store-driven patch/stamp gates (6c-1/6c-3/P2) keep
/// exercising their machinery on tables that would otherwise class-enter mid-test. Production
/// behavior is unconditional.
#[cfg(test)]
pub(crate) static CHUNK_CLASS_ENTRY_ENABLED_TEST: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

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

/// 6c-3 (audit MEDIUM): the eager commit-hook patches only deltas up to this many changed chains —
/// larger writes defer to the lazy read-path patch so a bulk insert never stalls the global commit
/// mutex on decode/build/spill work. Engine-internal, not config.
const EAGER_PATCH_MAX_DELTA_ROWS: usize = 4096;

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
        (StreamAgg::Min, SelectProjection::Min { column }) => Some((
            value_type(column)?,
            SelectProjection::Min { column: partial },
        )),
        (StreamAgg::Max, SelectProjection::Max { column }) => Some((
            value_type(column)?,
            SelectProjection::Max { column: partial },
        )),
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
    Ordered {
        top_n: Option<usize>,
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
        // P4-2b: a CLASS table's fold must NEVER scan the (frozen) store — a cold MISS (budget
        // re-chunk, below-boundary reader, eviction) routes to the CPU-pinned path, whose guard
        // de-authoritizes first. The probe load here is the folds' own load (a hit is reused).
        if self.table_chunk_authoritative(&select.table).is_some() {
            let chunk_target = (budget / 2).max(1);
            if self
                .load_streaming_cold(&select.table, &table, chunk_target, copin_s)
                .is_none()
            {
                return Some(self.execute_relational_select_cpu_pinned(select));
            }
        }
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
                let Ok(next) = self.stage_cold_chunk(chunk, copin_s) else {
                    // A failed replay (e.g. a bad spill read) evicts the entry so the next read
                    // rebuilds instead of defer-thrashing (audit LOW).
                    self.evict_streaming_cold(&select.table);
                    return self.execute_relational_select_cpu_pinned(select);
                };
                if let Some(prev) = staged.replace(next) {
                    let Ok((src, chunk_vis)) = prev.ready() else {
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
                        chunk_vis,
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
                    let Ok(next) =
                        self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
                    else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    if let Some(prev) = staged.replace(next) {
                        let Ok((src, chunk_vis)) = prev.ready() else {
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
                            chunk_vis,
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
            let Ok(next) =
                self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
            else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            if let Some(prev) = staged.replace(next) {
                let Ok((src, chunk_vis)) = prev.ready() else {
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
                    chunk_vis,
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
            let Ok((src, chunk_vis)) = prev.ready() else {
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
                chunk_vis,
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
                let Ok(next) = self.stage_cold_chunk(chunk, copin_s) else {
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
                    let Ok((src, chunk_vis)) = prev.ready() else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    match self.project_streaming_chunk(
                        &chunk_select,
                        table,
                        &chunk_bound,
                        predicate,
                        copin_s,
                        &src,
                        chunk_vis,
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
                    let Ok(next) =
                        self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
                    else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    let to_compute = if window_bound.is_some() {
                        Some(next) // limited: compute eagerly (early-exit fidelity)
                    } else {
                        staged.replace(next) // unbounded: pipeline one chunk ahead
                    };
                    if let Some(prev) = to_compute {
                        let Ok((src, chunk_vis)) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.project_streaming_chunk(
                            &chunk_select,
                            table,
                            &chunk_bound,
                            predicate,
                            copin_s,
                            &src,
                            chunk_vis,
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
            let Ok(next) =
                self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
            else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            let to_compute = if window_bound.is_some() {
                Some(next)
            } else {
                staged.replace(next)
            };
            if let Some(prev) = to_compute {
                let Ok((src, chunk_vis)) = prev.ready() else {
                    return self.execute_relational_select_cpu_pinned(select);
                };
                match self.project_streaming_chunk(
                    &chunk_select,
                    table,
                    &chunk_bound,
                    predicate,
                    copin_s,
                    &src,
                    chunk_vis,
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
            let Ok((src, chunk_vis)) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.project_streaming_chunk(
                &chunk_select,
                table,
                &chunk_bound,
                predicate,
                copin_s,
                &src,
                chunk_vis,
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
        visibility: Option<crate::engine_expr::ResidentVisibility>,
        rows_out: &mut Vec<Vec<SqlValue>>,
    ) -> ChunkOutcome {
        let result = match self.execute_resident_expr_select_with_binding(
            chunk_select,
            table,
            Some(src),
            chunk_bound.clone(),
            copin_s,
            predicate,
            visibility,
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
        let mut partial_columns: Vec<(String, SqlType)> = vec![(group_column.clone(), key_type)];
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
                matches!(aggregate.kind, GroupedAggKind::Count | GroupedAggKind::Sum)
                    && matches!(partial_ty, SqlType::Numeric { .. })
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
                let Ok(next) = self.stage_cold_chunk(chunk, copin_s) else {
                    // A failed replay (e.g. a bad spill read) evicts the entry so the next read
                    // rebuilds instead of defer-thrashing (audit LOW).
                    self.evict_streaming_cold(&select.table);
                    return self.execute_relational_select_cpu_pinned(select);
                };
                if let Some(prev) = staged.replace(next) {
                    let Ok((src, chunk_vis)) = prev.ready() else {
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
                        chunk_vis,
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
                        let Ok((src, chunk_vis)) = prev.ready() else {
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
                            chunk_vis,
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
                    let Ok(next) =
                        self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
                    else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    if let Some(prev) = staged.replace(next) {
                        let Ok((src, chunk_vis)) = prev.ready() else {
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
                            chunk_vis,
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
                            let Ok((src, chunk_vis)) = prev.ready() else {
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
                                chunk_vis,
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
            let Ok(next) =
                self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
            else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            if let Some(prev) = staged.replace(next) {
                let Ok((src, chunk_vis)) = prev.ready() else {
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
                    chunk_vis,
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
            let Ok((src, chunk_vis)) = prev.ready() else {
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
                chunk_vis,
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
        visibility: Option<crate::engine_expr::ResidentVisibility>,
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
            visibility,
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
                let Ok(next) = self.stage_cold_chunk(chunk, copin_s) else {
                    // A failed replay (e.g. a bad spill read) evicts the entry so the next read
                    // rebuilds instead of defer-thrashing (audit LOW).
                    self.evict_streaming_cold(&select.table);
                    return self.execute_relational_select_cpu_pinned(select);
                };
                if let Some(prev) = staged.replace(next) {
                    let Ok((src, chunk_vis)) = prev.ready() else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    match self.ordered_streaming_chunk(
                        &chunk_select,
                        table,
                        &chunk_bound,
                        predicate,
                        copin_s,
                        &src,
                        chunk_vis,
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
                        let Ok((src, chunk_vis)) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.ordered_streaming_chunk(
                            &chunk_select,
                            table,
                            &chunk_bound,
                            predicate,
                            copin_s,
                            &src,
                            chunk_vis,
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
                    let Ok(next) =
                        self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
                    else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    if let Some(prev) = staged.replace(next) {
                        let Ok((src, chunk_vis)) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.ordered_streaming_chunk(
                            &chunk_select,
                            table,
                            &chunk_bound,
                            predicate,
                            copin_s,
                            &src,
                            chunk_vis,
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
                            let Ok((src, chunk_vis)) = prev.ready() else {
                                return self.execute_relational_select_cpu_pinned(select);
                            };
                            match self.ordered_streaming_chunk(
                                &chunk_select,
                                table,
                                &chunk_bound,
                                predicate,
                                copin_s,
                                &src,
                                chunk_vis,
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
                                        return self.execute_relational_select_cpu_pinned(select)
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
            let Ok(next) =
                self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture)
            else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            if let Some(prev) = staged.replace(next) {
                let Ok((src, chunk_vis)) = prev.ready() else {
                    return self.execute_relational_select_cpu_pinned(select);
                };
                match self.ordered_streaming_chunk(
                    &chunk_select,
                    table,
                    &chunk_bound,
                    predicate,
                    copin_s,
                    &src,
                    chunk_vis,
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
            let Ok((src, chunk_vis)) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.ordered_streaming_chunk(
                &chunk_select,
                table,
                &chunk_bound,
                predicate,
                copin_s,
                &src,
                chunk_vis,
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
        visibility: Option<crate::engine_expr::ResidentVisibility>,
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
            visibility,
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
            merged_bytes = merged_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
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
            builder.push(
                payload,
                snapshot.clone(),
                chunk_rows.len() as u64,
                chunk_range,
            );
        }
        Ok(StagedChunk {
            snapshot,
            pending,
            row_count: chunk_rows.len() as u64,
            visibility: None,
        })
    }

    /// S-E.6b (audit LOW): evict a table's cold entry after a replay failure — a bad spill file
    /// (disk fault) would otherwise defer-thrash every future streaming read on the table; dropping
    /// the entry lets the next read rebuild it.
    fn evict_streaming_cold(&self, table_name: &str) {
        // Audit H2: a CHUNK-AUTHORITATIVE table's entry is the record-of-truth for post-freeze
        // writes — fold-failure eviction must never remove it (the fold falls to the CPU-pinned
        // path whose guard de-authoritizes WITH the entry present, replaying the delta).
        if self.table_chunk_authoritative(table_name).is_some() {
            return;
        }
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
    fn stage_cold_chunk(
        &self,
        chunk: &ColdChunk,
        reader_copin_s: Index,
    ) -> Result<StagedChunk, ()> {
        let runtime = self.cuda_driver_probe_runtime();
        // RAM chunks borrow; spilled chunks positional-read from the unlinked file (an IO error is
        // a defer, never a wrong answer).
        let payload = chunk.payload.read()?;
        // P2: a sidecar-bearing chunk uploads payload + 8-aligned deleted_by sidecar as ONE device
        // buffer (the transient source is one allocation; `ResidentVisibility` addresses the
        // sidecar by ABSOLUTE offset). The concat is one host memcpy paid ONLY by delete-bearing
        // chunks — delete-free chunks keep the zero-copy borrow. The mask (`deleted_by >
        // read_txn_id`, signed s64) is ANDed in-kernel by the executor's mask VM.
        let (bytes, visibility) = match &chunk.deleted_by {
            None => (payload, None),
            Some(sidecar) => {
                let padded = payload.len().next_multiple_of(8);
                let mut buf = Vec::with_capacity(padded + sidecar.len());
                buf.extend_from_slice(&payload);
                buf.resize(padded, 0);
                buf.extend_from_slice(sidecar);
                (
                    std::borrow::Cow::Owned(buf),
                    Some(crate::engine_expr::ResidentVisibility {
                        read_txn_id: reader_copin_s as i64,
                        deleted_by_offset: Some(padded as u64),
                        created_by_offset: None,
                    }),
                )
            }
        };
        let pending = runtime
            .retain_device_memory_copy_async(chunk.snapshot.gpu_id, &bytes)
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
            visibility,
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
        // 6c-3 (audit F4 adopted): payload + descriptor ONLY — no throwaway upload. The replay
        // (stage_cold_chunk) stamps a fresh proof when it actually uploads.
        self.build_transient_relation_payload_only(table, rows)
            .map_err(|_| ())
    }

    /// P2: classify one changed chain as a PURE DELETE of a payload-visible row — the only
    /// change tolerated by the sidecar STAMP downgrade. Returns the deleting commit seq when the
    /// old and new chains are identical EXCEPT exactly one version (a payload row: `created_by <=
    /// payload_copin_s`, previously live) gained a `deleted_by` stamp. Anything else — tail
    /// growth, value edits, same-id version appends, vanished chains, double deletes — returns
    /// `None` and the chunk keeps the 6c-1 REBUILD arm (correctness backstop; never a wrong
    /// answer). Control-plane version-METADATA comparison only (charter: no row values computed,
    /// the equality checks are structural).
    fn classify_pure_delete(
        old: &[gpu_db_storage::TupleVersion],
        new: &[gpu_db_storage::TupleVersion],
        payload_copin_s: Index,
    ) -> Option<Index> {
        if old.len() != new.len() {
            return None;
        }
        let mut stamp: Option<Index> = None;
        for (o, n) in old.iter().zip(new.iter()) {
            if o == n {
                continue;
            }
            if stamp.is_some() {
                return None; // more than one changed version
            }
            if o.tuple_id != n.tuple_id
                || o.key != n.key
                || o.value != n.value
                || o.created_by != n.created_by
            {
                return None;
            }
            if o.deleted_by.is_some() || n.deleted_by.is_none() {
                return None;
            }
            if o.created_by > payload_copin_s {
                return None; // not a payload row (defensive: interior inserts cannot happen)
            }
            stamp = n.deleted_by;
        }
        stamp
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
    #[allow(clippy::too_many_arguments)]
    fn patch_streaming_cold(
        &self,
        table_name: &str,
        table: &RelationalTable,
        stale: &Arc<ColdTableChunks>,
        current: &Arc<crate::resident_storage::TableVersionData>,
        copin_s: Index,
        chunk_target_bytes: u64,
        commit_lock_held: bool,
    ) -> Option<Arc<ColdTableChunks>> {
        // The ALTER guard: a shape-changing DDL republished the store too — cached payload layouts
        // would be reused with the WRONG column shape. Signature inequality -> evict.
        let signature: Vec<(String, SqlType)> = table
            .columns
            .iter()
            .map(|c| (c.name.clone(), c.ty))
            .collect();
        if signature != stale.column_signature || stale.chunk_target_bytes != chunk_target_bytes {
            return None;
        }
        let changed = stale.generation.rows.changed_tuple_ids(&current.rows);
        // Map changed ids to dirty chunks via the effective tiling; ids past the last hi = tail.
        let mut dirty = vec![false; stale.chunks.len()];
        let his: Vec<u64> = stale.chunks.iter().map(|c| c.tuple_range.1).collect();
        let last_hi = his.last().copied().unwrap_or(0);
        let mut tail_dirty = stale.chunks.is_empty();
        let mut per_chunk_ids: Vec<Vec<u64>> = vec![Vec::new(); stale.chunks.len()];
        for id in &changed {
            if *id > last_hi {
                tail_dirty = true;
                continue;
            }
            let idx = his.partition_point(|hi| *hi < *id);
            dirty[idx] = true;
            per_chunk_ids[idx].push(*id);
        }
        // P2 — the SIDECAR STAMP DOWNGRADE: a dirty chunk whose every change is a PURE DELETE of
        // one of its payload rows keeps its bytes and gains tombstone stamps (an O(8B x rows)
        // sidecar COW) instead of the O(chunk) decode+rebuild. The row's SLOT is its rank among
        // the ids visible at the chunk's OWN payload boundary within the chunk's effective range
        // (scan order IS TupleId order; the walk anchors at `payload_copin_s`, never the entry
        // boundary — a stamped row stays IN the payload, masked in-kernel at replay). Any
        // classification failure keeps the rebuild arm.
        let mut stamps: Vec<Option<Vec<(usize, Index)>>> = vec![None; stale.chunks.len()];
        'downgrade: for i in 0..stale.chunks.len() {
            if !dirty[i] || per_chunk_ids[i].is_empty() {
                continue;
            }
            let chunk = &stale.chunks[i];
            if chunk.row_count == 0 {
                continue;
            }
            let eff_lo_i = if i == 0 {
                0
            } else {
                his[i - 1].saturating_add(1)
            };
            let payload_vis = StorageVisibility {
                read_txn_id: chunk.payload_copin_s,
            };
            let mut list: Vec<(usize, Index)> = Vec::with_capacity(per_chunk_ids[i].len());
            for id in &per_chunk_ids[i] {
                let (Some(old_chain), Some(new_chain)) =
                    (stale.generation.rows.chain(*id), current.rows.chain(*id))
                else {
                    continue 'downgrade;
                };
                let Some(stamp) =
                    Self::classify_pure_delete(old_chain, new_chain, chunk.payload_copin_s)
                else {
                    continue 'downgrade;
                };
                let Ok(slot) = stale.generation.rows.visible_count_in_range(
                    payload_vis,
                    eff_lo_i,
                    id.saturating_sub(1),
                ) else {
                    continue 'downgrade;
                };
                if slot >= chunk.row_count as usize {
                    continue 'downgrade; // rank disagrees with the payload — rebuild (defensive)
                }
                list.push((slot, stamp));
            }
            stamps[i] = Some(list);
            dirty[i] = false;
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
                stamps[last] = None; // the tail absorption needs the rebuild arm
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
        let mut stamped_rows: u64 = 0;
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
                // P2: a stamp-downgraded chunk reuses its payload and COWs its sidecar (get-or-
                // materialize at the 0x7F live fill — a delete-free chunk pays only here, on its
                // FIRST delete); a plain reuse carries both through unchanged.
                let deleted_by = match &stamps[i] {
                    Some(list) if !list.is_empty() => {
                        let mut bytes = match &chunk.deleted_by {
                            Some(existing) => existing.as_ref().clone(),
                            None => {
                                vec![COLD_DELETED_BY_LIVE_FILL_BYTE; (chunk.row_count as usize) * 8]
                            }
                        };
                        for (slot, stamp) in list {
                            bytes[slot * 8..slot * 8 + 8].copy_from_slice(&stamp.to_le_bytes());
                        }
                        stamped_rows += list.len() as u64;
                        Some(Arc::new(bytes))
                    }
                    _ => chunk.deleted_by.as_ref().map(Arc::clone),
                };
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
                    // P2: reuse preserves the payload's OWN boundary (stamps do NOT advance it).
                    payload_copin_s: chunk.payload_copin_s,
                    deleted_by,
                });
            }
            eff_lo = eff_hi.saturating_add(1);
        }
        // The tail (unless the runt-coalesce already extended the last rebuild through MAX).
        let tail_absorbed = tail_dirty && !stale.chunks.is_empty() && dirty[stale.chunks.len() - 1];
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
            .map(|c| {
                let payload = match &c.payload {
                    ColdPayload::Ram(bytes) => bytes.len() as u64,
                    ColdPayload::Spilled { len, .. } => *len as u64,
                };
                // P2: sidecars count against the cap class too (they are held host bytes).
                payload + c.deleted_by.as_ref().map_or(0, |b| b.len() as u64)
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
        if !self.install_streaming_cold_inner(table_name, builder, true, commit_lock_held) {
            return None;
        }
        self.read_state
            .residency
            .streaming_cold_patches
            .fetch_add(1, Ordering::Relaxed);
        if stamped_rows > 0 {
            self.read_state
                .residency
                .streaming_cold_stamps
                .fetch_add(stamped_rows, Ordering::Relaxed);
        }
        self.read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
    }

    /// 6c-3 — EAGER COLD-TIER MAINTENANCE AT COMMIT: for each committed table that HAS a cold
    /// entry, patch it in place (O(delta) via the 6c-1 patcher) so subsequent READS never pay the
    /// maintenance. Self-gating on entry existence (no flag — the no-flag mandate); entirely
    /// best-effort (any failure -> the read path patches lazily as before; NEVER fails the
    /// already-durable commit); runs post-publish under the held commit mutex (committed_seq
    /// frozen -> the settled proof is trivial). Catalog resolution uses the PUBLISHED snapshot,
    /// never the latch (the rehydrate lesson). First builds stay LAZY on first read (an eager
    /// O(table) first build would stall the commit; its deletion is the sealed-shards-primary arc).
    pub(crate) fn maintain_streaming_cold_on_commit(
        &self,
        tables: &std::collections::BTreeSet<String>,
    ) {
        if tables.is_empty() {
            return;
        }
        let map = self.read_state.residency.streaming_cold_chunks.load();
        for table_name in tables {
            let Some(entry) = map.get(table_name).cloned() else {
                continue;
            };
            let current = self
                .read_state
                .mvcc
                .table_rows(table_name)
                .generation_payload();
            if Arc::ptr_eq(&entry.generation, &current) {
                continue; // already fresh
            }
            let Some(table) = self
                .catalog_snapshot()
                .relational_catalog
                .get(table_name)
                .cloned()
            else {
                continue;
            };
            // 6c-3 (audit MEDIUM): BOUND the eager work — the hook runs synchronously under the
            // GLOBAL commit mutex, so a bulk write's tail rebuild (possibly with spill-file IO)
            // must never head-of-line-block every committer. Oversized deltas defer to the lazy
            // read-path patch (the unchanged correctness backstop).
            let changed = entry.generation.rows.changed_tuple_ids(&current.rows);
            if changed.len() > EAGER_PATCH_MAX_DELTA_ROWS {
                continue;
            }
            let copin_s = self.committed_seq();
            let _ = self.patch_streaming_cold(
                table_name,
                &table,
                &entry,
                &current,
                copin_s,
                entry.chunk_target_bytes,
                true,
            );
        }
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
        let current = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
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
                false,
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
        self.install_streaming_cold_inner(table_name, builder, false, false)
    }

    /// `is_patch` keeps the BUILD counter honest (a patch re-install is not a fresh build — audit
    /// 6c-1 F3); everything else is identical.
    fn install_streaming_cold_inner(
        &self,
        table_name: &str,
        builder: ColdCacheBuilder,
        is_patch: bool,
        // 6c-3: the caller IS the serialized committer (both engine_commit hook sites hold the
        // commit mutex — one with the internal-read flag UNSET, so inference would deadlock;
        // explicit beats inference). NOTE (audit): committed_seq is NOT frozen under this mutex —
        // intent lanes publish it LOCK-FREE off this path — the actual safety is (a) the STRICT
        // EQUALITY guard below (a concurrent bump FAILS the install — a safe miss, never a
        // higher-stamp pass), (b) generation ptr identity (every write COW-publishes a fresh Arc),
        // and (c) per-read visibility at replay. Never weaken the generation check on a
        // frozen-seq assumption.
        commit_lock_held: bool,
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
        let _commit_guard = if commit_lock_held {
            None
        } else {
            if self.mvcc_read_skips_leader_check() {
                // Mid-commit INTERNAL READ (not our hook): acquiring the lock would self-deadlock
                // and the boundary is mid-mutation — skip installing (the read path rebuilds).
                return false;
            }
            Some(self.commit_state())
        };
        let current = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
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
            // P4-2b: a CHUNK-AUTHORITATIVE table's entry is its representation-of-record — cap
            // pressure must never evict it (the frozen store lacks the post-freeze writes).
            let protected = self.read_state.residency.chunk_authoritative_tables.load();
            map.retain(|name, c| c.spilled != spilled || protected.contains_key(name));
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
        visibility: Option<crate::engine_expr::ResidentVisibility>,
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
            visibility,
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
            visibility,
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

    /// P3 (sealed-shards-primary): the DML WHERE-locate as a STREAMING FOLD — a DELETE/UPDATE on a
    /// NON-ADMITTED table whose predicate the value index cannot bound (no Eq leaf) previously fell
    /// to the pure-host seq_scan + `select_filter_matches` loop (the CPU relational engine's core,
    /// ADR-006 debt; the resident device arm needs shards a non-admitted table does not have). Here
    /// the predicate runs ON THE DEVICE over bounded chunks instead: the table's visible rows are
    /// staged with a synthesized trailing `__row_id` int8 column (the S-E.3 synthesized-relation
    /// pattern — real columns keep their catalog indexes, so the DML predicate lowering binds
    /// unchanged), each chunk is device-filtered + gathered, and each survivor's `__row_id` maps it
    /// back to `(row_id, row_key, row image)` — the same `DmlResolvedMatch` triple every other arm
    /// yields. The host stages/decodes (the REGISTERED 6c-1 scan-build staging debt, deletion
    /// trigger = P4) and reassembles identity — it never evaluates the predicate. Visibility is
    /// exact by construction (the scan is boundary-pinned like the host arm); the device predicate
    /// is trusted exactly as every streaming READ trusts it (no host recheck — the read folds set
    /// the precedent and the differentials gate it). Returns `None` to DECLINE (no budget, elided,
    /// un-lowerable predicate, any staging/executor failure) — the caller falls to the host scan,
    /// never a wrong answer; `Some(vec![])` is a VALID zero-match resolve.
    pub(crate) fn try_streaming_dml_locate(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        visibility: StorageVisibility,
        table_rows: &crate::resident_storage::TableRowsView,
    ) -> Option<Vec<(u64, String, Vec<SqlValue>)>> {
        let gpu_id = self.planner.default_gpu_id();
        let budget = self.relational_residency_budget_bytes(gpu_id)?;
        if budget == 0 {
            return None;
        }
        // Never locate through an ELIDED table's host store (stale by design); its resolve is the
        // resident device arm upstream.
        if self.table_install_elided(&table.name) {
            return None;
        }
        let predicate =
            crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(table, filter_groups)?;
        // The synthesized locate relation: the real columns (catalog order, indexes unchanged) plus
        // the trailing row-identity column the survivors carry back.
        let mut locate_columns: Vec<(String, SqlType)> = table
            .columns
            .iter()
            .map(|c| (c.name.clone(), c.ty))
            .collect();
        locate_columns.push(("__row_id".to_string(), SqlType::Int8));
        let locate_refs: Vec<(&str, SqlType)> = locate_columns
            .iter()
            .map(|(name, ty)| (name.as_str(), *ty))
            .collect();
        let locate_table = catalog_relation_table(&table.schema, "__stream_locate", &locate_refs);
        let chunk_select = Select {
            table: locate_table.name.clone(),
            distinct: false,
            projection: SelectProjection::All,
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        let chunk_bound = bind_relational_select(&locate_table, &chunk_select).ok()?;
        let column_types: Vec<SqlType> = locate_columns.iter().map(|(_, ty)| *ty).collect();
        let chunk_target_bytes = (budget / 2).max(1);
        let copin_s = visibility.read_txn_id;
        let prefix = relational_key_prefix(&table.name);

        let mut matches: Vec<(u64, String, Vec<SqlValue>)> = Vec::new();
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        let mut capture: Option<ColdCacheBuilder> = None; // never capture: not the table's schema
        let mut run_chunk = |chunk_rows: &[Vec<SqlValue>],
                             matches: &mut Vec<(u64, String, Vec<SqlValue>)>|
         -> Result<(), ()> {
            let staged =
                self.stage_streaming_chunk(&locate_table, chunk_rows, (1, 0), &mut capture)?;
            let (src, _) = staged.ready()?; // P3 locate chunks are scan-built (no sidecar)
            let result = self
                .execute_resident_expr_select_with_binding(
                    &chunk_select,
                    &locate_table,
                    Some(&src),
                    chunk_bound.clone(),
                    copin_s,
                    Some(&predicate),
                    None,
                    &[],
                    &[],
                    None,
                    &[],
                )
                .map_err(|_| ())?;
            for row in result.rows.iter() {
                let mut row = row.to_vec();
                // The trailing survivor cell IS the identity (staged from tuple.tuple_id below).
                let Some(SqlValue::Int8(id)) = row.pop() else {
                    return Err(());
                };
                let row_id = id as u64;
                matches.push((row_id, relational_row_key(&table.name, row_id), row));
            }
            Ok(())
        };

        let mut cursor = table_rows.store().seq_scan_open(visibility).ok()?;
        while let Some(tuple) = cursor.next() {
            if !tuple.key.starts_with(&prefix) {
                continue;
            }
            let mut decoded = decode_relational_row(&tuple.value, &table.columns).ok()?;
            decoded.push(SqlValue::Int8(tuple.tuple_id as i64));
            chunk_bytes =
                chunk_bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
            chunk_rows.push(decoded);
            if chunk_bytes >= chunk_target_bytes {
                if run_chunk(&chunk_rows, &mut matches).is_err() {
                    return None;
                }
                chunk_rows.clear();
                chunk_bytes = 0;
            }
        }
        drop(cursor);
        if !chunk_rows.is_empty() && run_chunk(&chunk_rows, &mut matches).is_err() {
            return None;
        }
        self.read_state
            .residency
            .dml_streaming_resolve_hits
            .fetch_add(1, Ordering::Relaxed);
        Some(matches)
    }

    /// P4-1: the REVERSE GATHER driver — decode a table's ENTIRE cold entry into catalog-order
    /// rows visible at `rtx` (chunk order = TupleId scan order, so concatenation preserves the
    /// store's iteration order). The de-authoritization building block; `None` = no cold entry.
    // Production callers arrive with P4-2b (patch-failure de-auth) and P4-3 (the read-
    // completeness sticky exit); until then only the round-trip gate exercises it.
    #[allow(dead_code)]
    pub(crate) fn reverse_gather_streamed_rows(
        &self,
        table_name: &str,
        rtx: Index,
    ) -> Option<Result<Vec<Vec<SqlValue>>, EngineError>> {
        let entry = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()?;
        let table = self
            .catalog_snapshot()
            .relational_catalog
            .get(table_name)
            .cloned()?;
        let mut rows: Vec<Vec<SqlValue>> = Vec::new();
        for chunk in &entry.chunks {
            match decode_cold_chunk_rows(&table, chunk, rtx) {
                Ok(mut decoded) => rows.append(&mut decoded),
                Err(err) => return Some(Err(err)),
            }
        }
        Some(Ok(rows))
    }

    /// P4-2a — THE CHUNK-NATIVE LOCATE (design-review C1: the P3 locate derives identity from a
    /// STORE scan, unusable store-free): evaluate a DML predicate over the table's COLD CHUNKS
    /// THEMSELVES, returning matching `(chunk_idx, local slots)` coordinates. Each chunk replays
    /// through the SAME staging the read folds use (payload + sidecar upload) and the SAME
    /// slot-locate primitive the resident shard DML uses (`lower_resident_predicate` — the mask VM
    /// with the sidecar visibility ANDed on, so already-tombstoned slots never re-locate). No
    /// store, no row decode, no projection — the device returns slots natively; the host only
    /// orchestrates (charter: control plane). `None` = decline (no entry, a reader boundary below
    /// the entry, any staging/lowering failure) — the caller falls to its store-era arm while one
    /// exists.
    /// ## P4-2b CALLER OBLIGATIONS (audit, forward-looking)
    /// 1. COORDINATE TOKEN (MEDIUM latent): the slots are positions in the entry INSTALLED AT
    ///    LOCATE TIME. `stamp_streaming_cold_slots` reloads the CURRENT entry — an intervening
    ///    patch install (eager hook / lazy read patch) re-tiles chunks and the coordinates
    ///    mis-align (the install guard cannot catch it: the new entry's generation IS current).
    ///    The caller MUST run locate→stamp inside ONE commit-lock critical section with no
    ///    intervening patch, or carry an entry-identity token and refuse on mismatch.
    /// 2. STORE DIVERGENCE (LOW): a store-free stamp hides rows the store still holds; a LATER
    ///    store-driven patch REBUILD of that chunk rebuilds from the store and RESURRECTS them
    ///    (sidecar discarded). For chunk-authoritative tables the store must be dropped/frozen so
    ///    the rebuild arm is unreachable — until then this primitive must not run beside live
    ///    store writes to the same table.
    // Production caller = P4-2b (the class write path); the differential gate exercises it now.
    #[allow(dead_code)]
    pub(crate) fn locate_streaming_cold_slots(
        &self,
        table: &RelationalTable,
        predicate: &crate::engine_expr::ResidentExpr,
        rtx: Index,
    ) -> Option<Vec<(usize, Vec<u32>)>> {
        let entry = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(&table.name)
            .cloned()?;
        if rtx < entry.build_copin_s {
            return None; // below the entry boundary — the chunks cannot serve this reader
        }
        let mut out: Vec<(usize, Vec<u32>)> = Vec::new();
        for (idx, chunk) in entry.chunks.iter().enumerate() {
            if chunk.row_count == 0 {
                continue;
            }
            let staged = self.stage_cold_chunk(chunk, rtx).ok()?;
            let (src, vis) = staged.ready().ok()?;
            let slots = self
                .lower_resident_predicate(
                    predicate,
                    table,
                    &src.descriptor,
                    &src.device_memory,
                    chunk.row_count,
                    vis,
                )
                .ok()?;
            if !slots.is_empty() {
                out.push((idx, slots));
            }
        }
        Some(out)
    }

    /// P4-2a — THE LOCATE-DRIVEN STAMP (design-review C1: the P2 stamp rides the store-generation
    /// diff + chain classification, unusable store-free): tombstone the given `(chunk_idx, slot)`
    /// coordinates directly — sidecar COW (get-or-materialize at the 0x7F live fill), stamp value
    /// = the deleting commit's boundary, entry re-installed at that boundary under the standard
    /// strict-equality settled proof (the P4-2b caller stamps under the commit lock right after
    /// the publish, so equality holds by construction; a racing commit fails the install — a safe
    /// decline). The entry's pinned generation is UNCHANGED (a store-free write publishes no
    /// generation). Returns false on any invalid coordinate or a failed install.
    /// See `locate_streaming_cold_slots` — the two P4-2b caller obligations (the coordinate
    /// token / single-critical-section rule, and the store-divergence rebuild hazard) apply to
    /// this pair as a unit.
    // Production caller = P4-2b; the isolation gate exercises it now.
    #[allow(dead_code)]
    pub(crate) fn stamp_streaming_cold_slots(
        &self,
        table_name: &str,
        located: &[(usize, Vec<u32>)],
        stamp: Index,
        commit_lock_held: bool,
    ) -> bool {
        if located.is_empty() {
            return true;
        }
        let entry = match self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
        {
            Some(entry) => entry,
            None => return false,
        };
        let mut stamped_rows: u64 = 0;
        let mut chunks: Vec<ColdChunk> = Vec::with_capacity(entry.chunks.len());
        let mut sidecar_growth: u64 = 0;
        for (idx, chunk) in entry.chunks.iter().enumerate() {
            let slots = located
                .iter()
                .find(|(chunk_idx, _)| *chunk_idx == idx)
                .map(|(_, slots)| slots.as_slice())
                .unwrap_or(&[]);
            let deleted_by = if slots.is_empty() {
                chunk.deleted_by.as_ref().map(Arc::clone)
            } else {
                let mut bytes = match &chunk.deleted_by {
                    Some(existing) => existing.as_ref().clone(),
                    None => {
                        sidecar_growth += chunk.row_count * 8;
                        vec![COLD_DELETED_BY_LIVE_FILL_BYTE; (chunk.row_count as usize) * 8]
                    }
                };
                for slot in slots {
                    let slot = *slot as usize;
                    if slot >= chunk.row_count as usize {
                        return false; // an out-of-range coordinate: refuse the whole stamp
                    }
                    bytes[slot * 8..slot * 8 + 8].copy_from_slice(&stamp.to_le_bytes());
                }
                stamped_rows += slots.len() as u64;
                Some(Arc::new(bytes))
            };
            chunks.push(ColdChunk {
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
                payload_copin_s: chunk.payload_copin_s,
                deleted_by,
            });
        }
        let builder = ColdCacheBuilder {
            generation: Arc::clone(&entry.generation),
            build_copin_s: stamp,
            chunk_target_bytes: entry.chunk_target_bytes,
            total_payload_bytes: entry.total_payload_bytes + sidecar_growth,
            column_signature: entry.column_signature.clone(),
            chunks,
            spill: None,
            poisoned: false,
        };
        if !self.install_streaming_cold_inner(table_name, builder, true, commit_lock_held) {
            return false;
        }
        self.read_state
            .residency
            .streaming_cold_stamps
            .fetch_add(stamped_rows, Ordering::Relaxed);
        true
    }

    // ================= P4-2b-i (S-E.P4): THE CHUNK-AUTHORITATIVE CLASS =================
    //
    // A table whose ONLY representation-of-record for post-entry writes is its cold chunks. The
    // host store FREEZES at the entry boundary (writes skip the install; the frozen chains keep
    // serving readers pinned BELOW the boundary — exact MVCC time travel); everything at-or-above
    // streams from the chunks. The class is INTRINSIC (no flag): entered at the commit hook when
    // eligible, exited LOUDLY (de-authoritization) on any shape the chunks cannot serve. The
    // freeze — not a drop — closes the design-review C3 below-boundary-reader hole without a
    // reader tracker, and makes the P4-2a store-divergence rebuild hazard structurally
    // unreachable: a class table's writes publish NO store generation, so the entry's pinned
    // generation stays pointer-current forever (always a HIT; the patcher never fires). RAM
    // reclamation of the frozen rows is P4-5 (behind a reader fence).

    /// The class check: `Some(freeze boundary)` when `table` is chunk-authoritative.
    pub(crate) fn table_chunk_authoritative(&self, table: &str) -> Option<Index> {
        self.read_state
            .residency
            .chunk_authoritative_tables
            .load()
            .get(table)
            .copied()
    }

    /// Catalog eligibility (design review H1: RUNTIME state does the rest): NO unique index
    /// (per-insert uniqueness over chunks would be an O(table) fold scan) and NO FK edge in
    /// either direction (inbound-FK validation scans the provider host-side). Every scalar type
    /// is chunk-encodable, so types never gate — the cold entry's existence is the real
    /// structural gate.
    fn chunk_class_eligible(catalog: &crate::CatalogSnapshot, table_name: &str) -> bool {
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return false;
        };
        if table.indexes.iter().any(|index| index.unique) {
            return false;
        }
        if !table.foreign_keys.is_empty() {
            return false;
        }
        // Inbound FK: any OTHER table referencing this one.
        !catalog.relational_catalog.values().any(|other| {
            other
                .foreign_keys
                .iter()
                .any(|fk| fk.referenced_table == table_name)
        })
    }

    /// CLASS ENTRY — called from the applied-commit hook UNDER THE COMMIT LOCK, strictly in the
    /// `else` of the elision ENTER (mutual exclusion by construction, review H1). Enters when the
    /// table is eligible, NOT elided, has a FRESH cold entry (generation pointer-current AND
    /// boundary == committed_seq — the eager patch for this very commit just ran), and streaming
    /// is active (a budget is configured). The freeze boundary = the current commit index.
    pub(crate) fn maybe_enter_chunk_class(&self, table_name: &str) {
        #[cfg(test)]
        if !CHUNK_CLASS_ENTRY_ENABLED_TEST.load(Ordering::Relaxed) {
            return;
        }
        if self.table_chunk_authoritative(table_name).is_some()
            || self.table_install_elided(table_name)
        {
            return;
        }
        let gpu_id = self.planner.default_gpu_id();
        match self.relational_residency_budget_bytes(gpu_id) {
            Some(budget) if budget > 0 => {}
            _ => return,
        }
        let catalog = self.catalog_snapshot();
        if !Self::chunk_class_eligible(&catalog, table_name) {
            return;
        }
        let residency = &self.read_state.residency;
        let Some(entry) = residency.streaming_cold_chunks.load().get(table_name).cloned() else {
            return;
        };
        let current = self.read_state.mvcc.table_rows(table_name).generation_payload();
        if !Arc::ptr_eq(&entry.generation, &current)
            || entry.build_copin_s != self.committed_seq()
        {
            return; // not fresh at THIS commit — a later commit's hook will retry
        }
        let boundary = entry.build_copin_s;
        let mut map = std::collections::BTreeMap::clone(
            &residency.chunk_authoritative_tables.load(),
        );
        map.insert(table_name.to_string(), boundary);
        residency.chunk_authoritative_tables.store(Arc::new(map));
        residency
            .chunk_class_entries
            .fetch_add(1, Ordering::Relaxed);
    }

    /// THE CLASS INSERT MATERIALIZATION — called from the applied-commit hook UNDER THE COMMIT
    /// LOCK (obligation 1: one critical section, no intervening patch — the class entry's
    /// generation never changes so no patch can interpose). Appends the statement's OWN rows as
    /// fresh TAIL chunks (payload boundary = this commit — the born gate + de-auth read it) and
    /// re-installs at the commit boundary. `false` = the caller must DE-AUTHORITIZE (the commit
    /// is already durable in the WAL; the chunks just could not absorb it).
    pub(crate) fn append_streaming_cold_tail(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
        commit_seq: Index,
    ) -> bool {
        if rows.is_empty() {
            return true;
        }
        let residency = &self.read_state.residency;
        let Some(entry) = residency.streaming_cold_chunks.load().get(&table.name).cloned()
        else {
            return false;
        };
        let column_types: Vec<SqlType> = table.columns.iter().map(|c| c.ty).collect();
        let mut builder = ColdCacheBuilder {
            generation: Arc::clone(&entry.generation),
            build_copin_s: commit_seq,
            chunk_target_bytes: entry.chunk_target_bytes,
            total_payload_bytes: entry.total_payload_bytes,
            column_signature: entry.column_signature.clone(),
            chunks: entry
                .chunks
                .iter()
                .map(|chunk| ColdChunk {
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
                    payload_copin_s: chunk.payload_copin_s,
                    deleted_by: chunk.deleted_by.as_ref().map(Arc::clone),
                })
                .collect(),
            spill: None,
            poisoned: false,
        };
        // Chunk the statement rows by device bytes (the fold's own sizing); each tail chunk's
        // payload boundary = this commit. Class tables have no store TupleIds — the tuple_range
        // is the documented empty sentinel (the patcher never runs on a class table). Tail
        // chunks are built DIRECTLY as RAM chunks (audit L6: the builder's retro-spill would
        // POISON on a pre-spilled base; a mixed Spilled-base/Ram-tail entry is legal — P2). The
        // entry's unbounded growth is accepted BY DESIGN (record-of-truth; VACUUM compaction is
        // P4-5) and ledgered.
        let mut start = 0usize;
        while start < rows.len() {
            let mut bytes: u64 = 0;
            let mut end = start;
            while end < rows.len() {
                bytes = bytes.saturating_add(chunk_row_device_bytes(&rows[end], &column_types));
                end += 1;
                if bytes >= builder.chunk_target_bytes {
                    break;
                }
            }
            let Ok((snapshot, payload)) =
                self.build_transient_relation_payload_only(table, &rows[start..end])
            else {
                return false;
            };
            builder.total_payload_bytes += payload.len() as u64;
            builder.chunks.push(ColdChunk {
                payload: ColdPayload::Ram(Arc::new(payload)),
                snapshot,
                row_count: (end - start) as u64,
                tuple_range: (1, 0),
                payload_copin_s: commit_seq,
                deleted_by: None,
            });
            start = end;
        }
        self.install_streaming_cold_class(&table.name, builder)
    }

    /// The CLASS-PATH install: the general install's strict `committed_seq == build` proof cannot
    /// hold here — the tail append runs INSIDE the apply, BEFORE `publish_committed_seq` (the
    /// boundary is the commit being applied). Settledness comes from the structure instead: the
    /// caller holds the COMMIT LOCK, class tables are serial-path-only (no lock-free lane
    /// publishes touch them), and the FROZEN generation is verified pointer-current (a class
    /// table's store never republishes — inequality means the class was exited mid-flight and
    /// the append must fail into the de-auth backstop).
    fn install_streaming_cold_class(&self, table_name: &str, builder: ColdCacheBuilder) -> bool {
        if builder.poisoned {
            return false;
        }
        let current = self.read_state.mvcc.table_rows(table_name).generation_payload();
        if !Arc::ptr_eq(&builder.generation, &current) {
            return false;
        }
        let spilled = builder.spill.is_some()
            || builder
                .chunks
                .iter()
                .any(|c| matches!(c.payload, ColdPayload::Spilled { .. }));
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
        residency.streaming_cold_chunks.store(Arc::new(map));
        true
    }

    /// DE-AUTHORITIZATION — the STICKY EXIT (the class twin of `rehydrate_elided_serialized`):
    /// replay the POST-FREEZE DELTA back into the FROZEN store as normal chain mutations, so the
    /// store becomes whole again at EVERY boundary (tail rows insert with `created_by = their
    /// chunk's payload boundary` — readers below it keep not seeing them; the frozen chains
    /// below the boundary were never touched). Runs under the commit lock; the cold entry is
    /// EVICTED (its pinned generation is superseded by the replay's COW publishes; the next
    /// streaming read rebuilds a clean cache). Any read/DML shape the chunks cannot serve exits
    /// through here — loud, counted, correct.
    pub(crate) fn deauthoritize_chunk_table(
        &self,
        table_name: &str,
        // Audit C1 (the COPY-path deadlock): the commit mutex is NOT re-entrant and the
        // internal-read flag is NOT set on every locked path — the caller states lock ownership
        // EXPLICITLY (the install_streaming_cold_inner precedent).
        commit_lock_held: bool,
    ) -> Result<(), EngineError> {
        let exit = |engine: &Self| -> Result<(), EngineError> {
            let residency = &engine.read_state.residency;
            let Some(freeze) = engine.table_chunk_authoritative(table_name) else {
                return Ok(()); // another exiter won the race
            };
            let Some(table) = engine
                .catalog_snapshot()
                .relational_catalog
                .get(table_name)
                .cloned()
            else {
                return Ok(());
            };
            let entry = residency
                .streaming_cold_chunks
                .load()
                .get(table_name)
                .cloned();
            // Audit H2: a class table WITHOUT its cold entry has LOST post-freeze writes — a
            // silent freeze-only exit would serve wrong results. Fail LOUDLY (recovery = WAL
            // replay); the eviction guards below make this unreachable.
            if entry.is_none() {
                return Err(EngineError::ApplyFailed(format!(
                    "chunk-authoritative table \"{table_name}\" lost its cold entry — refusing \
                     a freeze-only de-authoritization (post-freeze writes live only in chunks)"
                )));
            }
            if let Some(entry) = entry {
                for chunk in &entry.chunks {
                    // P4-2b-i is INSERT-only: the post-freeze delta = tail chunks born above the
                    // freeze (sidecar stamps above the freeze arrive with P4-2b-ii).
                    if chunk.payload_copin_s <= freeze {
                        continue;
                    }
                    let rows =
                        decode_cold_chunk_rows(&table, chunk, chunk.payload_copin_s)
                            .map_err(|e| {
                                EngineError::ApplyFailed(format!(
                                    "de-authoritization decode failed: {e}"
                                ))
                            })?;
                    let born = chunk.payload_copin_s;
                    // Fresh row ids (the class INSERT advanced the allocator without assigning;
                    // ids are internal-only for a keyless FK-free table — divergence from the
                    // prepare-time ids is unobservable, and WAL replay derives its own).
                    let keyed: Vec<(String, Vec<SqlValue>)> = rows
                        .into_iter()
                        .map(|row| {
                            let row_id = engine
                                .read_state
                                .mvcc
                                .next_row_id
                                .fetch_add(1, Ordering::Relaxed);
                            (
                                crate::rel_exec_helpers::relational_row_key(
                                    table_name, row_id,
                                ),
                                row,
                            )
                        })
                        .collect();
                    let index_entries =
                        crate::rel_exec_helpers::relational_value_index_entries_for_rows(
                            &table.columns,
                            &keyed,
                        );
                    // Tuple ids come from the SHARED MvccData allocator (the store-local
                    // counter is NOT the authority — partition stores share one id space via
                    // reserve_tuple_id; a local allocation would COLLIDE and replace live chains).
                    let tuple_ids: Vec<gpu_db_storage::TupleId> = keyed
                        .iter()
                        .map(|_| engine.read_state.mvcc.reserve_tuple_id())
                        .collect();
                    engine.read_state.mvcc.with_table_mut(table_name, |data| {
                        for (tuple_id, (key, row)) in tuple_ids.iter().zip(keyed.iter()) {
                            data.rows
                                .tuple_insert_reserved_key_with_id(
                                    *tuple_id,
                                    gpu_db_storage::NewTuple {
                                        key: key.clone(),
                                        value: crate::rel_exec_helpers::encode_relational_row(
                                            row,
                                        ),
                                    },
                                    born,
                                )
                                .map_err(|err| {
                                    EngineError::ApplyFailed(err.to_string())
                                })?;
                        }
                        for (entry_key, row_keys) in &index_entries {
                            let mut slot = data
                                .value_index
                                .get(entry_key)
                                .cloned()
                                .unwrap_or_default();
                            slot.extend(row_keys.iter().cloned());
                            data.value_index.insert(entry_key.clone(), slot);
                        }
                        Ok::<(), EngineError>(())
                    })?;
                }
            }
            // Leave the class + evict the (now superseded-generation) cache entry.
            let mut map = std::collections::BTreeMap::clone(
                &residency.chunk_authoritative_tables.load(),
            );
            map.remove(table_name);
            residency.chunk_authoritative_tables.store(Arc::new(map));
            engine.evict_streaming_cold(table_name);
            residency
                .chunk_class_deauths
                .fetch_add(1, Ordering::Relaxed);
            Ok(())
        };
        if commit_lock_held || self.mvcc_read_skips_leader_check() {
            return exit(self);
        }
        let _commit_guard = self.commit_state();
        exit(self)
    }

    /// P4-2b telemetry.
    pub fn chunk_class_entries(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_entries
            .load(Ordering::Relaxed)
    }
    pub fn chunk_class_skipped_installs(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_skipped_installs
            .load(Ordering::Relaxed)
    }
    pub fn chunk_class_deauths(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_deauths
            .load(Ordering::Relaxed)
    }

    /// P2 telemetry: cold-chunk rows tombstone-stamped in place of a chunk rebuild.
    pub fn streaming_cold_stamps(&self) -> u64 {
        self.read_state
            .residency
            .streaming_cold_stamps
            .load(Ordering::Relaxed)
    }

    /// P3 telemetry: DML WHERE-locates resolved on-device via the streaming fold.
    pub fn dml_streaming_resolve_hits(&self) -> u64 {
        self.read_state
            .residency
            .dml_streaming_resolve_hits
            .load(Ordering::Relaxed)
    }

    /// P1 telemetry: cold-tier tables written into the durable checkpoint artifact.
    pub fn streaming_cold_checkpointed(&self) -> u64 {
        self.read_state
            .residency
            .streaming_cold_checkpointed
            .load(Ordering::Relaxed)
    }

    /// P1 telemetry: cold-tier tables restored from the checkpoint artifact at reopen.
    pub fn streaming_cold_restored(&self) -> u64 {
        self.read_state
            .residency
            .streaming_cold_restored
            .load(Ordering::Relaxed)
    }

    // ================= P1 (sealed-shards-primary): THE DURABLE COLD CHECKPOINT =================
    //
    // The cold tier's device-format chunk bytes become DURABLE via the checkpoint model (the
    // architecture reserves bulk/GDS-style paths for checkpoints, never the WAL): the lanes
    // checkpoint (`checkpoint_intent_lanes`) additionally writes a sibling COLD ARTIFACT —
    // `<base>.cold-checkpoint.<cut>` — holding each qualifying table's chunk payload bytes +
    // descriptors, stamped with the artifact BOUNDARY (the commit index whose store state the
    // chunks reflect). Recovery (`open_lanes_durable_wal_segment`) restores the artifact at the
    // SEAM between checkpoint-records replay and lane-suffix replay: at that point the rebuilt
    // store IS the boundary state (strict equality `boundary == committed_seq()` is verified, any
    // mismatch is a benign skip — the cache rebuilds on first read), so the restored entries pin
    // the current generation Arc and the WAL SUFFIX IS THE DELTA STREAM — each replayed suffix
    // record patches them forward through the existing 6c-1 patcher via the 6c-3 commit hooks.
    // The durable validity token is therefore (artifact boundary == replay commit index); the
    // process-local generation-Arc validity takes over from the seam exactly as a live build.
    //
    // WHAT THE SAFETY ACTUALLY IS (audit MEDIUM, adopted): once a restored entry INSTALLS, a hit
    // replays its bytes verbatim — the store is not re-consulted — so an installed restore is
    // exactly as load-bearing as a live build's install. The guarantees carrying that trust are
    // (a) strict boundary equality with the seam's committed_seq, (b) the column-signature guard,
    // and (c) RECOVERY DETERMINISM: replaying the same records reproduces the same visible set
    // and the same TupleId space (the engine's standing recovery invariant — by-key records
    // re-resolve deterministically, the seam asserts the seq space). Every guard failure
    // degrades to a benign skip and the first read rebuilds from the store.
    //
    // CHARTER: durability/staging/IO is control-plane work (CHARTER.md: WAL, checkpointing and
    // recovery orchestration are host-owned); the artifact stores DEVICE-format bytes produced by
    // the existing build machinery — no host relational computation is introduced. The artifact
    // is the cold tier's durable form in P1 beside the still-authoritative store, and becomes the
    // PRIMARY cold representation when P4 deletes the store for streamed tables.

    /// Write the cold-tier checkpoint artifact beside the lanes checkpoint. `boundary_index` is
    /// the RECOVERY-SEAM value stamped into the artifact — the INCLUSIVE index of the
    /// checkpoint's last record, which is what the seam's `committed_seq()` reaches (replay
    /// publishes each record's own index). The LIVE watermark is accepted in EITHER convention
    /// (audit HIGH): `boundary_index` (serial/replay-derived engines) or `frontier_index` (the
    /// lane pump's exclusive `visible_global_cut = base_seq + cut`) — both prove every stamp is
    /// <= `boundary_index` at a quiesced cut, so a generation-current entry's content IS the
    /// boundary state. A table qualifies when its entry's pinned generation is ptr-equal current
    /// (untouched since its settled install), or the 6c-1 patcher brings it current at the
    /// observed watermark (the patch install re-proves settledness; a concurrent commit fails
    /// it — a safe exclusion). A watermark moved by a racing commit after qualification ABORTS
    /// the artifact (post-loop re-check: a fresh entry installed mid-loop could otherwise embed
    /// a stamp above the boundary). NOTE: the patch arm rides the REGISTERED 6c-1 scan-build
    /// staging debt (host visibility decode; deletion trigger = P4) — the checkpoint adds no new
    /// host relational computation. Returns the number of tables written.
    pub(crate) fn write_streaming_cold_checkpoint(
        &self,
        base: &std::path::Path,
        cut: u64,
        boundary_index: Index,
        frontier_index: Index,
    ) -> Result<usize, EngineError> {
        use std::io::Write;
        let path = streaming_cold_checkpoint_path(base, cut);
        let watermark = self.committed_seq();
        if watermark != boundary_index && watermark != frontier_index {
            // Not quiesced at the cut (a commit raced the checkpoint): skip — the artifact would
            // never match the recovery seam. Stale artifacts from older cuts still get swept.
            remove_stale_cold_checkpoints(base, cut);
            return Ok(0);
        }
        let map = self.read_state.residency.streaming_cold_chunks.load();
        let mut qualified: Vec<(String, Arc<ColdTableChunks>)> = Vec::new();
        for (name, entry) in map.iter() {
            let current = self.read_state.mvcc.table_rows(name).generation_payload();
            let entry = if Arc::ptr_eq(&entry.generation, &current) {
                // Untouched since its settled install: every live stamp is <= boundary (the
                // watermark guard above), so its content is the boundary state whatever
                // watermark convention its own build pinned.
                Arc::clone(entry)
            } else {
                // Written since its build: bring it current through the 6c-1 patcher (also lands
                // in the live map). A failed patch (ALTER, install race, IO) excludes the table.
                let Some(table) = self
                    .catalog_snapshot()
                    .relational_catalog
                    .get(name)
                    .cloned()
                else {
                    continue;
                };
                match self.patch_streaming_cold(
                    name,
                    &table,
                    entry,
                    &current,
                    watermark,
                    entry.chunk_target_bytes,
                    false,
                ) {
                    Some(patched) => patched,
                    None => continue,
                }
            };
            qualified.push((name.clone(), entry));
        }
        if self.committed_seq() != watermark {
            // A commit landed DURING qualification: a mid-loop fresh install could have embedded
            // a stamp above the boundary — abort this artifact (a later checkpoint retries).
            remove_stale_cold_checkpoints(base, cut);
            return Ok(0);
        }
        if qualified.is_empty() {
            remove_stale_cold_checkpoints(base, cut);
            return Ok(0);
        }
        // Stream-encode to a temp sibling, fsync, then atomically rename into place (the artifact
        // is all-or-nothing; a crash mid-write leaves the old artifact or none). Payloads stream
        // chunk-at-a-time (a spilled table's bytes never accumulate in host RAM here).
        let tmp = streaming_cold_checkpoint_tmp_path(base);
        let file = std::fs::File::create(&tmp).map_err(|e| {
            EngineError::Durability(format!(
                "cold checkpoint: create {} failed: {e}",
                tmp.display()
            ))
        })?;
        let mut w = ColdCkptWriter {
            inner: std::io::BufWriter::new(file),
            hash: FNV_OFFSET,
        };
        let write_all = (|| -> std::io::Result<()> {
            w.put(COLD_CHECKPOINT_MAGIC)?;
            w.put_u64(boundary_index)?;
            w.put_u32(qualified.len() as u32)?;
            for (name, entry) in &qualified {
                w.put_str(name)?;
                w.put_u16(entry.column_signature.len() as u16)?;
                for (col, ty) in &entry.column_signature {
                    w.put_str(col)?;
                    w.put_sql_type(*ty)?;
                }
                w.put_u64(entry.chunk_target_bytes)?;
                w.put_u32(entry.chunks.len() as u32)?;
                for chunk in &entry.chunks {
                    w.put_u64(chunk.row_count)?;
                    w.put_u64(chunk.tuple_range.0)?;
                    w.put_u64(chunk.tuple_range.1)?;
                    // v2 (P2b): the payload's OWN boundary + the optional tombstone sidecar —
                    // stamped rows must stay masked across a restart.
                    w.put_u64(chunk.payload_copin_s)?;
                    match &chunk.deleted_by {
                        None => w.put_u8(0)?,
                        Some(sidecar) => {
                            w.put_u8(1)?;
                            w.put_u64(sidecar.len() as u64)?;
                            w.put(sidecar)?;
                        }
                    }
                    encode_cold_descriptor(&mut w, &chunk.snapshot)?;
                    let payload = chunk.payload.read().map_err(|_| {
                        std::io::Error::other("cold checkpoint: spill payload read failed")
                    })?;
                    w.put_u64(payload.len() as u64)?;
                    w.put(&payload)?;
                }
            }
            let hash = w.hash;
            w.inner.write_all(&hash.to_le_bytes())?;
            w.inner.flush()?;
            w.inner.get_ref().sync_all()
        })();
        if let Err(e) = write_all {
            let _ = std::fs::remove_file(&tmp);
            return Err(EngineError::Durability(format!(
                "cold checkpoint: writing {} failed: {e}",
                tmp.display()
            )));
        }
        std::fs::rename(&tmp, &path).map_err(|e| {
            EngineError::Durability(format!(
                "cold checkpoint: rename into {} failed: {e}",
                path.display()
            ))
        })?;
        if let Some(parent) = path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        remove_stale_cold_checkpoints(base, cut);
        self.read_state
            .residency
            .streaming_cold_checkpointed
            .fetch_add(qualified.len() as u64, Ordering::Relaxed);
        Ok(qualified.len())
    }

    /// Restore the cold-tier checkpoint artifact at the recovery SEAM (checkpoint records
    /// replayed, lane suffix not yet). Every GUARD FAILURE degrades to a benign skip (the cache
    /// rebuilds on first read): missing/corrupt artifact, boundary != the seam's
    /// `committed_seq()`, a table gone from the catalog, or a column-signature mismatch. An
    /// entry that DOES install is trusted like a live build (see the section note above:
    /// boundary equality + signature + recovery determinism carry it). Restored payloads route through the standard builder, so
    /// over-threshold tables spill exactly like a live capture. Returns tables restored.
    pub(crate) fn restore_streaming_cold_checkpoint(
        &self,
        base: &std::path::Path,
        cut: u64,
    ) -> usize {
        let path = streaming_cold_checkpoint_path(base, cut);
        let Ok(mut file) = std::fs::File::open(&path) else {
            return 0;
        };
        // PASS 1: verify the FNV trailer over the whole stream (bounded RAM), then re-read and
        // decode trusting the content. Startup-time sequential IO; two passes beat buffering a
        // possibly spill-class (over-RAM) artifact.
        if !cold_checkpoint_checksum_ok(&mut file) {
            eprintln!(
                "[gpu-db] cold checkpoint {} failed its checksum; skipping the cold-tier \
                 restore (streaming reads rebuild the cache)",
                path.display()
            );
            return 0;
        }
        use std::io::Seek;
        if file.seek(std::io::SeekFrom::Start(0)).is_err() {
            return 0;
        }
        let mut r = ColdCkptReader {
            inner: std::io::BufReader::new(file),
        };
        let mut restored = 0usize;
        let decode_all = (|| -> std::io::Result<()> {
            let mut magic = [0u8; COLD_CHECKPOINT_MAGIC.len()];
            r.take(&mut magic)?;
            if &magic != COLD_CHECKPOINT_MAGIC {
                return Ok(());
            }
            let boundary = r.take_u64()?;
            if boundary != self.committed_seq() {
                // Not this replay's seam state (e.g. the WAL gained records the artifact predates
                // in a way that changed the ordinal) — never install; the cache rebuilds.
                return Ok(());
            }
            let table_count = r.take_u32()?;
            for _ in 0..table_count {
                let name = r.take_str()?;
                let sig_len = r.take_u16()?;
                let mut signature: Vec<(String, SqlType)> = Vec::with_capacity(sig_len as usize);
                for _ in 0..sig_len {
                    let col = r.take_str()?;
                    let ty = r.take_sql_type()?;
                    signature.push((col, ty));
                }
                let chunk_target_bytes = r.take_u64()?;
                let chunk_count = r.take_u32()?;
                // The catalog + signature guards mirror the 6c-1 ALTER guard; a mismatching
                // table's bytes must still be CONSUMED to keep the stream aligned.
                let table = self
                    .catalog_snapshot()
                    .relational_catalog
                    .get(&name)
                    .cloned();
                let live_signature: Option<Vec<(String, SqlType)>> = table
                    .as_ref()
                    .map(|t| t.columns.iter().map(|c| (c.name.clone(), c.ty)).collect());
                let usable = live_signature.as_ref() == Some(&signature);
                let mut builder = usable.then(|| ColdCacheBuilder {
                    generation: self.read_state.mvcc.table_rows(&name).generation_payload(),
                    build_copin_s: boundary,
                    chunk_target_bytes,
                    total_payload_bytes: 0,
                    column_signature: signature,
                    chunks: Vec::new(),
                    spill: None,
                    poisoned: false,
                });
                for _ in 0..chunk_count {
                    let row_count = r.take_u64()?;
                    let lo = r.take_u64()?;
                    let hi = r.take_u64()?;
                    // v2 (P2b): the persisted payload boundary + tombstone sidecar.
                    let payload_copin_s = r.take_u64()?;
                    let deleted_by = match r.take_u8()? {
                        0 => None,
                        _ => {
                            let len = r.take_u64()? as usize;
                            if len != (row_count as usize) * 8 {
                                return Err(std::io::Error::other(
                                    "cold checkpoint: sidecar length != 8 * row_count",
                                ));
                            }
                            let mut sidecar = vec![0u8; len];
                            r.take(&mut sidecar)?;
                            Some(Arc::new(sidecar))
                        }
                    };
                    let snapshot = decode_cold_descriptor(&mut r)?;
                    let payload_len = r.take_u64()? as usize;
                    let mut payload = vec![0u8; payload_len];
                    r.take(&mut payload)?;
                    if let Some(builder) = builder.as_mut() {
                        builder.push(payload, snapshot, row_count, (lo, hi));
                        // push() built the chunk with fresh-scan defaults; restore the persisted
                        // identity. The sidecar's bytes are ADDED to the builder total here
                        // (audit LOW: install copies the builder total verbatim — no
                        // recomputation — and the live patch path counts sidecars, so the cap
                        // class must see them on the restore path too).
                        if let Some(sidecar) = &deleted_by {
                            builder.total_payload_bytes += sidecar.len() as u64;
                        }
                        if let Some(chunk) = builder.chunks.last_mut() {
                            chunk.payload_copin_s = payload_copin_s;
                            chunk.deleted_by = deleted_by;
                        }
                    }
                }
                if let Some(builder) = builder {
                    // is_patch=true: a restore is not a fresh scan-build (keeps the builds
                    // counter the out-of-core cost signal it is). The install re-proves the
                    // settled boundary (strict equality — trivially true at the single-threaded
                    // seam) and applies the standard cap policy.
                    if !builder.poisoned
                        && self.install_streaming_cold_inner(&name, builder, true, false)
                    {
                        restored += 1;
                    }
                }
            }
            Ok(())
        })();
        if decode_all.is_err() {
            // Truncated/torn beyond the verified trailer (should be impossible) — keep whatever
            // installed cleanly; later tables just rebuild.
            eprintln!(
                "[gpu-db] cold checkpoint {} ended mid-decode; restored {restored} table(s)",
                path.display()
            );
        }
        if restored > 0 {
            self.read_state
                .residency
                .streaming_cold_restored
                .fetch_add(restored as u64, Ordering::Relaxed);
        }
        restored
    }
}

// ---------------- P4-1: THE REVERSE GATHER (chunk bytes -> host rows) ----------------
//
// The DE-AUTHORITIZATION primitive of the chunk-authoritative class (PLAN §2 S-E.P4): host-decode a
// cold chunk's DEVICE-FORMAT payload back into catalog-order rows — the exit that rebuilds a store
// representation for any shape the streaming folds cannot serve (over-budget unbounded ORDER BY,
// JOINs, windows), for DDL preflights, and for patch failures. GREENFIELD BY NECESSITY (design
// review C2): the elision rehydrate reads resident device SHARDS, which an over-budget streamed
// table does not have — its only representation is these host-side chunk bytes.
//
// CHARTER / LEDGER (design review M4): this is a HOST MATERIALIZATION — registered debt, control
// plane only (a de-auth/import action, never the steady-state data path; the live path stays
// device-executed). Deletion trigger: the device-index-over-chunks route that lifts the class's
// no-uniqueness restriction (the locate/gather then runs on-device end to end).

/// Decode ONE cold chunk's payload into rows visible at `rtx`, honoring the deleted_by sidecar
/// with EXACTLY the kernel's semantics (visible ⟺ `deleted_by > rtx`; absent sidecar = all live)
/// and each column's NULL validity bitmap. Layout authority = the chunk's own DESCRIPTOR (the
/// self-describing text/bool/null offsets + the capacity-derived fixed-width section formulas —
/// the same helpers the device readers use).
#[allow(dead_code)] // P4-2b/P4-3 wire the production callers; the round-trip gate exercises it now.
pub(crate) fn decode_cold_chunk_rows(
    table: &RelationalTable,
    chunk: &ColdChunk,
    rtx: Index,
) -> Result<Vec<Vec<SqlValue>>, EngineError> {
    use crate::relational_model::{
        resident_device_bool_column_offset, resident_device_int4_column_offset,
        resident_device_int8_column_offset, resident_device_numeric_column_offset,
        resident_device_text_column_layout,
    };
    let map_err = |what: &str| EngineError::ApplyFailed(format!("reverse gather: {what}"));
    let bytes = chunk
        .payload
        .read()
        .map_err(|_| map_err("chunk payload read failed"))?;
    let bytes: &[u8] = &bytes;
    let d = &chunk.snapshot;
    let rows_n = chunk.row_count as usize;
    let read_u32 = |off: usize| -> Result<u32, EngineError> {
        bytes
            .get(off..off + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
            .ok_or_else(|| map_err("payload truncated (u32)"))
    };
    let read_u64 = |off: usize| -> Result<u64, EngineError> {
        bytes
            .get(off..off + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
            .ok_or_else(|| map_err("payload truncated (u64)"))
    };
    // Per-column NULL validity: name -> bitmap offset (absent = all valid). 1 = valid, LSB-first
    // u32 words (doc 21).
    let null_offset = |name: &str| -> Option<u64> {
        d.resident_device_null_columns
            .iter()
            .find(|n| n.name == name)
            .map(|n| n.bitmap_byte_offset)
    };
    let bit_is_set = |bitmap_offset: u64, slot: usize| -> Result<bool, EngineError> {
        let word = read_u32(bitmap_offset as usize + (slot / 32) * 4)?;
        Ok((word >> (slot % 32)) & 1 == 1)
    };
    // The sidecar mask, kernel-identical: visible ⟺ deleted_by > rtx (0x7F.. live fill is a large
    // positive i64, always > any real boundary). A mis-sized sidecar is impossible under the P2b
    // sizing invariant (row_count*8, decode-validated) — but if it ever regresses, ERROR loudly
    // rather than silently resurrect a deleted row (audit LOW).
    let slot_visible = |slot: usize| -> Result<bool, EngineError> {
        match &chunk.deleted_by {
            None => Ok(true),
            Some(sidecar) => {
                let raw = sidecar
                    .get(slot * 8..slot * 8 + 8)
                    .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
                    .ok_or_else(|| map_err("sidecar shorter than row_count*8"))?;
                Ok(raw > rtx as i64)
            }
        }
    };

    let mut out: Vec<Vec<SqlValue>> = Vec::new();
    for slot in 0..rows_n {
        if !slot_visible(slot)? {
            continue;
        }
        let mut row: Vec<SqlValue> = Vec::with_capacity(table.columns.len());
        for (idx, column) in table.columns.iter().enumerate() {
            if let Some(offset) = null_offset(&column.name) {
                if !bit_is_set(offset, slot)? {
                    row.push(SqlValue::Null);
                    continue;
                }
            }
            let value = match column.ty {
                SqlType::Int4 | SqlType::Date | SqlType::Int2 => {
                    let base = resident_device_int4_column_offset(d, table, idx)
                        .map_err(|_| map_err("int4 offset"))?;
                    let v = read_u32(base as usize + slot * 4)? as i32;
                    match column.ty {
                        SqlType::Date => SqlValue::Date(v),
                        SqlType::Int2 => SqlValue::Int2(v as i16),
                        _ => SqlValue::Int4(v),
                    }
                }
                SqlType::Int8 | SqlType::Timestamp => {
                    let base = resident_device_int8_column_offset(d, table, idx)
                        .map_err(|_| map_err("int8 offset"))?;
                    let v = read_u64(base as usize + slot * 8)? as i64;
                    if column.ty == SqlType::Timestamp {
                        SqlValue::Timestamp(v)
                    } else {
                        SqlValue::Int8(v)
                    }
                }
                SqlType::Numeric { .. } | SqlType::Uuid => {
                    let base = resident_device_numeric_column_offset(d, table, idx)
                        .map_err(|_| map_err("b128 offset"))?;
                    let off = base as usize + slot * 16;
                    let raw: [u8; 16] = bytes
                        .get(off..off + 16)
                        .and_then(|b| b.try_into().ok())
                        .ok_or_else(|| map_err("payload truncated (b128)"))?;
                    if column.ty == SqlType::Uuid {
                        SqlValue::Uuid(raw)
                    } else {
                        let SqlType::Numeric { scale, .. } = column.ty else {
                            unreachable!()
                        };
                        SqlValue::Numeric(gpu_db_sql::Decimal128::new(
                            i128::from_le_bytes(raw),
                            scale,
                        ))
                    }
                }
                SqlType::Bool => {
                    let base = resident_device_bool_column_offset(d, table, idx)
                        .map_err(|_| map_err("bool offset"))?;
                    SqlValue::Bool(bit_is_set(base, slot)?)
                }
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(d, table, idx)
                        .map_err(|_| map_err("text layout"))?;
                    let lo =
                        read_u64(layout.offsets_byte_offset as usize + slot * 8)? as usize;
                    let hi =
                        read_u64(layout.offsets_byte_offset as usize + (slot + 1) * 8)? as usize;
                    let span = bytes
                        .get(layout.bytes_byte_offset as usize + lo
                            ..layout.bytes_byte_offset as usize + hi)
                        .ok_or_else(|| map_err("payload truncated (text)"))?;
                    SqlValue::Text(
                        std::str::from_utf8(span)
                            .map_err(|_| map_err("text not UTF-8"))?
                            .to_string(),
                    )
                }
            };
            row.push(value);
        }
        out.push(row);
    }
    Ok(out)
}

// ---------------- P1: the cold-checkpoint artifact encoding (control plane, no serde) ----------------

/// Artifact magic — version-suffixed like the WAL magics (`GPUDBWAL1`); bump on layout change.
const COLD_CHECKPOINT_MAGIC: &[u8; 15] = b"GPUDBCOLDCKPT2\n";
pub(crate) const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// `<base>.cold-checkpoint.<cut>` beside the WAL — mirrors the lanes-checkpoint naming
/// (`<base>.lanes-checkpoint.seg.<cut>`), keyed by the SAME cut so recovery pairs them.
fn streaming_cold_checkpoint_path(base: &std::path::Path, cut: u64) -> std::path::PathBuf {
    let name = base
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    base.with_file_name(format!("{name}.cold-checkpoint.{cut}"))
}

fn streaming_cold_checkpoint_tmp_path(base: &std::path::Path) -> std::path::PathBuf {
    let name = base
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    base.with_file_name(format!("{name}.cold-checkpoint.tmp"))
}

/// Sweep artifacts for OTHER cuts (and the tmp) — the previous checkpoint generation's artifact
/// is dead once a newer cut committed (its seam can never be replayed again).
pub(crate) fn remove_stale_cold_checkpoints(base: &std::path::Path, keep_cut: u64) {
    let Some(parent) = base.parent() else { return };
    let Some(stem) = base.file_name() else { return };
    let prefix = format!("{}.cold-checkpoint.", stem.to_string_lossy());
    let keep = keep_cut.to_string();
    let Ok(entries) = std::fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(suffix) = name.strip_prefix(&prefix) {
            if suffix != keep {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

/// FNV-1a-hashing writer: every byte written folds into the running trailer checksum.
pub(crate) struct ColdCkptWriter<W: std::io::Write> {
    pub(crate) inner: W,
    pub(crate) hash: u64,
}

impl<W: std::io::Write> ColdCkptWriter<W> {
    fn put(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        for b in bytes {
            self.hash = (self.hash ^ u64::from(*b)).wrapping_mul(FNV_PRIME);
        }
        self.inner.write_all(bytes)
    }
    fn put_u8(&mut self, v: u8) -> std::io::Result<()> {
        self.put(&[v])
    }
    fn put_u16(&mut self, v: u16) -> std::io::Result<()> {
        self.put(&v.to_le_bytes())
    }
    fn put_u32(&mut self, v: u32) -> std::io::Result<()> {
        self.put(&v.to_le_bytes())
    }
    fn put_u64(&mut self, v: u64) -> std::io::Result<()> {
        self.put(&v.to_le_bytes())
    }
    fn put_i32(&mut self, v: i32) -> std::io::Result<()> {
        self.put(&v.to_le_bytes())
    }
    fn put_str(&mut self, s: &str) -> std::io::Result<()> {
        let bytes = s.as_bytes();
        if bytes.len() > u16::MAX as usize {
            return Err(std::io::Error::other("cold checkpoint: string too long"));
        }
        self.put_u16(bytes.len() as u16)?;
        self.put(bytes)
    }
    fn put_sql_type(&mut self, ty: SqlType) -> std::io::Result<()> {
        match ty {
            SqlType::Int2 => self.put_u8(0),
            SqlType::Int4 => self.put_u8(1),
            SqlType::Int8 => self.put_u8(2),
            SqlType::Numeric { precision, scale } => {
                self.put_u8(3)?;
                self.put_u8(precision)?;
                self.put_u8(scale)
            }
            SqlType::Bool => self.put_u8(4),
            SqlType::Text => self.put_u8(5),
            SqlType::Date => self.put_u8(6),
            SqlType::Timestamp => self.put_u8(7),
            SqlType::Uuid => self.put_u8(8),
        }
    }
}

pub(crate) struct ColdCkptReader<R: std::io::Read> {
    pub(crate) inner: R,
}

impl<R: std::io::Read> ColdCkptReader<R> {
    fn take(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.inner.read_exact(buf)
    }
    fn take_u8(&mut self) -> std::io::Result<u8> {
        let mut b = [0u8; 1];
        self.take(&mut b)?;
        Ok(b[0])
    }
    fn take_u16(&mut self) -> std::io::Result<u16> {
        let mut b = [0u8; 2];
        self.take(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn take_u32(&mut self) -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        self.take(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn take_u64(&mut self) -> std::io::Result<u64> {
        let mut b = [0u8; 8];
        self.take(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn take_i32(&mut self) -> std::io::Result<i32> {
        let mut b = [0u8; 4];
        self.take(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }
    fn take_str(&mut self) -> std::io::Result<String> {
        let len = self.take_u16()? as usize;
        let mut bytes = vec![0u8; len];
        self.take(&mut bytes)?;
        String::from_utf8(bytes)
            .map_err(|_| std::io::Error::other("cold checkpoint: invalid UTF-8"))
    }
    fn take_sql_type(&mut self) -> std::io::Result<SqlType> {
        Ok(match self.take_u8()? {
            0 => SqlType::Int2,
            1 => SqlType::Int4,
            2 => SqlType::Int8,
            3 => SqlType::Numeric {
                precision: self.take_u8()?,
                scale: self.take_u8()?,
            },
            4 => SqlType::Bool,
            5 => SqlType::Text,
            6 => SqlType::Date,
            7 => SqlType::Timestamp,
            8 => SqlType::Uuid,
            _ => {
                return Err(std::io::Error::other(
                    "cold checkpoint: unknown SqlType tag",
                ))
            }
        })
    }
}

/// Verify the trailing FNV-1a checksum over everything before it. Streams in 64KiB blocks
/// (bounded RAM for spill-class artifacts).
fn cold_checkpoint_checksum_ok(file: &mut std::fs::File) -> bool {
    use std::io::{Read, Seek};
    let Ok(total) = file.seek(std::io::SeekFrom::End(0)) else {
        return false;
    };
    if total < 8 + COLD_CHECKPOINT_MAGIC.len() as u64 {
        return false;
    }
    if file.seek(std::io::SeekFrom::Start(0)).is_err() {
        return false;
    }
    let body = total - 8;
    let mut hash = FNV_OFFSET;
    let mut remaining = body;
    let mut buf = vec![0u8; 64 << 10];
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        if file.read_exact(&mut buf[..want]).is_err() {
            return false;
        }
        for b in &buf[..want] {
            hash = (hash ^ u64::from(*b)).wrapping_mul(FNV_PRIME);
        }
        remaining -= want as u64;
    }
    let mut trailer = [0u8; 8];
    if file.read_exact(&mut trailer).is_err() {
        return false;
    }
    hash == u64::from_le_bytes(trailer)
}

/// Serialize the chunk DESCRIPTOR ([`RelationalResidencySnapshot`]) — the device-layout contract
/// the replay staging clones (a fresh memory proof is stamped per upload, so the proof and the
/// refresh/invalidation bookkeeping are NOT persisted; decode restores them to their transient
/// defaults, exactly what `build_transient_relation_payload_only` produces).
pub(crate) fn encode_cold_descriptor<W: std::io::Write>(
    w: &mut ColdCkptWriter<W>,
    d: &RelationalResidencySnapshot,
) -> std::io::Result<()> {
    w.put_u16(d.gpu_id)?;
    w.put_str(&d.schema)?;
    w.put_str(&d.table)?;
    w.put_u64(d.generation)?;
    w.put_u64(d.row_count as u64)?;
    w.put_u64(d.capacity as u64)?;
    w.put_u64(d.column_count as u64)?;
    w.put_u64(d.resident_bytes)?;
    w.put_u32(d.resident_device_int4_columns.len() as u32)?;
    for name in &d.resident_device_int4_columns {
        w.put_str(name)?;
    }
    w.put_u32(d.resident_device_int4_column_stats.len() as u32)?;
    for s in &d.resident_device_int4_column_stats {
        w.put_str(&s.name)?;
        w.put_i32(s.min)?;
        w.put_i32(s.max)?;
    }
    w.put_u32(d.resident_device_int8_columns.len() as u32)?;
    for name in &d.resident_device_int8_columns {
        w.put_str(name)?;
    }
    w.put_u32(d.resident_device_numeric_columns.len() as u32)?;
    for name in &d.resident_device_numeric_columns {
        w.put_str(name)?;
    }
    w.put_u32(d.resident_device_bool_columns.len() as u32)?;
    for b in &d.resident_device_bool_columns {
        w.put_str(&b.name)?;
        w.put_u64(b.bitmap_byte_offset)?;
    }
    w.put_u32(d.resident_device_text_columns.len() as u32)?;
    for t in &d.resident_device_text_columns {
        w.put_str(&t.name)?;
        w.put_u64(t.offsets_byte_offset)?;
        w.put_u64(t.bytes_byte_offset)?;
        w.put_u64(t.bytes_len)?;
    }
    w.put_u32(d.resident_device_null_columns.len() as u32)?;
    for n in &d.resident_device_null_columns {
        w.put_str(&n.name)?;
        w.put_u64(n.bitmap_byte_offset)?;
    }
    w.put_u64(d.valid_through_index)
}

pub(crate) fn decode_cold_descriptor<R: std::io::Read>(
    r: &mut ColdCkptReader<R>,
) -> std::io::Result<RelationalResidencySnapshot> {
    let gpu_id = r.take_u16()?;
    let schema = r.take_str()?;
    let table = r.take_str()?;
    let generation = r.take_u64()?;
    let row_count = r.take_u64()? as usize;
    let capacity = r.take_u64()? as usize;
    let column_count = r.take_u64()? as usize;
    let resident_bytes = r.take_u64()?;
    let mut resident_device_int4_columns = Vec::new();
    for _ in 0..r.take_u32()? {
        resident_device_int4_columns.push(r.take_str()?);
    }
    let mut resident_device_int4_column_stats = Vec::new();
    for _ in 0..r.take_u32()? {
        resident_device_int4_column_stats.push(ResidentDeviceInt4ColumnStats {
            name: r.take_str()?,
            min: r.take_i32()?,
            max: r.take_i32()?,
        });
    }
    let mut resident_device_int8_columns = Vec::new();
    for _ in 0..r.take_u32()? {
        resident_device_int8_columns.push(r.take_str()?);
    }
    let mut resident_device_numeric_columns = Vec::new();
    for _ in 0..r.take_u32()? {
        resident_device_numeric_columns.push(r.take_str()?);
    }
    let mut resident_device_bool_columns = Vec::new();
    for _ in 0..r.take_u32()? {
        resident_device_bool_columns.push(ResidentDeviceBoolColumnLayout {
            name: r.take_str()?,
            bitmap_byte_offset: r.take_u64()?,
        });
    }
    let mut resident_device_text_columns = Vec::new();
    for _ in 0..r.take_u32()? {
        resident_device_text_columns.push(ResidentDeviceTextColumnLayout {
            name: r.take_str()?,
            offsets_byte_offset: r.take_u64()?,
            bytes_byte_offset: r.take_u64()?,
            bytes_len: r.take_u64()?,
        });
    }
    let mut resident_device_null_columns = Vec::new();
    for _ in 0..r.take_u32()? {
        resident_device_null_columns.push(ResidentDeviceNullBitmapLayout {
            name: r.take_str()?,
            bitmap_byte_offset: r.take_u64()?,
        });
    }
    let valid_through_index = r.take_u64()?;
    Ok(RelationalResidencySnapshot {
        gpu_id,
        schema,
        table,
        generation,
        row_count,
        capacity,
        column_count,
        resident_bytes,
        resident_device_int4_columns,
        resident_device_int4_column_stats,
        resident_device_int8_columns,
        resident_device_numeric_columns,
        resident_device_bool_columns,
        resident_device_text_columns,
        resident_device_null_columns,
        valid_through_index,
        invalidated_by_txn_id: None,
        invalidated_at_index: None,
        invalidated_by_memory_pressure: false,
        memory_pressure_active: false,
        last_refresh_cost: None,
        admission_budget_bytes: None,
        resident_bytes_after_admission: resident_bytes,
        evicted_tables_on_admission: Vec::new(),
        device_memory_proof: None,
    })
}
