//! STRATA S-E.1 — the streaming out-of-core executor (ADR-012 / PLAN S-E).
//!
//! A query whose working set exceeds the GPU byte budget must still run ON THE DEVICE: the charter
//! forbids the host from being an execution tier, and ADR-006 deletes the CPU relational engine. The
//! streaming executor closes the over-VRAM gap with byte-bounded scalar reductions, filter/project,
//! GROUP/DISTINCT, ordered top-N, device window functions, and two-/N-way INNER/OUTER joins:
//!
//!   admit chunk -> push the reduction down (device kernel) -> combine the partial -> evict chunk -> next
//!
//! S-E.5 lookahead overlaps at most TWO transient chunks (one computing, one staged), each targeted at
//! half the query budget, so a single GPU can serve a relation far larger than its VRAM.
//! Host RAM is the cold STORAGE tier (the MVCC store the chunk bytes are staged from); the GPU is the SOLE
//! execution tier — WHERE, each reduction, and the final associative partial combine run on the device.
//! The host does only what the charter permits: MVCC visibility resolution, staging/upload orchestration,
//! and cardinality flow control — never relational value combination.
//!
//! Scalar folds reuse the whole predicate + aggregate executor. Relational operators retain opaque
//! device masks, coordinates, materialized runs, hash chains, rank lanes, and partial accumulators;
//! only final result framing crosses D2H. Activation gates on a CONFIGURED per-GPU
//! residency budget (the operator's VRAM-management signal); with no budget there is no notion of
//! "over-VRAM" and the read stays on the interim host path — so default behavior is byte-identical.
//!
//! **S-E.2 (here too): streaming filter/project.** A plain `All`/`Columns` projection folds by CONCAT
//! (the ARCHITECTURE §13 projection combine): each chunk's device-filtered + device-gathered survivors
//! append to the result, with LIMIT/OFFSET applied as cross-chunk windowing of the survivor stream (the
//! executor's own "LIMIT/OFFSET as control-plane WINDOWING" precedent) and a satisfied LIMIT stopping the
//! scan early — the table tail is never staged.
//!
//! Byte-replay, bounded LRU/spill eviction, async copy/compute lookahead, keyed chunk admission,
//! and round-robin multi-GPU scalar/group partial execution are built. Remaining follow-ons are
//! tracked in PLAN S-E rather than implied by this module header.

use super::*;

/// Candidate chunk positions plus the optional retained device index owner that proves their
/// generation remains live for the caller.
type ChunkKeyCandidates = (
    Vec<Vec<usize>>,
    Option<Arc<gpu_db_execution::CudaResidentDeviceMemory>>,
);

use crate::engine_expr::{
    grouped_projection_to_aggregates, resident_predicate_from_bound_filters, ResidentExecSource,
    ResidentExpr,
};
use crate::rel_exec_helpers::{
    bind_relational_select, catalog_relation_table, decode_relational_row, relational_key_prefix,
    relational_row_key,
};
use std::sync::atomic::Ordering;

mod materialized_column_decode;
mod materialized_join_run;
mod streaming_cold_admission;
mod streaming_grouped_fold;
mod streaming_join;
mod streaming_ordered_fold;
mod streaming_projection_fold;
mod streaming_reduction_fold;
mod streaming_select_route;

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
pub(crate) struct StagedChunk {
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
    pub(crate) fn ready(
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
    /// P4-2b-ii — THE COORDINATE TOKEN (the P4-2a obligation): a process-monotonic install epoch.
    /// (chunk_idx, slot) coordinates located against epoch E are valid ONLY while the installed
    /// entry still carries E — any re-install (tail append, stamp, patch) bumps it, and a stale
    /// token falls back to de-authoritization instead of mis-stamping re-tiled chunks.
    pub(crate) entry_epoch: u64,
    pub(crate) chunks: Vec<ColdChunk>,
}

/// P4-2b-ii: a class DML resolve's matches — the standard (pseudo_id, row_key, image) triples.
pub(crate) type ClassDmlMatches = Vec<(u64, String, Vec<SqlValue>)>;

/// P5-1: one chunk's retained device key index (fingerprint hash table).
// Production callers arrive with P5-2 (the uniqueness probe); the gate exercises it now.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct ChunkKeyIndex {
    pub(crate) device: Arc<gpu_db_execution::CudaResidentDeviceMemory>,
    pub(crate) table_mask: u32,
    pub(crate) hash_shift: u32,
    pub(crate) row_count: u32,
    pub(crate) bytes: u64,
    pub(crate) last_used: u64,
}

/// P5-later: compact candidate filter retained for every chunk when the full index set exceeds its cap.
#[derive(Clone, Debug)]
pub(crate) struct ChunkKeyBloom {
    pub(crate) device: Arc<gpu_db_execution::CudaResidentDeviceMemory>,
    pub(crate) bit_mask: u32,
    pub(crate) bytes: u64,
}

/// P5-1: the retained chunk-index VRAM cap (explicit accounting — the shard twin has none).
#[allow(dead_code)] // P5-2 wires the production path.
const CHUNK_KEY_INDEX_CAP_BYTES: u64 = 256 * 1024 * 1024;
const CHUNK_KEY_BLOOM_CAP_BYTES: u64 = 64 * 1024 * 1024;
#[cfg(test)]
pub(crate) static CHUNK_KEY_INDEX_CAP_BYTES_TEST: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn chunk_key_index_cap_bytes() -> u64 {
    #[cfg(test)]
    {
        let forced = CHUNK_KEY_INDEX_CAP_BYTES_TEST.load(Ordering::Relaxed);
        if forced > 0 {
            return forced;
        }
    }
    CHUNK_KEY_INDEX_CAP_BYTES
}

#[cfg(test)]
pub(crate) static CHUNK_KEY_BLOOM_CAP_BYTES_TEST: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
pub(crate) static CHUNK_KEY_BLOOM_ALL_POSITIVE_TEST: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn chunk_key_bloom_cap_bytes() -> u64 {
    #[cfg(test)]
    {
        let forced = CHUNK_KEY_BLOOM_CAP_BYTES_TEST.load(Ordering::Relaxed);
        if forced > 0 {
            return forced;
        }
    }
    CHUNK_KEY_BLOOM_CAP_BYTES
}
/// Until the device tuple-hash/group operator lands, exact within-statement uniqueness is a
/// bounded set of device predicate passes. The bound prevents an unscalable O(B^2) hot path;
/// larger SQL batches take the existing conservative de-authorize path. Normal OLTP waves are
/// single-row/small-batch. Deletion trigger: the device exact tuple-hash/group operator.
const CLASS_DEVICE_UNIQUE_BATCH_MAX_ROWS: usize = 256;

#[cfg(test)]
type ClassResolvePinHook = (
    Arc<std::sync::Barrier>,
    Arc<std::sync::Barrier>,
);

#[cfg(test)]
fn class_resolve_pin_hook() -> &'static std::sync::Mutex<Option<ClassResolvePinHook>> {
    static HOOK: std::sync::OnceLock<std::sync::Mutex<Option<ClassResolvePinHook>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(test)]
pub(crate) fn install_class_resolve_pin_hook() -> ClassResolvePinHook {
    let pinned = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    *class_resolve_pin_hook()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
        Some((Arc::clone(&pinned), Arc::clone(&resume)));
    (pinned, resume)
}

#[cfg(test)]
type ChunkKeyPrimePinHook = (
    Arc<std::sync::Barrier>,
    Arc<std::sync::Barrier>,
);

#[cfg(test)]
fn chunk_key_prime_pin_hook() -> &'static std::sync::Mutex<Option<ChunkKeyPrimePinHook>> {
    static HOOK: std::sync::OnceLock<std::sync::Mutex<Option<ChunkKeyPrimePinHook>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(test)]
pub(crate) fn install_chunk_key_prime_pin_hook() -> ChunkKeyPrimePinHook {
    let pinned = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    *chunk_key_prime_pin_hook()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
        Some((Arc::clone(&pinned), Arc::clone(&resume)));
    (pinned, resume)
}

/// P5-1: the chunk CONTENT-identity allocator (fresh iff the payload bytes are new).
static COLD_CHUNK_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// P4-2b-ii: the global entry-epoch allocator (never reused; process-local like the class map).
static COLD_ENTRY_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// One cached chunk: the exact device payload bytes + the descriptor template the build produced.
pub(crate) struct ColdChunk {
    payload: ColdPayload,
    pub(crate) snapshot: RelationalResidencySnapshot,
    pub(crate) row_count: u64,
    /// P5-1 — CONTENT IDENTITY (design review H3): a process-monotonic id allocated ONLY where
    /// the payload bytes are GENUINELY NEW (the builder's push, the tail constructor, the
    /// compaction survivor build) and PRESERVED by every verbatim clone. The chunk-index cache
    /// keys on it: a rebuilt payload inheriting its source's id would serve the OLD index over
    /// NEW bytes (probe-slot misalignment = a silent false-negative duplicate). COEXISTS with
    /// positional chunk_idx + entry_epoch (the stamp token) — never conflate; never cache a
    /// chunk_idx across entry Arcs.
    pub(crate) chunk_id: u64,
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
    pub(crate) payload_copin_s: Index,
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
            chunk_id: COLD_CHUNK_ID.fetch_add(1, Ordering::Relaxed),
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

/// A device reduction error that is a genuine arithmetic OVERFLOW (matched on the executor's stable PG
/// overflow phrases). Such an error must surface, not defer to the CPU path (audit Finding 2).
fn is_overflow_error(err: &ExecuteError) -> bool {
    let message = err.to_string();
    message.contains("overflow") || message.contains("out of range")
}

impl Engine {
    /// GPUs eligible to execute one query's byte-bounded chunks. The default GPU leads for stable
    /// single-device behavior; additional physical devices participate only when they have an equal
    /// or larger effective query budget and are not unavailable/memory-pressured. Cross-device
    /// traffic therefore consists only of the small associative partials returned to the final
    /// default-GPU combine—never full relation payloads or host relational execution.
    fn streaming_execution_gpus(&self, default_gpu: u16, query_budget: u64) -> Vec<u16> {
        let hardware = self.cuda_driver_probe_runtime().snapshot();
        let health = self.router.runtime().snapshot();
        let physical: Vec<u16> = hardware.devices.iter().map(|device| device.id).collect();
        // Use EFFECTIVE budgets, not only the explicit lock-free overrides. In production an
        // unconfigured device receives the 80%-of-physical-memory default; consulting only the
        // override map made every secondary GPU silently ineligible in the default configuration.
        let budgets: std::collections::BTreeMap<u16, u64> = physical
            .iter()
            .filter_map(|&gpu_id| {
                self.relational_residency_budget_bytes(gpu_id)
                    .map(|budget| (gpu_id, budget))
            })
            .collect();
        select_streaming_execution_gpus(
            default_gpu,
            query_budget,
            &physical,
            &health.unavailable_gpu_ids,
            &health.memory_pressured_gpu_ids,
            &budgets,
        )
    }

    fn record_streaming_chunk_gpu(&self, source: &ResidentExecSource, coordinator_gpu: u16) {
        if source.descriptor.gpu_id != coordinator_gpu {
            self.read_state
                .residency
                .streaming_fold_secondary_gpu_chunks
                .fetch_add(1, Ordering::Relaxed);
        }
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
        gpu_id: u16,
    ) -> Result<StagedChunk, ()> {
        let (snapshot, pending, payload) = self
            .build_transient_relation_residency_async(table, chunk_rows, gpu_id)
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
            self.purge_chunk_key_indexes_for_table(table_name);
            self.purge_chunk_key_blooms_for_table(table_name);
        }
    }

    /// S-E.6: stage one COLD chunk — re-upload the cached device payload bytes (async copy stream),
    /// with a fresh proof stamped onto the cached descriptor template. No decode, no assembly.
    pub(crate) fn stage_cold_chunk(
        &self,
        chunk: &ColdChunk,
        reader_copin_s: Index,
    ) -> Result<StagedChunk, ()> {
        self.stage_cold_chunk_on_gpu(chunk, reader_copin_s, chunk.snapshot.gpu_id)
    }

    fn stage_cold_chunk_on_gpu(
        &self,
        chunk: &ColdChunk,
        reader_copin_s: Index,
        gpu_id: u16,
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
            .retain_device_memory_copy_async(gpu_id, &bytes)
            .map_err(|_| ())?;
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(chunk.snapshot.resident_bytes, Ordering::Relaxed);
        let mut snapshot = chunk.snapshot.clone();
        snapshot.gpu_id = gpu_id;
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
                    chunk_id: chunk.chunk_id,
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
        if cold.chunk_target_bytes != chunk_target_bytes {
            return None;
        }
        // P4-3 — THE BORN GATE (design review C3): a CLASS table's entry boundary advances with
        // every tail append, so `copin_s >= build` would MISS any reader pinned below the latest
        // write and thrash de-auth. Class hits require only `copin_s >= the FREEZE boundary`
        // (below it the frozen chains serve exactly); the replay arms skip chunks BORN LATER
        // (`payload_copin_s > copin_s`) and the sidecar mask handles deletes — exact MVCC per
        // reader. Non-class entries keep the strict boundary rule (their chunks are rebuilt at
        // the entry boundary; no per-chunk born discipline exists for them).
        match self.table_chunk_authoritative(table_name) {
            Some(freeze) => {
                if copin_s < freeze {
                    return None;
                }
            }
            None => {
                if copin_s < cold.build_copin_s {
                    return None;
                }
            }
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
            entry_epoch: COLD_ENTRY_EPOCH.fetch_add(1, Ordering::Relaxed),
            chunks: builder.chunks,
        });
        let live_chunk_ids: std::collections::BTreeSet<u64> =
            entry.chunks.iter().map(|chunk| chunk.chunk_id).collect();
        let residency = &self.read_state.residency;
        let _publish = residency
            .streaming_cold_lock
            .lock()
            .expect("streaming cold-tier lock poisoned");
        let mut map = std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
        map.insert(table_name.to_string(), Arc::clone(&entry));
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
        drop(_publish);
        self.purge_stale_chunk_key_candidates(table_name, &live_chunk_ids);
        drop(_commit_guard);
        // Key candidate structures are primed only after releasing the global commit mutex. This
        // is load-bearing for spilled captures: staging may perform positional NVMe reads, which
        // must never occur in the later class-entry hook under the commit lock.
        if !commit_lock_held {
            self.prime_chunk_key_candidates(table_name, &entry);
        }
        true
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

    pub fn streaming_fold_secondary_gpu_chunks(&self) -> u64 {
        self.read_state
            .residency
            .streaming_fold_secondary_gpu_chunks
            .load(Ordering::Relaxed)
    }

    pub fn streaming_fold_peak_chunk_bytes(&self) -> u64 {
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .load(Ordering::Relaxed)
    }

    pub fn streaming_join_hits(&self) -> u64 {
        self.read_state
            .residency
            .streaming_join_hits
            .load(Ordering::Relaxed)
    }

    pub fn streaming_join_block_pairs(&self) -> u64 {
        self.read_state
            .residency
            .streaming_join_block_pairs
            .load(Ordering::Relaxed)
    }

    pub fn streaming_join_peak_device_bytes(&self) -> u64 {
        self.read_state
            .residency
            .streaming_join_peak_device_bytes
            .load(Ordering::Relaxed)
    }

    pub fn streaming_window_hits(&self) -> u64 {
        self.read_state
            .residency
            .streaming_window_hits
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
                self.stage_streaming_chunk(&locate_table, chunk_rows, (1, 0), &mut capture, gpu_id)?;
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
        // Audit LOW (P4-3): the same entry-level floor as the locate — a class gather below the
        // freeze would silently DROP freeze-rebuilt base chunks via the born skip; decline instead
        // (the frozen store serves those boundaries).
        match self.table_chunk_authoritative(table_name) {
            Some(freeze) => {
                if rtx < freeze {
                    return None;
                }
            }
            None => {
                if rtx < entry.build_copin_s {
                    return None;
                }
            }
        }
        let mut rows: Vec<Vec<SqlValue>> = Vec::new();
        for chunk in &entry.chunks {
            // P4-3 born gate: chunks born after `rtx` are invisible to that boundary.
            if chunk.payload_copin_s > rtx {
                continue;
            }
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
    #[cfg(test)]
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
        self.locate_streaming_cold_slots_in_entry(table, predicate, rtx, &entry, None)
    }

    /// P5 charter closure: run an EXACT predicate over only the chunks selected by the
    /// fingerprint index. The index is an addressing accelerator, never the relational
    /// authority: visibility, full-key equality (including collision resolution), residual
    /// predicates, and NULL/3VL all run through `lower_resident_predicate` on the device.
    /// `positions=None` is the ordinary full fold; `Some` is an over-approximating candidate
    /// set and therefore may add work but can never remove an exact match.
    fn locate_streaming_cold_slots_in_entry(
        &self,
        table: &RelationalTable,
        predicate: &crate::engine_expr::ResidentExpr,
        rtx: Index,
        entry: &Arc<ColdTableChunks>,
        positions: Option<&std::collections::BTreeSet<usize>>,
    ) -> Option<Vec<(usize, Vec<u32>)>> {
        // P4-3: class entries serve any boundary at-or-above the FREEZE (the born gate skips
        // later-born chunks below); non-class entries keep the strict entry-boundary rule.
        match self.table_chunk_authoritative(&table.name) {
            Some(freeze) => {
                if rtx < freeze {
                    return None;
                }
            }
            None => {
                if rtx < entry.build_copin_s {
                    return None;
                }
            }
        }
        let mut out: Vec<(usize, Vec<u32>)> = Vec::new();
        for (idx, chunk) in entry.chunks.iter().enumerate() {
            if positions.is_some_and(|selected| !selected.contains(&idx)) {
                continue;
            }
            if chunk.row_count == 0 || chunk.payload_copin_s > rtx {
                // Empty, or born after this boundary (the P4-3 born gate).
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
        if positions.is_some() {
            self.read_state
                .residency
                .chunk_class_device_exact_rechecks
                .fetch_add(1, Ordering::Relaxed);
        }
        Some(out)
    }

    /// Exact constraint probe over a chunk-authoritative table. The device predicate and sidecar
    /// visibility decide membership; the host receives only bounded approved coordinates so it can
    /// apply the statement's self-exclusion set and reduce them to one boolean verdict.
    pub(crate) fn chunk_class_visible_row_with_value(
        &self,
        table: &RelationalTable,
        rtx: Index,
        column_idx: usize,
        value: &SqlValue,
        exclude_keys: Option<&std::collections::BTreeSet<String>>,
    ) -> Option<bool> {
        let entry = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(&table.name)
            .cloned()?;
        let predicate = if matches!(value, SqlValue::Null) {
            crate::engine_expr::ResidentExpr::IsNull {
                col: column_idx,
                is_not_null: false,
            }
        } else {
            crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(
                table,
                &[vec![(column_idx, SelectFilterOp::Eq, value.clone())]],
            )?
        };
        let located = self.locate_streaming_cold_slots_in_entry(
            table,
            &predicate,
            rtx,
            &entry,
            None,
        )?;
        for (chunk_idx, slots) in located {
            for slot in slots {
                let pseudo_id = ((chunk_idx as u64) << 32) | u64::from(slot);
                let key = relational_row_key(&table.name, pseudo_id);
                if exclude_keys.is_none_or(|excluded| !excluded.contains(&key)) {
                    self.read_state
                        .residency
                        .chunk_class_device_exact_rechecks
                        .fetch_add(1, Ordering::Relaxed);
                    return Some(true);
                }
            }
        }
        self.read_state
            .residency
            .chunk_class_device_exact_rechecks
            .fetch_add(1, Ordering::Relaxed);
        Some(false)
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
    /// P4 COMPACTION (fence-free, the deletion directive): rebuild ONE heavily-stamped chunk
    /// from its SURVIVORS — a device projection gather (predicate=None + the sidecar mask over
    /// the staged chunk), re-encoded as a fresh sidecar-free payload born at the compacting
    /// boundary. SOUND without a fence by the reclamation argument: in-flight folds hold the OLD
    /// entry Arc; every later bind pins >= the current boundary >= `boundary`, so nobody can
    /// observe the re-slotting (the born gate would hide the chunk from a sub-boundary reader,
    /// but no such reader can bind). DELETES: the dead slots' payload bytes + the whole sidecar.
    /// `None` = the gather declined (device error) — the caller keeps the stamped, uncompacted
    /// chunk (compaction is an optimization, never load-bearing).
    fn compact_streaming_cold_chunk(
        &self,
        table: &RelationalTable,
        chunk: &ColdChunk,
        boundary: Index,
    ) -> Option<ColdChunk> {
        let select_all = Select {
            table: table.name.clone(),
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
        let bound = bind_relational_select(table, &select_all).ok()?;
        let staged = self.stage_cold_chunk(chunk, boundary).ok()?;
        let (src, vis) = staged.ready().ok()?;
        let result = self
            .execute_resident_expr_select_with_binding(
                &select_all,
                table,
                Some(&src),
                bound,
                boundary,
                None,
                vis,
                &[],
                &[],
                None,
                &[],
            )
            .ok()?;
        let survivors: Vec<Vec<SqlValue>> = result.rows.iter().map(|r| r.to_vec()).collect();
        let (snapshot, payload) = self
            .build_transient_relation_payload_only(table, &survivors)
            .ok()?;
        self.read_state
            .residency
            .chunk_class_compactions
            .fetch_add(1, Ordering::Relaxed);
        self.read_state
            .residency
            .chunk_class_compacted_slots
            .fetch_add(chunk.row_count - survivors.len() as u64, Ordering::Relaxed);
        Some(ColdChunk {
            chunk_id: COLD_CHUNK_ID.fetch_add(1, Ordering::Relaxed),
            payload: ColdPayload::Ram(Arc::new(payload)),
            snapshot,
            row_count: survivors.len() as u64,
            tuple_range: (1, 0), // class chunks carry no store ids (the sentinel)
            payload_copin_s: boundary,
            deleted_by: None,
        })
    }

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
                chunk_id: chunk.chunk_id,
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
        // P4-2b-ii: a CLASS table stamps mid-commit (pre-publish) — the general install's strict
        // committed==build equality cannot hold there; the class install's structural settledness
        // (held commit lock + serial-only class + frozen-generation check) applies (the tail-
        // append precedent). Non-class callers (the P4-2a isolation shape) keep the full proof.
        let installed = if self.table_chunk_authoritative(table_name).is_some() {
            self.install_streaming_cold_class(table_name, builder)
        } else {
            self.install_streaming_cold_inner(table_name, builder, true, commit_lock_held)
        };
        if !installed {
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

    /// Catalog eligibility (design review H1: RUNTIME state does the rest). Unique keys are served
    /// by P5's exact/Bloom candidate index. CHECK is row-local. Non-self foreign keys are served by
    /// exact device predicates over the parent/child chunks; a device decline de-authoritizes before
    /// the host validator runs. Self-FKs remain excluded because one statement's provider/consumer
    /// images interleave. Every scalar type is chunk-encodable, so types never gate.
    fn chunk_class_eligible(catalog: &crate::CatalogSnapshot, table_name: &str) -> bool {
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return false;
        };
        // P5-2 (the KEYED LIFT): a unique index no longer refuses the class WHEN it is
        // chunk-probe SERVABLE — its key positions resolve and no key column is Bool (the fold
        // kernel has no bool arm). The entry hook enforces the REST of the contract (H1: the
        // whole index set must fit the retained cap; H2: the indexes must BUILD at entry).
        for index in table.indexes.iter().filter(|index| index.unique) {
            let Some(positions) = crate::engine_residency::index_key_column_positions(table, index)
            else {
                return false;
            };
            if positions
                .iter()
                .any(|&position| matches!(table.columns[position].ty, SqlType::Bool))
            {
                return false;
            }
        }
        if table
            .foreign_keys
            .iter()
            .any(|foreign_key| foreign_key.referenced_table == table_name)
        {
            return false;
        }
        true
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
        let Some(entry) = residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
        else {
            return;
        };
        let current = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
        if !Arc::ptr_eq(&entry.generation, &current) || entry.build_copin_s != self.committed_seq()
        {
            return; // not fresh at THIS commit — a later commit's hook will retry
        }
        // P5-2 H1+H2 (the KEYED LIFT's entry contract): every unique index's chunk indexes
        // must BUILD NOW — entry time, under this commit lock, while the chunks are RAM-fresh —
        // never a lazy NVMe read later (H2); and the ESTIMATED set must fit the retained cap
        // (H1: a set that cannot co-reside would LRU-thrash on every preflight). Any decline =
        // no entry; the table simply stays store-authoritative.
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return;
        };
        let unique_key_ids: Vec<(usize, Vec<usize>)> = table
            .indexes
            .iter()
            .enumerate()
            .filter(|(_, index)| index.unique)
            .filter_map(|(key_id, index)| {
                crate::engine_residency::index_key_column_positions(table, index)
                    .map(|positions| (key_id, positions))
            })
            .collect();
        if !unique_key_ids.is_empty() {
            let per_index_bytes: u64 = entry
                .chunks
                .iter()
                .filter(|chunk| chunk.row_count > 0)
                .map(|chunk| {
                    (chunk.row_count * 2)
                        .checked_next_power_of_two()
                        .unwrap_or(u64::MAX)
                        .saturating_mul(8)
                })
                .fold(0u64, u64::saturating_add);
            if per_index_bytes.saturating_mul(unique_key_ids.len() as u64)
                <= chunk_key_index_cap_bytes()
            {
                if self.missing_chunk_key_candidates_require_spill(table, &entry, true) {
                    // Never stage spill payloads while this hook holds the global commit lock.
                    // The ordinary cold-capture install primes complete sets after releasing it.
                    return;
                }
                for (key_id, positions) in &unique_key_ids {
                    if self
                        .ensure_chunk_key_indexes(table, &entry, positions, *key_id)
                        .is_none()
                    {
                        return;
                    }
                }
            } else {
                let per_bloom_bytes: u64 = entry
                    .chunks
                    .iter()
                    .filter(|chunk| chunk.row_count > 0)
                    .map(|chunk| {
                        (chunk.row_count.saturating_mul(8).max(256))
                            .checked_next_power_of_two()
                            .unwrap_or(u64::MAX)
                            / 8
                    })
                    .fold(0u64, u64::saturating_add);
                if per_bloom_bytes.saturating_mul(unique_key_ids.len() as u64)
                    > chunk_key_bloom_cap_bytes()
                {
                    return;
                }
                if self.missing_chunk_key_candidates_require_spill(table, &entry, false) {
                    return;
                }
                for (key_id, positions) in &unique_key_ids {
                    if self
                        .ensure_chunk_key_blooms(table, &entry, positions, *key_id)
                        .is_none()
                    {
                        // No class was published, so discard any partial reservation from this
                        // attempt. A complete pre-primed set remains untouched on the success path.
                        self.purge_chunk_key_blooms_for_table(table_name);
                        return;
                    }
                }
            }
        }
        let boundary = entry.build_copin_s;
        let mut map =
            std::collections::BTreeMap::clone(&residency.chunk_authoritative_tables.load());
        map.insert(table_name.to_string(), boundary);
        residency.chunk_authoritative_tables.store(Arc::new(map));
        residency
            .chunk_class_entries
            .fetch_add(1, Ordering::Relaxed);
        // P4 RECLAMATION — THE STORE-ROW DELETION (the arc's payoff): the class table's host
        // chains + value-index entries are DROPPED at entry. Sound without a reader fence:
        // (a) in-flight readers hold COW generation Arcs — the clear publishes a NEW generation
        // and cannot touch their pinned rows; (b) every FUTURE reader of a class table either
        // streams (chunks) or passes the de-auth guard, which re-binds at the CURRENT boundary
        // (>= the freeze) — no reader can ever need the dropped sub-freeze history; (c) de-auth
        // v2 rebuilds chunk-only. The allocator is preserved (identities never reuse). The
        // cleared store publishes a fresh generation, so the entry RE-PINS it (the class
        // invariants key on generation pointer stability from here on).
        let reclaimed: u64 = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .store()
            .all_versions()
            .len() as u64;
        if self
            .read_state
            .mvcc
            .with_table_mut(table_name, |data| {
                data.rows.clear_versions_preserving_allocator();
                data.value_index = Default::default();
                Ok::<(), EngineError>(())
            })
            .is_err()
        {
            // Audit LOW: never run classed-but-unreclaimed on a swallowed error — exit loudly.
            let _ = self.deauthoritize_chunk_table(table_name, true);
            return;
        }
        let cleared_generation = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
        let repinned = ColdCacheBuilder {
            generation: cleared_generation,
            build_copin_s: entry.build_copin_s,
            chunk_target_bytes: entry.chunk_target_bytes,
            total_payload_bytes: entry.total_payload_bytes,
            column_signature: entry.column_signature.clone(),
            chunks: entry
                .chunks
                .iter()
                .map(|chunk| ColdChunk {
                    chunk_id: chunk.chunk_id,
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
        if !self.install_streaming_cold_class(table_name, repinned) {
            // The re-pin cannot legitimately fail (we hold the commit lock and just published
            // the cleared generation) — if it ever does, exit the class LOUDLY rather than run
            // with a mismatched pin.
            let _ = self.deauthoritize_chunk_table(table_name, true);
            return;
        }
        residency
            .chunk_class_reclaimed_rows
            .fetch_add(reclaimed, Ordering::Relaxed);
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
        let Some(entry) = residency
            .streaming_cold_chunks
            .load()
            .get(&table.name)
            .cloned()
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
                    chunk_id: chunk.chunk_id,
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
                chunk_id: COLD_CHUNK_ID.fetch_add(1, Ordering::Relaxed),
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
        let current = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
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
            entry_epoch: COLD_ENTRY_EPOCH.fetch_add(1, Ordering::Relaxed),
            chunks: builder.chunks,
        });
        let residency = &self.read_state.residency;
        let _publish = residency
            .streaming_cold_lock
            .lock()
            .expect("streaming cold-tier lock poisoned");
        let mut map = std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
        let live_chunk_ids: std::collections::BTreeSet<u64> =
            entry.chunks.iter().map(|chunk| chunk.chunk_id).collect();
        map.insert(table_name.to_string(), entry);
        residency.streaming_cold_chunks.store(Arc::new(map));
        drop(_publish);
        self.purge_stale_chunk_key_candidates(table_name, &live_chunk_ids);
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
                    // P4 DE-AUTH v2 (chunk-only — the store rows were RECLAIMED at class entry,
                    // so there is nothing to map into): EVERY chunk replays by inserting its
                    // slot-aligned unmasked rows at FRESH ids — base chunks BORN-VISIBLE
                    // (created 0: every post-de-auth reader binds at the current boundary,
                    // which is >= the freeze >= every base row's real birth; in-flight readers
                    // keep their COW generation pins), tail chunks at their born boundary —
                    // then replaying every sidecar stamp as a tombstone on the just-inserted
                    // id. The store is whole for every FUTURE boundary; the old slot->store-id
                    // rank enumeration is DELETED with the frozen rows it mapped into.
                    let rows = decode_cold_chunk_rows(&table, chunk, 0).map_err(|e| {
                        EngineError::ApplyFailed(format!("de-authoritization decode failed: {e}"))
                    })?;
                    let born = if chunk.payload_copin_s <= freeze {
                        // Base: effectively born-visible — created@1 <= every real boundary
                        // (commit seqs are positive; the storage API rejects a literal 0).
                        1
                    } else {
                        chunk.payload_copin_s
                    };
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
                                crate::rel_exec_helpers::relational_row_key(table_name, row_id),
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
                                        value: crate::rel_exec_helpers::encode_relational_row(row),
                                    },
                                    born,
                                )
                                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                        }
                        for (entry_key, row_keys) in &index_entries {
                            let mut slot =
                                data.value_index.get(entry_key).cloned().unwrap_or_default();
                            slot.extend(row_keys.iter().cloned());
                            data.value_index.insert(entry_key.clone(), slot);
                        }
                        // Every chunk's stamps replay onto the just-inserted ids: the chain
                        // gets created@born + deleted@stamp (a pre-freeze stamp on a base chunk
                        // = an already-dead chain — wasteful, MVCC-correct).
                        if let Some(sidecar) = &chunk.deleted_by {
                            for (slot, tuple_id) in tuple_ids.iter().enumerate() {
                                let raw = i64::from_le_bytes(
                                    sidecar[slot * 8..slot * 8 + 8].try_into().expect("8"),
                                );
                                let live = i64::from_le_bytes([COLD_DELETED_BY_LIVE_FILL_BYTE; 8]);
                                if raw != live {
                                    data.rows
                                        .tuple_delete(*tuple_id, raw as u64)
                                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                                }
                            }
                        }
                        Ok::<(), EngineError>(())
                    })?;
                }
            }
            // Leave the class + evict the (now superseded-generation) cache entry.
            let mut map =
                std::collections::BTreeMap::clone(&residency.chunk_authoritative_tables.load());
            map.remove(table_name);
            residency.chunk_authoritative_tables.store(Arc::new(map));
            engine.evict_streaming_cold(table_name);
            // P5-2 audit LOW: purge the table's chunk KEY-INDEX cache entries too — chunk ids
            // are monotonic, so post-de-auth entries can never be re-hit; leaving them inflates
            // `chunk_key_index_bytes` (VRAM retention + premature LRU eviction of live tables).
            engine.purge_chunk_key_indexes_for_table(table_name);
            engine.purge_chunk_key_blooms_for_table(table_name);
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

    /// P4-2b-ii — resolve a class table's DELETE/UPDATE matches FROM THE CHUNKS: the P4-2a
    /// locate yields (chunk_idx, slot) coordinates (sidecar mask composed — tombstoned slots
    /// never re-match), the P4-1 decoder extracts each coordinate's row image (an UNMASKED
    /// rtx=0 decode is slot-aligned: every slot visible, direct indexing), and the match triple
    /// fabricates its identity from the PACKED coordinate (class rows have no store row ids;
    /// the pseudo-id is unique within the entry epoch, which rides the delta as the P4-2a
    /// COORDINATE TOKEN — the commit hook stamps only while the installed entry still carries
    /// it). `None` = decline (unlowerable predicate, any locate/decode failure) — the caller
    /// de-authoritizes (the sticky exit stays the correctness backstop).
    pub(crate) fn resolve_class_dml_matches(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        visibility: StorageVisibility,
    ) -> Option<(ClassDmlMatches, u64)> {
        let entry = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(&table.name)
            .cloned()?;
        #[cfg(test)]
        let resolve_pin_hook = {
            class_resolve_pin_hook()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take()
        };
        #[cfg(test)]
        if let Some((pinned, resume)) = resolve_pin_hook {
            pinned.wait();
            resume.wait();
        }
        let epoch = entry.entry_epoch;
        // P5-3: an Eq-on-unique-key WHERE locates via the CHUNK KEY-INDEX PROBE — one device
        // locate + per-hit slot rechecks — instead of the full fold scan over every chunk, and
        // its matches materialize from the RECHECKED slots (the reverse-gather decoder stays
        // off the point-DML hot path). Anything else (range/OR/NULL/no covering key/any probe
        // failure) falls to the fold path below — never a decline.
        if let Some(matches) = self.resolve_class_dml_via_key_probe(
            table,
            filter_groups,
            visibility.read_txn_id,
            &entry,
        ) {
            self.read_state
                .residency
                .chunk_class_dml_key_locates
                .fetch_add(1, Ordering::Relaxed);
            return Some((matches, epoch));
        }
        let predicate =
            crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(table, filter_groups)?;
        // Keep coordinates, row images, and the returned epoch on ONE pinned entry Arc. This
        // resolver can run off-lock; reloading inside locate would let a concurrent tail/stamp/
        // compaction publish E2, then interpret E2 coordinates against E1 below and poison the
        // prepare-time write-set used by SI conflict detection.
        let located = self.locate_streaming_cold_slots_in_entry(
            table,
            &predicate,
            visibility.read_txn_id,
            &entry,
            None,
        )?;
        let mut matches: ClassDmlMatches = Vec::new();
        for (chunk_idx, slots) in &located {
            let chunk = entry.chunks.get(*chunk_idx)?;
            let staged = self.stage_cold_chunk(chunk, chunk.payload_copin_s).ok()?;
            let (src, _vis) = staged.ready().ok()?;
            for slot in slots {
                // The device locate already decided the exact predicate + visibility. Read
                // back only the approved row image; never decode the host cold payload here.
                let image =
                    self.read_cold_chunk_slot_values(table, chunk, &src, *slot as usize)?;
                let pseudo_id = ((*chunk_idx as u64) << 32) | u64::from(*slot);
                let key = crate::rel_exec_helpers::relational_row_key(&table.name, pseudo_id);
                matches.push((pseudo_id, key, image));
            }
        }
        Some((matches, epoch))
    }

    /// P5-3 — the by-key DML locate: serve a single-group ALL-Eq WHERE that covers some unique
    /// index's key columns through the chunk key-index probe. The probe only chooses candidate
    /// chunks; the complete predicate and visibility mask then run on-device over those chunks.
    /// This exact device pass resolves fingerprint collisions and residual predicates before the
    /// host receives final survivor coordinates or row values. `None` = NOT ELIGIBLE or any
    /// failure — the caller falls to the full device fold (never a host relational recheck).
    fn resolve_class_dml_via_key_probe(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        rtx: Index,
        entry: &Arc<ColdTableChunks>,
    ) -> Option<ClassDmlMatches> {
        // Audit MEDIUM (P5-3): mirror the fold locate's `rtx < freeze` DECLINE exactly — a
        // sub-freeze reader boundary must drive the caller's DE-AUTH (the frozen chains serve
        // it), never a silent 0-row DML (every class chunk is born at-or-above the freeze, so
        // the recheck's born gate would mask ALL hits and quietly bypass the safety valve).
        if let Some(freeze) = self.table_chunk_authoritative(&table.name) {
            if rtx < freeze {
                return None;
            }
        }
        let [group] = filter_groups else {
            return None; // OR groups keep the fold
        };
        if group.is_empty()
            || group
                .iter()
                .any(|(_, op, value)| *op != SelectFilterOp::Eq || matches!(value, SqlValue::Null))
        {
            return None; // range / NULL-Eq keep the fold (host WHERE-NULL semantics ride it)
        }
        let eq_positions: std::collections::BTreeMap<usize, &SqlValue> =
            group.iter().map(|(idx, _, value)| (*idx, value)).collect();
        // The FIRST unique index fully covered by the Eq columns carries the probe.
        let (key_id, positions) =
            table
                .indexes
                .iter()
                .enumerate()
                .find_map(|(key_id, index)| {
                    if !index.unique {
                        return None;
                    }
                    let positions =
                        crate::engine_residency::index_key_column_positions(table, index)?;
                    positions
                        .iter()
                        .all(|position| eq_positions.contains_key(position))
                        .then_some((key_id, positions))
                })?;
        // Synthesize the needle row: key positions carry the Eq values (chunk_key_needle reads
        // ONLY the key positions).
        let mut needle_row: Vec<SqlValue> = vec![SqlValue::Null; table.columns.len()];
        for &position in &positions {
            needle_row[position] = (*eq_positions.get(&position)?).clone();
        }
        let needle = Self::chunk_key_needle(table, &positions, &needle_row)?;
        let (hits, _) = self.chunk_key_candidate_positions(
            table,
            entry,
            &positions,
            key_id,
            &[needle],
        )?;
        let candidate_positions: std::collections::BTreeSet<usize> = hits
            .first()?
            .iter()
            .copied()
            .collect();
        if candidate_positions.is_empty() {
            return Some(Vec::new());
        }
        let exact_predicate =
            crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(table, filter_groups)?;
        let exact = self.locate_streaming_cold_slots_in_entry(
            table,
            &exact_predicate,
            rtx,
            entry,
            Some(&candidate_positions),
        )?;
        let mut matches: ClassDmlMatches = Vec::new();
        for (position, slots) in exact {
            let chunk = entry.chunks.get(position)?;
            let staged = self.stage_cold_chunk(chunk, chunk.payload_copin_s).ok()?;
            let (src, _vis) = staged.ready().ok()?;
            for slot in slots {
                // The exact device pass above already decided visibility + the complete
                // predicate. This is the one final value readback needed to stage the DML image.
                let row = self.read_cold_chunk_slot_values(table, chunk, &src, slot as usize)?;
                let pseudo_id = ((position as u64) << 32) | u64::from(slot);
                let key = crate::rel_exec_helpers::relational_row_key(&table.name, pseudo_id);
                matches.push((pseudo_id, key, row));
            }
        }
        Some(matches)
    }

    /// P4-2b-ii — the commit hook's STAMP arm: verify the COORDINATE TOKEN (the entry installed
    /// NOW must still carry the prepare-time epoch — any interposed install re-tiled or advanced
    /// it) and tombstone the packed coordinates at the committing boundary. `false` = the caller
    /// must de-authoritize (never a mis-stamp).
    pub(crate) fn stamp_class_coordinates(
        &self,
        table_name: &str,
        packed: &[u64],
        epoch: u64,
        stamp: Index,
    ) -> bool {
        let Some(entry) = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
        else {
            return false;
        };
        if entry.entry_epoch != epoch {
            return false; // the token expired — coordinates may be misaligned
        }
        let mut per_chunk: std::collections::BTreeMap<usize, Vec<u32>> =
            std::collections::BTreeMap::new();
        for p in packed {
            per_chunk
                .entry((p >> 32) as usize)
                .or_default()
                .push((p & 0xFFFF_FFFF) as u32);
        }
        let located: Vec<(usize, Vec<u32>)> = per_chunk.into_iter().collect();
        self.stamp_streaming_cold_slots(table_name, &located, stamp, true)
    }

    /// P4 COMPACTION driver — runs at the commit hook AFTER `publish_committed_seq` (the timing
    /// is load-bearing: the compacted chunk is born at the CURRENT PUBLISHED boundary, so every
    /// later bind pins at-or-above it and sees it; a PRE-publish install would let a concurrent
    /// boundary-minus-one bind load the new entry and born-skip the chunk — its SURVIVORS would
    /// vanish for that read. In-flight readers hold the old entry Arc either way). Scans the
    /// class entry's sidecars; every chunk past the dead-fraction threshold rebuilds from its
    /// survivors in ONE fresh install.
    pub(crate) fn maybe_compact_chunk_class(&self, table_name: &str) {
        if self.table_chunk_authoritative(table_name).is_none() {
            return;
        }
        let residency = &self.read_state.residency;
        let Some(entry) = residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
        else {
            return;
        };
        let live = i64::from_le_bytes([COLD_DELETED_BY_LIVE_FILL_BYTE; 8]);
        let needs: Vec<usize> = entry
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, chunk)| {
                let Some(sidecar) = &chunk.deleted_by else {
                    return false;
                };
                if chunk.row_count < 8 {
                    return false;
                }
                let dead = (0..chunk.row_count as usize)
                    .filter(|slot| {
                        i64::from_le_bytes(sidecar[slot * 8..slot * 8 + 8].try_into().expect("8"))
                            != live
                    })
                    .count() as u64;
                dead * 4 >= chunk.row_count
            })
            .map(|(idx, _)| idx)
            .collect();
        if needs.is_empty() {
            return;
        }
        let Some(table) = self
            .catalog_snapshot()
            .relational_catalog
            .get(table_name)
            .cloned()
        else {
            return;
        };
        let boundary = self.committed_seq();
        let mut chunks: Vec<ColdChunk> = Vec::with_capacity(entry.chunks.len());
        let mut total = entry.total_payload_bytes;
        for (idx, chunk) in entry.chunks.iter().enumerate() {
            if needs.contains(&idx) {
                if let Some(compacted) = self.compact_streaming_cold_chunk(&table, chunk, boundary)
                {
                    // Cap accounting: subtract the replaced payload + sidecar, add the new.
                    let old_payload = match &chunk.payload {
                        ColdPayload::Ram(b) => b.len() as u64,
                        ColdPayload::Spilled { len, .. } => *len as u64,
                    };
                    let old_sidecar = chunk.deleted_by.as_ref().map_or(0, |b| b.len() as u64);
                    let new_payload = match &compacted.payload {
                        ColdPayload::Ram(b) => b.len() as u64,
                        ColdPayload::Spilled { len, .. } => *len as u64,
                    };
                    total = total
                        .saturating_sub(old_payload + old_sidecar)
                        .saturating_add(new_payload);
                    chunks.push(compacted);
                    continue;
                }
            }
            chunks.push(ColdChunk {
                chunk_id: chunk.chunk_id,
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
            });
        }
        let builder = ColdCacheBuilder {
            generation: Arc::clone(&entry.generation),
            build_copin_s: entry.build_copin_s.max(boundary),
            chunk_target_bytes: entry.chunk_target_bytes,
            total_payload_bytes: total,
            column_signature: entry.column_signature.clone(),
            chunks,
            spill: None,
            poisoned: false,
        };
        let _ = self.install_streaming_cold_class(table_name, builder);
    }

    // ================= P5-1: THE PER-CHUNK DEVICE KEY-INDEX CACHE =================

    /// Build ONE chunk's key index: stage the payload, derive per-row KEY FINGERPRINTS (a single
    /// int4 key reads its column verbatim; every other shape folds ON-DEVICE via
    /// `submit_compound_fold_fingerprints` — raw key bytes never reach the host), build the
    /// hash table host-side from the fingerprints (ALL-VISIBLE: the sidecar applies at the
    /// probe's recheck), and RETAIN it as a persistent device buffer. `None` = decline (an
    /// in-chunk duplicate fingerprint under the non-dup-tolerant build, a stage/read failure) —
    /// the caller treats the table as probe-unservable (P5-2 de-auths).
    #[allow(dead_code)] // P5-2 wires the production caller.
    fn build_chunk_key_index(
        &self,
        table: &RelationalTable,
        chunk: &ColdChunk,
        key_positions: &[usize],
    ) -> Option<ChunkKeyIndex> {
        use crate::relational_model::{
            resident_device_int4_column_offset, resident_device_int8_column_offset,
            resident_device_numeric_column_offset, resident_device_text_column_layout,
        };
        if chunk.row_count == 0 {
            return None;
        }
        let staged = self.stage_cold_chunk(chunk, chunk.payload_copin_s).ok()?;
        let (src, _vis) = staged.ready().ok()?;
        let d = &chunk.snapshot;
        let row_count = chunk.row_count as usize;
        // blob_offsets is PER-COLUMN PARALLEL to offsets (audit HIGH: the fold wrapper errors on
        // a length mismatch and the kernel reads blob_offsets[k] only where widths[k]==0 — the
        // text sentinel; every non-text column carries a 0 placeholder, mirroring the shard
        // caller's construction).
        let mut offsets: Vec<u64> = Vec::with_capacity(key_positions.len());
        let mut blob_offsets: Vec<u64> = Vec::with_capacity(key_positions.len());
        let mut blob_lens: Vec<u64> = Vec::with_capacity(key_positions.len());
        for &pos in key_positions {
            let column = table.columns.get(pos)?;
            match column.ty {
                SqlType::Int4 | SqlType::Date | SqlType::Int2 => {
                    offsets.push(resident_device_int4_column_offset(d, table, pos).ok()?);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Int8 | SqlType::Timestamp => {
                    offsets.push(resident_device_int8_column_offset(d, table, pos).ok()?);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Numeric { .. } | SqlType::Uuid => {
                    offsets.push(resident_device_numeric_column_offset(d, table, pos).ok()?);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(d, table, pos).ok()?;
                    offsets.push(layout.offsets_byte_offset);
                    blob_offsets.push(layout.bytes_byte_offset);
                    blob_lens.push(layout.bytes_len);
                }
                SqlType::Bool => return None, // a bool key is not a real-world unique key
            }
        }
        let keys: Vec<i32> = if key_positions.len() == 1
            && matches!(
                table.columns[key_positions[0]].ty,
                SqlType::Int4 | SqlType::Date | SqlType::Int2
            ) {
            src.device_memory
                .read_resident_i32_column(offsets[0], row_count)
                .ok()?
        } else {
            let widths = key_positions
                .iter()
                .map(|&p| crate::engine_residency::key_column_width_words(table.columns[p].ty))
                .collect::<Option<Vec<u32>>>()?;
            let fold_columns = widths
                .iter()
                .enumerate()
                .map(|(idx, &width_words)| {
                    if width_words == 0 {
                        CudaCompoundFoldColumn::Text {
                            offsets_byte_offset: offsets[idx],
                            bytes_byte_offset: blob_offsets[idx],
                            bytes_len: blob_lens[idx],
                        }
                    } else {
                        CudaCompoundFoldColumn::Fixed {
                            byte_offset: offsets[idx],
                            width_words,
                        }
                    }
                })
                .collect::<Vec<_>>();
            src.device_memory
                .submit_compound_fold_fingerprints(&fold_columns, row_count)
                .ok()?
        };
        if keys.len() != row_count {
            return None;
        }
        // All-visible, non-dup-tolerant: a keyed class chunk's fingerprints are unique unless a
        // genuine fingerprint COLLISION exists in-chunk — decline then (probe-unservable).
        let (hash_table, table_mask, hash_shift) =
            crate::engine_retained_read::build_int4_pk_hash_table_host_visible(
                &keys,
                chunk.row_count,
                None,
                0,
                // DUP-TOLERANT (audit availability finding): a 32-bit fingerprint birthday
                // collision between DISTINCT keys must not decline the chunk (at 50k+ rows the
                // decline rate is material) — colliding entries chain to the next probe slot,
                // the write-locate walks ALL matches, and the full-tuple recheck resolves.
                true,
            )?;
        let bytes: Vec<u8> = hash_table.iter().flat_map(|w| w.to_le_bytes()).collect();
        let runtime = self.cuda_driver_probe_runtime();
        let device = runtime.retain_device_memory_copy(d.gpu_id, &bytes).ok()?;
        Some(ChunkKeyIndex {
            device: Arc::new(device),
            table_mask,
            hash_shift,
            row_count: u32::try_from(row_count).ok()?,
            bytes: bytes.len() as u64,
            last_used: 0,
        })
    }

    /// Build the compact Bloom twin of a chunk key index. Fingerprints are derived with the same device fold as
    /// the exact index; the host only packs the staging bitset. Candidate membership is decided by the GPU and
    /// every positive is rechecked by the exact device predicate, so Bloom false positives are harmless.
    fn build_chunk_key_bloom(
        &self,
        table: &RelationalTable,
        chunk: &ColdChunk,
        key_positions: &[usize],
    ) -> Option<ChunkKeyBloom> {
        use crate::relational_model::{
            resident_device_int4_column_offset, resident_device_int8_column_offset,
            resident_device_numeric_column_offset, resident_device_text_column_layout,
        };
        let staged = self.stage_cold_chunk(chunk, chunk.payload_copin_s).ok()?;
        let (src, _vis) = staged.ready().ok()?;
        let mut offsets = Vec::with_capacity(key_positions.len());
        let mut blob_offsets = Vec::with_capacity(key_positions.len());
        let mut blob_lens = Vec::with_capacity(key_positions.len());
        for &pos in key_positions {
            match table.columns.get(pos)?.ty {
                SqlType::Int4 | SqlType::Date | SqlType::Int2 => {
                    offsets.push(resident_device_int4_column_offset(&chunk.snapshot, table, pos).ok()?);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Int8 | SqlType::Timestamp => {
                    offsets.push(resident_device_int8_column_offset(&chunk.snapshot, table, pos).ok()?);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Numeric { .. } | SqlType::Uuid => {
                    offsets.push(resident_device_numeric_column_offset(&chunk.snapshot, table, pos).ok()?);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(&chunk.snapshot, table, pos).ok()?;
                    offsets.push(layout.offsets_byte_offset);
                    blob_offsets.push(layout.bytes_byte_offset);
                    blob_lens.push(layout.bytes_len);
                }
                SqlType::Bool => return None,
            }
        }
        let keys = if key_positions.len() == 1
            && matches!(table.columns[key_positions[0]].ty, SqlType::Int4 | SqlType::Date | SqlType::Int2)
        {
            src.device_memory
                .read_resident_i32_column(offsets[0], chunk.row_count as usize)
                .ok()?
        } else {
            let widths = key_positions
                .iter()
                .map(|&p| crate::engine_residency::key_column_width_words(table.columns[p].ty))
                .collect::<Option<Vec<_>>>()?;
            let fold_columns = widths
                .iter()
                .enumerate()
                .map(|(idx, &width_words)| {
                    if width_words == 0 {
                        CudaCompoundFoldColumn::Text {
                            offsets_byte_offset: offsets[idx],
                            bytes_byte_offset: blob_offsets[idx],
                            bytes_len: blob_lens[idx],
                        }
                    } else {
                        CudaCompoundFoldColumn::Fixed {
                            byte_offset: offsets[idx],
                            width_words,
                        }
                    }
                })
                .collect::<Vec<_>>();
            src.device_memory
                .submit_compound_fold_fingerprints(&fold_columns, chunk.row_count as usize)
                .ok()?
        };
        if keys.len() != chunk.row_count as usize {
            return None;
        }
        let bit_count = (chunk.row_count.saturating_mul(8).max(256))
            .checked_next_power_of_two()?
            .min(1u64 << 31);
        let bit_mask = u32::try_from(bit_count - 1).ok()?;
        let mut words = vec![0u32; (bit_count / 32) as usize];
        for key in keys {
            let key = key as u32;
            let h1 = key.wrapping_mul(2_654_435_761);
            let h2 = (key ^ (key >> 16)).wrapping_mul(2_246_822_519) | 1;
            for i in 0..3u32 {
                let bit = h1.wrapping_add(i.wrapping_mul(h2)) & bit_mask;
                words[(bit >> 5) as usize] |= 1u32 << (bit & 31);
            }
        }
        #[cfg(test)]
        if CHUNK_KEY_BLOOM_ALL_POSITIVE_TEST.load(Ordering::Relaxed) {
            words.fill(u32::MAX);
        }
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let device = self
            .cuda_driver_probe_runtime()
            .retain_device_memory_copy(chunk.snapshot.gpu_id, &bytes)
            .ok()?;
        Some(ChunkKeyBloom {
            device: Arc::new(device),
            bit_mask,
            bytes: bytes.len() as u64,
        })
    }

    fn ensure_chunk_key_blooms(
        &self,
        table: &RelationalTable,
        entry: &Arc<ColdTableChunks>,
        key_positions: &[usize],
        key_id: usize,
    ) -> Option<Vec<(usize, ChunkKeyBloom)>> {
        let residency = &self.read_state.residency;
        if residency.chunk_key_bloom_bytes.load(Ordering::Relaxed)
            > chunk_key_bloom_cap_bytes()
        {
            return None;
        }
        let mut out = Vec::new();
        for (position, chunk) in entry.chunks.iter().enumerate().filter(|(_, c)| c.row_count > 0) {
            let key = (table.name.clone(), chunk.chunk_id, key_id);
            let cached = {
                residency
                    .chunk_key_bloom
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .get(&key)
                    .cloned()
            };
            let bloom = if let Some(hit) = cached {
                hit
            } else {
                let built = self.build_chunk_key_bloom(table, chunk, key_positions)?;
                let mut cache = residency.chunk_key_bloom.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(hit) = cache.get(&key).cloned() {
                    hit
                } else {
                    let current = residency.chunk_key_bloom_bytes.load(Ordering::Relaxed);
                    if current.saturating_add(built.bytes) > chunk_key_bloom_cap_bytes() {
                        return None;
                    }
                    residency
                        .chunk_key_bloom_bytes
                        .fetch_add(built.bytes, Ordering::Relaxed);
                    cache.insert(key, built.clone());
                    built
                }
            };
            out.push((position, bloom));
        }
        Some(out)
    }

    fn probe_chunk_key_blooms(
        &self,
        blooms: &[(usize, ChunkKeyBloom)],
        needles: &[i32],
    ) -> Option<Vec<Vec<usize>>> {
        if blooms.is_empty() {
            return Some(vec![Vec::new(); needles.len()]);
        }
        let device_blooms: Vec<_> = blooms
            .iter()
            .map(|(_, bloom)| gpu_db_execution::ChunkBloomProbeShard {
                bloom: Arc::clone(&bloom.device),
                bit_mask: bloom.bit_mask,
            })
            .collect();
        let candidates = device_blooms[0]
            .bloom
            .probe_chunk_blooms(&device_blooms, needles)
            .ok()?;
        self.read_state.residency.chunk_key_bloom_probes.fetch_add(1, Ordering::Relaxed);
        candidates
            .into_iter()
            .map(|chunks| {
                chunks
                    .into_iter()
                    .map(|filtered| blooms.get(filtered as usize).map(|pair| pair.0))
                    .collect::<Option<Vec<_>>>()
            })
            .collect()
    }

    fn chunk_key_exact_set_bytes(table: &RelationalTable, entry: &ColdTableChunks) -> u64 {
        let unique_keys = table.indexes.iter().filter(|index| index.unique).count() as u64;
        entry
            .chunks
            .iter()
            .filter(|chunk| chunk.row_count > 0)
            .map(|chunk| {
                (chunk.row_count.saturating_mul(2))
                    .checked_next_power_of_two()
                    .unwrap_or(u64::MAX)
                    .saturating_mul(8)
            })
            .fold(0u64, u64::saturating_add)
            .saturating_mul(unique_keys)
    }

    fn chunk_key_unique_positions(table: &RelationalTable) -> Vec<(usize, Vec<usize>)> {
        table
            .indexes
            .iter()
            .enumerate()
            .filter(|(_, index)| index.unique)
            .filter_map(|(key_id, index)| {
                crate::engine_residency::index_key_column_positions(table, index)
                    .map(|positions| (key_id, positions))
            })
            .collect()
    }

    fn purge_chunk_key_indexes_for_table(&self, table_name: &str) {
        let residency = &self.read_state.residency;
        let mut cache = residency
            .chunk_key_index
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let stale: Vec<_> = cache
            .keys()
            .filter(|(name, _, _)| name == table_name)
            .cloned()
            .collect();
        for key in stale {
            if let Some(evicted) = cache.remove(&key) {
                residency
                    .chunk_key_index_bytes
                    .fetch_sub(evicted.bytes, Ordering::Relaxed);
            }
        }
    }

    fn purge_chunk_key_blooms_for_table(&self, table_name: &str) {
        let residency = &self.read_state.residency;
        let mut cache = residency
            .chunk_key_bloom
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let stale: Vec<_> = cache
            .keys()
            .filter(|(name, _, _)| name == table_name)
            .cloned()
            .collect();
        for key in stale {
            if let Some(evicted) = cache.remove(&key) {
                residency
                    .chunk_key_bloom_bytes
                    .fetch_sub(evicted.bytes, Ordering::Relaxed);
            }
        }
    }

    fn purge_stale_chunk_key_candidates(
        &self,
        table_name: &str,
        live_chunk_ids: &std::collections::BTreeSet<u64>,
    ) {
        let residency = &self.read_state.residency;
        {
            let mut cache = residency
                .chunk_key_index
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let stale: Vec<_> = cache
                .keys()
                .filter(|(name, chunk_id, _)| {
                    name == table_name && !live_chunk_ids.contains(chunk_id)
                })
                .cloned()
                .collect();
            for key in stale {
                if let Some(evicted) = cache.remove(&key) {
                    residency
                        .chunk_key_index_bytes
                        .fetch_sub(evicted.bytes, Ordering::Relaxed);
                }
            }
        }
        {
            let mut cache = residency
                .chunk_key_bloom
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let stale: Vec<_> = cache
                .keys()
                .filter(|(name, chunk_id, _)| {
                    name == table_name && !live_chunk_ids.contains(chunk_id)
                })
                .cloned()
                .collect();
            for key in stale {
                if let Some(evicted) = cache.remove(&key) {
                    residency
                        .chunk_key_bloom_bytes
                        .fetch_sub(evicted.bytes, Ordering::Relaxed);
                }
            }
        }
    }

    fn purge_chunk_key_candidates_for_ids(
        &self,
        table_name: &str,
        chunk_ids: &std::collections::BTreeSet<u64>,
    ) {
        if chunk_ids.is_empty() {
            return;
        }
        let residency = &self.read_state.residency;
        {
            let mut cache = residency
                .chunk_key_index
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let stale: Vec<_> = cache
                .keys()
                .filter(|(name, chunk_id, _)| {
                    name == table_name && chunk_ids.contains(chunk_id)
                })
                .cloned()
                .collect();
            for key in stale {
                if let Some(evicted) = cache.remove(&key) {
                    residency
                        .chunk_key_index_bytes
                        .fetch_sub(evicted.bytes, Ordering::Relaxed);
                }
            }
        }
        {
            let mut cache = residency
                .chunk_key_bloom
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let stale: Vec<_> = cache
                .keys()
                .filter(|(name, chunk_id, _)| {
                    name == table_name && chunk_ids.contains(chunk_id)
                })
                .cloned()
                .collect();
            for key in stale {
                if let Some(evicted) = cache.remove(&key) {
                    residency
                        .chunk_key_bloom_bytes
                        .fetch_sub(evicted.bytes, Ordering::Relaxed);
                }
            }
        }
    }

    fn missing_chunk_key_candidates_require_spill(
        &self,
        table: &RelationalTable,
        entry: &ColdTableChunks,
        exact: bool,
    ) -> bool {
        let keys = Self::chunk_key_unique_positions(table);
        if keys.is_empty() {
            return false;
        }
        let residency = &self.read_state.residency;
        if exact {
            let cache = residency
                .chunk_key_index
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            entry
                .chunks
                .iter()
                .filter(|chunk| {
                    chunk.row_count > 0 && matches!(chunk.payload, ColdPayload::Spilled { .. })
                })
                .any(|chunk| {
                    keys.iter().any(|(key_id, _)| {
                        !cache.contains_key(&(table.name.clone(), chunk.chunk_id, *key_id))
                    })
                })
        } else {
            let cache = residency
                .chunk_key_bloom
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            entry
                .chunks
                .iter()
                .filter(|chunk| {
                    chunk.row_count > 0 && matches!(chunk.payload, ColdPayload::Spilled { .. })
                })
                .any(|chunk| {
                    keys.iter().any(|(key_id, _)| {
                        !cache.contains_key(&(table.name.clone(), chunk.chunk_id, *key_id))
                    })
                })
        }
    }

    /// Build the complete candidate set after an ordinary cold capture has released the commit
    /// mutex. Spilled chunks may read NVMe here. Class entry later only verifies/reuses this set.
    fn prime_chunk_key_candidates(&self, table_name: &str, entry: &Arc<ColdTableChunks>) {
        let catalog = self.catalog_snapshot();
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return;
        };
        if !Self::chunk_class_eligible(&catalog, table_name) {
            return;
        }
        let keys = Self::chunk_key_unique_positions(table);
        if keys.is_empty() {
            return;
        }
        let entry_chunk_ids: std::collections::BTreeSet<u64> =
            entry.chunks.iter().map(|chunk| chunk.chunk_id).collect();
        #[cfg(test)]
        if let Some((pinned, resume)) = chunk_key_prime_pin_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            pinned.wait();
            resume.wait();
        }
        let mut complete = true;
        if Self::chunk_key_exact_set_bytes(table, entry) <= chunk_key_index_cap_bytes() {
            for (key_id, positions) in keys {
                if self
                    .ensure_chunk_key_indexes(table, entry, &positions, key_id)
                    .is_none()
                {
                    complete = false;
                    break;
                }
            }
        } else {
            for (key_id, positions) in keys {
                if self
                    .ensure_chunk_key_blooms(table, entry, &positions, key_id)
                    .is_none()
                {
                    complete = false;
                    break;
                }
            }
        }
        // Publication may have advanced while the off-lock GPU/NVMe work ran. Remove only IDs
        // belonging to this primed entry that are no longer live; never table-wide purge here,
        // because a newer entry may already have installed/built its own tail candidates. If this
        // exact entry is still current but priming was partial, roll the whole partial reservation
        // back so a retry cannot accumulate toward the global cap.
        let current = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned();
        let obsolete = match current {
            Some(current) if Arc::ptr_eq(&current, entry) => {
                if complete {
                    std::collections::BTreeSet::new()
                } else {
                    entry_chunk_ids
                }
            }
            Some(current) => {
                let live: std::collections::BTreeSet<u64> =
                    current.chunks.iter().map(|chunk| chunk.chunk_id).collect();
                entry_chunk_ids.difference(&live).copied().collect()
            }
            None => entry_chunk_ids,
        };
        self.purge_chunk_key_candidates_for_ids(table_name, &obsolete);
    }

    /// Select candidate chunks without ever making a host membership decision. The retained
    /// exact hash set is preferred while the complete class set fits its cap; otherwise the
    /// compact all-chunk Bloom set runs on the GPU. Both are candidate-only: callers must run
    /// the authoritative device predicate/visibility pass over every returned chunk.
    fn chunk_key_candidate_positions(
        &self,
        table: &RelationalTable,
        entry: &Arc<ColdTableChunks>,
        key_positions: &[usize],
        key_id: usize,
        needles: &[i32],
    ) -> Option<ChunkKeyCandidates> {
        if Self::chunk_key_exact_set_bytes(table, entry) <= chunk_key_index_cap_bytes() {
            if self.missing_chunk_key_candidates_require_spill(table, entry, true) {
                return None;
            }
            self.purge_chunk_key_blooms_for_table(&table.name);
            let indexes = self.ensure_chunk_key_indexes(table, entry, key_positions, key_id)?;
            let candidates = self
                .probe_chunk_key_indexes(&indexes, needles)?
                .into_iter()
                .map(|hits| hits.into_iter().map(|(position, _)| position).collect())
                .collect();
            return Some((
                candidates,
                indexes.first().map(|(_, index)| Arc::clone(&index.device)),
            ));
        }
        if self.missing_chunk_key_candidates_require_spill(table, entry, false) {
            return None;
        }
        self.purge_chunk_key_indexes_for_table(&table.name);
        let blooms = self.ensure_chunk_key_blooms(table, entry, key_positions, key_id)?;
        let candidates = self.probe_chunk_key_blooms(&blooms, needles)?;
        Some((
            candidates,
            blooms.first().map(|(_, bloom)| Arc::clone(&bloom.device)),
        ))
    }

    /// Get-or-build the key indexes for EVERY chunk of a class entry (entry-time in P5-2; the
    /// direct-call gate uses it now). Returns per-chunk (chunk_id, index) in entry order, or
    /// `None` if any chunk declines. Cap policy: evict the least-recently-used entries of OTHER
    /// chunks until the new total fits (class-agnostic — index buffers are small).
    pub(crate) fn ensure_chunk_key_indexes(
        &self,
        table: &RelationalTable,
        entry: &Arc<ColdTableChunks>,
        key_positions: &[usize],
        key_id: usize,
    ) -> Option<Vec<(usize, ChunkKeyIndex)>> {
        // Each element pairs the index with its ENTRY POSITION — empty (fully-compacted) chunks
        // are skipped, so the probe's shard_idx indexes THIS vec, and the caller translates back
        // through the position (never `entry.chunks[shard_idx]` directly: position drift).
        let residency = &self.read_state.residency;
        let mut out: Vec<(usize, ChunkKeyIndex)> = Vec::with_capacity(entry.chunks.len());
        for (position, chunk) in entry.chunks.iter().enumerate() {
            if chunk.row_count == 0 {
                continue;
            }
            let cache_key = (table.name.clone(), chunk.chunk_id, key_id);
            let cached = {
                let mut map = residency
                    .chunk_key_index
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                map.get_mut(&cache_key).map(|e| {
                    e.last_used = residency
                        .chunk_key_index_clock
                        .fetch_add(1, Ordering::Relaxed);
                    e.clone()
                })
            };
            let index = match cached {
                Some(index) => index,
                None => {
                    let built = self.build_chunk_key_index(table, chunk, key_positions)?;
                    let mut map = residency
                        .chunk_key_index
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    // Recheck under the lock (the double-build race: keep the first).
                    let entry_ref = map.entry(cache_key).or_insert_with(|| {
                        residency
                            .chunk_key_index_bytes
                            .fetch_add(built.bytes, Ordering::Relaxed);
                        built
                    });
                    entry_ref.last_used = residency
                        .chunk_key_index_clock
                        .fetch_add(1, Ordering::Relaxed);
                    let got = entry_ref.clone();
                    // Cap: evict LRU entries (never the just-inserted key) until under the cap.
                    let just_inserted = (table.name.clone(), chunk.chunk_id, key_id);
                    let mut total = residency.chunk_key_index_bytes.load(Ordering::Relaxed);
                    while total > chunk_key_index_cap_bytes() {
                        let victim = map
                            .iter()
                            .filter(|(k, _)| **k != just_inserted)
                            .min_by_key(|(_, e)| e.last_used)
                            .map(|(k, _)| k.clone());
                        let Some(victim) = victim else { break };
                        if let Some(evicted) = map.remove(&victim) {
                            total = residency
                                .chunk_key_index_bytes
                                .fetch_sub(evicted.bytes, Ordering::Relaxed)
                                .saturating_sub(evicted.bytes);
                        }
                    }
                    got
                }
            };
            out.push((position, index));
        }
        Some(out)
    }

    /// Probe every chunk index with the needle fingerprints — ONE multi-chunk write-locate
    /// launch. Returns per-needle (position-in-entry, slot) hits (the caller translates through
    /// ITS entry Arc; never cache positions across entries).
    /// The SHARED needle derivation (audit LOW — the build/probe parity contract): a probe
    /// needle MUST be derived exactly as the build derived its keys — the RAW i32 for a single
    /// int4-class key, the host `compound_key_fingerprint` over `sql_value_key_words` for every
    /// other shape (byte-identical to the device fold). A mismatch is a silent all-miss = a
    /// false-negative duplicate = the C2 RPO hazard.
    pub(crate) fn chunk_key_needle(
        table: &RelationalTable,
        key_positions: &[usize],
        values: &[SqlValue],
    ) -> Option<i32> {
        if key_positions.len() == 1
            && matches!(
                table.columns[key_positions[0]].ty,
                SqlType::Int4 | SqlType::Date | SqlType::Int2
            )
        {
            return match values.get(key_positions[0])? {
                SqlValue::Int4(v) => Some(*v),
                SqlValue::Date(v) => Some(*v),
                SqlValue::Int2(v) => Some(i32::from(*v)),
                _ => None,
            };
        }
        let mut words: Vec<i32> = Vec::new();
        for &pos in key_positions {
            let column = table.columns.get(pos)?;
            words.extend(crate::engine_residency::sql_value_key_words(
                column.ty,
                values.get(pos)?,
            )?);
        }
        Some(crate::engine_residency::compound_key_fingerprint(&words))
    }

    /// The candidate-index needle including NULL payload placeholders. Chunk fingerprints are
    /// built from raw device values and intentionally ignore validity; the payload encoder writes
    /// zero fixed-width words (or an empty text span) for NULL. Reproduce that representation so
    /// a NULL-bearing exact tuple can still use the index as a no-false-negative candidate
    /// selector; the following device `IS NULL` predicate remains authoritative.
    fn chunk_key_candidate_needle(
        table: &RelationalTable,
        key_positions: &[usize],
        values: &[SqlValue],
    ) -> Option<i32> {
        if key_positions.len() == 1
            && matches!(
                table.columns[key_positions[0]].ty,
                SqlType::Int4 | SqlType::Date | SqlType::Int2
            )
        {
            return match values.get(key_positions[0])? {
                SqlValue::Null => Some(0),
                _ => Self::chunk_key_needle(table, key_positions, values),
            };
        }
        let mut words: Vec<i32> = Vec::new();
        for &position in key_positions {
            let column = table.columns.get(position)?;
            match values.get(position)? {
                SqlValue::Null => match column.ty {
                    SqlType::Text => words.extend(
                        crate::engine_residency::sql_value_key_words(
                            SqlType::Text,
                            &SqlValue::Text(String::new()),
                        )?,
                    ),
                    SqlType::Int2 | SqlType::Int4 | SqlType::Date => words.push(0),
                    SqlType::Int8 | SqlType::Timestamp => words.extend([0, 0]),
                    SqlType::Numeric { .. } | SqlType::Uuid => words.extend([0, 0, 0, 0]),
                    SqlType::Bool => return None,
                },
                value => words.extend(crate::engine_residency::sql_value_key_words(
                    column.ty, value,
                )?),
            }
        }
        Some(crate::engine_residency::compound_key_fingerprint(&words))
    }

    pub(crate) fn probe_chunk_key_indexes(
        &self,
        indexes: &[(usize, ChunkKeyIndex)],
        needles: &[i32],
    ) -> Option<Vec<Vec<(usize, u32)>>> {
        // Hits translate shard_idx -> the paired ENTRY position before returning.
        if indexes.is_empty() || needles.is_empty() {
            return Some(vec![Vec::new(); needles.len()]);
        }
        let shards: Vec<gpu_db_execution::WriteLocateShard> = indexes
            .iter()
            .map(|(_, index)| gpu_db_execution::WriteLocateShard {
                index: Arc::clone(&index.device),
                table_mask: index.table_mask,
                hash_shift: index.hash_shift,
                row_count: index.row_count,
            })
            .collect();
        let ctx = Arc::clone(&shards[0].index);
        let result = ctx
            .submit_multi_shard_i32_write_locate(&shards, needles, 8)
            .ok()?;
        if result.count.len() != needles.len() {
            return None;
        }
        let mut out: Vec<Vec<(usize, u32)>> = Vec::with_capacity(needles.len());
        for n in 0..needles.len() {
            let count = result.count[n];
            if count == u32::MAX {
                // Overflow: more same-fingerprint hits than the window. The kernel MUST set the
                // u32::MAX sentinel (never truncate) — P5-3 made this load-bearing for DML: a
                // truncated window would be a silently MISSED DML match (data loss), not just a
                // missed uniqueness conflict. Decline -> the fold path serves the statement.
                return None;
            }
            let mut hits = Vec::with_capacity(count as usize);
            for h in 0..count as usize {
                let flat = n * result.max_hits as usize + h;
                let filtered = *result.shard_idx.get(flat)? as usize;
                hits.push((indexes.get(filtered)?.0, *result.slot.get(flat)?));
            }
            out.push(hits);
        }
        Some(out)
    }

    /// Build the exact equality predicate for one unique-key tuple. The statement values are
    /// already type-coerced by bind. This engine's current unique semantics are structural
    /// (`NULL == NULL`), so NULL key components lower to the device validity-mask `IS NULL`
    /// leaf rather than SQL `=` (which would be UNKNOWN).
    fn class_exact_key_predicate(
        table: &RelationalTable,
        positions: &[usize],
        row: &[SqlValue],
    ) -> Option<crate::engine_expr::ResidentExpr> {
        use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};
        let mut predicate: Option<ResidentExpr> = None;
        for &position in positions {
            let value = row.get(position)?.clone();
            let leaf = if matches!(value, SqlValue::Null) {
                ResidentExpr::IsNull {
                    col: position,
                    is_not_null: false,
                }
            } else {
                crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(
                    table,
                    &[vec![(position, SelectFilterOp::Eq, value)]],
                )?
            };
            predicate = Some(match predicate {
                None => leaf,
                Some(lhs) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(lhs),
                    rhs: Box::new(leaf),
                },
            });
        }
        predicate
    }

    /// Exact within-statement unique validation over a transient DEVICE relation. Every bound
    /// key tuple runs as a complete device predicate; there is no host grouping, NULL branch, or
    /// value comparison. Survivor coordinates feed the device threshold kernel, whose status bit
    /// is the final verdict readback. This is deliberately bounded until the device exact
    /// tuple-hash/group operator replaces it.
    fn validate_class_new_rows_unique_on_device(
        &self,
        table: &RelationalTable,
        new_rows: &[Vec<SqlValue>],
    ) -> Option<Result<(), EngineError>> {
        if new_rows.len() < 2 {
            return Some(Ok(()));
        }
        if new_rows.len() > CLASS_DEVICE_UNIQUE_BATCH_MAX_ROWS {
            return None;
        }
        let (snapshot, memory) = self
            .build_transient_relation_residency(table, new_rows)
            .ok()?;
        let src = ResidentExecSource {
            descriptor: Arc::new(snapshot),
            device_memory: Arc::new(memory),
            row_count: new_rows.len() as u64,
        };
        for index in table.indexes.iter().filter(|index| index.unique) {
            let Some(positions) =
                crate::engine_residency::index_key_column_positions(table, index)
            else {
                continue;
            };
            for row in new_rows {
                let predicate = Self::class_exact_key_predicate(
                    table,
                    &positions,
                    row,
                )?;
                let slots = self
                    .lower_resident_predicate(
                        &predicate,
                        table,
                        &src.descriptor,
                        &src.device_memory,
                        src.row_count,
                        None,
                    )
                    .ok()?;
                self.read_state
                    .residency
                    .chunk_class_device_exact_rechecks
                    .fetch_add(1, Ordering::Relaxed);
                let candidates: Vec<u64> = slots.into_iter().map(u64::from).collect();
                let duplicate = src
                    .device_memory
                    .unique_coordinate_threshold_reached(&candidates, &[], 2)
                    .ok()?;
                if duplicate {
                    return Some(Err(EngineError::ApplyFailed(format!(
                        "duplicate key value violates unique index \"{}\"",
                        index.name
                    ))));
                }
            }
        }
        Some(Ok(()))
    }

    /// P5-2 (S-E.P5) — the KEYED-CLASS uniqueness preflight: validate a statement's NEW key
    /// images against a chunk-authoritative table ON-DEVICE. Every host validator at the call
    /// sites sees the RECLAIMED (empty) store and passes VACUOUSLY — and a vacuous accept is the
    /// C2 hazard: a WAL-durable duplicate that recovery's host-path replay then REJECTS, i.e. an
    /// unreplayable acked commit. In-batch duplicates are checked over a transient device
    /// relation; existing-row conflicts probe the per-chunk indexes (ONE multi-chunk locate per
    /// unique index), then run exact key equality + visibility through the device predicate VM.
    /// A tombstoned slot is NOT a conflict, and a fingerprint collision fails exact equality.
    ///
    /// `exclude` — the C1 UPDATE self-exclusion: (the update's own located PACKED coordinates,
    /// the resolve-time entry epoch). An update's old version is LIVE at probe time (stamps land
    /// in the commit hook), so its own coordinates are SELF, not conflicts; the epoch must still
    /// match the probed entry or the coordinates may be misaligned (decline, never guess).
    ///
    /// `Some(Ok)` = validated; `Some(Err)` = duplicate (a statement error — the class stays);
    /// `None` = DECLINE, the caller must DE-AUTHORITIZE and fall through to host validation.
    /// NULL keys use the raw-payload placeholder fingerprint only to choose candidate chunks,
    /// then run an exact device `IS NULL` predicate, preserving structural NULL uniqueness
    /// without de-authorizing. Declines: an unfoldable needle, epoch drift, or a
    /// build/probe/stage/device error.
    pub(crate) fn validate_class_insert_uniqueness(
        &self,
        table: &RelationalTable,
        new_rows: &[Vec<SqlValue>],
        rtx: Index,
        exclude: Option<(&std::collections::BTreeSet<u64>, u64)>,
    ) -> Option<Result<(), EngineError>> {
        if new_rows.is_empty() {
            return Some(Ok(()));
        }
        let mut keyed: Vec<(usize, String, Vec<usize>)> = Vec::new();
        for (key_id, index) in table.indexes.iter().enumerate() {
            if !index.unique {
                continue;
            }
            // The host validator SKIPS an index whose key positions do not resolve
            // (`validate_unique_indexes_for_rows`) — mirror it exactly: parity, not strictness.
            let Some(positions) = crate::engine_residency::index_key_column_positions(table, index)
            else {
                continue;
            };
            keyed.push((key_id, index.name.clone(), positions));
        }
        if keyed.is_empty() {
            return Some(Ok(()));
        }
        if let Err(err) = self.validate_class_new_rows_unique_on_device(table, new_rows)? {
            return Some(Err(err));
        }
        let entry = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(&table.name)
            .cloned()?;
        if let Some((_, epoch)) = exclude {
            if entry.entry_epoch != epoch {
                return None;
            }
        }
        // Pure marshaling for the device verdict below: these are prepare-time packed coordinates,
        // not host-decoded values or a host-side membership oracle.
        let excluded_coordinates: Vec<u64> = exclude
            .map(|(set, _)| set.iter().copied().collect())
            .unwrap_or_default();
        for (key_id, index_name, positions) in &keyed {
            let needles: Vec<i32> = new_rows
                .iter()
                .map(|row| Self::chunk_key_candidate_needle(table, positions, row))
                .collect::<Option<Vec<_>>>()?;
            let (hits, verdict_device) = self.chunk_key_candidate_positions(
                table,
                &entry,
                positions,
                *key_id,
                &needles,
            )?;
            for (needle_idx, needle_hits) in hits.iter().enumerate() {
                let candidate_positions: std::collections::BTreeSet<usize> = needle_hits
                    .iter()
                    .copied()
                    .collect();
                if candidate_positions.is_empty() {
                    continue;
                }
                let predicate = Self::class_exact_key_predicate(
                    table,
                    positions,
                    new_rows.get(needle_idx)?,
                )?;
                let exact = self.locate_streaming_cold_slots_in_entry(
                    table,
                    &predicate,
                    rtx,
                    &entry,
                    Some(&candidate_positions),
                )?;
                let candidates: Vec<u64> = exact
                    .into_iter()
                    .flat_map(|(position, slots)| {
                        slots.into_iter().map(move |slot| {
                            ((position as u64) << 32) | u64::from(slot)
                        })
                    })
                    .collect();
                let conflict = verdict_device
                    .as_ref()?
                    .unique_coordinate_threshold_reached(
                        &candidates,
                        &excluded_coordinates,
                        1,
                    )
                    .ok()?;
                if conflict {
                    self.read_state
                        .residency
                        .chunk_class_unique_probe_conflicts
                        .fetch_add(1, Ordering::Relaxed);
                    return Some(Err(EngineError::ApplyFailed(format!(
                        "duplicate key value violates unique index \"{index_name}\""
                    ))));
                }
            }
        }
        self.read_state
            .residency
            .chunk_class_unique_probes
            .fetch_add(1, Ordering::Relaxed);
        Some(Ok(()))
    }

    /// P5-2 telemetry: keyed-class uniqueness preflights served on-device (non-vacuity).
    pub fn chunk_class_unique_probes(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_unique_probes
            .load(Ordering::Relaxed)
    }
    /// P5-2 telemetry: probe-rejected duplicates (recheck-confirmed conflicts).
    pub fn chunk_class_unique_probe_conflicts(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_unique_probe_conflicts
            .load(Ordering::Relaxed)
    }
    /// P5-3 telemetry: class DML statements whose locate RAN through the key-index probe —
    /// counts probe-eligible executions (including 0-hit misses and residual-filtered-out
    /// statements), NOT rows located.
    pub fn chunk_class_dml_key_locates(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_dml_key_locates
            .load(Ordering::Relaxed)
    }

    /// Candidate-routing launches served by compact all-chunk Bloom filters because the exact
    /// retained key-index set exceeded its VRAM cap.
    pub fn chunk_key_bloom_probes(&self) -> u64 {
        self.read_state
            .residency
            .chunk_key_bloom_probes
            .load(Ordering::Relaxed)
    }

    pub fn chunk_key_bloom_bytes(&self) -> u64 {
        self.read_state
            .residency
            .chunk_key_bloom_bytes
            .load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn stale_chunk_key_candidate_count(&self, table_name: &str) -> usize {
        let live: std::collections::BTreeSet<u64> = self
            .read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .map(|entry| entry.chunks.iter().map(|chunk| chunk.chunk_id).collect())
            .unwrap_or_default();
        let residency = &self.read_state.residency;
        let indexes = residency
            .chunk_key_index
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .keys()
            .filter(|(table, chunk_id, _)| table == table_name && !live.contains(chunk_id))
            .count();
        let blooms = residency
            .chunk_key_bloom
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .keys()
            .filter(|(table, chunk_id, _)| table == table_name && !live.contains(chunk_id))
            .count();
        indexes + blooms
    }

    /// Candidate-index and structural-NULL validations whose authoritative exact predicate
    /// completed on-device.
    pub fn chunk_class_device_exact_rechecks(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_device_exact_rechecks
            .load(Ordering::Relaxed)
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
    /// P4 reclamation telemetry: host store versions deleted at class entry.
    pub fn chunk_class_reclaimed_rows(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_reclaimed_rows
            .load(Ordering::Relaxed)
    }
    /// P4 compaction telemetry.
    pub fn chunk_class_compactions(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_compactions
            .load(Ordering::Relaxed)
    }
    pub fn chunk_class_compacted_slots(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_compacted_slots
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

fn select_streaming_execution_gpus(
    default_gpu: u16,
    query_budget: u64,
    physical_gpu_ids: &[u16],
    unavailable_gpu_ids: &[u16],
    memory_pressured_gpu_ids: &[u16],
    budgets: &std::collections::BTreeMap<u16, u64>,
) -> Vec<u16> {
    let eligible = |gpu_id: u16| {
        !unavailable_gpu_ids.contains(&gpu_id)
            && !memory_pressured_gpu_ids.contains(&gpu_id)
            && budgets
                .get(&gpu_id)
                .is_some_and(|budget| *budget >= query_budget)
    };
    if !physical_gpu_ids.contains(&default_gpu) || !eligible(default_gpu) {
        return vec![default_gpu];
    }
    let mut gpus = vec![default_gpu];
    gpus.extend(
        physical_gpu_ids
            .iter()
            .copied()
            .filter(|&gpu_id| gpu_id != default_gpu && eligible(gpu_id)),
    );
    gpus
}

// ---------------- P5: FINAL SLOT READBACK (one device-approved slot) ----------------
//
// Visibility and predicates have ALREADY been decided by `lower_resident_predicate`; this helper
// only performs the charter-sanctioned final readback of an approved row image. It MUST NOT inspect
// born/tombstone metadata or compare values — doing so would turn the host back into an execution
// tier. It reads only one slot from the already-staged device chunk, never the host cold payload.

impl Engine {
    /// CALLER CONTRACT: `src` is the staged buffer of THIS `chunk`, and `slot` came from an
    /// exact device predicate using the same source + snapshot visibility. Returns `None` only
    /// on a read/layout failure; the caller conservatively declines.
    pub(crate) fn read_cold_chunk_slot_values(
        &self,
        table: &RelationalTable,
        chunk: &ColdChunk,
        src: &crate::engine_expr::ResidentExecSource,
        slot: usize,
    ) -> Option<Vec<SqlValue>> {
        use crate::relational_model::{
            resident_device_bool_column_offset, resident_device_int4_column_offset,
            resident_device_int8_column_offset, resident_device_numeric_column_offset,
            resident_device_text_column_layout,
        };
        if slot >= chunk.row_count as usize {
            return None;
        }
        let d = &chunk.snapshot;
        let memory = &src.device_memory;
        let null_bit = |name: &str| -> Option<bool> {
            // 1 = valid; absent bitmap = all valid.
            match d
                .resident_device_null_columns
                .iter()
                .find(|n| n.name == name)
            {
                None => Some(true),
                Some(layout) => {
                    let word_off = layout.bitmap_byte_offset + ((slot as u64 / 32) * 4);
                    let words = memory.read_resident_i32_column(word_off, 1).ok()?;
                    Some((words.first().copied()? as u32 >> (slot % 32)) & 1 == 1)
                }
            }
        };
        let mut row: Vec<SqlValue> = Vec::with_capacity(table.columns.len());
        for (idx, column) in table.columns.iter().enumerate() {
            if !null_bit(&column.name)? {
                row.push(SqlValue::Null);
                continue;
            }
            let value = match column.ty {
                SqlType::Int4 | SqlType::Date | SqlType::Int2 => {
                    let base = resident_device_int4_column_offset(d, table, idx).ok()?;
                    let v = *memory
                        .read_resident_i32_column(base + (slot as u64) * 4, 1)
                        .ok()?
                        .first()?;
                    match column.ty {
                        SqlType::Date => SqlValue::Date(v),
                        SqlType::Int2 => SqlValue::Int2(v as i16),
                        _ => SqlValue::Int4(v),
                    }
                }
                SqlType::Int8 | SqlType::Timestamp => {
                    let base = resident_device_int8_column_offset(d, table, idx).ok()?;
                    let halves = memory
                        .read_resident_i32_column(base + (slot as u64) * 8, 2)
                        .ok()?;
                    let v = ((*halves.first()? as u32 as u64)
                        | ((*halves.get(1)? as u32 as u64) << 32))
                        as i64;
                    if column.ty == SqlType::Timestamp {
                        SqlValue::Timestamp(v)
                    } else {
                        SqlValue::Int8(v)
                    }
                }
                SqlType::Numeric { .. } | SqlType::Uuid => {
                    let base = resident_device_numeric_column_offset(d, table, idx).ok()?;
                    let words = memory
                        .read_resident_i32_column(base + (slot as u64) * 16, 4)
                        .ok()?;
                    let mut raw = [0u8; 16];
                    for (w, word) in words.iter().enumerate() {
                        raw[w * 4..w * 4 + 4].copy_from_slice(&word.to_le_bytes());
                    }
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
                    let base = resident_device_bool_column_offset(d, table, idx).ok()?;
                    let word_off = base + ((slot as u64 / 32) * 4);
                    let words = memory.read_resident_i32_column(word_off, 1).ok()?;
                    SqlValue::Bool((*words.first()? as u32 >> (slot % 32)) & 1 == 1)
                }
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(d, table, idx).ok()?;
                    let bounds = memory
                        .read_resident_u64_column(layout.offsets_byte_offset + (slot as u64) * 8, 2)
                        .ok()?;
                    let (lo, hi) = (*bounds.first()?, *bounds.get(1)?);
                    // Audit LOW: corrupt bounds (hi < lo) DECLINE like the host decoder — never
                    // a debug-panic on the subtraction.
                    let span_len = hi.checked_sub(lo)? as usize;
                    let span = memory
                        .read_resident_bytes(layout.bytes_byte_offset + lo, span_len)
                        .ok()?;
                    SqlValue::Text(String::from_utf8(span).ok()?)
                }
            };
            row.push(value);
        }
        Some(row)
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
                    let lo = read_u64(layout.offsets_byte_offset as usize + slot * 8)? as usize;
                    let hi =
                        read_u64(layout.offsets_byte_offset as usize + (slot + 1) * 8)? as usize;
                    let span = bytes
                        .get(
                            layout.bytes_byte_offset as usize + lo
                                ..layout.bytes_byte_offset as usize + hi,
                        )
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

#[cfg(test)]
mod streaming_scheduler_tests {
    use super::select_streaming_execution_gpus;
    use std::collections::BTreeMap;

    #[test]
    fn multi_gpu_scheduler_requires_physical_health_and_full_query_budget() {
        let budgets = BTreeMap::from([(0, 4096), (1, 4096), (2, 2048), (3, 8192)]);
        assert_eq!(
            select_streaming_execution_gpus(0, 4096, &[0, 1, 2, 3], &[], &[3], &budgets),
            vec![0, 1],
            "under-budget and pressured devices must not receive a chunk"
        );
        assert_eq!(
            select_streaming_execution_gpus(0, 4096, &[0, 1], &[1], &[], &budgets),
            vec![0],
            "an unavailable secondary must be excluded"
        );
        assert_eq!(
            select_streaming_execution_gpus(7, 4096, &[0, 1], &[], &[], &budgets),
            vec![7],
            "the coordinator fallback preserves the existing error surface when no GPU is eligible"
        );
    }
}
