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
//! "over-VRAM", so another GPU route must accept the read or it fails loudly.
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
mod streaming_chunk_keys;
mod streaming_cold_admission;
mod streaming_cold_lifecycle;
mod streaming_dml_class;
mod streaming_grouped_fold;
mod streaming_join;
mod streaming_ordered_fold;
mod streaming_projection_fold;
mod streaming_reduction_fold;
mod streaming_select_route;
mod streaming_transaction_cow;

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

/// A chunk's device reduction either combined cleanly, hit a shape the streaming path must decline
/// through the fail-loud GPU-required boundary (`Defer`), or produced a genuine SQL error (`Hard`).
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
pub(crate) type ClassDmlUpdate = (ClassDmlMatches, Vec<Vec<SqlValue>>);
pub(crate) type ClassDmlUpdateWithEpoch = (ClassDmlMatches, Vec<Vec<SqlValue>>, u64);

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
type ClassResolvePinHook = (Arc<std::sync::Barrier>, Arc<std::sync::Barrier>);

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

/// Test-only policy override for store-driven cold-cache and resident-baseline gates. Production
/// class entry is unconditional; ignored GPU tests run serially and restore this flag on drop.
#[cfg(test)]
pub(crate) static CHUNK_CLASS_ENTRY_ENABLED_TEST: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// P5-1: the chunk CONTENT-identity allocator (fresh iff the payload bytes are new).
static COLD_CHUNK_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// P4-2b-ii: the global entry-epoch allocator (never reused; process-local like the class map).
static COLD_ENTRY_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// One cached chunk: the exact device payload bytes + the descriptor template the build produced.
pub(crate) struct ColdChunk {
    payload: ColdPayload,
    pub(crate) snapshot: RelationalResidencySnapshot,
    pub(crate) row_count: u64,
    /// Stable logical entity identity, parallel to payload slots once a table becomes
    /// chunk-authoritative. Ordinary cache entries leave this empty; class entry resolves the
    /// canonical row keys from the still-pinned store before reclaiming it, and every tail/update
    /// carries the same identities forward. Placement remains `(entry_epoch, chunk, slot)`.
    pub(crate) entity_ids: Arc<Vec<u64>>,
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
            entity_ids: Arc::new(Vec::new()),
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
    // fails -> poison -> skip the incomplete cold-cache install; O_EXCL already blocks anything
    // worse — audit LOW).
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
/// A device reduction error that is a genuine arithmetic OVERFLOW (matched on the executor's stable PG
/// overflow phrases). Such an error must surface through the GPU-required boundary (audit Finding 2).
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

    /// Device-authoritative commits maintained through the chunk-tail path.
    pub fn chunk_class_device_commits(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_device_commits
            .load(Ordering::Relaxed)
    }
    /// Current number of chunk-authoritative relations.
    pub fn chunk_class_entries(&self) -> u64 {
        self.read_state
            .residency
            .chunk_authoritative_tables
            .load()
            .len() as u64
    }
    /// P4 compaction telemetry.
    #[cfg(test)]
    pub fn chunk_class_compactions(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_compactions
            .load(Ordering::Relaxed)
    }
    #[cfg(test)]
    pub fn chunk_class_compacted_slots(&self) -> u64 {
        self.read_state
            .residency
            .chunk_class_compacted_slots
            .load(Ordering::Relaxed)
    }
    /// Explicit RETIRE-002 repair exits from chunk authority.
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
    /// publishes each record's own index). The live watermark must equal `boundary_index`; a
    /// one-high watermark represents a genuinely newer commit and must abort the artifact. A table
    /// qualifies when its entry's pinned generation is ptr-equal current
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
    ) -> Result<usize, EngineError> {
        use std::io::Write;
        let path = streaming_cold_checkpoint_path(base, cut);
        let watermark = self.committed_seq();
        if watermark != boundary_index {
            // Not quiesced at the cut (a commit raced the checkpoint): skip — the artifact would
            // never match the recovery seam. Stale artifacts from older cuts still get swept.
            remove_stale_cold_checkpoints(base, cut)?;
            return Ok(0);
        }
        let map = self.read_state.residency.streaming_cold_chunks.load();
        let mut qualified: Vec<(String, Arc<ColdTableChunks>)> = Vec::new();
        for (name, entry) in map.iter() {
            let current = self.read_state.mvcc.table_rows(name).generation_payload();
            let entry = if Arc::ptr_eq(&entry.generation, &current) {
                // Untouched since its settled install: every live stamp is <= boundary (the
                // watermark guard above), so its content is the boundary state.
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
            remove_stale_cold_checkpoints(base, cut)?;
            return Ok(0);
        }
        if qualified.is_empty() {
            remove_stale_cold_checkpoints(base, cut)?;
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
            hash: {
                use sha2::Digest as _;
                sha2::Sha256::new()
            },
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
            use sha2::Digest as _;
            let hash = w.hash.finalize();
            w.inner.write_all(&hash)?;
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
        gpu_db_wal::sync_wal_parent_dir(&path)?;
        remove_stale_cold_checkpoints(base, cut)?;
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
        // PASS 1: verify the collision-resistant trailer over the whole stream (bounded RAM), then re-read and
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
            if &magic != COLD_CHECKPOINT_MAGIC && &magic != COLD_CHECKPOINT_MAGIC_V2 {
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
const COLD_CHECKPOINT_MAGIC_V2: &[u8; 15] = b"GPUDBCOLDCKPT2\n";
const COLD_CHECKPOINT_MAGIC: &[u8; 15] = b"GPUDBCOLDCKPT3\n";

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
pub(crate) fn remove_stale_cold_checkpoints(
    base: &std::path::Path,
    keep_cut: u64,
) -> Result<(), EngineError> {
    let Some(parent) = base.parent() else {
        return Ok(());
    };
    let Some(stem) = base.file_name() else {
        return Ok(());
    };
    let prefix = format!("{}.cold-checkpoint.", stem.to_string_lossy());
    let keep = keep_cut.to_string();
    let entries = std::fs::read_dir(parent).map_err(|err| {
        EngineError::Durability(format!(
            "cold checkpoint: enumerate {} failed: {err}",
            parent.display()
        ))
    })?;
    let mut removed = false;
    for entry in entries {
        let entry = entry.map_err(|err| {
            EngineError::Durability(format!(
                "cold checkpoint: enumerate entry in {} failed: {err}",
                parent.display()
            ))
        })?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(suffix) = name.strip_prefix(&prefix) {
            if suffix != keep {
                std::fs::remove_file(entry.path()).map_err(|err| {
                    EngineError::Durability(format!(
                        "cold checkpoint: remove stale artifact {} failed: {err}",
                        entry.path().display()
                    ))
                })?;
                removed = true;
            }
        }
    }
    if removed {
        gpu_db_wal::sync_wal_parent_dir(base)?;
    }
    Ok(())
}

/// SHA-256 hashing writer: every byte written folds into the running trailer authority.
pub(crate) struct ColdCkptWriter<W: std::io::Write> {
    pub(crate) inner: W,
    pub(crate) hash: sha2::Sha256,
}

impl<W: std::io::Write> ColdCkptWriter<W> {
    fn put(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        use sha2::Digest as _;
        self.hash.update(bytes);
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

/// Verify the v3 SHA-256 trailer (or legacy v2 FNV trailer). Streams in 64KiB blocks.
fn cold_checkpoint_checksum_ok(file: &mut std::fs::File) -> bool {
    use sha2::Digest as _;
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
    let mut magic = [0_u8; COLD_CHECKPOINT_MAGIC.len()];
    if file.read_exact(&mut magic).is_err() || file.seek(std::io::SeekFrom::Start(0)).is_err() {
        return false;
    }
    let trailer_len = if &magic == COLD_CHECKPOINT_MAGIC {
        32
    } else if &magic == COLD_CHECKPOINT_MAGIC_V2 {
        8
    } else {
        return false;
    };
    let body = total - trailer_len;
    let mut remaining = body;
    let mut buf = vec![0u8; 64 << 10];
    let mut sha = sha2::Sha256::new();
    let mut fnv = 0xcbf29ce484222325_u64;
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        if file.read_exact(&mut buf[..want]).is_err() {
            return false;
        }
        if trailer_len == 32 {
            use sha2::Digest as _;
            sha.update(&buf[..want]);
        } else {
            for b in &buf[..want] {
                fnv = (fnv ^ u64::from(*b)).wrapping_mul(0x100000001b3);
            }
        }
        remaining -= want as u64;
    }
    if trailer_len == 32 {
        use sha2::Digest as _;
        let mut trailer = [0_u8; 32];
        file.read_exact(&mut trailer).is_ok() && trailer == sha.finalize().as_slice()
    } else {
        let mut trailer = [0_u8; 8];
        file.read_exact(&mut trailer).is_ok() && fnv == u64::from_le_bytes(trailer)
    }
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
