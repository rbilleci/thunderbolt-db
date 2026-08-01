use std::fmt;
use std::os::raw::c_void;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use libloading::Library;

use crate::cuda_context::{check_cuda, CudaExternalAllocationReservation, GpuPrimaryContext};
use crate::{
    launch_cuda_resident_row_count, launch_cuda_resident_visible_count,
    launch_cuda_resident_visible_digest, CudaDeviceMemoryProof, CudaRuntimeProbeError,
    CudaVisibleDigestColumn, CudaVisibleSourceDigest,
};

mod prepared_u64_publish;
#[cfg(test)]
pub(crate) use prepared_u64_publish::fail_next_prepared_u64_htod_publication;
pub use prepared_u64_publish::PreparedU64HtoDPublication;

pub struct CudaResidentDeviceMemory {
    pub(super) metadata: CudaDeviceMemoryProof,
    pub(super) device_ptr: u64,
    /// The shared primary context this allocation lives in (plan §9.3). Residency owns no
    /// context of its own; it holds a refcount so the context outlives the allocation, and
    /// reaches the device pointer, library, module cache, and stream pool through it.
    pub(super) primary: Arc<GpuPrimaryContext>,
    /// Shared allocation lifetime. Read views clone this guard, so every safe owner or view that can
    /// submit a kernel keeps the exact device pointer allocated until its last clone is dropped.
    allocation: Arc<CudaResidentDeviceAllocation>,
    /// A provenance bit, not an accounting counter: it is set only by a constructor which has
    /// established that every byte of this allocation was initialized contiguously. In particular,
    /// `copied_bytes == allocated_bytes` is insufficient because chunk uploads can overlap.
    full_contiguous_initialization: AtomicBool,
    pub(super) last_kernel_event_elapsed_us: Mutex<Option<u64>>,
}

#[derive(Clone)]
pub struct CudaResidentDeviceMemoryReadView {
    metadata: CudaDeviceMemoryProof,
    device_ptr: u64,
    primary: Arc<GpuPrimaryContext>,
    allocation: Arc<CudaResidentDeviceAllocation>,
}

pub(super) struct CudaResidentDeviceAllocation {
    device_ptr: u64,
    primary: Arc<GpuPrimaryContext>,
    /// Present only for an explicitly scoped transient retained allocation. Long-lived residency
    /// keeps using the engine's canonical accounting; transactional cold validation opts into this
    /// owner so its source/recompaction bytes share one allocator-backed scratch high-water.
    _scope_reservation: Option<CudaExternalAllocationReservation>,
}

/// Test-only non-owning witness for the exact resident allocation. This lets lifetime regressions
/// prove that a read view and a deferred submission retain the allocation itself, independently of
/// CUDA's synchronizing `cuMemFree` behavior.
#[cfg(test)]
pub(crate) struct CudaResidentAllocationWeak(std::sync::Weak<CudaResidentDeviceAllocation>);

#[cfg(test)]
impl CudaResidentAllocationWeak {
    pub(crate) fn is_alive(&self) -> bool {
        self.0.upgrade().is_some()
    }
}

impl Drop for CudaResidentDeviceAllocation {
    fn drop(&mut self) {
        // The allocation is freed only after the last owner/read-view guard drops. Bind the shared
        // primary context first because that last guard may be released on an otherwise unbound thread.
        let _ = self.primary.set_current();
        unsafe {
            (self.primary.cu_mem_free)(self.device_ptr);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaDeviceMemoryChunk<'a> {
    pub byte_offset: u64,
    pub bytes: &'a [u8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaOwnedDeviceMemoryChunk {
    pub byte_offset: u64,
    pub bytes: Vec<u8>,
}

/// One device-to-device copy in an on-device recompaction (S10c slice 2a): copy `byte_len` bytes
/// from `src_device_ptr + src_byte_offset` (a SOURCE resident allocation, e.g. one partition's SoA
/// buffer) to `dst_byte_offset` within the freshly-allocated unified buffer. `byte_len == 0` is a
/// no-op (skipped). The host never touches the bytes — the copy is a `cuMemcpyDtoD` inside the
/// shared primary context, so the unified buffer is built fully on-device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecompactSegment {
    pub src_device_ptr: u64,
    pub src_byte_offset: u64,
    pub dst_byte_offset: u64,
    pub byte_len: u64,
}

/// SV3 (MVCC visibility): fill `len` bytes of the freshly-allocated unified buffer at `byte_offset` with
/// `fill_byte` (a `cuMemsetD8`) BEFORE the segment copies run. The recompaction buffer is `cuMemAlloc`'d
/// (uninitialized), so a section not fully covered by segments would read garbage; a fill initializes it.
/// Use `0x7F` to make a gathered `deleted_by` section born all-live (each u64 = `0x7F7F_7F7F_7F7F_7F7F`
/// ≈ 9.1e18, a LARGE POSITIVE i64 greater than every real commit `Index`), so delete-free shards' rows
/// (which contribute no `deleted_by` segment) read as live and versioned shards' segments overwrite only
/// their deleted slots. Do NOT use `0xFF`/`u64::MAX`: the read visibility compare `deleted_by > read_txn_id`
/// is SIGNED (s64), so `u64::MAX` = -1 and a live row would wrongly FAIL the compare (hidden). `len == 0`
/// is a no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecompactFill {
    pub byte_offset: u64,
    pub len: u64,
    pub fill_byte: u8,
}

impl fmt::Debug for CudaResidentDeviceMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaResidentDeviceMemory")
            .field("metadata", &self.metadata)
            .field("device_ptr", &self.device_ptr)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for CudaResidentDeviceMemoryReadView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaResidentDeviceMemoryReadView")
            .field("metadata", &self.metadata)
            .field("device_ptr", &self.device_ptr)
            .finish_non_exhaustive()
    }
}

// Both `CudaResidentDeviceMemory` and its read view are now **auto** `Send`/`Sync`: every
// field is thread-safe — `metadata` is plain data, `device_ptr` is a `u64`, allocation lifetime
// and context/library/module-cache/stream-pool all live behind `Arc` (with `GpuPrimaryContext`'s
// `unsafe impl Send + Sync` carrying the load-bearing argument), and telemetry uses a `Mutex`.
// This is the §9.3 payoff over the P1-M3 per-allocation model: residency owns
// no raw context, so the previously hand-written `unsafe impl`s here are gone — concurrent
// reads over a published generation are safe by construction, witnessed by the
// `published_resident_generation_*` and `concurrent_readers_*` GPU probes.

impl CudaResidentDeviceMemoryReadView {
    pub fn metadata(&self) -> &CudaDeviceMemoryProof {
        &self.metadata
    }

    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    pub fn context(&self) -> *mut c_void {
        self.primary.context()
    }
}

pub(super) trait CudaResidentReadSource {
    fn metadata(&self) -> &CudaDeviceMemoryProof;
    fn device_ptr(&self) -> u64;
    fn lib(&self) -> &Library;
    /// The shared primary context — entry point to the module cache + stream pool, so the
    /// generic read routes (over an owner or a read view) can migrate to the substrate.
    fn primary(&self) -> &GpuPrimaryContext;
    /// A cloned strong handle to the shared primary context — for deferred-completion routes
    /// (split `submit`/`complete`) that must carry pool re-entry across the boundary in a `Send`
    /// submission, where a borrow won't outlive the `submit` frame.
    fn primary_arc(&self) -> Arc<GpuPrimaryContext>;
    /// A strong guard for the exact resident allocation addressed by a deferred submission. Safe split
    /// submit/complete APIs return independently of the source borrow, so the submission must retain it.
    fn allocation_arc(&self) -> Arc<CudaResidentDeviceAllocation>;
    fn record_kernel_event_elapsed_us(&self, _elapsed_us: Option<u64>) {}
}

impl CudaResidentReadSource for CudaResidentDeviceMemory {
    fn metadata(&self) -> &CudaDeviceMemoryProof {
        &self.metadata
    }

    fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    fn lib(&self) -> &Library {
        self.primary.lib()
    }

    fn primary(&self) -> &GpuPrimaryContext {
        &self.primary
    }

    fn primary_arc(&self) -> Arc<GpuPrimaryContext> {
        Arc::clone(&self.primary)
    }

    fn allocation_arc(&self) -> Arc<CudaResidentDeviceAllocation> {
        Arc::clone(&self.allocation)
    }

    fn record_kernel_event_elapsed_us(&self, elapsed_us: Option<u64>) {
        CudaResidentDeviceMemory::record_kernel_event_elapsed_us(self, elapsed_us);
    }
}

impl CudaResidentReadSource for CudaResidentDeviceMemoryReadView {
    fn metadata(&self) -> &CudaDeviceMemoryProof {
        &self.metadata
    }

    fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    fn lib(&self) -> &Library {
        self.primary.lib()
    }

    fn primary(&self) -> &GpuPrimaryContext {
        &self.primary
    }

    fn primary_arc(&self) -> Arc<GpuPrimaryContext> {
        Arc::clone(&self.primary)
    }

    fn allocation_arc(&self) -> Arc<CudaResidentDeviceAllocation> {
        Arc::clone(&self.allocation)
    }
}

impl CudaResidentDeviceMemory {
    /// Stable identity of the underlying CUDA allocation, independent of this outer resource
    /// wrapper. Retained plan accounting uses it to avoid double-charging aliases created for a
    /// different primary-context witness in tests or recovery probes.
    pub fn allocation_identity(&self) -> usize {
        Arc::as_ptr(&self.allocation) as usize
    }

    /// Whether a sealed constructor proved this entire allocation was initialized exactly over
    /// `[0, allocated_bytes)`. This deliberately does not infer coverage from `copied_bytes`.
    pub fn has_full_contiguous_initialization(&self) -> bool {
        self.full_contiguous_initialization.load(Ordering::Acquire)
    }

    /// Mint the narrow full-allocation initialization proof after the owning CUDA path has
    /// synchronously established contiguous coverage. Kept crate-visible so the async-copy
    /// completion path can mint it only after its stream fence succeeds.
    pub(crate) fn mark_full_contiguous_initialization(&self) {
        self.full_contiguous_initialization
            .store(true, Ordering::Release);
    }

    /// Transfer one freshly allocated raw device pointer into the shared owner/read-view lifetime.
    /// Every construction path funnels through this helper so a safe read view can never outlive the
    /// allocation it addresses.
    pub(super) fn from_raw_parts(
        metadata: CudaDeviceMemoryProof,
        device_ptr: u64,
        primary: Arc<GpuPrimaryContext>,
    ) -> Self {
        Self::from_raw_parts_with_scope_reservation(metadata, device_ptr, primary, None)
    }

    pub(super) fn from_raw_parts_with_scope_reservation(
        metadata: CudaDeviceMemoryProof,
        device_ptr: u64,
        primary: Arc<GpuPrimaryContext>,
        scope_reservation: Option<CudaExternalAllocationReservation>,
    ) -> Self {
        let allocation = Arc::new(CudaResidentDeviceAllocation {
            device_ptr,
            primary: Arc::clone(&primary),
            _scope_reservation: scope_reservation,
        });
        Self {
            metadata,
            device_ptr,
            primary,
            allocation,
            full_contiguous_initialization: AtomicBool::new(false),
            last_kernel_event_elapsed_us: Mutex::new(None),
        }
    }

    pub fn metadata(&self) -> &CudaDeviceMemoryProof {
        &self.metadata
    }

    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Fence work submitted to the default stream before an input allocation is retired or reused.
    /// Relational run compaction uses this at explicit lifetime boundaries; it transfers no data.
    pub fn synchronize_default_stream(&self) -> Result<(), CudaRuntimeProbeError> {
        self.primary.set_current()?;
        check_cuda(unsafe { (self.primary.cu_stream_synchronize)(std::ptr::null_mut()) })
    }

    /// R1b: DtoH a contiguous int4 column `[byte_offset, byte_offset + count*4)` from THIS resident
    /// buffer into host `i32`s. The GPU index is built from these exact bytes — the same ones the scan
    /// kernel reads and the probe gathers from — so the index's row→value mapping is inherently
    /// consistent with the resident buffer (no host-rows cross-map reference, no NULL re-encoding: a NULL
    /// int4 is materialized as `0` here exactly as the scan sees it). Blocking copy on the calling thread.
    pub fn read_resident_i32_column(
        &self,
        byte_offset: u64,
        count: usize,
    ) -> Result<Vec<i32>, CudaRuntimeProbeError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let bytes = count
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let end = byte_offset
            .checked_add(bytes as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if end > self.metadata.allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(bytes));
        }
        self.primary.set_current()?;
        let mut out = vec![0_i32; count];
        check_cuda(unsafe {
            (self.primary.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                self.device_ptr + byte_offset,
                bytes,
            )
        })?;
        Ok(out)
    }

    /// U1: the u64 twin of [`Self::read_resident_i32_column`] — the visibility-aware index
    /// rebuild reads the shard's `deleted_by` stamps to skip rows dead below the GC boundary.
    pub fn read_resident_u64_column(
        &self,
        byte_offset: u64,
        count: usize,
    ) -> Result<Vec<u64>, CudaRuntimeProbeError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let bytes = count
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let end = byte_offset
            .checked_add(bytes as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if end > self.metadata.allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(bytes));
        }
        self.primary.set_current()?;
        let mut out = vec![0_u64; count];
        check_cuda(unsafe {
            (self.primary.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                self.device_ptr + byte_offset,
                bytes,
            )
        })?;
        Ok(out)
    }

    /// TYPE-COVERAGE #14 (text): read `len` RAW bytes from this buffer at `byte_offset` (a text column's
    /// bytes blob), for the device->host rehydration gather. Bounds-checked against `allocated_bytes`.
    pub fn read_resident_bytes(
        &self,
        byte_offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, CudaRuntimeProbeError> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let end = byte_offset
            .checked_add(len as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if end > self.metadata.allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(len));
        }
        self.primary.set_current()?;
        let mut out = vec![0_u8; len];
        check_cuda(unsafe {
            (self.primary.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                self.device_ptr + byte_offset,
                len,
            )
        })?;
        Ok(out)
    }

    /// Slice 1a (GPU-native writes): append `chunks` IN PLACE into this allocation's headroom — one
    /// `cuMemcpyHtoD` to `device_ptr + byte_offset` per chunk, with NO reallocation and NO
    /// device-to-device recompaction, so an OPEN shard grows without re-uploading the rows it already
    /// holds (the whole point — the dual-store tax was a full re-upload per commit). Each chunk must
    /// fit within `allocated_bytes` (the headroom reserved at admission); a chunk that would overrun
    /// the allocation is rejected (the caller must roll over to a fresh shard instead). Returns the
    /// total bytes written (the caller advances the shard's `resident_bytes` / `row_count`). The
    /// shared primary context is made current on this thread first (mirrors `read_resident_i32_column`
    /// and `launch_cuda_resident_device_memory_owned_chunks`). No cached sync HtoD pointer exists
    /// (only the async one), so the sync entry point is resolved from the context's library.
    ///
    /// PARTIAL-FAILURE CONTRACT (no rollback): chunks are written in order; if a `cuMemcpyHtoD` fails
    /// mid-list, the chunks already written STAY written and `Err` is returned with the allocation left
    /// PARTIALLY MUTATED. A caller appending multiple column sections must therefore treat any non-`Ok`
    /// return as "this shard is poisoned -- do NOT publish it; rebuild/drop it", and must not advance the
    /// live row count (the header) until all column writes have landed (so a partial append can never
    /// advertise rows whose column bytes are missing).
    pub fn append_owned_chunks<I>(&self, chunks: I) -> Result<u64, CudaRuntimeProbeError>
    where
        I: IntoIterator<Item = CudaOwnedDeviceMemoryChunk>,
    {
        type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
        self.primary.set_current()?;
        let cu_memcpy_htod = unsafe {
            *self
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| self.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let mut appended_bytes = 0_u64;
        for chunk in chunks {
            if chunk.bytes.is_empty() {
                continue;
            }
            let len = chunk.bytes.len();
            let end = chunk
                .byte_offset
                .checked_add(len as u64)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if end > self.metadata.allocated_bytes {
                return Err(CudaRuntimeProbeError::InvalidInputLength(len));
            }
            // checked_add on the destination address too (mirrors the launcher) — for an honestly-built
            // allocation `byte_offset < allocated_bytes` so this cannot wrap, but guard against a
            // debug-build panic / release-build silent wrap to a wrong device address.
            let dst = self
                .device_ptr
                .checked_add(chunk.byte_offset)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            check_cuda(unsafe { cu_memcpy_htod(dst, chunk.bytes.as_ptr().cast::<c_void>(), len) })?;
            appended_bytes = appended_bytes
                .checked_add(len as u64)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        }
        Ok(appended_bytes)
    }

    pub fn context(&self) -> *mut c_void {
        self.primary.context()
    }

    /// The shared primary context backing this allocation — the entry point to the module
    /// cache and stream pool for migrated launch routes (plan §9.3 / Phase 2).
    pub(super) fn primary(&self) -> &GpuPrimaryContext {
        &self.primary
    }

    /// Bind the shared primary context to the calling thread. The driver keeps the current
    /// context per-thread, so a reader thread must make it current before launching a
    /// kernel — otherwise the launch fails with `CUDA_ERROR_INVALID_CONTEXT` (201). Safe
    /// to call concurrently from many reader threads: a context may be current on multiple
    /// threads at once (driver ≥ 4.0). With the shared primary context (§9.3) this is a
    /// once-per-thread bind to the *same* context for every table, not a per-allocation
    /// context — which also removes the old cross-generation context-mismatch hazard.
    pub fn set_current_context(&self) -> Result<(), CudaRuntimeProbeError> {
        self.primary.set_current()
    }

    pub(super) fn lib(&self) -> &Library {
        self.primary.lib()
    }

    pub fn read_view(&self) -> CudaResidentDeviceMemoryReadView {
        CudaResidentDeviceMemoryReadView {
            metadata: self.metadata.clone(),
            device_ptr: self.device_ptr,
            primary: Arc::clone(&self.primary),
            allocation: Arc::clone(&self.allocation),
        }
    }

    #[cfg(test)]
    pub(crate) fn allocation_weak_for_test(&self) -> CudaResidentAllocationWeak {
        CudaResidentAllocationWeak(Arc::downgrade(&self.allocation))
    }

    #[cfg(test)]
    pub(crate) fn clone_with_primary_for_test(&self, primary: Arc<GpuPrimaryContext>) -> Self {
        Self {
            metadata: self.metadata.clone(),
            device_ptr: self.device_ptr,
            primary,
            allocation: Arc::clone(&self.allocation),
            full_contiguous_initialization: AtomicBool::new(
                self.has_full_contiguous_initialization(),
            ),
            last_kernel_event_elapsed_us: Mutex::new(None),
        }
    }

    /// Build a distinct outer resource wrapper which retains the same underlying allocation.
    /// Cross-crate probe tests use this to prove accounting keys the allocation owner, not the
    /// wrapper Arc identity.
    #[cfg(any(test, feature = "probe-timing"))]
    pub fn distinct_wrapper_for_accounting_test(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            device_ptr: self.device_ptr,
            primary: Arc::clone(&self.primary),
            allocation: Arc::clone(&self.allocation),
            full_contiguous_initialization: AtomicBool::new(
                self.has_full_contiguous_initialization(),
            ),
            last_kernel_event_elapsed_us: Mutex::new(None),
        }
    }

    pub fn last_kernel_event_elapsed_us(&self) -> Option<u64> {
        self.last_kernel_event_elapsed_us
            .lock()
            .ok()
            .and_then(|elapsed| *elapsed)
    }

    pub fn clear_last_kernel_event_elapsed_us(&self) {
        self.record_kernel_event_elapsed_us(None);
    }

    pub(super) fn record_kernel_event_elapsed_us(&self, elapsed_us: Option<u64>) {
        if let Ok(mut last) = self.last_kernel_event_elapsed_us.lock() {
            *last = elapsed_us;
        }
    }

    pub fn count_rows_from_header(&self) -> Result<u64, CudaRuntimeProbeError> {
        launch_cuda_resident_row_count(self)
    }

    /// GPU-reduce the rows visible at `read_txn_id` to one scalar. Version lanes may live in
    /// independent allocations (resident shards) or at byte offsets in this allocation (cold
    /// chunk replay); `None` denotes an all-visible side of the predicate.
    pub fn count_visible_rows(
        &self,
        row_count: u64,
        read_txn_id: i64,
        deleted_by: Option<(&Self, u64)>,
        created_by: Option<(&Self, u64)>,
    ) -> Result<u64, CudaRuntimeProbeError> {
        launch_cuda_resident_visible_count(self, row_count, read_txn_id, deleted_by, created_by)
    }

    /// GPU-reduce the exact visible logical source set to a constant-size digest. The digest binds
    /// every visible stable row identity, typed column value, and NULL bit; version lanes select
    /// visibility but are not representation-dependent digest input. This makes the result stable
    /// across equivalent hot-shard and staged-cold layouts while detecting same-cardinality content
    /// substitution. Only the 40-byte count/digest result crosses D2H.
    pub fn digest_visible_source(
        &self,
        row_count: u64,
        columns: &[CudaVisibleDigestColumn],
        row_ids: Option<(&Self, u64)>,
        read_txn_id: i64,
        deleted_by: Option<(&Self, u64)>,
        created_by: Option<(&Self, u64)>,
    ) -> Result<CudaVisibleSourceDigest, CudaRuntimeProbeError> {
        launch_cuda_resident_visible_digest(
            self,
            row_count,
            columns,
            row_ids,
            read_txn_id,
            deleted_by,
            created_by,
        )
    }
}
