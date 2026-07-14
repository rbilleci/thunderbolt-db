use std::fmt;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use libloading::Library;

use crate::cuda_context::{check_cuda, GpuPrimaryContext};
use crate::{launch_cuda_resident_row_count, CudaDeviceMemoryProof, CudaRuntimeProbeError};

pub struct CudaResidentDeviceMemory {
    pub(super) metadata: CudaDeviceMemoryProof,
    pub(super) device_ptr: u64,
    /// The shared primary context this allocation lives in (plan §9.3). Residency owns no
    /// context of its own; it holds a refcount so the context outlives the allocation, and
    /// reaches the device pointer, library, module cache, and stream pool through it.
    pub(super) primary: Arc<GpuPrimaryContext>,
    pub(super) last_kernel_event_elapsed_us: Mutex<Option<u64>>,
}

#[derive(Clone)]
pub struct CudaResidentDeviceMemoryReadView {
    metadata: CudaDeviceMemoryProof,
    device_ptr: u64,
    primary: Arc<GpuPrimaryContext>,
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
// field is thread-safe — `metadata` is plain data, `device_ptr` is a `u64`, the telemetry
// slot is a `Mutex`, and the context/library/module-cache/stream-pool all live behind
// `Arc<GpuPrimaryContext>` (whose own `unsafe impl Send + Sync` carries the load-bearing
// argument). This is the §9.3 payoff over the P1-M3 per-allocation model: residency owns
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
}

impl CudaResidentDeviceMemory {
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
}

impl Drop for CudaResidentDeviceMemory {
    fn drop(&mut self) {
        // §9.3: residency frees only its own device memory. The shared primary context is
        // owned by `GpuPrimaryContext` (released when the last `Arc` — registry + every
        // allocation — drops), not destroyed per allocation. Bind the context first so the
        // free lands in the right context even when the last reader drops this owner on a
        // thread that never bound it (otherwise cuMemFree would no-op + leak).
        let _ = self.primary.set_current();
        unsafe {
            (self.primary.cu_mem_free)(self.device_ptr);
        }
    }
}
