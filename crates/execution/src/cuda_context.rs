use std::collections::BTreeMap;
use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use libloading::Library;

use crate::{CudaDeviceMemoryProof, CudaResidentDeviceMemory, CudaRuntimeProbeError};

// ---------------------------------------------------------------------------
// GPU shared-context substrate (plan §9.3 + Phase 2 module cache / stream pool)
// ---------------------------------------------------------------------------

/// A loaded CUDA module and the entry function handle resolved from it. Cached on the
/// owning `GpuPrimaryContext` so a kernel's PTX is JIT-loaded once per process, not per
/// launch. The handles are valid for the lifetime of the context they were loaded into.
struct CachedModule {
    module: *mut c_void,
    function: *mut c_void,
}

/// One retained CUDA **primary** context per physical GPU (plan §9.3), shared by every
/// resident allocation and every reader thread. Replaces the prototype's per-allocation
/// `cuCtxCreate`/`cuCtxDestroy` (one heavyweight context per table-generation). It owns the
/// process-wide module/function cache (`cached_function`) — which kills the per-launch
/// `cuModuleLoadData`/`Unload` the P1-M3 step-4 benchmark found dominating concurrent reads
/// — and a small stream pool (`acquire_stream`/`release_stream`) so concurrent launches use
/// private streams synced individually (`cuStreamSynchronize`) instead of one global
/// `cuCtxSynchronize` barrier. Created lazily and cached in `gpu_primary_context`; retained
/// for process lifetime (the registry holds a strong `Arc`), so `Drop` — which unloads
/// cached modules, destroys pooled streams, then `cuDevicePrimaryCtxRelease` — runs at
/// process teardown (or if an entry is ever evicted). Residency holds an
/// `Arc<GpuPrimaryContext>` and owns no context of its own (the §9.3 "residency owns no
/// context" property).
/// Bytes of reusable device "scratch" output attached to each pooled stream. Migrated
/// scalar routes (e.g. COUNT) write their result here instead of `cuMemAlloc`-ing a fresh
/// output buffer per call — removing a driver-serialized allocation from the hot path.
pub(super) const POOLED_STREAM_SCRATCH_BYTES: usize = 64;

/// A pooled CUDA stream plus the reusable per-stream resources a migrated launch needs: a
/// small device output buffer and a pair of timing events. Pooling these removes the
/// per-call `cuMemAlloc`/`cuEventCreate`/`cuEventDestroy` (all driver-serialized) that the
/// P2-M1 step-4 benchmark found re-serializing concurrent reads once the module load and
/// whole-context sync were gone. `start_event`/`stop_event` are null when the event symbols
/// are unavailable (timing is then skipped).
pub(super) struct PooledStream {
    pub(super) stream: *mut c_void,
    pub(super) output: u64,
    pub(super) start_event: *mut c_void,
    pub(super) stop_event: *mut c_void,
}

/// Lower bound on a pooled output buffer — one bucket holds all the tiny counter/needle
/// allocations so they reuse too.
const MIN_POOLED_BUFFER_BYTES: usize = 256;
/// Cap on idle (freed-but-retained) pooled output bytes; a release beyond this frees instead
/// of pooling, so a one-off oversized request can't pin device memory for process lifetime.
const POOLED_OUTPUT_BYTES_CAP: usize = 1 << 30; // 1 GiB

/// Fallibly round a buffer request up to its pool bucket (power of two, floored at
/// `MIN_POOLED_BUFFER_BYTES`) so a release maps straight back to the bucket it came from.
/// Public prepared-geometry preflight uses this before it can reach an allocator.
pub(super) fn checked_output_buffer_bucket(min_bytes: usize) -> Option<usize> {
    min_bytes
        .max(MIN_POOLED_BUFFER_BYTES)
        .checked_next_power_of_two()
}

/// Round a buffer request up to its pool bucket (power of two, floored at
/// `MIN_POOLED_BUFFER_BYTES`) so a release maps straight back to the bucket it came from.
pub(super) fn output_buffer_bucket(min_bytes: usize) -> usize {
    min_bytes.max(MIN_POOLED_BUFFER_BYTES).next_power_of_two()
}

/// Power-of-two-bucketed free list of reusable device output buffers. The projection/gather
/// routes allocate several output buffers per call sized to the worst-case `row_count` (the
/// P2-M2 text-route benchmark measured ~9 allocs / ~2 MB per call); `cuMemAlloc`/`cuMemFree`
/// are driver-serialized and dominated those routes at high concurrency. Reusing buffers
/// across calls removes that churn. `pooled_bytes` bounds the idle retained set.
#[derive(Default)]
struct OutputBufferPool {
    free: BTreeMap<usize, Vec<u64>>,
    pooled_bytes: usize,
}

/// Power-of-two-bucketed free list of reusable **pinned (page-locked) host** buffers, mirroring
/// `OutputBufferPool`. The fused text route copies its results device→host with the async D2H
/// variant on its private stream; an async D2H is only truly asynchronous (and DMA-fast) into
/// page-locked host memory, but `cuMemHostAlloc`/`cuMemFreeHost` are themselves driver-
/// serialized, so a per-call alloc would re-introduce exactly the contention this fix removes.
/// Pooling the host staging buffers across calls removes that churn. Pointers are wrapped in
/// `PinnedHostPtr` so the map is `Send`/`Sync` under the context's existing `unsafe impl`.
#[derive(Default)]
struct PinnedHostBufferPool {
    free: BTreeMap<usize, Vec<PinnedHostPtr>>,
    pooled_bytes: usize,
}

/// A raw pinned-host allocation. `Send`/`Sync` is sound for the same reason the context's other
/// raw pointers are: the buffer is owned solely by the pool/lease (never aliased), and
/// `cuMemFreeHost` is valid from any thread.
#[derive(Clone, Copy)]
struct PinnedHostPtr(*mut c_void);
// SAFETY: see the type doc — the pointer is exclusively owned by the pool or an active lease and
// is freed via the thread-safe `cuMemFreeHost`; it is never concurrently aliased.
unsafe impl Send for PinnedHostPtr {}
unsafe impl Sync for PinnedHostPtr {}

pub(super) struct CudaAllocationTracker {
    live: std::sync::atomic::AtomicU64,
    peak: std::sync::atomic::AtomicU64,
    limit: u64,
}

thread_local! {
    static CUDA_ALLOCATION_TRACKER: std::cell::RefCell<Option<Arc<CudaAllocationTracker>>> =
        const { std::cell::RefCell::new(None) };
}

/// Thread-scoped, allocator-backed budget for relational scratch. Every pooled device-buffer lease
/// made while the scope is active reserves its actual power-of-two allocation capacity before CUDA
/// allocation/reuse. This makes an operator fail before crossing its VRAM allowance and exposes the
/// true live-allocation high-water rather than a post-hoc row-width estimate.
pub struct CudaAllocationScope {
    tracker: Arc<CudaAllocationTracker>,
    previous: Option<Arc<CudaAllocationTracker>>,
}

/// Reservation for a raw CUDA allocation performed by an engine-owned builder rather than the
/// pooled execution allocator. It charges the current scope before that allocation and releases on
/// drop, so conservative layout bounds participate in the same live high-water and hard limit.
pub struct CudaExternalAllocationReservation {
    tracker: Option<Arc<CudaAllocationTracker>>,
    bytes: usize,
}

impl CudaAllocationScope {
    pub fn with_budget(limit: u64) -> Self {
        let tracker = Arc::new(CudaAllocationTracker {
            live: std::sync::atomic::AtomicU64::new(0),
            peak: std::sync::atomic::AtomicU64::new(0),
            limit,
        });
        let previous =
            CUDA_ALLOCATION_TRACKER.with(|slot| slot.borrow_mut().replace(Arc::clone(&tracker)));
        Self { tracker, previous }
    }

    pub fn peak_bytes(&self) -> u64 {
        self.tracker.peak.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Fail closed when `bytes` of additional simultaneously-live device ownership would cross
    /// this scope's hard limit, without retaining a charge. Callers use this immediately before a
    /// compound operator whose individual pooled leases are still charged normally: it proves the
    /// complete operator geometry fits before the first lease is acquired, while the leases remain
    /// the source of truth for the measured high-water.
    pub fn ensure_available(bytes: u64) -> Result<(), CudaRuntimeProbeError> {
        CUDA_ALLOCATION_TRACKER.with(|slot| {
            let Some(tracker) = slot.borrow().as_ref().cloned() else {
                return Ok(());
            };
            let live = tracker.live.load(std::sync::atomic::Ordering::Relaxed);
            if live.saturating_add(bytes) > tracker.limit {
                return Err(CudaRuntimeProbeError::AllocationBudgetExceeded {
                    requested: bytes,
                    live,
                    limit: tracker.limit,
                });
            }
            Ok(())
        })
    }

    pub fn reserve_external(
        bytes: usize,
    ) -> Result<CudaExternalAllocationReservation, CudaRuntimeProbeError> {
        let tracker = CUDA_ALLOCATION_TRACKER
            .with(|slot| slot.borrow().as_ref().map(|tracker| tracker.reserve(bytes)))
            .transpose()?;
        Ok(CudaExternalAllocationReservation { tracker, bytes })
    }
}

impl Drop for CudaAllocationScope {
    fn drop(&mut self) {
        CUDA_ALLOCATION_TRACKER.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

impl Drop for CudaExternalAllocationReservation {
    fn drop(&mut self) {
        if let Some(tracker) = &self.tracker {
            tracker.release(self.bytes);
        }
    }
}

impl CudaAllocationTracker {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Arc<Self>, CudaRuntimeProbeError> {
        let bytes = bytes as u64;
        let mut live = self.live.load(std::sync::atomic::Ordering::Relaxed);
        loop {
            let next = live.saturating_add(bytes);
            if next > self.limit {
                return Err(CudaRuntimeProbeError::AllocationBudgetExceeded {
                    requested: bytes,
                    live,
                    limit: self.limit,
                });
            }
            match self.live.compare_exchange_weak(
                live,
                next,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.peak
                        .fetch_max(next, std::sync::atomic::Ordering::Relaxed);
                    return Ok(Arc::clone(self));
                }
                Err(actual) => live = actual,
            }
        }
    }

    fn release(&self, bytes: usize) {
        self.live
            .fetch_sub(bytes as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(super) struct GpuPrimaryContext {
    device: i32,
    context: *mut c_void,
    pub(super) cu_mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    pub(super) cu_mem_free: unsafe extern "C" fn(u64) -> i32,
    pub(super) cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    // P2-M2: stream-ordered (async) transfer + pinned-host primitives. The fused text route
    // issues its HtoD/memset/D2H on its pooled private stream via the `*Async` variants behind
    // a minimal pair of `cuStreamSynchronize` (instead of the legacy blocking default/NULL-
    // stream ops, which the driver serializes context-wide across concurrent readers). The
    // pinned-host alloc/free back a page-locked host-buffer pool so those D2H are truly async
    // and DMA-fast. All five are best-effort: a route only uses them when present (else it
    // keeps the blocking path), so an old driver still runs correctly, just unaccelerated.
    pub(super) cu_memcpy_htod_async:
        Option<unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> i32>,
    pub(super) cu_memcpy_dtoh_async:
        Option<unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> i32>,
    pub(super) cu_memset_d8_async: Option<unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> i32>,
    cu_mem_host_alloc: Option<unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32>,
    cu_mem_free_host: Option<unsafe extern "C" fn(*mut c_void) -> i32>,
    cu_ctx_set_current: unsafe extern "C" fn(*mut c_void) -> i32,
    cu_primary_ctx_release: unsafe extern "C" fn(i32) -> i32,
    cu_module_load_data: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32,
    cu_module_unload: unsafe extern "C" fn(*mut c_void) -> i32,
    cu_module_get_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32,
    cu_stream_create: unsafe extern "C" fn(*mut *mut c_void, u32) -> i32,
    cu_stream_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    pub(super) cu_stream_synchronize: unsafe extern "C" fn(*mut c_void) -> i32,
    pub(super) cu_event_create: unsafe extern "C" fn(*mut *mut c_void, u32) -> i32,
    pub(super) cu_event_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    pub(super) cu_event_record: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    pub(super) cu_event_elapsed_time:
        unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> i32,
    modules: Mutex<BTreeMap<&'static CStr, CachedModule>>,
    streams: Mutex<Vec<PooledStream>>,
    output_buffers: Mutex<OutputBufferPool>,
    pinned_host_buffers: Mutex<PinnedHostBufferPool>,
    lib: Arc<Library>,
}

// SAFETY: the only !Send/!Sync fields are the raw `context` and the cached module/stream
// pointers. The CUDA driver API is thread-safe (driver >= 4.0): one primary context may
// be current on many threads at once, and concurrent launches of one cached function on
// distinct streams are explicitly permitted. The context is immutable after creation; the
// module cache and stream pool are each behind a `Mutex`, so the only mutation is
// serialized. `cuMemFree`/`cuStreamDestroy`/`cuDevicePrimaryCtxRelease` are valid from any
// thread. This is the load-bearing context-layer `unsafe`; residency above it becomes
// auto-`Send`/`Sync` (it owns only `device_ptr: u64`, a `Mutex`, and `Arc<Self>`).
unsafe impl Send for GpuPrimaryContext {}
unsafe impl Sync for GpuPrimaryContext {}

impl GpuPrimaryContext {
    pub(super) fn context(&self) -> *mut c_void {
        self.context
    }

    pub(super) fn lib(&self) -> &Library {
        self.lib.as_ref()
    }

    /// Make this primary context current on the calling thread (idempotent; a context may
    /// be current on many threads). Every reader thread must call this before launching.
    pub(super) fn set_current(&self) -> Result<(), CudaRuntimeProbeError> {
        check_cuda(unsafe { (self.cu_ctx_set_current)(self.context) })
    }

    /// Return the entry function for `entry_name`, loading + caching its module the first
    /// time. `ptx_with_nul` must be NUL-terminated PTX. The returned handle is reused on
    /// every subsequent call and is safe to launch concurrently on distinct streams.
    pub(super) fn cached_function(
        &self,
        entry_name: &'static CStr,
        ptx_with_nul: &[u8],
    ) -> Result<*mut c_void, CudaRuntimeProbeError> {
        let mut modules = self.modules.lock().expect("gpu module cache poisoned");
        if let Some(cached) = modules.get(entry_name) {
            return Ok(cached.function);
        }
        let mut module = std::ptr::null_mut();
        check_cuda(unsafe {
            (self.cu_module_load_data)(&mut module, ptx_with_nul.as_ptr().cast::<c_void>())
        })?;
        let module_guard = CudaModuleGuard {
            module,
            unload: self.cu_module_unload,
        };
        let mut function = std::ptr::null_mut();
        check_cuda(unsafe {
            (self.cu_module_get_function)(&mut function, module, entry_name.as_ptr())
        })?;
        std::mem::forget(module_guard); // ownership moves into the cache (unloaded on Drop)
        modules.insert(entry_name, CachedModule { module, function });
        Ok(function)
    }

    /// Take a stream (with its reusable scratch output + timing events) from the pool,
    /// creating one if the pool is empty.
    pub(super) fn acquire_pooled_stream(&self) -> Result<PooledStream, CudaRuntimeProbeError> {
        if let Some(pooled) = self.streams.lock().expect("gpu stream pool poisoned").pop() {
            return Ok(pooled);
        }
        let mut stream = std::ptr::null_mut();
        check_cuda(unsafe { (self.cu_stream_create)(&mut stream, 0) })?;
        let mut output = 0_u64;
        if let Err(err) =
            check_cuda(unsafe { (self.cu_mem_alloc)(&mut output, POOLED_STREAM_SCRATCH_BYTES) })
        {
            unsafe { (self.cu_stream_destroy)(stream) };
            return Err(err);
        }
        // Best-effort timing events: if either fails, run untimed rather than fail the read.
        let mut start_event = std::ptr::null_mut();
        let mut stop_event = std::ptr::null_mut();
        let s1 = unsafe { (self.cu_event_create)(&mut start_event, 0) };
        let s2 = unsafe { (self.cu_event_create)(&mut stop_event, 0) };
        if s1 != 0 || s2 != 0 {
            if s1 == 0 {
                unsafe { (self.cu_event_destroy)(start_event) };
            }
            if s2 == 0 {
                unsafe { (self.cu_event_destroy)(stop_event) };
            }
            start_event = std::ptr::null_mut();
            stop_event = std::ptr::null_mut();
        }
        Ok(PooledStream {
            stream,
            output,
            start_event,
            stop_event,
        })
    }

    /// Return a pooled stream (and its scratch) to the pool for reuse.
    pub(super) fn release_pooled_stream(&self, pooled: PooledStream) {
        self.streams
            .lock()
            .expect("gpu stream pool poisoned")
            .push(pooled);
    }

    /// Lease a device output buffer of at least `min_bytes`, reusing a pooled one of the
    /// matching bucket if available (else allocating). The lease returns it to the pool on
    /// drop — removing the per-call `cuMemAlloc`/`cuMemFree` the projection/gather routes
    /// otherwise serialize on at high concurrency. The buffer is **not** zeroed; callers that
    /// need a zeroed region (e.g. the atomic-append counters) must memset it, and callers must
    /// only read back the region the kernel actually wrote (the routes read `[0, count)`).
    pub(super) fn lease_device_buffer(
        &self,
        min_bytes: usize,
    ) -> Result<PooledBufferLease<'_>, CudaRuntimeProbeError> {
        let capacity = output_buffer_bucket(min_bytes);
        let tracker = CUDA_ALLOCATION_TRACKER
            .with(|slot| {
                slot.borrow()
                    .as_ref()
                    .map(|tracker| tracker.reserve(capacity))
            })
            .transpose()?;
        let reused = {
            let mut pool = self
                .output_buffers
                .lock()
                .expect("gpu output-buffer pool poisoned");
            match pool.free.get_mut(&capacity).and_then(Vec::pop) {
                Some(ptr) => {
                    pool.pooled_bytes = pool.pooled_bytes.saturating_sub(capacity);
                    Some(ptr)
                }
                None => None,
            }
        };
        let ptr = match reused {
            Some(ptr) => ptr,
            None => {
                let mut ptr = 0_u64;
                if let Err(err) = check_cuda(unsafe { (self.cu_mem_alloc)(&mut ptr, capacity) }) {
                    if let Some(tracker) = &tracker {
                        tracker.release(capacity);
                    }
                    return Err(err);
                }
                ptr
            }
        };
        Ok(PooledBufferLease {
            primary: self,
            ptr,
            capacity,
            tracker,
        })
    }

    /// Owned (`'static`) variant of `lease_device_buffer`: same pool, same bucketing, but returns
    /// a guard that holds an `Arc<Self>` instead of a borrow so it can be moved into a deferred
    /// `Send` submission that outlives the leasing stack frame. `self` is taken as `&Arc<Self>`.
    pub(super) fn lease_device_buffer_owned(
        self: &Arc<Self>,
        min_bytes: usize,
    ) -> Result<PooledDeviceBufferOwned, CudaRuntimeProbeError> {
        let lease = self.lease_device_buffer(min_bytes)?;
        // Transfer the leased buffer into the owned guard without round-tripping it through the
        // pool: read its fields, then forget the borrow-scoped lease so its Drop does not release.
        let (ptr, capacity, tracker) = (lease.ptr, lease.capacity, lease.tracker.clone());
        std::mem::forget(lease);
        Ok(PooledDeviceBufferOwned {
            primary: Arc::clone(self),
            ptr,
            capacity,
            tracker,
        })
    }

    /// Allocate exact-size, engine-retained device state that is freed on owner drop rather than
    /// entering the scratch-output pool. Prepared route directories can be hundreds of MiB;
    /// returning them to the idle pool would retain physical VRAM after route accounting ends.
    pub(super) fn allocate_retained_device_buffer_owned(
        self: &Arc<Self>,
        bytes: usize,
    ) -> Result<RetainedDeviceBufferOwned, CudaRuntimeProbeError> {
        if bytes == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let tracker = CUDA_ALLOCATION_TRACKER
            .with(|slot| slot.borrow().as_ref().map(|tracker| tracker.reserve(bytes)))
            .transpose()?;
        let mut ptr = 0_u64;
        if let Err(error) = check_cuda(unsafe { (self.cu_mem_alloc)(&mut ptr, bytes) }) {
            if let Some(tracker) = &tracker {
                tracker.release(bytes);
            }
            return Err(error);
        }
        Ok(RetainedDeviceBufferOwned {
            primary: Arc::clone(self),
            ptr,
            capacity: bytes,
            tracker,
        })
    }

    /// Return a leased buffer to the pool, or free it if the idle cap is reached.
    fn release_device_buffer(&self, ptr: u64, capacity: usize) {
        let mut pool = self
            .output_buffers
            .lock()
            .expect("gpu output-buffer pool poisoned");
        if pool.pooled_bytes.saturating_add(capacity) > POOLED_OUTPUT_BYTES_CAP {
            drop(pool);
            unsafe { (self.cu_mem_free)(ptr) };
            return;
        }
        pool.pooled_bytes += capacity;
        pool.free.entry(capacity).or_default().push(ptr);
    }

    /// Lease a pinned (page-locked) host staging buffer of at least `min_bytes`, reusing a
    /// pooled one of the matching bucket if available (else allocating via `cuMemHostAlloc`).
    /// Returns `None` if the pinned-host symbols are unavailable (old driver) — callers then
    /// fall back to plain pageable host buffers + blocking D2H. The lease returns the buffer to
    /// the pool on drop, removing the per-call `cuMemHostAlloc`/`cuMemFreeHost` (driver-
    /// serialized) that would otherwise re-serialize concurrent readers. Bucketing mirrors the
    /// device pool so a release maps straight back to its bucket. The buffer is uninitialized;
    /// callers must only read back the bytes the matching D2H actually wrote.
    pub(super) fn lease_pinned_host_buffer(&self, min_bytes: usize) -> Option<PinnedHostLease<'_>> {
        let alloc = self.cu_mem_host_alloc?;
        self.cu_mem_free_host?; // required for release; bail to the pageable path if absent
        let capacity = output_buffer_bucket(min_bytes);
        let reused = {
            let mut pool = self
                .pinned_host_buffers
                .lock()
                .expect("gpu pinned-host pool poisoned");
            match pool.free.get_mut(&capacity).and_then(Vec::pop) {
                Some(ptr) => {
                    pool.pooled_bytes = pool.pooled_bytes.saturating_sub(capacity);
                    Some(ptr)
                }
                None => None,
            }
        };
        let ptr = match reused {
            Some(ptr) => ptr,
            None => {
                let mut ptr = std::ptr::null_mut();
                // flags = 0 (CU_MEMHOSTALLOC_PORTABLE/DEVICEMAP not needed for a staging buffer).
                if check_cuda(unsafe { alloc(&mut ptr, capacity, 0) }).is_err() {
                    return None;
                }
                PinnedHostPtr(ptr)
            }
        };
        Some(PinnedHostLease {
            primary: self,
            ptr: ptr.0,
            capacity,
        })
    }

    /// Return a leased pinned-host buffer to the pool, or free it (`cuMemFreeHost`) if the idle
    /// cap is reached.
    pub(super) fn release_pinned_host_buffer(&self, ptr: *mut c_void, capacity: usize) {
        let Some(free_host) = self.cu_mem_free_host else {
            return;
        };
        let mut pool = self
            .pinned_host_buffers
            .lock()
            .expect("gpu pinned-host pool poisoned");
        if pool.pooled_bytes.saturating_add(capacity) > POOLED_OUTPUT_BYTES_CAP {
            drop(pool);
            unsafe { free_host(ptr) };
            return;
        }
        pool.pooled_bytes += capacity;
        pool.free
            .entry(capacity)
            .or_default()
            .push(PinnedHostPtr(ptr));
    }

    /// Test-only: idle count in one specific pinned-host bucket. Bucket-scoped (like
    /// `is_module_cached`) so a pool-isolation test can assert "a different-bucket lease did not
    /// DRAIN this bucket" without reading the process-global total, which any *other* pool user in
    /// the binary perturbs (e.g. the migrated equal_any parity test now leases from this same
    /// shared pool — its `complete` stages result D2H through pooled pinned buffers).
    #[cfg(test)]
    pub(super) fn pooled_pinned_host_buffer_count_in_bucket(&self, min_bytes: usize) -> usize {
        let bucket = output_buffer_bucket(min_bytes);
        self.pinned_host_buffers
            .lock()
            .map(|pool| pool.free.get(&bucket).map_or(0, Vec::len))
            .unwrap_or(0)
    }

    /// Test-only: idle count in one specific device-buffer bucket. See
    /// `pooled_pinned_host_buffer_count_in_bucket` for why this is bucket-scoped.
    #[cfg(test)]
    pub(super) fn pooled_output_buffer_count_in_bucket(&self, min_bytes: usize) -> usize {
        let bucket = output_buffer_bucket(min_bytes);
        self.output_buffers
            .lock()
            .map(|pool| pool.free.get(&bucket).map_or(0, Vec::len))
            .unwrap_or(0)
    }

    /// Test-only: whether a specific kernel's module is present in the cache. The cache is
    /// keyed by entry name, so a given key can hold at most one entry no matter how many times
    /// it is launched — letting a test assert "this module is loaded once and reused" without
    /// reading the process-global module count (which sibling tests' kernels perturb).
    #[cfg(test)]
    pub(super) fn is_module_cached(&self, entry_name: &CStr) -> bool {
        self.modules
            .lock()
            .map(|modules| modules.contains_key(entry_name))
            .unwrap_or(false)
    }

    fn create(gpu_id: u16) -> Result<Self, CudaRuntimeProbeError> {
        type CuInit = unsafe extern "C" fn(u32) -> i32;
        type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
        type CuPrimaryCtxRetain = unsafe extern "C" fn(*mut *mut c_void, i32) -> i32;
        type CuPrimaryCtxRelease = unsafe extern "C" fn(i32) -> i32;
        type CuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> i32;
        type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
        type CuMemFree = unsafe extern "C" fn(u64) -> i32;
        type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
        type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
        type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
        type CuModuleGetFunction =
            unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
        type CuStreamCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
        type CuStreamDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
        type CuStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> i32;
        type CuEventCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
        type CuEventDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
        type CuEventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
        type CuEventElapsedTime = unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> i32;

        let lib = unsafe {
            Library::new("libcuda.so.1")
                .or_else(|_| Library::new("libcuda.so"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        macro_rules! sym {
            ($t:ty, $primary:literal $(, $fallback:literal)*) => {{
                let result = unsafe { lib.get::<$t>($primary) };
                $( let result = result.or_else(|_| unsafe { lib.get::<$t>($fallback) }); )*
                *result.map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
            }};
        }
        // Optional symbol: returns `Some(fn)` if present, `None` otherwise. Used for the async
        // transfer + pinned-host primitives, which the fused text route accelerates on when
        // available but does not *require* (it falls back to the blocking path on an old driver).
        macro_rules! opt_sym {
            ($t:ty, $primary:literal $(, $fallback:literal)*) => {{
                let result = unsafe { lib.get::<$t>($primary) };
                $( let result = result.or_else(|_| unsafe { lib.get::<$t>($fallback) }); )*
                result.map(|f| *f).ok()
            }};
        }
        type CuMemcpyHtoDAsync =
            unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> i32;
        type CuMemcpyDtoHAsync = unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> i32;
        type CuMemsetD8Async = unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> i32;
        type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
        type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> i32;
        let cu_init: CuInit = sym!(CuInit, b"cuInit\0");
        let cu_device_get: CuDeviceGet = sym!(CuDeviceGet, b"cuDeviceGet\0");
        let cu_primary_ctx_retain: CuPrimaryCtxRetain =
            sym!(CuPrimaryCtxRetain, b"cuDevicePrimaryCtxRetain\0");
        let cu_primary_ctx_release: CuPrimaryCtxRelease = sym!(
            CuPrimaryCtxRelease,
            b"cuDevicePrimaryCtxRelease_v2\0",
            b"cuDevicePrimaryCtxRelease\0"
        );
        let cu_ctx_set_current: CuCtxSetCurrent = sym!(CuCtxSetCurrent, b"cuCtxSetCurrent\0");
        let cu_mem_alloc: CuMemAlloc = sym!(CuMemAlloc, b"cuMemAlloc_v2\0", b"cuMemAlloc\0");
        let cu_mem_free: CuMemFree = sym!(CuMemFree, b"cuMemFree_v2\0", b"cuMemFree\0");
        let cu_memcpy_dtoh: CuMemcpyDtoH =
            sym!(CuMemcpyDtoH, b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0");
        let cu_memcpy_htod_async: Option<CuMemcpyHtoDAsync> = opt_sym!(
            CuMemcpyHtoDAsync,
            b"cuMemcpyHtoDAsync_v2\0",
            b"cuMemcpyHtoDAsync\0"
        );
        let cu_memcpy_dtoh_async: Option<CuMemcpyDtoHAsync> = opt_sym!(
            CuMemcpyDtoHAsync,
            b"cuMemcpyDtoHAsync_v2\0",
            b"cuMemcpyDtoHAsync\0"
        );
        let cu_memset_d8_async: Option<CuMemsetD8Async> =
            opt_sym!(CuMemsetD8Async, b"cuMemsetD8Async\0");
        let cu_mem_host_alloc: Option<CuMemHostAlloc> =
            opt_sym!(CuMemHostAlloc, b"cuMemHostAlloc\0");
        let cu_mem_free_host: Option<CuMemFreeHost> = opt_sym!(CuMemFreeHost, b"cuMemFreeHost\0");
        let cu_module_load_data: CuModuleLoadData = sym!(CuModuleLoadData, b"cuModuleLoadData\0");
        let cu_module_unload: CuModuleUnload = sym!(CuModuleUnload, b"cuModuleUnload\0");
        let cu_module_get_function: CuModuleGetFunction =
            sym!(CuModuleGetFunction, b"cuModuleGetFunction\0");
        let cu_stream_create: CuStreamCreate = sym!(CuStreamCreate, b"cuStreamCreate\0");
        let cu_stream_destroy: CuStreamDestroy = sym!(
            CuStreamDestroy,
            b"cuStreamDestroy_v2\0",
            b"cuStreamDestroy\0"
        );
        let cu_stream_synchronize: CuStreamSynchronize =
            sym!(CuStreamSynchronize, b"cuStreamSynchronize\0");
        let cu_event_create: CuEventCreate = sym!(CuEventCreate, b"cuEventCreate\0");
        let cu_event_destroy: CuEventDestroy =
            sym!(CuEventDestroy, b"cuEventDestroy_v2\0", b"cuEventDestroy\0");
        let cu_event_record: CuEventRecord = sym!(CuEventRecord, b"cuEventRecord\0");
        let cu_event_elapsed_time: CuEventElapsedTime =
            sym!(CuEventElapsedTime, b"cuEventElapsedTime\0");

        check_cuda(unsafe { cu_init(0) })?;
        let mut device = 0_i32;
        check_cuda(unsafe { cu_device_get(&mut device, i32::from(gpu_id)) })?;
        let mut context = std::ptr::null_mut();
        check_cuda(unsafe { cu_primary_ctx_retain(&mut context, device) })?;

        Ok(Self {
            device,
            context,
            cu_mem_alloc,
            cu_mem_free,
            cu_memcpy_dtoh,
            cu_memcpy_htod_async,
            cu_memcpy_dtoh_async,
            cu_memset_d8_async,
            cu_mem_host_alloc,
            cu_mem_free_host,
            cu_ctx_set_current,
            cu_primary_ctx_release,
            cu_module_load_data,
            cu_module_unload,
            cu_module_get_function,
            cu_stream_create,
            cu_stream_destroy,
            cu_stream_synchronize,
            cu_event_create,
            cu_event_destroy,
            cu_event_record,
            cu_event_elapsed_time,
            modules: Mutex::new(BTreeMap::new()),
            streams: Mutex::new(Vec::new()),
            output_buffers: Mutex::new(OutputBufferPool::default()),
            pinned_host_buffers: Mutex::new(PinnedHostBufferPool::default()),
            lib: Arc::new(lib),
        })
    }
}

impl Drop for GpuPrimaryContext {
    fn drop(&mut self) {
        // Unload cached modules and destroy pooled streams BEFORE releasing the primary
        // context (they belong to it), then release our retain.
        if let Ok(mut modules) = self.modules.lock() {
            for (_, cached) in std::mem::take(&mut *modules) {
                unsafe { (self.cu_module_unload)(cached.module) };
            }
        }
        if let Ok(mut streams) = self.streams.lock() {
            for pooled in std::mem::take(&mut *streams) {
                unsafe {
                    if !pooled.start_event.is_null() {
                        (self.cu_event_destroy)(pooled.start_event);
                    }
                    if !pooled.stop_event.is_null() {
                        (self.cu_event_destroy)(pooled.stop_event);
                    }
                    (self.cu_mem_free)(pooled.output);
                    (self.cu_stream_destroy)(pooled.stream);
                }
            }
        }
        if let Ok(mut pool) = self.output_buffers.lock() {
            for (_, ptrs) in std::mem::take(&mut pool.free) {
                for ptr in ptrs {
                    unsafe { (self.cu_mem_free)(ptr) };
                }
            }
        }
        if let Some(free_host) = self.cu_mem_free_host {
            if let Ok(mut pool) = self.pinned_host_buffers.lock() {
                for (_, ptrs) in std::mem::take(&mut pool.free) {
                    for ptr in ptrs {
                        unsafe { free_host(ptr.0) };
                    }
                }
            }
        }
        unsafe { (self.cu_primary_ctx_release)(self.device) };
    }
}

/// RAII handle to a pooled device output buffer; returns it to the pool on every exit
/// (success, error, or panic). `ptr` is a plain field — migrated routes read `lease.ptr`
/// exactly like the old `CudaDeviceAllocationGuard.ptr`, so only the construction site changes.
pub(super) struct PooledBufferLease<'a> {
    primary: &'a GpuPrimaryContext,
    pub(super) ptr: u64,
    pub(super) capacity: usize,
    pub(super) tracker: Option<Arc<CudaAllocationTracker>>,
}

impl PooledBufferLease<'_> {
    pub(super) fn primary_identity(&self) -> usize {
        std::ptr::from_ref(self.primary).addr()
    }
}

impl Drop for PooledBufferLease<'_> {
    fn drop(&mut self) {
        self.primary.release_device_buffer(self.ptr, self.capacity);
        if let Some(tracker) = &self.tracker {
            tracker.release(self.capacity);
        }
    }
}

/// Owned (`'static`) analogue of `PooledBufferLease`: returns the buffer to the pool on Drop, but
/// holds an `Arc<GpuPrimaryContext>` instead of a borrow so it can live inside a `Send`,
/// deferred-completion submission that outlives the `submit` stack frame (the split
/// `submit`→`complete` projection route). Same pool, same bucketed release — the only difference
/// from `PooledBufferLease` is owned vs borrowed context, so a buffer leased here is
/// indistinguishable to the pool from one leased the synchronous way.
pub(super) struct PooledDeviceBufferOwned {
    pub(super) primary: Arc<GpuPrimaryContext>,
    pub(super) ptr: u64,
    pub(super) capacity: usize,
    pub(super) tracker: Option<Arc<CudaAllocationTracker>>,
}

impl Drop for PooledDeviceBufferOwned {
    fn drop(&mut self) {
        self.primary.release_device_buffer(self.ptr, self.capacity);
        if let Some(tracker) = &self.tracker {
            tracker.release(self.capacity);
        }
    }
}

/// Exact-size device allocation for long-lived prepared-route state. Dropping this owner releases
/// physical VRAM immediately, keeping route residency accounting aligned with retained allocation.
pub(super) struct RetainedDeviceBufferOwned {
    pub(super) primary: Arc<GpuPrimaryContext>,
    pub(super) ptr: u64,
    pub(super) capacity: usize,
    pub(super) tracker: Option<Arc<CudaAllocationTracker>>,
}

impl Drop for RetainedDeviceBufferOwned {
    fn drop(&mut self) {
        let _ = self.primary.set_current();
        unsafe {
            (self.primary.cu_mem_free)(self.ptr);
        }
        if let Some(tracker) = &self.tracker {
            tracker.release(self.capacity);
        }
    }
}

/// Owned (`'static`) RAII handle to a pooled private stream: returns it to the pool on Drop,
/// holding an `Arc<GpuPrimaryContext>` so it can be carried across the `submit`→`complete`
/// boundary of the deferred-completion projection route (the synchronous routes use the
/// borrow-scoped `StreamLease` instead). Carrying the stream (rather than releasing it in
/// `submit`) keeps its timing events valid for `complete` and prevents another concurrent reader
/// from leasing — and enqueuing onto — the same stream while this route's kernel is still in
/// flight on it.
pub(super) struct PooledStreamOwned {
    pub(super) primary: Arc<GpuPrimaryContext>,
    pub(super) pooled: Option<PooledStream>,
}

impl Drop for PooledStreamOwned {
    fn drop(&mut self) {
        if let Some(pooled) = self.pooled.take() {
            self.primary.release_pooled_stream(pooled);
        }
    }
}

/// STRATA S-E.5: an IN-FLIGHT async HtoD upload of a retained device allocation (built by
/// [`CudaRuntime::retain_device_memory_copy_async`]). The copy is enqueued on a private pooled
/// stream from a pinned staging buffer; the allocation must not be read (no kernel launched
/// against it) until [`Self::wait`] returns. Dropping an un-waited handle still synchronizes the
/// stream (best-effort) before releasing the pinned buffer — the staging bytes are never handed
/// back to the pool while the DMA could still be reading them. NOTE the field DROP ORDER also
/// carries safety: `memory` drops first, and classic `cuMemFree` implicitly device-syncs, draining
/// the in-flight HtoD before the transport releases the pinned bytes — if the allocation path ever
/// migrates to stream-ordered `cuMemFreeAsync`, the transport sync becomes the ONLY fence and this
/// ordering must be revisited (audit F5).
pub struct PendingCudaResidentDeviceCopy {
    pub(super) memory: Option<CudaResidentDeviceMemory>,
    pub(super) in_flight: Option<PendingCopyTransport>,
}

pub(super) struct PendingCopyTransport {
    pub(super) primary: Arc<GpuPrimaryContext>,
    pub(super) pooled: Option<PooledStream>,
    pub(super) pinned: Option<(*mut c_void, usize)>,
}

// The transport carries a raw pinned-host pointer + a pooled stream handle across the staging →
// compute boundary of the streaming fold, exactly as `PooledStreamOwned` carries its stream across
// submit→complete. Access is externally synchronized (one fold thread drives it).
unsafe impl Send for PendingCopyTransport {}

impl PendingCopyTransport {
    /// Synchronize the copy stream, then return the stream + pinned buffer to their pools.
    /// Idempotent (fields are taken); called by `wait()` and (best-effort) by Drop.
    fn complete(&mut self) -> Result<(), CudaRuntimeProbeError> {
        let mut result = Ok(());
        if let Some(pooled) = self.pooled.take() {
            let set = self.primary.set_current();
            let sync = check_cuda(unsafe { (self.primary.cu_stream_synchronize)(pooled.stream) });
            self.primary.release_pooled_stream(pooled);
            result = set.and(sync);
        }
        if let Some((ptr, capacity)) = self.pinned.take() {
            self.primary.release_pinned_host_buffer(ptr, capacity);
        }
        result
    }
}

impl Drop for PendingCopyTransport {
    fn drop(&mut self) {
        let _ = self.complete();
    }
}

impl PendingCudaResidentDeviceCopy {
    /// The allocation's proof metadata — available IMMEDIATELY (the descriptor build does not need
    /// to wait for the copy; only kernel launches do).
    pub fn metadata(&self) -> &CudaDeviceMemoryProof {
        self.memory
            .as_ref()
            .expect("pending device copy already waited")
            .metadata()
    }

    /// Block until the upload is complete and return the (now kernel-safe) retained allocation.
    pub fn wait(mut self) -> Result<CudaResidentDeviceMemory, CudaRuntimeProbeError> {
        if let Some(mut transport) = self.in_flight.take() {
            transport.complete()?;
        }
        Ok(self
            .memory
            .take()
            .expect("pending device copy already waited"))
    }
}

/// RAII handle to a pooled pinned (page-locked) host staging buffer; returns it to the pool on
/// every exit (success, error, or panic). `ptr` is a raw host pointer the route reads back
/// through after the async D2H completes (i.e. after the stream sync).
pub(super) struct PinnedHostLease<'a> {
    primary: &'a GpuPrimaryContext,
    pub(super) ptr: *mut c_void,
    pub(super) capacity: usize,
}

impl Drop for PinnedHostLease<'_> {
    fn drop(&mut self) {
        self.primary
            .release_pinned_host_buffer(self.ptr, self.capacity);
    }
}

/// Process-wide registry of retained primary contexts, one per GPU id. The strong `Arc`
/// kept here retains each context for process lifetime; allocations clone it so the
/// context outlives every allocation made in it.
static GPU_PRIMARY_CONTEXTS: OnceLock<Mutex<BTreeMap<u16, Arc<GpuPrimaryContext>>>> =
    OnceLock::new();

/// Get (creating + caching on first use) the shared primary context for `gpu_id`.
pub(super) fn gpu_primary_context(
    gpu_id: u16,
) -> Result<Arc<GpuPrimaryContext>, CudaRuntimeProbeError> {
    let registry = GPU_PRIMARY_CONTEXTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut map = registry
        .lock()
        .expect("gpu primary-context registry poisoned");
    if let Some(existing) = map.get(&gpu_id) {
        return Ok(Arc::clone(existing));
    }
    let context = Arc::new(GpuPrimaryContext::create(gpu_id)?);
    map.insert(gpu_id, Arc::clone(&context));
    Ok(context)
}

#[cfg(test)]
pub(super) fn distinct_gpu_primary_context_for_test(
    gpu_id: u16,
) -> Result<Arc<GpuPrimaryContext>, CudaRuntimeProbeError> {
    Ok(Arc::new(GpuPrimaryContext::create(gpu_id)?))
}

pub(super) fn check_cuda(code: i32) -> Result<(), CudaRuntimeProbeError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaRuntimeProbeError::KernelLaunchFailed(code))
    }
}

pub(super) struct CudaContextGuard {
    pub(super) context: *mut c_void,
    pub(super) destroy: unsafe extern "C" fn(*mut c_void) -> i32,
}

impl Drop for CudaContextGuard {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context);
        }
    }
}

pub(super) struct CudaDeviceAllocationGuard {
    pub(super) ptr: u64,
    pub(super) free: unsafe extern "C" fn(u64) -> i32,
}

impl Drop for CudaDeviceAllocationGuard {
    fn drop(&mut self) {
        unsafe {
            (self.free)(self.ptr);
        }
    }
}

pub(super) struct CudaModuleGuard {
    pub(super) module: *mut c_void,
    pub(super) unload: unsafe extern "C" fn(*mut c_void) -> i32,
}

impl Drop for CudaModuleGuard {
    fn drop(&mut self) {
        unsafe {
            (self.unload)(self.module);
        }
    }
}

// Owns an event across every success/error exit of benchmark and serial-reference timing paths.
pub(super) struct CudaEventGuard {
    pub(super) event: *mut c_void,
    pub(super) destroy: unsafe extern "C" fn(*mut c_void) -> i32,
}

impl Drop for CudaEventGuard {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::checked_output_buffer_bucket;

    #[test]
    fn pooled_buffer_bucket_rounding_rejects_the_first_unroundable_request() {
        let largest_bucket = 1_usize << (usize::BITS - 1);
        assert_eq!(
            checked_output_buffer_bucket(largest_bucket),
            Some(largest_bucket)
        );
        assert_eq!(checked_output_buffer_bucket(largest_bucket + 1), None);
    }
}
