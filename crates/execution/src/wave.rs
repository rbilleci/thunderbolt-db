//! R2 wave read engine (ADR-009) — the persistent-kernel point-lookup read path, lifted into the crate
//! from the proven standalone probes (`examples/wave_devatomic_probe.rs` 1d-i) so it runs on the engine's
//! REAL shared primary context instead of a throwaway one.
//!
//! A single persistent kernel is launched once over a resident int4 table's GPU hash index (the SAME R1
//! index format `(key<<32)|(row+1)`, Fibonacci hash, 256-probe cap). Its worker threads drain a host-
//! pinned **device-mapped ring** of needle requests: each thread bounded-claims the next index via a
//! DEVICE-memory `atom.cas` (no block barriers — a barrier deadlock would evade the backstop and zombie
//! the shared context), hash-probes the index, **gathers up to `MAX_PROJECTIONS` int4 columns** at the
//! matched row (exactly like the R1 `gpu_db_resident_i32_index_probe` kernel), and writes a per-needle
//! result record (status + row_index + values). `submit_async` enqueues a wave and returns a `WaveTicket`
//! immediately; `harvest` turns a ticket into `CudaI32BatchProjectionRow`s once it drains — byte-identical
//! to the R1 index probe, one row per found needle. Pipelining many tickets (depth K) amortizes the
//! host<->device round-trip; the blocking `submit` is `submit_async` + spin-`harvest`.
//!
//! ## Completion gate (DECISIONS ADR-008 "R2 all_done ordering audit") — load-bearing
//! Counters (`claim`, `completed`) are **monotonic / cumulative** device-memory atomics, never reset; the
//! ring is **circular** (`idx & ring_mask`). `submit_async` advances a cumulative `head`; a wave is
//! complete once `completed >= base+len`. The kernel writes `completed` into a **host-mapped MIRROR** after
//! its `membar.sys` + device-`completed` bump, and `harvest` reads that mirror DIRECTLY (a local
//! system-RAM read, no DtoH — a DtoH spin congests with the kernel's PCIe polling). Because each worker
//! releases its record writes ahead of its `completed` bump (`membar.sys`), `mirror >= base+len` means
//! those records are host-visible (the audit's counter-acquire, host-mapped variant). The `all_done` flag
//! is unused now; the mirror is the gate.
//!
//! ## Bounded, clamped-batched claim (vs the probe's speculative `atom.add`)
//! Each thread CAS-claims a CLAMPED BATCH `[c, min(c+K, head))` in one `atom.cas` (K=`CLAIM_BATCH`), then
//! does ONE `membar.sys` + ONE `atom.add(completed, cnt)` for the whole batch. This amortizes the CAS +
//! system fence + completion atomic over up to K indices AND never overshoots `head` (correct for
//! incremental/continuous fill — no lost requests), unlike the probe's `atom.add(claim, K)` which
//! overshoots. The audited ordering (record writes -> `membar.sys` -> `completed` bump) is unchanged.
//!
//! ## Safety net (the `--gpu-reset`-denied-box rule, proven in probe 1a)
//! The kernel ALWAYS self-terminates: host doorbell OR a `%globaltimer` wall-clock backstop (`backstop_ns`).
//! `shutdown`/`Drop` ring the doorbell and synchronize the stream before freeing. SM-coexistence rule
//! (DECISIONS ADR-008): keep the persistent footprint small; `threads` is an explicit, modest knob.
//! ASCII-only PTX (the driver JIT rejects non-ASCII even where the local ptxas tolerates it).
//!
//! R2.2a = the data plane with multi-column projection, verified in ISOLATION vs the R1 index-probe
//! oracle; NOT wired into any engine query path yet (that is R2.2b).

// TODO(R2.2b): drop this once the engine wiring consumes `WaveReadEngine`. Until then the type is used
// only by its `#[cfg(test)]` GPU test, so a non-test build sees it as dead.
#![allow(dead_code)]

use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use libloading::Library;

use crate::{
    check_cuda, CudaI32BatchProjectionRow, CudaResidentDeviceMemory, CudaResidentReadSource,
    CudaRuntimeProbeError, GpuPrimaryContext,
};

const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
const CU_STREAM_NON_BLOCKING: u32 = 0x01;
/// Control block bytes (device-mapped pinned), TWO cachelines to avoid false sharing: cacheline 0 =
/// [doorbell@0, head@4, all_done@8] (GPU polls these every iteration); cacheline 1 = [completed_mirror@64]
/// (the host tight-spins reading this in `harvest`). Keeping the host-spun mirror off the GPU-polled
/// cacheline stops the host spin from starving the kernel's head/doorbell PCIe reads.
const CTRL_BYTES: usize = 128;
/// Byte offset of the host-mapped `completed` mirror (its own cacheline).
const COMPLETED_MIRROR_OFFSET: usize = 64;
/// Max projected int4 columns per needle (matches the R1 index probe's `MAX_PROJECTIONS`).
const MAX_PROJECTIONS: usize = 4;
/// Per-needle result record (device-mapped): [status@0 u32, pad@4, row@8 u64, v0@16, v1@20, v2@24, v3@28].
/// status: 0 = incomplete (bug), 1 = found, 2 = not found.
const RES_SLOT_BYTES: usize = 32;
/// Indices a thread claims per `atom.cas` (clamped to `head`). Amortizes the CAS + `membar.sys` +
/// completion atomic over up to K indices (the probe's 1d-ii sweet spot was K=8); no overshoot.
const CLAIM_BATCH: u32 = 8;

/// Handle to an in-flight `submit_async` wave: its cumulative ring range `[base, base+len)`. `harvest`
/// turns it into rows once `completed >= base + len`. `Copy` so the host can queue many in flight.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WaveTicket {
    base: u32,
    len: u32,
}
/// Host-side timeout waiting for a wave to drain (the kernel's own backstop is the device-side net).
const DRAIN_TIMEOUT: Duration = Duration::from_secs(20);

// FFI signatures resolved ad-hoc from the primary context's `Library` (the crate idiom).
// `cuMemHostGetDevicePointer` is not bound as a `GpuPrimaryContext` field; the device-mapped buffers need
// it to hand the kernel a device pointer.
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFree = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
type CuMemHostGetDevicePointer = unsafe extern "C" fn(*mut u64, *mut c_void, u32) -> i32;
type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> i32;
type CuStreamCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
type CuStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> i32;
type CuStreamDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
#[allow(clippy::type_complexity)]
type CuLaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

/// The persistent multi-projection data-plane kernel (proper port). Each thread CAS-claims a CLAMPED
/// BATCH `[c, min(c+K, head))` in one `atom.cas` (no overshoot — correct for incremental fill), processes
/// that range (R1's 4-way-unrolled multi-column gather, per-needle record [status,row,v0..v3]), then does
/// ONE `membar.sys` + ONE `atom.add(completed, cnt)` per batch and writes `completed` to a host-mapped
/// MIRROR (`ctrl+64`, own cacheline) for the host to read without a DtoH. record-writes -> `membar.sys` -> `completed`
/// bump (-> mirror) ordering preserves the `all_done` ordering audit's counter-acquire. Pure ASCII.
const WAVE_DATAPLANE_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_wave_read_dataplane(
    .param .u64 ctrl,
    .param .u64 counters,
    .param .u64 req,
    .param .u64 res,
    .param .u64 resident,
    .param .u64 index,
    .param .u32 hash_shift,
    .param .u32 table_mask,
    .param .u32 ring_mask,
    .param .u32 proj_count,
    .param .u64 proj_off0,
    .param .u64 proj_off1,
    .param .u64 proj_off2,
    .param .u64 proj_off3,
    .param .u32 claim_batch,
    .param .u64 max_ns
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<64>;

    ld.param.u64 %rd1, [ctrl];
    ld.param.u64 %rd2, [counters];
    ld.param.u64 %rd3, [req];
    ld.param.u64 %rd4, [res];
    ld.param.u64 %rd5, [resident];
    ld.param.u64 %rd6, [index];
    ld.param.u32 %r1, [hash_shift];
    ld.param.u32 %r2, [table_mask];
    ld.param.u32 %r3, [ring_mask];
    ld.param.u32 %r4, [proj_count];
    ld.param.u64 %rd7, [proj_off0];
    ld.param.u64 %rd30, [proj_off1];
    ld.param.u64 %rd31, [proj_off2];
    ld.param.u64 %rd32, [proj_off3];
    ld.param.u32 %r5, [claim_batch];
    ld.param.u64 %rd8, [max_ns];
    mov.u64 %rd9, %globaltimer;

$L_loop:
    ld.volatile.global.u32 %r6, [%rd1];
    setp.ne.s32 %p1, %r6, 0;
    @%p1 bra $L_done;
    mov.u64 %rd10, %globaltimer;
    sub.u64 %rd11, %rd10, %rd9;
    setp.ge.u64 %p1, %rd11, %rd8;
    @%p1 bra $L_done;
    ld.volatile.global.u32 %r7, [%rd1+4];
    ld.volatile.global.u32 %r8, [%rd2];
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra $L_loop;
    add.u32 %r9, %r8, %r5;
    min.u32 %r9, %r9, %r7;
    atom.global.cas.b32 %r10, [%rd2], %r8, %r9;
    setp.ne.u32 %p1, %r10, %r8;
    @%p1 bra $L_loop;
    sub.u32 %r11, %r9, %r8;
    mov.u32 %r12, %r8;

$L_proc:
    setp.ge.u32 %p2, %r12, %r9;
    @%p2 bra $L_after;
    and.b32 %r13, %r12, %r3;
    mul.wide.u32 %rd12, %r13, 4;
    add.u64 %rd13, %rd3, %rd12;
    ld.volatile.global.u32 %r14, [%rd13];
    mul.wide.u32 %rd14, %r13, 32;
    add.u64 %rd15, %rd4, %rd14;
    mul.lo.u32 %r15, %r14, 2654435761;
    shr.u32 %r15, %r15, %r1;
    mov.u32 %r16, 0;
    mov.u32 %r17, 2;

$L_probe:
    and.b32 %r15, %r15, %r2;
    mul.wide.u32 %rd16, %r15, 8;
    add.u64 %rd17, %rd6, %rd16;
    ld.global.u64 %rd18, [%rd17];
    setp.eq.u64 %p3, %rd18, 0;
    @%p3 bra $L_write;
    shr.u64 %rd19, %rd18, 32;
    cvt.u32.u64 %r18, %rd19;
    setp.ne.s32 %p3, %r18, %r14;
    @%p3 bra $L_probe_next;
    cvt.u32.u64 %r19, %rd18;
    sub.u32 %r19, %r19, 1;
    cvt.u64.u32 %rd20, %r19;
    mov.u32 %r17, 1;
    st.global.u64 [%rd15+8], %rd20;
    mul.lo.u64 %rd21, %rd20, 4;
    add.u64 %rd22, %rd5, %rd7;
    add.u64 %rd22, %rd22, %rd21;
    ld.global.s32 %r20, [%rd22];
    st.global.s32 [%rd15+16], %r20;
    setp.le.u32 %p4, %r4, 1;
    @%p4 bra $L_write;
    add.u64 %rd22, %rd5, %rd30;
    add.u64 %rd22, %rd22, %rd21;
    ld.global.s32 %r20, [%rd22];
    st.global.s32 [%rd15+20], %r20;
    setp.le.u32 %p4, %r4, 2;
    @%p4 bra $L_write;
    add.u64 %rd22, %rd5, %rd31;
    add.u64 %rd22, %rd22, %rd21;
    ld.global.s32 %r20, [%rd22];
    st.global.s32 [%rd15+24], %r20;
    setp.le.u32 %p4, %r4, 3;
    @%p4 bra $L_write;
    add.u64 %rd22, %rd5, %rd32;
    add.u64 %rd22, %rd22, %rd21;
    ld.global.s32 %r20, [%rd22];
    st.global.s32 [%rd15+28], %r20;
    bra $L_write;

$L_probe_next:
    add.u32 %r15, %r15, 1;
    add.u32 %r16, %r16, 1;
    setp.ge.u32 %p4, %r16, 256;
    @%p4 bra $L_write;
    bra $L_probe;

$L_write:
    st.global.u32 [%rd15], %r17;
    add.u32 %r12, %r12, 1;
    bra $L_proc;

$L_after:
    membar.sys;
    atom.global.add.u32 %r21, [%rd2+4], %r11;
    add.u32 %r21, %r21, %r11;
    st.volatile.global.u32 [%rd1+64], %r21;     // host-mapped completed MIRROR on its OWN cacheline (harvest reads it, no DtoH)
    setp.ne.u32 %p5, %r21, %r7;
    @%p5 bra $L_loop;
    membar.sys;
    mov.u32 %r22, 1;
    st.volatile.global.u32 [%rd1+8], %r22;
    bra $L_loop;

$L_done:
    ret;
}
"#;

/// Resolve a CUDA driver symbol from `lib` (trying each name in order) and COPY OUT the bare fn pointer.
///
/// # Safety
/// The returned fn pointer is valid only while `lib` stays loaded. The caller stores it inside a
/// `WaveReadEngine` that also holds the `Arc<GpuPrimaryContext>` (hence the `Arc<Library>`), so the
/// library outlives every stored pointer.
unsafe fn sym<T: Copy>(lib: &Library, names: &[&[u8]]) -> Result<T, CudaRuntimeProbeError> {
    for name in names {
        if let Ok(symbol) = lib.get::<T>(name) {
            return Ok(*symbol);
        }
    }
    Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
}

/// A persistent GPU "wave" read kernel resident on the engine's shared primary context, draining a
/// circular needle ring against one resident int4 table's hash index, projecting up to `MAX_PROJECTIONS`
/// columns. `submit` runs one wave at a time (serialized) and reads its rows back.
pub(crate) struct WaveReadEngine {
    /// Keep the shared context + the index/table device buffers alive for the engine's lifetime.
    _primary: Arc<GpuPrimaryContext>,
    _index: Arc<CudaResidentDeviceMemory>,
    _resident: Arc<CudaResidentDeviceMemory>,
    stream: *mut c_void,
    /// Device-mapped pinned control block (host view): [doorbell@0, head@4, all_done@8].
    ctrl_host: *mut c_void,
    /// Device-mapped pinned needle ring (host view): i32[ring_capacity].
    req_host: *mut c_void,
    /// Device-mapped pinned result ring (host view): RES_SLOT_BYTES * ring_capacity bytes.
    res_host: *mut c_void,
    /// Device counters [claim@0, completed@4] (monotonic; never reset).
    counters: u64,
    ring_capacity: usize,
    ring_mask: u32,
    proj_count: u32,
    /// Cumulative count of needles ever published (host-tracked; mirrors ctrl.head). Monotonic.
    head: u32,
    cu_stream_synchronize: CuStreamSynchronize,
    cu_stream_destroy: CuStreamDestroy,
    cu_mem_free: CuMemFree,
    cu_mem_free_host: CuMemFreeHost,
    shut: bool,
}

impl WaveReadEngine {
    /// Launch the persistent kernel over `index` (built with `table_mask`/`hash_shift`) and `resident`
    /// (the table whose columns are gathered at `projection_offsets[j] + row*4`), on the SAME shared
    /// context the `index` lives in. `projection_offsets` must be 1..=MAX_PROJECTIONS. `ring_capacity` is
    /// rounded up to a power of two and bounds a single wave. `threads` is the persistent grid (kept
    /// modest per the SM-coexistence rule). Runs until `shutdown`/`Drop`, or `backstop_ns` elapses.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        index: Arc<CudaResidentDeviceMemory>,
        resident: Arc<CudaResidentDeviceMemory>,
        projection_offsets: &[u64],
        table_mask: u32,
        hash_shift: u32,
        ring_capacity: usize,
        threads: u32,
        backstop_ns: u64,
    ) -> Result<Self, CudaRuntimeProbeError> {
        if projection_offsets.is_empty() || projection_offsets.len() > MAX_PROJECTIONS {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                projection_offsets.len(),
            ));
        }
        let proj_count = projection_offsets.len() as u32;
        let mut proj = [0u64; MAX_PROJECTIONS];
        proj[..projection_offsets.len()].copy_from_slice(projection_offsets);

        let ring_capacity = ring_capacity.next_power_of_two().max(8);
        let ring_mask = (ring_capacity - 1) as u32;
        let threads = threads.max(1);

        let primary = index.primary_arc();
        primary.set_current()?;
        let lib = primary.lib();

        let cu_mem_alloc: CuMemAlloc = unsafe { sym(lib, &[b"cuMemAlloc_v2\0", b"cuMemAlloc\0"]) }?;
        let cu_mem_free: CuMemFree = unsafe { sym(lib, &[b"cuMemFree_v2\0", b"cuMemFree\0"]) }?;
        let cu_memcpy_htod: CuMemcpyHtoD =
            unsafe { sym(lib, &[b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0"]) }?;
        let cu_mem_host_alloc: CuMemHostAlloc = unsafe { sym(lib, &[b"cuMemHostAlloc\0"]) }?;
        let cu_mem_host_get_device_pointer: CuMemHostGetDevicePointer = unsafe {
            sym(
                lib,
                &[
                    b"cuMemHostGetDevicePointer_v2\0",
                    b"cuMemHostGetDevicePointer\0",
                ],
            )
        }?;
        let cu_mem_free_host: CuMemFreeHost = unsafe { sym(lib, &[b"cuMemFreeHost\0"]) }?;
        let cu_stream_create: CuStreamCreate = unsafe { sym(lib, &[b"cuStreamCreate\0"]) }?;
        let cu_stream_synchronize: CuStreamSynchronize =
            unsafe { sym(lib, &[b"cuStreamSynchronize\0"]) }?;
        let cu_stream_destroy: CuStreamDestroy =
            unsafe { sym(lib, &[b"cuStreamDestroy_v2\0", b"cuStreamDestroy\0"]) }?;
        let cu_launch_kernel: CuLaunchKernel = unsafe { sym(lib, &[b"cuLaunchKernel\0"]) }?;

        let mut ptx = Vec::with_capacity(WAVE_DATAPLANE_PTX.len() + 1);
        ptx.extend_from_slice(WAVE_DATAPLANE_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_wave_read_dataplane", &ptx)?;

        let resident_ptr = resident.device_ptr();
        let index_ptr = index.device_ptr();
        let blocks = threads.div_ceil(256);
        let threads_per_block = threads.min(256);

        let mut counters: u64 = 0;
        let mut ctrl_host: *mut c_void = ptr::null_mut();
        let mut req_host: *mut c_void = ptr::null_mut();
        let mut res_host: *mut c_void = ptr::null_mut();
        let mut stream: *mut c_void = ptr::null_mut();
        let setup = (|| -> Result<(), CudaRuntimeProbeError> {
            check_cuda(unsafe { cu_mem_alloc(&mut counters, 8) })?;
            let zero = [0u32; 2];
            check_cuda(unsafe { cu_memcpy_htod(counters, zero.as_ptr().cast::<c_void>(), 8) })?;

            check_cuda(unsafe {
                cu_mem_host_alloc(&mut ctrl_host, CTRL_BYTES, CU_MEMHOSTALLOC_DEVICEMAP)
            })?;
            unsafe {
                for i in 0..(CTRL_BYTES / 4) {
                    ptr::write_volatile((ctrl_host as *mut u32).add(i), 0);
                }
            }
            check_cuda(unsafe {
                cu_mem_host_alloc(&mut req_host, ring_capacity * 4, CU_MEMHOSTALLOC_DEVICEMAP)
            })?;
            check_cuda(unsafe {
                cu_mem_host_alloc(
                    &mut res_host,
                    ring_capacity * RES_SLOT_BYTES,
                    CU_MEMHOSTALLOC_DEVICEMAP,
                )
            })?;
            unsafe {
                ptr::write_bytes(res_host as *mut u8, 0, ring_capacity * RES_SLOT_BYTES);
            }
            fence(Ordering::SeqCst);

            let mut ctrl_dptr: u64 = 0;
            let mut req_dptr: u64 = 0;
            let mut res_dptr: u64 = 0;
            check_cuda(unsafe { cu_mem_host_get_device_pointer(&mut ctrl_dptr, ctrl_host, 0) })?;
            check_cuda(unsafe { cu_mem_host_get_device_pointer(&mut req_dptr, req_host, 0) })?;
            check_cuda(unsafe { cu_mem_host_get_device_pointer(&mut res_dptr, res_host, 0) })?;

            check_cuda(unsafe { cu_stream_create(&mut stream, CU_STREAM_NON_BLOCKING) })?;

            let mut a_ctrl = ctrl_dptr;
            let mut a_counters = counters;
            let mut a_req = req_dptr;
            let mut a_res = res_dptr;
            let mut a_resident = resident_ptr;
            let mut a_index = index_ptr;
            let mut a_shift = hash_shift;
            let mut a_tmask = table_mask;
            let mut a_rmask = ring_mask;
            let mut a_pcount = proj_count;
            let mut a_off0 = proj[0];
            let mut a_off1 = proj[1];
            let mut a_off2 = proj[2];
            let mut a_off3 = proj[3];
            let mut a_claim_batch = CLAIM_BATCH;
            let mut a_max = backstop_ns;
            let mut params = [
                (&mut a_ctrl as *mut u64).cast::<c_void>(),
                (&mut a_counters as *mut u64).cast::<c_void>(),
                (&mut a_req as *mut u64).cast::<c_void>(),
                (&mut a_res as *mut u64).cast::<c_void>(),
                (&mut a_resident as *mut u64).cast::<c_void>(),
                (&mut a_index as *mut u64).cast::<c_void>(),
                (&mut a_shift as *mut u32).cast::<c_void>(),
                (&mut a_tmask as *mut u32).cast::<c_void>(),
                (&mut a_rmask as *mut u32).cast::<c_void>(),
                (&mut a_pcount as *mut u32).cast::<c_void>(),
                (&mut a_off0 as *mut u64).cast::<c_void>(),
                (&mut a_off1 as *mut u64).cast::<c_void>(),
                (&mut a_off2 as *mut u64).cast::<c_void>(),
                (&mut a_off3 as *mut u64).cast::<c_void>(),
                (&mut a_claim_batch as *mut u32).cast::<c_void>(),
                (&mut a_max as *mut u64).cast::<c_void>(),
            ];
            check_cuda(unsafe {
                cu_launch_kernel(
                    function,
                    blocks,
                    1,
                    1,
                    threads_per_block,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            })?;
            Ok(())
        })();

        if let Err(err) = setup {
            unsafe {
                if !stream.is_null() {
                    cu_stream_destroy(stream);
                }
                if !res_host.is_null() {
                    cu_mem_free_host(res_host);
                }
                if !req_host.is_null() {
                    cu_mem_free_host(req_host);
                }
                if !ctrl_host.is_null() {
                    cu_mem_free_host(ctrl_host);
                }
                if counters != 0 {
                    cu_mem_free(counters);
                }
            }
            return Err(err);
        }

        Ok(Self {
            _primary: primary,
            _index: index,
            _resident: resident,
            stream,
            ctrl_host,
            req_host,
            res_host,
            counters,
            ring_capacity,
            ring_mask,
            proj_count,
            head: 0,
            cu_stream_synchronize,
            cu_stream_destroy,
            cu_mem_free,
            cu_mem_free_host,
            shut: false,
        })
    }

    /// Enqueue a wave of `needles` (each a key value) and return IMMEDIATELY (non-blocking) with a
    /// `WaveTicket`. The host can keep MANY tickets in flight and `harvest` them as they drain — the only
    /// way the persistent kernel amortizes its host<->device round-trip (review directive B). The caller
    /// must bound total un-harvested needles to `ring_capacity` so the circular ring never overwrites an
    /// un-read wave. `needle_index` in the harvested rows is the position within THIS `needles` slice.
    pub(crate) fn submit_async(
        &mut self,
        needles: &[i32],
    ) -> Result<WaveTicket, CudaRuntimeProbeError> {
        if needles.is_empty() {
            return Ok(WaveTicket {
                base: self.head,
                len: 0,
            });
        }
        if needles.len() > self.ring_capacity {
            return Err(CudaRuntimeProbeError::InvalidInputLength(needles.len()));
        }
        let base = self.head;
        let req = self.req_host as *mut i32;
        let res = self.res_host as *mut u8;
        for (i, &needle) in needles.iter().enumerate() {
            let slot = (base as usize + i) & (self.ring_mask as usize);
            unsafe {
                ptr::write_volatile(req.add(slot), needle);
                // Clear the status word so a stale (reused-slot) record can't read as complete.
                ptr::write_volatile(res.add(slot * RES_SLOT_BYTES).cast::<u32>(), 0);
            }
        }
        fence(Ordering::SeqCst);
        let new_head = base + needles.len() as u32;
        self.head = new_head;
        unsafe { ptr::write_volatile((self.ctrl_host as *mut u32).add(1), new_head) }; // publish head
        fence(Ordering::SeqCst);
        Ok(WaveTicket {
            base,
            len: needles.len() as u32,
        })
    }

    /// Non-blocking completion check + read for a `submit_async` ticket. Reads the HOST-MAPPED `completed`
    /// mirror directly — a local system-RAM read (~ns), NOT a DtoH (a PCIe round-trip that also congests
    /// with the kernel's polling; the review flagged the DtoH spin). The kernel writes the mirror AFTER
    /// its `membar.sys` + device `completed` bump, so `mirror >= base+len` means this wave's records are
    /// host-visible (the audit's counter-acquire, host-mapped variant). If ready, read + return the rows
    /// (found needles only, byte-identical to the R1 index probe); else `None`. The mirror is last-writer-
    /// wins (`st.volatile`, untorn 4-byte) and can briefly under-read under contention — safe (an
    /// under-read only delays, never reads records early); the caller just retries.
    pub(crate) fn harvest(
        &self,
        ticket: WaveTicket,
    ) -> Result<Option<Vec<CudaI32BatchProjectionRow>>, CudaRuntimeProbeError> {
        if ticket.len == 0 {
            return Ok(Some(Vec::new()));
        }
        let completed = unsafe {
            ptr::read_volatile((self.ctrl_host as *const u8).add(COMPLETED_MIRROR_OFFSET).cast::<u32>())
        };
        if completed.wrapping_sub(ticket.base) < ticket.len {
            return Ok(None);
        }
        Ok(Some(self.read_records(ticket.base, ticket.len)))
    }

    /// Read the per-needle records for a fully-drained wave `[base, base+len)` into rows (found needles
    /// only). Caller MUST have confirmed completion (the `harvest` gate) before calling.
    fn read_records(&self, base: u32, len: u32) -> Vec<CudaI32BatchProjectionRow> {
        let res = self.res_host as *mut u8;
        let mut rows = Vec::new();
        for i in 0..len as usize {
            let slot = (base as usize + i) & (self.ring_mask as usize);
            let rec = unsafe { res.add(slot * RES_SLOT_BYTES) };
            let status = unsafe { ptr::read_volatile(rec.cast::<u32>()) };
            if status != 1 {
                continue; // 2 = not found (no row, like the R1 index probe); 0 would be a bug
            }
            let row_index = unsafe { ptr::read_volatile(rec.add(8).cast::<u64>()) };
            let mut values = Vec::with_capacity(self.proj_count as usize);
            for j in 0..self.proj_count as usize {
                values.push(unsafe { ptr::read_volatile(rec.add(16 + j * 4).cast::<i32>()) });
            }
            rows.push(CudaI32BatchProjectionRow {
                needle_index: i,
                row_index,
                values,
            });
        }
        rows
    }

    /// Blocking single-wave submit (`submit_async` + spin-`harvest`). Kept for correctness tests and
    /// request-response callers; for THROUGHPUT use `submit_async`/`harvest` pipelined (depth K).
    pub(crate) fn submit(
        &mut self,
        needles: &[i32],
    ) -> Result<Vec<CudaI32BatchProjectionRow>, CudaRuntimeProbeError> {
        let ticket = self.submit_async(needles)?;
        let deadline = Instant::now() + DRAIN_TIMEOUT;
        loop {
            if let Some(rows) = self.harvest(ticket)? {
                return Ok(rows);
            }
            if Instant::now() >= deadline {
                return Err(CudaRuntimeProbeError::KernelLaunchFailed(-1));
            }
            std::hint::spin_loop();
        }
    }

    /// Ring the doorbell, wait for the kernel to exit, then free the stream + buffers. Idempotent.
    pub(crate) fn shutdown(&mut self) {
        if self.shut {
            return;
        }
        self.shut = true;
        unsafe { ptr::write_volatile(self.ctrl_host as *mut u32, 1) }; // doorbell
        fence(Ordering::SeqCst);
        unsafe {
            (self.cu_stream_synchronize)(self.stream);
            (self.cu_stream_destroy)(self.stream);
            (self.cu_mem_free_host)(self.res_host);
            (self.cu_mem_free_host)(self.req_host);
            (self.cu_mem_free_host)(self.ctrl_host);
            (self.cu_mem_free)(self.counters);
        }
    }
}

impl Drop for WaveReadEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDriverRuntime;

    #[test]
    fn wave_dataplane_ptx_is_pure_ascii() {
        // The driver JIT's ptxas rejects a non-ASCII byte (INVALID_PTX 218) even where the local ptxas
        // tolerates it, so a stray em-dash / smart-quote in a comment fails every launch. Keep it ASCII.
        if let Some(pos) = WAVE_DATAPLANE_PTX.iter().position(|&byte| !byte.is_ascii()) {
            let line = WAVE_DATAPLANE_PTX[..pos]
                .iter()
                .filter(|&&byte| byte == b'\n')
                .count()
                + 1;
            panic!("WAVE_DATAPLANE_PTX has a non-ASCII byte at offset {pos} (line {line})");
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_engine_point_lookups_match_index_probe_oracle() {
        // Build a tiny resident int4 table + its R1-format hash index on the SHARED primary context, run
        // the persistent wave engine over it with a 2-column projection, and assert every wave's rows are
        // BYTE-IDENTICAL to the R1 index probe (`submit_match_project_i32_index_probe_from_payload`). Two
        // waves exercise the cumulative ring.
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return; // no GPU -> skip (also #[ignore]d by default)
        };
        let rows: u64 = 1000;
        let keys: Vec<i32> = (0..rows as i32).map(|r| r * 3 + 1).collect();
        let payload: Vec<i32> = (0..rows as i32).map(|r| r * 1000 + 7).collect();
        let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
        for &k in &keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &v in &payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &buf)
                .expect("resident device memory"),
        );
        let key_offset = 0_u64;
        let payload_offset = rows * 4;
        let projections = [key_offset, payload_offset];

        let table_size = ((rows * 2) as u32).next_power_of_two();
        let table_mask = table_size - 1;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (r, &k) in keys.iter().enumerate() {
            let key = k as u32;
            let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
            while index[h as usize] != 0 {
                h = (h + 1) & table_mask;
            }
            index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
        let index_resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &index_bytes)
                .expect("index device memory"),
        );

        // The R1 index probe is the oracle.
        let oracle = |needles: &[i32]| -> Vec<CudaI32BatchProjectionRow> {
            let submission = resident
                .submit_match_project_i32_index_probe_from_payload(
                    &index_resident,
                    table_mask,
                    hash_shift,
                    needles,
                    &projections,
                    rows,
                )
                .expect("index submit");
            let mut r = submission.complete(&resident).expect("index complete");
            r.sort_by_key(|row| (row.needle_index, row.row_index));
            r
        };

        let mut engine = WaveReadEngine::new(
            Arc::clone(&index_resident),
            Arc::clone(&resident),
            &projections,
            table_mask,
            hash_shift,
            1024,
            1024,
            30_000_000_000,
        )
        .expect("wave engine launches on the shared context");

        let waves = [
            vec![keys[5], keys[100], keys[999], -12345, keys[0]],
            vec![keys[10], keys[20], 2, keys[30]],
        ];
        // Run all waves on the persistent kernel FIRST (no interleaved GPU launches), then validate vs
        // the R1 oracle — isolates multi-wave persistence from any concurrent-launch interaction.
        let mut results = Vec::new();
        for needles in &waves {
            let mut got = engine.submit(needles).expect("wave submit");
            got.sort_by_key(|row| (row.needle_index, row.row_index));
            results.push(got);
        }
        // Non-vacuous spot-check gather (also a 3rd wave) — collected BEFORE any GPU oracle launch.
        let spot = engine.submit(&[keys[5]]).expect("spot submit");
        engine.shutdown();

        for (got, needles) in results.iter().zip(waves.iter()) {
            let want = oracle(needles);
            assert_eq!(
                *got, want,
                "wave rows must be byte-identical to the R1 index probe for {needles:?}"
            );
        }
        assert_eq!(spot.len(), 1, "present key -> exactly one row");
        assert_eq!(
            spot[0].values,
            vec![keys[5], payload[5]],
            "2-column gather (key, payload) at row 5"
        );
    }

    /// R2.2 diagnostic — persistent wave vs launch-per-batch R1 index probe, swept over (large) batch.
    /// CAVEAT (DECISIONS ADR-008 "R2.2 evidence gate CORRECTION"): this is SINGLE-THREADED serialized
    /// submit-and-wait (still blocking) — NOT the wave's intended pipelined/concurrent regime; the depth-K
    /// pipelined benchmark (proper-port step 3) is the real evidence gate. The right baseline is the 156k
    /// single-coalescer BATCHER cap (the production concurrent path), not the raw 1-thread index probe
    /// (7-24M, a microbenchmark ceiling). With the OPTIMIZED drain (clamped-batched CAS claim + amortized
    /// membar) the wave drain is now ~2.45M/s (was ~520k), already ~16x the batcher's 156k. Scaling peaks
    /// near 128 threads — CAS contention on the single `claim` counter is the remaining limiter (sharded
    /// counters would lift it toward the probe's 45M); more threads make it worse.
    /// Wave is measured FIRST then shut down, THEN the index probe — so the index probe's `cuMemAlloc`
    /// (which device-syncs) never runs while the wave kernel is live (the freeze root cause).
    ///
    /// Run: `GPU_DB_WAVE_BENCH_ROWS=1000000 cargo test -p gpu_db_execution \
    ///   wave::tests::wave_vs_launch_per_batch_throughput -- --ignored --nocapture`
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_vs_launch_per_batch_throughput() {
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return;
        };
        let rows: u64 = std::env::var("GPU_DB_WAVE_BENCH_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000);
        // Persistent-kernel thread count (PCIe-polling congestion suspect — sweep it).
        // NOTE: this naive port is CONGESTION-bound (per-needle host-mapped record write + `membar.sys`
        // per needle + per-needle CAS) — MORE threads make it WORSE (256+ times out on large batches),
        // unlike the optimized probe (batched claiming + device atomics) which scaled to 8192. 128 is the
        // sweet spot here. The probe hit ~45M/s; this port plateaus ~520k/s (~85x slower) — see the
        // "R2.2 evidence gate CORRECTION" in DECISIONS: the wave engine was NOT fairly tested.
        let wave_threads: u32 = std::env::var("GPU_DB_WAVE_BENCH_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        // Adaptive iters so total work per batch size is ~constant (large single waves = continuous fill).
        let target_lookups: usize = 2_000_000;
        let keys: Vec<i32> = (0..rows as i32).map(|r| r.wrapping_mul(3).wrapping_add(1)).collect();
        let payload: Vec<i32> = (0..rows as i32).map(|r| r.wrapping_mul(1000).wrapping_add(7)).collect();
        let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
        for &k in &keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &v in &payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let resident = Arc::new(runtime.retain_device_memory_copy(0, &buf).expect("resident"));
        let payload_offset = rows * 4;
        let projections = [payload_offset]; // "SELECT payload WHERE key = ?"

        let table_size = ((rows * 2) as u32).next_power_of_two();
        let table_mask = table_size - 1;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (r, &k) in keys.iter().enumerate() {
            let key = k as u32;
            let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
            while index[h as usize] != 0 {
                h = (h + 1) & table_mask;
            }
            index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
        let index_resident = Arc::new(runtime.retain_device_memory_copy(0, &index_bytes).expect("index"));

        // SMALL realistic point-lookup batches (where lpb is launch-overhead-bound). The wave is drain-
        // bound (persistent kernel, no per-batch launch), so it sustains throughput as the batch shrinks
        // while lpb falls off; plus 65536 for the wave drain ceiling.
        let batch_sizes = [1usize, 8, 32, 256, 65536];
        let ring_capacity = 131_072usize;
        let iters_for = |batch: usize| (target_lookups / batch).clamp(20, 50_000);
        let needles_for = |batch: usize, iter: usize| -> Vec<i32> {
            (0..batch)
                .map(|i| keys[(iter.wrapping_mul(batch).wrapping_add(i)) % rows as usize])
                .collect()
        };

        // --- Persistent wave engine (alloc-free submits) ---
        let mut wave_lps = [0f64; 5];
        {
            let mut engine = WaveReadEngine::new(
                Arc::clone(&index_resident),
                Arc::clone(&resident),
                &projections,
                table_mask,
                hash_shift,
                ring_capacity,
                wave_threads,
                30_000_000_000,
            )
            .expect("wave engine");
            for (bi, &batch) in batch_sizes.iter().enumerate() {
                let iters = iters_for(batch);
                // A timeout (Err) records 0 for this config rather than crashing (high thread counts can
                // congest the persistent kernel past the drain deadline — itself a finding).
                if (0..4).any(|w| engine.submit(&needles_for(batch, w)).is_err()) {
                    wave_lps[bi] = 0.0;
                    continue;
                }
                let t = Instant::now();
                let mut ok = true;
                for it in 0..iters {
                    if engine.submit(&needles_for(batch, it)).is_err() {
                        ok = false;
                        break;
                    }
                }
                let secs = t.elapsed().as_secs_f64();
                wave_lps[bi] = if ok { (iters * batch) as f64 / secs } else { 0.0 };
            }
            engine.shutdown(); // MUST shut down before the index probe's cuMemAlloc (freeze root cause)
        }

        // --- Launch-per-batch R1 index probe (cuMemAlloc safe now: no wave kernel alive) ---
        let mut lpb_lps = [0f64; 5];
        for (bi, &batch) in batch_sizes.iter().enumerate() {
            let iters = iters_for(batch);
            for w in 0..4 {
                let s = resident
                    .submit_match_project_i32_index_probe_from_payload(
                        &index_resident, table_mask, hash_shift, &needles_for(batch, w), &projections, rows,
                    )
                    .expect("warmup submit");
                let _ = s.complete(&resident).expect("warmup complete");
            }
            let t = Instant::now();
            for it in 0..iters {
                let s = resident
                    .submit_match_project_i32_index_probe_from_payload(
                        &index_resident, table_mask, hash_shift, &needles_for(batch, it), &projections, rows,
                    )
                    .expect("lpb submit");
                let _ = s.complete(&resident).expect("lpb complete");
            }
            let secs = t.elapsed().as_secs_f64();
            lpb_lps[bi] = (iters * batch) as f64 / secs;
        }

        println!("\n# R2.2 gate: persistent WAVE vs LAUNCH-PER-BATCH index probe  rows={rows} wave_threads={wave_threads}");
        println!(
            "  {:>6}  {:>16}  {:>16}  {:>9}  {:>12}  {:>12}",
            "batch", "wave lookups/s", "lpb lookups/s", "speedup", "wave us/sub", "lpb us/call"
        );
        for (bi, &batch) in batch_sizes.iter().enumerate() {
            let speedup = if lpb_lps[bi] > 0.0 { wave_lps[bi] / lpb_lps[bi] } else { 0.0 };
            let wave_us = 1.0e6 / (wave_lps[bi] / batch as f64);
            let lpb_us = 1.0e6 / (lpb_lps[bi] / batch as f64);
            println!(
                "  {batch:>6}  {:>16.0}  {:>16.0}  {speedup:>8.2}x  {wave_us:>11.2}  {lpb_us:>11.2}",
                wave_lps[bi], lpb_lps[bi]
            );
        }
        println!("# large batch = continuous-fill (the wave's intended regime); batcher ~156k single-coalescer cap.");
    }
}
