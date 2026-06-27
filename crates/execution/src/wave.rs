//! R2 wave read engine (ADR-009) — the persistent-kernel point-lookup read path, lifted into the crate
//! from the proven standalone probes (`examples/wave_devatomic_probe.rs` 1d-i) so it runs on the engine's
//! REAL shared primary context instead of a throwaway one.
//!
//! A single persistent kernel is launched once over a resident int4 table's GPU hash index (the SAME R1
//! index format `(key<<32)|(row+1)`, Fibonacci hash, 256-probe cap). Its worker threads drain a host-
//! pinned **device-mapped ring** of needle requests: each thread lock-free-claims the next index via a
//! DEVICE-memory `atom.add` (no block barriers — a barrier deadlock would evade the backstop and zombie
//! the shared context), hash-probes the index, gathers the payload, and writes a packed `(value<<32)|done`
//! result slot. `submit` enqueues a wave of needles and reads the slots back — no per-request host
//! orchestration (the wave model). This is the same kernel the R2 `all_done` ordering audit covered.
//!
//! ## Completion gate (DECISIONS ADR-008 "R2 all_done ordering audit") — load-bearing
//! Counters (`claim`, `completed`) are **monotonic / cumulative** device-memory atomics, never reset; the
//! ring is **circular** (`idx & ring_mask`). Each `submit` advances a cumulative `head` and waits until
//! `completed == head` read back via **`cuMemcpyDtoH` (the host ACQUIRING the device counter — the proven
//! 1b pattern)** before reading any result slot. The host-mapped `all_done` flag is ONLY a wake hint to
//! avoid busy-polling DtoH; it carries no happens-before for the workers' slot stores, so it is NEVER the
//! gate. Each worker releases its slot store ahead of its `completed` bump via `membar.sys`.
//!
//! ## Safety net (the `--gpu-reset`-denied-box rule, proven in probe 1a)
//! The kernel ALWAYS self-terminates: host doorbell OR a `%globaltimer` wall-clock backstop (`backstop_ns`).
//! `shutdown`/`Drop` ring the doorbell and synchronize the stream before freeing. SM-coexistence rule
//! (DECISIONS ADR-008): keep the persistent footprint small; `threads` is an explicit, modest knob.
//! ASCII-only PTX (the driver JIT rejects non-ASCII even where the local ptxas tolerates it).
//!
//! Sub-step 2 = the data plane, verified in ISOLATION (a GPU test vs a CPU oracle); NOT wired into any
//! engine query path yet (that is R2.2).

// TODO(R2.2): drop this once the engine wiring consumes `WaveReadEngine`. Until then the type is used
// only by its `#[cfg(test)]` GPU test, so a non-test build sees it as dead.
#![allow(dead_code)]

use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use libloading::Library;

use crate::{
    check_cuda, CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError,
    GpuPrimaryContext,
};

const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
const CU_STREAM_NON_BLOCKING: u32 = 0x01;
/// Control block bytes (device-mapped pinned): [doorbell@0, head@4, all_done@8]; rest reserved.
const CTRL_BYTES: usize = 64;
/// Host-side timeout waiting for a wave to drain (the kernel's own backstop is the device-side net).
const DRAIN_TIMEOUT: Duration = Duration::from_secs(20);

// FFI signatures resolved ad-hoc from the primary context's `Library` (the crate idiom — see how
// `cuLaunchKernel`/`cuMemcpyHtoD` are resolved in `submit_cuda_resident_i32_index_probe`).
// `cuMemHostGetDevicePointer` is not bound as a `GpuPrimaryContext` field; the device-mapped ring/result/
// ctrl buffers need it to hand the kernel a device pointer.
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

/// The persistent data-plane kernel — from `wave_devatomic_probe.rs` (probe 1d-i, the kernel the R2
/// `all_done` ordering audit covered), with ONE change for multi-wave persistence: the claim uses
/// `atom.cas` (bounded) instead of `atom.add` (speculative). The probe's `atom.add` lets many threads
/// overshoot `claim` past `head` (harmless in its single wave, but it leaves `claim` ~thread-count high,
/// which would stall the next wave under a cumulative `head`). `atom.cas` increments `claim` by exactly 1
/// only while `claim < head`, so `claim` ends exactly at `head` each wave and never overshoots. This is
/// orthogonal to the audited path: the result-store -> `membar.sys` -> `completed` bump -> `all_done`
/// ordering is UNCHANGED, so the `all_done` ordering audit (gate on the counter-acquire, not `all_done`)
/// still holds.
///
/// Control block = [doorbell@0, head@4, all_done@8]; counters (device) = [claim@0, completed@4]. Each
/// thread: exit on doorbell/backstop; read `claim` < head; `atom.cas` to claim that index; hash-probe
/// `table` + gather `col_payload`; atomically store packed `(value<<32)|done`; `membar.sys`; `atom.add`
/// `completed`; the completer that reaches `head` sets `all_done` (wake hint). Lock-free, no barriers,
/// pure ASCII.
const WAVE_DATAPLANE_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_wave_read_dataplane(
    .param .u64 ctrl,
    .param .u64 counters,
    .param .u64 req,
    .param .u64 res,
    .param .u64 table,
    .param .u64 col_payload,
    .param .u32 hash_shift,
    .param .u32 table_mask,
    .param .u32 ring_mask,
    .param .u64 max_ns
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<32>;

    ld.param.u64 %rd1, [ctrl];
    ld.param.u64 %rd2, [counters];
    ld.param.u64 %rd3, [req];
    ld.param.u64 %rd4, [res];
    ld.param.u64 %rd5, [table];
    ld.param.u64 %rd6, [col_payload];
    ld.param.u32 %r1, [hash_shift];
    ld.param.u32 %r2, [table_mask];
    ld.param.u32 %r3, [ring_mask];
    ld.param.u64 %rd7, [max_ns];
    mov.u64 %rd8, %globaltimer;

$L_loop:
    ld.volatile.global.u32 %r4, [%rd1];
    setp.ne.s32 %p1, %r4, 0;
    @%p1 bra $L_done;
    mov.u64 %rd9, %globaltimer;
    sub.u64 %rd10, %rd9, %rd8;
    setp.ge.u64 %p1, %rd10, %rd7;
    @%p1 bra $L_done;
    ld.volatile.global.u32 %r5, [%rd1+4];
    ld.volatile.global.u32 %r6, [%rd2];
    setp.ge.u32 %p1, %r6, %r5;
    @%p1 bra $L_loop;
    add.u32 %r20, %r6, 1;
    atom.global.cas.b32 %r7, [%rd2], %r6, %r20;
    setp.ne.u32 %p1, %r7, %r6;
    @%p1 bra $L_loop;
    and.b32 %r8, %r7, %r3;
    mul.wide.u32 %rd11, %r8, 4;
    add.u64 %rd12, %rd3, %rd11;
    ld.volatile.global.u32 %r9, [%rd12];
    mul.lo.u32 %r10, %r9, 2654435761;
    shr.u32 %r11, %r10, %r1;
    mov.u32 %r12, 2;
    mov.u32 %r13, 0;
    mov.u32 %r17, 0;

$L_probe:
    and.b32 %r11, %r11, %r2;
    mul.wide.u32 %rd13, %r11, 8;
    add.u64 %rd14, %rd5, %rd13;
    ld.global.u64 %rd15, [%rd14];
    setp.eq.u64 %p2, %rd15, 0;
    @%p2 bra $L_write;
    shr.u64 %rd16, %rd15, 32;
    cvt.u32.u64 %r14, %rd16;
    setp.ne.s32 %p2, %r14, %r9;
    @%p2 bra $L_probe_next;
    cvt.u32.u64 %r15, %rd15;
    sub.u32 %r15, %r15, 1;
    mul.wide.u32 %rd17, %r15, 4;
    add.u64 %rd18, %rd6, %rd17;
    ld.global.u32 %r13, [%rd18];
    mov.u32 %r12, 1;
    bra $L_write;
$L_probe_next:
    add.u32 %r11, %r11, 1;
    add.u32 %r17, %r17, 1;
    setp.ge.u32 %p3, %r17, 256;
    @%p3 bra $L_write;
    bra $L_probe;

$L_write:
    cvt.u64.u32 %rd19, %r13;
    shl.b64 %rd20, %rd19, 32;
    cvt.u64.u32 %rd21, %r12;
    or.b64 %rd22, %rd20, %rd21;
    mul.wide.u32 %rd23, %r8, 8;
    add.u64 %rd24, %rd4, %rd23;
    st.volatile.global.u64 [%rd24], %rd22;
    membar.sys;
    atom.global.add.u32 %r18, [%rd2+4], 1;
    add.u32 %r18, %r18, 1;
    setp.ne.u32 %p4, %r18, %r5;
    @%p4 bra $L_loop;
    membar.sys;
    mov.u32 %r19, 1;
    st.volatile.global.u32 [%rd1+8], %r19;
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
/// circular needle ring against one resident int4 table's hash index. Owns a dedicated (non-pooled)
/// stream the kernel occupies for its whole life, device-mapped ctrl/req/res buffers, and a device
/// counters buffer. `submit` runs one wave at a time (serialized) and reads its results back.
pub(crate) struct WaveReadEngine {
    /// Keep the shared context + the index/payload device buffers alive for the engine's lifetime.
    _primary: Arc<GpuPrimaryContext>,
    _index: Arc<CudaResidentDeviceMemory>,
    _payload: Arc<CudaResidentDeviceMemory>,
    stream: *mut c_void,
    /// Device-mapped pinned control block (host view): [doorbell@0, head@4, all_done@8].
    ctrl_host: *mut c_void,
    /// Device-mapped pinned needle ring (host view): i32[ring_capacity].
    req_host: *mut c_void,
    /// Device-mapped pinned result ring (host view): u64[ring_capacity], packed (value<<32)|done.
    res_host: *mut c_void,
    /// Device counters [claim@0, completed@4] (monotonic; never reset).
    counters: u64,
    ring_capacity: usize,
    ring_mask: u32,
    /// Cumulative count of needles ever published (host-tracked; mirrors ctrl.head). Monotonic.
    head: u32,
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cu_stream_synchronize: CuStreamSynchronize,
    cu_stream_destroy: CuStreamDestroy,
    cu_mem_free: CuMemFree,
    cu_mem_free_host: CuMemFreeHost,
    shut: bool,
}

impl WaveReadEngine {
    /// Launch the persistent data-plane kernel over `index` (built with `table_mask`/`hash_shift`) and
    /// `payload` (the resident column gathered at `payload_offset + row*4`), on the SAME shared context
    /// the `index` lives in. `ring_capacity` is rounded up to a power of two and bounds a single wave's
    /// size. `threads` is the persistent grid (kept modest per the SM-coexistence rule). The kernel runs
    /// until `shutdown`/`Drop`, or `backstop_ns` elapses (the wall-clock safety net).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        index: Arc<CudaResidentDeviceMemory>,
        payload: Arc<CudaResidentDeviceMemory>,
        payload_offset: u64,
        table_mask: u32,
        hash_shift: u32,
        ring_capacity: usize,
        threads: u32,
        backstop_ns: u64,
    ) -> Result<Self, CudaRuntimeProbeError> {
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
        let cu_memcpy_dtoh: CuMemcpyDtoH =
            unsafe { sym(lib, &[b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0"]) }?;
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

        let table_ptr = index.device_ptr();
        let payload_ptr = payload.device_ptr() + payload_offset;
        let blocks = threads.div_ceil(256);
        let threads_per_block = threads.min(256);

        // Allocate every resource through a try-block; free whatever was created on any error.
        let mut counters: u64 = 0;
        let mut ctrl_host: *mut c_void = ptr::null_mut();
        let mut req_host: *mut c_void = ptr::null_mut();
        let mut res_host: *mut c_void = ptr::null_mut();
        let mut stream: *mut c_void = ptr::null_mut();
        let setup = (|| -> Result<(), CudaRuntimeProbeError> {
            // Device counters [claim, completed], zeroed.
            check_cuda(unsafe { cu_mem_alloc(&mut counters, 8) })?;
            let zero = [0u32; 2];
            check_cuda(unsafe { cu_memcpy_htod(counters, zero.as_ptr().cast::<c_void>(), 8) })?;

            // Device-mapped ctrl / req / res; ctrl + res zeroed (req is overwritten per wave).
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
                cu_mem_host_alloc(&mut res_host, ring_capacity * 8, CU_MEMHOSTALLOC_DEVICEMAP)
            })?;
            unsafe {
                for i in 0..ring_capacity {
                    ptr::write_volatile((res_host as *mut u64).add(i), 0);
                }
            }
            fence(Ordering::SeqCst);

            let mut ctrl_dptr: u64 = 0;
            let mut req_dptr: u64 = 0;
            let mut res_dptr: u64 = 0;
            check_cuda(unsafe { cu_mem_host_get_device_pointer(&mut ctrl_dptr, ctrl_host, 0) })?;
            check_cuda(unsafe { cu_mem_host_get_device_pointer(&mut req_dptr, req_host, 0) })?;
            check_cuda(unsafe { cu_mem_host_get_device_pointer(&mut res_dptr, res_host, 0) })?;

            check_cuda(unsafe { cu_stream_create(&mut stream, CU_STREAM_NON_BLOCKING) })?;

            // Launch the persistent kernel ASYNC; it idles (head=0) until the first `submit`.
            let mut a_ctrl = ctrl_dptr;
            let mut a_counters = counters;
            let mut a_req = req_dptr;
            let mut a_res = res_dptr;
            let mut a_table = table_ptr;
            let mut a_pay = payload_ptr;
            let mut a_shift = hash_shift;
            let mut a_tmask = table_mask;
            let mut a_rmask = ring_mask;
            let mut a_max = backstop_ns;
            let mut params = [
                (&mut a_ctrl as *mut u64).cast::<c_void>(),
                (&mut a_counters as *mut u64).cast::<c_void>(),
                (&mut a_req as *mut u64).cast::<c_void>(),
                (&mut a_res as *mut u64).cast::<c_void>(),
                (&mut a_table as *mut u64).cast::<c_void>(),
                (&mut a_pay as *mut u64).cast::<c_void>(),
                (&mut a_shift as *mut u32).cast::<c_void>(),
                (&mut a_tmask as *mut u32).cast::<c_void>(),
                (&mut a_rmask as *mut u32).cast::<c_void>(),
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
            _payload: payload,
            stream,
            ctrl_host,
            req_host,
            res_host,
            counters,
            ring_capacity,
            ring_mask,
            head: 0,
            cu_memcpy_dtoh,
            cu_stream_synchronize,
            cu_stream_destroy,
            cu_mem_free,
            cu_mem_free_host,
            shut: false,
        })
    }

    /// Run one wave of point lookups: enqueue `needles` (each a key value), wait for the wave to drain,
    /// and return one result per needle (`Some(payload)` if the key was found, else `None`). Serialized:
    /// each call publishes its needles, waits for `completed == head`, then reads its slots — so the
    /// circular ring never overwrites an un-read wave (one wave in flight at a time). `needles.len()` must
    /// not exceed `ring_capacity`.
    pub(crate) fn submit(&mut self, needles: &[i32]) -> Result<Vec<Option<i32>>, CudaRuntimeProbeError> {
        if needles.is_empty() {
            return Ok(Vec::new());
        }
        if needles.len() > self.ring_capacity {
            return Err(CudaRuntimeProbeError::InvalidInputLength(needles.len()));
        }
        let base = self.head;
        let req = self.req_host as *mut i32;
        let res = self.res_host as *mut u64;
        for (i, &needle) in needles.iter().enumerate() {
            let slot = (base as usize + i) & (self.ring_mask as usize);
            unsafe {
                ptr::write_volatile(req.add(slot), needle);
                ptr::write_volatile(res.add(slot), 0); // clear stale (this slot was read last wave)
            }
        }
        // Reset the all_done WAKE HINT (not the gate) and publish the new cumulative head.
        unsafe { ptr::write_volatile((self.ctrl_host as *mut u32).add(2), 0) };
        fence(Ordering::SeqCst);
        let new_head = base + needles.len() as u32;
        self.head = new_head;
        unsafe { ptr::write_volatile((self.ctrl_host as *mut u32).add(1), new_head) };
        fence(Ordering::SeqCst);

        // Wake hint: cheap host-mapped poll of all_done so we don't hammer DtoH.
        let deadline = Instant::now() + DRAIN_TIMEOUT;
        loop {
            if unsafe { ptr::read_volatile((self.ctrl_host as *const u32).add(2)) } != 0 {
                break;
            }
            if Instant::now() >= deadline {
                break; // fall through to the authoritative gate (which also has the deadline)
            }
            std::hint::spin_loop();
        }
        // AUTHORITATIVE GATE (ADR-008 "R2 all_done ordering audit"): the host ACQUIRES the device
        // `completed` counter (== cumulative head) via DtoH BEFORE reading any slot. This is the sound
        // 1b release/acquire (each worker released its slot store ahead of its bump via membar.sys);
        // all_done above is only the wake hint and carries no happens-before for the slot stores.
        let mut counters = [0u32; 2];
        loop {
            check_cuda(unsafe {
                (self.cu_memcpy_dtoh)(counters.as_mut_ptr().cast::<c_void>(), self.counters, 8)
            })?;
            if counters[1] >= new_head {
                break;
            }
            if Instant::now() >= deadline {
                return Err(CudaRuntimeProbeError::KernelLaunchFailed(-1));
            }
            std::hint::spin_loop();
        }

        let mut out = Vec::with_capacity(needles.len());
        for i in 0..needles.len() {
            let slot = (base as usize + i) & (self.ring_mask as usize);
            let packed = unsafe { ptr::read_volatile(res.add(slot)) };
            let done = (packed & 0xffff_ffff) as u32; // 1 = found, 2 = not found
            let value = (packed >> 32) as i32;
            out.push(if done == 1 { Some(value) } else { None });
        }
        Ok(out)
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
            // The kernel exits on the doorbell (~us); the %globaltimer backstop is the ultimate net.
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
    fn wave_engine_point_lookups_match_oracle_on_shared_context() {
        // Build a tiny resident int4 table + its R1-format hash index on the SHARED primary context,
        // run the persistent wave engine over it, and assert every needle's result matches a CPU oracle
        // (present key -> its payload; absent value -> None). Two waves exercise the cumulative ring.
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return; // no GPU -> skip (also #[ignore]d by default)
        };
        let rows: u64 = 1000;
        // Distinct keys (a real map) + distinct payload (a load-bearing gather).
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
        let payload_offset = rows * 4;

        // Open-addressing index over the key column (same format/hash as R1 + the probes).
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

        let mut engine = WaveReadEngine::new(
            index_resident,
            resident,
            payload_offset,
            table_mask,
            hash_shift,
            1024,
            1024,
            30_000_000_000,
        )
        .expect("wave engine launches on the shared context");

        // CPU oracle: a needle is a KEY value; found -> its row's payload, else None.
        let oracle = |needle: i32| -> Option<i32> {
            keys.iter().position(|&k| k == needle).map(|r| payload[r])
        };

        // Wave 1: present keys (rows 5, 100, 999, 0) + one absent value (2 is not 1 mod 3).
        let w1: Vec<i32> = vec![keys[5], keys[100], keys[999], keys[0], 2];
        let got1 = engine.submit(&w1).expect("wave 1");
        let want1: Vec<Option<i32>> = w1.iter().map(|&n| oracle(n)).collect();
        assert_eq!(got1, want1, "wave 1 results must match the CPU oracle");
        assert_eq!(got1[0], Some(payload[5]), "needle keys[5] -> payload[5]");
        assert_eq!(got1[4], None, "absent needle 2 -> None");

        // Wave 2 (cumulative ring): a different set, proving multi-wave correctness.
        let w2: Vec<i32> = vec![keys[10], keys[20], keys[30], 5];
        let got2 = engine.submit(&w2).expect("wave 2");
        let want2: Vec<Option<i32>> = w2.iter().map(|&n| oracle(n)).collect();
        assert_eq!(got2, want2, "wave 2 results must match the CPU oracle");

        engine.shutdown(); // clean doorbell exit on the shared context
    }
}
