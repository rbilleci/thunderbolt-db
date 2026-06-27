//! R2 wave read engine (ADR-009) — the persistent-kernel point-lookup read path, lifted into the crate
//! from the proven standalone probes (`examples/wave_devatomic_probe.rs` 1d-i, `wave_batchclaim_probe.rs`
//! 1d-ii) so it can run on the engine's REAL shared primary context instead of a throwaway one.
//!
//! **Sub-step 1 (this file, so far) = BARE LIFECYCLE on the shared context.** It proves the one genuinely
//! new thing vs the probes: a persistent kernel launched on the engine's process-wide
//! `gpu_primary_context()` (alongside the engine's live allocations/modules/streams) starts, advances a
//! heartbeat, and EXITS CLEANLY on a host-rung doorbell — without zombie-ing the shared context. NO data
//! plane yet (no ring / index probe / result slots); that is sub-step 2.
//!
//! Safety net (the `--gpu-reset`-denied-box rule, proven in probe 1a): the kernel ALWAYS self-terminates
//! — host doorbell OR a `%globaltimer` wall-clock backstop (`backstop_ns`) — so a missed doorbell can
//! never hang the shared context; worst case it runs bounded time then exits. Single block / single
//! thread (1 SM) honoring the R2 SM-coexistence rule (DECISIONS ADR-008: never a fat co-resident).
//! ASCII-only PTX (the driver JIT rejects non-ASCII even where the local ptxas tolerates it).
//!
//! When the data plane lands (sub-step 2) the completion gate MUST be the host ACQUIRING the device
//! `completed` counter (`== requests`), NOT an `all_done` flag — per the R2 `all_done` ordering audit
//! (DECISIONS ADR-008): `all_done` carries no happens-before for the other workers' result stores.

// TODO(R2.2): drop this once the engine wiring consumes `WaveReadEngine`. Until then the type is used
// only by its `#[cfg(test)]` GPU test, so a non-test build sees it as dead.
#![allow(dead_code)]

use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::sync::Arc;

use libloading::Library;

use crate::{check_cuda, gpu_primary_context, CudaRuntimeProbeError, GpuPrimaryContext};

const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
const CU_STREAM_NON_BLOCKING: u32 = 0x01;
/// Control block bytes (device-mapped pinned): [doorbell@0, heartbeat@4]; rest reserved for sub-step 2.
const CTRL_BYTES: usize = 64;

// FFI signatures resolved ad-hoc from the primary context's `Library` (the crate idiom — see how
// `cuLaunchKernel`/`cuMemcpyHtoD` are resolved in `submit_cuda_resident_i32_index_probe`). NOTE:
// `cuMemHostGetDevicePointer` is NOT bound as a `GpuPrimaryContext` field, so the wave engine resolves
// it here; the device-mapped ring/result/ctrl buffers need it to hand the kernel a device pointer.
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

/// The bare persistent lifecycle kernel (probe 1a): a single thread polls the device-mapped doorbell
/// (offset 0), publishes a heartbeat (offset 4), and exits on the doorbell OR when `%globaltimer` passes
/// `max_ns` since launch. Pure ASCII (guarded by `wave_lifecycle_ptx_is_pure_ascii`).
const WAVE_LIFECYCLE_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_wave_read_lifecycle(
    .param .u64 ctrl,
    .param .u64 max_ns
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [ctrl];
    ld.param.u64 %rd2, [max_ns];
    mov.u64 %rd3, %globaltimer;
    mov.u32 %r2, 0;

$L_loop:
    ld.volatile.global.u32 %r1, [%rd1];
    setp.ne.s32 %p1, %r1, 0;
    @%p1 bra $L_done;
    add.s32 %r2, %r2, 1;
    st.volatile.global.u32 [%rd1+4], %r2;
    mov.u64 %rd4, %globaltimer;
    sub.u64 %rd5, %rd4, %rd3;
    setp.ge.u64 %p2, %rd5, %rd2;
    @%p2 bra $L_done;
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

/// A persistent GPU "wave" kernel resident on the engine's shared primary context.
///
/// Sub-step 1: lifecycle only. Owns a dedicated (non-pooled) stream the persistent kernel occupies for
/// its whole life, a device-mapped pinned control block, and the bare driver fns needed to ring the
/// doorbell + tear down. `shutdown`/`Drop` ring the doorbell and synchronize the stream so the kernel
/// exits before the stream/buffer are freed.
pub(crate) struct WaveReadEngine {
    /// Keeps the shared primary context (and its `Library`) alive for the engine's lifetime.
    _primary: Arc<GpuPrimaryContext>,
    stream: *mut c_void,
    /// Device-mapped pinned control block (host view): [doorbell@0, heartbeat@4].
    ctrl_host: *mut c_void,
    cu_stream_synchronize: CuStreamSynchronize,
    cu_stream_destroy: CuStreamDestroy,
    cu_mem_free_host: CuMemFreeHost,
    shut: bool,
}

impl WaveReadEngine {
    /// Launch the persistent lifecycle kernel on the shared primary context for `gpu_id`. The kernel runs
    /// until `shutdown`/`Drop` rings the doorbell, or `backstop_ns` elapses (the wall-clock safety net).
    pub(crate) fn new(gpu_id: u16, backstop_ns: u64) -> Result<Self, CudaRuntimeProbeError> {
        let primary = gpu_primary_context(gpu_id)?;
        primary.set_current()?;
        let lib = primary.lib();

        // Resolve the driver fns up front (these only fail with DriverLibraryUnavailable, before any
        // resource is allocated, so an early return here cannot leak).
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

        let mut ptx = Vec::with_capacity(WAVE_LIFECYCLE_PTX.len() + 1);
        ptx.extend_from_slice(WAVE_LIFECYCLE_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_wave_read_lifecycle", &ptx)?;

        // Device-mapped pinned control block, zeroed (doorbell=0, heartbeat=0).
        let mut ctrl_host: *mut c_void = ptr::null_mut();
        check_cuda(unsafe {
            cu_mem_host_alloc(&mut ctrl_host, CTRL_BYTES, CU_MEMHOSTALLOC_DEVICEMAP)
        })?;
        unsafe {
            for i in 0..(CTRL_BYTES / 4) {
                ptr::write_volatile((ctrl_host as *mut u32).add(i), 0);
            }
        }
        fence(Ordering::SeqCst);

        // From here a failure must free `ctrl_host` (and any stream created). Emulate a try-block.
        let stream = (|| -> Result<*mut c_void, CudaRuntimeProbeError> {
            let mut ctrl_dptr: u64 = 0;
            check_cuda(unsafe { cu_mem_host_get_device_pointer(&mut ctrl_dptr, ctrl_host, 0) })?;
            let mut stream: *mut c_void = ptr::null_mut();
            check_cuda(unsafe { cu_stream_create(&mut stream, CU_STREAM_NON_BLOCKING) })?;
            // Launch ASYNC: 1 block x 1 thread (1 SM). The kernel runs until the doorbell / backstop.
            let mut a_ctrl = ctrl_dptr;
            let mut a_max = backstop_ns;
            let mut params = [
                (&mut a_ctrl as *mut u64).cast::<c_void>(),
                (&mut a_max as *mut u64).cast::<c_void>(),
            ];
            if let Err(err) = check_cuda(unsafe {
                cu_launch_kernel(
                    function,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            }) {
                unsafe { cu_stream_destroy(stream) };
                return Err(err);
            }
            Ok(stream)
        })();
        let stream = match stream {
            Ok(stream) => stream,
            Err(err) => {
                unsafe { cu_mem_free_host(ctrl_host) };
                return Err(err);
            }
        };

        Ok(Self {
            _primary: primary,
            stream,
            ctrl_host,
            cu_stream_synchronize,
            cu_stream_destroy,
            cu_mem_free_host,
            shut: false,
        })
    }

    /// The persistent kernel's heartbeat counter (advances every poll while it runs) — liveness only.
    pub(crate) fn heartbeat(&self) -> u32 {
        unsafe { ptr::read_volatile((self.ctrl_host as *const u32).add(1)) }
    }

    /// Ring the doorbell and wait for the kernel to exit, then free the stream + control block.
    /// Idempotent (safe to call before `Drop`).
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
            (self.cu_mem_free_host)(self.ctrl_host);
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
    use std::time::{Duration, Instant};

    #[test]
    fn wave_lifecycle_ptx_is_pure_ascii() {
        // The driver JIT's ptxas rejects a non-ASCII byte (INVALID_PTX 218) even where the local ptxas
        // tolerates it, so a stray em-dash / smart-quote in a comment fails every launch. Keep it ASCII.
        if let Some(pos) = WAVE_LIFECYCLE_PTX.iter().position(|&byte| !byte.is_ascii()) {
            let line = WAVE_LIFECYCLE_PTX[..pos]
                .iter()
                .filter(|&&byte| byte == b'\n')
                .count()
                + 1;
            panic!("WAVE_LIFECYCLE_PTX has a non-ASCII byte at offset {pos} (line {line})");
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_engine_lifecycle_exits_cleanly_on_shared_context() {
        // Construct over the engine's SHARED primary context; prove the persistent kernel runs (heartbeat
        // advances) and exits CLEANLY on the doorbell (shutdown returns well inside the 30s backstop) with
        // no zombie context. This is the in-crate, shared-context version of probe 1a.
        let Ok(mut engine) = WaveReadEngine::new(0, 30_000_000_000) else {
            return; // no GPU -> skip (also #[ignore]d by default)
        };
        let hb0 = engine.heartbeat();
        std::thread::sleep(Duration::from_millis(50));
        let hb1 = engine.heartbeat();
        assert!(
            hb1 > hb0,
            "heartbeat did not advance (hb0={hb0} hb1={hb1}) - persistent kernel not running"
        );
        let rang = Instant::now();
        engine.shutdown();
        let exit = rang.elapsed();
        assert!(
            exit < Duration::from_secs(2),
            "kernel did not exit promptly on the doorbell (took {exit:?}) - it likely hit the backstop"
        );
    }
}
