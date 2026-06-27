//! Wave-engine increment 1a — bare persistent-kernel LIFECYCLE probe (de-risking).
//!
//! Proves a persistent CUDA kernel can poll a device-mapped doorbell, advance a heartbeat, and EXIT
//! CLEANLY on the doorbell — and on a hard wall-clock backstop — 3× in a row, without zombie-ing the
//! primary context on this `--gpu-reset`-denied shared box. NO data plane: this is purely the
//! lifecycle/exit mechanism the wave engine (ADR-009) depends on. If a kernel can't exit cleanly here,
//! the wave engine stops at this gate.
//!
//! Safety: the kernel ALWAYS terminates — doorbell OR a `%globaltimer` wall-clock backstop (default
//! 30s) — so a missed/incoherent doorbell can never hang the context; worst case it runs bounded time
//! then exits (a TIME bound is robust to unknown per-iteration PCIe latency, unlike an iteration count).
//! Single block, single thread (1 SM), lock-free, ASCII PTX. Standalone (its own libcuda + primary
//! context) so a bug here cannot touch the production engine.
//!
//! Run: `cargo run --release -p gpu_db_execution --example wave_lifecycle_probe`

use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

use libloading::{Library, Symbol};

/// The persistent lifecycle kernel: a single thread that polls the doorbell (offset 0), publishes a
/// heartbeat (offset 4), and exits on the doorbell OR when `%globaltimer` passes `max_ns` since launch.
const PTX_SRC: &str = r#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_wave_lifecycle(
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

#[repr(C, align(8))]
struct WaveControl {
    doorbell: u32,
    heartbeat: u32,
}

const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;

type CuInit = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuCtxRetain = unsafe extern "C" fn(*mut *mut c_void, i32) -> i32;
type CuCtxRelease = unsafe extern "C" fn(i32) -> i32;
type CuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> i32;
type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
type CuMemHostGetDevicePointer = unsafe extern "C" fn(*mut u64, *mut c_void, u32) -> i32;
type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> i32;
type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
type CuModuleGetFunction = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
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

fn sym<T>(lib: &Library, names: &[&[u8]]) -> Symbol<'static, T> {
    for name in names {
        if let Ok(symbol) = unsafe { lib.get::<T>(name) } {
            // The Library outlives the whole probe (leaked below), so extend the borrow to 'static.
            return unsafe { std::mem::transmute::<Symbol<'_, T>, Symbol<'static, T>>(symbol) };
        }
    }
    panic!("CUDA symbol not found: {names:?}");
}

fn check(code: i32, what: &str) {
    assert_eq!(code, 0, "{what} failed: CUDA driver error {code}");
}

fn main() {
    let max_ns: u64 = std::env::var("GPU_DB_WAVE_BACKSTOP_NS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30_000_000_000); // 30s wall-clock backstop

    let lib: &'static Library = Box::leak(Box::new(
        unsafe { Library::new("libcuda.so.1") }
            .or_else(|_| unsafe { Library::new("libcuda.so") })
            .expect("load libcuda (no NVIDIA driver?)"),
    ));

    let cu_init: Symbol<CuInit> = sym(lib, &[b"cuInit\0"]);
    let cu_device_get: Symbol<CuDeviceGet> = sym(lib, &[b"cuDeviceGet\0"]);
    let cu_ctx_retain: Symbol<CuCtxRetain> = sym(lib, &[b"cuDevicePrimaryCtxRetain\0"]);
    let cu_ctx_release: Symbol<CuCtxRelease> = sym(
        lib,
        &[
            b"cuDevicePrimaryCtxRelease_v2\0",
            b"cuDevicePrimaryCtxRelease\0",
        ],
    );
    let cu_ctx_set_current: Symbol<CuCtxSetCurrent> = sym(lib, &[b"cuCtxSetCurrent\0"]);
    let cu_mem_host_alloc: Symbol<CuMemHostAlloc> = sym(lib, &[b"cuMemHostAlloc\0"]);
    let cu_mem_host_get_device_pointer: Symbol<CuMemHostGetDevicePointer> = sym(
        lib,
        &[
            b"cuMemHostGetDevicePointer_v2\0",
            b"cuMemHostGetDevicePointer\0",
        ],
    );
    let cu_mem_free_host: Symbol<CuMemFreeHost> = sym(lib, &[b"cuMemFreeHost\0"]);
    let cu_module_load_data: Symbol<CuModuleLoadData> = sym(lib, &[b"cuModuleLoadData\0"]);
    let cu_module_get_function: Symbol<CuModuleGetFunction> = sym(lib, &[b"cuModuleGetFunction\0"]);
    let cu_module_unload: Symbol<CuModuleUnload> = sym(lib, &[b"cuModuleUnload\0"]);
    let cu_stream_create: Symbol<CuStreamCreate> = sym(lib, &[b"cuStreamCreate\0"]);
    let cu_stream_synchronize: Symbol<CuStreamSynchronize> = sym(lib, &[b"cuStreamSynchronize\0"]);
    let cu_stream_destroy: Symbol<CuStreamDestroy> =
        sym(lib, &[b"cuStreamDestroy_v2\0", b"cuStreamDestroy\0"]);
    let cu_launch_kernel: Symbol<CuLaunchKernel> = sym(lib, &[b"cuLaunchKernel\0"]);

    check(unsafe { cu_init(0) }, "cuInit");
    let mut device: i32 = 0;
    check(unsafe { cu_device_get(&mut device, 0) }, "cuDeviceGet");
    let mut context: *mut c_void = ptr::null_mut();
    check(
        unsafe { cu_ctx_retain(&mut context, device) },
        "cuDevicePrimaryCtxRetain",
    );
    check(unsafe { cu_ctx_set_current(context) }, "cuCtxSetCurrent");

    let ptx = CString::new(PTX_SRC).unwrap();
    let entry = CString::new("gpu_db_wave_lifecycle").unwrap();

    println!(
        "# wave-engine 1a lifecycle probe  backstop={}ms  (3 cycles)",
        max_ns / 1_000_000
    );
    for cycle in 1..=3 {
        // Device-mapped pinned control block (doorbell + heartbeat), zeroed.
        let mut host_ptr: *mut c_void = ptr::null_mut();
        check(
            unsafe { cu_mem_host_alloc(&mut host_ptr, 64, CU_MEMHOSTALLOC_DEVICEMAP) },
            "cuMemHostAlloc",
        );
        let ctrl = host_ptr as *mut WaveControl;
        unsafe {
            ptr::write_volatile(ptr::addr_of_mut!((*ctrl).doorbell), 0);
            ptr::write_volatile(ptr::addr_of_mut!((*ctrl).heartbeat), 0);
        }
        fence(Ordering::SeqCst);
        let mut device_ptr: u64 = 0;
        check(
            unsafe { cu_mem_host_get_device_pointer(&mut device_ptr, host_ptr, 0) },
            "cuMemHostGetDevicePointer",
        );

        let mut module: *mut c_void = ptr::null_mut();
        check(
            unsafe { cu_module_load_data(&mut module, ptx.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        );
        let mut func: *mut c_void = ptr::null_mut();
        check(
            unsafe { cu_module_get_function(&mut func, module, entry.as_ptr()) },
            "cuModuleGetFunction",
        );
        let mut stream: *mut c_void = ptr::null_mut();
        check(
            unsafe { cu_stream_create(&mut stream, 0) },
            "cuStreamCreate",
        );

        // Launch the persistent kernel ASYNC (it runs until the doorbell or the wall-clock backstop).
        let mut arg_ptr = device_ptr;
        let mut arg_max = max_ns;
        let mut params: [*mut c_void; 2] = [
            ptr::addr_of_mut!(arg_ptr) as *mut c_void,
            ptr::addr_of_mut!(arg_max) as *mut c_void,
        ];
        check(
            unsafe {
                cu_launch_kernel(
                    func,
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
            },
            "cuLaunchKernel",
        );

        // Confirm the kernel is alive: the heartbeat must advance.
        let hb0 = unsafe { ptr::read_volatile(ptr::addr_of!((*ctrl).heartbeat)) };
        std::thread::sleep(Duration::from_millis(50));
        let hb1 = unsafe { ptr::read_volatile(ptr::addr_of!((*ctrl).heartbeat)) };
        assert!(
            hb1 > hb0,
            "cycle {cycle}: heartbeat did not advance (hb0={hb0} hb1={hb1}) — kernel not running"
        );

        // Ring the doorbell and wait for the kernel to exit (the backstop is the ultimate safety net).
        unsafe { ptr::write_volatile(ptr::addr_of_mut!((*ctrl).doorbell), 1) };
        fence(Ordering::SeqCst);
        let rang = Instant::now();
        check(
            unsafe { cu_stream_synchronize(stream) },
            "cuStreamSynchronize",
        );
        let exit_latency = rang.elapsed();
        let hb_exit = unsafe { ptr::read_volatile(ptr::addr_of!((*ctrl).heartbeat)) };

        assert!(
            exit_latency < Duration::from_secs(2),
            "cycle {cycle}: kernel did not exit promptly on the doorbell (took {exit_latency:?}) — \
             it likely ran to the wall-clock backstop, i.e. the doorbell was not observed"
        );
        println!(
            "  cycle {cycle}: CLEAN EXIT on doorbell — heartbeat_at_exit={hb_exit}, exit_latency={exit_latency:?}"
        );

        check(unsafe { cu_stream_destroy(stream) }, "cuStreamDestroy");
        check(unsafe { cu_module_unload(module) }, "cuModuleUnload");
        check(unsafe { cu_mem_free_host(host_ptr) }, "cuMemFreeHost");
    }

    check(
        unsafe { cu_ctx_release(device) },
        "cuDevicePrimaryCtxRelease",
    );
    println!("OK: 3/3 clean persistent-kernel lifecycles — doorbell exit confirmed, context reusable, no zombie");
}
