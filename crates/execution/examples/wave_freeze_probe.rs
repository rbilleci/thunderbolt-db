//! Wave-engine R2.2 BLOCKER probe — pin down the INTERLEAVED-LAUNCH FREEZE.
//!
//! R2.2a found that launching the R1 index-probe (a flag-0/blocking pooled stream + a synchronous
//! NULL-stream memcpy + a stream sync) BETWEEN two waves freezes the idle persistent wave kernel — its
//! claim/heartbeat stops, with no CUDA error/fault/backstop. The SM-coexistence probe only tested
//! NON-blocking concurrent launches (which were fine). This probe isolates WHICH operation freezes a
//! non-blocking persistent kernel, so R2.2 can decide: fix coexistence (non-blocking streams / avoid
//! NULL-stream sync) OR have the wave kernel REPLACE the per-batch path.
//!
//! Method: launch a heartbeat-advancing persistent kernel on a NON-BLOCKING stream (clear liveness
//! signal), confirm it's alive, run ONE candidate interleave op, then check whether the heartbeat KEEPS
//! advancing. Each op runs against a FRESH persistent kernel (independent). Also reports the shutdown
//! cuStreamSynchronize latency: fast => the kernel had EXITED; ~backstop => it was HUNG (paused).
//!
//! Safety: the persistent kernel always self-terminates (doorbell OR a `%globaltimer` backstop, 5s here);
//! single block/thread; standalone own context; ASCII PTX. Run under `timeout`, never `--gpu-reset`.
//!
//! Run: `cargo run --release -p gpu_db_execution --example wave_freeze_probe`

use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

use libloading::{Library, Symbol};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

const PTX_SRC: &str = r#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_freeze_persistent(
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

$L0:
    ld.volatile.global.u32 %r1, [%rd1];
    setp.ne.s32 %p1, %r1, 0;
    @%p1 bra $L1;
    atom.global.add.u32 %r2, [%rd1+4], 1;
    mov.u64 %rd4, %globaltimer;
    sub.u64 %rd5, %rd4, %rd3;
    setp.ge.u64 %p2, %rd5, %rd2;
    @%p2 bra $L1;
    bra $L0;
$L1:
    ret;
}

.visible .entry gpu_db_freeze_noop(
    .param .u64 out
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [out];
    mov.u32 %r1, 12345;
    st.global.u32 [%rd1], %r1;
    ret;
}
"#;

const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
const CU_STREAM_NON_BLOCKING: u32 = 0x01;

type Fn1U32 = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuCtxRetain = unsafe extern "C" fn(*mut *mut c_void, i32) -> i32;
type CuCtxRelease = unsafe extern "C" fn(i32) -> i32;
type Fn1Ptr = unsafe extern "C" fn(*mut c_void) -> i32;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFree = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
type CuMemHostGetDevicePointer = unsafe extern "C" fn(*mut u64, *mut c_void, u32) -> i32;
type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> i32;
type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
type CuModuleGetFunction = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
type CuStreamCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
type CuStreamDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type CuEventCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
type CuEventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type CuEventDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
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
            return unsafe { std::mem::transmute::<Symbol<'_, T>, Symbol<'static, T>>(symbol) };
        }
    }
    panic!("CUDA symbol not found: {names:?}");
}

fn check(code: i32, what: &str) {
    assert_eq!(code, 0, "{what} failed: CUDA driver error {code}");
}

fn main() {
    let backstop_ns: u64 = 5_000_000_000; // 5s — bounds a HUNG kernel's shutdown wait

    let lib: &'static Library = Box::leak(Box::new(
        unsafe { Library::new("libcuda.so.1") }
            .or_else(|_| unsafe { Library::new("libcuda.so") })
            .expect("load libcuda"),
    ));
    let cu_init: Symbol<Fn1U32> = sym(lib, &[b"cuInit\0"]);
    let cu_device_get: Symbol<CuDeviceGet> = sym(lib, &[b"cuDeviceGet\0"]);
    let cu_ctx_retain: Symbol<CuCtxRetain> = sym(lib, &[b"cuDevicePrimaryCtxRetain\0"]);
    let cu_ctx_release: Symbol<CuCtxRelease> = sym(
        lib,
        &[
            b"cuDevicePrimaryCtxRelease_v2\0",
            b"cuDevicePrimaryCtxRelease\0",
        ],
    );
    let cu_ctx_set_current: Symbol<Fn1Ptr> = sym(lib, &[b"cuCtxSetCurrent\0"]);
    let cu_mem_alloc: Symbol<CuMemAlloc> = sym(lib, &[b"cuMemAlloc_v2\0", b"cuMemAlloc\0"]);
    let cu_mem_free: Symbol<CuMemFree> = sym(lib, &[b"cuMemFree_v2\0", b"cuMemFree\0"]);
    let cu_memcpy_htod: Symbol<CuMemcpyHtoD> = sym(lib, &[b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0"]);
    let cu_memcpy_dtoh: Symbol<CuMemcpyDtoH> = sym(lib, &[b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0"]);
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
    let cu_stream_create: Symbol<CuStreamCreate> = sym(lib, &[b"cuStreamCreate\0"]);
    let cu_stream_synchronize: Symbol<Fn1Ptr> = sym(lib, &[b"cuStreamSynchronize\0"]);
    let cu_stream_destroy: Symbol<CuStreamDestroy> =
        sym(lib, &[b"cuStreamDestroy_v2\0", b"cuStreamDestroy\0"]);
    let cu_event_create: Symbol<CuEventCreate> = sym(lib, &[b"cuEventCreate\0"]);
    let cu_event_record: Symbol<CuEventRecord> = sym(lib, &[b"cuEventRecord\0"]);
    let cu_event_destroy: Symbol<CuEventDestroy> =
        sym(lib, &[b"cuEventDestroy_v2\0", b"cuEventDestroy\0"]);
    let cu_launch_kernel: Symbol<CuLaunchKernel> = sym(lib, &[b"cuLaunchKernel\0"]);

    // Persistent kernel grid (data-plane uses 4 x 256); env-overridable to sweep the freeze.
    let pblocks = env_u32("GPU_DB_FREEZE_PBLOCKS", 4);
    let ptpb = env_u32("GPU_DB_FREEZE_PTPB", 256);

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
    let mut module: *mut c_void = ptr::null_mut();
    check(
        unsafe { cu_module_load_data(&mut module, ptx.as_ptr() as *const c_void) },
        "cuModuleLoadData",
    );
    let entry_persist = CString::new("gpu_db_freeze_persistent").unwrap();
    let entry_noop = CString::new("gpu_db_freeze_noop").unwrap();
    let mut func_persist: *mut c_void = ptr::null_mut();
    let mut func_noop: *mut c_void = ptr::null_mut();
    check(
        unsafe { cu_module_get_function(&mut func_persist, module, entry_persist.as_ptr()) },
        "get persist",
    );
    check(
        unsafe { cu_module_get_function(&mut func_noop, module, entry_noop.as_ptr()) },
        "get noop",
    );

    // Device-mapped pinned control [doorbell@0, heartbeat@4].
    let mut ctrl_host: *mut c_void = ptr::null_mut();
    check(
        unsafe { cu_mem_host_alloc(&mut ctrl_host, 64, CU_MEMHOSTALLOC_DEVICEMAP) },
        "alloc ctrl",
    );
    let mut ctrl_dptr: u64 = 0;
    check(
        unsafe { cu_mem_host_get_device_pointer(&mut ctrl_dptr, ctrl_host, 0) },
        "get ctrl dptr",
    );
    let ctrl = ctrl_host as *mut u32;

    // A scratch device buffer + host buffer for the interleave ops.
    let mut d_scratch: u64 = 0;
    check(unsafe { cu_mem_alloc(&mut d_scratch, 64) }, "alloc scratch");
    let mut h_scratch = [0u8; 64];

    // Run one interleave op against a FRESH persistent kernel and report ALIVE/FROZEN + shutdown latency.
    let run = |label: &str, interleave: &mut dyn FnMut()| {
        unsafe {
            ptr::write_volatile(ctrl, 0); // doorbell
            ptr::write_volatile(ctrl.add(1), 0); // heartbeat
        }
        fence(Ordering::SeqCst);
        let mut pstream: *mut c_void = ptr::null_mut();
        check(
            unsafe { cu_stream_create(&mut pstream, CU_STREAM_NON_BLOCKING) },
            "create pstream",
        );
        let mut a_ctrl = ctrl_dptr;
        let mut a_max = backstop_ns;
        let mut params: [*mut c_void; 2] = [
            ptr::addr_of_mut!(a_ctrl) as *mut c_void,
            ptr::addr_of_mut!(a_max) as *mut c_void,
        ];
        check(
            unsafe {
                cu_launch_kernel(
                    func_persist,
                    pblocks,
                    1,
                    1,
                    ptpb,
                    1,
                    1,
                    0,
                    pstream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launch persist",
        );
        // Confirm alive at baseline.
        std::thread::sleep(Duration::from_millis(40));
        let hb_base = unsafe { ptr::read_volatile(ctrl.add(1)) };
        assert!(hb_base > 0, "{label}: persistent kernel never started");

        // The interleave op (timed: a device-synchronizing op will block ~backstop waiting for the
        // never-ending persistent kernel, which is the smoking gun).
        let op_started = Instant::now();
        interleave();
        let op_dur = op_started.elapsed();

        // Did it keep advancing?
        let hb_before = unsafe { ptr::read_volatile(ctrl.add(1)) };
        std::thread::sleep(Duration::from_millis(40));
        let hb_after = unsafe { ptr::read_volatile(ctrl.add(1)) };
        let delta = hb_after.wrapping_sub(hb_before);
        let alive = delta > 0;

        // Shutdown: ring the doorbell, time the sync (fast => exited; ~backstop => hung).
        unsafe { ptr::write_volatile(ctrl, 1) };
        fence(Ordering::SeqCst);
        let rang = Instant::now();
        check(unsafe { cu_stream_synchronize(pstream) }, "sync pstream");
        let shutdown = rang.elapsed();
        check(unsafe { cu_stream_destroy(pstream) }, "destroy pstream");

        println!(
            "  {label:<34} {}  (hb +{delta:>8})  op={op_dur:>10.2?}  shutdown={:?}{}",
            if alive { "ALIVE " } else { "FROZEN" },
            shutdown,
            if !alive && op_dur >= Duration::from_secs(2) {
                "  <- the op DEVICE-SYNCED: blocked ~backstop until the kernel died"
            } else if !alive {
                "  <- kernel had EXITED"
            } else {
                ""
            },
        );
    };

    println!(
        "# wave R2.2 freeze probe  (persistent kernel {pblocks}x{ptpb} on a NON-BLOCKING stream; backstop={}s)",
        backstop_ns / 1_000_000_000
    );
    println!("# each row = one interleave op vs a fresh persistent kernel; FROZEN = the op stalled it\n");

    // 0) Control: no interleave -> must be ALIVE.
    run("none (control)", &mut || {});

    // 1) Non-blocking kernel launch + sync (the SM-coexistence-probe case -> expected ALIVE).
    run("nonblocking launch + sync", &mut || {
        let mut s: *mut c_void = ptr::null_mut();
        check(
            unsafe { cu_stream_create(&mut s, CU_STREAM_NON_BLOCKING) },
            "nb stream",
        );
        let mut out = d_scratch;
        let mut p: [*mut c_void; 1] = [ptr::addr_of_mut!(out) as *mut c_void];
        check(
            unsafe { cu_launch_kernel(func_noop, 64, 1, 1, 128, 1, 1, 0, s, p.as_mut_ptr(), ptr::null_mut()) },
            "nb launch",
        );
        check(unsafe { cu_stream_synchronize(s) }, "nb sync");
        check(unsafe { cu_stream_destroy(s) }, "nb destroy");
    });

    // 2) Blocking (flag-0) kernel launch + sync (the engine's pooled-stream type).
    run("blocking(flag0) launch + sync", &mut || {
        let mut s: *mut c_void = ptr::null_mut();
        check(unsafe { cu_stream_create(&mut s, 0) }, "blk stream");
        let mut out = d_scratch;
        let mut p: [*mut c_void; 1] = [ptr::addr_of_mut!(out) as *mut c_void];
        check(
            unsafe { cu_launch_kernel(func_noop, 64, 1, 1, 128, 1, 1, 0, s, p.as_mut_ptr(), ptr::null_mut()) },
            "blk launch",
        );
        check(unsafe { cu_stream_synchronize(s) }, "blk sync");
        check(unsafe { cu_stream_destroy(s) }, "blk destroy");
    });

    // 3) Synchronous NULL-stream HtoD memcpy (no kernel).
    run("null-stream HtoD memcpy", &mut || {
        check(
            unsafe { cu_memcpy_htod(d_scratch, h_scratch.as_ptr() as *const c_void, 64) },
            "null htod",
        );
    });

    // 4) Synchronous NULL-stream DtoH memcpy (no kernel).
    run("null-stream DtoH memcpy", &mut || {
        check(
            unsafe { cu_memcpy_dtoh(h_scratch.as_mut_ptr() as *mut c_void, d_scratch, 64) },
            "null dtoh",
        );
    });

    // 5) The index-probe combo: blocking launch + NULL HtoD + sync.
    run("blocking launch + null HtoD + sync", &mut || {
        let mut s: *mut c_void = ptr::null_mut();
        check(unsafe { cu_stream_create(&mut s, 0) }, "combo stream");
        check(
            unsafe { cu_memcpy_htod(d_scratch, h_scratch.as_ptr() as *const c_void, 64) },
            "combo htod",
        );
        let mut out = d_scratch;
        let mut p: [*mut c_void; 1] = [ptr::addr_of_mut!(out) as *mut c_void];
        check(
            unsafe { cu_launch_kernel(func_noop, 64, 1, 1, 128, 1, 1, 0, s, p.as_mut_ptr(), ptr::null_mut()) },
            "combo launch",
        );
        check(unsafe { cu_stream_synchronize(s) }, "combo sync");
        check(unsafe { cu_stream_destroy(s) }, "combo destroy");
    });

    // 6) cuMemAlloc + cuMemFree (the index probe leases device buffers; alloc/free are synchronizing ops).
    run("cuMemAlloc + cuMemFree", &mut || {
        let mut p: u64 = 0;
        check(unsafe { cu_mem_alloc(&mut p, 4096) }, "interleave alloc");
        check(unsafe { cu_mem_free(p) }, "interleave free");
    });

    // 7) CUDA event record + sync path (the pooled-stream timing the index probe uses).
    run("event create/record/destroy", &mut || {
        let mut s: *mut c_void = ptr::null_mut();
        check(unsafe { cu_stream_create(&mut s, 0) }, "evt stream");
        let mut ev: *mut c_void = ptr::null_mut();
        check(unsafe { cu_event_create(&mut ev, 0) }, "evt create");
        check(unsafe { cu_event_record(ev, s) }, "evt record");
        check(unsafe { cu_stream_synchronize(s) }, "evt sync");
        check(unsafe { cu_event_destroy(ev) }, "evt destroy");
        check(unsafe { cu_stream_destroy(s) }, "evt stream destroy");
    });

    unsafe {
        cu_mem_free(d_scratch);
        cu_mem_free_host(ctrl_host);
    }
    check(unsafe { cu_ctx_release(device) }, "ctx release");
    println!("\n# the FROZEN row(s) name the operation that stalls a non-blocking persistent kernel.");
}
