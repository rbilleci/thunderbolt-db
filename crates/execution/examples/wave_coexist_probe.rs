//! Wave-engine R2 GATE — persistent-kernel ⟷ engine-kernel SM-COEXISTENCE measurement.
//!
//! R2 wants ONE always-resident persistent wave kernel sitting on the SAME CUDA context as the engine's
//! existing launch-per-batch kernels. The recon's #1 unknown: does a never-exiting persistent kernel
//! (which holds its SM(s) forever) **coexist** with concurrent normal kernel launches, or does it
//! deadlock / starve them? This measures it directly on THIS box before any integration is attempted.
//!
//! Two kernels in one module on one primary context:
//!   - `gpu_db_wave_persistent` — the proven 1a persistent kernel (polls a device-mapped doorbell, bumps
//!     a heartbeat, exits on the doorbell OR a `%globaltimer` wall-clock backstop). Launched with
//!     `pblocks` blocks × 1 thread ⇒ it reserves ~`pblocks` SMs for as long as it runs.
//!   - `gpu_db_engine_scan` — a representative full-grid int4 scan (grid-stride over a resident column,
//!     count == needle): exactly the memory-bound shape the engine launches per point-read batch.
//!
//! Protocol: sweep `pblocks ∈ {0,1,8,32}`. For each, run a storm of `launches` scan kernels on a second
//! stream and measure scan throughput. `pblocks=0` is the baseline (no persistent kernel); every other
//! row is reported as a % of it ⇒ **the throughput cost of reserving N SMs for an always-resident wave
//! kernel**. For `pblocks>0` we also (a) confirm the persistent kernel is alive before the storm, (b)
//! confirm its heartbeat keeps advancing DURING the storm (⇒ the scan storm did not starve it), and
//! (c) confirm it exits promptly on the doorbell afterward (not via the backstop).
//!
//! Verdict: COEXISTENCE VIABLE if every `pblocks>0` storm completes at ~baseline throughput (no stall),
//! the persistent heartbeat advances throughout, and the kernel exits cleanly. A deadlock/starvation
//! shows up as a storm that only finishes when the backstop fires (~30s) — detectable, never a true hang.
//!
//! Safety: the persistent kernel ALWAYS self-terminates (doorbell + `%globaltimer` backstop); single
//! thread per block, lock-free; standalone (own libcuda + primary context) so a bug here can't touch the
//! engine. `pblocks` stays well under SM capacity so scan blocks can always be scheduled.
//!
//! Run: `cargo run --release -p gpu_db_execution --example wave_coexist_probe`
//! Env: GPU_DB_WAVE_COEXIST_BLOCKS (sweep, default "0,1,8,32"), GPU_DB_WAVE_ROWS (default 33554432),
//!      GPU_DB_WAVE_LAUNCHES (scan launches per storm, default 2000), GPU_DB_WAVE_POLL_NS (gentle-mode
//!      %globaltimer backoff per poll, default 1000), GPU_DB_WAVE_BACKSTOP_NS (default 30s).

use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

use libloading::{Library, Symbol};

/// Both kernels share one module. The persistent kernel is byte-for-byte 1a's lifecycle loop (1 thread
/// per block, multi-block-safe heartbeat via `atom.add`). The scan is a grid-stride equality count.
const PTX_SRC: &str = r#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_wave_persistent(
    .param .u64 ctrl,
    .param .u64 max_ns,
    .param .u32 sleep_ns
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [ctrl];
    ld.param.u64 %rd2, [max_ns];
    ld.param.u32 %r1, [sleep_ns];
    cvt.u64.u32 %rd6, %r1;        // sleep_ns as u64
    mov.u64 %rd3, %globaltimer;   // launch time (backstop reference)

$L0:
    ld.volatile.global.u32 %r2, [%rd1];
    setp.ne.s32 %p1, %r2, 0;
    @%p1 bra $L1;
    atom.global.add.u32 %r2, [%rd1+4], 1;
    mov.u64 %rd4, %globaltimer;
    sub.u64 %rd5, %rd4, %rd3;
    setp.ge.u64 %p2, %rd5, %rd2;
    @%p2 bra $L1;
    setp.eq.u64 %p3, %rd6, 0;
    @%p3 bra $L0;                 // sleep_ns==0 -> busy-spin (poll memory every iteration)
    mov.u64 %rd7, %globaltimer;   // else back off ~sleep_ns spinning ONLY on %globaltimer (no mem traffic)
$LW:
    mov.u64 %rd8, %globaltimer;
    sub.u64 %rd9, %rd8, %rd7;
    setp.lt.u64 %p4, %rd9, %rd6;
    @%p4 bra $LW;
    bra $L0;
$L1:
    ret;
}

.visible .entry gpu_db_engine_scan(
    .param .u64 col,
    .param .u32 rows,
    .param .u32 needle,
    .param .u64 result
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [col];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [needle];
    ld.param.u64 %rd2, [result];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;   // i = blockIdx.x*blockDim.x + threadIdx.x
    mov.u32 %r7, %nctaid.x;
    mul.lo.s32 %r8, %r7, %r4;        // stride = gridDim.x*blockDim.x

$S0:
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra $S1;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u32 %r9, [%rd4];
    setp.ne.s32 %p2, %r9, %r2;
    @%p2 bra $S2;
    atom.global.add.u32 %r10, [%rd2], 1;
$S2:
    add.u32 %r6, %r6, %r8;
    bra $S0;
$S1:
    ret;
}
"#;

const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
const CU_STREAM_NON_BLOCKING: u32 = 0x01;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;

type Fn1U32 = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
type CuCtxRetain = unsafe extern "C" fn(*mut *mut c_void, i32) -> i32;
type CuCtxRelease = unsafe extern "C" fn(i32) -> i32;
type Fn1Ptr = unsafe extern "C" fn(*mut c_void) -> i32;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFree = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
type CuMemHostGetDevicePointer = unsafe extern "C" fn(*mut u64, *mut c_void, u32) -> i32;
type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> i32;
type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
#[allow(clippy::type_complexity)]
type CuModuleLoadDataEx = unsafe extern "C" fn(
    *mut *mut c_void,
    *const c_void,
    u32,
    *mut u32,
    *mut *mut c_void,
) -> i32;
type CuModuleGetFunction = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
type CuStreamCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
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
            return unsafe { std::mem::transmute::<Symbol<'_, T>, Symbol<'static, T>>(symbol) };
        }
    }
    panic!("CUDA symbol not found: {names:?}");
}

fn check(code: i32, what: &str) {
    assert_eq!(code, 0, "{what} failed: CUDA driver error {code}");
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let max_ns: u64 = std::env::var("GPU_DB_WAVE_BACKSTOP_NS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30_000_000_000);
    let rows = env_u32("GPU_DB_WAVE_ROWS", 33_554_432).max(1024);
    let launches = env_u32("GPU_DB_WAVE_LAUNCHES", 2000).max(1);
    let sweep: Vec<u32> = std::env::var("GPU_DB_WAVE_COEXIST_BLOCKS")
        .unwrap_or_else(|_| "0,1,8,32".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse::<u32>().ok())
        .collect();
    // Gentle-poll backoff: re-run the sweep with the persistent kernel spinning on %globaltimer for
    // ~poll_ns between memory polls, to separate busy-wait memory-contention cost from SM-occupancy cost.
    let poll_ns = env_u32("GPU_DB_WAVE_POLL_NS", 1000);
    let scan_blocks: u32 = 4096; // full-grid; grid-stride covers any `rows`
    let scan_threads: u32 = 256;
    let needle = rows - 1; // present exactly once (last row) ⇒ count must be 1, non-vacuous

    let lib: &'static Library = Box::leak(Box::new(
        unsafe { Library::new("libcuda.so.1") }
            .or_else(|_| unsafe { Library::new("libcuda.so") })
            .expect("load libcuda (no NVIDIA driver?)"),
    ));
    let cu_init: Symbol<Fn1U32> = sym(lib, &[b"cuInit\0"]);
    let cu_device_get: Symbol<CuDeviceGet> = sym(lib, &[b"cuDeviceGet\0"]);
    let cu_device_get_attribute: Symbol<CuDeviceGetAttribute> =
        sym(lib, &[b"cuDeviceGetAttribute\0"]);
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
    let cu_mem_host_alloc: Symbol<CuMemHostAlloc> = sym(lib, &[b"cuMemHostAlloc\0"]);
    let cu_mem_host_get_device_pointer: Symbol<CuMemHostGetDevicePointer> = sym(
        lib,
        &[
            b"cuMemHostGetDevicePointer_v2\0",
            b"cuMemHostGetDevicePointer\0",
        ],
    );
    let cu_mem_free_host: Symbol<CuMemFreeHost> = sym(lib, &[b"cuMemFreeHost\0"]);
    let _cu_module_load_data: Symbol<CuModuleLoadData> = sym(lib, &[b"cuModuleLoadData\0"]);
    let cu_module_load_data_ex: Symbol<CuModuleLoadDataEx> = sym(lib, &[b"cuModuleLoadDataEx\0"]);
    let cu_module_get_function: Symbol<CuModuleGetFunction> = sym(lib, &[b"cuModuleGetFunction\0"]);
    let cu_module_unload: Symbol<CuModuleUnload> = sym(lib, &[b"cuModuleUnload\0"]);
    let cu_stream_create: Symbol<CuStreamCreate> = sym(lib, &[b"cuStreamCreate\0"]);
    let cu_stream_synchronize: Symbol<Fn1Ptr> = sym(lib, &[b"cuStreamSynchronize\0"]);
    let cu_stream_destroy: Symbol<CuStreamDestroy> =
        sym(lib, &[b"cuStreamDestroy_v2\0", b"cuStreamDestroy\0"]);
    let cu_launch_kernel: Symbol<CuLaunchKernel> = sym(lib, &[b"cuLaunchKernel\0"]);

    check(unsafe { cu_init(0) }, "cuInit");
    let mut device: i32 = 0;
    check(unsafe { cu_device_get(&mut device, 0) }, "cuDeviceGet");
    let mut sm_count: i32 = 0;
    check(
        unsafe {
            cu_device_get_attribute(
                &mut sm_count,
                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device,
            )
        },
        "cuDeviceGetAttribute(SM_COUNT)",
    );
    let mut context: *mut c_void = ptr::null_mut();
    check(
        unsafe { cu_ctx_retain(&mut context, device) },
        "cuDevicePrimaryCtxRetain",
    );
    check(unsafe { cu_ctx_set_current(context) }, "cuCtxSetCurrent");

    // Resident int4 column: col[i] = i. Device memory (what the engine scans).
    let col_host: Vec<i32> = (0..rows as i32).collect();
    let mut col: u64 = 0;
    check(
        unsafe { cu_mem_alloc(&mut col, rows as usize * 4) },
        "cuMemAlloc col",
    );
    check(
        unsafe { cu_memcpy_htod(col, col_host.as_ptr() as *const c_void, rows as usize * 4) },
        "HtoD col",
    );

    // Device-mapped pinned control [doorbell@0, heartbeat@4] and scan result counter.
    let mut ctrl_host: *mut c_void = ptr::null_mut();
    check(
        unsafe { cu_mem_host_alloc(&mut ctrl_host, 64, CU_MEMHOSTALLOC_DEVICEMAP) },
        "cuMemHostAlloc ctrl",
    );
    let mut ctrl_dptr: u64 = 0;
    check(
        unsafe { cu_mem_host_get_device_pointer(&mut ctrl_dptr, ctrl_host, 0) },
        "cuMemHostGetDevicePointer ctrl",
    );
    let mut res_host: *mut c_void = ptr::null_mut();
    check(
        unsafe { cu_mem_host_alloc(&mut res_host, 64, CU_MEMHOSTALLOC_DEVICEMAP) },
        "cuMemHostAlloc res",
    );
    let mut res_dptr: u64 = 0;
    check(
        unsafe { cu_mem_host_get_device_pointer(&mut res_dptr, res_host, 0) },
        "cuMemHostGetDevicePointer res",
    );
    let ctrl = ctrl_host as *mut u32; // [doorbell, heartbeat]
    let result = res_host as *mut u32;

    let ptx = CString::new(PTX_SRC).unwrap();
    let mut module: *mut c_void = ptr::null_mut();
    // Load via ...Ex so we can capture the DRIVER JIT's error log (the system ptxas may be newer than
    // the driver and accept PTX the driver rejects — without the log, a 218 is opaque).
    const CU_JIT_ERROR_LOG_BUFFER: u32 = 5;
    const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: u32 = 6;
    let mut err_log = vec![0u8; 8192];
    let mut opts = [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
    let mut opt_vals: [*mut c_void; 2] = [
        err_log.as_mut_ptr() as *mut c_void,
        err_log.len() as *mut c_void,
    ];
    let rc = unsafe {
        cu_module_load_data_ex(
            &mut module,
            ptx.as_ptr() as *const c_void,
            opts.len() as u32,
            opts.as_mut_ptr(),
            opt_vals.as_mut_ptr(),
        )
    };
    if rc != 0 {
        let msg = String::from_utf8_lossy(&err_log);
        panic!(
            "cuModuleLoadDataEx failed: CUDA driver error {rc}\n--- driver JIT log ---\n{}",
            msg.trim_end_matches('\0').trim()
        );
    }
    let entry_persist = CString::new("gpu_db_wave_persistent").unwrap();
    let entry_scan = CString::new("gpu_db_engine_scan").unwrap();
    let mut func_persist: *mut c_void = ptr::null_mut();
    let mut func_scan: *mut c_void = ptr::null_mut();
    check(
        unsafe { cu_module_get_function(&mut func_persist, module, entry_persist.as_ptr()) },
        "cuModuleGetFunction persist",
    );
    check(
        unsafe { cu_module_get_function(&mut func_scan, module, entry_scan.as_ptr()) },
        "cuModuleGetFunction scan",
    );

    // Two streams on the SAME context — non-blocking so they never serialize via the legacy NULL stream.
    let mut stream_persist: *mut c_void = ptr::null_mut();
    let mut stream_engine: *mut c_void = ptr::null_mut();
    check(
        unsafe { cu_stream_create(&mut stream_persist, CU_STREAM_NON_BLOCKING) },
        "cuStreamCreate persist",
    );
    check(
        unsafe { cu_stream_create(&mut stream_engine, CU_STREAM_NON_BLOCKING) },
        "cuStreamCreate engine",
    );

    println!(
        "# wave R2 SM-coexistence probe  device SMs={sm_count}  rows={rows}  scan_grid={scan_blocks}x{scan_threads}  launches/storm={launches}"
    );
    println!("# col[i]=i, needle={needle} (1 match); pblocks=0 is the no-persistent baseline");
    println!("# two poll modes: busy-spin (sleep_ns=0) vs gentle (~{poll_ns}ns %globaltimer backoff/poll)\n");

    // Run the scan storm; returns (mrows_per_s, launches_per_s).
    let run_storm = || -> (f64, f64) {
        let t = Instant::now();
        for _ in 0..launches {
            unsafe { ptr::write_volatile(result, 0) };
            fence(Ordering::SeqCst);
            let mut a_col = col;
            let mut a_rows = rows;
            let mut a_needle = needle;
            let mut a_res = res_dptr;
            let mut params: [*mut c_void; 4] = [
                ptr::addr_of_mut!(a_col) as *mut c_void,
                ptr::addr_of_mut!(a_rows) as *mut c_void,
                ptr::addr_of_mut!(a_needle) as *mut c_void,
                ptr::addr_of_mut!(a_res) as *mut c_void,
            ];
            check(
                unsafe {
                    cu_launch_kernel(
                        func_scan,
                        scan_blocks,
                        1,
                        1,
                        scan_threads,
                        1,
                        1,
                        0,
                        stream_engine,
                        params.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                },
                "cuLaunchKernel scan",
            );
            check(unsafe { cu_stream_synchronize(stream_engine) }, "sync engine");
            let cnt = unsafe { ptr::read_volatile(result) };
            assert_eq!(cnt, 1, "scan miscounted (got {cnt}, expected 1) — kernel wrong");
        }
        let wall = t.elapsed();
        let secs = wall.as_secs_f64();
        (
            (rows as f64 * launches as f64) / secs / 1.0e6,
            launches as f64 / secs,
        )
    };

    // Baseline (no persistent kernel) — poll-mode-independent.
    let (base_mrows, base_lps) = run_storm();
    println!("  baseline (no persistent kernel):  {base_mrows:.1} Mrows/s  ({base_lps:.0} launches/s)\n");

    let mut all_viable = true;
    let modes: [(&str, u32); 2] = [("busy-spin", 0), ("gentle", poll_ns)];
    for (mode_label, sleep_ns) in modes {
        println!("## persistent poll mode: {mode_label} (sleep_ns={sleep_ns})");
        println!(
            "  {:>8}  {:>9}  {:>14}  {:>11}  {:>11}  {:>12}",
            "pblocks", "SMs~%", "scan Mrows/s", "% of base", "hb/storm", "exit"
        );
        for &pblocks in sweep.iter().filter(|&&b| b > 0) {
            // Launch the persistent kernel reserving ~`pblocks` SMs; confirm it's alive before the storm.
            unsafe {
                ptr::write_volatile(ctrl, 0); // doorbell
                ptr::write_volatile(ctrl.add(1), 0); // heartbeat
            }
            fence(Ordering::SeqCst);
            let mut a_ctrl = ctrl_dptr;
            let mut a_max = max_ns;
            let mut a_sleep = sleep_ns;
            let mut pparams: [*mut c_void; 3] = [
                ptr::addr_of_mut!(a_ctrl) as *mut c_void,
                ptr::addr_of_mut!(a_max) as *mut c_void,
                ptr::addr_of_mut!(a_sleep) as *mut c_void,
            ];
            check(
                unsafe {
                    cu_launch_kernel(
                        func_persist,
                        pblocks,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        stream_persist,
                        pparams.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                },
                "cuLaunchKernel persist",
            );
            std::thread::sleep(Duration::from_millis(50));
            let hb_alive = unsafe { ptr::read_volatile(ctrl.add(1)) };
            assert!(
                hb_alive > 0,
                "pblocks={pblocks} ({mode_label}): persistent kernel never advanced its heartbeat — not running"
            );

            // The storm runs concurrently with the resident persistent kernel.
            let hb_before = unsafe { ptr::read_volatile(ctrl.add(1)) };
            let (mrows, lps) = run_storm();
            let hb_after = unsafe { ptr::read_volatile(ctrl.add(1)) };
            let hb_advanced = hb_after.wrapping_sub(hb_before);
            let starved = hb_advanced == 0;
            if starved {
                all_viable = false;
            }

            // Clean exit on the doorbell (not the backstop).
            unsafe { ptr::write_volatile(ctrl, 1) };
            fence(Ordering::SeqCst);
            let rang = Instant::now();
            check(unsafe { cu_stream_synchronize(stream_persist) }, "sync persist");
            let exit_latency = rang.elapsed();
            let clean_exit = exit_latency < Duration::from_secs(2);
            if !clean_exit {
                all_viable = false;
            }

            let pct = mrows / base_mrows * 100.0;
            let sm_pct = pblocks as f64 / sm_count as f64 * 100.0;
            println!(
                "  {:>8}  {:>8.1}%  {:>14.1}  {:>10.1}%  {:>11}  {:>12}  ({lps:.0} launches/s)",
                pblocks,
                sm_pct,
                mrows,
                pct,
                if starved {
                    "STARVED".to_string()
                } else {
                    format!("+{hb_advanced}")
                },
                if clean_exit {
                    format!("{}us", exit_latency.as_micros())
                } else {
                    format!("BACKSTOP {:?}", exit_latency)
                },
            );
        }
        println!();
    }

    // Teardown.
    check(unsafe { cu_stream_destroy(stream_persist) }, "destroy persist");
    check(unsafe { cu_stream_destroy(stream_engine) }, "destroy engine");
    check(unsafe { cu_module_unload(module) }, "cuModuleUnload");
    check(unsafe { cu_mem_free_host(ctrl_host) }, "free ctrl");
    check(unsafe { cu_mem_free_host(res_host) }, "free res");
    check(unsafe { cu_mem_free(col) }, "free col");
    check(
        unsafe { cu_ctx_release(device) },
        "cuDevicePrimaryCtxRelease",
    );

    if all_viable && sweep.iter().any(|&b| b > 0) {
        println!(
            "\nOK: COEXISTENCE VIABLE — every reserved-SM run kept the persistent kernel advancing through\n\
             the scan storm AND exited cleanly on the doorbell; scan throughput vs baseline is the SM-\n\
             reservation cost above. An always-resident wave kernel can share the engine's context."
        );
    } else if sweep.iter().any(|&b| b > 0) {
        println!(
            "\nWARN: a reserved-SM run STARVED the persistent kernel or hit the backstop — coexistence is\n\
             NOT free on this box; R2 needs SM reservation / MPS / time-slicing (see the rows above)."
        );
    } else {
        println!("\n(baseline-only sweep — add pblocks>0 to test coexistence)");
    }
}
