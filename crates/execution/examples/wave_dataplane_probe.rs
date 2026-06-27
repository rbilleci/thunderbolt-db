//! Wave-engine increment 1b (step i) — int4 point-lookup DATA PLANE on the persistent kernel.
//!
//! Builds on 1a's proven lifecycle ([`wave_lifecycle_probe`]). A persistent kernel drains a host-pinned
//! request ring of int4 needles, scans a RESIDENT key column for each needle, gathers a payload column
//! at the matching row, and writes (found, value) to a result slot; the host enqueues needles and reads
//! results back — NO per-request host materialization (the whole point of the wave model). This step is
//! SINGLE-THREAD (correctness first; 1b-ii parallelizes the scan). Verifies present needles (gather is
//! correct) AND absent needles (not-found is reported, not a wrong row). Standalone (own libcuda +
//! context) so a bug can't touch the engine; the kernel always terminates (doorbell OR `%globaltimer`
//! wall-clock backstop).
//!
//! Run: `cargo run --release -p gpu_db_execution --example wave_dataplane_probe`

use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

use libloading::{Library, Symbol};

/// Persistent single-thread data-plane kernel. Drains request slots [tail..head); for each, serial-scans
/// `col_key[0..rows]` for the needle and writes `(done, value)` to the result slot — done=1 + payload on
/// match, done=2 (not found) otherwise. Exits on the doorbell or the `%globaltimer` wall-clock backstop.
const PTX_SRC: &str = r#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_wave_dataplane(
    .param .u64 ctrl,
    .param .u64 req,
    .param .u64 res,
    .param .u64 col_key,
    .param .u64 col_payload,
    .param .u32 row_count,
    .param .u32 ring_mask,
    .param .u64 max_ns
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<24>;

    ld.param.u64 %rd1, [ctrl];
    ld.param.u64 %rd2, [req];
    ld.param.u64 %rd3, [res];
    ld.param.u64 %rd4, [col_key];
    ld.param.u64 %rd5, [col_payload];
    ld.param.u32 %r1, [row_count];
    ld.param.u32 %r2, [ring_mask];
    ld.param.u64 %rd6, [max_ns];
    mov.u64 %rd7, %globaltimer;

$L_outer:
    ld.volatile.global.u32 %r3, [%rd1];
    setp.ne.s32 %p1, %r3, 0;
    @%p1 bra $L_done;
    ld.volatile.global.u32 %r4, [%rd1+4];
    ld.volatile.global.u32 %r5, [%rd1+8];
    setp.ne.s32 %p2, %r4, %r5;
    @%p2 bra $L_process;
    mov.u64 %rd8, %globaltimer;
    sub.u64 %rd9, %rd8, %rd7;
    setp.ge.u64 %p3, %rd9, %rd6;
    @%p3 bra $L_done;
    bra $L_outer;

$L_process:
    and.b32 %r6, %r5, %r2;
    mul.wide.u32 %rd10, %r6, 4;
    add.u64 %rd11, %rd2, %rd10;
    ld.volatile.global.u32 %r7, [%rd11];
    mov.u32 %r8, 0;
    mov.u32 %r9, 2;
    mov.u32 %r10, 0;

$L_scan:
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra $L_write;
    mul.wide.u32 %rd12, %r8, 4;
    add.u64 %rd13, %rd4, %rd12;
    ld.global.u32 %r11, [%rd13];
    setp.ne.s32 %p2, %r11, %r7;
    @%p2 bra $L_next;
    add.u64 %rd14, %rd5, %rd12;
    ld.global.u32 %r10, [%rd14];
    mov.u32 %r9, 1;
    bra $L_write;
$L_next:
    add.u32 %r8, %r8, 1;
    bra $L_scan;

$L_write:
    mul.wide.u32 %rd15, %r6, 8;
    add.u64 %rd16, %rd3, %rd15;
    st.volatile.global.u32 [%rd16+4], %r10;
    membar.sys;
    st.volatile.global.u32 [%rd16], %r9;
    add.u32 %r5, %r5, 1;
    st.volatile.global.u32 [%rd1+8], %r5;
    bra $L_outer;

$L_done:
    ret;
}
"#;

const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
const RING: u32 = 4096; // power of two; >= request count below so the ring never wraps in this probe
const ROWS: u32 = 4000;
const REQUESTS: u32 = 2000;

type Fn1U32 = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
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
type CuModuleGetFunction = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
type CuStreamCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
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

fn map_alloc(
    alloc: &CuMemHostAlloc,
    get_dptr: &CuMemHostGetDevicePointer,
    bytes: usize,
) -> (*mut c_void, u64) {
    let mut host: *mut c_void = ptr::null_mut();
    check(
        unsafe { alloc(&mut host, bytes, CU_MEMHOSTALLOC_DEVICEMAP) },
        "cuMemHostAlloc",
    );
    let mut dptr: u64 = 0;
    check(
        unsafe { get_dptr(&mut dptr, host, 0) },
        "cuMemHostGetDevicePointer",
    );
    (host, dptr)
}

fn main() {
    let max_ns: u64 = 30_000_000_000;
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
    let cu_stream_destroy: Symbol<Fn1Ptr> =
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

    // Resident columns: key[i] = i, payload[i] = i*1000 + 7 (distinct, so the gather is load-bearing).
    let key: Vec<i32> = (0..ROWS as i32).collect();
    let payload: Vec<i32> = (0..ROWS as i32).map(|i| i * 1000 + 7).collect();
    let mut col_key: u64 = 0;
    let mut col_payload: u64 = 0;
    check(
        unsafe { cu_mem_alloc(&mut col_key, ROWS as usize * 4) },
        "cuMemAlloc key",
    );
    check(
        unsafe { cu_mem_alloc(&mut col_payload, ROWS as usize * 4) },
        "cuMemAlloc payload",
    );
    check(
        unsafe { cu_memcpy_htod(col_key, key.as_ptr() as *const c_void, ROWS as usize * 4) },
        "HtoD key",
    );
    check(
        unsafe {
            cu_memcpy_htod(
                col_payload,
                payload.as_ptr() as *const c_void,
                ROWS as usize * 4,
            )
        },
        "HtoD payload",
    );

    // Device-mapped pinned ring/control/result.
    let (ctrl_host, ctrl_dptr) = map_alloc(&cu_mem_host_alloc, &cu_mem_host_get_device_pointer, 16);
    let (req_host, req_dptr) = map_alloc(
        &cu_mem_host_alloc,
        &cu_mem_host_get_device_pointer,
        RING as usize * 4,
    );
    let (res_host, res_dptr) = map_alloc(
        &cu_mem_host_alloc,
        &cu_mem_host_get_device_pointer,
        RING as usize * 8,
    );
    let ctrl = ctrl_host as *mut u32; // [doorbell, head, tail, _]
    let reqs = req_host as *mut i32;
    let results = res_host as *mut u32; // pairs [done, value]
    unsafe {
        for i in 0..4 {
            ptr::write_volatile(ctrl.add(i), 0);
        }
    }

    // Enqueue REQUESTS needles: i in [0, REQUESTS) are PRESENT (i < ROWS); also a block of ABSENT needles.
    // First (REQUESTS-4) present, last 4 absent (>= ROWS) to test not-found.
    let absent_from = REQUESTS - 4;
    unsafe {
        for i in 0..REQUESTS {
            let needle = if i < absent_from {
                i as i32
            } else {
                (ROWS + i) as i32
            };
            ptr::write_volatile(reqs.add(i as usize), needle);
            // zero the result slot
            ptr::write_volatile(results.add(i as usize * 2), 0);
            ptr::write_volatile(results.add(i as usize * 2 + 1), 0);
        }
    }
    fence(Ordering::SeqCst);

    let ptx = CString::new(PTX_SRC).unwrap();
    let entry = CString::new("gpu_db_wave_dataplane").unwrap();
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

    let mut a_ctrl = ctrl_dptr;
    let mut a_req = req_dptr;
    let mut a_res = res_dptr;
    let mut a_key = col_key;
    let mut a_pay = col_payload;
    let mut a_rows = ROWS;
    let mut a_mask = RING - 1;
    let mut a_max = max_ns;
    let mut params: [*mut c_void; 8] = [
        ptr::addr_of_mut!(a_ctrl) as *mut c_void,
        ptr::addr_of_mut!(a_req) as *mut c_void,
        ptr::addr_of_mut!(a_res) as *mut c_void,
        ptr::addr_of_mut!(a_key) as *mut c_void,
        ptr::addr_of_mut!(a_pay) as *mut c_void,
        ptr::addr_of_mut!(a_rows) as *mut c_void,
        ptr::addr_of_mut!(a_mask) as *mut c_void,
        ptr::addr_of_mut!(a_max) as *mut c_void,
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

    // Publish all requests at once: head = REQUESTS. Then poll each result slot's done flag.
    let started = Instant::now();
    unsafe { ptr::write_volatile(ctrl.add(1), REQUESTS) };
    fence(Ordering::SeqCst);

    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let last_done = unsafe { ptr::read_volatile(results.add((REQUESTS as usize - 1) * 2)) };
        if last_done != 0 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the wave to drain"
        );
        std::hint::spin_loop();
    }
    let drain_elapsed = started.elapsed();

    // Verify correctness.
    let mut found = 0u32;
    let mut notfound = 0u32;
    for i in 0..REQUESTS as usize {
        let done = unsafe { ptr::read_volatile(results.add(i * 2)) };
        let value = unsafe { ptr::read_volatile(results.add(i * 2 + 1)) } as i32;
        assert_ne!(done, 0, "request {i} never completed");
        if (i as u32) < absent_from {
            assert_eq!(
                done, 1,
                "request {i} (present needle {i}) reported not-found"
            );
            assert_eq!(
                value,
                i as i32 * 1000 + 7,
                "request {i} gathered the wrong payload"
            );
            found += 1;
        } else {
            assert_eq!(done, 2, "request {i} (absent needle) reported found");
            notfound += 1;
        }
    }

    unsafe { ptr::write_volatile(ctrl, 1) }; // doorbell
    fence(Ordering::SeqCst);
    check(
        unsafe { cu_stream_synchronize(stream) },
        "cuStreamSynchronize",
    );

    let throughput = REQUESTS as f64 / drain_elapsed.as_secs_f64();
    println!(
        "# wave-engine 1b-i data-plane probe  rows={ROWS} requests={REQUESTS} (single-thread scan)"
    );
    println!("  correctness: {found} present (gather verified) + {notfound} absent (not-found) = ALL CORRECT");
    println!("  drain {REQUESTS} requests in {drain_elapsed:?}  ->  {throughput:.0} req/s (single-thread baseline)");

    check(unsafe { cu_stream_destroy(stream) }, "cuStreamDestroy");
    check(unsafe { cu_mem_free_host(ctrl_host) }, "free ctrl");
    check(unsafe { cu_mem_free_host(req_host) }, "free req");
    check(unsafe { cu_mem_free_host(res_host) }, "free res");
    check(unsafe { cu_mem_free(col_key) }, "free key");
    check(unsafe { cu_mem_free(col_payload) }, "free payload");
    check(
        unsafe { cu_ctx_release(device) },
        "cuDevicePrimaryCtxRelease",
    );
    println!("OK: data plane correct, clean exit");
}
