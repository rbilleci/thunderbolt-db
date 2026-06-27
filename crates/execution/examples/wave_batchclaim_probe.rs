//! Wave-engine increment 1d-ii — push further: BATCHED claiming (cut atomic + fence frequency K×).
//!
//! 1d-i moved the atomics to device memory (~30M req/s) but the single `claim`/`completed` counter is
//! now contention-bound (peaks at LOW thread count). This cuts that contention K× via batched claiming:
//! each thread reserves K requests per `atom.add(claim, K)`, processes the whole [base, min(base+K,head))
//! range, then bumps `completed` ONCE by the batch count under a single `membar.sys` (instead of one
//! atomic + one fence PER request). Same device-memory atomics, GPU index, host-mapped `all_done`.
//!
//! Run: `cargo run --release -p gpu_db_execution --example wave_batchclaim_probe`
//! Env: GPU_DB_WAVE_THREADS (8192), GPU_DB_WAVE_REQUESTS (200000), GPU_DB_WAVE_ROWS (1000000),
//!      GPU_DB_WAVE_CLAIM_BATCH (K, default 32).

use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

use libloading::{Library, Symbol};

/// Same as 1d-i but the claim/complete is BATCHED: `atom.add(claim, K)` reserves a range; the inner loop
/// hash-probes + gathers each request in it; one `membar.sys` + one `atom.add(completed, count)` per batch.
const PTX_SRC: &str = r#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_wave_batchclaim(
    .param .u64 ctrl,
    .param .u64 counters,
    .param .u64 req,
    .param .u64 res,
    .param .u64 table,
    .param .u64 col_payload,
    .param .u32 hash_shift,
    .param .u32 table_mask,
    .param .u32 ring_mask,
    .param .u32 claim_batch,
    .param .u64 max_ns
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<28>;
    .reg .b64 %rd<36>;

    ld.param.u64 %rd1, [ctrl];
    ld.param.u64 %rd2, [counters];
    ld.param.u64 %rd3, [req];
    ld.param.u64 %rd4, [res];
    ld.param.u64 %rd5, [table];
    ld.param.u64 %rd6, [col_payload];
    ld.param.u32 %r1, [hash_shift];
    ld.param.u32 %r2, [table_mask];
    ld.param.u32 %r3, [ring_mask];
    ld.param.u32 %r20, [claim_batch];
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
    atom.global.add.u32 %r7, [%rd2], %r20;
    setp.ge.u32 %p1, %r7, %r5;
    @%p1 bra $L_loop;
    add.u32 %r21, %r7, %r20;
    min.u32 %r22, %r21, %r5;
    mov.u32 %r23, 0;
    mov.u32 %r24, %r7;

$L_inner:
    setp.ge.u32 %p2, %r24, %r22;
    @%p2 bra $L_after;
    and.b32 %r8, %r24, %r3;
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
    setp.eq.u64 %p3, %rd15, 0;
    @%p3 bra $L_pwrite;
    shr.u64 %rd16, %rd15, 32;
    cvt.u32.u64 %r14, %rd16;
    setp.ne.s32 %p3, %r14, %r9;
    @%p3 bra $L_pnext;
    cvt.u32.u64 %r15, %rd15;
    sub.u32 %r15, %r15, 1;
    mul.wide.u32 %rd17, %r15, 4;
    add.u64 %rd18, %rd6, %rd17;
    ld.global.u32 %r13, [%rd18];
    mov.u32 %r12, 1;
    bra $L_pwrite;
$L_pnext:
    add.u32 %r11, %r11, 1;
    add.u32 %r17, %r17, 1;
    setp.ge.u32 %p4, %r17, 256;
    @%p4 bra $L_pwrite;
    bra $L_probe;

$L_pwrite:
    cvt.u64.u32 %rd19, %r13;
    shl.b64 %rd20, %rd19, 32;
    cvt.u64.u32 %rd21, %r12;
    or.b64 %rd22, %rd20, %rd21;
    mul.wide.u32 %rd23, %r8, 8;
    add.u64 %rd24, %rd4, %rd23;
    st.volatile.global.u64 [%rd24], %rd22;
    add.u32 %r23, %r23, 1;
    add.u32 %r24, %r24, 1;
    bra $L_inner;

$L_after:
    membar.sys;
    atom.global.add.u32 %r18, [%rd2+4], %r23;
    add.u32 %r18, %r18, %r23;
    setp.ne.u32 %p5, %r18, %r5;
    @%p5 bra $L_loop;
    membar.sys;
    mov.u32 %r19, 1;
    st.volatile.global.u32 [%rd1+8], %r19;
    bra $L_loop;

$L_done:
    ret;
}
"#;

const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;

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

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let max_ns: u64 = 30_000_000_000;
    let threads = env_u32("GPU_DB_WAVE_THREADS", 8192).max(256);
    let blocks = threads.div_ceil(256);
    let requests = env_u32("GPU_DB_WAVE_REQUESTS", 200_000);
    let rows = env_u32("GPU_DB_WAVE_ROWS", 1_000_000);
    // K=8 is the sweet spot: large enough to cut atomic frequency 8x, small enough that batches still
    // outnumber threads (so all stay busy). K>=32 collapses (fewer batches than threads → under-parallel
    // + each thread serializes K host-mapped result writes). See the commit message's sweep.
    let claim_batch = env_u32("GPU_DB_WAVE_CLAIM_BATCH", 8).max(1);
    let ring = (requests + 8).next_power_of_two();
    let absent_from = requests - (requests / 100).clamp(4, requests);

    let table_size = (rows.saturating_mul(2)).next_power_of_two().max(2);
    let table_mask = table_size - 1;
    let hash_shift = 32 - table_size.trailing_zeros();
    let mut table = vec![0u64; table_size as usize];
    for i in 0..rows {
        let mut h = (i.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
        while table[h as usize] != 0 {
            h = (h + 1) & table_mask;
        }
        table[h as usize] = ((i as u64) << 32) | (i as u64 + 1);
    }
    let payload: Vec<i32> = (0..rows as i32).map(|i| i * 1000 + 7).collect();

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

    let mut d_table: u64 = 0;
    let mut d_payload: u64 = 0;
    let mut d_counters: u64 = 0;
    check(
        unsafe { cu_mem_alloc(&mut d_table, table_size as usize * 8) },
        "cuMemAlloc table",
    );
    check(
        unsafe { cu_mem_alloc(&mut d_payload, rows as usize * 4) },
        "cuMemAlloc payload",
    );
    check(
        unsafe { cu_mem_alloc(&mut d_counters, 8) },
        "cuMemAlloc counters",
    );
    check(
        unsafe {
            cu_memcpy_htod(
                d_table,
                table.as_ptr() as *const c_void,
                table_size as usize * 8,
            )
        },
        "HtoD table",
    );
    check(
        unsafe {
            cu_memcpy_htod(
                d_payload,
                payload.as_ptr() as *const c_void,
                rows as usize * 4,
            )
        },
        "HtoD payload",
    );
    let zero = [0u32; 2];
    check(
        unsafe { cu_memcpy_htod(d_counters, zero.as_ptr() as *const c_void, 8) },
        "HtoD counters zero",
    );

    let (ctrl_host, ctrl_dptr) = map_alloc(&cu_mem_host_alloc, &cu_mem_host_get_device_pointer, 16);
    let (req_host, req_dptr) = map_alloc(
        &cu_mem_host_alloc,
        &cu_mem_host_get_device_pointer,
        ring as usize * 4,
    );
    let (res_host, res_dptr) = map_alloc(
        &cu_mem_host_alloc,
        &cu_mem_host_get_device_pointer,
        ring as usize * 8,
    );
    let ctrl = ctrl_host as *mut u32;
    let reqs = req_host as *mut i32;
    let results = res_host as *mut u64;
    unsafe {
        for i in 0..4 {
            ptr::write_volatile(ctrl.add(i), 0);
        }
        for i in 0..requests {
            let needle = if i < absent_from {
                (i % rows) as i32
            } else {
                (rows + i) as i32
            };
            ptr::write_volatile(reqs.add(i as usize), needle);
            ptr::write_volatile(results.add(i as usize), 0);
        }
    }
    fence(Ordering::SeqCst);

    let ptx = CString::new(PTX_SRC).unwrap();
    let entry = CString::new("gpu_db_wave_batchclaim").unwrap();
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
    let mut a_counters = d_counters;
    let mut a_req = req_dptr;
    let mut a_res = res_dptr;
    let mut a_table = d_table;
    let mut a_pay = d_payload;
    let mut a_shift = hash_shift;
    let mut a_tmask = table_mask;
    let mut a_rmask = ring - 1;
    let mut a_batch = claim_batch;
    let mut a_max = max_ns;
    let mut params: [*mut c_void; 11] = [
        ptr::addr_of_mut!(a_ctrl) as *mut c_void,
        ptr::addr_of_mut!(a_counters) as *mut c_void,
        ptr::addr_of_mut!(a_req) as *mut c_void,
        ptr::addr_of_mut!(a_res) as *mut c_void,
        ptr::addr_of_mut!(a_table) as *mut c_void,
        ptr::addr_of_mut!(a_pay) as *mut c_void,
        ptr::addr_of_mut!(a_shift) as *mut c_void,
        ptr::addr_of_mut!(a_tmask) as *mut c_void,
        ptr::addr_of_mut!(a_rmask) as *mut c_void,
        ptr::addr_of_mut!(a_batch) as *mut c_void,
        ptr::addr_of_mut!(a_max) as *mut c_void,
    ];
    check(
        unsafe {
            cu_launch_kernel(
                func,
                blocks,
                1,
                1,
                256,
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

    let started = Instant::now();
    unsafe { ptr::write_volatile(ctrl.add(1), requests) };
    fence(Ordering::SeqCst);
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let done = unsafe { ptr::read_volatile(ctrl.add(2)) };
        if done != 0 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the wave to drain"
        );
        std::hint::spin_loop();
    }
    let drain_elapsed = started.elapsed();

    let mut dev_counters = [0u32; 2];
    check(
        unsafe { cu_memcpy_dtoh(dev_counters.as_mut_ptr() as *mut c_void, d_counters, 8) },
        "DtoH counters",
    );
    assert_eq!(dev_counters[1], requests, "device completed != requests");

    let mut found = 0u32;
    let mut notfound = 0u32;
    for i in 0..requests as usize {
        let packed = unsafe { ptr::read_volatile(results.add(i)) };
        let done = (packed & 0xffff_ffff) as u32;
        let value = (packed >> 32) as i32;
        assert_ne!(done, 0, "request {i} never completed");
        if (i as u32) < absent_from {
            let expected = (i as u32 % rows) as i32 * 1000 + 7;
            assert_eq!(done, 1, "request {i} (present) reported not-found");
            assert_eq!(value, expected, "request {i} wrong payload");
            found += 1;
        } else {
            assert_eq!(done, 2, "request {i} (absent) reported found");
            notfound += 1;
        }
    }

    unsafe { ptr::write_volatile(ctrl, 1) };
    fence(Ordering::SeqCst);
    check(
        unsafe { cu_stream_synchronize(stream) },
        "cuStreamSynchronize",
    );

    let throughput = requests as f64 / drain_elapsed.as_secs_f64();
    println!(
        "# wave 1d-ii BATCHED-claim indexed point lookup  rows={rows} requests={requests} threads={} ({blocks}x256) K={claim_batch}",
        blocks * 256
    );
    println!("  correctness: {found} present (gather verified) + {notfound} absent = ALL CORRECT");
    println!(
        "  GPU drain {requests} INDEXED lookups in {drain_elapsed:?}  ->  {throughput:.0} req/s  \
         (vs 1d-i device-atomic K=1: ~30M)"
    );

    check(unsafe { cu_stream_destroy(stream) }, "cuStreamDestroy");
    check(unsafe { cu_mem_free_host(ctrl_host) }, "free ctrl");
    check(unsafe { cu_mem_free_host(req_host) }, "free req");
    check(unsafe { cu_mem_free_host(res_host) }, "free res");
    check(unsafe { cu_mem_free(d_table) }, "free table");
    check(unsafe { cu_mem_free(d_payload) }, "free payload");
    check(unsafe { cu_mem_free(d_counters) }, "free counters");
    check(
        unsafe { cu_ctx_release(device) },
        "cuDevicePrimaryCtxRelease",
    );
    println!("OK: batched-claim indexed data plane correct, clean exit");
}
