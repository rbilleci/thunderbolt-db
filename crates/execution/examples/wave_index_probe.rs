//! Wave-engine increment 1c — GPU **index** point lookup (removes the O(rows) scan).
//!
//! The 1b full-scan data plane is O(rows)/lookup, so a 1M-row table fell to ~485k req/s (scan-bound).
//! ADR-009 specifies a GPU index. This swaps the in-kernel serial scan for a hash-table probe: a
//! host-built open-addressing table (`(key<<32)|(row+1)`, 0 = empty, load factor 0.5) is uploaded once;
//! the persistent kernel hashes the needle (Fibonacci: `(needle * 0x9E3779B1) >> shift` — high bits, so
//! sequential keys still distribute, not a perfect-hash artifact), linear-probes to the match, and
//! gathers the payload at that row. O(1) average — so the wave engine should be atomic-ceiling-bound
//! again at ANY table size, not scan-bound. Probe count is hard-capped (defensive: a probe loop inside
//! a request would evade the `%globaltimer` backstop and zombie the context).
//!
//! Verifies present needles (gather correct) + absent (not-found), then measures vs the 1b scan curve
//! (1M rows: ~485k req/s). Standalone (own libcuda + context).
//!
//! Run: `cargo run --release -p gpu_db_execution --example wave_index_probe`
//! Env: GPU_DB_WAVE_THREADS (8192), GPU_DB_WAVE_REQUESTS (200000), GPU_DB_WAVE_ROWS (1000000).

use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

use libloading::{Library, Symbol};

const PTX_SRC: &str = r#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_wave_index(
    .param .u64 ctrl,
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
    .reg .pred %p<4>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<28>;

    ld.param.u64 %rd1, [ctrl];
    ld.param.u64 %rd2, [req];
    ld.param.u64 %rd3, [res];
    ld.param.u64 %rd4, [table];
    ld.param.u64 %rd5, [col_payload];
    ld.param.u32 %r1, [hash_shift];
    ld.param.u32 %r2, [table_mask];
    ld.param.u32 %r3, [ring_mask];
    ld.param.u64 %rd6, [max_ns];
    mov.u64 %rd7, %globaltimer;

$L_loop:
    ld.volatile.global.u32 %r4, [%rd1];
    setp.ne.s32 %p1, %r4, 0;
    @%p1 bra $L_done;
    mov.u64 %rd8, %globaltimer;
    sub.u64 %rd9, %rd8, %rd7;
    setp.ge.u64 %p1, %rd9, %rd6;
    @%p1 bra $L_done;
    ld.volatile.global.u32 %r5, [%rd1+4];
    ld.volatile.global.u32 %r6, [%rd1+8];
    setp.ge.u32 %p1, %r6, %r5;
    @%p1 bra $L_loop;
    atom.global.add.u32 %r7, [%rd1+8], 1;
    setp.ge.u32 %p1, %r7, %r5;
    @%p1 bra $L_loop;
    and.b32 %r8, %r7, %r3;
    mul.wide.u32 %rd10, %r8, 4;
    add.u64 %rd11, %rd2, %rd10;
    ld.volatile.global.u32 %r9, [%rd11];
    mul.lo.u32 %r10, %r9, 2654435761;
    shr.u32 %r11, %r10, %r1;
    mov.u32 %r12, 2;
    mov.u32 %r13, 0;
    mov.u32 %r17, 0;

$L_probe:
    and.b32 %r11, %r11, %r2;
    mul.wide.u32 %rd12, %r11, 8;
    add.u64 %rd13, %rd4, %rd12;
    ld.global.u64 %rd14, [%rd13];
    setp.eq.u64 %p2, %rd14, 0;
    @%p2 bra $L_write;
    shr.u64 %rd15, %rd14, 32;
    cvt.u32.u64 %r14, %rd15;
    setp.ne.s32 %p2, %r14, %r9;
    @%p2 bra $L_probe_next;
    cvt.u32.u64 %r15, %rd14;
    sub.u32 %r15, %r15, 1;
    mul.wide.u32 %rd16, %r15, 4;
    add.u64 %rd17, %rd5, %rd16;
    ld.global.u32 %r13, [%rd17];
    mov.u32 %r12, 1;
    bra $L_write;
$L_probe_next:
    add.u32 %r11, %r11, 1;
    add.u32 %r17, %r17, 1;
    setp.ge.u32 %p3, %r17, 256;
    @%p3 bra $L_write;
    bra $L_probe;

$L_write:
    cvt.u64.u32 %rd18, %r13;
    shl.b64 %rd19, %rd18, 32;
    cvt.u64.u32 %rd20, %r12;
    or.b64 %rd21, %rd19, %rd20;
    mul.wide.u32 %rd22, %r8, 8;
    add.u64 %rd23, %rd3, %rd22;
    st.volatile.global.u64 [%rd23], %rd21;
    membar.sys;
    atom.global.add.u32 %r16, [%rd1+12], 1;
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
    let ring = (requests + 8).next_power_of_two();
    let absent_from = requests - (requests / 100).clamp(4, requests);

    // Host-built open-addressing hash table: key[i]=i -> row i. load factor 0.5 (so an absent probe
    // always hits an empty slot fast). Fibonacci hash on the TOP bits => sequential keys still spread.
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
    check(
        unsafe { cu_mem_alloc(&mut d_table, table_size as usize * 8) },
        "cuMemAlloc table",
    );
    check(
        unsafe { cu_mem_alloc(&mut d_payload, rows as usize * 4) },
        "cuMemAlloc payload",
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
    let entry = CString::new("gpu_db_wave_index").unwrap();
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
    let mut a_table = d_table;
    let mut a_pay = d_payload;
    let mut a_shift = hash_shift;
    let mut a_tmask = table_mask;
    let mut a_rmask = ring - 1;
    let mut a_max = max_ns;
    let mut params: [*mut c_void; 9] = [
        ptr::addr_of_mut!(a_ctrl) as *mut c_void,
        ptr::addr_of_mut!(a_req) as *mut c_void,
        ptr::addr_of_mut!(a_res) as *mut c_void,
        ptr::addr_of_mut!(a_table) as *mut c_void,
        ptr::addr_of_mut!(a_pay) as *mut c_void,
        ptr::addr_of_mut!(a_shift) as *mut c_void,
        ptr::addr_of_mut!(a_tmask) as *mut c_void,
        ptr::addr_of_mut!(a_rmask) as *mut c_void,
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
        let done = unsafe { ptr::read_volatile(ctrl.add(3)) };
        if done >= requests {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the wave to drain"
        );
        std::hint::spin_loop();
    }
    let drain_elapsed = started.elapsed();

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
            assert_eq!(value, expected, "request {i} gathered the wrong payload");
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
        "# wave 1c GPU-INDEX point lookup  rows={rows} requests={requests} threads={} ({blocks}x256)",
        blocks * 256
    );
    println!("  correctness: {found} present (gather verified) + {notfound} absent (not-found) = ALL CORRECT");
    println!(
        "  drained {requests} INDEXED point lookups in {drain_elapsed:?}  ->  {throughput:.0} req/s  \
         (vs 1b full-scan @1M rows: ~485k req/s)"
    );

    check(unsafe { cu_stream_destroy(stream) }, "cuStreamDestroy");
    check(unsafe { cu_mem_free_host(ctrl_host) }, "free ctrl");
    check(unsafe { cu_mem_free_host(req_host) }, "free req");
    check(unsafe { cu_mem_free_host(res_host) }, "free res");
    check(unsafe { cu_mem_free(d_table) }, "free table");
    check(unsafe { cu_mem_free(d_payload) }, "free payload");
    check(
        unsafe { cu_ctx_release(device) },
        "cuDevicePrimaryCtxRelease",
    );
    println!("OK: indexed data plane correct, clean exit");
}
