//! Wave-engine write path — probe 1: CONCURRENT lock-free index INSERT throughput.
//!
//! Reads are settled (O(1), tens of millions/s). The write path is the unmeasured frontier, and its
//! novel + historically-hard piece is **concurrent index maintenance**: many GPU threads inserting
//! (key,row) into a SHARED open-addressing hash table with NO locks (locks deadlock on this box —
//! lock-free atomics only). This probe measures exactly that: each thread `atom.cas.b64`-installs its
//! keys into the table (linear-probe on collision, idempotent on same-key), and we measure inserts/s +
//! verify EVERY key landed exactly once at the right (key,row) (no lost / duplicate / torn insert).
//! Column append (`payload[row]=...`) is included (it's the trivial, un-contended part of an INSERT).
//!
//! One-shot grid-stride kernel (this measures raw insert throughput; the persistent-kernel/ring
//! integration is later). Probe count is hard-capped so a one-shot kernel can never infinite-loop and
//! hang `cuStreamSynchronize` (which would zombie the `--gpu-reset`-denied box) — a capped insert is
//! "lost" and caught by the count==N verification, never a hang.
//!
//! Run: `cargo run --release -p gpu_db_execution --example wave_index_insert_probe`
//! Env: GPU_DB_WAVE_INSERTS (N, default 1000000), GPU_DB_WAVE_THREADS (default 65536).

use std::ffi::{c_void, CString};
use std::ptr;
use std::time::Instant;

use libloading::{Library, Symbol};

/// Each thread grid-strides over key indices; for key i it writes `payload[i]=i*1000+7` then installs
/// `(i<<32)|(i+1)` into the hash table via `atom.cas.b64(slot, 0, val)` — linear-probe on a different-key
/// collision, idempotent if the same key is already present, hard-capped at 4096 probes (defensive).
const PTX_SRC: &str = r#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_index_insert(
    .param .u64 table,
    .param .u64 col_payload,
    .param .u32 table_mask,
    .param .u32 hash_shift,
    .param .u32 n
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<16>;

    ld.param.u64 %rd1, [table];
    ld.param.u64 %rd2, [col_payload];
    ld.param.u32 %r1, [table_mask];
    ld.param.u32 %r2, [hash_shift];
    ld.param.u32 %r3, [n];

    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r5, %r4, %r6;
    mov.u32 %r8, %nctaid.x;
    mul.lo.u32 %r9, %r8, %r4;
    mov.u32 %r10, %r7;

$L_loop:
    setp.ge.u32 %p1, %r10, %r3;
    @%p1 bra $L_done;
    mul.lo.u32 %r11, %r10, 1000;
    add.u32 %r11, %r11, 7;
    mul.wide.u32 %rd3, %r10, 4;
    add.u64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r11;
    cvt.u64.u32 %rd5, %r10;
    shl.b64 %rd6, %rd5, 32;
    add.u32 %r12, %r10, 1;
    cvt.u64.u32 %rd7, %r12;
    or.b64 %rd8, %rd6, %rd7;
    mul.lo.u32 %r13, %r10, 2654435761;
    shr.u32 %r14, %r13, %r2;
    mov.u32 %r16, 0;

$L_probe:
    and.b32 %r14, %r14, %r1;
    mul.wide.u32 %rd9, %r14, 8;
    add.u64 %rd10, %rd1, %rd9;
    mov.u64 %rd11, 0;
    atom.global.cas.b64 %rd12, [%rd10], %rd11, %rd8;
    setp.eq.u64 %p2, %rd12, 0;
    @%p2 bra $L_next;
    shr.u64 %rd13, %rd12, 32;
    cvt.u32.u64 %r15, %rd13;
    setp.eq.s32 %p3, %r15, %r10;
    @%p3 bra $L_next;
    add.u32 %r14, %r14, 1;
    add.u32 %r16, %r16, 1;
    setp.ge.u32 %p4, %r16, 4096;
    @%p4 bra $L_next;
    bra $L_probe;

$L_next:
    add.u32 %r10, %r10, %r9;
    bra $L_loop;

$L_done:
    ret;
}
"#;

type Fn1U32 = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuCtxRetain = unsafe extern "C" fn(*mut *mut c_void, i32) -> i32;
type CuCtxRelease = unsafe extern "C" fn(i32) -> i32;
type Fn1Ptr = unsafe extern "C" fn(*mut c_void) -> i32;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFree = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
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

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n = env_u32("GPU_DB_WAVE_INSERTS", 1_000_000);
    let threads = env_u32("GPU_DB_WAVE_THREADS", 65536).max(256);
    let blocks = threads.div_ceil(256);
    let table_size = (n.saturating_mul(2)).next_power_of_two().max(2);
    let table_mask = table_size - 1;
    let hash_shift = 32 - table_size.trailing_zeros();

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
        unsafe { cu_mem_alloc(&mut d_payload, n as usize * 4) },
        "cuMemAlloc payload",
    );
    // Zero the table (all slots empty) via HtoD of a zeroed buffer.
    let zeros = vec![0u64; table_size as usize];
    check(
        unsafe {
            cu_memcpy_htod(
                d_table,
                zeros.as_ptr() as *const c_void,
                table_size as usize * 8,
            )
        },
        "HtoD table zero",
    );

    let ptx = CString::new(PTX_SRC).unwrap();
    let entry = CString::new("gpu_db_index_insert").unwrap();
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

    let mut a_table = d_table;
    let mut a_pay = d_payload;
    let mut a_tmask = table_mask;
    let mut a_shift = hash_shift;
    let mut a_n = n;
    let mut params: [*mut c_void; 5] = [
        ptr::addr_of_mut!(a_table) as *mut c_void,
        ptr::addr_of_mut!(a_pay) as *mut c_void,
        ptr::addr_of_mut!(a_tmask) as *mut c_void,
        ptr::addr_of_mut!(a_shift) as *mut c_void,
        ptr::addr_of_mut!(a_n) as *mut c_void,
    ];

    let started = Instant::now();
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
    check(
        unsafe { cu_stream_synchronize(stream) },
        "cuStreamSynchronize",
    );
    let elapsed = started.elapsed();

    // Verify: DtoH the table, confirm EVERY key 0..n landed exactly once at the right (key, row).
    let mut table = vec![0u64; table_size as usize];
    check(
        unsafe {
            cu_memcpy_dtoh(
                table.as_mut_ptr() as *mut c_void,
                d_table,
                table_size as usize * 8,
            )
        },
        "DtoH table",
    );
    let mut seen = vec![false; n as usize];
    let mut count = 0u32;
    for &entry in &table {
        if entry != 0 {
            count += 1;
            let key = (entry >> 32) as u32;
            let row = (entry & 0xffff_ffff) as u32 - 1;
            assert!(key < n, "entry key {key} out of range");
            assert_eq!(row, key, "entry (key {key}) maps to wrong row {row}");
            assert!(!seen[key as usize], "key {key} inserted more than once");
            seen[key as usize] = true;
        }
    }
    assert_eq!(
        count, n,
        "expected {n} entries, found {count} (lost inserts)"
    );
    assert!(seen.iter().all(|&s| s), "some key missing from the index");

    let throughput = n as f64 / elapsed.as_secs_f64();
    println!(
        "# wave write probe 1: CONCURRENT lock-free index INSERT  n={n} threads={} ({blocks}x256)",
        blocks * 256
    );
    println!("  correctness: all {n} keys inserted exactly once at the right (key,row) = VERIFIED");
    println!(
        "  inserted {n} rows (payload append + CAS index insert) in {elapsed:?}  ->  {throughput:.0} inserts/s"
    );

    check(unsafe { cu_stream_destroy(stream) }, "cuStreamDestroy");
    check(unsafe { cu_mem_free(d_table) }, "free table");
    check(unsafe { cu_mem_free(d_payload) }, "free payload");
    check(
        unsafe { cu_ctx_release(device) },
        "cuDevicePrimaryCtxRelease",
    );
    println!("OK: concurrent lock-free index insert correct, clean exit");
}
