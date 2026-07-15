//! R3-001 build-only physical representation probe.
//!
//! Compares resident-input GPU mutation mechanics after removing the common WAL/FUA floor:
//! A = append a full new image, stamp the old version dead, publish latest-head coordinates;
//! B = copy the old dense latest image and visibility interval to undo, overwrite latest under a
//! seqlock, publish indexes.
//! This is decision evidence only. It does not migrate or serve engine state.
//!
//! Run (never `--gpu-reset`):
//! `timeout 300 cargo run --release -p gpu_db_execution --example write_path_candidate_ab`

use std::ffi::{c_void, CString};
use std::time::Instant;

use libloading::{Library, Symbol};

const ROWS: u32 = 1_000_000;
const MAX_BATCH: u32 = 4096;
const MAX_COLS: u32 = 32;
const MAX_FANOUT: u32 = 6;

const PTX: &str = r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry candidate_a_append(
    .param .u64 new_values,
    .param .u64 slots,
    .param .u64 appended,
    .param .u64 deleted_by,
    .param .u64 appended_deleted_by,
    .param .u64 created_by,
    .param .u64 row_ids,
    .param .u64 index_heads,
    .param .u64 index_history,
    .param .u32 rows,
    .param .u32 input_stride,
    .param .u32 n,
    .param .u32 cols,
    .param .u32 fanout,
    .param .u64 commit_seq
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<34>;

    ld.param.u64 %rd1, [new_values];
    ld.param.u64 %rd2, [slots];
    ld.param.u64 %rd3, [appended];
    ld.param.u64 %rd4, [deleted_by];
    ld.param.u64 %rd29, [appended_deleted_by];
    ld.param.u64 %rd5, [created_by];
    ld.param.u64 %rd6, [row_ids];
    ld.param.u64 %rd7, [index_heads];
    ld.param.u64 %rd8, [index_history];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [input_stride];
    ld.param.u32 %r3, [n];
    ld.param.u32 %r4, [cols];
    ld.param.u32 %r5, [fanout];
    ld.param.u64 %rd9, [commit_seq];

    mov.u32 %r6, %tid.x;
    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mad.lo.u32 %r9, %r7, %r8, %r6;
    setp.ge.u32 %p1, %r9, %r3;
    @%p1 bra A_DONE;

    mul.wide.u32 %rd32, %r9, 4;
    add.u64 %rd10, %rd2, %rd32;
    ld.global.u32 %r10, [%rd10];

    mov.u32 %r11, 0;
A_COL:
    setp.ge.u32 %p2, %r11, %r4;
    @%p2 bra A_META;
    mad.lo.u32 %r12, %r11, %r2, %r9;
    mul.wide.u32 %rd11, %r12, 4;
    add.u64 %rd12, %rd1, %rd11;
    ld.global.u32 %r13, [%rd12];
    mad.lo.u32 %r14, %r11, %r2, %r9;
    mul.wide.u32 %rd13, %r14, 4;
    add.u64 %rd14, %rd3, %rd13;
    st.global.u32 [%rd14], %r13;
    add.u32 %r11, %r11, 1;
    bra A_COL;

A_META:
    mul.wide.u32 %rd15, %r10, 8;
    add.u64 %rd16, %rd4, %rd15;
    st.global.u64 [%rd16], %rd9;
    mul.wide.u32 %rd17, %r9, 8;
    add.u64 %rd18, %rd5, %rd17;
    st.global.u64 [%rd18], %rd9;
    add.u64 %rd19, %rd6, %rd17;
    cvt.u64.u32 %rd20, %r10;
    st.global.u64 [%rd19], %rd20;
    add.u64 %rd30, %rd29, %rd17;
    mov.u64 %rd31, 0xffffffffffffffff;
    st.global.u64 [%rd30], %rd31;

    mov.u32 %r11, 0;
A_INDEX:
    setp.ge.u32 %p2, %r11, %r5;
    @%p2 bra A_DONE;
    mad.lo.u32 %r12, %r11, %r1, %r10;
    mul.wide.u32 %rd21, %r12, 8;
    add.u64 %rd22, %rd7, %rd21;
    cvt.u64.u32 %rd23, %r9;
    add.u64 %rd23, %rd23, 1;
    st.global.u64 [%rd22], %rd23;
    mad.lo.u32 %r13, %r11, %r2, %r9;
    mul.wide.u32 %rd24, %r13, 32;
    add.u64 %rd25, %rd8, %rd24;
    cvt.u64.u32 %rd26, %r10;
    st.global.u64 [%rd25], %rd26;
    st.global.u64 [%rd25+8], %rd23;
    st.global.u64 [%rd25+16], %rd9;
    st.global.u64 [%rd25+24], %rd31;
    add.u32 %r11, %r11, 1;
    bra A_INDEX;
A_DONE:
    ret;
}

.visible .entry candidate_b_dense_undo(
    .param .u64 latest,
    .param .u64 new_values,
    .param .u64 slots,
    .param .u64 undo,
    .param .u64 undo_created,
    .param .u64 undo_deleted,
    .param .u64 undo_row_ids,
    .param .u64 row_epochs,
    .param .u64 latest_created,
    .param .u64 latest_deleted,
    .param .u64 index_heads,
    .param .u64 index_history,
    .param .u32 rows,
    .param .u32 input_stride,
    .param .u32 n,
    .param .u32 cols,
    .param .u32 fanout,
    .param .u64 commit_seq
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<17>;
    .reg .b64 %rd<44>;

    ld.param.u64 %rd1, [latest];
    ld.param.u64 %rd2, [new_values];
    ld.param.u64 %rd3, [slots];
    ld.param.u64 %rd4, [undo];
    ld.param.u64 %rd5, [undo_created];
    ld.param.u64 %rd35, [undo_deleted];
    ld.param.u64 %rd6, [undo_row_ids];
    ld.param.u64 %rd7, [row_epochs];
    ld.param.u64 %rd33, [latest_created];
    ld.param.u64 %rd34, [latest_deleted];
    ld.param.u64 %rd8, [index_heads];
    ld.param.u64 %rd29, [index_history];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [input_stride];
    ld.param.u32 %r3, [n];
    ld.param.u32 %r4, [cols];
    ld.param.u32 %r5, [fanout];
    ld.param.u64 %rd9, [commit_seq];

    mov.u32 %r6, %tid.x;
    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mad.lo.u32 %r9, %r7, %r8, %r6;
    setp.ge.u32 %p1, %r9, %r3;
    @%p1 bra B_DONE;

    mul.wide.u32 %rd10, %r9, 4;
    add.u64 %rd11, %rd3, %rd10;
    ld.global.u32 %r10, [%rd11];
    mul.wide.u32 %rd12, %r10, 8;
    add.u64 %rd13, %rd7, %rd12;
    ld.global.u64 %rd14, [%rd13];
    add.u64 %rd15, %rd14, 1;
    st.global.u64 [%rd13], %rd15;
    membar.gl;

    mov.u32 %r11, 0;
B_COL:
    setp.ge.u32 %p2, %r11, %r4;
    @%p2 bra B_META;
    mad.lo.u32 %r12, %r11, %r1, %r10;
    mul.wide.u32 %rd16, %r12, 4;
    add.u64 %rd17, %rd1, %rd16;
    ld.global.u32 %r13, [%rd17];
    mad.lo.u32 %r14, %r11, %r2, %r9;
    mul.wide.u32 %rd18, %r14, 4;
    add.u64 %rd19, %rd4, %rd18;
    st.global.u32 [%rd19], %r13;
    add.u64 %rd20, %rd2, %rd18;
    ld.global.u32 %r15, [%rd20];
    st.global.u32 [%rd17], %r15;
    add.u32 %r11, %r11, 1;
    bra B_COL;

B_META:
    mul.wide.u32 %rd21, %r9, 8;
    add.u64 %rd36, %rd33, %rd12;
    ld.global.u64 %rd37, [%rd36];
    add.u64 %rd38, %rd34, %rd12;
    add.u64 %rd22, %rd5, %rd21;
    st.global.u64 [%rd22], %rd37;
    add.u64 %rd40, %rd35, %rd21;
    st.global.u64 [%rd40], %rd9;
    add.u64 %rd23, %rd6, %rd21;
    cvt.u64.u32 %rd24, %r10;
    st.global.u64 [%rd23], %rd24;
    st.global.u64 [%rd36], %rd9;
    mov.u64 %rd41, 0xffffffffffffffff;
    st.global.u64 [%rd38], %rd41;
    membar.gl;
    add.u64 %rd25, %rd14, 2;
    st.global.u64 [%rd13], %rd25;

    mov.u32 %r11, 0;
B_INDEX:
    setp.ge.u32 %p2, %r11, %r5;
    @%p2 bra B_DONE;
    mad.lo.u32 %r12, %r11, %r1, %r10;
    mul.wide.u32 %rd26, %r12, 8;
    add.u64 %rd27, %rd8, %rd26;
    cvt.u64.u32 %rd28, %r10;
    add.u64 %rd28, %rd28, 1;
    st.global.u64 [%rd27], %rd28;
    mad.lo.u32 %r13, %r11, %r2, %r9;
    mul.wide.u32 %rd30, %r13, 32;
    add.u64 %rd31, %rd29, %rd30;
    cvt.u64.u32 %rd32, %r10;
    st.global.u64 [%rd31], %rd32;
    st.global.u64 [%rd31+8], %rd28;
    st.global.u64 [%rd31+16], %rd9;
    st.global.u64 [%rd31+24], %rd41;
    add.u32 %r11, %r11, 1;
    bra B_INDEX;
B_DONE:
    ret;
}
"#;

type Fn1U32 = unsafe extern "C" fn(u32) -> i32;
type Fn1Ptr = unsafe extern "C" fn(*mut c_void) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuCtxRetain = unsafe extern "C" fn(*mut *mut c_void, i32) -> i32;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFree = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
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

fn sym<T>(lib: &'static Library, names: &[&[u8]]) -> Symbol<'static, T> {
    for name in names {
        if let Ok(symbol) = unsafe { lib.get::<T>(name) } {
            return symbol;
        }
    }
    panic!("CUDA symbol not found: {names:?}");
}

fn check(code: i32, what: &str) {
    assert_eq!(code, 0, "CUDA call failed ({code}): {what}");
}

fn percentile(samples: &mut [u64], p: f64) -> u64 {
    samples.sort_unstable();
    let rank = ((samples.len() as f64) * p).ceil() as usize;
    samples[rank.clamp(1, samples.len()) - 1]
}

struct DeviceBuffer {
    ptr: u64,
    bytes: usize,
    free: CuMemFree,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        check(unsafe { (self.free)(self.ptr) }, "cuMemFree");
    }
}

fn alloc(cu_mem_alloc: CuMemAlloc, cu_mem_free: CuMemFree, bytes: usize) -> DeviceBuffer {
    let mut ptr = 0;
    check(
        unsafe { cu_mem_alloc(&mut ptr, bytes.max(1)) },
        "cuMemAlloc",
    );
    DeviceBuffer {
        ptr,
        bytes,
        free: cu_mem_free,
    }
}

fn main() {
    let lib: &'static Library = match unsafe { Library::new("libcuda.so.1") }
        .or_else(|_| unsafe { Library::new("libcuda.so") })
    {
        Ok(lib) => Box::leak(Box::new(lib)),
        Err(err) => {
            println!("no CUDA ({err}) — skipping");
            return;
        }
    };

    let cu_init: Symbol<Fn1U32> = sym(lib, &[b"cuInit\0"]);
    let cu_device_get: Symbol<CuDeviceGet> = sym(lib, &[b"cuDeviceGet\0"]);
    let cu_ctx_retain: Symbol<CuCtxRetain> = sym(lib, &[b"cuDevicePrimaryCtxRetain\0"]);
    let cu_ctx_set_current: Symbol<Fn1Ptr> = sym(lib, &[b"cuCtxSetCurrent\0"]);
    let cu_mem_alloc: Symbol<CuMemAlloc> = sym(lib, &[b"cuMemAlloc_v2\0", b"cuMemAlloc\0"]);
    let cu_mem_free: Symbol<CuMemFree> = sym(lib, &[b"cuMemFree_v2\0", b"cuMemFree\0"]);
    let cu_memcpy_htod: Symbol<CuMemcpyHtoD> = sym(lib, &[b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0"]);
    let cu_memcpy_dtoh: Symbol<CuMemcpyDtoH> = sym(lib, &[b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0"]);
    let cu_memset_d8: Symbol<CuMemsetD8> = sym(lib, &[b"cuMemsetD8_v2\0", b"cuMemsetD8\0"]);
    let cu_module_load: Symbol<CuModuleLoadData> = sym(lib, &[b"cuModuleLoadData\0"]);
    let cu_module_get: Symbol<CuModuleGetFunction> = sym(lib, &[b"cuModuleGetFunction\0"]);
    let cu_stream_create: Symbol<CuStreamCreate> = sym(lib, &[b"cuStreamCreate\0"]);
    let cu_stream_sync: Symbol<Fn1Ptr> = sym(lib, &[b"cuStreamSynchronize\0"]);
    let cu_launch: Symbol<CuLaunchKernel> = sym(lib, &[b"cuLaunchKernel\0"]);

    check(unsafe { cu_init(0) }, "cuInit");
    let mut device = 0;
    check(unsafe { cu_device_get(&mut device, 0) }, "cuDeviceGet");
    let mut context = std::ptr::null_mut();
    check(
        unsafe { cu_ctx_retain(&mut context, device) },
        "cuDevicePrimaryCtxRetain",
    );
    check(unsafe { cu_ctx_set_current(context) }, "cuCtxSetCurrent");

    let ptx = CString::new(PTX).expect("PTX CString");
    let mut module = std::ptr::null_mut();
    check(
        unsafe { cu_module_load(&mut module, ptx.as_ptr().cast()) },
        "cuModuleLoadData",
    );
    let mut candidate_a = std::ptr::null_mut();
    let mut candidate_b = std::ptr::null_mut();
    check(
        unsafe {
            cu_module_get(
                &mut candidate_a,
                module,
                CString::new("candidate_a_append").unwrap().as_ptr(),
            )
        },
        "get candidate A",
    );
    check(
        unsafe {
            cu_module_get(
                &mut candidate_b,
                module,
                CString::new("candidate_b_dense_undo").unwrap().as_ptr(),
            )
        },
        "get candidate B",
    );
    let mut stream = std::ptr::null_mut();
    check(
        unsafe { cu_stream_create(&mut stream, 0) },
        "cuStreamCreate",
    );

    let latest = alloc(
        *cu_mem_alloc,
        *cu_mem_free,
        ROWS as usize * MAX_COLS as usize * 4,
    );
    let new_values = alloc(
        *cu_mem_alloc,
        *cu_mem_free,
        MAX_BATCH as usize * MAX_COLS as usize * 4,
    );
    let slots = alloc(*cu_mem_alloc, *cu_mem_free, MAX_BATCH as usize * 4);
    let appended = alloc(
        *cu_mem_alloc,
        *cu_mem_free,
        MAX_BATCH as usize * MAX_COLS as usize * 4,
    );
    let undo = alloc(
        *cu_mem_alloc,
        *cu_mem_free,
        MAX_BATCH as usize * MAX_COLS as usize * 4,
    );
    let deleted = alloc(*cu_mem_alloc, *cu_mem_free, ROWS as usize * 8);
    let appended_deleted = alloc(*cu_mem_alloc, *cu_mem_free, MAX_BATCH as usize * 8);
    let appended_created = alloc(*cu_mem_alloc, *cu_mem_free, MAX_BATCH as usize * 8);
    let undo_created = alloc(*cu_mem_alloc, *cu_mem_free, MAX_BATCH as usize * 8);
    let undo_deleted = alloc(*cu_mem_alloc, *cu_mem_free, MAX_BATCH as usize * 8);
    let row_ids = alloc(*cu_mem_alloc, *cu_mem_free, MAX_BATCH as usize * 8);
    let epochs = alloc(*cu_mem_alloc, *cu_mem_free, ROWS as usize * 8);
    let latest_created = alloc(*cu_mem_alloc, *cu_mem_free, ROWS as usize * 8);
    let latest_deleted = alloc(*cu_mem_alloc, *cu_mem_free, ROWS as usize * 8);
    let indexes = alloc(
        *cu_mem_alloc,
        *cu_mem_free,
        ROWS as usize * MAX_FANOUT as usize * 8,
    );
    let index_history = alloc(
        *cu_mem_alloc,
        *cu_mem_free,
        MAX_BATCH as usize * MAX_FANOUT as usize * 32,
    );

    for buffer in [
        &latest,
        &appended,
        &undo,
        &deleted,
        &appended_deleted,
        &appended_created,
        &undo_created,
        &undo_deleted,
        &row_ids,
        &epochs,
        &latest_created,
        &latest_deleted,
        &indexes,
        &index_history,
    ] {
        check(
            unsafe { cu_memset_d8(buffer.ptr, 0, buffer.bytes) },
            "zero device buffer",
        );
    }
    for buffer in [&appended_deleted, &latest_deleted] {
        check(
            unsafe { cu_memset_d8(buffer.ptr, 0xff, buffer.bytes) },
            "initialize live deleted-by buffer",
        );
    }
    let host_slots: Vec<u32> = (0..MAX_BATCH).map(|i| i * 211 % ROWS).collect();
    let mut host_values = vec![0_u32; MAX_BATCH as usize * MAX_COLS as usize];
    for col in 0..MAX_COLS as usize {
        for row in 0..MAX_BATCH as usize {
            host_values[col * MAX_BATCH as usize + row] = (col as u32 + 1)
                .wrapping_mul(1_000_003)
                .wrapping_add(row as u32);
        }
    }
    check(
        unsafe { cu_memcpy_htod(slots.ptr, host_slots.as_ptr().cast(), slots.bytes) },
        "upload slots",
    );
    check(
        unsafe {
            cu_memcpy_htod(
                new_values.ptr,
                host_values.as_ptr().cast(),
                new_values.bytes,
            )
        },
        "upload new values",
    );

    println!("R3-001 GPU physical representation A/B (resident inputs; wall launch+sync)");
    println!("rows={ROWS} max-batch={MAX_BATCH}; A=append/tombstone, B=dense-latest+undo+seqlock");
    println!(
        "{:>5} {:>3} {:>5} | {:>10} {:>10} | {:>10} {:>10} | {:>7}",
        "width", "idx", "batch", "A p50 us", "A p99.9", "B p50 us", "B p99.9", "B/A"
    );

    for &cols in &[2_u32, 8, 32] {
        for &fanout in &[1_u32, 3, 6] {
            for &batch in &[1_u32, 256, 4096] {
                let iterations = if batch == 1 {
                    2000
                } else if batch == 256 {
                    1000
                } else {
                    300
                };
                let blocks = batch.div_ceil(256);
                let mut a_commit_seq = 41_u64;
                let mut launch_a = || {
                    a_commit_seq = a_commit_seq.wrapping_add(1);
                    let mut p_new = new_values.ptr;
                    let mut p_slots = slots.ptr;
                    let mut p_appended = appended.ptr;
                    let mut p_deleted = deleted.ptr;
                    let mut p_appended_deleted = appended_deleted.ptr;
                    let mut p_created = appended_created.ptr;
                    let mut p_row_ids = row_ids.ptr;
                    let mut p_indexes = indexes.ptr;
                    let mut p_index_history = index_history.ptr;
                    let mut p_rows = ROWS;
                    let mut p_stride = MAX_BATCH;
                    let mut p_n = batch;
                    let mut p_cols = cols;
                    let mut p_fanout = fanout;
                    let mut p_seq = a_commit_seq;
                    let mut args: [*mut c_void; 15] = [
                        (&mut p_new as *mut u64).cast(),
                        (&mut p_slots as *mut u64).cast(),
                        (&mut p_appended as *mut u64).cast(),
                        (&mut p_deleted as *mut u64).cast(),
                        (&mut p_appended_deleted as *mut u64).cast(),
                        (&mut p_created as *mut u64).cast(),
                        (&mut p_row_ids as *mut u64).cast(),
                        (&mut p_indexes as *mut u64).cast(),
                        (&mut p_index_history as *mut u64).cast(),
                        (&mut p_rows as *mut u32).cast(),
                        (&mut p_stride as *mut u32).cast(),
                        (&mut p_n as *mut u32).cast(),
                        (&mut p_cols as *mut u32).cast(),
                        (&mut p_fanout as *mut u32).cast(),
                        (&mut p_seq as *mut u64).cast(),
                    ];
                    check(
                        unsafe {
                            cu_launch(
                                candidate_a,
                                blocks,
                                1,
                                1,
                                256,
                                1,
                                1,
                                0,
                                stream,
                                args.as_mut_ptr(),
                                std::ptr::null_mut(),
                            )
                        },
                        "launch candidate A",
                    );
                    check(unsafe { cu_stream_sync(stream) }, "sync candidate A");
                };
                let mut b_commit_seq = 41_u64;
                let mut launch_b = || {
                    b_commit_seq = b_commit_seq.wrapping_add(1);
                    let mut p_latest = latest.ptr;
                    let mut p_new = new_values.ptr;
                    let mut p_slots = slots.ptr;
                    let mut p_undo = undo.ptr;
                    let mut p_created = undo_created.ptr;
                    let mut p_undo_deleted = undo_deleted.ptr;
                    let mut p_row_ids = row_ids.ptr;
                    let mut p_epochs = epochs.ptr;
                    let mut p_latest_created = latest_created.ptr;
                    let mut p_latest_deleted = latest_deleted.ptr;
                    let mut p_indexes = indexes.ptr;
                    let mut p_index_history = index_history.ptr;
                    let mut p_rows = ROWS;
                    let mut p_stride = MAX_BATCH;
                    let mut p_n = batch;
                    let mut p_cols = cols;
                    let mut p_fanout = fanout;
                    let mut p_seq = b_commit_seq;
                    let mut args: [*mut c_void; 18] = [
                        (&mut p_latest as *mut u64).cast(),
                        (&mut p_new as *mut u64).cast(),
                        (&mut p_slots as *mut u64).cast(),
                        (&mut p_undo as *mut u64).cast(),
                        (&mut p_created as *mut u64).cast(),
                        (&mut p_undo_deleted as *mut u64).cast(),
                        (&mut p_row_ids as *mut u64).cast(),
                        (&mut p_epochs as *mut u64).cast(),
                        (&mut p_latest_created as *mut u64).cast(),
                        (&mut p_latest_deleted as *mut u64).cast(),
                        (&mut p_indexes as *mut u64).cast(),
                        (&mut p_index_history as *mut u64).cast(),
                        (&mut p_rows as *mut u32).cast(),
                        (&mut p_stride as *mut u32).cast(),
                        (&mut p_n as *mut u32).cast(),
                        (&mut p_cols as *mut u32).cast(),
                        (&mut p_fanout as *mut u32).cast(),
                        (&mut p_seq as *mut u64).cast(),
                    ];
                    check(
                        unsafe {
                            cu_launch(
                                candidate_b,
                                blocks,
                                1,
                                1,
                                256,
                                1,
                                1,
                                0,
                                stream,
                                args.as_mut_ptr(),
                                std::ptr::null_mut(),
                            )
                        },
                        "launch candidate B",
                    );
                    check(unsafe { cu_stream_sync(stream) }, "sync candidate B");
                };

                for _ in 0..20 {
                    launch_a();
                    launch_b();
                }
                let mut a = Vec::with_capacity(iterations);
                let mut b = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    let start = Instant::now();
                    launch_a();
                    a.push(start.elapsed().as_nanos() as u64);
                    let start = Instant::now();
                    launch_b();
                    b.push(start.elapsed().as_nanos() as u64);
                }
                let a50 = percentile(&mut a.clone(), 0.50) as f64 / 1000.0;
                let a999 = percentile(&mut a, 0.999) as f64 / 1000.0;
                let b50 = percentile(&mut b.clone(), 0.50) as f64 / 1000.0;
                let b999 = percentile(&mut b, 0.999) as f64 / 1000.0;
                println!(
                    "{:>5} {:>3} {:>5} | {:>10.2} {:>10.2} | {:>10.2} {:>10.2} | {:>7.2}",
                    cols * 4,
                    fanout,
                    batch,
                    a50,
                    a999,
                    b50,
                    b999,
                    b50 / a50,
                );
            }
        }
    }

    let mut a_value = 0_u32;
    let mut b_value = 0_u32;
    check(
        unsafe { cu_memcpy_dtoh((&mut a_value as *mut u32).cast(), appended.ptr, 4) },
        "read candidate A",
    );
    let latest_slot = u64::from(host_slots[0]) * 4;
    check(
        unsafe {
            cu_memcpy_dtoh(
                (&mut b_value as *mut u32).cast(),
                latest.ptr + latest_slot,
                4,
            )
        },
        "read candidate B",
    );
    assert_eq!(a_value, host_values[0]);
    assert_eq!(b_value, host_values[0]);

    let read_u64 = |ptr: u64, what: &str| {
        let mut value = 0_u64;
        check(
            unsafe { cu_memcpy_dtoh((&mut value as *mut u64).cast(), ptr, 8) },
            what,
        );
        value
    };
    let slot_meta = u64::from(host_slots[0]) * 8;
    let a_created = read_u64(appended_created.ptr, "read candidate A created-by");
    let a_new_deleted = read_u64(appended_deleted.ptr, "read candidate A new deleted-by");
    let a_old_deleted = read_u64(deleted.ptr + slot_meta, "read candidate A old deleted-by");
    let b_created = read_u64(
        latest_created.ptr + slot_meta,
        "read candidate B latest created-by",
    );
    let b_deleted = read_u64(
        latest_deleted.ptr + slot_meta,
        "read candidate B latest deleted-by",
    );
    let b_epoch = read_u64(epochs.ptr + slot_meta, "read candidate B seqlock");
    let b_undo_created = read_u64(undo_created.ptr, "read candidate B undo created-by");
    let b_undo_deleted = read_u64(undo_deleted.ptr, "read candidate B undo deleted-by");
    let history_deleted = read_u64(
        index_history.ptr + 24,
        "read candidate B index-history deleted-by",
    );
    assert_eq!(a_created, a_old_deleted);
    assert_eq!(a_new_deleted, u64::MAX);
    assert_eq!(b_created, a_created);
    assert_eq!(b_deleted, u64::MAX);
    assert_eq!(b_epoch & 1, 0);
    assert!(b_undo_created < b_created);
    assert_eq!(b_undo_deleted, b_created);
    assert_eq!(history_deleted, u64::MAX);
    let old_snapshot = b_undo_created;
    assert!(b_undo_created <= old_snapshot && old_snapshot < b_undo_deleted);
    assert!(!(b_created <= old_snapshot && old_snapshot < b_deleted));
    let current_snapshot = b_created;
    assert!(!(b_undo_created <= current_snapshot && current_snapshot < b_undo_deleted));
    assert!(b_created <= current_snapshot && current_snapshot < b_deleted);

    println!("\nExact logical representation bytes (excludes allocator rounding/scratch):");
    for &width in &[8_u64, 32, 128] {
        let a_history = width + 24;
        let b_history = width + 24;
        println!(
            "  width={width:>3}B: A history/update={a_history:>3}B; B undo/update={b_history:>3}B; \
             both add 32B/index-history record per maintained index/version"
        );
    }
    println!(
        "  Both allocate one 8B latest-head cell per logical row/index in this probe. A's newest \
         append carries row-id/created/deleted metadata; B's dense latest uses an implicit row slot \
         plus created/deleted/seqlock metadata. Both are width+24B per current row."
    );
}
