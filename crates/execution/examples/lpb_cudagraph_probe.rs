//! R2.2c spike — does CUDA-graph replay close most of the wave's launch-overhead win over lpb?
//!
//! The wave's single-coalescer A/B win (R2.2b-3) is mostly LAUNCH-OVERHEAD avoidance: a launch-per-batch
//! (lpb) point-read pays ~27us p50 at batch=1 (CPU-side launch + memcpy submission + sync round-trip),
//! while the persistent wave kernel pays ~10us (no per-batch launch). If a CUDA-GRAPH-captured lpb launch
//! replays in ~2-5us, graphs close most of that gap with NONE of the persistent-kernel complexity (no
//! at-most-one-resident, no coexistence, no watchdog, no SM tax) -- and the wave's unique edge shrinks to
//! the multi-PRODUCER concurrent regime only. This spike measures exactly that.
//!
//! The launch-overhead saving graphs provide is KERNEL-AGNOSTIC (it is the CPU-side submission cost, added
//! to both paths' kernel-execution time equally), so we use a trivial GATHER kernel with the SAME op shape
//! as the lpb index probe: HtoD(needles) -> kernel gathers table[needle & mask] -> DtoH(results). We time,
//! per batch size, the per-batch latency of:
//!   DIRECT : write needles -> cuMemcpyHtoDAsync -> cuLaunchKernel -> cuMemcpyDtoHAsync -> cuStreamSynchronize
//!   GRAPH  : write needles -> cuGraphLaunch (replays the captured HtoD+launch+DtoH) -> cuStreamSynchronize
//! and assert GRAPH results == DIRECT results. The DELTA (direct - graph) is the submission overhead graphs
//! eliminate; compare GRAPH p50 to the wave's ~10us to see if graphs beat the wave on the single-coalescer path.
//!
//! Run (RTX box; never `--gpu-reset`, always under `timeout`):
//!   timeout 180 cargo run --release --example lpb_cudagraph_probe -p gpu_db_execution

use std::ffi::{c_void, CString};
use std::time::Instant;

use libloading::{Library, Symbol};

const PTX_SRC: &str = r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry gather(
    .param .u64 needles,
    .param .u64 table,
    .param .u64 out,
    .param .u32 n,
    .param .u32 mask
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [needles];
    ld.param.u64 %rd2, [table];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r2, [mask];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra $L_end;
    mul.wide.u32 %rd4, %r6, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.u32 %r7, [%rd5];
    and.b32 %r7, %r7, %r2;
    mul.wide.u32 %rd6, %r7, 4;
    add.u64 %rd7, %rd2, %rd6;
    ld.global.u32 %r7, [%rd7];
    mul.wide.u32 %rd8, %r6, 4;
    add.u64 %rd9, %rd3, %rd8;
    st.global.u32 [%rd9], %r7;
$L_end:
    ret;
}
"#;

const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;

type Fn1U32 = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuCtxRetain = unsafe extern "C" fn(*mut *mut c_void, i32) -> i32;
type Fn1Ptr = unsafe extern "C" fn(*mut c_void) -> i32;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyHtoDAsync = unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> i32;
type CuMemcpyDtoHAsync = unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> i32;
type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
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
type CuStreamBeginCapture = unsafe extern "C" fn(*mut c_void, u32) -> i32;
type CuStreamEndCapture = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32;
type CuGraphInstantiate = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, u64) -> i32;
type CuGraphLaunch = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;

fn sym<T>(lib: &Library, names: &[&[u8]]) -> Symbol<'static, T> {
    for name in names {
        if let Ok(symbol) = unsafe { lib.get::<T>(name) } {
            return unsafe { std::mem::transmute::<Symbol<'_, T>, Symbol<'static, T>>(symbol) };
        }
    }
    panic!("CUDA symbol not found: {names:?}");
}

fn check(code: i32, what: &str) {
    assert_eq!(code, 0, "CUDA call failed ({code}): {what}");
}

fn p50(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    if v.is_empty() {
        0
    } else {
        v[v.len() / 2]
    }
}

fn main() {
    let lib: &'static Library = match unsafe { Library::new("libcuda.so.1") }
        .or_else(|_| unsafe { Library::new("libcuda.so") })
    {
        Ok(l) => Box::leak(Box::new(l)),
        Err(e) => {
            println!("no CUDA ({e}) — skipping");
            return;
        }
    };

    let cu_init: Symbol<Fn1U32> = sym(lib, &[b"cuInit\0"]);
    let cu_device_get: Symbol<CuDeviceGet> = sym(lib, &[b"cuDeviceGet\0"]);
    let cu_ctx_retain: Symbol<CuCtxRetain> = sym(lib, &[b"cuDevicePrimaryCtxRetain\0"]);
    let cu_ctx_set_current: Symbol<Fn1Ptr> = sym(lib, &[b"cuCtxSetCurrent\0"]);
    let cu_mem_alloc: Symbol<CuMemAlloc> = sym(lib, &[b"cuMemAlloc_v2\0", b"cuMemAlloc\0"]);
    let cu_memcpy_htod: Symbol<CuMemcpyHtoD> = sym(lib, &[b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0"]);
    let cu_memcpy_htod_async: Symbol<CuMemcpyHtoDAsync> =
        sym(lib, &[b"cuMemcpyHtoDAsync_v2\0", b"cuMemcpyHtoDAsync\0"]);
    let cu_memcpy_dtoh_async: Symbol<CuMemcpyDtoHAsync> =
        sym(lib, &[b"cuMemcpyDtoHAsync_v2\0", b"cuMemcpyDtoHAsync\0"]);
    let cu_mem_host_alloc: Symbol<CuMemHostAlloc> = sym(lib, &[b"cuMemHostAlloc\0"]);
    let cu_module_load_data: Symbol<CuModuleLoadData> = sym(lib, &[b"cuModuleLoadData\0"]);
    let cu_module_get_function: Symbol<CuModuleGetFunction> = sym(lib, &[b"cuModuleGetFunction\0"]);
    let cu_stream_create: Symbol<CuStreamCreate> = sym(lib, &[b"cuStreamCreate\0"]);
    let cu_stream_synchronize: Symbol<Fn1Ptr> = sym(lib, &[b"cuStreamSynchronize\0"]);
    let cu_launch_kernel: Symbol<CuLaunchKernel> = sym(lib, &[b"cuLaunchKernel\0"]);
    let cu_stream_begin_capture: Symbol<CuStreamBeginCapture> =
        sym(lib, &[b"cuStreamBeginCapture_v2\0", b"cuStreamBeginCapture\0"]);
    let cu_stream_end_capture: Symbol<CuStreamEndCapture> = sym(lib, &[b"cuStreamEndCapture\0"]);
    let cu_graph_instantiate: Symbol<CuGraphInstantiate> = sym(
        lib,
        &[
            b"cuGraphInstantiateWithFlags\0",
            b"cuGraphInstantiate_v2\0",
        ],
    );
    let cu_graph_launch: Symbol<CuGraphLaunch> = sym(lib, &[b"cuGraphLaunch\0"]);

    check(unsafe { cu_init(0) }, "cuInit");
    let mut dev: i32 = 0;
    check(unsafe { cu_device_get(&mut dev, 0) }, "cuDeviceGet");
    let mut ctx: *mut c_void = std::ptr::null_mut();
    check(unsafe { cu_ctx_retain(&mut ctx, dev) }, "cuDevicePrimaryCtxRetain");
    check(unsafe { cu_ctx_set_current(ctx) }, "cuCtxSetCurrent");

    // Resident table: table[i] = i*7 (so a gather result is verifiable).
    let table_size: u32 = 65536;
    let mask = table_size - 1;
    let table: Vec<i32> = (0..table_size as i32).map(|i| i.wrapping_mul(7)).collect();
    let mut table_dev: u64 = 0;
    check(unsafe { cu_mem_alloc(&mut table_dev, table_size as usize * 4) }, "cuMemAlloc table");
    check(
        unsafe { cu_memcpy_htod(table_dev, table.as_ptr().cast(), table_size as usize * 4) },
        "HtoD table",
    );

    let max_batch = 256usize;
    let mut needles_dev: u64 = 0;
    let mut out_dev: u64 = 0;
    check(unsafe { cu_mem_alloc(&mut needles_dev, max_batch * 4) }, "cuMemAlloc needles");
    check(unsafe { cu_mem_alloc(&mut out_dev, max_batch * 4) }, "cuMemAlloc out");
    // Pinned host staging (graph captures memcpy from/to these fixed addresses; we overwrite contents/read).
    let mut needles_host: *mut c_void = std::ptr::null_mut();
    let mut out_host: *mut c_void = std::ptr::null_mut();
    check(
        unsafe { cu_mem_host_alloc(&mut needles_host, max_batch * 4, CU_MEMHOSTALLOC_PORTABLE) },
        "host needles",
    );
    check(
        unsafe { cu_mem_host_alloc(&mut out_host, max_batch * 4, CU_MEMHOSTALLOC_PORTABLE) },
        "host out",
    );

    let ptx = CString::new(PTX_SRC).unwrap();
    let mut module: *mut c_void = std::ptr::null_mut();
    check(
        unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast()) },
        "cuModuleLoadData",
    );
    let fname = CString::new("gather").unwrap();
    let mut func: *mut c_void = std::ptr::null_mut();
    check(
        unsafe { cu_module_get_function(&mut func, module, fname.as_ptr()) },
        "cuModuleGetFunction",
    );
    let mut stream: *mut c_void = std::ptr::null_mut();
    check(unsafe { cu_stream_create(&mut stream, 0) }, "cuStreamCreate");

    let needles_host_p = needles_host as *mut i32;
    let out_host_p = out_host as *mut i32;
    let set_needles = |batch: usize, round: usize| {
        for k in 0..batch {
            unsafe { *needles_host_p.add(k) = ((round * batch + k) as u32 & mask) as i32 };
        }
    };
    let expected = |batch: usize| -> Vec<i32> {
        (0..batch)
            .map(|k| {
                let n = unsafe { *needles_host_p.add(k) } as u32 & mask;
                table[n as usize]
            })
            .collect()
    };
    let read_out = |batch: usize| -> Vec<i32> {
        (0..batch).map(|k| unsafe { *out_host_p.add(k) }).collect()
    };

    let iters = 2000usize;
    let warmup = 50usize;
    println!("# R2.2c CUDA-graph vs direct-launch spike (gather op, same shape as lpb point read)");
    println!("# DIRECT = HtoDAsync + launch + DtoHAsync + sync per batch;  GRAPH = cuGraphLaunch + sync");
    println!("# reference: wave single-flight p50 ~10us @batch1; lpb ~27us @batch1\n");
    println!(
        "  {:>6}  {:>14}  {:>14}  {:>12}  {:>16}",
        "batch", "DIRECT p50 us", "GRAPH p50 us", "delta us", "graph speedup"
    );

    for &batch in &[1usize, 8, 32, 256] {
        let n = batch as u32;
        let blocks = ((batch + 255) / 256) as u32;
        let tpb = if batch >= 256 { 256u32 } else { batch as u32 };

        // ---- DIRECT path ----
        let launch_direct = |round: usize| {
            set_needles(batch, round);
            let mut p_needles = needles_dev;
            let mut p_table = table_dev;
            let mut p_out = out_dev;
            let mut p_n = n;
            let mut p_mask = mask;
            let mut args: [*mut c_void; 5] = [
                (&mut p_needles as *mut u64).cast(),
                (&mut p_table as *mut u64).cast(),
                (&mut p_out as *mut u64).cast(),
                (&mut p_n as *mut u32).cast(),
                (&mut p_mask as *mut u32).cast(),
            ];
            unsafe {
                check(
                    cu_memcpy_htod_async(needles_dev, needles_host, batch * 4, stream),
                    "HtoDAsync needles",
                );
                check(
                    cu_launch_kernel(
                        func, blocks, 1, 1, tpb, 1, 1, 0, stream, args.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                    "launch",
                );
                check(
                    cu_memcpy_dtoh_async(out_host, out_dev, batch * 4, stream),
                    "DtoHAsync out",
                );
                check(cu_stream_synchronize(stream), "sync");
            }
        };
        for r in 0..warmup {
            launch_direct(r);
        }
        // correctness for the direct path
        set_needles(batch, 7);
        launch_direct(7);
        assert_eq!(read_out(batch), expected(batch), "direct gather wrong (batch={batch})");

        let mut direct = Vec::with_capacity(iters);
        for r in 0..iters {
            let t = Instant::now();
            launch_direct(r);
            direct.push(t.elapsed().as_micros());
        }

        // ---- GRAPH path: capture ONE batch's HtoDAsync+launch+DtoHAsync, instantiate, then replay ----
        let mut p_needles = needles_dev;
        let mut p_table = table_dev;
        let mut p_out = out_dev;
        let mut p_n = n;
        let mut p_mask = mask;
        let mut args: [*mut c_void; 5] = [
            (&mut p_needles as *mut u64).cast(),
            (&mut p_table as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_n as *mut u32).cast(),
            (&mut p_mask as *mut u32).cast(),
        ];
        let mut graph: *mut c_void = std::ptr::null_mut();
        let mut exec: *mut c_void = std::ptr::null_mut();
        unsafe {
            check(cu_stream_begin_capture(stream, 0), "begin capture"); // 0 = GLOBAL
            check(
                cu_memcpy_htod_async(needles_dev, needles_host, batch * 4, stream),
                "cap HtoD",
            );
            check(
                cu_launch_kernel(
                    func, blocks, 1, 1, tpb, 1, 1, 0, stream, args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cap launch",
            );
            check(
                cu_memcpy_dtoh_async(out_host, out_dev, batch * 4, stream),
                "cap DtoH",
            );
            check(cu_stream_end_capture(stream, &mut graph), "end capture");
            check(cu_graph_instantiate(&mut exec, graph, 0), "instantiate");
        }
        let launch_graph = |round: usize| {
            set_needles(batch, round);
            unsafe {
                check(cu_graph_launch(exec, stream), "graph launch");
                check(cu_stream_synchronize(stream), "graph sync");
            }
        };
        for r in 0..warmup {
            launch_graph(r);
        }
        set_needles(batch, 7);
        launch_graph(7);
        assert_eq!(read_out(batch), expected(batch), "graph gather wrong (batch={batch})");

        let mut graph_t = Vec::with_capacity(iters);
        for r in 0..iters {
            let t = Instant::now();
            launch_graph(r);
            graph_t.push(t.elapsed().as_micros());
        }

        let d = p50(direct);
        let g = p50(graph_t);
        let delta = d.saturating_sub(g);
        let speedup = if g > 0 { d as f64 / g as f64 } else { 0.0 };
        println!("  {batch:>6}  {d:>14}  {g:>14}  {delta:>12}  {speedup:>15.2}x");
    }

    println!("\n# If GRAPH p50 ~2-5us at batch1 (<< lpb ~27us, ~= or < wave ~10us) -> graphs obviate the wave");
    println!("# for the single-coalescer regime; the wave's only remaining edge is multi-producer (gate-2).");
}
