    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_primary_context_is_cached_and_loads_each_module_once() {
        // P2-M1 step 2 — the substrate's packaging contract, on real hardware:
        //   (a) the registry hands out the SAME primary context per GPU id (one shared
        //       context, not one per allocation/generation), and
        //   (b) a kernel's module is JIT-loaded ONCE and reused — running the migrated
        //       COUNT route many times adds at most one entry to the module cache (the
        //       load-once property the step-1 spike showed is the dominant concurrency
        //       win), not one per launch.
        use std::sync::Arc;

        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        // (a) registry caching: same Arc for the same GPU id.
        let ctx_a = gpu_primary_context(0).expect("primary context");
        let ctx_b = gpu_primary_context(0).expect("primary context");
        assert!(
            Arc::ptr_eq(&ctx_a, &ctx_b),
            "registry must hand out one shared primary context per GPU id"
        );

        // (b) load-once: run the migrated COUNT route several times and show its module is
        // JIT-loaded once and then reused, not reloaded per launch.
        //
        // The module cache is PROCESS-WIDE (one shared primary context per GPU id), so sibling
        // tests running in PARALLEL load their own kernels' modules into the same cache. Asserting
        // on the global `cached_module_count()` is therefore racy (a sibling loading a different
        // kernel between our before/after snapshots inflates the delta and fails the bound). We
        // instead assert against THIS route's specific module key (`gpu_db_resident_row_count`):
        // the cache is keyed by entry name, so this key can hold at most one entry regardless of
        // how many times we launch — and siblings' kernels touch other keys, so they cannot
        // perturb this assertion.
        const COUNT_ROUTE_MODULE: &CStr = c"gpu_db_resident_row_count";
        let row_count = 1234_u64;
        let resident = runtime
            .retain_device_memory_copy(0, &row_count.to_le_bytes())
            .expect("retain resident device memory");
        // The resident allocation must share the registry's context, not a private one.
        assert_eq!(
            resident.context(),
            ctx_a.context(),
            "the resident allocation must live in the registry's shared primary context"
        );
        for _ in 0..8 {
            assert_eq!(resident.count_rows_from_header().expect("count"), row_count);
        }
        // After repeated launches the route's module is present exactly once (keyed cache ⇒ a
        // single entry per key ⇒ loaded once and reused, not per launch).
        assert!(
            ctx_a.is_module_cached(COUNT_ROUTE_MODULE),
            "the COUNT route's module ({COUNT_ROUTE_MODULE:?}) must be JIT-loaded once and cached \
             for reuse, not reloaded per launch"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_output_buffer_pool_reuses_buffers_and_isolates_concurrent_leases() {
        // P2-M2 — the projection routes' c64 wall was per-call cuMemAlloc/cuMemFree of
        // worst-case-sized output buffers. This proves the pool's contract on real hardware:
        //   (a) a released buffer of a bucket is REUSED (same device ptr) on the next
        //       same-size lease — no fresh cuMemAlloc;
        //   (b) two simultaneously-held leases of one bucket get DISTINCT buffers (concurrent
        //       calls never alias each other's output);
        //   (c) released leases return to the idle pool, and a different size is a different
        //       bucket (no collision);
        //   (d) under real multi-thread contention, simultaneously-held leases never alias.
        // This is the binary's sole pool user, so the (c) counts are stable (no sibling race).
        let _runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
        let ctx = gpu_primary_context(0).expect("primary context");
        // The lease allocates via cuMemAlloc, so the context must be current on this thread —
        // exactly the precondition the engine dispatcher satisfies before a route runs.
        ctx.set_current()
            .expect("bind primary context on this thread");

        // (a) reuse: lease -> drop -> pool; the next same-bucket lease pops the same ptr.
        let big = 600_000_usize;
        let ptr1 = ctx.lease_device_buffer(big).expect("lease").ptr;
        let ptr2 = ctx.lease_device_buffer(big).expect("lease").ptr;
        assert_eq!(
            ptr1, ptr2,
            "a released buffer must be reused on the next same-bucket lease, not reallocated"
        );

        // (b) isolation: two leases held at once must be distinct device pointers.
        {
            let a = ctx.lease_device_buffer(big).expect("lease a");
            let b = ctx.lease_device_buffer(big).expect("lease b");
            assert_ne!(
                a.ptr, b.ptr,
                "concurrent leases of one bucket must not alias the same device buffer"
            );
        } // both returned to the pool here

        // (c) accounting: the big bucket now holds >= 2 idle buffers; a tiny lease uses a
        // different bucket (distinct ptr) and does not drain the big one. Counts are read
        // BUCKET-SCOPED (not the process-global total) so this stays correct even though the
        // migrated equal_any parity test now also leases from this shared pool — only the big
        // bucket's own idle count is load-bearing here.
        let big_idle = ctx.pooled_output_buffer_count_in_bucket(big);
        assert!(
            big_idle >= 2,
            "both released leases must return to the big bucket's idle pool (got {big_idle})"
        );
        let small = ctx.lease_device_buffer(4).expect("lease small").ptr;
        assert_ne!(
            small, ptr1,
            "a different bucket must yield a different buffer"
        );
        assert_eq!(
            ctx.pooled_output_buffer_count_in_bucket(big),
            big_idle,
            "a different-bucket (tiny) lease must NOT drain the big bucket's idle pool"
        );

        // (d) the load-bearing safety property under real contention: leases held at the same
        // instant on different threads must be DISTINCT device buffers — `pop()` removes the
        // ptr from the free list under the mutex, so no two live leases share a buffer (which
        // would let one concurrent kernel clobber another's output). A barrier holds all N
        // leases live simultaneously before their pointers are compared. Folded into this one
        // test (rather than its own) so it is the sole pool user in the binary — the (c)
        // counts above stay stable instead of racing a sibling test on the shared pool.
        const N: usize = 16;
        let barrier = std::sync::Barrier::new(N);
        let held = std::sync::Mutex::new(Vec::with_capacity(N));
        std::thread::scope(|scope| {
            for _ in 0..N {
                scope.spawn(|| {
                    ctx.set_current()
                        .expect("bind primary context on this thread");
                    let lease = ctx.lease_device_buffer(big).expect("lease");
                    held.lock().expect("held poisoned").push(lease.ptr);
                    // Keep every lease alive across the barrier so all N are live at once; no
                    // buffer is released until every thread has recorded its pointer.
                    barrier.wait();
                    drop(lease);
                });
            }
        });
        let mut ptrs = held.into_inner().expect("held poisoned");
        let total = ptrs.len();
        assert_eq!(total, N, "every thread must have recorded a lease");
        ptrs.sort_unstable();
        ptrs.dedup();
        assert_eq!(
            ptrs.len(),
            total,
            "concurrently-held leases must never alias the same device buffer"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_pinned_host_buffer_pool_reuses_buffers_and_isolates_concurrent_leases() {
        // P2-M2 — the fused text route stages its async D2H through pooled pinned (page-locked)
        // host buffers; `cuMemHostAlloc`/`cuMemFreeHost` are driver-serialized, so the buffers
        // are pooled. This mirrors the device-pool isolation test for the pinned pool and proves
        // the same contract on real hardware:
        //   (a) a released buffer of a bucket is REUSED (same host ptr) on the next same-bucket
        //       lease — no fresh cuMemHostAlloc;
        //   (b) two simultaneously-held leases of one bucket get DISTINCT buffers (concurrent
        //       calls never stage into each other's pinned region);
        //   (c) released leases return to the idle pool, and a different size is a different
        //       bucket (no collision);
        //   (d) under real multi-thread contention, simultaneously-held leases never alias.
        // This is the binary's sole pinned-pool user, so the (c) counts are stable.
        let _runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
        let ctx = gpu_primary_context(0).expect("primary context");
        // The lease allocates via cuMemHostAlloc, so the context must be current on this thread —
        // exactly the precondition the engine dispatcher satisfies before a route runs.
        ctx.set_current()
            .expect("bind primary context on this thread");

        // (a) reuse: lease -> drop -> pool; the next same-bucket lease pops the same ptr. The
        // first lease also serves as the driver-capability probe: the pinned path is gated on
        // optional driver symbols, so a `None` here means the local driver lacks
        // cuMemHostAlloc/cuMemFreeHost (the route falls back to pageable host buffers) — skip.
        // The probe is folded into this real lease rather than a throwaway one so it does not
        // pre-populate the tiny bucket that step (c)'s small lease later draws from.
        let big = 600_000_usize;
        let Some(first) = ctx.lease_pinned_host_buffer(big) else {
            eprintln!(
                "skipping: driver lacks cuMemHostAlloc/cuMemFreeHost (pinned-host pool inactive)"
            );
            return;
        };
        let ptr1 = first.ptr;
        drop(first); // return to the pool so the next same-bucket lease reuses it
        let ptr2 = ctx.lease_pinned_host_buffer(big).expect("lease").ptr;
        assert_eq!(
            ptr1, ptr2,
            "a released pinned buffer must be reused on the next same-bucket lease, not reallocated"
        );

        // (b) isolation: two leases held at once must be distinct host pointers.
        {
            let a = ctx.lease_pinned_host_buffer(big).expect("lease a");
            let b = ctx.lease_pinned_host_buffer(big).expect("lease b");
            assert_ne!(
                a.ptr, b.ptr,
                "concurrent leases of one bucket must not alias the same pinned host buffer"
            );
        } // both returned to the pool here

        // (c) accounting: the big bucket now holds >= 2 idle buffers; a tiny lease uses a
        // different bucket (distinct ptr) and does not drain the big one. Counts are read
        // BUCKET-SCOPED (not the process-global total) so this stays correct even though the
        // migrated equal_any parity test now also leases pinned buffers from this shared pool
        // (its `complete` stages result D2H through pooled pinned buffers) — only the big bucket's
        // own idle count is load-bearing here.
        let big_idle = ctx.pooled_pinned_host_buffer_count_in_bucket(big);
        assert!(
            big_idle >= 2,
            "both released pinned leases must return to the big bucket's idle pool (got {big_idle})"
        );
        let small = ctx.lease_pinned_host_buffer(4).expect("lease small").ptr;
        assert_ne!(
            small, ptr1,
            "a different bucket must yield a different pinned buffer"
        );
        assert_eq!(
            ctx.pooled_pinned_host_buffer_count_in_bucket(big),
            big_idle,
            "a different-bucket (tiny) lease must NOT drain the big bucket's idle pool"
        );

        // (d) the load-bearing safety property under real contention: leases held at the same
        // instant on different threads must be DISTINCT host buffers — `pop()` removes the ptr
        // from the free list under the mutex, so no two live leases share a buffer (which would
        // let one concurrent route's async D2H clobber another's staged result). A barrier holds
        // all N leases live simultaneously before their pointers are compared. Folded into this
        // one test (rather than its own) so it is the sole pinned-pool user in the binary — the
        // (c) counts above stay stable instead of racing a sibling test on the shared pool.
        const N: usize = 16;
        let barrier = std::sync::Barrier::new(N);
        let held = std::sync::Mutex::new(Vec::with_capacity(N));
        std::thread::scope(|scope| {
            for _ in 0..N {
                scope.spawn(|| {
                    ctx.set_current()
                        .expect("bind primary context on this thread");
                    let lease = ctx.lease_pinned_host_buffer(big).expect("lease");
                    held.lock().expect("held poisoned").push(lease.ptr as usize);
                    // Keep every lease alive across the barrier so all N are live at once; no
                    // buffer is released until every thread has recorded its pointer.
                    barrier.wait();
                    drop(lease);
                });
            }
        });
        let mut ptrs = held.into_inner().expect("held poisoned");
        let total = ptrs.len();
        assert_eq!(total, N, "every thread must have recorded a lease");
        ptrs.sort_unstable();
        ptrs.dedup();
        assert_eq!(
            ptrs.len(),
            total,
            "concurrently-held pinned leases must never alias the same host buffer"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_shared_primary_context_with_cached_module_and_per_stream_scales_concurrent_count() {
        // P2-M1 step 1 — the GPU concurrent-execution soundness/perf spike.
        //
        // The P1-M3 step-4 benchmark found the resident GPU read path *regresses* under
        // concurrency (p50 92µs→11.5ms at c64). The cause: each launch does
        // cuModuleLoadData/Unload (JIT the PTX every call) and then cuCtxSynchronize (a
        // WHOLE-context barrier) on the default stream — so N concurrent readers serialize
        // on one global sync and re-JIT the same kernel N times. This probe proves the fix
        // BEFORE refactoring production: it builds the target model in isolation —
        //   * ONE shared primary context (cuDevicePrimaryCtxRetain), not a context per
        //     allocation/generation;
        //   * the row_count module loaded ONCE, its function handle reused across every
        //     launch on every thread (a process-wide module/function cache stand-in);
        //   * each reader thread launches on its OWN stream and syncs only that stream
        //     (cuStreamSynchronize), never the whole context —
        // and A/B-times it against the current production shape (per-launch load/unload +
        // cuCtxSynchronize on the default stream) over the SAME shared context, so the
        // measured delta is exactly the two Phase-2 fixes (module cache + per-stream sync).
        // The context model is identical in both arms, so this isolates those two effects;
        // the shared-context change is validated here as "correct + concurrency-safe" (it
        // is the prerequisite for a cross-allocation module cache), not A/B'd.
        //
        // Hard gates: (a) EVERY concurrent launch returns the correct row count — the
        // soundness claim that concurrent reuse of one cached function over one shared
        // primary context from many threads is sound; (b) at the top concurrency the
        // cached/per-stream model out-throughputs the per-launch/ctx-sync model — the
        // milestone hypothesis. If (b) fails, the premise is wrong and the substrate
        // should not be built, so the probe is allowed to fail loudly.
        use std::sync::Barrier;
        use std::time::{Duration, Instant};

        // Raw CUDA handles (context, function) are `*mut c_void`, hence !Send. Wrap to
        // move them into reader threads: sound here because the retained primary context
        // and the loaded module's function handle are immutable for the probe's duration
        // and the CUDA driver API is thread-safe (concurrent launches of one function on
        // distinct streams are explicitly allowed).
        #[derive(Clone, Copy)]
        struct SendPtr(*mut c_void);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}
        impl SendPtr {
            // Access through `&self` so closures capture the whole (Send) struct rather
            // than the raw `*mut c_void` field (Rust 2021 disjoint capture would grab the
            // !Send field and refuse to cross the thread boundary).
            fn get(&self) -> *mut c_void {
                self.0
            }
        }

        // Bare driver-call signatures (mirroring the production launch sites above).
        type CuInit = unsafe extern "C" fn(u32) -> i32;
        type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
        type CuPrimaryCtxRetain = unsafe extern "C" fn(*mut *mut c_void, i32) -> i32;
        type CuPrimaryCtxRelease = unsafe extern "C" fn(i32) -> i32;
        type CuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> i32;
        type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
        type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
        type CuMemFree = unsafe extern "C" fn(u64) -> i32;
        type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
        type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
        type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
        type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
        type CuModuleGetFunction =
            unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
        type CuStreamCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
        type CuStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> i32;
        type CuStreamDestroy = unsafe extern "C" fn(*mut c_void) -> i32;

        // Snapshot of the exact production row-count kernel owned by
        // resident_header.rs (`launch_cuda_resident_row_count`):
        // reads the u64 row-count header at [resident] and stores it to [out].
        const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_row_count(
    .param .u64 resident_ptr,
    .param .u64 out_ptr
)
{
    .reg .u64 %resident;
    .reg .u64 %out;
    .reg .u64 %rows;
    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.global.u64 %rows, [%resident];
    st.global.u64 [%out], %rows;
    ret;
}
"#;

        fn check(code: i32, what: &str) {
            assert_eq!(code, 0, "{what} failed with CUDA driver code {code}");
        }

        let lib = unsafe {
            Library::new("libcuda.so.1")
                .or_else(|_| Library::new("libcuda.so"))
                .expect("requires a local NVIDIA driver and GPU")
        };

        macro_rules! load_sym {
            ($t:ty, $primary:literal $(, $fallback:literal)*) => {{
                let result = unsafe { lib.get::<$t>($primary) };
                $( let result = result.or_else(|_| unsafe { lib.get::<$t>($fallback) }); )*
                *result.expect("missing required CUDA driver symbol")
            }};
        }

        let cu_init: CuInit = load_sym!(CuInit, b"cuInit\0");
        let cu_device_get: CuDeviceGet = load_sym!(CuDeviceGet, b"cuDeviceGet\0");
        let cu_primary_ctx_retain: CuPrimaryCtxRetain =
            load_sym!(CuPrimaryCtxRetain, b"cuDevicePrimaryCtxRetain\0");
        let cu_primary_ctx_release: CuPrimaryCtxRelease = load_sym!(
            CuPrimaryCtxRelease,
            b"cuDevicePrimaryCtxRelease_v2\0",
            b"cuDevicePrimaryCtxRelease\0"
        );
        let cu_ctx_set_current: CuCtxSetCurrent = load_sym!(CuCtxSetCurrent, b"cuCtxSetCurrent\0");
        let cu_ctx_synchronize: CuCtxSynchronize =
            load_sym!(CuCtxSynchronize, b"cuCtxSynchronize\0");
        let cu_mem_alloc: CuMemAlloc = load_sym!(CuMemAlloc, b"cuMemAlloc_v2\0", b"cuMemAlloc\0");
        let cu_mem_free: CuMemFree = load_sym!(CuMemFree, b"cuMemFree_v2\0", b"cuMemFree\0");
        let cu_memcpy_htod: CuMemcpyHtoD =
            load_sym!(CuMemcpyHtoD, b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0");
        let cu_memcpy_dtoh: CuMemcpyDtoH =
            load_sym!(CuMemcpyDtoH, b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0");
        let cu_module_load_data: CuModuleLoadData =
            load_sym!(CuModuleLoadData, b"cuModuleLoadData\0");
        let cu_module_unload: CuModuleUnload = load_sym!(CuModuleUnload, b"cuModuleUnload\0");
        let cu_module_get_function: CuModuleGetFunction =
            load_sym!(CuModuleGetFunction, b"cuModuleGetFunction\0");
        let cu_launch_kernel: CuLaunchKernel = load_sym!(CuLaunchKernel, b"cuLaunchKernel\0");
        let cu_stream_create: CuStreamCreate = load_sym!(CuStreamCreate, b"cuStreamCreate\0");
        let cu_stream_synchronize: CuStreamSynchronize =
            load_sym!(CuStreamSynchronize, b"cuStreamSynchronize\0");
        let cu_stream_destroy: CuStreamDestroy = load_sym!(
            CuStreamDestroy,
            b"cuStreamDestroy_v2\0",
            b"cuStreamDestroy\0"
        );

        // One shared primary context for the whole probe.
        check(unsafe { cu_init(0) }, "cuInit");
        let mut device = 0_i32;
        check(unsafe { cu_device_get(&mut device, 0) }, "cuDeviceGet");
        let mut ctx: *mut c_void = std::ptr::null_mut();
        check(
            unsafe { cu_primary_ctx_retain(&mut ctx, device) },
            "cuDevicePrimaryCtxRetain",
        );
        check(unsafe { cu_ctx_set_current(ctx) }, "cuCtxSetCurrent(main)");

        // Payload = the 8-byte u64 row-count header the kernel reads.
        let row_count: u64 = 4096;
        let payload = row_count.to_le_bytes();
        let mut device_payload = 0_u64;
        check(
            unsafe { cu_mem_alloc(&mut device_payload, payload.len()) },
            "cuMemAlloc(payload)",
        );
        check(
            unsafe {
                cu_memcpy_htod(
                    device_payload,
                    payload.as_ptr().cast::<c_void>(),
                    payload.len(),
                )
            },
            "cuMemcpyHtoD(payload)",
        );

        // Load the module ONCE; reuse this function handle on every launch/thread.
        let mut ptx = PTX.to_vec();
        ptx.push(0);
        let mut module: *mut c_void = std::ptr::null_mut();
        check(
            unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) },
            "cuModuleLoadData(once)",
        );
        let mut function: *mut c_void = std::ptr::null_mut();
        check(
            unsafe {
                cu_module_get_function(&mut function, module, c"gpu_db_resident_row_count".as_ptr())
            },
            "cuModuleGetFunction(once)",
        );

        // Target model: cached function + a private stream per thread + per-stream sync.
        let run_cached = |threads: usize, ops: usize| -> Duration {
            let barrier = Arc::new(Barrier::new(threads + 1));
            let ctx = SendPtr(ctx);
            let function = SendPtr(function);
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        check(
                            unsafe { cu_ctx_set_current(ctx.get()) },
                            "cuCtxSetCurrent(cached)",
                        );
                        let mut stream: *mut c_void = std::ptr::null_mut();
                        check(
                            unsafe { cu_stream_create(&mut stream, 0) },
                            "cuStreamCreate",
                        );
                        let mut out = 0_u64;
                        check(
                            unsafe { cu_mem_alloc(&mut out, std::mem::size_of::<u64>()) },
                            "cuMemAlloc(out,cached)",
                        );
                        barrier.wait();
                        for _ in 0..ops {
                            let mut resident_arg = device_payload;
                            let mut out_arg = out;
                            let mut args = [
                                (&mut resident_arg as *mut u64).cast::<c_void>(),
                                (&mut out_arg as *mut u64).cast::<c_void>(),
                            ];
                            check(
                                unsafe {
                                    cu_launch_kernel(
                                        function.get(),
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        0,
                                        stream,
                                        args.as_mut_ptr(),
                                        std::ptr::null_mut(),
                                    )
                                },
                                "cuLaunchKernel(cached)",
                            );
                            check(
                                unsafe { cu_stream_synchronize(stream) },
                                "cuStreamSynchronize",
                            );
                            let mut got = 0_u64;
                            check(
                                unsafe {
                                    cu_memcpy_dtoh(
                                        (&mut got as *mut u64).cast::<c_void>(),
                                        out,
                                        std::mem::size_of::<u64>(),
                                    )
                                },
                                "cuMemcpyDtoH(cached)",
                            );
                            assert_eq!(
                                got, row_count,
                                "cached concurrent count returned wrong value"
                            );
                        }
                        unsafe {
                            cu_mem_free(out);
                            cu_stream_destroy(stream);
                        }
                    })
                })
                .collect();
            barrier.wait();
            let start = Instant::now();
            for handle in handles {
                handle.join().expect("cached reader panicked");
            }
            start.elapsed()
        };

        // Production shape: load+unload the module every call, launch on the default
        // stream, sync the whole context. Same shared primary context as the cached arm.
        let run_per_launch = |threads: usize, ops: usize| -> Duration {
            let barrier = Arc::new(Barrier::new(threads + 1));
            let ctx = SendPtr(ctx);
            let ptx = Arc::new({
                let mut p = PTX.to_vec();
                p.push(0);
                p
            });
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let barrier = Arc::clone(&barrier);
                    let ptx = Arc::clone(&ptx);
                    std::thread::spawn(move || {
                        check(
                            unsafe { cu_ctx_set_current(ctx.get()) },
                            "cuCtxSetCurrent(perlaunch)",
                        );
                        let mut out = 0_u64;
                        check(
                            unsafe { cu_mem_alloc(&mut out, std::mem::size_of::<u64>()) },
                            "cuMemAlloc(out,perlaunch)",
                        );
                        barrier.wait();
                        for _ in 0..ops {
                            let mut m: *mut c_void = std::ptr::null_mut();
                            check(
                                unsafe {
                                    cu_module_load_data(&mut m, ptx.as_ptr().cast::<c_void>())
                                },
                                "cuModuleLoadData(perlaunch)",
                            );
                            let mut f: *mut c_void = std::ptr::null_mut();
                            check(
                                unsafe {
                                    cu_module_get_function(
                                        &mut f,
                                        m,
                                        c"gpu_db_resident_row_count".as_ptr(),
                                    )
                                },
                                "cuModuleGetFunction(perlaunch)",
                            );
                            let mut resident_arg = device_payload;
                            let mut out_arg = out;
                            let mut args = [
                                (&mut resident_arg as *mut u64).cast::<c_void>(),
                                (&mut out_arg as *mut u64).cast::<c_void>(),
                            ];
                            check(
                                unsafe {
                                    cu_launch_kernel(
                                        f,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        0,
                                        std::ptr::null_mut(),
                                        args.as_mut_ptr(),
                                        std::ptr::null_mut(),
                                    )
                                },
                                "cuLaunchKernel(perlaunch)",
                            );
                            check(unsafe { cu_ctx_synchronize() }, "cuCtxSynchronize");
                            let mut got = 0_u64;
                            check(
                                unsafe {
                                    cu_memcpy_dtoh(
                                        (&mut got as *mut u64).cast::<c_void>(),
                                        out,
                                        std::mem::size_of::<u64>(),
                                    )
                                },
                                "cuMemcpyDtoH(perlaunch)",
                            );
                            assert_eq!(
                                got, row_count,
                                "per-launch concurrent count returned wrong value"
                            );
                            unsafe { cu_module_unload(m) };
                        }
                        unsafe { cu_mem_free(out) };
                    })
                })
                .collect();
            barrier.wait();
            let start = Instant::now();
            for handle in handles {
                handle.join().expect("per-launch reader panicked");
            }
            start.elapsed()
        };

        let ops = 100usize;
        let concurrencies = [1usize, 2, 4, 8, 16, 32, 64];
        // Warm both paths (driver lazy-init, JIT cache for the per-launch arm) so we
        // measure steady state — this is conservative, it helps the per-launch baseline.
        let _ = run_per_launch(1, 10);
        let _ = run_cached(1, 10);

        println!(
            "p2_m1_spike: shared primary ctx + load-once module + per-stream  vs  per-launch load + ctx-sync"
        );
        println!("row_count={row_count} ops/thread={ops}");
        println!("| conc | per-launch qps | cached qps | speedup |");
        println!("|---:|---:|---:|---:|");
        let mut json_cells: Vec<String> = Vec::new();
        let mut top_cached_qps = 0.0_f64;
        let mut top_per_launch_qps = 0.0_f64;
        for &c in &concurrencies {
            let total = (c * ops) as f64;
            let d_pl = run_per_launch(c, ops);
            let d_ca = run_cached(c, ops);
            let qps_pl = total / d_pl.as_secs_f64();
            let qps_ca = total / d_ca.as_secs_f64();
            let speedup = if qps_pl > 0.0 { qps_ca / qps_pl } else { 0.0 };
            println!("| {c} | {qps_pl:.0} | {qps_ca:.0} | {speedup:.2}x |");
            json_cells.push(format!(
                "{{\"concurrency\":{c},\"per_launch_qps\":{qps_pl:.1},\"cached_qps\":{qps_ca:.1},\"speedup\":{speedup:.3}}}"
            ));
            if c == 64 {
                top_cached_qps = qps_ca;
                top_per_launch_qps = qps_pl;
            }
        }
        println!(
            "json={{\"kind\":\"p2_m1_gpu_concurrency_spike\",\"row_count\":{row_count},\"ops_per_thread\":{ops},\"cells\":[{}]}}",
            json_cells.join(",")
        );

        unsafe {
            cu_mem_free(device_payload);
            cu_module_unload(module);
            cu_primary_ctx_release(device);
        }

        assert!(
            top_cached_qps > top_per_launch_qps,
            "milestone premise unmet: at c64 the cached/per-stream model ({top_cached_qps:.0} qps) \
             did not beat the per-launch/ctx-sync model ({top_per_launch_qps:.0} qps)"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_parallel_i32_equal_count_matches_serial_and_wins_on_large_tables() {
        // P2-M2 step 1 — parallel scan kernel spike.
        //
        // The resident filtered-count route (`gpu_db_resident_i32_equal_count`) is a
        // single-thread `(1,1,1)` serial loop — one GPU thread scanning every row, the
        // "shallow GPU" the plan §1.2 calls out. This probe proves the Phase-2 fix BEFORE
        // wiring it into the route: a real grid/block strided scan + global reduction
        // (`launch_cuda_resident_i32_equal_count_parallel`). It A/Bs the parallel kernel
        // vs the serial one across a row-count sweep, hard-asserting (a) the parallel count
        // EQUALS the previously-shipped serial (1,1,1) scan at every size — the serial kernel
        // is the GPU-native oracle (no CPU re-implementation of the count operator as the
        // expected), so matching it proves the atomic reduction + grid-stride bounds correct —
        // and (b) on a large table the parallel kernel is faster (the milestone hypothesis).
        use std::time::Instant;

        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        let needle = 3_i32;
        // Partial-block / sub-256 / odd sizes FIRST (255/257/513/1000): they exercise the per-block tree
        // reduction's zero-iteration threads + partial last block -- exactly the path the count-skeleton
        // fix's first (shfl) attempt UNDERCOUNTED. The serial (1,1,1) oracle validates the exact count
        // there. The large multiples-of-256 sizes (last) keep the speedup hypothesis.
        let sizes: [u64; 9] = [
            255,
            257,
            513,
            1_000,
            4_096,
            65_536,
            1 << 20,
            1 << 22,
            1 << 24,
        ];
        println!("p2_m2_parallel_scan_spike: needle={needle} (value[i] = i % 7)");
        println!("| rows | expected | serial ms | parallel ms | speedup |");
        println!("|---:|---:|---:|---:|---:|");

        let mut largest_serial_ms = 0.0_f64;
        let mut largest_parallel_ms = 0.0_f64;
        for &n in &sizes {
            let values: Vec<i32> = (0..n).map(|i| (i % 7) as i32).collect();
            // SAFETY: `i32` is plain-old-data with no padding; viewing the Vec as native
            // (little-endian) bytes matches what the kernel's `ld.global.s32` reads, and
            // `values` outlives this borrow (retain copies to the device synchronously).
            let column_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
            };
            let header = n.to_le_bytes();
            let allocated = std::mem::size_of::<u64>() as u64 + column_bytes.len() as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    allocated,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: std::mem::size_of::<u64>() as u64,
                            bytes: column_bytes,
                        },
                    ],
                )
                .expect("retain resident column");
            let offset = std::mem::size_of::<u64>() as u64;

            // Correctness (also warms each kernel's module load): the parallel kernel must EQUAL
            // the previously-shipped serial (1,1,1) scan ON THE GPU — the serial scan is the
            // GPU-native oracle (no CPU re-implementation of the count operator as the expected).
            let serial =
                launch_cuda_resident_i32_equal_count_serial(&resident, offset, n, needle, None)
                    .expect("serial count");
            let parallel = launch_cuda_resident_i32_equal_count(&resident, offset, n, needle, None)
                .expect("parallel count");
            assert_eq!(
                parallel, serial,
                "parallel count != serial reference at rows={n}"
            );

            // Timing: best of 3 (latency).
            let mut serial_ms = f64::MAX;
            let mut parallel_ms = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                launch_cuda_resident_i32_equal_count_serial(&resident, offset, n, needle, None)
                    .unwrap();
                serial_ms = serial_ms.min(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                launch_cuda_resident_i32_equal_count(&resident, offset, n, needle, None).unwrap();
                parallel_ms = parallel_ms.min(t.elapsed().as_secs_f64() * 1e3);
            }
            let speedup = if parallel_ms > 0.0 {
                serial_ms / parallel_ms
            } else {
                0.0
            };
            println!("| {n} | {serial} | {serial_ms:.3} | {parallel_ms:.3} | {speedup:.1}x |");
            largest_serial_ms = serial_ms;
            largest_parallel_ms = parallel_ms;
        }

        assert!(
            largest_parallel_ms < largest_serial_ms,
            "parallel kernel ({largest_parallel_ms:.3} ms) did not beat the serial (1,1,1) \
             kernel ({largest_serial_ms:.3} ms) on the largest table — milestone premise unmet"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_resident_i32_equal_count_excludes_null_rows_via_validity_bitmap() {
        // M3 (doc 21): the resident equal-count kernels honor the per-column NULL VALIDITY bitmap
        // (1 = valid, 0 = NULL) ON THE GPU — a NULL operand never matches (three-valued logic), even
        // though its value-section bytes are a 0 placeholder. GPU-native oracle: serial == parallel
        // (no CPU re-implementation of the operator), plus a construction count, plus a control that
        // shows the bitmap is what does the excluding.
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        let n: u64 = 200;
        // NULL every 7th row. Non-null values are (i % 5) + 1 ∈ {1,2,3,4,5}, so 0 is NEVER a real
        // value — it can only appear as a NULL row's placeholder.
        let is_null = |i: u64| i.is_multiple_of(7);
        let values: Vec<i32> = (0..n)
            .map(|i| if is_null(i) { 0 } else { (i % 5) as i32 + 1 })
            .collect();
        // Validity bitmap: ceil(n/32) little-endian u32 words, bit i = row i, 1 = valid.
        let mut bitmap = vec![0u32; (n as usize).div_ceil(32)];
        for i in 0..n as usize {
            if !is_null(i as u64) {
                bitmap[i / 32] |= 1u32 << (i % 32);
            }
        }
        // Payload: header(8) + int4 column (n*4) + validity bitmap (words*4). The bitmap is the
        // section build_relational_device_payload emits after the columns (slice 2a).
        let header = n.to_le_bytes();
        // SAFETY: i32/u32 are plain-old-data; viewing the Vecs as native (little-endian) bytes
        // matches the kernels' `ld.global.s32`/`ld.global.u32`, and both outlive this borrow.
        let column_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
        let bitmap_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bitmap.as_ptr().cast::<u8>(), bitmap.len() * 4) };
        let header_len = std::mem::size_of::<u64>() as u64;
        let null_offset = header_len + column_bytes.len() as u64;
        let allocated = null_offset + bitmap_bytes.len() as u64;
        let resident = runtime
            .retain_device_memory_chunks(
                0,
                allocated,
                &[
                    CudaDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes: &header,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: header_len,
                        bytes: column_bytes,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: null_offset,
                        bytes: bitmap_bytes,
                    },
                ],
            )
            .expect("retain resident column + validity bitmap");
        let offset = header_len;
        let n_null = (0..n).filter(|&i| is_null(i)).count() as u64;
        assert!(n_null > 0);

        // (1) needle 0 collides with the NULL placeholder: WITH the bitmap, count(col == 0) must be 0
        //     (no non-null row has value 0), and serial must equal parallel ON THE GPU.
        let serial0 =
            launch_cuda_resident_i32_equal_count_serial(&resident, offset, n, 0, Some(null_offset))
                .expect("serial count with bitmap");
        let parallel0 =
            launch_cuda_resident_i32_equal_count(&resident, offset, n, 0, Some(null_offset))
                .expect("parallel count with bitmap");
        assert_eq!(
            serial0, parallel0,
            "serial == parallel (GPU oracle) at needle=0"
        );
        assert_eq!(
            serial0, 0,
            "NULL rows (placeholder 0) must NOT match WHERE col = 0"
        );

        // Control: WITHOUT the bitmap (None), the 0 placeholders DO leak as matches — proving the
        // setup is real and the bitmap is precisely what excludes the NULL rows.
        let serial0_no_bitmap =
            launch_cuda_resident_i32_equal_count_serial(&resident, offset, n, 0, None)
                .expect("serial count without bitmap");
        assert_eq!(
            serial0_no_bitmap, n_null,
            "without the validity bitmap, every NULL placeholder leaks as a 0 match"
        );

        // (2) needle 3: non-null matching is preserved alongside the bitmap. Expected = non-null rows
        //     whose value is 3, i.e. (i % 5) + 1 == 3.
        let expected3 = (0..n).filter(|&i| !is_null(i) && (i % 5) + 1 == 3).count() as u64;
        let serial3 =
            launch_cuda_resident_i32_equal_count_serial(&resident, offset, n, 3, Some(null_offset))
                .expect("serial count needle=3");
        let parallel3 =
            launch_cuda_resident_i32_equal_count(&resident, offset, n, 3, Some(null_offset))
                .expect("parallel count needle=3");
        assert_eq!(
            serial3, parallel3,
            "serial == parallel (GPU oracle) at needle=3"
        );
        assert_eq!(serial3, expected3, "construction count excludes NULL rows");
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_resident_i32_between_stats_skips_null_values_via_validity_bitmap() {
        // M3 (doc 21): the BETWEEN stats kernel excludes NULL values ON THE GPU via the validity bitmap
        // (a NULL never satisfies a range; its bytes are a 0 placeholder). GPU-native oracle: a
        // construction check against hand-computed non-NULL-in-range stats, plus a control over a range
        // that INCLUDES the 0 placeholder so that without the bitmap the placeholders demonstrably leak.
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        let n: u64 = 200;
        let is_null = |i: u64| i.is_multiple_of(7);
        // Non-null values (i % 5) + 1 ∈ {1,2,3,4,5}; 0 is only ever a NULL row's placeholder.
        let values: Vec<i32> = (0..n)
            .map(|i| if is_null(i) { 0 } else { (i % 5) as i32 + 1 })
            .collect();
        let mut bitmap = vec![0u32; (n as usize).div_ceil(32)];
        for i in 0..n as usize {
            if !is_null(i as u64) {
                bitmap[i / 32] |= 1u32 << (i % 32);
            }
        }
        // SAFETY: i32/u32 are POD; native LE bytes match the kernel loads; the Vecs outlive the retain.
        let column_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
        let bitmap_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bitmap.as_ptr().cast::<u8>(), bitmap.len() * 4) };
        let header = n.to_le_bytes();
        let header_len = std::mem::size_of::<u64>() as u64;
        let null_off = header_len + column_bytes.len() as u64;
        let allocated = null_off + bitmap_bytes.len() as u64;
        let resident = runtime
            .retain_device_memory_chunks(
                0,
                allocated,
                &[
                    CudaDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes: &header,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: header_len,
                        bytes: column_bytes,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: null_off,
                        bytes: bitmap_bytes,
                    },
                ],
            )
            .expect("retain resident column + validity bitmap");
        let off = header_len;

        // Range [0,4] INCLUDES the placeholder 0. WITH the bitmap, only non-null values in [0,4] count
        // (i.e. {1,2,3,4}; value 5 excluded). Hand-computed construction over the non-null rows.
        let (mut exp_count, mut exp_sum, mut exp_min, mut exp_max) =
            (0u64, 0i64, i32::MAX, i32::MIN);
        for i in 0..n {
            if !is_null(i) {
                let v = (i % 5) as i32 + 1;
                if (0..=4).contains(&v) {
                    exp_count += 1;
                    exp_sum += i64::from(v);
                    exp_min = exp_min.min(v);
                    exp_max = exp_max.max(v);
                }
            }
        }
        let stats = resident
            .stats_i32_between_nullable_from_payload(off, n, 0, 4, Some(null_off))
            .expect("nullable between stats");
        assert_eq!(stats.count, exp_count, "between count excludes NULL");
        assert_eq!(stats.sum, exp_sum, "between sum excludes NULL");
        assert_eq!(
            stats.min,
            Some(exp_min),
            "between min excludes NULL (not the 0 placeholder)"
        );
        assert_eq!(stats.max, Some(exp_max), "between max excludes NULL");

        // Control: WITHOUT the bitmap, the NULL placeholder 0s satisfy [0,4] and leak — count rises by
        // n_null and min drops to 0 — proving the construction is non-vacuous and the bitmap excludes.
        let n_null = (0..n).filter(|&i| is_null(i)).count() as u64;
        assert!(n_null > 0);
        let leaked = resident
            .stats_i32_between_from_payload(off, n, 0, 4)
            .expect("between stats without bitmap");
        assert_eq!(
            leaked.count,
            exp_count + n_null,
            "without bitmap the 0 placeholders leak in"
        );
        assert_eq!(
            leaked.min,
            Some(0),
            "without bitmap min collapses to the 0 placeholder"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_parallel_i32_compare_count_matches_serial_and_wins_on_large_tables() {
        // P2-M2 — compare-count parallelization parity + perf gate (GPU-native oracle).
        //
        // `launch_cuda_resident_i32_compare_count` was a single-thread `(1,1,1)` serial scan; it is
        // now a grid-stride + `red.global.add.u64` parallel kernel (the same skeleton as
        // `gpu_db_resident_i32_equal_count_parallel`) on a pooled private stream. Mirroring the
        // sibling equal_count gate, this A/Bs the parallel kernel against the retained serial
        // reference ON THE GPU (no CPU oracle): it hard-asserts (a) the parallel count EQUALS the
        // serial count at every size for every comparison code (`<`/`<=`/`>`/`>=`) and for the
        // composed BETWEEN — across sub-block, block-boundary, multi-block, and a row count that
        // exceeds one full `gridDim*blockDim` grid-stride pass (1<<24 > 65_535*256, the wrap path)
        // — and (b) the parallel kernel beats the serial `(1,1,1)` kernel on the largest table.
        use std::time::Instant;

        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        let needle = 3_i32;
        let sizes: [u64; 7] = [1, 256, 257, 4_096, 65_536, 1 << 20, 1 << 24];
        let comparisons = [
            CudaI32Comparison::Lt,
            CudaI32Comparison::Lte,
            CudaI32Comparison::Gt,
            CudaI32Comparison::Gte,
        ];
        println!("p2_m2_compare_count: needle={needle} (value[i] = i % 7)");
        println!("| rows | serial ms | parallel ms | speedup |");
        println!("|---:|---:|---:|---:|");

        let mut largest_serial_ms = 0.0_f64;
        let mut largest_parallel_ms = 0.0_f64;
        for &n in &sizes {
            let values: Vec<i32> = (0..n).map(|i| (i % 7) as i32).collect();
            // SAFETY: `i32` is POD with no padding; viewing the Vec as native (little-endian)
            // bytes matches the kernel's `ld.global.s32`, and `values` outlives the retain copy.
            let column_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
            };
            let header = n.to_le_bytes();
            let allocated = std::mem::size_of::<u64>() as u64 + column_bytes.len() as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    allocated,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: std::mem::size_of::<u64>() as u64,
                            bytes: column_bytes,
                        },
                    ],
                )
                .expect("retain resident column");
            let offset = std::mem::size_of::<u64>() as u64;

            // Parity: parallel == serial for every comparison (GPU-vs-GPU; the serial reference is
            // the previously-shipped kernel, so matching it proves the parallel regression-safe).
            for &comparison in &comparisons {
                let serial = launch_cuda_resident_i32_compare_count_serial(
                    &resident, offset, n, needle, comparison,
                )
                .expect("serial compare count");
                let parallel = resident
                    .count_i32_compare_from_payload(offset, n, needle, comparison)
                    .expect("parallel compare count");
                assert_eq!(
                    parallel, serial,
                    "parallel != serial at rows={n} comparison={comparison:?}"
                );
            }

            // BETWEEN composes two compare-counts; check the parallel-based public route against a
            // serial-composed reference (Gte lower − Gt upper, saturating).
            let (lower, upper) = (2_i32, 5_i32);
            let serial_between = launch_cuda_resident_i32_compare_count_serial(
                &resident,
                offset,
                n,
                lower,
                CudaI32Comparison::Gte,
            )
            .expect("serial gte")
            .saturating_sub(
                launch_cuda_resident_i32_compare_count_serial(
                    &resident,
                    offset,
                    n,
                    upper,
                    CudaI32Comparison::Gt,
                )
                .expect("serial gt"),
            );
            let parallel_between = resident
                .count_i32_between_from_payload(offset, n, lower, upper)
                .expect("parallel between count");
            assert_eq!(
                parallel_between, serial_between,
                "between parallel != serial at rows={n}"
            );

            // Timing: best-of-3 latency on Gte (a 4/7-selectivity scan), like the equal_count gate.
            let mut serial_ms = f64::MAX;
            let mut parallel_ms = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                launch_cuda_resident_i32_compare_count_serial(
                    &resident,
                    offset,
                    n,
                    needle,
                    CudaI32Comparison::Gte,
                )
                .unwrap();
                serial_ms = serial_ms.min(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                resident
                    .count_i32_compare_from_payload(offset, n, needle, CudaI32Comparison::Gte)
                    .unwrap();
                parallel_ms = parallel_ms.min(t.elapsed().as_secs_f64() * 1e3);
            }
            let speedup = if parallel_ms > 0.0 {
                serial_ms / parallel_ms
            } else {
                0.0
            };
            println!("| {n} | {serial_ms:.3} | {parallel_ms:.3} | {speedup:.1}x |");
            largest_serial_ms = serial_ms;
            largest_parallel_ms = parallel_ms;
        }

        assert!(
            largest_parallel_ms < largest_serial_ms,
            "parallel kernel ({largest_parallel_ms:.3} ms) did not beat the serial (1,1,1) \
             kernel ({largest_serial_ms:.3} ms) on the largest table — milestone premise unmet"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_pooled_i32_sum_matches_closed_form_across_sizes() {
        // P2-M2 — sum pooled-stream launch-migration correctness gate (GPU-native, construction
        // oracle). The i32-sum kernel is already a parallel grid-stride reduction; this migrated its
        // launch off the default/null stream + per-call cuModuleLoadData/cuMemAlloc onto a pooled
        // private stream with a cached module + async-memset scratch. Assert the migrated GPU sum
        // EQUALS the CLOSED-FORM sum of the synthetic column (value[i] = i % 7 => sum =
        // 21*(n/7) + r*(r-1)/2 for r = n % 7) across sub-block .. multi-block sizes — no CPU operator
        // re-implementation, just the known-data constant. This gates the migrated launch wiring
        // (scratch zeroing, kernel atomic-add into scratch, scratch readback); end-to-end value
        // correctness through the engine is also covered by the scalar_aggregate probe.
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        let sizes: [u64; 5] = [1, 257, 4_096, 65_536, 1 << 24];
        for &n in &sizes {
            let values: Vec<i32> = (0..n).map(|i| (i % 7) as i32).collect();
            // SAFETY: `i32` is POD; viewing the Vec as native (little-endian) bytes matches the
            // resident column layout, and `values` outlives the retain copy.
            let column_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
            };
            let header = n.to_le_bytes();
            let allocated = std::mem::size_of::<u64>() as u64 + column_bytes.len() as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    allocated,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: std::mem::size_of::<u64>() as u64,
                            bytes: column_bytes,
                        },
                    ],
                )
                .expect("retain resident column");
            let offset = std::mem::size_of::<u64>() as u64;

            // Closed form: each full period of 7 sums to 0+1+..+6 = 21; the r = n % 7 remainder
            // adds 0+1+..+(r-1) = r*(r-1)/2. (value[i] = i % 7.)
            let q = (n / 7) as i64;
            let r = (n % 7) as i64;
            let expected = 21 * q + (r * (r - 1)) / 2;
            let gpu = resident.sum_i32_from_payload(offset, n).expect("gpu sum");
            assert_eq!(gpu, expected, "sum mismatch at rows={n}");
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_i32_sum_byte_identical_to_host_wrapping_oracle() {
        // ADVERSARIAL AUDIT (commit 12d2034a — per-block tree reduction for the i32 sum kernel).
        // The shipped closed-form test uses value[i] = i % 7 (all NON-negative, never wraps), so it
        // does NOT exercise the load-bearing CLAIM: that the barrier-synchronized SHARED-MEMORY tree
        // (u64 partials) is byte-identical to the old per-thread atom.add.u64 storm BECAUSE
        // two's-complement (u64) addition is associative AND commutative (mod 2^64). This oracle drives
        // `sum_i32_from_payload` and asserts the EXACT i64 against the host computed by `i64::wrapping_add`
        // over the column in LINEAR order. If the kernel's tree grouping/order produced a different bit
        // pattern under sign-reinterpretation or modular overflow, the linear host oracle would diverge.
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        // Host oracle: sum i32 values widened to i64, modulo 2^64 (wrapping) in LINEAR order. The kernel
        // groups/orders the adds differently (per-thread grid-stride, then a per-block halving tree, then
        // one atomic per block); equality here is exactly the associativity/commutativity claim.
        let host_wrapping_sum = |values: &[i32]| -> i64 {
            values
                .iter()
                .fold(0_i64, |acc, &v| acc.wrapping_add(i64::from(v)))
        };

        let run = |values: &[i32]| -> i64 {
            let n = values.len() as u64;
            // SAFETY: `i32` is POD; native (little-endian) bytes match the resident column layout, and
            // `values` outlives the synchronous retain copy.
            let column_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
            };
            let header = n.to_le_bytes();
            let allocated = std::mem::size_of::<u64>() as u64 + column_bytes.len() as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    allocated.max(std::mem::size_of::<u64>() as u64),
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: std::mem::size_of::<u64>() as u64,
                            bytes: column_bytes,
                        },
                    ],
                )
                .expect("retain resident column");
            resident
                .sum_i32_from_payload(std::mem::size_of::<u64>() as u64, n)
                .expect("gpu sum")
        };

        let check = |values: &[i32], label: &str| {
            let expected = host_wrapping_sum(values);
            let gpu = run(values);
            assert_eq!(
                gpu,
                expected,
                "GPU sum != host wrapping oracle for case '{label}' (n={})",
                values.len()
            );
        };

        // --- EDGE: row_count = 0 (grid clamped to 1 block; loop runs zero iterations in every thread;
        // every s_part[tid] must be the zero-init %sum). Expect 0. ---
        check(&[], "zero rows");

        // --- EDGE: row_count < 256 (most threads contribute 0 -- their s_part[tid] MUST be the
        // `mov.s64 %sum,0` init, not garbage). Mixed sign so the zero contributors are distinguishable
        // from a hypothetical garbage add. ---
        check(&[5], "single row");
        check(
            &[i32::MIN],
            "single i32::MIN (negative -> high-bit-set u64 partial)",
        );
        check(&[i32::MAX, i32::MIN, -1, 1, 7], "5 rows mixed sign");
        check(
            &[-7; 100],
            "100 negative rows (< one warp-block tail of 256)",
        );
        check(
            &[i32::MIN; 255],
            "255 negative rows (one short of a full block)",
        );

        // --- EDGE: partial last block (not a multiple of 256) + odd counts. ---
        check(
            &(0..257)
                .map(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN })
                .collect::<Vec<_>>(),
            "257 alternating MAX/MIN (partial block, odd)",
        );
        check(
            &(0..1001).map(|i| i - 500).collect::<Vec<_>>(),
            "1001 rows -500..500 (partial block, odd, mixed sign, true sum 500)",
        );

        // --- GRID-STRIDE WRAP: > 262144 rows (=1024 blocks * 256 threads), so each thread sums MANY
        // rows into a large i64 partial. 4M+1 is NOT a multiple of the grid width nor of 256 (partial
        // tail), with large-magnitude alternating values so partials are big and sign-mixed. ---
        let big: Vec<i32> = (0..4_000_001_u64)
            .map(|i| {
                if i % 2 == 0 {
                    i32::MAX - (i % 17) as i32
                } else {
                    i32::MIN + (i % 13) as i32
                }
            })
            .collect();
        check(
            &big,
            "4_000_001 large alternating (grid-stride wrap, partial tail)",
        );

        // --- THE CORE CLAIM: genuine mod-2^64 OVERFLOW in the u64 tree adds. A pure i32 sum cannot
        // overflow i64 at feasible row counts (would need ~2^32 rows of i32::MAX, ~17GB), so a true i64
        // wrap of the FINAL value is infeasible. BUT the tree/atomic adds happen in u64 and a NEGATIVE
        // true sum is stored as a high-bit-set u64 (e.g. -10 -> 2^64-10); summing many such negative
        // per-thread/per-block partials overflows 2^64 at EACH tree level and EACH block atomic and must
        // wrap-around to land on the correct negative i64 bits. This column has a large NEGATIVE true sum
        // produced by hundreds of thousands of threads each holding a negative partial, so the modular
        // wrap in the tree is genuinely exercised; the linear host `wrapping_add` oracle is the only
        // correct answer. We additionally assert below (non-vacuity) that the u64 intermediate sum of the
        // raw partials really does exceed 2^64. ---
        let neg_n = 2_000_003_u64; // > grid width, odd, partial tail
        let neg: Vec<i32> = (0..neg_n).map(|_| i32::MIN).collect();
        let expected_neg = host_wrapping_sum(&neg); // = (i32::MIN as i64) * neg_n, fits i64 (no FINAL wrap)
        let gpu_neg = run(&neg);
        assert_eq!(
            gpu_neg, expected_neg,
            "GPU sum != host oracle for the all-i32::MIN negative-partial wrap case (n={neg_n})"
        );
        // NON-VACUITY of the modular wrap: the sum of the per-thread u64 partials (each a negative-as-u64
        // value) really does overflow 2^64. With the grid clamped to 1024*256 = 262144 threads, every
        // active thread's partial is negative, and the u64 sum of all 262144 partials (= the same number
        // as `expected_neg` reinterpreted, but the INTERMEDIATE u64 accumulation across positive-as-u64
        // reinterpretations) crosses 2^64. Concretely: there are 262144 active threads, each holding a
        // partial whose u64 value is >= 2^63 (negative), so summing just two of them already exceeds
        // 2^64 -- the tree MUST wrap mod 2^64 at the very first level for the bits to stay correct.
        let threads = 262_144_u128;
        let min_partial_u64 = (i64::from(i32::MIN)) as u64 as u128; // negative -> ~2^63..2^64
        assert!(
            min_partial_u64 >= (1_u128 << 63),
            "expected i32::MIN partial to set the u64 high bit (got {min_partial_u64:#x})"
        );
        assert!(
            min_partial_u64 * 2 >= (1_u128 << 64),
            "two negative-as-u64 partials must already overflow 2^64 (proves the tree adds wrap)"
        );
        let _ = threads; // documented thread count; the 2-partial overflow above is the load-bearing proof

        // --- CONTROL: a large NON-negative grid-stride-wrap case (matches the shipped oracle's regime
        // but at a partial-tail odd size), to confirm the wrapping path also stays correct when no high
        // bit is set. ---
        check(
            &(0..3_000_007_u64)
                .map(|i| (i % 7) as i32)
                .collect::<Vec<_>>(),
            "3_000_007 non-negative i%7 (grid-stride wrap, partial tail, control)",
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_i32_scalar_stats_byte_identical_to_host_oracle() {
        // ADVERSARIAL AUDIT (commit 75aeb493 -- direct (count,sum,min,max) scalar-stats reduction
        // that replaced the self-grouped hash path for unfiltered non-nullable MIN/MAX/AVG).
        //
        // The load-bearing CLAIM is BYTE-IDENTITY: the direct kernel's (count, sum, min, max) must
        // equal an independent HOST oracle. (The dead self-grouped A/B oracle arm was dropped with the
        // engine-dead `grouped_stats` family; the host oracle still pins the direct kernel exactly.)
        // The NEW min/max logic is what this commit introduces, so the focus is
        // sentinel-valued (i32::MIN/MAX as REAL data, not init sentinels), all-negative (a leaked
        // INT_MAX init would surface), all-positive (a leaked INT_MIN init would surface), single row,
        // partial last block, ODD counts, and grid-stride wrap. count must == rows (non-nullable
        // unfiltered), and i64 sum must == the linear host `wrapping_add` oracle.
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        // Independent HOST oracle (linear order; the kernel groups/orders adds via a per-thread
        // grid-stride scan then a per-block halving tree then one atomic/block -- equality is exactly
        // the associativity/commutativity claim for u64 sum and the assoc/comm claim for min/max.s32).
        let host_oracle = |values: &[i32]| -> (u64, i64, i32, i32) {
            let count = values.len() as u64;
            let sum = values
                .iter()
                .fold(0_i64, |acc, &v| acc.wrapping_add(i64::from(v)));
            let min = values.iter().copied().min().unwrap_or(i32::MAX);
            let max = values.iter().copied().max().unwrap_or(i32::MIN);
            (count, sum, min, max)
        };

        let run = |values: &[i32]| -> (u64, i64, i32, i32) {
            let n = values.len() as u64;
            // SAFETY: `i32` is POD; native (little-endian) bytes match the resident column layout, and
            // `values` outlives the synchronous retain copy.
            let column_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
            };
            let header = n.to_le_bytes();
            let allocated = std::mem::size_of::<u64>() as u64 + column_bytes.len() as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    allocated.max(std::mem::size_of::<u64>() as u64),
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: std::mem::size_of::<u64>() as u64,
                            bytes: column_bytes,
                        },
                    ],
                )
                .expect("retain resident column");
            let off = std::mem::size_of::<u64>() as u64;

            // Direct scalar-stats path.
            resident
                .scalar_stats_i32_from_payload(off, n)
                .expect("direct scalar stats")
        };

        let check = |values: &[i32], label: &str| {
            let oracle = host_oracle(values);
            let direct = run(values);
            assert_eq!(
                direct,
                oracle,
                "DIRECT scalar_stats != host oracle for '{label}' (n={})",
                values.len()
            );
            // count must == rows on this unfiltered non-nullable path.
            assert_eq!(direct.0, values.len() as u64, "count != rows for '{label}'");
        };

        // --- min/max NEW logic: sentinel-valued REAL data (must return i32::MIN/MAX, NOT confuse with
        // the INT_MAX/INT_MIN init sentinels). ---
        check(&[i32::MIN], "single i32::MIN as REAL min");
        check(&[i32::MAX], "single i32::MAX as REAL max");
        check(&[i32::MIN, i32::MAX], "both extremes");
        check(&[i32::MAX, i32::MIN, 0, -1, 1], "extremes + small mixed");

        // --- all-negative: a leaked INT_MAX min-sentinel or mis-init min=0 would surface (max must be
        // negative; min must be the most-negative real value). ---
        check(&[-1, -5, -100, -7, -3], "all-negative small");
        check(
            &(1..=1000_i32).map(|i| -i).collect::<Vec<_>>(),
            "all-negative 1000 (-1..-1000)",
        );
        check(&[-42; 257], "all-equal negative, 257 (partial block, odd)");

        // --- all-positive: a leaked INT_MIN max-sentinel or mis-init min=0 (would wrongly win the min
        // of an all-positive column) would surface. ---
        check(&[1, 5, 100, 7, 3], "all-positive small");
        check(
            &(1..=1000_i32).collect::<Vec<_>>(),
            "all-positive 1000 (1..1000)",
        );
        check(&[42; 257], "all-equal positive, 257 (partial block, odd)");

        // --- mixed-sign, duplicates, monotonic, single, all-equal. ---
        check(&[7], "single positive");
        check(&[-7], "single negative");
        check(&[3, 3, 3, 3, 3], "all-equal");
        check(
            &(-500..=500_i32).collect::<Vec<_>>(),
            "monotonic -500..500 (1001, odd)",
        );
        check(
            &(0..1024_i32).rev().collect::<Vec<_>>(),
            "monotonic decreasing 1023..0",
        );
        check(
            &(0..777_u32)
                .map(|i| i.wrapping_mul(2_654_435_761) as i32)
                .collect::<Vec<_>>(),
            "scrambled 777 (random-ish, odd, partial block)",
        );

        // --- sizes: < one block, exactly one block, partial last block, ODD, rows >> grid*block. ---
        check(&(0..255_i32).collect::<Vec<_>>(), "255 (< one block)");
        check(&(0..256_i32).collect::<Vec<_>>(), "256 (exactly one block)");
        check(
            &(0..257_i32).collect::<Vec<_>>(),
            "257 (one over a block, partial)",
        );
        check(
            &(0..513_i32).collect::<Vec<_>>(),
            "513 (odd, two-block tail)",
        );

        // --- i64 SUM wrap: large negative partials forcing u64 wrap in the tree/atomics across
        // hundreds of thousands of threads; == linear host wrapping_add oracle. Also exercises
        // grid-stride wrap (> 1024*256 = 262144 threads). The min here is i32::MIN (a REAL extreme) and
        // max is i32::MIN too (all equal), so a sentinel leak would surface alongside the wrap. ---
        let neg_n = 2_000_003_usize; // > grid width, odd, partial tail
        check(
            &vec![i32::MIN; neg_n],
            "2_000_003x i32::MIN (i64-wrap + grid-stride + sentinel min/max)",
        );

        // --- grid-stride wrap with large-magnitude alternating extremes (partial tail, odd). min must
        // be i32::MIN and max i32::MAX -- both REAL extremes, scattered across many grid-stride iters. ---
        let big: Vec<i32> = (0..4_000_001_u64)
            .map(|i| {
                if i % 2 == 0 {
                    i32::MAX - (i % 17) as i32
                } else {
                    i32::MIN + (i % 13) as i32
                }
            })
            .collect();
        check(
            &big,
            "4_000_001 alternating near-extremes (grid-stride wrap, partial tail)",
        );

        // --- mixed positive grid-stride-wrap control (no high bit) with a single embedded extreme so
        // min/max are determined by ONE row reached deep in a grid-stride iteration. ---
        let mut spread: Vec<i32> = (0..3_000_007_u64).map(|i| (i % 100) as i32).collect();
        spread[2_500_000] = i32::MAX; // a single real max buried mid-stream
        spread[1_999_999] = i32::MIN; // a single real min buried mid-stream
        check(
            &spread,
            "3M spread with single buried i32::MIN/MAX (grid-stride, partial tail)",
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_filtered_nullable_scalar_stats_byte_identical_to_host_oracle() {
        // SLICE B (the direct (count,sum,min,max) reduction extended with an on-device FILTER + NULL-skip,
        // which replaced the now-removed self-grouped hash and gather-to-host CPU-reduction paths.
        //
        // The load-bearing CLAIM is BYTE-IDENTITY of the SURVIVING-row stats: the direct kernel's
        // (count, sum, min, max) over the rows passing `<col> <cmp> needle` AND the validity bitmap must
        // equal an independent HOST oracle. (The dead self-grouped A/B oracle arm was dropped with the
        // engine-dead `grouped_stats` family; the host oracle still pins the direct kernel exactly.)
        // Focus: NULL-skip (placeholder 0 must never leak), the four comparison codes, negatives, sentinel
        // i32::MIN/MAX as REAL data, all-survivors-NULL (=> count 0 => SQL NULL), and zero-match filter
        // (=> count 0 => SQL NULL). A no-bitmap (`None`) variant covers the filtered NON-nullable path.
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        // Retain a self-grouped resident column (group == value == filter) plus its validity bitmap.
        let retain = |values: &[i32], bitmap: &[u32]| {
            let n = values.len() as u64;
            // SAFETY: i32/u32 are POD; native little-endian bytes match the kernels' ld.global; the slices
            // outlive this synchronous retain.
            let column_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
            };
            let bitmap_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(bitmap.as_ptr().cast::<u8>(), bitmap.len() * 4)
            };
            let header = n.to_le_bytes();
            let header_len = std::mem::size_of::<u64>() as u64;
            let null_offset = header_len + column_bytes.len() as u64;
            let allocated =
                (null_offset + bitmap_bytes.len() as u64).max(std::mem::size_of::<u64>() as u64);
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    allocated,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: header_len,
                            bytes: column_bytes,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: null_offset,
                            bytes: bitmap_bytes,
                        },
                    ],
                )
                .expect("retain resident column + validity bitmap");
            (resident, header_len, null_offset)
        };

        // HOST oracle over the rows surviving the filter AND the validity predicate.
        let host_oracle = |values: &[i32],
                           valid: &dyn Fn(usize) -> bool,
                           keep: &dyn Fn(i32) -> bool|
         -> (u64, i64, i32, i32) {
            let survivors: Vec<i32> = values
                .iter()
                .enumerate()
                .filter(|(i, &v)| valid(*i) && keep(v))
                .map(|(_, &v)| v)
                .collect();
            let count = survivors.len() as u64;
            let sum = survivors
                .iter()
                .fold(0_i64, |acc, &v| acc.wrapping_add(i64::from(v)));
            let min = survivors.iter().copied().min().unwrap_or(i32::MAX);
            let max = survivors.iter().copied().max().unwrap_or(i32::MIN);
            (count, sum, min, max)
        };

        let cmp_keep = |cmp: CudaI32Comparison, needle: i32| -> Box<dyn Fn(i32) -> bool> {
            match cmp {
                CudaI32Comparison::Lt => Box::new(move |v| v < needle),
                CudaI32Comparison::Lte => Box::new(move |v| v <= needle),
                CudaI32Comparison::Gt => Box::new(move |v| v > needle),
                CudaI32Comparison::Gte => Box::new(move |v| v >= needle),
            }
        };

        // ---------- (A) FILTERED + NULLABLE: A/B the direct kernel vs the self-grouped nullable hash path
        // and the host oracle, over a column with NULLs (placeholder 0), negatives, and the i32 sentinels.
        // is_null every 9th row; non-null values include negatives, both sentinels, and zeros-as-real only
        // on NON-null rows would be ambiguous with the placeholder, so non-null values avoid 0.
        let n: usize = 4096 + 137; // > one block, odd tail, two grid-stride iters at clamp
        let is_null = |i: usize| i.is_multiple_of(9);
        let raw: Vec<i32> = (0..n)
            .map(|i| {
                if is_null(i) {
                    0 // NULL placeholder (must be excluded by the bitmap, never counted)
                } else {
                    match i % 11 {
                        0 => i32::MIN,
                        1 => i32::MAX,
                        2 => -1,
                        3 => -1_000_000,
                        4 => 500_000,
                        k => (k as i32) - 5, // small mixed incl negatives, never 0 collides harmlessly
                    }
                }
            })
            .collect();
        let mut bitmap = vec![0u32; n.div_ceil(32)];
        for i in 0..n {
            if !is_null(i) {
                bitmap[i / 32] |= 1u32 << (i % 32);
            }
        }
        let (resident, off, null_off) = retain(&raw, &bitmap);
        let rows = n as u64;
        let valid = |i: usize| !is_null(i);

        for cmp in [
            CudaI32Comparison::Lt,
            CudaI32Comparison::Lte,
            CudaI32Comparison::Gt,
            CudaI32Comparison::Gte,
        ] {
            for &needle in &[i32::MIN, -1_000_000, -1, 0, 500_000, i32::MAX, 7] {
                let keep = cmp_keep(cmp, needle);
                let oracle = host_oracle(&raw, &valid, &keep);

                let direct = resident
                    .filtered_scalar_stats_i32_from_payload(off, rows, needle, cmp, Some(null_off))
                    .expect("direct filtered nullable scalar stats");

                assert_eq!(
                    direct, oracle,
                    "DIRECT filtered+nullable != host oracle (cmp={cmp:?}, needle={needle})"
                );
                if oracle.0 == 0 {
                    // Zero survivors: the engine maps count==0 to SQL NULL; the kernel leaves the init
                    // sentinels (count 0, sum 0, min INT_MAX, max INT_MIN).
                    assert_eq!(direct.0, 0, "zero-survivor count must be 0 (=> SQL NULL)");
                }
                // The NULL placeholder 0 must never be folded in: a `<= 0`-ish filter that WOULD admit the
                // placeholder still excludes it (the bitmap, not the filter, is what drops NULL rows).
                if matches!(cmp, CudaI32Comparison::Gte) && needle <= 0 {
                    let n_null = (0..n).filter(|&i| is_null(i)).count() as u64;
                    let n_nonnull_ge = raw
                        .iter()
                        .enumerate()
                        .filter(|(i, &v)| !is_null(*i) && v >= needle)
                        .count() as u64;
                    assert_eq!(
                        direct.0, n_nonnull_ge,
                        "NULL placeholders must be excluded (n_null={n_null} not counted)"
                    );
                }
            }
        }

        // all-survivors-NULL: a needle that ALL non-null rows fail (here `< i32::MIN` is unsatisfiable) =>
        // zero survivors even ignoring NULLs; AND an all-NULL column => zero survivors via the bitmap.
        let unsat = resident
            .filtered_scalar_stats_i32_from_payload(
                off,
                rows,
                i32::MIN,
                CudaI32Comparison::Lt,
                Some(null_off),
            )
            .expect("unsatisfiable filter");
        assert_eq!(
            unsat.0, 0,
            "v < i32::MIN matches nothing => count 0 => SQL NULL"
        );

        // ---------- (B) FILTERED NON-NULLABLE (no bitmap, `None`): A/B vs the NON-null self-grouped path
        // and host oracle. This is the path that REPLACES the gather-to-host project+CPU reduce.
        let nn: Vec<i32> = (0..2000)
            .map(|i| match i % 13 {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => -i,
                k => i - k * 7,
            })
            .collect();
        let nn_bitmap = vec![0u32; nn.len().div_ceil(32)]; // unused (None path), but retain needs a slice
        let (resident_nn, off_nn, _null_nn) = retain(&nn, &nn_bitmap);
        let nn_rows = nn.len() as u64;
        let all_valid = |_: usize| true;
        for cmp in [
            CudaI32Comparison::Lt,
            CudaI32Comparison::Lte,
            CudaI32Comparison::Gt,
            CudaI32Comparison::Gte,
        ] {
            for &needle in &[i32::MIN, -50, 0, 50, i32::MAX] {
                let keep = cmp_keep(cmp, needle);
                let oracle = host_oracle(&nn, &all_valid, &keep);
                let direct = resident_nn
                    .filtered_scalar_stats_i32_from_payload(off_nn, nn_rows, needle, cmp, None)
                    .expect("direct filtered non-nullable scalar stats");
                assert_eq!(
                    direct, oracle,
                    "DIRECT filtered non-nullable != host oracle (cmp={cmp:?}, needle={needle})"
                );
            }
        }

        // ---------- (C) UNFILTERED NULLABLE (no filter, with bitmap): vs the host oracle -- the
        // `ResidentPredicate::All` nullable arm. Also covers the all-NULL => count 0 => NULL corner.
        let direct_all = resident
            .nullable_scalar_stats_i32_from_payload(off, rows, Some(null_off))
            .expect("direct unfiltered nullable scalar stats");
        let oracle_all = host_oracle(&raw, &valid, &|_| true);
        assert_eq!(
            direct_all, oracle_all,
            "DIRECT unfiltered nullable != host oracle"
        );

        // all-NULL column => zero survivors => count 0 (=> SQL NULL).
        let all_null: Vec<i32> = vec![0; 333];
        let all_null_bitmap = vec![0u32; all_null.len().div_ceil(32)]; // every bit 0 = every row NULL
        let (resident_an, off_an, null_an) = retain(&all_null, &all_null_bitmap);
        let direct_an = resident_an
            .nullable_scalar_stats_i32_from_payload(off_an, all_null.len() as u64, Some(null_an))
            .expect("direct all-null scalar stats");
        assert_eq!(direct_an.0, 0, "all-NULL column => count 0 => SQL NULL");
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_pooled_i32_between_stats_matches_expected() {
        // P2-M2 — between-stats pooled launch-migration correctness gate (GPU-native oracle).
        // The kernel is unchanged (parallel grid-stride count/sum/min/max for the [lo,hi] predicate,
        // atomic-reduced into one 24-byte struct); this gates the migrated launch wiring: the ASYNC
        // H2D of the {count=0, sum=0, min=INT_MAX, max=INT_MIN} init struct into the pooled scratch,
        // the on-stream ordering before the kernel, and the 24-byte scratch readback. Expected values
        // are hand-computed constants (no CPU operator re-implementation).
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        let run = |values: &[i32], lo: i32, hi: i32| -> CudaI32Stats {
            let n = values.len() as u64;
            // SAFETY: `i32` is POD; viewing the Vec as native bytes matches the resident column
            // layout, and `values` outlives the synchronous retain copy.
            let column_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
            };
            let header = n.to_le_bytes();
            let allocated = std::mem::size_of::<u64>() as u64 + column_bytes.len() as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    allocated,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: std::mem::size_of::<u64>() as u64,
                            bytes: column_bytes,
                        },
                    ],
                )
                .expect("retain resident column");
            resident
                .stats_i32_between_from_payload(std::mem::size_of::<u64>() as u64, n, lo, hi)
                .expect("between stats")
        };

        // Small, hand-verified: matching {3,4,5,6}.
        let s = run(&[1, 2, 3, 4, 5, 6, 7, 8], 3, 6);
        assert_eq!((s.count, s.sum, s.min, s.max), (4, 18, Some(3), Some(6)));

        // No match -> count 0, min/max None: exercises the kernel's count==0 guard so the init
        // sentinels are left in the scratch, then mapped to None by the host conversion.
        let s = run(&[1, 2, 3, 4, 5, 6, 7, 8], 100, 200);
        assert_eq!((s.count, s.sum, s.min, s.max), (0, 0, None, None));

        // Multi-block (299_999 = 7*42857 rows, value[i] = i % 7), range [2,5] -> {2,3,4,5} matches
        // every period; exercises the parallel atomic count/sum + min/max reduction across >1000
        // blocks and the grid-stride wrap.
        let q = 42_857_i64;
        let values: Vec<i32> = (0..7 * q).map(|i| (i % 7) as i32).collect();
        let s = run(&values, 2, 5);
        assert_eq!(
            (s.count, s.sum, s.min, s.max),
            (4 * q as u64, 14 * q, Some(2), Some(5))
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_bitonic_argsort_i64_sorts_stably_both_directions() {
        // P2 §9.5/S2 — bitonic argsort primitive correctness + benchmark (GPU-native property oracle).
        // No CPU sort oracle: verify the GPU permutation is (a) a permutation of 0..n, (b) sorted in
        // the requested direction, (c) STABLE (equal keys keep ascending original-index order). Covers
        // both directions across sizes incl. duplicates, negatives, and non-power-of-2 n (padding).
        use std::time::Instant;
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

        let retain_keys = |keys: &[i64]| {
            let n = keys.len() as u64;
            // SAFETY: `i64` is POD; native bytes; `keys` outlives the synchronous retain copy.
            let key_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(keys.as_ptr().cast::<u8>(), keys.len() * 8) };
            let header = n.to_le_bytes();
            let off = std::mem::size_of::<u64>() as u64;
            let allocated = off + key_bytes.len() as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    allocated,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: off,
                            bytes: key_bytes,
                        },
                    ],
                )
                .expect("retain keys");
            // Primitives take an absolute device key pointer; hand back device_ptr()+off.
            let key_ptr = resident.device_ptr() + off;
            (resident, key_ptr)
        };
        let verify = |keys: &[i64], idx: &[u32], descending: bool| {
            assert_eq!(idx.len(), keys.len());
            // (a) permutation of 0..n.
            let mut seen = vec![false; keys.len()];
            for &x in idx {
                assert!(
                    !seen[x as usize],
                    "index {x} appears twice — not a permutation"
                );
                seen[x as usize] = true;
            }
            // (b) sorted in the requested direction, and (c) STABLE across each ENTIRE run of
            // equal keys. Stability is checked NON-ADJACENTLY: within a maximal equal-key run
            // every original index must exceed the run's FIRST index (not merely the previous
            // one), so a violation between two equal rows separated by other equal rows can't
            // slip past. Equal keys are contiguous once (b) holds. Property oracle — no CPU sort.
            let mut run_start = 0usize;
            for p in 1..idx.len() {
                let (prev, cur) = (keys[idx[p - 1] as usize], keys[idx[p] as usize]);
                if descending {
                    assert!(prev >= cur, "not descending at {p}: {prev} < {cur}");
                } else {
                    assert!(prev <= cur, "not ascending at {p}: {prev} > {cur}");
                }
                if cur == prev {
                    assert!(
                        idx[p] > idx[run_start],
                        "not stable on ties: equal-key run [{run_start}..={p}] has idx[{p}]={} <= run-start idx={}",
                        idx[p],
                        idx[run_start]
                    );
                } else {
                    run_start = p;
                }
            }
        };

        for &n in &[1_usize, 2, 7, 100, 1_000, 4_096, 5_000] {
            let keys: Vec<i64> = (0..n)
                .map(|i| (((i as u64).wrapping_mul(2_654_435_761) % 1_000) as i64) - 500)
                .collect();
            for &desc in &[false, true] {
                let (resident, off) = retain_keys(&keys);
                let idx = launch_cuda_resident_i64_argsort_bitonic(&resident, off, n as u64, desc)
                    .expect("argsort");
                verify(&keys, &idx, desc);
            }
        }

        // Extreme-value keys: exercise the padding-sentinel COLLISION path on-device — real
        // keys equal to the ascending sentinel (i64::MAX) / descending sentinel (i64::MIN),
        // which the bounded-key loop above never reaches. Includes the all-sentinel worst
        // cases and duplicate-heavy arrays that stress non-adjacent stability over long runs.
        let (mn, mx) = (i64::MIN, i64::MAX);
        let extreme_cases: Vec<Vec<i64>> = vec![
            vec![mx; 64],                                                 // all == asc sentinel
            vec![mn; 64],                                                 // all == desc sentinel
            vec![mn, mx, 0, mx, mn, 7, mx, mn, -1, mx],                   // mixed, non-pow-2
            (0..300).map(|i| if i % 2 == 0 { mn } else { mx }).collect(), // alternating extremes
            (0..1_000).map(|i| (i % 3) as i64 - 1).collect(),             // dup-heavy: stability
        ];
        for keys in &extreme_cases {
            for &desc in &[false, true] {
                let (resident, off) = retain_keys(keys);
                let idx = launch_cuda_resident_i64_argsort_bitonic(
                    &resident,
                    off,
                    keys.len() as u64,
                    desc,
                )
                .expect("argsort extreme");
                verify(keys, &idx, desc);
            }
        }

        // Benchmark: bitonic latency vs N (informs the S3/S4 radix crossover — host-looped
        // O(log^2 N) per-step launches, so it grows faster than radix's constant passes).
        println!("p2_m2_s2_bitonic_argsort: latency vs N");
        println!("| n | ms |");
        println!("|---:|---:|");
        for &n in &[100_usize, 1_000, 10_000, 100_000, 1_000_000] {
            let keys: Vec<i64> = (0..n)
                .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % n.max(1) as u64) as i64)
                .collect();
            let (resident, off) = retain_keys(&keys);
            let mut ms = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                let idx = launch_cuda_resident_i64_argsort_bitonic(&resident, off, n as u64, false)
                    .unwrap();
                std::hint::black_box(&idx);
                ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("| {n} | {ms:.3} |");
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_radix_argsort_parallel_matches_serial_and_bitonic() {
        // P2 §9.5/S3 — validate the PARALLEL tiled radix against TWO GPU-native oracles: the serial
        // single-thread radix (SAME algorithm → isolates parallelization/scan bugs) and the
        // independently-verified bitonic arm (DIFFERENT algorithm → cross-checks semantics). All
        // three are stable with the SAME ascending-index tie-break, so for identical inputs the
        // returned permutations must be BYTE-IDENTICAL in both directions. GPU-vs-GPU; no CPU oracle.
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
        let retain_keys = |keys: &[i64]| {
            let n = keys.len() as u64;
            // SAFETY: `i64` is POD; native bytes; `keys` outlives the synchronous retain copy.
            let key_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(keys.as_ptr().cast::<u8>(), keys.len() * 8) };
            let header = n.to_le_bytes();
            let off = std::mem::size_of::<u64>() as u64;
            let allocated = off + key_bytes.len() as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    allocated,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: off,
                            bytes: key_bytes,
                        },
                    ],
                )
                .expect("retain keys");
            // Primitives take an absolute device key pointer; hand back device_ptr()+off.
            let key_ptr = resident.device_ptr() + off;
            (resident, key_ptr)
        };

        let mut cases: Vec<Vec<i64>> = Vec::new();
        for &n in &[1_usize, 2, 7, 16, 17, 100, 1_000, 4_096, 5_000] {
            cases.push(
                (0..n)
                    .map(|i| (((i as u64).wrapping_mul(2_654_435_761) % 1_000) as i64) - 500)
                    .collect(),
            );
        }
        let (mn, mx) = (i64::MIN, i64::MAX);
        cases.push(vec![mx; 64]); // all == asc sentinel
        cases.push(vec![mn; 64]); // all == desc sentinel
        cases.push(vec![mn, mx, 0, mx, mn, 7, mx, mn, -1, mx]); // mixed, non-pow-2
        cases.push((0..300).map(|i| if i % 2 == 0 { mn } else { mx }).collect());
        cases.push((0..1_000).map(|i| (i % 3) as i64 - 1).collect()); // dup-heavy

        for keys in &cases {
            let n = keys.len() as u64;
            for &desc in &[false, true] {
                let (resident, off) = retain_keys(keys);
                let parallel = launch_cuda_resident_i64_argsort_radix(&resident, off, n, desc)
                    .expect("parallel radix");
                let serial = launch_cuda_resident_i64_argsort_radix_serial(&resident, off, n, desc)
                    .expect("serial radix");
                let bitonic = launch_cuda_resident_i64_argsort_bitonic(&resident, off, n, desc)
                    .expect("bitonic");
                assert_eq!(
                    parallel, serial,
                    "parallel radix != serial radix for n={n} descending={desc}"
                );
                assert_eq!(
                    parallel, bitonic,
                    "parallel radix != bitonic for n={n} descending={desc}"
                );
            }
        }

        // Large-n parity crossing the multi-tile scan boundary: the scan handles 1024 entries per
        // tile, so its cross-tile carry chain only engages once 16*G > 1024, i.e. G > 64, i.e.
        // n > 64*chunk = 131072. These sizes also force many grid-stride waves per block. Validated
        // against the (fast, parallel) bitonic arm; the single-thread serial oracle is too slow here.
        for &n in &[131_073_usize, 200_000, 300_000] {
            let keys: Vec<i64> = (0..n)
                .map(|i| {
                    (((i as u64).wrapping_mul(11_400_714_819_323_198_485) >> 33) as i64)
                        - (n as i64 / 2)
                })
                .collect();
            let nn = n as u64;
            for &desc in &[false, true] {
                let (resident, off) = retain_keys(&keys);
                let parallel = launch_cuda_resident_i64_argsort_radix(&resident, off, nn, desc)
                    .expect("parallel radix large-n");
                let bitonic = launch_cuda_resident_i64_argsort_bitonic(&resident, off, nn, desc)
                    .expect("bitonic large-n");
                assert_eq!(
                    parallel, bitonic,
                    "parallel radix != bitonic for large n={n} descending={desc} (multi-tile scan)"
                );
            }
        }

        // Benchmark: parallel radix vs the bitonic arm (S2) vs the serial radix oracle, across N.
        // This (a) proves the parallel radix beats its serial oracle, and (b) locates the bitonic
        // -> radix CROSSOVER that the S4 adaptive dispatch will use. Serial is single-thread, so it
        // is only timed up to 10k (it would take seconds at 1M).
        use std::time::Instant;
        let bench = |f: &dyn Fn() -> Vec<u32>| -> f64 {
            let mut ms = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                let out = f();
                std::hint::black_box(&out);
                ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
            }
            ms
        };
        println!("p2_m2_s3_radix_argsort: latency vs N (ascending, best-of-3)");
        println!("| n | bitonic ms | radix ms | serial ms | radix vs bitonic |");
        println!("|---:|---:|---:|---:|---:|");
        for &n in &[100_usize, 1_000, 10_000, 100_000, 1_000_000, 10_000_000] {
            let keys: Vec<i64> = (0..n)
                .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % n.max(1) as u64) as i64)
                .collect();
            let (resident, off) = retain_keys(&keys);
            let nn = n as u64;
            let bitonic_ms = bench(&|| {
                launch_cuda_resident_i64_argsort_bitonic(&resident, off, nn, false).unwrap()
            });
            let radix_ms = bench(&|| {
                launch_cuda_resident_i64_argsort_radix(&resident, off, nn, false).unwrap()
            });
            let serial_str = if n <= 10_000 {
                let serial_ms = bench(&|| {
                    launch_cuda_resident_i64_argsort_radix_serial(&resident, off, nn, false)
                        .unwrap()
                });
                format!("{serial_ms:.3}")
            } else {
                "-".to_string()
            };
            let speedup = bitonic_ms / radix_ms;
            println!("| {n} | {bitonic_ms:.3} | {radix_ms:.3} | {serial_str} | {speedup:.2}x |");
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_adaptive_argsort_dispatches_and_matches_reference() {
        // P2 §9.5/S4 — the adaptive dispatch must return a CORRECT stable argsort on both sides of
        // the bitonic<->radix crossover. Both arms are already proven byte-identical (S3 gate), so
        // validating adaptive == bitonic across sizes straddling the threshold confirms the dispatch
        // picks a correct arm each side. GPU-vs-GPU; no CPU oracle.
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
        let retain_keys = |keys: &[i64]| {
            let n = keys.len() as u64;
            // SAFETY: `i64` is POD; native bytes; `keys` outlives the synchronous retain copy.
            let key_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(keys.as_ptr().cast::<u8>(), keys.len() * 8) };
            let header = n.to_le_bytes();
            let off = std::mem::size_of::<u64>() as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    off + key_bytes.len() as u64,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: off,
                            bytes: key_bytes,
                        },
                    ],
                )
                .expect("retain keys");
            // Primitives take an absolute device key pointer; hand back device_ptr()+off.
            let key_ptr = resident.device_ptr() + off;
            (resident, key_ptr)
        };

        // Sizes straddling ADAPTIVE_SORT_CROSSOVER_ROWS (10_000): below -> bitonic arm, at/above ->
        // radix arm. Include the exact boundary and ±1.
        for &n in &[100_usize, 9_999, 10_000, 10_001, 50_000] {
            let keys: Vec<i64> = (0..n)
                .map(|i| (((i as u64).wrapping_mul(2_654_435_761) % 4_096) as i64) - 2_048)
                .collect();
            let nn = n as u64;
            for &desc in &[false, true] {
                let (resident, off) = retain_keys(&keys);
                let adaptive = launch_cuda_resident_i64_argsort_adaptive(&resident, off, nn, desc)
                    .expect("adaptive argsort");
                let reference = launch_cuda_resident_i64_argsort_bitonic(&resident, off, nn, desc)
                    .expect("bitonic reference");
                assert_eq!(
                    adaptive, reference,
                    "adaptive argsort != bitonic reference for n={n} descending={desc}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_having_filter_dnf_matches_construction() {
        // P2 §9.5/S4 — GPU HAVING DNF filter. CONSTRUCTION oracle (no CPU filter re-impl): lay out
        // group[r] = 1000+r and agg[r] = r, so each predicate's survivors are a known arithmetic
        // range/union and the col selector is unambiguous (group values 1000.. never overlap agg
        // values 0..n). Covers all five ops, both columns, AND within a clause, OR across clauses,
        // mixed-column clauses, empty/all results, and n spanning multiple grid-stride waves.
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
        let retain = |group: &[i64], agg: &[i64]| {
            let n = group.len();
            // SAFETY: i64 is POD; native little-endian bytes; the slices outlive the retain copy.
            let group_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(group.as_ptr().cast::<u8>(), n * 8) };
            let agg_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(agg.as_ptr().cast::<u8>(), n * 8) };
            let agg_off = (n * 8) as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    (n * 16) as u64,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: group_bytes,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: agg_off,
                            bytes: agg_bytes,
                        },
                    ],
                )
                .expect("retain group+agg");
            (resident, agg_off)
        };

        // op encoding: 0=Eq 1=Lt 2=Lte 3=Gt 4=Gte ; col 0=group, 1=agg.
        for &n in &[10_usize, 256, 257, 1_000, 5_000] {
            let group: Vec<i64> = (0..n as i64).map(|r| 1_000 + r).collect();
            let agg: Vec<i64> = (0..n as i64).collect();
            let (resident, agg_off) = retain(&group, &agg);
            let nn = n as u64;
            let run = |clauses: &[Vec<(u32, u32, i64)>]| {
                launch_cuda_resident_having_filter(
                    &resident,
                    resident.device_ptr(),
                    resident.device_ptr() + agg_off,
                    nn,
                    clauses,
                )
                .expect("having filter")
            };
            let range = |lo: usize, hi: usize| -> Vec<u32> { (lo..hi).map(|r| r as u32).collect() };
            let (k, a, b) = ((n / 3) as i64, (n / 4) as i64, (3 * n / 4) as i64);

            // agg > K (Gt) -> (K, n)
            assert_eq!(
                run(&[vec![(1, 3, k)]]),
                range(k as usize + 1, n),
                "agg>K n={n}"
            );
            // agg <= K (Lte) -> [0, K]
            assert_eq!(
                run(&[vec![(1, 2, k)]]),
                range(0, k as usize + 1),
                "agg<=K n={n}"
            );
            // agg >= A AND agg < B (Gte, Lt) -> [A, B)
            assert_eq!(
                run(&[vec![(1, 4, a), (1, 1, b)]]),
                range(a as usize, b as usize),
                "A<=agg<B n={n}"
            );
            // group == 1000+target (Eq, col 0) -> single row `target`
            let target = (n / 2) as i64;
            assert_eq!(
                run(&[vec![(0, 0, 1_000 + target)]]),
                vec![target as u32],
                "group==v n={n}"
            );
            // (group < 1000+A) OR (agg >= B) -> [0, A) ∪ [B, n)  (mixed cols, OR)
            let mut expect_or = range(0, a as usize);
            expect_or.extend(range(b as usize, n));
            assert_eq!(
                run(&[vec![(0, 1, 1_000 + a)], vec![(1, 4, b)]]),
                expect_or,
                "group<A OR agg>=B n={n}"
            );
            // group >= 1000+A AND agg < B -> [A, B)  (mixed cols, AND)
            assert_eq!(
                run(&[vec![(0, 4, 1_000 + a), (1, 1, b)]]),
                range(a as usize, b as usize),
                "group>=A AND agg<B n={n}"
            );
            // empty (agg > n+10) and all (agg >= 0)
            assert!(
                run(&[vec![(1, 3, n as i64 + 10)]]).is_empty(),
                "empty n={n}"
            );
            assert_eq!(run(&[vec![(1, 4, 0)]]), range(0, n), "all n={n}");
            // deep AND (3 filters): agg > AA AND agg < BB AND group >= 1000+CC -> [CC, BB)
            let (aa, cc, bb) = ((n / 5) as i64, (2 * n / 5) as i64, (3 * n / 5) as i64);
            assert_eq!(
                run(&[vec![(1, 3, aa), (1, 1, bb), (0, 4, 1_000 + cc)]]),
                range(cc as usize, bb as usize),
                "deep-AND n={n}"
            );
            // three OR clauses: agg==1 OR agg==n/2 OR agg==n-1 -> {1, n/2, n-1} ascending
            assert_eq!(
                run(&[
                    vec![(1, 0, 1)],
                    vec![(1, 0, (n / 2) as i64)],
                    vec![(1, 0, n as i64 - 1)],
                ]),
                vec![1_u32, (n / 2) as u32, n as u32 - 1],
                "three-OR n={n}"
            );
            // vacuous: an empty clause (vacuous AND) matches all; OR short-circuits to all
            assert_eq!(run(&[vec![]]), range(0, n), "vacuous-empty-clause n={n}");
            assert_eq!(
                run(&[vec![], vec![(1, 3, n as i64 + 10)]]),
                range(0, n),
                "vacuous-clause-OR n={n}"
            );
        }

        // Negatives, i64 extremes, and scattered (non-contiguous) survivors — the construction loop
        // above only uses distinct/monotone non-negative data, so cover the auditor-flagged paths.
        let mk_range = |lo: usize, hi: usize| -> Vec<u32> { (lo..hi).map(|r| r as u32).collect() };
        {
            // negatives: agg[r] = r - 500 over n = 1000
            let n = 1_000_usize;
            let group: Vec<i64> = (0..n as i64).collect();
            let agg: Vec<i64> = (0..n as i64).map(|r| r - 500).collect();
            let (resident, agg_off) = retain(&group, &agg);
            let run = |c: &[Vec<(u32, u32, i64)>]| {
                launch_cuda_resident_having_filter(
                    &resident,
                    resident.device_ptr(),
                    resident.device_ptr() + agg_off,
                    n as u64,
                    c,
                )
                .expect("having neg")
            };
            assert_eq!(run(&[vec![(1, 4, 0)]]), mk_range(500, 1_000), "neg agg>=0");
            assert_eq!(run(&[vec![(1, 2, -1)]]), mk_range(0, 500), "neg agg<=-1");
            assert_eq!(
                run(&[vec![(1, 3, -250), (1, 1, 250)]]),
                mk_range(251, 750),
                "neg -250<agg<250"
            );
        }
        {
            // i64 extremes (explicit dataset)
            let group: Vec<i64> = vec![0, 1, 2, 3, 4];
            let agg: Vec<i64> = vec![i64::MIN, -1, 0, 1, i64::MAX];
            let (resident, agg_off) = retain(&group, &agg);
            let run = |c: &[Vec<(u32, u32, i64)>]| {
                launch_cuda_resident_having_filter(
                    &resident,
                    resident.device_ptr(),
                    resident.device_ptr() + agg_off,
                    5,
                    c,
                )
                .expect("having ext")
            };
            assert_eq!(run(&[vec![(1, 4, 0)]]), vec![2_u32, 3, 4], "ext agg>=0");
            assert_eq!(run(&[vec![(1, 2, 0)]]), vec![0_u32, 1, 2], "ext agg<=0");
            assert_eq!(
                run(&[vec![(1, 3, i64::MIN)]]),
                vec![1_u32, 2, 3, 4],
                "ext agg>MIN"
            );
            assert_eq!(run(&[vec![(1, 0, i64::MAX)]]), vec![4_u32], "ext agg==MAX");
            assert_eq!(
                run(&[vec![(1, 1, i64::MAX), (1, 3, i64::MIN)]]),
                vec![1_u32, 2, 3],
                "ext MIN<agg<MAX"
            );
        }
        {
            // scattered (non-contiguous) survivors: agg[r] = r % 2 over n = 100
            let n = 100_usize;
            let group: Vec<i64> = (0..n as i64).collect();
            let agg: Vec<i64> = (0..n as i64).map(|r| r % 2).collect();
            let (resident, agg_off) = retain(&group, &agg);
            let run = |c: &[Vec<(u32, u32, i64)>]| {
                launch_cuda_resident_having_filter(
                    &resident,
                    resident.device_ptr(),
                    resident.device_ptr() + agg_off,
                    n as u64,
                    c,
                )
                .expect("having scat")
            };
            let odds: Vec<u32> = (0..n / 2).map(|k| (2 * k + 1) as u32).collect();
            let evens: Vec<u32> = (0..n / 2).map(|k| (2 * k) as u32).collect();
            assert_eq!(run(&[vec![(1, 0, 1)]]), odds, "scattered odd rows");
            assert_eq!(run(&[vec![(1, 0, 0)]]), evens, "scattered even rows");
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_order_by_limit_windows_the_sorted_keys() {
        // P2 §9.5/S4 — ORDER BY ... LIMIT/OFFSET. CONSTRUCTION oracle: keys[r] = r, so the ascending
        // order is [0,1,..,n-1] and descending is [n-1,..,0]; each (offset, limit) window is a known
        // slice. Also checks the offset/limit clamps. GPU sort + host window; no CPU sort oracle.
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
        let retain_keys = |keys: &[i64]| {
            let n = keys.len() as u64;
            // SAFETY: i64 is POD; native bytes; `keys` outlives the synchronous retain copy.
            let key_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(keys.as_ptr().cast::<u8>(), keys.len() * 8) };
            let header = n.to_le_bytes();
            let off = std::mem::size_of::<u64>() as u64;
            let resident = runtime
                .retain_device_memory_chunks(
                    0,
                    off + key_bytes.len() as u64,
                    &[
                        CudaDeviceMemoryChunk {
                            byte_offset: 0,
                            bytes: &header,
                        },
                        CudaDeviceMemoryChunk {
                            byte_offset: off,
                            bytes: key_bytes,
                        },
                    ],
                )
                .expect("retain keys");
            // Primitives take an absolute device key pointer; hand back device_ptr()+off.
            let key_ptr = resident.device_ptr() + off;
            (resident, key_ptr)
        };

        let n = 100_usize;
        let keys: Vec<i64> = (0..n as i64).collect();
        let (resident, off) = retain_keys(&keys);
        let nn = n as u64;
        let run = |desc: bool, offset: u64, limit: Option<u64>| {
            launch_cuda_resident_i64_order_by_limit(&resident, off, nn, desc, offset, limit)
                .expect("order_by_limit")
        };

        // ascending order is [0, n): windows are contiguous slices.
        assert_eq!(run(false, 0, Some(5)), vec![0_u32, 1, 2, 3, 4]);
        assert_eq!(run(false, 10, Some(3)), vec![10_u32, 11, 12]);
        assert_eq!(run(false, 95, None), vec![95_u32, 96, 97, 98, 99]);
        assert_eq!(run(false, 0, None), (0..n as u32).collect::<Vec<_>>());
        // descending order is [n-1, .., 0].
        assert_eq!(run(true, 0, Some(3)), vec![99_u32, 98, 97]);
        assert_eq!(run(true, 2, Some(2)), vec![97_u32, 96]);
        // clamps: offset past the end -> empty; limit past the end -> clamped.
        assert!(run(false, 200, Some(5)).is_empty());
        assert_eq!(run(false, 98, Some(10)), vec![98_u32, 99]);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn gpu_s9_5_scaling_benchmarks() {
        // §9.5 scaling benchmarks for every GPU path: the argsort primitives (bitonic / radix /
        // adaptive / order_by_limit), the HAVING DNF filter, and the end-to-end resident grouped
        // pipeline (hash-agg + ORDER BY <col> [+ HAVING] + window). Best-of-3.
        use std::time::Instant;
        let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
        let bench = |f: &dyn Fn()| -> f64 {
            let mut ms = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                f();
                ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
            }
            ms
        };
        let chunk = |off: u64, bytes: &[u8]| CudaDeviceMemoryChunk {
            byte_offset: off,
            bytes: unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) },
        };
        let as_bytes_i64 = |v: &[i64]| unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        };

        // ---- 1. argsort primitives: latency vs N keys (i64) ----
        println!("\n### argsort primitives (i64 keys) latency vs N (ms, best-of-3)");
        println!("| N | bitonic | radix | adaptive | order_by_limit(LIMIT 100) |");
        println!("|---:|---:|---:|---:|---:|");
        for &n in &[1_000_usize, 10_000, 100_000, 1_000_000, 10_000_000] {
            let keys: Vec<i64> = (0..n)
                .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % n as u64) as i64)
                .collect();
            let header = (n as u64).to_le_bytes();
            let res = runtime
                .retain_device_memory_chunks(
                    0,
                    8 + (n * 8) as u64,
                    &[chunk(0, &header), chunk(8, as_bytes_i64(&keys))],
                )
                .expect("retain keys");
            let kp = res.device_ptr() + 8;
            let nn = n as u64;
            let bit = bench(&|| {
                launch_cuda_resident_i64_argsort_bitonic(&res, kp, nn, false).unwrap();
            });
            let rad = bench(&|| {
                launch_cuda_resident_i64_argsort_radix(&res, kp, nn, false).unwrap();
            });
            let adp = bench(&|| {
                launch_cuda_resident_i64_argsort_adaptive(&res, kp, nn, false).unwrap();
            });
            let obl = bench(&|| {
                launch_cuda_resident_i64_order_by_limit(&res, kp, nn, false, 0, Some(100)).unwrap();
            });
            println!("| {n} | {bit:.3} | {rad:.3} | {adp:.3} | {obl:.3} |");
        }

        // ---- 2. HAVING DNF filter: latency vs M groups (i64 group + agg) ----
        println!("\n### HAVING DNF filter (single-block) latency vs M groups (ms, best-of-3)");
        println!("| M (groups) | HAVING agg > M/2 |");
        println!("|---:|---:|");
        for &m in &[1_000_usize, 10_000, 100_000, 1_000_000] {
            let col: Vec<i64> = (0..m as i64).collect();
            let header = (m as u64).to_le_bytes();
            let go = 8_u64;
            let ao = 8 + (m * 8) as u64;
            let res = runtime
                .retain_device_memory_chunks(
                    0,
                    8 + (m * 16) as u64,
                    &[
                        chunk(0, &header),
                        chunk(go, as_bytes_i64(&col)),
                        chunk(ao, as_bytes_i64(&col)),
                    ],
                )
                .expect("retain group+agg");
            let (gp, ap) = (res.device_ptr() + go, res.device_ptr() + ao);
            let clauses = vec![vec![(1_u32, 3_u32, (m / 2) as i64)]];
            let h = bench(&|| {
                launch_cuda_resident_having_filter(&res, gp, ap, m as u64, &clauses).unwrap();
            });
            println!("| {m} | {h:.3} |");
        }
        // NOTE: the resident GROUPED PIPELINE benchmark (hash-agg + ORDER BY + HAVING + window) was
        // removed with the engine-dead `grouped_stats` hash-agg family. The LIVE per-group path is the
        // two-level kernel (`group_by_i32_count_sum_minmax_from_payload`), benchmarked by the
        // `grouped_cardinality_probe` / `read_kernel_roofline` examples.
    }
