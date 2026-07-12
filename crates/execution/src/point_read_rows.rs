use std::ffi::c_void;

use super::{
    check_cuda, copy_pinned_into, launch_on_pooled_stream, stage_result_dtoh_async,
    CudaResidentDeviceMemory, CudaRuntimeProbeError, GpuPrimaryContext, PooledStream,
};

pub(super) fn launch_cuda_resident_i32_equal_row_indices(
    resident: &CudaResidentDeviceMemory,
    filters: &[(u64, i32)],
    row_count: u64,
) -> Result<Vec<u64>, CudaRuntimeProbeError> {
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
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

    const MAX_FILTERS: usize = 4;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_equal_row_indices(
    .param .u64 resident_ptr,
    .param .u64 row_count,
    .param .u32 filter_count,
    .param .u64 offset0,
    .param .u64 offset1,
    .param .u64 offset2,
    .param .u64 offset3,
    .param .s32 needle0,
    .param .s32 needle1,
    .param .s32 needle2,
    .param .s32 needle3,
    .param .u64 out_indices_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p_out;
    .reg .pred %p_done;
    .reg .pred %p_match;
    .reg .pred %p_check;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %idx32;
    .reg .u64 %idx;
    .reg .u64 %rows;
    .reg .u32 %filters;
    .reg .u64 %resident;
    .reg .u64 %offset0;
    .reg .u64 %offset1;
    .reg .u64 %offset2;
    .reg .u64 %offset3;
    .reg .s32 %needle0;
    .reg .s32 %needle1;
    .reg .s32 %needle2;
    .reg .s32 %needle3;
    .reg .u64 %out_indices;
    .reg .u64 %out_count;
    .reg .u64 %row_byte;
    .reg .u64 %addr;
    .reg .u32 %slot;
    .reg .u64 %out_addr;
    .reg .u64 %slot64;
    .reg .u32 %one;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %rows, [row_count];
    ld.param.u32 %filters, [filter_count];
    ld.param.u64 %offset0, [offset0];
    ld.param.u64 %offset1, [offset1];
    ld.param.u64 %offset2, [offset2];
    ld.param.u64 %offset3, [offset3];
    ld.param.s32 %needle0, [needle0];
    ld.param.s32 %needle1, [needle1];
    ld.param.s32 %needle2, [needle2];
    ld.param.s32 %needle3, [needle3];
    ld.param.u64 %out_indices, [out_indices_ptr];
    ld.param.u64 %out_count, [out_count_ptr];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %idx32, %r_block, %r_block_dim, %r_tid;
    cvt.u64.u32 %idx, %idx32;

    setp.ge.u64 %p_out, %idx, %rows;
    @%p_out bra DONE;

    setp.eq.u32 %p_done, %filters, 0;
    @%p_done bra DONE;

    mul.lo.u64 %row_byte, %idx, 4;

    add.u64 %addr, %resident, %offset0;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle0;
    @!%p_match bra DONE;

    setp.le.u32 %p_check, %filters, 1;
    @%p_check bra MATCHED;
    add.u64 %addr, %resident, %offset1;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle1;
    @!%p_match bra DONE;

    setp.le.u32 %p_check, %filters, 2;
    @%p_check bra MATCHED;
    add.u64 %addr, %resident, %offset2;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle2;
    @!%p_match bra DONE;

    setp.le.u32 %p_check, %filters, 3;
    @%p_check bra MATCHED;
    add.u64 %addr, %resident, %offset3;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle3;
    @!%p_match bra DONE;

MATCHED:
    mov.u32 %one, 1;
    atom.global.add.u32 %slot, [%out_count], %one;
    cvt.u64.u32 %slot64, %slot;
    mul.lo.u64 %out_addr, %slot64, 8;
    add.u64 %out_addr, %out_indices, %out_addr;
    st.global.u64 [%out_addr], %idx;

DONE:
    ret;
}
"#;

    if filters.is_empty() || filters.len() > MAX_FILTERS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(filters.len()));
    }
    if row_count == 0 {
        return Ok(Vec::new());
    }
    for (byte_offset, _) in filters {
        let bytes = row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .and_then(|bytes| byte_offset.checked_add(bytes))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if bytes > resident.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
        }
    }
    let row_count_u32 = u32::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<u64>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    // P2-M2 (row-index-gather async lever): this gather route had the same default/NULL-stream-
    // serialized wall as the other resident routes — a per-call `cuMemAlloc` (indices + count) +
    // `cuMemFree`, a blocking default-stream `cuMemsetD8` to zero the atomic-append counter, a
    // per-call `cuModuleLoadData` re-JIT, a whole-context sync (the legacy event-timing helper
    // falls back to `cuCtxSynchronize`, and even its timed path records events on the NULL stream),
    // and two blocking default-stream `cuMemcpyDtoH` (the count, then the indices). The driver
    // serializes those memory ops + the JIT + the whole-context sync across concurrent readers, so
    // a per-section breakdown localized ~73% of the engine's c64 multi-column cascade wall to this
    // single launch (~20.7 ms/call @c64) — no scaling. Unlike the serial compare/range kernels,
    // THIS kernel is already parallel (one thread per row + an `atom.global.add` append), so the
    // wall is purely the per-call orchestration, not the kernel — removing the serializer should
    // restore concurrency. The fix moves the counter-zero (memset), the kernel, and both D2H onto
    // the route's pooled private stream via the `*Async` variants behind exactly TWO
    // `cuStreamSynchronize` (one after the kernel so the device-computed match count is readable to
    // size the indices read; one after the indices D2H), stages both readbacks through pooled
    // pinned (page-locked) host buffers, leases its device output buffers + private stream + module
    // from the shared pools/cache (no per-call alloc/JIT), and removes the whole-context sync. No
    // HtoD is needed: the kernel takes the offsets/needles as scalar params; the count buffer is
    // the only region that must be zeroed (the indices buffer is read back only over [0, count) so
    // pooled stale bytes are never observed). The PTX kernel + grid shape are byte-for-byte
    // unchanged, so the appended indices — and thus the result bytes — are identical.
    //
    // The whole async path is gated on the optional async + pinned-host driver symbols; on an old
    // driver lacking them the route keeps a blocking path (still cached-module + pooled-buffer +
    // pooled-stream, just blocking memset/D2H), so correctness is unconditional and only the
    // acceleration is best-effort.
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let async_ops = match (
        resident.primary().cu_memcpy_dtoh_async,
        resident.primary().cu_memset_d8_async,
    ) {
        (Some(dtoh), Some(memset)) => Some((dtoh, memset)),
        _ => None,
    };

    // Pooled device output buffers (no per-call cuMemAlloc/cuMemFree). Reused buffers are NOT
    // zeroed; only the atomic-append counter is memset, and the indices buffer is read back only
    // over the [0, count) prefix — so stale bytes are never observed.
    let indices_guard = resident
        .primary()
        .lease_device_buffer(output_bytes.max(1))?;
    const COUNT_BYTES: usize = std::mem::size_of::<u32>();
    let count_guard = resident.primary().lease_device_buffer(COUNT_BYTES)?;

    // P2-M2: cached module (no per-launch cuModuleLoadData) — keyed by the kernel entry name and
    // launched concurrently on distinct streams.
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_equal_row_indices", &ptx)?;

    let mut resident_arg = resident.device_ptr();
    let mut rows_arg = row_count;
    let mut filter_count_arg = u32::try_from(filters.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(filters.len()))?;
    let mut offsets = [0_u64; MAX_FILTERS];
    let mut needles = [0_i32; MAX_FILTERS];
    for (idx, (offset, needle)) in filters.iter().enumerate() {
        offsets[idx] = *offset;
        needles[idx] = *needle;
    }
    let mut output_arg = indices_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut filter_count_arg as *mut u32).cast::<c_void>(),
        (&mut offsets[0] as *mut u64).cast::<c_void>(),
        (&mut offsets[1] as *mut u64).cast::<c_void>(),
        (&mut offsets[2] as *mut u64).cast::<c_void>(),
        (&mut offsets[3] as *mut u64).cast::<c_void>(),
        (&mut needles[0] as *mut i32).cast::<c_void>(),
        (&mut needles[1] as *mut i32).cast::<c_void>(),
        (&mut needles[2] as *mut i32).cast::<c_void>(),
        (&mut needles[3] as *mut i32).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count_u32.div_ceil(threads_per_block);

    // This kernel is already parallel (one thread per row + an `atom.global.add` ordered append);
    // its grid shape is preserved exactly so the appended order — and thus the result bytes — is
    // identical (the append order is the atomic schedule, which the grid shape does not change).
    if let Some((dtoh_async, memset_async)) = async_ops {
        // ---- async-on-pooled-stream path (the lever) ----
        // Bind the shared primary context (idempotent) and lease the pooled private stream.
        resident.primary().set_current()?;
        struct StreamLease<'a> {
            primary: &'a GpuPrimaryContext,
            pooled: Option<PooledStream>,
        }
        impl Drop for StreamLease<'_> {
            fn drop(&mut self) {
                if let Some(pooled) = self.pooled.take() {
                    self.primary.release_pooled_stream(pooled);
                }
            }
        }
        let lease = StreamLease {
            primary: resident.primary(),
            pooled: Some(resident.primary().acquire_pooled_stream()?),
        };
        let pooled = lease.pooled.as_ref().expect("pooled stream just set");
        let stream = pooled.stream;
        let timed = !pooled.start_event.is_null() && !pooled.stop_event.is_null();

        // HARDENING (error-path stream drain): once an async op is enqueued on this private
        // stream, an early `?` would unwind the locals — returning the device + pinned buffer
        // leases to their shared pools while enqueued ops may still be in flight, a
        // use-after-free window for whoever leases those buffers next. So every fallible op from
        // the first async enqueue through each covering `cuStreamSynchronize` propagates its error
        // through `drain_err`, a best-effort blocking sync (result ignored) that drains the stream
        // FIRST, *then* yields the original error. `map_err` runs the closure at the error site
        // BEFORE `?` returns and hence before ANY local Drop, so it precedes the release of every
        // lease regardless of scope; on `Ok` the closure is not invoked, so the success path adds
        // nothing beyond the two explicit syncs below.
        let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
            // SAFETY: `stream` is the live pooled private stream; a blocking synchronize on it is
            // valid from this thread (the primary context is current). The result is intentionally
            // ignored — this is a best-effort drain on an already-failing path.
            unsafe {
                let _ = (resident.primary().cu_stream_synchronize)(stream);
            }
            err
        };

        // (1) Stream-ordered zero of the atomic-append counter, then the kernel launch (no HtoD
        // needed for this route — the offsets/needles are scalar params).
        check_cuda(unsafe { memset_async(count_guard.ptr, 0, COUNT_BYTES, stream) })
            .map_err(drain_err)?;
        if timed {
            check_cuda(unsafe { (resident.primary().cu_event_record)(pooled.start_event, stream) })
                .map_err(drain_err)?;
        }
        check_cuda(unsafe {
            cu_launch_kernel(
                function,
                blocks,
                1,
                1,
                threads_per_block,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;
        if timed {
            check_cuda(unsafe { (resident.primary().cu_event_record)(pooled.stop_event, stream) })
                .map_err(drain_err)?;
        }

        // (2) Stream-ordered read of the 4-byte match count, then sync #1: now the kernel is
        // complete and the count is known, so the indices read can be sized. The count stages
        // through a small pooled pinned host buffer for a truly-async DMA.
        let mut match_count = 0_u32;
        let count_pinned = resident.primary().lease_pinned_host_buffer(COUNT_BYTES);
        let count_dst: *mut c_void = count_pinned
            .as_ref()
            .map(|p| p.ptr)
            .unwrap_or_else(|| (&mut match_count as *mut u32).cast::<c_void>());
        check_cuda(unsafe { dtoh_async(count_dst, count_guard.ptr, COUNT_BYTES, stream) })
            .map_err(drain_err)?;
        // Covering sync #1: drains on its own error too (the count D2H is still enqueued).
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        if let Some(pinned) = &count_pinned {
            // SAFETY: the sync above completed the 4-byte D2H into the pinned region; read the
            // u32 out by typed pointer (pinned host memory is page-aligned).
            unsafe {
                match_count = pinned.ptr.cast::<u32>().read();
            }
        }
        if timed {
            let mut elapsed_ms = 0.0_f32;
            check_cuda(unsafe {
                (resident.primary().cu_event_elapsed_time)(
                    &mut elapsed_ms,
                    pooled.start_event,
                    pooled.stop_event,
                )
            })?;
            resident.record_kernel_event_elapsed_us(Some(
                (f64::from(elapsed_ms) * 1_000.0).ceil() as u64
            ));
        } else {
            resident.record_kernel_event_elapsed_us(None);
        }

        let match_count = u64::from(match_count);
        if match_count > row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let match_count_usize = usize::try_from(match_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

        // (3) Stream-ordered result D2H of the populated [0, count) indices prefix into a pooled
        // pinned host buffer, then ONE sync #2; copy the pinned bytes into the owned Vec.
        let mut indices = vec![0_u64; match_count_usize];
        let indices_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            indices_guard.ptr,
            &mut indices,
        )
        .map_err(drain_err)?;
        // Covering sync #2: drains on its own error too (the indices D2H is still enqueued).
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        copy_pinned_into(&indices_pinned, &mut indices);
        drop(lease);
        // Correctness (deterministic ascending order): the kernel appends matching row indices in
        // `atom.global.add` SCHEDULE order, which is ascending only when all matches land in a
        // single warp (<=32). For >32 matches the append order is non-deterministic across warps,
        // so a positional gather would emit rows in a non-deterministic order that diverges from
        // the CPU/non-resident reference (ascending) and breaks the partitioned ascending-merge.
        // Sort host-side over [0, count) so the route always returns ascending indices identical to
        // the reference. Cost is O(k log k) host-side, dominated by the per-row D2H gather above.
        indices.sort_unstable();
        Ok(indices)
    } else {
        // ---- legacy blocking fallback (old driver: no async/pinned symbols) ----
        // Still cached-module + pooled-buffer + pooled-stream (per-stream sync, no whole-context
        // cuCtxSynchronize); just blocking memset/D2H. The route owns its output buffers, so no
        // pooled scratch is requested.
        check_cuda(unsafe { cu_memset_d8(count_guard.ptr, 0, COUNT_BYTES) })?;
        launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
            cu_launch_kernel(
                function,
                blocks,
                1,
                1,
                threads_per_block,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;

        let mut match_count = 0_u32;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                (&mut match_count as *mut u32).cast::<c_void>(),
                count_guard.ptr,
                COUNT_BYTES,
            )
        })?;
        let match_count = u64::from(match_count);
        if match_count > row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let match_count_usize = usize::try_from(match_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let mut indices = vec![0_u64; match_count_usize];
        if !indices.is_empty() {
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    indices.as_mut_ptr().cast::<c_void>(),
                    indices_guard.ptr,
                    indices.len() * std::mem::size_of::<u64>(),
                )
            })?;
        }
        // Same deterministic-ascending ordering guarantee as the async path above (the kernel's
        // atom-schedule append order is non-deterministic for >32 matches); sort host-side so this
        // fallback returns indices identical to the CPU/non-resident reference.
        indices.sort_unstable();
        Ok(indices)
    }
}
