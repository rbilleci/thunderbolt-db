use std::os::raw::c_void;
use std::sync::Arc;

use crate::cuda_context::{check_cuda, PooledStreamOwned};
use crate::resident_memory::{CudaResidentDeviceMemory, CudaResidentReadSource};
use crate::{
    validate_i32_index_geometry, CudaI32EqualAnyProjectSubmission, CudaRuntimeProbeError,
    I32NeedlesHostGuard,
};

pub(super) fn submit_cuda_resident_i32_equal_any_project<R: CudaResidentReadSource>(
    resident: &R,
    filter_offset: u64,
    needles: &[i32],
    projection_offsets: &[u64],
    row_count: u64,
) -> Result<CudaI32EqualAnyProjectSubmission, CudaRuntimeProbeError> {
    // Only the blocking-fallback HtoD/memset and the (always-used) kernel launch are still loaded
    // here; the alloc/free/module/event symbols the un-migrated path used are replaced by the
    // shared pool, the module cache, and the pooled stream's own events.
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
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

    const MAX_PROJECTIONS: usize = 4;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_equal_any_project(
    .param .u64 resident_ptr,
    .param .u64 row_count,
    .param .u32 needle_count,
    .param .u32 projection_count,
    .param .u64 filter_offset,
    .param .u64 projection_offset0,
    .param .u64 projection_offset1,
    .param .u64 projection_offset2,
    .param .u64 projection_offset3,
    .param .u64 needles_ptr,
    .param .u64 out_values_ptr,
    .param .u64 out_needle_indices_ptr,
    .param .u64 out_row_indices_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p_out;
    .reg .pred %p_done;
    .reg .pred %p_match;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %idx32;
    .reg .u32 %needle_count;
    .reg .u32 %projection_count;
    .reg .u32 %needle_idx;
    .reg .u32 %slot;
    .reg .u32 %one;
    .reg .u64 %idx;
    .reg .u64 %rows;
    .reg .u64 %resident;
    .reg .u64 %filter_offset;
    .reg .u64 %projection_offset0;
    .reg .u64 %projection_offset1;
    .reg .u64 %projection_offset2;
    .reg .u64 %projection_offset3;
    .reg .u64 %needles;
    .reg .u64 %out_values;
    .reg .u64 %out_needle_indices;
    .reg .u64 %out_row_indices;
    .reg .u64 %out_count;
    .reg .u64 %row_byte;
    .reg .u64 %addr;
    .reg .u64 %needle_byte;
    .reg .u64 %slot64;
    .reg .u64 %projection_count64;
    .reg .u64 %base_slot;
    .reg .u64 %out_addr;
    .reg .s32 %row_value;
    .reg .s32 %needle_value;
    .reg .s32 %projection_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %rows, [row_count];
    ld.param.u32 %needle_count, [needle_count];
    ld.param.u32 %projection_count, [projection_count];
    ld.param.u64 %filter_offset, [filter_offset];
    ld.param.u64 %projection_offset0, [projection_offset0];
    ld.param.u64 %projection_offset1, [projection_offset1];
    ld.param.u64 %projection_offset2, [projection_offset2];
    ld.param.u64 %projection_offset3, [projection_offset3];
    ld.param.u64 %needles, [needles_ptr];
    ld.param.u64 %out_values, [out_values_ptr];
    ld.param.u64 %out_needle_indices, [out_needle_indices_ptr];
    ld.param.u64 %out_row_indices, [out_row_indices_ptr];
    ld.param.u64 %out_count, [out_count_ptr];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %idx32, %r_block, %r_block_dim, %r_tid;
    cvt.u64.u32 %idx, %idx32;

    setp.ge.u64 %p_out, %idx, %rows;
    @%p_out bra DONE;
    setp.eq.u32 %p_done, %needle_count, 0;
    @%p_done bra DONE;
    setp.eq.u32 %p_done, %projection_count, 0;
    @%p_done bra DONE;

    mul.lo.u64 %row_byte, %idx, 4;
    add.u64 %addr, %resident, %filter_offset;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %row_value, [%addr];

    mov.u32 %needle_idx, 0;
NEEDLE_LOOP:
    setp.ge.u32 %p_done, %needle_idx, %needle_count;
    @%p_done bra DONE;
    cvt.u64.u32 %needle_byte, %needle_idx;
    mul.lo.u64 %needle_byte, %needle_byte, 4;
    add.u64 %addr, %needles, %needle_byte;
    ld.global.s32 %needle_value, [%addr];
    setp.eq.s32 %p_match, %row_value, %needle_value;
    @%p_match bra MATCHED;
    add.u32 %needle_idx, %needle_idx, 1;
    bra NEEDLE_LOOP;

MATCHED:
    mov.u32 %one, 1;
    atom.global.add.u32 %slot, [%out_count], %one;
    cvt.u64.u32 %slot64, %slot;

    mul.lo.u64 %out_addr, %slot64, 4;
    add.u64 %out_addr, %out_needle_indices, %out_addr;
    st.global.u32 [%out_addr], %needle_idx;

    mul.lo.u64 %out_addr, %slot64, 8;
    add.u64 %out_addr, %out_row_indices, %out_addr;
    st.global.u64 [%out_addr], %idx;

    cvt.u64.u32 %projection_count64, %projection_count;
    mul.lo.u64 %base_slot, %slot64, %projection_count64;
    mul.lo.u64 %base_slot, %base_slot, 4;

    add.u64 %addr, %resident, %projection_offset0;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 1;
    @%p_done bra DONE;
    add.u64 %addr, %resident, %projection_offset1;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 4;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 2;
    @%p_done bra DONE;
    add.u64 %addr, %resident, %projection_offset2;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 8;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 3;
    @%p_done bra DONE;
    add.u64 %addr, %resident, %projection_offset3;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 12;
    st.global.s32 [%out_addr], %projection_value;

DONE:
    ret;
}
"#;

    if needles.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if projection_offsets.is_empty() || projection_offsets.len() > MAX_PROJECTIONS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            projection_offsets.len(),
        ));
    }
    if row_count == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if !filter_offset.is_multiple_of(std::mem::align_of::<i32>() as u64) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(filter_offset).unwrap_or(usize::MAX),
        ));
    }
    let filter_bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| filter_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if filter_bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            filter_bytes as usize,
        ));
    }
    for byte_offset in projection_offsets {
        if byte_offset % std::mem::align_of::<i32>() as u64 != 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(*byte_offset).unwrap_or(usize::MAX),
            ));
        }
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
    let needle_count_u32 = u32::try_from(needles.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;
    let output_cells = row_count
        .checked_mul(projection_offsets.len() as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = usize::try_from(
        output_cells
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_indices_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_row_indices_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<u64>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let needle_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    // P2-M2 (equal_any split-route migration — mirrors the text route's async-on-pooled-stream
    // lever, adapted to this route's deferred `submit`→`complete`). The un-migrated path was the
    // projection-route default-stream wall: per-call `cuMemAlloc`/`cuMemFree` (5 driver-serialized
    // allocs), a blocking default-stream `cuMemcpyHtoD`(needles) + `cuMemsetD8`(count), a per-call
    // `cuModuleLoadData` (re-JIT), and a kernel on the NULL stream — every one a context-wide
    // barrier across concurrent readers. This `submit` now: leases the device buffers from the
    // shared `OutputBufferPool` (no per-call alloc/free), uses the cached module (no re-JIT), and
    // ENQUEUES HtoD(needles) + memset(count) + the kernel on a pooled private stream via the
    // `*Async` variants when present (blocking variants on an old driver) — without syncing, so
    // the kernel overlaps the host work the caller does before `complete`. `complete` syncs the
    // stream and reads the results (async-pinned when available). The pooled buffers + stream are
    // carried as owned (`Arc`-holding) guards because they must outlive this frame.
    //
    // Only `cu_memcpy_htod` / `cu_memset_d8` (blocking fallback) and `cu_launch_kernel` (always)
    // are still loaded here; the alloc/free/module/event symbols are gone — the pool, the module
    // cache, and the pooled stream's own events replace them.
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let primary = resident.primary_arc();
    primary.set_current()?;
    let needles_host_guard = I32NeedlesHostGuard::stage(&primary, needles);

    // Pooled device buffers (owned guards — returned to the pool when the submission drops, after
    // `complete` reads them). Reused buffers are NOT zeroed; only the count is memset, the needles
    // buffer is fully overwritten by the HtoD, and every output is read back only over [0, count),
    // so stale bytes are never observed.
    let needles_guard = primary.lease_device_buffer_owned(needle_bytes)?;
    let values_guard = primary.lease_device_buffer_owned(output_bytes)?;
    let indices_guard = primary.lease_device_buffer_owned(output_indices_bytes)?;
    let row_indices_guard = primary.lease_device_buffer_owned(output_row_indices_bytes)?;
    let count_guard = primary.lease_device_buffer_owned(std::mem::size_of::<u32>())?;

    // Cached module (no per-call cuModuleLoadData) — the kernel is already parallel (one thread
    // per row + atomic-append); the wall was this per-call orchestration, not the kernel.
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_resident_i32_equal_any_project", &ptx)?;

    let mut resident_arg = resident.device_ptr();
    let mut rows_arg = row_count;
    let mut needle_count_arg = needle_count_u32;
    let mut projection_count_arg = u32::try_from(projection_offsets.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(projection_offsets.len()))?;
    let mut filter_offset_arg = filter_offset;
    let mut projected_offsets = [0_u64; MAX_PROJECTIONS];
    for (idx, offset) in projection_offsets.iter().enumerate() {
        projected_offsets[idx] = *offset;
    }
    let mut needles_arg = needles_guard.ptr;
    let mut output_arg = values_guard.ptr;
    let mut indices_arg = indices_guard.ptr;
    let mut row_indices_arg = row_indices_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut needle_count_arg as *mut u32).cast::<c_void>(),
        (&mut projection_count_arg as *mut u32).cast::<c_void>(),
        (&mut filter_offset_arg as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[0] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[1] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[2] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[3] as *mut u64).cast::<c_void>(),
        (&mut needles_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut indices_arg as *mut u64).cast::<c_void>(),
        (&mut row_indices_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count_u32.div_ceil(threads_per_block);

    // Establish the complete draining owner before the first asynchronous enqueue. In particular, it owns an
    // exact host copy of `needles`: this safe deferred API may return while H2D is still consuming that source.
    let mut submission = CudaI32EqualAnyProjectSubmission {
        projection_count: projection_offsets.len(),
        needles_len: needles.len(),
        row_count,
        primary: Arc::clone(&primary),
        _resident_allocation_guard: resident.allocation_arc(),
        values_guard,
        indices_guard,
        row_indices_guard,
        count_guard,
        _needles_guard: needles_guard,
        _needles_host_guard: needles_host_guard,
        stream: Some(PooledStreamOwned {
            primary: Arc::clone(&primary),
            pooled: Some(primary.acquire_pooled_stream()?),
        }),
        timed: false,
        _wave_index_guard: None,
    };
    let pooled = submission
        .stream
        .as_ref()
        .expect("submission owns stream")
        .pooled
        .as_ref()
        .expect("pooled stream just leased");
    let stream = pooled.stream;
    let timed = !pooled.start_event.is_null() && !pooled.stop_event.is_null();
    submission.timed = timed;
    let needles_host_ptr = submission._needles_host_guard.as_ptr();

    // Error-path stream drain (same contract as the text route): every op below ENQUEUES async
    // work on `stream`; an early `?` would unwind the owned buffer/stream guards (returning them
    // to their pools) while that work may still be in flight — a use-after-free for the next
    // leaser. `drain_err` blocking-syncs the stream FIRST (at the error site, before any guard
    // Drop), then yields the original error. Zero success-path cost.
    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        // SAFETY: `stream` is the live pooled stream; a blocking sync on it is valid here (the
        // primary context is current). Result intentionally ignored — best-effort error-path drain.
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    // Optional stream-ordered transfer symbols; absent → the blocking fallback (still correct,
    // just a default-stream barrier on an old driver). The kernel always launches on the pooled
    // stream regardless.
    let async_ops = match (primary.cu_memcpy_htod_async, primary.cu_memset_d8_async) {
        (Some(htod), Some(memset)) => Some((htod, memset)),
        _ => None,
    };

    if let Some((htod_async, memset_async)) = async_ops {
        check_cuda(unsafe { htod_async(needles_arg, needles_host_ptr, needle_bytes, stream) })
            .map_err(drain_err)?;
        check_cuda(unsafe { memset_async(count_arg, 0, std::mem::size_of::<u32>(), stream) })
            .map_err(drain_err)?;
    } else {
        // Blocking fallback: these default-stream ops complete (host-blocking) before the kernel
        // is enqueued on the pooled stream below, so the kernel still observes the uploaded needles
        // and the zeroed counter.
        check_cuda(unsafe { cu_memcpy_htod(needles_arg, needles_host_ptr, needle_bytes) })
            .map_err(drain_err)?;
        check_cuda(unsafe { cu_memset_d8(count_arg, 0, std::mem::size_of::<u32>()) })
            .map_err(drain_err)?;
    }

    if timed {
        check_cuda(unsafe { (primary.cu_event_record)(pooled.start_event, stream) })
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
        check_cuda(unsafe { (primary.cu_event_record)(pooled.stop_event, stream) })
            .map_err(drain_err)?;
    }
    // NB: deliberately NOT synced here — `complete` does the single covering sync, so the kernel
    // overlaps the caller's host work between `submit` and `complete`.

    Ok(submission)
}

/// R1a — the GPU-INDEX point-lookup analogue of `submit_cuda_resident_i32_equal_any_project`. Instead of
/// scanning every row for the needles (thread-per-row), it launches ONE thread per needle, hash-probes a
/// device-resident open-addressing index (`(key<<32)|(row+1)`, 0 = empty; Fibonacci
/// `(needle*0x9E3779B1)>>hash_shift` + linear probe, hard-capped) for the matching row, then appends
/// `(needle_index, row_index, projected values)` via the IDENTICAL atomic-`out_count` protocol the scan
/// uses — so the submission + `complete_detached` path is reused byte-for-byte. Output buffers are sized
/// for `needle_count` (a unique-key point lookup matches at most one row per needle), not the scan's
/// worst-case `row_count`. The index is a SEPARATE device buffer (built on admission over the key column),
/// passed by pointer; projections are still gathered from `resident` at the probed row.
#[allow(clippy::too_many_arguments)]
pub(super) fn submit_cuda_resident_i32_index_probe<R: CudaResidentReadSource>(
    resident: &R,
    index: &Arc<CudaResidentDeviceMemory>,
    index_table_mask: u32,
    index_hash_shift: u32,
    needles: &[i32],
    projection_offsets: &[u64],
    row_count: u64,
) -> Result<CudaI32EqualAnyProjectSubmission, CudaRuntimeProbeError> {
    let index_ptr = index.device_ptr();
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
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

    const MAX_PROJECTIONS: usize = 4;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_index_probe(
    .param .u64 resident_ptr,
    .param .u64 index_ptr,
    .param .u32 table_mask,
    .param .u32 hash_shift,
    .param .u32 row_count,
    .param .u32 needle_count,
    .param .u32 projection_count,
    .param .u64 projection_offset0,
    .param .u64 projection_offset1,
    .param .u64 projection_offset2,
    .param .u64 projection_offset3,
    .param .u64 needles_ptr,
    .param .u64 out_values_ptr,
    .param .u64 out_needle_indices_ptr,
    .param .u64 out_row_indices_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<28>;

    ld.param.u64 %rd1, [resident_ptr];
    ld.param.u64 %rd2, [index_ptr];
    ld.param.u32 %r1, [table_mask];
    ld.param.u32 %r2, [hash_shift];
    ld.param.u32 %r18, [row_count];
    ld.param.u32 %r3, [needle_count];
    ld.param.u32 %r4, [projection_count];
    ld.param.u64 %rd3, [projection_offset0];
    ld.param.u64 %rd4, [projection_offset1];
    ld.param.u64 %rd5, [projection_offset2];
    ld.param.u64 %rd6, [projection_offset3];
    ld.param.u64 %rd7, [needles_ptr];
    ld.param.u64 %rd8, [out_values_ptr];
    ld.param.u64 %rd9, [out_needle_indices_ptr];
    ld.param.u64 %rd10, [out_row_indices_ptr];
    ld.param.u64 %rd11, [out_count_ptr];

    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mad.lo.u32 %r8, %r6, %r7, %r5;
    setp.ge.u32 %p1, %r8, %r3;
    @%p1 bra DONE;
    setp.eq.u32 %p1, %r4, 0;
    @%p1 bra DONE;

    mul.wide.u32 %rd12, %r8, 4;
    add.u64 %rd13, %rd7, %rd12;
    ld.global.s32 %r9, [%rd13];
    mul.lo.u32 %r10, %r9, 2654435761;
    shr.u32 %r11, %r10, %r2;
    mov.u32 %r12, 0;

PROBE:
    and.b32 %r11, %r11, %r1;
    mul.wide.u32 %rd14, %r11, 8;
    add.u64 %rd15, %rd2, %rd14;
    ld.global.u64 %rd16, [%rd15];
    setp.eq.u64 %p2, %rd16, 0;
    @%p2 bra DONE;
    shr.u64 %rd17, %rd16, 32;
    cvt.u32.u64 %r13, %rd17;
    setp.eq.s32 %p2, %r13, %r9;
    @%p2 bra FOUND;
    add.u32 %r11, %r11, 1;
    add.u32 %r12, %r12, 1;
    setp.ge.u32 %p3, %r12, 256;
    @%p3 bra DONE;
    bra PROBE;

FOUND:
    cvt.u32.u64 %r14, %rd16;
    and.b32 %r14, %r14, 2147483647;
    sub.u32 %r14, %r14, 1;
    setp.ge.u32 %p2, %r14, %r18;
    @%p2 bra DONE;
    cvt.u64.u32 %rd18, %r14;

    mov.u32 %r15, 1;
    atom.global.add.u32 %r16, [%rd11], %r15;
    cvt.u64.u32 %rd19, %r16;

    mul.lo.u64 %rd20, %rd19, 4;
    add.u64 %rd21, %rd9, %rd20;
    st.global.u32 [%rd21], %r8;

    mul.lo.u64 %rd20, %rd19, 8;
    add.u64 %rd21, %rd10, %rd20;
    st.global.u64 [%rd21], %rd18;

    cvt.u64.u32 %rd22, %r4;
    mul.lo.u64 %rd23, %rd19, %rd22;
    mul.lo.u64 %rd23, %rd23, 4;
    mul.lo.u64 %rd24, %rd18, 4;

    add.u64 %rd25, %rd1, %rd3;
    add.u64 %rd25, %rd25, %rd24;
    ld.global.s32 %r17, [%rd25];
    add.u64 %rd26, %rd8, %rd23;
    st.global.s32 [%rd26], %r17;

    setp.le.u32 %p4, %r4, 1;
    @%p4 bra DONE;
    add.u64 %rd25, %rd1, %rd4;
    add.u64 %rd25, %rd25, %rd24;
    ld.global.s32 %r17, [%rd25];
    add.u64 %rd26, %rd8, %rd23;
    add.u64 %rd26, %rd26, 4;
    st.global.s32 [%rd26], %r17;

    setp.le.u32 %p4, %r4, 2;
    @%p4 bra DONE;
    add.u64 %rd25, %rd1, %rd5;
    add.u64 %rd25, %rd25, %rd24;
    ld.global.s32 %r17, [%rd25];
    add.u64 %rd26, %rd8, %rd23;
    add.u64 %rd26, %rd26, 8;
    st.global.s32 [%rd26], %r17;

    setp.le.u32 %p4, %r4, 3;
    @%p4 bra DONE;
    add.u64 %rd25, %rd1, %rd6;
    add.u64 %rd25, %rd25, %rd24;
    ld.global.s32 %r17, [%rd25];
    add.u64 %rd26, %rd8, %rd23;
    add.u64 %rd26, %rd26, 12;
    st.global.s32 [%rd26], %r17;

DONE:
    ret;
}
"#;

    if needles.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if projection_offsets.is_empty() || projection_offsets.len() > MAX_PROJECTIONS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            projection_offsets.len(),
        ));
    }
    if row_count == 0 || row_count >= (1_u64 << 31) || index_ptr == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if !Arc::ptr_eq(&resident.primary_arc(), &index.primary_arc()) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    validate_i32_index_geometry(
        index.metadata().allocated_bytes,
        index_table_mask,
        index_hash_shift,
    )?;
    for byte_offset in projection_offsets {
        if byte_offset % std::mem::align_of::<i32>() as u64 != 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(*byte_offset).unwrap_or(usize::MAX),
            ));
        }
        let bytes = row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .and_then(|bytes| byte_offset.checked_add(bytes))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if bytes > resident.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
        }
    }
    let needle_count_u32 = u32::try_from(needles.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;
    let output_cells = (needles.len() as u64)
        .checked_mul(projection_offsets.len() as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = usize::try_from(
        output_cells
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_indices_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_row_indices_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let needle_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let primary = resident.primary_arc();
    primary.set_current()?;
    let needles_host_guard = I32NeedlesHostGuard::stage(&primary, needles);

    let needles_guard = primary.lease_device_buffer_owned(needle_bytes)?;
    let values_guard = primary.lease_device_buffer_owned(output_bytes)?;
    let indices_guard = primary.lease_device_buffer_owned(output_indices_bytes)?;
    let row_indices_guard = primary.lease_device_buffer_owned(output_row_indices_bytes)?;
    let count_guard = primary.lease_device_buffer_owned(std::mem::size_of::<u32>())?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_resident_i32_index_probe", &ptx)?;

    let mut resident_arg = resident.device_ptr();
    let mut index_arg = index_ptr;
    let mut table_mask_arg = index_table_mask;
    let mut hash_shift_arg = index_hash_shift;
    let mut row_count_arg = row_count as u32;
    let mut needle_count_arg = needle_count_u32;
    let mut projection_count_arg = u32::try_from(projection_offsets.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(projection_offsets.len()))?;
    let mut projected_offsets = [0_u64; MAX_PROJECTIONS];
    for (idx, offset) in projection_offsets.iter().enumerate() {
        projected_offsets[idx] = *offset;
    }
    let mut needles_arg = needles_guard.ptr;
    let mut output_arg = values_guard.ptr;
    let mut indices_arg = indices_guard.ptr;
    let mut row_indices_arg = row_indices_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut index_arg as *mut u64).cast::<c_void>(),
        (&mut table_mask_arg as *mut u32).cast::<c_void>(),
        (&mut hash_shift_arg as *mut u32).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
        (&mut needle_count_arg as *mut u32).cast::<c_void>(),
        (&mut projection_count_arg as *mut u32).cast::<c_void>(),
        (&mut projected_offsets[0] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[1] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[2] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[3] as *mut u64).cast::<c_void>(),
        (&mut needles_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut indices_arg as *mut u64).cast::<c_void>(),
        (&mut row_indices_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = needle_count_u32.div_ceil(threads_per_block);

    let mut submission = CudaI32EqualAnyProjectSubmission {
        projection_count: projection_offsets.len(),
        needles_len: needles.len(),
        row_count,
        primary: Arc::clone(&primary),
        _resident_allocation_guard: resident.allocation_arc(),
        values_guard,
        indices_guard,
        row_indices_guard,
        count_guard,
        _needles_guard: needles_guard,
        _needles_host_guard: needles_host_guard,
        stream: Some(PooledStreamOwned {
            primary: Arc::clone(&primary),
            pooled: Some(primary.acquire_pooled_stream()?),
        }),
        timed: false,
        _wave_index_guard: Some(Arc::clone(index)),
    };
    let pooled = submission
        .stream
        .as_ref()
        .expect("submission owns stream")
        .pooled
        .as_ref()
        .expect("pooled stream just leased");
    let stream = pooled.stream;
    let timed = !pooled.start_event.is_null() && !pooled.stop_event.is_null();
    submission.timed = timed;
    let needles_host_ptr = submission._needles_host_guard.as_ptr();

    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    let async_ops = match (primary.cu_memcpy_htod_async, primary.cu_memset_d8_async) {
        (Some(htod), Some(memset)) => Some((htod, memset)),
        _ => None,
    };
    if let Some((htod_async, memset_async)) = async_ops {
        check_cuda(unsafe { htod_async(needles_arg, needles_host_ptr, needle_bytes, stream) })
            .map_err(drain_err)?;
        check_cuda(unsafe { memset_async(count_arg, 0, std::mem::size_of::<u32>(), stream) })
            .map_err(drain_err)?;
    } else {
        check_cuda(unsafe { cu_memcpy_htod(needles_arg, needles_host_ptr, needle_bytes) })
            .map_err(drain_err)?;
        check_cuda(unsafe { cu_memset_d8(count_arg, 0, std::mem::size_of::<u32>()) })
            .map_err(drain_err)?;
    }

    if timed {
        check_cuda(unsafe { (primary.cu_event_record)(pooled.start_event, stream) })
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
        check_cuda(unsafe { (primary.cu_event_record)(pooled.stop_event, stream) })
            .map_err(drain_err)?;
    }

    Ok(submission)
}
