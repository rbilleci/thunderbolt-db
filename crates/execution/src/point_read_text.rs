use std::ffi::c_void;

use super::{
    check_cuda, copy_pinned_into, launch_on_pooled_stream, stage_result_dtoh_async,
    CudaI32TextBatchProjectionRow, CudaResidentReadSource, CudaRuntimeProbeError,
    GpuPrimaryContext, PooledStream,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_cuda_resident_i32_equal_any_project_text<R: CudaResidentReadSource>(
    resident: &R,
    filter_offset: u64,
    needles: &[i32],
    projection_offsets: &[u64],
    text_offsets_byte_offset: u64,
    text_bytes_byte_offset: u64,
    text_bytes_len: u64,
    row_count: u64,
) -> Result<Vec<CudaI32TextBatchProjectionRow>, CudaRuntimeProbeError> {
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
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

    const MAX_PROJECTIONS: usize = 4;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_equal_any_project_text(
    .param .u64 resident_ptr,
    .param .u64 row_count,
    .param .u32 needle_count,
    .param .u32 projection_count,
    .param .u64 filter_offset,
    .param .u64 projection_offset0,
    .param .u64 projection_offset1,
    .param .u64 projection_offset2,
    .param .u64 projection_offset3,
    .param .u64 text_offsets_offset,
    .param .u64 text_bytes_offset,
    .param .u32 text_bytes_len,
    .param .u64 needles_ptr,
    .param .u64 out_values_ptr,
    .param .u64 out_needle_indices_ptr,
    .param .u64 out_row_indices_ptr,
    .param .u64 out_text_starts_ptr,
    .param .u64 out_text_lens_ptr,
    .param .u64 out_text_bytes_ptr,
    .param .u64 out_count_ptr,
    .param .u64 out_text_count_ptr
)
{
    .reg .pred %p_out;
    .reg .pred %p_done;
    .reg .pred %p_match;
    .reg .pred %p_copy_done;
    .reg .u16 %byte_value;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %idx32;
    .reg .u32 %needle_count;
    .reg .u32 %projection_count;
    .reg .u32 %needle_idx;
    .reg .u32 %slot;
    .reg .u32 %text_start32;
    .reg .u32 %text_end32;
    .reg .u32 %text_len;
    .reg .u32 %text_slot;
    .reg .u32 %copy_idx;
    .reg .u32 %one;
    .reg .u64 %idx;
    .reg .u64 %rows;
    .reg .u64 %resident;
    .reg .u64 %filter_offset;
    .reg .u64 %projection_offset0;
    .reg .u64 %projection_offset1;
    .reg .u64 %projection_offset2;
    .reg .u64 %projection_offset3;
    .reg .u64 %text_offsets_offset;
    .reg .u64 %text_bytes_offset;
    .reg .u64 %needles;
    .reg .u64 %out_values;
    .reg .u64 %out_needle_indices;
    .reg .u64 %out_row_indices;
    .reg .u64 %out_text_starts;
    .reg .u64 %out_text_lens;
    .reg .u64 %out_text_bytes;
    .reg .u64 %out_count;
    .reg .u64 %out_text_count;
    .reg .u64 %row_byte;
    .reg .u64 %addr;
    .reg .u64 %addr2;
    .reg .u64 %needle_byte;
    .reg .u64 %slot64;
    .reg .u64 %projection_count64;
    .reg .u64 %base_slot;
    .reg .u64 %out_addr;
    .reg .u64 %text_offset_addr;
    .reg .u64 %text_start64;
    .reg .u64 %text_end64;
    .reg .u64 %text_slot64;
    .reg .u64 %copy64;
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
    ld.param.u64 %text_offsets_offset, [text_offsets_offset];
    ld.param.u64 %text_bytes_offset, [text_bytes_offset];
    ld.param.u64 %needles, [needles_ptr];
    ld.param.u64 %out_values, [out_values_ptr];
    ld.param.u64 %out_needle_indices, [out_needle_indices_ptr];
    ld.param.u64 %out_row_indices, [out_row_indices_ptr];
    ld.param.u64 %out_text_starts, [out_text_starts_ptr];
    ld.param.u64 %out_text_lens, [out_text_lens_ptr];
    ld.param.u64 %out_text_bytes, [out_text_bytes_ptr];
    ld.param.u64 %out_count, [out_count_ptr];
    ld.param.u64 %out_text_count, [out_text_count_ptr];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %idx32, %r_block, %r_block_dim, %r_tid;
    cvt.u64.u32 %idx, %idx32;

    setp.ge.u64 %p_out, %idx, %rows;
    @%p_out bra DONE;
    setp.eq.u32 %p_done, %needle_count, 0;
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
    mul.lo.u64 %text_offset_addr, %idx, 8;
    add.u64 %addr, %resident, %text_offsets_offset;
    add.u64 %addr, %addr, %text_offset_addr;
    ld.global.u32 %text_start32, [%addr];
    add.u64 %addr, %addr, 8;
    ld.global.u32 %text_end32, [%addr];
    cvt.u64.u32 %text_start64, %text_start32;
    cvt.u64.u32 %text_end64, %text_end32;
    sub.u32 %text_len, %text_end32, %text_start32;

    mov.u32 %one, 1;
    atom.global.add.u32 %slot, [%out_count], %one;
    cvt.u64.u32 %slot64, %slot;
    atom.global.add.u32 %text_slot, [%out_text_count], %text_len;
    cvt.u64.u32 %text_slot64, %text_slot;

    mul.lo.u64 %out_addr, %slot64, 4;
    add.u64 %out_addr, %out_needle_indices, %out_addr;
    st.global.u32 [%out_addr], %needle_idx;

    mul.lo.u64 %out_addr, %slot64, 8;
    add.u64 %out_addr, %out_row_indices, %out_addr;
    st.global.u64 [%out_addr], %idx;

    mul.lo.u64 %out_addr, %slot64, 4;
    add.u64 %addr, %out_text_starts, %out_addr;
    st.global.u32 [%addr], %text_slot;
    add.u64 %addr, %out_text_lens, %out_addr;
    st.global.u32 [%addr], %text_len;

    setp.eq.u32 %p_done, %projection_count, 0;
    @%p_done bra COPY_TEXT;
    cvt.u64.u32 %projection_count64, %projection_count;
    mul.lo.u64 %base_slot, %slot64, %projection_count64;
    mul.lo.u64 %base_slot, %base_slot, 4;

    add.u64 %addr, %resident, %projection_offset0;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 1;
    @%p_done bra COPY_TEXT;
    add.u64 %addr, %resident, %projection_offset1;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 4;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 2;
    @%p_done bra COPY_TEXT;
    add.u64 %addr, %resident, %projection_offset2;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 8;
    st.global.s32 [%out_addr], %projection_value;

    setp.le.u32 %p_done, %projection_count, 3;
    @%p_done bra COPY_TEXT;
    add.u64 %addr, %resident, %projection_offset3;
    add.u64 %addr, %addr, %row_byte;
    ld.global.s32 %projection_value, [%addr];
    add.u64 %out_addr, %out_values, %base_slot;
    add.u64 %out_addr, %out_addr, 12;
    st.global.s32 [%out_addr], %projection_value;

COPY_TEXT:
    mov.u32 %copy_idx, 0;
COPY_LOOP:
    setp.ge.u32 %p_copy_done, %copy_idx, %text_len;
    @%p_copy_done bra DONE;
    cvt.u64.u32 %copy64, %copy_idx;
    add.u64 %addr, %resident, %text_bytes_offset;
    add.u64 %addr, %addr, %text_start64;
    add.u64 %addr, %addr, %copy64;
    ld.global.u8 %byte_value, [%addr];
    add.u64 %addr2, %out_text_bytes, %text_slot64;
    add.u64 %addr2, %addr2, %copy64;
    st.global.u8 [%addr2], %byte_value;
    add.u32 %copy_idx, %copy_idx, 1;
    bra COPY_LOOP;

DONE:
    ret;
}
"#;

    if needles.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if projection_offsets.len() > MAX_PROJECTIONS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            projection_offsets.len(),
        ));
    }
    if row_count == 0 {
        return Ok(Vec::new());
    }
    let _text_bytes_len_u32 = u32::try_from(text_bytes_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let filter_bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| filter_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let text_offsets_end = row_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
        .and_then(|bytes| text_offsets_byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let text_bytes_end = text_bytes_byte_offset
        .checked_add(text_bytes_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if filter_bytes > resident.metadata().allocated_bytes
        || text_offsets_end > resident.metadata().allocated_bytes
        || text_bytes_end > resident.metadata().allocated_bytes
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            filter_bytes.max(text_offsets_end).max(text_bytes_end) as usize,
        ));
    }
    for byte_offset in projection_offsets {
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
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?
    .max(1);
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
    let output_text_bytes = usize::try_from(text_bytes_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?
        .max(1);
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

    // P2-M2 (text-route async lever): the c64 wall of this route was its 11 synchronous,
    // default/NULL-stream memory ops (1 HtoD + 2 memset + 8 D2H) — the driver serializes those
    // context-wide across concurrent readers, so 64 threads × 11 ops queued behind one barrier
    // (measured 99.7 % of a 12.8 ms c64 wall; the kernel + its private-stream sync were < 0.5 %).
    // The fix moves every one of those ops onto the route's **already-pooled private stream**
    // via the `*Async` variants behind exactly TWO `cuStreamSynchronize` (one after the kernel
    // so the device-computed counts are readable to size the result reads; one after the result
    // D2H). Two extra reductions ride along: the two atomic-append counters are fused into ONE
    // 8-byte device buffer read back in ONE D2H (the count read was the single largest post-fix
    // section because it forces the mid-pipeline sync), and all device→host copies stage through
    // **pooled pinned (page-locked) host buffers** so the async D2H is truly async + DMA-fast.
    // `cuMemAlloc`/`cuMemHostAlloc`/`cuStreamCreate` are themselves driver-serialized, so every
    // buffer (device + pinned-host) and the stream are POOLED — a per-call alloc would
    // re-introduce the very contention this removes (that was why the throwaway knob experiment,
    // which created an un-pooled per-call stream, only reached 1.64× instead of more).
    //
    // The whole async path is gated on the optional async + pinned-host driver symbols; on an
    // old driver lacking them the route keeps the original blocking default-stream path below,
    // so correctness is unconditional and only the acceleration is best-effort.
    let async_ops = match (
        resident.primary().cu_memcpy_htod_async,
        resident.primary().cu_memcpy_dtoh_async,
        resident.primary().cu_memset_d8_async,
    ) {
        (Some(htod), Some(dtoh), Some(memset)) => Some((htod, dtoh, memset)),
        _ => None,
    };

    // Pooled device output buffers (no per-call cuMemAlloc/cuMemFree). Reused buffers are NOT
    // zeroed; only the fused counter buffer is memset, the needles buffer is fully overwritten
    // by the HtoD upload, and every output is read back only over [0, count) — so stale bytes
    // are never observed.
    let needles_guard = resident.primary().lease_device_buffer(needle_bytes)?;
    let values_guard = resident.primary().lease_device_buffer(output_bytes)?;
    let indices_guard = resident
        .primary()
        .lease_device_buffer(output_indices_bytes)?;
    let row_indices_guard = resident
        .primary()
        .lease_device_buffer(output_row_indices_bytes)?;
    let text_starts_guard = resident
        .primary()
        .lease_device_buffer(output_indices_bytes)?;
    let text_lens_guard = resident
        .primary()
        .lease_device_buffer(output_indices_bytes)?;
    let text_bytes_guard = resident.primary().lease_device_buffer(output_text_bytes)?;
    // Fused counters: `count` at +0, `text_count` at +4 of one 8-byte device buffer, so both
    // are zeroed by one memset and read back by one D2H. The kernel still receives two distinct
    // pointers (it does `atom.add` into each independently), so no kernel change is required.
    const COUNT_OFFSET: usize = 0;
    const TEXT_COUNT_OFFSET: usize = std::mem::size_of::<u32>();
    const COUNTERS_BYTES: usize = 2 * std::mem::size_of::<u32>();
    let counters_guard = resident.primary().lease_device_buffer(COUNTERS_BYTES)?;
    let count_ptr = counters_guard.ptr + COUNT_OFFSET as u64;
    let text_count_ptr = counters_guard.ptr + TEXT_COUNT_OFFSET as u64;

    // P2-M2: cached module (no per-launch cuModuleLoadData) — the projection kernel is
    // already parallel (one thread per row + atomic-append); the c64 wall was this per-call
    // orchestration (per-launch JIT + whole-context sync), not the kernel.
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_equal_any_project_text", &ptx)?;

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
    let mut text_offsets_arg = text_offsets_byte_offset;
    let mut text_bytes_arg = text_bytes_byte_offset;
    let mut text_bytes_len_arg = _text_bytes_len_u32;
    let mut needles_arg = needles_guard.ptr;
    let mut output_arg = values_guard.ptr;
    let mut indices_arg = indices_guard.ptr;
    let mut row_indices_arg = row_indices_guard.ptr;
    let mut text_starts_arg = text_starts_guard.ptr;
    let mut text_lens_arg = text_lens_guard.ptr;
    let mut text_output_arg = text_bytes_guard.ptr;
    let mut count_arg = count_ptr;
    let mut text_count_arg = text_count_ptr;
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
        (&mut text_offsets_arg as *mut u64).cast::<c_void>(),
        (&mut text_bytes_arg as *mut u64).cast::<c_void>(),
        (&mut text_bytes_len_arg as *mut u32).cast::<c_void>(),
        (&mut needles_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut indices_arg as *mut u64).cast::<c_void>(),
        (&mut row_indices_arg as *mut u64).cast::<c_void>(),
        (&mut text_starts_arg as *mut u64).cast::<c_void>(),
        (&mut text_lens_arg as *mut u64).cast::<c_void>(),
        (&mut text_output_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
        (&mut text_count_arg as *mut u64).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count_u32.div_ceil(threads_per_block);
    let projection_count = projection_offsets.len();

    // Read back the fused 8-byte counter pair; on the async path it stages through a pooled
    // pinned host buffer, on the blocking path through a plain stack array.
    let mut counters_host = [0_u32; 2];

    let (match_count, compact_text_len) = if let Some((htod_async, dtoh_async, memset_async)) =
        async_ops
    {
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
        // stream, an early `?` would propagate and unwind the locals — returning the device +
        // pinned buffer leases to their pools while enqueued ops may still be in flight, a
        // use-after-free window for whoever leases those buffers next. So every fallible op
        // from the first async enqueue through each covering `cuStreamSynchronize` propagates
        // its error through `drain_err`, which does a best-effort blocking sync (ignoring its
        // result) to drain the stream FIRST, *then* yields the original error.
        //
        // Ordering proof (drain-before-release on EVERY error path): `.map_err(drain_err)`
        // runs the closure at the error site, BEFORE the `?` returns and hence before ANY local
        // Drop runs — so it precedes the release of every lease regardless of scope (the device
        // guards live in the outer fn scope and drop last on unwind; the pinned leases live in
        // this block and drop first). A declaration-order Drop guard could not cover both,
        // because the pinned leases are created incrementally across this region; draining at
        // the error site sidesteps scope/order entirely. Zero success-path cost: `map_err` does
        // not invoke the closure on `Ok`, so a successful op adds nothing — the two explicit
        // syncs below remain the only synchronizations on the hot path. The drain is applied
        // only to ops that may leave async work in flight (enqueues + the covering syncs); the
        // pure-host steps between/after the syncs run when the stream is already idle, so they
        // need no drain.
        let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
            // SAFETY: `stream` is the live pooled private stream; a blocking synchronize on it is
            // valid from this thread (the primary context is current). The result is intentionally
            // ignored — this is a best-effort drain on an already-failing path.
            unsafe {
                let _ = (resident.primary().cu_stream_synchronize)(stream);
            }
            err
        };

        // (1) Stream-ordered upload + zero + kernel: HtoD(needles), memset(counters=0), kernel.
        check_cuda(unsafe {
            htod_async(
                needles_guard.ptr,
                needles.as_ptr().cast::<c_void>(),
                needle_bytes,
                stream,
            )
        })
        .map_err(drain_err)?;
        check_cuda(unsafe { memset_async(counters_guard.ptr, 0, COUNTERS_BYTES, stream) })
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
        // (2) Stream-ordered read of the fused 8-byte counter pair, then sync #1: now the
        // counters (and the kernel) are complete and the result-array sizes are known. The
        // counters stage through a small pooled pinned host buffer for a truly-async DMA.
        let counters_pinned = resident.primary().lease_pinned_host_buffer(COUNTERS_BYTES);
        let counters_dst: *mut c_void = counters_pinned
            .as_ref()
            .map(|p| p.ptr)
            .unwrap_or_else(|| counters_host.as_mut_ptr().cast::<c_void>());
        check_cuda(unsafe { dtoh_async(counters_dst, counters_guard.ptr, COUNTERS_BYTES, stream) })
            .map_err(drain_err)?;
        // Covering sync #1: drains on its own error too (the counter D2H is still enqueued).
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        if let Some(pinned) = &counters_pinned {
            // SAFETY: the sync above completed the 8-byte D2H into the pinned region; copy the
            // two u32 counters out by typed pointer (no alignment hazard: pinned host memory is
            // page-aligned and we read u32s from a u32-array layout).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pinned.ptr.cast::<u32>(),
                    counters_host.as_mut_ptr(),
                    counters_host.len(),
                );
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

        let match_count = u64::from(counters_host[0]);
        let compact_text_len = counters_host[1];
        if match_count > row_count || u64::from(compact_text_len) > text_bytes_len {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let match_count_usize = usize::try_from(match_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let compact_text_usize = usize::try_from(compact_text_len)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

        // (3) Stream-ordered result D2H into pooled pinned host buffers (one queued copy per
        // result array, each reading back only the populated [0, count) prefix of its worst-
        // case device buffer), then ONE sync #2 — so the six copies overlap on the copy engine
        // instead of serializing as six blocking default-stream barriers. After the sync, the
        // pinned bytes are copied into owned Vecs (pinned buffers return to the pool on drop).
        let mut values = vec![0_i32; match_count_usize.saturating_mul(projection_count)];
        let mut needle_indices = vec![0_u32; match_count_usize];
        let mut row_indices = vec![0_u64; match_count_usize];
        let mut text_starts = vec![0_u32; match_count_usize];
        let mut text_lens = vec![0_u32; match_count_usize];
        let mut text_bytes = vec![0_u8; compact_text_usize];

        // Each staged D2H enqueues an async copy before it can return `Err`, so its error path
        // drains the stream first (via `drain_err`) before any lease unwinds. Earlier copies in
        // this batch are also still in flight on a later copy's failure — the single covering
        // sync #2 below would normally wait on them, but on the error path we must drain
        // explicitly since that sync is skipped.
        let values_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            values_guard.ptr,
            &mut values,
        )
        .map_err(drain_err)?;
        let needle_indices_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            indices_guard.ptr,
            &mut needle_indices,
        )
        .map_err(drain_err)?;
        let row_indices_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            row_indices_guard.ptr,
            &mut row_indices,
        )
        .map_err(drain_err)?;
        let text_starts_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            text_starts_guard.ptr,
            &mut text_starts,
        )
        .map_err(drain_err)?;
        let text_lens_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            text_lens_guard.ptr,
            &mut text_lens,
        )
        .map_err(drain_err)?;
        let text_bytes_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            text_bytes_guard.ptr,
            &mut text_bytes,
        )
        .map_err(drain_err)?;
        // Covering sync #2: drains on its own error too (six result D2H still enqueued).
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;

        copy_pinned_into(&values_pinned, &mut values);
        copy_pinned_into(&needle_indices_pinned, &mut needle_indices);
        copy_pinned_into(&row_indices_pinned, &mut row_indices);
        copy_pinned_into(&text_starts_pinned, &mut text_starts);
        copy_pinned_into(&text_lens_pinned, &mut text_lens);
        copy_pinned_into(&text_bytes_pinned, &mut text_bytes);
        drop(lease);

        return assemble_i32_text_batch_projection_rows(
            needles,
            row_count,
            projection_count,
            &values,
            &needle_indices,
            &row_indices,
            &text_starts,
            &text_lens,
            &text_bytes,
        );
    } else {
        // ---- legacy blocking default-stream fallback (old driver: no async/pinned symbols) ----
        check_cuda(unsafe {
            cu_memcpy_htod(
                needles_guard.ptr,
                needles.as_ptr().cast::<c_void>(),
                needle_bytes,
            )
        })?;
        check_cuda(unsafe { cu_memset_d8(counters_guard.ptr, 0, COUNTERS_BYTES) })?;
        // P2-M2: launch on a pooled private stream synced individually (no whole-context
        // cuCtxSynchronize); the route owns its output buffers, so no pooled scratch.
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
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                counters_host.as_mut_ptr().cast::<c_void>(),
                counters_guard.ptr,
                COUNTERS_BYTES,
            )
        })?;
        let match_count = u64::from(counters_host[0]);
        let compact_text_len = counters_host[1];
        if match_count > row_count || u64::from(compact_text_len) > text_bytes_len {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        (match_count, compact_text_len)
    };

    let match_count_usize = usize::try_from(match_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut values = vec![0_i32; match_count_usize.saturating_mul(projection_count)];
    if !values.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                values.as_mut_ptr().cast::<c_void>(),
                values_guard.ptr,
                values.len() * std::mem::size_of::<i32>(),
            )
        })?;
    }
    let mut needle_indices = vec![0_u32; match_count_usize];
    if !needle_indices.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                needle_indices.as_mut_ptr().cast::<c_void>(),
                indices_guard.ptr,
                needle_indices.len() * std::mem::size_of::<u32>(),
            )
        })?;
    }
    let mut row_indices = vec![0_u64; match_count_usize];
    if !row_indices.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                row_indices.as_mut_ptr().cast::<c_void>(),
                row_indices_guard.ptr,
                row_indices.len() * std::mem::size_of::<u64>(),
            )
        })?;
    }
    let mut text_starts = vec![0_u32; match_count_usize];
    if !text_starts.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                text_starts.as_mut_ptr().cast::<c_void>(),
                text_starts_guard.ptr,
                text_starts.len() * std::mem::size_of::<u32>(),
            )
        })?;
    }
    let mut text_lens = vec![0_u32; match_count_usize];
    if !text_lens.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                text_lens.as_mut_ptr().cast::<c_void>(),
                text_lens_guard.ptr,
                text_lens.len() * std::mem::size_of::<u32>(),
            )
        })?;
    }
    let mut text_bytes = vec![
        0_u8;
        usize::try_from(compact_text_len).map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
        })?
    ];
    if !text_bytes.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                text_bytes.as_mut_ptr().cast::<c_void>(),
                text_bytes_guard.ptr,
                text_bytes.len(),
            )
        })?;
    }

    drop(counters_guard);
    drop(text_bytes_guard);
    drop(text_lens_guard);
    drop(text_starts_guard);
    drop(row_indices_guard);
    drop(indices_guard);
    drop(values_guard);
    drop(needles_guard);
    assemble_i32_text_batch_projection_rows(
        needles,
        row_count,
        projection_count,
        &values,
        &needle_indices,
        &row_indices,
        &text_starts,
        &text_lens,
        &text_bytes,
    )
}

/// Shared row assembler for both the async and blocking text-route paths: validate the
/// device-returned indices and stitch the per-row arrays into `CudaI32TextBatchProjectionRow`s
/// with byte-identical layout to the pre-async path (same field semantics, same column order).
#[allow(clippy::too_many_arguments)]
fn assemble_i32_text_batch_projection_rows(
    needles: &[i32],
    row_count: u64,
    projection_count: usize,
    values: &[i32],
    needle_indices: &[u32],
    row_indices: &[u64],
    text_starts: &[u32],
    text_lens: &[u32],
    text_bytes: &[u8],
) -> Result<Vec<CudaI32TextBatchProjectionRow>, CudaRuntimeProbeError> {
    values
        .chunks_exact(projection_count)
        .zip(needle_indices.iter().copied())
        .zip(row_indices.iter().copied())
        .zip(text_starts.iter().copied())
        .zip(text_lens.iter().copied())
        .map(
            |((((row, needle_index), row_index), text_start), text_len)| {
                let needle_index = usize::try_from(needle_index)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                if needle_index >= needles.len() {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(needle_index));
                }
                if row_index >= row_count {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
                }
                let text_start = usize::try_from(text_start)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                let text_len = usize::try_from(text_len)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                let text_end = text_start
                    .checked_add(text_len)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                if text_end > text_bytes.len() {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(text_end));
                }
                let text = std::str::from_utf8(&text_bytes[text_start..text_end])
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(text_len))?
                    .to_string();
                Ok(CudaI32TextBatchProjectionRow {
                    needle_index,
                    row_index,
                    values: row.to_vec(),
                    text,
                })
            },
        )
        .collect()
}
