use crate::{CudaResidentDeviceMemory, CudaRuntimeProbeError, check_cuda, launch_on_pooled_stream};
use std::ffi::c_void;

/// Device-final uniqueness verdict over packed `(chunk, slot)` coordinates. One thread is
/// intentional: P5 currently bounds statement batches at 256 rows, and this validation kernel is
/// an authorization seam rather than a read-throughput operator. The replacement milestone is
/// the tuple hash/group operator, which will keep the exact survivors resident end-to-end.
pub(super) fn launch_cuda_unique_coordinate_threshold(
    resident: &CudaResidentDeviceMemory,
    candidates: &[u64],
    exclusions: &[u64],
    reject_at: u32,
) -> Result<bool, CudaRuntimeProbeError> {
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

    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_unique_coordinate_threshold(
    .param .u64 candidates_ptr,
    .param .u64 candidate_count,
    .param .u64 exclusions_ptr,
    .param .u64 exclusion_count,
    .param .u32 reject_at,
    .param .u64 out_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_excluded;
    .reg .pred %p_equal;
    .reg .pred %p_reject;
    .reg .u32 %reject_at;
    .reg .u32 %accepted;
    .reg .u64 %candidates;
    .reg .u64 %candidate_count;
    .reg .u64 %exclusions;
    .reg .u64 %exclusion_count;
    .reg .u64 %out;
    .reg .u64 %ci;
    .reg .u64 %ei;
    .reg .u64 %offset;
    .reg .u64 %addr;
    .reg .u64 %candidate;
    .reg .u64 %excluded_coordinate;

    ld.param.u64 %candidates, [candidates_ptr];
    ld.param.u64 %candidate_count, [candidate_count];
    ld.param.u64 %exclusions, [exclusions_ptr];
    ld.param.u64 %exclusion_count, [exclusion_count];
    ld.param.u32 %reject_at, [reject_at];
    ld.param.u64 %out, [out_ptr];

    st.global.u32 [%out], 0;
    mov.u32 %accepted, 0;
    mov.u64 %ci, 0;

candidate_loop:
    setp.ge.u64 %p_done, %ci, %candidate_count;
    @%p_done bra done;
    mul.lo.u64 %offset, %ci, 8;
    add.u64 %addr, %candidates, %offset;
    ld.global.u64 %candidate, [%addr];
    mov.pred %p_excluded, 0;
    mov.u64 %ei, 0;

exclusion_loop:
    setp.ge.u64 %p_done, %ei, %exclusion_count;
    @%p_done bra exclusion_done;
    mul.lo.u64 %offset, %ei, 8;
    add.u64 %addr, %exclusions, %offset;
    ld.global.u64 %excluded_coordinate, [%addr];
    setp.eq.u64 %p_equal, %candidate, %excluded_coordinate;
    @%p_equal mov.pred %p_excluded, 1;
    @%p_equal bra exclusion_done;
    add.u64 %ei, %ei, 1;
    bra exclusion_loop;

exclusion_done:
    @%p_excluded bra next_candidate;
    add.u32 %accepted, %accepted, 1;
    setp.ge.u32 %p_reject, %accepted, %reject_at;
    @%p_reject st.global.u32 [%out], 1;
    @%p_reject bra done;

next_candidate:
    add.u64 %ci, %ci, 1;
    bra candidate_loop;

done:
    ret;
}
"#;

    if reject_at == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let candidate_bytes = candidates
        .len()
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let exclusion_bytes = exclusions
        .len()
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let primary = resident.primary();
    primary.set_current()?;
    let candidates_device = primary.lease_device_buffer(candidate_bytes.max(8))?;
    let exclusions_device = primary.lease_device_buffer(exclusion_bytes.max(8))?;
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    if candidate_bytes > 0 {
        check_cuda(unsafe {
            cu_memcpy_htod(
                candidates_device.ptr,
                candidates.as_ptr().cast::<c_void>(),
                candidate_bytes,
            )
        })?;
    }
    if exclusion_bytes > 0 {
        check_cuda(unsafe {
            cu_memcpy_htod(
                exclusions_device.ptr,
                exclusions.as_ptr().cast::<c_void>(),
                exclusion_bytes,
            )
        })?;
    }

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_unique_coordinate_threshold", &ptx)?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut output_bytes = [0_u8; std::mem::size_of::<u32>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        let mut candidates_arg = candidates_device.ptr;
        let mut candidate_count_arg = candidates.len() as u64;
        let mut exclusions_arg = exclusions_device.ptr;
        let mut exclusion_count_arg = exclusions.len() as u64;
        let mut reject_at_arg = reject_at;
        let mut output_arg = output_ptr;
        let mut args = [
            (&mut candidates_arg as *mut u64).cast::<c_void>(),
            (&mut candidate_count_arg as *mut u64).cast::<c_void>(),
            (&mut exclusions_arg as *mut u64).cast::<c_void>(),
            (&mut exclusion_count_arg as *mut u64).cast::<c_void>(),
            (&mut reject_at_arg as *mut u32).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                function,
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
        }
    })?;

    Ok(u32::from_le_bytes(output_bytes) != 0)
}
