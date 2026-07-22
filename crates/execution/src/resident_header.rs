use crate::{launch_on_pooled_stream, CudaResidentDeviceMemory, CudaRuntimeProbeError};
use std::ffi::c_void;

pub(super) fn launch_cuda_resident_row_count(
    resident: &CudaResidentDeviceMemory,
) -> Result<u64, CudaRuntimeProbeError> {
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

    if resident.metadata().allocated_bytes < std::mem::size_of::<u64>() as u64 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            resident.metadata().allocated_bytes as usize,
        ));
    }

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    // P2-M1: reuse the process-wide cached module/function (no per-launch
    // cuModuleLoadData/Unload) and launch on a pooled private stream with a pooled scratch
    // output + events (no per-call cuMemAlloc/cuEvent* and no whole-context
    // cuCtxSynchronize) — the fixes the step-1 spike and step-4 benchmark located.
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    // A cache miss loads the CUDA module before the launch helper gets a chance to bind. Make
    // first use safe when reset proof runs on a worker other than the allocation thread.
    resident.primary().set_current()?;
    let function = resident
        .primary()
        .cached_function(c"gpu_db_resident_row_count", &ptx)?;

    let mut output_bytes = [0_u8; std::mem::size_of::<u64>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        let mut resident_arg = resident.device_ptr();
        let mut output_arg = output_ptr;
        let mut args = [
            (&mut resident_arg as *mut u64).cast::<c_void>(),
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

    Ok(u64::from_le_bytes(output_bytes))
}
