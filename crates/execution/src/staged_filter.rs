//! Minimal CUDA driver smoke operation.
//!
//! This scalar add-one probe verifies that the driver can create a context, load a module, launch a
//! kernel, and read back one word. It is not a relational execution or result path.

use std::os::raw::c_void;

use libloading::Library;

use crate::cuda_context::{
    check_cuda, CudaContextGuard, CudaDeviceAllocationGuard, CudaModuleGuard,
};
use crate::CudaRuntimeProbeError;

pub(super) fn launch_cuda_smoke_add_one(input: u32) -> Result<u32, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_smoke_add_one(
    .param .u64 out_ptr,
    .param .u32 input
)
{
    .reg .u64 %out;
    .reg .u32 %r_value;
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %r_value, [input];
    add.u32 %r_value, %r_value, 1;
    st.global.u32 [%out], %r_value;
    ret;
}
"#;

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, std::mem::size_of::<u32>()) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(&mut function, module, c"gpu_db_cuda_smoke_add_one".as_ptr())
    })?;

    let mut output_arg = allocation_guard.ptr;
    let mut input_arg = input;
    let mut args = [
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut input_arg as *mut u32).cast::<c_void>(),
    ];
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
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
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut output = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output as *mut u32).cast::<c_void>(),
            allocation_guard.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;

    drop(module_guard);
    drop(allocation_guard);
    drop(context_guard);

    Ok(output)
}
