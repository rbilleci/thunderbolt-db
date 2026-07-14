//! Host-staged CUDA smoke and filter bootstrap operations.
//!
//! These launchers preserve the generic CUDA-MVCC compatibility path. They upload host materialized
//! rows on every call and are RETIRE-003 debt, not the GPU-resident product execution boundary.

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

pub(super) fn launch_cuda_all_mask(row_count: usize) -> Result<Vec<bool>, CudaRuntimeProbeError> {
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

.visible .entry gpu_db_cuda_all_mask(
    .param .u64 mask_ptr,
    .param .u32 row_count
)
{
    .reg .pred %p_out;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_row_count, [row_count];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset, %r_idx, 4;
    add.u64 %rd_mask_addr, %rd_mask, %rd_offset;
    st.global.u32 [%rd_mask_addr], 1;

DONE:
    ret;
}
"#;

    let row_count = u32::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(row_count))?;
    if row_count == 0 {
        return Ok(Vec::new());
    }

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

    let mask_len = row_count as usize;
    let mask_bytes = mask_len * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
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
        cu_module_get_function(&mut function, module, c"gpu_db_cuda_all_mask".as_ptr())
    })?;

    let mut mask_arg = mask_guard.ptr;
    let mut row_count_arg = row_count;
    let mut args = [
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count.div_ceil(threads_per_block);
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
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; mask_len];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

pub(super) fn launch_cuda_u32_equal_mask(
    input: &[u32],
    needle: u32,
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_u32_equal_mask(
    .param .u64 input_ptr,
    .param .u64 mask_ptr,
    .param .u32 len,
    .param .u32 needle
)
{
    .reg .pred %p_out;
    .reg .pred %p_match;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_len;
    .reg .u32 %r_needle;
    .reg .u32 %r_value;
    .reg .u32 %r_mask;
    .reg .u64 %rd_input;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset;
    .reg .u64 %rd_input_addr;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_input, [input_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_len, [len];
    ld.param.u32 %r_needle, [needle];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_len;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset, %r_idx, 4;
    add.u64 %rd_input_addr, %rd_input, %rd_offset;
    ld.global.u32 %r_value, [%rd_input_addr];
    setp.eq.u32 %p_match, %r_value, %r_needle;
    selp.u32 %r_mask, 1, 0, %p_match;
    add.u64 %rd_mask_addr, %rd_mask, %rd_offset;
    st.global.u32 [%rd_mask_addr], %r_mask;

DONE:
    ret;
}
"#;

    let len = u32::try_from(input.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(input.len()))?;
    if input.is_empty() {
        return Ok(Vec::new());
    }

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
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
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

    let input_bytes = std::mem::size_of_val(input);
    let mut device_input = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_input, input_bytes) })?;
    let input_guard = CudaDeviceAllocationGuard {
        ptr: device_input,
        free: *cu_mem_free,
    };

    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, input_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    check_cuda(unsafe {
        cu_memcpy_htod(
            input_guard.ptr,
            input.as_ptr().cast::<c_void>(),
            input_bytes,
        )
    })?;

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
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_u32_equal_mask".as_ptr(),
        )
    })?;

    let mut input_arg = input_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut len_arg = len;
    let mut needle_arg = needle;
    let mut args = [
        (&mut input_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
        (&mut needle_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = len.div_ceil(threads_per_block);
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
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; input.len()];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            input_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(input_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

pub(super) fn launch_cuda_bytes_equal_mask(
    input: &[&[u8]],
    needle: &[u8],
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_bytes_equal_mask(
    .param .u64 bytes_ptr,
    .param .u64 offsets_ptr,
    .param .u64 needle_ptr,
    .param .u64 mask_ptr,
    .param .u32 row_count,
    .param .u32 needle_len
)
{
    .reg .pred %p_out;
    .reg .pred %p_len_diff;
    .reg .pred %p_loop_done;
    .reg .pred %p_byte_diff;
    .reg .u16 %h_byte;
    .reg .u16 %n_byte;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_needle_len;
    .reg .u32 %r_start;
    .reg .u32 %r_end;
    .reg .u32 %r_len;
    .reg .u32 %r_i;
    .reg .u32 %r_mask;
    .reg .u64 %rd_bytes;
    .reg .u64 %rd_offsets;
    .reg .u64 %rd_needle;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset_addr;
    .reg .u64 %rd_next_offset_addr;
    .reg .u64 %rd_byte_offset;
    .reg .u64 %rd_hay_addr;
    .reg .u64 %rd_needle_addr;
    .reg .u64 %rd_mask_offset;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_bytes, [bytes_ptr];
    ld.param.u64 %rd_offsets, [offsets_ptr];
    ld.param.u64 %rd_needle, [needle_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_row_count, [row_count];
    ld.param.u32 %r_needle_len, [needle_len];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset_addr, %r_idx, 4;
    add.u64 %rd_offset_addr, %rd_offsets, %rd_offset_addr;
    add.u64 %rd_next_offset_addr, %rd_offset_addr, 4;
    ld.global.u32 %r_start, [%rd_offset_addr];
    ld.global.u32 %r_end, [%rd_next_offset_addr];
    sub.u32 %r_len, %r_end, %r_start;
    setp.ne.u32 %p_len_diff, %r_len, %r_needle_len;
    @%p_len_diff bra NO_MATCH;

    mov.u32 %r_i, 0;
LOOP:
    setp.ge.u32 %p_loop_done, %r_i, %r_needle_len;
    @%p_loop_done bra MATCH;
    add.u32 %r_len, %r_start, %r_i;
    cvt.u64.u32 %rd_byte_offset, %r_len;
    add.u64 %rd_hay_addr, %rd_bytes, %rd_byte_offset;
    cvt.u64.u32 %rd_byte_offset, %r_i;
    add.u64 %rd_needle_addr, %rd_needle, %rd_byte_offset;
    ld.global.u8 %h_byte, [%rd_hay_addr];
    ld.global.u8 %n_byte, [%rd_needle_addr];
    setp.ne.u16 %p_byte_diff, %h_byte, %n_byte;
    @%p_byte_diff bra NO_MATCH;
    add.u32 %r_i, %r_i, 1;
    bra LOOP;

MATCH:
    mov.u32 %r_mask, 1;
    bra STORE;

NO_MATCH:
    mov.u32 %r_mask, 0;

STORE:
    mul.wide.u32 %rd_mask_offset, %r_idx, 4;
    add.u64 %rd_mask_addr, %rd_mask, %rd_mask_offset;
    st.global.u32 [%rd_mask_addr], %r_mask;

DONE:
    ret;
}
"#;

    let row_count = u32::try_from(input.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(input.len()))?;
    let needle_len = u32::try_from(needle.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needle.len()))?;
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut offsets = Vec::with_capacity(input.len() + 1);
    let mut flattened = Vec::new();
    offsets.push(0_u32);
    for value in input {
        flattened.extend_from_slice(value);
        offsets.push(
            u32::try_from(flattened.len())
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(flattened.len()))?,
        );
    }

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
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
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

    let bytes_len = flattened.len().max(1);
    let mut device_bytes = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_bytes, bytes_len) })?;
    let bytes_guard = CudaDeviceAllocationGuard {
        ptr: device_bytes,
        free: *cu_mem_free,
    };

    let offsets_bytes = std::mem::size_of_val(offsets.as_slice());
    let mut device_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_offsets, offsets_bytes) })?;
    let offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_offsets,
        free: *cu_mem_free,
    };

    let needle_bytes = needle.len().max(1);
    let mut device_needle = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_needle, needle_bytes) })?;
    let needle_guard = CudaDeviceAllocationGuard {
        ptr: device_needle,
        free: *cu_mem_free,
    };

    let mask_bytes = input.len() * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    if !flattened.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                bytes_guard.ptr,
                flattened.as_ptr().cast::<c_void>(),
                flattened.len(),
            )
        })?;
    }
    check_cuda(unsafe {
        cu_memcpy_htod(
            offsets_guard.ptr,
            offsets.as_ptr().cast::<c_void>(),
            offsets_bytes,
        )
    })?;
    if !needle.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                needle_guard.ptr,
                needle.as_ptr().cast::<c_void>(),
                needle.len(),
            )
        })?;
    }

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
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_bytes_equal_mask".as_ptr(),
        )
    })?;

    let mut bytes_arg = bytes_guard.ptr;
    let mut offsets_arg = offsets_guard.ptr;
    let mut needle_arg = needle_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut row_count_arg = row_count;
    let mut needle_len_arg = needle_len;
    let mut args = [
        (&mut bytes_arg as *mut u64).cast::<c_void>(),
        (&mut offsets_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
        (&mut needle_len_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count.div_ceil(threads_per_block);
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
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; input.len()];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(needle_guard);
    drop(offsets_guard);
    drop(bytes_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

pub(super) fn launch_cuda_bytes_range_mask(
    input: &[&[u8]],
    start_inclusive: &[u8],
    end_exclusive: &[u8],
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_bytes_range_mask(
    .param .u64 bytes_ptr,
    .param .u64 offsets_ptr,
    .param .u64 start_ptr,
    .param .u64 end_ptr,
    .param .u64 mask_ptr,
    .param .u32 row_count,
    .param .u32 start_len,
    .param .u32 end_len
)
{
    .reg .pred %p_out;
    .reg .pred %p_loop_done;
    .reg .pred %p_row_done;
    .reg .pred %p_bound_done;
    .reg .pred %p_lt;
    .reg .pred %p_gt;
    .reg .pred %p_ge_start;
    .reg .pred %p_lt_end;
    .reg .u16 %h_byte;
    .reg .u16 %b_byte;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_start_len;
    .reg .u32 %r_end_len;
    .reg .u32 %r_row_start;
    .reg .u32 %r_row_end;
    .reg .u32 %r_row_len;
    .reg .u32 %r_i;
    .reg .u32 %r_pos;
    .reg .u32 %r_mask;
    .reg .u64 %rd_bytes;
    .reg .u64 %rd_offsets;
    .reg .u64 %rd_start;
    .reg .u64 %rd_end;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset_addr;
    .reg .u64 %rd_next_offset_addr;
    .reg .u64 %rd_byte_offset;
    .reg .u64 %rd_hay_addr;
    .reg .u64 %rd_bound_addr;
    .reg .u64 %rd_mask_offset;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_bytes, [bytes_ptr];
    ld.param.u64 %rd_offsets, [offsets_ptr];
    ld.param.u64 %rd_start, [start_ptr];
    ld.param.u64 %rd_end, [end_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_row_count, [row_count];
    ld.param.u32 %r_start_len, [start_len];
    ld.param.u32 %r_end_len, [end_len];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset_addr, %r_idx, 4;
    add.u64 %rd_offset_addr, %rd_offsets, %rd_offset_addr;
    add.u64 %rd_next_offset_addr, %rd_offset_addr, 4;
    ld.global.u32 %r_row_start, [%rd_offset_addr];
    ld.global.u32 %r_row_end, [%rd_next_offset_addr];
    sub.u32 %r_row_len, %r_row_end, %r_row_start;

    mov.u32 %r_i, 0;
START_LOOP:
    setp.ge.u32 %p_row_done, %r_i, %r_row_len;
    setp.ge.u32 %p_bound_done, %r_i, %r_start_len;
    or.pred %p_loop_done, %p_row_done, %p_bound_done;
    @%p_loop_done bra START_PREFIX_DONE;

    add.u32 %r_pos, %r_row_start, %r_i;
    cvt.u64.u32 %rd_byte_offset, %r_pos;
    add.u64 %rd_hay_addr, %rd_bytes, %rd_byte_offset;
    cvt.u64.u32 %rd_byte_offset, %r_i;
    add.u64 %rd_bound_addr, %rd_start, %rd_byte_offset;
    ld.global.u8 %h_byte, [%rd_hay_addr];
    ld.global.u8 %b_byte, [%rd_bound_addr];
    setp.lt.u16 %p_lt, %h_byte, %b_byte;
    @%p_lt bra NO_MATCH;
    setp.gt.u16 %p_gt, %h_byte, %b_byte;
    @%p_gt bra START_MATCH;
    add.u32 %r_i, %r_i, 1;
    bra START_LOOP;

START_PREFIX_DONE:
    setp.ge.u32 %p_ge_start, %r_row_len, %r_start_len;
    @%p_ge_start bra START_MATCH;
    bra NO_MATCH;

START_MATCH:
    mov.u32 %r_i, 0;
END_LOOP:
    setp.ge.u32 %p_row_done, %r_i, %r_row_len;
    setp.ge.u32 %p_bound_done, %r_i, %r_end_len;
    or.pred %p_loop_done, %p_row_done, %p_bound_done;
    @%p_loop_done bra END_PREFIX_DONE;

    add.u32 %r_pos, %r_row_start, %r_i;
    cvt.u64.u32 %rd_byte_offset, %r_pos;
    add.u64 %rd_hay_addr, %rd_bytes, %rd_byte_offset;
    cvt.u64.u32 %rd_byte_offset, %r_i;
    add.u64 %rd_bound_addr, %rd_end, %rd_byte_offset;
    ld.global.u8 %h_byte, [%rd_hay_addr];
    ld.global.u8 %b_byte, [%rd_bound_addr];
    setp.lt.u16 %p_lt, %h_byte, %b_byte;
    @%p_lt bra MATCH;
    setp.gt.u16 %p_gt, %h_byte, %b_byte;
    @%p_gt bra NO_MATCH;
    add.u32 %r_i, %r_i, 1;
    bra END_LOOP;

END_PREFIX_DONE:
    setp.lt.u32 %p_lt_end, %r_row_len, %r_end_len;
    @%p_lt_end bra MATCH;
    bra NO_MATCH;

MATCH:
    mov.u32 %r_mask, 1;
    bra STORE;

NO_MATCH:
    mov.u32 %r_mask, 0;

STORE:
    mul.wide.u32 %rd_mask_offset, %r_idx, 4;
    add.u64 %rd_mask_addr, %rd_mask, %rd_mask_offset;
    st.global.u32 [%rd_mask_addr], %r_mask;

DONE:
    ret;
}
"#;

    let row_count = u32::try_from(input.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(input.len()))?;
    let start_len = u32::try_from(start_inclusive.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(start_inclusive.len()))?;
    let end_len = u32::try_from(end_exclusive.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(end_exclusive.len()))?;
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut offsets = Vec::with_capacity(input.len() + 1);
    let mut flattened = Vec::new();
    offsets.push(0_u32);
    for value in input {
        flattened.extend_from_slice(value);
        offsets.push(
            u32::try_from(flattened.len())
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(flattened.len()))?,
        );
    }

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
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
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

    let bytes_len = flattened.len().max(1);
    let mut device_bytes = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_bytes, bytes_len) })?;
    let bytes_guard = CudaDeviceAllocationGuard {
        ptr: device_bytes,
        free: *cu_mem_free,
    };

    let offsets_bytes = std::mem::size_of_val(offsets.as_slice());
    let mut device_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_offsets, offsets_bytes) })?;
    let offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_offsets,
        free: *cu_mem_free,
    };

    let start_bytes = start_inclusive.len().max(1);
    let mut device_start = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_start, start_bytes) })?;
    let start_guard = CudaDeviceAllocationGuard {
        ptr: device_start,
        free: *cu_mem_free,
    };

    let end_bytes = end_exclusive.len().max(1);
    let mut device_end = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_end, end_bytes) })?;
    let end_guard = CudaDeviceAllocationGuard {
        ptr: device_end,
        free: *cu_mem_free,
    };

    let mask_bytes = input.len() * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    if !flattened.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                bytes_guard.ptr,
                flattened.as_ptr().cast::<c_void>(),
                flattened.len(),
            )
        })?;
    }
    check_cuda(unsafe {
        cu_memcpy_htod(
            offsets_guard.ptr,
            offsets.as_ptr().cast::<c_void>(),
            offsets_bytes,
        )
    })?;
    if !start_inclusive.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                start_guard.ptr,
                start_inclusive.as_ptr().cast::<c_void>(),
                start_inclusive.len(),
            )
        })?;
    }
    if !end_exclusive.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                end_guard.ptr,
                end_exclusive.as_ptr().cast::<c_void>(),
                end_exclusive.len(),
            )
        })?;
    }

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
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_bytes_range_mask".as_ptr(),
        )
    })?;

    let mut bytes_arg = bytes_guard.ptr;
    let mut offsets_arg = offsets_guard.ptr;
    let mut start_arg = start_guard.ptr;
    let mut end_arg = end_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut row_count_arg = row_count;
    let mut start_len_arg = start_len;
    let mut end_len_arg = end_len;
    let mut args = [
        (&mut bytes_arg as *mut u64).cast::<c_void>(),
        (&mut offsets_arg as *mut u64).cast::<c_void>(),
        (&mut start_arg as *mut u64).cast::<c_void>(),
        (&mut end_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
        (&mut start_len_arg as *mut u32).cast::<c_void>(),
        (&mut end_len_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count.div_ceil(threads_per_block);
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
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; input.len()];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(end_guard);
    drop(start_guard);
    drop(offsets_guard);
    drop(bytes_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}
