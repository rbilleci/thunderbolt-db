//! Host-staged CUDA MVCC bootstrap operations.
//!
//! These launchers preserve the generic CUDA-MVCC compatibility path. They upload host MVCC arrays
//! on every call and are RETIRE-003 debt, not the GPU-resident product execution boundary.

use std::os::raw::c_void;

use libloading::Library;

use crate::cuda_context::{
    check_cuda, CudaContextGuard, CudaDeviceAllocationGuard, CudaModuleGuard,
};
use crate::{CudaMvccRowBatch, CudaRuntimeProbeError};

pub(super) fn launch_cuda_mvcc_row_batch_lengths(
    batch: &CudaMvccRowBatch,
) -> Result<Vec<(u32, u32)>, CudaRuntimeProbeError> {
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

.visible .entry gpu_db_cuda_mvcc_row_batch_lengths(
    .param .u64 key_offsets_ptr,
    .param .u64 value_offsets_ptr,
    .param .u64 output_ptr,
    .param .u32 row_count
)
{
    .reg .pred %p_out;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_key_start;
    .reg .u32 %r_key_end;
    .reg .u32 %r_value_start;
    .reg .u32 %r_value_end;
    .reg .u32 %r_key_len;
    .reg .u32 %r_value_len;
    .reg .u64 %rd_key_offsets;
    .reg .u64 %rd_value_offsets;
    .reg .u64 %rd_output;
    .reg .u64 %rd_offset;
    .reg .u64 %rd_next_offset;
    .reg .u64 %rd_output_offset;
    .reg .u64 %addr;

    ld.param.u64 %rd_key_offsets, [key_offsets_ptr];
    ld.param.u64 %rd_value_offsets, [value_offsets_ptr];
    ld.param.u64 %rd_output, [output_ptr];
    ld.param.u32 %r_row_count, [row_count];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset, %r_idx, 4;
    add.u64 %addr, %rd_key_offsets, %rd_offset;
    add.u64 %rd_next_offset, %addr, 4;
    ld.global.u32 %r_key_start, [%addr];
    ld.global.u32 %r_key_end, [%rd_next_offset];
    sub.u32 %r_key_len, %r_key_end, %r_key_start;

    add.u64 %addr, %rd_value_offsets, %rd_offset;
    add.u64 %rd_next_offset, %addr, 4;
    ld.global.u32 %r_value_start, [%addr];
    ld.global.u32 %r_value_end, [%rd_next_offset];
    sub.u32 %r_value_len, %r_value_end, %r_value_start;

    mul.wide.u32 %rd_output_offset, %r_idx, 8;
    add.u64 %addr, %rd_output, %rd_output_offset;
    st.global.u32 [%addr], %r_key_len;
    add.u64 %addr, %addr, 4;
    st.global.u32 [%addr], %r_value_len;

DONE:
    ret;
}
"#;

    batch.validate()?;
    if batch.row_count == 0 {
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

    let key_offsets_bytes = std::mem::size_of_val(batch.key_offsets.as_slice());
    let mut device_key_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_key_offsets, key_offsets_bytes) })?;
    let key_offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_key_offsets,
        free: *cu_mem_free,
    };

    let value_offsets_bytes = std::mem::size_of_val(batch.value_offsets.as_slice());
    let mut device_value_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_value_offsets, value_offsets_bytes) })?;
    let value_offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_value_offsets,
        free: *cu_mem_free,
    };

    let output_words = batch.row_count as usize * 2;
    let output_bytes = output_words * std::mem::size_of::<u32>();
    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, output_bytes) })?;
    let output_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    check_cuda(unsafe {
        cu_memcpy_htod(
            key_offsets_guard.ptr,
            batch.key_offsets.as_ptr().cast::<c_void>(),
            key_offsets_bytes,
        )
    })?;
    check_cuda(unsafe {
        cu_memcpy_htod(
            value_offsets_guard.ptr,
            batch.value_offsets.as_ptr().cast::<c_void>(),
            value_offsets_bytes,
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
            c"gpu_db_cuda_mvcc_row_batch_lengths".as_ptr(),
        )
    })?;

    let mut key_offsets_arg = key_offsets_guard.ptr;
    let mut value_offsets_arg = value_offsets_guard.ptr;
    let mut output_arg = output_guard.ptr;
    let mut row_count_arg = batch.row_count;
    let mut args = [
        (&mut key_offsets_arg as *mut u64).cast::<c_void>(),
        (&mut value_offsets_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = batch.row_count.div_ceil(threads_per_block);
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

    let mut output = vec![0_u32; output_words];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            output.as_mut_ptr().cast::<c_void>(),
            output_guard.ptr,
            output_bytes,
        )
    })?;

    drop(module_guard);
    drop(output_guard);
    drop(value_offsets_guard);
    drop(key_offsets_guard);
    drop(context_guard);

    Ok(output
        .chunks_exact(2)
        .map(|lengths| (lengths[0], lengths[1]))
        .collect())
}

pub(super) fn launch_cuda_mvcc_visibility_mask(
    batch: &CudaMvccRowBatch,
    read_txn_id: u64,
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

.visible .entry gpu_db_cuda_mvcc_visibility_mask(
    .param .u64 begin_txn_ids_ptr,
    .param .u64 end_txn_ids_ptr,
    .param .u64 mask_ptr,
    .param .u64 read_txn_id,
    .param .u32 row_count
)
{
    .reg .pred %p_out;
    .reg .pred %p_created;
    .reg .pred %p_not_deleted;
    .reg .pred %p_visible;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_mask_value;
    .reg .u64 %rd_begin;
    .reg .u64 %rd_end;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_read_txn_id;
    .reg .u64 %rd_offset8;
    .reg .u64 %rd_offset4;
    .reg .u64 %addr;
    .reg .u64 %rd_created_by;
    .reg .u64 %rd_deleted_by;

    ld.param.u64 %rd_begin, [begin_txn_ids_ptr];
    ld.param.u64 %rd_end, [end_txn_ids_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u64 %rd_read_txn_id, [read_txn_id];
    ld.param.u32 %r_row_count, [row_count];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset8, %r_idx, 8;
    add.u64 %addr, %rd_begin, %rd_offset8;
    ld.global.u64 %rd_created_by, [%addr];
    add.u64 %addr, %rd_end, %rd_offset8;
    ld.global.u64 %rd_deleted_by, [%addr];

    setp.le.u64 %p_created, %rd_created_by, %rd_read_txn_id;
    setp.gt.u64 %p_not_deleted, %rd_deleted_by, %rd_read_txn_id;
    and.pred %p_visible, %p_created, %p_not_deleted;
    selp.u32 %r_mask_value, 1, 0, %p_visible;

    mul.wide.u32 %rd_offset4, %r_idx, 4;
    add.u64 %addr, %rd_mask, %rd_offset4;
    st.global.u32 [%addr], %r_mask_value;

DONE:
    ret;
}
"#;

    batch.validate()?;
    if batch.row_count == 0 {
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

    let begin_bytes = std::mem::size_of_val(batch.begin_txn_ids.as_slice());
    let mut device_begin = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_begin, begin_bytes) })?;
    let begin_guard = CudaDeviceAllocationGuard {
        ptr: device_begin,
        free: *cu_mem_free,
    };

    let end_bytes = std::mem::size_of_val(batch.end_txn_ids.as_slice());
    let mut device_end = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_end, end_bytes) })?;
    let end_guard = CudaDeviceAllocationGuard {
        ptr: device_end,
        free: *cu_mem_free,
    };

    let mask_bytes = batch.row_count as usize * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    check_cuda(unsafe {
        cu_memcpy_htod(
            begin_guard.ptr,
            batch.begin_txn_ids.as_ptr().cast::<c_void>(),
            begin_bytes,
        )
    })?;
    check_cuda(unsafe {
        cu_memcpy_htod(
            end_guard.ptr,
            batch.end_txn_ids.as_ptr().cast::<c_void>(),
            end_bytes,
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
            c"gpu_db_cuda_mvcc_visibility_mask".as_ptr(),
        )
    })?;

    let mut begin_arg = begin_guard.ptr;
    let mut end_arg = end_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut read_txn_id_arg = read_txn_id;
    let mut row_count_arg = batch.row_count;
    let mut args = [
        (&mut begin_arg as *mut u64).cast::<c_void>(),
        (&mut end_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut read_txn_id_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = batch.row_count.div_ceil(threads_per_block);
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

    let mut mask = vec![0_u32; batch.row_count as usize];
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
    drop(begin_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}
