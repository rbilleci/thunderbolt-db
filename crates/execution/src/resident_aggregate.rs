use std::ffi::c_void;

use super::{CudaResidentDeviceMemory, CudaRuntimeProbeError, check_cuda, launch_on_pooled_stream};

/// SUM(int4) over a filtered set of row indices (the operator axis, doc 19): H2D the surviving u32
/// indices, zero a single i64 accumulator, run the gather-reduce kernel (local sum per thread + one
/// `atom.add.u64`), and D2H the bigint sum. `indices` must be non-empty (an empty aggregate is NULL,
/// hard-errored upstream until M3).
pub(super) fn launch_cuda_resident_i32_sum_at_indices(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    indices: &[u32],
) -> Result<i64, CudaRuntimeProbeError> {
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
    type CuMemsetD8Async = unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if indices.is_empty() {
        return Ok(0);
    }
    let count = indices.len();
    let idx_bytes = count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count))?;
    let count_u64 = count as u64;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_memset_d8_async = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8Async>(b"cuMemsetD8Async\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_resident_i32_sum_at_indices", &ptx)?;
    let indices_dev = primary.lease_device_buffer(idx_bytes)?;
    let out = primary.lease_device_buffer(std::mem::size_of::<i64>())?;

    const BLOCK: u32 = 256;
    let grid = count_u64.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = byte_offset;
    let mut a2 = indices_dev.ptr;
    let mut a3 = count_u64;
    let mut a4 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                idx_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe { cu_memset_d8_async(out.ptr, 0, std::mem::size_of::<i64>(), stream) };
        if rc != 0 {
            return rc;
        }
        unsafe {
            cu_launch_kernel(
                kernel_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    let mut sum = 0_i64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut sum as *mut i64).cast::<c_void>(),
            out.ptr,
            std::mem::size_of::<i64>(),
        )
    })?;
    Ok(sum)
}

/// SUM(int8) over a filtered set of row indices, as i128 (the operator axis, doc 19): H2D the u32
/// indices, zero a 16-byte (two-limb i128) accumulator, run the i128 gather-reduce kernel (local i128
/// sum per thread + a two-64-bit-atomic carry add), and D2H the 16 bytes as a little-endian i128.
/// `indices` must be non-empty (an empty SUM is NULL, hard-errored upstream until M3).
pub(super) fn launch_cuda_resident_i64_sum_at_indices_i128(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    indices: &[u32],
) -> Result<i128, CudaRuntimeProbeError> {
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
    type CuMemsetD8Async = unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if indices.is_empty() {
        return Ok(0);
    }
    let count = indices.len();
    let idx_bytes = count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count))?;
    let count_u64 = count as u64;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_memset_d8_async = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8Async>(b"cuMemsetD8Async\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_resident_i64_sum_at_indices_i128", &ptx)?;
    let indices_dev = primary.lease_device_buffer(idx_bytes)?;
    let out = primary.lease_device_buffer(std::mem::size_of::<i128>())?;

    const BLOCK: u32 = 256;
    let grid = count_u64.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = byte_offset;
    let mut a2 = indices_dev.ptr;
    let mut a3 = count_u64;
    let mut a4 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                idx_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe { cu_memset_d8_async(out.ptr, 0, std::mem::size_of::<i128>(), stream) };
        if rc != 0 {
            return rc;
        }
        unsafe {
            cu_launch_kernel(
                kernel_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    let mut bytes = [0u8; std::mem::size_of::<i128>()];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            bytes.as_mut_ptr().cast::<c_void>(),
            out.ptr,
            std::mem::size_of::<i128>(),
        )
    })?;
    Ok(i128::from_le_bytes(bytes))
}

/// MIN (`is_max=false`) or MAX (`is_max=true`) of a resident int4 column over a filtered set of row
/// indices (the operator axis, doc 19): H2D the surviving u32 indices, H2D the accumulator's identity
/// (i32::MAX for min / i32::MIN for max -- not a memset byte pattern), run the reduction kernel (local
/// min/max per thread + one predicated `atom.min/max.s32`), and D2H the i32. `indices` must be
/// non-empty (an empty MIN/MAX is NULL, hard-errored upstream until M3).
pub(super) fn launch_cuda_resident_i32_minmax_at_indices(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    indices: &[u32],
    is_max: bool,
) -> Result<i32, CudaRuntimeProbeError> {
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
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if indices.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let count = indices.len();
    let idx_bytes = count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count))?;
    let count_u64 = count as u64;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_resident_i32_minmax_at_indices", &ptx)?;
    let indices_dev = primary.lease_device_buffer(idx_bytes)?;
    let out = primary.lease_device_buffer(std::mem::size_of::<i32>())?;

    // Accumulator identity: min reduces against i32::MAX, max against i32::MIN.
    let init: i32 = if is_max { i32::MIN } else { i32::MAX };
    let op: u32 = u32::from(is_max);

    const BLOCK: u32 = 256;
    let grid = count_u64.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = byte_offset;
    let mut a2 = indices_dev.ptr;
    let mut a3 = count_u64;
    let mut a4 = op;
    let mut a5 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u32).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                idx_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                out.ptr,
                (&init as *const i32).cast::<c_void>(),
                std::mem::size_of::<i32>(),
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        unsafe {
            cu_launch_kernel(
                kernel_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    let mut result = 0_i32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut result as *mut i32).cast::<c_void>(),
            out.ptr,
            std::mem::size_of::<i32>(),
        )
    })?;
    Ok(result)
}

/// MIN (`is_max=false`) / MAX (`is_max=true`) of a resident INT8 column over a filtered set of row
/// indices (the operator axis, doc 19): H2D the surviving u32 indices, H2D the i64 identity (i64::MAX
/// for min / i64::MIN for max), run the reduction kernel (2x4-byte i64 loads + predicated
/// `atom.min/max.s64`), D2H the i64. `indices` must be non-empty.
pub(super) fn launch_cuda_resident_i64_minmax_at_indices(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    indices: &[u32],
    is_max: bool,
) -> Result<i64, CudaRuntimeProbeError> {
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
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if indices.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let count = indices.len();
    let idx_bytes = count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count))?;
    let count_u64 = count as u64;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_resident_i64_minmax_at_indices", &ptx)?;
    let indices_dev = primary.lease_device_buffer(idx_bytes)?;
    let out = primary.lease_device_buffer(std::mem::size_of::<i64>())?;

    let init: i64 = if is_max { i64::MIN } else { i64::MAX };
    let op: u32 = u32::from(is_max);

    const BLOCK: u32 = 256;
    let grid = count_u64.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = byte_offset;
    let mut a2 = indices_dev.ptr;
    let mut a3 = count_u64;
    let mut a4 = op;
    let mut a5 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u32).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                idx_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                out.ptr,
                (&init as *const i64).cast::<c_void>(),
                std::mem::size_of::<i64>(),
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        unsafe {
            cu_launch_kernel(
                kernel_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    let mut result = 0_i64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut result as *mut i64).cast::<c_void>(),
            out.ptr,
            std::mem::size_of::<i64>(),
        )
    })?;
    Ok(result)
}

/// MIN (`is_max=false`) / MAX (`is_max=true`) of a resident NUMERIC (i128 mantissa) column over a
/// filtered set of row indices, returned as the i128 mantissa (the operator axis, doc 19). No native
/// 128-bit atomic min/max, so the kernel runs a BOUNDED grid where each thread reduces its strided
/// slice to a local i128 partial; the host combines the (<= 16384) partials. `indices` must be
/// non-empty.
pub(super) fn launch_cuda_resident_i128_minmax_partials_at_indices(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    indices: &[u32],
    is_max: bool,
) -> Result<i128, CudaRuntimeProbeError> {
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
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if indices.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let count = indices.len();
    let idx_bytes = count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count))?;
    let count_u64 = count as u64;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn =
        primary.cached_function(c"gpu_db_resident_i128_minmax_partials_at_indices", &ptx)?;

    // Bounded grid: cap the partials count (and the host combine) at 64 * 256 = 16384.
    const BLOCK: u32 = 256;
    const MAX_GRID: u32 = 64;
    let grid = (count_u64
        .div_ceil(u64::from(BLOCK))
        .clamp(1, u64::from(MAX_GRID))) as u32;
    let num_threads = (grid * BLOCK) as usize;
    let partials_bytes = num_threads
        .checked_mul(std::mem::size_of::<i128>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(num_threads))?;
    let indices_dev = primary.lease_device_buffer(idx_bytes)?;
    let partials = primary.lease_device_buffer(partials_bytes)?;

    let op: u32 = u32::from(is_max);
    let mut a0 = resident.device_ptr();
    let mut a1 = byte_offset;
    let mut a2 = indices_dev.ptr;
    let mut a3 = count_u64;
    let mut a4 = op;
    let mut a5 = partials.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u32).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                idx_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        unsafe {
            cu_launch_kernel(
                kernel_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    let mut bytes = vec![0u8; partials_bytes];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            bytes.as_mut_ptr().cast::<c_void>(),
            partials.ptr,
            partials_bytes,
        )
    })?;
    // Host-combine the bounded partials (idle threads wrote the identity, which min/max ignores).
    let mut result: Option<i128> = None;
    for chunk in bytes.chunks_exact(std::mem::size_of::<i128>()) {
        let value = i128::from_le_bytes(chunk.try_into().expect("16-byte chunk"));
        result = Some(match result {
            None => value,
            Some(acc) if is_max => acc.max(value),
            Some(acc) => acc.min(value),
        });
    }
    result.ok_or(CudaRuntimeProbeError::InvalidInputLength(0))
}

/// SUM of a resident NUMERIC (i128 mantissa) column over a filtered set of row indices, as the i128
/// mantissa with CHECKED i128 overflow (the operator axis, doc 19). Bounded-grid partials reduction:
/// each thread sums its strided slice into a local i128 (and sets a device overflow flag if any of
/// its adds overflows i128); the host checked-combines the partials. Either the device flag OR a
/// host-combine overflow yields `NumericFieldOverflow` (PG `numeric field overflow`). `indices`
/// must be non-empty.
pub(super) fn launch_cuda_resident_i128_sum_partials_at_indices(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    indices: &[u32],
) -> Result<i128, CudaRuntimeProbeError> {
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
    type CuMemsetD8Async = unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if indices.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let count = indices.len();
    let idx_bytes = count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count))?;
    let count_u64 = count as u64;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_memset_d8_async = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8Async>(b"cuMemsetD8Async\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn =
        primary.cached_function(c"gpu_db_resident_i128_sum_partials_at_indices", &ptx)?;

    const BLOCK: u32 = 256;
    const MAX_GRID: u32 = 64;
    let grid = (count_u64
        .div_ceil(u64::from(BLOCK))
        .clamp(1, u64::from(MAX_GRID))) as u32;
    let num_threads = (grid * BLOCK) as usize;
    let partials_bytes = num_threads
        .checked_mul(std::mem::size_of::<i128>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(num_threads))?;
    let indices_dev = primary.lease_device_buffer(idx_bytes)?;
    let partials = primary.lease_device_buffer(partials_bytes)?;
    let flag = primary.lease_device_buffer(std::mem::size_of::<u32>())?;

    let mut a0 = resident.device_ptr();
    let mut a1 = byte_offset;
    let mut a2 = indices_dev.ptr;
    let mut a3 = count_u64;
    let mut a4 = partials.ptr;
    let mut a5 = flag.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                idx_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe { cu_memset_d8_async(flag.ptr, 0, std::mem::size_of::<u32>(), stream) };
        if rc != 0 {
            return rc;
        }
        unsafe {
            cu_launch_kernel(
                kernel_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    let mut flag_host = 0u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut flag_host as *mut u32).cast::<c_void>(),
            flag.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;
    if flag_host != 0 {
        return Err(CudaRuntimeProbeError::NumericFieldOverflow);
    }
    let mut bytes = vec![0u8; partials_bytes];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            bytes.as_mut_ptr().cast::<c_void>(),
            partials.ptr,
            partials_bytes,
        )
    })?;
    // Host checked-combine of the bounded partials (idle threads wrote 0). A combine overflow is the
    // same PG `numeric field overflow` as a per-thread overflow.
    let mut sum: i128 = 0;
    for chunk in bytes.chunks_exact(std::mem::size_of::<i128>()) {
        let value = i128::from_le_bytes(chunk.try_into().expect("16-byte chunk"));
        sum = sum
            .checked_add(value)
            .ok_or(CudaRuntimeProbeError::NumericFieldOverflow)?;
    }
    Ok(sum)
}
