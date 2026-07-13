use std::os::raw::c_void;

use super::resident_window::validate_text_windows;
use super::{check_cuda, launch_on_pooled_stream, CudaResidentDeviceMemory, CudaRuntimeProbeError};

impl CudaResidentDeviceMemory {
    pub fn count_text_prefix_from_payload(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        row_count: u64,
        prefix: &[u8],
    ) -> Result<u64, CudaRuntimeProbeError> {
        launch_cuda_resident_text_prefix_count(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            row_count,
            prefix,
        )
    }

    pub fn project_text_rows_from_payload(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        row_count: u64,
        row_indices: &[u64],
    ) -> Result<Vec<String>, CudaRuntimeProbeError> {
        copy_cuda_resident_text_rows(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            row_count,
            row_indices,
        )
    }
}

fn launch_cuda_resident_text_prefix_count(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    row_count: u64,
    prefix: &[u8],
) -> Result<u64, CudaRuntimeProbeError> {
    type CuMemsetD8Async = unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> i32;
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
    const PTX: &[u8] = include_bytes!("resident_text_prefix.ptx");

    validate_text_windows(
        resident.metadata().allocated_bytes,
        offsets_byte_offset,
        bytes_byte_offset,
        bytes_len,
        row_count,
    )?;
    let prefix_len = u64::try_from(prefix.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8_async = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8Async>(b"cuMemsetD8Async\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_resident_text_prefix_count", &ptx)?;
    let prefix_lease = primary.lease_device_buffer(prefix.len().max(1))?;

    const BLOCK: u32 = 256;
    let grid = row_count.div_ceil(u64::from(BLOCK)).clamp(1, 4096) as u32;
    let mut output = [0_u8; 16];
    launch_on_pooled_stream(resident, Some(&mut output), |stream, output_ptr| {
        let memset_rc = unsafe { cu_memset_d8_async(output_ptr, 0, 16, stream) };
        if memset_rc != 0 {
            return memset_rc;
        }
        if !prefix.is_empty() {
            let copy_rc = unsafe {
                htod_async(
                    prefix_lease.ptr,
                    prefix.as_ptr().cast::<c_void>(),
                    prefix.len(),
                    stream,
                )
            };
            if copy_rc != 0 {
                return copy_rc;
            }
        }
        let mut a0 = resident.device_ptr();
        let mut a1 = offsets_byte_offset;
        let mut a2 = bytes_byte_offset;
        let mut a3 = bytes_len;
        let mut a4 = prefix_lease.ptr;
        let mut a5 = prefix_len;
        let mut a6 = row_count;
        let mut a7 = output_ptr;
        let mut a8 = output_ptr + 8;
        let mut args = [
            (&mut a0 as *mut u64).cast::<c_void>(),
            (&mut a1 as *mut u64).cast::<c_void>(),
            (&mut a2 as *mut u64).cast::<c_void>(),
            (&mut a3 as *mut u64).cast::<c_void>(),
            (&mut a4 as *mut u64).cast::<c_void>(),
            (&mut a5 as *mut u64).cast::<c_void>(),
            (&mut a6 as *mut u64).cast::<c_void>(),
            (&mut a7 as *mut u64).cast::<c_void>(),
            (&mut a8 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                function,
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

    let malformed = u32::from_le_bytes(output[8..12].try_into().unwrap());
    if malformed != 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    Ok(u64::from_le_bytes(output[..8].try_into().unwrap()))
}

fn copy_cuda_resident_text_rows(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    row_count: u64,
    row_indices: &[u64],
) -> Result<Vec<String>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    validate_text_windows(
        resident.metadata().allocated_bytes,
        offsets_byte_offset,
        bytes_byte_offset,
        bytes_len,
        row_count,
    )?;
    if row_indices.iter().any(|&row_idx| row_idx >= row_count) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    let primary = resident.primary();
    primary.set_current()?;
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut first_offset = 0_u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut first_offset as *mut u64).cast::<c_void>(),
            resident.device_ptr() + offsets_byte_offset,
            std::mem::size_of::<u64>(),
        )
    })?;
    if first_offset != 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(first_offset).unwrap_or(usize::MAX),
        ));
    }
    if row_indices.is_empty() {
        return Ok(Vec::new());
    }

    let min_row_idx = row_indices.iter().copied().min().unwrap_or(0);
    let max_row_idx = row_indices.iter().copied().max().unwrap_or(0);
    let offset_count = max_row_idx
        .checked_sub(min_row_idx)
        .and_then(|span| span.checked_add(2))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let offsets_offset = min_row_idx
        .checked_mul(std::mem::size_of::<u64>() as u64)
        .and_then(|offset| offsets_byte_offset.checked_add(offset))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let offsets_end = offsets_byte_offset
        .checked_add(
            max_row_idx
                .checked_add(2)
                .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        )
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_end = bytes_byte_offset
        .checked_add(bytes_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if offsets_end > resident.metadata().allocated_bytes
        || bytes_end > resident.metadata().allocated_bytes
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            offsets_end.max(bytes_end) as usize,
        ));
    }

    let mut offsets = vec![
        0_u64;
        usize::try_from(offset_count).map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
        })?
    ];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            offsets.as_mut_ptr().cast::<c_void>(),
            resident.device_ptr() + offsets_offset,
            offsets
                .len()
                .checked_mul(std::mem::size_of::<u64>())
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        )
    })?;

    let mut spans = Vec::with_capacity(row_indices.len());
    let mut min_text_start = u64::MAX;
    let mut max_text_end = 0_u64;
    for row_idx in row_indices {
        let offset_idx = row_idx
            .checked_sub(min_row_idx)
            .and_then(|idx| usize::try_from(idx).ok())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let start = offsets[offset_idx];
        let end = offsets[offset_idx + 1];
        if start > end || end > bytes_len {
            return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
        }
        min_text_start = min_text_start.min(start);
        max_text_end = max_text_end.max(end);
        spans.push((start, end));
    }

    let text_span_len = max_text_end
        .checked_sub(min_text_start)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut text_bytes = vec![
        0_u8;
        usize::try_from(text_span_len).map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
        })?
    ];
    if text_span_len > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                text_bytes.as_mut_ptr().cast::<c_void>(),
                resident.device_ptr() + bytes_byte_offset + min_text_start,
                text_bytes.len(),
            )
        })?;
    }

    let mut values = Vec::with_capacity(row_indices.len());
    for (start, end) in spans {
        let value_len = usize::try_from(end - start)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let value_start = start
            .checked_sub(min_text_start)
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let value_end = value_start
            .checked_add(value_len)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let value = std::str::from_utf8(&text_bytes[value_start..value_end])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(value_len))?;
        values.push(value.to_string());
    }
    Ok(values)
}
