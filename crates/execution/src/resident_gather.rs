use std::ffi::c_void;

use super::{check_cuda, launch_on_pooled_stream, CudaResidentDeviceMemory, CudaRuntimeProbeError};

/// Shared fixed-width row gather (the type matrix): HtoD `row_indices` once, launch `kernel_name`
/// (out[i] = the `elem_size`-byte resident column element at the row index `row_indices[i]`, grid-stride),
/// then ONE bulk D2H into `out_host` (the caller sized it to `row_indices.len() * elem_size` bytes).
/// Replaces the per-row synchronous D2H gather for i32 / i64 / i128 projections (one kernel + one bulk D2H
/// vs n tiny D2H copies). The indices are bounds-checked host-side FIRST -- an OOB index would be an
/// illegal DEVICE read in the kernel (CUDA 700, context-corrupting on a shared box). `n == 0` is a no-op.
/// SAFETY: `out_host` must point to >= `row_indices.len() * elem_size` writable bytes and `kernel_name`
/// must read/write `elem_size`-byte elements addressed `resident_ptr + byte_offset + idx*elem_size`.
/// Raw resident row-gather launch, shared by the fixed-width AND bool gathers: HtoD `row_indices` once,
/// launch the 5-arg `kernel_name` (resident_ptr, `arg1`, indices, n, out), then ONE bulk D2H of `out_bytes`
/// into the host buffer `out_host`. `arg1` is the kernel's second parameter -- a column `byte_offset` for
/// the fixed-width gathers, a `bitmap_byte_offset` for the bool gather. `n == 0` is a no-op.
///
/// NO bounds check here -- the CALLER MUST validate every index in-bounds first (an OOB index is an illegal
/// DEVICE read = CUDA 700, context-corrupting on a shared box). The kernel's writes to `out` are visible
/// before the D2H: `launch_on_pooled_stream` blocking-syncs the stream before returning, then the legacy
/// synchronous D2H reads `out`. SAFETY: `out_host` must point to >= `out_bytes` writable bytes matching the
/// kernel's output element width.
fn gather_resident_kernel(
    resident: &CudaResidentDeviceMemory,
    arg1: u64,
    row_indices: &[u64],
    kernel_name: &'static core::ffi::CStr,
    out_host: *mut c_void,
    out_bytes: usize,
) -> Result<(), CudaRuntimeProbeError> {
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

    let n = row_indices.len();
    if n == 0 {
        return Ok(());
    }

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
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
    let gather_fn = primary.cached_function(kernel_name, &ptx)?;

    // HtoD the indices once (blocking; on-device on return), run the gather in ONE kernel, then ONE bulk
    // D2H of the whole result buffer. Both leases are held alive past the D2H below.
    let idx_buf = resident.upload_u64_device(row_indices)?;
    let out = primary.lease_device_buffer(out_bytes)?;

    const BLOCK: u32 = 256;
    let grid = (n as u64).div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = arg1;
    let mut a2 = idx_buf.device_ptr();
    let mut a3 = n as u64;
    let mut a4 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            gather_fn,
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
    })?;

    // The explicit cuStreamSynchronize inside launch_on_pooled_stream (plus the legacy synchronous D2H)
    // makes the kernel's writes to `out` visible before this read.
    check_cuda(unsafe { cu_memcpy_dtoh(out_host, out.ptr, out_bytes) })?;
    drop(idx_buf);
    Ok(())
}

/// Fixed-width row gather (i32/i64/i128): bounds-check every index, then [`gather_resident_kernel`] with
/// `out_bytes = n * elem_size`. `out_host` must hold `row_indices.len() * elem_size` writable bytes, and
/// `kernel_name` must read/write `elem_size`-byte elements addressed `resident_ptr + byte_offset +
/// idx*elem_size`. `n == 0` is a no-op.
fn gather_resident_fixed_rows(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_indices: &[u64],
    elem_size: u64,
    kernel_name: &'static core::ffi::CStr,
    out_host: *mut c_void,
) -> Result<(), CudaRuntimeProbeError> {
    let n = row_indices.len();
    if n == 0 {
        return Ok(());
    }
    let out_bytes = n
        .checked_mul(elem_size as usize)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;

    // Bounds safety: the kernel reads `resident_ptr + byte_offset + idx*elem_size` for `elem_size` bytes,
    // so validate every index's element lies within the allocation FIRST (an OOB index is CUDA 700). O(n)
    // checked arithmetic, no transfers (the win was removing the per-row D2H, not this check).
    let allocated = resident.metadata().allocated_bytes;
    for &row_idx in row_indices {
        let value_end = row_idx
            .checked_mul(elem_size)
            .and_then(|offset| byte_offset.checked_add(offset))
            .and_then(|offset| offset.checked_add(elem_size))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if value_end > allocated {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                value_end as usize,
            ));
        }
    }

    gather_resident_kernel(
        resident,
        byte_offset,
        row_indices,
        kernel_name,
        out_host,
        out_bytes,
    )
}

/// Gather a resident int4 column at `row_indices` into host `i32` values: one
/// `gpu_db_resident_i32_gather_rows` launch + one bulk D2H, via [`gather_resident_fixed_rows`].
pub(super) fn copy_cuda_resident_i32_rows(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_indices: &[u64],
) -> Result<Vec<i32>, CudaRuntimeProbeError> {
    let mut values = vec![0_i32; row_indices.len()];
    gather_resident_fixed_rows(
        resident,
        byte_offset,
        row_indices,
        std::mem::size_of::<i32>() as u64,
        c"gpu_db_resident_i32_gather_rows",
        values.as_mut_ptr().cast::<c_void>(),
    )?;
    Ok(values)
}

/// Gather a resident bool BITMAP (1 bit/row) at `row_indices` into host `bool` values: one
/// `gpu_db_resident_bool_gather_rows` launch (each thread reads the u32 word at `bitmap_byte_offset +
/// (idx>>5)*4`, extracts bit `idx&31`, writes a 0/1 u8) + ONE bulk D2H, decoded to bool. Mirrors
/// [`gather_resident_fixed_rows`] but with bitmap-word addressing + a u8-per-row output. Runs for bool
/// projections AND the per-column NULL VALIDITY bitmap, so it fires on every nullable projected column.
pub(super) fn copy_cuda_resident_bool_rows(
    resident: &CudaResidentDeviceMemory,
    bitmap_byte_offset: u64,
    row_indices: &[u64],
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    let n = row_indices.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    // Bounds safety: the kernel reads the u32 word at `bitmap_byte_offset + (idx>>5)*4` -- validate every
    // word lies within the allocation FIRST (an OOB read is CUDA 700). O(n) checked arithmetic, no transfers.
    let allocated = resident.metadata().allocated_bytes;
    for &row_idx in row_indices {
        let word_end = (row_idx >> 5)
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .and_then(|offset| bitmap_byte_offset.checked_add(offset))
            .and_then(|offset| offset.checked_add(std::mem::size_of::<u32>() as u64))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if word_end > allocated {
            return Err(CudaRuntimeProbeError::InvalidInputLength(word_end as usize));
        }
    }

    // One u8 per row: the kernel writes 0/1, gathered in bulk, then decoded to bool.
    let mut bytes = vec![0_u8; n];
    gather_resident_kernel(
        resident,
        bitmap_byte_offset,
        row_indices,
        c"gpu_db_resident_bool_gather_rows",
        bytes.as_mut_ptr().cast::<c_void>(),
        n,
    )?;
    Ok(bytes.into_iter().map(|b| b != 0).collect())
}

/// int8 (s64) analog of [`copy_cuda_resident_i32_rows`] (the type matrix, docs/architecture/19):
/// gather the int8 column at `byte_offset` for the given device row indices into host `i64` values,
/// 8 bytes per row. Used to materialize an int8 projection at the surviving rows.
pub(super) fn copy_cuda_resident_i64_rows(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_indices: &[u64],
) -> Result<Vec<i64>, CudaRuntimeProbeError> {
    let mut values = vec![0_i64; row_indices.len()];
    gather_resident_fixed_rows(
        resident,
        byte_offset,
        row_indices,
        std::mem::size_of::<i64>() as u64,
        c"gpu_db_resident_i64_gather_rows",
        values.as_mut_ptr().cast::<c_void>(),
    )?;
    Ok(values)
}

/// Gather the 16-byte i128 mantissas of `row_indices` from a resident numeric column (the type matrix,
/// doc 19). Little-endian on both device and host, so a 16-byte DtoH lands directly in an `i128`.
pub(super) fn copy_cuda_resident_i128_rows(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_indices: &[u64],
) -> Result<Vec<i128>, CudaRuntimeProbeError> {
    let mut values = vec![0_i128; row_indices.len()];
    gather_resident_fixed_rows(
        resident,
        byte_offset,
        row_indices,
        std::mem::size_of::<i128>() as u64,
        c"gpu_db_resident_i128_gather_rows",
        values.as_mut_ptr().cast::<c_void>(),
    )?;
    Ok(values)
}
