use std::ffi::c_void;

use super::resident_window::{validate_text_windows, validate_window};
use super::{
    CudaResidentDeviceMemory, CudaRuntimeProbeError, compact_mask_i32_to_indices,
    launch_on_pooled_stream,
};

fn validate_bitmap_windows(
    allocated_bytes: u64,
    bitmap_offsets: &[u64],
    row_count: u64,
) -> Result<(), CudaRuntimeProbeError> {
    let word_count = row_count.div_ceil(32);
    for &offset in bitmap_offsets {
        validate_window(
            allocated_bytes,
            offset,
            word_count,
            std::mem::size_of::<u32>() as u64,
        )?;
    }
    Ok(())
}

/// Evaluate `col <cmp> scalar` (or `scalar <cmp> col` when `scalar_on_left`) over a resident int8
/// column to surviving row indices (the type matrix, doc 19): run
/// `gpu_db_resident_i64_compare_scalar_to_mask` to a 0/1 mask, then compact it with the type-agnostic
/// `gpu_db_mask_compact_to_indices`. `comparison` 0=eq/1=lt/2=le/3=gt/4=ge/5=ne; indices host-sorted.
pub(super) fn launch_cuda_resident_i64_compare_scalar_filter(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    scalar: i64,
    scalar_on_left: bool,
    comparison: u32,
    n: u64,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    validate_window(
        resident.metadata().allocated_bytes,
        byte_offset,
        n,
        std::mem::size_of::<i64>() as u64,
    )?;
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let compare_fn =
        primary.cached_function(c"gpu_db_resident_i64_compare_scalar_to_mask", &ptx)?;
    let mask = primary.lease_device_buffer(mask_bytes)?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = byte_offset;
    let mut a2 = scalar;
    let mut a3 = u32::from(scalar_on_left);
    let mut a4 = comparison;
    let mut a5 = n;
    let mut a6 = mask.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut i64).cast::<c_void>(),
        (&mut a3 as *mut u32).cast::<c_void>(),
        (&mut a4 as *mut u32).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
        (&mut a6 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            compare_fn,
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
    compact_mask_i32_to_indices(resident, mask.ptr, n)
}

/// Evaluate `a <cmp> b` over two resident int8 columns to surviving row indices (the type matrix, doc
/// 19): `gpu_db_resident_i64_compare_columns_to_mask` to a mask, then the type-agnostic compactor.
pub(super) fn launch_cuda_resident_i64_compare_columns_filter(
    resident: &CudaResidentDeviceMemory,
    a_byte_offset: u64,
    b_byte_offset: u64,
    comparison: u32,
    n: u64,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    for offset in [a_byte_offset, b_byte_offset] {
        validate_window(
            resident.metadata().allocated_bytes,
            offset,
            n,
            std::mem::size_of::<i64>() as u64,
        )?;
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let compare_fn =
        primary.cached_function(c"gpu_db_resident_i64_compare_columns_to_mask", &ptx)?;
    let mask = primary.lease_device_buffer(mask_bytes)?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = a_byte_offset;
    let mut a2 = b_byte_offset;
    let mut a3 = comparison;
    let mut a4 = n;
    let mut a5 = mask.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u32).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            compare_fn,
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
    compact_mask_i32_to_indices(resident, mask.ptr, n)
}

/// Evaluate `col <cmp> scalar` (or `scalar <cmp> col` when `scalar_on_left`) over a resident numeric
/// (i128) column to surviving row indices (the type matrix, doc 19): run
/// `gpu_db_resident_i128_compare_scalar_to_mask` (the scalar mantissa passed as low/high u64 limbs) to
/// a 0/1 mask, then the type-agnostic compactor. `comparison` 0=eq/1=lt/2=le/3=gt/4=ge/5=ne.
pub(super) fn launch_cuda_resident_i128_compare_scalar_filter(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    scalar: i128,
    scalar_on_left: bool,
    comparison: u32,
    n: u64,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    validate_window(
        resident.metadata().allocated_bytes,
        byte_offset,
        n,
        std::mem::size_of::<i128>() as u64,
    )?;
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let compare_fn =
        primary.cached_function(c"gpu_db_resident_i128_compare_scalar_to_mask", &ptx)?;
    let mask = primary.lease_device_buffer(mask_bytes)?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = byte_offset;
    let mut a2 = scalar as u64; // low 64 mantissa bits
    let mut a3 = (scalar >> 64) as u64; // high 64 mantissa bits (arithmetic shift keeps the sign)
    let mut a4 = u32::from(scalar_on_left);
    let mut a5 = comparison;
    let mut a6 = n;
    let mut a7 = mask.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u32).cast::<c_void>(),
        (&mut a5 as *mut u32).cast::<c_void>(),
        (&mut a6 as *mut u64).cast::<c_void>(),
        (&mut a7 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            compare_fn,
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
    compact_mask_i32_to_indices(resident, mask.ptr, n)
}

/// Evaluate `text[i] == needle` (or `<>` when `negate`) over a resident TEXT column to surviving row
/// indices (the type matrix, doc 19): copy the needle bytes H2D into a leased buffer, run
/// `gpu_db_resident_text_eq_scalar_to_mask` to a mask, then the shared compactor. The needle host
/// slice outlives the stream's covering sync, so the async H2D source stays valid.
pub(super) fn launch_cuda_resident_text_eq_scalar_filter(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    needle: &[u8],
    negate: bool,
    n: u64,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    let text_bytes_limit = validate_text_windows(
        resident.metadata().allocated_bytes,
        offsets_byte_offset,
        bytes_byte_offset,
        bytes_len,
        n,
    )?;
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

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
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let compare_fn = primary.cached_function(c"gpu_db_resident_text_eq_scalar_to_mask", &ptx)?;

    // Needle on device (lease >= 1 byte so the pointer is valid even for the empty string, which the
    // kernel never dereferences since needle_len == 0).
    let needle_lease = primary.lease_device_buffer(needle.len().max(1))?;
    let mask = primary.lease_device_buffer(mask_bytes)?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = offsets_byte_offset;
    let mut a2 = bytes_byte_offset;
    let mut a3 = text_bytes_limit;
    let mut a4 = needle_lease.ptr;
    let mut a5 = needle.len() as u64;
    let mut a6 = u32::from(negate);
    let mut a7 = n;
    let mut a8 = mask.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
        (&mut a6 as *mut u32).cast::<c_void>(),
        (&mut a7 as *mut u64).cast::<c_void>(),
        (&mut a8 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        if !needle.is_empty() {
            let rc = unsafe {
                htod_async(
                    needle_lease.ptr,
                    needle.as_ptr().cast::<c_void>(),
                    needle.len(),
                    stream,
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        unsafe {
            cu_launch_kernel(
                compare_fn,
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
    compact_mask_i32_to_indices(resident, mask.ptr, n)
}

/// Evaluate `text[i] <cmp> needle` over a resident TEXT column to surviving row indices (the type
/// matrix, doc 19): copy the needle bytes H2D, run `gpu_db_resident_text_compare_scalar_to_mask`
/// (LEXICOGRAPHIC unsigned byte compare, shorter-sorts-first — matches the host `compare_sql_values`
/// Text order), AND any nullable-column validity mask, then compact. `cmp`: 0=eq/1=lt/2=le/3=gt/4=ge/
/// 5=ne; `scalar_on_left` reverses the operand order. `validity_offsets` holds the text column's
/// validity bitmap offset when nullable (empty otherwise; a NULL operand is excluded — UNKNOWN).
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_cuda_resident_text_compare_scalar_filter(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    needle: &[u8],
    scalar_on_left: bool,
    comparison: u32,
    n: u64,
    validity_offsets: &[u64],
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    let text_bytes_limit = validate_text_windows(
        resident.metadata().allocated_bytes,
        offsets_byte_offset,
        bytes_byte_offset,
        bytes_len,
        n,
    )?;
    validate_bitmap_windows(resident.metadata().allocated_bytes, validity_offsets, n)?;
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

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
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let compare_fn =
        primary.cached_function(c"gpu_db_resident_text_compare_scalar_to_mask", &ptx)?;

    let needle_lease = primary.lease_device_buffer(needle.len().max(1))?;
    let mask = primary.lease_device_buffer(mask_bytes)?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = offsets_byte_offset;
    let mut a2 = bytes_byte_offset;
    let mut a3 = text_bytes_limit;
    let mut a4 = needle_lease.ptr;
    let mut a5 = needle.len() as u64;
    let mut a6 = u32::from(scalar_on_left);
    let mut a7 = comparison;
    let mut a8 = n;
    let mut a9 = mask.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
        (&mut a6 as *mut u32).cast::<c_void>(),
        (&mut a7 as *mut u32).cast::<c_void>(),
        (&mut a8 as *mut u64).cast::<c_void>(),
        (&mut a9 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        if !needle.is_empty() {
            let rc = unsafe {
                htod_async(
                    needle_lease.ptr,
                    needle.as_ptr().cast::<c_void>(),
                    needle.len(),
                    stream,
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        unsafe {
            cu_launch_kernel(
                compare_fn,
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
    compact_mask_with_validity(resident, mask.ptr, validity_offsets, n)
}

/// AND each nullable operand's NULL validity mask (1 = valid) into the i32 comparison `mask` ON THE GPU,
/// then compact to surviving indices. For M3 (doc 21) WHERE 3VL over a nullable column whose compare
/// kernel writes a plain mask but has no VM step (uuid memcmp): a NULL operand has validity bit 0, so the
/// AND clears its mask bit and the row is excluded (UNKNOWN ⇒ not selected). `validity_offsets` holds the
/// byte offset of each NULLABLE operand's validity bitmap (the caller omits non-nullable operands); EMPTY
/// ⇒ just compact, byte-identical to the no-NULL path. The compare that wrote `mask` ran under a
/// `launch_on_pooled_stream` whose covering sync completed it, so this separate stream is ordered after
/// it; the bitmap→mask + mask-AND kernels are the same the predicate VM uses (NO new/changed kernel).
fn compact_mask_with_validity(
    resident: &CudaResidentDeviceMemory,
    mask_ptr: u64,
    validity_offsets: &[u64],
    n: u64,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    if validity_offsets.is_empty() || n == 0 {
        return compact_mask_i32_to_indices(resident, mask_ptr, n);
    }
    validate_bitmap_windows(resident.metadata().allocated_bytes, validity_offsets, n)?;
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    // `gpu_db_resident_bool_to_mask` reads the validity bitmap (1=valid) -> an i32 0/1 mask;
    // `gpu_db_mask_binary` op 0 = AND, elementwise (in-place out==lhs is safe: each thread reads lhs[i]
    // before writing out[i], no cross-index dependency).
    let bool_mask_fn = primary.cached_function(c"gpu_db_resident_bool_to_mask", &ptx)?;
    let mask_binary_fn = primary.cached_function(c"gpu_db_mask_binary", &ptx)?;
    let validity_mask = primary.lease_device_buffer(mask_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let base = resident.device_ptr();
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        for &off in validity_offsets {
            // validity_mask = bitmap[off] expanded (negate=false: valid->1, NULL->0)
            let mut b0 = base;
            let mut b1 = off;
            let mut b2 = 0u32;
            let mut b3 = n;
            let mut b4 = validity_mask.ptr;
            let mut bargs = [
                (&mut b0 as *mut u64).cast::<c_void>(),
                (&mut b1 as *mut u64).cast::<c_void>(),
                (&mut b2 as *mut u32).cast::<c_void>(),
                (&mut b3 as *mut u64).cast::<c_void>(),
                (&mut b4 as *mut u64).cast::<c_void>(),
            ];
            let rc = unsafe {
                cu_launch_kernel(
                    bool_mask_fn,
                    grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    stream,
                    bargs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                return rc;
            }
            // mask = mask AND validity_mask (in-place)
            let mut m0 = mask_ptr;
            let mut m1 = validity_mask.ptr;
            let mut m2 = 0u32; // op 0 = AND
            let mut m3 = n;
            let mut m4 = mask_ptr;
            let mut margs = [
                (&mut m0 as *mut u64).cast::<c_void>(),
                (&mut m1 as *mut u64).cast::<c_void>(),
                (&mut m2 as *mut u32).cast::<c_void>(),
                (&mut m3 as *mut u64).cast::<c_void>(),
                (&mut m4 as *mut u64).cast::<c_void>(),
            ];
            let rc = unsafe {
                cu_launch_kernel(
                    mask_binary_fn,
                    grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    stream,
                    margs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        0
    })?;
    compact_mask_i32_to_indices(resident, mask_ptr, n)
}

/// Evaluate `uuid[i] <cmp> needle` over a resident UUID column (16 raw bytes/row) to surviving row
/// indices (the type matrix, doc 19): copy the 16 needle bytes H2D, run the byte-wise compare kernel
/// to a mask, then the shared nullable-mask compactor.
pub(super) fn launch_cuda_resident_uuid_compare_scalar_filter(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    needle: &[u8; 16],
    scalar_on_left: bool,
    comparison: u32,
    n: u64,
    validity_offsets: &[u64],
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    validate_window(resident.metadata().allocated_bytes, byte_offset, n, 16)?;
    validate_bitmap_windows(resident.metadata().allocated_bytes, validity_offsets, n)?;
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

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
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let compare_fn =
        primary.cached_function(c"gpu_db_resident_uuid_compare_scalar_to_mask", &ptx)?;

    let needle_lease = primary.lease_device_buffer(16)?;
    let mask = primary.lease_device_buffer(mask_bytes)?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = byte_offset;
    let mut a2 = needle_lease.ptr;
    let mut a3 = u32::from(scalar_on_left);
    let mut a4 = comparison;
    let mut a5 = n;
    let mut a6 = mask.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u32).cast::<c_void>(),
        (&mut a4 as *mut u32).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
        (&mut a6 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                needle_lease.ptr,
                needle.as_ptr().cast::<c_void>(),
                16,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        unsafe {
            cu_launch_kernel(
                compare_fn,
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
    compact_mask_with_validity(resident, mask.ptr, validity_offsets, n)
}

/// Evaluate `a <cmp> b` over two resident UUID columns (16 raw bytes/row each) to surviving row
/// indices (the type matrix, doc 19): the byte-wise compare kernel to a mask, then the compactor.
pub(super) fn launch_cuda_resident_uuid_compare_columns_filter(
    resident: &CudaResidentDeviceMemory,
    a_byte_offset: u64,
    b_byte_offset: u64,
    comparison: u32,
    n: u64,
    validity_offsets: &[u64],
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    for offset in [a_byte_offset, b_byte_offset] {
        validate_window(resident.metadata().allocated_bytes, offset, n, 16)?;
    }
    validate_bitmap_windows(resident.metadata().allocated_bytes, validity_offsets, n)?;
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let compare_fn =
        primary.cached_function(c"gpu_db_resident_uuid_compare_columns_to_mask", &ptx)?;
    let mask = primary.lease_device_buffer(mask_bytes)?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = a_byte_offset;
    let mut a2 = b_byte_offset;
    let mut a3 = comparison;
    let mut a4 = n;
    let mut a5 = mask.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u32).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            compare_fn,
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
    compact_mask_with_validity(resident, mask.ptr, validity_offsets, n)
}

/// Expand a resident bool column's 1-bit-per-row bitmap to surviving row indices (the type matrix,
/// doc 19): run the bitmap->i32-mask kernel (bit i -> row i, XOR `negate`), then the shared compactor.
pub(super) fn launch_cuda_resident_bool_to_mask_filter(
    resident: &CudaResidentDeviceMemory,
    bitmap_byte_offset: u64,
    negate: bool,
    n: u64,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    validate_bitmap_windows(
        resident.metadata().allocated_bytes,
        &[bitmap_byte_offset],
        n,
    )?;
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_resident_bool_to_mask", &ptx)?;
    let mask = primary.lease_device_buffer(mask_bytes)?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = bitmap_byte_offset;
    let mut a2 = u32::from(negate);
    let mut a3 = n;
    let mut a4 = mask.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u32).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
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
    })?;
    compact_mask_i32_to_indices(resident, mask.ptr, n)
}

/// Evaluate `text[i] LIKE pattern` over a resident TEXT column to surviving row indices (the type
/// matrix, doc 19): copy the compiled u32 token array H2D into a leased buffer, run
/// `gpu_db_resident_text_like_scalar_to_mask` to a mask, then the shared compactor. The `tokens` host
/// slice outlives the stream's covering sync, so the async H2D source stays valid.
pub(super) fn launch_cuda_resident_text_like_scalar_filter(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    tokens: &[u32],
    n: u64,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    let text_bytes_limit = validate_text_windows(
        resident.metadata().allocated_bytes,
        offsets_byte_offset,
        bytes_byte_offset,
        bytes_len,
        n,
    )?;
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let token_bytes = tokens
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(tokens.len()))?;

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
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let match_fn = primary.cached_function(c"gpu_db_resident_text_like_scalar_to_mask", &ptx)?;

    // Token array on device (lease >= 1 byte so the pointer is valid even for the empty pattern, which
    // the kernel never dereferences since ntok == 0).
    let tokens_lease = primary.lease_device_buffer(token_bytes.max(1))?;
    let mask = primary.lease_device_buffer(mask_bytes)?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = offsets_byte_offset;
    let mut a2 = bytes_byte_offset;
    let mut a3 = text_bytes_limit;
    let mut a4 = tokens_lease.ptr;
    let mut a5 = tokens.len() as u64;
    let mut a6 = n;
    let mut a7 = mask.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
        (&mut a6 as *mut u64).cast::<c_void>(),
        (&mut a7 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        if !tokens.is_empty() {
            let rc = unsafe {
                htod_async(
                    tokens_lease.ptr,
                    tokens.as_ptr().cast::<c_void>(),
                    token_bytes,
                    stream,
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        unsafe {
            cu_launch_kernel(
                match_fn,
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
    compact_mask_i32_to_indices(resident, mask.ptr, n)
}

/// Evaluate `a <cmp> b` over two resident numeric (i128) columns to surviving row indices (the type
/// matrix, doc 19): `gpu_db_resident_i128_compare_columns_to_mask` to a mask, then the compactor.
pub(super) fn launch_cuda_resident_i128_compare_columns_filter(
    resident: &CudaResidentDeviceMemory,
    a_byte_offset: u64,
    b_byte_offset: u64,
    comparison: u32,
    n: u64,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    for offset in [a_byte_offset, b_byte_offset] {
        validate_window(
            resident.metadata().allocated_bytes,
            offset,
            n,
            std::mem::size_of::<i128>() as u64,
        )?;
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let mask_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let compare_fn =
        primary.cached_function(c"gpu_db_resident_i128_compare_columns_to_mask", &ptx)?;
    let mask = primary.lease_device_buffer(mask_bytes)?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = a_byte_offset;
    let mut a2 = b_byte_offset;
    let mut a3 = comparison;
    let mut a4 = n;
    let mut a5 = mask.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u32).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            compare_fn,
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
    compact_mask_i32_to_indices(resident, mask.ptr, n)
}

#[cfg(test)]
mod tests {
    use super::{validate_bitmap_windows, validate_text_windows, validate_window};

    #[test]
    fn typed_filter_window_validation_is_checked_and_boundary_exact() {
        validate_window(64, 32, 2, 16).unwrap();
        assert!(validate_window(63, 32, 2, 16).is_err());
        assert!(validate_window(u64::MAX, u64::MAX - 1, 1, 8).is_err());
        assert!(validate_window(u64::MAX, 0, u64::MAX, 16).is_err());
    }

    #[test]
    fn typed_filter_bitmap_validation_covers_every_nullable_operand() {
        validate_bitmap_windows(24, &[0, 16], 33).unwrap();
        assert!(validate_bitmap_windows(23, &[0, 16], 33).is_err());
        assert!(validate_bitmap_windows(u64::MAX, &[u64::MAX - 1], 33).is_err());
    }

    #[test]
    fn typed_filter_text_validation_bounds_offsets_and_byte_base() {
        assert_eq!(validate_text_windows(80, 0, 40, 20, 4).unwrap(), 20);
        assert!(validate_text_windows(39, 0, 20, 1, 4).is_err());
        assert!(validate_text_windows(80, 0, 81, 0, 4).is_err());
        assert!(validate_text_windows(80, 0, 70, 11, 4).is_err());
        assert!(validate_text_windows(u64::MAX, 0, 0, 0, u64::MAX).is_err());
    }
}
