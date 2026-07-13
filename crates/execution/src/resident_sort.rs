use std::ffi::c_void;

use super::{
    check_cuda, launch_cuda_resident_i64_argsort_radix, launch_on_pooled_stream,
    CudaResidentDeviceMemory, CudaRuntimeProbeError,
};

/// GPU bitonic sort of `keys` -> the row positions (0..n) in ascending (or `descending`) key order.
/// Pads to a power of two; padding positions sort last and are dropped from the result. O(log^2 n)
/// compare-exchange passes, each a `gpu_db_bitonic_sort_i64_step` launch on a pooled stream. The
/// reusable core of the charter-native GPU ORDER BY (later slices add multi-key / typed / expr keys).
pub(super) fn launch_cuda_bitonic_sort_i64(
    resident: &CudaResidentDeviceMemory,
    keys: &[i64],
    descending: bool,
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
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("resident_sort.ptx");

    let n = keys.len();
    if n <= 1 {
        return Ok((0..n as u32).collect());
    }
    let npot = n.next_power_of_two();
    let perm_bytes = npot
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(npot))?;
    let keys_bytes = n
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    let n_u64 = n as u64;
    let npot_u64 = npot as u64;
    let desc_u64 = u64::from(descending);

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
    let sort_fn = primary.cached_function(c"gpu_db_bitonic_sort_i64_step", &ptx)?;
    let perm_dev = primary.lease_device_buffer(perm_bytes)?;
    let keys_dev = primary.lease_device_buffer(keys_bytes)?;
    let identity: Vec<u32> = (0..npot as u32).collect();

    const BLOCK: u32 = 256;
    let grid = (npot.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);

    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                perm_dev.ptr,
                identity.as_ptr().cast::<c_void>(),
                perm_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                keys_dev.ptr,
                keys.as_ptr().cast::<c_void>(),
                keys_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        // Bitonic network: kk = the sorted-subsequence size (doubles), jj = the compare distance
        // (halves). Each (kk, jj) is one compare-exchange pass over all npot indices on the stream.
        let mut kk = 2_u64;
        while kk <= npot_u64 {
            let mut jj = kk >> 1;
            loop {
                let mut p0 = perm_dev.ptr;
                let mut p1 = keys_dev.ptr;
                let mut p2 = n_u64;
                let mut p3 = npot_u64;
                let mut p4 = kk;
                let mut p5 = jj;
                let mut p6 = desc_u64;
                let mut args = [
                    (&mut p0 as *mut u64).cast::<c_void>(),
                    (&mut p1 as *mut u64).cast::<c_void>(),
                    (&mut p2 as *mut u64).cast::<c_void>(),
                    (&mut p3 as *mut u64).cast::<c_void>(),
                    (&mut p4 as *mut u64).cast::<c_void>(),
                    (&mut p5 as *mut u64).cast::<c_void>(),
                    (&mut p6 as *mut u64).cast::<c_void>(),
                ];
                let rc = unsafe {
                    cu_launch_kernel(
                        sort_fn,
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
                };
                if rc != 0 {
                    return rc;
                }
                if jj == 1 {
                    break;
                }
                jj >>= 1;
            }
            kk <<= 1;
        }
        0
    })?;

    let mut perm_host = vec![0_u32; npot];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            perm_host.as_mut_ptr().cast::<c_void>(),
            perm_dev.ptr,
            perm_bytes,
        )
    })?;
    Ok(perm_host
        .into_iter()
        .filter(|&p| (p as usize) < n)
        .collect())
}

/// The RADIX arm of `order_by_sort_i64` (n >= the adaptive crossover): upload the host `keys` to a
/// device buffer with a SYNCHRONOUS HtoD (so the radix reads valid keys regardless of which stream it
/// runs on), then reuse the proven resident LSD-radix argsort. The lease `keys_dev` MUST outlive the
/// radix call -- the explicit `drop` AFTER it stops NLL from returning the buffer to the pool (where the
/// radix could re-lease it as scratch) while the radix still reads it.
pub(super) fn launch_cuda_order_by_sort_i64_radix(
    resident: &CudaResidentDeviceMemory,
    keys: &[i64],
    descending: bool,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    let n = keys.len();
    if n <= 1 {
        return Ok((0..n as u32).collect());
    }
    let primary = resident.primary();
    primary.set_current()?;
    let keys_bytes = n
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    let keys_dev = primary.lease_device_buffer(keys_bytes)?;
    let cu_memcpy_htod = unsafe {
        *primary
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe {
        cu_memcpy_htod(keys_dev.ptr, keys.as_ptr().cast::<c_void>(), keys_bytes)
    })?;
    let result =
        launch_cuda_resident_i64_argsort_radix(resident, keys_dev.ptr, n as u64, descending);
    drop(keys_dev);
    result
}

/// GPU bitonic sort keyed by a resident TEXT column (lexicographic, unsigned bytes, a prefix sorts
/// smaller). `indices` are the surviving row ids (after WHERE); the comparator reads each row's text via
/// the indices indirection (`text[ indices[pos] ]`). Returns positions into `indices` in sorted text
/// order. Pads to a power of two (padding sorts last by position, dropped). The text leg of the
/// charter-native GPU ORDER BY; mirrors `launch_cuda_bitonic_sort_i64` (perm + the kk/jj network),
/// swapping the i64 key buffer for the indices buffer + the resident text section pointers.
pub(super) fn launch_cuda_bitonic_sort_text(
    resident: &CudaResidentDeviceMemory,
    indices: &[u64],
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    descending: bool,
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
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("resident_sort.ptx");

    let n = indices.len();
    if n <= 1 {
        return Ok((0..n as u32).collect());
    }
    let npot = n.next_power_of_two();
    let perm_bytes = npot
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(npot))?;
    let indices_bytes = n
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    let n_u64 = n as u64;
    let npot_u64 = npot as u64;
    let desc_u64 = u64::from(descending);
    // The kernel reads the resident text sections directly; resolve their absolute device pointers
    // here (device_ptr is the resident base, same as the text-eq/LIKE filters pass it).
    let offsets_base = resident.device_ptr().wrapping_add(offsets_byte_offset);
    let bytes_base = resident.device_ptr().wrapping_add(bytes_byte_offset);

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
    let sort_fn = primary.cached_function(c"gpu_db_bitonic_sort_text_step", &ptx)?;
    let perm_dev = primary.lease_device_buffer(perm_bytes)?;
    let indices_dev = primary.lease_device_buffer(indices_bytes)?;
    let identity: Vec<u32> = (0..npot as u32).collect();

    const BLOCK: u32 = 256;
    let grid = (npot.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);

    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                perm_dev.ptr,
                identity.as_ptr().cast::<c_void>(),
                perm_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                indices_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let mut kk = 2_u64;
        while kk <= npot_u64 {
            let mut jj = kk >> 1;
            loop {
                let mut p0 = perm_dev.ptr;
                let mut p1 = indices_dev.ptr;
                let mut p2 = offsets_base;
                let mut p3 = bytes_base;
                let mut p4 = n_u64;
                let mut p5 = npot_u64;
                let mut p6 = kk;
                let mut p7 = jj;
                let mut p8 = desc_u64;
                let mut args = [
                    (&mut p0 as *mut u64).cast::<c_void>(),
                    (&mut p1 as *mut u64).cast::<c_void>(),
                    (&mut p2 as *mut u64).cast::<c_void>(),
                    (&mut p3 as *mut u64).cast::<c_void>(),
                    (&mut p4 as *mut u64).cast::<c_void>(),
                    (&mut p5 as *mut u64).cast::<c_void>(),
                    (&mut p6 as *mut u64).cast::<c_void>(),
                    (&mut p7 as *mut u64).cast::<c_void>(),
                    (&mut p8 as *mut u64).cast::<c_void>(),
                ];
                let rc = unsafe {
                    cu_launch_kernel(
                        sort_fn,
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
                };
                if rc != 0 {
                    return rc;
                }
                if jj == 1 {
                    break;
                }
                jj >>= 1;
            }
            kk <<= 1;
        }
        0
    })?;

    let mut perm_host = vec![0_u32; npot];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            perm_host.as_mut_ptr().cast::<c_void>(),
            perm_dev.ptr,
            perm_bytes,
        )
    })?;
    Ok(perm_host
        .into_iter()
        .filter(|&p| (p as usize) < n)
        .collect())
}

/// GPU multi-key bitonic sort. `keys` is a row-major `n x k` matrix of i64 (`keys[row*k + key]`, key 0
/// most significant); `desc_mask` bit j set => key j sorts descending. Returns the row positions (0..n)
/// in the multi-key order. Pads to a power of two (padding sorts last, dropped from the result), driving
/// the same O(log^2 n) kk/jj network as the single-key path but with a `gpu_db_bitonic_sort_multikey_step`
/// comparator that walks the K keys. The general multi-key core of the charter-native GPU ORDER BY.
pub(super) fn launch_cuda_bitonic_sort_multikey(
    resident: &CudaResidentDeviceMemory,
    keys: &[i64],
    n: usize,
    k: usize,
    desc_mask: u64,
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
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("resident_sort.ptx");

    let expected = n
        .checked_mul(k)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    if keys.len() != expected {
        return Err(CudaRuntimeProbeError::InvalidInputLength(keys.len()));
    }
    if n <= 1 {
        return Ok((0..n as u32).collect());
    }
    let npot = n.next_power_of_two();
    let perm_bytes = npot
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(npot))?;
    let keys_bytes = keys
        .len()
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(keys.len()))?;
    let n_u64 = n as u64;
    let npot_u64 = npot as u64;
    let k_u64 = k as u64;

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
    let sort_fn = primary.cached_function(c"gpu_db_bitonic_sort_multikey_step", &ptx)?;
    let perm_dev = primary.lease_device_buffer(perm_bytes)?;
    let keys_dev = primary.lease_device_buffer(keys_bytes)?;
    let identity: Vec<u32> = (0..npot as u32).collect();

    const BLOCK: u32 = 256;
    let grid = (npot.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);

    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                perm_dev.ptr,
                identity.as_ptr().cast::<c_void>(),
                perm_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                keys_dev.ptr,
                keys.as_ptr().cast::<c_void>(),
                keys_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        // Same bitonic network as the single-key path: kk = the sorted-subsequence size (doubles), jj =
        // the compare distance (halves). The kernel takes kdim (= K) and desc_mask as extra params.
        let mut kk = 2_u64;
        while kk <= npot_u64 {
            let mut jj = kk >> 1;
            loop {
                let mut p0 = perm_dev.ptr;
                let mut p1 = keys_dev.ptr;
                let mut p2 = n_u64;
                let mut p3 = npot_u64;
                let mut p4 = k_u64;
                let mut p5 = kk;
                let mut p6 = jj;
                let mut p7 = desc_mask;
                let mut args = [
                    (&mut p0 as *mut u64).cast::<c_void>(),
                    (&mut p1 as *mut u64).cast::<c_void>(),
                    (&mut p2 as *mut u64).cast::<c_void>(),
                    (&mut p3 as *mut u64).cast::<c_void>(),
                    (&mut p4 as *mut u64).cast::<c_void>(),
                    (&mut p5 as *mut u64).cast::<c_void>(),
                    (&mut p6 as *mut u64).cast::<c_void>(),
                    (&mut p7 as *mut u64).cast::<c_void>(),
                ];
                let rc = unsafe {
                    cu_launch_kernel(
                        sort_fn,
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
                };
                if rc != 0 {
                    return rc;
                }
                if jj == 1 {
                    break;
                }
                jj >>= 1;
            }
            kk <<= 1;
        }
        0
    })?;

    let mut perm_host = vec![0_u32; npot];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            perm_host.as_mut_ptr().cast::<c_void>(),
            perm_dev.ptr,
            perm_bytes,
        )
    })?;
    Ok(perm_host
        .into_iter()
        .filter(|&p| (p as usize) < n)
        .collect())
}

/// GPU HETEROGENEOUS multi-key bitonic sort — a key tuple mixing INT and TEXT keys (`ORDER BY name
/// /*text*/, age /*int*/, id /*int*/`). `int_keys` is a row-major `n x num_int` i64 matrix BY POSITION
/// (only the int keys); `text_cols[t]` = (offsets_byte_offset, bytes_byte_offset) of text key t's
/// resident column; `key_plan[k]` selects key k's source (bit31=is_text, bits0..30=idx into the int
/// columns or the text_cols). `desc_mask` bit k => key k DESC. Returns the row positions (0..n) in the
/// tuple order. The comparator walks the keys; the first non-equal decides; ties fall through.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_cuda_bitonic_sort_hetero(
    resident: &CudaResidentDeviceMemory,
    indices: &[u64],
    int_keys: &[i64],
    num_int: usize,
    text_cols: &[(u64, u64)],
    b128_cols: &[u64],
    key_plan: &[u32],
    desc_mask: u64,
    // M3 (doc 21): one NULL validity bitmap byte offset per key (u64::MAX = the key holds no NULL). The
    // comparator reads it ON-DEVICE. Empty ⇒ treated as all-sentinel (no NULL handling). Offsets are into
    // `resident_base`.
    null_offs: &[u64],
    // M3 (doc 21): per-key effective "NULLS FIRST" bitmask (bit k = key k places NULLs first). The engine
    // sets it to the explicit NULLS FIRST/LAST override, or the key's DESC bit (PG default: NULLS LAST ASC,
    // NULLS FIRST DESC) when unspecified, so the NULL placement is direction-resolved in the comparator.
    nulls_first_mask: u64,
    // When Some, the text/numeric/uuid legs read from THIS uploaded payload (a resident-LIKE buffer
    // built from a non-resident result, e.g. a grouped result) instead of `resident`'s own columns;
    // `resident` is then used only for the CUDA context/stream. text_cols/b128_cols offsets are into it.
    resident_base_payload: Option<&[u8]>,
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
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("resident_sort.ptx");

    let n = indices.len();
    let expected_int = n
        .checked_mul(num_int)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    if int_keys.len() != expected_int {
        return Err(CudaRuntimeProbeError::InvalidInputLength(int_keys.len()));
    }
    if n <= 1 {
        return Ok((0..n as u32).collect());
    }
    let num_text = text_cols.len();
    let num_keys = key_plan.len();
    let npot = n.next_power_of_two();
    let perm_bytes = npot
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(npot))?;
    let indices_bytes = n
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    // int_keys / text_cols / key_plan may be empty along one axis (all-text or all-int tuple); lease a
    // non-zero floor so the device buffer is valid even when the kernel never reads it.
    let int_keys_bytes = std::mem::size_of_val(int_keys).max(8);
    let text_offs: Vec<u64> = text_cols.iter().map(|&(o, _)| o).collect();
    let text_bytes_vec: Vec<u64> = text_cols.iter().map(|&(_, b)| b).collect();
    let text_meta_bytes = (num_text * std::mem::size_of::<u64>()).max(8);
    let plan_bytes = std::mem::size_of_val(key_plan).max(4);
    let num_b128 = b128_cols.len();
    let b128_meta_bytes = std::mem::size_of_val(b128_cols).max(8);
    // M3 (doc 21): one validity offset per key (sentinel u64::MAX when no NULL handling). Normalize so a
    // caller that passes none gets all-sentinel (byte-identical to the no-NULL behavior).
    let null_offs_vec: Vec<u64> = if null_offs.len() == num_keys {
        null_offs.to_vec()
    } else {
        vec![u64::MAX; num_keys]
    };
    let null_offs_bytes = std::mem::size_of_val(null_offs_vec.as_slice()).max(8);
    let n_u64 = n as u64;
    let npot_u64 = npot as u64;
    let num_int_u64 = num_int as u64;
    let num_keys_u64 = num_keys as u64;

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
    let sort_fn = primary.cached_function(c"gpu_db_bitonic_sort_hetero_step", &ptx)?;
    let perm_dev = primary.lease_device_buffer(perm_bytes)?;
    let indices_dev = primary.lease_device_buffer(indices_bytes)?;
    let int_keys_dev = primary.lease_device_buffer(int_keys_bytes)?;
    let text_offs_dev = primary.lease_device_buffer(text_meta_bytes)?;
    let text_bytes_dev = primary.lease_device_buffer(text_meta_bytes)?;
    let plan_dev = primary.lease_device_buffer(plan_bytes)?;
    let b128_offs_dev = primary.lease_device_buffer(b128_meta_bytes)?;
    let null_offs_dev = primary.lease_device_buffer(null_offs_bytes)?;
    // The text/numeric/uuid legs read `resident_base`: default to `resident`'s own columns, or a leased
    // copy of `resident_base_payload` (a grouped result's resident-like buffer) when provided.
    let payload_dev = match resident_base_payload {
        Some(p) => Some(primary.lease_device_buffer(p.len().max(8))?),
        None => None,
    };
    let resident_base = payload_dev
        .as_ref()
        .map_or_else(|| resident.device_ptr(), |d| d.ptr);
    let identity: Vec<u32> = (0..npot as u32).collect();

    const BLOCK: u32 = 256;
    let grid = (npot.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);

    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        // Upload the grouped result's resident-like payload FIRST (the kernel reads it as resident_base).
        if let (Some(p), Some(pd)) = (resident_base_payload, payload_dev.as_ref()) {
            let rc = unsafe { htod_async(pd.ptr, p.as_ptr().cast::<c_void>(), p.len(), stream) };
            if rc != 0 {
                return rc;
            }
        }
        let rc = unsafe {
            htod_async(
                perm_dev.ptr,
                identity.as_ptr().cast::<c_void>(),
                perm_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                indices_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        if !int_keys.is_empty() {
            let rc = unsafe {
                htod_async(
                    int_keys_dev.ptr,
                    int_keys.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(int_keys),
                    stream,
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        if num_text != 0 {
            let rc = unsafe {
                htod_async(
                    text_offs_dev.ptr,
                    text_offs.as_ptr().cast::<c_void>(),
                    num_text * std::mem::size_of::<u64>(),
                    stream,
                )
            };
            if rc != 0 {
                return rc;
            }
            let rc = unsafe {
                htod_async(
                    text_bytes_dev.ptr,
                    text_bytes_vec.as_ptr().cast::<c_void>(),
                    num_text * std::mem::size_of::<u64>(),
                    stream,
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        if num_b128 != 0 {
            let rc = unsafe {
                htod_async(
                    b128_offs_dev.ptr,
                    b128_cols.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(b128_cols),
                    stream,
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        let rc = unsafe {
            htod_async(
                plan_dev.ptr,
                key_plan.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(key_plan),
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                null_offs_dev.ptr,
                null_offs_vec.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(null_offs_vec.as_slice()),
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let mut kk = 2_u64;
        while kk <= npot_u64 {
            let mut jj = kk >> 1;
            loop {
                let mut p0 = perm_dev.ptr;
                let mut p1 = indices_dev.ptr;
                let mut p2 = int_keys_dev.ptr;
                let mut p3 = num_int_u64;
                let mut p4 = text_offs_dev.ptr;
                let mut p5 = text_bytes_dev.ptr;
                let mut p6 = resident_base;
                let mut p7 = plan_dev.ptr;
                let mut p8 = num_keys_u64;
                let mut p9 = desc_mask;
                let mut p10 = n_u64;
                let mut p11 = npot_u64;
                let mut p12 = kk;
                let mut p13 = jj;
                let mut p14 = b128_offs_dev.ptr;
                let mut p15 = null_offs_dev.ptr;
                let mut p16 = nulls_first_mask;
                let mut args = [
                    (&mut p0 as *mut u64).cast::<c_void>(),
                    (&mut p1 as *mut u64).cast::<c_void>(),
                    (&mut p2 as *mut u64).cast::<c_void>(),
                    (&mut p3 as *mut u64).cast::<c_void>(),
                    (&mut p4 as *mut u64).cast::<c_void>(),
                    (&mut p5 as *mut u64).cast::<c_void>(),
                    (&mut p6 as *mut u64).cast::<c_void>(),
                    (&mut p7 as *mut u64).cast::<c_void>(),
                    (&mut p8 as *mut u64).cast::<c_void>(),
                    (&mut p9 as *mut u64).cast::<c_void>(),
                    (&mut p10 as *mut u64).cast::<c_void>(),
                    (&mut p11 as *mut u64).cast::<c_void>(),
                    (&mut p12 as *mut u64).cast::<c_void>(),
                    (&mut p13 as *mut u64).cast::<c_void>(),
                    (&mut p14 as *mut u64).cast::<c_void>(),
                    (&mut p15 as *mut u64).cast::<c_void>(),
                    (&mut p16 as *mut u64).cast::<c_void>(),
                ];
                let rc = unsafe {
                    cu_launch_kernel(
                        sort_fn,
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
                };
                if rc != 0 {
                    return rc;
                }
                if jj == 1 {
                    break;
                }
                jj >>= 1;
            }
            kk <<= 1;
        }
        0
    })?;

    let mut perm_host = vec![0_u32; npot];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            perm_host.as_mut_ptr().cast::<c_void>(),
            perm_dev.ptr,
            perm_bytes,
        )
    })?;
    Ok(perm_host
        .into_iter()
        .filter(|&p| (p as usize) < n)
        .collect())
}
