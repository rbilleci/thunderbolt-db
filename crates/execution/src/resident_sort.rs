use std::ffi::c_void;

use super::{
    check_cuda, launch_on_pooled_stream, CudaResidentDeviceMemory, CudaRuntimeProbeError,
    GpuPrimaryContext, PooledBufferLease, PooledStream,
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

/// The radix arm of `order_by_sort_i64` (n >= the measured radix crossover): upload the host `keys`
/// to a device buffer with a SYNCHRONOUS HtoD (so the radix reads valid keys regardless of which
/// stream it runs on), then reuse the proven resident LSD-radix argsort. The lease `keys_dev` MUST
/// outlive the radix call -- the explicit `drop` AFTER it stops NLL from returning the buffer to the
/// pool (where the radix could re-lease it as scratch) while the radix still reads it. The host-key
/// H2D and returned-permutation D2H remain RETIRE-003 result-path debt.
pub(super) fn launch_cuda_order_by_sort_i64_radix(
    resident: &CudaResidentDeviceMemory,
    keys: &[i64],
    descending: bool,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    let n = keys.len();
    // Validate the u32 permutation domain and exact byte size before touching CUDA. In particular,
    // an oversized host slice must not select a context or attempt a multi-gigabyte allocation first.
    let (n_u64, keys_bytes) = validate_i64_argsort_host_len(n)?;
    if n <= 1 {
        return Ok((0..n as u32).collect());
    }
    let primary = resident.primary();
    primary.set_current()?;
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
    let result = launch_cuda_resident_i64_argsort_radix(resident, &keys_dev, n_u64, descending);
    drop(keys_dev);
    result
}

/// Validate a host-slice length for the u32-indexed radix contract without touching CUDA.
pub(super) fn validate_i64_argsort_host_len(
    n: usize,
) -> Result<(u64, usize), CudaRuntimeProbeError> {
    let n_u64 =
        u64::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if n_u64 > u64::from(u32::MAX) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(n));
    }
    let required = n
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    Ok((n_u64, required))
}

/// Validate the typed production radix input before CUDA setup. The permutation stores u32 row
/// indices, and the four kernels read exactly the aligned i64 window `[keys.ptr, keys.ptr + n*8)`.
pub(super) fn validate_i64_argsort_input(
    expected_primary: usize,
    input_primary: usize,
    input_ptr: u64,
    input_capacity: usize,
    n: u64,
) -> Result<usize, CudaRuntimeProbeError> {
    if n > u64::from(u32::MAX) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(n).unwrap_or(usize::MAX),
        ));
    }
    let n_usize =
        usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let (_, required) = validate_i64_argsort_host_len(n_usize)?;
    if input_primary != expected_primary
        || !input_ptr.is_multiple_of(std::mem::align_of::<i64>() as u64)
        || required > input_capacity
        || input_ptr.checked_add(required as u64).is_none()
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(input_capacity));
    }
    Ok(n_usize)
}

/// Parallel GPU LSD-radix argsort over a resident i64 key column -- the production large-result
/// ORDER BY arm. Returns a Vec<u32> permutation of 0..n ordering the rows by key (ascending when
/// `descending=false`, descending when true), STABLE (equal keys keep ascending original index) in
/// both directions. O(n), constant 16 LSD passes of 4 bits -- beats the bitonic arm's O(n log^2 n)
/// launch count at large n.
///
/// Keys are mapped signed->unsigned-order by XOR with a direction mask (0x8000…0 ascending so i64
/// order == u64 order; 0x7FFF…F descending = the complement, so one ascending radix yields
/// descending keys with the SAME ascending-index tie-break). Each pass is three kernels on one
/// pooled stream: (1) `radix_histogram` — each block counts its contiguous chunk's 4-bit digits
/// into a bucket-major block_hist[d*G+b] via a shared per-digit histogram; (2) `radix_scan` —
/// exclusive prefix sum over the whole 16*G matrix, so block_hist[d*G+b] becomes the global output
/// offset where block b's digit-d run begins; (3) `radix_scatter` — each block STABLY scatters its
/// chunk to keys_dst/idx_dst at base + per-digit-running + within-block rank, the within-block rank
/// computed per grid-stride wave by `match.any.sync` warp ranking + a cross-warp per-digit combine,
/// with a per-digit running offset carried across waves. Ping-pong (keys/idx a<->b) over 16 passes.
pub(super) fn launch_cuda_resident_i64_argsort_radix(
    resident: &CudaResidentDeviceMemory,
    keys: &PooledBufferLease<'_>,
    n: u64,
    descending: bool,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
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

    const PTX: &[u8] = include_bytes!("resident_argsort.ptx");

    let n_usize = validate_i64_argsort_input(
        std::ptr::from_ref(resident.primary()).addr(),
        keys.primary_identity(),
        keys.ptr,
        keys.capacity,
        n,
    )?;
    if n == 0 {
        return Ok(Vec::new());
    }

    // Contiguous partition: block `b` owns rows [b*chunk, min(b*chunk+chunk, n)). `chunk` is sized
    // so the block count G stays within the grid-x max (65_535) for any n.
    const BLOCK: u32 = 256;
    // The scatter kernel's shared arrays s_wh[128]/s_wb[128] are sized for nwarps = BLOCK/32 = 8
    // (128 = 16 digits * 8 warps), and the scan kernel's s[1024] + its 1024-thread launch assume
    // that block. Raising BLOCK without resizing those hard-coded PTX shared arrays would corrupt
    // shared memory, so pin it at compile time.
    const _: () = assert!(
        BLOCK == 256,
        "radix PTX shared-memory sizes are hard-coded for BLOCK=256"
    );
    const CHUNK_ROWS: u64 = 2_048;
    const MAX_GRID: u64 = 65_535;
    let chunk = CHUNK_ROWS.max(n.div_ceil(MAX_GRID));
    let grid_g_u64 = n.div_ceil(chunk);
    let grid_g = u32::try_from(grid_g_u64)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let hist_len = grid_g_u64
        .checked_mul(16)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let hist_len_usize = usize::try_from(hist_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

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
    let init_fn = resident
        .primary()
        .cached_function(c"gpu_db_radix_init", &ptx)?;
    let hist_fn = resident
        .primary()
        .cached_function(c"gpu_db_radix_histogram", &ptx)?;
    let scan_fn = resident
        .primary()
        .cached_function(c"gpu_db_radix_scan", &ptx)?;
    let scatter_fn = resident
        .primary()
        .cached_function(c"gpu_db_radix_scatter", &ptx)?;

    let primary = resident.primary();
    let bytes8 = n_usize
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes4 = n_usize
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let hist_bytes = hist_len_usize
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let keys_a = primary.lease_device_buffer(bytes8)?;
    let keys_b = primary.lease_device_buffer(bytes8)?;
    let idx_a = primary.lease_device_buffer(bytes4)?;
    let idx_b = primary.lease_device_buffer(bytes4)?;
    let block_hist = primary.lease_device_buffer(hist_bytes)?;

    primary.set_current()?;
    struct StreamLease<'a> {
        primary: &'a GpuPrimaryContext,
        pooled: Option<PooledStream>,
    }
    impl Drop for StreamLease<'_> {
        fn drop(&mut self) {
            if let Some(pooled) = self.pooled.take() {
                self.primary.release_pooled_stream(pooled);
            }
        }
    }
    let lease = StreamLease {
        primary,
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let stream = lease
        .pooled
        .as_ref()
        .expect("pooled stream just set")
        .stream;
    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    let src_keys_base = keys.ptr;
    let mask: u64 = if descending {
        0x7FFF_FFFF_FFFF_FFFF
    } else {
        0x8000_0000_0000_0000
    };

    // init: transform source keys into keys_a, idx_a = identity.
    {
        let mut src_arg = src_keys_base;
        let mut n_arg = n;
        let mut mask_arg = mask;
        let mut ka_arg = keys_a.ptr;
        let mut ia_arg = idx_a.ptr;
        let mut init_args = [
            (&mut src_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
            (&mut mask_arg as *mut u64).cast::<c_void>(),
            (&mut ka_arg as *mut u64).cast::<c_void>(),
            (&mut ia_arg as *mut u64).cast::<c_void>(),
        ];
        let init_grid = n.div_ceil(u64::from(BLOCK)).clamp(1, MAX_GRID) as u32;
        check_cuda(unsafe {
            cu_launch_kernel(
                init_fn,
                init_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                init_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;
    }

    // 16 LSD passes of 4 bits, ping-ponging (keys/idx) a<->b.
    let mut keys_src = keys_a.ptr;
    let mut keys_dst = keys_b.ptr;
    let mut idx_src = idx_a.ptr;
    let mut idx_dst = idx_b.ptr;
    for pass in 0u32..16 {
        let mut shift_arg = pass * 4;
        let mut n_arg = n;
        let mut chunk_arg = chunk;
        let mut g_arg = grid_g_u64;
        let mut bh_arg = block_hist.ptr;
        let mut total_arg = hist_len;

        // histogram (reads keys_src)
        let mut ksrc_arg = keys_src;
        let mut hist_args = [
            (&mut ksrc_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
            (&mut shift_arg as *mut u32).cast::<c_void>(),
            (&mut chunk_arg as *mut u64).cast::<c_void>(),
            (&mut g_arg as *mut u64).cast::<c_void>(),
            (&mut bh_arg as *mut u64).cast::<c_void>(),
        ];
        check_cuda(unsafe {
            cu_launch_kernel(
                hist_fn,
                grid_g,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                hist_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;

        // scan (single block, exclusive prefix over the 16*G matrix)
        let mut scan_args = [
            (&mut bh_arg as *mut u64).cast::<c_void>(),
            (&mut total_arg as *mut u64).cast::<c_void>(),
        ];
        check_cuda(unsafe {
            cu_launch_kernel(
                scan_fn,
                1,
                1,
                1,
                1_024,
                1,
                1,
                0,
                stream,
                scan_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;

        // scatter (reads keys_src/idx_src, writes keys_dst/idx_dst)
        let mut isrc_arg = idx_src;
        let mut kdst_arg = keys_dst;
        let mut idst_arg = idx_dst;
        let mut scatter_args = [
            (&mut ksrc_arg as *mut u64).cast::<c_void>(),
            (&mut isrc_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
            (&mut shift_arg as *mut u32).cast::<c_void>(),
            (&mut chunk_arg as *mut u64).cast::<c_void>(),
            (&mut g_arg as *mut u64).cast::<c_void>(),
            (&mut bh_arg as *mut u64).cast::<c_void>(),
            (&mut kdst_arg as *mut u64).cast::<c_void>(),
            (&mut idst_arg as *mut u64).cast::<c_void>(),
        ];
        check_cuda(unsafe {
            cu_launch_kernel(
                scatter_fn,
                grid_g,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                scatter_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;

        std::mem::swap(&mut keys_src, &mut keys_dst);
        std::mem::swap(&mut idx_src, &mut idx_dst);
    }

    check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;
    resident.record_kernel_event_elapsed_us(None);

    // After 16 (even) passes the final result is in the buffer `idx_src` now points at.
    let mut indices = vec![0_u32; n_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            indices.as_mut_ptr().cast::<c_void>(),
            idx_src,
            n_usize * std::mem::size_of::<u32>(),
        )
    })
    .map_err(drain_err)?;
    drop(lease);
    Ok(indices)
}

const RADIX_SORT_CROSSOVER_ROWS: u64 = 10_000;

/// Production int4/int8-compatible ORDER BY dispatcher. Small host-key batches retain the existing
/// bitonic route; large batches use the typed, stable LSD-radix owner above.
pub(super) fn launch_cuda_order_by_sort_i64(
    resident: &CudaResidentDeviceMemory,
    keys: &[i64],
    descending: bool,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    if keys.len() as u64 >= RADIX_SORT_CROSSOVER_ROWS {
        launch_cuda_order_by_sort_i64_radix(resident, keys, descending)
    } else {
        launch_cuda_bitonic_sort_i64(resident, keys, descending)
    }
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
