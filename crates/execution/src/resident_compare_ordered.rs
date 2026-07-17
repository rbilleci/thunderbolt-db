use std::os::raw::c_void;

use super::resident_window;
use super::{
    check_cuda, copy_pinned_into, launch_on_pooled_stream, stage_result_dtoh_async,
    CudaI32Comparison, CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError,
    GpuPrimaryContext, PooledBufferLease, PooledDeviceBufferOwned, PooledStream,
};

/// Ordered parallel-compaction PTX shared by the VALUE-emit launch
/// (`launch_cuda_resident_i32_compare_project`) and the INDEX-emit launch
/// (`launch_cuda_resident_i32_compare_indices_ordered`). One module, two `.entry`
/// kernels (`..._count_blocks` Pass A, `..._scatter_blocks` Pass B). The scatter kernel's
/// trailing `out_is_index` param selects what it stores at each ascending output slot: the
/// matching i32 VALUE (mode 0) or the surviving ROW INDEX as u32 (mode != 0). Both kernels
/// support comparison codes 0=eq, 1=lt, 2=lte, 3=gt, 4=gte, 5=ne (eq/ne are folded into
/// `p_match` IDENTICALLY in count and scatter, so Pass A's per-block count equals Pass B's
/// per-block scatter count for every predicate).
pub(super) const COMPARE_ORDERED_PTX: &[u8] = include_bytes!("resident_compare_ordered.ptx");

/// VALUE-emit launch: returns the matching i32 VALUES in ASCENDING ROW ORDER (the ordered
/// parallel-compaction backbone, `COMPARE_ORDERED_PTX`). A thin wrapper over the shared core with
/// `out_is_index = 0`; behavior is unchanged from before the index-emit mode was added.
pub(super) fn launch_cuda_resident_i32_compare_project<R: CudaResidentReadSource>(
    resident: &R,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    comparison: CudaI32Comparison,
) -> Result<Vec<i32>, CudaRuntimeProbeError> {
    launch_cuda_resident_i32_compare_ordered_core(
        resident,
        OrderedI32InputWindow::resident(resident, byte_offset),
        row_count,
        needle,
        comparison.code(),
        0,
    )
}

pub(super) fn i32_bits_into_u32(values: Vec<i32>) -> Vec<u32> {
    // SAFETY: i32 and u32 have identical size/alignment and every 32-bit pattern is valid for both.
    // ManuallyDrop transfers the allocation exactly once; the reconstructed Vec retains the same
    // pointer, length, and capacity, so the index path does not allocate/copy millions of result slots.
    let mut values = std::mem::ManuallyDrop::new(values);
    let ptr = values.as_mut_ptr().cast::<u32>();
    let len = values.len();
    let capacity = values.capacity();
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

#[derive(Clone, Copy)]
struct OrderedI32InputWindow {
    device_base: u64,
    allocated_bytes: u64,
    byte_offset: u64,
}

impl OrderedI32InputWindow {
    fn resident<R: CudaResidentReadSource>(resident: &R, byte_offset: u64) -> Self {
        Self {
            device_base: resident.device_ptr(),
            allocated_bytes: resident.metadata().allocated_bytes,
            byte_offset,
        }
    }

    fn pooled(input: &PooledBufferLease<'_>) -> Self {
        Self {
            device_base: input.ptr,
            allocated_bytes: input.capacity as u64,
            byte_offset: 0,
        }
    }

    fn pooled_owned(input: &PooledDeviceBufferOwned) -> Self {
        Self {
            device_base: input.ptr,
            allocated_bytes: input.capacity as u64,
            byte_offset: 0,
        }
    }
}

pub(super) fn validate_ordered_i32_comparison(
    comparison: u32,
    max_code: u32,
) -> Result<(), CudaRuntimeProbeError> {
    if comparison > max_code {
        return Err(CudaRuntimeProbeError::UnsupportedComparison(comparison));
    }
    Ok(())
}

pub(super) fn validate_ordered_i32_index_domain(
    row_count: u64,
) -> Result<(), CudaRuntimeProbeError> {
    if row_count > u64::from(u32::MAX) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(row_count).unwrap_or(usize::MAX),
        ));
    }
    Ok(())
}

pub(super) fn validate_ordered_i32_context_identity(
    expected: usize,
    actual: usize,
    capacity: usize,
) -> Result<(), CudaRuntimeProbeError> {
    if actual != expected {
        return Err(CudaRuntimeProbeError::InvalidInputLength(capacity));
    }
    Ok(())
}

pub(super) fn validate_ordered_i32_input_window(
    allocated_bytes: u64,
    byte_offset: u64,
    row_count: u64,
) -> Result<(), CudaRuntimeProbeError> {
    let width = std::mem::size_of::<i32>() as u64;
    resident_window::validate_aligned_window(
        allocated_bytes,
        byte_offset,
        row_count,
        width,
        std::mem::align_of::<i32>() as u64,
    )
}

/// INDEX-emit launch: returns the surviving ROW INDICES (`Vec<u32>`) in ASCENDING ORDER via the SAME
/// ordered parallel compaction, with the scatter kernel storing each match's row index (u32) instead
/// of its value (`out_is_index = 1`). The ascending-by-construction guarantee is identical to the
/// value path — the scatter slot is the same row's rank either way, only the stored payload differs —
/// so this replaces the atomic-append + host-`sort_unstable` index path with no host sort. Takes the
/// raw comparison code (0=eq, 1=lt, 2=lte, 3=gt, 4=gte, 5=ne).
pub(super) fn launch_cuda_resident_i32_compare_indices_ordered<R: CudaResidentReadSource>(
    resident: &R,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    validate_ordered_i32_comparison(comparison, 5)?;
    validate_ordered_i32_index_domain(row_count)?;
    let slots = launch_cuda_resident_i32_compare_ordered_core(
        resident,
        OrderedI32InputWindow::resident(resident, byte_offset),
        row_count,
        needle,
        comparison,
        1,
    )?;
    // The scatter kernel wrote each surviving row index as a u32 via `st.global.b32`; the host buffer
    // is `Vec<i32>` 4-byte slots, so reinterpret each slot's bits back to u32 (bit-exact — a row index
    // is `< row_count`, always non-negative, and fits u32 since row_count <= u32::MAX in every sized
    // grid). The order is already ascending by construction (no host sort).
    Ok(i32_bits_into_u32(slots))
}

pub(super) fn launch_cuda_buffer_i32_compare_indices_ordered(
    resident: &CudaResidentDeviceMemory,
    input: &PooledBufferLease<'_>,
    row_count: u64,
    needle: i32,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    validate_ordered_i32_comparison(comparison, 5)?;
    validate_ordered_i32_index_domain(row_count)?;
    validate_ordered_i32_context_identity(
        std::ptr::from_ref(resident.primary()).addr(),
        input.primary_identity(),
        input.capacity,
    )?;
    let slots = launch_cuda_resident_i32_compare_ordered_core(
        resident,
        OrderedI32InputWindow::pooled(input),
        row_count,
        needle,
        comparison,
        1,
    )?;
    Ok(i32_bits_into_u32(slots))
}

pub(super) fn launch_cuda_owned_i32_compare_indices_ordered(
    resident: &CudaResidentDeviceMemory,
    input: &PooledDeviceBufferOwned,
    row_count: u64,
    needle: i32,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    validate_ordered_i32_comparison(comparison, 5)?;
    validate_ordered_i32_index_domain(row_count)?;
    validate_ordered_i32_context_identity(
        std::ptr::from_ref(resident.primary()).addr(),
        std::sync::Arc::as_ptr(&input.primary).addr(),
        input.capacity,
    )?;
    let slots = launch_cuda_resident_i32_compare_ordered_core(
        resident,
        OrderedI32InputWindow::pooled_owned(input),
        row_count,
        needle,
        comparison,
        1,
    )?;
    Ok(i32_bits_into_u32(slots))
}

/// TWO-INPUT (col-vs-col / expr-vs-expr) ordered compare-compaction: compare `lhs[i] <cmp> rhs[i]`
/// elementwise and return the surviving ROW INDICES (`Vec<u32>`) in ASCENDING ORDER, with NO host
/// sort. Mirrors `launch_cuda_resident_i32_compare_indices_ordered` (and shares the orchestration of
/// `launch_cuda_resident_i32_compare_ordered_core`: parallel per-block count -> tiny host
/// exclusive-scan of the per-block counts -> parallel ordered-scatter, the same chunk/grid sizing,
/// lease lifetimes, async-on-pooled-stream lever + blocking fallback, and the `output_count >
/// row_count` guard), but drives the TWO-INPUT kernels `gpu_db_buffers_i32_compare_count_blocks` /
/// `..._scatter_blocks` with two absolute device bases and NO needle/byte_offset. The scatter always
/// stores the surviving row index (col-vs-col is a predicate). `comparison` is the raw code
/// (0=eq,1=lt,2=lte,3=gt,4=gte; ne is rejected at the wrapper).
///
/// Both typed leases must belong to `resident` and cover `[0, n*4)`: the wrapper validates those
/// exact capacities before CUDA setup. Their borrows span both count and scatter launches, so both
/// inputs outlive every device read.
pub(super) fn launch_cuda_resident_i32_compare_buffers_indices_ordered(
    resident: &CudaResidentDeviceMemory,
    lhs: &PooledBufferLease<'_>,
    rhs: &PooledBufferLease<'_>,
    n: u64,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
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

    const PTX: &[u8] = COMPARE_ORDERED_PTX;

    validate_ordered_i32_comparison(comparison, 4)?;
    validate_ordered_i32_index_domain(n)?;
    let primary_identity = std::ptr::from_ref(resident.primary()).addr();
    validate_ordered_i32_context_identity(primary_identity, lhs.primary_identity(), lhs.capacity)?;
    validate_ordered_i32_context_identity(primary_identity, rhs.primary_identity(), rhs.capacity)?;
    validate_ordered_i32_input_window(lhs.capacity as u64, 0, n)?;
    validate_ordered_i32_input_window(rhs.capacity as u64, 0, n)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    // CUDA current context is thread-local. Bind before module-cache lookup, buffer leasing, or
    // any launch so off-lock DML predicates are safe on arbitrary writer threads.
    resident.primary().set_current()?;
    let lhs_base = lhs.ptr;
    let rhs_base = rhs.ptr;

    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let values_bytes = usize::try_from(
        n.checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    // ---- ordered-compaction grid shape (identical to the single-input core) ----
    const BLOCK: u32 = 256;
    const CHUNK_ROWS: u64 = 256;
    const MAX_GRID: u64 = 65_535;
    let chunk = CHUNK_ROWS.max(n.div_ceil(MAX_GRID));
    let grid_u64 = n.div_ceil(chunk);
    debug_assert!((1..=MAX_GRID).contains(&grid_u64));
    let grid = u32::try_from(grid_u64)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let block_counts_len = grid as usize;
    let block_scratch_bytes = block_counts_len
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let async_ops = match (
        resident.primary().cu_memcpy_dtoh_async,
        resident.primary().cu_memcpy_htod_async,
        resident.primary().cu_memset_d8_async,
    ) {
        (Some(dtoh), Some(htod), Some(memset)) => Some((dtoh, htod, memset)),
        _ => None,
    };

    let values_guard = resident
        .primary()
        .lease_device_buffer(values_bytes.max(1))?;
    let block_counts_guard = resident
        .primary()
        .lease_device_buffer(block_scratch_bytes)?;
    let block_base_guard = resident
        .primary()
        .lease_device_buffer(block_scratch_bytes)?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let count_fn = resident
        .primary()
        .cached_function(c"gpu_db_buffers_i32_compare_count_blocks", &ptx)?;
    let scatter_fn = resident
        .primary()
        .cached_function(c"gpu_db_buffers_i32_compare_scatter_blocks", &ptx)?;

    // Shared kernel scalar args. The two-input kernels read lhs[idx] / rhs[idx] at `base + idx*4` (no
    // needle / byte_offset). The scatter ALWAYS stores the row index (col-vs-col is a predicate).
    let mut lhs_arg = lhs_base;
    let mut rhs_arg = rhs_base;
    let mut rows_arg = n;
    let mut chunk_arg = chunk;
    let mut comparison_arg = comparison;
    let mut block_counts_arg = block_counts_guard.ptr;
    let mut block_base_arg = block_base_guard.ptr;
    let mut values_arg = values_guard.ptr;
    let mut count_args = [
        (&mut lhs_arg as *mut u64).cast::<c_void>(),
        (&mut rhs_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut chunk_arg as *mut u64).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut block_counts_arg as *mut u64).cast::<c_void>(),
    ];
    let mut scatter_args = [
        (&mut lhs_arg as *mut u64).cast::<c_void>(),
        (&mut rhs_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut chunk_arg as *mut u64).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut block_base_arg as *mut u64).cast::<c_void>(),
        (&mut values_arg as *mut u64).cast::<c_void>(),
    ];

    // Host exclusive scan of the per-block match counts into per-block base output slots; returns
    // (block_base, total_matches). Identical to the single-input core.
    fn exclusive_scan_blocks(counts: &[u64]) -> (Vec<u64>, u64) {
        let mut base = Vec::with_capacity(counts.len());
        let mut running = 0_u64;
        for &c in counts {
            base.push(running);
            running = running.saturating_add(c);
        }
        (base, running)
    }

    if let Some((dtoh_async, htod_async, memset_async)) = async_ops {
        // ---- async-on-pooled-stream path ----
        resident.primary().set_current()?;
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
            primary: resident.primary(),
            pooled: Some(resident.primary().acquire_pooled_stream()?),
        };
        let pooled = lease.pooled.as_ref().expect("pooled stream just set");
        let stream = pooled.stream;
        let timed = !pooled.start_event.is_null() && !pooled.stop_event.is_null();

        // HARDENING (error-path stream drain): drain the private stream before yielding an error so
        // enqueued ops cannot use leases freed by unwinding. Identical to the single-input core.
        let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
            unsafe {
                let _ = (resident.primary().cu_stream_synchronize)(stream);
            }
            err
        };

        if timed {
            check_cuda(unsafe { (resident.primary().cu_event_record)(pooled.start_event, stream) })
                .map_err(drain_err)?;
        }
        check_cuda(unsafe { memset_async(block_counts_guard.ptr, 0, block_scratch_bytes, stream) })
            .map_err(drain_err)?;
        check_cuda(unsafe {
            cu_launch_kernel(
                count_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                count_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;

        let mut block_counts = vec![0_u64; block_counts_len];
        let counts_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            block_counts_guard.ptr,
            &mut block_counts,
        )
        .map_err(drain_err)?;
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        copy_pinned_into(&counts_pinned, &mut block_counts);
        drop(counts_pinned);

        let (block_base, output_count) = exclusive_scan_blocks(&block_counts);
        if output_count > n {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(output_count).unwrap_or(usize::MAX),
            ));
        }

        let base_pinned = resident
            .primary()
            .lease_pinned_host_buffer(block_scratch_bytes);
        if let Some(pinned) = &base_pinned {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    block_base.as_ptr(),
                    pinned.ptr.cast::<u64>(),
                    block_base.len(),
                );
            }
        }
        let base_src: *const c_void = base_pinned
            .as_ref()
            .map(|p| p.ptr.cast_const())
            .unwrap_or_else(|| block_base.as_ptr().cast::<c_void>());
        check_cuda(unsafe {
            htod_async(block_base_guard.ptr, base_src, block_scratch_bytes, stream)
        })
        .map_err(drain_err)?;
        check_cuda(unsafe {
            cu_launch_kernel(
                scatter_fn,
                grid,
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
        if timed {
            check_cuda(unsafe { (resident.primary().cu_event_record)(pooled.stop_event, stream) })
                .map_err(drain_err)?;
        }

        let mut output = vec![
            0_i32;
            usize::try_from(output_count).map_err(|_| {
                CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
            })?
        ];

        let values_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            values_guard.ptr,
            &mut output,
        )
        .map_err(drain_err)?;
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        drop(base_pinned);
        copy_pinned_into(&values_pinned, &mut output);

        if timed {
            let mut elapsed_ms = 0.0_f32;
            check_cuda(unsafe {
                (resident.primary().cu_event_elapsed_time)(
                    &mut elapsed_ms,
                    pooled.start_event,
                    pooled.stop_event,
                )
            })?;
            resident.record_kernel_event_elapsed_us(Some(
                (f64::from(elapsed_ms) * 1_000.0).ceil() as u64
            ));
        } else {
            resident.record_kernel_event_elapsed_us(None);
        }

        drop(lease);
        // The scatter wrote each surviving row index as a u32 via `st.global.b32`; the host buffer is
        // `Vec<i32>` 4-byte slots, so reinterpret each slot's bits back to u32 (bit-exact - a row index
        // is `< n`, non-negative, fits u32 since n <= u32::MAX). Already ascending by construction.
        Ok(output.into_iter().map(|slot| slot as u32).collect())
    } else {
        // ---- legacy blocking fallback (old driver: no async/pinned symbols) ----
        check_cuda(unsafe { cu_memset_d8(block_counts_guard.ptr, 0, block_scratch_bytes) })?;
        launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
            cu_launch_kernel(
                count_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                count_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;

        let mut block_counts = vec![0_u64; block_counts_len];
        if !block_counts.is_empty() {
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    block_counts.as_mut_ptr().cast::<c_void>(),
                    block_counts_guard.ptr,
                    block_scratch_bytes,
                )
            })?;
        }
        let (block_base, output_count) = exclusive_scan_blocks(&block_counts);
        if output_count > n {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(output_count).unwrap_or(usize::MAX),
            ));
        }

        check_cuda(unsafe {
            cu_memcpy_htod(
                block_base_guard.ptr,
                block_base.as_ptr().cast::<c_void>(),
                block_scratch_bytes,
            )
        })?;
        launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
            cu_launch_kernel(
                scatter_fn,
                grid,
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
        })?;

        let mut output = vec![
            0_i32;
            usize::try_from(output_count).map_err(|_| {
                CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
            })?
        ];
        if !output.is_empty() {
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    output.as_mut_ptr().cast::<c_void>(),
                    values_guard.ptr,
                    output.len() * std::mem::size_of::<i32>(),
                )
            })?;
        }
        Ok(output.into_iter().map(|slot| slot as u32).collect())
    }
}

/// Shared core for the ordered i32 compare-compaction (`COMPARE_ORDERED_PTX`). `out_is_index` selects
/// what each ascending output slot stores: the matching i32 VALUE (0) or the surviving ROW INDEX as a
/// u32 (1, returned as `i32` bits the caller reinterprets). `comparison` is the raw kernel code
/// (0=eq, 1=lt, 2=lte, 3=gt, 4=gte, 5=ne). All count/host-scan/scatter structure, chunk/grid sizing,
/// and lease lifetimes are shared, so the value and index paths are byte-identical except the payload.
///
/// `input` keeps the device base, exact owner capacity, and byte offset together. Kernels read
/// `row_count` contiguous i32s from `input.device_base + input.byte_offset + idx*4`; the complete
/// aligned window is checked against the extent before CUDA setup, so both passes are total.
fn launch_cuda_resident_i32_compare_ordered_core<R: CudaResidentReadSource>(
    resident: &R,
    input: OrderedI32InputWindow,
    row_count: u64,
    needle: i32,
    comparison: u32,
    out_is_index: u32,
) -> Result<Vec<i32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
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

    // P2-M2 (compare-route parallel-kernel lever): the legacy kernel was a single-thread
    // `<<<1,1,1>>>` ascending scan that ordered-appended every matching i32 VALUE, so its output
    // is the matches in ASCENDING ROW ORDER. That serialized the whole 50k-row scan into one thread
    // (~277 µs kernel; the route plateaued serial-kernel-bound). This route now runs an ORDERED
    // PARALLEL COMPACTION over a CONTIGUOUS block partition — byte-identical ascending output, no
    // atomic-append (which would yield non-deterministic atomic-SCHEDULE order, the hazard fixed in
    // `row_indices`). The partition is the ordering backbone: block `b` owns the contiguous row
    // range `[b*chunk, min(b*chunk+chunk, rows))`, so "block order" == "row order".
    //
    //   Pass A (`..._count_blocks`, parallel: G blocks x BLOCK threads): each block grid-strides
    //   its own range and `red.global.add`s its local match count into `block_counts[b]`. This is
    //   the parallel scan of all rows (analogous to `equal_count`'s parallel reduction).
    //
    //   Host (between passes): exclusive-scan `block_counts[0..G]` -> `block_base[b]` = number of
    //   matches in all blocks `< b` (the base output slot for block `b`); the total match count is
    //   `block_base[G-1] + block_counts[G-1]`. Tiny (G is the block count), so a host scan avoids a
    //   third device scan kernel + a hand-authored shared-memory prefix sum.
    //
    //   Pass B (`..._scatter_blocks`, G blocks x BLOCK threads): block `b` compacts ITS range in
    //   PARALLEL via an ORDERED intra-block prefix-sum — each thread tests its row(s) to a 0/1 flag,
    //   an intra-block EXCLUSIVE scan (warp `shfl` scan + a tiny shared cross-warp combine) gives
    //   each match its within-block rank, and the value is scattered at `block_base[b] + rank`
    //   (+ a per-iteration running base when `chunk > blockDim`). The scan is monotonic in row index,
    //   so the within-block order is ASCENDING BY CONSTRUCTION (no atomics in the ordering path), and
    //   disjoint `block_base` ranges keep blocks independent — so the global output is exactly the
    //   ascending per-row matches, byte-identical to the old serial kernel for ANY row_count and
    //   match pattern. The serial span drops from one `chunk` (the prior one-thread-per-block scan)
    //   to `ceil(chunk / blockDim)` ordered-scan steps.
    //
    // Both kernels loop over `[start, end)` (block grid-stride), so they are correct for any
    // `chunk`/grid; the host sizes `chunk` so `G = ceil(rows/chunk) <= 65535` for every row_count
    // (the CUDA grid-x max), rounding `chunk` up when rows would exceed `65535 * CHUNK_ROWS`.
    const PTX: &[u8] = COMPARE_ORDERED_PTX;

    validate_ordered_i32_comparison(comparison, 5)?;
    if out_is_index > 1 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            out_is_index as usize,
        ));
    }
    if out_is_index == 1 {
        validate_ordered_i32_index_domain(row_count)?;
    }
    validate_ordered_i32_input_window(input.allocated_bytes, input.byte_offset, row_count)?;
    if row_count == 0 {
        return Ok(Vec::new());
    }
    // CUDA current context is thread-local. Bind before module-cache lookup, buffer leasing, or
    // any launch so off-lock DML predicates are safe on arbitrary writer threads.
    resident.primary().set_current()?;
    let input_base = input.device_base;
    let byte_offset = input.byte_offset;

    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    // Blocking memset/HtoD for the fallback path (zero the block-count scratch; upload the
    // host-scanned per-block base offsets).
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let values_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    // ---- ordered-compaction grid shape ----
    // Contiguous partition: block `b` owns rows `[b*chunk, min(b*chunk+chunk, rows))`. `chunk` is
    // sized so the block count `G = ceil(rows/chunk)` stays within the CUDA grid-x max (65_535) for
    // ANY row_count: start at CHUNK_ROWS rows/block and round `chunk` up if rows would need more
    // than 65_535 blocks. Both kernels loop over their range, so any `chunk` is correct.
    const BLOCK: u32 = 256;
    const CHUNK_ROWS: u64 = 256;
    const MAX_GRID: u64 = 65_535;
    let chunk = CHUNK_ROWS.max(row_count.div_ceil(MAX_GRID));
    let grid_u64 = row_count.div_ceil(chunk);
    debug_assert!((1..=MAX_GRID).contains(&grid_u64));
    let grid = u32::try_from(grid_u64)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let block_counts_len = grid as usize;
    let block_scratch_bytes = block_counts_len
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    // P2-M2 (compare-route parallel-kernel lever): this single-frame range/comparison projection
    // already ran on the pooled-async substrate (private stream, pinned D2H, two covering syncs, no
    // per-call alloc/JIT/whole-context sync), but its kernel was the legacy single-thread
    // `<<<1,1,1>>>` scan, so the route plateaued serial-kernel-bound (~277 µs for a 50k-row scan).
    // The kernel is now an ORDERED PARALLEL COMPACTION (two passes over a contiguous block
    // partition; see the PTX header): a parallel per-block match count, a tiny host exclusive scan
    // of those counts into per-block base offsets, then a parallel scatter that compacts each block's
    // matches ascending at its base via an ordered intra-block prefix-sum (warp `shfl` scan + a tiny
    // shared cross-warp combine). The output is byte-identical ascending-per-row values for any
    // row_count and match pattern (no atomic-append, so no atomic-schedule non-determinism). The
    // device scratch is the `block_counts` / `block_base` arrays (G u64 each), leased from the same
    // `OutputBufferPool` as the values buffer.
    //
    // The whole async path is gated on the optional async + pinned-host driver symbols; on an old
    // driver lacking them the route keeps a blocking path (still cached-module + pooled-buffer +
    // pooled-stream, just blocking memset/HtoD/D2H), so correctness is unconditional and only the
    // acceleration is best-effort.
    let async_ops = match (
        resident.primary().cu_memcpy_dtoh_async,
        resident.primary().cu_memcpy_htod_async,
        resident.primary().cu_memset_d8_async,
    ) {
        (Some(dtoh), Some(htod), Some(memset)) => Some((dtoh, htod, memset)),
        _ => None,
    };

    // Pooled device buffers (no per-call cuMemAlloc/cuMemFree): the values output plus the two
    // block-offset scratch arrays. `block_counts` is zeroed before pass A (the count kernel
    // red-adds into it); `block_base` is overwritten by the HtoD upload; `values` is read back only
    // over the [0, total) prefix, so pooled stale bytes are never observed.
    let values_guard = resident
        .primary()
        .lease_device_buffer(values_bytes.max(1))?;
    let block_counts_guard = resident
        .primary()
        .lease_device_buffer(block_scratch_bytes)?;
    let block_base_guard = resident
        .primary()
        .lease_device_buffer(block_scratch_bytes)?;

    // P2-M2: cached modules (no per-launch cuModuleLoadData) — both kernel entries live in one PTX
    // module; the cache is keyed per entry name and launched concurrently on distinct streams.
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let count_fn = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_compare_count_blocks", &ptx)?;
    let scatter_fn = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_compare_scatter_blocks", &ptx)?;

    // Shared kernel scalar args (pointers/needle/comparison/chunk are identical across both passes;
    // each pass binds its own output pointer). The kernels read the input at `input_base + byte_offset
    // + idx*4`; `input_base` is the resident column ptr for the column path and a leased buffer ptr for
    // a generalized caller (mask / arithmetic-result compaction).
    let mut resident_arg = input_base;
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut chunk_arg = chunk;
    let mut needle_arg = needle;
    let mut comparison_arg = comparison;
    let mut block_counts_arg = block_counts_guard.ptr;
    let mut block_base_arg = block_base_guard.ptr;
    let mut values_arg = values_guard.ptr;
    // out_is_index selects the scatter payload: 0 = matching i32 VALUE, 1 = surviving ROW INDEX (u32).
    // Both produce the SAME ascending output slots; only the stored bytes differ.
    let mut out_is_index_arg: u32 = out_is_index;
    let mut count_args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut chunk_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut block_counts_arg as *mut u64).cast::<c_void>(),
    ];
    let mut scatter_args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut chunk_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut block_base_arg as *mut u64).cast::<c_void>(),
        (&mut values_arg as *mut u64).cast::<c_void>(),
        (&mut out_is_index_arg as *mut u32).cast::<c_void>(),
    ];

    // Host exclusive scan of the per-block match counts into per-block base output slots; returns
    // (block_base, total_matches). The base of block 0 is 0, and total is the sum of all counts —
    // both kernels see the SAME contiguous partition, so prefix-by-block == prefix-by-row.
    fn exclusive_scan_blocks(counts: &[u64]) -> (Vec<u64>, u64) {
        let mut base = Vec::with_capacity(counts.len());
        let mut running = 0_u64;
        for &c in counts {
            base.push(running);
            running = running.saturating_add(c);
        }
        (base, running)
    }

    if let Some((dtoh_async, htod_async, memset_async)) = async_ops {
        // ---- async-on-pooled-stream path (the lever) ----
        // Bind the shared primary context (idempotent) and lease the pooled private stream.
        resident.primary().set_current()?;
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
            primary: resident.primary(),
            pooled: Some(resident.primary().acquire_pooled_stream()?),
        };
        let pooled = lease.pooled.as_ref().expect("pooled stream just set");
        let stream = pooled.stream;
        let timed = !pooled.start_event.is_null() && !pooled.stop_event.is_null();

        // HARDENING (error-path stream drain): once an async op is enqueued on this private
        // stream, an early `?` would unwind the locals — returning the device + pinned buffer
        // leases to their shared pools while enqueued ops may still be in flight, a
        // use-after-free window for whoever leases those buffers next. So every fallible op from
        // the first async enqueue through each covering `cuStreamSynchronize` propagates its error
        // through `drain_err`, a best-effort blocking sync (result ignored) that drains the stream
        // FIRST, *then* yields the original error. `map_err` runs the closure at the error site
        // BEFORE `?` returns and hence before ANY local Drop, so it precedes the release of every
        // lease regardless of scope; on `Ok` the closure is not invoked, so the success path adds
        // nothing beyond the two explicit syncs below.
        let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
            // SAFETY: `stream` is the live pooled private stream; a blocking synchronize on it is
            // valid from this thread (the primary context is current). The result is intentionally
            // ignored — this is a best-effort drain on an already-failing path.
            unsafe {
                let _ = (resident.primary().cu_stream_synchronize)(stream);
            }
            err
        };

        // (1) Stream-ordered zero of the block-count scratch, then the parallel COUNT kernel
        // (pass A). Start the timer before pass A so the recorded span covers BOTH kernels — the
        // meaningful "new kernel time" vs the old serial kernel.
        if timed {
            check_cuda(unsafe { (resident.primary().cu_event_record)(pooled.start_event, stream) })
                .map_err(drain_err)?;
        }
        check_cuda(unsafe { memset_async(block_counts_guard.ptr, 0, block_scratch_bytes, stream) })
            .map_err(drain_err)?;
        check_cuda(unsafe {
            cu_launch_kernel(
                count_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                count_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;

        // (2) Stream-ordered read of the per-block counts, then sync #1: the count kernel is now
        // complete so the host can exclusive-scan the counts into base offsets and size the values
        // read. The counts stage through a pooled pinned host buffer for a truly-async DMA.
        let mut block_counts = vec![0_u64; block_counts_len];
        let counts_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            block_counts_guard.ptr,
            &mut block_counts,
        )
        .map_err(drain_err)?;
        // Covering sync #1: drains on its own error too (the counts D2H is still enqueued).
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        copy_pinned_into(&counts_pinned, &mut block_counts);
        drop(counts_pinned);

        let (block_base, output_count) = exclusive_scan_blocks(&block_counts);
        if output_count > row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(output_count).unwrap_or(usize::MAX),
            ));
        }

        // (3) Upload the host-scanned base offsets (HtoD, staged through a pooled pinned buffer for
        // a truly-async DMA), then launch the parallel SCATTER kernel (pass B: BLOCK threads/block,
        // ordered intra-block compaction — each match scattered ascending at base+rank), then stop
        // the timer.
        let base_pinned = resident
            .primary()
            .lease_pinned_host_buffer(block_scratch_bytes);
        if let Some(pinned) = &base_pinned {
            // SAFETY: leased with capacity >= block_scratch_bytes; copy the base offsets into the
            // pinned region (page-aligned) before the async HtoD reads from it.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    block_base.as_ptr(),
                    pinned.ptr.cast::<u64>(),
                    block_base.len(),
                );
            }
        }
        let base_src: *const c_void = base_pinned
            .as_ref()
            .map(|p| p.ptr.cast_const())
            .unwrap_or_else(|| block_base.as_ptr().cast::<c_void>());
        check_cuda(unsafe {
            htod_async(block_base_guard.ptr, base_src, block_scratch_bytes, stream)
        })
        .map_err(drain_err)?;
        check_cuda(unsafe {
            cu_launch_kernel(
                scatter_fn,
                grid,
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
        if timed {
            check_cuda(unsafe { (resident.primary().cu_event_record)(pooled.stop_event, stream) })
                .map_err(drain_err)?;
        }

        let mut output = vec![
            0_i32;
            usize::try_from(output_count).map_err(|_| {
                CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
            })?
        ];

        // (4) Stream-ordered result D2H of the populated [0, total) values prefix into a pooled
        // pinned host buffer, then ONE sync #2; copy the pinned bytes into the owned Vec. This sync
        // also covers the still-enqueued base HtoD + scatter kernel (both ordered before it).
        let values_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            values_guard.ptr,
            &mut output,
        )
        .map_err(drain_err)?;
        // Covering sync #2: drains on its own error too (the HtoD/scatter/values D2H are enqueued).
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        // Keep the base staging buffer alive until after the sync that completes its HtoD.
        drop(base_pinned);
        copy_pinned_into(&values_pinned, &mut output);

        if timed {
            let mut elapsed_ms = 0.0_f32;
            check_cuda(unsafe {
                (resident.primary().cu_event_elapsed_time)(
                    &mut elapsed_ms,
                    pooled.start_event,
                    pooled.stop_event,
                )
            })?;
            resident.record_kernel_event_elapsed_us(Some(
                (f64::from(elapsed_ms) * 1_000.0).ceil() as u64
            ));
        } else {
            resident.record_kernel_event_elapsed_us(None);
        }

        drop(lease);
        Ok(output)
    } else {
        // ---- legacy blocking fallback (old driver: no async/pinned symbols) ----
        // Still cached-module + pooled-buffer + pooled-stream (per-stream sync, no whole-context
        // cuCtxSynchronize); just blocking memset/HtoD/D2H. Run the same two passes with a blocking
        // counts D2H + host scan + base HtoD between them.
        check_cuda(unsafe { cu_memset_d8(block_counts_guard.ptr, 0, block_scratch_bytes) })?;
        launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
            cu_launch_kernel(
                count_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                count_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;

        let mut block_counts = vec![0_u64; block_counts_len];
        if !block_counts.is_empty() {
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    block_counts.as_mut_ptr().cast::<c_void>(),
                    block_counts_guard.ptr,
                    block_scratch_bytes,
                )
            })?;
        }
        let (block_base, output_count) = exclusive_scan_blocks(&block_counts);
        if output_count > row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(output_count).unwrap_or(usize::MAX),
            ));
        }

        check_cuda(unsafe {
            cu_memcpy_htod(
                block_base_guard.ptr,
                block_base.as_ptr().cast::<c_void>(),
                block_scratch_bytes,
            )
        })?;
        launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
            cu_launch_kernel(
                scatter_fn,
                grid,
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
        })?;

        let mut output = vec![
            0_i32;
            usize::try_from(output_count).map_err(|_| {
                CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
            })?
        ];
        if !output.is_empty() {
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    output.as_mut_ptr().cast::<c_void>(),
                    values_guard.ptr,
                    output.len() * std::mem::size_of::<i32>(),
                )
            })?;
        }
        Ok(output)
    }
}
