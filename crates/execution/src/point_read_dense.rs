use std::ffi::c_void;
use std::sync::Arc;

use super::resident_memory::CudaResidentDeviceAllocation;
use super::{
    check_cuda, copy_pinned_into, stage_result_dtoh_async, validate_i32_index_geometry,
    CudaI32BatchProjectionColumns, CudaResidentDeviceMemory, CudaResidentReadSource,
    CudaRuntimeProbeError, GpuPrimaryContext, I32NeedlesHostGuard, PooledDeviceBufferOwned,
    PooledStreamOwned, Probe,
};

#[cfg(test)]
std::thread_local! {
    static DENSE_PANIC_PHASE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn force_next_dense_panic(phase: u8) {
    DENSE_PANIC_PHASE.with(|pending| pending.set(phase));
}

#[cfg(test)]
fn panic_at_dense_phase(phase: u8) {
    DENSE_PANIC_PHASE.with(|pending| {
        if pending.get() == phase {
            pending.set(0);
            panic!("injected dense asynchronous ownership panic at phase {phase}");
        }
    });
}

/// Compact dense point-probe result used between the execution and engine hot paths. Device status remains
/// byte-sized through D2H and assembly: `1` means exactly one visible row was found, `2` means no visible row
/// was found, and `3` means multiple visible rows matched so the unique-route result was declined. The
/// compatibility completion converts status `3` to [`CudaRuntimeProbeError::DuplicatePointReadMatch`] rather
/// than silently treating it as not-found; otherwise it preserves the established `Vec<u32>` status contract.
#[derive(Debug)]
pub struct CudaI32DenseBatchProjection {
    values: Vec<i32>,
    projection_count: usize,
    status: Vec<u8>,
}

impl CudaI32DenseBatchProjection {
    /// Per-needle dense status: `1` found, `2` not found, `3` duplicate visible match / unique-route decline.
    pub fn status(&self) -> &[u8] {
        &self.status
    }

    pub fn projection_count(&self) -> usize {
        self.projection_count
    }

    /// Decompose into `(values, projection_count, status)`. Status values are `1` found, `2` not found, and
    /// `3` duplicate visible match / unique-route decline; compatibility completion maps `3` to
    /// [`CudaRuntimeProbeError::DuplicatePointReadMatch`].
    pub fn into_parts(self) -> (Vec<i32>, usize, Vec<u8>) {
        (self.values, self.projection_count, self.status)
    }

    fn into_compat(self) -> Result<CudaI32BatchProjectionColumns, CudaRuntimeProbeError> {
        if let Some(needle_index) = self.status.iter().position(|status| *status == 3) {
            return Err(CudaRuntimeProbeError::DuplicatePointReadMatch(needle_index));
        }
        if let Some(status) = self.status.iter().find(|status| !matches!(**status, 1 | 2)) {
            return Err(CudaRuntimeProbeError::InvalidInputLength(*status as usize));
        }
        Ok(CudaI32BatchProjectionColumns {
            values: self.values,
            needle_indices: Vec::new(),
            row_indices: Vec::new(),
            projection_count: self.projection_count,
            status: self.status.into_iter().map(u32::from).collect(),
        })
    }
}

/// DENSE-emit variant of the unique int4 index probe (DECISIONS "lpb read levers" #1). On the single-shard
/// route, thread `i` writes `values[i*proj]` plus `status[i]` (`1` found, `2` not found) to its own slot. The
/// multi-shard route uses the same submission owner and may additionally write `3` when multiple visible rows
/// match and the unique route declines. Both avoid `atom.global.add`, needle/count/row-index outputs, and host
/// random scatter. The non-unique scan keeps the atomic kernel and row indices. Gaps are guarded: status is
/// initialized to zero, every in-bounds thread writes a definitive `1`, `2`, or (multi-shard only) `3`, and
/// compatibility/engine consumers reject any remaining zero/unknown status while compact completion preserves
/// the opaque byte for production decline handling.
pub struct CudaI32IndexProbeDenseSubmission {
    projection_count: usize,
    needles_len: usize,
    primary: Arc<GpuPrimaryContext>,
    _resident_allocation_guard: Arc<CudaResidentDeviceAllocation>,
    values_guard: PooledDeviceBufferOwned,
    status_guard: PooledDeviceBufferOwned,
    _needles_guard: PooledDeviceBufferOwned,
    _needles_host_guard: I32NeedlesHostGuard,
    stream: Option<PooledStreamOwned>,
    timed: bool,
    _wave_index_guard: Option<Arc<CudaResidentDeviceMemory>>,
    /// Sub-slice 8 v2 (multi-shard kernel): pins the per-shard device indexes + the descriptor-array device
    /// buffer alive until the kernel completes (the kernel reads them). Empty/None for the single-shard path.
    _multi_shard_plan_guard: Option<Arc<CudaI32MultiShardProbePlan>>,
    /// Sub-slice 8 v3 (O(1) routing): true when the multi-shard kernel took the BINARY-SEARCH path (the shards
    /// were host-proven ascending-disjoint). Pure telemetry for the non-vacuity assert that binary mode fired;
    /// the result rows are byte-identical to the linear path.
    pub multi_shard_binary_mode: bool,
}

/// Panic guard for completion's asynchronous device-to-host destinations. The submission itself drains its
/// device buffers and stream, but function locals normally drop before the by-value `self` parameter. Declaring
/// this guard after every host destination makes its Drop run first and synchronize before a pinned lease can
/// return to the shared pool or a pageable Vec can free its backing memory.
struct InFlightHostCopyDrain {
    primary: Arc<GpuPrimaryContext>,
    stream: *mut c_void,
    armed: bool,
}

impl InFlightHostCopyDrain {
    fn new(primary: Arc<GpuPrimaryContext>, stream: *mut c_void) -> Self {
        Self {
            primary,
            stream,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for InFlightHostCopyDrain {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.primary.set_current();
            unsafe {
                let _ = (self.primary.cu_stream_synchronize)(self.stream);
            }
        }
    }
}

// The dense submission owns pooled device-buffer/stream guards (raw `*mut c_void` pointers into
// device memory plus a pooled stream). Engine ownership moves it between connection threads and
// completion re-binds the primary context; the pointers are never concurrently shared.
unsafe impl Send for CudaI32IndexProbeDenseSubmission {}

pub(super) fn submit_cuda_resident_i32_index_probe_dense<R: CudaResidentReadSource>(
    resident: &R,
    index: &Arc<CudaResidentDeviceMemory>,
    index_table_mask: u32,
    index_hash_shift: u32,
    needles: &[i32],
    projection_offsets: &[u64],
    row_count: u64,
) -> Result<CudaI32IndexProbeDenseSubmission, CudaRuntimeProbeError> {
    let index_ptr = index.device_ptr();
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
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

    const MAX_PROJECTIONS: usize = 4;
    // Byte-for-byte the same params/probe as `gpu_db_resident_i32_index_probe` EXCEPT: `out_needle_indices_ptr`
    // / `out_row_indices_ptr` / `out_count_ptr` are replaced by a single `out_status_ptr`; the FOUND emit
    // writes to slot `idx` (no atomic) + `status[idx]=1`; a new NOTFOUND block writes `status[idx]=2`. The
    // out-of-bounds / proj==0 guards still `ret` without writing (those slots are never read).
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_index_probe_dense(
    .param .u64 resident_ptr,
    .param .u64 index_ptr,
    .param .u32 table_mask,
    .param .u32 hash_shift,
    .param .u32 row_count,
    .param .u32 needle_count,
    .param .u32 projection_count,
    .param .u64 projection_offset0,
    .param .u64 projection_offset1,
    .param .u64 projection_offset2,
    .param .u64 projection_offset3,
    .param .u64 needles_ptr,
    .param .u64 out_values_ptr,
    .param .u64 out_status_ptr
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<28>;

    ld.param.u64 %rd1, [resident_ptr];
    ld.param.u64 %rd2, [index_ptr];
    ld.param.u32 %r1, [table_mask];
    ld.param.u32 %r2, [hash_shift];
    ld.param.u32 %r18, [row_count];
    ld.param.u32 %r3, [needle_count];
    ld.param.u32 %r4, [projection_count];
    ld.param.u64 %rd3, [projection_offset0];
    ld.param.u64 %rd4, [projection_offset1];
    ld.param.u64 %rd5, [projection_offset2];
    ld.param.u64 %rd6, [projection_offset3];
    ld.param.u64 %rd7, [needles_ptr];
    ld.param.u64 %rd8, [out_values_ptr];
    ld.param.u64 %rd9, [out_status_ptr];

    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mad.lo.u32 %r8, %r6, %r7, %r5;
    setp.ge.u32 %p1, %r8, %r3;
    @%p1 bra DONE;
    setp.eq.u32 %p1, %r4, 0;
    @%p1 bra DONE;

    mul.wide.u32 %rd12, %r8, 4;
    add.u64 %rd13, %rd7, %rd12;
    ld.global.s32 %r9, [%rd13];
    mul.lo.u32 %r10, %r9, 2654435761;
    shr.u32 %r11, %r10, %r2;
    mov.u32 %r12, 0;

PROBE:
    and.b32 %r11, %r11, %r1;
    mul.wide.u32 %rd14, %r11, 8;
    add.u64 %rd15, %rd2, %rd14;
    ld.global.u64 %rd16, [%rd15];
    setp.eq.u64 %p2, %rd16, 0;
    @%p2 bra NOTFOUND;
    shr.u64 %rd17, %rd16, 32;
    cvt.u32.u64 %r13, %rd17;
    setp.eq.s32 %p2, %r13, %r9;
    @%p2 bra FOUND;
    add.u32 %r11, %r11, 1;
    add.u32 %r12, %r12, 1;
    setp.ge.u32 %p3, %r12, 256;
    @%p3 bra NOTFOUND;
    bra PROBE;

FOUND:
    cvt.u32.u64 %r14, %rd16;
    sub.u32 %r14, %r14, 1;
    setp.ge.u32 %p2, %r14, %r18;
    @%p2 bra NOTFOUND;
    cvt.u64.u32 %rd18, %r14;

    cvt.u64.u32 %rd19, %r8;

    add.u64 %rd21, %rd9, %rd19;
    mov.u32 %r15, 1;
    st.global.u8 [%rd21], %r15;

    cvt.u64.u32 %rd22, %r4;
    mul.lo.u64 %rd23, %rd19, %rd22;
    mul.lo.u64 %rd23, %rd23, 4;
    mul.lo.u64 %rd24, %rd18, 4;

    add.u64 %rd25, %rd1, %rd3;
    add.u64 %rd25, %rd25, %rd24;
    ld.global.s32 %r17, [%rd25];
    add.u64 %rd26, %rd8, %rd23;
    st.global.s32 [%rd26], %r17;

    setp.le.u32 %p4, %r4, 1;
    @%p4 bra DONE;
    add.u64 %rd25, %rd1, %rd4;
    add.u64 %rd25, %rd25, %rd24;
    ld.global.s32 %r17, [%rd25];
    add.u64 %rd26, %rd8, %rd23;
    add.u64 %rd26, %rd26, 4;
    st.global.s32 [%rd26], %r17;

    setp.le.u32 %p4, %r4, 2;
    @%p4 bra DONE;
    add.u64 %rd25, %rd1, %rd5;
    add.u64 %rd25, %rd25, %rd24;
    ld.global.s32 %r17, [%rd25];
    add.u64 %rd26, %rd8, %rd23;
    add.u64 %rd26, %rd26, 8;
    st.global.s32 [%rd26], %r17;

    setp.le.u32 %p4, %r4, 3;
    @%p4 bra DONE;
    add.u64 %rd25, %rd1, %rd6;
    add.u64 %rd25, %rd25, %rd24;
    ld.global.s32 %r17, [%rd25];
    add.u64 %rd26, %rd8, %rd23;
    add.u64 %rd26, %rd26, 12;
    st.global.s32 [%rd26], %r17;
    bra DONE;

NOTFOUND:
    cvt.u64.u32 %rd19, %r8;
    add.u64 %rd21, %rd9, %rd19;
    mov.u32 %r15, 2;
    st.global.u8 [%rd21], %r15;

DONE:
    ret;
}
"#;

    if needles.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if projection_offsets.is_empty() || projection_offsets.len() > MAX_PROJECTIONS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            projection_offsets.len(),
        ));
    }
    if row_count == 0 || row_count >= u32::MAX as u64 || index_ptr == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let primary = resident.primary_arc();
    if !Arc::ptr_eq(&primary, &index.primary_arc()) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    validate_i32_index_geometry(
        index.metadata().allocated_bytes,
        index_table_mask,
        index_hash_shift,
    )?;
    for byte_offset in projection_offsets {
        if byte_offset % std::mem::align_of::<i32>() as u64 != 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(*byte_offset).unwrap_or(usize::MAX),
            ));
        }
        let bytes = row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .and_then(|bytes| byte_offset.checked_add(bytes))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if bytes > resident.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
        }
    }
    let needle_count_u32 = u32::try_from(needles.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;
    // DENSE single-shard layout: one slot per needle (gaps for absent), so values is sized
    // `needle_count*proj` (same as the atomic worst case). Status is initialized to 0 and the kernel writes
    // one byte per needle: 1 found or 2 not found.
    let output_cells = (needles.len() as u64)
        .checked_mul(projection_offsets.len() as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = usize::try_from(
        output_cells
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let status_bytes = needles.len();
    let needle_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

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
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    primary.set_current()?;

    let needles_host_guard = I32NeedlesHostGuard::stage(&primary, needles);

    let needles_guard = primary.lease_device_buffer_owned(needle_bytes)?;
    let values_guard = primary.lease_device_buffer_owned(output_bytes)?;
    let status_guard = primary.lease_device_buffer_owned(status_bytes)?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_resident_i32_index_probe_dense", &ptx)?;

    let mut resident_arg = resident.device_ptr();
    let mut index_arg = index_ptr;
    let mut table_mask_arg = index_table_mask;
    let mut hash_shift_arg = index_hash_shift;
    let mut row_count_arg = row_count as u32;
    let mut needle_count_arg = needle_count_u32;
    let mut projection_count_arg = u32::try_from(projection_offsets.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(projection_offsets.len()))?;
    let mut projected_offsets = [0_u64; MAX_PROJECTIONS];
    for (idx, offset) in projection_offsets.iter().enumerate() {
        projected_offsets[idx] = *offset;
    }
    let mut needles_arg = needles_guard.ptr;
    let mut output_arg = values_guard.ptr;
    let mut status_arg = status_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut index_arg as *mut u64).cast::<c_void>(),
        (&mut table_mask_arg as *mut u32).cast::<c_void>(),
        (&mut hash_shift_arg as *mut u32).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
        (&mut needle_count_arg as *mut u32).cast::<c_void>(),
        (&mut projection_count_arg as *mut u32).cast::<c_void>(),
        (&mut projected_offsets[0] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[1] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[2] as *mut u64).cast::<c_void>(),
        (&mut projected_offsets[3] as *mut u64).cast::<c_void>(),
        (&mut needles_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut status_arg as *mut u64).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = needle_count_u32.div_ceil(threads_per_block);

    let stream_owned = PooledStreamOwned {
        primary: Arc::clone(&primary),
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let pooled = stream_owned
        .pooled
        .as_ref()
        .expect("pooled stream just leased");
    let stream = pooled.stream;
    let start_event = pooled.start_event;
    let stop_event = pooled.stop_event;
    let timed = !start_event.is_null() && !stop_event.is_null();

    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    // Own the H2D source and every asynchronous destination before the first enqueue. The public safe API
    // returns this submission independently of the caller's slice borrow, so retaining only `needles.as_ptr()`
    // would permit pinned input memory to be freed/reused while DMA was still consuming it.
    let submission = CudaI32IndexProbeDenseSubmission {
        projection_count: projection_offsets.len(),
        needles_len: needles.len(),
        primary: Arc::clone(&primary),
        _resident_allocation_guard: resident.allocation_arc(),
        values_guard,
        status_guard,
        _needles_guard: needles_guard,
        _needles_host_guard: needles_host_guard,
        stream: Some(stream_owned),
        timed,
        _wave_index_guard: Some(Arc::clone(index)),
        _multi_shard_plan_guard: None,
        multi_shard_binary_mode: false,
    };

    // GAP GUARD: memset `status` to 0 before the kernel so any slot a thread fails to write surfaces as 0.
    // Every in-bounds thread writes a definitive 1/2; compatibility conversion and engine assembly reject 0.
    let async_ops = match (primary.cu_memcpy_htod_async, primary.cu_memset_d8_async) {
        (Some(htod), Some(memset)) => Some((htod, memset)),
        _ => None,
    };
    if let Some((htod_async, memset_async)) = async_ops {
        check_cuda(unsafe {
            htod_async(
                submission._needles_guard.ptr,
                submission._needles_host_guard.as_ptr(),
                needle_bytes,
                stream,
            )
        })
        .map_err(drain_err)?;
        check_cuda(unsafe { memset_async(submission.status_guard.ptr, 0, status_bytes, stream) })
            .map_err(drain_err)?;
    } else {
        check_cuda(unsafe {
            cu_memcpy_htod(
                submission._needles_guard.ptr,
                submission._needles_host_guard.as_ptr(),
                needle_bytes,
            )
        })
        .map_err(drain_err)?;
        check_cuda(unsafe { cu_memset_d8(submission.status_guard.ptr, 0, status_bytes) })
            .map_err(drain_err)?;
    }

    if timed {
        check_cuda(unsafe { (primary.cu_event_record)(start_event, stream) }).map_err(drain_err)?;
    }
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
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })
    .map_err(drain_err)?;
    if timed {
        check_cuda(unsafe { (primary.cu_event_record)(stop_event, stream) }).map_err(drain_err)?;
    }

    Ok(submission)
}

/// Sub-slice 8 v2: one shard's inputs for the multi-shard dense-emit probe. One kernel launch reads these from
/// an 88-byte-per-shard device descriptor array, probes every linear-mode candidate or one binary-range candidate
/// per needle, and emits one dense slot (1xN output, no SxN D2H or host merge).
pub struct MultiShardProbeShard {
    /// The shard's column BUFFER (probed at the capacity-strided projection offsets).
    pub resident: Arc<CudaResidentDeviceMemory>,
    /// The shard's DEVICE hash index (`(key<<32)|(row+1)`), pinned until the kernel completes.
    pub index: Arc<CudaResidentDeviceMemory>,
    pub table_mask: u32,
    pub hash_shift: u32,
    /// Byte offsets of the projected int4 columns in `resident` (1..=4, capacity-strided).
    pub projection_offsets: Vec<u64>,
    /// The shard's live row count (for the projection bounds check).
    pub row_count: u64,
    /// Per-row birth/death stamps from the SAME published shard snapshot. `None` means born-visible/all-live.
    /// The Arcs pin these regions through completion; the descriptor passes only their device pointers.
    pub created_by: Option<Arc<CudaResidentDeviceMemory>>,
    pub deleted_by: Option<Arc<CudaResidentDeviceMemory>>,
    /// Zone map [min, max] of the FILTER column for this shard (from `resident_device_int4_column_stats`);
    /// the kernel skips this shard for a needle outside [min, max] (on-device prune). Pass `(i32::MIN,
    /// i32::MAX)` when the shard has no stat for the column -> always in-range (matches the scan, which keeps
    /// a shard with no zone-map stat). NULLs are EXCLUDED from the stat, so the keep-shard-0 fallback below
    /// handles a needle (0) that would match a NULL-stored-as-0 row in a shard whose [min,max] excludes 0.
    pub min: i32,
    pub max: i32,
}

const MULTI_SHARD_MAX_PROJECTIONS: usize = 4;
const MULTI_SHARD_DESC_U64_PER_SHARD: usize = 11;

/// Generation-owned device route for repeated multi-shard point batches. The immutable descriptor array
/// lives on the GPU and this owner pins every payload, index, and MVCC sidecar it names. Reusing the plan
/// removes per-batch O(shards) host descriptor encoding, Arc churn, and descriptor H2D while preserving the
/// exact published device generations that established the route.
pub struct CudaI32MultiShardProbePlan {
    primary: Arc<GpuPrimaryContext>,
    projection_count: usize,
    shard_count: u32,
    binary_mode: bool,
    descriptor_guard: PooledDeviceBufferOwned,
    _resource_guards: Vec<Arc<CudaResidentDeviceMemory>>,
}

impl std::fmt::Debug for CudaI32MultiShardProbePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaI32MultiShardProbePlan")
            .field("projection_count", &self.projection_count)
            .field("shard_count", &self.shard_count)
            .field("binary_mode", &self.binary_mode)
            .finish_non_exhaustive()
    }
}

impl CudaI32MultiShardProbePlan {
    /// Exact live device bytes held by the prepared descriptor table (the pooled allocation bucket,
    /// not merely the requested descriptor payload). Engines retaining a plan must charge this value.
    pub fn descriptor_allocated_bytes(&self) -> u64 {
        self.descriptor_guard.capacity as u64
    }
}

pub(super) fn prepare_cuda_resident_i32_multi_shard_index_probe_dense(
    ctx: &CudaResidentDeviceMemory,
    shards: &[MultiShardProbeShard],
) -> Result<CudaI32MultiShardProbePlan, CudaRuntimeProbeError> {
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;

    if shards.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let probe = Probe::start();
    let primary = ctx.primary_arc();
    let projection_count = shards[0].projection_offsets.len();
    if projection_count == 0 || projection_count > MULTI_SHARD_MAX_PROJECTIONS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(projection_count));
    }
    let shard_count = u32::try_from(shards.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(shards.len()))?;
    let descriptor_capacity = shards
        .len()
        .checked_mul(MULTI_SHARD_DESC_U64_PER_SHARD)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let guard_capacity = shards
        .len()
        .checked_mul(4)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut desc = Vec::with_capacity(descriptor_capacity);
    let mut resource_guards = Vec::with_capacity(guard_capacity);
    for shard in shards {
        if shard.projection_offsets.len() != projection_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                shard.projection_offsets.len(),
            ));
        }
        if shard.row_count == 0
            || shard.row_count >= u32::MAX as u64
            || shard.resident.device_ptr() == 0
            || shard.index.device_ptr() == 0
            || shard.min > shard.max
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        if !Arc::ptr_eq(&primary, &shard.resident.primary_arc())
            || !Arc::ptr_eq(&primary, &shard.index.primary_arc())
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        validate_i32_index_geometry(
            shard.index.metadata().allocated_bytes,
            shard.table_mask,
            shard.hash_shift,
        )?;
        let allocated = shard.resident.metadata().allocated_bytes;
        for &byte_offset in &shard.projection_offsets {
            if byte_offset % std::mem::align_of::<i32>() as u64 != 0 {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    usize::try_from(byte_offset).unwrap_or(usize::MAX),
                ));
            }
            let end = shard
                .row_count
                .checked_mul(std::mem::size_of::<i32>() as u64)
                .and_then(|bytes| byte_offset.checked_add(bytes))
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if end > allocated {
                return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
            }
        }
        desc.push(shard.resident.device_ptr());
        desc.push(shard.index.device_ptr());
        desc.push((shard.table_mask as u64) | ((shard.hash_shift as u64) << 32));
        let mut offsets = [0_u64; MULTI_SHARD_MAX_PROJECTIONS];
        for (idx, &offset) in shard.projection_offsets.iter().enumerate() {
            offsets[idx] = offset;
        }
        desc.extend_from_slice(&offsets);
        desc.push((shard.min as u32 as u64) | ((shard.max as u32 as u64) << 32));
        for region in [&shard.created_by, &shard.deleted_by] {
            if let Some(region) = region {
                if region.device_ptr() == 0 || !Arc::ptr_eq(&primary, &region.primary_arc()) {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
                }
                let required = shard
                    .row_count
                    .checked_mul(std::mem::size_of::<u64>() as u64)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                if required > region.metadata().allocated_bytes {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(required as usize));
                }
                desc.push(region.device_ptr());
                resource_guards.push(Arc::clone(region));
            } else {
                desc.push(0);
            }
        }
        desc.push(shard.row_count);
        resource_guards.push(Arc::clone(&shard.resident));
        resource_guards.push(Arc::clone(&shard.index));
    }
    probe.lap("point_multi_descriptor_encode");

    let binary_mode = shards.len() >= 2 && shards.windows(2).all(|w| w[0].max < w[1].min);
    let desc_bytes = desc
        .len()
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    Probe::value("point_multi_descriptor_bytes", desc_bytes as u64, "B");
    Probe::value(
        "point_multi_descriptor_shards",
        u64::from(shard_count),
        "shards",
    );
    let cu_memcpy_htod = unsafe {
        ctx.lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| ctx.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    primary.set_current()?;
    let descriptor_guard = primary.lease_device_buffer_owned(desc_bytes)?;
    check_cuda(unsafe {
        cu_memcpy_htod(
            descriptor_guard.ptr,
            desc.as_ptr().cast::<c_void>(),
            desc_bytes,
        )
    })?;
    probe.lap("point_multi_plan_upload");
    Ok(CudaI32MultiShardProbePlan {
        primary,
        projection_count,
        shard_count,
        binary_mode,
        descriptor_guard,
        _resource_guards: resource_guards,
    })
}

pub(super) fn submit_cuda_resident_i32_multi_shard_index_probe_dense(
    ctx: &CudaResidentDeviceMemory,
    shards: &[MultiShardProbeShard],
    needles: &[i32],
    read_snapshot: u64,
) -> Result<CudaI32IndexProbeDenseSubmission, CudaRuntimeProbeError> {
    let plan = Arc::new(prepare_cuda_resident_i32_multi_shard_index_probe_dense(
        ctx, shards,
    )?);
    submit_cuda_resident_i32_multi_shard_index_probe_dense_prepared(
        ctx,
        plan,
        needles,
        read_snapshot,
    )
}

/// Sub-slice 8 v2 (charter-faithful): the MULTI-SHARD dense-emit point-lookup probe. In one kernel launch, each
/// needle probes every candidate shard from the device descriptor array: all shards in linear mode or the one
/// range candidate in binary mode. Zero visible matches emits status 2, one gathers the projected values and emits
/// status 1, and a second emits status 3 so the unique route declines. This replaces the per-shard kernel plus host
/// merge: D2H is `needle_count*projection_count`, not `shards*needle_count`, and output is needle-indexed directly.
/// All shards must share `projection_count` (1..=4).
pub(super) fn submit_cuda_resident_i32_multi_shard_index_probe_dense_prepared(
    ctx: &CudaResidentDeviceMemory,
    plan: Arc<CudaI32MultiShardProbePlan>,
    needles: &[i32],
    read_snapshot: u64,
) -> Result<CudaI32IndexProbeDenseSubmission, CudaRuntimeProbeError> {
    let probe = Probe::start();
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
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

    // PTX: outer SHARD loop over an 88 B/shard descriptor array. Each thread probes every candidate shard: zero
    // visible matches emits status=2, one emits values + status=1, and a second emits status=3 so the unique route
    // declines. It reuses the single-shard hash lookup and gather-address contract while adding descriptor reads,
    // linear/binary candidate routing, zone pruning, MVCC visibility, and duplicate detection. ASCII-only.
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_multi_shard_i32_index_probe_dense(
    .param .u64 desc_array_ptr,
    .param .u32 shard_count,
    .param .u32 needle_count,
    .param .u32 projection_count,
    .param .u64 needles_ptr,
    .param .u64 out_values_ptr,
    .param .u64 out_status_ptr,
    .param .u32 binary_mode,
    .param .u64 read_snapshot
)
{
    .reg .pred %p<9>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<48>;

    ld.param.u64 %rd1, [desc_array_ptr];
    ld.param.u32 %r1, [shard_count];
    ld.param.u32 %r2, [needle_count];
    ld.param.u32 %r3, [projection_count];
    ld.param.u64 %rd2, [needles_ptr];
    ld.param.u64 %rd3, [out_values_ptr];
    ld.param.u64 %rd4, [out_status_ptr];
    ld.param.u32 %r26, [binary_mode];
    ld.param.u64 %rd36, [read_snapshot];

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mad.lo.u32 %r7, %r5, %r6, %r4;
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra DONE;
    setp.eq.u32 %p1, %r3, 0;
    @%p1 bra DONE;

    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd6, %rd2, %rd5;
    ld.global.s32 %r8, [%rd6];

    // BINARY MODE (set by the host ONLY when it proves the shards are ascending-disjoint: max[i-1] < min[i]).
    // Then a needle maps to EXACTLY ONE shard, so binary-search the min-sorted (== table-order) descriptors for
    // the containing shard in O(log shards) instead of scanning all. Reuses the linear PROBE/FOUND/emit path by
    // jumping into SHARD with the candidate index k, bound=k+1, force=1. A gap / out-of-all-ranges falls to
    // keep-shard-0 (descriptor[0] == table_shards[0]) exactly like the linear ALLDONE fallback.
    setp.eq.u32 %p1, %r26, 0;
    @%p1 bra LINEARINIT;
    mov.u32 %r27, 0;
    mov.u32 %r28, %r1;
BSEARCH:
    setp.ge.u32 %p1, %r27, %r28;
    @%p1 bra BDONE;
    add.u32 %r29, %r27, %r28;
    shr.u32 %r29, %r29, 1;
    mul.wide.u32 %rd32, %r29, 88;
    add.u64 %rd33, %rd1, %rd32;
    ld.global.u64 %rd34, [%rd33+56];
    cvt.u32.u64 %r30, %rd34;
    setp.le.s32 %p1, %r30, %r8;
    @%p1 bra BLO;
    mov.u32 %r28, %r29;
    bra BSEARCH;
BLO:
    add.u32 %r27, %r29, 1;
    bra BSEARCH;
BDONE:
    setp.eq.u32 %p1, %r27, 0;
    @%p1 bra BKEEP0;
    sub.u32 %r9, %r27, 1;
    mul.wide.u32 %rd32, %r9, 88;
    add.u64 %rd33, %rd1, %rd32;
    ld.global.u64 %rd34, [%rd33+56];
    shr.u64 %rd35, %rd34, 32;
    cvt.u32.u64 %r21, %rd35;
    setp.gt.s32 %p1, %r8, %r21;
    @%p1 bra BKEEP0;
    add.u32 %r23, %r9, 1;
    mov.u32 %r24, 1;
    mov.u32 %r25, 0;
    mov.u32 %r19, 0;
    mov.u32 %r22, 1;
    bra SHARD;
BKEEP0:
    mov.u32 %r9, 0;
    mov.u32 %r23, 1;
    mov.u32 %r24, 1;
    mov.u32 %r25, 1;
    mov.u32 %r19, 0;
    mov.u32 %r22, 0;
    bra SHARD;

LINEARINIT:
    mov.u32 %r9, 0;
    mov.u32 %r19, 0;
    mov.u32 %r22, 0;
    mov.u32 %r23, %r1;
    mov.u32 %r24, 0;
    mov.u32 %r25, 0;

SHARD:
    setp.ge.u32 %p1, %r9, %r23;
    @%p1 bra ALLDONE;
    mul.wide.u32 %rd7, %r9, 88;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u64 %rd9, [%rd8];
    ld.global.u64 %rd10, [%rd8+8];
    ld.global.u64 %rd11, [%rd8+16];
    ld.global.u64 %rd30, [%rd8+56];
    ld.global.u64 %rd37, [%rd8+64];
    ld.global.u64 %rd38, [%rd8+72];
    ld.global.u64 %rd44, [%rd8+80];
    cvt.u32.u64 %r31, %rd44;
    cvt.u32.u64 %r20, %rd30;
    shr.u64 %rd31, %rd30, 32;
    cvt.u32.u64 %r21, %rd31;
    setp.ne.u32 %p1, %r24, 0;
    @%p1 bra INRANGE;
    setp.lt.s32 %p1, %r8, %r20;
    @%p1 bra NEXTSHARD;
    setp.gt.s32 %p1, %r8, %r21;
    @%p1 bra NEXTSHARD;
INRANGE:
    mov.u32 %r22, 1;
    cvt.u32.u64 %r10, %rd11;
    shr.u64 %rd12, %rd11, 32;
    cvt.u32.u64 %r11, %rd12;
    mul.lo.u32 %r12, %r8, 2654435761;
    shr.u32 %r13, %r12, %r11;
    mov.u32 %r14, 0;

PROBE:
    and.b32 %r13, %r13, %r10;
    mul.wide.u32 %rd13, %r13, 8;
    add.u64 %rd14, %rd10, %rd13;
    ld.global.u64 %rd15, [%rd14];
    setp.eq.u64 %p2, %rd15, 0;
    @%p2 bra NEXTSHARD;
    shr.u64 %rd16, %rd15, 32;
    cvt.u32.u64 %r15, %rd16;
    setp.eq.s32 %p2, %r15, %r8;
    @%p2 bra FOUND;
    bra ADVANCEPROBE;

NEXTSHARD:
    add.u32 %r9, %r9, 1;
    bra SHARD;

FOUND:
    cvt.u32.u64 %r16, %rd15;
    sub.u32 %r16, %r16, 1;
    setp.ge.u32 %p8, %r16, %r31;
    @%p8 bra ADVANCEPROBE;
    cvt.u64.u32 %rd17, %r16;
    mul.lo.u64 %rd39, %rd17, 8;
    setp.eq.u64 %p6, %rd37, 0;
    @%p6 bra CREATEDOK;
    add.u64 %rd40, %rd37, %rd39;
    ld.global.u64 %rd41, [%rd40];
    setp.gt.u64 %p6, %rd41, %rd36;
    @%p6 bra ADVANCEPROBE;
CREATEDOK:
    setp.eq.u64 %p7, %rd38, 0;
    @%p7 bra VISIBLE;
    add.u64 %rd42, %rd38, %rd39;
    ld.global.u64 %rd43, [%rd42];
    setp.le.u64 %p7, %rd43, %rd36;
    @%p7 bra ADVANCEPROBE;
VISIBLE:
    setp.eq.u32 %p5, %r19, 1;
    @%p5 bra DUP;
    mov.u32 %r19, 1;
    mul.lo.u64 %rd24, %rd17, 4;

    cvt.u64.u32 %rd18, %r7;
    add.u64 %rd20, %rd4, %rd18;
    mov.u32 %r17, 1;
    st.global.u8 [%rd20], %r17;

    cvt.u64.u32 %rd21, %r3;
    mul.lo.u64 %rd22, %rd18, %rd21;
    mul.lo.u64 %rd22, %rd22, 4;
    add.u64 %rd23, %rd3, %rd22;

    ld.global.u64 %rd25, [%rd8+24];
    add.u64 %rd26, %rd9, %rd25;
    add.u64 %rd26, %rd26, %rd24;
    ld.global.s32 %r18, [%rd26];
    st.global.s32 [%rd23], %r18;

    setp.le.u32 %p4, %r3, 1;
    @%p4 bra AFTEREMIT;
    ld.global.u64 %rd25, [%rd8+32];
    add.u64 %rd26, %rd9, %rd25;
    add.u64 %rd26, %rd26, %rd24;
    ld.global.s32 %r18, [%rd26];
    add.u64 %rd27, %rd23, 4;
    st.global.s32 [%rd27], %r18;

    setp.le.u32 %p4, %r3, 2;
    @%p4 bra AFTEREMIT;
    ld.global.u64 %rd25, [%rd8+40];
    add.u64 %rd26, %rd9, %rd25;
    add.u64 %rd26, %rd26, %rd24;
    ld.global.s32 %r18, [%rd26];
    add.u64 %rd27, %rd23, 8;
    st.global.s32 [%rd27], %r18;

    setp.le.u32 %p4, %r3, 3;
    @%p4 bra AFTEREMIT;
    ld.global.u64 %rd25, [%rd8+48];
    add.u64 %rd26, %rd9, %rd25;
    add.u64 %rd26, %rd26, %rd24;
    ld.global.s32 %r18, [%rd26];
    add.u64 %rd27, %rd23, 12;
    st.global.s32 [%rd27], %r18;

AFTEREMIT:
ADVANCEPROBE:
    add.u32 %r13, %r13, 1;
    add.u32 %r14, %r14, 1;
    setp.ge.u32 %p3, %r14, 256;
    @%p3 bra NEXTSHARD;
    bra PROBE;

DUP:
    cvt.u64.u32 %rd18, %r7;
    add.u64 %rd20, %rd4, %rd18;
    mov.u32 %r17, 3;
    st.global.u8 [%rd20], %r17;
    bra DONE;

ALLDONE:
    setp.eq.u32 %p5, %r19, 1;
    @%p5 bra DONE;
    setp.eq.u32 %p5, %r25, 1;
    @%p5 bra WRITEABSENT;
    setp.eq.u32 %p5, %r22, 1;
    @%p5 bra WRITEABSENT;
    mov.u32 %r9, 0;
    mov.u32 %r23, 1;
    mov.u32 %r24, 1;
    mov.u32 %r25, 1;
    bra SHARD;

WRITEABSENT:
    cvt.u64.u32 %rd18, %r7;
    add.u64 %rd20, %rd4, %rd18;
    mov.u32 %r17, 2;
    st.global.u8 [%rd20], %r17;

DONE:
    ret;
}
"#;

    if needles.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if !Arc::ptr_eq(&plan.primary, &ctx.primary_arc()) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    let projection_count = plan.projection_count;
    let shard_count_u32 = plan.shard_count;
    let binary_mode = u32::from(plan.binary_mode);
    let needle_count_u32 = u32::try_from(needles.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;
    let projection_count_u32 = projection_count as u32;

    let output_bytes = usize::try_from(
        (needles.len() as u64)
            .checked_mul(projection_count as u64)
            .and_then(|c| c.checked_mul(std::mem::size_of::<i32>() as u64))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let status_bytes = needles.len();
    let needle_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    Probe::value("point_multi_h2d_bytes", needle_bytes as u64, "B");
    Probe::value(
        "point_multi_d2h_bytes",
        output_bytes.saturating_add(status_bytes) as u64,
        "B",
    );
    probe.lap("point_multi_submission_prepare");

    let cu_memset_d8 = unsafe {
        ctx.lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| ctx.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        ctx.lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| ctx.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        ctx.lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let primary = ctx.primary_arc();
    primary.set_current()?;

    let needles_guard = primary.lease_device_buffer_owned(needle_bytes)?;
    let values_guard = primary.lease_device_buffer_owned(output_bytes)?;
    let status_guard = primary.lease_device_buffer_owned(status_bytes)?;
    let needles_host_guard = I32NeedlesHostGuard::stage(&primary, needles);
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function =
        primary.cached_function(c"gpu_db_resident_multi_shard_i32_index_probe_dense", &ptx)?;
    probe.lap("point_multi_resource_prepare");

    let mut desc_arg = plan.descriptor_guard.ptr;
    let mut shard_count_arg = shard_count_u32;
    let mut needle_count_arg = needle_count_u32;
    let mut projection_count_arg = projection_count_u32;
    let mut needles_arg = needles_guard.ptr;
    let mut output_arg = values_guard.ptr;
    let mut status_arg = status_guard.ptr;
    let mut binary_mode_arg = binary_mode;
    let mut read_snapshot_arg = read_snapshot;
    let mut args = [
        (&mut desc_arg as *mut u64).cast::<c_void>(),
        (&mut shard_count_arg as *mut u32).cast::<c_void>(),
        (&mut needle_count_arg as *mut u32).cast::<c_void>(),
        (&mut projection_count_arg as *mut u32).cast::<c_void>(),
        (&mut needles_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut status_arg as *mut u64).cast::<c_void>(),
        (&mut binary_mode_arg as *mut u32).cast::<c_void>(),
        (&mut read_snapshot_arg as *mut u64).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = needle_count_u32.div_ceil(threads_per_block);

    let stream_owned = PooledStreamOwned {
        primary: Arc::clone(&primary),
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let pooled = stream_owned
        .pooled
        .as_ref()
        .expect("pooled stream just leased");
    let stream = pooled.stream;
    let start_event = pooled.start_event;
    let stop_event = pooled.stop_event;
    let timed = !start_event.is_null() && !stop_event.is_null();

    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    // Establish the draining owner before the FIRST asynchronous operation, not merely before launch. H2D and
    // memset also retain the host needle bytes and pooled device buffers until the stream reaches them. Any
    // error or panic from this point therefore synchronizes before those resources or the stream return to a
    // shared pool.
    let submission = CudaI32IndexProbeDenseSubmission {
        projection_count,
        needles_len: needles.len(),
        primary: Arc::clone(&primary),
        _resident_allocation_guard: ctx.allocation_arc(),
        values_guard,
        status_guard,
        _needles_guard: needles_guard,
        _needles_host_guard: needles_host_guard,
        stream: Some(stream_owned),
        timed,
        _wave_index_guard: None,
        _multi_shard_plan_guard: Some(plan),
        multi_shard_binary_mode: binary_mode == 1,
    };

    // The generation-owned plan keeps the immutable descriptor array resident. Per batch, upload only needles and
    // clear status; compatibility conversion and engine assembly treat any unwritten zero as invalid.
    let async_ops = match (primary.cu_memcpy_htod_async, primary.cu_memset_d8_async) {
        (Some(htod), Some(memset)) => Some((htod, memset)),
        _ => None,
    };
    if let Some((htod_async, memset_async)) = async_ops {
        check_cuda(unsafe {
            htod_async(
                submission._needles_guard.ptr,
                submission._needles_host_guard.as_ptr(),
                needle_bytes,
                stream,
            )
        })
        .map_err(drain_err)?;
        check_cuda(unsafe { memset_async(submission.status_guard.ptr, 0, status_bytes, stream) })
            .map_err(drain_err)?;
    } else {
        check_cuda(unsafe {
            cu_memcpy_htod(
                submission._needles_guard.ptr,
                submission._needles_host_guard.as_ptr(),
                needle_bytes,
            )
        })
        .map_err(drain_err)?;
        check_cuda(unsafe { cu_memset_d8(submission.status_guard.ptr, 0, status_bytes) })
            .map_err(drain_err)?;
    }
    #[cfg(test)]
    panic_at_dense_phase(1);
    probe.lap("point_multi_transfer_enqueue");

    if timed {
        check_cuda(unsafe { (primary.cu_event_record)(start_event, stream) }).map_err(drain_err)?;
    }
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
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })
    .map_err(drain_err)?;
    if timed {
        check_cuda(unsafe { (primary.cu_event_record)(stop_event, stream) }).map_err(drain_err)?;
    }
    probe.lap("point_multi_launch_enqueue");
    Ok(submission)
}

impl CudaI32IndexProbeDenseSubmission {
    /// Drain into the established dense compatibility layout: one value slot per needle, byte status widened to
    /// `Vec<u32>`, and empty needle/row-index vectors. Status 3 becomes `DuplicatePointReadMatch`; other unknown
    /// statuses are rejected. One covering `cuStreamSynchronize` suffices because `needles_len` is known a priori,
    /// so there is no count round-trip.
    pub fn complete_detached_columnar(
        self,
    ) -> Result<(CudaI32BatchProjectionColumns, Option<u64>), CudaRuntimeProbeError> {
        let (compact, elapsed_us) = self.complete_detached_columnar_compact()?;
        Ok((compact.into_compat()?, elapsed_us))
    }

    /// Hot-path completion retaining the device/D2H byte-sized status representation. This is additive to
    /// `complete_detached_columnar`; it does not alter that method's public `Vec<u32>` result contract.
    pub fn complete_detached_columnar_compact(
        mut self,
    ) -> Result<(CudaI32DenseBatchProjection, Option<u64>), CudaRuntimeProbeError> {
        let probe = Probe::start();
        let primary = Arc::clone(&self.primary);
        primary.set_current()?;
        // Keep the stream inside `self` until the covering sync and event read are complete. Submission Drop
        // protects device resources; the host-copy guard below separately precedes local D2H destinations in
        // unwind order.
        let (stream, start_event, stop_event) = {
            let stream_owned = self
                .stream
                .as_ref()
                .expect("pooled stream held until complete");
            let pooled = stream_owned
                .pooled
                .as_ref()
                .expect("pooled stream held until complete");
            (pooled.stream, pooled.start_event, pooled.stop_event)
        };

        let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
            unsafe {
                let _ = (primary.cu_stream_synchronize)(stream);
            }
            err
        };

        let nc = self.needles_len;
        let proj = self.projection_count;
        let mut values_raw = vec![0_i32; nc.saturating_mul(proj)];
        let mut status_bytes = vec![0_u8; nc];
        let mut values_pinned = None;
        let mut status_pinned = None;
        // Declared after every D2H destination so unwind order is drain first, then pinned leases/Vecs. The
        // by-value `self` parameter drops after these locals, so submission Drop alone is too late to protect
        // host memory from an in-flight asynchronous copy.
        let mut host_copy_drain = InFlightHostCopyDrain::new(Arc::clone(&primary), stream);
        if let Some(dtoh_async) = primary.cu_memcpy_dtoh_async {
            // Queue both known-size result transfers immediately behind the kernel on its stream. Host Vec/
            // pinned-buffer preparation can overlap the already-running launch, and one covering sync below
            // establishes visibility for both kernel stores and D2H writes. The former pre-transfer kernel
            // sync needlessly serialized that preparation and paid a second stream barrier per point batch.
            values_pinned = stage_result_dtoh_async(
                primary.as_ref(),
                dtoh_async,
                stream,
                self.values_guard.ptr,
                &mut values_raw,
            )
            .map_err(drain_err)?;
            status_pinned = stage_result_dtoh_async(
                primary.as_ref(),
                dtoh_async,
                stream,
                self.status_guard.ptr,
                &mut status_bytes,
            )
            .map_err(drain_err)?;
        }
        #[cfg(test)]
        panic_at_dense_phase(2);
        probe.lap("point_multi_result_prepare");

        // The covering sync is the visibility barrier — plain `st.global` status writes in the kernel and the
        // stream-ordered result copies complete before the host reads below. The blocking-copy fallback also
        // requires the kernel to be idle before copying on the default stream.
        check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;
        host_copy_drain.disarm();
        probe.lap("point_multi_launch_readback_sync");

        let elapsed_us = if self.timed {
            let mut elapsed_ms = 0.0_f32;
            check_cuda(unsafe {
                (primary.cu_event_elapsed_time)(&mut elapsed_ms, start_event, stop_event)
            })?;
            Some((f64::from(elapsed_ms) * 1_000.0).ceil() as u64)
        } else {
            None
        };
        // CUDA work and timing-event reads are complete. From here onward, unwinding may return the stream
        // directly because no device operation still references the owned buffers or prepared plan.
        let _stream_owned = self
            .stream
            .take()
            .expect("pooled stream held through covering synchronization");
        if let Some(elapsed_us) = elapsed_us {
            Probe::value("point_multi_kernel_event", elapsed_us, "us");
        }

        if primary.cu_memcpy_dtoh_async.is_some() {
            copy_pinned_into(&values_pinned, &mut values_raw);
            copy_pinned_into(&status_pinned, &mut status_bytes);
        } else if !values_raw.is_empty() {
            check_cuda(unsafe {
                (primary.cu_memcpy_dtoh)(
                    values_raw.as_mut_ptr().cast::<c_void>(),
                    self.values_guard.ptr,
                    values_raw.len() * std::mem::size_of::<i32>(),
                )
            })
            .map_err(drain_err)?;
            check_cuda(unsafe {
                (primary.cu_memcpy_dtoh)(
                    status_bytes.as_mut_ptr().cast::<c_void>(),
                    self.status_guard.ptr,
                    status_bytes.len(),
                )
            })
            .map_err(drain_err)?;
        }
        probe.lap("point_multi_result_host_copy");

        // Return the DENSE LAYOUT as-is (NO compaction here): `values_raw` is one slot per needle (gaps) +
        // `status`. The engine's `assemble_batched_rows` compacts it in ONE sequential pass — compacting here
        // AND letting the engine re-scatter would be two passes (measured slower than the atomic scatter). The
        // compatibility conversion and engine assembly each validate status before consuming a slot.
        let columns = CudaI32DenseBatchProjection {
            values: values_raw,
            projection_count: proj,
            status: status_bytes,
        };
        probe.lap("point_multi_completion_frame");
        Ok((columns, elapsed_us))
    }
}

impl Drop for CudaI32IndexProbeDenseSubmission {
    fn drop(&mut self) {
        // Same drop-without-complete safety drain as `CudaI32EqualAnyProjectSubmission`: if dropped before
        // `complete` (early `Err`/cancel/panic), sync the held stream BEFORE the field guards return the
        // device buffers + stream to the SHARED pools (else a concurrent leaser re-leases memory the in-flight
        // kernel/HtoD/memset still writes). A successful `complete_detached_columnar` `take()`s `stream`, so
        // this is skipped on the success path. Best-effort, NO PANIC.
        if let Some(stream_owned) = self.stream.as_ref() {
            if let Some(pooled) = stream_owned.pooled.as_ref() {
                let _ = self.primary.set_current();
                unsafe {
                    let _ = (self.primary.cu_stream_synchronize)(pooled.stream);
                }
            }
        }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::{validate_i32_index_geometry, CudaI32DenseBatchProjection};
    use crate::CudaRuntimeProbeError;

    #[test]
    fn multi_shard_index_geometry_rejects_malformed_descriptor_bounds() {
        validate_i32_index_geometry(128, 15, 28).expect("16 u64 slots");
        assert!(validate_i32_index_geometry(127, 15, 28).is_err());
        assert!(validate_i32_index_geometry(128, 14, 28).is_err());
        assert!(validate_i32_index_geometry(128, 15, 27).is_err());
    }

    #[test]
    fn dense_compatibility_rejects_duplicate_status_instead_of_dropping_it() {
        let result = CudaI32DenseBatchProjection {
            values: vec![41],
            projection_count: 1,
            status: vec![3],
        }
        .into_compat();
        assert_eq!(
            result,
            Err(CudaRuntimeProbeError::DuplicatePointReadMatch(0))
        );
    }
}
