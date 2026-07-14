use std::ffi::c_void;
use std::sync::Arc;

use super::{
    check_cuda, CudaI32BatchProjectionColumns, CudaResidentDeviceMemory, CudaResidentReadSource,
    CudaRuntimeProbeError, GpuPrimaryContext, PooledDeviceBufferOwned, PooledStreamOwned,
};

/// DENSE-emit variant of the unique int4 index probe (DECISIONS "lpb read levers" #1). A unique index has
/// <=1 match/needle, so thread `i` writes `values[i*proj]` + `status[i]` (1=found, 2=not-found) to its OWN
/// slot — NO `atom.global.add`, NO `needle_indices`, NO `out_count`, NO `row_indices`. The host then compacts
/// sequentially by status (no random scatter). Only valid for the unique index route (the non-unique scan
/// keeps the atomic kernel + row_indices). The hash/probe logic is byte-identical to
/// `gpu_db_resident_i32_index_probe`; only the emit differs. Gaps are guarded: `status` is memset to 0, every
/// in-bounds thread writes a definitive 1/2, and the completion `debug_assert`s `status != 0`.
pub struct CudaI32IndexProbeDenseSubmission {
    projection_count: usize,
    needles_len: usize,
    primary: Arc<GpuPrimaryContext>,
    values_guard: PooledDeviceBufferOwned,
    status_guard: PooledDeviceBufferOwned,
    _needles_guard: PooledDeviceBufferOwned,
    stream: Option<PooledStreamOwned>,
    timed: bool,
    _wave_index_guard: Option<Arc<CudaResidentDeviceMemory>>,
    /// Sub-slice 8 v2 (multi-shard kernel): pins the per-shard device indexes + the descriptor-array device
    /// buffer alive until the kernel completes (the kernel reads them). Empty/None for the single-shard path.
    _multi_shard_index_guards: Vec<Arc<CudaResidentDeviceMemory>>,
    _multi_shard_desc_guard: Option<PooledDeviceBufferOwned>,
    /// Sub-slice 8 v3 (O(1) routing): true when the multi-shard kernel took the BINARY-SEARCH path (the shards
    /// were host-proven ascending-disjoint). Pure telemetry for the non-vacuity assert that binary mode fired;
    /// the result rows are byte-identical to the linear path.
    pub multi_shard_binary_mode: bool,
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
    cvt.u64.u32 %rd18, %r14;

    cvt.u64.u32 %rd19, %r8;

    mul.lo.u64 %rd20, %rd19, 4;
    add.u64 %rd21, %rd9, %rd20;
    mov.u32 %r15, 1;
    st.global.u32 [%rd21], %r15;

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
    mul.lo.u64 %rd20, %rd19, 4;
    add.u64 %rd21, %rd9, %rd20;
    mov.u32 %r15, 2;
    st.global.u32 [%rd21], %r15;

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
    if row_count == 0 || index_ptr == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    for byte_offset in projection_offsets {
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
    // DENSE: one slot per needle (gaps for absent), so values is sized `needle_count*proj` (same as the
    // atomic worst case) and `status` is `needle_count` u32s.
    let output_cells = (needles.len() as u64)
        .checked_mul(projection_offsets.len() as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = usize::try_from(
        output_cells
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let status_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
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

    let primary = resident.primary_arc();
    primary.set_current()?;

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
    let timed = !pooled.start_event.is_null() && !pooled.stop_event.is_null();

    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    // GAP GUARD: memset `status` to 0 BEFORE the kernel so any slot a thread fails to write surfaces as 0
    // (the completion `debug_assert`s != 0). Every in-bounds thread then writes a definitive 1/2.
    let async_ops = match (primary.cu_memcpy_htod_async, primary.cu_memset_d8_async) {
        (Some(htod), Some(memset)) => Some((htod, memset)),
        _ => None,
    };
    if let Some((htod_async, memset_async)) = async_ops {
        check_cuda(unsafe {
            htod_async(
                needles_guard.ptr,
                needles.as_ptr().cast::<c_void>(),
                needle_bytes,
                stream,
            )
        })
        .map_err(drain_err)?;
        check_cuda(unsafe { memset_async(status_guard.ptr, 0, status_bytes, stream) })
            .map_err(drain_err)?;
    } else {
        check_cuda(unsafe {
            cu_memcpy_htod(
                needles_guard.ptr,
                needles.as_ptr().cast::<c_void>(),
                needle_bytes,
            )
        })
        .map_err(drain_err)?;
        check_cuda(unsafe { cu_memset_d8(status_guard.ptr, 0, status_bytes) })
            .map_err(drain_err)?;
    }

    if timed {
        check_cuda(unsafe { (primary.cu_event_record)(pooled.start_event, stream) })
            .map_err(drain_err)?;
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
        check_cuda(unsafe { (primary.cu_event_record)(pooled.stop_event, stream) })
            .map_err(drain_err)?;
    }

    Ok(CudaI32IndexProbeDenseSubmission {
        projection_count: projection_offsets.len(),
        needles_len: needles.len(),
        primary,
        values_guard,
        status_guard,
        _needles_guard: needles_guard,
        stream: Some(stream_owned),
        timed,
        _wave_index_guard: Some(Arc::clone(index)),
        _multi_shard_index_guards: Vec::new(),
        _multi_shard_desc_guard: None,
        multi_shard_binary_mode: false,
    })
}

/// Sub-slice 8 v2: one shard's inputs for the MULTI-SHARD dense-emit probe. The kernel reads these from a
/// device-resident descriptor array (88 bytes/shard) so ONE kernel launch probes ALL shards per needle,
/// emits ONE dense slot (1xN output, no S*N DtoH, no host merge).
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

/// Sub-slice 8 v2 (charter-faithful): the MULTI-SHARD dense-emit point-lookup probe. ONE kernel launch where
/// each thread (needle) loops over ALL `shards` reading a device DESCRIPTOR ARRAY, probes each shard's device
/// hash index, and on the FIRST hit gathers the projected columns from that shard's buffer + dense-emits ONE
/// slot + status; a needle in no shard emits status=2. This replaces the per-shard-kernel + HOST-merge (v1):
/// the DtoH is `needle_count*projection_count` (NOT `shards*needle_count`), and the merge is eliminated
/// (the kernel writes needle-indexed output directly). All shards must share `projection_count` (1..=4).
pub(super) fn submit_cuda_resident_i32_multi_shard_index_probe_dense(
    ctx: &CudaResidentDeviceMemory,
    shards: &[MultiShardProbeShard],
    needles: &[i32],
    read_snapshot: u64,
) -> Result<CudaI32IndexProbeDenseSubmission, CudaRuntimeProbeError> {
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
    const DESC_U64_PER_SHARD: usize = 11; // resident, index, mask|shift, proj0..3, zone, created, deleted, rows
                                          // PTX: outer SHARD loop over an 88 B/shard descriptor array.
                                          // gather. Each thread probes shard 0, 1, ... until a FOUND (emit slot + status=1) or all miss (status=2).
                                          // The hash/probe/gather is byte-identical to `gpu_db_resident_i32_index_probe_dense`; only the shard loop +
                                          // per-shard descriptor reads are new. ASCII-only.
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
    mul.lo.u64 %rd19, %rd18, 4;
    add.u64 %rd20, %rd4, %rd19;
    mov.u32 %r17, 1;
    st.global.u32 [%rd20], %r17;

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
    mul.lo.u64 %rd19, %rd18, 4;
    add.u64 %rd20, %rd4, %rd19;
    mov.u32 %r17, 3;
    st.global.u32 [%rd20], %r17;
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
    mul.lo.u64 %rd19, %rd18, 4;
    add.u64 %rd20, %rd4, %rd19;
    mov.u32 %r17, 2;
    st.global.u32 [%rd20], %r17;

DONE:
    ret;
}
"#;

    if shards.is_empty() || needles.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let projection_count = shards[0].projection_offsets.len();
    if projection_count == 0 || projection_count > MAX_PROJECTIONS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(projection_count));
    }
    // Build the descriptor array + per-shard bounds check; all shards share projection_count.
    let mut desc: Vec<u64> = Vec::with_capacity(shards.len() * DESC_U64_PER_SHARD);
    let mut index_guards: Vec<Arc<CudaResidentDeviceMemory>> = Vec::with_capacity(shards.len() * 4);
    for shard in shards {
        if shard.projection_offsets.len() != projection_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                shard.projection_offsets.len(),
            ));
        }
        if shard.row_count == 0
            || shard.row_count > u32::MAX as u64
            || shard.index.device_ptr() == 0
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let allocated = shard.resident.metadata().allocated_bytes;
        for &byte_offset in &shard.projection_offsets {
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
        let mut offs = [0u64; MAX_PROJECTIONS];
        for (i, &o) in shard.projection_offsets.iter().enumerate() {
            offs[i] = o;
        }
        desc.extend_from_slice(&offs);
        // Slot 7: the filter column's zone map [min, max] packed as `min | (max << 32)` (the kernel skips
        // this shard for a needle out of range).
        desc.push((shard.min as u32 as u64) | ((shard.max as u32 as u64) << 32));
        for region in [&shard.created_by, &shard.deleted_by] {
            if let Some(region) = region {
                let required = shard
                    .row_count
                    .checked_mul(std::mem::size_of::<u64>() as u64)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                if required > region.metadata().allocated_bytes {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(required as usize));
                }
                desc.push(region.device_ptr());
                index_guards.push(Arc::clone(region));
            } else {
                desc.push(0);
            }
        }
        desc.push(shard.row_count);
        // Pin the shard's column buffer + its device index alive until the kernel completes.
        index_guards.push(Arc::clone(&shard.resident));
        index_guards.push(Arc::clone(&shard.index));
    }
    // O(1)-routing gate: when the shards are ASCENDING-DISJOINT (max[i-1] < min[i] for every i), each needle
    // maps to exactly one shard, so the kernel binary-searches (O(log shards)) instead of scanning all shards.
    // Requires >= 2 shards (1 shard is already O(1) linearly) AND real disjoint zone maps (a missing stat is
    // (i32::MIN, i32::MAX), which fails the check -> safe linear fallback). Shards are NOT reordered, so binary
    // mode requires table order to already be min-ascending (the clustered/ordered-insert regime); anything else
    // (descending, overlapping, unclustered) fails the check and takes the linear path. Byte-identical either way.
    let binary_mode: u32 = if shards.len() >= 2 && shards.windows(2).all(|w| w[0].max < w[1].min) {
        1
    } else {
        0
    };
    let shard_count_u32 = u32::try_from(shards.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(shards.len()))?;
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
    let status_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let needle_bytes = needles
        .len()
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let desc_bytes = desc
        .len()
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

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
    let desc_guard = primary.lease_device_buffer_owned(desc_bytes)?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function =
        primary.cached_function(c"gpu_db_resident_multi_shard_i32_index_probe_dense", &ptx)?;

    let mut desc_arg = desc_guard.ptr;
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
    let timed = !pooled.start_event.is_null() && !pooled.stop_event.is_null();

    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    // HtoD needles + descriptor array, memset status to 0 (the gap guard the completion debug_asserts).
    let async_ops = match (primary.cu_memcpy_htod_async, primary.cu_memset_d8_async) {
        (Some(htod), Some(memset)) => Some((htod, memset)),
        _ => None,
    };
    if let Some((htod_async, memset_async)) = async_ops {
        check_cuda(unsafe {
            htod_async(
                needles_guard.ptr,
                needles.as_ptr().cast::<c_void>(),
                needle_bytes,
                stream,
            )
        })
        .map_err(drain_err)?;
        check_cuda(unsafe {
            htod_async(
                desc_guard.ptr,
                desc.as_ptr().cast::<c_void>(),
                desc_bytes,
                stream,
            )
        })
        .map_err(drain_err)?;
        check_cuda(unsafe { memset_async(status_guard.ptr, 0, status_bytes, stream) })
            .map_err(drain_err)?;
    } else {
        check_cuda(unsafe {
            cu_memcpy_htod(
                needles_guard.ptr,
                needles.as_ptr().cast::<c_void>(),
                needle_bytes,
            )
        })
        .map_err(drain_err)?;
        check_cuda(unsafe {
            cu_memcpy_htod(desc_guard.ptr, desc.as_ptr().cast::<c_void>(), desc_bytes)
        })
        .map_err(drain_err)?;
        check_cuda(unsafe { cu_memset_d8(status_guard.ptr, 0, status_bytes) })
            .map_err(drain_err)?;
    }

    if timed {
        check_cuda(unsafe { (primary.cu_event_record)(pooled.start_event, stream) })
            .map_err(drain_err)?;
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
        check_cuda(unsafe { (primary.cu_event_record)(pooled.stop_event, stream) })
            .map_err(drain_err)?;
    }

    Ok(CudaI32IndexProbeDenseSubmission {
        projection_count,
        needles_len: needles.len(),
        primary,
        values_guard,
        status_guard,
        _needles_guard: needles_guard,
        stream: Some(stream_owned),
        timed,
        _wave_index_guard: None,
        _multi_shard_index_guards: index_guards,
        _multi_shard_desc_guard: Some(desc_guard),
        multi_shard_binary_mode: binary_mode == 1,
    })
}

impl CudaI32IndexProbeDenseSubmission {
    /// Drain the dense output and SEQUENTIALLY compact it (by `status`) into the SAME compacted columnar form
    /// the atomic path produces — same byte-identical rows, minus `row_indices` (dense unique has no
    /// within-needle sort, so the engine never needs it). ONE covering `cuStreamSynchronize` (the size is
    /// `needles_len`, known a priori, so there is no count round-trip — folds in lever #2). `needle_indices`
    /// comes out ASCENDING (slot order), exactly what the engine's scatter expects.
    pub fn complete_detached_columnar(
        mut self,
    ) -> Result<(CudaI32BatchProjectionColumns, Option<u64>), CudaRuntimeProbeError> {
        let primary = Arc::clone(&self.primary);
        primary.set_current()?;
        let stream_owned = self
            .stream
            .take()
            .expect("pooled stream held until complete");
        let pooled = stream_owned
            .pooled
            .as_ref()
            .expect("pooled stream held until complete");
        let stream = pooled.stream;

        // The covering sync is the visibility barrier — plain `st.global` status writes in the kernel are
        // ordered before the host reads below by this sync (NO release-acquire needed; that machinery is only
        // for the host-mapped-polling path).
        check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) })?;

        let elapsed_us = if self.timed {
            let mut elapsed_ms = 0.0_f32;
            check_cuda(unsafe {
                (primary.cu_event_elapsed_time)(
                    &mut elapsed_ms,
                    pooled.start_event,
                    pooled.stop_event,
                )
            })?;
            Some((f64::from(elapsed_ms) * 1_000.0).ceil() as u64)
        } else {
            None
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
        let mut status = vec![0_u32; nc];
        if !values_raw.is_empty() {
            check_cuda(unsafe {
                (primary.cu_memcpy_dtoh)(
                    values_raw.as_mut_ptr().cast::<c_void>(),
                    self.values_guard.ptr,
                    values_raw.len() * std::mem::size_of::<i32>(),
                )
            })
            .map_err(drain_err)?;
        }
        check_cuda(unsafe {
            (primary.cu_memcpy_dtoh)(
                status.as_mut_ptr().cast::<c_void>(),
                self.status_guard.ptr,
                status.len() * std::mem::size_of::<u32>(),
            )
        })
        .map_err(drain_err)?;

        // Return the DENSE LAYOUT as-is (NO compaction here): `values_raw` is one slot per needle (gaps) +
        // `status`. The engine's `assemble_batched_rows` compacts it in ONE sequential pass — compacting here
        // AND letting the engine re-scatter would be two passes (measured slower than the atomic scatter). The
        // gap guard (`status != 0`) lives at the compaction site.
        let columns = CudaI32BatchProjectionColumns {
            values: values_raw,
            needle_indices: Vec::new(),
            row_indices: Vec::new(),
            projection_count: proj,
            status,
        };
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
