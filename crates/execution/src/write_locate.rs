use std::os::raw::c_void;
use std::sync::Arc;

use super::resident_memory::CudaResidentReadSource;
use super::{CudaResidentDeviceMemory, CudaRuntimeProbeError, check_cuda};

/// M1 (charter-pure device WRITE-LOCATE): one shard's DEVICE hash index + its packing params. The
/// write path (A2 DML resolve, A3 validators) probes these ON THE DEVICE — replacing the host
/// `shard_pk_index` hash probe (the charter ruling, 2026-07-03: key->slot ADDRESSING is device work).
pub struct WriteLocateShard {
    /// The shard's DEVICE hash index (`(key<<32)|(row+1)`, 0 = empty), pinned until the kernel completes.
    pub index: Arc<CudaResidentDeviceMemory>,
    pub table_mask: u32,
    pub hash_shift: u32,
}

/// Per-needle device-locate result: a fixed `max_hits` window of `(shard_idx, slot)` pairs +
/// a per-needle count. `count[n] == u32::MAX` = OVERFLOW (more than `max_hits` shards held the key —
/// the host declines that needle to the scan, exactly like a cross-shard-dup host decline).
pub struct WriteLocateResult {
    pub max_hits: u32,
    /// `needle_count * max_hits`: the shard index (into the passed `shards` slice) of each hit.
    pub shard_idx: Vec<u32>,
    /// `needle_count * max_hits`: the LOCAL slot of each hit within its shard.
    pub slot: Vec<u32>,
    /// `needle_count`: hits found (u32::MAX = overflow).
    pub count: Vec<u32>,
}

/// U1 (lane DELETE intents): one shard's device hash index + its VERSION regions for the
/// VISIBLE-LOCATE kernel — visibility (`created_by <= snap && deleted_by > snap`) is evaluated
/// ON THE DEVICE per hit, so the host never rechecks. A zero region pointer means the region was
/// never allocated: `created_by == 0` = born-visible (fill 0x00), `deleted_by == 0` = all-live
/// (fill 0x7F). The caller must PIN the regions (hold their Arcs) across the submit call — this
/// struct carries raw pointers for the descriptor image.
pub struct VisibleLocateShard {
    /// The shard's DEVICE hash index (`(key<<32)|(row+1)`, 0 = empty), pinned until the kernel completes.
    pub index: Arc<CudaResidentDeviceMemory>,
    pub table_mask: u32,
    pub hash_shift: u32,
    /// `created_by[row]` u64 region base, or 0 (absent = every row born-visible).
    pub created_by_ptr: u64,
    /// `deleted_by[row]` u64 region base, or 0 (absent = every row live).
    pub deleted_by_ptr: u64,
}

/// Per-needle VISIBLE-locate result: the count of VISIBLE matches at the needle's snapshot
/// (dead twins and not-yet-visible versions are skipped on-device) and the (shard_idx, slot) of
/// the FIRST visible match (meaningful iff `count >= 1`). For a unique key, `count == 1` is the
/// tombstone target, `count == 0` is a clean miss (0 rows affected / no duplicate), and
/// `count > 1` is an ambiguity the caller must decline (uniqueness invariant violation net).
pub struct VisibleLocateResult {
    pub shard_idx: Vec<u32>,
    pub slot: Vec<u32>,
    pub count: Vec<u32>,
}

/// The VISIBLE-LOCATE kernel (U1): write-locate's probe loop + ON-DEVICE MVCC visibility. Each
/// thread (needle) loops ALL shards; on a key match it loads the row's `created_by`/`deleted_by`
/// stamps and counts the match only if `created_by <= snap && deleted_by > snap` (unsigned u64 —
/// the live fill 0x7F7F.. and visible fill 0x0 are chosen for exactly these compares). Unlike
/// write-locate it ADVANCES PAST every match (visible or not): dup-tolerant indexes hold a dead
/// twin and its live reinsert in separate slots of the SAME shard. ASCII-only PTX.
const VISIBLE_LOCATE_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_multi_shard_i32_visible_locate(
    .param .u64 desc_array_ptr,
    .param .u32 shard_count,
    .param .u32 needle_count,
    .param .u64 needles_ptr,
    .param .u64 snapshots_ptr,
    .param .u64 out_shard_ptr,
    .param .u64 out_slot_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<32>;

    ld.param.u64 %rd1, [desc_array_ptr];
    ld.param.u32 %r1, [shard_count];
    ld.param.u32 %r2, [needle_count];
    ld.param.u64 %rd2, [needles_ptr];
    ld.param.u64 %rd3, [snapshots_ptr];
    ld.param.u64 %rd4, [out_shard_ptr];
    ld.param.u64 %rd5, [out_slot_ptr];
    ld.param.u64 %rd6, [out_count_ptr];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;

    mul.wide.u32 %rd7, %r6, 4;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.s32 %r7, [%rd8];
    mul.wide.u32 %rd9, %r6, 8;
    add.u64 %rd10, %rd3, %rd9;
    ld.global.u64 %rd11, [%rd10];

    mov.u32 %r8, 0;
    mov.u32 %r9, 0;
    mov.u32 %r10, 0;
    mov.u32 %r11, 0;

SHARD:
    setp.ge.u32 %p1, %r11, %r1;
    @%p1 bra WRITEOUT;
    mul.wide.u32 %rd12, %r11, 32;
    add.u64 %rd13, %rd1, %rd12;
    ld.global.u64 %rd14, [%rd13];
    ld.global.u64 %rd15, [%rd13+8];
    cvt.u32.u64 %r12, %rd15;
    shr.u64 %rd16, %rd15, 32;
    cvt.u32.u64 %r13, %rd16;
    ld.global.u64 %rd17, [%rd13+16];
    ld.global.u64 %rd18, [%rd13+24];

    mul.lo.u32 %r14, %r7, 2654435761;
    shr.u32 %r15, %r14, %r13;
    and.b32 %r15, %r15, %r12;
    mov.u32 %r16, 0;

PROBE:
    mul.wide.u32 %rd19, %r15, 8;
    add.u64 %rd20, %rd14, %rd19;
    ld.global.u64 %rd21, [%rd20];
    setp.eq.u64 %p2, %rd21, 0;
    @%p2 bra NEXTSHARD;
    shr.u64 %rd22, %rd21, 32;
    cvt.u32.u64 %r17, %rd22;
    setp.ne.s32 %p2, %r17, %r7;
    @%p2 bra ADVANCE;
    cvt.u32.u64 %r18, %rd21;
    sub.u32 %r18, %r18, 1;
    setp.eq.u64 %p3, %rd17, 0;
    @%p3 bra CBOK;
    mul.wide.u32 %rd23, %r18, 8;
    add.u64 %rd24, %rd17, %rd23;
    ld.global.u64 %rd25, [%rd24];
    setp.gt.u64 %p4, %rd25, %rd11;
    @%p4 bra ADVANCE;
CBOK:
    setp.eq.u64 %p5, %rd18, 0;
    @%p5 bra VISHIT;
    mul.wide.u32 %rd26, %r18, 8;
    add.u64 %rd27, %rd18, %rd26;
    ld.global.u64 %rd28, [%rd27];
    setp.le.u64 %p6, %rd28, %rd11;
    @%p6 bra ADVANCE;
VISHIT:
    setp.ne.u32 %p7, %r8, 0;
    @%p7 bra VCOUNT;
    mov.u32 %r9, %r11;
    mov.u32 %r10, %r18;
VCOUNT:
    add.u32 %r8, %r8, 1;
ADVANCE:
    add.u32 %r15, %r15, 1;
    and.b32 %r15, %r15, %r12;
    add.u32 %r16, %r16, 1;
    setp.ge.u32 %p2, %r16, 256;
    @%p2 bra NEXTSHARD;
    bra PROBE;

NEXTSHARD:
    add.u32 %r11, %r11, 1;
    bra SHARD;

WRITEOUT:
    mul.wide.u32 %rd29, %r6, 4;
    add.u64 %rd30, %rd4, %rd29;
    st.global.u32 [%rd30], %r9;
    add.u64 %rd31, %rd5, %rd29;
    st.global.u32 [%rd31], %r10;
    add.u64 %rd7, %rd6, %rd29;
    st.global.u32 [%rd7], %r8;

DONE:
    ret;
}
"#;

/// The WRITE-LOCATE kernel: each thread (needle) loops ALL shards, probes each shard's device hash
/// index (fib-hash + linear probe, 256 cap — BYTE-IDENTICAL to the read probe + the host
/// `build_int4_pk_hash_table_host`), and emits EVERY hit (shard_idx, local slot) into the needle's
/// fixed window. Unlike the read kernel (first-hit + decline-on-dup), the write path needs ALL hits
/// (an SV5 update-append puts a key in two shards: tombstoned-old + new). NO bloom/zone prune here —
/// those only skip shards a hit can't be in, so the hash probe is authoritative and the hits are
/// identical to the host probe's. ASCII-only PTX.
const WRITE_LOCATE_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_multi_shard_i32_write_locate(
    .param .u64 desc_array_ptr,
    .param .u32 shard_count,
    .param .u32 needle_count,
    .param .u64 needles_ptr,
    .param .u32 max_hits,
    .param .u64 out_shard_ptr,
    .param .u64 out_slot_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;

    ld.param.u64 %rd1, [desc_array_ptr];
    ld.param.u32 %r1, [shard_count];
    ld.param.u32 %r2, [needle_count];
    ld.param.u64 %rd2, [needles_ptr];
    ld.param.u32 %r3, [max_hits];
    ld.param.u64 %rd3, [out_shard_ptr];
    ld.param.u64 %rd4, [out_slot_ptr];
    ld.param.u64 %rd5, [out_count_ptr];

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mad.lo.u32 %r7, %r5, %r6, %r4;
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra DONE;

    mul.wide.u32 %rd6, %r7, 4;
    add.u64 %rd7, %rd2, %rd6;
    ld.global.s32 %r8, [%rd7];

    mov.u32 %r9, 0;
    mov.u32 %r10, 0;

SHARD:
    setp.ge.u32 %p1, %r10, %r1;
    @%p1 bra WRITECOUNT;
    mul.wide.u32 %rd8, %r10, 16;
    add.u64 %rd9, %rd1, %rd8;
    ld.global.u64 %rd10, [%rd9];
    ld.global.u64 %rd11, [%rd9+8];
    cvt.u32.u64 %r11, %rd11;
    shr.u64 %rd12, %rd11, 32;
    cvt.u32.u64 %r12, %rd12;

    mul.lo.u32 %r13, %r8, 2654435761;
    shr.u32 %r14, %r13, %r12;
    and.b32 %r14, %r14, %r11;
    mov.u32 %r15, 0;

PROBE:
    mul.wide.u32 %rd13, %r14, 8;
    add.u64 %rd14, %rd10, %rd13;
    ld.global.u64 %rd15, [%rd14];
    setp.eq.u64 %p2, %rd15, 0;
    @%p2 bra NEXTSHARD;
    shr.u64 %rd16, %rd15, 32;
    cvt.u32.u64 %r16, %rd16;
    setp.eq.s32 %p2, %r16, %r8;
    @%p2 bra FOUND;
    add.u32 %r14, %r14, 1;
    and.b32 %r14, %r14, %r11;
    add.u32 %r15, %r15, 1;
    setp.ge.u32 %p3, %r15, 256;
    @%p3 bra NEXTSHARD;
    bra PROBE;

FOUND:
    cvt.u32.u64 %r17, %rd15;
    sub.u32 %r17, %r17, 1;
    setp.ge.u32 %p4, %r9, %r3;
    @%p4 bra INCCOUNT;
    mad.lo.u32 %r18, %r7, %r3, %r9;
    mul.wide.u32 %rd18, %r18, 4;
    add.u64 %rd19, %rd3, %rd18;
    st.global.u32 [%rd19], %r10;
    add.u64 %rd20, %rd4, %rd18;
    st.global.u32 [%rd20], %r17;
INCCOUNT:
    add.u32 %r9, %r9, 1;
    // F3/U4: ADVANCE past this match and keep probing THIS shard for MVCC version twins (the
    // dup-tolerant index now holds an updated key's old + new physical rows in the same shard).
    // Only an empty slot / 256-cap ends the shard. The caller resolves visibility across the
    // returned hits exactly as it already does for the cross-shard (old-in-A, new-in-B) case.
    add.u32 %r14, %r14, 1;
    and.b32 %r14, %r14, %r11;
    add.u32 %r15, %r15, 1;
    setp.ge.u32 %p3, %r15, 256;
    @%p3 bra NEXTSHARD;
    bra PROBE;
NEXTSHARD:
    add.u32 %r10, %r10, 1;
    bra SHARD;

WRITECOUNT:
    mul.wide.u32 %rd21, %r7, 4;
    add.u64 %rd22, %rd5, %rd21;
    setp.gt.u32 %p5, %r9, %r3;
    @%p5 bra WOVER;
    st.global.u32 [%rd22], %r9;
    bra DONE;
WOVER:
    mov.u32 %r19, 4294967295;
    st.global.u32 [%rd22], %r19;

DONE:
    ret;
}
"#;

impl CudaResidentDeviceMemory {
    /// M1 (charter-pure): probe a BATCH of int4 `needles` against ALL `shards`' DEVICE hash indexes in
    /// ONE kernel launch, emitting each needle's `(shard_idx, slot)` hits. Replaces the host
    /// `shard_pk_index` hash probe for the write path (A2/A3). Synchronous (null stream + context sync):
    /// the write path's batches are small (one wave), so the launch+readback round-trip is the design,
    /// not a throughput bottleneck. `self` is only the allocation/launch context (any device buffer on
    /// the GPU works). `max_hits` bounds the per-needle window (2 suffices for SV5 old+new; overflow
    /// declines).
    pub fn submit_multi_shard_i32_write_locate(
        &self,
        shards: &[WriteLocateShard],
        needles: &[i32],
        max_hits: u32,
    ) -> Result<WriteLocateResult, CudaRuntimeProbeError> {
        type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
        type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
        type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
        #[allow(clippy::type_complexity)]
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

        if shards.is_empty() || needles.is_empty() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        // COUNT-ONLY fast path (`max_hits == 0`): the kernel emits only per-needle counts (any hit
        // -> u32::MAX), NO shard/slot output — so wave-batch validation skips 2 device buffers +
        // 2 DtoH reads per wave (the shard/slot outputs it never consumes). The FOUND arm takes
        // INCCOUNT (count >= max_hits == 0 always), so the shard/slot pointers are never
        // dereferenced; a valid dummy (the count buffer) is passed for them.
        let count_only = max_hits == 0;
        const DESC_U64_PER_SHARD: usize = 2; // index_ptr, mask|shift
        let mut desc: Vec<u64> = Vec::with_capacity(shards.len() * DESC_U64_PER_SHARD);
        let mut index_guards: Vec<Arc<CudaResidentDeviceMemory>> = Vec::with_capacity(shards.len());
        for shard in shards {
            if shard.index.device_ptr() == 0 {
                return Err(CudaRuntimeProbeError::InvalidInputLength(0));
            }
            desc.push(shard.index.device_ptr());
            desc.push((shard.table_mask as u64) | ((shard.hash_shift as u64) << 32));
            index_guards.push(Arc::clone(&shard.index));
        }
        let shard_count_u32 = u32::try_from(shards.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(shards.len()))?;
        let needle_count_u32 = u32::try_from(needles.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;

        let window = (needles.len() as u64)
            .checked_mul(max_hits as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let window_bytes = usize::try_from(
            window
                .checked_mul(std::mem::size_of::<u32>() as u64)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        )
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let count_bytes = needles
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

        let primary = self.primary_arc();
        primary.set_current()?;
        let cu_memset_d8 = unsafe {
            primary
                .lib()
                .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
                .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_memcpy_htod = unsafe {
            primary
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_memcpy_dtoh = unsafe {
            primary
                .lib()
                .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_launch_kernel = unsafe {
            primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };

        let needles_guard = primary.lease_device_buffer_owned(needle_bytes)?;
        let count_guard = primary.lease_device_buffer_owned(count_bytes)?;
        let desc_guard = primary.lease_device_buffer_owned(desc_bytes)?;
        // Count-only skips the shard/slot buffers (unused); their args point at the count buffer
        // (never written — the kernel takes INCCOUNT for every hit at max_hits==0).
        let shard_out_guard = if count_only {
            None
        } else {
            Some(primary.lease_device_buffer_owned(window_bytes)?)
        };
        let slot_out_guard = if count_only {
            None
        } else {
            Some(primary.lease_device_buffer_owned(window_bytes)?)
        };
        let shard_out_ptr = shard_out_guard.as_ref().map_or(count_guard.ptr, |g| g.ptr);
        let slot_out_ptr = slot_out_guard.as_ref().map_or(count_guard.ptr, |g| g.ptr);

        let mut ptx = Vec::with_capacity(WRITE_LOCATE_PTX.len() + 1);
        ptx.extend_from_slice(WRITE_LOCATE_PTX);
        ptx.push(0);
        let function =
            primary.cached_function(c"gpu_db_resident_multi_shard_i32_write_locate", &ptx)?;

        // HtoD needles + descriptor (the count buffer's zero-init was dropped — see below).
        check_cuda(unsafe {
            cu_memcpy_htod(
                needles_guard.ptr,
                needles.as_ptr().cast::<c_void>(),
                needle_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_htod(desc_guard.ptr, desc.as_ptr().cast::<c_void>(), desc_bytes)
        })?;
        // FUSE (driver-call trim): the count-buffer zero-init is DEAD — every needle-thread
        // (tid < needle_count) stores its own final count at WRITECOUNT/overflow, so no slot is
        // ever read stale. (Threads tid >= needle_count early-return but own no count slot.)
        // Dropping the memset removes one driver call per wave; `_` binds the unused fn ptr.
        let _ = cu_memset_d8;

        let mut desc_arg = desc_guard.ptr;
        let mut shard_count_arg = shard_count_u32;
        let mut needle_count_arg = needle_count_u32;
        let mut needles_arg = needles_guard.ptr;
        let mut max_hits_arg = max_hits;
        let mut shard_out_arg = shard_out_ptr;
        let mut slot_out_arg = slot_out_ptr;
        let mut count_arg = count_guard.ptr;
        let mut args = [
            (&mut desc_arg as *mut u64).cast::<c_void>(),
            (&mut shard_count_arg as *mut u32).cast::<c_void>(),
            (&mut needle_count_arg as *mut u32).cast::<c_void>(),
            (&mut needles_arg as *mut u64).cast::<c_void>(),
            (&mut max_hits_arg as *mut u32).cast::<c_void>(),
            (&mut shard_out_arg as *mut u64).cast::<c_void>(),
            (&mut slot_out_arg as *mut u64).cast::<c_void>(),
            (&mut count_arg as *mut u64).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = needle_count_u32.div_ceil(threads_per_block);
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
        // FUSE (driver-call trim): the explicit stream-synchronize is REDUNDANT — the count DtoH
        // below (and, on the full path, the shard/slot DtoH) is a BLOCKING `cuMemcpyDtoH` on the
        // null stream, which already waits for this kernel before copying. Dropping the sync
        // removes one driver call per wave with zero semantic change (the readback still blocks).
        // FENCE INVARIANT (do not break): this only fences the kernel — so the pooled guards below
        // are not recycled mid-flight — because (a) the kernel launches on the NULL stream and (b) a
        // NON-ZERO-length blocking DtoH runs on that same null stream on EVERY path before return.
        // If you move the launch to a pooled/non-blocking stream or make the readback conditional,
        // restore an explicit synchronize.
        let mut shard_idx = vec![0u32; window as usize];
        let mut slot = vec![0u32; window as usize];
        let mut count = vec![0u32; needles.len()];
        if let (Some(sg), Some(lg)) = (shard_out_guard.as_ref(), slot_out_guard.as_ref()) {
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    shard_idx.as_mut_ptr().cast::<c_void>(),
                    sg.ptr,
                    window_bytes,
                )
            })?;
            check_cuda(unsafe {
                cu_memcpy_dtoh(slot.as_mut_ptr().cast::<c_void>(), lg.ptr, window_bytes)
            })?;
        }
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                count.as_mut_ptr().cast::<c_void>(),
                count_guard.ptr,
                count_bytes,
            )
        })?;
        // The guards (device buffers + pinned indexes) drop here — after the sync, so the kernel is done.
        drop(index_guards);
        Ok(WriteLocateResult {
            max_hits,
            shard_idx,
            slot,
            count,
        })
    }

    /// U1 VISIBLE-LOCATE submit (see [`VISIBLE_LOCATE_PTX`]): one coalesced launch resolving
    /// every needle to its VISIBLE match count + first visible (shard_idx, slot) at the needle's
    /// OWN snapshot. `needles` and `snapshots` are parallel. The caller must hold the version
    /// regions' Arcs across this call (the descriptor carries raw pointers).
    pub fn submit_multi_shard_i32_visible_locate(
        &self,
        shards: &[VisibleLocateShard],
        needles: &[i32],
        snapshots: &[u64],
    ) -> Result<VisibleLocateResult, CudaRuntimeProbeError> {
        type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
        type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
        #[allow(clippy::type_complexity)]
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

        if shards.is_empty() || needles.is_empty() || needles.len() != snapshots.len() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(needles.len()));
        }
        const DESC_U64_PER_SHARD: usize = 4; // index_ptr, mask|shift, created_by, deleted_by
        let mut desc: Vec<u64> = Vec::with_capacity(shards.len() * DESC_U64_PER_SHARD);
        let mut index_guards: Vec<Arc<CudaResidentDeviceMemory>> = Vec::with_capacity(shards.len());
        for shard in shards {
            if shard.index.device_ptr() == 0 {
                return Err(CudaRuntimeProbeError::InvalidInputLength(0));
            }
            desc.push(shard.index.device_ptr());
            desc.push((shard.table_mask as u64) | ((shard.hash_shift as u64) << 32));
            desc.push(shard.created_by_ptr);
            desc.push(shard.deleted_by_ptr);
            index_guards.push(Arc::clone(&shard.index));
        }
        let shard_count_u32 = u32::try_from(shards.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(shards.len()))?;
        let needle_count_u32 = u32::try_from(needles.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;

        let out_bytes = needles
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let needle_bytes = std::mem::size_of_val(needles);
        let snapshot_bytes = std::mem::size_of_val(snapshots);
        let desc_bytes = std::mem::size_of_val(desc.as_slice());

        let primary = self.primary_arc();
        primary.set_current()?;
        let cu_memcpy_htod = unsafe {
            primary
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_memcpy_dtoh = unsafe {
            primary
                .lib()
                .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_launch_kernel = unsafe {
            primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };

        let needles_guard = primary.lease_device_buffer_owned(needle_bytes)?;
        let snapshots_guard = primary.lease_device_buffer_owned(snapshot_bytes)?;
        let desc_guard = primary.lease_device_buffer_owned(desc_bytes)?;
        let shard_out_guard = primary.lease_device_buffer_owned(out_bytes)?;
        let slot_out_guard = primary.lease_device_buffer_owned(out_bytes)?;
        let count_guard = primary.lease_device_buffer_owned(out_bytes)?;

        let mut ptx = Vec::with_capacity(VISIBLE_LOCATE_PTX.len() + 1);
        ptx.extend_from_slice(VISIBLE_LOCATE_PTX);
        ptx.push(0);
        let function =
            primary.cached_function(c"gpu_db_resident_multi_shard_i32_visible_locate", &ptx)?;

        check_cuda(unsafe {
            cu_memcpy_htod(
                needles_guard.ptr,
                needles.as_ptr().cast::<c_void>(),
                needle_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_htod(
                snapshots_guard.ptr,
                snapshots.as_ptr().cast::<c_void>(),
                snapshot_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_htod(desc_guard.ptr, desc.as_ptr().cast::<c_void>(), desc_bytes)
        })?;
        // No output memset: every needle-thread stores its own shard/slot/count at WRITEOUT.

        let mut desc_arg = desc_guard.ptr;
        let mut shard_count_arg = shard_count_u32;
        let mut needle_count_arg = needle_count_u32;
        let mut needles_arg = needles_guard.ptr;
        let mut snapshots_arg = snapshots_guard.ptr;
        let mut shard_out_arg = shard_out_guard.ptr;
        let mut slot_out_arg = slot_out_guard.ptr;
        let mut count_arg = count_guard.ptr;
        let mut args = [
            (&mut desc_arg as *mut u64).cast::<c_void>(),
            (&mut shard_count_arg as *mut u32).cast::<c_void>(),
            (&mut needle_count_arg as *mut u32).cast::<c_void>(),
            (&mut needles_arg as *mut u64).cast::<c_void>(),
            (&mut snapshots_arg as *mut u64).cast::<c_void>(),
            (&mut shard_out_arg as *mut u64).cast::<c_void>(),
            (&mut slot_out_arg as *mut u64).cast::<c_void>(),
            (&mut count_arg as *mut u64).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = needle_count_u32.div_ceil(threads_per_block);
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
        // Same FENCE INVARIANT as write-locate: null-stream launch + blocking non-zero DtoH
        // below fences the kernel before the pooled guards recycle.
        let mut shard_idx = vec![0u32; needles.len()];
        let mut slot = vec![0u32; needles.len()];
        let mut count = vec![0u32; needles.len()];
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                shard_idx.as_mut_ptr().cast::<c_void>(),
                shard_out_guard.ptr,
                out_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                slot.as_mut_ptr().cast::<c_void>(),
                slot_out_guard.ptr,
                out_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                count.as_mut_ptr().cast::<c_void>(),
                count_guard.ptr,
                out_bytes,
            )
        })?;
        drop(index_guards);
        Ok(VisibleLocateResult {
            shard_idx,
            slot,
            count,
        })
    }
}
