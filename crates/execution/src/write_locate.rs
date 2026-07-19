use std::os::raw::c_void;
use std::sync::Arc;

use super::resident_memory::CudaResidentReadSource;
use super::{check_cuda, CudaResidentDeviceMemory, CudaRuntimeProbeError};

/// M1 (charter-pure device WRITE-LOCATE): one shard's DEVICE hash index + its packing params. The
/// write path (A2 DML resolve, A3 validators) probes these ON THE DEVICE — replacing the host
/// `shard_pk_index` hash probe (the charter ruling, 2026-07-03: key->slot ADDRESSING is device work).
pub struct WriteLocateShard {
    /// The shard's DEVICE hash index (`(key<<32)|(row+1)`, 0 = empty), pinned until the kernel completes.
    pub index: Arc<CudaResidentDeviceMemory>,
    pub table_mask: u32,
    pub hash_shift: u32,
    /// Logical number of rows whose packed slots are valid in this index.
    pub row_count: u32,
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
/// ON THE DEVICE per hit, so the host never rechecks. An absent region means it was never allocated:
/// absent `created_by` = born-visible and absent `deleted_by` = all-live. Owners remain pinned by
/// this descriptor through the synchronous launch.
pub struct VisibleLocateShard {
    /// The shard's DEVICE hash index (`(key<<32)|(row+1)`, 0 = empty), pinned until the kernel completes.
    pub index: Arc<CudaResidentDeviceMemory>,
    pub table_mask: u32,
    pub hash_shift: u32,
    pub row_count: u32,
    /// `created_by[row]` u64 region owner; absent means every row was born visible.
    pub created_by: Option<Arc<CudaResidentDeviceMemory>>,
    /// `deleted_by[row]` u64 region owner; absent means every row remains live.
    pub deleted_by: Option<Arc<CudaResidentDeviceMemory>>,
    /// Stable logical entity identity for each physical version slot. An absent region makes the
    /// identity result `u64::MAX`; mutation callers must decline rather than invent an identity.
    pub row_id: Option<Arc<CudaResidentDeviceMemory>>,
}

fn validate_index_geometry(
    allocated_bytes: u64,
    table_mask: u32,
    hash_shift: u32,
    row_count: u32,
) -> Result<(), CudaRuntimeProbeError> {
    let table_slots = u64::from(table_mask)
        .checked_add(1)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if table_slots < 2 || !table_slots.is_power_of_two() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            table_mask as usize,
        ));
    }
    let expected_shift = 32 - table_slots.trailing_zeros();
    let required_bytes = crate::resident_index_allocated_bytes(table_mask, u64::from(row_count))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if hash_shift != expected_shift || required_bytes > allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(required_bytes).unwrap_or(usize::MAX),
        ));
    }
    Ok(())
}

fn validate_version_region(
    region: &CudaResidentDeviceMemory,
    row_count: u32,
) -> Result<(), CudaRuntimeProbeError> {
    let required_bytes = u64::from(row_count)
        .checked_mul(std::mem::size_of::<u64>() as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if required_bytes > region.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(required_bytes).unwrap_or(usize::MAX),
        ));
    }
    Ok(())
}

type CuStreamSync = unsafe extern "C" fn(*mut c_void) -> i32;

struct DefaultStreamDrain {
    sync: CuStreamSync,
    armed: bool,
}

impl Drop for DefaultStreamDrain {
    fn drop(&mut self) {
        if self.armed {
            unsafe { (self.sync)(std::ptr::null_mut()) };
        }
    }
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
    /// Stable entity identity of the first visible match, or `u64::MAX` when no identity region
    /// was supplied. Parallel to `count`.
    pub row_id: Vec<u64>,
    /// Highest real `created_by` or `deleted_by` stamp among every physical version matching the
    /// needle. The all-live deleted sentinel is ignored. This is independent of the requested
    /// visibility snapshot and lets writers detect a claim/release history newer than their read
    /// snapshot, including a key that was inserted and deleted again before validation.
    pub latest_write: Vec<u64>,
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
    .param .u64 out_count_ptr,
    .param .u64 out_row_id_ptr,
    .param .u64 out_latest_write_ptr
)
{
    .reg .pred %p<13>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<48>;

    ld.param.u64 %rd1, [desc_array_ptr];
    ld.param.u32 %r1, [shard_count];
    ld.param.u32 %r2, [needle_count];
    ld.param.u64 %rd2, [needles_ptr];
    ld.param.u64 %rd3, [snapshots_ptr];
    ld.param.u64 %rd4, [out_shard_ptr];
    ld.param.u64 %rd5, [out_slot_ptr];
    ld.param.u64 %rd6, [out_count_ptr];
    ld.param.u64 %rd7, [out_row_id_ptr];
    ld.param.u64 %rd39, [out_latest_write_ptr];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;

    mul.wide.u32 %rd8, %r6, 4;
    add.u64 %rd9, %rd2, %rd8;
    ld.global.s32 %r7, [%rd9];
    mul.wide.u32 %rd10, %r6, 8;
    add.u64 %rd11, %rd3, %rd10;
    ld.global.u64 %rd12, [%rd11];

    mov.u32 %r8, 0;
    mov.u32 %r9, 0;
    mov.u32 %r10, 0;
    mov.u32 %r11, 0;
    mov.u64 %rd30, 18446744073709551615;
    mov.u64 %rd40, 0;

SHARD:
    setp.ge.u32 %p1, %r11, %r1;
    @%p1 bra WRITEOUT;
    mul.wide.u32 %rd13, %r11, 48;
    add.u64 %rd14, %rd1, %rd13;
    ld.global.u64 %rd15, [%rd14];
    ld.global.u64 %rd16, [%rd14+8];
    cvt.u32.u64 %r12, %rd16;
    shr.u64 %rd17, %rd16, 32;
    cvt.u32.u64 %r13, %rd17;
    ld.global.u64 %rd18, [%rd14+16];
    ld.global.u64 %rd19, [%rd14+24];
    ld.global.u64 %rd20, [%rd14+32];
    ld.global.u32 %r19, [%rd14+40];
    add.u32 %r21, %r12, 1;
    cvt.u64.u32 %rd42, %r21;
    shl.b64 %rd42, %rd42, 3;
    add.u64 %rd42, %rd15, %rd42;

    mul.lo.u32 %r14, %r7, 2654435761;
    shr.u32 %r15, %r14, %r13;
    and.b32 %r15, %r15, %r12;
    mov.u32 %r16, 0;

PROBE:
    mul.wide.u32 %rd21, %r15, 8;
    add.u64 %rd22, %rd15, %rd21;
    ld.global.u64 %rd23, [%rd22];
    setp.eq.u64 %p2, %rd23, 0;
    @%p2 bra NEXTSHARD;
    shr.u64 %rd24, %rd23, 32;
    cvt.u32.u64 %r17, %rd24;
    setp.ne.s32 %p2, %r17, %r7;
    @%p2 bra ADVANCE;
    cvt.u32.u64 %r18, %rd23;
    // Bit 31 marks a singleton directory head: inspect it, then skip the dependent link load.
    setp.lt.s32 %p12, %r18, 0;
    and.b32 %r18, %r18, 2147483647;
    mov.u32 %r22, 0;
CHAIN:
    setp.eq.u32 %p2, %r18, 0;
    @%p2 bra NEXTSHARD;
    setp.ge.u32 %p8, %r22, %r19;
    @%p8 bra BADINDEX;
    add.u32 %r22, %r22, 1;
    sub.u32 %r18, %r18, 1;
    setp.ge.u32 %p8, %r18, %r19;
    @%p8 bra BADINDEX;

    mov.u64 %rd27, 0;
    setp.eq.u64 %p3, %rd18, 0;
    @%p3 bra CBLOADED;
    mul.wide.u32 %rd25, %r18, 8;
    add.u64 %rd26, %rd18, %rd25;
    ld.global.u64 %rd27, [%rd26];
CBLOADED:
    setp.gt.u64 %p10, %rd27, %rd40;
    @%p10 mov.u64 %rd40, %rd27;

    mov.u64 %rd29, 9187201950435737471;
    setp.eq.u64 %p5, %rd19, 0;
    @%p5 bra DBLOADED;
    mul.wide.u32 %rd25, %r18, 8;
    add.u64 %rd28, %rd19, %rd25;
    ld.global.u64 %rd29, [%rd28];
DBLOADED:
    setp.eq.u64 %p10, %rd29, 9187201950435737471;
    @%p10 bra HISTORYDONE;
    setp.gt.u64 %p11, %rd29, %rd40;
    @%p11 mov.u64 %rd40, %rd29;
HISTORYDONE:

    setp.gt.u64 %p4, %rd27, %rd12;
    @%p4 bra CHAINNEXT;
    setp.le.u64 %p6, %rd29, %rd12;
    @%p6 bra CHAINNEXT;
VISHIT:
    setp.ne.u32 %p7, %r8, 0;
    @%p7 bra VCOUNT;
    mov.u32 %r9, %r11;
    mov.u32 %r10, %r18;
    setp.eq.u64 %p9, %rd20, 0;
    @%p9 bra VCOUNT;
    mul.wide.u32 %rd31, %r18, 8;
    add.u64 %rd32, %rd20, %rd31;
    ld.global.u64 %rd30, [%rd32];
VCOUNT:
    add.u32 %r8, %r8, 1;
CHAINNEXT:
    @%p12 bra NEXTSHARD;
    mul.wide.u32 %rd43, %r18, 4;
    add.u64 %rd44, %rd42, %rd43;
    ld.global.u32 %r18, [%rd44];
    bra CHAIN;
BADINDEX:
    mov.u32 %r8, 4294967295;
    bra WRITEOUT;
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
    mul.wide.u32 %rd33, %r6, 4;
    add.u64 %rd34, %rd4, %rd33;
    st.global.u32 [%rd34], %r9;
    add.u64 %rd35, %rd5, %rd33;
    st.global.u32 [%rd35], %r10;
    add.u64 %rd36, %rd6, %rd33;
    st.global.u32 [%rd36], %r8;
    mul.wide.u32 %rd37, %r6, 8;
    add.u64 %rd38, %rd7, %rd37;
    st.global.u64 [%rd38], %rd30;
    add.u64 %rd41, %rd39, %rd37;
    st.global.u64 [%rd41], %rd40;

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
    .reg .pred %p<8>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<28>;

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
    mul.wide.u32 %rd8, %r10, 24;
    add.u64 %rd9, %rd1, %rd8;
    ld.global.u64 %rd10, [%rd9];
    ld.global.u64 %rd11, [%rd9+8];
    cvt.u32.u64 %r11, %rd11;
    shr.u64 %rd12, %rd11, 32;
    cvt.u32.u64 %r12, %rd12;
    ld.global.u32 %r20, [%rd9+16];
    add.u32 %r21, %r11, 1;
    cvt.u64.u32 %rd21, %r21;
    shl.b64 %rd21, %rd21, 3;
    add.u64 %rd21, %rd10, %rd21;

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
    // Bit 31 marks a singleton directory head: emit it, then skip the dependent link load.
    setp.lt.s32 %p7, %r17, 0;
    and.b32 %r17, %r17, 2147483647;
    mov.u32 %r22, 0;
CHAIN:
    setp.eq.u32 %p2, %r17, 0;
    @%p2 bra NEXTSHARD;
    setp.ge.u32 %p6, %r22, %r20;
    @%p6 bra BADINDEX;
    add.u32 %r22, %r22, 1;
    sub.u32 %r17, %r17, 1;
    setp.ge.u32 %p6, %r17, %r20;
    @%p6 bra BADINDEX;
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
    @%p7 bra NEXTSHARD;
    mul.wide.u32 %rd22, %r17, 4;
    add.u64 %rd23, %rd21, %rd22;
    ld.global.u32 %r17, [%rd23];
    bra CHAIN;
BADINDEX:
    mov.u32 %r9, 4294967294;
    bra WRITECOUNT;
NEXTSHARD:
    add.u32 %r10, %r10, 1;
    bra SHARD;

WRITECOUNT:
    mul.wide.u32 %rd21, %r7, 4;
    add.u64 %rd22, %rd5, %rd21;
    setp.eq.u32 %p6, %r9, 4294967294;
    @%p6 bra WINVALID;
    setp.gt.u32 %p5, %r9, %r3;
    @%p5 bra WOVER;
    st.global.u32 [%rd22], %r9;
    bra DONE;
WOVER:
    mov.u32 %r19, 4294967295;
    st.global.u32 [%rd22], %r19;
    bra DONE;
WINVALID:
    st.global.u32 [%rd22], %r9;

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
        const INVALID_COUNT: u32 = u32::MAX - 1;
        let max_count = shards
            .iter()
            .try_fold(0usize, |total, shard| {
                total.checked_add(shard.row_count as usize)
            })
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let max_count_u32 = u32::try_from(max_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(max_count))?;
        if max_count_u32 >= INVALID_COUNT {
            return Err(CudaRuntimeProbeError::InvalidInputLength(max_count));
        }
        let shard_count_u32 = u32::try_from(shards.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(shards.len()))?;
        let needle_count_u32 = u32::try_from(needles.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;
        // COUNT-ONLY fast path (`max_hits == 0`): the kernel emits only per-needle counts (any hit
        // -> u32::MAX), NO shard/slot output — so wave-batch validation skips 2 device buffers +
        // 2 DtoH reads per wave (the shard/slot outputs it never consumes). The FOUND arm takes
        // INCCOUNT (count >= max_hits == 0 always), so the shard/slot pointers are never
        // dereferenced; a valid dummy (the count buffer) is passed for them.
        let count_only = max_hits == 0;
        const DESC_U64_PER_SHARD: usize = 3; // index_ptr, mask|shift, logical row_count
        let desc_capacity = shards
            .len()
            .checked_mul(DESC_U64_PER_SHARD)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let mut desc: Vec<u64> = Vec::with_capacity(desc_capacity);
        let mut index_guards: Vec<Arc<CudaResidentDeviceMemory>> = Vec::with_capacity(shards.len());
        let primary = self.primary_arc();
        for shard in shards {
            if shard.index.device_ptr() == 0 {
                return Err(CudaRuntimeProbeError::InvalidInputLength(0));
            }
            if !Arc::ptr_eq(&primary, &shard.index.primary_arc()) {
                return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
            }
            validate_index_geometry(
                shard.index.metadata().allocated_bytes,
                shard.table_mask,
                shard.hash_shift,
                shard.row_count,
            )?;
            desc.push(shard.index.device_ptr());
            desc.push((shard.table_mask as u64) | ((shard.hash_shift as u64) << 32));
            desc.push(u64::from(shard.row_count));
            index_guards.push(Arc::clone(&shard.index));
        }
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
        let cu_stream_sync = unsafe {
            *primary
                .lib()
                .get::<CuStreamSync>(b"cuStreamSynchronize\0")
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

        // HtoD needles + the typed, preflighted descriptor image.
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
        let mut stream_drain = DefaultStreamDrain {
            sync: cu_stream_sync,
            armed: true,
        };
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
        if count.contains(&INVALID_COUNT) {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        stream_drain.armed = false;
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
    /// OWN snapshot. `needles` and `snapshots` are parallel; typed shard owners pin every region.
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
        const INVALID_COUNT: u32 = u32::MAX;
        let max_count = shards
            .iter()
            .try_fold(0usize, |total, shard| {
                total.checked_add(shard.row_count as usize)
            })
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let max_count_u32 = u32::try_from(max_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(max_count))?;
        if max_count_u32 == INVALID_COUNT {
            return Err(CudaRuntimeProbeError::InvalidInputLength(max_count));
        }
        let shard_count_u32 = u32::try_from(shards.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(shards.len()))?;
        let needle_count_u32 = u32::try_from(needles.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;
        const DESC_U64_PER_SHARD: usize = 6; // index, geometry, created, deleted, identity, row_count
        let desc_capacity = shards
            .len()
            .checked_mul(DESC_U64_PER_SHARD)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let mut desc: Vec<u64> = Vec::with_capacity(desc_capacity);
        let mut index_guards: Vec<Arc<CudaResidentDeviceMemory>> = Vec::with_capacity(shards.len());
        let primary = self.primary_arc();
        for shard in shards {
            if shard.index.device_ptr() == 0 {
                return Err(CudaRuntimeProbeError::InvalidInputLength(0));
            }
            if !Arc::ptr_eq(&primary, &shard.index.primary_arc()) {
                return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
            }
            validate_index_geometry(
                shard.index.metadata().allocated_bytes,
                shard.table_mask,
                shard.hash_shift,
                shard.row_count,
            )?;
            for region in [
                shard.created_by.as_ref(),
                shard.deleted_by.as_ref(),
                shard.row_id.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if !Arc::ptr_eq(&primary, &region.primary_arc()) {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
                }
                validate_version_region(region, shard.row_count)?;
            }
            desc.push(shard.index.device_ptr());
            desc.push((shard.table_mask as u64) | ((shard.hash_shift as u64) << 32));
            desc.push(
                shard
                    .created_by
                    .as_ref()
                    .map_or(0, |region| region.device_ptr()),
            );
            desc.push(
                shard
                    .deleted_by
                    .as_ref()
                    .map_or(0, |region| region.device_ptr()),
            );
            desc.push(
                shard
                    .row_id
                    .as_ref()
                    .map_or(0, |region| region.device_ptr()),
            );
            desc.push(u64::from(shard.row_count));
            index_guards.push(Arc::clone(&shard.index));
        }
        let out_bytes = needles
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let row_id_out_bytes = needles
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let needle_bytes = std::mem::size_of_val(needles);
        let snapshot_bytes = std::mem::size_of_val(snapshots);
        let desc_bytes = std::mem::size_of_val(desc.as_slice());

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
        let cu_stream_sync = unsafe {
            *primary
                .lib()
                .get::<CuStreamSync>(b"cuStreamSynchronize\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };

        let needles_guard = primary.lease_device_buffer_owned(needle_bytes)?;
        let snapshots_guard = primary.lease_device_buffer_owned(snapshot_bytes)?;
        let desc_guard = primary.lease_device_buffer_owned(desc_bytes)?;
        let shard_out_guard = primary.lease_device_buffer_owned(out_bytes)?;
        let slot_out_guard = primary.lease_device_buffer_owned(out_bytes)?;
        let count_guard = primary.lease_device_buffer_owned(out_bytes)?;
        let row_id_out_guard = primary.lease_device_buffer_owned(row_id_out_bytes)?;
        let latest_write_out_guard = primary.lease_device_buffer_owned(row_id_out_bytes)?;

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

        let mut desc_arg = desc_guard.ptr;
        let mut shard_count_arg = shard_count_u32;
        let mut needle_count_arg = needle_count_u32;
        let mut needles_arg = needles_guard.ptr;
        let mut snapshots_arg = snapshots_guard.ptr;
        let mut shard_out_arg = shard_out_guard.ptr;
        let mut slot_out_arg = slot_out_guard.ptr;
        let mut count_arg = count_guard.ptr;
        let mut row_id_out_arg = row_id_out_guard.ptr;
        let mut latest_write_out_arg = latest_write_out_guard.ptr;
        let mut args = [
            (&mut desc_arg as *mut u64).cast::<c_void>(),
            (&mut shard_count_arg as *mut u32).cast::<c_void>(),
            (&mut needle_count_arg as *mut u32).cast::<c_void>(),
            (&mut needles_arg as *mut u64).cast::<c_void>(),
            (&mut snapshots_arg as *mut u64).cast::<c_void>(),
            (&mut shard_out_arg as *mut u64).cast::<c_void>(),
            (&mut slot_out_arg as *mut u64).cast::<c_void>(),
            (&mut count_arg as *mut u64).cast::<c_void>(),
            (&mut row_id_out_arg as *mut u64).cast::<c_void>(),
            (&mut latest_write_out_arg as *mut u64).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = needle_count_u32.div_ceil(threads_per_block);
        let mut stream_drain = DefaultStreamDrain {
            sync: cu_stream_sync,
            armed: true,
        };
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
        let mut shard_idx = vec![0u32; needles.len()];
        let mut slot = vec![0u32; needles.len()];
        let mut count = vec![0u32; needles.len()];
        let mut row_id = vec![u64::MAX; needles.len()];
        let mut latest_write = vec![0u64; needles.len()];
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
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                row_id.as_mut_ptr().cast::<c_void>(),
                row_id_out_guard.ptr,
                row_id_out_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                latest_write.as_mut_ptr().cast::<c_void>(),
                latest_write_out_guard.ptr,
                row_id_out_bytes,
            )
        })?;
        if count.contains(&INVALID_COUNT) {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        stream_drain.armed = false;
        drop(index_guards);
        Ok(VisibleLocateResult {
            shard_idx,
            slot,
            count,
            row_id,
            latest_write,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::validate_index_geometry;

    #[test]
    fn write_locate_geometry_requires_exact_power_of_two_addressing() {
        validate_index_geometry(132, 15, 28, 1).unwrap();
        assert!(validate_index_geometry(131, 15, 28, 1).is_err());
        assert!(validate_index_geometry(132, 14, 28, 1).is_err());
        assert!(validate_index_geometry(132, 15, 27, 1).is_err());
        assert!(validate_index_geometry(u64::MAX, 0, 32, 1).is_err());
    }
}
