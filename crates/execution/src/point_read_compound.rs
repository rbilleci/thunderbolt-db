//! Prepared fixed-width compound point reads.
//!
//! The route builds one generation-owned table-level hash directory directly from resident
//! `(int4, int8)` key columns. Probe threads derive the candidate fingerprint on-device, then
//! compare both full typed key components before applying MVCC visibility and gathering fixed-width
//! projections. The hash is therefore only a candidate selector; collisions cannot manufacture a
//! row. The host uploads typed parameters and performs only the bounded terminal values-and-status readback phase.

use std::ffi::c_void;
use std::sync::Arc;

use super::cuda_context::{PooledStreamOwned, RetainedDeviceBufferOwned};
use super::{
    check_cuda, validate_i32_index_geometry, CudaResidentDeviceMemory, CudaResidentReadSource,
    CudaRuntimeProbeError, GpuPrimaryContext,
};

const MAX_PROJECTIONS: usize = 4;
const DESC_U64S: usize = 11;

/// One fixed-width result column gathered by the compound route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaFixedPointProjectionKind {
    /// A resident i32-section value. This includes SQL int2, whose resident representation is widened i32.
    I32,
    /// A resident i64-section value. The kernel uses two 32-bit loads because the section may be 4-mod-8.
    I64,
}

/// One projected column in a shard payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaFixedPointProjection {
    pub byte_offset: u64,
    pub kind: CudaFixedPointProjectionKind,
}

/// Typed engine-internal key input. Its explicit three-word layout is the device ABI:
/// `[int4, int8-low32, int8-high32]`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaI32I64PointKey {
    words: [i32; 3],
}

impl CudaI32I64PointKey {
    pub fn new(first: i32, second: i64) -> Self {
        let bits = second as u64;
        Self {
            words: [first, bits as u32 as i32, (bits >> 32) as u32 as i32],
        }
    }
}

/// One shard captured by a prepared compound point route.
pub struct CompoundI32I64ProbeShard {
    pub resident: Arc<CudaResidentDeviceMemory>,
    pub key_i32_offset: u64,
    pub key_i64_offset: u64,
    pub projections: Vec<CudaFixedPointProjection>,
    pub row_count: u64,
    pub created_by: Option<Arc<CudaResidentDeviceMemory>>,
    pub deleted_by: Option<Arc<CudaResidentDeviceMemory>>,
}

/// Dense fixed-width result. Status is one byte per input key: 1=one visible exact match, 2=absent,
/// 3=multiple visible exact matches (the unique route must fail/decline).
pub struct CudaI32I64PointBatchProjection {
    values: Vec<u64>,
    projection_kinds: Vec<CudaFixedPointProjectionKind>,
    status: Vec<u8>,
}

impl CudaI32I64PointBatchProjection {
    pub fn status(&self) -> &[u8] {
        &self.status
    }

    pub fn projection_kinds(&self) -> &[CudaFixedPointProjectionKind] {
        &self.projection_kinds
    }

    pub fn values(&self) -> &[u64] {
        &self.values
    }
}

/// Generation-owned global compound point directory. The descriptor table pins every resident payload and
/// visibility sidecar it names; the index stores `(fingerprint, global_row_ordinal)` and the row map resolves
/// that ordinal to one descriptor in O(1). Exact key verification remains on-device.
pub struct CudaI32I64MultiShardProbePlan {
    primary: Arc<GpuPrimaryContext>,
    projection_kinds: Vec<CudaFixedPointProjectionKind>,
    shard_count: u32,
    total_rows: u32,
    table_mask: u32,
    hash_shift: u32,
    gc_boundary: u64,
    descriptor_guard: RetainedDeviceBufferOwned,
    index_guard: RetainedDeviceBufferOwned,
    row_to_shard_guard: RetainedDeviceBufferOwned,
    _resource_guards: Vec<Arc<CudaResidentDeviceMemory>>,
}

impl std::fmt::Debug for CudaI32I64MultiShardProbePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaI32I64MultiShardProbePlan")
            .field("projection_kinds", &self.projection_kinds)
            .field("shard_count", &self.shard_count)
            .field("total_rows", &self.total_rows)
            .finish_non_exhaustive()
    }
}

impl CudaI32I64MultiShardProbePlan {
    /// Exact live dedicated allocation bytes retained by this route.
    pub fn allocated_bytes(&self) -> u64 {
        [
            self.descriptor_guard.capacity,
            self.index_guard.capacity,
            self.row_to_shard_guard.capacity,
        ]
        .into_iter()
        .map(|bytes| bytes as u64)
        .sum()
    }

    /// Exact dedicated allocation charge needed for this many physical rows and shards.
    pub fn estimated_allocated_bytes(
        shard_count: usize,
        total_rows: u64,
    ) -> Result<u64, CudaRuntimeProbeError> {
        if shard_count == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let (_, _, index_bytes) = index_geometry(total_rows)?;
        let row_map_bytes = usize::try_from(
            total_rows
                .checked_mul(std::mem::size_of::<u32>() as u64)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        )
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let descriptor_bytes = shard_count
            .checked_mul(DESC_U64S)
            .and_then(|words| words.checked_mul(std::mem::size_of::<u64>()))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        [index_bytes, row_map_bytes, descriptor_bytes]
            .into_iter()
            .map(|bytes| {
                u64::try_from(bytes)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))
            })
            .try_fold(0_u64, |sum, bytes| {
                sum.checked_add(bytes?)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))
            })
    }
}

fn validate_plan_read_snapshot(
    gc_boundary: u64,
    read_snapshot: u64,
) -> Result<(), CudaRuntimeProbeError> {
    if read_snapshot < gc_boundary {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(read_snapshot).unwrap_or(usize::MAX),
        ));
    }
    Ok(())
}

fn index_geometry(total_rows: u64) -> Result<(u32, u32, usize), CudaRuntimeProbeError> {
    if total_rows == 0 || total_rows >= u32::MAX as u64 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(total_rows).unwrap_or(usize::MAX),
        ));
    }
    let slots = total_rows
        .checked_mul(2)
        .and_then(u64::checked_next_power_of_two)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if slots > (1_u64 << 30) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(total_rows).unwrap_or(usize::MAX),
        ));
    }
    let table_mask = (slots - 1) as u32;
    let hash_shift = 32 - slots.trailing_zeros();
    let bytes = usize::try_from(
        slots
            .checked_mul(std::mem::size_of::<u64>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    Ok((table_mask, hash_shift, bytes))
}

const BUILD_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_compound_i32_i64_global_index_build(
    .param .u64 desc_ptr,
    .param .u32 shard_count,
    .param .u32 total_rows,
    .param .u64 index_ptr,
    .param .u64 row_to_shard_ptr,
    .param .u32 table_mask,
    .param .u32 hash_shift,
    .param .u64 gc_boundary,
    .param .u64 decline_ptr
)
{
    .reg .pred %p<12>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<48>;

    ld.param.u64 %rd1, [desc_ptr];
    ld.param.u32 %r1, [shard_count];
    ld.param.u32 %r2, [total_rows];
    ld.param.u64 %rd2, [index_ptr];
    ld.param.u64 %rd3, [row_to_shard_ptr];
    ld.param.u32 %r3, [table_mask];
    ld.param.u32 %r4, [hash_shift];
    ld.param.u64 %rd4, [gc_boundary];
    ld.param.u64 %rd5, [decline_ptr];

    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mad.lo.u32 %r8, %r6, %r7, %r5;
    setp.ge.u32 %p1, %r8, %r2;
    @%p1 bra DONE;

    // Find the last descriptor whose global row base is <= this global row.
    mov.u32 %r9, 0;
    mov.u32 %r10, %r1;
BSEARCH:
    setp.ge.u32 %p2, %r9, %r10;
    @%p2 bra BFOUND;
    add.u32 %r11, %r9, %r10;
    shr.u32 %r11, %r11, 1;
    mul.wide.u32 %rd6, %r11, 88;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u64 %rd8, [%rd7+80];
    cvt.u32.u64 %r12, %rd8;
    setp.le.u32 %p2, %r12, %r8;
    @%p2 bra BLO;
    mov.u32 %r10, %r11;
    bra BSEARCH;
BLO:
    add.u32 %r9, %r11, 1;
    bra BSEARCH;
BFOUND:
    setp.eq.u32 %p2, %r9, 0;
    @%p2 bra DECLINE;
    sub.u32 %r13, %r9, 1;
    mul.wide.u32 %rd6, %r13, 88;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u64 %rd8, [%rd7+80];
    cvt.u32.u64 %r12, %rd8;
    sub.u32 %r14, %r8, %r12;
    ld.global.u64 %rd9, [%rd7+72];
    cvt.u32.u64 %r15, %rd9;
    setp.ge.u32 %p2, %r14, %r15;
    @%p2 bra DECLINE;

    // Omit only rows dead at/below the route's build boundary.
    ld.global.u64 %rd10, [%rd7+64];
    setp.eq.u64 %p3, %rd10, 0;
    @%p3 bra LOADKEY;
    mul.wide.u32 %rd11, %r14, 8;
    add.u64 %rd12, %rd10, %rd11;
    ld.global.u64 %rd13, [%rd12];
    setp.le.u64 %p3, %rd13, %rd4;
    @%p3 bra DONE;

LOADKEY:
    ld.global.u64 %rd14, [%rd7];
    ld.global.u64 %rd15, [%rd7+8];
    ld.global.u64 %rd16, [%rd7+16];
    mul.wide.u32 %rd17, %r14, 4;
    add.u64 %rd18, %rd14, %rd15;
    add.u64 %rd18, %rd18, %rd17;
    ld.global.u32 %r16, [%rd18];
    mul.wide.u32 %rd19, %r14, 8;
    add.u64 %rd20, %rd14, %rd16;
    add.u64 %rd20, %rd20, %rd19;
    ld.global.u32 %r17, [%rd20];
    ld.global.u32 %r18, [%rd20+4];

    // Canonical three-word FNV/rotate fold: int4, int8 low, int8 high.
    mov.u32 %r19, 2166136261;
    xor.b32 %r19, %r19, %r16;
    mul.lo.u32 %r19, %r19, 16777619;
    shl.b32 %r20, %r19, 13;
    shr.b32 %r21, %r19, 19;
    or.b32 %r19, %r20, %r21;
    add.u32 %r19, %r19, 2654435761;
    xor.b32 %r19, %r19, %r17;
    mul.lo.u32 %r19, %r19, 16777619;
    shl.b32 %r20, %r19, 13;
    shr.b32 %r21, %r19, 19;
    or.b32 %r19, %r20, %r21;
    add.u32 %r19, %r19, 2654435761;
    xor.b32 %r19, %r19, %r18;
    mul.lo.u32 %r19, %r19, 16777619;
    shl.b32 %r20, %r19, 13;
    shr.b32 %r21, %r19, 19;
    or.b32 %r19, %r20, %r21;
    add.u32 %r19, %r19, 2654435761;

    mul.lo.u32 %r22, %r19, 2654435761;
    shr.u32 %r23, %r22, %r4;
    mov.u32 %r24, 0;
    add.u32 %r25, %r8, 1;
    cvt.u64.u32 %rd21, %r19;
    shl.b64 %rd21, %rd21, 32;
    cvt.u64.u32 %rd22, %r25;
    or.b64 %rd23, %rd21, %rd22;
INSERT:
    and.b32 %r23, %r23, %r3;
    mul.wide.u32 %rd24, %r23, 8;
    add.u64 %rd25, %rd2, %rd24;
    atom.global.cas.b64 %rd26, [%rd25], 0, %rd23;
    setp.eq.u64 %p4, %rd26, 0;
    @%p4 bra STORESHARD;
    add.u32 %r23, %r23, 1;
    add.u32 %r24, %r24, 1;
    setp.ge.u32 %p4, %r24, 256;
    @%p4 bra DECLINE;
    bra INSERT;

STORESHARD:
    mul.wide.u32 %rd27, %r8, 4;
    add.u64 %rd28, %rd3, %rd27;
    st.global.u32 [%rd28], %r13;
    bra DONE;

DECLINE:
    mov.u32 %r30, 1;
    atom.global.exch.b32 %r31, [%rd5], %r30;
DONE:
    ret;
}
"#;

const PROBE_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_compound_i32_i64_global_probe(
    .param .u64 desc_ptr,
    .param .u32 shard_count,
    .param .u32 total_rows,
    .param .u64 index_ptr,
    .param .u64 row_to_shard_ptr,
    .param .u32 table_mask,
    .param .u32 hash_shift,
    .param .u32 projection_count,
    .param .u32 projection_i64_mask,
    .param .u64 keys_ptr,
    .param .u32 key_count,
    .param .u64 read_snapshot,
    .param .u64 out_values_ptr,
    .param .u64 out_status_ptr
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<72>;

    ld.param.u64 %rd1, [desc_ptr];
    ld.param.u32 %r1, [shard_count];
    ld.param.u32 %r2, [total_rows];
    ld.param.u64 %rd2, [index_ptr];
    ld.param.u64 %rd3, [row_to_shard_ptr];
    ld.param.u32 %r3, [table_mask];
    ld.param.u32 %r4, [hash_shift];
    ld.param.u32 %r5, [projection_count];
    ld.param.u32 %r6, [projection_i64_mask];
    ld.param.u64 %rd4, [keys_ptr];
    ld.param.u32 %r7, [key_count];
    ld.param.u64 %rd5, [read_snapshot];
    ld.param.u64 %rd6, [out_values_ptr];
    ld.param.u64 %rd7, [out_status_ptr];

    mov.u32 %r8, %tid.x;
    mov.u32 %r9, %ctaid.x;
    mov.u32 %r10, %ntid.x;
    mad.lo.u32 %r11, %r9, %r10, %r8;
    setp.ge.u32 %p1, %r11, %r7;
    @%p1 bra DONE;

    mul.wide.u32 %rd8, %r11, 12;
    add.u64 %rd9, %rd4, %rd8;
    ld.global.u32 %r12, [%rd9];
    ld.global.u32 %r13, [%rd9+4];
    ld.global.u32 %r14, [%rd9+8];

    mov.u32 %r15, 2166136261;
    xor.b32 %r15, %r15, %r12;
    mul.lo.u32 %r15, %r15, 16777619;
    shl.b32 %r16, %r15, 13;
    shr.b32 %r17, %r15, 19;
    or.b32 %r15, %r16, %r17;
    add.u32 %r15, %r15, 2654435761;
    xor.b32 %r15, %r15, %r13;
    mul.lo.u32 %r15, %r15, 16777619;
    shl.b32 %r16, %r15, 13;
    shr.b32 %r17, %r15, 19;
    or.b32 %r15, %r16, %r17;
    add.u32 %r15, %r15, 2654435761;
    xor.b32 %r15, %r15, %r14;
    mul.lo.u32 %r15, %r15, 16777619;
    shl.b32 %r16, %r15, 13;
    shr.b32 %r17, %r15, 19;
    or.b32 %r15, %r16, %r17;
    add.u32 %r15, %r15, 2654435761;

    mul.lo.u32 %r18, %r15, 2654435761;
    shr.u32 %r19, %r18, %r4;
    mov.u32 %r20, 0;
    mov.u32 %r21, 0;
    mov.u32 %r22, 0;
    mov.u32 %r23, 0;

PROBE:
    and.b32 %r19, %r19, %r3;
    mul.wide.u32 %rd10, %r19, 8;
    add.u64 %rd11, %rd2, %rd10;
    ld.global.u64 %rd12, [%rd11];
    setp.eq.u64 %p2, %rd12, 0;
    @%p2 bra PROBEDONE;
    shr.u64 %rd13, %rd12, 32;
    cvt.u32.u64 %r24, %rd13;
    setp.ne.u32 %p2, %r24, %r15;
    @%p2 bra ADVANCE;
    cvt.u32.u64 %r25, %rd12;
    setp.eq.u32 %p2, %r25, 0;
    @%p2 bra ADVANCE;
    sub.u32 %r26, %r25, 1;
    setp.ge.u32 %p2, %r26, %r2;
    @%p2 bra ADVANCE;
    mul.wide.u32 %rd14, %r26, 4;
    add.u64 %rd15, %rd3, %rd14;
    ld.global.u32 %r27, [%rd15];
    setp.ge.u32 %p2, %r27, %r1;
    @%p2 bra ADVANCE;
    mul.wide.u32 %rd16, %r27, 88;
    add.u64 %rd17, %rd1, %rd16;
    ld.global.u64 %rd18, [%rd17+80];
    cvt.u32.u64 %r28, %rd18;
    sub.u32 %r29, %r26, %r28;
    ld.global.u64 %rd19, [%rd17+72];
    cvt.u32.u64 %r30, %rd19;
    setp.ge.u32 %p2, %r29, %r30;
    @%p2 bra ADVANCE;

    // Full typed collision recheck. The i64 component is always read as two aligned words.
    ld.global.u64 %rd20, [%rd17];
    ld.global.u64 %rd21, [%rd17+8];
    ld.global.u64 %rd22, [%rd17+16];
    mul.wide.u32 %rd23, %r29, 4;
    add.u64 %rd24, %rd20, %rd21;
    add.u64 %rd24, %rd24, %rd23;
    ld.global.u32 %r31, [%rd24];
    setp.ne.u32 %p3, %r31, %r12;
    @%p3 bra ADVANCE;
    mul.wide.u32 %rd25, %r29, 8;
    add.u64 %rd26, %rd20, %rd22;
    add.u64 %rd26, %rd26, %rd25;
    ld.global.u32 %r32, [%rd26];
    ld.global.u32 %r33, [%rd26+4];
    setp.ne.u32 %p3, %r32, %r13;
    @%p3 bra ADVANCE;
    setp.ne.u32 %p3, %r33, %r14;
    @%p3 bra ADVANCE;

    // MVCC visibility against the exact descriptor generation.
    ld.global.u64 %rd27, [%rd17+56];
    setp.eq.u64 %p4, %rd27, 0;
    @%p4 bra CREATEDOK;
    mul.wide.u32 %rd28, %r29, 8;
    add.u64 %rd29, %rd27, %rd28;
    ld.global.u64 %rd30, [%rd29];
    setp.gt.u64 %p4, %rd30, %rd5;
    @%p4 bra ADVANCE;
CREATEDOK:
    ld.global.u64 %rd31, [%rd17+64];
    setp.eq.u64 %p5, %rd31, 0;
    @%p5 bra VISIBLE;
    mul.wide.u32 %rd32, %r29, 8;
    add.u64 %rd33, %rd31, %rd32;
    ld.global.u64 %rd34, [%rd33];
    setp.le.u64 %p5, %rd34, %rd5;
    @%p5 bra ADVANCE;
VISIBLE:
    setp.ne.u32 %p6, %r21, 0;
    @%p6 bra DUP;
    mov.u32 %r21, 1;
    mov.u32 %r22, %r27;
    mov.u32 %r23, %r29;
    bra ADVANCE;

DUP:
    cvt.u64.u32 %rd35, %r11;
    add.u64 %rd36, %rd7, %rd35;
    mov.u32 %r34, 3;
    st.global.u8 [%rd36], %r34;
    bra DONE;

ADVANCE:
    add.u32 %r19, %r19, 1;
    add.u32 %r20, %r20, 1;
    setp.ge.u32 %p7, %r20, 256;
    @%p7 bra PROBEDONE;
    bra PROBE;

PROBEDONE:
    setp.eq.u32 %p8, %r21, 0;
    @%p8 bra ABSENT;
    mul.wide.u32 %rd37, %r22, 88;
    add.u64 %rd38, %rd1, %rd37;
    ld.global.u64 %rd39, [%rd38];
    cvt.u64.u32 %rd40, %r11;
    cvt.u64.u32 %rd41, %r5;
    mul.lo.u64 %rd42, %rd40, %rd41;
    mul.lo.u64 %rd42, %rd42, 8;
    add.u64 %rd43, %rd6, %rd42;

    // Projection 0.
    ld.global.u64 %rd44, [%rd38+24];
    and.b32 %r35, %r6, 1;
    setp.ne.u32 %p9, %r35, 0;
    @%p9 bra P0I64;
    mul.wide.u32 %rd45, %r23, 4;
    add.u64 %rd46, %rd39, %rd44;
    add.u64 %rd46, %rd46, %rd45;
    ld.global.u32 %r36, [%rd46];
    cvt.u64.u32 %rd47, %r36;
    bra P0STORE;
P0I64:
    mul.wide.u32 %rd45, %r23, 8;
    add.u64 %rd46, %rd39, %rd44;
    add.u64 %rd46, %rd46, %rd45;
    ld.global.u32 %r36, [%rd46];
    ld.global.u32 %r37, [%rd46+4];
    cvt.u64.u32 %rd47, %r37;
    shl.b64 %rd47, %rd47, 32;
    cvt.u64.u32 %rd48, %r36;
    or.b64 %rd47, %rd47, %rd48;
P0STORE:
    st.global.u64 [%rd43], %rd47;
    setp.le.u32 %p10, %r5, 1;
    @%p10 bra FOUND;

    // Projection 1.
    ld.global.u64 %rd44, [%rd38+32];
    shr.u32 %r35, %r6, 1;
    and.b32 %r35, %r35, 1;
    setp.ne.u32 %p9, %r35, 0;
    @%p9 bra P1I64;
    mul.wide.u32 %rd45, %r23, 4;
    add.u64 %rd46, %rd39, %rd44;
    add.u64 %rd46, %rd46, %rd45;
    ld.global.u32 %r36, [%rd46];
    cvt.u64.u32 %rd47, %r36;
    bra P1STORE;
P1I64:
    mul.wide.u32 %rd45, %r23, 8;
    add.u64 %rd46, %rd39, %rd44;
    add.u64 %rd46, %rd46, %rd45;
    ld.global.u32 %r36, [%rd46];
    ld.global.u32 %r37, [%rd46+4];
    cvt.u64.u32 %rd47, %r37;
    shl.b64 %rd47, %rd47, 32;
    cvt.u64.u32 %rd48, %r36;
    or.b64 %rd47, %rd47, %rd48;
P1STORE:
    st.global.u64 [%rd43+8], %rd47;
    setp.le.u32 %p10, %r5, 2;
    @%p10 bra FOUND;

    // Projection 2.
    ld.global.u64 %rd44, [%rd38+40];
    shr.u32 %r35, %r6, 2;
    and.b32 %r35, %r35, 1;
    setp.ne.u32 %p9, %r35, 0;
    @%p9 bra P2I64;
    mul.wide.u32 %rd45, %r23, 4;
    add.u64 %rd46, %rd39, %rd44;
    add.u64 %rd46, %rd46, %rd45;
    ld.global.u32 %r36, [%rd46];
    cvt.u64.u32 %rd47, %r36;
    bra P2STORE;
P2I64:
    mul.wide.u32 %rd45, %r23, 8;
    add.u64 %rd46, %rd39, %rd44;
    add.u64 %rd46, %rd46, %rd45;
    ld.global.u32 %r36, [%rd46];
    ld.global.u32 %r37, [%rd46+4];
    cvt.u64.u32 %rd47, %r37;
    shl.b64 %rd47, %rd47, 32;
    cvt.u64.u32 %rd48, %r36;
    or.b64 %rd47, %rd47, %rd48;
P2STORE:
    st.global.u64 [%rd43+16], %rd47;
    setp.le.u32 %p10, %r5, 3;
    @%p10 bra FOUND;

    // Projection 3.
    ld.global.u64 %rd44, [%rd38+48];
    shr.u32 %r35, %r6, 3;
    and.b32 %r35, %r35, 1;
    setp.ne.u32 %p9, %r35, 0;
    @%p9 bra P3I64;
    mul.wide.u32 %rd45, %r23, 4;
    add.u64 %rd46, %rd39, %rd44;
    add.u64 %rd46, %rd46, %rd45;
    ld.global.u32 %r36, [%rd46];
    cvt.u64.u32 %rd47, %r36;
    bra P3STORE;
P3I64:
    mul.wide.u32 %rd45, %r23, 8;
    add.u64 %rd46, %rd39, %rd44;
    add.u64 %rd46, %rd46, %rd45;
    ld.global.u32 %r36, [%rd46];
    ld.global.u32 %r37, [%rd46+4];
    cvt.u64.u32 %rd47, %r37;
    shl.b64 %rd47, %rd47, 32;
    cvt.u64.u32 %rd48, %r36;
    or.b64 %rd47, %rd47, %rd48;
P3STORE:
    st.global.u64 [%rd43+24], %rd47;

FOUND:
    cvt.u64.u32 %rd49, %r11;
    add.u64 %rd50, %rd7, %rd49;
    mov.u32 %r38, 1;
    st.global.u8 [%rd50], %r38;
    bra DONE;

ABSENT:
    cvt.u64.u32 %rd49, %r11;
    add.u64 %rd50, %rd7, %rd49;
    mov.u32 %r38, 2;
    st.global.u8 [%rd50], %r38;
DONE:
    ret;
}
"#;

pub(super) fn prepare_cuda_compound_i32_i64_multi_shard_probe(
    ctx: &CudaResidentDeviceMemory,
    shards: &[CompoundI32I64ProbeShard],
    gc_boundary: u64,
) -> Result<CudaI32I64MultiShardProbePlan, CudaRuntimeProbeError> {
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

    if shards.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let primary = ctx.primary_arc();
    let projection_kinds = shards[0]
        .projections
        .iter()
        .map(|projection| projection.kind)
        .collect::<Vec<_>>();
    if projection_kinds.is_empty() || projection_kinds.len() > MAX_PROJECTIONS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            projection_kinds.len(),
        ));
    }
    let mut total_rows = 0_u64;
    let mut descriptors = Vec::with_capacity(shards.len() * DESC_U64S);
    let mut resource_guards = Vec::with_capacity(shards.len() * 3);
    for shard in shards {
        if shard.projections.len() != projection_kinds.len()
            || shard
                .projections
                .iter()
                .map(|projection| projection.kind)
                .ne(projection_kinds.iter().copied())
            || shard.row_count == 0
            || shard.row_count >= u32::MAX as u64
            || shard.resident.device_ptr() == 0
            || !Arc::ptr_eq(&primary, &shard.resident.primary_arc())
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let allocated = shard.resident.metadata().allocated_bytes;
        for (offset, width) in [
            (shard.key_i32_offset, std::mem::size_of::<i32>() as u64),
            (shard.key_i64_offset, std::mem::size_of::<i64>() as u64),
        ] {
            if !offset.is_multiple_of(std::mem::align_of::<i32>() as u64) {
                return Err(CudaRuntimeProbeError::InvalidInputLength(offset as usize));
            }
            let end = offset
                .checked_add(
                    shard
                        .row_count
                        .checked_mul(width)
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
                )
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if end > allocated {
                return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
            }
        }
        for projection in &shard.projections {
            if !projection
                .byte_offset
                .is_multiple_of(std::mem::align_of::<i32>() as u64)
            {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    projection.byte_offset as usize,
                ));
            }
            let width = match projection.kind {
                CudaFixedPointProjectionKind::I32 => std::mem::size_of::<i32>() as u64,
                CudaFixedPointProjectionKind::I64 => std::mem::size_of::<i64>() as u64,
            };
            let end = projection
                .byte_offset
                .checked_add(
                    shard
                        .row_count
                        .checked_mul(width)
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
                )
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if end > allocated {
                return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
            }
        }
        for region in [&shard.created_by, &shard.deleted_by].into_iter().flatten() {
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
        }
        descriptors.push(shard.resident.device_ptr());
        descriptors.push(shard.key_i32_offset);
        descriptors.push(shard.key_i64_offset);
        let mut projection_offsets = [0_u64; MAX_PROJECTIONS];
        for (index, projection) in shard.projections.iter().enumerate() {
            projection_offsets[index] = projection.byte_offset;
        }
        descriptors.extend_from_slice(&projection_offsets);
        descriptors.push(
            shard
                .created_by
                .as_ref()
                .map_or(0, |region| region.device_ptr()),
        );
        descriptors.push(
            shard
                .deleted_by
                .as_ref()
                .map_or(0, |region| region.device_ptr()),
        );
        descriptors.push(shard.row_count);
        descriptors.push(total_rows);
        total_rows = total_rows
            .checked_add(shard.row_count)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        resource_guards.push(Arc::clone(&shard.resident));
        resource_guards.extend(
            [&shard.created_by, &shard.deleted_by]
                .into_iter()
                .flatten()
                .cloned(),
        );
    }
    let shard_count = u32::try_from(shards.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(shards.len()))?;
    let total_rows_u32 = u32::try_from(total_rows)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let (table_mask, hash_shift, index_bytes) = index_geometry(total_rows)?;
    let descriptor_bytes = std::mem::size_of_val(descriptors.as_slice());
    let row_map_bytes = usize::try_from(
        total_rows
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_memcpy_htod = unsafe {
        ctx.lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| ctx.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8 = unsafe {
        ctx.lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| ctx.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        ctx.lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    primary.set_current()?;
    let descriptor_guard = primary.allocate_retained_device_buffer_owned(descriptor_bytes)?;
    let index_guard = primary.allocate_retained_device_buffer_owned(index_bytes)?;
    let row_to_shard_guard = primary.allocate_retained_device_buffer_owned(row_map_bytes)?;
    let decline_guard = primary.lease_device_buffer_owned(std::mem::size_of::<u32>())?;
    check_cuda(unsafe {
        cu_memcpy_htod(
            descriptor_guard.ptr,
            descriptors.as_ptr().cast::<c_void>(),
            descriptor_bytes,
        )
    })?;
    check_cuda(unsafe { cu_memset_d8(index_guard.ptr, 0, index_bytes) })?;
    check_cuda(unsafe { cu_memset_d8(decline_guard.ptr, 0, std::mem::size_of::<u32>()) })?;

    let mut ptx = Vec::with_capacity(BUILD_PTX.len() + 1);
    ptx.extend_from_slice(BUILD_PTX);
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_compound_i32_i64_global_index_build", &ptx)?;
    let mut desc_arg = descriptor_guard.ptr;
    let mut shard_count_arg = shard_count;
    let mut total_rows_arg = total_rows_u32;
    let mut index_arg = index_guard.ptr;
    let mut row_map_arg = row_to_shard_guard.ptr;
    let mut table_mask_arg = table_mask;
    let mut hash_shift_arg = hash_shift;
    let mut gc_boundary_arg = gc_boundary;
    let mut decline_arg = decline_guard.ptr;
    let mut args = [
        (&mut desc_arg as *mut u64).cast::<c_void>(),
        (&mut shard_count_arg as *mut u32).cast::<c_void>(),
        (&mut total_rows_arg as *mut u32).cast::<c_void>(),
        (&mut index_arg as *mut u64).cast::<c_void>(),
        (&mut row_map_arg as *mut u64).cast::<c_void>(),
        (&mut table_mask_arg as *mut u32).cast::<c_void>(),
        (&mut hash_shift_arg as *mut u32).cast::<c_void>(),
        (&mut gc_boundary_arg as *mut u64).cast::<c_void>(),
        (&mut decline_arg as *mut u64).cast::<c_void>(),
    ];
    let stream_owned = PooledStreamOwned {
        primary: Arc::clone(&primary),
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let stream = stream_owned
        .pooled
        .as_ref()
        .expect("compound build stream just leased")
        .stream;
    let threads = 128_u32;
    let blocks = total_rows_u32.div_ceil(threads);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) })?;
    let mut declined = 0_u32;
    check_cuda(unsafe {
        (primary.cu_memcpy_dtoh)(
            (&mut declined as *mut u32).cast::<c_void>(),
            decline_guard.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;
    if declined != 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            total_rows as usize,
        ));
    }
    validate_i32_index_geometry(index_guard.capacity as u64, table_mask, hash_shift)?;
    Ok(CudaI32I64MultiShardProbePlan {
        primary,
        projection_kinds,
        shard_count,
        total_rows: total_rows_u32,
        table_mask,
        hash_shift,
        gc_boundary,
        descriptor_guard,
        index_guard,
        row_to_shard_guard,
        _resource_guards: resource_guards,
    })
}

pub(super) fn execute_cuda_compound_i32_i64_multi_shard_probe(
    ctx: &CudaResidentDeviceMemory,
    plan: &Arc<CudaI32I64MultiShardProbePlan>,
    keys: &[CudaI32I64PointKey],
    read_snapshot: u64,
) -> Result<CudaI32I64PointBatchProjection, CudaRuntimeProbeError> {
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

    if keys.is_empty() || !Arc::ptr_eq(&plan.primary, &ctx.primary_arc()) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(keys.len()));
    }
    validate_plan_read_snapshot(plan.gc_boundary, read_snapshot)?;
    let key_count = u32::try_from(keys.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(keys.len()))?;
    let key_bytes = std::mem::size_of_val(keys);
    let projection_count = plan.projection_kinds.len();
    let output_cells = keys
        .len()
        .checked_mul(projection_count)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output_bytes = output_cells
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let status_bytes = keys.len();
    let primary = ctx.primary_arc();
    primary.set_current()?;
    let keys_guard = primary.lease_device_buffer_owned(key_bytes)?;
    let values_guard = primary.lease_device_buffer_owned(output_bytes)?;
    let status_guard = primary.lease_device_buffer_owned(status_bytes)?;
    let cu_memcpy_htod = unsafe {
        ctx.lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| ctx.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8 = unsafe {
        ctx.lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| ctx.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        ctx.lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe {
        cu_memcpy_htod(keys_guard.ptr, keys.as_ptr().cast::<c_void>(), key_bytes)
    })?;
    check_cuda(unsafe { cu_memset_d8(values_guard.ptr, 0, output_bytes) })?;
    check_cuda(unsafe { cu_memset_d8(status_guard.ptr, 0, status_bytes) })?;
    let mut ptx = Vec::with_capacity(PROBE_PTX.len() + 1);
    ptx.extend_from_slice(PROBE_PTX);
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_compound_i32_i64_global_probe", &ptx)?;
    let projection_i64_mask =
        plan.projection_kinds
            .iter()
            .enumerate()
            .fold(0_u32, |mask, (index, kind)| {
                mask | (u32::from(*kind == CudaFixedPointProjectionKind::I64) << index)
            });
    let mut desc_arg = plan.descriptor_guard.ptr;
    let mut shard_count_arg = plan.shard_count;
    let mut total_rows_arg = plan.total_rows;
    let mut index_arg = plan.index_guard.ptr;
    let mut row_map_arg = plan.row_to_shard_guard.ptr;
    let mut table_mask_arg = plan.table_mask;
    let mut hash_shift_arg = plan.hash_shift;
    let mut projection_count_arg = projection_count as u32;
    let mut projection_i64_mask_arg = projection_i64_mask;
    let mut keys_arg = keys_guard.ptr;
    let mut key_count_arg = key_count;
    let mut read_snapshot_arg = read_snapshot;
    let mut values_arg = values_guard.ptr;
    let mut status_arg = status_guard.ptr;
    let mut args = [
        (&mut desc_arg as *mut u64).cast::<c_void>(),
        (&mut shard_count_arg as *mut u32).cast::<c_void>(),
        (&mut total_rows_arg as *mut u32).cast::<c_void>(),
        (&mut index_arg as *mut u64).cast::<c_void>(),
        (&mut row_map_arg as *mut u64).cast::<c_void>(),
        (&mut table_mask_arg as *mut u32).cast::<c_void>(),
        (&mut hash_shift_arg as *mut u32).cast::<c_void>(),
        (&mut projection_count_arg as *mut u32).cast::<c_void>(),
        (&mut projection_i64_mask_arg as *mut u32).cast::<c_void>(),
        (&mut keys_arg as *mut u64).cast::<c_void>(),
        (&mut key_count_arg as *mut u32).cast::<c_void>(),
        (&mut read_snapshot_arg as *mut u64).cast::<c_void>(),
        (&mut values_arg as *mut u64).cast::<c_void>(),
        (&mut status_arg as *mut u64).cast::<c_void>(),
    ];
    let stream_owned = PooledStreamOwned {
        primary: Arc::clone(&primary),
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let stream = stream_owned
        .pooled
        .as_ref()
        .expect("compound probe stream just leased")
        .stream;
    let threads = 128_u32;
    let blocks = key_count.div_ceil(threads);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) })?;
    let mut values = vec![0_u64; output_cells];
    let mut status = vec![0_u8; status_bytes];
    check_cuda(unsafe {
        (primary.cu_memcpy_dtoh)(
            values.as_mut_ptr().cast::<c_void>(),
            values_guard.ptr,
            output_bytes,
        )
    })?;
    check_cuda(unsafe {
        (primary.cu_memcpy_dtoh)(
            status.as_mut_ptr().cast::<c_void>(),
            status_guard.ptr,
            status_bytes,
        )
    })?;
    if status.iter().any(|value| !matches!(*value, 1..=3)) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            status
                .iter()
                .copied()
                .find(|value| !matches!(*value, 1..=3))
                .unwrap_or(0) as usize,
        ));
    }
    Ok(CudaI32I64PointBatchProjection {
        values,
        projection_kinds: plan.projection_kinds.clone(),
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_key_abi_and_plan_charge_are_checked() {
        assert_eq!(std::mem::size_of::<CudaI32I64PointKey>(), 12);
        let key = CudaI32I64PointKey::new(-7, 0x1122_3344_5566_7788);
        assert_eq!(key.words, [-7, 0x5566_7788, 0x1122_3344]);
        assert_eq!(
            CudaI32I64MultiShardProbePlan::estimated_allocated_bytes(4, 200).unwrap(),
            5_248
        );
        assert!(CudaI32I64MultiShardProbePlan::estimated_allocated_bytes(0, 0).is_err());
        assert!(CudaI32I64MultiShardProbePlan::estimated_allocated_bytes(0, 1).is_err());
        assert!(validate_plan_read_snapshot(12, 11).is_err());
        validate_plan_read_snapshot(12, 12).unwrap();
        validate_plan_read_snapshot(12, 13).unwrap();
    }
}
