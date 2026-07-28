use std::{os::raw::c_void, sync::Arc};

use crate::{
    check_cuda, CudaCompoundFoldColumn, CudaResidentDeviceMemory, CudaResidentReadSource,
    CudaRuntimeProbeError,
};

const COMPOUND_FOLD_VALIDITY_WIDTH: u32 = u32::MAX - 1;

fn validate_index_fold_columns(
    columns: &[CudaCompoundFoldColumn],
) -> Result<(), CudaRuntimeProbeError> {
    let mut data_columns = 0usize;
    let mut saw_validity = false;
    let mut validity_offsets = Vec::new();
    for column in columns {
        match column {
            CudaCompoundFoldColumn::Validity { bitmap_byte_offset } => {
                saw_validity = true;
                if validity_offsets.contains(bitmap_byte_offset) {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(columns.len()));
                }
                validity_offsets.push(*bitmap_byte_offset);
            }
            _ if saw_validity => {
                // The raw-key fast path relies on the first descriptor being data. Keeping every
                // predicate descriptor as a suffix also makes the device ABI canonical.
                return Err(CudaRuntimeProbeError::InvalidInputLength(columns.len()));
            }
            _ => data_columns += 1,
        }
    }
    if data_columns == 0 || validity_offsets.len() > data_columns {
        return Err(CudaRuntimeProbeError::InvalidInputLength(columns.len()));
    }
    Ok(())
}

/// Bytes occupied by the open-addressed key directory. The posting/version links immediately
/// follow this prefix in the same allocation.
pub fn resident_index_hash_bytes(table_mask: u32) -> Option<u64> {
    u64::from(table_mask)
        .checked_add(1)?
        .checked_mul(std::mem::size_of::<u64>() as u64)
}

/// Exact allocation for one resident index: one 64-bit directory slot per distinct-key bucket,
/// followed by one 32-bit previous-version link per physical row of shard capacity. Directory
/// heads reserve bit 31 of their low word as the singleton/no-link marker, so physical row ids
/// must fit in the remaining 31 bits (the engine's stricter 1 << 30 shard cap normally wins).
pub fn resident_index_allocated_bytes(table_mask: u32, row_capacity: u64) -> Option<u64> {
    if row_capacity >= (1_u64 << 31) {
        return None;
    }
    resident_index_hash_bytes(table_mask)?
        .checked_add(row_capacity.checked_mul(std::mem::size_of::<u32>() as u64)?)
}

// R3-002 / PRODUCT-002: construct or extend a resident key directory directly from resident typed
// columns. Each distinct raw key/fingerprint owns one open-addressed slot. Physical duplicates and
// MVCC versions form a row-addressed posting chain behind that slot, so the 256-probe bound applies
// only to DISTINCT hash collisions, never to a hot key's version count. Descriptor arrays and the
// four-byte verdict are control-plane data; keys, fingerprints, deleted stamps, and postings never
// cross the host.
const RESIDENT_INDEX_BUILD_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_typed_index_build(
    .param .u64 base_ptr,
    .param .u64 offsets_ptr,
    .param .u64 widths_ptr,
    .param .u64 blob_offsets_ptr,
    .param .u64 blob_lens_ptr,
    .param .u32 ncols,
    .param .u32 base_row,
    .param .u32 row_count,
    .param .u64 index_ptr,
    .param .u64 next_ptr,
    .param .u32 table_mask,
    .param .u32 hash_shift,
    .param .u64 deleted_ptr,
    .param .u64 gc_boundary,
    .param .u32 dup_tolerant,
    .param .u64 decline_ptr
)
{
    .reg .pred %p<18>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<56>;

    ld.param.u64 %rd1, [base_ptr];
    ld.param.u64 %rd2, [offsets_ptr];
    ld.param.u64 %rd3, [widths_ptr];
    ld.param.u64 %rd20, [blob_offsets_ptr];
    ld.param.u64 %rd31, [blob_lens_ptr];
    ld.param.u32 %r1, [ncols];
    ld.param.u32 %r40, [base_row];
    ld.param.u32 %r2, [row_count];
    ld.param.u64 %rd4, [index_ptr];
    ld.param.u64 %rd53, [next_ptr];
    ld.param.u32 %r30, [table_mask];
    ld.param.u32 %r31, [hash_shift];
    ld.param.u64 %rd40, [deleted_ptr];
    ld.param.u64 %rd41, [gc_boundary];
    ld.param.u32 %r32, [dup_tolerant];
    ld.param.u64 %rd42, [decline_ptr];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;
    add.u32 %r40, %r40, %r6;

    // A missing deleted_by region means all-live. Otherwise omit only rows dead no later than the
    // oldest active snapshot; newer tombstones remain indexed for old pinned readers.
    setp.eq.u64 %p10, %rd40, 0;
    @%p10 bra VALIDITYSTART;
    mul.wide.u32 %rd43, %r40, 8;
    add.u64 %rd44, %rd40, %rd43;
    ld.global.u64 %rd45, [%rd44];
    setp.le.u64 %p11, %rd45, %rd41;
    @%p11 bra DONE;

    // Validity descriptors carry the reserved width 0xfffffffe and are appended after the typed
    // key descriptors. They are predicates, not key material: one NULL in a compound key omits the
    // complete row from the directory (SQL NULLS DISTINCT). Count only data descriptors when
    // selecting the raw single-i32 ABI.
VALIDITYSTART:
    mov.u32 %r45, 0;
    mov.u32 %r46, 0;
VALIDITYLOOP:
    setp.ge.u32 %p16, %r45, %r1;
    @%p16 bra KEYMODE;
    mul.wide.u32 %rd5, %r45, 4;
    add.u64 %rd6, %rd3, %rd5;
    ld.global.u32 %r9, [%rd6];
    setp.eq.u32 %p17, %r9, 4294967294;
    @!%p17 bra VALIDDATA;
    mul.wide.u32 %rd5, %r45, 8;
    add.u64 %rd6, %rd2, %rd5;
    ld.global.u64 %rd7, [%rd6];
    shr.u32 %r19, %r40, 3;
    and.b32 %r20, %r40, 7;
    add.u64 %rd35, %rd1, %rd7;
    cvt.u64.u32 %rd38, %r19;
    add.u64 %rd35, %rd35, %rd38;
    ld.global.u8 %r21, [%rd35];
    shr.u32 %r21, %r21, %r20;
    and.b32 %r21, %r21, 1;
    setp.eq.u32 %p17, %r21, 0;
    @%p17 bra DONE;
    bra VALIDNEXT;
VALIDDATA:
    add.u32 %r46, %r46, 1;
VALIDNEXT:
    add.u32 %r45, %r45, 1;
    bra VALIDITYLOOP;

KEYMODE:
    // The unflagged single i32/date/int2 ABI stores the resident word verbatim. Every other shape
    // (multi-column, wide fixed, BOOL, TEXT) uses the canonical FNV/rotate fingerprint below.
    setp.ne.u32 %p12, %r46, 1;
    @%p12 bra FOLDINIT;
    ld.global.u32 %r33, [%rd3];
    setp.ne.u32 %p12, %r33, 1;
    @%p12 bra FOLDINIT;
    ld.global.u64 %rd7, [%rd2];
    mul.wide.u32 %rd10, %r40, 4;
    add.u64 %rd11, %rd1, %rd7;
    add.u64 %rd12, %rd11, %rd10;
    ld.global.u32 %r7, [%rd12];
    bra INSERT;

FOLDINIT:
    mov.u32 %r7, 2166136261;
    mov.u32 %r8, 0;

FOLDLOOP:
    mul.wide.u32 %rd5, %r8, 8;
    add.u64 %rd6, %rd2, %rd5;
    ld.global.u64 %rd7, [%rd6];
    mul.wide.u32 %rd8, %r8, 4;
    add.u64 %rd9, %rd3, %rd8;
    ld.global.u32 %r9, [%rd9];
    setp.eq.u32 %p4, %r9, 4294967294;
    @%p4 bra NEXTCOL;
    setp.eq.u32 %p4, %r9, 0;
    @%p4 bra TEXTCOL;
    setp.eq.u32 %p4, %r9, 4294967295;
    @%p4 bra BOOLCOL;
    mul.lo.u32 %r10, %r40, %r9;
    mul.wide.u32 %rd10, %r10, 4;
    add.u64 %rd11, %rd1, %rd7;
    add.u64 %rd12, %rd11, %rd10;
    mov.u32 %r11, 0;

WORDLOOP:
    mul.wide.u32 %rd13, %r11, 4;
    add.u64 %rd14, %rd12, %rd13;
    ld.global.u32 %r12, [%rd14];
    xor.b32 %r7, %r7, %r12;
    mul.lo.u32 %r7, %r7, 16777619;
    shl.b32 %r13, %r7, 13;
    shr.b32 %r14, %r7, 19;
    or.b32 %r7, %r13, %r14;
    add.u32 %r7, %r7, 2654435761;
    add.u32 %r11, %r11, 1;
    setp.lt.u32 %p3, %r11, %r9;
    @%p3 bra WORDLOOP;
    bra NEXTCOL;

BOOLCOL:
    shr.u32 %r19, %r40, 3;
    and.b32 %r20, %r40, 7;
    add.u64 %rd35, %rd1, %rd7;
    cvt.u64.u32 %rd38, %r19;
    add.u64 %rd35, %rd35, %rd38;
    ld.global.u8 %r21, [%rd35];
    shr.u32 %r21, %r21, %r20;
    and.b32 %r21, %r21, 1;
    xor.b32 %r7, %r7, %r21;
    mul.lo.u32 %r7, %r7, 16777619;
    shl.b32 %r13, %r7, 13;
    shr.b32 %r14, %r7, 19;
    or.b32 %r7, %r13, %r14;
    add.u32 %r7, %r7, 2654435761;
    bra NEXTCOL;

TEXTCOL:
    add.u64 %rd21, %rd1, %rd7;
    mul.wide.u32 %rd22, %r40, 8;
    add.u64 %rd23, %rd21, %rd22;
    ld.global.u64 %rd24, [%rd23];
    ld.global.u64 %rd25, [%rd23+8];
    add.u64 %rd33, %rd31, %rd5;
    ld.global.u64 %rd34, [%rd33];
    setp.gt.u64 %p8, %rd24, %rd25;
    @%p8 bra DECLINE;
    setp.gt.u64 %p9, %rd25, %rd34;
    @%p9 bra DECLINE;
    add.u64 %rd26, %rd20, %rd5;
    ld.global.u64 %rd27, [%rd26];
    add.u64 %rd28, %rd1, %rd27;
    mov.u32 %r15, 2166136261;
    mov.u64 %rd29, %rd24;
BYTELOOP:
    setp.ge.u64 %p5, %rd29, %rd25;
    @%p5 bra BYTEDONE;
    add.u64 %rd30, %rd28, %rd29;
    ld.global.u8 %r16, [%rd30];
    xor.b32 %r15, %r15, %r16;
    mul.lo.u32 %r15, %r15, 16777619;
    add.u64 %rd29, %rd29, 1;
    bra BYTELOOP;
BYTEDONE:
    xor.b32 %r7, %r7, %r15;
    mul.lo.u32 %r7, %r7, 16777619;
    shl.b32 %r13, %r7, 13;
    shr.b32 %r14, %r7, 19;
    or.b32 %r7, %r13, %r14;
    add.u32 %r7, %r7, 2654435761;

NEXTCOL:
    add.u32 %r8, %r8, 1;
    setp.lt.u32 %p2, %r8, %r1;
    @%p2 bra FOLDLOOP;

INSERT:
    add.u32 %r34, %r40, 1;
    or.b32 %r42, %r34, 2147483648;
    cvt.u64.u32 %rd46, %r42;
    cvt.u64.u32 %rd47, %r7;
    shl.b64 %rd47, %rd47, 32;
    or.b64 %rd47, %rd47, %rd46;
    mul.lo.u32 %r35, %r7, 2654435761;
    shr.u32 %r36, %r35, %r31;
    and.b32 %r36, %r36, %r30;
    mov.u32 %r37, 0;

PROBE:
    mul.wide.u32 %rd48, %r36, 8;
    add.u64 %rd49, %rd4, %rd48;
    mov.u64 %rd50, 0;
    atom.global.cas.b64 %rd51, [%rd49], %rd50, %rd47;
    setp.eq.u64 %p6, %rd51, 0;
    @%p6 bra DONE;
    shr.u64 %rd52, %rd51, 32;
    cvt.u32.u64 %r38, %rd52;
    setp.ne.u32 %p14, %r38, %r7;
    @%p14 bra ADVANCE;
    setp.eq.u32 %p13, %r32, 0;
    @%p13 bra DECLINE;

    // Same key/fingerprint: publish this physical row as the new posting head. Store its previous
    // link before the slot CAS; a losing concurrent inserter retries against the newer head.
    cvt.u32.u64 %r41, %rd51;
    and.b32 %r41, %r41, 2147483647;
    mul.wide.u32 %rd54, %r40, 4;
    add.u64 %rd54, %rd53, %rd54;
    st.global.u32 [%rd54], %r41;
    membar.gl;
    cvt.u64.u32 %rd46, %r34;
    cvt.u64.u32 %rd47, %r7;
    shl.b64 %rd47, %rd47, 32;
    or.b64 %rd47, %rd47, %rd46;
    atom.global.cas.b64 %rd55, [%rd49], %rd51, %rd47;
    setp.eq.u64 %p15, %rd55, %rd51;
    @%p15 bra POSTED;
    bra PROBE;

POSTED:
    mov.u32 %r43, 2;
    atom.global.or.b32 %r44, [%rd42], %r43;
    bra DONE;

ADVANCE:
    add.u32 %r36, %r36, 1;
    and.b32 %r36, %r36, %r30;
    add.u32 %r37, %r37, 1;
    setp.ge.u32 %p7, %r37, 256;
    @%p7 bra DECLINE;
    bra PROBE;

DECLINE:
    mov.u32 %r39, 1;
    atom.global.or.b32 %r47, [%rd42], %r39;

DONE:
    ret;
}
"#;

/// One existing resident index maintained by the multi-index tail-insert kernel.
#[derive(Debug, Clone)]
pub struct CudaResidentTypedIndexInsert {
    pub index: Arc<CudaResidentDeviceMemory>,
    pub table_mask: u32,
    pub hash_shift: u32,
    pub columns: Vec<CudaCompoundFoldColumn>,
}

// PRODUCT-002 mutation fanout: one thread owns one (index, appended-row) pair. All named compound
// indexes fold their ordered typed columns and prepend their posting in ONE launch. A compact
// descriptor image is the sole H2D transfer and the shared verdict is the sole D2H transfer.
const RESIDENT_MULTI_INDEX_INSERT_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_typed_multi_index_insert(
    .param .u64 base_ptr,
    .param .u64 descriptors_ptr,
    .param .u32 index_count,
    .param .u32 base_row,
    .param .u32 row_count,
    .param .u64 decline_ptr
)
{
    .reg .pred %p<14>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<56>;

    ld.param.u64 %rd1, [base_ptr];
    ld.param.u64 %rd2, [descriptors_ptr];
    ld.param.u32 %r1, [index_count];
    ld.param.u32 %r2, [base_row];
    ld.param.u32 %r3, [row_count];
    ld.param.u64 %rd3, [decline_ptr];

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mad.lo.u32 %r7, %r5, %r6, %r4;
    mul.lo.u32 %r8, %r1, %r3;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra DONE;
    div.u32 %r9, %r7, %r3;
    rem.u32 %r10, %r7, %r3;
    add.u32 %r11, %r2, %r10;

    // Index descriptor = [index_ptr, mask|shift, column_start|column_count].
    mul.wide.u32 %rd4, %r9, 24;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.u64 %rd6, [%rd5];
    ld.global.u64 %rd7, [%rd5+8];
    ld.global.u64 %rd8, [%rd5+16];
    cvt.u32.u64 %r12, %rd7;
    shr.u64 %rd9, %rd7, 32;
    cvt.u32.u64 %r13, %rd9;
    cvt.u32.u64 %r14, %rd8;
    shr.u64 %rd10, %rd8, 32;
    cvt.u32.u64 %r15, %rd10;
    mul.wide.u32 %rd11, %r1, 24;
    add.u64 %rd12, %rd2, %rd11;

    // Validity descriptors (reserved width 0xfffffffe) are predicates, not key material. Omit the
    // index/row work item when any indexed column is NULL and count only data descriptors when
    // choosing the raw single-i32 ABI.
    mov.u32 %r43, 0;
    mov.u32 %r44, 0;
VALIDITYLOOP:
    setp.ge.u32 %p13, %r43, %r15;
    @%p13 bra KEYMODE;
    add.u32 %r45, %r14, %r43;
    mul.wide.u32 %rd44, %r45, 32;
    add.u64 %rd45, %rd12, %rd44;
    ld.global.u64 %rd46, [%rd45+8];
    cvt.u32.u64 %r46, %rd46;
    setp.eq.u32 %p13, %r46, 4294967294;
    @!%p13 bra VALIDDATA;
    ld.global.u64 %rd46, [%rd45];
    shr.u32 %r26, %r11, 3;
    and.b32 %r27, %r11, 7;
    add.u64 %rd47, %rd1, %rd46;
    cvt.u64.u32 %rd48, %r26;
    add.u64 %rd47, %rd47, %rd48;
    ld.global.u8 %r28, [%rd47];
    shr.u32 %r28, %r28, %r27;
    and.b32 %r28, %r28, 1;
    setp.eq.u32 %p13, %r28, 0;
    @%p13 bra DONE;
    bra VALIDNEXT;
VALIDDATA:
    add.u32 %r44, %r44, 1;
VALIDNEXT:
    add.u32 %r43, %r43, 1;
    bra VALIDITYLOOP;

KEYMODE:
    // One raw i32 column retains the established key ABI. All other shapes use the canonical fold.
    setp.ne.u32 %p2, %r44, 1;
    @%p2 bra FOLDINIT;
    mul.wide.u32 %rd13, %r14, 32;
    add.u64 %rd14, %rd12, %rd13;
    ld.global.u64 %rd15, [%rd14+8];
    cvt.u32.u64 %r16, %rd15;
    setp.ne.u32 %p2, %r16, 1;
    @%p2 bra FOLDINIT;
    ld.global.u64 %rd16, [%rd14];
    mul.wide.u32 %rd17, %r11, 4;
    add.u64 %rd18, %rd1, %rd16;
    add.u64 %rd18, %rd18, %rd17;
    ld.global.u32 %r17, [%rd18];
    bra INSERT;

FOLDINIT:
    mov.u32 %r17, 2166136261;
    mov.u32 %r18, 0;

FOLDLOOP:
    add.u32 %r19, %r14, %r18;
    mul.wide.u32 %rd13, %r19, 32;
    add.u64 %rd14, %rd12, %rd13;
    ld.global.u64 %rd16, [%rd14];
    ld.global.u64 %rd15, [%rd14+8];
    cvt.u32.u64 %r20, %rd15;
    setp.eq.u32 %p3, %r20, 4294967294;
    @%p3 bra NEXTCOL;
    setp.eq.u32 %p3, %r20, 0;
    @%p3 bra TEXTCOL;
    setp.eq.u32 %p3, %r20, 4294967295;
    @%p3 bra BOOLCOL;
    mul.lo.u32 %r21, %r11, %r20;
    mul.wide.u32 %rd17, %r21, 4;
    add.u64 %rd18, %rd1, %rd16;
    add.u64 %rd18, %rd18, %rd17;
    mov.u32 %r22, 0;
WORDLOOP:
    mul.wide.u32 %rd19, %r22, 4;
    add.u64 %rd20, %rd18, %rd19;
    ld.global.u32 %r23, [%rd20];
    xor.b32 %r17, %r17, %r23;
    mul.lo.u32 %r17, %r17, 16777619;
    shl.b32 %r24, %r17, 13;
    shr.b32 %r25, %r17, 19;
    or.b32 %r17, %r24, %r25;
    add.u32 %r17, %r17, 2654435761;
    add.u32 %r22, %r22, 1;
    setp.lt.u32 %p4, %r22, %r20;
    @%p4 bra WORDLOOP;
    bra NEXTCOL;

BOOLCOL:
    shr.u32 %r26, %r11, 3;
    and.b32 %r27, %r11, 7;
    add.u64 %rd21, %rd1, %rd16;
    cvt.u64.u32 %rd22, %r26;
    add.u64 %rd21, %rd21, %rd22;
    ld.global.u8 %r28, [%rd21];
    shr.u32 %r28, %r28, %r27;
    and.b32 %r28, %r28, 1;
    xor.b32 %r17, %r17, %r28;
    mul.lo.u32 %r17, %r17, 16777619;
    shl.b32 %r24, %r17, 13;
    shr.b32 %r25, %r17, 19;
    or.b32 %r17, %r24, %r25;
    add.u32 %r17, %r17, 2654435761;
    bra NEXTCOL;

TEXTCOL:
    add.u64 %rd23, %rd1, %rd16;
    mul.wide.u32 %rd24, %r11, 8;
    add.u64 %rd25, %rd23, %rd24;
    ld.global.u64 %rd26, [%rd25];
    ld.global.u64 %rd27, [%rd25+8];
    ld.global.u64 %rd28, [%rd14+24];
    setp.gt.u64 %p5, %rd26, %rd27;
    @%p5 bra DECLINE;
    setp.gt.u64 %p6, %rd27, %rd28;
    @%p6 bra DECLINE;
    ld.global.u64 %rd29, [%rd14+16];
    add.u64 %rd30, %rd1, %rd29;
    mov.u32 %r29, 2166136261;
    mov.u64 %rd31, %rd26;
BYTELOOP:
    setp.ge.u64 %p7, %rd31, %rd27;
    @%p7 bra BYTEDONE;
    add.u64 %rd32, %rd30, %rd31;
    ld.global.u8 %r30, [%rd32];
    xor.b32 %r29, %r29, %r30;
    mul.lo.u32 %r29, %r29, 16777619;
    add.u64 %rd31, %rd31, 1;
    bra BYTELOOP;
BYTEDONE:
    xor.b32 %r17, %r17, %r29;
    mul.lo.u32 %r17, %r17, 16777619;
    shl.b32 %r24, %r17, 13;
    shr.b32 %r25, %r17, 19;
    or.b32 %r17, %r24, %r25;
    add.u32 %r17, %r17, 2654435761;

NEXTCOL:
    add.u32 %r18, %r18, 1;
    setp.lt.u32 %p8, %r18, %r15;
    @%p8 bra FOLDLOOP;

INSERT:
    add.u32 %r31, %r11, 1;
    or.b32 %r40, %r31, 2147483648;
    cvt.u64.u32 %rd33, %r40;
    cvt.u64.u32 %rd34, %r17;
    shl.b64 %rd34, %rd34, 32;
    or.b64 %rd34, %rd34, %rd33;
    mul.lo.u32 %r32, %r17, 2654435761;
    shr.u32 %r33, %r32, %r13;
    and.b32 %r33, %r33, %r12;
    add.u32 %r34, %r12, 1;
    cvt.u64.u32 %rd35, %r34;
    shl.b64 %rd35, %rd35, 3;
    add.u64 %rd35, %rd6, %rd35;
    mov.u32 %r35, 0;

PROBE:
    mul.wide.u32 %rd36, %r33, 8;
    add.u64 %rd37, %rd6, %rd36;
    mov.u64 %rd38, 0;
    atom.global.cas.b64 %rd39, [%rd37], %rd38, %rd34;
    setp.eq.u64 %p9, %rd39, 0;
    @%p9 bra DONE;
    shr.u64 %rd40, %rd39, 32;
    cvt.u32.u64 %r36, %rd40;
    setp.ne.u32 %p10, %r36, %r17;
    @%p10 bra ADVANCE;
    cvt.u32.u64 %r37, %rd39;
    and.b32 %r37, %r37, 2147483647;
    mul.wide.u32 %rd41, %r11, 4;
    add.u64 %rd42, %rd35, %rd41;
    st.global.u32 [%rd42], %r37;
    membar.gl;
    cvt.u64.u32 %rd33, %r31;
    cvt.u64.u32 %rd34, %r17;
    shl.b64 %rd34, %rd34, 32;
    or.b64 %rd34, %rd34, %rd33;
    atom.global.cas.b64 %rd43, [%rd37], %rd39, %rd34;
    setp.eq.u64 %p11, %rd43, %rd39;
    @%p11 bra POSTED;
    bra PROBE;

POSTED:
    mov.u32 %r41, 2;
    atom.global.or.b32 %r42, [%rd3], %r41;
    bra DONE;

ADVANCE:
    add.u32 %r33, %r33, 1;
    and.b32 %r33, %r33, %r12;
    add.u32 %r35, %r35, 1;
    setp.ge.u32 %p12, %r35, 256;
    @%p12 bra DECLINE;
    bra PROBE;

DECLINE:
    mov.u32 %r38, 1;
    atom.global.or.b32 %r39, [%rd3], %r38;

DONE:
    ret;
}
"#;

/// Four-byte device verdict shared by resident index build/append kernels. `created_posting` reports
/// whether this operation published at least one posting link. A zero-based build can therefore use it
/// as the allocation's initial state; incremental callers must monotonically OR it into retained state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaResidentIndexStatus {
    pub declined: bool,
    pub created_posting: bool,
}

impl CudaResidentIndexStatus {
    pub(crate) fn from_bits(bits: u32) -> Self {
        Self {
            declined: bits & 1 != 0,
            created_posting: bits & 2 != 0,
        }
    }
}

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

/// Drain queued default-stream work before its pooled buffers return; an unsubmitted token keeps
/// this armed guard so descriptor setup cannot outlive its leases.
struct NullStreamDrain {
    primary: Arc<crate::GpuPrimaryContext>,
    armed: bool,
}

impl Drop for NullStreamDrain {
    fn drop(&mut self) {
        if self.armed {
            #[cfg(test)]
            PREPARED_MULTI_INDEX_DRAINS.with(|drains| drains.set(drains.get() + 1));
            let _ = self.primary.set_current();
            unsafe {
                let _ = (self.primary.cu_stream_synchronize)(std::ptr::null_mut());
            }
        }
    }
}

/// Exact pre-WAL pooled bytes: three u64 words/index, four words/typed-or-validity column, and
/// the separately bucketed four-byte terminal; this excludes resident source/index allocations.
pub fn resident_typed_indexes_insert_preparation_bytes(
    index_count: usize,
    descriptor_column_count: usize,
) -> Option<u64> {
    let descriptor_words = index_count
        .checked_mul(3)?
        .checked_add(descriptor_column_count.checked_mul(4)?)?;
    let descriptor_bytes = descriptor_words.checked_mul(std::mem::size_of::<u64>())?;
    let descriptor_pool = crate::cuda_context::checked_output_buffer_bucket(descriptor_bytes)?;
    let verdict_pool =
        crate::cuda_context::checked_output_buffer_bucket(std::mem::size_of::<u32>())?;
    u64::try_from(descriptor_pool)
        .ok()?
        .checked_add(u64::try_from(verdict_pool).ok()?)
}

#[cfg(test)]
thread_local! {
    static PREPARED_MULTI_INDEX_PREPARES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static PREPARED_MULTI_INDEX_SUBMITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static PREPARED_MULTI_INDEX_DRAINS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static FAIL_PREPARED_MULTI_INDEX_AFTER_LEASES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_PREPARED_MULTI_INDEX_AFTER_LAUNCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PreparedResidentTypedIndexesInsertCounters {
    pub prepares: u64,
    pub submits: u64,
    pub drains: u64,
}

#[cfg(test)]
pub(crate) fn prepared_resident_typed_indexes_insert_counters(
) -> PreparedResidentTypedIndexesInsertCounters {
    PreparedResidentTypedIndexesInsertCounters {
        prepares: PREPARED_MULTI_INDEX_PREPARES.with(std::cell::Cell::get),
        submits: PREPARED_MULTI_INDEX_SUBMITS.with(std::cell::Cell::get),
        drains: PREPARED_MULTI_INDEX_DRAINS.with(std::cell::Cell::get),
    }
}

/// Fail immediately after both pooled leases have been acquired.  The next prepare consumes this
/// arm, proving that an abandoned pre-WAL token drains its leases and leaves the shared context
/// reusable before any launch can occur.
#[cfg(test)]
pub(crate) fn fail_next_prepared_resident_typed_indexes_insert_after_leases() {
    FAIL_PREPARED_MULTI_INDEX_AFTER_LEASES.with(|fail| fail.set(true));
}

/// Inject a return after a successful fused launch but before its terminal D2H.  The token's
/// drain guard must synchronize before its pooled leases can return to the shared pool.
#[cfg(test)]
pub(crate) fn fail_next_prepared_resident_typed_indexes_insert_after_launch() {
    FAIL_PREPARED_MULTI_INDEX_AFTER_LAUNCH.with(|fail| fail.set(true));
}

#[cfg(test)]
fn take_fail_after_prepared_multi_index_launch() -> bool {
    FAIL_PREPARED_MULTI_INDEX_AFTER_LAUNCH.with(|fail| fail.replace(false))
}

/// Move-only launch ownership for one fused resident-index tail insert.
///
/// Preparation validates every index basis, resolves the cached kernel, queues the compact
/// descriptor image and verdict initialization, and leases both device buffers. Submission can
/// therefore cross a durability boundary without allocating, reloading a module, looking up a
/// cache entry, or rebuilding host descriptors. Dropping an unsubmitted token drains setup before
/// returning its owned leases to the execution pool; it cannot mutate an index.
pub struct PreparedResidentTypedIndexesInsert {
    primary: Arc<crate::GpuPrimaryContext>,
    /// Strong allocation guards keep both source and every destination index alive even if their
    /// cache entries are retired after preparation and before the consuming launch.
    _source_owner: Arc<crate::resident_memory::CudaResidentDeviceAllocation>,
    _index_owners: Box<[Arc<crate::resident_memory::CudaResidentDeviceAllocation>]>,
    // Field order is load-bearing: drain queued null-stream setup before either pooled buffer
    // returns to the shared pool on an abandoned token or an error path.
    preparation_drain: Option<NullStreamDrain>,
    descriptor_guard: crate::PooledDeviceBufferOwned,
    decline_guard: crate::PooledDeviceBufferOwned,
    function: *mut c_void,
    source_ptr: u64,
    index_count: u32,
    base_row: u32,
    row_count: u32,
    work_items: u32,
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cu_launch_kernel: CuLaunchKernel,
}

// SAFETY: exclusive allocations/leases and `primary` pin the module function; submit/drop rebind
// that context before CUDA use. The single-consumption token is intentionally not `Sync`.
unsafe impl Send for PreparedResidentTypedIndexesInsert {}

impl PreparedResidentTypedIndexesInsert {
    /// Exact pooled descriptor + terminal capacity held across the WAL boundary.
    pub fn preparation_bytes(&self) -> u64 {
        u64::try_from(self.descriptor_guard.capacity)
            .unwrap_or(u64::MAX)
            .saturating_add(u64::try_from(self.decline_guard.capacity).unwrap_or(u64::MAX))
    }

    /// Consume the sealed launch exactly once.  All allocation and module-cache work happened
    /// during preparation; this performs only the kernel launch and its bounded four-byte D2H
    /// status fence.
    pub fn submit(mut self) -> Result<CudaResidentIndexStatus, CudaRuntimeProbeError> {
        #[cfg(test)]
        PREPARED_MULTI_INDEX_SUBMITS.with(|submits| submits.set(submits.get() + 1));
        self.primary.set_current()?;
        let mut stream_drain = self
            .preparation_drain
            .take()
            .expect("prepared resident index launch retains its drain guard");
        let mut source_arg = self.source_ptr;
        let mut descriptors_arg = self.descriptor_guard.ptr;
        let mut index_count_arg = self.index_count;
        let mut base_row_arg = self.base_row;
        let mut row_count_arg = self.row_count;
        let mut decline_arg = self.decline_guard.ptr;
        let mut args = [
            (&mut source_arg as *mut u64).cast::<c_void>(),
            (&mut descriptors_arg as *mut u64).cast::<c_void>(),
            (&mut index_count_arg as *mut u32).cast::<c_void>(),
            (&mut base_row_arg as *mut u32).cast::<c_void>(),
            (&mut row_count_arg as *mut u32).cast::<c_void>(),
            (&mut decline_arg as *mut u64).cast::<c_void>(),
        ];
        let threads = 128_u32;
        check_cuda(unsafe {
            (self.cu_launch_kernel)(
                self.function,
                self.work_items.div_ceil(threads),
                1,
                1,
                threads,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;
        #[cfg(test)]
        if take_fail_after_prepared_multi_index_launch() {
            return Err(CudaRuntimeProbeError::KernelLaunchFailed(-1));
        }
        // Blocking null-stream D2H is the completion fence before owned leases return to pool.
        let mut decline = 0_u32;
        check_cuda(unsafe {
            (self.cu_memcpy_dtoh)(
                (&mut decline as *mut u32).cast::<c_void>(),
                self.decline_guard.ptr,
                std::mem::size_of::<u32>(),
            )
        })?;
        stream_drain.armed = false;
        Ok(CudaResidentIndexStatus::from_bits(decline))
    }
}

impl CudaResidentDeviceMemory {
    /// Build `index` from typed columns in this resident allocation. Returns `true` only for a
    /// semantic decline (duplicate in a non-tolerant index, malformed text offsets, or probe-cap
    /// exhaustion); CUDA/input failures are errors so callers can retry rather than cache decline.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_resident_typed_index_build(
        &self,
        index: &CudaResidentDeviceMemory,
        table_mask: u32,
        hash_shift: u32,
        columns: &[CudaCompoundFoldColumn],
        row_count: usize,
        deleted_by: Option<&CudaResidentDeviceMemory>,
        gc_boundary: u64,
        dup_tolerant: bool,
    ) -> Result<bool, CudaRuntimeProbeError> {
        Ok(self
            .submit_resident_typed_index_build_status(
                index,
                table_mask,
                hash_shift,
                columns,
                row_count,
                deleted_by,
                gc_boundary,
                dup_tolerant,
            )?
            .declined)
    }

    /// Status-bearing form of [`Self::submit_resident_typed_index_build`]. It reuses the existing
    /// four-byte verdict readback to report whether this zero-based build created posting chains.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_resident_typed_index_build_status(
        &self,
        index: &CudaResidentDeviceMemory,
        table_mask: u32,
        hash_shift: u32,
        columns: &[CudaCompoundFoldColumn],
        row_count: usize,
        deleted_by: Option<&CudaResidentDeviceMemory>,
        gc_boundary: u64,
        dup_tolerant: bool,
    ) -> Result<CudaResidentIndexStatus, CudaRuntimeProbeError> {
        self.submit_resident_typed_index_range(
            index,
            table_mask,
            hash_shift,
            columns,
            0,
            row_count,
            deleted_by,
            gc_boundary,
            dup_tolerant,
        )
    }

    /// Insert the appended resident row range into an existing index without host key folding or
    /// key H2D. The source column descriptors address this resident shard; only descriptors and the
    /// four-byte decline verdict cross the host.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_resident_typed_index_insert(
        &self,
        index: &CudaResidentDeviceMemory,
        table_mask: u32,
        hash_shift: u32,
        columns: &[CudaCompoundFoldColumn],
        base_row: usize,
        row_count: usize,
        dup_tolerant: bool,
    ) -> Result<bool, CudaRuntimeProbeError> {
        Ok(self
            .submit_resident_typed_index_insert_status(
                index,
                table_mask,
                hash_shift,
                columns,
                base_row,
                row_count,
                dup_tolerant,
            )?
            .declined)
    }

    /// Status-bearing form of [`Self::submit_resident_typed_index_insert`].
    #[allow(clippy::too_many_arguments)]
    pub fn submit_resident_typed_index_insert_status(
        &self,
        index: &CudaResidentDeviceMemory,
        table_mask: u32,
        hash_shift: u32,
        columns: &[CudaCompoundFoldColumn],
        base_row: usize,
        row_count: usize,
        dup_tolerant: bool,
    ) -> Result<CudaResidentIndexStatus, CudaRuntimeProbeError> {
        self.submit_resident_typed_index_range(
            index,
            table_mask,
            hash_shift,
            columns,
            base_row,
            row_count,
            None,
            0,
            dup_tolerant,
        )
    }

    /// Maintain all supplied named indexes for one appended resident row range in a single GPU
    /// launch. This is the mutation-fanout path for compound/wide indexes: no host key vectors and
    /// no per-index launch/readback loop.
    pub fn submit_resident_typed_indexes_insert(
        &self,
        indexes: &[CudaResidentTypedIndexInsert],
        base_row: usize,
        row_count: usize,
    ) -> Result<bool, CudaRuntimeProbeError> {
        Ok(self
            .submit_resident_typed_indexes_insert_status(indexes, base_row, row_count)?
            .declined)
    }

    /// Status-bearing fused fanout append. No extra launch or transfer is added; bit 1 shares the
    /// existing four-byte verdict with the bounded-probe decline bit.
    pub fn submit_resident_typed_indexes_insert_status(
        &self,
        indexes: &[CudaResidentTypedIndexInsert],
        base_row: usize,
        row_count: usize,
    ) -> Result<CudaResidentIndexStatus, CudaRuntimeProbeError> {
        self.prepare_resident_typed_indexes_insert(indexes, base_row, row_count)?
            .submit()
    }

    /// Preallocate and seal the one fused resident-index tail-insert launch.  The returned token
    /// pins the source plus every index owner, retains the uploaded descriptor and verdict leases,
    /// and resolves the kernel before the caller crosses WAL.  Its consuming `submit` has no
    /// allocation, module-cache lookup, or revalidation path.
    pub fn prepare_resident_typed_indexes_insert(
        &self,
        indexes: &[CudaResidentTypedIndexInsert],
        base_row: usize,
        row_count: usize,
    ) -> Result<PreparedResidentTypedIndexesInsert, CudaRuntimeProbeError> {
        if indexes.is_empty() || row_count == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(row_count));
        }
        let base_row_u32 = u32::try_from(base_row)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(base_row))?;
        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(row_count))?;
        let end_row = base_row
            .checked_add(row_count)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if end_row >= u32::MAX as usize {
            return Err(CudaRuntimeProbeError::InvalidInputLength(end_row));
        }
        let index_count_u32 = u32::try_from(indexes.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(indexes.len()))?;
        let work_items = index_count_u32
            .checked_mul(row_count_u32)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let primary = self.primary_arc();
        let index_descriptor_capacity = indexes
            .len()
            .checked_mul(3)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let column_count = indexes
            .iter()
            .try_fold(0usize, |total, index| {
                total.checked_add(index.columns.len())
            })
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let column_descriptor_capacity = column_count
            .checked_mul(4)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let mut index_descriptors = Vec::<u64>::with_capacity(index_descriptor_capacity);
        let mut column_descriptors = Vec::<u64>::with_capacity(column_descriptor_capacity);
        // The token pins all allocations until submit, so these device addresses are stable for
        // this validation interval.  Reject physical aliases here: logical key-id coalescing is
        // an upstream concern, while one fused launch must never use its source as a destination
        // or insert the same directory twice concurrently.
        let source_ptr = self.device_ptr();
        let mut destination_ptrs = std::collections::BTreeSet::new();
        let mut column_start = 0_u32;
        for request in indexes {
            if request.columns.is_empty()
                || !Arc::ptr_eq(&primary, &request.index.primary_arc())
                || request.index.device_ptr() == 0
                || request.index.device_ptr() == source_ptr
                || !destination_ptrs.insert(request.index.device_ptr())
            {
                return Err(CudaRuntimeProbeError::InvalidInputLength(0));
            }
            validate_index_fold_columns(&request.columns)?;
            let table_size = u64::from(request.table_mask) + 1;
            let expected_shift = 32_u32.checked_sub(table_size.trailing_zeros()).ok_or(
                CudaRuntimeProbeError::InvalidInputLength(table_size as usize),
            )?;
            let required = resident_index_allocated_bytes(request.table_mask, end_row as u64)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if table_size < 2
                || !table_size.is_power_of_two()
                || table_size > (1_u64 << 30)
                || request.hash_shift != expected_shift
                || request.index.metadata().allocated_bytes < required
            {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    usize::try_from(required).unwrap_or(usize::MAX),
                ));
            }
            let count = u32::try_from(request.columns.len())
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(request.columns.len()))?;
            index_descriptors.push(request.index.device_ptr());
            index_descriptors
                .push(u64::from(request.table_mask) | (u64::from(request.hash_shift) << 32));
            index_descriptors.push(u64::from(column_start) | (u64::from(count) << 32));
            column_start = column_start
                .checked_add(count)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

            for column in &request.columns {
                let (offset, width, blob_offset, blob_len, extent) = match *column {
                    CudaCompoundFoldColumn::Fixed {
                        byte_offset,
                        width_words,
                    } => {
                        if width_words == 0 || !byte_offset.is_multiple_of(4) {
                            return Err(CudaRuntimeProbeError::InvalidInputLength(
                                byte_offset as usize,
                            ));
                        }
                        let bytes = (end_row as u64)
                            .checked_mul(u64::from(width_words))
                            .and_then(|words| words.checked_mul(4))
                            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                        (
                            byte_offset,
                            u64::from(width_words),
                            0,
                            0,
                            byte_offset.checked_add(bytes),
                        )
                    }
                    CudaCompoundFoldColumn::Bool { bitmap_byte_offset } => (
                        bitmap_byte_offset,
                        u64::from(u32::MAX),
                        0,
                        0,
                        bitmap_byte_offset.checked_add(end_row.div_ceil(8) as u64),
                    ),
                    CudaCompoundFoldColumn::Validity { bitmap_byte_offset } => {
                        if !bitmap_byte_offset.is_multiple_of(4) {
                            return Err(CudaRuntimeProbeError::InvalidInputLength(
                                bitmap_byte_offset as usize,
                            ));
                        }
                        (
                            bitmap_byte_offset,
                            u64::from(COMPOUND_FOLD_VALIDITY_WIDTH),
                            0,
                            0,
                            bitmap_byte_offset.checked_add(end_row.div_ceil(8) as u64),
                        )
                    }
                    CudaCompoundFoldColumn::Text {
                        offsets_byte_offset,
                        bytes_byte_offset,
                        bytes_len,
                    } => {
                        if !offsets_byte_offset.is_multiple_of(8) {
                            return Err(CudaRuntimeProbeError::InvalidInputLength(
                                offsets_byte_offset as usize,
                            ));
                        }
                        let offsets_end =
                            offsets_byte_offset.checked_add((end_row as u64 + 1).saturating_mul(8));
                        let bytes_end = bytes_byte_offset.checked_add(bytes_len);
                        (
                            offsets_byte_offset,
                            0,
                            bytes_byte_offset,
                            bytes_len,
                            offsets_end.zip(bytes_end).map(|(a, b)| a.max(b)),
                        )
                    }
                };
                let extent = extent.ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                if extent > self.metadata().allocated_bytes {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(
                        usize::try_from(extent).unwrap_or(usize::MAX),
                    ));
                }
                column_descriptors.extend_from_slice(&[offset, width, blob_offset, blob_len]);
            }
        }
        let mut descriptors = index_descriptors;
        descriptors.extend_from_slice(&column_descriptors);
        let descriptor_bytes = std::mem::size_of_val(&*descriptors);
        let exact_preparation_bytes =
            resident_typed_indexes_insert_preparation_bytes(indexes.len(), column_count)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        // Refuse the complete two-lease geometry before taking either lease.  A budget failure is
        // therefore pre-WAL and leaves no partially prepared token or pooled scratch ownership.
        crate::CudaAllocationScope::ensure_available(exact_preparation_bytes)?;

        primary.set_current()?;
        let descriptor_guard = primary.lease_device_buffer_owned(descriptor_bytes)?;
        let decline_guard = primary.lease_device_buffer_owned(std::mem::size_of::<u32>())?;
        #[cfg(test)]
        if FAIL_PREPARED_MULTI_INDEX_AFTER_LEASES.with(|fail| fail.replace(false)) {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let cu_memset_d8 = unsafe {
            *primary
                .lib()
                .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
                .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_memcpy_htod = unsafe {
            *primary
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_memcpy_dtoh = unsafe {
            *primary
                .lib()
                .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_launch_kernel = unsafe {
            *primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        // Resolve every potentially failing driver/module dependency before queueing the first
        // default-stream operation. The token then owns queued descriptor setup plus both leases;
        // its later same-stream launch is ordered after this setup without an extra sync.
        let mut ptx = Vec::with_capacity(RESIDENT_MULTI_INDEX_INSERT_PTX.len() + 1);
        ptx.extend_from_slice(RESIDENT_MULTI_INDEX_INSERT_PTX);
        ptx.push(0);
        let function =
            primary.cached_function(c"gpu_db_resident_typed_multi_index_insert", &ptx)?;
        // Arm before the first null-stream HtoD/memset. Any setup error after this point drains
        // queued work before the owned descriptor/verdict leases return to the pool.
        let stream_drain = NullStreamDrain {
            primary: Arc::clone(&primary),
            armed: true,
        };
        check_cuda(unsafe {
            cu_memcpy_htod(
                descriptor_guard.ptr,
                descriptors.as_ptr().cast::<c_void>(),
                descriptor_bytes,
            )
        })?;
        check_cuda(unsafe { cu_memset_d8(decline_guard.ptr, 0, std::mem::size_of::<u32>()) })?;
        #[cfg(test)]
        PREPARED_MULTI_INDEX_PREPARES.with(|prepares| prepares.set(prepares.get() + 1));
        Ok(PreparedResidentTypedIndexesInsert {
            primary,
            _source_owner: self.allocation_arc(),
            _index_owners: indexes
                .iter()
                .map(|request| request.index.allocation_arc())
                .collect::<Box<[_]>>(),
            preparation_drain: Some(stream_drain),
            descriptor_guard,
            decline_guard,
            function,
            source_ptr: self.device_ptr(),
            index_count: index_count_u32,
            base_row: base_row_u32,
            row_count: row_count_u32,
            work_items,
            cu_memcpy_dtoh,
            cu_launch_kernel,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_resident_typed_index_range(
        &self,
        index: &CudaResidentDeviceMemory,
        table_mask: u32,
        hash_shift: u32,
        columns: &[CudaCompoundFoldColumn],
        base_row: usize,
        row_count: usize,
        deleted_by: Option<&CudaResidentDeviceMemory>,
        gc_boundary: u64,
        dup_tolerant: bool,
    ) -> Result<CudaResidentIndexStatus, CudaRuntimeProbeError> {
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

        let base_row_u32 = u32::try_from(base_row)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(base_row))?;
        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(row_count))?;
        let end_row = base_row
            .checked_add(row_count)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if end_row >= u32::MAX as usize {
            return Err(CudaRuntimeProbeError::InvalidInputLength(end_row));
        }
        if row_count == 0 || columns.is_empty() || self.context() != index.context() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(row_count));
        }
        validate_index_fold_columns(columns)?;
        let table_size = u64::from(table_mask) + 1;
        let expected_shift = 32_u32.checked_sub(table_size.trailing_zeros()).ok_or(
            CudaRuntimeProbeError::InvalidInputLength(table_size as usize),
        )?;
        let hash_bytes = resident_index_hash_bytes(table_mask)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let index_bytes = resident_index_allocated_bytes(table_mask, end_row as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if !table_size.is_power_of_two()
            || table_size > (1_u64 << 30)
            || hash_shift != expected_shift
            || index.metadata().allocated_bytes < index_bytes
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(index_bytes).unwrap_or(usize::MAX),
            ));
        }
        if let Some(deleted) = deleted_by {
            let needed = (end_row as u64)
                .checked_mul(std::mem::size_of::<u64>() as u64)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if deleted.context() != self.context() || deleted.metadata().allocated_bytes < needed {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    usize::try_from(needed).unwrap_or(usize::MAX),
                ));
            }
        }

        let mut offsets = Vec::with_capacity(columns.len());
        let mut widths = Vec::with_capacity(columns.len());
        let mut blob_offsets = Vec::with_capacity(columns.len());
        let mut blob_lens = Vec::with_capacity(columns.len());
        for column in columns {
            match *column {
                CudaCompoundFoldColumn::Fixed {
                    byte_offset,
                    width_words,
                } => {
                    let bytes = (end_row as u64)
                        .checked_mul(u64::from(width_words))
                        .and_then(|words| words.checked_mul(4))
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                    let end = byte_offset
                        .checked_add(bytes)
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                    if width_words == 0
                        || !byte_offset.is_multiple_of(4)
                        || end > self.metadata().allocated_bytes
                    {
                        return Err(CudaRuntimeProbeError::InvalidInputLength(
                            usize::try_from(end).unwrap_or(usize::MAX),
                        ));
                    }
                    offsets.push(byte_offset);
                    widths.push(width_words);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                CudaCompoundFoldColumn::Bool { bitmap_byte_offset } => {
                    let bytes = end_row.div_ceil(8) as u64;
                    let end = bitmap_byte_offset
                        .checked_add(bytes)
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                    if end > self.metadata().allocated_bytes {
                        return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
                    }
                    offsets.push(bitmap_byte_offset);
                    widths.push(u32::MAX);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                CudaCompoundFoldColumn::Validity { bitmap_byte_offset } => {
                    let bytes = end_row.div_ceil(8) as u64;
                    let end = bitmap_byte_offset
                        .checked_add(bytes)
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                    if !bitmap_byte_offset.is_multiple_of(4)
                        || end > self.metadata().allocated_bytes
                    {
                        return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
                    }
                    offsets.push(bitmap_byte_offset);
                    widths.push(COMPOUND_FOLD_VALIDITY_WIDTH);
                    blob_offsets.push(0);
                    blob_lens.push(0);
                }
                CudaCompoundFoldColumn::Text {
                    offsets_byte_offset,
                    bytes_byte_offset,
                    bytes_len,
                } => {
                    let offsets_end = offsets_byte_offset
                        .checked_add((end_row as u64 + 1).saturating_mul(8))
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                    let bytes_end = bytes_byte_offset
                        .checked_add(bytes_len)
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                    if !offsets_byte_offset.is_multiple_of(8)
                        || offsets_end > self.metadata().allocated_bytes
                        || bytes_end > self.metadata().allocated_bytes
                    {
                        return Err(CudaRuntimeProbeError::InvalidInputLength(
                            usize::try_from(offsets_end.max(bytes_end)).unwrap_or(usize::MAX),
                        ));
                    }
                    offsets.push(offsets_byte_offset);
                    widths.push(0);
                    blob_offsets.push(bytes_byte_offset);
                    blob_lens.push(bytes_len);
                }
            }
        }

        let primary = self.primary_arc();
        primary.set_current()?;
        let offsets_guard = primary.lease_device_buffer_owned(std::mem::size_of_val(&*offsets))?;
        let widths_guard = primary.lease_device_buffer_owned(std::mem::size_of_val(&*widths))?;
        let blob_offsets_guard =
            primary.lease_device_buffer_owned(std::mem::size_of_val(&*blob_offsets))?;
        let blob_lens_guard =
            primary.lease_device_buffer_owned(std::mem::size_of_val(&*blob_lens))?;
        let decline_guard = primary.lease_device_buffer_owned(std::mem::size_of::<u32>())?;
        let cu_memset_d8 = unsafe {
            *primary
                .lib()
                .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
                .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_memcpy_htod = unsafe {
            *primary
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_memcpy_dtoh = unsafe {
            *primary
                .lib()
                .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_launch_kernel = unsafe {
            *primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        for (guard, ptr, bytes) in [
            (
                &offsets_guard,
                offsets.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(&*offsets),
            ),
            (
                &widths_guard,
                widths.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(&*widths),
            ),
            (
                &blob_offsets_guard,
                blob_offsets.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(&*blob_offsets),
            ),
            (
                &blob_lens_guard,
                blob_lens.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(&*blob_lens),
            ),
        ] {
            check_cuda(unsafe { cu_memcpy_htod(guard.ptr, ptr, bytes) })?;
        }
        check_cuda(unsafe { cu_memset_d8(decline_guard.ptr, 0, std::mem::size_of::<u32>()) })?;

        let mut ptx = Vec::with_capacity(RESIDENT_INDEX_BUILD_PTX.len() + 1);
        ptx.extend_from_slice(RESIDENT_INDEX_BUILD_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_resident_typed_index_build", &ptx)?;
        let mut base_arg = self.device_ptr();
        let mut offsets_arg = offsets_guard.ptr;
        let mut widths_arg = widths_guard.ptr;
        let mut blob_offsets_arg = blob_offsets_guard.ptr;
        let mut blob_lens_arg = blob_lens_guard.ptr;
        let mut ncols_arg = columns.len() as u32;
        let mut base_row_arg = base_row_u32;
        let mut rows_arg = row_count_u32;
        let mut index_arg = index.device_ptr();
        let mut next_arg = index
            .device_ptr()
            .checked_add(hash_bytes)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let mut mask_arg = table_mask;
        let mut shift_arg = hash_shift;
        let mut deleted_arg = deleted_by.map_or(0, CudaResidentDeviceMemory::device_ptr);
        let mut boundary_arg = gc_boundary;
        let mut tolerant_arg = u32::from(dup_tolerant);
        let mut decline_arg = decline_guard.ptr;
        let mut args = [
            (&mut base_arg as *mut u64).cast::<c_void>(),
            (&mut offsets_arg as *mut u64).cast::<c_void>(),
            (&mut widths_arg as *mut u64).cast::<c_void>(),
            (&mut blob_offsets_arg as *mut u64).cast::<c_void>(),
            (&mut blob_lens_arg as *mut u64).cast::<c_void>(),
            (&mut ncols_arg as *mut u32).cast::<c_void>(),
            (&mut base_row_arg as *mut u32).cast::<c_void>(),
            (&mut rows_arg as *mut u32).cast::<c_void>(),
            (&mut index_arg as *mut u64).cast::<c_void>(),
            (&mut next_arg as *mut u64).cast::<c_void>(),
            (&mut mask_arg as *mut u32).cast::<c_void>(),
            (&mut shift_arg as *mut u32).cast::<c_void>(),
            (&mut deleted_arg as *mut u64).cast::<c_void>(),
            (&mut boundary_arg as *mut u64).cast::<c_void>(),
            (&mut tolerant_arg as *mut u32).cast::<c_void>(),
            (&mut decline_arg as *mut u64).cast::<c_void>(),
        ];
        let threads = 128_u32;
        check_cuda(unsafe {
            cu_launch_kernel(
                function,
                row_count_u32.div_ceil(threads),
                1,
                1,
                threads,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;
        // Blocking null-stream DtoH is the completion fence for the build and descriptor leases.
        let mut decline = 0_u32;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                (&mut decline as *mut u32).cast::<c_void>(),
                decline_guard.ptr,
                std::mem::size_of::<u32>(),
            )
        })?;
        Ok(CudaResidentIndexStatus::from_bits(decline))
    }
}
