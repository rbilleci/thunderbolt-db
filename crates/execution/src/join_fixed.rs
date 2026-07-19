//! Accumulated fixed, text, and composite device coordinate-join ownership.

use std::os::raw::c_void;

use super::{
    check_cuda, CudaJoinCoordinatesU32, CudaJoinPayloadKey, CudaMatchBitmapU32,
    CudaPredicateMaskI32, CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError,
};

impl CudaResidentDeviceMemory {
    /// Extend a device-resident coordinate relation by one fixed-width equi-join input. The first
    /// step passes `accumulated=None` and `left_row_count`; later steps pass the prior coordinates.
    /// Eligibility masks are evaluated by the kernel (typically visibility plus an INNER-pushable
    /// WHERE predicate). OUTER complements and NULL pads are also emitted on-device.
    #[allow(clippy::too_many_arguments)]
    pub fn join_fixed_payload_coordinates(
        &self,
        accumulated: Option<&CudaJoinCoordinatesU32>,
        left_row_count: u32,
        left_key_relations: &[u32],
        left_keys: &[CudaJoinPayloadKey<'_>],
        right_row_count: u32,
        right_keys: &[CudaJoinPayloadKey<'_>],
        left_eligibility: Option<&CudaPredicateMaskI32>,
        right_eligibility: Option<&CudaPredicateMaskI32>,
        outer_left: bool,
        outer_right: bool,
    ) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
        launch_cuda_join_fixed_payload_coordinates(
            self,
            accumulated,
            left_row_count,
            left_key_relations,
            left_keys,
            right_row_count,
            right_keys,
            left_eligibility,
            right_eligibility,
            outer_left,
            outer_right,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn join_fixed_payload_coordinate_matches(
        &self,
        accumulated: Option<&CudaJoinCoordinatesU32>,
        left_row_count: u32,
        left_key_relations: &[u32],
        left_keys: &[CudaJoinPayloadKey<'_>],
        right_row_count: u32,
        right_keys: &[CudaJoinPayloadKey<'_>],
        left_eligibility: Option<&CudaPredicateMaskI32>,
        right_eligibility: Option<&CudaPredicateMaskI32>,
        left_matches: &CudaMatchBitmapU32,
        right_matches: Option<&CudaMatchBitmapU32>,
    ) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
        launch_cuda_join_fixed_payload_coordinates(
            self,
            accumulated,
            left_row_count,
            left_key_relations,
            left_keys,
            right_row_count,
            right_keys,
            left_eligibility,
            right_eligibility,
            false,
            false,
            Some(left_matches),
            right_matches,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_join_fixed_payload_coordinates(
    ctx: &CudaResidentDeviceMemory,
    accumulated: Option<&CudaJoinCoordinatesU32>,
    left_row_count: u32,
    left_key_relations: &[u32],
    left_keys: &[CudaJoinPayloadKey<'_>],
    right_row_count: u32,
    right_keys: &[CudaJoinPayloadKey<'_>],
    left_eligibility: Option<&CudaPredicateMaskI32>,
    right_eligibility: Option<&CudaPredicateMaskI32>,
    outer_left: bool,
    outer_right: bool,
    persistent_left_marks: Option<&CudaMatchBitmapU32>,
    persistent_right_marks: Option<&CudaMatchBitmapU32>,
) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
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
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

// Match/count or match/emit one accumulated tuple x one new-relation row. Keys are read straight
// from their source payloads; `coords` carries the accumulated relation's absolute row coordinate.
.visible .entry gpu_db_join_fixed_match(
    .param .u64 coords, .param .u32 left_n, .param .u32 left_rels,
    .param .u32 has_coords, .param .u64 left_key_rels,
    .param .u64 lb0, .param .u64 lo0, .param .u64 lv0, .param .u32 lw0,
    .param .u64 rb0, .param .u64 ro0, .param .u64 rv0, .param .u32 rw0,
    .param .u64 lb1, .param .u64 lo1, .param .u64 lv1, .param .u32 lw1,
    .param .u64 rb1, .param .u64 ro1, .param .u64 rv1, .param .u32 rw1,
    .param .u32 right_n, .param .u64 left_mask, .param .u64 right_mask,
    .param .u64 left_marks, .param .u64 right_marks,
    .param .u64 out, .param .u64 cursor)
{
    .reg .pred %p<32>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<96>;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r1, [left_n];
    ld.param.u32 %r2, [left_rels];
    ld.param.u32 %r3, [has_coords];
    ld.param.u64 %rd64, [left_key_rels]; cvt.u32.u64 %r34, %rd64;
    shr.u64 %rd64, %rd64, 32; cvt.u32.u64 %r4, %rd64;
    ld.param.u32 %r5, [right_n];
    ld.param.u64 %rd2, [left_mask];
    ld.param.u64 %rd3, [right_mask];
    ld.param.u64 %rd4, [left_marks];
    ld.param.u64 %rd5, [right_marks];
    ld.param.u64 %rd6, [out];
    ld.param.u64 %rd7, [cursor];

    mov.u32 %r6, %tid.x;
    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mov.u32 %r9, %nctaid.x;
    mad.lo.u32 %r10, %r7, %r8, %r6;
    mul.lo.u32 %r11, %r9, %r8;
    cvt.u64.u32 %rd8, %r10;
    cvt.u64.u32 %rd9, %r11;
    cvt.u64.u32 %rd10, %r1;
    cvt.u64.u32 %rd11, %r5;
    mul.lo.u64 %rd12, %rd10, %rd11;
PAIR_LOOP:
    setp.ge.u64 %p1, %rd8, %rd12;
    @%p1 bra MATCH_DONE;
    div.u64 %rd13, %rd8, %rd11;
    rem.u64 %rd14, %rd8, %rd11;
    cvt.u32.u64 %r12, %rd13;
    cvt.u32.u64 %r13, %rd14;
    setp.eq.u32 %p2, %r3, 0;
    @%p2 mov.u32 %r14, %r12;
    @%p2 bra LEFT_ROW_READY;
    mul.lo.u32 %r15, %r12, %r2;
    add.u32 %r15, %r15, %r4;
    mul.wide.u32 %rd15, %r15, 4;
    add.u64 %rd16, %rd1, %rd15;
    ld.global.u32 %r14, [%rd16];
LEFT_ROW_READY:
    setp.eq.u32 %p3, %r14, 4294967295;
    @%p3 bra PAIR_NEXT;

    mov.u64 %rd17, 18446744073709551615;
    setp.eq.u64 %p4, %rd2, %rd17;
    @%p4 bra LEFT_MASK_OK;
    mul.wide.u32 %rd18, %r14, 4;
    add.u64 %rd19, %rd2, %rd18;
    ld.global.u32 %r16, [%rd19];
    setp.eq.u32 %p5, %r16, 0;
    @%p5 bra PAIR_NEXT;
LEFT_MASK_OK:
    setp.eq.u64 %p4, %rd3, %rd17;
    @%p4 bra RIGHT_MASK_OK;
    mul.wide.u32 %rd18, %r13, 4;
    add.u64 %rd19, %rd3, %rd18;
    ld.global.u32 %r16, [%rd19];
    setp.eq.u32 %p5, %r16, 0;
    @%p5 bra PAIR_NEXT;
RIGHT_MASK_OK:

    // Component 0 validity.
    ld.param.u64 %rd20, [lv0];
    setp.eq.u64 %p6, %rd20, %rd17;
    @%p6 bra L0_VALID;
    shr.u32 %r17, %r14, 5;
    mul.wide.u32 %rd21, %r17, 4;
    add.u64 %rd21, %rd20, %rd21;
    ld.global.u32 %r18, [%rd21];
    and.b32 %r17, %r14, 31;
    shr.u32 %r18, %r18, %r17;
    and.b32 %r18, %r18, 1;
    setp.eq.u32 %p7, %r18, 0;
    @%p7 bra PAIR_NEXT;
L0_VALID:
    ld.param.u64 %rd22, [rv0];
    setp.eq.u64 %p6, %rd22, %rd17;
    @%p6 bra R0_VALID;
    shr.u32 %r17, %r13, 5;
    mul.wide.u32 %rd21, %r17, 4;
    add.u64 %rd21, %rd22, %rd21;
    ld.global.u32 %r18, [%rd21];
    and.b32 %r17, %r13, 31;
    shr.u32 %r18, %r18, %r17;
    and.b32 %r18, %r18, 1;
    setp.eq.u32 %p7, %r18, 0;
    @%p7 bra PAIR_NEXT;
R0_VALID:
    ld.param.u64 %rd23, [lb0];
    ld.param.u64 %rd24, [lo0];
    ld.param.u64 %rd25, [rb0];
    ld.param.u64 %rd26, [ro0];
    ld.param.u32 %r19, [lw0];
    mul.wide.u32 %rd27, %r14, %r19;
    mul.wide.u32 %rd28, %r13, %r19;
    add.u64 %rd27, %rd27, %rd23;
    add.u64 %rd27, %rd27, %rd24;
    add.u64 %rd28, %rd28, %rd25;
    add.u64 %rd28, %rd28, %rd26;
    setp.eq.u32 %p20, %r19, 255;
    @%p20 bra CMP0_TEXT;
    setp.eq.u32 %p8, %r19, 4;
    @%p8 bra CMP0_4;
    ld.global.u32 %r35,[%rd27]; ld.global.u32 %r36,[%rd27+4];
    cvt.u64.u32 %rd29,%r36; shl.b64 %rd29,%rd29,32; cvt.u64.u32 %rd72,%r35; or.b64 %rd29,%rd29,%rd72;
    ld.global.u32 %r37,[%rd28]; ld.global.u32 %r38,[%rd28+4];
    cvt.u64.u32 %rd30,%r38; shl.b64 %rd30,%rd30,32; cvt.u64.u32 %rd72,%r37; or.b64 %rd30,%rd30,%rd72;
    setp.ne.u64 %p9, %rd29, %rd30;
    @%p9 bra PAIR_NEXT;
    setp.eq.u32 %p10, %r19, 16;
    @!%p10 bra CMP0_DONE;
    ld.global.u32 %r35,[%rd27+8]; ld.global.u32 %r36,[%rd27+12];
    cvt.u64.u32 %rd29,%r36; shl.b64 %rd29,%rd29,32; cvt.u64.u32 %rd72,%r35; or.b64 %rd29,%rd29,%rd72;
    ld.global.u32 %r37,[%rd28+8]; ld.global.u32 %r38,[%rd28+12];
    cvt.u64.u32 %rd30,%r38; shl.b64 %rd30,%rd30,32; cvt.u64.u32 %rd72,%r37; or.b64 %rd30,%rd30,%rd72;
    setp.ne.u64 %p9, %rd29, %rd30;
    @%p9 bra PAIR_NEXT;
    bra CMP0_DONE;
CMP0_4:
    ld.global.u32 %r20, [%rd27];
    ld.global.u32 %r21, [%rd28];
    setp.ne.u32 %p9, %r20, %r21;
    @%p9 bra PAIR_NEXT;
    bra CMP0_DONE;
CMP0_TEXT:
    // lo0/ro0 point at u64 row offsets; lo1/ro1 carry the corresponding bytes-section offsets.
    mul.wide.u32 %rd51, %r14, 8;
    mul.wide.u32 %rd52, %r13, 8;
    add.u64 %rd53, %rd23, %rd24;
    add.u64 %rd54, %rd25, %rd26;
    add.u64 %rd55, %rd53, %rd51;
    add.u64 %rd56, %rd54, %rd52;
    ld.global.u64 %rd57, [%rd55];
    ld.global.u64 %rd58, [%rd55+8];
    ld.global.u64 %rd59, [%rd56];
    ld.global.u64 %rd60, [%rd56+8];
    sub.u64 %rd61, %rd58, %rd57;
    sub.u64 %rd62, %rd60, %rd59;
    setp.ne.u64 %p21, %rd61, %rd62;
    @%p21 bra PAIR_NEXT;
    ld.param.u64 %rd63, [lb1];
    ld.param.u64 %rd64, [lo1];
    ld.param.u64 %rd65, [rb1];
    ld.param.u64 %rd66, [ro1];
    add.u64 %rd67, %rd63, %rd64;
    add.u64 %rd67, %rd67, %rd57;
    add.u64 %rd68, %rd65, %rd66;
    add.u64 %rd68, %rd68, %rd59;
    mov.u64 %rd69, 0;
TEXT_LOOP:
    setp.ge.u64 %p22, %rd69, %rd61;
    @%p22 bra CMP0_DONE;
    add.u64 %rd70, %rd67, %rd69;
    add.u64 %rd71, %rd68, %rd69;
    ld.global.u8 %r33, [%rd70];
    ld.global.u8 %r34, [%rd71];
    setp.ne.u32 %p23, %r33, %r34;
    @%p23 bra PAIR_NEXT;
    add.u64 %rd69, %rd69, 1;
    bra TEXT_LOOP;
CMP0_DONE:

    // Optional component 1 (a two-column fixed-width composite key).
    ld.param.u32 %r22, [lw1];
    setp.eq.u32 %p11, %r22, 0;
    @%p11 bra PAIR_MATCH;
    ld.param.u64 %rd31, [lv1];
    setp.eq.u64 %p12, %rd31, %rd17;
    @%p12 bra L1_VALID;
    shr.u32 %r23, %r14, 5;
    mul.wide.u32 %rd32, %r23, 4;
    add.u64 %rd32, %rd31, %rd32;
    ld.global.u32 %r24, [%rd32];
    and.b32 %r23, %r14, 31;
    shr.u32 %r24, %r24, %r23;
    and.b32 %r24, %r24, 1;
    setp.eq.u32 %p13, %r24, 0;
    @%p13 bra PAIR_NEXT;
L1_VALID:
    ld.param.u64 %rd33, [rv1];
    setp.eq.u64 %p12, %rd33, %rd17;
    @%p12 bra R1_VALID;
    shr.u32 %r23, %r13, 5;
    mul.wide.u32 %rd32, %r23, 4;
    add.u64 %rd32, %rd33, %rd32;
    ld.global.u32 %r24, [%rd32];
    and.b32 %r23, %r13, 31;
    shr.u32 %r24, %r24, %r23;
    and.b32 %r24, %r24, 1;
    setp.eq.u32 %p13, %r24, 0;
    @%p13 bra PAIR_NEXT;
R1_VALID:
    ld.param.u64 %rd34, [lb1];
    ld.param.u64 %rd35, [lo1];
    ld.param.u64 %rd36, [rb1];
    ld.param.u64 %rd37, [ro1];
    mul.wide.u32 %rd38, %r14, %r22;
    mul.wide.u32 %rd39, %r13, %r22;
    add.u64 %rd38, %rd38, %rd34;
    add.u64 %rd38, %rd38, %rd35;
    add.u64 %rd39, %rd39, %rd36;
    add.u64 %rd39, %rd39, %rd37;
    setp.eq.u32 %p14, %r22, 4;
    @%p14 bra CMP1_4;
    ld.global.u32 %r35,[%rd38]; ld.global.u32 %r36,[%rd38+4];
    cvt.u64.u32 %rd40,%r36; shl.b64 %rd40,%rd40,32; cvt.u64.u32 %rd72,%r35; or.b64 %rd40,%rd40,%rd72;
    ld.global.u32 %r37,[%rd39]; ld.global.u32 %r38,[%rd39+4];
    cvt.u64.u32 %rd41,%r38; shl.b64 %rd41,%rd41,32; cvt.u64.u32 %rd72,%r37; or.b64 %rd41,%rd41,%rd72;
    setp.ne.u64 %p15, %rd40, %rd41;
    @%p15 bra PAIR_NEXT;
    setp.eq.u32 %p16, %r22, 16;
    @!%p16 bra PAIR_MATCH;
    ld.global.u32 %r35,[%rd38+8]; ld.global.u32 %r36,[%rd38+12];
    cvt.u64.u32 %rd40,%r36; shl.b64 %rd40,%rd40,32; cvt.u64.u32 %rd72,%r35; or.b64 %rd40,%rd40,%rd72;
    ld.global.u32 %r37,[%rd39+8]; ld.global.u32 %r38,[%rd39+12];
    cvt.u64.u32 %rd41,%r38; shl.b64 %rd41,%rd41,32; cvt.u64.u32 %rd72,%r37; or.b64 %rd41,%rd41,%rd72;
    setp.ne.u64 %p15, %rd40, %rd41;
    @%p15 bra PAIR_NEXT;
    bra PAIR_MATCH;
CMP1_4:
    ld.global.u32 %r25, [%rd38];
    ld.global.u32 %r26, [%rd39];
    setp.ne.u32 %p15, %r25, %r26;
    @%p15 bra PAIR_NEXT;

PAIR_MATCH:
    mul.wide.u32 %rd42, %r12, 4;
    add.u64 %rd43, %rd4, %rd42;
    mov.u32 %r27, 1;
    atom.global.exch.b32 %r28, [%rd43], %r27;
    mul.wide.u32 %rd42, %r13, 4;
    add.u64 %rd43, %rd5, %rd42;
    atom.global.exch.b32 %r28, [%rd43], %r27;
    atom.global.add.u64 %rd44, [%rd7], 1;
    setp.eq.u64 %p17, %rd6, 0;
    @%p17 bra PAIR_NEXT;
    add.u32 %r29, %r2, 1;
    cvt.u64.u32 %rd45, %r29;
    mul.lo.u64 %rd46, %rd44, %rd45;
    mov.u32 %r30, 0;
COPY_LEFT:
    setp.ge.u32 %p18, %r30, %r2;
    @%p18 bra WRITE_RIGHT;
    setp.eq.u32 %p19, %r3, 0;
    @%p19 mov.u32 %r31, %r12;
    @%p19 bra COPY_STORE;
    mul.lo.u32 %r32, %r12, %r2;
    add.u32 %r32, %r32, %r30;
    mul.wide.u32 %rd47, %r32, 4;
    add.u64 %rd48, %rd1, %rd47;
    ld.global.u32 %r31, [%rd48];
COPY_STORE:
    cvt.u64.u32 %rd49, %r30;
    add.u64 %rd50, %rd46, %rd49;
    mul.lo.u64 %rd50, %rd50, 4;
    add.u64 %rd50, %rd6, %rd50;
    st.global.u32 [%rd50], %r31;
    add.u32 %r30, %r30, 1;
    bra COPY_LEFT;
WRITE_RIGHT:
    cvt.u64.u32 %rd49, %r2;
    add.u64 %rd50, %rd46, %rd49;
    mul.lo.u64 %rd50, %rd50, 4;
    add.u64 %rd50, %rd6, %rd50;
    st.global.u32 [%rd50], %r13;
PAIR_NEXT:
    add.u64 %rd8, %rd8, %rd9;
    bra PAIR_LOOP;
MATCH_DONE:
    ret;
}

// Build a chained hash table directly from fixed-width right-payload keys. Excluded/NULL rows are
// not chained. Buckets contain row indices, so N:N duplicates remain distinct.
.visible .entry gpu_db_join_fixed_hash_build(
    .param .u64 rb0, .param .u64 ro0, .param .u64 rv0, .param .u32 rw0,
    .param .u64 rb1, .param .u64 ro1, .param .u64 rv1, .param .u32 rw1,
    .param .u32 right_n, .param .u64 right_mask,
    .param .u64 heads, .param .u64 next, .param .u32 bucket_mask)
{
    .reg .pred %p<16>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<48>;
    ld.param.u64 %rd1, [rb0]; ld.param.u64 %rd2, [ro0];
    ld.param.u64 %rd3, [rv0]; ld.param.u32 %r1, [rw0];
    ld.param.u64 %rd4, [rb1]; ld.param.u64 %rd5, [ro1];
    ld.param.u64 %rd6, [rv1]; ld.param.u32 %r2, [rw1];
    ld.param.u32 %r3, [right_n]; ld.param.u64 %rd7, [right_mask];
    ld.param.u64 %rd8, [heads]; ld.param.u64 %rd9, [next];
    ld.param.u32 %r4, [bucket_mask];
    mov.u32 %r5, %tid.x; mov.u32 %r6, %ctaid.x; mov.u32 %r7, %ntid.x;
    mov.u32 %r8, %nctaid.x; mad.lo.u32 %r9, %r6, %r7, %r5;
    mul.lo.u32 %r10, %r8, %r7; mov.u64 %rd10, 18446744073709551615;
BUILD_ROW:
    setp.ge.u32 %p1, %r9, %r3; @%p1 bra BUILD_DONE;
    setp.eq.u64 %p2, %rd7, %rd10; @%p2 bra BUILD_MASK_OK;
    mul.wide.u32 %rd11, %r9, 4; add.u64 %rd12, %rd7, %rd11;
    ld.global.u32 %r11, [%rd12]; setp.eq.u32 %p3, %r11, 0; @%p3 bra BUILD_NEXT;
BUILD_MASK_OK:
    setp.eq.u64 %p4, %rd3, %rd10; @%p4 bra BUILD_VALID0;
    shr.u32 %r12, %r9, 5; mul.wide.u32 %rd13, %r12, 4; add.u64 %rd13, %rd3, %rd13;
    ld.global.u32 %r13, [%rd13]; and.b32 %r12, %r9, 31; shr.u32 %r13, %r13, %r12;
    and.b32 %r13, %r13, 1; setp.eq.u32 %p5, %r13, 0; @%p5 bra BUILD_NEXT;
BUILD_VALID0:
    setp.eq.u32 %p6, %r2, 0; @%p6 bra BUILD_HASH;
    setp.eq.u64 %p4, %rd6, %rd10; @%p4 bra BUILD_HASH;
    shr.u32 %r12, %r9, 5; mul.wide.u32 %rd13, %r12, 4; add.u64 %rd13, %rd6, %rd13;
    ld.global.u32 %r13, [%rd13]; and.b32 %r12, %r9, 31; shr.u32 %r13, %r13, %r12;
    and.b32 %r13, %r13, 1; setp.eq.u32 %p5, %r13, 0; @%p5 bra BUILD_NEXT;
BUILD_HASH:
    mov.u64 %rd14, 14695981039346656037;
    setp.eq.u32 %p9, %r1, 255; @%p9 bra BUILD_HASH0_TEXT;
    mul.wide.u32 %rd15, %r9, %r1; add.u64 %rd15, %rd15, %rd1; add.u64 %rd15, %rd15, %rd2;
    setp.eq.u32 %p9, %r1, 4; @%p9 bra BUILD_HASH0_I4;
    setp.eq.u32 %p9, %r1, 8; @%p9 bra BUILD_HASH0_I8;
    mov.u32 %r14, 0;
BUILD_HASH0_LOOP:
    setp.ge.u32 %p7, %r14, %r1; @%p7 bra BUILD_HASH1_START;
    cvt.u64.u32 %rd16, %r14; add.u64 %rd17, %rd15, %rd16; ld.global.u8 %r15, [%rd17];
    cvt.u64.u32 %rd18, %r15; xor.b64 %rd14, %rd14, %rd18;
    mul.lo.u64 %rd14, %rd14, 1099511628211; add.u32 %r14, %r14, 1; bra BUILD_HASH0_LOOP;
BUILD_HASH0_I4:
    ld.global.s32 %r33,[%rd15]; cvt.s64.s32 %rd32,%r33; bra BUILD_HASH0_INT_START;
BUILD_HASH0_I8:
    ld.global.u32 %r33,[%rd15]; ld.global.u32 %r34,[%rd15+4];
    cvt.u64.u32 %rd32,%r34; shl.b64 %rd32,%rd32,32; cvt.u64.u32 %rd33,%r33;
    or.b64 %rd32,%rd32,%rd33;
BUILD_HASH0_INT_START:
    mov.u32 %r14,0;
BUILD_HASH0_INT_LOOP:
    setp.ge.u32 %p7,%r14,8; @%p7 bra BUILD_HASH1_START;
    and.b64 %rd18,%rd32,255; xor.b64 %rd14,%rd14,%rd18;
    mul.lo.u64 %rd14,%rd14,1099511628211; shr.u64 %rd32,%rd32,8;
    add.u32 %r14,%r14,1; bra BUILD_HASH0_INT_LOOP;
BUILD_HASH0_TEXT:
    mul.wide.u32 %rd23, %r9, 8; add.u64 %rd24, %rd1, %rd2; add.u64 %rd24, %rd24, %rd23;
    ld.global.u64 %rd25, [%rd24]; ld.global.u64 %rd26, [%rd24+8];
    add.u64 %rd27, %rd4, %rd5; add.u64 %rd27, %rd27, %rd25;
    sub.u64 %rd28, %rd26, %rd25; mov.u64 %rd29, 0;
BUILD_HASH0_TEXT_LOOP:
    setp.ge.u64 %p10, %rd29, %rd28; @%p10 bra BUILD_HASH1_START;
    add.u64 %rd30, %rd27, %rd29; ld.global.u8 %r15, [%rd30]; cvt.u64.u32 %rd18, %r15;
    xor.b64 %rd14, %rd14, %rd18; mul.lo.u64 %rd14, %rd14, 1099511628211;
    add.u64 %rd29, %rd29, 1; bra BUILD_HASH0_TEXT_LOOP;
BUILD_HASH1_START:
    setp.eq.u32 %p8, %r2, 0; @%p8 bra BUILD_INSERT;
    mul.wide.u32 %rd15, %r9, %r2; add.u64 %rd15, %rd15, %rd4; add.u64 %rd15, %rd15, %rd5;
    setp.eq.u32 %p9,%r2,4; @%p9 bra BUILD_HASH1_I4;
    setp.eq.u32 %p9,%r2,8; @%p9 bra BUILD_HASH1_I8;
    mov.u32 %r14, 0;
BUILD_HASH1_LOOP:
    setp.ge.u32 %p7, %r14, %r2; @%p7 bra BUILD_INSERT;
    cvt.u64.u32 %rd16, %r14; add.u64 %rd17, %rd15, %rd16; ld.global.u8 %r15, [%rd17];
    cvt.u64.u32 %rd18, %r15; xor.b64 %rd14, %rd14, %rd18;
    mul.lo.u64 %rd14, %rd14, 1099511628211; add.u32 %r14, %r14, 1; bra BUILD_HASH1_LOOP;
BUILD_HASH1_I4:
    ld.global.s32 %r33,[%rd15]; cvt.s64.s32 %rd32,%r33; bra BUILD_HASH1_INT_START;
BUILD_HASH1_I8:
    ld.global.u32 %r33,[%rd15]; ld.global.u32 %r34,[%rd15+4];
    cvt.u64.u32 %rd32,%r34; shl.b64 %rd32,%rd32,32; cvt.u64.u32 %rd33,%r33;
    or.b64 %rd32,%rd32,%rd33;
BUILD_HASH1_INT_START:
    mov.u32 %r14,0;
BUILD_HASH1_INT_LOOP:
    setp.ge.u32 %p7,%r14,8; @%p7 bra BUILD_INSERT;
    and.b64 %rd18,%rd32,255; xor.b64 %rd14,%rd14,%rd18;
    mul.lo.u64 %rd14,%rd14,1099511628211; shr.u64 %rd32,%rd32,8;
    add.u32 %r14,%r14,1; bra BUILD_HASH1_INT_LOOP;
BUILD_INSERT:
    cvt.u32.u64 %r16, %rd14; and.b32 %r16, %r16, %r4;
    mul.wide.u32 %rd19, %r16, 4; add.u64 %rd20, %rd8, %rd19;
    atom.global.exch.b32 %r17, [%rd20], %r9;
    mul.wide.u32 %rd21, %r9, 4; add.u64 %rd22, %rd9, %rd21; st.global.u32 [%rd22], %r17;
BUILD_NEXT:
    add.u32 %r9, %r9, %r10; bra BUILD_ROW;
BUILD_DONE:
    ret;
}

// Probe one accumulated tuple per thread. Hash collisions are verified byte-for-byte before the
// coordinate is emitted, and chained right rows preserve the full N:N cross product.
.visible .entry gpu_db_join_fixed_hash_probe_v2(
    .param .u64 coords, .param .u32 left_n, .param .u32 left_rels,
    .param .u32 has_coords, .param .u64 left_key_rels,
    .param .u64 lb0, .param .u64 lo0, .param .u64 lv0, .param .u32 lw0,
    .param .u64 rb0, .param .u64 ro0, .param .u32 rw0,
    .param .u64 lb1, .param .u64 lo1, .param .u64 lv1, .param .u32 lw1,
    .param .u64 rb1, .param .u64 ro1, .param .u32 rw1,
    .param .u64 left_mask, .param .u64 heads, .param .u64 next,
    .param .u32 bucket_mask, .param .u64 left_marks, .param .u64 right_marks,
    .param .u64 out, .param .u64 cursor)
{
    .reg .pred %p<24>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<80>;
    ld.param.u64 %rd1, [coords]; ld.param.u32 %r1, [left_n];
    ld.param.u32 %r2, [left_rels]; ld.param.u32 %r3, [has_coords];
    // Packed as relation 0 in the high word and relation 1 in the low word. Keeping the
    // component coordinates in one launch parameter avoids another ABI slot while still letting
    // each composite-key component address its own accumulated relation.
    ld.param.u64 %rd64, [left_key_rels]; cvt.u32.u64 %r34, %rd64;
    shr.u64 %rd64, %rd64, 32; cvt.u32.u64 %r4, %rd64;
    ld.param.u64 %rd2, [lb0]; ld.param.u64 %rd3, [lo0]; ld.param.u64 %rd4, [lv0];
    ld.param.u32 %r5, [lw0]; ld.param.u64 %rd5, [rb0]; ld.param.u64 %rd6, [ro0];
    ld.param.u32 %r6, [rw0];
    ld.param.u64 %rd7, [lb1]; ld.param.u64 %rd8, [lo1]; ld.param.u64 %rd9, [lv1];
    ld.param.u32 %r7, [lw1]; ld.param.u64 %rd10, [rb1]; ld.param.u64 %rd11, [ro1];
    ld.param.u32 %r8, [rw1]; ld.param.u64 %rd12, [left_mask];
    ld.param.u64 %rd13, [heads]; ld.param.u64 %rd14, [next];
    ld.param.u32 %r9, [bucket_mask]; ld.param.u64 %rd15, [left_marks];
    ld.param.u64 %rd16, [right_marks]; ld.param.u64 %rd17, [out];
    ld.param.u64 %rd18, [cursor];
    mov.u32 %r10, %tid.x; mov.u32 %r11, %ctaid.x; mov.u32 %r12, %ntid.x;
    mov.u32 %r13, %nctaid.x; mad.lo.u32 %r14, %r11, %r12, %r10;
    mul.lo.u32 %r15, %r13, %r12; mov.u64 %rd19, 18446744073709551615;
PROBE_ROW:
    setp.ge.u32 %p1, %r14, %r1; @%p1 bra PROBE_DONE;
    setp.eq.u32 %p2, %r3, 0; @%p2 mov.u32 %r16, %r14; @%p2 bra PROBE_LEFT_READY;
    mul.lo.u32 %r17, %r14, %r2; add.u32 %r17, %r17, %r4;
    mul.wide.u32 %rd20, %r17, 4; add.u64 %rd21, %rd1, %rd20; ld.global.u32 %r16, [%rd21];
PROBE_LEFT_READY:
    setp.eq.u32 %p3, %r16, 4294967295; @%p3 bra PROBE_NEXT;
    mov.u32 %r35, %r16;
    setp.eq.u32 %p23, %r7, 0; @%p23 bra PROBE_LEFT1_READY;
    setp.eq.u32 %p23, %r3, 0; @%p23 bra PROBE_LEFT1_READY;
    mul.lo.u32 %r36, %r14, %r2; add.u32 %r36, %r36, %r34;
    mul.wide.u32 %rd62, %r36, 4; add.u64 %rd63, %rd1, %rd62; ld.global.u32 %r35, [%rd63];
    setp.eq.u32 %p23, %r35, 4294967295; @%p23 bra PROBE_NEXT;
PROBE_LEFT1_READY:
    setp.eq.u64 %p4, %rd12, %rd19; @%p4 bra PROBE_MASK_OK;
    mul.wide.u32 %rd22, %r16, 4; add.u64 %rd23, %rd12, %rd22; ld.global.u32 %r18, [%rd23];
    setp.eq.u32 %p5, %r18, 0; @%p5 bra PROBE_NEXT;
PROBE_MASK_OK:
    setp.eq.u64 %p6, %rd4, %rd19; @%p6 bra PROBE_VALID0;
    shr.u32 %r19, %r16, 5; mul.wide.u32 %rd24, %r19, 4; add.u64 %rd24, %rd4, %rd24;
    ld.global.u32 %r20, [%rd24]; and.b32 %r19, %r16, 31; shr.u32 %r20, %r20, %r19;
    and.b32 %r20, %r20, 1; setp.eq.u32 %p7, %r20, 0; @%p7 bra PROBE_NEXT;
PROBE_VALID0:
    setp.eq.u32 %p8, %r7, 0; @%p8 bra PROBE_HASH;
    setp.eq.u64 %p6, %rd9, %rd19; @%p6 bra PROBE_HASH;
    shr.u32 %r19, %r35, 5; mul.wide.u32 %rd24, %r19, 4; add.u64 %rd24, %rd9, %rd24;
    ld.global.u32 %r20, [%rd24]; and.b32 %r19, %r35, 31; shr.u32 %r20, %r20, %r19;
    and.b32 %r20, %r20, 1; setp.eq.u32 %p7, %r20, 0; @%p7 bra PROBE_NEXT;
PROBE_HASH:
    mov.u64 %rd25, 14695981039346656037;
    setp.eq.u32 %p18, %r5, 255; @%p18 bra PROBE_HASH0_TEXT;
    mul.wide.u32 %rd26, %r16, %r5; add.u64 %rd26, %rd26, %rd2; add.u64 %rd26, %rd26, %rd3;
    setp.eq.u32 %p18,%r5,4; @%p18 bra PROBE_HASH0_I4;
    setp.eq.u32 %p18,%r5,8; @%p18 bra PROBE_HASH0_I8;
    mov.u32 %r21, 0;
PROBE_HASH0_LOOP:
    setp.ge.u32 %p9, %r21, %r5; @%p9 bra PROBE_HASH1_START;
    cvt.u64.u32 %rd27, %r21; add.u64 %rd28, %rd26, %rd27; ld.global.u8 %r22, [%rd28];
    cvt.u64.u32 %rd29, %r22; xor.b64 %rd25, %rd25, %rd29;
    mul.lo.u64 %rd25, %rd25, 1099511628211; add.u32 %r21, %r21, 1; bra PROBE_HASH0_LOOP;
PROBE_HASH0_I4:
    ld.global.s32 %r37,[%rd26]; cvt.s64.s32 %rd64,%r37; bra PROBE_HASH0_INT_START;
PROBE_HASH0_I8:
    ld.global.u32 %r37,[%rd26]; ld.global.u32 %r38,[%rd26+4];
    cvt.u64.u32 %rd64,%r38; shl.b64 %rd64,%rd64,32; cvt.u64.u32 %rd66,%r37;
    or.b64 %rd64,%rd64,%rd66;
PROBE_HASH0_INT_START:
    mov.u32 %r21,0;
PROBE_HASH0_INT_LOOP:
    setp.ge.u32 %p9,%r21,8; @%p9 bra PROBE_HASH1_START;
    and.b64 %rd29,%rd64,255; xor.b64 %rd25,%rd25,%rd29;
    mul.lo.u64 %rd25,%rd25,1099511628211; shr.u64 %rd64,%rd64,8;
    add.u32 %r21,%r21,1; bra PROBE_HASH0_INT_LOOP;
PROBE_HASH0_TEXT:
    mul.wide.u32 %rd48, %r16, 8; add.u64 %rd49, %rd2, %rd3; add.u64 %rd49, %rd49, %rd48;
    ld.global.u64 %rd50, [%rd49]; ld.global.u64 %rd51, [%rd49+8];
    add.u64 %rd52, %rd7, %rd8; add.u64 %rd52, %rd52, %rd50;
    sub.u64 %rd53, %rd51, %rd50; mov.u64 %rd54, 0;
PROBE_HASH0_TEXT_LOOP:
    setp.ge.u64 %p19, %rd54, %rd53; @%p19 bra PROBE_HASH1_START;
    add.u64 %rd55, %rd52, %rd54; ld.global.u8 %r22, [%rd55]; cvt.u64.u32 %rd29, %r22;
    xor.b64 %rd25, %rd25, %rd29; mul.lo.u64 %rd25, %rd25, 1099511628211;
    add.u64 %rd54, %rd54, 1; bra PROBE_HASH0_TEXT_LOOP;
PROBE_HASH1_START:
    setp.eq.u32 %p10, %r7, 0; @%p10 bra PROBE_BUCKET;
    mul.wide.u32 %rd26, %r35, %r7; add.u64 %rd26, %rd26, %rd7; add.u64 %rd26, %rd26, %rd8;
    setp.eq.u32 %p18,%r7,4; @%p18 bra PROBE_HASH1_I4;
    setp.eq.u32 %p18,%r7,8; @%p18 bra PROBE_HASH1_I8;
    mov.u32 %r21, 0;
PROBE_HASH1_LOOP:
    setp.ge.u32 %p9, %r21, %r7; @%p9 bra PROBE_BUCKET;
    cvt.u64.u32 %rd27, %r21; add.u64 %rd28, %rd26, %rd27; ld.global.u8 %r22, [%rd28];
    cvt.u64.u32 %rd29, %r22; xor.b64 %rd25, %rd25, %rd29;
    mul.lo.u64 %rd25, %rd25, 1099511628211; add.u32 %r21, %r21, 1; bra PROBE_HASH1_LOOP;
PROBE_HASH1_I4:
    ld.global.s32 %r37,[%rd26]; cvt.s64.s32 %rd64,%r37; bra PROBE_HASH1_INT_START;
PROBE_HASH1_I8:
    ld.global.u32 %r37,[%rd26]; ld.global.u32 %r38,[%rd26+4];
    cvt.u64.u32 %rd64,%r38; shl.b64 %rd64,%rd64,32; cvt.u64.u32 %rd66,%r37;
    or.b64 %rd64,%rd64,%rd66;
PROBE_HASH1_INT_START:
    mov.u32 %r21,0;
PROBE_HASH1_INT_LOOP:
    setp.ge.u32 %p9,%r21,8; @%p9 bra PROBE_BUCKET;
    and.b64 %rd29,%rd64,255; xor.b64 %rd25,%rd25,%rd29;
    mul.lo.u64 %rd25,%rd25,1099511628211; shr.u64 %rd64,%rd64,8;
    add.u32 %r21,%r21,1; bra PROBE_HASH1_INT_LOOP;
PROBE_BUCKET:
    cvt.u32.u64 %r23, %rd25; and.b32 %r23, %r23, %r9;
    mul.wide.u32 %rd30, %r23, 4; add.u64 %rd31, %rd13, %rd30; ld.global.u32 %r24, [%rd31];
CHAIN_LOOP:
    setp.eq.u32 %p11, %r24, 4294967295; @%p11 bra PROBE_NEXT;
    setp.eq.u32 %p20, %r5, 255; @%p20 bra CMP_HASH0_TEXT;
    mul.wide.u32 %rd32, %r16, %r5; add.u64 %rd32, %rd32, %rd2; add.u64 %rd32, %rd32, %rd3;
    mul.wide.u32 %rd33, %r24, %r6; add.u64 %rd33, %rd33, %rd5; add.u64 %rd33, %rd33, %rd6;
    setp.eq.u32 %p23,%r5,4; @%p23 bra CMP_HASH0_LEFT_I4;
    setp.eq.u32 %p23,%r5,8; @%p23 bra CMP_HASH0_LEFT_I8;
    mov.u32 %r25, 0;
CMP_HASH0_LOOP:
    setp.ge.u32 %p12, %r25, %r5; @%p12 bra CMP_HASH1_START;
    cvt.u64.u32 %rd34, %r25; add.u64 %rd35, %rd32, %rd34; add.u64 %rd36, %rd33, %rd34;
    ld.global.u8 %r26, [%rd35]; ld.global.u8 %r27, [%rd36];
    setp.ne.u32 %p13, %r26, %r27; @%p13 bra CHAIN_NEXT;
    add.u32 %r25, %r25, 1; bra CMP_HASH0_LOOP;
CMP_HASH0_LEFT_I4:
    ld.global.s32 %r37,[%rd32]; cvt.s64.s32 %rd64,%r37; bra CMP_HASH0_RIGHT;
CMP_HASH0_LEFT_I8:
    ld.global.u32 %r37,[%rd32]; ld.global.u32 %r38,[%rd32+4];
    cvt.u64.u32 %rd64,%r38; shl.b64 %rd64,%rd64,32; cvt.u64.u32 %rd66,%r37;
    or.b64 %rd64,%rd64,%rd66;
CMP_HASH0_RIGHT:
    setp.eq.u32 %p23,%r6,4; @%p23 bra CMP_HASH0_RIGHT_I4;
    ld.global.u32 %r37,[%rd33]; ld.global.u32 %r38,[%rd33+4];
    cvt.u64.u32 %rd65,%r38; shl.b64 %rd65,%rd65,32; cvt.u64.u32 %rd66,%r37;
    or.b64 %rd65,%rd65,%rd66; bra CMP_HASH0_INT_COMPARE;
CMP_HASH0_RIGHT_I4:
    ld.global.s32 %r37,[%rd33]; cvt.s64.s32 %rd65,%r37;
CMP_HASH0_INT_COMPARE:
    setp.ne.u64 %p13,%rd64,%rd65; @%p13 bra CHAIN_NEXT; bra CMP_HASH1_START;
CMP_HASH0_TEXT:
    mul.wide.u32 %rd48, %r16, 8; add.u64 %rd49, %rd2, %rd3; add.u64 %rd49, %rd49, %rd48;
    ld.global.u64 %rd50, [%rd49]; ld.global.u64 %rd51, [%rd49+8];
    mul.wide.u32 %rd48, %r24, 8; add.u64 %rd56, %rd5, %rd6; add.u64 %rd56, %rd56, %rd48;
    ld.global.u64 %rd57, [%rd56]; ld.global.u64 %rd58, [%rd56+8];
    sub.u64 %rd53, %rd51, %rd50; sub.u64 %rd59, %rd58, %rd57;
    setp.ne.u64 %p21, %rd53, %rd59; @%p21 bra CHAIN_NEXT;
    add.u64 %rd52, %rd7, %rd8; add.u64 %rd52, %rd52, %rd50;
    add.u64 %rd60, %rd10, %rd11; add.u64 %rd60, %rd60, %rd57; mov.u64 %rd54, 0;
CMP_HASH0_TEXT_LOOP:
    setp.ge.u64 %p22, %rd54, %rd53; @%p22 bra HASH_MATCH;
    add.u64 %rd55, %rd52, %rd54; add.u64 %rd61, %rd60, %rd54;
    ld.global.u8 %r26, [%rd55]; ld.global.u8 %r27, [%rd61];
    setp.ne.u32 %p13, %r26, %r27; @%p13 bra CHAIN_NEXT;
    add.u64 %rd54, %rd54, 1; bra CMP_HASH0_TEXT_LOOP;
CMP_HASH1_START:
    setp.eq.u32 %p14, %r7, 0; @%p14 bra HASH_MATCH;
    mul.wide.u32 %rd32, %r35, %r7; add.u64 %rd32, %rd32, %rd7; add.u64 %rd32, %rd32, %rd8;
    mul.wide.u32 %rd33, %r24, %r8; add.u64 %rd33, %rd33, %rd10; add.u64 %rd33, %rd33, %rd11;
    setp.eq.u32 %p23,%r7,4; @%p23 bra CMP_HASH1_LEFT_I4;
    setp.eq.u32 %p23,%r7,8; @%p23 bra CMP_HASH1_LEFT_I8;
    mov.u32 %r25, 0;
CMP_HASH1_LOOP:
    setp.ge.u32 %p12, %r25, %r7; @%p12 bra HASH_MATCH;
    cvt.u64.u32 %rd34, %r25; add.u64 %rd35, %rd32, %rd34; add.u64 %rd36, %rd33, %rd34;
    ld.global.u8 %r26, [%rd35]; ld.global.u8 %r27, [%rd36];
    setp.ne.u32 %p13, %r26, %r27; @%p13 bra CHAIN_NEXT;
    add.u32 %r25, %r25, 1; bra CMP_HASH1_LOOP;
CMP_HASH1_LEFT_I4:
    ld.global.s32 %r37,[%rd32]; cvt.s64.s32 %rd64,%r37; bra CMP_HASH1_RIGHT;
CMP_HASH1_LEFT_I8:
    ld.global.u32 %r37,[%rd32]; ld.global.u32 %r38,[%rd32+4];
    cvt.u64.u32 %rd64,%r38; shl.b64 %rd64,%rd64,32; cvt.u64.u32 %rd66,%r37;
    or.b64 %rd64,%rd64,%rd66;
CMP_HASH1_RIGHT:
    setp.eq.u32 %p23,%r8,4; @%p23 bra CMP_HASH1_RIGHT_I4;
    ld.global.u32 %r37,[%rd33]; ld.global.u32 %r38,[%rd33+4];
    cvt.u64.u32 %rd65,%r38; shl.b64 %rd65,%rd65,32; cvt.u64.u32 %rd66,%r37;
    or.b64 %rd65,%rd65,%rd66; bra CMP_HASH1_INT_COMPARE;
CMP_HASH1_RIGHT_I4:
    ld.global.s32 %r37,[%rd33]; cvt.s64.s32 %rd65,%r37;
CMP_HASH1_INT_COMPARE:
    setp.ne.u64 %p13,%rd64,%rd65; @%p13 bra CHAIN_NEXT;
HASH_MATCH:
    mul.wide.u32 %rd37, %r14, 4; add.u64 %rd38, %rd15, %rd37; mov.u32 %r28, 1;
    atom.global.exch.b32 %r29, [%rd38], %r28;
    mul.wide.u32 %rd37, %r24, 4; add.u64 %rd38, %rd16, %rd37;
    atom.global.exch.b32 %r29, [%rd38], %r28;
    atom.global.add.u64 %rd39, [%rd18], 1; setp.eq.u64 %p15, %rd17, 0; @%p15 bra CHAIN_NEXT;
    add.u32 %r30, %r2, 1; cvt.u64.u32 %rd40, %r30; mul.lo.u64 %rd41, %rd39, %rd40;
    mov.u32 %r31, 0;
HASH_COPY_LEFT:
    setp.ge.u32 %p16, %r31, %r2; @%p16 bra HASH_WRITE_RIGHT;
    setp.eq.u32 %p17, %r3, 0; @%p17 mov.u32 %r32, %r14; @%p17 bra HASH_COPY_STORE;
    mul.lo.u32 %r33, %r14, %r2; add.u32 %r33, %r33, %r31;
    mul.wide.u32 %rd42, %r33, 4; add.u64 %rd43, %rd1, %rd42; ld.global.u32 %r32, [%rd43];
HASH_COPY_STORE:
    cvt.u64.u32 %rd44, %r31; add.u64 %rd45, %rd41, %rd44; mul.lo.u64 %rd45, %rd45, 4;
    add.u64 %rd45, %rd17, %rd45; st.global.u32 [%rd45], %r32;
    add.u32 %r31, %r31, 1; bra HASH_COPY_LEFT;
HASH_WRITE_RIGHT:
    cvt.u64.u32 %rd44, %r2; add.u64 %rd45, %rd41, %rd44; mul.lo.u64 %rd45, %rd45, 4;
    add.u64 %rd45, %rd17, %rd45; st.global.u32 [%rd45], %r24;
CHAIN_NEXT:
    mul.wide.u32 %rd46, %r24, 4; add.u64 %rd47, %rd14, %rd46; ld.global.u32 %r24, [%rd47];
    bra CHAIN_LOOP;
PROBE_NEXT:
    add.u32 %r14, %r14, %r15; bra PROBE_ROW;
PROBE_DONE:
    ret;
}

.visible .entry gpu_db_join_fixed_unmatched(
    .param .u64 coords, .param .u32 left_n, .param .u32 left_rels,
    .param .u32 has_coords, .param .u32 right_n,
    .param .u64 left_mask, .param .u64 right_mask,
    .param .u64 left_marks, .param .u64 right_marks,
    .param .u32 outer_left, .param .u32 outer_right,
    .param .u64 out, .param .u64 cursor)
{
    .reg .pred %p<20>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<64>;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r1, [left_n];
    ld.param.u32 %r2, [left_rels];
    ld.param.u32 %r3, [has_coords];
    ld.param.u32 %r4, [right_n];
    ld.param.u64 %rd2, [left_mask];
    ld.param.u64 %rd3, [right_mask];
    ld.param.u64 %rd4, [left_marks];
    ld.param.u64 %rd5, [right_marks];
    ld.param.u32 %r5, [outer_left];
    ld.param.u32 %r6, [outer_right];
    ld.param.u64 %rd6, [out];
    ld.param.u64 %rd7, [cursor];
    mov.u32 %r7, %tid.x;
    mov.u32 %r8, %ctaid.x;
    mov.u32 %r9, %ntid.x;
    mov.u32 %r10, %nctaid.x;
    mad.lo.u32 %r11, %r8, %r9, %r7;
    mul.lo.u32 %r12, %r10, %r9;
    mov.u64 %rd8, 18446744073709551615;
    setp.eq.u32 %p1, %r5, 0;
    @%p1 bra RIGHT_PHASE;
    mov.u32 %r13, %r11;
LEFT_LOOP:
    setp.ge.u32 %p2, %r13, %r1;
    @%p2 bra RIGHT_PHASE;
    mul.wide.u32 %rd9, %r13, 4;
    add.u64 %rd10, %rd4, %rd9;
    ld.global.u32 %r14, [%rd10];
    setp.ne.u32 %p3, %r14, 0;
    @%p3 bra LEFT_NEXT;
    setp.ne.u32 %p4, %r3, 0;
    @%p4 bra LEFT_ELIGIBLE;
    setp.eq.u64 %p5, %rd2, %rd8;
    @%p5 bra LEFT_ELIGIBLE;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.u32 %r14, [%rd10];
    setp.eq.u32 %p6, %r14, 0;
    @%p6 bra LEFT_NEXT;
LEFT_ELIGIBLE:
    atom.global.add.u64 %rd11, [%rd7], 1;
    setp.eq.u64 %p7, %rd6, 0;
    @%p7 bra LEFT_NEXT;
    add.u32 %r15, %r2, 1;
    cvt.u64.u32 %rd12, %r15;
    mul.lo.u64 %rd13, %rd11, %rd12;
    mov.u32 %r16, 0;
LEFT_COPY:
    setp.ge.u32 %p8, %r16, %r2;
    @%p8 bra LEFT_PAD;
    setp.eq.u32 %p9, %r3, 0;
    @%p9 mov.u32 %r17, %r13;
    @%p9 bra LEFT_STORE;
    mul.lo.u32 %r18, %r13, %r2;
    add.u32 %r18, %r18, %r16;
    mul.wide.u32 %rd14, %r18, 4;
    add.u64 %rd15, %rd1, %rd14;
    ld.global.u32 %r17, [%rd15];
LEFT_STORE:
    cvt.u64.u32 %rd16, %r16;
    add.u64 %rd17, %rd13, %rd16;
    mul.lo.u64 %rd17, %rd17, 4;
    add.u64 %rd17, %rd6, %rd17;
    st.global.u32 [%rd17], %r17;
    add.u32 %r16, %r16, 1;
    bra LEFT_COPY;
LEFT_PAD:
    cvt.u64.u32 %rd16, %r2;
    add.u64 %rd17, %rd13, %rd16;
    mul.lo.u64 %rd17, %rd17, 4;
    add.u64 %rd17, %rd6, %rd17;
    mov.u32 %r17, 4294967295;
    st.global.u32 [%rd17], %r17;
LEFT_NEXT:
    add.u32 %r13, %r13, %r12;
    bra LEFT_LOOP;

RIGHT_PHASE:
    setp.eq.u32 %p10, %r6, 0;
    @%p10 bra UNMATCHED_DONE;
    mov.u32 %r13, %r11;
RIGHT_LOOP:
    setp.ge.u32 %p11, %r13, %r4;
    @%p11 bra UNMATCHED_DONE;
    mul.wide.u32 %rd9, %r13, 4;
    add.u64 %rd10, %rd5, %rd9;
    ld.global.u32 %r14, [%rd10];
    setp.ne.u32 %p12, %r14, 0;
    @%p12 bra RIGHT_NEXT;
    setp.eq.u64 %p13, %rd3, %rd8;
    @%p13 bra RIGHT_ELIGIBLE;
    add.u64 %rd10, %rd3, %rd9;
    ld.global.u32 %r14, [%rd10];
    setp.eq.u32 %p14, %r14, 0;
    @%p14 bra RIGHT_NEXT;
RIGHT_ELIGIBLE:
    atom.global.add.u64 %rd11, [%rd7], 1;
    setp.eq.u64 %p15, %rd6, 0;
    @%p15 bra RIGHT_NEXT;
    add.u32 %r15, %r2, 1;
    cvt.u64.u32 %rd12, %r15;
    mul.lo.u64 %rd13, %rd11, %rd12;
    mov.u32 %r16, 0;
RIGHT_PAD_LOOP:
    setp.ge.u32 %p16, %r16, %r2;
    @%p16 bra RIGHT_VALUE;
    cvt.u64.u32 %rd16, %r16;
    add.u64 %rd17, %rd13, %rd16;
    mul.lo.u64 %rd17, %rd17, 4;
    add.u64 %rd17, %rd6, %rd17;
    mov.u32 %r17, 4294967295;
    st.global.u32 [%rd17], %r17;
    add.u32 %r16, %r16, 1;
    bra RIGHT_PAD_LOOP;
RIGHT_VALUE:
    cvt.u64.u32 %rd16, %r2;
    add.u64 %rd17, %rd13, %rd16;
    mul.lo.u64 %rd17, %rd17, 4;
    add.u64 %rd17, %rd6, %rd17;
    st.global.u32 [%rd17], %r13;
RIGHT_NEXT:
    add.u32 %r13, %r13, %r12;
    bra RIGHT_LOOP;
UNMATCHED_DONE:
    ret;
}
"#;

    if left_keys.is_empty()
        || left_keys.len() > 2
        || left_keys.len() != right_keys.len()
        || left_key_relations.len() != left_keys.len()
        || left_keys.iter().zip(right_keys).any(|(l, r)| {
            !matches!(l.width, 4 | 8 | 16 | 255)
                || !matches!(r.width, 4 | 8 | 16 | 255)
                || (l.width != r.width && !matches!((l.width, r.width), (4, 8) | (8, 4)))
        })
        || (left_keys.len() > 1 && left_keys.iter().any(|key| key.width == 255))
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(left_keys.len()));
    }
    let left_key_relation0 = left_key_relations[0];
    let left_key_relation1 = left_key_relations
        .get(1)
        .copied()
        .unwrap_or(left_key_relation0);
    let (left_n, left_rels, coords_ptr, has_coords) = match accumulated {
        Some(coords) => {
            if left_row_count != 0
                || left_key_relations
                    .iter()
                    .any(|relation| *relation >= coords.relation_count)
            {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    left_key_relation0 as usize,
                ));
            }
            (
                coords.row_count,
                coords.relation_count,
                coords.coordinates.as_ref().map_or(0, |b| b.ptr),
                1_u32,
            )
        }
        None => {
            if left_key_relations.iter().any(|relation| *relation != 0) {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    left_key_relation0 as usize,
                ));
            }
            (left_row_count, 1_u32, 0_u64, 0_u32)
        }
    };
    if left_eligibility.is_some_and(|m| m.row_count != left_row_count)
        || right_eligibility.is_some_and(|m| m.row_count != right_row_count)
        || (accumulated.is_some() && left_eligibility.is_some())
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(left_n as usize));
    }
    if persistent_left_marks.is_some_and(|marks| marks.row_count != left_n) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(left_n as usize));
    }
    if persistent_right_marks.is_some_and(|marks| marks.row_count != right_row_count) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            right_row_count as usize,
        ));
    }
    let gpu_id = ctx.metadata.gpu_id;
    let valid_key = |key: &CudaJoinPayloadKey<'_>, rows: u32| {
        if rows == 0 {
            return key.payload.metadata.gpu_id == gpu_id;
        }
        let values_end = if key.width == 255 {
            key.byte_offset
                .checked_add((u64::from(rows) + 1).saturating_mul(8))
        } else {
            key.byte_offset
                .checked_add(u64::from(rows).saturating_mul(u64::from(key.width)))
        };
        key.payload.metadata.gpu_id == gpu_id
            && values_end.is_some_and(|end| end <= key.payload.metadata.allocated_bytes)
            && (key.width != 255
                || key
                    .text_bytes_byte_offset
                    .and_then(|off| off.checked_add(key.text_bytes_len))
                    .is_some_and(|end| end <= key.payload.metadata.allocated_bytes))
            && key
                .validity_bitmap_offset
                .and_then(|off| off.checked_add(u64::from(rows).div_ceil(32) * 4))
                .is_none_or(|end| end <= key.payload.metadata.allocated_bytes)
    };
    // An accumulated key addresses original relation rows, whose cardinality is not `left_n`; its
    // descriptor bounds were already validated when that resident layout was admitted. Validate the
    // right side exactly and the left offset itself here; engine-side layout construction supplies the
    // original source allocation.
    if right_keys
        .iter()
        .any(|key| !valid_key(key, right_row_count))
        || left_keys
            .iter()
            .any(|key| key.payload.metadata.gpu_id != gpu_id)
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            right_row_count as usize,
        ));
    }
    if left_n == 0 && !outer_right {
        let mut relation_row_counts = accumulated.map_or_else(
            || vec![left_row_count],
            |coordinates| coordinates.relation_row_counts.clone(),
        );
        relation_row_counts.push(right_row_count);
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: left_rels + 1,
            relation_row_counts,
            allocated_bytes: 0,
        });
    }
    if right_row_count == 0 && !outer_left {
        let mut relation_row_counts = accumulated.map_or_else(
            || vec![left_row_count],
            |coordinates| coordinates.relation_row_counts.clone(),
        );
        relation_row_counts.push(right_row_count);
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: left_rels + 1,
            relation_row_counts,
            allocated_bytes: 0,
        });
    }

    let primary = ctx.primary_arc();
    primary.set_current()?;
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let dtoh = unsafe {
        primary
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let memset = unsafe {
        primary
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let left_marks_bytes = (left_n as usize).saturating_mul(4).max(1);
    let right_marks_bytes = (right_row_count as usize).saturating_mul(4).max(1);
    let owned_left_marks = if persistent_left_marks.is_none() {
        Some(primary.lease_device_buffer_owned(left_marks_bytes)?)
    } else {
        None
    };
    let left_marks_ptr = persistent_left_marks
        .map(|marks| marks.marks.device_ptr)
        .or_else(|| owned_left_marks.as_ref().map(|marks| marks.ptr))
        .expect("left match storage");
    let owned_right_marks = if persistent_right_marks.is_none() {
        Some(primary.lease_device_buffer_owned(right_marks_bytes)?)
    } else {
        None
    };
    let right_marks_ptr = persistent_right_marks
        .map(|marks| marks.marks.device_ptr)
        .or_else(|| owned_right_marks.as_ref().map(|marks| marks.ptr))
        .expect("right match storage");
    let cursor = primary.lease_device_buffer_owned(8)?;
    if persistent_left_marks.is_none() {
        check_cuda(unsafe { memset(left_marks_ptr, 0, left_marks_bytes) })?;
    }
    if persistent_right_marks.is_none() {
        check_cuda(unsafe { memset(right_marks_ptr, 0, right_marks_bytes) })?;
    }
    check_cuda(unsafe { memset(cursor.ptr, 0, 8) })?;
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let hash_build_fn = primary.cached_function(c"gpu_db_join_fixed_hash_build", &ptx)?;
    let hash_probe_fn = primary.cached_function(c"gpu_db_join_fixed_hash_probe_v2", &ptx)?;
    let unmatched_fn = primary.cached_function(c"gpu_db_join_fixed_unmatched", &ptx)?;
    let sentinel = u64::MAX;
    let key_args = |keys: &[CudaJoinPayloadKey<'_>], i: usize| {
        keys.get(i).map_or((0, 0, sentinel, 0), |key| {
            (
                key.payload.device_ptr,
                key.byte_offset,
                key.validity_bitmap_offset
                    .map_or(sentinel, |off| key.payload.device_ptr + off),
                u32::from(key.width),
            )
        })
    };
    let (lb0, lo0, lv0, lw0) = key_args(left_keys, 0);
    let (rb0, ro0, rv0, rw0) = key_args(right_keys, 0);
    let (lb1, lo1, lv1, lw1) = if left_keys[0].width == 255 {
        (
            left_keys[0].payload.device_ptr,
            left_keys[0]
                .text_bytes_byte_offset
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?,
            sentinel,
            0,
        )
    } else {
        key_args(left_keys, 1)
    };
    let (rb1, ro1, rv1, rw1) = if right_keys[0].width == 255 {
        (
            right_keys[0].payload.device_ptr,
            right_keys[0]
                .text_bytes_byte_offset
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?,
            sentinel,
            0,
        )
    } else {
        key_args(right_keys, 1)
    };
    let left_mask = left_eligibility.map_or(sentinel, |m| m.mask.ptr);
    let right_mask = right_eligibility.map_or(sentinel, |m| m.mask.ptr);
    const BLOCK: u32 = 256;
    let hash_slots = (right_row_count as usize)
        .saturating_mul(2)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
        .max(2);
    if hash_slots > u32::MAX as usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(hash_slots));
    }
    let hash_heads = primary.lease_device_buffer_owned(hash_slots.saturating_mul(4))?;
    let hash_next = primary.lease_device_buffer_owned(right_marks_bytes)?;
    let hash_mask = hash_slots.saturating_sub(1) as u32;
    {
        check_cuda(unsafe { memset(hash_heads.ptr, 0xff, hash_slots * 4) })?;
        check_cuda(unsafe { memset(hash_next.ptr, 0xff, right_marks_bytes) })?;
        let mut b0 = rb0;
        let mut b1 = ro0;
        let mut b2 = rv0;
        let mut b3 = rw0;
        let mut b4 = rb1;
        let mut b5 = ro1;
        let mut b6 = rv1;
        let mut b7 = rw1;
        let mut b8 = right_row_count;
        let mut b9 = right_mask;
        let mut b10 = hash_heads.ptr;
        let mut b11 = hash_next.ptr;
        let mut b12 = hash_mask;
        let mut build_args = [
            (&mut b0 as *mut u64).cast(),
            (&mut b1 as *mut u64).cast(),
            (&mut b2 as *mut u64).cast(),
            (&mut b3 as *mut u32).cast(),
            (&mut b4 as *mut u64).cast(),
            (&mut b5 as *mut u64).cast(),
            (&mut b6 as *mut u64).cast(),
            (&mut b7 as *mut u32).cast(),
            (&mut b8 as *mut u32).cast(),
            (&mut b9 as *mut u64).cast(),
            (&mut b10 as *mut u64).cast(),
            (&mut b11 as *mut u64).cast(),
            (&mut b12 as *mut u32).cast(),
        ];
        let build_grid = u64::from(right_row_count)
            .div_ceil(u64::from(BLOCK))
            .clamp(1, 65_535) as u32;
        if right_row_count > 0 {
            check_cuda(unsafe {
                launch(
                    hash_build_fn,
                    build_grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    build_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            })?;
        }
    }
    let unmatched_n = left_n.max(right_row_count);

    let run_phase = |out_ptr: u64| -> Result<(), CudaRuntimeProbeError> {
        if left_n > 0 && right_row_count > 0 {
            let mut h0 = coords_ptr;
            let mut h1 = left_n;
            let mut h2 = left_rels;
            let mut h3 = has_coords;
            let mut h4 = u64::from(left_key_relation1) | (u64::from(left_key_relation0) << 32);
            let mut h5 = lb0;
            let mut h6 = lo0;
            let mut h7 = lv0;
            let mut h8 = lw0;
            let mut h9 = rb0;
            let mut h10 = ro0;
            let mut h11 = rw0;
            let mut h12 = lb1;
            let mut h13 = lo1;
            let mut h14 = lv1;
            let mut h15 = lw1;
            let mut h16 = rb1;
            let mut h17 = ro1;
            let mut h18 = rw1;
            let mut h19 = left_mask;
            let mut h20 = hash_heads.ptr;
            let mut h21 = hash_next.ptr;
            let mut h22 = hash_mask;
            let mut h23 = left_marks_ptr;
            let mut h24 = right_marks_ptr;
            let mut h25 = out_ptr;
            let mut h26 = cursor.ptr;
            let mut hash_args = [
                (&mut h0 as *mut u64).cast(),
                (&mut h1 as *mut u32).cast(),
                (&mut h2 as *mut u32).cast(),
                (&mut h3 as *mut u32).cast(),
                (&mut h4 as *mut u64).cast(),
                (&mut h5 as *mut u64).cast(),
                (&mut h6 as *mut u64).cast(),
                (&mut h7 as *mut u64).cast(),
                (&mut h8 as *mut u32).cast(),
                (&mut h9 as *mut u64).cast(),
                (&mut h10 as *mut u64).cast(),
                (&mut h11 as *mut u32).cast(),
                (&mut h12 as *mut u64).cast(),
                (&mut h13 as *mut u64).cast(),
                (&mut h14 as *mut u64).cast(),
                (&mut h15 as *mut u32).cast(),
                (&mut h16 as *mut u64).cast(),
                (&mut h17 as *mut u64).cast(),
                (&mut h18 as *mut u32).cast(),
                (&mut h19 as *mut u64).cast(),
                (&mut h20 as *mut u64).cast(),
                (&mut h21 as *mut u64).cast(),
                (&mut h22 as *mut u32).cast(),
                (&mut h23 as *mut u64).cast(),
                (&mut h24 as *mut u64).cast(),
                (&mut h25 as *mut u64).cast(),
                (&mut h26 as *mut u64).cast(),
            ];
            let probe_grid = u64::from(left_n)
                .div_ceil(u64::from(BLOCK))
                .clamp(1, 65_535) as u32;
            check_cuda(unsafe {
                launch(
                    hash_probe_fn,
                    probe_grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    hash_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            })?;
        }
        let mut u0 = coords_ptr;
        let mut u1 = left_n;
        let mut u2 = left_rels;
        let mut u3 = has_coords;
        let mut u4 = right_row_count;
        let mut u5 = left_mask;
        let mut u6 = right_mask;
        let mut u7 = left_marks_ptr;
        let mut u8 = right_marks_ptr;
        let mut u9 = u32::from(outer_left);
        let mut u10 = u32::from(outer_right);
        let mut u11 = out_ptr;
        let mut u12 = cursor.ptr;
        let mut uargs = [
            (&mut u0 as *mut u64).cast(),
            (&mut u1 as *mut u32).cast(),
            (&mut u2 as *mut u32).cast(),
            (&mut u3 as *mut u32).cast(),
            (&mut u4 as *mut u32).cast(),
            (&mut u5 as *mut u64).cast(),
            (&mut u6 as *mut u64).cast(),
            (&mut u7 as *mut u64).cast(),
            (&mut u8 as *mut u64).cast(),
            (&mut u9 as *mut u32).cast(),
            (&mut u10 as *mut u32).cast(),
            (&mut u11 as *mut u64).cast(),
            (&mut u12 as *mut u64).cast(),
        ];
        if (outer_left || outer_right) && unmatched_n > 0 {
            check_cuda(unsafe {
                launch(
                    unmatched_fn,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    uargs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            })?;
        }
        Ok(())
    };

    run_phase(0)?;
    let mut total = 0_u64;
    check_cuda(unsafe { dtoh((&mut total as *mut u64).cast(), cursor.ptr, 8) })?;
    let total_u32 =
        u32::try_from(total).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if total == 0 {
        let mut relation_row_counts = accumulated.map_or_else(
            || vec![left_row_count],
            |coordinates| coordinates.relation_row_counts.clone(),
        );
        relation_row_counts.push(right_row_count);
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: left_rels + 1,
            relation_row_counts,
            allocated_bytes: 0,
        });
    }
    let output_bytes = total
        .checked_mul(u64::from(left_rels + 1))
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output = primary.lease_device_buffer_owned(output_bytes)?;
    check_cuda(unsafe { memset(cursor.ptr, 0, 8) })?;
    run_phase(output.ptr)?;
    let mut emitted = 0_u64;
    check_cuda(unsafe { dtoh((&mut emitted as *mut u64).cast(), cursor.ptr, 8) })?;
    if emitted != total {
        return Err(CudaRuntimeProbeError::InvalidInputLength(emitted as usize));
    }
    let mut relation_row_counts = accumulated.map_or_else(
        || vec![left_row_count],
        |coordinates| coordinates.relation_row_counts.clone(),
    );
    relation_row_counts.push(right_row_count);
    Ok(CudaJoinCoordinatesU32 {
        coordinates: Some(output),
        row_count: total_u32,
        relation_count: left_rels + 1,
        relation_row_counts,
        allocated_bytes: output_bytes as u64,
    })
}
