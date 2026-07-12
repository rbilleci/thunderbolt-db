use crate::{CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError, check_cuda};
use std::{ffi::c_void, sync::Arc};

/// E2.5c FUSED APPLY (2M+ push (b)): ONE kernel for the whole merged-apply device pass —
/// column scatter (each appended row's i32 values into every column section's headroom slots)
/// + created_by stamps + row-id stamps + the incremental PK hash-index CAS insert — replacing
///   the ~8 driver calls of the unfused chain (C column HtoDs + 2 stamp HtoDs + the 4-call index
///   insert) with 1 staging HtoD + 1 launch + 1 decline DtoH (which also serializes the stamps
///   ahead of the host's row_count publish, preserving the SV6 stamp-before-publish order).
///
/// Everything travels in ONE staging buffer read by the kernel (fixed 80B header + a per-column
/// dest-pointer table + col-major values + stamps + optional row ids); the decline flag lives
/// INSIDE the header (zeroed by the same HtoD that uploads it — no separate memset). Hash/pack
/// math is byte-identical to `gpu_db_resident_i32_index_insert` above. `pk_col == 0xFFFFFFFF`
/// skips the index insert (no cached device index); `has_row_ids == 0` skips row-id stamps.
/// ASCII-only.
///
/// Header layout (u64-aligned; offsets are load-bearing — `submit_i32_fused_apply` mirrors them):
///   0: u32 k             4: u32 num_cols     8: u32 pk_col      12: u32 base_row
///  16: u32 index_mask   20: u32 index_shift 24: u32 has_row_ids 28: u32 decline (kernel-set)
///  32: u64 index_ptr    40: u64 created_by_dest                48: u64 row_id_dest
///  56: u64 header_dest (the shard's device row-count word)     64: u64 header_value
///  72: u64 reserved
///  80: u64 col_dest[num_cols]
///  80 + num_cols*8:                     i32 values[num_cols][k] (col-major)
///  ^ + round8(num_cols*k*4):            u64 stamps[k]
///  ^ + k*8:                             u64 row_ids[k]   (present iff has_row_ids)
const FUSED_APPLY_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_fused_apply(
    .param .u64 staging_ptr
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<28>;

    ld.param.u64 %rd1, [staging_ptr];
    ld.global.u32 %r1, [%rd1+0];      // k
    ld.global.u32 %r2, [%rd1+4];      // num_cols

    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mad.lo.u32 %r8, %r6, %r7, %r5;    // j
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra DONE;

    // thread 0 publishes the shard's DEVICE row-count header (the unfused path's final
    // append_owned_chunks chunk) -- same launch, completed before the caller's sync DtoH.
    setp.ne.u32 %p2, %r8, 0;
    @%p2 bra HDRDONE;
    ld.global.u64 %rd6, [%rd1+56];
    ld.global.u64 %rd7, [%rd1+64];
    st.global.u64 [%rd6], %rd7;
HDRDONE:

    // values base = staging + 80 + num_cols*8
    cvt.u64.u32 %rd2, %r2;
    shl.b64 %rd3, %rd2, 3;
    add.u64 %rd4, %rd1, 80;           // col_dest table base
    add.u64 %rd5, %rd4, %rd3;         // values base

    // COLUMN SCATTER: for c in 0..num_cols: col_dest[c][j] = values[c*k + j]
    mov.u32 %r9, 0;                   // c
COLLOOP:
    setp.ge.u32 %p2, %r9, %r2;
    @%p2 bra COLDONE;
    // src = values_base + (c*k + j)*4
    mad.lo.u32 %r10, %r9, %r1, %r8;
    mul.wide.u32 %rd6, %r10, 4;
    add.u64 %rd7, %rd5, %rd6;
    ld.global.s32 %r11, [%rd7];
    // dst = col_dest[c] + j*4
    mul.wide.u32 %rd8, %r9, 8;
    add.u64 %rd9, %rd4, %rd8;
    ld.global.u64 %rd10, [%rd9];
    mul.wide.u32 %rd11, %r8, 4;
    add.u64 %rd12, %rd10, %rd11;
    st.global.s32 [%rd12], %r11;
    add.u32 %r9, %r9, 1;
    bra COLLOOP;
COLDONE:

    // stamps base = values_base + round8(num_cols*k*4)
    mul.lo.u32 %r12, %r2, %r1;
    mul.wide.u32 %rd13, %r12, 4;
    add.u64 %rd13, %rd13, 7;
    and.b64 %rd13, %rd13, 0xfffffffffffffff8;
    add.u64 %rd14, %rd5, %rd13;       // stamps base
    // created_by_dest[j] = stamps[j]
    mul.wide.u32 %rd15, %r8, 8;
    add.u64 %rd16, %rd14, %rd15;
    ld.global.u64 %rd17, [%rd16];
    ld.global.u64 %rd18, [%rd1+40];
    add.u64 %rd19, %rd18, %rd15;
    st.global.u64 [%rd19], %rd17;

    // row ids (optional): row_id_dest[j] = row_ids[j]
    ld.global.u32 %r13, [%rd1+24];
    setp.eq.u32 %p3, %r13, 0;
    @%p3 bra RIDONE;
    cvt.u64.u32 %rd20, %r1;
    shl.b64 %rd20, %rd20, 3;
    add.u64 %rd21, %rd14, %rd20;      // row_ids base = stamps base + k*8
    add.u64 %rd21, %rd21, %rd15;
    ld.global.u64 %rd22, [%rd21];
    ld.global.u64 %rd23, [%rd1+48];
    add.u64 %rd24, %rd23, %rd15;
    st.global.u64 [%rd24], %rd22;
RIDONE:

    // PK INDEX INSERT (optional): identical math to gpu_db_resident_i32_index_insert.
    ld.global.u32 %r14, [%rd1+8];     // pk_col
    setp.eq.u32 %p4, %r14, 4294967295;
    @%p4 bra DONE;
    // key = values[pk_col*k + j]
    mad.lo.u32 %r15, %r14, %r1, %r8;
    mul.wide.u32 %rd25, %r15, 4;
    add.u64 %rd26, %rd5, %rd25;
    ld.global.s32 %r16, [%rd26];
    // packed = (key << 32) | (base_row + j + 1)
    ld.global.u32 %r17, [%rd1+12];    // base_row
    add.u32 %r17, %r17, %r8;
    add.u32 %r17, %r17, 1;
    cvt.u64.u32 %rd6, %r17;
    cvt.u64.u32 %rd7, %r16;
    shl.b64 %rd8, %rd7, 32;
    or.b64 %rd9, %rd8, %rd6;
    // slot = (key * 0x9E3779B1) >> shift & mask
    ld.global.u32 %r18, [%rd1+16];    // mask
    ld.global.u32 %r19, [%rd1+20];    // shift
    ld.global.u64 %rd10, [%rd1+32];   // index_ptr
    mul.lo.u32 %r20, %r16, 2654435761;
    shr.u32 %r21, %r20, %r19;
    and.b32 %r21, %r21, %r18;
    mov.u32 %r22, 0;
INSLOOP:
    mul.wide.u32 %rd11, %r21, 8;
    add.u64 %rd12, %rd10, %rd11;
    mov.u64 %rd13, 0;
    atom.global.cas.b64 %rd27, [%rd12], %rd13, %rd9;
    setp.eq.u64 %p5, %rd27, 0;
    @%p5 bra DONE;
    // F3/U4: occupied slot (ANY key, incl. our own = a version twin) -> probe onward, place the
    // twin at the next empty slot (dup-tolerant visible-locate resolves versions at probe time).
    // Only a 256-probe overflow declines.
    add.u32 %r21, %r21, 1;
    and.b32 %r21, %r21, %r18;
    add.u32 %r22, %r22, 1;
    setp.ge.u32 %p5, %r22, 256;
    @%p5 bra DUP;
    bra INSLOOP;
DUP:
    mov.u32 %r23, 1;
    st.global.u32 [%rd1+28], %r23;    // decline flag lives in the header

DONE:
    ret;
}
"#;

/// M1 (charter-pure, ledger #24): the INCREMENTAL device-index INSERT kernel — lock-free
/// open-addressing insert of the k APPENDED keys into an existing device hash, so the device
/// index is APPEND-MAINTAINED (O(k)) instead of REBUILT O(rows) every wave (the measured
/// 305us/wave bottleneck: DtoH keys + host hash + HtoD). Each thread inserts one key via
/// `atom.global.cas.b64` (the charter's advance-on-failure discipline — NO spin-locks): claim an
/// empty slot (CAS 0 -> packed), while every occupied slot—including the same key's prior MVCC
/// version—advances the probe. Packing + hash are BYTE-IDENTICAL to
/// `build_int4_pk_hash_table_host` + the probe kernels
/// (`(key<<32)|(row+1)`, fib `key*0x9E3779B1 >> shift & mask`, 256 cap). Overflow (256 probes) ->
/// decline (the caller drops the cache -> a full rebuild at the grown size, mirroring the host
/// extend's load rule). ASCII-only.
const INDEX_INSERT_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_index_insert(
    .param .u64 index_ptr,
    .param .u32 table_mask,
    .param .u32 hash_shift,
    .param .u64 keys_ptr,
    .param .u32 key_count,
    .param .u32 base_row,
    .param .u64 decline_ptr
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<20>;

    ld.param.u64 %rd1, [index_ptr];
    ld.param.u32 %r1, [table_mask];
    ld.param.u32 %r2, [hash_shift];
    ld.param.u64 %rd2, [keys_ptr];
    ld.param.u32 %r3, [key_count];
    ld.param.u32 %r4, [base_row];
    ld.param.u64 %rd3, [decline_ptr];

    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mad.lo.u32 %r8, %r6, %r7, %r5;
    setp.ge.u32 %p1, %r8, %r3;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r8, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.s32 %r9, [%rd5];

    // packed = ((u64)key_bits << 32) | (base_row + tid + 1)
    add.u32 %r10, %r4, %r8;
    add.u32 %r10, %r10, 1;
    cvt.u64.u32 %rd6, %r10;
    cvt.u64.u32 %rd7, %r9;
    shl.b64 %rd8, %rd7, 32;
    or.b64 %rd9, %rd8, %rd6;

    // slot = (key * 0x9E3779B1) >> shift & mask
    mul.lo.u32 %r11, %r9, 2654435761;
    shr.u32 %r12, %r11, %r2;
    and.b32 %r12, %r12, %r1;
    mov.u32 %r13, 0;

INSLOOP:
    mul.wide.u32 %rd10, %r12, 8;
    add.u64 %rd11, %rd1, %rd10;
    // atomicCAS(slot, 0, packed) -> old
    mov.u64 %rd12, 0;
    atom.global.cas.b64 %rd13, [%rd11], %rd12, %rd9;
    setp.eq.u64 %p2, %rd13, 0;
    @%p2 bra DONE;
    // F3/U4: an occupied slot (ANY key, INCLUDING our own = an MVCC version twin) is a collision
    // -> probe onward to the next empty slot and place the twin there. The dup-tolerant
    // visible-locate walks the whole chain and resolves the snapshot-visible version at probe time,
    // so a key legitimately holds >1 physical slot (old + new) until the old drops below the GC
    // boundary and a rebuild reclaims it. Only a 256-probe overflow declines (rebuild at grown size).
    add.u32 %r12, %r12, 1;
    and.b32 %r12, %r12, %r1;
    add.u32 %r13, %r13, 1;
    setp.ge.u32 %p3, %r13, 256;
    @%p3 bra DUP;
    bra INSLOOP;

DUP:
    mov.u32 %r15, 1;
    st.global.u32 [%rd3], %r15;

DONE:
    ret;
}
"#;

// COMPOUND KEYS (TYPE-COVERAGE #14 Track 3, charter close + wider types): fold a compound key's ORDERED
// key columns into the 32-bit surrogate FINGERPRINT entirely ON THE DEVICE, so the index rebuild never
// reads the raw resident key columns back to the host to hash them (the host reads only the derived
// fingerprint buffer, exactly as the single-column build reads its one key column). Byte-identical to
// the host `compound_key_fingerprint`/`sql_value_key_words` (FNV-1a: h=0x811C9DC5; per WORD h^=w;
// h*=0x01000193; h=rotl(h,13)+0x9E3779B1) so the device-built index and the host-derived probe needle
// agree. Each key column contributes `widths[k]` consecutive i32 words (Int4/Date/Int2 -> 1; Int8/
// Timestamp -> 2 = the i64 section's LE lo/hi), read column-major at `base + offsets[k] + row*widths[k]*4`
// with `widths[k]` words folded in ascending (little-endian) order. One thread per row. The text branch's
// byte loop hashes raw bytes (`ld.global.u8` zero-extends), so it is byte-exact for all UTF-8, not just ASCII.
//
// TEXT key columns are variable-length (no fixed section width), so `widths[k] == 0` is the TEXT SENTINEL:
// the kernel then treats `offsets[k]` as the byte offset of that column's OFFSETS array (n+1 i64 entries)
// and `blob_offsets[k]` as the byte offset of its blob, reads the row's [start,end) byte span, folds ONE
// word = the FNV-1a hash of those bytes (h=0x811C9DC5; per BYTE h^=b; h*=0x01000193 — byte-identical to the
// host `fnv1a_bytes`), then applies the same outer per-word mix. `blob_offsets` is ignored for fixed-width
// columns (pass 0). One thread per row.
const COMPOUND_FOLD_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_compound_fold_fingerprints(
    .param .u64 base_ptr,
    .param .u64 offsets_ptr,
    .param .u64 widths_ptr,
    .param .u64 blob_offsets_ptr,
    .param .u32 ncols,
    .param .u32 row_count,
    .param .u64 out_ptr
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<40>;

    ld.param.u64 %rd1, [base_ptr];
    ld.param.u64 %rd2, [offsets_ptr];
    ld.param.u64 %rd3, [widths_ptr];
    ld.param.u64 %rd20, [blob_offsets_ptr];
    ld.param.u32 %r1, [ncols];
    ld.param.u32 %r2, [row_count];
    ld.param.u64 %rd4, [out_ptr];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;      // row = global thread id
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;

    mov.u32 %r7, 2166136261;            // fp = 0x811C9DC5 (FNV offset basis)
    mov.u32 %r8, 0;                     // k = 0 (column index)
    setp.ge.u32 %p2, %r8, %r1;
    @%p2 bra STORE;                     // ncols == 0 -> empty fold (defensive)

FOLDLOOP:
    mul.wide.u32 %rd5, %r8, 8;          // k * 8 (offsets are u64)
    add.u64 %rd6, %rd2, %rd5;
    ld.global.u64 %rd7, [%rd6];         // off = offsets[k]
    mul.wide.u32 %rd8, %r8, 4;          // k * 4 (widths are u32)
    add.u64 %rd9, %rd3, %rd8;
    ld.global.u32 %r9, [%rd9];          // w = widths[k] (words in this column; 0 = TEXT sentinel)
    setp.eq.u32 %p4, %r9, 0;
    @%p4 bra TEXTCOL;                   // w == 0 -> variable-length text column
    mul.lo.u32 %r10, %r6, %r9;          // row * w (word index of this column's row-0-relative start)
    mul.wide.u32 %rd10, %r10, 4;        // row * w * 4 (byte offset)
    add.u64 %rd11, %rd1, %rd7;
    add.u64 %rd12, %rd11, %rd10;        // col_base = base + off + row*w*4
    mov.u32 %r11, 0;                    // j = 0 (word within column)

WORDLOOP:
    mul.wide.u32 %rd13, %r11, 4;        // j * 4
    add.u64 %rd14, %rd12, %rd13;        // addr = col_base + j*4
    ld.global.s32 %r12, [%rd14];        // word value (i32)
    xor.b32 %r7, %r7, %r12;             // fp ^= w
    mul.lo.u32 %r7, %r7, 16777619;      // fp *= 0x01000193 (FNV prime)
    shl.b32 %r13, %r7, 13;              // rotl(fp, 13)
    shr.b32 %r14, %r7, 19;
    or.b32 %r7, %r13, %r14;
    add.u32 %r7, %r7, 2654435761;       // fp += 0x9E3779B1
    add.u32 %r11, %r11, 1;
    setp.lt.u32 %p3, %r11, %r9;
    @%p3 bra WORDLOOP;
    bra NEXTCOL;

TEXTCOL:
    // offsets array for this column starts at base + off; entries are i64, one per row + 1.
    add.u64 %rd21, %rd1, %rd7;          // off_arr = base + off
    mul.wide.u32 %rd22, %r6, 8;         // row * 8
    add.u64 %rd23, %rd21, %rd22;        // &offsets[row]
    ld.global.u64 %rd24, [%rd23];       // start = offsets[row]
    ld.global.u64 %rd25, [%rd23+8];     // end   = offsets[row+1]
    // blob base = base + blob_offsets[k]
    add.u64 %rd26, %rd20, %rd5;         // &blob_offsets[k] (rd5 = k*8 from above)
    ld.global.u64 %rd27, [%rd26];       // blob_off = blob_offsets[k]
    add.u64 %rd28, %rd1, %rd27;         // blob_base = base + blob_off
    mov.u32 %r15, 2166136261;           // h_text = 0x811C9DC5 (inner byte FNV)
    mov.u64 %rd29, %rd24;               // i = start
BYTELOOP:
    setp.ge.u64 %p5, %rd29, %rd25;      // i >= end ?
    @%p5 bra BYTEDONE;
    add.u64 %rd30, %rd28, %rd29;        // &blob[i]
    ld.global.u8 %r16, [%rd30];         // byte
    xor.b32 %r15, %r15, %r16;           // h_text ^= b
    mul.lo.u32 %r15, %r15, 16777619;    // h_text *= 0x01000193
    add.u64 %rd29, %rd29, 1;
    bra BYTELOOP;
BYTEDONE:
    // fold the single text word h_text into fp with the outer per-word mix.
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

STORE:
    mul.wide.u32 %rd15, %r6, 4;         // row * 4
    add.u64 %rd16, %rd4, %rd15;         // out + row*4
    st.global.u32 [%rd16], %r7;

DONE:
    ret;
}
"#;

/// One fused merged-apply pass (see [`FUSED_APPLY_PTX`]): everything the kernel needs, staged
/// into one buffer by [`CudaResidentDeviceMemory::submit_i32_fused_apply`].
pub struct FusedApplyRequest<'a> {
    /// Absolute device address of each column's FIRST new slot (shard base + section offset +
    /// row_count * 4), catalog order. `values` is col-major over exactly these columns.
    pub col_dests: &'a [u64],
    /// Col-major i32 values: `values[c * k + j]` = row j's value for column c.
    pub values: &'a [i32],
    /// One created_by birth stamp per appended row (k of them).
    pub stamps: &'a [u64],
    /// Absolute device address of `created_by[first_slot]`.
    pub created_by_dest: u64,
    /// Optional row-id stamps + the absolute device address of `row_id[first_slot]`.
    pub row_ids: Option<(&'a [u64], u64)>,
    /// Optional PK hash-index insert: (index_ptr, table_mask, hash_shift, pk_col, base_row).
    pub index: Option<(u64, u32, u32, u32, u32)>,
    /// The shard's device row-count header word (absolute address) and the value to publish
    /// there (`row_count + k`) — the unfused path's final append chunk, written by thread 0.
    pub header_dest: u64,
    pub header_value: u64,
}

impl CudaResidentDeviceMemory {
    /// E2.5c FUSED APPLY (2M+ push (b)): run the WHOLE merged-apply device pass in one staging
    /// HtoD + one launch + one decline DtoH (which also completes the launch, so the caller's
    /// row_count publish happens strictly after the stamps land — the SV6 order). Returns
    /// `true` when the index insert hit its bounded-probe overflow (caller declines the index entry, same
    /// contract as [`Self::submit_i32_index_insert`]). Any error leaves the shard partially
    /// mutated in INVISIBLE headroom only — the caller must not publish and must re-admit,
    /// exactly the `append_owned_chunks` partial-failure contract.
    pub fn submit_i32_fused_apply(
        &self,
        request: &FusedApplyRequest<'_>,
    ) -> Result<bool, CudaRuntimeProbeError> {
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

        let num_cols = request.col_dests.len();
        let k = request.stamps.len();
        if k == 0 || num_cols == 0 {
            return Ok(false);
        }
        if request.values.len() != num_cols * k {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                request.values.len(),
            ));
        }
        if let Some((row_ids, _)) = request.row_ids {
            if row_ids.len() != k {
                return Err(CudaRuntimeProbeError::InvalidInputLength(row_ids.len()));
            }
        }
        let k_u32 = u32::try_from(k).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(k))?;
        let cols_u32 = u32::try_from(num_cols)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(num_cols))?;

        // Assemble the staging image (header layout documented at FUSED_APPLY_PTX).
        let values_bytes = num_cols * k * 4;
        let values_padded = values_bytes.div_ceil(8) * 8;
        let header_bytes = 80 + num_cols * 8;
        let total = header_bytes
            + values_padded
            + k * 8
            + if request.row_ids.is_some() { k * 8 } else { 0 };
        let mut staging = vec![0_u8; total];
        let (pk_col, base_row, index_ptr, index_mask, index_shift) = match request.index {
            Some((ptr, mask, shift, pk_col, base_row)) => (pk_col, base_row, ptr, mask, shift),
            None => (u32::MAX, 0, 0, 0, 0),
        };
        staging[0..4].copy_from_slice(&k_u32.to_le_bytes());
        staging[4..8].copy_from_slice(&cols_u32.to_le_bytes());
        staging[8..12].copy_from_slice(&pk_col.to_le_bytes());
        staging[12..16].copy_from_slice(&base_row.to_le_bytes());
        staging[16..20].copy_from_slice(&index_mask.to_le_bytes());
        staging[20..24].copy_from_slice(&index_shift.to_le_bytes());
        staging[24..28].copy_from_slice(&u32::from(request.row_ids.is_some()).to_le_bytes());
        // 28..32 = decline flag, zeroed by this very upload (no separate memset).
        staging[32..40].copy_from_slice(&index_ptr.to_le_bytes());
        staging[40..48].copy_from_slice(&request.created_by_dest.to_le_bytes());
        let row_id_dest = request.row_ids.map(|(_, dest)| dest).unwrap_or(0);
        staging[48..56].copy_from_slice(&row_id_dest.to_le_bytes());
        staging[56..64].copy_from_slice(&request.header_dest.to_le_bytes());
        staging[64..72].copy_from_slice(&request.header_value.to_le_bytes());
        // 72..80 reserved (zero).
        for (c, dest) in request.col_dests.iter().enumerate() {
            staging[80 + c * 8..80 + c * 8 + 8].copy_from_slice(&dest.to_le_bytes());
        }
        let values_off = header_bytes;
        for (i, value) in request.values.iter().enumerate() {
            staging[values_off + i * 4..values_off + i * 4 + 4]
                .copy_from_slice(&value.to_le_bytes());
        }
        let stamps_off = values_off + values_padded;
        for (i, stamp) in request.stamps.iter().enumerate() {
            staging[stamps_off + i * 8..stamps_off + i * 8 + 8]
                .copy_from_slice(&stamp.to_le_bytes());
        }
        if let Some((row_ids, _)) = request.row_ids {
            let row_ids_off = stamps_off + k * 8;
            for (i, row_id) in row_ids.iter().enumerate() {
                staging[row_ids_off + i * 8..row_ids_off + i * 8 + 8]
                    .copy_from_slice(&row_id.to_le_bytes());
            }
        }

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

        let staging_guard = primary.lease_device_buffer_owned(total)?;
        let mut ptx = Vec::with_capacity(FUSED_APPLY_PTX.len() + 1);
        ptx.extend_from_slice(FUSED_APPLY_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_resident_i32_fused_apply", &ptx)?;

        check_cuda(unsafe {
            cu_memcpy_htod(staging_guard.ptr, staging.as_ptr().cast::<c_void>(), total)
        })?;
        let mut staging_arg = staging_guard.ptr;
        let mut args = [(&mut staging_arg as *mut u64).cast::<c_void>()];
        let threads_per_block: u32 = 128;
        let blocks = k_u32.div_ceil(threads_per_block);
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
        // The decline read completes the launch (default stream), which is what lets the caller
        // publish row_count AFTER the stamps are device-visible.
        let mut decline = 0_u32;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                (&mut decline as *mut u32).cast::<c_void>(),
                staging_guard.ptr + 28,
                std::mem::size_of::<u32>(),
            )
        })?;
        Ok(decline != 0)
    }

    /// M1 (ledger #24): insert `keys` (the APPENDED tail, at device rows `base_row..`) INTO this
    /// device hash index IN PLACE via the lock-free `index_insert` kernel. Same-key MVCC versions
    /// occupy distinct probe slots; `true` means the 256-probe bound was exhausted, so the caller
    /// transitions the cache to the DECLINED
    /// state (monotone, like the host path). The caller MUST enforce the load rule
    /// (`2*(base_row+keys.len()) <= table_size`) BEFORE calling (else drop + rebuild). Synchronous.
    pub fn submit_i32_index_insert(
        &self,
        index: &Arc<CudaResidentDeviceMemory>,
        table_mask: u32,
        hash_shift: u32,
        keys: &[i32],
        base_row: u32,
    ) -> Result<bool, CudaRuntimeProbeError> {
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

        if keys.is_empty() {
            return Ok(false);
        }
        if index.device_ptr() == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let key_count = u32::try_from(keys.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(keys.len()))?;
        let keys_bytes = keys
            .len()
            .checked_mul(std::mem::size_of::<i32>())
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

        let keys_guard = primary.lease_device_buffer_owned(keys_bytes)?;
        let decline_guard = primary.lease_device_buffer_owned(std::mem::size_of::<u32>())?;

        let mut ptx = Vec::with_capacity(INDEX_INSERT_PTX.len() + 1);
        ptx.extend_from_slice(INDEX_INSERT_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_resident_i32_index_insert", &ptx)?;

        check_cuda(unsafe {
            cu_memcpy_htod(keys_guard.ptr, keys.as_ptr().cast::<c_void>(), keys_bytes)
        })?;
        check_cuda(unsafe { cu_memset_d8(decline_guard.ptr, 0, std::mem::size_of::<u32>()) })?;

        let mut index_arg = index.device_ptr();
        let mut mask_arg = table_mask;
        let mut shift_arg = hash_shift;
        let mut keys_arg = keys_guard.ptr;
        let mut count_arg = key_count;
        let mut base_arg = base_row;
        let mut decline_arg = decline_guard.ptr;
        let mut args = [
            (&mut index_arg as *mut u64).cast::<c_void>(),
            (&mut mask_arg as *mut u32).cast::<c_void>(),
            (&mut shift_arg as *mut u32).cast::<c_void>(),
            (&mut keys_arg as *mut u64).cast::<c_void>(),
            (&mut count_arg as *mut u32).cast::<c_void>(),
            (&mut base_arg as *mut u32).cast::<c_void>(),
            (&mut decline_arg as *mut u64).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = key_count.div_ceil(threads_per_block);
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
        // FUSE (driver-call trim): the explicit stream-synchronize is REDUNDANT — the decline DtoH
        // below is a BLOCKING `cuMemcpyDtoH` on the null stream, which already waits for the kernel
        // before copying. Dropping the sync removes one driver call per wave, zero semantic change.
        // FENCE INVARIANT (do not break): this fences the kernel's `atom.cas` index mutation (so the
        // pooled guards below are not recycled mid-flight) ONLY because the launch is on the NULL
        // stream and this 4-byte blocking DtoH runs on that same stream before return (keys.is_empty
        // early-returns above, so it is always a real transfer). Keep both true, or re-add the sync.
        let mut decline = [0u32; 1];
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                decline.as_mut_ptr().cast::<c_void>(),
                decline_guard.ptr,
                std::mem::size_of::<u32>(),
            )
        })?;
        Ok(decline[0] != 0)
    }

    /// COMPOUND KEYS (charter close + wider types): fold a compound key's columns of the shard buffer at
    /// `base_ptr` into per-row 32-bit fingerprints ON THE DEVICE (`COMPOUND_FOLD_PTX`), returning the
    /// `row_count` fingerprints. Each key column `k` contributes `widths[k]` consecutive i32 words at its
    /// capacity-strided section `offsets[k]` (1 for i32-section, 2 for the i64 section). This lets the
    /// PK-index rebuild derive the compound key on the GPU instead of reading the raw resident key columns
    /// back to the host to hash them — the host then reads only this derived fingerprint column, exactly
    /// as the single-column build reads its one key column. Byte-identical to the host
    /// `compound_key_fingerprint` / `sql_value_key_words`. `offsets.len() == widths.len() ==
    /// blob_offsets.len()`. A TEXT key column sets `widths[k] == 0` (the sentinel); `offsets[k]` is then
    /// the byte offset of its OFFSETS array and `blob_offsets[k]` the byte offset of its blob (ignored,
    /// pass 0, for fixed-width columns). Synchronous (the blocking output DtoH on the null stream fences
    /// the launch).
    pub fn submit_compound_fold_fingerprints(
        &self,
        base_ptr: u64,
        offsets: &[u64],
        widths: &[u32],
        blob_offsets: &[u64],
        row_count: usize,
    ) -> Result<Vec<i32>, CudaRuntimeProbeError> {
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

        if row_count == 0 {
            return Ok(Vec::new());
        }
        if offsets.is_empty()
            || base_ptr == 0
            || offsets.len() != widths.len()
            || offsets.len() != blob_offsets.len()
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(offsets.len()));
        }
        let ncols = u32::try_from(offsets.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(offsets.len()))?;
        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(row_count))?;
        let offsets_bytes = offsets
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let widths_bytes = widths
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let blob_offsets_bytes = blob_offsets
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let out_bytes = row_count
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

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

        let offsets_guard = primary.lease_device_buffer_owned(offsets_bytes)?;
        let widths_guard = primary.lease_device_buffer_owned(widths_bytes)?;
        let blob_offsets_guard = primary.lease_device_buffer_owned(blob_offsets_bytes)?;
        let out_guard = primary.lease_device_buffer_owned(out_bytes)?;

        let mut ptx = Vec::with_capacity(COMPOUND_FOLD_PTX.len() + 1);
        ptx.extend_from_slice(COMPOUND_FOLD_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_compound_fold_fingerprints", &ptx)?;

        check_cuda(unsafe {
            cu_memcpy_htod(
                offsets_guard.ptr,
                offsets.as_ptr().cast::<c_void>(),
                offsets_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_htod(
                widths_guard.ptr,
                widths.as_ptr().cast::<c_void>(),
                widths_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_htod(
                blob_offsets_guard.ptr,
                blob_offsets.as_ptr().cast::<c_void>(),
                blob_offsets_bytes,
            )
        })?;

        let mut base_arg = base_ptr;
        let mut offsets_arg = offsets_guard.ptr;
        let mut widths_arg = widths_guard.ptr;
        let mut blob_offsets_arg = blob_offsets_guard.ptr;
        let mut ncols_arg = ncols;
        let mut rows_arg = row_count_u32;
        let mut out_arg = out_guard.ptr;
        let mut args = [
            (&mut base_arg as *mut u64).cast::<c_void>(),
            (&mut offsets_arg as *mut u64).cast::<c_void>(),
            (&mut widths_arg as *mut u64).cast::<c_void>(),
            (&mut blob_offsets_arg as *mut u64).cast::<c_void>(),
            (&mut ncols_arg as *mut u32).cast::<c_void>(),
            (&mut rows_arg as *mut u32).cast::<c_void>(),
            (&mut out_arg as *mut u64).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = row_count_u32.div_ceil(threads_per_block);
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
        // The blocking output DtoH on the null stream fences the kernel (same discipline as
        // `submit_i32_index_insert`'s decline read); out_bytes > 0 here, so it always transfers.
        let mut out = vec![0_i32; row_count];
        check_cuda(unsafe {
            cu_memcpy_dtoh(out.as_mut_ptr().cast::<c_void>(), out_guard.ptr, out_bytes)
        })?;
        Ok(out)
    }
}
