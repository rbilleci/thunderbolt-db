use crate::{launch_on_pooled_stream, CudaResidentDeviceMemory, CudaRuntimeProbeError};
use std::ffi::c_void;

/// One typed column in the canonical visible-source digest. Offsets address the resident payload;
/// an optional validity bitmap uses the ordinary LSB-first `1 = present` representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaVisibleDigestColumn {
    Fixed {
        byte_offset: u64,
        width_bytes: u32,
        validity_byte_offset: Option<u64>,
    },
    Bool {
        bitmap_byte_offset: u64,
        validity_byte_offset: Option<u64>,
    },
    Text {
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        validity_byte_offset: Option<u64>,
    },
}

/// Constant-size, order-independent digest of a visible logical row set. Four independent 64-bit
/// lanes are reduced modulo 2^64; stable row identity is part of every row hash, so physical shard
/// ordering and hot/cold partitioning do not affect the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaVisibleSourceDigest {
    pub visible_rows: u64,
    pub lanes: [u64; 4],
}

const DESCRIPTOR_WORDS: usize = 6;
const KIND_FIXED: u64 = 1;
const KIND_BOOL: u64 = 2;
const KIND_TEXT: u64 = 3;
const NO_VALIDITY: u64 = u64::MAX;

fn checked_region_ptr(
    resident: &CudaResidentDeviceMemory,
    region: Option<(&CudaResidentDeviceMemory, u64)>,
    row_count: u64,
) -> Result<u64, CudaRuntimeProbeError> {
    let Some((memory, offset)) = region else {
        return Ok(0);
    };
    if memory.metadata().gpu_id != resident.metadata().gpu_id
        || !std::ptr::eq(memory.primary(), resident.primary())
        || !offset.is_multiple_of(std::mem::align_of::<u64>() as u64)
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(row_count).unwrap_or(usize::MAX),
        ));
    }
    let bytes = row_count
        .checked_mul(std::mem::size_of::<u64>() as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let end = offset
        .checked_add(bytes)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if end > memory.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(end).unwrap_or(usize::MAX),
        ));
    }
    memory
        .device_ptr()
        .checked_add(offset)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))
}

fn checked_payload_window(
    resident: &CudaResidentDeviceMemory,
    offset: u64,
    bytes: u64,
) -> Result<(), CudaRuntimeProbeError> {
    let end = offset
        .checked_add(bytes)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if end > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(end).unwrap_or(usize::MAX),
        ));
    }
    Ok(())
}

pub(super) fn launch_cuda_resident_visible_digest(
    resident: &CudaResidentDeviceMemory,
    row_count: u64,
    columns: &[CudaVisibleDigestColumn],
    row_ids: Option<(&CudaResidentDeviceMemory, u64)>,
    read_txn_id: i64,
    deleted_by: Option<(&CudaResidentDeviceMemory, u64)>,
    created_by: Option<(&CudaResidentDeviceMemory, u64)>,
) -> Result<CudaVisibleSourceDigest, CudaRuntimeProbeError> {
    type CuMemcpyHtoDAsync = unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> i32;
    type CuMemsetD8Async = unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> i32;
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

    if row_count > u64::from(u32::MAX) || columns.len() > u32::MAX as usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    if row_count == 0 {
        return Ok(CudaVisibleSourceDigest {
            visible_rows: 0,
            lanes: [0; 4],
        });
    }
    if columns.is_empty() || row_ids.is_none() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(row_count).unwrap_or(usize::MAX),
        ));
    }

    let row_ids_ptr = checked_region_ptr(resident, row_ids, row_count)?;
    let deleted_ptr = checked_region_ptr(resident, deleted_by, row_count)?;
    let created_ptr = checked_region_ptr(resident, created_by, row_count)?;
    let bitmap_bytes = row_count
        .checked_add(7)
        .map(|bits| bits / 8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut descriptors = Vec::with_capacity(columns.len() * DESCRIPTOR_WORDS);
    for column in columns {
        let (kind, offset, width, aux_offset, aux_len, validity) = match *column {
            CudaVisibleDigestColumn::Fixed {
                byte_offset,
                width_bytes,
                validity_byte_offset,
            } => {
                if !matches!(width_bytes, 4 | 8 | 16)
                    || !byte_offset.is_multiple_of(std::mem::align_of::<u32>() as u64)
                {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(
                        usize::try_from(byte_offset).unwrap_or(usize::MAX),
                    ));
                }
                let bytes = row_count
                    .checked_mul(u64::from(width_bytes))
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                checked_payload_window(resident, byte_offset, bytes)?;
                (
                    KIND_FIXED,
                    byte_offset,
                    u64::from(width_bytes / 4),
                    0,
                    0,
                    validity_byte_offset.unwrap_or(NO_VALIDITY),
                )
            }
            CudaVisibleDigestColumn::Bool {
                bitmap_byte_offset,
                validity_byte_offset,
            } => {
                checked_payload_window(resident, bitmap_byte_offset, bitmap_bytes)?;
                (
                    KIND_BOOL,
                    bitmap_byte_offset,
                    0,
                    0,
                    0,
                    validity_byte_offset.unwrap_or(NO_VALIDITY),
                )
            }
            CudaVisibleDigestColumn::Text {
                offsets_byte_offset,
                bytes_byte_offset,
                bytes_len,
                validity_byte_offset,
            } => {
                if !offsets_byte_offset.is_multiple_of(std::mem::align_of::<u64>() as u64) {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(
                        usize::try_from(offsets_byte_offset).unwrap_or(usize::MAX),
                    ));
                }
                let offsets_bytes = row_count
                    .checked_add(1)
                    .and_then(|rows| rows.checked_mul(std::mem::size_of::<u64>() as u64))
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                checked_payload_window(resident, offsets_byte_offset, offsets_bytes)?;
                checked_payload_window(resident, bytes_byte_offset, bytes_len)?;
                (
                    KIND_TEXT,
                    offsets_byte_offset,
                    0,
                    bytes_byte_offset,
                    bytes_len,
                    validity_byte_offset.unwrap_or(NO_VALIDITY),
                )
            }
        };
        if validity != NO_VALIDITY {
            checked_payload_window(resident, validity, bitmap_bytes)?;
        }
        descriptors.extend_from_slice(&[kind, offset, width, aux_offset, aux_len, validity]);
    }

    const OUTPUT_WORDS: usize = 6;
    const OUTPUT_BYTES: usize = OUTPUT_WORDS * std::mem::size_of::<u64>();
    let descriptor_bytes = std::mem::size_of_val(descriptors.as_slice());
    let primary = resident.primary();
    primary.set_current()?;
    let descriptor_guard = primary.lease_device_buffer(descriptor_bytes)?;
    let cu_memcpy_htod_async = unsafe {
        primary
            .lib()
            .get::<CuMemcpyHtoDAsync>(b"cuMemcpyHtoDAsync_v2\0")
            .or_else(|_| {
                primary
                    .lib()
                    .get::<CuMemcpyHtoDAsync>(b"cuMemcpyHtoDAsync\0")
            })
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8_async = unsafe {
        primary
            .lib()
            .get::<CuMemsetD8Async>(b"cuMemsetD8Async\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_visible_source_digest", &ptx)?;

    let mut output_bytes = [0_u8; OUTPUT_BYTES];
    const BLOCK: u32 = 256;
    let grid = row_count.div_ceil(u64::from(BLOCK)).clamp(1, 4096) as u32;
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        let copy_rc = unsafe {
            cu_memcpy_htod_async(
                descriptor_guard.ptr,
                descriptors.as_ptr().cast::<c_void>(),
                descriptor_bytes,
                stream,
            )
        };
        if copy_rc != 0 {
            return copy_rc;
        }
        let memset_rc = unsafe { cu_memset_d8_async(output_ptr, 0, OUTPUT_BYTES, stream) };
        if memset_rc != 0 {
            return memset_rc;
        }
        let mut payload_arg = resident.device_ptr();
        let mut descriptors_arg = descriptor_guard.ptr;
        let mut columns_arg = columns.len() as u32;
        let mut rows_arg = row_count;
        let mut row_ids_arg = row_ids_ptr;
        let mut deleted_arg = deleted_ptr;
        let mut created_arg = created_ptr;
        let mut boundary_arg = read_txn_id;
        let mut output_arg = output_ptr;
        let mut args = [
            (&mut payload_arg as *mut u64).cast::<c_void>(),
            (&mut descriptors_arg as *mut u64).cast::<c_void>(),
            (&mut columns_arg as *mut u32).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut row_ids_arg as *mut u64).cast::<c_void>(),
            (&mut deleted_arg as *mut u64).cast::<c_void>(),
            (&mut created_arg as *mut u64).cast::<c_void>(),
            (&mut boundary_arg as *mut i64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                function,
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
        }
    })?;

    let words = output_bytes
        .chunks_exact(8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("8-byte digest word")))
        .collect::<Vec<_>>();
    if words[5] != 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(words[5] as usize));
    }
    Ok(CudaVisibleSourceDigest {
        visible_rows: words[0],
        lanes: [words[1], words[2], words[3], words[4]],
    })
}

// One thread grid-strides over rows. Visible rows are hashed from stable row id plus catalog-ordered
// typed values/NULL markers. Four independent 64-bit lanes are block-reduced before five global
// atomics, avoiding the per-row atomic bottleneck. The reduction is commutative, so shard/chunk
// partition and physical row order do not affect the logical source-set digest.
const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_visible_source_digest(
    .param .u64 payload_ptr,
    .param .u64 descriptors_ptr,
    .param .u32 column_count,
    .param .u64 row_count,
    .param .u64 row_ids_ptr,
    .param .u64 deleted_ptr,
    .param .u64 created_ptr,
    .param .s64 read_txn_id,
    .param .u64 out_ptr
)
{
    .shared .align 8 .b64 s_part[1280];
    .reg .pred %p<16>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<80>;

    ld.param.u64 %rd1, [payload_ptr];
    ld.param.u64 %rd2, [descriptors_ptr];
    ld.param.u32 %r1, [column_count];
    ld.param.u64 %rd3, [row_count];
    ld.param.u64 %rd4, [row_ids_ptr];
    ld.param.u64 %rd5, [deleted_ptr];
    ld.param.u64 %rd6, [created_ptr];
    ld.param.s64 %rd7, [read_txn_id];
    ld.param.u64 %rd8, [out_ptr];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %nctaid.x;
    mad.lo.u32 %r6, %r4, %r3, %r2;
    cvt.u64.u32 %rd9, %r6;
    mul.lo.u32 %r7, %r5, %r3;
    cvt.u64.u32 %rd10, %r7;

    mov.u64 %rd11, 0; // local visible count
    mov.u64 %rd12, 0; // accumulated lane 0
    mov.u64 %rd13, 0;
    mov.u64 %rd14, 0;
    mov.u64 %rd15, 0;

ROW_LOOP:
    setp.ge.u64 %p1, %rd9, %rd3;
    @%p1 bra ROWS_DONE;
    mul.lo.u64 %rd16, %rd9, 8;

    setp.eq.u64 %p2, %rd5, 0;
    @%p2 bra CREATED_CHECK;
    add.u64 %rd17, %rd5, %rd16;
    ld.global.s64 %rd18, [%rd17];
    setp.gt.s64 %p3, %rd18, %rd7;
    @!%p3 bra ROW_NEXT;
CREATED_CHECK:
    setp.eq.u64 %p2, %rd6, 0;
    @%p2 bra ROW_HASH_INIT;
    add.u64 %rd17, %rd6, %rd16;
    ld.global.s64 %rd18, [%rd17];
    setp.le.s64 %p3, %rd18, %rd7;
    @!%p3 bra ROW_NEXT;

ROW_HASH_INIT:
    add.u64 %rd17, %rd4, %rd16;
    ld.global.u64 %rd19, [%rd17];
    mov.u64 %rd20, 14695981039346656037;
    mov.u64 %rd21, 7809847782465536322;
    mov.u64 %rd22, 9650029242287828579;
    mov.u64 %rd23, 2870177450012600261;
    xor.b64 %rd20, %rd20, %rd19;
    mul.lo.u64 %rd20, %rd20, 1099511628211;
    xor.b64 %rd21, %rd21, %rd19;
    mul.lo.u64 %rd21, %rd21, 14029467366897019727;
    xor.b64 %rd22, %rd22, %rd19;
    mul.lo.u64 %rd22, %rd22, 1609587929392839161;
    xor.b64 %rd23, %rd23, %rd19;
    mul.lo.u64 %rd23, %rd23, 9650029242287828579;
    mov.u32 %r8, 0;

COLUMN_LOOP:
    setp.ge.u32 %p4, %r8, %r1;
    @%p4 bra ROW_HASH_DONE;
    mul.wide.u32 %rd24, %r8, 48;
    add.u64 %rd25, %rd2, %rd24;
    ld.global.u64 %rd26, [%rd25+0];  // kind
    ld.global.u64 %rd27, [%rd25+8];  // data offset
    ld.global.u64 %rd28, [%rd25+16]; // fixed width words
    ld.global.u64 %rd29, [%rd25+24]; // text blob offset
    ld.global.u64 %rd30, [%rd25+32]; // text blob length
    ld.global.u64 %rd31, [%rd25+40]; // validity offset / UINT64_MAX

    cvt.u64.u32 %rd32, %r8;
    shl.b64 %rd32, %rd32, 8;
    or.b64 %rd32, %rd32, %rd26;      // column boundary + kind marker
    xor.b64 %rd20, %rd20, %rd32;
    mul.lo.u64 %rd20, %rd20, 1099511628211;
    xor.b64 %rd21, %rd21, %rd32;
    mul.lo.u64 %rd21, %rd21, 14029467366897019727;
    xor.b64 %rd22, %rd22, %rd32;
    mul.lo.u64 %rd22, %rd22, 1609587929392839161;
    xor.b64 %rd23, %rd23, %rd32;
    mul.lo.u64 %rd23, %rd23, 9650029242287828579;

    setp.eq.u64 %p5, %rd31, 18446744073709551615;
    @%p5 bra VALUE_PRESENT;
    shr.u64 %rd33, %rd9, 3;
    and.b64 %rd34, %rd9, 7;
    add.u64 %rd35, %rd1, %rd31;
    add.u64 %rd35, %rd35, %rd33;
    ld.global.u8 %r9, [%rd35];
    cvt.u32.u64 %r10, %rd34;
    shr.u32 %r9, %r9, %r10;
    and.b32 %r9, %r9, 1;
    setp.ne.u32 %p6, %r9, 0;
    @%p6 bra VALUE_PRESENT;
    mov.u64 %rd36, 18446744073709551358; // canonical NULL marker
    xor.b64 %rd20, %rd20, %rd36;
    mul.lo.u64 %rd20, %rd20, 1099511628211;
    xor.b64 %rd21, %rd21, %rd36;
    mul.lo.u64 %rd21, %rd21, 14029467366897019727;
    xor.b64 %rd22, %rd22, %rd36;
    mul.lo.u64 %rd22, %rd22, 1609587929392839161;
    xor.b64 %rd23, %rd23, %rd36;
    mul.lo.u64 %rd23, %rd23, 9650029242287828579;
    bra COLUMN_NEXT;

VALUE_PRESENT:
    setp.eq.u64 %p7, %rd26, 1;
    @%p7 bra FIXED_VALUE;
    setp.eq.u64 %p7, %rd26, 2;
    @%p7 bra BOOL_VALUE;
    bra TEXT_VALUE;

FIXED_VALUE:
    mul.lo.u64 %rd33, %rd28, 4;
    mul.lo.u64 %rd34, %rd9, %rd33;
    add.u64 %rd35, %rd1, %rd27;
    add.u64 %rd35, %rd35, %rd34;
    mov.u64 %rd36, 0;
FIXED_WORD_LOOP:
    setp.ge.u64 %p8, %rd36, %rd28;
    @%p8 bra COLUMN_NEXT;
    mul.lo.u64 %rd37, %rd36, 4;
    add.u64 %rd38, %rd35, %rd37;
    ld.global.u32 %r11, [%rd38];
    cvt.u64.u32 %rd39, %r11;
    xor.b64 %rd20, %rd20, %rd39;
    mul.lo.u64 %rd20, %rd20, 1099511628211;
    xor.b64 %rd21, %rd21, %rd39;
    mul.lo.u64 %rd21, %rd21, 14029467366897019727;
    xor.b64 %rd22, %rd22, %rd39;
    mul.lo.u64 %rd22, %rd22, 1609587929392839161;
    xor.b64 %rd23, %rd23, %rd39;
    mul.lo.u64 %rd23, %rd23, 9650029242287828579;
    add.u64 %rd36, %rd36, 1;
    bra FIXED_WORD_LOOP;

BOOL_VALUE:
    shr.u64 %rd33, %rd9, 3;
    and.b64 %rd34, %rd9, 7;
    add.u64 %rd35, %rd1, %rd27;
    add.u64 %rd35, %rd35, %rd33;
    ld.global.u8 %r11, [%rd35];
    cvt.u32.u64 %r12, %rd34;
    shr.u32 %r11, %r11, %r12;
    and.b32 %r11, %r11, 1;
    cvt.u64.u32 %rd39, %r11;
    xor.b64 %rd20, %rd20, %rd39;
    mul.lo.u64 %rd20, %rd20, 1099511628211;
    xor.b64 %rd21, %rd21, %rd39;
    mul.lo.u64 %rd21, %rd21, 14029467366897019727;
    xor.b64 %rd22, %rd22, %rd39;
    mul.lo.u64 %rd22, %rd22, 1609587929392839161;
    xor.b64 %rd23, %rd23, %rd39;
    mul.lo.u64 %rd23, %rd23, 9650029242287828579;
    bra COLUMN_NEXT;

TEXT_VALUE:
    mul.lo.u64 %rd33, %rd9, 8;
    add.u64 %rd34, %rd1, %rd27;
    add.u64 %rd34, %rd34, %rd33;
    ld.global.u64 %rd35, [%rd34];
    ld.global.u64 %rd36, [%rd34+8];
    setp.gt.u64 %p9, %rd35, %rd36;
    @%p9 bra BAD_TEXT;
    setp.gt.u64 %p9, %rd36, %rd30;
    @%p9 bra BAD_TEXT;
    sub.u64 %rd37, %rd36, %rd35;
    xor.b64 %rd20, %rd20, %rd37;
    mul.lo.u64 %rd20, %rd20, 1099511628211;
    xor.b64 %rd21, %rd21, %rd37;
    mul.lo.u64 %rd21, %rd21, 14029467366897019727;
    xor.b64 %rd22, %rd22, %rd37;
    mul.lo.u64 %rd22, %rd22, 1609587929392839161;
    xor.b64 %rd23, %rd23, %rd37;
    mul.lo.u64 %rd23, %rd23, 9650029242287828579;
    add.u64 %rd38, %rd1, %rd29;
    add.u64 %rd38, %rd38, %rd35;
TEXT_BYTE_LOOP:
    setp.ge.u64 %p10, %rd35, %rd36;
    @%p10 bra COLUMN_NEXT;
    ld.global.u8 %r11, [%rd38];
    cvt.u64.u32 %rd39, %r11;
    xor.b64 %rd20, %rd20, %rd39;
    mul.lo.u64 %rd20, %rd20, 1099511628211;
    xor.b64 %rd21, %rd21, %rd39;
    mul.lo.u64 %rd21, %rd21, 14029467366897019727;
    xor.b64 %rd22, %rd22, %rd39;
    mul.lo.u64 %rd22, %rd22, 1609587929392839161;
    xor.b64 %rd23, %rd23, %rd39;
    mul.lo.u64 %rd23, %rd23, 9650029242287828579;
    add.u64 %rd35, %rd35, 1;
    add.u64 %rd38, %rd38, 1;
    bra TEXT_BYTE_LOOP;

BAD_TEXT:
    mov.u32 %r13, 1;
    atom.global.or.b32 %r14, [%rd8+40], %r13;
    bra ROW_NEXT;

COLUMN_NEXT:
    add.u32 %r8, %r8, 1;
    bra COLUMN_LOOP;

ROW_HASH_DONE:
    add.u64 %rd11, %rd11, 1;
    add.u64 %rd12, %rd12, %rd20;
    add.u64 %rd13, %rd13, %rd21;
    add.u64 %rd14, %rd14, %rd22;
    add.u64 %rd15, %rd15, %rd23;

ROW_NEXT:
    add.u64 %rd9, %rd9, %rd10;
    bra ROW_LOOP;

ROWS_DONE:
    mov.u64 %rd40, s_part;
    mul.wide.u32 %rd41, %r2, 8;
    add.u64 %rd42, %rd40, %rd41;
    st.shared.u64 [%rd42+0], %rd11;
    st.shared.u64 [%rd42+2048], %rd12;
    st.shared.u64 [%rd42+4096], %rd13;
    st.shared.u64 [%rd42+6144], %rd14;
    st.shared.u64 [%rd42+8192], %rd15;
    bar.sync 0;

    shr.u32 %r15, %r3, 1;
REDUCE_LOOP:
    setp.eq.u32 %p11, %r15, 0;
    @%p11 bra REDUCE_DONE;
    setp.lt.u32 %p12, %r2, %r15;
    @!%p12 bra REDUCE_SKIP;
    add.u32 %r16, %r2, %r15;
    mul.wide.u32 %rd43, %r16, 8;
    add.u64 %rd44, %rd40, %rd43;
    ld.shared.u64 %rd45, [%rd42+0];
    ld.shared.u64 %rd46, [%rd44+0];
    add.u64 %rd45, %rd45, %rd46;
    st.shared.u64 [%rd42+0], %rd45;
    ld.shared.u64 %rd45, [%rd42+2048];
    ld.shared.u64 %rd46, [%rd44+2048];
    add.u64 %rd45, %rd45, %rd46;
    st.shared.u64 [%rd42+2048], %rd45;
    ld.shared.u64 %rd45, [%rd42+4096];
    ld.shared.u64 %rd46, [%rd44+4096];
    add.u64 %rd45, %rd45, %rd46;
    st.shared.u64 [%rd42+4096], %rd45;
    ld.shared.u64 %rd45, [%rd42+6144];
    ld.shared.u64 %rd46, [%rd44+6144];
    add.u64 %rd45, %rd45, %rd46;
    st.shared.u64 [%rd42+6144], %rd45;
    ld.shared.u64 %rd45, [%rd42+8192];
    ld.shared.u64 %rd46, [%rd44+8192];
    add.u64 %rd45, %rd45, %rd46;
    st.shared.u64 [%rd42+8192], %rd45;
REDUCE_SKIP:
    bar.sync 0;
    shr.u32 %r15, %r15, 1;
    bra REDUCE_LOOP;

REDUCE_DONE:
    setp.ne.u32 %p13, %r2, 0;
    @%p13 bra DONE;
    ld.shared.u64 %rd45, [%rd40+0];
    red.global.add.u64 [%rd8+0], %rd45;
    ld.shared.u64 %rd45, [%rd40+2048];
    red.global.add.u64 [%rd8+8], %rd45;
    ld.shared.u64 %rd45, [%rd40+4096];
    red.global.add.u64 [%rd8+16], %rd45;
    ld.shared.u64 %rd45, [%rd40+6144];
    red.global.add.u64 [%rd8+24], %rd45;
    ld.shared.u64 %rd45, [%rd40+8192];
    red.global.add.u64 [%rd8+32], %rd45;
DONE:
    ret;
}
"#;
