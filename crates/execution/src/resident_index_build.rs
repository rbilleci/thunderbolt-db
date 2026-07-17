use std::os::raw::c_void;

use crate::{
    check_cuda, CudaCompoundFoldColumn, CudaResidentDeviceMemory, CudaResidentReadSource,
    CudaRuntimeProbeError,
};

// R3-002: construct the resident unique-key hash directly from resident typed columns. A thread
// derives its raw/fingerprint key, skips a version dead at/below the GC boundary, then claims an
// open-addressing slot with atomicCAS. The layout and 256-probe bound are byte-identical to the
// read/write locate kernels. Descriptor arrays and the four-byte verdict are control-plane data;
// keys, fingerprints, deleted stamps, and the hash table never cross the host.
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
    .param .u32 row_count,
    .param .u64 index_ptr,
    .param .u32 table_mask,
    .param .u32 hash_shift,
    .param .u64 deleted_ptr,
    .param .u64 gc_boundary,
    .param .u32 dup_tolerant,
    .param .u64 decline_ptr
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<56>;

    ld.param.u64 %rd1, [base_ptr];
    ld.param.u64 %rd2, [offsets_ptr];
    ld.param.u64 %rd3, [widths_ptr];
    ld.param.u64 %rd20, [blob_offsets_ptr];
    ld.param.u64 %rd31, [blob_lens_ptr];
    ld.param.u32 %r1, [ncols];
    ld.param.u32 %r2, [row_count];
    ld.param.u64 %rd4, [index_ptr];
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

    // A missing deleted_by region means all-live. Otherwise omit only rows dead no later than the
    // oldest active snapshot; newer tombstones remain indexed for old pinned readers.
    setp.eq.u64 %p10, %rd40, 0;
    @%p10 bra KEYMODE;
    mul.wide.u32 %rd43, %r6, 8;
    add.u64 %rd44, %rd40, %rd43;
    ld.global.u64 %rd45, [%rd44];
    setp.le.u64 %p11, %rd45, %rd41;
    @%p11 bra DONE;

KEYMODE:
    // The unflagged single i32/date/int2 ABI stores the resident word verbatim. Every other shape
    // (multi-column, wide fixed, BOOL, TEXT) uses the canonical FNV/rotate fingerprint below.
    setp.ne.u32 %p12, %r1, 1;
    @%p12 bra FOLDINIT;
    ld.global.u32 %r33, [%rd3];
    setp.ne.u32 %p12, %r33, 1;
    @%p12 bra FOLDINIT;
    ld.global.u64 %rd7, [%rd2];
    mul.wide.u32 %rd10, %r6, 4;
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
    setp.eq.u32 %p4, %r9, 0;
    @%p4 bra TEXTCOL;
    setp.eq.u32 %p4, %r9, 4294967295;
    @%p4 bra BOOLCOL;
    mul.lo.u32 %r10, %r6, %r9;
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
    shr.u32 %r19, %r6, 3;
    and.b32 %r20, %r6, 7;
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
    mul.wide.u32 %rd22, %r6, 8;
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
    add.u32 %r34, %r6, 1;
    cvt.u64.u32 %rd46, %r34;
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
    setp.ne.u32 %p13, %r32, 0;
    @%p13 bra ADVANCE;
    shr.u64 %rd52, %rd51, 32;
    cvt.u32.u64 %r38, %rd52;
    setp.eq.u32 %p14, %r38, %r7;
    @%p14 bra DECLINE;

ADVANCE:
    add.u32 %r36, %r36, 1;
    and.b32 %r36, %r36, %r30;
    add.u32 %r37, %r37, 1;
    setp.ge.u32 %p7, %r37, 256;
    @%p7 bra DECLINE;
    bra PROBE;

DECLINE:
    mov.u32 %r39, 1;
    atom.global.exch.b32 %r40, [%rd42], %r39;

DONE:
    ret;
}
"#;

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

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(row_count))?;
        if row_count == 0 || columns.is_empty() || self.context() != index.context() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(row_count));
        }
        let table_size = u64::from(table_mask) + 1;
        let expected_shift = 32_u32.checked_sub(table_size.trailing_zeros()).ok_or(
            CudaRuntimeProbeError::InvalidInputLength(table_size as usize),
        )?;
        let index_bytes = table_size
            .checked_mul(std::mem::size_of::<u64>() as u64)
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
            let needed = (row_count as u64)
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
                    let bytes = (row_count as u64)
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
                    let bytes = row_count.div_ceil(8) as u64;
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
                CudaCompoundFoldColumn::Text {
                    offsets_byte_offset,
                    bytes_byte_offset,
                    bytes_len,
                } => {
                    let offsets_end = offsets_byte_offset
                        .checked_add((row_count as u64 + 1).saturating_mul(8))
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
        let mut rows_arg = row_count_u32;
        let mut index_arg = index.device_ptr();
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
            (&mut rows_arg as *mut u32).cast::<c_void>(),
            (&mut index_arg as *mut u64).cast::<c_void>(),
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
        Ok(decline != 0)
    }
}
