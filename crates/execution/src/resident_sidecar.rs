use crate::{CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError, check_cuda};
use std::{ffi::c_void, sync::Arc};

#[derive(Debug, Clone)]
pub struct CudaSidecarSource {
    pub memory: Arc<CudaResidentDeviceMemory>,
    pub byte_offset: u64,
}

#[derive(Debug, Clone)]
pub struct CudaTextOffsetSource {
    pub memory: Arc<CudaResidentDeviceMemory>,
    pub offsets_byte_offset: u64,
    pub bytes_byte_offset: u64,
    pub bytes_len: u64,
}

fn checked_window(
    memory: &CudaResidentDeviceMemory,
    byte_offset: u64,
    byte_len: u64,
    alignment: u64,
) -> Result<u64, CudaRuntimeProbeError> {
    if memory.device_ptr() == 0 || alignment == 0 || !byte_offset.is_multiple_of(alignment) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            byte_offset as usize,
        ));
    }
    let end = byte_offset
        .checked_add(byte_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if end > memory.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(end).unwrap_or(usize::MAX),
        ));
    }
    memory
        .device_ptr()
        .checked_add(byte_offset)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))
}

fn bitmap_window_bytes(base_row: u32, count: u32) -> Result<u64, CudaRuntimeProbeError> {
    let end_row = base_row
        .checked_add(count)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count as usize))?;
    u64::from(end_row)
        .checked_add(31)
        .and_then(|bits| bits.checked_div(32))
        .and_then(|words| words.checked_mul(std::mem::size_of::<u32>() as u64))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count as usize))
}

fn entry_window_bytes(base: u32, count: u32, width: u64) -> Result<u64, CudaRuntimeProbeError> {
    let entries = base
        .checked_add(count)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count as usize))?;
    u64::from(entries)
        .checked_mul(width)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count as usize))
}

fn checked_source(
    primary: &Arc<crate::GpuPrimaryContext>,
    source: &CudaSidecarSource,
    byte_len: u64,
    alignment: u64,
) -> Result<u64, CudaRuntimeProbeError> {
    if !Arc::ptr_eq(primary, &source.memory.primary_arc()) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    checked_window(&source.memory, source.byte_offset, byte_len, alignment)
}

fn ranges_overlap(left_offset: u64, left_len: u64, right_offset: u64, right_len: u64) -> bool {
    let left_end = left_offset + left_len;
    let right_end = right_offset + right_len;
    left_offset < right_end && right_offset < left_end
}

type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

struct ContextDrain {
    sync: CuCtxSynchronize,
    armed: bool,
}

impl Drop for ContextDrain {
    fn drop(&mut self) {
        if self.armed {
            unsafe { (self.sync)() };
        }
    }
}

/// U1 perf lever B: the SCATTER kernel — `region[slots[t]] = values[t]` (u64 store), one thread
/// per (slot, value) pair. Replaces N per-slot HtoD chunks with 2 HtoDs + 1 launch on the lane
/// tombstone path. Bounds are the caller's contract (slots pre-checked < capacity). ASCII-only.
const SCATTER_U64_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_scatter_u64(
    .param .u64 region_ptr,
    .param .u32 count,
    .param .u64 slots_ptr,
    .param .u64 values_ptr
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [region_ptr];
    ld.param.u32 %r1, [count];
    ld.param.u64 %rd2, [slots_ptr];
    ld.param.u64 %rd3, [values_ptr];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    // value = values[t]
    mul.wide.u32 %rd4, %r5, 8;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.u64 %rd6, [%rd5];
    // slot = slots[t]  (u32)
    mul.wide.u32 %rd7, %r5, 4;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.u32 %r2, [%rd8];
    // region[slot] = value  (slot * 8 bytes)
    mul.wide.u32 %rd9, %r2, 8;
    add.u64 %rd9, %rd1, %rd9;
    st.global.u64 [%rd9], %rd6;

DONE:
    ret;
}
"#;

// TYPE-COVERAGE #14 (bool): set the value bits of `count` appended rows into a resident bool column's
// 1-bit/row bitmap section (LE u32 words, LSB-first). Thread t owns appended row `base_row + t`; it reads
// its value byte (0/1) and atomically sets OR clears the bit in word `(base_row+t)>>5`. Both logical states
// are explicit, so correctness does not depend on destination prefill. The atomics preserve unrelated bits
// when several appended rows land in the SAME 32-bit word.
const BOOL_BITMAP_SET_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_bool_bitmap_set_range(
    .param .u64 region_ptr,
    .param .u64 bitmap_byte_offset,
    .param .u32 base_row,
    .param .u32 count,
    .param .u64 values_ptr
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<13>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [region_ptr];
    ld.param.u64 %rd2, [bitmap_byte_offset];
    ld.param.u32 %r1, [base_row];
    ld.param.u32 %r2, [count];
    ld.param.u64 %rd3, [values_ptr];

    // t = ctaid.x * ntid.x + tid.x
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;

    // val = values[t]  (u8, zero-extended)
    cvt.u64.u32 %rd4, %r6;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.u8 %r7, [%rd5];
    setp.eq.u32 %p2, %r7, 0;

    // row = base_row + t ; word = row >> 5 ; bit = row & 31 ; mask = 1 << bit
    add.u32 %r8, %r1, %r6;
    shr.u32 %r9, %r8, 5;
    and.b32 %r10, %r8, 31;
    mov.u32 %r11, 1;
    shl.b32 %r11, %r11, %r10;

    // addr = region_ptr + bitmap_byte_offset + word*4
    mul.wide.u32 %rd6, %r9, 4;
    add.u64 %rd7, %rd1, %rd2;
    add.u64 %rd7, %rd7, %rd6;
    @%p2 bra CLEAR;
    atom.global.or.b32 %r12, [%rd7], %r11;
    bra DONE;

CLEAR:
    not.b32 %r11, %r11;
    atom.global.and.b32 %r12, [%rd7], %r11;

DONE:
    ret;
}
"#;

// TYPE-COVERAGE #14 (bool): recompact ONE shard's bool bitmap into the unified buffer's bool bitmap at an
// ARBITRARY (not necessarily 32-row-aligned) destination base. Thread `l` owns the shard's local row `l`;
// it reads source bit `l` and atomically sets OR clears destination bit `dst_base_row + l`. A byte-copy cannot
// do this when `dst_base_row % 32 != 0` (shards seal at arbitrary row counts), so this per-bit gather is
// the alignment-free repack. Both states are explicit and disjoint-bit atomics commute when rows from
// different shards land in the SAME destination word.
const BOOL_BITMAP_GATHER_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_bool_bitmap_gather_shard(
    .param .u64 dst_ptr,
    .param .u64 dst_bitmap_offset,
    .param .u32 dst_base_row,
    .param .u64 src_ptr,
    .param .u64 src_bitmap_offset,
    .param .u32 count
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [dst_ptr];
    ld.param.u64 %rd2, [dst_bitmap_offset];
    ld.param.u32 %r1, [dst_base_row];
    ld.param.u64 %rd3, [src_ptr];
    ld.param.u64 %rd4, [src_bitmap_offset];
    ld.param.u32 %r2, [count];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;

    // src bit l: word = src[src_off + (l>>5)*4]; bit = (word >> (l&31)) & 1
    shr.u32 %r7, %r6, 5;
    and.b32 %r8, %r6, 31;
    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd6, %rd3, %rd4;
    add.u64 %rd6, %rd6, %rd5;
    ld.global.u32 %r9, [%rd6];
    shr.u32 %r10, %r9, %r8;
    and.b32 %r10, %r10, 1;
    setp.eq.u32 %p2, %r10, 0;

    // dst bit (dst_base_row + l): word = (base+l)>>5 ; bit = (base+l)&31 ; mask = 1<<bit
    add.u32 %r11, %r1, %r6;
    shr.u32 %r12, %r11, 5;
    and.b32 %r13, %r11, 31;
    mov.u32 %r14, 1;
    shl.b32 %r14, %r14, %r13;
    mul.wide.u32 %rd7, %r12, 4;
    add.u64 %rd8, %rd1, %rd2;
    add.u64 %rd8, %rd8, %rd7;
    @%p2 bra CLEAR;
    atom.global.or.b32 %r15, [%rd8], %r14;
    bra DONE;

CLEAR:
    not.b32 %r14, %r14;
    atom.global.and.b32 %r15, [%rd8], %r14;

DONE:
    ret;
}
"#;

// ADR-006 (NULL coverage): recompact ONE shard's VALIDITY bitmap into the unified buffer at an ARBITRARY
// (not necessarily 32-row-aligned) destination base — the alignment-free twin of the bool gather. Validity
// semantics are INVERTED vs bool: 1 = valid, 0 = NULL, and a row/shard WITHOUT a bitmap is all-valid. Each
// source row explicitly sets or clears its destination validity bit; correctness does not depend on the
// initial destination word, and disjoint-bit atomics commute across shard boundaries.
const NULL_BITMAP_GATHER_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_null_bitmap_gather_shard(
    .param .u64 dst_ptr,
    .param .u64 dst_bitmap_offset,
    .param .u32 dst_base_row,
    .param .u64 src_ptr,
    .param .u64 src_bitmap_offset,
    .param .u32 count
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [dst_ptr];
    ld.param.u64 %rd2, [dst_bitmap_offset];
    ld.param.u32 %r1, [dst_base_row];
    ld.param.u64 %rd3, [src_ptr];
    ld.param.u64 %rd4, [src_bitmap_offset];
    ld.param.u32 %r2, [count];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;

    // src bit l: word = src[src_off + (l>>5)*4]; bit = (word >> (l&31)) & 1
    shr.u32 %r7, %r6, 5;
    and.b32 %r8, %r6, 31;
    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd6, %rd3, %rd4;
    add.u64 %rd6, %rd6, %rd5;
    ld.global.u32 %r9, [%rd6];
    shr.u32 %r10, %r9, %r8;
    and.b32 %r10, %r10, 1;
    setp.eq.u32 %p2, %r10, 0;

    // dst bit (dst_base_row + l): word = (base+l)>>5 ; bit = (base+l)&31 ; clear mask = ~(1<<bit)
    add.u32 %r11, %r1, %r6;
    shr.u32 %r12, %r11, 5;
    and.b32 %r13, %r11, 31;
    mov.u32 %r14, 1;
    shl.b32 %r14, %r14, %r13;
    mul.wide.u32 %rd7, %r12, 4;
    add.u64 %rd8, %rd1, %rd2;
    add.u64 %rd8, %rd8, %rd7;
    @%p2 bra CLEAR;
    atom.global.or.b32 %r15, [%rd8], %r14;
    bra DONE;

CLEAR:
    not.b32 %r14, %r14;
    atom.global.and.b32 %r15, [%rd8], %r14;

DONE:
    ret;
}
"#;

// TYPE-COVERAGE #14 (text): rebase ONE source shard's text offsets into the unified buffer's offsets
// section. A shard's offsets are RELATIVE to its own bytes blob (start 0); the unified buffer concatenates
// blobs, so each shard's offsets must have that shard's running `blob_base` added. Thread `i` reads source
// offset `i` (u64) and writes `src_off + blob_base` to unified offset `dst_base_row + i`. `count` =
// row_count+1 (the offsets are one-longer than the rows). DtoD, no HtoD; the unified offsets section is
// fully written across shards (contiguous, boundaries overlap with the identical value).
const TEXT_OFFSET_REBASE_PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_text_offset_rebase(
    .param .u64 dst_ptr,
    .param .u64 dst_offsets_byte_offset,
    .param .u32 dst_base_row,
    .param .u64 blob_base,
    .param .u64 src_ptr,
    .param .u64 src_offsets_byte_offset,
    .param .u32 count,
    .param .u64 src_blob_len,
    .param .u64 error_ptr
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<18>;

    ld.param.u64 %rd1, [dst_ptr];
    ld.param.u64 %rd2, [dst_offsets_byte_offset];
    ld.param.u32 %r1, [dst_base_row];
    ld.param.u64 %rd3, [blob_base];
    ld.param.u64 %rd4, [src_ptr];
    ld.param.u64 %rd5, [src_offsets_byte_offset];
    ld.param.u32 %r2, [count];
    ld.param.u64 %rd13, [src_blob_len];
    ld.param.u64 %rd12, [error_ptr];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;

    // src_off = *(u64*)(src_ptr + src_offsets_byte_offset + i*8)
    mul.wide.u32 %rd6, %r6, 8;
    add.u64 %rd7, %rd4, %rd5;
    add.u64 %rd7, %rd7, %rd6;
    ld.global.u64 %rd8, [%rd7];
    // A text offset vector starts at zero, is monotonic, and never exceeds its blob.
    setp.gt.u64 %p2, %rd8, %rd13;
    @%p2 bra INVALID;
    setp.eq.u32 %p3, %r6, 0;
    @%p3 bra CHECK_ZERO;
    sub.u64 %rd15, %rd7, 8;
    ld.global.u64 %rd14, [%rd15];
    setp.lt.u64 %p4, %rd8, %rd14;
    @%p4 bra INVALID;
    bra CHECK_LAST;

CHECK_ZERO:
    setp.ne.u64 %p4, %rd8, 0;
    @%p4 bra INVALID;

CHECK_LAST:
    sub.u32 %r9, %r2, 1;
    setp.eq.u32 %p2, %r6, %r9;
    @!%p2 bra REBASE;
    setp.ne.u64 %p4, %rd8, %rd13;
    @%p4 bra INVALID;

REBASE:
    // dst_val = src_off + blob_base
    add.u64 %rd9, %rd8, %rd3;
    setp.lt.u64 %p4, %rd9, %rd8;
    @%p4 bra INVALID;
    // dst_idx = dst_base_row + i ; addr = dst_ptr + dst_offsets_byte_offset + dst_idx*8
    add.u32 %r7, %r1, %r6;
    mul.wide.u32 %rd10, %r7, 8;
    add.u64 %rd11, %rd1, %rd2;
    add.u64 %rd11, %rd11, %rd10;
    st.global.u64 [%rd11], %rd9;
    bra DONE;

INVALID:
    mov.u32 %r8, 1;
    atom.global.exch.b32 %r9, [%rd12], %r8;

DONE:
    ret;
}
"#;

impl CudaResidentDeviceMemory {
    /// U1 perf lever B: SCATTER `values[t]` into `deleted_by[slots[t]]` in ONE launch (a single
    /// `slots` HtoD + a single `values` HtoD + one kernel), replacing the per-slot
    /// `append_owned_chunks` HtoD loop the lane tombstone pass used (measured device-apply
    /// ~468us/wave at ~75 tombstones = N tiny HtoDs). `self` is the `deleted_by` region (u64
    /// array, slot `s` at byte `s*8`); this API derives and checks the highest written byte before
    /// launch. Synchronous: null-stream launch + `cuCtxSynchronize`, so the stamps are
    /// device-visible before return (the caller then publishes / settles). ASCII-only PTX.
    pub fn scatter_u64_slots(
        &self,
        slots: &[u32],
        values: &[u64],
    ) -> Result<(), CudaRuntimeProbeError> {
        type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
        type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
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

        if slots.len() != values.len() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(slots.len()));
        }
        if slots.is_empty() {
            return Ok(());
        }
        let max_slot = slots.iter().copied().max().unwrap_or(0);
        let required_bytes = entry_window_bytes(max_slot, 1, std::mem::size_of::<u64>() as u64)?;
        checked_window(self, 0, required_bytes, std::mem::align_of::<u64>() as u64)?;
        let count = u32::try_from(slots.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(slots.len()))?;
        let slots_bytes = std::mem::size_of_val(slots);
        let values_bytes = std::mem::size_of_val(values);

        let primary = self.primary_arc();
        primary.set_current()?;
        let cu_memcpy_htod = unsafe {
            primary
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_launch_kernel = unsafe {
            primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_ctx_synchronize = unsafe {
            primary
                .lib()
                .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };

        let slots_guard = primary.lease_device_buffer_owned(slots_bytes)?;
        let values_guard = primary.lease_device_buffer_owned(values_bytes)?;

        let mut ptx = Vec::with_capacity(SCATTER_U64_PTX.len() + 1);
        ptx.extend_from_slice(SCATTER_U64_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_resident_scatter_u64", &ptx)?;

        check_cuda(unsafe {
            cu_memcpy_htod(
                slots_guard.ptr,
                slots.as_ptr().cast::<c_void>(),
                slots_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_htod(
                values_guard.ptr,
                values.as_ptr().cast::<c_void>(),
                values_bytes,
            )
        })?;

        let mut region_arg = self.device_ptr();
        let mut count_arg = count;
        let mut slots_arg = slots_guard.ptr;
        let mut values_arg = values_guard.ptr;
        let mut args = [
            (&mut region_arg as *mut u64).cast::<c_void>(),
            (&mut count_arg as *mut u32).cast::<c_void>(),
            (&mut slots_arg as *mut u64).cast::<c_void>(),
            (&mut values_arg as *mut u64).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = count.div_ceil(threads_per_block);
        let mut context_drain = ContextDrain {
            sync: *cu_ctx_synchronize,
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
        // No DtoH output to fence the scatter; ctx-synchronize so the stamps are device-visible
        // before the caller publishes (matches the append_owned_chunks synchronous contract).
        check_cuda(unsafe { cu_ctx_synchronize() })?;
        context_drain.armed = false;
        drop(slots_guard);
        drop(values_guard);
        Ok(())
    }

    /// TYPE-COVERAGE #14 (bool): set the value bits of `values.len()` appended rows into THIS shard
    /// buffer's bool column bitmap at `bitmap_byte_offset` (the column's section start within the buffer),
    /// starting at local row `base_row`. `values[i]` is 0/1 for appended row `base_row + i`. The kernel
    /// atomically sets or clears every addressed bit while preserving neighboring rows — the bool analog
    /// of `scatter_u64_slots`. Synchronous (ctx-sync) so the bits are
    /// device-visible before the caller bumps `row_count` / publishes.
    pub fn set_bool_bitmap_range(
        &self,
        bitmap_byte_offset: u64,
        base_row: u32,
        values: &[u8],
    ) -> Result<(), CudaRuntimeProbeError> {
        type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
        type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
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

        if values.is_empty() {
            return Ok(());
        }
        if values.iter().any(|value| *value > 1) {
            return Err(CudaRuntimeProbeError::InvalidInputLength(values.len()));
        }
        let count = u32::try_from(values.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(values.len()))?;
        let bitmap_bytes = bitmap_window_bytes(base_row, count)?;
        checked_window(
            self,
            bitmap_byte_offset,
            bitmap_bytes,
            std::mem::align_of::<u32>() as u64,
        )?;
        let values_bytes = std::mem::size_of_val(values);

        let primary = self.primary_arc();
        primary.set_current()?;
        let cu_memcpy_htod = unsafe {
            primary
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_launch_kernel = unsafe {
            primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_ctx_synchronize = unsafe {
            primary
                .lib()
                .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };

        let values_guard = primary.lease_device_buffer_owned(values_bytes)?;

        let mut ptx = Vec::with_capacity(BOOL_BITMAP_SET_PTX.len() + 1);
        ptx.extend_from_slice(BOOL_BITMAP_SET_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_resident_bool_bitmap_set_range", &ptx)?;

        check_cuda(unsafe {
            cu_memcpy_htod(
                values_guard.ptr,
                values.as_ptr().cast::<c_void>(),
                values_bytes,
            )
        })?;

        let mut region_arg = self.device_ptr();
        let mut offset_arg = bitmap_byte_offset;
        let mut base_row_arg = base_row;
        let mut count_arg = count;
        let mut values_arg = values_guard.ptr;
        let mut args = [
            (&mut region_arg as *mut u64).cast::<c_void>(),
            (&mut offset_arg as *mut u64).cast::<c_void>(),
            (&mut base_row_arg as *mut u32).cast::<c_void>(),
            (&mut count_arg as *mut u32).cast::<c_void>(),
            (&mut values_arg as *mut u64).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = count.div_ceil(threads_per_block);
        let mut context_drain = ContextDrain {
            sync: *cu_ctx_synchronize,
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
        check_cuda(unsafe { cu_ctx_synchronize() })?;
        context_drain.armed = false;
        drop(values_guard);
        Ok(())
    }

    /// TYPE-COVERAGE #14 (bool): gather ONE source shard's bool bitmap (`count` local rows at
    /// `src_bitmap_offset` in `src_device_ptr`) into THIS (unified) buffer's bool bitmap at
    /// `dst_bitmap_offset`, placing the shard's row `l` at unified row `dst_base_row + l`. Device->device
    /// (no HtoD): the kernel atomically sets or clears every destination bit, so it works at ANY
    /// `dst_base_row` and with any destination prefill. Synchronous (ctx-sync).
    pub fn gather_bool_bitmap_from_shard(
        &self,
        dst_bitmap_offset: u64,
        dst_base_row: u32,
        source: &CudaSidecarSource,
        count: u32,
    ) -> Result<(), CudaRuntimeProbeError> {
        type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
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

        if count == 0 {
            return Ok(());
        }
        let primary = self.primary_arc();
        let dst_bytes = bitmap_window_bytes(dst_base_row, count)?;
        checked_window(
            self,
            dst_bitmap_offset,
            dst_bytes,
            std::mem::align_of::<u32>() as u64,
        )?;
        let src_bytes = bitmap_window_bytes(0, count)?;
        if self.device_ptr() == source.memory.device_ptr() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let src_device_ptr = checked_source(
            &primary,
            source,
            src_bytes,
            std::mem::align_of::<u32>() as u64,
        )?;
        primary.set_current()?;
        let cu_launch_kernel = unsafe {
            primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_ctx_synchronize = unsafe {
            primary
                .lib()
                .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };

        let mut ptx = Vec::with_capacity(BOOL_BITMAP_GATHER_PTX.len() + 1);
        ptx.extend_from_slice(BOOL_BITMAP_GATHER_PTX);
        ptx.push(0);
        let function =
            primary.cached_function(c"gpu_db_resident_bool_bitmap_gather_shard", &ptx)?;

        let mut dst_ptr_arg = self.device_ptr();
        let mut dst_off_arg = dst_bitmap_offset;
        let mut dst_base_arg = dst_base_row;
        let mut src_ptr_arg = src_device_ptr;
        let mut src_off_arg = 0_u64;
        let mut count_arg = count;
        let mut args = [
            (&mut dst_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut dst_off_arg as *mut u64).cast::<c_void>(),
            (&mut dst_base_arg as *mut u32).cast::<c_void>(),
            (&mut src_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut src_off_arg as *mut u64).cast::<c_void>(),
            (&mut count_arg as *mut u32).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = count.div_ceil(threads_per_block);
        let mut context_drain = ContextDrain {
            sync: *cu_ctx_synchronize,
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
        check_cuda(unsafe { cu_ctx_synchronize() })?;
        context_drain.armed = false;
        Ok(())
    }

    /// ADR-006 (NULL coverage): gather ONE source shard's VALIDITY bitmap (`count` local rows at
    /// `src_bitmap_offset` in `src_device_ptr`) into THIS (unified) buffer's validity bitmap at
    /// `dst_bitmap_offset`, placing the shard's row `l` at unified row `dst_base_row + l`. The unified region
    /// kernel atomically sets each valid bit and clears each NULL bit — the alignment-free twin of
    /// `gather_bool_bitmap_from_shard`, working at ANY `dst_base_row` and with any destination prefill.
    /// Synchronous (ctx-sync).
    pub fn gather_null_bitmap_from_shard(
        &self,
        dst_bitmap_offset: u64,
        dst_base_row: u32,
        source: &CudaSidecarSource,
        count: u32,
    ) -> Result<(), CudaRuntimeProbeError> {
        type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
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

        if count == 0 {
            return Ok(());
        }
        let primary = self.primary_arc();
        let dst_bytes = bitmap_window_bytes(dst_base_row, count)?;
        checked_window(
            self,
            dst_bitmap_offset,
            dst_bytes,
            std::mem::align_of::<u32>() as u64,
        )?;
        let src_bytes = bitmap_window_bytes(0, count)?;
        if self.device_ptr() == source.memory.device_ptr() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let src_device_ptr = checked_source(
            &primary,
            source,
            src_bytes,
            std::mem::align_of::<u32>() as u64,
        )?;
        primary.set_current()?;
        let cu_launch_kernel = unsafe {
            primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let cu_ctx_synchronize = unsafe {
            primary
                .lib()
                .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };

        let mut ptx = Vec::with_capacity(NULL_BITMAP_GATHER_PTX.len() + 1);
        ptx.extend_from_slice(NULL_BITMAP_GATHER_PTX);
        ptx.push(0);
        let function =
            primary.cached_function(c"gpu_db_resident_null_bitmap_gather_shard", &ptx)?;

        let mut dst_ptr_arg = self.device_ptr();
        let mut dst_off_arg = dst_bitmap_offset;
        let mut dst_base_arg = dst_base_row;
        let mut src_ptr_arg = src_device_ptr;
        let mut src_off_arg = 0_u64;
        let mut count_arg = count;
        let mut args = [
            (&mut dst_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut dst_off_arg as *mut u64).cast::<c_void>(),
            (&mut dst_base_arg as *mut u32).cast::<c_void>(),
            (&mut src_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut src_off_arg as *mut u64).cast::<c_void>(),
            (&mut count_arg as *mut u32).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = count.div_ceil(threads_per_block);
        let mut context_drain = ContextDrain {
            sync: *cu_ctx_synchronize,
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
        check_cuda(unsafe { cu_ctx_synchronize() })?;
        context_drain.armed = false;
        Ok(())
    }

    /// TYPE-COVERAGE #14 (text): rebase ONE source shard's `count` (= row_count+1) text offsets into THIS
    /// (unified) buffer's offsets section at `dst_offsets_byte_offset`, placing the shard's offset `i` at
    /// unified offset `dst_base_row + i` with `blob_base` added (the shard's running byte position in the
    /// concatenated unified blob). The typed source owns the allocation and blob extent; the kernel rejects
    /// a nonzero first offset, descending offsets, offsets beyond the blob, and rebase overflow before the
    /// unified snapshot can be published. Device->device (no HtoD). Synchronous (blocking error readback).
    pub fn rebase_text_offsets_from_shard(
        &self,
        dst_offsets_byte_offset: u64,
        dst_base_row: u32,
        blob_base: u64,
        source: &CudaTextOffsetSource,
        count: u32,
    ) -> Result<(), CudaRuntimeProbeError> {
        type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
        type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
        type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
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

        if count == 0 {
            return Ok(());
        }
        let primary = self.primary_arc();
        let dst_bytes = entry_window_bytes(dst_base_row, count, std::mem::size_of::<u64>() as u64)?;
        checked_window(
            self,
            dst_offsets_byte_offset,
            dst_bytes,
            std::mem::align_of::<u64>() as u64,
        )?;
        let src_bytes = entry_window_bytes(0, count, std::mem::size_of::<u64>() as u64)?;
        if self.device_ptr() == source.memory.device_ptr() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        if !Arc::ptr_eq(&primary, &source.memory.primary_arc()) {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let src_device_ptr = checked_window(
            &source.memory,
            source.offsets_byte_offset,
            src_bytes,
            std::mem::align_of::<u64>() as u64,
        )?;
        checked_window(
            &source.memory,
            source.bytes_byte_offset,
            source.bytes_len,
            1,
        )?;
        if ranges_overlap(
            source.offsets_byte_offset,
            src_bytes,
            source.bytes_byte_offset,
            source.bytes_len,
        ) {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        primary.set_current()?;
        let cu_memset_d8 = unsafe {
            primary
                .lib()
                .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
                .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
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
        let cu_ctx_synchronize = unsafe {
            primary
                .lib()
                .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };

        let mut ptx = Vec::with_capacity(TEXT_OFFSET_REBASE_PTX.len() + 1);
        ptx.extend_from_slice(TEXT_OFFSET_REBASE_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_resident_text_offset_rebase", &ptx)?;
        let error_guard = primary.lease_device_buffer_owned(std::mem::size_of::<u32>())?;
        check_cuda(unsafe { cu_memset_d8(error_guard.ptr, 0, std::mem::size_of::<u32>()) })?;

        let mut dst_ptr_arg = self.device_ptr();
        let mut dst_off_arg = dst_offsets_byte_offset;
        let mut dst_base_arg = dst_base_row;
        let mut blob_base_arg = blob_base;
        let mut src_ptr_arg = src_device_ptr;
        let mut src_off_arg = 0_u64;
        let mut count_arg = count;
        let mut src_blob_len_arg = source.bytes_len;
        let mut error_arg = error_guard.ptr;
        let mut args = [
            (&mut dst_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut dst_off_arg as *mut u64).cast::<c_void>(),
            (&mut dst_base_arg as *mut u32).cast::<c_void>(),
            (&mut blob_base_arg as *mut u64).cast::<c_void>(),
            (&mut src_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut src_off_arg as *mut u64).cast::<c_void>(),
            (&mut count_arg as *mut u32).cast::<c_void>(),
            (&mut src_blob_len_arg as *mut u64).cast::<c_void>(),
            (&mut error_arg as *mut u64).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = count.div_ceil(threads_per_block);
        let mut context_drain = ContextDrain {
            sync: *cu_ctx_synchronize,
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
        let mut input_error = 0_u32;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                (&mut input_error as *mut u32).cast::<c_void>(),
                error_guard.ptr,
                std::mem::size_of::<u32>(),
            )
        })?;
        if input_error != 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        context_drain.armed = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{bitmap_window_bytes, entry_window_bytes, ranges_overlap};

    #[test]
    fn sidecar_entry_windows_are_boundary_exact_and_overflow_safe() {
        assert_eq!(entry_window_bytes(3, 2, 8).unwrap(), 40);
        assert_eq!(entry_window_bytes(0, 0, 8).unwrap(), 0);
        assert!(entry_window_bytes(u32::MAX, 1, 8).is_err());
    }

    #[test]
    fn sidecar_bitmap_windows_round_words_and_reject_row_overflow() {
        assert_eq!(bitmap_window_bytes(0, 1).unwrap(), 4);
        assert_eq!(bitmap_window_bytes(31, 1).unwrap(), 4);
        assert_eq!(bitmap_window_bytes(32, 1).unwrap(), 8);
        assert!(bitmap_window_bytes(u32::MAX, 1).is_err());
    }

    #[test]
    fn sidecar_source_ranges_reject_only_real_overlap() {
        assert!(ranges_overlap(0, 16, 8, 16));
        assert!(!ranges_overlap(0, 16, 16, 3));
        assert!(!ranges_overlap(0, 0, 0, 0));
    }
}
