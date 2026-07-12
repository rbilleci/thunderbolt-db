use crate::{CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError, check_cuda};
use std::ffi::c_void;

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
// its value byte (0/1) and, iff 1, atomic-ORs the bit into word `(base_row+t)>>5`. FALSE rows write
// nothing — the open shard's bitmap headroom is pre-zeroed at admission, and prior appends only set (never
// clear) bits, so ORing new true-bits is correct without reading the boundary word (no host mirror, no
// readback). atom.or handles the case where several appended rows land in the SAME 32-bit word.
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

    // val = values[t]  (u8, zero-extended); false -> leave bit 0
    cvt.u64.u32 %rd4, %r6;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.u8 %r7, [%rd5];
    setp.eq.u32 %p2, %r7, 0;
    @%p2 bra DONE;

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
    atom.global.or.b32 %r12, [%rd7], %r11;

DONE:
    ret;
}
"#;

// TYPE-COVERAGE #14 (bool): recompact ONE shard's bool bitmap into the unified buffer's bool bitmap at an
// ARBITRARY (not necessarily 32-row-aligned) destination base. Thread `l` owns the shard's local row `l`;
// it reads source bit `l` and, iff set, atomic-ORs destination bit `dst_base_row + l`. A byte-copy cannot
// do this when `dst_base_row % 32 != 0` (shards seal at arbitrary row counts), so this per-bit gather is
// the alignment-free repack. The unified region is pre-zeroed (RecompactFill 0x00) so only set bits are
// written; atom.or handles rows from different shards that land in the SAME destination word.
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
    @%p2 bra DONE;

    // dst bit (dst_base_row + l): word = (base+l)>>5 ; bit = (base+l)&31 ; mask = 1<<bit
    add.u32 %r11, %r1, %r6;
    shr.u32 %r12, %r11, 5;
    and.b32 %r13, %r11, 31;
    mov.u32 %r14, 1;
    shl.b32 %r14, %r14, %r13;
    mul.wide.u32 %rd7, %r12, 4;
    add.u64 %rd8, %rd1, %rd2;
    add.u64 %rd8, %rd8, %rd7;
    atom.global.or.b32 %r15, [%rd8], %r14;

DONE:
    ret;
}
"#;

// ADR-006 (NULL coverage): recompact ONE shard's VALIDITY bitmap into the unified buffer at an ARBITRARY
// (not necessarily 32-row-aligned) destination base — the alignment-free twin of the bool gather. Validity
// semantics are INVERTED vs bool: 1 = valid, 0 = NULL, and a row/shard WITHOUT a bitmap is all-valid. So the
// unified region is PRE-FILLED 0xFF (all valid) and this kernel only acts on NULL rows: thread `l` reads
// source bit `l`, and iff it is 0 (NULL) atomic-AND-CLEARS destination bit `dst_base_row + l`. Valid bits
// leave the pre-filled 1 untouched; atom.and handles rows from different shards sharing a destination word.
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
    // valid (bit==1) -> leave the pre-filled 0xFF; only a NULL (bit==0) clears the dst bit.
    setp.ne.u32 %p2, %r10, 0;
    @%p2 bra DONE;

    // dst bit (dst_base_row + l): word = (base+l)>>5 ; bit = (base+l)&31 ; clear mask = ~(1<<bit)
    add.u32 %r11, %r1, %r6;
    shr.u32 %r12, %r11, 5;
    and.b32 %r13, %r11, 31;
    mov.u32 %r14, 1;
    shl.b32 %r14, %r14, %r13;
    not.b32 %r14, %r14;
    mul.wide.u32 %rd7, %r12, 4;
    add.u64 %rd8, %rd1, %rd2;
    add.u64 %rd8, %rd8, %rd7;
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
    .param .u32 count
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<16>;

    ld.param.u64 %rd1, [dst_ptr];
    ld.param.u64 %rd2, [dst_offsets_byte_offset];
    ld.param.u32 %r1, [dst_base_row];
    ld.param.u64 %rd3, [blob_base];
    ld.param.u64 %rd4, [src_ptr];
    ld.param.u64 %rd5, [src_offsets_byte_offset];
    ld.param.u32 %r2, [count];

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
    // dst_val = src_off + blob_base
    add.u64 %rd9, %rd8, %rd3;
    // dst_idx = dst_base_row + i ; addr = dst_ptr + dst_offsets_byte_offset + dst_idx*8
    add.u32 %r7, %r1, %r6;
    mul.wide.u32 %rd10, %r7, 8;
    add.u64 %rd11, %rd1, %rd2;
    add.u64 %rd11, %rd11, %rd10;
    st.global.u64 [%rd11], %rd9;

DONE:
    ret;
}
"#;

impl CudaResidentDeviceMemory {
    /// U1 perf lever B: SCATTER `values[t]` into `deleted_by[slots[t]]` in ONE launch (a single
    /// `slots` HtoD + a single `values` HtoD + one kernel), replacing the per-slot
    /// `append_owned_chunks` HtoD loop the lane tombstone pass used (measured device-apply
    /// ~468us/wave at ~75 tombstones = N tiny HtoDs). `self` is the `deleted_by` region (u64
    /// array, slot `s` at byte `s*8`); the caller guarantees every `slot < capacity` (bounds
    /// pre-checked host-side, exactly as the chunk path relied on `append_owned_chunks`'
    /// per-chunk check). Synchronous: null-stream launch + `cuCtxSynchronize`, so the stamps are
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

        if slots.is_empty() {
            return Ok(());
        }
        if slots.len() != values.len() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(slots.len()));
        }
        if self.device_ptr() == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
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
        drop(slots_guard);
        drop(values_guard);
        Ok(())
    }

    /// TYPE-COVERAGE #14 (bool): set the value bits of `values.len()` appended rows into THIS shard
    /// buffer's bool column bitmap at `bitmap_byte_offset` (the column's section start within the buffer),
    /// starting at local row `base_row`. `values[i]` is 0/1 for appended row `base_row + i`. The kernel
    /// atomic-ORs only the TRUE bits (the headroom is pre-zeroed, so false rows and untouched prior bits
    /// stay correct) — the bool analog of `scatter_u64_slots`. Synchronous (ctx-sync) so the bits are
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
        if self.device_ptr() == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let count = u32::try_from(values.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(values.len()))?;
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
        drop(values_guard);
        Ok(())
    }

    /// TYPE-COVERAGE #14 (bool): gather ONE source shard's bool bitmap (`count` local rows at
    /// `src_bitmap_offset` in `src_device_ptr`) into THIS (unified) buffer's bool bitmap at
    /// `dst_bitmap_offset`, placing the shard's row `l` at unified row `dst_base_row + l`. Device->device
    /// (no HtoD): the kernel atomic-ORs each set source bit into the pre-zeroed unified region, so it works
    /// at ANY `dst_base_row` (shards seal at arbitrary, non-32-aligned row counts). Synchronous (ctx-sync).
    pub fn gather_bool_bitmap_from_shard(
        &self,
        dst_bitmap_offset: u64,
        dst_base_row: u32,
        src_device_ptr: u64,
        src_bitmap_offset: u64,
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
        if self.device_ptr() == 0 || src_device_ptr == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }

        let primary = self.primary_arc();
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
        let mut src_off_arg = src_bitmap_offset;
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
        Ok(())
    }

    /// ADR-006 (NULL coverage): gather ONE source shard's VALIDITY bitmap (`count` local rows at
    /// `src_bitmap_offset` in `src_device_ptr`) into THIS (unified) buffer's validity bitmap at
    /// `dst_bitmap_offset`, placing the shard's row `l` at unified row `dst_base_row + l`. The unified region
    /// is PRE-FILLED 0xFF (all valid); the kernel atomic-AND-CLEARS each NULL source bit — the alignment-free
    /// twin of `gather_bool_bitmap_from_shard`, working at ANY `dst_base_row`. Synchronous (ctx-sync).
    pub fn gather_null_bitmap_from_shard(
        &self,
        dst_bitmap_offset: u64,
        dst_base_row: u32,
        src_device_ptr: u64,
        src_bitmap_offset: u64,
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
        if self.device_ptr() == 0 || src_device_ptr == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }

        let primary = self.primary_arc();
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
        let mut src_off_arg = src_bitmap_offset;
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
        Ok(())
    }

    /// TYPE-COVERAGE #14 (text): rebase ONE source shard's `count` (= row_count+1) text offsets into THIS
    /// (unified) buffer's offsets section at `dst_offsets_byte_offset`, placing the shard's offset `i` at
    /// unified offset `dst_base_row + i` with `blob_base` added (the shard's running byte position in the
    /// concatenated unified blob). Device->device (no HtoD). Synchronous (ctx-sync).
    pub fn rebase_text_offsets_from_shard(
        &self,
        dst_offsets_byte_offset: u64,
        dst_base_row: u32,
        blob_base: u64,
        src_device_ptr: u64,
        src_offsets_byte_offset: u64,
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
        if self.device_ptr() == 0 || src_device_ptr == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }

        let primary = self.primary_arc();
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

        let mut ptx = Vec::with_capacity(TEXT_OFFSET_REBASE_PTX.len() + 1);
        ptx.extend_from_slice(TEXT_OFFSET_REBASE_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_resident_text_offset_rebase", &ptx)?;

        let mut dst_ptr_arg = self.device_ptr();
        let mut dst_off_arg = dst_offsets_byte_offset;
        let mut dst_base_arg = dst_base_row;
        let mut blob_base_arg = blob_base;
        let mut src_ptr_arg = src_device_ptr;
        let mut src_off_arg = src_offsets_byte_offset;
        let mut count_arg = count;
        let mut args = [
            (&mut dst_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut dst_off_arg as *mut u64).cast::<c_void>(),
            (&mut dst_base_arg as *mut u32).cast::<c_void>(),
            (&mut blob_base_arg as *mut u64).cast::<c_void>(),
            (&mut src_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut src_off_arg as *mut u64).cast::<c_void>(),
            (&mut count_arg as *mut u32).cast::<c_void>(),
        ];
        let threads_per_block: u32 = 128;
        let blocks = count.div_ceil(threads_per_block);
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
        Ok(())
    }
}
