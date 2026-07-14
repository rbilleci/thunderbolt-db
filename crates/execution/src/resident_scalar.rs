use std::os::raw::c_void;

use super::resident_count::validity_bitmap_kernel_arg;
use super::{
    launch_on_pooled_stream, CudaI32Comparison, CudaResidentDeviceMemory, CudaRuntimeProbeError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaI32Stats {
    pub count: u64,
    pub sum: i64,
    pub min: Option<i32>,
    pub max: Option<i32>,
}

impl CudaI32Stats {
    fn from_values(values: &[i32]) -> Self {
        Self {
            count: values.len() as u64,
            sum: values.iter().map(|value| i64::from(*value)).sum(),
            min: values.iter().copied().min(),
            max: values.iter().copied().max(),
        }
    }
}

impl CudaResidentDeviceMemory {
    pub fn sum_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
    ) -> Result<i64, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_sum(self, byte_offset, row_count)
    }

    /// DIRECT scalar (count, sum, min, max) over a resident, non-nullable, UNFILTERED int4 column in
    /// ONE streaming pass — the MIN/MAX/AVG analogue of [`sum_i32_from_payload`]. Replaced the
    /// (now-removed) self-grouped hash kernel with group==value, which built an O(distinct)-entry hash
    /// table just to reduce; over a high-distinct
    /// column that hash table dominates (~182 Melem/s and falling at 8M distinct). This is a grid-stride
    /// scan + a `bar.sync` shared-memory block tree reduction of all four partials + ONE set of four
    /// global atomics per block (the audited `gpu_db_resident_i32_sum` pattern), so it runs at the same
    /// memory-bound roofline regardless of distinctness. Returns `(count, sum, min, max)`; `count == 0`
    /// (empty input) leaves min/max at their `INT_MAX`/`INT_MIN` init sentinels and the caller maps it
    /// to SQL NULL.
    pub fn scalar_stats_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
    ) -> Result<(u64, i64, i32, i32), CudaRuntimeProbeError> {
        launch_cuda_resident_i32_scalar_stats(self, byte_offset, row_count, None, None)
    }

    /// NULL-aware DIRECT scalar (count, sum, min, max) — the unfiltered nullable analogue of
    /// [`scalar_stats_i32_from_payload`]. A NULL value (per `null_bitmap_offset`, 1 = valid) contributes
    /// to NO statistic and is NOT counted, modelled EXACTLY on the grouped hash kernel's validity-bitmap
    /// logic (sentinel `0xFFFF...` = no bitmap = every row valid). Replaced the (now-removed) self-grouped
    /// NULL-aware hash kernel with group==value on the scalar arm. `count` is the SURVIVING (non-NULL) row
    /// count; the caller maps `count == 0` (all-NULL column) to SQL NULL.
    pub fn nullable_scalar_stats_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        null_bitmap_offset: Option<u64>,
    ) -> Result<(u64, i64, i32, i32), CudaRuntimeProbeError> {
        launch_cuda_resident_i32_scalar_stats(
            self,
            byte_offset,
            row_count,
            None,
            null_bitmap_offset,
        )
    }

    /// FILTERED (+ optionally NULL-aware) DIRECT scalar (count, sum, min, max) — the filtered analogue of
    /// [`scalar_stats_i32_from_payload`]. The filter `<col> <cmp> needle` runs ON-DEVICE (per-row
    /// predication, comparison codes 1=lt/2=lte/3=gt/4=gte matching the grouped hash kernel EXACTLY) and
    /// non-matching rows are SKIPPED; a NULL value (per `null_bitmap_offset`, 1 = valid; `None` = no
    /// bitmap) is ALSO skipped (3VL — a NULL contributes to no statistic). `count` is the count of
    /// SURVIVING (matching, non-NULL) rows; the caller maps `count == 0` (zero matches / all-NULL
    /// survivors) to SQL NULL. Replaced the now-removed self-grouped filtered hash kernel and
    /// gather-to-host projection plus CPU-reduction path. Byte-identical to those for non-empty results.
    pub fn filtered_scalar_stats_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
        null_bitmap_offset: Option<u64>,
    ) -> Result<(u64, i64, i32, i32), CudaRuntimeProbeError> {
        launch_cuda_resident_i32_scalar_stats(
            self,
            byte_offset,
            row_count,
            Some((byte_offset, needle, comparison)),
            null_bitmap_offset,
        )
    }

    pub fn stats_i32_between_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        lower_inclusive: i32,
        upper_inclusive: i32,
    ) -> Result<CudaI32Stats, CudaRuntimeProbeError> {
        if lower_inclusive > upper_inclusive {
            return Ok(CudaI32Stats::from_values(&[]));
        }
        launch_cuda_resident_i32_between_stats(
            self,
            byte_offset,
            row_count,
            lower_inclusive,
            upper_inclusive,
            None,
        )
    }

    /// NULL-aware BETWEEN stats (M3 — doc 21): like [`Self::stats_i32_between_from_payload`] but a NULL
    /// value (per `null_bitmap_offset`, 1 = valid) never satisfies the range, so it is excluded from
    /// count/sum/min/max. `None` reduces to the plain path. The caller maps a zero `count` to SQL NULL.
    pub fn stats_i32_between_nullable_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        lower_inclusive: i32,
        upper_inclusive: i32,
        null_bitmap_offset: Option<u64>,
    ) -> Result<CudaI32Stats, CudaRuntimeProbeError> {
        if lower_inclusive > upper_inclusive {
            return Ok(CudaI32Stats::from_values(&[]));
        }
        launch_cuda_resident_i32_between_stats(
            self,
            byte_offset,
            row_count,
            lower_inclusive,
            upper_inclusive,
            null_bitmap_offset,
        )
    }
}

fn launch_cuda_resident_i32_sum(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
) -> Result<i64, CudaRuntimeProbeError> {
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

    // P2-M2 — the i32-sum kernel is already a parallel grid-stride reduction (each thread sums a
    // strided slice into an s64, then `atom.global.add.u64`s it into one output). This migrates the
    // LAUNCH off the default/null stream + per-call cuModuleLoadData (re-JIT) + per-call cuMemAlloc
    // onto a pooled private stream with a cached module + async-memset scratch via
    // `launch_on_pooled_stream` (event-timed, covering-synced, drained-on-error). Kernel unchanged.

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_sum(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .u64 out_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_active;
    .reg .pred %p_isthr0;
    .shared .align 8 .b64 s_part[1024];
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %addr;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %thread;
    .reg .u32 %grid_dim;
    .reg .u64 %wide_block;
    .reg .u64 %wide_thread;
    .reg .u64 %wide_block_dim;
    .reg .u64 %wide_grid_dim;
    .reg .u64 %sum_bits;
    .reg .u64 %ignored;
    .reg .s64 %sum;
    .reg .s64 %wide;
    .reg .s32 %r_value;
    .reg .u32 %rstride;
    .reg .u32 %peer;
    .reg .u64 %sh_base;
    .reg .u64 %sh_self;
    .reg .u64 %sh_peer;
    .reg .u64 %peer_val;
    .reg .u64 %block_tot;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %out, [out_ptr];

    add.u64 %base, %resident, %offset;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mov.u32 %thread, %tid.x;
    mov.u32 %grid_dim, %nctaid.x;
    cvt.u64.u32 %wide_block, %r_block;
    cvt.u64.u32 %wide_thread, %thread;
    cvt.u64.u32 %wide_block_dim, %r_block_dim;
    cvt.u64.u32 %wide_grid_dim, %grid_dim;
    mul.lo.u64 %idx, %wide_block, %wide_block_dim;
    add.u64 %idx, %idx, %wide_thread;
    mul.lo.u64 %stride, %wide_grid_dim, %wide_block_dim;
    mov.s64 %sum, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %addr, %idx, 4;
    add.u64 %addr, %base, %addr;
    ld.global.s32 %r_value, [%addr];
    cvt.s64.s32 %wide, %r_value;
    add.s64 %sum, %sum, %wide;
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    // ---- per-block reduction: sum every thread's i64 partial, then ONE atomic per block ----
    // The grid is clamped to a saturating constant (<=1024 blocks), so each thread accumulates a REAL
    // i64 partial over its grid-stride rows. Summing the per-thread partials in a barrier-synchronized
    // SHARED-MEMORY tree, then a single atom.add per block, replaces the old one-atomic-per-thread storm
    // (grid*BLOCK atomics on one address). Byte-identical to the old per-thread accumulation: two's-
    // complement (u64) addition is associative AND commutative, so the i64 sum is independent of the
    // grouping/order of the adds (mod 2^64). A shfl.sync would be WRONG here -- the grid-stride loop
    // exits per-thread (idx >= rows), so lanes within a warp run a DIFFERENT number of iterations and are
    // NOT converged at done:; bar.sync synchronizes the WHOLE block regardless, and every barrier below
    // is on the straight-line path (outside the @!%p_active guard) so all threads reach it. %thread =
    // %tid.x (in-block id); %r_block_dim = %ntid.x (block width, a power of two so the tree terminates).
    cvt.u64.s64 %sum_bits, %sum;
    mov.u64 %sh_base, s_part;
    mul.wide.u32 %sh_self, %thread, 8;
    add.u64 %sh_self, %sh_base, %sh_self;
    st.shared.u64 [%sh_self], %sum_bits;
    bar.sync 0;

    // tree reduce: for rstride = bdim/2, bdim/4, ..., 1: s_part[t] += s_part[t + rstride] for t < rstride.
    shr.u32 %rstride, %r_block_dim, 1;
red_loop:
    setp.eq.u32 %p_done, %rstride, 0;
    @%p_done bra red_done;
    setp.lt.u32 %p_active, %thread, %rstride;
    @!%p_active bra red_skip;
    add.u32 %peer, %thread, %rstride;
    mul.wide.u32 %sh_peer, %peer, 8;
    add.u64 %sh_peer, %sh_base, %sh_peer;
    ld.shared.u64 %peer_val, [%sh_peer];
    ld.shared.u64 %sum_bits, [%sh_self];
    add.u64 %sum_bits, %sum_bits, %peer_val;
    st.shared.u64 [%sh_self], %sum_bits;
red_skip:
    bar.sync 0;
    shr.u32 %rstride, %rstride, 1;
    bra red_loop;
red_done:
    // thread 0 holds the block total at s_part[0]; ONE global atomic add for the whole block.
    setp.eq.u32 %p_isthr0, %thread, 0;
    @!%p_isthr0 bra block_done;
    ld.shared.u64 %block_tot, [%sh_base];
    atom.global.add.u64 %ignored, [%out], %block_tot;
block_done:
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8_async = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8Async>(b"cuMemsetD8Async\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_sum", &ptx)?;

    let block_dim = 256_u32;
    let grid_dim = if row_count == 0 {
        1
    } else {
        row_count.div_ceil(u64::from(block_dim)).min(1024) as u32
    };

    let mut output_bytes = [0_u8; std::mem::size_of::<i64>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        // Zero the 8-byte scratch on the stream (the kernel atom-adds into it), ordered before the
        // kernel launch on the same stream.
        let memset_rc =
            unsafe { cu_memset_d8_async(output_ptr, 0, std::mem::size_of::<i64>(), stream) };
        if memset_rc != 0 {
            return memset_rc;
        }
        let mut resident_arg = resident.device_ptr();
        let mut offset_arg = byte_offset;
        let mut rows_arg = row_count;
        let mut output_arg = output_ptr;
        let mut args = [
            (&mut resident_arg as *mut u64).cast::<c_void>(),
            (&mut offset_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                function,
                grid_dim,
                1,
                1,
                block_dim,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;

    Ok(i64::from_le_bytes(output_bytes))
}

fn launch_cuda_resident_i32_scalar_stats(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    // `Some((filter_byte_offset, needle, comparison))` = an on-device per-row filter `<col> <cmp>
    // needle` (non-matches skipped); `None` = no filter (the unfiltered fast path). The scalar arms
    // always pass `filter_byte_offset == byte_offset` (the filter and aggregate are the SAME column).
    filter: Option<(u64, i32, CudaI32Comparison)>,
    // M3 (doc 21): `Some(off)` = the value column's NULL validity bitmap byte offset (1 = valid); `None`
    // = no bitmap ⇒ every row valid (the no-NULL fast path). A NULL value contributes to NO statistic.
    null_bitmap_offset: Option<u64>,
) -> Result<(u64, i64, i32, i32), CudaRuntimeProbeError> {
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

    // DIRECT scalar-stats reduction — the MIN/MAX/AVG analogue of `gpu_db_resident_i32_sum`, with an
    // OPTIONAL on-device per-row filter and an OPTIONAL NULL-skip. Each thread accumulates (count, i64
    // sum, i32 min, i32 max) over its grid-stride slice; per row it (1) optionally evaluates the filter
    // `<col> <cmp> needle` (comparison 0=none/1=lt/2=lte/3=gt/4=gte) and SKIPS non-matches, then (2)
    // optionally reads the validity bit (sentinel `0xFFFF...` =
    // no bitmap ⇒ valid; the standard per-row validity-bitmap convention) and
    // SKIPS NULL rows. Surviving rows do count++, sum+=v, min/max. The block then reduces all four
    // partials in a `bar.sync` SHARED-MEMORY tree (NOT shfl — the grid-stride loop exits per-thread at
    // `done:` so warp lanes run a DIFFERENT iteration count and are NOT converged; bar.sync synchronizes
    // the WHOLE block and every barrier below is on the straight-line path), and thread 0 issues ONE set
    // of four global atomics for the block: add count, add sum (u64 two's-complement), min.s32, max.s32.
    // count/sum atomic-adds are order-independent (mod 2^64), min/max are associative/commutative, so the
    // result is byte-identical to the self-grouped path's reduced (count, sum, min, max). The UNFILTERED
    // NON-NULLABLE path (comparison==0, null_off==sentinel) takes a fast straight-line branch with NO
    // per-row filter/bitmap load — byte- and speed-identical to the slice-a kernel. Host inits the
    // 24-byte out struct count=0, sum=0, min=INT_MAX, max=INT_MIN (the min/max sentinels can't be a
    // plain memset, so we H2D the init struct on-stream before the kernel). `.target sm_60` for the
    // global min/max atomics. Saturating grid (.min) like sum; row_count==0 (or zero survivors) leaves
    // the out struct at its init (count 0 => caller maps to SQL NULL).
    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_resident_i32_scalar_stats(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .u64 out_ptr,
    .param .u64 filter_byte_offset,
    .param .s32 needle,
    .param .u32 comparison,
    .param .u64 value_null_bitmap_offset
)
{
    .reg .pred %p_done;
    .reg .pred %p_active;
    .reg .pred %p_isthr0;
    .reg .pred %p_check;
    .reg .pred %p_match;
    .reg .pred %p_no_bitmap;
    .reg .pred %p_valid;
    .shared .align 8 .b64 s_count[1024];
    .shared .align 8 .b64 s_sum[1024];
    .shared .align 4 .b32 s_min[1024];
    .shared .align 4 .b32 s_max[1024];
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %filter_base;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %addr;
    .reg .u64 %roff;
    .reg .u32 %comparison;
    .reg .s32 %needle;
    .reg .s32 %filter_value;
    .reg .u64 %val_null_off;
    .reg .u64 %sentinel;
    .reg .u64 %word_byte;
    .reg .u64 %bitmap_addr;
    .reg .u32 %bitmap_word;
    .reg .u32 %bit_pos;
    .reg .u32 %valid_bit;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %thread;
    .reg .u32 %grid_dim;
    .reg .u64 %wide_block;
    .reg .u64 %wide_thread;
    .reg .u64 %wide_block_dim;
    .reg .u64 %wide_grid_dim;
    .reg .u64 %count;
    .reg .u64 %sum_bits;
    .reg .u64 %ignored64;
    .reg .s32 %ignored32;
    .reg .s64 %sum;
    .reg .s64 %wide;
    .reg .s32 %r_value;
    .reg .s32 %min;
    .reg .s32 %max;
    .reg .u32 %rstride;
    .reg .u32 %peer;
    .reg .u64 %sh_count_base;
    .reg .u64 %sh_sum_base;
    .reg .u64 %sh_min_base;
    .reg .u64 %sh_max_base;
    .reg .u64 %sh_count_self;
    .reg .u64 %sh_sum_self;
    .reg .u64 %sh_min_self;
    .reg .u64 %sh_max_self;
    .reg .u64 %sh_count_peer;
    .reg .u64 %sh_sum_peer;
    .reg .u64 %sh_min_peer;
    .reg .u64 %sh_max_peer;
    .reg .u64 %off8;
    .reg .u64 %off4;
    .reg .u64 %peer_count;
    .reg .u64 %peer_sum;
    .reg .s32 %peer_min;
    .reg .s32 %peer_max;
    .reg .u64 %count_addr;
    .reg .u64 %sum_addr;
    .reg .u64 %min_addr;
    .reg .u64 %max_addr;
    .reg .u64 %block_count;
    .reg .u64 %block_sum;
    .reg .s32 %block_min;
    .reg .s32 %block_max;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %filter_base, [filter_byte_offset];
    ld.param.s32 %needle, [needle];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %val_null_off, [value_null_bitmap_offset];

    add.u64 %base, %resident, %offset;
    add.u64 %filter_base, %resident, %filter_base;
    mov.u64 %sentinel, 0xFFFFFFFFFFFFFFFF;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mov.u32 %thread, %tid.x;
    mov.u32 %grid_dim, %nctaid.x;
    cvt.u64.u32 %wide_block, %r_block;
    cvt.u64.u32 %wide_thread, %thread;
    cvt.u64.u32 %wide_block_dim, %r_block_dim;
    cvt.u64.u32 %wide_grid_dim, %grid_dim;
    mul.lo.u64 %idx, %wide_block, %wide_block_dim;
    add.u64 %idx, %idx, %wide_thread;
    mul.lo.u64 %stride, %wide_grid_dim, %wide_block_dim;
    mov.u64 %count, 0;
    mov.s64 %sum, 0;
    mov.s32 %min, 2147483647;
    mov.s32 %max, -2147483648;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %roff, %idx, 4;

    // Optional on-device filter `<col> <cmp> needle` (comparison 0=none/1=lt/2=lte/3=gt/4=gte).
    // comparison==0 => no filter, fall
    // straight through (the unfiltered fast branch, no filter load). Non-matches skip this row.
    setp.eq.u32 %p_check, %comparison, 0;
    @%p_check bra after_filter;
    add.u64 %addr, %filter_base, %roff;
    ld.global.s32 %filter_value, [%addr];
    mov.pred %p_match, 0;
    setp.eq.u32 %p_check, %comparison, 1;
    @%p_check bra f_lt;
    setp.eq.u32 %p_check, %comparison, 2;
    @%p_check bra f_lte;
    setp.eq.u32 %p_check, %comparison, 3;
    @%p_check bra f_gt;
    setp.eq.u32 %p_check, %comparison, 4;
    @%p_check bra f_gte;
    bra next_row;
f_lt:
    setp.lt.s32 %p_match, %filter_value, %needle;
    bra f_done;
f_lte:
    setp.le.s32 %p_match, %filter_value, %needle;
    bra f_done;
f_gt:
    setp.gt.s32 %p_match, %filter_value, %needle;
    bra f_done;
f_gte:
    setp.ge.s32 %p_match, %filter_value, %needle;
f_done:
    @!%p_match bra next_row;

after_filter:
    // Optional NULL-skip (M3 3VL): a NULL value contributes to no statistic. val_null_off == sentinel
    // (0xFFFF...) => no validity bitmap => every row valid (skip the load). Modelled EXACTLY on the
    // grouped hash kernel's validity-bitmap logic (1 = valid/present, 0 = NULL).
    setp.eq.u64 %p_no_bitmap, %val_null_off, %sentinel;
    @%p_no_bitmap bra accumulate;
    shr.u64 %word_byte, %idx, 5;          // idx / 32 (the validity word index)
    mul.lo.u64 %word_byte, %word_byte, 4; // * 4 bytes per u32 word
    add.u64 %bitmap_addr, %resident, %val_null_off;
    add.u64 %bitmap_addr, %bitmap_addr, %word_byte;
    ld.global.u32 %bitmap_word, [%bitmap_addr];
    cvt.u32.u64 %bit_pos, %idx;
    and.b32 %bit_pos, %bit_pos, 31;       // idx % 32
    bfe.u32 %valid_bit, %bitmap_word, %bit_pos, 1;
    setp.eq.u32 %p_valid, %valid_bit, 1;  // 1 = valid/present, 0 = NULL
    @!%p_valid bra next_row;              // NULL value => skip this row

accumulate:
    add.u64 %addr, %base, %roff;
    ld.global.s32 %r_value, [%addr];
    cvt.s64.s32 %wide, %r_value;
    add.s64 %sum, %sum, %wide;
    add.u64 %count, %count, 1;
    min.s32 %min, %min, %r_value;
    max.s32 %max, %max, %r_value;

next_row:
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    // ---- per-block reduction: tree-reduce all four partials in shared memory, then ONE set of four
    // atomics per block. Grid is saturating-clamped (<=1024 blocks) so each thread holds a REAL partial
    // over its grid-stride rows. count/sum (u64 two's-complement add) are associative+commutative mod
    // 2^64; min/max are associative+commutative; so the tree grouping is byte-identical to a flat
    // accumulation. %thread = %tid.x (in-block id); %r_block_dim = %ntid.x (a power of two so the tree
    // terminates). Every barrier below is on the straight-line path (outside the @!%p_active guard) so
    // all threads in the block reach it even though the grid-stride loop exited per-thread above.
    mov.u64 %sh_count_base, s_count;
    mov.u64 %sh_sum_base, s_sum;
    mov.u64 %sh_min_base, s_min;
    mov.u64 %sh_max_base, s_max;
    mul.wide.u32 %off8, %thread, 8;
    mul.wide.u32 %off4, %thread, 4;
    add.u64 %sh_count_self, %sh_count_base, %off8;
    add.u64 %sh_sum_self, %sh_sum_base, %off8;
    add.u64 %sh_min_self, %sh_min_base, %off4;
    add.u64 %sh_max_self, %sh_max_base, %off4;
    cvt.u64.s64 %sum_bits, %sum;
    st.shared.u64 [%sh_count_self], %count;
    st.shared.u64 [%sh_sum_self], %sum_bits;
    st.shared.s32 [%sh_min_self], %min;
    st.shared.s32 [%sh_max_self], %max;
    bar.sync 0;

    // tree reduce: for rstride = bdim/2, bdim/4, ..., 1: combine s[t] with s[t + rstride] for t < rstride.
    shr.u32 %rstride, %r_block_dim, 1;
red_loop:
    setp.eq.u32 %p_done, %rstride, 0;
    @%p_done bra red_done;
    setp.lt.u32 %p_active, %thread, %rstride;
    @!%p_active bra red_skip;
    add.u32 %peer, %thread, %rstride;
    mul.wide.u32 %off8, %peer, 8;
    mul.wide.u32 %off4, %peer, 4;
    add.u64 %sh_count_peer, %sh_count_base, %off8;
    add.u64 %sh_sum_peer, %sh_sum_base, %off8;
    add.u64 %sh_min_peer, %sh_min_base, %off4;
    add.u64 %sh_max_peer, %sh_max_base, %off4;
    ld.shared.u64 %peer_count, [%sh_count_peer];
    ld.shared.u64 %peer_sum, [%sh_sum_peer];
    ld.shared.s32 %peer_min, [%sh_min_peer];
    ld.shared.s32 %peer_max, [%sh_max_peer];
    ld.shared.u64 %count, [%sh_count_self];
    ld.shared.u64 %sum_bits, [%sh_sum_self];
    ld.shared.s32 %min, [%sh_min_self];
    ld.shared.s32 %max, [%sh_max_self];
    add.u64 %count, %count, %peer_count;
    add.u64 %sum_bits, %sum_bits, %peer_sum;
    min.s32 %min, %min, %peer_min;
    max.s32 %max, %max, %peer_max;
    st.shared.u64 [%sh_count_self], %count;
    st.shared.u64 [%sh_sum_self], %sum_bits;
    st.shared.s32 [%sh_min_self], %min;
    st.shared.s32 [%sh_max_self], %max;
red_skip:
    bar.sync 0;
    shr.u32 %rstride, %rstride, 1;
    bra red_loop;
red_done:
    // thread 0 holds the block totals at index 0; ONE set of four global atomics for the whole block.
    setp.eq.u32 %p_isthr0, %thread, 0;
    @!%p_isthr0 bra block_done;
    ld.shared.u64 %block_count, [%sh_count_base];
    ld.shared.u64 %block_sum, [%sh_sum_base];
    ld.shared.s32 %block_min, [%sh_min_base];
    ld.shared.s32 %block_max, [%sh_max_base];
    mov.u64 %count_addr, %out;
    atom.global.add.u64 %ignored64, [%count_addr], %block_count;
    add.u64 %sum_addr, %out, 8;
    atom.global.add.u64 %ignored64, [%sum_addr], %block_sum;
    add.u64 %min_addr, %out, 16;
    atom.global.min.s32 %ignored32, [%min_addr], %block_min;
    add.u64 %max_addr, %out, 20;
    atom.global.max.s32 %ignored32, [%max_addr], %block_max;
block_done:
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }

    // Filter args: `None` => comparison 0 (no filter; filter_base unused, point it at the value column).
    // `Some` => the bounds-checked filter column + needle + the comparison `.code()` (1..4, the SAME
    // mapping the grouped hash kernel uses).
    let (filter_byte_offset, needle, comparison_code) = match filter {
        Some((filter_byte_offset, needle, comparison)) => {
            let filter_bytes = row_count
                .checked_mul(std::mem::size_of::<i32>() as u64)
                .and_then(|bytes| filter_byte_offset.checked_add(bytes))
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if filter_bytes > resident.metadata().allocated_bytes {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    filter_bytes as usize,
                ));
            }
            (filter_byte_offset, needle, comparison.code())
        }
        None => (byte_offset, 0, 0),
    };
    // u64::MAX sentinel when there is no validity bitmap; otherwise the bounds-checked byte offset.
    let null_off_value = validity_bitmap_kernel_arg(null_bitmap_offset, row_count, resident)?;

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = resident
        .primary()
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_scalar_stats", &ptx)?;

    let block_dim = 256_u32;
    let grid_dim = if row_count == 0 {
        1
    } else {
        row_count.div_ceil(u64::from(block_dim)).min(1024) as u32
    };

    // Init struct H2D'd into the scratch on-stream BEFORE the kernel: count/sum start at 0 (atomic
    // add), min/max at the INT sentinels (atomic min/max — they can't be a plain memset). `initial`
    // outlives the helper's covering sync, so the async source stays valid until the copy completes.
    let initial = CudaI32StatsRaw {
        count: 0,
        sum: 0,
        min: i32::MAX,
        max: i32::MIN,
    };
    let mut output_bytes = [0_u8; std::mem::size_of::<CudaI32StatsRaw>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        let htod_rc = unsafe {
            htod_async(
                output_ptr,
                (&initial as *const CudaI32StatsRaw).cast::<c_void>(),
                std::mem::size_of::<CudaI32StatsRaw>(),
                stream,
            )
        };
        if htod_rc != 0 {
            return htod_rc;
        }
        let mut resident_arg = resident.device_ptr();
        let mut offset_arg = byte_offset;
        let mut rows_arg = row_count;
        let mut output_arg = output_ptr;
        let mut filter_off_arg = filter_byte_offset;
        let mut needle_arg = needle;
        let mut comparison_arg = comparison_code;
        let mut null_off_arg = null_off_value;
        let mut args = [
            (&mut resident_arg as *mut u64).cast::<c_void>(),
            (&mut offset_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut filter_off_arg as *mut u64).cast::<c_void>(),
            (&mut needle_arg as *mut i32).cast::<c_void>(),
            (&mut comparison_arg as *mut u32).cast::<c_void>(),
            (&mut null_off_arg as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                function,
                grid_dim,
                1,
                1,
                block_dim,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;

    let count = u64::from_le_bytes(output_bytes[0..8].try_into().unwrap());
    let sum = i64::from_le_bytes(output_bytes[8..16].try_into().unwrap());
    let min = i32::from_le_bytes(output_bytes[16..20].try_into().unwrap());
    let max = i32::from_le_bytes(output_bytes[20..24].try_into().unwrap());
    Ok((count, sum, min, max))
}

#[repr(C)]
struct CudaI32StatsRaw {
    count: u64,
    sum: i64,
    min: i32,
    max: i32,
}

fn launch_cuda_resident_i32_between_stats(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    lower_inclusive: i32,
    upper_inclusive: i32,
    // M3 (doc 21): `Some(off)` = the column's NULL validity bitmap byte offset (1 = valid); `None` = no
    // bitmap ⇒ every row valid. A NULL value never satisfies BETWEEN (3VL).
    null_bitmap_offset: Option<u64>,
) -> Result<CudaI32Stats, CudaRuntimeProbeError> {
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

    // P2-M2 — the between-stats kernel is already a parallel grid-stride reduction (each thread
    // computes count/sum/min/max for the [lower,upper] predicate, then atomic add count/sum +
    // atomic min/max into one 24-byte output struct). Migrate the LAUNCH off the default/null stream
    // + per-call cuModuleLoadData (re-JIT) + per-call cuMemAlloc + blocking H2D/D2H onto a pooled
    // private stream with a cached module and an ASYNC H2D of the init struct
    // (count=0,sum=0,min=INT_MAX,max=INT_MIN — the min/max sentinels can't be memset like sum's plain
    // zero) into the pooled scratch BEFORE the kernel, via launch_on_pooled_stream (event-timed,
    // covering-synced, drained-on-error). Kernel unchanged.

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_between_stats(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .s32 lower_inclusive,
    .param .s32 upper_inclusive,
    .param .u64 out_ptr,
    .param .u64 null_bitmap_offset
)
{
    .reg .pred %p_done;
    .reg .pred %p_ge_lower;
    .reg .pred %p_le_upper;
    .reg .pred %p_match;
    .reg .pred %p_no_bitmap;
    .reg .pred %p_valid;
    .reg .u64 %null_off;
    .reg .u64 %sentinel;
    .reg .u64 %word_byte;
    .reg .u64 %bitmap_addr;
    .reg .u32 %bitmap_word;
    .reg .u32 %bit_pos;
    .reg .u32 %valid_bit;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %addr;
    .reg .u64 %count_addr;
    .reg .u64 %sum_addr;
    .reg .u64 %min_addr;
    .reg .u64 %max_addr;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %thread;
    .reg .u32 %grid_dim;
    .reg .u64 %wide_block;
    .reg .u64 %wide_thread;
    .reg .u64 %wide_block_dim;
    .reg .u64 %wide_grid_dim;
    .reg .u64 %count;
    .reg .u64 %sum_bits;
    .reg .u64 %ignored64;
    .reg .s32 %ignored32;
    .reg .s64 %sum;
    .reg .s64 %wide;
    .reg .s32 %r_value;
    .reg .s32 %lower;
    .reg .s32 %upper;
    .reg .s32 %min;
    .reg .s32 %max;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %lower, [lower_inclusive];
    ld.param.s32 %upper, [upper_inclusive];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %null_off, [null_bitmap_offset];

    add.u64 %base, %resident, %offset;
    mov.u64 %sentinel, 0xFFFFFFFFFFFFFFFF;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mov.u32 %thread, %tid.x;
    mov.u32 %grid_dim, %nctaid.x;
    cvt.u64.u32 %wide_block, %r_block;
    cvt.u64.u32 %wide_thread, %thread;
    cvt.u64.u32 %wide_block_dim, %r_block_dim;
    cvt.u64.u32 %wide_grid_dim, %grid_dim;
    mul.lo.u64 %idx, %wide_block, %wide_block_dim;
    add.u64 %idx, %idx, %wide_thread;
    mul.lo.u64 %stride, %wide_grid_dim, %wide_block_dim;
    mov.u64 %count, 0;
    mov.s64 %sum, 0;
    mov.s32 %min, 2147483647;
    mov.s32 %max, -2147483648;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %addr, %idx, 4;
    add.u64 %addr, %base, %addr;
    ld.global.s32 %r_value, [%addr];
    setp.ge.s32 %p_ge_lower, %r_value, %lower;
    setp.le.s32 %p_le_upper, %r_value, %upper;
    and.pred %p_match, %p_ge_lower, %p_le_upper;
    @!%p_match bra next;
    // M3 3VL: a NULL value never satisfies BETWEEN (its bytes are a 0 placeholder). null_off ==
    // sentinel (0xFFFF...) => no validity bitmap => every row valid (skip the load).
    setp.eq.u64 %p_no_bitmap, %null_off, %sentinel;
    @%p_no_bitmap bra accumulate;
    shr.u64 %word_byte, %idx, 5;          // idx / 32 (validity word index)
    mul.lo.u64 %word_byte, %word_byte, 4; // * 4 bytes per u32 word
    add.u64 %bitmap_addr, %resident, %null_off;
    add.u64 %bitmap_addr, %bitmap_addr, %word_byte;
    ld.global.u32 %bitmap_word, [%bitmap_addr];
    cvt.u32.u64 %bit_pos, %idx;
    and.b32 %bit_pos, %bit_pos, 31;       // idx % 32
    bfe.u32 %valid_bit, %bitmap_word, %bit_pos, 1;
    setp.eq.u32 %p_valid, %valid_bit, 1;  // 1 = valid/present, 0 = NULL
    @!%p_valid bra next;                  // NULL => not a match, skip

accumulate:
    cvt.s64.s32 %wide, %r_value;
    add.s64 %sum, %sum, %wide;
    add.u64 %count, %count, 1;
    min.s32 %min, %min, %r_value;
    max.s32 %max, %max, %r_value;

next:
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    setp.eq.u64 %p_done, %count, 0;
    @%p_done bra ret_done;
    mov.u64 %count_addr, %out;
    atom.global.add.u64 %ignored64, [%count_addr], %count;
    add.u64 %sum_addr, %out, 8;
    cvt.u64.s64 %sum_bits, %sum;
    atom.global.add.u64 %ignored64, [%sum_addr], %sum_bits;
    add.u64 %min_addr, %out, 16;
    atom.global.min.s32 %ignored32, [%min_addr], %min;
    add.u64 %max_addr, %out, 20;
    atom.global.max.s32 %ignored32, [%max_addr], %max;

ret_done:
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = resident
        .primary()
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_between_stats", &ptx)?;

    // u64::MAX sentinel when there is no validity bitmap; otherwise the bounds-checked byte offset.
    let null_off_value = validity_bitmap_kernel_arg(null_bitmap_offset, row_count, resident)?;

    let block_dim = 256_u32;
    let grid_dim = if row_count == 0 {
        1
    } else {
        row_count.div_ceil(u64::from(block_dim)).min(1024) as u32
    };

    // Init struct H2D'd into the scratch on-stream BEFORE the kernel: count/sum start at 0 (atomic
    // add), min/max at the INT sentinels (atomic min/max). `initial` outlives the helper's covering
    // sync, so the async source stays valid until the copy completes.
    let initial = CudaI32StatsRaw {
        count: 0,
        sum: 0,
        min: i32::MAX,
        max: i32::MIN,
    };
    let mut output_bytes = [0_u8; std::mem::size_of::<CudaI32StatsRaw>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        let htod_rc = unsafe {
            htod_async(
                output_ptr,
                (&initial as *const CudaI32StatsRaw).cast::<c_void>(),
                std::mem::size_of::<CudaI32StatsRaw>(),
                stream,
            )
        };
        if htod_rc != 0 {
            return htod_rc;
        }
        let mut resident_arg = resident.device_ptr();
        let mut offset_arg = byte_offset;
        let mut rows_arg = row_count;
        let mut lower_arg = lower_inclusive;
        let mut upper_arg = upper_inclusive;
        let mut output_arg = output_ptr;
        let mut null_off_arg = null_off_value;
        let mut args = [
            (&mut resident_arg as *mut u64).cast::<c_void>(),
            (&mut offset_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut lower_arg as *mut i32).cast::<c_void>(),
            (&mut upper_arg as *mut i32).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut null_off_arg as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                function,
                grid_dim,
                1,
                1,
                block_dim,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;

    let raw = CudaI32StatsRaw {
        count: u64::from_le_bytes(output_bytes[0..8].try_into().unwrap()),
        sum: i64::from_le_bytes(output_bytes[8..16].try_into().unwrap()),
        min: i32::from_le_bytes(output_bytes[16..20].try_into().unwrap()),
        max: i32::from_le_bytes(output_bytes[20..24].try_into().unwrap()),
    };

    Ok(CudaI32Stats {
        count: raw.count,
        sum: raw.sum,
        min: (raw.count > 0).then_some(raw.min),
        max: (raw.count > 0).then_some(raw.max),
    })
}
