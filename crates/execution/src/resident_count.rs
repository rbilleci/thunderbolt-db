use std::ffi::c_void;

#[cfg(test)]
use super::{
    check_cuda, launch_with_optional_cuda_event_timing, CudaDeviceAllocationGuard, CudaModuleGuard,
};
use super::{
    launch_on_pooled_stream, CudaI32Comparison, CudaResidentDeviceMemory, CudaRuntimeProbeError,
};

/// Map a column's optional NULL validity-bitmap byte offset to the `u64` the resident count kernels
/// expect (M3 — doc 21): `None` ⇒ the sentinel `u64::MAX` ("no bitmap, every row valid"); `Some(off)`
/// ⇒ `off`, after bounds-checking the bitmap region (`off + ceil(row_count/32) * 4`) fits the
/// allocation so the kernel's `ld.global.u32` can never read out of bounds.
pub(super) fn validity_bitmap_kernel_arg(
    null_bitmap_offset: Option<u64>,
    row_count: u64,
    resident: &CudaResidentDeviceMemory,
) -> Result<u64, CudaRuntimeProbeError> {
    match null_bitmap_offset {
        None => Ok(u64::MAX),
        Some(off) => {
            let end = row_count
                .div_ceil(32)
                .checked_mul(std::mem::size_of::<u32>() as u64)
                .and_then(|bitmap_bytes| off.checked_add(bitmap_bytes))
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if end > resident.metadata().allocated_bytes {
                return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
            }
            Ok(off)
        }
    }
}

/// Single-thread `(1,1,1)` serial filtered-count — retained ONLY as the A/B baseline for
/// the P2-M2 parallel-scan spike. The production route is the parallel
/// `launch_cuda_resident_i32_equal_count`.
#[cfg(test)]
pub(super) fn launch_cuda_resident_i32_equal_count_serial(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    // M3 (doc 21): `Some(off)` = the column's NULL validity bitmap byte offset (1 = valid, 0 = NULL);
    // `None` = no bitmap ⇒ all rows valid. NULL rows never match the needle (3VL).
    null_bitmap_offset: Option<u64>,
) -> Result<u64, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_equal_count(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .s32 needle,
    .param .u64 out_ptr,
    .param .u64 null_bitmap_offset
)
{
    .reg .pred %p_done;
    .reg .pred %p_match;
    .reg .pred %p_no_bitmap;
    .reg .pred %p_valid;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %addr;
    .reg .u64 %matches;
    .reg .u64 %null_off;
    .reg .u64 %sentinel;
    .reg .u64 %word_byte;
    .reg .u64 %bitmap_addr;
    .reg .u32 %bitmap_word;
    .reg .u32 %bit_pos;
    .reg .u32 %valid_bit;
    .reg .s32 %needle;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %needle, [needle];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %null_off, [null_bitmap_offset];

    add.u64 %base, %resident, %offset;
    mov.u64 %idx, 0;
    mov.u64 %matches, 0;
    mov.u64 %sentinel, 0xFFFFFFFFFFFFFFFF;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %addr, %idx, 4;
    add.u64 %addr, %base, %addr;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle;
    @!%p_match bra next;
    // The value matches the needle; in 3VL a NULL operand can never match, so exclude NULL rows.
    // null_off == 0xFFFF... (sentinel) means the column has no validity bitmap => every row valid.
    setp.eq.u64 %p_no_bitmap, %null_off, %sentinel;
    @%p_no_bitmap bra count;
    shr.u64 %word_byte, %idx, 5;          // idx / 32 (the validity word index)
    mul.lo.u64 %word_byte, %word_byte, 4; // * 4 bytes per u32 word
    add.u64 %bitmap_addr, %resident, %null_off;
    add.u64 %bitmap_addr, %bitmap_addr, %word_byte;
    ld.global.u32 %bitmap_word, [%bitmap_addr];
    cvt.u32.u64 %bit_pos, %idx;
    and.b32 %bit_pos, %bit_pos, 31;       // idx % 32
    bfe.u32 %valid_bit, %bitmap_word, %bit_pos, 1;
    setp.eq.u32 %p_valid, %valid_bit, 1;  // 1 = valid/present, 0 = NULL
    @!%p_valid bra next;                   // NULL => does not match, skip

count:
    add.u64 %matches, %matches, 1;

next:
    add.u64 %idx, %idx, 1;
    bra loop;

done:
    st.global.u64 [%out], %matches;
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

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, std::mem::size_of::<u64>()) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_equal_count".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut needle_arg = needle;
    let mut output_arg = allocation_guard.ptr;
    let mut null_bitmap_arg = validity_bitmap_kernel_arg(null_bitmap_offset, row_count, resident)?;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut null_bitmap_arg as *mut u64).cast::<c_void>(),
    ];
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut output = 0_u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output as *mut u64).cast::<c_void>(),
            allocation_guard.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;
    drop(module_guard);
    drop(allocation_guard);
    Ok(output)
}

/// Filtered-count over an i32 column: a real grid/block strided scan + global reduction
/// (Phase 2). Each thread grid-strides over the column counting matches into a register,
/// then `red.global.add.u64` accumulates into a zeroed scratch. Runs on the P2-M1 substrate
/// (cached module — no per-launch JIT — and a pooled private stream). This replaced the
/// original single-thread `(1,1,1)` serial loop (kept as `_serial`, test-only, for the A/B);
/// ~60× faster on a 16M-row scan (run report `2026-06-14-p2-m2-...`).
pub(super) fn launch_cuda_resident_i32_equal_count(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    // M3 (doc 21): `Some(off)` = the column's NULL validity bitmap (1 = valid, 0 = NULL); `None` = no
    // bitmap ⇒ all rows valid. A NULL operand never matches the needle (three-valued logic).
    null_bitmap_offset: Option<u64>,
) -> Result<u64, CudaRuntimeProbeError> {
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

    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_resident_i32_equal_count_parallel(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .s32 needle,
    .param .u64 out_ptr,
    .param .u64 null_bitmap_offset
)
{
    // Per-block reduction scratch: one u32 partial slot per thread (block <= 1024 threads).
    .shared .align 4 .b32 s_part[1024];

    .reg .pred %p_done;
    .reg .pred %p_match;
    .reg .pred %p_no_bitmap;
    .reg .pred %p_valid;
    .reg .pred %p_active;
    .reg .pred %p_isthr0;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %gdim;
    .reg .u32 %tmp32;
    .reg .u32 %part32;
    .reg .u32 %rstride;
    .reg .u32 %peer;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %addr;
    .reg .u64 %off_bytes;
    .reg .u64 %matches;
    .reg .u64 %null_off;
    .reg .u64 %sentinel;
    .reg .u64 %word_byte;
    .reg .u64 %bitmap_addr;
    .reg .u64 %sh_base;
    .reg .u64 %sh_self;
    .reg .u64 %sh_peer;
    .reg .u64 %block_tot;
    .reg .u32 %bitmap_word;
    .reg .u32 %bit_pos;
    .reg .u32 %valid_bit;
    .reg .s32 %needle;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %needle, [needle];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %null_off, [null_bitmap_offset];

    add.u64 %base, %resident, %offset;
    mov.u64 %sentinel, 0xFFFFFFFFFFFFFFFF;

    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %tmp32, %bid, %bdim, %lane;
    cvt.u64.u32 %idx, %tmp32;
    mul.lo.u32 %tmp32, %gdim, %bdim;
    cvt.u64.u32 %stride, %tmp32;

    mov.u64 %matches, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %off_bytes, %idx, 4;
    add.u64 %addr, %base, %off_bytes;
    ld.global.s32 %r_value, [%addr];
    setp.eq.s32 %p_match, %r_value, %needle;
    @!%p_match bra next;
    // 3VL: a NULL operand never matches. null_off == sentinel => no bitmap => every row valid.
    setp.eq.u64 %p_no_bitmap, %null_off, %sentinel;
    @%p_no_bitmap bra count;
    shr.u64 %word_byte, %idx, 5;          // idx / 32
    mul.lo.u64 %word_byte, %word_byte, 4; // * 4 bytes/word
    add.u64 %bitmap_addr, %resident, %null_off;
    add.u64 %bitmap_addr, %bitmap_addr, %word_byte;
    ld.global.u32 %bitmap_word, [%bitmap_addr];
    cvt.u32.u64 %bit_pos, %idx;
    and.b32 %bit_pos, %bit_pos, 31;       // idx % 32
    bfe.u32 %valid_bit, %bitmap_word, %bit_pos, 1;
    setp.eq.u32 %p_valid, %valid_bit, 1;  // 1 = valid, 0 = NULL
    @!%p_valid bra next;                   // NULL => skip

count:
    add.u64 %matches, %matches, 1;

next:
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    // ---- per-block reduction: sum every thread's %matches partial, then ONE atomic per block ----
    // The grid is clamped to a saturating constant, so each thread's partial = ceil(rows/(grid*BLOCK))
    // rows of matches, which fits u32 for any plausible row_count (the per-block sum likewise:
    // BLOCK=256 threads * partial << 2^32). The block total is widened to u64 for the single global
    // red.add so the global COUNT stays exact and byte-identical to the old per-thread accumulation
    // (sum is associative/commutative - only the grouping of the adds changed).
    //
    // This is a barrier-synchronized SHARED-MEMORY tree reduction, NOT a warp shuffle: the
    // grid-stride loop exits per-thread (idx >= rows), so lanes within a warp execute a DIFFERENT
    // number of iterations and are NOT guaranteed converged at done: -- a shfl.sync here would
    // silently drop partials. bar.sync synchronizes the WHOLE block regardless of how many
    // iterations each thread ran, and every barrier below is on the straight-line path (outside the
    // @!%p_active guard), so all threads reach it. %lane holds %tid.x (the in-block thread id).
    cvt.u32.u64 %part32, %matches;
    mov.u64 %sh_base, s_part;
    mul.wide.u32 %sh_self, %lane, 4;
    add.u64 %sh_self, %sh_base, %sh_self;
    st.shared.u32 [%sh_self], %part32;
    bar.sync 0;

    // tree reduce: for rstride = bdim/2, bdim/4, ..., 1: s_part[t] += s_part[t + rstride] for t < rstride.
    shr.u32 %rstride, %bdim, 1;
red_loop:
    setp.eq.u32 %p_done, %rstride, 0;
    @%p_done bra red_done;
    setp.lt.u32 %p_active, %lane, %rstride;
    @!%p_active bra red_skip;
    add.u32 %peer, %lane, %rstride;
    mul.wide.u32 %sh_peer, %peer, 4;
    add.u64 %sh_peer, %sh_base, %sh_peer;
    ld.shared.u32 %tmp32, [%sh_peer];
    ld.shared.u32 %part32, [%sh_self];
    add.u32 %part32, %part32, %tmp32;
    st.shared.u32 [%sh_self], %part32;
red_skip:
    bar.sync 0;
    shr.u32 %rstride, %rstride, 1;
    bra red_loop;
red_done:
    // thread 0 holds the block total at s_part[0]; widen to u64 and do the single global red.add.
    setp.eq.u32 %p_isthr0, %lane, 0;
    @!%p_isthr0 bra block_done;
    ld.shared.u32 %part32, [%sh_base];
    cvt.u64.u32 %block_tot, %part32;
    red.global.add.u64 [%out], %block_tot;
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
        .cached_function(c"gpu_db_resident_i32_equal_count_parallel", &ptx)?;
    let null_bitmap_kernel_arg =
        validity_bitmap_kernel_arg(null_bitmap_offset, row_count, resident)?;

    // SATURATING grid (not one-thread-per-row): clamp to a moderate constant that fills the GPU.
    // The grid-stride loop covers any row_count regardless of grid size, so clamping is
    // correctness-safe — each thread now strides over `ceil(row_count/(grid*BLOCK))` rows = a REAL
    // partial, and the in-kernel per-block reduction means each block does ONE global atomic. Before:
    // `clamp(1, 65_535)` = one thread per row => ~N serialized `red.global.add` on a single address
    // (the ~60x-below-roofline bug). The kernel computes `stride = gridDim * blockDim` in u32: with
    // grid ≤ 4096 and BLOCK = 256 the product is ≤ ~1M, well within u32.
    const BLOCK: u32 = 256;
    let grid: u32 = row_count.div_ceil(u64::from(BLOCK)).clamp(1, 4096) as u32;

    let mut output_bytes = [0_u8; std::mem::size_of::<u64>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        // Zero the scratch on the stream (the kernel red-adds into it), ordered before the
        // kernel launch on the same stream.
        let memset_rc =
            unsafe { cu_memset_d8_async(output_ptr, 0, std::mem::size_of::<u64>(), stream) };
        if memset_rc != 0 {
            return memset_rc;
        }
        let mut resident_arg = resident.device_ptr();
        let mut offset_arg = byte_offset;
        let mut rows_arg = row_count;
        let mut needle_arg = needle;
        let mut output_arg = output_ptr;
        let mut null_bitmap_arg = null_bitmap_kernel_arg;
        let mut args = [
            (&mut resident_arg as *mut u64).cast::<c_void>(),
            (&mut offset_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut needle_arg as *mut i32).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut null_bitmap_arg as *mut u64).cast::<c_void>(),
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

    Ok(u64::from_le_bytes(output_bytes))
}

pub(super) fn launch_cuda_resident_i32_compare_count(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    comparison: CudaI32Comparison,
) -> Result<u64, CudaRuntimeProbeError> {
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

    // P2-M2 parallelization of the comparison-count scan: this is the serial
    // `gpu_db_resident_i32_compare_count` predicate (4-way `<`/`<=`/`>`/`>=` against `needle`)
    // dropped into the proven `gpu_db_resident_i32_equal_count_parallel` grid-stride skeleton —
    // each thread accumulates its partial match count over a `gridDim*blockDim`-strided slice and
    // `red.global.add.u64`s it into the single output counter. `target sm_60` is required for the
    // reduction op (the old serial kernel targeted sm_30). Launched on a pooled private stream via
    // `launch_on_pooled_stream` (cached module — no per-call re-JIT, pooled scratch — no per-call
    // `cuMemAlloc`/`cuMemFree`, async memset + event timing + centralized error drain).
    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_resident_i32_compare_count_parallel(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .s32 needle,
    .param .u32 comparison,
    .param .u64 out_ptr
)
{
    // Per-block reduction scratch: one u32 partial slot per thread (block <= 1024 threads).
    .shared .align 4 .b32 s_part[1024];

    .reg .pred %p_done;
    .reg .pred %p_lt;
    .reg .pred %p_lte;
    .reg .pred %p_gt;
    .reg .pred %p_gte;
    .reg .pred %p_code_lt;
    .reg .pred %p_code_lte;
    .reg .pred %p_code_gt;
    .reg .pred %p_code_gte;
    .reg .pred %p_match;
    .reg .pred %p_active;
    .reg .pred %p_isthr0;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %gdim;
    .reg .u32 %tmp32;
    .reg .u32 %part32;
    .reg .u32 %rstride;
    .reg .u32 %peer;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %addr;
    .reg .u64 %off_bytes;
    .reg .u64 %matches;
    .reg .u64 %sh_base;
    .reg .u64 %sh_self;
    .reg .u64 %sh_peer;
    .reg .u64 %block_tot;
    .reg .u32 %comparison;
    .reg .s32 %needle;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %needle, [needle];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %out, [out_ptr];

    add.u64 %base, %resident, %offset;

    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %tmp32, %bid, %bdim, %lane;
    cvt.u64.u32 %idx, %tmp32;
    mul.lo.u32 %tmp32, %gdim, %bdim;
    cvt.u64.u32 %stride, %tmp32;

    mov.u64 %matches, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %off_bytes, %idx, 4;
    add.u64 %addr, %base, %off_bytes;
    ld.global.s32 %r_value, [%addr];
    setp.lt.s32 %p_lt, %r_value, %needle;
    setp.le.s32 %p_lte, %r_value, %needle;
    setp.gt.s32 %p_gt, %r_value, %needle;
    setp.ge.s32 %p_gte, %r_value, %needle;
    setp.eq.u32 %p_code_lt, %comparison, 1;
    setp.eq.u32 %p_code_lte, %comparison, 2;
    setp.eq.u32 %p_code_gt, %comparison, 3;
    setp.eq.u32 %p_code_gte, %comparison, 4;
    mov.pred %p_match, 0;
    and.pred %p_lt, %p_lt, %p_code_lt;
    or.pred %p_match, %p_match, %p_lt;
    and.pred %p_lte, %p_lte, %p_code_lte;
    or.pred %p_match, %p_match, %p_lte;
    and.pred %p_gt, %p_gt, %p_code_gt;
    or.pred %p_match, %p_match, %p_gt;
    and.pred %p_gte, %p_gte, %p_code_gte;
    or.pred %p_match, %p_match, %p_gte;
    @!%p_match bra next;
    add.u64 %matches, %matches, 1;

next:
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    // ---- per-block reduction: sum every thread's %matches partial, then ONE atomic per block ----
    // The grid is clamped to a saturating constant, so each thread's partial = ceil(rows/(grid*BLOCK))
    // rows of matches, which fits u32 for any plausible row_count (the per-block sum likewise:
    // BLOCK=256 threads * partial << 2^32). The block total is widened to u64 for the single global
    // red.add so the global COUNT stays exact and byte-identical to the old per-thread accumulation
    // (sum is associative/commutative - only the grouping of the adds changed).
    //
    // This is a barrier-synchronized SHARED-MEMORY tree reduction, NOT a warp shuffle: the
    // grid-stride loop exits per-thread (idx >= rows), so lanes within a warp execute a DIFFERENT
    // number of iterations and are NOT guaranteed converged at done: -- a shfl.sync here would
    // silently drop partials. bar.sync synchronizes the WHOLE block regardless of how many
    // iterations each thread ran, and every barrier below is on the straight-line path (outside the
    // @!%p_active guard), so all threads reach it. %lane holds %tid.x (the in-block thread id).
    cvt.u32.u64 %part32, %matches;
    mov.u64 %sh_base, s_part;
    mul.wide.u32 %sh_self, %lane, 4;
    add.u64 %sh_self, %sh_base, %sh_self;
    st.shared.u32 [%sh_self], %part32;
    bar.sync 0;

    // tree reduce: for rstride = bdim/2, bdim/4, ..., 1: s_part[t] += s_part[t + rstride] for t < rstride.
    shr.u32 %rstride, %bdim, 1;
red_loop:
    setp.eq.u32 %p_done, %rstride, 0;
    @%p_done bra red_done;
    setp.lt.u32 %p_active, %lane, %rstride;
    @!%p_active bra red_skip;
    add.u32 %peer, %lane, %rstride;
    mul.wide.u32 %sh_peer, %peer, 4;
    add.u64 %sh_peer, %sh_base, %sh_peer;
    ld.shared.u32 %tmp32, [%sh_peer];
    ld.shared.u32 %part32, [%sh_self];
    add.u32 %part32, %part32, %tmp32;
    st.shared.u32 [%sh_self], %part32;
red_skip:
    bar.sync 0;
    shr.u32 %rstride, %rstride, 1;
    bra red_loop;
red_done:
    // thread 0 holds the block total at s_part[0]; widen to u64 and do the single global red.add.
    setp.eq.u32 %p_isthr0, %lane, 0;
    @!%p_isthr0 bra block_done;
    ld.shared.u32 %part32, [%sh_base];
    cvt.u64.u32 %block_tot, %part32;
    red.global.add.u64 [%out], %block_tot;
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
        .cached_function(c"gpu_db_resident_i32_compare_count_parallel", &ptx)?;

    // SATURATING grid (not one-thread-per-row): clamp to a moderate constant that fills the GPU.
    // The grid-stride loop covers any row_count regardless of grid size, so clamping is
    // correctness-safe — each thread now strides over `ceil(row_count/(grid*BLOCK))` rows = a REAL
    // partial, and the in-kernel per-block reduction means each block does ONE global atomic. Before:
    // `clamp(1, 65_535)` = one thread per row => ~N serialized `red.global.add` on a single address
    // (the ~60x-below-roofline bug). The kernel computes `stride = gridDim * blockDim` in u32: with
    // grid ≤ 4096 and BLOCK = 256 the product is ≤ ~1M, well within u32.
    const BLOCK: u32 = 256;
    let grid: u32 = row_count.div_ceil(u64::from(BLOCK)).clamp(1, 4096) as u32;

    let mut output_bytes = [0_u8; std::mem::size_of::<u64>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        // Zero the scratch on the stream (the kernel red-adds into it), ordered before the
        // kernel launch on the same stream.
        let memset_rc =
            unsafe { cu_memset_d8_async(output_ptr, 0, std::mem::size_of::<u64>(), stream) };
        if memset_rc != 0 {
            return memset_rc;
        }
        let mut resident_arg = resident.device_ptr();
        let mut offset_arg = byte_offset;
        let mut rows_arg = row_count;
        let mut needle_arg = needle;
        let mut comparison_arg = comparison.code();
        let mut output_arg = output_ptr;
        let mut args = [
            (&mut resident_arg as *mut u64).cast::<c_void>(),
            (&mut offset_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut needle_arg as *mut i32).cast::<c_void>(),
            (&mut comparison_arg as *mut u32).cast::<c_void>(),
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

    Ok(u64::from_le_bytes(output_bytes))
}

/// Single-thread `(1,1,1)` serial comparison-count — retained ONLY (under `#[cfg(test)]`) as the
/// on-GPU A/B parity + perf baseline for the P2-M2 parallel compare-count migration. The production
/// route is the parallel `launch_cuda_resident_i32_compare_count`; this is the previously-shipped
/// serial kernel kept as a device-side reference oracle (no CPU operator re-implementation).
#[cfg(test)]
pub(super) fn launch_cuda_resident_i32_compare_count_serial(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    comparison: CudaI32Comparison,
) -> Result<u64, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_compare_count(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .s32 needle,
    .param .u32 comparison,
    .param .u64 out_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_lt;
    .reg .pred %p_lte;
    .reg .pred %p_gt;
    .reg .pred %p_gte;
    .reg .pred %p_code_lt;
    .reg .pred %p_code_lte;
    .reg .pred %p_code_gt;
    .reg .pred %p_code_gte;
    .reg .pred %p_match;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %addr;
    .reg .u64 %matches;
    .reg .u32 %comparison;
    .reg .s32 %needle;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %needle, [needle];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %out, [out_ptr];

    add.u64 %base, %resident, %offset;
    mov.u64 %idx, 0;
    mov.u64 %matches, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %addr, %idx, 4;
    add.u64 %addr, %base, %addr;
    ld.global.s32 %r_value, [%addr];
    setp.lt.s32 %p_lt, %r_value, %needle;
    setp.le.s32 %p_lte, %r_value, %needle;
    setp.gt.s32 %p_gt, %r_value, %needle;
    setp.ge.s32 %p_gte, %r_value, %needle;
    setp.eq.u32 %p_code_lt, %comparison, 1;
    setp.eq.u32 %p_code_lte, %comparison, 2;
    setp.eq.u32 %p_code_gt, %comparison, 3;
    setp.eq.u32 %p_code_gte, %comparison, 4;
    mov.pred %p_match, 0;
    and.pred %p_lt, %p_lt, %p_code_lt;
    or.pred %p_match, %p_match, %p_lt;
    and.pred %p_lte, %p_lte, %p_code_lte;
    or.pred %p_match, %p_match, %p_lte;
    and.pred %p_gt, %p_gt, %p_code_gt;
    or.pred %p_match, %p_match, %p_gt;
    and.pred %p_gte, %p_gte, %p_code_gte;
    or.pred %p_match, %p_match, %p_gte;
    @!%p_match bra next;
    add.u64 %matches, %matches, 1;

next:
    add.u64 %idx, %idx, 1;
    bra loop;

done:
    st.global.u64 [%out], %matches;
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

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, std::mem::size_of::<u64>()) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_compare_count".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut needle_arg = needle;
    let mut comparison_arg = comparison.code();
    let mut output_arg = allocation_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
    ];
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut output = 0_u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output as *mut u64).cast::<c_void>(),
            allocation_guard.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;
    drop(module_guard);
    drop(allocation_guard);
    Ok(output)
}
