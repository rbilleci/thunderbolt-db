use crate::{launch_on_pooled_stream, CudaResidentDeviceMemory, CudaRuntimeProbeError};
use std::ffi::c_void;

/// Reduce one resident generation's MVCC visibility lanes to a scalar row count. The payload is
/// used as the launch/context owner; `deleted_by` and `created_by` may be independent resident
/// allocations or byte ranges inside that same payload. A missing bound means every row satisfies
/// that side of the visibility predicate.
pub(super) fn launch_cuda_resident_visible_count(
    resident: &CudaResidentDeviceMemory,
    row_count: u64,
    read_txn_id: i64,
    deleted_by: Option<(&CudaResidentDeviceMemory, u64)>,
    created_by: Option<(&CudaResidentDeviceMemory, u64)>,
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

    if row_count > u64::from(u32::MAX) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    let version_bytes = row_count
        .checked_mul(std::mem::size_of::<i64>() as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let version_ptr = |region: Option<(&CudaResidentDeviceMemory, u64)>| {
        let Some((memory, offset)) = region else {
            return Ok(0_u64);
        };
        if memory.metadata().gpu_id != resident.metadata().gpu_id
            || !std::ptr::eq(memory.primary(), resident.primary())
            || offset % std::mem::align_of::<i64>() as u64 != 0
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                row_count as usize,
            ));
        }
        let end = offset
            .checked_add(version_bytes)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if end > memory.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
        }
        memory
            .device_ptr()
            .checked_add(offset)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))
    };
    let deleted_ptr = version_ptr(deleted_by)?;
    let created_ptr = version_ptr(created_by)?;
    if row_count == 0 {
        return Ok(0);
    }

    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_resident_visible_count(
    .param .u64 deleted_ptr,
    .param .u64 created_ptr,
    .param .u64 row_count,
    .param .s64 read_txn_id,
    .param .u64 out_ptr
)
{
    .shared .align 4 .b32 s_part[1024];

    .reg .pred %p_done;
    .reg .pred %p_absent;
    .reg .pred %p_visible;
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
    .reg .u64 %deleted;
    .reg .u64 %created;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %off_bytes;
    .reg .u64 %addr;
    .reg .u64 %matches;
    .reg .u64 %sh_base;
    .reg .u64 %sh_self;
    .reg .u64 %sh_peer;
    .reg .u64 %block_tot;
    .reg .s64 %bound;
    .reg .s64 %stamp;

    ld.param.u64 %deleted, [deleted_ptr];
    ld.param.u64 %created, [created_ptr];
    ld.param.u64 %rows, [row_count];
    ld.param.s64 %bound, [read_txn_id];
    ld.param.u64 %out, [out_ptr];

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
    mul.lo.u64 %off_bytes, %idx, 8;

    setp.eq.u64 %p_absent, %deleted, 0;
    @%p_absent bra created_check;
    add.u64 %addr, %deleted, %off_bytes;
    ld.global.s64 %stamp, [%addr];
    setp.gt.s64 %p_visible, %stamp, %bound;
    @!%p_visible bra next;

created_check:
    setp.eq.u64 %p_absent, %created, 0;
    @%p_absent bra count;
    add.u64 %addr, %created, %off_bytes;
    ld.global.s64 %stamp, [%addr];
    setp.le.s64 %p_visible, %stamp, %bound;
    @!%p_visible bra next;

count:
    add.u64 %matches, %matches, 1;
next:
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    cvt.u32.u64 %part32, %matches;
    mov.u64 %sh_base, s_part;
    mul.wide.u32 %sh_self, %lane, 4;
    add.u64 %sh_self, %sh_base, %sh_self;
    st.shared.u32 [%sh_self], %part32;
    bar.sync 0;

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
    setp.eq.u32 %p_isthr0, %lane, 0;
    @!%p_isthr0 bra block_done;
    ld.shared.u32 %part32, [%sh_base];
    cvt.u64.u32 %block_tot, %part32;
    red.global.add.u64 [%out], %block_tot;
block_done:
    ret;
}
"#;

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
    // Module load is context-scoped and precedes the self-binding launch helper on a cache miss.
    // Bind here so this public reduction is safe on its first call from any worker thread.
    resident.primary().set_current()?;
    let function = resident
        .primary()
        .cached_function(c"gpu_db_resident_visible_count", &ptx)?;

    const BLOCK: u32 = 256;
    let grid = row_count.div_ceil(u64::from(BLOCK)).clamp(1, 4096) as u32;
    let mut output_bytes = [0_u8; std::mem::size_of::<u64>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        let memset_rc =
            unsafe { cu_memset_d8_async(output_ptr, 0, std::mem::size_of::<u64>(), stream) };
        if memset_rc != 0 {
            return memset_rc;
        }
        let mut deleted_arg = deleted_ptr;
        let mut created_arg = created_ptr;
        let mut rows_arg = row_count;
        let mut boundary_arg = read_txn_id;
        let mut output_arg = output_ptr;
        let mut args = [
            (&mut deleted_arg as *mut u64).cast::<c_void>(),
            (&mut created_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
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
    Ok(u64::from_le_bytes(output_bytes))
}
