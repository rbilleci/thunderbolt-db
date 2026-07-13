//! Device-resident join-coordinate initialization and post-join filtering ownership.

use std::os::raw::c_void;

use super::{
    check_cuda, CudaJoinCoordinatesU32, CudaPredicateMaskI32, CudaResidentDeviceMemory,
    CudaResidentReadSource, CudaRuntimeProbeError,
};

impl CudaResidentDeviceMemory {
    /// Apply post-join WHERE masks to a coordinate relation without materializing tuples. For a real
    /// coordinate the referenced device mask decides SQL-WHERE truth; for an OUTER NULL pad the
    /// corresponding retained one-row device mask decides without an intermediate host verdict.
    pub fn filter_join_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        masks: &[Option<&CudaPredicateMaskI32>],
        pad_masks: &[Option<&CudaPredicateMaskI32>],
    ) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
        launch_cuda_filter_join_coordinates(self, coordinates, masks, pad_masks)
    }

    pub fn identity_join_coordinates(
        &self,
        row_count: u32,
        eligibility: Option<&CudaPredicateMaskI32>,
    ) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
        launch_cuda_identity_join_coordinates(self, row_count, eligibility)
    }
}

fn launch_cuda_identity_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    row_count: u32,
    eligibility: Option<&CudaPredicateMaskI32>,
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
.visible .entry gpu_db_identity_join_coordinates(
    .param .u32 rows, .param .u64 mask, .param .u64 out, .param .u64 cursor)
{
    .reg .pred %p<5>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    ld.param.u32 %r1, [rows];
    ld.param.u64 %rd1, [mask];
    ld.param.u64 %rd2, [out];
    ld.param.u64 %rd3, [cursor];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %nctaid.x;
    mad.lo.u32 %r6, %r3, %r4, %r2;
    mul.lo.u32 %r7, %r5, %r4;
LOOP:
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra DONE;
    mov.u64 %rd4, 18446744073709551615;
    setp.eq.u64 %p2, %rd1, %rd4;
    @%p2 bra KEEP;
    mul.wide.u32 %rd5, %r6, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u32 %r8, [%rd6];
    setp.eq.u32 %p3, %r8, 0;
    @%p3 bra NEXT;
KEEP:
    atom.global.add.u64 %rd7, [%rd3], 1;
    setp.eq.u64 %p4, %rd2, 0;
    @%p4 bra NEXT;
    mul.lo.u64 %rd8, %rd7, 4;
    add.u64 %rd9, %rd2, %rd8;
    st.global.u32 [%rd9], %r6;
NEXT:
    add.u32 %r6, %r6, %r7;
    bra LOOP;
DONE:
    ret;
}
"#;
    if eligibility.is_some_and(|mask| mask.row_count != row_count) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            row_count as usize,
        ));
    }
    if row_count == 0 {
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: 1,
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
    let cursor = primary.lease_device_buffer_owned(8)?;
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_identity_join_coordinates", &ptx)?;
    let run = |out: u64| -> Result<(), CudaRuntimeProbeError> {
        let mut a0 = row_count;
        let mut a1 = eligibility.map_or(u64::MAX, |mask| mask.mask.ptr);
        let mut a2 = out;
        let mut a3 = cursor.ptr;
        let mut args = [
            (&mut a0 as *mut u32).cast(),
            (&mut a1 as *mut u64).cast(),
            (&mut a2 as *mut u64).cast(),
            (&mut a3 as *mut u64).cast(),
        ];
        check_cuda(unsafe {
            launch(
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
        })
    };
    check_cuda(unsafe { memset(cursor.ptr, 0, 8) })?;
    run(0)?;
    let mut count = 0_u64;
    check_cuda(unsafe { dtoh((&mut count as *mut u64).cast(), cursor.ptr, 8) })?;
    if count == 0 {
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: 1,
            allocated_bytes: 0,
        });
    }
    let count_u32 =
        u32::try_from(count).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes = count_u32 as usize * 4;
    let output = primary.lease_device_buffer_owned(bytes)?;
    check_cuda(unsafe { memset(cursor.ptr, 0, 8) })?;
    run(output.ptr)?;
    Ok(CudaJoinCoordinatesU32 {
        coordinates: Some(output),
        row_count: count_u32,
        relation_count: 1,
        // The scalar compaction cursor is temporary and has already been released.
        allocated_bytes: bytes as u64,
    })
}

fn launch_cuda_filter_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    coordinates: &CudaJoinCoordinatesU32,
    masks: &[Option<&CudaPredicateMaskI32>],
    pad_masks: &[Option<&CudaPredicateMaskI32>],
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
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64
.visible .entry gpu_db_filter_join_coordinates(
    .param .u64 coords, .param .u32 rows, .param .u32 rels,
    .param .u64 desc, .param .u64 out, .param .u64 cursor)
{
    .reg .pred %p<12>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<40>;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [rels];
    ld.param.u64 %rd2, [desc];
    ld.param.u64 %rd3, [out];
    ld.param.u64 %rd4, [cursor];
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %nctaid.x;
    mad.lo.u32 %r7, %r4, %r5, %r3;
    mul.lo.u32 %r8, %r6, %r5;
ROW_LOOP:
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra DONE;
    mov.u32 %r9, 0;
REL_LOOP:
    setp.ge.u32 %p2, %r9, %r2;
    @%p2 bra KEEP;
    mul.lo.u32 %r10, %r7, %r2;
    add.u32 %r10, %r10, %r9;
    mul.wide.u32 %rd5, %r10, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u32 %r11, [%rd6];
    mul.wide.u32 %rd7, %r9, 16;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.u64 %rd9, [%rd8];
    ld.global.u64 %rd10, [%rd8+8];
    mov.u64 %rd11, 18446744073709551615;
    setp.eq.u32 %p3, %r11, 4294967295;
    @%p3 bra PAD_TEST;
    setp.eq.u64 %p4, %rd9, %rd11;
    @%p4 bra REL_NEXT;
    mul.wide.u32 %rd12, %r11, 4;
    add.u64 %rd13, %rd9, %rd12;
    ld.global.u32 %r12, [%rd13];
    setp.eq.u32 %p5, %r12, 0;
    @%p5 bra ROW_NEXT;
    bra REL_NEXT;
PAD_TEST:
    setp.eq.u64 %p6, %rd9, %rd11;
    @%p6 bra REL_NEXT;
    setp.eq.u64 %p7, %rd10, %rd11;
    @%p7 bra REL_NEXT;
    ld.global.u32 %r12, [%rd10];
    setp.eq.u32 %p7, %r12, 0;
    @%p7 bra ROW_NEXT;
REL_NEXT:
    add.u32 %r9, %r9, 1;
    bra REL_LOOP;
KEEP:
    atom.global.add.u64 %rd14, [%rd4], 1;
    setp.eq.u64 %p8, %rd3, 0;
    @%p8 bra ROW_NEXT;
    cvt.u64.u32 %rd15, %r2;
    mul.lo.u64 %rd16, %rd14, %rd15;
    mov.u32 %r13, 0;
COPY_LOOP:
    setp.ge.u32 %p9, %r13, %r2;
    @%p9 bra ROW_NEXT;
    mul.lo.u32 %r14, %r7, %r2;
    add.u32 %r14, %r14, %r13;
    mul.wide.u32 %rd17, %r14, 4;
    add.u64 %rd18, %rd1, %rd17;
    ld.global.u32 %r15, [%rd18];
    cvt.u64.u32 %rd19, %r13;
    add.u64 %rd20, %rd16, %rd19;
    mul.lo.u64 %rd20, %rd20, 4;
    add.u64 %rd20, %rd3, %rd20;
    st.global.u32 [%rd20], %r15;
    add.u32 %r13, %r13, 1;
    bra COPY_LOOP;
ROW_NEXT:
    add.u32 %r7, %r7, %r8;
    bra ROW_LOOP;
DONE:
    ret;
}
"#;
    if masks.len() != coordinates.relation_count as usize
        || pad_masks.len() != masks.len()
        || masks.iter().flatten().any(|mask| mask.row_count == 0)
        || pad_masks.iter().flatten().any(|mask| mask.row_count != 1)
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(masks.len()));
    }
    if coordinates.row_count == 0 {
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: coordinates.relation_count,
            allocated_bytes: 0,
        });
    }
    let source = coordinates
        .coordinates
        .as_ref()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    let primary = ctx.primary_arc();
    primary.set_current()?;
    let htod = unsafe {
        primary
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
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
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let descriptor: Vec<u64> = masks
        .iter()
        .zip(pad_masks)
        .flat_map(|(mask, pad)| {
            [
                mask.map_or(u64::MAX, |mask| mask.mask.ptr),
                pad.map_or(u64::MAX, |mask| mask.mask.ptr),
            ]
        })
        .collect();
    let desc_bytes = std::mem::size_of_val(descriptor.as_slice());
    let desc = primary.lease_device_buffer_owned(desc_bytes.max(1))?;
    check_cuda(unsafe { htod(desc.ptr, descriptor.as_ptr().cast(), desc_bytes) })?;
    let cursor = primary.lease_device_buffer_owned(8)?;
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_filter_join_coordinates", &ptx)?;
    const BLOCK: u32 = 256;
    let grid = coordinates.row_count.div_ceil(BLOCK).clamp(1, 65_535);
    let run = |out: u64| -> Result<(), CudaRuntimeProbeError> {
        let mut a0 = source.ptr;
        let mut a1 = coordinates.row_count;
        let mut a2 = coordinates.relation_count;
        let mut a3 = desc.ptr;
        let mut a4 = out;
        let mut a5 = cursor.ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast(),
            (&mut a1 as *mut u32).cast(),
            (&mut a2 as *mut u32).cast(),
            (&mut a3 as *mut u64).cast(),
            (&mut a4 as *mut u64).cast(),
            (&mut a5 as *mut u64).cast(),
        ];
        check_cuda(unsafe {
            launch(
                function,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    };
    check_cuda(unsafe { memset(cursor.ptr, 0, 8) })?;
    run(0)?;
    let mut total = 0_u64;
    check_cuda(unsafe { dtoh((&mut total as *mut u64).cast(), cursor.ptr, 8) })?;
    let total_u32 =
        u32::try_from(total).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if total == 0 {
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: coordinates.relation_count,
            allocated_bytes: 0,
        });
    }
    let output_bytes = total
        .checked_mul(u64::from(coordinates.relation_count))
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let output = primary.lease_device_buffer_owned(output_bytes)?;
    check_cuda(unsafe { memset(cursor.ptr, 0, 8) })?;
    run(output.ptr)?;
    Ok(CudaJoinCoordinatesU32 {
        coordinates: Some(output),
        row_count: total_u32,
        relation_count: coordinates.relation_count,
        allocated_bytes: output_bytes as u64 + desc_bytes as u64 + 8,
    })
}
