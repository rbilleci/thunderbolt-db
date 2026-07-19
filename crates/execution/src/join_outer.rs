//! GPU OUTER-join match marking and unmatched-coordinate completion ownership.

use std::os::raw::c_void;

use super::{
    check_cuda, launch_cuda_resident_device_memory, CudaDeviceMemoryProof, CudaJoinCoordinatesU32,
    CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError,
};

impl CudaResidentDeviceMemory {
    pub fn create_match_bitmap_u32(
        &self,
        row_count: u32,
    ) -> Result<CudaMatchBitmapU32, CudaRuntimeProbeError> {
        if row_count == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let byte_len = row_count as usize * std::mem::size_of::<u32>();
        let allocation =
            launch_cuda_resident_device_memory(self.metadata.gpu_id, &vec![0_u8; byte_len])?;
        Ok(CudaMatchBitmapU32 {
            marks: CudaResidentDeviceMemory::from_raw_parts(
                CudaDeviceMemoryProof {
                    gpu_id: self.metadata.gpu_id,
                    device_name: self.metadata.device_name.clone(),
                    allocated_bytes: byte_len as u64,
                    copied_bytes: byte_len as u64,
                    retained: true,
                },
                allocation.device_ptr,
                allocation.primary,
            ),
            row_count,
        })
    }
}

/// Bounded device-resident match state for streaming OUTER completion. Pair kernels may produce
/// duplicate coordinates (N:N); atomic marking collapses them without any host membership verdict.
pub struct CudaMatchBitmapU32 {
    pub(super) marks: CudaResidentDeviceMemory,
    pub(super) row_count: u32,
}

impl CudaMatchBitmapU32 {
    /// Mark one coordinate column from a retained device relation. Pads are skipped. No coordinate
    /// vector crosses D2H/H2D, and N:N duplicates collapse through the same atomic bitmap update.
    pub fn mark_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        relation: u32,
    ) -> Result<(), CudaRuntimeProbeError> {
        launch_cuda_mark_coordinate_column_u32(&self.marks, self.row_count, coordinates, relation)
    }

    /// Compact this bitmap's complement directly into a padded coordinate relation. `real_relation`
    /// receives the unmatched row index; every other relation receives the OUTER NULL sentinel.
    pub fn unmatched_coordinates(
        &self,
        relation_count: u32,
        real_relation: u32,
    ) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
        launch_cuda_unmatched_coordinate_relation(
            &self.marks,
            self.row_count,
            relation_count,
            real_relation,
        )
    }

    /// Extend each unmatched accumulated tuple with one OUTER pad. The accumulated coordinates are
    /// copied D2D; only the compacted cardinality scalar crosses to the scheduler.
    pub fn unmatched_extended_coordinates(
        &self,
        accumulated: &CudaJoinCoordinatesU32,
    ) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
        launch_cuda_unmatched_extended_coordinates(&self.marks, self.row_count, accumulated)
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.marks.metadata().allocated_bytes
    }
}

fn launch_cuda_unmatched_coordinate_relation(
    marks: &CudaResidentDeviceMemory,
    row_count: u32,
    relation_count: u32,
    real_relation: u32,
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
    const PTX:&[u8]=br#"
.version 6.0
.target sm_60
.address_size 64
.visible .entry gpu_db_unmatched_coordinate_relation(
 .param .u64 marks,.param .u32 rows,.param .u32 rels,.param .u32 real_rel,
 .param .u64 out,.param .u64 cursor)
{
 .reg .pred %p<5>; .reg .b32 %r<24>; .reg .b64 %rd<24>;
 ld.param.u64 %rd1,[marks];ld.param.u32 %r1,[rows];ld.param.u32 %r2,[rels];ld.param.u32 %r3,[real_rel];ld.param.u64 %rd2,[out];ld.param.u64 %rd3,[cursor];
 mov.u32 %r4,0;
ROW_LOOP:setp.ge.u32 %p1,%r4,%r1;@%p1 bra DONE;mul.wide.u32 %rd4,%r4,4;add.u64 %rd5,%rd1,%rd4;ld.global.u32 %r5,[%rd5];setp.ne.u32 %p2,%r5,0;@%p2 bra NEXT;
 atom.global.add.u64 %rd6,[%rd3],1;setp.eq.u64 %p3,%rd2,0;@%p3 bra NEXT;cvt.u64.u32 %rd7,%r2;mul.lo.u64 %rd8,%rd6,%rd7;mov.u32 %r6,0;
COL_LOOP:setp.ge.u32 %p4,%r6,%r2;@%p4 bra NEXT;setp.eq.u32 %p3,%r6,%r3;selp.u32 %r7,%r4,4294967295,%p3;cvt.u64.u32 %rd9,%r6;add.u64 %rd10,%rd8,%rd9;mul.lo.u64 %rd10,%rd10,4;add.u64 %rd10,%rd2,%rd10;st.global.u32 [%rd10],%r7;add.u32 %r6,%r6,1;bra COL_LOOP;
NEXT:add.u32 %r4,%r4,1;bra ROW_LOOP;DONE:ret;
}
"#;
    if relation_count == 0 || real_relation >= relation_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            real_relation as usize,
        ));
    }
    let primary = marks.primary_arc();
    primary.set_current()?;
    let cursor = primary.lease_device_buffer_owned(8)?;
    let memset = unsafe {
        primary
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let dtoh = unsafe {
        primary
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_unmatched_coordinate_relation", &ptx)?;
    let run = |out: u64| -> Result<(), CudaRuntimeProbeError> {
        let mut a0 = marks.device_ptr;
        let mut a1 = row_count;
        let mut a2 = relation_count;
        let mut a3 = real_relation;
        let mut a4 = out;
        let mut a5 = cursor.ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast(),
            (&mut a1 as *mut u32).cast(),
            (&mut a2 as *mut u32).cast(),
            (&mut a3 as *mut u32).cast(),
            (&mut a4 as *mut u64).cast(),
            (&mut a5 as *mut u64).cast(),
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
    let count =
        u32::try_from(count).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if count == 0 {
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count,
            relation_row_counts: (0..relation_count)
                .map(|relation| u32::from(relation == real_relation) * row_count)
                .collect(),
            allocated_bytes: 0,
        });
    }
    let bytes = count as usize * relation_count as usize * 4;
    let output = primary.lease_device_buffer_owned(bytes)?;
    check_cuda(unsafe { memset(cursor.ptr, 0, 8) })?;
    run(output.ptr)?;
    Ok(CudaJoinCoordinatesU32 {
        coordinates: Some(output),
        row_count: count,
        relation_count,
        relation_row_counts: (0..relation_count)
            .map(|relation| u32::from(relation == real_relation) * row_count)
            .collect(),
        allocated_bytes: bytes as u64 + 8,
    })
}

fn launch_cuda_unmatched_extended_coordinates(
    marks: &CudaResidentDeviceMemory,
    row_count: u32,
    accumulated: &CudaJoinCoordinatesU32,
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
.visible .entry gpu_db_unmatched_extended_coordinates(
    .param .u64 marks, .param .u64 coords, .param .u32 rows, .param .u32 rels,
    .param .u64 out, .param .u64 cursor)
{
    .reg .pred %p<5>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<28>;
    ld.param.u64 %rd1, [marks];
    ld.param.u64 %rd2, [coords];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [rels];
    ld.param.u64 %rd3, [out];
    ld.param.u64 %rd4, [cursor];
    mov.u32 %r3, 0;
ROW_LOOP:
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra DONE;
    mul.wide.u32 %rd5, %r3, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u32 %r4, [%rd6];
    setp.ne.u32 %p2, %r4, 0;
    @%p2 bra NEXT;
    atom.global.add.u64 %rd7, [%rd4], 1;
    setp.eq.u64 %p3, %rd3, 0;
    @%p3 bra NEXT;
    add.u32 %r5, %r2, 1;
    cvt.u64.u32 %rd8, %r5;
    mul.lo.u64 %rd9, %rd7, %rd8;
    mov.u32 %r6, 0;
COPY_LOOP:
    setp.ge.u32 %p4, %r6, %r2;
    @%p4 bra WRITE_PAD;
    mul.lo.u32 %r7, %r3, %r2;
    add.u32 %r7, %r7, %r6;
    mul.wide.u32 %rd10, %r7, 4;
    add.u64 %rd11, %rd2, %rd10;
    ld.global.u32 %r8, [%rd11];
    cvt.u64.u32 %rd12, %r6;
    add.u64 %rd13, %rd9, %rd12;
    mul.lo.u64 %rd13, %rd13, 4;
    add.u64 %rd14, %rd3, %rd13;
    st.global.u32 [%rd14], %r8;
    add.u32 %r6, %r6, 1;
    bra COPY_LOOP;
WRITE_PAD:
    cvt.u64.u32 %rd12, %r2;
    add.u64 %rd13, %rd9, %rd12;
    mul.lo.u64 %rd13, %rd13, 4;
    add.u64 %rd14, %rd3, %rd13;
    mov.u32 %r8, 4294967295;
    st.global.u32 [%rd14], %r8;
NEXT:
    add.u32 %r3, %r3, 1;
    bra ROW_LOOP;
DONE:
    ret;
}
"#;
    if accumulated.row_count != row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            row_count as usize,
        ));
    }
    if row_count == 0 {
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: accumulated.relation_count + 1,
            relation_row_counts: {
                let mut counts = accumulated.relation_row_counts.clone();
                counts.push(0);
                counts
            },
            allocated_bytes: 0,
        });
    }
    let source = accumulated
        .coordinates
        .as_ref()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    let primary = marks.primary_arc();
    primary.set_current()?;
    let cursor = primary.lease_device_buffer_owned(8)?;
    let memset = unsafe {
        primary
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let dtoh = unsafe {
        primary
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_unmatched_extended_coordinates", &ptx)?;
    let run = |out: u64| -> Result<(), CudaRuntimeProbeError> {
        let mut a0 = marks.device_ptr;
        let mut a1 = source.ptr;
        let mut a2 = row_count;
        let mut a3 = accumulated.relation_count;
        let mut a4 = out;
        let mut a5 = cursor.ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast(),
            (&mut a1 as *mut u64).cast(),
            (&mut a2 as *mut u32).cast(),
            (&mut a3 as *mut u32).cast(),
            (&mut a4 as *mut u64).cast(),
            (&mut a5 as *mut u64).cast(),
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
    let count =
        u32::try_from(count).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if count == 0 {
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: accumulated.relation_count + 1,
            relation_row_counts: {
                let mut counts = accumulated.relation_row_counts.clone();
                counts.push(0);
                counts
            },
            allocated_bytes: 0,
        });
    }
    let bytes = count as usize * (accumulated.relation_count as usize + 1) * 4;
    let output = primary.lease_device_buffer_owned(bytes)?;
    check_cuda(unsafe { memset(cursor.ptr, 0, 8) })?;
    run(output.ptr)?;
    Ok(CudaJoinCoordinatesU32 {
        coordinates: Some(output),
        row_count: count,
        relation_count: accumulated.relation_count + 1,
        relation_row_counts: {
            let mut counts = accumulated.relation_row_counts.clone();
            counts.push(0);
            counts
        },
        allocated_bytes: bytes as u64,
    })
}

fn launch_cuda_mark_coordinate_column_u32(
    marks: &CudaResidentDeviceMemory,
    row_count: u32,
    coordinates: &CudaJoinCoordinatesU32,
    relation: u32,
) -> Result<(), CudaRuntimeProbeError> {
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
.target sm_30
.address_size 64
.visible .entry gpu_db_mark_coordinate_column_u32(
    .param .u64 coords, .param .u32 tuple_count, .param .u32 relation_count,
    .param .u32 relation, .param .u64 marks, .param .u32 row_count)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r1, [tuple_count];
    ld.param.u32 %r2, [relation_count];
    ld.param.u32 %r3, [relation];
    ld.param.u64 %rd2, [marks];
    ld.param.u32 %r4, [row_count];
    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mov.u32 %r8, %nctaid.x;
    mad.lo.u32 %r9, %r6, %r7, %r5;
    mul.lo.u32 %r10, %r8, %r7;
LOOP:
    setp.ge.u32 %p1, %r9, %r1;
    @%p1 bra DONE;
    mul.lo.u32 %r11, %r9, %r2;
    add.u32 %r11, %r11, %r3;
    mul.wide.u32 %rd3, %r11, 4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u32 %r12, [%rd4];
    setp.ge.u32 %p2, %r12, %r4;
    @%p2 bra NEXT;
    mul.wide.u32 %rd5, %r12, 4;
    add.u64 %rd6, %rd2, %rd5;
    mov.u32 %r13, 1;
    atom.global.exch.b32 %r14, [%rd6], %r13;
NEXT:
    add.u32 %r9, %r9, %r10;
    bra LOOP;
DONE:
    ret;
}
"#;
    if relation >= coordinates.relation_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(relation as usize));
    }
    if coordinates.row_count == 0 {
        return Ok(());
    }
    let source = coordinates
        .coordinates
        .as_ref()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    let primary = marks.primary_arc();
    primary.set_current()?;
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_mark_coordinate_column_u32", &ptx)?;
    let mut a0 = source.ptr;
    let mut a1 = coordinates.row_count;
    let mut a2 = coordinates.relation_count;
    let mut a3 = relation;
    let mut a4 = marks.device_ptr;
    let mut a5 = row_count;
    let mut args = [
        (&mut a0 as *mut u64).cast(),
        (&mut a1 as *mut u32).cast(),
        (&mut a2 as *mut u32).cast(),
        (&mut a3 as *mut u32).cast(),
        (&mut a4 as *mut u64).cast(),
        (&mut a5 as *mut u32).cast(),
    ];
    check_cuda(unsafe {
        launch(
            function,
            coordinates.row_count.div_ceil(256).clamp(1, 65_535),
            1,
            1,
            256,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })
}
