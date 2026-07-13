//! Device-resident join-coordinate final typed projection ownership.

use std::os::raw::c_void;

use super::{
    check_cuda, CudaJoinCoordinatesU32, CudaResidentDeviceMemory, CudaResidentReadSource,
    CudaRuntimeProbeError,
};

impl CudaResidentDeviceMemory {
    /// Final fixed-width projection from retained join coordinates. The relational coordinate and
    /// NULL decisions remain on-device; returned bytes/validity are the final result materialization.
    pub fn project_fixed_from_join_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        relation: u32,
        payload: &CudaResidentDeviceMemory,
        byte_offset: u64,
        validity_bitmap_offset: Option<u64>,
        width: u8,
    ) -> Result<(Vec<u8>, Vec<bool>), CudaRuntimeProbeError> {
        launch_cuda_project_fixed_from_join_coordinates(
            self,
            coordinates,
            relation,
            payload,
            byte_offset,
            validity_bitmap_offset,
            width,
        )
    }

    pub fn project_bool_from_join_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        relation: u32,
        payload: &CudaResidentDeviceMemory,
        bitmap_byte_offset: u64,
        validity_bitmap_offset: Option<u64>,
    ) -> Result<Vec<Option<bool>>, CudaRuntimeProbeError> {
        launch_cuda_project_bool_from_join_coordinates(
            self,
            coordinates,
            relation,
            payload,
            bitmap_byte_offset,
            validity_bitmap_offset,
        )
    }

    /// Final UTF-8 projection from retained join coordinates. Device code resolves pads, NULL
    /// validity, and source varlen offsets; the host performs only final result-buffer framing.
    #[allow(clippy::too_many_arguments)]
    pub fn project_text_from_join_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        relation: u32,
        payload: &CudaResidentDeviceMemory,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        validity_bitmap_offset: Option<u64>,
    ) -> Result<Vec<Option<String>>, CudaRuntimeProbeError> {
        launch_cuda_project_text_from_join_coordinates(
            self,
            coordinates,
            relation,
            payload,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            validity_bitmap_offset,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_project_text_from_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    coordinates: &CudaJoinCoordinatesU32,
    relation: u32,
    payload: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    validity_bitmap_offset: Option<u64>,
) -> Result<Vec<Option<String>>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64
.visible .entry gpu_db_project_text_join_lengths(
    .param .u64 coords, .param .u32 rows, .param .u32 rels, .param .u32 relation,
    .param .u64 payload, .param .u64 offsets_off, .param .u64 validity,
    .param .u64 out_lengths, .param .u64 out_valid)
{
    .reg .pred %p<8>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<32>;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [rels];
    ld.param.u32 %r3, [relation];
    ld.param.u64 %rd2, [payload];
    ld.param.u64 %rd3, [offsets_off];
    ld.param.u64 %rd4, [validity];
    ld.param.u64 %rd5, [out_lengths];
    ld.param.u64 %rd6, [out_valid];
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %nctaid.x;
    mad.lo.u32 %r8, %r5, %r6, %r4;
    mul.lo.u32 %r9, %r7, %r6;
L_LOOP:
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra L_DONE;
    mov.u32 %r10, 0;
    mov.u64 %rd7, 0;
    mul.lo.u32 %r11, %r8, %r2;
    add.u32 %r11, %r11, %r3;
    mul.wide.u32 %rd8, %r11, 4;
    add.u64 %rd9, %rd1, %rd8;
    ld.global.u32 %r12, [%rd9];
    setp.eq.u32 %p2, %r12, 4294967295;
    @%p2 bra L_WRITE;
    mov.u64 %rd10, 18446744073709551615;
    setp.eq.u64 %p3, %rd4, %rd10;
    @%p3 bra L_VALID;
    shr.u32 %r13, %r12, 5;
    mul.wide.u32 %rd11, %r13, 4;
    add.u64 %rd12, %rd4, %rd11;
    ld.global.u32 %r14, [%rd12];
    and.b32 %r13, %r12, 31;
    shr.u32 %r14, %r14, %r13;
    and.b32 %r14, %r14, 1;
    setp.eq.u32 %p4, %r14, 0;
    @%p4 bra L_WRITE;
L_VALID:
    mov.u32 %r10, 1;
    mul.wide.u32 %rd13, %r12, 8;
    add.u64 %rd14, %rd2, %rd3;
    add.u64 %rd15, %rd14, %rd13;
    ld.global.u64 %rd16, [%rd15];
    ld.global.u64 %rd17, [%rd15+8];
    sub.u64 %rd7, %rd17, %rd16;
L_WRITE:
    mul.wide.u32 %rd18, %r8, 8;
    add.u64 %rd19, %rd5, %rd18;
    st.global.u64 [%rd19], %rd7;
    cvt.u64.u32 %rd20, %r8;
    add.u64 %rd21, %rd6, %rd20;
    st.global.u8 [%rd21], %r10;
    add.u32 %r8, %r8, %r9;
    bra L_LOOP;
L_DONE:
    ret;
}

.visible .entry gpu_db_project_text_join_copy(
    .param .u64 coords, .param .u32 rows, .param .u32 rels, .param .u32 relation,
    .param .u64 payload, .param .u64 offsets_off, .param .u64 bytes_off,
    .param .u64 validity, .param .u64 dest_offsets, .param .u64 out_bytes)
{
    .reg .pred %p<8>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<40>;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [rels];
    ld.param.u32 %r3, [relation];
    ld.param.u64 %rd2, [payload];
    ld.param.u64 %rd3, [offsets_off];
    ld.param.u64 %rd4, [bytes_off];
    ld.param.u64 %rd5, [validity];
    ld.param.u64 %rd6, [dest_offsets];
    ld.param.u64 %rd7, [out_bytes];
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %nctaid.x;
    mad.lo.u32 %r8, %r5, %r6, %r4;
    mul.lo.u32 %r9, %r7, %r6;
C_ROW:
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra C_DONE;
    mul.lo.u32 %r10, %r8, %r2;
    add.u32 %r10, %r10, %r3;
    mul.wide.u32 %rd8, %r10, 4;
    add.u64 %rd9, %rd1, %rd8;
    ld.global.u32 %r11, [%rd9];
    setp.eq.u32 %p2, %r11, 4294967295;
    @%p2 bra C_NEXT;
    mov.u64 %rd10, 18446744073709551615;
    setp.eq.u64 %p3, %rd5, %rd10;
    @%p3 bra C_VALID;
    shr.u32 %r12, %r11, 5;
    mul.wide.u32 %rd11, %r12, 4;
    add.u64 %rd12, %rd5, %rd11;
    ld.global.u32 %r13, [%rd12];
    and.b32 %r12, %r11, 31;
    shr.u32 %r13, %r13, %r12;
    and.b32 %r13, %r13, 1;
    setp.eq.u32 %p4, %r13, 0;
    @%p4 bra C_NEXT;
C_VALID:
    mul.wide.u32 %rd13, %r11, 8;
    add.u64 %rd14, %rd2, %rd3;
    add.u64 %rd15, %rd14, %rd13;
    ld.global.u64 %rd16, [%rd15];
    ld.global.u64 %rd17, [%rd15+8];
    mul.wide.u32 %rd18, %r8, 8;
    add.u64 %rd19, %rd6, %rd18;
    ld.global.u64 %rd20, [%rd19];
    add.u64 %rd21, %rd2, %rd4;
    add.u64 %rd21, %rd21, %rd16;
    add.u64 %rd22, %rd7, %rd20;
    sub.u64 %rd23, %rd17, %rd16;
    mov.u64 %rd24, 0;
C_BYTE:
    setp.ge.u64 %p5, %rd24, %rd23;
    @%p5 bra C_NEXT;
    add.u64 %rd25, %rd21, %rd24;
    add.u64 %rd26, %rd22, %rd24;
    ld.global.u8 %r14, [%rd25];
    st.global.u8 [%rd26], %r14;
    add.u64 %rd24, %rd24, 1;
    bra C_BYTE;
C_NEXT:
    add.u32 %r8, %r8, %r9;
    bra C_ROW;
C_DONE:
    ret;
}
"#;
    if relation >= coordinates.relation_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(relation as usize));
    }
    if coordinates.row_count == 0 {
        return Ok(Vec::new());
    }
    if payload.metadata.gpu_id != ctx.metadata.gpu_id
        || offsets_byte_offset > payload.metadata.allocated_bytes
        || bytes_byte_offset
            .checked_add(bytes_len)
            .is_none_or(|end| end > payload.metadata.allocated_bytes)
        || validity_bitmap_offset.is_some_and(|off| off > payload.metadata.allocated_bytes)
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            bytes_len as usize,
        ));
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
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let n = coordinates.row_count as usize;
    let lengths_dev = primary.lease_device_buffer_owned(n * 8)?;
    let valid_dev = primary.lease_device_buffer_owned(n)?;
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let lengths_fn = primary.cached_function(c"gpu_db_project_text_join_lengths", &ptx)?;
    let copy_fn = primary.cached_function(c"gpu_db_project_text_join_copy", &ptx)?;
    let validity = validity_bitmap_offset.map_or(u64::MAX, |off| payload.device_ptr + off);
    let mut a0 = source.ptr;
    let mut a1 = coordinates.row_count;
    let mut a2 = coordinates.relation_count;
    let mut a3 = relation;
    let mut a4 = payload.device_ptr;
    let mut a5 = offsets_byte_offset;
    let mut a6 = validity;
    let mut a7 = lengths_dev.ptr;
    let mut a8 = valid_dev.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast(),
        (&mut a1 as *mut u32).cast(),
        (&mut a2 as *mut u32).cast(),
        (&mut a3 as *mut u32).cast(),
        (&mut a4 as *mut u64).cast(),
        (&mut a5 as *mut u64).cast(),
        (&mut a6 as *mut u64).cast(),
        (&mut a7 as *mut u64).cast(),
        (&mut a8 as *mut u64).cast(),
    ];
    let grid = coordinates.row_count.div_ceil(256).clamp(1, 65_535);
    check_cuda(unsafe {
        launch(
            lengths_fn,
            grid,
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
    })?;
    let mut lengths = vec![0_u64; n];
    let mut valid = vec![0_u8; n];
    check_cuda(unsafe { dtoh(lengths.as_mut_ptr().cast(), lengths_dev.ptr, n * 8) })?;
    check_cuda(unsafe { dtoh(valid.as_mut_ptr().cast(), valid_dev.ptr, n) })?;
    let mut destinations = Vec::with_capacity(n + 1);
    destinations.push(0_u64);
    for &len in &lengths {
        destinations.push(
            destinations
                .last()
                .copied()
                .unwrap()
                .checked_add(len)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        );
    }
    let total = usize::try_from(*destinations.last().unwrap())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let destination_dev = primary.lease_device_buffer_owned(destinations.len() * 8)?;
    check_cuda(unsafe {
        htod(
            destination_dev.ptr,
            destinations.as_ptr().cast(),
            destinations.len() * 8,
        )
    })?;
    let bytes_dev = primary.lease_device_buffer_owned(total.max(1))?;
    let mut c0 = source.ptr;
    let mut c1 = coordinates.row_count;
    let mut c2 = coordinates.relation_count;
    let mut c3 = relation;
    let mut c4 = payload.device_ptr;
    let mut c5 = offsets_byte_offset;
    let mut c6 = bytes_byte_offset;
    let mut c7 = validity;
    let mut c8 = destination_dev.ptr;
    let mut c9 = bytes_dev.ptr;
    let mut cargs = [
        (&mut c0 as *mut u64).cast(),
        (&mut c1 as *mut u32).cast(),
        (&mut c2 as *mut u32).cast(),
        (&mut c3 as *mut u32).cast(),
        (&mut c4 as *mut u64).cast(),
        (&mut c5 as *mut u64).cast(),
        (&mut c6 as *mut u64).cast(),
        (&mut c7 as *mut u64).cast(),
        (&mut c8 as *mut u64).cast(),
        (&mut c9 as *mut u64).cast(),
    ];
    check_cuda(unsafe {
        launch(
            copy_fn,
            grid,
            1,
            1,
            256,
            1,
            1,
            0,
            std::ptr::null_mut(),
            cargs.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    let mut bytes = vec![0_u8; total];
    if total > 0 {
        check_cuda(unsafe { dtoh(bytes.as_mut_ptr().cast(), bytes_dev.ptr, total) })?;
    }
    (0..n)
        .map(|row| {
            if valid[row] == 0 {
                return Ok(None);
            }
            let start = destinations[row] as usize;
            let end = destinations[row + 1] as usize;
            String::from_utf8(bytes[start..end].to_vec())
                .map(Some)
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(end - start))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_project_bool_from_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    coordinates: &CudaJoinCoordinatesU32,
    relation: u32,
    payload: &CudaResidentDeviceMemory,
    bitmap_byte_offset: u64,
    validity_bitmap_offset: Option<u64>,
) -> Result<Vec<Option<bool>>, CudaRuntimeProbeError> {
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
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64
.visible .entry gpu_db_project_bool_join_coordinates(
    .param .u64 coords, .param .u32 rows, .param .u32 rels, .param .u32 relation,
    .param .u64 payload, .param .u64 bitmap, .param .u64 validity,
    .param .u64 out_values, .param .u64 out_valid)
{
    .reg .pred %p<8>;
    .reg .b32 %r<28>;
    .reg .b64 %rd<32>;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [rels];
    ld.param.u32 %r3, [relation];
    ld.param.u64 %rd2, [payload];
    ld.param.u64 %rd3, [bitmap];
    ld.param.u64 %rd4, [validity];
    ld.param.u64 %rd5, [out_values];
    ld.param.u64 %rd6, [out_valid];
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %nctaid.x;
    mad.lo.u32 %r8, %r5, %r6, %r4;
    mul.lo.u32 %r9, %r7, %r6;
LOOP:
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra DONE;
    mov.u32 %r10, 0;
    mov.u32 %r11, 0;
    mul.lo.u32 %r12, %r8, %r2;
    add.u32 %r12, %r12, %r3;
    mul.wide.u32 %rd7, %r12, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r13, [%rd8];
    setp.eq.u32 %p2, %r13, 4294967295;
    @%p2 bra WRITE;
    mov.u64 %rd9, 18446744073709551615;
    setp.eq.u64 %p3, %rd4, %rd9;
    @%p3 bra VALID;
    shr.u32 %r14, %r13, 5;
    mul.wide.u32 %rd10, %r14, 4;
    add.u64 %rd11, %rd4, %rd10;
    ld.global.u32 %r15, [%rd11];
    and.b32 %r14, %r13, 31;
    shr.u32 %r15, %r15, %r14;
    and.b32 %r15, %r15, 1;
    setp.eq.u32 %p4, %r15, 0;
    @%p4 bra WRITE;
VALID:
    mov.u32 %r10, 1;
    shr.u32 %r14, %r13, 5;
    mul.wide.u32 %rd10, %r14, 4;
    add.u64 %rd11, %rd2, %rd3;
    add.u64 %rd11, %rd11, %rd10;
    ld.global.u32 %r15, [%rd11];
    and.b32 %r14, %r13, 31;
    shr.u32 %r15, %r15, %r14;
    and.b32 %r11, %r15, 1;
WRITE:
    cvt.u64.u32 %rd12, %r8;
    add.u64 %rd13, %rd5, %rd12;
    add.u64 %rd14, %rd6, %rd12;
    st.global.u8 [%rd13], %r11;
    st.global.u8 [%rd14], %r10;
    add.u32 %r8, %r8, %r9;
    bra LOOP;
DONE:
    ret;
}
"#;
    if relation >= coordinates.relation_count || coordinates.row_count == 0 {
        return if coordinates.row_count == 0 {
            Ok(Vec::new())
        } else {
            Err(CudaRuntimeProbeError::InvalidInputLength(relation as usize))
        };
    }
    let source = coordinates
        .coordinates
        .as_ref()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    let primary = ctx.primary_arc();
    primary.set_current()?;
    let n = coordinates.row_count as usize;
    let values = primary.lease_device_buffer_owned(n)?;
    let valid = primary.lease_device_buffer_owned(n)?;
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
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_project_bool_join_coordinates", &ptx)?;
    let mut a0 = source.ptr;
    let mut a1 = coordinates.row_count;
    let mut a2 = coordinates.relation_count;
    let mut a3 = relation;
    let mut a4 = payload.device_ptr;
    let mut a5 = bitmap_byte_offset;
    let mut a6 = validity_bitmap_offset.map_or(u64::MAX, |off| payload.device_ptr + off);
    let mut a7 = values.ptr;
    let mut a8 = valid.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast(),
        (&mut a1 as *mut u32).cast(),
        (&mut a2 as *mut u32).cast(),
        (&mut a3 as *mut u32).cast(),
        (&mut a4 as *mut u64).cast(),
        (&mut a5 as *mut u64).cast(),
        (&mut a6 as *mut u64).cast(),
        (&mut a7 as *mut u64).cast(),
        (&mut a8 as *mut u64).cast(),
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
    })?;
    let mut values_host = vec![0_u8; n];
    let mut valid_host = vec![0_u8; n];
    check_cuda(unsafe { dtoh(values_host.as_mut_ptr().cast(), values.ptr, n) })?;
    check_cuda(unsafe { dtoh(valid_host.as_mut_ptr().cast(), valid.ptr, n) })?;
    Ok(values_host
        .into_iter()
        .zip(valid_host)
        .map(|(value, valid)| (valid != 0).then_some(value != 0))
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_project_fixed_from_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    coordinates: &CudaJoinCoordinatesU32,
    relation: u32,
    payload: &CudaResidentDeviceMemory,
    byte_offset: u64,
    validity_bitmap_offset: Option<u64>,
    width: u8,
) -> Result<(Vec<u8>, Vec<bool>), CudaRuntimeProbeError> {
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
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64
.visible .entry gpu_db_project_fixed_join_coordinates(
    .param .u64 coords, .param .u32 rows, .param .u32 rels, .param .u32 relation,
    .param .u64 payload, .param .u64 offset, .param .u64 validity, .param .u32 width,
    .param .u64 out_values, .param .u64 out_valid)
{
    .reg .pred %p<10>;
    .reg .b32 %r<28>;
    .reg .b64 %rd<40>;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [rels];
    ld.param.u32 %r3, [relation];
    ld.param.u64 %rd2, [payload];
    ld.param.u64 %rd3, [offset];
    ld.param.u64 %rd4, [validity];
    ld.param.u32 %r4, [width];
    ld.param.u64 %rd5, [out_values];
    ld.param.u64 %rd6, [out_valid];
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
    mul.wide.u32 %rd7, %r11, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r12, [%rd8];
    mov.u32 %r13, 0;
    setp.eq.u32 %p2, %r12, 4294967295;
    @%p2 bra WRITE_VALID;
    mov.u64 %rd9, 18446744073709551615;
    setp.eq.u64 %p3, %rd4, %rd9;
    @%p3 bra VALUE_VALID;
    shr.u32 %r14, %r12, 5;
    mul.wide.u32 %rd10, %r14, 4;
    add.u64 %rd11, %rd4, %rd10;
    ld.global.u32 %r15, [%rd11];
    and.b32 %r14, %r12, 31;
    shr.u32 %r15, %r15, %r14;
    and.b32 %r15, %r15, 1;
    setp.eq.u32 %p4, %r15, 0;
    @%p4 bra WRITE_VALID;
VALUE_VALID:
    mov.u32 %r13, 1;
    mul.wide.u32 %rd12, %r12, %r4;
    add.u64 %rd12, %rd12, %rd2;
    add.u64 %rd12, %rd12, %rd3;
    mul.wide.u32 %rd13, %r9, %r4;
    add.u64 %rd13, %rd5, %rd13;
    setp.eq.u32 %p5, %r4, 4;
    @%p5 bra COPY4;
    ld.global.u64 %rd14, [%rd12];
    st.global.u64 [%rd13], %rd14;
    setp.eq.u32 %p6, %r4, 16;
    @!%p6 bra WRITE_VALID;
    ld.global.u64 %rd14, [%rd12+8];
    st.global.u64 [%rd13+8], %rd14;
    bra WRITE_VALID;
COPY4:
    ld.global.u32 %r16, [%rd12];
    st.global.u32 [%rd13], %r16;
WRITE_VALID:
    cvt.u64.u32 %rd15, %r9;
    add.u64 %rd15, %rd6, %rd15;
    st.global.u8 [%rd15], %r13;
    add.u32 %r9, %r9, %r10;
    bra LOOP;
DONE:
    ret;
}
"#;
    if relation >= coordinates.relation_count || !matches!(width, 4 | 8 | 16) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(relation as usize));
    }
    if coordinates.row_count == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if payload.metadata.gpu_id != ctx.metadata.gpu_id
        || byte_offset > payload.metadata.allocated_bytes
        || validity_bitmap_offset.is_some_and(|off| off > payload.metadata.allocated_bytes)
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            byte_offset as usize,
        ));
    }
    let source = coordinates
        .coordinates
        .as_ref()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
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
    let value_bytes = coordinates.row_count as usize * width as usize;
    let values = primary.lease_device_buffer_owned(value_bytes)?;
    let valid = primary.lease_device_buffer_owned(coordinates.row_count as usize)?;
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_project_fixed_join_coordinates", &ptx)?;
    let mut a0 = source.ptr;
    let mut a1 = coordinates.row_count;
    let mut a2 = coordinates.relation_count;
    let mut a3 = relation;
    let mut a4 = payload.device_ptr;
    let mut a5 = byte_offset;
    let mut a6 = validity_bitmap_offset.map_or(u64::MAX, |off| payload.device_ptr + off);
    let mut a7 = u32::from(width);
    let mut a8 = values.ptr;
    let mut a9 = valid.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast(),
        (&mut a1 as *mut u32).cast(),
        (&mut a2 as *mut u32).cast(),
        (&mut a3 as *mut u32).cast(),
        (&mut a4 as *mut u64).cast(),
        (&mut a5 as *mut u64).cast(),
        (&mut a6 as *mut u64).cast(),
        (&mut a7 as *mut u32).cast(),
        (&mut a8 as *mut u64).cast(),
        (&mut a9 as *mut u64).cast(),
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
    })?;
    let mut host_values = vec![0_u8; value_bytes];
    let mut host_valid = vec![0_u8; coordinates.row_count as usize];
    check_cuda(unsafe { dtoh(host_values.as_mut_ptr().cast(), values.ptr, value_bytes) })?;
    check_cuda(unsafe { dtoh(host_valid.as_mut_ptr().cast(), valid.ptr, host_valid.len()) })?;
    Ok((
        host_values,
        host_valid.into_iter().map(|value| value != 0).collect(),
    ))
}
