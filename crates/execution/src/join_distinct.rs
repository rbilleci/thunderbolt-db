//! Exact DISTINCT over a device-sorted, materialized coordinate relation.

use std::os::raw::c_void;

use super::{
    check_cuda, CudaJoinCoordinatesU32, CudaJoinPayloadKey, CudaResidentDeviceMemory,
    CudaResidentReadSource, CudaRuntimeProbeError,
};

impl CudaResidentDeviceMemory {
    /// Compact adjacent-equal rows from coordinates already sorted by every `keys` member.
    ///
    /// Keys and coordinates remain device-resident. Only the resulting cardinality scalar crosses
    /// D2H; row equality and survivor selection execute on the GPU.
    pub fn distinct_sorted_join_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        keys: &[CudaJoinPayloadKey<'_>],
    ) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
        launch_cuda_distinct_sorted_join_coordinates(self, coordinates, keys)
    }
}

fn launch_cuda_distinct_sorted_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    coordinates: &CudaJoinCoordinatesU32,
    keys: &[CudaJoinPayloadKey<'_>],
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
.visible .entry gpu_db_distinct_sorted_join_coordinates(
    .param .u64 coords, .param .u32 rows, .param .u64 desc,
    .param .u32 key_count, .param .u64 out, .param .u64 cursor)
{
    .reg .pred %p<32>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<96>;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r1, [rows];
    ld.param.u64 %rd2, [desc];
    ld.param.u32 %r2, [key_count];
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
    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u32 %r9, [%rd6];
    setp.eq.u32 %p2, %r7, 0;
    @%p2 bra KEEP;
    sub.u32 %r10, %r7, 1;
    mul.wide.u32 %rd7, %r10, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r11, [%rd8];
    mov.u32 %r12, 0;
KEY_LOOP:
    setp.ge.u32 %p3, %r12, %r2;
    @%p3 bra ROW_NEXT;
    mul.wide.u32 %rd9, %r12, 48;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.u64 %rd11, [%rd10];
    ld.global.u64 %rd12, [%rd10+8];
    ld.global.u64 %rd13, [%rd10+16];
    ld.global.u64 %rd14, [%rd10+24];
    ld.global.u64 %rd15, [%rd10+32];
    mov.u32 %r13, 1;
    mov.u32 %r14, 1;
    mov.u64 %rd16, 18446744073709551615;
    setp.eq.u64 %p4, %rd13, %rd16;
    @%p4 bra VALID_READY;
    shr.u32 %r15, %r9, 5;
    mul.wide.u32 %rd17, %r15, 4;
    add.u64 %rd18, %rd13, %rd17;
    ld.global.u32 %r16, [%rd18];
    and.b32 %r17, %r9, 31;
    shr.u32 %r16, %r16, %r17;
    and.b32 %r13, %r16, 1;
    shr.u32 %r15, %r11, 5;
    mul.wide.u32 %rd17, %r15, 4;
    add.u64 %rd18, %rd13, %rd17;
    ld.global.u32 %r16, [%rd18];
    and.b32 %r17, %r11, 31;
    shr.u32 %r16, %r16, %r17;
    and.b32 %r14, %r16, 1;
VALID_READY:
    setp.ne.u32 %p5, %r13, %r14;
    @%p5 bra KEEP;
    setp.eq.u32 %p6, %r13, 0;
    @%p6 bra KEY_NEXT;
    add.u64 %rd19, %rd11, %rd12;
    cvt.u32.u64 %r18, %rd14;
    setp.eq.u32 %p7, %r18, 255;
    @%p7 bra CMP_TEXT;
    mul.wide.u32 %rd20, %r9, %r18;
    mul.wide.u32 %rd21, %r11, %r18;
    add.u64 %rd22, %rd19, %rd20;
    add.u64 %rd23, %rd19, %rd21;
    setp.eq.u32 %p8, %r18, 4;
    @%p8 bra CMP_4;
    setp.eq.u32 %p9, %r18, 8;
    @%p9 bra CMP_8;
    ld.global.u64 %rd24, [%rd22];
    ld.global.u64 %rd25, [%rd23];
    setp.ne.u64 %p10, %rd24, %rd25;
    @%p10 bra KEEP;
    ld.global.u64 %rd24, [%rd22+8];
    ld.global.u64 %rd25, [%rd23+8];
    setp.ne.u64 %p11, %rd24, %rd25;
    @%p11 bra KEEP;
    bra KEY_NEXT;
CMP_4:
    ld.global.u32 %r19, [%rd22];
    ld.global.u32 %r20, [%rd23];
    setp.ne.u32 %p12, %r19, %r20;
    @%p12 bra KEEP;
    bra KEY_NEXT;
CMP_8:
    ld.global.u64 %rd24, [%rd22];
    ld.global.u64 %rd25, [%rd23];
    setp.ne.u64 %p13, %rd24, %rd25;
    @%p13 bra KEEP;
    bra KEY_NEXT;
CMP_TEXT:
    mul.wide.u32 %rd26, %r9, 8;
    mul.wide.u32 %rd27, %r11, 8;
    add.u64 %rd28, %rd19, %rd26;
    add.u64 %rd29, %rd19, %rd27;
    ld.global.u64 %rd30, [%rd28];
    ld.global.u64 %rd31, [%rd28+8];
    ld.global.u64 %rd32, [%rd29];
    ld.global.u64 %rd33, [%rd29+8];
    sub.u64 %rd34, %rd31, %rd30;
    sub.u64 %rd35, %rd33, %rd32;
    setp.ne.u64 %p14, %rd34, %rd35;
    @%p14 bra KEEP;
    mov.u64 %rd36, 0;
TEXT_LOOP:
    setp.ge.u64 %p15, %rd36, %rd34;
    @%p15 bra KEY_NEXT;
    add.u64 %rd37, %rd15, %rd30;
    add.u64 %rd37, %rd37, %rd36;
    add.u64 %rd38, %rd15, %rd32;
    add.u64 %rd38, %rd38, %rd36;
    ld.global.u8 %r21, [%rd37];
    ld.global.u8 %r22, [%rd38];
    setp.ne.u32 %p16, %r21, %r22;
    @%p16 bra KEEP;
    add.u64 %rd36, %rd36, 1;
    bra TEXT_LOOP;
KEY_NEXT:
    add.u32 %r12, %r12, 1;
    bra KEY_LOOP;
KEEP:
    atom.global.add.u64 %rd39, [%rd4], 1;
    mul.lo.u64 %rd40, %rd39, 4;
    add.u64 %rd41, %rd3, %rd40;
    st.global.u32 [%rd41], %r9;
ROW_NEXT:
    add.u32 %r7, %r7, %r8;
    bra ROW_LOOP;
DONE:
    ret;
}
"#;

    if keys.is_empty()
        || coordinates.relation_count != 1
        || coordinates.relation_row_counts.len() != 1
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(keys.len()));
    }
    if coordinates.row_count <= 1 {
        return ctx.window_join_coordinates(coordinates, 0, Some(coordinates.row_count));
    }
    let source = coordinates
        .coordinates
        .as_ref()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    let context_identity = std::ptr::from_ref(ctx.primary()).addr();
    let source_rows = u64::from(coordinates.relation_row_counts[0]);
    if source.capacity < coordinates.row_count as usize * 4
        || std::ptr::from_ref(source.primary.as_ref()).addr() != context_identity
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            coordinates.row_count as usize,
        ));
    }
    let descriptor = keys
        .iter()
        .map(|key| {
            if !matches!(key.width, 4 | 8 | 16 | 255)
                || std::ptr::from_ref(key.payload.primary()).addr() != context_identity
            {
                return Err(CudaRuntimeProbeError::InvalidInputLength(keys.len()));
            }
            let allocation_bytes = key.payload.metadata.allocated_bytes;
            let value_end = if key.width == 255 {
                key.byte_offset.checked_add(
                    source_rows
                        .checked_add(1)
                        .and_then(|rows| rows.checked_mul(8))
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(keys.len()))?,
                )
            } else {
                key.byte_offset.checked_add(
                    source_rows
                        .checked_mul(u64::from(key.width))
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(keys.len()))?,
                )
            }
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(keys.len()))?;
            let validity_end = key.validity_bitmap_offset.and_then(|offset| {
                source_rows
                    .div_ceil(32)
                    .checked_mul(4)
                    .and_then(|bytes| offset.checked_add(bytes))
            });
            let text_end = key
                .text_bytes_byte_offset
                .and_then(|offset| offset.checked_add(key.text_bytes_len));
            if value_end > allocation_bytes
                || key
                    .validity_bitmap_offset
                    .is_some_and(|_| validity_end.is_none_or(|end| end > allocation_bytes))
                || key.width == 255
                    && (key.text_bytes_byte_offset.is_none()
                        || text_end.is_none_or(|end| end > allocation_bytes))
                || key.width != 255
                    && (key.text_bytes_byte_offset.is_some() || key.text_bytes_len != 0)
            {
                return Err(CudaRuntimeProbeError::InvalidInputLength(keys.len()));
            }
            Ok([
                key.payload.device_ptr,
                key.byte_offset,
                key.validity_bitmap_offset
                    .map_or(u64::MAX, |offset| key.payload.device_ptr + offset),
                u64::from(key.width),
                key.text_bytes_byte_offset
                    .map_or(0, |offset| key.payload.device_ptr + offset),
                key.text_bytes_len,
            ])
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

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
    let descriptor_bytes = std::mem::size_of_val(descriptor.as_slice());
    let device_descriptor = primary.lease_device_buffer_owned(descriptor_bytes)?;
    check_cuda(unsafe {
        htod(
            device_descriptor.ptr,
            descriptor.as_ptr().cast(),
            descriptor_bytes,
        )
    })?;
    let output_bytes = coordinates.row_count as usize * 4;
    let output = primary.lease_device_buffer_owned(output_bytes)?;
    let cursor = primary.lease_device_buffer_owned(8)?;
    check_cuda(unsafe { memset(cursor.ptr, 0, 8) })?;
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_distinct_sorted_join_coordinates", &ptx)?;
    let mut a0 = source.ptr;
    let mut a1 = coordinates.row_count;
    let mut a2 = device_descriptor.ptr;
    let mut a3 = keys.len() as u32;
    let mut a4 = output.ptr;
    let mut a5 = cursor.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast(),
        (&mut a1 as *mut u32).cast(),
        (&mut a2 as *mut u64).cast(),
        (&mut a3 as *mut u32).cast(),
        (&mut a4 as *mut u64).cast(),
        (&mut a5 as *mut u64).cast(),
    ];
    if let Err(error) = check_cuda(unsafe {
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
    }) {
        let _ = ctx.synchronize_default_stream();
        return Err(error);
    }
    let mut distinct_rows = 0_u64;
    check_cuda(unsafe { dtoh((&mut distinct_rows as *mut u64).cast(), cursor.ptr, 8) })?;
    ctx.synchronize_default_stream()?;
    let distinct_rows = u32::try_from(distinct_rows)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if distinct_rows == 0 || distinct_rows > coordinates.row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            distinct_rows as usize,
        ));
    }
    Ok(CudaJoinCoordinatesU32 {
        coordinates: Some(output),
        row_count: distinct_rows,
        relation_count: 1,
        relation_row_counts: coordinates.relation_row_counts.clone(),
        allocated_bytes: output_bytes as u64,
    })
}
