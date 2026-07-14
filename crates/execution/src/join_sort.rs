//! Stable device-coordinate sort ownership.

use std::os::raw::c_void;

use super::join_window::launch_cuda_window_join_coordinates;
use super::{
    check_cuda, CudaJoinCoordinatesU32, CudaJoinOrderKey, CudaResidentDeviceMemory,
    CudaResidentReadSource, CudaRuntimeProbeError,
};

impl CudaResidentDeviceMemory {
    /// Stable device merge-sort over coordinate rows. Comparators dereference the original resident
    /// payloads, including NULL placement and varlen text, so no projected key array or row crosses the
    /// host boundary.
    pub fn sort_join_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        order: &[CudaJoinOrderKey<'_>],
    ) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
        launch_cuda_sort_join_coordinates(self, coordinates, order)
    }
}

fn launch_cuda_sort_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    coordinates: &CudaJoinCoordinatesU32,
    order: &[CudaJoinOrderKey<'_>],
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
    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64
.visible .entry gpu_db_merge_join_coordinates(
    .param .u64 src, .param .u64 dst, .param .u32 rows, .param .u32 rels,
    .param .u64 width, .param .u64 desc, .param .u32 key_count)
{
    .reg .pred %p<40>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<128>;
    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [dst];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [rels];
    ld.param.u64 %rd3, [width];
    ld.param.u64 %rd4, [desc];
    ld.param.u32 %r3, [key_count];
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mad.lo.u32 %r7, %r5, %r6, %r4;
    cvt.u64.u32 %rd5, %r7;
    mul.lo.u64 %rd6, %rd3, 2;
    mul.lo.u64 %rd7, %rd5, %rd6;
    cvt.u64.u32 %rd8, %r1;
    setp.ge.u64 %p1, %rd7, %rd8;
    @%p1 bra DONE;
    add.u64 %rd9, %rd7, %rd3;
    min.u64 %rd9, %rd9, %rd8;
    add.u64 %rd10, %rd7, %rd6;
    min.u64 %rd10, %rd10, %rd8;
    mov.u64 %rd11, %rd7;
    mov.u64 %rd12, %rd9;
    mov.u64 %rd13, %rd7;
MERGE_LOOP:
    setp.ge.u64 %p2, %rd13, %rd10;
    @%p2 bra DONE;
    setp.ge.u64 %p3, %rd11, %rd9;
    @%p3 bra TAKE_RIGHT;
    setp.ge.u64 %p4, %rd12, %rd10;
    @%p4 bra TAKE_LEFT;
    mov.u32 %r8, 0;
KEY_LOOP:
    setp.ge.u32 %p5, %r8, %r3;
    @%p5 bra TAKE_LEFT; // stable tie
    mul.wide.u32 %rd14, %r8, 72;
    add.u64 %rd15, %rd4, %rd14;
    ld.global.u64 %rd16, [%rd15];      // payload base
    ld.global.u64 %rd17, [%rd15+8];    // value/offsets off
    ld.global.u64 %rd18, [%rd15+16];   // validity absolute or sentinel
    ld.global.u64 %rd19, [%rd15+24];   // width
    ld.global.u64 %rd20, [%rd15+32];   // text bytes off
    ld.global.u64 %rd21, [%rd15+40];   // relation
    ld.global.u64 %rd22, [%rd15+48];   // descending
    ld.global.u64 %rd23, [%rd15+56];   // nulls first
    ld.global.u64 %rd24, [%rd15+64];   // lexicographic 16
    cvt.u32.u64 %r9, %rd21;
    cvt.u32.u64 %r10, %rd11;
    cvt.u32.u64 %r11, %rd12;
    mul.lo.u32 %r12, %r10, %r2;
    add.u32 %r12, %r12, %r9;
    mul.lo.u32 %r13, %r11, %r2;
    add.u32 %r13, %r13, %r9;
    mul.wide.u32 %rd25, %r12, 4;
    mul.wide.u32 %rd26, %r13, 4;
    add.u64 %rd27, %rd1, %rd25;
    add.u64 %rd28, %rd1, %rd26;
    ld.global.u32 %r14, [%rd27];
    ld.global.u32 %r15, [%rd28];
    setp.ne.u32 %p6, %r14, 4294967295;
    selp.u32 %r16, 1, 0, %p6;
    setp.ne.u32 %p7, %r15, 4294967295;
    selp.u32 %r17, 1, 0, %p7;
    mov.u64 %rd29, 18446744073709551615;
    setp.eq.u64 %p8, %rd18, %rd29;
    @%p8 bra VALID_READY;
    setp.eq.u32 %p9, %r16, 0;
    @%p9 bra A_VALID_DONE;
    shr.u32 %r18, %r14, 5;
    mul.wide.u32 %rd30, %r18, 4;
    add.u64 %rd31, %rd18, %rd30;
    ld.global.u32 %r19, [%rd31];
    and.b32 %r18, %r14, 31;
    shr.u32 %r19, %r19, %r18;
    and.b32 %r16, %r19, 1;
A_VALID_DONE:
    setp.eq.u32 %p10, %r17, 0;
    @%p10 bra VALID_READY;
    shr.u32 %r18, %r15, 5;
    mul.wide.u32 %rd30, %r18, 4;
    add.u64 %rd31, %rd18, %rd30;
    ld.global.u32 %r19, [%rd31];
    and.b32 %r18, %r15, 31;
    shr.u32 %r19, %r19, %r18;
    and.b32 %r17, %r19, 1;
VALID_READY:
    setp.eq.u32 %p11, %r16, %r17;
    @%p11 bra BOTH_SAME_VALID;
    // A NULL comes first iff nulls_first; otherwise B/non-NULL comes first.
    setp.eq.u32 %p12, %r16, 0;
    setp.ne.u64 %p13, %rd23, 0;
    and.pred %p14, %p12, %p13;
    @%p14 bra TAKE_LEFT;
    @%p12 bra TAKE_RIGHT;
    @%p13 bra TAKE_RIGHT;
    bra TAKE_LEFT;
BOTH_SAME_VALID:
    setp.eq.u32 %p15, %r16, 0;
    @%p15 bra KEY_NEXT; // NULL peer
    add.u64 %rd32, %rd16, %rd17;
    cvt.u32.u64 %r20, %rd19;
    setp.eq.u32 %p16, %r20, 255;
    @%p16 bra CMP_TEXT;
    mul.wide.u32 %rd33, %r14, %r20;
    mul.wide.u32 %rd34, %r15, %r20;
    add.u64 %rd35, %rd32, %rd33;
    add.u64 %rd36, %rd32, %rd34;
    setp.eq.u32 %p17, %r20, 4;
    @%p17 bra CMP_I32;
    setp.eq.u32 %p18, %r20, 8;
    @%p18 bra CMP_I64;
    setp.ne.u64 %p19, %rd24, 0;
    @%p19 bra CMP_UUID;
    ld.global.u32 %r25,[%rd35+8]; ld.global.u32 %r26,[%rd35+12];
    cvt.u64.u32 %rd37,%r26; shl.b64 %rd37,%rd37,32; cvt.u64.u32 %rd42,%r25; or.b64 %rd37,%rd37,%rd42;
    ld.global.u32 %r27,[%rd36+8]; ld.global.u32 %r28,[%rd36+12];
    cvt.u64.u32 %rd38,%r28; shl.b64 %rd38,%rd38,32; cvt.u64.u32 %rd42,%r27; or.b64 %rd38,%rd38,%rd42;
    setp.lt.s64 %p20, %rd37, %rd38;
    @%p20 bra A_LESS;
    setp.gt.s64 %p21, %rd37, %rd38;
    @%p21 bra A_GREATER;
    ld.global.u32 %r25,[%rd35]; ld.global.u32 %r26,[%rd35+4];
    cvt.u64.u32 %rd37,%r26; shl.b64 %rd37,%rd37,32; cvt.u64.u32 %rd42,%r25; or.b64 %rd37,%rd37,%rd42;
    ld.global.u32 %r27,[%rd36]; ld.global.u32 %r28,[%rd36+4];
    cvt.u64.u32 %rd38,%r28; shl.b64 %rd38,%rd38,32; cvt.u64.u32 %rd42,%r27; or.b64 %rd38,%rd38,%rd42;
    setp.lt.u64 %p20, %rd37, %rd38;
    @%p20 bra A_LESS;
    setp.gt.u64 %p21, %rd37, %rd38;
    @%p21 bra A_GREATER;
    bra KEY_NEXT;
CMP_I32:
    ld.global.s32 %r21, [%rd35];
    ld.global.s32 %r22, [%rd36];
    setp.lt.s32 %p20, %r21, %r22;
    @%p20 bra A_LESS;
    setp.gt.s32 %p21, %r21, %r22;
    @%p21 bra A_GREATER;
    bra KEY_NEXT;
CMP_I64:
    ld.global.u32 %r25,[%rd35]; ld.global.u32 %r26,[%rd35+4];
    cvt.u64.u32 %rd37,%r26; shl.b64 %rd37,%rd37,32; cvt.u64.u32 %rd42,%r25; or.b64 %rd37,%rd37,%rd42;
    ld.global.u32 %r27,[%rd36]; ld.global.u32 %r28,[%rd36+4];
    cvt.u64.u32 %rd38,%r28; shl.b64 %rd38,%rd38,32; cvt.u64.u32 %rd42,%r27; or.b64 %rd38,%rd38,%rd42;
    setp.lt.s64 %p20, %rd37, %rd38;
    @%p20 bra A_LESS;
    setp.gt.s64 %p21, %rd37, %rd38;
    @%p21 bra A_GREATER;
    bra KEY_NEXT;
CMP_UUID:
    mov.u64 %rd39, 0;
UUID_LOOP:
    setp.ge.u64 %p22, %rd39, 16;
    @%p22 bra KEY_NEXT;
    add.u64 %rd40, %rd35, %rd39;
    add.u64 %rd41, %rd36, %rd39;
    ld.global.u8 %r23, [%rd40];
    ld.global.u8 %r24, [%rd41];
    setp.lt.u32 %p23, %r23, %r24;
    @%p23 bra A_LESS;
    setp.gt.u32 %p24, %r23, %r24;
    @%p24 bra A_GREATER;
    add.u64 %rd39, %rd39, 1;
    bra UUID_LOOP;
CMP_TEXT:
    mul.wide.u32 %rd42, %r14, 8;
    mul.wide.u32 %rd43, %r15, 8;
    add.u64 %rd44, %rd32, %rd42;
    add.u64 %rd45, %rd32, %rd43;
    ld.global.u64 %rd46, [%rd44];
    ld.global.u64 %rd47, [%rd44+8];
    ld.global.u64 %rd48, [%rd45];
    ld.global.u64 %rd49, [%rd45+8];
    sub.u64 %rd50, %rd47, %rd46;
    sub.u64 %rd51, %rd49, %rd48;
    min.u64 %rd52, %rd50, %rd51;
    add.u64 %rd53, %rd16, %rd20;
    add.u64 %rd53, %rd53, %rd46;
    add.u64 %rd54, %rd16, %rd20;
    add.u64 %rd54, %rd54, %rd48;
    mov.u64 %rd55, 0;
TEXT_LOOP:
    setp.ge.u64 %p25, %rd55, %rd52;
    @%p25 bra TEXT_LENGTH;
    add.u64 %rd56, %rd53, %rd55;
    add.u64 %rd57, %rd54, %rd55;
    ld.global.u8 %r25, [%rd56];
    ld.global.u8 %r26, [%rd57];
    setp.lt.u32 %p26, %r25, %r26;
    @%p26 bra A_LESS;
    setp.gt.u32 %p27, %r25, %r26;
    @%p27 bra A_GREATER;
    add.u64 %rd55, %rd55, 1;
    bra TEXT_LOOP;
TEXT_LENGTH:
    setp.lt.u64 %p28, %rd50, %rd51;
    @%p28 bra A_LESS;
    setp.gt.u64 %p29, %rd50, %rd51;
    @%p29 bra A_GREATER;
    bra KEY_NEXT;
A_LESS:
    setp.eq.u64 %p30, %rd22, 0;
    @%p30 bra TAKE_LEFT;
    bra TAKE_RIGHT;
A_GREATER:
    setp.eq.u64 %p31, %rd22, 0;
    @%p31 bra TAKE_RIGHT;
    bra TAKE_LEFT;
KEY_NEXT:
    add.u32 %r8, %r8, 1;
    bra KEY_LOOP;

TAKE_LEFT:
    mov.u64 %rd58, %rd11;
    add.u64 %rd11, %rd11, 1;
    bra COPY_ROW;
TAKE_RIGHT:
    mov.u64 %rd58, %rd12;
    add.u64 %rd12, %rd12, 1;
COPY_ROW:
    cvt.u64.u32 %rd59, %r2;
    mul.lo.u64 %rd60, %rd58, %rd59;
    mul.lo.u64 %rd61, %rd13, %rd59;
    mov.u32 %r27, 0;
COPY_LOOP:
    setp.ge.u32 %p32, %r27, %r2;
    @%p32 bra COPY_DONE;
    cvt.u64.u32 %rd62, %r27;
    add.u64 %rd63, %rd60, %rd62;
    add.u64 %rd64, %rd61, %rd62;
    mul.lo.u64 %rd63, %rd63, 4;
    mul.lo.u64 %rd64, %rd64, 4;
    add.u64 %rd63, %rd1, %rd63;
    add.u64 %rd64, %rd2, %rd64;
    ld.global.u32 %r28, [%rd63];
    st.global.u32 [%rd64], %r28;
    add.u32 %r27, %r27, 1;
    bra COPY_LOOP;
COPY_DONE:
    add.u64 %rd13, %rd13, 1;
    bra MERGE_LOOP;
DONE:
    ret;
}
"#;
    if order.is_empty() || coordinates.row_count <= 1 {
        return launch_cuda_window_join_coordinates(
            ctx,
            coordinates,
            0,
            Some(coordinates.row_count),
        );
    }
    if order.iter().any(|key| {
        key.relation >= coordinates.relation_count
            || !matches!(key.key.width, 4 | 8 | 16 | 255)
            || key.key.payload.metadata.gpu_id != ctx.metadata.gpu_id
    }) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(order.len()));
    }
    let source = coordinates
        .coordinates
        .as_ref()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    let primary = ctx.primary_arc();
    primary.set_current()?;
    let descriptor: Vec<u64> = order
        .iter()
        .flat_map(|item| {
            [
                item.key.payload.device_ptr,
                item.key.byte_offset,
                item.key
                    .validity_bitmap_offset
                    .map_or(u64::MAX, |off| item.key.payload.device_ptr + off),
                u64::from(item.key.width),
                item.key.text_bytes_byte_offset.unwrap_or(0),
                u64::from(item.relation),
                u64::from(item.descending),
                u64::from(item.nulls_first),
                u64::from(item.lexicographic_16),
            ]
        })
        .collect();
    let desc_bytes = std::mem::size_of_val(descriptor.as_slice());
    let desc = primary.lease_device_buffer_owned(desc_bytes)?;
    let htod = unsafe {
        primary
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { htod(desc.ptr, descriptor.as_ptr().cast(), desc_bytes) })?;
    let bytes = coordinates.row_count as usize * coordinates.relation_count as usize * 4;
    let mut a = Some(primary.lease_device_buffer_owned(bytes)?);
    let mut b = Some(primary.lease_device_buffer_owned(bytes)?);
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_merge_join_coordinates", &ptx)?;
    let mut width = 1_u64;
    let mut pass = 0_u32;
    let mut src_ptr = source.ptr;
    while width < u64::from(coordinates.row_count) {
        let dst_ptr = if pass.is_multiple_of(2) {
            a.as_ref().unwrap().ptr
        } else {
            b.as_ref().unwrap().ptr
        };
        let merge_count = u64::from(coordinates.row_count).div_ceil(width * 2);
        let mut a0 = src_ptr;
        let mut a1 = dst_ptr;
        let mut a2 = coordinates.row_count;
        let mut a3 = coordinates.relation_count;
        let mut a4 = width;
        let mut a5 = desc.ptr;
        let mut a6 = order.len() as u32;
        let mut args = [
            (&mut a0 as *mut u64).cast(),
            (&mut a1 as *mut u64).cast(),
            (&mut a2 as *mut u32).cast(),
            (&mut a3 as *mut u32).cast(),
            (&mut a4 as *mut u64).cast(),
            (&mut a5 as *mut u64).cast(),
            (&mut a6 as *mut u32).cast(),
        ];
        check_cuda(unsafe {
            launch(
                function,
                merge_count.div_ceil(256).clamp(1, 65_535) as u32,
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
        src_ptr = dst_ptr;
        pass += 1;
        width = width.saturating_mul(2);
    }
    let output = if pass % 2 == 1 {
        a.take().unwrap()
    } else {
        b.take().unwrap()
    };
    Ok(CudaJoinCoordinatesU32 {
        coordinates: Some(output),
        row_count: coordinates.row_count,
        relation_count: coordinates.relation_count,
        // Only the selected output buffer is retained; the alternate merge buffer and descriptor
        // are returned to their pools before this coordinate relation escapes.
        allocated_bytes: bytes as u64,
    })
}
