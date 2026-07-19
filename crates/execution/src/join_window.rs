//! Device-resident join-coordinate window, rank, and shift ownership.

use super::*;

impl CudaResidentDeviceMemory {
    /// Rank-family columns over an already ordered device-coordinate relation. Source keys are
    /// dereferenced from their resident payloads and rank triples remain device-resident.
    pub fn window_ranks_from_join_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        partition: &[CudaJoinOrderKey<'_>],
        order: &[CudaJoinOrderKey<'_>],
    ) -> Result<CudaWindowRanksU64, CudaRuntimeProbeError> {
        launch_cuda_window_ranks_from_join_coordinates(self, coordinates, partition, order)
    }

    /// Shift each ordered coordinate within its partition for LAG/LEAD. Rows that cross a partition
    /// boundary become OUTER pads; the selected value is gathered only at final result framing.
    pub fn shift_join_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        partition: &[CudaJoinOrderKey<'_>],
        delta: i32,
    ) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
        launch_cuda_shift_join_coordinates(self, coordinates, partition, delta)
    }

    pub fn window_join_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        offset: u32,
        limit: Option<u32>,
    ) -> Result<CudaJoinCoordinatesU32, CudaRuntimeProbeError> {
        launch_cuda_window_join_coordinates(self, coordinates, offset, limit)
    }
}

pub struct CudaWindowRanksU64 {
    values: PooledDeviceBufferOwned,
    row_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaWindowRankKind {
    RowNumber,
    Rank,
    DenseRank,
}

impl CudaWindowRanksU64 {
    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.values.capacity as u64
    }

    /// Final result readback for one requested rank-family column.
    pub fn readback(
        &self,
        kind: CudaWindowRankKind,
        offset: u32,
        limit: Option<u32>,
    ) -> Result<Vec<u64>, CudaRuntimeProbeError> {
        type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
        let start = offset.min(self.row_count);
        let count = limit.map_or(self.row_count - start, |limit| {
            limit.min(self.row_count - start)
        });
        if count == 0 {
            return Ok(Vec::new());
        }
        self.values.primary.set_current()?;
        let dtoh = unsafe {
            self.values
                .primary
                .lib()
                .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| {
                    self.values
                        .primary
                        .lib()
                        .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0")
                })
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        // Triples are interleaved, so gather the requested lane on-device before the one final D2H.
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
.visible .entry gpu_db_gather_rank_lane(
    .param .u64 src, .param .u32 start, .param .u32 count,
    .param .u32 lane, .param .u64 dst)
{
    .reg .pred %p;
    .reg .b32 %r<12>;
    .reg .b64 %rd<16>;
    ld.param.u64 %rd1, [src];
    ld.param.u32 %r1, [start];
    ld.param.u32 %r2, [count];
    ld.param.u32 %r3, [lane];
    ld.param.u64 %rd2, [dst];
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %nctaid.x;
    mad.lo.u32 %r8, %r5, %r6, %r4;
    mul.lo.u32 %r9, %r7, %r6;
LOOP:
    setp.ge.u32 %p, %r8, %r2;
    @%p bra DONE;
    add.u32 %r10, %r8, %r1;
    mul.lo.u32 %r10, %r10, 3;
    add.u32 %r10, %r10, %r3;
    mul.wide.u32 %rd3, %r10, 8;
    mul.wide.u32 %rd4, %r8, 8;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.u64 %rd7, [%rd5];
    st.global.u64 [%rd6], %rd7;
    add.u32 %r8, %r8, %r9;
    bra LOOP;
DONE:
    ret;
}
"#;
        let output = self
            .values
            .primary
            .lease_device_buffer_owned(count as usize * 8)?;
        let launch = unsafe {
            self.values
                .primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let mut ptx = PTX.to_vec();
        ptx.push(0);
        let function = self
            .values
            .primary
            .cached_function(c"gpu_db_gather_rank_lane", &ptx)?;
        let mut a0 = self.values.ptr;
        let mut a1 = start;
        let mut a2 = count;
        let mut a3 = match kind {
            CudaWindowRankKind::RowNumber => 0_u32,
            CudaWindowRankKind::Rank => 1,
            CudaWindowRankKind::DenseRank => 2,
        };
        let mut a4 = output.ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast(),
            (&mut a1 as *mut u32).cast(),
            (&mut a2 as *mut u32).cast(),
            (&mut a3 as *mut u32).cast(),
            (&mut a4 as *mut u64).cast(),
        ];
        check_cuda(unsafe {
            launch(
                function,
                count.div_ceil(256).clamp(1, 65_535),
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
        let mut result = vec![0_u64; count as usize];
        check_cuda(unsafe { dtoh(result.as_mut_ptr().cast(), output.ptr, result.len() * 8) })?;
        Ok(result)
    }
}

fn launch_cuda_window_ranks_from_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    coordinates: &CudaJoinCoordinatesU32,
    partition: &[CudaJoinOrderKey<'_>],
    order: &[CudaJoinOrderKey<'_>],
) -> Result<CudaWindowRanksU64, CudaRuntimeProbeError> {
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
.visible .entry gpu_db_window_rank_boundaries(
    .param .u64 coords, .param .u32 rows, .param .u32 rels,
    .param .u64 part_desc, .param .u32 part_n,
    .param .u64 order_desc, .param .u32 order_n, .param .u64 out)
{
    .reg .pred %p<32>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<112>;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r26, %nctaid.x;
    mad.lo.u32 %r8, %r2, %r3, %r1;
    mul.lo.u32 %r27, %r26, %r3;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r4, [rows];
    ld.param.u32 %r5, [rels];
    ld.param.u64 %rd2, [part_desc];
    ld.param.u32 %r6, [part_n];
    ld.param.u64 %rd3, [order_desc];
    ld.param.u32 %r7, [order_n];
    ld.param.u64 %rd4, [out];
ROW_LOOP:
    setp.ge.u32 %p2, %r8, %r4;
    @%p2 bra DONE;
    setp.eq.u32 %p3, %r8, 0;
    @%p3 bra NEW_PART;
    sub.u32 %r9, %r8, 1;
    mov.u32 %r10, 0;
    mov.u64 %rd8, %rd2;
    mov.u32 %r11, %r6;
COMPARE_PART:
    setp.ge.u32 %p4, %r10, %r11;
    @%p4 bra SAME_PART;
    bra COMPARE_KEY;
SAME_PART:
    mov.u32 %r10, 0;
    mov.u64 %rd8, %rd3;
    mov.u32 %r11, %r7;
COMPARE_ORDER:
    setp.ge.u32 %p4, %r10, %r11;
    @%p4 bra SAME_PEER;
COMPARE_KEY:
    mul.wide.u32 %rd9, %r10, 72;
    add.u64 %rd10, %rd8, %rd9;
    ld.global.u64 %rd11, [%rd10];
    ld.global.u64 %rd12, [%rd10+8];
    ld.global.u64 %rd13, [%rd10+16];
    ld.global.u64 %rd14, [%rd10+24];
    ld.global.u64 %rd15, [%rd10+32];
    ld.global.u64 %rd16, [%rd10+40];
    cvt.u32.u64 %r12, %rd16;
    mul.lo.u32 %r13, %r8, %r5;
    add.u32 %r13, %r13, %r12;
    mul.lo.u32 %r14, %r9, %r5;
    add.u32 %r14, %r14, %r12;
    mul.wide.u32 %rd17, %r13, 4;
    mul.wide.u32 %rd18, %r14, 4;
    add.u64 %rd19, %rd1, %rd17;
    add.u64 %rd20, %rd1, %rd18;
    ld.global.u32 %r15, [%rd19];
    ld.global.u32 %r16, [%rd20];
    setp.ne.u32 %p5, %r15, 4294967295;
    selp.u32 %r17, 1, 0, %p5;
    setp.ne.u32 %p6, %r16, 4294967295;
    selp.u32 %r18, 1, 0, %p6;
    mov.u64 %rd21, 18446744073709551615;
    setp.eq.u64 %p7, %rd13, %rd21;
    @%p7 bra VALID_READY;
    setp.eq.u32 %p8, %r17, 0;
    @%p8 bra CUR_VALID_DONE;
    shr.u32 %r19, %r15, 5;
    mul.wide.u32 %rd22, %r19, 4;
    add.u64 %rd23, %rd13, %rd22;
    ld.global.u32 %r20, [%rd23];
    and.b32 %r19, %r15, 31;
    shr.u32 %r20, %r20, %r19;
    and.b32 %r17, %r20, 1;
CUR_VALID_DONE:
    setp.eq.u32 %p9, %r18, 0;
    @%p9 bra VALID_READY;
    shr.u32 %r19, %r16, 5;
    mul.wide.u32 %rd22, %r19, 4;
    add.u64 %rd23, %rd13, %rd22;
    ld.global.u32 %r20, [%rd23];
    and.b32 %r19, %r16, 31;
    shr.u32 %r20, %r20, %r19;
    and.b32 %r18, %r20, 1;
VALID_READY:
    setp.ne.u32 %p10, %r17, %r18;
    @%p10 bra KEY_DIFFERENT;
    setp.eq.u32 %p11, %r17, 0;
    @%p11 bra KEY_EQUAL;
    cvt.u32.u64 %r21, %rd14;
    setp.eq.u32 %p12, %r21, 255;
    @%p12 bra TEXT_EQUALITY;
    mul.wide.u32 %rd24, %r15, %r21;
    mul.wide.u32 %rd25, %r16, %r21;
    add.u64 %rd26, %rd11, %rd12;
    add.u64 %rd27, %rd26, %rd24;
    add.u64 %rd28, %rd26, %rd25;
    setp.eq.u32 %p13, %r21, 4;
    @%p13 bra EQ4;
    ld.global.u32 %r30,[%rd27]; ld.global.u32 %r31,[%rd27+4];
    cvt.u64.u32 %rd29,%r31; shl.b64 %rd29,%rd29,32; cvt.u64.u32 %rd47,%r30; or.b64 %rd29,%rd29,%rd47;
    ld.global.u32 %r32,[%rd28]; ld.global.u32 %r33,[%rd28+4];
    cvt.u64.u32 %rd30,%r33; shl.b64 %rd30,%rd30,32; cvt.u64.u32 %rd47,%r32; or.b64 %rd30,%rd30,%rd47;
    setp.ne.u64 %p14, %rd29, %rd30;
    @%p14 bra KEY_DIFFERENT;
    setp.eq.u32 %p15, %r21, 16;
    @!%p15 bra KEY_EQUAL;
    ld.global.u32 %r30,[%rd27+8]; ld.global.u32 %r31,[%rd27+12];
    cvt.u64.u32 %rd29,%r31; shl.b64 %rd29,%rd29,32; cvt.u64.u32 %rd47,%r30; or.b64 %rd29,%rd29,%rd47;
    ld.global.u32 %r32,[%rd28+8]; ld.global.u32 %r33,[%rd28+12];
    cvt.u64.u32 %rd30,%r33; shl.b64 %rd30,%rd30,32; cvt.u64.u32 %rd47,%r32; or.b64 %rd30,%rd30,%rd47;
    setp.ne.u64 %p14, %rd29, %rd30;
    @%p14 bra KEY_DIFFERENT;
    bra KEY_EQUAL;
EQ4:
    ld.global.u32 %r22, [%rd27];
    ld.global.u32 %r23, [%rd28];
    setp.ne.u32 %p14, %r22, %r23;
    @%p14 bra KEY_DIFFERENT;
    bra KEY_EQUAL;
TEXT_EQUALITY:
    mul.wide.u32 %rd31, %r15, 8;
    mul.wide.u32 %rd32, %r16, 8;
    add.u64 %rd33, %rd11, %rd12;
    add.u64 %rd34, %rd33, %rd31;
    add.u64 %rd35, %rd33, %rd32;
    ld.global.u64 %rd36, [%rd34];
    ld.global.u64 %rd37, [%rd34+8];
    ld.global.u64 %rd38, [%rd35];
    ld.global.u64 %rd39, [%rd35+8];
    sub.u64 %rd40, %rd37, %rd36;
    sub.u64 %rd41, %rd39, %rd38;
    setp.ne.u64 %p16, %rd40, %rd41;
    @%p16 bra KEY_DIFFERENT;
    add.u64 %rd42, %rd11, %rd15;
    add.u64 %rd42, %rd42, %rd36;
    add.u64 %rd43, %rd11, %rd15;
    add.u64 %rd43, %rd43, %rd38;
    mov.u64 %rd44, 0;
TEXT_EQ_LOOP:
    setp.ge.u64 %p17, %rd44, %rd40;
    @%p17 bra KEY_EQUAL;
    add.u64 %rd45, %rd42, %rd44;
    add.u64 %rd46, %rd43, %rd44;
    ld.global.u8 %r24, [%rd45];
    ld.global.u8 %r25, [%rd46];
    setp.ne.u32 %p18, %r24, %r25;
    @%p18 bra KEY_DIFFERENT;
    add.u64 %rd44, %rd44, 1;
    bra TEXT_EQ_LOOP;
KEY_EQUAL:
    add.u32 %r10, %r10, 1;
    setp.eq.u64 %p19, %rd8, %rd2;
    @%p19 bra COMPARE_PART;
    bra COMPARE_ORDER;
KEY_DIFFERENT:
    setp.eq.u64 %p20, %rd8, %rd2;
    @%p20 bra NEW_PART;
    bra NEW_PEER;
SAME_PEER:
    mov.u64 %rd5, 0;
    mov.u64 %rd6, 0;
    mov.u64 %rd7, 0;
    bra WRITE;
NEW_PART:
    mov.u64 %rd5, 1;
    cvt.u64.u32 %rd6, %r8;
    mov.u64 %rd7, %rd6;
    bra WRITE;
NEW_PEER:
    mov.u64 %rd5, 1;
    mov.u64 %rd6, 0;
    cvt.u64.u32 %rd7, %r8;
WRITE:
    mul.wide.u32 %rd47, %r8, 24;
    add.u64 %rd48, %rd4, %rd47;
    st.global.u64 [%rd48], %rd5;
    st.global.u64 [%rd48+8], %rd6;
    st.global.u64 [%rd48+16], %rd7;
    add.u32 %r8, %r8, %r27;
    bra ROW_LOOP;
DONE:
    ret;
}

// Hillis-Steele passes compute, in parallel, the inclusive peer-boundary count and the latest
// partition/peer boundary row. Ping-pong buffers make every pass race-free.
.visible .entry gpu_db_window_rank_scan(
    .param .u64 src, .param .u64 dst, .param .u32 rows, .param .u32 stride)
{
    .reg .pred %p<4>; .reg .b32 %r<12>; .reg .b64 %rd<24>;
    ld.param.u64 %rd1,[src]; ld.param.u64 %rd2,[dst];
    ld.param.u32 %r1,[rows]; ld.param.u32 %r2,[stride];
    mov.u32 %r3,%tid.x; mov.u32 %r4,%ctaid.x; mov.u32 %r5,%ntid.x;
    mov.u32 %r6,%nctaid.x; mad.lo.u32 %r7,%r4,%r5,%r3; mul.lo.u32 %r8,%r6,%r5;
SCAN_ROW:
    setp.ge.u32 %p1,%r7,%r1; @%p1 bra SCAN_DONE;
    mul.wide.u32 %rd3,%r7,24; add.u64 %rd4,%rd1,%rd3;
    ld.global.u64 %rd5,[%rd4]; ld.global.u64 %rd6,[%rd4+8]; ld.global.u64 %rd7,[%rd4+16];
    setp.lt.u32 %p2,%r7,%r2; @%p2 bra SCAN_STORE;
    sub.u32 %r9,%r7,%r2; mul.wide.u32 %rd8,%r9,24; add.u64 %rd9,%rd1,%rd8;
    ld.global.u64 %rd10,[%rd9]; ld.global.u64 %rd11,[%rd9+8]; ld.global.u64 %rd12,[%rd9+16];
    add.u64 %rd5,%rd5,%rd10; max.u64 %rd6,%rd6,%rd11; max.u64 %rd7,%rd7,%rd12;
SCAN_STORE:
    add.u64 %rd13,%rd2,%rd3; st.global.u64 [%rd13],%rd5;
    st.global.u64 [%rd13+8],%rd6; st.global.u64 [%rd13+16],%rd7;
    add.u32 %r7,%r7,%r8; bra SCAN_ROW;
SCAN_DONE: ret;
}

.visible .entry gpu_db_window_rank_finalize(
    .param .u64 src, .param .u64 dst, .param .u32 rows)
{
    .reg .pred %p; .reg .b32 %r<12>; .reg .b64 %rd<28>;
    ld.param.u64 %rd1,[src]; ld.param.u64 %rd2,[dst]; ld.param.u32 %r1,[rows];
    mov.u32 %r2,%tid.x; mov.u32 %r3,%ctaid.x; mov.u32 %r4,%ntid.x;
    mov.u32 %r5,%nctaid.x; mad.lo.u32 %r6,%r3,%r4,%r2; mul.lo.u32 %r7,%r5,%r4;
FINAL_ROW:
    setp.ge.u32 %p,%r6,%r1; @%p bra FINAL_DONE;
    mul.wide.u32 %rd3,%r6,24; add.u64 %rd4,%rd1,%rd3;
    ld.global.u64 %rd5,[%rd4]; ld.global.u64 %rd6,[%rd4+8]; ld.global.u64 %rd7,[%rd4+16];
    mul.lo.u64 %rd8,%rd6,24; add.u64 %rd9,%rd1,%rd8; ld.global.u64 %rd10,[%rd9];
    cvt.u64.u32 %rd11,%r6; sub.u64 %rd12,%rd11,%rd6; add.u64 %rd12,%rd12,1;
    sub.u64 %rd13,%rd7,%rd6; add.u64 %rd13,%rd13,1;
    sub.u64 %rd14,%rd5,%rd10; add.u64 %rd14,%rd14,1;
    add.u64 %rd15,%rd2,%rd3; st.global.u64 [%rd15],%rd12;
    st.global.u64 [%rd15+8],%rd13; st.global.u64 [%rd15+16],%rd14;
    add.u32 %r6,%r6,%r7; bra FINAL_ROW;
FINAL_DONE: ret;
}
"#;
    if partition.iter().chain(order).any(|key| {
        key.relation >= coordinates.relation_count
            || key.key.payload.metadata.gpu_id != ctx.metadata.gpu_id
    }) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            coordinates.row_count as usize,
        ));
    }
    let primary = ctx.primary_arc();
    let bytes = (coordinates.row_count as usize).saturating_mul(24).max(1);
    let mut output = Some(primary.lease_device_buffer_owned(bytes)?);
    if coordinates.row_count == 0 {
        return Ok(CudaWindowRanksU64 {
            values: output.take().expect("rank output"),
            row_count: 0,
        });
    }
    let source = coordinates
        .coordinates
        .as_ref()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !std::ptr::eq(source.primary.as_ref(), ctx.primary()) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            coordinates.row_count as usize,
        ));
    }
    let pack = |keys: &[CudaJoinOrderKey<'_>]| {
        keys.iter()
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
                    0,
                    0,
                    0,
                ]
            })
            .collect::<Vec<_>>()
    };
    let part_host = pack(partition);
    let order_host = pack(order);
    let part_dev =
        primary.lease_device_buffer_owned(std::mem::size_of_val(part_host.as_slice()).max(1))?;
    let order_dev =
        primary.lease_device_buffer_owned(std::mem::size_of_val(order_host.as_slice()).max(1))?;
    primary.set_current()?;
    let htod = unsafe {
        primary
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    if !part_host.is_empty() {
        check_cuda(unsafe {
            htod(
                part_dev.ptr,
                part_host.as_ptr().cast(),
                std::mem::size_of_val(part_host.as_slice()),
            )
        })?;
    }
    if !order_host.is_empty() {
        check_cuda(unsafe {
            htod(
                order_dev.ptr,
                order_host.as_ptr().cast(),
                std::mem::size_of_val(order_host.as_slice()),
            )
        })?;
    }
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let boundary_fn = primary.cached_function(c"gpu_db_window_rank_boundaries", &ptx)?;
    let scan_fn = primary.cached_function(c"gpu_db_window_rank_scan", &ptx)?;
    let finalize_fn = primary.cached_function(c"gpu_db_window_rank_finalize", &ptx)?;
    let mut scratch = Some(primary.lease_device_buffer_owned(bytes)?);
    let mut a0 = source.ptr;
    let mut a1 = coordinates.row_count;
    let mut a2 = coordinates.relation_count;
    let mut a3 = part_dev.ptr;
    let mut a4 = partition.len() as u32;
    let mut a5 = order_dev.ptr;
    let mut a6 = order.len() as u32;
    let mut a7 = output.as_ref().expect("rank output").ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast(),
        (&mut a1 as *mut u32).cast(),
        (&mut a2 as *mut u32).cast(),
        (&mut a3 as *mut u64).cast(),
        (&mut a4 as *mut u32).cast(),
        (&mut a5 as *mut u64).cast(),
        (&mut a6 as *mut u32).cast(),
        (&mut a7 as *mut u64).cast(),
    ];
    if let Err(err) = check_cuda(unsafe {
        launch(
            boundary_fn,
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
        return Err(err);
    }
    let mut src_ptr = output.as_ref().expect("rank output").ptr;
    let mut dst_ptr = scratch.as_ref().expect("rank scratch").ptr;
    let mut stride = 1_u32;
    while stride < coordinates.row_count {
        let mut s0 = src_ptr;
        let mut s1 = dst_ptr;
        let mut s2 = coordinates.row_count;
        let mut s3 = stride;
        let mut scan_args = [
            (&mut s0 as *mut u64).cast(),
            (&mut s1 as *mut u64).cast(),
            (&mut s2 as *mut u32).cast(),
            (&mut s3 as *mut u32).cast(),
        ];
        check_cuda(unsafe {
            launch(
                scan_fn,
                coordinates.row_count.div_ceil(256).clamp(1, 65_535),
                1,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                scan_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;
        std::mem::swap(&mut src_ptr, &mut dst_ptr);
        stride = stride.saturating_mul(2);
    }
    // Finalize into the buffer not holding the scan. It reads the partition-start peer prefix from
    // the stable scan buffer, so this must remain a distinct allocation for the whole launch.
    let mut f0 = src_ptr;
    let mut f1 = dst_ptr;
    let mut f2 = coordinates.row_count;
    let mut finalize_args = [
        (&mut f0 as *mut u64).cast(),
        (&mut f1 as *mut u64).cast(),
        (&mut f2 as *mut u32).cast(),
    ];
    check_cuda(unsafe {
        launch(
            finalize_fn,
            coordinates.row_count.div_ceil(256).clamp(1, 65_535),
            1,
            1,
            256,
            1,
            1,
            0,
            std::ptr::null_mut(),
            finalize_args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    ctx.synchronize_default_stream()?;
    let values = if dst_ptr == output.as_ref().expect("rank output").ptr {
        output.take().expect("rank output")
    } else {
        scratch.take().expect("rank scratch")
    };
    Ok(CudaWindowRanksU64 {
        values,
        row_count: coordinates.row_count,
    })
}

fn launch_cuda_shift_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    coordinates: &CudaJoinCoordinatesU32,
    partition: &[CudaJoinOrderKey<'_>],
    delta: i32,
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
.visible .entry gpu_db_shift_join_coordinates(
    .param .u64 coords, .param .u32 rows, .param .u32 rels,
    .param .u64 part_desc, .param .u32 part_n, .param .s32 delta, .param .u64 out)
{
    .reg .pred %p<24>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<64>;
    ld.param.u64 %rd1, [coords];
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [rels];
    ld.param.u64 %rd2, [part_desc];
    ld.param.u32 %r3, [part_n];
    ld.param.s32 %r4, [delta];
    ld.param.u64 %rd3, [out];
    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mov.u32 %r8, %nctaid.x;
    mad.lo.u32 %r9, %r6, %r7, %r5;
    mul.lo.u32 %r10, %r8, %r7;
ROW_LOOP:
    setp.ge.u32 %p1, %r9, %r1;
    @%p1 bra DONE;
    cvt.s64.s32 %rd4, %r4;
    cvt.u64.u32 %rd5, %r9;
    cvt.s64.u64 %rd6, %rd5;
    add.s64 %rd7, %rd6, %rd4;
    setp.lt.s64 %p2, %rd7, 0;
    @%p2 bra WRITE_PAD;
    cvt.u64.s64 %rd8, %rd7;
    cvt.u64.u32 %rd9, %r1;
    setp.ge.u64 %p3, %rd8, %rd9;
    @%p3 bra WRITE_PAD;
    mov.u32 %r11, 0;
PART_LOOP:
    setp.ge.u32 %p4, %r11, %r3;
    @%p4 bra COPY_CANDIDATE;
    mul.wide.u32 %rd10, %r11, 48;
    add.u64 %rd11, %rd2, %rd10;
    ld.global.u64 %rd12, [%rd11];
    ld.global.u64 %rd13, [%rd11+8];
    ld.global.u64 %rd14, [%rd11+16];
    ld.global.u64 %rd15, [%rd11+24];
    ld.global.u64 %rd16, [%rd11+32];
    ld.global.u64 %rd33, [%rd11+40];
    cvt.u32.u64 %r12, %rd33;
    mul.lo.u32 %r13, %r9, %r2;
    add.u32 %r13, %r13, %r12;
    cvt.u32.u64 %r14, %rd8;
    mul.lo.u32 %r15, %r14, %r2;
    add.u32 %r15, %r15, %r12;
    mul.wide.u32 %rd17, %r13, 4;
    mul.wide.u32 %rd18, %r15, 4;
    add.u64 %rd19, %rd1, %rd17;
    add.u64 %rd20, %rd1, %rd18;
    ld.global.u32 %r16, [%rd19];
    ld.global.u32 %r17, [%rd20];
    setp.eq.u32 %p5, %r16, 4294967295;
    @%p5 bra WRITE_PAD;
    setp.eq.u32 %p5, %r17, 4294967295;
    @%p5 bra WRITE_PAD;
    mov.u32 %r18, 1;
    mov.u32 %r19, 1;
    mov.u64 %rd21, 18446744073709551615;
    setp.eq.u64 %p6, %rd14, %rd21;
    @%p6 bra VALID_READY;
    shr.u32 %r20, %r16, 5;
    mul.wide.u32 %rd22, %r20, 4;
    add.u64 %rd23, %rd14, %rd22;
    ld.global.u32 %r21, [%rd23];
    and.b32 %r20, %r16, 31;
    shr.u32 %r21, %r21, %r20;
    and.b32 %r18, %r21, 1;
    shr.u32 %r20, %r17, 5;
    mul.wide.u32 %rd22, %r20, 4;
    add.u64 %rd23, %rd14, %rd22;
    ld.global.u32 %r21, [%rd23];
    and.b32 %r20, %r17, 31;
    shr.u32 %r21, %r21, %r20;
    and.b32 %r19, %r21, 1;
VALID_READY:
    setp.ne.u32 %p7, %r18, %r19;
    @%p7 bra WRITE_PAD;
    setp.eq.u32 %p8, %r18, 0;
    @%p8 bra PART_NEXT;
    add.u64 %rd24, %rd12, %rd13;
    cvt.u32.u64 %r22, %rd15;
    setp.eq.u32 %p9, %r22, 255;
    @%p9 bra CMP_TEXT;
    mul.wide.u32 %rd25, %r16, %r22;
    mul.wide.u32 %rd26, %r17, %r22;
    add.u64 %rd25, %rd24, %rd25;
    add.u64 %rd26, %rd24, %rd26;
    setp.eq.u32 %p9, %r22, 4;
    @%p9 bra CMP4;
    ld.global.u32 %r30,[%rd25]; ld.global.u32 %r31,[%rd25+4];
    cvt.u64.u32 %rd27,%r31; shl.b64 %rd27,%rd27,32; cvt.u64.u32 %rd44,%r30; or.b64 %rd27,%rd27,%rd44;
    ld.global.u32 %r32,[%rd26]; ld.global.u32 %r33,[%rd26+4];
    cvt.u64.u32 %rd28,%r33; shl.b64 %rd28,%rd28,32; cvt.u64.u32 %rd44,%r32; or.b64 %rd28,%rd28,%rd44;
    setp.ne.u64 %p10, %rd27, %rd28;
    @%p10 bra WRITE_PAD;
    setp.ne.u32 %p11, %r22, 16;
    @%p11 bra PART_NEXT;
    ld.global.u32 %r30,[%rd25+8]; ld.global.u32 %r31,[%rd25+12];
    cvt.u64.u32 %rd27,%r31; shl.b64 %rd27,%rd27,32; cvt.u64.u32 %rd44,%r30; or.b64 %rd27,%rd27,%rd44;
    ld.global.u32 %r32,[%rd26+8]; ld.global.u32 %r33,[%rd26+12];
    cvt.u64.u32 %rd28,%r33; shl.b64 %rd28,%rd28,32; cvt.u64.u32 %rd44,%r32; or.b64 %rd28,%rd28,%rd44;
    setp.ne.u64 %p10, %rd27, %rd28;
    @%p10 bra WRITE_PAD;
    bra PART_NEXT;
CMP_TEXT:
    mul.wide.u32 %rd25,%r16,8; mul.wide.u32 %rd26,%r17,8;
    add.u64 %rd25,%rd24,%rd25; add.u64 %rd26,%rd24,%rd26;
    ld.global.u64 %rd27,[%rd25]; ld.global.u64 %rd28,[%rd25+8];
    ld.global.u64 %rd34,[%rd26]; ld.global.u64 %rd35,[%rd26+8];
    sub.u64 %rd36,%rd28,%rd27; sub.u64 %rd37,%rd35,%rd34;
    setp.ne.u64 %p10,%rd36,%rd37; @%p10 bra WRITE_PAD;
    add.u64 %rd38,%rd12,%rd16; add.u64 %rd39,%rd38,%rd27; add.u64 %rd40,%rd38,%rd34;
    mov.u64 %rd41,0;
CMP_TEXT_LOOP:
    setp.ge.u64 %p14,%rd41,%rd36; @%p14 bra PART_NEXT;
    add.u64 %rd42,%rd39,%rd41; add.u64 %rd43,%rd40,%rd41;
    ld.global.u8 %r30,[%rd42]; ld.global.u8 %r31,[%rd43];
    setp.ne.u32 %p10,%r30,%r31; @%p10 bra WRITE_PAD;
    add.u64 %rd41,%rd41,1; bra CMP_TEXT_LOOP;
CMP4:
    ld.global.u32 %r23, [%rd25];
    ld.global.u32 %r24, [%rd26];
    setp.ne.u32 %p10, %r23, %r24;
    @%p10 bra WRITE_PAD;
PART_NEXT:
    add.u32 %r11, %r11, 1;
    bra PART_LOOP;
COPY_CANDIDATE:
    mov.u32 %r25, 0;
COPY_LOOP:
    setp.ge.u32 %p12, %r25, %r2;
    @%p12 bra NEXT_ROW;
    cvt.u32.u64 %r26, %rd8;
    mul.lo.u32 %r27, %r26, %r2;
    add.u32 %r27, %r27, %r25;
    mul.lo.u32 %r28, %r9, %r2;
    add.u32 %r28, %r28, %r25;
    mul.wide.u32 %rd29, %r27, 4;
    mul.wide.u32 %rd30, %r28, 4;
    add.u64 %rd31, %rd1, %rd29;
    add.u64 %rd32, %rd3, %rd30;
    ld.global.u32 %r29, [%rd31];
    st.global.u32 [%rd32], %r29;
    add.u32 %r25, %r25, 1;
    bra COPY_LOOP;
WRITE_PAD:
    mov.u32 %r25, 0;
PAD_LOOP:
    setp.ge.u32 %p13, %r25, %r2;
    @%p13 bra NEXT_ROW;
    mul.lo.u32 %r28, %r9, %r2;
    add.u32 %r28, %r28, %r25;
    mul.wide.u32 %rd30, %r28, 4;
    add.u64 %rd32, %rd3, %rd30;
    mov.u32 %r29, 4294967295;
    st.global.u32 [%rd32], %r29;
    add.u32 %r25, %r25, 1;
    bra PAD_LOOP;
NEXT_ROW:
    add.u32 %r9, %r9, %r10;
    bra ROW_LOOP;
DONE:
    ret;
}
"#;
    if coordinates.row_count == 0 {
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: coordinates.relation_count,
            relation_row_counts: coordinates.relation_row_counts.clone(),
            allocated_bytes: 0,
        });
    }
    if partition.iter().any(|key| {
        key.relation >= coordinates.relation_count
            || !matches!(key.key.width, 4 | 8 | 16 | 255)
            || (key.key.width == 255 && key.key.text_bytes_byte_offset.is_none())
            || !std::ptr::eq(key.key.payload.primary(), ctx.primary())
    }) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(partition.len()));
    }
    let source = coordinates
        .coordinates
        .as_ref()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !std::ptr::eq(source.primary.as_ref(), ctx.primary()) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            coordinates.row_count as usize,
        ));
    }
    let primary = ctx.primary_arc();
    primary.set_current()?;
    let descriptors = partition
        .iter()
        .flat_map(|item| {
            [
                item.key.payload.device_ptr,
                item.key.byte_offset,
                item.key
                    .validity_bitmap_offset
                    .map_or(u64::MAX, |offset| item.key.payload.device_ptr + offset),
                u64::from(item.key.width),
                item.key.text_bytes_byte_offset.unwrap_or(0),
                u64::from(item.relation),
            ]
        })
        .collect::<Vec<_>>();
    let desc = if descriptors.is_empty() {
        None
    } else {
        let buffer = primary.lease_device_buffer_owned(descriptors.len() * 8)?;
        let htod = unsafe {
            primary
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        check_cuda(unsafe {
            htod(
                buffer.ptr,
                descriptors.as_ptr().cast(),
                descriptors.len() * 8,
            )
        })?;
        Some(buffer)
    };
    let bytes = coordinates.row_count as usize * coordinates.relation_count as usize * 4;
    let output = primary.lease_device_buffer_owned(bytes)?;
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_shift_join_coordinates", &ptx)?;
    let mut a0 = source.ptr;
    let mut a1 = coordinates.row_count;
    let mut a2 = coordinates.relation_count;
    let mut a3 = desc.as_ref().map_or(0, |buffer| buffer.ptr);
    let mut a4 = partition.len() as u32;
    let mut a5 = delta;
    let mut a6 = output.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast(),
        (&mut a1 as *mut u32).cast(),
        (&mut a2 as *mut u32).cast(),
        (&mut a3 as *mut u64).cast(),
        (&mut a4 as *mut u32).cast(),
        (&mut a5 as *mut i32).cast(),
        (&mut a6 as *mut u64).cast(),
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
    Ok(CudaJoinCoordinatesU32 {
        coordinates: Some(output),
        row_count: coordinates.row_count,
        relation_count: coordinates.relation_count,
        relation_row_counts: coordinates.relation_row_counts.clone(),
        allocated_bytes: bytes as u64,
    })
}

pub(super) fn launch_cuda_window_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    coordinates: &CudaJoinCoordinatesU32,
    offset: u32,
    limit: Option<u32>,
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
    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64
.visible .entry gpu_db_window_join_coordinates(
    .param .u64 src, .param .u32 start, .param .u32 rows,
    .param .u32 rels, .param .u64 dst)
{
    .reg .pred %p;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    ld.param.u64 %rd1, [src];
    ld.param.u32 %r1, [start];
    ld.param.u32 %r2, [rows];
    ld.param.u32 %r3, [rels];
    ld.param.u64 %rd2, [dst];
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %nctaid.x;
    mad.lo.u32 %r8, %r5, %r6, %r4;
    mul.lo.u32 %r9, %r7, %r6;
LOOP:
    setp.ge.u32 %p, %r8, %r2;
    @%p bra DONE;
    add.u32 %r10, %r8, %r1;
    mul.lo.u32 %r10, %r10, %r3;
    mul.lo.u32 %r11, %r8, %r3;
    mov.u32 %r12, 0;
COPY:
    setp.ge.u32 %p, %r12, %r3;
    @%p bra NEXT;
    add.u32 %r13, %r10, %r12;
    add.u32 %r14, %r11, %r12;
    mul.wide.u32 %rd3, %r13, 4;
    mul.wide.u32 %rd4, %r14, 4;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.u32 %r15, [%rd5];
    st.global.u32 [%rd6], %r15;
    add.u32 %r12, %r12, 1;
    bra COPY;
NEXT:
    add.u32 %r8, %r8, %r9;
    bra LOOP;
DONE:
    ret;
}
"#;
    let start = offset.min(coordinates.row_count);
    let available = coordinates.row_count - start;
    let rows = limit.map_or(available, |limit| limit.min(available));
    if rows == 0 {
        return Ok(CudaJoinCoordinatesU32 {
            coordinates: None,
            row_count: 0,
            relation_count: coordinates.relation_count,
            relation_row_counts: coordinates.relation_row_counts.clone(),
            allocated_bytes: 0,
        });
    }
    if start == 0 && rows == coordinates.row_count {
        // A no-op window still returns a distinct device relation because the coordinate guard is not
        // cloneable by design; one D2D pass keeps ownership explicit.
    }
    let source = coordinates
        .coordinates
        .as_ref()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !std::ptr::eq(source.primary.as_ref(), ctx.primary()) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(rows as usize));
    }
    let primary = ctx.primary_arc();
    primary.set_current()?;
    let bytes = rows as usize * coordinates.relation_count as usize * 4;
    let output = primary.lease_device_buffer_owned(bytes)?;
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_window_join_coordinates", &ptx)?;
    let mut a0 = source.ptr;
    let mut a1 = start;
    let mut a2 = rows;
    let mut a3 = coordinates.relation_count;
    let mut a4 = output.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast(),
        (&mut a1 as *mut u32).cast(),
        (&mut a2 as *mut u32).cast(),
        (&mut a3 as *mut u32).cast(),
        (&mut a4 as *mut u64).cast(),
    ];
    if let Err(err) = check_cuda(unsafe {
        launch(
            function,
            rows.div_ceil(256).clamp(1, 65_535),
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
        return Err(err);
    }
    // The returned guard owns only `output`; the borrowed source may be returned to the shared
    // pool by the caller immediately after this function returns. Complete the D2D read before that
    // ownership boundary (and drain a deferred CUDA fault while every operand is still alive).
    ctx.synchronize_default_stream()?;
    Ok(CudaJoinCoordinatesU32 {
        coordinates: Some(output),
        row_count: rows,
        relation_count: coordinates.relation_count,
        relation_row_counts: coordinates.relation_row_counts.clone(),
        allocated_bytes: bytes as u64,
    })
}
