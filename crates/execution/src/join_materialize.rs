//! Device-resident join materialization ownership.

use std::{os::raw::c_void, sync::Arc};

use super::{
    check_cuda, CudaAllocationScope, CudaDeviceMemoryProof, CudaExternalAllocationReservation,
    CudaJoinCoordinatesU32, CudaJoinPayloadKey, CudaResidentDeviceMemory, CudaResidentReadSource,
    CudaRuntimeProbeError, PooledDeviceBufferOwned,
};

impl CudaResidentDeviceMemory {
    pub fn materialize_join_coordinates(
        &self,
        coordinates: &CudaJoinCoordinatesU32,
        columns: &[CudaMaterializeJoinColumn<'_>],
    ) -> Result<CudaMaterializedRelation, CudaRuntimeProbeError> {
        launch_cuda_materialize_join_coordinates(self, coordinates, columns)
    }

    pub fn concat_materialized_relations(
        &self,
        left: &CudaMaterializedRelation,
        right: &CudaMaterializedRelation,
    ) -> Result<CudaMaterializedRelation, CudaRuntimeProbeError> {
        launch_cuda_concat_materialized_relations(self, left, right)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CudaMaterializeJoinColumn<'a> {
    Fixed {
        relation: u32,
        payload: &'a CudaResidentDeviceMemory,
        byte_offset: u64,
        validity_bitmap_offset: Option<u64>,
        width: u8,
    },
    Bool {
        relation: u32,
        payload: &'a CudaResidentDeviceMemory,
        bitmap_byte_offset: u64,
        validity_bitmap_offset: Option<u64>,
    },
    Text {
        relation: u32,
        payload: &'a CudaResidentDeviceMemory,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        validity_bitmap_offset: Option<u64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaMaterializedColumnKind {
    Fixed { width: u8 },
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMaterializedColumnLayout {
    pub kind: CudaMaterializedColumnKind,
    pub value_byte_offset: u64,
    pub text_bytes_byte_offset: Option<u64>,
    pub text_bytes_len: u64,
    pub validity_bitmap_offset: u64,
}

/// An opaque, self-contained device relation produced by a relational operator. It is safe to retain
/// after source chunks are evicted because every selected value and validity bit was gathered D2D.
pub struct CudaMaterializedRelation {
    memory: CudaResidentDeviceMemory,
    _reservation: CudaExternalAllocationReservation,
    columns: Vec<CudaMaterializedColumnLayout>,
    row_count: u32,
}

/// The terminal host-visible representation of a device-materialized result. Relational execution
/// must finish before this boundary: constructing a frame performs one contiguous D2H transfer, and
/// consumers may only decode/framing the copied bytes for the wire protocol.
#[derive(Debug)]
pub struct CudaMaterializedResultFrame {
    bytes: Vec<u8>,
    columns: Vec<CudaMaterializedColumnLayout>,
    row_count: u32,
}

impl CudaMaterializedResultFrame {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn columns(&self) -> &[CudaMaterializedColumnLayout] {
        &self.columns
    }

    pub fn row_count(&self) -> u32 {
        self.row_count
    }
}

impl CudaMaterializedRelation {
    pub fn memory(&self) -> &CudaResidentDeviceMemory {
        &self.memory
    }

    pub fn columns(&self) -> &[CudaMaterializedColumnLayout] {
        &self.columns
    }

    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.memory.metadata.allocated_bytes
    }

    /// Finish a device result with exactly one contiguous D2H transfer. Column-wise or row-wise
    /// readbacks belong before this boundary and are not exposed by the frame.
    pub fn read_result_frame(&self) -> Result<CudaMaterializedResultFrame, CudaRuntimeProbeError> {
        let byte_len = usize::try_from(self.allocated_bytes())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        Ok(CudaMaterializedResultFrame {
            bytes: self.memory.read_resident_bytes(0, byte_len)?,
            columns: self.columns.clone(),
            row_count: self.row_count,
        })
    }

    pub fn payload_key(&self, column: usize) -> Option<CudaJoinPayloadKey<'_>> {
        let layout = *self.columns.get(column)?;
        Some(match layout.kind {
            CudaMaterializedColumnKind::Fixed { width } => CudaJoinPayloadKey {
                payload: &self.memory,
                byte_offset: layout.value_byte_offset,
                validity_bitmap_offset: Some(layout.validity_bitmap_offset),
                width,
                text_bytes_byte_offset: None,
                text_bytes_len: 0,
            },
            CudaMaterializedColumnKind::Text => CudaJoinPayloadKey {
                payload: &self.memory,
                byte_offset: layout.value_byte_offset,
                validity_bitmap_offset: Some(layout.validity_bitmap_offset),
                width: 255,
                text_bytes_byte_offset: layout.text_bytes_byte_offset,
                text_bytes_len: layout.text_bytes_len,
            },
        })
    }
}

fn text_scan_partition_chunk(
    row_count: u32,
    block_count: u32,
) -> Result<u32, CudaRuntimeProbeError> {
    if block_count == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let quotient = row_count / block_count;
    quotient
        .checked_add(u32::from(!row_count.is_multiple_of(block_count)))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(
            row_count as usize,
        ))
}

fn validate_materialized_coordinate_extent(
    row_count: u32,
    relation_count: u32,
) -> Result<(), CudaRuntimeProbeError> {
    // The materialization PTX uses u32 grid-stride cursors. Keep their additions comfortably below
    // wrap instead of pretending the theoretical coordinate maximum is executable; larger results
    // fail loud before any allocation or launch.
    let coordinate_elements = u64::from(row_count)
        .checked_mul(u64::from(relation_count))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if relation_count == 0
        || row_count > i32::MAX as u32
        || coordinate_elements > u64::from(u32::MAX) + 1
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            row_count as usize,
        ));
    }
    Ok(())
}

fn launch_cuda_concat_materialized_relations(
    ctx: &CudaResidentDeviceMemory,
    left: &CudaMaterializedRelation,
    right: &CudaMaterializedRelation,
) -> Result<CudaMaterializedRelation, CudaRuntimeProbeError> {
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
    type CuMemcpyDtoD = unsafe extern "C" fn(u64, u64, usize) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    const PTX:&[u8]=br#"
.version 6.0
.target sm_30
.address_size 64
.visible .entry gpu_db_concat_validity(
 .param .u64 src,.param .u32 src_rows,.param .u64 dst,.param .u32 dst_base)
{
 .reg .pred %p<3>; .reg .b32 %r<20>; .reg .b64 %rd<16>;
 ld.param.u64 %rd1,[src];ld.param.u32 %r1,[src_rows];ld.param.u64 %rd2,[dst];ld.param.u32 %r2,[dst_base];
 mov.u32 %r3,%tid.x;mov.u32 %r4,%ctaid.x;mov.u32 %r5,%ntid.x;mov.u32 %r6,%nctaid.x;mad.lo.u32 %r7,%r4,%r5,%r3;mul.lo.u32 %r8,%r6,%r5;
V_LOOP:setp.ge.u32 %p1,%r7,%r1;@%p1 bra V_DONE;shr.u32 %r9,%r7,5;mul.wide.u32 %rd3,%r9,4;add.u64 %rd4,%rd1,%rd3;ld.global.u32 %r10,[%rd4];and.b32 %r11,%r7,31;shr.u32 %r10,%r10,%r11;and.b32 %r10,%r10,1;setp.eq.u32 %p2,%r10,0;@%p2 bra V_NEXT;
 add.u32 %r12,%r7,%r2;shr.u32 %r13,%r12,5;mul.wide.u32 %rd5,%r13,4;add.u64 %rd6,%rd2,%rd5;and.b32 %r14,%r12,31;mov.u32 %r15,1;shl.b32 %r15,%r15,%r14;atom.global.or.b32 %r16,[%rd6],%r15;
V_NEXT:add.u32 %r7,%r7,%r8;bra V_LOOP;V_DONE:ret;
}

.visible .entry gpu_db_concat_text_offsets(
 .param .u64 src,.param .u32 src_rows,.param .u64 dst,.param .u32 dst_base,.param .u64 byte_base)
{
 .reg .pred %p; .reg .b32 %r<16>; .reg .b64 %rd<20>;
 ld.param.u64 %rd1,[src];ld.param.u32 %r1,[src_rows];ld.param.u64 %rd2,[dst];ld.param.u32 %r2,[dst_base];ld.param.u64 %rd3,[byte_base];
 mov.u32 %r3,%tid.x;mov.u32 %r4,%ctaid.x;mov.u32 %r5,%ntid.x;mov.u32 %r6,%nctaid.x;mad.lo.u32 %r7,%r4,%r5,%r3;mul.lo.u32 %r8,%r6,%r5;add.u32 %r1,%r1,1;
O_LOOP:setp.ge.u32 %p,%r7,%r1;@%p bra O_DONE;mul.wide.u32 %rd4,%r7,8;add.u64 %rd5,%rd1,%rd4;ld.global.u64 %rd6,[%rd5];add.u64 %rd6,%rd6,%rd3;add.u32 %r9,%r7,%r2;mul.wide.u32 %rd7,%r9,8;add.u64 %rd8,%rd2,%rd7;st.global.u64 [%rd8],%rd6;add.u32 %r7,%r7,%r8;bra O_LOOP;O_DONE:ret;
}
"#;
    if left.columns.len() != right.columns.len()
        || left
            .columns
            .iter()
            .zip(&right.columns)
            .any(|(a, b)| a.kind != b.kind)
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            left.columns.len(),
        ));
    }
    let rows = left
        .row_count
        .checked_add(right.row_count)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let align = |v: u64, a: u64| v.div_ceil(a) * a;
    let valid_bytes = u64::from(rows).div_ceil(32) * 4;
    let mut cursor = 0_u64;
    let mut layouts = Vec::with_capacity(left.columns.len());
    for (l, r) in left.columns.iter().zip(&right.columns) {
        match l.kind {
            CudaMaterializedColumnKind::Fixed { width } => {
                cursor = align(cursor, u64::from(width.min(8)));
                let value = cursor;
                cursor += u64::from(rows) * u64::from(width);
                cursor = align(cursor, 4);
                let valid = cursor;
                cursor += valid_bytes;
                layouts.push(CudaMaterializedColumnLayout {
                    kind: l.kind,
                    value_byte_offset: value,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                    validity_bitmap_offset: valid,
                });
            }
            CudaMaterializedColumnKind::Text => {
                cursor = align(cursor, 8);
                let offsets = cursor;
                cursor += (u64::from(rows) + 1) * 8;
                let bytes = cursor;
                let text_len = l
                    .text_bytes_len
                    .checked_add(r.text_bytes_len)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                cursor += text_len;
                cursor = align(cursor, 4);
                let valid = cursor;
                cursor += valid_bytes;
                layouts.push(CudaMaterializedColumnLayout {
                    kind: l.kind,
                    value_byte_offset: offsets,
                    text_bytes_byte_offset: Some(bytes),
                    text_bytes_len: text_len,
                    validity_bitmap_offset: valid,
                });
            }
        }
    }
    let primary = ctx.primary_arc();
    primary.set_current()?;
    let allocated = cursor.max(1);
    let allocated_usize = usize::try_from(allocated)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let reservation = CudaAllocationScope::reserve_external(allocated_usize)?;
    let mut ptr = 0;
    check_cuda(unsafe { (primary.cu_mem_alloc)(&mut ptr, allocated_usize) })?;
    let memory = CudaResidentDeviceMemory::from_raw_parts(
        CudaDeviceMemoryProof {
            gpu_id: ctx.metadata.gpu_id,
            device_name: ctx.metadata.device_name.clone(),
            allocated_bytes: allocated,
            copied_bytes: 0,
            retained: true,
        },
        ptr,
        Arc::clone(&primary),
    );
    let memset = unsafe {
        primary
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { memset(memory.device_ptr, 0, allocated_usize) })?;
    let dtod = unsafe {
        primary
            .lib()
            .get::<CuMemcpyDtoD>(b"cuMemcpyDtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyDtoD>(b"cuMemcpyDtoD\0"))
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
    let valid_fn = primary.cached_function(c"gpu_db_concat_validity", &ptx)?;
    let offsets_fn = primary.cached_function(c"gpu_db_concat_text_offsets", &ptx)?;
    for (index, &out) in layouts.iter().enumerate() {
        let l = left.columns[index];
        let r = right.columns[index];
        match out.kind {
            CudaMaterializedColumnKind::Fixed { width } => {
                let lb = left.row_count as usize * width as usize;
                let rb = right.row_count as usize * width as usize;
                if lb > 0 {
                    check_cuda(unsafe {
                        dtod(
                            memory.device_ptr + out.value_byte_offset,
                            left.memory.device_ptr + l.value_byte_offset,
                            lb,
                        )
                    })?;
                }
                if rb > 0 {
                    check_cuda(unsafe {
                        dtod(
                            memory.device_ptr + out.value_byte_offset + lb as u64,
                            right.memory.device_ptr + r.value_byte_offset,
                            rb,
                        )
                    })?;
                }
            }
            CudaMaterializedColumnKind::Text => {
                let dst_bytes = memory.device_ptr + out.text_bytes_byte_offset.unwrap();
                if l.text_bytes_len > 0 {
                    check_cuda(unsafe {
                        dtod(
                            dst_bytes,
                            left.memory.device_ptr + l.text_bytes_byte_offset.unwrap(),
                            l.text_bytes_len as usize,
                        )
                    })?;
                }
                if r.text_bytes_len > 0 {
                    check_cuda(unsafe {
                        dtod(
                            dst_bytes + l.text_bytes_len,
                            right.memory.device_ptr + r.text_bytes_byte_offset.unwrap(),
                            r.text_bytes_len as usize,
                        )
                    })?;
                }
                for (src, src_rows, dst_base, byte_base) in [
                    (
                        left.memory.device_ptr + l.value_byte_offset,
                        left.row_count,
                        0_u32,
                        0_u64,
                    ),
                    (
                        right.memory.device_ptr + r.value_byte_offset,
                        right.row_count,
                        left.row_count,
                        l.text_bytes_len,
                    ),
                ] {
                    let mut a0 = src;
                    let mut a1 = src_rows;
                    let mut a2 = memory.device_ptr + out.value_byte_offset;
                    let mut a3 = dst_base;
                    let mut a4 = byte_base;
                    let mut args = [
                        (&mut a0 as *mut u64).cast(),
                        (&mut a1 as *mut u32).cast(),
                        (&mut a2 as *mut u64).cast(),
                        (&mut a3 as *mut u32).cast(),
                        (&mut a4 as *mut u64).cast(),
                    ];
                    check_cuda(unsafe {
                        launch(
                            offsets_fn,
                            (src_rows + 1).div_ceil(256).clamp(1, 65_535),
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
                }
            }
        }
        for (src, src_rows, dst_base) in [
            (
                left.memory.device_ptr + l.validity_bitmap_offset,
                left.row_count,
                0_u32,
            ),
            (
                right.memory.device_ptr + r.validity_bitmap_offset,
                right.row_count,
                left.row_count,
            ),
        ] {
            if src_rows == 0 {
                continue;
            }
            let mut a0 = src;
            let mut a1 = src_rows;
            let mut a2 = memory.device_ptr + out.validity_bitmap_offset;
            let mut a3 = dst_base;
            let mut args = [
                (&mut a0 as *mut u64).cast(),
                (&mut a1 as *mut u32).cast(),
                (&mut a2 as *mut u64).cast(),
                (&mut a3 as *mut u32).cast(),
            ];
            check_cuda(unsafe {
                launch(
                    valid_fn,
                    src_rows.div_ceil(256).clamp(1, 65_535),
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
        }
    }
    Ok(CudaMaterializedRelation {
        memory,
        _reservation: reservation,
        columns: layouts,
        row_count: rows,
    })
}

fn launch_cuda_materialize_join_coordinates(
    ctx: &CudaResidentDeviceMemory,
    coordinates: &CudaJoinCoordinatesU32,
    columns: &[CudaMaterializeJoinColumn<'_>],
) -> Result<CudaMaterializedRelation, CudaRuntimeProbeError> {
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
    type CuMemcpyDtoD = unsafe extern "C" fn(u64, u64, usize) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64
.visible .entry gpu_db_materialize_fixed(
    .param .u64 coords, .param .u32 rows, .param .u32 rels, .param .u32 relation,
    .param .u64 src, .param .u64 src_off, .param .u64 src_valid, .param .u32 width,
    .param .u64 dst, .param .u64 dst_off, .param .u64 dst_valid)
{
    .reg .pred %p<8>; .reg .b32 %r<32>; .reg .b64 %rd<40>;
    ld.param.u64 %rd1,[coords]; ld.param.u32 %r1,[rows]; ld.param.u32 %r2,[rels];
    ld.param.u32 %r3,[relation]; ld.param.u64 %rd2,[src]; ld.param.u64 %rd3,[src_off];
    ld.param.u64 %rd4,[src_valid]; ld.param.u32 %r4,[width]; ld.param.u64 %rd5,[dst];
    ld.param.u64 %rd6,[dst_off]; ld.param.u64 %rd7,[dst_valid];
    mov.u32 %r5,%tid.x; mov.u32 %r6,%ctaid.x; mov.u32 %r7,%ntid.x; mov.u32 %r8,%nctaid.x;
    mad.lo.u32 %r9,%r6,%r7,%r5; mul.lo.u32 %r10,%r8,%r7;
F_LOOP:
    setp.ge.u32 %p1,%r9,%r1; @%p1 bra F_DONE;
    mul.lo.u32 %r11,%r9,%r2; add.u32 %r11,%r11,%r3; mul.wide.u32 %rd8,%r11,4;
    add.u64 %rd9,%rd1,%rd8; ld.global.u32 %r12,[%rd9];
    setp.eq.u32 %p2,%r12,4294967295; @%p2 bra F_NEXT;
    mov.u64 %rd10,18446744073709551615; setp.eq.u64 %p3,%rd4,%rd10; @%p3 bra F_VALID;
    shr.u32 %r13,%r12,5; mul.wide.u32 %rd11,%r13,4; add.u64 %rd12,%rd4,%rd11;
    ld.global.u32 %r14,[%rd12]; and.b32 %r13,%r12,31; shr.u32 %r14,%r14,%r13;
    and.b32 %r14,%r14,1; setp.eq.u32 %p4,%r14,0; @%p4 bra F_NEXT;
F_VALID:
    mul.wide.u32 %rd13,%r12,%r4; add.u64 %rd13,%rd13,%rd2; add.u64 %rd13,%rd13,%rd3;
    mul.wide.u32 %rd14,%r9,%r4; add.u64 %rd14,%rd14,%rd5; add.u64 %rd14,%rd14,%rd6;
    setp.eq.u32 %p5,%r4,4; @%p5 bra F_COPY4;
    // The engine packs wide sections on a four-byte boundary. Copy 8/16-byte values as u32 words:
    // a single misaligned ld.global.u64 can fault when an odd-sized int4 section precedes this one.
    ld.global.u32 %r15,[%rd13]; st.global.u32 [%rd14],%r15;
    ld.global.u32 %r16,[%rd13+4]; st.global.u32 [%rd14+4],%r16;
    setp.eq.u32 %p6,%r4,16; @!%p6 bra F_MARK;
    ld.global.u32 %r17,[%rd13+8]; st.global.u32 [%rd14+8],%r17;
    ld.global.u32 %r18,[%rd13+12]; st.global.u32 [%rd14+12],%r18; bra F_MARK;
F_COPY4: ld.global.u32 %r15,[%rd13]; st.global.u32 [%rd14],%r15;
F_MARK:
    shr.u32 %r16,%r9,5; mul.wide.u32 %rd16,%r16,4; add.u64 %rd17,%rd5,%rd7; add.u64 %rd17,%rd17,%rd16;
    and.b32 %r17,%r9,31; mov.u32 %r18,1; shl.b32 %r18,%r18,%r17; atom.global.or.b32 %r19,[%rd17],%r18;
F_NEXT: add.u32 %r9,%r9,%r10; bra F_LOOP;
F_DONE: ret;
}

.visible .entry gpu_db_materialize_bool(
    .param .u64 coords, .param .u32 rows, .param .u32 rels, .param .u32 relation,
    .param .u64 src, .param .u64 src_bitmap, .param .u64 src_valid,
    .param .u64 dst, .param .u64 dst_off, .param .u64 dst_valid)
{
    .reg .pred %p<7>; .reg .b32 %r<32>; .reg .b64 %rd<36>;
    ld.param.u64 %rd1,[coords]; ld.param.u32 %r1,[rows]; ld.param.u32 %r2,[rels]; ld.param.u32 %r3,[relation];
    ld.param.u64 %rd2,[src]; ld.param.u64 %rd3,[src_bitmap]; ld.param.u64 %rd4,[src_valid];
    ld.param.u64 %rd5,[dst]; ld.param.u64 %rd6,[dst_off]; ld.param.u64 %rd7,[dst_valid];
    mov.u32 %r4,%tid.x; mov.u32 %r5,%ctaid.x; mov.u32 %r6,%ntid.x; mov.u32 %r7,%nctaid.x;
    mad.lo.u32 %r8,%r5,%r6,%r4; mul.lo.u32 %r9,%r7,%r6;
B_LOOP: setp.ge.u32 %p1,%r8,%r1; @%p1 bra B_DONE;
    mul.lo.u32 %r10,%r8,%r2; add.u32 %r10,%r10,%r3; mul.wide.u32 %rd8,%r10,4; add.u64 %rd9,%rd1,%rd8;
    ld.global.u32 %r11,[%rd9]; setp.eq.u32 %p2,%r11,4294967295; @%p2 bra B_NEXT;
    mov.u64 %rd10,18446744073709551615; setp.eq.u64 %p3,%rd4,%rd10; @%p3 bra B_VALID;
    shr.u32 %r12,%r11,5; mul.wide.u32 %rd11,%r12,4; add.u64 %rd12,%rd4,%rd11; ld.global.u32 %r13,[%rd12];
    and.b32 %r12,%r11,31; shr.u32 %r13,%r13,%r12; and.b32 %r13,%r13,1; setp.eq.u32 %p4,%r13,0; @%p4 bra B_NEXT;
B_VALID:
    shr.u32 %r12,%r11,5; mul.wide.u32 %rd11,%r12,4; add.u64 %rd12,%rd2,%rd3; add.u64 %rd12,%rd12,%rd11;
    ld.global.u32 %r13,[%rd12]; and.b32 %r12,%r11,31; shr.u32 %r13,%r13,%r12; and.b32 %r13,%r13,1;
    mul.wide.u32 %rd13,%r8,4; add.u64 %rd14,%rd5,%rd6; add.u64 %rd14,%rd14,%rd13; st.global.u32 [%rd14],%r13;
    shr.u32 %r14,%r8,5; mul.wide.u32 %rd15,%r14,4; add.u64 %rd16,%rd5,%rd7; add.u64 %rd16,%rd16,%rd15;
    and.b32 %r15,%r8,31; mov.u32 %r16,1; shl.b32 %r16,%r16,%r15; atom.global.or.b32 %r17,[%rd16],%r16;
B_NEXT: add.u32 %r8,%r8,%r9; bra B_LOOP;
B_DONE: ret;
}

.visible .entry gpu_db_materialize_text_lengths(
    .param .u64 coords,.param .u32 rows,.param .u32 rels,.param .u32 relation,
    .param .u64 src,.param .u64 src_offsets,.param .u64 src_bytes,.param .u64 src_bytes_len,
    .param .u64 src_rows,.param .u64 src_valid,.param .u64 lengths,.param .u64 status)
{
    .reg .pred %p<10>; .reg .b32 %r<28>; .reg .b64 %rd<40>;
    ld.param.u64 %rd1,[coords]; ld.param.u32 %r1,[rows]; ld.param.u32 %r2,[rels]; ld.param.u32 %r3,[relation];
    ld.param.u64 %rd2,[src]; ld.param.u64 %rd3,[src_offsets]; ld.param.u64 %rd4,[src_bytes];
    ld.param.u64 %rd5,[src_bytes_len]; ld.param.u64 %rd6,[src_rows]; ld.param.u64 %rd7,[src_valid];
    ld.param.u64 %rd8,[lengths]; ld.param.u64 %rd9,[status];
    mov.u32 %r4,%tid.x; mov.u32 %r5,%ctaid.x; mov.u32 %r6,%ntid.x; mov.u32 %r7,%nctaid.x;
    mad.lo.u32 %r8,%r5,%r6,%r4; mul.lo.u32 %r9,%r7,%r6;
TL_LOOP: setp.ge.u32 %p1,%r8,%r1; @%p1 bra TL_DONE; mov.u64 %rd30,0;
    mul.lo.u32 %r10,%r8,%r2; add.u32 %r10,%r10,%r3; mul.wide.u32 %rd10,%r10,4; add.u64 %rd11,%rd1,%rd10;
    ld.global.u32 %r11,[%rd11]; setp.eq.u32 %p2,%r11,4294967295; @%p2 bra TL_WRITE;
    cvt.u64.u32 %rd12,%r11; setp.ge.u64 %p3,%rd12,%rd6; @%p3 bra TL_INVALID;
    add.u64 %rd12,%rd12,2; mul.lo.u64 %rd12,%rd12,8; add.u64 %rd13,%rd3,%rd12;
    setp.lt.u64 %p3,%rd13,%rd3; @%p3 bra TL_INVALID; setp.gt.u64 %p4,%rd13,%rd4; @%p4 bra TL_INVALID;
    mov.u64 %rd14,18446744073709551615; setp.eq.u64 %p5,%rd7,%rd14; @%p5 bra TL_VALID;
    shr.u32 %r13,%r11,5; mul.wide.u32 %rd15,%r13,4; add.u64 %rd16,%rd7,%rd15; ld.global.u32 %r14,[%rd16];
    and.b32 %r12,%r11,31; shr.u32 %r14,%r14,%r12; and.b32 %r14,%r14,1; setp.eq.u32 %p4,%r14,0; @%p4 bra TL_WRITE;
TL_VALID: mul.wide.u32 %rd17,%r11,8; add.u64 %rd18,%rd2,%rd3; add.u64 %rd18,%rd18,%rd17;
    ld.global.u64 %rd19,[%rd18]; ld.global.u64 %rd20,[%rd18+8];
    setp.lt.u64 %p6,%rd20,%rd19; @%p6 bra TL_INVALID; setp.gt.u64 %p7,%rd20,%rd5; @%p7 bra TL_INVALID;
    sub.u64 %rd30,%rd20,%rd19; bra TL_WRITE;
TL_INVALID: mov.u32 %r15,1; atom.global.or.b32 %r16,[%rd9],%r15;
TL_WRITE: mul.wide.u32 %rd21,%r8,8; add.u64 %rd22,%rd8,%rd21; st.global.u64 [%rd22],%rd30;
    add.u32 %r8,%r8,%r9; bra TL_LOOP; TL_DONE: ret;
}

.visible .entry gpu_db_materialize_text_block_scan(
    .param .u64 lengths,.param .u32 rows,.param .u32 chunk,.param .u64 offsets,.param .u64 block_sums,.param .u64 status)
{
    .reg .pred %p<12>; .reg .b32 %r<32>; .reg .b64 %rd<40>;
    .shared .align 8 .b8 scan_shared[2064];
    ld.param.u64 %rd1,[lengths]; ld.param.u32 %r1,[rows]; ld.param.u32 %r6,[chunk]; ld.param.u64 %rd2,[offsets];
    ld.param.u64 %rd3,[block_sums]; ld.param.u64 %rd4,[status];
    mov.u32 %r2,%tid.x; mov.u32 %r3,%ntid.x; mov.u32 %r4,%ctaid.x; mov.u32 %r5,%nctaid.x;
    mul.lo.u32 %r7,%r4,%r6; add.u32 %r8,%r7,%r6; min.u32 %r8,%r8,%r1;
    setp.eq.u32 %p1,%r2,0; @!%p1 bra TBS_INIT_DONE; mov.u64 %rd5,0; st.shared.u64 [scan_shared+2048],%rd5;
TBS_INIT_DONE: bar.sync 0; mov.u32 %r9,%r7;
TBS_TILE: setp.ge.u32 %p2,%r9,%r8; @%p2 bra TBS_BLOCK_DONE;
    add.u32 %r10,%r9,%r2; setp.lt.u32 %p3,%r10,%r8; mov.u64 %rd6,0; @!%p3 bra TBS_LOAD_DONE;
    mul.wide.u32 %rd7,%r10,8; add.u64 %rd8,%rd1,%rd7; ld.global.u64 %rd6,[%rd8];
TBS_LOAD_DONE: mul.wide.u32 %rd9,%r2,8; mov.u64 %rd10,scan_shared; add.u64 %rd10,%rd10,%rd9; st.shared.u64 [%rd10],%rd6; bar.sync 0;
    mov.u32 %r11,1;
TBS_SCAN: setp.ge.u32 %p4,%r11,%r3; @%p4 bra TBS_SCANNED;
    ld.shared.u64 %rd11,[%rd10]; mov.u64 %rd12,0; setp.ge.u32 %p5,%r2,%r11; @!%p5 bra TBS_PREV_DONE;
    sub.u32 %r12,%r2,%r11; mul.wide.u32 %rd13,%r12,8; mov.u64 %rd14,scan_shared; add.u64 %rd14,%rd14,%rd13; ld.shared.u64 %rd12,[%rd14];
TBS_PREV_DONE: bar.sync 0; add.u64 %rd15,%rd11,%rd12; setp.lt.u64 %p6,%rd15,%rd11; @!%p6 bra TBS_NO_LOCAL_OVERFLOW;
    mov.u32 %r13,2; atom.global.or.b32 %r14,[%rd4],%r13;
TBS_NO_LOCAL_OVERFLOW: st.shared.u64 [%rd10],%rd15; bar.sync 0; shl.b32 %r11,%r11,1; bra TBS_SCAN;
TBS_SCANNED:
    sub.u32 %r14,%r8,%r9; min.u32 %r14,%r14,%r3;
    @!%p1 bra TBS_TOTAL_READY; sub.u32 %r15,%r14,1; mul.wide.u32 %rd16,%r15,8; mov.u64 %rd17,scan_shared; add.u64 %rd17,%rd17,%rd16;
    ld.shared.u64 %rd18,[%rd17]; ld.shared.u64 %rd19,[scan_shared+2048]; add.u64 %rd20,%rd19,%rd18;
    setp.lt.u64 %p7,%rd20,%rd19; @!%p7 bra TBS_STORE_TOTAL; mov.u32 %r16,2; atom.global.or.b32 %r17,[%rd4],%r16;
TBS_STORE_TOTAL: st.shared.u64 [scan_shared+2056],%rd20;
TBS_TOTAL_READY: bar.sync 0; @!%p3 bra TBS_OUTPUT_DONE;
    mov.u64 %rd21,0; setp.eq.u32 %p8,%r2,0; @%p8 bra TBS_HAVE_EXCLUSIVE; sub.u32 %r17,%r2,1;
    mul.wide.u32 %rd22,%r17,8; mov.u64 %rd23,scan_shared; add.u64 %rd23,%rd23,%rd22; ld.shared.u64 %rd21,[%rd23];
TBS_HAVE_EXCLUSIVE: ld.shared.u64 %rd24,[scan_shared+2048]; add.u64 %rd25,%rd24,%rd21;
    setp.lt.u64 %p9,%rd25,%rd24; @!%p9 bra TBS_STORE_OFFSET; mov.u32 %r18,2; atom.global.or.b32 %r19,[%rd4],%r18;
TBS_STORE_OFFSET: mul.wide.u32 %rd26,%r10,8; add.u64 %rd27,%rd2,%rd26; st.global.u64 [%rd27],%rd25;
TBS_OUTPUT_DONE: bar.sync 0; @!%p1 bra TBS_BASE_DONE; ld.shared.u64 %rd28,[scan_shared+2056]; st.shared.u64 [scan_shared+2048],%rd28;
TBS_BASE_DONE: bar.sync 0; add.u32 %r9,%r9,%r3; bra TBS_TILE;
TBS_BLOCK_DONE: @!%p1 bra TBS_DONE; ld.shared.u64 %rd29,[scan_shared+2048]; mul.wide.u32 %rd30,%r4,8; add.u64 %rd31,%rd3,%rd30; st.global.u64 [%rd31],%rd29;
TBS_DONE: ret;
}

.visible .entry gpu_db_materialize_text_block_prefix(
    .param .u64 block_sums,.param .u32 blocks,.param .u32 rows,.param .u64 offsets,.param .u64 metadata)
{
    .reg .pred %p<5>; .reg .b32 %r<12>; .reg .b64 %rd<24>;
    mov.u32 %r1,%tid.x; mov.u32 %r2,%ctaid.x; setp.ne.u32 %p1,%r1,0; @%p1 bra TBP_DONE; setp.ne.u32 %p2,%r2,0; @%p2 bra TBP_DONE;
    ld.param.u64 %rd1,[block_sums]; ld.param.u32 %r3,[blocks]; ld.param.u32 %r4,[rows]; ld.param.u64 %rd2,[offsets]; ld.param.u64 %rd3,[metadata];
    mov.u64 %rd4,0; mov.u32 %r5,0;
TBP_LOOP: setp.ge.u32 %p3,%r5,%r3; @%p3 bra TBP_TOTAL; mul.wide.u32 %rd5,%r5,8; add.u64 %rd6,%rd1,%rd5;
    ld.global.u64 %rd7,[%rd6]; st.global.u64 [%rd6],%rd4; add.u64 %rd8,%rd4,%rd7; setp.lt.u64 %p4,%rd8,%rd4; @!%p4 bra TBP_NEXT;
    mov.u32 %r6,2; atom.global.or.b32 %r7,[%rd3],%r6;
TBP_NEXT: mov.u64 %rd4,%rd8; add.u32 %r5,%r5,1; bra TBP_LOOP;
TBP_TOTAL: st.global.u64 [%rd3+8],%rd4; mul.wide.u32 %rd9,%r4,8; add.u64 %rd10,%rd2,%rd9; st.global.u64 [%rd10],%rd4;
TBP_DONE: ret;
}

.visible .entry gpu_db_materialize_text_add_bases(
    .param .u64 offsets,.param .u32 rows,.param .u32 chunk,.param .u64 block_bases,.param .u64 status)
{
    .reg .pred %p<4>; .reg .b32 %r<16>; .reg .b64 %rd<20>;
    ld.param.u64 %rd1,[offsets]; ld.param.u32 %r1,[rows]; ld.param.u32 %r6,[chunk]; ld.param.u64 %rd2,[block_bases]; ld.param.u64 %rd3,[status];
    mov.u32 %r2,%tid.x; mov.u32 %r3,%ntid.x; mov.u32 %r4,%ctaid.x; mov.u32 %r5,%nctaid.x;
    mul.lo.u32 %r7,%r4,%r6; add.u32 %r8,%r7,%r6; min.u32 %r8,%r8,%r1;
    mul.wide.u32 %rd4,%r4,8; add.u64 %rd5,%rd2,%rd4; ld.global.u64 %rd6,[%rd5]; add.u32 %r9,%r7,%r2;
TBA_LOOP: setp.ge.u32 %p1,%r9,%r8; @%p1 bra TBA_DONE; mul.wide.u32 %rd7,%r9,8; add.u64 %rd8,%rd1,%rd7; ld.global.u64 %rd9,[%rd8];
    add.u64 %rd10,%rd9,%rd6; setp.lt.u64 %p2,%rd10,%rd9; @!%p2 bra TBA_STORE; mov.u32 %r10,2; atom.global.or.b32 %r11,[%rd3],%r10;
TBA_STORE: st.global.u64 [%rd8],%rd10; add.u32 %r9,%r9,%r3; bra TBA_LOOP;
TBA_DONE: ret;
}

.visible .entry gpu_db_materialize_text_copy(
    .param .u64 coords,.param .u32 rows,.param .u32 rels,.param .u32 relation,
    .param .u64 src,.param .u64 src_offsets,.param .u64 src_bytes,.param .u64 src_valid,
    .param .u64 dst,.param .u64 dst_offsets,.param .u64 dst_bytes,.param .u64 dst_valid)
{
    .reg .pred %p<7>; .reg .b32 %r<28>; .reg .b64 %rd<48>;
    ld.param.u64 %rd1,[coords]; ld.param.u32 %r1,[rows]; ld.param.u32 %r2,[rels]; ld.param.u32 %r3,[relation];
    ld.param.u64 %rd2,[src]; ld.param.u64 %rd3,[src_offsets]; ld.param.u64 %rd4,[src_bytes]; ld.param.u64 %rd5,[src_valid];
    ld.param.u64 %rd6,[dst]; ld.param.u64 %rd7,[dst_offsets]; ld.param.u64 %rd8,[dst_bytes]; ld.param.u64 %rd9,[dst_valid];
    mov.u32 %r4,%tid.x; mov.u32 %r5,%ctaid.x; mov.u32 %r6,%ntid.x; mov.u32 %r7,%nctaid.x;
    mad.lo.u32 %r8,%r5,%r6,%r4; mul.lo.u32 %r9,%r7,%r6;
TC_LOOP: setp.ge.u32 %p1,%r8,%r1; @%p1 bra TC_DONE;
    mul.lo.u32 %r10,%r8,%r2; add.u32 %r10,%r10,%r3; mul.wide.u32 %rd10,%r10,4; add.u64 %rd11,%rd1,%rd10;
    ld.global.u32 %r11,[%rd11]; setp.eq.u32 %p2,%r11,4294967295; @%p2 bra TC_NEXT;
    mov.u64 %rd12,18446744073709551615; setp.eq.u64 %p3,%rd5,%rd12; @%p3 bra TC_VALID;
    shr.u32 %r12,%r11,5; mul.wide.u32 %rd13,%r12,4; add.u64 %rd14,%rd5,%rd13; ld.global.u32 %r13,[%rd14];
    and.b32 %r12,%r11,31; shr.u32 %r13,%r13,%r12; and.b32 %r13,%r13,1; setp.eq.u32 %p4,%r13,0; @%p4 bra TC_NEXT;
TC_VALID:
    mul.wide.u32 %rd15,%r11,8; add.u64 %rd16,%rd2,%rd3; add.u64 %rd16,%rd16,%rd15;
    ld.global.u64 %rd17,[%rd16]; ld.global.u64 %rd18,[%rd16+8]; sub.u64 %rd19,%rd18,%rd17;
    mul.wide.u32 %rd20,%r8,8; add.u64 %rd21,%rd6,%rd7; add.u64 %rd21,%rd21,%rd20; ld.global.u64 %rd22,[%rd21];
    add.u64 %rd23,%rd2,%rd4; add.u64 %rd23,%rd23,%rd17; add.u64 %rd24,%rd6,%rd8; add.u64 %rd24,%rd24,%rd22;
    mov.u64 %rd25,0;
TC_BYTES: setp.ge.u64 %p5,%rd25,%rd19; @%p5 bra TC_MARK; add.u64 %rd26,%rd23,%rd25; add.u64 %rd27,%rd24,%rd25;
    ld.global.u8 %r14,[%rd26]; st.global.u8 [%rd27],%r14; add.u64 %rd25,%rd25,1; bra TC_BYTES;
TC_MARK: shr.u32 %r15,%r8,5; mul.wide.u32 %rd28,%r15,4; add.u64 %rd29,%rd6,%rd9; add.u64 %rd29,%rd29,%rd28;
    and.b32 %r16,%r8,31; mov.u32 %r17,1; shl.b32 %r17,%r17,%r16; atom.global.or.b32 %r18,[%rd29],%r17;
TC_NEXT: add.u32 %r8,%r8,%r9; bra TC_LOOP; TC_DONE: ret;
}
"#;
    if columns.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    validate_materialized_coordinate_extent(coordinates.row_count, coordinates.relation_count)?;
    if columns.iter().any(|column| {
        let relation = match column {
            CudaMaterializeJoinColumn::Fixed { relation, .. }
            | CudaMaterializeJoinColumn::Bool { relation, .. }
            | CudaMaterializeJoinColumn::Text { relation, .. } => *relation,
        };
        relation >= coordinates.relation_count
    }) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            coordinates.relation_count as usize,
        ));
    }
    if coordinates.relation_row_counts.len() != coordinates.relation_count as usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            coordinates.relation_row_counts.len(),
        ));
    }
    for column in columns {
        let (relation, payload, value_offset, validity_offset, width, text) = match *column {
            CudaMaterializeJoinColumn::Fixed {
                relation,
                payload,
                byte_offset,
                validity_bitmap_offset,
                width,
            } => (
                relation,
                payload,
                byte_offset,
                validity_bitmap_offset,
                u64::from(width),
                None,
            ),
            CudaMaterializeJoinColumn::Bool {
                relation,
                payload,
                bitmap_byte_offset,
                validity_bitmap_offset,
            } => (
                relation,
                payload,
                bitmap_byte_offset,
                validity_bitmap_offset,
                0,
                None,
            ),
            CudaMaterializeJoinColumn::Text {
                relation,
                payload,
                offsets_byte_offset,
                bytes_byte_offset,
                bytes_len,
                validity_bitmap_offset,
            } => (
                relation,
                payload,
                offsets_byte_offset,
                validity_bitmap_offset,
                8,
                Some((bytes_byte_offset, bytes_len)),
            ),
        };
        let source_rows = u64::from(coordinates.relation_row_counts[relation as usize]);
        let same_context = std::ptr::eq(payload.primary(), ctx.primary());
        let allocated = payload.metadata.allocated_bytes;
        let validity_end = validity_offset.and_then(|offset| {
            source_rows
                .div_ceil(32)
                .checked_mul(4)
                .and_then(|bytes| offset.checked_add(bytes))
        });
        let value_end = if matches!(column, CudaMaterializeJoinColumn::Bool { .. }) {
            source_rows
                .div_ceil(32)
                .checked_mul(4)
                .and_then(|bytes| value_offset.checked_add(bytes))
        } else if text.is_some() {
            source_rows
                .checked_add(1)
                .and_then(|rows| rows.checked_mul(8))
                .and_then(|bytes| value_offset.checked_add(bytes))
        } else {
            source_rows
                .checked_mul(width)
                .and_then(|bytes| value_offset.checked_add(bytes))
        };
        let text_valid = text.is_none_or(|(offset, len)| {
            offset.checked_add(len).is_some_and(|end| end <= allocated)
        });
        let aligned = match column {
            CudaMaterializeJoinColumn::Fixed { width, .. } => {
                // Resident sections are packed on the engine's four-byte layout boundary. CUDA
                // fixed-width copies use four-byte words for i64/i128 just as the resident
                // expression and comparison families do; requiring natural 8-byte alignment here
                // would incorrectly reject valid int8/numeric/UUID columns after an odd section.
                matches!(width, 4 | 8 | 16) && value_offset.is_multiple_of(4)
            }
            CudaMaterializeJoinColumn::Bool { .. } => value_offset.is_multiple_of(4),
            CudaMaterializeJoinColumn::Text { .. } => value_offset.is_multiple_of(8),
        };
        if !same_context
            || !aligned
            || value_end.is_none_or(|end| end > allocated)
            || validity_offset.is_some_and(|offset| {
                !offset.is_multiple_of(4) || validity_end.is_none_or(|end| end > allocated)
            })
            || !text_valid
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                source_rows as usize,
            ));
        }
    }
    if coordinates.row_count > 0 && coordinates.coordinates.is_none() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            coordinates.row_count as usize,
        ));
    }
    if coordinates
        .coordinates
        .as_ref()
        .is_some_and(|buffer| !std::ptr::eq(buffer.primary.as_ref(), ctx.primary()))
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            coordinates.row_count as usize,
        ));
    }
    let source_coords = coordinates
        .coordinates
        .as_ref()
        .map_or(0, |buffer| buffer.ptr);
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
    let dtod = unsafe {
        primary
            .lib()
            .get::<CuMemcpyDtoD>(b"cuMemcpyDtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyDtoD>(b"cuMemcpyDtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let memset = unsafe {
        primary
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let fixed_fn = primary.cached_function(c"gpu_db_materialize_fixed", &ptx)?;
    let bool_fn = primary.cached_function(c"gpu_db_materialize_bool", &ptx)?;
    let text_lengths_fn = primary.cached_function(c"gpu_db_materialize_text_lengths", &ptx)?;
    let text_block_scan_fn =
        primary.cached_function(c"gpu_db_materialize_text_block_scan", &ptx)?;
    let text_block_prefix_fn =
        primary.cached_function(c"gpu_db_materialize_text_block_prefix", &ptx)?;
    let text_add_bases_fn = primary.cached_function(c"gpu_db_materialize_text_add_bases", &ptx)?;
    let text_copy_fn = primary.cached_function(c"gpu_db_materialize_text_copy", &ptx)?;
    let n = coordinates.row_count as usize;
    let grid = coordinates.row_count.div_ceil(256).clamp(1, 65_535);
    let chunk = text_scan_partition_chunk(coordinates.row_count, grid)?;
    let mut text_offsets_device: Vec<Option<PooledDeviceBufferOwned>> =
        (0..columns.len()).map(|_| None).collect();
    let mut text_lengths = vec![0_u64; columns.len()];
    if n > 0 {
        for (index, column) in columns.iter().enumerate() {
            let CudaMaterializeJoinColumn::Text {
                relation,
                payload,
                offsets_byte_offset,
                bytes_byte_offset,
                bytes_len,
                validity_bitmap_offset,
            } = *column
            else {
                continue;
            };
            let source_rows = u64::from(coordinates.relation_row_counts[relation as usize]);
            let lengths_bytes = n
                .checked_mul(8)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
            let offsets_bytes = n
                .checked_add(1)
                .and_then(|count| count.checked_mul(8))
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
            let block_count = grid as usize;
            let block_bytes = block_count
                .checked_mul(8)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(block_count))?;
            // Every fallible allocation happens before the first enqueue so no error can return a
            // buffer to the shared pool while an earlier kernel is still writing it.
            let lengths_dev = primary.lease_device_buffer_owned(lengths_bytes)?;
            let offsets_dev = primary.lease_device_buffer_owned(offsets_bytes)?;
            let block_sums_dev = primary.lease_device_buffer_owned(block_bytes)?;
            let metadata_dev = primary.lease_device_buffer_owned(16)?;
            check_cuda(unsafe { memset(metadata_dev.ptr, 0, 16) })?;
            let drain_err = |err| {
                let _ = ctx.synchronize_default_stream();
                err
            };
            let mut a0 = source_coords;
            let mut a1 = coordinates.row_count;
            let mut a2 = coordinates.relation_count;
            let mut a3 = relation;
            let mut a4 = payload.device_ptr;
            let mut a5 = offsets_byte_offset;
            let mut a6 = bytes_byte_offset;
            let mut a7 = bytes_len;
            let mut a8 = source_rows;
            let mut a9 = validity_bitmap_offset.map_or(u64::MAX, |off| payload.device_ptr + off);
            let mut a10 = lengths_dev.ptr;
            let mut a11 = metadata_dev.ptr;
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
                (&mut a9 as *mut u64).cast(),
                (&mut a10 as *mut u64).cast(),
                (&mut a11 as *mut u64).cast(),
            ];
            check_cuda(unsafe {
                launch(
                    text_lengths_fn,
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
            })
            .map_err(&drain_err)?;
            let mut s0 = lengths_dev.ptr;
            let mut s1 = coordinates.row_count;
            let mut s2 = chunk;
            let mut s3 = offsets_dev.ptr;
            let mut s4 = block_sums_dev.ptr;
            let mut s5 = metadata_dev.ptr;
            let mut scan_args = [
                (&mut s0 as *mut u64).cast(),
                (&mut s1 as *mut u32).cast(),
                (&mut s2 as *mut u32).cast(),
                (&mut s3 as *mut u64).cast(),
                (&mut s4 as *mut u64).cast(),
                (&mut s5 as *mut u64).cast(),
            ];
            check_cuda(unsafe {
                launch(
                    text_block_scan_fn,
                    grid,
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
            })
            .map_err(&drain_err)?;
            let mut p0 = block_sums_dev.ptr;
            let mut p1 = grid;
            let mut p2 = coordinates.row_count;
            let mut p3 = offsets_dev.ptr;
            let mut p4 = metadata_dev.ptr;
            let mut prefix_args = [
                (&mut p0 as *mut u64).cast(),
                (&mut p1 as *mut u32).cast(),
                (&mut p2 as *mut u32).cast(),
                (&mut p3 as *mut u64).cast(),
                (&mut p4 as *mut u64).cast(),
            ];
            check_cuda(unsafe {
                launch(
                    text_block_prefix_fn,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    prefix_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            })
            .map_err(&drain_err)?;
            let mut b0 = offsets_dev.ptr;
            let mut b1 = coordinates.row_count;
            let mut b2 = chunk;
            let mut b3 = block_sums_dev.ptr;
            let mut b4 = metadata_dev.ptr;
            let mut base_args = [
                (&mut b0 as *mut u64).cast(),
                (&mut b1 as *mut u32).cast(),
                (&mut b2 as *mut u32).cast(),
                (&mut b3 as *mut u64).cast(),
                (&mut b4 as *mut u64).cast(),
            ];
            check_cuda(unsafe {
                launch(
                    text_add_bases_fn,
                    grid,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    base_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            })
            .map_err(&drain_err)?;
            // Allocation sizing needs only status + final byte cardinality. Every per-row length
            // and offset remains device-resident and is copied D2D into the materialized relation.
            let mut metadata = [0_u64; 2];
            check_cuda(unsafe {
                dtoh(
                    metadata.as_mut_ptr().cast(),
                    metadata_dev.ptr,
                    std::mem::size_of_val(&metadata),
                )
            })
            .map_err(&drain_err)?;
            if metadata[0] != 0 {
                return Err(CudaRuntimeProbeError::InvalidInputLength(n));
            }
            text_lengths[index] = metadata[1];
            text_offsets_device[index] = Some(offsets_dev);
        }
    }
    let align = |value: u64, alignment: u64| {
        value
            .checked_add(alignment - 1)
            .map(|sum| sum / alignment * alignment)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))
    };
    let add = |value: u64, increment: u64| {
        value
            .checked_add(increment)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))
    };
    let validity_bytes = u64::from(coordinates.row_count).div_ceil(32) * 4;
    let mut cursor = 0_u64;
    let mut layouts = Vec::with_capacity(columns.len());
    for (index, column) in columns.iter().enumerate() {
        match *column {
            CudaMaterializeJoinColumn::Fixed { width, .. } => {
                if !matches!(width, 4 | 8 | 16) {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(width as usize));
                }
                cursor = align(cursor, u64::from(width.min(8)))?;
                let value = cursor;
                cursor = add(cursor, u64::from(coordinates.row_count) * u64::from(width))?;
                cursor = align(cursor, 4)?;
                let valid = cursor;
                cursor = add(cursor, validity_bytes)?;
                layouts.push(CudaMaterializedColumnLayout {
                    kind: CudaMaterializedColumnKind::Fixed { width },
                    value_byte_offset: value,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                    validity_bitmap_offset: valid,
                });
            }
            CudaMaterializeJoinColumn::Bool { .. } => {
                cursor = align(cursor, 4)?;
                let value = cursor;
                cursor = add(cursor, u64::from(coordinates.row_count) * 4)?;
                cursor = align(cursor, 4)?;
                let valid = cursor;
                cursor = add(cursor, validity_bytes)?;
                layouts.push(CudaMaterializedColumnLayout {
                    kind: CudaMaterializedColumnKind::Fixed { width: 4 },
                    value_byte_offset: value,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                    validity_bitmap_offset: valid,
                });
            }
            CudaMaterializeJoinColumn::Text { .. } => {
                cursor = align(cursor, 8)?;
                let offsets = cursor;
                cursor = add(cursor, (u64::from(coordinates.row_count) + 1) * 8)?;
                let text_len = text_lengths[index];
                let bytes = cursor;
                cursor = add(cursor, text_len)?;
                cursor = align(cursor, 4)?;
                let valid = cursor;
                cursor = add(cursor, validity_bytes)?;
                layouts.push(CudaMaterializedColumnLayout {
                    kind: CudaMaterializedColumnKind::Text,
                    value_byte_offset: offsets,
                    text_bytes_byte_offset: Some(bytes),
                    text_bytes_len: text_len,
                    validity_bitmap_offset: valid,
                });
            }
        }
    }
    let allocated = cursor.max(1);
    let mut ptr = 0_u64;
    let allocated_usize = usize::try_from(allocated)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let reservation = CudaAllocationScope::reserve_external(allocated_usize)?;
    check_cuda(unsafe { (primary.cu_mem_alloc)(&mut ptr, allocated_usize) })?;
    let memory = CudaResidentDeviceMemory::from_raw_parts(
        CudaDeviceMemoryProof {
            gpu_id: ctx.metadata.gpu_id,
            device_name: ctx.metadata.device_name.clone(),
            allocated_bytes: allocated,
            copied_bytes: 0,
            retained: true,
        },
        ptr,
        Arc::clone(&primary),
    );
    check_cuda(unsafe { memset(memory.device_ptr, 0, allocated_usize) })?;
    let drain_err = |err| {
        let _ = ctx.synchronize_default_stream();
        err
    };
    for (index, column) in columns.iter().enumerate() {
        let layout = layouts[index];
        match *column {
            CudaMaterializeJoinColumn::Fixed {
                relation,
                payload,
                byte_offset,
                validity_bitmap_offset,
                width,
            } => {
                let mut a0 = source_coords;
                let mut a1 = coordinates.row_count;
                let mut a2 = coordinates.relation_count;
                let mut a3 = relation;
                let mut a4 = payload.device_ptr;
                let mut a5 = byte_offset;
                let mut a6 =
                    validity_bitmap_offset.map_or(u64::MAX, |off| payload.device_ptr + off);
                let mut a7 = u32::from(width);
                let mut a8 = memory.device_ptr;
                let mut a9 = layout.value_byte_offset;
                let mut a10 = layout.validity_bitmap_offset;
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
                    (&mut a10 as *mut u64).cast(),
                ];
                if n > 0 {
                    check_cuda(unsafe {
                        launch(
                            fixed_fn,
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
                    })
                    .map_err(&drain_err)?;
                }
            }
            CudaMaterializeJoinColumn::Bool {
                relation,
                payload,
                bitmap_byte_offset,
                validity_bitmap_offset,
            } => {
                let mut a0 = source_coords;
                let mut a1 = coordinates.row_count;
                let mut a2 = coordinates.relation_count;
                let mut a3 = relation;
                let mut a4 = payload.device_ptr;
                let mut a5 = bitmap_byte_offset;
                let mut a6 =
                    validity_bitmap_offset.map_or(u64::MAX, |off| payload.device_ptr + off);
                let mut a7 = memory.device_ptr;
                let mut a8 = layout.value_byte_offset;
                let mut a9 = layout.validity_bitmap_offset;
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
                    (&mut a9 as *mut u64).cast(),
                ];
                if n > 0 {
                    check_cuda(unsafe {
                        launch(
                            bool_fn,
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
                    })
                    .map_err(&drain_err)?;
                }
            }
            CudaMaterializeJoinColumn::Text {
                relation,
                payload,
                offsets_byte_offset,
                bytes_byte_offset,
                validity_bitmap_offset,
                ..
            } => {
                if let Some(offsets) = text_offsets_device[index].as_ref() {
                    check_cuda(unsafe {
                        dtod(
                            memory.device_ptr + layout.value_byte_offset,
                            offsets.ptr,
                            (n + 1) * 8,
                        )
                    })
                    .map_err(&drain_err)?;
                }
                let mut a0 = source_coords;
                let mut a1 = coordinates.row_count;
                let mut a2 = coordinates.relation_count;
                let mut a3 = relation;
                let mut a4 = payload.device_ptr;
                let mut a5 = offsets_byte_offset;
                let mut a6 = bytes_byte_offset;
                let mut a7 =
                    validity_bitmap_offset.map_or(u64::MAX, |off| payload.device_ptr + off);
                let mut a8 = memory.device_ptr;
                let mut a9 = layout.value_byte_offset;
                let mut a10 = layout.text_bytes_byte_offset.unwrap();
                let mut a11 = layout.validity_bitmap_offset;
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
                    (&mut a9 as *mut u64).cast(),
                    (&mut a10 as *mut u64).cast(),
                    (&mut a11 as *mut u64).cast(),
                ];
                if n > 0 {
                    check_cuda(unsafe {
                        launch(
                            text_copy_fn,
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
                    })
                    .map_err(&drain_err)?;
                }
            }
        }
    }
    ctx.synchronize_default_stream()?;
    Ok(CudaMaterializedRelation {
        memory,
        _reservation: reservation,
        columns: layouts,
        row_count: coordinates.row_count,
    })
}

#[cfg(test)]
mod tests {
    use super::{text_scan_partition_chunk, validate_materialized_coordinate_extent};

    #[test]
    fn text_scan_partition_chunk_is_total_at_u32_boundary() {
        assert_eq!(text_scan_partition_chunk(0, 1).unwrap(), 0);
        assert_eq!(text_scan_partition_chunk(1_025, 5).unwrap(), 205);
        assert_eq!(
            text_scan_partition_chunk(i32::MAX as u32, 65_535).unwrap(),
            32_769
        );
        assert!(text_scan_partition_chunk(1, 0).is_err());
        assert!(validate_materialized_coordinate_extent(i32::MAX as u32, 1).is_ok());
        assert!(validate_materialized_coordinate_extent(i32::MAX as u32 + 1, 1).is_err());
        assert!(validate_materialized_coordinate_extent(1_500_000_000, 3).is_err());
        assert!(validate_materialized_coordinate_extent(1, 0).is_err());
    }
}
