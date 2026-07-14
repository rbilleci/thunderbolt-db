//! Device-resident join materialization ownership.

use std::{
    os::raw::c_void,
    sync::{Arc, Mutex},
};

use super::{
    check_cuda, CudaDeviceMemoryProof, CudaJoinCoordinatesU32, CudaJoinPayloadKey,
    CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError,
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

#[derive(Debug, Clone, Copy)]
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
    columns: Vec<CudaMaterializedColumnLayout>,
    row_count: u32,
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
    let mut ptr = 0;
    check_cuda(unsafe { (primary.cu_mem_alloc)(&mut ptr, allocated as usize) })?;
    let memory = CudaResidentDeviceMemory {
        metadata: CudaDeviceMemoryProof {
            gpu_id: ctx.metadata.gpu_id,
            device_name: ctx.metadata.device_name.clone(),
            allocated_bytes: allocated,
            copied_bytes: 0,
            retained: true,
        },
        device_ptr: ptr,
        primary: Arc::clone(&primary),
        last_kernel_event_elapsed_us: Mutex::new(None),
    };
    let memset = unsafe {
        primary
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { memset(memory.device_ptr, 0, allocated as usize) })?;
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
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
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
    ld.global.u64 %rd15,[%rd13]; st.global.u64 [%rd14],%rd15;
    setp.eq.u32 %p6,%r4,16; @!%p6 bra F_MARK;
    ld.global.u64 %rd15,[%rd13+8]; st.global.u64 [%rd14+8],%rd15; bra F_MARK;
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
    .param .u64 src,.param .u64 src_offsets,.param .u64 src_valid,.param .u64 lengths)
{
    .reg .pred %p<6>; .reg .b32 %r<24>; .reg .b64 %rd<32>;
    ld.param.u64 %rd1,[coords]; ld.param.u32 %r1,[rows]; ld.param.u32 %r2,[rels]; ld.param.u32 %r3,[relation];
    ld.param.u64 %rd2,[src]; ld.param.u64 %rd3,[src_offsets]; ld.param.u64 %rd4,[src_valid]; ld.param.u64 %rd5,[lengths];
    mov.u32 %r4,%tid.x; mov.u32 %r5,%ctaid.x; mov.u32 %r6,%ntid.x; mov.u32 %r7,%nctaid.x;
    mad.lo.u32 %r8,%r5,%r6,%r4; mul.lo.u32 %r9,%r7,%r6;
TL_LOOP: setp.ge.u32 %p1,%r8,%r1; @%p1 bra TL_DONE; mov.u64 %rd6,0;
    mul.lo.u32 %r10,%r8,%r2; add.u32 %r10,%r10,%r3; mul.wide.u32 %rd7,%r10,4; add.u64 %rd8,%rd1,%rd7;
    ld.global.u32 %r11,[%rd8]; setp.eq.u32 %p2,%r11,4294967295; @%p2 bra TL_WRITE;
    mov.u64 %rd9,18446744073709551615; setp.eq.u64 %p3,%rd4,%rd9; @%p3 bra TL_VALID;
    shr.u32 %r12,%r11,5; mul.wide.u32 %rd10,%r12,4; add.u64 %rd11,%rd4,%rd10; ld.global.u32 %r13,[%rd11];
    and.b32 %r12,%r11,31; shr.u32 %r13,%r13,%r12; and.b32 %r13,%r13,1; setp.eq.u32 %p4,%r13,0; @%p4 bra TL_WRITE;
TL_VALID: mul.wide.u32 %rd12,%r11,8; add.u64 %rd13,%rd2,%rd3; add.u64 %rd13,%rd13,%rd12;
    ld.global.u64 %rd14,[%rd13]; ld.global.u64 %rd15,[%rd13+8]; sub.u64 %rd6,%rd15,%rd14;
TL_WRITE: mul.wide.u32 %rd16,%r8,8; add.u64 %rd17,%rd5,%rd16; st.global.u64 [%rd17],%rd6;
    add.u32 %r8,%r8,%r9; bra TL_LOOP; TL_DONE: ret;
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
    if coordinates.row_count > 0 && coordinates.coordinates.is_none() {
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
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let fixed_fn = primary.cached_function(c"gpu_db_materialize_fixed", &ptx)?;
    let bool_fn = primary.cached_function(c"gpu_db_materialize_bool", &ptx)?;
    let text_lengths_fn = primary.cached_function(c"gpu_db_materialize_text_lengths", &ptx)?;
    let text_copy_fn = primary.cached_function(c"gpu_db_materialize_text_copy", &ptx)?;
    let n = coordinates.row_count as usize;
    let grid = coordinates.row_count.div_ceil(256).clamp(1, 65_535);
    let mut text_offsets: Vec<Option<Vec<u64>>> = vec![None; columns.len()];
    if n > 0 {
        for (index, column) in columns.iter().enumerate() {
            let CudaMaterializeJoinColumn::Text {
                relation,
                payload,
                offsets_byte_offset,
                validity_bitmap_offset,
                ..
            } = *column
            else {
                continue;
            };
            let lengths_dev = primary.lease_device_buffer_owned(n * 8)?;
            let mut a0 = source_coords;
            let mut a1 = coordinates.row_count;
            let mut a2 = coordinates.relation_count;
            let mut a3 = relation;
            let mut a4 = payload.device_ptr;
            let mut a5 = offsets_byte_offset;
            let mut a6 = validity_bitmap_offset.map_or(u64::MAX, |off| payload.device_ptr + off);
            let mut a7 = lengths_dev.ptr;
            let mut args = [
                (&mut a0 as *mut u64).cast(),
                (&mut a1 as *mut u32).cast(),
                (&mut a2 as *mut u32).cast(),
                (&mut a3 as *mut u32).cast(),
                (&mut a4 as *mut u64).cast(),
                (&mut a5 as *mut u64).cast(),
                (&mut a6 as *mut u64).cast(),
                (&mut a7 as *mut u64).cast(),
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
            })?;
            let mut lengths = vec![0_u64; n];
            check_cuda(unsafe { dtoh(lengths.as_mut_ptr().cast(), lengths_dev.ptr, n * 8) })?;
            let mut offsets = Vec::with_capacity(n + 1);
            offsets.push(0_u64);
            for len in lengths {
                offsets.push(
                    offsets
                        .last()
                        .copied()
                        .unwrap()
                        .checked_add(len)
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
                );
            }
            text_offsets[index] = Some(offsets);
        }
    }
    let align = |value: u64, alignment: u64| value.div_ceil(alignment) * alignment;
    let validity_bytes = u64::from(coordinates.row_count).div_ceil(32) * 4;
    let mut cursor = 0_u64;
    let mut layouts = Vec::with_capacity(columns.len());
    for (index, column) in columns.iter().enumerate() {
        match *column {
            CudaMaterializeJoinColumn::Fixed { width, .. } => {
                if !matches!(width, 4 | 8 | 16) {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(width as usize));
                }
                cursor = align(cursor, u64::from(width.min(8)));
                let value = cursor;
                cursor = cursor.saturating_add(u64::from(coordinates.row_count) * u64::from(width));
                cursor = align(cursor, 4);
                let valid = cursor;
                cursor = cursor.saturating_add(validity_bytes);
                layouts.push(CudaMaterializedColumnLayout {
                    kind: CudaMaterializedColumnKind::Fixed { width },
                    value_byte_offset: value,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                    validity_bitmap_offset: valid,
                });
            }
            CudaMaterializeJoinColumn::Bool { .. } => {
                cursor = align(cursor, 4);
                let value = cursor;
                cursor = cursor.saturating_add(u64::from(coordinates.row_count) * 4);
                cursor = align(cursor, 4);
                let valid = cursor;
                cursor = cursor.saturating_add(validity_bytes);
                layouts.push(CudaMaterializedColumnLayout {
                    kind: CudaMaterializedColumnKind::Fixed { width: 4 },
                    value_byte_offset: value,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                    validity_bitmap_offset: valid,
                });
            }
            CudaMaterializeJoinColumn::Text { .. } => {
                cursor = align(cursor, 8);
                let offsets = cursor;
                cursor = cursor.saturating_add((u64::from(coordinates.row_count) + 1) * 8);
                let text_len = text_offsets[index]
                    .as_ref()
                    .and_then(|v| v.last())
                    .copied()
                    .unwrap_or(0);
                let bytes = cursor;
                cursor = cursor.saturating_add(text_len);
                cursor = align(cursor, 4);
                let valid = cursor;
                cursor = cursor.saturating_add(validity_bytes);
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
    check_cuda(unsafe { (primary.cu_mem_alloc)(&mut ptr, allocated as usize) })?;
    let memory = CudaResidentDeviceMemory {
        metadata: CudaDeviceMemoryProof {
            gpu_id: ctx.metadata.gpu_id,
            device_name: ctx.metadata.device_name.clone(),
            allocated_bytes: allocated,
            copied_bytes: 0,
            retained: true,
        },
        device_ptr: ptr,
        primary: Arc::clone(&primary),
        last_kernel_event_elapsed_us: Mutex::new(None),
    };
    check_cuda(unsafe { memset(memory.device_ptr, 0, allocated as usize) })?;
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
                    })?;
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
                    })?;
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
                let offsets = text_offsets[index]
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| vec![0]);
                check_cuda(unsafe {
                    htod(
                        memory.device_ptr + layout.value_byte_offset,
                        offsets.as_ptr().cast(),
                        offsets.len() * 8,
                    )
                })?;
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
                    })?;
                }
            }
        }
    }
    Ok(CudaMaterializedRelation {
        memory,
        columns: layouts,
        row_count: coordinates.row_count,
    })
}
