use std::os::raw::c_void;

use super::{
    check_cuda, launch_cuda_buffer_i32_compare_indices_ordered,
    launch_cuda_owned_i32_compare_indices_ordered, run_resident_arith_program,
    CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError, ExprStep,
    ExprTerminal, PooledBufferLease, PooledDeviceBufferOwned, Probe, ResidentElemType,
};

impl CudaResidentDeviceMemory {
    /// Run a boolean predicate bytecode `program` (comparisons producing masks, combined by
    /// `MaskBinary` AND/OR) to one mask buffer, then compact it to the matching row indices. The
    /// general boolean-predicate VM behind the engine's `AND`/`OR`/`Ne` lowering. `elem` selects the
    /// value-buffer element type (int4 / int8 — the type matrix, doc 19); mask/compact are type-agnostic.
    pub fn run_expr_predicate_filter(
        &self,
        program: &[ExprStep],
        row_count: u64,
        elem: ResidentElemType,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_expr_predicate_filter(self, program, &[], row_count, elem)
    }

    /// `run_expr_predicate_filter` with the out-of-band varlen needles a `TextEqMask` step references
    /// (`text_needles[needle_idx]`). For text `IN` / multi-text-condition WHERE: the text comparisons
    /// run on the GPU as mask steps the VM combines with AND/OR (no host-side relational filtering).
    pub fn run_expr_predicate_filter_with_text(
        &self,
        program: &[ExprStep],
        text_needles: &[Vec<u8>],
        row_count: u64,
        elem: ResidentElemType,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_expr_predicate_filter(self, program, text_needles, row_count, elem)
    }

    /// Evaluate the boolean predicate VM but retain its TRUE/FALSE/UNKNOWN mask on the device for a
    /// following relational operator.  Only launch metadata is returned to the host.
    pub fn run_expr_predicate_mask_with_text(
        &self,
        program: &[ExprStep],
        text_needles: &[Vec<u8>],
        row_count: u32,
        elem: ResidentElemType,
    ) -> Result<CudaPredicateMaskI32, CudaRuntimeProbeError> {
        launch_cuda_resident_expr_predicate_mask(self, program, text_needles, row_count, elem)
    }

    /// A scheduler block range becomes a device mask without uploading an O(block_rows) index vector.
    pub fn row_range_mask_u32(
        &self,
        row_count: u32,
        start: u32,
        end: u32,
    ) -> Result<CudaPredicateMaskI32, CudaRuntimeProbeError> {
        launch_cuda_row_range_mask_u32(self, row_count, start, end)
    }

    /// Materialize an ascending device-generated row range as indices. This is the all-slots
    /// terminal for predicate-free DML: the host receives only the compacted result, never builds
    /// or uploads an O(rows) identity vector.
    pub fn row_range_indices_u32(
        &self,
        row_count: u32,
        start: u32,
        end: u32,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        let mask = launch_cuda_row_range_mask_u32(self, row_count, start, end)?;
        launch_cuda_owned_i32_compare_indices_ordered(self, &mask.mask, u64::from(row_count), 0, 5)
    }

    pub fn and_predicate_masks(
        &self,
        left: &CudaPredicateMaskI32,
        right: &CudaPredicateMaskI32,
    ) -> Result<CudaPredicateMaskI32, CudaRuntimeProbeError> {
        launch_cuda_and_predicate_masks(self, left, right)
    }

    /// Compact a retained predicate mask to ascending row indices. This is the device terminal used
    /// after independently typed predicate leaves have been combined on-device; the host receives
    /// only the approved coordinates, never the component masks or a relational verdict.
    pub fn predicate_mask_indices_u32(
        &self,
        mask: &CudaPredicateMaskI32,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_owned_i32_compare_indices_ordered(
            self,
            &mask.mask,
            u64::from(mask.row_count),
            0,
            5,
        )
    }
}

/// A one-i32-per-row SQL-WHERE mask kept on the device. Zero means FALSE/UNKNOWN; non-zero means
/// TRUE. This is the non-materializing terminal used by relational pipelines; unlike the legacy
/// `run_expr_predicate_filter*` API it does not copy survivor indices to the host.
pub struct CudaPredicateMaskI32 {
    pub(super) mask: PooledDeviceBufferOwned,
    pub(super) row_count: u32,
}

impl CudaPredicateMaskI32 {
    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.mask.capacity as u64
    }

    pub(super) fn device_ptr(&self) -> u64 {
        self.mask.ptr
    }
}

pub(super) fn retain_predicate_mask_i32(
    resident: &CudaResidentDeviceMemory,
    mask: PooledBufferLease<'_>,
    row_count: u32,
) -> Result<CudaPredicateMaskI32, CudaRuntimeProbeError> {
    let required = row_count as usize * std::mem::size_of::<i32>();
    if mask.capacity < required
        || mask.primary_identity() != std::ptr::from_ref(resident.primary()).addr()
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(required));
    }
    let ptr = mask.ptr;
    let capacity = mask.capacity;
    let tracker = mask.tracker.clone();
    std::mem::forget(mask);
    Ok(CudaPredicateMaskI32 {
        mask: PooledDeviceBufferOwned {
            primary: resident.primary_arc(),
            ptr,
            capacity,
            tracker,
        },
        row_count,
    })
}

/// Compact a leased 0/1 mask buffer to matching row indices ascending. The terminal of the boolean
/// predicate VM.
pub(super) fn compact_mask_i32_to_indices(
    resident: &CudaResidentDeviceMemory,
    mask: &PooledBufferLease<'_>,
    n: u64,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    // probe-timing (VM lever): time the mask->indices compaction. The legacy path was an atomic-counter
    // compact + count/index D2H + a host `sort_unstable`; the ~93%-of-cost host sort is what the ordered
    // compaction eliminates (the share of the predicate that decided this lever).
    let _compact_scope = Probe::scope("compact");

    // Re-route to the ORDERED parallel compaction (`COMPARE_ORDERED_PTX`) in INDEX-emit mode. The mask
    // is a 0/1 i32 PER ROW (the compare-to-mask / mask-binary kernels write `selp.b32 %mask,1,0` then
    // `st.global.b32 [out + idx*4]` — one i32 per row, NOT bit-packed), so "row is set" == "mask i32 !=
    // 0". Read the typed MASK lease as the contiguous i32 input and select set rows with `needle = 0`,
    // `comparison = 5 (ne)`: the ordered
    // count/scatter kernels both test `mask[idx] != 0` and emit the surviving ROW INDICES ascending by
    // construction (contiguous block partition + ordered intra-block prefix sum) — identical indices to
    // the legacy atomic-append, but with no host `sort_unstable`.
    //
    // Lease lifetime: the ordered core does count -> host-scan -> scatter (TWO launches reading
    // mask lease). The typed wrapper validates the exact capacity and context ownership before CUDA
    // setup; this borrow spans the call, so the input outlives both reads.
    launch_cuda_buffer_i32_compare_indices_ordered(resident, mask, n, 0, 5)
}

/// Evaluate a boolean predicate `program` (comparisons -> masks, combined by `MaskBinary`) to one
/// mask buffer, then compact it to the matching row indices. The general boolean-predicate VM behind
/// the engine's AND/OR / Ne lowering.
fn launch_cuda_resident_expr_predicate_filter(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    text_needles: &[Vec<u8>],
    n: u64,
    elem: ResidentElemType,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut stack =
        run_resident_arith_program(resident, program, text_needles, n, elem, ExprTerminal::Mask)?;
    let mask = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        // A well-formed predicate program leaves exactly one mask on the stack.
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    compact_mask_i32_to_indices(resident, &mask, n)
}

fn launch_cuda_resident_expr_predicate_mask(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    text_needles: &[Vec<u8>],
    n: u32,
    elem: ResidentElemType,
) -> Result<CudaPredicateMaskI32, CudaRuntimeProbeError> {
    let n64 = u64::from(n);
    if n == 0 {
        return Ok(CudaPredicateMaskI32 {
            mask: resident.primary_arc().lease_device_buffer_owned(1)?,
            row_count: 0,
        });
    }
    let mut stack = run_resident_arith_program(
        resident,
        program,
        text_needles,
        n64,
        elem,
        ExprTerminal::Mask,
    )?;
    let mask = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    // Transfer the pool lease without a device copy. The owned guard keeps the shared primary
    // context alive and returns exactly the same bucket to the pool on drop.
    retain_predicate_mask_i32(resident, mask, n)
}

fn launch_cuda_row_range_mask_u32(
    resident: &CudaResidentDeviceMemory,
    row_count: u32,
    start: u32,
    end: u32,
) -> Result<CudaPredicateMaskI32, CudaRuntimeProbeError> {
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
.visible .entry gpu_db_row_range_mask(
    .param .u32 rows, .param .u32 start, .param .u32 end, .param .u64 mask)
{
    .reg .pred %p<4>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<5>;
    ld.param.u32 %r1, [rows];
    ld.param.u32 %r2, [start];
    ld.param.u32 %r3, [end];
    ld.param.u64 %rd1, [mask];
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %nctaid.x;
    mad.lo.u32 %r8, %r5, %r6, %r4;
    mul.lo.u32 %r9, %r7, %r6;
LOOP:
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra DONE;
    setp.ge.u32 %p2, %r8, %r2;
    setp.lt.u32 %p3, %r8, %r3;
    and.pred %p2, %p2, %p3;
    selp.u32 %r10, 1, 0, %p2;
    mul.wide.u32 %rd2, %r8, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r10;
    add.u32 %r8, %r8, %r9;
    bra LOOP;
DONE:
    ret;
}
"#;
    if start > end || end > row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
    }
    let primary = resident.primary_arc();
    primary.set_current()?;
    let bytes = (row_count as usize).saturating_mul(4).max(1);
    let mask = primary.lease_device_buffer_owned(bytes)?;
    if row_count > 0 {
        let launch = unsafe {
            primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let mut ptx = PTX.to_vec();
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_row_range_mask", &ptx)?;
        let mut a0 = row_count;
        let mut a1 = start;
        let mut a2 = end;
        let mut a3 = mask.ptr;
        let mut args = [
            (&mut a0 as *mut u32).cast(),
            (&mut a1 as *mut u32).cast(),
            (&mut a2 as *mut u32).cast(),
            (&mut a3 as *mut u64).cast(),
        ];
        check_cuda(unsafe {
            launch(
                function,
                row_count.div_ceil(256).clamp(1, 65_535),
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
    Ok(CudaPredicateMaskI32 { mask, row_count })
}

fn launch_cuda_and_predicate_masks(
    resident: &CudaResidentDeviceMemory,
    left: &CudaPredicateMaskI32,
    right: &CudaPredicateMaskI32,
) -> Result<CudaPredicateMaskI32, CudaRuntimeProbeError> {
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
.visible .entry gpu_db_and_predicate_masks(
    .param .u64 a, .param .u64 b, .param .u32 n, .param .u64 out)
{
    .reg .pred %p;
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u32 %r1, [n];
    ld.param.u64 %rd3, [out];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %nctaid.x;
    mad.lo.u32 %r6, %r3, %r4, %r2;
    mul.lo.u32 %r7, %r5, %r4;
LOOP:
    setp.ge.u32 %p, %r6, %r1;
    @%p bra DONE;
    mul.wide.u32 %rd4, %r6, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;
    ld.global.u32 %r8, [%rd5];
    ld.global.u32 %r9, [%rd6];
    and.b32 %r10, %r8, %r9;
    st.global.u32 [%rd7], %r10;
    add.u32 %r6, %r6, %r7;
    bra LOOP;
DONE:
    ret;
}
"#;
    if left.row_count != right.row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            left.row_count as usize,
        ));
    }
    let primary = resident.primary_arc();
    primary.set_current()?;
    let bytes = (left.row_count as usize).saturating_mul(4).max(1);
    let output = primary.lease_device_buffer_owned(bytes)?;
    if left.row_count > 0 {
        let launch = unsafe {
            primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let mut ptx = PTX.to_vec();
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_and_predicate_masks", &ptx)?;
        let mut a0 = left.mask.ptr;
        let mut a1 = right.mask.ptr;
        let mut a2 = left.row_count;
        let mut a3 = output.ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast(),
            (&mut a1 as *mut u64).cast(),
            (&mut a2 as *mut u32).cast(),
            (&mut a3 as *mut u64).cast(),
        ];
        check_cuda(unsafe {
            launch(
                function,
                a2.div_ceil(256).clamp(1, 65_535),
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
    Ok(CudaPredicateMaskI32 {
        mask: output,
        row_count: left.row_count,
    })
}
