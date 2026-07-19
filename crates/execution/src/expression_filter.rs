//! Resident expression filter and value-materialization orchestration.
//!
//! GPU evaluation stays in the typed postfix VM and ordered compaction. The fixed/nullable
//! value-at-indices routes still perform full-column D2H plus host selected gathering; that is
//! explicit RETIRE-003 result-path debt, not the target GPU-resident final-result boundary.

use std::os::raw::c_void;

use crate::cuda_context::{check_cuda, PooledBufferLease};
use crate::expression_vm::{
    run_resident_arith_program, run_resident_arith_program_at_indices, ExprStep, ExprTerminal,
    ResidentElemType,
};
use crate::resident_compare_ordered::{
    launch_cuda_buffer_i32_compare_indices_ordered,
    launch_cuda_resident_i32_compare_buffers_indices_ordered, validate_ordered_i32_comparison,
    validate_ordered_i32_index_domain,
};
use crate::{launch_on_pooled_stream, CudaResidentDeviceMemory, CudaRuntimeProbeError, Probe};

/// Evaluate `(a <op> b) <cmp> needle` over two resident int4 columns through the typed
/// postfix VM, then use the ordered two-pass compactor for ascending row indices. This path shares
/// the general executor's checked arithmetic, allocation-window validation, stream draining, and
/// typed intermediate leases instead of maintaining a raw-offset atomic-append special case.
pub(super) fn launch_cuda_resident_expr_two_col_filter(
    resident: &CudaResidentDeviceMemory,
    a_byte_offset: u64,
    b_byte_offset: u64,
    op_code: u32,
    n: u64,
    needle: i32,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    if op_code > 2 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(op_code as usize));
    }
    validate_ordered_i32_comparison(comparison, 4)?;
    validate_ordered_i32_index_domain(n)?;
    if n == 0 {
        return Ok(Vec::new());
    }

    let program = [
        ExprStep::LoadColumn {
            byte_offset: a_byte_offset,
        },
        ExprStep::LoadColumn {
            byte_offset: b_byte_offset,
        },
        ExprStep::BufferBinary { op: op_code },
    ];
    launch_cuda_resident_expr_arith_filter(resident, &program, n, comparison, needle)
}
/// Compare an int4 value-buffer lease to `needle` and return matching row indices ascending.
/// Shared compare-and-compact tail for the arithmetic VM and specialized two-column facade.
/// `comparison`: 0=eq, 1=lt, 2=le, 3=gt, 4=ge.
fn compact_buffer_i32_compare_to_indices(
    resident: &CudaResidentDeviceMemory,
    value: &PooledBufferLease<'_>,
    n: u64,
    needle: i32,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    // The scalar compact kernel switched on codes 0=eq..4=ge only; reject 5=ne (mask-path only) so a
    // caller gets an error, not a silently-empty result. Preserved byte-identically across the
    // re-route to the ordered compaction (no production caller passes 5 here — `ne` lowers to a mask).
    validate_ordered_i32_comparison(comparison, 4)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    // probe-timing (VM lever): the int4 simple-comparison compaction (`col <cmp> scalar` -> indices). The
    // legacy path was a fused compare+atomic-append kernel + count/index D2H + a host `sort_unstable` of
    // the surviving indices; the ~93%-of-cost host sort is what the ordered compaction eliminates.
    let _compact_scope = Probe::scope("compact");

    // Re-route to the ORDERED parallel compaction (`COMPARE_ORDERED_PTX`) in INDEX-emit mode: read the
    // leased VALUE buffer as the contiguous i32 input. The typed wrapper verifies that this lease
    // belongs to the resident context and covers every `value[idx]` read before either kernel launch.
    // Surviving row indices are emitted ASCENDING BY CONSTRUCTION (contiguous block partition +
    // ordered intra-block prefix sum), replacing the host `sort_unstable`.
    //
    // Lease lifetime: the ordered core does count -> host-scan -> scatter (TWO launches reading
    // value lease). The borrow spans this whole call, so the input outlives both reads.
    launch_cuda_buffer_i32_compare_indices_ordered(resident, value, n, needle, comparison)
}

/// PROTOTYPE — the resident arithmetic bytecode VM (docs/architecture/17 section 2.3). Runs a postfix
/// `program` over a stack of leased device buffers to evaluate an ARBITRARY int4 arithmetic tree on
/// device (each step launches one buffer->buffer primitive), then compares the single result buffer
/// to `needle` and returns the matching row indices. This is the general recursive interpreter that
/// generalizes the specialized two-column facade: arbitrary depth lowers to a longer program of the same
/// primitives. Correctness-first: each step runs on a pooled stream that syncs, so an intermediate
/// is valid before the next step reads it (pipelining/fusion is a later perf lever).
/// Compare two int4 value-buffer leases elementwise and return the matching row indices in ASCENDING
/// order. The col-vs-col / expr-vs-expr analogue of
/// `compact_buffer_i32_compare_to_indices`. `comparison`: 0=eq, 1=lt, 2=le, 3=gt, 4=ge.
fn compact_buffers_i32_compare_to_indices(
    resident: &CudaResidentDeviceMemory,
    lhs: &PooledBufferLease<'_>,
    rhs: &PooledBufferLease<'_>,
    n: u64,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    // The buffer compact kernel switches on codes 0=eq..4=ge only; reject 5=ne (mask-path only) so a
    // caller gets an error, not a silently-empty result. Preserved byte-identically across the re-route
    // to the ordered two-input compaction (no production caller passes 5 here - `ne` lowers to a mask).
    validate_ordered_i32_comparison(comparison, 4)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    // Re-route to the TWO-INPUT ORDERED parallel compaction (`COMPARE_ORDERED_PTX`,
    // `gpu_db_buffers_i32_compare_*_blocks`): compare `lhs[i] <cmp> rhs[i]` over the two leased value
    // buffers (each contiguous i32 at `base + idx*4` - the same layout the legacy
    // `gpu_db_buffer_i32_compare_buffers_to_indices` atomic kernel read). The typed wrapper verifies
    // both exact lease capacities and context ownership before launch. Surviving row indices are
    // IDENTICAL (operand order lhs,rhs; codes 0..4), but emitted ASCENDING BY CONSTRUCTION.
    //
    // Lease lifetime: the ordered launch does count -> host-scan -> scatter (TWO launches, each reading
    // BOTH buffers). The caller leases both value buffers and holds those leases across this whole call
    // (borrowed for the duration), so both inputs outlive both reads - unchanged from the legacy
    // single-launch path, which also read the same two buffers.
    launch_cuda_resident_i32_compare_buffers_indices_ordered(resident, lhs, rhs, n, comparison)
}

/// Evaluate an arithmetic `program` to one value buffer, then compare it to `needle` -> row indices.
pub(super) fn launch_cuda_resident_expr_arith_filter(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    n: u64,
    compare_code: u32,
    needle: i32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut stack = run_resident_arith_program(
        resident,
        program,
        &[],
        n,
        ResidentElemType::I32,
        ExprTerminal::Value,
    )?;
    let value = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        // A well-formed arithmetic program leaves exactly one value on the stack.
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    compact_buffer_i32_compare_to_indices(resident, &value, n, needle, compare_code)
}

pub(super) fn launch_cuda_arith_value_column_at_indices(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    n_rows: u64,
    indices: &[u32],
    elem: ResidentElemType,
) -> Result<Vec<i64>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    let n = usize::try_from(n_rows).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    // Run over ALL rows at the program's element width (I32 for int4 exprs, I64 for int8 -- the arith
    // VM checks the matching overflow bounds internally -> PG error, no wrap); the value buffer (lease)
    // is the single result + must stay alive through the D2H. Read it at that width, gather at the
    // survivor indices, widen to i64 (the universal sort-key width).
    let mut stack =
        run_resident_arith_program(resident, program, &[], n_rows, elem, ExprTerminal::Value)?;
    let value = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        // A well-formed arithmetic program leaves exactly one value on the stack.
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    let mut out = Vec::with_capacity(indices.len());
    match elem {
        ResidentElemType::I32 => {
            let byte_len = n
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
            let mut host = vec![0i32; n];
            check_cuda(unsafe {
                cu_memcpy_dtoh(host.as_mut_ptr().cast::<c_void>(), value.ptr, byte_len)
            })?;
            drop(value);
            for &i in indices {
                let v = *host
                    .get(i as usize)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(i as usize))?;
                out.push(i64::from(v));
            }
        }
        ResidentElemType::I64 => {
            let byte_len = n
                .checked_mul(std::mem::size_of::<i64>())
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
            let mut host = vec![0i64; n];
            check_cuda(unsafe {
                cu_memcpy_dtoh(host.as_mut_ptr().cast::<c_void>(), value.ptr, byte_len)
            })?;
            drop(value);
            for &i in indices {
                let v = *host
                    .get(i as usize)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(i as usize))?;
                out.push(v);
            }
        }
        // i128 (numeric) sort expressions are not yet supported here.
        ResidentElemType::I128 => {
            drop(value);
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
    }
    Ok(out)
}

/// Evaluate checked arithmetic only at a bounded host-provided coordinate list. The coordinates
/// are validated before CUDA, uploaded once, and dereferenced by the indexed VM load; the compact
/// result is the only value data copied back. This is the mutation twin of the read-path survivor
/// evaluator: rows rejected by UPDATE's WHERE cannot trigger an overflow or be read as operands.
pub(super) fn launch_cuda_arith_value_column_at_selected_indices(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    source_row_count: u64,
    indices: &[u32],
    elem: ResidentElemType,
) -> Result<Vec<i64>, CudaRuntimeProbeError> {
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    if indices.is_empty() {
        return Ok(Vec::new());
    }
    if source_row_count == 0
        || indices
            .iter()
            .any(|index| u64::from(*index) >= source_row_count)
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(indices.len()));
    }
    let primary = resident.primary();
    primary.set_current()?;
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let index_bytes = indices
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(indices.len()))?;
    let index_device = primary.lease_device_buffer(index_bytes)?;
    check_cuda(unsafe {
        cu_memcpy_htod(
            index_device.ptr,
            indices.as_ptr().cast::<c_void>(),
            index_bytes,
        )
    })?;
    let survivor_count = indices.len() as u64;
    let mut stack = run_resident_arith_program_at_indices(
        resident,
        program,
        index_device.ptr,
        survivor_count,
        source_row_count,
        elem,
    )?;
    let value = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    let mut out = Vec::with_capacity(indices.len());
    match elem {
        ResidentElemType::I32 => {
            let mut host = vec![0i32; indices.len()];
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    host.as_mut_ptr().cast::<c_void>(),
                    value.ptr,
                    indices.len() * std::mem::size_of::<i32>(),
                )
            })?;
            out.extend(host.into_iter().map(i64::from));
        }
        ResidentElemType::I64 => {
            let mut host = vec![0i64; indices.len()];
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    host.as_mut_ptr().cast::<c_void>(),
                    value.ptr,
                    indices.len() * std::mem::size_of::<i64>(),
                )
            })?;
            out = host;
        }
        ResidentElemType::I128 => {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
    }
    Ok(out)
}

/// As [`launch_cuda_arith_value_column_at_indices`] (I32 arith) but the expression is NULLABLE: run the
/// arith program (value buffer) AND a `validity_program` (a mask VM program yielding 1 where every
/// operand column is non-NULL), then blend ON-DEVICE into an i64 key buffer -- `i64::MAX` (PG's
/// default-end sentinel; no widened int4 value can equal it) where NULL, else the sign-extended value.
/// D2H the i64 keys and gather at `indices` (M3 -- doc 21).
pub(super) fn launch_cuda_arith_value_column_at_indices_nullable(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    validity_program: &[ExprStep],
    n_rows: u64,
    indices: &[u32],
) -> Result<Vec<i64>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
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
    const PTX: &[u8] = include_bytes!("device_fill.ptx");
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    let n = usize::try_from(n_rows).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let primary = resident.primary();
    primary.set_current()?;
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    // The arith VALUE (I32) and the VALIDITY mask (I32, 0/1) -- two independent VM runs over all rows; each
    // top-of-stack buffer is the single result and must stay alive through the blend.
    let mut value_stack = run_resident_arith_program(
        resident,
        program,
        &[],
        n_rows,
        ResidentElemType::I32,
        ExprTerminal::Value,
    )?;
    let value = value_stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !value_stack.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    let mut mask_stack = run_resident_arith_program(
        resident,
        validity_program,
        &[],
        n_rows,
        ResidentElemType::I32,
        ExprTerminal::Mask,
    )?;
    let mask = mask_stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !mask_stack.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            validity_program.len(),
        ));
    }
    // Blend ON-DEVICE: out_i64[i] = mask[i] ? sign_extend(value[i]) : i64::MAX.
    let out_bytes = n
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    let out_buf = primary.lease_device_buffer(out_bytes)?;
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let blend_fn = primary.cached_function(c"gpu_db_blend_widen_null_sentinel", &ptx)?;
    const BLOCK: u32 = 256;
    let grid = n_rows.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = value.ptr;
    let mut a1 = mask.ptr;
    let mut a2 = out_buf.ptr;
    let mut a3 = n_rows;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            blend_fn,
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
    })?;
    let mut host = vec![0i64; n];
    check_cuda(unsafe {
        cu_memcpy_dtoh(host.as_mut_ptr().cast::<c_void>(), out_buf.ptr, out_bytes)
    })?;
    drop(out_buf);
    drop(mask);
    drop(value);
    let mut out = Vec::with_capacity(indices.len());
    for &i in indices {
        let v = *host
            .get(i as usize)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(i as usize))?;
        out.push(v);
    }
    Ok(out)
}

/// Evaluate a `program` that leaves TWO value buffers (the compiled lhs then rhs of a comparison),
/// then compare them elementwise (`lhs <cmp> rhs`) -> row indices. The col-vs-col / expr-vs-expr
/// filter behind the engine's `Compare(expr, expr)` lowering.
pub(super) fn launch_cuda_resident_expr_compare_buffers_filter(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    n: u64,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut stack = run_resident_arith_program(
        resident,
        program,
        &[],
        n,
        ResidentElemType::I32,
        ExprTerminal::TwoValues,
    )?;
    let rhs = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    let lhs = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    compact_buffers_i32_compare_to_indices(resident, &lhs, &rhs, n, comparison)
}
