use std::os::raw::c_void;

use super::resident_window::validate_text_windows;
use super::{
    CudaResidentDeviceMemory, CudaRuntimeProbeError, PooledBufferLease, check_cuda,
    launch_on_pooled_stream,
};

/// One step of the resident expression bytecode (docs/architecture/17 section 2.3): a postfix
/// program over a stack of device buffers. Value steps build an int4 arithmetic expression into a
/// value buffer; comparison steps turn value buffer(s) into a 0/1 MASK buffer; `MaskBinary` combines
/// masks. A boolean predicate program leaves one mask (compacted to indices by the predicate VM); the
/// arithmetic / col-vs-col filters use only the value steps and fuse the terminal comparison. `op` is
/// 0=add/1=sub/2=mul (arithmetic) or 0=and/1=or (mask); `cmp` is 0=eq/1=lt/2=le/3=gt/4=ge/5=ne.
#[derive(Debug, Clone, Copy)]
pub enum ExprStep {
    /// Push the resident int4 column at `byte_offset` (loaded into a fresh buffer).
    LoadColumn { byte_offset: u64 },
    /// SV3b (mixed-width VM): push an i64 (u64) column at `byte_offset` into an 8-byte-per-elem buffer via
    /// the i64 load kernel — REGARDLESS of the program's run `elem`. This is what lets a mixed program
    /// combine an i32 value predicate with an i64 `deleted_by > read_txn_id` visibility compare
    /// (`CompareScalarI64`) in ONE program: masks are width-agnostic, only the value buffers differ. An
    /// all-i32 / all-i64 program never emits this step, so those programs are byte-identical.
    ///
    /// **CALLER CONTRACT (audit follow-up):** in a NON-I64 run, a `CompareScalarI64` MUST consume a buffer
    /// produced by a preceding `LoadColumnI64` (its 8-byte operand) — the i64 compare kernel reads 8 bytes
    /// per element, so pairing it with a 4-byte `LoadColumn` buffer is an out-of-bounds DEVICE read. The
    /// program builder is responsible for this pairing (the VM does not structurally enforce it).
    LoadColumnI64 { byte_offset: u64 },
    /// Pop b, pop a, push `a <op> b` (buffer x buffer).
    BufferBinary { op: u32 },
    /// Pop a, push `a <op> scalar` (or `scalar <op> a` if `scalar_on_left`) — folds an immediate
    /// literal operand without materializing a constant buffer.
    ScalarBinary {
        op: u32,
        scalar: i32,
        scalar_on_left: bool,
    },
    /// Pop a (value), push the mask `(scalar_on_left ? scalar <cmp> a : a <cmp> scalar) ? 1 : 0`.
    CompareScalar {
        cmp: u32,
        scalar: i32,
        scalar_on_left: bool,
    },
    /// Like `CompareScalar` but with a full-width i64 scalar — for an i64 literal that exceeds `i32`
    /// (a timestamp's microseconds, or a large int8 literal). ONLY valid in an I64 program (it launches
    /// the i64 compare-scalar-to-mask kernel, which reads an s64 scalar). The dispatch errors otherwise.
    CompareScalarI64 {
        cmp: u32,
        scalar: i64,
        scalar_on_left: bool,
    },
    /// Like `CompareScalar` but with a full-width signed i128 scalar — for a numeric mantissa (already
    /// rescaled to the column scale) that exceeds `i32`. ONLY valid in an I128 program (it launches the
    /// i128 compare-scalar-to-mask kernel, which reads the scalar as two u64 limbs). Errors otherwise.
    CompareScalarI128 {
        cmp: u32,
        scalar: i128,
        scalar_on_left: bool,
    },
    /// Pop b, pop a (values), push the mask `(a <cmp> b) ? 1 : 0`.
    CompareBuffers { cmp: u32 },
    /// Pop b, pop a (masks), push `op==0 ? a&&b : a||b` (0/1).
    MaskBinary { op: u32 },
    /// Push the mask `(textcol[i] == needle) ^ negate ? 1 : 0` for the resident TEXT column at
    /// (`offsets_byte_offset`, `bytes_byte_offset`). The needle bytes are `text_needles[needle_idx]`
    /// (varlen, so threaded out-of-band to keep this step `Copy`). Lets the mask VM combine text
    /// equality/inequality with AND/OR (and with int4 comparisons) -- text `IN`, multi-text WHERE.
    TextEqMask {
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        needle_idx: u32,
        negate: bool,
    },
    /// Push the mask `(scalar_on_left ? needle <cmp> textcol[i] : textcol[i] <cmp> needle) ? 1 : 0`
    /// for the resident TEXT column at (`offsets_byte_offset`, `bytes_byte_offset`) — LEXICOGRAPHIC
    /// unsigned byte compare (memcmp of the common prefix, shorter sorts first), matching Rust
    /// `str::cmp`. `cmp`: 0=eq/1=lt/2=le/3=gt/4=ge/5=ne. The needle bytes are
    /// `text_needles[needle_idx]` (the same out-of-band channel as `TextEqMask`). Lets the mask VM
    /// combine text INEQUALITIES with AND/OR — text ranges (`name >= 'a' AND name < 'm'`) and
    /// nullable-text inequalities (via a following validity MaskBinary). Launches the SAME kernel as
    /// the standalone `expr_text_compare_scalar_filter` (`gpu_db_resident_text_compare_scalar_to_mask`).
    TextCmpMask {
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        needle_idx: u32,
        scalar_on_left: bool,
        cmp: u32,
    },
    /// Push the mask `(scalar_on_left ? needle <cmp> uuid[i] : uuid[i] <cmp> needle) ? 1 : 0` for the
    /// resident UUID column (16 raw bytes/row in the b128 section at `byte_offset`) — unsigned
    /// big-endian 16-byte memcmp (PG's uuid order == Rust `[u8;16]::cmp`). `cmp`: 0=eq/1=lt/2=le/
    /// 3=gt/4=ge/5=ne. The 16 needle bytes are `text_needles[needle_idx]` (the same out-of-band varlen
    /// channel). Lets the mask VM combine uuid comparisons with AND/OR — uuid IN, uuid ranges, mixed
    /// uuid+int4/text WHEREs. Launches the SAME kernel as the standalone
    /// `expr_uuid_compare_scalar_filter` (`gpu_db_resident_uuid_compare_scalar_to_mask`).
    UuidCmpMask {
        byte_offset: u64,
        needle_idx: u32,
        scalar_on_left: bool,
        cmp: u32,
    },
    /// Push the mask `(text_a[i] <cmp> text_b[i]) ? 1 : 0` for TWO resident TEXT columns — the
    /// COLUMN-VS-COLUMN twin of `TextCmpMask` (ADR-006): per-row lexicographic unsigned byte
    /// memcmp (shorter sorts first), matching Rust `str::cmp` == the host recheck. `cmp` codes
    /// 0=eq/1=lt/2=le/3=gt/4=ge/5=ne. Column order is positional (`a <cmp> b`) — no side flag.
    /// Launches the NEW `gpu_db_resident_text_compare_columns_to_mask` kernel.
    TextCmpColumnsMask {
        a_offsets_byte_offset: u64,
        a_bytes_byte_offset: u64,
        b_offsets_byte_offset: u64,
        b_bytes_byte_offset: u64,
        cmp: u32,
    },
    /// Push the mask `(uuid_a[i] <cmp> uuid_b[i]) ? 1 : 0` for TWO resident UUID columns (16 raw
    /// bytes/row each in the b128 section) — the COLUMN-VS-COLUMN twin of `UuidCmpMask` (ADR-006):
    /// unsigned big-endian 16-byte memcmp per row, `cmp` codes 0=eq/1=lt/2=le/3=gt/4=ge/5=ne. Lets
    /// the mask VM combine `ua <cmp> ub` with AND/OR. Launches the SAME kernel as the standalone
    /// col-vs-col path (`gpu_db_resident_uuid_compare_columns_to_mask`).
    UuidCmpColumnsMask {
        a_byte_offset: u64,
        b_byte_offset: u64,
        cmp: u32,
    },
    /// Push the mask `(textcol[i] LIKE pattern) ? 1 : 0` for the resident TEXT column at
    /// (`offsets_byte_offset`, `bytes_byte_offset`). The pattern is the COMPILED u32 token array (escapes
    /// resolved on the host: each token `(op<<8)|byte`, op 0=literal/1=`_`/2=`%`), stored LE-serialized in
    /// `text_needles[pattern_idx]` (the SAME out-of-band channel as `TextEqMask` — `ntok = len/4`), so
    /// this step stays `Copy`. Lets the mask VM combine LIKE with AND/OR and (via a following validity
    /// MaskBinary) with NULL 3VL — the path a NULLABLE-text `LIKE` predicate takes (a non-null text LIKE
    /// keeps the standalone `expr_text_like_scalar_filter` fast path). Launches the SAME on-device matcher
    /// kernel as that fast path (`gpu_db_resident_text_like_scalar_to_mask`).
    TextLikeMask {
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        pattern_idx: u32,
    },
    /// Push the mask `bitmap[i] ^ negate ? 1 : 0` for the resident BOOL column whose 1-bit-per-row
    /// bitmap is at `bitmap_byte_offset` (`negate` selects the clear bits, i.e. `flag = false` / `NOT
    /// flag`). Lets the mask VM combine a bool column with AND/OR (and int4/text) -- e.g. `flag AND x>0`.
    BoolMask {
        bitmap_byte_offset: u64,
        negate: bool,
    },
    /// Push a CONSTANT per-row mask (every row `value ? 1 : 0`), filled with a device memset (no kernel).
    /// Used by `col IS NULL` / `IS NOT NULL` on a column with no NULL validity bitmap (M3 -- doc 21): the
    /// column holds no NULLs, so every row is valid -- IS NOT NULL is all-1, IS NULL all-0.
    ConstMask { value: bool },
}

/// Execute an arithmetic bytecode `program` over a stack of leased device buffers and return the
/// resulting stack (each step launches one buffer->buffer primitive on a syncing pooled stream). The
/// returned leases borrow `resident`; the caller consumes the stack (one value for a scalar compare,
/// two for a buffer-vs-buffer compare). Callers handle `n == 0` before calling.
/// The element type of a resident arithmetic VM program (the type matrix, doc 19): int4 buffers
/// (s32, 4 bytes), int8 buffers (s64, 8 bytes), or numeric buffers (i128, 16 bytes). A program is
/// mono-typed; the VM selects per-step kernels + intermediate-buffer sizes by this. The mask AND/OR +
/// mask->indices compaction stages are type-agnostic (they operate on i32 0/1 masks) and are shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentElemType {
    I32,
    I64,
    I128,
}

impl ResidentElemType {
    pub(super) fn elem_size(self) -> usize {
        match self {
            Self::I32 => std::mem::size_of::<i32>(),
            Self::I64 => std::mem::size_of::<i64>(),
            Self::I128 => std::mem::size_of::<i128>(),
        }
    }
}

pub(super) fn run_resident_arith_program<'r>(
    resident: &'r CudaResidentDeviceMemory,
    program: &[ExprStep],
    text_needles: &[Vec<u8>],
    n: u64,
    elem: ResidentElemType,
) -> Result<Vec<PooledBufferLease<'r>>, CudaRuntimeProbeError> {
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
    type CuMemsetD32Async = unsafe extern "C" fn(u64, u32, usize, *mut c_void) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let byte_len = n_usize
        .checked_mul(elem.elem_size())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
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
    // Only constant-mask programs require this symbol. Keep every other VM route's driver contract
    // unchanged while ensuring the constant path never materializes or uploads an O(rows) host vector.
    let cu_memset_d32_async = if program
        .iter()
        .any(|step| matches!(step, ExprStep::ConstMask { .. }))
    {
        Some(unsafe {
            *resident
                .lib()
                .get::<CuMemsetD32Async>(b"cuMemsetD32Async\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        })
    } else {
        None
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    // Per-element-type kernels (the type matrix, doc 19); the mask-binary stage is type-agnostic.
    let (load_name, binary_name, scalar_name, compare_scalar_name, compare_buffers_name) =
        match elem {
            ResidentElemType::I32 => (
                c"gpu_db_resident_i32_load_column",
                c"gpu_db_buffer_i32_binary",
                c"gpu_db_buffer_i32_binary_scalar",
                c"gpu_db_buffer_i32_compare_scalar_to_mask",
                c"gpu_db_buffer_i32_compare_buffers_to_mask",
            ),
            ResidentElemType::I64 => (
                c"gpu_db_resident_i64_load_column",
                c"gpu_db_buffer_i64_binary",
                c"gpu_db_buffer_i64_binary_scalar",
                c"gpu_db_buffer_i64_compare_scalar_to_mask",
                c"gpu_db_buffer_i64_compare_buffers_to_mask",
            ),
            ResidentElemType::I128 => (
                c"gpu_db_resident_i128_load_column",
                c"gpu_db_buffer_i128_binary",
                c"gpu_db_buffer_i128_binary_scalar",
                c"gpu_db_buffer_i128_compare_scalar_to_mask",
                c"gpu_db_buffer_i128_compare_buffers_to_mask",
            ),
        };
    let load_fn = primary.cached_function(load_name, &ptx)?;
    let buffer_binary_fn = primary.cached_function(binary_name, &ptx)?;
    let scalar_binary_fn = primary.cached_function(scalar_name, &ptx)?;
    let compare_scalar_mask_fn = primary.cached_function(compare_scalar_name, &ptx)?;
    let compare_buffers_mask_fn = primary.cached_function(compare_buffers_name, &ptx)?;
    let mask_binary_fn = primary.cached_function(c"gpu_db_mask_binary", &ptx)?;
    // Text equality -> i32 mask, so the VM can combine text `=`/`<>` with AND/OR (text IN, multi-text
    // WHERE). The needle is the varlen bytes `text_needles[needle_idx]`. Loaded lazily (only if used).
    let text_eq_mask_fn = if program
        .iter()
        .any(|s| matches!(s, ExprStep::TextEqMask { .. }))
    {
        Some(primary.cached_function(c"gpu_db_resident_text_eq_scalar_to_mask", &ptx)?)
    } else {
        None
    };
    // Text ordering (< <= > >=) -> i32 mask via the lexicographic byte-compare kernel (the SAME kernel
    // the standalone text-inequality fast path launches), so the VM can combine text ranges
    // (`name >= 'a' AND name < 'm'`) with AND/OR. Lazy (only if a TextCmpMask step is present).
    let text_cmp_mask_fn = if program
        .iter()
        .any(|s| matches!(s, ExprStep::TextCmpMask { .. }))
    {
        Some(primary.cached_function(c"gpu_db_resident_text_compare_scalar_to_mask", &ptx)?)
    } else {
        None
    };
    // uuid comparison -> i32 mask via the byte-wise b128 compare kernel (the SAME kernel the standalone
    // uuid fast path launches), so the VM can combine uuid `=`/`<`/`IN` with AND/OR. Lazy (only if used).
    let uuid_cmp_mask_fn = if program
        .iter()
        .any(|s| matches!(s, ExprStep::UuidCmpMask { .. }))
    {
        Some(primary.cached_function(c"gpu_db_resident_uuid_compare_scalar_to_mask", &ptx)?)
    } else {
        None
    };
    // text COL-VS-COL -> i32 mask (ADR-006): the per-row two-column lexicographic byte-compare
    // kernel, so the VM can combine `ta <cmp> tb` with AND/OR. Lazy (only if used).
    let text_cmp_columns_mask_fn = if program
        .iter()
        .any(|s| matches!(s, ExprStep::TextCmpColumnsMask { .. }))
    {
        Some(primary.cached_function(c"gpu_db_resident_text_compare_columns_to_mask", &ptx)?)
    } else {
        None
    };
    // uuid COL-VS-COL -> i32 mask (ADR-006: the SAME kernel the standalone col-vs-col path
    // launches), so the VM can combine `ua <cmp> ub` with AND/OR. Lazy (only if used).
    let uuid_cmp_columns_mask_fn = if program
        .iter()
        .any(|s| matches!(s, ExprStep::UuidCmpColumnsMask { .. }))
    {
        Some(primary.cached_function(c"gpu_db_resident_uuid_compare_columns_to_mask", &ptx)?)
    } else {
        None
    };
    // Text LIKE -> i32 mask (the SAME matcher kernel the standalone `expr_text_like_scalar_filter` uses),
    // so the VM can combine LIKE with AND/OR and the NULL 3VL validity AND. The pattern tokens are the
    // varlen bytes `text_needles[pattern_idx]` (LE u32). Loaded lazily (only if a TextLikeMask step is used).
    let text_like_mask_fn = if program
        .iter()
        .any(|s| matches!(s, ExprStep::TextLikeMask { .. }))
    {
        Some(primary.cached_function(c"gpu_db_resident_text_like_scalar_to_mask", &ptx)?)
    } else {
        None
    };
    // Bool column bitmap -> i32 mask, so the VM can combine a bool column with AND/OR. Lazy (only if used).
    let bool_mask_fn = if program
        .iter()
        .any(|s| matches!(s, ExprStep::BoolMask { .. }))
    {
        Some(primary.cached_function(c"gpu_db_resident_bool_to_mask", &ptx)?)
    } else {
        None
    };
    // SV3b (mixed-width VM): the i64 load + i64 compare-scalar kernels, loaded lazily only when the program
    // contains a `LoadColumnI64` step — i.e. a MIXED program running at `elem = I32` that also needs an i64
    // `deleted_by` compare. They let those i64 steps run at full 8-byte width inside an otherwise-i32 program;
    // the value buffers differ (8 vs 4 bytes) but the resulting masks are width-agnostic and AND together.
    // An all-i64 program keeps using `elem`'s `load_fn`/`compare_scalar_mask_fn` (already the i64 kernels).
    let has_i64_step = program
        .iter()
        .any(|s| matches!(s, ExprStep::LoadColumnI64 { .. }));
    let i64_byte_len = n_usize
        .checked_mul(ResidentElemType::I64.elem_size())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let (i64_load_fn, i64_compare_scalar_mask_fn) = if has_i64_step {
        (
            Some(primary.cached_function(c"gpu_db_resident_i64_load_column", &ptx)?),
            Some(primary.cached_function(c"gpu_db_buffer_i64_compare_scalar_to_mask", &ptx)?),
        )
    } else {
        (None, None)
    };
    // numeric (i128) multiply is a SEPARATE kernel — the signed 128x128->256 product is too large to
    // inline into the add/sub binary kernel — loaded only for I128. int4/int8 multiply lives in their
    // binary kernel (mul.hi), so this stays None there.
    let i128_mul_scalar_fn = match elem {
        ResidentElemType::I128 => {
            Some(primary.cached_function(c"gpu_db_buffer_i128_mul_scalar", &ptx)?)
        }
        ResidentElemType::I32 | ResidentElemType::I64 => None,
    };
    // column*column multiply for I128 is its own kernel too; None for int4/int8 (their binary kernel
    // handles multiply), so the BufferBinary arm keeps using buffer_binary_fn there.
    let i128_mul_buffer_fn = match elem {
        ResidentElemType::I128 => Some(primary.cached_function(c"gpu_db_buffer_i128_mul", &ptx)?),
        ResidentElemType::I32 | ResidentElemType::I64 => None,
    };

    // CHECKED int4 arithmetic (Charter rule 2 PG-fidelity): ONE overflow flag shared by every
    // arithmetic step of the program. The checked kernels OR 1 into it when an op overflows int32;
    // after the program runs (each pooled-stream launch syncs) the host reads it once and raises
    // `integer out of range`, exactly like Postgres — never a silent wrap, and on-device (no CPU
    // fallback). Zeroed with a blocking HtoD so it is initialized before the first launch reads it.
    let overflow_buf = primary.lease_device_buffer(std::mem::size_of::<u32>())?;
    let overflow_zero = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_htod(
            overflow_buf.ptr,
            (&overflow_zero as *const u32).cast::<c_void>(),
            std::mem::size_of::<u32>(),
        )
    })?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let resident_base = resident.device_ptr();
    let launch =
        |function: *mut c_void, args: &mut [*mut c_void]| -> Result<(), CudaRuntimeProbeError> {
            launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
                cu_launch_kernel(
                    function,
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
            })
        };

    // Stack of leased intermediate buffers. Popped operands drop (return to the pool) after their
    // consuming step's stream has synced, so the pool churns at most ~tree-depth live buffers.
    let mut stack: Vec<PooledBufferLease> = Vec::new();
    for step in program {
        match *step {
            ExprStep::LoadColumn { byte_offset } => {
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = resident_base;
                let mut a1 = byte_offset;
                let mut a2 = n;
                let mut a3 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u64).cast::<c_void>(),
                    (&mut a3 as *mut u64).cast::<c_void>(),
                ];
                launch(load_fn, &mut args)?;
                stack.push(out);
            }
            ExprStep::LoadColumnI64 { byte_offset } => {
                // SV3b: load an i64 column at full 8-byte width even in a mixed (elem=I32) program.
                let i64_load_fn =
                    i64_load_fn.ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(i64_byte_len)?;
                let mut a0 = resident_base;
                let mut a1 = byte_offset;
                let mut a2 = n;
                let mut a3 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u64).cast::<c_void>(),
                    (&mut a3 as *mut u64).cast::<c_void>(),
                ];
                launch(i64_load_fn, &mut args)?;
                stack.push(out);
            }
            ExprStep::BufferBinary { op } => {
                let rhs = stack
                    .pop()
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let lhs = stack
                    .pop()
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = lhs.ptr;
                let mut a1 = rhs.ptr;
                let mut a2 = op;
                let mut a3 = n;
                let mut a4 = out.ptr;
                let mut a5 = overflow_buf.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u32).cast::<c_void>(),
                    (&mut a3 as *mut u64).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                    (&mut a5 as *mut u64).cast::<c_void>(),
                ];
                // op 2 (multiply) over two i128 buffers uses the dedicated i128 column*column mul
                // kernel; add/sub (and all int4/int8 ops, where i128_mul_buffer_fn is None) use the
                // binary kernel.
                let function = match i128_mul_buffer_fn {
                    Some(mul_fn) if op == 2 => mul_fn,
                    _ => buffer_binary_fn,
                };
                launch(function, &mut args)?;
                stack.push(out);
            }
            ExprStep::ScalarBinary {
                op,
                scalar,
                scalar_on_left,
            } => {
                let lhs = stack
                    .pop()
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = lhs.ptr;
                let mut a2 = op;
                let mut a3 = u32::from(scalar_on_left);
                let mut a4 = n;
                let mut a5 = out.ptr;
                let mut a6 = overflow_buf.ptr;
                // The scalar param is s32 (int4 kernel), s64 (int8 kernel), or two u64 limbs (the
                // numeric i128 kernel). The i32 ExprStep literal widens to the element type (PG's
                // int4->int8 / int4->numeric coercion).
                match elem {
                    ResidentElemType::I32 => {
                        let mut a1 = scalar;
                        let mut args = [
                            (&mut a0 as *mut u64).cast::<c_void>(),
                            (&mut a1 as *mut i32).cast::<c_void>(),
                            (&mut a2 as *mut u32).cast::<c_void>(),
                            (&mut a3 as *mut u32).cast::<c_void>(),
                            (&mut a4 as *mut u64).cast::<c_void>(),
                            (&mut a5 as *mut u64).cast::<c_void>(),
                            (&mut a6 as *mut u64).cast::<c_void>(),
                        ];
                        launch(scalar_binary_fn, &mut args)?;
                    }
                    ResidentElemType::I64 => {
                        let mut a1 = i64::from(scalar);
                        let mut args = [
                            (&mut a0 as *mut u64).cast::<c_void>(),
                            (&mut a1 as *mut i64).cast::<c_void>(),
                            (&mut a2 as *mut u32).cast::<c_void>(),
                            (&mut a3 as *mut u32).cast::<c_void>(),
                            (&mut a4 as *mut u64).cast::<c_void>(),
                            (&mut a5 as *mut u64).cast::<c_void>(),
                            (&mut a6 as *mut u64).cast::<c_void>(),
                        ];
                        launch(scalar_binary_fn, &mut args)?;
                    }
                    ResidentElemType::I128 => {
                        let s = i128::from(scalar);
                        let mut s_lo = s as u64;
                        let mut s_hi = (s >> 64) as u64;
                        let mut args = [
                            (&mut a0 as *mut u64).cast::<c_void>(),
                            (&mut s_lo as *mut u64).cast::<c_void>(),
                            (&mut s_hi as *mut u64).cast::<c_void>(),
                            (&mut a2 as *mut u32).cast::<c_void>(),
                            (&mut a3 as *mut u32).cast::<c_void>(),
                            (&mut a4 as *mut u64).cast::<c_void>(),
                            (&mut a5 as *mut u64).cast::<c_void>(),
                            (&mut a6 as *mut u64).cast::<c_void>(),
                        ];
                        // op 2 (multiply) uses the dedicated i128 mul kernel; add/sub the binary kernel.
                        let function = if op == 2 {
                            i128_mul_scalar_fn
                                .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?
                        } else {
                            scalar_binary_fn
                        };
                        launch(function, &mut args)?;
                    }
                }
                stack.push(out);
            }
            ExprStep::CompareScalar {
                cmp,
                scalar,
                scalar_on_left,
            } => {
                let value = stack
                    .pop()
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = value.ptr;
                let mut a2 = u32::from(scalar_on_left);
                let mut a3 = cmp;
                let mut a4 = n;
                let mut a5 = out.ptr;
                match elem {
                    ResidentElemType::I32 => {
                        let mut a1 = scalar;
                        let mut args = [
                            (&mut a0 as *mut u64).cast::<c_void>(),
                            (&mut a1 as *mut i32).cast::<c_void>(),
                            (&mut a2 as *mut u32).cast::<c_void>(),
                            (&mut a3 as *mut u32).cast::<c_void>(),
                            (&mut a4 as *mut u64).cast::<c_void>(),
                            (&mut a5 as *mut u64).cast::<c_void>(),
                        ];
                        launch(compare_scalar_mask_fn, &mut args)?;
                    }
                    ResidentElemType::I64 => {
                        let mut a1 = i64::from(scalar);
                        let mut args = [
                            (&mut a0 as *mut u64).cast::<c_void>(),
                            (&mut a1 as *mut i64).cast::<c_void>(),
                            (&mut a2 as *mut u32).cast::<c_void>(),
                            (&mut a3 as *mut u32).cast::<c_void>(),
                            (&mut a4 as *mut u64).cast::<c_void>(),
                            (&mut a5 as *mut u64).cast::<c_void>(),
                        ];
                        launch(compare_scalar_mask_fn, &mut args)?;
                    }
                    ResidentElemType::I128 => {
                        let s = i128::from(scalar);
                        let mut s_lo = s as u64;
                        let mut s_hi = (s >> 64) as u64;
                        let mut args = [
                            (&mut a0 as *mut u64).cast::<c_void>(),
                            (&mut s_lo as *mut u64).cast::<c_void>(),
                            (&mut s_hi as *mut u64).cast::<c_void>(),
                            (&mut a2 as *mut u32).cast::<c_void>(),
                            (&mut a3 as *mut u32).cast::<c_void>(),
                            (&mut a4 as *mut u64).cast::<c_void>(),
                            (&mut a5 as *mut u64).cast::<c_void>(),
                        ];
                        launch(compare_scalar_mask_fn, &mut args)?;
                    }
                }
                stack.push(out);
            }
            ExprStep::CompareScalarI64 {
                cmp,
                scalar,
                scalar_on_left,
            } => {
                // A full-width i64 scalar (timestamp micros / large int8 literal / `deleted_by` commit seq).
                // The i64 compare-scalar-to-mask kernel reads an s64 scalar + an 8-byte operand, so it is
                // valid EITHER in an all-I64 program (the run's `compare_scalar_mask_fn` is the i64 kernel)
                // OR in a MIXED program where a `LoadColumnI64` supplied the 8-byte operand and the lazy i64
                // kernel was loaded (SV3b). Reject otherwise (an i32/i128 run kernel would mis-read the arg).
                // The mask OUTPUT is width-agnostic (i32), so `byte_len` fits it in either run.
                let compare_fn = if elem == ResidentElemType::I64 {
                    compare_scalar_mask_fn
                } else {
                    i64_compare_scalar_mask_fn
                        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?
                };
                let value = stack
                    .pop()
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = value.ptr;
                let mut a1 = scalar;
                let mut a2 = u32::from(scalar_on_left);
                let mut a3 = cmp;
                let mut a4 = n;
                let mut a5 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut i64).cast::<c_void>(),
                    (&mut a2 as *mut u32).cast::<c_void>(),
                    (&mut a3 as *mut u32).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                    (&mut a5 as *mut u64).cast::<c_void>(),
                ];
                launch(compare_fn, &mut args)?;
                stack.push(out);
            }
            ExprStep::CompareScalarI128 {
                cmp,
                scalar,
                scalar_on_left,
            } => {
                // A full-width signed i128 scalar (a numeric mantissa). The i128 compare-scalar-to-mask
                // kernel reads the scalar as two u64 limbs (lo, hi), so this is valid ONLY for an I128
                // program; reject any other elem (an i32/i64 kernel would mis-read the 16-byte arg).
                if elem != ResidentElemType::I128 {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(0));
                }
                let value = stack
                    .pop()
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = value.ptr;
                let mut s_lo = scalar as u64;
                let mut s_hi = (scalar >> 64) as u64;
                let mut a2 = u32::from(scalar_on_left);
                let mut a3 = cmp;
                let mut a4 = n;
                let mut a5 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut s_lo as *mut u64).cast::<c_void>(),
                    (&mut s_hi as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u32).cast::<c_void>(),
                    (&mut a3 as *mut u32).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                    (&mut a5 as *mut u64).cast::<c_void>(),
                ];
                launch(compare_scalar_mask_fn, &mut args)?;
                stack.push(out);
            }
            ExprStep::CompareBuffers { cmp } => {
                let rhs = stack
                    .pop()
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let lhs = stack
                    .pop()
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = lhs.ptr;
                let mut a1 = rhs.ptr;
                let mut a2 = cmp;
                let mut a3 = n;
                let mut a4 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u32).cast::<c_void>(),
                    (&mut a3 as *mut u64).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                ];
                launch(compare_buffers_mask_fn, &mut args)?;
                stack.push(out);
            }
            ExprStep::MaskBinary { op } => {
                let rhs = stack
                    .pop()
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let lhs = stack
                    .pop()
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = lhs.ptr;
                let mut a1 = rhs.ptr;
                let mut a2 = op;
                let mut a3 = n;
                let mut a4 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u32).cast::<c_void>(),
                    (&mut a3 as *mut u64).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                ];
                launch(mask_binary_fn, &mut args)?;
                stack.push(out);
            }
            ExprStep::TextEqMask {
                offsets_byte_offset,
                bytes_byte_offset,
                bytes_len,
                needle_idx,
                negate,
            } => {
                // textcol[i] == needle (XOR negate) -> i32 mask pushed on the stack; the VM combines it
                // with AND/OR like any other mask. The needle is uploaded H2D into a fresh lease (>=1
                // byte so the pointer is valid for the empty string, which the kernel never reads).
                let needle = text_needles.get(needle_idx as usize).ok_or(
                    CudaRuntimeProbeError::InvalidInputLength(needle_idx as usize),
                )?;
                let function =
                    text_eq_mask_fn.ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let text_bytes_limit = validate_text_windows(
                    resident.metadata().allocated_bytes,
                    offsets_byte_offset,
                    bytes_byte_offset,
                    bytes_len,
                    n,
                )?;
                let needle_lease = primary.lease_device_buffer(needle.len().max(1))?;
                if !needle.is_empty() {
                    check_cuda(unsafe {
                        cu_memcpy_htod(
                            needle_lease.ptr,
                            needle.as_ptr().cast::<c_void>(),
                            needle.len(),
                        )
                    })?;
                }
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = resident_base;
                let mut a1 = offsets_byte_offset;
                let mut a2 = bytes_byte_offset;
                let mut a3 = text_bytes_limit;
                let mut a4 = needle_lease.ptr;
                let mut a5 = needle.len() as u64;
                let mut a6 = u32::from(negate);
                let mut a7 = n;
                let mut a8 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u64).cast::<c_void>(),
                    (&mut a3 as *mut u64).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                    (&mut a5 as *mut u64).cast::<c_void>(),
                    (&mut a6 as *mut u32).cast::<c_void>(),
                    (&mut a7 as *mut u64).cast::<c_void>(),
                    (&mut a8 as *mut u64).cast::<c_void>(),
                ];
                launch(function, &mut args)?;
                stack.push(out);
            }
            ExprStep::TextCmpMask {
                offsets_byte_offset,
                bytes_byte_offset,
                bytes_len,
                needle_idx,
                scalar_on_left,
                cmp,
            } => {
                // textcol[i] <cmp> needle (lexicographic unsigned bytes, shorter-first) -> i32 mask
                // pushed on the stack. Same needle H2D discipline as TextEqMask (lease >= 1 byte; the
                // kernel never reads the pointer when needle_len == 0 since minlen == 0).
                let needle = text_needles.get(needle_idx as usize).ok_or(
                    CudaRuntimeProbeError::InvalidInputLength(needle_idx as usize),
                )?;
                let function =
                    text_cmp_mask_fn.ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let text_bytes_limit = validate_text_windows(
                    resident.metadata().allocated_bytes,
                    offsets_byte_offset,
                    bytes_byte_offset,
                    bytes_len,
                    n,
                )?;
                let needle_lease = primary.lease_device_buffer(needle.len().max(1))?;
                if !needle.is_empty() {
                    check_cuda(unsafe {
                        cu_memcpy_htod(
                            needle_lease.ptr,
                            needle.as_ptr().cast::<c_void>(),
                            needle.len(),
                        )
                    })?;
                }
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = resident_base;
                let mut a1 = offsets_byte_offset;
                let mut a2 = bytes_byte_offset;
                let mut a3 = text_bytes_limit;
                let mut a4 = needle_lease.ptr;
                let mut a5 = needle.len() as u64;
                let mut a6 = u32::from(scalar_on_left);
                let mut a7 = cmp;
                let mut a8 = n;
                let mut a9 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u64).cast::<c_void>(),
                    (&mut a3 as *mut u64).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                    (&mut a5 as *mut u64).cast::<c_void>(),
                    (&mut a6 as *mut u32).cast::<c_void>(),
                    (&mut a7 as *mut u32).cast::<c_void>(),
                    (&mut a8 as *mut u64).cast::<c_void>(),
                    (&mut a9 as *mut u64).cast::<c_void>(),
                ];
                launch(function, &mut args)?;
                stack.push(out);
            }
            ExprStep::UuidCmpMask {
                byte_offset,
                needle_idx,
                scalar_on_left,
                cmp,
            } => {
                // uuid[i] <cmp> needle (unsigned big-endian 16-byte memcmp) -> i32 mask pushed on the
                // stack. The needle MUST be exactly 16 bytes (the kernel reads a fixed 16); a
                // wrong-length needle is a compile-side bug surfaced loudly here.
                let needle = text_needles.get(needle_idx as usize).ok_or(
                    CudaRuntimeProbeError::InvalidInputLength(needle_idx as usize),
                )?;
                if needle.len() != 16 {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(needle.len()));
                }
                let function =
                    uuid_cmp_mask_fn.ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let needle_lease = primary.lease_device_buffer(16)?;
                check_cuda(unsafe {
                    cu_memcpy_htod(
                        needle_lease.ptr,
                        needle.as_ptr().cast::<c_void>(),
                        needle.len(),
                    )
                })?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = resident_base;
                let mut a1 = byte_offset;
                let mut a2 = needle_lease.ptr;
                let mut a3 = u32::from(scalar_on_left);
                let mut a4 = cmp;
                let mut a5 = n;
                let mut a6 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u64).cast::<c_void>(),
                    (&mut a3 as *mut u32).cast::<c_void>(),
                    (&mut a4 as *mut u32).cast::<c_void>(),
                    (&mut a5 as *mut u64).cast::<c_void>(),
                    (&mut a6 as *mut u64).cast::<c_void>(),
                ];
                launch(function, &mut args)?;
                stack.push(out);
            }
            ExprStep::TextCmpColumnsMask {
                a_offsets_byte_offset,
                a_bytes_byte_offset,
                b_offsets_byte_offset,
                b_bytes_byte_offset,
                cmp,
            } => {
                // text_a[i] <cmp> text_b[i] (per-row lexicographic unsigned byte memcmp, shorter
                // sorts first) -> i32 mask pushed on the stack. ABI mirrors the kernel param
                // order EXACTLY: (resident_ptr, a_offsets, a_bytes, b_offsets, b_bytes,
                // comparison, n, out_mask_ptr).
                let function =
                    text_cmp_columns_mask_fn.ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = resident_base;
                let mut a1 = a_offsets_byte_offset;
                let mut a2 = a_bytes_byte_offset;
                let mut a3 = b_offsets_byte_offset;
                let mut a4 = b_bytes_byte_offset;
                let mut a5 = cmp;
                let mut a6 = n;
                let mut a7 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u64).cast::<c_void>(),
                    (&mut a3 as *mut u64).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                    (&mut a5 as *mut u32).cast::<c_void>(),
                    (&mut a6 as *mut u64).cast::<c_void>(),
                    (&mut a7 as *mut u64).cast::<c_void>(),
                ];
                launch(function, &mut args)?;
                stack.push(out);
            }
            ExprStep::UuidCmpColumnsMask {
                a_byte_offset,
                b_byte_offset,
                cmp,
            } => {
                // uuid_a[i] <cmp> uuid_b[i] (unsigned big-endian 16-byte memcmp per row) -> i32
                // mask pushed on the stack. ABI mirrors the standalone
                // `launch_cuda_resident_uuid_compare_columns_filter` launcher EXACTLY:
                // (resident_ptr, a_byte_offset, b_byte_offset, comparison, n, out_mask_ptr).
                let function =
                    uuid_cmp_columns_mask_fn.ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = resident_base;
                let mut a1 = a_byte_offset;
                let mut a2 = b_byte_offset;
                let mut a3 = cmp;
                let mut a4 = n;
                let mut a5 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u64).cast::<c_void>(),
                    (&mut a3 as *mut u32).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                    (&mut a5 as *mut u64).cast::<c_void>(),
                ];
                launch(function, &mut args)?;
                stack.push(out);
            }
            ExprStep::TextLikeMask {
                offsets_byte_offset,
                bytes_byte_offset,
                bytes_len,
                pattern_idx,
            } => {
                // (textcol[i] LIKE pattern) -> i32 mask pushed on the stack; the VM combines it with AND/OR
                // (and a following validity MaskBinary for NULL 3VL) like any other mask. The pattern is the
                // LE-serialized u32 token array `text_needles[pattern_idx]` (`ntok = len/4`), uploaded H2D
                // into a fresh lease (>=1 byte so the pointer is valid for the empty pattern, never read).
                let pattern = text_needles.get(pattern_idx as usize).ok_or(
                    CudaRuntimeProbeError::InvalidInputLength(pattern_idx as usize),
                )?;
                if pattern.len() % std::mem::size_of::<u32>() != 0 {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(pattern.len()));
                }
                // ABI/window discipline must match the standalone LIKE launcher exactly. The VM used
                // to omit `text_bytes_limit`, shifting every following argument left by one: the token
                // pointer was interpreted as a byte limit and the final output pointer was read from an
                // absent eighth parameter, causing a deterministic CUDA 700 on nullable-text LIKE.
                let text_bytes_limit = validate_text_windows(
                    resident.metadata().allocated_bytes,
                    offsets_byte_offset,
                    bytes_byte_offset,
                    bytes_len,
                    n,
                )?;
                let ntok = (pattern.len() / std::mem::size_of::<u32>()) as u64;
                let function =
                    text_like_mask_fn.ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let pattern_lease = primary.lease_device_buffer(pattern.len().max(1))?;
                if !pattern.is_empty() {
                    check_cuda(unsafe {
                        cu_memcpy_htod(
                            pattern_lease.ptr,
                            pattern.as_ptr().cast::<c_void>(),
                            pattern.len(),
                        )
                    })?;
                }
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = resident_base;
                let mut a1 = offsets_byte_offset;
                let mut a2 = bytes_byte_offset;
                let mut a3 = text_bytes_limit;
                let mut a4 = pattern_lease.ptr;
                let mut a5 = ntok;
                let mut a6 = n;
                let mut a7 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u64).cast::<c_void>(),
                    (&mut a3 as *mut u64).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                    (&mut a5 as *mut u64).cast::<c_void>(),
                    (&mut a6 as *mut u64).cast::<c_void>(),
                    (&mut a7 as *mut u64).cast::<c_void>(),
                ];
                launch(function, &mut args)?;
                stack.push(out);
            }
            ExprStep::BoolMask {
                bitmap_byte_offset,
                negate,
            } => {
                // bitmap[i] ^ negate -> i32 mask pushed on the stack; the VM combines it with AND/OR.
                let function = bool_mask_fn.ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
                let out = primary.lease_device_buffer(byte_len)?;
                let mut a0 = resident_base;
                let mut a1 = bitmap_byte_offset;
                let mut a2 = u32::from(negate);
                let mut a3 = n;
                let mut a4 = out.ptr;
                let mut args = [
                    (&mut a0 as *mut u64).cast::<c_void>(),
                    (&mut a1 as *mut u64).cast::<c_void>(),
                    (&mut a2 as *mut u32).cast::<c_void>(),
                    (&mut a3 as *mut u64).cast::<c_void>(),
                    (&mut a4 as *mut u64).cast::<c_void>(),
                ];
                launch(function, &mut args)?;
                stack.push(out);
            }
            ExprStep::ConstMask { value } => {
                // A constant per-row i32 mask (no kernel, no host materialization) -- `col IS NULL`/
                // `IS NOT NULL` on a column with no validity bitmap. Fill exactly `n` i32 words with 0/1
                // on a pooled stream; the helper drains before this lease can return to the buffer pool.
                let out = primary.lease_device_buffer(byte_len)?;
                let memset =
                    cu_memset_d32_async.ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
                launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
                    memset(out.ptr, u32::from(value), n_usize, stream)
                })?;
                stack.push(out);
            }
        }
    }

    // Every arithmetic launch has synced; read the shared overflow flag once. Any overflowing op set
    // it, so the whole query must error like Postgres (int4 -> "integer out of range", int8 ->
    // "bigint out of range") rather than return rows computed from a silently-wrapped value.
    let mut overflow = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut overflow as *mut u32).cast::<c_void>(),
            overflow_buf.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;
    if overflow != 0 {
        return Err(match elem {
            ResidentElemType::I32 => CudaRuntimeProbeError::IntegerOutOfRange,
            ResidentElemType::I64 => CudaRuntimeProbeError::BigintOutOfRange,
            ResidentElemType::I128 => CudaRuntimeProbeError::NumericFieldOverflow,
        });
    }

    Ok(stack)
}
