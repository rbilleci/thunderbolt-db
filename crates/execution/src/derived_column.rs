use std::marker::PhantomData;
use std::os::raw::c_void;

use super::resident_window::{CudaGroupTextSource, validate_aligned_window, validate_text_windows};
use super::{
    CudaResidentDeviceMemory, CudaRuntimeProbeError, ExprStep, ExprTerminal, PooledBufferLease,
    ResidentElemType, check_cuda, launch_on_pooled_stream, run_resident_arith_program,
};

#[derive(Debug, Clone, Copy)]
pub struct CudaGroupDeviceView<'a> {
    pub(super) ptr: u64,
    pub(super) initialized_bytes: u64,
    pub(super) context_identity: usize,
    _owner: PhantomData<&'a ()>,
}

impl CudaGroupDeviceView<'_> {
    pub(super) fn new(ptr: u64, initialized_bytes: u64, context_identity: usize) -> Self {
        Self {
            ptr,
            initialized_bytes,
            context_identity,
            _owner: PhantomData,
        }
    }
}

/// Typed source for one member of a device-built fixed-width composite key.
#[derive(Debug, Clone, Copy)]
pub enum CudaWideKeySource<'a> {
    ResidentI32 { byte_offset: u64 },
    ResidentI64 { byte_offset: u64 },
    ResidentI128 { byte_offset: u64 },
    ResidentBool { bitmap_byte_offset: u64 },
    DerivedI32 { buffer: CudaGroupDeviceView<'a> },
    DerivedI64 { buffer: CudaGroupDeviceView<'a> },
}

/// Destination geometry for one member of a device-built fixed-width composite key.
#[derive(Debug, Clone, Copy)]
pub struct CudaWideKeyDescriptor<'a> {
    pub source: CudaWideKeySource<'a>,
    pub destination_byte_offset: u64,
}

/// NULL state for a wide-key member. A non-empty validity slice reserves a trailing u64 word in each
/// output row and must contain exactly one entry per descriptor.
#[derive(Debug, Clone, Copy)]
pub enum CudaWideKeyValidity {
    NonNullable,
    Bitmap { byte_offset: u64 },
}

/// Evaluate an arithmetic `program` over all `n_rows`, then GATHER the resulting i32 value column at
/// the survivor `indices` (sign-extended to i64). The ORDER BY-expression key column: the device Expr
/// interpreter computes `a+b` etc. with CHECKED int4 arithmetic (overflow -> `IntegerOutOfRange`,
/// inherited from `run_resident_arith_program` -- no wrap, no CPU), and the result feeds the GPU sort
/// exactly like a materialized int key.
/// A resident arith-program result kept ON-DEVICE (the value buffer, one element per row) + its pooled
/// lease. Returned by [`CudaResidentDeviceMemory::arith_value_column_device`]; callers bind its typed
/// [`CudaGroupDeviceView`] into a later GROUP BY launch, so this owner MUST stay alive across that
/// launch. Dropping it returns the buffer to the pool.
pub struct DeviceArithBuffer<'a> {
    _lease: PooledBufferLease<'a>,
    ptr: u64,
    initialized_bytes: u64,
}

impl DeviceArithBuffer<'_> {
    fn new(lease: PooledBufferLease<'_>, initialized_bytes: usize) -> DeviceArithBuffer<'_> {
        debug_assert!(initialized_bytes <= lease.capacity);
        let ptr = lease.ptr;
        DeviceArithBuffer {
            _lease: lease,
            ptr,
            initialized_bytes: initialized_bytes as u64,
        }
    }

    pub(super) fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Borrow this allocation with its exact initialized extent and originating CUDA context.
    pub fn group_view(&self) -> CudaGroupDeviceView<'_> {
        CudaGroupDeviceView::new(
            self.ptr,
            self.initialized_bytes,
            self._lease.primary_identity(),
        )
    }
}

/// Run an arith program over all `n_rows` and return the result value buffer RESIDENT (no D2H). See
/// [`CudaResidentDeviceMemory::arith_value_column_device`]. cuCtxSynchronize'd so a later launch reads
/// valid keys. Checked overflow -> PG error inherited from `run_resident_arith_program`.
pub(super) fn launch_cuda_arith_value_column_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    program: &[ExprStep],
    n_rows: u64,
    elem: ResidentElemType,
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    let n = usize::try_from(n_rows).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let mut stack =
        run_resident_arith_program(resident, program, &[], n_rows, elem, ExprTerminal::Value)?;
    let value = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        // A well-formed arithmetic program leaves exactly one value on the stack.
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    // Block until the arith program finishes, so the SEPARATE GROUP BY launch that reads this buffer
    // sees the completed keys (not a racing/stale buffer).
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    let initialized_bytes = n
        .checked_mul(elem.elem_size())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    Ok(DeviceArithBuffer::new(value, initialized_bytes))
}

/// Run `gpu_db_resident_bool_to_mask` (negate=0) into a leased int4 buffer (it writes 0/1 per row) and
/// return it as a `DeviceArithBuffer` -- the derived int4 column for a bool GROUP BY key / bool MIN/MAX
/// value. Synchronizes so a separate GROUP BY launch borrowing its typed view sees the completed
/// buffer, not a racing/stale one. The lease lives in the returned buffer (caller-owned).
pub(super) fn launch_cuda_bool_to_int4_column_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    bitmap_byte_offset: u64,
    n: u64,
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let bitmap_words = n
        .checked_add(31)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?
        / 32;
    validate_aligned_window(
        resident.metadata().allocated_bytes,
        bitmap_byte_offset,
        bitmap_words,
        4,
        4,
    )?;
    let out_bytes = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_resident_bool_to_mask", &ptx)?;
    let out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = bitmap_byte_offset;
    let mut a2 = 0u32; // negate = false: bool true -> int4 1, false -> 0
    let mut a3 = n;
    let mut a4 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u32).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            kernel_fn,
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
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok(DeviceArithBuffer::new(out, out_bytes))
}

/// Run `gpu_db_pack_two_int4_cols` into a leased i64 buffer (col0<<32 | col1 per row) and return it as a
/// `DeviceArithBuffer` -- the derived composite GROUP BY key. cuCtxSynchronize'd so the SEPARATE GROUP BY
/// launch borrowing its typed view sees the completed buffer, not a racing/stale one.
pub(super) fn launch_cuda_pack_two_int4_cols_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    off0: u64,
    off1: u64,
    n: u64,
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    validate_aligned_window(resident.metadata().allocated_bytes, off0, n, 4, 4)?;
    validate_aligned_window(resident.metadata().allocated_bytes, off1, n, 4, 4)?;
    let out_bytes = n_usize
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_pack_two_int4_cols", &ptx)?;
    let out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = off0;
    let mut a2 = off1;
    let mut a3 = n;
    let mut a4 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            kernel_fn,
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
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok(DeviceArithBuffer::new(out, out_bytes))
}

/// Run `gpu_db_pack_two_cols_i128` into a leased [i128; n] buffer (col0 high 64 bits, col1 low 64) and
/// return it as a `DeviceArithBuffer` -- the derived composite GROUP BY key for a wider (int8/timestamp
/// member) composite. `w0`/`w1` are each member's read width (4 or 8). cuCtxSynchronize'd so the
/// separate b128 GROUP BY launch borrowing its typed view sees the completed buffer.
pub(super) fn launch_cuda_pack_two_cols_i128_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    off0: u64,
    w0: u64,
    off1: u64,
    w1: u64,
    n: u64,
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if (w0 != 4 && w0 != 8) || (w1 != 4 && w1 != 8) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    validate_aligned_window(resident.metadata().allocated_bytes, off0, n, w0, 4)?;
    validate_aligned_window(resident.metadata().allocated_bytes, off1, n, w1, 4)?;
    let out_bytes = n_usize
        .checked_mul(std::mem::size_of::<i128>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_pack_two_cols_i128", &ptx)?;
    let out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = off0;
    let mut a2 = w0;
    let mut a3 = off1;
    let mut a4 = w1;
    let mut a5 = n;
    let mut a6 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
        (&mut a6 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            kernel_fn,
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
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok(DeviceArithBuffer::new(out, out_bytes))
}

/// Run `gpu_db_widen_col_to_i64` into a leased [i64; n] buffer (the column sign-extended to i64) for the
/// fixed member of a composite (fixed-width, text) GROUP BY key. `w` = 4 or 8. cuCtxSynchronize'd so the
/// separate text-key GROUP BY launch borrowing its typed view reads the completed buffer.
pub(super) fn launch_cuda_widen_col_to_i64_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    off: u64,
    w: u64,
    n: u64,
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if w != 4 && w != 8 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    validate_aligned_window(resident.metadata().allocated_bytes, off, n, w, 4)?;
    let out_bytes = n_usize
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_widen_col_to_i64", &ptx)?;
    let out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = off;
    let mut a2 = w;
    let mut a3 = n;
    let mut a4 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            kernel_fn,
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
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok(DeviceArithBuffer::new(out, out_bytes))
}

/// Run `gpu_db_build_wide_key` into a leased [u8; wbytes*n] buffer (the all-fixed composite wide key per
/// row) and return it. `descriptors` = (kind, src_off, dst_off) per member, uploaded as 3 u64 each.
/// cuCtxSynchronize'd so the separate GROUP BY launch borrowing its typed view reads completed data.
pub(super) fn launch_cuda_upload_u64_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    data: &[u64],
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    if data.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let bytes = std::mem::size_of_val(data);
    let primary = resident.primary();
    primary.set_current()?;
    let buf = primary.lease_device_buffer(bytes)?;
    let cu_memcpy_htod = unsafe {
        *primary
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    // cuMemcpyHtoD is host-synchronous: the data is fully on device when it returns, so the GROUP BY
    // kernel (on the pooled stream) sees it without a further sync.
    check_cuda(unsafe { cu_memcpy_htod(buf.ptr, data.as_ptr().cast::<c_void>(), bytes) })?;
    Ok(DeviceArithBuffer::new(buf, bytes))
}

fn validate_wide_key_descriptors(
    resident: &CudaResidentDeviceMemory,
    descriptors: &[CudaWideKeyDescriptor<'_>],
    wbytes: u64,
    n: u64,
    validity: &[CudaWideKeyValidity],
) -> Result<(Vec<u64>, Vec<u64>, u64), CudaRuntimeProbeError> {
    validate_wide_key_descriptor_parts(
        resident.metadata().allocated_bytes,
        std::ptr::from_ref(resident.primary()).addr(),
        descriptors,
        wbytes,
        n,
        validity,
    )
}

fn validate_wide_key_descriptor_parts(
    resident_bytes: u64,
    context_identity: usize,
    descriptors: &[CudaWideKeyDescriptor<'_>],
    wbytes: u64,
    n: u64,
    validity: &[CudaWideKeyValidity],
) -> Result<(Vec<u64>, Vec<u64>, u64), CudaRuntimeProbeError> {
    if n == 0 || descriptors.is_empty() || wbytes == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if !validity.is_empty() && (validity.len() != descriptors.len() || descriptors.len() > 64) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(validity.len()));
    }
    let bitmap_words = n
        .checked_add(31)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?
        / 32;
    let value_limit = if validity.is_empty() {
        wbytes
    } else {
        wbytes
            .checked_sub(8)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(wbytes).unwrap_or(usize::MAX),
            ))?
    };
    let mut flat = Vec::with_capacity(descriptors.len() * 3);
    let mut destination_spans = Vec::with_capacity(descriptors.len());
    let mut derived_ptr = None;
    for descriptor in descriptors {
        let destination = descriptor.destination_byte_offset;
        if destination % 8 != 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(destination).unwrap_or(usize::MAX),
            ));
        }
        let (kind, source_offset, source_width) = match descriptor.source {
            CudaWideKeySource::ResidentI32 { byte_offset } => (0, byte_offset, 4),
            CudaWideKeySource::ResidentI64 { byte_offset } => (1, byte_offset, 8),
            CudaWideKeySource::ResidentI128 { byte_offset } => (2, byte_offset, 16),
            CudaWideKeySource::ResidentBool { bitmap_byte_offset } => {
                validate_aligned_window(resident_bytes, bitmap_byte_offset, bitmap_words, 4, 4)?;
                (3, bitmap_byte_offset, 0)
            }
            CudaWideKeySource::DerivedI32 { buffer } => {
                let required = n
                    .checked_mul(4)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                if buffer.context_identity != context_identity
                    || buffer.initialized_bytes < required
                    || derived_ptr.is_some_and(|ptr| ptr != buffer.ptr)
                {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(
                        usize::try_from(required).unwrap_or(usize::MAX),
                    ));
                }
                derived_ptr = Some(buffer.ptr);
                (4, 0, 0)
            }
            CudaWideKeySource::DerivedI64 { buffer } => {
                let required = n
                    .checked_mul(8)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                if buffer.context_identity != context_identity
                    || buffer.initialized_bytes < required
                    || derived_ptr.is_some_and(|ptr| ptr != buffer.ptr)
                {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(
                        usize::try_from(required).unwrap_or(usize::MAX),
                    ));
                }
                derived_ptr = Some(buffer.ptr);
                (5, 0, 0)
            }
        };
        if source_width != 0 {
            validate_aligned_window(resident_bytes, source_offset, n, source_width, 4)?;
        }
        let output_width = if kind == 2 { 16 } else { 8 };
        let end = destination
            .checked_add(output_width)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if end > value_limit
            || destination_spans
                .iter()
                .any(|&(start, prior_end)| destination < prior_end && start < end)
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(end).unwrap_or(usize::MAX),
            ));
        }
        destination_spans.push((destination, end));
        flat.extend_from_slice(&[kind, source_offset, destination]);
    }
    destination_spans.sort_unstable_by_key(|&(start, _)| start);
    let mut covered = 0;
    for (start, end) in destination_spans {
        if start != covered {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(start).unwrap_or(usize::MAX),
            ));
        }
        covered = end;
    }
    if covered != value_limit {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(covered).unwrap_or(usize::MAX),
        ));
    }
    let mut validity_flat = Vec::with_capacity(validity.len());
    for state in validity {
        match *state {
            CudaWideKeyValidity::NonNullable => validity_flat.push(u64::MAX),
            CudaWideKeyValidity::Bitmap { byte_offset } => {
                validate_aligned_window(resident_bytes, byte_offset, bitmap_words, 4, 4)?;
                validity_flat.push(byte_offset);
            }
        }
    }
    Ok((flat, validity_flat, derived_ptr.unwrap_or(0)))
}

pub(super) fn launch_cuda_build_wide_key_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    descriptors: &[CudaWideKeyDescriptor<'_>],
    wbytes: u64,
    n: u64,
    validity: &[CudaWideKeyValidity],
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    let (desc_flat, validity_flat, derived_ptr) =
        validate_wide_key_descriptors(resident, descriptors, wbytes, n, validity)?;
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let wbytes_usize =
        usize::try_from(wbytes).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let out_bytes = n_usize
        .checked_mul(wbytes_usize)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let desc_bytes = std::mem::size_of_val(desc_flat.as_slice());
    let n_members = descriptors.len() as u64;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_build_wide_key", &ptx)?;
    let desc_dev = primary.lease_device_buffer(desc_bytes)?;
    // M3 (doc 21): the per-member validity offsets (one u64 per member). Leased + uploaded only when
    // present (nullable composite); else `vdesc_ptr` stays 0 (the kernel skips all validity handling).
    let vdesc_bytes = std::mem::size_of_val(validity_flat.as_slice());
    let vdesc_dev = if vdesc_bytes > 0 {
        Some(primary.lease_device_buffer(vdesc_bytes)?)
    } else {
        None
    };
    let vdesc_ptr = vdesc_dev.as_ref().map_or(0, |b| b.ptr);
    let out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                desc_dev.ptr,
                desc_flat.as_ptr().cast::<c_void>(),
                desc_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        if let Some(vdev) = &vdesc_dev {
            let rc = unsafe {
                htod_async(
                    vdev.ptr,
                    validity_flat.as_ptr().cast::<c_void>(),
                    vdesc_bytes,
                    stream,
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        let mut a0 = resident.device_ptr();
        let mut a1 = desc_dev.ptr;
        let mut a2 = n_members;
        let mut a3 = wbytes;
        let mut a4 = n;
        let mut a5 = out.ptr;
        let mut a6 = derived_ptr;
        let mut a7 = vdesc_ptr;
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
        unsafe {
            cu_launch_kernel(
                kernel_fn,
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
        }
    })?;
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok(DeviceArithBuffer::new(out, out_bytes))
}

/// COUNT(DISTINCT v) mark pass (see [`CudaResidentDeviceMemory::mark_new_distinct_device`]). Uploads
/// the sorted `k`-wide i64 tuple matrix (key0 = group key) + the permutation, runs
/// `gpu_db_mark_new_distinct`, and returns
/// the two derived i64 device columns `(g_sorted, new_distinct)`. cuCtxSynchronize'd so the SEPARATE
/// GROUP BY launch reads completed buffers (a fully-drained launch, off the bool-GROUP-BY hazard). The
/// `keys`/`perm` upload buffers are temporary -- the synchronize guarantees the kernel read them before
/// they return to the pool at function end; the output leases live in the returned buffers.
fn validate_permutation(perm: &[u32], n: usize) -> Result<(), CudaRuntimeProbeError> {
    if perm.len() != n {
        return Err(CudaRuntimeProbeError::InvalidInputLength(perm.len()));
    }
    let mut seen = vec![false; n];
    for &position in perm {
        let position = usize::try_from(position)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if position >= n || std::mem::replace(&mut seen[position], true) {
            return Err(CudaRuntimeProbeError::InvalidInputLength(position));
        }
    }
    Ok(())
}

pub(super) fn launch_cuda_mark_new_distinct_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    keys: &[i64],
    perm: &[u32],
    n: u64,
    k: usize,
) -> Result<(DeviceArithBuffer<'r>, DeviceArithBuffer<'r>), CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n_usize == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if k < 2 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(k));
    }
    let expected_keys = n_usize
        .checked_mul(k)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    if keys.len() != expected_keys {
        return Err(CudaRuntimeProbeError::InvalidInputLength(keys.len()));
    }
    validate_permutation(perm, n_usize)?;
    let keys_bytes = keys
        .len()
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(keys.len()))?;
    let perm_bytes = n_usize
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let out_bytes = n_usize
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_mark_new_distinct", &ptx)?;
    let keys_dev = primary.lease_device_buffer(keys_bytes)?;
    let perm_dev = primary.lease_device_buffer(perm_bytes)?;
    let g_out = primary.lease_device_buffer(out_bytes)?;
    let nd_out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = (n_usize.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                keys_dev.ptr,
                keys.as_ptr().cast::<c_void>(),
                keys_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                perm_dev.ptr,
                perm.as_ptr().cast::<c_void>(),
                perm_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let mut a0 = keys_dev.ptr;
        let mut a1 = perm_dev.ptr;
        let mut a2 = n;
        let mut a3 = k as u64;
        let mut a4 = g_out.ptr;
        let mut a5 = nd_out.ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast::<c_void>(),
            (&mut a1 as *mut u64).cast::<c_void>(),
            (&mut a2 as *mut u64).cast::<c_void>(),
            (&mut a3 as *mut u64).cast::<c_void>(),
            (&mut a4 as *mut u64).cast::<c_void>(),
            (&mut a5 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                kernel_fn,
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
        }
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok((
        DeviceArithBuffer::new(g_out, out_bytes),
        DeviceArithBuffer::new(nd_out, out_bytes),
    ))
}

/// COUNT(DISTINCT v) mark pass for a TEXT value (see
/// [`CudaResidentDeviceMemory::mark_new_distinct_text_device`]). Uploads the hetero-sort permutation,
/// the surviving absolute rows, and the per-position group key; runs `gpu_db_mark_new_distinct_text`
/// (which reads the value text from the resident payload via text_off/text_bytes), and returns the two
/// derived i64 device columns `(g_sorted, new_distinct)`. cuCtxSynchronize'd so the SEPARATE GROUP BY
/// launch reads completed buffers (a fully-drained launch, off the bool-GROUP-BY hazard).
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_cuda_mark_new_distinct_text_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    perm: &[u32],
    indices: &[u64],
    g_keys: &[i64],
    text: CudaGroupTextSource,
    n: u64,
) -> Result<(DeviceArithBuffer<'r>, DeviceArithBuffer<'r>), CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n_usize == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    validate_permutation(perm, n_usize)?;
    if indices.len() != n_usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(indices.len()));
    }
    if g_keys.len() != n_usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(g_keys.len()));
    }
    validate_text_windows(
        resident.metadata().allocated_bytes,
        text.offsets_byte_offset,
        text.bytes_byte_offset,
        text.bytes_len,
        text.row_count,
    )?;
    if let Some(&row) = indices.iter().find(|&&row| row >= text.row_count) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(row).unwrap_or(usize::MAX),
        ));
    }
    let perm_bytes = n_usize
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let indices_bytes = n_usize
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let gkeys_bytes = n_usize
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let out_bytes = gkeys_bytes;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let memset_d8_async = primary
        .cu_memset_d8_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_mark_new_distinct_text", &ptx)?;
    let validate_fn = primary.cached_function(c"gpu_db_validate_distinct_text_offsets", &ptx)?;
    let resident_base = resident.device_ptr();
    let indices_dev = primary.lease_device_buffer(indices_bytes)?;
    let validation_error = primary.lease_device_buffer(std::mem::size_of::<u32>())?;
    const BLOCK: u32 = 256;
    let grid = (n_usize.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                indices_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc =
            unsafe { memset_d8_async(validation_error.ptr, 0, std::mem::size_of::<u32>(), stream) };
        if rc != 0 {
            return rc;
        }
        let mut a0 = resident_base;
        let mut a1 = text.offsets_byte_offset;
        let mut a2 = text.bytes_len;
        let mut a3 = indices_dev.ptr;
        let mut a4 = n;
        let mut a5 = validation_error.ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast::<c_void>(),
            (&mut a1 as *mut u64).cast::<c_void>(),
            (&mut a2 as *mut u64).cast::<c_void>(),
            (&mut a3 as *mut u64).cast::<c_void>(),
            (&mut a4 as *mut u64).cast::<c_void>(),
            (&mut a5 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                validate_fn,
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
        }
    })?;
    let mut invalid_offsets = 0_u32;
    check_cuda(unsafe {
        (primary.cu_memcpy_dtoh)(
            (&mut invalid_offsets as *mut u32).cast::<c_void>(),
            validation_error.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;
    if invalid_offsets != 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(text.bytes_len).unwrap_or(usize::MAX),
        ));
    }
    let perm_dev = primary.lease_device_buffer(perm_bytes)?;
    let gkeys_dev = primary.lease_device_buffer(gkeys_bytes)?;
    let g_out = primary.lease_device_buffer(out_bytes)?;
    let nd_out = primary.lease_device_buffer(out_bytes)?;
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                perm_dev.ptr,
                perm.as_ptr().cast::<c_void>(),
                perm_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                gkeys_dev.ptr,
                g_keys.as_ptr().cast::<c_void>(),
                gkeys_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let mut a0 = perm_dev.ptr;
        let mut a1 = indices_dev.ptr;
        let mut a2 = gkeys_dev.ptr;
        let mut a3 = resident_base;
        let mut a4 = text.offsets_byte_offset;
        let mut a5 = text.bytes_byte_offset;
        let mut a6 = n;
        let mut a7 = g_out.ptr;
        let mut a8 = nd_out.ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast::<c_void>(),
            (&mut a1 as *mut u64).cast::<c_void>(),
            (&mut a2 as *mut u64).cast::<c_void>(),
            (&mut a3 as *mut u64).cast::<c_void>(),
            (&mut a4 as *mut u64).cast::<c_void>(),
            (&mut a5 as *mut u64).cast::<c_void>(),
            (&mut a6 as *mut u64).cast::<c_void>(),
            (&mut a7 as *mut u64).cast::<c_void>(),
            (&mut a8 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                kernel_fn,
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
        }
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok((
        DeviceArithBuffer::new(g_out, out_bytes),
        DeviceArithBuffer::new(nd_out, out_bytes),
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        CudaGroupDeviceView, CudaWideKeyDescriptor, CudaWideKeySource, CudaWideKeyValidity,
        validate_permutation, validate_wide_key_descriptor_parts,
    };

    #[test]
    fn distinct_permutations_are_exact() {
        assert!(validate_permutation(&[2, 0, 1], 3).is_ok());
        assert!(validate_permutation(&[0, 0, 2], 3).is_err());
        assert!(validate_permutation(&[0, 1, 3], 3).is_err());
        assert!(validate_permutation(&[0, 1], 3).is_err());
    }

    #[test]
    fn wide_key_descriptors_check_windows_geometry_and_validity() {
        let descriptors = [
            CudaWideKeyDescriptor {
                source: CudaWideKeySource::ResidentI32 { byte_offset: 0 },
                destination_byte_offset: 0,
            },
            CudaWideKeyDescriptor {
                source: CudaWideKeySource::ResidentI128 { byte_offset: 16 },
                destination_byte_offset: 8,
            },
        ];
        let validity = [
            CudaWideKeyValidity::NonNullable,
            CudaWideKeyValidity::Bitmap { byte_offset: 80 },
        ];
        let (flat, valid_flat, derived) =
            validate_wide_key_descriptor_parts(84, 7, &descriptors, 32, 4, &validity)
                .expect("boundary-valid descriptors");
        assert_eq!(flat, [0, 0, 0, 2, 16, 8]);
        assert_eq!(valid_flat, [u64::MAX, 80]);
        assert_eq!(derived, 0);

        let overlapping = [
            descriptors[0],
            CudaWideKeyDescriptor {
                source: CudaWideKeySource::ResidentI64 { byte_offset: 0 },
                destination_byte_offset: 0,
            },
        ];
        assert!(validate_wide_key_descriptor_parts(64, 7, &overlapping, 16, 4, &[]).is_err());
        assert!(validate_wide_key_descriptor_parts(84, 7, &descriptors, 40, 4, &validity).is_err());
        assert!(validate_wide_key_descriptor_parts(83, 7, &descriptors, 32, 4, &validity).is_err());
        assert!(validate_wide_key_descriptor_parts(84, 7, &descriptors, 24, 4, &validity).is_err());
        assert!(
            validate_wide_key_descriptor_parts(84, 7, &descriptors, 32, 4, &validity[..1]).is_err()
        );

        let misaligned_source = [CudaWideKeyDescriptor {
            source: CudaWideKeySource::ResidentI32 { byte_offset: 1 },
            destination_byte_offset: 0,
        }];
        assert!(validate_wide_key_descriptor_parts(64, 7, &misaligned_source, 8, 4, &[]).is_err());
        let misaligned_bool = [CudaWideKeyDescriptor {
            source: CudaWideKeySource::ResidentBool {
                bitmap_byte_offset: 1,
            },
            destination_byte_offset: 0,
        }];
        assert!(validate_wide_key_descriptor_parts(64, 7, &misaligned_bool, 8, 4, &[]).is_err());
        assert!(
            validate_wide_key_descriptor_parts(
                64,
                7,
                &descriptors[..1],
                16,
                4,
                &[CudaWideKeyValidity::Bitmap { byte_offset: 1 }],
            )
            .is_err()
        );
    }

    #[test]
    fn wide_key_derived_views_require_context_and_initialized_extent() {
        let valid_view = CudaGroupDeviceView::new(99, 32, 7);
        let descriptor = [CudaWideKeyDescriptor {
            source: CudaWideKeySource::DerivedI64 { buffer: valid_view },
            destination_byte_offset: 0,
        }];
        assert_eq!(
            validate_wide_key_descriptor_parts(0, 7, &descriptor, 8, 4, &[])
                .expect("valid view")
                .2,
            99
        );
        assert!(validate_wide_key_descriptor_parts(0, 8, &descriptor, 8, 4, &[]).is_err());
        let short = [CudaWideKeyDescriptor {
            source: CudaWideKeySource::DerivedI64 {
                buffer: CudaGroupDeviceView::new(99, 31, 7),
            },
            destination_byte_offset: 0,
        }];
        assert!(validate_wide_key_descriptor_parts(0, 7, &short, 8, 4, &[]).is_err());
    }
}
