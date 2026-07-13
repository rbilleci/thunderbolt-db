use std::marker::PhantomData;
use std::os::raw::c_void;

use super::{
    CudaResidentDeviceMemory, CudaRuntimeProbeError, ExprStep, PooledBufferLease, ResidentElemType,
    check_cuda, launch_on_pooled_stream, run_resident_arith_program,
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

/// Evaluate an arithmetic `program` over all `n_rows`, then GATHER the resulting i32 value column at
/// the survivor `indices` (sign-extended to i64). The ORDER BY-expression key column: the device Expr
/// interpreter computes `a+b` etc. with CHECKED int4 arithmetic (overflow -> `IntegerOutOfRange`,
/// inherited from `run_resident_arith_program` -- no wrap, no CPU), and the result feeds the GPU sort
/// exactly like a materialized int key.
/// A resident arith-program result kept ON-DEVICE (the value buffer, one element per row) + its pooled
/// lease. Returned by [`CudaResidentDeviceMemory::arith_value_column_device`]; `device_ptr()` is read by
/// a LATER kernel launch (the GROUP BY group-key via `key_base_override`), so this MUST be kept alive
/// across every such launch -- dropping it returns the buffer to the pool (a UAF under reuse).
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

    /// Device address of the value buffer (one element per row; width = the program's element type).
    pub fn device_ptr(&self) -> u64 {
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
    let mut stack = run_resident_arith_program(resident, program, &[], n_rows, elem)?;
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
/// value. Synchronizes so the SEPARATE GROUP BY launch (which reads it via key/value_base_override) sees
/// the completed buffer, not a racing/stale one. The lease lives in the returned buffer (caller-owned).
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
/// launch (which reads it via key_base_override) sees the completed buffer, not a racing/stale one.
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
/// SEPARATE b128 GROUP BY launch (key_is_i128 + key_base_override) sees the completed buffer.
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
/// SEPARATE text-key GROUP BY launch reads the completed buffer via key_base_override.
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
/// cuCtxSynchronize'd so the SEPARATE GROUP BY launch reads the completed buffer via key_base_override.
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

pub(super) fn launch_cuda_build_wide_key_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    descriptors: &[(u64, u64, u64)],
    wbytes: u64,
    n: u64,
    // A separate per-row buffer for a DERIVED member (the expression group key); read by descriptor
    // kinds 4 (i32) / 5 (i64). 0 when the wide key has only column members.
    derived_ptr: u64,
    // M3 (doc 21): per-member NULL validity offsets (EMPTY = none). See the wrapper's doc.
    validity_descs: &[u64],
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
    if n == 0 || descriptors.is_empty() || wbytes == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let wbytes_usize =
        usize::try_from(wbytes).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let out_bytes = n_usize
        .checked_mul(wbytes_usize)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    // Flatten the descriptors to a [u64] (kind, src_off, dst_off per member) for the device upload.
    let mut desc_flat: Vec<u64> = Vec::with_capacity(descriptors.len() * 3);
    for &(kind, src_off, dst_off) in descriptors {
        desc_flat.push(kind);
        desc_flat.push(src_off);
        desc_flat.push(dst_off);
    }
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
    let vdesc_bytes = std::mem::size_of_val(validity_descs);
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
                    validity_descs.as_ptr().cast::<c_void>(),
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
    if perm.len() != n_usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(perm.len()));
    }
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
    text_off: u64,
    text_bytes: u64,
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
    if perm.len() != n_usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(perm.len()));
    }
    if indices.len() != n_usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(indices.len()));
    }
    if g_keys.len() != n_usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(g_keys.len()));
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
    let resident_base = resident.device_ptr();
    let perm_dev = primary.lease_device_buffer(perm_bytes)?;
    let indices_dev = primary.lease_device_buffer(indices_bytes)?;
    let gkeys_dev = primary.lease_device_buffer(gkeys_bytes)?;
    let g_out = primary.lease_device_buffer(out_bytes)?;
    let nd_out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = (n_usize.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);
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
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                indices_bytes,
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
        let mut a4 = text_off;
        let mut a5 = text_bytes;
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
