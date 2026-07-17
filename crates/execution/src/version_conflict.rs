//! Fixed-size device verdict for exact-key physical-version history.
//!
//! The predicate VM retains exact typed/NULL equality as a device mask. This terminal consumes
//! that mask and version sidecars on-device and reads back one four-byte conflict bit, rather than
//! materializing matching coordinates or version stamps on the host.

use std::os::raw::c_void;
use std::sync::Arc;

use super::{
    check_cuda, CudaPredicateMaskI32, CudaResidentDeviceMemory, CudaResidentReadSource,
    CudaRuntimeProbeError,
};

/// One device-computed exact-history verdict. `readback_bytes` is deliberately part of the
/// contract so callers and tests can prove this path never regresses to coordinate-sized D2H.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaVersionConflictVerdict {
    pub conflict: bool,
    pub readback_bytes: usize,
}

impl CudaResidentDeviceMemory {
    /// Return whether a predicate-matching physical version has a real create/delete stamp newer
    /// than `read_snapshot`. Version sources may be separate resident allocations (hot shards) or
    /// byte ranges in `self` (cold chunks). An absent source uses its supplied uniform default.
    /// `deleted_live` is ignored rather than treated as a write stamp.
    #[allow(clippy::too_many_arguments)]
    pub fn predicate_mask_version_conflict(
        &self,
        mask: &CudaPredicateMaskI32,
        created_by: Option<(&CudaResidentDeviceMemory, u64)>,
        created_default: u64,
        deleted_by: Option<(&CudaResidentDeviceMemory, u64)>,
        deleted_default: u64,
        deleted_live: u64,
        read_snapshot: u64,
    ) -> Result<CudaVersionConflictVerdict, CudaRuntimeProbeError> {
        type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
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

        const VERDICT_BYTES: usize = std::mem::size_of::<u32>();
        const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_predicate_mask_version_conflict(
    .param .u64 mask_ptr,
    .param .u32 row_count,
    .param .u64 created_ptr,
    .param .u64 created_default,
    .param .u64 deleted_ptr,
    .param .u64 deleted_default,
    .param .u64 deleted_live,
    .param .u64 read_snapshot,
    .param .u64 verdict_ptr
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<24>;

    ld.param.u64 %rd1, [mask_ptr];
    ld.param.u32 %r1, [row_count];
    ld.param.u64 %rd2, [created_ptr];
    ld.param.u64 %rd3, [created_default];
    ld.param.u64 %rd4, [deleted_ptr];
    ld.param.u64 %rd5, [deleted_default];
    ld.param.u64 %rd6, [deleted_live];
    ld.param.u64 %rd7, [read_snapshot];
    ld.param.u64 %rd8, [verdict_ptr];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %nctaid.x;
    mad.lo.u32 %r6, %r3, %r4, %r2;
    mul.lo.u32 %r7, %r5, %r4;

LOOP:
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra DONE;
    mul.wide.u32 %rd9, %r6, 4;
    add.u64 %rd10, %rd1, %rd9;
    ld.global.u32 %r8, [%rd10];
    setp.eq.u32 %p2, %r8, 0;
    @%p2 bra NEXT;

    mov.u64 %rd11, %rd3;
    setp.eq.u64 %p3, %rd2, 0;
    @%p3 bra CREATED_READY;
    mul.wide.u32 %rd12, %r6, 8;
    add.u64 %rd13, %rd2, %rd12;
    ld.global.u32 %r10, [%rd13];
    ld.global.u32 %r11, [%rd13+4];
    cvt.u64.u32 %rd17, %r10;
    cvt.u64.u32 %rd18, %r11;
    shl.b64 %rd18, %rd18, 32;
    or.b64 %rd11, %rd17, %rd18;
CREATED_READY:
    setp.gt.u64 %p4, %rd11, %rd7;
    @%p4 bra CONFLICT;

    mov.u64 %rd14, %rd5;
    setp.eq.u64 %p5, %rd4, 0;
    @%p5 bra DELETED_READY;
    mul.wide.u32 %rd15, %r6, 8;
    add.u64 %rd16, %rd4, %rd15;
    ld.global.u32 %r10, [%rd16];
    ld.global.u32 %r11, [%rd16+4];
    cvt.u64.u32 %rd19, %r10;
    cvt.u64.u32 %rd20, %r11;
    shl.b64 %rd20, %rd20, 32;
    or.b64 %rd14, %rd19, %rd20;
DELETED_READY:
    setp.eq.u64 %p6, %rd14, %rd6;
    @%p6 bra NEXT;
    setp.gt.u64 %p7, %rd14, %rd7;
    @!%p7 bra NEXT;
CONFLICT:
    atom.global.exch.b32 %r9, [%rd8], 1;

NEXT:
    add.u32 %r6, %r6, %r7;
    bra LOOP;
DONE:
    ret;
}
"#;

        let primary = self.primary_arc();
        if !Arc::ptr_eq(&primary, &mask.mask.primary) {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let mask_bytes = usize::try_from(mask.row_count)
            .ok()
            .and_then(|rows| rows.checked_mul(std::mem::size_of::<u32>()))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if mask_bytes > mask.mask.capacity {
            return Err(CudaRuntimeProbeError::InvalidInputLength(mask_bytes));
        }
        let version_bytes = u64::from(mask.row_count)
            .checked_mul(std::mem::size_of::<u64>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let validate_source = |source: Option<(&CudaResidentDeviceMemory, u64)>|
         -> Result<u64, CudaRuntimeProbeError> {
            let Some((memory, offset)) = source else {
                return Ok(0);
            };
            if !Arc::ptr_eq(&primary, &memory.primary_arc()) {
                return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
            }
            let end = offset
                .checked_add(version_bytes)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if end > memory.metadata().allocated_bytes {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    usize::try_from(end).unwrap_or(usize::MAX),
                ));
            }
            memory
                .device_ptr()
                .checked_add(offset)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))
        };
        let created_ptr = validate_source(created_by)?;
        let deleted_ptr = validate_source(deleted_by)?;

        primary.set_current()?;
        let memcpy_htod = unsafe {
            primary
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let memcpy_dtoh = unsafe {
            primary
                .lib()
                .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let launch = unsafe {
            primary
                .lib()
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let verdict = primary.lease_device_buffer_owned(VERDICT_BYTES)?;
        let zero = 0u32;
        check_cuda(unsafe {
            memcpy_htod(
                verdict.ptr,
                (&zero as *const u32).cast::<c_void>(),
                VERDICT_BYTES,
            )
        })?;
        if mask.row_count != 0 {
            let mut ptx = PTX.to_vec();
            ptx.push(0);
            let function =
                primary.cached_function(c"gpu_db_predicate_mask_version_conflict", &ptx)?;
            let mut a0 = mask.mask.ptr;
            let mut a1 = mask.row_count;
            let mut a2 = created_ptr;
            let mut a3 = created_default;
            let mut a4 = deleted_ptr;
            let mut a5 = deleted_default;
            let mut a6 = deleted_live;
            let mut a7 = read_snapshot;
            let mut a8 = verdict.ptr;
            let mut args = [
                (&mut a0 as *mut u64).cast(),
                (&mut a1 as *mut u32).cast(),
                (&mut a2 as *mut u64).cast(),
                (&mut a3 as *mut u64).cast(),
                (&mut a4 as *mut u64).cast(),
                (&mut a5 as *mut u64).cast(),
                (&mut a6 as *mut u64).cast(),
                (&mut a7 as *mut u64).cast(),
                (&mut a8 as *mut u64).cast(),
            ];
            check_cuda(unsafe {
                launch(
                    function,
                    mask.row_count.div_ceil(256).clamp(1, 65_535),
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
        let mut host_verdict = 0u32;
        check_cuda(unsafe {
            memcpy_dtoh(
                (&mut host_verdict as *mut u32).cast::<c_void>(),
                verdict.ptr,
                VERDICT_BYTES,
            )
        })?;
        if host_verdict > 1 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                host_verdict as usize,
            ));
        }
        Ok(CudaVersionConflictVerdict {
            conflict: host_verdict == 1,
            readback_bytes: VERDICT_BYTES,
        })
    }
}
