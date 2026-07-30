//! Exact device verdict for an incoming INSERT key against one resident shard.
//!
//! The directory is deliberately an incoming-row directory, not a resident index: it is short
//! lived, contains only candidate fingerprints, and every candidate is rechecked with typed
//! equality on the GPU.  The host receives only the earliest incoming row for each of the two
//! MVCC predicates.

#[cfg(test)]
use std::cell::Cell;
use std::os::raw::c_void;
use std::sync::Arc;

use super::{
    check_cuda, insert_key_verdict::validate_insert_batch_key_descriptors, CudaAllocationScope,
    CudaCompoundFoldColumn, CudaResidentDeviceMemory, CudaResidentReadSource,
    CudaRuntimeProbeError,
};

/// The terminal is `[visible_row, history_row, status, reserved]`.
pub const INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES: usize = 16;

const STATUS_MALFORMED_TEXT: u32 = 1;
const STATUS_PROBE_EXHAUSTED: u32 = 2;
const STATUS_MALFORMED_VERSION: u32 = 4;
const RESIDENT_KEY_VERDICT_PTX: &[u8] = include_bytes!("insert_resident_key_verdict.ptx");

#[cfg(test)]
thread_local! {
    static FAIL_AFTER_INITIALIZATION: Cell<bool> = const { Cell::new(false) };
    static FAIL_AFTER_FIRST_LAUNCH: Cell<bool> = const { Cell::new(false) };
}

/// Inject one error after the default-stream input initialization.  The test-only seam proves
/// that every queued buffer survives an error before module lookup/launch.
#[cfg(test)]
pub(crate) fn fail_next_insert_resident_key_verdict_after_initialization() {
    FAIL_AFTER_INITIALIZATION.with(|fail| fail.set(true));
}

/// Inject one error after the directory launch.  Unlike the initialization seam this proves the
/// null-stream drain retains the directory and descriptor buffers after real device work queued.
#[cfg(test)]
pub(crate) fn fail_next_insert_resident_key_verdict_after_first_launch() {
    FAIL_AFTER_FIRST_LAUNCH.with(|fail| fail.set(true));
}

#[cfg(test)]
fn take_fail_after_initialization() -> bool {
    FAIL_AFTER_INITIALIZATION.with(|fail| fail.replace(false))
}

#[cfg(test)]
fn take_fail_after_first_launch() -> bool {
    FAIL_AFTER_FIRST_LAUNCH.with(|fail| fail.replace(false))
}

/// One optional version sidecar. `byte_offset` names the first `u64` stamp for this shard.
/// An absent sidecar uses the matching default supplied on [`CudaInsertResidentKeyShard`].
#[derive(Clone, Copy)]
pub struct CudaInsertResidentKeySidecar<'a> {
    pub memory: &'a CudaResidentDeviceMemory,
    pub byte_offset: u64,
}

/// One pinned resident shard examined by the exact INSERT key proof.
///
/// The payload, descriptors, and sidecars are captured from one immutable resident generation;
/// callers must not synthesize the sidecars from a later map lookup.
pub struct CudaInsertResidentKeyShard<'a> {
    pub payload: &'a CudaResidentDeviceMemory,
    pub columns: &'a [CudaCompoundFoldColumn],
    pub row_count: u32,
    pub created_by: Option<CudaInsertResidentKeySidecar<'a>>,
    pub created_default: u64,
    pub deleted_by: Option<CudaInsertResidentKeySidecar<'a>>,
    pub deleted_default: u64,
    pub deleted_live: u64,
}

/// Bounded terminal from an exact incoming/resident key comparison.  The visible predicate is
/// evaluated at `published_constraint_boundary`; history is independently evaluated relative to
/// `original_read_snapshot`.  SQL arbitration remains the engine owner's responsibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaInsertResidentKeyVerdict {
    pub first_visible_conflict_row: Option<u32>,
    pub first_history_conflict_row: Option<u32>,
    pub readback_bytes: usize,
}

/// Exact transient allocation high-water excluding both caller-owned payloads and sidecars.
///
/// `S = max(16, next_pow2(2 * incoming_rows))`: one 64-bit open-addressed candidate entry per
/// non-NULL incoming row (duplicates intentionally remain separate), plus both descriptor images
/// and one 16-byte terminal.  Each independently-live allocation is charged at its pool bucket.
pub fn insert_resident_key_verdict_scratch_bytes(
    incoming_rows: usize,
    incoming_descriptor_count: usize,
    resident_descriptor_count: usize,
) -> Option<u64> {
    let slots = key_verdict_slot_count(incoming_rows)?;
    key_verdict_bucket(slots.checked_mul(8)?)?
        .checked_add(key_verdict_bucket(
            incoming_descriptor_count.checked_mul(32)?,
        )?)?
        .checked_add(key_verdict_bucket(
            resident_descriptor_count.checked_mul(32)?,
        )?)?
        .checked_add(key_verdict_bucket(
            INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES,
        )?)
}

fn key_verdict_slot_count(row_count: usize) -> Option<usize> {
    row_count
        .checked_mul(2)?
        .checked_next_power_of_two()
        .map(|n| n.max(16))
}

fn key_verdict_bucket(bytes: usize) -> Option<u64> {
    u64::try_from(bytes.max(256).checked_next_power_of_two()?).ok()
}

fn invalid_input(value: u64) -> CudaRuntimeProbeError {
    CudaRuntimeProbeError::InvalidInputLength(usize::try_from(value).unwrap_or(usize::MAX))
}

pub(crate) fn data_count(columns: &[CudaCompoundFoldColumn]) -> Result<u32, CudaRuntimeProbeError> {
    u32::try_from(
        columns
            .iter()
            .take_while(|column| !matches!(column, CudaCompoundFoldColumn::Validity { .. }))
            .count(),
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(columns.len()))
}

pub(crate) fn same_key_data_layout(
    incoming: &[CudaCompoundFoldColumn],
    resident: &[CudaCompoundFoldColumn],
) -> Result<(), CudaRuntimeProbeError> {
    let incoming_data = incoming
        .iter()
        .take_while(|column| !matches!(column, CudaCompoundFoldColumn::Validity { .. }));
    let resident_data = resident
        .iter()
        .take_while(|column| !matches!(column, CudaCompoundFoldColumn::Validity { .. }));
    let mut matched = 0_usize;
    for (left, right) in incoming_data.zip(resident_data) {
        let same = match (left, right) {
            (
                CudaCompoundFoldColumn::Fixed {
                    width_words: left, ..
                },
                CudaCompoundFoldColumn::Fixed {
                    width_words: right, ..
                },
            ) => left == right,
            (CudaCompoundFoldColumn::Text { .. }, CudaCompoundFoldColumn::Text { .. })
            | (CudaCompoundFoldColumn::Bool { .. }, CudaCompoundFoldColumn::Bool { .. }) => true,
            _ => false,
        };
        if !same {
            return Err(CudaRuntimeProbeError::InvalidInputLength(matched));
        }
        matched += 1;
    }
    if data_count(incoming)? != data_count(resident)? || matched == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(matched));
    }
    Ok(())
}

pub(crate) fn sidecar_ptr(
    primary: &Arc<super::GpuPrimaryContext>,
    sidecar: Option<CudaInsertResidentKeySidecar<'_>>,
    row_count: u32,
) -> Result<u64, CudaRuntimeProbeError> {
    let Some(sidecar) = sidecar else {
        return Ok(0);
    };
    if !Arc::ptr_eq(primary, &sidecar.memory.primary_arc())
        || !sidecar
            .byte_offset
            .is_multiple_of(std::mem::align_of::<u64>() as u64)
    {
        return Err(invalid_input(u64::MAX));
    }
    let bytes = u64::from(row_count)
        .checked_mul(std::mem::size_of::<u64>() as u64)
        .ok_or_else(|| invalid_input(u64::MAX))?;
    let end = sidecar
        .byte_offset
        .checked_add(bytes)
        .ok_or_else(|| invalid_input(u64::MAX))?;
    if sidecar.memory.device_ptr() == 0
        || end > sidecar.memory.metadata().copied_bytes
        || sidecar.memory.metadata().copied_bytes > sidecar.memory.metadata().allocated_bytes
    {
        return Err(invalid_input(end));
    }
    sidecar
        .memory
        .device_ptr()
        .checked_add(sidecar.byte_offset)
        .ok_or_else(|| invalid_input(u64::MAX))
}

/// A null-stream operation can be queued before a later driver/PTX error.  Keep every pooled
/// input alive through a synchronizing drain on that error path.
struct NullStreamDrain {
    synchronize: unsafe extern "C" fn(*mut c_void) -> i32,
    armed: bool,
}

impl Drop for NullStreamDrain {
    fn drop(&mut self) {
        if self.armed {
            let _ = check_cuda(unsafe { (self.synchronize)(std::ptr::null_mut()) });
        }
    }
}

impl CudaResidentDeviceMemory {
    /// Compare one dense incoming key payload with one resident shard entirely on the GPU.
    ///
    /// `published_constraint_boundary` is the current serialized constraint boundary.  A visible
    /// exact match at that boundary is a UNIQUE violation; a non-visible exact version whose
    /// create/release stamp is newer than `original_read_snapshot` is a separate retry witness.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_resident_key_verdict_against_shard(
        &self,
        incoming_columns: &[CudaCompoundFoldColumn],
        incoming_row_count: u32,
        shard: &CudaInsertResidentKeyShard<'_>,
        published_constraint_boundary: u64,
        original_read_snapshot: u64,
    ) -> Result<CudaInsertResidentKeyVerdict, CudaRuntimeProbeError> {
        type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
        type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
        type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
        type CuStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> i32;
        #[allow(clippy::type_complexity)]
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

        // Directory entries reserve zero as EMPTY and encode `incoming_row + 1` in their low
        // word.  The full u32 maximum therefore cannot be represented without becoming EMPTY.
        if incoming_row_count == u32::MAX || published_constraint_boundary < original_read_snapshot
        {
            return Err(invalid_input(u64::from(incoming_row_count)));
        }
        if shard.deleted_default != shard.deleted_live
            && shard.created_default > shard.deleted_default
        {
            return Err(invalid_input(shard.deleted_default));
        }
        if incoming_row_count == 0 || shard.row_count == 0 {
            return Ok(CudaInsertResidentKeyVerdict {
                first_visible_conflict_row: None,
                first_history_conflict_row: None,
                readback_bytes: 0,
            });
        }
        if self.device_ptr() == 0 || shard.payload.device_ptr() == 0 {
            return Err(invalid_input(0));
        }
        let incoming_descriptors = validate_insert_batch_key_descriptors(
            self.metadata().copied_bytes,
            self.device_ptr(),
            incoming_columns,
            incoming_row_count,
        )?;
        let resident_descriptors = validate_insert_batch_key_descriptors(
            shard.payload.metadata().copied_bytes,
            shard.payload.device_ptr(),
            shard.columns,
            shard.row_count,
        )?;
        same_key_data_layout(incoming_columns, shard.columns)?;
        let incoming_data_count = data_count(incoming_columns)?;
        let resident_data_count = data_count(shard.columns)?;
        let incoming_descriptor_count = u32::try_from(incoming_descriptors.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(incoming_descriptors.len()))?;
        let resident_descriptor_count = u32::try_from(resident_descriptors.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(resident_descriptors.len()))?;
        let slots = key_verdict_slot_count(incoming_row_count as usize)
            .ok_or_else(|| invalid_input(u64::from(incoming_row_count)))?;
        let slots_u64 =
            u64::try_from(slots).map_err(|_| invalid_input(u64::from(incoming_row_count)))?;
        let directory_bytes = slots
            .checked_mul(8)
            .ok_or_else(|| invalid_input(u64::from(incoming_row_count)))?;
        let incoming_descriptor_bytes = incoming_descriptors
            .len()
            .checked_mul(32)
            .ok_or_else(|| invalid_input(u64::from(incoming_row_count)))?;
        let resident_descriptor_bytes = resident_descriptors
            .len()
            .checked_mul(32)
            .ok_or_else(|| invalid_input(u64::from(shard.row_count)))?;
        let scratch = insert_resident_key_verdict_scratch_bytes(
            incoming_row_count as usize,
            incoming_descriptors.len(),
            resident_descriptors.len(),
        )
        .ok_or_else(|| invalid_input(u64::from(incoming_row_count)))?;
        CudaAllocationScope::ensure_available(scratch)?;

        let primary = self.primary_arc();
        if !Arc::ptr_eq(&primary, &shard.payload.primary_arc()) {
            return Err(invalid_input(u64::MAX));
        }
        let created_ptr = sidecar_ptr(&primary, shard.created_by, shard.row_count)?;
        let deleted_ptr = sidecar_ptr(&primary, shard.deleted_by, shard.row_count)?;
        primary.set_current()?;
        let memset = unsafe {
            primary
                .lib()
                .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
                .or_else(|_| primary.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
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
        let stream_synchronize = unsafe {
            *primary
                .lib()
                .get::<CuStreamSynchronize>(b"cuStreamSynchronize\0")
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };

        let directory = primary.lease_device_buffer_owned(directory_bytes)?;
        let incoming_image = primary.lease_device_buffer_owned(incoming_descriptor_bytes)?;
        let resident_image = primary.lease_device_buffer_owned(resident_descriptor_bytes)?;
        let terminal =
            primary.lease_device_buffer_owned(INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES)?;
        let mut stream_drain = NullStreamDrain {
            synchronize: stream_synchronize,
            armed: true,
        };
        check_cuda(unsafe { memset(directory.ptr, 0, directory_bytes) })?;
        check_cuda(unsafe {
            memcpy_htod(
                incoming_image.ptr,
                incoming_descriptors.as_ptr().cast::<c_void>(),
                incoming_descriptor_bytes,
            )
        })?;
        check_cuda(unsafe {
            memcpy_htod(
                resident_image.ptr,
                resident_descriptors.as_ptr().cast::<c_void>(),
                resident_descriptor_bytes,
            )
        })?;
        let initial = [u8::MAX; INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES];
        check_cuda(unsafe {
            memcpy_htod(
                terminal.ptr,
                initial.as_ptr().cast::<c_void>(),
                INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES,
            )
        })?;
        // The status word begins clear; the two row minima use MAX sentinels.
        let zero = 0_u32;
        check_cuda(unsafe {
            memcpy_htod(
                terminal.ptr + 8,
                (&zero as *const u32).cast::<c_void>(),
                std::mem::size_of::<u32>(),
            )
        })?;

        #[cfg(test)]
        if take_fail_after_initialization() {
            return Err(CudaRuntimeProbeError::KernelLaunchFailed(-1));
        }

        let mut ptx = RESIDENT_KEY_VERDICT_PTX.to_vec();
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_insert_resident_key_verdict", &ptx)?;
        let launch_mode = |mode: u32, work_rows: u32| -> Result<(), CudaRuntimeProbeError> {
            let mut a0 = self.device_ptr();
            let mut a1 = incoming_image.ptr;
            let mut a2 = incoming_data_count;
            let mut a3 = incoming_descriptor_count;
            let mut a4 = incoming_row_count;
            let mut a5 = shard.payload.device_ptr();
            let mut a6 = resident_image.ptr;
            let mut a7 = resident_data_count;
            let mut a8 = resident_descriptor_count;
            let mut a9 = shard.row_count;
            let mut a10 = directory.ptr;
            let mut a11 = slots_u64;
            let mut a12 = created_ptr;
            let mut a13 = shard.created_default;
            let mut a14 = deleted_ptr;
            let mut a15 = shard.deleted_default;
            let mut a16 = shard.deleted_live;
            let mut a17 = published_constraint_boundary;
            let mut a18 = original_read_snapshot;
            let mut a19 = terminal.ptr;
            let mut a20 = mode;
            let mut args = [
                (&mut a0 as *mut u64).cast::<c_void>(),
                (&mut a1 as *mut u64).cast::<c_void>(),
                (&mut a2 as *mut u32).cast::<c_void>(),
                (&mut a3 as *mut u32).cast::<c_void>(),
                (&mut a4 as *mut u32).cast::<c_void>(),
                (&mut a5 as *mut u64).cast::<c_void>(),
                (&mut a6 as *mut u64).cast::<c_void>(),
                (&mut a7 as *mut u32).cast::<c_void>(),
                (&mut a8 as *mut u32).cast::<c_void>(),
                (&mut a9 as *mut u32).cast::<c_void>(),
                (&mut a10 as *mut u64).cast::<c_void>(),
                (&mut a11 as *mut u64).cast::<c_void>(),
                (&mut a12 as *mut u64).cast::<c_void>(),
                (&mut a13 as *mut u64).cast::<c_void>(),
                (&mut a14 as *mut u64).cast::<c_void>(),
                (&mut a15 as *mut u64).cast::<c_void>(),
                (&mut a16 as *mut u64).cast::<c_void>(),
                (&mut a17 as *mut u64).cast::<c_void>(),
                (&mut a18 as *mut u64).cast::<c_void>(),
                (&mut a19 as *mut u64).cast::<c_void>(),
                (&mut a20 as *mut u32).cast::<c_void>(),
            ];
            check_cuda(unsafe {
                launch(
                    function,
                    work_rows.div_ceil(128).clamp(1, 65_535),
                    1,
                    1,
                    128,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            })
        };
        launch_mode(0, incoming_row_count)?;
        #[cfg(test)]
        if take_fail_after_first_launch() {
            return Err(CudaRuntimeProbeError::KernelLaunchFailed(-1));
        }
        launch_mode(1, shard.row_count)?;
        let mut host_terminal = [0_u8; INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES];
        check_cuda(unsafe {
            memcpy_dtoh(
                host_terminal.as_mut_ptr().cast::<c_void>(),
                terminal.ptr,
                INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES,
            )
        })?;
        stream_drain.armed = false;
        let visible = u32::from_le_bytes(host_terminal[..4].try_into().expect("fixed terminal"));
        let history = u32::from_le_bytes(host_terminal[4..8].try_into().expect("fixed terminal"));
        let status = u32::from_le_bytes(host_terminal[8..12].try_into().expect("fixed terminal"));
        if status & !(STATUS_MALFORMED_TEXT | STATUS_PROBE_EXHAUSTED | STATUS_MALFORMED_VERSION)
            != 0
            || status != 0
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        Ok(CudaInsertResidentKeyVerdict {
            first_visible_conflict_row: (visible != u32::MAX).then_some(visible),
            first_history_conflict_row: (history != u32::MAX).then_some(history),
            readback_bytes: INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES,
        })
    }
}
