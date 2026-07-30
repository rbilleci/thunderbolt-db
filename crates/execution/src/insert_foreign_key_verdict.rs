//! GPU-only immediate foreign-key absence/presence verdict for one INSERT batch.
//!
//! This leaf owns no catalog or live mutation state. It accepts immutable child and pinned parent
//! device payloads, reduces all state on the device, and returns one bounded terminal for the
//! caller that owns SQL constraint arbitration.

#[cfg(test)]
use std::cell::Cell;
use std::os::raw::c_void;
use std::sync::Arc;

use super::{
    check_cuda,
    insert_key_verdict::{validate_insert_batch_key_descriptors, KeyDescriptor},
    insert_resident_key_verdict::{data_count, same_key_data_layout, sidecar_ptr},
    CudaAllocationScope, CudaCompoundFoldColumn, CudaInsertResidentKeySidecar,
    CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError,
};

/// The terminal is `[first_missing_row, first_history_row, status, reserved]`.
pub const INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES: usize = 16;

const STATUS_MALFORMED_TEXT: u32 = 1;
const STATUS_MALFORMED_REPRESENTATIVE: u32 = 2;
const STATUS_PROBE_EXHAUSTED: u32 = 4;
const STATUS_MALFORMED_VERSION: u32 = 8;
const FOREIGN_KEY_VERDICT_PTX: &[u8] = include_bytes!("insert_foreign_key_verdict.ptx");
const FOREIGN_KEY_THREADS_PER_BLOCK: u32 = 128;
const FOREIGN_KEY_MAX_BLOCKS: u32 = 65_535;
const FOREIGN_KEY_MAX_GRID_STRIDE: u32 = FOREIGN_KEY_THREADS_PER_BLOCK * FOREIGN_KEY_MAX_BLOCKS;

/// `r += grid_stride` remains in PTX's u32 domain. The final in-range iteration must therefore
/// be able to take one terminating increment without wrapping back below `row_count`.
pub(crate) const FOREIGN_KEY_MAX_SAFE_ROWS: u32 = u32::MAX - FOREIGN_KEY_MAX_GRID_STRIDE + 1;

#[cfg(test)]
thread_local! {
    static FAIL_AFTER_INITIALIZATION: Cell<bool> = const { Cell::new(false) };
    static FAIL_AFTER_FIRST_PARENT_LAUNCH: Cell<bool> = const { Cell::new(false) };
}

/// Inject one error after null-stream setup and before PTX lookup/launch.
#[cfg(test)]
pub(crate) fn fail_next_insert_foreign_key_verdict_after_initialization() {
    FAIL_AFTER_INITIALIZATION.with(|fail| fail.set(true));
}

/// Inject one error after the first parent scan, proving that every pooled lease is retained
/// until the null stream has drained real device work.
#[cfg(test)]
pub(crate) fn fail_next_insert_foreign_key_verdict_after_first_parent_launch() {
    FAIL_AFTER_FIRST_PARENT_LAUNCH.with(|fail| fail.set(true));
}

#[cfg(test)]
fn take_fail_after_initialization() -> bool {
    FAIL_AFTER_INITIALIZATION.with(|fail| fail.replace(false))
}

#[cfg(test)]
fn take_fail_after_first_parent_launch() -> bool {
    FAIL_AFTER_FIRST_PARENT_LAUNCH.with(|fail| fail.replace(false))
}

/// One immutable parent shard pinned by the caller's snapshot/commit gate.
pub struct CudaInsertForeignKeyParentShard<'a> {
    pub payload: &'a CudaResidentDeviceMemory,
    pub columns: &'a [CudaCompoundFoldColumn],
    pub row_count: u32,
    pub created_by: Option<CudaInsertResidentKeySidecar<'a>>,
    pub created_default: u64,
    pub deleted_by: Option<CudaInsertResidentKeySidecar<'a>>,
    pub deleted_default: u64,
    pub deleted_live: u64,
}

/// Optional same-batch parent provider for an immediate self-referential foreign key.
///
/// A successful exact match marks both the current constraint world and the original read world:
/// values supplied by this batch are visible to an immediate self FK without creating a history
/// witness. It reuses the child allocation and exact child row domain, but may select a distinct
/// referenced-column descriptor from that allocation.
pub struct CudaInsertForeignKeySelfProvider<'a> {
    pub columns: &'a [CudaCompoundFoldColumn],
}

/// One bounded GPU FK result.  A child key is satisfied iff it matched in both worlds; neither
/// match is missing, and exactly one match is a current/original history witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaInsertForeignKeyVerdict {
    pub first_missing_row: Option<u32>,
    pub first_history_row: Option<u32>,
    pub readback_bytes: usize,
}

/// Exact pooled-device high-water excluding caller-owned child, parent, and sidecar payloads.
///
/// `S = max(16, next_pow2(2 * child_rows))`.  The complete simultaneous allocation is
/// `P(8S)` for the child directory, `P(4S)` each for aligned current/original atomic match
/// buckets, the child
/// descriptor image, one reusable largest-parent descriptor image, an optional self-provider
/// descriptor image, and the 16-byte terminal.  Every `P` is the execution pool's actual bucket.
pub fn insert_foreign_key_verdict_scratch_bytes(
    child_rows: usize,
    child_descriptor_count: usize,
    max_parent_descriptor_count: usize,
    self_descriptor_count: Option<usize>,
) -> Option<u64> {
    if child_rows == 0 {
        return Some(0);
    }
    let slots = foreign_key_slot_count(child_rows)?;
    let self_descriptor_bucket = match self_descriptor_count {
        Some(count) => descriptor_bucket(count)?,
        None => 0,
    };
    foreign_key_bucket(slots.checked_mul(8)?)?
        .checked_add(foreign_key_bucket(
            slots.checked_mul(std::mem::size_of::<u32>())?,
        )?)?
        .checked_add(foreign_key_bucket(
            slots.checked_mul(std::mem::size_of::<u32>())?,
        )?)?
        .checked_add(descriptor_bucket(child_descriptor_count)?)?
        .checked_add(optional_descriptor_bucket(max_parent_descriptor_count)?)?
        .checked_add(self_descriptor_bucket)?
        .checked_add(foreign_key_bucket(
            INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
        )?)
}

fn foreign_key_slot_count(row_count: usize) -> Option<usize> {
    row_count
        .checked_mul(2)?
        .checked_next_power_of_two()
        .map(|slots| slots.max(16))
}

fn foreign_key_bucket(bytes: usize) -> Option<u64> {
    u64::try_from(bytes.max(256).checked_next_power_of_two()?).ok()
}

fn descriptor_bucket(descriptor_count: usize) -> Option<u64> {
    foreign_key_bucket(descriptor_count.checked_mul(std::mem::size_of::<KeyDescriptor>())?)
}

fn optional_descriptor_bucket(descriptor_count: usize) -> Option<u64> {
    if descriptor_count == 0 {
        Some(0)
    } else {
        descriptor_bucket(descriptor_count)
    }
}

fn invalid_input(value: u64) -> CudaRuntimeProbeError {
    CudaRuntimeProbeError::InvalidInputLength(usize::try_from(value).unwrap_or(usize::MAX))
}

pub(crate) fn foreign_key_grid_stride_is_safe(row_count: u32) -> bool {
    row_count <= FOREIGN_KEY_MAX_SAFE_ROWS
}

fn require_grid_stride_safe(row_count: u32) -> Result<(), CudaRuntimeProbeError> {
    foreign_key_grid_stride_is_safe(row_count)
        .then_some(())
        .ok_or_else(|| invalid_input(u64::from(row_count)))
}

fn require_single_column(
    child_columns: &[CudaCompoundFoldColumn],
    candidate_columns: &[CudaCompoundFoldColumn],
) -> Result<(), CudaRuntimeProbeError> {
    if data_count(child_columns)? != 1 || data_count(candidate_columns)? != 1 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            child_columns.len().max(candidate_columns.len()),
        ));
    }
    same_key_data_layout(child_columns, candidate_columns)
}

fn checked_descriptor_bytes(
    descriptors: &[KeyDescriptor],
    error_value: u64,
) -> Result<usize, CudaRuntimeProbeError> {
    descriptors
        .len()
        .checked_mul(std::mem::size_of::<KeyDescriptor>())
        .ok_or_else(|| invalid_input(error_value))
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

struct ValidatedParent<'a> {
    shard: &'a CudaInsertForeignKeyParentShard<'a>,
    descriptors: Vec<KeyDescriptor>,
    created_ptr: u64,
    deleted_ptr: u64,
    data_count: u32,
}

impl CudaResidentDeviceMemory {
    /// Evaluate one immediate, single-column `MATCH SIMPLE` child FK entirely on the GPU.
    ///
    /// The host merely validates captured device contracts and launches the fixed passes.  It
    /// never receives parent rows, child keys, directory entries, or per-shard results.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_foreign_key_verdict_against_shards(
        &self,
        child_columns: &[CudaCompoundFoldColumn],
        child_row_count: u32,
        parent_shards: &[CudaInsertForeignKeyParentShard<'_>],
        self_provider: Option<CudaInsertForeignKeySelfProvider<'_>>,
        published_constraint_boundary: u64,
        original_read_snapshot: u64,
    ) -> Result<CudaInsertForeignKeyVerdict, CudaRuntimeProbeError> {
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

        // Low zero is the EMPTY directory encoding, so the final u32 child ordinal is not
        // representable after the required `row + 1` encoding.
        if published_constraint_boundary < original_read_snapshot || self.device_ptr() == 0 {
            return Err(invalid_input(u64::from(child_row_count)));
        }
        require_grid_stride_safe(child_row_count)?;
        let child_descriptors = validate_insert_batch_key_descriptors(
            self.metadata().copied_bytes,
            self.device_ptr(),
            child_columns,
            child_row_count,
        )?;
        require_single_column(child_columns, child_columns)?;
        let child_data_count = data_count(child_columns)?;
        let child_descriptor_count = u32::try_from(child_descriptors.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(child_descriptors.len()))?;
        let child_descriptor_bytes =
            checked_descriptor_bytes(&child_descriptors, u64::from(child_row_count))?;

        let primary = self.primary_arc();
        let mut parents = Vec::with_capacity(parent_shards.len());
        let mut max_parent_descriptor_count = 0_usize;
        for shard in parent_shards {
            if shard.deleted_default != shard.deleted_live
                && shard.created_default > shard.deleted_default
                || shard.payload.device_ptr() == 0
                || !Arc::ptr_eq(&primary, &shard.payload.primary_arc())
            {
                return Err(invalid_input(u64::from(shard.row_count)));
            }
            require_grid_stride_safe(shard.row_count)?;
            let descriptors = validate_insert_batch_key_descriptors(
                shard.payload.metadata().copied_bytes,
                shard.payload.device_ptr(),
                shard.columns,
                shard.row_count,
            )?;
            require_single_column(child_columns, shard.columns)?;
            let created_ptr = sidecar_ptr(&primary, shard.created_by, shard.row_count)?;
            let deleted_ptr = sidecar_ptr(&primary, shard.deleted_by, shard.row_count)?;
            max_parent_descriptor_count = max_parent_descriptor_count.max(descriptors.len());
            parents.push(ValidatedParent {
                shard,
                data_count: data_count(shard.columns)?,
                descriptors,
                created_ptr,
                deleted_ptr,
            });
        }

        let self_provider = if let Some(provider) = self_provider {
            // The provider's data authority is exactly the child allocation and its launched
            // domain; only its referenced-column descriptors are independently supplied.
            require_grid_stride_safe(child_row_count)?;
            let descriptors = validate_insert_batch_key_descriptors(
                self.metadata().copied_bytes,
                self.device_ptr(),
                provider.columns,
                child_row_count,
            )?;
            require_single_column(child_columns, provider.columns)?;
            let data_count = data_count(provider.columns)?;
            Some((descriptors, data_count))
        } else {
            None
        };

        if child_row_count == 0 {
            return Ok(CudaInsertForeignKeyVerdict {
                first_missing_row: None,
                first_history_row: None,
                readback_bytes: 0,
            });
        }

        let slots = foreign_key_slot_count(child_row_count as usize)
            .ok_or_else(|| invalid_input(u64::from(child_row_count)))?;
        let slots_u64 =
            u64::try_from(slots).map_err(|_| invalid_input(u64::from(child_row_count)))?;
        let directory_bytes = slots
            .checked_mul(8)
            .ok_or_else(|| invalid_input(u64::from(child_row_count)))?;
        let match_bytes = slots
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| invalid_input(u64::from(child_row_count)))?;
        let max_parent_descriptor_bytes = max_parent_descriptor_count
            .checked_mul(std::mem::size_of::<KeyDescriptor>())
            .ok_or_else(|| invalid_input(u64::from(child_row_count)))?;
        let self_descriptor_bytes = self_provider
            .as_ref()
            .map(|(descriptors, _)| {
                checked_descriptor_bytes(descriptors, u64::from(child_row_count))
            })
            .transpose()?;
        let scratch = insert_foreign_key_verdict_scratch_bytes(
            child_row_count as usize,
            child_descriptors.len(),
            max_parent_descriptor_count,
            self_provider
                .as_ref()
                .map(|(descriptors, _)| descriptors.len()),
        )
        .ok_or_else(|| invalid_input(u64::from(child_row_count)))?;
        CudaAllocationScope::ensure_available(scratch)?;

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
        let current_matches = primary.lease_device_buffer_owned(match_bytes)?;
        let original_matches = primary.lease_device_buffer_owned(match_bytes)?;
        let child_image = primary.lease_device_buffer_owned(child_descriptor_bytes)?;
        let parent_image = (max_parent_descriptor_bytes != 0)
            .then(|| primary.lease_device_buffer_owned(max_parent_descriptor_bytes))
            .transpose()?;
        let self_image = self_descriptor_bytes
            .map(|bytes| primary.lease_device_buffer_owned(bytes))
            .transpose()?;
        let terminal =
            primary.lease_device_buffer_owned(INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES)?;
        let mut stream_drain = NullStreamDrain {
            synchronize: stream_synchronize,
            armed: true,
        };
        check_cuda(unsafe { memset(directory.ptr, 0, directory_bytes) })?;
        check_cuda(unsafe { memset(current_matches.ptr, 0, match_bytes) })?;
        check_cuda(unsafe { memset(original_matches.ptr, 0, match_bytes) })?;
        check_cuda(unsafe {
            memcpy_htod(
                child_image.ptr,
                child_descriptors.as_ptr().cast::<c_void>(),
                child_descriptor_bytes,
            )
        })?;
        let mut initial = [0_u8; INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES];
        initial[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        check_cuda(unsafe {
            memcpy_htod(
                terminal.ptr,
                initial.as_ptr().cast::<c_void>(),
                INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
            )
        })?;

        #[cfg(test)]
        if take_fail_after_initialization() {
            return Err(CudaRuntimeProbeError::KernelLaunchFailed(-1));
        }

        let mut ptx = FOREIGN_KEY_VERDICT_PTX.to_vec();
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_insert_foreign_key_verdict", &ptx)?;
        let launch_mode = |mode: u32,
                           parent_base: u64,
                           parent_descriptors: u64,
                           parent_data_count: u32,
                           parent_descriptor_count: u32,
                           parent_rows: u32,
                           created_ptr: u64,
                           created_default: u64,
                           deleted_ptr: u64,
                           deleted_default: u64,
                           deleted_live: u64|
         -> Result<(), CudaRuntimeProbeError> {
            let mut a0 = self.device_ptr();
            let mut a1 = child_image.ptr;
            let mut a2 = child_data_count;
            let mut a3 = child_descriptor_count;
            let mut a4 = child_row_count;
            let mut a5 = parent_base;
            let mut a6 = parent_descriptors;
            let mut a7 = parent_data_count;
            let mut a8 = parent_descriptor_count;
            let mut a9 = parent_rows;
            let mut a10 = directory.ptr;
            let mut a11 = current_matches.ptr;
            let mut a12 = original_matches.ptr;
            let mut a13 = slots_u64;
            let mut a14 = created_ptr;
            let mut a15 = created_default;
            let mut a16 = deleted_ptr;
            let mut a17 = deleted_default;
            let mut a18 = deleted_live;
            let mut a19 = published_constraint_boundary;
            let mut a20 = original_read_snapshot;
            let mut a21 = terminal.ptr;
            let mut a22 = mode;
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
                (&mut a20 as *mut u64).cast::<c_void>(),
                (&mut a21 as *mut u64).cast::<c_void>(),
                (&mut a22 as *mut u32).cast::<c_void>(),
            ];
            let work_rows = match mode {
                0 | 2 => child_row_count,
                _ => parent_rows,
            };
            check_cuda(unsafe {
                launch(
                    function,
                    work_rows
                        .div_ceil(FOREIGN_KEY_THREADS_PER_BLOCK)
                        .clamp(1, FOREIGN_KEY_MAX_BLOCKS),
                    1,
                    1,
                    FOREIGN_KEY_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            })
        };

        launch_mode(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)?;
        let mut launched_parent = false;
        for parent in &parents {
            let image = parent_image
                .as_ref()
                .expect("validated parent needs descriptor image");
            let descriptor_bytes =
                checked_descriptor_bytes(&parent.descriptors, u64::from(parent.shard.row_count))?;
            check_cuda(unsafe {
                memcpy_htod(
                    image.ptr,
                    parent.descriptors.as_ptr().cast::<c_void>(),
                    descriptor_bytes,
                )
            })?;
            let descriptor_count = u32::try_from(parent.descriptors.len())
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(parent.descriptors.len()))?;
            launch_mode(
                1,
                parent.shard.payload.device_ptr(),
                image.ptr,
                parent.data_count,
                descriptor_count,
                parent.shard.row_count,
                parent.created_ptr,
                parent.shard.created_default,
                parent.deleted_ptr,
                parent.shard.deleted_default,
                parent.shard.deleted_live,
            )?;
            if !launched_parent {
                launched_parent = true;
                #[cfg(test)]
                if take_fail_after_first_parent_launch() {
                    return Err(CudaRuntimeProbeError::KernelLaunchFailed(-1));
                }
            }
        }
        if let Some((descriptors, provider_data_count)) = &self_provider {
            let image = self_image
                .as_ref()
                .expect("self provider has descriptor image");
            check_cuda(unsafe {
                memcpy_htod(
                    image.ptr,
                    descriptors.as_ptr().cast::<c_void>(),
                    self_descriptor_bytes.expect("self provider bytes"),
                )
            })?;
            let descriptor_count = u32::try_from(descriptors.len())
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(descriptors.len()))?;
            launch_mode(
                3,
                self.device_ptr(),
                image.ptr,
                *provider_data_count,
                descriptor_count,
                child_row_count,
                0,
                0,
                0,
                0,
                0,
            )?;
        }
        launch_mode(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)?;
        let mut host_terminal = [0_u8; INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES];
        check_cuda(unsafe {
            memcpy_dtoh(
                host_terminal.as_mut_ptr().cast::<c_void>(),
                terminal.ptr,
                INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
            )
        })?;
        stream_drain.armed = false;
        let missing = u32::from_le_bytes(host_terminal[..4].try_into().expect("fixed terminal"));
        let history = u32::from_le_bytes(host_terminal[4..8].try_into().expect("fixed terminal"));
        let status = u32::from_le_bytes(host_terminal[8..12].try_into().expect("fixed terminal"));
        let known_status = STATUS_MALFORMED_TEXT
            | STATUS_MALFORMED_REPRESENTATIVE
            | STATUS_PROBE_EXHAUSTED
            | STATUS_MALFORMED_VERSION;
        if status & !known_status != 0 || status != 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        Ok(CudaInsertForeignKeyVerdict {
            first_missing_row: (missing != u32::MAX).then_some(missing),
            first_history_row: (history != u32::MAX).then_some(history),
            readback_bytes: INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
        })
    }
}
