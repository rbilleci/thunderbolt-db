//! Device-resident exact-key verdict for one dense INSERT batch.
//!
//! This execution primitive owns neither relation state nor publication. It accepts one immutable
//! resident payload and returns only a bounded NULL/duplicate terminal after two GPU passes.

#[cfg(test)]
use std::cell::Cell;
use std::collections::BTreeSet;
use std::os::raw::c_void;

use super::{
    check_cuda, CudaAllocationScope, CudaCompoundFoldColumn, CudaResidentDeviceMemory,
    CudaResidentReadSource, CudaRuntimeProbeError,
};

/// The sole host-visible terminal for [`CudaResidentDeviceMemory::insert_batch_key_verdict_from_payload`].
pub const INSERT_BATCH_KEY_VERDICT_READBACK_BYTES: usize = 16;

const TEXT_TAG: u32 = 0;
const BOOL_TAG: u32 = u32::MAX;
const VALIDITY_TAG: u32 = u32::MAX - 1;
const STATUS_MALFORMED_TEXT: u32 = 1;
const STATUS_MALFORMED_REPRESENTATIVE: u32 = 2;
const STATUS_PROBE_EXHAUSTED: u32 = 4;
const KEY_VERDICT_PTX: &[u8] = include_bytes!("insert_key_verdict.ptx");

#[cfg(test)]
thread_local! {
    static FAIL_AFTER_INITIALIZATION: Cell<bool> = const { Cell::new(false) };
}

/// Inject one error after the default-stream buffer initialization and before PTX lookup/launch.
/// This is test-only because it proves the armed drain owns initialization work on every error path.
#[cfg(test)]
pub(crate) fn fail_next_insert_key_verdict_after_initialization() {
    FAIL_AFTER_INITIALIZATION.with(|fail| fail.set(true));
}

#[cfg(test)]
fn take_fail_after_initialization() -> bool {
    FAIL_AFTER_INITIALIZATION.with(|fail| fail.replace(false))
}

/// A launch that reached the null stream must finish before its pooled inputs may drop. The normal
/// 16-byte terminal D2H provides that fence; this guard covers a later launch/D2H error.
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

/// One earliest SQL-NULL key component. `validity_ordinal` is within the validity suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaInsertBatchNull {
    pub row: u32,
    pub validity_ordinal: u32,
}

/// A bounded GPU-only dense-batch key verdict. No per-row hashes or coordinates cross D2H.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaInsertBatchKeyVerdict {
    pub first_null: Option<CudaInsertBatchNull>,
    pub first_duplicate_row: Option<u32>,
    pub readback_bytes: usize,
}

/// One 32-byte device descriptor: `[offset, tag, padding, blob_offset, blob_len]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct KeyDescriptor {
    offset: u64,
    tag: u32,
    reserved: u32,
    blob_offset: u64,
    blob_len: u64,
}

const _: () = assert!(std::mem::size_of::<KeyDescriptor>() == 32);

/// Exact pooled-device high-water excluding caller-owned payload.
///
/// `S = max(16, next_pow2(2 * rows))`; every independently-live allocation receives its actual
/// output-pool bucket: `P(8S) + P(4S) + P(32D) + P(16)`.
pub fn insert_batch_key_verdict_scratch_bytes(
    row_count: usize,
    descriptor_count: usize,
) -> Option<u64> {
    let slots = key_verdict_slot_count(row_count)?;
    key_verdict_bucket(slots.checked_mul(8)?)?
        .checked_add(key_verdict_bucket(slots.checked_mul(4)?)?)?
        .checked_add(key_verdict_bucket(descriptor_count.checked_mul(32)?)?)?
        .checked_add(key_verdict_bucket(INSERT_BATCH_KEY_VERDICT_READBACK_BYTES)?)
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

fn checked_span_end(
    allocated_bytes: u64,
    byte_offset: u64,
    byte_len: u64,
) -> Result<(), CudaRuntimeProbeError> {
    let end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| invalid_input(u64::MAX))?;
    if end > allocated_bytes {
        return Err(invalid_input(end));
    }
    Ok(())
}

/// Validate the public descriptor ABI and every payload extent before CUDA allocation/launch.
/// Repeated data descriptors are legal key members; only validity offsets must be unique.
pub(crate) fn validate_insert_batch_key_descriptors(
    allocated_bytes: u64,
    device_ptr: u64,
    columns: &[CudaCompoundFoldColumn],
    row_count: u32,
) -> Result<Vec<KeyDescriptor>, CudaRuntimeProbeError> {
    if columns.is_empty() || device_ptr == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(columns.len()));
    }
    device_ptr
        .checked_add(allocated_bytes)
        .ok_or_else(|| invalid_input(u64::MAX))?;
    let rows = u64::from(row_count);
    let bitmap_bytes = rows
        .checked_add(7)
        .map(|bits| bits / 8)
        .ok_or_else(|| invalid_input(u64::MAX))?;
    let text_offsets_bytes = rows
        .checked_add(1)
        .and_then(|count| count.checked_mul(8))
        .ok_or_else(|| invalid_input(u64::MAX))?;
    let mut descriptors = Vec::with_capacity(columns.len());
    let mut validity_offsets = BTreeSet::new();
    let mut saw_validity = false;
    let mut data_count = 0_usize;
    for column in columns {
        match *column {
            CudaCompoundFoldColumn::Fixed {
                byte_offset,
                width_words,
            } => {
                if saw_validity
                    || !matches!(width_words, 1 | 2 | 4)
                    || !byte_offset.is_multiple_of(4)
                {
                    return Err(invalid_input(byte_offset));
                }
                let byte_len = rows
                    .checked_mul(u64::from(width_words))
                    .and_then(|words| words.checked_mul(4))
                    .ok_or_else(|| invalid_input(u64::MAX))?;
                checked_span_end(allocated_bytes, byte_offset, byte_len)?;
                descriptors.push(KeyDescriptor {
                    offset: byte_offset,
                    tag: width_words,
                    reserved: 0,
                    blob_offset: 0,
                    blob_len: 0,
                });
                data_count += 1;
            }
            CudaCompoundFoldColumn::Text {
                offsets_byte_offset,
                bytes_byte_offset,
                bytes_len,
            } => {
                if saw_validity || !offsets_byte_offset.is_multiple_of(8) {
                    return Err(invalid_input(offsets_byte_offset));
                }
                checked_span_end(allocated_bytes, offsets_byte_offset, text_offsets_bytes)?;
                checked_span_end(allocated_bytes, bytes_byte_offset, bytes_len)?;
                descriptors.push(KeyDescriptor {
                    offset: offsets_byte_offset,
                    tag: TEXT_TAG,
                    reserved: 0,
                    blob_offset: bytes_byte_offset,
                    blob_len: bytes_len,
                });
                data_count += 1;
            }
            CudaCompoundFoldColumn::Bool { bitmap_byte_offset } => {
                if saw_validity {
                    return Err(invalid_input(bitmap_byte_offset));
                }
                checked_span_end(allocated_bytes, bitmap_byte_offset, bitmap_bytes)?;
                descriptors.push(KeyDescriptor {
                    offset: bitmap_byte_offset,
                    tag: BOOL_TAG,
                    reserved: 0,
                    blob_offset: 0,
                    blob_len: 0,
                });
                data_count += 1;
            }
            CudaCompoundFoldColumn::Validity { bitmap_byte_offset } => {
                saw_validity = true;
                if !validity_offsets.insert(bitmap_byte_offset) {
                    return Err(invalid_input(bitmap_byte_offset));
                }
                checked_span_end(allocated_bytes, bitmap_byte_offset, bitmap_bytes)?;
                descriptors.push(KeyDescriptor {
                    offset: bitmap_byte_offset,
                    tag: VALIDITY_TAG,
                    reserved: 0,
                    blob_offset: 0,
                    blob_len: 0,
                });
            }
        }
    }
    if data_count == 0 || validity_offsets.len() > data_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(columns.len()));
    }
    Ok(descriptors)
}

impl CudaResidentDeviceMemory {
    /// Validate a dense INSERT payload's exact compound keys on the GPU.
    ///
    /// The first launch constructs hash-directory representatives and records first rows. The
    /// second full rescan re-probes exact groups and records only `row > first_row`, making the
    /// returned second occurrence deterministic independent of winning `atom.cas` timing.
    pub fn insert_batch_key_verdict_from_payload(
        &self,
        columns: &[CudaCompoundFoldColumn],
        row_count: u32,
    ) -> Result<CudaInsertBatchKeyVerdict, CudaRuntimeProbeError> {
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

        let descriptors = validate_insert_batch_key_descriptors(
            self.metadata().allocated_bytes,
            self.device_ptr(),
            columns,
            row_count,
        )?;
        let descriptor_count = u32::try_from(descriptors.len())
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(descriptors.len()))?;
        let data_count = u32::try_from(
            columns
                .iter()
                .take_while(|column| !matches!(column, CudaCompoundFoldColumn::Validity { .. }))
                .count(),
        )
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(columns.len()))?;
        let slots = key_verdict_slot_count(row_count as usize)
            .ok_or_else(|| invalid_input(u64::from(row_count)))?;
        let slots_u64 = u64::try_from(slots).map_err(|_| invalid_input(u64::from(row_count)))?;
        let directory_bytes = slots
            .checked_mul(8)
            .ok_or_else(|| invalid_input(u64::from(row_count)))?;
        let first_rows_bytes = slots
            .checked_mul(4)
            .ok_or_else(|| invalid_input(u64::from(row_count)))?;
        let descriptor_bytes = descriptors
            .len()
            .checked_mul(std::mem::size_of::<KeyDescriptor>())
            .ok_or_else(|| invalid_input(u64::from(row_count)))?;
        let scratch = insert_batch_key_verdict_scratch_bytes(row_count as usize, descriptors.len())
            .ok_or_else(|| invalid_input(u64::from(row_count)))?;
        CudaAllocationScope::ensure_available(scratch)?;

        let primary = self.primary_arc();
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
        let first_rows = primary.lease_device_buffer_owned(first_rows_bytes)?;
        let descriptor_image = primary.lease_device_buffer_owned(descriptor_bytes)?;
        let verdict = primary.lease_device_buffer_owned(INSERT_BATCH_KEY_VERDICT_READBACK_BYTES)?;
        // This must be armed before the first default-stream memset/HtoD. Those driver calls can
        // enqueue work even though their setup returned synchronously; every later `?` therefore
        // retains the pooled buffers until this guard drains the null stream on unwind.
        let mut stream_drain = NullStreamDrain {
            synchronize: stream_synchronize,
            armed: true,
        };
        check_cuda(unsafe { memset(directory.ptr, 0, directory_bytes) })?;
        check_cuda(unsafe { memset(first_rows.ptr, u8::MAX, first_rows_bytes) })?;
        check_cuda(unsafe {
            memcpy_htod(
                descriptor_image.ptr,
                descriptors.as_ptr().cast::<c_void>(),
                descriptor_bytes,
            )
        })?;
        let mut initial = [0_u8; INSERT_BATCH_KEY_VERDICT_READBACK_BYTES];
        initial[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        initial[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        check_cuda(unsafe {
            memcpy_htod(
                verdict.ptr,
                initial.as_ptr().cast::<c_void>(),
                INSERT_BATCH_KEY_VERDICT_READBACK_BYTES,
            )
        })?;

        #[cfg(test)]
        if take_fail_after_initialization() {
            return Err(CudaRuntimeProbeError::KernelLaunchFailed(-1));
        }

        let mut ptx = KEY_VERDICT_PTX.to_vec();
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_insert_batch_key_verdict", &ptx)?;
        let launch_verdict = |mode: u32| -> Result<(), CudaRuntimeProbeError> {
            let mut base_arg = self.device_ptr();
            let mut descriptors_arg = descriptor_image.ptr;
            let mut data_count_arg = data_count;
            let mut descriptor_count_arg = descriptor_count;
            let mut rows_arg = row_count;
            let mut directory_arg = directory.ptr;
            let mut first_rows_arg = first_rows.ptr;
            let mut slots_arg = slots_u64;
            let mut verdict_arg = verdict.ptr;
            let mut mode_arg = mode;
            let mut args = [
                (&mut base_arg as *mut u64).cast::<c_void>(),
                (&mut descriptors_arg as *mut u64).cast::<c_void>(),
                (&mut data_count_arg as *mut u32).cast::<c_void>(),
                (&mut descriptor_count_arg as *mut u32).cast::<c_void>(),
                (&mut rows_arg as *mut u32).cast::<c_void>(),
                (&mut directory_arg as *mut u64).cast::<c_void>(),
                (&mut first_rows_arg as *mut u64).cast::<c_void>(),
                (&mut slots_arg as *mut u64).cast::<c_void>(),
                (&mut verdict_arg as *mut u64).cast::<c_void>(),
                (&mut mode_arg as *mut u32).cast::<c_void>(),
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
            })
        };

        launch_verdict(0)?;
        launch_verdict(1)?;
        let mut host_verdict = [0_u8; INSERT_BATCH_KEY_VERDICT_READBACK_BYTES];
        check_cuda(unsafe {
            memcpy_dtoh(
                host_verdict.as_mut_ptr().cast::<c_void>(),
                verdict.ptr,
                INSERT_BATCH_KEY_VERDICT_READBACK_BYTES,
            )
        })?;
        stream_drain.armed = false;

        let first_null = u64::from_le_bytes(host_verdict[..8].try_into().expect("fixed verdict"));
        let duplicate = u32::from_le_bytes(host_verdict[8..12].try_into().expect("fixed verdict"));
        let status = u32::from_le_bytes(host_verdict[12..].try_into().expect("fixed verdict"));
        let known_status =
            STATUS_MALFORMED_TEXT | STATUS_MALFORMED_REPRESENTATIVE | STATUS_PROBE_EXHAUSTED;
        let unknown_status = status & !known_status;
        if unknown_status != 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let known_error = status
            & (STATUS_MALFORMED_TEXT | STATUS_MALFORMED_REPRESENTATIVE | STATUS_PROBE_EXHAUSTED);
        if known_error != 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        Ok(CudaInsertBatchKeyVerdict {
            first_null: (first_null != u64::MAX).then_some(CudaInsertBatchNull {
                row: (first_null >> 32) as u32,
                validity_ordinal: first_null as u32,
            }),
            first_duplicate_row: (duplicate != u32::MAX).then_some(duplicate),
            readback_bytes: INSERT_BATCH_KEY_VERDICT_READBACK_BYTES,
        })
    }
}
