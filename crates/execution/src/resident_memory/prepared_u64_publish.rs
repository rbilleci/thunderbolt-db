//! Move-only pre-WAL publication of one fixed-width resident value.
//!
//! A visibility header is a publication boundary, not an append payload. This token therefore
//! prepares its exact destination, allocation lifetime, primary context, inline `u64` image, and
//! synchronous driver entry point before durability. Its only consuming operation is the final
//! host-to-device write; it cannot rebuild a destination or resolve CUDA state after WAL.

#[cfg(test)]
use std::cell::Cell;
use std::os::raw::c_void;
use std::sync::Arc;

use crate::cuda_context::{check_cuda, GpuPrimaryContext};
use crate::CudaRuntimeProbeError;

use super::{CudaResidentDeviceAllocation, CudaResidentDeviceMemory};

type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;

const U64_BYTES: usize = std::mem::size_of::<u64>();

/// A one-shot, pre-WAL, synchronous `u64` publication into resident memory.
///
/// The token owns no heap-backed payload: `value` is the exact eight-byte little-endian image
/// inline. Its strong allocation and primary-context guards keep the exact destination valid even
/// when the outer resident-memory owner is retired between preparation and publication.
#[must_use]
pub struct PreparedU64HtoDPublication {
    primary: Arc<GpuPrimaryContext>,
    _allocation: Arc<CudaResidentDeviceAllocation>,
    destination: u64,
    value: [u8; U64_BYTES],
    cu_memcpy_htod: CuMemcpyHtoD,
}

// SAFETY: publication is consuming, the exact allocation and its primary context are strongly
// pinned, and the driver context is rebound on the publishing thread. The token has no mutable
// shared state and no borrowed host backing.
unsafe impl Send for PreparedU64HtoDPublication {}

impl PreparedU64HtoDPublication {
    /// Publish the prepared eight bytes synchronously.
    ///
    /// This is deliberately the entire post-WAL surface: it makes the retained primary context
    /// current and calls the function pointer resolved at preparation. There is no allocation,
    /// module/cache operation, symbol lookup, destination reconstruction, or fallback path here.
    pub fn publish(self) -> Result<(), CudaRuntimeProbeError> {
        self.primary.set_current()?;
        #[cfg(test)]
        if take_fail_next_prepared_u64_htod_publication() {
            // Exercise the consuming failure edge without making a test depend on a driver fault.
            // This models a synchronous HtoD failure, before any asynchronous work can exist.
            return check_cuda(-1);
        }
        check_cuda(unsafe {
            (self.cu_memcpy_htod)(
                self.destination,
                self.value.as_ptr().cast::<c_void>(),
                self.value.len(),
            )
        })
    }
}

impl CudaResidentDeviceMemory {
    /// Prepare the exact synchronous eight-byte publication that may occur after WAL.
    ///
    /// `byte_offset` names a naturally aligned `u64` wholly within this resident allocation.
    /// Preparation pins the allocation's own primary context (rather than trusting an outer
    /// wrapper witness), binds it on this thread, and resolves `cuMemcpyHtoD` once. Consequently
    /// [`PreparedU64HtoDPublication::publish`] has no build, cache, or symbol-resolution work.
    pub fn prepare_u64_htod_publication(
        &self,
        byte_offset: u64,
        value: u64,
    ) -> Result<PreparedU64HtoDPublication, CudaRuntimeProbeError> {
        let allocation = Arc::clone(&self.allocation);
        let destination = checked_u64_publication_destination(self, &allocation, byte_offset)?;
        // The allocation is the source of truth for the context that owns `destination`. A test
        // can deliberately construct an outer wrapper with a different witness; production code
        // cannot, but this keeps the capability exact rather than relying on that invariant.
        let primary = Arc::clone(&allocation.primary);
        primary.set_current()?;
        let cu_memcpy_htod = unsafe {
            *primary
                .lib()
                .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        Ok(PreparedU64HtoDPublication {
            primary,
            _allocation: allocation,
            destination,
            value: value.to_le_bytes(),
            cu_memcpy_htod,
        })
    }
}

fn checked_u64_publication_destination(
    memory: &CudaResidentDeviceMemory,
    allocation: &Arc<CudaResidentDeviceAllocation>,
    byte_offset: u64,
) -> Result<u64, CudaRuntimeProbeError> {
    validate_u64_publication_span(memory.metadata.allocated_bytes, byte_offset)?;
    if allocation.device_ptr == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(byte_offset).unwrap_or(usize::MAX),
        ));
    }
    let destination = allocation
        .device_ptr
        .checked_add(byte_offset)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    // CUDA receives a pointer plus a length. Checking the final addressed byte makes the device
    // address arithmetic total even for malformed/raw-test construction at the top of `u64`.
    destination
        .checked_add((U64_BYTES - 1) as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    Ok(destination)
}

fn validate_u64_publication_span(
    allocated_bytes: u64,
    byte_offset: u64,
) -> Result<(), CudaRuntimeProbeError> {
    if !byte_offset.is_multiple_of(U64_BYTES as u64) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(byte_offset).unwrap_or(usize::MAX),
        ));
    }
    let end = byte_offset
        .checked_add(U64_BYTES as u64)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if end > allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(end).unwrap_or(usize::MAX),
        ));
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    static FAIL_NEXT_PREPARED_U64_HTOD_PUBLICATION: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn fail_next_prepared_u64_htod_publication() {
    FAIL_NEXT_PREPARED_U64_HTOD_PUBLICATION.with(|value| value.set(true));
}

#[cfg(test)]
fn take_fail_next_prepared_u64_htod_publication() -> bool {
    FAIL_NEXT_PREPARED_U64_HTOD_PUBLICATION.with(|value| value.replace(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u64_publication_destination_validation_is_total() {
        assert_eq!(validate_u64_publication_span(16, 0), Ok(()));
        assert_eq!(validate_u64_publication_span(16, 8), Ok(()));
        assert!(validate_u64_publication_span(16, 16).is_err());
        assert!(validate_u64_publication_span(16, 4).is_err());
        assert!(validate_u64_publication_span(u64::MAX, u64::MAX - 7).is_err());
    }
}
