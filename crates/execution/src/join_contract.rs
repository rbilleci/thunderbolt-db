#[cfg(test)]
use std::os::raw::c_void;

#[cfg(test)]
use super::{check_cuda, CudaRuntimeProbeError};
use super::{CudaResidentDeviceMemory, PooledDeviceBufferOwned};

/// One fixed-width equi-join key read directly from a resident payload.  The descriptor is host-side
/// launch metadata only: key bytes and NULL validity remain in the referenced device allocation.
/// `width` is 4, 8, or 16 bytes.  Two 4-byte descriptors may be supplied to form a composite key.
#[derive(Debug, Clone, Copy)]
pub struct CudaJoinPayloadKey<'a> {
    pub payload: &'a CudaResidentDeviceMemory,
    /// Fixed: value-column offset. Text (`width=255`): u64 offsets-column offset.
    pub byte_offset: u64,
    pub validity_bitmap_offset: Option<u64>,
    /// 4/8/16 for fixed values; 255 for a varlen text key.
    pub width: u8,
    /// Text bytes-section offset/length; `None/0` for fixed-width keys.
    pub text_bytes_byte_offset: Option<u64>,
    pub text_bytes_len: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct CudaJoinOrderKey<'a> {
    pub relation: u32,
    pub key: CudaJoinPayloadKey<'a>,
    pub descending: bool,
    pub nulls_first: bool,
    /// Width-16 keys are signed i128 by default; UUID uses bytewise canonical ordering.
    pub lexicographic_16: bool,
}

/// Device-resident row coordinates produced by a relational join. Coordinates are row-major:
/// `[tuple0.rel0, tuple0.rel1, ..., tuple1.rel0, ...]`; `u32::MAX` is an OUTER NULL pad.  The type is
/// deliberately opaque outside this crate so host code cannot turn an intermediate relation into a
/// semantic loop.  Only bounded scalar cardinalities cross D2H while the pipeline is executing.
pub struct CudaJoinCoordinatesU32 {
    pub(super) coordinates: Option<PooledDeviceBufferOwned>,
    pub(super) row_count: u32,
    pub(super) relation_count: u32,
    pub(super) allocated_bytes: u64,
}

impl CudaJoinCoordinatesU32 {
    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    pub fn relation_count(&self) -> u32 {
        self.relation_count
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    #[cfg(test)]
    pub(super) fn readback_for_test(&self) -> Result<Vec<Vec<u32>>, CudaRuntimeProbeError> {
        if self.row_count == 0 {
            return Ok(Vec::new());
        }
        type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
        let buffer = self
            .coordinates
            .as_ref()
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
        buffer.primary.set_current()?;
        let dtoh = unsafe {
            buffer
                .primary
                .lib()
                .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| buffer.primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        let len = self.row_count as usize * self.relation_count as usize;
        let mut flat = vec![0_u32; len];
        check_cuda(unsafe { dtoh(flat.as_mut_ptr().cast(), buffer.ptr, len * 4) })?;
        Ok(flat
            .chunks_exact(self.relation_count as usize)
            .map(<[u32]>::to_vec)
            .collect())
    }
}
