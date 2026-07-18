use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaRuntimeSnapshot {
    pub driver_available: bool,
    pub driver_version: Option<i32>,
    pub device_count: u16,
    pub devices: Vec<CudaDeviceSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaDeviceSnapshot {
    pub id: u16,
    pub name: String,
    pub total_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaDeviceMemoryProof {
    pub gpu_id: u16,
    pub device_name: String,
    pub allocated_bytes: u64,
    pub copied_bytes: u64,
    pub retained: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaRuntimeProbeError {
    DriverLibraryUnavailable,
    DriverInitFailed(i32),
    DeviceCountFailed(i32),
    InvalidDeviceCount(i32),
    InvalidInputLength(usize),
    AllocationBudgetExceeded {
        requested: u64,
        live: u64,
        limit: u64,
    },
    KernelLaunchFailed(i32),
    /// A supposedly unique point-read route found the same needle in more than one resident shard. The
    /// compact production result uses status 3 to decline/re-resolve; compatibility APIs surface it as an
    /// error rather than silently translating the duplicate into a not-found row.
    DuplicatePointReadMatch(usize),
    /// A comparison code outside the range the called primitive supports (the fused
    /// scalar/buffer compact kernels handle 0=eq..4=ge; `5=ne` is mask-path only).
    UnsupportedComparison(u32),
    /// On-device int4 arithmetic (`+`/`-`/`*`) overflowed the int32 range on at least one row. The
    /// device VM evaluates each op in 64-bit, range-checks it, and sets an overflow flag; the host
    /// surfaces this so the query errors exactly as PostgreSQL's `integer out of range` (Charter
    /// rule 2 PG-fidelity), never silently wrapping. GPU-native: detected on the device, no CPU
    /// fallback.
    IntegerOutOfRange,
    /// On-device int8 arithmetic (`+`/`-`/`*`) overflowed the int64 range — same checked-on-device
    /// model as [`Self::IntegerOutOfRange`], surfaced as PostgreSQL's `bigint out of range` (the type
    /// matrix, doc 19; int8 differs from int4 only in the error type).
    BigintOutOfRange,
    /// On-device int128 (NUMERIC) arithmetic overflowed the i128 mantissa range — same checked model,
    /// surfaced as PostgreSQL's `numeric field overflow` (the type matrix, doc 19).
    NumericFieldOverflow,
}

impl fmt::Display for CudaRuntimeProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DriverLibraryUnavailable => write!(f, "CUDA driver library is unavailable"),
            Self::DriverInitFailed(code) => write!(f, "CUDA driver initialization failed: {code}"),
            Self::DeviceCountFailed(code) => {
                write!(f, "CUDA device count query failed: {code}")
            }
            Self::InvalidDeviceCount(count) => write!(f, "invalid CUDA device count: {count}"),
            Self::InvalidInputLength(len) => write!(f, "invalid CUDA input length: {len}"),
            Self::AllocationBudgetExceeded {
                requested,
                live,
                limit,
            } => write!(
                f,
                "CUDA allocation budget exceeded: {live} live + {requested} requested > {limit} bytes"
            ),
            Self::KernelLaunchFailed(code) => write!(f, "CUDA kernel launch failed: {code}"),
            Self::DuplicatePointReadMatch(needle_index) => write!(
                f,
                "duplicate CUDA point-read match for needle index {needle_index}"
            ),
            Self::UnsupportedComparison(code) => {
                write!(f, "unsupported comparison code for this primitive: {code}")
            }
            // Surfaced verbatim as the engine's error message, so it reads as PostgreSQL's
            // `integer out of range` / `bigint out of range` (SQLSTATE 22003) to a client.
            Self::IntegerOutOfRange => write!(f, "integer out of range"),
            Self::BigintOutOfRange => write!(f, "bigint out of range"),
            Self::NumericFieldOverflow => write!(f, "numeric field overflow"),
        }
    }
}

impl std::error::Error for CudaRuntimeProbeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_errors_preserve_client_facing_text() {
        assert_eq!(
            CudaRuntimeProbeError::DriverLibraryUnavailable.to_string(),
            "CUDA driver library is unavailable"
        );
        assert_eq!(
            CudaRuntimeProbeError::AllocationBudgetExceeded {
                requested: 4,
                live: 8,
                limit: 10,
            }
            .to_string(),
            "CUDA allocation budget exceeded: 8 live + 4 requested > 10 bytes"
        );
        assert_eq!(
            CudaRuntimeProbeError::IntegerOutOfRange.to_string(),
            "integer out of range"
        );
        assert_eq!(
            CudaRuntimeProbeError::DuplicatePointReadMatch(7).to_string(),
            "duplicate CUDA point-read match for needle index 7"
        );
        assert_eq!(
            CudaRuntimeProbeError::BigintOutOfRange.to_string(),
            "bigint out of range"
        );
        assert_eq!(
            CudaRuntimeProbeError::NumericFieldOverflow.to_string(),
            "numeric field overflow"
        );
    }
}
