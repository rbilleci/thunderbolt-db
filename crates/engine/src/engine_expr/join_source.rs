//! Device-source ownership contracts shared by resident, catalog, and streaming joins.

use std::sync::Arc;

use super::ResidentVisibility;
use crate::relational_model::RelationalResidencyEntry;
use gpu_db_execution::{
    CudaExternalAllocationReservation, CudaPredicateMaskI32, CudaResidentDeviceMemory,
};

/// Sentinel carried in a join's per-relation index vectors meaning "no row -> emit NULL for this
/// relation's columns" -- a LEFT OUTER join's NULL pad for an unmatched left row (M3 -- doc 21). A real
/// absolute row index can never be `u32::MAX` (residency row counts are far smaller), so it is unambiguous.
pub(super) const JOIN_NULL_ROW: u32 = u32::MAX;

/// The device memory backing a join relation: a RESIDENT user table's published `Arc` (shared), or a
/// SYNTHESIZED catalog relation's freshly-uploaded TRANSIENT payload (owned for the query). `.mem()`
/// yields the `&CudaResidentDeviceMemory` the GPU pre-filter / key-projection / hash-join kernels run on.
pub(crate) enum JoinDeviceMemory {
    Resident(Arc<CudaResidentDeviceMemory>),
    Transient(CudaResidentDeviceMemory),
}

pub(super) struct JoinNullPadMask {
    pub(super) mask: Option<CudaPredicateMaskI32>,
    pub(super) _source: CudaResidentDeviceMemory,
    pub(super) _allocation: CudaExternalAllocationReservation,
}

pub(crate) type JoinExecSide = (
    RelationalResidencyEntry,
    JoinDeviceMemory,
    usize,
    Option<ResidentVisibility>,
);

impl JoinDeviceMemory {
    pub(crate) fn mem(&self) -> &CudaResidentDeviceMemory {
        match self {
            JoinDeviceMemory::Resident(memory) => memory,
            JoinDeviceMemory::Transient(memory) => memory,
        }
    }
}
