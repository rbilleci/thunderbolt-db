//! Shared pre-WAL INSERT proofs and the codec-5 physical-materialization capability.
//!
//! The former queued `PreparedDeviceInsertPlan`/`BoundDeviceInsertPlan` authority was deleted
//! with the direct commit-wave INSERT route. Production INSERTs now retain their typed batch in
//! the transaction overlay and compile the existing residency `DeviceInsertPlan` only inside the
//! codec-5 terminal.

use crate::{CatalogSnapshot, Engine, EngineError, Index};

pub(crate) mod batch_key_constraints;
pub(crate) mod constraint_arbitration;
pub(crate) mod host_retention;
mod pre_wal_capacity;
pub(crate) mod pre_wal_constraints;
mod pre_wal_effects;
pub(crate) use pre_wal_effects::PreparedInsertEffectPlan;
pub(crate) mod pre_wal_footprint;
mod reserved_pre_wal;
pub(crate) mod resident_constraint_generation;
pub(crate) mod resident_key_constraints;
pub(crate) mod row_local_constraints;

/// Capability for the one bounded indexed physical-materialization boundary.
///
/// This carries no WAL, status, row-id, allocator, or publication authority. It only proves the
/// codec-5 terminal has closed common semantic/currentness checks before residency allocates.
#[must_use]
pub(crate) struct IndexedPhysicalMaterializationPermit {
    _only_engine_insert_plan_may_issue: (),
}

pub(crate) fn issue_codec5_terminal_indexed_physical_materialization_permit(
) -> IndexedPhysicalMaterializationPermit {
    IndexedPhysicalMaterializationPermit {
        _only_engine_insert_plan_may_issue: (),
    }
}
