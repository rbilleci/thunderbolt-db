//! Typed device-plan compilation for the accepted INSERT residency route.
//!
//! This module owns no CUDA allocation, device write, descriptor publication, side-map mutation,
//! or WAL. It consumes a sealed source into an opaque plan, then the plan re-enters mutation's one
//! publisher at apply time. A post-WAL apply failure is fatal rather than permission to fall back.

use super::append_source::ResidentAppendSource;
use super::*;
use crate::engine_insert_plan::host_retention::{HostRetentionGeometry, HostRetentionReport};
use crate::engine_insert_plan::IndexedPhysicalMaterializationPermit;
use crate::typed_insert_batch::{
    PreparedResidentAppendSource, PreparedResidentFixedBoolUpload, PreparedResidentFixedChunk,
    PreparedResidentFixedChunkOwners, TypedInsertBatch,
};
use std::sync::MutexGuard;

mod indexed_fused;
mod retained_host;
pub(super) use indexed_fused::{
    indexed_in_place_fused_footprint_forecast, indexed_in_place_fused_host_retention_forecast,
    indexed_in_place_fused_materialization_scratch_forecast, source_is_all_i32_fixed,
    PreparedIndexedInPlaceFusedApply,
};
use retained_host::{
    append_pending_bool_layout_box, append_pending_int4_stats_box,
    indexed_fixed_rollover_materialization_scratch_prediction,
    indexed_fixed_rollover_plan_owned_host_retention_prediction,
    indexed_in_place_plan_owned_host_retention_prediction, PreparedOpenShardIdentity,
};

/// Allocation-free host owner geometry for an indexed in-place append before its sidecar/chunk
/// owners exist.  The forecast boundary is the only sibling permitted to request it.
pub(super) fn indexed_in_place_append_host_retention_forecast(
    source: &PreparedResidentAppendSource,
    row_ids: &DeviceInsertRowIds,
    open: &RelationalResidentShard,
    table: &RelationalTable,
) -> Result<HostRetentionGeometry, EngineError> {
    indexed_in_place_plan_owned_host_retention_prediction(source, row_ids, open, table)
}

/// Equivalent scalar host geometry for the fixed-rollover append branch.
pub(super) fn indexed_fixed_rollover_append_host_retention_forecast(
    source: &PreparedResidentAppendSource,
    row_ids: &DeviceInsertRowIds,
    open: &RelationalResidentShard,
    table: &RelationalTable,
) -> Result<HostRetentionGeometry, EngineError> {
    indexed_fixed_rollover_plan_owned_host_retention_prediction(source, row_ids, open, table)
}

pub(super) fn indexed_fixed_rollover_append_host_scratch_forecast(
    source: &PreparedResidentAppendSource,
    row_ids: &DeviceInsertRowIds,
    table: &RelationalTable,
) -> Result<HostRetentionGeometry, EngineError> {
    indexed_fixed_rollover_materialization_scratch_prediction(source, row_ids, table)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceInsertPlanPrepareError {
    /// The fixed route is not applicable to this source or resident descriptor. Callers may
    /// retain the established pre-WAL legacy preparation in this case.
    UnsupportedShape,
    /// A source proved the exact CREATE bootstrap sentinel, but its first rollover cannot fit the
    /// current residency budget. Legacy must not claim WAL/row identities, then find the same
    /// allocation miss before canonical WAL/status buffering, device apply, or group durability.
    RetryableBoundBootstrapResource,
    /// The exact CREATE bootstrap binding changed while its rollover reservation was being
    /// sealed. This is retryable rather than a license to replace the bound path with legacy.
    RetryableBoundBootstrapState,
}

#[allow(dead_code)] // reservation modes compile before live indexed selection is opened
#[derive(Clone, Copy)]
enum ResidentOpenShardAppendPreparationMode {
    LiveUnindexed,
    IndexedInPlaceReservation { index_scratch_bytes: u64 },
    IndexedFixedRolloverReservation { max_index_scratch_bytes: u64 },
}

/// A post-WAL device-apply failure is terminal before physical group durability; the active wave
/// wedges with durable count unchanged rather than attempting legacy re-application or publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceInsertPlanApplyError {
    PlanDrift,
    ShapeDrift,
    PublisherFailure,
}

/// Exact durable row identities sealed into a typed append before WAL. Construction is limited to
/// owned IDs; the synthetic form is test-only so a production cutover cannot accidentally lose a
/// bound entity-identity sidecar.
pub(crate) struct DeviceInsertRowIds {
    kind: DeviceInsertRowIdsKind,
}

enum DeviceInsertRowIdsKind {
    Exact(Box<[u64]>),
    SyntheticNoIdentity,
    ConsumedExact,
}

impl DeviceInsertRowIds {
    pub(crate) fn exact(ids: Box<[u64]>) -> Self {
        Self {
            kind: DeviceInsertRowIdsKind::Exact(ids),
        }
    }

    #[cfg(test)]
    pub(crate) fn synthetic_no_identity() -> Self {
        Self {
            kind: DeviceInsertRowIdsKind::SyntheticNoIdentity,
        }
    }

    pub(super) fn is_exact(&self) -> bool {
        matches!(
            &self.kind,
            DeviceInsertRowIdsKind::Exact(_) | DeviceInsertRowIdsKind::ConsumedExact
        )
    }

    pub(super) fn exact_len_matches(&self, rows: usize) -> bool {
        match &self.kind {
            DeviceInsertRowIdsKind::Exact(ids) => ids.len() == rows,
            DeviceInsertRowIdsKind::SyntheticNoIdentity => true,
            DeviceInsertRowIdsKind::ConsumedExact => false,
        }
    }

    fn take_exact(&mut self) -> Option<Box<[u64]>> {
        match std::mem::replace(&mut self.kind, DeviceInsertRowIdsKind::SyntheticNoIdentity) {
            DeviceInsertRowIdsKind::Exact(ids) => {
                self.kind = DeviceInsertRowIdsKind::ConsumedExact;
                Some(ids)
            }
            DeviceInsertRowIdsKind::SyntheticNoIdentity | DeviceInsertRowIdsKind::ConsumedExact => {
                None
            }
        }
    }

    /// Borrow the one exact, still-unconsumed identity owner while a pre-WAL fused staging image
    /// copies it.  The append plan retains ownership until the later publication-capable slice;
    /// this method never permits replacing or consuming those identities.
    fn exact_slice(&self) -> Option<&[u64]> {
        match &self.kind {
            DeviceInsertRowIdsKind::Exact(ids) => Some(ids),
            DeviceInsertRowIdsKind::SyntheticNoIdentity | DeviceInsertRowIdsKind::ConsumedExact => {
                None
            }
        }
    }

    /// Row identities are fixed before WAL. Dense plans upload these into their private sidecar
    /// allocation during preflight; apply still owns the IDs to prevent a caller from replacing
    /// identity authority between WAL and descriptor publication.
    fn pre_wal_payload(&self) -> Option<Vec<u8>> {
        match &self.kind {
            DeviceInsertRowIdsKind::Exact(ids) => {
                Some(ids.iter().flat_map(|id| id.to_le_bytes()).collect())
            }
            DeviceInsertRowIdsKind::SyntheticNoIdentity => None,
            DeviceInsertRowIdsKind::ConsumedExact => None,
        }
    }

    /// Row-ID bytes are charged exclusively by the row-id domain. Their exact boxed owner is
    /// still one generic host allocation slot while this plan retains it.
    pub(super) fn append_host_allocation_slot(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), EngineError> {
        match &self.kind {
            DeviceInsertRowIdsKind::Exact(ids) => report.retain_slot_only(ids.as_ptr() as usize),
            DeviceInsertRowIdsKind::SyntheticNoIdentity | DeviceInsertRowIdsKind::ConsumedExact => {
                Err(EngineError::ApplyFailed(
                    "resident INSERT host retention requires exact unconsumed row identities"
                        .to_string(),
                ))
            }
        }
    }

    /// Allocation-free pre-lease geometry. The row-ID bytes are charged by their own domain;
    /// this owner contributes only its one live boxed-allocation slot.
    pub(super) fn host_allocation_geometry(
        &self,
    ) -> Result<crate::engine_insert_plan::host_retention::HostRetentionGeometry, EngineError> {
        match &self.kind {
            DeviceInsertRowIdsKind::Exact(_) => {
                let mut geometry =
                    crate::engine_insert_plan::host_retention::HostRetentionGeometry::default();
                geometry.checked_add_allocation_slot()?;
                Ok(geometry)
            }
            DeviceInsertRowIdsKind::SyntheticNoIdentity | DeviceInsertRowIdsKind::ConsumedExact => {
                Err(EngineError::ApplyFailed(
                    "resident INSERT host retention requires exact unconsumed row identities"
                        .to_string(),
                ))
            }
        }
    }

    pub(super) fn predict_host_allocation_slot(
        &self,
        geometry: &mut HostRetentionGeometry,
    ) -> Result<(), EngineError> {
        match &self.kind {
            DeviceInsertRowIdsKind::Exact(_) => geometry.checked_add_allocation_slot(),
            DeviceInsertRowIdsKind::SyntheticNoIdentity | DeviceInsertRowIdsKind::ConsumedExact => {
                Err(EngineError::ApplyFailed(
                    "resident INSERT host retention requires exact unconsumed row identities"
                        .to_string(),
                ))
            }
        }
    }
}

fn same_optional_device_region(
    left: &Option<Arc<CudaResidentDeviceMemory>>,
    right: &Option<Arc<CudaResidentDeviceMemory>>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        (None, None) => true,
        _ => false,
    }
}

/// CREATE auto-admission deliberately publishes this one zero-capacity descriptor without
/// capacity-sized sidecars. It is not a general missing-identity escape hatch: only the sole,
/// generation-bound shard-0 sentinel may roll over into the first identity-bearing generation.
fn is_empty_bootstrap_sentinel(
    engine: &Engine,
    table: &str,
    table_shards: &[RelationalResidentShard],
    identity: &PreparedOpenShardIdentity,
) -> bool {
    table_shards.len() == 1
        && identity.shard_id == 0
        && identity.row_start == 0
        && identity.row_count == 0
        && identity.capacity == 0
        && identity.created_by_region.is_none()
        && identity.row_id_region.is_none()
        && table_shards[0].deleted_by_region.is_none()
        && engine
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table.to_string(), 0))
            .is_none()
        && engine
            .read_state
            .residency
            .shard_created_by_memory
            .get(&(table.to_string(), 0))
            .is_none()
        && engine
            .read_state
            .residency
            .shard_row_id_memory
            .get(&(table.to_string(), 0))
            .is_none()
}

enum PreparedResidentAppendBranch {
    InPlace(PreparedInPlaceAppend),
    FixedRollover(PreparedFixedRollover),
    DenseRollover(PreparedDenseRollover),
    ConsumedDense,
}

/// The fixed in-place branch keeps the existing descriptor's created-by Arc or a private
/// capacity-sized replacement reserved before WAL. Mutation installs only the latter.
struct PreparedInPlaceAppend {
    pending_created_by: Option<super::rollover::PendingInPlaceCreatedBy>,
}

/// A fixed-width successor generation fully reserved before WAL. The allocation owner has already
/// uploaded all immutable bytes with a zero row-count header; mutation only stamps and publishes.
pub(super) struct PreparedFixedRollover {
    pub(super) pending: super::rollover::PendingFixedResidentShard,
    pub(super) capacity: usize,
    pub(super) new_shard_id: u32,
    pub(super) new_row_start: usize,
    pub(super) capacity_fit_evaluations: u64,
    pub(super) budget_scan_entries: u64,
    #[allow(dead_code)] // held for the unreachable indexed rollover reservation
    pub(super) planned_index_allocation_bytes: u64,
    #[allow(dead_code)] // held for the unreachable indexed rollover reservation
    pub(super) max_index_scratch_bytes: u64,
}

pub(super) enum PreparedInPlaceCreatedBy {
    Existing(Arc<CudaResidentDeviceMemory>),
    Reserved(super::rollover::PendingInPlaceCreatedBy),
}

/// Private, unpublished dense allocation set. All three CUDA allocations are created while the
/// plan owns the mutation and budget gates, before WAL. Apply may only upload/write them and hand
/// their Arcs to mutation's existing descriptor publisher.
pub(super) struct PreparedDenseRollover {
    pub(super) pending: super::rollover::PendingDenseResidentShard,
    pub(super) new_shard_id: u32,
    pub(super) new_row_start: usize,
    pub(super) budget_scan_entries: u64,
}

fn checked_rollover_coordinates(
    shard_id: u32,
    row_start: usize,
    row_count: usize,
) -> Option<(u32, usize)> {
    Some((shard_id.checked_add(1)?, row_start.checked_add(row_count)?))
}

/// Move-only, opaque pre-WAL device-append plan. `chunks` are host bytes only; device apply occurs
/// after canonical WAL/status buffering but before physical group durability and publication/ack.
pub(super) struct ResidentOpenShardAppendPlan<'a> {
    source: PreparedResidentAppendSource,
    identity: PreparedOpenShardIdentity,
    catalog_seq: Index,
    row_ids: DeviceInsertRowIds,
    bootstrap_sentinel: bool,
    branch: PreparedResidentAppendBranch,
    fixed_chunks: Option<Box<[PreparedResidentFixedChunk]>>,
    fixed_chunk_offsets: Option<Box<[u64]>>,
    int4_min_max: Box<[(i32, i32)]>,
    bool_uploads: Option<Box<[PreparedResidentFixedBoolUpload]>>,
    // Scalar-only, allocation-free geometry captured before any fixed plan owner is built.  The
    // identity-aware comparison occurs later in diagnostic retention reporting, never admission.
    host_retention_prediction: Option<HostRetentionGeometry>,
    // Field order is load-bearing: Rust drops fields in declaration order, so a failed or
    // completed apply releases its budget reservation before it releases the device gate.
    budget_allocation: Option<MutexGuard<'a, ()>>,
    // The plan crosses WAL while owning the only locks that can change its descriptor or consume
    // its sealed budget. This is a reservation, not a best-effort budget snapshot: no other
    // normal device publisher/allocation transaction can invalidate this geometry before apply.
    _device_apply: Option<MutexGuard<'a, ()>>,
}

#[allow(dead_code)] // scalar-only test inspection reads the reservation basis
pub(super) struct ResidentFixedRolloverReservationBasis<'a> {
    pub(super) predecessor_shard_id: u32,
    pub(super) predecessor_row_count: usize,
    pub(super) catalog_seq: Index,
    pub(super) incoming_rows: usize,
    pub(super) capacity: usize,
    pub(super) new_shard_id: u32,
    pub(super) new_row_start: usize,
    pub(super) capacity_fit_evaluations: u64,
    pub(super) budget_scan_entries: u64,
    pub(super) payload: &'a Arc<CudaResidentDeviceMemory>,
    pub(super) bool_layouts: &'a [ResidentDeviceBoolColumnLayout],
    pub(super) payload_bytes: u64,
    pub(super) created_by_bytes: u64,
    pub(super) row_id_bytes: u64,
    pub(super) payload_sidecar_allocation_bytes: u64,
    pub(super) payload_sidecar_allocation_count: u64,
    pub(super) planned_index_allocation_bytes: u64,
    pub(super) max_index_scratch_bytes: u64,
}

impl ResidentOpenShardAppendPlan<'_> {
    pub(crate) fn table_name(&self) -> &str {
        self.source.table_name()
    }

    pub(super) fn source(&self) -> &PreparedResidentAppendSource {
        &self.source
    }

    pub(super) fn row_count(&self) -> usize {
        self.source.row_count()
    }

    /// Retained host plan allocations for the materialized append plan. GPU allocation guards,
    /// point-route generation, lock guards, and row-ID bytes are intentionally absent; the row
    /// ID box contributes only its separate generic allocation slot.
    fn plan_owned_host_retention_report(&self) -> Result<HostRetentionReport, ExecuteError> {
        let mut report = HostRetentionReport::default();
        self.identity.append_host_retention(&mut report)?;
        self.row_ids.append_host_allocation_slot(&mut report)?;
        match &self.branch {
            PreparedResidentAppendBranch::InPlace(_)
            | PreparedResidentAppendBranch::ConsumedDense => {}
            PreparedResidentAppendBranch::FixedRollover(fixed) => {
                append_pending_bool_layout_box(&mut report, &fixed.pending.bool_layouts)?;
                append_pending_int4_stats_box(&mut report, &fixed.pending.int4_stats)?;
            }
            PreparedResidentAppendBranch::DenseRollover(dense) => {
                report.merge(dense.pending.payload.host_retention_report()?)?;
            }
        }
        if let Some(chunks) = self.fixed_chunks.as_ref() {
            report.retain_boxed_slice(chunks)?;
            for chunk in chunks.iter() {
                chunk.append_host_retention(&mut report)?;
            }
        }
        if let Some(offsets) = self.fixed_chunk_offsets.as_ref() {
            report.retain_boxed_slice(offsets)?;
        }
        report.retain_boxed_slice(&self.int4_min_max)?;
        if let Some(uploads) = self.bool_uploads.as_ref() {
            report.retain_boxed_slice(uploads)?;
            for upload in uploads.iter() {
                upload.append_host_retention(&mut report)?;
            }
        }
        Ok(report)
    }

    pub(super) fn host_retention_report(&self) -> Result<HostRetentionReport, ExecuteError> {
        let plan_owned = self.plan_owned_host_retention_report()?;
        if let Some(predicted) = self.host_retention_prediction {
            if !predicted.matches(&plan_owned)? {
                return Err(EngineError::ApplyFailed(
                    "materialized fixed append host retention diverged from its pre-materialization prediction"
                        .to_string(),
                )
                .into());
            }
        }
        let mut report = self.source.host_retention_report()?;
        report.merge(plan_owned)?;
        Ok(report)
    }

    pub(super) fn catalog_matches(&self, table: &RelationalTable, catalog_seq: Index) -> bool {
        self.catalog_seq == catalog_seq && source_matches_table(&self.source, table)
    }

    pub(super) fn identity_matches(&self, open: &RelationalResidentShard, pressured: bool) -> bool {
        self.identity.matches(open, pressured)
    }

    pub(super) fn int4_min_max(&self) -> &[(i32, i32)] {
        &self.int4_min_max
    }

    pub(crate) fn is_bound_bootstrap_sentinel(&self) -> bool {
        self.bootstrap_sentinel
    }

    fn take_row_ids(&mut self) -> Option<Box<[u64]>> {
        self.row_ids.take_exact()
    }

    pub(super) fn holds_budget_reservation(&self) -> bool {
        self.budget_allocation.is_some()
    }

    pub(super) fn indexed_fixed_rollover_host_materialization_scratch(
        &self,
    ) -> Option<HostRetentionGeometry> {
        match &self.branch {
            PreparedResidentAppendBranch::FixedRollover(fixed) => {
                Some(fixed.pending.host_materialization_scratch)
            }
            PreparedResidentAppendBranch::InPlace(_)
            | PreparedResidentAppendBranch::DenseRollover(_)
            | PreparedResidentAppendBranch::ConsumedDense => None,
        }
    }

    #[allow(dead_code)] // consumed by the production-compiled, unreachable reservation
    pub(super) fn indexed_in_place_reservation_basis(&self) -> (u32, usize, Index) {
        (
            self.identity.shard_id,
            self.identity.row_count,
            self.catalog_seq,
        )
    }

    pub(super) fn prepare_indexed_in_place_fused_apply_inputs(
        &self,
        expected_commit_seq: Index,
    ) -> Result<indexed_fused::IndexedInPlaceFusedApplyInputs<'_>, DeviceInsertPlanPrepareError>
    {
        indexed_fused::prepare_inputs(self, expected_commit_seq)
    }

    /// Recheck the exact private pre-WAL device envelope after the fused preparation footprint
    /// is known, while the append reservation still owns the budget gate.  A missing
    /// `created_by` sidecar was allocated by this same reservation but is not yet published in
    /// the shard map, so it remains part of this private envelope rather than the resident scan.
    pub(super) fn ensure_indexed_in_place_preparation_budget(
        &self,
        engine: &Engine,
        simultaneous_preparation_bytes: u64,
    ) -> Result<(), DeviceInsertPlanPrepareError> {
        let PreparedResidentAppendBranch::InPlace(in_place) = &self.branch else {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        };
        if self.budget_allocation.is_none() {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let pending_created_by_bytes = in_place
            .pending_created_by
            .as_ref()
            .map_or(0, |pending| pending.allocation_bytes);
        let required = pending_created_by_bytes
            .checked_add(simultaneous_preparation_bytes)
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        let (resident_bytes, _) =
            engine.relational_resident_bytes_and_entries_for_gpu(self.identity.gpu_id);
        let remaining = match engine.relational_residency_budget_bytes(self.identity.gpu_id) {
            Some(budget) => budget
                .checked_sub(resident_bytes)
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?,
            None => return Ok(()),
        };
        if required > remaining {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        Ok(())
    }

    #[allow(dead_code)] // consumed by the production-compiled, unreachable reservation
    pub(super) fn indexed_fixed_rollover_reservation_basis(
        &self,
    ) -> Option<ResidentFixedRolloverReservationBasis<'_>> {
        let PreparedResidentAppendBranch::FixedRollover(fixed) = &self.branch else {
            return None;
        };
        Some(ResidentFixedRolloverReservationBasis {
            predecessor_shard_id: self.identity.shard_id,
            predecessor_row_count: self.identity.row_count,
            catalog_seq: self.catalog_seq,
            incoming_rows: self.source.row_count(),
            capacity: fixed.capacity,
            new_shard_id: fixed.new_shard_id,
            new_row_start: fixed.new_row_start,
            capacity_fit_evaluations: fixed.capacity_fit_evaluations,
            budget_scan_entries: fixed.budget_scan_entries,
            payload: &fixed.pending.device_memory,
            bool_layouts: &fixed.pending.bool_layouts,
            payload_bytes: fixed.pending.payload_bytes,
            created_by_bytes: fixed.pending.created_by_bytes,
            row_id_bytes: fixed.pending.row_id_bytes,
            payload_sidecar_allocation_bytes: fixed.pending.allocation_bytes,
            payload_sidecar_allocation_count: fixed.pending.persistent_allocation_count,
            planned_index_allocation_bytes: fixed.planned_index_allocation_bytes,
            max_index_scratch_bytes: fixed.max_index_scratch_bytes,
        })
    }

    fn sidecars_still_match(&self, engine: &Engine) -> bool {
        let table = self.source.table_name();
        let shard_id = self.identity.shard_id;
        same_optional_device_region(
            &engine
                .read_state
                .residency
                .shard_created_by_memory
                .get(&(table.to_string(), shard_id)),
            &self.identity.created_by_region,
        ) && same_optional_device_region(
            &engine
                .read_state
                .residency
                .shard_row_id_memory
                .get(&(table.to_string(), shard_id)),
            &self.identity.row_id_region,
        )
    }

    pub(super) fn is_rollover(&self) -> bool {
        matches!(
            self.branch,
            PreparedResidentAppendBranch::FixedRollover(_)
                | PreparedResidentAppendBranch::DenseRollover(_)
        )
    }

    pub(super) fn chunks_for_in_place(
        &mut self,
        capacity: usize,
        row_start: usize,
    ) -> Option<PreparedResidentFixedChunkOwners> {
        if self.is_rollover()
            || self.identity.capacity != capacity
            || self.identity.row_count != row_start
        {
            return None;
        }
        let chunks = self.fixed_chunks.take()?;
        let offsets = self.fixed_chunk_offsets.take()?;
        (chunks.len() == offsets.len())
            .then_some(PreparedResidentFixedChunkOwners { chunks, offsets })
    }

    fn apply_shape_matches(&self, created_by: &AppendCreatedBy<'_>) -> bool {
        created_by.stamps_for(self.row_count()).is_some()
    }

    pub(super) fn take_bool_uploads(
        &mut self,
        layouts: &[ResidentDeviceBoolColumnLayout],
    ) -> Option<Box<[PreparedResidentFixedBoolUpload]>> {
        let uploads = self.bool_uploads.take()?;
        (uploads.len() == layouts.len()
            && uploads.iter().zip(layouts).all(|(upload, layout)| {
                upload.name.as_ref() == layout.name && upload.values.len() == self.row_count()
            }))
        .then_some(uploads)
    }

    pub(super) fn dense_rollover_payload_len(&self) -> Option<u64> {
        match &self.branch {
            PreparedResidentAppendBranch::DenseRollover(dense) => Some(dense.pending.payload_bytes),
            _ => None,
        }
    }

    pub(super) fn fixed_rollover_payload_len(&self) -> Option<u64> {
        match &self.branch {
            PreparedResidentAppendBranch::FixedRollover(fixed) => Some(fixed.pending.payload_bytes),
            _ => None,
        }
    }

    pub(super) fn take_fixed_rollover(&mut self) -> Option<PreparedFixedRollover> {
        match std::mem::replace(
            &mut self.branch,
            PreparedResidentAppendBranch::ConsumedDense,
        ) {
            PreparedResidentAppendBranch::FixedRollover(fixed) => Some(fixed),
            branch => {
                self.branch = branch;
                None
            }
        }
    }

    pub(super) fn take_in_place_created_by(
        &mut self,
        capacity: usize,
        row_start: usize,
    ) -> Option<PreparedInPlaceCreatedBy> {
        if self.identity.capacity != capacity || self.identity.row_count != row_start {
            return None;
        }
        let PreparedResidentAppendBranch::InPlace(in_place) = &mut self.branch else {
            return None;
        };
        match in_place.pending_created_by.take() {
            Some(reserved) => Some(PreparedInPlaceCreatedBy::Reserved(reserved)),
            None => self
                .identity
                .created_by_region
                .as_ref()
                .map(|region| PreparedInPlaceCreatedBy::Existing(Arc::clone(region))),
        }
    }

    pub(super) fn take_dense_rollover(&mut self) -> Option<PreparedDenseRollover> {
        match std::mem::replace(
            &mut self.branch,
            PreparedResidentAppendBranch::ConsumedDense,
        ) {
            PreparedResidentAppendBranch::DenseRollover(dense) => Some(dense),
            branch => {
                self.branch = branch;
                None
            }
        }
    }
}

/// The only post-compile token that can reach typed device append. Its physical variant is
/// deliberately private to residency: the concurrent wave can own this opaque plan, but cannot
/// inspect, replace, or construct a resident append source.
pub(crate) struct DeviceInsertPlan<'a>(DeviceInsertPlanKind<'a>);

enum DeviceInsertPlanKind<'a> {
    ResidentOpenShardAppend(ResidentOpenShardAppendPlan<'a>),
}

impl DeviceInsertPlan<'_> {
    pub(crate) fn table_name(&self) -> &str {
        match &self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan) => plan.table_name(),
        }
    }

    pub(crate) fn is_bound_bootstrap_sentinel(&self) -> bool {
        match &self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan) => {
                plan.is_bound_bootstrap_sentinel()
            }
        }
    }

    /// Consume this plan only after the canonical typed-INSERT claim has been issued.  The
    /// opaque permit is minted exclusively by that claim, so residency never accepts a detached
    /// commit stamp or an independently assembled post-WAL apply request.
    pub(crate) fn apply_after_typed_wal_claim(
        self,
        engine: &Engine,
        permit: crate::engine_dml_concurrent::TypedInsertPostWalApplyPermit,
    ) -> Result<(), DeviceInsertPlanApplyError> {
        let created_by = permit.into_append_created_by();
        match self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan) => {
                engine.apply_resident_open_shard_append(plan, created_by)
            }
        }
    }
}

impl Engine {
    /// Compile the one move-only semantic batch into the opaque device plan. Physical branches
    /// are fixed-width append/rollover and dense nullable-or-text rollover; neither adds a
    /// parallel wave carrier or publisher.
    pub(crate) fn compile_typed_insert_device_plan<'a>(
        &'a self,
        batch: TypedInsertBatch,
        row_ids: DeviceInsertRowIds,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        let source = batch
            .into_resident_append_source()
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        self.prepare_resident_open_shard_append(source, row_ids)
            .map(|plan| DeviceInsertPlan(DeviceInsertPlanKind::ResidentOpenShardAppend(plan)))
    }

    /// Consume a sealed source and prepare its exact resident append shape before WAL. A decline is
    /// side-effect-free; the source is dropped and the caller may still take the legacy route.
    ///
    /// The caller must consume the returned plan after canonical WAL/status buffering and before
    /// physical group durability, while it owns the canonical commit boundary. The plan deliberately
    /// retains the device gate (and, when needed, allocation gate) only across that device-apply
    /// interval; it is not an asynchronous durability-tail or publication/ack handle.
    fn prepare_resident_open_shard_append<'a>(
        &'a self,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
    ) -> Result<ResidentOpenShardAppendPlan<'a>, DeviceInsertPlanPrepareError> {
        self.prepare_resident_open_shard_append_core(
            source,
            row_ids,
            ResidentOpenShardAppendPreparationMode::LiveUnindexed,
            None,
        )
    }

    fn prepare_resident_open_shard_append_core<'a>(
        &'a self,
        mut source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        mode: ResidentOpenShardAppendPreparationMode,
        preheld_device_apply: Option<MutexGuard<'a, ()>>,
    ) -> Result<ResidentOpenShardAppendPlan<'a>, DeviceInsertPlanPrepareError> {
        let live_unindexed = matches!(mode, ResidentOpenShardAppendPreparationMode::LiveUnindexed);
        let indexed_in_place_proof = matches!(
            mode,
            ResidentOpenShardAppendPreparationMode::IndexedInPlaceReservation { .. }
        );
        let indexed_fixed_rollover_proof = matches!(
            mode,
            ResidentOpenShardAppendPreparationMode::IndexedFixedRolloverReservation { .. }
        );
        let apply_leader =
            crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(std::cell::Cell::get);
        // A lane leader already owns this lock, but the current intentional general-path policy
        // does not export a cross-WAL plan after its leader scope ends. It declines before WAL;
        // ordinary callers retain the lock in the returned move-only plan, closing
        // descriptor/generation races through apply.
        if live_unindexed && apply_leader {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let device_apply = match (mode, preheld_device_apply) {
            (ResidentOpenShardAppendPreparationMode::LiveUnindexed, None) => Some(
                self.read_state
                    .residency
                    .mutation_gate
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ),
            (
                ResidentOpenShardAppendPreparationMode::IndexedInPlaceReservation { .. },
                Some(held),
            ) => Some(held),
            (
                ResidentOpenShardAppendPreparationMode::IndexedFixedRolloverReservation { .. },
                Some(held),
            ) => Some(held),
            _ => return Err(DeviceInsertPlanPrepareError::UnsupportedShape),
        };
        let catalog = self.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(source.table_name())
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        let catalog_matches = if live_unindexed {
            source.prepared_catalog_seq() == catalog.commit_seq
        } else {
            source.prepared_catalog_seq() <= catalog.commit_seq
        };
        let source_matches = if live_unindexed {
            source_matches_table(&source, table)
        } else {
            source_matches_indexed_in_place_reservation(&source, table)
        };
        if source.row_count() == 0 || !catalog_matches || !source_matches {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let row_ids_present = row_ids.is_exact();
        if !row_ids.exact_len_matches(source.row_count()) {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        // Dense sources are selected before any fixed-width descriptor geometry. Their fresh
        // generation owns exact text/validity layouts, while this identity only binds the OLD
        // open descriptor that will be followed by that rollover.
        let dense_rollover = source.requires_dense_rollover();
        let pressured = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .clone();
        let (identity, gpu_id, bootstrap_sentinel, host_retention_prediction) = {
            let shards = self.read_state.residency.shards.load();
            let table_shards = shards
                .get(source.table_name())
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let open = table_shards
                .last()
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            // The indexed proof paths capture this scalar shape before the identity, final
            // chunks, or device generation exist.  It performs no CUDA work and allocates no
            // retention identity map; post-materialization reporting verifies exact parity.
            let host_retention_prediction = if indexed_in_place_proof {
                Some(
                    indexed_in_place_plan_owned_host_retention_prediction(
                        &source, &row_ids, open, table,
                    )
                    .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?,
                )
            } else if indexed_fixed_rollover_proof {
                Some(
                    indexed_fixed_rollover_plan_owned_host_retention_prediction(
                        &source, &row_ids, open, table,
                    )
                    .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?,
                )
            } else {
                None
            };
            let identity = PreparedOpenShardIdentity::from_open(open);
            // `matches` binds validity and the descriptor generation; the source checks above
            // bind the catalog. Keep the no-sidecar exception narrower than either ordinary
            // empty or ordinary missing-sidecar states, which must still decline before WAL.
            let bootstrap_sentinel =
                is_empty_bootstrap_sentinel(self, source.table_name(), table_shards, &identity);
            let expected_i32 = table
                .columns
                .iter()
                .filter(|column| matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
                .count();
            if !identity.matches(open, pressured.contains(&open.gpu_id))
                || !identity.matches_scalar_table_layout(table)
                || (!dense_rollover && !identity.matches_fixed_table_layout(table))
                || open.resident_device_int4_column_stats.len() != expected_i32
                || (row_ids_present && identity.row_id_region.is_none() && !bootstrap_sentinel)
                || (!row_ids_present && (identity.row_id_region.is_some() || bootstrap_sentinel))
                || identity.created_by_region.as_ref().is_some_and(|region| {
                    u64::try_from(identity.capacity)
                        .ok()
                        .and_then(|capacity| capacity.checked_mul(8))
                        .is_none_or(|required| region.metadata().allocated_bytes < required)
                })
                || identity.row_id_region.as_ref().is_some_and(|region| {
                    u64::try_from(identity.capacity)
                        .ok()
                        .and_then(|capacity| capacity.checked_mul(8))
                        .is_none_or(|required| region.metadata().allocated_bytes < required)
                })
            {
                return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
            }
            (
                identity,
                open.gpu_id,
                bootstrap_sentinel,
                host_retention_prediction,
            )
        };
        let k = source.row_count();
        if indexed_in_place_proof
            && (dense_rollover
                || bootstrap_sentinel
                || identity
                    .row_count
                    .checked_add(k)
                    .is_none_or(|end| end > identity.capacity))
        {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        if indexed_fixed_rollover_proof
            && (dense_rollover
                || bootstrap_sentinel
                || table.indexes.is_empty()
                || identity
                    .row_count
                    .checked_add(k)
                    .is_none_or(|end| end <= identity.capacity))
        {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        // NULL validity or varlen text has no capacity-strided OPEN-shard representation. It was
        // selected before descriptor geometry, so it cannot inherit an in-place arm from headroom.
        let (
            branch,
            fixed_chunks,
            fixed_chunk_offsets,
            int4_min_max,
            bool_uploads,
            budget_allocation,
        ) = if dense_rollover {
            let payload = source
                .checked_dense_payload(table)
                .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let sidecar_bytes = u64::try_from(k)
                .ok()
                .and_then(|rows| rows.checked_mul(std::mem::size_of::<u64>() as u64))
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let row_id_bytes = if row_ids_present { sidecar_bytes } else { 0 };
            let payload_bytes = payload
                .device_payload_len()
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let requested_bytes = payload_bytes
                .checked_add(sidecar_bytes)
                .and_then(|bytes| bytes.checked_add(row_id_bytes))
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let row_id_payload = row_ids.pre_wal_payload();
            if row_ids_present != row_id_payload.is_some() {
                return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
            }
            let (new_shard_id, new_row_start) = checked_rollover_coordinates(
                identity.shard_id,
                identity.row_start,
                identity.row_count,
            )
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let budget_allocation = self
                .read_state
                .residency
                .budget_allocation_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let still_matches = self
                .read_state
                .residency
                .shards
                .load()
                .get(source.table_name())
                .and_then(|shards| shards.last())
                .is_some_and(|open| identity.matches(open, pressured.contains(&open.gpu_id)));
            let (resident_bytes, budget_scan_entries) =
                self.relational_resident_bytes_and_entries_for_gpu(gpu_id);
            let remaining_budget = match self.relational_residency_budget_bytes(gpu_id) {
                Some(budget) => match budget.checked_sub(resident_bytes) {
                    Some(remaining) => Some(remaining),
                    None => {
                        self.read_state
                            .residency
                            .rollover_budget_declines
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return Err(if bootstrap_sentinel {
                            DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                        } else {
                            DeviceInsertPlanPrepareError::UnsupportedShape
                        });
                    }
                },
                None => None,
            };
            if !still_matches
                || remaining_budget.is_some_and(|remaining| requested_bytes > remaining)
            {
                self.read_state
                    .residency
                    .rollover_budget_declines
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(if bootstrap_sentinel {
                    DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                } else {
                    DeviceInsertPlanPrepareError::UnsupportedShape
                });
            }
            let pending = super::rollover::PendingDenseResidentShard::reserve_pre_wal(
                self,
                gpu_id,
                payload,
                sidecar_bytes,
                row_id_payload,
            )
            .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
            if remaining_budget.is_some_and(|remaining| pending.allocation_bytes > remaining) {
                self.read_state
                    .residency
                    .rollover_budget_declines
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(if bootstrap_sentinel {
                    DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                } else {
                    DeviceInsertPlanPrepareError::UnsupportedShape
                });
            }
            (
                PreparedResidentAppendBranch::DenseRollover(PreparedDenseRollover {
                    pending,
                    new_shard_id,
                    new_row_start,
                    budget_scan_entries,
                }),
                None,
                None,
                Box::default(),
                None,
                Some(budget_allocation),
            )
        } else if !bootstrap_sentinel
            && identity
                .row_count
                .checked_add(k)
                .is_some_and(|end| end <= identity.capacity)
        {
            let chunks = source
                .checked_append_chunks(identity.capacity, identity.row_count)
                .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let index_scratch_bytes = match mode {
                ResidentOpenShardAppendPreparationMode::LiveUnindexed => 0,
                ResidentOpenShardAppendPreparationMode::IndexedInPlaceReservation {
                    index_scratch_bytes,
                } => index_scratch_bytes,
                ResidentOpenShardAppendPreparationMode::IndexedFixedRolloverReservation {
                    ..
                } => {
                    return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                }
            };
            let reserve_created_by = identity.created_by_region.is_none();
            // The proof mode always carries the budget guard, including when the append has
            // an existing sidecar.  Its exact pooled index preparation footprint is charged
            // alongside any append-sidecar reservation before the latter allocates.
            let retain_budget_guard = reserve_created_by || !live_unindexed;
            let (pending_created_by, budget_allocation) = if retain_budget_guard {
                let budget_allocation = self
                    .read_state
                    .residency
                    .budget_allocation_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let still_matches = self
                    .read_state
                    .residency
                    .shards
                    .load()
                    .get(source.table_name())
                    .and_then(|shards| shards.last())
                    .is_some_and(|open| identity.matches(open, pressured.contains(&open.gpu_id)));
                let sidecar_bytes = if reserve_created_by {
                    u64::try_from(identity.capacity)
                        .ok()
                        .and_then(|capacity| {
                            capacity.checked_mul(std::mem::size_of::<u64>() as u64)
                        })
                        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?
                } else {
                    0
                };
                let required_before_allocation = sidecar_bytes
                    .checked_add(index_scratch_bytes)
                    .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                let (resident_bytes, _) =
                    self.relational_resident_bytes_and_entries_for_gpu(gpu_id);
                let remaining_budget = match self.relational_residency_budget_bytes(gpu_id) {
                    Some(budget) => match budget.checked_sub(resident_bytes) {
                        Some(remaining) => Some(remaining),
                        None => return Err(DeviceInsertPlanPrepareError::UnsupportedShape),
                    },
                    None => None,
                };
                if !still_matches
                    || remaining_budget
                        .is_some_and(|remaining| required_before_allocation > remaining)
                {
                    return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                }
                let pending_created_by = if reserve_created_by {
                    // Reserve and allocate the capacity-sized created_by sidecar before WAL.
                    // Device apply only installs this exact Arc and stamps its still-invisible slots.
                    let pending = super::rollover::PendingInPlaceCreatedBy::reserve_pre_wal(
                        self,
                        gpu_id,
                        identity.capacity,
                    )
                    .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
                    let actual_required = pending
                        .allocation_bytes
                        .checked_add(index_scratch_bytes)
                        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                    if pending.capacity_bytes != sidecar_bytes
                        || pending.allocation_bytes < sidecar_bytes
                        || remaining_budget.is_some_and(|remaining| actual_required > remaining)
                    {
                        return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                    }
                    Some(pending)
                } else {
                    None
                };
                (pending_created_by, Some(budget_allocation))
            } else {
                (None, None)
            };
            (
                PreparedResidentAppendBranch::InPlace(PreparedInPlaceAppend { pending_created_by }),
                Some(chunks.chunks),
                Some(chunks.offsets),
                source
                    .int4_min_max()
                    .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?,
                Some(
                    source
                        .fixed_bool_uploads(table)
                        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?,
                ),
                budget_allocation,
            )
        } else {
            if indexed_in_place_proof {
                return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
            }
            // Rollover capacity is an allocation transaction even though this pre-WAL phase makes
            // no allocation. Snapshot the exact budget under the same lock the publisher rechecks.
            let budget_allocation = self
                .read_state
                .residency
                .budget_allocation_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let still_matches = self
                .read_state
                .residency
                .shards
                .load()
                .get(source.table_name())
                .and_then(|shards| shards.last())
                .is_some_and(|open| identity.matches(open, pressured.contains(&open.gpu_id)));
            if !still_matches {
                return Err(if bootstrap_sentinel {
                    DeviceInsertPlanPrepareError::RetryableBoundBootstrapState
                } else {
                    DeviceInsertPlanPrepareError::UnsupportedShape
                });
            }
            let (resident_bytes, budget_scan_entries) =
                self.relational_resident_bytes_and_entries_for_gpu(gpu_id);
            let remaining_budget = match self.relational_residency_budget_bytes(gpu_id) {
                Some(budget) => match budget.checked_sub(resident_bytes) {
                    Some(remaining) => Some(remaining),
                    None => {
                        self.read_state
                            .residency
                            .rollover_budget_declines
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return Err(if bootstrap_sentinel {
                            DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                        } else {
                            DeviceInsertPlanPrepareError::UnsupportedShape
                        });
                    }
                },
                None => None,
            };
            let (named_indexes_required, max_index_scratch_bytes) = match mode {
                ResidentOpenShardAppendPreparationMode::IndexedFixedRolloverReservation {
                    max_index_scratch_bytes,
                } => (true, max_index_scratch_bytes),
                ResidentOpenShardAppendPreparationMode::LiveUnindexed => (false, 0),
                ResidentOpenShardAppendPreparationMode::IndexedInPlaceReservation { .. } => {
                    return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                }
            };
            let persistent_budget = match remaining_budget {
                Some(remaining) => match remaining.checked_sub(max_index_scratch_bytes) {
                    Some(persistent) => Some(persistent),
                    None => {
                        self.read_state
                            .residency
                            .rollover_budget_declines
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return Err(if bootstrap_sentinel {
                            DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                        } else {
                            DeviceInsertPlanPrepareError::UnsupportedShape
                        });
                    }
                },
                None => None,
            };
            let desired = super::rollover::ResidentRolloverPlan::fixed_width_desired_capacity(
                k,
                Some(self.shard_size_target()),
            )
            .ok_or(if bootstrap_sentinel {
                DeviceInsertPlanPrepareError::RetryableBoundBootstrapState
            } else {
                DeviceInsertPlanPrepareError::UnsupportedShape
            })?;
            let rollover = match super::rollover::ResidentRolloverPlan::fixed_width_null_free_table(
                table,
                k,
                desired,
                row_ids_present,
                named_indexes_required,
                persistent_budget,
            ) {
                Ok(Some(plan)) => plan,
                Ok(None) => {
                    self.read_state
                        .residency
                        .rollover_budget_declines
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(if bootstrap_sentinel {
                        DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                    } else {
                        DeviceInsertPlanPrepareError::UnsupportedShape
                    });
                }
                Err(_) => {
                    self.read_state
                        .residency
                        .rollover_budget_declines
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(if bootstrap_sentinel {
                        DeviceInsertPlanPrepareError::RetryableBoundBootstrapState
                    } else {
                        DeviceInsertPlanPrepareError::UnsupportedShape
                    });
                }
            };
            let chunks = source
                .checked_append_chunks(rollover.capacity(), 0)
                .map_err(|_| {
                    if bootstrap_sentinel {
                        DeviceInsertPlanPrepareError::RetryableBoundBootstrapState
                    } else {
                        DeviceInsertPlanPrepareError::UnsupportedShape
                    }
                })?;
            let int4_stats = source
                .fixed_int4_stats(table)
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let bool_uploads = source
                .fixed_bool_uploads(table)
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let row_id_payload = row_ids.pre_wal_payload();
            if row_ids_present != row_id_payload.is_some() {
                return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
            }
            let (new_shard_id, new_row_start) = checked_rollover_coordinates(
                identity.shard_id,
                identity.row_start,
                identity.row_count,
            )
            .ok_or(if bootstrap_sentinel {
                DeviceInsertPlanPrepareError::RetryableBoundBootstrapState
            } else {
                DeviceInsertPlanPrepareError::UnsupportedShape
            })?;
            let pending = super::rollover::PendingFixedResidentShard::reserve_pre_wal(
                self,
                gpu_id,
                &rollover,
                k,
                chunks,
                bool_uploads,
                int4_stats,
                row_id_payload,
            )
            .map_err(|_| {
                if bootstrap_sentinel {
                    DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                } else {
                    DeviceInsertPlanPrepareError::UnsupportedShape
                }
            })?;
            let planned_index_allocation_bytes = rollover.named_index_bytes();
            let actual_peak_bytes = pending
                .allocation_bytes
                .checked_add(planned_index_allocation_bytes)
                .and_then(|bytes| bytes.checked_add(max_index_scratch_bytes))
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            if pending.allocation_bytes != rollover.allocation_bytes_before_indexes()
                || remaining_budget.is_some_and(|remaining| actual_peak_bytes > remaining)
            {
                self.read_state
                    .residency
                    .rollover_budget_declines
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(if bootstrap_sentinel {
                    DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                } else {
                    DeviceInsertPlanPrepareError::UnsupportedShape
                });
            }
            (
                PreparedResidentAppendBranch::FixedRollover(PreparedFixedRollover {
                    pending,
                    capacity: rollover.capacity(),
                    new_shard_id,
                    new_row_start,
                    capacity_fit_evaluations: rollover.capacity_scan_entries(),
                    budget_scan_entries,
                    planned_index_allocation_bytes,
                    max_index_scratch_bytes,
                }),
                None,
                None,
                Box::default(),
                None,
                Some(budget_allocation),
            )
        };
        Ok(ResidentOpenShardAppendPlan {
            source,
            identity,
            catalog_seq: catalog.commit_seq,
            row_ids,
            bootstrap_sentinel,
            branch,
            fixed_chunks,
            fixed_chunk_offsets,
            int4_min_max,
            bool_uploads,
            host_retention_prediction,
            budget_allocation,
            _device_apply: device_apply,
        })
    }

    /// Apply exactly one pre-WAL plan. Any failure is fatal: a caller that has already written WAL
    /// must wedge/recover instead of falling back to a second write path.
    fn apply_resident_open_shard_append(
        &self,
        mut plan: ResidentOpenShardAppendPlan<'_>,
        created_by: AppendCreatedBy<'_>,
    ) -> Result<(), DeviceInsertPlanApplyError> {
        if !plan.apply_shape_matches(&created_by) {
            return Err(DeviceInsertPlanApplyError::ShapeDrift);
        }
        if !plan_matches_live_open(self, &plan) {
            return Err(DeviceInsertPlanApplyError::PlanDrift);
        }
        let row_ids = plan.take_row_ids();
        let table = plan.source().table_name().to_string();
        self.try_append_to_resident_open_shard(
            &table,
            ResidentAppendSource::DevicePlan(&mut plan),
            created_by,
            row_ids.as_deref(),
        )
        .then_some(())
        .ok_or(DeviceInsertPlanApplyError::PublisherFailure)
    }

    /// Internal reservation adapter over the one append compiler. It is intentionally not used
    /// by live `DeviceInsertPlan` selection: the returned append reservation is owned only by
    /// the private indexed physical-reservation owner until the future handoff is designed.
    #[allow(dead_code)] // reached only by scalar-only inspection until live selection is designed
    pub(super) fn prepare_resident_open_shard_append_indexed_in_place_reservation<'a>(
        &'a self,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        device_apply: MutexGuard<'a, ()>,
        index_scratch_bytes: u64,
        _permit: IndexedPhysicalMaterializationPermit,
    ) -> Result<ResidentOpenShardAppendPlan<'a>, DeviceInsertPlanPrepareError> {
        self.prepare_resident_open_shard_append_core(
            source,
            row_ids,
            ResidentOpenShardAppendPreparationMode::IndexedInPlaceReservation {
                index_scratch_bytes,
            },
            Some(device_apply),
        )
    }

    /// Internal fixed-rollover reservation adapter. It reserves the private payload/sidecars
    /// plus the exact index and scratch envelope, without constructing a `DeviceInsertPlan` or
    /// exposing WAL, apply, cache, or descriptor-publication authority.
    #[allow(dead_code)] // reached only by scalar-only inspection until live selection is designed
    pub(super) fn prepare_resident_open_shard_append_indexed_fixed_rollover_reservation<'a>(
        &'a self,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        device_apply: MutexGuard<'a, ()>,
        max_index_scratch_bytes: u64,
        _permit: IndexedPhysicalMaterializationPermit,
    ) -> Result<ResidentOpenShardAppendPlan<'a>, DeviceInsertPlanPrepareError> {
        self.prepare_resident_open_shard_append_core(
            source,
            row_ids,
            ResidentOpenShardAppendPreparationMode::IndexedFixedRolloverReservation {
                max_index_scratch_bytes,
            },
            Some(device_apply),
        )
    }
}

fn source_matches_table(source: &PreparedResidentAppendSource, table: &RelationalTable) -> bool {
    source.table_name() == table.name
        && source.exact_single_table_dependency()
        && source.table_oid() == table.oid
        && table.indexes.is_empty()
        && table.foreign_keys.is_empty()
        && crate::engine_transaction_reset::table_schema_digest(table)
            .is_ok_and(|digest| digest == source.schema_digest())
        && table.columns.len() == source.columns().len()
        && table
            .columns
            .iter()
            .zip(source.columns())
            .all(|(column, source_column)| {
                is_live_resident_append_type(column.ty)
                    // Scalar defaults were already materialized in the sealed vectors, and
                    // metadata-only domain bindings were revalidated by the prepared plan before
                    // row identities/device compilation. They do not change resident layout.
                    && column.table_oid == table.oid
                    && column.id == source_column.column_id()
                    && column.attnum == source_column.attnum()
                    && column.ty == source_column.ty()
                    && column.type_oid == source_column.type_oid()
                    && column.type_size == source_column.type_size()
                    && source_column.ty() == column.ty
            })
}

/// The private reservation can inspect an already-published indexed table while preserving the
/// production predicate above verbatim.  It deliberately shares every non-index source/catalog
/// witness with the live route, then lets the proof owner establish complete raw-index coverage.
pub(super) fn source_matches_indexed_in_place_reservation(
    source: &PreparedResidentAppendSource,
    table: &RelationalTable,
) -> bool {
    source.table_name() == table.name
        && source.exact_single_table_dependency()
        && source.table_oid() == table.oid
        && table.foreign_keys.is_empty()
        && crate::engine_transaction_reset::table_schema_digest(table)
            .is_ok_and(|digest| digest == source.schema_digest())
        && table.columns.len() == source.columns().len()
        && table
            .columns
            .iter()
            .zip(source.columns())
            .all(|(column, source_column)| {
                is_live_resident_append_type(column.ty)
                    && column.table_oid == table.oid
                    && column.id == source_column.column_id()
                    && column.attnum == source_column.attnum()
                    && column.ty == source_column.ty()
                    && column.type_oid == source_column.type_oid()
                    && column.type_size == source_column.type_size()
                    && source_column.ty() == column.ty
            })
}

fn is_live_resident_append_type(ty: SqlType) -> bool {
    matches!(
        ty,
        SqlType::Int2
            | SqlType::Int4
            | SqlType::Date
            | SqlType::Int8
            | SqlType::Timestamp
            | SqlType::Numeric { .. }
            | SqlType::Uuid
            | SqlType::Bool
            | SqlType::Text
    )
}

fn plan_matches_live_open(engine: &Engine, plan: &ResidentOpenShardAppendPlan) -> bool {
    let catalog = engine.catalog_snapshot();
    let Some(table) = catalog.relational_catalog.get(plan.source().table_name()) else {
        return false;
    };
    if !plan.catalog_matches(table, catalog.commit_seq) {
        return false;
    }
    let pressured = engine
        .router
        .runtime()
        .snapshot()
        .memory_pressured_gpu_ids
        .clone();
    plan.sidecars_still_match(engine)
        && engine
            .read_state
            .residency
            .shards
            .load()
            .get(plan.source().table_name())
            .and_then(|shards| shards.last())
            .is_some_and(|open| plan.identity_matches(open, pressured.contains(&open.gpu_id)))
}

#[cfg(test)]
mod ownership_tests {
    #[test]
    fn live_device_plan_apply_requires_the_post_wal_typed_claim_permit() {
        let source = include_str!("fixed_insert.rs")
            .split("\n#[cfg(test)]\nmod ownership_tests")
            .next()
            .expect("implementation precedes tests");
        assert!(source.contains("fn apply_after_typed_wal_claim"));
        assert!(source.contains("TypedInsertPostWalApplyPermit"));
        assert!(source.contains("permit.into_append_created_by()"));
        assert!(
            !source.contains("pub(crate) fn apply(\n"),
            "DeviceInsertPlan must not expose an unclaimed apply entry"
        );
    }

    #[test]
    fn exact_row_id_box_is_slot_only_and_synthetic_or_consumed_forms_fail_closed() {
        let row_ids = super::DeviceInsertRowIds::exact(vec![7_u64, 8].into());
        let mut report = crate::engine_insert_plan::host_retention::HostRetentionReport::default();
        row_ids.append_host_allocation_slot(&mut report).unwrap();
        assert_eq!(report.retained_bytes(), 0);
        assert_eq!(report.allocation_slots().unwrap(), 1);

        let synthetic = super::DeviceInsertRowIds::synthetic_no_identity();
        assert!(synthetic.append_host_allocation_slot(&mut report).is_err());
        let mut consumed = super::DeviceInsertRowIds::exact(vec![9_u64].into());
        assert!(consumed.take_exact().is_some());
        assert!(consumed.append_host_allocation_slot(&mut report).is_err());
    }

    #[test]
    fn indexed_reservation_mode_reuses_the_single_append_preparation_core() {
        let source = include_str!("fixed_insert.rs")
            .split("\n#[cfg(test)]\nmod ownership_tests")
            .next()
            .expect("implementation precedes tests");
        let core = source
            .split("fn prepare_resident_open_shard_append_core")
            .nth(1)
            .and_then(|section| {
                section
                    .split("\n    /// Apply exactly one pre-WAL plan")
                    .next()
            })
            .expect("one append preparation core");
        assert!(core.contains("LiveUnindexed"));
        assert!(core.contains("IndexedInPlaceReservation"));
        assert!(core.contains("source_matches_table(&source, table)"));
        assert!(core.contains("source_matches_indexed_in_place_reservation(&source, table)"));
        let reservation_decline = core
            .find("&& (dense_rollover")
            .expect("proof mode rejects dense/bootstrap/headroom shapes");
        for later in [
            ".checked_dense_payload(table)",
            "fixed_width_desired_capacity",
            "PendingInPlaceCreatedBy::reserve_pre_wal",
            "budget_allocation_lock",
        ] {
            assert!(
                reservation_decline < core.find(later).expect("ordinary later preparation branch"),
                "reservation mode must decline before {later}"
            );
        }
        assert!(
            [
                "dense_rollover",
                "bootstrap_sentinel",
                ".checked_add(k)",
                "end > identity.capacity"
            ]
            .into_iter()
            .all(|decline| core[reservation_decline..].contains(decline)),
            "reservation mode must explicitly decline every fixed/bootstrap/dense shape before allocation"
        );
        let adapter = source
            .split("fn prepare_resident_open_shard_append_indexed_in_place_reservation")
            .nth(1)
            .and_then(|section| section.split("\n}\n\nfn source_matches_table").next())
            .expect("proof adapter");
        assert!(adapter.contains("prepare_resident_open_shard_append_core"));
        assert!(!adapter.contains("let catalog ="));
        assert!(!adapter.contains("reserve_pre_wal"));

        let live_selector = source
            .split("pub(crate) fn compile_typed_insert_device_plan")
            .nth(1)
            .and_then(|section| {
                section
                    .split("fn prepare_resident_open_shard_append")
                    .next()
            })
            .expect("sole live device-plan selector");
        assert!(live_selector.contains("ResidentOpenShardAppend"));
        assert!(
            !live_selector.contains("IndexedInPlaceReservation")
                && !live_selector.contains("IndexedFixedRolloverReservation"),
            "the live DeviceInsertPlan selector must not choose an indexed reservation"
        );
    }

    #[test]
    fn indexed_in_place_fused_owner_is_all_i32_only_and_has_no_publication_surface() {
        let source = include_str!("fixed_insert/indexed_fused.rs");
        let fused = source
            .split("pub(in super::super) struct IndexedInPlaceFusedApplyInputs")
            .nth(1)
            .and_then(|section| {
                section
                    .split("pub(in super::super) fn prepare_inputs")
                    .next()
            })
            .expect("private fused input and owner section");
        for required in [
            "prepare_i32_fused_apply",
            "index: None",
            "created_by_stamps",
            "stamps_match_expected_commit",
            "owner_array_backing_identity",
            "staging_backing_identity",
        ] {
            assert!(
                fused.contains(required),
                "indexed fused owner must retain {required}"
            );
        }
        for forbidden in [
            "apply_before_header",
            ".apply(",
            "submit_resident",
            ".publish(",
            "shard_pk_device_index",
        ] {
            assert!(
                !fused.contains(forbidden),
                "indexed fused owner must not expose {forbidden}"
            );
        }
        let inputs = source
            .split("pub(in super::super) fn prepare_inputs")
            .nth(1)
            .and_then(|section| {
                section
                    .split("pub(in super::super) fn source_is_all_i32_fixed")
                    .next()
            })
            .expect("sealed fused input derivation");
        let compact_inputs: String = inputs
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        assert!(compact_inputs.contains("source_is_all_i32_fixed(&plan.source)"));
        assert!(compact_inputs.contains("plan.identity.device_memory"));
        assert!(compact_inputs.contains("plan.row_ids"));
        assert!(compact_inputs.contains("expected_commit_seq"));
    }
    #[test]
    fn rollover_coordinates_are_checked_before_allocation() {
        assert_eq!(
            super::checked_rollover_coordinates(7, 11, 13),
            Some((8, 24))
        );
        assert_eq!(super::checked_rollover_coordinates(u32::MAX, 0, 0), None);
        assert_eq!(super::checked_rollover_coordinates(7, usize::MAX, 1), None);
    }
    #[test]
    fn typed_post_wal_paths_cannot_allocate_a_replacement_generation() {
        let mutation = include_str!("mutation.rs");
        let fixed_apply = mutation
            .split("} else if preallocated_fixed_plan {")
            .nth(1)
            .and_then(|section| {
                section
                    .split("} else if !has_text && !batch_has_null {")
                    .next()
            })
            .expect("typed fixed post-WAL apply section");
        assert!(fixed_apply.contains("finish_post_wal"));
        for forbidden in [
            "relational_residency_device_memory",
            "retain_device_memory_",
            "PendingResidentShard::build",
            "fixed_width_desired_capacity",
        ] {
            assert!(
                !fixed_apply.contains(forbidden),
                "typed fixed post-WAL apply must not {forbidden}"
            );
        }

        let rollover = include_str!("rollover.rs");
        let fixed_finish = rollover
            .split("fn finish_post_wal")
            .nth(1)
            .and_then(|section| section.split("impl PendingInPlaceCreatedBy").next())
            .expect("fixed post-WAL finalization");
        for forbidden in [
            "relational_residency_device_memory",
            "retain_device_memory_",
        ] {
            assert!(
                !fixed_finish.contains(forbidden),
                "fixed post-WAL finalization must not {forbidden}"
            );
        }

        let typed_in_place = mutation
            .split("let typed_created_by_region = match &mut source {")
            .nth(1)
            .and_then(|section| section.split("let fused = if").next())
            .expect("typed in-place sidecar handoff");
        assert!(
            typed_in_place.contains("pending.into_region")
                && typed_in_place.contains("get_or_alloc_created_by_region"),
            "mutation must consume and publish the sealed in-place Arc"
        );
        assert!(!typed_in_place.contains("install_preallocated_created_by_region"));
        let pending_in_place = rollover
            .split("impl PendingInPlaceCreatedBy")
            .nth(1)
            .and_then(|section| section.split("impl PendingResidentShard").next())
            .expect("in-place allocation lifecycle");
        for forbidden in [
            "with_shards_mut",
            ".insert_shard(",
            "shard_created_by_memory",
        ] {
            assert!(
                !pending_in_place.contains(forbidden),
                "rollover lifecycle leaf must not publish {forbidden}"
            );
        }
    }

    #[test]
    fn adapter_has_no_second_residency_or_durability_publisher() {
        let source = include_str!("fixed_insert.rs");
        let implementation = source
            .split("#[cfg(test)]\nmod ownership_tests")
            .next()
            .expect("source has an implementation prefix");
        for (prefix, suffix) in [
            ("RelationalResident", "Shard {"),
            ("with_shards_mut", "_for_table("),
            (".insert_", "shard("),
            ("retain_device_", "memory_"),
            (".append_owned_", "chunks("),
            ("write_", "wal"),
            ("append_", "wal"),
        ] {
            let forbidden = format!("{prefix}{suffix}");
            assert!(
                !implementation.contains(&forbidden),
                "fixed INSERT adapter must not own {forbidden}"
            );
        }
        for forbidden in [
            "relational_residency_device_memory",
            "retain_device_memory_",
            "shard_created_by_memory.insert",
            "shard_row_id_memory.insert",
        ] {
            assert!(
                !implementation.contains(forbidden),
                "fixed INSERT adapter must delegate {forbidden} to the allocation/publisher owner"
            );
        }
        assert!(implementation.contains("ResidentAppendSource::DevicePlan"));
        assert_eq!(
            implementation
                .matches("try_append_to_resident_open_shard")
                .count(),
            1,
            "the typed plan must enter mutation through exactly one publisher"
        );
        assert!(implementation.contains("_device_apply"));
        assert!(implementation.contains("budget_allocation"));
        assert!(
            !implementation.contains("pub(crate) fn new"),
            "the move-only plan must have no raw public constructor"
        );
        assert!(
            implementation.contains("row_ids: DeviceInsertRowIds"),
            "the plan must own its row-id input before WAL"
        );
        assert!(
            !implementation.contains("row_ids_present: bool"),
            "a caller-provided row-id presence bit must not cross the WAL/apply boundary"
        );
        let apply = implementation
            .split("fn apply_resident_open_shard_append")
            .nth(1)
            .expect("sealed apply exists")
            .split("\n    /// Internal reservation adapter")
            .next()
            .expect("sealed apply precedes reservation adapters");
        assert!(
            !apply.contains("row_ids:"),
            "apply must consume only row IDs sealed into the opaque plan"
        );
        assert!(
            source.contains("#[cfg(test)]\n    pub(crate) fn synthetic_no_identity"),
            "synthetic no-identity construction must remain test-only"
        );
    }
}
