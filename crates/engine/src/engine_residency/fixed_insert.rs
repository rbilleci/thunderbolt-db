//! Typed device-plan compilation for the accepted INSERT residency route.
//!
//! This module owns no CUDA allocation, device write, descriptor publication, side-map mutation,
//! or WAL. It consumes a sealed source into an opaque plan, then the plan re-enters mutation's one
//! publisher at apply time. A post-WAL apply failure is fatal rather than permission to fall back.

use super::append_source::ResidentAppendSource;
use super::*;
use crate::engine_insert_plan::host_retention::{HostRetentionGeometry, HostRetentionReport};
use crate::engine_insert_plan::IndexedPhysicalMaterializationPermit;
#[cfg(test)]
use crate::typed_insert_batch::TypedInsertBatch;
use crate::typed_insert_batch::{
    PreparedResidentAppendSource, PreparedResidentFixedBoolUpload, PreparedResidentFixedChunk,
    PreparedResidentFixedChunkOwners,
};
use std::sync::{Arc, MutexGuard};

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
    TransactionTerminalUnindexed,
    IndexedInPlaceReservation {
        index_scratch_bytes: u64,
    },
    IndexedFixedRolloverReservation {
        max_index_scratch_bytes: u64,
        allow_s3_index_schema_transition: bool,
    },
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
    pending: Option<super::rollover::PendingFixedResidentShard>,
    uniform_publication: Option<super::rollover::PreparedFixedResidentShardPublication>,
    pub(super) capacity: usize,
    pub(super) new_shard_id: u32,
    pub(super) new_row_start: usize,
    pub(super) capacity_fit_evaluations: u64,
    pub(super) budget_scan_entries: u64,
    pub(super) planned_index_allocation_bytes: u64,
    pub(super) max_index_scratch_bytes: u64,
}

impl PreparedFixedRollover {
    fn pending(&self) -> Option<&super::rollover::PendingFixedResidentShard> {
        self.pending.as_ref().or_else(|| {
            self.uniform_publication
                .as_ref()
                .map(|owner| owner.pending())
        })
    }

    fn prepare_uniform_commit_pre_wal(
        &mut self,
        commit_seq: Index,
    ) -> Result<(), DeviceInsertPlanPrepareError> {
        if self.uniform_publication.is_some() {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let pending = self
            .pending
            .take()
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        self.uniform_publication = Some(
            pending
                .prepare_uniform_commit_pre_wal(commit_seq)
                .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?,
        );
        Ok(())
    }

    pub(super) fn publish_uniform_post_wal(
        self,
    ) -> Result<super::rollover::PendingFixedResidentShard, DeviceInsertPlanApplyError> {
        match (self.pending, self.uniform_publication) {
            (None, Some(publication)) => publication
                .publish_post_wal()
                .map_err(|_| DeviceInsertPlanApplyError::PublisherFailure),
            _ => Err(DeviceInsertPlanApplyError::PlanDrift),
        }
    }
}

pub(super) enum PreparedInPlaceCreatedBy {
    Existing(Arc<CudaResidentDeviceMemory>),
    Reserved(super::rollover::PendingInPlaceCreatedBy),
}

/// Private, unpublished dense allocation set. All three CUDA allocations are created while the
/// plan owns the mutation and budget gates, before WAL. Apply may only upload/write them and hand
/// their Arcs to mutation's existing descriptor publisher.
pub(super) struct PreparedDenseRollover {
    pending: Option<super::rollover::PendingDenseResidentShard>,
    uniform_publication: Option<super::rollover::PreparedDenseResidentShardPublication>,
    pub(super) new_shard_id: u32,
    pub(super) new_row_start: usize,
    pub(super) budget_scan_entries: u64,
    pub(super) planned_index_allocation_bytes: u64,
    pub(super) max_index_scratch_bytes: u64,
}

impl PreparedDenseRollover {
    fn pending(&self) -> Option<&super::rollover::PendingDenseResidentShard> {
        self.pending.as_ref().or_else(|| {
            self.uniform_publication
                .as_ref()
                .map(|owner| owner.pending())
        })
    }

    fn prepare_uniform_commit_pre_wal(
        &mut self,
        commit_seq: Index,
    ) -> Result<(), DeviceInsertPlanPrepareError> {
        if self.uniform_publication.is_some() {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let pending = self
            .pending
            .take()
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        self.uniform_publication = Some(
            pending
                .prepare_uniform_commit_pre_wal(commit_seq)
                .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?,
        );
        Ok(())
    }

    pub(super) fn into_pending_for_nonuniform(
        self,
    ) -> Result<super::rollover::PendingDenseResidentShard, ExecuteError> {
        match (self.pending, self.uniform_publication) {
            (Some(pending), None) => Ok(pending),
            _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed dense rollover publication state drifted".to_string(),
            ))),
        }
    }

    pub(super) fn publish_uniform_post_wal(
        self,
    ) -> Result<super::rollover::PendingDenseResidentShard, DeviceInsertPlanApplyError> {
        match (self.pending, self.uniform_publication) {
            (None, Some(publication)) => publication
                .publish_post_wal()
                .map_err(|_| DeviceInsertPlanApplyError::PublisherFailure),
            _ => Err(DeviceInsertPlanApplyError::PlanDrift),
        }
    }
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
    foreign_keys_prevalidated: bool,
    row_ids: DeviceInsertRowIds,
    resets_existing_rows: bool,
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
    // Plural compilation moves the one guard forward into a later plan. The earlier plan remains
    // in the same owned vector and may apply only while that later owner is still retained by the
    // vector's consuming iterator; this marker records that proven handoff without minting a
    // second guard or treating an unlocked budget snapshot as sufficient.
    budget_reservation_retained_by_transaction: bool,
    // The plan crosses WAL while owning the only locks that can change its descriptor or consume
    // its sealed budget. This is a reservation, not a best-effort budget snapshot: no other
    // normal device publisher/allocation transaction can invalidate this geometry before apply.
    _device_apply: Option<MutexGuard<'a, ()>>,
}

pub(super) struct ResidentFixedRolloverReservationBasis<'a> {
    pub(super) predecessor_shard_id: u32,
    pub(super) predecessor_row_count: usize,
    pub(super) catalog_seq: Index,
    pub(super) incoming_rows: usize,
    pub(super) capacity: usize,
    pub(super) new_shard_id: u32,
    pub(super) new_row_start: usize,
    pub(super) capacity_fit_evaluations: u64,
    pub(super) payload: &'a Arc<CudaResidentDeviceMemory>,
    pub(super) created_by_region: &'a Arc<CudaResidentDeviceMemory>,
    pub(super) row_id_region: Option<&'a Arc<CudaResidentDeviceMemory>>,
    pub(super) bool_layouts: &'a [ResidentDeviceBoolColumnLayout],
    pub(super) int4_stats: &'a [ResidentDeviceInt4ColumnStats],
    pub(super) payload_bytes: u64,
    pub(super) created_by_bytes: u64,
    pub(super) row_id_bytes: u64,
    pub(super) payload_sidecar_allocation_bytes: u64,
    pub(super) payload_sidecar_allocation_count: u64,
    pub(super) planned_index_allocation_bytes: u64,
    pub(super) max_index_scratch_bytes: u64,
}

#[derive(Clone, Copy)]
pub(super) struct ResidentDenseRolloverReservationBasis<'a> {
    pub(super) predecessor_shard_id: u32,
    pub(super) predecessor_row_count: usize,
    pub(super) catalog_seq: Index,
    pub(super) incoming_rows: usize,
    pub(super) capacity: usize,
    pub(super) new_shard_id: u32,
    pub(super) new_row_start: usize,
    pub(super) payload: &'a Arc<CudaResidentDeviceMemory>,
    pub(super) created_by_region: &'a Arc<CudaResidentDeviceMemory>,
    pub(super) row_id_region: Option<&'a Arc<CudaResidentDeviceMemory>>,
    pub(super) bool_layouts: &'a [ResidentDeviceBoolColumnLayout],
    pub(super) text_layouts: &'a [ResidentDeviceTextColumnLayout],
    pub(super) null_layouts: &'a [ResidentDeviceNullBitmapLayout],
    pub(super) int4_stats: &'a [ResidentDeviceInt4ColumnStats],
    pub(super) payload_bytes: u64,
    pub(super) payload_sidecar_allocation_bytes: u64,
    pub(super) payload_sidecar_allocation_count: u64,
    pub(super) planned_index_allocation_bytes: u64,
    pub(super) max_index_scratch_bytes: u64,
}

impl<'a> ResidentOpenShardAppendPlan<'a> {
    pub(crate) fn table_name(&self) -> &str {
        self.source.table_name()
    }

    pub(super) fn source(&self) -> &PreparedResidentAppendSource {
        &self.source
    }

    pub(super) fn row_count(&self) -> usize {
        self.source.row_count()
    }

    pub(super) fn resets_existing_rows(&self) -> bool {
        self.resets_existing_rows
    }

    fn pre_wal_reserved_persistent_bytes(&self) -> u64 {
        match &self.branch {
            PreparedResidentAppendBranch::InPlace(in_place) => in_place
                .pending_created_by
                .as_ref()
                .map_or(0, |pending| pending.allocation_bytes),
            PreparedResidentAppendBranch::FixedRollover(fixed) => fixed
                .pending()
                .map_or(0, |pending| pending.allocation_bytes)
                .checked_add(fixed.planned_index_allocation_bytes)
                .expect("validated fixed rollover reservation bytes fit u64"),
            PreparedResidentAppendBranch::DenseRollover(dense) => dense
                .pending()
                .map_or(0, |pending| pending.allocation_bytes)
                .checked_add(dense.planned_index_allocation_bytes)
                .expect("validated dense rollover reservation bytes fit u64"),
            PreparedResidentAppendBranch::ConsumedDense => 0,
        }
    }

    pub(super) fn take_transaction_terminal_device_apply_guard(
        &mut self,
    ) -> Option<MutexGuard<'a, ()>> {
        self._device_apply.take()
    }

    pub(super) fn take_transaction_terminal_budget_guard(
        &mut self,
    ) -> Option<(MutexGuard<'a, ()>, u64)> {
        let bytes = self.pre_wal_reserved_persistent_bytes();
        let retained = self.budget_allocation.take().map(|guard| (guard, bytes));
        if retained.is_some() {
            self.budget_reservation_retained_by_transaction = true;
        }
        retained
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
                let pending = fixed.pending().ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "materialized fixed rollover lost its private generation".to_string(),
                    ))
                })?;
                append_pending_bool_layout_box(&mut report, &pending.bool_layouts)?;
                append_pending_int4_stats_box(&mut report, &pending.int4_stats)?;
            }
            PreparedResidentAppendBranch::DenseRollover(dense) => {
                let pending = dense.pending().ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "materialized dense rollover lost its private generation".to_string(),
                    )
                })?;
                report.merge(pending.payload.host_retention_report()?)?;
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
        self.catalog_seq == catalog_seq
            && source_matches_table(&self.source, table, self.foreign_keys_prevalidated)
    }

    pub(super) fn identity_matches(&self, open: &RelationalResidentShard, pressured: bool) -> bool {
        self.identity.matches(open, pressured)
    }

    pub(super) fn int4_min_max(&self) -> &[(i32, i32)] {
        &self.int4_min_max
    }

    fn take_row_ids(&mut self) -> Option<Box<[u64]>> {
        self.row_ids.take_exact()
    }

    pub(super) fn holds_budget_reservation(&self) -> bool {
        self.budget_allocation.is_some() || self.budget_reservation_retained_by_transaction
    }

    pub(super) fn indexed_fixed_rollover_host_materialization_scratch(
        &self,
    ) -> Option<HostRetentionGeometry> {
        match &self.branch {
            PreparedResidentAppendBranch::FixedRollover(fixed) => fixed
                .pending()
                .map(|pending| pending.host_materialization_scratch),
            PreparedResidentAppendBranch::InPlace(_)
            | PreparedResidentAppendBranch::DenseRollover(_)
            | PreparedResidentAppendBranch::ConsumedDense => None,
        }
    }

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

    pub(super) fn indexed_fixed_rollover_reservation_basis(
        &self,
    ) -> Option<ResidentFixedRolloverReservationBasis<'_>> {
        let PreparedResidentAppendBranch::FixedRollover(fixed) = &self.branch else {
            return None;
        };
        let pending = fixed.pending()?;
        Some(ResidentFixedRolloverReservationBasis {
            predecessor_shard_id: self.identity.shard_id,
            predecessor_row_count: self.identity.row_count,
            catalog_seq: self.catalog_seq,
            incoming_rows: self.source.row_count(),
            capacity: fixed.capacity,
            new_shard_id: fixed.new_shard_id,
            new_row_start: fixed.new_row_start,
            capacity_fit_evaluations: fixed.capacity_fit_evaluations,
            payload: &pending.device_memory,
            created_by_region: &pending.created_by_region,
            row_id_region: pending.row_id_region.as_ref(),
            bool_layouts: &pending.bool_layouts,
            int4_stats: &pending.int4_stats,
            payload_bytes: pending.payload_bytes,
            created_by_bytes: pending.created_by_bytes,
            row_id_bytes: pending.row_id_bytes,
            payload_sidecar_allocation_bytes: pending.allocation_bytes,
            payload_sidecar_allocation_count: pending.persistent_allocation_count,
            planned_index_allocation_bytes: fixed.planned_index_allocation_bytes,
            max_index_scratch_bytes: fixed.max_index_scratch_bytes,
        })
    }

    pub(super) fn prepare_fixed_rollover_uniform_commit(
        &mut self,
        expected_commit_seq: Index,
    ) -> Result<(), DeviceInsertPlanPrepareError> {
        match &mut self.branch {
            PreparedResidentAppendBranch::FixedRollover(fixed) => {
                fixed.prepare_uniform_commit_pre_wal(expected_commit_seq)
            }
            PreparedResidentAppendBranch::DenseRollover(_)
            | PreparedResidentAppendBranch::InPlace(_) => Ok(()),
            PreparedResidentAppendBranch::ConsumedDense => {
                Err(DeviceInsertPlanPrepareError::UnsupportedShape)
            }
        }
    }

    pub(super) fn publish_fixed_rollover_header(
        mut self,
    ) -> Result<super::rollover::PendingFixedResidentShard, DeviceInsertPlanApplyError> {
        let fixed = self
            .take_fixed_rollover()
            .ok_or(DeviceInsertPlanApplyError::PlanDrift)?;
        fixed.publish_uniform_post_wal()
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
            PreparedResidentAppendBranch::DenseRollover(dense) => {
                dense.pending().map(|pending| pending.payload_bytes)
            }
            _ => None,
        }
    }

    pub(super) fn fixed_rollover_payload_len(&self) -> Option<u64> {
        match &self.branch {
            PreparedResidentAppendBranch::FixedRollover(fixed) => {
                fixed.pending().map(|pending| pending.payload_bytes)
            }
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

    /// Complete the descriptor half of an indexed fused append after its payload, index tail,
    /// device row-count header, and cache coverage have all succeeded. The fused branch writes
    /// the same open allocation as the ordinary append publisher, but it cannot call that
    /// publisher without uploading the payload and extending the index a second time. Consume
    /// this plan here to publish only the already-sealed descriptor metadata through the common
    /// residency owner.
    pub(super) fn publish_indexed_in_place_descriptor_after_fused_apply(
        mut self,
        engine: &Engine,
        expected_commit_seq: Index,
    ) -> Result<(), DeviceInsertPlanApplyError> {
        let table = self.source.table_name().to_string();
        let base_row = self.identity.row_count;
        let capacity = self.identity.capacity;
        let incoming_rows = self.source.row_count();
        let end_row = base_row
            .checked_add(incoming_rows)
            .filter(|end| *end <= capacity)
            .ok_or(DeviceInsertPlanApplyError::PlanDrift)?;
        if incoming_rows == 0
            || !source_is_all_i32_fixed(&self.source)
            || !self.row_ids.exact_len_matches(incoming_rows)
            || self.int4_min_max.len() != self.identity.int4_columns.len()
        {
            return Err(DeviceInsertPlanApplyError::PlanDrift);
        }

        let pressured = engine.router.runtime().snapshot().memory_pressured_gpu_ids;
        let still_current = engine
            .read_state
            .residency
            .shards
            .load()
            .get(&table)
            .and_then(|shards| shards.last())
            .is_some_and(|open| {
                self.identity
                    .matches(open, pressured.contains(&open.gpu_id))
            });
        if !still_current {
            return Err(DeviceInsertPlanApplyError::PlanDrift);
        }

        let created_by_region = match self.take_in_place_created_by(capacity, base_row) {
            Some(PreparedInPlaceCreatedBy::Existing(region)) => {
                let current = engine
                    .read_state
                    .residency
                    .shard_created_by_memory
                    .get(&(table.clone(), self.identity.shard_id));
                if !current
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, &region))
                {
                    return Err(DeviceInsertPlanApplyError::PlanDrift);
                }
                region
            }
            Some(PreparedInPlaceCreatedBy::Reserved(pending)) => {
                let region = pending
                    .into_region(capacity)
                    .ok_or(DeviceInsertPlanApplyError::PlanDrift)?;
                engine
                    .get_or_alloc_created_by_region(
                        &table,
                        self.identity.shard_id,
                        capacity,
                        self.identity.gpu_id,
                        true,
                        Some(region),
                    )
                    .ok_or(DeviceInsertPlanApplyError::PublisherFailure)?
            }
            None => return Err(DeviceInsertPlanApplyError::PlanDrift),
        };

        let appended_bytes = incoming_rows
            .checked_mul(self.identity.int4_columns.len())
            .and_then(|values| values.checked_mul(std::mem::size_of::<i32>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(DeviceInsertPlanApplyError::PlanDrift)?;
        let mut published = false;
        engine
            .read_state
            .residency
            .with_shards_mut_for_table(&table, |shards| {
                let Some(open) = shards.get_mut(&table).and_then(|shards| shards.last_mut()) else {
                    return;
                };
                let payload_matches =
                    same_optional_device_region(&open.device_memory, &self.identity.device_memory);
                let created_by_matches = open
                    .created_by_region
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, &created_by_region));
                let row_ids_match =
                    same_optional_device_region(&open.row_id_region, &self.identity.row_id_region);
                if open.shard_id != self.identity.shard_id
                    || open.capacity != capacity
                    || open.row_count != base_row
                    || open.row_start != self.identity.row_start
                    || open.gpu_id != self.identity.gpu_id
                    || !payload_matches
                    || !created_by_matches
                    || !row_ids_match
                    || open.resident_device_int4_column_stats.len() != self.int4_min_max.len()
                {
                    return;
                }
                open.row_count = end_row;
                open.max_created_by = open.max_created_by.max(expected_commit_seq);
                open.resident_bytes = open.resident_bytes.saturating_add(appended_bytes);
                for (stat, (lo, hi)) in open
                    .resident_device_int4_column_stats
                    .iter_mut()
                    .zip(self.int4_min_max.iter())
                {
                    stat.min = stat.min.min(*lo);
                    stat.max = stat.max.max(*hi);
                }
                published = true;
            });
        if !published {
            return Err(DeviceInsertPlanApplyError::PublisherFailure);
        }
        engine
            .read_state
            .residency
            .open_shard_append_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
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

    pub(super) fn indexed_dense_rollover_reservation_basis(
        &self,
    ) -> Option<ResidentDenseRolloverReservationBasis<'_>> {
        let PreparedResidentAppendBranch::DenseRollover(dense) = &self.branch else {
            return None;
        };
        let pending = dense.pending()?;
        Some(ResidentDenseRolloverReservationBasis {
            predecessor_shard_id: self.identity.shard_id,
            predecessor_row_count: self.identity.row_count,
            catalog_seq: self.catalog_seq,
            incoming_rows: self.source.row_count(),
            capacity: self.source.row_count(),
            new_shard_id: dense.new_shard_id,
            new_row_start: dense.new_row_start,
            payload: &pending.device_memory,
            created_by_region: &pending.created_by_region,
            row_id_region: pending.row_id_region.as_ref(),
            bool_layouts: pending.payload.bool_layouts(),
            text_layouts: pending.payload.text_layouts(),
            null_layouts: pending.payload.null_layouts(),
            int4_stats: pending.payload.int4_stats(),
            payload_bytes: pending.payload_bytes,
            payload_sidecar_allocation_bytes: pending.allocation_bytes,
            payload_sidecar_allocation_count: pending.persistent_allocation_count,
            planned_index_allocation_bytes: dense.planned_index_allocation_bytes,
            max_index_scratch_bytes: dense.max_index_scratch_bytes,
        })
    }

    pub(super) fn prepare_indexed_dense_rollover_uniform_commit(
        &mut self,
        expected_commit_seq: Index,
    ) -> Result<(), DeviceInsertPlanPrepareError> {
        let PreparedResidentAppendBranch::DenseRollover(dense) = &mut self.branch else {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        };
        dense.prepare_uniform_commit_pre_wal(expected_commit_seq)
    }

    pub(super) fn publish_indexed_dense_rollover_header(
        mut self,
    ) -> Result<super::rollover::PendingDenseResidentShard, DeviceInsertPlanApplyError> {
        let dense = self
            .take_dense_rollover()
            .ok_or(DeviceInsertPlanApplyError::PlanDrift)?;
        dense.publish_uniform_post_wal()
    }
}

/// The only post-compile token that can reach typed device append. Its physical variant is
/// deliberately private to residency: the concurrent wave can own this opaque plan, but cannot
/// inspect, replace, or construct a resident append source.
pub(crate) struct DeviceInsertPlan<'a>(DeviceInsertPlanKind<'a>);

/// Opaque pre-WAL handoff from one indexed rollover plan to the next plan in the same transaction.
/// It contains only the first plan's already-built successor roots and cannot apply or publish.
pub(crate) struct DeviceInsertPlanManifestPredecessor(
    super::prepared_table_index_manifest::PreparedIndexedRolloverManifestPredecessor,
);

#[allow(clippy::large_enum_variant)] // opaque plan variants preserve move-only GPU resource ownership
enum DeviceInsertPlanKind<'a> {
    ResidentOpenShardAppend(ResidentOpenShardAppendPlan<'a>),
    /// The first physical generation for a relation created by the transaction's S3 catalog
    /// composition.  It is still the one codec-5 device-plan lifecycle: the table is absent
    /// from the public catalog until the common publication step, so this plan retains the
    /// exact private dense generation rather than manufacturing a public empty predecessor.
    TransactionCreatedDenseTable(TransactionCreatedDenseTablePlan<'a>),
    IndexedInPlace(super::indexed_reservation::PreparedIndexedPhysicalReservation<'a>),
}

/// A transaction-created table has no public descriptor to append to before its catalog
/// composition commits.  This is the dense first-row counterpart of the ordinary preallocated
/// rollover: all bytes and the uniform header capability are sealed before WAL; post-WAL only
/// resolves that capability and installs the resulting descriptor through the common residency
/// maps.  It deliberately owns neither catalog publication nor a second row representation.
pub(super) struct TransactionCreatedDenseTablePlan<'a> {
    pub(super) table: RelationalTable,
    pub(super) expected_commit_seq: Index,
    pub(super) row_count: usize,
    pub(super) row_ids: DeviceInsertRowIds,
    pub(super) dense: PreparedDenseRollover,
    // The generic plural coordinator keeps the one budget guard while it prepares every table.
    // This exact private dense allocation must remain in that coordinator's running pre-WAL
    // charge when a later table receives the guard; otherwise two transaction-created tables
    // could each fit the same remaining-budget snapshot independently.
    pub(super) reserved_bytes: u64,
    // The same transaction-wide named-index lifecycle follows a private first-row table through
    // the dense reservation and back to the generic codec-5 finalizer.  The finalizer remains
    // the sole catalog/index publication owner after the durable cut.
    pub(super) named_index_lifecycle:
        Option<crate::engine_state::TransactionNamedIndexPublicationGuard<'a>>,
    pub(super) budget_allocation: Option<MutexGuard<'a, ()>>,
    pub(super) _device_apply: Option<MutexGuard<'a, ()>>,
}

impl<'a> DeviceInsertPlan<'a> {
    pub(crate) fn indexed_rollover_manifest_successor_predecessor(
        &self,
    ) -> Option<DeviceInsertPlanManifestPredecessor> {
        match &self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(_) => None,
            DeviceInsertPlanKind::TransactionCreatedDenseTable(_) => None,
            DeviceInsertPlanKind::IndexedInPlace(plan) => plan
                .rollover_manifest_successor_predecessor()
                .map(DeviceInsertPlanManifestPredecessor),
        }
    }

    /// Move the sole common lifecycle guard back to the generic transaction finalizer before it
    /// enters final publication.  The indexed plan owned it during pre-WAL reservation so no
    /// detached or second named-index lifecycle can race cache retirement.
    pub(crate) fn take_named_index_publication_guard(
        &mut self,
    ) -> Option<crate::engine_state::TransactionNamedIndexPublicationGuard<'a>> {
        match &mut self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(_) => None,
            DeviceInsertPlanKind::TransactionCreatedDenseTable(plan) => {
                plan.named_index_lifecycle.take()
            }
            DeviceInsertPlanKind::IndexedInPlace(plan) => plan.take_named_index_publication_guard(),
        }
    }

    /// Move the one global mutation gate between unindexed table reservations while the generic
    /// transaction owner composes a plural plan. No plan can apply through this handoff, and the
    /// final plan retains the guard across WAL for the whole vector.
    pub(crate) fn take_transaction_terminal_unindexed_device_apply_guard(
        &mut self,
    ) -> Option<MutexGuard<'a, ()>> {
        match &mut self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan) => plan._device_apply.take(),
            DeviceInsertPlanKind::TransactionCreatedDenseTable(plan) => plan._device_apply.take(),
            DeviceInsertPlanKind::IndexedInPlace(plan) => {
                plan.take_transaction_terminal_device_apply_guard()
            }
        }
    }

    pub(crate) fn take_transaction_terminal_unindexed_budget_guard(
        &mut self,
    ) -> Option<(MutexGuard<'a, ()>, u64)> {
        match &mut self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan) => {
                plan.take_transaction_terminal_budget_guard()
            }
            DeviceInsertPlanKind::TransactionCreatedDenseTable(plan) => plan
                .budget_allocation
                .take()
                .map(|guard| (guard, plan.reserved_bytes)),
            DeviceInsertPlanKind::IndexedInPlace(plan) => {
                plan.take_transaction_terminal_budget_guard()
            }
        }
    }

    #[cfg(feature = "probe-timing")]
    pub(crate) fn probe_retains_exclusive_budget_guard(&self) -> bool {
        match &self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan) => plan.budget_allocation.is_some(),
            DeviceInsertPlanKind::TransactionCreatedDenseTable(plan) => {
                plan.budget_allocation.is_some()
            }
            DeviceInsertPlanKind::IndexedInPlace(_) => true,
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
            DeviceInsertPlanKind::TransactionCreatedDenseTable(plan) => {
                engine.apply_transaction_created_dense_table_generation(plan, created_by)
            }
            DeviceInsertPlanKind::IndexedInPlace(plan) => {
                plan.apply_after_transaction_wal_claim(engine, created_by)
            }
        }
    }

    /// Transaction-terminal sibling of the typed wave handoff. Its permit is still opaque and
    /// may only be issued after the canonical transaction outcome is durable; this method owns
    /// no WAL, status, allocator, or publication authority.
    pub(crate) fn apply_after_transaction_wal_claim(
        self,
        engine: &Engine,
        permit: crate::engine_dml_concurrent::TypedInsertPostWalApplyPermit,
    ) -> Result<(), DeviceInsertPlanApplyError> {
        self.apply_after_typed_wal_claim(engine, permit)
    }
}

impl Engine {
    /// Exercise the production transaction-terminal physical compiler from a sealed batch in
    /// low-level tests. This fixture owns no selector: it only performs the same batch-to-source
    /// move that the transaction overlay completes before entering the canonical compiler.
    #[cfg(test)]
    pub(crate) fn compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test<'a>(
        &'a self,
        batch: TypedInsertBatch,
        row_ids: DeviceInsertRowIds,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        self.compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test_at_commit(
            batch,
            row_ids,
            self.committed_seq(),
        )
    }

    /// Test-only physical fixture for a sealed post-WAL permit. Production callers receive the
    /// exact commit sequence from the canonical transaction finalizer before WAL.
    #[cfg(test)]
    pub(crate) fn compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test_at_commit<
        'a,
    >(
        &'a self,
        batch: TypedInsertBatch,
        row_ids: DeviceInsertRowIds,
        expected_commit_seq: Index,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        let source = batch
            .into_codec5_resident_append_source_for_test(&self.catalog_snapshot())
            .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
        let mut plan = self.prepare_resident_open_shard_append_core(
            source,
            row_ids,
            ResidentOpenShardAppendPreparationMode::TransactionTerminalUnindexed,
            None,
            None,
            0,
            false,
            None,
        )?;
        plan.prepare_fixed_rollover_uniform_commit(expected_commit_seq)?;
        Ok(DeviceInsertPlan(
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan),
        ))
    }

    pub(crate) fn compile_transaction_terminal_typed_insert_device_plan<'a>(
        &'a self,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        expected_commit_seq: Index,
        resets_existing_rows: bool,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        let mut plan = self.prepare_resident_open_shard_append_core(
            source,
            row_ids,
            ResidentOpenShardAppendPreparationMode::TransactionTerminalUnindexed,
            None,
            None,
            0,
            resets_existing_rows,
            None,
        )?;
        plan.prepare_fixed_rollover_uniform_commit(expected_commit_seq)?;
        Ok(DeviceInsertPlan(
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan),
        ))
    }

    /// Compile the first rowset for a relation introduced by the transaction's catalog
    /// composition.  The public catalog and shard map must both still be absent here: publishing
    /// either before the canonical WAL/status claim would create a second write authority.
    pub(crate) fn compile_transaction_created_table_typed_insert_device_plan<'a>(
        &'a self,
        table: &RelationalTable,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        expected_commit_seq: Index,
        named_index_lifecycle: Option<
            crate::engine_state::TransactionNamedIndexPublicationGuard<'a>,
        >,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        self.compile_transaction_created_table_typed_insert_device_plan_core(
            table,
            source,
            row_ids,
            expected_commit_seq,
            named_index_lifecycle,
            None,
            None,
            0,
        )
    }

    /// Plural codec-5 preparation moves its one mutation/budget reservation through each table
    /// plan.  A transaction-created table participates in that same handoff: it still owns its
    /// private first-generation allocation, but never manufactures a second writer or publishes
    /// before the common terminal.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_transaction_created_table_typed_insert_device_plan_with_gate<'a>(
        &'a self,
        table: &RelationalTable,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        expected_commit_seq: Index,
        named_index_lifecycle: Option<
            crate::engine_state::TransactionNamedIndexPublicationGuard<'a>,
        >,
        device_apply: MutexGuard<'a, ()>,
        budget_allocation: Option<MutexGuard<'a, ()>>,
        prior_reserved_bytes: u64,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        self.compile_transaction_created_table_typed_insert_device_plan_core(
            table,
            source,
            row_ids,
            expected_commit_seq,
            named_index_lifecycle,
            Some(device_apply),
            budget_allocation,
            prior_reserved_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_transaction_created_table_typed_insert_device_plan_core<'a>(
        &'a self,
        table: &RelationalTable,
        mut source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        expected_commit_seq: Index,
        named_index_lifecycle: Option<
            crate::engine_state::TransactionNamedIndexPublicationGuard<'a>,
        >,
        preheld_device_apply: Option<MutexGuard<'a, ()>>,
        preheld_budget_allocation: Option<MutexGuard<'a, ()>>,
        prior_reserved_bytes: u64,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        let nonempty = source.row_count() != 0;
        let source_matches = source_matches_indexed_in_place_reservation(&source, table);
        let exact_row_ids = row_ids.is_exact() && row_ids.exact_len_matches(source.row_count());
        let catalog_absent = !self
            .catalog_snapshot()
            .relational_catalog
            .contains_key(source.table_name());
        let shards_absent = !self
            .read_state
            .residency
            .shards
            .load()
            .contains_key(source.table_name());
        if !(nonempty
            && source_matches
            && exact_row_ids
            && catalog_absent
            && shards_absent
            && (table.indexes.is_empty() == named_index_lifecycle.is_none()))
        {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }

        // A transaction-created relation has no prior fixed-width shard. Keep it on the existing
        // dense first-generation branch even when its immutable source is NULL-free/fixed-width;
        // manufacturing a public empty predecessor here would split the codec-5 lifecycle.
        source.require_dense_first_generation();

        let device_apply = preheld_device_apply.unwrap_or_else(|| {
            self.read_state
                .residency
                .mutation_gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        let budget_allocation = preheld_budget_allocation.unwrap_or_else(|| {
            self.read_state
                .residency
                .budget_allocation_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        let gpu_id = self.planner.default_gpu_id();
        let payload = source
            .checked_dense_payload(table)
            .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
        let rows = source.row_count();
        let sidecar_bytes = u64::try_from(rows)
            .ok()
            .and_then(|rows| rows.checked_mul(std::mem::size_of::<u64>() as u64))
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        let row_id_payload = row_ids.pre_wal_payload();
        if row_id_payload
            .as_ref()
            .is_none_or(|payload| u64::try_from(payload.len()).ok() != Some(sidecar_bytes))
        {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let requested_bytes = payload
            .device_payload_len()
            .and_then(|bytes| bytes.checked_add(sidecar_bytes))
            .and_then(|bytes| bytes.checked_add(sidecar_bytes))
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        let (resident_bytes, budget_scan_entries) =
            self.relational_resident_bytes_and_entries_for_gpu(gpu_id);
        let remaining_budget = match self.relational_residency_budget_bytes(gpu_id) {
            Some(budget) => match budget
                .checked_sub(resident_bytes)
                .and_then(|remaining| remaining.checked_sub(prior_reserved_bytes))
            {
                Some(remaining) => Some(remaining),
                None => {
                    self.read_state
                        .residency
                        .rollover_budget_declines
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                }
            },
            None => None,
        };
        if remaining_budget.is_some_and(|remaining| requested_bytes > remaining) {
            self.read_state
                .residency
                .rollover_budget_declines
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
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
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let mut dense = PreparedDenseRollover {
            pending: Some(pending),
            uniform_publication: None,
            new_shard_id: 0,
            new_row_start: 0,
            budget_scan_entries,
            planned_index_allocation_bytes: 0,
            max_index_scratch_bytes: 0,
        };
        dense.prepare_uniform_commit_pre_wal(expected_commit_seq)?;
        let reserved_bytes = dense
            .pending()
            .map(|pending| pending.allocation_bytes)
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        Ok(DeviceInsertPlan(
            DeviceInsertPlanKind::TransactionCreatedDenseTable(TransactionCreatedDenseTablePlan {
                table: table.clone(),
                expected_commit_seq,
                row_count: rows,
                row_ids,
                dense,
                reserved_bytes,
                named_index_lifecycle,
                budget_allocation: Some(budget_allocation),
                _device_apply: Some(device_apply),
            }),
        ))
    }

    #[allow(clippy::too_many_arguments)] // One move-only DeviceInsertPlan consumes every pre-WAL ownership input.
    pub(crate) fn compile_transaction_terminal_typed_insert_device_plan_with_gate<'a>(
        &'a self,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        expected_commit_seq: Index,
        resets_existing_rows: bool,
        device_apply: MutexGuard<'a, ()>,
        budget_allocation: Option<MutexGuard<'a, ()>>,
        prior_reserved_bytes: u64,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        let mut plan = self.prepare_resident_open_shard_append_core(
            source,
            row_ids,
            ResidentOpenShardAppendPreparationMode::TransactionTerminalUnindexed,
            Some(device_apply),
            budget_allocation,
            prior_reserved_bytes,
            resets_existing_rows,
            None,
        )?;
        plan.prepare_fixed_rollover_uniform_commit(expected_commit_seq)?;
        Ok(DeviceInsertPlan(
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan),
        ))
    }

    /// Predict whether this source needs the exclusive allocation budget. Plural preparation
    /// compiles an indexed rollover first because its prebuilt global roots must publish before
    /// table-local in-place successors; the existing budget guard is then handed through every
    /// remaining plan rather than recursively acquiring the non-reentrant lock.
    pub(crate) fn transaction_terminal_typed_insert_requires_rollover(
        &self,
        source: &PreparedResidentAppendSource,
        resets_existing_rows: bool,
    ) -> bool {
        resets_existing_rows
            || source.requires_dense_rollover()
            || !source_is_all_i32_fixed(source)
            || !self
                .read_state
                .residency
                .shards
                .load()
                .get(source.table_name())
                .and_then(|shards| shards.last())
                .and_then(|open| {
                    open.row_count
                        .checked_add(source.row_count())
                        .map(|end| end <= open.capacity)
                })
                .unwrap_or(false)
    }

    /// Prepare the one production-reachable indexed physical branch. It returns the same opaque
    /// `DeviceInsertPlan` consumed by the generic finalizer/replay owner; UNIQUE/PRIMARY verdicts
    /// have already closed before this physical reservation and create no second terminal.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_transaction_terminal_indexed_typed_insert_device_plan<'a>(
        &'a self,
        table: &RelationalTable,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        named_index_lifecycle: crate::engine_state::TransactionNamedIndexPublicationGuard<'a>,
        expected_commit_seq: Index,
        resets_existing_rows: bool,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        self.compile_transaction_terminal_indexed_typed_insert_device_plan_core(
            table,
            source,
            row_ids,
            named_index_lifecycle,
            expected_commit_seq,
            resets_existing_rows,
            None,
            None,
            0,
            None,
            None,
            &[],
            &[],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_transaction_terminal_indexed_typed_insert_device_plan_with_gate<'a>(
        &'a self,
        table: &RelationalTable,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        named_index_lifecycle: crate::engine_state::TransactionNamedIndexPublicationGuard<'a>,
        expected_commit_seq: Index,
        resets_existing_rows: bool,
        device_apply: MutexGuard<'a, ()>,
        budget_allocation: Option<MutexGuard<'a, ()>>,
        prior_reserved_bytes: u64,
        manifest_predecessor: Option<DeviceInsertPlanManifestPredecessor>,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        self.compile_transaction_terminal_indexed_typed_insert_device_plan_core(
            table,
            source,
            row_ids,
            named_index_lifecycle,
            expected_commit_seq,
            resets_existing_rows,
            Some(device_apply),
            budget_allocation,
            prior_reserved_bytes,
            manifest_predecessor,
            None,
            &[],
            &[],
        )
    }

    /// S3-created named indexes retain the public catalog predecessor through pre-WAL
    /// preparation.  The composed table is still the sole typed source/codec-5 successor;
    /// this merely prevents a private index from masquerading as a publicly enrolled one.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_transaction_terminal_s3_created_index_typed_insert_device_plan_with_gate<
        'a,
    >(
        &'a self,
        table: &RelationalTable,
        public_table: &RelationalTable,
        created_index_ids: &[u64],
        retired_index_ids: &[u64],
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        named_index_lifecycle: crate::engine_state::TransactionNamedIndexPublicationGuard<'a>,
        expected_commit_seq: Index,
        device_apply: MutexGuard<'a, ()>,
        budget_allocation: Option<MutexGuard<'a, ()>>,
        prior_reserved_bytes: u64,
        manifest_predecessor: Option<DeviceInsertPlanManifestPredecessor>,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        self.compile_transaction_terminal_indexed_typed_insert_device_plan_core(
            table,
            source,
            row_ids,
            named_index_lifecycle,
            expected_commit_seq,
            false,
            Some(device_apply),
            budget_allocation,
            prior_reserved_bytes,
            manifest_predecessor,
            Some(public_table),
            created_index_ids,
            retired_index_ids,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_transaction_terminal_s3_created_index_typed_insert_device_plan<'a>(
        &'a self,
        table: &RelationalTable,
        public_table: &RelationalTable,
        created_index_ids: &[u64],
        retired_index_ids: &[u64],
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        named_index_lifecycle: crate::engine_state::TransactionNamedIndexPublicationGuard<'a>,
        expected_commit_seq: Index,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        self.compile_transaction_terminal_indexed_typed_insert_device_plan_core(
            table,
            source,
            row_ids,
            named_index_lifecycle,
            expected_commit_seq,
            false,
            None,
            None,
            0,
            None,
            Some(public_table),
            created_index_ids,
            retired_index_ids,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_transaction_terminal_indexed_typed_insert_device_plan_core<'a>(
        &'a self,
        table: &RelationalTable,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        named_index_lifecycle: crate::engine_state::TransactionNamedIndexPublicationGuard<'a>,
        expected_commit_seq: Index,
        resets_existing_rows: bool,
        preheld_device_apply: Option<MutexGuard<'a, ()>>,
        preheld_budget_allocation: Option<MutexGuard<'a, ()>>,
        prior_reserved_bytes: u64,
        manifest_predecessor: Option<DeviceInsertPlanManifestPredecessor>,
        public_table: Option<&RelationalTable>,
        created_index_ids: &[u64],
        retired_index_ids: &[u64],
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        let current_catalog = self.catalog_snapshot();
        let predecessor_boundary = current_catalog.commit_seq;
        let public_table = public_table.unwrap_or(table);
        let created_indexes_are_exact =
            created_index_ids.iter().enumerate().all(|(ordinal, id)| {
                *id != 0
                    && (ordinal == 0 || created_index_ids[ordinal - 1] < *id)
                    && table
                        .indexes
                        .iter()
                        .any(|index| u64::from(index.oid) == *id)
                    && !public_table
                        .indexes
                        .iter()
                        .any(|index| u64::from(index.oid) == *id)
            });
        let retired_indexes_are_exact =
            retired_index_ids.iter().enumerate().all(|(ordinal, id)| {
                *id != 0
                    && (ordinal == 0 || retired_index_ids[ordinal - 1] < *id)
                    && public_table
                        .indexes
                        .iter()
                        .any(|index| u64::from(index.oid) == *id)
                    && !table
                        .indexes
                        .iter()
                        .any(|index| u64::from(index.oid) == *id)
            });
        let has_s3_index_transition =
            !created_index_ids.is_empty() || !retired_index_ids.is_empty();
        if current_catalog.relational_catalog.get(&table.name) != Some(public_table)
            || expected_commit_seq <= predecessor_boundary
            || !created_indexes_are_exact
            || !retired_indexes_are_exact
            || created_index_ids
                .iter()
                .any(|id| retired_index_ids.binary_search(id).is_ok())
            || (public_table == table) != !has_s3_index_transition
        {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let key_proof =
            crate::engine_insert_plan::batch_key_constraints::seal_current_index_witness(
                table,
                predecessor_boundary,
            )
            .map_err(|_error| {
                #[cfg(feature = "probe-timing")]
                eprintln!(
                    "[probe] codec5_indexed_plan decline stage=key_witness table={} error={_error:?}",
                    table.name
                );
                DeviceInsertPlanPrepareError::UnsupportedShape
            })?;
        let mutation_gate = preheld_device_apply.unwrap_or_else(|| {
            self.read_state
                .residency
                .mutation_gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        let validation = if !has_s3_index_transition {
            crate::engine_insert_plan::resident_key_constraints::validate_current_index_in_place_generation(
                self,
                table,
                &current_catalog,
                predecessor_boundary,
                &mutation_gate,
            )
        } else {
            crate::engine_insert_plan::resident_key_constraints::validate_s3_created_index_generation(
                self,
                public_table,
                table,
                &current_catalog,
                predecessor_boundary,
                &mutation_gate,
            )
        }
        .map_err(|_error| {
            #[cfg(feature = "probe-timing")]
            eprintln!(
                "[probe] codec5_indexed_plan decline stage=resident_validation table={} error={_error:?}",
                table.name
            );
            DeviceInsertPlanPrepareError::UnsupportedShape
        })?;
        let dense_rollover = source.requires_dense_rollover();
        let fits_in_place = !resets_existing_rows
            && !dense_rollover
            && source_is_all_i32_fixed(&source)
            && self
                .read_state
                .residency
                .shards
                .load()
                .get(&table.name)
                .and_then(|shards| shards.last())
                .and_then(|open| {
                    open.row_count
                        .checked_add(source.row_count())
                        .map(|end| (open, end))
                })
                .is_some_and(|(open, end)| {
                    open.resident_device_null_columns.is_empty() && end <= open.capacity
                });
        if manifest_predecessor.is_some() && fits_in_place {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        if has_s3_index_transition && (!dense_rollover || resets_existing_rows) {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let manifest_predecessor = manifest_predecessor.map(|predecessor| predecessor.0);
        let permit = crate::engine_insert_plan::
            issue_codec5_terminal_indexed_physical_materialization_permit();
        let reservation = if dense_rollover {
            super::index_rollover::materialize_dense(
                self,
                table,
                has_s3_index_transition.then_some(public_table),
                created_index_ids,
                retired_index_ids,
                predecessor_boundary,
                source,
                row_ids,
                key_proof,
                validation,
                mutation_gate,
                named_index_lifecycle,
                expected_commit_seq,
                permit,
                preheld_budget_allocation,
                prior_reserved_bytes,
                manifest_predecessor,
                resets_existing_rows,
            )
            .map_err(|error| {
                #[cfg(feature = "probe-timing")]
                eprintln!(
                    "[probe] codec5_indexed_plan decline stage=dense_rollover table={} error={error:?}",
                    table.name
                );
                error
            })
            .map(super::indexed_reservation::PreparedIndexedPhysicalReservation::dense_rollover)
        } else if fits_in_place {
            let preview = super::index_delta_preview::prepare_from_resident_source(
                self,
                table,
                predecessor_boundary,
                source,
                row_ids,
                key_proof,
                validation,
                mutation_gate,
                named_index_lifecycle,
            )
            .map_err(|_error| {
                #[cfg(feature = "probe-timing")]
                eprintln!(
                    "[probe] codec5_indexed_plan decline stage=in_place_preview table={} error={_error:?}",
                    table.name
                );
                DeviceInsertPlanPrepareError::UnsupportedShape
            })?;
            super::index_delta::prepare(
                self,
                table,
                preview,
                expected_commit_seq,
                permit,
                preheld_budget_allocation,
                prior_reserved_bytes,
            )
                .map_err(|error| {
                    #[cfg(feature = "probe-timing")]
                    eprintln!(
                        "[probe] codec5_indexed_plan decline stage=in_place_materialize table={} error={error:?}",
                        table.name
                    );
                    error
                })
                .map(super::indexed_reservation::PreparedIndexedPhysicalReservation::in_place)
        } else {
            let preview = super::index_rollover::prepare_fixed_rollover_preview(
                self,
                table,
                predecessor_boundary,
                source,
                row_ids,
                key_proof,
                validation,
                mutation_gate,
                named_index_lifecycle,
                resets_existing_rows,
            )
            .map_err(|_error| {
                #[cfg(feature = "probe-timing")]
                eprintln!(
                    "[probe] codec5_indexed_plan decline stage=fixed_rollover_preview table={} error={_error:?}",
                    table.name
                );
                DeviceInsertPlanPrepareError::UnsupportedShape
            })?;
            super::index_rollover::materialize(
                self,
                table,
                preview,
                expected_commit_seq,
                permit,
                preheld_budget_allocation,
                prior_reserved_bytes,
                manifest_predecessor,
            )
                .map_err(|error| {
                    #[cfg(feature = "probe-timing")]
                    eprintln!(
                        "[probe] codec5_indexed_plan decline stage=fixed_rollover_materialize table={} error={error:?}",
                        table.name
                    );
                    error
                })
                .map(super::indexed_reservation::PreparedIndexedPhysicalReservation::fixed_rollover)
        }
        .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
        Ok(DeviceInsertPlan(DeviceInsertPlanKind::IndexedInPlace(
            reservation,
        )))
    }

    #[allow(clippy::too_many_arguments)] // each argument names a distinct lifecycle guard or invariant
    fn prepare_resident_open_shard_append_core<'a>(
        &'a self,
        mut source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        mode: ResidentOpenShardAppendPreparationMode,
        preheld_device_apply: Option<MutexGuard<'a, ()>>,
        mut preheld_budget_allocation: Option<MutexGuard<'a, ()>>,
        prior_reserved_bytes: u64,
        resets_existing_rows: bool,
        indexed_final_table: Option<&RelationalTable>,
    ) -> Result<ResidentOpenShardAppendPlan<'a>, DeviceInsertPlanPrepareError> {
        let unindexed = matches!(
            mode,
            ResidentOpenShardAppendPreparationMode::TransactionTerminalUnindexed
        );
        let indexed_in_place_proof = matches!(
            mode,
            ResidentOpenShardAppendPreparationMode::IndexedInPlaceReservation { .. }
        );
        let indexed_fixed_rollover_proof = matches!(
            mode,
            ResidentOpenShardAppendPreparationMode::IndexedFixedRolloverReservation { .. }
        );
        let allow_s3_index_schema_transition = matches!(
            mode,
            ResidentOpenShardAppendPreparationMode::IndexedFixedRolloverReservation {
                allow_s3_index_schema_transition: true,
                ..
            }
        );
        let apply_leader =
            crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(std::cell::Cell::get);
        // A lane leader already owns this lock, but the current intentional general-path policy
        // does not export a cross-WAL plan after its leader scope ends. It declines before WAL;
        // ordinary callers retain the lock in the returned move-only plan, closing
        // descriptor/generation races through apply.
        if unindexed && apply_leader {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let device_apply = match (mode, preheld_device_apply) {
            (ResidentOpenShardAppendPreparationMode::TransactionTerminalUnindexed, None) => Some(
                self.read_state
                    .residency
                    .mutation_gate
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ),
            (ResidentOpenShardAppendPreparationMode::TransactionTerminalUnindexed, Some(held)) => {
                Some(held)
            }
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
        let index_table = indexed_final_table.unwrap_or(table);
        let catalog_matches = if unindexed {
            source.prepared_catalog_seq() != 0
                && source.prepared_catalog_seq() <= catalog.commit_seq
        } else {
            source.prepared_catalog_seq() <= catalog.commit_seq
        };
        let source_matches = if unindexed {
            source_matches_table(&source, table, true)
        } else {
            source_matches_indexed_in_place_reservation(&source, index_table)
                || (allow_s3_index_schema_transition
                    && source_matches_s3_created_index_reservation(&source, index_table))
        };
        #[cfg(feature = "probe-timing")]
        if source.row_count() == 0 || !catalog_matches || !source_matches {
            eprintln!(
                "[probe] typed_insert_plan_decline stage=source table={} rows={} catalog_matches={} source_matches={} source_schema={:02x?} catalog_schema={:02x?}",
                source.table_name(),
                source.row_count(),
                catalog_matches,
                source_matches,
                source.schema_digest(),
                crate::engine_transaction_reset::table_schema_digest(table).unwrap_or([0; 32]),
            );
        }
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
            } else if indexed_fixed_rollover_proof && !dense_rollover {
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
            && (resets_existing_rows
                || dense_rollover
                || bootstrap_sentinel
                || !identity.supports_in_place_append()
                || identity
                    .row_count
                    .checked_add(k)
                    .is_none_or(|end| end > identity.capacity))
        {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        if indexed_fixed_rollover_proof
            && ((!dense_rollover && bootstrap_sentinel)
                || index_table.indexes.is_empty()
                || (!resets_existing_rows
                    && !dense_rollover
                    && source_is_all_i32_fixed(&source)
                    && identity.supports_in_place_append()
                    && identity
                        .row_count
                        .checked_add(k)
                        .is_none_or(|end| end <= identity.capacity)))
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
            let (named_indexes_required, max_index_scratch_bytes) = match mode {
                ResidentOpenShardAppendPreparationMode::IndexedFixedRolloverReservation {
                    max_index_scratch_bytes,
                    ..
                } => (true, max_index_scratch_bytes),
                ResidentOpenShardAppendPreparationMode::TransactionTerminalUnindexed => (false, 0),
                ResidentOpenShardAppendPreparationMode::IndexedInPlaceReservation { .. } => {
                    return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                }
            };
            let payload = source
                .checked_dense_payload(index_table)
                .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let sidecar_bytes = u64::try_from(k)
                .ok()
                .and_then(|rows| rows.checked_mul(std::mem::size_of::<u64>() as u64))
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let row_id_bytes = if row_ids_present { sidecar_bytes } else { 0 };
            let payload_bytes = payload
                .device_payload_len()
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let planned_index_allocation_bytes = if named_indexes_required {
                super::estimated_named_index_bytes_for_shard(index_table, k, k)
                    .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?
            } else {
                0
            };
            let requested_bytes = payload_bytes
                .checked_add(sidecar_bytes)
                .and_then(|bytes| bytes.checked_add(row_id_bytes))
                .and_then(|bytes| bytes.checked_add(planned_index_allocation_bytes))
                .and_then(|bytes| bytes.checked_add(max_index_scratch_bytes))
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let row_id_payload = row_ids.pre_wal_payload();
            if row_ids_present != row_id_payload.is_some() {
                return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
            }
            let (new_shard_id, new_row_start) = if resets_existing_rows {
                identity
                    .shard_id
                    .checked_add(1)
                    .map(|shard_id| (shard_id, 0))
            } else {
                checked_rollover_coordinates(
                    identity.shard_id,
                    identity.row_start,
                    identity.row_count,
                )
            }
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let budget_allocation = preheld_budget_allocation.take().unwrap_or_else(|| {
                self.read_state
                    .residency
                    .budget_allocation_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            });
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
                Some(budget) => match budget
                    .checked_sub(resident_bytes)
                    .and_then(|remaining| remaining.checked_sub(prior_reserved_bytes))
                {
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
            let actual_peak_bytes = pending
                .allocation_bytes
                .checked_add(planned_index_allocation_bytes)
                .and_then(|bytes| bytes.checked_add(max_index_scratch_bytes))
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            if remaining_budget.is_some_and(|remaining| actual_peak_bytes > remaining) {
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
                    pending: Some(pending),
                    uniform_publication: None,
                    new_shard_id,
                    new_row_start,
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
        } else if !resets_existing_rows
            && !bootstrap_sentinel
            && identity.supports_in_place_append()
            && (!indexed_fixed_rollover_proof || source_is_all_i32_fixed(&source))
            && identity
                .row_count
                .checked_add(k)
                .is_some_and(|end| end <= identity.capacity)
        {
            let chunks = source
                .checked_append_chunks(identity.capacity, identity.row_count)
                .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let index_scratch_bytes = match mode {
                ResidentOpenShardAppendPreparationMode::TransactionTerminalUnindexed => 0,
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
            let retain_budget_guard = reserve_created_by || !unindexed;
            let (pending_created_by, budget_allocation) = if retain_budget_guard {
                let budget_allocation = preheld_budget_allocation.take().unwrap_or_else(|| {
                    self.read_state
                        .residency
                        .budget_allocation_lock
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                });
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
                    Some(budget) => match budget
                        .checked_sub(resident_bytes)
                        .and_then(|remaining| remaining.checked_sub(prior_reserved_bytes))
                    {
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
                (None, preheld_budget_allocation.take())
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
            let budget_allocation = preheld_budget_allocation.take().unwrap_or_else(|| {
                self.read_state
                    .residency
                    .budget_allocation_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            });
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
                Some(budget) => match budget
                    .checked_sub(resident_bytes)
                    .and_then(|remaining| remaining.checked_sub(prior_reserved_bytes))
                {
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
                    ..
                } => (true, max_index_scratch_bytes),
                ResidentOpenShardAppendPreparationMode::TransactionTerminalUnindexed => (false, 0),
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
            let (new_shard_id, new_row_start) = if resets_existing_rows {
                identity
                    .shard_id
                    .checked_add(1)
                    .map(|shard_id| (shard_id, 0))
            } else {
                checked_rollover_coordinates(
                    identity.shard_id,
                    identity.row_start,
                    identity.row_count,
                )
            }
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
                    pending: Some(pending),
                    uniform_publication: None,
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
            foreign_keys_prevalidated: unindexed,
            row_ids,
            resets_existing_rows,
            bootstrap_sentinel,
            branch,
            fixed_chunks,
            fixed_chunk_offsets,
            int4_min_max,
            bool_uploads,
            host_retention_prediction,
            budget_allocation,
            budget_reservation_retained_by_transaction: false,
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

    /// Internal reservation adapter over the one append compiler. The returned append reservation
    /// is owned only by the private indexed physical owner and reaches production solely through
    /// the generic codec-5 `DeviceInsertPlan` handoff.
    #[allow(clippy::too_many_arguments)] // exact indexed reservation capability inputs
    pub(super) fn prepare_resident_open_shard_append_indexed_in_place_reservation<'a>(
        &'a self,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        device_apply: MutexGuard<'a, ()>,
        index_scratch_bytes: u64,
        _permit: IndexedPhysicalMaterializationPermit,
        budget_allocation: Option<MutexGuard<'a, ()>>,
        prior_reserved_bytes: u64,
    ) -> Result<ResidentOpenShardAppendPlan<'a>, DeviceInsertPlanPrepareError> {
        self.prepare_resident_open_shard_append_core(
            source,
            row_ids,
            ResidentOpenShardAppendPreparationMode::IndexedInPlaceReservation {
                index_scratch_bytes,
            },
            Some(device_apply),
            budget_allocation,
            prior_reserved_bytes,
            false,
            None,
        )
    }

    /// Internal fixed-rollover reservation adapter. It reserves the private payload/sidecars
    /// plus the exact index and scratch envelope, without constructing a `DeviceInsertPlan` or
    /// exposing WAL, apply, cache, or descriptor-publication authority.
    #[allow(clippy::too_many_arguments)] // exact rollover reservation capability inputs
    pub(super) fn prepare_resident_open_shard_append_indexed_fixed_rollover_reservation<'a>(
        &'a self,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        device_apply: MutexGuard<'a, ()>,
        max_index_scratch_bytes: u64,
        _permit: IndexedPhysicalMaterializationPermit,
        budget_allocation: Option<MutexGuard<'a, ()>>,
        prior_reserved_bytes: u64,
        resets_existing_rows: bool,
        allow_s3_index_schema_transition: bool,
        s3_final_table: Option<&RelationalTable>,
    ) -> Result<ResidentOpenShardAppendPlan<'a>, DeviceInsertPlanPrepareError> {
        self.prepare_resident_open_shard_append_core(
            source,
            row_ids,
            ResidentOpenShardAppendPreparationMode::IndexedFixedRolloverReservation {
                max_index_scratch_bytes,
                allow_s3_index_schema_transition,
            },
            Some(device_apply),
            budget_allocation,
            prior_reserved_bytes,
            resets_existing_rows,
            s3_final_table,
        )
    }
}

fn source_matches_table(
    source: &PreparedResidentAppendSource,
    table: &RelationalTable,
    foreign_keys_prevalidated: bool,
) -> bool {
    source.table_name() == table.name
        && source.exact_single_table_dependency()
        && source.table_oid() == table.oid
        && table.indexes.is_empty()
        && (foreign_keys_prevalidated || table.foreign_keys.is_empty())
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
    // Transaction-terminal FK validation closes before codec-5 compiles this physical plan. FK
    // metadata and its additional table dependencies therefore constrain the semantic preflight,
    // not the single-target indexed append layout.
    source.table_name() == table.name
        && source.table_oid() == table.oid
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

/// The final S3 catalog shape may append exact index identities after the source's statement
/// snapshot. Its relation identity and physical column layout remain exact; only the catalog
/// schema digest is statement-time for this narrow pre-WAL reservation.
pub(super) fn source_matches_s3_created_index_reservation(
    source: &PreparedResidentAppendSource,
    table: &RelationalTable,
) -> bool {
    source.table_name() == table.name
        && source.table_oid() == table.oid
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
#[path = "fixed_insert/tests.rs"]
mod ownership_tests;
