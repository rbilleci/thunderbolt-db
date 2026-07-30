//! Physical ownership for one already-published indexed in-place reservation.
//!
//! This leaf prepares no canonical operation and exposes no device mutation entry point.  It
//! exists to make the exact resource/lifetime handoff auditable before WRITE-001 intentionally
//! connects it to the one live commit path.

#![allow(dead_code)] // deliberately compiled reservation; live strategy selection remains closed

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::fixed_insert::ResidentOpenShardAppendPlan;
use super::index_delta_preview::{
    IndexDeltaResourceLedger, IndexedInPlacePreview, PhysicalIndexBinding, RawIndexLogicalBinding,
};
use super::indexed_forecast::{
    IndexedPhysicalGenerationWitness, IndexedPhysicalResourceForecast, IndexedPhysicalTargetWitness,
};
use crate::engine_insert_plan::batch_key_constraints::BatchKeyConstraintProof;
use crate::engine_insert_plan::host_retention::{HostRetentionGeometry, HostRetentionReport};
use crate::engine_insert_plan::resident_key_constraints::ResidentKeyValidationSeal;
use crate::engine_state::TransactionNamedIndexPublicationGuard;
use crate::relational_model::RelationalTable;
#[cfg(test)]
use crate::typed_insert_batch::TypedInsertBatch;
use crate::{Engine, ExecuteError, Index};
use gpu_db_execution::{
    CudaAllocationScope, CudaResidentDeviceMemory, PreparedResidentTypedIndexesInsert,
};

#[cfg(test)]
pub(crate) use super::index_delta_preview::IndexedInPlacePreviewReport;

/// Scalar-only observation available to reservation tests. Device pointers, CUDA tokens, append
/// sources, cache entries, and lifecycle guards never cross this reporting boundary.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexedInPlaceProofReport {
    pub(crate) raw_index_count: usize,
    pub(crate) physical_index_count: usize,
    pub(crate) base_row: usize,
    pub(crate) incoming_rows: usize,
    pub(crate) preparation_bytes: u64,
    pub(crate) index_preparation_bytes: u64,
    pub(crate) fused_preparation_bytes: u64,
    pub(crate) fused_pooled_allocation_slots: u64,
    pub(crate) original_read_snapshot: Index,
    pub(crate) predecessor_boundary: Index,
    pub(crate) pinned_persistent_index_bytes: u64,
    pub(crate) descriptor_bytes: u64,
    pub(crate) descriptor_count: usize,
    pub(crate) bounded_readback_bytes: u64,
    pub(crate) allocation_pin_count: usize,
    pub(crate) host_retained_bytes: u64,
    pub(crate) host_allocation_slots: u64,
    pub(crate) host_generation_pin_slots: u64,
    pub(crate) peak_host_retained_bytes: u64,
    pub(crate) peak_host_allocation_slots: u64,
    pub(crate) peak_host_generation_pin_slots: u64,
}

/// Move-only device-index preparation and its exact proof/pin ledger.  Launch setup drops before
/// semantic witness and cache pins, so no retained basis can outlive the prepared CUDA drain.
struct PreparedResidentIndexDelta {
    launch: PreparedResidentTypedIndexesInsert,
    source_payload: Arc<CudaResidentDeviceMemory>,
    _created_by_region: Option<Arc<CudaResidentDeviceMemory>>,
    _row_id_region: Option<Arc<CudaResidentDeviceMemory>>,
    _deleted_by_region: Option<Arc<CudaResidentDeviceMemory>>,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    raw_bindings: Box<[RawIndexLogicalBinding]>,
    physical_bindings: Box<[PhysicalIndexBinding]>,
    mutation_epoch: Arc<AtomicU64>,
    mutation_epoch_expected_even: u64,
    ledger: IndexDeltaResourceLedger,
    host_retention_peak_before_finalization: HostRetentionGeometry,
}

impl PreparedResidentIndexDelta {
    fn host_retention_report(&self) -> Result<HostRetentionReport, ExecuteError> {
        let mut report = HostRetentionReport::default();
        self.key_proof.append_host_retention(&mut report)?;
        self.validation.append_host_retention(&mut report)?;
        report.retain_boxed_slice(&self.raw_bindings)?;
        report.retain_boxed_slice(&self.physical_bindings)?;
        for binding in self.physical_bindings.iter() {
            binding.append_host_retention(&mut report)?;
        }
        report.retain_arc_owner(&self.mutation_epoch)?;
        let launch = self.launch.host_retention_report().map_err(|error| {
            decline(format!(
                "indexed append launch host retention failed: {error}"
            ))
        })?;
        if let Some(identity) = launch.owner_array_backing_identity {
            report.retain_external_backing(identity, launch.owner_array_backing_bytes)?;
        }
        Ok(report)
    }
}

/// Move-only physical reservation. Declaration order is load-bearing: abandoning it drains its
/// whole prepared fused/index tail first, then releases the ordinary append reservation, and only
/// then releases named-index lifecycle protection.
pub(super) struct PreparedIndexedInPlaceReservation<'a> {
    fused: super::fixed_insert::PreparedIndexedInPlaceFusedApply,
    index_delta: PreparedResidentIndexDelta,
    append: ResidentOpenShardAppendPlan<'a>,
    forecast: IndexedPhysicalResourceForecast,
    fused_materialization_peak: HostRetentionGeometry,
    target_witness: IndexedPhysicalTargetWitness,
    generation_witness: IndexedPhysicalGenerationWitness,
    _named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
}

impl PreparedIndexedInPlaceReservation<'_> {
    fn host_retention_report(&self) -> Result<HostRetentionReport, ExecuteError> {
        let mut report = self.append.host_retention_report()?;
        report.merge(self.fused.host_retention_report()?)?;
        report.merge(self.index_delta.host_retention_report()?)?;
        Ok(report)
    }

    fn host_retention_peak(&self) -> Result<HostRetentionGeometry, ExecuteError> {
        let fused_final = self.fused.host_retention_report()?.geometry()?;
        let mut index_materialization_peak =
            self.index_delta.host_retention_peak_before_finalization;
        index_materialization_peak.checked_add_disjoint(
            fused_final,
            "indexed in-place fused final owner alongside index preparation",
        )?;
        Ok(self
            .fused_materialization_peak
            .peak(index_materialization_peak)
            .peak(self.host_retention_report()?.geometry()?))
    }

    fn actual_resource_forecast(&self) -> Result<IndexedPhysicalResourceForecast, ExecuteError> {
        let host = self.host_retention_report()?.geometry()?;
        let peak = self.host_retention_peak()?;
        let delta = &self.index_delta;
        let fused = &self.fused;
        let fused_footprint = fused.preparation_footprint();
        let incremental_allocation_slots = u64::try_from(
            delta
                .ledger
                .pending_created_by_allocation_count
                .checked_add(delta.ledger.transient_allocation_slot_count)
                .and_then(|slots| {
                    usize::try_from(fused_footprint.pooled_device_scratch_slots)
                        .ok()
                        .and_then(|fused_slots| slots.checked_add(fused_slots))
                })
                .ok_or_else(|| decline("indexed append forecast allocation-slot overflow"))?,
        )
        .map_err(|_| decline("indexed append forecast allocation-slot conversion overflow"))?;
        let simultaneous_preparation_bytes = delta
            .ledger
            .transient_preparation_bytes
            .checked_add(fused.preparation_bytes())
            .ok_or_else(|| decline("indexed append combined preparation-byte overflow"))?;
        Ok(IndexedPhysicalResourceForecast {
            final_host_retained_bytes: host.retained_bytes(),
            final_host_allocation_slots: host.allocation_slots(),
            final_host_generation_pin_slots: host.generation_pin_slots(),
            peak_host_retained_bytes: peak.retained_bytes(),
            peak_host_allocation_slots: peak.allocation_slots(),
            peak_host_generation_pin_slots: peak.generation_pin_slots(),
            old_generation_pinned_bytes: delta.ledger.retained_persistent_bytes,
            new_persistent_bytes: delta.ledger.pending_created_by_bytes,
            retained_device_transient_bytes: simultaneous_preparation_bytes,
            retained_device_result_bytes: 0,
            incremental_allocation_slots,
            generation_pin_slots: u64::try_from(delta.ledger.retained_allocation_pin_count)
                .map_err(|_| decline("indexed append forecast generation-pin overflow"))?,
            maximum_concurrent_device_scratch_bytes: simultaneous_preparation_bytes,
            maximum_host_readback_bytes: delta
                .ledger
                .bounded_readback_bytes
                .max(fused_footprint.status_readback_bytes),
        })
    }

    fn forecast_matches_actual(&self) -> Result<bool, ExecuteError> {
        let actual = self.actual_resource_forecast()?;
        let delta = &self.index_delta;
        let (_, base_row, catalog_seq) = self.append.indexed_in_place_reservation_basis();
        Ok(actual == self.forecast
            && self.target_witness.gpu_id == delta.source_payload.metadata().gpu_id
            && self.generation_witness.catalog_seq == catalog_seq
            && self.generation_witness.open_shard_id
                == self.append.indexed_in_place_reservation_basis().0
            && self.generation_witness.row_count == u64::try_from(base_row).unwrap_or(u64::MAX)
            && self.generation_witness.index_mutation_epoch_even
                == delta.mutation_epoch_expected_even
            && self.fused.stamps_match_expected_commit())
    }
}

#[cfg(test)]
impl PreparedIndexedInPlaceReservation<'_> {
    /// Consume the inert owner after exposing only its scalar accounting report.  There is no
    /// forwarding surface for the prepared CUDA token or append carrier.
    pub(super) fn inspect<R>(
        self,
        inspect: impl FnOnce(IndexedInPlaceProofReport) -> R,
    ) -> Result<R, ExecuteError> {
        let report = self.scalar_report_if_intact()?;
        Ok(inspect(report))
    }

    fn scalar_report_if_intact(&self) -> Result<IndexedInPlaceProofReport, ExecuteError> {
        let delta = &self.index_delta;
        let host = self.host_retention_report()?;
        let host_geometry = host.geometry()?;
        let host_peak = self.host_retention_peak()?;
        let (shard_id, base_row, _) = self.append.indexed_in_place_reservation_basis();
        let fused_footprint = self.fused.preparation_footprint();
        let index_preparation_bytes = delta.ledger.transient_preparation_bytes;
        let fused_preparation_bytes = self.fused.preparation_bytes();
        let preparation_bytes = index_preparation_bytes
            .checked_add(fused_preparation_bytes)
            .ok_or_else(|| decline("indexed append proof combined preparation bytes overflow"))?;
        let report = IndexedInPlaceProofReport {
            raw_index_count: delta.ledger.raw_index_count,
            physical_index_count: delta.ledger.physical_index_count,
            base_row,
            incoming_rows: self.append.row_count(),
            preparation_bytes,
            index_preparation_bytes,
            fused_preparation_bytes,
            fused_pooled_allocation_slots: fused_footprint.pooled_device_scratch_slots,
            original_read_snapshot: delta.validation.original_read_snapshot(),
            predecessor_boundary: delta.validation.predecessor_boundary(),
            pinned_persistent_index_bytes: delta.ledger.pinned_persistent_index_bytes,
            descriptor_bytes: delta.ledger.descriptor_bytes,
            descriptor_count: delta.ledger.descriptor_count,
            bounded_readback_bytes: delta
                .ledger
                .bounded_readback_bytes
                .max(fused_footprint.status_readback_bytes),
            allocation_pin_count: delta.ledger.retained_allocation_pin_count,
            host_retained_bytes: host_geometry.retained_bytes(),
            host_allocation_slots: host_geometry.allocation_slots(),
            host_generation_pin_slots: host_geometry.generation_pin_slots(),
            peak_host_retained_bytes: host_peak.retained_bytes(),
            peak_host_allocation_slots: host_peak.allocation_slots(),
            peak_host_generation_pin_slots: host_peak.generation_pin_slots(),
        };
        if !self.append.holds_budget_reservation()
            || base_row != report.base_row
            || self.append.row_count() != report.incoming_rows
            || delta.launch.preparation_bytes() != delta.ledger.transient_preparation_bytes
            || report.index_preparation_bytes != delta.ledger.transient_preparation_bytes
            || report.fused_preparation_bytes != self.fused.preparation_bytes()
            || report.preparation_bytes
                != report
                    .index_preparation_bytes
                    .saturating_add(report.fused_preparation_bytes)
            || report.fused_pooled_allocation_slots != fused_footprint.pooled_device_scratch_slots
            || self.fused.preparation_bytes() != fused_footprint.pooled_device_scratch_bytes
            || !self.fused.stamps_match_expected_commit()
            || report.original_read_snapshot != delta.validation.original_read_snapshot()
            || report.predecessor_boundary != delta.validation.predecessor_boundary()
            || delta.mutation_epoch.load(Ordering::Acquire) != delta.mutation_epoch_expected_even
            || delta.mutation_epoch_expected_even & 1 != 0
            || delta.key_proof.indexes().len() != delta.raw_bindings.len()
            || delta.source_payload.device_ptr() == 0
        {
            return Err(decline(
                "indexed append proof retained resource integrity drifted",
            ));
        }
        let pinned_bytes = delta
            .physical_bindings
            .iter()
            .try_fold(0_u64, |total, binding| {
                total
                    .checked_add(binding.allocated_bytes)
                    .ok_or_else(|| decline("indexed append proof retained byte sum overflows"))
            })?;
        let mut retained_allocations = std::collections::BTreeMap::<usize, u64>::new();
        let mut retain = |memory: &Arc<CudaResidentDeviceMemory>| -> Result<(), ExecuteError> {
            let identity = memory.allocation_identity();
            let bytes = memory.metadata().allocated_bytes;
            if let Some(existing) = retained_allocations.insert(identity, bytes) {
                if existing != bytes {
                    return Err(decline(
                        "indexed append proof allocation identity byte drifted",
                    ));
                }
            }
            Ok(())
        };
        retain(&delta.source_payload)?;
        for binding in delta.physical_bindings.iter() {
            retain(&binding.device_index)?;
        }
        for memory in [
            delta._created_by_region.as_ref(),
            delta._row_id_region.as_ref(),
            delta._deleted_by_region.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            retain(memory)?;
        }
        let retained_bytes = retained_allocations
            .values()
            .try_fold(0_u64, |total, bytes| {
                total
                    .checked_add(*bytes)
                    .ok_or_else(|| decline("indexed append proof retained total overflows"))
            })?;
        if pinned_bytes != delta.ledger.pinned_persistent_index_bytes
            || delta.ledger.raw_index_count != delta.raw_bindings.len()
            || delta.ledger.physical_index_count != delta.physical_bindings.len()
            || delta.ledger.retained_persistent_bytes != retained_bytes
            || delta.ledger.retained_allocation_pin_count != retained_allocations.len()
            || delta.ledger.bounded_readback_bytes != std::mem::size_of::<u32>() as u64
            || report.raw_index_count != delta.ledger.raw_index_count
            || report.physical_index_count != delta.ledger.physical_index_count
            || report.pinned_persistent_index_bytes != delta.ledger.pinned_persistent_index_bytes
            || report.descriptor_bytes != delta.ledger.descriptor_bytes
            || report.descriptor_count != delta.ledger.descriptor_count
            || report.bounded_readback_bytes
                != delta
                    .ledger
                    .bounded_readback_bytes
                    .max(fused_footprint.status_readback_bytes)
            || report.allocation_pin_count != delta.ledger.retained_allocation_pin_count
        {
            return Err(decline("indexed append proof retained ledger drifted"));
        }
        for (raw_ordinal, raw) in delta.raw_bindings.iter().enumerate() {
            let physical = delta
                .physical_bindings
                .get(raw.physical_ordinal)
                .ok_or_else(|| decline("indexed append proof raw mapping lost a physical pin"))?;
            if raw.raw_ordinal != raw_ordinal
                || raw.key_id != physical.key_id
                || delta
                    .key_proof
                    .indexes()
                    .get(raw_ordinal)
                    .is_none_or(|binding| binding.raw_ordinal() != raw_ordinal)
            {
                return Err(decline("indexed append proof raw mapping drifted"));
            }
        }
        for physical in delta.physical_bindings.iter() {
            if physical.cache_key.1 != shard_id
                || physical.cache_key.2 != physical.key_id
                || physical.source_ptr != delta.source_payload.device_ptr()
                || physical.base_row != base_row
                || physical.end_row < physical.base_row
                || physical.end_row > physical.capacity
                || physical.device_index.device_ptr() == physical.source_ptr
                || physical.device_index.metadata().allocated_bytes != physical.allocated_bytes
                || physical.published_row_count.load(Ordering::Acquire) != physical.base_row
                || physical.published_has_postings.load(Ordering::Acquire)
                    != physical.has_postings_at_preview
                || physical.gc_boundary > delta.validation.original_read_snapshot()
                || physical.table_mask == 0
                || physical.hash_shift != 32 - (u64::from(physical.table_mask) + 1).trailing_zeros()
            {
                return Err(decline("indexed append proof physical pin drifted"));
            }
        }
        Ok(report)
    }
}

/// Assemble and inspect the inert proof while the caller retains the canonical commit guard and
/// passes only a short proof alongside its already-held residency mutation gate.
#[allow(clippy::too_many_arguments)] // proof ownership stays explicit; no opaque live-operation carrier
#[cfg(test)]
pub(crate) fn inspect_prepared_indexed_in_place<'a, R>(
    engine: &'a Engine,
    table: &RelationalTable,
    predecessor_boundary: Index,
    batch: TypedInsertBatch,
    row_ids: super::DeviceInsertRowIds,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    mutation_gate: std::sync::MutexGuard<'a, ()>,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
    commit_proof: &std::sync::MutexGuard<'_, crate::CommitState>,
    permit: crate::engine_insert_plan::IndexedPhysicalMaterializationPermit,
    inspect: impl FnOnce(IndexedInPlaceProofReport) -> R,
) -> Result<R, ExecuteError> {
    let preview = super::index_delta_preview::prepare(
        engine,
        table,
        predecessor_boundary,
        batch,
        row_ids,
        key_proof,
        validation,
        mutation_gate,
        named_index_lifecycle,
        commit_proof,
    )?;
    prepare(
        engine,
        table,
        preview,
        commit_proof.repl.peek_next_index(),
        permit,
    )
    .map(super::indexed_reservation::PreparedIndexedPhysicalReservation::in_place)
    .and_then(|prepared| prepared.inspect_in_place(inspect))
}

/// Scalar-only preview inspection. This deliberately stops before the append compiler and every
/// CUDA preparation resource while retaining the same commit -> named-index -> mutation guards.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn inspect_prepared_indexed_in_place_preview<'a, R>(
    engine: &'a Engine,
    table: &RelationalTable,
    predecessor_boundary: Index,
    batch: TypedInsertBatch,
    row_ids: super::DeviceInsertRowIds,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    mutation_gate: std::sync::MutexGuard<'a, ()>,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
    commit_proof: &std::sync::MutexGuard<'_, crate::CommitState>,
    inspect: impl FnOnce(IndexedInPlacePreviewReport) -> R,
) -> Result<R, ExecuteError> {
    super::index_delta_preview::inspect_prepared_indexed_in_place_preview(
        engine,
        table,
        predecessor_boundary,
        batch,
        row_ids,
        key_proof,
        validation,
        mutation_gate,
        named_index_lifecycle,
        commit_proof,
        inspect,
    )
}

pub(super) fn prepare<'a>(
    engine: &'a Engine,
    table: &RelationalTable,
    preview: IndexedInPlacePreview<'a>,
    expected_commit_seq: Index,
    permit: crate::engine_insert_plan::IndexedPhysicalMaterializationPermit,
) -> Result<PreparedIndexedInPlaceReservation<'a>, ExecuteError> {
    let parts = preview.into_append_and_parts(engine, table, permit)?;
    let materialized_host_retention = parts.host_retention_report()?;
    let preview_host_retention = parts.preview_host_retention;
    let super::index_delta_preview::PreviewPreparedParts {
        append,
        source_payload,
        created_by_region,
        row_id_region,
        deleted_by_region,
        key_proof,
        validation,
        logical,
        requests,
        physical_bindings,
        mutation_epoch,
        mutation_epoch_expected_even,
        ledger,
        fused_footprint,
        fused_final_host_geometry,
        fused_materialization_scratch,
        fused_materialization_peak: forecast_fused_materialization_peak,
        target_witness,
        generation_witness,
        preview_host_retention: _,
        resource_forecast,
        materialized_append_host_geometry: _,
        named_index_lifecycle,
    } = parts;
    let (_, base_row, _) = append.indexed_in_place_reservation_basis();
    let fused_inputs = append
        .prepare_indexed_in_place_fused_apply_inputs(expected_commit_seq)
        .map_err(|_| decline("indexed append fused preparation inputs declined"))?;
    let fused_preparation_bytes = fused_inputs.preparation_bytes();
    if fused_inputs.preparation_footprint() != fused_footprint
        || fused_inputs.materialization_host_scratch() != fused_materialization_scratch
    {
        return Err(decline(
            "indexed append fused preparation inputs diverged from the admitted forecast",
        ));
    }
    let simultaneous_preparation_bytes = logical
        .preparation_bytes()
        .checked_add(fused_preparation_bytes)
        .ok_or_else(|| decline("indexed append combined preparation-byte overflow"))?;
    append
        .ensure_indexed_in_place_preparation_budget(engine, simultaneous_preparation_bytes)
        .map_err(|_| decline("indexed append combined preparation budget declined"))?;
    let allocation_scope = CudaAllocationScope::with_budget(simultaneous_preparation_bytes);
    let fused_materialization_scratch = fused_inputs.materialization_host_scratch();
    let fused = fused_inputs
        .materialize()
        .map_err(|_| decline("indexed append fused CUDA preparation declined"))?;
    let fused_host_retention = fused.host_retention_report()?;
    if fused_host_retention.geometry()? != fused_final_host_geometry {
        return Err(decline(
            "indexed append fused final host retention diverged from the admitted forecast",
        ));
    }
    let mut fused_materialization_overlap = materialized_host_retention.clone();
    fused_materialization_overlap.merge(fused_host_retention)?;
    let mut fused_materialization_peak = fused_materialization_overlap.geometry()?;
    fused_materialization_peak.checked_add_disjoint(
        fused_materialization_scratch,
        "indexed in-place fused temporary materialization backing",
    )?;
    if fused_materialization_peak != forecast_fused_materialization_peak {
        return Err(decline(
            "indexed append fused materialization peak diverged from the admitted forecast",
        ));
    }
    let index_delta = source_payload
        .prepare_resident_typed_indexes_insert(&requests, base_row, append.row_count())
        .map_err(|_| decline("indexed append proof CUDA preparation declined"))?;
    let launch_host_retention = index_delta.host_retention_report().map_err(|error| {
        decline(format!(
            "indexed append launch host retention failed: {error}"
        ))
    })?;
    let mut materialized_launch_overlap = materialized_host_retention;
    if let Some(identity) = launch_host_retention.owner_array_backing_identity {
        materialized_launch_overlap
            .retain_external_backing(identity, launch_host_retention.owner_array_backing_bytes)?;
    }
    let host_retention_peak_before_finalization =
        preview_host_retention.peak(materialized_launch_overlap.geometry()?);
    if index_delta.preparation_bytes() != logical.preparation_bytes()
        || fused.preparation_bytes() != fused_preparation_bytes
        || mutation_epoch.load(Ordering::Acquire) != mutation_epoch_expected_even
        || mutation_epoch_expected_even & 1 != 0
        || allocation_scope.peak_bytes() != simultaneous_preparation_bytes
    {
        return Err(decline(
            "indexed append proof allocating preparation drifted",
        ));
    }
    drop(allocation_scope);
    let reservation = PreparedIndexedInPlaceReservation {
        fused,
        index_delta: PreparedResidentIndexDelta {
            launch: index_delta,
            source_payload,
            _created_by_region: created_by_region,
            _row_id_region: row_id_region,
            _deleted_by_region: deleted_by_region,
            key_proof,
            validation,
            raw_bindings: logical.raw,
            physical_bindings,
            mutation_epoch,
            mutation_epoch_expected_even,
            ledger,
            host_retention_peak_before_finalization,
        },
        append,
        forecast: resource_forecast,
        fused_materialization_peak,
        target_witness,
        generation_witness,
        _named_index_lifecycle: named_index_lifecycle,
    };
    if !reservation.forecast_matches_actual()? {
        return Err(decline(
            "indexed append materialization owner ledger diverged from forecast",
        ));
    }
    Ok(reservation)
}

fn decline(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Serialization(message.into())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn indexed_resident_engine() -> Option<crate::Engine> {
        let mut engine = crate::Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return None;
        }
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);
        engine
            .execute_text(1, "CREATE TABLE inert_index_delta_other (value int4)")
            .unwrap();
        engine
            .execute_text(
                2,
                "CREATE TABLE inert_index_delta (id int4 PRIMARY KEY, shared int4, code int4 UNIQUE)",
            )
            .unwrap();
        engine
            .execute_text(
                3,
                "CREATE INDEX inert_index_delta_shared_a ON inert_index_delta (shared)",
            )
            .unwrap();
        engine
            .execute_text(
                4,
                "CREATE INDEX inert_index_delta_shared_b ON inert_index_delta (shared)",
            )
            .unwrap();
        engine
            .execute_text(
                5,
                "CREATE INDEX inert_index_delta_compound ON inert_index_delta (shared, code)",
            )
            .unwrap();
        engine
            .execute_text(6, "INSERT INTO inert_index_delta VALUES (1, 10, 100)")
            .unwrap();
        engine
            .populate_relational_residency_snapshot("inert_index_delta")
            .unwrap();
        engine
            .publish_relational_resident_indexes("inert_index_delta")
            .unwrap();
        Some(engine)
    }

    fn indexed_proof_plan(
        engine: &crate::Engine,
    ) -> crate::engine_insert_plan::PreparedDeviceInsertPlan {
        let catalog = engine.catalog_snapshot();
        let command =
            gpu_db_sql::parse_command("INSERT INTO inert_index_delta VALUES (2, 20, 200)").unwrap();
        let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
            &command,
            &catalog,
            catalog.commit_seq,
        )
        .unwrap()
        .expect("indexed proof-only builder remains eligible");
        crate::engine_insert_plan::PreparedDeviceInsertPlan::from_typed_batch(
            batch, engine, &catalog,
        )
        .unwrap()
    }

    #[test]
    fn indexed_in_place_preparation_is_inert_and_pins_the_open_generation() {
        let Some(engine) = indexed_resident_engine() else {
            return;
        };
        let catalog = engine.catalog_snapshot();
        let plan = indexed_proof_plan(&engine);
        let epoch = engine
            .read_state
            .residency
            .point_index_mutation_epoch_for_table(
                &engine.read_state,
                catalog
                    .relational_catalog
                    .get("inert_index_delta")
                    .expect("inert proof table remains catalog-visible"),
            )
            .expect("exact inert proof table installs its point slot");
        let epoch_before = epoch.load(Ordering::Acquire);
        let wal_before = engine.durable_wal_records().len();
        let boundary_before = engine.committed_seq();
        let allocator_before = engine.read_state.mvcc.current_row_id();
        let (shard_id, source_ptr, source_header) = {
            let shards = engine.read_state.residency.shards.load_full();
            let open = shards
                .get("inert_index_delta")
                .and_then(|shards| shards.last())
                .unwrap();
            let payload = open.device_memory.as_ref().unwrap();
            (
                open.shard_id,
                payload.device_ptr(),
                payload.read_resident_i32_column(0, 2).unwrap(),
            )
        };
        let key_ids = catalog
            .relational_catalog
            .get("inert_index_delta")
            .unwrap()
            .indexes
            .iter()
            .enumerate()
            .map(|(ordinal, index)| {
                crate::engine_residency::index_probe_key_id(
                    catalog.relational_catalog.get("inert_index_delta").unwrap(),
                    index,
                    ordinal,
                )
                .unwrap()
            })
            .collect::<BTreeSet<_>>();
        let primary_key_id = crate::engine_residency::index_probe_key_id(
            catalog.relational_catalog.get("inert_index_delta").unwrap(),
            &catalog
                .relational_catalog
                .get("inert_index_delta")
                .unwrap()
                .indexes[0],
            0,
        )
        .unwrap();
        let (primary_index, primary_table_mask, primary_hash_shift, primary_row_count) =
            super::super::capacity_payload_tests::resident_named_index_cache_entry(
                &engine,
                "inert_index_delta",
                shard_id,
                primary_key_id,
            );
        assert_eq!(
            super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                &primary_index,
                primary_table_mask,
                primary_hash_shift,
                primary_row_count,
                1,
            ),
            1
        );
        assert_eq!(
            super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                &primary_index,
                primary_table_mask,
                primary_hash_shift,
                primary_row_count,
                2,
            ),
            0
        );
        let publication_before = engine
            .read_state
            .residency
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let proposal = plan.prepare_row_id_proposal(allocator_before).unwrap();
        let report = plan
            .inspect_current_resident_index_delta(&engine, proposal, |report| {
                let cache = engine
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let coverage = engine
                    .read_state
                    .residency
                    .named_index_coverage
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                for key_id in &key_ids {
                    let key = ("inert_index_delta".to_string(), shard_id, *key_id);
                    let entry = cache.get(&key).unwrap();
                    assert_eq!(entry.resident_device_ptr, source_ptr);
                    assert_eq!(entry.row_count, 1);
                    assert_eq!(entry.published_row_count.load(Ordering::Acquire), 1);
                    assert_eq!(coverage.get(&key), Some(&(source_ptr, 1)));
                }
                drop(coverage);
                drop(cache);
                let shards = engine.read_state.residency.shards.load_full();
                let payload = shards
                    .get("inert_index_delta")
                    .and_then(|shards| shards.last())
                    .and_then(|shard| shard.device_memory.as_ref())
                    .unwrap();
                assert_eq!(payload.device_ptr(), source_ptr);
                assert_eq!(
                    payload.read_resident_i32_column(0, 2).unwrap(),
                    source_header
                );
                assert_eq!(
                    super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                        &primary_index,
                        primary_table_mask,
                        primary_hash_shift,
                        primary_row_count,
                        1,
                    ),
                    1
                );
                assert_eq!(
                    super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                        &primary_index,
                        primary_table_mask,
                        primary_hash_shift,
                        primary_row_count,
                        2,
                    ),
                    0
                );
                engine
                    .read_state
                    .residency
                    .purge_shard_pk_index_for_table("inert_index_delta");
                assert!(engine
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .keys()
                    .any(|(table, _, _)| table == "inert_index_delta"));
                report
            })
            .unwrap();
        assert_eq!(report.raw_index_count, 5);
        assert_eq!(report.physical_index_count, 4);
        assert_eq!(report.base_row, 1);
        assert_eq!(report.incoming_rows, 1);
        assert_eq!(report.index_preparation_bytes, 512);
        assert!(report.fused_preparation_bytes > 0);
        assert_eq!(
            report.preparation_bytes,
            report
                .index_preparation_bytes
                .checked_add(report.fused_preparation_bytes)
                .unwrap()
        );
        assert!(report.fused_pooled_allocation_slots > 0);
        assert_eq!(report.bounded_readback_bytes, 4);
        assert_eq!(report.allocation_pin_count, 7);
        assert!(report.pinned_persistent_index_bytes > 0);
        assert!(report.host_retained_bytes > 0);
        assert!(report.host_allocation_slots > 0);
        assert_eq!(report.host_generation_pin_slots, 0);
        assert!(report.peak_host_retained_bytes >= report.host_retained_bytes);
        assert!(report.peak_host_allocation_slots >= report.host_allocation_slots);
        assert!(report.peak_host_generation_pin_slots >= report.host_generation_pin_slots);
        assert_eq!(report.descriptor_bytes, 256);
        assert_eq!(report.descriptor_count, 5);
        assert_eq!(epoch.load(Ordering::Acquire), epoch_before);
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.committed_seq(), boundary_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), allocator_before);
        // The cache is now retired, but this pre-proof Arc remains a live device pin.  It must
        // still answer the old key and not the merely proposed key; no write launch occurred.
        assert_eq!(
            super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                &primary_index,
                primary_table_mask,
                primary_hash_shift,
                primary_row_count,
                1,
            ),
            1
        );
        assert_eq!(
            super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                &primary_index,
                primary_table_mask,
                primary_hash_shift,
                primary_row_count,
                2,
            ),
            0
        );
        assert_eq!(
            *engine
                .read_state
                .residency
                .named_index_publications
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            publication_before
        );
        assert!(!engine
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .any(|(table, _, _)| table == "inert_index_delta"));
        assert!(!engine
            .read_state
            .residency
            .named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .any(|(table, _, _)| table == "inert_index_delta"));
    }

    #[test]
    fn indexed_preparation_declines_missing_stale_aliased_and_incomplete_cache_evidence() {
        for sabotage in [
            "missing",
            "stale_geometry",
            "short_published_geometry",
            "stale_coverage",
            "destination_alias",
            "source_arc_pin_mismatch",
            "horizon_metadata_mismatch",
            "shared_posting_verdict_mismatch",
            "incomplete",
        ] {
            let Some(engine) = indexed_resident_engine() else {
                return;
            };
            let catalog = engine.catalog_snapshot();
            let table = catalog.relational_catalog.get("inert_index_delta").unwrap();
            let key_id =
                crate::engine_residency::index_probe_key_id(table, &table.indexes[0], 0).unwrap();
            let distinct_key_id =
                crate::engine_residency::index_probe_key_id(table, &table.indexes[1], 1).unwrap();
            let shard_id = engine
                .read_state
                .residency
                .shards
                .load_full()
                .get("inert_index_delta")
                .and_then(|shards| shards.last())
                .unwrap()
                .shard_id;
            let cache_key = ("inert_index_delta".to_string(), shard_id, key_id);
            match sabotage {
                "missing" => {
                    engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .remove(&cache_key);
                }
                "stale_geometry" => {
                    engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .get_mut(&cache_key)
                        .unwrap()
                        .row_count = 0;
                }
                "short_published_geometry" => {
                    engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .get(&cache_key)
                        .unwrap()
                        .published_row_count
                        .store(0, Ordering::Release);
                }
                "stale_coverage" => {
                    engine
                        .read_state
                        .residency
                        .named_index_coverage
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .insert(cache_key.clone(), (0, 0));
                }
                "destination_alias" => {
                    let distinct_cache_key =
                        ("inert_index_delta".to_string(), shard_id, distinct_key_id);
                    let mut cache = engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let aliased_index = cache
                        .get(&cache_key)
                        .and_then(|entry| entry.device_index.as_ref())
                        .cloned()
                        .unwrap();
                    cache.get_mut(&distinct_cache_key).unwrap().device_index = Some(aliased_index);
                }
                "source_arc_pin_mismatch" => {
                    let mut cache = engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let different_allocation = cache
                        .get(&cache_key)
                        .and_then(|entry| entry.device_index.as_ref())
                        .cloned()
                        .unwrap();
                    cache.get_mut(&cache_key).unwrap()._resident_guard = different_allocation;
                }
                "horizon_metadata_mismatch" => {
                    let mut cache = engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let entry = cache.get_mut(&cache_key).unwrap();
                    entry.table_mask ^= 1;
                    assert_ne!(entry.table_mask, 0);
                }
                "shared_posting_verdict_mismatch" => {
                    let cache = engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let entry = cache.get(&cache_key).unwrap();
                    entry
                        .published_has_postings
                        .store(!entry.has_postings, Ordering::Release);
                }
                "incomplete" => {
                    engine
                        .read_state
                        .residency
                        .named_index_coverage
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .remove(&cache_key);
                }
                _ => unreachable!(),
            }
            let plan = indexed_proof_plan(&engine);
            let proposal = plan
                .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
                .unwrap();
            let wal_before = engine.durable_wal_records().len();
            let boundary_before = engine.committed_seq();
            assert!(matches!(
                plan.inspect_current_resident_index_delta(&engine, proposal, |_| ()),
                Err(ExecuteError::Serialization(_))
            ));
            assert_eq!(engine.durable_wal_records().len(), wal_before, "{sabotage}");
            assert_eq!(engine.committed_seq(), boundary_before, "{sabotage}");
        }
    }

    #[test]
    fn indexed_preparation_declines_when_exact_pooled_bytes_exceed_remaining_budget() {
        let Some(mut engine) = indexed_resident_engine() else {
            return;
        };
        let plan = indexed_proof_plan(&engine);
        let gpu_id = engine
            .read_state
            .residency
            .shards
            .load_full()
            .get("inert_index_delta")
            .and_then(|shards| shards.last())
            .unwrap()
            .gpu_id;
        let resident = engine.relational_resident_bytes_for_gpu(gpu_id);
        // The exact fanout geometry is a 512-byte two-lease pool reservation on this shape.
        // Keep validation's small source/verdict allocation available, then refuse the append
        // core's sidecar(0)+prepared-token(512) reservation before the token can allocate.
        engine.set_relational_residency_budget_bytes(gpu_id, resident + 511);
        let proposal = plan
            .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        let boundary_before = engine.committed_seq();
        assert!(matches!(
            plan.inspect_current_resident_index_delta(&engine, proposal, |_| ()),
            Err(ExecuteError::Serialization(_))
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.committed_seq(), boundary_before);
    }

    #[test]
    fn indexed_preparation_refuses_the_fused_plus_index_sum_before_wal() {
        let Some(mut engine) = indexed_resident_engine() else {
            return;
        };
        let baseline_plan = indexed_proof_plan(&engine);
        let baseline_proposal = baseline_plan
            .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
            .unwrap();
        let baseline = baseline_plan
            .inspect_current_resident_index_delta(&engine, baseline_proposal, |report| report)
            .unwrap();
        assert!(baseline.fused_preparation_bytes > 0);
        assert_eq!(
            baseline.preparation_bytes,
            baseline
                .index_preparation_bytes
                .checked_add(baseline.fused_preparation_bytes)
                .unwrap()
        );
        // Construct the ordinary typed plan while unrestricted. The budget below targets the
        // indexed physical materializer, where the append core may admit the index lease alone
        // but must reject the exact simultaneous index-plus-fused lease before CUDA setup.
        let plan = indexed_proof_plan(&engine);
        let gpu_id = engine
            .read_state
            .residency
            .shards
            .load_full()
            .get("inert_index_delta")
            .and_then(|shards| shards.last())
            .unwrap()
            .gpu_id;
        let resident = engine.relational_resident_bytes_for_gpu(gpu_id);
        // This admits the indexed multi-lease alone but not the simultaneously retained fused
        // lease. The second reservation check must reject before a WAL record or commit step.
        engine.set_relational_residency_budget_bytes(
            gpu_id,
            resident + baseline.preparation_bytes - 1,
        );
        let proposal = plan
            .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        let boundary_before = engine.committed_seq();
        assert!(matches!(
            plan.inspect_current_resident_index_delta(&engine, proposal, |_| ()),
            Err(ExecuteError::Serialization(_))
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.committed_seq(), boundary_before);
    }

    #[test]
    fn indexed_preparation_revalidates_target_binding_at_an_advanced_predecessor_boundary() {
        let Some(engine) = indexed_resident_engine() else {
            return;
        };
        let original_read_snapshot = engine.committed_seq();
        let plan = indexed_proof_plan(&engine);
        // Advance the global catalog/commit sequence through unrelated DML only.  The target
        // relation, index enrollment, and resident generation remain unchanged.
        engine
            .execute_text(77, "INSERT INTO inert_index_delta_other VALUES (7)")
            .unwrap();
        let predecessor_boundary = engine.committed_seq();
        assert!(predecessor_boundary > original_read_snapshot);
        let proposal = plan
            .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
            .unwrap();
        let report = plan
            .inspect_current_resident_index_delta(&engine, proposal, |report| report)
            .unwrap();
        assert_eq!(report.original_read_snapshot, original_read_snapshot);
        assert_eq!(report.predecessor_boundary, predecessor_boundary);
        assert!(report.original_read_snapshot < report.predecessor_boundary);
        assert_eq!(engine.committed_seq(), predecessor_boundary);
    }

    #[test]
    fn reservation_owner_exposes_only_scalar_inspection() {
        let source = include_str!("index_delta.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production reservation owner precedes tests");
        assert!(source.contains("struct PreparedIndexedInPlaceReservation"));
        assert!(source.contains("struct PreparedResidentIndexDelta"));
        assert!(source.contains("PreparedIndexedInPlaceFusedApply"));
        assert!(source.contains("launch: PreparedResidentTypedIndexesInsert"));
        assert!(source.contains("fused: super::fixed_insert::PreparedIndexedInPlaceFusedApply"));
        assert!(source.contains("index_delta: PreparedResidentIndexDelta"));
        assert!(source.contains("append: ResidentOpenShardAppendPlan"));
        assert!(source.contains("key_proof: BatchKeyConstraintProof"));
        assert!(source.contains("validation: ResidentKeyValidationSeal"));
        let preview_source = include_str!("index_delta_preview.rs");
        assert!(preview_source.contains("struct IndexDeltaResourceLedger"));
        assert!(source.contains("mutation_epoch_expected_even"));
        assert!(source.contains("CudaAllocationScope::with_budget(simultaneous_preparation_bytes)"));
        assert!(source.contains("ensure_indexed_in_place_preparation_budget"));
        assert!(source.contains("fused.preparation_bytes() != fused_preparation_bytes"));
        assert!(source.contains("_named_index_lifecycle"));
        assert!(!source.contains("MutexGuard<'a, crate::CommitState>"));
        assert!(!source.contains("begin_point_index_mutation"));
        let wrapper = source
            .split("struct PreparedIndexedInPlaceReservation")
            .nth(1)
            .and_then(|section| {
                section
                    .split("\n}\n\n#[cfg(test)]\nimpl PreparedIndexedInPlaceReservation")
                    .next()
            })
            .expect("wrapper declaration");
        assert!(
            wrapper.find("fused:") < wrapper.find("index_delta:")
                && wrapper.find("index_delta:") < wrapper.find("append:")
                && wrapper.find("append:") < wrapper.find("_named_index_lifecycle:"),
            "drop order must remain fused/index delta -> append -> lifecycle"
        );
        let owner = include_str!("indexed_reservation.rs")
            .split("impl PreparedIndexedPhysicalReservation")
            .nth(1)
            .and_then(|section| section.split("\n#[cfg(test)]\nmod tests").next())
            .expect("scalar-only physical reservation inspection");
        for forbidden in ["submit", "apply", "into_parts", "wal", "status", "poison"] {
            assert!(
                !owner.contains(forbidden),
                "physical reservation owner must not expose {forbidden}"
            );
        }
        assert!(owner.contains("inspect_in_place"));
    }
}
