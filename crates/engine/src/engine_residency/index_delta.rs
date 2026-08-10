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
use crate::engine_state::{PreparedPointIndexMutationGuard, TransactionNamedIndexPublicationGuard};
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
    launch: Option<PreparedResidentTypedIndexesInsert>,
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
    point_epoch_transition: PreparedPointIndexMutationGuard,
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
        let launch = self
            .launch
            .as_ref()
            .ok_or_else(|| decline("indexed append launch owner was consumed"))?
            .host_retention_report()
            .map_err(|error| {
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
    // The common generic finalizer moves this exact guard back out immediately before it enters
    // final publication.  Retaining it through all pre-WAL preparation prevents a second
    // lifecycle authority while cache retirement is deferred for the sealed physical basis.
    named_index_lifecycle: Option<TransactionNamedIndexPublicationGuard<'a>>,
}

impl<'a> PreparedIndexedInPlaceReservation<'a> {
    pub(super) fn table_name(&self) -> &str {
        self.append.table_name()
    }

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

    pub(super) fn take_named_index_publication_guard(
        &mut self,
    ) -> Option<TransactionNamedIndexPublicationGuard<'a>> {
        self.named_index_lifecycle.take()
    }

    pub(super) fn take_transaction_terminal_device_apply_guard(
        &mut self,
    ) -> Option<std::sync::MutexGuard<'a, ()>> {
        self.append.take_transaction_terminal_device_apply_guard()
    }

    pub(super) fn take_transaction_terminal_budget_guard(
        &mut self,
    ) -> Option<(std::sync::MutexGuard<'a, ()>, u64)> {
        self.append.take_transaction_terminal_budget_guard()
    }

    /// Consume the one already-prepared indexed append after the generic finalizer has crossed
    /// its canonical WAL/status boundary.  The order is load-bearing: payload/sidecars become
    /// device-visible to the index tail first, the tail completes, then exactly one prebuilt
    /// header store exposes the row extent.  Every failure after terminal arm leaves the exact
    /// point-index epoch poisoned rather than permitting a cache or legacy retry.
    pub(super) fn apply_after_transaction_wal_claim(
        self,
        engine: &Engine,
        created_by: super::AppendCreatedBy<'_>,
    ) -> Result<(), super::DeviceInsertPlanApplyError> {
        let Self {
            fused,
            mut index_delta,
            append,
            forecast: _,
            fused_materialization_peak: _,
            target_witness,
            generation_witness,
            named_index_lifecycle: _,
        } = self;
        let expected_commit = fused.expected_commit_sequence();
        if !matches!(created_by, super::AppendCreatedBy::InsertUniform(sequence) if sequence == expected_commit)
        {
            return Err(super::DeviceInsertPlanApplyError::PlanDrift);
        }
        let catalog = engine.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(append.table_name())
            .ok_or(super::DeviceInsertPlanApplyError::PlanDrift)?;
        let (shard_id, base_row, catalog_seq) = append.indexed_in_place_reservation_basis();
        let shards = engine.read_state.residency.shards.load_full();
        let open = shards
            .get(append.table_name())
            .and_then(|shards| shards.last())
            .ok_or(super::DeviceInsertPlanApplyError::PlanDrift)?;
        if catalog.commit_seq != catalog_seq
            || table.oid != target_witness.table_oid
            || crate::engine_transaction_reset::table_schema_digest(table).ok()
                != Some(target_witness.schema_digest)
            || !super::fixed_insert::source_matches_indexed_in_place_reservation(
                append.source(),
                table,
            )
            || table.indexes.len() != index_delta.raw_bindings.len()
            || table.indexes.is_empty()
            || table.indexes.iter().any(|index| {
                (index.primary_key && !index.unique)
                    || (index.unique_constraint && !index.unique)
                    || index.key_columns.is_empty()
            })
            || open.shard_id != shard_id
            || open.row_count != base_row
            || !index_delta.validation.matches_in_place_append(
                table,
                catalog_seq,
                open,
                generation_witness.predecessor_boundary,
            )
            || index_delta.mutation_epoch.load(Ordering::Acquire)
                != index_delta.mutation_epoch_expected_even
            || index_delta.mutation_epoch_expected_even & 1 != 0
        {
            return Err(super::DeviceInsertPlanApplyError::PlanDrift);
        }

        index_delta.point_epoch_transition.arm_post_wal();
        index_delta
            .point_epoch_transition
            .start()
            .map_err(|_| super::DeviceInsertPlanApplyError::PlanDrift)?;

        let (payload_status, header) = fused.apply_payload_before_header()?;
        if payload_status.declined {
            return Err(super::DeviceInsertPlanApplyError::PublisherFailure);
        }
        let index_status = index_delta
            .launch
            .take()
            .ok_or(super::DeviceInsertPlanApplyError::PlanDrift)?
            .submit()
            .map_err(|_| super::DeviceInsertPlanApplyError::PublisherFailure)?;
        if index_status.declined {
            return Err(super::DeviceInsertPlanApplyError::PublisherFailure);
        }
        validate_tail_cache_current(engine, table, &index_delta)?;
        header
            .publish()
            .map_err(|_| super::DeviceInsertPlanApplyError::PublisherFailure)?;
        publish_tail_cache_after_header(engine, table, &index_delta, index_status.created_posting)?;
        append.publish_indexed_in_place_descriptor_after_fused_apply(engine, expected_commit)?;
        engine
            .read_state
            .residency
            .fused_apply_hits
            .fetch_add(1, Ordering::Relaxed);
        index_delta.point_epoch_transition.complete_started();
        Ok(())
    }
}

fn validate_tail_cache_current(
    engine: &Engine,
    table: &RelationalTable,
    delta: &PreparedResidentIndexDelta,
) -> Result<(), super::DeviceInsertPlanApplyError> {
    let _route_publish = engine
        .read_state
        .residency
        .sharded_point_route_publish_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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
    let complete = engine
        .read_state
        .residency
        .named_index_coverage_complete
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let publications = engine
        .read_state
        .residency
        .named_index_publications
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if publications.get(&table.oid) != Some(&table.indexes)
        || !complete
            .get(&table.name)
            .is_some_and(|(oid, indexes)| *oid == table.oid && indexes == &table.indexes)
    {
        return Err(super::DeviceInsertPlanApplyError::PlanDrift);
    }
    for binding in delta.physical_bindings.iter() {
        let Some(entry) = cache.get(&binding.cache_key) else {
            return Err(super::DeviceInsertPlanApplyError::PlanDrift);
        };
        if entry.resident_device_ptr != binding.source_ptr
            || entry.row_count != binding.base_row
            || entry.table_mask != binding.table_mask
            || entry.hash_shift != binding.hash_shift
            || !Arc::ptr_eq(&entry._resident_guard, &binding.resident_guard)
            || !entry
                .device_index
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &binding.device_index))
            || binding.published_row_count.load(Ordering::Acquire) != binding.base_row
            || binding.published_has_postings.load(Ordering::Acquire)
                != binding.has_postings_at_preview
            || coverage.get(&binding.cache_key) != Some(&(binding.source_ptr, binding.base_row))
        {
            return Err(super::DeviceInsertPlanApplyError::PlanDrift);
        }
    }
    Ok(())
}

fn publish_tail_cache_after_header(
    engine: &Engine,
    table: &RelationalTable,
    delta: &PreparedResidentIndexDelta,
    created_posting: bool,
) -> Result<(), super::DeviceInsertPlanApplyError> {
    // The preceding currentness check and the still-active common named-index lifecycle guard
    // make this a pure update of existing entries: no cache/map allocation, replacement build,
    // or second publication authority is permitted after the terminal header write.
    let _route_publish = engine
        .read_state
        .residency
        .sharded_point_route_publish_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut cache = engine
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut coverage = engine
        .read_state
        .residency
        .named_index_coverage
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if table.indexes.len() != delta.raw_bindings.len() {
        return Err(super::DeviceInsertPlanApplyError::PlanDrift);
    }
    for binding in delta.physical_bindings.iter() {
        let entry = cache
            .get_mut(&binding.cache_key)
            .ok_or(super::DeviceInsertPlanApplyError::PlanDrift)?;
        let covered = coverage
            .get_mut(&binding.cache_key)
            .ok_or(super::DeviceInsertPlanApplyError::PlanDrift)?;
        if entry.resident_device_ptr != binding.source_ptr
            || entry.row_count != binding.base_row
            || !entry
                .device_index
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &binding.device_index))
            || *covered != (binding.source_ptr, binding.base_row)
        {
            return Err(super::DeviceInsertPlanApplyError::PlanDrift);
        }
        if created_posting {
            binding
                .published_has_postings
                .store(true, Ordering::Relaxed);
            entry.has_postings = true;
        }
        binding
            .published_row_count
            .store(binding.end_row, Ordering::Release);
        entry.row_count = binding.end_row;
        *covered = (binding.source_ptr, binding.end_row);
    }
    Ok(())
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
            || delta.launch.as_ref().is_none_or(|launch| {
                launch.preparation_bytes() != delta.ledger.transient_preparation_bytes
            })
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
        None,
        0,
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
    preheld_budget_allocation: Option<std::sync::MutexGuard<'a, ()>>,
    prior_reserved_bytes: u64,
) -> Result<PreparedIndexedInPlaceReservation<'a>, ExecuteError> {
    let parts = preview.into_append_and_parts(
        engine,
        table,
        permit,
        preheld_budget_allocation,
        prior_reserved_bytes,
    )?;
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
    let point_epoch_transition = engine
        .read_state
        .residency
        .prepare_exact_point_index_mutation(
            Arc::clone(&mutation_epoch),
            mutation_epoch_expected_even,
        );
    let reservation = PreparedIndexedInPlaceReservation {
        fused,
        index_delta: PreparedResidentIndexDelta {
            launch: Some(index_delta),
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
            point_epoch_transition,
            ledger,
            host_retention_peak_before_finalization,
        },
        append,
        forecast: resource_forecast,
        fused_materialization_peak,
        target_witness,
        generation_witness,
        named_index_lifecycle: Some(named_index_lifecycle),
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
