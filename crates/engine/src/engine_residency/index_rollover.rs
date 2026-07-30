//! Physical ownership for one indexed fixed-width rollover reservation.
//!
//! This leaf builds a complete private payload plus every distinct physical index before WAL.
//! It exposes only scalar inspection, never a canonical operation, mutation launcher, cache
//! insertion, descriptor publication, or durability surface.

#![allow(dead_code)] // deliberately compiled reservation; live strategy selection remains closed

#[cfg(test)]
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::engine_insert_plan::batch_key_constraints::BatchKeyConstraintProof;
use crate::engine_insert_plan::host_retention::{HostRetentionGeometry, HostRetentionReport};
use crate::engine_insert_plan::{
    resident_key_constraints::ResidentKeyValidationSeal, IndexedPhysicalMaterializationPermit,
};
use crate::engine_state::TransactionNamedIndexPublicationGuard;
use crate::relational_model::{RelationalIndex, RelationalTable};
use crate::typed_insert_batch::TypedInsertBatch;
use crate::{Engine, ExecuteError, Index};
use gpu_db_execution::{
    multi_shard_i32_write_locate_resource_geometry, resident_index_allocated_bytes,
    resident_typed_index_build_host_scratch_geometry, resident_typed_index_build_preparation_bytes,
    CudaAllocationScope, CudaCompoundFoldColumn, CudaHostScratchGeometry, CudaResidentDeviceMemory,
};

use super::indexed_forecast::{
    IndexedPhysicalGenerationWitness, IndexedPhysicalResourceForecast, IndexedPhysicalTargetWitness,
};

#[path = "index_rollover/physical_probe.rs"]
mod physical_probe;

struct RawIndexLogicalBinding {
    raw_ordinal: usize,
    key_id: usize,
    physical_ordinal: usize,
}

struct PhysicalIndexLogicalBinding {
    raw_ordinal: usize,
    key_id: usize,
    descriptor_count: usize,
    duplicate_tolerant: bool,
}

struct PreparedFixedRolloverLogicalBindings {
    raw: Box<[RawIndexLogicalBinding]>,
    physical: Box<[PhysicalIndexLogicalBinding]>,
    max_concurrent_scratch_bytes: u64,
    descriptor_count: usize,
    max_descriptor_count: usize,
}

impl PreparedFixedRolloverLogicalBindings {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), ExecuteError> {
        report.retain_boxed_slice(&self.raw)?;
        report.retain_boxed_slice(&self.physical)?;
        Ok(())
    }

    fn host_retention_geometry(&self) -> Result<HostRetentionGeometry, ExecuteError> {
        fixed_rollover_logical_host_geometry(self.raw.len(), self.physical.len())
    }
}

struct PreparedPrivateIndexGeneration {
    raw_ordinal: usize,
    key_id: usize,
    memory: Arc<CudaResidentDeviceMemory>,
    table_mask: u32,
    hash_shift: u32,
    allocated_bytes: u64,
    descriptor_count: usize,
    /// Catalog-derived physical semantics retained for the future manifest publisher. This is
    /// independent of whether this particular zero-based build happened to create a posting.
    duplicate_tolerant: bool,
    created_posting: bool,
}

struct ReservedPrivateIndexGeneration {
    raw_ordinal: usize,
    key_id: usize,
    memory: Arc<CudaResidentDeviceMemory>,
    table_mask: u32,
    hash_shift: u32,
    allocated_bytes: u64,
    descriptor_count: usize,
    duplicate_tolerant: bool,
    build_columns: Vec<gpu_db_execution::CudaCompoundFoldColumn>,
}

struct FixedRolloverResourceLedger {
    payload_bytes: u64,
    created_by_bytes: u64,
    row_id_bytes: u64,
    payload_sidecar_persistent_bytes: u64,
    index_persistent_bytes: u64,
    total_persistent_bytes: u64,
    max_concurrent_scratch_bytes: u64,
    observed_scratch_peak_bytes: u64,
    bounded_readback_bytes: u64,
    max_concurrent_readback_bytes: u64,
    persistent_allocation_count: u64,
    raw_index_count: usize,
    physical_index_count: usize,
    descriptor_count: usize,
    max_descriptor_count: usize,
    gpu_probe_count: usize,
}

/// Scalar raw/physical geometry derived without building the materialized binding boxes.  The
/// O(n²) duplicate walk is intentional: this pre-capacity path must not allocate a map merely to
/// deduplicate a bounded catalog index list.
#[derive(Clone, Copy)]
struct FixedRolloverLogicalForecast {
    raw_index_count: usize,
    physical_index_count: usize,
    descriptor_count: usize,
    max_descriptor_count: usize,
    max_build_scratch_bytes: u64,
}

#[derive(Clone, Copy, Default)]
struct FixedRolloverHostPeakTracker {
    observed: HostRetentionGeometry,
}

impl FixedRolloverHostPeakTracker {
    fn observe(&mut self, report: HostRetentionReport) -> Result<(), ExecuteError> {
        self.observed = self.observed.peak(report.geometry()?);
        Ok(())
    }

    fn observe_with_disjoint_scratch(
        &mut self,
        report: HostRetentionReport,
        scratch: HostRetentionGeometry,
        domain: &'static str,
    ) -> Result<(), ExecuteError> {
        let mut geometry = report.geometry()?;
        geometry.checked_add_disjoint(scratch, domain)?;
        self.observed = self.observed.peak(geometry);
        Ok(())
    }

    fn peak(self) -> HostRetentionGeometry {
        self.observed
    }
}

/// Move-only fixed-rollover preview before any private CUDA work exists.  It retains only the
/// sealed source/proof/lock owners and old-generation Arcs needed to reject a witness drift.
pub(super) struct IndexedFixedRolloverPreview<'a> {
    source: crate::typed_insert_batch::PreparedResidentAppendSource,
    row_ids: super::DeviceInsertRowIds,
    table_name: Box<str>,
    table_oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
    predecessor_boundary: Index,
    catalog_seq: Index,
    predecessor_shard_id: u32,
    predecessor_row_start: usize,
    predecessor_row_count: usize,
    predecessor_capacity: usize,
    predecessor_payload: Arc<CudaResidentDeviceMemory>,
    predecessor_generation: Arc<()>,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    logical: FixedRolloverLogicalForecast,
    rollover: super::rollover::FixedRolloverGeometryForecast,
    target_witness: IndexedPhysicalTargetWitness,
    generation_witness: IndexedPhysicalGenerationWitness,
    resource_forecast: IndexedPhysicalResourceForecast,
    mutation_gate: std::sync::MutexGuard<'a, ()>,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
}

impl IndexedFixedRolloverPreview<'_> {
    pub(super) fn resource_forecast(&self) -> IndexedPhysicalResourceForecast {
        self.resource_forecast
    }

    pub(super) fn target_witness(&self) -> &IndexedPhysicalTargetWitness {
        &self.target_witness
    }

    pub(super) fn generation_witness(&self) -> &IndexedPhysicalGenerationWitness {
        &self.generation_witness
    }

    fn host_retention_report(&self) -> Result<HostRetentionReport, ExecuteError> {
        let mut report = self.source.host_retention_report()?;
        self.row_ids.append_host_allocation_slot(&mut report)?;
        report.retain_boxed_str(&self.table_name)?;
        self.key_proof.append_host_retention(&mut report)?;
        self.validation.append_host_retention(&mut report)?;
        Ok(report)
    }

    fn host_retention_geometry(&self) -> Result<HostRetentionGeometry, ExecuteError> {
        fixed_rollover_preview_host_geometry(
            &self.source,
            &self.row_ids,
            &self.table_name,
            &self.key_proof,
            &self.validation,
        )
    }
}

/// Scalar-only evidence from the private generation. Device pointers, allocations, CUDA state,
/// semantic carriers, and guards cannot cross this boundary.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexedFixedRolloverProofReport {
    pub(crate) raw_index_count: usize,
    pub(crate) physical_index_count: usize,
    pub(crate) predecessor_shard_id: u32,
    pub(crate) successor_shard_id: u32,
    pub(crate) successor_row_start: usize,
    pub(crate) incoming_rows: usize,
    pub(crate) capacity: usize,
    pub(crate) original_read_snapshot: Index,
    pub(crate) predecessor_boundary: Index,
    pub(crate) payload_bytes: u64,
    pub(crate) created_by_bytes: u64,
    pub(crate) row_id_bytes: u64,
    pub(crate) payload_sidecar_persistent_bytes: u64,
    pub(crate) index_persistent_bytes: u64,
    pub(crate) total_persistent_bytes: u64,
    pub(crate) max_concurrent_scratch_bytes: u64,
    pub(crate) observed_scratch_peak_bytes: u64,
    pub(crate) bounded_readback_bytes: u64,
    pub(crate) max_concurrent_readback_bytes: u64,
    pub(crate) persistent_allocation_count: u64,
    pub(crate) descriptor_count: usize,
    pub(crate) max_descriptor_count: usize,
    pub(crate) capacity_fit_evaluations: u64,
    pub(crate) budget_scan_entries: u64,
    pub(crate) gpu_build_count: usize,
    pub(crate) gpu_probe_count: usize,
    pub(crate) posting_index_count: usize,
    pub(crate) duplicate_tolerant_index_count: usize,
    pub(crate) private_header_zero: bool,
    pub(crate) final_host_retained_bytes: u64,
    pub(crate) final_host_allocation_slots: u64,
    pub(crate) final_host_generation_pin_slots: u64,
    pub(crate) observed_peak_host_retained_bytes: u64,
    pub(crate) observed_peak_host_allocation_slots: u64,
    pub(crate) observed_peak_host_generation_pin_slots: u64,
    pub(crate) forecast_final_host_retained_bytes: u64,
    pub(crate) forecast_final_host_allocation_slots: u64,
    pub(crate) forecast_peak_host_retained_bytes: u64,
    pub(crate) forecast_peak_host_allocation_slots: u64,
    pub(crate) concrete_generation_box_bytes: u64,
    pub(crate) concrete_generation_box_slots: u64,
}

/// Every completed zero-based build and its exact proof ledger. Index allocations drop before the
/// private payload owner, so no index can retain a source pointer after rollover abandonment.
struct PreparedPrivateRolloverIndexes {
    generations: Box<[PreparedPrivateIndexGeneration]>,
    source_payload: Arc<CudaResidentDeviceMemory>,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    raw_bindings: Box<[RawIndexLogicalBinding]>,
    mutation_epoch: Arc<AtomicU64>,
    mutation_epoch_expected_even: u64,
    ledger: FixedRolloverResourceLedger,
}

impl PreparedPrivateRolloverIndexes {
    fn host_retention_report(&self) -> Result<HostRetentionReport, ExecuteError> {
        let mut report = HostRetentionReport::default();
        report.retain_boxed_slice(&self.generations)?;
        self.key_proof.append_host_retention(&mut report)?;
        self.validation.append_host_retention(&mut report)?;
        report.retain_boxed_slice(&self.raw_bindings)?;
        report.retain_arc_owner(&self.mutation_epoch)?;
        Ok(report)
    }

    #[cfg(test)]
    fn host_retention_report_without_generation_box(
        &self,
    ) -> Result<HostRetentionReport, ExecuteError> {
        let mut report = HostRetentionReport::default();
        self.key_proof.append_host_retention(&mut report)?;
        self.validation.append_host_retention(&mut report)?;
        report.retain_boxed_slice(&self.raw_bindings)?;
        report.retain_arc_owner(&self.mutation_epoch)?;
        Ok(report)
    }
}

/// Move-only physical reservation. Declaration order is load-bearing: private indexes drop first, then the
/// payload/sidecar append reservation, then the named-index lifecycle guard.
pub(super) struct PreparedIndexedFixedRolloverReservation<'a> {
    indexes: PreparedPrivateRolloverIndexes,
    append: super::fixed_insert::ResidentOpenShardAppendPlan<'a>,
    forecast: IndexedPhysicalResourceForecast,
    target_witness: IndexedPhysicalTargetWitness,
    generation_witness: IndexedPhysicalGenerationWitness,
    old_generation_pinned_bytes: u64,
    generation_pin_slots: u64,
    host_retention_peak_before_finalization: HostRetentionGeometry,
    #[cfg(test)]
    report: IndexedFixedRolloverProofReport,
    _named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
}

impl PreparedIndexedFixedRolloverReservation<'_> {
    fn host_retention_report(&self) -> Result<HostRetentionReport, ExecuteError> {
        let mut report = self.append.host_retention_report()?;
        report.merge(self.indexes.host_retention_report()?)?;
        Ok(report)
    }

    fn forecast_matches_actual(&self) -> Result<bool, ExecuteError> {
        let basis = self
            .append
            .indexed_fixed_rollover_reservation_basis()
            .ok_or_else(|| decline("indexed rollover forecast lost materialized basis"))?;
        let host = self.host_retention_report()?.geometry()?;
        let peak = self.host_retention_peak_before_finalization.peak(host);
        let ledger = &self.indexes.ledger;
        let actual = IndexedPhysicalResourceForecast {
            final_host_retained_bytes: host.retained_bytes(),
            final_host_allocation_slots: host.allocation_slots(),
            final_host_generation_pin_slots: host.generation_pin_slots(),
            peak_host_retained_bytes: peak.retained_bytes(),
            peak_host_allocation_slots: peak.allocation_slots(),
            peak_host_generation_pin_slots: peak.generation_pin_slots(),
            old_generation_pinned_bytes: self.old_generation_pinned_bytes,
            new_persistent_bytes: ledger.total_persistent_bytes,
            retained_device_transient_bytes: 0,
            retained_device_result_bytes: 0,
            incremental_allocation_slots: ledger.persistent_allocation_count,
            generation_pin_slots: self.generation_pin_slots,
            maximum_concurrent_device_scratch_bytes: ledger.max_concurrent_scratch_bytes,
            maximum_host_readback_bytes: ledger.max_concurrent_readback_bytes,
        };
        Ok(actual == self.forecast
            && self.target_witness.gpu_id == self.indexes.source_payload.metadata().gpu_id
            && self.generation_witness.catalog_seq == basis.catalog_seq
            && self.generation_witness.open_shard_id == basis.predecessor_shard_id
            && self.generation_witness.row_start
                == u64::try_from(
                    basis
                        .new_row_start
                        .saturating_sub(basis.predecessor_row_count),
                )
                .unwrap_or(u64::MAX)
            && self.generation_witness.row_count
                == u64::try_from(basis.predecessor_row_count).unwrap_or(u64::MAX)
            && self.generation_witness.capacity
                == u64::try_from(self.predecessor_capacity_for_basis(basis.capacity))
                    .unwrap_or(u64::MAX)
            && self.generation_witness.index_mutation_epoch_even
                == self.indexes.mutation_epoch_expected_even)
    }

    fn predecessor_capacity_for_basis(&self, _successor_capacity: usize) -> usize {
        self.generation_witness
            .capacity
            .try_into()
            .unwrap_or(usize::MAX)
    }
}

#[cfg(test)]
impl PreparedIndexedFixedRolloverReservation<'_> {
    pub(super) fn inspect<R>(
        self,
        inspect: impl FnOnce(IndexedFixedRolloverProofReport) -> R,
    ) -> Result<R, ExecuteError> {
        let report = self.scalar_report_if_intact()?;
        Ok(inspect(report))
    }

    fn scalar_report_if_intact(&self) -> Result<IndexedFixedRolloverProofReport, ExecuteError> {
        let basis = self
            .append
            .indexed_fixed_rollover_reservation_basis()
            .ok_or_else(|| decline("indexed rollover proof lost its fixed private generation"))?;
        let indexes = &self.indexes;
        let ledger = &indexes.ledger;
        let final_host = self.host_retention_report()?.geometry()?;
        let observed_peak = self
            .host_retention_peak_before_finalization
            .peak(final_host);
        let mut without_generation_box = self.append.host_retention_report()?;
        without_generation_box.merge(
            self.indexes
                .host_retention_report_without_generation_box()?,
        )?;
        let without_generation_box = without_generation_box.geometry()?;
        let mut report = self.report;
        report.final_host_retained_bytes = final_host.retained_bytes();
        report.final_host_allocation_slots = final_host.allocation_slots();
        report.final_host_generation_pin_slots = final_host.generation_pin_slots();
        report.observed_peak_host_retained_bytes = observed_peak.retained_bytes();
        report.observed_peak_host_allocation_slots = observed_peak.allocation_slots();
        report.observed_peak_host_generation_pin_slots = observed_peak.generation_pin_slots();
        report.forecast_final_host_retained_bytes = self.forecast.final_host_retained_bytes;
        report.forecast_final_host_allocation_slots = self.forecast.final_host_allocation_slots;
        report.forecast_peak_host_retained_bytes = self.forecast.peak_host_retained_bytes;
        report.forecast_peak_host_allocation_slots = self.forecast.peak_host_allocation_slots;
        report.concrete_generation_box_bytes = final_host
            .retained_bytes()
            .checked_sub(without_generation_box.retained_bytes())
            .ok_or_else(|| decline("indexed rollover generation-box byte report underflowed"))?;
        report.concrete_generation_box_slots = final_host
            .allocation_slots()
            .checked_sub(without_generation_box.allocation_slots())
            .ok_or_else(|| decline("indexed rollover generation-box slot report underflowed"))?;
        if !self.append.holds_budget_reservation()
            || basis.predecessor_shard_id != self.report.predecessor_shard_id
            || basis.new_shard_id != self.report.successor_shard_id
            || basis.new_row_start != self.report.successor_row_start
            || basis.incoming_rows != self.report.incoming_rows
            || basis.capacity != self.report.capacity
            || basis.planned_index_allocation_bytes != ledger.index_persistent_bytes
            || basis.max_index_scratch_bytes != ledger.max_concurrent_scratch_bytes
            || !Arc::ptr_eq(basis.payload, &indexes.source_payload)
            || indexes.source_payload.device_ptr() == 0
            || indexes.mutation_epoch.load(Ordering::Acquire)
                != indexes.mutation_epoch_expected_even
            || indexes.mutation_epoch_expected_even & 1 != 0
            || indexes.key_proof.indexes().len() != indexes.raw_bindings.len()
            || indexes.validation.original_read_snapshot() != self.report.original_read_snapshot
            || indexes.validation.predecessor_boundary() != self.report.predecessor_boundary
            || indexes
                .source_payload
                .read_resident_u64_column(0, 1)
                .map_err(|_| decline("indexed rollover proof private header readback failed"))?
                != [0]
        {
            return Err(decline(
                "indexed rollover proof retained resource integrity drifted",
            ));
        }

        let mut distinct_destinations = BTreeSet::new();
        let mut actual_index_bytes = 0_u64;
        let mut descriptor_count = 0_usize;
        for generation in indexes.generations.iter() {
            if generation.raw_ordinal >= indexes.key_proof.indexes().len()
                || generation.memory.device_ptr() == indexes.source_payload.device_ptr()
                || !distinct_destinations.insert(generation.memory.device_ptr())
                || generation.memory.metadata().allocated_bytes != generation.allocated_bytes
                || generation.table_mask == 0
                || generation.hash_shift
                    != 32 - (u64::from(generation.table_mask) + 1).trailing_zeros()
                || generation.descriptor_count == 0
            {
                return Err(decline(
                    "indexed rollover proof retained private index drifted",
                ));
            }
            actual_index_bytes = actual_index_bytes
                .checked_add(generation.allocated_bytes)
                .ok_or_else(|| decline("indexed rollover proof index byte sum overflows"))?;
            descriptor_count = descriptor_count
                .checked_add(generation.descriptor_count)
                .ok_or_else(|| decline("indexed rollover proof descriptor sum overflows"))?;
        }
        for (raw_ordinal, raw) in indexes.raw_bindings.iter().enumerate() {
            let physical = indexes
                .generations
                .get(raw.physical_ordinal)
                .ok_or_else(|| decline("indexed rollover proof lost raw-to-physical mapping"))?;
            if raw.raw_ordinal != raw_ordinal
                || raw.key_id != physical.key_id
                || indexes
                    .key_proof
                    .indexes()
                    .get(raw_ordinal)
                    .is_none_or(|binding| binding.raw_ordinal() != raw_ordinal)
            {
                return Err(decline("indexed rollover proof raw mapping drifted"));
            }
        }
        let expected_total = basis
            .payload_sidecar_allocation_bytes
            .checked_add(actual_index_bytes)
            .ok_or_else(|| decline("indexed rollover proof persistent byte sum overflows"))?;
        let expected_payload_sidecars = basis
            .payload_bytes
            .checked_add(basis.created_by_bytes)
            .and_then(|bytes| bytes.checked_add(basis.row_id_bytes))
            .ok_or_else(|| decline("indexed rollover proof payload/sidecar bytes overflow"))?;
        if actual_index_bytes != ledger.index_persistent_bytes
            || expected_total != ledger.total_persistent_bytes
            || expected_payload_sidecars != basis.payload_sidecar_allocation_bytes
            || ledger.payload_bytes != basis.payload_bytes
            || ledger.created_by_bytes != basis.created_by_bytes
            || ledger.row_id_bytes != basis.row_id_bytes
            || basis.payload_sidecar_allocation_bytes != ledger.payload_sidecar_persistent_bytes
            || basis
                .payload_sidecar_allocation_count
                .checked_add(indexes.generations.len() as u64)
                != Some(ledger.persistent_allocation_count)
            || ledger.raw_index_count != indexes.raw_bindings.len()
            || ledger.physical_index_count != indexes.generations.len()
            || descriptor_count != ledger.descriptor_count
            || indexes
                .generations
                .iter()
                .map(|generation| generation.descriptor_count)
                .max()
                .unwrap_or(0)
                != ledger.max_descriptor_count
            || ledger.max_concurrent_scratch_bytes != ledger.observed_scratch_peak_bytes
            || physical_probe::expected_resource_ledger(
                indexes.generations.len(),
                self.report.incoming_rows,
            ) != Some((
                ledger.bounded_readback_bytes,
                ledger.max_concurrent_readback_bytes,
            ))
            || self.report.payload_sidecar_persistent_bytes
                != ledger.payload_sidecar_persistent_bytes
            || self.report.payload_bytes != ledger.payload_bytes
            || self.report.created_by_bytes != ledger.created_by_bytes
            || self.report.row_id_bytes != ledger.row_id_bytes
            || self.report.index_persistent_bytes != ledger.index_persistent_bytes
            || self.report.total_persistent_bytes != ledger.total_persistent_bytes
            || self.report.max_concurrent_scratch_bytes != ledger.max_concurrent_scratch_bytes
            || self.report.observed_scratch_peak_bytes != ledger.observed_scratch_peak_bytes
            || self.report.bounded_readback_bytes != ledger.bounded_readback_bytes
            || self.report.max_concurrent_readback_bytes != ledger.max_concurrent_readback_bytes
            || self.report.persistent_allocation_count != ledger.persistent_allocation_count
            || self.report.raw_index_count != ledger.raw_index_count
            || self.report.physical_index_count != ledger.physical_index_count
            || self.report.descriptor_count != ledger.descriptor_count
            || self.report.max_descriptor_count != ledger.max_descriptor_count
            || self.report.capacity_fit_evaluations != basis.capacity_fit_evaluations
            || self.report.budget_scan_entries != basis.budget_scan_entries
            || self.report.gpu_build_count != indexes.generations.len()
            || self.report.gpu_probe_count != ledger.gpu_probe_count
            || ledger.gpu_probe_count != indexes.generations.len()
            || self.report.posting_index_count
                != indexes
                    .generations
                    .iter()
                    .filter(|generation| generation.created_posting)
                    .count()
            || self.report.duplicate_tolerant_index_count
                != indexes
                    .generations
                    .iter()
                    .filter(|generation| generation.duplicate_tolerant)
                    .count()
            || !self.report.private_header_zero
            || report.final_host_retained_bytes != report.forecast_final_host_retained_bytes
            || report.final_host_allocation_slots != report.forecast_final_host_allocation_slots
            || report.observed_peak_host_retained_bytes != report.forecast_peak_host_retained_bytes
            || report.observed_peak_host_allocation_slots
                != report.forecast_peak_host_allocation_slots
            || report.concrete_generation_box_bytes
                != u64::try_from(indexes.generations.len())
                    .ok()
                    .and_then(|count| {
                        count.checked_mul(
                            std::mem::size_of::<PreparedPrivateIndexGeneration>() as u64
                        )
                    })
                    .unwrap_or(u64::MAX)
            || report.concrete_generation_box_slots != 1
        {
            return Err(decline("indexed rollover proof retained ledger drifted"));
        }
        Ok(report)
    }
}

/// Prepare only the fixed-rollover metadata and exact scalar capacity forecast.  No CUDA
/// allocation, driver call, lookup oracle, private payload, sidecar, or index build is permitted
/// before the resulting owner consumes an `engine_insert_plan` materialization permit.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_fixed_rollover_preview<'a>(
    engine: &'a Engine,
    table: &RelationalTable,
    predecessor_boundary: Index,
    batch: TypedInsertBatch,
    row_ids: super::DeviceInsertRowIds,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    mutation_gate: std::sync::MutexGuard<'a, ()>,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
    _commit_proof: &std::sync::MutexGuard<'_, crate::CommitState>,
) -> Result<IndexedFixedRolloverPreview<'a>, ExecuteError> {
    let source_retention_prediction = batch
        .resident_append_source_host_retention_prediction()
        .map_err(|_| decline("indexed rollover forecast source retention prediction declined"))?;
    let source = batch
        .into_resident_append_source()
        .ok_or_else(|| decline("indexed rollover forecast lost its resident append source"))?;
    if source_retention_prediction != source.host_retention_geometry()? {
        return Err(decline(
            "indexed rollover forecast source retention materialization drifted",
        ));
    }
    let catalog = engine.catalog_snapshot();
    let current_table = catalog
        .relational_catalog
        .get(&table.name)
        .ok_or_else(|| decline("indexed rollover forecast lost current table catalog"))?;
    if current_table != table
        || catalog.commit_seq < source.prepared_catalog_seq()
        || !super::fixed_insert::source_matches_indexed_in_place_reservation(&source, table)
        || source.requires_dense_rollover()
        || source.row_count() == 0
        || !row_ids.is_exact()
        || !row_ids.exact_len_matches(source.row_count())
    {
        return Err(decline(
            "indexed rollover forecast source is not fixed eligible",
        ));
    }
    let logical = fixed_rollover_logical_forecast(table, &key_proof)?;
    let shards = engine.read_state.residency.shards.load_full();
    let open = shards
        .get(&table.name)
        .and_then(|shards| shards.last())
        .ok_or_else(|| decline("indexed rollover forecast has no current open shard"))?;
    if open
        .row_count
        .checked_add(source.row_count())
        .is_none_or(|end| end <= open.capacity)
        || !validation.matches_fixed_rollover_predecessor(
            table,
            catalog.commit_seq,
            open,
            predecessor_boundary,
        )
        || validation.original_read_snapshot() > predecessor_boundary
        || !physical_probe::published_index_enrollment_is_complete(engine, table)
    {
        return Err(decline(
            "indexed rollover forecast predecessor witness drifted",
        ));
    }
    let payload = open
        .device_memory
        .as_ref()
        .cloned()
        .filter(|payload| payload.device_ptr() != 0)
        .ok_or_else(|| decline("indexed rollover forecast lost predecessor payload"))?;
    let lookup_geometry = multi_shard_i32_write_locate_resource_geometry(
        1,
        source.row_count(),
        u32::try_from(source.row_count())
            .map_err(|_| decline("indexed rollover forecast row count overflows"))?,
    )
    .ok_or_else(|| decline("indexed rollover forecast lookup geometry overflows"))?;
    let build_host_scratch =
        resident_typed_index_build_host_scratch_geometry(logical.max_descriptor_count)
            .ok_or_else(|| decline("indexed rollover forecast build host scratch overflows"))?;
    let lookup_host_peak = CudaHostScratchGeometry {
        bytes: lookup_geometry.host_peak_bytes,
        allocation_slots: lookup_geometry.host_peak_allocation_slots,
    };
    let max_scratch = logical
        .max_build_scratch_bytes
        .max(lookup_geometry.preparation_bytes);
    let (resident_bytes, _) = engine.relational_resident_bytes_and_entries_for_gpu(open.gpu_id);
    let remaining_budget = match engine.relational_residency_budget_bytes(open.gpu_id) {
        Some(budget) => Some(budget.checked_sub(resident_bytes).ok_or_else(|| {
            decline("indexed rollover forecast resident budget already exceeded")
        })?),
        None => None,
    };
    let persistent_budget =
        match remaining_budget {
            Some(remaining) => Some(remaining.checked_sub(max_scratch).ok_or_else(|| {
                decline("indexed rollover forecast scratch exceeds resident budget")
            })?),
            None => None,
        };
    let desired = super::rollover::ResidentRolloverPlan::fixed_width_desired_capacity(
        source.row_count(),
        Some(engine.shard_size_target()),
    )
    .ok_or_else(|| decline("indexed rollover forecast desired capacity overflows"))?;
    let rollover = super::rollover::ResidentRolloverPlan::fixed_width_null_free_table_forecast(
        table,
        source.row_count(),
        desired,
        true,
        true,
        persistent_budget,
    )?
    .ok_or_else(|| decline("indexed rollover forecast capacity does not fit"))?;
    let one_index_bytes =
        resident_index_bytes_for_rollover(source.row_count(), rollover.capacity())?;
    let index_bytes = one_index_bytes
        .checked_mul(
            u64::try_from(logical.physical_index_count)
                .map_err(|_| decline("indexed rollover forecast physical index count overflows"))?,
        )
        .ok_or_else(|| decline("indexed rollover forecast index bytes overflow"))?;
    if index_bytes != rollover.named_index_bytes() {
        return Err(decline("indexed rollover forecast index geometry drifted"));
    }
    let (old_generation_pinned_bytes, generation_pin_slots) =
        old_generation_pinned_ledger(engine, table, open, &payload)?;
    let append_host = super::fixed_insert::indexed_fixed_rollover_append_host_retention_forecast(
        &source, &row_ids, open, table,
    )
    .map_err(|_| decline("indexed rollover forecast append host geometry declined"))?;
    let append_host_scratch =
        super::fixed_insert::indexed_fixed_rollover_append_host_scratch_forecast(
            &source, &row_ids, table,
        )
        .map_err(|_| decline("indexed rollover forecast append host scratch declined"))?;
    let new_persistent_bytes = rollover
        .allocation_bytes_before_indexes()
        .and_then(|bytes| bytes.checked_add(index_bytes))
        .ok_or_else(|| decline("indexed rollover forecast persistent bytes overflow"))?;
    let payload_sidecar_slots = 2_u64
        .checked_add(u64::from(rollover.row_id_bytes() != 0))
        .ok_or_else(|| decline("indexed rollover forecast sidecar slots overflow"))?;
    let incremental_allocation_slots = payload_sidecar_slots
        .checked_add(
            u64::try_from(logical.physical_index_count)
                .map_err(|_| decline("indexed rollover forecast allocation slots overflow"))?,
        )
        .ok_or_else(|| decline("indexed rollover forecast allocation slots overflow"))?;
    let (_bounded_readback, max_readback) =
        physical_probe::expected_resource_ledger(logical.physical_index_count, source.row_count())
            .ok_or_else(|| decline("indexed rollover forecast readback geometry overflows"))?;
    let (final_host, peak_host) = fixed_rollover_peak_host_geometry(
        &source,
        &row_ids,
        &table.name,
        &key_proof,
        &validation,
        logical,
        append_host,
        append_host_scratch,
        build_host_scratch,
        lookup_host_peak,
    )?;
    let schema_digest = crate::engine_transaction_reset::table_schema_digest(table)
        .map_err(|_| decline("indexed rollover forecast schema digest declined"))?;
    let mutation_epoch = engine
        .read_state
        .residency
        .point_index_mutation_epoch_for_table(&engine.read_state, table)
        .ok_or_else(|| decline("indexed rollover forecast relation identity changed"))?;
    let mutation_epoch_even = mutation_epoch.load(Ordering::Acquire);
    if mutation_epoch_even & 1 != 0 {
        return Err(decline(
            "indexed rollover forecast observed active mutation epoch",
        ));
    }
    Ok(IndexedFixedRolloverPreview {
        source,
        row_ids,
        table_name: Box::from(table.name.as_str()),
        table_oid: table.oid,
        schema_digest,
        predecessor_boundary,
        catalog_seq: catalog.commit_seq,
        predecessor_shard_id: open.shard_id,
        predecessor_row_start: open.row_start,
        predecessor_row_count: open.row_count,
        predecessor_capacity: open.capacity,
        predecessor_payload: payload,
        predecessor_generation: Arc::clone(&open.point_route_generation),
        key_proof,
        validation,
        logical,
        rollover,
        target_witness: IndexedPhysicalTargetWitness {
            gpu_id: open.gpu_id,
            table_oid: table.oid,
            schema_digest,
        },
        generation_witness: IndexedPhysicalGenerationWitness {
            catalog_seq: catalog.commit_seq,
            predecessor_boundary,
            open_shard_id: open.shard_id,
            row_start: u64::try_from(open.row_start)
                .map_err(|_| decline("indexed rollover forecast row start overflows"))?,
            row_count: u64::try_from(open.row_count)
                .map_err(|_| decline("indexed rollover forecast row count overflows"))?,
            capacity: u64::try_from(open.capacity)
                .map_err(|_| decline("indexed rollover forecast capacity overflows"))?,
            index_mutation_epoch_even: mutation_epoch_even,
        },
        resource_forecast: IndexedPhysicalResourceForecast {
            final_host_retained_bytes: final_host.retained_bytes(),
            final_host_allocation_slots: final_host.allocation_slots(),
            final_host_generation_pin_slots: final_host.generation_pin_slots(),
            peak_host_retained_bytes: peak_host.retained_bytes(),
            peak_host_allocation_slots: peak_host.allocation_slots(),
            peak_host_generation_pin_slots: peak_host.generation_pin_slots(),
            old_generation_pinned_bytes,
            new_persistent_bytes,
            retained_device_transient_bytes: 0,
            retained_device_result_bytes: 0,
            incremental_allocation_slots,
            generation_pin_slots,
            maximum_concurrent_device_scratch_bytes: max_scratch,
            maximum_host_readback_bytes: max_readback,
        },
        mutation_gate,
        named_index_lifecycle,
    })
}

/// Test-only construction of the shared opaque forecast. The returned owner has performed no
/// private CUDA allocation or launch; only the terminal pre-WAL carrier can issue its later
/// materialization permit.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_indexed_fixed_rollover_forecast<'a>(
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
) -> Result<super::indexed_forecast::PreparedIndexedPhysicalForecast<'a>, ExecuteError> {
    prepare_fixed_rollover_preview(
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
    )
    .map(super::indexed_forecast::PreparedIndexedPhysicalForecast::fixed_rollover)
}

fn fixed_rollover_logical_forecast(
    table: &RelationalTable,
    keys: &BatchKeyConstraintProof,
) -> Result<FixedRolloverLogicalForecast, ExecuteError> {
    if table.indexes.is_empty() || keys.indexes().len() != table.indexes.len() {
        return Err(decline(
            "indexed rollover forecast lost exact raw catalog enrollment",
        ));
    }
    let mut physical_index_count = 0_usize;
    let mut descriptor_count = 0_usize;
    let mut max_descriptor_count = 0_usize;
    let mut max_build_scratch_bytes = 0_u64;
    for (raw_ordinal, index) in table.indexes.iter().enumerate() {
        let binding = keys
            .indexes()
            .get(raw_ordinal)
            .filter(|binding| binding.raw_ordinal() == raw_ordinal)
            .ok_or_else(|| decline("indexed rollover forecast lost raw catalog binding"))?;
        if !super::index_all_key_columns_foldable(table, index) {
            return Err(decline(
                "indexed rollover forecast found non-foldable named index",
            ));
        }
        let key_id = super::index_probe_key_id(table, index, raw_ordinal)
            .ok_or_else(|| decline("indexed rollover forecast found no key id"))?;
        let descriptor = binding.key_column_count();
        if descriptor == 0 {
            return Err(decline("indexed rollover forecast found empty descriptor"));
        }
        let mut prior_descriptor = None;
        for prior in 0..raw_ordinal {
            let prior_index = table
                .indexes
                .get(prior)
                .ok_or_else(|| decline("indexed rollover forecast prior index drifted"))?;
            if super::index_probe_key_id(table, prior_index, prior) == Some(key_id) {
                let prior_binding = keys
                    .indexes()
                    .get(prior)
                    .filter(|binding| binding.raw_ordinal() == prior)
                    .ok_or_else(|| decline("indexed rollover forecast prior binding drifted"))?;
                prior_descriptor = Some(prior_binding.key_column_count());
                break;
            }
        }
        if let Some(prior_descriptor) = prior_descriptor {
            if prior_descriptor != descriptor {
                return Err(decline(
                    "indexed rollover forecast duplicate descriptor geometry drifted",
                ));
            }
            continue;
        }
        physical_index_count = physical_index_count
            .checked_add(1)
            .ok_or_else(|| decline("indexed rollover forecast physical count overflows"))?;
        descriptor_count = descriptor_count
            .checked_add(descriptor)
            .ok_or_else(|| decline("indexed rollover forecast descriptor count overflows"))?;
        max_descriptor_count = max_descriptor_count.max(descriptor);
        max_build_scratch_bytes = max_build_scratch_bytes.max(
            resident_typed_index_build_preparation_bytes(descriptor)
                .ok_or_else(|| decline("indexed rollover forecast scratch geometry overflows"))?,
        );
    }
    if physical_index_count == 0 || max_build_scratch_bytes == 0 {
        return Err(decline("indexed rollover forecast has no physical build"));
    }
    Ok(FixedRolloverLogicalForecast {
        raw_index_count: table.indexes.len(),
        physical_index_count,
        descriptor_count,
        max_descriptor_count,
        max_build_scratch_bytes,
    })
}

fn resident_index_bytes_for_rollover(rows: usize, capacity: usize) -> Result<u64, ExecuteError> {
    let table_size = super::resident_shard_index_table_size(
        u64::try_from(rows).map_err(|_| decline("indexed rollover forecast rows overflow"))?,
        u64::try_from(capacity)
            .map_err(|_| decline("indexed rollover forecast capacity overflow"))?,
    )
    .ok_or_else(|| decline("indexed rollover forecast index horizon overflows"))?;
    resident_index_allocated_bytes(
        u32::try_from(table_size - 1)
            .map_err(|_| decline("indexed rollover forecast table mask overflows"))?,
        u64::try_from(capacity.max(rows))
            .map_err(|_| decline("indexed rollover forecast index capacity overflows"))?,
    )
    .ok_or_else(|| decline("indexed rollover forecast index bytes overflow"))
}

fn old_generation_pinned_ledger(
    engine: &Engine,
    table: &RelationalTable,
    open: &crate::RelationalResidentShard,
    payload: &Arc<CudaResidentDeviceMemory>,
) -> Result<(u64, u64), ExecuteError> {
    let direct_regions = [
        Some(payload.as_ref()),
        open.created_by_region.as_deref(),
        open.row_id_region.as_deref(),
        open.deleted_by_region.as_deref(),
    ];
    let mut bytes = 0_u64;
    let mut slots = 0_u64;
    for (position, memory) in direct_regions.iter().enumerate() {
        let Some(memory) = memory else {
            continue;
        };
        if direct_regions[..position]
            .iter()
            .flatten()
            .any(|prior| prior.allocation_identity() == memory.allocation_identity())
        {
            continue;
        }
        bytes = bytes
            .checked_add(memory.metadata().allocated_bytes)
            .ok_or_else(|| decline("indexed rollover forecast old generation bytes overflow"))?;
        slots = slots
            .checked_add(1)
            .ok_or_else(|| decline("indexed rollover forecast old generation slots overflow"))?;
    }
    let cache = engine
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (raw_ordinal, index) in table.indexes.iter().enumerate() {
        let key_id = super::index_probe_key_id(table, index, raw_ordinal)
            .ok_or_else(|| decline("indexed rollover forecast old index key drifted"))?;
        if (0..raw_ordinal).any(|prior| {
            table.indexes.get(prior).is_some_and(|prior_index| {
                super::index_probe_key_id(table, prior_index, prior) == Some(key_id)
            })
        }) {
            continue;
        }
        let entry = cache
            .iter()
            .find(|((name, shard_id, entry_key), _)| {
                name == &table.name && *shard_id == open.shard_id && *entry_key == key_id
            })
            .map(|(_, entry)| entry)
            .ok_or_else(|| decline("indexed rollover forecast lost old index pin"))?;
        let memory = entry
            .device_index
            .as_deref()
            .ok_or_else(|| decline("indexed rollover forecast old index allocation missing"))?;
        if memory.device_ptr() == payload.device_ptr() {
            return Err(decline(
                "indexed rollover forecast old index aliases payload",
            ));
        }
        bytes = bytes
            .checked_add(memory.metadata().allocated_bytes)
            .ok_or_else(|| decline("indexed rollover forecast old index bytes overflow"))?;
        slots = slots
            .checked_add(1)
            .ok_or_else(|| decline("indexed rollover forecast old index slots overflow"))?;
    }
    Ok((bytes, slots))
}

fn fixed_rollover_preview_host_geometry(
    source: &crate::typed_insert_batch::PreparedResidentAppendSource,
    row_ids: &super::DeviceInsertRowIds,
    table_name: &str,
    key_proof: &BatchKeyConstraintProof,
    validation: &ResidentKeyValidationSeal,
) -> Result<HostRetentionGeometry, ExecuteError> {
    let mut geometry = source.host_retention_geometry()?;
    geometry.checked_add_disjoint(
        row_ids.host_allocation_geometry()?,
        "indexed rollover preview row-id owner",
    )?;
    geometry.checked_add_backing_elements::<u8>(
        table_name.len(),
        "indexed rollover preview table name",
    )?;
    geometry.checked_add_disjoint(
        key_proof.host_retention_geometry()?,
        "indexed rollover preview key proof",
    )?;
    geometry.checked_add_disjoint(
        validation.host_retention_geometry()?,
        "indexed rollover preview validation seal",
    )?;
    Ok(geometry)
}

fn fixed_rollover_logical_host_geometry(
    raw_index_count: usize,
    physical_index_count: usize,
) -> Result<HostRetentionGeometry, ExecuteError> {
    let mut geometry = HostRetentionGeometry::default();
    geometry.checked_add_backing_elements::<RawIndexLogicalBinding>(
        raw_index_count,
        "indexed rollover logical raw binding box",
    )?;
    geometry.checked_add_backing_elements::<PhysicalIndexLogicalBinding>(
        physical_index_count,
        "indexed rollover logical physical binding box",
    )?;
    Ok(geometry)
}

fn fixed_rollover_append_base_host_geometry(
    source: &crate::typed_insert_batch::PreparedResidentAppendSource,
    append_host: HostRetentionGeometry,
    key_proof: &BatchKeyConstraintProof,
    validation: &ResidentKeyValidationSeal,
) -> Result<HostRetentionGeometry, ExecuteError> {
    let mut geometry = source.host_retention_geometry()?;
    geometry.checked_add_disjoint(append_host, "indexed rollover append plan owner")?;
    geometry.checked_add_disjoint(
        key_proof.host_retention_geometry()?,
        "indexed rollover append key proof",
    )?;
    geometry.checked_add_disjoint(
        validation.host_retention_geometry()?,
        "indexed rollover append validation seal",
    )?;
    Ok(geometry)
}

fn fixed_rollover_materialized_host_report(
    append: &super::fixed_insert::ResidentOpenShardAppendPlan<'_>,
    key_proof: &BatchKeyConstraintProof,
    validation: &ResidentKeyValidationSeal,
    logical: &PreparedFixedRolloverLogicalBindings,
    oracle: &physical_probe::PrivateGpuLookupOracle,
    mutation_epoch: Option<&Arc<AtomicU64>>,
) -> Result<HostRetentionReport, ExecuteError> {
    let mut report = append.host_retention_report()?;
    key_proof.append_host_retention(&mut report)?;
    validation.append_host_retention(&mut report)?;
    logical.append_host_retention(&mut report)?;
    oracle.append_host_retention(&mut report)?;
    if let Some(epoch) = mutation_epoch {
        report.retain_arc_owner(epoch)?;
    }
    Ok(report)
}

fn fixed_rollover_final_host_geometry(
    source: &crate::typed_insert_batch::PreparedResidentAppendSource,
    key_proof: &BatchKeyConstraintProof,
    validation: &ResidentKeyValidationSeal,
    raw_index_count: usize,
    physical_index_count: usize,
    append_host: HostRetentionGeometry,
) -> Result<HostRetentionGeometry, ExecuteError> {
    let mut geometry =
        fixed_rollover_append_base_host_geometry(source, append_host, key_proof, validation)?;
    geometry.checked_add_backing_elements::<RawIndexLogicalBinding>(
        raw_index_count,
        "indexed rollover final raw binding box",
    )?;
    geometry.checked_add_backing_elements::<PreparedPrivateIndexGeneration>(
        physical_index_count,
        "indexed rollover final generation box",
    )?;
    geometry.checked_add_backing_elements::<AtomicU64>(
        1,
        "indexed rollover final mutation epoch owner",
    )?;
    Ok(geometry)
}

#[allow(clippy::too_many_arguments)]
fn fixed_rollover_peak_host_geometry(
    source: &crate::typed_insert_batch::PreparedResidentAppendSource,
    row_ids: &super::DeviceInsertRowIds,
    table_name: &str,
    key_proof: &BatchKeyConstraintProof,
    validation: &ResidentKeyValidationSeal,
    logical: FixedRolloverLogicalForecast,
    append_host: HostRetentionGeometry,
    append_host_scratch: HostRetentionGeometry,
    build_host_scratch: CudaHostScratchGeometry,
    lookup_host_peak: CudaHostScratchGeometry,
) -> Result<(HostRetentionGeometry, HostRetentionGeometry), ExecuteError> {
    let preview =
        fixed_rollover_preview_host_geometry(source, row_ids, table_name, key_proof, validation)?;
    let mut preview_revalidation = preview;
    preview_revalidation.checked_add_generation_pin_slots(2)?;
    let logical_host = fixed_rollover_logical_host_geometry(
        logical.raw_index_count,
        logical.physical_index_count,
    )?;
    let oracle_host = physical_probe::lookup_oracle_host_retention_geometry(
        logical.physical_index_count,
        source.row_count(),
    )?;

    // The table-name witness retires after currentness validation. Before the append compiler
    // consumes source/row IDs, the exact logical and lookup boxes overlap the remaining preview
    // owners.
    let mut pre_append = source.host_retention_geometry()?;
    pre_append.checked_add_disjoint(
        row_ids.host_allocation_geometry()?,
        "indexed rollover materializing row-id owner",
    )?;
    pre_append.checked_add_disjoint(
        key_proof.host_retention_geometry()?,
        "indexed rollover materializing key proof",
    )?;
    pre_append.checked_add_disjoint(
        validation.host_retention_geometry()?,
        "indexed rollover materializing validation seal",
    )?;
    pre_append.checked_add_disjoint(logical_host, "indexed rollover materializing logical")?;
    pre_append.checked_add_disjoint(oracle_host, "indexed rollover materializing oracle")?;

    let append_base =
        fixed_rollover_append_base_host_geometry(source, append_host, key_proof, validation)?;
    let mut append_materializing = append_base;
    append_materializing
        .checked_add_disjoint(logical_host, "indexed rollover append-time logical")?;
    append_materializing
        .checked_add_disjoint(oracle_host, "indexed rollover append-time oracle")?;
    append_materializing.checked_add_disjoint(
        append_host_scratch,
        "indexed rollover append-time host scratch",
    )?;
    let mut materialized = append_base;
    materialized.checked_add_disjoint(logical_host, "indexed rollover retained logical")?;
    materialized.checked_add_disjoint(oracle_host, "indexed rollover retained oracle")?;
    materialized.checked_add_backing_elements::<AtomicU64>(
        1,
        "indexed rollover materializing mutation epoch",
    )?;

    let mut reserved = materialized;
    reserved.checked_add_backing_elements::<ReservedPrivateIndexGeneration>(
        logical.physical_index_count,
        "indexed rollover reserved generation vector",
    )?;
    let build_column_bytes = u64::try_from(logical.descriptor_count)
        .ok()
        .and_then(|count| count.checked_mul(std::mem::size_of::<CudaCompoundFoldColumn>() as u64))
        .ok_or_else(|| decline("indexed rollover build-column host bytes overflow"))?;
    reserved.checked_add_backing_bytes_slots(
        build_column_bytes,
        u64::try_from(logical.physical_index_count)
            .map_err(|_| decline("indexed rollover build-column host slots overflow"))?,
        "indexed rollover build-column vectors",
    )?;

    // The destination generation vector reserves its final backing before the first reserved
    // build is consumed. At this instant every build-column vector is still live.
    let mut build_transition = reserved;
    build_transition.checked_add_backing_elements::<PreparedPrivateIndexGeneration>(
        logical.physical_index_count,
        "indexed rollover prepared generation vector",
    )?;
    build_transition.checked_add_backing_bytes_slots(
        build_host_scratch.bytes,
        build_host_scratch.allocation_slots,
        "indexed rollover execution build host scratch",
    )?;

    // Lookup owns its descriptor, index-guard, PTX, and three non-empty result vectors
    // simultaneously. It occurs after all build-column vectors have retired but while
    // logical/oracle/final generations live.
    let mut probe = materialized;
    probe.checked_add_backing_elements::<PreparedPrivateIndexGeneration>(
        logical.physical_index_count,
        "indexed rollover probing generation vector",
    )?;
    probe.checked_add_backing_bytes_slots(
        lookup_host_peak.bytes,
        lookup_host_peak.allocation_slots,
        "indexed rollover execution lookup host peak",
    )?;

    let final_host = fixed_rollover_final_host_geometry(
        source,
        key_proof,
        validation,
        logical.raw_index_count,
        logical.physical_index_count,
        append_host,
    )?;
    Ok((
        final_host,
        preview
            .peak(preview_revalidation)
            .peak(pre_append)
            .peak(append_materializing)
            .peak(materialized)
            .peak(reserved)
            .peak(build_transition)
            .peak(probe)
            .peak(final_host),
    ))
}

fn host_geometry_from_execution_scratch(
    scratch: CudaHostScratchGeometry,
) -> Result<HostRetentionGeometry, ExecuteError> {
    let mut geometry = HostRetentionGeometry::default();
    geometry.checked_add_backing_bytes_slots(
        scratch.bytes,
        scratch.allocation_slots,
        "indexed rollover execution host scratch",
    )?;
    Ok(geometry)
}

fn prepare_logical_bindings(
    table: &RelationalTable,
    keys: &BatchKeyConstraintProof,
    expected: FixedRolloverLogicalForecast,
) -> Result<PreparedFixedRolloverLogicalBindings, ExecuteError> {
    if table.indexes.is_empty() || keys.indexes().len() != table.indexes.len() {
        return Err(decline(
            "indexed rollover proof lost exact raw catalog enrollment",
        ));
    }
    let mut raw = Vec::with_capacity(expected.raw_index_count);
    let mut physical =
        Vec::<PhysicalIndexLogicalBinding>::with_capacity(expected.physical_index_count);
    let mut max_concurrent_scratch_bytes = 0_u64;
    let mut descriptor_count = 0_usize;
    let mut max_descriptor_count = 0_usize;
    for (raw_ordinal, index) in table.indexes.iter().enumerate() {
        let binding = keys
            .indexes()
            .get(raw_ordinal)
            .filter(|binding| binding.raw_ordinal() == raw_ordinal)
            .ok_or_else(|| decline("indexed rollover proof lost a raw catalog binding"))?;
        if !super::index_all_key_columns_foldable(table, index) {
            return Err(decline(
                "indexed rollover proof found a non-foldable named index",
            ));
        }
        let key_id = super::index_probe_key_id(table, index, raw_ordinal)
            .ok_or_else(|| decline("indexed rollover proof found no resident index key id"))?;
        let binding_descriptor_count = binding.key_column_count();
        if binding_descriptor_count == 0 {
            return Err(decline(
                "indexed rollover proof found an empty physical descriptor",
            ));
        }
        let physical_ordinal = if let Some(ordinal) =
            physical.iter().position(|binding| binding.key_id == key_id)
        {
            let prior = physical
                .get_mut(ordinal)
                .ok_or_else(|| decline("indexed rollover proof physical map drifted"))?;
            if prior.descriptor_count != binding_descriptor_count {
                return Err(decline(
                    "indexed rollover proof coalesced descriptor geometry drifted",
                ));
            }
            prior.duplicate_tolerant |= catalog_duplicate_tolerant(index, key_id);
            ordinal
        } else {
            let ordinal = physical.len();
            let scratch = resident_typed_index_build_preparation_bytes(binding_descriptor_count)
                .ok_or_else(|| decline("indexed rollover proof scratch geometry overflows"))?;
            max_concurrent_scratch_bytes = max_concurrent_scratch_bytes.max(scratch);
            descriptor_count = descriptor_count
                .checked_add(binding_descriptor_count)
                .ok_or_else(|| decline("indexed rollover proof descriptor total overflows"))?;
            max_descriptor_count = max_descriptor_count.max(binding_descriptor_count);
            physical.push(PhysicalIndexLogicalBinding {
                raw_ordinal,
                key_id,
                descriptor_count: binding_descriptor_count,
                duplicate_tolerant: catalog_duplicate_tolerant(index, key_id),
            });
            ordinal
        };
        raw.push(RawIndexLogicalBinding {
            raw_ordinal,
            key_id,
            physical_ordinal,
        });
    }
    if physical.is_empty()
        || max_concurrent_scratch_bytes == 0
        || raw.len() != expected.raw_index_count
        || physical.len() != expected.physical_index_count
        || descriptor_count != expected.descriptor_count
        || max_descriptor_count != expected.max_descriptor_count
        || max_concurrent_scratch_bytes != expected.max_build_scratch_bytes
    {
        return Err(decline(
            "indexed rollover proof has no physical index build",
        ));
    }
    Ok(PreparedFixedRolloverLogicalBindings {
        raw: raw.into(),
        physical: physical.into(),
        max_concurrent_scratch_bytes,
        descriptor_count,
        max_descriptor_count,
    })
}

fn catalog_duplicate_tolerant(index: &RelationalIndex, key_id: usize) -> bool {
    !index.unique || key_id & super::COMPOUND_KEY_ID_FLAG != 0
}

/// Check the catalog-order raw bindings, coalesced physical bindings, and final prepared
/// generations agree on the semantic flag a later manifest publisher needs. `created_posting`
/// deliberately is not substituted here: it reports a build outcome, while duplicate tolerance
/// is a catalog property even when the first build has no duplicate rows to post.
fn prepared_generations_preserve_catalog_duplicate_tolerance(
    table: &RelationalTable,
    logical: &PreparedFixedRolloverLogicalBindings,
    generations: &[PreparedPrivateIndexGeneration],
) -> bool {
    if generations.len() != logical.physical.len()
        || logical.raw.iter().enumerate().any(|(ordinal, raw)| {
            raw.raw_ordinal != ordinal || raw.physical_ordinal >= logical.physical.len()
        })
    {
        return false;
    }
    for (physical_ordinal, physical) in logical.physical.iter().enumerate() {
        let Some(generation) = generations.get(physical_ordinal) else {
            return false;
        };
        if generation.raw_ordinal != physical.raw_ordinal
            || generation.key_id != physical.key_id
            || generation.descriptor_count != physical.descriptor_count
            || generation.duplicate_tolerant != physical.duplicate_tolerant
        {
            return false;
        }
        let mut first_raw_ordinal = None;
        let mut expected_duplicate_tolerant = false;
        for raw in logical
            .raw
            .iter()
            .filter(|raw| raw.physical_ordinal == physical_ordinal)
        {
            let Some(index) = table.indexes.get(raw.raw_ordinal) else {
                return false;
            };
            let Some(key_id) = super::index_probe_key_id(table, index, raw.raw_ordinal) else {
                return false;
            };
            if raw.key_id != physical.key_id || key_id != physical.key_id {
                return false;
            }
            first_raw_ordinal.get_or_insert(raw.raw_ordinal);
            expected_duplicate_tolerant |= catalog_duplicate_tolerant(index, key_id);
        }
        if first_raw_ordinal != Some(physical.raw_ordinal)
            || expected_duplicate_tolerant != physical.duplicate_tolerant
            || expected_duplicate_tolerant != generation.duplicate_tolerant
        {
            return false;
        }
    }
    true
}

fn fixed_rollover_build_columns(
    table: &RelationalTable,
    basis: &super::fixed_insert::ResidentFixedRolloverReservationBasis<'_>,
    binding: &crate::engine_insert_plan::batch_key_constraints::IndexBinding,
    descriptor_count: usize,
) -> Result<Vec<CudaCompoundFoldColumn>, ExecuteError> {
    let capacity = u64::try_from(basis.capacity)
        .map_err(|_| decline("indexed rollover descriptor capacity overflows"))?;
    let int4_count = u64::try_from(
        table
            .columns
            .iter()
            .filter(|column| {
                matches!(
                    column.ty,
                    crate::SqlType::Int2 | crate::SqlType::Int4 | crate::SqlType::Date
                )
            })
            .count(),
    )
    .map_err(|_| decline("indexed rollover descriptor int4 count overflows"))?;
    let int8_count = u64::try_from(
        table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, crate::SqlType::Int8 | crate::SqlType::Timestamp))
            .count(),
    )
    .map_err(|_| decline("indexed rollover descriptor int8 count overflows"))?;
    let header = std::mem::size_of::<u64>() as u64;
    let int8_base = int4_count
        .checked_mul(capacity)
        .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<i32>() as u64))
        .and_then(|bytes| header.checked_add(bytes))
        .ok_or_else(|| decline("indexed rollover descriptor int8 base overflows"))?;
    let numeric_base = int8_count
        .checked_mul(capacity)
        .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<i64>() as u64))
        .and_then(|bytes| int8_base.checked_add(bytes))
        .ok_or_else(|| decline("indexed rollover descriptor numeric base overflows"))?;

    let mut columns = Vec::with_capacity(descriptor_count);
    binding.try_for_each_resolved_catalog_column(table, |column_idx, column| {
        let ordinal_before = |accepts: fn(&crate::SqlType) -> bool| {
            table
                .columns
                .iter()
                .take(column_idx)
                .filter(|candidate| accepts(&candidate.ty))
                .count()
        };
        let descriptor = match column.ty {
            crate::SqlType::Int2 | crate::SqlType::Int4 | crate::SqlType::Date => {
                let ordinal = u64::try_from(ordinal_before(|ty| {
                    matches!(
                        ty,
                        crate::SqlType::Int2 | crate::SqlType::Int4 | crate::SqlType::Date
                    )
                }))
                .map_err(|_| decline("indexed rollover descriptor int4 ordinal overflows"))?;
                let byte_offset = ordinal
                    .checked_mul(capacity)
                    .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<i32>() as u64))
                    .and_then(|bytes| header.checked_add(bytes))
                    .ok_or_else(|| decline("indexed rollover descriptor int4 offset overflows"))?;
                CudaCompoundFoldColumn::Fixed {
                    byte_offset,
                    width_words: 1,
                }
            }
            crate::SqlType::Int8 | crate::SqlType::Timestamp => {
                let ordinal = u64::try_from(ordinal_before(|ty| {
                    matches!(ty, crate::SqlType::Int8 | crate::SqlType::Timestamp)
                }))
                .map_err(|_| decline("indexed rollover descriptor int8 ordinal overflows"))?;
                let byte_offset = ordinal
                    .checked_mul(capacity)
                    .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<i64>() as u64))
                    .and_then(|bytes| int8_base.checked_add(bytes))
                    .ok_or_else(|| decline("indexed rollover descriptor int8 offset overflows"))?;
                CudaCompoundFoldColumn::Fixed {
                    byte_offset,
                    width_words: 2,
                }
            }
            crate::SqlType::Numeric { .. } | crate::SqlType::Uuid => {
                let ordinal = u64::try_from(ordinal_before(|ty| {
                    matches!(ty, crate::SqlType::Numeric { .. } | crate::SqlType::Uuid)
                }))
                .map_err(|_| decline("indexed rollover descriptor numeric ordinal overflows"))?;
                let byte_offset = ordinal
                    .checked_mul(capacity)
                    .and_then(|bytes| bytes.checked_mul(16))
                    .and_then(|bytes| numeric_base.checked_add(bytes))
                    .ok_or_else(|| {
                        decline("indexed rollover descriptor numeric offset overflows")
                    })?;
                CudaCompoundFoldColumn::Fixed {
                    byte_offset,
                    width_words: 4,
                }
            }
            crate::SqlType::Bool => {
                let bitmap_byte_offset = basis
                    .bool_layouts
                    .iter()
                    .find(|layout| layout.name == column.name)
                    .map(|layout| layout.bitmap_byte_offset)
                    .ok_or_else(|| {
                        decline("indexed rollover descriptor bool layout disappeared")
                    })?;
                CudaCompoundFoldColumn::Bool { bitmap_byte_offset }
            }
            crate::SqlType::Text => {
                return Err(decline(
                    "indexed fixed rollover cannot prepare a text index descriptor",
                ));
            }
        };
        columns.push(descriptor);
        Ok(())
    })?;
    if columns.len() != descriptor_count {
        return Err(decline(
            "indexed rollover proof build descriptor count drifted",
        ));
    }
    Ok(columns)
}

/// Consume the capacity forecast only after `engine_insert_plan` has issued the move-only
/// capability.  All CUDA-facing preparation is deliberately below this line.
pub(super) fn materialize<'a>(
    engine: &'a Engine,
    table: &RelationalTable,
    preview: IndexedFixedRolloverPreview<'a>,
    permit: IndexedPhysicalMaterializationPermit,
) -> Result<PreparedIndexedFixedRolloverReservation<'a>, ExecuteError> {
    let preview_geometry = preview.host_retention_geometry()?;
    let preview_report = preview.host_retention_report()?;
    if !preview_geometry.matches(&preview_report)? {
        return Err(decline(
            "indexed rollover preview host geometry diverged from actual ownership",
        ));
    }
    let mut host_peak = FixedRolloverHostPeakTracker::default();
    host_peak.observe(preview_report)?;
    let IndexedFixedRolloverPreview {
        source,
        row_ids,
        table_name,
        table_oid,
        schema_digest,
        predecessor_boundary,
        catalog_seq,
        predecessor_shard_id,
        predecessor_row_start,
        predecessor_row_count,
        predecessor_capacity,
        predecessor_payload,
        predecessor_generation,
        key_proof,
        validation,
        logical: expected_logical,
        rollover: expected_rollover,
        target_witness,
        generation_witness,
        resource_forecast,
        mutation_gate,
        named_index_lifecycle,
    } = preview;
    {
        let catalog = engine.catalog_snapshot();
        let current_table = catalog
            .relational_catalog
            .get(table_name.as_ref())
            .ok_or_else(|| decline("indexed rollover materialization lost table"))?;
        let shards = engine.read_state.residency.shards.load_full();
        let open = shards
            .get(table_name.as_ref())
            .and_then(|shards| shards.last())
            .ok_or_else(|| decline("indexed rollover materialization lost open shard"))?;
        let mut revalidation_host = source.host_retention_report()?;
        row_ids.append_host_allocation_slot(&mut revalidation_host)?;
        revalidation_host.retain_boxed_str(&table_name)?;
        key_proof.append_host_retention(&mut revalidation_host)?;
        validation.append_host_retention(&mut revalidation_host)?;
        revalidation_host.retain_generation_pin(&catalog)?;
        revalidation_host.retain_generation_pin(&shards)?;
        host_peak.observe(revalidation_host)?;
        if current_table != table
            || table.oid != table_oid
            || crate::engine_transaction_reset::table_schema_digest(current_table)
                .ok()
                .as_ref()
                != Some(&schema_digest)
            || catalog.commit_seq != catalog_seq
            || open.shard_id != predecessor_shard_id
            || open.row_start != predecessor_row_start
            || open.row_count != predecessor_row_count
            || open.capacity != predecessor_capacity
            || open.gpu_id != target_witness.gpu_id
            || open
                .device_memory
                .as_ref()
                .is_none_or(|payload| !Arc::ptr_eq(payload, &predecessor_payload))
            || !Arc::ptr_eq(&open.point_route_generation, &predecessor_generation)
            || generation_witness.catalog_seq != catalog_seq
            || generation_witness.predecessor_boundary != predecessor_boundary
            || generation_witness.open_shard_id != predecessor_shard_id
            || generation_witness.row_start
                != u64::try_from(predecessor_row_start).unwrap_or(u64::MAX)
            || generation_witness.row_count
                != u64::try_from(predecessor_row_count).unwrap_or(u64::MAX)
            || generation_witness.capacity
                != u64::try_from(predecessor_capacity).unwrap_or(u64::MAX)
            || target_witness.table_oid != table_oid
            || target_witness.schema_digest != schema_digest
            || !validation.matches_fixed_rollover_predecessor(
                current_table,
                catalog_seq,
                open,
                predecessor_boundary,
            )
            || !physical_probe::published_index_enrollment_is_complete(engine, current_table)
        {
            return Err(decline("indexed rollover materialization witness drifted"));
        }
    }
    drop(table_name);
    drop(predecessor_payload);
    drop(predecessor_generation);

    let mut logical = prepare_logical_bindings(table, &key_proof, expected_logical)?;
    if logical.raw.len() != expected_logical.raw_index_count
        || logical.physical.len() != expected_logical.physical_index_count
        || logical.descriptor_count != expected_logical.descriptor_count
        || logical.max_descriptor_count != expected_logical.max_descriptor_count
        || logical.max_concurrent_scratch_bytes != expected_logical.max_build_scratch_bytes
    {
        return Err(decline(
            "indexed rollover materialization logical forecast drifted",
        ));
    }
    let gpu_lookup_oracle = physical_probe::prepare_gpu_lookup_oracle(table, &logical, &source)?;
    logical.max_concurrent_scratch_bytes = logical
        .max_concurrent_scratch_bytes
        .max(gpu_lookup_oracle.max_concurrent_scratch_bytes());
    if logical.max_concurrent_scratch_bytes
        != resource_forecast.maximum_concurrent_device_scratch_bytes
    {
        return Err(decline(
            "indexed rollover materialization scratch forecast drifted",
        ));
    }
    let mut pre_append_host = source.host_retention_report()?;
    row_ids.append_host_allocation_slot(&mut pre_append_host)?;
    key_proof.append_host_retention(&mut pre_append_host)?;
    validation.append_host_retention(&mut pre_append_host)?;
    logical.append_host_retention(&mut pre_append_host)?;
    gpu_lookup_oracle.append_host_retention(&mut pre_append_host)?;
    host_peak.observe(pre_append_host)?;
    let append = engine
        .prepare_resident_open_shard_append_indexed_fixed_rollover_reservation(
            source,
            row_ids,
            mutation_gate,
            logical.max_concurrent_scratch_bytes,
            permit,
        )
        .map_err(|_| decline("indexed rollover materialization append reservation declined"))?;
    let append_host_scratch = append
        .indexed_fixed_rollover_host_materialization_scratch()
        .ok_or_else(|| decline("indexed rollover materialization lost append host scratch"))?;
    host_peak.observe_with_disjoint_scratch(
        fixed_rollover_materialized_host_report(
            &append,
            &key_proof,
            &validation,
            &logical,
            &gpu_lookup_oracle,
            None,
        )?,
        append_host_scratch,
        "indexed rollover actual append host scratch",
    )?;
    let basis = append
        .indexed_fixed_rollover_reservation_basis()
        .ok_or_else(|| decline("indexed rollover materialization lost fixed basis"))?;
    if basis.capacity != expected_rollover.capacity()
        || basis.payload_bytes != expected_rollover.payload_bytes()
        || basis.created_by_bytes != expected_rollover.created_by_bytes()
        || basis.row_id_bytes != expected_rollover.row_id_bytes()
        || basis.planned_index_allocation_bytes != expected_rollover.named_index_bytes()
        || basis.capacity_fit_evaluations != expected_rollover.capacity_scan_entries()
    {
        return Err(decline(
            "indexed rollover materialization payload forecast drifted",
        ));
    }
    prepare(
        engine,
        table,
        predecessor_boundary,
        append,
        key_proof,
        validation,
        logical,
        gpu_lookup_oracle,
        named_index_lifecycle,
        resource_forecast,
        target_witness,
        generation_witness,
        resource_forecast.old_generation_pinned_bytes,
        resource_forecast.generation_pin_slots,
        host_peak,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare<'a>(
    engine: &'a Engine,
    table: &RelationalTable,
    predecessor_boundary: Index,
    append: super::fixed_insert::ResidentOpenShardAppendPlan<'a>,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    logical: PreparedFixedRolloverLogicalBindings,
    gpu_lookup_oracle: physical_probe::PrivateGpuLookupOracle,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
    resource_forecast: IndexedPhysicalResourceForecast,
    target_witness: IndexedPhysicalTargetWitness,
    generation_witness: IndexedPhysicalGenerationWitness,
    old_generation_pinned_bytes: u64,
    generation_pin_slots: u64,
    mut host_peak: FixedRolloverHostPeakTracker,
) -> Result<PreparedIndexedFixedRolloverReservation<'a>, ExecuteError> {
    let basis = append
        .indexed_fixed_rollover_reservation_basis()
        .ok_or_else(|| decline("indexed rollover proof did not select fixed rollover"))?;
    if gpu_lookup_oracle.incoming_rows() != basis.incoming_rows {
        return Err(decline(
            "indexed rollover proof GPU lookup oracle row geometry drifted",
        ));
    }
    let shard_map = engine.read_state.residency.shards.load_full();
    let predecessor = shard_map
        .get(&table.name)
        .and_then(|shards| {
            shards
                .iter()
                .find(|shard| shard.shard_id == basis.predecessor_shard_id)
        })
        .ok_or_else(|| decline("indexed rollover proof lost its predecessor shard"))?;
    if !validation.matches_fixed_rollover_predecessor(
        table,
        basis.catalog_seq,
        predecessor,
        predecessor_boundary,
    ) || validation.original_read_snapshot() > predecessor_boundary
        || predecessor.row_count != basis.predecessor_row_count
        || basis
            .predecessor_row_count
            .checked_add(predecessor.row_start)
            != Some(basis.new_row_start)
        || basis.predecessor_shard_id.checked_add(1) != Some(basis.new_shard_id)
        || basis.payload.metadata().gpu_id != predecessor.gpu_id
        || basis.payload.metadata().allocated_bytes == 0
    {
        return Err(decline(
            "indexed rollover proof predecessor generation witness drifted",
        ));
    }
    if !physical_probe::published_index_enrollment_is_complete(engine, table) {
        return Err(decline(
            "indexed rollover proof requires complete named-index publication",
        ));
    }
    let private_header = basis
        .payload
        .read_resident_u64_column(0, 1)
        .map_err(|_| decline("indexed rollover proof private header readback failed"))?;
    if private_header != [0] {
        return Err(decline(
            "indexed rollover proof exposed its private row-count header",
        ));
    }
    drop(shard_map);

    let table_size =
        super::resident_shard_index_table_size(basis.incoming_rows as u64, basis.capacity as u64)
            .ok_or_else(|| decline("indexed rollover proof index horizon overflows"))?;
    let table_mask = u32::try_from(table_size - 1)
        .map_err(|_| decline("indexed rollover proof table mask overflows"))?;
    let hash_shift = 32 - table_size.trailing_zeros();
    let one_index_bytes =
        resident_index_allocated_bytes(table_mask, basis.capacity.max(basis.incoming_rows) as u64)
            .ok_or_else(|| decline("indexed rollover proof index allocation overflows"))?;
    let expected_index_bytes = one_index_bytes
        .checked_mul(logical.physical.len() as u64)
        .ok_or_else(|| decline("indexed rollover proof persistent index bytes overflow"))?;
    if expected_index_bytes != basis.planned_index_allocation_bytes
        || logical.max_concurrent_scratch_bytes != basis.max_index_scratch_bytes
    {
        return Err(decline("indexed rollover proof reserved geometry drifted"));
    }

    let mutation_epoch = engine
        .read_state
        .residency
        .point_index_mutation_epoch_for_table(&engine.read_state, table)
        .ok_or_else(|| decline("indexed rollover proof relation identity changed"))?;
    let mutation_epoch_expected_even = mutation_epoch.load(Ordering::Acquire);
    if mutation_epoch_expected_even & 1 != 0 {
        return Err(decline(
            "indexed rollover proof observed an active mutation epoch",
        ));
    }
    host_peak.observe(fixed_rollover_materialized_host_report(
        &append,
        &key_proof,
        &validation,
        &logical,
        &gpu_lookup_oracle,
        Some(&mutation_epoch),
    )?)?;

    let source_payload = Arc::clone(basis.payload);
    let gpu_id = source_payload.metadata().gpu_id;
    let runtime = engine.cuda_driver_probe_runtime();
    // Allocate and validate every persistent destination before launching the first build. If
    // any allocation fails, all already-reserved directories and the private payload drop
    // without a partially built generation ever existing.
    let mut reserved = Vec::with_capacity(logical.physical.len());
    for physical in logical.physical.iter() {
        let binding = key_proof
            .indexes()
            .get(physical.raw_ordinal)
            .filter(|binding| binding.raw_ordinal() == physical.raw_ordinal)
            .ok_or_else(|| decline("indexed rollover proof build binding drifted"))?;
        let columns =
            fixed_rollover_build_columns(table, &basis, binding, physical.descriptor_count)?;
        let memory = Arc::new(
            runtime
                .retain_device_memory_zeroed(gpu_id, one_index_bytes)
                .map_err(|error| {
                    decline(format!(
                        "indexed rollover proof private index allocation failed: {error}"
                    ))
                })?,
        );
        if memory.metadata().allocated_bytes != one_index_bytes
            || memory.device_ptr() == source_payload.device_ptr()
        {
            return Err(decline(
                "indexed rollover proof private index allocation drifted",
            ));
        }
        #[cfg(test)]
        PRIVATE_INDEX_ALLOCATIONS.with(|count| {
            count.set(
                count
                    .get()
                    .checked_add(1)
                    .expect("test-only private index allocation counter overflowed"),
            );
        });
        reserved.push(ReservedPrivateIndexGeneration {
            raw_ordinal: physical.raw_ordinal,
            key_id: physical.key_id,
            memory,
            build_columns: columns,
            table_mask,
            hash_shift,
            allocated_bytes: one_index_bytes,
            descriptor_count: physical.descriptor_count,
            duplicate_tolerant: physical.duplicate_tolerant,
        });
    }
    let reserved_index_bytes = reserved.iter().try_fold(0_u64, |total, generation| {
        total
            .checked_add(generation.allocated_bytes)
            .ok_or_else(|| decline("indexed rollover proof reserved index bytes overflow"))
    })?;
    if reserved_index_bytes != basis.planned_index_allocation_bytes {
        return Err(decline(
            "indexed rollover proof reserved persistent bytes drifted",
        ));
    }
    let mut reserved_peak = fixed_rollover_materialized_host_report(
        &append,
        &key_proof,
        &validation,
        &logical,
        &gpu_lookup_oracle,
        Some(&mutation_epoch),
    )?;
    reserved_peak.retain_vec(&reserved)?;
    for generation in reserved.iter() {
        reserved_peak.retain_vec(&generation.build_columns)?;
    }
    host_peak.observe(reserved_peak)?;

    #[cfg(test)]
    physical_probe::apply_test_build_sabotage(&mut reserved)?;

    // Build the widest descriptor first. Every reserved build-column owner and the final
    // generation Vec backing are then simultaneously live with the execution primitive's maximum
    // descriptor/PTX scratch, making the forecasted high-water both exact and directly observable.
    reserved.sort_unstable_by_key(|generation| {
        (
            std::cmp::Reverse(generation.descriptor_count),
            generation.raw_ordinal,
        )
    });

    // The complete persistent generation is now live. Zero-based builds are synchronous and
    // sequential, so their exact concurrent scratch high-water is max(S_i), never sum(S_i).
    let allocation_scope = CudaAllocationScope::with_budget(logical.max_concurrent_scratch_bytes);
    let mut generations = Vec::with_capacity(reserved.len());
    let mut transition_peak = fixed_rollover_materialized_host_report(
        &append,
        &key_proof,
        &validation,
        &logical,
        &gpu_lookup_oracle,
        Some(&mutation_epoch),
    )?;
    transition_peak.retain_vec(&reserved)?;
    for generation in reserved.iter() {
        transition_peak.retain_vec(&generation.build_columns)?;
    }
    transition_peak.retain_vec(&generations)?;
    host_peak.observe(transition_peak.clone())?;
    let expected_build_host_scratch =
        resident_typed_index_build_host_scratch_geometry(logical.max_descriptor_count)
            .ok_or_else(|| decline("indexed rollover proof build host scratch overflows"))?;
    let mut observed_build_host_scratch = None;
    for reserved in reserved {
        let (status, call_host_scratch) = source_payload
            .submit_resident_typed_index_build_status_observed(
                &reserved.memory,
                reserved.table_mask,
                reserved.hash_shift,
                &reserved.build_columns,
                basis.incoming_rows,
                None,
                0,
                reserved.duplicate_tolerant,
            )
            .map_err(|error| {
                decline(format!(
                    "indexed rollover proof zero-based GPU build failed: {error}"
                ))
            })?;
        if observed_build_host_scratch.is_none() {
            if reserved.descriptor_count != logical.max_descriptor_count
                || call_host_scratch != expected_build_host_scratch
            {
                return Err(decline(
                    "indexed rollover proof widest build host scratch drifted",
                ));
            }
            host_peak.observe_with_disjoint_scratch(
                transition_peak.clone(),
                host_geometry_from_execution_scratch(call_host_scratch)?,
                "indexed rollover actual execution build host scratch",
            )?;
            observed_build_host_scratch = Some(call_host_scratch);
        } else if call_host_scratch.bytes > expected_build_host_scratch.bytes
            || call_host_scratch.allocation_slots > expected_build_host_scratch.allocation_slots
        {
            return Err(decline(
                "indexed rollover proof later build exceeded widest host scratch",
            ));
        }
        if status.declined {
            return Err(decline(
                "indexed rollover proof zero-based GPU build declined",
            ));
        }
        #[cfg(test)]
        PRIVATE_INDEX_BUILDS.with(|count| {
            count.set(
                count
                    .get()
                    .checked_add(1)
                    .expect("test-only private index build counter overflowed"),
            );
        });
        generations.push(PreparedPrivateIndexGeneration {
            raw_ordinal: reserved.raw_ordinal,
            key_id: reserved.key_id,
            memory: reserved.memory,
            table_mask: reserved.table_mask,
            hash_shift: reserved.hash_shift,
            allocated_bytes: reserved.allocated_bytes,
            descriptor_count: reserved.descriptor_count,
            duplicate_tolerant: reserved.duplicate_tolerant,
            created_posting: status.created_posting,
        });
        #[cfg(test)]
        if take_fail_after_build(generations.len()) {
            return Err(decline(
                "indexed rollover proof injected failure after private index build",
            ));
        }
    }
    if observed_build_host_scratch != Some(expected_build_host_scratch) {
        return Err(decline(
            "indexed rollover proof observed no exact execution build host peak",
        ));
    }
    generations.sort_unstable_by_key(|generation| generation.raw_ordinal);
    if !prepared_generations_preserve_catalog_duplicate_tolerance(table, &logical, &generations) {
        return Err(decline(
            "indexed rollover proof prepared duplicate-tolerance mapping drifted",
        ));
    }
    let probe_evidence = physical_probe::verify_private_gpu_lookups(
        &source_payload,
        &generations,
        &gpu_lookup_oracle,
    )?;
    let mut probe_peak = fixed_rollover_materialized_host_report(
        &append,
        &key_proof,
        &validation,
        &logical,
        &gpu_lookup_oracle,
        Some(&mutation_epoch),
    )?;
    probe_peak.retain_vec(&generations)?;
    host_peak.observe_with_disjoint_scratch(
        probe_peak,
        probe_evidence.host_call_peak,
        "indexed rollover actual execution lookup host peak",
    )?;
    let observed_scratch_peak_bytes = allocation_scope.peak_bytes();
    if observed_scratch_peak_bytes != logical.max_concurrent_scratch_bytes
        || mutation_epoch.load(Ordering::Acquire) != mutation_epoch_expected_even
        || source_payload
            .read_resident_u64_column(0, 1)
            .map_err(|_| decline("indexed rollover proof post-build header readback failed"))?
            != [0]
    {
        return Err(decline(
            "indexed rollover proof build high-water or generation drifted",
        ));
    }
    drop(allocation_scope);

    let actual_index_bytes = generations.iter().try_fold(0_u64, |total, generation| {
        total
            .checked_add(generation.allocated_bytes)
            .ok_or_else(|| decline("indexed rollover proof actual index bytes overflow"))
    })?;
    let total_persistent_bytes = basis
        .payload_sidecar_allocation_bytes
        .checked_add(actual_index_bytes)
        .ok_or_else(|| decline("indexed rollover proof total persistent bytes overflow"))?;
    let persistent_allocation_count = basis
        .payload_sidecar_allocation_count
        .checked_add(generations.len() as u64)
        .ok_or_else(|| decline("indexed rollover proof allocation count overflows"))?;
    let build_readback_bytes = (generations.len() as u64)
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .ok_or_else(|| decline("indexed rollover proof readback bytes overflow"))?;
    let bounded_readback_bytes = build_readback_bytes
        .checked_add(probe_evidence.bounded_readback_bytes)
        .ok_or_else(|| decline("indexed rollover proof probe readback bytes overflow"))?;
    let ledger = FixedRolloverResourceLedger {
        payload_bytes: basis.payload_bytes,
        created_by_bytes: basis.created_by_bytes,
        row_id_bytes: basis.row_id_bytes,
        payload_sidecar_persistent_bytes: basis.payload_sidecar_allocation_bytes,
        index_persistent_bytes: actual_index_bytes,
        total_persistent_bytes,
        max_concurrent_scratch_bytes: logical.max_concurrent_scratch_bytes,
        observed_scratch_peak_bytes,
        bounded_readback_bytes,
        max_concurrent_readback_bytes: (std::mem::size_of::<u32>() as u64)
            .max(probe_evidence.max_concurrent_readback_bytes),
        persistent_allocation_count,
        raw_index_count: logical.raw.len(),
        physical_index_count: generations.len(),
        descriptor_count: logical.descriptor_count,
        max_descriptor_count: logical.max_descriptor_count,
        gpu_probe_count: probe_evidence.gpu_probe_count,
    };
    let PreparedFixedRolloverLogicalBindings {
        raw: raw_bindings,
        physical,
        max_concurrent_scratch_bytes: _,
        descriptor_count: _,
        max_descriptor_count: _,
    } = logical;
    drop(physical);
    drop(gpu_lookup_oracle);
    let generations = generations.into_boxed_slice();
    #[cfg(test)]
    let report = IndexedFixedRolloverProofReport {
        raw_index_count: ledger.raw_index_count,
        physical_index_count: ledger.physical_index_count,
        predecessor_shard_id: basis.predecessor_shard_id,
        successor_shard_id: basis.new_shard_id,
        successor_row_start: basis.new_row_start,
        incoming_rows: basis.incoming_rows,
        capacity: basis.capacity,
        original_read_snapshot: validation.original_read_snapshot(),
        predecessor_boundary,
        payload_bytes: ledger.payload_bytes,
        created_by_bytes: ledger.created_by_bytes,
        row_id_bytes: ledger.row_id_bytes,
        payload_sidecar_persistent_bytes: ledger.payload_sidecar_persistent_bytes,
        index_persistent_bytes: ledger.index_persistent_bytes,
        total_persistent_bytes: ledger.total_persistent_bytes,
        max_concurrent_scratch_bytes: ledger.max_concurrent_scratch_bytes,
        observed_scratch_peak_bytes: ledger.observed_scratch_peak_bytes,
        bounded_readback_bytes: ledger.bounded_readback_bytes,
        max_concurrent_readback_bytes: ledger.max_concurrent_readback_bytes,
        persistent_allocation_count: ledger.persistent_allocation_count,
        descriptor_count: ledger.descriptor_count,
        max_descriptor_count: ledger.max_descriptor_count,
        capacity_fit_evaluations: basis.capacity_fit_evaluations,
        budget_scan_entries: basis.budget_scan_entries,
        gpu_build_count: generations.len(),
        gpu_probe_count: ledger.gpu_probe_count,
        posting_index_count: generations
            .iter()
            .filter(|generation| generation.created_posting)
            .count(),
        duplicate_tolerant_index_count: generations
            .iter()
            .filter(|generation| generation.duplicate_tolerant)
            .count(),
        private_header_zero: true,
        final_host_retained_bytes: 0,
        final_host_allocation_slots: 0,
        final_host_generation_pin_slots: 0,
        observed_peak_host_retained_bytes: 0,
        observed_peak_host_allocation_slots: 0,
        observed_peak_host_generation_pin_slots: 0,
        forecast_final_host_retained_bytes: 0,
        forecast_final_host_allocation_slots: 0,
        forecast_peak_host_retained_bytes: 0,
        forecast_peak_host_allocation_slots: 0,
        concrete_generation_box_bytes: 0,
        concrete_generation_box_slots: 0,
    };
    let reservation = PreparedIndexedFixedRolloverReservation {
        indexes: PreparedPrivateRolloverIndexes {
            generations,
            source_payload,
            key_proof,
            validation,
            raw_bindings,
            mutation_epoch,
            mutation_epoch_expected_even,
            ledger,
        },
        append,
        forecast: resource_forecast,
        target_witness,
        generation_witness,
        old_generation_pinned_bytes,
        generation_pin_slots,
        host_retention_peak_before_finalization: host_peak.peak(),
        #[cfg(test)]
        report,
        _named_index_lifecycle: named_index_lifecycle,
    };
    if !reservation.forecast_matches_actual()? {
        return Err(decline(
            "indexed rollover materialization owner ledger diverged from forecast",
        ));
    }
    Ok(reservation)
}

#[cfg(test)]
thread_local! {
    static FAIL_AFTER_PRIVATE_INDEX_BUILDS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static PRIVATE_INDEX_ALLOCATIONS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
    static PRIVATE_INDEX_BUILDS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}
#[cfg(test)]
fn take_fail_after_build(completed: usize) -> bool {
    FAIL_AFTER_PRIVATE_INDEX_BUILDS.with(|slot| {
        if slot.get() == Some(completed) {
            slot.set(None);
            true
        } else {
            false
        }
    })
}
#[cfg(test)]
fn fail_next_after_private_index_builds(completed: usize) {
    FAIL_AFTER_PRIVATE_INDEX_BUILDS.with(|slot| slot.set(Some(completed)));
}
#[cfg(test)]
fn private_index_work_counts() -> (u64, u64) {
    (
        PRIVATE_INDEX_ALLOCATIONS.with(std::cell::Cell::get),
        PRIVATE_INDEX_BUILDS.with(std::cell::Cell::get),
    )
}
fn decline(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Serialization(message.into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug, PartialEq, Eq)]
    struct PublicStateFingerprint {
        wal_records: usize,
        committed_seq: Index,
        next_row_id: u64,
        resident_bytes: u64,
        mutation_epoch: u64,
        shards: Vec<String>,
        cache: Vec<String>,
        coverage: Vec<String>,
        complete: String,
        publications: String,
    }
    fn device_ptr(memory: Option<&Arc<CudaResidentDeviceMemory>>) -> u64 {
        memory.map_or(0, |memory| memory.device_ptr())
    }
    fn indexed_rollover_engine() -> (Engine, i32) {
        let mut engine = Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        assert!(
            hardware.driver_available && hardware.device_count != 0,
            "this ignored proof test requires a real CUDA device"
        );
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(2);
        for (txn_id, sql) in [
            (1, "CREATE TABLE inert_index_rollover_other (value int4)"),
            (
                2,
                "CREATE TABLE inert_index_rollover \
                 (id int4 PRIMARY KEY, shared int4, code int4 UNIQUE)",
            ),
            (
                3,
                "CREATE INDEX inert_index_rollover_shared_a \
                 ON inert_index_rollover (shared)",
            ),
            (
                4,
                "CREATE INDEX inert_index_rollover_shared_b \
                 ON inert_index_rollover (shared)",
            ),
            (
                5,
                "CREATE INDEX inert_index_rollover_compound \
                 ON inert_index_rollover (shared, code)",
            ),
            (6, "INSERT INTO inert_index_rollover VALUES (1, 10, 100)"),
            (7, "INSERT INTO inert_index_rollover VALUES (2, 20, 200)"),
        ] {
            engine.execute_text(txn_id, sql).unwrap();
        }
        engine
            .populate_relational_residency_snapshot("inert_index_rollover")
            .unwrap();
        engine
            .publish_relational_resident_indexes("inert_index_rollover")
            .unwrap();
        {
            let shards = engine.read_state.residency.shards.load();
            let table_shards = shards
                .get("inert_index_rollover")
                .expect("resident indexed fixture");
            assert!(!table_shards.is_empty());
            let open = table_shards.last().unwrap();
            assert_eq!(open.row_count, open.capacity);
        }
        (engine, 3)
    }
    fn indexed_rollover_plan(
        engine: &Engine,
        next_id: i32,
    ) -> crate::engine_insert_plan::PreparedDeviceInsertPlan {
        let catalog = engine.catalog_snapshot();
        let command = gpu_db_sql::parse_command(&format!(
            "INSERT INTO inert_index_rollover VALUES \
             ({next_id}, {}, {})",
            next_id * 10,
            next_id * 100
        ))
        .unwrap();
        let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
            &command,
            &catalog,
            catalog.commit_seq,
        )
        .unwrap()
        .expect("indexed fixed-rollover proof-only builder remains eligible");
        crate::engine_insert_plan::PreparedDeviceInsertPlan::from_typed_batch(
            batch, engine, &catalog,
        )
        .unwrap()
    }
    fn inspect_rollover(
        engine: &Engine,
        next_id: i32,
    ) -> Result<IndexedFixedRolloverProofReport, ExecuteError> {
        let plan = indexed_rollover_plan(engine, next_id);
        let proposal = plan
            .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
            .unwrap();
        plan.inspect_current_resident_index_rollover(engine, proposal, |report| report)
    }
    fn public_state(engine: &Engine) -> PublicStateFingerprint {
        let shards = engine.read_state.residency.shards.load();
        let table_shards = shards
            .get("inert_index_rollover")
            .into_iter()
            .flatten()
            .map(|shard| {
                format!(
                    "{}:{}:{}:{}:{}:{}:{}:{:p}",
                    shard.shard_id,
                    shard.row_count,
                    shard.capacity,
                    device_ptr(shard.device_memory.as_ref()),
                    device_ptr(shard.created_by_region.as_ref()),
                    device_ptr(shard.row_id_region.as_ref()),
                    device_ptr(shard.deleted_by_region.as_ref()),
                    Arc::as_ptr(&shard.point_route_generation),
                )
            })
            .collect();
        drop(shards);
        let cache = engine
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|((table, _, _), _)| table == "inert_index_rollover")
            .map(|(key, entry)| {
                format!(
                    "{key:?}:{}:{}:{}:{}:{}:{}:{}:{}",
                    entry.resident_device_ptr,
                    entry.row_count,
                    entry.published_row_count.load(Ordering::Acquire),
                    device_ptr(entry.device_index.as_ref()),
                    entry.table_mask,
                    entry.hash_shift,
                    entry.has_postings,
                    entry.published_has_postings.load(Ordering::Acquire),
                )
            })
            .collect();
        let coverage = engine
            .read_state
            .residency
            .named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|((table, _, _), _)| table == "inert_index_rollover")
            .map(|(key, value)| format!("{key:?}:{value:?}"))
            .collect();
        let complete = format!(
            "{:?}",
            engine
                .read_state
                .residency
                .named_index_coverage_complete
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get("inert_index_rollover")
        );
        let publications = {
            let catalog = engine.catalog_snapshot();
            let oid = catalog.relational_catalog["inert_index_rollover"].oid;
            format!(
                "{:?}",
                engine
                    .read_state
                    .residency
                    .named_index_publications
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .get(&oid)
            )
        };
        PublicStateFingerprint {
            wal_records: engine.durable_wal_records().len(),
            committed_seq: engine.committed_seq(),
            next_row_id: engine.read_state.mvcc.current_row_id(),
            resident_bytes: engine.relational_resident_bytes_for_gpu(0),
            mutation_epoch: {
                let catalog = engine.catalog_snapshot();
                engine
                    .read_state
                    .residency
                    .point_index_mutation_epoch_for_table(
                        &engine.read_state,
                        catalog
                            .relational_catalog
                            .get("inert_index_rollover")
                            .expect("inert rollover table remains catalog-visible"),
                    )
                    .expect("exact inert rollover table installs its point slot")
                    .load(Ordering::Acquire)
            },
            shards: table_shards,
            cache,
            coverage,
            complete,
            publications,
        }
    }
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn indexed_fixed_rollover_private_generation_is_exact_gpu_work_and_inert() {
        let (engine, next_id) = indexed_rollover_engine();
        let before = public_state(&engine);
        let reservations_before = super::super::rollover::fixed_rollover_reservation_count();
        let report = {
            let plan = indexed_rollover_plan(&engine, next_id);
            let proposal = plan
                .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
                .unwrap();
            plan.inspect_current_resident_index_rollover(&engine, proposal, |report| report)
                .unwrap()
        };
        assert_eq!(public_state(&engine), before);
        assert_eq!(
            super::super::rollover::fixed_rollover_reservation_count(),
            reservations_before + 1
        );
        assert_eq!(report.raw_index_count, 5);
        assert_eq!(report.physical_index_count, 4);
        assert_eq!(report.gpu_build_count, 4);
        assert_eq!(report.gpu_probe_count, 4);
        assert_eq!(report.descriptor_count, 5);
        assert_eq!(report.max_descriptor_count, 2);
        assert_eq!(report.posting_index_count, 0);
        assert_eq!(
            report.duplicate_tolerant_index_count, 2,
            "two physical generations retain non-unique/compound posting semantics even though this first build created no postings"
        );
        assert_eq!(report.bounded_readback_bytes, 64);
        assert_eq!(report.max_concurrent_readback_bytes, 12);
        assert_eq!(report.persistent_allocation_count, 7);
        assert_eq!(
            report.payload_sidecar_persistent_bytes,
            report
                .payload_bytes
                .checked_add(report.created_by_bytes)
                .and_then(|bytes| bytes.checked_add(report.row_id_bytes))
                .unwrap()
        );
        assert_eq!(
            report.total_persistent_bytes,
            report
                .payload_sidecar_persistent_bytes
                .checked_add(report.index_persistent_bytes)
                .unwrap()
        );
        assert_eq!(
            report.max_concurrent_scratch_bytes,
            resident_typed_index_build_preparation_bytes(2).unwrap()
        );
        assert_eq!(
            report.observed_scratch_peak_bytes,
            report.max_concurrent_scratch_bytes
        );
        assert!(report.index_persistent_bytes > 0);
        assert!(report.payload_bytes > 0);
        assert!(report.capacity_fit_evaluations > 0);
        assert!(report.budget_scan_entries > 0);
        assert_eq!(report.successor_shard_id, report.predecessor_shard_id + 1);
        assert!(report.successor_row_start > 0);
        assert_eq!(report.incoming_rows, 1);
        assert!(report.private_header_zero);
        assert_eq!(
            report.final_host_retained_bytes,
            report.forecast_final_host_retained_bytes
        );
        assert_eq!(
            report.final_host_allocation_slots,
            report.forecast_final_host_allocation_slots
        );
        assert_eq!(
            report.observed_peak_host_retained_bytes,
            report.forecast_peak_host_retained_bytes
        );
        assert_eq!(
            report.observed_peak_host_allocation_slots,
            report.forecast_peak_host_allocation_slots
        );
        assert!(
            report.observed_peak_host_retained_bytes > report.final_host_retained_bytes,
            "permit-time host scratch must be a non-vacuous measured high-water"
        );
        assert!(
            report.observed_peak_host_allocation_slots > report.final_host_allocation_slots,
            "permit-time temporary owners must increase the allocation-slot high-water"
        );
        assert_eq!(report.final_host_generation_pin_slots, 0);
        assert_eq!(
            report.observed_peak_host_generation_pin_slots, 2,
            "catalog and shard-map generations must both be charged during revalidation"
        );
        assert_eq!(
            report.concrete_generation_box_bytes,
            report.physical_index_count as u64
                * std::mem::size_of::<PreparedPrivateIndexGeneration>() as u64
        );
        assert_eq!(report.concrete_generation_box_slots, 1);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn indexed_fixed_rollover_actual_host_high_water_one_short_rejects_before_permit() {
        let (engine, next_id) = indexed_rollover_engine();
        let actual = inspect_rollover(&engine, next_id).unwrap();
        let materialization_scratch = actual
            .observed_peak_host_retained_bytes
            .checked_sub(actual.final_host_retained_bytes)
            .expect("actual peak covers final host retention");
        let execution_build_scratch =
            resident_typed_index_build_host_scratch_geometry(actual.max_descriptor_count)
                .expect("reported descriptor count has exact execution host geometry");
        assert!(
            materialization_scratch > actual.max_concurrent_readback_bytes,
            "the test must target the concrete owner-overlap peak, not only readback"
        );
        assert!(
            materialization_scratch >= execution_build_scratch.bytes,
            "the one-short limit must cover the execution-owned descriptor/PTX build scratch"
        );
        assert!(
            actual
                .observed_peak_host_allocation_slots
                .checked_sub(actual.final_host_allocation_slots)
                .expect("actual peak covers final host slots")
                >= execution_build_scratch.allocation_slots,
            "the one-short limit must cover every execution-owned build allocation slot"
        );
        let before = public_state(&engine);
        let reservations_before = super::super::rollover::fixed_rollover_reservation_count();
        let work_before = private_index_work_counts();
        crate::engine_insert_plan::limit_next_test_indexed_host_scratch_to(
            materialization_scratch - 1,
        );
        assert!(matches!(
            inspect_rollover(&engine, next_id),
            Err(ExecuteError::Serialization(_))
        ));
        assert_eq!(public_state(&engine), before);
        assert_eq!(
            super::super::rollover::fixed_rollover_reservation_count(),
            reservations_before,
            "actual-high-water one-short must reject before private payload allocation"
        );
        assert_eq!(
            private_index_work_counts(),
            work_before,
            "actual-high-water one-short must reject before index allocation or launch"
        );
        let retry = inspect_rollover(&engine, next_id).unwrap();
        assert_eq!(
            retry.observed_peak_host_retained_bytes,
            actual.observed_peak_host_retained_bytes
        );
        assert_eq!(public_state(&engine), before);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn indexed_fixed_rollover_exact_minimum_peak_refuses_then_shrinks() {
        let (mut engine, next_id) = indexed_rollover_engine();
        let unlimited = inspect_rollover(&engine, next_id).unwrap();
        assert!(unlimited.capacity > unlimited.incoming_rows);
        let catalog = engine.catalog_snapshot();
        let table = &catalog.relational_catalog["inert_index_rollover"];
        let column_types = table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect::<Vec<_>>();
        let minimum = super::super::rollover::ResidentRolloverPlan::fixed_width_null_free(
            table,
            &column_types,
            1,
            1,
            true,
            true,
            None,
        )
        .unwrap()
        .unwrap();
        let minimum_persistent = minimum.total_allocation_bytes();
        let resident = engine.relational_resident_bytes_for_gpu(0);
        let minimum_peak = minimum_persistent
            .checked_add(unlimited.max_concurrent_scratch_bytes)
            .unwrap();
        engine.set_relational_residency_budget_bytes(0, resident + minimum_peak - 1);
        let refused_state = public_state(&engine);
        let reservations_before = super::super::rollover::fixed_rollover_reservation_count();
        assert!(matches!(
            inspect_rollover(&engine, next_id),
            Err(ExecuteError::Serialization(_))
        ));
        assert_eq!(public_state(&engine), refused_state);
        assert_eq!(
            super::super::rollover::fixed_rollover_reservation_count(),
            reservations_before,
            "minimum-peak refusal must happen before private payload allocation"
        );
        engine.set_relational_residency_budget_bytes(0, resident + minimum_peak);
        let exact_state = public_state(&engine);
        let report = inspect_rollover(&engine, next_id).unwrap();
        assert_eq!(public_state(&engine), exact_state);
        assert_eq!(report.capacity, 1);
        assert_eq!(report.total_persistent_bytes, minimum_persistent);
        assert_eq!(
            report
                .total_persistent_bytes
                .checked_add(report.max_concurrent_scratch_bytes),
            Some(minimum_peak)
        );
        assert_eq!(
            super::super::rollover::fixed_rollover_reservation_count(),
            reservations_before + 1
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn indexed_fixed_rollover_first_middle_last_failure_is_atomic_and_reusable() {
        for completed_before_failure in [1, 2, 4] {
            let (engine, next_id) = indexed_rollover_engine();
            let before = public_state(&engine);
            let reservations_before = super::super::rollover::fixed_rollover_reservation_count();
            let work_before = private_index_work_counts();
            fail_next_after_private_index_builds(completed_before_failure);
            assert!(matches!(
                inspect_rollover(&engine, next_id),
                Err(ExecuteError::Serialization(_))
            ));
            assert_eq!(
                public_state(&engine),
                before,
                "failure after build {completed_before_failure} changed public state"
            );
            assert_eq!(
                super::super::rollover::fixed_rollover_reservation_count(),
                reservations_before + 1
            );
            assert_eq!(
                private_index_work_counts(),
                (
                    work_before.0 + 4,
                    work_before.1 + completed_before_failure as u64
                ),
                "all four persistent directories must exist before the first build"
            );
            let retry = inspect_rollover(&engine, next_id).unwrap();
            assert_eq!(retry.gpu_build_count, 4);
            assert_eq!(public_state(&engine), before);
            assert_eq!(
                private_index_work_counts(),
                (
                    work_before.0 + 8,
                    work_before.1 + completed_before_failure as u64 + 4
                )
            );
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn indexed_fixed_rollover_gpu_oracle_rejects_misbound_and_reversed_directories() {
        for (name, sabotage) in [
            (
                "swapped",
                physical_probe::PrivateIndexBuildSabotage::SwapFirstTwoDirectories,
            ),
            (
                "reversed_compound",
                physical_probe::PrivateIndexBuildSabotage::ReverseCompoundDescriptors,
            ),
        ] {
            let (engine, next_id) = indexed_rollover_engine();
            let before = public_state(&engine);
            let work_before = private_index_work_counts();
            physical_probe::arm_private_index_build_sabotage(sabotage);
            assert!(matches!(
                inspect_rollover(&engine, next_id),
                Err(ExecuteError::Serialization(_))
            ));
            assert_eq!(
                public_state(&engine),
                before,
                "{name} private build must not publish, write WAL, or advance eligibility"
            );
            assert_eq!(
                private_index_work_counts(),
                (work_before.0 + 4, work_before.1 + 4),
                "{name} must build every destination before the independent lookup rejects it"
            );
            let retry = inspect_rollover(&engine, next_id).unwrap();
            assert_eq!(retry.gpu_probe_count, retry.physical_index_count, "{name}");
            assert_eq!(public_state(&engine), before, "{name} retry remains inert");
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn indexed_fixed_rollover_rejects_catalog_and_physical_coverage_drift() {
        {
            let (engine, next_id) = indexed_rollover_engine();
            let plan = indexed_rollover_plan(&engine, next_id);
            let proposal = plan
                .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
                .unwrap();
            engine
                .execute_text(
                    9_001,
                    "CREATE INDEX inert_index_rollover_extra \
                     ON inert_index_rollover (code, shared)",
                )
                .unwrap();
            let drifted = public_state(&engine);
            assert!(plan
                .inspect_current_resident_index_rollover(&engine, proposal, |_| ())
                .is_err());
            assert_eq!(public_state(&engine), drifted);
        }
        for sabotage in ["coverage", "cache_geometry"] {
            let (engine, next_id) = indexed_rollover_engine();
            let catalog = engine.catalog_snapshot();
            let table = &catalog.relational_catalog["inert_index_rollover"];
            let key_id = super::super::index_probe_key_id(table, &table.indexes[0], 0).unwrap();
            let shard_id = engine
                .read_state
                .residency
                .shards
                .load()
                .get("inert_index_rollover")
                .and_then(|shards| shards.last())
                .unwrap()
                .shard_id;
            let plan = indexed_rollover_plan(&engine, next_id);
            let proposal = plan
                .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
                .unwrap();
            let cache_key = ("inert_index_rollover".to_string(), shard_id, key_id);
            match sabotage {
                "coverage" => {
                    engine
                        .read_state
                        .residency
                        .named_index_coverage
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .remove(&cache_key);
                }
                "cache_geometry" => {
                    engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .get_mut(&cache_key)
                        .unwrap()
                        .table_mask ^= 1;
                }
                _ => unreachable!(),
            }
            let drifted = public_state(&engine);
            assert!(matches!(
                plan.inspect_current_resident_index_rollover(&engine, proposal, |_| ()),
                Err(ExecuteError::Serialization(_))
            ));
            assert_eq!(public_state(&engine), drifted, "{sabotage}");
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn indexed_fixed_rollover_null_shape_declines_but_canonical_null_semantics_survive() {
        let mut engine = Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        assert!(
            hardware.driver_available && hardware.device_count != 0,
            "this ignored differential requires a real CUDA device"
        );
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(1);
        for (txn_id, sql) in [
            (
                1,
                "CREATE TABLE nullable_index_rollover \
                 (id int4 PRIMARY KEY, value int4)",
            ),
            (
                2,
                "CREATE INDEX nullable_index_rollover_value \
                 ON nullable_index_rollover (value)",
            ),
            (3, "INSERT INTO nullable_index_rollover VALUES (1, NULL)"),
        ] {
            engine.execute_text(txn_id, sql).unwrap();
        }
        engine
            .populate_relational_residency_snapshot("nullable_index_rollover")
            .unwrap();
        engine
            .publish_relational_resident_indexes("nullable_index_rollover")
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let command =
            gpu_db_sql::parse_command("INSERT INTO nullable_index_rollover VALUES (2, NULL)")
                .unwrap();
        let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
            &command,
            &catalog,
            catalog.commit_seq,
        )
        .unwrap()
        .expect("NULL batch still reaches the proof-only semantic carrier");
        let plan = crate::engine_insert_plan::PreparedDeviceInsertPlan::from_typed_batch(
            batch, &engine, &catalog,
        )
        .unwrap();
        let proposal = plan
            .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        let boundary_before = engine.committed_seq();
        assert!(matches!(
            plan.inspect_current_resident_index_rollover(&engine, proposal, |_| ()),
            Err(ExecuteError::Serialization(_))
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.committed_seq(), boundary_before);
        engine
            .execute_text(4, "INSERT INTO nullable_index_rollover VALUES (2, NULL)")
            .unwrap();
        let rows = engine
            .execute_relational_select_text(
                "SELECT id, value FROM nullable_index_rollover ORDER BY id",
            )
            .unwrap()
            .rows;
        assert_eq!(
            rows,
            vec![
                vec![crate::SqlValue::Int4(1), crate::SqlValue::Null],
                vec![crate::SqlValue::Int4(2), crate::SqlValue::Null],
            ]
        );
    }

    #[test]
    fn indexed_fixed_rollover_owner_and_entrypoint_remain_production_ineligible() {
        let source = include_str!("index_rollover.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("reservation implementation precedes tests");
        assert!(!source.contains("MutexGuard<'a, crate::CommitState>"));
        let wrapper = source
            .split("struct PreparedIndexedFixedRolloverReservation")
            .nth(1)
            .and_then(|section| {
                section
                    .split("\n}\n\n#[cfg(test)]\nimpl PreparedIndexedFixedRolloverReservation")
                    .next()
            })
            .expect("move-only rollover owner");
        assert!(
            wrapper.find("indexes:") < wrapper.find("append:")
                && wrapper.find("append:") < wrapper.find("report:")
                && wrapper.find("report:") < wrapper.find("_named_index_lifecycle:"),
            "drop order must remain private indexes -> append -> report -> lifecycle"
        );
        let owner = source
            .split("\n#[cfg(test)]\nimpl PreparedIndexedFixedRolloverReservation")
            .nth(1)
            .and_then(|section| section.split("\n/// Prepare only").next())
            .expect("scalar-only owner inspection");
        assert!(owner.contains("fn inspect"));
        for forbidden in [
            "submit_resident",
            "apply_resident",
            "bind_for_current_commit",
            "durable_wal",
            "publish_",
            "cache.insert",
            "into_parts",
        ] {
            assert!(
                !owner.contains(forbidden),
                "scalar owner inspection must not expose {forbidden}"
            );
        }
        assert!(!source.contains("pub(crate) fn new"));
        assert!(!source.contains("DeviceInsertPlan"));
        assert!(!source.contains("begin_point_index_mutation"));
        assert!(!source.contains("enter_final_publication"));
        let prepared_generation = source
            .split("struct PreparedPrivateIndexGeneration")
            .nth(1)
            .and_then(|section| {
                section
                    .split("struct ReservedPrivateIndexGeneration")
                    .next()
            })
            .expect("prepared private generation contract");
        assert!(prepared_generation.contains("duplicate_tolerant: bool"));
        assert!(source.contains("duplicate_tolerant: reserved.duplicate_tolerant"));
        assert!(source.contains("prepared_generations_preserve_catalog_duplicate_tolerance"));
        assert!(source.contains("expected_duplicate_tolerant != generation.duplicate_tolerant"));
        assert!(source.contains("generation.created_posting"));
        assert!(source.contains("generation.duplicate_tolerant"));
        let physical_probe = include_str!("index_rollover/physical_probe.rs");
        assert!(physical_probe.contains("submit_multi_shard_i32_write_locate"));
        for forbidden in [
            "apply_resident",
            "bind_for_current_commit",
            "durable_wal",
            "begin_point_index_mutation",
            "enter_final_publication",
        ] {
            assert!(
                !physical_probe.contains(forbidden),
                "private GPU lookup leaf must not expose {forbidden}"
            );
        }
        let evidence = physical_probe
            .split("pub(super) struct PrivateGpuLookupEvidence")
            .nth(1)
            .and_then(|section| section.split("}\n\nimpl PrivateGpuLookupOracle").next())
            .expect("scalar private GPU lookup evidence");
        assert!(evidence.contains("gpu_probe_count"));
        assert!(evidence.contains("bounded_readback_bytes"));
        assert!(evidence.contains("max_concurrent_readback_bytes"));
        assert!(!evidence.contains("Vec<"));
        let module = include_str!("../engine_residency.rs");
        assert!(module.contains("pub(crate) mod index_rollover;"));
        assert!(module.contains("mod indexed_reservation;"));
        let entrypoint = include_str!("../engine_insert_plan.rs")
            .split("pub(crate) fn inspect_current_resident_index_rollover")
            .nth(1)
            .and_then(|section| {
                section
                    .split("\n    /// Revalidate the catalog-order semantic carrier")
                    .next()
            })
            .expect("test-only engine-plan inspection entrypoint");
        for forbidden in [
            "bind_for_current_commit",
            "apply_resident",
            "durable_wal",
            "enter_final_publication",
        ] {
            assert!(
                !entrypoint.contains(forbidden),
                "test-only entrypoint must not expose {forbidden}"
            );
        }
        let fixed = include_str!("fixed_insert.rs");
        let production_eligibility = fixed
            .split("fn source_matches_table")
            .nth(1)
            .and_then(|section| section.split("\n}\n").next())
            .expect("production append eligibility");
        assert!(production_eligibility.contains("table.indexes.is_empty()"));
    }
}
