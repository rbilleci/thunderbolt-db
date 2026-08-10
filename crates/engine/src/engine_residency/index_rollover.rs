//! Physical ownership for one indexed fixed-width rollover reservation.
//!
//! This leaf builds a complete private payload plus every distinct physical index before WAL.
//! The common codec-5 transaction finalizer owns WAL and calls the narrow post-claim apply method;
//! this leaf owns no canonical operation, row-id allocator, durability loop, or acknowledgement.

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
    total_persistent_bytes: u64,
    max_concurrent_scratch_bytes: u64,
    max_concurrent_readback_bytes: u64,
    persistent_allocation_count: u64,
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
    resets_existing_rows: bool,
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
}

/// Move-only physical reservation. Declaration order is load-bearing: private indexes drop first, then the
/// payload/sidecar append reservation, then the named-index lifecycle guard.
pub(super) struct PreparedIndexedFixedRolloverReservation<'a> {
    indexes: PreparedPrivateRolloverIndexes,
    append: super::fixed_insert::ResidentOpenShardAppendPlan<'a>,
    manifest: super::prepared_table_index_manifest::PreparedTableIndexManifest<'a>,
    forecast: IndexedPhysicalResourceForecast,
    target_witness: IndexedPhysicalTargetWitness,
    generation_witness: IndexedPhysicalGenerationWitness,
    old_generation_pinned_bytes: u64,
    generation_pin_slots: u64,
    host_retention_peak_before_finalization: HostRetentionGeometry,
    named_index_lifecycle: Option<TransactionNamedIndexPublicationGuard<'a>>,
}

/// Dense text/validity rollover beneath the same generic codec-5 terminal. Unlike the older
/// fixed proof carrier, this owner is built directly from the exact dense payload descriptors;
/// it owns no WAL encoder, allocator, status transition, or acknowledgement path.
pub(super) struct PreparedIndexedDenseRolloverReservation<'a> {
    _indexes: PreparedPrivateRolloverIndexes,
    append: super::fixed_insert::ResidentOpenShardAppendPlan<'a>,
    manifest: super::prepared_table_index_manifest::PreparedTableIndexManifest<'a>,
    named_index_lifecycle: Option<TransactionNamedIndexPublicationGuard<'a>>,
}

impl<'a> PreparedIndexedDenseRolloverReservation<'a> {
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

    pub(super) fn manifest_successor_predecessor(
        &self,
    ) -> super::prepared_table_index_manifest::PreparedIndexedRolloverManifestPredecessor {
        self.manifest.indexed_rollover_successor_predecessor()
    }

    pub(super) fn apply_after_transaction_wal_claim(
        self,
        engine: &Engine,
        created_by: super::AppendCreatedBy<'_>,
    ) -> Result<(), super::DeviceInsertPlanApplyError> {
        let Self {
            _indexes: _,
            append,
            manifest,
            named_index_lifecycle: _,
        } = self;
        let expected_commit = match created_by {
            super::AppendCreatedBy::InsertUniform(sequence) => sequence,
            _ => return Err(super::DeviceInsertPlanApplyError::PlanDrift),
        };
        if engine.catalog_snapshot().commit_seq >= expected_commit {
            return Err(super::DeviceInsertPlanApplyError::PlanDrift);
        }
        let armed = manifest
            .arm_post_wal_for_indexed_rollover()
            .map_err(|_| super::DeviceInsertPlanApplyError::PlanDrift)?;
        let _published_generation = append.publish_indexed_dense_rollover_header()?;
        armed
            .publish_after_physical_completion(&engine.read_state.residency, &engine.read_state)
            .map_err(|_| super::DeviceInsertPlanApplyError::PublisherFailure)
    }
}

impl<'a> PreparedIndexedFixedRolloverReservation<'a> {
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

    pub(super) fn manifest_successor_predecessor(
        &self,
    ) -> super::prepared_table_index_manifest::PreparedIndexedRolloverManifestPredecessor {
        self.manifest.indexed_rollover_successor_predecessor()
    }

    pub(super) fn apply_after_transaction_wal_claim(
        self,
        engine: &Engine,
        created_by: super::AppendCreatedBy<'_>,
    ) -> Result<(), super::DeviceInsertPlanApplyError> {
        let Self {
            indexes: _,
            append,
            manifest,
            forecast: _,
            target_witness: _,
            generation_witness: _,
            old_generation_pinned_bytes: _,
            generation_pin_slots: _,
            host_retention_peak_before_finalization: _,
            named_index_lifecycle: _,
        } = self;
        let expected_commit = match created_by {
            super::AppendCreatedBy::InsertUniform(sequence) => sequence,
            _ => return Err(super::DeviceInsertPlanApplyError::PlanDrift),
        };
        let catalog = engine.catalog_snapshot();
        if catalog.commit_seq >= expected_commit {
            return Err(super::DeviceInsertPlanApplyError::PlanDrift);
        }
        let armed = manifest
            .arm_post_wal_for_indexed_rollover()
            .map_err(|_| super::DeviceInsertPlanApplyError::PlanDrift)?;
        let _published_generation = append.publish_fixed_rollover_header()?;
        armed
            .publish_after_physical_completion(&engine.read_state.residency, &engine.read_state)
            .map_err(|_| super::DeviceInsertPlanApplyError::PublisherFailure)
    }

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

/// Prepare only the fixed-rollover metadata and exact scalar capacity forecast.  No CUDA
/// allocation, driver call, lookup oracle, private payload, sidecar, or index build is permitted
/// before the resulting owner consumes an `engine_insert_plan` materialization permit.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_fixed_rollover_preview<'a>(
    engine: &'a Engine,
    table: &RelationalTable,
    predecessor_boundary: Index,
    source: crate::typed_insert_batch::PreparedResidentAppendSource,
    row_ids: super::DeviceInsertRowIds,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    mutation_gate: std::sync::MutexGuard<'a, ()>,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
    resets_existing_rows: bool,
) -> Result<IndexedFixedRolloverPreview<'a>, ExecuteError> {
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
    let logical = fixed_rollover_logical_forecast(table, &key_proof, &source)?;
    let shards = engine.read_state.residency.shards.load_full();
    let open = shards
        .get(&table.name)
        .and_then(|shards| shards.last())
        .ok_or_else(|| decline("indexed rollover forecast has no current open shard"))?;
    if (!resets_existing_rows
        && super::fixed_insert::source_is_all_i32_fixed(&source)
        && open.resident_device_null_columns.is_empty()
        && open
            .row_count
            .checked_add(source.row_count())
            .is_none_or(|end| end <= open.capacity))
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
        resets_existing_rows,
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

fn fixed_rollover_logical_forecast(
    table: &RelationalTable,
    keys: &BatchKeyConstraintProof,
    source: &crate::typed_insert_batch::PreparedResidentAppendSource,
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
        let descriptor = rollover_binding_descriptor_count(table, binding, source)?;
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
                prior_descriptor = Some(rollover_binding_descriptor_count(
                    table,
                    prior_binding,
                    source,
                )?);
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

fn rollover_binding_descriptor_count(
    table: &RelationalTable,
    binding: &crate::engine_insert_plan::batch_key_constraints::IndexBinding,
    source: &crate::typed_insert_batch::PreparedResidentAppendSource,
) -> Result<usize, ExecuteError> {
    let mut validity = BTreeSet::new();
    binding.try_for_each_resolved_catalog_column(table, |_, column| {
        let source_column = source
            .columns()
            .iter()
            .find(|candidate| {
                candidate.column_id() == column.id && candidate.attnum() == column.attnum
            })
            .ok_or_else(|| decline("indexed rollover descriptor lost its source column"))?;
        if source_column.has_validity_bitmap() {
            validity.insert((column.attnum, column.id));
        }
        Ok(())
    })?;
    binding
        .key_column_count()
        .checked_add(validity.len())
        .ok_or_else(|| decline("indexed rollover descriptor count overflows"))
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
    source: &crate::typed_insert_batch::PreparedResidentAppendSource,
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
        let binding_descriptor_count = rollover_binding_descriptor_count(table, binding, source)?;
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

fn fixed_rollover_successor_shard(
    table: &RelationalTable,
    basis: super::fixed_insert::ResidentFixedRolloverReservationBasis<'_>,
    commit_seq: Index,
) -> Result<crate::RelationalResidentShard, ExecuteError> {
    let incoming_rows = basis.incoming_rows;
    let fixed_row_bytes = table.columns.iter().try_fold(0_usize, |bytes, column| {
        let width = match column.ty {
            crate::SqlType::Int2 | crate::SqlType::Int4 | crate::SqlType::Date => 4,
            crate::SqlType::Int8 | crate::SqlType::Timestamp => 8,
            crate::SqlType::Numeric { .. } | crate::SqlType::Uuid => 16,
            crate::SqlType::Bool => 0,
            _ => return None,
        };
        bytes.checked_add(width)
    });
    let resident_bytes = fixed_row_bytes
        .and_then(|width| incoming_rows.checked_mul(width))
        .and_then(|bytes| {
            basis
                .bool_layouts
                .len()
                .checked_mul(incoming_rows.div_ceil(32))
                .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
                .and_then(|bool_bytes| bytes.checked_add(bool_bytes))
        })
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| decline("indexed rollover resident-byte geometry overflowed"))?;
    let names = |include: fn(crate::SqlType) -> bool| {
        table
            .columns
            .iter()
            .filter(|column| include(column.ty))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>()
    };
    Ok(crate::RelationalResidentShard {
        shard_id: basis.new_shard_id,
        row_start: basis.new_row_start,
        row_count: incoming_rows,
        history_floor_index: 0,
        capacity: basis.capacity,
        int4_appendable: true,
        resident_device_int4_column_stats: basis.int4_stats.to_vec(),
        resident_bytes,
        allocated_bytes: basis.payload.metadata().allocated_bytes,
        count_header_byte_offset: 0,
        resident_device_int4_columns: names(|ty| {
            matches!(
                ty,
                crate::SqlType::Int2 | crate::SqlType::Int4 | crate::SqlType::Date
            )
        }),
        resident_device_int8_columns: names(|ty| {
            matches!(ty, crate::SqlType::Int8 | crate::SqlType::Timestamp)
        }),
        resident_device_numeric_columns: names(|ty| {
            matches!(ty, crate::SqlType::Numeric { .. } | crate::SqlType::Uuid)
        }),
        resident_device_bool_columns: basis.bool_layouts.to_vec(),
        resident_device_text_columns: Vec::new(),
        resident_device_null_columns: Vec::new(),
        gpu_id: basis.payload.metadata().gpu_id,
        schema: table.schema.clone(),
        table: table.name.clone(),
        point_route_generation: Arc::new(()),
        device_memory_proof: Some(basis.payload.metadata().clone()),
        invalidated_by_txn_id: None,
        invalidated_at_index: None,
        invalidated_by_memory_pressure: false,
        memory_pressure_active: false,
        device_memory: Some(Arc::clone(basis.payload)),
        deleted_by_region: None,
        created_by_region: Some(Arc::clone(basis.created_by_region)),
        row_id_region: basis.row_id_region.map(Arc::clone),
        max_created_by: commit_seq,
    })
}

fn dense_rollover_successor_shard(
    table: &RelationalTable,
    basis: super::fixed_insert::ResidentDenseRolloverReservationBasis<'_>,
    commit_seq: Index,
) -> Result<crate::RelationalResidentShard, ExecuteError> {
    let incoming_rows = basis.incoming_rows;
    let names = |include: fn(crate::SqlType) -> bool| {
        table
            .columns
            .iter()
            .filter(|column| include(column.ty))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>()
    };
    Ok(crate::RelationalResidentShard {
        shard_id: basis.new_shard_id,
        row_start: basis.new_row_start,
        row_count: incoming_rows,
        history_floor_index: 0,
        capacity: basis.capacity,
        int4_appendable: true,
        resident_device_int4_column_stats: basis.int4_stats.to_vec(),
        resident_bytes: basis.payload_bytes,
        allocated_bytes: basis.payload.metadata().allocated_bytes,
        count_header_byte_offset: 0,
        resident_device_int4_columns: names(|ty| {
            matches!(
                ty,
                crate::SqlType::Int2 | crate::SqlType::Int4 | crate::SqlType::Date
            )
        }),
        resident_device_int8_columns: names(|ty| {
            matches!(ty, crate::SqlType::Int8 | crate::SqlType::Timestamp)
        }),
        resident_device_numeric_columns: names(|ty| {
            matches!(ty, crate::SqlType::Numeric { .. } | crate::SqlType::Uuid)
        }),
        resident_device_bool_columns: basis.bool_layouts.to_vec(),
        resident_device_text_columns: basis.text_layouts.to_vec(),
        resident_device_null_columns: basis.null_layouts.to_vec(),
        gpu_id: basis.payload.metadata().gpu_id,
        schema: table.schema.clone(),
        table: table.name.clone(),
        point_route_generation: Arc::new(()),
        device_memory_proof: Some(basis.payload.metadata().clone()),
        invalidated_by_txn_id: None,
        invalidated_at_index: None,
        invalidated_by_memory_pressure: false,
        memory_pressure_active: false,
        device_memory: Some(Arc::clone(basis.payload)),
        deleted_by_region: None,
        created_by_region: Some(Arc::clone(basis.created_by_region)),
        row_id_region: basis.row_id_region.map(Arc::clone),
        max_created_by: commit_seq,
    })
}

fn dense_rollover_build_columns(
    engine: &Engine,
    table: &RelationalTable,
    shard: &crate::RelationalResidentShard,
    binding: &crate::engine_insert_plan::batch_key_constraints::IndexBinding,
    descriptor_count: usize,
) -> Result<Vec<CudaCompoundFoldColumn>, ExecuteError> {
    let columns = binding.resident_constraint_columns(table)?;
    let snapshot = engine.resident_snapshot_for_shard(shard, table);
    let descriptors =
        crate::engine_insert_plan::resident_constraint_generation::resident_columns_for_snapshot(
            table, &snapshot, &columns,
        )?;
    if descriptors.len() != descriptor_count {
        return Err(decline(
            "indexed dense rollover descriptor geometry drifted",
        ));
    }
    Ok(descriptors)
}

/// Materialize a dense indexed rollover beneath the common transaction terminal. Every payload,
/// sidecar, directory, descriptor map and header capability exists before this returns; the
/// caller may then claim WAL and consume only the opaque physical reservation.
#[allow(clippy::too_many_arguments)]
pub(super) fn materialize_dense<'a>(
    engine: &'a Engine,
    table: &RelationalTable,
    public_table: Option<&RelationalTable>,
    created_index_ids: &[u64],
    retired_index_ids: &[u64],
    predecessor_boundary: Index,
    source: crate::typed_insert_batch::PreparedResidentAppendSource,
    row_ids: super::DeviceInsertRowIds,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    mutation_gate: std::sync::MutexGuard<'a, ()>,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
    expected_commit_seq: Index,
    permit: IndexedPhysicalMaterializationPermit,
    preheld_budget_allocation: Option<std::sync::MutexGuard<'a, ()>>,
    prior_reserved_bytes: u64,
    manifest_predecessor: Option<
        super::prepared_table_index_manifest::PreparedIndexedRolloverManifestPredecessor,
    >,
    resets_existing_rows: bool,
) -> Result<PreparedIndexedDenseRolloverReservation<'a>, ExecuteError> {
    let catalog = engine.catalog_snapshot();
    let public_table = public_table.unwrap_or(table);
    let current_table = catalog
        .relational_catalog
        .get(&table.name)
        .ok_or_else(|| decline("indexed dense rollover lost current table"))?;
    let shards = engine.read_state.residency.shards.load_full();
    let open = shards
        .get(&table.name)
        .and_then(|shards| shards.last())
        .ok_or_else(|| decline("indexed dense rollover has no predecessor shard"))?;
    let current_table_matches = current_table == public_table;
    let catalog_covers_source = catalog.commit_seq >= source.prepared_catalog_seq();
    let dense_source = source.requires_dense_rollover();
    let nonempty_source = source.row_count() != 0;
    let exact_row_ids = row_ids.is_exact() && row_ids.exact_len_matches(source.row_count());
    let has_s3_index_transition = !created_index_ids.is_empty() || !retired_index_ids.is_empty();
    let source_matches =
        super::fixed_insert::source_matches_indexed_in_place_reservation(&source, table)
            || (has_s3_index_transition
                && super::fixed_insert::source_matches_s3_created_index_reservation(
                    &source, table,
                ));
    let predecessor_matches = validation.matches_fixed_rollover_predecessor(
        table,
        catalog.commit_seq,
        open,
        predecessor_boundary,
    );
    let snapshot_is_covered = validation.original_read_snapshot() <= predecessor_boundary;
    let enrollment_is_complete = public_table.indexes.is_empty()
        || physical_probe::published_index_enrollment_is_complete(engine, public_table);
    let created_indexes_are_exact = created_index_ids.iter().enumerate().all(|(ordinal, id)| {
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
    let retired_indexes_are_exact = retired_index_ids.iter().enumerate().all(|(ordinal, id)| {
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
    if !current_table_matches
        || !catalog_covers_source
        || !dense_source
        || !nonempty_source
        || !exact_row_ids
        || !source_matches
        || !predecessor_matches
        || !snapshot_is_covered
        || !enrollment_is_complete
        || !created_indexes_are_exact
        || !retired_indexes_are_exact
        || created_index_ids
            .iter()
            .any(|id| retired_index_ids.binary_search(id).is_ok())
    {
        #[cfg(feature = "probe-timing")]
        eprintln!(
            "[probe] indexed_dense_witness table={} current_table={} catalog={} dense={} nonempty={} row_ids={} source={} predecessor={} snapshot={} enrollment={}",
            table.name,
            current_table_matches,
            catalog_covers_source,
            dense_source,
            nonempty_source,
            exact_row_ids,
            source_matches,
            predecessor_matches,
            snapshot_is_covered,
            enrollment_is_complete,
        );
        return Err(decline(
            "indexed dense rollover source or predecessor witness drifted",
        ));
    }
    let logical_forecast = fixed_rollover_logical_forecast(table, &key_proof, &source)?;
    let logical = prepare_logical_bindings(table, &key_proof, &source, logical_forecast)?;
    let predecessor_shard_id = open.shard_id;
    let predecessor_row_start = open.row_start;
    let predecessor_row_count = open.row_count;
    let predecessor_gpu_id = open.gpu_id;
    let predecessor_shards = shards
        .get(&table.name)
        .cloned()
        .ok_or_else(|| decline("indexed dense rollover lost predecessor shard set"))?;
    drop(shards);

    let append = engine
        .prepare_resident_open_shard_append_indexed_fixed_rollover_reservation(
            source,
            row_ids,
            mutation_gate,
            logical.max_concurrent_scratch_bytes,
            permit,
            preheld_budget_allocation,
            prior_reserved_bytes,
            resets_existing_rows,
            has_s3_index_transition,
            has_s3_index_transition.then_some(table),
        )
        .map_err(|_| decline("indexed dense rollover append reservation declined"))?;
    let basis = append
        .indexed_dense_rollover_reservation_basis()
        .ok_or_else(|| decline("indexed dense rollover lost its private payload"))?;
    if basis.predecessor_shard_id != predecessor_shard_id
        || basis.predecessor_row_count != predecessor_row_count
        || if resets_existing_rows {
            basis.new_row_start != 0
        } else {
            predecessor_row_start.checked_add(predecessor_row_count) != Some(basis.new_row_start)
        }
        || basis.catalog_seq != catalog.commit_seq
        || basis.capacity != basis.incoming_rows
        || basis.payload.metadata().gpu_id != predecessor_gpu_id
        || basis.max_index_scratch_bytes != logical.max_concurrent_scratch_bytes
    {
        return Err(decline("indexed dense rollover append geometry drifted"));
    }
    let successor = dense_rollover_successor_shard(table, basis, expected_commit_seq)?;
    let table_size =
        super::resident_shard_index_table_size(basis.incoming_rows as u64, basis.capacity as u64)
            .ok_or_else(|| decline("indexed dense rollover index horizon overflows"))?;
    let table_mask = u32::try_from(table_size - 1)
        .map_err(|_| decline("indexed dense rollover table mask overflows"))?;
    let hash_shift = 32 - table_size.trailing_zeros();
    let one_index_bytes = resident_index_allocated_bytes(
        table_mask,
        u64::try_from(basis.capacity)
            .map_err(|_| decline("indexed dense rollover capacity overflows"))?,
    )
    .ok_or_else(|| decline("indexed dense rollover index allocation overflows"))?;
    let expected_index_bytes = one_index_bytes
        .checked_mul(logical.physical.len() as u64)
        .ok_or_else(|| decline("indexed dense rollover total index bytes overflow"))?;
    if expected_index_bytes != basis.planned_index_allocation_bytes {
        return Err(decline(
            "indexed dense rollover planned index bytes drifted",
        ));
    }

    let mutation_epoch = engine
        .read_state
        .residency
        .point_index_mutation_epoch_for_table(&engine.read_state, public_table)
        .ok_or_else(|| decline("indexed dense rollover relation identity changed"))?;
    let mutation_epoch_expected_even = mutation_epoch.load(Ordering::Acquire);
    if mutation_epoch_expected_even & 1 != 0
        || basis
            .payload
            .read_resident_u64_column(0, 1)
            .map_err(|_| decline("indexed dense rollover header readback failed"))?
            != [0]
    {
        return Err(decline(
            "indexed dense rollover observed an active or public generation",
        ));
    }

    let runtime = engine.cuda_driver_probe_runtime();
    let mut reserved = Vec::with_capacity(logical.physical.len());
    for physical in logical.physical.iter() {
        let binding = key_proof
            .indexes()
            .get(physical.raw_ordinal)
            .filter(|binding| binding.raw_ordinal() == physical.raw_ordinal)
            .ok_or_else(|| decline("indexed dense rollover build binding drifted"))?;
        let build_columns = dense_rollover_build_columns(
            engine,
            table,
            &successor,
            binding,
            physical.descriptor_count,
        )?;
        let memory = Arc::new(
            runtime
                .retain_device_memory_zeroed(predecessor_gpu_id, one_index_bytes)
                .map_err(|error| {
                    decline(format!(
                        "indexed dense rollover private index allocation failed: {error}"
                    ))
                })?,
        );
        reserved.push(ReservedPrivateIndexGeneration {
            raw_ordinal: physical.raw_ordinal,
            key_id: physical.key_id,
            memory,
            table_mask,
            hash_shift,
            allocated_bytes: one_index_bytes,
            descriptor_count: physical.descriptor_count,
            duplicate_tolerant: physical.duplicate_tolerant,
            build_columns,
        });
    }
    reserved.sort_unstable_by_key(|generation| {
        (
            std::cmp::Reverse(generation.descriptor_count),
            generation.raw_ordinal,
        )
    });
    let allocation_scope = CudaAllocationScope::with_budget(logical.max_concurrent_scratch_bytes);
    let mut generations = Vec::with_capacity(reserved.len());
    let expected_build_host_scratch =
        resident_typed_index_build_host_scratch_geometry(logical.max_descriptor_count)
            .ok_or_else(|| decline("indexed dense rollover host scratch overflows"))?;
    for reserved in reserved {
        let (status, host_scratch) = basis
            .payload
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
                    "indexed dense rollover GPU index build failed: {error}"
                ))
            })?;
        if status.declined
            || host_scratch.bytes > expected_build_host_scratch.bytes
            || host_scratch.allocation_slots > expected_build_host_scratch.allocation_slots
        {
            return Err(decline("indexed dense rollover GPU build declined"));
        }
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
    }
    if allocation_scope.peak_bytes() > logical.max_concurrent_scratch_bytes
        || mutation_epoch.load(Ordering::Acquire) != mutation_epoch_expected_even
        || basis
            .payload
            .read_resident_u64_column(0, 1)
            .map_err(|_| decline("indexed dense rollover post-build header readback failed"))?
            != [0]
    {
        return Err(decline(
            "indexed dense rollover build escaped its sealed resource envelope",
        ));
    }
    drop(allocation_scope);
    generations.sort_unstable_by_key(|generation| generation.raw_ordinal);
    if !prepared_generations_preserve_catalog_duplicate_tolerance(table, &logical, &generations) {
        return Err(decline(
            "indexed dense rollover duplicate-tolerance mapping drifted",
        ));
    }
    let actual_index_bytes = generations.iter().try_fold(0_u64, |total, generation| {
        total
            .checked_add(generation.allocated_bytes)
            .ok_or_else(|| decline("indexed dense rollover actual index bytes overflow"))
    })?;
    let total_persistent_bytes = basis
        .payload_sidecar_allocation_bytes
        .checked_add(actual_index_bytes)
        .ok_or_else(|| decline("indexed dense rollover persistent bytes overflow"))?;
    let persistent_allocation_count = basis
        .payload_sidecar_allocation_count
        .checked_add(generations.len() as u64)
        .ok_or_else(|| decline("indexed dense rollover allocation count overflows"))?;
    let ledger = FixedRolloverResourceLedger {
        total_persistent_bytes,
        max_concurrent_scratch_bytes: logical.max_concurrent_scratch_bytes,
        max_concurrent_readback_bytes: std::mem::size_of::<u32>() as u64,
        persistent_allocation_count,
    };
    let PreparedFixedRolloverLogicalBindings {
        raw: raw_bindings,
        physical,
        ..
    } = logical;
    drop(physical);
    let generations = generations.into_boxed_slice();
    let source_payload = Arc::clone(basis.payload);

    let mut append = append;
    append
        .prepare_indexed_dense_rollover_uniform_commit(expected_commit_seq)
        .map_err(|_| decline("indexed dense rollover header preparation declined"))?;
    let publication_basis = append
        .indexed_dense_rollover_reservation_basis()
        .ok_or_else(|| decline("indexed dense rollover publication basis disappeared"))?;
    let new_shard = dense_rollover_successor_shard(table, publication_basis, expected_commit_seq)?;
    let manifest_entries = generations
        .iter()
        .map(|generation| {
            super::prepared_table_index_manifest::PreparedRolloverIndexManifestEntry {
                key_id: generation.key_id,
                memory: Arc::clone(&generation.memory),
                table_mask: generation.table_mask,
                hash_shift: generation.hash_shift,
                duplicate_tolerant: generation.duplicate_tolerant,
                has_postings: generation.created_posting,
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let prefix_entries = created_index_ids
        .iter()
        .map(|created_id| {
            table
                .indexes
                .iter()
                .enumerate()
                .find(|(_, index)| u64::from(index.oid) == *created_id)
                .ok_or_else(|| decline("indexed dense rollover lost S3-created index"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flat_map(|(ordinal, index)| {
            predecessor_shards
                .iter()
                .filter(|shard| shard.row_count != 0)
                .map(move |shard| (ordinal, index, shard))
        })
        .map(|(ordinal, index, shard)| {
            engine.prepare_s3_created_index_manifest_entry(
                table,
                index,
                ordinal,
                shard,
                predecessor_boundary,
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_boxed_slice();
    let final_key_ids = table
        .indexes
        .iter()
        .enumerate()
        .map(|(ordinal, index)| crate::engine_residency::index_probe_key_id(table, index, ordinal))
        .collect::<Option<std::collections::BTreeSet<_>>>()
        .ok_or_else(|| decline("indexed dense rollover lost a final index key"))?;
    let retired_key_ids = retired_index_ids
        .iter()
        .map(|retired_id| {
            public_table
                .indexes
                .iter()
                .enumerate()
                .find(|(_, index)| u64::from(index.oid) == *retired_id)
                .and_then(|(ordinal, index)| {
                    crate::engine_residency::index_probe_key_id(public_table, index, ordinal)
                })
                // A raw one-column directory may be shared by a surviving catalog index. In
                // that case it remains the same device directory; only keys absent from the
                // final catalog are retired from the manifest.
                .filter(|key_id| !final_key_ids.contains(key_id))
                .ok_or_else(|| decline("indexed dense rollover lost an S3-retired index key"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let manifest =
        super::prepared_table_index_manifest::PreparedTableIndexManifest::prepare_indexed_rollover(
            &engine.read_state.residency,
            &engine.read_state,
            table,
            has_s3_index_transition.then_some(public_table),
            new_shard,
            manifest_entries,
            prefix_entries,
            retired_key_ids,
            expected_commit_seq,
            &mutation_epoch,
            mutation_epoch_expected_even,
            manifest_predecessor,
            resets_existing_rows,
        )
        .map_err(|error| {
            decline(format!(
                "indexed dense rollover manifest preparation declined: {error:?}"
            ))
        })?;
    Ok(PreparedIndexedDenseRolloverReservation {
        _indexes: PreparedPrivateRolloverIndexes {
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
        manifest,
        named_index_lifecycle: Some(named_index_lifecycle),
    })
}

/// Consume the capacity forecast only after `engine_insert_plan` has issued the move-only
/// capability.  All CUDA-facing preparation is deliberately below this line.
#[allow(clippy::too_many_arguments)] // explicit predecessor/budget capabilities prevent a second authority
pub(super) fn materialize<'a>(
    engine: &'a Engine,
    table: &RelationalTable,
    preview: IndexedFixedRolloverPreview<'a>,
    expected_commit_seq: Index,
    permit: IndexedPhysicalMaterializationPermit,
    preheld_budget_allocation: Option<std::sync::MutexGuard<'a, ()>>,
    prior_reserved_bytes: u64,
    manifest_predecessor: Option<
        super::prepared_table_index_manifest::PreparedIndexedRolloverManifestPredecessor,
    >,
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
        resets_existing_rows,
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

    let mut logical = prepare_logical_bindings(table, &key_proof, &source, expected_logical)?;
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
            preheld_budget_allocation,
            prior_reserved_bytes,
            resets_existing_rows,
            false,
            None,
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
        expected_commit_seq,
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
        manifest_predecessor,
        resets_existing_rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare<'a>(
    engine: &'a Engine,
    table: &RelationalTable,
    predecessor_boundary: Index,
    expected_commit_seq: Index,
    mut append: super::fixed_insert::ResidentOpenShardAppendPlan<'a>,
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
    manifest_predecessor: Option<
        super::prepared_table_index_manifest::PreparedIndexedRolloverManifestPredecessor,
    >,
    resets_existing_rows: bool,
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
        || if resets_existing_rows {
            basis.new_row_start != 0
        } else {
            basis
                .predecessor_row_count
                .checked_add(predecessor.row_start)
                != Some(basis.new_row_start)
        }
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
    let ledger = FixedRolloverResourceLedger {
        total_persistent_bytes,
        max_concurrent_scratch_bytes: logical.max_concurrent_scratch_bytes,
        max_concurrent_readback_bytes: (std::mem::size_of::<u32>() as u64)
            .max(probe_evidence.max_concurrent_readback_bytes),
        persistent_allocation_count,
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
    append
        .prepare_fixed_rollover_uniform_commit(expected_commit_seq)
        .map_err(|_| decline("indexed rollover header publication preparation declined"))?;
    let publication_basis = append
        .indexed_fixed_rollover_reservation_basis()
        .ok_or_else(|| decline("indexed rollover publication lost its fixed basis"))?;
    let new_shard = fixed_rollover_successor_shard(table, publication_basis, expected_commit_seq)?;
    let manifest_entries = generations
        .iter()
        .map(|generation| {
            super::prepared_table_index_manifest::PreparedRolloverIndexManifestEntry {
                key_id: generation.key_id,
                memory: Arc::clone(&generation.memory),
                table_mask: generation.table_mask,
                hash_shift: generation.hash_shift,
                duplicate_tolerant: generation.duplicate_tolerant,
                has_postings: generation.created_posting,
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let manifest =
        super::prepared_table_index_manifest::PreparedTableIndexManifest::prepare_indexed_rollover(
            &engine.read_state.residency,
            &engine.read_state,
            table,
            None,
            new_shard,
            manifest_entries,
            Box::new([]),
            vec![],
            expected_commit_seq,
            &mutation_epoch,
            mutation_epoch_expected_even,
            manifest_predecessor,
            resets_existing_rows,
        )
        .map_err(|error| {
            decline(format!(
                "indexed rollover manifest preparation declined: {error:?}"
            ))
        })?;
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
        manifest,
        forecast: resource_forecast,
        target_witness,
        generation_witness,
        old_generation_pinned_bytes,
        generation_pin_slots,
        host_retention_peak_before_finalization: host_peak.peak(),
        named_index_lifecycle: Some(named_index_lifecycle),
    };
    if !reservation.forecast_matches_actual()? {
        return Err(decline(
            "indexed rollover materialization owner ledger diverged from forecast",
        ));
    }
    Ok(reservation)
}

fn decline(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Serialization(message.into())
}
