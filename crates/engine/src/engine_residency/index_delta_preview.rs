//! Zero-CUDA preflight for the unreachable indexed in-place INSERT reservation.
//!
//! This owner pins every generation, cache, and descriptor witness needed by the allocating
//! index-delta tail.  It deliberately has no CUDA allocation, launch, cache publication, WAL,
//! or mutation surface: a later owner must consume and revalidate this preview before it may
//! reserve either append-sidecar or pooled index-preparation resources.

#![allow(dead_code)] // private preflight is compiled before the live reservation handoff opens

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, MutexGuard};

use super::fixed_insert::ResidentOpenShardAppendPlan;
use super::indexed_forecast::{
    IndexedPhysicalGenerationWitness, IndexedPhysicalResourceForecast, IndexedPhysicalTargetWitness,
};
use crate::engine_insert_plan::batch_key_constraints::BatchKeyConstraintProof;
use crate::engine_insert_plan::host_retention::{HostRetentionGeometry, HostRetentionReport};
use crate::engine_insert_plan::{
    resident_constraint_generation, resident_key_constraints::ResidentKeyValidationSeal,
};
use crate::engine_state::TransactionNamedIndexPublicationGuard;
use crate::relational_model::RelationalTable;
use crate::typed_insert_batch::{PreparedResidentAppendSource, TypedInsertBatch};
use crate::{Engine, ExecuteError, Index, RelationalResidentShard};
use gpu_db_execution::{
    resident_index_allocated_bytes, resident_typed_indexes_insert_preparation_bytes,
    CudaCompoundFoldColumn, CudaResidentDeviceMemory, CudaResidentTypedIndexInsert,
};

/// Every catalog index remains represented in raw catalog order. Multiple raw indexes reuse a
/// physical directory only when their established probe key says that they may.
pub(super) struct RawIndexLogicalBinding {
    pub(super) raw_ordinal: usize,
    pub(super) key_id: usize,
    pub(super) physical_ordinal: usize,
}

pub(super) struct PhysicalIndexLogicalBinding {
    pub(super) raw_ordinal: usize,
    pub(super) key_id: usize,
    pub(super) descriptor_count: usize,
}

pub(super) struct PreparedIndexDeltaLogicalBindings {
    pub(super) raw: Box<[RawIndexLogicalBinding]>,
    pub(super) physical: Box<[PhysicalIndexLogicalBinding]>,
    preparation_bytes: u64,
}

impl PreparedIndexDeltaLogicalBindings {
    pub(super) fn preparation_bytes(&self) -> u64 {
        self.preparation_bytes
    }

    pub(super) fn append_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), ExecuteError> {
        report.retain_boxed_slice(&self.raw)?;
        report.retain_boxed_slice(&self.physical)?;
        Ok(())
    }

    fn host_retention_geometry(&self) -> Result<HostRetentionGeometry, ExecuteError> {
        let mut geometry = HostRetentionGeometry::default();
        geometry.checked_add_backing_elements::<RawIndexLogicalBinding>(
            self.raw.len(),
            "indexed preview raw logical binding box",
        )?;
        geometry.checked_add_backing_elements::<PhysicalIndexLogicalBinding>(
            self.physical.len(),
            "indexed preview physical logical binding box",
        )?;
        Ok(geometry)
    }
}

/// Exact cache evidence for one physical descriptor request. These Arcs remain distinct from the
/// cache maps so retirement cannot invalidate a preview's basis before its consumer revalidates.
pub(super) struct PhysicalIndexBinding {
    pub(super) cache_key: (String, u32, usize),
    pub(super) key_id: usize,
    pub(super) device_index: Arc<CudaResidentDeviceMemory>,
    pub(super) resident_guard: Arc<CudaResidentDeviceMemory>,
    pub(super) published_row_count: Arc<AtomicUsize>,
    pub(super) published_has_postings: Arc<AtomicBool>,
    pub(super) has_postings_at_preview: bool,
    pub(super) source_ptr: u64,
    pub(super) base_row: usize,
    pub(super) end_row: usize,
    pub(super) capacity: usize,
    pub(super) table_mask: u32,
    pub(super) hash_shift: u32,
    pub(super) gc_boundary: Index,
    pub(super) allocated_bytes: u64,
}

impl PhysicalIndexBinding {
    pub(super) fn append_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), ExecuteError> {
        report.retain_string(&self.cache_key.0)?;
        // Device-memory Arcs are GPU allocation pins, while these metadata atomics are ordinary
        // statement-retired host owners and must charge both payload bytes and allocation slots.
        report.retain_arc_owner(&self.published_row_count)?;
        report.retain_arc_owner(&self.published_has_postings)?;
        Ok(())
    }

    fn append_host_retention_geometry(
        &self,
        geometry: &mut HostRetentionGeometry,
    ) -> Result<(), ExecuteError> {
        append_string_geometry(geometry, &self.cache_key.0, "indexed preview cache key")?;
        geometry.checked_add_backing_elements::<AtomicUsize>(
            1,
            "indexed preview published-row owner",
        )?;
        geometry.checked_add_backing_elements::<AtomicBool>(1, "indexed preview posting owner")?;
        Ok(())
    }
}

/// Scalar ledger determined before any CUDA allocation scope or typed-index preparation exists.
/// Persistent bytes are pins, not new allocations. A missing `created_by` region records the
/// exact later capacity allocation owned by the append compiler.
pub(super) struct IndexDeltaResourceLedger {
    pub(super) source_payload_bytes: u64,
    pub(super) pinned_persistent_index_bytes: u64,
    pub(super) pinned_created_by_bytes: u64,
    pub(super) pinned_row_id_bytes: u64,
    pub(super) pinned_deleted_by_bytes: u64,
    pub(super) retained_persistent_bytes: u64,
    pub(super) pending_created_by_bytes: u64,
    pub(super) transient_preparation_bytes: u64,
    pub(super) transient_allocation_slot_count: usize,
    pub(super) descriptor_bytes: u64,
    pub(super) descriptor_count: usize,
    pub(super) bounded_readback_bytes: u64,
    pub(super) retained_allocation_pin_count: usize,
    pub(super) pending_created_by_allocation_count: usize,
    pub(super) raw_index_count: usize,
    pub(super) physical_index_count: usize,
}

/// Exact immutable in-place identity captured below commit -> named-index lifecycle -> mutation.
/// The private source and row identities remain sealed until consumption; no caller can pair a
/// preview with independently supplied append input.
pub(super) struct IndexedInPlacePreview<'a> {
    pub(super) source: PreparedResidentAppendSource,
    pub(super) row_ids: super::DeviceInsertRowIds,
    pub(super) table_name: String,
    pub(super) table_oid: u32,
    pub(super) schema_digest: gpu_db_wal::CanonicalDigest,
    pub(super) catalog_seq: Index,
    pub(super) predecessor_boundary: Index,
    pub(super) shard_id: u32,
    pub(super) row_start: usize,
    pub(super) base_row: usize,
    pub(super) end_row: usize,
    pub(super) capacity: usize,
    pub(super) gpu_id: u16,
    pub(super) schema: String,
    pub(super) source_payload: Arc<CudaResidentDeviceMemory>,
    pub(super) point_route_generation: Arc<()>,
    pub(super) created_by_region: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) row_id_region: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) deleted_by_region: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) key_proof: BatchKeyConstraintProof,
    pub(super) validation: ResidentKeyValidationSeal,
    pub(super) logical: PreparedIndexDeltaLogicalBindings,
    pub(super) requests: Box<[CudaResidentTypedIndexInsert]>,
    pub(super) physical_bindings: Box<[PhysicalIndexBinding]>,
    pub(super) mutation_epoch: Arc<AtomicU64>,
    pub(super) mutation_epoch_expected_even: u64,
    pub(super) ledger: IndexDeltaResourceLedger,
    fused_footprint: gpu_db_execution::FusedApplyPreparationFootprint,
    fused_final_host_geometry: HostRetentionGeometry,
    fused_materialization_scratch: HostRetentionGeometry,
    fused_materialization_peak: HostRetentionGeometry,
    target_witness: IndexedPhysicalTargetWitness,
    generation_witness: IndexedPhysicalGenerationWitness,
    resource_forecast: IndexedPhysicalResourceForecast,
    materialized_append_host_geometry: HostRetentionGeometry,
    pub(super) mutation_gate: MutexGuard<'a, ()>,
    pub(super) named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
}

/// Scalar-only preview evidence. Device resources remain private and can only be compared by the
/// narrow test inspection methods below.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexedInPlacePreviewReport {
    pub(crate) raw_index_count: usize,
    pub(crate) physical_index_count: usize,
    pub(crate) base_row: usize,
    pub(crate) incoming_rows: usize,
    pub(crate) end_row: usize,
    pub(crate) capacity: usize,
    pub(crate) preparation_bytes: u64,
    pub(crate) source_payload_bytes: u64,
    pub(crate) pinned_persistent_index_bytes: u64,
    pub(crate) pinned_created_by_bytes: u64,
    pub(crate) pinned_row_id_bytes: u64,
    pub(crate) pinned_deleted_by_bytes: u64,
    pub(crate) retained_persistent_bytes: u64,
    pub(crate) pending_created_by_bytes: u64,
    pub(crate) descriptor_bytes: u64,
    pub(crate) descriptor_count: usize,
    pub(crate) bounded_readback_bytes: u64,
    pub(crate) retained_allocation_pin_count: usize,
    pub(crate) pending_created_by_allocation_count: usize,
    pub(crate) transient_allocation_slot_count: usize,
    pub(crate) mutation_epoch: u64,
    pub(crate) fused_preparation_bytes: u64,
    pub(crate) fused_pooled_allocation_slots: u64,
    pub(crate) fused_owner_array_bytes: u64,
    pub(crate) fused_staging_bytes: u64,
    pub(crate) fused_stamp_bytes: u64,
    pub(crate) fused_materialization_scratch_bytes: u64,
    pub(crate) combined_preparation_bytes: u64,
    pub(crate) combined_allocation_slots: u64,
    pub(crate) host_retained_bytes: u64,
    pub(crate) host_allocation_slots: u64,
    pub(crate) host_generation_pin_slots: u64,
}

impl<'a> IndexedInPlacePreview<'a> {
    pub(super) fn resource_forecast(&self) -> IndexedPhysicalResourceForecast {
        self.resource_forecast
    }

    pub(super) fn target_witness(&self) -> &IndexedPhysicalTargetWitness {
        &self.target_witness
    }

    pub(super) fn generation_witness(&self) -> &IndexedPhysicalGenerationWitness {
        &self.generation_witness
    }

    /// Complete host retention for the zero-CUDA preview. Row-ID bytes, device allocation guards,
    /// point-route generation and mutation/lifecycle locks belong to their own domains; the
    /// commit proof is intentionally borrowed and never retained. The exact row-ID box
    /// contributes only one generic allocation slot.
    pub(super) fn host_retention_report(&self) -> Result<HostRetentionReport, ExecuteError> {
        let mut report = self.source.host_retention_report()?;
        self.row_ids.append_host_allocation_slot(&mut report)?;
        report.retain_string(&self.table_name)?;
        report.retain_string(&self.schema)?;
        self.key_proof.append_host_retention(&mut report)?;
        self.validation.append_host_retention(&mut report)?;
        self.logical.append_host_retention(&mut report)?;
        report.retain_boxed_slice(&self.requests)?;
        for request in self.requests.iter() {
            report.retain_vec(&request.columns)?;
        }
        report.retain_boxed_slice(&self.physical_bindings)?;
        for binding in self.physical_bindings.iter() {
            binding.append_host_retention(&mut report)?;
        }
        report.retain_arc_owner(&self.mutation_epoch)?;
        Ok(report)
    }

    /// Complete pre-lease scalar geometry. Every direct owner below is private to this preview;
    /// shared source identities were already de-duplicated by the source's bounded scalar walk.
    /// The identity-map report remains a post-materialization/test diagnostic only.
    fn host_retention_geometry(&self) -> Result<HostRetentionGeometry, ExecuteError> {
        let mut geometry = self.source.host_retention_geometry()?;
        geometry.checked_add_disjoint(
            self.row_ids.host_allocation_geometry()?,
            "indexed preview row-id owner",
        )?;
        append_string_geometry(
            &mut geometry,
            &self.table_name,
            "indexed preview table name",
        )?;
        append_string_geometry(&mut geometry, &self.schema, "indexed preview schema")?;
        geometry.checked_add_disjoint(
            self.key_proof.host_retention_geometry()?,
            "indexed preview key proof",
        )?;
        geometry.checked_add_disjoint(
            self.validation.host_retention_geometry()?,
            "indexed preview validation seal",
        )?;
        geometry.checked_add_disjoint(
            self.logical.host_retention_geometry()?,
            "indexed preview logical bindings",
        )?;
        geometry.checked_add_backing_elements::<CudaResidentTypedIndexInsert>(
            self.requests.len(),
            "indexed preview request box",
        )?;
        for request in self.requests.iter() {
            if request.columns.capacity() != 0 {
                geometry.checked_add_backing_elements::<CudaCompoundFoldColumn>(
                    request.columns.capacity(),
                    "indexed preview request columns",
                )?;
            }
        }
        geometry.checked_add_backing_elements::<PhysicalIndexBinding>(
            self.physical_bindings.len(),
            "indexed preview physical binding box",
        )?;
        for binding in self.physical_bindings.iter() {
            binding.append_host_retention_geometry(&mut geometry)?;
        }
        geometry
            .checked_add_backing_elements::<AtomicU64>(1, "indexed preview mutation epoch owner")?;
        Ok(geometry)
    }

    /// Scalar final-owner geometry after requests/temporary logical bindings retire.  This must
    /// stay allocation-free: the materialized owner validates it against its identity-aware
    /// report before private CUDA preparation can escape this branch.
    fn final_host_retention_geometry(&self) -> Result<HostRetentionGeometry, ExecuteError> {
        let mut geometry = self.source.host_retention_geometry()?;
        geometry.checked_add_disjoint(
            self.materialized_append_host_geometry,
            "indexed in-place materialized append owner",
        )?;
        geometry.checked_add_disjoint(
            self.key_proof.host_retention_geometry()?,
            "indexed in-place final key proof",
        )?;
        geometry.checked_add_disjoint(
            self.validation.host_retention_geometry()?,
            "indexed in-place final validation seal",
        )?;
        geometry.checked_add_backing_elements::<RawIndexLogicalBinding>(
            self.logical.raw.len(),
            "indexed in-place final raw binding box",
        )?;
        geometry.checked_add_backing_elements::<PhysicalIndexBinding>(
            self.physical_bindings.len(),
            "indexed in-place final physical binding box",
        )?;
        for binding in self.physical_bindings.iter() {
            binding.append_host_retention_geometry(&mut geometry)?;
        }
        geometry.checked_add_backing_elements::<AtomicU64>(
            1,
            "indexed in-place final mutation epoch owner",
        )?;
        geometry.checked_add_backing_elements::<Arc<CudaResidentDeviceMemory>>(
            self.physical_bindings.len(),
            "indexed in-place prepared launch owner array",
        )?;
        Ok(geometry)
    }

    fn materializing_base_host_retention_geometry(
        &self,
    ) -> Result<HostRetentionGeometry, ExecuteError> {
        let mut geometry = self.source.host_retention_geometry()?;
        geometry.checked_add_disjoint(
            self.materialized_append_host_geometry,
            "indexed in-place materializing append owner",
        )?;
        geometry.checked_add_disjoint(
            self.key_proof.host_retention_geometry()?,
            "indexed in-place materializing key proof",
        )?;
        geometry.checked_add_disjoint(
            self.validation.host_retention_geometry()?,
            "indexed in-place materializing validation seal",
        )?;
        geometry.checked_add_disjoint(
            self.logical.host_retention_geometry()?,
            "indexed in-place materializing logical bindings",
        )?;
        geometry.checked_add_backing_elements::<CudaResidentTypedIndexInsert>(
            self.requests.len(),
            "indexed in-place materializing request box",
        )?;
        for request in self.requests.iter() {
            geometry.checked_add_backing_elements::<CudaCompoundFoldColumn>(
                request.columns.capacity(),
                "indexed in-place materializing request columns",
            )?;
        }
        geometry.checked_add_backing_elements::<PhysicalIndexBinding>(
            self.physical_bindings.len(),
            "indexed in-place materializing physical binding box",
        )?;
        for binding in self.physical_bindings.iter() {
            binding.append_host_retention_geometry(&mut geometry)?;
        }
        geometry.checked_add_backing_elements::<AtomicU64>(
            1,
            "indexed in-place materializing mutation epoch owner",
        )?;
        Ok(geometry)
    }

    fn materializing_launch_host_retention_geometry(
        &self,
    ) -> Result<HostRetentionGeometry, ExecuteError> {
        let mut geometry = self.materializing_base_host_retention_geometry()?;
        geometry.checked_add_backing_elements::<Arc<CudaResidentDeviceMemory>>(
            self.physical_bindings.len(),
            "indexed in-place materializing prepared launch owner array",
        )?;
        Ok(geometry)
    }

    #[cfg(test)]
    pub(super) fn scalar_report(&self) -> IndexedInPlacePreviewReport {
        let host = self
            .host_retention_report()
            .expect("indexed preview host retention remains checked");
        let scalar = self
            .host_retention_geometry()
            .expect("indexed preview scalar host geometry remains checked");
        assert_eq!(
            scalar,
            host.geometry().expect("indexed preview host geometry fits"),
            "pre-lease scalar geometry must match the post-materialization identity report"
        );
        IndexedInPlacePreviewReport {
            raw_index_count: self.ledger.raw_index_count,
            physical_index_count: self.ledger.physical_index_count,
            base_row: self.base_row,
            incoming_rows: self.source.row_count(),
            end_row: self.end_row,
            capacity: self.capacity,
            preparation_bytes: self.ledger.transient_preparation_bytes,
            source_payload_bytes: self.ledger.source_payload_bytes,
            pinned_persistent_index_bytes: self.ledger.pinned_persistent_index_bytes,
            pinned_created_by_bytes: self.ledger.pinned_created_by_bytes,
            pinned_row_id_bytes: self.ledger.pinned_row_id_bytes,
            pinned_deleted_by_bytes: self.ledger.pinned_deleted_by_bytes,
            retained_persistent_bytes: self.ledger.retained_persistent_bytes,
            pending_created_by_bytes: self.ledger.pending_created_by_bytes,
            descriptor_bytes: self.ledger.descriptor_bytes,
            descriptor_count: self.ledger.descriptor_count,
            bounded_readback_bytes: self.ledger.bounded_readback_bytes,
            retained_allocation_pin_count: self.ledger.retained_allocation_pin_count,
            pending_created_by_allocation_count: self.ledger.pending_created_by_allocation_count,
            transient_allocation_slot_count: self.ledger.transient_allocation_slot_count,
            mutation_epoch: self.mutation_epoch_expected_even,
            fused_preparation_bytes: self.fused_footprint.pooled_device_scratch_bytes,
            fused_pooled_allocation_slots: self.fused_footprint.pooled_device_scratch_slots,
            fused_owner_array_bytes: self.fused_footprint.owner_array_backing_bytes,
            fused_staging_bytes: self.fused_footprint.staging_backing_bytes,
            fused_stamp_bytes: u64::try_from(self.source.row_count())
                .ok()
                .and_then(|rows| rows.checked_mul(std::mem::size_of::<Index>() as u64))
                .expect("indexed fused stamp geometry was checked before preview construction"),
            fused_materialization_scratch_bytes: self
                .fused_materialization_scratch
                .retained_bytes(),
            combined_preparation_bytes: self.resource_forecast.retained_device_transient_bytes,
            combined_allocation_slots: self.resource_forecast.incremental_allocation_slots,
            host_retained_bytes: host.retained_bytes(),
            host_allocation_slots: host
                .allocation_slots()
                .expect("indexed preview host allocation slots fit"),
            host_generation_pin_slots: host
                .generation_pin_slots()
                .expect("indexed preview host generation-pin slots fit"),
        }
    }

    #[cfg(test)]
    pub(super) fn source_payload(&self) -> &Arc<CudaResidentDeviceMemory> {
        &self.source_payload
    }

    #[cfg(test)]
    pub(super) fn point_route_generation(&self) -> &Arc<()> {
        &self.point_route_generation
    }

    #[cfg(test)]
    pub(super) fn physical_bindings(&self) -> &[PhysicalIndexBinding] {
        &self.physical_bindings
    }

    #[cfg(test)]
    pub(super) fn revalidate_for_test(
        &self,
        engine: &Engine,
        table: &RelationalTable,
    ) -> Result<(), ExecuteError> {
        self.revalidate(engine, table)
    }

    pub(super) fn revalidate(
        &self,
        engine: &Engine,
        table: &RelationalTable,
    ) -> Result<(), ExecuteError> {
        let current_catalog = engine.catalog_snapshot();
        let current_table = current_catalog
            .relational_catalog
            .get(&self.table_name)
            .ok_or_else(|| decline("indexed in-place preview lost its table"))?;
        if current_catalog.commit_seq != self.catalog_seq
            || current_table.oid != self.table_oid
            || current_table != table
            || crate::engine_transaction_reset::table_schema_digest(current_table)
                .ok()
                .as_ref()
                != Some(&self.schema_digest)
            || !super::fixed_insert::source_matches_indexed_in_place_reservation(
                &self.source,
                current_table,
            )
            || !self.row_ids.is_exact()
            || !self.row_ids.exact_len_matches(self.source.row_count())
            || self.source.requires_dense_rollover()
            || self.source.row_count() == 0
            || self.key_proof.indexes().len() != self.logical.raw.len()
            || self.mutation_epoch.load(Ordering::Acquire) != self.mutation_epoch_expected_even
            || self.mutation_epoch_expected_even & 1 != 0
            || self.target_witness.table_oid != self.table_oid
            || self.target_witness.gpu_id != self.gpu_id
            || self.target_witness.schema_digest != self.schema_digest
            || self.generation_witness.catalog_seq != self.catalog_seq
            || self.generation_witness.predecessor_boundary != self.predecessor_boundary
            || self.generation_witness.open_shard_id != self.shard_id
            || self.generation_witness.row_start
                != u64::try_from(self.row_start).unwrap_or(u64::MAX)
            || self.generation_witness.row_count != u64::try_from(self.base_row).unwrap_or(u64::MAX)
            || self.generation_witness.capacity != u64::try_from(self.capacity).unwrap_or(u64::MAX)
            || self.generation_witness.index_mutation_epoch_even
                != self.mutation_epoch_expected_even
        {
            return Err(decline("indexed in-place preview currentness drifted"));
        }
        let shards = engine.read_state.residency.shards.load_full();
        let open = shards
            .get(&self.table_name)
            .and_then(|shards| shards.last())
            .ok_or_else(|| decline("indexed in-place preview lost its open shard"))?;
        let source = open
            .device_memory
            .as_ref()
            .ok_or_else(|| decline("indexed in-place preview lost its source payload"))?;
        if open.shard_id != self.shard_id
            || open.row_start != self.row_start
            || open.row_count != self.base_row
            || open.capacity != self.capacity
            || open.gpu_id != self.gpu_id
            || open.schema != self.schema
            || !Arc::ptr_eq(source, &self.source_payload)
            || !Arc::ptr_eq(&open.point_route_generation, &self.point_route_generation)
            || !same_optional_arc(&open.created_by_region, &self.created_by_region)
            || !same_optional_arc(&open.row_id_region, &self.row_id_region)
            || !same_optional_arc(&open.deleted_by_region, &self.deleted_by_region)
            || source.device_ptr() == 0
            || source.device_ptr() != self.source_payload.device_ptr()
            || self.base_row.checked_add(self.source.row_count()) != Some(self.end_row)
            || self.end_row > self.capacity
            || !self.validation.matches_in_place_append(
                current_table,
                self.catalog_seq,
                open,
                self.predecessor_boundary,
            )
        {
            return Err(decline(
                "indexed in-place preview generation identity drifted",
            ));
        }
        validate_physical_bindings(
            engine,
            current_table,
            open,
            &self.source_payload,
            &self.key_proof,
            &self.logical,
            self.validation.original_read_snapshot(),
            self.end_row,
            &self.requests,
            &self.physical_bindings,
        )?;
        validate_ledger(self)?;
        let fused_footprint =
            super::fixed_insert::indexed_in_place_fused_footprint_forecast(&self.source, open)
                .map_err(|_| decline("indexed in-place preview fused footprint drifted"))?;
        if fused_footprint != self.fused_footprint
            || super::fixed_insert::indexed_in_place_fused_host_retention_forecast(
                &self.source,
                fused_footprint,
            )
            .map_err(|_| decline("indexed in-place preview fused host geometry drifted"))?
                != self.fused_final_host_geometry
            || super::fixed_insert::indexed_in_place_fused_materialization_scratch_forecast(
                &self.source,
                fused_footprint,
            )
            .map_err(|_| decline("indexed in-place preview fused scratch geometry drifted"))?
                != self.fused_materialization_scratch
        {
            return Err(decline("indexed in-place preview fused forecast drifted"));
        }
        Ok(())
    }

    pub(super) fn into_append_and_parts(
        self,
        engine: &'a Engine,
        table: &RelationalTable,
        permit: crate::engine_insert_plan::IndexedPhysicalMaterializationPermit,
    ) -> Result<PreviewPreparedParts<'a>, ExecuteError> {
        self.revalidate(engine, table)?;
        let preview_host_retention = self.host_retention_geometry()?;
        let IndexedInPlacePreview {
            source,
            row_ids,
            catalog_seq,
            shard_id,
            base_row,
            end_row,
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
            fused_materialization_peak,
            target_witness,
            generation_witness,
            resource_forecast,
            materialized_append_host_geometry,
            mutation_gate,
            named_index_lifecycle,
            ..
        } = self;
        let append = engine
            .prepare_resident_open_shard_append_indexed_in_place_reservation(
                source,
                row_ids,
                mutation_gate,
                logical.preparation_bytes(),
                permit,
            )
            .map_err(|_| decline("indexed in-place preview append reservation declined"))?;
        let (append_shard, append_base, append_catalog) =
            append.indexed_in_place_reservation_basis();
        if append_shard != shard_id
            || append_base != base_row
            || append_catalog != catalog_seq
            || append.row_count() == 0
            || append_base.checked_add(append.row_count()) != Some(end_row)
        {
            return Err(decline("indexed in-place preview append geometry drifted"));
        }
        Ok(PreviewPreparedParts {
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
            fused_materialization_peak,
            target_witness,
            generation_witness,
            preview_host_retention,
            resource_forecast,
            materialized_append_host_geometry,
            named_index_lifecycle,
        })
    }
}

/// Fields transferred only to the allocating index-delta owner after the preview has revalidated.
pub(super) struct PreviewPreparedParts<'a> {
    pub(super) append: ResidentOpenShardAppendPlan<'a>,
    pub(super) source_payload: Arc<CudaResidentDeviceMemory>,
    pub(super) created_by_region: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) row_id_region: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) deleted_by_region: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) key_proof: BatchKeyConstraintProof,
    pub(super) validation: ResidentKeyValidationSeal,
    pub(super) logical: PreparedIndexDeltaLogicalBindings,
    pub(super) requests: Box<[CudaResidentTypedIndexInsert]>,
    pub(super) physical_bindings: Box<[PhysicalIndexBinding]>,
    pub(super) mutation_epoch: Arc<AtomicU64>,
    pub(super) mutation_epoch_expected_even: u64,
    pub(super) ledger: IndexDeltaResourceLedger,
    pub(super) fused_footprint: gpu_db_execution::FusedApplyPreparationFootprint,
    pub(super) fused_final_host_geometry: HostRetentionGeometry,
    pub(super) fused_materialization_scratch: HostRetentionGeometry,
    pub(super) fused_materialization_peak: HostRetentionGeometry,
    pub(super) target_witness: IndexedPhysicalTargetWitness,
    pub(super) generation_witness: IndexedPhysicalGenerationWitness,
    pub(super) preview_host_retention: HostRetentionGeometry,
    pub(super) resource_forecast: IndexedPhysicalResourceForecast,
    pub(super) materialized_append_host_geometry: HostRetentionGeometry,
    pub(super) named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
}

impl PreviewPreparedParts<'_> {
    /// The materialization phase retains append source/row-ID ownership while requests still
    /// exist. The caller merges the prepared launch array before that request owner drops.
    pub(super) fn host_retention_report(&self) -> Result<HostRetentionReport, ExecuteError> {
        let mut report = self.append.host_retention_report()?;
        self.key_proof.append_host_retention(&mut report)?;
        self.validation.append_host_retention(&mut report)?;
        self.logical.append_host_retention(&mut report)?;
        report.retain_boxed_slice(&self.requests)?;
        for request in self.requests.iter() {
            report.retain_vec(&request.columns)?;
        }
        report.retain_boxed_slice(&self.physical_bindings)?;
        for binding in self.physical_bindings.iter() {
            binding.append_host_retention(&mut report)?;
        }
        report.retain_arc_owner(&self.mutation_epoch)?;
        Ok(report)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare<'a>(
    engine: &'a Engine,
    table: &RelationalTable,
    predecessor_boundary: Index,
    batch: TypedInsertBatch,
    row_ids: super::DeviceInsertRowIds,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    mutation_gate: MutexGuard<'a, ()>,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
    _commit_proof: &MutexGuard<'_, crate::CommitState>,
) -> Result<IndexedInPlacePreview<'a>, ExecuteError> {
    let source_retention_prediction = batch
        .resident_append_source_host_retention_prediction()
        .map_err(|_| decline("indexed in-place preview source retention prediction declined"))?;
    let source = batch
        .into_resident_append_source()
        .ok_or_else(|| decline("indexed in-place preview lost its resident append source"))?;
    if source_retention_prediction != source.host_retention_geometry()? {
        return Err(decline(
            "indexed in-place preview source retention materialization drifted",
        ));
    }
    // The current physical reservation owns one prepared fused i32 payload/sidecar pass.  Mixed
    // fixed widths, NULL-bearing inputs, and text must decline here, before any physical index
    // or append-side resource crosses the pre-WAL boundary.
    if !super::fixed_insert::source_is_all_i32_fixed(&source) {
        return Err(decline(
            "indexed in-place fused reservation requires an all-i32 null-free source",
        ));
    }
    let current_catalog = engine.catalog_snapshot();
    let current_table = current_catalog
        .relational_catalog
        .get(&table.name)
        .ok_or_else(|| decline("indexed in-place preview lost current table catalog"))?;
    if current_catalog.commit_seq < source.prepared_catalog_seq()
        || current_table != table
        || !super::fixed_insert::source_matches_indexed_in_place_reservation(&source, table)
        || source.requires_dense_rollover()
        || source.row_count() == 0
        || !row_ids.is_exact()
        || !row_ids.exact_len_matches(source.row_count())
    {
        return Err(decline(
            "indexed in-place preview source is not fixed in-place eligible",
        ));
    }
    let logical = prepare_logical_bindings(engine, table, &key_proof)?;
    let shards = engine.read_state.residency.shards.load_full();
    let open = shards
        .get(&table.name)
        .and_then(|shards| shards.last())
        .ok_or_else(|| decline("indexed in-place preview has no current open shard"))?;
    let base_row = open.row_count;
    let end_row = base_row
        .checked_add(source.row_count())
        .filter(|end| *end <= open.capacity)
        .ok_or_else(|| decline("indexed in-place preview requires an in-place extent"))?;
    let source_payload = open
        .device_memory
        .as_ref()
        .cloned()
        .filter(|payload| payload.device_ptr() != 0)
        .ok_or_else(|| decline("indexed in-place preview lost resident source allocation"))?;
    if !validation.matches_in_place_append(
        table,
        current_catalog.commit_seq,
        open,
        predecessor_boundary,
    ) || validation.original_read_snapshot() > predecessor_boundary
    {
        return Err(decline(
            "indexed in-place preview validation generation drifted",
        ));
    }
    let (requests, physical_bindings) = bind_physical_indexes(
        engine,
        table,
        open,
        &source_payload,
        base_row,
        end_row,
        &key_proof,
        &logical,
        validation.original_read_snapshot(),
    )?;
    let expected_columns = logical
        .physical
        .iter()
        .try_fold(0_usize, |total, binding| {
            total.checked_add(binding.descriptor_count)
        })
        .ok_or_else(|| decline("indexed in-place preview descriptor total overflows"))?;
    if requests.len() != logical.physical.len()
        || requests
            .iter()
            .map(|request| request.columns.len())
            .sum::<usize>()
            != expected_columns
    {
        return Err(decline(
            "indexed in-place preview descriptor authority drifted",
        ));
    }
    let mutation_epoch = engine
        .read_state
        .residency
        .point_index_mutation_epoch_for_table(&engine.read_state, table)
        .ok_or_else(|| decline("indexed in-place preview relation identity changed"))?;
    let mutation_epoch_expected_even = mutation_epoch.load(Ordering::Acquire);
    if mutation_epoch_expected_even & 1 != 0 {
        return Err(decline(
            "indexed in-place preview observed an active mutation epoch",
        ));
    }
    let ledger = resource_ledger(&logical, &physical_bindings, &source_payload, open)?;
    let schema_digest = crate::engine_transaction_reset::table_schema_digest(table)
        .map_err(|_| decline("indexed in-place preview schema digest declined"))?;
    let materialized_append_host_geometry =
        super::fixed_insert::indexed_in_place_append_host_retention_forecast(
            &source, &row_ids, open, table,
        )
        .map_err(|_| decline("indexed in-place preview append host forecast declined"))?;
    let fused_footprint =
        super::fixed_insert::indexed_in_place_fused_footprint_forecast(&source, open)
            .map_err(|_| decline("indexed in-place preview fused footprint forecast declined"))?;
    let fused_final_host_geometry =
        super::fixed_insert::indexed_in_place_fused_host_retention_forecast(
            &source,
            fused_footprint,
        )
        .map_err(|_| decline("indexed in-place preview fused final host forecast declined"))?;
    let fused_materialization_scratch =
        super::fixed_insert::indexed_in_place_fused_materialization_scratch_forecast(
            &source,
            fused_footprint,
        )
        .map_err(|_| decline("indexed in-place preview fused scratch forecast declined"))?;
    let target_witness = IndexedPhysicalTargetWitness {
        gpu_id: open.gpu_id,
        table_oid: table.oid,
        schema_digest,
    };
    let generation_witness = IndexedPhysicalGenerationWitness {
        catalog_seq: current_catalog.commit_seq,
        predecessor_boundary,
        open_shard_id: open.shard_id,
        row_start: u64::try_from(open.row_start)
            .map_err(|_| decline("indexed in-place preview row start forecast overflows"))?,
        row_count: u64::try_from(open.row_count)
            .map_err(|_| decline("indexed in-place preview row count forecast overflows"))?,
        capacity: u64::try_from(open.capacity)
            .map_err(|_| decline("indexed in-place preview capacity forecast overflows"))?,
        index_mutation_epoch_even: mutation_epoch_expected_even,
    };
    let mut preview = IndexedInPlacePreview {
        source,
        row_ids,
        table_name: table.name.clone(),
        table_oid: table.oid,
        schema_digest,
        catalog_seq: current_catalog.commit_seq,
        predecessor_boundary,
        shard_id: open.shard_id,
        row_start: open.row_start,
        base_row,
        end_row,
        capacity: open.capacity,
        gpu_id: open.gpu_id,
        schema: open.schema.clone(),
        source_payload,
        point_route_generation: Arc::clone(&open.point_route_generation),
        created_by_region: open.created_by_region.clone(),
        row_id_region: open.row_id_region.clone(),
        deleted_by_region: open.deleted_by_region.clone(),
        key_proof,
        validation,
        logical,
        requests: requests.into(),
        physical_bindings: physical_bindings.into(),
        mutation_epoch,
        mutation_epoch_expected_even,
        ledger,
        fused_footprint,
        fused_final_host_geometry,
        fused_materialization_scratch,
        fused_materialization_peak: HostRetentionGeometry::default(),
        target_witness,
        generation_witness,
        resource_forecast: IndexedPhysicalResourceForecast::default(),
        materialized_append_host_geometry,
        mutation_gate,
        named_index_lifecycle,
    };
    let preview_host = preview.host_retention_geometry()?;
    let mut final_host = preview.final_host_retention_geometry()?;
    final_host.checked_add_disjoint(
        preview.fused_final_host_geometry,
        "indexed in-place fused final host owner",
    )?;
    let materializing_base = preview.materializing_base_host_retention_geometry()?;
    let mut fused_materialization_peak = materializing_base;
    fused_materialization_peak.checked_add_disjoint(
        preview.fused_final_host_geometry,
        "indexed in-place fused final owner during materialization",
    )?;
    fused_materialization_peak.checked_add_disjoint(
        preview.fused_materialization_scratch,
        "indexed in-place fused temporary materialization backing",
    )?;
    preview.fused_materialization_peak = fused_materialization_peak;
    let mut materializing_launch = preview.materializing_launch_host_retention_geometry()?;
    materializing_launch.checked_add_disjoint(
        preview.fused_final_host_geometry,
        "indexed in-place fused final owner alongside index preparation",
    )?;
    let incremental_allocation_slots = u64::try_from(
        preview
            .ledger
            .pending_created_by_allocation_count
            .checked_add(preview.ledger.transient_allocation_slot_count)
            .and_then(|slots| {
                usize::try_from(preview.fused_footprint.pooled_device_scratch_slots)
                    .ok()
                    .and_then(|fused_slots| slots.checked_add(fused_slots))
            })
            .ok_or_else(|| decline("indexed in-place forecast allocation-slot overflow"))?,
    )
    .map_err(|_| decline("indexed in-place forecast allocation-slot conversion overflow"))?;
    preview.resource_forecast = IndexedPhysicalResourceForecast {
        final_host_retained_bytes: final_host.retained_bytes(),
        final_host_allocation_slots: final_host.allocation_slots(),
        final_host_generation_pin_slots: final_host.generation_pin_slots(),
        peak_host_retained_bytes: preview_host
            .peak(preview.fused_materialization_peak)
            .peak(materializing_launch)
            .peak(final_host)
            .retained_bytes(),
        peak_host_allocation_slots: preview_host
            .peak(preview.fused_materialization_peak)
            .peak(materializing_launch)
            .peak(final_host)
            .allocation_slots(),
        peak_host_generation_pin_slots: preview_host
            .peak(preview.fused_materialization_peak)
            .peak(materializing_launch)
            .peak(final_host)
            .generation_pin_slots(),
        old_generation_pinned_bytes: preview.ledger.retained_persistent_bytes,
        new_persistent_bytes: preview.ledger.pending_created_by_bytes,
        retained_device_transient_bytes: preview
            .ledger
            .transient_preparation_bytes
            .checked_add(preview.fused_footprint.pooled_device_scratch_bytes)
            .ok_or_else(|| decline("indexed in-place combined preparation bytes overflow"))?,
        retained_device_result_bytes: 0,
        incremental_allocation_slots,
        generation_pin_slots: u64::try_from(preview.ledger.retained_allocation_pin_count)
            .map_err(|_| decline("indexed in-place forecast generation-pin overflow"))?,
        maximum_concurrent_device_scratch_bytes: preview
            .ledger
            .transient_preparation_bytes
            .checked_add(preview.fused_footprint.pooled_device_scratch_bytes)
            .ok_or_else(|| decline("indexed in-place combined scratch bytes overflow"))?,
        maximum_host_readback_bytes: preview
            .ledger
            .bounded_readback_bytes
            .max(preview.fused_footprint.status_readback_bytes),
    };
    Ok(preview)
}

/// Test-only inspection stops before the append compiler, allocation scope, and typed-index
/// preparation. Only a scalar report crosses this boundary; resource Arcs remain private.
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
    mutation_gate: MutexGuard<'a, ()>,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
    commit_proof: &MutexGuard<'_, crate::CommitState>,
    inspect: impl FnOnce(IndexedInPlacePreviewReport) -> R,
) -> Result<R, ExecuteError> {
    let preview = prepare(
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
    preview.revalidate(engine, table)?;
    Ok(inspect(preview.scalar_report()))
}

pub(super) fn prepare_logical_bindings(
    engine: &Engine,
    table: &RelationalTable,
    keys: &BatchKeyConstraintProof,
) -> Result<PreparedIndexDeltaLogicalBindings, ExecuteError> {
    if keys.indexes().len() != table.indexes.len() || table.indexes.is_empty() {
        return Err(decline(
            "indexed in-place preview lost exact raw catalog enrollment",
        ));
    }
    let mut raw = Vec::with_capacity(table.indexes.len());
    let mut physical = Vec::with_capacity(table.indexes.len());
    let mut physical_by_key = BTreeMap::new();
    let mut descriptor_count = 0_usize;
    let shard_map = engine.read_state.residency.shards.load_full();
    let open = shard_map
        .get(&table.name)
        .and_then(|shards| shards.last())
        .ok_or_else(|| decline("indexed in-place preview has no current open shard"))?;
    for (raw_ordinal, index) in table.indexes.iter().enumerate() {
        let binding = keys
            .indexes()
            .get(raw_ordinal)
            .filter(|binding| binding.raw_ordinal() == raw_ordinal)
            .ok_or_else(|| decline("indexed in-place preview lost a raw catalog binding"))?;
        if !crate::engine_residency::index_all_key_columns_foldable(table, index) {
            return Err(decline(
                "indexed in-place preview found a non-foldable named index",
            ));
        }
        let key_id = crate::engine_residency::index_probe_key_id(table, index, raw_ordinal)
            .ok_or_else(|| decline("indexed in-place preview found no resident index key id"))?;
        let physical_ordinal = if let Some(&ordinal) = physical_by_key.get(&key_id) {
            ordinal
        } else {
            let ordinal = physical.len();
            let columns = binding.resident_constraint_columns(table)?;
            let count =
                resident_constraint_generation::resident_columns(engine, table, open, &columns)?
                    .len();
            descriptor_count = descriptor_count
                .checked_add(count)
                .ok_or_else(|| decline("indexed in-place preview descriptor count overflows"))?;
            physical.push(PhysicalIndexLogicalBinding {
                raw_ordinal,
                key_id,
                descriptor_count: count,
            });
            physical_by_key.insert(key_id, ordinal);
            ordinal
        };
        raw.push(RawIndexLogicalBinding {
            raw_ordinal,
            key_id,
            physical_ordinal,
        });
    }
    let preparation_bytes =
        resident_typed_indexes_insert_preparation_bytes(physical.len(), descriptor_count)
            .ok_or_else(|| decline("indexed in-place preview preparation geometry overflows"))?;
    Ok(PreparedIndexDeltaLogicalBindings {
        raw: raw.into(),
        physical: physical.into(),
        preparation_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn bind_physical_indexes(
    engine: &Engine,
    table: &RelationalTable,
    open: &RelationalResidentShard,
    source_payload: &Arc<CudaResidentDeviceMemory>,
    base_row: usize,
    end_row: usize,
    keys: &BatchKeyConstraintProof,
    logical: &PreparedIndexDeltaLogicalBindings,
    original_read_snapshot: Index,
) -> Result<(Vec<CudaResidentTypedIndexInsert>, Vec<PhysicalIndexBinding>), ExecuteError> {
    let source_ptr = source_payload.device_ptr();
    let horizon = expected_index_horizon(open)?;
    let route_publish = engine
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
        return Err(decline(
            "indexed in-place preview requires complete named-index enrollment",
        ));
    }
    let mut requests = Vec::with_capacity(logical.physical.len());
    let mut physical = Vec::with_capacity(logical.physical.len());
    let mut destinations = BTreeSet::new();
    for logical_binding in logical.physical.iter() {
        let binding = keys
            .indexes()
            .get(logical_binding.raw_ordinal)
            .filter(|binding| binding.raw_ordinal() == logical_binding.raw_ordinal)
            .ok_or_else(|| {
                decline("indexed in-place preview raw binding changed before cache bind")
            })?;
        let index = table
            .indexes
            .get(logical_binding.raw_ordinal)
            .ok_or_else(|| decline("indexed in-place preview catalog index disappeared"))?;
        if crate::engine_residency::index_probe_key_id(table, index, logical_binding.raw_ordinal)
            != Some(logical_binding.key_id)
        {
            return Err(decline("indexed in-place preview logical key id drifted"));
        }
        let cache_key = (table.name.clone(), open.shard_id, logical_binding.key_id);
        let covered = coverage.get(&cache_key) == Some(&(source_ptr, base_row));
        let entry = cache
            .get(&cache_key)
            .ok_or_else(|| decline("indexed in-place preview has no enrolled physical index"))?;
        let device_index = entry
            .device_index
            .as_ref()
            .filter(|memory| {
                entry.resident_device_ptr == source_ptr
                    && entry.row_count == base_row
                    && Arc::ptr_eq(&entry._resident_guard, source_payload)
                    && entry.gc_boundary <= original_read_snapshot
                    && entry.published_row_count.load(Ordering::Acquire) == base_row
                    && entry.published_has_postings.load(Ordering::Acquire) == entry.has_postings
                    && entry.table_mask == horizon.table_mask
                    && entry.hash_shift == horizon.hash_shift
                    && memory.metadata().allocated_bytes == horizon.allocated_bytes
            })
            .cloned()
            .ok_or_else(|| decline("indexed in-place preview physical index basis drifted"))?;
        if !covered
            || device_index.device_ptr() == source_ptr
            || !destinations.insert(device_index.device_ptr())
        {
            return Err(decline(
                "indexed in-place preview lost complete distinct cache coverage",
            ));
        }
        let key_columns = binding.resident_constraint_columns(table)?;
        let columns =
            resident_constraint_generation::resident_columns(engine, table, open, &key_columns)?;
        if columns.len() != logical_binding.descriptor_count {
            return Err(decline(
                "indexed in-place preview input descriptor count drifted",
            ));
        }
        requests.push(CudaResidentTypedIndexInsert {
            index: Arc::clone(&device_index),
            table_mask: entry.table_mask,
            hash_shift: entry.hash_shift,
            columns,
        });
        physical.push(PhysicalIndexBinding {
            cache_key,
            key_id: logical_binding.key_id,
            device_index,
            resident_guard: Arc::clone(&entry._resident_guard),
            published_row_count: Arc::clone(&entry.published_row_count),
            published_has_postings: Arc::clone(&entry.published_has_postings),
            has_postings_at_preview: entry.has_postings,
            source_ptr,
            base_row,
            end_row,
            capacity: open.capacity,
            table_mask: entry.table_mask,
            hash_shift: entry.hash_shift,
            gc_boundary: entry.gc_boundary,
            allocated_bytes: horizon.allocated_bytes,
        });
    }
    drop(publications);
    drop(complete);
    drop(coverage);
    drop(cache);
    drop(route_publish);
    Ok((requests, physical))
}

#[allow(clippy::too_many_arguments)]
fn validate_physical_bindings(
    engine: &Engine,
    table: &RelationalTable,
    open: &RelationalResidentShard,
    source_payload: &Arc<CudaResidentDeviceMemory>,
    keys: &BatchKeyConstraintProof,
    logical: &PreparedIndexDeltaLogicalBindings,
    original_read_snapshot: Index,
    end_row: usize,
    requests: &[CudaResidentTypedIndexInsert],
    physical: &[PhysicalIndexBinding],
) -> Result<(), ExecuteError> {
    if physical.len() != logical.physical.len() || requests.len() != physical.len() {
        return Err(decline("indexed in-place preview physical count drifted"));
    }
    let source_ptr = source_payload.device_ptr();
    let horizon = expected_index_horizon(open)?;
    let route_publish = engine
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
        return Err(decline(
            "indexed in-place preview enrollment currentness drifted",
        ));
    }
    let mut destinations = BTreeSet::new();
    for (ordinal, ((logical_binding, physical_binding), request)) in logical
        .physical
        .iter()
        .zip(physical.iter())
        .zip(requests.iter())
        .enumerate()
    {
        let raw = keys
            .indexes()
            .get(logical_binding.raw_ordinal)
            .filter(|binding| binding.raw_ordinal() == logical_binding.raw_ordinal)
            .ok_or_else(|| decline("indexed in-place preview raw mapping currentness drifted"))?;
        let index = table
            .indexes
            .get(logical_binding.raw_ordinal)
            .ok_or_else(|| decline("indexed in-place preview catalog currentness drifted"))?;
        let key =
            crate::engine_residency::index_probe_key_id(table, index, logical_binding.raw_ordinal);
        let key_columns = raw.resident_constraint_columns(table)?;
        let current_columns =
            resident_constraint_generation::resident_columns(engine, table, open, &key_columns)?;
        let entry = cache
            .get(&physical_binding.cache_key)
            .ok_or_else(|| decline("indexed in-place preview cache entry retired"))?;
        if ordinal >= physical.len()
            || raw.raw_ordinal() != logical_binding.raw_ordinal
            || key != Some(logical_binding.key_id)
            || physical_binding.key_id != logical_binding.key_id
            || physical_binding.cache_key
                != (table.name.clone(), open.shard_id, logical_binding.key_id)
            || coverage.get(&physical_binding.cache_key) != Some(&(source_ptr, open.row_count))
            || !Arc::ptr_eq(&entry._resident_guard, source_payload)
            || !Arc::ptr_eq(&entry._resident_guard, &physical_binding.resident_guard)
            || !entry
                .device_index
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &physical_binding.device_index))
            || !Arc::ptr_eq(
                &entry.published_row_count,
                &physical_binding.published_row_count,
            )
            || !Arc::ptr_eq(
                &entry.published_has_postings,
                &physical_binding.published_has_postings,
            )
            || entry.resident_device_ptr != source_ptr
            || entry.row_count != open.row_count
            || entry.gc_boundary != physical_binding.gc_boundary
            || entry.gc_boundary > original_read_snapshot
            || entry.table_mask != physical_binding.table_mask
            || entry.hash_shift != physical_binding.hash_shift
            || entry.table_mask != horizon.table_mask
            || entry.hash_shift != horizon.hash_shift
            || entry.has_postings != physical_binding.has_postings_at_preview
            || entry.published_row_count.load(Ordering::Acquire) != physical_binding.base_row
            || entry.published_has_postings.load(Ordering::Acquire)
                != physical_binding.has_postings_at_preview
            || physical_binding.source_ptr != source_ptr
            || physical_binding.base_row != open.row_count
            || physical_binding.end_row != end_row
            || physical_binding.capacity != open.capacity
            || !Arc::ptr_eq(&request.index, &physical_binding.device_index)
            || request.table_mask != physical_binding.table_mask
            || request.hash_shift != physical_binding.hash_shift
            || request.columns != current_columns
            || request.columns.len() != logical_binding.descriptor_count
            || physical_binding.device_index.device_ptr() == source_ptr
            || physical_binding.device_index.metadata().allocated_bytes
                != physical_binding.allocated_bytes
            || physical_binding.allocated_bytes != horizon.allocated_bytes
            || !destinations.insert(physical_binding.device_index.device_ptr())
        {
            return Err(decline(
                "indexed in-place preview cache pin currentness drifted",
            ));
        }
    }
    drop(publications);
    drop(complete);
    drop(coverage);
    drop(cache);
    drop(route_publish);
    Ok(())
}

struct IndexHorizon {
    table_mask: u32,
    hash_shift: u32,
    allocated_bytes: u64,
}

fn expected_index_horizon(open: &RelationalResidentShard) -> Result<IndexHorizon, ExecuteError> {
    let rows = u64::try_from(open.row_count)
        .map_err(|_| decline("indexed in-place preview row count overflows"))?;
    let capacity = u64::try_from(open.capacity)
        .map_err(|_| decline("indexed in-place preview capacity overflows"))?;
    let table_size = crate::engine_residency::resident_shard_index_table_size(rows, capacity)
        .ok_or_else(|| decline("indexed in-place preview has no index horizon geometry"))?;
    let table_mask = (table_size - 1) as u32;
    let allocated_bytes = resident_index_allocated_bytes(table_mask, capacity.max(rows))
        .ok_or_else(|| decline("indexed in-place preview index allocation geometry overflows"))?;
    Ok(IndexHorizon {
        table_mask,
        hash_shift: 32 - table_size.trailing_zeros(),
        allocated_bytes,
    })
}

fn resource_ledger(
    logical: &PreparedIndexDeltaLogicalBindings,
    physical: &[PhysicalIndexBinding],
    source_payload: &Arc<CudaResidentDeviceMemory>,
    open: &RelationalResidentShard,
) -> Result<IndexDeltaResourceLedger, ExecuteError> {
    let descriptor_count = logical
        .physical
        .iter()
        .try_fold(0_usize, |total, binding| {
            total.checked_add(binding.descriptor_count)
        })
        .ok_or_else(|| decline("indexed in-place preview ledger descriptor count overflows"))?;
    let descriptor_words = logical
        .physical
        .len()
        .checked_mul(3)
        .and_then(|words| words.checked_add(descriptor_count.checked_mul(4)?))
        .ok_or_else(|| decline("indexed in-place preview ledger descriptor words overflow"))?;
    let descriptor_bytes = u64::try_from(descriptor_words)
        .ok()
        .and_then(|words| words.checked_mul(std::mem::size_of::<u64>() as u64))
        .ok_or_else(|| decline("indexed in-place preview ledger descriptor bytes overflow"))?;
    let pinned_persistent_index_bytes = physical.iter().try_fold(0_u64, |total, binding| {
        total
            .checked_add(binding.allocated_bytes)
            .ok_or_else(|| decline("indexed in-place preview ledger persistent bytes overflow"))
    })?;
    let pending_created_by_bytes = if open.created_by_region.is_none() {
        u64::try_from(open.capacity)
            .ok()
            .and_then(|capacity| capacity.checked_mul(std::mem::size_of::<u64>() as u64))
            .ok_or_else(|| decline("indexed in-place preview created_by extent overflows"))?
    } else {
        0
    };
    let pinned_created_by_bytes = open
        .created_by_region
        .as_ref()
        .map(|memory| memory.metadata().allocated_bytes)
        .unwrap_or(0);
    let pinned_row_id_bytes = open
        .row_id_region
        .as_ref()
        .map(|memory| memory.metadata().allocated_bytes)
        .unwrap_or(0);
    let (retained_persistent_bytes, retained_allocation_pin_count) = retained_allocation_ledger(
        source_payload,
        physical,
        open.created_by_region.as_ref(),
        open.row_id_region.as_ref(),
        open.deleted_by_region.as_ref(),
    )?;
    Ok(IndexDeltaResourceLedger {
        source_payload_bytes: source_payload.metadata().allocated_bytes,
        pinned_persistent_index_bytes,
        pinned_created_by_bytes,
        pinned_row_id_bytes,
        pinned_deleted_by_bytes: open
            .deleted_by_region
            .as_ref()
            .map(|memory| memory.metadata().allocated_bytes)
            .unwrap_or(0),
        retained_persistent_bytes,
        pending_created_by_bytes,
        transient_preparation_bytes: logical.preparation_bytes,
        transient_allocation_slot_count: 2,
        descriptor_bytes,
        descriptor_count,
        bounded_readback_bytes: std::mem::size_of::<u32>() as u64,
        retained_allocation_pin_count,
        pending_created_by_allocation_count: usize::from(pending_created_by_bytes != 0),
        raw_index_count: logical.raw.len(),
        physical_index_count: logical.physical.len(),
    })
}

fn validate_ledger(preview: &IndexedInPlacePreview<'_>) -> Result<(), ExecuteError> {
    let ledger = &preview.ledger;
    let pinned = preview
        .physical_bindings
        .iter()
        .try_fold(0_u64, |total, binding| {
            total
                .checked_add(binding.allocated_bytes)
                .ok_or_else(|| decline("indexed in-place preview retained byte sum overflows"))
        })?;
    let (retained_persistent_bytes, retained_allocation_pin_count) = retained_allocation_ledger(
        &preview.source_payload,
        &preview.physical_bindings,
        preview.created_by_region.as_ref(),
        preview.row_id_region.as_ref(),
        preview.deleted_by_region.as_ref(),
    )?;
    if ledger.source_payload_bytes != preview.source_payload.metadata().allocated_bytes
        || ledger.pinned_persistent_index_bytes != pinned
        || ledger.pinned_created_by_bytes
            != preview
                .created_by_region
                .as_ref()
                .map(|memory| memory.metadata().allocated_bytes)
                .unwrap_or(0)
        || ledger.pinned_row_id_bytes
            != preview
                .row_id_region
                .as_ref()
                .map(|memory| memory.metadata().allocated_bytes)
                .unwrap_or(0)
        || ledger.pinned_deleted_by_bytes
            != preview
                .deleted_by_region
                .as_ref()
                .map(|memory| memory.metadata().allocated_bytes)
                .unwrap_or(0)
        || ledger.retained_persistent_bytes != retained_persistent_bytes
        || ledger.transient_preparation_bytes != preview.logical.preparation_bytes
        || ledger.transient_allocation_slot_count != 2
        || ledger.raw_index_count != preview.logical.raw.len()
        || ledger.physical_index_count != preview.logical.physical.len()
        || ledger.physical_index_count != preview.physical_bindings.len()
        || ledger.retained_allocation_pin_count != retained_allocation_pin_count
        || ledger.pending_created_by_allocation_count
            != usize::from(preview.created_by_region.is_none())
        || ledger.bounded_readback_bytes != std::mem::size_of::<u32>() as u64
        || preview
            .logical
            .raw
            .iter()
            .enumerate()
            .any(|(ordinal, raw)| {
                raw.raw_ordinal != ordinal
                    || preview
                        .logical
                        .physical
                        .get(raw.physical_ordinal)
                        .is_none_or(|physical| physical.key_id != raw.key_id)
            })
    {
        return Err(decline("indexed in-place preview scalar ledger drifted"));
    }
    Ok(())
}

fn retained_allocation_ledger(
    source_payload: &Arc<CudaResidentDeviceMemory>,
    physical: &[PhysicalIndexBinding],
    created_by: Option<&Arc<CudaResidentDeviceMemory>>,
    row_id: Option<&Arc<CudaResidentDeviceMemory>>,
    deleted_by: Option<&Arc<CudaResidentDeviceMemory>>,
) -> Result<(u64, usize), ExecuteError> {
    let mut allocations = BTreeMap::<usize, u64>::new();
    let mut retain = |memory: &Arc<CudaResidentDeviceMemory>| -> Result<(), ExecuteError> {
        let identity = memory.allocation_identity();
        let bytes = memory.metadata().allocated_bytes;
        if let Some(existing) = allocations.insert(identity, bytes) {
            if existing != bytes {
                return Err(decline(
                    "indexed in-place preview allocation identity byte drifted",
                ));
            }
        }
        Ok(())
    };
    retain(source_payload)?;
    for binding in physical {
        retain(&binding.device_index)?;
    }
    for memory in [created_by, row_id, deleted_by].into_iter().flatten() {
        retain(memory)?;
    }
    let bytes = allocations.values().try_fold(0_u64, |total, bytes| {
        total
            .checked_add(*bytes)
            .ok_or_else(|| decline("indexed in-place preview retained byte total overflows"))
    })?;
    Ok((bytes, allocations.len()))
}

#[cfg(all(test, feature = "probe-timing"))]
pub(super) fn distinct_wrapper_alias_deduplicates_for_test(
    source: &Arc<CudaResidentDeviceMemory>,
) -> Result<bool, ExecuteError> {
    let alias = Arc::new(source.distinct_wrapper_for_accounting_test());
    let (bytes, slots) = retained_allocation_ledger(source, &[], Some(&alias), None, None)?;
    Ok(!Arc::ptr_eq(source, &alias)
        && source.allocation_identity() == alias.allocation_identity()
        && bytes == source.metadata().allocated_bytes
        && slots == 1)
}

fn same_optional_arc(
    left: &Option<Arc<CudaResidentDeviceMemory>>,
    right: &Option<Arc<CudaResidentDeviceMemory>>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        (None, None) => true,
        _ => false,
    }
}

fn append_string_geometry(
    geometry: &mut HostRetentionGeometry,
    value: &String,
    domain: &'static str,
) -> Result<(), ExecuteError> {
    if value.capacity() == 0 {
        return Ok(());
    }
    let bytes = u64::try_from(value.capacity())
        .map_err(|_| decline("indexed preview string capacity overflows"))?;
    geometry.checked_add_backing_bytes_slots(bytes, 1, domain)?;
    Ok(())
}

fn decline(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Serialization(message.into())
}
