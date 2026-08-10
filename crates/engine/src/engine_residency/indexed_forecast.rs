//! Forecast-only ownership at the indexed physical capacity boundary.
//!
//! The owner below intentionally has no CUDA allocation, launch, WAL, apply, cache, or
//! publication surface.  Its sole transition consumes the unforgeable permit issued by
//! `engine_insert_plan`; the contained branch preview then creates its existing private work.

#![allow(dead_code)] // the live indexed INSERT handoff remains deliberately closed

use crate::engine_insert_plan::IndexedPhysicalMaterializationPermit;
use crate::relational_model::RelationalTable;
use crate::{Engine, ExecuteError, Index};

use super::index_delta_preview::IndexedInPlacePreview;
use super::index_rollover::IndexedFixedRolloverPreview;
use super::indexed_reservation::PreparedIndexedPhysicalReservation;

/// Stable physical target identity. Device pointers are deliberately absent: the forecast is
/// made before a rollover payload/private indexes exist and must be comparable after the permit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct IndexedPhysicalTargetWitness {
    pub(super) gpu_id: u16,
    pub(super) table_oid: u32,
    pub(super) schema_digest: gpu_db_wal::CanonicalDigest,
}

/// Exact old-generation identity that a materialized branch must still observe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct IndexedPhysicalGenerationWitness {
    pub(super) catalog_seq: Index,
    pub(super) predecessor_boundary: Index,
    pub(super) open_shard_id: u32,
    pub(super) row_start: u64,
    pub(super) row_count: u64,
    pub(super) capacity: u64,
    pub(super) index_mutation_epoch_even: u64,
}

/// Allocation-free scalar accounting for one sealed indexed physical branch.
///
/// `old_generation_pinned_bytes` is an existing authoritative-generation envelope, never a new
/// allocation.  `new_persistent_bytes` and the retained transient/result domains name only work
/// made private after the materialization permit.  Slots are separate from bytes because a
/// capacity authority must reject one-slot drift even where a domain happens to be zero bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct IndexedPhysicalResourceForecast {
    pub(super) final_host_retained_bytes: u64,
    pub(super) final_host_allocation_slots: u64,
    pub(super) final_host_generation_pin_slots: u64,
    pub(super) peak_host_retained_bytes: u64,
    pub(super) peak_host_allocation_slots: u64,
    pub(super) peak_host_generation_pin_slots: u64,
    pub(super) old_generation_pinned_bytes: u64,
    pub(super) new_persistent_bytes: u64,
    pub(super) retained_device_transient_bytes: u64,
    pub(super) retained_device_result_bytes: u64,
    pub(super) incremental_allocation_slots: u64,
    pub(super) generation_pin_slots: u64,
    pub(super) maximum_concurrent_device_scratch_bytes: u64,
    pub(super) maximum_host_readback_bytes: u64,
}

/// The one opaque, move-only forecast for indexed physical preparation.
///
/// Both variants retain only already-admitted semantic/proof/lock/generation owners until a
/// permit is consumed.  In particular, the fixed branch has not constructed a payload, sidecar,
/// private index directory, lookup oracle, or CUDA launch when it reaches this enum.
#[must_use]
#[allow(clippy::large_enum_variant)] // boxing would add a statement-retired host allocation
enum PreparedIndexedPhysicalForecastBranch<'a> {
    InPlace(IndexedInPlacePreview<'a>),
    FixedRollover(IndexedFixedRolloverPreview<'a>),
}

/// Opaque crate-visible handle; the branch variants and their private owners stay residency-only.
#[must_use]
pub(crate) struct PreparedIndexedPhysicalForecast<'a> {
    branch: PreparedIndexedPhysicalForecastBranch<'a>,
}

impl<'a> PreparedIndexedPhysicalForecast<'a> {
    pub(super) fn in_place(preview: IndexedInPlacePreview<'a>) -> Self {
        Self {
            branch: PreparedIndexedPhysicalForecastBranch::InPlace(preview),
        }
    }

    pub(super) fn fixed_rollover(preview: IndexedFixedRolloverPreview<'a>) -> Self {
        Self {
            branch: PreparedIndexedPhysicalForecastBranch::FixedRollover(preview),
        }
    }

    pub(super) fn resource_forecast(&self) -> IndexedPhysicalResourceForecast {
        match &self.branch {
            PreparedIndexedPhysicalForecastBranch::InPlace(preview) => preview.resource_forecast(),
            PreparedIndexedPhysicalForecastBranch::FixedRollover(preview) => {
                preview.resource_forecast()
            }
        }
    }

    pub(super) fn target_witness(&self) -> &IndexedPhysicalTargetWitness {
        match &self.branch {
            PreparedIndexedPhysicalForecastBranch::InPlace(preview) => preview.target_witness(),
            PreparedIndexedPhysicalForecastBranch::FixedRollover(preview) => {
                preview.target_witness()
            }
        }
    }

    pub(super) fn generation_witness(&self) -> &IndexedPhysicalGenerationWitness {
        match &self.branch {
            PreparedIndexedPhysicalForecastBranch::InPlace(preview) => preview.generation_witness(),
            PreparedIndexedPhysicalForecastBranch::FixedRollover(preview) => {
                preview.generation_witness()
            }
        }
    }

    /// Project the opaque residency preview into the exact scalar accounting/identity value
    /// retained by the pre-WAL lease. No pointer, allocation, lock, or materialization authority
    /// crosses this boundary.
    pub(crate) fn pre_wal_binding(
        &self,
    ) -> crate::engine_insert_plan::pre_wal_footprint::IndexedPreWalPhysicalForecast {
        let resource = self.resource_forecast();
        let target = self.target_witness();
        let generation = self.generation_witness();
        crate::engine_insert_plan::pre_wal_footprint::IndexedPreWalPhysicalForecast {
            target: crate::engine_insert_plan::pre_wal_footprint::GpuTargetKey {
                gpu_id: target.gpu_id,
                table_oid: target.table_oid,
                schema_digest: target.schema_digest,
            },
            generation: crate::engine_insert_plan::pre_wal_footprint::GpuGenerationWitness {
                catalog_seq: generation.catalog_seq,
                predecessor_boundary: generation.predecessor_boundary,
                open_shard_id: generation.open_shard_id,
                row_start: generation.row_start,
                row_count: generation.row_count,
                capacity: generation.capacity,
                index_mutation_epoch_even: generation.index_mutation_epoch_even,
            },
            final_host_retained_bytes: resource.final_host_retained_bytes,
            final_host_allocation_slots: resource.final_host_allocation_slots,
            final_host_generation_pin_slots: resource.final_host_generation_pin_slots,
            peak_host_retained_bytes: resource.peak_host_retained_bytes,
            peak_host_allocation_slots: resource.peak_host_allocation_slots,
            peak_host_generation_pin_slots: resource.peak_host_generation_pin_slots,
            old_generation_pinned_bytes: resource.old_generation_pinned_bytes,
            new_persistent_bytes: resource.new_persistent_bytes,
            retained_device_transient_bytes: resource.retained_device_transient_bytes,
            retained_device_result_bytes: resource.retained_device_result_bytes,
            incremental_allocation_slots: resource.incremental_allocation_slots,
            generation_pin_slots: resource.generation_pin_slots,
            maximum_concurrent_device_scratch_bytes: resource
                .maximum_concurrent_device_scratch_bytes,
            maximum_host_readback_bytes: resource.maximum_host_readback_bytes,
        }
    }

    /// Consume the only materialization capability.  Each branch revalidates its immutable
    /// witness and compares the actual owner ledger before returning any private reservation.
    pub(crate) fn materialize(
        self,
        engine: &'a Engine,
        table: &RelationalTable,
        expected_commit_seq: Index,
        permit: IndexedPhysicalMaterializationPermit,
    ) -> Result<PreparedIndexedPhysicalReservation<'a>, ExecuteError> {
        match self.branch {
            PreparedIndexedPhysicalForecastBranch::InPlace(preview) => super::index_delta::prepare(
                engine,
                table,
                preview,
                expected_commit_seq,
                permit,
                None,
                0,
            )
            .map(PreparedIndexedPhysicalReservation::in_place),
            PreparedIndexedPhysicalForecastBranch::FixedRollover(preview) => {
                super::index_rollover::materialize(
                    engine,
                    table,
                    preview,
                    expected_commit_seq,
                    permit,
                    None,
                    0,
                    None,
                )
                .map(PreparedIndexedPhysicalReservation::fixed_rollover)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fixed_forecast_has_no_cuda_work_before_the_permit_boundary() {
        let source = include_str!("index_rollover.rs");
        let forecast = source
            .split("pub(super) fn prepare_fixed_rollover_preview")
            .nth(1)
            .and_then(|section| section.split("fn fixed_rollover_logical_forecast").next())
            .expect("fixed metadata-only preview body");
        for forbidden in [
            "prepare_gpu_lookup_oracle",
            "prepare_resident_open_shard_append_indexed_fixed_rollover_reservation",
            "retain_device_memory_zeroed",
            "submit_resident",
            "CudaAllocationScope",
            "Vec",
            "BTree",
        ] {
            assert!(
                !forecast.contains(forbidden),
                "fixed forecast must remain allocation/CUDA-free before permit: {forbidden}"
            );
        }
    }

    #[test]
    fn indexed_materialization_requires_the_engine_insert_plan_permit() {
        let plan = include_str!("../engine_insert_plan.rs");
        assert!(plan.contains("struct IndexedPhysicalMaterializationPermit"));
        assert!(plan.contains("fn issue_codec5_terminal_indexed_physical_materialization_permit"));
        let fixed = include_str!("fixed_insert.rs");
        assert!(fixed.contains("_permit: IndexedPhysicalMaterializationPermit"));
        let in_place = include_str!("index_delta.rs");
        assert!(in_place
            .contains("permit: crate::engine_insert_plan::IndexedPhysicalMaterializationPermit"));
        let rollover = include_str!("index_rollover.rs");
        assert!(rollover.contains("permit: IndexedPhysicalMaterializationPermit"));
        assert!(rollover.contains("pub(super) fn materialize"));
        let forecast = include_str!("indexed_forecast.rs");
        assert!(forecast.contains("expected_commit_seq: Index"));
        assert!(forecast.contains("preview, expected_commit_seq, permit"));
    }
}
