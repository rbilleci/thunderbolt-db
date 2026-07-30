//! Exact all-i32 fused-tail ownership for the inert indexed in-place reservation.
//!
//! This leaf owns the allocation-free forecast, sealed append derivation, and opaque prepared
//! token. It deliberately has no WAL, apply, cache, descriptor, or publication operation.

use super::{
    DeviceInsertPlanPrepareError, PreparedResidentAppendBranch, ResidentOpenShardAppendPlan,
};
use crate::engine_insert_plan::host_retention::{HostRetentionGeometry, HostRetentionReport};
use crate::typed_insert_batch::PreparedResidentAppendSource;
use crate::{EngineError, ExecuteError, Index, RelationalResidentShard, SqlType};

/// Temporary exact inputs used only while constructing the final fused CUDA token. The boxed
/// values and destinations retire after copying into token staging; only the token and stamps
/// cross the pre-WAL boundary.
pub(in super::super) struct IndexedInPlaceFusedApplyInputs<'plan> {
    source_payload: std::sync::Arc<gpu_db_execution::CudaResidentDeviceMemory>,
    columns: Box<[gpu_db_execution::CudaWriteDestination]>,
    values: Box<[i32]>,
    created_by: gpu_db_execution::CudaWriteDestination,
    row_ids: &'plan [u64],
    row_id_destination: gpu_db_execution::CudaWriteDestination,
    base_row: u32,
    expected_commit_seq: Index,
    stamps: Box<[Index]>,
    footprint: gpu_db_execution::FusedApplyPreparationFootprint,
    materialization_host_scratch: HostRetentionGeometry,
}

/// Opaque pre-WAL fused payload/identity preparation. It exposes no apply, submit, header,
/// cache, descriptor, or publication operation.
pub(in super::super) struct PreparedIndexedInPlaceFusedApply {
    fused: gpu_db_execution::PreparedI32FusedApply,
    expected_commit_seq: Index,
    created_by_stamps: Box<[Index]>,
    footprint: gpu_db_execution::FusedApplyPreparationFootprint,
}

impl IndexedInPlaceFusedApplyInputs<'_> {
    pub(in super::super) fn preparation_bytes(&self) -> u64 {
        self.footprint.pooled_device_scratch_bytes
    }

    pub(in super::super) fn preparation_footprint(
        &self,
    ) -> gpu_db_execution::FusedApplyPreparationFootprint {
        self.footprint
    }

    pub(in super::super) fn materialization_host_scratch(&self) -> HostRetentionGeometry {
        self.materialization_host_scratch
    }

    pub(in super::super) fn materialize(
        self,
    ) -> Result<PreparedIndexedInPlaceFusedApply, DeviceInsertPlanPrepareError> {
        let preparation = gpu_db_execution::FusedApplyPreparation {
            columns: &self.columns,
            values: &self.values,
            created_by: self.created_by,
            row_ids: Some((self.row_ids, self.row_id_destination)),
            // The fused pass owns payload and sidecars only; indexed maintenance stays in its
            // independently prepared tail and cannot be reached through this token.
            index: None,
            base_row: self.base_row,
            header: gpu_db_execution::CudaWriteDestination {
                memory: std::sync::Arc::clone(&self.source_payload),
                byte_offset: 0,
            },
        };
        let fused = self
            .source_payload
            .prepare_i32_fused_apply(&preparation)
            .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
        if fused.preparation_bytes() != self.footprint.pooled_device_scratch_bytes {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        Ok(PreparedIndexedInPlaceFusedApply {
            fused,
            expected_commit_seq: self.expected_commit_seq,
            created_by_stamps: self.stamps,
            footprint: self.footprint,
        })
    }
}

impl PreparedIndexedInPlaceFusedApply {
    pub(in super::super) fn preparation_bytes(&self) -> u64 {
        self.fused.preparation_bytes()
    }

    pub(in super::super) fn preparation_footprint(
        &self,
    ) -> gpu_db_execution::FusedApplyPreparationFootprint {
        self.footprint
    }

    pub(in super::super) fn host_retention_report(
        &self,
    ) -> Result<HostRetentionReport, ExecuteError> {
        let retention = self.fused.host_retention_report().map_err(|error| {
            EngineError::ApplyFailed(format!(
                "indexed in-place fused host retention inspection failed: {error}"
            ))
        })?;
        if retention.owner_array_element_count != self.footprint.pinned_cuda_allocation_count
            || retention.owner_array_backing_bytes != self.footprint.owner_array_backing_bytes
            || retention.owner_array_allocation_slots != self.footprint.owner_array_allocation_slots
            || retention.staging_backing_bytes != self.footprint.staging_backing_bytes
            || retention.staging_allocation_slots != self.footprint.staging_allocation_slots
        {
            return Err(EngineError::ApplyFailed(
                "indexed in-place fused host retention drifted from its preflight footprint"
                    .to_string(),
            )
            .into());
        }
        let mut report = HostRetentionReport::default();
        if let Some(identity) = retention.owner_array_backing_identity {
            report.retain_external_backing(identity, retention.owner_array_backing_bytes)?;
        }
        if let Some(identity) = retention.staging_backing_identity {
            report.retain_external_backing(identity, retention.staging_backing_bytes)?;
        }
        report.retain_boxed_slice(&self.created_by_stamps)?;
        Ok(report)
    }

    pub(in super::super) fn stamps_match_expected_commit(&self) -> bool {
        self.created_by_stamps
            .iter()
            .all(|stamp| *stamp == self.expected_commit_seq)
    }
}

/// Derive fused inputs only from this plan's sealed source and exact open-shard identity.
pub(in super::super) fn prepare_inputs<'plan>(
    plan: &'plan ResidentOpenShardAppendPlan<'_>,
    expected_commit_seq: Index,
) -> Result<IndexedInPlaceFusedApplyInputs<'plan>, DeviceInsertPlanPrepareError> {
    let PreparedResidentAppendBranch::InPlace(in_place) = &plan.branch else {
        return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
    };
    if !source_is_all_i32_fixed(&plan.source)
        || plan.bootstrap_sentinel
        || plan
            .identity
            .row_count
            .checked_add(plan.source.row_count())
            .is_none_or(|end| end > plan.identity.capacity)
    {
        return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
    }
    let source_payload = plan
        .identity
        .device_memory
        .as_ref()
        .cloned()
        .filter(|memory| memory.device_ptr() != 0)
        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
    let created_by_memory = in_place
        .pending_created_by
        .as_ref()
        .map(|pending| std::sync::Arc::clone(&pending.region))
        .or_else(|| plan.identity.created_by_region.as_ref().cloned())
        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
    let row_ids = plan
        .row_ids
        .exact_slice()
        .filter(|ids| ids.len() == plan.source.row_count())
        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
    let row_id_memory = plan
        .identity
        .row_id_region
        .as_ref()
        .cloned()
        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
    let column_count = plan.source.columns().len();
    let expected_offsets = column_count
        .checked_add(1)
        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
    let offsets = plan
        .fixed_chunk_offsets
        .as_ref()
        .filter(|offsets| offsets.len() == expected_offsets && offsets.last() == Some(&0))
        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
    let rows = plan.source.row_count();
    let value_count = column_count
        .checked_mul(rows)
        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
    let mut values = Box::<[i32]>::new_uninit_slice(value_count);
    let mut value_offset = 0_usize;
    for column in plan.source.columns() {
        let values_source = column
            .i32_values()
            .filter(|values| values.len() == rows)
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        for value in values_source.iter().copied() {
            values[value_offset].write(value);
            value_offset += 1;
        }
    }
    if value_offset != value_count {
        return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
    }
    // SAFETY: the checked all-i32 source fills every exact column-major output slot once.
    let values = unsafe { values.assume_init() };
    let mut columns =
        Box::<[gpu_db_execution::CudaWriteDestination]>::new_uninit_slice(column_count);
    for (position, byte_offset) in offsets[..column_count].iter().copied().enumerate() {
        columns[position].write(gpu_db_execution::CudaWriteDestination {
            memory: std::sync::Arc::clone(&source_payload),
            byte_offset,
        });
    }
    // SAFETY: every slot corresponds to one checked all-i32 source column.
    let columns = unsafe { columns.assume_init() };
    let mut stamps = Box::<[Index]>::new_uninit_slice(rows);
    for stamp in stamps.iter_mut() {
        stamp.write(expected_commit_seq);
    }
    // SAFETY: the exact uniform loop initializes every stamp slot.
    let stamps = unsafe { stamps.assume_init() };
    let base_row = u32::try_from(plan.identity.row_count)
        .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
    let created_by = gpu_db_execution::CudaWriteDestination {
        memory: created_by_memory,
        byte_offset: u64::try_from(plan.identity.row_count)
            .ok()
            .and_then(|rows| rows.checked_mul(std::mem::size_of::<u64>() as u64))
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?,
    };
    let row_id_destination = gpu_db_execution::CudaWriteDestination {
        memory: row_id_memory,
        byte_offset: created_by.byte_offset,
    };
    let preparation = gpu_db_execution::FusedApplyPreparation {
        columns: &columns,
        values: &values,
        created_by: created_by.clone(),
        row_ids: Some((row_ids, row_id_destination.clone())),
        index: None,
        base_row,
        header: gpu_db_execution::CudaWriteDestination {
            memory: std::sync::Arc::clone(&source_payload),
            byte_offset: 0,
        },
    };
    let footprint = preparation
        .footprint(&source_payload)
        .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
    let materialization_host_scratch =
        indexed_in_place_fused_materialization_scratch_forecast(&plan.source, footprint)
            .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
    Ok(IndexedInPlaceFusedApplyInputs {
        source_payload,
        columns,
        values,
        created_by,
        row_ids,
        row_id_destination,
        base_row,
        expected_commit_seq,
        stamps,
        footprint,
        materialization_host_scratch,
    })
}

pub(in super::super) fn source_is_all_i32_fixed(source: &PreparedResidentAppendSource) -> bool {
    let rows = source.row_count();
    rows != 0
        && !source.requires_dense_rollover()
        && !source.columns().is_empty()
        && source.columns().iter().all(|column| {
            matches!(column.ty(), SqlType::Int2 | SqlType::Int4 | SqlType::Date)
                && column
                    .i32_values()
                    .is_some_and(|values| values.len() == rows)
        })
}

pub(in super::super) fn indexed_in_place_fused_footprint_forecast(
    source: &PreparedResidentAppendSource,
    open: &RelationalResidentShard,
) -> Result<gpu_db_execution::FusedApplyPreparationFootprint, EngineError> {
    if !source_is_all_i32_fixed(source) {
        return Err(EngineError::ApplyFailed(
            "indexed fused footprint requires an all-i32 source".to_string(),
        ));
    }
    let payload = open.device_memory.as_ref().ok_or_else(|| {
        EngineError::ApplyFailed("indexed fused footprint lost the open payload".to_string())
    })?;
    let row_ids = open.row_id_region.as_ref().ok_or_else(|| {
        EngineError::ApplyFailed("indexed fused footprint lost the row-id sidecar".to_string())
    })?;
    let base_row = u32::try_from(open.row_count).map_err(|_| {
        EngineError::ApplyFailed("indexed fused footprint base row overflows u32".to_string())
    })?;
    let rows = u32::try_from(source.row_count()).map_err(|_| {
        EngineError::ApplyFailed("indexed fused footprint row count overflows u32".to_string())
    })?;
    if base_row.checked_add(rows).is_none() {
        return Err(EngineError::ApplyFailed(
            "indexed fused footprint end row overflows u32".to_string(),
        ));
    }
    let mut identities = [payload.allocation_identity(), 0, 0];
    let mut identity_count = 1_usize;
    let mut pinned_count = 1_u64;
    if let Some(created_by) = open.created_by_region.as_ref() {
        let identity = created_by.allocation_identity();
        if !identities[..identity_count].contains(&identity) {
            identities[identity_count] = identity;
            identity_count += 1;
            pinned_count += 1;
        }
    } else {
        pinned_count += 1;
    }
    let row_id_identity = row_ids.allocation_identity();
    if !identities[..identity_count].contains(&row_id_identity) {
        pinned_count += 1;
    }
    gpu_db_execution::i32_fused_apply_footprint_for_shape(
        source.row_count(),
        source.columns().len(),
        true,
        pinned_count,
    )
    .map_err(|_| EngineError::ApplyFailed("indexed fused footprint geometry declined".to_string()))
}

pub(in super::super) fn indexed_in_place_fused_host_retention_forecast(
    source: &PreparedResidentAppendSource,
    footprint: gpu_db_execution::FusedApplyPreparationFootprint,
) -> Result<HostRetentionGeometry, EngineError> {
    let mut geometry = HostRetentionGeometry::default();
    geometry.checked_add_backing_bytes_slots(
        footprint.owner_array_backing_bytes,
        footprint.owner_array_allocation_slots,
        "indexed fused prepared owner array",
    )?;
    geometry.checked_add_backing_bytes_slots(
        footprint.staging_backing_bytes,
        footprint.staging_allocation_slots,
        "indexed fused prepared staging",
    )?;
    geometry.checked_add_backing_elements::<Index>(
        source.row_count(),
        "indexed fused uniform created-by stamps",
    )?;
    Ok(geometry)
}

pub(in super::super) fn indexed_in_place_fused_materialization_scratch_forecast(
    source: &PreparedResidentAppendSource,
    footprint: gpu_db_execution::FusedApplyPreparationFootprint,
) -> Result<HostRetentionGeometry, EngineError> {
    let mut geometry = HostRetentionGeometry::default();
    let value_count = source
        .columns()
        .len()
        .checked_mul(source.row_count())
        .ok_or_else(|| EngineError::ApplyFailed("indexed fused values overflow".to_string()))?;
    geometry.checked_add_backing_elements::<i32>(value_count, "indexed fused values staging")?;
    geometry.checked_add_backing_elements::<gpu_db_execution::CudaWriteDestination>(
        source.columns().len(),
        "indexed fused destination staging",
    )?;
    geometry.checked_add_backing_bytes_slots(
        footprint.maximum_temporary_host_scratch_bytes,
        u64::from(footprint.maximum_temporary_host_scratch_bytes != 0),
        "indexed fused execution preparation scratch",
    )?;
    Ok(geometry)
}
