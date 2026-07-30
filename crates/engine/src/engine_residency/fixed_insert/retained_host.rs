//! Exact host owners retained by a sealed fixed-width append plan.
//!
//! This child owns only deterministic plan metadata and host-upload storage.  It deliberately
//! knows nothing about CUDA allocation, WAL, descriptor publication, or the live carrier.

use super::{same_optional_device_region, DeviceInsertRowIds};
use crate::engine_insert_plan::host_retention::{HostRetentionGeometry, HostRetentionReport};
use crate::relational_model::{
    RelationalTable, ResidentDeviceBoolColumnLayout, ResidentDeviceInt4ColumnStats,
    ResidentDeviceNullBitmapLayout, ResidentDeviceTextColumnLayout,
};
use crate::typed_insert_batch::{
    PreparedResidentAppendSource, PreparedResidentFixedBoolUpload, PreparedResidentFixedChunk,
};
use crate::{EngineError, RelationalResidentShard, SqlType};
use gpu_db_execution::CudaResidentDeviceMemory;
use std::sync::Arc;

pub(super) struct PreparedOpenShardIdentity {
    pub(super) shard_id: u32,
    pub(super) capacity: usize,
    pub(super) row_count: usize,
    pub(super) row_start: usize,
    pub(super) gpu_id: u16,
    // This identity crosses the pre-WAL boundary. Keep every host owner exact-length so a
    // side-effect-free reservation prediction does not depend on allocator growth history.
    schema: Box<str>,
    pub(super) int4_columns: Box<[Box<str>]>,
    pub(super) int8_columns: Box<[Box<str>]>,
    pub(super) numeric_columns: Box<[Box<str>]>,
    pub(super) bool_layouts: Box<[PreparedBoolLayout]>,
    text_layouts: Box<[PreparedTextLayout]>,
    null_layouts: Box<[PreparedNullLayout]>,
    generation: Arc<()>,
    /// The exact published payload generation that owns every fused destination.  This is a
    /// device allocation pin, not a host-retention backing; retaining it here keeps the future
    /// prepared fused launch independent of cache lookup or later descriptor loads.
    pub(super) device_memory: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) created_by_region: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) row_id_region: Option<Arc<CudaResidentDeviceMemory>>,
}

impl PreparedOpenShardIdentity {
    pub(super) fn from_open(open: &RelationalResidentShard) -> Self {
        Self {
            shard_id: open.shard_id,
            capacity: open.capacity,
            row_count: open.row_count,
            row_start: open.row_start,
            gpu_id: open.gpu_id,
            schema: Box::from(open.schema.as_str()),
            int4_columns: exact_names(&open.resident_device_int4_columns),
            int8_columns: exact_names(&open.resident_device_int8_columns),
            numeric_columns: exact_names(&open.resident_device_numeric_columns),
            bool_layouts: open
                .resident_device_bool_columns
                .iter()
                .map(PreparedBoolLayout::from_live)
                .collect(),
            text_layouts: open
                .resident_device_text_columns
                .iter()
                .map(PreparedTextLayout::from_live)
                .collect(),
            null_layouts: open
                .resident_device_null_columns
                .iter()
                .map(PreparedNullLayout::from_live)
                .collect(),
            generation: Arc::clone(&open.point_route_generation),
            device_memory: open.device_memory.clone(),
            created_by_region: open.created_by_region.clone(),
            row_id_region: open.row_id_region.clone(),
        }
    }

    pub(super) fn matches(&self, open: &RelationalResidentShard, pressured: bool) -> bool {
        open.int4_appendable
            && open.is_valid(pressured)
            && open.shard_id == self.shard_id
            && open.capacity == self.capacity
            && open.row_count == self.row_count
            && open.row_start == self.row_start
            && open.gpu_id == self.gpu_id
            && open.schema == self.schema.as_ref()
            && names_match(&open.resident_device_int4_columns, &self.int4_columns)
            && names_match(&open.resident_device_int8_columns, &self.int8_columns)
            && names_match(&open.resident_device_numeric_columns, &self.numeric_columns)
            && bool_layouts_match(&open.resident_device_bool_columns, &self.bool_layouts)
            && text_layouts_match(&open.resident_device_text_columns, &self.text_layouts)
            && null_layouts_match(&open.resident_device_null_columns, &self.null_layouts)
            && Arc::ptr_eq(&open.point_route_generation, &self.generation)
            && same_optional_device_region(&open.device_memory, &self.device_memory)
            && same_optional_device_region(&open.created_by_region, &self.created_by_region)
            && same_optional_device_region(&open.row_id_region, &self.row_id_region)
    }

    /// Validate the fixed descriptor against borrowed catalog columns.  This deliberately avoids
    /// cloning source types or names merely to compare a sealed identity.
    pub(super) fn matches_scalar_table_layout(&self, table: &RelationalTable) -> bool {
        exact_names_match_table(&self.int4_columns, table, |ty| {
            matches!(ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date)
        }) && exact_names_match_table(&self.int8_columns, table, |ty| {
            matches!(ty, SqlType::Int8 | SqlType::Timestamp)
        }) && exact_names_match_table(&self.numeric_columns, table, |ty| {
            matches!(ty, SqlType::Numeric { .. } | SqlType::Uuid)
        })
    }

    pub(super) fn matches_fixed_table_layout(&self, table: &RelationalTable) -> bool {
        self.matches_scalar_table_layout(table)
            && descriptor_bool_layouts_match(table, self.capacity, &self.bool_layouts)
    }

    pub(super) fn append_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), EngineError> {
        report.retain_boxed_str(&self.schema)?;
        append_exact_name_box(report, &self.int4_columns)?;
        append_exact_name_box(report, &self.int8_columns)?;
        append_exact_name_box(report, &self.numeric_columns)?;
        append_exact_bool_layout_box(report, &self.bool_layouts)?;
        append_exact_text_layout_box(report, &self.text_layouts)?;
        append_exact_null_layout_box(report, &self.null_layouts)
    }

    fn host_retention_prediction(
        open: &RelationalResidentShard,
    ) -> Result<HostRetentionGeometry, EngineError> {
        let mut geometry = HostRetentionGeometry::default();
        append_boxed_str_prediction(&mut geometry, &open.schema)?;
        append_exact_name_prediction(&mut geometry, &open.resident_device_int4_columns)?;
        append_exact_name_prediction(&mut geometry, &open.resident_device_int8_columns)?;
        append_exact_name_prediction(&mut geometry, &open.resident_device_numeric_columns)?;
        append_bool_layout_prediction(&mut geometry, &open.resident_device_bool_columns)?;
        append_text_layout_prediction(&mut geometry, &open.resident_device_text_columns)?;
        append_null_layout_prediction(&mut geometry, &open.resident_device_null_columns)?;
        Ok(geometry)
    }
}

pub(super) struct PreparedBoolLayout {
    name: Box<str>,
    bitmap_byte_offset: u64,
}

impl PreparedBoolLayout {
    fn from_live(value: &ResidentDeviceBoolColumnLayout) -> Self {
        Self {
            name: Box::from(value.name.as_str()),
            bitmap_byte_offset: value.bitmap_byte_offset,
        }
    }
}

struct PreparedTextLayout {
    name: Box<str>,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
}

impl PreparedTextLayout {
    fn from_live(value: &ResidentDeviceTextColumnLayout) -> Self {
        Self {
            name: Box::from(value.name.as_str()),
            offsets_byte_offset: value.offsets_byte_offset,
            bytes_byte_offset: value.bytes_byte_offset,
            bytes_len: value.bytes_len,
        }
    }
}

struct PreparedNullLayout {
    name: Box<str>,
    bitmap_byte_offset: u64,
}

impl PreparedNullLayout {
    fn from_live(value: &ResidentDeviceNullBitmapLayout) -> Self {
        Self {
            name: Box::from(value.name.as_str()),
            bitmap_byte_offset: value.bitmap_byte_offset,
        }
    }
}

fn exact_names(values: &[String]) -> Box<[Box<str>]> {
    values
        .iter()
        .map(|value| Box::from(value.as_str()))
        .collect()
}

fn exact_names_match_table(
    prepared: &[Box<str>],
    table: &RelationalTable,
    accepts: impl Fn(&SqlType) -> bool,
) -> bool {
    prepared.len()
        == table
            .columns
            .iter()
            .filter(|column| accepts(&column.ty))
            .count()
        && prepared
            .iter()
            .zip(table.columns.iter().filter(|column| accepts(&column.ty)))
            .all(|(prepared, column)| prepared.as_ref() == column.name)
}

#[allow(dead_code)] // exact live-to-prepared tests exercise table comparison below
pub(super) fn names_match(live: &[String], prepared: &[Box<str>]) -> bool {
    live.len() == prepared.len()
        && live
            .iter()
            .zip(prepared)
            .all(|(live, prepared)| live == prepared.as_ref())
}

fn bool_layouts_match(
    live: &[ResidentDeviceBoolColumnLayout],
    prepared: &[PreparedBoolLayout],
) -> bool {
    live.len() == prepared.len()
        && live.iter().zip(prepared).all(|(live, prepared)| {
            live.name == prepared.name.as_ref()
                && live.bitmap_byte_offset == prepared.bitmap_byte_offset
        })
}

fn text_layouts_match(
    live: &[ResidentDeviceTextColumnLayout],
    prepared: &[PreparedTextLayout],
) -> bool {
    live.len() == prepared.len()
        && live.iter().zip(prepared).all(|(live, prepared)| {
            live.name == prepared.name.as_ref()
                && live.offsets_byte_offset == prepared.offsets_byte_offset
                && live.bytes_byte_offset == prepared.bytes_byte_offset
                && live.bytes_len == prepared.bytes_len
        })
}

fn null_layouts_match(
    live: &[ResidentDeviceNullBitmapLayout],
    prepared: &[PreparedNullLayout],
) -> bool {
    live.len() == prepared.len()
        && live.iter().zip(prepared).all(|(live, prepared)| {
            live.name == prepared.name.as_ref()
                && live.bitmap_byte_offset == prepared.bitmap_byte_offset
        })
}

fn append_exact_name_box(
    report: &mut HostRetentionReport,
    values: &[Box<str>],
) -> Result<(), EngineError> {
    report.retain_boxed_slice(values)?;
    for value in values {
        report.retain_boxed_str(value)?;
    }
    Ok(())
}

fn append_exact_bool_layout_box(
    report: &mut HostRetentionReport,
    values: &[PreparedBoolLayout],
) -> Result<(), EngineError> {
    report.retain_boxed_slice(values)?;
    for value in values {
        report.retain_boxed_str(&value.name)?;
    }
    Ok(())
}

fn append_exact_text_layout_box(
    report: &mut HostRetentionReport,
    values: &[PreparedTextLayout],
) -> Result<(), EngineError> {
    report.retain_boxed_slice(values)?;
    for value in values {
        report.retain_boxed_str(&value.name)?;
    }
    Ok(())
}

fn append_exact_null_layout_box(
    report: &mut HostRetentionReport,
    values: &[PreparedNullLayout],
) -> Result<(), EngineError> {
    report.retain_boxed_slice(values)?;
    for value in values {
        report.retain_boxed_str(&value.name)?;
    }
    Ok(())
}

pub(super) fn descriptor_bool_layouts_match(
    table: &RelationalTable,
    capacity: usize,
    actual: &[PreparedBoolLayout],
) -> bool {
    if table.columns.is_empty()
        || table.columns.iter().any(|column| {
            !matches!(
                column.ty,
                SqlType::Int2
                    | SqlType::Int4
                    | SqlType::Date
                    | SqlType::Int8
                    | SqlType::Timestamp
                    | SqlType::Numeric { .. }
                    | SqlType::Uuid
                    | SqlType::Bool
            )
        })
    {
        return false;
    }
    let capacity = match u64::try_from(capacity) {
        Ok(capacity) => capacity,
        Err(_) => return false,
    };
    let mut bytes = std::mem::size_of::<u64>() as u64;
    for width in [4_u64, 8, 16] {
        let count = table
            .columns
            .iter()
            .filter(|column| match width {
                4 => matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date),
                8 => matches!(column.ty, SqlType::Int8 | SqlType::Timestamp),
                16 => matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid),
                _ => false,
            })
            .count();
        let section = u64::try_from(count)
            .ok()
            .and_then(|count| count.checked_mul(capacity))
            .and_then(|bytes| bytes.checked_mul(width));
        let Some(section) = section else {
            return false;
        };
        let Some(next) = bytes.checked_add(section) else {
            return false;
        };
        bytes = next;
    }
    let Some(bool_bytes) = capacity
        .div_ceil(32)
        .checked_mul(std::mem::size_of::<u32>() as u64)
    else {
        return false;
    };
    let mut actual = actual.iter();
    for column in table
        .columns
        .iter()
        .filter(|column| column.ty == SqlType::Bool)
    {
        let Some(layout) = actual.next() else {
            return false;
        };
        if layout.name.as_ref() != column.name || layout.bitmap_byte_offset != bytes {
            return false;
        }
        let Some(next) = bytes.checked_add(bool_bytes) else {
            return false;
        };
        bytes = next;
    }
    actual.next().is_none()
}

pub(super) fn append_pending_bool_layout_box(
    report: &mut HostRetentionReport,
    values: &[ResidentDeviceBoolColumnLayout],
) -> Result<(), EngineError> {
    report.retain_boxed_slice(values)?;
    for value in values {
        report.retain_string(&value.name)?;
    }
    Ok(())
}

pub(super) fn append_pending_int4_stats_box(
    report: &mut HostRetentionReport,
    values: &[ResidentDeviceInt4ColumnStats],
) -> Result<(), EngineError> {
    report.retain_boxed_slice(values)?;
    for value in values {
        report.retain_string(&value.name)?;
    }
    Ok(())
}

fn append_boxed_str_prediction(
    geometry: &mut HostRetentionGeometry,
    value: &str,
) -> Result<(), EngineError> {
    geometry.checked_add_backing_elements::<u8>(value.len(), "boxed host string")
}

fn append_exact_name_prediction(
    geometry: &mut HostRetentionGeometry,
    values: &[String],
) -> Result<(), EngineError> {
    geometry.checked_add_backing_elements::<Box<str>>(values.len(), "boxed host name array")?;
    for value in values {
        append_boxed_str_prediction(geometry, value)?;
    }
    Ok(())
}

fn append_bool_layout_prediction(
    geometry: &mut HostRetentionGeometry,
    values: &[ResidentDeviceBoolColumnLayout],
) -> Result<(), EngineError> {
    geometry.checked_add_backing_elements::<PreparedBoolLayout>(
        values.len(),
        "boxed host bool layout array",
    )?;
    for value in values {
        append_boxed_str_prediction(geometry, &value.name)?;
    }
    Ok(())
}

fn append_text_layout_prediction(
    geometry: &mut HostRetentionGeometry,
    values: &[ResidentDeviceTextColumnLayout],
) -> Result<(), EngineError> {
    geometry.checked_add_backing_elements::<PreparedTextLayout>(
        values.len(),
        "boxed host text layout array",
    )?;
    for value in values {
        append_boxed_str_prediction(geometry, &value.name)?;
    }
    Ok(())
}

fn append_null_layout_prediction(
    geometry: &mut HostRetentionGeometry,
    values: &[ResidentDeviceNullBitmapLayout],
) -> Result<(), EngineError> {
    geometry.checked_add_backing_elements::<PreparedNullLayout>(
        values.len(),
        "boxed host NULL layout array",
    )?;
    for value in values {
        append_boxed_str_prediction(geometry, &value.name)?;
    }
    Ok(())
}

fn append_pending_bool_layout_prediction(
    geometry: &mut HostRetentionGeometry,
    table: &RelationalTable,
) -> Result<usize, EngineError> {
    let count = table
        .columns
        .iter()
        .filter(|column| column.ty == SqlType::Bool)
        .count();
    geometry.checked_add_backing_elements::<ResidentDeviceBoolColumnLayout>(
        count,
        "boxed fixed rollover bool layout array",
    )?;
    for value in table
        .columns
        .iter()
        .filter(|column| column.ty == SqlType::Bool)
    {
        append_boxed_str_prediction(geometry, &value.name)?;
    }
    Ok(count)
}

fn append_pending_int4_stats_prediction(
    geometry: &mut HostRetentionGeometry,
    table: &RelationalTable,
) -> Result<usize, EngineError> {
    let count = table
        .columns
        .iter()
        .filter(|column| matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
        .count();
    geometry.checked_add_backing_elements::<ResidentDeviceInt4ColumnStats>(
        count,
        "boxed fixed rollover int4 statistics array",
    )?;
    for value in table
        .columns
        .iter()
        .filter(|column| matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
    {
        append_boxed_str_prediction(geometry, &value.name)?;
    }
    Ok(count)
}

/// Allocation-free geometry for the new plan-owned in-place backings. It deliberately excludes
/// source/proof owners, whose real identities are only merged by post-materialization diagnostics.
pub(super) fn indexed_in_place_plan_owned_host_retention_prediction(
    source: &PreparedResidentAppendSource,
    row_ids: &DeviceInsertRowIds,
    open: &RelationalResidentShard,
    table: &RelationalTable,
) -> Result<HostRetentionGeometry, EngineError> {
    let source_geometry = source
        .fixed_append_host_owner_geometry(open.capacity, open.row_count)
        .ok_or_else(|| {
            EngineError::ApplyFailed(
                "indexed in-place host retention prediction lost fixed append geometry".to_string(),
            )
        })?;
    let table_bool_count = table
        .columns
        .iter()
        .filter(|column| column.ty == SqlType::Bool)
        .count();
    if source_geometry.bool_count != table_bool_count {
        return Err(EngineError::ApplyFailed(
            "indexed in-place host retention prediction bool geometry drifted".to_string(),
        ));
    }
    let mut geometry = PreparedOpenShardIdentity::host_retention_prediction(open)?;
    row_ids.predict_host_allocation_slot(&mut geometry)?;
    geometry.checked_add_backing_elements::<PreparedResidentFixedChunk>(
        source_geometry.chunk_count,
        "boxed fixed append chunk array",
    )?;
    let chunk_slots = u64::try_from(source_geometry.chunk_count).map_err(|_| {
        EngineError::Durability("fixed append chunk allocation slots overflow".to_string())
    })?;
    geometry.checked_add_backing_bytes_slots(
        source_geometry.chunk_payload_bytes,
        chunk_slots,
        "boxed fixed append chunk bytes",
    )?;
    geometry.checked_add_backing_elements::<u64>(
        source_geometry.chunk_count,
        "boxed fixed append chunk offsets",
    )?;
    geometry.checked_add_backing_elements::<(i32, i32)>(
        source_geometry.int4_count,
        "boxed fixed append int4 min/max",
    )?;
    geometry.checked_add_backing_elements::<PreparedResidentFixedBoolUpload>(
        source_geometry.bool_count,
        "boxed fixed append bool upload array",
    )?;
    let rows = u64::try_from(source.row_count())
        .map_err(|_| EngineError::Durability("fixed append bool rows overflow".to_string()))?;
    let bool_slots = u64::try_from(source_geometry.bool_count).map_err(|_| {
        EngineError::Durability("fixed append bool allocation slots overflow".to_string())
    })?;
    geometry.checked_add_backing_bytes_slots(
        rows.checked_mul(bool_slots).ok_or_else(|| {
            EngineError::Durability("fixed append bool bytes overflow".to_string())
        })?,
        bool_slots,
        "boxed fixed append bool bytes",
    )?;
    for column in table
        .columns
        .iter()
        .filter(|column| column.ty == SqlType::Bool)
    {
        append_boxed_str_prediction(&mut geometry, &column.name)?;
    }
    Ok(geometry)
}

/// Allocation-free geometry for fixed rollover's plan-owned descriptor metadata.
pub(super) fn indexed_fixed_rollover_plan_owned_host_retention_prediction(
    source: &PreparedResidentAppendSource,
    row_ids: &DeviceInsertRowIds,
    open: &RelationalResidentShard,
    table: &RelationalTable,
) -> Result<HostRetentionGeometry, EngineError> {
    let source_geometry = source
        .fixed_append_host_owner_geometry(source.row_count(), 0)
        .ok_or_else(|| {
            EngineError::ApplyFailed(
                "indexed fixed rollover host retention prediction lost fixed geometry".to_string(),
            )
        })?;
    let mut geometry = PreparedOpenShardIdentity::host_retention_prediction(open)?;
    row_ids.predict_host_allocation_slot(&mut geometry)?;
    if append_pending_bool_layout_prediction(&mut geometry, table)? != source_geometry.bool_count
        || append_pending_int4_stats_prediction(&mut geometry, table)? != source_geometry.int4_count
    {
        return Err(EngineError::ApplyFailed(
            "indexed fixed rollover host retention prediction descriptor drifted".to_string(),
        ));
    }
    Ok(geometry)
}

/// Exact permit-time host scratch consumed while the fixed payload is encoded and uploaded.
///
/// Catalog names and descriptor arrays move into the retained pending shard, so this counts only
/// owners that retire before the append plan escapes: fixed chunks/offsets, expanded BOOL bytes,
/// and the row-ID upload image.
pub(super) fn indexed_fixed_rollover_materialization_scratch_prediction(
    source: &PreparedResidentAppendSource,
    row_ids: &DeviceInsertRowIds,
    table: &RelationalTable,
) -> Result<HostRetentionGeometry, EngineError> {
    if !row_ids.is_exact() {
        return Err(EngineError::ApplyFailed(
            "indexed fixed rollover scratch prediction requires exact row identities".to_string(),
        ));
    }
    let source_geometry = source
        .fixed_append_host_owner_geometry(source.row_count(), 0)
        .ok_or_else(|| {
            EngineError::ApplyFailed(
                "indexed fixed rollover scratch prediction lost fixed geometry".to_string(),
            )
        })?;
    let table_bool_count = table
        .columns
        .iter()
        .filter(|column| column.ty == SqlType::Bool)
        .count();
    if source_geometry.bool_count != table_bool_count {
        return Err(EngineError::ApplyFailed(
            "indexed fixed rollover scratch prediction bool geometry drifted".to_string(),
        ));
    }
    let mut geometry = HostRetentionGeometry::default();
    geometry.checked_add_backing_elements::<PreparedResidentFixedChunk>(
        source_geometry.chunk_count,
        "fixed rollover materialization chunk array",
    )?;
    geometry.checked_add_backing_bytes_slots(
        source_geometry.chunk_payload_bytes,
        u64::try_from(source_geometry.chunk_count).map_err(|_| {
            EngineError::Durability(
                "fixed rollover materialization chunk slots overflow".to_string(),
            )
        })?,
        "fixed rollover materialization chunk payloads",
    )?;
    geometry.checked_add_backing_elements::<u64>(
        source_geometry.chunk_count,
        "fixed rollover materialization chunk offsets",
    )?;
    geometry.checked_add_backing_elements::<PreparedResidentFixedBoolUpload>(
        source_geometry.bool_count,
        "fixed rollover materialization bool upload array",
    )?;
    let bool_slots = u64::try_from(source_geometry.bool_count).map_err(|_| {
        EngineError::Durability("fixed rollover materialization bool slots overflow".to_string())
    })?;
    let rows = u64::try_from(source.row_count()).map_err(|_| {
        EngineError::Durability("fixed rollover materialization row count overflows".to_string())
    })?;
    geometry.checked_add_backing_bytes_slots(
        rows.checked_mul(bool_slots).ok_or_else(|| {
            EngineError::Durability(
                "fixed rollover materialization bool bytes overflow".to_string(),
            )
        })?,
        bool_slots,
        "fixed rollover materialization bool values",
    )?;
    geometry.checked_add_backing_bytes_slots(
        rows.checked_mul(std::mem::size_of::<u64>() as u64)
            .ok_or_else(|| {
                EngineError::Durability(
                    "fixed rollover materialization row-id bytes overflow".to_string(),
                )
            })?,
        1,
        "fixed rollover materialization row-id upload",
    )?;
    Ok(geometry)
}
