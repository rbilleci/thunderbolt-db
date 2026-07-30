//! Physical resident-append source sealed from typed INSERT vectors.
//!
//! The parent module owns SQL semantic binding. This leaf owns the move-only source vocabulary
//! and dense payload boundary so physical layout cannot grow back into the semantic builder.

use super::*;

/// Move-only, catalog-order resident payload derived solely by consuming a sealed batch.
///
/// This is deliberately not a row carrier: the private vector arms remain the only values
/// authority between semantic binding, WAL templating, and resident-plan compilation.
pub(crate) struct PreparedResidentAppendSource {
    pub(super) table: TypedInsertBatchTable,
    pub(super) row_count: u32,
    pub(super) columns: Box<[PreparedResidentAppendColumn]>,
    pub(super) dependencies: Box<[TypedInsertDependencyBinding]>,
    pub(super) requires_dense_rollover: bool,
}

pub(crate) struct PreparedResidentAppendColumn {
    pub(super) column_id: u32,
    pub(super) attnum: i16,
    pub(super) ty: SqlType,
    pub(super) type_oid: u32,
    pub(super) type_size: i16,
    /// Present for fixed-width planning. Dense encoding consumes both logical vectors before the
    /// plan crosses WAL, retaining only the immutable catalog binding above.
    pub(super) validity: Option<TypedInsertColumnValidity>,
    pub(super) values: Option<TypedInsertColumnValues>,
}

/// One exact fixed-width device upload retained across WAL.  The source encoder writes this
/// final boxed owner directly; it must not first build a growable CUDA chunk and re-box it in
/// the plan compiler.
pub(crate) struct PreparedResidentFixedChunk {
    pub(crate) byte_offset: u64,
    pub(crate) bytes: Box<[u8]>,
}

impl PreparedResidentFixedChunk {
    pub(crate) fn append_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), EngineError> {
        report.retain_boxed_slice(&self.bytes)
    }
}

/// Paired exact owners move together so fused offset descriptors cannot outlive, or require
/// rebuilding from, their fixed upload owner after WAL.
pub(crate) struct PreparedResidentFixedChunkOwners {
    pub(crate) chunks: Box<[PreparedResidentFixedChunk]>,
    pub(crate) offsets: Box<[u64]>,
}

impl PreparedResidentFixedChunkOwners {
    pub(crate) fn append_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), EngineError> {
        report.retain_boxed_slice(&self.chunks)?;
        for chunk in self.chunks.iter() {
            chunk.append_host_retention(report)?;
        }
        report.retain_boxed_slice(&self.offsets)
    }
}

/// A plan-owned bitmap upload. Bool values stay bit-packed in the semantic source; this
/// byte-per-row view and its exact catalog name are the final fixed-plan owners.
pub(crate) struct PreparedResidentFixedBoolUpload {
    pub(crate) name: Box<str>,
    pub(crate) values: Box<[u8]>,
}

impl PreparedResidentFixedBoolUpload {
    pub(crate) fn append_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), EngineError> {
        report.retain_boxed_str(&self.name)?;
        report.retain_boxed_slice(&self.values)
    }
}

/// Exact dense device bytes and descriptor metadata sealed from the catalog-order vectors before
/// WAL. This is intentionally a physical payload, not a second logical row carrier.
pub(crate) struct PreparedResidentDensePayload {
    /// Present until the private pre-WAL allocation consumes it. Keeping it optional makes a
    /// second O(payload) zero buffer impossible while the plan crosses WAL.
    device_payload: Option<Vec<u8>>,
    final_count_header: [u8; std::mem::size_of::<u64>()],
    text_layouts: Vec<ResidentDeviceTextColumnLayout>,
    bool_layouts: Vec<ResidentDeviceBoolColumnLayout>,
    int4_stats: Vec<ResidentDeviceInt4ColumnStats>,
    null_layouts: Vec<ResidentDeviceNullBitmapLayout>,
}

/// Descriptor material retained after the one host payload has been moved into its private
/// pre-WAL device allocation. No logical values or O(payload) host copy survives this boundary.
pub(crate) struct PreparedDenseDescriptorParts {
    pub(crate) final_count_header: [u8; std::mem::size_of::<u64>()],
    pub(crate) text_layouts: Vec<ResidentDeviceTextColumnLayout>,
    pub(crate) bool_layouts: Vec<ResidentDeviceBoolColumnLayout>,
    pub(crate) int4_stats: Vec<ResidentDeviceInt4ColumnStats>,
    pub(crate) null_layouts: Vec<ResidentDeviceNullBitmapLayout>,
}

impl PreparedResidentDensePayload {
    /// Host backings retained by the dense rollover payload before its private device allocation
    /// consumes it. Device buffers and row IDs belong to physical/row-id domains, respectively.
    #[allow(dead_code)] // Adopted by the inert reservation carrier next.
    pub(crate) fn host_retention_report(&self) -> Result<HostRetentionReport, EngineError> {
        let mut report = HostRetentionReport::default();
        if let Some(payload) = self.device_payload.as_ref() {
            report.retain_vec(payload)?;
        }
        report.retain_vec(&self.text_layouts)?;
        for layout in self.text_layouts.iter() {
            report.retain_string(&layout.name)?;
        }
        report.retain_vec(&self.bool_layouts)?;
        for layout in self.bool_layouts.iter() {
            report.retain_string(&layout.name)?;
        }
        report.retain_vec(&self.int4_stats)?;
        for stat in self.int4_stats.iter() {
            report.retain_string(&stat.name)?;
        }
        report.retain_vec(&self.null_layouts)?;
        for layout in self.null_layouts.iter() {
            report.retain_string(&layout.name)?;
        }
        Ok(report)
    }

    pub(crate) fn device_payload_len(&self) -> Option<u64> {
        self.device_payload
            .as_ref()
            .and_then(|payload| u64::try_from(payload.len()).ok())
    }

    /// Move the one host payload into a private allocation upload. The count is deliberately
    /// zeroed so only post-WAL apply can expose the rows with its header-last write.
    pub(crate) fn take_pre_wal_upload(&mut self) -> Option<Vec<u8>> {
        let mut payload = self.device_payload.take()?;
        if payload.len() < self.final_count_header.len() {
            return None;
        }
        payload[..self.final_count_header.len()].fill(0);
        Some(payload)
    }

    pub(crate) fn into_descriptor_parts(self) -> PreparedDenseDescriptorParts {
        PreparedDenseDescriptorParts {
            final_count_header: self.final_count_header,
            text_layouts: self.text_layouts,
            bool_layouts: self.bool_layouts,
            int4_stats: self.int4_stats,
            null_layouts: self.null_layouts,
        }
    }
}

/// Build the dense rollover representation directly from sealed vectors before WAL. The source
/// remains the only logical values authority; this leaf merely fixes byte layout and metadata.
pub(super) fn checked_dense_payload(
    source: &mut PreparedResidentAppendSource,
    table: &RelationalTable,
) -> Result<PreparedResidentDensePayload, ExecuteError> {
    let rows = source.row_count();
    if rows == 0
        || table.columns.len() != source.columns.len()
        || !source.requires_dense_rollover()
        || !source.columns.iter().all(|column| {
            column
                .values
                .as_ref()
                .is_some_and(|values| values.rows_match(rows))
                && column
                    .validity
                    .as_ref()
                    .is_some_and(|validity| validity.shape_is_exact(rows))
                && column
                    .values
                    .as_ref()
                    .is_some_and(|values| values.text_invariants_hold(rows))
        })
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "sealed typed dense resident payload lost its vector geometry".to_string(),
        )));
    }
    if !table
        .columns
        .iter()
        .zip(&source.columns)
        .all(|(live, column)| {
            live.id == column.column_id
                && live.attnum == column.attnum
                && live.ty == column.ty
                && live.name.len() <= u16::MAX as usize
        })
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "sealed typed dense resident payload no longer matches its catalog".to_string(),
        )));
    }

    let mut device_payload = vec![0; std::mem::size_of::<u64>()];
    let mut text_layouts = Vec::new();
    let mut bool_layouts = Vec::new();
    let mut int4_stats = Vec::new();
    let mut null_layouts = Vec::new();

    for (column, live) in source.columns.iter().zip(&table.columns) {
        if !matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date) {
            continue;
        }
        let Some(TypedInsertColumnValues::I32(values)) = column.values.as_ref() else {
            return Err(super::dense_payload_shape_error());
        };
        let mut min = i32::MAX;
        let mut max = i32::MIN;
        for (row, value) in values.iter().copied().enumerate() {
            if column
                .validity
                .as_ref()
                .is_some_and(|validity| validity.is_valid(row))
            {
                min = min.min(value);
                max = max.max(value);
            }
            device_payload.extend_from_slice(&value.to_le_bytes());
        }
        int4_stats.push(ResidentDeviceInt4ColumnStats {
            name: live.name.clone(),
            min,
            max,
        });
    }
    for column in &source.columns {
        if !matches!(column.ty, SqlType::Int8 | SqlType::Timestamp) {
            continue;
        }
        let Some(TypedInsertColumnValues::I64(values)) = column.values.as_ref() else {
            return Err(super::dense_payload_shape_error());
        };
        for value in values {
            device_payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    for column in &source.columns {
        if !matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid) {
            continue;
        }
        match column.values.as_ref() {
            Some(TypedInsertColumnValues::I128(values))
                if matches!(column.ty, SqlType::Numeric { .. }) =>
            {
                for value in values {
                    device_payload.extend_from_slice(&value.to_le_bytes());
                }
            }
            Some(TypedInsertColumnValues::Bytes16(values)) if column.ty == SqlType::Uuid => {
                for value in values {
                    device_payload.extend_from_slice(value);
                }
            }
            _ => return Err(super::dense_payload_shape_error()),
        }
    }
    for (column, live) in source.columns.iter().zip(&table.columns) {
        if column.ty != SqlType::Bool {
            continue;
        }
        let Some(TypedInsertColumnValues::BoolBits(words)) = column.values.as_ref() else {
            return Err(super::dense_payload_shape_error());
        };
        if !super::bitmap_shape_is_exact(words, rows) {
            return Err(super::dense_payload_shape_error());
        }
        let bitmap_byte_offset =
            u64::try_from(device_payload.len()).map_err(|_| super::dense_payload_shape_error())?;
        for word in words {
            device_payload.extend_from_slice(&word.to_le_bytes());
        }
        bool_layouts.push(ResidentDeviceBoolColumnLayout {
            name: live.name.clone(),
            bitmap_byte_offset,
        });
    }
    for (column, live) in source.columns.iter().zip(&table.columns) {
        let Some(words) = column
            .validity
            .as_ref()
            .and_then(TypedInsertColumnValidity::bitmap_words)
        else {
            continue;
        };
        if !super::bitmap_shape_is_exact(words, rows) {
            return Err(super::dense_payload_shape_error());
        }
        let bitmap_byte_offset =
            u64::try_from(device_payload.len()).map_err(|_| super::dense_payload_shape_error())?;
        for word in words {
            device_payload.extend_from_slice(&word.to_le_bytes());
        }
        null_layouts.push(ResidentDeviceNullBitmapLayout {
            name: live.name.clone(),
            bitmap_byte_offset,
        });
    }
    for (column, live) in source.columns.iter().zip(&table.columns) {
        if column.ty != SqlType::Text {
            continue;
        }
        let Some(TypedInsertColumnValues::Text { offsets, bytes }) = column.values.as_ref() else {
            return Err(super::dense_payload_shape_error());
        };
        while !device_payload.len().is_multiple_of(8) {
            device_payload.push(0);
        }
        let offsets_byte_offset =
            u64::try_from(device_payload.len()).map_err(|_| super::dense_payload_shape_error())?;
        for offset in offsets {
            device_payload.extend_from_slice(&offset.to_le_bytes());
        }
        let bytes_byte_offset =
            u64::try_from(device_payload.len()).map_err(|_| super::dense_payload_shape_error())?;
        device_payload.extend_from_slice(bytes);
        text_layouts.push(ResidentDeviceTextColumnLayout {
            name: live.name.clone(),
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len: u64::try_from(bytes.len())
                .map_err(|_| super::dense_payload_shape_error())?,
        });
    }
    let final_count_header = u64::try_from(rows)
        .map_err(|_| super::dense_payload_shape_error())?
        .to_le_bytes();
    device_payload[..std::mem::size_of::<u64>()].copy_from_slice(&final_count_header);
    let prepared = PreparedResidentDensePayload {
        device_payload: Some(device_payload),
        final_count_header,
        text_layouts,
        bool_layouts,
        int4_stats,
        null_layouts,
    };
    // The dense allocation now owns the only physical payload bytes. Retaining logical vectors
    // through WAL would create a second authority and an O(payload) host lifetime.
    for column in source.columns.iter_mut() {
        column.values = None;
        column.validity = None;
    }
    Ok(prepared)
}
