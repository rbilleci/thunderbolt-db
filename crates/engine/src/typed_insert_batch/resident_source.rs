//! Physical resident-append source sealed from typed INSERT vectors.
//!
//! The parent module owns SQL semantic binding. This leaf owns the move-only source vocabulary
//! and dense payload boundary so physical layout cannot grow back into the semantic builder.

use super::*;
#[cfg(test)]
use sha2::{Digest, Sha256};

/// Move-only, catalog-order resident payload derived solely by consuming a sealed batch.
///
/// This is deliberately not a row carrier: the private vector arms remain the only values
/// authority between semantic binding, WAL templating, and resident-plan compilation.
pub(crate) struct PreparedResidentAppendSource {
    pub(super) table: TypedInsertBatchTable,
    pub(super) row_count: u32,
    pub(super) columns: Box<[PreparedResidentAppendColumn]>,
    /// Exact payload geometry derived while the strict final image still owns the vectors.
    /// Runtime generation needs this scalar to reserve its encoder, but must not rescan every
    /// logical cell immediately before `write_rows` consumes those same vectors.
    pub(super) runtime_value_bytes: usize,
    pub(super) dependencies: Box<[TypedInsertDependencyBinding]>,
    pub(super) requires_dense_rollover: bool,
}

impl PreparedResidentAppendSource {
    /// Strict constructor for a decoded final-table image. The image remains move-only:
    /// successful construction transfers its typed vector owners, while a catalog mismatch drops
    /// the complete image and exposes no partially materialized append source.
    pub(crate) fn from_decoded_final_table_image(
        image: DecodedTypedImage,
        table: &RelationalTable,
        table_schema_digest: gpu_db_wal::CanonicalDigest,
        prepared_catalog_seq: Index,
    ) -> Result<Self, EngineError> {
        image.into_resident_append_source(table, table_schema_digest, prepared_catalog_seq)
    }

    /// Compose the table-local statements and bind their shared image to its plural S7 table
    /// reference before either CUDA generation or WAL closure observes the layout digest.
    pub(crate) fn from_decoded_final_table_images_for_table_ref(
        mut images: Vec<(DecodedTypedImage, Arc<[u8]>)>,
        table_ref: u32,
        table: &RelationalTable,
        table_schema_digest: gpu_db_wal::CanonicalDigest,
        prepared_catalog_seq: Index,
    ) -> Result<(Self, Arc<[u8]>), EngineError> {
        let (image, encoded) = if images.len() == 1 {
            images.pop().expect("one checked final image")
        } else {
            return DecodedTypedImage::concatenate_final_table_images_for_table_ref(
                images.into_iter().map(|(image, _)| image).collect(),
                table_ref,
            )
            .and_then(|(image, encoded)| {
                image
                    .into_resident_append_source(table, table_schema_digest, prepared_catalog_seq)
                    .map(|source| (source, encoded))
            });
        };
        // Statement sealing uses table reference zero. The first table in the deterministic S7
        // order is also zero, so the strict source image is already the exact final authority.
        // Do not produce a byte-identical replacement and then strictly decode it a second time:
        // the original decode above authenticated its grammar and layout digest, and the opaque
        // Arc remains the same sole S7/GPU image authority. Nonzero table references still take
        // the explicit rebind because that reference participates in the layout digest.
        if image.facts().role == TypedImageRole::FinalTableImage
            && image.columns().all(|column| column.table_ref == table_ref)
        {
            return image
                .into_resident_append_source(table, table_schema_digest, prepared_catalog_seq)
                .map(|source| (source, encoded));
        }
        let (image, encoded) = image.rebind_final_table_ref(table_ref)?;
        let source =
            image.into_resident_append_source(table, table_schema_digest, prepared_catalog_seq)?;
        Ok((source, encoded))
    }
}

/// Borrowed, fully checked row-major view of the same catalog-order source used by resident
/// apply. It creates no row matrix and owns no fallback bytes; the type-neutral runtime encoder
/// consumes logical cells through this view before the source moves into physical apply.
pub(crate) struct PreparedResidentRuntimeGenerationView<'a> {
    pub(super) source: &'a PreparedResidentAppendSource,
    pub(super) geometry: gpu_db_execution::RuntimeTypedInsertGenerationGeometry,
    pub(super) first_row_id: u64,
    pub(super) row_sources:
        &'a [crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource],
}

impl PreparedResidentRuntimeGenerationView<'_> {
    pub(crate) fn geometry(&self) -> gpu_db_execution::RuntimeTypedInsertGenerationGeometry {
        self.geometry
    }

    pub(crate) fn first_row_id(&self) -> u64 {
        self.first_row_id
    }

    #[cfg(test)]
    pub(crate) fn final_row_digest(
        &self,
        row_ordinal: usize,
        image_ref: u32,
        image_row_ordinal: u32,
    ) -> Result<[u8; 32], EngineError> {
        if row_ordinal >= self.geometry.rows {
            return Err(generation_source_error("final-row ordinal"));
        }
        let stable_row_id = self
            .first_row_id
            .checked_add(
                u64::try_from(row_ordinal)
                    .map_err(|_| generation_source_error("final-row identity ordinal"))?,
            )
            .ok_or_else(|| generation_source_error("final-row identity"))?;
        let domain = b"gpu-db/write001/s7-final-row/v2";
        let mut digest = Sha256::new();
        digest.update((domain.len() as u64).to_le_bytes());
        digest.update(domain);
        digest.update(self.source.table.stable_table_id.to_le_bytes());
        digest.update(stable_row_id.to_le_bytes());
        digest.update(image_ref.to_le_bytes());
        digest.update(image_row_ordinal.to_le_bytes());
        digest.update(
            u32::try_from(self.source.columns.len())
                .map_err(|_| generation_source_error("final-row column count"))?
                .to_le_bytes(),
        );
        for (ordinal, column) in self.source.columns.iter().enumerate() {
            digest.update(
                u32::try_from(ordinal)
                    .map_err(|_| generation_source_error("final-row column ordinal"))?
                    .to_le_bytes(),
            );
            digest.update(column.column_id.to_le_bytes());
            digest.update(column.attnum.to_le_bytes());
            digest.update(typed_image_sql_storage(column.ty));
            digest.update(column.type_oid.to_le_bytes());
            digest.update(column.type_size.to_le_bytes());
            let validity = column
                .validity
                .as_ref()
                .ok_or_else(|| generation_source_error("final-row validity"))?;
            let values = column
                .values
                .as_ref()
                .ok_or_else(|| generation_source_error("final-row values"))?;
            with_logical_cell(
                validity,
                values,
                column.ty,
                row_ordinal,
                |is_null, value| {
                    digest.update([u8::from(is_null)]);
                    digest.update(
                        u32::try_from(value.len())
                            .expect("typed final-row value length fits u32")
                            .to_le_bytes(),
                    );
                    digest.update(value);
                },
            )?;
        }
        Ok(digest.finalize().into())
    }

    pub(crate) fn write_rows(
        &self,
        encoder: &mut gpu_db_execution::RuntimeTypedInsertGenerationEncoder<'_>,
    ) {
        let table_id = self.source.table.stable_table_id;
        for row in 0..self.geometry.rows {
            let row_source = self.row_sources[row];
            encoder.write_row(gpu_db_execution::RuntimeTypedInsertGenerationRow {
                stable_table_id: table_id,
                stable_row_id: row_source.stable_row_id,
                source_statement_ordinal: row_source.statement_ordinal,
                source_row_ordinal: row_source.source_row_ordinal,
                cell_count: u32::try_from(self.source.columns.len())
                    .expect("prepared generation column count fits u32"),
            });
            for (ordinal, column) in self.source.columns.iter().enumerate() {
                let validity = column
                    .validity
                    .as_ref()
                    .expect("prepared generation retains validity");
                let values = column
                    .values
                    .as_ref()
                    .expect("prepared generation retains values");
                with_logical_cell(validity, values, column.ty, row, |is_null, value| {
                    encoder.write_cell(gpu_db_execution::RuntimeTypedInsertGenerationCell {
                        catalog_column_ordinal: u32::try_from(ordinal)
                            .expect("typed column ordinal fits u32"),
                        stable_column_id: column.column_id,
                        attnum: column.attnum,
                        storage: typed_image_sql_storage(column.ty),
                        declared_type_oid: column.type_oid,
                        signed_type_size: column.type_size,
                        is_null,
                        value,
                    });
                })
                .expect("prepared generation cell remains valid");
            }
        }
    }
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

/// Compute the exact variable payload extent without re-walking every logical cell at runtime.
///
/// The only constructor consumes `DecodedTypedImage`, whose strict decoder has already checked
/// vector shapes, bitmap tails, text offsets, UTF-8, and type/vector agreement. The source is
/// then private and move-only. Keep the scalar computation at that ownership boundary so runtime
/// generation can retain its identity checks without duplicating the logical-value traversal that
/// `write_rows` performs to emit its ABI.
pub(super) fn runtime_generation_value_bytes(
    rows: usize,
    columns: &[PreparedResidentAppendColumn],
) -> Result<usize, EngineError> {
    let mut total = 0_usize;
    for column in columns {
        let validity = column
            .validity
            .as_ref()
            .ok_or_else(|| generation_source_error("value-byte validity"))?;
        let values = column
            .values
            .as_ref()
            .ok_or_else(|| generation_source_error("value-byte values"))?;
        let valid_rows = match validity {
            TypedInsertColumnValidity::AllValid => rows,
            TypedInsertColumnValidity::Bitmap(words) => {
                words.iter().try_fold(0_usize, |sum, word| {
                    sum.checked_add(word.count_ones() as usize)
                        .ok_or_else(|| generation_source_error("value-byte valid-row count"))
                })?
            }
        };
        if valid_rows > rows {
            return Err(generation_source_error("value-byte validity count"));
        }
        let bytes = match (values, column.ty) {
            (TypedInsertColumnValues::I32(_), SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
                valid_rows.checked_mul(std::mem::size_of::<i32>())
            }
            (TypedInsertColumnValues::I64(_), SqlType::Int8 | SqlType::Timestamp) => {
                valid_rows.checked_mul(std::mem::size_of::<i64>())
            }
            (TypedInsertColumnValues::I128(_), SqlType::Numeric { .. }) => {
                valid_rows.checked_mul(std::mem::size_of::<i128>())
            }
            (TypedInsertColumnValues::Bytes16(_), SqlType::Uuid) => {
                valid_rows.checked_mul(std::mem::size_of::<[u8; 16]>())
            }
            (TypedInsertColumnValues::BoolBits(_), SqlType::Bool) => Some(valid_rows),
            (TypedInsertColumnValues::Text { bytes, .. }, SqlType::Text) => Some(bytes.len()),
            _ => return Err(generation_source_error("value-byte SQL storage arm")),
        }
        .ok_or_else(|| generation_source_error("value-byte total"))?;
        total = total
            .checked_add(bytes)
            .ok_or_else(|| generation_source_error("value-byte total"))?;
    }
    Ok(total)
}

pub(super) fn with_logical_cell<R>(
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    row: usize,
    consume: impl FnOnce(bool, &[u8]) -> R,
) -> Result<R, EngineError> {
    if !validity.is_valid(row) {
        return Ok(consume(true, &[]));
    }
    match (values, ty) {
        (TypedInsertColumnValues::I32(values), SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            let value = values
                .get(row)
                .ok_or_else(|| generation_source_error("i32 row"))?
                .to_le_bytes();
            Ok(consume(false, &value))
        }
        (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => {
            let value = values
                .get(row)
                .ok_or_else(|| generation_source_error("i64 row"))?
                .to_le_bytes();
            Ok(consume(false, &value))
        }
        (TypedInsertColumnValues::I128(values), SqlType::Numeric { .. }) => {
            let value = values
                .get(row)
                .ok_or_else(|| generation_source_error("numeric row"))?
                .to_le_bytes();
            Ok(consume(false, &value))
        }
        (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => values
            .get(row)
            .map(|value| consume(false, value))
            .ok_or_else(|| generation_source_error("UUID row")),
        (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => {
            let word = words
                .get(row / 32)
                .ok_or_else(|| generation_source_error("BOOL row"))?;
            let value = [u8::from(word & (1_u32 << (row % 32)) != 0)];
            Ok(consume(false, &value))
        }
        (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
            let start = offsets
                .get(row)
                .and_then(|value| usize::try_from(*value).ok())
                .ok_or_else(|| generation_source_error("TEXT start"))?;
            let end = offsets
                .get(row + 1)
                .and_then(|value| usize::try_from(*value).ok())
                .ok_or_else(|| generation_source_error("TEXT end"))?;
            let value = bytes
                .get(start..end)
                .ok_or_else(|| generation_source_error("TEXT range"))?;
            Ok(consume(false, value))
        }
        _ => Err(generation_source_error("SQL storage arm")),
    }
}

fn generation_source_error(part: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT runtime generation source has invalid {part}"
    ))
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

    pub(crate) fn final_row_count(&self) -> u64 {
        u64::from_le_bytes(self.final_count_header)
    }

    pub(crate) fn text_layouts(&self) -> &[ResidentDeviceTextColumnLayout] {
        &self.text_layouts
    }

    pub(crate) fn bool_layouts(&self) -> &[ResidentDeviceBoolColumnLayout] {
        &self.bool_layouts
    }

    pub(crate) fn int4_stats(&self) -> &[ResidentDeviceInt4ColumnStats] {
        &self.int4_stats
    }

    pub(crate) fn null_layouts(&self) -> &[ResidentDeviceNullBitmapLayout] {
        &self.null_layouts
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
