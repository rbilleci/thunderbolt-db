//! Borrow-only physical source for pre-WAL row-local device operators.
//!
//! This leaf is intentionally below the sealed typed batch rather than the mutation owner.  It
//! serializes catalog-order vectors to the established resident payload ABI, but never converts
//! them to logical rows, retains them, or decides a constraint verdict.

use super::*;
use gpu_db_execution::{CudaCompoundFoldColumn, CudaResidentDeviceMemory};
#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static CONSTRAINT_SOURCE_UPLOAD_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_constraint_source_upload_count() {
    CONSTRAINT_SOURCE_UPLOAD_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn constraint_source_upload_count() -> usize {
    CONSTRAINT_SOURCE_UPLOAD_COUNT.with(Cell::get)
}

/// Geometry-only lookup for one catalog column in the common pre-WAL payload.  The key proof
/// receives this only after the batch/table binding is already sealed; it contains no values and
/// cannot make a host-side relational decision.
struct ConstraintSourceKeyColumnLayout {
    column_id: u32,
    name: String,
    data: CudaCompoundFoldColumn,
    validity_bitmap_byte_offset: Option<u64>,
}

/// A short-lived device operator input derived from borrowed sealed vectors.  Its descriptor and
/// memory disappear before the off-lock plan is queued; neither is semantic authority.
pub(crate) struct TypedInsertConstraintDeviceSource {
    descriptor: RelationalResidencySnapshot,
    memory: CudaResidentDeviceMemory,
    payload_bytes: u64,
    key_column_layouts: Box<[ConstraintSourceKeyColumnLayout]>,
}

impl TypedInsertConstraintDeviceSource {
    pub(crate) fn descriptor(&self) -> &RelationalResidencySnapshot {
        &self.descriptor
    }

    pub(crate) fn memory(&self) -> &CudaResidentDeviceMemory {
        &self.memory
    }

    pub(crate) fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Resolve one key descriptor from the single uploaded payload. The caller keeps key order
    /// and repeated occurrences; lookup is by sealed column id and fail-closed name confirmation.
    pub(crate) fn key_column_layout(
        &self,
        column_id: u32,
        name: &str,
    ) -> Result<(CudaCompoundFoldColumn, Option<u64>), EngineError> {
        let layout = self
            .key_column_layouts
            .iter()
            .find(|layout| layout.column_id == column_id)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "typed constraint source is missing column id {column_id}"
                ))
            })?;
        if layout.name != name {
            return Err(EngineError::ApplyFailed(format!(
                "typed constraint source column id {column_id} name drifted from \"{name}\""
            )));
        }
        Ok((layout.data, layout.validity_bitmap_byte_offset))
    }
}

pub(super) fn build(
    batch: &TypedInsertBatch,
    engine: &Engine,
    table: &RelationalTable,
) -> Result<TypedInsertConstraintDeviceSource, ExecuteError> {
    let rows = usize::try_from(batch.row_count).expect("u32 rows fit usize on supported hosts");
    if rows == 0
        || table.columns.len() != batch.columns.len()
        || !table
            .columns
            .iter()
            .zip(&batch.columns)
            .all(|(live, column)| {
                live.id == column.column_id
                    && live.attnum == column.attnum
                    && live.ty == column.ty
                    && column.values.rows_match(rows)
                    && column.validity.shape_is_exact(rows)
                    && column.values.text_extent_matches(rows)
            })
    {
        return Err(source_error(
            "sealed typed CHECK source lost vector geometry",
        ));
    }

    let expected_payload_bytes = payload_bytes(batch)?;
    let expected_payload_len = usize::try_from(expected_payload_bytes)
        .map_err(|_| source_error("typed CHECK payload length exceeds host address space"))?;
    let mut payload = Vec::with_capacity(expected_payload_len);
    payload.resize(std::mem::size_of::<u64>(), 0);
    let mut text_layouts = Vec::new();
    let mut bool_layouts = Vec::new();
    let mut int4_stats = Vec::new();
    let mut null_layouts = Vec::new();
    let mut key_column_layouts = Vec::with_capacity(table.columns.len());
    let mut validity_offsets = Vec::new();

    // The resident VM indexes these sections by catalog order and type.  This writes values only
    // as payload bytes; it never scans them for a host predicate or verdict.  Int4 stats are
    // intentionally conservative so a source cannot make a host-derived semantic decision.
    for (column, live) in batch.columns.iter().zip(&table.columns) {
        if !matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date) {
            continue;
        }
        let TypedInsertColumnValues::I32(values) = &column.values else {
            return Err(source_error(
                "typed CHECK int4 vector arm disagrees with catalog",
            ));
        };
        let byte_offset = u64::try_from(payload.len())
            .map_err(|_| source_error("typed CHECK int4 payload offset overflows"))?;
        for value in values.iter() {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        key_column_layouts.push(ConstraintSourceKeyColumnLayout {
            column_id: column.column_id,
            name: live.name.clone(),
            data: CudaCompoundFoldColumn::Fixed {
                byte_offset,
                width_words: 1,
            },
            validity_bitmap_byte_offset: None,
        });
        int4_stats.push(ResidentDeviceInt4ColumnStats {
            name: live.name.clone(),
            min: i32::MIN,
            max: i32::MAX,
        });
    }
    for (column, live) in batch.columns.iter().zip(&table.columns) {
        if !matches!(column.ty, SqlType::Int8 | SqlType::Timestamp) {
            continue;
        }
        let TypedInsertColumnValues::I64(values) = &column.values else {
            return Err(source_error(
                "typed CHECK int8 vector arm disagrees with catalog",
            ));
        };
        let byte_offset = u64::try_from(payload.len())
            .map_err(|_| source_error("typed CHECK int8 payload offset overflows"))?;
        for value in values.iter() {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        key_column_layouts.push(ConstraintSourceKeyColumnLayout {
            column_id: column.column_id,
            name: live.name.clone(),
            data: CudaCompoundFoldColumn::Fixed {
                byte_offset,
                width_words: 2,
            },
            validity_bitmap_byte_offset: None,
        });
    }
    for (column, live) in batch.columns.iter().zip(&table.columns) {
        match (&column.ty, &column.values) {
            (SqlType::Numeric { .. }, TypedInsertColumnValues::I128(values)) => {
                let byte_offset = u64::try_from(payload.len())
                    .map_err(|_| source_error("typed CHECK numeric payload offset overflows"))?;
                for value in values.iter() {
                    payload.extend_from_slice(&value.to_le_bytes());
                }
                key_column_layouts.push(ConstraintSourceKeyColumnLayout {
                    column_id: column.column_id,
                    name: live.name.clone(),
                    data: CudaCompoundFoldColumn::Fixed {
                        byte_offset,
                        width_words: 4,
                    },
                    validity_bitmap_byte_offset: None,
                });
            }
            (SqlType::Uuid, TypedInsertColumnValues::Bytes16(values)) => {
                let byte_offset = u64::try_from(payload.len())
                    .map_err(|_| source_error("typed CHECK UUID payload offset overflows"))?;
                for value in values.iter() {
                    payload.extend_from_slice(value);
                }
                key_column_layouts.push(ConstraintSourceKeyColumnLayout {
                    column_id: column.column_id,
                    name: live.name.clone(),
                    data: CudaCompoundFoldColumn::Fixed {
                        byte_offset,
                        width_words: 4,
                    },
                    validity_bitmap_byte_offset: None,
                });
            }
            (SqlType::Numeric { .. } | SqlType::Uuid, _) => {
                return Err(source_error(
                    "typed CHECK wide vector arm disagrees with catalog",
                ));
            }
            _ => {}
        }
    }
    for (column, live) in batch.columns.iter().zip(&table.columns) {
        if column.ty != SqlType::Bool {
            continue;
        }
        let TypedInsertColumnValues::BoolBits(words) = &column.values else {
            return Err(source_error(
                "typed CHECK bool vector arm disagrees with catalog",
            ));
        };
        if !bitmap_shape_is_exact(words, rows) {
            return Err(source_error("typed CHECK bool bitmap extent is invalid"));
        }
        let bitmap_byte_offset = u64::try_from(payload.len())
            .map_err(|_| source_error("typed CHECK bool payload offset overflows"))?;
        for word in words.iter() {
            payload.extend_from_slice(&word.to_le_bytes());
        }
        bool_layouts.push(ResidentDeviceBoolColumnLayout {
            name: live.name.clone(),
            bitmap_byte_offset,
        });
        key_column_layouts.push(ConstraintSourceKeyColumnLayout {
            column_id: column.column_id,
            name: live.name.clone(),
            data: CudaCompoundFoldColumn::Bool { bitmap_byte_offset },
            validity_bitmap_byte_offset: None,
        });
    }
    for (column, live) in batch.columns.iter().zip(&table.columns) {
        let Some(words) = column.validity.bitmap_words() else {
            continue;
        };
        if !bitmap_shape_is_exact(words, rows) {
            return Err(source_error(
                "typed CHECK validity bitmap extent is invalid",
            ));
        }
        let bitmap_byte_offset = u64::try_from(payload.len())
            .map_err(|_| source_error("typed CHECK validity payload offset overflows"))?;
        for word in words.iter() {
            payload.extend_from_slice(&word.to_le_bytes());
        }
        null_layouts.push(ResidentDeviceNullBitmapLayout {
            name: live.name.clone(),
            bitmap_byte_offset,
        });
        validity_offsets.push((column.column_id, bitmap_byte_offset));
    }
    for (column, live) in batch.columns.iter().zip(&table.columns) {
        if column.ty != SqlType::Text {
            continue;
        }
        let TypedInsertColumnValues::Text { offsets, bytes } = &column.values else {
            return Err(source_error(
                "typed CHECK text vector arm disagrees with catalog",
            ));
        };
        while !payload.len().is_multiple_of(8) {
            payload.push(0);
        }
        let offsets_byte_offset = u64::try_from(payload.len())
            .map_err(|_| source_error("typed CHECK text offset position overflows"))?;
        for offset in offsets.iter() {
            payload.extend_from_slice(&offset.to_le_bytes());
        }
        let bytes_byte_offset = u64::try_from(payload.len())
            .map_err(|_| source_error("typed CHECK text byte position overflows"))?;
        payload.extend_from_slice(bytes);
        text_layouts.push(ResidentDeviceTextColumnLayout {
            name: live.name.clone(),
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len: u64::try_from(bytes.len())
                .map_err(|_| source_error("typed CHECK text byte length overflows"))?,
        });
        key_column_layouts.push(ConstraintSourceKeyColumnLayout {
            column_id: column.column_id,
            name: live.name.clone(),
            data: CudaCompoundFoldColumn::Text {
                offsets_byte_offset,
                bytes_byte_offset,
                bytes_len: u64::try_from(bytes.len())
                    .map_err(|_| source_error("typed CHECK text byte length overflows"))?,
            },
            validity_bitmap_byte_offset: None,
        });
    }
    for (column_id, bitmap_byte_offset) in validity_offsets {
        let layout = key_column_layouts
            .iter_mut()
            .find(|layout| layout.column_id == column_id)
            .ok_or_else(|| source_error("typed CHECK validity column lacks data layout"))?;
        layout.validity_bitmap_byte_offset = Some(bitmap_byte_offset);
    }
    let row_header = u64::try_from(rows)
        .map_err(|_| source_error("typed CHECK row count overflows"))?
        .to_le_bytes();
    payload[..row_header.len()].copy_from_slice(&row_header);
    let payload_bytes = u64::try_from(payload.len())
        .map_err(|_| source_error("typed CHECK payload length overflows"))?;
    if payload_bytes != expected_payload_bytes {
        return Err(source_error(
            "typed CHECK payload geometry changed during encoding",
        ));
    }
    let gpu_id = engine.planner.default_gpu_id();
    #[cfg(test)]
    CONSTRAINT_SOURCE_UPLOAD_COUNT.with(|count| count.set(count.get().saturating_add(1)));
    let memory = engine
        .cuda_driver_probe_runtime()
        .retain_device_memory_copy_scoped(gpu_id, &payload)
        .map_err(|error| source_error(&format!("typed CHECK source upload failed: {error}")))?;
    let descriptor = RelationalResidencySnapshot {
        gpu_id,
        schema: table.schema.clone(),
        table: table.name.clone(),
        generation: 1,
        row_count: rows,
        capacity: rows,
        column_count: table.columns.len(),
        resident_bytes: payload_bytes,
        resident_device_int4_columns: table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
            .map(|column| column.name.clone())
            .collect(),
        resident_device_int4_column_stats: int4_stats,
        resident_device_int8_columns: table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect(),
        resident_device_numeric_columns: table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect(),
        resident_device_bool_columns: bool_layouts,
        resident_device_text_columns: text_layouts,
        resident_device_null_columns: null_layouts,
        valid_through_index: engine.committed_seq(),
        invalidated_by_txn_id: None,
        invalidated_at_index: None,
        invalidated_by_memory_pressure: false,
        memory_pressure_active: false,
        last_refresh_cost: None,
        admission_budget_bytes: None,
        resident_bytes_after_admission: 0,
        evicted_tables_on_admission: Vec::new(),
        device_memory_proof: Some(memory.metadata().clone()),
    };
    Ok(TypedInsertConstraintDeviceSource {
        descriptor,
        memory,
        payload_bytes,
        key_column_layouts: key_column_layouts.into(),
    })
}

/// Exact byte geometry for the resident payload ABI, computed from lengths and catalog type only.
/// No value is inspected, compared, or used to make a relational decision here.
pub(super) fn payload_bytes(batch: &TypedInsertBatch) -> Result<u64, EngineError> {
    let mut bytes = std::mem::size_of::<u64>();
    for column in batch
        .columns
        .iter()
        .filter(|column| matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
    {
        let TypedInsertColumnValues::I32(values) = &column.values else {
            return Err(EngineError::ApplyFailed(
                "typed CHECK int4 vector arm disagrees with catalog".to_string(),
            ));
        };
        bytes = bytes
            .checked_add(values.len().checked_mul(4).ok_or_else(|| {
                EngineError::ApplyFailed("typed CHECK payload size overflows".to_string())
            })?)
            .ok_or_else(|| {
                EngineError::ApplyFailed("typed CHECK payload size overflows".to_string())
            })?;
    }
    for column in batch
        .columns
        .iter()
        .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
    {
        let TypedInsertColumnValues::I64(values) = &column.values else {
            return Err(EngineError::ApplyFailed(
                "typed CHECK int8 vector arm disagrees with catalog".to_string(),
            ));
        };
        bytes = bytes
            .checked_add(values.len().checked_mul(8).ok_or_else(|| {
                EngineError::ApplyFailed("typed CHECK payload size overflows".to_string())
            })?)
            .ok_or_else(|| {
                EngineError::ApplyFailed("typed CHECK payload size overflows".to_string())
            })?;
    }
    for column in batch
        .columns
        .iter()
        .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
    {
        let value_bytes = match &column.values {
            TypedInsertColumnValues::I128(values)
                if matches!(column.ty, SqlType::Numeric { .. }) =>
            {
                values.len().checked_mul(16)
            }
            TypedInsertColumnValues::Bytes16(values) if column.ty == SqlType::Uuid => {
                values.len().checked_mul(16)
            }
            _ => None,
        }
        .ok_or_else(|| {
            EngineError::ApplyFailed(
                "typed CHECK wide vector arm disagrees with catalog".to_string(),
            )
        })?;
        bytes = bytes.checked_add(value_bytes).ok_or_else(|| {
            EngineError::ApplyFailed("typed CHECK payload size overflows".to_string())
        })?;
    }
    for column in batch
        .columns
        .iter()
        .filter(|column| column.ty == SqlType::Bool)
    {
        let TypedInsertColumnValues::BoolBits(words) = &column.values else {
            return Err(EngineError::ApplyFailed(
                "typed CHECK bool vector arm disagrees with catalog".to_string(),
            ));
        };
        bytes = bytes
            .checked_add(words.len().checked_mul(4).ok_or_else(|| {
                EngineError::ApplyFailed("typed CHECK payload size overflows".to_string())
            })?)
            .ok_or_else(|| {
                EngineError::ApplyFailed("typed CHECK payload size overflows".to_string())
            })?;
    }
    for column in &batch.columns {
        if let TypedInsertColumnValidity::Bitmap(words) = &column.validity {
            let validity_bytes = words.len().checked_mul(4).ok_or_else(|| {
                EngineError::ApplyFailed("typed CHECK validity payload size overflows".to_string())
            })?;
            bytes = bytes.checked_add(validity_bytes).ok_or_else(|| {
                EngineError::ApplyFailed("typed CHECK validity payload size overflows".to_string())
            })?;
        }
    }
    for column in batch
        .columns
        .iter()
        .filter(|column| column.ty == SqlType::Text)
    {
        let TypedInsertColumnValues::Text {
            offsets,
            bytes: text,
        } = &column.values
        else {
            return Err(EngineError::ApplyFailed(
                "typed CHECK text vector arm disagrees with catalog".to_string(),
            ));
        };
        bytes = bytes.checked_next_multiple_of(8).ok_or_else(|| {
            EngineError::ApplyFailed("typed CHECK text alignment overflows".to_string())
        })?;
        let text_bytes = offsets
            .len()
            .checked_mul(8)
            .and_then(|offset_bytes| offset_bytes.checked_add(text.len()))
            .ok_or_else(|| {
                EngineError::ApplyFailed("typed CHECK text payload size overflows".to_string())
            })?;
        bytes = bytes.checked_add(text_bytes).ok_or_else(|| {
            EngineError::ApplyFailed("typed CHECK text payload size overflows".to_string())
        })?;
    }
    u64::try_from(bytes)
        .map_err(|_| EngineError::ApplyFailed("typed CHECK payload length overflows".to_string()))
}

fn source_error(message: &str) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_source_payload_keeps_nullable_bitmap_tails_for_31_32_and_33_rows() {
        for rows in [31_usize, 32, 33] {
            let engine = Engine::new_local_test_engine();
            engine
                .execute_text(
                    1,
                    "CREATE TABLE check_tail (id int4, note text, \
                     CONSTRAINT check_tail_id_positive CHECK (id > 0))",
                )
                .unwrap();
            let insert = crate::Insert {
                table: "check_tail".to_string(),
                columns: Vec::new(),
                rows: gpu_db_sql::Insert::programmatic_rows(
                    (0..rows)
                        .map(|row| {
                            vec![
                                SqlValue::Int4(i32::try_from(row + 1).unwrap()),
                                if row + 1 == rows {
                                    SqlValue::Null
                                } else {
                                    SqlValue::Text(format!("note-{row}"))
                                },
                            ]
                        })
                        .collect(),
                ),
                returning: Vec::new(),
            };
            let catalog = engine.catalog_snapshot();
            let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch(
                &crate::Command::Insert(insert),
                &catalog,
                catalog.commit_seq,
                None,
            )
            .unwrap()
            .expect("CHECK-only typed batch remains eligible");
            let TypedInsertColumnValidity::Bitmap(validity) = &batch.columns[1].validity else {
                panic!("last NULL must use a typed validity bitmap");
            };
            assert!(bitmap_shape_is_exact(validity, rows));
            assert_eq!(validity.len(), rows.div_ceil(32));
            assert!(batch.row_local_constraint_device_payload_bytes().unwrap() >= 8);
        }
    }
}
