//! Terminal typed and nullable projection from resident GPU payloads into result rows.

use crate::relational_model::{
    resident_device_bool_column_offset, resident_device_int4_column_offset,
    resident_device_int8_column_offset, resident_device_null_column_offset,
    resident_device_numeric_column_offset, resident_device_text_column_layout,
    RelationalAccessPath, RelationalResidencySnapshot, RelationalSelectResult, RelationalTable,
    RowBlock,
};
use crate::resident_route::BoundRelationalSelect;
use crate::ExecuteError;
use gpu_db_execution::{CudaResidentDeviceMemory, DeviceTarget};
use gpu_db_sql::{Decimal128, SqlType, SqlValue};
use gpu_db_types::EngineError;
use std::sync::Arc;

pub(super) fn materialize_projected_rows(
    table: &RelationalTable,
    bound: BoundRelationalSelect,
    access_path: RelationalAccessPath,
    snapshot: &RelationalResidencySnapshot,
    device_memory: &CudaResidentDeviceMemory,
    row_count: u64,
    indices_u64: Vec<u64>,
) -> Result<RelationalSelectResult, ExecuteError> {
    // Materialize: gather each projected column at the surviving row indices on the GPU, by type
    // (int4 -> i32 gather, int8 -> i64 gather; the type matrix, doc 19).
    enum ProjectedColumn {
        Int4(Vec<i32>),
        Int8(Vec<i64>),
        Numeric(Vec<i128>, u8),
        Date(Vec<i32>),
        Timestamp(Vec<i64>),
        // Stored as the i128 the i128 projector returns; the raw 16 uuid bytes are its LE form.
        Uuid(Vec<i128>),
        // Stored as the widened i32s the int4 projector returns; narrowed back to i16 per row.
        Int2(Vec<i32>),
        // Gathered straight from the 1-bit-per-row bool bitmap.
        Bool(Vec<bool>),
        // The text VALUE is gathered ON-DEVICE from the resident payload's offsets+bytes sections
        // (project_text_rows_from_payload) at the GPU-sorted surviving indices -- no host_rows read
        // (the host is control plane only). A NULL/empty cell returns ""; the validity override below
        // restores SqlValue::Null for a NULL row.
        Text(Vec<String>),
    }
    let mut projected_columns: Vec<ProjectedColumn> =
        Vec::with_capacity(bound.selected_indexes.len());
    for &col in &bound.selected_indexes {
        let column = match table.columns[col].ty {
            SqlType::Int8 => {
                let byte_offset = resident_device_int8_column_offset(snapshot, table, col)?;
                let values = device_memory
                    .project_i64_rows_from_payload(byte_offset, &indices_u64)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                ProjectedColumn::Int8(values)
            }
            SqlType::Numeric { scale, .. } => {
                let byte_offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                let values = device_memory
                    .project_i128_rows_from_payload(byte_offset, &indices_u64)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                ProjectedColumn::Numeric(values, scale)
            }
            SqlType::Date => {
                // Date rides the i32 section; project it as i32 then tag it as a date.
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                let values = device_memory
                    .project_i32_rows_from_payload(byte_offset, &indices_u64)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                ProjectedColumn::Date(values)
            }
            SqlType::Int2 => {
                // Smallint rides the i32 section widened; project as i32, narrow per row below.
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                let values = device_memory
                    .project_i32_rows_from_payload(byte_offset, &indices_u64)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                ProjectedColumn::Int2(values)
            }
            SqlType::Bool => {
                // Bool is a 1-bit-per-row bitmap; gather the selected rows' bits.
                let byte_offset = resident_device_bool_column_offset(snapshot, table, col)?;
                let values = device_memory
                    .project_bool_rows_from_payload(byte_offset, &indices_u64)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                ProjectedColumn::Bool(values)
            }
            SqlType::Timestamp => {
                // Timestamp rides the i64 section; project it as i64 then tag it as a timestamp.
                let byte_offset = resident_device_int8_column_offset(snapshot, table, col)?;
                let values = device_memory
                    .project_i64_rows_from_payload(byte_offset, &indices_u64)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                ProjectedColumn::Timestamp(values)
            }
            SqlType::Uuid => {
                // Uuid rides the i128 section; project as i128 (its LE bytes are the raw uuid).
                let byte_offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                let values = device_memory
                    .project_i128_rows_from_payload(byte_offset, &indices_u64)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                ProjectedColumn::Uuid(values)
            }
            SqlType::Text => {
                // Gather the text VALUE ON-DEVICE from the resident payload's offsets+bytes sections at
                // the GPU-sorted surviving indices -- no host_rows read (the host is control plane
                // only; the GPU did the filter + sort AND now materializes the strings). A NULL/empty
                // cell returns ""; the projected_validity override below restores SqlValue::Null.
                let layout = resident_device_text_column_layout(snapshot, table, col)?;
                let values = device_memory
                    .project_text_rows_from_payload(
                        layout.offsets_byte_offset,
                        layout.bytes_byte_offset,
                        layout.bytes_len,
                        row_count,
                        &indices_u64,
                    )
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                ProjectedColumn::Text(values)
            }
            _ => {
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                let values = device_memory
                    .project_i32_rows_from_payload(byte_offset, &indices_u64)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                ProjectedColumn::Int4(values)
            }
        };
        projected_columns.push(column);
    }
    // M3 (doc 21): per projected column, the NULL validity of each surviving row (1 = present). A
    // nullable column's device value is a 0/empty PLACEHOLDER for a NULL row, so the result must emit
    // SqlValue::Null there. The validity bitmap is 1-bit-per-row like a bool column, so the bool
    // projector gathers it at the surviving indices. `None` = the column holds no NULLs (all valid),
    // so non-nullable projections are unchanged. (Text's device gather returns "" for a NULL cell, so
    // this override is what restores its SqlValue::Null.)
    let projected_validity: Vec<Option<Vec<bool>>> = bound
        .selected_indexes
        .iter()
        .map(|&col| -> Result<Option<Vec<bool>>, ExecuteError> {
            match resident_device_null_column_offset(snapshot, table, col)? {
                Some(off) => Ok(Some(
                    device_memory
                        .project_bool_rows_from_payload(off, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?,
                )),
                None => Ok(None),
            }
        })
        .collect::<Result<_, _>>()?;
    // Build the result rows FLAT, directly into the RowBlock (DECISIONS read-path lever): the dominant
    // general-route cost was the per-row `Vec<SqlValue>` boxing PLUS the `rows.into()` RowBlock
    // conversion (a 2nd O(n) pass) -- together ~75% of the per-row time (measured 38+39 of 102 ns/row).
    // Emit ONE row-major `Vec<SqlValue>` (no inner Vec, no conversion). Byte-identical: same values in
    // the same row-major order `RowBlock::from(Vec<Vec<_>>)` produced.
    let ncols = projected_columns.len();
    let mut flat: Vec<SqlValue> = Vec::with_capacity(indices_u64.len() * ncols);
    for row in 0..indices_u64.len() {
        for (c, column) in projected_columns.iter().enumerate() {
            // A NULL row (validity bit 0) projects as SQL NULL regardless of its placeholder.
            if projected_validity[c]
                .as_ref()
                .is_some_and(|validity| !validity[row])
            {
                flat.push(SqlValue::Null);
                continue;
            }
            flat.push(match column {
                ProjectedColumn::Int4(values) => SqlValue::Int4(values[row]),
                ProjectedColumn::Int8(values) => SqlValue::Int8(values[row]),
                ProjectedColumn::Numeric(values, scale) => {
                    SqlValue::Numeric(Decimal128::new(values[row], *scale))
                }
                ProjectedColumn::Date(values) => SqlValue::Date(values[row]),
                ProjectedColumn::Timestamp(values) => SqlValue::Timestamp(values[row]),
                ProjectedColumn::Uuid(values) => SqlValue::Uuid(values[row].to_le_bytes()),
                // The stored i32 is a widened i16, so the narrowing is exact.
                ProjectedColumn::Int2(values) => SqlValue::Int2(values[row] as i16),
                ProjectedColumn::Bool(values) => SqlValue::Bool(values[row]),
                // Device-gathered String; a NULL row was already handled by the validity override
                // above, so a bare value here is a real (possibly empty) string.
                ProjectedColumn::Text(values) => SqlValue::Text(values[row].clone()),
            });
        }
    }
    // OFFSET/LIMIT was already applied as a control-plane window of `indices_u64` above (before the
    // gather), so `flat` is the final windowed result -- no host drain/truncate on result data.
    Ok(RelationalSelectResult {
        columns: Arc::new(bound.selected_columns),
        rows: RowBlock::flat(flat, ncols),
        planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
        executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
        fallback_reason: None,
        access_path: Arc::new(access_path),
    })
}
