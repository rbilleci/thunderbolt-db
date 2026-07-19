//! Scalar aggregate execution over GPU-filtered resident row indices.

use super::grouped_count_distinct::count_distinct_groups;
use crate::rel_exec_helpers::{average_sql_value, avg_numeric_sql_value, relational_column_index};
use crate::relational_model::{
    resident_device_int4_column_offset, resident_device_int8_column_offset,
    resident_device_numeric_column_offset, RelationalAccessPath, RelationalColumn,
    RelationalResidencySnapshot, RelationalSelectResult, RelationalTable,
};
use crate::ExecuteError;
use gpu_db_execution::{CudaResidentDeviceMemory, DeviceTarget};
use gpu_db_sql::{Decimal128, Select, SelectProjection, SqlType, SqlValue};
use gpu_db_types::EngineError;
use std::sync::Arc;

// Scalar aggregate? Compute it from the filtered indices and return a single row.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_scalar_aggregate(
    select: &Select,
    table: &RelationalTable,
    selected_columns: Vec<RelationalColumn>,
    access_path: RelationalAccessPath,
    snapshot: &RelationalResidencySnapshot,
    device_memory: &CudaResidentDeviceMemory,
    row_count: u64,
    indices: &[u32],
    indices_u64: &[u64],
) -> Result<RelationalSelectResult, ExecuteError> {
    // Capability is a property of the SQL shape, never of result cardinality. Validate the value
    // domain before the empty-set NULL fast path so an unsupported aggregate cannot appear
    // GPU-served merely because WHERE removed every row.
    match &select.projection {
        SelectProjection::Sum { column } => {
            let col_idx = relational_column_index(table, column)?;
            if !matches!(
                table.columns[col_idx].ty,
                SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. }
            ) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "SUM supports int4 / int8 / numeric columns on the Expr path".to_string(),
                )));
            }
        }
        SelectProjection::Avg { column } => {
            let col_idx = relational_column_index(table, column)?;
            if !matches!(
                table.columns[col_idx].ty,
                SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. }
            ) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "AVG supports int4 / int8 / numeric columns on the Expr path".to_string(),
                )));
            }
        }
        SelectProjection::Min { column } | SelectProjection::Max { column } => {
            let col_idx = relational_column_index(table, column)?;
            if !matches!(
                table.columns[col_idx].ty,
                SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. }
            ) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "MIN / MAX support int4 / int8 / numeric columns on the Expr path".to_string(),
                )));
            }
        }
        _ => {}
    }
    // PG: an aggregate over ZERO surviving rows -- SUM/AVG/MIN/MAX are SQL NULL (COUNT(*) and
    // COUNT(DISTINCT) are 0, handled in their arms below). NULL support is present now, so this
    // resolves the former "the engine cannot represent NULL yet (M3)" hard-error. The result
    // schema (selected_columns) carries the aggregate's column type, so the NULL is typed.
    if indices.is_empty()
        && matches!(
            select.projection,
            SelectProjection::Sum { .. }
                | SelectProjection::Avg { .. }
                | SelectProjection::Min { .. }
                | SelectProjection::Max { .. }
        )
    {
        return Ok(RelationalSelectResult {
            columns: Arc::new(selected_columns),
            rows: (vec![vec![SqlValue::Null]]).into(),
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path: Arc::new(access_path),
        });
    }
    let value = match &select.projection {
        // COUNT(*): the surviving row count IS the result (PG returns bigint). The GPU filter
        // + compaction already produced the count; no per-row materialization.
        SelectProjection::CountAll => SqlValue::Int8(indices.len() as i64),
        // SUM(int4): a GPU reduction over the filtered column (gather col[indices] + reduce);
        // PG returns bigint. (An empty filtered set => SQL NULL is handled above the match.)
        SelectProjection::Sum { column } => {
            let col_idx = relational_column_index(table, column)?;
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            // PG: SUM(int4) -> bigint; SUM(int8) -> numeric (the sum can exceed i64, so it
            // reduces into i128 via the two-atomic carry kernel).
            match table.columns[col_idx].ty {
                SqlType::Int4 => {
                    let byte_offset = resident_device_int4_column_offset(snapshot, table, col_idx)?;
                    let sum = device_memory
                        .sum_i32_at_indices_from_payload(byte_offset, indices)
                        .map_err(map_err)?;
                    SqlValue::Int8(sum)
                }
                SqlType::Int8 => {
                    let byte_offset = resident_device_int8_column_offset(snapshot, table, col_idx)?;
                    let sum = device_memory
                        .sum_i64_at_indices_i128_from_payload(byte_offset, indices)
                        .map_err(map_err)?;
                    SqlValue::Numeric(Decimal128::new(sum, 0))
                }
                // SUM(numeric) -> numeric at the column scale: sum the i128 mantissas (the
                // partials kernel with CHECKED i128 overflow -> numeric field overflow).
                SqlType::Numeric { scale, .. } => {
                    let byte_offset =
                        resident_device_numeric_column_offset(snapshot, table, col_idx)?;
                    let mantissa = device_memory
                        .sum_i128_at_indices_from_payload(byte_offset, indices)
                        .map_err(map_err)?;
                    SqlValue::Numeric(Decimal128::new(mantissa, scale))
                }
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "SUM supports int4 / int8 / numeric columns on the Expr path".to_string(),
                    )));
                }
            }
        }
        // MIN/MAX(int4): a GPU reduction; PG MIN/MAX preserve the column type (int4 -> int4).
        // (An empty filtered set => SQL NULL is handled above the match.)
        SelectProjection::Min { column } | SelectProjection::Max { column } => {
            let col_idx = relational_column_index(table, column)?;
            let is_max = matches!(select.projection, SelectProjection::Max { .. });
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            // MIN/MAX preserve the column type (PG): int4 -> int4 (i32 reduce), int8 -> int8
            // (i64 reduce). Other types are follow-ons.
            match table.columns[col_idx].ty {
                SqlType::Int4 => {
                    let byte_offset = resident_device_int4_column_offset(snapshot, table, col_idx)?;
                    let value = if is_max {
                        device_memory.max_i32_at_indices_from_payload(byte_offset, indices)
                    } else {
                        device_memory.min_i32_at_indices_from_payload(byte_offset, indices)
                    }
                    .map_err(map_err)?;
                    SqlValue::Int4(value)
                }
                SqlType::Int8 => {
                    let byte_offset = resident_device_int8_column_offset(snapshot, table, col_idx)?;
                    let value = if is_max {
                        device_memory.max_i64_at_indices_from_payload(byte_offset, indices)
                    } else {
                        device_memory.min_i64_at_indices_from_payload(byte_offset, indices)
                    }
                    .map_err(map_err)?;
                    SqlValue::Int8(value)
                }
                // MIN/MAX(numeric) -> numeric: reduce the i128 mantissas (no native 128-bit
                // atomic, so a partials + host-combine reduction); the result carries the
                // column scale.
                SqlType::Numeric { scale, .. } => {
                    let byte_offset =
                        resident_device_numeric_column_offset(snapshot, table, col_idx)?;
                    let mantissa = if is_max {
                        device_memory.max_i128_at_indices_from_payload(byte_offset, indices)
                    } else {
                        device_memory.min_i128_at_indices_from_payload(byte_offset, indices)
                    }
                    .map_err(map_err)?;
                    SqlValue::Numeric(Decimal128::new(mantissa, scale))
                }
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "MIN / MAX support int4 / int8 / numeric columns on the Expr path"
                            .to_string(),
                    )));
                }
            }
        }
        // AVG(int4) = the GPU SUM / the count, as numeric (PG). The reduction is on the GPU;
        // the final scalar divide reuses `average_sql_value` (scale-16, matching the enumerated
        // path). (An empty filtered set => SQL NULL is handled above the match.)
        SelectProjection::Avg { column } => {
            let col_idx = relational_column_index(table, column)?;
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            // AVG -> numeric = the GPU sum / the count. int4 reduces to i64 (widened), int8 to
            // i128 (the two-atomic carry); both divide via average_sql_value (integer sum).
            // numeric sums the i128 mantissas and divides via avg_numeric_sql_value, which
            // carries the column scale into PG's division-scale derivation.
            match table.columns[col_idx].ty {
                SqlType::Int4 => {
                    let byte_offset = resident_device_int4_column_offset(snapshot, table, col_idx)?;
                    let sum = i128::from(
                        device_memory
                            .sum_i32_at_indices_from_payload(byte_offset, indices)
                            .map_err(map_err)?,
                    );
                    average_sql_value(sum, indices.len())
                }
                SqlType::Int8 => {
                    let byte_offset = resident_device_int8_column_offset(snapshot, table, col_idx)?;
                    let sum = device_memory
                        .sum_i64_at_indices_i128_from_payload(byte_offset, indices)
                        .map_err(map_err)?;
                    average_sql_value(sum, indices.len())
                }
                SqlType::Numeric { scale, .. } => {
                    let byte_offset =
                        resident_device_numeric_column_offset(snapshot, table, col_idx)?;
                    let mantissa = device_memory
                        .sum_i128_at_indices_from_payload(byte_offset, indices)
                        .map_err(map_err)?;
                    avg_numeric_sql_value(mantissa, indices.len(), scale)
                }
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "AVG supports int4 / int8 / numeric columns on the Expr path".to_string(),
                    )));
                }
            }
        }
        // Scalar COUNT(DISTINCT v) (no GROUP BY) = ONE group: sort/mark/SUM over (g=0, v) via
        // `count_distinct_groups` with a constant group key, so the single group's count is the
        // total distinct. PG: COUNT(DISTINCT) over zero rows is 0 (NOT NULL), so an empty
        // filtered set returns 0 (the lone exception to the SUM/AVG empty-set NULL hard-error).
        SelectProjection::CountDistinct { column } => {
            let value_idx = relational_column_index(table, column)?;
            // M3 (doc 21): COUNT(DISTINCT v) counts distinct NON-NULL values (PG). A nullable
            // v is handled by the aggregate validity conjunct ANDed into the predicate above
            // (`v IS NOT NULL`), so the surviving `indices` here contain no NULL values — the
            // raw sort/mark/SUM pass over them is PG-exact. (The GROUPED COUNT(DISTINCT)
            // paths still guard nullable values with a clean error.)
            if indices_u64.is_empty() {
                SqlValue::Int8(0)
            } else {
                let g_vals = vec![0i64; indices_u64.len()];
                let groups = count_distinct_groups(
                    value_idx,
                    &g_vals,
                    indices_u64,
                    table,
                    snapshot,
                    device_memory,
                    row_count,
                )?;
                // The constant key yields exactly one group; its SUM = the total distinct.
                SqlValue::Int8(groups.first().map_or(0, |g| g.sum))
            }
        }
        _ => unreachable!("is_aggregate gates on CountAll | Sum | Min | Max | Avg | CountDistinct"),
    };
    Ok(RelationalSelectResult {
        columns: Arc::new(selected_columns),
        rows: (vec![vec![value]]).into(),
        planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
        executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
        fallback_reason: None,
        access_path: Arc::new(access_path),
    })
}
