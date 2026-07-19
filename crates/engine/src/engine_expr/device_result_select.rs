//! Device-resident terminal pipeline for ordinary (non-grouped, non-aggregate) SELECT results.

use super::execution_source::ResidentVisibility;
use super::predicate_compiler::{
    compile_arith_program, predicate_references_nullable_column, push_leaf_validity_and,
};
use super::ResidentExpr;
use crate::rel_exec_helpers::relational_column_index;
use crate::relational_model::{
    resident_device_bool_column_offset, resident_device_int4_column_offset,
    resident_device_int8_column_offset, resident_device_null_column_offset,
    resident_device_numeric_column_offset, resident_device_text_column_layout,
    RelationalAccessPath, RelationalResidencySnapshot, RelationalSelectResult, RelationalTable,
};
use crate::resident_route::BoundRelationalSelect;
use crate::{Engine, ExecuteError};
use gpu_db_execution::{
    CudaJoinOrderKey, CudaJoinPayloadKey, CudaJoinSortKey, CudaMaterializeJoinColumn,
    CudaResidentDeviceMemory, DeviceTarget, ExprStep, ResidentElemType,
};
use gpu_db_sql::{Select, SqlType};
use gpu_db_types::EngineError;
use std::sync::Arc;

fn result_error(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message.into()))
}

fn order_arithmetic_elem(
    expression: &ResidentExpr,
    table: &RelationalTable,
) -> Result<ResidentElemType, ExecuteError> {
    fn visit(
        expression: &ResidentExpr,
        table: &RelationalTable,
        i32_column: &mut bool,
        i64_column: &mut bool,
    ) -> Result<(), ExecuteError> {
        match expression {
            ResidentExpr::Column(index) => {
                match table.columns.get(*index).map(|column| column.ty) {
                    Some(SqlType::Int2) => {
                        return Err(result_error(
                            "ORDER BY smallint arithmetic is not supported by the device VM's checked int4 domain",
                        ));
                    }
                    Some(SqlType::Int4) => *i32_column = true,
                    Some(SqlType::Int8) => *i64_column = true,
                    Some(SqlType::Timestamp) => {
                        return Err(result_error(
                            "ORDER BY timestamp arithmetic is not supported on the device",
                        ));
                    }
                    _ => {
                        return Err(result_error(
                            "ORDER BY arithmetic requires int2, int4, or int8 operands",
                        ));
                    }
                }
            }
            ResidentExpr::Binary { lhs, rhs, .. } => {
                visit(lhs, table, i32_column, i64_column)?;
                visit(rhs, table, i32_column, i64_column)?;
            }
            ResidentExpr::Int4Literal(_) | ResidentExpr::Int8Literal(_) => {}
            ResidentExpr::IsNull { .. }
            | ResidentExpr::NumericLiteral(_)
            | ResidentExpr::TextLiteral(_)
            | ResidentExpr::BoolLiteral(_) => {
                return Err(result_error(
                    "ORDER BY expression is not supported by the integer arithmetic device VM",
                ));
            }
        }
        Ok(())
    }

    let mut i32_column = false;
    let mut i64_column = false;
    visit(expression, table, &mut i32_column, &mut i64_column)?;
    if i32_column && i64_column {
        return Err(result_error(
            "ORDER BY a mixed int2/int4/int8 arithmetic expression is not supported by the mono-typed device VM",
        ));
    }
    Ok(if i64_column {
        ResidentElemType::I64
    } else {
        ResidentElemType::I32
    })
}

fn payload_key<'a>(
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    memory: &'a CudaResidentDeviceMemory,
    column: usize,
) -> Result<CudaJoinPayloadKey<'a>, ExecuteError> {
    let validity_bitmap_offset = resident_device_null_column_offset(snapshot, table, column)?;
    Ok(match table.columns[column].ty {
        SqlType::Text => {
            let layout = resident_device_text_column_layout(snapshot, table, column)?;
            CudaJoinPayloadKey {
                payload: memory,
                byte_offset: layout.offsets_byte_offset,
                validity_bitmap_offset,
                width: 255,
                text_bytes_byte_offset: Some(layout.bytes_byte_offset),
                text_bytes_len: layout.bytes_len,
            }
        }
        SqlType::Numeric { .. } | SqlType::Uuid => CudaJoinPayloadKey {
            payload: memory,
            byte_offset: resident_device_numeric_column_offset(snapshot, table, column)?,
            validity_bitmap_offset,
            width: 16,
            text_bytes_byte_offset: None,
            text_bytes_len: 0,
        },
        SqlType::Int8 | SqlType::Timestamp => CudaJoinPayloadKey {
            payload: memory,
            byte_offset: resident_device_int8_column_offset(snapshot, table, column)?,
            validity_bitmap_offset,
            width: 8,
            text_bytes_byte_offset: None,
            text_bytes_len: 0,
        },
        SqlType::Int2 | SqlType::Int4 | SqlType::Date => CudaJoinPayloadKey {
            payload: memory,
            byte_offset: resident_device_int4_column_offset(snapshot, table, column)?,
            validity_bitmap_offset,
            width: 4,
            text_bytes_byte_offset: None,
            text_bytes_len: 0,
        },
        SqlType::Bool => {
            return Err(result_error(
                "ORDER BY a bool column requires a device fixed-width derived key",
            ));
        }
    })
}

fn projection_spec<'a>(
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    memory: &'a CudaResidentDeviceMemory,
    column: usize,
) -> Result<CudaMaterializeJoinColumn<'a>, ExecuteError> {
    let validity_bitmap_offset = resident_device_null_column_offset(snapshot, table, column)?;
    Ok(match table.columns[column].ty {
        SqlType::Text => {
            let layout = resident_device_text_column_layout(snapshot, table, column)?;
            CudaMaterializeJoinColumn::Text {
                relation: 0,
                payload: memory,
                offsets_byte_offset: layout.offsets_byte_offset,
                bytes_byte_offset: layout.bytes_byte_offset,
                bytes_len: layout.bytes_len,
                validity_bitmap_offset,
            }
        }
        SqlType::Bool => CudaMaterializeJoinColumn::Bool {
            relation: 0,
            payload: memory,
            bitmap_byte_offset: resident_device_bool_column_offset(snapshot, table, column)?,
            validity_bitmap_offset,
        },
        ty => {
            let (byte_offset, width) = match ty {
                SqlType::Int8 | SqlType::Timestamp => (
                    resident_device_int8_column_offset(snapshot, table, column)?,
                    8,
                ),
                SqlType::Numeric { .. } | SqlType::Uuid => (
                    resident_device_numeric_column_offset(snapshot, table, column)?,
                    16,
                ),
                SqlType::Int2 | SqlType::Int4 | SqlType::Date => (
                    resident_device_int4_column_offset(snapshot, table, column)?,
                    4,
                ),
                SqlType::Text | SqlType::Bool => unreachable!(),
            };
            CudaMaterializeJoinColumn::Fixed {
                relation: 0,
                payload: memory,
                byte_offset,
                validity_bitmap_offset,
                width,
            }
        }
    })
}

enum PreparedOrderKey<'a> {
    Resident(CudaJoinOrderKey<'a>),
    Derived {
        program: Vec<ExprStep>,
        validity_program: Option<Vec<ExprStep>>,
        elem: ResidentElemType,
        width: u8,
        descending: bool,
        nulls_first: bool,
    },
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_device_result_select(
    engine: &Engine,
    select: &Select,
    table: &RelationalTable,
    bound: BoundRelationalSelect,
    access_path: RelationalAccessPath,
    snapshot: &RelationalResidencySnapshot,
    memory: &CudaResidentDeviceMemory,
    row_count: u64,
    predicate: Option<&ResidentExpr>,
    visibility: Option<ResidentVisibility>,
    order_by_exprs: &[Option<ResidentExpr>],
    order_by_nulls_first: &[Option<bool>],
) -> Result<RelationalSelectResult, ExecuteError> {
    let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| result_error(err.to_string());
    let row_count = u32::try_from(row_count)
        .map_err(|_| result_error("resident SELECT row count exceeds device coordinates"))?;
    let mask = engine.resident_predicate_device_mask(
        predicate, table, snapshot, memory, row_count, visibility,
    )?;
    let mut coordinates = memory
        .identity_join_coordinates(row_count, mask.as_ref())
        .map_err(map_err)?;

    if !select.order_by.is_empty() {
        // Classify and validate every ORDER shape independently of result cardinality. Unsupported
        // device shapes must fail loud even for an empty table or an empty WHERE result.
        let prepared = select
            .order_by
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let nulls_first = order_by_nulls_first
                    .get(index)
                    .copied()
                    .flatten()
                    .unwrap_or(item.descending);
                let Some(expression) = order_by_exprs.get(index).and_then(Option::as_ref) else {
                    let column = relational_column_index(table, &item.column)?;
                    return Ok(PreparedOrderKey::Resident(CudaJoinOrderKey {
                        relation: 0,
                        key: payload_key(table, snapshot, memory, column)?,
                        descending: item.descending,
                        nulls_first,
                        lexicographic_16: table.columns[column].ty == SqlType::Uuid,
                    }));
                };
                let elem = order_arithmetic_elem(expression, table)?;
                let mut program = Vec::new();
                compile_arith_program(expression, table, snapshot, &mut program)?;
                let nullable = predicate_references_nullable_column(expression, table, snapshot)?;
                if nullable {
                    if elem == ResidentElemType::I64 {
                    return Err(result_error(
                        "ORDER BY a nullable int8 expression is not supported on the device",
                    ));
                }
                if order_by_nulls_first.get(index).copied().flatten().is_some() {
                    return Err(result_error(
                        "explicit NULLS FIRST/LAST on a nullable ORDER BY expression is not supported",
                    ));
                }
                let mut validity_program = vec![ExprStep::ConstMask { value: true }];
                push_leaf_validity_and(
                    std::slice::from_ref(&expression),
                    table,
                    snapshot,
                    &mut validity_program,
                )?;
                    Ok(PreparedOrderKey::Derived {
                        program,
                        validity_program: Some(validity_program),
                        elem: ResidentElemType::I32,
                        width: 8,
                        descending: item.descending,
                        nulls_first,
                    })
                } else {
                    Ok(PreparedOrderKey::Derived {
                        program,
                        validity_program: None,
                        elem,
                        width: match elem {
                            ResidentElemType::I32 => 4,
                            ResidentElemType::I64 => 8,
                            ResidentElemType::I128 => 16,
                        },
                        descending: item.descending,
                        nulls_first,
                    })
                }
            })
            .collect::<Result<Vec<_>, ExecuteError>>()?;

        if coordinates.row_count() > 0 {
            let derived = prepared
                .iter()
                .map(|key| match key {
                    PreparedOrderKey::Resident(_) => Ok(None),
                    PreparedOrderKey::Derived {
                        program,
                        validity_program: Some(validity_program),
                        ..
                    } => memory
                        .arith_value_column_device_at_coordinates_nullable(
                            program,
                            validity_program,
                            &coordinates,
                            row_count,
                        )
                        .map(Some)
                        .map_err(map_err),
                    PreparedOrderKey::Derived { program, elem, .. } => memory
                        .arith_value_column_device_at_coordinates(
                            program,
                            &coordinates,
                            row_count,
                            *elem,
                        )
                        .map(Some)
                        .map_err(map_err),
                })
                .collect::<Result<Vec<_>, ExecuteError>>()?;
            let order = prepared
                .iter()
                .zip(&derived)
                .map(|(key, value)| match key {
                    PreparedOrderKey::Resident(key) => CudaJoinSortKey::Resident(*key),
                    PreparedOrderKey::Derived {
                        width,
                        descending,
                        nulls_first,
                        ..
                    } => CudaJoinSortKey::Derived {
                        relation: 0,
                        values: value
                            .as_ref()
                            .expect("derived ORDER owner exists")
                            .group_view(),
                        width: *width,
                        descending: *descending,
                        nulls_first: *nulls_first,
                    },
                })
                .collect::<Vec<_>>();
            let sorted = if coordinates.row_count() > 1 {
                Some(
                    memory
                        .sort_join_coordinates_with_keys(&coordinates, &order)
                        .map_err(map_err)?,
                )
            } else {
                None
            };
            drop(order);
            drop(derived);
            if let Some(sorted) = sorted {
                coordinates = sorted;
            }
        }
    }

    if select.offset.is_some() || select.limit.is_some() {
        let offset = select.offset.unwrap_or(0).min(row_count as usize) as u32;
        let limit = select
            .limit
            .map(|limit| u32::try_from(limit).unwrap_or(u32::MAX));
        coordinates = memory
            .window_join_coordinates(&coordinates, offset, limit)
            .map_err(map_err)?;
    }

    let specs = bound
        .selected_indexes
        .iter()
        .map(|&column| projection_spec(table, snapshot, memory, column))
        .collect::<Result<Vec<_>, _>>()?;
    let terminal = memory
        .materialize_join_coordinates(&coordinates, &specs)
        .map_err(map_err)?;
    let frame = terminal.read_result_frame().map_err(map_err)?;
    let rows = engine.decode_materialized_result_frame(&frame, &bound.selected_columns)?;
    Ok(RelationalSelectResult {
        columns: Arc::new(bound.selected_columns),
        rows: rows.into(),
        planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
        executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
        fallback_reason: None,
        access_path: Arc::new(access_path),
    })
}
