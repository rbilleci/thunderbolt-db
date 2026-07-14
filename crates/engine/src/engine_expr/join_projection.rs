//! Join projection resolution and device-result materialization.
//!
//! This leaf owns projection source and alias expansion plus typed coordinate materialization.
//! Join planning, coordinate filtering/execution, and streaming orchestration remain with their
//! established owners.

use super::join_source::JoinExecSide;
use crate::engine_join_ir::{JoinColRef, JoinPlan, JoinProjItem};
use crate::rel_exec_helpers::relational_column_index;
use crate::relational_model::{
    resident_device_bool_column_offset, resident_device_int4_column_offset,
    resident_device_int8_column_offset, resident_device_null_column_offset,
    resident_device_numeric_column_offset, resident_device_text_column_layout, RelationalColumn,
    RelationalTable,
};
use crate::{Engine, ExecuteError};
use gpu_db_sql::{SqlType, SqlValue};
use gpu_db_types::EngineError;
use std::sync::Arc;

impl Engine {
    pub(crate) fn materialize_join_projection_coordinates(
        &self,
        plan: &JoinPlan,
        tables: &[RelationalTable],
        sides: &[JoinExecSide],
        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32,
    ) -> Result<
        (
            Arc<Vec<RelationalColumn>>,
            gpu_db_execution::CudaMaterializedRelation,
        ),
        ExecuteError,
    > {
        let resolve = |column: &JoinColRef| -> Result<(usize, usize), ExecuteError> {
            if let Some(qualifier) = &column.qualifier {
                let relation = plan
                    .relations
                    .iter()
                    .position(|relation| relation.alias == *qualifier)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "missing FROM-clause entry for table \"{qualifier}\""
                        )))
                    })?;
                return Ok((
                    relation,
                    relational_column_index(&tables[relation], &column.column)?,
                ));
            }
            let matches = tables
                .iter()
                .enumerate()
                .filter_map(|(relation, table)| {
                    relational_column_index(table, &column.column)
                        .ok()
                        .map(|column| (relation, column))
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [resolved] => Ok(*resolved),
                [] => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column \"{}\" does not exist",
                    column.column
                )))),
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column reference \"{}\" is ambiguous",
                    column.column
                )))),
            }
        };
        let coalesced = plan
            .steps
            .iter()
            .flat_map(|step| step.coalesce.iter().cloned())
            .collect::<Vec<_>>();
        let mut projection = Vec::new();
        for item in &plan.projection {
            match item {
                JoinProjItem::Column(column)
                    if column.qualifier.is_none() && coalesced.contains(&column.column) =>
                {
                    projection.push((0, relational_column_index(&tables[0], &column.column)?));
                }
                JoinProjItem::Column(column) => projection.push(resolve(column)?),
                JoinProjItem::Star(None) if !coalesced.is_empty() => {
                    for name in &coalesced {
                        projection.push((0, relational_column_index(&tables[0], name)?));
                    }
                    for (relation, table) in tables.iter().enumerate() {
                        projection.extend(
                            table
                                .columns
                                .iter()
                                .enumerate()
                                .filter(|(_, column)| !coalesced.contains(&column.name))
                                .map(|(column, _)| (relation, column)),
                        );
                    }
                }
                JoinProjItem::Star(None) => {
                    for (relation, table) in tables.iter().enumerate() {
                        projection.extend(
                            table
                                .columns
                                .iter()
                                .enumerate()
                                .map(|(column, _)| (relation, column)),
                        );
                    }
                }
                JoinProjItem::Star(Some(alias)) => {
                    let relation = plan
                        .relations
                        .iter()
                        .position(|relation| relation.alias == *alias)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "missing FROM-clause entry for table \"{alias}\""
                            )))
                        })?;
                    projection.extend(
                        tables[relation]
                            .columns
                            .iter()
                            .enumerate()
                            .map(|(column, _)| (relation, column)),
                    );
                }
            }
        }
        let output_aliases = self.join_projection_output_aliases(plan, tables)?;
        let mut columns = Vec::with_capacity(projection.len());
        let mut specs = Vec::with_capacity(projection.len());
        for (output_index, &(relation, column)) in projection.iter().enumerate() {
            let mut output = tables[relation].columns[column].clone();
            output.attnum = (output_index + 1) as i16;
            if let Some(alias) = &output_aliases[output_index] {
                output.name.clone_from(alias);
            }
            columns.push(output);
            let entry = &sides[relation].0;
            let payload = sides[relation].1.mem();
            let validity =
                resident_device_null_column_offset(&entry.descriptor, &tables[relation], column)?;
            specs.push(match tables[relation].columns[column].ty {
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?;
                    gpu_db_execution::CudaMaterializeJoinColumn::Text {
                        relation: relation as u32,
                        payload,
                        offsets_byte_offset: layout.offsets_byte_offset,
                        bytes_byte_offset: layout.bytes_byte_offset,
                        bytes_len: layout.bytes_len,
                        validity_bitmap_offset: validity,
                    }
                }
                SqlType::Bool => gpu_db_execution::CudaMaterializeJoinColumn::Bool {
                    relation: relation as u32,
                    payload,
                    bitmap_byte_offset: resident_device_bool_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset: validity,
                },
                ty => {
                    let (byte_offset, width) = match ty {
                        SqlType::Int8 | SqlType::Timestamp => (
                            resident_device_int8_column_offset(
                                &entry.descriptor,
                                &tables[relation],
                                column,
                            )?,
                            8,
                        ),
                        SqlType::Numeric { .. } | SqlType::Uuid => (
                            resident_device_numeric_column_offset(
                                &entry.descriptor,
                                &tables[relation],
                                column,
                            )?,
                            16,
                        ),
                        SqlType::Int2 | SqlType::Int4 | SqlType::Date => (
                            resident_device_int4_column_offset(
                                &entry.descriptor,
                                &tables[relation],
                                column,
                            )?,
                            4,
                        ),
                        SqlType::Text | SqlType::Bool => unreachable!(),
                    };
                    gpu_db_execution::CudaMaterializeJoinColumn::Fixed {
                        relation: relation as u32,
                        payload,
                        byte_offset,
                        validity_bitmap_offset: validity,
                        width,
                    }
                }
            });
        }
        let run = sides[0]
            .1
            .mem()
            .materialize_join_coordinates(coordinates, &specs)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        Ok((Arc::new(columns), run))
    }

    pub(crate) fn join_projection_sources(
        &self,
        plan: &JoinPlan,
        tables: &[RelationalTable],
    ) -> Result<Vec<(usize, usize)>, ExecuteError> {
        let resolve = |column: &JoinColRef| -> Result<(usize, usize), ExecuteError> {
            if let Some(qualifier) = &column.qualifier {
                let relation = plan
                    .relations
                    .iter()
                    .position(|relation| relation.alias == *qualifier)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "missing FROM-clause entry for table \"{qualifier}\""
                        )))
                    })?;
                return Ok((
                    relation,
                    relational_column_index(&tables[relation], &column.column)?,
                ));
            }
            let matches = tables
                .iter()
                .enumerate()
                .filter_map(|(relation, table)| {
                    relational_column_index(table, &column.column)
                        .ok()
                        .map(|column| (relation, column))
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [resolved] => Ok(*resolved),
                [] => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column \"{}\" does not exist",
                    column.column
                )))),
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column reference \"{}\" is ambiguous",
                    column.column
                )))),
            }
        };
        let coalesced = plan
            .steps
            .iter()
            .flat_map(|step| step.coalesce.iter().cloned())
            .collect::<Vec<_>>();
        let mut projection = Vec::new();
        for item in &plan.projection {
            match item {
                JoinProjItem::Column(column)
                    if column.qualifier.is_none() && coalesced.contains(&column.column) =>
                {
                    projection.push((0, relational_column_index(&tables[0], &column.column)?));
                }
                JoinProjItem::Column(column) => projection.push(resolve(column)?),
                JoinProjItem::Star(None) if !coalesced.is_empty() => {
                    for name in &coalesced {
                        projection.push((0, relational_column_index(&tables[0], name)?));
                    }
                    for (relation, table) in tables.iter().enumerate() {
                        projection.extend(
                            table
                                .columns
                                .iter()
                                .enumerate()
                                .filter(|(_, column)| !coalesced.contains(&column.name))
                                .map(|(column, _)| (relation, column)),
                        );
                    }
                }
                JoinProjItem::Star(None) => {
                    for (relation, table) in tables.iter().enumerate() {
                        projection.extend(
                            table
                                .columns
                                .iter()
                                .enumerate()
                                .map(|(column, _)| (relation, column)),
                        );
                    }
                }
                JoinProjItem::Star(Some(alias)) => {
                    let relation = plan
                        .relations
                        .iter()
                        .position(|relation| relation.alias == *alias)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "missing FROM-clause entry for table \"{alias}\""
                            )))
                        })?;
                    projection.extend(
                        tables[relation]
                            .columns
                            .iter()
                            .enumerate()
                            .map(|(column, _)| (relation, column)),
                    );
                }
            }
        }
        Ok(projection)
    }

    pub(crate) fn join_projection_output_aliases(
        &self,
        plan: &JoinPlan,
        tables: &[RelationalTable],
    ) -> Result<Vec<Option<String>>, ExecuteError> {
        if plan.projection.len() != plan.projection_aliases.len() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "join projection alias metadata is misaligned".to_string(),
            )));
        }
        let coalesced = plan
            .steps
            .iter()
            .flat_map(|step| step.coalesce.iter())
            .collect::<Vec<_>>();
        let mut aliases = Vec::new();
        for (item, alias) in plan.projection.iter().zip(&plan.projection_aliases) {
            match item {
                JoinProjItem::Column(_) => aliases.push(alias.clone()),
                JoinProjItem::Star(None) if !coalesced.is_empty() => {
                    aliases.extend((0..coalesced.len()).map(|_| None));
                    aliases.extend(
                        tables
                            .iter()
                            .flat_map(|table| &table.columns)
                            .filter(|column| !coalesced.iter().any(|name| ***name == column.name))
                            .map(|_| None),
                    );
                }
                JoinProjItem::Star(None) => {
                    aliases.extend(tables.iter().flat_map(|table| &table.columns).map(|_| None))
                }
                JoinProjItem::Star(Some(qualifier)) => {
                    let relation = plan
                        .relations
                        .iter()
                        .position(|relation| relation.alias == *qualifier)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "missing FROM-clause entry for table \"{qualifier}\""
                            )))
                        })?;
                    aliases.extend((0..tables[relation].columns.len()).map(|_| None));
                }
            }
        }
        Ok(aliases)
    }

    pub(crate) fn project_join_column_values(
        &self,
        table: &RelationalTable,
        side: &JoinExecSide,
        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32,
        relation: u32,
        column: usize,
    ) -> Result<Vec<SqlValue>, ExecuteError> {
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let payload = side.1.mem();
        let validity = resident_device_null_column_offset(&side.0.descriptor, table, column)?;
        let ty = table.columns[column].ty;
        Ok(match ty {
            SqlType::Text => {
                let layout = resident_device_text_column_layout(&side.0.descriptor, table, column)?;
                payload
                    .project_text_from_join_coordinates(
                        coordinates,
                        relation,
                        payload,
                        layout.offsets_byte_offset,
                        layout.bytes_byte_offset,
                        layout.bytes_len,
                        validity,
                    )
                    .map_err(map_err)?
                    .into_iter()
                    .map(|value| value.map_or(SqlValue::Null, SqlValue::Text))
                    .collect()
            }
            SqlType::Bool => payload
                .project_bool_from_join_coordinates(
                    coordinates,
                    relation,
                    payload,
                    resident_device_bool_column_offset(&side.0.descriptor, table, column)?,
                    validity,
                )
                .map_err(map_err)?
                .into_iter()
                .map(|value| value.map_or(SqlValue::Null, SqlValue::Bool))
                .collect(),
            _ => {
                let (byte_offset, width) = match ty {
                    SqlType::Int8 | SqlType::Timestamp => (
                        resident_device_int8_column_offset(&side.0.descriptor, table, column)?,
                        8_u8,
                    ),
                    SqlType::Numeric { .. } | SqlType::Uuid => (
                        resident_device_numeric_column_offset(&side.0.descriptor, table, column)?,
                        16_u8,
                    ),
                    SqlType::Int2 | SqlType::Int4 | SqlType::Date => (
                        resident_device_int4_column_offset(&side.0.descriptor, table, column)?,
                        4_u8,
                    ),
                    SqlType::Text | SqlType::Bool => unreachable!(),
                };
                let (raw, valid) = payload
                    .project_fixed_from_join_coordinates(
                        coordinates,
                        relation,
                        payload,
                        byte_offset,
                        validity,
                        width,
                    )
                    .map_err(map_err)?;
                raw.chunks_exact(width as usize)
                    .zip(valid)
                    .map(|(bytes, valid)| {
                        if !valid {
                            return SqlValue::Null;
                        }
                        match ty {
                            SqlType::Int2 => SqlValue::Int2(i32::from_le_bytes(
                                bytes.try_into().expect("int2 width"),
                            ) as i16),
                            SqlType::Int4 => SqlValue::Int4(i32::from_le_bytes(
                                bytes.try_into().expect("int4 width"),
                            )),
                            SqlType::Date => SqlValue::Date(i32::from_le_bytes(
                                bytes.try_into().expect("date width"),
                            )),
                            SqlType::Int8 => SqlValue::Int8(i64::from_le_bytes(
                                bytes.try_into().expect("int8 width"),
                            )),
                            SqlType::Timestamp => SqlValue::Timestamp(i64::from_le_bytes(
                                bytes.try_into().expect("timestamp width"),
                            )),
                            SqlType::Numeric { scale, .. } => {
                                SqlValue::Numeric(gpu_db_sql::Decimal128::new(
                                    i128::from_le_bytes(bytes.try_into().expect("numeric width")),
                                    scale,
                                ))
                            }
                            SqlType::Uuid => SqlValue::Uuid(bytes.try_into().expect("uuid width")),
                            SqlType::Text | SqlType::Bool => unreachable!(),
                        }
                    })
                    .collect()
            }
        })
    }
}
