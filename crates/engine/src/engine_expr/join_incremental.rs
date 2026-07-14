//! Streaming/incremental join coordinate construction: identity seeding and typed match
//! extension. Main join orchestration and projection materialization remain elsewhere.

use super::join_source::JoinExecSide;
use crate::engine_join_ir::{JoinColRef, JoinPlan};
use crate::rel_exec_helpers::relational_column_index;
use crate::relational_model::{
    resident_device_int4_column_offset, resident_device_int8_column_offset,
    resident_device_null_column_offset, resident_device_numeric_column_offset,
    resident_device_text_column_layout, RelationalTable,
};
use crate::{Engine, ExecuteError};
use gpu_db_execution::{CudaJoinCoordinatesU32, CudaJoinPayloadKey, CudaMatchBitmapU32};
use gpu_db_sql::SqlType;
use gpu_db_types::EngineError;

impl Engine {
    pub(crate) fn resident_join_identity_coordinates(
        &self,
        table: &RelationalTable,
        side: &JoinExecSide,
        row_range: Option<(u32, u32)>,
    ) -> Result<CudaJoinCoordinatesU32, ExecuteError> {
        let mut mask = self.resident_predicate_device_mask(
            None,
            table,
            &side.0.descriptor,
            side.1.mem(),
            side.2 as u32,
            side.3,
        )?;
        if let Some((start, end)) = row_range {
            let range = side
                .1
                .mem()
                .row_range_mask_u32(side.2 as u32, start, end)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            mask = match mask {
                Some(mask) => Some(side.1.mem().and_predicate_masks(&mask, &range).map_err(
                    |err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())),
                )?),
                None => Some(range),
            };
        }
        side.1
            .mem()
            .identity_join_coordinates(side.2 as u32, mask.as_ref())
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_resident_join_coordinate_matches(
        &self,
        plan: &JoinPlan,
        tables: &[RelationalTable],
        sides: &[JoinExecSide],
        step_index: usize,
        accumulated: &CudaJoinCoordinatesU32,
        left_matches: &CudaMatchBitmapU32,
        right_matches: Option<&CudaMatchBitmapU32>,
        right_range: Option<(u32, u32)>,
    ) -> Result<CudaJoinCoordinatesU32, ExecuteError> {
        use CudaJoinPayloadKey;

        let right_relation = step_index + 1;
        if right_relation >= sides.len() || accumulated.relation_count() != right_relation as u32 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "incremental join step arity does not match its accumulated coordinates"
                    .to_string(),
            )));
        }
        let resolve = |column: &JoinColRef| -> Result<(usize, usize), ExecuteError> {
            if let Some(qualifier) = &column.qualifier {
                let relation = plan
                    .relations
                    .iter()
                    .take(right_relation + 1)
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
                .take(right_relation + 1)
                .enumerate()
                .filter_map(|(relation, table)| {
                    relational_column_index(table, &column.column)
                        .ok()
                        .map(|column| (relation, column))
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [value] => Ok(*value),
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
        let descriptor = |relation: usize,
                          column: usize|
         -> Result<CudaJoinPayloadKey<'_>, ExecuteError> {
            let entry = &sides[relation].0;
            let payload = sides[relation].1.mem();
            let validity =
                resident_device_null_column_offset(&entry.descriptor, &tables[relation], column)?;
            Ok(match tables[relation].columns[column].ty {
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?;
                    CudaJoinPayloadKey {
                        payload,
                        byte_offset: layout.offsets_byte_offset,
                        validity_bitmap_offset: validity,
                        width: 255,
                        text_bytes_byte_offset: Some(layout.bytes_byte_offset),
                        text_bytes_len: layout.bytes_len,
                    }
                }
                SqlType::Numeric { .. } | SqlType::Uuid => CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_numeric_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset: validity,
                    width: 16,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                },
                SqlType::Int8 | SqlType::Timestamp => CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_int8_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset: validity,
                    width: 8,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                },
                SqlType::Int2 | SqlType::Int4 | SqlType::Date => CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_int4_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset: validity,
                    width: 4,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                },
                SqlType::Bool => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "bool equi-join keys are not supported".to_string(),
                    )))
                }
            })
        };
        let step = &plan.steps[step_index];
        if step.natural || step.conjuncts.is_empty() || step.conjuncts.len() > 2 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "incremental streaming OUTER joins require one or two explicit equality keys"
                    .to_string(),
            )));
        }
        let mut left_keys = Vec::with_capacity(step.conjuncts.len());
        let mut right_keys = Vec::with_capacity(step.conjuncts.len());
        let mut left_key_relations = Vec::with_capacity(step.conjuncts.len());
        for (left, right) in &step.conjuncts {
            let a = resolve(left)?;
            let b = resolve(right)?;
            let (acc, new) = if a.0 == right_relation && b.0 < right_relation {
                (b, a)
            } else if b.0 == right_relation && a.0 < right_relation {
                (a, b)
            } else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "each incremental join key must connect the new relation to the accumulated side"
                        .to_string(),
                )));
            };
            left_key_relations.push(acc.0 as u32);
            left_keys.push(descriptor(acc.0, acc.1)?);
            right_keys.push(descriptor(new.0, new.1)?);
        }
        let mut right_mask = self.resident_predicate_device_mask(
            None,
            &tables[right_relation],
            &sides[right_relation].0.descriptor,
            sides[right_relation].1.mem(),
            sides[right_relation].2 as u32,
            sides[right_relation].3,
        )?;
        if let Some((start, end)) = right_range {
            let range = sides[right_relation]
                .1
                .mem()
                .row_range_mask_u32(sides[right_relation].2 as u32, start, end)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            right_mask = match right_mask {
                Some(mask) => Some(
                    sides[right_relation]
                        .1
                        .mem()
                        .and_predicate_masks(&mask, &range)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?,
                ),
                None => Some(range),
            };
        }
        sides[0]
            .1
            .mem()
            .join_fixed_payload_coordinate_matches(
                Some(accumulated),
                0,
                &left_key_relations,
                &left_keys,
                sides[right_relation].2 as u32,
                &right_keys,
                None,
                right_mask.as_ref(),
                left_matches,
                right_matches,
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }
}
