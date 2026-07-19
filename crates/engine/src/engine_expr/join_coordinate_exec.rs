//! Main device-coordinate join execution.
//!
//! This leaf owns the direct non-streaming join coordinate pipeline, including predicate and
//! visibility masks, typed keys, OUTER completion, ordering/windowing, and final result framing.

use super::execution_source::ResidentVisibility;
use super::join_source::{JoinExecSide, JoinNullPadMask};
use crate::engine_expr_ir::ResidentExpr;
use crate::engine_join_ir::JoinPlan;
use crate::relational_model::{
    resident_device_bool_column_offset, resident_device_int4_column_offset,
    resident_device_int8_column_offset, resident_device_null_column_offset,
    resident_device_numeric_column_offset, resident_device_text_column_layout,
    RelationalAccessPath, RelationalSelectResult, RelationalTable,
};
use crate::{Engine, ExecuteError};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{SqlType, SqlValue};
use gpu_db_types::EngineError;
use std::sync::Arc;

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_resident_device_coordinate_join(
        &self,
        plan: &JoinPlan,
        tables: &[RelationalTable],
        sides: &[JoinExecSide],
        step_keys: &[Vec<(usize, usize, usize)>],
        _step_is_text: &[bool],
        _step_is_b128: &[bool],
        projection: &[(usize, usize)],
        predicates: &[Option<ResidentExpr>],
        outer_where: bool,
        row_ranges: Option<&[(u32, u32)]>,
        resolved_order: &[(usize, usize, bool, Option<bool>)],
        gpu_id: u16,
        coordinate_out: Option<&mut Option<gpu_db_execution::CudaJoinCoordinatesU32>>,
        materialized_out: Option<&mut Option<gpu_db_execution::CudaMaterializedRelation>>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        use gpu_db_execution::{CudaJoinOrderKey, CudaJoinPayloadKey, CudaPredicateMaskI32};
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        for (_, _, row_count, _) in sides {
            if *row_count > u32::MAX as usize {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "join row count exceeds the device u32 coordinate range".to_string(),
                )));
            }
        }

        let compile_mask = |relation: usize,
                            predicate: Option<&ResidentExpr>,
                            visibility: Option<ResidentVisibility>|
         -> Result<Option<CudaPredicateMaskI32>, ExecuteError> {
            self.resident_predicate_device_mask(
                predicate,
                &tables[relation],
                &sides[relation].0.descriptor,
                sides[relation].1.mem(),
                sides[relation].2 as u32,
                visibility,
            )
        };

        // Input eligibility always includes MVCC visibility. INNER joins may push their per-relation
        // WHERE into the input mask; OUTER joins retain WHERE as a post-coordinate mask. Streaming block
        // subsets are scheduler metadata and become one additional device mask.
        let mut input_masks: Vec<Option<CudaPredicateMaskI32>> = Vec::with_capacity(sides.len());
        let mut post_masks: Vec<Option<CudaPredicateMaskI32>> = Vec::with_capacity(sides.len());
        let mut pad_masks: Vec<Option<JoinNullPadMask>> = (0..sides.len()).map(|_| None).collect();
        for relation in 0..sides.len() {
            let _mask_scope = gpu_db_execution::Probe::scope("join_input_mask");
            let pushed = (!outer_where)
                .then_some(predicates[relation].as_ref())
                .flatten();
            let mut input = compile_mask(relation, pushed, sides[relation].3)?;
            if let Some(ranges) = row_ranges {
                let (start, end) = ranges[relation];
                let range = sides[relation]
                    .1
                    .mem()
                    .row_range_mask_u32(sides[relation].2 as u32, start, end)
                    .map_err(map_err)?;
                input = match input {
                    Some(mask) => Some(
                        sides[relation]
                            .1
                            .mem()
                            .and_predicate_masks(&mask, &range)
                            .map_err(map_err)?,
                    ),
                    None => Some(range),
                };
            }
            input_masks.push(input);
            if outer_where && predicates[relation].is_some() {
                post_masks.push(compile_mask(relation, predicates[relation].as_ref(), None)?);
                pad_masks[relation] = Some(self.predicate_mask_on_null_pad(
                    predicates[relation].as_ref().expect("checked"),
                    &tables[relation],
                )?);
            } else {
                post_masks.push(None);
            }
        }

        let key_descriptor = |relation: usize,
                              column: usize|
         -> Result<CudaJoinPayloadKey<'_>, ExecuteError> {
            let entry = &sides[relation].0;
            let payload = sides[relation].1.mem();
            let validity_bitmap_offset =
                resident_device_null_column_offset(&entry.descriptor, &tables[relation], column)?;
            match tables[relation].columns[column].ty {
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?;
                    Ok(CudaJoinPayloadKey {
                        payload,
                        byte_offset: layout.offsets_byte_offset,
                        validity_bitmap_offset,
                        width: 255,
                        text_bytes_byte_offset: Some(layout.bytes_byte_offset),
                        text_bytes_len: layout.bytes_len,
                    })
                }
                SqlType::Numeric { .. } | SqlType::Uuid => Ok(CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_numeric_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset,
                    width: 16,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                }),
                SqlType::Int8 | SqlType::Timestamp => Ok(CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_int8_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset,
                    width: 8,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                }),
                SqlType::Int4 | SqlType::Int2 | SqlType::Date => Ok(CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_int4_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset,
                    width: 4,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                }),
                SqlType::Bool => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "bool equi-join keys are not represented as fixed payload keys".to_string(),
                ))),
            }
        };

        let context = sides[0].1.mem();
        let mut coordinates: Option<gpu_db_execution::CudaJoinCoordinatesU32> = None;
        for (step_index, conjuncts) in step_keys.iter().enumerate() {
            let _coordinate_scope =
                gpu_db_execution::Probe::scope("join_fixed_payload_coordinates");
            let right_relation = step_index + 1;
            let left_keys = conjuncts
                .iter()
                .map(|&(relation, column, _)| key_descriptor(relation, column))
                .collect::<Result<Vec<_>, _>>()?;
            let right_keys = conjuncts
                .iter()
                .map(|&(_, _, column)| key_descriptor(right_relation, column))
                .collect::<Result<Vec<_>, _>>()?;
            let left_key_relations = conjuncts
                .iter()
                .map(|&(relation, _, _)| relation as u32)
                .collect::<Vec<_>>();
            coordinates = Some(
                context
                    .join_fixed_payload_coordinates(
                        coordinates.as_ref(),
                        if step_index == 0 {
                            sides[0].2 as u32
                        } else {
                            0
                        },
                        &left_key_relations,
                        &left_keys,
                        sides[right_relation].2 as u32,
                        &right_keys,
                        if step_index == 0 {
                            input_masks[0].as_ref()
                        } else {
                            None
                        },
                        input_masks[right_relation].as_ref(),
                        plan.steps[step_index].outer_left,
                        plan.steps[step_index].outer_right,
                    )
                    .map_err(map_err)?,
            );
        }
        let mut coordinates = coordinates.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "a join plan requires at least one join step".to_string(),
            ))
        })?;

        // Streaming match state consumes an ON-coordinate copy before OUTER-WHERE, ORDER BY, or LIMIT.
        if let Some(out) = coordinate_out {
            *out = Some(
                context
                    .window_join_coordinates(&coordinates, 0, None)
                    .map_err(map_err)?,
            );
        }
        if outer_where {
            coordinates = context
                .filter_join_coordinates(
                    &coordinates,
                    &post_masks.iter().map(Option::as_ref).collect::<Vec<_>>(),
                    &pad_masks
                        .iter()
                        .map(|guard| guard.as_ref().and_then(|guard| guard.mask.as_ref()))
                        .collect::<Vec<_>>(),
                )
                .map_err(map_err)?;
        }

        if !resolved_order.is_empty() {
            let order = resolved_order
                .iter()
                .map(|&(relation, column, descending, nulls_first)| {
                    Ok(CudaJoinOrderKey {
                        relation: relation as u32,
                        key: key_descriptor(relation, column)?,
                        descending,
                        nulls_first: nulls_first.unwrap_or(descending),
                        lexicographic_16: tables[relation].columns[column].ty == SqlType::Uuid,
                    })
                })
                .collect::<Result<Vec<_>, ExecuteError>>()?;
            coordinates = context
                .sort_join_coordinates(&coordinates, &order)
                .map_err(map_err)?;
        }
        if plan.offset.is_some() || plan.limit.is_some() {
            let offset = u32::try_from(plan.offset.unwrap_or(0)).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "join OFFSET exceeds the device coordinate range".to_string(),
                ))
            })?;
            let limit = plan.limit.map(u32::try_from).transpose().map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "join LIMIT exceeds the device coordinate range".to_string(),
                ))
            })?;
            coordinates = context
                .window_join_coordinates(&coordinates, offset, limit)
                .map_err(map_err)?;
        }

        let output_aliases = self.join_projection_output_aliases(plan, tables)?;
        let mut columns = Vec::with_capacity(projection.len());
        for (index, &(relation, column)) in projection.iter().enumerate() {
            let mut output = tables[relation].columns[column].clone();
            output.attnum = (index + 1) as i16;
            if let Some(alias) = &output_aliases[index] {
                output.name.clone_from(alias);
            }
            columns.push(output);
        }
        let mut specs = Vec::with_capacity(projection.len());
        for &(relation, column) in projection {
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
        let materialized = context
            .materialize_join_coordinates(&coordinates, &specs)
            .map_err(map_err)?;
        if let Some(out) = materialized_out {
            *out = Some(materialized);
            return Ok(RelationalSelectResult {
                columns: Arc::new(columns),
                rows: Vec::<Vec<SqlValue>>::new().into(),
                planned_target: DeviceTarget::Gpu(gpu_id),
                executed_target: DeviceTarget::Gpu(gpu_id),
                fallback_reason: None,
                access_path: Arc::new(RelationalAccessPath::FullTableScan),
            });
        }
        let frame = materialized.read_result_frame().map_err(map_err)?;
        let result_rows = self.decode_materialized_result_frame(&frame, &columns)?;
        Ok(RelationalSelectResult {
            columns: Arc::new(columns),
            rows: result_rows.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        })
    }
}
