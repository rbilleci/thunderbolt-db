//! Device-materialized streaming join run planning, merging, and decoding.

use super::*;

impl Engine {
    pub(super) fn materialized_join_run_specs<'a>(
        run: &'a gpu_db_execution::CudaMaterializedRelation,
    ) -> Vec<gpu_db_execution::CudaMaterializeJoinColumn<'a>> {
        use gpu_db_execution::{CudaMaterializeJoinColumn, CudaMaterializedColumnKind};

        run.columns()
            .iter()
            .map(|layout| match layout.kind {
                CudaMaterializedColumnKind::Fixed { width } => CudaMaterializeJoinColumn::Fixed {
                    relation: 0,
                    payload: run.memory(),
                    byte_offset: layout.value_byte_offset,
                    validity_bitmap_offset: Some(layout.validity_bitmap_offset),
                    width,
                },
                CudaMaterializedColumnKind::Text => CudaMaterializeJoinColumn::Text {
                    relation: 0,
                    payload: run.memory(),
                    offsets_byte_offset: layout.value_byte_offset,
                    bytes_byte_offset: layout.text_bytes_byte_offset.expect("text bytes layout"),
                    bytes_len: layout.text_bytes_len,
                    validity_bitmap_offset: Some(layout.validity_bitmap_offset),
                },
            })
            .collect()
    }

    pub(super) fn materialized_join_run_order<'a>(
        &self,
        plan: &crate::engine_expr::JoinPlan,
        tables: &[RelationalTable],
        columns: &[RelationalColumn],
        run: &'a gpu_db_execution::CudaMaterializedRelation,
    ) -> Result<Vec<gpu_db_execution::CudaJoinOrderKey<'a>>, ExecuteError> {
        let sources = self.join_projection_sources(plan, tables)?;
        let aliases = self.join_projection_output_aliases(plan, tables)?;
        plan.order_by
            .iter()
            .enumerate()
            .map(|(order_index, (key, descending))| {
                let alias_matches = if key.qualifier.is_none() {
                    aliases
                        .iter()
                        .enumerate()
                        .filter_map(|(index, alias)| {
                            (alias.as_deref() == Some(&key.column)).then_some(index)
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                let alias_column = match alias_matches.as_slice() {
                    [column] => Some(*column),
                    [] => None,
                    _ => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "ORDER BY \"{}\" is ambiguous",
                            key.column
                        ))))
                    }
                };
                let resolved_relation = if alias_column.is_some() {
                    None
                } else if let Some(qualifier) = &key.qualifier {
                    Some(
                        plan.relations
                            .iter()
                            .position(|relation| relation.alias == *qualifier)
                            .ok_or_else(|| {
                                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                    "missing FROM-clause entry for table \"{qualifier}\""
                                )))
                            })?,
                    )
                } else {
                    let relations = tables
                        .iter()
                        .enumerate()
                        .filter_map(|(relation, table)| {
                            relational_column_index(table, &key.column)
                                .ok()
                                .map(|_| relation)
                        })
                        .collect::<Vec<_>>();
                    match relations.as_slice() {
                        [relation] => Some(*relation),
                        [] => None,
                        _ => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "column reference \"{}\" is ambiguous",
                                key.column
                            ))))
                        }
                    }
                };
                let source_matches = sources
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &(relation, column))| {
                        (Some(relation) == resolved_relation
                            && tables[relation].columns[column].name == key.column)
                            .then_some(index)
                    })
                    .collect::<Vec<_>>();
                let column = if let Some(column) = alias_column {
                    column
                } else if let [column] = source_matches.as_slice() {
                    *column
                } else {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "streaming JOIN ORDER BY column `{}` must identify exactly one projected result column",
                        key.column
                    ))));
                };
                Ok(gpu_db_execution::CudaJoinOrderKey {
                    relation: 0,
                    key: run.payload_key(column).ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "streaming JOIN run column layout is missing".to_string(),
                        ))
                    })?,
                    descending: *descending,
                    nulls_first: plan
                        .order_by_nulls_first
                        .get(order_index)
                        .copied()
                        .flatten()
                        .unwrap_or(*descending),
                    lexicographic_16: columns[column].ty == SqlType::Uuid,
                })
            })
            .collect()
    }

    pub(super) fn join_run_plan(
        &self,
        plan: &crate::engine_expr::JoinPlan,
        tables: &[RelationalTable],
    ) -> Result<(crate::engine_expr::JoinPlan, usize), ExecuteError> {
        let mut run_plan = plan.clone();
        let visible = self.join_projection_sources(plan, tables)?.len();
        let mut sources = self.join_projection_sources(&run_plan, tables)?;
        let visible_aliases = self.join_projection_output_aliases(plan, tables)?;
        for (key, _) in &plan.order_by {
            let alias_matches = if key.qualifier.is_none() {
                visible_aliases
                    .iter()
                    .enumerate()
                    .filter_map(|(index, alias)| {
                        (alias.as_deref() == Some(&key.column)).then_some(index)
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let source = match alias_matches.as_slice() {
                [index] => sources[*index],
                [] => {
                    let mut key_plan = plan.clone();
                    key_plan.projection =
                        vec![crate::engine_expr::JoinProjItem::Column(key.clone())];
                    key_plan.projection_aliases = vec![None];
                    self.join_projection_sources(&key_plan, tables)?
                        .into_iter()
                        .next()
                        .expect("one ORDER BY source")
                }
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "ORDER BY \"{}\" is ambiguous",
                        key.column
                    ))))
                }
            };
            if !sources.contains(&source) {
                run_plan
                    .projection
                    .push(crate::engine_expr::JoinProjItem::Column(key.clone()));
                run_plan.projection_aliases.push(None);
                sources.push(source);
            }
        }
        Ok((run_plan, visible))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn merge_materialized_join_run(
        &self,
        accumulator: &mut Option<gpu_db_execution::CudaMaterializedRelation>,
        next: gpu_db_execution::CudaMaterializedRelation,
        plan: &crate::engine_expr::JoinPlan,
        tables: &[RelationalTable],
        columns: &[RelationalColumn],
        top_n: Option<u32>,
        budget: u64,
        peak: &mut u64,
    ) -> Result<(), ExecuteError> {
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let Some(previous) = accumulator.take() else {
            *peak = (*peak).max(next.allocated_bytes());
            if *peak > budget {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "streaming JOIN allocator high-water ({peak}) exceeds the query budget ({budget})"
                ))));
            }
            *accumulator = Some(next);
            return Ok(());
        };
        let live_before = previous
            .allocated_bytes()
            .saturating_add(next.allocated_bytes());
        budget.checked_sub(live_before).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "streaming JOIN allocator live set ({live_before}) exceeds the query budget ({budget})"
            )))
        })?;
        let combined = previous
            .memory()
            .concat_materialized_relations(&previous, &next)
            .map_err(map_err)?;
        drop(previous);
        drop(next);
        let identity = combined
            .memory()
            .identity_join_coordinates(combined.row_count(), None)
            .map_err(map_err)?;
        let keys = self.materialized_join_run_order(plan, tables, columns, &combined)?;
        let ordered = combined
            .memory()
            .sort_join_coordinates(&identity, &keys)
            .map_err(map_err)?;
        let retained = if let Some(limit) = top_n {
            let window = combined
                .memory()
                .window_join_coordinates(&ordered, 0, Some(limit))
                .map_err(map_err)?;
            combined
                .memory()
                .synchronize_default_stream()
                .map_err(map_err)?;
            drop(ordered);
            window
        } else {
            combined
                .memory()
                .synchronize_default_stream()
                .map_err(map_err)?;
            ordered
        };
        let specs = Self::materialized_join_run_specs(&combined);
        let compacted = combined
            .memory()
            .materialize_join_coordinates(&retained, &specs)
            .map_err(map_err)?;
        *peak = (*peak).max(live_before.saturating_add(compacted.allocated_bytes()));
        if *peak > budget {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "streaming JOIN allocator high-water ({peak}) exceeds the query budget ({budget})"
            ))));
        }
        *accumulator = Some(compacted);
        Ok(())
    }

    pub(super) fn merge_unordered_materialized_join_run(
        &self,
        accumulator: &mut Option<gpu_db_execution::CudaMaterializedRelation>,
        next: gpu_db_execution::CudaMaterializedRelation,
        retain: u32,
        budget: u64,
        peak: &mut u64,
    ) -> Result<(), ExecuteError> {
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let previous = accumulator.take();
        let live_before = previous
            .as_ref()
            .map_or(next.allocated_bytes(), |previous| {
                previous
                    .allocated_bytes()
                    .saturating_add(next.allocated_bytes())
            });
        budget.checked_sub(live_before).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "streaming JOIN allocator live set ({live_before}) exceeds the query budget ({budget})"
            )))
        })?;
        let combined = if let Some(previous) = previous {
            let combined = previous
                .memory()
                .concat_materialized_relations(&previous, &next)
                .map_err(map_err)?;
            combined
        } else {
            next
        };
        let retained = if combined.row_count() > retain {
            let identity = combined
                .memory()
                .identity_join_coordinates(combined.row_count(), None)
                .map_err(map_err)?;
            let window = combined
                .memory()
                .window_join_coordinates(&identity, 0, Some(retain))
                .map_err(map_err)?;
            let specs = Self::materialized_join_run_specs(&combined);
            let compacted = combined
                .memory()
                .materialize_join_coordinates(&window, &specs)
                .map_err(map_err)?;
            compacted
        } else {
            combined
        };
        *peak = (*peak).max(live_before.saturating_add(retained.allocated_bytes()));
        if *peak > budget {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "streaming JOIN allocator high-water ({peak}) exceeds the query budget ({budget})"
            ))));
        }
        *accumulator = Some(retained);
        Ok(())
    }

    pub(super) fn decode_materialized_join_run(
        &self,
        run: &gpu_db_execution::CudaMaterializedRelation,
        columns: &[RelationalColumn],
        offset: u32,
        limit: Option<u32>,
    ) -> Result<Vec<Vec<SqlValue>>, ExecuteError> {
        use gpu_db_execution::CudaMaterializedColumnKind;

        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let identity = run
            .memory()
            .identity_join_coordinates(run.row_count(), None)
            .map_err(map_err)?;
        let window = run
            .memory()
            .window_join_coordinates(&identity, offset, limit)
            .map_err(map_err)?;
        let row_count = window.row_count() as usize;
        let mut rows = vec![Vec::with_capacity(columns.len()); row_count];
        for (column_index, column) in columns.iter().enumerate() {
            let layout = run.columns().get(column_index).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "streaming JOIN run schema does not match its result columns".to_string(),
                ))
            })?;
            let values = match layout.kind {
                CudaMaterializedColumnKind::Text => run
                    .memory()
                    .project_text_from_join_coordinates(
                        &window,
                        0,
                        run.memory(),
                        layout.value_byte_offset,
                        layout.text_bytes_byte_offset.expect("text bytes layout"),
                        layout.text_bytes_len,
                        Some(layout.validity_bitmap_offset),
                    )
                    .map_err(map_err)?
                    .into_iter()
                    .map(|value| value.map_or(SqlValue::Null, SqlValue::Text))
                    .collect::<Vec<_>>(),
                CudaMaterializedColumnKind::Fixed { width } => {
                    let (raw, valid) = run
                        .memory()
                        .project_fixed_from_join_coordinates(
                            &window,
                            0,
                            run.memory(),
                            layout.value_byte_offset,
                            Some(layout.validity_bitmap_offset),
                            width,
                        )
                        .map_err(map_err)?;
                    raw.chunks_exact(width as usize)
                        .zip(valid)
                        .map(|(bytes, valid)| {
                            if !valid {
                                return SqlValue::Null;
                            }
                            match column.ty {
                                SqlType::Bool => SqlValue::Bool(
                                    i32::from_le_bytes(bytes.try_into().expect("bool width")) != 0,
                                ),
                                SqlType::Int2 => SqlValue::Int2(i32::from_le_bytes(
                                    bytes.try_into().expect("int2 width"),
                                )
                                    as i16),
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
                                        i128::from_le_bytes(
                                            bytes.try_into().expect("numeric width"),
                                        ),
                                        scale,
                                    ))
                                }
                                SqlType::Uuid => {
                                    SqlValue::Uuid(bytes.try_into().expect("uuid width"))
                                }
                                SqlType::Text => unreachable!(),
                            }
                        })
                        .collect::<Vec<_>>()
                }
            };
            for (row, value) in rows.iter_mut().zip(values) {
                row.push(value);
            }
        }
        Ok(rows)
    }
}
