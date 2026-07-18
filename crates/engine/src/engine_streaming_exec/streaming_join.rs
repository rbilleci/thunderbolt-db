//! Device-resident N-way streaming join orchestration.

use super::*;

type JoinCoordinateEmitter<'a> = dyn FnMut(
        &mut Vec<crate::engine_expr::JoinExecSide>,
        &gpu_db_execution::CudaJoinCoordinatesU32,
    ) -> Result<bool, ExecuteError>
    + 'a;

/// Advance a mixed-radix Cartesian-product cursor. This is scheduler metadata only; it never
/// inspects relation values or decides relational membership.
fn advance_product_cursor(cursor: &mut [usize], extents: &[usize]) -> bool {
    for idx in (0..cursor.len()).rev() {
        cursor[idx] += 1;
        if cursor[idx] < extents[idx] {
            return true;
        }
        cursor[idx] = 0;
    }
    false
}

impl Engine {
    /// Enumerate one complete left-deep prefix without materializing host tuples. RIGHT/FULL
    /// completion replays the preceding prefix once per bounded right chunk while that chunk's
    /// match bitmap is resident. Recursive replay is deliberate: it lets an earlier RIGHT/FULL
    /// complement participate in every later join step while retaining only one bitmap per active
    /// recursion level.
    #[allow(clippy::too_many_arguments)]
    fn stream_nway_prefix(
        &self,
        plan: &crate::engine_expr::JoinPlan,
        tables: &[RelationalTable],
        chunks: &[Vec<&ColdChunk>],
        copin_s: Index,
        block_rows: usize,
        relation_count: usize,
        budget_base: u64,
        budget: u64,
        live_bitmap_bytes: &std::cell::Cell<u64>,
        allocator_peak: &std::cell::Cell<u64>,
        blocks_run: &std::cell::Cell<u64>,
        emit: &mut JoinCoordinateEmitter<'_>,
    ) -> Result<bool, ExecuteError> {
        let account_bitmap = |bytes: u64,
                              live: &std::cell::Cell<u64>,
                              peak: &std::cell::Cell<u64>|
         -> Result<(), ExecuteError> {
            live.set(live.get().saturating_add(bytes));
            let total = budget_base.saturating_add(live.get());
            if total > budget {
                live.set(live.get().saturating_sub(bytes));
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "N-way streaming OUTER bitmap state ({total}) exceeds budget ({budget})"
                ))));
            }
            peak.set(peak.get().max(total));
            Ok(())
        };
        if relation_count == 1 {
            for chunk in &chunks[0] {
                let (source, visibility) = self
                    .stage_cold_chunk(chunk, copin_s)
                    .and_then(StagedChunk::ready)
                    .map_err(|_| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "failed to stage the first relation of an N-way OUTER join".to_string(),
                        ))
                    })?;
                for start in (0..source.row_count as usize).step_by(block_rows) {
                    let end = (start + block_rows).min(source.row_count as usize);
                    let first_side = (
                        RelationalResidencyEntry::new(Arc::clone(&source.descriptor)),
                        crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(
                            &source.device_memory,
                        )),
                        source.row_count as usize,
                        visibility,
                    );
                    let identity = self.resident_join_identity_coordinates(
                        &tables[0],
                        &first_side,
                        Some((start as u32, end as u32)),
                    )?;
                    let mut sides = vec![first_side];
                    if identity.row_count() > 0 && emit(&mut sides, &identity)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }

        let step_index = relation_count - 2;
        let right_relation = relation_count - 1;
        {
            let mut emit_matches = |sides: &mut Vec<crate::engine_expr::JoinExecSide>,
                                    accumulated: &gpu_db_execution::CudaJoinCoordinatesU32|
             -> Result<bool, ExecuteError> {
                let bitmap = sides[0]
                    .1
                    .mem()
                    .create_match_bitmap_u32(accumulated.row_count())
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                let bitmap_bytes = bitmap.allocated_bytes();
                account_bitmap(bitmap_bytes, live_bitmap_bytes, allocator_peak)?;
                let result = (|| {
                    for chunk in &chunks[right_relation] {
                        let (source, visibility) = self
                        .stage_cold_chunk(chunk, copin_s)
                        .and_then(StagedChunk::ready)
                        .map_err(|_| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "failed to stage relation {right_relation} for a streaming OUTER join"
                            )))
                        })?;
                        for start in (0..source.row_count as usize).step_by(block_rows) {
                            let end = (start + block_rows).min(source.row_count as usize);
                            sides.push((
                                RelationalResidencyEntry::new(Arc::clone(&source.descriptor)),
                                crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(
                                    &source.device_memory,
                                )),
                                source.row_count as usize,
                                visibility,
                            ));
                            let matches = self.execute_resident_join_coordinate_matches(
                                plan,
                                tables,
                                sides,
                                step_index,
                                accumulated,
                                &bitmap,
                                None,
                                Some((start as u32, end as u32)),
                            )?;
                            blocks_run.set(blocks_run.get().saturating_add(1));
                            let stop = matches.row_count() > 0 && emit(sides, &matches)?;
                            sides.pop();
                            if stop {
                                return Ok(true);
                            }
                        }
                    }
                    if plan.steps[step_index].outer_left {
                        let unmatched = bitmap
                            .unmatched_extended_coordinates(accumulated)
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?;
                        if unmatched.row_count() > 0 {
                            sides.push(self.resolve_join_side(
                                &tables[right_relation].name,
                                &tables[right_relation],
                                Some(Vec::new()),
                                copin_s,
                            )?);
                            let stop = emit(sides, &unmatched)?;
                            sides.pop();
                            if stop {
                                return Ok(true);
                            }
                        }
                    }
                    Ok(false)
                })();
                live_bitmap_bytes.set(live_bitmap_bytes.get().saturating_sub(bitmap_bytes));
                result
            };
            if self.stream_nway_prefix(
                plan,
                tables,
                chunks,
                copin_s,
                block_rows,
                relation_count - 1,
                budget_base,
                budget,
                live_bitmap_bytes,
                allocator_peak,
                blocks_run,
                &mut emit_matches,
            )? {
                return Ok(true);
            }
        }

        if !plan.steps[step_index].outer_right {
            return Ok(false);
        }
        for chunk in &chunks[right_relation] {
            let (source, visibility) = self
                .stage_cold_chunk(chunk, copin_s)
                .and_then(StagedChunk::ready)
                .map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "failed to restage relation {right_relation} for RIGHT completion"
                    )))
                })?;
            let bitmap = source
                .device_memory
                .create_match_bitmap_u32(source.row_count as u32)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            let bitmap_bytes = bitmap.allocated_bytes();
            account_bitmap(bitmap_bytes, live_bitmap_bytes, allocator_peak)?;
            let replay = (|| {
                let mut mark_matches = |sides: &mut Vec<crate::engine_expr::JoinExecSide>,
                                        accumulated: &gpu_db_execution::CudaJoinCoordinatesU32|
                 -> Result<bool, ExecuteError> {
                    let left_bitmap = sides[0]
                        .1
                        .mem()
                        .create_match_bitmap_u32(accumulated.row_count())
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    let left_bytes = left_bitmap.allocated_bytes();
                    account_bitmap(left_bytes, live_bitmap_bytes, allocator_peak)?;
                    let result = (|| {
                        for start in (0..source.row_count as usize).step_by(block_rows) {
                            let end = (start + block_rows).min(source.row_count as usize);
                            sides.push((
                                RelationalResidencyEntry::new(Arc::clone(&source.descriptor)),
                                crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(
                                    &source.device_memory,
                                )),
                                source.row_count as usize,
                                visibility,
                            ));
                            let matches = self.execute_resident_join_coordinate_matches(
                                plan,
                                tables,
                                sides,
                                step_index,
                                accumulated,
                                &left_bitmap,
                                Some(&bitmap),
                                Some((start as u32, end as u32)),
                            );
                            sides.pop();
                            matches?;
                            blocks_run.set(blocks_run.get().saturating_add(1));
                        }
                        Ok(false)
                    })();
                    live_bitmap_bytes.set(live_bitmap_bytes.get().saturating_sub(left_bytes));
                    result
                };
                self.stream_nway_prefix(
                    plan,
                    tables,
                    chunks,
                    copin_s,
                    block_rows,
                    relation_count - 1,
                    budget_base,
                    budget,
                    live_bitmap_bytes,
                    allocator_peak,
                    blocks_run,
                    &mut mark_matches,
                )?;
                Ok::<(), ExecuteError>(())
            })();
            if let Err(err) = replay {
                live_bitmap_bytes.set(live_bitmap_bytes.get().saturating_sub(bitmap_bytes));
                return Err(err);
            }
            let coordinates = bitmap
                .unmatched_coordinates(relation_count as u32, right_relation as u32)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            live_bitmap_bytes.set(live_bitmap_bytes.get().saturating_sub(bitmap_bytes));
            if coordinates.row_count() == 0 {
                continue;
            }
            let mut sides = Vec::with_capacity(relation_count);
            for table in tables.iter().take(right_relation) {
                sides.push(self.resolve_join_side(
                    &table.name,
                    table,
                    Some(Vec::new()),
                    copin_s,
                )?);
            }
            sides.push((
                RelationalResidencyEntry::new(Arc::clone(&source.descriptor)),
                crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(&source.device_memory)),
                source.row_count as usize,
                visibility,
            ));
            let coordinates = self.filter_join_visibility_coordinates(
                &tables[..=right_relation],
                &sides,
                &coordinates,
            )?;
            if coordinates.row_count() > 0 && emit(&mut sides, &coordinates)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn try_streaming_nway_join(
        &self,
        plan: &crate::engine_expr::JoinPlan,
        tables: &[RelationalTable],
        predicates: &[Option<crate::engine_expr::ResidentExpr>],
        copin_s: Index,
    ) -> Option<Result<RelationalSelectResult, ExecuteError>> {
        let n = plan.relations.len();
        if n < 3
            || plan.steps.len() + 1 != n
            || tables.len() != n
            || predicates.len() != n
            || plan
                .relations
                .iter()
                .any(|r| self.table_device_authoritative(&r.table))
        {
            return None;
        }
        let gpu_id = self.planner.default_gpu_id();
        let budget = self.relational_residency_budget_bytes(gpu_id)?;
        if budget < 256 {
            return None;
        }
        let all_resident = plan.relations.iter().all(|relation| {
            self.relational_residency_entry(&relation.table).is_some()
                || self
                    .read_state
                    .residency
                    .shards
                    .load()
                    .get(&relation.table)
                    .is_some_and(|shards| !shards.is_empty())
        });
        if all_resident {
            return None;
        }
        // All resident inputs together consume <=1/4 budget; a packed variable-width key copy may
        // consume another <=1/4, leaving half for the worst intermediate pair/index state.
        let input_cap = (budget / (4 * n as u64)).max(1);
        let cold: Vec<Arc<ColdTableChunks>> = tables
            .iter()
            .map(|table| self.ensure_streaming_join_cold(table, copin_s, gpu_id, input_cap))
            .collect::<Option<_>>()?;
        let chunks: Vec<Vec<&ColdChunk>> = cold
            .iter()
            .map(|entry| {
                entry
                    .chunks
                    .iter()
                    .filter(|chunk| chunk.payload_copin_s <= copin_s && chunk.row_count > 0)
                    .collect()
            })
            .collect();
        if chunks.iter().flatten().any(|chunk| {
            chunk.row_count > u64::from(u32::MAX) || chunk.snapshot.resident_bytes > input_cap
        }) {
            return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "an N-way streaming join input exceeds its device slice".to_string(),
            ))));
        }

        // Choose the largest B whose worst N-way result and penultimate-key/hash state fit half
        // the query budget: 16B/final tuple (widest fixed gather) + 96B/penultimate tuple/key state.
        let mut block_rows = 1_usize;
        loop {
            let candidate = block_rows.saturating_add(1);
            let final_tuples = candidate.saturating_pow(n as u32);
            let prior_tuples = candidate.saturating_pow((n - 1) as u32);
            let scratch = final_tuples
                .saturating_mul(16)
                .saturating_add(prior_tuples.saturating_add(candidate).saturating_mul(96));
            if scratch as u64 > budget / 2 || candidate == usize::MAX {
                break;
            }
            block_rows = candidate;
        }
        let final_tuples = block_rows.saturating_pow(n as u32);
        let prior_tuples = block_rows.saturating_pow((n - 1) as u32);
        let scratch_peak = final_tuples
            .saturating_mul(16)
            .saturating_add(prior_tuples.saturating_add(block_rows).saturating_mul(96))
            as u64;
        let input_peak: u64 = chunks
            .iter()
            .map(|relation| {
                relation
                    .iter()
                    .map(|chunk| chunk.snapshot.resident_bytes)
                    .max()
                    .unwrap_or(0)
            })
            .sum();
        let planned_peak = input_peak.saturating_mul(2).saturating_add(scratch_peak);
        if planned_peak > budget {
            return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                format!(
            "N-way streaming join live device bytes ({planned_peak}) exceed budget ({budget})"
        ),
            ))));
        }
        let scratch_budget = match budget.checked_sub(input_peak) {
            Some(bytes) => bytes,
            None => {
                return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    format!("N-way streaming join inputs ({input_peak}) exceed budget ({budget})"),
                ))))
            }
        };
        let allocation_scope = gpu_db_execution::CudaAllocationScope::with_budget(scratch_budget);
        self.read_state
            .residency
            .streaming_join_peak_device_bytes
            .fetch_max(planned_peak, Ordering::Relaxed);

        let (run_plan, visible_columns) = match self.join_run_plan(plan, tables) {
            Ok(value) => value,
            Err(err) => return Some(Err(err)),
        };
        let mut chunk_plan = run_plan.clone();
        chunk_plan.limit = None;
        chunk_plan.offset = None;
        chunk_plan.order_by.clear();
        chunk_plan.order_by_nulls_first.clear();
        let collect_bound = plan
            .limit
            .map(|limit| plan.offset.unwrap_or(0).saturating_add(limit));
        let top_n = match collect_bound.map(u32::try_from).transpose() {
            Ok(value) => value,
            Err(_) => {
                return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "N-way streaming JOIN LIMIT/OFFSET exceeds u32".to_string(),
                ))))
            }
        };
        let needs_device_window =
            plan.order_by.is_empty() && (plan.limit.is_some() || plan.offset.unwrap_or(0) > 0);
        let mut output_rows = Vec::new();
        let mut output_columns: Option<Arc<Vec<RelationalColumn>>> = None;
        let mut run_columns: Option<Arc<Vec<RelationalColumn>>> = None;
        let mut accumulator = None;
        let allocator_peak = std::cell::Cell::new(planned_peak);
        let blocks_run = std::cell::Cell::new(0_u64);
        let has_outer = plan
            .steps
            .iter()
            .any(|step| step.outer_left || step.outer_right);
        let chunk_extents: Vec<usize> = chunks.iter().map(Vec::len).collect();
        let have_data = chunk_extents.iter().all(|&extent| extent > 0);
        let mut stop = false;
        if has_outer {
            let mut emit = |result_columns: Arc<Vec<RelationalColumn>>,
                            run: gpu_db_execution::CudaMaterializedRelation|
             -> Result<bool, ExecuteError> {
                if output_columns.is_none() {
                    output_columns = Some(Arc::new(result_columns[..visible_columns].to_vec()));
                    run_columns = Some(Arc::clone(&result_columns));
                }
                if plan.order_by.is_empty() && !needs_device_window {
                    let remaining =
                        top_n.map(|bound| bound.saturating_sub(output_rows.len() as u32));
                    output_rows.extend(self.decode_materialized_join_run(
                        &run,
                        &result_columns[..visible_columns],
                        0,
                        remaining,
                    )?);
                    Ok(top_n.is_some_and(|bound| output_rows.len() >= bound as usize))
                } else if plan.order_by.is_empty() {
                    let mut peak = allocator_peak.get();
                    self.merge_unordered_materialized_join_run(
                        &mut accumulator,
                        run,
                        top_n.unwrap_or(0),
                        budget,
                        &mut peak,
                    )?;
                    allocator_peak.set(peak);
                    Ok(top_n.is_some_and(|bound| {
                        accumulator
                            .as_ref()
                            .is_some_and(|run| run.row_count() >= bound)
                    }))
                } else {
                    let mut peak = allocator_peak.get();
                    self.merge_materialized_join_run(
                        &mut accumulator,
                        run,
                        &run_plan,
                        tables,
                        result_columns.as_slice(),
                        top_n,
                        budget,
                        &mut peak,
                    )?;
                    allocator_peak.set(peak);
                    Ok(false)
                }
            };
            let live_bitmap_bytes = std::cell::Cell::new(0_u64);
            let mut emit_coordinates = |sides: &mut Vec<crate::engine_expr::JoinExecSide>,
                                        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32|
             -> Result<bool, ExecuteError> {
                let filtered = self.filter_outer_join_projection_coordinates(
                    tables,
                    sides,
                    predicates,
                    coordinates,
                )?;
                if filtered.row_count() == 0 {
                    return Ok(false);
                }
                let (columns, run) = self
                    .materialize_join_projection_coordinates(&run_plan, tables, sides, &filtered)?;
                emit(columns, run)
            };
            let outer_early_satisfied = match self.stream_nway_prefix(
                plan,
                tables,
                &chunks,
                copin_s,
                block_rows,
                n,
                planned_peak,
                budget,
                &live_bitmap_bytes,
                &allocator_peak,
                &blocks_run,
                &mut emit_coordinates,
            ) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let _ = outer_early_satisfied;
        } else if have_data {
            let mut chunk_cursor = vec![0_usize; n];
            loop {
                let mut sources = Vec::with_capacity(n);
                for relation in 0..n {
                    let ready = match self
                        .stage_cold_chunk(chunks[relation][chunk_cursor[relation]], copin_s)
                        .and_then(StagedChunk::ready)
                    {
                        Ok(ready) => ready,
                        Err(_) => {
                            return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "failed to stage an N-way streaming join chunk".to_string(),
                            ))))
                        }
                    };
                    sources.push(ready);
                }
                let block_extents: Vec<usize> = sources
                    .iter()
                    .map(|(src, _)| (src.row_count as usize).div_ceil(block_rows))
                    .collect();
                let mut block_cursor = vec![0_usize; n];
                loop {
                    let mut ranges = Vec::with_capacity(n);
                    let mut sides = Vec::with_capacity(n);
                    for relation in 0..n {
                        let (src, visibility) = &sources[relation];
                        let start = block_cursor[relation] * block_rows;
                        let end = (start + block_rows).min(src.row_count as usize);
                        ranges.push((start as u32, end as u32));
                        sides.push((
                            RelationalResidencyEntry::new(Arc::clone(&src.descriptor)),
                            crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(
                                &src.device_memory,
                            )),
                            src.row_count as usize,
                            *visibility,
                        ));
                    }
                    let mut run = None;
                    let result = match self.execute_resident_expr_join_with_device_ranges(
                        &chunk_plan,
                        tables.to_vec(),
                        vec![None; n],
                        predicates.to_vec(),
                        copin_s,
                        Some(sides),
                        ranges,
                        None,
                        Some(&mut run),
                        false,
                    ) {
                        Ok(result) => result,
                        Err(err) => return Some(Err(err)),
                    };
                    if output_columns.is_none() {
                        output_columns = Some(Arc::new(result.columns[..visible_columns].to_vec()));
                        run_columns = Some(Arc::clone(&result.columns));
                    }
                    let Some(run) = run else {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "N-way device JOIN did not return its materialized run".to_string(),
                        ))));
                    };
                    if plan.order_by.is_empty() && !needs_device_window {
                        let remaining =
                            top_n.map(|bound| bound.saturating_sub(output_rows.len() as u32));
                        match self.decode_materialized_join_run(
                            &run,
                            &result.columns[..visible_columns],
                            0,
                            remaining,
                        ) {
                            Ok(rows) => output_rows.extend(rows),
                            Err(err) => return Some(Err(err)),
                        }
                    } else if plan.order_by.is_empty() {
                        let mut peak = allocator_peak.get();
                        if let Err(err) = self.merge_unordered_materialized_join_run(
                            &mut accumulator,
                            run,
                            top_n.unwrap_or(0),
                            budget,
                            &mut peak,
                        ) {
                            return Some(Err(err));
                        }
                        allocator_peak.set(peak);
                    } else {
                        let mut peak = allocator_peak.get();
                        if let Err(err) = self.merge_materialized_join_run(
                            &mut accumulator,
                            run,
                            &run_plan,
                            tables,
                            result.columns.as_slice(),
                            top_n,
                            budget,
                            &mut peak,
                        ) {
                            return Some(Err(err));
                        }
                        allocator_peak.set(peak);
                    }
                    blocks_run.set(blocks_run.get().saturating_add(1));
                    if needs_device_window
                        && top_n.is_some_and(|bound| {
                            accumulator
                                .as_ref()
                                .is_some_and(|run| run.row_count() >= bound)
                        })
                    {
                        stop = true;
                        break;
                    }
                    if !advance_product_cursor(&mut block_cursor, &block_extents) {
                        break;
                    }
                }
                if stop || !advance_product_cursor(&mut chunk_cursor, &chunk_extents) {
                    break;
                }
            }
        }
        if output_columns.is_none() {
            let sides = tables
                .iter()
                .map(|table| self.resolve_join_side(&table.name, table, Some(Vec::new()), copin_s))
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            let mut run = None;
            let result = match self.execute_resident_expr_join_with_device_run(
                &chunk_plan,
                tables.to_vec(),
                vec![None; n],
                predicates.to_vec(),
                copin_s,
                Some(sides),
                None,
                Some(&mut run),
                false,
            ) {
                Ok(result) => result,
                Err(err) => return Some(Err(err)),
            };
            output_columns = Some(Arc::new(result.columns[..visible_columns].to_vec()));
            run_columns = Some(result.columns);
        }
        let columns = output_columns.expect("N-way join schema resolved");
        let materialized_columns = run_columns.expect("N-way materialized schema resolved");
        if !plan.order_by.is_empty() {
            if let Some(run) = accumulator {
                let identity = match run
                    .memory()
                    .identity_join_coordinates(run.row_count(), None)
                {
                    Ok(value) => value,
                    Err(err) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.to_string(),
                        ))))
                    }
                };
                let keys = match self.materialized_join_run_order(
                    &run_plan,
                    tables,
                    materialized_columns.as_slice(),
                    &run,
                ) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                let ordered = match run.memory().sort_join_coordinates(&identity, &keys) {
                    Ok(value) => value,
                    Err(err) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.to_string(),
                        ))))
                    }
                };
                let specs = Self::materialized_join_run_specs(&run);
                let sorted = match run.memory().materialize_join_coordinates(&ordered, &specs) {
                    Ok(value) => value,
                    Err(err) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.to_string(),
                        ))))
                    }
                };
                output_rows = match self.decode_materialized_join_run(
                    &sorted,
                    columns.as_slice(),
                    u32::try_from(plan.offset.unwrap_or(0)).unwrap_or(u32::MAX),
                    plan.limit.and_then(|value| u32::try_from(value).ok()),
                ) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
            }
        } else if needs_device_window {
            if let Some(run) = accumulator {
                output_rows = match self.decode_materialized_join_run(
                    &run,
                    columns.as_slice(),
                    u32::try_from(plan.offset.unwrap_or(0)).unwrap_or(u32::MAX),
                    plan.limit.and_then(|value| u32::try_from(value).ok()),
                ) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
            }
        }
        let allocator_high_water = input_peak.saturating_add(allocation_scope.peak_bytes());
        self.read_state
            .residency
            .streaming_join_peak_device_bytes
            .fetch_max(allocator_high_water, Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_join_hits
            .fetch_add(1, Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_join_block_pairs
            .fetch_add(blocks_run.get(), Ordering::Relaxed);
        Some(Ok(RelationalSelectResult {
            columns,
            rows: output_rows.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        }))
    }

    /// ADR-012 two-relation join: stage one cold chunk from each side, then split each pair
    /// into bounded logical row blocks before invoking the existing GPU hash join. CONCAT across
    /// disjoint block pairs is control-plane combine; predicate, MVCC visibility, key equality,
    /// NULL handling, and result gathers remain device-executed. OUTER joins run the match phase as
    /// INNER while recording device-approved coordinates, then device-gather unmatched rows once.
    pub(crate) fn try_streaming_inner_join(
        &self,
        plan: &crate::engine_expr::JoinPlan,
        tables: &[RelationalTable],
        predicates: &[Option<crate::engine_expr::ResidentExpr>],
        copin_s: Index,
    ) -> Option<Result<RelationalSelectResult, ExecuteError>> {
        if plan.relations.len() > 2 {
            return self.try_streaming_nway_join(plan, tables, predicates, copin_s);
        }
        if plan.relations.len() != 2
            || plan.steps.len() != 1
            || tables.len() != 2
            || predicates.len() != 2
        {
            return None;
        }
        let gpu_id = self.planner.default_gpu_id();
        let budget = self.relational_residency_budget_bytes(gpu_id)?;
        if budget < 128 {
            return None;
        }
        let all_resident = plan.relations.iter().all(|relation| {
            self.relational_residency_entry(&relation.table).is_some()
                || self
                    .read_state
                    .residency
                    .shards
                    .load()
                    .get(&relation.table)
                    .is_some_and(|shards| !shards.is_empty())
        });
        if all_resident
            || plan
                .relations
                .iter()
                .any(|r| self.table_device_authoritative(&r.table))
        {
            return None;
        }
        let input_cap = (budget / 8).max(1);
        let left = self.ensure_streaming_join_cold(&tables[0], copin_s, gpu_id, input_cap)?;
        let right = self.ensure_streaming_join_cold(&tables[1], copin_s, gpu_id, input_cap)?;
        let left_chunks = left
            .chunks
            .iter()
            .filter(|chunk| chunk.payload_copin_s <= copin_s && chunk.row_count > 0)
            .collect::<Vec<_>>();
        let right_chunks = right
            .chunks
            .iter()
            .filter(|chunk| chunk.payload_copin_s <= copin_s && chunk.row_count > 0)
            .collect::<Vec<_>>();
        if left_chunks.iter().chain(right_chunks.iter()).any(|chunk| {
            chunk.row_count > u64::from(u32::MAX) || chunk.snapshot.resident_bytes > input_cap
        }) {
            return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a streaming JOIN input chunk exceeds its device slice or u32 row range"
                    .to_string(),
            ))));
        }

        let is_outer = plan.steps[0].outer_left || plan.steps[0].outer_right;
        let (run_plan, visible_columns) = match self.join_run_plan(plan, tables) {
            Ok(value) => value,
            Err(err) => return Some(Err(err)),
        };
        let mut pair_plan = run_plan.clone();
        pair_plan.limit = None;
        pair_plan.offset = None;
        pair_plan.order_by.clear();
        pair_plan.order_by_nulls_first.clear();
        pair_plan.steps[0].outer_left = false;
        pair_plan.steps[0].outer_right = false;

        let max_pairs = usize::try_from((budget / 4 / 8).max(1)).ok()?;
        let mut block_rows = (max_pairs as f64).sqrt() as usize;
        block_rows = block_rows.max(1);
        while block_rows.saturating_mul(block_rows) > max_pairs {
            block_rows -= 1;
        }
        while block_rows > 1
            && (block_rows
                .saturating_mul(block_rows)
                .saturating_mul(8)
                .saturating_add(block_rows.saturating_mul(96)) as u64)
                > budget.saturating_mul(3) / 8
        {
            block_rows -= 1;
        }
        let input_peak = left_chunks
            .iter()
            .map(|chunk| chunk.snapshot.resident_bytes)
            .max()
            .unwrap_or(0)
            .saturating_add(
                right_chunks
                    .iter()
                    .map(|chunk| chunk.snapshot.resident_bytes)
                    .max()
                    .unwrap_or(0),
            );
        let bitmap_peak = if is_outer {
            left_chunks
                .iter()
                .chain(right_chunks.iter())
                .map(|chunk| chunk.row_count.saturating_mul(4))
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        let scratch_peak = (block_rows
            .saturating_mul(block_rows)
            .saturating_mul(8)
            .saturating_add(block_rows.saturating_mul(96))) as u64;
        let planned_peak = input_peak
            .saturating_mul(2)
            .saturating_add(bitmap_peak)
            .saturating_add(scratch_peak);
        if planned_peak > budget {
            return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                format!(
            "streaming JOIN live device bytes ({planned_peak}) exceed the query budget ({budget})"
        ),
            ))));
        }
        let scratch_budget = match budget.checked_sub(input_peak) {
            Some(bytes) => bytes,
            None => {
                return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    format!("streaming JOIN inputs ({input_peak}) exceed budget ({budget})"),
                ))))
            }
        };
        let allocation_scope = gpu_db_execution::CudaAllocationScope::with_budget(scratch_budget);
        let collect_bound = plan
            .limit
            .map(|limit| plan.offset.unwrap_or(0).saturating_add(limit));
        let top_n = match collect_bound.map(u32::try_from).transpose() {
            Ok(value) => value,
            Err(_) => {
                return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "streaming JOIN LIMIT/OFFSET exceeds the device coordinate range".to_string(),
                ))))
            }
        };
        let needs_device_window =
            plan.order_by.is_empty() && (plan.limit.is_some() || plan.offset.unwrap_or(0) > 0);
        let mut output_rows = Vec::<Vec<SqlValue>>::new();
        let mut output_columns: Option<Arc<Vec<RelationalColumn>>> = None;
        let mut run_columns: Option<Arc<Vec<RelationalColumn>>> = None;
        let mut accumulator = None;
        let mut allocator_peak = planned_peak;
        let mut blocks_run = 0_u64;
        let mut satisfied = top_n == Some(0) && plan.order_by.is_empty();

        let mut consume = |columns: Arc<Vec<RelationalColumn>>,
                           run: gpu_db_execution::CudaMaterializedRelation|
         -> Result<bool, ExecuteError> {
            if output_columns.is_none() {
                output_columns = Some(Arc::new(columns[..visible_columns].to_vec()));
                run_columns = Some(Arc::clone(&columns));
            }
            if !plan.order_by.is_empty() {
                self.merge_materialized_join_run(
                    &mut accumulator,
                    run,
                    &run_plan,
                    tables,
                    columns.as_slice(),
                    top_n,
                    budget,
                    &mut allocator_peak,
                )?;
                return Ok(false);
            }
            if needs_device_window {
                self.merge_unordered_materialized_join_run(
                    &mut accumulator,
                    run,
                    top_n.unwrap_or(0),
                    budget,
                    &mut allocator_peak,
                )?;
                return Ok(top_n.is_some_and(|bound| {
                    accumulator
                        .as_ref()
                        .is_some_and(|run| run.row_count() >= bound)
                }));
            }
            let decoded = self.decode_materialized_join_run(
                &run,
                &columns[..visible_columns],
                0,
                top_n.map(|bound| bound.saturating_sub(output_rows.len() as u32)),
            )?;
            output_rows.extend(decoded);
            if let Some(bound) = top_n {
                output_rows.truncate(bound as usize);
                return Ok(output_rows.len() >= bound as usize);
            }
            Ok(false)
        };

        'left_chunks: for left_chunk in &left_chunks {
            if satisfied {
                break;
            }
            let (left_src, left_vis) = match self
                .stage_cold_chunk(left_chunk, copin_s)
                .and_then(StagedChunk::ready)
            {
                Ok(value) => value,
                Err(_) => {
                    return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "failed to stage the left streaming JOIN chunk".to_string(),
                    ))))
                }
            };
            let left_bitmap = if plan.steps[0].outer_left {
                match left_src
                    .device_memory
                    .create_match_bitmap_u32(left_src.row_count as u32)
                {
                    Ok(value) => Some(value),
                    Err(err) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.to_string(),
                        ))))
                    }
                }
            } else {
                None
            };
            for right_chunk in &right_chunks {
                let (right_src, right_vis) = match self
                    .stage_cold_chunk(right_chunk, copin_s)
                    .and_then(StagedChunk::ready)
                {
                    Ok(value) => value,
                    Err(_) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "failed to stage the right streaming JOIN chunk".to_string(),
                        ))))
                    }
                };
                for left_start in (0..left_src.row_count as usize).step_by(block_rows) {
                    let left_end = (left_start + block_rows).min(left_src.row_count as usize);
                    for right_start in (0..right_src.row_count as usize).step_by(block_rows) {
                        let right_end =
                            (right_start + block_rows).min(right_src.row_count as usize);
                        let sides = vec![
                            (
                                RelationalResidencyEntry::new(Arc::clone(&left_src.descriptor)),
                                crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(
                                    &left_src.device_memory,
                                )),
                                left_src.row_count as usize,
                                left_vis,
                            ),
                            (
                                RelationalResidencyEntry::new(Arc::clone(&right_src.descriptor)),
                                crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(
                                    &right_src.device_memory,
                                )),
                                right_src.row_count as usize,
                                right_vis,
                            ),
                        ];
                        let mut coordinates = None;
                        let mut run = None;
                        let result = match self.execute_resident_expr_join_with_device_ranges(
                            &pair_plan,
                            tables.to_vec(),
                            vec![None, None],
                            predicates.to_vec(),
                            copin_s,
                            Some(sides),
                            vec![
                                (left_start as u32, left_end as u32),
                                (right_start as u32, right_end as u32),
                            ],
                            Some(&mut coordinates),
                            Some(&mut run),
                            is_outer,
                        ) {
                            Ok(value) => value,
                            Err(err) => return Some(Err(err)),
                        };
                        if let (Some(bitmap), Some(coordinates)) =
                            (left_bitmap.as_ref(), coordinates.as_ref())
                        {
                            if let Err(err) = bitmap.mark_coordinates(coordinates, 0) {
                                return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                    err.to_string(),
                                ))));
                            }
                        }
                        blocks_run += 1;
                        let Some(run) = run else {
                            return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "device JOIN did not return its materialized run".to_string(),
                            ))));
                        };
                        match consume(result.columns, run) {
                            Ok(done) => satisfied = done,
                            Err(err) => return Some(Err(err)),
                        }
                        if satisfied {
                            break 'left_chunks;
                        }
                    }
                }
            }
            if let Some(bitmap) = left_bitmap {
                let coordinates = match bitmap.unmatched_coordinates(2, 0) {
                    Ok(value) => value,
                    Err(err) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.to_string(),
                        ))))
                    }
                };
                let empty_right = self
                    .resolve_join_side(&tables[1].name, &tables[1], Some(Vec::new()), copin_s)
                    .ok()?;
                let sides = vec![
                    (
                        RelationalResidencyEntry::new(Arc::clone(&left_src.descriptor)),
                        crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(
                            &left_src.device_memory,
                        )),
                        left_src.row_count as usize,
                        left_vis,
                    ),
                    empty_right,
                ];
                let coordinates = match self.filter_outer_join_projection_coordinates(
                    tables,
                    &sides,
                    predicates,
                    &coordinates,
                ) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                let (columns, run) = match self.materialize_join_projection_coordinates(
                    &run_plan,
                    tables,
                    &sides,
                    &coordinates,
                ) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                match consume(columns, run) {
                    Ok(done) => satisfied = done,
                    Err(err) => return Some(Err(err)),
                }
            }
        }

        if plan.steps[0].outer_right && !satisfied {
            'right_chunks: for right_chunk in &right_chunks {
                let (right_src, right_vis) = match self
                    .stage_cold_chunk(right_chunk, copin_s)
                    .and_then(StagedChunk::ready)
                {
                    Ok(value) => value,
                    Err(_) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "failed to stage a right OUTER JOIN chunk".to_string(),
                        ))))
                    }
                };
                let bitmap = match right_src
                    .device_memory
                    .create_match_bitmap_u32(right_src.row_count as u32)
                {
                    Ok(value) => value,
                    Err(err) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.to_string(),
                        ))))
                    }
                };
                for left_chunk in &left_chunks {
                    let (left_src, left_vis) = match self
                        .stage_cold_chunk(left_chunk, copin_s)
                        .and_then(StagedChunk::ready)
                    {
                        Ok(value) => value,
                        Err(_) => {
                            return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "failed to stage a left OUTER match chunk".to_string(),
                            ))))
                        }
                    };
                    for left_start in (0..left_src.row_count as usize).step_by(block_rows) {
                        let left_end = (left_start + block_rows).min(left_src.row_count as usize);
                        for right_start in (0..right_src.row_count as usize).step_by(block_rows) {
                            let right_end =
                                (right_start + block_rows).min(right_src.row_count as usize);
                            let sides = vec![
                                (
                                    RelationalResidencyEntry::new(Arc::clone(&left_src.descriptor)),
                                    crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(
                                        &left_src.device_memory,
                                    )),
                                    left_src.row_count as usize,
                                    left_vis,
                                ),
                                (
                                    RelationalResidencyEntry::new(Arc::clone(
                                        &right_src.descriptor,
                                    )),
                                    crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(
                                        &right_src.device_memory,
                                    )),
                                    right_src.row_count as usize,
                                    right_vis,
                                ),
                            ];
                            let mut coordinates = None;
                            if let Err(err) = self.execute_resident_expr_join_with_device_ranges(
                                &pair_plan,
                                tables.to_vec(),
                                vec![None, None],
                                predicates.to_vec(),
                                copin_s,
                                Some(sides),
                                vec![
                                    (left_start as u32, left_end as u32),
                                    (right_start as u32, right_end as u32),
                                ],
                                Some(&mut coordinates),
                                None,
                                is_outer,
                            ) {
                                return Some(Err(err));
                            }
                            if let Some(coordinates) = coordinates.as_ref() {
                                if let Err(err) = bitmap.mark_coordinates(coordinates, 1) {
                                    return Some(Err(ExecuteError::Engine(
                                        EngineError::ApplyFailed(err.to_string()),
                                    )));
                                }
                            }
                            blocks_run += 1;
                        }
                    }
                }
                let coordinates = match bitmap.unmatched_coordinates(2, 1) {
                    Ok(value) => value,
                    Err(err) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.to_string(),
                        ))))
                    }
                };
                let empty_left = self
                    .resolve_join_side(&tables[0].name, &tables[0], Some(Vec::new()), copin_s)
                    .ok()?;
                let sides = vec![
                    empty_left,
                    (
                        RelationalResidencyEntry::new(Arc::clone(&right_src.descriptor)),
                        crate::engine_expr::JoinDeviceMemory::Resident(Arc::clone(
                            &right_src.device_memory,
                        )),
                        right_src.row_count as usize,
                        right_vis,
                    ),
                ];
                let coordinates = match self.filter_outer_join_projection_coordinates(
                    tables,
                    &sides,
                    predicates,
                    &coordinates,
                ) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                let (columns, run) = match self.materialize_join_projection_coordinates(
                    &run_plan,
                    tables,
                    &sides,
                    &coordinates,
                ) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                match consume(columns, run) {
                    Ok(done) => satisfied = done,
                    Err(err) => return Some(Err(err)),
                }
                if satisfied {
                    break 'right_chunks;
                }
            }
        }

        if output_columns.is_none() {
            let sides = vec![
                self.resolve_join_side(&tables[0].name, &tables[0], Some(Vec::new()), copin_s)
                    .ok()?,
                self.resolve_join_side(&tables[1].name, &tables[1], Some(Vec::new()), copin_s)
                    .ok()?,
            ];
            let mut run = None;
            let result = match self.execute_resident_expr_join_with_device_run(
                &pair_plan,
                tables.to_vec(),
                vec![None, None],
                predicates.to_vec(),
                copin_s,
                Some(sides),
                None,
                Some(&mut run),
                false,
            ) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            output_columns = Some(Arc::new(result.columns[..visible_columns].to_vec()));
            run_columns = Some(result.columns);
        }
        let columns = output_columns.expect("streaming JOIN schema resolved");
        let materialized_columns =
            run_columns.expect("streaming JOIN materialized schema resolved");
        if !plan.order_by.is_empty() {
            if let Some(run) = accumulator {
                let identity = match run
                    .memory()
                    .identity_join_coordinates(run.row_count(), None)
                {
                    Ok(value) => value,
                    Err(err) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.to_string(),
                        ))))
                    }
                };
                let keys = match self.materialized_join_run_order(
                    &run_plan,
                    tables,
                    materialized_columns.as_slice(),
                    &run,
                ) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                let ordered = match run.memory().sort_join_coordinates(&identity, &keys) {
                    Ok(value) => value,
                    Err(err) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.to_string(),
                        ))))
                    }
                };
                let specs = Self::materialized_join_run_specs(&run);
                let sorted_run = match run.memory().materialize_join_coordinates(&ordered, &specs) {
                    Ok(value) => value,
                    Err(err) => {
                        return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            err.to_string(),
                        ))))
                    }
                };
                output_rows = match self.decode_materialized_join_run(
                    &sorted_run,
                    columns.as_slice(),
                    u32::try_from(plan.offset.unwrap_or(0)).unwrap_or(u32::MAX),
                    plan.limit.and_then(|value| u32::try_from(value).ok()),
                ) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
            }
        } else if needs_device_window {
            if let Some(run) = accumulator {
                output_rows = match self.decode_materialized_join_run(
                    &run,
                    columns.as_slice(),
                    u32::try_from(plan.offset.unwrap_or(0)).unwrap_or(u32::MAX),
                    plan.limit.and_then(|value| u32::try_from(value).ok()),
                ) {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
            }
        }
        let allocator_high_water = input_peak.saturating_add(allocation_scope.peak_bytes());
        self.read_state
            .residency
            .streaming_join_peak_device_bytes
            .fetch_max(allocator_high_water, Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_join_hits
            .fetch_add(1, Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_join_block_pairs
            .fetch_add(blocks_run, Ordering::Relaxed);
        Some(Ok(RelationalSelectResult {
            columns,
            rows: output_rows.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        }))
    }
}
