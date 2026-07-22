use super::{
    Arc, Engine, EngineError, ExecuteError, Instant, RelationalPointBatchResult,
    RelationalRetainedInt4ProjectionSubmission, RelationalRetainedReadSubmission,
    RelationalRetainedReadSubmissionInner, RelationalRetainedReadTemplate, Select,
};

impl Engine {
    /// Prepare the **needle-invariant** template for a batchable int4-equality point lookup — the plan,
    /// binding, result columns, and MVCC access path that every needle of this shape shares. Built ONCE
    /// per shape; `submit_relational_retained_template_point_lookups` then reuses it across all needles
    /// with no per-request re-plan/re-bind (removing the ~15µs/item coalescer cost — DECISIONS ADR-008).
    /// Reuses `prepare_relational_retained_read_job`'s validation, then binds once more for the result
    /// columns + access path (amortized across the whole batch).
    pub fn prepare_relational_retained_read_template(
        &self,
        select: &Select,
    ) -> Result<RelationalRetainedReadTemplate, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let _table_access = self.acquire_autocommit_table_access(&select.table)?;
        let job = self.prepare_relational_retained_read_job(select)?;
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if filter_groups.len() != 1 || filter_groups[0].len() != 1 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "retained read template requires one equality predicate".to_string(),
            )));
        }
        let filter_idx = filter_groups[0][0].0;
        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let projected_column_is_null_bearing = |name: &str| {
            self.read_residency_shards()
                .get(&table.name)
                .is_some_and(|shards| {
                    shards.iter().any(|shard| {
                        shard
                            .resident_device_null_columns
                            .iter()
                            .any(|layout| layout.name == name)
                    })
                })
                || self
                    .relational_retained_snapshot_handle(&table.name)
                    .is_some_and(|snapshot| {
                        snapshot
                            .resident_device_null_columns
                            .iter()
                            .any(|layout| layout.name == name)
                    })
        };
        let flat_i32_projection_null_free = bound
            .selected_columns
            .iter()
            .all(|column| !projected_column_is_null_bearing(&column.name));
        Ok(RelationalRetainedReadTemplate {
            route_id: job.route_id,
            schema: job.schema,
            snapshot_generation: job.snapshot_generation,
            selected_indexes: bound.selected_indexes,
            result_columns: bound.selected_columns,
            flat_i32_projection_null_free,
            filter_idx,
            access_path,
            table,
            select: select.clone(),
        })
    }

    /// Submit a whole batch of point lookups for one prepared `template`, varying only the `needles`.
    /// Validates the resident snapshot generation against the template ONCE (no per-needle re-plan or
    /// re-bind), launches the single `equal_any` GPU submission for all needles, and stamps every
    /// per-needle result with the template's shared `result_columns` + `access_path`. This is the
    /// per-batch host path the coalescer (and the future wave engine) drive.
    pub fn submit_relational_retained_template_point_lookups(
        &self,
        template: &RelationalRetainedReadTemplate,
        needles: &[i32],
    ) -> Result<RelationalRetainedReadSubmission, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let table_access = Some(self.acquire_autocommit_table_access(&template.table.name)?);
        let submit_started = Instant::now();
        if needles.is_empty() {
            return Ok(RelationalRetainedReadSubmission {
                route_id: template.route_id.clone(),
                table: template.table.name.clone(),
                snapshot_generation: template.snapshot_generation,
                job_count: 0,
                submit_wall_micros: submit_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
                commit_path_wedged: Arc::clone(&self.commit_path_wedged),
                table_access,
                inner: RelationalRetainedReadSubmissionInner::Ready(Vec::new()),
            });
        }
        let query_shape = template.route_id.split(':').next().unwrap_or("unknown");
        if matches!(
            query_shape,
            "sharded_int4_equality_projection"
                | "sharded_int4_equality_multi_column_projection"
                | "sharded_int4_equality_mixed_column_projection"
        ) {
            let current_generation = self.committed_seq();
            if current_generation != template.snapshot_generation {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "retained sharded read template snapshot generation mismatch for relation \"{}\": template={}, current={}",
                    template.table.name, template.snapshot_generation, current_generation
                ))));
            }
            if template.is_int4_only_projection() {
                if let Some(projected) = self.gather_sharded_int4_point_lookups_batched(
                    current_generation,
                    &template.table,
                    template.filter_idx,
                    &template.selected_indexes,
                    needles,
                )? {
                    let gpu_id = self
                        .read_residency_shards()
                        .get(&template.table.name)
                        .and_then(|shards| shards.first())
                        .map(|shard| shard.gpu_id)
                        .unwrap_or(0);
                    return Ok(RelationalRetainedReadSubmission {
                        route_id: template.route_id.clone(),
                        table: template.table.name.clone(),
                        snapshot_generation: template.snapshot_generation,
                        job_count: needles.len(),
                        submit_wall_micros: submit_started
                            .elapsed()
                            .as_micros()
                            .try_into()
                            .unwrap_or(u64::MAX),
                        commit_path_wedged: Arc::clone(&self.commit_path_wedged),
                        table_access,
                        inner: RelationalRetainedReadSubmissionInner::ReadyBatched(Box::new(
                            RelationalPointBatchResult::new(
                                Arc::new(template.result_columns.clone()),
                                Arc::new(template.access_path.clone()),
                                gpu_id,
                                projected.values,
                                projected.ncols,
                                projected.needle_ranges,
                            )
                            .into_compat(),
                        )),
                    });
                }
            }
            let filter_column = &template.table.columns[template.filter_idx].name;
            let results = needles
                .iter()
                .map(|&needle| {
                    let mut select = template.select.clone();
                    if let Some(filter) = &mut select.filter {
                        if filter.column == *filter_column {
                            filter.value = gpu_db_sql::SqlValue::Int4(needle);
                        }
                    }
                    for filter in &mut select.filters {
                        if filter.column == *filter_column {
                            filter.value = gpu_db_sql::SqlValue::Int4(needle);
                        }
                    }
                    for filter in select.filter_groups.iter_mut().flatten() {
                        if filter.column == *filter_column {
                            filter.value = gpu_db_sql::SqlValue::Int4(needle);
                        }
                    }
                    self.execute_relational_select(&select)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(RelationalRetainedReadSubmission {
                route_id: template.route_id.clone(),
                table: template.table.name.clone(),
                snapshot_generation: template.snapshot_generation,
                job_count: needles.len(),
                submit_wall_micros: submit_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
                commit_path_wedged: Arc::clone(&self.commit_path_wedged),
                table_access,
                inner: RelationalRetainedReadSubmissionInner::Ready(results),
            });
        }
        // Validate the resident snapshot still matches the generation the template was prepared against
        // (the per-job equivalent of the jobs-path handle checks, done once for the whole batch).
        let handle = self
            .relational_retained_snapshot_handle(&template.table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained snapshot handle",
                    template.table.name
                )))
            })?;
        if handle.generation != template.snapshot_generation {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "retained read template snapshot generation mismatch for relation \"{}\": template={}, current={}",
                template.table.name, template.snapshot_generation, handle.generation
            ))));
        }
        if !handle.valid {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "retained read template snapshot handle for relation \"{}\" is invalid",
                template.table.name
            ))));
        }
        if !handle.has_retained_device_memory {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "retained read template snapshot handle for relation \"{}\" has no device memory",
                template.table.name
            ))));
        }
        let (snapshot_gpu_id, before_metrics, batch_started, payload) = self
            .submit_resident_int4_equal_any_payload(
                &template.table,
                &template.selected_indexes,
                template.filter_idx,
                needles,
            )?
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" resident snapshot became unusable between prepare and submit",
                    template.table.name
                )))
            })?;
        // The projected schema (columns, access_path) is needle-invariant + `Arc`-SHARED: built ONCE here,
        // refcount-cloned by the completion — NOT deep-cloned N times, and NO per-needle Vec (DECISIONS
        // "Result-path optimization"). The hot batched completion needs only `needle_count` + this schema.
        let shared_columns = Arc::new(template.result_columns.clone());
        let shared_access_path = Arc::new(template.access_path.clone());
        Ok(RelationalRetainedReadSubmission {
            route_id: template.route_id.clone(),
            table: template.table.name.clone(),
            snapshot_generation: template.snapshot_generation,
            job_count: needles.len(),
            submit_wall_micros: submit_started
                .elapsed()
                .as_micros()
                .try_into()
                .unwrap_or(u64::MAX),
            commit_path_wedged: Arc::clone(&self.commit_path_wedged),
            table_access,
            inner: RelationalRetainedReadSubmissionInner::PendingInt4Projection(Box::new(
                RelationalRetainedInt4ProjectionSubmission {
                    table: template.table.clone(),
                    snapshot_gpu_id,
                    selected_indexes: template.selected_indexes.clone(),
                    needle_count: needles.len(),
                    shared_columns,
                    shared_access_path,
                    before_metrics,
                    batch_started,
                    payload,
                },
            )),
        })
    }
}
