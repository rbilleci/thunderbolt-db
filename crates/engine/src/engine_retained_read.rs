//! Retained-read async job lifecycle (P0 §9.6 decomposition, behavior-preserving):
//! a focused `impl Engine` block for preparing, submitting, and completing
//! retained relational read jobs against resident device memory — including the
//! int4-projection retained submissions (try_submit / complete[_detached]) that
//! let a caller hold a device read view across calls.

use super::*;

impl Engine {
    // Stage-0 (Thread-3 batched/async submission): this takes `&self`, not `&mut self`.
    // Its body only calls `plan_relational_resident_route`, `bind_relational_select_for_execution`,
    // and `relational_retained_snapshot_handle` — all `&self` — so job preparation needs no
    // exclusive access. Flipping to `&self` lets the façade build a whole batch of jobs under a
    // single shared read lock (the "one read-lock per batch" invariant), exactly as the `&self`
    // read path `execute_relational_select` already does.
    pub fn prepare_relational_retained_read_job(
        &self,
        select: &Select,
    ) -> Result<RelationalRetainedReadJob, ExecuteError> {
        let decision = self.plan_relational_resident_route(select);
        if !decision.accepted {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "retained read job rejected: {}",
                decision.reason
            ))));
        }
        if !matches!(
            decision.query_shape.as_str(),
            "int4_equality_projection"
                | "int4_equality_multi_column_projection"
                | "int4_equality_mixed_column_projection"
        ) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "retained read jobs currently support only int4 equality projection routes, got {}",
                decision.query_shape
            ))));
        }
        let (table, bound, _copin_s) = self.bind_relational_select_for_execution(select)?;
        let handle = self
            .relational_retained_snapshot_handle(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained snapshot handle",
                    table.name
                )))
            })?;
        if !handle.valid {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" retained snapshot handle is invalid",
                table.name
            ))));
        }
        if !handle.has_retained_device_memory {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" retained snapshot handle has no device memory",
                table.name
            ))));
        }
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
                "retained read jobs currently require one equality predicate".to_string(),
            )));
        }
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "retained read jobs currently require an int4 equality parameter".to_string(),
            )));
        };
        if op != SelectFilterOp::Eq || table.columns[filter_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "retained read jobs currently require an int4 equality parameter".to_string(),
            )));
        }
        let projection_columns = bound
            .selected_indexes
            .iter()
            .map(|idx| table.columns[*idx].name.clone())
            .collect::<Vec<_>>();
        let filter_column = table.columns[filter_idx].name.clone();
        let route_id = format!(
            "{}:{}:{}:{}:{}",
            decision.query_shape,
            table.schema,
            table.name,
            projection_columns.join(","),
            filter_column
        );
        Ok(RelationalRetainedReadJob {
            route_id,
            schema: table.schema,
            table: table.name,
            snapshot_generation: handle.generation,
            params: vec![RelationalRetainedReadParam::Int4Eq {
                column: filter_column,
                value: needle,
            }],
            select: select.clone(),
        })
    }

    pub fn execute_relational_retained_read_jobs_with_resident_device_memory_probe(
        &self,
        jobs: &[RelationalRetainedReadJob],
    ) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
        let submission =
            self.submit_relational_retained_read_jobs_with_resident_device_memory_probe(jobs)?;
        self.complete_relational_retained_read_submission(submission)
    }

    pub fn submit_relational_retained_read_jobs_with_resident_device_memory_probe(
        &self,
        jobs: &[RelationalRetainedReadJob],
    ) -> Result<RelationalRetainedReadSubmission, ExecuteError> {
        for job in jobs {
            let handle = self
                .relational_retained_snapshot_handle(&job.table)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" has no retained snapshot handle",
                        job.table
                    )))
                })?;
            if handle.generation != job.snapshot_generation {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "retained read job snapshot generation mismatch for relation \"{}\": job={}, current={}",
                    job.table, job.snapshot_generation, handle.generation
                ))));
            }
            if !handle.valid {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "retained read job snapshot handle for relation \"{}\" is invalid",
                    job.table
                ))));
            }
            if !handle.has_retained_device_memory {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "retained read job snapshot handle for relation \"{}\" has no device memory",
                    job.table
                ))));
            }
        }
        let submit_started = Instant::now();
        if let Some(submission) =
            self.try_submit_relational_retained_int4_projection_jobs(jobs, submit_started)?
        {
            return Ok(submission);
        }
        let selects = jobs
            .iter()
            .map(|job| job.select.clone())
            .collect::<Vec<_>>();
        let results = self.execute_relational_equality_multi_column_projection_batch_inner(
            &selects,
            Some(jobs),
            true,
        )?;
        let first_job = jobs.first();
        Ok(RelationalRetainedReadSubmission {
            route_id: first_job
                .map(|job| job.route_id.clone())
                .unwrap_or_else(|| "empty".to_string()),
            table: first_job
                .map(|job| job.table.clone())
                .unwrap_or_else(|| "empty".to_string()),
            snapshot_generation: first_job.map(|job| job.snapshot_generation).unwrap_or(0),
            job_count: jobs.len(),
            submit_wall_micros: submit_started
                .elapsed()
                .as_micros()
                .try_into()
                .unwrap_or(u64::MAX),
            inner: RelationalRetainedReadSubmissionInner::Ready(results),
        })
    }

    pub fn complete_relational_retained_read_submission(
        &self,
        submission: RelationalRetainedReadSubmission,
    ) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
        match submission.inner {
            RelationalRetainedReadSubmissionInner::Ready(results) => Ok(results),
            RelationalRetainedReadSubmissionInner::PendingInt4Projection(pending) => {
                self.complete_relational_retained_int4_projection_submission(*pending)
            }
        }
    }

    fn try_submit_relational_retained_int4_projection_jobs(
        &self,
        jobs: &[RelationalRetainedReadJob],
        submit_started: Instant,
    ) -> Result<Option<RelationalRetainedReadSubmission>, ExecuteError> {
        if jobs.is_empty() {
            return Ok(Some(RelationalRetainedReadSubmission {
                route_id: "empty".to_string(),
                table: "empty".to_string(),
                snapshot_generation: 0,
                job_count: 0,
                submit_wall_micros: submit_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
                inner: RelationalRetainedReadSubmissionInner::Ready(Vec::new()),
            }));
        }

        let mut members = Vec::with_capacity(jobs.len());
        let mut batch_table: Option<RelationalTable> = None;
        let mut batch_filter_idx: Option<usize> = None;
        let mut batch_selected_indexes: Option<Vec<usize>> = None;
        for job in jobs {
            let query_shape = job.route_id.split(':').next().unwrap_or("unknown");
            if !matches!(
                query_shape,
                "int4_equality_projection" | "int4_equality_multi_column_projection"
            ) {
                return Ok(None);
            }
            let (table, bound, copin_s) = self.bind_relational_select_for_execution(&job.select)?;
            let filter_groups = if !bound.filter_groups.is_empty() {
                bound.filter_groups.clone()
            } else if !bound.filters.is_empty() {
                vec![bound.filters.clone()]
            } else if let Some(filter) = bound.filter.clone() {
                vec![vec![filter]]
            } else {
                Vec::new()
            };
            if job.schema != table.schema
                || job.table != table.name
                || job.params.len() != 1
                || bound.selected_indexes.is_empty()
                || !bound
                    .selected_indexes
                    .iter()
                    .all(|idx| table.columns[*idx].ty == SqlType::Int4)
                || filter_groups.len() != 1
                || filter_groups[0].len() != 1
            {
                return Ok(None);
            }
            let (filter_idx, op, value) = filter_groups[0][0].clone();
            let SqlValue::Int4(needle) = value else {
                return Ok(None);
            };
            if op != SelectFilterOp::Eq || table.columns[filter_idx].ty != SqlType::Int4 {
                return Ok(None);
            }
            match &job.params[0] {
                RelationalRetainedReadParam::Int4Eq { column, value }
                    if column == &table.columns[filter_idx].name && *value == needle => {}
                _ => return Ok(None),
            }
            if let Some(existing) = &batch_table {
                if existing.name != table.name
                    || existing.schema != table.schema
                    || existing.columns != table.columns
                {
                    return Ok(None);
                }
            } else {
                batch_table = Some(table.clone());
            }
            if batch_filter_idx.is_some_and(|existing| existing != filter_idx) {
                return Ok(None);
            }
            batch_filter_idx = Some(filter_idx);
            if batch_selected_indexes
                .as_ref()
                .is_some_and(|existing| existing != &bound.selected_indexes)
            {
                return Ok(None);
            }
            batch_selected_indexes = Some(bound.selected_indexes.clone());
            let (_query, access_path) =
                self.relational_select_mvcc_query_pinned(&job.select, &table, &bound, copin_s)?;
            members.push((bound, access_path, needle));
        }

        let table = batch_table.expect("non-empty batch has table");
        let filter_idx = batch_filter_idx.expect("non-empty batch has filter");
        let selected_indexes = batch_selected_indexes.expect("non-empty batch has projections");
        let snapshot = self
            .relational_residency_snapshot_ref(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident snapshot",
                    table.name
                )))
            })?;
        if snapshot.schema != table.schema || snapshot.table != table.name || !snapshot.is_valid() {
            return Ok(None);
        }
        let snapshot_gpu_id = snapshot.gpu_id;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let filter_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let projection_offsets = selected_indexes
            .iter()
            .map(|idx| resident_device_int4_column_offset(&snapshot, &table, *idx))
            .collect::<Result<Vec<_>, ExecuteError>>()?;
        let needles = members
            .iter()
            .map(|(_bound, _access_path, needle)| *needle)
            .collect::<Vec<_>>();
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        device_memory.clear_last_kernel_event_elapsed_us();
        let before_metrics = self.metrics.snapshot();
        let batch_started = Instant::now();
        let cuda_submission = device_memory
            .submit_match_project_i32_equal_any_from_payload(
                filter_offset,
                &needles,
                &projection_offsets,
                row_count,
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let first_job = jobs.first().expect("non-empty jobs");

        Ok(Some(RelationalRetainedReadSubmission {
            route_id: first_job.route_id.clone(),
            table: first_job.table.clone(),
            snapshot_generation: first_job.snapshot_generation,
            job_count: jobs.len(),
            submit_wall_micros: submit_started
                .elapsed()
                .as_micros()
                .try_into()
                .unwrap_or(u64::MAX),
            inner: RelationalRetainedReadSubmissionInner::PendingInt4Projection(Box::new(
                RelationalRetainedInt4ProjectionSubmission {
                    table,
                    snapshot_gpu_id,
                    selected_indexes,
                    members,
                    before_metrics,
                    batch_started,
                    submission: cuda_submission,
                },
            )),
        }))
    }

    fn complete_relational_retained_int4_projection_submission(
        &self,
        pending: RelationalRetainedInt4ProjectionSubmission,
    ) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
        if !self
            .read_state
            .residency
            .device_memory
            .contains_key(&pending.table.name)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no retained resident device memory",
                pending.table.name
            ))));
        }
        let completion =
            Self::complete_relational_retained_int4_projection_submission_detached(pending)?;
        let row_metadata_d2h_bytes = u64::try_from(completion.total_rows)
            .unwrap_or(u64::MAX)
            .saturating_mul((std::mem::size_of::<u32>() + std::mem::size_of::<u64>()) as u64)
            .saturating_add(std::mem::size_of::<u32>() as u64);
        let result_d2h_bytes = u64::try_from(completion.total_rows)
            .unwrap_or(u64::MAX)
            .saturating_mul(
                u64::try_from(completion.int4_result_columns)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(std::mem::size_of::<i32>() as u64),
            )
            .saturating_add(row_metadata_d2h_bytes);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.metrics
            .observe_kernel_exec_ms(completion.batch_micros.div_ceil(1000).max(1));
        if let Some(elapsed_us) = completion.kernel_event_elapsed_us {
            self.metrics.observe_kernel_event_elapsed_us(elapsed_us);
        }
        self.read_state
            .route_telemetry
            .record_route_selected_projection_micros(
                &completion.table_name,
                completion.batch_micros,
                completion.batch_micros,
                completion.materialization_micros,
                completion.total_rows,
            );
        let after_metrics = self.metrics.snapshot();
        self.read_state
            .route_telemetry
            .record_route_execution_observation(
                &completion.table_name,
                RelationalResidentRouteExecutionObservation {
                    h2d_bytes: after_metrics
                        .h2d_bytes_total
                        .saturating_sub(completion.before_metrics.h2d_bytes_total),
                    d2h_bytes: after_metrics
                        .d2h_bytes_total
                        .saturating_sub(completion.before_metrics.d2h_bytes_total),
                    kernel_samples: after_metrics
                        .kernel_exec_samples
                        .saturating_sub(completion.before_metrics.kernel_exec_samples),
                    kernel_ms: after_metrics
                        .kernel_exec_total_ms
                        .saturating_sub(completion.before_metrics.kernel_exec_total_ms),
                    kernel_event_elapsed_us: completion.kernel_event_elapsed_us,
                    rows: completion.total_rows,
                    wall_micros: completion.wall_micros,
                },
            );

        Ok(completion.results)
    }

    pub(crate) fn complete_relational_retained_int4_projection_submission_detached(
        pending: RelationalRetainedInt4ProjectionSubmission,
    ) -> Result<RelationalRetainedInt4ProjectionCompletion, ExecuteError> {
        let (projected_rows, kernel_event_elapsed_us) = pending
            .submission
            .complete_detached()
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let batch_micros = pending
            .batch_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        let materialize_started = Instant::now();
        // Stable-order fix (Thread-3 Stage 4): the `equal_any` kernel appends matches in
        // `atom.global.add` SCHEDULE order, which is non-deterministic for >32 matches (multi-warp)
        // and differs from the per-query path's order. Tag each scattered row with the kernel's
        // `row_index` and sort each needle's slice ASCENDING by it, so the batched output is
        // deterministic and byte-identical to the per-query ascending order (the `row_indices`
        // order class established by `4b750a94`). For the single-column self-projection every value
        // equals the needle, so this reorder is a no-op on the emitted value sequence (it only
        // makes the output deterministic); for multi-column the projected values differ per row, so
        // the sort is load-bearing for parity.
        let mut rows_by_select: Vec<Vec<(u64, Vec<SqlValue>)>> =
            vec![Vec::new(); pending.members.len()];
        for projected in &projected_rows {
            rows_by_select[projected.needle_index].push((
                projected.row_index,
                projected
                    .values
                    .iter()
                    .copied()
                    .map(SqlValue::Int4)
                    .collect::<Vec<_>>(),
            ));
        }
        let rows_by_select: Vec<Vec<Vec<SqlValue>>> = rows_by_select
            .into_iter()
            .map(|mut slice| {
                slice.sort_by_key(|(row_index, _)| *row_index);
                slice.into_iter().map(|(_, row)| row).collect()
            })
            .collect();
        let materialization_micros = materialize_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        let total_rows = rows_by_select.iter().map(Vec::len).sum::<usize>();
        let table_name = pending.table.name.clone();
        let results = pending
            .members
            .into_iter()
            .zip(rows_by_select)
            .map(
                |((bound, access_path, _needle), rows)| RelationalSelectResult {
                    columns: bound.selected_columns,
                    rows,
                    planned_target: DeviceTarget::Gpu(pending.snapshot_gpu_id),
                    executed_target: DeviceTarget::Gpu(pending.snapshot_gpu_id),
                    fallback_reason: None,
                    access_path,
                },
            )
            .collect();
        let wall_micros = pending
            .batch_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        Ok(RelationalRetainedInt4ProjectionCompletion {
            table_name,
            before_metrics: pending.before_metrics,
            batch_micros,
            wall_micros,
            materialization_micros,
            total_rows,
            int4_result_columns: pending.selected_indexes.len(),
            kernel_event_elapsed_us,
            results,
        })
    }
}
