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

        // Cold per-job path. All jobs in a batch are asserted identical-shape below, so the projected schema
        // is shared (captured once from the first job). Collect just the needles + the shared schema — no
        // per-needle Vec (DECISIONS "Result-path optimization").
        let mut needles: Vec<i32> = Vec::with_capacity(jobs.len());
        let mut shared_schema: Option<(Arc<Vec<RelationalColumn>>, Arc<RelationalAccessPath>)> =
            None;
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
            needles.push(needle);
            // Identical-shape across jobs (asserted above) -> capture the shared schema once. The
            // `access_path` is shared too: each job is a SINGLE-key point read (`job.params.len() == 1`,
            // checked above), so every job's `EqualityIndex { matched_keys: 1, .. }` is identical — the
            // first is representative. (audit P3: `access_path` is a diagnostic field, never on the wire;
            // the production batcher uses the needle-invariant template path, not this jobs path.)
            if shared_schema.is_none() {
                shared_schema = Some((Arc::new(bound.selected_columns), Arc::new(access_path)));
            }
        }

        let table = batch_table.expect("non-empty batch has table");
        let filter_idx = batch_filter_idx.expect("non-empty batch has filter");
        let selected_indexes = batch_selected_indexes.expect("non-empty batch has projections");
        let (shared_columns, shared_access_path) =
            shared_schema.expect("non-empty batch has a shared schema");
        let (snapshot_gpu_id, before_metrics, batch_started, payload) = match self
            .submit_resident_int4_equal_any_payload(
                &table,
                &selected_indexes,
                filter_idx,
                &needles,
            )? {
            Some(out) => out,
            // Resident snapshot present but schema-mismatched / invalidated (the prior `Ok(None)`
            // case): fall through to the slower non-fast-path batch executor, unchanged.
            None => return Ok(None),
        };
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
                    needle_count: needles.len(),
                    shared_columns,
                    shared_access_path,
                    before_metrics,
                    batch_started,
                    payload,
                },
            )),
        }))
    }

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
        Ok(RelationalRetainedReadTemplate {
            route_id: job.route_id,
            schema: job.schema,
            snapshot_generation: job.snapshot_generation,
            selected_indexes: bound.selected_indexes,
            result_columns: bound.selected_columns,
            filter_idx,
            access_path,
            table,
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
                inner: RelationalRetainedReadSubmissionInner::Ready(Vec::new()),
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

    /// Shared GPU-submit core for an already-validated, same-shape int4 equality-projection batch over
    /// the resident snapshot: derive the filter + projection column offsets ONCE and launch the single
    /// `equal_any` submission for all `needles`. `Ok(None)` signals the resident snapshot is present but
    /// schema-mismatched / invalidated (the caller chooses fallback vs error); a missing snapshot or
    /// device memory stays a hard error (preserving the jobs path's prior semantics).
    fn submit_resident_int4_equal_any_payload(
        &self,
        table: &RelationalTable,
        selected_indexes: &[usize],
        filter_idx: usize,
        needles: &[i32],
    ) -> Result<Option<(u16, RuntimeMetricsSnapshot, Instant, DeferredProbe)>, ExecuteError> {
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
        let filter_offset = resident_device_int4_column_offset(&snapshot, table, filter_idx)?;
        let projection_offsets = selected_indexes
            .iter()
            .map(|idx| resident_device_int4_column_offset(&snapshot, table, *idx))
            .collect::<Result<Vec<_>, ExecuteError>>()?;
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
        // ADR-009 R1: route this resident int4 unique-key batch among two byte-identical producers (each
        // returns the SAME matched rows; only HOW they're found differs):
        //   (1) lpb GPU hash-index probe — `index_probe_enabled` ON, column unique/buildable: O(1)/needle.
        //   (2) full-scan kernel        — flag off, or column non-unique / un-buildable: O(rows).
        // Both are DEFERRED submissions drained at completion.
        // CONTRACT: the index route requires `needles` to be DISTINCT — the facade batcher's `dedup_needles`
        // guarantees this. The thread-per-needle gather emits one match per found needle vs the scan's one
        // per matched row; for a unique key + distinct needles these coincide (a bijection). Duplicate
        // needles would over-count relative to the scan, so the distinct-needle invariant is debug-asserted
        // INSIDE the index arm only — the scan arm is reachable with duplicate needles from the jobs-batch
        // caller (`submit_relational_retained_int4_projection_batch`) and must NOT be guarded.
        // launch-per-batch: GPU index probe when buildable, else the full scan -> a DeferredProbe.
        let payload = {
            match self
                .index_probe_enabled()
                .then(|| {
                    self.wave_resident_int4_index(
                        &table.name,
                        &device_memory,
                        filter_offset,
                        filter_idx,
                        row_count,
                    )
                })
                .flatten()
            {
                Some((index, table_mask, hash_shift)) => {
                    debug_assert!(
                        {
                            let mut seen = std::collections::HashSet::with_capacity(needles.len());
                            needles.iter().all(|needle| seen.insert(*needle))
                        },
                        "index route requires distinct needles (batcher dedup_needles contract)"
                    );
                    // The index route is unique (<=1 match/needle), so it can take the DENSE-emit kernel
                    // (DECISIONS "lpb read levers" #1) when the flag is on — byte-identical, no atomic/scatter.
                    if self.dense_index_probe_enabled() {
                        let dense = device_memory
                            .submit_match_project_i32_index_probe_dense_from_payload(
                                &index,
                                table_mask,
                                hash_shift,
                                needles,
                                &projection_offsets,
                                row_count,
                            )
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?;
                        self.read_state
                            .residency
                            .dense_index_probe_hits
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        DeferredProbe::Dense(dense)
                    } else {
                        DeferredProbe::Atomic(
                            device_memory
                                .submit_match_project_i32_index_probe_from_payload(
                                    &index,
                                    table_mask,
                                    hash_shift,
                                    needles,
                                    &projection_offsets,
                                    row_count,
                                )
                                .map_err(|err| {
                                    ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                                })?,
                        )
                    }
                }
                // The non-unique SCAN keeps the atomic kernel (>1 match/needle needs the atomic compaction +
                // row_indices for the within-needle sort).
                None => DeferredProbe::Atomic(
                    device_memory
                        .submit_match_project_i32_equal_any_from_payload(
                            filter_offset,
                            needles,
                            &projection_offsets,
                            row_count,
                        )
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?,
                ),
            }
        };
        Ok(Some((
            snapshot_gpu_id,
            before_metrics,
            batch_started,
            payload,
        )))
    }

    /// ADR-009 R1: fetch (building + caching on demand) the GPU hash index over resident int4 key column
    /// `filter_idx` of the resident buffer `device_memory`. `Some((index, table_mask, hash_shift))` drives
    /// the index-probe route; `None` means "use the scan" — the column is non-unique / un-buildable. The
    /// cache is keyed by table and validated by `(column_idx, resident_device_ptr)`: the index is built
    /// from + used with the SAME device buffer (its address pinned via `_resident_guard`, so a re-admission
    /// allocates a new buffer with a new address → cache miss → rebuild against the live bytes). This makes
    /// the index inherently consistent with the buffer the probe gathers from — no host-rows cross-map race.
    fn wave_resident_int4_index(
        &self,
        table_name: &str,
        device_memory: &Arc<CudaResidentDeviceMemory>,
        filter_offset: u64,
        filter_idx: usize,
        row_count: u64,
    ) -> Option<(Arc<CudaResidentDeviceMemory>, u32, u32)> {
        let resident_device_ptr = device_memory.device_ptr();
        {
            let cache = self
                .read_state
                .residency
                .wave_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(existing) = cache.get(table_name) {
                if existing.column_idx == filter_idx
                    && existing.resident_device_ptr == resident_device_ptr
                {
                    return existing.index_memory.as_ref().map(|memory| {
                        (Arc::clone(memory), existing.table_mask, existing.hash_shift)
                    });
                }
            }
        }
        // Miss / different column / different buffer: build OUTSIDE the lock (a DtoH of the key column + a
        // host hash pass + one HtoD upload), then publish. A concurrent builder for the same buffer merely
        // rebuilds + overwrites — rare (once per residency generation) and harmless: each index is
        // self-contained and its in-flight kernels pin their own `Arc`, so a replaced entry's index buffer
        // is freed only once no submission still holds it.
        // Serialize the budget check, device allocation, and cache publication with admission.
        // A cap decline is intentionally just an index miss: the caller executes the same query
        // with the resident GPU scan, never with host relational execution.
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let built = self.build_wave_resident_int4_index(device_memory, filter_offset, row_count);
        let (index_memory, table_mask, hash_shift) = match &built {
            Some((memory, mask, shift)) => (Some(Arc::clone(memory)), *mask, *shift),
            None => (None, 0, 0),
        };
        let mut cache = self
            .read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.insert(
            table_name.to_string(),
            WaveResidentIndex {
                column_idx: filter_idx,
                resident_device_ptr,
                _resident_guard: Arc::clone(device_memory),
                index_memory,
                table_mask,
                hash_shift,
            },
        );
        built
    }

    /// ADR-009 R1: build the open-addressing GPU hash index (`(key<<32)|(row+1)`, 0 = empty; Fibonacci
    /// `(key*0x9E3779B1)>>hash_shift` + linear probe) over int4 column at `filter_offset` of `device_memory`,
    /// then upload it once (HtoD). The keys are DtoH-read from the resident buffer ITSELF — the same bytes
    /// the scan reads and the probe gathers — so the row→value mapping is inherently consistent with the
    /// buffer and a NULL int4 (materialized as `0`) is indexed exactly as the scan matches it (no skip, so
    /// `WHERE col = 0` agrees on both routes). `None` (→ caller scans) when: the table is empty / too large
    /// to pack a row index into 32 bits; the DtoH fails; a key would exceed the kernel's 256-probe cap; or
    /// — critically for correctness — the column has DUPLICATE keys (incl. multiple NULLs-as-0): a hash
    /// index holds one row per key but the scan returns EVERY match, so a duplicate makes the index decline.
    fn build_wave_resident_int4_index(
        &self,
        device_memory: &Arc<CudaResidentDeviceMemory>,
        filter_offset: u64,
        row_count: u64,
    ) -> Option<(Arc<CudaResidentDeviceMemory>, u32, u32)> {
        // `row + 1` is packed into the low 32 bits, so the row index must fit in u32.
        if row_count == 0 || row_count >= (u32::MAX as u64) {
            return None;
        }
        let row_count_usize = usize::try_from(row_count).ok()?;
        let keys = device_memory
            .read_resident_i32_column(filter_offset, row_count_usize)
            .ok()?;
        if keys.len() != row_count_usize {
            return None;
        }
        let (index, table_mask, hash_shift) = build_int4_pk_hash_table_host(&keys, row_count)?;
        let index_bytes: Vec<u8> = index.iter().flat_map(|entry| entry.to_le_bytes()).collect();
        let runtime = self.cuda_driver_probe_runtime();
        let gpu_id = device_memory.metadata().gpu_id;
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(index_bytes.len() as u64)
                    > budget
            })
        {
            return None;
        }
        let memory = runtime
            .retain_device_memory_copy(gpu_id, &index_bytes)
            .ok()?;
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(memory.metadata().allocated_bytes)
                    > budget
            })
        {
            return None;
        }
        Some((Arc::new(memory), table_mask, hash_shift))
    }

    /// CROSS-SHARD PK INDEX (sub-slice 1): resolve `filter_idx = key` to the resident `(shard_id, LOCAL row)`
    /// positions via a PER-SHARD host-built PK hash index. Thin projection of
    /// [`Self::locate_resident_pk_via_shard_index_detailed`] to just `(shard_id, slot)` -- the shape the
    /// scan-locate differential + the (future) DELETE/UPDATE resolution compare against. See the detailed
    /// method for the semantics (returns the IDENTICAL physical `(shard, slot)` a scan finds; `None` to fall
    /// back on any decline / invalid shard / missing offset). Used by the scan-locate differential tests +
    /// the (future) DELETE/UPDATE resolution; the 3b read route calls the `_detailed` variant directly.
    #[cfg_attr(not(test), allow(dead_code))] // the GPU differentials' oracle-facing wrapper
    pub(crate) fn locate_resident_pk_via_shard_index(
        &self,
        table: &RelationalTable,
        filter_idx: usize,
        key: i32,
    ) -> Option<Vec<(u32, u32)>> {
        self.locate_resident_pk_via_shard_index_detailed(table, filter_idx, key)
            .map(|hits| hits.into_iter().map(|h| (h.shard_id, h.slot)).collect())
    }

    /// W0 (concurrent-invalidation liveness): the CONCURRENT commit path invalidates residency by
    /// tombstoning the device-memory CELL only (`invalidate_relational_residency_tables_concurrent`)
    /// — the `shards` descriptor flags are owned by the SERIALIZED path (they are not interior-
    /// mutable via `&self`), so `is_valid()` alone cannot prove a shard's bytes are current. Any
    /// WRITE-locate that trusts the descriptor's riding buffer must ALSO require the authoritative
    /// cell to still publish EXACTLY that Arc (ptr-identical). A tombstoned (`None`) or re-admitted
    /// (different-ptr) cell declines the locate, sending the caller to the always-correct host
    /// ladder. Without this gate, a concurrent host-installed write purges the PK cache but leaves
    /// the descriptor valid-looking, and the next probe REBUILDS the cache from STALE device bytes
    /// — where a physical miss is load-bearing ("no visible duplicate" / "0 matches"): a duplicate
    /// key FALSE-PASSES or an Eq-resolved UPDATE/DELETE loses its row. Repro + regression:
    /// `w0_concurrent_invalidation_must_not_leave_write_locate_trusting_stale_shards`.
    pub(crate) fn shard_write_locate_cell_live(
        &self,
        table_name: &str,
        shard_id: u32,
        descriptor_memory: &Arc<CudaResidentDeviceMemory>,
    ) -> bool {
        self.read_state
            .residency
            .shard_device_memory
            .get(&(table_name.to_string(), shard_id))
            .is_some_and(|cell| Arc::ptr_eq(&cell, descriptor_memory))
    }

    /// CROSS-SHARD PK INDEX: the GENERATION-CONSISTENT locate. Resolves `filter_idx = key` per shard via the
    /// cached hash+bloom index and, for each HIT, CAPTURES the exact `(descriptor, device_memory,
    /// deleted_by)` the slot was resolved against -- all from the SAME `shards.load()` snapshot, with the
    /// device buffer PINNED by the returned `Arc`. A caller (the 3b point-lookup route) that materializes the
    /// row from these captured handles reads the slot from the buffer it was computed for, closing the
    /// concurrent TOCTOU: reads are lock-free and straddle commits, so a concurrent DELETE re-admit can
    /// republish a shard_id's buffer with COMPACTED/reordered slots; resolving the slot against one
    /// generation and then re-`get`-ing the buffer (a second, independent `ArcSwap` load) could read the slot
    /// out of a DIFFERENT generation -> a wrong row. Returning the pinned buffer (not just `(shard,slot)`)
    /// makes slot + capacity-stride + buffer + deleted_by generation-consistent by construction. `None` to
    /// fall back to the scan when ANY shard declines (duplicate / 256-probe overflow / oversize) or is
    /// invalid / missing its offset or device memory. Filter column must be int4.
    pub(crate) fn locate_resident_pk_via_shard_index_detailed(
        &self,
        table: &RelationalTable,
        // COMPOUND KEYS: the probe key id (single-column column-index, or `FLAG | ordinal`); `key`
        // is the raw key or the compound fingerprint.
        key_id: usize,
        key: i32,
    ) -> Option<Vec<ShardPkHit>> {
        // M1 (charter ruling 2026-07-03): the DEVICE write-locate replaces the host PK-hash probe.
        // Same Vec<ShardPkHit> output (region-Arc capture unchanged) -> a drop-in the consumers
        // never see. The host-probe path below is the flag-off oracle until M3 deletes it.
        if self.device_write_locate_enabled() {
            return self.locate_resident_pk_via_device(table, key_id, key);
        }
        // COMPOUND KEYS: the host-oracle probe below is single-column (`probe_shard_pk_index_cached`
        // keys on one filter column); a compound key can't ride it, so decline -> the recheck falls
        // to the host rehydrate+scan ladder (correct, just not device-accelerated when the device
        // write-locate is disabled).
        if key_id & crate::engine_residency::COMPOUND_KEY_ID_FLAG != 0 {
            return None;
        }
        let filter_idx = key_id;
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut out: Vec<ShardPkHit> = Vec::new();
        for shard in table_shards.iter() {
            // Identity/validity prechecks (mirror `locate_resident_delete_slots` / the scan's `source_for`):
            // an invalidated / memory-pressured / catalog-mismatched shard forces None so the caller scans,
            // rather than reading a stale generation's device bytes (audit P3).
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            // An empty shard contributes no keys (the scan matches 0 rows there): SKIP it -- both to match the
            // scan (which continues to other rows) and to avoid the 0-row hash-build decline that would
            // otherwise drop hits from OTHER shards (audit P2).
            if shard.row_count == 0 {
                continue;
            }
            // The filter column's BYTE offset within this shard's own (capacity-strided) buffer -- the SAME
            // offset the scan reads, so the row indices line up 1:1 with the scan + the deleted_by gather.
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            let filter_offset =
                resident_device_int4_column_offset(&descriptor, table, filter_idx).ok()?;
            // D4 (ADR-013 pre2): the buffer rides the loaded descriptor — the SAME generation as
            // the metadata/zone-map this loop already read (no second map load to race a re-admit).
            let device_memory = shard.device_memory.clone()?;
            // W0: the descriptor flags don't see concurrent invalidations — require the
            // authoritative cell to still publish THIS buffer, else decline to the host ladder.
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            // Sub-slice 3: probe the CACHED per-shard hash+bloom index (built once per shard generation,
            // ptr-validated) instead of a per-lookup DtoH + rebuild. Bloom-prunes then hash-probes.
            match self.probe_shard_pk_index_cached(
                &table.name,
                shard.shard_id,
                filter_idx,
                &device_memory,
                filter_offset,
                shard.row_count,
                key,
            ) {
                ShardPkProbe::Hit(row) => {
                    // D4: the regions ride the SAME loaded descriptor as the buffer — the
                    // visibility gates read `deleted_by[slot]`/`created_by[slot]` aligned to the
                    // SAME generation as the slot + buffer, by construction (previously three
                    // separate map loads could straddle a republish).
                    let deleted_by = shard.deleted_by_region.clone();
                    let created_by = shard.created_by_region.clone();
                    let row_id = shard.row_id_region.clone();
                    out.push(ShardPkHit {
                        shard_id: shard.shard_id,
                        slot: row,
                        descriptor,
                        device_memory,
                        deleted_by,
                        created_by,
                        row_id,
                    });
                }
                ShardPkProbe::Miss => {}
                // A duplicate key in ANY shard declines the whole locate (the scan returns every match; a
                // hash holds one row/key) -> the caller scans.
                ShardPkProbe::Declined => {
                    return None;
                }
            }
        }
        Some(out)
    }

    /// M1 (ledger #24): INCREMENTALLY maintain a cached DEVICE PK index over an in-place append —
    /// insert only the k appended keys via the `index_insert` kernel (O(k)) instead of the O(rows)
    /// rebuild (`ensure_shard_pk_device_index`) the (ptr,row_count) validation would otherwise force
    /// every wave (the measured 305us/wave bottleneck). Device analog of the host
    /// `extend_shard_pk_index_cache_on_append`. Called at the append chokepoint with the appended
    /// values in hand (no DtoH). Per entry: a different ptr (re-admit) or a basis != `base_row_count`
    /// (a prober rebuilt) is skipped; a DECLINED entry stays declined (monotone); a dup/overflow ->
    /// DECLINED; past the load rule (`2*new_count > table_size`) the entry is DROPPED (the next probe
    /// rebuilds at the grown size).
    pub(crate) fn extend_shard_pk_device_index_on_append(
        &self,
        table_name: &str,
        shard_id: u32,
        device_ptr: u64,
        base_row_count: usize,
        column_values: &[Vec<i32>],
        // COMPOUND KEYS (wider types): the appended rows' full SqlValues, so a compound index's tail
        // fingerprint can fold WIDER key columns (i64) whose values are not in the i32 `column_values`.
        new_rows: &[Vec<SqlValue>],
    ) {
        let appended = column_values.first().map_or(0, Vec::len);
        if appended == 0 {
            return;
        }
        let new_count = base_row_count + appended;
        // Single-column keys: the cache is keyed by the catalog COLUMN INDEX, and the appended tail
        // is that column's values verbatim (`column_values[col_idx]`).
        for (col_idx, tail) in column_values.iter().enumerate() {
            self.extend_shard_pk_device_index_entry(
                (table_name.to_string(), shard_id, col_idx),
                device_ptr,
                base_row_count,
                new_count,
                tail,
            );
        }
        // COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): each compound unique index's cache entry is keyed
        // by `FLAG | ordinal`, and its appended tail is the per-row FINGERPRINT folded from the key
        // columns' appended values. Only compound indexes need this second pass (single-column keys
        // rode the loop above); resolved from the catalog (rare relative to the append itself).
        let catalog = self.catalog_snapshot();
        if let Some(table) = catalog.relational_catalog.get(table_name) {
            for (ord, index) in table.indexes.iter().enumerate() {
                if !index.unique || !crate::engine_residency::index_is_compound(index) {
                    continue;
                }
                // Fold each appended row's key TUPLE into its fingerprint from the full SqlValues
                // (handles any supported key type, incl. i64). A row whose key can't fold (e.g. a NULL
                // key column) makes the whole tail unfoldable -> skip this index's incremental extend;
                // its cache entry stays at the old row_count and the next probe rebuilds it.
                let mut fp_tail: Vec<i32> = Vec::with_capacity(new_rows.len());
                let mut foldable = true;
                for row in new_rows {
                    match crate::engine_residency::compound_index_row_fingerprint(table, index, row)
                    {
                        Some(fp) => fp_tail.push(fp),
                        None => {
                            foldable = false;
                            break;
                        }
                    }
                }
                if !foldable {
                    continue;
                }
                let key_id = crate::engine_residency::COMPOUND_KEY_ID_FLAG | ord;
                self.extend_shard_pk_device_index_entry(
                    (table_name.to_string(), shard_id, key_id),
                    device_ptr,
                    base_row_count,
                    new_count,
                    &fp_tail,
                );
            }
        }
    }

    /// M1 (ledger #24): maintain ONE cached device PK-index entry over an append — insert the k
    /// appended keys/fingerprints (`tail`) via the `index_insert` kernel. Shared by the single-column
    /// and compound passes of `extend_shard_pk_device_index_on_append` (`tail` is a column's raw
    /// values or the folded compound fingerprints; the device index treats both as opaque keys).
    fn extend_shard_pk_device_index_entry(
        &self,
        key: (String, u32, usize),
        device_ptr: u64,
        base_row_count: usize,
        new_count: usize,
        tail: &[i32],
    ) {
        {
            // Snapshot the entry basis under the lock (index Arc is cheap-cloned for the launch).
            let (index, table_mask, hash_shift) = {
                let cache = self
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let Some(entry) = cache.get(&key) else {
                    return; // not built on the device -> nothing to maintain
                };
                if entry.resident_device_ptr != device_ptr || entry.row_count != base_row_count {
                    return; // re-admit / a prober advanced it -> the prober path converges it
                }
                let Some(index) = entry.device_index.clone() else {
                    return; // DECLINED is monotone under appends
                };
                (index, entry.table_mask, entry.hash_shift)
            };
            let table_size = (table_mask as u64) + 1;
            if (new_count as u64).saturating_mul(2) > table_size {
                // Past the builder's load rule -> drop so the next probe rebuilds at the grown size.
                self.read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&key);
                return;
            }
            let Ok(base_row_u32) = u32::try_from(base_row_count) else {
                return;
            };
            // The kernel mutates the device index buffer IN PLACE (atom.cas). A launch failure ->
            // drop the entry (rebuild next probe); never a wrong index.
            match index.submit_i32_index_insert(&index, table_mask, hash_shift, tail, base_row_u32)
            {
                Ok(dup) => {
                    let mut cache = self
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let Some(entry) = cache.get_mut(&key) else {
                        return;
                    };
                    // Re-validate the basis (a racing rebuild could have replaced it).
                    if entry.resident_device_ptr != device_ptr || entry.row_count != base_row_count
                    {
                        return;
                    }
                    if dup {
                        // F3/U4: the insert kernel now PLACES version twins, so `dup` no longer
                        // means "duplicate key" — it fires ONLY on a 256-probe OVERFLOW (a shard
                        // whose live+twin fan-out overran the probe cap). Drop the index so the
                        // next probe rebuilds at the grown, boundary-gated size (dead-below-GC
                        // twins are dropped there). A pathological hot-key with >256 un-GC'd
                        // versions stays declined until its readers release — a bounded transient.
                        entry.device_index = None;
                    }
                    entry.row_count = new_count;
                }
                Err(_) => {
                    self.read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .remove(&key);
                }
            }
        }
    }

    /// M1 design B (wave-time batched validation): probe a BATCH of `needles` against the table's
    /// DEVICE hash indexes in ONE kernel launch, returning each needle's HIT COUNT (any shard). A
    /// FAST-PATH FILTER for the wave's PK-unique validation: `count == 0` proves NO physical slot
    /// holds the key -> no visible dup -> the INSERT passes with zero further work (the common
    /// case: unique keys). `count > 0` (incl u32::MAX overflow) -> the caller runs the
    /// authoritative per-item `visible_row_with_value` (a tombstoned/invisible slot is a
    /// false-positive here, filtered there). `None` (caller validates per-item) on: no shards,
    /// any invalid/pressured/mismatched shard, a dup-key index (== host Declined), a device-probe
    /// failure. One launch amortizes across the whole wave (the amortization curve: launch cost
    /// is flat vs batch size).
    pub(crate) fn wave_batch_locate_hit_counts(
        &self,
        table: &RelationalTable,
        key_id: usize,
        needles: &[i32],
    ) -> Option<Vec<u32>> {
        // E2.5b-2 device-stage aggregation (v1): under lanes, funnel locate
        // calls through the cross-lane coalescer — one kernel launch covers
        // every lane's concurrently-pending wave (fixed-per-launch device cost
        // was the measured scaling bound past 4 lanes).
        if self.intent_lanes.is_some() {
            return self.wave_batch_locate_coalesced(table, key_id, needles);
        }
        self.wave_batch_locate_hit_counts_direct(table, key_id, needles)
    }

    /// The cross-lane coalescing front of the device locate (see
    /// `IntentLaneState::validate_queue`). Push the request, then either lead
    /// (drain every same-target request, ONE launch, scatter counts) or spin
    /// until a leader completes ours.
    fn wave_batch_locate_coalesced(
        &self,
        table: &RelationalTable,
        key_id: usize,
        needles: &[i32],
    ) -> Option<Vec<u32>> {
        use std::sync::atomic::Ordering as AOrd;
        let lanes = self
            .intent_lanes
            .as_ref()
            .expect("coalesced locate requires lanes");
        let slot = std::sync::Arc::new(crate::engine_intent_lanes::ValidateSlot {
            done: std::sync::atomic::AtomicBool::new(false),
            result: std::sync::Mutex::new(None),
        });
        lanes
            .validate_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(crate::engine_intent_lanes::ValidateRequest {
                table: table.name.clone(),
                key_id,
                needles: needles.to_vec(),
                slot: std::sync::Arc::clone(&slot),
            });
        loop {
            if slot.done.load(AOrd::Acquire) {
                return slot
                    .result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
                    .expect("done implies result written");
            }
            let Ok(_leader) = lanes.validate_leader.try_lock() else {
                std::hint::spin_loop();
                continue;
            };
            // LEADER: drain every request for THIS (table, filter) target —
            // including our own — into one concatenated launch.
            let batch: Vec<crate::engine_intent_lanes::ValidateRequest> = {
                let mut queue = lanes
                    .validate_queue
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let mut matched = Vec::new();
                let mut rest = Vec::with_capacity(queue.len());
                for request in queue.drain(..) {
                    if request.table == table.name && request.key_id == key_id {
                        matched.push(request);
                    } else {
                        rest.push(request);
                    }
                }
                *queue = rest;
                matched
            };
            if batch.is_empty() {
                // someone else's leader round already served us; loop re-checks
                continue;
            }
            let leader_started = std::time::Instant::now();
            let mut all_needles: Vec<i32> =
                Vec::with_capacity(batch.iter().map(|r| r.needles.len()).sum());
            for request in &batch {
                all_needles.extend_from_slice(&request.needles);
            }
            lanes.stat_coalesced_launches.fetch_add(1, AOrd::Relaxed);
            lanes
                .stat_coalesced_requests
                .fetch_add(batch.len() as u64, AOrd::Relaxed);
            let counts = self.wave_batch_locate_hit_counts_direct(table, key_id, &all_needles);
            let mut offset = 0usize;
            for request in batch {
                let take = request.needles.len();
                let piece = counts
                    .as_ref()
                    .map(|all| all[offset..offset + take].to_vec());
                offset += take;
                *request
                    .slot
                    .result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(piece);
                request.slot.done.store(true, AOrd::Release);
            }
            lanes
                .stat_validate_leader_ns
                .fetch_add(leader_started.elapsed().as_nanos() as u64, AOrd::Relaxed);
            // our own slot was in the batch; the loop's next pass returns it
        }
    }

    pub(crate) fn wave_batch_locate_hit_counts_direct(
        &self,
        table: &RelationalTable,
        // COMPOUND KEYS: the probe key id (single-column column-index, or `FLAG | ordinal`). The
        // `needles` are the raw i32 keys for a single-column key, or the host-computed compound
        // fingerprints — the device index treats both as opaque 32-bit keys.
        key_id: usize,
        needles: &[i32],
    ) -> Option<Vec<u32>> {
        let positions = crate::engine_residency::probe_key_id_positions(table, key_id)?;
        // COUNT-ONLY (max_hits=0): the kernel emits per-needle counts only (0 = no dup, else
        // u32::MAX), skipping the shard/slot buffers + 2 DtoH reads this fn never consumes.
        const MAX_HITS: u32 = 0;
        if needles.is_empty() {
            return Some(Vec::new());
        }
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut descs: Vec<WriteLocateShard> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            if !shard.is_valid(
                runtime_snapshot
                    .memory_pressured_gpu_ids
                    .contains(&shard.gpu_id),
            ) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            // PERF (this runs SERIALLY on the sequencer, per wave): compute the i32 filter offset
            // DIRECTLY from the shard's own fields — `resident_snapshot_for_shard` would clone the
            // whole descriptor (int4/int8/text/null name vectors) per shard per wave for nothing.
            let offsets = positions
                .iter()
                .map(|&p| shard_fixed_width_key_offset(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let blob_offsets = positions
                .iter()
                .map(|&p| shard_key_column_blob_offset(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let device_memory = shard.device_memory.clone()?;
            // W0: same cell-liveness gate as the host-probe locate (descriptor flags don't see
            // concurrent invalidations); a stale shard declines the whole wave-batch probe.
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            let (device_index, table_mask, hash_shift) = self.ensure_shard_pk_device_index(
                table,
                &table.name,
                shard.shard_id,
                key_id,
                &positions,
                &device_memory,
                &offsets,
                &blob_offsets,
                shard.row_count,
            )?;
            descs.push(WriteLocateShard {
                index: device_index,
                table_mask,
                hash_shift,
            });
        }
        if descs.is_empty() {
            return Some(vec![0u32; needles.len()]); // no probed shards -> every needle misses
        }
        let ctx = Arc::clone(&descs[0].index);
        let result = ctx
            .submit_multi_shard_i32_write_locate(&descs, needles, MAX_HITS)
            .ok()?;
        if result.count.len() != needles.len() {
            return None;
        }
        self.read_state
            .residency
            .device_write_locate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(result.count)
    }

    /// U1 (lane DELETE intents): the batched DEVICE VISIBLE-LOCATE — one coalesced launch
    /// resolving every needle to its VISIBLE match count + first visible (shard_id, slot) at the
    /// needle's OWN snapshot (visibility evaluated ON-DEVICE from the shards' created_by /
    /// deleted_by regions; absent region = born-visible / all-live, matching the fills).
    /// Declines (`None`) exactly like `wave_batch_locate_hit_counts_direct`: any invalid /
    /// pressured / mismatched / stale-cell shard, or an index that can't be ensured — the caller
    /// falls back per-needle or aborts retryably. `targets[i]` carries the probed shard's
    /// identity handles for the APPLY-TIME liveness recheck (a VACUUM/re-admit between locate
    /// and the coalesced tombstone apply rebuilds the shard and re-clusters slots — the apply
    /// must decline on identity mismatch, never stamp a re-clustered slot).
    pub(crate) fn wave_batch_visible_locate(
        &self,
        table: &RelationalTable,
        // COMPOUND KEYS: the probe key id. This visibility-blind LANE tombstone/update path consumes
        // the located `(shard, slot)` WITHOUT re-verifying the row's key, so a fingerprint collision
        // must never reach it — its callers pass single-column key ids ONLY. That is structurally
        // guaranteed: a compound-keyed table cannot take the covered-DELETE/UPDATE lane (it needs a
        // covered-INSERT route, which rejects compound). Compound DELETE/UPDATE ARE implemented — via
        // the SQL resolve path (`resolve_dml_matches_via_device` -> `dml_device_probe_key`), which
        // probes the fingerprint index and then re-verifies the FULL tuple with the `filter_groups`
        // recheck. So `probe_key_id_positions` resolves `[key_id]` here in practice.
        key_id: usize,
        needles: &[i32],
        snapshots: &[u64],
    ) -> Option<WaveVisibleLocate> {
        if needles.is_empty() {
            return Some(WaveVisibleLocate::default());
        }
        let positions = crate::engine_residency::probe_key_id_positions(table, key_id)?;
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut descs: Vec<VisibleLocateShard> = Vec::new();
        // Parallel to `descs`: the probed shard's id + its MAIN device region (the W0 cell-
        // liveness identity) — plus pins for the version regions the kernel dereferences.
        let mut probed: Vec<(u32, Arc<CudaResidentDeviceMemory>)> = Vec::new();
        let mut region_pins: Vec<Arc<CudaResidentDeviceMemory>> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            if !shard.is_valid(
                runtime_snapshot
                    .memory_pressured_gpu_ids
                    .contains(&shard.gpu_id),
            ) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            let offsets = positions
                .iter()
                .map(|&p| shard_fixed_width_key_offset(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let blob_offsets = positions
                .iter()
                .map(|&p| shard_key_column_blob_offset(shard, table, p))
                .collect::<Option<Vec<u64>>>()?;
            let device_memory = shard.device_memory.clone()?;
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            let (device_index, table_mask, hash_shift) = self.ensure_shard_pk_device_index(
                table,
                &table.name,
                shard.shard_id,
                key_id,
                &positions,
                &device_memory,
                &offsets,
                &blob_offsets,
                shard.row_count,
            )?;
            let created_by_ptr = shard
                .created_by_region
                .as_ref()
                .map(|region| {
                    region_pins.push(Arc::clone(region));
                    region.device_ptr()
                })
                .unwrap_or(0);
            let deleted_by_ptr = shard
                .deleted_by_region
                .as_ref()
                .map(|region| {
                    region_pins.push(Arc::clone(region));
                    region.device_ptr()
                })
                .unwrap_or(0);
            descs.push(VisibleLocateShard {
                index: device_index,
                table_mask,
                hash_shift,
                created_by_ptr,
                deleted_by_ptr,
            });
            probed.push((shard.shard_id, device_memory));
        }
        if descs.is_empty() {
            // No probed shards: every needle has zero visible matches.
            return Some(WaveVisibleLocate {
                counts: vec![0u32; needles.len()],
                shard_ids: vec![0u32; needles.len()],
                slots: vec![0u32; needles.len()],
                probed: Vec::new(),
            });
        }
        let ctx = Arc::clone(&descs[0].index);
        let result = ctx
            .submit_multi_shard_i32_visible_locate(&descs, needles, snapshots)
            .ok()?;
        drop(region_pins); // kernel fenced by the blocking DtoH inside the submit
        if result.count.len() != needles.len() {
            return None;
        }
        self.read_state
            .residency
            .device_visible_locate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Map probed-descriptor indexes back to real shard ids + identity handles.
        let mut shard_ids = vec![0u32; needles.len()];
        for (needle, (&count, &desc_idx)) in
            result.count.iter().zip(result.shard_idx.iter()).enumerate()
        {
            if count >= 1 {
                let (shard_id, _) = probed.get(desc_idx as usize)?;
                shard_ids[needle] = *shard_id;
            }
        }
        Some(WaveVisibleLocate {
            counts: result.count,
            shard_ids,
            slots: result.slot,
            probed,
        })
    }

    /// M1 (charter-pure): the DEVICE write-locate — probe the per-shard DEVICE hash indexes in ONE
    /// kernel launch (`submit_multi_shard_i32_write_locate`) instead of the host `shard_pk_index`
    /// hash cache. Builds the SAME `Vec<ShardPkHit>` the host path does (region Arcs captured from
    /// the same loaded descriptor), so callers are identical. Declines (None -> caller scans) on:
    /// any invalid/pressured/mismatched shard (parity with the host path's precheck), a shard whose
    /// device index can't be built (dup keys — matches `ShardPkProbe::Declined`), a device-probe
    /// failure, or a per-needle overflow past `MAX_HITS` (a cross-shard multiplicity the host path
    /// likewise declines). NO host hash probe anywhere on this path.
    fn locate_resident_pk_via_device(
        &self,
        table: &RelationalTable,
        // COMPOUND KEYS: the probe key id; `key` is the raw i32 key (single-column) or the compound
        // fingerprint. The returned hits are still (shard, slot) — the CALLER (the authoritative
        // recheck) materializes each and compares the FULL tuple, so a fingerprint collision is
        // filtered there.
        key_id: usize,
        key: i32,
    ) -> Option<Vec<ShardPkHit>> {
        const MAX_HITS: u32 = 4;
        let positions = crate::engine_residency::probe_key_id_positions(table, key_id)?;
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        // Parallel Vecs: the kernel descriptors + the per-descriptor shard context (descriptor,
        // buffer, region Arcs) so a hit's shard_idx maps back to build the ShardPkHit. Empty +
        // 0-row shards are SKIPPED (parity with the host path); a hit's shard_idx indexes into
        // `ctxs`, which lists only the probed shards in order.
        let mut descs: Vec<WriteLocateShard> = Vec::new();
        struct ShardCtx {
            shard_id: u32,
            descriptor: RelationalResidencySnapshot,
            device_memory: Arc<CudaResidentDeviceMemory>,
            deleted_by: Option<Arc<CudaResidentDeviceMemory>>,
            created_by: Option<Arc<CudaResidentDeviceMemory>>,
            row_id: Option<Arc<CudaResidentDeviceMemory>>,
        }
        let mut ctxs: Vec<ShardCtx> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            // COMPOUND KEYS (wider types): dispatch each key column to its section's descriptor offset
            // helper (i32-section vs i64 section vs b128 vs text-offsets), matching
            // `shard_fixed_width_key_offset`. A TEXT key column contributes its OFFSETS-array byte offset.
            let offsets = positions
                .iter()
                .map(|&p| match table.columns.get(p).map(|c| c.ty) {
                    Some(SqlType::Int8) | Some(SqlType::Timestamp) => {
                        resident_device_int8_column_offset(&descriptor, table, p).ok()
                    }
                    Some(SqlType::Numeric { .. }) | Some(SqlType::Uuid) => {
                        resident_device_numeric_column_offset(&descriptor, table, p).ok()
                    }
                    Some(SqlType::Text) => {
                        resident_device_text_column_layout(&descriptor, table, p)
                            .ok()
                            .map(|layout| layout.offsets_byte_offset)
                    }
                    _ => resident_device_int4_column_offset(&descriptor, table, p).ok(),
                })
                .collect::<Option<Vec<u64>>>()?;
            // COMPOUND KEYS (text): the parallel blob byte offsets (nonzero only for a text column).
            let blob_offsets = positions
                .iter()
                .map(|&p| match table.columns.get(p).map(|c| c.ty) {
                    Some(SqlType::Text) => {
                        resident_device_text_column_layout(&descriptor, table, p)
                            .ok()
                            .map(|layout| layout.bytes_byte_offset)
                    }
                    _ => Some(0),
                })
                .collect::<Option<Vec<u64>>>()?;
            let device_memory = shard.device_memory.clone()?;
            // W0: same cell-liveness gate as the host-probe locate (descriptor flags don't see
            // concurrent invalidations); a stale shard declines the device locate to the ladder.
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            // Build/reuse the shard's DEVICE hash index (uploaded once per generation,
            // (ptr,row_count)-validated). None = the shard has DUP keys -> decline the whole
            // locate to the scan, exactly like the host `ShardPkProbe::Declined`.
            let (device_index, table_mask, hash_shift) = self.ensure_shard_pk_device_index(
                table,
                &table.name,
                shard.shard_id,
                key_id,
                &positions,
                &device_memory,
                &offsets,
                &blob_offsets,
                shard.row_count,
            )?;
            descs.push(WriteLocateShard {
                index: device_index,
                table_mask,
                hash_shift,
            });
            ctxs.push(ShardCtx {
                shard_id: shard.shard_id,
                descriptor,
                device_memory,
                deleted_by: shard.deleted_by_region.clone(),
                created_by: shard.created_by_region.clone(),
                row_id: shard.row_id_region.clone(),
            });
        }
        if descs.is_empty() {
            return Some(Vec::new()); // no probed shards -> no hits (parity with the host loop)
        }
        // ONE device launch: the launch context is any device buffer on the GPU (the first shard's).
        let ctx = Arc::clone(&descs[0].index);
        let result = ctx
            .submit_multi_shard_i32_write_locate(&descs, &[key], MAX_HITS)
            .ok()?;
        let count = *result.count.first()?;
        if count == u32::MAX {
            return None; // overflow past MAX_HITS -> decline to the scan (cross-shard multiplicity)
        }
        self.read_state
            .residency
            .device_write_locate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut out: Vec<ShardPkHit> = Vec::with_capacity(count as usize);
        for h in 0..count as usize {
            let shard_idx = *result.shard_idx.get(h)? as usize;
            let slot = *result.slot.get(h)?;
            let c = ctxs.get(shard_idx)?;
            out.push(ShardPkHit {
                shard_id: c.shard_id,
                slot,
                descriptor: c.descriptor.clone(),
                device_memory: Arc::clone(&c.device_memory),
                deleted_by: c.deleted_by.clone(),
                created_by: c.created_by.clone(),
                row_id: c.row_id.clone(),
            });
        }
        Some(out)
    }

    /// TYPE-COVERAGE track 1 (ledger #3, WRITER-side maintenance): extend every cached
    /// `(table, shard, col)` PK-index entry over the rows an in-place append just wrote. The
    /// appended values are known HOST-SIDE at the append chokepoint, so maintenance is O(k)
    /// hash+bloom inserts with NO device read — probers stop paying the per-flush tail DtoH
    /// (the prober-side `try_extend_cached_shard_pk_index` remains the fallback for entries
    /// whose basis this call skips). Per entry: a different ptr (re-admit raced) or a basis
    /// other than `base_row_count` (a prober's DtoH extension raced ahead) is skipped — the
    /// prober ladder converges it; a DECLINED entry is left untouched (advancing its count
    /// would shrink the monotone-decline window for probers pinned between the dup point and
    /// this append); a duplicate appended key transitions the entry to DECLINED (the same
    /// conclusion a full rebuild reaches — e.g. an SV5 update-append duplicating its key
    /// against the old slot); past the builder's load rule the entry is DROPPED so the next
    /// probe rebuilds + resizes off the hot flush path.
    pub(crate) fn extend_shard_pk_index_cache_on_append(
        &self,
        table_name: &str,
        shard_id: u32,
        device_ptr: u64,
        base_row_count: usize,
        column_values: &[Vec<i32>],
    ) {
        let appended = column_values.first().map_or(0, Vec::len);
        if appended == 0 {
            return;
        }
        let mut cache = self
            .read_state
            .residency
            .shard_pk_index
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (col_idx, tail) in column_values.iter().enumerate() {
            let key = (table_name.to_string(), shard_id, col_idx);
            let mut drop_entry = false;
            {
                let Some(entry) = cache.get_mut(&key) else {
                    continue; // never probed: built on demand later
                };
                if entry.resident_device_ptr != device_ptr
                    || entry.row_count != base_row_count
                    || entry.index.is_none()
                {
                    continue;
                }
                let new_count = base_row_count + appended;
                let data = entry.index.as_mut().expect("checked above");
                if (new_count as u64).saturating_mul(2) > data.hash_table.len() as u64 {
                    drop_entry = true; // resize belongs to the prober's rebuild, not the flush
                } else if extend_int4_pk_hash_table_host(
                    &mut data.hash_table,
                    data.table_mask,
                    data.hash_shift,
                    tail,
                    base_row_count,
                ) {
                    extend_int4_pk_bloom_host(
                        &mut data.bloom_words,
                        data.bloom_num_bits,
                        data.bloom_num_hashes,
                        tail,
                    );
                    entry.row_count = new_count;
                    self.read_state
                        .residency
                        .pk_index_writer_extends
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                } else {
                    entry.index = None; // dup appended key: monotone decline at this basis
                    entry.row_count = new_count;
                }
            }
            if drop_entry {
                cache.remove(&key);
            }
        }
    }

    /// The batch twin of `probe_shard_pk_index_fast_path`: probe EVERY needle under ONE lock
    /// against a covering entry. `Some(true)` = all needles answered (`on_hit` called per Hit
    /// within the caller's slot bound); `Some(false)` = the shard DECLINES (monotone dup state);
    /// `None` = extend/rebuild. A `Some(index)` entry never yields `Declined` mid-batch
    /// (`Declined` only comes from `index: None`), so `on_hit` sees no partial batch.
    fn probe_shard_pk_index_batch_fast_path<F: FnMut(u32, u32)>(
        &self,
        cache_key: &(String, u32, usize),
        device_ptr: u64,
        row_count: usize,
        needles: &[i32],
        on_hit: &mut F,
    ) -> Option<bool> {
        let cache = self
            .read_state
            .residency
            .shard_pk_index
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = cache.get(cache_key)?;
        if entry.resident_device_ptr != device_ptr {
            return None;
        }
        if entry.index.is_none() {
            if row_count >= entry.row_count {
                return Some(false);
            }
            return None; // declined at MORE rows: the shorter prefix may be dup-free -> rebuild
        }
        if entry.row_count < row_count {
            return None; // stale: appended since the build -> extend (or rebuild)
        }
        for (ni, &key) in needles.iter().enumerate() {
            match probe_cached_shard_pk(entry, key) {
                ShardPkProbe::Hit(slot) if (slot as usize) < row_count => on_hit(ni as u32, slot),
                ShardPkProbe::Hit(_) => {} // appended after this caller's pinned snapshot -> miss
                ShardPkProbe::Miss => {}
                ShardPkProbe::Declined => return Some(false),
            }
        }
        Some(true)
    }

    /// The cached-entry fast path shared by the single and batch probes: answer from the cache
    /// when the entry's ptr matches and its row_count COVERS the caller's pinned `row_count`.
    /// `Some(probe)` = answered; `None` = the caller must extend or rebuild.
    ///
    /// AHEAD entries (`entry.row_count > row_count`: a prober pinned to a NEWER shard descriptor
    /// extended first) are probeable with a SLOT-BOUND filter — the hash holds at most one row
    /// per key (dups decline the whole entry), so a Hit at `slot >= row_count` proves the key's
    /// only occurrence is newer than this caller's snapshot -> Miss. This also removes the
    /// two-direction rebuild thrash the old EXACT row_count rule caused between probers pinned
    /// at different generations. An ahead DECLINED entry is NOT declinable here: dup-ness at
    /// MORE rows says nothing about the shorter prefix -> fall to rebuild at the caller's count.
    fn probe_shard_pk_index_fast_path(
        &self,
        cache_key: &(String, u32, usize),
        device_ptr: u64,
        row_count: usize,
        key: i32,
    ) -> Option<ShardPkProbe> {
        let cache = self
            .read_state
            .residency
            .shard_pk_index
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = cache.get(cache_key)?;
        if entry.resident_device_ptr != device_ptr {
            return None; // re-admit/rollover: a different buffer -> rebuild against live bytes
        }
        // RETIREMENT A2 (measured cliff): a DECLINED entry (`index: None` — duplicate keys, e.g.
        // an SV5 update-append duplicating its key across old+new slots) stays declined under
        // FURTHER APPENDS on the same buffer: dup-ness is MONOTONE under appends (MEASURED:
        // single-row UPDATE p50 went linear, 358->887us at 64k->262k, rebuild-to-decline each
        // statement). Only a ptr change (re-admit / VACUUM re-clustering) can clear a dup.
        if entry.index.is_none() {
            if row_count >= entry.row_count {
                return Some(ShardPkProbe::Declined);
            }
            return None; // declined at MORE rows: the shorter prefix may be dup-free -> rebuild
        }
        if entry.row_count < row_count {
            return None; // stale: appended since the build -> extend (or rebuild)
        }
        match probe_cached_shard_pk(entry, key) {
            ShardPkProbe::Hit(slot) if (slot as usize) >= row_count => Some(ShardPkProbe::Miss),
            other => Some(other),
        }
    }

    /// TYPE-COVERAGE track 1 (ledger #3): bring a cached shard PK index CURRENT after in-place
    /// appends by inserting ONLY the appended tail keys — O(delta) instead of the O(shard)
    /// rebuild that made every constrained-INSERT probe pay ~1ms under per-commit append churn.
    /// Returns `true` when the cache entry is now current for `(device_ptr, row_count)` (either
    /// extended live, transitioned to the monotone DECLINED state on a dup/overflow tail key, or
    /// another prober already brought it current); `false` when no extension applies (absent
    /// entry, ptr changed, load rule exceeded — the builder's `2*count <= table_size`) and the
    /// caller must full-rebuild (which re-sizes both hash and bloom).
    ///
    /// Locking: the tail DtoH read happens OUTSIDE the lock (it can stall ~10s of µs); the
    /// mutation re-validates `(ptr, base_count)` under the lock and retries once if a concurrent
    /// extender advanced the entry meanwhile (their tail may already cover ours).
    fn try_extend_cached_shard_pk_index(
        &self,
        cache_key: &(String, u32, usize),
        device_memory: &Arc<CudaResidentDeviceMemory>,
        filter_offset: u64,
        row_count: usize,
    ) -> bool {
        let device_ptr = device_memory.device_ptr();
        for _attempt in 0..2 {
            // Snapshot the extension basis under the lock.
            let (base_count, table_size) = {
                let cache = self
                    .read_state
                    .residency
                    .shard_pk_index
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let Some(entry) = cache.get(cache_key) else {
                    return false;
                };
                if entry.resident_device_ptr != device_ptr {
                    return false; // re-admit/rollover: a different buffer -> full rebuild
                }
                if entry.row_count >= row_count {
                    return true; // already current (or ahead: a fresher probe won)
                }
                let Some(data) = entry.index.as_ref() else {
                    return true; // DECLINED is monotone under appends: current by definition
                };
                (entry.row_count, data.hash_table.len() as u64)
            };
            // The builder sizes `table_size = next_pow2(2*count)`; extending past its own load
            // rule risks probe-cap overflows a fresh build would not have -> rebuild/resize.
            if (row_count as u64).saturating_mul(2) > table_size {
                return false;
            }
            let tail_len = row_count - base_count;
            let Ok(tail_keys) = device_memory
                .read_resident_i32_column(filter_offset + (base_count as u64) * 4, tail_len)
            else {
                return false; // read failure -> the rebuild path's conservative decline
            };
            if tail_keys.len() != tail_len {
                return false;
            }
            let mut cache = self
                .read_state
                .residency
                .shard_pk_index
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(entry) = cache.get_mut(cache_key) else {
                return false;
            };
            if entry.resident_device_ptr != device_ptr {
                return false;
            }
            if entry.row_count != base_count {
                continue; // a concurrent extender moved the base: re-snapshot and retry once
            }
            let Some(data) = entry.index.as_mut() else {
                return true;
            };
            if extend_int4_pk_hash_table_host(
                &mut data.hash_table,
                data.table_mask,
                data.hash_shift,
                &tail_keys,
                base_count,
            ) {
                extend_int4_pk_bloom_host(
                    &mut data.bloom_words,
                    data.bloom_num_bits,
                    data.bloom_num_hashes,
                    &tail_keys,
                );
                self.read_state
                    .residency
                    .pk_index_prober_extends
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                // A dup/overflow tail key: the shard is now dup-bearing — the same conclusion a
                // full rebuild reaches, recorded WITHOUT the O(shard) re-discovery (A2's monotone
                // decline discipline; only a ptr change can clear it).
                entry.index = None;
            }
            entry.row_count = row_count;
            return true;
        }
        false // two basis moves in a row: give up, the rebuild path is always correct
    }

    /// Sub-slice 3: probe the CACHED per-shard host PK index (hash + bloom) for `key`. Builds + caches the
    /// index ONCE per shard generation -- keyed `(table, shard_id, col_idx)`, VALIDATED by
    /// `resident_device_ptr` so a re-admit / rollover (new device buffer -> new ptr) misses and rebuilds
    /// against the live bytes (the R1 `wave_index` staleness discipline, per shard). Reuse makes a point
    /// lookup an O(1) host probe instead of a per-lookup DtoH + rebuild. Build happens OUTSIDE the cache lock
    /// (a concurrent rebuild of the same entry merely overwrites -- harmless, rare). A DECLINED shard
    /// (duplicate / oversize key column) is CACHED as `index: None` so it is not rebuilt every lookup.
    fn probe_shard_pk_index_cached(
        &self,
        table_name: &str,
        shard_id: u32,
        col_idx: usize,
        device_memory: &Arc<CudaResidentDeviceMemory>,
        filter_offset: u64,
        row_count: usize,
        key: i32,
    ) -> ShardPkProbe {
        let device_ptr = device_memory.device_ptr();
        let cache_key = (table_name.to_string(), shard_id, col_idx);
        // Fast path: a cached entry whose ptr still matches the live buffer -> probe under the lock.
        if let Some(result) =
            self.probe_shard_pk_index_fast_path(&cache_key, device_ptr, row_count, key)
        {
            return result;
        }
        // TYPE-COVERAGE track 1 (ledger #3): same ptr + larger live row_count = an in-place
        // append — EXTEND the cached index with the tail keys (O(delta)) instead of rebuilding
        // O(shard) per probe (the measured constrained-INSERT cliff: ~1ms prepare under
        // per-commit append churn). On success the entry is current -> the fast path answers.
        if self.try_extend_cached_shard_pk_index(
            &cache_key,
            device_memory,
            filter_offset,
            row_count,
        ) {
            if let Some(result) =
                self.probe_shard_pk_index_fast_path(&cache_key, device_ptr, row_count, key)
            {
                return result;
            }
        }
        // Miss / stale ptr: build OUTSIDE the lock (DtoH the key column + host hash + bloom), then publish.
        self.read_state
            .residency
            .pk_index_rebuilds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Ok(keys) = device_memory.read_resident_i32_column(filter_offset, row_count) else {
            return ShardPkProbe::Declined; // a read failure forces the conservative scan (not cached)
        };
        if keys.len() != row_count {
            return ShardPkProbe::Declined;
        }
        let index = match build_int4_pk_hash_table_host(&keys, row_count as u64) {
            Some((hash_table, table_mask, hash_shift)) => {
                // build_int4_pk_bloom_host only declines on 0 rows (excluded above) -> Some; the fallback
                // (0,0) makes `bloom_maybe_contains` conservatively "maybe" (never wrongly skips).
                let (bloom_words, bloom_num_bits, bloom_num_hashes) =
                    build_int4_pk_bloom_host(&keys).unwrap_or((Vec::new(), 0, 0));
                Some(CachedShardPkIndexData {
                    hash_table,
                    table_mask,
                    hash_shift,
                    bloom_words,
                    bloom_num_bits,
                    bloom_num_hashes,
                })
            }
            None => None, // duplicate / oversize key column -> declined (cached so we don't rebuild)
        };
        let entry = CachedShardPkIndex {
            resident_device_ptr: device_ptr,
            row_count,
            // Pin the buffer so its address can't be reused while cached (ABA guard; see the struct doc).
            _resident_guard: Arc::clone(device_memory),
            index,
        };
        let result = probe_cached_shard_pk(&entry, key);
        self.read_state
            .residency
            .shard_pk_index
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(cache_key, entry);
        result
    }

    /// Step 1 (lpb-for-shards): BATCHED per-shard PK probe. Ensures the shard's cached hash+bloom index is
    /// built + `(ptr,row_count)`-validated ONCE (not per needle), then probes EVERY needle against it under a
    /// SINGLE cache lock, calling `on_hit(needle_index, slot)` per Hit. Returns `false` if the shard DECLINES
    /// (duplicate / oversize key column, or a device read failure) -> the caller falls back to the scan for
    /// the whole batch (a hash holds one row/key; the scan returns every match). Same `(ptr,row_count)`
    /// validation + build-outside-the-lock discipline as the single-key `probe_shard_pk_index_cached`. A
    /// DECLINED shard's `index` is `None`, so `probe_cached_shard_pk` declines the FIRST needle -> no partial
    /// `on_hit` before a decline.
    fn probe_shard_pk_index_cached_batch<F: FnMut(u32, u32)>(
        &self,
        table_name: &str,
        shard_id: u32,
        col_idx: usize,
        device_memory: &Arc<CudaResidentDeviceMemory>,
        filter_offset: u64,
        row_count: usize,
        needles: &[i32],
        mut on_hit: F,
    ) -> bool {
        let device_ptr = device_memory.device_ptr();
        let cache_key = (table_name.to_string(), shard_id, col_idx);
        // Fast path: a COVERING cached entry -> probe ALL needles under ONE lock (ahead entries
        // slot-bound filtered, declined entries monotone — the single-probe fast-path rules).
        if let Some(answer) = self.probe_shard_pk_index_batch_fast_path(
            &cache_key,
            device_ptr,
            row_count,
            needles,
            &mut on_hit,
        ) {
            return answer;
        }
        // TYPE-COVERAGE track 1 (ledger #3): extend the cached index over an in-place append
        // (O(delta)) before falling back to the O(shard) rebuild.
        if self.try_extend_cached_shard_pk_index(
            &cache_key,
            device_memory,
            filter_offset,
            row_count,
        ) {
            if let Some(answer) = self.probe_shard_pk_index_batch_fast_path(
                &cache_key,
                device_ptr,
                row_count,
                needles,
                &mut on_hit,
            ) {
                return answer;
            }
        }
        // Miss / stale ptr: build OUTSIDE the lock, probe against the built entry, then publish it.
        self.read_state
            .residency
            .pk_index_rebuilds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Ok(keys) = device_memory.read_resident_i32_column(filter_offset, row_count) else {
            return false;
        };
        if keys.len() != row_count {
            return false;
        }
        let index = match build_int4_pk_hash_table_host(&keys, row_count as u64) {
            Some((hash_table, table_mask, hash_shift)) => {
                let (bloom_words, bloom_num_bits, bloom_num_hashes) =
                    build_int4_pk_bloom_host(&keys).unwrap_or((Vec::new(), 0, 0));
                Some(CachedShardPkIndexData {
                    hash_table,
                    table_mask,
                    hash_shift,
                    bloom_words,
                    bloom_num_bits,
                    bloom_num_hashes,
                })
            }
            None => None,
        };
        let entry = CachedShardPkIndex {
            resident_device_ptr: device_ptr,
            row_count,
            _resident_guard: Arc::clone(device_memory),
            index,
        };
        let mut declined = false;
        for (ni, &key) in needles.iter().enumerate() {
            match probe_cached_shard_pk(&entry, key) {
                ShardPkProbe::Hit(slot) => on_hit(ni as u32, slot),
                ShardPkProbe::Miss => {}
                ShardPkProbe::Declined => {
                    declined = true;
                    break;
                }
            }
        }
        self.read_state
            .residency
            .shard_pk_index
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(cache_key, entry);
        !declined
    }

    /// Step 1 (lpb-for-shards): the BATCHED generation-consistent locate. Loads the table's shards ONCE and,
    /// per shard, builds the descriptor + captures the (PINNED) buffer + deleted_by region ONCE, then probes
    /// ALL needles against that shard's cached index -> a `BatchShardGroup` per shard with >=1 hit, carrying
    /// the captured handles + the `(needle_index, slot)` hits. Same generation-consistency guarantee as
    /// `locate_resident_pk_via_shard_index_detailed`: every hit's slot is read from the exact pinned buffer it
    /// was resolved against. `None` (fall back to the scan) if ANY shard is invalid / declines (dup) or a
    /// needle hits >1 shard (a cross-shard duplicate — the scan returns every match). Filter column int4.
    fn locate_sharded_pk_batch(
        &self,
        table: &RelationalTable,
        filter_idx: usize,
        needles: &[i32],
    ) -> Option<Vec<BatchShardGroup>> {
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut groups: Vec<BatchShardGroup> = Vec::new();
        let mut hit_shard_count = vec![0u32; needles.len()];
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            let filter_offset =
                resident_device_int4_column_offset(&descriptor, table, filter_idx).ok()?;
            // D4: the buffer rides the loaded descriptor (one-snapshot capture).
            let device_memory = shard.device_memory.clone()?;
            let mut hits: Vec<(u32, u32)> = Vec::new();
            let ok = self.probe_shard_pk_index_cached_batch(
                &table.name,
                shard.shard_id,
                filter_idx,
                &device_memory,
                filter_offset,
                shard.row_count,
                needles,
                |ni, slot| hits.push((ni, slot)),
            );
            if !ok {
                return None; // this shard declined -> whole batch falls back to the scan
            }
            if hits.is_empty() {
                continue;
            }
            for &(ni, _) in &hits {
                hit_shard_count[ni as usize] += 1;
            }
            // D4: regions from the SAME loaded descriptor as the buffer.
            let deleted_by = shard.deleted_by_region.clone();
            let created_by = shard.created_by_region.clone();
            groups.push(BatchShardGroup {
                descriptor,
                device_memory,
                deleted_by,
                created_by,
                hits,
            });
        }
        // A needle that Hit in >1 shard is a cross-shard duplicate -> fall back (the scan returns every match).
        if hit_shard_count.iter().any(|&c| c > 1) {
            return None;
        }
        Some(groups)
    }

    /// Step 1 (lpb-for-shards): BATCHED cross-shard point-lookup GATHER — the throughput lever over the
    /// single-flight 3b route. Routes a batch of int4 `needles` through the cross-shard PK index
    /// (`locate_sharded_pk_batch`), then per shard-group gathers the projected int4 columns at the group's
    /// slots with ONE kernel + one bulk DtoH PER (shard, column) (`project_i32_rows_from_payload`) —
    /// amortizing the per-needle launch that caps the single-flight route — and applies the SV3b
    /// `deleted_by[slot] > read_txn_id` gate (one batched i64 gather per versioned shard). Scatters back to
    /// NEEDLE ORDER (unique-PK -> each needle 0 or 1 row). `None` (caller falls back to the per-needle route)
    /// on decline / dup / int4 shape / error. The sharded path is NULL-blind (raw i32), byte-identical to the
    /// single-flight route by construction. Increments `sharded_point_batch_hits`. The read snapshot is
    /// `committed_seq()` (matches the single-flight route's pin when no writes interleave).
    pub(crate) fn gather_sharded_int4_point_lookups_batched(
        &self,
        // SC5 rider (ADR-013 adjunct): the READER'S pinned boundary — previously this fn re-read
        // `committed_seq()` internally, breaking the statement's catalog<->data co-pinning.
        read_boundary: Index,
        table: &RelationalTable,
        filter_idx: usize,
        selected_indexes: &[usize],
        needles: &[i32],
    ) -> Option<BatchedShardProjection> {
        if selected_indexes.is_empty() {
            return None;
        }
        if table.columns.get(filter_idx).map(|c| c.ty) != Some(SqlType::Int4) {
            return None;
        }
        for &idx in selected_indexes {
            if table.columns.get(idx).map(|c| c.ty) != Some(SqlType::Int4) {
                return None;
            }
        }
        // M3-for-shards: the batched gather (GPU dense-emit + host paths) emits RAW i32 with NO validity
        // channel, so a NULL in the FILTER or any PROJECTED column would surface as a phantom 0. NULLs in
        // UNREFERENCED columns are irrelevant: neither the device index nor the result kernel reads those
        // bytes. Decline iff a referenced column has a bitmap; the caller's per-query NULL-aware scan serves
        // that shape. This metadata-only eligibility check performs no host relational decision.
        let mut referenced_names: std::collections::BTreeSet<&str> = selected_indexes
            .iter()
            .filter_map(|&idx| table.columns.get(idx).map(|column| column.name.as_str()))
            .collect();
        referenced_names.insert(table.columns.get(filter_idx)?.name.as_str());
        if self
            .read_state
            .residency
            .shards
            .load()
            .get(&table.name)
            .is_some_and(|shards| {
                shards.iter().any(|shard| {
                    shard
                        .resident_device_null_columns
                        .iter()
                        .any(|layout| referenced_names.contains(layout.name.as_str()))
                })
            })
        {
            return None;
        }
        // Sub-slice 8: PREFER the fully-GPU dense-emit path (device-resident per-shard index + the
        // `gpu_db_resident_i32_index_probe_dense` kernel probes+gathers+emits on the GPU — no host per-needle
        // probe, one bulk DtoH per shard). Returns None -> fall through to the host-probe path below when a
        // shape/ncols is unsupported (>4 cols), or the index declines / errors. The dense kernel now evaluates
        // created_by/deleted_by visibility itself; the host gather remains a correctness fallback, not the normal
        // append-window route. Byte-identical either way.
        if let Some(gpu) = self.gather_sharded_int4_point_lookups_batched_gpu(
            table,
            filter_idx,
            selected_indexes,
            needles,
            read_boundary,
        ) {
            return Some(gpu);
        }
        let read_txn_id = read_boundary as i64;
        let groups = self.locate_sharded_pk_batch(table, filter_idx, needles)?;
        let ncols = selected_indexes.len();
        // per needle: the projected row (Some) or absent/hidden (None). Unique-PK -> <=1 row/needle.
        let mut per_needle: Vec<Option<Vec<i32>>> = vec![None; needles.len()];
        for group in &groups {
            let slots: Vec<u64> = group.hits.iter().map(|&(_, slot)| slot as u64).collect();
            // SV3b visibility: ONE batched i64 gather of deleted_by at the slots (versioned shard), else live.
            let mut visible: Vec<bool> = match &group.deleted_by {
                Some(region) => {
                    let dby = region.project_i64_rows_from_payload(0, &slots).ok()?;
                    if dby.len() != slots.len() {
                        return None;
                    }
                    dby.iter().map(|&d| d > read_txn_id).collect()
                }
                None => vec![true; slots.len()],
            };
            // SV6 lower bound: AND `created_by <= read_txn_id` (one batched i64 gather) so an
            // UPDATE-appended version whose commit exceeds the read snapshot stays hidden (the
            // double-read gate). An un-stamped shard (no region) is born-visible.
            if let Some(region) = &group.created_by {
                let cby = region.project_i64_rows_from_payload(0, &slots).ok()?;
                if cby.len() != slots.len() {
                    return None;
                }
                for (v, &c) in visible.iter_mut().zip(cby.iter()) {
                    *v = *v && c <= read_txn_id;
                }
            }
            // ONE batched i32 gather per projected column at the group's slots.
            let mut col_values: Vec<Vec<i32>> = Vec::with_capacity(ncols);
            for &idx in selected_indexes {
                let col_base =
                    resident_device_int4_column_offset(&group.descriptor, table, idx).ok()?;
                let vals = group
                    .device_memory
                    .project_i32_rows_from_payload(col_base, &slots)
                    .ok()?;
                if vals.len() != slots.len() {
                    return None;
                }
                col_values.push(vals);
            }
            // Scatter to needle order (unique-PK -> at most one visible hit per needle).
            for (j, &(ni, _)) in group.hits.iter().enumerate() {
                if !visible[j] {
                    continue;
                }
                let mut row = Vec::with_capacity(ncols);
                for col in col_values.iter() {
                    row.push(col[j]);
                }
                per_needle[ni as usize] = Some(row);
            }
        }
        // Flatten to needle order + per-needle ranges (row-major, ncols wide).
        let mut values: Vec<i32> = Vec::new();
        let mut needle_ranges: Vec<(u32, u32)> = Vec::with_capacity(needles.len());
        for row in &per_needle {
            let start = (values.len() / ncols) as u32;
            match row {
                Some(r) => {
                    values.extend_from_slice(r);
                    needle_ranges.push((start, 1));
                }
                None => needle_ranges.push((start, 0)),
            }
        }
        self.read_state
            .residency
            .sharded_point_batch_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(BatchedShardProjection {
            ncols,
            values,
            needle_ranges,
        })
    }

    /// Sub-slice 8 (GPU-native probe): ensure the shard's PK hash index is resident ON THE DEVICE (uploaded
    /// once per shard generation), returning `(device_index, table_mask, hash_shift)` for the dense-emit
    /// probe kernel. Mirrors R1's `build_wave_resident_int4_index` per shard: DtoH the key column -> build the
    /// host hash table (`(key<<32)|(row+1)`) -> HtoD upload via `retain_device_memory_copy` -> cache keyed
    /// `(table, shard_id, col_idx)` validated by `(ptr, row_count)` + the ABA `_resident_guard` pin. `None`
    /// (caller falls back to the host path) when the shard is empty / oversize, the DtoH fails, the upload
    /// fails, or the key column has DUPLICATES (the hash declines — cached as `device_index: None` so it is
    /// not rebuilt every batch).
    fn ensure_shard_pk_device_index(
        &self,
        table: &RelationalTable,
        table_name: &str,
        shard_id: u32,
        // COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): `key_id` identifies WHICH unique index this index
        // serves — a single-column key's catalog COLUMN INDEX (byte-compatible with every prior cache
        // entry), or `COMPOUND_KEY_ID_FLAG | ordinal` for a compound key. `positions` are the ordered
        // catalog indices of the key column(s); `offsets` are those columns' capacity-strided
        // i32-section byte offsets, caller-computed. NON-lanes builds from the caller's shard, so the
        // caller offsets are exact. UNDER LANES the rebuild reads the LIVE shard whose capacity a
        // concurrent re-admit may have GROWN since the caller's snapshot — so offsets are RECOMPUTED
        // from the live shard here (a capacity-strided offset for any key column past int4-ordinal 0
        // would otherwise mis-address the buffer -> garbage fingerprints -> a missed duplicate). One
        // offset = single column (raw keys); >1 = compound (the per-row values FOLD into the surrogate
        // fingerprint the index stores as an opaque key).
        key_id: usize,
        positions: &[usize],
        device_memory: &Arc<CudaResidentDeviceMemory>,
        offsets: &[u64],
        // COMPOUND KEYS (text): the per-key-column BLOB byte offsets, parallel to `offsets` — nonzero only
        // for a TEXT column (its blob), 0 for fixed-width columns. Recomputed from the live shard under lanes.
        blob_offsets: &[u64],
        row_count: usize,
    ) -> Option<(Arc<CudaResidentDeviceMemory>, u32, u32)> {
        let device_ptr = device_memory.device_ptr();
        let cache_key = (table_name.to_string(), shard_id, key_id);
        // Fast path: a valid cached device index -> return it (or None if it declined at build).
        {
            let cache = self
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(entry) = cache.get(&cache_key) {
                // E2.5b-2: tolerate a NEWER index than the caller's shard snapshot
                // (entry.row_count >= row_count, same buffer). Concurrent lane
                // applies extend the index at the append chokepoint; a probe
                // against a superset is safe — extra rows only add count>0 hits,
                // which the authoritative visible_row_with_value check filters at
                // the needle's read snapshot. Requiring EQUALITY here caused a
                // rebuild ping-pong under lanes (a stale-snapshot rebuild kept
                // clobbering the newer entry): measured 16.5ms/wave validate.
                if entry.resident_device_ptr == device_ptr && entry.row_count >= row_count {
                    return entry
                        .device_index
                        .clone()
                        .map(|di| (di, entry.table_mask, entry.hash_shift));
                }
            }
        }
        // Miss / stale ptr: rebuild. UNDER LANES the rebuild takes the device-apply
        // lock and uses the LIVE shard basis: a rebuild at a stale caller snapshot
        // while append-side extensions continue would leave a HOLE (rows S..B
        // absent from the index) => false-negative duplicate checks. The guard
        // excludes applies during the rebuild, and the live count re-converges the
        // extension chain (entry.row_count == the next apply's base) instead of
        // looping through rebuild-per-wave. Cache HITS above stay lock-free.
        // U1 WAL-FIRST: the apply LEADER already holds `device_apply_lock` (the delete
        // visible-locate rebuilds under it), so re-taking it here would self-deadlock — skip the
        // guard when the leader thread-local is set; the leader's exclusivity already gives the
        // rebuild what the guard provides.
        let apply_leader = crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(|f| f.get());
        let _lane_rebuild_guard = if apply_leader {
            None
        } else {
            self.intent_lanes.as_ref().map(|lanes| {
                lanes
                    .device_apply_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            })
        };
        let (build_memory, build_offsets, build_blob_offsets, build_row_count, build_capacity_rows) =
            if self.intent_lanes.is_some() {
                // Re-check under the guard: another prober may have rebuilt already.
                {
                    let cache = self
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if let Some(entry) = cache.get(&cache_key) {
                        if entry.row_count >= row_count {
                            return entry
                                .device_index
                                .clone()
                                .map(|di| (di, entry.table_mask, entry.hash_shift));
                        }
                    }
                }
                let shards = self.read_state.residency.shards.load();
                let live = shards
                    .get(table_name)?
                    .iter()
                    .find(|shard| shard.shard_id == shard_id)?
                    .clone();
                let live_memory = live.device_memory.clone()?;
                let live_rows = live.row_count;
                // AUDIT FIX (compound): recompute the key-column offsets from the LIVE shard — a
                // concurrent re-admit may have grown its capacity since the caller's snapshot, and
                // the offsets are capacity-strided, so the caller's offsets could mis-address every
                // key column past int4-ordinal 0.
                let live_offsets = positions
                    .iter()
                    .map(|&p| shard_fixed_width_key_offset(&live, table, p))
                    .collect::<Option<Vec<u64>>>()?;
                // COMPOUND KEYS (text): the blob byte offsets are ALSO capacity/layout-dependent, so
                // recompute them from the live shard alongside the fixed-width offsets.
                let live_blob_offsets = positions
                    .iter()
                    .map(|&p| shard_key_column_blob_offset(&live, table, p))
                    .collect::<Option<Vec<u64>>>()?;
                // CAPACITY-SIZED INDEX: size the hash table once for the shard's
                // full capacity (clamped to the builder's 2^30 slot limit via the
                // sizing_rows argument), so capacity-exhaustion rebuilds are
                // impossible for the shard's lifetime — only ptr changes
                // (re-admission) rebuild, and the floor above makes those rare.
                let capacity_rows = live.capacity as u64;
                (
                    live_memory,
                    live_offsets,
                    live_blob_offsets,
                    live_rows,
                    capacity_rows,
                )
            } else {
                // NON-lanes: the build reads the CALLER's `device_memory` (same generation the caller
                // computed `offsets` against, no concurrent re-admit), so the caller offsets are exact.
                (
                    Arc::clone(device_memory),
                    offsets.to_vec(),
                    blob_offsets.to_vec(),
                    row_count,
                    0_u64,
                )
            };
        let device_ptr = build_memory.device_ptr();
        self.read_state
            .residency
            .lane_diag_rebuilds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let row_count = build_row_count;
        let row_count_u64 = row_count as u64;
        if row_count == 0 || row_count_u64 >= u32::MAX as u64 {
            return None;
        }
        // COMPOUND KEYS: obtain the per-row keys the index stores. A single-column key reads its one
        // resident column verbatim (raw keys), byte-identical to the prior path. A COMPOUND key folds
        // its columns into the surrogate fingerprint ON THE DEVICE (`submit_compound_fold_fingerprints`,
        // byte-matching the host `compound_key_fingerprint`) — the raw key columns are NEVER read back
        // to the host to be hashed (the charter close); the host reads only the derived fingerprint
        // column, exactly as the single-column build reads its one key column.
        let keys: Vec<i32> = if build_offsets.len() == 1 {
            let keys = build_memory
                .read_resident_i32_column(build_offsets[0], row_count)
                .ok()?;
            if keys.len() != row_count {
                return None;
            }
            keys
        } else {
            // Per-column WORD widths (i32-section -> 1, i64 section -> 2), parallel to `build_offsets`
            // in `positions` order; the device fold reads `widths[k]` words per column.
            let widths = positions
                .iter()
                .map(|&p| crate::engine_residency::key_column_width_words(table.columns[p].ty))
                .collect::<Option<Vec<u32>>>()?;
            let fps = build_memory
                .submit_compound_fold_fingerprints(
                    device_ptr,
                    &build_offsets,
                    &widths,
                    &build_blob_offsets,
                    row_count,
                )
                .ok()?;
            if fps.len() != row_count {
                return None;
            }
            fps
        };
        // U1 (visibility-aware rebuild): read the shard's deleted_by stamps (absent region =
        // all-live) and skip rows dead at or below the GC boundary — see
        // `build_int4_pk_hash_table_host_visible`. Boundary = the oldest registered snapshot
        // (or committed_seq if none): a row dead at or below it is invisible to every current
        // AND future reader (future snapshots bind at >= committed_seq >= any published stamp).
        let deleted_stamps: Option<Vec<u64>> = self
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table_name.to_string(), shard_id))
            .and_then(|region| region.read_resident_u64_column(0, row_count).ok());
        let gc_boundary = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .oldest()
            .unwrap_or_else(|| self.committed_seq());
        // GROWTH HEADROOM (E2.5b-2): size the rebuilt table for 2x the current
        // rows, not 1x. The builder's natural rule next_pow2(rows*2) can land
        // capacity EXACTLY at the current row count (whenever rows*2 is a power
        // of two), so the very next append re-drops the entry and the probe
        // rebuilds again — measured as 222 O(rows) rebuilds in one 8s lane run
        // (~seconds of DtoH+build+HtoD). Sizing for 2x makes the drop->rebuild
        // cadence geometric: log2(final/initial) rebuilds per shard lifetime.
        // The table only ever holds `keys` (real rows); the extra slots are
        // empty probe space (sparser = faster linear probing).
        let sizing_rows = if self.intent_lanes.is_some() {
            // lanes: size for the shard's capacity once (see live rebuild note)
            row_count_u64
                .saturating_mul(2)
                .max(build_capacity_rows.saturating_mul(2))
                .min(1_u64 << 29)
        } else {
            row_count_u64.saturating_mul(2)
        };
        // The host hash build is outside the allocation lock. Only its retained GPU result affects
        // the residency cap, so serialize from this point through cache publication.
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (device_index, table_mask, hash_shift) = match build_int4_pk_hash_table_host_visible(
            &keys,
            sizing_rows,
            deleted_stamps.as_deref(),
            gc_boundary,
            // F3/U4: place MVCC version twins (dead-old + live-new sharing a pk) rather than
            // declining — BUT only on a VERSIONED shard (one that carries a `deleted_by` region).
            // A version twin can only exist where a delete/update tombstoned the old, so a versioned
            // shard's same-key duplicates are twins the dup-tolerant visible/write-locate AND dense-read
            // probes resolve by advancing past invisible hits. A DELETE-FREE shard has no versions, so a
            // same-key duplicate is a genuine
            // DATA duplicate (a non-unique-key table) that MUST still decline the whole shard — the
            // first-match probe cannot resolve it. `deleted_stamps.is_some()` is exactly that gate.
            deleted_stamps.is_some(),
        ) {
            Some((index, table_mask, hash_shift)) => {
                let index_bytes: Vec<u8> =
                    index.iter().flat_map(|entry| entry.to_le_bytes()).collect();
                let runtime = self.cuda_driver_probe_runtime();
                let gpu_id = build_memory.metadata().gpu_id;
                if self
                    .relational_residency_budget_bytes(gpu_id)
                    .is_some_and(|budget| {
                        self.relational_resident_bytes_for_gpu(gpu_id)
                            .saturating_add(index_bytes.len() as u64)
                            > budget
                    })
                {
                    return None;
                }
                // An upload failure (e.g. OOM) is TRANSIENT -> return None WITHOUT caching (retry next
                // batch); the caller falls back to the host path meanwhile.
                let Ok(mem) = runtime.retain_device_memory_copy(gpu_id, &index_bytes) else {
                    return None;
                };
                if self
                    .relational_residency_budget_bytes(gpu_id)
                    .is_some_and(|budget| {
                        self.relational_resident_bytes_for_gpu(gpu_id)
                            .saturating_add(mem.metadata().allocated_bytes)
                            > budget
                    })
                {
                    return None;
                }
                (Some(Arc::new(mem)), table_mask, hash_shift)
            }
            // Duplicate / oversize key column -> declined; CACHE `None` so it is not rebuilt every batch.
            None => (None, 0, 0),
        };
        let result = device_index.clone().map(|di| (di, table_mask, hash_shift));
        let entry = CachedShardPkDeviceIndex {
            resident_device_ptr: device_ptr,
            row_count,
            _resident_guard: Arc::clone(&build_memory),
            device_index,
            table_mask,
            hash_shift,
        };
        {
            let mut cache = self
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Do not replace a NEWER entry built/extended concurrently (same
            // ping-pong hazard as the hit path, from the insert side).
            let newer_exists = cache.get(&cache_key).is_some_and(|existing| {
                existing.resident_device_ptr == device_ptr && existing.row_count > row_count
            });
            if !newer_exists {
                cache.insert(cache_key, entry);
            }
        }
        result
    }

    /// Sub-slice 8 (GPU-native probe): the FULLY-GPU batched cross-shard point-lookup — the charter-faithful
    /// completion of lpb-for-shards. It ensures each shard's device-resident PK index, then launches ONE
    /// multi-shard kernel which probes, applies MVCC visibility, gathers, and dense-emits in needle order.
    /// There is no per-shard launch, per-needle host probe, or host merge; completion performs one flat status
    /// compaction over the single needle-indexed output.
    ///
    /// Returns `None` (the caller falls back to the host-probe `gather_sharded_int4_point_lookups_batched`
    /// body, which applies the same visibility gates) when: the projection is >4 int4 columns (the dense
    /// kernel gathers <=4); a shard is invalid; the device index declines / fails; a needle has >1 VISIBLE
    /// match (uniqueness violation); or any device error. The
    /// DELETE-FREE majority (incl. the benchmark) takes this fully-GPU path. Increments
    /// `sharded_point_gpu_probe_hits` + `sharded_point_batch_hits`.
    fn gather_sharded_int4_point_lookups_batched_gpu(
        &self,
        table: &RelationalTable,
        filter_idx: usize,
        selected_indexes: &[usize],
        needles: &[i32],
        // D3: the reader's pinned boundary is consumed by the dense kernel's per-hit visibility gate.
        read_boundary: Index,
    ) -> Option<BatchedShardProjection> {
        let ncols = selected_indexes.len();
        // The dense kernel gathers 1..=4 projection columns.
        if ncols == 0 || ncols > 4 {
            return None;
        }
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let runtime_snapshot = self.router.runtime().snapshot();
        let n = needles.len();
        // Build the per-shard descriptor list for the MULTI-SHARD kernel: for each non-empty,
        // valid shard, ensure its DEVICE index + capture (device buffer, device index, mask, shift, capacity-
        // strided projection offsets, row_count). ONE kernel then probes ALL shards per needle + dense-emits a
        // single needle-indexed output (no S*N DtoH, no host merge).
        let mut probe_shards: Vec<gpu_db_execution::MultiShardProbeShard> = Vec::new();
        for shard in table_shards.iter() {
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            if shard.row_count == 0 {
                continue;
            }
            // D3/D4: version regions come from this SAME loaded shard descriptor and ride the kernel submission
            // as pinned Arcs. The dense probe applies `created_by <= read_boundary < deleted_by` per candidate,
            // including readers pinned inside append publication and dead/live version twins in one hash index.
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            let filter_offset =
                resident_device_int4_column_offset(&descriptor, table, filter_idx).ok()?;
            // D4: the buffer rides the loaded descriptor (one-snapshot capture).
            let device_memory = shard.device_memory.clone()?;
            let (device_index, table_mask, hash_shift) = self.ensure_shard_pk_device_index(
                table,
                &table.name,
                shard.shard_id,
                filter_idx,
                std::slice::from_ref(&filter_idx),
                &device_memory,
                &[filter_offset],
                &[0], // single-column key -> blob offsets unused (fixed-width fold path)
                shard.row_count,
            )?;
            let mut projection_offsets: Vec<u64> = Vec::with_capacity(ncols);
            for &idx in selected_indexes {
                projection_offsets
                    .push(resident_device_int4_column_offset(&descriptor, table, idx).ok()?);
            }
            // Sub-slice 8 v3: the filter column's zone map [min,max] for on-device pruning, read the SAME way
            // the scan's `shard_zone_map_excludes` does (by column NAME from the int4-ordinal-compacted stats).
            // No stat for the column -> (i32::MIN, i32::MAX) = always in-range (matching the scan, which keeps
            // a shard with no zone-map stat). NULLs are excluded from the stat -> the kernel's keep-shard-0
            // fallback handles a needle 0 that would match a NULL-stored-as-0 row in an out-of-[min,max] shard.
            let (min, max) = table
                .columns
                .get(filter_idx)
                .and_then(|col| {
                    shard
                        .resident_device_int4_column_stats
                        .iter()
                        .find(|s| s.name == col.name)
                })
                .map(|s| (s.min, s.max))
                .unwrap_or((i32::MIN, i32::MAX));
            probe_shards.push(gpu_db_execution::MultiShardProbeShard {
                resident: device_memory,
                index: device_index,
                table_mask,
                hash_shift,
                projection_offsets,
                row_count: shard.row_count as u64,
                created_by: shard.created_by_region.clone(),
                deleted_by: shard.deleted_by_region.clone(),
                min,
                max,
            });
        }
        // Compact a needle-indexed dense output (status[i]==1 -> 1 row, else 0) in ONE pass -- the SAME
        // compaction the single-buffer dense path uses; the kernel already wrote needle order, so there is NO
        // cross-shard host merge. Empty output (no non-empty shards) -> all needles absent.
        let (values, needle_ranges) = if probe_shards.is_empty() {
            (Vec::new(), vec![(0u32, 0u32); n])
        } else {
            // `self` context = the first shard's buffer (allocation/launch only; the kernel reads each shard's
            // own ptr from the descriptor array). ONE kernel launch, ONE bulk DtoH.
            let submission = probe_shards[0]
                .resident
                .submit_multi_shard_i32_index_probe_dense(
                    &probe_shards,
                    needles,
                    read_boundary as u64,
                )
                .ok()?;
            let binary_mode = submission.multi_shard_binary_mode;
            let (cols, _elapsed) = submission.complete_detached_columnar().ok()?;
            if cols.status.len() != n {
                return None;
            }
            let mut values: Vec<i32> = Vec::with_capacity(n * ncols);
            let mut needle_ranges: Vec<(u32, u32)> = Vec::with_capacity(n);
            for i in 0..n {
                let start = (values.len() / ncols) as u32;
                match cols.status[i] {
                    1 => {
                        // Found in exactly one shard -> its projected row.
                        values.extend_from_slice(&cols.values[i * ncols..(i + 1) * ncols]);
                        needle_ranges.push((start, 1));
                    }
                    2 => needle_ranges.push((start, 0)), // absent in every shard
                    // 3 = the multi-shard kernel found this needle in >1 shard (a CROSS-shard duplicate): it
                    // can emit only one slot, and the scan returns EVERY match -> decline the WHOLE batch to
                    // the host path (which also declines cross-shard dups -> the per-query scan). 0 = a thread
                    // that never wrote (gap guard) -> also decline (never a wrong result).
                    _ => return None,
                }
            }
            if binary_mode {
                self.read_state
                    .residency
                    .sharded_point_binary_route_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            (values, needle_ranges)
        };
        self.read_state
            .residency
            .sharded_point_gpu_probe_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.read_state
            .residency
            .sharded_point_batch_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(BatchedShardProjection {
            ncols,
            values,
            needle_ranges,
        })
    }

    /// Step 1 (lpb-for-shards) benchmark + telemetry entry: resolve the table + columns, run the BATCHED
    /// cross-shard point-lookup gather over `needles`, and return the number of needles that materialized a
    /// row (a sanity signal for the benchmark), or `None` if the batched path declined. `sharded_point_batch_hits`
    /// counts the served batches.
    pub fn bench_sharded_point_lookup_batch(
        &self,
        table_name: &str,
        filter_col: &str,
        proj_cols: &[String],
        needles: &[i32],
    ) -> Option<usize> {
        let table = self.relational_catalog_table(table_name)?;
        let filter_idx =
            crate::rel_exec_helpers::relational_column_index(&table, filter_col).ok()?;
        let mut selected_indexes = Vec::with_capacity(proj_cols.len());
        for c in proj_cols {
            selected_indexes
                .push(crate::rel_exec_helpers::relational_column_index(&table, c).ok()?);
        }
        let proj = self.gather_sharded_int4_point_lookups_batched(
            self.committed_seq(),
            &table,
            filter_idx,
            &selected_indexes,
            needles,
        )?;
        Some(proj.needle_ranges.iter().filter(|&&(_, c)| c > 0).count())
    }

    /// lpb-for-shards WIRING: serve a shard-resident int4 point-lookup BATCH as ONE dispatchable
    /// `RelationalRetainedBatchResult` — the SAME type the single-buffer lpb coalescer produces, so the facade
    /// batcher's `distribute_results_batched` handles it UNCHANGED (columns mapped once, sliced per needle).
    /// Binds `select`, verifies it is a shard-resident single int4-`Eq` point lookup, runs
    /// `gather_sharded_int4_point_lookups_batched` over `needles`, and wraps the projection with the shared
    /// schema. Returns `None` (the batcher keeps its existing behavior — byte-identical) when the flag is OFF,
    /// the table is not shard-resident, the shape is not a single int4 equality, or the gather declines
    /// (dup / non-int4 / error). `needles` must be DISTINCT (the caller dedups), per the gather's contract.
    /// `access_path` is a label (`FullTableScan`) — the batcher's dispatch consumes only `columns` +
    /// `needle_values`, never `access_path`.
    pub fn submit_sharded_point_lookups_batched(
        &self,
        select: &Select,
        needles: &[i32],
    ) -> Option<RelationalRetainedBatchResult> {
        if !self.shard_batched_point_read_enabled() {
            return None;
        }
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select).ok()?;
        if self.resident_shard_count(&table.name) == 0 {
            return None; // not shard-resident -> the batcher's single-buffer / per-query path
        }
        let (filter_idx, _needle) = crate::engine_expr::shard_point_lookup_int4_eq(&bound, &table)?;
        // SC5 rider: the gather reads at the STATEMENT'S pinned boundary (was: an internal
        // committed_seq re-read that broke catalog<->data co-pinning).
        let proj = self.gather_sharded_int4_point_lookups_batched(
            copin_s,
            &table,
            filter_idx,
            &bound.selected_indexes,
            needles,
        )?;
        let gpu_id = self
            .read_state
            .residency
            .shards
            .load()
            .get(&table.name)
            .and_then(|s| s.first())
            .map(|s| s.gpu_id)
            .unwrap_or(0);
        Some(RelationalRetainedBatchResult {
            columns: Arc::new(bound.selected_columns),
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
            gpu_id,
            values: proj.values,
            ncols: proj.ncols,
            needle_ranges: proj.needle_ranges,
        })
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
        // `observe_kernel_exec_ms` is recorded from the batch WALL time (`batch_micros`) — it models "a batch
        // executed on the GPU" and keeps `kernel_exec_samples` comparable across the scan and index routes.
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
        // ADR-009 R1: the deferred (lpb/scan/R1-index) probe drains its enqueued GPU submission here, yielding
        // a `Vec<CudaI32BatchProjectionRow>`. Everything downstream (stable sort by `row_index`,
        // `SqlValue::Int4` mapping, metrics, results assembly) is shared, so atomic and dense are byte-
        // identical by construction. This per-needle path is COLD (the hot batcher uses the batched
        // completion); it bridges the columnar dense form back to per-row via `into_rows` (DECISIONS "Tail
        // latency").
        let (projected_rows, kernel_event_elapsed_us) = match pending.payload {
            DeferredProbe::Atomic(submission) => submission
                .complete_detached()
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?,
            DeferredProbe::Dense(submission) => {
                let (columns, elapsed) =
                    submission.complete_detached_columnar().map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                (columns.into_rows(), elapsed)
            }
        };
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
        // Materialize FLAT per needle (DECISIONS "Result-path optimization" step-1): group the matched rows
        // by needle BY REFERENCE (no per-row Vec), sort each needle's slice by `row_index` for the
        // deterministic byte-identical order (the stable-order contract above), then flatten the projected
        // i32 values straight into a row-major `RowBlock` — skipping the per-row `Vec<SqlValue>` boxing +
        // `Vec<Vec<..>>` assembly that capped end-to-end read at ~7.8M (153ns/row -> ~2ns/row).
        let ncols = pending.selected_indexes.len();
        let mut by_needle: Vec<Vec<(u64, &[i32])>> = vec![Vec::new(); pending.needle_count];
        for projected in &projected_rows {
            by_needle[projected.needle_index]
                .push((projected.row_index, projected.values.as_slice()));
        }
        let row_blocks: Vec<RowBlock> = by_needle
            .into_iter()
            .map(|mut slice| {
                slice.sort_by_key(|(row_index, _)| *row_index);
                let mut values = Vec::with_capacity(slice.len() * ncols);
                for (_, vals) in slice {
                    values.extend(vals.iter().copied().map(SqlValue::Int4));
                }
                RowBlock::flat(values, ncols)
            })
            .collect();
        let materialization_micros = materialize_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        let total_rows = row_blocks.iter().map(RowBlock::len).sum::<usize>();
        let table_name = pending.table.name.clone();
        // Cold path: stamp each needle's result with the shared schema (refcount-cloned per needle here, at
        // completion, instead of N times at submit — DECISIONS "Result-path optimization").
        let results = row_blocks
            .into_iter()
            .map(|rows| RelationalSelectResult {
                columns: Arc::clone(&pending.shared_columns),
                rows,
                planned_target: DeviceTarget::Gpu(pending.snapshot_gpu_id),
                executed_target: DeviceTarget::Gpu(pending.snapshot_gpu_id),
                fallback_reason: None,
                access_path: Arc::clone(&pending.shared_access_path),
            })
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

    /// BATCHED completion (DECISIONS "Result-path optimization"): like
    /// `complete_relational_retained_read_submission` but returns a SINGLE flat `RelationalRetainedBatchResult`
    /// (shared schema once + one flat `RowBlock` over all needles + per-needle ranges) the batcher slices per
    /// needle — avoiding the N per-needle `RelationalSelectResult` structs + N grouping Vecs + 2N Arc clones +
    /// N column re-maps that capped end-to-end point reads below the GPU drain. An already-`Ready` submission
    /// (general query / empty batch) folds its per-result rows into the batched layout.
    pub fn complete_relational_retained_read_submission_batched(
        &self,
        submission: RelationalRetainedReadSubmission,
    ) -> Result<RelationalRetainedBatchResult, ExecuteError> {
        match submission.inner {
            RelationalRetainedReadSubmissionInner::PendingInt4Projection(pending) => {
                Self::complete_int4_projection_batched_detached(*pending)
            }
            RelationalRetainedReadSubmissionInner::Ready(results) => {
                Ok(Self::fold_ready_results_batched(results))
            }
        }
    }

    /// Build the batched result from a drained int4 point-read submission: ONE sort by
    /// `(needle_index, row_index)` (groups needles contiguously AND gives each needle's rows the ascending
    /// row_index order the byte-identity contract requires) + ONE flat value buffer + per-needle ranges. The
    /// per-needle rows are byte-identical to `..._detached`'s `RelationalSelectResult.rows`.
    pub(crate) fn complete_int4_projection_batched_detached(
        pending: RelationalRetainedInt4ProjectionSubmission,
    ) -> Result<RelationalRetainedBatchResult, ExecuteError> {
        let (projected, _kernel_event_elapsed_us) = match pending.payload {
            DeferredProbe::Atomic(submission) => submission
                .complete_detached_columnar()
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?,
            DeferredProbe::Dense(submission) => submission
                .complete_detached_columnar()
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?,
        };
        let n = pending.needle_count;
        let ncols = pending.selected_indexes.len();
        Ok(Self::assemble_batched_rows(
            &projected,
            n,
            ncols,
            pending.shared_columns,
            pending.shared_access_path,
            pending.snapshot_gpu_id,
        ))
    }

    /// Group the COLUMNAR matched rows (in kernel emit order) into the flat needle-ordered
    /// `RelationalRetainedBatchResult` — reading the three flat arrays directly, NO per-row `Vec`
    /// (DECISIONS "Tail latency"). Pure (no GPU) so it is unit-testable with a hand-built UNSORTED input —
    /// the ONLY way to prove the within-needle sort is NECESSARY (a GPU fixture's 2-row emit order
    /// coincidentally equals ascending row_index, so a `no-sort` regression slips past the differentials — P3).
    pub(crate) fn assemble_batched_rows(
        projected: &CudaI32BatchProjectionColumns,
        n: usize,
        ncols: usize,
        columns: Arc<Vec<RelationalColumn>>,
        access_path: Arc<RelationalAccessPath>,
        gpu_id: u16,
    ) -> RelationalRetainedBatchResult {
        // DENSE LAYOUT (DECISIONS "lpb read levers" #1): `values` is one slot per needle (gaps), `status[i]`
        // is 1 (found) / 2 (not-found). Compact in ONE sequential pass — read slot `i` in order (cache-
        // friendly), keep status==1, write packed. No host scatter, no `needle_indices` (slot == needle), no
        // count. (The atomic/wave compacted form has empty `status` and takes the scatter path below.)
        if !projected.status.is_empty() {
            debug_assert_eq!(
                projected.status.len(),
                n,
                "dense status must be one per needle"
            );
            let mut values = Vec::with_capacity(projected.values.len());
            let mut needle_ranges = Vec::with_capacity(n);
            let mut acc = 0u32;
            for i in 0..n {
                debug_assert!(
                    projected.status[i] != 0,
                    "dense index-probe: status slot {i} never written (gap) — undrained kernel",
                );
                let present = projected.status[i] == 1;
                needle_ranges.push((acc, u32::from(present)));
                if present {
                    values.extend_from_slice(&projected.values[i * ncols..i * ncols + ncols]);
                    acc += 1;
                }
            }
            return RelationalRetainedBatchResult {
                columns,
                access_path,
                gpu_id,
                values,
                ncols,
                needle_ranges,
            };
        }
        let nrows = projected.nrows();
        // Per-needle counts -> prefix-sum ranges + whether ANY needle matched >1 row.
        let mut counts = vec![0u32; n];
        for &needle_index in &projected.needle_indices {
            counts[needle_index as usize] += 1;
        }
        let mut needle_ranges = Vec::with_capacity(n);
        let mut acc = 0u32;
        let mut any_multi = false;
        for &c in &counts {
            needle_ranges.push((acc, c));
            acc += c;
            any_multi |= c > 1;
        }
        let total = acc as usize;
        let values = if any_multi {
            // GENERAL path (a non-unique predicate gave some needle >1 row): O(n) counting-sort scatter by
            // needle_index (`cursor` walks each needle's range), then sort each multi-row sub-range by
            // row_index (row_index unique within a needle => the total order the byte-identity contract
            // wants), then flatten as raw i32.
            let mut slot = vec![0u32; total];
            let mut cursor: Vec<u32> = needle_ranges.iter().map(|&(start, _)| start).collect();
            for i in 0..nrows {
                let ni = projected.needle_indices[i] as usize;
                let d = cursor[ni] as usize;
                slot[d] = i as u32;
                cursor[ni] += 1;
            }
            for &(start, count) in &needle_ranges {
                if count > 1 {
                    let s = start as usize;
                    let e = s + count as usize;
                    slot[s..e].sort_by_key(|&i| projected.row_indices[i as usize]);
                }
            }
            let mut values = Vec::with_capacity(total * ncols);
            for &i in &slot {
                values.extend_from_slice(projected.row_values(i as usize));
            }
            values
        } else {
            // UNIQUE FAST-PATH (the dominant point read: <=1 row/needle): place each row's i32 values
            // DIRECTLY at its needle's offset in ONE pass — no `slot`/`cursor` indirection (-512KB allocs at
            // b65536), no separate flatten. Byte-identical to the general path (count==1 => the scatter would
            // land row `i` at exactly `needle_ranges[ni].0`, and no within-needle order question arises).
            let mut values = vec![0i32; total * ncols];
            for i in 0..nrows {
                let ni = projected.needle_indices[i] as usize;
                let dst = needle_ranges[ni].0 as usize * ncols;
                values[dst..dst + ncols].copy_from_slice(projected.row_values(i));
            }
            values
        };
        RelationalRetainedBatchResult {
            columns,
            access_path,
            gpu_id,
            values,
            ncols,
            needle_ranges,
        }
    }

    /// Fold already-materialized `Ready` results (general-query / empty-batch path) into the batched layout:
    /// each result is one needle, rows concatenated, schema from the first (the batcher's `Ready` case is the
    /// empty batch -> 0 needles; the general fold is defensive).
    fn fold_ready_results_batched(
        results: Vec<RelationalSelectResult>,
    ) -> RelationalRetainedBatchResult {
        let (columns, access_path, gpu_id) = match results.first() {
            Some(first) => (
                Arc::clone(&first.columns),
                Arc::clone(&first.access_path),
                match first.executed_target {
                    DeviceTarget::Gpu(id) => id,
                    _ => 0,
                },
            ),
            None => (
                Arc::new(Vec::new()),
                Arc::new(RelationalAccessPath::FullTableScan),
                0,
            ),
        };
        // Width from the first NON-EMPTY result: an empty first result has ncols 0 but a later result may
        // carry rows, and flattening at ncols 0 would drop them (audit P3). All Ready results of one query
        // shape share the width, so the first non-zero is authoritative.
        let ncols = results
            .iter()
            .map(|r| r.rows.ncols())
            .find(|&n| n > 0)
            .unwrap_or(0);
        let mut values: Vec<i32> = Vec::new();
        let mut needle_ranges = Vec::with_capacity(results.len());
        let mut acc = 0u32;
        for result in &results {
            let count = result.rows.len() as u32;
            needle_ranges.push((acc, count));
            acc += count;
            for row in result.rows.iter() {
                for value in row {
                    // The int4 batched route is always Int4; the batcher's only Ready case is the EMPTY
                    // batch (0 rows), so this defensive non-empty fold never runs in production.
                    debug_assert!(
                        matches!(value, SqlValue::Int4(_)),
                        "batched int4 result expects Int4 values, got {value:?}"
                    );
                    values.push(match value {
                        SqlValue::Int4(v) => *v,
                        _ => 0,
                    });
                }
            }
        }
        RelationalRetainedBatchResult {
            columns,
            access_path,
            gpu_id,
            values,
            ncols,
            needle_ranges,
        }
    }
}

/// Cross-shard PK index (sub-slice 1): the PURE host build of the open-addressing int4 hash table
/// `(key<<32)|(row+1)` (0 = empty), Fibonacci hash `(key*0x9E37_79B1)>>hash_shift` + linear probe with the
/// kernel's hard 256-probe cap. Extracted verbatim from `build_wave_resident_int4_index` so the per-shard
/// index uses the IDENTICAL layout + dup/overflow rules as the R1 single-buffer index. Returns
/// `(table, table_mask, hash_shift)` or `None` when: the table would exceed 2^30 entries; or the column has
/// DUPLICATE int4 keys / a key exceeds the 256-probe cap (a hash index holds one row per key but the scan
/// returns every match, so a duplicate MUST decline → caller scans). `row + 1` packs into the low 32 bits.
pub(crate) fn build_int4_pk_hash_table_host(
    keys: &[i32],
    row_count: u64,
) -> Option<(Vec<u64>, u32, u32)> {
    // Legacy HOST index (first-match probe): keep the unique-key decline on a duplicate.
    build_int4_pk_hash_table_host_visible(keys, row_count, None, 0, false)
}

/// U1 (visibility-aware rebuild): like [`build_int4_pk_hash_table_host`], but rows whose
/// `deleted_by` stamp is at or below `gc_boundary` are SKIPPED — dead at or below the oldest
/// active snapshot means invisible to EVERY current and future reader, so omitting them from
/// the index loses nothing (the index also serves point reads at old snapshots — rows dead
/// ABOVE the boundary must stay indexed, which is why this is boundary-gated and not a blanket
/// tombstone skip). This is what keeps a delete→reinsert shard indexable: without the skip the
/// rebuild sees the dead twin + the live reinsert as a duplicate and DECLINES PERMANENTLY
/// (every later locate degrades to the scan). Twins deleted ABOVE the boundary still collide
/// and decline — a transient bounded by the active-snapshot window, sized by the mixed bench.
/// The LIVE fill (0x7F7F..) is above any real boundary, so live rows are never skipped, and a
/// missing region (`None`) means every row is live — no special cases.
pub(crate) fn build_int4_pk_hash_table_host_visible(
    keys: &[i32],
    row_count: u64,
    deleted_by: Option<&[u64]>,
    gc_boundary: u64,
    // F3/U4: when true, a same-key occupant is a MVCC VERSION TWIN (an updated key's dead-old +
    // live-new both above the GC boundary) and is placed at the next probe slot rather than
    // declining — the DEVICE index is probed by the dup-tolerant `visible-locate` kernel, which
    // walks the chain and resolves the snapshot-visible version. When false (the legacy HOST index,
    // probed first-match) a dup key still declines the whole table (that probe cannot resolve
    // versions). Boundary-gated dedup below still drops dead-below-GC rows first in both modes.
    dup_tolerant: bool,
) -> Option<(Vec<u64>, u32, u32)> {
    // A 0-row shard makes `table_size = 1` -> `hash_shift = 32`, and `key >> 32` is a u32 shift-overflow
    // (panics in debug/test, silently masks in release). Decline (the scan handles 0 rows as an empty match);
    // R1's caller `build_wave_resident_int4_index` already returns None for 0 rows BEFORE delegating, so this
    // is byte-identical for R1 and also protects the per-shard locate caller (audit P2).
    if row_count == 0 {
        return None;
    }
    let table_size = row_count
        .checked_mul(2)
        .and_then(|doubled| doubled.checked_next_power_of_two())?;
    if table_size > (1_u64 << 30) {
        return None;
    }
    let table_mask = (table_size - 1) as u32;
    let hash_shift = 32 - table_size.trailing_zeros();
    let mut index = vec![0_u64; table_size as usize];
    for (row, &key) in keys.iter().enumerate() {
        if deleted_by
            .and_then(|stamps| stamps.get(row))
            .is_some_and(|&stamp| stamp <= gc_boundary)
        {
            continue; // dead below the boundary: invisible to every possible reader
        }
        let key_bits = key as u32;
        let mut slot = (key_bits.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
        let mut probes = 0_u32;
        loop {
            let occupant = index[slot as usize];
            if occupant == 0 {
                index[slot as usize] = ((key_bits as u64) << 32) | (row as u64 + 1);
                break;
            }
            if !dup_tolerant && (occupant >> 32) as u32 == key_bits {
                return None; // unique (host) index: a duplicate physical key declines the table
            }
            // F3/U4 dup_tolerant OR a different-key collision: probe onward to the next slot.
            slot = (slot + 1) & table_mask;
            probes += 1;
            if probes >= 256 {
                return None;
            }
        }
    }
    Some((index, table_mask, hash_shift))
}

/// TYPE-COVERAGE track 1 (ledger #3 incremental-maintenance gate, measured on the PK'd-table SLO):
/// EXTEND an existing host hash table with the shard's APPENDED tail keys — the in-place
/// open-shard append grows `row_count` under the SAME device ptr every commit/wave-flush, and a
/// full O(shard) rebuild per probe made the constrained-INSERT prepare ~1ms (923→5.5k TPS was the
/// scan fix alone; this is the rest). IDENTICAL probing scheme to the builder (Fibonacci hash +
/// linear probe, 256 cap, `(key<<32)|(row+1)` packing). Returns `false` on a DUPLICATE tail key or
/// probe overflow — the shard has become dup-bearing and the entry must transition to the
/// monotone DECLINED state (exactly what a full rebuild would conclude, without paying O(shard)
/// to re-discover it). The caller enforces the builder's load rule (`2*count <= table_size`)
/// BEFORE calling; within it, insertion is always possible absent dups/overflow.
fn extend_int4_pk_hash_table_host(
    index: &mut [u64],
    table_mask: u32,
    hash_shift: u32,
    tail_keys: &[i32],
    base_row: usize,
) -> bool {
    for (offset, &key) in tail_keys.iter().enumerate() {
        let row = base_row + offset;
        let key_bits = key as u32;
        let mut slot = (key_bits.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
        let mut probes = 0_u32;
        loop {
            let occupant = index[slot as usize];
            if occupant == 0 {
                index[slot as usize] = ((key_bits as u64) << 32) | (row as u64 + 1);
                break;
            }
            if (occupant >> 32) as u32 == key_bits {
                return false; // duplicate: the shard declines (monotone under appends)
            }
            slot = (slot + 1) & table_mask;
            probes += 1;
            if probes >= 256 {
                return false;
            }
        }
    }
    true
}

/// TYPE-COVERAGE track 1: set the bloom bits for appended tail keys. The bit array is sized at
/// build time (10 bits/key THEN), so post-append inserts raise the false-POSITIVE rate slightly
/// (perf-only: an FP costs one hash probe of the shard) — never a false NEGATIVE, the
/// load-bearing invariant. The periodic load-factor rebuild re-sizes both structures. A `(0,0)`
/// fallback bloom (num_bits == 0) means "maybe contains everything": extending it is a no-op and
/// stays conservative.
fn extend_int4_pk_bloom_host(words: &mut [u64], num_bits: u64, num_hashes: u32, tail_keys: &[i32]) {
    if num_bits == 0 || words.is_empty() {
        return;
    }
    for &key in tail_keys {
        let (h1, h2) = bloom_hashes(key);
        for i in 0..num_hashes {
            let bit = bloom_bit(h1, h2, i, num_bits);
            words[(bit / 64) as usize] |= 1_u64 << (bit % 64);
        }
    }
}

/// Cross-shard PK index (sub-slice 1): probe the host hash table built by `build_int4_pk_hash_table_host`
/// for `key`, returning the LOCAL row index (0-based) or `None` (absent). Mirrors the device probe kernel:
/// Fibonacci hash → linear probe up to the 256 cap, matching the high 32 bits (the key) and unpacking
/// `row = (entry & 0xFFFF_FFFF) - 1`. An empty slot (0) terminates the probe = not found. A NULL int4 is
/// materialized as `0`, so `key = 0` probes exactly as the build indexed it (agrees with the scan).
pub(crate) fn probe_int4_pk_hash_table(
    table: &[u64],
    table_mask: u32,
    hash_shift: u32,
    key: i32,
) -> Option<u32> {
    let key_bits = key as u32;
    let mut slot = (key_bits.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
    for _ in 0..256 {
        let occupant = table[slot as usize];
        if occupant == 0 {
            return None;
        }
        if (occupant >> 32) as u32 == key_bits {
            return Some(((occupant & 0xFFFF_FFFF) as u32) - 1);
        }
        slot = (slot + 1) & table_mask;
    }
    None
}

/// Cross-shard PK index (sub-slice 2): two well-distributed 64-bit hashes of an int4 key for the bloom's
/// double hashing (Kirsch-Mitzenmacher `bit_i = h1 + i*h2`). Each key is run through a splitmix64-style
/// finalizer -- a PLAIN multiplicative hash leaves poorly-distributed LOW bits, so we mix and the bit index
/// is taken from the HIGH bits (see `bloom_bit`). Key is zero-extended (NULL-as-0 hashes as key 0,
/// consistent with the hash index + the scan's `WHERE col = 0`).
fn bloom_hashes(key: i32) -> (u64, u64) {
    let mix = |mut z: u64| -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let k = key as u32 as u64;
    let h1 = mix(k.wrapping_add(0x9E37_79B9_7F4A_7C15));
    // h2 forced ODD so the k probes land on distinct bit slots.
    let h2 = mix(k.wrapping_add(0x1234_5678_9ABC_DEF0)) | 1;
    (h1, h2)
}

/// The i-th bloom bit index for a key with double-hash `(h1, h2)` over `num_bits = 2^b`: take the HIGH `b`
/// bits of the 64-bit `h1 + i*h2` (`>> (64 - b)`), where the entropy of the mixed hash lives.
fn bloom_bit(h1: u64, h2: u64, i: u32, num_bits: u64) -> u64 {
    let shift = 64 - num_bits.trailing_zeros();
    h1.wrapping_add((i as u64).wrapping_mul(h2)) >> shift
}

/// M1 (perf): the i32-SECTION byte offset of column `filter_idx` in a shard's payload, computed
/// DIRECTLY from the shard's fields (no descriptor clone). Mirrors
/// `resident_device_int4_column_offset` byte-for-byte: header (u64) + `int4_ordinal * capacity * 4`,
/// where `int4_ordinal` = the count of i32-section columns before `filter_idx` in catalog order,
/// validated against the shard's `resident_device_int4_columns` label. `None` on any mismatch
/// (non-i32 column / stale layout) -> the caller declines (per-item full validation).
fn shard_i32_filter_offset(
    shard: &RelationalResidentShard,
    table: &RelationalTable,
    filter_idx: usize,
) -> Option<u64> {
    let column = table.columns.get(filter_idx)?;
    if !matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2) {
        return None;
    }
    let int4_ordinal = table
        .columns
        .iter()
        .take(filter_idx)
        .filter(|c| matches!(c.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
        .count();
    if shard
        .resident_device_int4_columns
        .get(int4_ordinal)
        .is_none_or(|name| name != &column.name)
    {
        return None;
    }
    let capacity = u64::try_from(shard.capacity).ok()?;
    let int4_width = std::mem::size_of::<i32>() as u64;
    capacity
        .checked_mul(int4_width)
        .and_then(|col_bytes| (int4_ordinal as u64).checked_mul(col_bytes))
        .and_then(|prefix| (std::mem::size_of::<u64>() as u64).checked_add(prefix))
}

/// COMPOUND KEYS (wider types): the capacity-strided byte offset of a FIXED-WIDTH key column in the
/// shard buffer, dispatched by section — i32-section (Int4/Date/Int2) via `shard_i32_filter_offset`, or
/// the i64 section (Int8/Timestamp): `header + int4_section_bytes + int8_ordinal * capacity * 8`
/// (matching `resident_device_int8_column_offset`). Returns `None` for an unsupported/absent column.
fn shard_fixed_width_key_offset(
    shard: &RelationalResidentShard,
    table: &RelationalTable,
    col_idx: usize,
) -> Option<u64> {
    let column = table.columns.get(col_idx)?;
    match column.ty {
        SqlType::Int4 | SqlType::Date | SqlType::Int2 => {
            shard_i32_filter_offset(shard, table, col_idx)
        }
        SqlType::Int8 | SqlType::Timestamp => {
            let int8_ordinal = table
                .columns
                .iter()
                .take(col_idx)
                .filter(|c| matches!(c.ty, SqlType::Int8 | SqlType::Timestamp))
                .count();
            if shard
                .resident_device_int8_columns
                .get(int8_ordinal)
                .is_none_or(|name| name != &column.name)
            {
                return None;
            }
            let capacity = u64::try_from(shard.capacity).ok()?;
            let int4_section_bytes = capacity
                .checked_mul(std::mem::size_of::<i32>() as u64)?
                .checked_mul(shard.resident_device_int4_columns.len() as u64)?;
            let int8_prefix = capacity
                .checked_mul(std::mem::size_of::<i64>() as u64)?
                .checked_mul(int8_ordinal as u64)?;
            (std::mem::size_of::<u64>() as u64)
                .checked_add(int4_section_bytes)
                .and_then(|after_i32| after_i32.checked_add(int8_prefix))
        }
        SqlType::Numeric { .. } | SqlType::Uuid => {
            // b128 section: header + int4_section + int8_section + numeric_ordinal * capacity * 16
            // (matching `resident_device_numeric_column_offset`).
            let numeric_ordinal = table
                .columns
                .iter()
                .take(col_idx)
                .filter(|c| matches!(c.ty, SqlType::Numeric { .. } | SqlType::Uuid))
                .count();
            if shard
                .resident_device_numeric_columns
                .get(numeric_ordinal)
                .is_none_or(|name| name != &column.name)
            {
                return None;
            }
            let capacity = u64::try_from(shard.capacity).ok()?;
            let int4_section_bytes = capacity
                .checked_mul(std::mem::size_of::<i32>() as u64)?
                .checked_mul(shard.resident_device_int4_columns.len() as u64)?;
            let int8_section_bytes = capacity
                .checked_mul(std::mem::size_of::<i64>() as u64)?
                .checked_mul(shard.resident_device_int8_columns.len() as u64)?;
            let numeric_prefix = capacity
                .checked_mul(std::mem::size_of::<i128>() as u64)?
                .checked_mul(numeric_ordinal as u64)?;
            (std::mem::size_of::<u64>() as u64)
                .checked_add(int4_section_bytes)
                .and_then(|after_i32| after_i32.checked_add(int8_section_bytes))
                .and_then(|after_i64| after_i64.checked_add(numeric_prefix))
        }
        SqlType::Text => {
            // TEXT key column: the fold reads the column's OFFSETS array (self-describing absolute byte
            // offset in the shard buffer). The BLOB is addressed separately via `shard_key_column_blob_offset`.
            shard
                .resident_device_text_columns
                .iter()
                .find(|layout| layout.name == column.name)
                .map(|layout| layout.offsets_byte_offset)
        }
        _ => None,
    }
}

/// COMPOUND KEYS (text): the BLOB byte offset of a TEXT key column in the shard buffer (the device fold
/// kernel's `blob_offsets[k]`, consulted only for the text sentinel `widths[k] == 0`). Returns `0` for a
/// fixed-width column (the kernel ignores it there), so callers build a `blob_offsets` array parallel to
/// the fixed-width `offsets` array uniformly across key column types.
fn shard_key_column_blob_offset(
    shard: &RelationalResidentShard,
    table: &RelationalTable,
    col_idx: usize,
) -> Option<u64> {
    let column = table.columns.get(col_idx)?;
    match column.ty {
        SqlType::Text => shard
            .resident_device_text_columns
            .iter()
            .find(|layout| layout.name == column.name)
            .map(|layout| layout.bytes_byte_offset),
        _ => Some(0),
    }
}

/// Cross-shard PK index (sub-slice 2): build a per-shard membership BLOOM over the int4 key column. `m` =
/// 10 bits/key rounded up to a power of two (mask-friendly, >= 64), `k` = 7 double-hashed probes. A false
/// POSITIVE (all k bits set by OTHER keys) only costs one extra hash-index probe of a shard; there is NEVER
/// a false NEGATIVE -- every inserted key sets ALL k of its bits, so `bloom_maybe_contains` returns true for
/// it. That no-false-negative property is the LOAD-BEARING correctness invariant: the bloom may only SKIP a
/// shard it PROVES cannot hold the key. Sizing (10 bits/key, k=7 → ~1% FP) is a tunable perf/memory knob,
/// NOT a correctness parameter. `None` on 0 rows / oversize (>2^34 bits). Returns `(words, num_bits, k)`.
pub(crate) fn build_int4_pk_bloom_host(keys: &[i32]) -> Option<(Vec<u64>, u64, u32)> {
    let n = keys.len() as u64;
    if n == 0 {
        return None;
    }
    const BITS_PER_KEY: u64 = 10;
    const NUM_HASHES: u32 = 7;
    let num_bits = n
        .saturating_mul(BITS_PER_KEY)
        .checked_next_power_of_two()?
        .max(64);
    if num_bits > (1_u64 << 34) {
        return None;
    }
    let mut words = vec![0_u64; (num_bits / 64) as usize];
    for &key in keys {
        let (h1, h2) = bloom_hashes(key);
        for i in 0..NUM_HASHES {
            let bit = bloom_bit(h1, h2, i, num_bits);
            words[(bit / 64) as usize] |= 1_u64 << (bit % 64);
        }
    }
    Some((words, num_bits, NUM_HASHES))
}

/// Cross-shard PK index (sub-slice 2): `true` = key MAYBE present (probe the shard's hash), `false` =
/// DEFINITELY absent (skip the shard). No false negatives by construction (see `build_int4_pk_bloom_host`).
pub(crate) fn bloom_maybe_contains(
    words: &[u64],
    num_bits: u64,
    num_hashes: u32,
    key: i32,
) -> bool {
    if num_bits == 0 {
        return true; // no bloom -> can't prune -> conservatively "maybe" (never skip)
    }
    let (h1, h2) = bloom_hashes(key);
    for i in 0..num_hashes {
        let bit = bloom_bit(h1, h2, i, num_bits);
        if words[(bit / 64) as usize] & (1_u64 << (bit % 64)) == 0 {
            return false;
        }
    }
    true
}

/// U1: the batched visible-locate verdicts for one wave's delete needles. Parallel per-needle
/// vectors (`counts[i]` visible matches at needle i's snapshot; `shard_ids[i]`/`slots[i]` = the
/// first visible target, meaningful iff `counts[i] >= 1`) + `probed`, the (shard_id, MAIN device
/// region) identity handles of every probed shard captured from the SAME `shards.load()`
/// snapshot — the apply-time tombstone MUST recheck cell liveness against these exact regions
/// (a VACUUM/re-admit between locate and apply re-clusters slots; stamping a stale slot would
/// tombstone the wrong row).
#[derive(Default)]
pub(crate) struct WaveVisibleLocate {
    pub(crate) counts: Vec<u32>,
    pub(crate) shard_ids: Vec<u32>,
    pub(crate) slots: Vec<u32>,
    pub(crate) probed: Vec<(u32, Arc<CudaResidentDeviceMemory>)>,
}

/// Sub-slice 3b: a GENERATION-CONSISTENT cross-shard PK-index hit. Carries the resolved `(shard_id, slot)`
/// TOGETHER WITH the exact device handles the slot was resolved against -- all captured inside ONE
/// `shards.load()` snapshot in `locate_resident_pk_via_shard_index_detailed`. The point-lookup route
/// materializes the row from these captured handles (NOT a fresh `shard_device_memory.get`), so the slot,
/// the capacity-stride (via `descriptor.capacity`), the int4 buffer, and the `deleted_by` region are all the
/// SAME generation. The `Arc<CudaResidentDeviceMemory>` PINS the buffer live for the read, so a concurrent
/// re-admit that compacts/reorders the shard cannot free it or make the slot point at a different row.
pub(crate) struct ShardPkHit {
    pub(crate) shard_id: u32,
    pub(crate) slot: u32,
    pub(crate) descriptor: RelationalResidencySnapshot,
    pub(crate) device_memory: Arc<CudaResidentDeviceMemory>,
    pub(crate) deleted_by: Option<Arc<CudaResidentDeviceMemory>>,
    /// SV6: the shard's `created_by` region (same snapshot/pin discipline as `deleted_by`) — the route's
    /// `created_by[slot] <= read_txn_id` lower-bound gate hides an UPDATE-appended version from a reader
    /// bound to an older snapshot. `None` = un-stamped shard, born-visible.
    pub(crate) created_by: Option<Arc<CudaResidentDeviceMemory>>,
    /// RETIREMENT A2: the shard's ROW-IDENTITY region (A1), captured in the SAME snapshot — the
    /// device DML resolve reads `row_id[slot]` to derive the host key. `None` = identity-unknown
    /// lineage (benchmark/synthetic) -> the resolve declines to the host path.
    pub(crate) row_id: Option<Arc<CudaResidentDeviceMemory>>,
}

/// Step 1 (lpb-for-shards): a shard's BATCHED hits — the generation-consistent captured handles (descriptor,
/// PINNED int4 buffer, deleted_by region, all from ONE `shards.load()` snapshot) + the `(needle_index, slot)`
/// list of the batch's needles that Hit in this shard. Same pin/consistency discipline as `ShardPkHit`.
struct BatchShardGroup {
    descriptor: RelationalResidencySnapshot,
    device_memory: Arc<CudaResidentDeviceMemory>,
    deleted_by: Option<Arc<CudaResidentDeviceMemory>>,
    /// SV6: the shard's `created_by` region (same snapshot/pin discipline) for the batched lower-bound gate.
    created_by: Option<Arc<CudaResidentDeviceMemory>>,
    hits: Vec<(u32, u32)>, // (needle_index, local slot)
}

/// Step 1 (lpb-for-shards): the batched point-lookup projection. `values` is row-major int4, `ncols` wide, in
/// NEEDLE ORDER; `needle_ranges[i] = (start_row, row_count)` slices needle i's rows (unique-PK -> count 0 or
/// 1). The schema (columns / access_path) is shape metadata the caller wraps around this raw projection.
/// Consumed by the production batch-result wiring (`submit_sharded_point_lookups_batched`), the
/// differential tests, and the benches.
pub(crate) struct BatchedShardProjection {
    pub(crate) ncols: usize,
    pub(crate) values: Vec<i32>,
    pub(crate) needle_ranges: Vec<(u32, u32)>,
}

/// Sub-slice 3: the result of probing a shard's cached PK index for a key.
enum ShardPkProbe {
    /// Local row index of the (unique) matching row in the shard.
    Hit(u32),
    /// Absent in this shard (bloom-pruned, or the hash found no match).
    Miss,
    /// The shard's key column has duplicates / oversize -> the caller must fall back to the scan.
    Declined,
}

/// Probe a cached shard PK index entry for `key`: bloom-prune, then hash-probe. `index = None` = the shard
/// declined at build (dup/oversize) -> `Declined`. No false negative (see the bloom/hash builds), so a
/// present key is never wrongly Missed.
fn probe_cached_shard_pk(entry: &CachedShardPkIndex, key: i32) -> ShardPkProbe {
    match &entry.index {
        None => ShardPkProbe::Declined,
        Some(data) => {
            if !bloom_maybe_contains(
                &data.bloom_words,
                data.bloom_num_bits,
                data.bloom_num_hashes,
                key,
            ) {
                return ShardPkProbe::Miss;
            }
            match probe_int4_pk_hash_table(&data.hash_table, data.table_mask, data.hash_shift, key)
            {
                Some(row) => ShardPkProbe::Hit(row),
                None => ShardPkProbe::Miss,
            }
        }
    }
}

#[cfg(test)]
mod cross_shard_pk_index_tests {
    use super::{
        bloom_maybe_contains, build_int4_pk_bloom_host, build_int4_pk_hash_table_host,
        probe_int4_pk_hash_table,
    };

    /// The pure per-shard PK hash index (sub-slice 1) round-trips: every built key probes back to its own
    /// row; an absent key returns None; NULL-as-0 is indexed + found; a DUPLICATE key declines the build
    /// (so the caller scans). Non-vacuous: a wrong row/absent would mismatch the enumerate() oracle.
    #[test]
    fn pk_hash_table_build_probe_roundtrips_and_declines_dups() {
        let keys: Vec<i32> = vec![0, 5, 130, -7, 42, 1_000_000];
        let (table, mask, shift) =
            build_int4_pk_hash_table_host(&keys, keys.len() as u64).expect("unique keys build");
        for (row, &key) in keys.iter().enumerate() {
            assert_eq!(
                probe_int4_pk_hash_table(&table, mask, shift, key),
                Some(row as u32),
                "key {key} must probe back to its own row {row}"
            );
        }
        // Absent keys (not in the set) return None.
        for absent in [7, 131, -1, 999] {
            assert_eq!(probe_int4_pk_hash_table(&table, mask, shift, absent), None);
        }
        // A duplicate key declines the build (a hash index holds one row/key; the scan returns all).
        assert!(
            build_int4_pk_hash_table_host(&[5, 5], 2).is_none(),
            "duplicate keys must decline (fall back to scan)"
        );
        // NULL-as-0: key 0 is indexed at its row and probes found (agrees with WHERE col = 0).
        assert_eq!(probe_int4_pk_hash_table(&table, mask, shift, 0), Some(0));
        // Audit P2: a 0-row shard declines WITHOUT a shift-by-32 overflow (would panic in debug otherwise).
        assert!(
            build_int4_pk_hash_table_host(&[], 0).is_none(),
            "0-row build declines (no shift-by-32 panic)"
        );
    }

    /// U1 (sabotage-sensitive twin of the GPU reinsert test, whose row outcomes survive a
    /// broken skip via the decline->scan net): a dead-below-boundary twin must be SKIPPED so
    /// the rebuild succeeds; the same twin above the boundary (or with no skip) collides and
    /// declines. Deleting the skip arm in `build_int4_pk_hash_table_host_visible` FAILS this
    /// (sabotage-verified 2026-07-07).
    #[test]
    fn visibility_aware_rebuild_skips_dead_twin_below_boundary() {
        let keys = [10_i32, 20, 10, 30]; // row 0 = dead twin of row 2
        let live = 0x7F7F_7F7F_7F7F_7F7Fu64;
        let stamps = [5_u64, live, live, live];
        let (index, mask, shift) =
            crate::engine_retained_read::build_int4_pk_hash_table_host_visible(
                &keys,
                8,
                Some(&stamps),
                7,
                false, // unique (host) mode: this test asserts the decline-on-dup behavior
            )
            .expect("dead twin below boundary is skipped");
        let mut probe = ((10_u32.wrapping_mul(0x9E37_79B1)) >> shift) & mask;
        let found = loop {
            let entry = index[probe as usize];
            assert_ne!(entry, 0, "key 10 must be indexed");
            if (entry >> 32) as u32 == 10 {
                break (entry & 0xFFFF_FFFF) as u32;
            }
            probe = (probe + 1) & mask;
        };
        assert_eq!(found, 3, "the LIVE twin (row 2, packed row+1=3) is indexed");
        assert!(
            crate::engine_retained_read::build_int4_pk_hash_table_host_visible(
                &keys,
                8,
                Some(&stamps),
                4,
                false, // unique mode: above-boundary twin collides -> declines
            )
            .is_none(),
            "a twin dead ABOVE the boundary still collides (readers may need it)"
        );
        assert!(
            crate::engine_retained_read::build_int4_pk_hash_table_host_visible(
                &keys, 8, None, 7, false
            )
            .is_none()
        );
    }

    /// F3/U4 (dup-tolerant DEVICE index): an above-boundary version twin is PLACED at the next
    /// probe slot (not declined), so the dup-tolerant visible-locate can resolve the snapshot-
    /// visible version. Sabotage: forcing `dup_tolerant = false` makes this build decline (None).
    #[test]
    fn dup_tolerant_build_places_above_boundary_twin() {
        let keys = [10_i32, 20, 10, 30]; // rows 0 and 2 share key 10 (an updated key's twin)
        let live = 0x7F7F_7F7F_7F7F_7F7Fu64;
        // Old (row 0) tombstoned at seq 6, ABOVE the gc_boundary (4) -> still reader-visible ->
        // both versions must be indexed.
        let stamps = [6_u64, live, live, live];
        let (index, mask, shift) =
            crate::engine_retained_read::build_int4_pk_hash_table_host_visible(
                &keys,
                8,
                Some(&stamps),
                4,
                true, // dup-tolerant DEVICE mode
            )
            .expect("dup-tolerant build must place the twin, not decline");
        // BOTH physical rows for key 10 (row 0 packed 1, row 2 packed 3) are present in the chain.
        let mut probe = ((10_u32.wrapping_mul(0x9E37_79B1)) >> shift) & mask;
        let mut found: Vec<u32> = Vec::new();
        loop {
            let entry = index[probe as usize];
            if entry == 0 {
                break;
            }
            if (entry >> 32) as u32 == 10 {
                found.push((entry & 0xFFFF_FFFF) as u32);
            }
            probe = (probe + 1) & mask;
        }
        found.sort_unstable();
        assert_eq!(
            found,
            vec![1, 3],
            "both twins (packed row+1 = 1 and 3) are indexed"
        );
    }

    /// The per-shard bloom (sub-slice 2) has NO FALSE NEGATIVES (every built key -> maybe_contains true --
    /// the load-bearing membership-prune invariant, so a present key's shard is NEVER skipped) and a sane
    /// false-positive rate (most absent keys -> false = skippable). Includes 0, negatives, and a large set.
    #[test]
    fn pk_bloom_no_false_negatives_and_sane_fp() {
        let keys: Vec<i32> = (0..1000_i32).map(|i| i * 3 - 500).collect(); // -500..2497, incl 0-adjacent + neg
        let (words, num_bits, k) = build_int4_pk_bloom_host(&keys).expect("build");
        // NO FALSE NEGATIVES: every inserted key MUST test present. A single miss = a dropped hit = WRONG.
        for &key in &keys {
            assert!(
                bloom_maybe_contains(&words, num_bits, k, key),
                "bloom false negative for a BUILT key {key} -- would drop a real hit"
            );
        }
        // False-positive sanity: 10k absent candidates far from the built range -> mostly "definitely absent".
        let (mut fp, mut total) = (0_usize, 0_usize);
        for cand in 1_000_000..1_010_000_i32 {
            total += 1;
            if bloom_maybe_contains(&words, num_bits, k, cand) {
                fp += 1;
            }
        }
        let fp_rate = fp as f64 / total as f64;
        assert!(
            fp_rate < 0.05,
            "bloom FP rate {fp_rate:.4} must be well under 5% at 10 bits/key, k=7"
        );
        // Empty -> None (no bloom to prune with).
        assert!(build_int4_pk_bloom_host(&[]).is_none());
        // A single-key bloom still contains its key (>= 64-bit min size, no sub-word issue).
        let (w1, b1, k1) = build_int4_pk_bloom_host(&[42]).unwrap();
        assert!(bloom_maybe_contains(&w1, b1, k1, 42));
    }
}
