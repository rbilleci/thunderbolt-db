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
        let mut shared_schema: Option<(Arc<Vec<RelationalColumn>>, Arc<RelationalAccessPath>)> = None;
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
            .submit_resident_int4_equal_any_payload(&table, &selected_indexes, filter_idx, &needles)?
        {
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
    ) -> Result<
        Option<(
            u16,
            RuntimeMetricsSnapshot,
            Instant,
            DeferredProbe,
        )>,
        ExecuteError,
    > {
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
                "resident snapshot row count exceeds retained device-memory proof range".to_string(),
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
        Ok(Some((snapshot_gpu_id, before_metrics, batch_started, payload)))
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
                    return existing
                        .index_memory
                        .as_ref()
                        .map(|memory| (Arc::clone(memory), existing.table_mask, existing.hash_shift));
                }
            }
        }
        // Miss / different column / different buffer: build OUTSIDE the lock (a DtoH of the key column + a
        // host hash pass + one HtoD upload), then publish. A concurrent builder for the same buffer merely
        // rebuilds + overwrites — rare (once per residency generation) and harmless: each index is
        // self-contained and its in-flight kernels pin their own `Arc`, so a replaced entry's index buffer
        // is freed only once no submission still holds it.
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
        let memory = runtime
            .retain_device_memory_copy(gpu_id, &index_bytes)
            .ok()?;
        Some((Arc::new(memory), table_mask, hash_shift))
    }

    /// CROSS-SHARD PK INDEX (sub-slice 1): resolve `filter_idx = key` to the resident `(shard_id, LOCAL row)`
    /// positions via a PER-SHARD host-built PK hash index. Thin projection of
    /// [`Self::locate_resident_pk_via_shard_index_detailed`] to just `(shard_id, slot)` -- the shape the
    /// scan-locate differential + the (future) DELETE/UPDATE resolution compare against. See the detailed
    /// method for the semantics (returns the IDENTICAL physical `(shard, slot)` a scan finds; `None` to fall
    /// back on any decline / invalid shard / missing offset). Used by the scan-locate differential tests +
    /// the (future) DELETE/UPDATE resolution; the 3b read route calls the `_detailed` variant directly.
    #[allow(dead_code)]
    pub(crate) fn locate_resident_pk_via_shard_index(
        &self,
        table: &RelationalTable,
        filter_idx: usize,
        key: i32,
    ) -> Option<Vec<(u32, u32)>> {
        self.locate_resident_pk_via_shard_index_detailed(table, filter_idx, key)
            .map(|hits| hits.into_iter().map(|h| (h.shard_id, h.slot)).collect())
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
        filter_idx: usize,
        key: i32,
    ) -> Option<Vec<ShardPkHit>> {
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
            // scan (which continues to other shards) and to avoid the 0-row hash-build decline that would
            // otherwise drop hits from OTHER shards (audit P2).
            if shard.row_count == 0 {
                continue;
            }
            // The filter column's BYTE offset within this shard's own (capacity-strided) buffer -- the SAME
            // offset the scan reads, so the row indices line up 1:1 with the scan + the deleted_by gather.
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            let filter_offset =
                resident_device_int4_column_offset(&descriptor, table, filter_idx).ok()?;
            let device_memory = self
                .read_state
                .residency
                .shard_device_memory
                .get(&(table.name.clone(), shard.shard_id))?;
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
                    // Capture the deleted_by region (if versioned) from the SAME snapshot -> the visibility
                    // gate reads a `deleted_by[slot]` aligned to the SAME generation as the slot + buffer.
                    let deleted_by = self
                        .read_state
                        .residency
                        .shard_deleted_by_memory
                        .get(&(table.name.clone(), shard.shard_id));
                    out.push(ShardPkHit {
                        shard_id: shard.shard_id,
                        slot: row,
                        descriptor,
                        device_memory,
                        deleted_by,
                    });
                }
                ShardPkProbe::Miss => {}
                // A duplicate key in ANY shard declines the whole locate (the scan returns every match; a
                // hash holds one row/key) -> the caller scans.
                ShardPkProbe::Declined => return None,
            }
        }
        Some(out)
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
        {
            let cache = self
                .read_state
                .residency
                .shard_pk_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(entry) = cache.get(&cache_key) {
                // Validate (ptr, row_count): a re-admit/rollover changes the ptr; an in-place open-shard
                // APPEND grows row_count with the SAME ptr -> either mismatch rebuilds against the live shard.
                if entry.resident_device_ptr == device_ptr && entry.row_count == row_count {
                    return probe_cached_shard_pk(entry, key);
                }
            }
        }
        // Miss / stale ptr: build OUTSIDE the lock (DtoH the key column + host hash + bloom), then publish.
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
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(cache_key, entry);
        result
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
            DeferredProbe::Atomic(submission) => {
                submission
                    .complete_detached()
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?
            }
            DeferredProbe::Dense(submission) => {
                let (columns, elapsed) = submission
                    .complete_detached_columnar()
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
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
            by_needle[projected.needle_index].push((projected.row_index, projected.values.as_slice()));
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
            debug_assert_eq!(projected.status.len(), n, "dense status must be one per needle");
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
            None => (Arc::new(Vec::new()), Arc::new(RelationalAccessPath::FullTableScan), 0),
        };
        // Width from the first NON-EMPTY result: an empty first result has ncols 0 but a later result may
        // carry rows, and flattening at ncols 0 would drop them (audit P3). All Ready results of one query
        // shape share the width, so the first non-zero is authoritative.
        let ncols = results.iter().map(|r| r.rows.ncols()).find(|&n| n > 0).unwrap_or(0);
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
                return None;
            }
            slot = (slot + 1) & table_mask;
            probes += 1;
            if probes >= 256 {
                return None;
            }
        }
    }
    Some((index, table_mask, hash_shift))
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
pub(crate) fn bloom_maybe_contains(words: &[u64], num_bits: u64, num_hashes: u32, key: i32) -> bool {
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
            if !bloom_maybe_contains(&data.bloom_words, data.bloom_num_bits, data.bloom_num_hashes, key) {
                return ShardPkProbe::Miss;
            }
            match probe_int4_pk_hash_table(&data.hash_table, data.table_mask, data.hash_shift, key) {
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
        assert!(fp_rate < 0.05, "bloom FP rate {fp_rate:.4} must be well under 5% at 10 bits/key, k=7");
        // Empty -> None (no bloom to prune with).
        assert!(build_int4_pk_bloom_host(&[]).is_none());
        // A single-key bloom still contains its key (>= 64-bit min size, no sub-word issue).
        let (w1, b1, k1) = build_int4_pk_bloom_host(&[42]).unwrap();
        assert!(bloom_maybe_contains(&w1, b1, k1, 42));
    }
}
