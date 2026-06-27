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
            members.push((bound.selected_columns, access_path, needle));
        }

        let table = batch_table.expect("non-empty batch has table");
        let filter_idx = batch_filter_idx.expect("non-empty batch has filter");
        let selected_indexes = batch_selected_indexes.expect("non-empty batch has projections");
        let needles = members
            .iter()
            .map(|(_columns, _access_path, needle)| *needle)
            .collect::<Vec<_>>();
        let (snapshot_gpu_id, before_metrics, batch_started, cuda_submission) = match self
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
                    members,
                    before_metrics,
                    batch_started,
                    submission: cuda_submission,
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
        let (snapshot_gpu_id, before_metrics, batch_started, cuda_submission) = self
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
        // Stamp every per-needle result with the shared (needle-invariant) columns + access path.
        let members = needles
            .iter()
            .map(|needle| {
                (
                    template.result_columns.clone(),
                    template.access_path.clone(),
                    *needle,
                )
            })
            .collect::<Vec<_>>();
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
                    members,
                    before_metrics,
                    batch_started,
                    submission: cuda_submission,
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
        Option<(u16, RuntimeMetricsSnapshot, Instant, CudaI32EqualAnyProjectSubmission)>,
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
        // ADR-009 R1: with the wave engine on, prefer a GPU hash-index probe (O(1) per needle) over the
        // full-scan kernel for this resident int4 unique-key column. `wave_resident_int4_index` returns
        // `None` (→ scan) when the flag is off, the column is non-unique / un-buildable, or a generation
        // race is detected — so the index route NEVER changes results, it only changes how they're found.
        // Both arms produce the SAME `CudaI32EqualAnyProjectSubmission`, so completion is byte-identical.
        // CONTRACT: `needles` are distinct (the batcher's `dedup_needles` collapses identical lookups). The
        // thread-per-needle index emits one match per found needle vs the scan's one per matched row; for a
        // UNIQUE key + distinct needles these coincide (a bijection, match_count ≤ row_count). Were a caller
        // to pass duplicates, `complete`'s `match_count > row_count` guard fail-safes to an error, never wrong rows.
        let cuda_submission = match self
            .wave_engine_enabled()
            .then(|| self.wave_resident_int4_index(&table.name, &snapshot, filter_idx))
            .flatten()
        {
            Some((index, table_mask, hash_shift)) => device_memory
                .submit_match_project_i32_index_probe_from_payload(
                    &index,
                    table_mask,
                    hash_shift,
                    needles,
                    &projection_offsets,
                    row_count,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?,
            None => device_memory
                .submit_match_project_i32_equal_any_from_payload(
                    filter_offset,
                    needles,
                    &projection_offsets,
                    row_count,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?,
        };
        Ok(Some((
            snapshot_gpu_id,
            before_metrics,
            batch_started,
            cuda_submission,
        )))
    }

    /// ADR-009 R1: fetch (building + caching on demand) the GPU hash index over resident int4 key column
    /// `filter_idx` for `table_name` at `snapshot.generation`. `Some((index, table_mask, hash_shift))`
    /// drives the index-probe route; `None` means "use the scan" — the column is non-unique, not an int4
    /// key, un-buildable, or a re-admission moved the generation out from under us. The cache is keyed by
    /// table and validated by `(generation, column_idx)`, so a stale entry is detected by mismatch and
    /// rebuilt — the index can never serve a wrong generation, and never needs lock-step invalidation.
    fn wave_resident_int4_index(
        &self,
        table_name: &str,
        snapshot: &RelationalResidencySnapshot,
        filter_idx: usize,
    ) -> Option<(Arc<CudaResidentDeviceMemory>, u32, u32)> {
        let generation = snapshot.generation;
        {
            let cache = self
                .read_state
                .residency
                .wave_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(existing) = cache.get(table_name) {
                if existing.generation == generation && existing.column_idx == filter_idx {
                    return existing
                        .index_memory
                        .as_ref()
                        .map(|memory| (Arc::clone(memory), existing.table_mask, existing.hash_shift));
                }
            }
        }
        // Miss / stale / different column: build OUTSIDE the lock (a host hash pass + one HtoD upload),
        // then publish. A concurrent builder for the same (table, generation) merely rebuilds + overwrites
        // — rare (once per generation) and harmless: each index is self-contained and its in-flight kernels
        // pin their own `Arc`, so a replaced entry's buffer is freed only once no submission still holds it.
        let built = self.build_wave_resident_int4_index(snapshot, filter_idx);
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
                generation,
                column_idx: filter_idx,
                index_memory,
                table_mask,
                hash_shift,
            },
        );
        built
    }

    /// ADR-009 R1: host-build the open-addressing GPU hash index (`(key<<32)|(row+1)`, 0 = empty;
    /// Fibonacci `(key*0x9E3779B1)>>hash_shift` + linear probe) over int4 column `filter_idx`, then upload
    /// it once (HtoD) into a resident device buffer. `None` (→ caller scans) when: a re-admission moved the
    /// generation; host rows don't match the device row count; the table is empty / too large to pack a
    /// row index into 32 bits; a non-int4 / missing key value appears; or — critically for correctness —
    /// the column has DUPLICATE keys (a hash index holds one row per key but the scan returns every match).
    /// NULL keys are skipped (SQL 3VL: `col = needle` is never true for NULL, exactly as the scan behaves),
    /// which is what keeps results byte-identical to the scan on NULL data.
    fn build_wave_resident_int4_index(
        &self,
        snapshot: &RelationalResidencySnapshot,
        filter_idx: usize,
    ) -> Option<(Arc<CudaResidentDeviceMemory>, u32, u32)> {
        let entry = self.relational_residency_entry(&snapshot.table)?;
        if entry.descriptor.generation != snapshot.generation {
            return None;
        }
        let host_rows = entry.host_rows.as_ref();
        let row_count = snapshot.row_count;
        if host_rows.len() != row_count {
            return None;
        }
        // `row + 1` is packed into the low 32 bits, so the row index must fit in u32; also bound the table
        // so a pathological row count can't allocate an absurd host vector (fall back to the scan instead).
        if row_count == 0 || row_count >= (u32::MAX as usize) {
            return None;
        }
        let table_size = (row_count as u64)
            .checked_mul(2)
            .and_then(|doubled| doubled.checked_next_power_of_two())?;
        if table_size > (1_u64 << 30) {
            return None;
        }
        let table_mask = (table_size - 1) as u32;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (row, values) in host_rows.iter().enumerate() {
            let key = match values.get(filter_idx) {
                Some(SqlValue::Int4(value)) => *value,
                Some(SqlValue::Null) => continue,
                _ => return None,
            };
            let key_bits = key as u32;
            let mut slot = (key_bits.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
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
            }
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|entry| entry.to_le_bytes()).collect();
        let runtime = self.cuda_driver_probe_runtime();
        let memory = runtime
            .retain_device_memory_copy(snapshot.gpu_id, &index_bytes)
            .ok()?;
        Some((Arc::new(memory), table_mask, hash_shift))
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
                |((columns, access_path, _needle), rows)| RelationalSelectResult {
                    columns,
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
