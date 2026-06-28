//! Retained-read async job lifecycle (P0 §9.6 decomposition, behavior-preserving):
//! a focused `impl Engine` block for preparing, submitting, and completing
//! retained relational read jobs against resident device memory — including the
//! int4-projection retained submissions (try_submit / complete[_detached]) that
//! let a caller hold a device read view across calls.

use super::*;

// ADR-009 R2.2b persistent wave-read-engine launch config (single-flight first; tunable for the R2.2b-3
// A/B). The engine is built lazily per (filter_col, projection set) and owned for the residency generation.
//
/// Ring capacity bounds ONE wave; a needle batch larger than this falls back to the lpb index probe
/// (blocker#1: total un-harvested needles <= ring_capacity, trivially satisfied single-flight). 65536
/// covers the measured batch range; `WaveReadEngine::new` rounds up to a power of two.
const WAVE_ENGINE_RING_CAPACITY: usize = 65536;
/// Persistent grid; `WaveReadEngine::new` CLAMPS this to device occupancy (correctness gate C1), so an
/// over-large value can never silently hang (un-resident blocks would never drain the ring). ~8k threads
/// is where the device-result rewrite peaked (~31.8M lookups/s).
const WAVE_ENGINE_THREADS: u32 = 8192;
/// Near-infinite fixed backstop: the engine is long-lived, so crash-safety is the watchdog's job, not a
/// fixed timeout that would kill a healthy idle engine mid-life.
const WAVE_ENGINE_BACKSTOP_NS: u64 = u64::MAX;
/// Crash-safe watchdog window: if the host dies/hangs (no heartbeat), thread 0 self-terminates the kernel
/// ~2s after the heartbeat goes stale — vs zombie-ing until the backstop on the `--gpu-reset`-denied box.
const WAVE_ENGINE_WATCHDOG_NS: u64 = 2_000_000_000;

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
            // Identical-shape across jobs (asserted above) -> capture the shared schema once.
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
            RelationalRetainedInt4ProjectionPayload,
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
        // ADR-009 R1/R2.2b: route this resident int4 unique-key batch among three byte-identical producers
        // (each returns the SAME matched rows; only HOW/WHEN they're found differs):
        //   (1) PERSISTENT wave kernel  — `wave_persistent_engine_enabled` ON (nested under
        //       `wave_engine_enabled`): no per-batch launch; rows harvested SYNCHRONOUSLY -> Materialized.
        //   (2) lpb GPU hash-index probe — `wave_engine_enabled` ON, column unique/buildable: O(1)/needle.
        //   (3) full-scan kernel        — flag off, or column non-unique / un-buildable: O(rows).
        // (2)+(3) are DEFERRED submissions drained at completion. The wave (1) is built over the SAME R1
        // index as (2) (so NULL-as-0 + gather semantics match), and falls through to (2)/(3) on ANY wave
        // error / harvest timeout / oversize batch — never a wrong result, only a slower path.
        // CONTRACT: the index + wave routes require `needles` to be DISTINCT — the facade batcher's
        // `dedup_needles` guarantees this. The thread-per-needle gather emits one match per found needle vs
        // the scan's one per matched row; for a unique key + distinct needles these coincide (a bijection).
        // Duplicate needles would over-count relative to the scan, so the distinct-needle invariant is
        // debug-asserted INSIDE the index + wave arms only — the scan arm is reachable with duplicate needles
        // from the jobs-batch caller (`submit_relational_retained_int4_projection_batch`) and must NOT be
        // guarded.
        let payload = 'route: {
            // (1) Persistent wave route. blocker#1: a batch larger than the ring falls back to lpb (the
            // wave can't hold it). blocker#2: `WaveReadEngine::submit`'s internal `DRAIN_TIMEOUT` bounds the
            // single-flight harvest. Single-flight: lock the per-(col,proj) engine for this one wave.
            if self.wave_engine_enabled()
                && self.wave_persistent_engine_enabled()
                && needles.len() <= WAVE_ENGINE_RING_CAPACITY
            {
                if let Some(engine) = self.wave_read_engine_for(
                    &table.name,
                    &device_memory,
                    filter_offset,
                    filter_idx,
                    &projection_offsets,
                    row_count,
                ) {
                    debug_assert!(
                        {
                            let mut seen = std::collections::HashSet::with_capacity(needles.len());
                            needles.iter().all(|needle| seen.insert(*needle))
                        },
                        "wave route requires distinct needles (batcher dedup_needles contract)"
                    );
                    let mut guard = engine.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    if let Ok(columns) = guard.submit_columnar(needles) {
                        self.read_state
                            .residency
                            .wave_route_hits
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        break 'route RelationalRetainedInt4ProjectionPayload::Materialized(columns);
                    }
                    // else: wave error / harvest timeout -> fall through to the lpb route below (the engine
                    // is left cached; a transient timeout does not invalidate it).
                }
            }
            // (2)/(3) launch-per-batch: GPU index probe when buildable, else the full scan -> Deferred.
            let cuda_submission = match self
                .wave_engine_enabled()
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
                        "wave index route requires distinct needles (batcher dedup_needles contract)"
                    );
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
                        })?
                }
                None => device_memory
                    .submit_match_project_i32_equal_any_from_payload(
                        filter_offset,
                        needles,
                        &projection_offsets,
                        row_count,
                    )
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?,
            };
            RelationalRetainedInt4ProjectionPayload::Deferred(cuda_submission)
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

    /// ADR-009 R2.2b: fetch (building + caching on demand) the PERSISTENT wave read engine for resident
    /// int4 key column `filter_idx` projecting `projection_offsets` over resident buffer `device_memory`.
    /// `Some(engine)` drives the persistent-kernel route (`WaveReadEngine::submit`); `None` means "use the
    /// lpb index probe" — the column is non-unique / un-buildable (no R1 index) OR the kernel launch failed.
    ///
    /// Mirrors [`Engine::wave_resident_int4_index`]'s ptr-keyed cache discipline (the engine is built over +
    /// gathers through the SAME device buffer, whose address the engine's own `Arc` pins), EXTENDED with the
    /// projection set in the identity because the wave kernel BAKES the projection offsets at launch. The
    /// engine is built over the SAME R1 index (reusing `wave_resident_int4_index`), so its NULL-as-0 key +
    /// gather semantics are byte-identical to the lpb index probe by construction.
    ///
    /// AT-MOST-ONE-RESIDENT INVARIANT (load-bearing for liveness): two full-occupancy persistent spin-kernels
    /// in the shared context MUTUALLY STARVE — neither yields its SMs, so a kernel whose blocks lost the SMs
    /// never runs (can't observe the doorbell / watchdog), and tearing it down (`cuStreamSynchronize`, with an
    /// infinite backstop) HANGS FOREVER. So on a miss this DRAINS every existing wave engine (each then the
    /// sole persistent kernel -> clean doorbell teardown) BEFORE launching the new one, all under the
    /// `wave_build_latch` so two concurrent misses can't both launch. [LIMITATION: this means ONE wave engine
    /// TOTAL across all tables — alternating tables/projections rebuilds each time. Fine for R2.2b single-
    /// flight + the one-shape A/B; lifting it (multi-engine coexistence) needs sub-occupancy sizing or a
    /// shared multi-table kernel — deferred to R2.2c. KNOWN single-flight EDGE: a concurrent reader holding a
    /// to-be-drained engine across a rebuild keeps it alive past the new launch (-> transient two-kernel
    /// window); the supported model is single-flight (one submit at a time), so this is out of scope here and
    /// is the central concern multi-producer R2.2b-3 must resolve.]
    fn wave_read_engine_for(
        &self,
        table_name: &str,
        device_memory: &Arc<CudaResidentDeviceMemory>,
        filter_offset: u64,
        filter_idx: usize,
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Option<Arc<Mutex<WaveReadEngine>>> {
        let resident_device_ptr = device_memory.device_ptr();
        // Cache lookup with the full (col, buffer, proj) identity check.
        let lookup = || -> Option<Arc<Mutex<WaveReadEngine>>> {
            let cache = self
                .read_state
                .residency
                .wave_read_engine
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cache.get(table_name).and_then(|existing| {
                (existing.column_idx == filter_idx
                    && existing.resident_device_ptr == resident_device_ptr
                    && existing.projection_offsets == projection_offsets)
                    .then(|| Arc::clone(&existing.engine))
            })
        };
        // Fast path: cache hit (no build latch -> no contention on the hot path).
        if let Some(engine) = lookup() {
            return Some(engine);
        }
        // Miss: serialize the BUILD so two concurrent misses can't both launch a persistent kernel.
        let _build_guard = self
            .read_state
            .residency
            .wave_build_latch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Re-check under the build latch: another builder may have just published exactly what we need.
        if let Some(engine) = lookup() {
            return Some(engine);
        }
        // Drain EVERY existing wave engine BEFORE launching the new one (at-most-one-resident invariant).
        // Take the whole map out under the cache lock, then drop OUTSIDE it: each drained engine is now the
        // sole persistent kernel, so `WaveReadEngine::Drop` (doorbell + petter join + stream sync) completes
        // cleanly. Dropping outside the cache lock keeps cache-hit readers unblocked; the build latch (held)
        // is what serializes builders.
        let stale = {
            let mut cache = self
                .read_state
                .residency
                .wave_read_engine
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut *cache)
        };
        drop(stale);
        // Get-or-build the R1 index over THIS buffer (so the wave gathers with the SAME NULL-as-0 semantics
        // as the scan/index); `None` here -> caller uses lpb. Then launch the persistent kernel — no other
        // persistent wave kernel is resident now. `WaveReadEngine::new` set_current's the shared primary
        // context, clamps the grid to occupancy (C1), and runs until Drop / backstop / stale-heartbeat
        // watchdog.
        let (index, table_mask, hash_shift) = self.wave_resident_int4_index(
            table_name,
            device_memory,
            filter_offset,
            filter_idx,
            row_count,
        )?;
        let engine = WaveReadEngine::new(
            index,
            Arc::clone(device_memory),
            projection_offsets,
            table_mask,
            hash_shift,
            WAVE_ENGINE_RING_CAPACITY,
            WAVE_ENGINE_THREADS,
            WAVE_ENGINE_BACKSTOP_NS,
            WAVE_ENGINE_WATCHDOG_NS,
        )
        .ok()?;
        let engine = Arc::new(Mutex::new(engine));
        {
            let mut cache = self
                .read_state
                .residency
                .wave_read_engine
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // The map was drained above and builds are serialized by the latch, so this slot is absent.
            cache.insert(
                table_name.to_string(),
                WaveResidentReadEngine {
                    column_idx: filter_idx,
                    resident_device_ptr,
                    projection_offsets: projection_offsets.to_vec(),
                    engine: Arc::clone(&engine),
                },
            );
        }
        Some(engine)
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
        // Bound the table so a pathological row count can't allocate an absurd host vector (scan instead).
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
            // Mirror the kernel's hard 256-probe cap: a key the kernel could not reach within the cap must
            // NOT be silently placed here (it would read as a false not-found at probe time) — decline.
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
        let index_bytes: Vec<u8> = index.iter().flat_map(|entry| entry.to_le_bytes()).collect();
        let runtime = self.cuda_driver_probe_runtime();
        let gpu_id = device_memory.metadata().gpu_id;
        let memory = runtime
            .retain_device_memory_copy(gpu_id, &index_bytes)
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
        // R2.2b (Slice-1 audit note): `observe_kernel_exec_ms` is recorded for BOTH payload arms from the
        // batch WALL time (`batch_micros`), intentionally — it models "a batch executed on the GPU" and keeps
        // `kernel_exec_samples` comparable across the lpb and wave routes. The wave (Materialized) arm has no
        // per-batch CUDA EVENT (the persistent kernel is not timed per wave), so only the separate
        // `kernel_event_*` counter pair below is omitted (via the `None` elapsed), not this exec-ms sample.
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
        // ADR-009 R2.2b: both payload arms yield the SAME `Vec<CudaI32BatchProjectionRow>` — the
        // `Deferred` (lpb/scan/R1-index) arm drains its enqueued GPU submission here; the `Materialized`
        // (persistent wave engine) arm already drained the wave synchronously in `submit`, so the rows are
        // in hand and there is no per-batch kernel event (the persistent kernel is not timed per wave ->
        // `None`). Everything downstream (stable sort by `row_index`, `SqlValue::Int4` mapping, metrics,
        // results assembly) is shared and arm-agnostic, so the two routes are byte-identical by construction.
        // This per-needle path is COLD (the hot batcher uses the batched completion); it bridges the columnar
        // Materialized arm back to per-row via `into_rows` (DECISIONS "Tail latency").
        let (projected_rows, kernel_event_elapsed_us) = match pending.payload {
            RelationalRetainedInt4ProjectionPayload::Deferred(submission) => submission
                .complete_detached()
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?,
            RelationalRetainedInt4ProjectionPayload::Materialized(columns) => {
                (columns.into_rows(), None)
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
            RelationalRetainedInt4ProjectionPayload::Deferred(submission) => submission
                .complete_detached_columnar()
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?,
            RelationalRetainedInt4ProjectionPayload::Materialized(columns) => (columns, None),
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
        // O(n) COUNTING-SORT SCATTER by needle_index — replaces an O(n log n) global
        // `sort_by_key((needle_index, row_index))` that profiling showed was ~80% of the assembly
        // (~1800us/65536-batch). `cursor` walks each needle's contiguous range; `slot[d]` indexes into the
        // columnar rows. The scatter is stable, so it preserves the kernel's emit order within a needle.
        let mut slot = vec![0u32; total];
        let mut cursor: Vec<u32> = needle_ranges.iter().map(|&(start, _)| start).collect();
        for i in 0..nrows {
            let ni = projected.needle_indices[i] as usize;
            let d = cursor[ni] as usize;
            slot[d] = i as u32;
            cursor[ni] += 1;
        }
        // Byte-identity contract: each needle's rows ascend by row_index. The unique-key point-read case has
        // <=1 row/needle (already ordered). Only a NON-unique predicate yields multi-row needles, whose
        // kernel-emit order is atomic-race (not row_index) — sort just those sub-ranges (row_index is unique
        // within a needle, so this is a total order matching the old global sort). Skipped entirely otherwise.
        if any_multi {
            for &(start, count) in &needle_ranges {
                if count > 1 {
                    let s = start as usize;
                    let e = s + count as usize;
                    slot[s..e].sort_by_key(|&i| projected.row_indices[i as usize]);
                }
            }
        }
        let mut values = Vec::with_capacity(total * ncols);
        for &i in &slot {
            values.extend(projected.row_values(i as usize).iter().copied().map(SqlValue::Int4));
        }
        RelationalRetainedBatchResult {
            columns,
            access_path,
            gpu_id,
            rows: RowBlock::flat(values, ncols),
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
        let mut values = Vec::new();
        let mut needle_ranges = Vec::with_capacity(results.len());
        let mut acc = 0u32;
        for result in &results {
            let count = result.rows.len() as u32;
            needle_ranges.push((acc, count));
            acc += count;
            for row in result.rows.iter() {
                values.extend_from_slice(row);
            }
        }
        RelationalRetainedBatchResult {
            columns,
            access_path,
            gpu_id,
            rows: RowBlock::flat(values, ncols),
            needle_ranges,
        }
    }
}
