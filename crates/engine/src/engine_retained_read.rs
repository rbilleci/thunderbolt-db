//! Retained-read async job lifecycle (P0 §9.6 decomposition, behavior-preserving):
//! a focused `impl Engine` block for preparing, submitting, and completing
//! retained relational read jobs against resident device memory — including the
//! int4-projection retained submissions (try_submit / complete[_detached]) that
//! let a caller hold a device read view across calls.

use super::*;

mod device_index_append;
mod shard_point_lookup;
mod submission;
mod template;
mod wave_index;
mod wave_locate;

#[cfg(test)]
type RetainedCompletionPostHook = (
    usize,
    std::sync::Arc<std::sync::Barrier>,
    std::sync::Arc<std::sync::Barrier>,
);

#[cfg(test)]
fn retained_completion_post_hook() -> &'static Mutex<Option<RetainedCompletionPostHook>> {
    static HOOK: OnceLock<Mutex<Option<RetainedCompletionPostHook>>> = OnceLock::new();
    HOOK.get_or_init(|| Mutex::new(None))
}

/// Device index key layout. The parallel slices are validated together before any device read.
struct ShardDeviceIndexKey<'a> {
    key_id: usize,
    positions: &'a [usize],
    offsets: &'a [u64],
    blob_offsets: &'a [u64],
    blob_lens: &'a [u64],
}

impl Engine {
    fn ensure_retained_submission_available(
        &self,
        origin_wedge: &Arc<AtomicBool>,
    ) -> Result<(), ExecuteError> {
        if !Arc::ptr_eq(origin_wedge, &self.commit_path_wedged) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "retained read submission belongs to a different engine".to_string(),
            )));
        }
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if origin_wedge.load(AtomicOrdering::Acquire) {
            return Err(ExecuteError::Engine(self.commit_path_unavailable_error()));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn set_retained_completion_post_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *retained_completion_post_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    #[cfg(test)]
    fn run_retained_completion_post_hook(&self) {
        let hook = retained_completion_post_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some((engine, reached, resume)) = hook {
            if engine == self as *const Self as usize {
                reached.wait();
                resume.wait();
            }
        }
    }

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
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
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
                | "sharded_int4_equality_projection"
                | "sharded_int4_equality_multi_column_projection"
                | "sharded_int4_equality_mixed_column_projection"
        ) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "retained read jobs currently support only int4 equality projection routes, got {}",
                decision.query_shape
            ))));
        }
        let (table, bound, _copin_s) = self.bind_relational_select_for_execution(select)?;
        let snapshot_generation = if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_equality_projection"
                | "sharded_int4_equality_multi_column_projection"
                | "sharded_int4_equality_mixed_column_projection"
        ) {
            // A shard generation is published as one ArcSwap bundle rather than a legacy
            // single-snapshot descriptor. The route decision already proved every shard valid and
            // device-backed; use the commit boundary as the retained job's conservative freshness
            // token so any intervening publication makes submission decline.
            self.committed_seq()
        } else {
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
            handle.generation
        };
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
            snapshot_generation,
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
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        for job in jobs {
            let query_shape = job.route_id.split(':').next().unwrap_or("unknown");
            if matches!(
                query_shape,
                "sharded_int4_equality_projection"
                    | "sharded_int4_equality_multi_column_projection"
                    | "sharded_int4_equality_mixed_column_projection"
            ) {
                let decision = self.plan_relational_resident_route(&job.select);
                if !decision.accepted || decision.query_shape != query_shape {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "retained sharded read job for relation \"{}\" is no longer route-valid: {}",
                        job.table, decision.reason
                    ))));
                }
                let current_generation = self.committed_seq();
                if current_generation != job.snapshot_generation {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "retained sharded read job snapshot generation mismatch for relation \"{}\": job={}, current={}",
                        job.table, job.snapshot_generation, current_generation
                    ))));
                }
                continue;
            }
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
        if jobs.iter().all(|job| {
            job.route_id.split(':').next().is_some_and(|shape| {
                matches!(
                    shape,
                    "sharded_int4_equality_projection"
                        | "sharded_int4_equality_multi_column_projection"
                        | "sharded_int4_equality_mixed_column_projection"
                )
            })
        }) {
            // Projections over the authoritative shard set use the sharded general executor. It
            // recompacts referenced columns device-to-device and materializes only selected rows;
            // the legacy single-buffer equal-any submission cannot address shard-local offsets.
            // Keep the retained API's submission framing while returning a ready batch.
            let results = jobs
                .iter()
                .map(|job| self.execute_relational_select(&job.select))
                .collect::<Result<Vec<_>, _>>()?;
            let first_job = jobs.first();
            return Ok(RelationalRetainedReadSubmission {
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
                commit_path_wedged: Arc::clone(&self.commit_path_wedged),
                inner: RelationalRetainedReadSubmissionInner::Ready(results),
            });
        }
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
            commit_path_wedged: Arc::clone(&self.commit_path_wedged),
            inner: RelationalRetainedReadSubmissionInner::Ready(results),
        })
    }

    pub fn complete_relational_retained_read_submission(
        &self,
        submission: RelationalRetainedReadSubmission,
    ) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
        let origin_wedge = Arc::clone(&submission.commit_path_wedged);
        self.ensure_retained_submission_available(&origin_wedge)?;
        let results = match submission.inner {
            RelationalRetainedReadSubmissionInner::Ready(results) => results,
            RelationalRetainedReadSubmissionInner::ReadyBatched(result) => {
                Self::expand_ready_batched_result(*result)
            }
            RelationalRetainedReadSubmissionInner::PendingInt4Projection(pending) => {
                self.complete_relational_retained_int4_projection_submission(*pending)?
            }
        };
        #[cfg(test)]
        self.run_retained_completion_post_hook();
        self.ensure_retained_submission_available(&origin_wedge)?;
        Ok(results)
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
                commit_path_wedged: Arc::clone(&self.commit_path_wedged),
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
            commit_path_wedged: Arc::clone(&self.commit_path_wedged),
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

    /// CROSS-SHARD PK INDEX (sub-slice 1): resolve `filter_idx = key` to the resident `(shard_id, LOCAL row)`
    /// positions via the per-shard device-resident PK hash index. Thin projection of
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

    /// A write locate may trust a shard buffer only while the captured authoritative generation
    /// owns exactly that `Arc` (pointer-identical). Transaction-scoped reads validate against their
    /// captured shard generation; other callers validate against the current device-memory cell.
    /// A tombstoned or replaced cell declines so no load-bearing miss can be answered from stale
    /// device bytes.
    pub(crate) fn shard_write_locate_cell_live(
        &self,
        table_name: &str,
        shard_id: u32,
        descriptor_memory: &Arc<CudaResidentDeviceMemory>,
    ) -> bool {
        if let Some(snapshot) = self.current_transaction_read_snapshot() {
            return snapshot
                .transaction_shards()
                .get(table_name)
                .and_then(|shards| shards.iter().find(|shard| shard.shard_id == shard_id))
                .and_then(|shard| shard.device_memory.as_ref())
                .is_some_and(|memory| Arc::ptr_eq(memory, descriptor_memory));
        }
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
        // R3-004: key-to-slot addressing is unconditionally device work. A device-index decline
        // returns `None` to the GPU scan; there is no host index or host relational probe route.
        self.locate_resident_pk_via_device(table, key_id, key)
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
    ) -> Result<Option<usize>, ExecuteError> {
        self.ensure_commit_path_available()?;
        let Some(table) = self.relational_catalog_table(table_name) else {
            return Ok(None);
        };
        let Ok(filter_idx) = crate::rel_exec_helpers::relational_column_index(&table, filter_col)
        else {
            return Ok(None);
        };
        let mut selected_indexes = Vec::with_capacity(proj_cols.len());
        for c in proj_cols {
            let Ok(index) = crate::rel_exec_helpers::relational_column_index(&table, c) else {
                return Ok(None);
            };
            selected_indexes.push(index);
        }
        let Some(proj) = self.gather_sharded_int4_point_lookups_batched(
            self.committed_seq(),
            &table,
            filter_idx,
            &selected_indexes,
            needles,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(proj.matched_needle_count()))
    }

    /// lpb-for-shards WIRING: serve a shard-resident int4 point-lookup BATCH as ONE dispatchable
    /// Compatibility wrapper for callers that require the established public one-range-per-needle result.
    /// Binds `select`, verifies it is a shard-resident single int4-`Eq` point lookup, runs
    /// `gather_sharded_int4_point_lookups_batched` over `needles`, and wraps the projection with the shared
    /// schema. Returns `None` (the batcher keeps its existing behavior — byte-identical) when the flag is OFF,
    /// the table is not shard-resident, the shape is not a single int4 equality, or the gather declines
    /// (dup / non-int4). Runtime failures are returned as typed errors and never become eligibility declines.
    /// `needles` must be DISTINCT (the caller dedups), per the gather's contract.
    /// `access_path` is a label (`FullTableScan`) — the batcher's dispatch consumes only `columns` +
    /// `needle_values`, never `access_path`.
    pub fn submit_sharded_point_lookups_batched(
        &self,
        select: &Select,
        needles: &[i32],
    ) -> Result<Option<RelationalRetainedBatchResult>, ExecuteError> {
        Ok(self
            .submit_sharded_point_lookups_batched_compact(select, needles)?
            .map(RelationalPointBatchResult::into_compat))
    }

    /// Production point-batcher entry. Dense all-present results keep an internal identity mapping and avoid
    /// the 8-byte-per-needle compatibility range allocation; runtime failures remain typed errors.
    pub fn submit_sharded_point_lookups_batched_compact(
        &self,
        select: &Select,
        needles: &[i32],
    ) -> Result<Option<RelationalPointBatchResult>, ExecuteError> {
        self.ensure_commit_path_available()?;
        if !self.shard_batched_point_read_enabled() {
            return Ok(None);
        }
        let Ok((table, bound, copin_s)) = self.bind_relational_select_for_execution(select) else {
            return Ok(None);
        };
        if self.resident_shard_count(&table.name) == 0 {
            return Ok(None); // not shard-resident -> the batcher's single-buffer / per-query path
        }
        let Some((filter_idx, _needle)) =
            crate::engine_expr::shard_point_lookup_int4_eq(&bound, &table)
        else {
            return Ok(None);
        };
        // SC5 rider: the gather reads at the STATEMENT'S pinned boundary (was: an internal
        // committed_seq re-read that broke catalog<->data co-pinning).
        let Some(proj) = self.gather_sharded_int4_point_lookups_batched(
            copin_s,
            &table,
            filter_idx,
            &bound.selected_indexes,
            needles,
        )?
        else {
            return Ok(None);
        };
        let gpu_id = self
            .read_residency_shards()
            .get(&table.name)
            .and_then(|s| s.first())
            .map(|s| s.gpu_id)
            .unwrap_or(0);
        Ok(Some(RelationalPointBatchResult::new(
            Arc::new(bound.selected_columns),
            Arc::new(RelationalAccessPath::FullTableScan),
            gpu_id,
            proj.values,
            proj.ncols,
            proj.needle_ranges,
        )))
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
        let origin_wedge = Arc::clone(&submission.commit_path_wedged);
        self.ensure_retained_submission_available(&origin_wedge)?;
        let result = match submission.inner {
            RelationalRetainedReadSubmissionInner::PendingInt4Projection(pending) => {
                Self::complete_int4_projection_batched_detached(*pending)?
            }
            RelationalRetainedReadSubmissionInner::ReadyBatched(result) => *result,
            RelationalRetainedReadSubmissionInner::Ready(results) => {
                Self::fold_ready_results_batched(results)
            }
        };
        #[cfg(test)]
        self.run_retained_completion_post_hook();
        self.ensure_retained_submission_available(&origin_wedge)?;
        Ok(result)
    }

    pub(crate) fn expand_ready_batched_result(
        result: RelationalRetainedBatchResult,
    ) -> Vec<RelationalSelectResult> {
        let RelationalRetainedBatchResult {
            columns,
            access_path,
            gpu_id,
            values,
            ncols,
            needle_ranges,
        } = result;
        needle_ranges
            .into_iter()
            .map(|(start, count)| {
                let first = start as usize * ncols;
                let last = first + count as usize * ncols;
                let rows = RowBlock::flat(
                    values[first..last]
                        .iter()
                        .copied()
                        .map(SqlValue::Int4)
                        .collect(),
                    ncols,
                );
                RelationalSelectResult {
                    columns: Arc::clone(&columns),
                    rows,
                    planned_target: DeviceTarget::Gpu(gpu_id),
                    executed_target: DeviceTarget::Gpu(gpu_id),
                    fallback_reason: None,
                    access_path: Arc::clone(&access_path),
                }
            })
            .collect()
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

/// Legacy single-buffer wave-index helper: the pure host build of the open-addressing int4 hash table
/// `(key<<32)|(row+1)` (0 = empty), Fibonacci hash `(key*0x9E37_79B1)>>hash_shift` + linear probe with the
/// kernel's hard 256-probe cap. The production shard index is device-built using this layout; it does not call
/// this helper. Returns
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

/// Test-only layout oracle: probe the host wave-index table built by `build_int4_pk_hash_table_host`
/// for `key`, returning the local row index (0-based) or `None` (absent). Mirrors the device probe kernel:
/// Fibonacci hash → linear probe up to the 256 cap, matching the high 32 bits (the key) and unpacking
/// `row = (entry & 0xFFFF_FFFF) - 1`. An empty slot (0) terminates the probe = not found. A NULL int4 is
/// materialized as `0`, so `key = 0` probes exactly as the build indexed it (agrees with the scan).
#[cfg(test)]
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
        SqlType::Bool => shard
            .resident_device_bool_columns
            .iter()
            .find(|layout| layout.name == column.name)
            .map(|layout| layout.bitmap_byte_offset),
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

fn shard_key_column_blob_len(
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
            .map(|layout| layout.bytes_len),
        _ => Some(0),
    }
}

/// U1: the batched visible-locate verdicts for one wave's delete needles. Parallel per-needle
/// vectors (`counts[i]` visible matches at needle i's snapshot; `shard_ids[i]`/`slots[i]` = the
/// first visible target and `row_ids[i]` = its stable entity identity, meaningful iff
/// `counts[i] >= 1`) + `probed`, the (shard_id, MAIN device
/// region) identity handles of every probed shard captured from the SAME `shards.load()`
/// snapshot — the apply-time tombstone MUST recheck cell liveness against these exact regions
/// (a VACUUM/re-admit between locate and apply re-clusters slots; stamping a stale slot would
/// tombstone the wrong row).
#[derive(Default)]
pub(crate) struct WaveVisibleLocate {
    pub(crate) counts: Vec<u32>,
    pub(crate) shard_ids: Vec<u32>,
    pub(crate) slots: Vec<u32>,
    pub(crate) row_ids: Vec<u64>,
    pub(crate) latest_write: Vec<u64>,
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
    /// device DML resolve reads `row_id[slot]` to derive the control-plane key. `None` =
    /// identity-unknown lineage (benchmark/synthetic), which callers reject with a loud availability error.
    pub(crate) row_id: Option<Arc<CudaResidentDeviceMemory>>,
}

/// Step 1 (lpb-for-shards): the batched point-lookup projection. `values` is row-major int4, `ncols` wide, in
/// NEEDLE ORDER. Mixed/absent batches use `needle_ranges[i] = (start_row, row_count)`; an empty range vector
/// with non-empty values is the private all-present identity mapping. The schema
/// (columns / access_path) is shape metadata the caller wraps around this raw projection.
/// Consumed by the production batch-result wiring (`submit_sharded_point_lookups_batched`), the
/// differential tests, and the benches.
#[derive(Debug)]
pub(crate) struct BatchedShardProjection {
    pub(crate) ncols: usize,
    pub(crate) values: Vec<i32>,
    pub(crate) needle_ranges: Vec<(u32, u32)>,
}

impl BatchedShardProjection {
    fn matched_needle_count(&self) -> usize {
        if self.needle_ranges.is_empty() && !self.values.is_empty() {
            debug_assert!(self.ncols > 0 && self.values.len().is_multiple_of(self.ncols));
            self.values.len() / self.ncols
        } else {
            self.needle_ranges
                .iter()
                .filter(|&&(_, count)| count > 0)
                .count()
        }
    }

    #[cfg(test)]
    pub(crate) fn needle_count(&self) -> usize {
        if self.needle_ranges.is_empty() && !self.values.is_empty() {
            self.values.len() / self.ncols
        } else {
            self.needle_ranges.len()
        }
    }

    #[cfg(test)]
    pub(crate) fn needle_range(&self, needle: usize) -> (u32, u32) {
        if self.needle_ranges.is_empty() && !self.values.is_empty() {
            (needle as u32, 1)
        } else {
            self.needle_ranges[needle]
        }
    }
}

#[cfg(test)]
mod cross_shard_pk_index_tests {
    use super::{build_int4_pk_hash_table_host, probe_int4_pk_hash_table};

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
}
