//! Retained-read async job lifecycle (P0 §9.6 decomposition, behavior-preserving):
//! a focused `impl Engine` block for preparing, submitting, and completing
//! retained relational read jobs against resident device memory — including the
//! int4-projection retained submissions (try_submit / complete[_detached]) that
//! let a caller hold a device read view across calls.

use super::*;

mod device_index_append;
mod shard_point_lookup;
mod template;
mod submission;
mod wave_index;
mod wave_locate;

/// One generation-validated host PK-cache source. The cache key, resident owner, exact column
/// offset, and live row extent travel together so probe helpers cannot mix shard generations.
struct ShardPkCacheSource<'a> {
    table_name: &'a str,
    shard_id: u32,
    col_idx: usize,
    device_memory: &'a Arc<CudaResidentDeviceMemory>,
    filter_offset: u64,
    row_count: usize,
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
                ShardPkCacheSource {
                    table_name: &table.name,
                    shard_id: shard.shard_id,
                    col_idx: filter_idx,
                    device_memory: &device_memory,
                    filter_offset,
                    row_count: shard.row_count,
                },
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
