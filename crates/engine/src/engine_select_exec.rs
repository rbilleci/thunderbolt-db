//! Relational SELECT entry + dispatch (P0 §9.6 decomposition, behavior-preserving):
//! a focused `impl Engine` block for the public SELECT entry points
//! (execute_relational_select, _text, _instrumented, _cpu_pinned[_instrumented],
//! execute_relational_function) and the device-route dispatch variants
//! (_with_cuda_driver_probe, _with_resident_snapshot_probe, _with_resident_route)
//! that pick the CPU/GPU path and hand off to the resident-probe executors.

use super::*;

impl Engine {
    pub fn execute_relational_select(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_relational_select_instrumented(select, || {})
    }

    /// Parse and execute a relational SELECT from text, accepting native
    /// `pg_catalog`/`information_schema` catalog relations (Phase-3 M2). This is the
    /// text -> rows entry a consolidated server uses for catalog introspection; user SQL
    /// without a catalog reference parses and runs exactly as via the strict path.
    ///
    /// Routing (Charter rule 2, the general GPU executor): the hand-rolled parser is tried FIRST — it
    /// gates the tuned enumerated resident-route fast-paths + the catalog, and it strictly REJECTS
    /// (never mis-parses) the predicates it cannot express. When it rejects (an arithmetic / boolean
    /// `WHERE` it cannot represent — e.g. `a + b > 400`, possibly mixed with `AND`/`OR`), the SELECT is
    /// routed to the GENERAL Expr executor: SQL text -> libpg_query -> `ResidentExpr` -> evaluated on
    /// the table's GPU-resident snapshot. This only ADDS coverage; it never shadows the hand-rolled
    /// path, so simple shapes keep their fused kernels and there is no perf regression. The general
    /// path runs on the GPU only when the residency snapshot reflects the visible committed set (it
    /// enforces the snapshot validity/identity invariant) and raises a hard error otherwise — there is
    /// no CPU re-execution of an arithmetic predicate (the hand-rolled CPU path cannot express it).
    /// The strict `parse_command` entry the legacy pgwire server uses is untouched (dual-entry).
    pub fn execute_relational_select_text(
        &self,
        text: &str,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        match parse_command_allowing_catalog(text) {
            Ok(Command::Select(select)) => self.execute_relational_select(&select),
            Ok(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "expected a SELECT statement".to_string(),
            ))),
            // The hand-rolled parser cannot express this SELECT; route it to the general GPU Expr
            // executor. (For a SELECT it does parse, the line above already ran it — the fast-paths and
            // catalog are unchanged.)
            Err(_) => self.execute_resident_expr_select_sql(text),
        }
    }

    /// [`Engine::execute_relational_select`] with a hook invoked at the START of the read — AFTER the
    /// reader would load its pin boundary but conceptually BEFORE it binds the catalog / pins data
    /// (PART B test seam). The concurrency-correctness suite uses this to rendezvous a reader at a
    /// barrier so it deterministically STRADDLES a concurrent shape-changing DDL commit: the reader
    /// parks at the hook, a writer commits an ADD/DROP COLUMN, then the reader proceeds to bind + pin.
    /// With co-pinning the reader selects the catalog as-of its boundary and pins data at the SAME
    /// boundary, so its (catalog, data) pair is always consistent; without it the bind and the data
    /// pin could land on different generations and the decode would mismatch the catalog shape.
    /// Production passes an empty hook, so this is a zero-overhead extraction, not a separate path.
    pub fn execute_relational_select_instrumented(
        &self,
        select: &Select,
        on_pinned: impl FnOnce(),
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // PART B test seam. The hook is threaded to the CPU pinned read, where it fires in the window
        // BETWEEN binding the catalog and pinning the data — exactly the window co-pinning closes. The
        // test parks a reader there while a writer commits a shape-changing DDL: with co-pinning the
        // data pin reuses the SAME boundary the bind selected its catalog at, so the (catalog, data)
        // pair stays consistent; without it the pin re-loads `committed_seq` (now newer) while the
        // catalog is older, and the decode mismatches the catalog shape. A view/matview is resolved
        // as-of the boundary too; for a plain table SELECT (the test's case) the read goes straight to
        // the co-pinned CPU path.
        //
        // The read pins ONE `committed_seq` boundary and resolves the catalog as-of it. It does NOT
        // register an active snapshot (reads stay OFF the `active_snapshots` mutex — true lock-free):
        // the catalog ring's COUNT floor (`MIN_RETAINED_CATALOG_GENERATIONS`) guarantees this read's
        // generation is still present even if a flurry of concurrent DDLs commit while the statement
        // runs, so `catalog_as_of(s)` never falls back to a too-new generation.
        let s = self.committed_seq();
        let catalog = self.read_state.catalog_as_of(s);
        if let Some(view) = catalog.relational_views.get(&select.table).cloned() {
            if !select_is_plain_view_scan(select) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "only plain SELECT * FROM view is supported for views".to_string(),
                )));
            }
            return self.execute_relational_select(&view.query);
        }
        if let Some(view) = catalog
            .relational_materialized_views
            .get(&select.table)
            .cloned()
        {
            if !select_is_plain_view_scan(select) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "only plain SELECT * FROM materialized view is supported for materialized views"
                        .to_string(),
                )));
            }
            return Ok(RelationalSelectResult {
                columns: view.columns,
                rows: view.rows,
                planned_target: DeviceTarget::Cpu,
                executed_target: DeviceTarget::Cpu,
                fallback_reason: Some(FallbackReason::NotGpuEligible),
                access_path: RelationalAccessPath::FullTableScan,
            });
        }
        // Phase-3 M2: a SELECT against a synthesized pg_catalog/information_schema relation
        // runs through the SAME bind -> filter -> project -> order/limit core as a user
        // table, over rows projected from this pinned catalog generation (MVCC-consistent).
        // Resolved AFTER user tables/views so a real relation always shadows a catalog name.
        if !catalog.relational_catalog.contains_key(&select.table) {
            if let Some((catalog_table, catalog_rows)) =
                synthesize_catalog_relation(&select.table, &catalog)
            {
                let bound = bind_relational_select(&catalog_table, select)?;
                let result = MvccReadResult {
                    planned_target: DeviceTarget::Cpu,
                    executed_target: DeviceTarget::Cpu,
                    fallback_reason: Some(FallbackReason::NotGpuEligible),
                    rows: catalog_rows
                        .iter()
                        .map(|row| MvccReadRow {
                            source_key: None,
                            key: None,
                            value: Some(encode_relational_row(row)),
                        })
                        .collect(),
                };
                return self.finalize_relational_select(
                    select,
                    catalog_table,
                    bound,
                    RelationalAccessPath::FullTableScan,
                    result,
                );
            }
        }
        let resident_route = self.plan_relational_resident_route(select);
        if resident_route.accepted {
            // The resident route does not use the inter-bind-and-pin window the hook targets; fire the
            // hook now (so a barrier'd test still rendezvouses) and run the resident route.
            on_pinned();
            match self.execute_relational_select_with_resident_route(select) {
                Ok(result) => return Ok(result),
                // A concurrent committer can tombstone the table's GPU residency (publish(None))
                // under the commit_mutex AFTER we accepted the resident route but BEFORE the probe
                // loaded the device-memory cell (the writer holds only the engine READ lock, so it
                // races our read). That surfaces as the precise "no retained resident device memory"
                // probe error — NOT a genuine device failure. Transparently fall back to the CPU
                // pinned-read path (which reads the current published data generation at one pinned
                // boundary), exactly as a non-resident table would. Any OTHER error (a real
                // GPU/CUDA failure, a bind error, etc.) propagates unchanged so we never mask it.
                Err(err) if err.is_residency_invalidated() => {
                    return self.execute_relational_select_cpu_pinned(select);
                }
                Err(err) => return Err(err),
            }
        }
        self.execute_relational_select_cpu_pinned_instrumented(select, on_pinned)
    }

    /// The CPU pinned-read path for a relational SELECT (write-half MVCC, Stage 4): bind, pin ONE
    /// generation + ONE visibility boundary for the whole statement (prereq #1 — the value-index
    /// lookup AND the row resolution both read from `pin`, never two `load_table()`s), build + run
    /// the MVCC query, finalize. Used both when the table is not GPU-resident AND as the transparent
    /// fallback when a resident route's residency was invalidated mid-statement by a concurrent
    /// committer (see [`Engine::execute_relational_select`]).
    pub(crate) fn execute_relational_select_cpu_pinned(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_relational_select_cpu_pinned_instrumented(select, || {})
    }

    /// [`Engine::execute_relational_select_cpu_pinned`] with the PART B test hook fired in the window
    /// BETWEEN binding the catalog (which captures the co-pin boundary `copin_s`) and pinning the data
    /// at that SAME `copin_s`. This is precisely the window co-pinning closes: the pin reuses
    /// `copin_s`, so a DDL committed while the hook is parked cannot make the data pin a different
    /// generation than the bound catalog. Production passes an empty hook (zero overhead).
    fn execute_relational_select_cpu_pinned_instrumented(
        &self,
        select: &Select,
        on_bound_before_pin: impl FnOnce(),
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        on_bound_before_pin();
        let pin = self.pin_relational_read_at(&select.table, copin_s);
        let (query, access_path) =
            self.relational_select_mvcc_query(select, &table, &bound, &pin)?;
        let result = self.execute_mvcc_query_on_pin(&pin, &query)?;
        self.finalize_relational_select(select, table, bound, access_path, result)
    }

    pub fn execute_relational_function(
        &self,
        call: &SelectFunction,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Lock-free read path (Stage 2 — blocker #1): pin the catalog snapshot for the function lookup.
        let catalog = self.catalog_snapshot();
        let Some(function) = catalog.relational_functions.get(&call.name) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "function \"{}\" does not exist",
                call.name
            ))));
        };
        let value = parse_bounded_sql_function_body(&function.body, function.return_type)?;
        self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
        Ok(RelationalSelectResult {
            columns: vec![RelationalColumn {
                id: 0,
                table_oid: function.oid,
                attnum: 1,
                name: function.name.clone(),
                ty: function.return_type,
                domain: None,
                default: None,
                type_oid: function.return_type.postgres_oid(),
                type_size: function.return_type.type_size(),
            }],
            rows: vec![vec![value]],
            planned_target: DeviceTarget::Cpu,
            executed_target: DeviceTarget::Cpu,
            fallback_reason: Some(FallbackReason::NotGpuEligible),
            access_path: RelationalAccessPath::FullTableScan,
        })
    }

    pub fn execute_relational_select_with_cuda_driver_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let pin = self.pin_relational_read_at(&select.table, copin_s);
        let (query, access_path) =
            self.relational_select_mvcc_query(select, &table, &bound, &pin)?;
        let result =
            self.execute_mvcc_query_with_cuda_driver_probe_on_store(pin.store(), &query)?;
        self.finalize_relational_select(select, table, bound, access_path, result)
    }

    pub fn execute_relational_select_with_resident_snapshot_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let pin = self.pin_relational_read_at(&select.table, copin_s);
        let (_query, access_path) =
            self.relational_select_mvcc_query(select, &table, &bound, &pin)?;
        let snapshot = self
            .relational_residency_snapshot_ref(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident snapshot",
                    table.name
                )))
            })?;
        if snapshot.schema != table.schema || snapshot.table != table.name {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot no longer matches catalog table identity".to_string(),
            )));
        }
        if !snapshot.is_valid() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" resident snapshot is invalid",
                table.name
            ))));
        }

        let result = MvccReadResult {
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            rows: snapshot
                .resident_rows
                .iter()
                .map(|row| MvccReadRow {
                    source_key: None,
                    key: None,
                    value: Some(encode_relational_row(row)),
                })
                .collect(),
        };
        self.finalize_relational_select(select, table, bound, access_path, result)
    }

    pub fn execute_relational_select_with_resident_route(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let decision = self.plan_relational_resident_route(select);
        if !decision.accepted {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident route rejected: {}",
                decision.reason
            ))));
        }

        let before_metrics = self.metrics.snapshot();
        if let Some(device_memory) = self.read_state.residency.device_memory.get(&decision.table) {
            // Make this allocation's CUDA context current on the calling thread so a
            // concurrent reader (not the context's creator) can launch — without it the
            // kernel fails with INVALID_CONTEXT (P1-M3 step 3c / gate 2).
            let _ = device_memory.set_current_context();
            device_memory.clear_last_kernel_event_elapsed_us();
        }
        for device_memory in self
            .read_state
            .residency
            .partition_device_memory
            .published_owners_for_table(&decision.table)
        {
            let _ = device_memory.set_current_context();
            device_memory.clear_last_kernel_event_elapsed_us();
        }
        let route_started = Instant::now();
        let result = match decision.query_shape.as_str() {
            "count_all" | "int4_equality_count" | "int4_range_count" | "int4_scalar_aggregate"
            | "int4_filtered_scalar_aggregate" | "int4_between_scalar_aggregate" => {
                self.execute_resident_plan(select)
            }
            "partitioned_count_all" => self
                .execute_relational_partitioned_count_with_resident_device_memory_probe(select),
            "partitioned_int4_equality_projection" => self
                .execute_relational_partitioned_equality_projection_with_resident_device_memory_probe(
                    select,
                ),
            "partitioned_int4_equality_multi_column_projection" => self
                .execute_relational_partitioned_equality_multi_column_projection_with_resident_device_memory_probe(
                    select,
                ),
            "partitioned_int4_equality_sum" => self
                .execute_relational_partitioned_equality_sum_with_resident_device_memory_probe(
                    select,
                ),
            "partitioned_int4_between_avg" => self
                .execute_relational_partitioned_between_avg_with_resident_device_memory_probe(
                    select,
                ),
            "partitioned_int4_filtered_min" => self
                .execute_relational_partitioned_filtered_min_with_resident_device_memory_probe(
                    select,
                ),
            "partitioned_int4_filtered_avg" => self
                .execute_relational_partitioned_filtered_avg_with_resident_device_memory_probe(
                    select,
                ),
            "partitioned_int4_filtered_max" => self
                .execute_relational_partitioned_filtered_max_with_resident_device_memory_probe(
                    select,
                ),
            "text_prefix_like_count" => {
                self.execute_relational_text_prefix_count_with_resident_device_memory_probe(select)
            }
            "int4_filter_group_count" => {
                self.execute_relational_filter_group_count_with_resident_device_memory_probe(select)
            }
            "int4_grouped_aggregate" => {
                self.execute_relational_grouped_aggregate_with_resident_device_memory_probe(select)
            }
            "int4_filtered_grouped_aggregate" => self
                .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(
                    select,
                ),
            "int4_projection" => {
                self.execute_relational_projection_with_resident_device_memory_probe(select)
            }
            "int4_equality_projection" => self
                .execute_relational_equality_projection_with_resident_device_memory_probe(select),
            "int4_equality_multi_column_projection"
            | "int4_composite_equality_multi_column_projection"
            | "int4_equality_mixed_column_projection" => self
                .execute_relational_equality_multi_column_projection_with_resident_device_memory_probe(
                    select,
                ),
            "int4_ordered_projection" => {
                self.execute_relational_ordered_projection_with_resident_device_memory_probe(select)
            }
            "int4_distinct_projection" => self
                .execute_relational_distinct_projection_with_resident_device_memory_probe(select),
            "int4_filtered_distinct_projection" => self
                .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(
                    select,
                ),
            shape => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident route accepted unsupported execution shape: {shape}"
            )))),
        }?;
        let kernel_event_elapsed_us = self
            .read_state
            .residency
            .device_memory
            .get(&decision.table)
            .and_then(|device_memory| device_memory.last_kernel_event_elapsed_us());
        if let Some(elapsed_us) = kernel_event_elapsed_us {
            self.metrics.observe_kernel_event_elapsed_us(elapsed_us);
        }
        let after_metrics = self.metrics.snapshot();
        self.read_state
            .route_telemetry
            .record_route_execution_observation(
                &decision.table,
                RelationalResidentRouteExecutionObservation {
                    h2d_bytes: after_metrics
                        .h2d_bytes_total
                        .saturating_sub(before_metrics.h2d_bytes_total),
                    d2h_bytes: after_metrics
                        .d2h_bytes_total
                        .saturating_sub(before_metrics.d2h_bytes_total),
                    kernel_samples: after_metrics
                        .kernel_exec_samples
                        .saturating_sub(before_metrics.kernel_exec_samples),
                    kernel_ms: after_metrics
                        .kernel_exec_total_ms
                        .saturating_sub(before_metrics.kernel_exec_total_ms),
                    kernel_event_elapsed_us,
                    rows: result.rows.len(),
                    wall_micros: route_started
                        .elapsed()
                        .as_micros()
                        .try_into()
                        .unwrap_or(u64::MAX),
                },
            );
        Ok(result)
    }
}
