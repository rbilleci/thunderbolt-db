//! Relational SELECT entry + dispatch (P0 §9.6 decomposition, behavior-preserving):
//! a focused `impl Engine` block for the public SELECT entry points
//! (execute_relational_select, _text, _instrumented, _cpu_pinned[_instrumented],
//! execute_relational_function) and the device-route dispatch variants
//! (_with_cuda_driver_probe, _with_resident_snapshot_probe, _with_resident_route)
//! that pick the CPU/GPU path and hand off to the resident-probe executors.

use super::*;

/// A non-grouped projection (no aggregate / GROUP BY / HAVING) with an ORDER BY whose keys are ALL
/// i64-sortable int columns (int2/int4/int8/date/timestamp) -- one OR several keys (e.g.
/// `ORDER BY a ASC, b DESC, c`). Such a query is routed to the general GPU Expr executor, which sorts
/// the surviving rows on the GPU (multi-key bitonic) -- the charter-native path, retiring the
/// enumerated ordered-projection shape for this case. A SINGLE text key also routes here (the byte-wise
/// GPU text sort), and a MIXED int+text tuple uses the heterogeneous comparator. Other shapes --
/// numeric/uuid keys, expressions -- stay on the existing path transitionally; the enumerated/CPU path
/// rejects multi-key, so a multi-key sort with a non-routable key is a clean error, never first-key-only.
fn select_is_gpu_sortable_projection(select: &Select, table: &RelationalTable) -> bool {
    if select.group_by.is_some() || !select.having_groups.is_empty() {
        return false;
    }
    if select.order_by.is_empty() {
        return false;
    }
    if !matches!(
        select.projection,
        SelectProjection::All | SelectProjection::Columns(_)
    ) {
        return false;
    }
    // Every ORDER BY key must be a base column the GPU sort handles: an i64-sortable int
    // (int2/4/8/date/timestamp), text (byte-wise), or numeric/uuid (the 16-byte comparator). A single
    // text key uses the text leg, an all-int tuple the i64 key matrix, anything with a text/numeric/uuid
    // key the heterogeneous comparator. Expression keys go via the libpg_query (Err-arm) path, not here.
    select.order_by.iter().all(|order| {
        table
            .columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(&order.column))
            .is_some_and(|c| {
                matches!(
                    c.ty,
                    SqlType::Int4
                        | SqlType::Int8
                        | SqlType::Int2
                        | SqlType::Date
                        | SqlType::Timestamp
                        | SqlType::Text
                        | SqlType::Numeric { .. }
                        | SqlType::Uuid
                )
            })
    })
}

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
            Ok(Command::Select(select)) => {
                // A non-grouped ORDER BY over an i64-sortable int key is a charter-native GPU sort: run
                // it through the general Expr executor (which sorts on the GPU via the bitonic sort),
                // NOT the enumerated ordered-projection shape (Charter rule 2) or the CPU path.
                if let Some(table) = self.relational_catalog_table(&select.table) {
                    // Only route to the general GPU path when the table is GPU-resident: that path has
                    // no CPU fallback, so for a non-resident table it would hard-error -- whereas the
                    // strict/CPU-pinned path below serves non-resident tables correctly. (The general
                    // path is the GPU-native one; this gate just preserves the entry's contract for
                    // tables not yet resident.) SLICE B: SHARD-resident tables qualify too — the general
                    // path now recompacts them into the unified exec source, so a sortable projection
                    // gets the SAME NULL-correct on-device sort instead of falling through the rejected
                    // shape route to the CPU engine's host sort (whose NULL placement diverges from PG —
                    // caught by the sharded-vs-single-buffer ORDER BY differential). The shard arm
                    // requires a PURELY int4-section table (int4/int2/date): the unified exec source
                    // gathers only the int4 sections (+ null bitmaps), so a text/int8/numeric/bool
                    // column reference would hard-error on the no-fallback general path where the CPU
                    // pinned path previously returned rows (audit P2: `mt (id INT, name TEXT)` sharded +
                    // `ORDER BY id`). Mixed-type sharded tables keep the CPU pinned path until the
                    // unified source gathers every section.
                    let shard_resident_int4_only = || {
                        table.columns.iter().all(|c| {
                            // TYPE-COVERAGE track 2 slice 2: the unified exec source now
                            // gathers the i64 sections too, so Int8/Timestamp columns route
                            // to the GPU general path when the flag admitted them to shards
                            // (without the flag such tables are never shard-resident and the
                            // shards.load() check below keeps this arm false). TYPE-COVERAGE
                            // numeric slice: the b128 sections (Numeric / Uuid, 16-byte) are
                            // now gathered by the unified exec source too, so those columns
                            // route to the GPU sort as well -- a plain-column ORDER BY over a
                            // numeric/uuid table no longer falls through to the CPU pinned
                            // path (the sharded b128 ORDER BY differential).
                            matches!(
                                c.ty,
                                SqlType::Int4
                                    | SqlType::Int2
                                    | SqlType::Date
                                    | SqlType::Int8
                                    | SqlType::Timestamp
                                    | SqlType::Numeric { .. }
                                    | SqlType::Uuid
                            )
                        }) && self
                            .read_state
                            .residency
                            .shards
                            .load()
                            .get(&table.name)
                            .is_some_and(|shards| !shards.is_empty())
                    };
                    if select_is_gpu_sortable_projection(&select, &table)
                        && (self.relational_residency_snapshot(&select.table).is_some()
                            || shard_resident_int4_only())
                    {
                        return self.execute_resident_expr_select_sql(text);
                    }
                }
                self.execute_relational_select(&select)
            }
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
        // Multi-key ORDER BY (`ORDER BY a, b, ...`) is GPU-only: it runs on the general Expr executor's
        // bitonic-sort path, routed in `execute_relational_select_text` when every key is an i64-sortable
        // base column on a GPU-resident table. This enumerated/CPU path has no multi-key sort and must
        // NOT silently sort by the first key only -- so reject it. A multi-key sort reaching here means a
        // non-routable key (text/numeric/uuid/expression) or a non-resident table: a clean error, never a
        // wrong (first-key-only) result.
        if select.order_by.len() > 1 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "multi-key ORDER BY runs on the GPU sort path (every key must be an i64-sortable \
                 base column on a GPU-resident table)"
                    .to_string(),
            )));
        }
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
                columns: Arc::new(view.columns),
                rows: (view.rows).into(),
                planned_target: DeviceTarget::Cpu,
                executed_target: DeviceTarget::Cpu,
                fallback_reason: Some(FallbackReason::NotGpuEligible),
                access_path: Arc::new(RelationalAccessPath::FullTableScan),
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
        // RETIREMENT A4e — the READ-SIDE ladder seam: a host-path read on an ELIDED table would
        // scan the STALE store (elided commits never installed). Rehydrate first (device gather +
        // reconciliation, sticky de-elision), exactly like the DML ladder — any read shape the
        // device routes cannot serve costs one O(table) rehydration instead of wrong results.
        if self.host_install_elision_enabled() && self.table_install_elided(&select.table) {
            // Audit B3: the rehydration store-write must hold the COMMIT LOCK (readers hold no
            // lock; a lost COW update would leave the table de-elided WITH a stale store). The
            // helper detects mid-commit internal reads (matview refresh) and skips the
            // self-deadlocking re-acquisition.
            self.rehydrate_elided_serialized(&select.table)
                .map_err(ExecuteError::Engine)?;
        }
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
            columns: Arc::new(vec![RelationalColumn {
                id: 0,
                table_oid: function.oid,
                attnum: 1,
                name: function.name.clone(),
                ty: function.return_type,
                domain: None,
                default: None,
                type_oid: function.return_type.postgres_oid(),
                type_size: function.return_type.type_size(),
            }]),
            rows: (vec![vec![value]]).into(),
            planned_target: DeviceTarget::Cpu,
            executed_target: DeviceTarget::Cpu,
            fallback_reason: Some(FallbackReason::NotGpuEligible),
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
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
        // Fetch the descriptor + host rows as ONE atomic entry (a single `load()`), so the metadata
        // checked here and the rows materialized below come from the same residency generation.
        let entry = self
            .relational_residency_entry(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident snapshot",
                    table.name
                )))
            })?;
        let snapshot = &entry.descriptor;
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

        // The host-row materialization is the other (Arc-shared) half of the SAME entry fetched above
        // -- one consistent generation, no second load.
        let result = MvccReadResult {
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            rows: entry
                .host_rows_iter()
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
            .shard_device_memory
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
            // S10c slice 2a: the 8 MULTI-PARTITION resident shapes route to the `&Select`->general bridge
            // (single-GPU). It RECOMPACTS the table's shard buffers ON-DEVICE (device-to-device copies)
            // into ONE unified int4 SoA buffer, then runs the general resident-Expr executor ONCE over it --
            // so COUNT/SUM/MIN/MAX/AVG and projection are all computed on the device with the host FULLY out
            // (no per-shard host combine; AVG is a single on-device quotient). Retired the 8
            // `execute_relational_sharded_*_with_resident_device_memory_probe` methods (slice 1) and then
            // the per-shard combine (slice 2a). Byte-identical to the probes on non-NULL data; the
            // empty-set aggregates take the same placeholders via a COUNT precheck.
            "sharded_count_all"
            | "sharded_int4_scalar_aggregate"
            // THE FLIP audit F1: filtered/range shapes the sharded bridge serves via the general
            // executor (pre-fix they fell to the CPU host scan on the now-default sharded layout).
            | "sharded_int4_equality_count"
            | "sharded_int4_range_count"
            | "sharded_int4_filtered_scalar_aggregate"
            | "sharded_int4_between_scalar_aggregate"
            | "sharded_int4_projection"
            // R-ver: the UNFILTERED int4 projection (SELECT <cols> FROM t, no WHERE). The general
            // executor runs its plain-projection path with predicate = None and threads the SV3b
            // `deleted_by` visibility conjunct for versioned shards — so this read stays on the
            // resident route instead of falling to the CPU-pinned path (which rehydrates + de-elides).
            | "sharded_int4_projection_all"
            | "sharded_int4_composite_equality_multi_column_projection"
            | "sharded_int4_equality_projection"
            | "sharded_int4_equality_multi_column_projection"
            | "sharded_int4_equality_sum"
            | "sharded_int4_between_avg"
            | "sharded_int4_filtered_min"
            | "sharded_int4_filtered_avg"
            | "sharded_int4_filtered_max"
            // S10c slice 2b: sharded DISTINCT / GROUP BY / ORDER-BY projection. The unified
            // recompacted buffer holds the WHOLE table, so these dedup/group/sort/window shapes are
            // CORRECT over it; the sharded bridge dispatches each to the grouped/distinct sub-bridge
            // with the unified source injected (no new device/kernel code).
            | "sharded_int4_distinct_projection"
            | "sharded_int4_filtered_distinct_projection"
            | "sharded_int4_grouped_aggregate"
            | "sharded_int4_filtered_grouped_aggregate"
            | "sharded_int4_ordered_projection" => {
                self.execute_resident_sharded_via_general(select)
            }
            // S10a: a text-prefix `COUNT(*)` (`SELECT COUNT(*) ... WHERE col LIKE 'al%'`) routes to the
            // `&Select`->general bridge as a CountAll + a reconstructed `LIKE` predicate. The bridge rebuilds
            // the faithful `LIKE '<prefix>%'` pattern from the bare `LikePrefix` bound filter (the parser
            // strips the trailing `%`); the general WHERE runs it ON-DEVICE -- a nullable-text LIKE via the
            // new `TextLikeMask` mask-VM step, a non-null LIKE via the standalone `expr_text_like_scalar_filter`
            // fast path. Byte-identical to the retired probe on non-NULL data; PG-correct on NULLs (a NULL is
            // UNKNOWN under LIKE 3VL so it is excluded, whereas the NULL-blind probe's empty placeholder span
            // matched `LIKE '%'`). Covers text + CTAS + view uniformly.
            "text_prefix_like_count" => self.execute_resident_grouped_via_general(select, None, None),
            // S10a: an int4 COUNT(*) with multiple filter groups (OR of AND-groups, all int4) routes to the
            // `&Select`->general bridge as a CountAll + the rebuilt int4 DNF predicate -- byte-identical to the
            // retired probe on non-NULL data; PG-correct on NULLs (a NULL fails the predicate via 3VL instead
            // of the probe's phantom Int4(0)).
            "int4_filter_group_count" => self.execute_resident_grouped_via_general(select, None, None),
            // S8: grouped int4 aggregates route to the general on-device executor via the
            // `&Select`->general BRIDGE (it does ORDER BY / HAVING / LIMIT ON-DEVICE), retiring the
            // legacy resident-probe grouped methods whose `!gpu_ordered` branch host-finalized
            // sort/HAVING/LIMIT. Because the dispatch sees `&Select`, this covers the text entry AND
            // CTAS AND view/matview uniformly.
            "int4_grouped_aggregate" | "int4_filtered_grouped_aggregate" => {
                self.execute_resident_grouped_via_general(select, None, None)
            }
            // S10a (projection batch): the non-grouped int4 projection shapes (a single-column range-filtered
            // projection; equality; multi-column / composite-AND equality; and a mixed text+int4 projection)
            // route to the SAME `&Select`->general bridge. The binding executor runs the plain-projection path
            // (WHERE predicate VM + on-device column/text gather, S1) -- byte-identical to the retired probes on
            // non-NULL data; NULL/empty results become PG-correct (a NULL fails the filter via 3VL instead of
            // the probe's phantom Int4(0)). Every FILTER is int4 (text only in the projection), so the int4-only
            // predicate builder suffices. Covers text + CTAS + view uniformly.
            "int4_projection"
            // R-ver: the UNFILTERED int4 projection on a NON-sharded snapshot-resident table (the
            // sharded twin is `sharded_int4_projection_all` above). The general executor's
            // plain-projection path (predicate = None -> every row) serves it, so an unfiltered
            // scan of a resident table stays on the resident route instead of the CPU-pinned path.
            | "int4_projection_all"
            | "int4_equality_projection"
            | "int4_equality_multi_column_projection"
            | "int4_composite_equality_multi_column_projection"
            | "int4_equality_mixed_column_projection" => {
                self.execute_resident_grouped_via_general(select, None, None)
            }
            // S10a: a single-int4-column ordered projection (`SELECT a FROM t WHERE a <range> ORDER BY a
            // LIMIT n`) routes to the SAME `&Select`->general bridge as the grouped shapes. For a
            // non-grouped select the bridge builds EMPTY group keys, so the binding executor runs the
            // plain-projection path (WHERE predicate VM + GPU sort + LIMIT/OFFSET window, S1/S2.1/S4) --
            // byte-identical to the legacy `int4_ordered_projection` probe (the projected column IS the
            // sort key, so tied values are identical output rows). Covers text + CTAS + view uniformly.
            "int4_ordered_projection" => self.execute_resident_grouped_via_general(select, None, None),
            // S10b: a single-int4-column SELECT DISTINCT routes through the `&Select`->general DISTINCT
            // bridge (`SELECT DISTINCT a` == `GROUP BY a` `COUNT(*)` with the count dropped) -- one row per
            // distinct key ON THE DEVICE, retiring the probes that deduped on a HOST `BTreeSet` (a §1
            // violation relabeled as GPU). Covers text + CTAS + view uniformly. NULL/no-ORDER-BY results
            // become PG-correct (NULL group kept, not a phantom 0; default order key-ASC, deterministic).
            "int4_distinct_projection" | "int4_filtered_distinct_projection" => {
                self.execute_resident_distinct_via_general(select, None, None)
            }
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
