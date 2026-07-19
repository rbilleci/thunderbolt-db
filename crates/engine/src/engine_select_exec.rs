//! Relational SELECT entry + dispatch (P0 §9.6 decomposition, behavior-preserving):
//! a focused `impl Engine` block for the public SELECT entry points
//! (execute_relational_select, _text, _instrumented, and the retired-host decline seam,
//! execute_relational_function) and the device-route dispatch variants
//! (_with_cuda_driver_probe, _with_resident_snapshot_probe, _with_resident_route)
//! that pick a GPU path or fail loudly and hand off to the resident-probe executors.

use super::*;
use crate::engine_expr::ResidentExecSource;

/// A non-grouped projection (no aggregate / GROUP BY / HAVING) with an ORDER BY whose keys are ALL
/// i64-sortable int columns (int2/int4/int8/date/timestamp) -- one OR several keys (e.g.
/// `ORDER BY a ASC, b DESC, c`). Such a query is routed to the general GPU Expr executor, which sorts
/// the surviving rows on the GPU (multi-key bitonic) -- the charter-native path, retiring the
/// enumerated ordered-projection shape for this case. A SINGLE text key also routes here (the byte-wise
/// GPU text sort), and a MIXED int+text tuple uses the heterogeneous comparator. Other shapes --
/// numeric/uuid keys, expressions -- stay on the existing path transitionally; the enumerated route
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
        if self.is_commit_path_poisoned() {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "commit path is wedged; restart recovery required".to_string(),
            )));
        }
        self.execute_relational_select_instrumented(select, || {})
    }

    /// Execute a SELECT against the generation bundle retained by explicit `txn_id`. The scoped
    /// context is connection-neutral and panic-safe: deep catalog, MVCC, and resident GPU lookups
    /// resolve through the retained bundle for this call, then the previous thread context is
    /// restored. Transaction control remains the sole owner of the bundle's lifetime.
    pub fn execute_relational_select_in_transaction(
        &self,
        txn_id: TxnId,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_relational_select_in_transaction_with_hook(txn_id, select, || {})
    }

    #[cfg(test)]
    pub(crate) fn execute_relational_select_in_transaction_instrumented(
        &self,
        txn_id: TxnId,
        select: &Select,
        on_statement_locked: impl FnOnce(),
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_relational_select_in_transaction_with_hook(txn_id, select, on_statement_locked)
    }

    fn execute_relational_select_in_transaction_with_hook(
        &self,
        txn_id: TxnId,
        select: &Select,
        on_statement_locked: impl FnOnce(),
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        let _statement = snapshot
            .statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        on_statement_locked();
        let _scope = self.enter_transaction_read(Arc::clone(&snapshot));
        self.execute_relational_select(select)
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
    /// no host re-execution of an arithmetic predicate.
    /// The strict `parse_command` entry the legacy pgwire server uses is untouched (dual-entry).
    pub fn execute_relational_select_text(
        &self,
        text: &str,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        if self.is_commit_path_poisoned() {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "commit path is wedged; restart recovery required".to_string(),
            )));
        }
        match parse_command_allowing_catalog(text) {
            Ok(Command::Select(select)) => {
                // A non-grouped ORDER BY over an i64-sortable int key is a charter-native GPU sort: run
                // it through the general Expr executor (which sorts on the GPU via the bitonic sort),
                // NOT the enumerated ordered-projection shape (Charter rule 2).
                if let Some(table) = self.relational_catalog_table(&select.table) {
                    // Only route to the general GPU path when the table is GPU-resident; a non-resident
                    // table reaches the strict dispatcher below and fails loudly if no GPU route accepts.
                    // SLICE B: SHARD-resident tables qualify too — the general
                    // path now recompacts them into the unified exec source, so a sortable projection
                    // gets the SAME NULL-correct on-device sort instead of falling through the rejected
                    // shape route. The shard arm
                    // requires a PURELY int4-section table (int4/int2/date): the unified exec source
                    // gathers only the int4 sections (+ null bitmaps), so a text/int8/numeric/bool
                    // column reference hard-errors on the no-fallback general path. Mixed-type sharded
                    // tables therefore require every referenced section in the unified source.
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
                            // numeric/uuid table remains on the GPU path.
                            // TYPE-COVERAGE #14 (bool): the bool bitmap section is gathered into the
                            // unified source too, so a bool column no longer forces an ORDER BY over
                            // a bool-bearing elided table to decline. Bool is carried/projected here,
                            // not a sort key.
                            matches!(
                                c.ty,
                                SqlType::Int4
                                    | SqlType::Int2
                                    | SqlType::Date
                                    | SqlType::Int8
                                    | SqlType::Timestamp
                                    | SqlType::Numeric { .. }
                                    | SqlType::Uuid
                                    | SqlType::Bool
                            )
                        }) && self
                            .read_residency_shards()
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
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        // Autocommit reads retain one immutable catalog+device generation for the full statement,
        // just like an explicit transaction retains one for its lifetime. Capture is serialized
        // only with the publication cut; kernel execution runs after the commit lock is released,
        // so readers remain mutually concurrent. A shape-changing DDL can then invalidate the
        // current maps without retiring or mismatching the generation this statement owns.
        let statement_snapshot = if self.current_transaction_read_snapshot().is_none()
            && !self.mvcc_read_skips_leader_check()
        {
            self.transition_oversized_device_table_to_streaming_repair(&select.table)
                .map_err(ExecuteError::Engine)?;
            let commit = self.commit_state();
            self.ensure_commit_path_available()
                .map_err(ExecuteError::Engine)?;
            let snapshot = self.capture_statement_snapshot(self.committed_seq());
            drop(commit);
            Some(snapshot)
        } else {
            None
        };
        let _statement_scope =
            statement_snapshot.map(|snapshot| self.enter_transaction_read(snapshot));
        // Multi-key ORDER BY (`ORDER BY a, b, ...`) is GPU-only: it runs on the general Expr executor's
        // bitonic-sort path, routed in `execute_relational_select_text` when every key is an i64-sortable
        // base column on a GPU-resident table. This enumerated path has no multi-key sort and must
        // NOT silently sort by the first key only -- so reject it. A multi-key sort reaching here means a
        // non-routable key (text/numeric/uuid/expression) or a non-resident table: a clean error, never a
        // wrong (first-key-only) result.
        if select.order_by.len() > 1 {
            if self.table_is_gpu_resident(&select.table) {
                on_pinned();
                return self.execute_resident_select_via_general(select);
            }
            if let Some(result) = self.try_streaming_select(select) {
                on_pinned();
                return result;
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "multi-key ORDER BY runs on the GPU sort path (every key must be an i64-sortable \
                 base column on a GPU-resident table)"
                    .to_string(),
            )));
        }
        // PART B test seam. Device routes fire the hook after retaining the statement generation; if
        // no route accepts, the retired-host decline seam fires it before returning the loud error so
        // a barrier-based concurrency test cannot deadlock. Views/materialized views are resolved at
        // the same retained boundary.
        //
        // The read pins ONE `committed_seq` boundary and resolves the catalog as-of it. It does NOT
        // register an active snapshot (reads stay OFF the `active_snapshots` mutex — true lock-free):
        // the catalog ring's COUNT floor (`MIN_RETAINED_CATALOG_GENERATIONS`) guarantees this read's
        // generation is still present even if a flurry of concurrent DDLs commit while the statement
        // runs, so `catalog_as_of(s)` never falls back to a too-new generation.
        let s = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(s);
        if let Some(view) = catalog.relational_views.get(&select.table).cloned() {
            if !select_is_plain_view_scan(select) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "only plain SELECT * FROM view is supported for views".to_string(),
                )));
            }
            on_pinned();
            return self.execute_relational_view_at(&view.query, s);
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
            let table = RelationalTable {
                schema: view.schema,
                name: view.name,
                oid: view.oid,
                columns: view.columns,
                indexes: Vec::new(),
                check_constraints: Vec::new(),
                foreign_keys: Vec::new(),
                acl: BTreeMap::new(),
            };
            return self.execute_transient_rows_via_general(select, table, view.rows, s);
        }
        // Phase-3 M2: a SELECT against a synthesized pg_catalog/information_schema relation
        // runs through the SAME bind -> filter -> project -> order/limit core as a user
        // table, over rows projected from this pinned catalog generation (MVCC-consistent).
        // Resolved AFTER user tables/views so a real relation always shadows a catalog name.
        if !catalog.relational_catalog.contains_key(&select.table) {
            if let Some((catalog_table, catalog_rows)) =
                synthesize_catalog_relation(&select.table, &catalog)
            {
                return self.execute_transient_rows_via_general(
                    select,
                    catalog_table,
                    catalog_rows,
                    s,
                );
            }
        }
        // A table that existed but had never published a data or residency generation at BEGIN is
        // exactly empty in the transaction snapshot. If a later first INSERT admits it, using the
        // current GPU buffer would pair old visibility with new bytes. Execute the empty generation
        // as a zero-row transient GPU relation instead: fail-free, device-native, and independent of
        // the later allocation. Non-empty captured tables continue through their retained resident
        // source below (or fail loud if that old generation had no GPU representation).
        if let Some(snapshot) = self.current_transaction_read_snapshot() {
            let captured_empty_without_residency = snapshot.boundary == s
                && !snapshot.table_versions.contains_key(&select.table)
                && !snapshot.resident_snapshots.contains_key(&select.table)
                && snapshot
                    .resident_shards
                    .get(&select.table)
                    .is_none_or(|shards| shards.is_empty());
            if captured_empty_without_residency {
                if let Some(table) = snapshot
                    .catalog
                    .relational_catalog
                    .get(&select.table)
                    .cloned()
                {
                    on_pinned();
                    return self.execute_transient_rows_via_general(select, table, Vec::new(), s);
                }
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
                // probe error — NOT a genuine device failure. The route declines through the common
                // fail-loud boundary. Any OTHER error (a real
                // GPU/CUDA failure, a bind error, etc.) propagates unchanged so we never mask it.
                Err(err) if err.is_residency_invalidated() => {
                    return self.execute_relational_select_cpu_pinned(select);
                }
                Err(err) => return Err(err),
            }
        }
        // CPU-ENGINE RETIREMENT (ADR-006): the SPECIALIZED resident route declined this shape (a
        // wider-type filtered projection / aggregate / GROUP BY / DISTINCT / ORDER BY / OFFSET the
        // enumerated matcher does not recognize). Route it to the GENERAL GPU Expr executor:
        // `execute_resident_grouped_via_general`
        // binds the `&Select`, rebuilds the WHERE as a `ResidentExpr`, and runs projection / aggregate /
        // GROUP BY / DISTINCT / ORDER BY / LIMIT / OFFSET ON THE DEVICE for every retained type, over the
        // whole-table buffer or the unified shard source (versioned-aware). It is GATED on residency: the
        // general path has no host fallback and hard-errors on a non-resident table, so a table with no
        // snapshot/shards skips it and ultimately fails loudly if streaming also declines.
        // The general executor ERRORS (never mis-answers) on a shape it cannot express, so on ANY error we
        // fall through to the fail-loud boundary. `general_read_fallback_hits` proves the on-device path
        // fired (a silent de-elide would pass output equality while abandoning the elision).
        if self.table_is_gpu_resident(&select.table) {
            on_pinned();
            match self.execute_resident_select_via_general(select) {
                Ok(result) => {
                    self.read_state
                        .residency
                        .general_read_fallback_hits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(result);
                }
                // Residency invalidated mid-statement, OR a shape the general executor cannot express:
                // decline loudly. The hook already fired above, so this uses the no-op-hook entry.
                Err(_) => return self.execute_relational_select_cpu_pinned(select),
            }
        }
        // STRATA S-E.1/S-E.2 (streaming executor, ADR-012): for an over-VRAM read, try the OUT-OF-CORE
        // streaming fold before reaching the fail-loud boundary — chunk the table's visible rows to the residency
        // budget and run each chunk ON THE DEVICE: a scalar reduction (COUNT(*)/SUM/MIN/MAX) combines
        // partials in one final device pass; a filter/project CONCATs survivors with LIMIT/OFFSET as
        // cross-chunk windowing (a satisfied LIMIT stops the scan early). Never all shards resident at
        // once. Gated on a configured per-GPU budget + a foldable shape; `None` = not applicable -> the
        // caller reaches the fail-loud boundary. A fold that cannot express a shape declines rather
        // than returning a host-computed result.
        if let Some(result) = self.try_streaming_select(select) {
            on_pinned();
            return result;
        }
        self.execute_relational_select_cpu_pinned_instrumented(select, on_pinned)
    }

    fn execute_relational_view_at(
        &self,
        select: &Select,
        copin_s: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let catalog = self.read_catalog_as_of(copin_s);
        if let Some(view) = catalog.relational_views.get(&select.table).cloned() {
            if !select_is_plain_view_scan(select) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "only plain SELECT * FROM view is supported for views".to_string(),
                )));
            }
            return self.execute_relational_view_at(&view.query, copin_s);
        }
        if let Some(result) = self.try_streaming_select_at(select, copin_s) {
            return result;
        }
        if self.table_is_gpu_resident(&select.table) && !select.distinct {
            let (table, bound, _) = self.bind_relational_select_at(select, copin_s)?;
            return self.execute_resident_grouped_via_general_with_binding(
                select, &table, bound, copin_s, None, None,
            );
        }
        self.execute_relational_select_cpu_pinned_at(select, copin_s, || {})
    }

    /// CPU-ENGINE RETIREMENT (ADR-006): is `table` GPU-resident right now — i.e. does it have a published
    /// single-buffer residency snapshot OR at least one live shard? The general Expr executor has no CPU
    /// fallback and hard-errors on a non-resident table, so the declined-route fallback gates on this
    /// (mirrors the `execute_relational_select_text` residency gate). A racing invalidation between this
    /// check and the executor's own load surfaces as a loud `is_residency_invalidated` route decline.
    pub(crate) fn table_is_gpu_resident(&self, table: &str) -> bool {
        self.relational_residency_snapshot(table).is_some()
            || self
                .read_residency_shards()
                .get(table)
                .is_some_and(|shards| !shards.is_empty())
    }

    pub(crate) fn execute_transient_rows_via_general(
        &self,
        select: &Select,
        table: RelationalTable,
        rows: Vec<Vec<SqlValue>>,
        copin_s: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let bound = bind_relational_select(&table, select)?;
        let row_count = rows.len() as u64;
        let (snapshot, memory) = self.build_transient_relation_residency(&table, &rows)?;
        let source = ResidentExecSource {
            descriptor: Arc::new(snapshot),
            device_memory: Arc::new(memory),
            row_count,
        };
        self.execute_resident_grouped_via_general_with_binding(
            select,
            &table,
            bound,
            copin_s,
            Some(&source),
            None,
        )
    }

    /// Project a DML statement's final resolved row images through the general GPU result path.
    /// Callers invoke this before any durable side effect; concurrent autocommit passes the
    /// under-lock re-resolved delta, so returned rows cannot describe a stale off-lock prepare.
    pub(crate) fn project_dml_returning(
        &self,
        command: &Command,
        delta: &WriteDelta,
        boundary: Index,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let (table_name, returning) = match command {
            Command::Insert(insert) => (&insert.table, &insert.returning),
            Command::Update(update) => (&update.table, &update.returning),
            Command::Delete(delete) => (&delete.table, &delete.returning),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "DML RETURNING projection received a non-DML command".to_string(),
                )))
            }
        };
        if returning.is_empty() {
            return Ok(None);
        }
        let table = self
            .catalog_snapshot()
            .relational_catalog
            .get(table_name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table_name}\" does not exist for RETURNING"
                )))
            })?;
        let rows = match &delta.mutation {
            PreparedMutation::Insert {
                table,
                inserted_rows,
                ..
            } if table == table_name => inserted_rows.iter().map(|(_, row)| row.clone()).collect(),
            PreparedMutation::Update {
                table, installs, ..
            } if table == table_name => installs.iter().map(|(_, _, row)| row.clone()).collect(),
            PreparedMutation::Delete {
                table,
                deleted_rows,
                ..
            } if table == table_name => deleted_rows.clone(),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "DML RETURNING command/delta shape mismatch".to_string(),
                )))
            }
        };
        let select = Select {
            table: table_name.clone(),
            distinct: false,
            projection: SelectProjection::Columns(returning.clone()),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        self.execute_transient_rows_via_general(&select, table, rows, boundary)
            .map(Some)
    }

    /// Evaluate a non-resident relational fixture without claiming an execution route.
    ///
    /// Device- or chunk-authoritative relations are rejected so this specification seam cannot
    /// become a host fallback for product state whose rows live outside the tuple store.
    #[cfg(test)]
    pub(crate) fn evaluate_relational_select_specification(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectSpecificationResult, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if self.table_device_authoritative(&select.table)
            || self.table_chunk_authoritative(&select.table).is_some()
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relational specification requires tuple-store authority for relation \"{}\"",
                select.table
            ))));
        }

        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let pin = self.pin_relational_read_at(&select.table, copin_s);
        let (query, access_path) =
            self.relational_select_mvcc_query(select, &table, &bound, &pin)?;
        let specification = self.evaluate_mvcc_query_specification_on_pin(&pin, &query)?;
        self.finalize_relational_select_specification(
            select,
            table,
            bound,
            access_path,
            specification.rows,
        )
    }

    /// Compatibility name for the retired host-dispatch boundary. No relational work executes here:
    /// every caller receives the same fail-loud GPU-required error without fallback telemetry.
    pub(crate) fn execute_relational_select_cpu_pinned(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_relational_select_cpu_pinned_instrumented(select, || {})
    }

    /// Retired host-dispatch seam. The hook is fired so concurrency tests cannot deadlock when a
    /// route declines, then the read fails loudly under the same policy in every build profile.
    fn execute_relational_select_cpu_pinned_instrumented(
        &self,
        select: &Select,
        on_bound_before_pin: impl FnOnce(),
    ) -> Result<RelationalSelectResult, ExecuteError> {
        on_bound_before_pin();
        Err(self.gpu_read_required(select, "no GPU route accepted the statement"))
    }

    fn execute_relational_select_cpu_pinned_at(
        &self,
        select: &Select,
        _statement_copin_s: Index,
        on_bound_before_pin: impl FnOnce(),
    ) -> Result<RelationalSelectResult, ExecuteError> {
        on_bound_before_pin();
        Err(self.gpu_read_required(select, "the pinned GPU route became unavailable"))
    }

    fn gpu_read_required(&self, select: &Select, detail: &str) -> ExecuteError {
        ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "GPU execution is required for SELECT on relation \"{}\": {detail}",
            select.table
        )))
    }

    pub fn execute_relational_function(
        &self,
        call: &SelectFunction,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        // Lock-free read path (Stage 2 — blocker #1): pin the catalog snapshot for the function lookup.
        let catalog = self.catalog_snapshot();
        let Some(function) = catalog.relational_functions.get(&call.name) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "function \"{}\" does not exist",
                call.name
            ))));
        };
        let value = parse_bounded_sql_function_body(&function.body, function.return_type)?;
        // A bounded SQL function body is a typed literal. Parsing it is control-plane work;
        // returning it is relational execution. Upload the literal as a one-row transient
        // relation and run the same device projection used by catalog/materialized-view rows.
        let column = RelationalColumn {
            id: 0,
            table_oid: function.oid,
            attnum: 1,
            name: function.name.clone(),
            ty: function.return_type,
            domain: None,
            default: None,
            type_oid: function.return_type.postgres_oid(),
            type_size: function.return_type.type_size(),
        };
        let table = RelationalTable {
            schema: function.schema.clone(),
            name: format!("__gpu_function_{}", function.oid),
            oid: function.oid,
            columns: vec![column],
            indexes: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
            acl: BTreeMap::new(),
        };
        let select = Select {
            table: table.name.clone(),
            distinct: false,
            projection: SelectProjection::All,
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        self.execute_transient_rows_via_general(
            &select,
            table,
            vec![vec![value]],
            self.committed_seq(),
        )
    }

    pub fn execute_relational_select_with_cuda_driver_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        if self.is_commit_path_poisoned() {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "commit path is wedged; restart recovery required".to_string(),
            )));
        }
        if self.table_is_gpu_resident(&select.table) {
            return self.execute_resident_select_via_general(select);
        }
        if let Some(result) = self.try_streaming_select(select) {
            return result;
        }
        Err(self.gpu_read_required(
            select,
            "the legacy CUDA-probe entry has no resident or streaming source",
        ))
    }

    pub fn execute_relational_select_with_resident_route(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        if self.is_commit_path_poisoned() {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "commit path is wedged; restart recovery required".to_string(),
            )));
        }
        let decision = self.plan_relational_resident_route(select);
        if !decision.accepted {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident route rejected: {}",
                decision.reason
            ))));
        }

        let before_metrics = self.metrics.snapshot();
        if let Some(device_memory) = self.read_resident_device_memory(&decision.table) {
            // Make this allocation's CUDA context current on the calling thread so a
            // concurrent reader (not the context's creator) can launch — without it the
            // kernel fails with INVALID_CONTEXT (P1-M3 step 3c / gate 2).
            let _ = device_memory.set_current_context();
            device_memory.clear_last_kernel_event_elapsed_us();
        }
        if let Some(shards) = self.read_residency_shards().get(&decision.table) {
            for device_memory in shards
                .iter()
                .filter_map(|shard| shard.device_memory.as_ref())
            {
                let _ = device_memory.set_current_context();
                device_memory.clear_last_kernel_event_elapsed_us();
            }
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
            // resident route instead of declining through the retired-host boundary.
            | "sharded_int4_projection_all"
            | "sharded_int4_composite_equality_multi_column_projection"
            | "sharded_int4_equality_projection"
            | "sharded_int4_equality_multi_column_projection"
            | "sharded_int4_equality_mixed_column_projection"
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
            // scan of a resident table stays on the resident route instead of declining.
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
