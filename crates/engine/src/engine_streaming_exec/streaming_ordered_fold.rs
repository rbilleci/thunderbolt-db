//! Streaming ORDER BY chunking, compaction, and final device sort/window.

use super::*;

impl Engine {
    /// STRATA S-E.4 — the ORDER BY fold. THE SORT IS ALWAYS ON THE DEVICE (the charter forbids a host
    /// k-way merge): TOP-N (`ORDER BY k LIMIT n [OFFSET m]`) runs each chunk's projection through the
    /// device sort + device window — a chunk's local top-(m+n) is its ONLY possible contribution to the
    /// global window — concats the runs (control plane), COMPACTS the accumulator by device re-sort +
    /// re-window whenever it outgrows the chunk target, and finishes with ONE device sort + the REAL
    /// window over the synthesized runs relation. UNBOUNDED ORDER BY skips the per-chunk sort (plain
    /// device filter/project per chunk — a final re-sort makes chunk runs pointless) and defers honestly
    /// mid-scan if the survivor set outgrows the budget (its final device sort could not fit).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_streaming_ordered_fold(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        top_n: Option<usize>,
        gpu_id: u16,
        budget: u64,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let chunk_target_bytes = (budget / 2).max(1);
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        // The per-chunk select: top-N keeps the ORDER BY + a device LIMIT of the window bound (the run
        // truncation); unbounded strips the ORDER BY (plain filter/project — sorted once at the end).
        let mut chunk_select = select.clone();
        chunk_select.offset = None;
        match top_n {
            Some(bound_n) => chunk_select.limit = Some(bound_n),
            None => {
                chunk_select.order_by = Vec::new();
                chunk_select.limit = None;
            }
        }
        let Ok(chunk_bound) = bind_relational_select(table, &chunk_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        // The synthesized RUNS relation: the projected columns by name/type (the classifier guarantees
        // all sort keys are among them), consumed by the FINAL device sort + window pass.
        let runs_column_refs: Vec<(&str, SqlType)> = bound
            .selected_columns
            .iter()
            .map(|column| (column.name.as_str(), column.ty))
            .collect();
        let partial_types: Vec<SqlType> = runs_column_refs.iter().map(|(_, ty)| *ty).collect();
        let runs_table = catalog_relation_table(&table.schema, "__stream_runs", &runs_column_refs);
        let final_select = Select {
            table: runs_table.name.clone(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Columns(
                runs_column_refs
                    .iter()
                    .map(|(name, _)| (*name).to_string())
                    .collect(),
            ),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: select.order_by.clone(),
            limit: select.limit,
            offset: select.offset,
        };
        let Ok(final_bound) = bind_relational_select(&runs_table, &final_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };
        // The COMPACTION select: same device sort but windowed to the top-N bound only (OFFSET stays 0 —
        // the real window slices once, at the end).
        let compact_select = Select {
            limit: top_n,
            offset: None,
            ..final_select.clone()
        };
        let Ok(compact_bound) = bind_relational_select(&runs_table, &compact_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        let mut runs_acc: Vec<Vec<SqlValue>> = Vec::new();
        let mut runs_bytes: u64 = 0;
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        // 6c-1: the current chunk's inclusive TupleId range ((1,0) = empty sentinel).
        let mut chunk_range: (u64, u64) = (1, 0);
        let mut chunks_run: u64 = 0;
        // S-E.5: the one-chunk lookahead (drained BEFORE any accumulator sort upload).
        let mut staged: Option<StagedChunk> = None;
        // S-E.6: cold-tier replay on a hit; a miss scans + captures for later reads.
        let cold = self.load_streaming_cold(&select.table, table, chunk_target_bytes, copin_s);
        let mut capture: Option<ColdCacheBuilder> = None;

        let prefix = relational_key_prefix(&select.table);
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        if let Some(cold) = &cold {
            for chunk in &cold.chunks {
                // P4-3 born gate: a chunk born after this reader's boundary is invisible to it.
                if chunk.payload_copin_s > copin_s {
                    continue;
                }
                let Ok(next) = self.stage_cold_chunk(chunk, copin_s) else {
                    // A failed replay (e.g. a bad spill read) evicts the entry so the next read
                    // rebuilds instead of defer-thrashing (audit LOW).
                    self.evict_streaming_cold(&select.table);
                    return self.execute_relational_select_cpu_pinned(select);
                };
                if let Some(prev) = staged.replace(next) {
                    let Ok((src, chunk_vis)) = prev.ready() else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    match self.ordered_streaming_chunk(
                        &chunk_select,
                        table,
                        &chunk_bound,
                        predicate,
                        copin_s,
                        &src,
                        chunk_vis,
                        &partial_types,
                        &mut runs_acc,
                        &mut runs_bytes,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                            return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    chunks_run += 1;
                }
                // The loop's compaction/defer checks, drain-first (as the scan loop).
                if runs_bytes >= chunk_target_bytes {
                    if let Some(prev) = staged.take() {
                        let Ok((src, chunk_vis)) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.ordered_streaming_chunk(
                            &chunk_select,
                            table,
                            &chunk_bound,
                            predicate,
                            copin_s,
                            &src,
                            chunk_vis,
                            &partial_types,
                            &mut runs_acc,
                            &mut runs_bytes,
                        ) {
                            ChunkOutcome::Ok => {}
                            ChunkOutcome::Defer => {
                                return self.execute_relational_select_cpu_pinned(select)
                            }
                            ChunkOutcome::Hard(err) => return Err(err),
                        }
                        chunks_run += 1;
                    }
                }
                if runs_bytes >= chunk_target_bytes {
                    match top_n {
                        Some(_) => {
                            match self.sort_streaming_runs(
                                &compact_select,
                                &runs_table,
                                &compact_bound,
                                copin_s,
                                &partial_types,
                                &mut runs_acc,
                                &mut runs_bytes,
                                budget,
                                None,
                            ) {
                                ChunkOutcome::Ok => {}
                                ChunkOutcome::Defer => {
                                    return self.execute_relational_select_cpu_pinned(select)
                                }
                                ChunkOutcome::Hard(err) => return Err(err),
                            }
                            if runs_bytes > budget {
                                return self.execute_relational_select_cpu_pinned(select);
                            }
                        }
                        None => {
                            if runs_bytes > budget {
                                return self.execute_relational_select_cpu_pinned(select);
                            }
                        }
                    }
                }
            }
        } else {
            let table_rows = self.read_state.mvcc.table_rows(&select.table);
            capture = Some(ColdCacheBuilder {
                generation: table_rows.generation_payload(),
                build_copin_s: copin_s,
                column_signature: table
                    .columns
                    .iter()
                    .map(|c| (c.name.clone(), c.ty))
                    .collect(),
                chunk_target_bytes,
                total_payload_bytes: 0,
                chunks: Vec::new(),
                spill: None,
                poisoned: false,
            });
            let mut cursor = table_rows.store().seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let decoded = decode_relational_row(&tuple.value, &table.columns)?;
                chunk_bytes =
                    chunk_bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
                chunk_range = if chunk_rows.is_empty() {
                    (tuple.tuple_id, tuple.tuple_id)
                } else {
                    (chunk_range.0, tuple.tuple_id)
                };
                chunk_rows.push(decoded);
                if chunk_bytes >= chunk_target_bytes {
                    let Ok(next) = self.stage_streaming_chunk(
                        table,
                        &chunk_rows,
                        chunk_range,
                        &mut capture,
                        gpu_id,
                    ) else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    if let Some(prev) = staged.replace(next) {
                        let Ok((src, chunk_vis)) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.ordered_streaming_chunk(
                            &chunk_select,
                            table,
                            &chunk_bound,
                            predicate,
                            copin_s,
                            &src,
                            chunk_vis,
                            &partial_types,
                            &mut runs_acc,
                            &mut runs_bytes,
                        ) {
                            ChunkOutcome::Ok => {}
                            ChunkOutcome::Defer => {
                                return self.execute_relational_select_cpu_pinned(select)
                            }
                            ChunkOutcome::Hard(err) => return Err(err),
                        }
                        chunks_run += 1;
                    }
                    chunk_rows.clear();
                    chunk_bytes = 0;
                    chunk_range = (1, 0);
                    // Drain the in-flight chunk FIRST so an accumulator sort upload never overlaps it.
                    if runs_bytes >= chunk_target_bytes {
                        if let Some(prev) = staged.take() {
                            let Ok((src, chunk_vis)) = prev.ready() else {
                                return self.execute_relational_select_cpu_pinned(select);
                            };
                            match self.ordered_streaming_chunk(
                                &chunk_select,
                                table,
                                &chunk_bound,
                                predicate,
                                copin_s,
                                &src,
                                chunk_vis,
                                &partial_types,
                                &mut runs_acc,
                                &mut runs_bytes,
                            ) {
                                ChunkOutcome::Ok => {}
                                ChunkOutcome::Defer => {
                                    return self.execute_relational_select_cpu_pinned(select)
                                }
                                ChunkOutcome::Hard(err) => return Err(err),
                            }
                            chunks_run += 1;
                        }
                    }
                    if runs_bytes >= chunk_target_bytes {
                        match top_n {
                            // TOP-N compaction: device re-sort + truncate to the window bound.
                            Some(_) => {
                                match self.sort_streaming_runs(
                                    &compact_select,
                                    &runs_table,
                                    &compact_bound,
                                    copin_s,
                                    &partial_types,
                                    &mut runs_acc,
                                    &mut runs_bytes,
                                    budget,
                                    None,
                                ) {
                                    ChunkOutcome::Ok => {}
                                    ChunkOutcome::Defer => {
                                        return self.execute_relational_select_cpu_pinned(select)
                                    }
                                    ChunkOutcome::Hard(err) => return Err(err),
                                }
                                // A window bound too large to compact under the budget cannot final-
                                // sort either: defer (mirrors the grouped over-cardinality defer).
                                if runs_bytes > budget {
                                    return self.execute_relational_select_cpu_pinned(select);
                                }
                            }
                            // UNBOUNDED: the survivor set itself outgrew the budget — its final
                            // device sort cannot fit. Decline honestly and fail loudly.
                            None => {
                                if runs_bytes > budget {
                                    return self.execute_relational_select_cpu_pinned(select);
                                }
                            }
                        }
                    }
                }
            }
        }
        if !chunk_rows.is_empty() {
            let Ok(next) =
                self.stage_streaming_chunk(table, &chunk_rows, chunk_range, &mut capture, gpu_id)
            else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            if let Some(prev) = staged.replace(next) {
                let Ok((src, chunk_vis)) = prev.ready() else {
                    return self.execute_relational_select_cpu_pinned(select);
                };
                match self.ordered_streaming_chunk(
                    &chunk_select,
                    table,
                    &chunk_bound,
                    predicate,
                    copin_s,
                    &src,
                    chunk_vis,
                    &partial_types,
                    &mut runs_acc,
                    &mut runs_bytes,
                ) {
                    ChunkOutcome::Ok => {}
                    ChunkOutcome::Defer => {
                        return self.execute_relational_select_cpu_pinned(select)
                    }
                    ChunkOutcome::Hard(err) => return Err(err),
                }
                chunks_run += 1;
            }
        }
        // Pipeline-lag re-check (mirrors the grouped fold): the tail-block compute lands one chunk's
        // runs after the loop's last check — re-compact (top-N) / re-gate (unbounded) before draining.
        if runs_bytes >= chunk_target_bytes {
            match top_n {
                Some(_) => {
                    match self.sort_streaming_runs(
                        &compact_select,
                        &runs_table,
                        &compact_bound,
                        copin_s,
                        &partial_types,
                        &mut runs_acc,
                        &mut runs_bytes,
                        budget,
                        None,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                            return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    if runs_bytes > budget {
                        return self.execute_relational_select_cpu_pinned(select);
                    }
                }
                None => {
                    if runs_bytes > budget {
                        return self.execute_relational_select_cpu_pinned(select);
                    }
                }
            }
        }
        // DRAIN the pipeline before the final sort (no chunk upload alongside the sort upload).
        if let Some(prev) = staged.take() {
            let Ok((src, chunk_vis)) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.ordered_streaming_chunk(
                &chunk_select,
                table,
                &chunk_bound,
                predicate,
                copin_s,
                &src,
                chunk_vis,
                &partial_types,
                &mut runs_acc,
                &mut runs_bytes,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            chunks_run += 1;
        }

        // FINAL: one device sort + the REAL [OFFSET, OFFSET+LIMIT) window over the accumulated runs.
        // An empty accumulator (empty table / all filtered) is the empty ordered result.
        if !runs_acc.is_empty() {
            match self.sort_streaming_runs(
                &final_select,
                &runs_table,
                &final_bound,
                copin_s,
                &partial_types,
                &mut runs_acc,
                &mut runs_bytes,
                budget,
                None,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
        }
        // S-E.6: a COMPLETE scan installs its captured chunks for byte-replay by later reads.
        if let Some(builder) = capture {
            self.install_streaming_cold(&select.table, builder);
        }
        let rows_out = std::mem::take(&mut runs_acc);

        self.read_state
            .residency
            .streaming_fold_hits
            .fetch_add(1, Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_fold_chunks
            .fetch_add(chunks_run, Ordering::Relaxed);

        Ok(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns.clone()),
            rows: rows_out.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        })
    }

    /// One ordered-fold chunk: upload, run the per-chunk select on the device (top-N = sort + window to
    /// the bound; unbounded = plain filter/project), and CONCAT the resulting run into the accumulator.
    #[allow(clippy::too_many_arguments)]
    fn ordered_streaming_chunk(
        &self,
        chunk_select: &Select,
        table: &RelationalTable,
        chunk_bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        src: &ResidentExecSource,
        visibility: Option<crate::engine_expr::ResidentVisibility>,
        partial_types: &[SqlType],
        runs_acc: &mut Vec<Vec<SqlValue>>,
        runs_bytes: &mut u64,
    ) -> ChunkOutcome {
        let order_by_exprs: Vec<Option<ResidentExpr>> = vec![None; chunk_select.order_by.len()];
        let order_by_nulls_first: Vec<Option<bool>> = vec![None; chunk_select.order_by.len()];
        let result = match self.execute_resident_expr_select_with_binding(
            chunk_select,
            table,
            Some(src),
            chunk_bound.clone(),
            copin_s,
            predicate,
            visibility,
            &order_by_exprs,
            &order_by_nulls_first,
            None,
            &[],
        ) {
            Ok(result) => result,
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        for row in result.rows.into_boxed() {
            *runs_bytes = runs_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
            runs_acc.push(row);
        }
        ChunkOutcome::Ok
    }

    /// Device-sort (+ window) the accumulated runs as a transient synthesized relation, replacing the
    /// accumulator with the sorted/windowed rows. Serves BOTH the top-N compaction and the final pass.
    /// BUDGET GATE (the S-E.3 lesson): defer WITHOUT uploading when the accumulator exceeds the budget.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sort_streaming_runs(
        &self,
        sort_select: &Select,
        runs_table: &RelationalTable,
        sort_bound: &BoundRelationalSelect,
        copin_s: Index,
        partial_types: &[SqlType],
        runs_acc: &mut Vec<Vec<SqlValue>>,
        runs_bytes: &mut u64,
        budget: u64,
        order_by_nulls_first: Option<&[Option<bool>]>,
    ) -> ChunkOutcome {
        if *runs_bytes > budget {
            return ChunkOutcome::Defer;
        }
        let (descriptor, device_memory) =
            match self.build_transient_relation_residency(runs_table, runs_acc) {
                Ok(pair) => pair,
                Err(_) => return ChunkOutcome::Defer,
            };
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(descriptor.resident_bytes, Ordering::Relaxed);
        let src = ResidentExecSource {
            descriptor: Arc::new(descriptor),
            device_memory: Arc::new(device_memory),
            row_count: runs_acc.len() as u64,
        };
        let order_by_exprs: Vec<Option<ResidentExpr>> = vec![None; sort_select.order_by.len()];
        let order_by_nulls_first: Vec<Option<bool>> = order_by_nulls_first
            .map(<[Option<bool>]>::to_vec)
            .unwrap_or_else(|| vec![None; sort_select.order_by.len()]);
        let result = match self.execute_resident_expr_select_with_binding(
            sort_select,
            runs_table,
            Some(&src),
            sort_bound.clone(),
            copin_s,
            None,
            None,
            &order_by_exprs,
            &order_by_nulls_first,
            None,
            &[],
        ) {
            Ok(result) => result,
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        let mut sorted: Vec<Vec<SqlValue>> = Vec::new();
        let mut sorted_bytes: u64 = 0;
        for row in result.rows.into_boxed() {
            sorted_bytes = sorted_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
            sorted.push(row);
        }
        *runs_acc = sorted;
        *runs_bytes = sorted_bytes;
        ChunkOutcome::Ok
    }
}
