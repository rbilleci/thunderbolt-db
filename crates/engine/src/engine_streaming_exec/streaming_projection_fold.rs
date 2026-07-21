//! Streaming filter/project fold, chunk execution, and final device window.

use super::*;

impl Engine {
    /// STRATA S-E.2 — the filter/project fold: scan the visible rows into byte-bounded chunks; per chunk,
    /// run the projection (predicate + column gather ON THE DEVICE, with a device-side LIMIT bounding the
    /// gather to the rows still needed) and CONCAT the returned rows — the ARCHITECTURE §13 projection
    /// combine. LIMIT/OFFSET cross-chunk WINDOWING runs as one final device pass over the concatenated
    /// survivor stream; the control plane only bounds per-chunk device work and stops staging when a
    /// LIMIT is satisfied. LIMIT without ORDER BY is any-N-rows per SQL, so
    /// scan-order windowing is a valid instance. A satisfied LIMIT stops the scan EARLY: the tail of the
    /// table is never even staged (the out-of-core win compounds).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_streaming_projection_fold(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        gpu_id: u16,
        budget: u64,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let chunk_target_bytes = (budget / 2).max(1);
        let execution_gpus = self.streaming_execution_gpus(gpu_id, budget);
        let mut gpu_cursor = 0_usize;
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        // 6c-0: the WINDOW BOUND = offset+limit. Per chunk the DEVICE limit is this constant (a
        // chunk's first `bound` survivors are its only possible global-window contribution); the
        // cross-chunk window itself runs as ONE final device pass. The fold's early-exit is pure
        // cardinality flow-control (`collected >= bound`), never value-based windowing on the host.
        let window_bound: Option<usize> = select
            .limit
            .map(|limit| select.offset.unwrap_or(0).saturating_add(limit));
        let mut chunk_select = select.clone();
        chunk_select.offset = None;
        chunk_select.limit = window_bound;
        let Ok(chunk_bound) = bind_relational_select(table, &chunk_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        let mut rows_out: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        // 6c-1: the current chunk's inclusive TupleId range ((1,0) = empty sentinel).
        let mut chunk_range: (u64, u64) = (1, 0);
        let mut chunks_run: u64 = 0;
        // S-E.5 lookahead — for the UNBOUNDED scan only: with a LIMIT the early-exit decision needs
        // THIS chunk's contribution before scanning further, so limited queries compute eagerly
        // (semantics identical to pre-pipeline, including "the tail is never staged").
        let mut staged: Option<StagedChunk> = None;
        // S-E.6: cold-tier replay on a hit; a miss scans + captures (discarded on a LIMIT early-exit
        // — only a COMPLETE scan installs).
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
                if window_bound.is_some_and(|bound| rows_out.len() >= bound) {
                    break;
                }
                let target_gpu = execution_gpus[gpu_cursor % execution_gpus.len()];
                gpu_cursor = gpu_cursor.wrapping_add(1);
                let Ok(next) = self.stage_cold_chunk_on_gpu(chunk, copin_s, target_gpu) else {
                    // A failed replay (e.g. a bad spill read) evicts the entry so the next read
                    // rebuilds instead of defer-thrashing (audit LOW).
                    self.evict_streaming_cold(&select.table);
                    return self.execute_relational_select_cpu_pinned(select);
                };
                let to_compute = if window_bound.is_some() {
                    Some(next)
                } else {
                    staged.replace(next)
                };
                if let Some(prev) = to_compute {
                    let Ok((src, chunk_vis)) = prev.ready() else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    match self.project_streaming_chunk(
                        &chunk_select,
                        table,
                        &chunk_bound,
                        predicate,
                        copin_s,
                        &src,
                        chunk_vis,
                        &mut rows_out,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                            return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    self.record_streaming_chunk_gpu(&src, gpu_id);
                    chunks_run += 1;
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
                // LIMIT satisfied -> STOP the scan: no further row is staged, decoded, or uploaded.
                // The capture is INCOMPLETE at an early exit — discard it (never install a partial set).
                if window_bound.is_some_and(|bound| rows_out.len() >= bound) {
                    capture = None;
                    break;
                }
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
                    let target_gpu = execution_gpus[gpu_cursor % execution_gpus.len()];
                    gpu_cursor = gpu_cursor.wrapping_add(1);
                    let Ok(next) = self.stage_streaming_chunk(
                        table,
                        &chunk_rows,
                        chunk_range,
                        &mut capture,
                        target_gpu,
                    ) else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    let to_compute = if window_bound.is_some() {
                        Some(next) // limited: compute eagerly (early-exit fidelity)
                    } else {
                        staged.replace(next) // unbounded: pipeline one chunk ahead
                    };
                    if let Some(prev) = to_compute {
                        let Ok((src, chunk_vis)) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.project_streaming_chunk(
                            &chunk_select,
                            table,
                            &chunk_bound,
                            predicate,
                            copin_s,
                            &src,
                            chunk_vis,
                            &mut rows_out,
                        ) {
                            ChunkOutcome::Ok => {}
                            ChunkOutcome::Defer => {
                                return self.execute_relational_select_cpu_pinned(select)
                            }
                            ChunkOutcome::Hard(err) => return Err(err),
                        }
                        self.record_streaming_chunk_gpu(&src, gpu_id);
                        chunks_run += 1;
                    }
                    chunk_rows.clear();
                    chunk_bytes = 0;
                    chunk_range = (1, 0);
                }
            }
        }
        // The final (partial) chunk — skipped when the LIMIT already filled (rows staged before the
        // early-exit tripped would be dropped by the window anyway; don't upload them). Then DRAIN the
        // unbounded pipeline (the last staged chunk still needs its compute).
        if !chunk_rows.is_empty() && window_bound.is_none_or(|bound| rows_out.len() < bound) {
            let target_gpu = execution_gpus[gpu_cursor % execution_gpus.len()];
            let Ok(next) = self.stage_streaming_chunk(
                table,
                &chunk_rows,
                chunk_range,
                &mut capture,
                target_gpu,
            ) else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            let to_compute = if window_bound.is_some() {
                Some(next)
            } else {
                staged.replace(next)
            };
            if let Some(prev) = to_compute {
                let Ok((src, chunk_vis)) = prev.ready() else {
                    return self.execute_relational_select_cpu_pinned(select);
                };
                match self.project_streaming_chunk(
                    &chunk_select,
                    table,
                    &chunk_bound,
                    predicate,
                    copin_s,
                    &src,
                    chunk_vis,
                    &mut rows_out,
                ) {
                    ChunkOutcome::Ok => {}
                    ChunkOutcome::Defer => {
                        return self.execute_relational_select_cpu_pinned(select)
                    }
                    ChunkOutcome::Hard(err) => return Err(err),
                }
                self.record_streaming_chunk_gpu(&src, gpu_id);
                chunks_run += 1;
            }
        }
        if let Some(prev) = staged.take() {
            let Ok((src, chunk_vis)) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.project_streaming_chunk(
                &chunk_select,
                table,
                &chunk_bound,
                predicate,
                copin_s,
                &src,
                chunk_vis,
                &mut rows_out,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            self.record_streaming_chunk_gpu(&src, gpu_id);
            chunks_run += 1;
        }

        // 6c-0: the cross-chunk OFFSET/LIMIT window — ONE device pass over the collected survivors
        // (the executor's own window path); without a window the concat IS the result. A window-pass
        // decline (e.g. the collected set over budget) fails loudly — never a wrong window.
        if (select.limit.is_some() || select.offset.is_some())
            && self
                .window_streaming_rows(select, bound, copin_s, &mut rows_out, budget)
                .is_err()
        {
            return self.execute_relational_select_cpu_pinned(select);
        }
        // S-E.6: a COMPLETE scan installs its captured chunks (None after a LIMIT early-exit).
        if let Some(builder) = capture {
            self.install_streaming_cold(&select.table, builder);
        }
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

    /// Upload one chunk as a transient resident source, run the projection on the device (bounded by a
    /// device-side LIMIT of the window bound — a chunk's first `offset+limit` survivors are its only
    /// possible contribution to the global window), and CONCAT the survivors. 6c-0 (charter-drift
    /// ruling): the former host drain/truncate windowing is DELETED — the cross-chunk window runs as
    /// ONE final device pass (`window_streaming_rows`); the fold's early-exit is pure CARDINALITY
    /// flow-control on the collected count.
    #[allow(clippy::too_many_arguments)]
    fn project_streaming_chunk(
        &self,
        chunk_select: &Select,
        table: &RelationalTable,
        chunk_bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        src: &ResidentExecSource,
        visibility: Option<crate::engine_expr::ResidentVisibility>,
        rows_out: &mut Vec<Vec<SqlValue>>,
    ) -> ChunkOutcome {
        let result = match self.execute_resident_expr_select_with_binding(
            chunk_select,
            table,
            Some(src),
            chunk_bound.clone(),
            copin_s,
            predicate,
            visibility,
            &[],
            &[],
            None,
            &[],
        ) {
            Ok(result) => result,
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        rows_out.append(&mut result.rows.into_boxed());
        ChunkOutcome::Ok
    }

    /// 6c-0: the cross-chunk OFFSET/LIMIT window as ONE device pass — the collected survivors upload
    /// as a synthesized relation and the executor applies `[OFFSET, OFFSET+LIMIT)` on its own window
    /// path (`sort_streaming_runs` with an EMPTY ORDER BY — the S-E.4 machinery minus the sort).
    pub(crate) fn window_streaming_rows(
        &self,
        select: &Select,
        bound: &BoundRelationalSelect,
        copin_s: Index,
        rows_out: &mut Vec<Vec<SqlValue>>,
        budget: u64,
    ) -> Result<(), ()> {
        let window_column_refs: Vec<(&str, SqlType)> = bound
            .selected_columns
            .iter()
            .map(|column| (column.name.as_str(), column.ty))
            .collect();
        let partial_types: Vec<SqlType> = window_column_refs.iter().map(|(_, ty)| *ty).collect();
        let window_table =
            catalog_relation_table(&select.table, "__stream_window", &window_column_refs);
        let window_select = Select {
            table: window_table.name.clone(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Columns(
                window_column_refs
                    .iter()
                    .map(|(name, _)| (*name).to_string())
                    .collect(),
            ),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: select.limit,
            offset: select.offset,
        };
        let window_bound = bind_relational_select(&window_table, &window_select).map_err(|_| ())?;
        let mut rows_bytes: u64 = rows_out
            .iter()
            .map(|row| chunk_row_device_bytes(row, &partial_types))
            .sum();
        match self.sort_streaming_runs(
            &window_select,
            &window_table,
            &window_bound,
            copin_s,
            &partial_types,
            rows_out,
            &mut rows_bytes,
            budget,
            None,
        ) {
            ChunkOutcome::Ok => Ok(()),
            _ => Err(()),
        }
    }
}
