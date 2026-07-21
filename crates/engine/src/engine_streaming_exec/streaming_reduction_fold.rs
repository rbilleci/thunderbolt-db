//! Streaming scalar reduction planning, chunk execution, and final device combine.

use super::*;

/// 6c-0 (charter-drift ruling): the SCALAR combine runs ON THE DEVICE. Per-chunk partials collect as
/// rows of a synthesized one-column relation and ONE final device aggregate pass folds them — the
/// S-E.3 synthesized-relation merge shape (`catalog_relation_table` + injected src). The former host
/// accumulator (StreamAccum: Decimal128 adds, compare_sql_values extremes) is DELETED in this merge.
/// The partial column's type + the fold aggregate per original shape (the S-E.3 typing rules):
/// COUNT -> Int8 partials folded by SUM; SUM(int2/4) -> Int8 by SUM; SUM(int8) -> Numeric{38,0} by
/// SUM; SUM(numeric{p,s}) -> same numeric by SUM; MIN/MAX -> the value type by MIN/MAX. Returns None
/// when the value type is not device-reducible (the per-chunk pass would have deferred anyway).
fn scalar_partial_plan(
    table: &RelationalTable,
    select: &Select,
    agg: StreamAgg,
) -> Option<(SqlType, SelectProjection)> {
    let value_type = |column: &String| {
        table
            .columns
            .iter()
            .find(|c| &c.name == column)
            .map(|c| c.ty)
    };
    let partial = "__p0".to_string();
    match (agg, &select.projection) {
        (StreamAgg::Count, _) => Some((SqlType::Int8, SelectProjection::Sum { column: partial })),
        (StreamAgg::Sum, SelectProjection::Sum { column }) => match value_type(column)? {
            SqlType::Int2 | SqlType::Int4 => {
                Some((SqlType::Int8, SelectProjection::Sum { column: partial }))
            }
            SqlType::Int8 => Some((
                SqlType::Numeric {
                    precision: 38,
                    scale: 0,
                },
                SelectProjection::Sum { column: partial },
            )),
            numeric @ SqlType::Numeric { .. } => {
                Some((numeric, SelectProjection::Sum { column: partial }))
            }
            _ => None,
        },
        (StreamAgg::Min, SelectProjection::Min { column }) => Some((
            value_type(column)?,
            SelectProjection::Min { column: partial },
        )),
        (StreamAgg::Max, SelectProjection::Max { column }) => Some((
            value_type(column)?,
            SelectProjection::Max { column: partial },
        )),
        _ => None,
    }
}

impl Engine {
    /// The fold driver: scan the table's MVCC-visible rows at the pinned boundary, accumulate them into
    /// byte-bounded chunks, and reduce+combine each chunk on the device. S-E.5 may overlap one computing
    /// and one staged chunk, each targeted at half the budget; descriptor overhead and one-row threshold
    /// overshoot remain visible in the per-descriptor peak gauge. Any unsupported executor error on a
    /// chunk declines the whole query through the fail-loud boundary.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_streaming_reduction_fold(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        agg: StreamAgg,
        gpu_id: u16,
        budget: u64,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Chunk to HALF the budget so the transient device payload + the executor's scratch/output buffers
        // stay within the budget together (the out-of-core invariant the peak-bytes gauge proves).
        let chunk_target_bytes = (budget / 2).max(1);
        let execution_gpus = self.streaming_execution_gpus(gpu_id, budget);
        let mut gpu_cursor = 0_usize;
        let count_select = {
            let mut s = select.clone();
            s.projection = SelectProjection::CountAll;
            s
        };
        let count_bound = bind_relational_select(table, &count_select)?;
        // 6c-0: the device-combine plan (partial column type + the fold aggregate). A value type the
        // device cannot reduce declines here (the per-chunk pass would defer on it anyway).
        let Some((partial_type, fold_projection)) = scalar_partial_plan(table, select, agg) else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        let mut partials: Vec<Vec<SqlValue>> = Vec::new();
        let mut total_matched: i64 = 0;
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        // 6c-1: the current chunk's inclusive TupleId range ((1,0) = empty sentinel).
        let mut chunk_range: (u64, u64) = (1, 0);
        let mut chunks_run: u64 = 0;
        // S-E.5: the ONE-chunk lookahead — the staged chunk's upload is in flight while the previous
        // chunk computes and the next chunk's rows stage on the host.
        let mut staged: Option<StagedChunk> = None;
        // S-E.6: cold-tier replay (no MVCC decode) on a hit; a miss scans + CAPTURES for next time.
        let cold = self.load_streaming_cold(&select.table, table, chunk_target_bytes, copin_s);
        let mut capture: Option<ColdCacheBuilder> = None;

        let prefix = relational_key_prefix(&select.table);
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        if let Some(cold) = &cold {
            for chunk in &cold.chunks {
                // P4-3 born gate: a chunk born after this reader's boundary is invisible to it.
                if chunk.payload_copin_s > copin_s {
                    continue;
                }
                let target_gpu = execution_gpus[gpu_cursor % execution_gpus.len()];
                gpu_cursor = gpu_cursor.wrapping_add(1);
                let Ok(next) = self.stage_cold_chunk_on_gpu(chunk, copin_s, target_gpu) else {
                    // A failed replay (e.g. a bad spill read) evicts the entry so the next read
                    // rebuilds instead of defer-thrashing (audit LOW).
                    self.evict_streaming_cold(&select.table);
                    return self.execute_relational_select_cpu_pinned(select);
                };
                if let Some(prev) = staged.replace(next) {
                    let Ok((src, chunk_vis)) = prev.ready() else {
                        return self.execute_relational_select_cpu_pinned(select);
                    };
                    match self.reduce_streaming_chunk(
                        select,
                        table,
                        bound,
                        predicate,
                        copin_s,
                        agg,
                        &src,
                        chunk_vis,
                        &count_select,
                        &count_bound,
                        &mut partials,
                        &mut total_matched,
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
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let decoded = decode_relational_row(&tuple.value, &table.columns)?;
                // Size by the DEVICE payload bytes this row occupies (a NULL / empty-text cell still
                // takes its fixed typed slot on the device), NOT the logical value bytes — else a
                // null-heavy table never flushes and the whole table uploads as one chunk (the
                // out-of-core bound would break; audit Finding 1).
                chunk_bytes =
                    chunk_bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
                chunk_range = if chunk_rows.is_empty() {
                    (tuple.tuple_id, tuple.tuple_id)
                } else {
                    (chunk_range.0, tuple.tuple_id)
                };
                chunk_rows.push(decoded);
                if chunk_bytes >= chunk_target_bytes {
                    // S-E.5 lookahead: stage this chunk (its upload overlaps), compute the PREVIOUS.
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
                    if let Some(prev) = staged.replace(next) {
                        let Ok((src, chunk_vis)) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.reduce_streaming_chunk(
                            select,
                            table,
                            bound,
                            predicate,
                            copin_s,
                            agg,
                            &src,
                            chunk_vis,
                            &count_select,
                            &count_bound,
                            &mut partials,
                            &mut total_matched,
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
        // The final (partial) chunk. When the table is EMPTY (or the tail cleared exactly), still run one
        // chunk so the aggregate gets its PG empty-set semantics (COUNT -> 0, SUM/MIN/MAX -> NULL). Then
        // DRAIN the pipeline (the last staged chunk still needs its compute).
        if !chunk_rows.is_empty() || (chunks_run == 0 && staged.is_none()) {
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
            if let Some(prev) = staged.replace(next) {
                let Ok((src, chunk_vis)) = prev.ready() else {
                    return self.execute_relational_select_cpu_pinned(select);
                };
                match self.reduce_streaming_chunk(
                    select,
                    table,
                    bound,
                    predicate,
                    copin_s,
                    agg,
                    &src,
                    chunk_vis,
                    &count_select,
                    &count_bound,
                    &mut partials,
                    &mut total_matched,
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
            match self.reduce_streaming_chunk(
                select,
                table,
                bound,
                predicate,
                copin_s,
                agg,
                &src,
                chunk_vis,
                &count_select,
                &count_bound,
                &mut partials,
                &mut total_matched,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            self.record_streaming_chunk_gpu(&src, gpu_id);
            chunks_run += 1;
        }

        // 6c-0: the CROSS-CHUNK COMBINE runs ON THE DEVICE — one aggregate pass over the collected
        // partials as a synthesized one-column relation. Zero matched rows emits PG's empty-set
        // result (COUNT->0, else NULL) from CARDINALITY bookkeeping alone — charter basis: kernel
        // orchestration (the host may decide WHETHER to launch from row counts, and the device SUM
        // fold over an empty partial set could not represent COUNT's 0 anyway); no data VALUE is
        // read or combined on the host here (audit 6c-0 LOW: justified from charter text, not
        // precedent, per the charter-drift ruling).
        let value = if total_matched == 0 {
            match agg {
                StreamAgg::Count => SqlValue::Int8(0),
                _ => SqlValue::Null,
            }
        } else {
            match self.combine_scalar_partials_on_device(
                select,
                table,
                partial_type,
                &fold_projection,
                copin_s,
                &partials,
                budget,
            ) {
                Ok(value) => value,
                // An all-NULL partial set (every matched row NULL in every chunk) that the device
                // scalar path cannot serve declines loudly rather than returning a host answer.
                Err(()) => return self.execute_relational_select_cpu_pinned(select),
            }
        };
        // 6c-0 readback boundary: the device SUM over Int8 partials returns Numeric(_,0) (PG's SUM
        // ladder); COUNT / SUM(int4) present as bigint on the wire — ONE checked narrow of the single
        // result cell at materialization (the charter's readback carve-out; a narrow overflow is PG's
        // own "bigint out of range").
        let value = match (&value, agg, matches!(partial_type, SqlType::Int8)) {
            (SqlValue::Numeric(d), StreamAgg::Count | StreamAgg::Sum, true) if d.scale == 0 => {
                match i64::try_from(d.mantissa) {
                    Ok(narrowed) => SqlValue::Int8(narrowed),
                    Err(_) => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "bigint out of range".to_string(),
                        )))
                    }
                }
            }
            _ => value,
        };
        // S-E.6: a COMPLETE scan installs its captured chunks for byte-replay by later reads.
        if let Some(builder) = capture {
            self.install_streaming_cold(&select.table, builder);
        }
        // Non-vacuity + out-of-core telemetry: the fold fired on the GPU, over `chunks_run` bounded chunks.
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
            rows: vec![vec![value]].into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        })
    }

    /// Upload one chunk as a transient resident source, reduce it on the device, and fold the partial in.
    /// One upload; a COUNT(*) launch (the empty-filtered-set guard + the COUNT value); and, for
    /// SUM/MIN/MAX with survivors, the reduction launch. This transient source drops at the end of the
    /// call; S-E.5 may concurrently retain one other half-budget chunk in its staged lookahead.
    #[allow(clippy::too_many_arguments)]
    fn reduce_streaming_chunk(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        agg: StreamAgg,
        src: &ResidentExecSource,
        visibility: Option<crate::engine_expr::ResidentVisibility>,
        count_select: &Select,
        count_bound: &BoundRelationalSelect,
        partials: &mut Vec<Vec<SqlValue>>,
        total_matched: &mut i64,
    ) -> ChunkOutcome {
        // COUNT(*) over the chunk (predicate applied on-device): both the COUNT value AND the empty-set
        // guard for SUM/MIN/MAX (the general reduction hard-errors over an empty filtered set).
        let chunk_count = match self.execute_resident_expr_select_with_binding(
            count_select,
            table,
            Some(src),
            count_bound.clone(),
            copin_s,
            predicate,
            visibility,
            &[],
            &[],
            None,
            &[],
        ) {
            Ok(result) => match result.rows.iter().next().and_then(|row| row.first()) {
                Some(SqlValue::Int8(n)) => *n,
                _ => return ChunkOutcome::Defer,
            },
            Err(_) => return ChunkOutcome::Defer,
        };

        *total_matched = total_matched.saturating_add(chunk_count);
        if agg == StreamAgg::Count {
            // 6c-0: the chunk's COUNT becomes an Int8 partial row; the device SUM pass folds them.
            partials.push(vec![SqlValue::Int8(chunk_count)]);
            return ChunkOutcome::Ok;
        }
        if chunk_count == 0 {
            // No surviving row in this chunk -> nothing to reduce (avoids the empty-set hard-error).
            return ChunkOutcome::Ok;
        }
        let value = match self.execute_resident_expr_select_with_binding(
            select,
            table,
            Some(src),
            bound.clone(),
            copin_s,
            predicate,
            visibility,
            &[],
            &[],
            None,
            &[],
        ) {
            Ok(result) => match result.rows.iter().next().and_then(|row| row.first()) {
                Some(cell) => cell.clone(),
                None => return ChunkOutcome::Defer,
            },
            // A genuine arithmetic OVERFLOW must SURFACE (PG errors on it); converting it to a generic
            // decline would mask it with a misleading message and make per-chunk overflow disagree with
            // cross-chunk overflow (audit Finding 2). Any other error declines this GPU route.
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        // 6c-0: the chunk's device partial (Null for an all-NULL matched set — the final pass's M3
        // validity conjunct skips it in-kernel) collects as a one-column row; NO host combine.
        partials.push(vec![value]);
        ChunkOutcome::Ok
    }

    /// 6c-0: the final SCALAR combine — ONE device aggregate pass over the collected partials
    /// uploaded as a synthesized one-column relation (the S-E.3 merge shape; the same pre-upload
    /// budget gate). Err(()) = the caller declines through the fail-loud boundary.
    #[allow(clippy::too_many_arguments)]
    fn combine_scalar_partials_on_device(
        &self,
        select: &Select,
        table: &RelationalTable,
        partial_type: SqlType,
        fold_projection: &SelectProjection,
        copin_s: Index,
        partials: &[Vec<SqlValue>],
        budget: u64,
    ) -> Result<SqlValue, ()> {
        let _ = (select, table);
        let partials_bytes: u64 = partials
            .iter()
            .map(|row| chunk_row_device_bytes(row, &[partial_type]))
            .sum();
        if partials_bytes > budget {
            return Err(());
        }
        let scalar_table =
            catalog_relation_table("public", "__stream_scalar", &[("__p0", partial_type)]);
        let fold_select = Select {
            table: scalar_table.name.clone(),
            public_only: false,
            distinct: false,
            projection: fold_projection.clone(),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        let fold_bound = bind_relational_select(&scalar_table, &fold_select).map_err(|_| ())?;
        let (descriptor, device_memory) = self
            .build_transient_relation_residency(&scalar_table, partials)
            .map_err(|_| ())?;
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(descriptor.resident_bytes, Ordering::Relaxed);
        let src = ResidentExecSource {
            descriptor: Arc::new(descriptor),
            device_memory: Arc::new(device_memory),
            row_count: partials.len() as u64,
        };
        let result = self
            .execute_resident_expr_select_with_binding(
                &fold_select,
                &scalar_table,
                Some(&src),
                fold_bound,
                copin_s,
                None,
                None,
                &[],
                &[],
                None,
                &[],
            )
            .map_err(|_| ())?;
        match result.rows.iter().next().and_then(|row| row.first()) {
            Some(cell) => Ok(cell.clone()),
            None => Err(()),
        }
    }
}
