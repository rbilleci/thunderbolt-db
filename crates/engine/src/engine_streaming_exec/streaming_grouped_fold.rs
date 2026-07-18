//! Streaming GROUP BY and DISTINCT two-level fold and partial merge.

use super::*;

impl Engine {
    /// STRATA S-E.3 — the GROUP BY / DISTINCT two-level fold. LEVEL 1: each byte-bounded chunk runs the
    /// (normalized) grouped aggregate ON THE DEVICE, producing partial group rows `(key, agg_1..agg_N)`.
    /// LEVEL 2: the partials CONCAT (control-plane, the S-E.2 combine) into an accumulator that is itself
    /// a synthesized relation, and ONE final device grouped pass MERGES them — COUNT folds as SUM(count),
    /// SUM as SUM(sum), MIN as MIN(min), MAX as MAX(max) — so the host never groups or aggregates; it only
    /// stages type-stable partials. PG-bigint aggregates remain Numeric(38,0) across merge rounds and
    /// narrow once at final result materialization. If the accumulator outgrows the chunk budget mid-scan,
    /// it is COMPACTED by the same device merge (the "persistent accumulator" realized as periodic
    /// re-merge); if even the compacted (true-cardinality) partials exceed the budget, the query declines
    /// through the fail-loud GPU-required boundary.
    /// DISTINCT rides this fold via the distinct bridge's own synthesis (GROUP BY col + COUNT dropped).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_streaming_grouped_fold(
        &self,
        select: &Select,
        table: &RelationalTable,
        normalized: SelectProjection,
        distinct_key_only: bool,
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
        let SelectProjection::GroupedAggregates {
            group_column,
            aggregates,
        } = &normalized
        else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        // The per-chunk grouped select: the normalized projection over the ORIGINAL table (WHERE rides
        // the predicate; ORDER BY/LIMIT/HAVING are absent by the classifier).
        let mut grouped_select = select.clone();
        grouped_select.distinct = false;
        grouped_select.group_by = Some(group_column.clone());
        grouped_select.projection = normalized.clone();
        let Ok(grouped_bound) = bind_relational_select(table, &grouped_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };

        // The synthesized PARTIALS relation: `(key, __p0..__pN)` with each partial column typed by its
        // aggregate — Count and Sum(int2/int4/int8) -> Numeric(38,0), Sum(numeric) -> the column's
        // numeric type; Min/Max -> the value column's own type. An unsupported combination
        // (e.g. SUM over text) declines through the common fail-loud boundary.
        let key_type = match table
            .columns
            .iter()
            .find(|column| &column.name == group_column)
        {
            Some(column) => column.ty,
            None => return self.execute_relational_select_cpu_pinned(select),
        };
        let mut partial_columns: Vec<(String, SqlType)> = vec![(group_column.clone(), key_type)];
        // Guard the reserved partial names (a user column literally named `__pN` would collide).
        if group_column.starts_with("__p") {
            return self.execute_relational_select_cpu_pinned(select);
        }
        for (i, aggregate) in aggregates.iter().enumerate() {
            let partial_type = match aggregate.kind {
                // 6c-0(c) RE-LANDED: Count/Sum-int32 partials are DECLARED Numeric{38,0} so every
                // merge round's device SUM output matches the column type directly — the per-round
                // host narrow loop is DELETED; the single PG-type narrow happens once at result
                // materialization (the readback carve-out). (The first landing exposed the
                // masked-pass2 phantom-group kernel bug, now fixed + regression-gated.)
                GroupedAggKind::Count => Some(SqlType::Numeric {
                    precision: 38,
                    scale: 0,
                }),
                GroupedAggKind::Sum | GroupedAggKind::Min | GroupedAggKind::Max => {
                    let value_type = aggregate.value_column.as_ref().and_then(|name| {
                        table
                            .columns
                            .iter()
                            .find(|column| &column.name == name)
                            .map(|column| column.ty)
                    });
                    match (aggregate.kind, value_type) {
                        (GroupedAggKind::Sum, Some(SqlType::Int2 | SqlType::Int4)) => {
                            Some(SqlType::Numeric {
                                precision: 38,
                                scale: 0,
                            })
                        }
                        (GroupedAggKind::Sum, Some(SqlType::Int8)) => Some(SqlType::Numeric {
                            precision: 38,
                            scale: 0,
                        }),
                        (GroupedAggKind::Sum, Some(numeric @ SqlType::Numeric { .. })) => {
                            Some(numeric)
                        }
                        (GroupedAggKind::Min | GroupedAggKind::Max, Some(value_type)) => {
                            Some(value_type)
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            match partial_type {
                Some(ty) => partial_columns.push((format!("__p{i}"), ty)),
                None => return self.execute_relational_select_cpu_pinned(select),
            }
        }
        let partial_types: Vec<SqlType> = partial_columns.iter().map(|(_, ty)| *ty).collect();
        let partial_column_refs: Vec<(&str, SqlType)> = partial_columns
            .iter()
            .map(|(name, ty)| (name.as_str(), *ty))
            .collect();
        let partials_table =
            catalog_relation_table(&table.schema, "__stream_partials", &partial_column_refs);

        // The MERGE select over the partials relation: GROUP BY key with the fold aggregate per column.
        let merge_select = Select {
            table: partials_table.name.clone(),
            distinct: false,
            projection: SelectProjection::GroupedAggregates {
                group_column: group_column.clone(),
                aggregates: aggregates
                    .iter()
                    .enumerate()
                    .map(|(i, aggregate)| GroupedAggregate {
                        kind: match aggregate.kind {
                            // COUNT partials fold by SUMMING; SUM partials by SUMMING.
                            GroupedAggKind::Count | GroupedAggKind::Sum => GroupedAggKind::Sum,
                            other => other,
                        },
                        value_column: Some(format!("__p{i}")),
                    })
                    .collect(),
            },
            group_by: Some(group_column.clone()),
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        let Ok(merge_bound) = bind_relational_select(&partials_table, &merge_select) else {
            return self.execute_relational_select_cpu_pinned(select);
        };
        // 6c-0(c): which agg columns are PG-bigint results carried as Numeric{38,0} partials — their
        // chunk cells (the executor emits Int8 for COUNT / SUM(int2/4)) WRAP to Numeric at the staging
        // encode, and the merged cells NARROW back to Int8 exactly once, at result materialization
        // (the readback carve-out; a narrow overflow declines loudly). TYPE-CONSISTENT by
        // construction: keyed off the DECLARED partial column type.
        let bigint_as_numeric: Vec<bool> = aggregates
            .iter()
            .zip(partial_types.iter().skip(1))
            .map(|(aggregate, partial_ty)| {
                matches!(aggregate.kind, GroupedAggKind::Count | GroupedAggKind::Sum)
                    && matches!(partial_ty, SqlType::Numeric { .. })
                    && aggregate.value_column.as_ref().is_none_or(|name| {
                        table
                            .columns
                            .iter()
                            .find(|c| &c.name == name)
                            .is_some_and(|c| matches!(c.ty, SqlType::Int2 | SqlType::Int4))
                    })
            })
            .collect();

        let mut partials_acc: Vec<Vec<SqlValue>> = Vec::new();
        let mut partials_bytes: u64 = 0;
        let mut chunk_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut chunk_bytes: u64 = 0;
        // 6c-1: the current chunk's inclusive TupleId range ((1,0) = empty sentinel).
        let mut chunk_range: (u64, u64) = (1, 0);
        let mut chunks_run: u64 = 0;
        // S-E.5: the one-chunk lookahead (drained BEFORE any accumulator merge so a chunk upload never
        // rides alongside the merge upload — the residency invariant stays two-chunks-max).
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
                    match self.grouped_streaming_chunk(
                        &grouped_select,
                        table,
                        &grouped_bound,
                        predicate,
                        copin_s,
                        group_column,
                        &src,
                        chunk_vis,
                        &partial_types,
                        &bigint_as_numeric,
                        &mut partials_acc,
                        &mut partials_bytes,
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
                // The loop's compaction, with the same drain-first discipline.
                if partials_bytes >= chunk_target_bytes {
                    if let Some(prev) = staged.take() {
                        let Ok((src, chunk_vis)) = prev.ready() else {
                            return self.execute_relational_select_cpu_pinned(select);
                        };
                        match self.grouped_streaming_chunk(
                            &grouped_select,
                            table,
                            &grouped_bound,
                            predicate,
                            copin_s,
                            group_column,
                            &src,
                            chunk_vis,
                            &partial_types,
                            &bigint_as_numeric,
                            &mut partials_acc,
                            &mut partials_bytes,
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
                if partials_bytes >= chunk_target_bytes {
                    match self.merge_streaming_partials(
                        &merge_select,
                        &partials_table,
                        &merge_bound,
                        copin_s,
                        group_column,
                        &mut partials_acc,
                        &mut partials_bytes,
                        &partial_types,
                        budget,
                    ) {
                        ChunkOutcome::Ok => {}
                        ChunkOutcome::Defer => {
                            return self.execute_relational_select_cpu_pinned(select)
                        }
                        ChunkOutcome::Hard(err) => return Err(err),
                    }
                    if partials_bytes >= chunk_target_bytes {
                        return self.execute_relational_select_cpu_pinned(select);
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
                        match self.grouped_streaming_chunk(
                            &grouped_select,
                            table,
                            &grouped_bound,
                            predicate,
                            copin_s,
                            group_column,
                            &src,
                            chunk_vis,
                            &partial_types,
                            &bigint_as_numeric,
                            &mut partials_acc,
                            &mut partials_bytes,
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
                    // COMPACTION: the accumulator outgrew the chunk budget — device-merge it down to
                    // one row per true group. If even the compacted form exceeds the budget, the group
                    // cardinality itself is over-budget: defer (S-E.4+ may spill; v1 is honest).
                    // Drain the in-flight chunk FIRST so the merge upload never overlaps a chunk upload.
                    if partials_bytes >= chunk_target_bytes {
                        if let Some(prev) = staged.take() {
                            let Ok((src, chunk_vis)) = prev.ready() else {
                                return self.execute_relational_select_cpu_pinned(select);
                            };
                            match self.grouped_streaming_chunk(
                                &grouped_select,
                                table,
                                &grouped_bound,
                                predicate,
                                copin_s,
                                group_column,
                                &src,
                                chunk_vis,
                                &partial_types,
                                &bigint_as_numeric,
                                &mut partials_acc,
                                &mut partials_bytes,
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
                    if partials_bytes >= chunk_target_bytes {
                        match self.merge_streaming_partials(
                            &merge_select,
                            &partials_table,
                            &merge_bound,
                            copin_s,
                            group_column,
                            &mut partials_acc,
                            &mut partials_bytes,
                            &partial_types,
                            budget,
                        ) {
                            ChunkOutcome::Ok => {}
                            ChunkOutcome::Defer => {
                                return self.execute_relational_select_cpu_pinned(select)
                            }
                            ChunkOutcome::Hard(err) => return Err(err),
                        }
                        if partials_bytes >= chunk_target_bytes {
                            return self.execute_relational_select_cpu_pinned(select);
                        }
                    }
                }
            }
        }
        if !chunk_rows.is_empty() {
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
                match self.grouped_streaming_chunk(
                    &grouped_select,
                    table,
                    &grouped_bound,
                    predicate,
                    copin_s,
                    group_column,
                    &src,
                    chunk_vis,
                    &partial_types,
                    &bigint_as_numeric,
                    &mut partials_acc,
                    &mut partials_bytes,
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
        // Pipeline-lag compaction: the tail-block compute lands one whole chunk's partials AFTER the
        // loop's last compaction check, so re-check here — else the accumulator can reach the drain
        // compute already over-target and overshoot the merge budget gate (defer where the pre-pipeline
        // fold compacted and succeeded).
        if partials_bytes >= chunk_target_bytes {
            match self.merge_streaming_partials(
                &merge_select,
                &partials_table,
                &merge_bound,
                copin_s,
                group_column,
                &mut partials_acc,
                &mut partials_bytes,
                &partial_types,
                budget,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            if partials_bytes >= chunk_target_bytes {
                return self.execute_relational_select_cpu_pinned(select);
            }
        }
        // DRAIN the pipeline before the final merge (no chunk upload alongside the merge upload).
        if let Some(prev) = staged.take() {
            let Ok((src, chunk_vis)) = prev.ready() else {
                return self.execute_relational_select_cpu_pinned(select);
            };
            match self.grouped_streaming_chunk(
                &grouped_select,
                table,
                &grouped_bound,
                predicate,
                copin_s,
                group_column,
                &src,
                chunk_vis,
                &partial_types,
                &bigint_as_numeric,
                &mut partials_acc,
                &mut partials_bytes,
            ) {
                ChunkOutcome::Ok => {}
                ChunkOutcome::Defer => return self.execute_relational_select_cpu_pinned(select),
                ChunkOutcome::Hard(err) => return Err(err),
            }
            self.record_streaming_chunk_gpu(&src, gpu_id);
            chunks_run += 1;
        }

        // FINAL MERGE: fold duplicate keys across chunks into the one true group table. An empty
        // accumulator (empty table / all rows filtered) is PG's empty grouped result: ZERO rows.
        if !partials_acc.is_empty() {
            match self.merge_streaming_partials(
                &merge_select,
                &partials_table,
                &merge_bound,
                copin_s,
                group_column,
                &mut partials_acc,
                &mut partials_bytes,
                &partial_types,
                budget,
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
        let mut rows_out = std::mem::take(&mut partials_acc);
        // 6c-0(c) readback boundary: the PG-bigint aggregates (COUNT / SUM(int2/4)) rode as
        // Numeric{38,0} partials; narrow each merged cell to Int8 exactly ONCE, at result
        // materialization (a narrow overflow declines loudly at PG's own "bigint out of range"
        // surface).
        if !distinct_key_only {
            for row in &mut rows_out {
                for (agg_idx, narrow) in bigint_as_numeric.iter().enumerate() {
                    if !narrow {
                        continue;
                    }
                    let cell = &mut row[agg_idx + 1];
                    match cell {
                        SqlValue::Numeric(d) if d.scale == 0 => match i64::try_from(d.mantissa) {
                            Ok(narrowed) => *cell = SqlValue::Int8(narrowed),
                            Err(_) => return self.execute_relational_select_cpu_pinned(select),
                        },
                        SqlValue::Null | SqlValue::Int8(_) => {}
                        _ => return self.execute_relational_select_cpu_pinned(select),
                    }
                }
            }
        }
        // DISTINCT: drop the synthesized COUNT column — the bare distinct keys.
        if distinct_key_only {
            for row in &mut rows_out {
                row.truncate(1);
            }
        }

        self.read_state
            .residency
            .streaming_fold_hits
            .fetch_add(1, Ordering::Relaxed);
        self.read_state
            .residency
            .streaming_fold_chunks
            .fetch_add(chunks_run, Ordering::Relaxed);

        // Columns: DISTINCT keeps the outer (single-column) binding; grouped uses the NORMALIZED
        // binding (byte-identical to the grouped bridge, which also binds the normalized form).
        let columns = if distinct_key_only {
            bound.selected_columns.clone()
        } else {
            grouped_bound.selected_columns.clone()
        };
        Ok(RelationalSelectResult {
            columns: Arc::new(columns),
            rows: rows_out.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        })
    }

    /// LEVEL 1 of the grouped fold: upload one chunk, run the grouped aggregate ON THE DEVICE, and append
    /// its partial group rows to the accumulator (concat — the control-plane combine).
    #[allow(clippy::too_many_arguments)]
    fn grouped_streaming_chunk(
        &self,
        grouped_select: &Select,
        table: &RelationalTable,
        grouped_bound: &BoundRelationalSelect,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        group_column: &str,
        src: &ResidentExecSource,
        visibility: Option<crate::engine_expr::ResidentVisibility>,
        partial_types: &[SqlType],
        bigint_as_numeric: &[bool],
        partials_acc: &mut Vec<Vec<SqlValue>>,
        partials_bytes: &mut u64,
    ) -> ChunkOutcome {
        let group_key_columns = [group_column.to_string()];
        let result = match self.execute_resident_expr_select_with_binding(
            grouped_select,
            table,
            Some(src),
            grouped_bound.clone(),
            copin_s,
            predicate,
            visibility,
            &[],
            &[],
            None,
            &group_key_columns,
        ) {
            Ok(result) => result,
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        for mut row in result.rows.into_boxed() {
            // 6c-0(c) staging encode: the executor emits Int8 for COUNT/SUM(int2/4); the partials
            // relation declares those columns Numeric{38,0} — wrap losslessly (i64 -> i128 mantissa)
            // so every merge round is type-stable with NO per-round narrow.
            for (agg_idx, wrap) in bigint_as_numeric.iter().enumerate() {
                if !wrap {
                    continue;
                }
                if let SqlValue::Int8(n) = row[agg_idx + 1] {
                    row[agg_idx + 1] = SqlValue::Numeric(Decimal128::new(i128::from(n), 0));
                }
            }
            *partials_bytes =
                partials_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
            partials_acc.push(row);
        }
        ChunkOutcome::Ok
    }

    /// LEVEL 2 of the grouped fold: upload the accumulated partials as a transient relation and run ONE
    /// device grouped pass that MERGES duplicate keys (SUM/SUM/MIN/MAX per column), producing the declared
    /// partial types directly. Numeric(38,0) PG-bigint partials compose without per-round retyping; the
    /// single final narrow defers on overflow, matching PG's own error surface. Replaces the
    /// accumulator in place. Serves BOTH the mid-scan compaction and the final merge.
    ///
    /// BUDGET GATE (S-E.3 audit Finding 1): a partial row can be WIDER than its source rows (a 4-byte
    /// key + an 8-byte count = 12B partials from 4B rows), so a near-unique-key chunk can inflate the
    /// accumulator past the budget before the over-cardinality defer triggers. The merge must never be
    /// the thing that busts the budget it exists to honor — defer WITHOUT uploading when the accumulator
    /// exceeds it (the query declines loudly; the peak-bytes gauge invariant stays <= budget).
    #[allow(clippy::too_many_arguments)]
    fn merge_streaming_partials(
        &self,
        merge_select: &Select,
        partials_table: &RelationalTable,
        merge_bound: &BoundRelationalSelect,
        copin_s: Index,
        group_column: &str,
        partials_acc: &mut Vec<Vec<SqlValue>>,
        partials_bytes: &mut u64,
        partial_types: &[SqlType],
        budget: u64,
    ) -> ChunkOutcome {
        if *partials_bytes > budget {
            return ChunkOutcome::Defer;
        }
        let (descriptor, device_memory) =
            match self.build_transient_relation_residency(partials_table, partials_acc) {
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
            row_count: partials_acc.len() as u64,
        };
        let group_key_columns = [group_column.to_string()];
        let result = match self.execute_resident_expr_select_with_binding(
            merge_select,
            partials_table,
            Some(&src),
            merge_bound.clone(),
            copin_s,
            None,
            None,
            &[],
            &[],
            None,
            &group_key_columns,
        ) {
            Ok(result) => result,
            Err(err) if is_overflow_error(&err) => return ChunkOutcome::Hard(err),
            Err(_) => return ChunkOutcome::Defer,
        };
        let mut merged: Vec<Vec<SqlValue>> = Vec::new();
        let mut merged_bytes: u64 = 0;
        // 6c-0(c): NO per-round re-typing — the merge output's cell types ARE the declared partial
        // column types (Numeric{38,0} for the PG-bigint aggregates); rounds compose with no host
        // work. The single PG-type narrow happens at result materialization.
        for row in result.rows.into_boxed() {
            merged_bytes = merged_bytes.saturating_add(chunk_row_device_bytes(&row, partial_types));
            merged.push(row);
        }
        *partials_acc = merged;
        *partials_bytes = merged_bytes;
        ChunkOutcome::Ok
    }
}
