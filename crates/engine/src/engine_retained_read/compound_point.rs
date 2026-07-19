use super::*;

/// Typed engine-internal parameter for the BENCH-001 canonical compound key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationalCompoundI32I64PointReadParam {
    pub first: i32,
    pub second: i64,
}

/// Prepared route identity for one exact resident table generation and projection shape. The caller
/// owns no GPU allocation: the engine's bounded route cache owns and accounts the generation plan.
#[derive(Debug, Clone)]
pub struct RelationalCompoundI32I64PointReadTemplate {
    pub route_id: String,
    pub schema: String,
    pub table: String,
    pub snapshot_generation: u64,
    pub key_columns: [String; 2],
    projection_columns: Arc<Vec<RelationalColumn>>,
    pub(crate) table_generation: Arc<()>,
    pub(crate) route_key: CompoundPointRouteKey,
    pub(crate) projection_positions: Vec<usize>,
}

impl RelationalCompoundI32I64PointReadTemplate {
    pub fn projection_columns(&self) -> &[RelationalColumn] {
        self.projection_columns.as_slice()
    }
}

impl Engine {
    /// Prepare the canonical GPU-native `(int4, int8)` equality route without SQL/wire lowering.
    /// The table must expose an exact unique key in the supplied order and every referenced key/result
    /// cell must be resident and non-NULL in the captured generation. Wider SQL/protocol type handling
    /// remains PRODUCT-002; this API accepts already-typed engine values only.
    pub fn prepare_relational_compound_i32_i64_point_read_template(
        &self,
        schema: &str,
        table_name: &str,
        key_columns: [&str; 2],
        projection_columns: &[&str],
    ) -> Result<RelationalCompoundI32I64PointReadTemplate, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if self.current_transaction_read_snapshot().is_some() {
            return Err(compound_route_error(
                "compound prepared routes cannot be built inside an existing transaction snapshot",
            ));
        }
        if projection_columns.is_empty() || projection_columns.len() > 4 {
            return Err(compound_route_error(
                "compound prepared routes require one to four fixed-width projection columns",
            ));
        }

        // Capture catalog, boundary, shards, and sidecars at one publication cut while keeping the
        // expensive GPU build outside the commit lock. The statement scope pins that exact generation.
        let commit = self.commit_state();
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let snapshot = self.capture_statement_snapshot(self.committed_seq());
        drop(commit);
        let read_boundary = snapshot.boundary;
        let table = snapshot
            .catalog
            .relational_catalog
            .get(table_name)
            .cloned()
            .ok_or_else(|| {
                compound_route_error(format!("relation \"{table_name}\" does not exist"))
            })?;
        let _scope = self.enter_transaction_read(snapshot);
        if table.schema != schema {
            return Err(compound_route_error(format!(
                "relation \"{table_name}\" is in schema \"{}\", not \"{schema}\"",
                table.schema
            )));
        }
        let key_positions = key_columns
            .iter()
            .map(|name| {
                table
                    .columns
                    .iter()
                    .position(|column| column.name == *name)
                    .ok_or_else(|| {
                        compound_route_error(format!(
                            "compound route key column \"{name}\" does not exist"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if table.columns[key_positions[0]].ty != SqlType::Int4
            || table.columns[key_positions[1]].ty != SqlType::Int8
        {
            return Err(compound_route_error(
                "compound prepared route requires ordered (int4, int8) key columns",
            ));
        }
        let expected_key_columns = key_columns.map(str::to_string).to_vec();
        let (index_ordinal, index) = table
            .indexes
            .iter()
            .enumerate()
            .find(|(_, index)| index.unique && index.key_columns == expected_key_columns)
            .ok_or_else(|| {
                compound_route_error(
                    "compound prepared route requires an exact unique (int4, int8) index",
                )
            })?;
        let key_id = crate::engine_residency::index_probe_key_id(&table, index, index_ordinal)
            .filter(|key_id| key_id & crate::engine_residency::COMPOUND_KEY_ID_FLAG != 0)
            .ok_or_else(|| {
                compound_route_error("compound index has no device fingerprint key id")
            })?;

        let projection_positions = projection_columns
            .iter()
            .map(|name| {
                table
                    .columns
                    .iter()
                    .position(|column| column.name == *name)
                    .ok_or_else(|| {
                        compound_route_error(format!(
                            "compound route projection column \"{name}\" does not exist"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let projection_schema = projection_positions
            .iter()
            .map(|position| table.columns[*position].clone())
            .collect::<Vec<_>>();
        if projection_schema
            .iter()
            .any(|column| !matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Int8))
        {
            return Err(compound_route_error(
                "compound prepared route projections currently require int2/int4/int8 columns",
            ));
        }

        if self.read_streaming_cold_chunks().contains_key(table_name) {
            return Err(compound_route_error(
                "compound prepared route requires zero cold accesses and a fully resident table",
            ));
        }
        let shards = self.read_residency_shards();
        let table_shards = shards.get(table_name).ok_or_else(|| {
            compound_route_error("compound prepared route requires resident table shards")
        })?;
        if table_shards.is_empty() {
            return Err(compound_route_error(
                "compound prepared route requires at least one resident data shard",
            ));
        }
        let table_generation = Arc::clone(&table_shards[0].point_route_generation);
        if table_shards.iter().any(|shard| {
            shard.schema != schema
                || shard.table != table_name
                || !Arc::ptr_eq(&shard.point_route_generation, &table_generation)
        }) {
            return Err(compound_route_error(
                "compound prepared route observed a torn resident table generation",
            ));
        }
        let referenced_names = key_columns
            .iter()
            .copied()
            .chain(projection_columns.iter().copied())
            .collect::<std::collections::BTreeSet<_>>();
        if table_shards.iter().any(|shard| {
            shard
                .resident_device_null_columns
                .iter()
                .any(|layout| referenced_names.contains(layout.name.as_str()))
        }) {
            return Err(compound_route_error(
                "compound prepared route referenced a NULL-bearing key or projection column",
            ));
        }

        let route_key = (table.name.clone(), key_id, projection_positions.clone());
        let template = RelationalCompoundI32I64PointReadTemplate {
            route_id: format!(
                "compound_i32_i64_equality_projection:{}:{}:{}:{}",
                table.schema,
                table.name,
                key_columns.join(","),
                projection_columns.join(",")
            ),
            schema: table.schema.clone(),
            table: table.name.clone(),
            snapshot_generation: read_boundary,
            key_columns: key_columns.map(str::to_string),
            projection_columns: Arc::new(projection_schema),
            table_generation: Arc::clone(&table_generation),
            route_key: route_key.clone(),
            projection_positions: projection_positions.clone(),
        };
        if self
            .read_state
            .residency
            .compound_point_routes
            .load()
            .get(&route_key)
            .is_some_and(|route| {
                Arc::ptr_eq(&route.table_generation, &table_generation)
                    && route.read_boundary <= read_boundary
            })
        {
            return Ok(template);
        }

        let runtime_snapshot = self.router.runtime().snapshot();
        let mut gpu_id = None;
        let mut probe_shards = Vec::with_capacity(table_shards.len());
        let mut total_rows = 0_u64;
        for shard in table_shards {
            if shard.row_count == 0 {
                continue;
            }
            if runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id)
                || !shard.is_valid(false)
            {
                return Err(compound_route_error(
                    "compound prepared route requires valid, non-pressured resident shards",
                ));
            }
            if gpu_id
                .replace(shard.gpu_id)
                .is_some_and(|prior| prior != shard.gpu_id)
            {
                return Err(compound_route_error(
                    "compound prepared route currently requires one GPU ownership domain",
                ));
            }
            let descriptor = self.resident_snapshot_for_shard(shard, &table);
            let resident = shard.device_memory.clone().ok_or_else(|| {
                compound_route_error("compound prepared route shard has no device allocation")
            })?;
            if !self.shard_write_locate_cell_live(table_name, shard.shard_id, &resident) {
                return Err(compound_route_error(
                    "compound prepared route shard allocation is no longer current",
                ));
            }
            let key_i32_offset =
                resident_device_int4_column_offset(&descriptor, &table, key_positions[0])?;
            let key_i64_offset =
                resident_device_int8_column_offset(&descriptor, &table, key_positions[1])?;
            let projections = projection_positions
                .iter()
                .map(|position| match table.columns[*position].ty {
                    SqlType::Int2 | SqlType::Int4 => {
                        resident_device_int4_column_offset(&descriptor, &table, *position).map(
                            |byte_offset| CudaFixedPointProjection {
                                byte_offset,
                                kind: CudaFixedPointProjectionKind::I32,
                            },
                        )
                    }
                    SqlType::Int8 => {
                        resident_device_int8_column_offset(&descriptor, &table, *position).map(
                            |byte_offset| CudaFixedPointProjection {
                                byte_offset,
                                kind: CudaFixedPointProjectionKind::I64,
                            },
                        )
                    }
                    _ => unreachable!("projection type checked above"),
                })
                .collect::<Result<Vec<_>, _>>()?;
            total_rows = total_rows
                .checked_add(shard.row_count as u64)
                .ok_or_else(|| compound_route_error("compound route row count overflowed"))?;
            probe_shards.push(CompoundI32I64ProbeShard {
                resident,
                key_i32_offset,
                key_i64_offset,
                projections,
                row_count: shard.row_count as u64,
                created_by: shard.created_by_region.clone(),
                deleted_by: shard.deleted_by_region.clone(),
            });
        }
        let gpu_id = gpu_id.ok_or_else(|| {
            compound_route_error("compound prepared route has no non-empty resident shard")
        })?;
        let estimated = CudaI32I64MultiShardProbePlan::estimated_allocated_bytes(
            probe_shards.len(),
            total_rows,
        )
        .map_err(|error| compound_route_error(format!("compound route sizing failed: {error}")))?;

        // Serialize budget preflight through route publication. The preflight includes the existing
        // route rather than assuming its allocation drops immediately; an in-flight reader may still pin it.
        #[cfg(test)]
        self.run_sharded_point_route_pre_publish_hook();
        let _budget = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self
            .read_state
            .residency
            .compound_point_routes
            .load()
            .get(&route_key)
            .is_some_and(|route| {
                Arc::ptr_eq(&route.table_generation, &table_generation)
                    && route.read_boundary <= read_boundary
            })
        {
            return Ok(template);
        }
        let live_before = self.relational_resident_bytes_for_gpu(gpu_id);
        if let Some(limit) = self.relational_residency_budget_bytes(gpu_id) {
            if live_before.saturating_add(estimated) > limit {
                return Err(compound_route_error(format!(
                    "compound route residency budget exceeded: {live_before} live + {estimated} route > {limit} bytes"
                )));
            }
        }
        let launch_resident = Arc::clone(&probe_shards[0].resident);
        let plan = launch_resident
            .prepare_compound_i32_i64_multi_shard_probe(&probe_shards, read_boundary)
            .map_err(|error| {
                compound_route_error(format!(
                    "GPU compound point-route preparation failed: {error}"
                ))
            })?;
        let actual = plan.allocated_bytes();
        if actual != estimated {
            return Err(compound_route_error(format!(
                "compound route allocation estimate drifted: estimated {estimated} bytes, retained {actual} bytes"
            )));
        }

        let _publish = self
            .read_state
            .residency
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let generation_is_current = self
            .read_state
            .residency
            .shards
            .load()
            .get(table_name)
            .and_then(|current| current.first())
            .is_some_and(|current| Arc::ptr_eq(&table_generation, &current.point_route_generation));
        if !generation_is_current {
            return Err(compound_route_error(
                "compound prepared route generation changed during preparation",
            ));
        }
        let plan = Arc::new(CachedCompoundI32I64PointPlan {
            _charge: CompoundPointRouteCharge::new(
                gpu_id,
                table.name.clone(),
                actual,
                Arc::clone(&self.read_state.residency.live_compound_point_route_bytes),
            ),
            plan,
        });
        let current = self.read_state.residency.compound_point_routes.load();
        let mut next = (**current).clone();
        next.retain(|(cached_table, _, _), _| cached_table != table_name);
        if next.len() >= MAX_CACHED_SHARDED_POINT_ROUTES {
            if let Some(evicted) = next.keys().next().cloned() {
                next.remove(&evicted);
            }
        }
        next.insert(
            route_key,
            CachedCompoundI32I64PointRoute {
                table_generation,
                read_boundary,
                gpu_id,
                launch_resident,
                plan,
            },
        );
        self.read_state
            .residency
            .compound_point_routes
            .store(Arc::new(next));
        Ok(template)
    }

    /// Execute a batch through an already-prepared canonical compound route. A statement snapshot pins
    /// the exact generation. Missing/evicted/stale routes fail explicitly so callers reprepare; they never
    /// fall through to cold storage or CPU relational execution.
    pub fn execute_relational_compound_i32_i64_point_reads(
        &self,
        template: &RelationalCompoundI32I64PointReadTemplate,
        params: &[RelationalCompoundI32I64PointReadParam],
    ) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if params.is_empty() {
            return Err(compound_route_error(
                "compound prepared route requires at least one typed parameter tuple",
            ));
        }
        if self.current_transaction_read_snapshot().is_some() {
            return Err(compound_route_error(
                "compound prepared route execution requires an autocommit statement snapshot",
            ));
        }
        let commit = self.commit_state();
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let snapshot = self.capture_statement_snapshot(self.committed_seq());
        drop(commit);
        let read_boundary = snapshot.boundary;
        let _scope = self.enter_transaction_read(snapshot);
        let current_generation = self
            .read_residency_shards()
            .get(&template.table)
            .and_then(|shards| shards.first())
            .map(|shard| Arc::clone(&shard.point_route_generation))
            .ok_or_else(|| compound_route_error("compound route table is no longer resident"))?;
        if !Arc::ptr_eq(&current_generation, &template.table_generation) {
            return Err(compound_route_error(
                "compound prepared route template is stale; reprepare against the current generation",
            ));
        }
        let (gpu_id, launch_resident, plan) = self
            .read_state
            .residency
            .compound_point_routes
            .load()
            .get(&template.route_key)
            .and_then(|route| {
                (Arc::ptr_eq(&route.table_generation, &template.table_generation)
                    && route.read_boundary <= read_boundary)
                    .then(|| {
                        (
                            route.gpu_id,
                            Arc::clone(&route.launch_resident),
                            Arc::clone(&route.plan),
                        )
                    })
            })
            .ok_or_else(|| {
                compound_route_error(
                    "compound prepared route is no longer cached; reprepare before execution",
                )
            })?;
        if self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id)
        {
            return Err(compound_route_error(
                "compound prepared route is unavailable while its GPU is memory pressured",
            ));
        }
        self.read_state
            .residency
            .sharded_point_route_cache_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let keys = params
            .iter()
            .map(|param| CudaI32I64PointKey::new(param.first, param.second))
            .collect::<Vec<_>>();
        let projection = launch_resident
            .execute_prepared_compound_i32_i64_multi_shard_probe(&plan.plan, &keys, read_boundary)
            .map_err(|error| {
                compound_route_error(format!(
                    "GPU compound point-route execution failed: {error}"
                ))
            })?;
        if projection.projection_kinds().len() != template.projection_positions.len()
            || projection
                .projection_kinds()
                .iter()
                .zip(template.projection_columns.iter())
                .any(|(kind, column)| {
                    !matches!(
                        (kind, column.ty),
                        (
                            CudaFixedPointProjectionKind::I32,
                            SqlType::Int2 | SqlType::Int4
                        ) | (CudaFixedPointProjectionKind::I64, SqlType::Int8)
                    )
                })
            || projection.values().len()
                != params
                    .len()
                    .checked_mul(template.projection_positions.len())
                    .ok_or_else(|| compound_route_error("compound result cardinality overflowed"))?
        {
            return Err(compound_route_error(
                "GPU compound point-route returned an invalid projection shape",
            ));
        }
        if let Some(needle) = projection.status().iter().position(|status| *status == 3) {
            return Err(compound_route_error(format!(
                "GPU compound point-route found multiple visible exact matches for parameter {needle}"
            )));
        }
        let matched_keys = projection
            .status()
            .iter()
            .filter(|status| **status == 1)
            .count();
        let access_path = Arc::new(RelationalAccessPath::ConjunctiveFilteredKeyBatch {
            table: template.table.clone(),
            predicate_count: 2,
            matched_keys,
        });
        let ncols = template.projection_positions.len();
        let mut results = Vec::with_capacity(params.len());
        for (needle, status) in projection.status().iter().copied().enumerate() {
            let mut values = Vec::with_capacity(if status == 1 { ncols } else { 0 });
            if status == 1 {
                for (column, raw) in template
                    .projection_columns
                    .iter()
                    .zip(&projection.values()[needle * ncols..(needle + 1) * ncols])
                {
                    values.push(match column.ty {
                        SqlType::Int2 => SqlValue::Int2((*raw as u32 as i32) as i16),
                        SqlType::Int4 => SqlValue::Int4(*raw as u32 as i32),
                        SqlType::Int8 => SqlValue::Int8(*raw as i64),
                        _ => unreachable!("projection type checked at preparation"),
                    });
                }
            }
            results.push(RelationalSelectResult {
                columns: Arc::clone(&template.projection_columns),
                rows: RowBlock::flat(values, ncols),
                planned_target: DeviceTarget::Gpu(gpu_id),
                executed_target: DeviceTarget::Gpu(gpu_id),
                fallback_reason: None,
                access_path: Arc::clone(&access_path),
            });
        }
        self.read_state
            .residency
            .sharded_point_gpu_probe_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.read_state
            .residency
            .sharded_point_batch_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(results)
    }
}

fn compound_route_error(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message.into()))
}
