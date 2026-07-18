//! Residency warmup, route planning, and status ownership.

use super::*;

impl Engine {
    pub fn warm_relational_residency_with_policy(
        &mut self,
        policy: RelationalResidencyWarmupPolicy,
    ) -> RelationalResidencyWarmupReport {
        let policy_sets_gpu = policy.gpu_id.is_some();
        let gpu_id = policy
            .gpu_id
            .unwrap_or_else(|| self.planner.default_gpu_id());
        let policy_sets_budget = policy.budget_bytes.is_some();
        if let Some(budget_bytes) = policy.budget_bytes {
            self.set_relational_residency_budget_bytes(gpu_id, budget_bytes);
        }
        let budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let requested_tables = if policy.tables.is_empty() {
            self.ddl_catalog_mut()
                .relational_catalog
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        } else {
            policy.tables.clone()
        };
        let mut selected_tables = requested_tables.clone();
        selected_tables.sort();
        selected_tables.dedup();
        if let Some(max_table_count) = policy.max_table_count {
            selected_tables.truncate(max_table_count);
        }

        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let mut entries = Vec::new();
        for table in selected_tables {
            if memory_pressure_active {
                entries.push(RelationalResidencyWarmupEntry {
                    table,
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: format!("GPU {gpu_id} is memory pressured"),
                    resident_bytes: 0,
                    evicted_tables: Vec::new(),
                    route_decision: None,
                });
                continue;
            }
            if !self
                .ddl_catalog_mut()
                .relational_catalog
                .contains_key(&table)
            {
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: "only supported public base tables can be warmed".to_string(),
                    resident_bytes: 0,
                    evicted_tables: Vec::new(),
                    route_decision: None,
                });
                continue;
            }

            let existing = self.relational_residency_snapshot(&table);
            let existing_valid = existing
                .as_ref()
                .is_some_and(|snapshot| snapshot.is_valid());
            let existing_retained = self.read_state.residency.device_memory.contains_key(&table);
            if existing_valid && existing_retained && !policy_sets_budget && !policy_sets_gpu {
                let route_decision = self.warmup_route_readiness_decision(&table);
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::AlreadyResident,
                    reason: "resident snapshot is already valid and retained".to_string(),
                    resident_bytes: existing
                        .as_ref()
                        .map(|snapshot| snapshot.resident_bytes)
                        .unwrap_or(0),
                    evicted_tables: Vec::new(),
                    route_decision,
                });
                continue;
            }
            if existing.is_some() && !policy.refresh_invalidated && !existing_valid {
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: "resident snapshot is invalidated and refresh is disabled".to_string(),
                    resident_bytes: existing
                        .as_ref()
                        .map(|snapshot| snapshot.resident_bytes)
                        .unwrap_or(0),
                    evicted_tables: Vec::new(),
                    route_decision: self.warmup_route_readiness_decision(&table),
                });
                continue;
            }

            let refreshing = existing.is_some();
            match self.populate_relational_residency_snapshot_on_gpu(&table, gpu_id) {
                Ok(snapshot) => {
                    let route_decision = self.warmup_route_readiness_decision(&table);
                    entries.push(RelationalResidencyWarmupEntry {
                        table: table.clone(),
                        action: if refreshing {
                            RelationalResidencyWarmupAction::Refreshed
                        } else {
                            RelationalResidencyWarmupAction::Warmed
                        },
                        reason: self
                            .ddl_catalog_mut()
                            .relational_resident_cache
                            .last_decision(&table)
                            .map(|decision| decision.reason.clone())
                            .unwrap_or_else(|| "resident snapshot warmed".to_string()),
                        resident_bytes: snapshot.resident_bytes,
                        evicted_tables: snapshot.evicted_tables_on_admission,
                        route_decision,
                    });
                }
                Err(err) => {
                    entries.push(RelationalResidencyWarmupEntry {
                        table: table.clone(),
                        action: RelationalResidencyWarmupAction::Error,
                        reason: err.to_string(),
                        resident_bytes: 0,
                        evicted_tables: Vec::new(),
                        route_decision: self.warmup_route_readiness_decision(&table),
                    });
                }
            }
        }

        RelationalResidencyWarmupReport {
            gpu_id,
            budget_bytes,
            requested_tables,
            entries,
        }
    }

    pub fn maintain_relational_residency_with_policy(
        &mut self,
        policy: RelationalResidencyMaintenancePolicy,
    ) -> RelationalResidencyMaintenanceReport {
        let warmup = self.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
            gpu_id: policy.gpu_id,
            tables: policy.tables,
            max_table_count: policy.max_table_count,
            budget_bytes: policy.budget_bytes,
            refresh_invalidated: policy.refresh_invalidated,
        });
        let mut warmed_count = 0;
        let mut refreshed_count = 0;
        let mut already_resident_count = 0;
        let mut skipped_count = 0;
        let mut error_count = 0;
        let mut route_ready_tables = Vec::new();
        let mut route_blockers = Vec::new();

        for entry in &warmup.entries {
            match entry.action {
                RelationalResidencyWarmupAction::Warmed => warmed_count += 1,
                RelationalResidencyWarmupAction::Refreshed => refreshed_count += 1,
                RelationalResidencyWarmupAction::AlreadyResident => already_resident_count += 1,
                RelationalResidencyWarmupAction::Skipped => skipped_count += 1,
                RelationalResidencyWarmupAction::Error => error_count += 1,
            }

            match entry.route_decision.as_ref() {
                Some(route) if route.accepted => route_ready_tables.push(entry.table.clone()),
                Some(route) => route_blockers.push(RelationalResidencyMaintenanceBlocker {
                    table: entry.table.clone(),
                    reason: if matches!(
                        entry.action,
                        RelationalResidencyWarmupAction::Skipped
                            | RelationalResidencyWarmupAction::Error
                    ) {
                        entry.reason.clone()
                    } else {
                        route.reason.clone()
                    },
                }),
                None => route_blockers.push(RelationalResidencyMaintenanceBlocker {
                    table: entry.table.clone(),
                    reason: entry.reason.clone(),
                }),
            }
        }

        let entry_count = warmup.entries.len();
        RelationalResidencyMaintenanceReport {
            gpu_id: warmup.gpu_id,
            budget_bytes: warmup.budget_bytes,
            requested_tables: warmup.requested_tables,
            entry_count,
            warmed_count,
            refreshed_count,
            already_resident_count,
            skipped_count,
            error_count,
            route_ready_count: route_ready_tables.len(),
            route_blocked_count: route_blockers.len(),
            route_ready_tables,
            route_blockers,
            entries: warmup.entries,
        }
    }

    fn warmup_route_readiness_decision(
        &mut self,
        table: &str,
    ) -> Option<RelationalResidentRouteDecisionStatus> {
        let sql = format!("SELECT COUNT(*) FROM {table}");
        let Ok(Command::Select(select)) = parse_command(&sql) else {
            return None;
        };
        Some(self.plan_relational_resident_route(&select))
    }

    fn relational_snapshot_cache_state(
        snapshot: &RelationalResidencySnapshot,
        memory_pressure_active: bool,
    ) -> &'static str {
        if memory_pressure_active || snapshot.invalidated_by_memory_pressure {
            "InvalidatedByMemoryPressure"
        } else if snapshot.invalidated_by_txn_id.is_some()
            || snapshot.invalidated_at_index.is_some()
        {
            "Invalidated"
        } else {
            "Valid"
        }
    }

    fn resident_route_reject(
        table: &str,
        reason: impl Into<String>,
        query_shape: impl Into<String>,
    ) -> RelationalResidentRouteDecisionStatus {
        RelationalResidentRouteDecisionStatus {
            table: table.to_string(),
            gpu_id: None,
            snapshot_generation: None,
            shard_count: 0,
            accepted: false,
            reason: reason.into(),
            query_shape: query_shape.into(),
            cache_state: "Absent".to_string(),
            valid: false,
            has_retained_device_memory: false,
            estimated_rows: 0,
            resident_bytes: 0,
            budget_bytes: None,
            refresh_resident_bytes: None,
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: 0,
            d2h_bytes_estimate: 0,
            d2h_rows_estimate: 0,
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        }
    }

    pub fn plan_relational_resident_route(
        &self,
        select: &Select,
    ) -> RelationalResidentRouteDecisionStatus {
        let decision = self.plan_relational_resident_route_inner(select);
        self.read_state
            .route_telemetry
            .record_route_decision(decision.clone());
        decision
    }

    fn plan_relational_resident_route_inner(
        &self,
        select: &Select,
    ) -> RelationalResidentRouteDecisionStatus {
        // Lock-free read path (Stage 2 — blocker #1): pin the catalog snapshot for the relation-kind
        // check (the subsequent table bind pins its own; both are immutable published snapshots).
        let catalog = self.catalog_snapshot();
        if catalog.relational_views.contains_key(&select.table)
            || catalog
                .relational_materialized_views
                .contains_key(&select.table)
        {
            return Self::resident_route_reject(
                &select.table,
                "resident routing currently supports only public base tables",
                "unsupported_relation_kind",
            );
        }

        let (table, bound, _copin_s) = match self.bind_relational_select_for_execution(select) {
            Ok(bound) => bound,
            Err(err) => {
                return Self::resident_route_reject(
                    &select.table,
                    format!("unsupported select shape: {err}"),
                    "unsupported_select",
                );
            }
        };

        // Stage 3 — blocker #2: pin the published resident shard + snapshot maps for the rest of
        // the planning decision (the shard slice is passed by reference into the sharded-route
        // planner, and the snapshot is read field-by-field below — both must outlive those uses, so the
        // guards are bound here and held to the end of the function).
        let shards_guard = self.read_residency_shards();
        let snapshots_guard = self.read_residency_snapshots();

        let query_shape = match resident_route_query_shape(select, &table, &bound) {
            Some(shape) => shape,
            None => {
                if let Some(shards) = shards_guard.get(&table.name) {
                    if let Some(shape) = sharded_resident_route_query_shape(select, &table, &bound)
                    {
                        return self
                            .plan_relational_sharded_resident_route(select, &table, shape, shards);
                    }
                }
                return Self::resident_route_reject(
                    &table.name,
                    "resident routing has no retained-kernel proof for this SELECT shape",
                    "unsupported_select",
                );
            }
        };

        if let Some(shards) = shards_guard.get(&table.name) {
            return self.plan_relational_sharded_resident_route(
                select,
                &table,
                query_shape,
                shards,
            );
        }

        let Some(entry) = snapshots_guard.get(&table.name) else {
            return Self::resident_route_reject(
                &table.name,
                "relation has no resident snapshot",
                query_shape,
            );
        };
        let snapshot = &entry.descriptor;
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&snapshot.gpu_id);
        let cache_state = Self::relational_snapshot_cache_state(snapshot, memory_pressure_active);
        let valid = snapshot.invalidated_by_txn_id.is_none()
            && snapshot.invalidated_at_index.is_none()
            && !snapshot.invalidated_by_memory_pressure
            && !memory_pressure_active;
        let has_retained_device_memory = self.read_resident_device_memory(&table.name).is_some();
        let d2h_bytes_estimate = resident_route_d2h_bytes_estimate(select, &query_shape, snapshot);
        let mut decision = RelationalResidentRouteDecisionStatus {
            table: table.name.clone(),
            gpu_id: Some(snapshot.gpu_id),
            snapshot_generation: Some(snapshot.generation),
            shard_count: 1,
            accepted: false,
            reason: String::new(),
            query_shape,
            cache_state: cache_state.to_string(),
            valid,
            has_retained_device_memory,
            estimated_rows: snapshot.row_count,
            resident_bytes: snapshot.resident_bytes,
            budget_bytes: snapshot.admission_budget_bytes,
            refresh_resident_bytes: snapshot
                .last_refresh_cost
                .as_ref()
                .map(|cost| cost.refreshed_resident_bytes),
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: snapshot.resident_bytes,
            d2h_bytes_estimate,
            d2h_rows_estimate: resident_route_d2h_rows_estimate(select, snapshot.row_count),
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        };

        if snapshot.schema != table.schema || snapshot.table != table.name {
            decision.reason =
                "resident snapshot no longer matches catalog table identity".to_string();
        } else if !valid {
            decision.reason = format!("resident snapshot is {cache_state}");
        } else if !has_retained_device_memory {
            decision.reason = "resident snapshot has no retained device memory".to_string();
        } else {
            decision.accepted = true;
            decision.reason = "resident route accepted".to_string();
        }
        decision
    }

    fn plan_relational_sharded_resident_route(
        &self,
        select: &Select,
        table: &RelationalTable,
        query_shape: String,
        shards: &[RelationalResidentShard],
    ) -> RelationalResidentRouteDecisionStatus {
        let total_rows = shards.iter().map(|shard| shard.row_count).sum::<usize>();
        let total_resident_bytes = shards.iter().map(|shard| shard.resident_bytes).sum::<u64>();
        let gpu_id = shards.first().map(|shard| shard.gpu_id);
        let sharded_query_shape = if query_shape == "count_all" {
            "sharded_count_all".to_string()
        } else if query_shape == "int4_equality_projection" {
            "sharded_int4_equality_projection".to_string()
        } else if query_shape == "int4_equality_multi_column_projection" {
            "sharded_int4_equality_multi_column_projection".to_string()
        } else if query_shape == "int4_equality_mixed_column_projection" {
            "sharded_int4_equality_mixed_column_projection".to_string()
        } else if matches!(
            query_shape.as_str(),
            "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
        ) {
            query_shape
        } else if matches!(
            query_shape.as_str(),
            "int4_equality_count"
                | "int4_range_count"
                | "int4_between_scalar_aggregate"
                | "int4_projection"
                | "int4_projection_all"
                | "int4_composite_equality_multi_column_projection"
        ) {
            // THE FLIP audit F1: these filtered/range int4 shapes had NO sharded mapping, so the
            // now-default sharded layout demoted them to the CPU host scan (GPU-served pre-flip).
            // The `sharded_` prefix routes them to the sharded BRIDGE (their unprefixed names
            // dispatch to the single-buffer enumerated kernels), whose general executor evaluates
            // the predicate + projection/aggregate on-device over the unified (or zero-copy
            // single-shard) source.
            format!("sharded_{query_shape}")
        } else if query_shape == "int4_filter_group_count" {
            // THE FLIP (burn-in): an OR-of-int4-equalities COUNT fell to the CPU engine on a sharded
            // table (no sharded mapping — the SUM cliff's sibling). The shape keeps its single-buffer
            // name: the dispatch arm routes it to the grouped bridge, whose `src: None` now resolves
            // the sharded unified source inside `execute_resident_expr_select_with_binding`.
            query_shape
        } else if query_shape == "int4_scalar_aggregate" {
            // FLIP slice (measured): an UNFILTERED scalar aggregate (bare SUM/AVG/MIN/MAX) had NO sharded
            // mapping, so it fell through the dispatch to the CPU engine's host scan — MEASURED p50
            // 496,554us vs the bridge-served sharded COUNT's 460us at 524k rows (~1000x, a charter
            // violation in the hot path). The bridge's COUNT-precheck + general run computes scalar
            // aggregates on the unified device buffer, so route it there.
            "sharded_int4_scalar_aggregate".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Avg { .. })
        {
            "sharded_int4_filtered_avg".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Min { .. })
        {
            "sharded_int4_filtered_min".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Max { .. })
        {
            "sharded_int4_filtered_max".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate" {
            // THE FLIP audit F1 (residue): the filtered aggregates NOT covered by the tuned
            // avg/min/max mappings above (a filtered SUM) route to the sharded bridge's general
            // executor instead of falling to the CPU host scan.
            "sharded_int4_filtered_scalar_aggregate".to_string()
        } else if query_shape == "int4_distinct_projection" {
            "sharded_int4_distinct_projection".to_string()
        } else if query_shape == "int4_filtered_distinct_projection" {
            "sharded_int4_filtered_distinct_projection".to_string()
        } else if query_shape == "int4_grouped_aggregate" {
            "sharded_int4_grouped_aggregate".to_string()
        } else if query_shape == "int4_filtered_grouped_aggregate" {
            "sharded_int4_filtered_grouped_aggregate".to_string()
        } else if query_shape == "int4_ordered_projection" {
            "sharded_int4_ordered_projection".to_string()
        } else {
            query_shape
        };
        let d2h_bytes_estimate = if matches!(
            sharded_query_shape.as_str(),
            "sharded_count_all"
                | "sharded_int4_equality_projection"
                | "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
        ) {
            shards
                .len()
                .checked_mul(std::mem::size_of::<u64>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(u64::MAX)
        } else if sharded_query_shape == "sharded_int4_equality_multi_column_projection" {
            let SelectProjection::Columns(columns) = &select.projection else {
                return Self::resident_route_reject(
                    &table.name,
                    "sharded resident routing has no retained-kernel proof for this SELECT shape",
                    sharded_query_shape,
                );
            };
            u64::try_from(total_rows)
                .unwrap_or(u64::MAX)
                .saturating_mul(
                    u64::try_from(columns.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(std::mem::size_of::<i32>() as u64)
                        .saturating_add(std::mem::size_of::<u64>() as u64),
                )
                .saturating_add(
                    u64::try_from(shards.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(std::mem::size_of::<u64>() as u64),
                )
        } else if sharded_query_shape == "sharded_int4_equality_mixed_column_projection" {
            // TEXT is variable-width. Conservatively bound the result by the complete resident
            // footprint plus one row index per possible match.
            total_resident_bytes.saturating_add(
                u64::try_from(total_rows)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(std::mem::size_of::<u64>() as u64),
            )
        } else {
            0
        };
        let mut decision = RelationalResidentRouteDecisionStatus {
            table: table.name.clone(),
            gpu_id,
            snapshot_generation: None,
            shard_count: shards.len(),
            accepted: false,
            reason: String::new(),
            query_shape: sharded_query_shape,
            cache_state: "Valid".to_string(),
            valid: true,
            has_retained_device_memory: false,
            estimated_rows: total_rows,
            resident_bytes: total_resident_bytes,
            budget_bytes: gpu_id.and_then(|gpu_id| self.relational_residency_budget_bytes(gpu_id)),
            refresh_resident_bytes: None,
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: total_resident_bytes,
            d2h_bytes_estimate,
            d2h_rows_estimate: resident_route_d2h_rows_estimate(select, total_rows),
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        };

        if !matches!(
            decision.query_shape.as_str(),
            "sharded_count_all"
                | "sharded_int4_scalar_aggregate"
                | "int4_filter_group_count"
                | "sharded_int4_equality_count"
                | "sharded_int4_range_count"
                | "sharded_int4_filtered_scalar_aggregate"
                | "sharded_int4_between_scalar_aggregate"
                | "sharded_int4_projection"
                | "sharded_int4_projection_all"
                | "sharded_int4_composite_equality_multi_column_projection"
                | "sharded_int4_equality_projection"
                | "sharded_int4_equality_multi_column_projection"
                | "sharded_int4_equality_mixed_column_projection"
                | "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
                | "sharded_int4_distinct_projection"
                | "sharded_int4_filtered_distinct_projection"
                | "sharded_int4_grouped_aggregate"
                | "sharded_int4_filtered_grouped_aggregate"
                | "sharded_int4_ordered_projection"
        ) {
            decision.cache_state = "Absent".to_string();
            decision.valid = false;
            decision.reason =
                "sharded resident routing currently supports only unfiltered COUNT(*), same-column int4 equality projection, int4 equality multi-column projection, int4 equality SUM, int4 BETWEEN AVG, int4 filtered AVG, int4 filtered MIN, int4 filtered MAX, int4 [filtered] DISTINCT projection, int4 [filtered] grouped aggregate, and int4 ordered projection"
                    .to_string();
            return decision;
        }
        let mut required_int4_columns = BTreeSet::new();
        // R-ver: the UNFILTERED projection (`sharded_int4_projection_all`) needs exactly its
        // projected columns resident — no filter columns (there is no WHERE by shape definition).
        // It shares the multi-column projection's extraction (the filter loops are no-ops here).
        if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_equality_multi_column_projection"
                | "sharded_int4_equality_mixed_column_projection"
        ) || decision.query_shape == "sharded_int4_projection_all"
        {
            match &select.projection {
                SelectProjection::Columns(columns) => {
                    for column in columns {
                        required_int4_columns.insert(column.clone());
                    }
                }
                // R-ver: `SELECT * FROM t` needs EVERY column resident (the classifier already
                // proved all columns are int4 for the `sharded_int4_projection_all` shape).
                SelectProjection::All => {
                    for column in &table.columns {
                        required_int4_columns.insert(column.name.clone());
                    }
                }
                _ => {
                    decision.cache_state = "Absent".to_string();
                    decision.valid = false;
                    decision.reason =
                        "sharded resident routing requires projected columns".to_string();
                    return decision;
                }
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_scalar_aggregate" {
            // FLIP slice: an UNFILTERED scalar aggregate (bare SUM/AVG/MIN/MAX over an int4 column —
            // the single-buffer `int4_scalar_aggregate` shape, remapped). Only the aggregate column is
            // required; there are no filters by shape definition.
            let (SelectProjection::Sum { column }
            | SelectProjection::Avg { column }
            | SelectProjection::Min { column }
            | SelectProjection::Max { column }) = &select.projection
            else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires SUM/AVG/MIN/MAX(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
        } else if decision.query_shape == "sharded_int4_equality_sum" {
            let SelectProjection::Sum { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires SUM(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_between_avg" | "sharded_int4_filtered_avg"
        ) {
            let SelectProjection::Avg { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires AVG(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_filtered_min" {
            let SelectProjection::Min { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires MIN(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_filtered_max" {
            let SelectProjection::Max { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = "sharded resident routing requires MAX(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_grouped_aggregate" | "sharded_int4_filtered_grouped_aggregate"
        ) {
            // Grouped int4 aggregate (S10c slice 2b): the per-shard layout check must cover both the
            // GROUP BY key column AND the aggregated value column, plus every filter column. Mirror the
            // route classifier's group/value extraction (resident_route.rs `resident_route_query_shape`):
            // GroupedCount groups by `column` and counts it; the other grouped projections carry an
            // explicit group/value column pair.
            match &select.projection {
                SelectProjection::GroupedCount { column } => {
                    required_int4_columns.insert(column.clone());
                }
                SelectProjection::GroupedSum {
                    group_column,
                    sum_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(sum_column.clone());
                }
                SelectProjection::GroupedAvg {
                    group_column,
                    avg_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(avg_column.clone());
                }
                SelectProjection::GroupedMin {
                    group_column,
                    min_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(min_column.clone());
                }
                SelectProjection::GroupedMax {
                    group_column,
                    max_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(max_column.clone());
                }
                _ => {
                    decision.cache_state = "Absent".to_string();
                    decision.valid = false;
                    decision.reason =
                        "sharded resident routing requires a grouped int4 aggregate projection"
                            .to_string();
                    return decision;
                }
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_distinct_projection"
                | "sharded_int4_filtered_distinct_projection"
                | "sharded_int4_ordered_projection"
        ) {
            // Single-column DISTINCT / ordered int4 projection (S10c slice 2b): the per-shard layout
            // check must cover the single projected/distinct column plus every filter column. The route
            // classifier accepts only a single projected column for these shapes.
            let SelectProjection::Columns(columns) = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires a single projected int4 column".to_string();
                return decision;
            };
            for column in columns {
                required_int4_columns.insert(column.clone());
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        }
        if shards.is_empty() {
            decision.cache_state = "Absent".to_string();
            decision.valid = false;
            decision.reason = "relation has no resident shards".to_string();
            return decision;
        }

        let mut has_all_device_memory = true;
        for shard in shards {
            let memory_pressure_active = self
                .router
                .runtime()
                .snapshot()
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            let valid = shard.is_valid(memory_pressure_active);
            decision.valid &= valid;
            if memory_pressure_active || shard.invalidated_by_memory_pressure {
                decision.cache_state = "InvalidatedByMemoryPressure".to_string();
            } else if shard.invalidated_by_txn_id.is_some() || shard.invalidated_at_index.is_some()
            {
                decision.cache_state = "Invalidated".to_string();
            }
            if shard.schema != table.schema || shard.table != table.name {
                decision.reason =
                    "resident shard no longer matches catalog table identity".to_string();
                return decision;
            }
            // D4: the planner's device check reads the loaded descriptor (advisory — execution
            // re-validates from its own snapshot).
            if shard.device_memory.is_none() {
                has_all_device_memory = false;
            }
            // TYPE-COVERAGE #14: a required column must sit in SOME device section the general
            // executor + recompaction gather serve — int4 / int8 / b128 (Numeric/Uuid) / bool bitmap.
            // (Filter columns are still int4 by shape definition; only PROJECTED columns can be
            // i64/b128/bool.) Text required columns are never classified here.
            if !required_int4_columns.is_empty()
                && required_int4_columns.iter().any(|column| {
                    !shard.resident_device_int4_columns.contains(column)
                        && !shard.resident_device_int8_columns.contains(column)
                        && !shard.resident_device_numeric_columns.contains(column)
                        && !shard
                            .resident_device_bool_columns
                            .iter()
                            .any(|b| &b.name == column)
                        && !shard
                            .resident_device_text_columns
                            .iter()
                            .any(|t| &t.name == column)
                })
            {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                // (Message kept as "int4 projection layout" for the p8 route tests; the check now
                // also accepts i64/b128 sections — a truly-missing column still rejects here.)
                decision.reason = format!(
                    "resident shard {} lacks required int4 projection layout",
                    shard.shard_id
                );
                return decision;
            }
        }
        decision.has_retained_device_memory = has_all_device_memory;
        if !decision.valid {
            decision.reason = format!("resident shard set is {}", decision.cache_state);
        } else if !has_all_device_memory {
            decision.reason = "resident shard set has missing retained device memory".to_string();
        } else {
            decision.accepted = true;
            decision.reason = "sharded resident route accepted".to_string();
        }
        decision
    }

    pub(crate) fn relational_residency_status(&self) -> RelationalResidencyStatus {
        // Stage 3 — blocker #2: iterate a pinned snapshot generation (the per-table `last_decision` it
        // joins to still lives on the resident cache and is read via `&self` inside the closure).
        let snapshots_guard = self.read_state.residency.snapshots.load();
        let mut tables = snapshots_guard
            .values()
            .map(|entry| {
                let snapshot = &entry.descriptor;
                let memory_pressure_active = self
                    .router
                    .runtime()
                    .snapshot()
                    .memory_pressured_gpu_ids
                    .contains(&snapshot.gpu_id);
                let last_decision = self
                    .ddl_catalog()
                    .relational_resident_cache
                    .last_decision(&snapshot.table)
                    .cloned();
                let last_decision = last_decision.as_ref();
                let cache_state =
                    Self::relational_snapshot_cache_state(snapshot, memory_pressure_active);
                RelationalResidencyTableStatus {
                    schema: snapshot.schema.clone(),
                    table: snapshot.table.clone(),
                    gpu_id: snapshot.gpu_id,
                    snapshot_generation: snapshot.generation,
                    cache_state: cache_state.to_string(),
                    row_count: snapshot.row_count,
                    column_count: snapshot.column_count,
                    resident_bytes: snapshot.resident_bytes,
                    valid_through_index: snapshot.valid_through_index,
                    valid: snapshot.invalidated_by_txn_id.is_none()
                        && snapshot.invalidated_at_index.is_none()
                        && !snapshot.invalidated_by_memory_pressure
                        && !memory_pressure_active,
                    invalidated_by_txn_id: snapshot.invalidated_by_txn_id,
                    invalidated_at_index: snapshot.invalidated_at_index,
                    invalidated_by_memory_pressure: snapshot.invalidated_by_memory_pressure,
                    memory_pressure_active,
                    admission_budget_bytes: snapshot.admission_budget_bytes,
                    resident_bytes_after_admission: snapshot.resident_bytes_after_admission,
                    evicted_tables_on_admission: snapshot.evicted_tables_on_admission.clone(),
                    last_decision_accepted: last_decision.map(|decision| decision.accepted),
                    last_decision_reason: last_decision.map(|decision| decision.reason.clone()),
                    last_decision_current_bytes_before: last_decision
                        .map(|decision| decision.current_bytes_before),
                    last_decision_current_bytes_after: last_decision
                        .map(|decision| decision.current_bytes_after),
                    device_memory_proof: snapshot.device_memory_proof.clone(),
                }
            })
            .collect::<Vec<_>>();
        tables.sort_by(|left, right| {
            left.gpu_id
                .cmp(&right.gpu_id)
                .then_with(|| left.schema.cmp(&right.schema))
                .then_with(|| left.table.cmp(&right.table))
        });

        let mut resident_bytes_by_gpu = BTreeMap::new();
        for table in &tables {
            *resident_bytes_by_gpu.entry(table.gpu_id).or_insert(0) += table.resident_bytes;
        }

        RelationalResidencyStatus {
            tables,
            latest_route_decisions: self
                .read_state
                .route_telemetry
                .route_decisions()
                .values()
                .cloned()
                .collect(),
            resident_bytes_by_gpu,
            budget_bytes_by_gpu: self
                .ddl_catalog()
                .relational_resident_cache
                .budget_bytes_by_gpu
                .clone(),
        }
    }
}
