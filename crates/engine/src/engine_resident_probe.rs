//! Resident-device-memory probe execution (P0 §9.6 decomposition, behavior-
//! preserving): a focused `impl Engine` block holding the GPU resident-route
//! execution surface — every execute_relational_*_with_resident_device_memory_probe
//! method (counts, filtered/partitioned counts, scalar + grouped aggregates,
//! projections incl. equality/multi-column/distinct/ordered), the multi-column
//! projection batch path, plus the select-with-backend dispatch and route-
//! execution observation counter they feed.

use super::*;

impl Engine {
    pub fn execute_relational_count_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || !matches!(select.projection, SelectProjection::CountAll)
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.filter.is_some()
            || !select.filters.is_empty()
            || !select.filter_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory query proof currently supports only unfiltered SELECT COUNT(*)"
                    .to_string(),
            )));
        }
        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
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
        let snapshot_gpu_id = snapshot.gpu_id;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let lookup_started = Instant::now();
        let row_count = device_memory
            .count_rows_from_header()
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let lookup_micros = lookup_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        if row_count != snapshot.row_count as u64 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory row-count proof returned {row_count}, expected {}",
                snapshot.row_count
            ))));
        }
        let count = i64::try_from(row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory row count {row_count} exceeds supported COUNT(*) result range"
            )))
        })?;
        self.read_state
            .route_telemetry
            .record_route_device_lookup_micros(&table.name, lookup_micros, 1);

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![SqlValue::Int8(count)]],
            planned_target: DeviceTarget::Gpu(snapshot_gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot_gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_partitioned_count_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || !matches!(select.projection, SelectProjection::CountAll)
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.filter.is_some()
            || !select.filters.is_empty()
            || !select.filter_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident device-memory query proof currently supports only unfiltered SELECT COUNT(*)"
                    .to_string(),
            )));
        }
        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let partitions = self
            .read_state
            .residency
            .partitions
            .load()
            .get(&table.name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident partitions",
                    table.name
                )))
            })?;
        if partitions.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no resident partitions",
                table.name
            ))));
        }

        let snapshot = self.router.runtime().snapshot();
        let mut total_count = 0_u64;
        let mut lookup_micros = 0_u64;
        let mut gpu_id = partitions[0].gpu_id;
        for partition in &partitions {
            if partition.schema != table.schema || partition.table != table.name {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition no longer matches catalog table identity".to_string(),
                )));
            }
            let memory_pressure_active = snapshot
                .memory_pressured_gpu_ids
                .contains(&partition.gpu_id);
            if !partition.is_valid(memory_pressure_active) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident partition {} is invalid",
                    partition.partition_id
                ))));
            }
            let device_memory = self
                .read_state
                .residency
                .partition_device_memory
                .get(&(table.name.clone(), partition.partition_id))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident partition {} has no retained device memory",
                        partition.partition_id
                    )))
                })?;
            let lookup_started = Instant::now();
            let row_count = device_memory
                .count_rows_from_header()
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            lookup_micros = lookup_micros.saturating_add(
                lookup_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            if row_count != partition.row_count as u64 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident partition {} row-count proof returned {row_count}, expected {}",
                    partition.partition_id, partition.row_count
                ))));
            }
            total_count = total_count.checked_add(row_count).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "partitioned resident count overflowed".to_string(),
                ))
            })?;
            gpu_id = partition.gpu_id;
        }
        let count = i64::try_from(total_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "partitioned resident row count {total_count} exceeds supported COUNT(*) result range"
            )))
        })?;
        self.read_state
            .route_telemetry
            .record_route_device_lookup_micros(&table.name, lookup_micros, partitions.len());

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![SqlValue::Int8(count)]],
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_partitioned_equality_projection_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.selected_indexes.len() != 1
            || filter_groups.len() != 1
            || filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality projection proof currently supports only SELECT one_int4_column with one same-column int4 equality predicate"
                    .to_string(),
            )));
        }
        let projection_idx = bound.selected_indexes[0];
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality projection proof currently supports only int4 equality predicates"
                    .to_string(),
            )));
        };
        if op != SelectFilterOp::Eq
            || projection_idx != filter_idx
            || table.columns[projection_idx].ty != SqlType::Int4
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality projection proof currently requires the projected int4 column to be the equality predicate column"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let partitions = self
            .read_state
            .residency
            .partitions
            .load()
            .get(&table.name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident partitions",
                    table.name
                )))
            })?;
        if partitions.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no resident partitions",
                table.name
            ))));
        }

        let snapshot = self.router.runtime().snapshot();
        let mut rows = Vec::new();
        let mut lookup_micros = 0_u64;
        let mut gpu_id = partitions[0].gpu_id;
        for partition in &partitions {
            if partition.schema != table.schema || partition.table != table.name {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition no longer matches catalog table identity".to_string(),
                )));
            }
            let memory_pressure_active = snapshot
                .memory_pressured_gpu_ids
                .contains(&partition.gpu_id);
            if !partition.is_valid(memory_pressure_active) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident partition {} is invalid",
                    partition.partition_id
                ))));
            }
            let device_memory = self
                .read_state
                .residency
                .partition_device_memory
                .get(&(table.name.clone(), partition.partition_id))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident partition {} has no retained device memory",
                        partition.partition_id
                    )))
                })?;
            let byte_offset = resident_partition_int4_column_offset(partition, &table, filter_idx)?;
            let row_count = u64::try_from(partition.row_count).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition row count exceeds retained device-memory proof range"
                        .to_string(),
                ))
            })?;
            let lookup_started = Instant::now();
            let matched_count = device_memory
                .count_i32_equal_from_payload(byte_offset, row_count, needle)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            lookup_micros = lookup_micros.saturating_add(
                lookup_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            let matched_len = usize::try_from(matched_count).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "partitioned resident equality projection count {matched_count} exceeds host result range"
                )))
            })?;
            rows.extend(std::iter::repeat_with(|| vec![SqlValue::Int4(needle)]).take(matched_len));
            gpu_id = partition.gpu_id;
        }
        self.metrics.observe_d2h_bytes(
            u64::try_from(partitions.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(std::mem::size_of::<u64>() as u64),
        );
        self.read_state
            .route_telemetry
            .record_route_device_lookup_micros(&table.name, lookup_micros, rows.len());

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_partitioned_equality_multi_column_projection_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.selected_indexes.len() < 2
            || filter_groups.len() != 1
            || filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality multi-column projection proof currently supports SELECT int4_columns with one int4 equality predicate"
                    .to_string(),
            )));
        }
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality multi-column projection proof currently supports only int4 equality predicates"
                    .to_string(),
            )));
        };
        if op != SelectFilterOp::Eq || table.columns[filter_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality multi-column projection proof currently supports only int4 equality predicates"
                    .to_string(),
            )));
        }
        if bound
            .selected_indexes
            .iter()
            .any(|idx| table.columns[*idx].ty != SqlType::Int4)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality multi-column projection proof currently supports only int4 projection columns"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let partitions = self
            .read_state
            .residency
            .partitions
            .load()
            .get(&table.name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident partitions",
                    table.name
                )))
            })?;
        if partitions.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no resident partitions",
                table.name
            ))));
        }

        let snapshot = self.router.runtime().snapshot();
        let mut rows = Vec::new();
        let mut match_index_micros = 0_u64;
        let mut selected_projection_micros = 0_u64;
        let mut materialization_micros = 0_u64;
        let mut gpu_id = partitions[0].gpu_id;
        let mut int4_d2h_bytes = 0_u64;
        let mut match_index_d2h_bytes = 0_u64;
        for partition in &partitions {
            if partition.schema != table.schema || partition.table != table.name {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition no longer matches catalog table identity".to_string(),
                )));
            }
            let memory_pressure_active = snapshot
                .memory_pressured_gpu_ids
                .contains(&partition.gpu_id);
            if !partition.is_valid(memory_pressure_active) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident partition {} is invalid",
                    partition.partition_id
                ))));
            }
            let device_memory = self
                .read_state
                .residency
                .partition_device_memory
                .get(&(table.name.clone(), partition.partition_id))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident partition {} has no retained device memory",
                        partition.partition_id
                    )))
                })?;
            let filter_offset =
                resident_partition_int4_column_offset(partition, &table, filter_idx)?;
            let row_count = u64::try_from(partition.row_count).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition row count exceeds retained device-memory proof range"
                        .to_string(),
                ))
            })?;
            let match_started = Instant::now();
            let matching_row_indices = device_memory
                .match_i32_equal_row_indices_from_payload(&[(filter_offset, needle)], row_count)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            match_index_micros = match_index_micros.saturating_add(
                match_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            match_index_d2h_bytes = match_index_d2h_bytes.saturating_add(
                u64::try_from(matching_row_indices.len())
                    .unwrap_or(u64::MAX)
                    .saturating_mul(std::mem::size_of::<u64>() as u64)
                    .saturating_add(std::mem::size_of::<u64>() as u64),
            );

            let started = Instant::now();
            let mut column_values = BTreeMap::new();
            for idx in &bound.selected_indexes {
                let byte_offset = resident_partition_int4_column_offset(partition, &table, *idx)?;
                let values = device_memory
                    .project_i32_rows_from_payload(byte_offset, &matching_row_indices)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                if values.len() != matching_row_indices.len() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "partitioned resident equality multi-column selected-row projection column returned {} rows, expected {}",
                        values.len(),
                        matching_row_indices.len()
                    ))));
                }
                column_values.insert(*idx, values);
            }
            let elapsed = started.elapsed();
            selected_projection_micros = selected_projection_micros
                .saturating_add(elapsed.as_micros().try_into().unwrap_or(u64::MAX));
            int4_d2h_bytes = int4_d2h_bytes.saturating_add(
                bound
                    .selected_indexes
                    .len()
                    .checked_mul(matching_row_indices.len())
                    .and_then(|cells| cells.checked_mul(std::mem::size_of::<i32>()))
                    .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .unwrap_or(u64::MAX),
            );
            for _ in 0..bound.selected_indexes.len().saturating_add(1) {
                self.metrics.observe_kernel_exec_ms(
                    elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1),
                );
            }

            let materialize_started = Instant::now();
            for selected_idx in 0..matching_row_indices.len() {
                let row = bound
                    .selected_indexes
                    .iter()
                    .map(|idx| {
                        column_values.get(idx).map(|values| SqlValue::Int4(values[selected_idx])).ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(
                                "partitioned resident equality multi-column projection missing projected column"
                                    .to_string(),
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()?;
                rows.push(row);
            }
            materialization_micros = materialization_micros.saturating_add(
                materialize_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            gpu_id = partition.gpu_id;
        }
        self.metrics
            .observe_d2h_bytes(int4_d2h_bytes.saturating_add(match_index_d2h_bytes));
        self.read_state
            .route_telemetry
            .record_route_device_lookup_micros(&table.name, match_index_micros, rows.len());
        self.read_state
            .route_telemetry
            .record_route_selected_projection_micros(
                &table.name,
                match_index_micros,
                selected_projection_micros,
                materialization_micros,
                rows.len(),
            );

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_partitioned_equality_sum_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let SelectProjection::Sum { column } = &select.projection else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality SUM proof currently supports only SELECT SUM(int4_column)"
                    .to_string(),
            )));
        };
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || filter_groups.len() != 1
            || filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality SUM proof currently supports SELECT SUM(int4_column) with one int4 equality predicate"
                    .to_string(),
            )));
        }
        let aggregate_idx = relational_column_index(&table, column)?;
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality SUM proof currently supports only int4 equality predicates"
                    .to_string(),
            )));
        };
        if op != SelectFilterOp::Eq
            || filter_idx == aggregate_idx
            || table.columns[filter_idx].ty != SqlType::Int4
            || table.columns[aggregate_idx].ty != SqlType::Int4
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident equality SUM proof currently requires an int4 equality predicate on a different int4 column"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let partitions = self
            .read_state
            .residency
            .partitions
            .load()
            .get(&table.name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident partitions",
                    table.name
                )))
            })?;
        if partitions.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no resident partitions",
                table.name
            ))));
        }

        let snapshot = self.router.runtime().snapshot();
        let mut total_sum = 0_i64;
        let mut matched_rows = 0_usize;
        let mut match_index_micros = 0_u64;
        let mut selected_projection_micros = 0_u64;
        let mut reduction_micros = 0_u64;
        let mut gpu_id = partitions[0].gpu_id;
        let mut result_d2h_bytes = 0_u64;
        for partition in &partitions {
            if partition.schema != table.schema || partition.table != table.name {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition no longer matches catalog table identity".to_string(),
                )));
            }
            let memory_pressure_active = snapshot
                .memory_pressured_gpu_ids
                .contains(&partition.gpu_id);
            if !partition.is_valid(memory_pressure_active) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident partition {} is invalid",
                    partition.partition_id
                ))));
            }
            let device_memory = self
                .read_state
                .residency
                .partition_device_memory
                .get(&(table.name.clone(), partition.partition_id))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident partition {} has no retained device memory",
                        partition.partition_id
                    )))
                })?;
            let filter_offset =
                resident_partition_int4_column_offset(partition, &table, filter_idx)?;
            let aggregate_offset =
                resident_partition_int4_column_offset(partition, &table, aggregate_idx)?;
            let row_count = u64::try_from(partition.row_count).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition row count exceeds retained device-memory proof range"
                        .to_string(),
                ))
            })?;
            let match_started = Instant::now();
            let matching_row_indices = device_memory
                .match_i32_equal_row_indices_from_payload(&[(filter_offset, needle)], row_count)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            match_index_micros = match_index_micros.saturating_add(
                match_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            result_d2h_bytes = result_d2h_bytes.saturating_add(
                u64::try_from(matching_row_indices.len())
                    .unwrap_or(u64::MAX)
                    .saturating_mul(std::mem::size_of::<u64>() as u64)
                    .saturating_add(std::mem::size_of::<u64>() as u64),
            );

            let projection_started = Instant::now();
            let values = device_memory
                .project_i32_rows_from_payload(aggregate_offset, &matching_row_indices)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            selected_projection_micros = selected_projection_micros.saturating_add(
                projection_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            if values.len() != matching_row_indices.len() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "partitioned resident equality SUM projection returned {} rows, expected {}",
                    values.len(),
                    matching_row_indices.len()
                ))));
            }
            result_d2h_bytes = result_d2h_bytes.saturating_add(
                values
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .unwrap_or(u64::MAX),
            );
            for _ in 0..2 {
                self.metrics.observe_kernel_exec_ms(
                    projection_started
                        .elapsed()
                        .as_millis()
                        .try_into()
                        .unwrap_or(u64::MAX)
                        .max(1),
                );
            }

            let reduction_started = Instant::now();
            for value in values {
                total_sum = total_sum.checked_add(i64::from(value)).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "partitioned resident equality SUM overflowed".to_string(),
                    ))
                })?;
            }
            reduction_micros = reduction_micros.saturating_add(
                reduction_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            matched_rows = matched_rows.saturating_add(matching_row_indices.len());
            gpu_id = partition.gpu_id;
        }
        result_d2h_bytes = result_d2h_bytes.saturating_add(std::mem::size_of::<i64>() as u64);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.read_state
            .route_telemetry
            .record_route_selected_projection_micros(
                &table.name,
                match_index_micros,
                selected_projection_micros,
                reduction_micros,
                matched_rows,
            );

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![SqlValue::Int8(total_sum)]],
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_partitioned_between_avg_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let SelectProjection::Avg { column } = &select.projection else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident BETWEEN AVG proof currently supports only SELECT AVG(int4_column)"
                    .to_string(),
            )));
        };
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || filter_groups.len() != 1
            || filter_groups[0].len() != 2
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident BETWEEN AVG proof currently supports SELECT AVG(int4_column) with one int4 BETWEEN predicate"
                    .to_string(),
            )));
        }
        let aggregate_idx = relational_column_index(&table, column)?;
        let mut filter_idx = None;
        let mut lower = None;
        let mut upper = None;
        for (idx, op, value) in &filter_groups[0] {
            if filter_idx
                .replace(*idx)
                .is_some_and(|existing| existing != *idx)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "partitioned resident BETWEEN AVG proof requires both range bounds to target the same column"
                        .to_string(),
                )));
            }
            let SqlValue::Int4(value) = value else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "partitioned resident BETWEEN AVG proof supports only int4 bounds".to_string(),
                )));
            };
            match op {
                SelectFilterOp::Gte => lower = Some(*value),
                SelectFilterOp::Lte => upper = Some(*value),
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "partitioned resident BETWEEN AVG proof currently supports only inclusive int4 bounds"
                            .to_string(),
                    )));
                }
            }
        }
        let filter_idx = filter_idx.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident BETWEEN AVG proof requires an int4 predicate column"
                    .to_string(),
            ))
        })?;
        let lower = lower.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident BETWEEN AVG proof requires a lower inclusive bound"
                    .to_string(),
            ))
        })?;
        let upper = upper.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident BETWEEN AVG proof requires an upper inclusive bound"
                    .to_string(),
            ))
        })?;
        if filter_idx == aggregate_idx
            || table.columns[filter_idx].ty != SqlType::Int4
            || table.columns[aggregate_idx].ty != SqlType::Int4
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident BETWEEN AVG proof currently requires an int4 BETWEEN predicate on a different int4 column"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let partitions = self
            .read_state
            .residency
            .partitions
            .load()
            .get(&table.name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident partitions",
                    table.name
                )))
            })?;
        if partitions.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no resident partitions",
                table.name
            ))));
        }

        let snapshot = self.router.runtime().snapshot();
        let mut total_sum = 0_i128;
        let mut matched_rows = 0_usize;
        let mut match_index_micros = 0_u64;
        let mut selected_projection_micros = 0_u64;
        let mut reduction_micros = 0_u64;
        let mut gpu_id = partitions[0].gpu_id;
        let mut result_d2h_bytes = 0_u64;
        for partition in &partitions {
            if partition.schema != table.schema || partition.table != table.name {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition no longer matches catalog table identity".to_string(),
                )));
            }
            let memory_pressure_active = snapshot
                .memory_pressured_gpu_ids
                .contains(&partition.gpu_id);
            if !partition.is_valid(memory_pressure_active) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident partition {} is invalid",
                    partition.partition_id
                ))));
            }
            let device_memory = self
                .read_state
                .residency
                .partition_device_memory
                .get(&(table.name.clone(), partition.partition_id))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident partition {} has no retained device memory",
                        partition.partition_id
                    )))
                })?;
            let filter_offset =
                resident_partition_int4_column_offset(partition, &table, filter_idx)?;
            let aggregate_offset =
                resident_partition_int4_column_offset(partition, &table, aggregate_idx)?;
            let row_count = u64::try_from(partition.row_count).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition row count exceeds retained device-memory proof range"
                        .to_string(),
                ))
            })?;
            let match_started = Instant::now();
            let matching_row_indices = device_memory
                .match_i32_between_row_indices_from_payload(filter_offset, row_count, lower, upper)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            match_index_micros = match_index_micros.saturating_add(
                match_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            result_d2h_bytes = result_d2h_bytes.saturating_add(
                u64::try_from(partition.row_count)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(std::mem::size_of::<i32>() as u64)
                    .saturating_add(
                        u64::try_from(matching_row_indices.len())
                            .unwrap_or(u64::MAX)
                            .saturating_mul(std::mem::size_of::<u64>() as u64),
                    ),
            );

            let projection_started = Instant::now();
            let values = device_memory
                .project_i32_rows_from_payload(aggregate_offset, &matching_row_indices)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            selected_projection_micros = selected_projection_micros.saturating_add(
                projection_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            if values.len() != matching_row_indices.len() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "partitioned resident BETWEEN AVG projection returned {} rows, expected {}",
                    values.len(),
                    matching_row_indices.len()
                ))));
            }
            result_d2h_bytes = result_d2h_bytes.saturating_add(
                values
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .unwrap_or(u64::MAX),
            );
            for _ in 0..2 {
                self.metrics.observe_kernel_exec_ms(
                    projection_started
                        .elapsed()
                        .as_millis()
                        .try_into()
                        .unwrap_or(u64::MAX)
                        .max(1),
                );
            }

            let reduction_started = Instant::now();
            for value in values {
                total_sum = total_sum.checked_add(i128::from(value)).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "partitioned resident BETWEEN AVG overflowed".to_string(),
                    ))
                })?;
            }
            reduction_micros = reduction_micros.saturating_add(
                reduction_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            matched_rows = matched_rows.saturating_add(matching_row_indices.len());
            gpu_id = partition.gpu_id;
        }
        result_d2h_bytes = result_d2h_bytes
            .saturating_add(std::mem::size_of::<u64>() as u64)
            .saturating_add(std::mem::size_of::<i64>() as u64);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.read_state
            .route_telemetry
            .record_route_selected_projection_micros(
                &table.name,
                match_index_micros,
                selected_projection_micros,
                reduction_micros,
                matched_rows,
            );

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![average_sql_value(total_sum, matched_rows)]],
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_partitioned_filtered_max_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let SelectProjection::Max { column } = &select.projection else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered MAX proof currently supports only SELECT MAX(int4_column)"
                    .to_string(),
            )));
        };
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || filter_groups.len() != 1
            || filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered MAX proof currently supports SELECT MAX(int4_column) with one same-column int4 comparison predicate"
                    .to_string(),
            )));
        }
        let aggregate_idx = relational_column_index(&table, column)?;
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        let Some(comparison) = resident_device_i32_comparison(op) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered MAX proof currently supports only non-equality int4 comparisons"
                    .to_string(),
            )));
        };
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered MAX proof currently supports only int4 comparison literals"
                    .to_string(),
            )));
        };
        if filter_idx != aggregate_idx || table.columns[aggregate_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered MAX proof currently requires the predicate column to match the MAX int4 column"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let partitions = self
            .read_state
            .residency
            .partitions
            .load()
            .get(&table.name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident partitions",
                    table.name
                )))
            })?;
        if partitions.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no resident partitions",
                table.name
            ))));
        }

        let snapshot = self.router.runtime().snapshot();
        let mut max_value = None;
        let mut matched_rows = 0_usize;
        let mut match_index_micros = 0_u64;
        let mut reduction_micros = 0_u64;
        let mut gpu_id = partitions[0].gpu_id;
        let mut result_d2h_bytes = 0_u64;
        for partition in &partitions {
            if partition.schema != table.schema || partition.table != table.name {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition no longer matches catalog table identity".to_string(),
                )));
            }
            let memory_pressure_active = snapshot
                .memory_pressured_gpu_ids
                .contains(&partition.gpu_id);
            if !partition.is_valid(memory_pressure_active) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident partition {} is invalid",
                    partition.partition_id
                ))));
            }
            let device_memory = self
                .read_state
                .residency
                .partition_device_memory
                .get(&(table.name.clone(), partition.partition_id))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident partition {} has no retained device memory",
                        partition.partition_id
                    )))
                })?;
            let aggregate_offset =
                resident_partition_int4_column_offset(partition, &table, aggregate_idx)?;
            let row_count = u64::try_from(partition.row_count).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition row count exceeds retained device-memory proof range"
                        .to_string(),
                ))
            })?;
            let match_started = Instant::now();
            let stats = device_memory
                .filtered_stats_i32_compare_from_payload(
                    aggregate_offset,
                    row_count,
                    needle,
                    comparison,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            match_index_micros = match_index_micros.saturating_add(
                match_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            result_d2h_bytes = result_d2h_bytes.saturating_add(
                std::mem::size_of::<u64>() as u64
                    + std::mem::size_of::<i64>() as u64
                    + (2 * std::mem::size_of::<i32>()) as u64
                    + std::mem::size_of::<u64>() as u64,
            );
            self.metrics.observe_kernel_exec_ms(
                match_started
                    .elapsed()
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX)
                    .max(1),
            );

            let reduction_started = Instant::now();
            if let Some(partition_max) = stats.max {
                max_value = Some(
                    max_value.map_or(partition_max, |current: i32| current.max(partition_max)),
                );
            }
            matched_rows = matched_rows.saturating_add(
                usize::try_from(stats.count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "partitioned resident filtered MAX count {} exceeds matched-row telemetry range",
                        stats.count
                    )))
                })?,
            );
            reduction_micros = reduction_micros.saturating_add(
                reduction_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            gpu_id = partition.gpu_id;
        }
        result_d2h_bytes = result_d2h_bytes.saturating_add(std::mem::size_of::<i32>() as u64);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.read_state
            .route_telemetry
            .record_route_selected_projection_micros(
                &table.name,
                match_index_micros,
                0,
                reduction_micros,
                matched_rows,
            );

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![max_value
                .map(SqlValue::Int4)
                .unwrap_or_else(|| SqlValue::Text(String::new()))]],
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_partitioned_filtered_avg_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let SelectProjection::Avg { column } = &select.projection else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered AVG proof currently supports only SELECT AVG(int4_column)"
                    .to_string(),
            )));
        };
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || filter_groups.len() != 1
            || filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered AVG proof currently supports SELECT AVG(int4_column) with one same-column int4 comparison predicate"
                    .to_string(),
            )));
        }
        let aggregate_idx = relational_column_index(&table, column)?;
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        let Some(comparison) = resident_device_i32_comparison(op) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered AVG proof currently supports only non-equality int4 comparisons"
                    .to_string(),
            )));
        };
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered AVG proof currently supports only int4 comparison literals"
                    .to_string(),
            )));
        };
        if filter_idx != aggregate_idx || table.columns[aggregate_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered AVG proof currently requires the predicate column to match the AVG int4 column"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let partitions = self
            .read_state
            .residency
            .partitions
            .load()
            .get(&table.name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident partitions",
                    table.name
                )))
            })?;
        if partitions.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no resident partitions",
                table.name
            ))));
        }

        let snapshot = self.router.runtime().snapshot();
        let mut total_sum = 0_i128;
        let mut matched_rows = 0_usize;
        let mut match_index_micros = 0_u64;
        let mut reduction_micros = 0_u64;
        let mut gpu_id = partitions[0].gpu_id;
        let mut result_d2h_bytes = 0_u64;
        for partition in &partitions {
            if partition.schema != table.schema || partition.table != table.name {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition no longer matches catalog table identity".to_string(),
                )));
            }
            let memory_pressure_active = snapshot
                .memory_pressured_gpu_ids
                .contains(&partition.gpu_id);
            if !partition.is_valid(memory_pressure_active) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident partition {} is invalid",
                    partition.partition_id
                ))));
            }
            let device_memory = self
                .read_state
                .residency
                .partition_device_memory
                .get(&(table.name.clone(), partition.partition_id))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident partition {} has no retained device memory",
                        partition.partition_id
                    )))
                })?;
            let aggregate_offset =
                resident_partition_int4_column_offset(partition, &table, aggregate_idx)?;
            let row_count = u64::try_from(partition.row_count).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition row count exceeds retained device-memory proof range"
                        .to_string(),
                ))
            })?;
            let match_started = Instant::now();
            let stats = device_memory
                .filtered_stats_i32_compare_from_payload(
                    aggregate_offset,
                    row_count,
                    needle,
                    comparison,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            match_index_micros = match_index_micros.saturating_add(
                match_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            result_d2h_bytes = result_d2h_bytes.saturating_add(
                std::mem::size_of::<u64>() as u64
                    + std::mem::size_of::<i64>() as u64
                    + (2 * std::mem::size_of::<i32>()) as u64
                    + std::mem::size_of::<u64>() as u64,
            );
            self.metrics.observe_kernel_exec_ms(
                match_started
                    .elapsed()
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX)
                    .max(1),
            );

            let reduction_started = Instant::now();
            total_sum = total_sum
                .checked_add(i128::from(stats.sum))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "partitioned resident filtered AVG overflowed".to_string(),
                    ))
                })?;
            matched_rows = matched_rows.saturating_add(
                usize::try_from(stats.count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "partitioned resident filtered AVG count {} exceeds matched-row telemetry range",
                        stats.count
                    )))
                })?,
            );
            reduction_micros = reduction_micros.saturating_add(
                reduction_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            gpu_id = partition.gpu_id;
        }
        result_d2h_bytes = result_d2h_bytes
            .saturating_add(std::mem::size_of::<u64>() as u64)
            .saturating_add(std::mem::size_of::<i64>() as u64);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.read_state
            .route_telemetry
            .record_route_selected_projection_micros(
                &table.name,
                match_index_micros,
                0,
                reduction_micros,
                matched_rows,
            );

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![average_sql_value(total_sum, matched_rows)]],
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_partitioned_filtered_min_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let SelectProjection::Min { column } = &select.projection else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered MIN proof currently supports only SELECT MIN(int4_column)"
                    .to_string(),
            )));
        };
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || filter_groups.len() != 1
            || filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered MIN proof currently supports SELECT MIN(int4_column) with one same-column int4 comparison predicate"
                    .to_string(),
            )));
        }
        let aggregate_idx = relational_column_index(&table, column)?;
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        let Some(comparison) = resident_device_i32_comparison(op) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered MIN proof currently supports only non-equality int4 comparisons"
                    .to_string(),
            )));
        };
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered MIN proof currently supports only int4 comparison literals"
                    .to_string(),
            )));
        };
        if filter_idx != aggregate_idx || table.columns[aggregate_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "partitioned resident filtered MIN proof currently requires the predicate column to match the MIN int4 column"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let partitions = self
            .read_state
            .residency
            .partitions
            .load()
            .get(&table.name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident partitions",
                    table.name
                )))
            })?;
        if partitions.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no resident partitions",
                table.name
            ))));
        }

        let snapshot = self.router.runtime().snapshot();
        let mut min_value = None;
        let mut matched_rows = 0_usize;
        let mut match_index_micros = 0_u64;
        let mut reduction_micros = 0_u64;
        let mut gpu_id = partitions[0].gpu_id;
        let mut result_d2h_bytes = 0_u64;
        for partition in &partitions {
            if partition.schema != table.schema || partition.table != table.name {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition no longer matches catalog table identity".to_string(),
                )));
            }
            let memory_pressure_active = snapshot
                .memory_pressured_gpu_ids
                .contains(&partition.gpu_id);
            if !partition.is_valid(memory_pressure_active) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident partition {} is invalid",
                    partition.partition_id
                ))));
            }
            let device_memory = self
                .read_state
                .residency
                .partition_device_memory
                .get(&(table.name.clone(), partition.partition_id))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident partition {} has no retained device memory",
                        partition.partition_id
                    )))
                })?;
            let aggregate_offset =
                resident_partition_int4_column_offset(partition, &table, aggregate_idx)?;
            let row_count = u64::try_from(partition.row_count).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident partition row count exceeds retained device-memory proof range"
                        .to_string(),
                ))
            })?;
            let match_started = Instant::now();
            let stats = device_memory
                .filtered_stats_i32_compare_from_payload(
                    aggregate_offset,
                    row_count,
                    needle,
                    comparison,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            match_index_micros = match_index_micros.saturating_add(
                match_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            result_d2h_bytes = result_d2h_bytes.saturating_add(
                std::mem::size_of::<u64>() as u64
                    + std::mem::size_of::<i64>() as u64
                    + (2 * std::mem::size_of::<i32>()) as u64
                    + std::mem::size_of::<u64>() as u64,
            );
            self.metrics.observe_kernel_exec_ms(
                match_started
                    .elapsed()
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX)
                    .max(1),
            );

            let reduction_started = Instant::now();
            if let Some(partition_min) = stats.min {
                min_value = Some(
                    min_value.map_or(partition_min, |current: i32| current.min(partition_min)),
                );
            }
            matched_rows = matched_rows.saturating_add(
                usize::try_from(stats.count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "partitioned resident filtered MIN count {} exceeds matched-row telemetry range",
                        stats.count
                    )))
                })?,
            );
            reduction_micros = reduction_micros.saturating_add(
                reduction_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            gpu_id = partition.gpu_id;
        }
        result_d2h_bytes = result_d2h_bytes.saturating_add(std::mem::size_of::<i32>() as u64);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.read_state
            .route_telemetry
            .record_route_selected_projection_micros(
                &table.name,
                match_index_micros,
                0,
                reduction_micros,
                matched_rows,
            );

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![min_value
                .map(SqlValue::Int4)
                .unwrap_or_else(|| SqlValue::Text(String::new()))]],
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_filtered_count_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || !matches!(select.projection, SelectProjection::CountAll)
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.filter_groups.len() != 1
            || bound.filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered count proof currently supports only SELECT COUNT(*) with one int4 equality predicate"
                    .to_string(),
            )));
        }
        let (filter_idx, op, value) = bound.filter_groups[0][0].clone();
        if op != SelectFilterOp::Eq {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered count proof currently supports only int4 equality predicates"
                    .to_string(),
            )));
        }
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered count proof currently supports only int4 equality predicates"
                    .to_string(),
            )));
        };
        if table.columns[filter_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered count proof currently supports only int4 equality predicates"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let filtered_count = device_memory
            .count_i32_equal_from_payload(byte_offset, row_count, needle)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let count = i64::try_from(filtered_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory filtered count {filtered_count} exceeds supported COUNT(*) result range"
            )))
        })?;

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![SqlValue::Int8(count)]],
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_text_prefix_count_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || !matches!(select.projection, SelectProjection::CountAll)
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.filter_groups.len() != 1
            || bound.filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory text-prefix count proof currently supports only SELECT COUNT(*) with one text prefix LIKE predicate"
                    .to_string(),
            )));
        }
        let (filter_idx, op, value) = bound.filter_groups[0][0].clone();
        if op != SelectFilterOp::LikePrefix {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory text-prefix count proof currently supports only text prefix LIKE predicates"
                    .to_string(),
            )));
        }
        let SqlValue::Text(prefix) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory text-prefix count proof currently supports only text prefix LIKE predicates"
                    .to_string(),
            )));
        };

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let layout = resident_device_text_column_layout(&snapshot, &table, filter_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let matched_count = device_memory
            .count_text_prefix_from_payload(
                layout.offsets_byte_offset,
                layout.bytes_byte_offset,
                layout.bytes_len,
                row_count,
                prefix.as_bytes(),
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        self.metrics.observe_d2h_bytes(
            (row_count + 1)
                .saturating_mul(std::mem::size_of::<u64>() as u64)
                .saturating_add(layout.bytes_len),
        );
        let count = i64::try_from(matched_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory text-prefix count {matched_count} exceeds supported COUNT(*) result range"
            )))
        })?;

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![SqlValue::Int8(count)]],
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_membership_count_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || !matches!(select.projection, SelectProjection::CountAll)
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.filter_groups.len() < 2
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory membership count proof currently supports only SELECT COUNT(*) with one int4 IN membership predicate"
                    .to_string(),
            )));
        }

        let mut filter_idx = None;
        let mut needles = BTreeSet::new();
        for group in &bound.filter_groups {
            if group.len() != 1 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory membership count proof currently supports only one int4 IN membership predicate"
                        .to_string(),
                )));
            }
            let (idx, op, value) = group[0].clone();
            if op != SelectFilterOp::Eq {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory membership count proof currently supports only int4 equality membership predicates"
                        .to_string(),
                )));
            }
            if filter_idx
                .replace(idx)
                .is_some_and(|existing| existing != idx)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory membership count proof requires all membership values to target the same column"
                        .to_string(),
                )));
            }
            let SqlValue::Int4(needle) = value else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory membership count proof currently supports only int4 membership literals"
                        .to_string(),
                )));
            };
            needles.insert(needle);
        }
        let filter_idx = filter_idx.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory membership count proof requires at least one membership literal"
                    .to_string(),
            ))
        })?;
        if table.columns[filter_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory membership count proof currently supports only int4 membership predicates"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let needles = needles.into_iter().collect::<Vec<_>>();
        let membership_count = device_memory
            .count_i32_in_from_payload(byte_offset, row_count, &needles)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let count = i64::try_from(membership_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory membership count {membership_count} exceeds supported COUNT(*) result range"
            )))
        })?;

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![SqlValue::Int8(count)]],
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_range_count_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || !matches!(select.projection, SelectProjection::CountAll)
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.filter_groups.len() != 1
            || bound.filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory range count proof currently supports only SELECT COUNT(*) with one int4 range predicate"
                    .to_string(),
            )));
        }
        let (filter_idx, op, value) = bound.filter_groups[0][0].clone();
        let Some(comparison) = resident_device_i32_comparison(op) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory range count proof currently supports only int4 non-equality predicates"
                    .to_string(),
            )));
        };
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory range count proof currently supports only int4 non-equality predicates"
                    .to_string(),
            )));
        };
        if table.columns[filter_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory range count proof currently supports only int4 non-equality predicates"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let filtered_count = device_memory
            .count_i32_compare_from_payload(byte_offset, row_count, needle, comparison)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let count = i64::try_from(filtered_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory range count {filtered_count} exceeds supported COUNT(*) result range"
            )))
        })?;

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![SqlValue::Int8(count)]],
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_between_count_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || !matches!(select.projection, SelectProjection::CountAll)
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.filter_groups.len() != 1
            || bound.filter_groups[0].len() != 2
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN count proof currently supports only SELECT COUNT(*) with one int4 BETWEEN predicate"
                    .to_string(),
            )));
        }

        let mut filter_idx = None;
        let mut lower = None;
        let mut upper = None;
        for (idx, op, value) in bound.filter_groups[0].iter().cloned() {
            if filter_idx
                .replace(idx)
                .is_some_and(|existing| existing != idx)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory BETWEEN count proof requires both range bounds to target the same column"
                        .to_string(),
                )));
            }
            let SqlValue::Int4(value) = value else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory BETWEEN count proof currently supports only int4 bounds"
                        .to_string(),
                )));
            };
            match op {
                SelectFilterOp::Gte => lower = Some(value),
                SelectFilterOp::Lte => upper = Some(value),
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident device-memory BETWEEN count proof currently supports only inclusive int4 bounds"
                            .to_string(),
                    )));
                }
            }
        }
        let filter_idx = filter_idx.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN count proof requires an int4 predicate column"
                    .to_string(),
            ))
        })?;
        let lower = lower.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN count proof requires a lower inclusive bound"
                    .to_string(),
            ))
        })?;
        let upper = upper.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN count proof requires an upper inclusive bound"
                    .to_string(),
            ))
        })?;
        if table.columns[filter_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN count proof currently supports only int4 predicates"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let between_count = device_memory
            .count_i32_between_from_payload(byte_offset, row_count, lower, upper)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        self.metrics
            .observe_d2h_bytes(2 * std::mem::size_of::<u64>() as u64);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));
        let count = i64::try_from(between_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory BETWEEN count {between_count} exceeds supported COUNT(*) result range"
            )))
        })?;

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![SqlValue::Int8(count)]],
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_filter_group_count_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || !matches!(select.projection, SelectProjection::CountAll)
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.filter_groups.is_empty()
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filter-group count proof currently supports only SELECT COUNT(*) with int4 WHERE filter groups"
                    .to_string(),
            )));
        }
        for group in &bound.filter_groups {
            if group.is_empty() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory filter-group count proof requires non-empty filter groups"
                        .to_string(),
                )));
            }
            for (idx, op, value) in group {
                if *op == SelectFilterOp::LikePrefix
                    || table
                        .columns
                        .get(*idx)
                        .is_none_or(|column| column.ty != SqlType::Int4)
                    || !matches!(value, SqlValue::Int4(_))
                {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident device-memory filter-group count proof currently supports only int4 literal predicates"
                            .to_string(),
                    )));
                }
            }
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let predicate_columns = bound
            .filter_groups
            .iter()
            .flat_map(|group| group.iter().map(|(idx, _op, _value)| *idx))
            .collect::<BTreeSet<_>>();
        let started = Instant::now();
        let mut column_values = BTreeMap::new();
        for idx in predicate_columns {
            let byte_offset = resident_device_int4_column_offset(&snapshot, &table, idx)?;
            let values = device_memory
                .project_i32_from_payload(byte_offset, row_count)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            if values.len() != snapshot.row_count {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident device-memory filter-group column returned {} rows, expected {}",
                    values.len(),
                    snapshot.row_count
                ))));
            }
            column_values.insert(idx, values);
        }
        let elapsed = started.elapsed();

        let mut matched_count = 0_u64;
        for row_idx in 0..snapshot.row_count {
            let row_matches = bound.filter_groups.iter().any(|group| {
                group.iter().all(|(idx, op, value)| {
                    let Some(values) = column_values.get(idx) else {
                        return false;
                    };
                    let left = SqlValue::Int4(values[row_idx]);
                    select_filter_matches(&left, *op, value)
                })
            });
            if row_matches {
                matched_count = matched_count.saturating_add(1);
            }
        }
        self.metrics.observe_d2h_bytes(
            (column_values.len() as u64)
                .saturating_mul(row_count)
                .saturating_mul(std::mem::size_of::<i32>() as u64),
        );
        for _ in 0..column_values.len() {
            self.metrics
                .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));
        }
        let count = i64::try_from(matched_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory filter-group count {matched_count} exceeds supported COUNT(*) result range"
            )))
        })?;

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![SqlValue::Int8(count)]],
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_sum_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let SelectProjection::Sum { column } = &select.projection else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory SUM proof currently supports only SELECT SUM(int4_column)"
                    .to_string(),
            )));
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.filter.is_some()
            || !select.filters.is_empty()
            || !select.filter_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory SUM proof currently supports only unfiltered SELECT SUM(int4_column)"
                    .to_string(),
            )));
        }
        let sum_idx = relational_column_index(&table, column)?;
        if table.columns[sum_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory SUM proof currently supports only int4 columns".to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, sum_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let sum = device_memory
            .sum_i32_from_payload(byte_offset, row_count)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        self.metrics
            .observe_d2h_bytes(std::mem::size_of::<i64>() as u64);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![SqlValue::Int8(sum)]],
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let (aggregate_column, aggregate_name) = match &select.projection {
            SelectProjection::Sum { column } => (column, "SUM"),
            SelectProjection::Avg { column } => (column, "AVG"),
            SelectProjection::Min { column } => (column, "MIN"),
            SelectProjection::Max { column } => (column, "MAX"),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory filtered scalar aggregate proof currently supports only SUM/AVG/MIN/MAX(int4_column)"
                        .to_string(),
                )));
            }
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.filter_groups.len() != 1
            || bound.filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered scalar aggregate proof currently supports only one int4 comparison predicate"
                    .to_string(),
            )));
        }
        let aggregate_idx = relational_column_index(&table, aggregate_column)?;
        let (filter_idx, op, value) = bound.filter_groups[0][0].clone();
        if filter_idx != aggregate_idx {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered scalar aggregate proof currently requires the predicate column to match the aggregate column"
                    .to_string(),
            )));
        }
        let Some(comparison) = resident_device_i32_comparison(op) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered scalar aggregate proof currently supports only non-equality int4 comparisons"
                    .to_string(),
            )));
        };
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered scalar aggregate proof currently supports only int4 comparison literals"
                    .to_string(),
            )));
        };
        if table.columns[aggregate_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory filtered scalar aggregate proof currently supports only int4 columns for {aggregate_name}"
            ))));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        if matches!(select.projection, SelectProjection::Max { .. })
            && resident_device_int4_column_stats(&snapshot, &table, aggregate_idx).is_some_and(
                |stats| resident_i32_comparison_domain_is_empty(stats, needle, comparison),
            )
        {
            self.metrics
                .observe_d2h_bytes(std::mem::size_of::<i64>() as u64);
            return Ok(RelationalSelectResult {
                columns: bound.selected_columns,
                rows: vec![vec![SqlValue::Text(String::new())]],
                planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
                executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
                fallback_reason: None,
                access_path,
            });
        }
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, aggregate_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let stats = device_memory
            .filtered_stats_i32_compare_from_payload(byte_offset, row_count, needle, comparison)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        let result_value = match &select.projection {
            SelectProjection::Sum { .. } => SqlValue::Int8(stats.sum),
            SelectProjection::Avg { .. } => average_sql_value(
                i128::from(stats.sum),
                usize::try_from(stats.count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident device-memory filtered aggregate count {} exceeds AVG result range",
                        stats.count
                    )))
                })?,
            ),
            SelectProjection::Min { .. } => stats
                .min
                .map(SqlValue::Int4)
                .unwrap_or_else(|| SqlValue::Text(String::new())),
            SelectProjection::Max { .. } => stats
                .max
                .map(SqlValue::Int4)
                .unwrap_or_else(|| SqlValue::Text(String::new())),
            _ => unreachable!(),
        };
        let result_d2h_bytes = (std::mem::size_of::<u64>()
            + std::mem::size_of::<i64>()
            + (2 * std::mem::size_of::<i32>())
            + std::mem::size_of::<u64>()) as u64;
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![result_value]],
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let (aggregate_column, aggregate_name) = match &select.projection {
            SelectProjection::Sum { column } => (column, "SUM"),
            SelectProjection::Avg { column } => (column, "AVG"),
            SelectProjection::Min { column } => (column, "MIN"),
            SelectProjection::Max { column } => (column, "MAX"),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory BETWEEN scalar aggregate proof currently supports only SUM/AVG/MIN/MAX(int4_column)"
                        .to_string(),
                )));
            }
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.filter_groups.len() != 1
            || bound.filter_groups[0].len() != 2
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN scalar aggregate proof currently supports only one int4 BETWEEN predicate"
                    .to_string(),
            )));
        }
        let aggregate_idx = relational_column_index(&table, aggregate_column)?;
        let mut filter_idx = None;
        let mut lower = None;
        let mut upper = None;
        for (idx, op, value) in &bound.filter_groups[0] {
            if filter_idx
                .replace(*idx)
                .is_some_and(|existing| existing != *idx)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory BETWEEN scalar aggregate proof requires both range bounds to target the same column"
                        .to_string(),
                )));
            }
            let SqlValue::Int4(value) = value else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory BETWEEN scalar aggregate proof supports only int4 bounds"
                        .to_string(),
                )));
            };
            match op {
                SelectFilterOp::Gte => lower = Some(*value),
                SelectFilterOp::Lte => upper = Some(*value),
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident device-memory BETWEEN scalar aggregate proof currently supports only inclusive int4 bounds"
                            .to_string(),
                    )));
                }
            }
        }
        let filter_idx = filter_idx.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN scalar aggregate proof requires an int4 predicate column"
                    .to_string(),
            ))
        })?;
        if filter_idx != aggregate_idx {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN scalar aggregate proof currently requires the predicate column to match the aggregate column"
                    .to_string(),
            )));
        }
        let lower = lower.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN scalar aggregate proof requires a lower inclusive bound"
                    .to_string(),
            ))
        })?;
        let upper = upper.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN scalar aggregate proof requires an upper inclusive bound"
                    .to_string(),
            ))
        })?;
        if table.columns[aggregate_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory BETWEEN scalar aggregate proof currently supports only int4 columns for {aggregate_name}"
            ))));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, aggregate_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let stats = device_memory
            .stats_i32_between_from_payload(byte_offset, row_count, lower, upper)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        let result_value = match &select.projection {
            SelectProjection::Sum { .. } => SqlValue::Int8(stats.sum),
            SelectProjection::Avg { .. } => average_sql_value(
                i128::from(stats.sum),
                usize::try_from(stats.count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident device-memory BETWEEN aggregate count {} exceeds AVG result range",
                        stats.count
                    )))
                })?,
            ),
            SelectProjection::Min { .. } => stats
                .min
                .map(SqlValue::Int4)
                .unwrap_or_else(|| SqlValue::Text(String::new())),
            SelectProjection::Max { .. } => stats
                .max
                .map(SqlValue::Int4)
                .unwrap_or_else(|| SqlValue::Text(String::new())),
            _ => unreachable!(),
        };
        let result_d2h_bytes = if lower > upper {
            0
        } else {
            (std::mem::size_of::<u64>()
                + std::mem::size_of::<i64>()
                + (2 * std::mem::size_of::<i32>())
                + std::mem::size_of::<u64>()) as u64
        };
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        if lower <= upper {
            self.metrics
                .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));
        }

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![result_value]],
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_scalar_aggregate_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let aggregate_column = match &select.projection {
            SelectProjection::Avg { column }
            | SelectProjection::Min { column }
            | SelectProjection::Max { column } => column,
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory scalar aggregate proof currently supports only AVG/MIN/MAX(int4_column)"
                        .to_string(),
                )));
            }
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.filter.is_some()
            || !select.filters.is_empty()
            || !select.filter_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory scalar aggregate proof currently supports only unfiltered AVG/MIN/MAX(int4_column)"
                    .to_string(),
            )));
        }
        let aggregate_idx = relational_column_index(&table, aggregate_column)?;
        if table.columns[aggregate_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory scalar aggregate proof currently supports only int4 columns"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, aggregate_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let grouped_stats = device_memory
            .grouped_stats_i32_from_payload(byte_offset, byte_offset, row_count)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        let copied_group_count = grouped_stats.len();
        let total_count = grouped_stats.iter().map(|group| group.count).sum::<u64>();
        let total_sum = grouped_stats.iter().map(|group| group.sum).sum::<i64>();
        let result_value = match &select.projection {
            SelectProjection::Avg { .. } => average_sql_value(
                i128::from(total_sum),
                usize::try_from(total_count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident device-memory scalar aggregate count {total_count} exceeds AVG result range"
                    )))
                })?,
            ),
            SelectProjection::Min { .. } => grouped_stats
                .iter()
                .map(|group| group.min)
                .min()
                .map(SqlValue::Int4)
                .unwrap_or_else(|| SqlValue::Text(String::new())),
            SelectProjection::Max { .. } => grouped_stats
                .iter()
                .map(|group| group.max)
                .max()
                .map(SqlValue::Int4)
                .unwrap_or_else(|| SqlValue::Text(String::new())),
            _ => unreachable!(),
        };
        let result_d2h_bytes = copied_group_count
            .checked_mul(
                std::mem::size_of::<i32>()
                    + std::mem::size_of::<u64>()
                    + std::mem::size_of::<i64>()
                    + (2 * std::mem::size_of::<i32>()),
            )
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: vec![vec![result_value]],
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_grouped_sum_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_relational_grouped_aggregate_with_resident_device_memory_probe(select)
    }

    pub fn execute_relational_grouped_aggregate_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let (group_column, value_column) = match &select.projection {
            SelectProjection::GroupedCount { column } => (column, column),
            SelectProjection::GroupedSum {
                group_column,
                sum_column,
            } => (group_column, sum_column),
            SelectProjection::GroupedAvg {
                group_column,
                avg_column,
            } => (group_column, avg_column),
            SelectProjection::GroupedMin {
                group_column,
                min_column,
            } => (group_column, min_column),
            SelectProjection::GroupedMax {
                group_column,
                max_column,
            } => (group_column, max_column),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory grouped aggregate proof currently supports only grouped COUNT/SUM/AVG/MIN/MAX"
                        .to_string(),
                )));
            }
        };
        if select.distinct
            || select.filter.is_some()
            || !select.filters.is_empty()
            || !select.filter_groups.is_empty()
            || select.offset.is_some()
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory grouped aggregate proof currently supports only unfiltered grouped aggregates with optional HAVING, ORDER BY, and LIMIT"
                    .to_string(),
            )));
        }
        let Some(group_by) = &select.group_by else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory grouped aggregate proof requires GROUP BY".to_string(),
            )));
        };
        if !group_by.eq_ignore_ascii_case(group_column) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory grouped aggregate proof requires GROUP BY to match the projected group column"
                    .to_string(),
            )));
        }
        let group_idx = relational_column_index(&table, group_column)?;
        let value_idx = relational_column_index(&table, value_column)?;
        if table.columns[group_idx].ty != SqlType::Int4
            || table.columns[value_idx].ty != SqlType::Int4
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory grouped aggregate proof currently supports only int4 group and value columns"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let group_offset = resident_device_int4_column_offset(&snapshot, &table, group_idx)?;
        let value_offset = resident_device_int4_column_offset(&snapshot, &table, value_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        // §9.5/S5a.3: ORDER BY + HAVING + LIMIT run on the GPU over the resident output (no
        // re-upload). The projection's aggregate as an i64 sort column (None for AVG -> Numeric,
        // stays host). COUNT packs (count, group) so counts must fit u32; SUM is direct i64; MIN/MAX
        // pack the i32 value + group; ORDER BY the group column packs group-by-group.
        let aggregate_col = match &select.projection {
            SelectProjection::GroupedCount { .. } => Some(GroupedI64SortColumn::Count),
            SelectProjection::GroupedSum { .. } => Some(GroupedI64SortColumn::Sum),
            SelectProjection::GroupedMin { .. } => Some(GroupedI64SortColumn::Min),
            SelectProjection::GroupedMax { .. } => Some(GroupedI64SortColumn::Max),
            _ => None,
        };
        let gpu_order: Option<GroupedI64Order> = {
            let limit = select.limit.map(|limit| limit as u64);
            let order_col = match &select.order_by {
                Some(order) => {
                    let by_aggregate = select_is_aggregate_result_column(select, &order.column);
                    match (aggregate_col, by_aggregate) {
                        // ORDER BY count needs counts to fit u32 (the composite pack).
                        (Some(GroupedI64SortColumn::Count), true) => (row_count
                            <= u64::from(u32::MAX))
                        .then_some(GroupedI64SortColumn::Count),
                        (Some(col), true) => Some(col), // Sum / Min / Max
                        (Some(_), false) => Some(GroupedI64SortColumn::Group), // ORDER BY group col
                        (None, _) => None,              // AVG
                    }
                }
                // HAVING with no ORDER BY: the host group-sorts first, so mirror with ORDER BY group.
                None if !select.having_groups.is_empty() && aggregate_col.is_some() => {
                    Some(GroupedI64SortColumn::Group)
                }
                None => None,
            };
            order_col.map(|column| GroupedI64Order {
                column,
                descending: select.order_by.as_ref().is_some_and(|o| o.descending),
                offset: 0,
                limit,
            })
        };
        // Translate HAVING (DNF) to GPU clauses (col 0=group / 1=aggregate; op 0..4; i64 value).
        // None when HAVING is present but not GPU-able (non-group/agg column, LikePrefix, non-i64
        // value, or AVG) -> the whole query falls back to the host.
        let gpu_having: Option<Vec<Vec<(u32, u32, i64)>>> = if select.having_groups.is_empty() {
            None
        } else {
            let aggregate_name = select_aggregate_result_column_name(select);
            (|| -> Option<Vec<Vec<(u32, u32, i64)>>> {
                aggregate_col?; // AVG has no i64 aggregate column
                let agg_name = aggregate_name?;
                select
                    .having_groups
                    .iter()
                    .map(|clause| {
                        clause
                            .iter()
                            .map(|filter| {
                                let col = if &filter.column == group_column {
                                    0_u32
                                } else if filter.column.eq_ignore_ascii_case(agg_name) {
                                    1
                                } else {
                                    return None;
                                };
                                let op = match filter.op {
                                    SelectFilterOp::Eq => 0_u32,
                                    SelectFilterOp::Lt => 1,
                                    SelectFilterOp::Lte => 2,
                                    SelectFilterOp::Gt => 3,
                                    SelectFilterOp::Gte => 4,
                                    SelectFilterOp::LikePrefix => return None,
                                };
                                let val = match &filter.value {
                                    SqlValue::Int4(v) => i64::from(*v),
                                    SqlValue::Int8(v) => *v,
                                    _ => return None,
                                };
                                Some((col, op, val))
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .collect::<Option<Vec<_>>>()
            })()
        };
        let having_translatable = select.having_groups.is_empty() || gpu_having.is_some();
        let use_gpu = gpu_order.is_some() && having_translatable;

        let started = Instant::now();
        let (mut grouped_stats, copied_group_count) = if use_gpu {
            let spec = gpu_order.expect("use_gpu implies gpu_order");
            let having = gpu_having.as_ref().map(|clauses| {
                (
                    aggregate_col.expect("HAVING translated => agg col"),
                    clauses.as_slice(),
                )
            });
            device_memory
                .grouped_stats_i32_ordered_from_payload(
                    group_offset,
                    value_offset,
                    row_count,
                    spec,
                    having,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?
        } else {
            let stats = device_memory
                .grouped_stats_i32_from_payload(group_offset, value_offset, row_count)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            let group_count = stats.len();
            (stats, group_count)
        };
        let elapsed = started.elapsed();
        // When the order ran on the GPU, the rows are already ordered + windowed; skip the host
        // sort/HAVING/LIMIT below (kept for the not-yet-GPU cases + as the parity reference).
        let gpu_ordered = use_gpu;
        let mut rows = grouped_stats
            .drain(..)
            .map(|group| {
                let aggregate: SqlValue = match &select.projection {
                    SelectProjection::GroupedCount { .. } => {
                        let count = i64::try_from(group.count).map_err(|_| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "resident device-memory grouped count {} exceeds supported COUNT(*) result range",
                                group.count
                            )))
                        })?;
                        Ok::<SqlValue, ExecuteError>(SqlValue::Int8(count))
                    }
                    SelectProjection::GroupedSum { .. } => {
                        Ok::<SqlValue, ExecuteError>(SqlValue::Int8(group.sum))
                    }
                    SelectProjection::GroupedAvg { .. } => {
                        Ok::<SqlValue, ExecuteError>(average_sql_value(
                            i128::from(group.sum),
                            group.count as usize,
                        ))
                    }
                    SelectProjection::GroupedMin { .. } => {
                        Ok::<SqlValue, ExecuteError>(SqlValue::Int4(group.min))
                    }
                    SelectProjection::GroupedMax { .. } => {
                        Ok::<SqlValue, ExecuteError>(SqlValue::Int4(group.max))
                    }
                    _ => unreachable!(),
                }?;
                Ok(vec![SqlValue::Int4(group.group), aggregate])
            })
            .collect::<Result<Vec<_>, ExecuteError>>()?;
        if !gpu_ordered {
            rows.sort_by(|left, right| compare_sql_values(&left[0], &right[0]));
            if !select.having_groups.is_empty() {
                let aggregate_name = select_aggregate_result_column_name(select)
                    .expect("grouped aggregate projection");
                rows = rows
                    .into_iter()
                    .filter_map(|row| {
                        let matches = grouped_row_matches_having(
                            select,
                            group_column,
                            &row[0],
                            aggregate_name,
                            &row[1],
                        );
                        match matches {
                            Ok(true) => Some(Ok(row)),
                            Ok(false) => None,
                            Err(err) => Some(Err(err)),
                        }
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()?;
            }
            if let Some(order) = &select.order_by {
                let order_by_sum = select_is_aggregate_result_column(select, &order.column);
                rows.sort_by(|left, right| {
                    let ordering = if order_by_sum {
                        compare_sql_values(&left[1], &right[1])
                    } else {
                        compare_sql_values(&left[0], &right[0])
                    };
                    ordering.then_with(|| compare_sql_values(&left[0], &right[0]))
                });
                if order.descending {
                    rows.reverse();
                }
            }
            if let Some(limit) = select.limit {
                rows.truncate(limit);
            }
        }
        let result_d2h_bytes = copied_group_count
            .checked_mul(
                std::mem::size_of::<i32>()
                    + std::mem::size_of::<u64>()
                    + std::mem::size_of::<i64>()
                    + (2 * std::mem::size_of::<i32>()),
            )
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let (group_column, value_column) = match &select.projection {
            SelectProjection::GroupedCount { column } => (column, column),
            SelectProjection::GroupedSum {
                group_column,
                sum_column,
            } => (group_column, sum_column),
            SelectProjection::GroupedAvg {
                group_column,
                avg_column,
            } => (group_column, avg_column),
            SelectProjection::GroupedMin {
                group_column,
                min_column,
            } => (group_column, min_column),
            SelectProjection::GroupedMax {
                group_column,
                max_column,
            } => (group_column, max_column),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory filtered grouped aggregate proof currently supports only grouped COUNT/SUM/AVG/MIN/MAX"
                        .to_string(),
                )));
            }
        };
        if select.distinct
            || select.offset.is_some()
            || bound.filter_groups.len() != 1
            || bound.filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered grouped aggregate proof currently supports only one int4 comparison predicate with optional HAVING, ORDER BY, and LIMIT"
                    .to_string(),
            )));
        }
        let Some(group_by) = &select.group_by else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered grouped aggregate proof requires GROUP BY"
                    .to_string(),
            )));
        };
        if !group_by.eq_ignore_ascii_case(group_column) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered grouped aggregate proof requires GROUP BY to match the projected group column"
                    .to_string(),
            )));
        }
        let group_idx = relational_column_index(&table, group_column)?;
        let value_idx = relational_column_index(&table, value_column)?;
        let (filter_idx, op, value) = bound.filter_groups[0][0].clone();
        let Some(comparison) = resident_device_i32_comparison(op) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered grouped aggregate proof currently supports only non-equality int4 comparisons"
                    .to_string(),
            )));
        };
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered grouped aggregate proof currently supports only int4 comparison literals"
                    .to_string(),
            )));
        };
        if table.columns[group_idx].ty != SqlType::Int4
            || table.columns[value_idx].ty != SqlType::Int4
            || table.columns[filter_idx].ty != SqlType::Int4
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered grouped aggregate proof currently supports only int4 group, aggregate, and filter columns"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
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
        let snapshot_gpu_id = snapshot.gpu_id;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let group_offset = resident_device_int4_column_offset(&snapshot, &table, group_idx)?;
        let value_offset = resident_device_int4_column_offset(&snapshot, &table, value_idx)?;
        let filter_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let mut grouped_stats = device_memory
            .filtered_grouped_stats_i32_compare_from_payload(
                group_offset,
                value_offset,
                filter_offset,
                row_count,
                needle,
                comparison,
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        let copied_group_count = grouped_stats.len();
        // The filtered grouped probe does not yet run ORDER BY on the GPU (slice 1 wired the
        // unfiltered probe); HAVING/ORDER BY/LIMIT stay on the host here.
        let gpu_ordered = false;
        let mut rows = grouped_stats
            .drain(..)
            .map(|group| {
                let aggregate: SqlValue = match &select.projection {
                    SelectProjection::GroupedCount { .. } => {
                        let count = i64::try_from(group.count).map_err(|_| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "resident device-memory filtered grouped count {} exceeds supported COUNT(*) result range",
                                group.count
                            )))
                        })?;
                        Ok::<SqlValue, ExecuteError>(SqlValue::Int8(count))
                    }
                    SelectProjection::GroupedSum { .. } => {
                        Ok::<SqlValue, ExecuteError>(SqlValue::Int8(group.sum))
                    }
                    SelectProjection::GroupedAvg { .. } => {
                        Ok::<SqlValue, ExecuteError>(average_sql_value(
                            i128::from(group.sum),
                            group.count as usize,
                        ))
                    }
                    SelectProjection::GroupedMin { .. } => {
                        Ok::<SqlValue, ExecuteError>(SqlValue::Int4(group.min))
                    }
                    SelectProjection::GroupedMax { .. } => {
                        Ok::<SqlValue, ExecuteError>(SqlValue::Int4(group.max))
                    }
                    _ => unreachable!(),
                }?;
                Ok(vec![SqlValue::Int4(group.group), aggregate])
            })
            .collect::<Result<Vec<_>, ExecuteError>>()?;
        if !gpu_ordered {
            rows.sort_by(|left, right| compare_sql_values(&left[0], &right[0]));
            if !select.having_groups.is_empty() {
                let aggregate_name = select_aggregate_result_column_name(select)
                    .expect("grouped aggregate projection");
                rows = rows
                    .into_iter()
                    .filter_map(|row| {
                        let matches = grouped_row_matches_having(
                            select,
                            group_column,
                            &row[0],
                            aggregate_name,
                            &row[1],
                        );
                        match matches {
                            Ok(true) => Some(Ok(row)),
                            Ok(false) => None,
                            Err(err) => Some(Err(err)),
                        }
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()?;
            }
            if let Some(order) = &select.order_by {
                let order_by_sum = select_is_aggregate_result_column(select, &order.column);
                rows.sort_by(|left, right| {
                    let ordering = if order_by_sum {
                        compare_sql_values(&left[1], &right[1])
                    } else {
                        compare_sql_values(&left[0], &right[0])
                    };
                    ordering.then_with(|| compare_sql_values(&left[0], &right[0]))
                });
                if order.descending {
                    rows.reverse();
                }
            }
            if let Some(limit) = select.limit {
                rows.truncate(limit);
            }
        }
        let result_d2h_bytes = copied_group_count
            .checked_mul(
                std::mem::size_of::<i32>()
                    + std::mem::size_of::<u64>()
                    + std::mem::size_of::<i64>()
                    + (2 * std::mem::size_of::<i32>()),
            )
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: DeviceTarget::Gpu(snapshot_gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot_gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_projection_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.selected_indexes.len() != 1
            || filter_groups.len() != 1
            || filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory projection proof currently supports only SELECT one_int4_column with one int4 range predicate"
                    .to_string(),
            )));
        }
        let projection_idx = bound.selected_indexes[0];
        if table.columns[projection_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory projection proof currently supports only int4 projection columns"
                    .to_string(),
            )));
        }
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        let Some(comparison) = resident_device_i32_comparison(op) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory projection proof currently supports only int4 non-equality predicates"
                    .to_string(),
            )));
        };
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory projection proof currently supports only int4 non-equality predicates"
                    .to_string(),
            )));
        };
        if table.columns[filter_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory projection proof currently supports only int4 non-equality predicates"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let filter_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let projection_offset =
            resident_device_int4_column_offset(&snapshot, &table, projection_idx)?;
        if filter_offset != projection_offset {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory projection proof currently requires predicate and projection to use the same int4 column"
                    .to_string(),
            )));
        }
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let values = device_memory
            .project_i32_compare_from_payload(projection_offset, row_count, needle, comparison)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        let result_d2h_bytes = values
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: values
                .into_iter()
                .map(|value| vec![SqlValue::Int4(value)])
                .collect(),
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_equality_projection_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.selected_indexes.len() != 1
            || filter_groups.len() != 1
            || filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory equality projection proof currently supports only SELECT one_int4_column with one same-column int4 equality predicate"
                    .to_string(),
            )));
        }
        let projection_idx = bound.selected_indexes[0];
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory equality projection proof currently supports only int4 equality predicates"
                    .to_string(),
            )));
        };
        if op != SelectFilterOp::Eq
            || projection_idx != filter_idx
            || table.columns[projection_idx].ty != SqlType::Int4
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory equality projection proof currently requires the projected int4 column to be the equality predicate column"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        // Borrow the stored snapshot (`_ref`) instead of `relational_residency_snapshot`, whose
        // `.clone()` deep-copies `resident_rows: Vec<Vec<SqlValue>>` — an O(rows) host allocation on
        // EVERY per-call lookup. That clone (not the kernel) was this route's per-call wall: ~99% of
        // the time on a 50k-row table, dwarfing the ~40µs migrated parallel count kernel. The proven
        // sibling routes (multi-column / projection / count) already borrow via `_ref`, and the live
        // memory-pressure gate runs in the planner before this route is entered, so reading the
        // stored snapshot here is behaviorally identical to them (and to the prior clone).
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let lookup_started = Instant::now();
        let matched_count = device_memory
            .count_i32_equal_from_payload(byte_offset, row_count, needle)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let lookup_micros = lookup_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        let matched_len = usize::try_from(matched_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory equality projection count {matched_count} exceeds host result range"
            )))
        })?;
        self.metrics
            .observe_d2h_bytes(std::mem::size_of::<u64>() as u64);
        self.read_state
            .route_telemetry
            .record_route_device_lookup_micros(&table.name, lookup_micros, matched_len);

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: std::iter::repeat_with(|| vec![SqlValue::Int4(needle)])
                .take(matched_len)
                .collect(),
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_equality_multi_column_projection_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let filter_groups = if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            Vec::new()
        };
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.order_by.is_some()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.selected_indexes.len() < 2
            || filter_groups.len() != 1
            || filter_groups[0].is_empty()
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory equality multi-column projection proof currently supports SELECT int4_columns with int4 equality predicates"
                    .to_string(),
            )));
        }
        let filters = filter_groups[0]
            .iter()
            .map(|(filter_idx, op, value)| {
                let SqlValue::Int4(needle) = value else {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident device-memory equality multi-column projection proof currently supports only int4 equality predicates"
                            .to_string(),
                    )));
                };
                if *op != SelectFilterOp::Eq || table.columns[*filter_idx].ty != SqlType::Int4 {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident device-memory equality multi-column projection proof currently supports only int4 equality predicates"
                            .to_string(),
                    )));
                }
                Ok((*filter_idx, *needle))
            })
            .collect::<Result<Vec<_>, ExecuteError>>()?;
        if bound
            .selected_indexes
            .iter()
            .any(|idx| !matches!(table.columns[*idx].ty, SqlType::Int4 | SqlType::Text))
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory equality multi-column projection proof currently supports only int4 or text projection columns"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
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
        let snapshot_gpu_id = snapshot.gpu_id;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let filter_offsets = filters
            .iter()
            .map(|(filter_idx, needle)| {
                resident_device_int4_column_offset(&snapshot, &table, *filter_idx)
                    .map(|offset| (offset, *needle))
            })
            .collect::<Result<Vec<_>, ExecuteError>>()?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        if bound
            .selected_indexes
            .iter()
            .all(|idx| table.columns[*idx].ty == SqlType::Int4)
        {
            let projection_offsets = bound
                .selected_indexes
                .iter()
                .map(|idx| resident_device_int4_column_offset(&snapshot, &table, *idx))
                .collect::<Result<Vec<_>, ExecuteError>>()?;
            let fused_started = Instant::now();
            let projected_rows = device_memory
                .match_project_i32_equal_from_payload(
                    &filter_offsets,
                    &projection_offsets,
                    row_count,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            let fused_micros = fused_started
                .elapsed()
                .as_micros()
                .try_into()
                .unwrap_or(u64::MAX);
            let materialize_started = Instant::now();
            let rows = projected_rows
                .into_iter()
                .map(|row| row.into_iter().map(SqlValue::Int4).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let materialization_micros = materialize_started
                .elapsed()
                .as_micros()
                .try_into()
                .unwrap_or(u64::MAX);
            let result_d2h_bytes = rows
                .len()
                .checked_mul(bound.selected_indexes.len())
                .and_then(|cells| cells.checked_mul(std::mem::size_of::<i32>()))
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u32>()))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(u64::MAX);
            self.metrics.observe_d2h_bytes(result_d2h_bytes);
            self.metrics
                .observe_kernel_exec_ms(fused_micros.div_ceil(1000).max(1));
            self.read_state
                .route_telemetry
                .record_route_selected_projection_micros(
                    &table.name,
                    fused_micros,
                    fused_micros,
                    materialization_micros,
                    rows.len(),
                );

            return Ok(RelationalSelectResult {
                columns: bound.selected_columns,
                rows,
                planned_target: DeviceTarget::Gpu(snapshot_gpu_id),
                executed_target: DeviceTarget::Gpu(snapshot_gpu_id),
                fallback_reason: None,
                access_path,
            });
        }
        // Mixed int4+text projection. The legacy path below issues a CASCADE of separate
        // synchronous GPU launches — `match_i32_equal_row_indices` (one launch) followed by a
        // per-projected-column `project_i32_rows` / `project_text_rows` launch — each on the
        // default/NULL stream with a whole-context `cuCtxSynchronize`, a per-call
        // `cuModuleLoadData` re-JIT, and a per-call `cuMemAlloc`/`cuMemFree`. A per-section
        // wall-clock breakdown localized ~99.8% of this route's c64 wall to that cascade (the
        // row-indices launch alone was ~20.7 ms/call @c64; ~73% of the wall), all on the
        // un-migrated synchronous substrate.
        //
        // For the common SINGLE-predicate int4+text shape (the `mixed_int_text` benchmark route),
        // delegate to the single-statement batch path, which fuses int4 + a single text column into
        // ONE pooled-stream launch (`match_project_i32_equal_any_text_from_payload`, the
        // P2-M1/P2-M2 substrate: cached module, private pooled stream, pooled output buffers) and
        // falls back internally to a cascade only for the rare multi-text shape. The all-int4
        // branch above is already a single fused launch and is left untouched. The MULTI-predicate
        // mixed shape (e.g. `WHERE a = 1 AND b = 2` with a text projection) is not yet accepted by
        // the batch path, so it retains the legacy cascade below — correct, just unmigrated; it is
        // not on any benchmarked hot path.
        if filter_offsets.len() == 1 {
            // The dispatcher (`execute_relational_select_with_resident_route`) that called this
            // method records the route-execution observation for the whole route, so the
            // delegated batch path must NOT record its own — otherwise the route telemetry is
            // double-counted. Pass `record_route_observation: false`.
            let mut results = self
                .execute_relational_equality_multi_column_projection_batch_inner(
                    std::slice::from_ref(select),
                    None,
                    false,
                )?;
            return results.pop().ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality mixed-column projection returned no result"
                        .to_string(),
                ))
            });
        }
        let match_started = Instant::now();
        let matching_row_indices = device_memory
            .match_i32_equal_row_indices_from_payload(&filter_offsets, row_count)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let match_index_micros = match_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        let mut projected_columns = bound
            .selected_indexes
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let started = Instant::now();
        let mut column_values = BTreeMap::new();
        let mut text_values = BTreeMap::new();
        for idx in std::mem::take(&mut projected_columns) {
            match table.columns[idx].ty {
                SqlType::Int4 => {
                    let byte_offset = resident_device_int4_column_offset(&snapshot, &table, idx)?;
                    let values = device_memory
                        .project_i32_rows_from_payload(byte_offset, &matching_row_indices)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    if values.len() != matching_row_indices.len() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "resident device-memory equality multi-column selected-row projection column returned {} rows, expected {}",
                            values.len(),
                            matching_row_indices.len()
                        ))));
                    }
                    column_values.insert(idx, values);
                }
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(&snapshot, &table, idx)?;
                    let values = device_memory
                        .project_text_rows_from_payload(
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            layout.bytes_len,
                            &matching_row_indices,
                        )
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    if values.len() != matching_row_indices.len() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "resident device-memory equality multi-column selected-row text projection column returned {} rows, expected {}",
                            values.len(),
                            matching_row_indices.len()
                        ))));
                    }
                    text_values.insert(idx, values);
                }
                // Typed (int8/numeric/bool) columns never reach a GPU-resident projection route
                // — `resident_route_shape` rejects them upstream so they take the CPU path. Guard
                // defensively in case a future route admits them before the kernels support them.
                SqlType::Int8 | SqlType::Numeric { .. } | SqlType::Bool => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident device-memory projection supports only int4/text columns"
                            .to_string(),
                    )));
                }
            }
        }
        let elapsed = started.elapsed();
        let materialize_started = Instant::now();
        let rows = (0..matching_row_indices.len())
            .map(|selected_idx| {
                bound
                    .selected_indexes
                    .iter()
                    .map(|idx| {
                        if let Some(values) = column_values.get(idx) {
                            return Ok(SqlValue::Int4(values[selected_idx]));
                        }
                        if let Some(values) = text_values.get(idx) {
                            return Ok(SqlValue::Text(values[selected_idx].clone()));
                        }
                        Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "resident device-memory equality multi-column projection missing projected column"
                                .to_string(),
                        )))
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()
            })
            .collect::<Result<Vec<_>, ExecuteError>>()?;
        let materialization_micros = materialize_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        let int4_d2h_bytes = column_values
            .len()
            .checked_mul(matching_row_indices.len())
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<i32>()))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        let text_d2h_bytes = text_values
            .values()
            .map(|values| {
                u64::try_from(values.len())
                    .unwrap_or(u64::MAX)
                    .saturating_mul(2 * std::mem::size_of::<u64>() as u64)
                    .saturating_add(
                        values
                            .iter()
                            .map(|value| u64::try_from(value.len()).unwrap_or(u64::MAX))
                            .fold(0_u64, u64::saturating_add),
                    )
            })
            .fold(0_u64, u64::saturating_add);
        let match_index_d2h_bytes = u64::try_from(matching_row_indices.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(std::mem::size_of::<u64>() as u64)
            .saturating_add(std::mem::size_of::<u64>() as u64);
        let result_d2h_bytes = int4_d2h_bytes
            .saturating_add(text_d2h_bytes)
            .saturating_add(match_index_d2h_bytes);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        for _ in 0..column_values
            .len()
            .saturating_add(text_values.len())
            .saturating_add(1)
        {
            self.metrics
                .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));
        }
        self.read_state
            .route_telemetry
            .record_route_selected_projection_micros(
                &table.name,
                match_index_micros,
                elapsed.as_micros().try_into().unwrap_or(u64::MAX),
                materialization_micros,
                matching_row_indices.len(),
            );

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: DeviceTarget::Gpu(snapshot_gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot_gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
        &self,
        selects: &[Select],
    ) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
        self.execute_relational_equality_multi_column_projection_batch_inner(selects, None, true)
    }

    pub(crate) fn execute_relational_equality_multi_column_projection_batch_inner(
        &self,
        selects: &[Select],
        planned_jobs: Option<&[RelationalRetainedReadJob]>,
        record_route_observation: bool,
    ) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
        if selects.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(jobs) = planned_jobs {
            if jobs.len() != selects.len() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "retained read job count {} does not match SELECT count {}",
                    jobs.len(),
                    selects.len()
                ))));
            }
        }
        let mut members = Vec::with_capacity(selects.len());
        let mut batch_table: Option<RelationalTable> = None;
        let mut batch_filter_idx: Option<usize> = None;
        let mut batch_selected_indexes: Option<Vec<usize>> = None;
        for (select_idx, select) in selects.iter().enumerate() {
            let query_shape = if let Some(jobs) = planned_jobs {
                jobs[select_idx]
                    .route_id
                    .split(':')
                    .next()
                    .unwrap_or("unknown")
                    .to_string()
            } else {
                let decision = self.plan_relational_resident_route(select);
                if !decision.accepted {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident device-memory equality batch currently supports only accepted int4 equality projection, got {}: {}",
                        decision.query_shape, decision.reason
                    ))));
                }
                decision.query_shape
            };
            if !matches!(
                query_shape.as_str(),
                "int4_equality_projection"
                    | "int4_equality_multi_column_projection"
                    | "int4_equality_mixed_column_projection"
            ) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident device-memory equality batch currently supports only accepted int4 equality projection, got {}: {}",
                    query_shape, "preplanned retained read job"
                ))));
            }
            let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
            let filter_groups = if !bound.filter_groups.is_empty() {
                bound.filter_groups.clone()
            } else if !bound.filters.is_empty() {
                vec![bound.filters.clone()]
            } else if let Some(filter) = bound.filter.clone() {
                vec![vec![filter]]
            } else {
                Vec::new()
            };
            if select.distinct
                || select.group_by.is_some()
                || !select.having_groups.is_empty()
                || select.order_by.is_some()
                || select.limit.is_some()
                || select.offset.is_some()
                || bound.selected_indexes.is_empty()
                || !bound
                    .selected_indexes
                    .iter()
                    .all(|idx| matches!(table.columns[*idx].ty, SqlType::Int4 | SqlType::Text))
                || filter_groups.len() != 1
                || filter_groups[0].len() != 1
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch currently supports SELECT one_or_more_int4_or_text_columns with one int4 equality predicate"
                        .to_string(),
                )));
            }
            let (filter_idx, op, value) = filter_groups[0][0].clone();
            let SqlValue::Int4(needle) = value else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch currently supports only int4 equality predicates"
                        .to_string(),
                )));
            };
            if op != SelectFilterOp::Eq || table.columns[filter_idx].ty != SqlType::Int4 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch currently supports only int4 equality predicates"
                        .to_string(),
                )));
            }
            if let Some(existing) = &batch_table {
                if existing.name != table.name
                    || existing.schema != table.schema
                    || existing.columns != table.columns
                {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident device-memory equality batch cannot mix tables".to_string(),
                    )));
                }
            } else {
                batch_table = Some(table.clone());
            }
            if batch_filter_idx.is_some_and(|existing| existing != filter_idx) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch cannot mix predicate columns"
                        .to_string(),
                )));
            }
            batch_filter_idx = Some(filter_idx);
            if batch_selected_indexes
                .as_ref()
                .is_some_and(|existing| existing != &bound.selected_indexes)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch cannot mix projection columns"
                        .to_string(),
                )));
            }
            batch_selected_indexes = Some(bound.selected_indexes.clone());
            let (_query, access_path) =
                self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
            members.push((bound, access_path, needle));
        }

        let table = batch_table.expect("non-empty batch has table");
        let filter_idx = batch_filter_idx.expect("non-empty batch has filter");
        let selected_indexes = batch_selected_indexes.expect("non-empty batch has projections");
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
        let snapshot_gpu_id = snapshot.gpu_id;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let filter_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let all_int4_projection = selected_indexes
            .iter()
            .all(|idx| table.columns[*idx].ty == SqlType::Int4);
        let projection_offsets = if all_int4_projection {
            selected_indexes
                .iter()
                .map(|idx| resident_device_int4_column_offset(&snapshot, &table, *idx))
                .collect::<Result<Vec<_>, ExecuteError>>()?
        } else {
            vec![filter_offset]
        };
        let needles = members
            .iter()
            .map(|(_bound, _access_path, needle)| *needle)
            .collect::<Vec<_>>();

        if let Some(device_memory) = self.read_state.residency.device_memory.get(&table.name) {
            device_memory.clear_last_kernel_event_elapsed_us();
        }
        let text_projection_indexes = selected_indexes
            .iter()
            .copied()
            .filter(|idx| table.columns[*idx].ty == SqlType::Text)
            .collect::<Vec<_>>();
        let compact_text_projection_idx = (!all_int4_projection
            && text_projection_indexes.len() == 1)
            .then(|| text_projection_indexes[0]);
        let int4_projection_indexes = selected_indexes
            .iter()
            .copied()
            .filter(|idx| table.columns[*idx].ty == SqlType::Int4)
            .collect::<Vec<_>>();
        let int4_projection_offsets = int4_projection_indexes
            .iter()
            .map(|idx| resident_device_int4_column_offset(&snapshot, &table, *idx))
            .collect::<Result<Vec<_>, ExecuteError>>()?;
        let before_metrics = self.metrics.snapshot();
        let batch_started = Instant::now();
        let compact_text_rows = if let Some(text_idx) = compact_text_projection_idx {
            let device_memory = self
                .read_state
                .residency
                .device_memory
                .get(&table.name)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" has no retained resident device memory",
                        table.name
                    )))
                })?;
            let layout = resident_device_text_column_layout(&snapshot, &table, text_idx)?;
            Some(
                device_memory
                    .match_project_i32_equal_any_text_from_payload(
                        filter_offset,
                        &needles,
                        &int4_projection_offsets,
                        layout.offsets_byte_offset,
                        layout.bytes_byte_offset,
                        layout.bytes_len,
                        row_count,
                    )
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?,
            )
        } else {
            None
        };
        let projected_rows = if compact_text_rows.is_none() {
            let device_memory = self
                .read_state
                .residency
                .device_memory
                .get(&table.name)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" has no retained resident device memory",
                        table.name
                    )))
                })?;
            Some(
                device_memory
                    .match_project_i32_equal_any_from_payload(
                        filter_offset,
                        &needles,
                        &projection_offsets,
                        row_count,
                    )
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?,
            )
        } else {
            None
        };
        let batch_micros = batch_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        let materialize_started = Instant::now();
        // Stable-order fix (Thread-3 Stage 4): every branch below scatters matched rows into
        // per-needle slices in the kernel's `atom.global.add` SCHEDULE order, which is
        // non-deterministic for >32 matches (multi-warp). Tag each row with its kernel `row_index`
        // and sort each needle's slice ASCENDING by it (after the branch), so the output is
        // deterministic and byte-identical to the per-query ascending order (the `row_indices`
        // order class established by `4b750a94`). The single-element delegation from the per-query
        // mixed/multi-column path flows through here too, so the per-query and batched paths share
        // this one sorted assembly and stay byte-identical by construction.
        let rows_by_select: Vec<Vec<(u64, Vec<SqlValue>)>> = if all_int4_projection {
            let mut rows_by_select = vec![Vec::new(); members.len()];
            let projected_rows = projected_rows.as_ref().ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch missing int4 projection rows"
                        .to_string(),
                ))
            })?;
            for projected in projected_rows {
                rows_by_select[projected.needle_index].push((
                    projected.row_index,
                    projected
                        .values
                        .iter()
                        .copied()
                        .map(SqlValue::Int4)
                        .collect::<Vec<_>>(),
                ));
            }
            rows_by_select
        } else if let (Some(text_idx), Some(compact_rows)) =
            (compact_text_projection_idx, compact_text_rows.as_ref())
        {
            let int4_positions = int4_projection_indexes
                .iter()
                .copied()
                .enumerate()
                .map(|(position, idx)| (idx, position))
                .collect::<BTreeMap<_, _>>();
            let mut rows_by_select = vec![Vec::new(); members.len()];
            for projected in compact_rows {
                let row = selected_indexes
                    .iter()
                    .map(|idx| {
                        if *idx == text_idx {
                            return Ok(SqlValue::Text(projected.text.clone()));
                        }
                        if let Some(position) = int4_positions.get(idx) {
                            return Ok(SqlValue::Int4(projected.values[*position]));
                        }
                        Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "resident device-memory equality batch compact text projection missing projected column"
                                .to_string(),
                        )))
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()?;
                rows_by_select[projected.needle_index].push((projected.row_index, row));
            }
            rows_by_select
        } else {
            let projected_rows = projected_rows.as_ref().ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch missing mixed projection rows"
                        .to_string(),
                ))
            })?;
            let matched_row_indices = projected_rows
                .iter()
                .map(|projected| projected.row_index)
                .collect::<Vec<_>>();
            let device_memory = self
                .read_state
                .residency
                .device_memory
                .get(&table.name)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" has no retained resident device memory",
                        table.name
                    )))
                })?;
            let mut int4_values = BTreeMap::new();
            let mut text_values = BTreeMap::new();
            for idx in &selected_indexes {
                match table.columns[*idx].ty {
                    SqlType::Int4 => {
                        let byte_offset =
                            resident_device_int4_column_offset(&snapshot, &table, *idx)?;
                        let values = device_memory
                            .project_i32_rows_from_payload(byte_offset, &matched_row_indices)
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?;
                        if values.len() != matched_row_indices.len() {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "resident device-memory equality batch mixed projection int4 column returned {} rows, expected {}",
                                values.len(),
                                matched_row_indices.len()
                            ))));
                        }
                        int4_values.insert(*idx, values);
                    }
                    SqlType::Text => {
                        let layout = resident_device_text_column_layout(&snapshot, &table, *idx)?;
                        let values = device_memory
                            .project_text_rows_from_payload(
                                layout.offsets_byte_offset,
                                layout.bytes_byte_offset,
                                layout.bytes_len,
                                &matched_row_indices,
                            )
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?;
                        if values.len() != matched_row_indices.len() {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "resident device-memory equality batch mixed projection text column returned {} rows, expected {}",
                                values.len(),
                                matched_row_indices.len()
                            ))));
                        }
                        text_values.insert(*idx, values);
                    }
                    // See the multi-column route above: typed columns take the CPU path; this
                    // GPU projection only handles int4/text.
                    SqlType::Int8 | SqlType::Numeric { .. } | SqlType::Bool => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "resident device-memory projection supports only int4/text columns"
                                .to_string(),
                        )));
                    }
                }
            }
            let mut rows_by_select = vec![Vec::new(); members.len()];
            for (projected_idx, projected) in projected_rows.iter().enumerate() {
                let row = selected_indexes
                    .iter()
                    .map(|idx| {
                        if let Some(values) = int4_values.get(idx) {
                            return Ok(SqlValue::Int4(values[projected_idx]));
                        }
                        if let Some(values) = text_values.get(idx) {
                            return Ok(SqlValue::Text(values[projected_idx].clone()));
                        }
                        Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "resident device-memory equality batch mixed projection missing projected column"
                                .to_string(),
                        )))
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()?;
                rows_by_select[projected.needle_index].push((projected.row_index, row));
            }
            rows_by_select
        };
        // Apply the ascending-by-`row_index` order to every needle's slice (see the stable-order
        // note above), then strip the index tag back to the materialized rows.
        let rows_by_select: Vec<Vec<Vec<SqlValue>>> = rows_by_select
            .into_iter()
            .map(|mut slice| {
                slice.sort_by_key(|(row_index, _)| *row_index);
                slice.into_iter().map(|(_, row)| row).collect()
            })
            .collect();
        let materialization_micros = materialize_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        let total_rows = rows_by_select.iter().map(Vec::len).sum::<usize>();
        let int4_result_columns = selected_indexes
            .iter()
            .filter(|idx| table.columns[**idx].ty == SqlType::Int4)
            .count();
        let text_result_bytes = if all_int4_projection {
            0
        } else {
            rows_by_select
                .iter()
                .flatten()
                .flat_map(|row| row.iter())
                .filter_map(|value| match value {
                    SqlValue::Text(value) => Some(u64::try_from(value.len()).unwrap_or(u64::MAX)),
                    _ => None,
                })
                .fold(0_u64, u64::saturating_add)
                .saturating_add(
                    u64::try_from(total_rows)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(2 * std::mem::size_of::<u64>() as u64),
                )
        };
        let row_metadata_d2h_bytes = u64::try_from(total_rows)
            .unwrap_or(u64::MAX)
            .saturating_mul((std::mem::size_of::<u32>() + std::mem::size_of::<u64>()) as u64)
            .saturating_add(std::mem::size_of::<u32>() as u64);
        let result_d2h_bytes = u64::try_from(total_rows)
            .unwrap_or(u64::MAX)
            .saturating_mul(
                u64::try_from(int4_result_columns)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(std::mem::size_of::<i32>() as u64),
            )
            .saturating_add(text_result_bytes)
            .saturating_add(row_metadata_d2h_bytes);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        let kernel_samples = if all_int4_projection {
            1
        } else {
            selected_indexes.len().saturating_add(1)
        };
        for _ in 0..kernel_samples {
            self.metrics
                .observe_kernel_exec_ms(batch_micros.div_ceil(1000).max(1));
        }
        let kernel_event_elapsed_us = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .and_then(|device_memory| device_memory.last_kernel_event_elapsed_us());
        if let Some(elapsed_us) = kernel_event_elapsed_us {
            self.metrics.observe_kernel_event_elapsed_us(elapsed_us);
        }
        self.read_state
            .route_telemetry
            .record_route_selected_projection_micros(
                &table.name,
                batch_micros,
                batch_micros,
                materialization_micros,
                total_rows,
            );
        // The route-execution observation is recorded once per route execution. When the
        // single-predicate mixed int4+text dispatcher path delegates here (a 1-element slice),
        // that caller's `execute_relational_select_with_resident_route` already records the
        // observation for the whole route, so the delegated call suppresses its own to avoid a
        // double-count (telemetry-only; results are unaffected). The standalone batch/submit
        // callers are not wrapped by the dispatcher and own the record themselves.
        if record_route_observation {
            let after_metrics = self.metrics.snapshot();
            self.read_state
                .route_telemetry
                .record_route_execution_observation(
                    &table.name,
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
                        rows: total_rows,
                        wall_micros: batch_started
                            .elapsed()
                            .as_micros()
                            .try_into()
                            .unwrap_or(u64::MAX),
                    },
                );
        }

        Ok(members
            .into_iter()
            .zip(rows_by_select)
            .map(
                |((bound, access_path, _needle), rows)| RelationalSelectResult {
                    columns: bound.selected_columns,
                    rows,
                    planned_target: DeviceTarget::Gpu(snapshot_gpu_id),
                    executed_target: DeviceTarget::Gpu(snapshot_gpu_id),
                    fallback_reason: None,
                    access_path,
                },
            )
            .collect())
    }

    pub fn execute_relational_distinct_projection_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if !select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || select.filter.is_some()
            || !select.filters.is_empty()
            || !select.filter_groups.is_empty()
            || bound.selected_indexes.len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory distinct projection proof currently supports only SELECT DISTINCT one_int4_column with optional same-column ORDER BY, LIMIT, and OFFSET"
                    .to_string(),
            )));
        }
        if select.offset.is_some() && (bound.order.is_none() || select.limit.is_none()) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory distinct projection OFFSET proof currently requires same-column ORDER BY and LIMIT"
                    .to_string(),
            )));
        }
        let projection_idx = bound.selected_indexes[0];
        if table.columns[projection_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory distinct projection proof currently supports only int4 projection columns"
                    .to_string(),
            )));
        }
        let descending = if let Some((order_idx, descending)) = bound.order {
            if order_idx != projection_idx {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory distinct projection proof currently requires ORDER BY to use the projected int4 column"
                        .to_string(),
                )));
            }
            Some(descending)
        } else {
            None
        };

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let projection_offset =
            resident_device_int4_column_offset(&snapshot, &table, projection_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let values = device_memory
            .project_i32_from_payload(projection_offset, row_count)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        // Kernel-less projection D2Hs only the i32 column (no device `out_count` readback), so the
        // d2h estimate is exactly the value bytes — drop the old `+ size_of::<u64>()` count term.
        let result_d2h_bytes = values
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));

        let mut rows = Vec::new();
        let mut seen = BTreeSet::new();
        for value in values {
            if seen.insert(value) {
                rows.push(vec![SqlValue::Int4(value)]);
            }
        }
        if let Some(descending) = descending {
            rows.sort_by(|left, right| compare_sql_values(&left[0], &right[0]));
            if descending {
                rows.reverse();
            }
        }
        if let Some(offset) = select.offset {
            rows = rows.into_iter().skip(offset).collect();
        }
        if let Some(limit) = select.limit {
            rows.truncate(limit);
        }

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if !select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || bound.selected_indexes.len() != 1
            || bound.filter_groups.len() != 1
            || bound.filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered distinct projection proof currently supports only SELECT DISTINCT one_int4_column with one same-column int4 comparison predicate, optional same-column ORDER BY, LIMIT, and OFFSET"
                    .to_string(),
            )));
        }
        if select.offset.is_some() && (bound.order.is_none() || select.limit.is_none()) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered distinct projection OFFSET proof currently requires same-column ORDER BY and LIMIT"
                    .to_string(),
            )));
        }
        let projection_idx = bound.selected_indexes[0];
        if table.columns[projection_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered distinct projection proof currently supports only int4 projection columns"
                    .to_string(),
            )));
        }
        let (filter_idx, op, value) = bound.filter_groups[0][0].clone();
        if filter_idx != projection_idx {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered distinct projection proof currently requires predicate and projection to use the same int4 column"
                    .to_string(),
            )));
        }
        let Some(comparison) = resident_device_i32_comparison(op) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered distinct projection proof currently supports only non-equality int4 comparisons"
                    .to_string(),
            )));
        };
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory filtered distinct projection proof currently supports only int4 comparison literals"
                    .to_string(),
            )));
        };
        let descending = if let Some((order_idx, descending)) = bound.order {
            if order_idx != projection_idx {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory filtered distinct projection proof currently requires ORDER BY to use the projected int4 column"
                        .to_string(),
                )));
            }
            Some(descending)
        } else {
            None
        };

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let projection_offset =
            resident_device_int4_column_offset(&snapshot, &table, projection_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let values = device_memory
            .project_i32_compare_from_payload(projection_offset, row_count, needle, comparison)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        let result_d2h_bytes = values
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));

        let mut rows = Vec::new();
        let mut seen = BTreeSet::new();
        for value in values {
            if seen.insert(value) {
                rows.push(vec![SqlValue::Int4(value)]);
            }
        }
        if let Some(descending) = descending {
            rows.sort_by(|left, right| compare_sql_values(&left[0], &right[0]));
            if descending {
                rows.reverse();
            }
        }
        if let Some(offset) = select.offset {
            rows = rows.into_iter().skip(offset).collect();
        }
        if let Some(limit) = select.limit {
            rows.truncate(limit);
        }

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    pub fn execute_relational_ordered_projection_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || bound.selected_indexes.len() != 1
            || bound.filter_groups.len() != 1
            || bound.filter_groups[0].len() != 1
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection proof currently supports only SELECT one_int4_column with one int4 range predicate, ORDER BY that column, and LIMIT"
                    .to_string(),
            )));
        }
        let projection_idx = bound.selected_indexes[0];
        if table.columns[projection_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection proof currently supports only int4 projection columns"
                    .to_string(),
            )));
        }
        let Some((order_idx, descending)) = bound.order else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection proof currently requires ORDER BY"
                    .to_string(),
            )));
        };
        if order_idx != projection_idx {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection proof currently requires ORDER BY to use the projected int4 column"
                    .to_string(),
            )));
        }
        let Some(limit) = select.limit else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection proof currently requires LIMIT"
                    .to_string(),
            )));
        };
        let limit = u64::try_from(limit).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection LIMIT exceeds proof range".to_string(),
            ))
        })?;
        let offset = u64::try_from(select.offset.unwrap_or(0)).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection OFFSET exceeds proof range".to_string(),
            ))
        })?;
        let (filter_idx, op, value) = bound.filter_groups[0][0].clone();
        let Some(comparison) = resident_device_i32_comparison(op) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection proof currently supports only int4 non-equality predicates"
                    .to_string(),
            )));
        };
        let SqlValue::Int4(needle) = value else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection proof currently supports only int4 non-equality predicates"
                    .to_string(),
            )));
        };
        if table.columns[filter_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection proof currently supports only int4 non-equality predicates"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
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
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let filter_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let projection_offset =
            resident_device_int4_column_offset(&snapshot, &table, projection_idx)?;
        if filter_offset != projection_offset {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory ordered projection proof currently requires predicate and projection to use the same int4 column"
                    .to_string(),
            )));
        }
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let values = device_memory
            .project_i32_compare_ordered_from_payload(
                projection_offset,
                row_count,
                needle,
                comparison,
                descending,
                (offset, limit),
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        let result_d2h_bytes = values
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows: values
                .into_iter()
                .map(|value| vec![SqlValue::Int4(value)])
                .collect(),
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    #[cfg(test)]
    pub(crate) fn execute_relational_select_with_backend<B: MvccExecutionBackend>(
        &mut self,
        select: &Select,
        backend: &B,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let pin = self.pin_relational_read_at(&select.table, copin_s);
        let (query, access_path) =
            self.relational_select_mvcc_query(select, &table, &bound, &pin)?;
        let result = self.execute_mvcc_query_with_fallback_reason(
            pin.store(),
            &query,
            backend,
            None,
            false,
        )?;
        self.finalize_relational_select(select, table, bound, access_path, result)
    }

    /// Test-only: total number of route-execution telemetry observations recorded so far.
    /// Used to assert that a route records its observation exactly once per execution.
    #[cfg(test)]
    pub(crate) fn route_execution_observation_count(&self) -> u64 {
        self.read_state
            .route_telemetry
            .route_execution_observation_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
