//! Relational SELECT binding + finalization (P0 §9.6 decomposition, behavior-
//! preserving): a focused `impl Engine` block that binds a parsed Select to the
//! catalog (bind_relational_select_for_execution), resolves rows via the MVCC
//! read query (pinned + unpinned), matches/sorts/orders relational keys against
//! filters & filter-groups, and finalizes the select result (host ORDER BY /
//! LIMIT / projection over the resolved rows).

use super::*;

impl Engine {
    /// Bind a relational SELECT against the catalog generation visible AS-OF the read's pinned
    /// boundary `s` (PART B catalog↔data co-pinning). Loads `committed_seq` ONCE → `s`, selects
    /// `catalog_as_of(s)`, clones the bound table out (a stable owned definition), and RETURNS `s`
    /// alongside so the caller pins the DATA at the SAME `s` (via [`Engine::pin_relational_read_at`] /
    /// [`Engine::relational_select_mvcc_query_pinned`]). Because `s` is loaded once here and threaded
    /// to the data pin, the catalog and data a statement reads are the same generation — a concurrent
    /// shape-changing DDL (which pushes its catalog gen BEFORE bumping `committed_seq`) can never split
    /// them. Every resident-route projection/aggregate method binds through here, so this one redirect
    /// co-pins the whole read path.
    pub(crate) fn bind_relational_select_for_execution(
        &self,
        select: &Select,
    ) -> Result<(RelationalTable, BoundRelationalSelect, Index), ExecuteError> {
        let s = self.committed_seq();
        let table = self
            .read_state
            .catalog_as_of(s)
            .relational_catalog
            .get(&select.table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" does not exist",
                    select.table
                )))
            })?
            .clone();
        let bound = bind_relational_select(&table, select)?;
        Ok((table, bound, s))
    }

    /// `relational_select_mvcc_query` with the read pin taken internally at the co-pinned boundary `s`
    /// (PART B). For callers that resolve the result rows from a SEPARATE residency generation (the GPU
    /// resident-route projection/aggregate methods) rather than the pinned CPU store — they need only
    /// the `MvccReadQuery` (key set / access path) and resolve against device memory, so a per-call pin
    /// is sufficient (the residency↔data snapshot consistency for those is enforced by the residency
    /// generation's `valid_through_index`/`invalidated_at_index`, not this pin). `s` is the boundary
    /// `bind_relational_select_for_execution` returned, so the pin matches the bound catalog.
    pub(crate) fn relational_select_mvcc_query_pinned(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        s: Index,
    ) -> Result<(MvccReadQuery, RelationalAccessPath), ExecuteError> {
        let pin = self.pin_relational_read_at(&select.table, s);
        self.relational_select_mvcc_query(select, table, bound, &pin)
    }

    pub(crate) fn relational_select_mvcc_query(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        pin: &RelationalReadPin,
    ) -> Result<(MvccReadQuery, RelationalAccessPath), ExecuteError> {
        let visibility = pin.visibility;
        let query_order = if select_is_aggregate(select) {
            None
        } else {
            bound.order
        };
        if bound.filter_groups.len() > 1 {
            if let Some((column_idx, keys)) = self
                .relational_keys_matching_same_column_equality_groups(
                    table,
                    &bound.filter_groups,
                    pin,
                )
            {
                let mut keys = keys;
                let order_column = query_order
                    .as_ref()
                    .map(|(idx, _)| table.columns[*idx].clone());
                if let Some((order_idx, descending)) = query_order {
                    keys = self
                        .relational_sort_keys_by_column(table, keys, order_idx, descending, pin)?;
                }
                let matched_keys = keys.len();
                let query = MvccReadQuery {
                    source: MvccReadSource::KeyBatchLookup { keys },
                    visibility,
                    filter: None,
                    order: query_order.is_none().then_some(MvccReadOrder::KeyAsc),
                    projection: MvccProjection::KeyValue,
                    limit: relational_select_pushed_limit(select, query_order.is_some()),
                };
                let table_column = table
                    .columns
                    .get(column_idx)
                    .expect("bound filter column came from table");
                let access_path = if let Some(order_column) = order_column {
                    RelationalAccessPath::OrderedKeyBatch {
                        table: select.table.clone(),
                        predicate_column: Some(table_column.name.clone()),
                        predicate_op: Some(SelectFilterOp::Eq),
                        order_column: order_column.name,
                        descending: query_order
                            .map(|(_, descending)| descending)
                            .unwrap_or(false),
                        matched_keys,
                    }
                } else {
                    RelationalAccessPath::EqualityIndex {
                        table: select.table.clone(),
                        column: table_column.name.clone(),
                        matched_keys,
                    }
                };
                return Ok((query, access_path));
            }
            let mut keys =
                self.relational_keys_matching_filter_groups(table, &bound.filter_groups, pin)?;
            let order_column = query_order
                .as_ref()
                .map(|(idx, _)| table.columns[*idx].clone());
            if let Some((order_idx, descending)) = query_order {
                keys =
                    self.relational_sort_keys_by_column(table, keys, order_idx, descending, pin)?;
            }
            let matched_keys = keys.len();
            let query = MvccReadQuery {
                source: MvccReadSource::KeyBatchLookup { keys },
                visibility,
                filter: None,
                order: query_order.is_none().then_some(MvccReadOrder::KeyAsc),
                projection: MvccProjection::KeyValue,
                limit: relational_select_pushed_limit(select, query_order.is_some()),
            };
            let access_path = if let Some(order_column) = order_column {
                RelationalAccessPath::OrderedKeyBatch {
                    table: select.table.clone(),
                    predicate_column: Some("<disjunction>".to_string()),
                    predicate_op: None,
                    order_column: order_column.name,
                    descending: query_order
                        .map(|(_, descending)| descending)
                        .unwrap_or(false),
                    matched_keys,
                }
            } else {
                RelationalAccessPath::DisjunctiveFilteredKeyBatch {
                    table: select.table.clone(),
                    predicate_group_count: bound.filter_groups.len(),
                    matched_keys,
                }
            };
            return Ok((query, access_path));
        }

        if bound.filters.len() > 1 {
            let mut keys = self.relational_keys_matching_filters(table, &bound.filters, pin)?;
            let order_column = query_order
                .as_ref()
                .map(|(idx, _)| table.columns[*idx].clone());
            if let Some((order_idx, descending)) = query_order {
                keys =
                    self.relational_sort_keys_by_column(table, keys, order_idx, descending, pin)?;
            }
            let matched_keys = keys.len();
            let query = MvccReadQuery {
                source: MvccReadSource::KeyBatchLookup { keys },
                visibility,
                filter: None,
                order: query_order.is_none().then_some(MvccReadOrder::KeyAsc),
                projection: MvccProjection::KeyValue,
                limit: relational_select_pushed_limit(select, query_order.is_some()),
            };
            let access_path = if let Some(order_column) = order_column {
                RelationalAccessPath::OrderedKeyBatch {
                    table: select.table.clone(),
                    predicate_column: Some("<conjunction>".to_string()),
                    predicate_op: None,
                    order_column: order_column.name,
                    descending: query_order
                        .map(|(_, descending)| descending)
                        .unwrap_or(false),
                    matched_keys,
                }
            } else {
                RelationalAccessPath::ConjunctiveFilteredKeyBatch {
                    table: select.table.clone(),
                    predicate_count: bound.filters.len(),
                    matched_keys,
                }
            };
            return Ok((query, access_path));
        }

        if let Some((column_idx, op, value)) = &bound.filter {
            let table_column = table
                .columns
                .get(*column_idx)
                .expect("bound filter column came from table");
            let mut keys = if *op == SelectFilterOp::Eq {
                // Equality fast-path: read the value-index from the SAME pinned generation the rows
                // will be resolved against (prereq #1, Stage 4). No second `load_table()` — so a
                // concurrent publish cannot slip a newer generation between the index lookup and the
                // row resolution. Stays O(log) + snapshot-consistent; does NOT scan version chains.
                //
                // The value-index is APPEND-ONLY, so a row updated in place appends its row_key once
                // per version that wrote this `(column, value)` slot — the same key can appear more
                // than once. Dedup before resolution; otherwise `KeyBatchLookup` would fetch (and
                // return) the row's single visible version multiple times. (The multi-predicate
                // equality paths already dedup via a `BTreeSet`; this single-predicate path is the
                // one that returned a raw `Vec`.)
                let mut keys = pin.index_keys(&table_column.name, &relational_index_value(value));
                keys.sort();
                keys.dedup();
                keys
            } else {
                self.relational_keys_matching_filter(table, *column_idx, *op, value, pin)?
            };
            let order_column = query_order
                .as_ref()
                .map(|(idx, _)| table.columns[*idx].clone());
            if let Some((order_idx, descending)) = query_order {
                keys =
                    self.relational_sort_keys_by_column(table, keys, order_idx, descending, pin)?;
            }
            let matched_keys = keys.len();
            let query = MvccReadQuery {
                source: MvccReadSource::KeyBatchLookup { keys },
                visibility,
                filter: None,
                order: query_order.is_none().then_some(MvccReadOrder::KeyAsc),
                projection: MvccProjection::KeyValue,
                limit: relational_select_pushed_limit(select, query_order.is_some()),
            };
            let access_path = if let Some(order_column) = order_column {
                RelationalAccessPath::OrderedKeyBatch {
                    table: select.table.clone(),
                    predicate_column: Some(table_column.name.clone()),
                    predicate_op: Some(*op),
                    order_column: order_column.name,
                    descending: query_order
                        .map(|(_, descending)| descending)
                        .unwrap_or(false),
                    matched_keys,
                }
            } else if *op == SelectFilterOp::Eq {
                RelationalAccessPath::EqualityIndex {
                    table: select.table.clone(),
                    column: table_column.name.clone(),
                    matched_keys,
                }
            } else {
                RelationalAccessPath::FilteredKeyBatch {
                    table: select.table.clone(),
                    predicate_column: table_column.name.clone(),
                    predicate_op: *op,
                    matched_keys,
                }
            };
            return Ok((query, access_path));
        }

        if let Some((order_idx, descending)) = query_order {
            let keys =
                self.relational_ordered_table_keys(table, visibility, order_idx, descending, pin)?;
            let matched_keys = keys.len();
            return Ok((
                MvccReadQuery {
                    source: MvccReadSource::KeyBatchLookup { keys },
                    visibility,
                    filter: None,
                    order: None,
                    projection: MvccProjection::KeyValue,
                    limit: relational_select_pushed_limit(select, true),
                },
                RelationalAccessPath::OrderedKeyBatch {
                    table: select.table.clone(),
                    predicate_column: None,
                    predicate_op: None,
                    order_column: table.columns[order_idx].name.clone(),
                    descending,
                    matched_keys,
                },
            ));
        }

        Ok((
            MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility,
                filter: Some(MvccReadFilter::KeyPrefix(relational_key_prefix(
                    &select.table,
                ))),
                order: Some(MvccReadOrder::KeyAsc),
                projection: MvccProjection::KeyValue,
                limit: relational_select_pushed_limit(select, false),
            },
            RelationalAccessPath::FullTableScan,
        ))
    }

    fn relational_keys_matching_same_column_equality_groups(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        pin: &RelationalReadPin,
    ) -> Option<(usize, Vec<String>)> {
        // Every equality slot reads the value-index of the SINGLE pinned generation (prereq #1):
        // rows + value-index are mutually consistent at one `commit_seq`.
        let mut column_idx = None;
        let mut keys = BTreeSet::new();
        for group in filter_groups {
            let [(idx, op, value)] = group.as_slice() else {
                return None;
            };
            if *op != SelectFilterOp::Eq {
                return None;
            }
            match column_idx {
                Some(existing_idx) if existing_idx != *idx => return None,
                Some(_) => {}
                None => column_idx = Some(*idx),
            }
            let column = table
                .columns
                .get(*idx)
                .expect("bound filter column came from table");
            keys.extend(pin.index_keys(&column.name, &relational_index_value(value)));
        }
        column_idx.map(|idx| (idx, keys.into_iter().collect()))
    }

    fn relational_sort_keys_by_column(
        &self,
        table: &RelationalTable,
        keys: Vec<String>,
        order_idx: usize,
        descending: bool,
        pin: &RelationalReadPin,
    ) -> Result<Vec<String>, ExecuteError> {
        let visibility = pin.visibility;
        let mut keyed_rows = Vec::new();
        for key in keys {
            let Some(tuple) = pin.store().tuple_fetch_by_key(&key, visibility)? else {
                continue;
            };
            let decoded = decode_relational_row(&tuple.value, &table.columns)?;
            keyed_rows.push((key, decoded[order_idx].clone()));
        }
        sort_relational_keys(&mut keyed_rows, descending);
        Ok(keyed_rows.into_iter().map(|(key, _)| key).collect())
    }

    fn relational_keys_matching_filter(
        &self,
        table: &RelationalTable,
        filter_idx: usize,
        op: SelectFilterOp,
        value: &SqlValue,
        pin: &RelationalReadPin,
    ) -> Result<Vec<String>, ExecuteError> {
        let visibility = pin.visibility;
        let mut cursor = pin.store().seq_scan_open(visibility)?;
        let prefix = relational_key_prefix(&table.name);
        let mut keys = Vec::new();
        while let Some(tuple) = cursor.next() {
            if !tuple.key.starts_with(&prefix) {
                continue;
            }
            let decoded = decode_relational_row(&tuple.value, &table.columns)?;
            if select_filter_matches(&decoded[filter_idx], op, value) {
                keys.push(tuple.key);
            }
        }
        keys.sort();
        Ok(keys)
    }

    fn relational_keys_matching_filters(
        &self,
        table: &RelationalTable,
        filters: &[(usize, SelectFilterOp, SqlValue)],
        pin: &RelationalReadPin,
    ) -> Result<Vec<String>, ExecuteError> {
        if filters.iter().all(|(_, op, _)| *op == SelectFilterOp::Eq) {
            // Conjunctive equality fast-path: intersect the per-column value-index hit sets, all
            // read from the SINGLE pinned generation (prereq #1, snapshot-consistent with the rows).
            let mut sets = filters
                .iter()
                .map(|(idx, _op, value)| {
                    let column = table
                        .columns
                        .get(*idx)
                        .expect("bound filter column came from table");
                    pin.index_keys(&column.name, &relational_index_value(value))
                        .into_iter()
                        .collect::<BTreeSet<_>>()
                })
                .collect::<Vec<_>>();
            if sets.is_empty() {
                return Ok(Vec::new());
            }
            sets.sort_by_key(|set| set.len());
            let mut matched = sets.remove(0);
            for set in sets {
                matched = matched.intersection(&set).cloned().collect();
                if matched.is_empty() {
                    break;
                }
            }
            return Ok(matched.into_iter().collect());
        }

        let visibility = pin.visibility;
        let mut cursor = pin.store().seq_scan_open(visibility)?;
        let prefix = relational_key_prefix(&table.name);
        let mut keys = Vec::new();
        while let Some(tuple) = cursor.next() {
            if !tuple.key.starts_with(&prefix) {
                continue;
            }
            let decoded = decode_relational_row(&tuple.value, &table.columns)?;
            if filters
                .iter()
                .all(|(idx, op, value)| select_filter_matches(&decoded[*idx], *op, value))
            {
                keys.push(tuple.key);
            }
        }
        keys.sort();
        Ok(keys)
    }

    fn relational_keys_matching_filter_groups(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        pin: &RelationalReadPin,
    ) -> Result<Vec<String>, ExecuteError> {
        let visibility = pin.visibility;
        let mut cursor = pin.store().seq_scan_open(visibility)?;
        let prefix = relational_key_prefix(&table.name);
        let mut keys = BTreeSet::new();
        while let Some(tuple) = cursor.next() {
            if !tuple.key.starts_with(&prefix) {
                continue;
            }
            let decoded = decode_relational_row(&tuple.value, &table.columns)?;
            if filter_groups.iter().any(|filters| {
                filters
                    .iter()
                    .all(|(idx, op, value)| select_filter_matches(&decoded[*idx], *op, value))
            }) {
                keys.insert(tuple.key);
            }
        }
        Ok(keys.into_iter().collect())
    }

    fn relational_ordered_table_keys(
        &self,
        table: &RelationalTable,
        visibility: StorageVisibility,
        order_idx: usize,
        descending: bool,
        pin: &RelationalReadPin,
    ) -> Result<Vec<String>, ExecuteError> {
        let mut cursor = pin.store().seq_scan_open(visibility)?;
        let prefix = relational_key_prefix(&table.name);
        let mut keyed_rows = Vec::new();
        while let Some(tuple) = cursor.next() {
            if !tuple.key.starts_with(&prefix) {
                continue;
            }
            let decoded = decode_relational_row(&tuple.value, &table.columns)?;
            keyed_rows.push((tuple.key, decoded[order_idx].clone()));
        }
        sort_relational_keys(&mut keyed_rows, descending);
        Ok(keyed_rows.into_iter().map(|(key, _)| key).collect())
    }

    pub(crate) fn finalize_relational_select(
        &self,
        select: &Select,
        table: RelationalTable,
        bound: BoundRelationalSelect,
        access_path: RelationalAccessPath,
        mvcc_result: MvccReadResult,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let mut rows = Vec::new();
        for row in mvcc_result.rows {
            let Some(value) = row.value else {
                continue;
            };
            let decoded = decode_relational_row(&value, &table.columns)?;
            let filter_matches = if bound.filter_groups.is_empty() {
                bound
                    .filters
                    .iter()
                    .all(|(idx, op, value)| select_filter_matches(&decoded[*idx], *op, value))
            } else {
                bound.filter_groups.iter().any(|filters| {
                    filters
                        .iter()
                        .all(|(idx, op, value)| select_filter_matches(&decoded[*idx], *op, value))
                })
            };
            if !filter_matches {
                continue;
            }
            rows.push(decoded);
        }

        if select_is_aggregate(select) {
            let mut aggregate_rows = match &select.projection {
                SelectProjection::CountAll => vec![vec![SqlValue::Int8(rows.len() as i64)]],
                SelectProjection::GroupedCount { .. } => {
                    let group_idx = bound
                        .group_by_index
                        .expect("grouped COUNT(*) validation requires GROUP BY");
                    let mut counts: BTreeMap<SqlValue, usize> = BTreeMap::new();
                    for row in rows {
                        *counts.entry(row[group_idx].clone()).or_default() += 1;
                    }
                    counts
                        .into_iter()
                        .map(|(value, count)| vec![value, SqlValue::Int8(count as i64)])
                        .collect::<Vec<_>>()
                }
                SelectProjection::Sum { column } => {
                    let sum_idx = validate_sum_column(&table, column)?;
                    let sum = rows
                        .iter()
                        .map(|row| int4_aggregate_value(&row[sum_idx], "SUM").map(i64::from))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .sum::<i64>();
                    vec![vec![SqlValue::Int8(sum)]]
                }
                SelectProjection::GroupedSum { sum_column, .. } => {
                    let group_idx = bound
                        .group_by_index
                        .expect("grouped SUM validation requires GROUP BY");
                    let sum_idx = validate_sum_column(&table, sum_column)?;
                    let mut sums: BTreeMap<SqlValue, i64> = BTreeMap::new();
                    for row in rows {
                        let value = i64::from(int4_aggregate_value(&row[sum_idx], "SUM")?);
                        *sums.entry(row[group_idx].clone()).or_default() += value;
                    }
                    sums.into_iter()
                        .map(|(value, sum)| vec![value, SqlValue::Int8(sum)])
                        .collect::<Vec<_>>()
                }
                SelectProjection::Avg { column } => {
                    let avg_idx = validate_avg_column(&table, column)?;
                    let mut sum = 0_i128;
                    let mut count = 0_usize;
                    for row in &rows {
                        sum += i128::from(int4_aggregate_value(&row[avg_idx], "AVG")?);
                        count += 1;
                    }
                    vec![vec![average_sql_value(sum, count)]]
                }
                SelectProjection::GroupedAvg { avg_column, .. } => {
                    let group_idx = bound
                        .group_by_index
                        .expect("grouped AVG validation requires GROUP BY");
                    let avg_idx = validate_avg_column(&table, avg_column)?;
                    let mut averages: BTreeMap<SqlValue, (i128, usize)> = BTreeMap::new();
                    for row in rows {
                        let value = int4_aggregate_value(&row[avg_idx], "AVG")?;
                        let entry = averages.entry(row[group_idx].clone()).or_default();
                        entry.0 += i128::from(value);
                        entry.1 += 1;
                    }
                    averages
                        .into_iter()
                        .map(|(value, (sum, count))| vec![value, average_sql_value(sum, count)])
                        .collect::<Vec<_>>()
                }
                SelectProjection::Min { column } | SelectProjection::Max { column } => {
                    let value_idx = relational_column_index(&table, column)?;
                    let value = if matches!(select.projection, SelectProjection::Min { .. }) {
                        rows.iter()
                            .map(|row| row[value_idx].clone())
                            .min_by(compare_sql_values)
                    } else {
                        rows.iter()
                            .map(|row| row[value_idx].clone())
                            .max_by(compare_sql_values)
                    }
                    .unwrap_or_else(|| SqlValue::Text(String::new()));
                    vec![vec![value]]
                }
                SelectProjection::GroupedMin { min_column, .. }
                | SelectProjection::GroupedMax {
                    max_column: min_column,
                    ..
                } => {
                    let group_idx = bound
                        .group_by_index
                        .expect("grouped MIN/MAX validation requires GROUP BY");
                    let value_idx = relational_column_index(&table, min_column)?;
                    let choose_min =
                        matches!(select.projection, SelectProjection::GroupedMin { .. });
                    let mut extrema: BTreeMap<SqlValue, SqlValue> = BTreeMap::new();
                    for row in rows {
                        extrema
                            .entry(row[group_idx].clone())
                            .and_modify(|current| {
                                let ordering = compare_sql_values(&row[value_idx], current);
                                if (choose_min && ordering.is_lt())
                                    || (!choose_min && ordering.is_gt())
                                {
                                    *current = row[value_idx].clone();
                                }
                            })
                            .or_insert_with(|| row[value_idx].clone());
                    }
                    extrema
                        .into_iter()
                        .map(|(value, extreme)| vec![value, extreme])
                        .collect::<Vec<_>>()
                }
                SelectProjection::All | SelectProjection::Columns(_) => unreachable!(),
            };

            if !select.having_groups.is_empty() {
                let group_idx = bound.group_by_index.ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "HAVING requires GROUP BY".to_string(),
                    ))
                })?;
                let group_column = &table.columns[group_idx].name;
                let aggregate_name = select_aggregate_result_column_name(select)
                    .expect("aggregate projection has result column name");
                aggregate_rows = aggregate_rows
                    .into_iter()
                    .filter_map(|row| {
                        let matches = grouped_row_matches_having(
                            select,
                            group_column,
                            &row[0],
                            aggregate_name,
                            row.last().expect("aggregate row has result value"),
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
                let order_idx = if select_is_aggregate_result_column(select, &order.column) {
                    aggregate_rows.first().map_or(0, |row| row.len() - 1)
                } else {
                    0
                };
                aggregate_rows.sort_by(|left, right| {
                    compare_sql_values(&left[order_idx], &right[order_idx])
                        .then_with(|| compare_sql_values(&left[0], &right[0]))
                });
                if order.descending {
                    aggregate_rows.reverse();
                }
            }
            if let Some(offset) = select.offset {
                aggregate_rows = aggregate_rows.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                aggregate_rows.truncate(limit);
            }

            return Ok(RelationalSelectResult {
                columns: bound.selected_columns,
                rows: aggregate_rows,
                planned_target: mvcc_result.planned_target,
                executed_target: mvcc_result.executed_target,
                fallback_reason: mvcc_result.fallback_reason,
                access_path,
            });
        }

        if select.distinct {
            let mut projected = Vec::new();
            let mut seen = BTreeSet::new();
            for row in rows {
                let selected = bound
                    .selected_indexes
                    .iter()
                    .map(|idx| row[*idx].clone())
                    .collect::<Vec<_>>();
                if seen.insert(selected.clone()) {
                    projected.push(selected);
                }
            }
            if let Some((order_idx, descending)) = &bound.order {
                let selected_order_idx = bound
                    .selected_indexes
                    .iter()
                    .position(|idx| idx == order_idx)
                    .expect("DISTINCT ORDER BY was validated against selected columns");
                projected.sort_by(|left, right| {
                    compare_sql_values(&left[selected_order_idx], &right[selected_order_idx])
                });
                if *descending {
                    projected.reverse();
                }
            }
            if let Some(offset) = select.offset {
                projected = projected.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                projected.truncate(limit);
            }

            return Ok(RelationalSelectResult {
                columns: bound.selected_columns,
                rows: projected,
                planned_target: mvcc_result.planned_target,
                executed_target: mvcc_result.executed_target,
                fallback_reason: mvcc_result.fallback_reason,
                access_path,
            });
        }

        if !matches!(access_path, RelationalAccessPath::OrderedKeyBatch { .. }) {
            if let Some((idx, descending)) = &bound.order {
                rows.sort_by(|left, right| compare_sql_values(&left[*idx], &right[*idx]));
                if *descending {
                    rows.reverse();
                }
            }
        }
        if !relational_select_limit_satisfied_by_access_path(select, &access_path) {
            if let Some(offset) = select.offset {
                rows = rows.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                rows.truncate(limit);
            }
        }

        let rows = rows
            .into_iter()
            .map(|row| {
                bound
                    .selected_indexes
                    .iter()
                    .map(|idx| row[*idx].clone())
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut fallback_reason = mvcc_result.fallback_reason;
        if fallback_reason.is_none()
            && relational_select_needs_host_sql_finalization(select, &access_path)
        {
            fallback_reason = Some(FallbackReason::GpuMvccReadParityGap);
            self.metrics
                .inc_fallback(FallbackReason::GpuMvccReadParityGap);
        }

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: mvcc_result.planned_target,
            executed_target: mvcc_result.executed_target,
            fallback_reason,
            access_path,
        })
    }
}
