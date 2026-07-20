// Legacy host-backed select execution. This is parity/bootstrap debt, not a product execution path.

use super::{
    bool_text, column_for_sql_type, int8_column, numeric_column, object_access_permission_error,
    sql_value_matches_type, write_error, write_select_rows, CatalogColumn, Column, ErrorField,
    ReadWrite, SelectResult, Session, Table,
};
use gpu_db_protocol::{Command, SelectFilterOp, SelectProjection, SqlValue, TablePrivilege};
use std::collections::BTreeMap;
use std::io;

fn compare_sql_values(left: &SqlValue, right: &SqlValue) -> std::cmp::Ordering {
    use gpu_db_protocol::Decimal128;
    match (left, right) {
        (SqlValue::Parameter { .. }, _) | (_, SqlValue::Parameter { .. }) => {
            unreachable!("legacy execution never receives an unbound prepared parameter")
        }
        // NULL sorts lowest in this internal total order (this server does not produce NULL
        // values; the arm only keeps the comparator total now that SqlValue has a Null variant).
        (SqlValue::Null, SqlValue::Null) => std::cmp::Ordering::Equal,
        (SqlValue::Null, _) => std::cmp::Ordering::Less,
        (_, SqlValue::Null) => std::cmp::Ordering::Greater,
        // smallint widens to int4 for every comparison (PG's numeric tower); recurse widened.
        (SqlValue::Int2(left), right) => {
            compare_sql_values(&SqlValue::Int4(i32::from(*left)), right)
        }
        (left, SqlValue::Int2(right)) => {
            compare_sql_values(left, &SqlValue::Int4(i32::from(*right)))
        }
        (SqlValue::Int4(left), SqlValue::Int4(right)) => left.cmp(right),
        (SqlValue::Int8(left), SqlValue::Int8(right)) => left.cmp(right),
        (SqlValue::Int4(left), SqlValue::Int8(right)) => i64::from(*left).cmp(right),
        (SqlValue::Int8(left), SqlValue::Int4(right)) => left.cmp(&i64::from(*right)),
        // Numeric is now a fixed-point Decimal128: compare scale-aligned (1.0 == 1.00), and
        // promote an integer to a scale-0 decimal for cross-type comparison.
        (SqlValue::Numeric(left), SqlValue::Numeric(right)) => left.cmp(right),
        (SqlValue::Int4(left), SqlValue::Numeric(right)) => {
            Decimal128::new(i128::from(*left), 0).cmp(right)
        }
        (SqlValue::Int8(left), SqlValue::Numeric(right)) => {
            Decimal128::new(i128::from(*left), 0).cmp(right)
        }
        (SqlValue::Numeric(left), SqlValue::Int4(right)) => {
            left.cmp(&Decimal128::new(i128::from(*right), 0))
        }
        (SqlValue::Numeric(left), SqlValue::Int8(right)) => {
            left.cmp(&Decimal128::new(i128::from(*right), 0))
        }
        (SqlValue::Bool(left), SqlValue::Bool(right)) => left.cmp(right),
        (SqlValue::Text(left), SqlValue::Text(right)) => left.cmp(right),
        (
            SqlValue::Int4(_) | SqlValue::Int8(_) | SqlValue::Numeric(_),
            SqlValue::Bool(_) | SqlValue::Text(_),
        ) => std::cmp::Ordering::Less,
        (SqlValue::Bool(_), SqlValue::Int4(_) | SqlValue::Int8(_) | SqlValue::Numeric(_)) => {
            std::cmp::Ordering::Greater
        }
        (SqlValue::Bool(_), SqlValue::Text(_)) => std::cmp::Ordering::Less,
        (
            SqlValue::Text(_),
            SqlValue::Int4(_) | SqlValue::Int8(_) | SqlValue::Numeric(_) | SqlValue::Bool(_),
        ) => std::cmp::Ordering::Greater,
        // Date is its own ordering tier (it sorts after the other types); same-type dates compare by
        // their day count. Cross-type date comparisons are type errors the engine rejects upstream.
        (SqlValue::Date(left), SqlValue::Date(right)) => left.cmp(right),
        (
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_),
            SqlValue::Date(_),
        ) => std::cmp::Ordering::Less,
        (
            SqlValue::Date(_),
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_),
        ) => std::cmp::Ordering::Greater,
        // Timestamp is the last tier (sorts after date); same-type compares by microsecond count.
        (SqlValue::Timestamp(left), SqlValue::Timestamp(right)) => left.cmp(right),
        (
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_)
            | SqlValue::Date(_),
            SqlValue::Timestamp(_),
        ) => std::cmp::Ordering::Less,
        (
            SqlValue::Timestamp(_),
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_)
            | SqlValue::Date(_),
        ) => std::cmp::Ordering::Greater,
        // Uuid is the final tier; same-type compares byte-wise (PG's uuid order).
        (SqlValue::Uuid(left), SqlValue::Uuid(right)) => left.cmp(right),
        (
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_)
            | SqlValue::Date(_)
            | SqlValue::Timestamp(_),
            SqlValue::Uuid(_),
        ) => std::cmp::Ordering::Less,
        (
            SqlValue::Uuid(_),
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_)
            | SqlValue::Date(_)
            | SqlValue::Timestamp(_),
        ) => std::cmp::Ordering::Greater,
    }
}

pub(super) fn select_filter_matches(left: &SqlValue, op: SelectFilterOp, right: &SqlValue) -> bool {
    match op {
        SelectFilterOp::Eq => left == right,
        SelectFilterOp::Lt => compare_sql_values(left, right).is_lt(),
        SelectFilterOp::Lte => !compare_sql_values(left, right).is_gt(),
        SelectFilterOp::Gt => compare_sql_values(left, right).is_gt(),
        SelectFilterOp::Gte => !compare_sql_values(left, right).is_lt(),
        SelectFilterOp::LikePrefix => match (left, right) {
            (SqlValue::Text(left), SqlValue::Text(prefix)) => left.starts_with(prefix),
            _ => false,
        },
    }
}

pub(super) fn row_matches_select_filters(
    table: &Table,
    row: &[SqlValue],
    select: &gpu_db_protocol::Select,
) -> Result<bool, ErrorField> {
    let filter_groups = if select.filter_groups.is_empty() {
        if select.filters.is_empty() {
            select
                .filter
                .iter()
                .cloned()
                .map(|filter| vec![filter])
                .collect::<Vec<_>>()
        } else {
            vec![select.filters.clone()]
        }
    } else {
        select.filter_groups.clone()
    };

    if filter_groups.is_empty() {
        return Ok(true);
    }

    for filters in filter_groups {
        let mut group_matches = true;
        for filter in filters {
            let Some(idx) = table
                .columns
                .iter()
                .position(|column| column.def.name == filter.column)
            else {
                return Err(ErrorField {
                    code: "42703",
                    message: "column does not exist",
                    position: None,
                });
            };
            if !select_filter_matches(&row[idx], filter.op, &filter.value) {
                group_matches = false;
                break;
            }
        }
        if group_matches {
            return Ok(true);
        }
    }

    Ok(false)
}

pub(super) fn row_matches_delete_filters(
    table: &Table,
    row: &[SqlValue],
    delete: &gpu_db_protocol::Delete,
) -> Result<bool, ErrorField> {
    let filter_groups = if delete.filter_groups.is_empty() {
        if delete.filters.is_empty() {
            delete
                .filter
                .iter()
                .cloned()
                .map(|filter| vec![filter])
                .collect::<Vec<_>>()
        } else {
            vec![delete.filters.clone()]
        }
    } else {
        delete.filter_groups.clone()
    };

    if filter_groups.is_empty() {
        return Err(ErrorField {
            code: "42601",
            message: "DELETE requires WHERE filters",
            position: None,
        });
    }

    for filters in filter_groups {
        let mut group_matches = true;
        for filter in filters {
            let Some(idx) = table
                .columns
                .iter()
                .position(|column| column.def.name == filter.column)
            else {
                return Err(ErrorField {
                    code: "42703",
                    message: "column does not exist",
                    position: None,
                });
            };
            if !sql_value_matches_type(&filter.value, table.columns[idx].def.ty) {
                return Err(ErrorField {
                    code: "42804",
                    message: "column type mismatch",
                    position: None,
                });
            }
            if !select_filter_matches(&row[idx], filter.op, &filter.value) {
                group_matches = false;
                break;
            }
        }
        if group_matches {
            return Ok(true);
        }
    }

    Ok(false)
}

fn grouped_row_matches_having(
    select: &gpu_db_protocol::Select,
    group_column: &str,
    group_value: &SqlValue,
    aggregate_name: &'static str,
    aggregate_value: &SqlValue,
) -> Result<bool, ErrorField> {
    if select.having_groups.is_empty() {
        return Ok(true);
    }

    for filters in &select.having_groups {
        let mut group_matches = true;
        for filter in filters {
            let value = if filter.column == group_column {
                group_value
            } else if filter.column.eq_ignore_ascii_case(aggregate_name) {
                aggregate_value
            } else {
                return Err(ErrorField {
                    code: "0A000",
                    message: "HAVING must reference grouped column or aggregate result",
                    position: None,
                });
            };
            if !select_filter_matches(value, filter.op, &filter.value) {
                group_matches = false;
                break;
            }
        }
        if group_matches {
            return Ok(true);
        }
    }

    Ok(false)
}

pub(super) fn execute_select_result(
    session: &Session,
    select: &gpu_db_protocol::Select,
) -> Result<SelectResult, ErrorField> {
    execute_select_result_inner(session, select, true)
}

fn execute_select_result_inner(
    session: &Session,
    select: &gpu_db_protocol::Select,
    enforce_relation_acl: bool,
) -> Result<SelectResult, ErrorField> {
    // Multi-key ORDER BY (`ORDER BY a, b, ...`) is GPU-only -- it runs on the general Expr executor's
    // bitonic-sort path (the consolidated server). This legacy pgwire handler sorts on the CPU off the
    // first key only, so rather than silently mis-order a multi-key request, reject it cleanly.
    if select.order_by.len() > 1 {
        return Err(ErrorField {
            code: "0A000",
            message:
                "multi-key ORDER BY is supported only on the GPU executor (consolidated server)",
            position: None,
        });
    }
    let Some(table) = session.tables.get(&select.table) else {
        if let Some(view) = session.views.get(&select.table) {
            if !select_is_plain_view_scan(select) {
                return Err(ErrorField {
                    code: "0A000",
                    message: "only plain SELECT * FROM view is supported for views",
                    position: None,
                });
            }
            if enforce_relation_acl {
                if let Some(error) =
                    object_access_permission_error(session, &select.table, TablePrivilege::Select)
                {
                    return Err(error);
                }
            }
            return execute_select_result_inner(session, &view.query, false);
        }
        if let Some(view) = session.materialized_views.get(&select.table) {
            if !select_is_plain_view_scan(select) {
                return Err(ErrorField {
                    code: "0A000",
                    message:
                        "only plain SELECT * FROM materialized view is supported for materialized views",
                    position: None,
                });
            }
            if enforce_relation_acl {
                if let Some(error) =
                    object_access_permission_error(session, &select.table, TablePrivilege::Select)
                {
                    return Err(error);
                }
            }
            return Ok(SelectResult {
                columns: view
                    .columns
                    .iter()
                    .map(|column| column_for_sql_type(column.def.ty, &column.def.name))
                    .collect(),
                rows: view
                    .rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|value| Some(format_sql_value(value)))
                            .collect()
                    })
                    .collect(),
            });
        }
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    };
    if enforce_relation_acl {
        if let Some(error) =
            object_access_permission_error(session, &select.table, TablePrivilege::Select)
        {
            return Err(error);
        }
    }
    if matches!(
        select.projection,
        SelectProjection::CountAll
            | SelectProjection::GroupedCount { .. }
            | SelectProjection::Sum { .. }
            | SelectProjection::GroupedSum { .. }
            | SelectProjection::Avg { .. }
            | SelectProjection::GroupedAvg { .. }
            | SelectProjection::Min { .. }
            | SelectProjection::GroupedMin { .. }
            | SelectProjection::Max { .. }
            | SelectProjection::GroupedMax { .. }
    ) {
        return execute_aggregate_select_result(table, select);
    }

    if select.group_by.is_some() {
        return Err(ErrorField {
            code: "0A000",
            message: "GROUP BY requires COUNT(*) projection",
            position: None,
        });
    }

    let selected_columns = match &select.projection {
        SelectProjection::All => table.columns.clone(),
        SelectProjection::Columns(columns) => {
            let mut selected = Vec::with_capacity(columns.len());
            for column in columns {
                let Some(def) = table
                    .columns
                    .iter()
                    .find(|candidate| candidate.def.name == *column)
                else {
                    return Err(ErrorField {
                        code: "42703",
                        message: "column does not exist",
                        position: None,
                    });
                };
                selected.push(def.clone());
            }
            selected
        }
        SelectProjection::CountAll
        | SelectProjection::GroupedCount { .. }
        | SelectProjection::Sum { .. }
        | SelectProjection::GroupedSum { .. }
        | SelectProjection::Avg { .. }
        | SelectProjection::GroupedAvg { .. }
        | SelectProjection::Min { .. }
        | SelectProjection::GroupedMin { .. }
        | SelectProjection::Max { .. }
        | SelectProjection::GroupedMax { .. }
        | SelectProjection::CountDistinct { .. }
        | SelectProjection::GroupedAggregates { .. } => unreachable!(),
    };
    if select.distinct {
        match &select.projection {
            SelectProjection::All => {
                return Err(ErrorField {
                    code: "0A000",
                    message: "SELECT DISTINCT * is unsupported",
                    position: None,
                });
            }
            SelectProjection::Columns(columns) => {
                if let Some(order) = select.order_by.first() {
                    if !columns.iter().any(|column| column == &order.column) {
                        return Err(ErrorField {
                            code: "0A000",
                            message: "SELECT DISTINCT ORDER BY must reference a selected column",
                            position: None,
                        });
                    }
                }
            }
            SelectProjection::CountAll
            | SelectProjection::GroupedCount { .. }
            | SelectProjection::Sum { .. }
            | SelectProjection::GroupedSum { .. }
            | SelectProjection::Avg { .. }
            | SelectProjection::GroupedAvg { .. }
            | SelectProjection::Min { .. }
            | SelectProjection::GroupedMin { .. }
            | SelectProjection::Max { .. }
            | SelectProjection::GroupedMax { .. }
            | SelectProjection::CountDistinct { .. }
            | SelectProjection::GroupedAggregates { .. } => unreachable!(),
        }
    }
    let selected_indexes = selected_columns
        .iter()
        .map(|selected| {
            table
                .columns
                .iter()
                .position(|column| column.def.name == selected.def.name)
                .expect("selected column came from table")
        })
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for row in &table.rows {
        match row_matches_select_filters(table, row, select) {
            Ok(true) => rows.push(row.clone()),
            Ok(false) => {}
            Err(error) => return Err(error),
        }
    }
    if select.distinct {
        let mut projected = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for row in &rows {
            let selected = selected_indexes
                .iter()
                .map(|idx| row[*idx].clone())
                .collect::<Vec<_>>();
            if seen.insert(selected.clone()) {
                projected.push(selected);
            }
        }
        if let Some(order) = select.order_by.first() {
            let Some(selected_order_idx) = selected_columns
                .iter()
                .position(|column| column.def.name == order.column)
            else {
                return Err(ErrorField {
                    code: "42703",
                    message: "column does not exist",
                    position: None,
                });
            };
            projected.sort_by(|left, right| {
                compare_sql_values(&left[selected_order_idx], &right[selected_order_idx])
            });
            if order.descending {
                projected.reverse();
            }
        }
        if let Some(offset) = select.offset {
            projected = projected.into_iter().skip(offset).collect();
        }
        if let Some(limit) = select.limit {
            projected.truncate(limit);
        }
        let columns = selected_columns
            .iter()
            .map(|column| column_for_sql_type(column.def.ty, &column.def.name))
            .collect::<Vec<_>>();
        let rows = projected
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| Some(format_sql_value(value)))
                    .collect()
            })
            .collect::<Vec<_>>();
        return Ok(SelectResult { columns, rows });
    }
    if let Some(order) = select.order_by.first() {
        let Some(idx) = table
            .columns
            .iter()
            .position(|column| column.def.name == order.column)
        else {
            return Err(ErrorField {
                code: "42703",
                message: "column does not exist",
                position: None,
            });
        };
        rows.sort_by(|left, right| compare_sql_values(&left[idx], &right[idx]));
        if order.descending {
            rows.reverse();
        }
    }
    if let Some(offset) = select.offset {
        rows = rows.into_iter().skip(offset).collect();
    }
    if let Some(limit) = select.limit {
        rows.truncate(limit);
    }
    let columns = selected_columns
        .iter()
        .map(|column| column_for_sql_type(column.def.ty, &column.def.name))
        .collect::<Vec<_>>();
    let rows = rows
        .iter()
        .map(|row| {
            selected_indexes
                .iter()
                .map(|idx| Some(format_sql_value(&row[*idx])))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(SelectResult { columns, rows })
}

fn select_is_plain_view_scan(select: &gpu_db_protocol::Select) -> bool {
    !select.distinct
        && matches!(select.projection, SelectProjection::All)
        && select.group_by.is_none()
        && select.having_groups.is_empty()
        && select.filter.is_none()
        && select.filters.is_empty()
        && select.filter_groups.is_empty()
        && select.order_by.is_empty()
        && select.limit.is_none()
        && select.offset.is_none()
}

fn execute_aggregate_select_result(
    table: &Table,
    select: &gpu_db_protocol::Select,
) -> Result<SelectResult, ErrorField> {
    let mut rows = Vec::new();
    for row in &table.rows {
        match row_matches_select_filters(table, row, select) {
            Ok(true) => rows.push(row.clone()),
            Ok(false) => {}
            Err(error) => return Err(error),
        }
    }
    if !select.having_groups.is_empty() && select.group_by.is_none() {
        return Err(ErrorField {
            code: "0A000",
            message: "HAVING requires GROUP BY",
            position: None,
        });
    }

    match &select.projection {
        // GroupedAggregates (N aggregates per group) and a bare COUNT(DISTINCT v) are produced only by
        // the engine Expr parser, not the wire-protocol parser that feeds this handler.
        SelectProjection::GroupedAggregates { .. } | SelectProjection::CountDistinct { .. } => {
            unreachable!()
        }
        SelectProjection::CountAll => {
            if select.group_by.is_some() {
                return Err(ErrorField {
                    code: "0A000",
                    message: "GROUP BY requires grouped COUNT(*) projection",
                    position: None,
                });
            }
            if let Some(order) = select.order_by.first() {
                if !order.column.eq_ignore_ascii_case("count") {
                    return Err(ErrorField {
                        code: "0A000",
                        message: "COUNT(*) ORDER BY only supports count",
                        position: None,
                    });
                }
            }
            let mut aggregate_rows = vec![vec![Some(rows.len().to_string())]];
            if let Some(offset) = select.offset {
                aggregate_rows = aggregate_rows.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                aggregate_rows.truncate(limit);
            }
            Ok(SelectResult {
                columns: vec![int8_column("count")],
                rows: aggregate_rows,
            })
        }
        SelectProjection::GroupedCount { column } => {
            let Some(group_by) = &select.group_by else {
                return Err(ErrorField {
                    code: "0A000",
                    message: "grouped COUNT(*) requires GROUP BY",
                    position: None,
                });
            };
            if group_by != column {
                return Err(ErrorField {
                    code: "0A000",
                    message: "GROUP BY column must match grouped COUNT(*) projection",
                    position: None,
                });
            }
            let Some(group_idx) = table
                .columns
                .iter()
                .position(|candidate| candidate.def.name == *column)
            else {
                return Err(ErrorField {
                    code: "42703",
                    message: "column does not exist",
                    position: None,
                });
            };
            let mut counts: BTreeMap<SqlValue, usize> = BTreeMap::new();
            for row in rows {
                *counts.entry(row[group_idx].clone()).or_default() += 1;
            }
            let mut grouped = counts.into_iter().collect::<Vec<_>>();
            grouped = grouped
                .into_iter()
                .filter_map(|(group_value, count)| {
                    let aggregate = SqlValue::Int8(count as i64);
                    match grouped_row_matches_having(
                        select,
                        column,
                        &group_value,
                        "count",
                        &aggregate,
                    ) {
                        Ok(true) => Some(Ok((group_value, count))),
                        Ok(false) => None,
                        Err(error) => Some(Err(error)),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(order) = select.order_by.first() {
                if order.column == *column {
                    grouped.sort_by(|(left, _), (right, _)| compare_sql_values(left, right));
                } else if order.column.eq_ignore_ascii_case("count") {
                    grouped.sort_by(|(left_value, left_count), (right_value, right_count)| {
                        left_count
                            .cmp(right_count)
                            .then_with(|| compare_sql_values(left_value, right_value))
                    });
                } else {
                    return Err(ErrorField {
                        code: "0A000",
                        message: "GROUP BY ORDER BY must reference grouped column or count",
                        position: None,
                    });
                }
                if order.descending {
                    grouped.reverse();
                }
            }
            if let Some(offset) = select.offset {
                grouped = grouped.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                grouped.truncate(limit);
            }
            let group_column = table
                .columns
                .get(group_idx)
                .expect("group column index came from table");
            let columns = vec![
                column_for_sql_type(group_column.def.ty, &group_column.def.name),
                int8_column("count"),
            ];
            let rows = grouped
                .into_iter()
                .map(|(value, count)| vec![Some(format_sql_value(&value)), Some(count.to_string())])
                .collect();
            Ok(SelectResult { columns, rows })
        }
        SelectProjection::Sum { column } => {
            if select.group_by.is_some() {
                return Err(ErrorField {
                    code: "0A000",
                    message: "GROUP BY requires grouped SUM projection",
                    position: None,
                });
            }
            if let Some(order) = select.order_by.first() {
                if !order.column.eq_ignore_ascii_case("sum") {
                    return Err(ErrorField {
                        code: "0A000",
                        message: "SUM ORDER BY only supports sum",
                        position: None,
                    });
                }
            }
            let sum_idx = int4_column_index(table, column)?;
            let sum = rows
                .iter()
                .map(|row| int4_value(&row[sum_idx]))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(i64::from)
                .sum::<i64>();
            let mut aggregate_rows = vec![vec![Some(sum.to_string())]];
            if let Some(offset) = select.offset {
                aggregate_rows = aggregate_rows.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                aggregate_rows.truncate(limit);
            }
            Ok(SelectResult {
                columns: vec![int8_column("sum")],
                rows: aggregate_rows,
            })
        }
        SelectProjection::GroupedSum {
            group_column,
            sum_column,
        } => {
            let Some(group_by) = &select.group_by else {
                return Err(ErrorField {
                    code: "0A000",
                    message: "grouped SUM requires GROUP BY",
                    position: None,
                });
            };
            if group_by != group_column {
                return Err(ErrorField {
                    code: "0A000",
                    message: "GROUP BY column must match grouped SUM projection",
                    position: None,
                });
            }
            let Some(group_idx) = table
                .columns
                .iter()
                .position(|candidate| candidate.def.name == *group_column)
            else {
                return Err(ErrorField {
                    code: "42703",
                    message: "column does not exist",
                    position: None,
                });
            };
            let sum_idx = int4_column_index(table, sum_column)?;
            let mut sums: BTreeMap<SqlValue, i64> = BTreeMap::new();
            for row in rows {
                let value = int4_value(&row[sum_idx])?;
                *sums.entry(row[group_idx].clone()).or_default() += i64::from(value);
            }
            let mut grouped = sums.into_iter().collect::<Vec<_>>();
            grouped = grouped
                .into_iter()
                .filter_map(|(group_value, sum)| {
                    let aggregate = SqlValue::Int8(sum);
                    match grouped_row_matches_having(
                        select,
                        group_column,
                        &group_value,
                        "sum",
                        &aggregate,
                    ) {
                        Ok(true) => Some(Ok((group_value, sum))),
                        Ok(false) => None,
                        Err(error) => Some(Err(error)),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(order) = select.order_by.first() {
                if order.column == *group_column {
                    grouped.sort_by(|(left, _), (right, _)| compare_sql_values(left, right));
                } else if order.column.eq_ignore_ascii_case("sum") {
                    grouped.sort_by(|(left_value, left_sum), (right_value, right_sum)| {
                        left_sum
                            .cmp(right_sum)
                            .then_with(|| compare_sql_values(left_value, right_value))
                    });
                } else {
                    return Err(ErrorField {
                        code: "0A000",
                        message: "GROUP BY ORDER BY must reference grouped column or sum",
                        position: None,
                    });
                }
                if order.descending {
                    grouped.reverse();
                }
            }
            if let Some(offset) = select.offset {
                grouped = grouped.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                grouped.truncate(limit);
            }
            let group_column_def = table
                .columns
                .get(group_idx)
                .expect("group column index came from table");
            let columns = vec![
                column_for_sql_type(group_column_def.def.ty, &group_column_def.def.name),
                int8_column("sum"),
            ];
            let rows = grouped
                .into_iter()
                .map(|(value, sum)| vec![Some(format_sql_value(&value)), Some(sum.to_string())])
                .collect();
            Ok(SelectResult { columns, rows })
        }
        SelectProjection::Avg { column } => {
            if select.group_by.is_some() {
                return Err(ErrorField {
                    code: "0A000",
                    message: "GROUP BY requires grouped AVG projection",
                    position: None,
                });
            }
            if let Some(order) = select.order_by.first() {
                if !order.column.eq_ignore_ascii_case("avg") {
                    return Err(ErrorField {
                        code: "0A000",
                        message: "AVG ORDER BY only supports avg",
                        position: None,
                    });
                }
            }
            let avg_idx = int4_column_index_for_aggregate(table, column, "AVG")?;
            let mut sum = 0_i128;
            let mut count = 0_usize;
            for row in &rows {
                sum += i128::from(int4_value_for_aggregate(&row[avg_idx], "AVG")?);
                count += 1;
            }
            let mut aggregate_rows = vec![vec![average_text(sum, count)]];
            if let Some(offset) = select.offset {
                aggregate_rows = aggregate_rows.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                aggregate_rows.truncate(limit);
            }
            Ok(SelectResult {
                columns: vec![numeric_column("avg")],
                rows: aggregate_rows,
            })
        }
        SelectProjection::GroupedAvg {
            group_column,
            avg_column,
        } => {
            let Some(group_by) = &select.group_by else {
                return Err(ErrorField {
                    code: "0A000",
                    message: "grouped AVG requires GROUP BY",
                    position: None,
                });
            };
            if group_by != group_column {
                return Err(ErrorField {
                    code: "0A000",
                    message: "GROUP BY column must match grouped AVG projection",
                    position: None,
                });
            }
            let group_idx = column_index(table, group_column)?;
            let avg_idx = int4_column_index_for_aggregate(table, avg_column, "AVG")?;
            let mut averages: BTreeMap<SqlValue, (i128, usize)> = BTreeMap::new();
            for row in rows {
                let value = int4_value_for_aggregate(&row[avg_idx], "AVG")?;
                let entry = averages.entry(row[group_idx].clone()).or_default();
                entry.0 += i128::from(value);
                entry.1 += 1;
            }
            let mut grouped = averages.into_iter().collect::<Vec<_>>();
            grouped = grouped
                .into_iter()
                .filter_map(|(group_value, (sum, count))| {
                    // HAVING evaluates the AVG against a numeric literal; both must be Decimal128.
                    // `average_text` yields a scale-16 string, so parse it back at scale 16.
                    let aggregate = SqlValue::Numeric(
                        gpu_db_protocol::Decimal128::parse_at_scale(
                            &average_text(sum, count)
                                .unwrap_or_else(|| "0.0000000000000000".to_string()),
                            16,
                        )
                        .unwrap_or(gpu_db_protocol::Decimal128::ZERO),
                    );
                    match grouped_row_matches_having(
                        select,
                        group_column,
                        &group_value,
                        "avg",
                        &aggregate,
                    ) {
                        Ok(true) => Some(Ok((group_value, (sum, count)))),
                        Ok(false) => None,
                        Err(error) => Some(Err(error)),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(order) = select.order_by.first() {
                if order.column == *group_column {
                    grouped.sort_by(|(left, _), (right, _)| compare_sql_values(left, right));
                } else if order.column.eq_ignore_ascii_case("avg") {
                    grouped.sort_by(
                        |(left_value, (left_sum, left_count)),
                         (right_value, (right_sum, right_count))| {
                            compare_averages(*left_sum, *left_count, *right_sum, *right_count)
                                .then_with(|| compare_sql_values(left_value, right_value))
                        },
                    );
                } else {
                    return Err(ErrorField {
                        code: "0A000",
                        message: "GROUP BY ORDER BY must reference grouped column or avg",
                        position: None,
                    });
                }
                if order.descending {
                    grouped.reverse();
                }
            }
            if let Some(offset) = select.offset {
                grouped = grouped.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                grouped.truncate(limit);
            }
            let columns = vec![
                aggregate_result_column(table, group_idx, group_column),
                numeric_column("avg"),
            ];
            let rows = grouped
                .into_iter()
                .map(|(value, (sum, count))| {
                    vec![Some(format_sql_value(&value)), average_text(sum, count)]
                })
                .collect();
            Ok(SelectResult { columns, rows })
        }
        SelectProjection::Min { column } | SelectProjection::Max { column } => {
            let aggregate_name = match &select.projection {
                SelectProjection::Min { .. } => "min",
                SelectProjection::Max { .. } => "max",
                _ => unreachable!(),
            };
            if select.group_by.is_some() {
                return Err(ErrorField {
                    code: "0A000",
                    message: "GROUP BY requires grouped MIN/MAX projection",
                    position: None,
                });
            }
            if let Some(order) = select.order_by.first() {
                if !order.column.eq_ignore_ascii_case(aggregate_name) {
                    return Err(ErrorField {
                        code: "0A000",
                        message: "MIN/MAX ORDER BY only supports the aggregate result",
                        position: None,
                    });
                }
            }
            let value_idx = column_index(table, column)?;
            let value = if matches!(select.projection, SelectProjection::Min { .. }) {
                rows.iter()
                    .map(|row| row[value_idx].clone())
                    .min_by(compare_sql_values)
            } else {
                rows.iter()
                    .map(|row| row[value_idx].clone())
                    .max_by(compare_sql_values)
            };
            let mut aggregate_rows = vec![vec![value.as_ref().map(format_sql_value)]];
            if let Some(offset) = select.offset {
                aggregate_rows = aggregate_rows.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                aggregate_rows.truncate(limit);
            }
            Ok(SelectResult {
                columns: vec![aggregate_result_column(table, value_idx, aggregate_name)],
                rows: aggregate_rows,
            })
        }
        SelectProjection::GroupedMin {
            group_column,
            min_column,
        }
        | SelectProjection::GroupedMax {
            group_column,
            max_column: min_column,
        } => {
            let aggregate_name = match &select.projection {
                SelectProjection::GroupedMin { .. } => "min",
                SelectProjection::GroupedMax { .. } => "max",
                _ => unreachable!(),
            };
            let Some(group_by) = &select.group_by else {
                return Err(ErrorField {
                    code: "0A000",
                    message: "grouped MIN/MAX requires GROUP BY",
                    position: None,
                });
            };
            if group_by != group_column {
                return Err(ErrorField {
                    code: "0A000",
                    message: "GROUP BY column must match grouped MIN/MAX projection",
                    position: None,
                });
            }
            let group_idx = column_index(table, group_column)?;
            let value_idx = column_index(table, min_column)?;
            let mut extrema: BTreeMap<SqlValue, SqlValue> = BTreeMap::new();
            let choose_min = matches!(select.projection, SelectProjection::GroupedMin { .. });
            for row in rows {
                extrema
                    .entry(row[group_idx].clone())
                    .and_modify(|current| {
                        let ordering = compare_sql_values(&row[value_idx], current);
                        if (choose_min && ordering.is_lt()) || (!choose_min && ordering.is_gt()) {
                            *current = row[value_idx].clone();
                        }
                    })
                    .or_insert_with(|| row[value_idx].clone());
            }
            let mut grouped = extrema.into_iter().collect::<Vec<_>>();
            grouped = grouped
                .into_iter()
                .filter_map(|(group_value, extreme)| {
                    match grouped_row_matches_having(
                        select,
                        group_column,
                        &group_value,
                        aggregate_name,
                        &extreme,
                    ) {
                        Ok(true) => Some(Ok((group_value, extreme))),
                        Ok(false) => None,
                        Err(error) => Some(Err(error)),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(order) = select.order_by.first() {
                if order.column == *group_column {
                    grouped.sort_by(|(left, _), (right, _)| compare_sql_values(left, right));
                } else if order.column.eq_ignore_ascii_case(aggregate_name) {
                    grouped.sort_by(|(left_value, left_extreme), (right_value, right_extreme)| {
                        compare_sql_values(left_extreme, right_extreme)
                            .then_with(|| compare_sql_values(left_value, right_value))
                    });
                } else {
                    return Err(ErrorField {
                        code: "0A000",
                        message: "GROUP BY ORDER BY must reference grouped column or min/max",
                        position: None,
                    });
                }
                if order.descending {
                    grouped.reverse();
                }
            }
            if let Some(offset) = select.offset {
                grouped = grouped.into_iter().skip(offset).collect();
            }
            if let Some(limit) = select.limit {
                grouped.truncate(limit);
            }
            let columns = vec![
                aggregate_result_column(table, group_idx, group_column),
                aggregate_result_column(table, value_idx, aggregate_name),
            ];
            let rows = grouped
                .into_iter()
                .map(|(value, extreme)| {
                    vec![
                        Some(format_sql_value(&value)),
                        Some(format_sql_value(&extreme)),
                    ]
                })
                .collect();
            Ok(SelectResult { columns, rows })
        }
        SelectProjection::All | SelectProjection::Columns(_) => unreachable!(),
    }
}

fn column_index(table: &Table, column: &str) -> Result<usize, ErrorField> {
    table
        .columns
        .iter()
        .position(|candidate| candidate.def.name == column)
        .ok_or(ErrorField {
            code: "42703",
            message: "column does not exist",
            position: None,
        })
}

fn aggregate_result_column(table: &Table, idx: usize, name: &str) -> Column {
    column_for_sql_type(table.columns[idx].def.ty, name)
}

fn int4_column_index(table: &Table, column: &str) -> Result<usize, ErrorField> {
    int4_column_index_for_aggregate(table, column, "SUM")
}

fn int4_column_index_for_aggregate(
    table: &Table,
    column: &str,
    aggregate: &'static str,
) -> Result<usize, ErrorField> {
    let Some(idx) = table
        .columns
        .iter()
        .position(|candidate| candidate.def.name == column)
    else {
        return Err(ErrorField {
            code: "42703",
            message: "column does not exist",
            position: None,
        });
    };
    if !matches!(table.columns[idx].def.ty, gpu_db_protocol::SqlType::Int4) {
        return Err(ErrorField {
            code: "0A000",
            message: aggregate_int4_error_message(aggregate),
            position: None,
        });
    }
    Ok(idx)
}

fn int4_value(value: &SqlValue) -> Result<i32, ErrorField> {
    int4_value_for_aggregate(value, "SUM")
}

fn int4_value_for_aggregate(value: &SqlValue, aggregate: &'static str) -> Result<i32, ErrorField> {
    match value {
        SqlValue::Int4(value) => Ok(*value),
        SqlValue::Null
        | SqlValue::Int2(_)
        | SqlValue::Int8(_)
        | SqlValue::Numeric(_)
        | SqlValue::Bool(_)
        | SqlValue::Text(_)
        | SqlValue::Date(_)
        | SqlValue::Timestamp(_)
        | SqlValue::Uuid(_)
        | SqlValue::Parameter { .. } => Err(ErrorField {
            code: "0A000",
            message: aggregate_int4_error_message(aggregate),
            position: None,
        }),
    }
}

fn aggregate_int4_error_message(aggregate: &str) -> &'static str {
    match aggregate {
        "AVG" => "AVG only supports int4 columns",
        _ => "SUM only supports int4 columns",
    }
}

fn average_text(sum: i128, count: usize) -> Option<String> {
    if count == 0 {
        return None;
    }
    let count = count as i128;
    let negative = sum.is_negative();
    let abs_sum = sum.abs();
    let whole = abs_sum / count;
    let mut remainder = abs_sum % count;
    let mut fractional = String::with_capacity(16);
    for _ in 0..16 {
        remainder *= 10;
        fractional.push(char::from(b'0' + u8::try_from(remainder / count).unwrap()));
        remainder %= count;
    }
    Some(format!(
        "{}{}.{fractional}",
        if negative { "-" } else { "" },
        whole
    ))
}

fn compare_averages(
    left_sum: i128,
    left_count: usize,
    right_sum: i128,
    right_count: usize,
) -> std::cmp::Ordering {
    (left_sum * right_count as i128).cmp(&(right_sum * left_count as i128))
}

pub(super) fn format_sql_value(value: &SqlValue) -> String {
    match value {
        // Dead arm (this server does not produce NULL); NULL reaches the wire as a `-1` field
        // length via the `Option<String>` result rows, not through this text renderer.
        SqlValue::Null => "NULL".to_string(),
        SqlValue::Int2(value) => value.to_string(),
        SqlValue::Int4(value) => value.to_string(),
        SqlValue::Int8(value) => value.to_string(),
        SqlValue::Numeric(value) => value.to_decimal_string(),
        SqlValue::Bool(value) => bool_text(*value),
        SqlValue::Text(value) => value.clone(),
        SqlValue::Date(value) => gpu_db_protocol::datetime::format_date(*value),
        SqlValue::Timestamp(value) => gpu_db_protocol::datetime::format_timestamp(*value),
        SqlValue::Uuid(value) => gpu_db_protocol::uuid::format_uuid(value),
        SqlValue::Parameter { .. } => {
            unreachable!("legacy result rendering never receives an unbound prepared parameter")
        }
    }
}

fn parse_materialized_row_value(
    value: &str,
    ty: gpu_db_protocol::SqlType,
) -> Result<SqlValue, ErrorField> {
    match ty {
        gpu_db_protocol::SqlType::Int2 => {
            value
                .parse::<i16>()
                .map(SqlValue::Int2)
                .map_err(|_| ErrorField {
                    code: "22P02",
                    message: "invalid input syntax for type smallint",
                    position: None,
                })
        }
        gpu_db_protocol::SqlType::Int4 => {
            value
                .parse::<i32>()
                .map(SqlValue::Int4)
                .map_err(|_| ErrorField {
                    code: "22P02",
                    message: "invalid input syntax for type integer",
                    position: None,
                })
        }
        gpu_db_protocol::SqlType::Int8 => {
            value
                .parse::<i64>()
                .map(SqlValue::Int8)
                .map_err(|_| ErrorField {
                    code: "22P02",
                    message: "invalid input syntax for type bigint",
                    position: None,
                })
        }
        gpu_db_protocol::SqlType::Numeric { scale, .. } => {
            gpu_db_protocol::Decimal128::parse_at_scale(value, scale)
                .map(SqlValue::Numeric)
                .ok_or(ErrorField {
                    code: "22P02",
                    message: "invalid input syntax for type numeric",
                    position: None,
                })
        }
        gpu_db_protocol::SqlType::Bool => match value.to_ascii_lowercase().as_str() {
            "t" | "true" => Ok(SqlValue::Bool(true)),
            "f" | "false" => Ok(SqlValue::Bool(false)),
            _ => Err(ErrorField {
                code: "22P02",
                message: "invalid input syntax for type boolean",
                position: None,
            }),
        },
        gpu_db_protocol::SqlType::Text => Ok(SqlValue::Text(value.to_string())),
        gpu_db_protocol::SqlType::Date => gpu_db_protocol::datetime::parse_date(value)
            .map(SqlValue::Date)
            .ok_or(ErrorField {
                code: "22P02",
                message: "invalid input syntax for type date",
                position: None,
            }),
        gpu_db_protocol::SqlType::Timestamp => gpu_db_protocol::datetime::parse_timestamp(value)
            .map(SqlValue::Timestamp)
            .ok_or(ErrorField {
                code: "22P02",
                message: "invalid input syntax for type timestamp",
                position: None,
            }),
        gpu_db_protocol::SqlType::Uuid => gpu_db_protocol::uuid::parse_uuid(value)
            .map(SqlValue::Uuid)
            .ok_or(ErrorField {
                code: "22P02",
                message: "invalid input syntax for type uuid",
                position: None,
            }),
    }
}

pub(super) fn materialize_select_rows(
    result: SelectResult,
    expected_columns: Option<&[CatalogColumn]>,
) -> Result<(Vec<CatalogColumn>, Vec<Vec<SqlValue>>), ErrorField> {
    let columns = result
        .columns
        .into_iter()
        .enumerate()
        .map(|(idx, column)| {
            Ok(CatalogColumn {
                attnum: i16::try_from(idx + 1).map_err(|_| ErrorField {
                    code: "54000",
                    message: "too many columns for bootstrap catalog",
                    position: None,
                })?,
                def: gpu_db_protocol::ColumnDef {
                    name: column.name,
                    ty: match column.oid {
                        23 => gpu_db_protocol::SqlType::Int4,
                        25 => gpu_db_protocol::SqlType::Text,
                        _ => {
                            return Err(ErrorField {
                                code: "0A000",
                                message: "materialized view column type is unsupported",
                                position: None,
                            })
                        }
                    },
                    domain: None,
                    default: None,
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(expected_columns) = expected_columns {
        if columns.len() != expected_columns.len()
            || columns
                .iter()
                .zip(expected_columns.iter())
                .any(|(left, right)| left.def.name != right.def.name || left.def.ty != right.def.ty)
        {
            return Err(ErrorField {
                code: "0A000",
                message: "materialized view refresh changed the result shape",
                position: None,
            });
        }
    }
    let column_types = columns
        .iter()
        .map(|column| column.def.ty)
        .collect::<Vec<_>>();
    let rows = result
        .rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .zip(column_types.iter().copied())
                .map(|(value, ty)| match value {
                    Some(value) => parse_materialized_row_value(&value, ty),
                    None => Err(ErrorField {
                        code: "0A000",
                        message: "NULL materialized view rows are unsupported",
                        position: None,
                    }),
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((columns, rows))
}

pub(super) fn execute_simple_select(
    stream: &mut dyn ReadWrite,
    session: &Session,
    command: Command,
    include_row_description: bool,
) -> io::Result<()> {
    match command {
        Command::Select(select) => {
            let result = match execute_select_result(session, &select) {
                Ok(result) => result,
                Err(error) => return write_error(stream, &error),
            };
            write_select_rows(
                stream,
                &result.columns,
                &result.rows,
                include_row_description,
            )
        }
        _ => unreachable!("simple-query SELECT executor received an unrelated command"),
    }
}
