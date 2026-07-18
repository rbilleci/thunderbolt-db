//! Resident-route query-shape binding (P0 §9.6 decomposition, behavior-
//! preserving): BoundRelationalSelect (the resolved column/index/filter binding
//! a resident read executes against) and the resident-route shape matchers
//! (resident_route_query_shape, ordered/distinct/grouped/count variants, the
//! sharded variant) plus the D2H rows/bytes estimates. Pure shape analysis
//! over a parsed Select; the Engine routes on the result.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundRelationalSelect {
    pub(crate) selected_columns: Vec<RelationalColumn>,
    pub(crate) selected_indexes: Vec<usize>,
    pub(crate) group_by_index: Option<usize>,
    pub(crate) filter: Option<(usize, SelectFilterOp, SqlValue)>,
    pub(crate) filters: Vec<(usize, SelectFilterOp, SqlValue)>,
    pub(crate) filter_groups: Vec<Vec<(usize, SelectFilterOp, SqlValue)>>,
    pub(crate) order: Option<(usize, bool)>,
}

pub(crate) fn resident_route_query_shape(
    select: &Select,
    table: &RelationalTable,
    bound: &BoundRelationalSelect,
) -> Option<String> {
    if select.distinct {
        return resident_route_distinct_projection_shape(select, table, bound);
    }

    match &select.projection {
        SelectProjection::GroupedCount { column } => {
            return resident_route_grouped_aggregate_shape(select, table, bound, column, column);
        }
        SelectProjection::GroupedSum {
            group_column,
            sum_column,
        } => {
            return resident_route_grouped_aggregate_shape(
                select,
                table,
                bound,
                group_column,
                sum_column,
            );
        }
        SelectProjection::GroupedAvg {
            group_column,
            avg_column,
        } => {
            return resident_route_grouped_aggregate_shape(
                select,
                table,
                bound,
                group_column,
                avg_column,
            );
        }
        SelectProjection::GroupedMin {
            group_column,
            min_column,
        } => {
            return resident_route_grouped_aggregate_shape(
                select,
                table,
                bound,
                group_column,
                min_column,
            );
        }
        SelectProjection::GroupedMax {
            group_column,
            max_column,
        } => {
            return resident_route_grouped_aggregate_shape(
                select,
                table,
                bound,
                group_column,
                max_column,
            );
        }
        _ => {}
    }

    if !select.order_by.is_empty() {
        return resident_route_ordered_projection_shape(select, table, bound);
    }

    if select.offset.is_some() {
        return None;
    }
    if select.group_by.is_some() || !select.having_groups.is_empty() {
        return None;
    }

    match &select.projection {
        SelectProjection::CountAll => resident_route_count_shape(table, bound),
        SelectProjection::Sum { column }
        | SelectProjection::Avg { column }
        | SelectProjection::Min { column }
        | SelectProjection::Max { column } => {
            let idx = table
                .columns
                .iter()
                .position(|candidate| candidate.name == *column)?;
            if table.columns[idx].ty != SqlType::Int4 || select.limit.is_some() {
                return None;
            }
            if bound.filter.is_none() && bound.filters.is_empty() && bound.filter_groups.is_empty()
            {
                return Some("int4_scalar_aggregate".to_string());
            }
            let filter_groups = if !bound.filter_groups.is_empty() {
                bound.filter_groups.clone()
            } else if !bound.filters.is_empty() {
                vec![bound.filters.clone()]
            } else {
                vec![vec![bound.filter.clone()?]]
            };
            if filter_groups.len() == 1 && filter_groups[0].len() == 1 {
                let (filter_idx, op, value) = filter_groups[0][0].clone();
                return (filter_idx == idx
                    && resident_device_i32_comparison(op).is_some()
                    && matches!(value, SqlValue::Int4(_)))
                .then(|| "int4_filtered_scalar_aggregate".to_string());
            }
            if filter_groups.len() == 1 && filter_groups[0].len() == 2 {
                let mut lower = false;
                let mut upper = false;
                for (filter_idx, op, value) in &filter_groups[0] {
                    if *filter_idx != idx || !matches!(value, SqlValue::Int4(_)) {
                        return None;
                    }
                    match op {
                        SelectFilterOp::Gte => lower = true,
                        SelectFilterOp::Lte => upper = true,
                        _ => return None,
                    }
                }
                return (lower && upper).then(|| "int4_between_scalar_aggregate".to_string());
            }
            None
        }
        SelectProjection::Columns(columns) => {
            if columns.is_empty() || select.distinct || select.limit.is_some() {
                return None;
            }
            // R-ver (read version resolution) + TYPE-COVERAGE #14 (numeric): an UNFILTERED projection
            // (`SELECT <cols> FROM t`, no WHERE) needs an explicit resident-route shape to avoid a
            // fail-loud decline. Route it on-device iff every projected column is a DEVICE-SERVABLE
            // FIXED-WIDTH type
            // (int4/int8/date/timestamp/int2/numeric/uuid — the general executor + the recompaction
            // gather serve every fixed-width section, with the SV3b visibility conjunct threaded). A
            // text/bool column is handled by the general GPU executor when this matcher declines it.
            // Placed BEFORE the int4-only FILTERED-shape logic below.
            if bound.filter.is_none() && bound.filters.is_empty() && bound.filter_groups.is_empty()
            {
                let all_servable = bound.selected_indexes.iter().all(|&idx| {
                    matches!(
                        table.columns[idx].ty,
                        SqlType::Int4
                            | SqlType::Int8
                            | SqlType::Date
                            | SqlType::Timestamp
                            | SqlType::Int2
                            | SqlType::Numeric { .. }
                            | SqlType::Uuid
                            | SqlType::Bool
                            | SqlType::Text
                    )
                });
                return all_servable.then(|| "int4_projection_all".to_string());
            }
            // FILTERED projection shapes below are int4-only (the device locate + int4 filter VM).
            if columns.iter().any(|column| {
                table
                    .columns
                    .iter()
                    .position(|candidate| candidate.name == *column)
                    .is_none_or(|idx| {
                        !matches!(table.columns[idx].ty, SqlType::Int4 | SqlType::Text)
                    })
            }) {
                return None;
            }
            let selected_has_text = bound
                .selected_indexes
                .iter()
                .any(|idx| table.columns[*idx].ty == SqlType::Text);
            let filter_groups = if !bound.filter_groups.is_empty() {
                bound.filter_groups.clone()
            } else if !bound.filters.is_empty() {
                vec![bound.filters.clone()]
            } else {
                vec![vec![bound.filter.clone()?]]
            };
            let filter_group = filter_groups.first()?;
            if filter_groups.len() == 1
                && !filter_group.is_empty()
                && filter_group.iter().all(|(filter_idx, op, value)| {
                    *op == SelectFilterOp::Eq
                        && matches!(value, SqlValue::Int4(_))
                        && table.columns[*filter_idx].ty == SqlType::Int4
                })
            {
                if selected_has_text {
                    return Some("int4_equality_mixed_column_projection".to_string());
                }
                if filter_group.len() == 1 {
                    let (filter_idx, _op, _value) = filter_group[0].clone();
                    if columns.len() == 1 && bound.selected_indexes[0] == filter_idx {
                        return Some("int4_equality_projection".to_string());
                    }
                    return Some("int4_equality_multi_column_projection".to_string());
                }
                return Some("int4_composite_equality_multi_column_projection".to_string());
            }
            if columns.len() != 1 {
                return None;
            }
            let (filter_idx, op, value) = filter_group.first()?.clone();
            let idx = bound.selected_indexes[0];
            (idx == filter_idx
                && resident_device_i32_comparison(op).is_some()
                && matches!(value, SqlValue::Int4(_))
                && filter_groups.len() == 1
                && filter_groups[0].len() == 1)
                .then(|| "int4_projection".to_string())
        }
        SelectProjection::All => {
            // R-ver: `SELECT * FROM t` (no WHERE) over an ALL-INT4 resident table routes on-device
            // like the explicit-column unfiltered projection (order_by/group_by/offset/having are
            // excluded by the guards above; distinct at the top). Any non-int4 column or a LIMIT
            // declines this specialized matcher for the general CUDA executor (which treats All ==
            // Columns of every column) or the common fail-loud boundary.
            let unfiltered = bound.filter.is_none()
                && bound.filters.is_empty()
                && bound.filter_groups.is_empty();
            // TYPE-COVERAGE #14: `SELECT *` routes on-device iff EVERY column is a device-servable
            // fixed-width type (int4/int8/date/timestamp/int2/numeric/uuid). A text/bool column
            // declines to the general CUDA executor.
            let all_fixed_width = table.columns.iter().all(|column| {
                matches!(
                    column.ty,
                    SqlType::Int4
                        | SqlType::Int8
                        | SqlType::Date
                        | SqlType::Timestamp
                        | SqlType::Int2
                        | SqlType::Numeric { .. }
                        | SqlType::Uuid
                        | SqlType::Bool
                )
            });
            // NB: TEXT is intentionally NOT in this `SELECT *` arm. On-device text reads go through
            // the explicit-column projection arm above or the general GPU executor; the `SELECT *`
            // shape keeps its legacy specialized-route classification so route-decision tests remain stable.
            (unfiltered && all_fixed_width && select.limit.is_none())
                .then(|| "int4_projection_all".to_string())
        }
        _ => None,
    }
}

pub(crate) fn resident_route_ordered_projection_shape(
    select: &Select,
    table: &RelationalTable,
    bound: &BoundRelationalSelect,
) -> Option<String> {
    if select.group_by.is_some()
        || !select.having_groups.is_empty()
        || bound.selected_indexes.len() != 1
        || select.limit.is_none()
        || bound.filter_groups.len() != 1
        || bound.filter_groups[0].len() != 1
    {
        return None;
    }
    let projection_idx = bound.selected_indexes[0];
    if table.columns[projection_idx].ty != SqlType::Int4 {
        return None;
    }
    let (order_idx, _) = bound.order?;
    if order_idx != projection_idx {
        return None;
    }
    let (filter_idx, op, value) = bound.filter_groups[0][0].clone();
    (filter_idx == projection_idx
        && resident_device_i32_comparison(op).is_some()
        && matches!(value, SqlValue::Int4(_)))
    .then(|| "int4_ordered_projection".to_string())
}

pub(crate) fn sharded_resident_route_query_shape(
    select: &Select,
    table: &RelationalTable,
    bound: &BoundRelationalSelect,
) -> Option<String> {
    let aggregate_column = match &select.projection {
        SelectProjection::Sum { column }
        | SelectProjection::Avg { column }
        | SelectProjection::Min { column }
        | SelectProjection::Max { column } => column,
        _ => return None,
    };
    if select.distinct
        || select.group_by.is_some()
        || !select.having_groups.is_empty()
        || !select.order_by.is_empty()
        || select.limit.is_some()
        || select.offset.is_some()
    {
        return None;
    }
    let aggregate_idx = table
        .columns
        .iter()
        .position(|candidate| candidate.name == *aggregate_column)?;
    if table.columns[aggregate_idx].ty != SqlType::Int4 {
        return None;
    }
    let filter_groups = if !bound.filter_groups.is_empty() {
        bound.filter_groups.clone()
    } else if !bound.filters.is_empty() {
        vec![bound.filters.clone()]
    } else {
        vec![vec![bound.filter.clone()?]]
    };
    if filter_groups.len() != 1 {
        return None;
    }
    if matches!(&select.projection, SelectProjection::Sum { .. }) {
        if filter_groups[0].len() != 1 {
            return None;
        }
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        return (op == SelectFilterOp::Eq
            && filter_idx != aggregate_idx
            && table.columns[filter_idx].ty == SqlType::Int4
            && matches!(value, SqlValue::Int4(_)))
        .then(|| "sharded_int4_equality_sum".to_string());
    }
    if matches!(
        &select.projection,
        SelectProjection::Min { .. } | SelectProjection::Max { .. }
    ) || (matches!(&select.projection, SelectProjection::Avg { .. })
        && filter_groups[0].len() == 1)
    {
        if filter_groups[0].len() != 1 {
            return None;
        }
        let (filter_idx, op, value) = filter_groups[0][0].clone();
        return (filter_idx == aggregate_idx
            && resident_device_i32_comparison(op).is_some()
            && matches!(value, SqlValue::Int4(_)))
        .then(|| match &select.projection {
            SelectProjection::Avg { .. } => "sharded_int4_filtered_avg".to_string(),
            SelectProjection::Min { .. } => "sharded_int4_filtered_min".to_string(),
            SelectProjection::Max { .. } => "sharded_int4_filtered_max".to_string(),
            _ => unreachable!("filtered aggregate route prechecked projection"),
        });
    }
    if filter_groups[0].len() != 2 {
        return None;
    }
    let mut filter_idx = None;
    let mut lower = false;
    let mut upper = false;
    for (idx, op, value) in &filter_groups[0] {
        if filter_idx
            .replace(*idx)
            .is_some_and(|existing| existing != *idx)
            || *idx == aggregate_idx
            || table.columns[*idx].ty != SqlType::Int4
            || !matches!(value, SqlValue::Int4(_))
        {
            return None;
        }
        match op {
            SelectFilterOp::Gte => lower = true,
            SelectFilterOp::Lte => upper = true,
            _ => return None,
        }
    }
    (lower && upper).then(|| "sharded_int4_between_avg".to_string())
}

pub(crate) fn resident_route_distinct_projection_shape(
    select: &Select,
    table: &RelationalTable,
    bound: &BoundRelationalSelect,
) -> Option<String> {
    if select.group_by.is_some()
        || !select.having_groups.is_empty()
        || bound.selected_indexes.len() != 1
    {
        return None;
    }
    let SelectProjection::Columns(_columns) = &select.projection else {
        return None;
    };
    if select.offset.is_some() && (bound.order.is_none() || select.limit.is_none()) {
        return None;
    }
    let projection_idx = bound.selected_indexes[0];
    if table.columns[projection_idx].ty != SqlType::Int4 {
        return None;
    }
    if let Some((order_idx, _descending)) = bound.order {
        if order_idx != projection_idx {
            return None;
        }
    }
    if bound.filter.is_none() && bound.filters.is_empty() && bound.filter_groups.is_empty() {
        return Some("int4_distinct_projection".to_string());
    }
    let filter_groups = if !bound.filter_groups.is_empty() {
        bound.filter_groups.clone()
    } else if !bound.filters.is_empty() {
        vec![bound.filters.clone()]
    } else {
        vec![vec![bound.filter.clone()?]]
    };
    if filter_groups.len() != 1 || filter_groups[0].len() != 1 {
        return None;
    }
    let (filter_idx, op, value) = filter_groups[0][0].clone();
    (filter_idx == projection_idx
        && resident_device_i32_comparison(op).is_some()
        && matches!(value, SqlValue::Int4(_)))
    .then(|| "int4_filtered_distinct_projection".to_string())
}

pub(crate) fn resident_route_grouped_aggregate_shape(
    select: &Select,
    table: &RelationalTable,
    bound: &BoundRelationalSelect,
    group_column: &str,
    value_column: &str,
) -> Option<String> {
    if select.offset.is_some() {
        return None;
    }
    let group_by = select.group_by.as_ref()?;
    if !group_by.eq_ignore_ascii_case(group_column) {
        return None;
    }
    let group_idx = table
        .columns
        .iter()
        .position(|candidate| candidate.name == group_column)?;
    let value_idx = table
        .columns
        .iter()
        .position(|candidate| candidate.name == value_column)?;
    if table.columns[group_idx].ty != SqlType::Int4 || table.columns[value_idx].ty != SqlType::Int4
    {
        return None;
    }
    if bound.filter.is_none() && bound.filters.is_empty() && bound.filter_groups.is_empty() {
        return Some("int4_grouped_aggregate".to_string());
    }
    let filter_groups = if !bound.filter_groups.is_empty() {
        bound.filter_groups.clone()
    } else if !bound.filters.is_empty() {
        vec![bound.filters.clone()]
    } else {
        vec![vec![bound.filter.clone()?]]
    };
    if filter_groups.len() != 1 || filter_groups[0].len() != 1 {
        return None;
    }
    let (filter_idx, op, value) = filter_groups[0][0].clone();
    (table.columns.get(filter_idx)?.ty == SqlType::Int4
        && resident_device_i32_comparison(op).is_some()
        && matches!(value, SqlValue::Int4(_)))
    .then(|| "int4_filtered_grouped_aggregate".to_string())
}

pub(crate) fn resident_route_count_shape(
    table: &RelationalTable,
    bound: &BoundRelationalSelect,
) -> Option<String> {
    if bound.filter.is_none() && bound.filters.is_empty() && bound.filter_groups.is_empty() {
        return Some("count_all".to_string());
    }
    let filter_groups = if !bound.filter_groups.is_empty() {
        bound.filter_groups.clone()
    } else if !bound.filters.is_empty() {
        vec![bound.filters.clone()]
    } else {
        vec![vec![bound.filter.clone()?]]
    };
    if filter_groups.is_empty() {
        return Some("count_all".to_string());
    }
    if filter_groups.len() == 1 && filter_groups[0].len() == 1 {
        let (idx, op, value) = &filter_groups[0][0];
        let column = table.columns.get(*idx)?;
        return match (column.ty, op, value) {
            (SqlType::Int4, SelectFilterOp::Eq, SqlValue::Int4(_)) => {
                Some("int4_equality_count".to_string())
            }
            (
                SqlType::Int4,
                SelectFilterOp::Lt | SelectFilterOp::Lte | SelectFilterOp::Gt | SelectFilterOp::Gte,
                SqlValue::Int4(_),
            ) => Some("int4_range_count".to_string()),
            (SqlType::Text, SelectFilterOp::LikePrefix, SqlValue::Text(_)) => {
                Some("text_prefix_like_count".to_string())
            }
            _ => None,
        };
    }
    filter_groups
        .iter()
        .flatten()
        .all(|(idx, _op, value)| {
            table.columns.get(*idx).is_some_and(|column| {
                column.ty == SqlType::Int4 && matches!(value, SqlValue::Int4(_))
            })
        })
        .then(|| "int4_filter_group_count".to_string())
}

pub(crate) fn resident_route_d2h_rows_estimate(
    select: &Select,
    resident_row_count: usize,
) -> usize {
    match select.projection {
        SelectProjection::CountAll
        | SelectProjection::Sum { .. }
        | SelectProjection::Avg { .. }
        | SelectProjection::Min { .. }
        | SelectProjection::Max { .. } => 1,
        SelectProjection::GroupedCount { .. }
        | SelectProjection::GroupedSum { .. }
        | SelectProjection::GroupedAvg { .. }
        | SelectProjection::GroupedMin { .. }
        | SelectProjection::GroupedMax { .. } => select
            .limit
            .unwrap_or(resident_row_count)
            .min(resident_row_count),
        _ => select
            .limit
            .unwrap_or(resident_row_count)
            .min(resident_row_count),
    }
}

pub(crate) fn resident_route_d2h_bytes_estimate(
    select: &Select,
    query_shape: &str,
    snapshot: &RelationalResidencySnapshot,
) -> u64 {
    const COUNT_RESULT_BYTES: u64 = std::mem::size_of::<u64>() as u64;
    const I32_RESULT_BYTES: u64 = std::mem::size_of::<i32>() as u64;
    const I64_RESULT_BYTES: u64 = std::mem::size_of::<i64>() as u64;
    const RESULT_LEN_BYTES: u64 = std::mem::size_of::<u64>() as u64;
    const SCALAR_STATS_BYTES: u64 = (std::mem::size_of::<u64>()
        + std::mem::size_of::<i64>()
        + (2 * std::mem::size_of::<i32>())) as u64;
    const GROUPED_STATS_BYTES: u64 = (std::mem::size_of::<i32>()
        + std::mem::size_of::<u64>()
        + std::mem::size_of::<i64>()
        + (2 * std::mem::size_of::<i32>())) as u64;

    let rows = resident_route_d2h_rows_estimate(select, snapshot.row_count);
    let row_bytes = |bytes_per_row: u64| {
        u64::try_from(rows)
            .unwrap_or(u64::MAX)
            .saturating_mul(bytes_per_row)
            .saturating_add(RESULT_LEN_BYTES)
    };
    let resident_row_bytes = |bytes_per_row: u64| {
        u64::try_from(snapshot.row_count)
            .unwrap_or(u64::MAX)
            .saturating_mul(bytes_per_row)
            .saturating_add(RESULT_LEN_BYTES)
    };

    match query_shape {
        "text_prefix_like_count" => snapshot.resident_bytes,
        "int4_grouped_aggregate" | "int4_filtered_grouped_aggregate" => {
            row_bytes(GROUPED_STATS_BYTES)
        }
        "int4_projection" | "int4_ordered_projection" => row_bytes(I32_RESULT_BYTES),
        "int4_equality_projection" => COUNT_RESULT_BYTES,
        "int4_equality_multi_column_projection"
        | "int4_composite_equality_multi_column_projection"
        | "int4_equality_mixed_column_projection" => {
            let SelectProjection::Columns(columns) = &select.projection else {
                return 0;
            };
            let mut unique_columns = columns.iter().collect::<BTreeSet<_>>();
            if let Some(filter) = &select.filter {
                unique_columns.insert(&filter.column);
            }
            for filter in &select.filters {
                unique_columns.insert(&filter.column);
            }
            for filter in select.filter_groups.iter().flatten() {
                unique_columns.insert(&filter.column);
            }
            let int4_columns = unique_columns
                .iter()
                .filter(|column| snapshot.resident_device_int4_columns.contains(column))
                .count();
            let text_bytes = snapshot
                .resident_device_text_columns
                .iter()
                .filter(|layout| unique_columns.contains(&layout.name))
                .map(|layout| {
                    (u64::try_from(snapshot.row_count)
                        .unwrap_or(u64::MAX)
                        .saturating_add(1))
                    .saturating_mul(std::mem::size_of::<u64>() as u64)
                    .saturating_add(layout.bytes_len)
                })
                .fold(0_u64, u64::saturating_add);
            resident_row_bytes(
                u64::try_from(int4_columns)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(I32_RESULT_BYTES),
            )
            .saturating_add(text_bytes)
        }
        "int4_distinct_projection" | "int4_filtered_distinct_projection" => {
            resident_row_bytes(I32_RESULT_BYTES)
        }
        "int4_scalar_aggregate" if matches!(select.projection, SelectProjection::Sum { .. }) => {
            I64_RESULT_BYTES
        }
        "int4_scalar_aggregate" => GROUPED_STATS_BYTES.saturating_add(RESULT_LEN_BYTES),
        "int4_filtered_scalar_aggregate" => SCALAR_STATS_BYTES.saturating_add(RESULT_LEN_BYTES),
        "int4_between_scalar_aggregate" => SCALAR_STATS_BYTES.saturating_add(RESULT_LEN_BYTES),
        "count_all" | "int4_equality_count" | "int4_range_count" | "int4_filter_group_count" => {
            COUNT_RESULT_BYTES
        }
        _ => 0,
    }
}
