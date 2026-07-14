use gpu_db_protocol::{Select, SelectFilterOp, SelectProjection, SqlValue};

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub(in super::super) enum RetainedSelectBatchCandidate {
    Literal {
        batch_key: RetainedSelectLiteralBatchKey,
        exact_key: String,
        select: Select,
        sql: String,
    },
    Exact {
        exact_key: String,
    },
}

impl RetainedSelectBatchCandidate {
    pub(in super::super) fn exact_key(&self) -> &str {
        match self {
            Self::Literal { exact_key, .. } | Self::Exact { exact_key } => exact_key,
        }
    }

    pub(in super::super) fn route_key(&self) -> String {
        match self {
            Self::Literal { batch_key, .. } => format!(
                "literal:{}:{}:{}",
                batch_key.table,
                batch_key.projection_columns.join(","),
                batch_key.filter_column
            ),
            Self::Exact { exact_key } => format!("exact:{exact_key}"),
        }
    }

    pub(in super::super) fn projected_payload_weight(&self) -> usize {
        match self {
            Self::Literal { batch_key, .. } => batch_key
                .projection_columns
                .iter()
                .map(|column| {
                    if column.ends_with("_info") || column.ends_with("_data") {
                        4
                    } else {
                        1
                    }
                })
                .sum::<usize>()
                .max(1),
            Self::Exact { .. } => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(in super::super) struct RetainedSelectLiteralBatchKey {
    pub(in super::super) table: String,
    pub(in super::super) projection_columns: Vec<String>,
    pub(in super::super) filter_column: String,
}

pub(in super::super) fn retained_select_literal_needle(select: &Select) -> Option<i32> {
    if select.filter_groups.len() > 1 {
        return None;
    }
    let filters = if !select.filter_groups.is_empty() {
        select.filter_groups[0].clone()
    } else if !select.filters.is_empty() {
        select.filters.clone()
    } else {
        let filter = select.filter.clone()?;
        vec![filter]
    };
    if filters.len() != 1 || filters[0].op != SelectFilterOp::Eq {
        return None;
    }
    let SqlValue::Int4(needle) = filters[0].value else {
        return None;
    };
    Some(needle)
}

pub(in super::super) fn retained_select_literal_batch_candidate(
    sql: &str,
    select: Select,
) -> Option<RetainedSelectBatchCandidate> {
    let exact_key = sql.to_string();
    if select.distinct
        || select.group_by.is_some()
        || !select.having_groups.is_empty()
        || !select.order_by.is_empty()
        || select.limit.is_some()
        || select.offset.is_some()
        || select.filter_groups.len() > 1
    {
        return None;
    }
    let SelectProjection::Columns(projection_columns) = &select.projection else {
        return None;
    };
    if projection_columns.is_empty() {
        return None;
    }
    let filters = if !select.filter_groups.is_empty() {
        select.filter_groups[0].clone()
    } else if !select.filters.is_empty() {
        select.filters.clone()
    } else {
        let filter = select.filter.clone()?;
        vec![filter]
    };
    if filters.len() != 1 || filters[0].op != SelectFilterOp::Eq {
        return None;
    }
    let _needle = retained_select_literal_needle(&select)?;
    Some(RetainedSelectBatchCandidate::Literal {
        batch_key: RetainedSelectLiteralBatchKey {
            table: select.table.clone(),
            projection_columns: projection_columns.clone(),
            filter_column: filters[0].column.clone(),
        },
        exact_key,
        select,
        sql: sql.to_string(),
    })
}
