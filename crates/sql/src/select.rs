//! SELECT contracts and projection, filter, and ordering parsing.

use super::{
    find_keyword_outside_quotes, find_matching_paren, next_clause_pos, normalize_identifier,
    normalize_relation_identifier, normalize_select_relation_identifier, parse_sql_value,
    split_csv, split_keyword_chain_outside_quotes, split_select_and_chain_outside_quotes,
    strip_keyword_prefix_case_insensitive, ParseError, SqlType, SqlValue,
};

/// Internal marker for an in-place bare `*` inside an otherwise explicit projection list.
///
/// PostgreSQL identifiers cannot contain NUL, so this value can never collide with a quoted
/// identifier such as `"*"`. Consumers compare against this marker when expanding a mixed
/// projection such as `SELECT oid, *`.
pub const PROJECTION_WILDCARD_SENTINEL: &str = "\0gpu_db_projection_wildcard";

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Select {
    pub table: String,
    /// An explicitly qualified `public.table` lookup must resolve only a user relation. Bare names
    /// retain PostgreSQL's user-first catalog search behavior, while system-schema names remain
    /// qualified in `table`. This provenance is part of the bound AST and survives prepared
    /// Parse/Describe/Execute without consulting or reparsing the original SQL text.
    #[serde(default, skip_serializing_if = "is_false")]
    pub public_only: bool,
    pub distinct: bool,
    pub projection: SelectProjection,
    pub group_by: Option<String>,
    pub having_groups: Vec<Vec<SelectFilter>>,
    pub filter: Option<SelectFilter>,
    pub filters: Vec<SelectFilter>,
    pub filter_groups: Vec<Vec<SelectFilter>>,
    /// ORDER BY keys, in significance order (key 0 is primary). Empty = no ORDER BY. Multi-key sorts
    /// run ONLY on the general GPU bitonic-sort path; the CPU/enumerated sort sites honor key 0 and
    /// reject `len() > 1`.
    pub order_by: Vec<SelectOrder>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

impl Select {
    /// Returns the one-based parameter index encoded in an unbound prepared `LIMIT` clause.
    ///
    /// The ordinary execution AST deliberately retains `Option<usize>` so every GPU route sees a
    /// fully bound window. A value in this reserved range exists only between prepared Parse and
    /// Bind; ordinary command parsing rejects unbound parameters before it can enter execution.
    pub fn prepared_limit_parameter_index(&self) -> Option<usize> {
        let limit = self.limit?;
        (limit > i32::MAX as usize)
            .then(|| usize::MAX.checked_sub(limit)?.checked_add(1))
            .flatten()
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum SelectProjection {
    All,
    /// Explicit columns with an in-place bare `*` expansion. The `Columns` representation keeps
    /// projection order and duplicates (for example `oid, *`) while binding expands `*` against
    /// the exact retained table definition.
    Columns(Vec<String>),
    CountAll,
    GroupedCount {
        column: String,
    },
    Sum {
        column: String,
    },
    GroupedSum {
        group_column: String,
        sum_column: String,
    },
    Avg {
        column: String,
    },
    GroupedAvg {
        group_column: String,
        avg_column: String,
    },
    Min {
        column: String,
    },
    GroupedMin {
        group_column: String,
        min_column: String,
    },
    Max {
        column: String,
    },
    GroupedMax {
        group_column: String,
        max_column: String,
    },
    /// `COUNT(DISTINCT v)` -- a distinct count over one column. Only the GROUPED form (folded into
    /// [`SelectProjection::GroupedAggregates`]) runs on the GPU path; the bare/scalar form is a clean
    /// follow-up (rejected at execution).
    CountDistinct {
        column: String,
    },
    /// A grouped SELECT projecting N aggregates over one GROUP BY key (the general grouped form on the
    /// Expr path). Each aggregate carries its own function + value column (None for COUNT(*)). The
    /// single Grouped{Count,Sum,Avg,Min,Max} variants above remain the legacy 1-aggregate shapes used
    /// by the hand-rolled enumerated GPU route.
    GroupedAggregates {
        group_column: String,
        aggregates: Vec<GroupedAggregate>,
    },
}

/// One aggregate within a `GroupedAggregates` projection.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct GroupedAggregate {
    pub kind: GroupedAggKind,
    /// The aggregated value column, or `None` for COUNT(*).
    pub value_column: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupedAggKind {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    /// `COUNT(DISTINCT v)` -- the per-group count of distinct `value_column` values. Computed on the
    /// GPU via a sort-of-`(group, v)` + mark-new-distinct + SUM pass (no CPU), not the direct hash.
    CountDistinct,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SelectFilter {
    pub column: String,
    pub op: SelectFilterOp,
    pub value: SqlValue,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectFilterOp {
    Eq,
    Lt,
    Lte,
    Gt,
    Gte,
    LikePrefix,
}

impl SelectFilterOp {
    pub(super) fn flipped(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Lt => Self::Gt,
            Self::Lte => Self::Gte,
            Self::Gt => Self::Lt,
            Self::Gte => Self::Lte,
            Self::LikePrefix => Self::LikePrefix,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SelectOrder {
    pub column: String,
    pub descending: bool,
}

pub(super) fn parse_select(input: &str, allow_catalog_schemas: bool) -> Result<Select, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "SELECT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let from_pos =
        find_keyword_outside_quotes(rest, "FROM").ok_or(ParseError::InvalidRelationalSql)?;
    let mut projection_input = rest[..from_pos].trim();
    let distinct = if let Some(after_distinct) =
        strip_keyword_prefix_case_insensitive(projection_input, "DISTINCT")
    {
        projection_input = after_distinct.trim_start();
        true
    } else {
        false
    };
    let projection = parse_projection(projection_input)?;
    if distinct
        && matches!(
            projection,
            SelectProjection::All
                | SelectProjection::CountAll
                | SelectProjection::GroupedCount { .. }
                | SelectProjection::Sum { .. }
                | SelectProjection::GroupedSum { .. }
                | SelectProjection::Avg { .. }
                | SelectProjection::GroupedAvg { .. }
                | SelectProjection::Min { .. }
                | SelectProjection::GroupedMin { .. }
                | SelectProjection::Max { .. }
                | SelectProjection::GroupedMax { .. }
        )
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let mut tail = rest[from_pos + "FROM".len()..].trim_start();
    if let Some(after_only) = strip_keyword_prefix_case_insensitive(tail, "ONLY") {
        tail = after_only.trim_start();
    }
    let table_end = tail.find(char::is_whitespace).unwrap_or(tail.len());
    // Only the engine's catalog-aware entry carries a `pg_catalog.`/`information_schema.`
    // qualifier through; the strict path keeps rejecting non-public schemas (so the legacy
    // server's compatibility layer still handles catalog queries unchanged).
    let relation = &tail[..table_end];
    let (table, public_only) = if allow_catalog_schemas {
        normalize_select_relation_identifier(relation)?
    } else {
        (
            normalize_relation_identifier(relation)?,
            relation.split_once('.').is_some(),
        )
    };
    tail = tail[table_end..].trim_start();

    let mut filter_groups = Vec::new();
    let mut group_by = None;
    let mut having_groups = Vec::new();
    let mut order_by = Vec::new();
    let mut limit = None;
    let mut offset = None;
    while !tail.is_empty() {
        if let Some(after_where) = strip_keyword_prefix_case_insensitive(tail, "WHERE") {
            let after_where = after_where.trim_start();
            let next = next_clause_pos(after_where).unwrap_or(after_where.len());
            filter_groups = parse_select_filter_groups(after_where[..next].trim())?;
            tail = after_where[next..].trim_start();
        } else if let Some(after_group) = strip_keyword_prefix_case_insensitive(tail, "GROUP") {
            let after_by = strip_keyword_prefix_case_insensitive(after_group.trim_start(), "BY")
                .ok_or(ParseError::InvalidRelationalSql)?
                .trim_start();
            let next = next_clause_pos(after_by).unwrap_or(after_by.len());
            group_by = Some(normalize_identifier(after_by[..next].trim())?);
            tail = after_by[next..].trim_start();
        } else if let Some(after_having) = strip_keyword_prefix_case_insensitive(tail, "HAVING") {
            let after_having = after_having.trim_start();
            let next = next_clause_pos(after_having).unwrap_or(after_having.len());
            having_groups = parse_select_filter_groups(after_having[..next].trim())?;
            tail = after_having[next..].trim_start();
        } else if let Some(after_order) = strip_keyword_prefix_case_insensitive(tail, "ORDER") {
            let after_by = strip_keyword_prefix_case_insensitive(after_order.trim_start(), "BY")
                .ok_or(ParseError::InvalidRelationalSql)?
                .trim_start();
            let next = next_clause_pos(after_by).unwrap_or(after_by.len());
            order_by = parse_select_order(after_by[..next].trim())?;
            tail = after_by[next..].trim_start();
        } else if let Some(after_limit) = strip_keyword_prefix_case_insensitive(tail, "LIMIT") {
            let after_limit = after_limit.trim_start();
            let next = next_clause_pos(after_limit).unwrap_or(after_limit.len());
            limit = Some(parse_select_limit(after_limit[..next].trim())?);
            tail = after_limit[next..].trim_start();
        } else if let Some(after_offset) = strip_keyword_prefix_case_insensitive(tail, "OFFSET") {
            let after_offset = after_offset.trim_start();
            let next = next_clause_pos(after_offset).unwrap_or(after_offset.len());
            offset = Some(parse_select_offset(after_offset[..next].trim())?);
            tail = after_offset[next..].trim_start();
        } else {
            return Err(ParseError::InvalidRelationalSql);
        }
    }

    let filters = filter_groups.first().cloned().unwrap_or_default();
    Ok(Select {
        table,
        public_only,
        distinct,
        projection,
        group_by,
        having_groups,
        filter: filters.first().cloned(),
        filters,
        filter_groups,
        order_by,
        limit,
        offset,
    })
}

fn parse_projection(input: &str) -> Result<SelectProjection, ParseError> {
    if input == "*" {
        return Ok(SelectProjection::All);
    }
    if input.eq_ignore_ascii_case("COUNT(*)") {
        return Ok(SelectProjection::CountAll);
    }
    if let Some(column) = parse_aggregate_call(input, "SUM")? {
        return Ok(SelectProjection::Sum { column });
    }
    if let Some(column) = parse_aggregate_call(input, "AVG")? {
        return Ok(SelectProjection::Avg { column });
    }
    if let Some(column) = parse_aggregate_call(input, "MIN")? {
        return Ok(SelectProjection::Min { column });
    }
    if let Some(column) = parse_aggregate_call(input, "MAX")? {
        return Ok(SelectProjection::Max { column });
    }
    let items = split_csv(input)?;
    if items.len() == 2 && items[1].trim().eq_ignore_ascii_case("COUNT(*)") {
        return Ok(SelectProjection::GroupedCount {
            column: normalize_identifier(items[0].trim())?,
        });
    }
    if items.len() == 2 {
        if let Some(sum_column) = parse_aggregate_call(items[1].trim(), "SUM")? {
            return Ok(SelectProjection::GroupedSum {
                group_column: normalize_identifier(items[0].trim())?,
                sum_column,
            });
        }
        if let Some(avg_column) = parse_aggregate_call(items[1].trim(), "AVG")? {
            return Ok(SelectProjection::GroupedAvg {
                group_column: normalize_identifier(items[0].trim())?,
                avg_column,
            });
        }
        if let Some(min_column) = parse_aggregate_call(items[1].trim(), "MIN")? {
            return Ok(SelectProjection::GroupedMin {
                group_column: normalize_identifier(items[0].trim())?,
                min_column,
            });
        }
        if let Some(max_column) = parse_aggregate_call(items[1].trim(), "MAX")? {
            return Ok(SelectProjection::GroupedMax {
                group_column: normalize_identifier(items[0].trim())?,
                max_column,
            });
        }
    }
    if items.iter().any(|item| {
        item.trim().eq_ignore_ascii_case("COUNT(*)") || aggregate_call_name(item.trim()).is_some()
    }) {
        return Err(ParseError::InvalidRelationalSql);
    }
    let columns = items
        .into_iter()
        .map(|column| {
            let column = column.trim();
            if column == "*" {
                Ok(PROJECTION_WILDCARD_SENTINEL.to_string())
            } else {
                normalize_identifier(column)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if columns.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(SelectProjection::Columns(columns))
}

fn parse_aggregate_call(input: &str, expected: &str) -> Result<Option<String>, ParseError> {
    let Some(name) = aggregate_call_name(input) else {
        return Ok(None);
    };
    if !name.eq_ignore_ascii_case(expected) {
        return Ok(None);
    }
    let open = input.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = input.rfind(')').ok_or(ParseError::InvalidRelationalSql)?;
    if close + 1 != input.len() || close <= open + 1 {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(Some(normalize_identifier(input[open + 1..close].trim())?))
}

fn aggregate_call_name(input: &str) -> Option<&str> {
    let open = input.find('(')?;
    if !input.ends_with(')') {
        return None;
    }
    let name = input[..open].trim();
    let mut chars = name.chars();
    let first = chars.next()?;
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return None;
    }
    if chars.any(|ch| !(ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())) {
        return None;
    }
    (!name.is_empty()).then_some(name)
}

fn parse_select_limit(input: &str) -> Result<usize, ParseError> {
    match parse_sql_value(input)? {
        SqlValue::Int4(value) if value >= 0 => Ok(value as usize),
        SqlValue::Int4(_) => Err(ParseError::NegativeLimit),
        SqlValue::Parameter { index, cast } if cast.is_none() || cast == Some(SqlType::Int4) => {
            // SQL literals are bounded to int4, so this high range cannot collide with a parsed
            // literal. PreparedCommand replaces it with the decoded int4 before the AST reaches
            // any engine route.
            usize::MAX
                .checked_sub(index.saturating_sub(1))
                .filter(|encoded| *encoded > i32::MAX as usize)
                .ok_or(ParseError::InvalidParameterReference)
        }
        SqlValue::Parameter { .. } => Err(ParseError::InvalidRelationalSql),
        SqlValue::Null
        | SqlValue::Int8(_)
        | SqlValue::Int2(_)
        | SqlValue::Numeric(_)
        | SqlValue::Bool(_)
        | SqlValue::Text(_)
        | SqlValue::Date(_)
        | SqlValue::Timestamp(_)
        | SqlValue::Uuid(_) => Err(ParseError::InvalidRelationalSql),
    }
}

fn parse_select_offset(input: &str) -> Result<usize, ParseError> {
    match parse_sql_value(input)? {
        SqlValue::Int4(value) if value >= 0 => Ok(value as usize),
        SqlValue::Int4(_) => Err(ParseError::NegativeOffset),
        SqlValue::Null
        | SqlValue::Int8(_)
        | SqlValue::Int2(_)
        | SqlValue::Numeric(_)
        | SqlValue::Bool(_)
        | SqlValue::Text(_)
        | SqlValue::Date(_)
        | SqlValue::Timestamp(_)
        | SqlValue::Uuid(_)
        | SqlValue::Parameter { .. } => Err(ParseError::InvalidRelationalSql),
    }
}

pub(super) fn parse_select_filter(input: &str) -> Result<SelectFilter, ParseError> {
    let input = trim_wrapping_parentheses(input)?;
    let (left, op, right) = split_select_filter(input)?;
    let left = left.trim();
    let right = right.trim();
    if let Ok(value) = parse_sql_value(right) {
        return Ok(SelectFilter {
            column: normalize_identifier(left)?,
            op,
            value,
        });
    }
    if let Ok(value) = parse_sql_value(left) {
        return Ok(SelectFilter {
            column: normalize_identifier(right)?,
            op: op.flipped(),
            value,
        });
    }
    Err(ParseError::InvalidRelationalSql)
}

pub(super) fn parse_select_filter_groups(
    input: &str,
) -> Result<Vec<Vec<SelectFilter>>, ParseError> {
    let groups = parse_select_filter_or_groups(input)?;
    if groups.is_empty() || groups.iter().any(Vec::is_empty) {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(groups)
}

fn parse_select_filter_or_groups(input: &str) -> Result<Vec<Vec<SelectFilter>>, ParseError> {
    let input = trim_wrapping_parentheses(input)?;
    let parts = split_keyword_chain_outside_quotes(input, "OR")?;
    if parts.len() == 1 {
        return parse_select_filter_and_groups(input);
    }
    let groups = parts
        .into_iter()
        .map(parse_select_filter_and_groups)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    Ok(groups)
}

fn parse_select_filter_and_groups(input: &str) -> Result<Vec<Vec<SelectFilter>>, ParseError> {
    let parts = split_select_and_chain_outside_quotes(input)?;
    if parts.len() == 1 {
        return parse_select_filter_factor(input);
    }
    let mut groups = vec![Vec::new()];
    for part in parts {
        let factor_groups = parse_select_filter_factor(part)?;
        let mut combined = Vec::new();
        for existing in &groups {
            for factor_group in &factor_groups {
                let mut group = existing.clone();
                group.extend(factor_group.iter().cloned());
                combined.push(group);
            }
        }
        groups = combined;
    }
    Ok(groups)
}

fn parse_select_filter_factor(input: &str) -> Result<Vec<Vec<SelectFilter>>, ParseError> {
    let input = input.trim();
    if input.starts_with('(') {
        let close = find_matching_paren(input, 0).ok_or(ParseError::InvalidRelationalSql)?;
        if close == input.len() - 1 {
            return parse_select_filter_or_groups(&input[1..close]);
        }
    }
    if let Some(group) = parse_select_between_filter_group(input)? {
        return Ok(vec![group]);
    }
    if let Some(groups) = parse_select_in_filter_groups(input)? {
        return Ok(groups);
    }
    if let Some(filter) = parse_select_like_prefix_filter(input)? {
        return Ok(vec![vec![filter]]);
    }
    Ok(vec![vec![parse_select_filter(input)?]])
}

fn parse_select_between_filter_group(input: &str) -> Result<Option<Vec<SelectFilter>>, ParseError> {
    let Some(pos) = find_keyword_outside_quotes(input, "BETWEEN") else {
        return Ok(None);
    };
    let column = normalize_identifier(input[..pos].trim())?;
    let bounds = input[pos + "BETWEEN".len()..].trim();
    let Some(and_pos) = find_keyword_outside_quotes(bounds, "AND") else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let lower = bounds[..and_pos].trim();
    let upper = bounds[and_pos + "AND".len()..].trim();
    if lower.is_empty() || upper.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(Some(vec![
        SelectFilter {
            column: column.clone(),
            op: SelectFilterOp::Gte,
            value: parse_sql_value(lower)?,
        },
        SelectFilter {
            column,
            op: SelectFilterOp::Lte,
            value: parse_sql_value(upper)?,
        },
    ]))
}

fn parse_select_in_filter_groups(
    input: &str,
) -> Result<Option<Vec<Vec<SelectFilter>>>, ParseError> {
    let Some(pos) = find_keyword_outside_quotes(input, "IN") else {
        return Ok(None);
    };
    let column = normalize_identifier(input[..pos].trim())?;
    let values = input[pos + "IN".len()..].trim();
    if !values.starts_with('(') {
        return Err(ParseError::InvalidRelationalSql);
    }
    let close = find_matching_paren(values, 0).ok_or(ParseError::InvalidRelationalSql)?;
    if close != values.len() - 1 {
        return Err(ParseError::InvalidRelationalSql);
    }
    let values = split_csv(&values[1..close])?;
    let groups = values
        .into_iter()
        .map(|value| {
            Ok(vec![SelectFilter {
                column: column.clone(),
                op: SelectFilterOp::Eq,
                value: parse_sql_value(value)?,
            }])
        })
        .collect::<Result<Vec<_>, ParseError>>()?;
    Ok(Some(groups))
}

fn parse_select_like_prefix_filter(input: &str) -> Result<Option<SelectFilter>, ParseError> {
    let Some(pos) = find_keyword_outside_quotes(input, "LIKE") else {
        return Ok(None);
    };
    let column = normalize_identifier(input[..pos].trim())?;
    let pattern = parse_sql_value(input[pos + "LIKE".len()..].trim())?;
    let SqlValue::Text(pattern) = pattern else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let Some(prefix) = pattern.strip_suffix('%') else {
        return Err(ParseError::InvalidRelationalSql);
    };
    if prefix.contains('%') || prefix.contains('_') {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(Some(SelectFilter {
        column,
        op: SelectFilterOp::LikePrefix,
        value: SqlValue::Text(prefix.to_string()),
    }))
}

pub(super) fn split_select_filter(input: &str) -> Result<(&str, SelectFilterOp, &str), ParseError> {
    for (token, op) in [
        ("<=", SelectFilterOp::Lte),
        (">=", SelectFilterOp::Gte),
        ("=", SelectFilterOp::Eq),
        ("<", SelectFilterOp::Lt),
        (">", SelectFilterOp::Gt),
    ] {
        if let Some((column, value)) = input.split_once(token) {
            if column.trim().is_empty() || value.trim().is_empty() {
                return Err(ParseError::InvalidRelationalSql);
            }
            return Ok((column, op, value));
        }
    }
    Err(ParseError::InvalidRelationalSql)
}

fn trim_wrapping_parentheses(input: &str) -> Result<&str, ParseError> {
    let mut trimmed = input.trim();
    loop {
        if !trimmed.starts_with('(') {
            return Ok(trimmed);
        }
        let close = find_matching_paren(trimmed, 0).ok_or(ParseError::InvalidRelationalSql)?;
        if close != trimmed.len() - 1 {
            return Ok(trimmed);
        }
        trimmed = trimmed[1..close].trim();
        if trimmed.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
    }
}

fn parse_select_order(input: &str) -> Result<Vec<SelectOrder>, ParseError> {
    let mut keys = Vec::new();
    for part in input.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        let mut parts = part.split_whitespace();
        let column = parts
            .next()
            .ok_or(ParseError::InvalidRelationalSql)
            .and_then(normalize_identifier)?;
        let descending = match parts.next() {
            None => false,
            Some(direction) if direction.eq_ignore_ascii_case("ASC") => false,
            Some(direction) if direction.eq_ignore_ascii_case("DESC") => true,
            _ => return Err(ParseError::InvalidRelationalSql),
        };
        if parts.next().is_some() {
            return Err(ParseError::InvalidRelationalSql);
        }
        keys.push(SelectOrder { column, descending });
    }
    if keys.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(keys)
}
