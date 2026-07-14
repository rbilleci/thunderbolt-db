//! State-free normalization from legacy/bound select forms into the resident expression IR.

use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};
use crate::resident_route::BoundRelationalSelect;
use crate::ExecuteError;
use gpu_db_sql::{Decimal128, GroupedAggKind, GroupedAggregate, SelectProjection, SqlValue};
use gpu_db_types::EngineError;

/// Map a HAVING/result-filter comparison operator to the resident-predicate IR operator.
pub(super) fn having_op_to_resident(op: crate::SelectFilterOp) -> ResidentBinaryOp {
    use crate::SelectFilterOp;
    match op {
        SelectFilterOp::Eq => ResidentBinaryOp::Eq,
        SelectFilterOp::Lt => ResidentBinaryOp::Lt,
        SelectFilterOp::Lte => ResidentBinaryOp::Le,
        SelectFilterOp::Gt => ResidentBinaryOp::Gt,
        SelectFilterOp::Gte => ResidentBinaryOp::Ge,
        SelectFilterOp::LikePrefix => ResidentBinaryOp::Like,
    }
}

/// Normalize a legacy 1-aggregate grouped projection (`GroupedCount` / `GroupedSum` / `GroupedAvg` /
/// `GroupedMin` / `GroupedMax`, produced by the hand-rolled parser) to the general
/// [`SelectProjection::GroupedAggregates`] form the on-device Expr executor consumes -- the S8
/// `&Select`->general bridge. `GroupedCount { column }` groups by `column` with a COUNT(*) (no value
/// column); the others carry `(group_column, value_column)`. Returns `None` for a projection that is
/// not a legacy grouped form (already `GroupedAggregates`, or not grouped) -- the caller leaves it as-is.
pub(crate) fn grouped_projection_to_aggregates(
    projection: &SelectProjection,
) -> Option<SelectProjection> {
    let (group_column, aggregate) = match projection {
        SelectProjection::GroupedCount { column } => (
            column.clone(),
            GroupedAggregate {
                kind: GroupedAggKind::Count,
                value_column: None,
            },
        ),
        SelectProjection::GroupedSum {
            group_column,
            sum_column,
        } => (
            group_column.clone(),
            GroupedAggregate {
                kind: GroupedAggKind::Sum,
                value_column: Some(sum_column.clone()),
            },
        ),
        SelectProjection::GroupedAvg {
            group_column,
            avg_column,
        } => (
            group_column.clone(),
            GroupedAggregate {
                kind: GroupedAggKind::Avg,
                value_column: Some(avg_column.clone()),
            },
        ),
        SelectProjection::GroupedMin {
            group_column,
            min_column,
        } => (
            group_column.clone(),
            GroupedAggregate {
                kind: GroupedAggKind::Min,
                value_column: Some(min_column.clone()),
            },
        ),
        SelectProjection::GroupedMax {
            group_column,
            max_column,
        } => (
            group_column.clone(),
            GroupedAggregate {
                kind: GroupedAggKind::Max,
                value_column: Some(max_column.clone()),
            },
        ),
        _ => return None,
    };
    Some(SelectProjection::GroupedAggregates {
        group_column,
        aggregates: vec![aggregate],
    })
}

/// Reconstruct a faithful SQL `LIKE` pattern from the BARE literal prefix a `LikePrefix` bound filter
/// carries. The parser produces `LikePrefix` only for `LIKE '<prefix>%'` where `<prefix>` has no `%`/`_`
/// (sql/lib.rs `parse_select_like_prefix_filter`), and strips the trailing `%`, so the bound value is the
/// literal prefix bytes (a `\` is literal, not a LIKE escape, since `parse_sql_value` does not resolve LIKE
/// escapes). To match the legacy probe's literal-byte prefix match EXACTLY, escape every LIKE-special byte
/// in the prefix (`%`, `_`, `\`) and append a `%`: the resulting `LIKE '<escaped>%'` means "starts with the
/// literal prefix" — `compile_like_pattern` then yields the same byte-prefix semantics the probe had.
pub(crate) fn like_pattern_for_literal_prefix(prefix: &str) -> String {
    let mut pattern = Vec::with_capacity(prefix.len() + 1);
    for &byte in prefix.as_bytes() {
        if matches!(byte, b'%' | b'_' | b'\\') {
            pattern.push(b'\\');
        }
        pattern.push(byte);
    }
    pattern.push(b'%');
    // The prefix is valid UTF-8 and only ASCII escape bytes are inserted, so the result stays valid UTF-8.
    String::from_utf8(pattern).expect("ascii-escaped utf8 stays valid utf8")
}

/// Build the WHERE-predicate [`ResidentExpr`] DNF for the `&Select`->general grouped BRIDGE (S8) from a
/// bound select's resolved filters. Normalizes to the canonical DNF (OR of AND-groups) exactly as the
/// route classifiers do -- `filter_groups`, else `filters` as one AND-group, else the single `filter`,
/// else no predicate -- then turns each `(column_idx, op, value)` leaf into `Column(idx) <op> literal`
/// (AND within a group, OR across groups), mirroring the S3 HAVING DNF construction. This bound-select
/// normalization path produces the SAME `ResidentExpr` the SQL->Expr path's `map_predicate_node` builds
/// for `WHERE col <op> 5` (an int4 `Column <cmp> Int4Literal`), so routing a grouped probe shape through
/// it is behavior-preserving. `None` = no WHERE (a full-table scan). The grouped route classifier supplies
/// only int4 `Column <cmp> Int4Literal` leaves, but shared callers also supply DATE and `LikePrefix` leaves,
/// which are normalized explicitly below. The general DNF remains robust to any int4 OR-of-AND filter the
/// bound may carry; a `LikePrefix` becomes an on-device `LIKE` (see [`like_pattern_for_literal_prefix`]).
pub(crate) fn resident_predicate_from_bound_filters(
    bound: &BoundRelationalSelect,
) -> Result<Option<ResidentExpr>, ExecuteError> {
    let groups: Vec<Vec<(usize, crate::SelectFilterOp, SqlValue)>> =
        if !bound.filter_groups.is_empty() {
            bound.filter_groups.clone()
        } else if !bound.filters.is_empty() {
            vec![bound.filters.clone()]
        } else if let Some(filter) = bound.filter.clone() {
            vec![vec![filter]]
        } else {
            return Ok(None);
        };
    let mut dnf: Option<ResidentExpr> = None;
    for group in &groups {
        let mut conj: Option<ResidentExpr> = None;
        for (idx, op, value) in group {
            // A `LikePrefix` filter carries the BARE literal prefix; reconstruct the faithful `LIKE
            // '<prefix>%'` pattern (escaped) so the bridge runs the SAME text-prefix match the retired probe
            // did — on-device via the general LIKE mask. Other ops build a `Column <cmp> literal` leaf.
            let leaf = if matches!(op, crate::SelectFilterOp::LikePrefix) {
                let SqlValue::Text(prefix) = value else {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "a LIKE-prefix filter requires a text prefix value".to_string(),
                    )));
                };
                ResidentExpr::Binary {
                    op: ResidentBinaryOp::Like,
                    lhs: Box::new(ResidentExpr::Column(*idx)),
                    rhs: Box::new(ResidentExpr::TextLiteral(like_pattern_for_literal_prefix(
                        prefix,
                    ))),
                }
            } else if let SqlValue::Date(days) = value {
                // A BOUND date filter (ADR-006 date compound): emit the CANONICAL date string, which
                // the date peephole / DATE VM leaf parse back to the identical days (`format_date` /
                // `parse_date` round-trip — the uuid pattern). NOT a raw-days `Int4Literal`: that node
                // is what a BARE INTEGER lowers to, and `date = 5` must stay a hard error (PG
                // semantics), so the date lowerers reject Int4Literal by design.
                ResidentExpr::Binary {
                    op: having_op_to_resident(*op),
                    lhs: Box::new(ResidentExpr::Column(*idx)),
                    rhs: Box::new(ResidentExpr::TextLiteral(
                        gpu_db_sql::datetime::format_date(*days),
                    )),
                }
            } else {
                ResidentExpr::Binary {
                    op: having_op_to_resident(*op),
                    lhs: Box::new(ResidentExpr::Column(*idx)),
                    // numeric_mode = false: a grouped-route filter constant is an int4 literal (the
                    // classifier guarantees `SqlValue::Int4`), matching `map_predicate_node`'s Int4Literal.
                    rhs: Box::new(having_value_to_resident_literal(value, false)?),
                }
            };
            conj = Some(match conj {
                None => leaf,
                Some(prev) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(prev),
                    rhs: Box::new(leaf),
                },
            });
        }
        if let Some(c) = conj {
            dnf = Some(match dnf {
                None => c,
                Some(prev) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::Or,
                    lhs: Box::new(prev),
                    rhs: Box::new(c),
                },
            });
        }
    }
    Ok(dnf)
}

/// A HAVING comparison constant (`SqlValue`) as a resident-predicate literal node, so the HAVING DNF can
/// be evaluated by the SAME device predicate VM as WHERE. The HAVING transient promotes every integer
/// column to ONE width (the VM is single-width); `numeric_mode` says which: in NUMERIC mode (the DNF
/// touches a numeric column) integer constants compare as `Numeric(scale 0)`; otherwise (all integer) they
/// ride the i32 literal (the int8 compare path sign-extends a small one). A genuine `Numeric` constant is
/// a numeric literal regardless. (A Timestamp/Uuid/Text/Bool HAVING constant is parser-rejected upstream;
/// the arm exists for safety, not reach.)
pub(super) fn having_value_to_resident_literal(
    value: &SqlValue,
    numeric_mode: bool,
) -> Result<ResidentExpr, ExecuteError> {
    if numeric_mode {
        return Ok(match value {
            SqlValue::Int4(v) | SqlValue::Date(v) => {
                ResidentExpr::NumericLiteral(Decimal128::new(i128::from(*v), 0))
            }
            SqlValue::Int2(v) => ResidentExpr::NumericLiteral(Decimal128::new(i128::from(*v), 0)),
            SqlValue::Int8(v) => ResidentExpr::NumericLiteral(Decimal128::new(i128::from(*v), 0)),
            SqlValue::Numeric(d) => ResidentExpr::NumericLiteral(*d),
            other => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "HAVING comparison against a {other:?} constant is not yet on the GPU predicate path"
                ))));
            }
        });
    }
    Ok(match value {
        SqlValue::Int4(v) | SqlValue::Date(v) => ResidentExpr::Int4Literal(*v),
        SqlValue::Int2(v) => ResidentExpr::Int4Literal(i32::from(*v)),
        SqlValue::Int8(v) => match i32::try_from(*v) {
            Ok(small) => ResidentExpr::Int4Literal(small),
            Err(_) => ResidentExpr::NumericLiteral(Decimal128::new(i128::from(*v), 0)),
        },
        SqlValue::Numeric(d) => ResidentExpr::NumericLiteral(*d),
        SqlValue::Text(s) => ResidentExpr::TextLiteral(s.clone()),
        SqlValue::Bool(b) => ResidentExpr::BoolLiteral(*b),
        other => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "HAVING comparison against a {other:?} constant is not yet on the GPU predicate path"
            ))));
        }
    })
}
