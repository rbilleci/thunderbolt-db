//! Aggregate/group semantic binding for the bounded empty-catalog query subset.

use super::*;

fn catalog_column_reference_key(
    column: &pg_query::protobuf::ColumnRef,
) -> Result<Vec<String>, ExecuteError> {
    column
        .fields
        .iter()
        .map(|field| match node_enum(field)? {
            NodeEnum::String(name) => Ok(name.sval.clone()),
            NodeEnum::AStar(_) => Err(sql_pg_error(
                "catalog grouped SELECT cannot project an unbound star".to_string(),
            )),
            _ => Err(sql_pg_error(
                "catalog grouped column reference is malformed".to_string(),
            )),
        })
        .collect()
}

fn collect_current_level_unaggregated_columns(
    node: &Node,
    columns: &mut Vec<Vec<String>>,
) -> Result<(), ExecuteError> {
    match node_enum(node)? {
        NodeEnum::SubLink(_) | NodeEnum::AConst(_) | NodeEnum::CaseTestExpr(_) => {}
        NodeEnum::ColumnRef(column) => columns.push(catalog_column_reference_key(column)?),
        NodeEnum::FuncCall(function) => {
            if catalog_function_is_aggregate(function) {
                return Ok(());
            }
            for argument in &function.args {
                collect_current_level_unaggregated_columns(argument, columns)?;
            }
        }
        NodeEnum::AExpr(expression) => {
            if let Some(left) = expression.lexpr.as_deref() {
                collect_current_level_unaggregated_columns(left, columns)?;
            }
            if let Some(right) = expression.rexpr.as_deref() {
                collect_current_level_unaggregated_columns(right, columns)?;
            }
        }
        NodeEnum::BoolExpr(expression) => {
            for argument in &expression.args {
                collect_current_level_unaggregated_columns(argument, columns)?;
            }
        }
        NodeEnum::NullTest(test) => {
            if let Some(argument) = test.arg.as_deref() {
                collect_current_level_unaggregated_columns(argument, columns)?;
            }
        }
        NodeEnum::BooleanTest(test) => {
            if let Some(argument) = test.arg.as_deref() {
                collect_current_level_unaggregated_columns(argument, columns)?;
            }
        }
        NodeEnum::TypeCast(cast) => {
            if let Some(argument) = cast.arg.as_deref() {
                collect_current_level_unaggregated_columns(argument, columns)?;
            }
        }
        NodeEnum::CollateClause(collate) => {
            if let Some(argument) = collate.arg.as_deref() {
                collect_current_level_unaggregated_columns(argument, columns)?;
            }
        }
        NodeEnum::CaseExpr(case) => {
            if let Some(argument) = case.arg.as_deref() {
                collect_current_level_unaggregated_columns(argument, columns)?;
            }
            for arm in &case.args {
                collect_current_level_unaggregated_columns(arm, columns)?;
            }
            if let Some(otherwise) = case.defresult.as_deref() {
                collect_current_level_unaggregated_columns(otherwise, columns)?;
            }
        }
        NodeEnum::CaseWhen(arm) => {
            if let Some(condition) = arm.expr.as_deref() {
                collect_current_level_unaggregated_columns(condition, columns)?;
            }
            if let Some(result) = arm.result.as_deref() {
                collect_current_level_unaggregated_columns(result, columns)?;
            }
        }
        NodeEnum::AArrayExpr(array) => {
            for element in &array.elements {
                collect_current_level_unaggregated_columns(element, columns)?;
            }
        }
        NodeEnum::AIndirection(indirection) => {
            if let Some(source) = indirection.arg.as_deref() {
                collect_current_level_unaggregated_columns(source, columns)?;
            }
            for item in &indirection.indirection {
                collect_current_level_unaggregated_columns(item, columns)?;
            }
        }
        NodeEnum::AIndices(indices) => {
            if let Some(lower) = indices.lidx.as_deref() {
                collect_current_level_unaggregated_columns(lower, columns)?;
            }
            if let Some(upper) = indices.uidx.as_deref() {
                collect_current_level_unaggregated_columns(upper, columns)?;
            }
        }
        NodeEnum::List(list) => {
            for item in &list.items {
                collect_current_level_unaggregated_columns(item, columns)?;
            }
        }
        NodeEnum::SortBy(sort) => {
            if let Some(key) = sort.node.as_deref() {
                collect_current_level_unaggregated_columns(key, columns)?;
            }
        }
        NodeEnum::ResTarget(target) => {
            if let Some(value) = target.val.as_deref() {
                collect_current_level_unaggregated_columns(value, columns)?;
            }
        }
        _ => {
            return Err(sql_pg_error(
                "catalog grouped expression is outside the statically bound subset".to_string(),
            ))
        }
    }
    Ok(())
}

fn require_grouped_columns(
    node: &Node,
    group_keys: &[Vec<String>],
    context: &str,
) -> Result<(), ExecuteError> {
    let mut columns = Vec::new();
    collect_current_level_unaggregated_columns(node, &mut columns)?;
    if let Some(column) = columns
        .into_iter()
        .find(|column| !group_keys.iter().any(|group| group == column))
    {
        return Err(sql_pg_error(format!(
            "catalog {context} column {} must appear in GROUP BY or be used in an aggregate",
            column.join(".")
        )));
    }
    Ok(())
}

pub(super) fn validate_group_semantics(
    stmt: &SelectStmt,
    output: &[EmptyCatalogColumn],
) -> Result<(), ExecuteError> {
    let aggregate_query = !stmt.group_clause.is_empty()
        || stmt
            .target_list
            .iter()
            .any(node_has_current_level_aggregate)
        || stmt
            .having_clause
            .as_deref()
            .is_some_and(node_has_current_level_aggregate)
        || stmt
            .sort_clause
            .iter()
            .any(node_has_current_level_aggregate);
    if !aggregate_query {
        return Ok(());
    }

    let group_keys = stmt
        .group_clause
        .iter()
        .map(|group| match node_enum(group)? {
            NodeEnum::ColumnRef(column) => catalog_column_reference_key(column),
            _ => Err(sql_pg_error(
                "catalog GROUP BY is limited to exact column references".to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;

    for target in &stmt.target_list {
        require_grouped_columns(target, &group_keys, "SELECT target")?;
    }
    if let Some(having) = stmt.having_clause.as_deref() {
        require_grouped_columns(having, &group_keys, "HAVING")?;
    }
    for sort in &stmt.sort_clause {
        let NodeEnum::SortBy(sort) = node_enum(sort)? else {
            continue;
        };
        let Some(key) = sort.node.as_deref() else {
            continue;
        };
        let output_reference = match node_enum(key)? {
            NodeEnum::AConst(constant) => matches!(
                &constant.val,
                Some(a_const::Val::Ival(position))
                    if usize::try_from(position.ival)
                        .ok()
                        .is_some_and(|position| (1..=output.len()).contains(&position))
            ),
            NodeEnum::ColumnRef(column) => match column.fields.as_slice() {
                [field] => match node_enum(field)? {
                    NodeEnum::String(name) => {
                        output
                            .iter()
                            .filter(|column| column.name == name.sval)
                            .count()
                            == 1
                    }
                    _ => false,
                },
                _ => false,
            },
            _ => false,
        };
        if !output_reference {
            require_grouped_columns(key, &group_keys, "ORDER BY")?;
        }
    }
    Ok(())
}
