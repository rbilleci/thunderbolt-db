//! SELECT-clause and output binding for the bounded empty-catalog query subset.

use super::*;

pub(super) fn array_sublink_element_type(
    link: &pg_query::protobuf::SubLink,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
) -> Result<SqlType, ExecuteError> {
    if link.testexpr.is_some() || !link.oper_name.is_empty() {
        return Err(sql_pg_error(
            "catalog ARRAY subquery has unexpected comparison state".to_string(),
        ));
    }
    let NodeEnum::SelectStmt(select) = node_enum(
        link.subselect
            .as_deref()
            .ok_or_else(|| sql_pg_error("catalog ARRAY subquery is missing".to_string()))?,
    )?
    else {
        return Err(sql_pg_error(
            "catalog ARRAY subquery must contain SELECT".to_string(),
        ));
    };
    let output = validate_select(select, catalog, scopes)?;
    let [column] = output.as_slice() else {
        return Err(sql_pg_error(
            "catalog ARRAY subquery must return exactly one column".to_string(),
        ));
    };
    column.ty.ok_or_else(|| {
        sql_pg_error("catalog ARRAY subquery has no exactly typed column".to_string())
    })
}

fn target_columns(
    targets: &[Node],
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
) -> Result<Vec<EmptyCatalogColumn>, ExecuteError> {
    let mut columns = Vec::with_capacity(targets.len());
    for target in targets {
        let NodeEnum::ResTarget(target) = node_enum(target)? else {
            return Err(sql_pg_error("malformed empty catalog target".to_string()));
        };
        if !target.indirection.is_empty() {
            return Err(sql_pg_error(
                "empty catalog target indirection is not supported".to_string(),
            ));
        }
        let value = target
            .val
            .as_deref()
            .ok_or_else(|| sql_pg_error("empty catalog target has no value".to_string()))?;
        validate_expression(value, catalog, scopes)?;
        let metadata = expression_metadata(value, catalog, scopes);
        let name = if target.name.is_empty() {
            metadata
                .as_ref()
                .map(|(name, _)| name.clone())
                .unwrap_or_else(|| "?column?".to_string())
        } else {
            target.name.clone()
        };
        columns.push(EmptyCatalogColumn {
            name,
            ty: metadata.map(|(_, ty)| ty),
        });
    }
    Ok(columns)
}

fn merge_set_columns(
    left: Vec<EmptyCatalogColumn>,
    right: Vec<EmptyCatalogColumn>,
) -> Result<Vec<EmptyCatalogColumn>, ExecuteError> {
    if left.len() != right.len() {
        return Err(sql_pg_error(format!(
            "each catalog set-operation arm must have the same width ({} versus {})",
            left.len(),
            right.len()
        )));
    }
    left.into_iter()
        .zip(right)
        .map(|(left, right)| {
            let ty = match (left.ty, right.ty) {
                (Some(left), Some(right)) if left != right => {
                    return Err(sql_pg_error(format!(
                        "catalog set-operation column types differ ({left:?} versus {right:?})"
                    )))
                }
                (Some(ty), _) | (_, Some(ty)) => Some(ty),
                (None, None) => None,
            };
            Ok(EmptyCatalogColumn {
                name: left.name,
                ty,
            })
        })
        .collect()
}

fn widest_catalog_integer(left: SqlType, right: SqlType) -> SqlType {
    if left == SqlType::Int8 || right == SqlType::Int8 {
        SqlType::Int8
    } else if left == SqlType::Int4 || right == SqlType::Int4 {
        SqlType::Int4
    } else {
        SqlType::Int2
    }
}

pub(super) fn merge_catalog_types(
    left: Option<SqlType>,
    right: Option<SqlType>,
    context: &str,
) -> Result<Option<SqlType>, ExecuteError> {
    match (left, right) {
        (Some(left), Some(right)) if left == right => Ok(Some(left)),
        (Some(left), Some(right)) if catalog_integer_type(left) && catalog_integer_type(right) => {
            Ok(Some(widest_catalog_integer(left, right)))
        }
        (Some(left), Some(right)) => Err(sql_pg_error(format!(
            "{context} types differ ({left:?} versus {right:?})"
        ))),
        (Some(ty), None) | (None, Some(ty)) => Ok(Some(ty)),
        (None, None) => Ok(None),
    }
}

fn validate_distinct_clause(
    clause: &[Node],
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
) -> Result<(), ExecuteError> {
    if clause.is_empty() {
        return Ok(());
    }
    if clause.len() == 1 && clause[0].node.is_none() {
        return Ok(());
    }
    for expression in clause {
        validate_expression(expression, catalog, scopes)?;
        catalog_expression_type(
            expression,
            catalog,
            scopes,
            "catalog DISTINCT ON expression",
        )?;
    }
    Err(sql_pg_error(
        "catalog DISTINCT ON is outside the typed empty-result subset".to_string(),
    ))
}

fn validate_sort(
    sort_clause: &[Node],
    catalog: &CatalogSnapshot,
    scope: &EmptyCatalogScope,
    outers: &[&EmptyCatalogScope],
    output: &[EmptyCatalogColumn],
) -> Result<(), ExecuteError> {
    for sort in sort_clause {
        let NodeEnum::SortBy(sort) = node_enum(sort)? else {
            return Err(sql_pg_error("malformed catalog ORDER BY".to_string()));
        };
        if sort.sortby_dir == SortByDir::SortbyUsing as i32 || !sort.use_op.is_empty() {
            return Err(sql_pg_error(
                "catalog ORDER BY USING is not supported".to_string(),
            ));
        }
        let key = sort
            .node
            .as_deref()
            .ok_or_else(|| sql_pg_error("catalog ORDER BY has no key".to_string()))?;
        if let NodeEnum::AConst(constant) = node_enum(key)? {
            if let Some(a_const::Val::Ival(position)) = &constant.val {
                if usize::try_from(position.ival)
                    .ok()
                    .filter(|position| (1..=output.len()).contains(position))
                    .is_some()
                {
                    continue;
                }
                return Err(sql_pg_error(
                    "catalog ORDER BY position is out of range".to_string(),
                ));
            }
        }
        if let NodeEnum::ColumnRef(column) = node_enum(key)? {
            if let [field] = column.fields.as_slice() {
                if let Some(NodeEnum::String(name)) = field.node.as_ref() {
                    let count = output
                        .iter()
                        .filter(|column| column.name == name.sval)
                        .count();
                    if count == 1 {
                        continue;
                    }
                    if count > 1 {
                        return Err(sql_pg_error(format!(
                            "ORDER BY \"{}\" is ambiguous",
                            name.sval
                        )));
                    }
                }
            }
        }
        let mut scopes = vec![scope];
        scopes.extend_from_slice(outers);
        validate_expression(key, catalog, &scopes)?;
        catalog_expression_type(key, catalog, &scopes, "catalog ORDER BY expression")?;
    }
    Ok(())
}

fn validate_values(
    stmt: &SelectStmt,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
) -> Result<Vec<EmptyCatalogColumn>, ExecuteError> {
    let mut width = None;
    let mut columns = Vec::new();
    for row in &stmt.values_lists {
        let NodeEnum::List(row) = node_enum(row)? else {
            return Err(sql_pg_error("catalog VALUES row is malformed".to_string()));
        };
        if width
            .replace(row.items.len())
            .is_some_and(|width| width != row.items.len())
        {
            return Err(sql_pg_error(
                "catalog VALUES rows have different widths".to_string(),
            ));
        }
        for value in &row.items {
            validate_expression(value, catalog, scopes)?;
        }
        if columns.is_empty() {
            columns = row
                .items
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    Ok(EmptyCatalogColumn {
                        name: format!("column{}", index + 1),
                        ty: catalog_expression_type(
                            value,
                            catalog,
                            scopes,
                            "catalog VALUES expression",
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, ExecuteError>>()?;
        } else {
            for (column, value) in columns.iter_mut().zip(&row.items) {
                let candidate =
                    catalog_expression_type(value, catalog, scopes, "catalog VALUES expression")?;
                column.ty = merge_catalog_types(column.ty, candidate, "catalog VALUES column")?;
            }
        }
    }
    Ok(columns)
}

fn validate_limit_expression(
    node: &Node,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
    context: &str,
) -> Result<(), ExecuteError> {
    if node_has_current_level_aggregate(node) {
        return Err(sql_pg_error(format!(
            "{context} cannot contain an aggregate"
        )));
    }
    validate_expression(node, catalog, scopes)?;
    let ty = catalog_expression_type(node, catalog, scopes, context)?;
    if ty.is_some_and(|ty| !catalog_integer_type(ty)) {
        return Err(sql_pg_error(format!(
            "{context} requires an integer expression"
        )));
    }
    Ok(())
}

pub(super) fn validate_select(
    stmt: &SelectStmt,
    catalog: &CatalogSnapshot,
    outers: &[&EmptyCatalogScope],
) -> Result<Vec<EmptyCatalogColumn>, ExecuteError> {
    if stmt.into_clause.is_some()
        || stmt.with_clause.is_some()
        || !stmt.locking_clause.is_empty()
        || !stmt.window_clause.is_empty()
        || stmt.group_distinct
    {
        return Err(sql_pg_error(
            "catalog empty-result SELECT has unsupported stateful/window clauses".to_string(),
        ));
    }
    if stmt.op != SetOperation::SetopNone as i32 {
        let left = validate_select(
            stmt.larg
                .as_deref()
                .ok_or_else(|| sql_pg_error("catalog set operation has no left arm".to_string()))?,
            catalog,
            outers,
        )?;
        let right = validate_select(
            stmt.rarg.as_deref().ok_or_else(|| {
                sql_pg_error("catalog set operation has no right arm".to_string())
            })?,
            catalog,
            outers,
        )?;
        let output = merge_set_columns(left, right)?;
        let scope = EmptyCatalogScope::default();
        let scopes = [&scope];
        validate_distinct_clause(&stmt.distinct_clause, catalog, &scopes)?;
        validate_sort(&stmt.sort_clause, catalog, &scope, outers, &output)?;
        if let Some(limit) = stmt.limit_count.as_deref() {
            validate_limit_expression(limit, catalog, &scopes, "catalog set-operation LIMIT")?;
        }
        if let Some(offset) = stmt.limit_offset.as_deref() {
            validate_limit_expression(offset, catalog, &scopes, "catalog set-operation OFFSET")?;
        }
        return Ok(output);
    }

    let multiple_sources =
        stmt.from_clause.len() > 1 || stmt.from_clause.iter().any(from_node_is_join);
    if multiple_sources && stmt.from_clause.iter().any(from_node_has_column_alias_list) {
        return Err(sql_pg_error(
            "JOIN and comma-join relation column-alias lists are not supported".to_string(),
        ));
    }

    let mut scope = EmptyCatalogScope::default();
    for from in &stmt.from_clause {
        bind_from_node(from, catalog, &mut scope, outers)?;
    }
    let mut scopes = vec![&scope];
    scopes.extend_from_slice(outers);
    let output = if !stmt.values_lists.is_empty() {
        if !stmt.from_clause.is_empty() || !stmt.target_list.is_empty() {
            return Err(sql_pg_error(
                "catalog VALUES cannot be combined with FROM/targets".to_string(),
            ));
        }
        validate_values(stmt, catalog, &scopes)?
    } else {
        target_columns(&stmt.target_list, catalog, &scopes)?
    };
    validate_distinct_clause(&stmt.distinct_clause, catalog, &scopes)?;
    if let Some(predicate) = stmt.where_clause.as_deref() {
        if node_has_current_level_aggregate(predicate) {
            return Err(sql_pg_error(
                "catalog WHERE cannot contain an aggregate".to_string(),
            ));
        }
        validate_expression(predicate, catalog, &scopes)?;
        require_catalog_boolean(predicate, catalog, &scopes, "catalog WHERE predicate")?;
    }
    for group in &stmt.group_clause {
        if matches!(group.node.as_ref(), Some(NodeEnum::GroupingSet(_))) {
            return Err(sql_pg_error(
                "catalog grouping sets are outside the typed empty-result subset".to_string(),
            ));
        }
        if node_has_current_level_aggregate(group) {
            return Err(sql_pg_error(
                "catalog GROUP BY cannot contain an aggregate".to_string(),
            ));
        }
        validate_expression(group, catalog, &scopes)?;
    }
    if let Some(having) = stmt.having_clause.as_deref() {
        validate_expression(having, catalog, &scopes)?;
        require_catalog_boolean(having, catalog, &scopes, "catalog HAVING predicate")?;
    }
    if let Some(limit) = stmt.limit_count.as_deref() {
        validate_limit_expression(limit, catalog, &scopes, "catalog LIMIT")?;
    }
    if let Some(offset) = stmt.limit_offset.as_deref() {
        validate_limit_expression(offset, catalog, &scopes, "catalog OFFSET")?;
    }
    validate_group_semantics(stmt, &output)?;
    validate_sort(&stmt.sort_clause, catalog, &scope, outers, &output)?;
    Ok(output)
}
