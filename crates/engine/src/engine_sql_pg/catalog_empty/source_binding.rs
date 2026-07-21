//! FROM-source binding for the bounded empty-catalog query subset.

use super::*;

fn range_function_call(node: &Node) -> Option<&pg_query::protobuf::FuncCall> {
    match node.node.as_ref()? {
        NodeEnum::FuncCall(function) => Some(function),
        NodeEnum::List(list) => list.items.iter().find_map(range_function_call),
        _ => None,
    }
}

pub(super) fn bind_from_node(
    node: &Node,
    catalog: &CatalogSnapshot,
    scope: &mut EmptyCatalogScope,
    outers: &[&EmptyCatalogScope],
) -> Result<(), ExecuteError> {
    match node_enum(node)? {
        NodeEnum::RangeVar(range) => push_range_binding(range, catalog, scope),
        NodeEnum::JoinExpr(join) => {
            if !matches!(
                join.jointype,
                value if value == JoinType::JoinInner as i32
                    || value == JoinType::JoinLeft as i32
                    || value == JoinType::JoinRight as i32
                    || value == JoinType::JoinFull as i32
            ) || join.is_natural
                || join.alias.is_some()
                || join.join_using_alias.is_some()
                || !join.using_clause.is_empty()
            {
                return Err(sql_pg_error(
                    "empty catalog binding supports qualified JOIN ... ON only".to_string(),
                ));
            }
            bind_from_node(
                join.larg
                    .as_deref()
                    .ok_or_else(|| sql_pg_error("catalog JOIN has no left side".to_string()))?,
                catalog,
                scope,
                outers,
            )?;
            bind_from_node(
                join.rarg
                    .as_deref()
                    .ok_or_else(|| sql_pg_error("catalog JOIN has no right side".to_string()))?,
                catalog,
                scope,
                outers,
            )?;
            let mut scopes = vec![&*scope];
            scopes.extend_from_slice(outers);
            if let Some(predicate) = join.quals.as_deref() {
                if node_has_current_level_aggregate(predicate) {
                    return Err(sql_pg_error(
                        "catalog JOIN condition cannot contain an aggregate".to_string(),
                    ));
                }
                validate_expression(predicate, catalog, &scopes)?;
                require_catalog_boolean(predicate, catalog, &scopes, "catalog JOIN condition")
            } else if join.jointype == JoinType::JoinInner as i32 {
                Ok(())
            } else {
                Err(sql_pg_error(
                    "catalog outer JOIN requires an ON condition".to_string(),
                ))
            }
        }
        NodeEnum::RangeFunction(function) => {
            if function.functions.len() != 1
                || function.ordinality
                || function.is_rowsfrom
                || !function.coldeflist.is_empty()
            {
                return Err(sql_pg_error(
                    "catalog range function is outside the bounded generate_series subset"
                        .to_string(),
                ));
            }
            let call = range_function_call(&function.functions[0]).ok_or_else(|| {
                sql_pg_error("catalog range function has no function call".to_string())
            })?;
            if catalog_function_name(call)? != "generate_series" {
                return Err(sql_pg_error(
                    "only pg_catalog.generate_series is supported as a catalog range function"
                        .to_string(),
                ));
            }
            let mut scopes = vec![&*scope];
            scopes.extend_from_slice(outers);
            validate_function(call, catalog, &scopes)?;
            let alias = function.alias.as_ref().ok_or_else(|| {
                sql_pg_error("catalog generate_series requires an alias".to_string())
            })?;
            if alias.colnames.len() > 1 {
                return Err(sql_pg_error(
                    "catalog generate_series supports one output column".to_string(),
                ));
            }
            let column_name = match alias.colnames.first() {
                Some(column) => match node_enum(column)? {
                    NodeEnum::String(column) => column.sval.clone(),
                    _ => {
                        return Err(sql_pg_error(
                            "catalog range-function alias is malformed".to_string(),
                        ))
                    }
                },
                None => alias.aliasname.clone(),
            };
            if scope
                .bindings
                .iter()
                .any(|binding| binding.qualifier == alias.aliasname)
            {
                return Err(sql_pg_error(format!(
                    "table name \"{}\" specified more than once",
                    alias.aliasname
                )));
            }
            scope.bindings.push(EmptyCatalogBinding {
                qualifier: alias.aliasname.clone(),
                table: catalog_relation_table(
                    "pg_catalog",
                    "__empty_generate_series",
                    &[(column_name.as_str(), SqlType::Int4)],
                ),
                array_elements: BTreeMap::new(),
            });
            Ok(())
        }
        _ => Err(sql_pg_error(
            "empty catalog FROM item is outside the statically bound subset".to_string(),
        )),
    }
}
