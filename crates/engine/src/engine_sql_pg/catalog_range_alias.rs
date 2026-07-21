use super::*;

/// Preserve the lookup class carried by a PostgreSQL `RangeVar`. Public relations use their bare
/// residency key plus separate `public_only` provenance; synthesized system relations keep their
/// schema-qualified key so a public shadow can never replace them.
pub(super) fn catalog_range_relation_key(
    range: &pg_query::protobuf::RangeVar,
) -> Result<String, ExecuteError> {
    if !range.catalogname.is_empty() {
        return Err(sql_pg_error(
            "cross-database relation references are not supported".to_string(),
        ));
    }
    match range.schemaname.as_str() {
        "" | "public" => Ok(range.relname.clone()),
        "pg_catalog" | "information_schema" => {
            Ok(format!("{}.{}", range.schemaname, range.relname))
        }
        schema => Err(sql_pg_error(format!(
            "schema \"{schema}\" does not exist on the GPU relation path"
        ))),
    }
}

pub(super) fn rank_window_relation_binding(from: &Node) -> Result<(String, String), ExecuteError> {
    let NodeEnum::RangeVar(range) = node_enum(from)? else {
        return Err(sql_pg_error(
            "the GPU rank-window path supports one base relation".to_string(),
        ));
    };
    if from_node_has_column_alias_list(from) {
        return Err(sql_pg_error(
            "relation column-alias lists are not supported on the GPU rank-window path".to_string(),
        ));
    }
    let table_name = catalog_range_relation_key(range)?;
    let qualifier = range
        .alias
        .as_ref()
        .map(|alias| alias.aliasname.clone())
        .unwrap_or_else(|| range.relname.clone());
    Ok((table_name, qualifier))
}

pub(super) fn apply_catalog_range_column_aliases(
    table: &mut RelationalTable,
    range: &pg_query::protobuf::RangeVar,
    exposed_width: usize,
) -> Result<(), ExecuteError> {
    let Some(alias) = range.alias.as_ref() else {
        return Ok(());
    };
    if alias.colnames.is_empty() {
        return Ok(());
    }
    if !matches!(table.schema.as_str(), "pg_catalog" | "information_schema") {
        return Err(sql_pg_error(
            "relation column-alias lists are supported only for synthesized catalog relations"
                .to_string(),
        ));
    }
    if exposed_width > table.columns.len() || alias.colnames.len() > exposed_width {
        return Err(sql_pg_error(format!(
            "table alias {:?} specifies {} columns for a relation with {exposed_width} exposed columns",
            alias.aliasname,
            alias.colnames.len()
        )));
    }
    let names = alias
        .colnames
        .iter()
        .map(|column| match node_enum(column)? {
            NodeEnum::String(column) if !column.sval.is_empty() => Ok(column.sval.clone()),
            _ => Err(sql_pg_error(
                "catalog relation column alias is malformed".to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut final_names = table.columns[..exposed_width]
        .iter()
        .map(|column| column.name.clone())
        .collect::<Vec<_>>();
    for (slot, name) in final_names.iter_mut().zip(&names) {
        *slot = name.clone();
    }
    for (index, name) in final_names.iter().enumerate() {
        if final_names[..index].iter().any(|earlier| earlier == name) {
            return Err(sql_pg_error(format!(
                "catalog relation column alias {name:?} is ambiguous"
            )));
        }
    }
    for (column, name) in table.columns.iter_mut().zip(names) {
        column.name = name;
    }
    Ok(())
}

pub(super) fn from_node_has_column_alias_list(node: &Node) -> bool {
    match node.node.as_ref() {
        Some(NodeEnum::RangeVar(range)) => range
            .alias
            .as_ref()
            .is_some_and(|alias| !alias.colnames.is_empty()),
        Some(NodeEnum::JoinExpr(join)) => {
            join.larg
                .as_deref()
                .is_some_and(from_node_has_column_alias_list)
                || join
                    .rarg
                    .as_deref()
                    .is_some_and(from_node_has_column_alias_list)
        }
        _ => false,
    }
}

pub(super) fn from_node_is_join(node: &Node) -> bool {
    matches!(node.node.as_ref(), Some(NodeEnum::JoinExpr(_)))
}
