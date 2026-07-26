//! Fail-closed binding for catalog queries whose inner input is provably empty.
//!
//! PostgreSQL still binds every relation, column, function, clause, and set-operation arm when a
//! relation has zero rows. This module performs that control-plane binding before constructing the
//! typed zero-row transient relation that is executed on the GPU.

use super::catalog_presentation::catalog_function_name;
use super::*;

mod cast_validation;
mod grouping;
mod select_validation;
mod source_binding;

use cast_validation::validate_catalog_scalar_cast;
use grouping::validate_group_semantics;
use select_validation::{array_sublink_element_type, merge_catalog_types, validate_select};
use source_binding::bind_from_node;

#[derive(Clone)]
struct EmptyCatalogBinding {
    qualifier: String,
    table: RelationalTable,
    /// Array capability follows the immutable source column position through RangeVar renaming.
    /// It is never inferred from the final exposed name.
    array_elements: BTreeMap<String, SqlType>,
}

#[derive(Default)]
struct EmptyCatalogScope {
    bindings: Vec<EmptyCatalogBinding>,
}

#[derive(Clone)]
struct EmptyCatalogColumn {
    name: String,
    ty: Option<SqlType>,
}

fn collect_from_relations(node: &Node, relations: &mut Vec<String>) {
    match node.node.as_ref() {
        Some(NodeEnum::RangeVar(range)) => {
            let relation = if !range.catalogname.is_empty() {
                format!(
                    "{}.{}.{}",
                    range.catalogname, range.schemaname, range.relname
                )
            } else {
                match range.schemaname.as_str() {
                    "" => range.relname.clone(),
                    "pg_catalog" | "information_schema" => {
                        format!("{}.{}", range.schemaname, range.relname)
                    }
                    schema => format!("{schema}.{}", range.relname),
                }
            };
            relations.push(relation);
        }
        Some(NodeEnum::JoinExpr(join)) => {
            if let Some(left) = join.larg.as_deref() {
                collect_from_relations(left, relations);
            }
            if let Some(right) = join.rarg.as_deref() {
                collect_from_relations(right, relations);
            }
        }
        _ => {}
    }
}

fn from_node_has_outer_join(node: &Node) -> bool {
    match node.node.as_ref() {
        Some(NodeEnum::JoinExpr(join)) => {
            join.jointype != JoinType::JoinInner as i32
                || join.larg.as_deref().is_some_and(from_node_has_outer_join)
                || join.rarg.as_deref().is_some_and(from_node_has_outer_join)
        }
        _ => false,
    }
}

fn catalog_function_is_aggregate(function: &pg_query::protobuf::FuncCall) -> bool {
    catalog_function_name(function)
        .map(|name| {
            matches!(
                name.as_str(),
                "avg" | "bool_and" | "bool_or" | "count" | "max" | "min" | "string_agg" | "sum"
            )
        })
        .unwrap_or(true)
        || function.agg_distinct
        || function.agg_filter.is_some()
        || function.agg_within_group
        || !function.agg_order.is_empty()
        || function.agg_star
}

fn node_has_current_level_aggregate(node: &Node) -> bool {
    match node.node.as_ref() {
        Some(NodeEnum::SubLink(_)) => false,
        Some(NodeEnum::FuncCall(function)) => {
            catalog_function_is_aggregate(function)
                || function.args.iter().any(node_has_current_level_aggregate)
        }
        Some(NodeEnum::AExpr(expression)) => {
            expression
                .lexpr
                .as_deref()
                .is_some_and(node_has_current_level_aggregate)
                || expression
                    .rexpr
                    .as_deref()
                    .is_some_and(node_has_current_level_aggregate)
        }
        Some(NodeEnum::BoolExpr(expression)) => {
            expression.args.iter().any(node_has_current_level_aggregate)
        }
        Some(NodeEnum::CaseExpr(expression)) => {
            expression
                .arg
                .as_deref()
                .is_some_and(node_has_current_level_aggregate)
                || expression.args.iter().any(node_has_current_level_aggregate)
                || expression
                    .defresult
                    .as_deref()
                    .is_some_and(node_has_current_level_aggregate)
        }
        Some(NodeEnum::CaseWhen(arm)) => {
            arm.expr
                .as_deref()
                .is_some_and(node_has_current_level_aggregate)
                || arm
                    .result
                    .as_deref()
                    .is_some_and(node_has_current_level_aggregate)
        }
        Some(NodeEnum::TypeCast(cast)) => cast
            .arg
            .as_deref()
            .is_some_and(node_has_current_level_aggregate),
        Some(NodeEnum::CollateClause(collate)) => collate
            .arg
            .as_deref()
            .is_some_and(node_has_current_level_aggregate),
        Some(NodeEnum::AIndirection(indirection)) => {
            indirection
                .arg
                .as_deref()
                .is_some_and(node_has_current_level_aggregate)
                || indirection
                    .indirection
                    .iter()
                    .any(node_has_current_level_aggregate)
        }
        Some(NodeEnum::AArrayExpr(array)) => {
            array.elements.iter().any(node_has_current_level_aggregate)
        }
        Some(NodeEnum::ResTarget(target)) => target
            .val
            .as_deref()
            .is_some_and(node_has_current_level_aggregate),
        Some(NodeEnum::List(list)) => list.items.iter().any(node_has_current_level_aggregate),
        _ => false,
    }
}

fn select_tree_is_proven_empty_catalog(stmt: &SelectStmt, catalog: &CatalogSnapshot) -> bool {
    if stmt.op != SetOperation::SetopNone as i32 {
        return stmt
            .larg
            .as_deref()
            .is_some_and(|left| select_tree_is_proven_empty_catalog(left, catalog))
            && stmt
                .rarg
                .as_deref()
                .is_some_and(|right| select_tree_is_proven_empty_catalog(right, catalog));
    }
    if stmt.from_clause.iter().any(from_node_has_outer_join)
        || stmt
            .group_clause
            .iter()
            .any(|group| matches!(group.node.as_ref(), Some(NodeEnum::GroupingSet(_))))
        || (!stmt.group_clause.is_empty() && stmt.from_clause.is_empty())
        || (stmt.group_clause.is_empty()
            && (stmt.having_clause.is_some()
                || stmt
                    .target_list
                    .iter()
                    .any(node_has_current_level_aggregate)
                || stmt
                    .sort_clause
                    .iter()
                    .any(node_has_current_level_aggregate)))
    {
        return false;
    }
    let mut relations = Vec::new();
    for from in &stmt.from_clause {
        collect_from_relations(from, &mut relations);
    }
    let mut has_empty = false;
    for relation in relations {
        if public_relation_name_exists(catalog, &relation) {
            continue;
        }
        match synthesize_catalog_relation(&relation, catalog) {
            Some((_, rows)) => has_empty |= rows.is_empty(),
            None => return false,
        }
    }
    has_empty
}

fn select_tree_has_empty_catalog_source(stmt: &SelectStmt, catalog: &CatalogSnapshot) -> bool {
    if stmt.op != SetOperation::SetopNone as i32 {
        return stmt
            .larg
            .as_deref()
            .is_some_and(|left| select_tree_has_empty_catalog_source(left, catalog))
            || stmt
                .rarg
                .as_deref()
                .is_some_and(|right| select_tree_has_empty_catalog_source(right, catalog));
    }
    let mut relations = Vec::new();
    for from in &stmt.from_clause {
        collect_from_relations(from, &mut relations);
    }
    relations.into_iter().any(|relation| {
        if !relation.contains('.') && public_relation_name_exists(catalog, &relation) {
            return false;
        }
        synthesize_catalog_relation(&relation, catalog).is_some_and(|(_, rows)| rows.is_empty())
    })
}

fn resolve_range_table(
    range: &pg_query::protobuf::RangeVar,
    catalog: &CatalogSnapshot,
) -> Result<(RelationalTable, BTreeMap<String, SqlType>), ExecuteError> {
    let relation_key = catalog_range_relation_key(range)?;
    let user = || catalog.relational_catalog.get(&range.relname).cloned();
    let synthesized = || {
        synthesize_catalog_relation(&relation_key, catalog)
            .map(|(table, _)| table)
            .or_else(|| empty_binding_only_catalog_relation(&range.relname))
    };
    let mut table = match range.schemaname.as_str() {
        "" if public_relation_name_exists(catalog, &range.relname) => user(),
        "" => user().or_else(synthesized),
        "public" => user().filter(|table| table.schema == "public"),
        "pg_catalog" | "information_schema" => {
            synthesized().filter(|table| table.schema == range.schemaname)
        }
        schema => {
            return Err(sql_pg_error(format!(
                "schema \"{schema}\" does not exist on the GPU catalog path"
            )))
        }
    }
    .ok_or_else(|| sql_pg_error(format!("relation \"{}\" does not exist", range.relname)))?;
    add_empty_binding_only_columns(&mut table)?;
    let array_elements_by_position = table
        .columns
        .iter()
        .map(|column| match table.schema.as_str() {
            "pg_catalog" | "information_schema" => catalog_array_column_element(&column.name),
            _ => None,
        })
        .collect::<Vec<_>>();
    let exposed_width = table.columns.len();
    apply_catalog_range_column_aliases(&mut table, range, exposed_width)?;
    let array_elements = table
        .columns
        .iter()
        .zip(array_elements_by_position)
        .filter_map(|(column, element)| element.map(|element| (column.name.clone(), element)))
        .collect();
    Ok((table, array_elements))
}

fn empty_binding_only_catalog_relation(name: &str) -> Option<RelationalTable> {
    match name {
        // psql's empty pg_policy query contains a correlated role-name subquery. The outer relation
        // is proven empty, so the subquery cannot execute, but PostgreSQL still binds these columns.
        "pg_roles" => Some(catalog_relation_table(
            "pg_catalog",
            "pg_roles",
            &[("oid", SqlType::Int4), ("rolname", SqlType::Text)],
        )),
        _ => None,
    }
}

fn add_empty_binding_only_columns(table: &mut RelationalTable) -> Result<(), ExecuteError> {
    let columns = match table.name.as_str() {
        // Referenced only by psql's pg_inherits query, whose other input is proven empty.
        "pg_class" => [("relpartbound", SqlType::Text)].as_slice(),
        _ => &[],
    };
    for (name, ty) in columns {
        if table.columns.iter().any(|column| column.name == *name) {
            continue;
        }
        let attnum = i16::try_from(table.columns.len() + 1)
            .map_err(|_| sql_pg_error("empty binding relation is too wide".to_string()))?;
        table.columns.push(RelationalColumn {
            id: 0,
            table_oid: table.oid,
            attnum,
            name: (*name).to_string(),
            ty: *ty,
            domain: None,
            default: None,
            type_oid: ty.postgres_oid(),
            type_size: ty.type_size(),
        });
    }
    Ok(())
}

fn push_range_binding(
    range: &pg_query::protobuf::RangeVar,
    catalog: &CatalogSnapshot,
    scope: &mut EmptyCatalogScope,
) -> Result<(), ExecuteError> {
    let (table, array_elements) = resolve_range_table(range, catalog)?;
    let qualifier = range
        .alias
        .as_ref()
        .map(|alias| alias.aliasname.clone())
        .unwrap_or_else(|| range.relname.clone());
    if scope
        .bindings
        .iter()
        .any(|binding| binding.qualifier == qualifier)
    {
        return Err(sql_pg_error(format!(
            "table name \"{qualifier}\" specified more than once"
        )));
    }
    scope.bindings.push(EmptyCatalogBinding {
        qualifier,
        table,
        array_elements,
    });
    Ok(())
}

fn resolve_column(
    column: &pg_query::protobuf::ColumnRef,
    scopes: &[&EmptyCatalogScope],
) -> Result<(String, SqlType, Option<SqlType>), ExecuteError> {
    let parts = column
        .fields
        .iter()
        .map(|field| match field.node.as_ref() {
            Some(NodeEnum::String(name)) => Ok(name.sval.as_str()),
            _ => Err(sql_pg_error(
                "empty catalog binding requires a named column reference".to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    match parts.as_slice() {
        [name] => {
            for scope in scopes {
                let matches = scope
                    .bindings
                    .iter()
                    .filter_map(|binding| {
                        binding
                            .table
                            .columns
                            .iter()
                            .find(|column| column.name == *name)
                            .map(|column| (binding, column))
                    })
                    .collect::<Vec<_>>();
                match matches.as_slice() {
                    [] => continue,
                    [(binding, column)] => {
                        return Ok((
                            column.name.clone(),
                            column.ty,
                            binding.array_elements.get(&column.name).copied(),
                        ))
                    }
                    _ => {
                        return Err(sql_pg_error(format!(
                            "column reference \"{name}\" is ambiguous"
                        )))
                    }
                }
            }
            Err(sql_pg_error(format!("column \"{name}\" does not exist")))
        }
        [qualifier, name] => {
            for scope in scopes {
                if let Some(binding) = scope
                    .bindings
                    .iter()
                    .find(|binding| binding.qualifier == *qualifier)
                {
                    let column = binding
                        .table
                        .columns
                        .iter()
                        .find(|column| column.name == *name)
                        .ok_or_else(|| {
                            sql_pg_error(format!("column \"{qualifier}.{name}\" does not exist"))
                        })?;
                    return Ok((
                        column.name.clone(),
                        column.ty,
                        binding.array_elements.get(&column.name).copied(),
                    ));
                }
            }
            Err(sql_pg_error(format!(
                "missing FROM-clause entry for table \"{qualifier}\""
            )))
        }
        _ => Err(sql_pg_error(
            "multi-part empty catalog column references are not supported".to_string(),
        )),
    }
}

fn validate_type_name(type_name: &pg_query::protobuf::TypeName) -> Result<String, ExecuteError> {
    if type_name.setof || type_name.pct_type || !type_name.typmods.is_empty() {
        return Err(sql_pg_error(
            "complex catalog casts are not supported by the empty-result binder".to_string(),
        ));
    }
    let parts = type_name
        .names
        .iter()
        .map(|part| match node_enum(part)? {
            NodeEnum::String(part) => Ok(part.sval.to_ascii_lowercase()),
            _ => Err(sql_pg_error(
                "catalog cast type name is malformed".to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let name = match parts.as_slice() {
        [name] => name.as_str(),
        [schema, name] if schema == "pg_catalog" => name.as_str(),
        _ => {
            return Err(sql_pg_error(format!(
                "catalog cast type {} is not an unqualified or pg_catalog type",
                parts.join(".")
            )))
        }
    };
    if !matches!(
        name,
        "bool"
            | "int2"
            | "int4"
            | "int8"
            | "oid"
            | "regclass"
            | "regnamespace"
            | "regtype"
            | "text"
    ) {
        return Err(sql_pg_error(format!(
            "catalog cast type {name} is outside the empty-result subset"
        )));
    }
    Ok(name.to_string())
}

fn function_arity_is_valid(name: &str, len: usize) -> bool {
    match name {
        "array_to_string" => (2..=3).contains(&len),
        "array_upper" | "col_description" | "format_type" | "string_agg" => len == 2,
        "count" => len <= 1,
        "generate_series" | "pg_get_expr" => (2..=3).contains(&len),
        "pg_get_indexdef" => (1..=3).contains(&len),
        "obj_description" | "pg_get_constraintdef" | "pg_get_triggerdef" => (1..=2).contains(&len),
        "pg_size_pretty" | "pg_table_size" => len == 1,
        "set_config" => len == 3,
        "pg_function_is_visible"
        | "pg_get_partkeydef"
        | "pg_get_statisticsobjdef_columns"
        | "pg_get_userbyid"
        | "pg_partition_ancestors"
        | "pg_relation_is_publishable"
        | "pg_table_is_visible"
        | "pg_type_is_visible" => len == 1,
        _ => false,
    }
}

fn validate_function(
    function: &pg_query::protobuf::FuncCall,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
) -> Result<(), ExecuteError> {
    let name = catalog_function_name(function)?;
    if !function_arity_is_valid(&name, function.args.len()) {
        return Err(sql_pg_error(format!(
            "catalog function {name} has an unsupported argument count {}",
            function.args.len()
        )));
    }
    if function.over.is_some() || function.func_variadic {
        return Err(sql_pg_error(format!(
            "catalog function {name} has unsupported modifiers"
        )));
    }
    if function.agg_within_group {
        return Err(sql_pg_error(format!(
            "catalog aggregate {name} does not support WITHIN GROUP"
        )));
    }
    if function.agg_star && name != "count" {
        return Err(sql_pg_error(format!(
            "catalog function {name} does not accept aggregate star"
        )));
    }
    if name == "count"
        && ((function.agg_star && !function.args.is_empty())
            || (!function.agg_star && function.args.is_empty()))
    {
        return Err(sql_pg_error(
            "catalog count requires either star or one argument".to_string(),
        ));
    }
    if catalog_function_is_aggregate(function)
        && function.args.iter().any(node_has_current_level_aggregate)
    {
        return Err(sql_pg_error(
            "nested catalog aggregate functions are not supported".to_string(),
        ));
    }
    for arg in &function.args {
        validate_expression(arg, catalog, scopes)?;
    }
    let arg_type = |index: usize| {
        function
            .args
            .get(index)
            .and_then(|arg| expression_metadata(arg, catalog, scopes))
            .map(|(_, ty)| ty)
    };
    let require = |index: usize, expected: SqlType| -> Result<(), ExecuteError> {
        if arg_type(index) != Some(expected) {
            return Err(sql_pg_error(format!(
                "catalog function {name} argument {} must bind as {expected:?}",
                index + 1
            )));
        }
        Ok(())
    };
    let require_oid = |index: usize| -> Result<(), ExecuteError> {
        let argument = function.args.get(index).ok_or_else(|| {
            sql_pg_error(format!(
                "catalog function {name} is missing argument {}",
                index + 1
            ))
        })?;
        let valid = arg_type(index) == Some(SqlType::Int4)
            || matches!(
                node_enum(argument),
                Ok(NodeEnum::AConst(constant))
                    if matches!(&constant.val, Some(a_const::Val::Sval(value)) if value.sval.parse::<i32>().is_ok())
            )
            || matches!(
                node_enum(argument),
                Ok(NodeEnum::TypeCast(cast))
                    if cast
                        .type_name
                        .as_ref()
                        .and_then(|target| validate_type_name(target).ok())
                        .is_some_and(|target| matches!(target.as_str(), "oid" | "regclass"))
            );
        if !valid {
            return Err(sql_pg_error(format!(
                "catalog function {name} argument {} must bind as an OID",
                index + 1
            )));
        }
        Ok(())
    };
    match name.as_str() {
        "array_to_string" => {
            catalog_array_element_type(&function.args[0], catalog, scopes)?;
            require(1, SqlType::Text)?;
        }
        "array_upper" => {
            catalog_array_element_type(&function.args[0], catalog, scopes)?;
            require(1, SqlType::Int4)?;
        }
        "col_description" | "format_type" => {
            require(0, SqlType::Int4)?;
            require(1, SqlType::Int4)?;
        }
        "generate_series" => {
            for index in 0..function.args.len() {
                require(index, SqlType::Int4)?;
            }
        }
        "obj_description" => {
            require(0, SqlType::Int4)?;
            if function.args.len() == 2 {
                require(1, SqlType::Text)?;
            }
        }
        "pg_function_is_visible"
        | "pg_get_partkeydef"
        | "pg_get_statisticsobjdef_columns"
        | "pg_get_userbyid"
        | "pg_partition_ancestors"
        | "pg_relation_is_publishable"
        | "pg_table_is_visible"
        | "pg_type_is_visible" => require_oid(0)?,
        "pg_get_constraintdef" | "pg_get_triggerdef" => {
            require_oid(0)?;
            if function.args.len() == 2 {
                require(1, SqlType::Bool)?;
            }
        }
        "pg_get_expr" => {
            require(0, SqlType::Text)?;
            require(1, SqlType::Int4)?;
            if function.args.len() == 3 {
                require(2, SqlType::Bool)?;
            }
        }
        "pg_get_indexdef" => {
            require_oid(0)?;
            if function.args.len() >= 2 {
                require(1, SqlType::Int4)?;
            }
            if function.args.len() == 3 {
                require(2, SqlType::Bool)?;
            }
        }
        "pg_size_pretty" => {
            let ty = arg_type(0).ok_or_else(|| {
                sql_pg_error("pg_size_pretty argument has no catalog type".to_string())
            })?;
            if !catalog_integer_type(ty) {
                return Err(sql_pg_error(
                    "pg_size_pretty argument must be an integer".to_string(),
                ));
            }
        }
        "pg_table_size" => require_oid(0)?,
        "string_agg" => {
            require(0, SqlType::Text)?;
            require(1, SqlType::Text)?;
        }
        "set_config" => {
            require(0, SqlType::Text)?;
            require(1, SqlType::Text)?;
            require(2, SqlType::Bool)?;
        }
        _ => {}
    }
    for order in &function.agg_order {
        if node_has_current_level_aggregate(order) {
            return Err(sql_pg_error(
                "catalog aggregate ORDER BY cannot contain another aggregate".to_string(),
            ));
        }
        validate_expression(order, catalog, scopes)?;
    }
    if let Some(filter) = function.agg_filter.as_deref() {
        if node_has_current_level_aggregate(filter) {
            return Err(sql_pg_error(
                "catalog aggregate FILTER cannot contain another aggregate".to_string(),
            ));
        }
        validate_expression(filter, catalog, scopes)?;
        require_catalog_boolean(filter, catalog, scopes, "catalog aggregate FILTER")?;
    }
    Ok(())
}

fn catalog_null_literal(node: &Node) -> bool {
    matches!(
        node.node.as_ref(),
        Some(NodeEnum::AConst(constant)) if constant.isnull
    )
}

fn catalog_integer_type(ty: SqlType) -> bool {
    matches!(ty, SqlType::Int2 | SqlType::Int4 | SqlType::Int8)
}

fn catalog_named_scalar_type(name: &str) -> Option<SqlType> {
    match name {
        "bool" => Some(SqlType::Bool),
        "int2" => Some(SqlType::Int2),
        "int4" | "oid" | "regclass" | "regnamespace" | "regtype" => Some(SqlType::Int4),
        "int8" => Some(SqlType::Int8),
        "text" => Some(SqlType::Text),
        _ => None,
    }
}

fn catalog_array_column_element(name: &str) -> Option<SqlType> {
    match name {
        "polroles" => Some(SqlType::Int4),
        "prattrs" => Some(SqlType::Int2),
        "stxkind" => Some(SqlType::Text),
        _ => None,
    }
}

fn catalog_array_element_type(
    node: &Node,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
) -> Result<SqlType, ExecuteError> {
    match node_enum(node)? {
        NodeEnum::ColumnRef(column) => {
            let (name, _, element) = resolve_column(column, scopes)?;
            element.ok_or_else(|| {
                sql_pg_error(format!(
                    "catalog column {name:?} is not a modeled array source"
                ))
            })
        }
        NodeEnum::TypeCast(cast) => {
            let target = cast
                .type_name
                .as_ref()
                .ok_or_else(|| sql_pg_error("catalog array cast has no target type".to_string()))?;
            if target.array_bounds.len() != 1 {
                return Err(sql_pg_error(
                    "catalog array cast requires exactly one dimension".to_string(),
                ));
            }
            let target = validate_type_name(target)?;
            let target = catalog_named_scalar_type(&target).ok_or_else(|| {
                sql_pg_error("catalog array cast has an unsupported element type".to_string())
            })?;
            let source = cast
                .arg
                .as_deref()
                .ok_or_else(|| sql_pg_error("catalog array cast has no source".to_string()))?;
            let NodeEnum::ColumnRef(source) = node_enum(source)? else {
                return Err(sql_pg_error(
                    "catalog array cast requires a modeled array column".to_string(),
                ));
            };
            let (name, _, element) = resolve_column(source, scopes)?;
            let actual = element.ok_or_else(|| {
                sql_pg_error(format!(
                    "catalog column {name:?} is not a modeled array source"
                ))
            })?;
            if actual != target {
                return Err(sql_pg_error(format!(
                    "catalog array cast changes {name:?} from {actual:?} to {target:?}"
                )));
            }
            Ok(target)
        }
        NodeEnum::AArrayExpr(array) => {
            let mut ty = None;
            for element in &array.elements {
                let candidate =
                    catalog_expression_type(element, catalog, scopes, "catalog array element")?;
                ty = merge_catalog_types(ty, candidate, "catalog array elements")?;
            }
            ty.ok_or_else(|| sql_pg_error("catalog array has no exactly typed element".to_string()))
        }
        NodeEnum::SubLink(link) if link.sub_link_type == SubLinkType::ArraySublink as i32 => {
            array_sublink_element_type(link, catalog, scopes)
        }
        _ => Err(sql_pg_error(
            "catalog expression is not a modeled array source".to_string(),
        )),
    }
}

fn catalog_expression_type(
    node: &Node,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
    context: &str,
) -> Result<Option<SqlType>, ExecuteError> {
    if let Some((_, ty)) = expression_metadata(node, catalog, scopes) {
        return Ok(Some(ty));
    }
    if catalog_null_literal(node) {
        return Ok(None);
    }
    Err(sql_pg_error(format!(
        "cannot derive an exact type for {context}"
    )))
}

fn require_catalog_type(
    node: &Node,
    expected: SqlType,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
    context: &str,
) -> Result<(), ExecuteError> {
    if catalog_expression_type(node, catalog, scopes, context)?
        .is_some_and(|actual| actual != expected)
    {
        return Err(sql_pg_error(format!("{context} must bind as {expected:?}")));
    }
    Ok(())
}

fn catalog_oid_cast_literal(node: &Node) -> bool {
    let Some(NodeEnum::TypeCast(cast)) = node.node.as_ref() else {
        return false;
    };
    let Some(target) = cast.type_name.as_ref() else {
        return false;
    };
    validate_type_name(target)
        .ok()
        .is_some_and(|target| matches!(target.as_str(), "oid" | "regclass"))
        && cast
            .arg
            .as_deref()
            .is_some_and(catalog_numeric_string_literal)
}

fn require_catalog_compatible_operands(
    left: &Node,
    right: &Node,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
    context: &str,
) -> Result<(Option<SqlType>, Option<SqlType>), ExecuteError> {
    let left_type = catalog_expression_type(left, catalog, scopes, context)?;
    let right_type = catalog_expression_type(right, catalog, scopes, context)?;
    let compatible = match (left_type, right_type) {
        (None, None) => false,
        (None, Some(_)) | (Some(_), None) => true,
        (Some(left_type), Some(right_type)) => {
            left_type == right_type
                || (catalog_integer_type(left_type) && catalog_integer_type(right_type))
                || (catalog_integer_type(left_type)
                    && (catalog_numeric_string_literal(right) || catalog_oid_cast_literal(right)))
                || (catalog_integer_type(right_type)
                    && (catalog_numeric_string_literal(left) || catalog_oid_cast_literal(left)))
        }
    };
    if !compatible {
        return Err(sql_pg_error(format!(
            "{context} has incompatible or indeterminate operand types ({left_type:?} and {right_type:?})"
        )));
    }
    Ok((left_type, right_type))
}

fn require_catalog_boolean(
    node: &Node,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
    context: &str,
) -> Result<(), ExecuteError> {
    require_catalog_type(node, SqlType::Bool, catalog, scopes, context)
}

fn validate_expression(
    node: &Node,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
) -> Result<(), ExecuteError> {
    match node_enum(node)? {
        NodeEnum::ColumnRef(column) => {
            resolve_column(column, scopes)?;
        }
        NodeEnum::AConst(_) | NodeEnum::CaseTestExpr(_) => {}
        NodeEnum::AExpr(expression) => {
            let token = aexpr_op_token(expression)?;
            let left = expression
                .lexpr
                .as_deref()
                .ok_or_else(|| sql_pg_error("catalog operator has no left operand".to_string()))?;
            let right = expression
                .rexpr
                .as_deref()
                .ok_or_else(|| sql_pg_error("catalog operator has no right operand".to_string()))?;
            validate_expression(left, catalog, scopes)?;
            if expression.kind == AExprKind::AexprIn as i32 {
                if !matches!(token, "=" | "<>") {
                    return Err(sql_pg_error(
                        "catalog IN has an unsupported comparison operator".to_string(),
                    ));
                }
                let NodeEnum::List(values) = node_enum(right)? else {
                    return Err(sql_pg_error(
                        "catalog IN requires a parenthesized value list".to_string(),
                    ));
                };
                if values.items.is_empty() {
                    return Err(sql_pg_error(
                        "catalog IN requires at least one value".to_string(),
                    ));
                }
                for value in &values.items {
                    validate_expression(value, catalog, scopes)?;
                    require_catalog_compatible_operands(
                        left,
                        value,
                        catalog,
                        scopes,
                        "catalog IN comparison",
                    )?;
                }
                return Ok(());
            }
            if matches!(
                expression.kind,
                value if value == AExprKind::AexprOpAny as i32
                    || value == AExprKind::AexprOpAll as i32
            ) {
                if !matches!(token, "=" | "<>" | "<" | "<=" | ">" | ">=") {
                    return Err(sql_pg_error(
                        "catalog ANY/ALL has an unsupported comparison operator".to_string(),
                    ));
                }
                validate_expression(right, catalog, scopes)?;
                let element = catalog_array_element_type(right, catalog, scopes)?;
                let left_type =
                    catalog_expression_type(left, catalog, scopes, "catalog ANY/ALL left operand")?;
                let compatible = left_type.is_none_or(|left_type| {
                    left_type == element
                        || (catalog_integer_type(left_type) && catalog_integer_type(element))
                        || (catalog_integer_type(element) && catalog_numeric_string_literal(left))
                });
                if !compatible {
                    return Err(sql_pg_error(format!(
                        "catalog ANY/ALL compares {left_type:?} with array element {element:?}"
                    )));
                }
                return Ok(());
            }
            if expression.kind != AExprKind::AexprOp as i32
                && expression.kind != AExprKind::AexprLike as i32
            {
                return Err(sql_pg_error(
                    "catalog expression kind is outside the typed empty-result subset".to_string(),
                ));
            }
            validate_expression(right, catalog, scopes)?;
            match token {
                "+" | "-" | "*" => {
                    let (left_type, right_type) = require_catalog_compatible_operands(
                        left,
                        right,
                        catalog,
                        scopes,
                        "catalog arithmetic operator",
                    )?;
                    if left_type.is_some_and(|ty| !catalog_integer_type(ty))
                        || right_type.is_some_and(|ty| !catalog_integer_type(ty))
                    {
                        return Err(sql_pg_error(
                            "catalog arithmetic requires integer operands".to_string(),
                        ));
                    }
                }
                "=" | "<>" | "<" | "<=" | ">" | ">=" => {
                    require_catalog_compatible_operands(
                        left,
                        right,
                        catalog,
                        scopes,
                        "catalog comparison operator",
                    )?;
                }
                "~" | "!~" | "~~" | "!~~" => {
                    require_catalog_type(
                        left,
                        SqlType::Text,
                        catalog,
                        scopes,
                        "catalog pattern left operand",
                    )?;
                    require_catalog_type(
                        right,
                        SqlType::Text,
                        catalog,
                        scopes,
                        "catalog pattern right operand",
                    )?;
                }
                _ => {
                    return Err(sql_pg_error(format!(
                        "catalog operator {token:?} is outside the typed empty-result subset"
                    )))
                }
            }
        }
        NodeEnum::BoolExpr(expression) => {
            if expression.args.is_empty() {
                return Err(sql_pg_error(
                    "catalog boolean expression is empty".to_string(),
                ));
            }
            let expected_arity = if expression.boolop == BoolExprType::NotExpr as i32 {
                1
            } else if matches!(
                expression.boolop,
                value if value == BoolExprType::AndExpr as i32
                    || value == BoolExprType::OrExpr as i32
            ) {
                2
            } else {
                return Err(sql_pg_error(
                    "catalog boolean expression has an unknown operator".to_string(),
                ));
            };
            if expression.args.len() < expected_arity {
                return Err(sql_pg_error(
                    "catalog boolean expression has too few operands".to_string(),
                ));
            }
            for arg in &expression.args {
                validate_expression(arg, catalog, scopes)?;
                require_catalog_boolean(arg, catalog, scopes, "catalog boolean operand")?;
            }
        }
        NodeEnum::NullTest(test) => validate_expression(
            test.arg
                .as_deref()
                .ok_or_else(|| sql_pg_error("catalog NULL test has no argument".to_string()))?,
            catalog,
            scopes,
        )?,
        NodeEnum::BooleanTest(test) => {
            let argument = test
                .arg
                .as_deref()
                .ok_or_else(|| sql_pg_error("catalog boolean test has no argument".to_string()))?;
            validate_expression(argument, catalog, scopes)?;
            require_catalog_boolean(argument, catalog, scopes, "catalog boolean test operand")?;
        }
        NodeEnum::TypeCast(cast) => {
            let target_name = cast
                .type_name
                .as_ref()
                .ok_or_else(|| sql_pg_error("catalog cast has no target type".to_string()))?;
            let target = validate_type_name(target_name)?;
            let argument = cast
                .arg
                .as_deref()
                .ok_or_else(|| sql_pg_error("catalog cast has no argument".to_string()))?;
            validate_expression(argument, catalog, scopes)?;
            if target_name.array_bounds.is_empty() {
                let source =
                    catalog_expression_type(argument, catalog, scopes, "catalog cast source")?;
                validate_catalog_scalar_cast(&target, argument, source, catalog)?;
            } else {
                catalog_array_element_type(node, catalog, scopes)?;
            }
        }
        NodeEnum::CollateClause(collate) => {
            let name = collate
                .collname
                .iter()
                .map(|part| match node_enum(part)? {
                    NodeEnum::String(part) => Ok(part.sval.to_ascii_lowercase()),
                    _ => Err(sql_pg_error(
                        "catalog collation name is malformed".to_string(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?
                .join(".");
            if !matches!(name.as_str(), "default" | "pg_catalog.default") {
                return Err(sql_pg_error(format!(
                    "catalog collation {name} is outside the empty-result subset"
                )));
            }
            validate_expression(
                collate
                    .arg
                    .as_deref()
                    .ok_or_else(|| sql_pg_error("catalog COLLATE has no argument".to_string()))?,
                catalog,
                scopes,
            )?;
            require_catalog_type(
                collate.arg.as_deref().expect("validated COLLATE source"),
                SqlType::Text,
                catalog,
                scopes,
                "catalog COLLATE source",
            )?;
        }
        NodeEnum::FuncCall(function) => validate_function(function, catalog, scopes)?,
        NodeEnum::SubLink(link) => {
            let NodeEnum::SelectStmt(select) = node_enum(
                link.subselect
                    .as_deref()
                    .ok_or_else(|| sql_pg_error("catalog subquery is missing".to_string()))?,
            )?
            else {
                return Err(sql_pg_error(
                    "catalog subquery must contain SELECT".to_string(),
                ));
            };
            let output = validate_select(select, catalog, scopes)?;
            if matches!(
                link.sub_link_type,
                value if value == SubLinkType::ExprSublink as i32
                    || value == SubLinkType::ArraySublink as i32
            ) {
                if link.testexpr.is_some() || !link.oper_name.is_empty() || output.len() != 1 {
                    return Err(sql_pg_error(
                        "catalog scalar/ARRAY subquery must have no comparison state and return exactly one column"
                            .to_string(),
                    ));
                }
            } else if link.sub_link_type == SubLinkType::ExistsSublink as i32 {
                if link.testexpr.is_some() || !link.oper_name.is_empty() {
                    return Err(sql_pg_error(
                        "catalog EXISTS subquery has unexpected comparison state".to_string(),
                    ));
                }
            } else if matches!(
                link.sub_link_type,
                value if value == SubLinkType::AnySublink as i32
                    || value == SubLinkType::AllSublink as i32
            ) {
                let test = link.testexpr.as_deref().ok_or_else(|| {
                    sql_pg_error("catalog ANY/ALL subquery has no test expression".to_string())
                })?;
                let operator = match link.oper_name.as_slice() {
                    [] => "=",
                    [operator] => {
                        let NodeEnum::String(operator) = node_enum(operator)? else {
                            return Err(sql_pg_error(
                                "catalog ANY/ALL subquery operator is malformed".to_string(),
                            ));
                        };
                        operator.sval.as_str()
                    }
                    _ => {
                        return Err(sql_pg_error(
                            "catalog ANY/ALL subquery has multiple operators".to_string(),
                        ))
                    }
                };
                if !matches!(operator, "=" | "<>") || output.len() != 1 {
                    return Err(sql_pg_error(
                        "catalog ANY/ALL subquery requires equality and one output column"
                            .to_string(),
                    ));
                }
                validate_expression(test, catalog, scopes)?;
                let test_type = catalog_expression_type(
                    test,
                    catalog,
                    scopes,
                    "catalog ANY/ALL test expression",
                )?;
                merge_catalog_types(test_type, output[0].ty, "catalog ANY/ALL comparison")?;
            } else {
                return Err(sql_pg_error(
                    "catalog subquery kind is outside the scalar/ARRAY/EXISTS/ANY subset"
                        .to_string(),
                ));
            }
        }
        NodeEnum::CaseExpr(case) => {
            if let Some(arg) = case.arg.as_deref() {
                validate_expression(arg, catalog, scopes)?;
            }
            let mut result_type = None;
            for arm in &case.args {
                validate_expression(arm, catalog, scopes)?;
                let NodeEnum::CaseWhen(arm) = node_enum(arm)? else {
                    return Err(sql_pg_error("catalog CASE arm is malformed".to_string()));
                };
                let condition = arm
                    .expr
                    .as_deref()
                    .ok_or_else(|| sql_pg_error("catalog CASE arm has no condition".to_string()))?;
                if let Some(source) = case.arg.as_deref() {
                    require_catalog_compatible_operands(
                        source,
                        condition,
                        catalog,
                        scopes,
                        "simple catalog CASE comparison",
                    )?;
                } else {
                    require_catalog_boolean(
                        condition,
                        catalog,
                        scopes,
                        "searched catalog CASE condition",
                    )?;
                }
                let result = arm
                    .result
                    .as_deref()
                    .ok_or_else(|| sql_pg_error("catalog CASE arm has no result".to_string()))?;
                let candidate =
                    catalog_expression_type(result, catalog, scopes, "catalog CASE result")?;
                result_type = merge_catalog_types(result_type, candidate, "catalog CASE result")?;
            }
            if let Some(otherwise) = case.defresult.as_deref() {
                validate_expression(otherwise, catalog, scopes)?;
                let candidate =
                    catalog_expression_type(otherwise, catalog, scopes, "catalog CASE result")?;
                merge_catalog_types(result_type, candidate, "catalog CASE result")?;
            }
        }
        NodeEnum::CaseWhen(arm) => {
            validate_expression(
                arm.expr
                    .as_deref()
                    .ok_or_else(|| sql_pg_error("catalog CASE arm has no condition".to_string()))?,
                catalog,
                scopes,
            )?;
            validate_expression(
                arm.result
                    .as_deref()
                    .ok_or_else(|| sql_pg_error("catalog CASE arm has no result".to_string()))?,
                catalog,
                scopes,
            )?;
        }
        NodeEnum::AArrayExpr(array) => {
            let mut element_type = None;
            for element in &array.elements {
                validate_expression(element, catalog, scopes)?;
                let candidate =
                    catalog_expression_type(element, catalog, scopes, "catalog array element")?;
                if let (Some(expected), Some(candidate)) = (element_type, candidate) {
                    if expected != candidate
                        && !(catalog_integer_type(expected) && catalog_integer_type(candidate))
                    {
                        return Err(sql_pg_error(
                            "catalog array elements have incompatible types".to_string(),
                        ));
                    }
                } else if element_type.is_none() {
                    element_type = candidate;
                }
            }
        }
        NodeEnum::AIndirection(indirection) => {
            validate_expression(
                indirection
                    .arg
                    .as_deref()
                    .ok_or_else(|| sql_pg_error("catalog indirection has no source".to_string()))?,
                catalog,
                scopes,
            )?;
            for item in &indirection.indirection {
                validate_expression(item, catalog, scopes)?;
            }
        }
        NodeEnum::AIndices(indices) => {
            if let Some(lower) = indices.lidx.as_deref() {
                validate_expression(lower, catalog, scopes)?;
                let ty =
                    catalog_expression_type(lower, catalog, scopes, "catalog array lower index")?;
                if ty.is_some_and(|ty| !catalog_integer_type(ty)) {
                    return Err(sql_pg_error(
                        "catalog array lower index must be an integer".to_string(),
                    ));
                }
            }
            if let Some(upper) = indices.uidx.as_deref() {
                validate_expression(upper, catalog, scopes)?;
                let ty =
                    catalog_expression_type(upper, catalog, scopes, "catalog array upper index")?;
                if ty.is_some_and(|ty| !catalog_integer_type(ty)) {
                    return Err(sql_pg_error(
                        "catalog array upper index must be an integer".to_string(),
                    ));
                }
            }
        }
        NodeEnum::List(list) => {
            for item in &list.items {
                validate_expression(item, catalog, scopes)?;
            }
        }
        NodeEnum::SortBy(sort) => validate_expression(
            sort.node
                .as_deref()
                .ok_or_else(|| sql_pg_error("catalog sort has no key".to_string()))?,
            catalog,
            scopes,
        )?,
        _ => {
            return Err(sql_pg_error(
                "catalog empty-result expression is outside the statically bound subset"
                    .to_string(),
            ))
        }
    }
    Ok(())
}

fn catalog_numeric_string_literal(node: &Node) -> bool {
    matches!(
        node.node.as_ref(),
        Some(NodeEnum::AConst(constant))
            if matches!(&constant.val, Some(a_const::Val::Sval(value)) if value.sval.parse::<i64>().is_ok())
    )
}

fn expression_metadata(
    node: &Node,
    catalog: &CatalogSnapshot,
    scopes: &[&EmptyCatalogScope],
) -> Option<(String, SqlType)> {
    match node_enum(node).ok()? {
        NodeEnum::ColumnRef(column) => resolve_column(column, scopes)
            .ok()
            .map(|(name, ty, _)| (name, ty)),
        NodeEnum::AConst(constant) => match &constant.val {
            Some(a_const::Val::Ival(_)) => Some(("?column?".to_string(), SqlType::Int4)),
            Some(a_const::Val::Sval(_)) => Some(("?column?".to_string(), SqlType::Text)),
            Some(a_const::Val::Boolval(_)) => Some(("?column?".to_string(), SqlType::Bool)),
            _ => None,
        },
        NodeEnum::TypeCast(cast) => {
            let type_name = cast.type_name.as_ref()?;
            if !type_name.array_bounds.is_empty() {
                return None;
            }
            let target = validate_type_name(type_name).ok()?;
            let ty = match target.as_str() {
                "bool" => SqlType::Bool,
                "int2" => SqlType::Int2,
                "int4" | "oid" => SqlType::Int4,
                "int8" => SqlType::Int8,
                "regclass" | "regnamespace" | "regtype" => SqlType::Int4,
                "text" => SqlType::Text,
                _ => return None,
            };
            let name = cast
                .arg
                .as_deref()
                .and_then(|arg| expression_metadata(arg, catalog, scopes))
                .map(|(name, _)| name)
                .unwrap_or_else(|| "?column?".to_string());
            Some((name, ty))
        }
        NodeEnum::FuncCall(function) => {
            let name = catalog_function_name(function).ok()?;
            let ty = match name.as_str() {
                "array_upper" | "generate_series" | "pg_partition_ancestors" => SqlType::Int4,
                "count" | "pg_table_size" => SqlType::Int8,
                "pg_function_is_visible"
                | "pg_relation_is_publishable"
                | "pg_table_is_visible"
                | "pg_type_is_visible" => SqlType::Bool,
                _ if function_arity_is_valid(&name, function.args.len()) => SqlType::Text,
                _ => return None,
            };
            Some((name, ty))
        }
        NodeEnum::AExpr(expression) => {
            let token = aexpr_op_token(expression).ok()?;
            if matches!(
                expression.kind,
                value if value == AExprKind::AexprIn as i32
                    || value == AExprKind::AexprLike as i32
                    || value == AExprKind::AexprOpAny as i32
                    || value == AExprKind::AexprOpAll as i32
            ) || matches!(token, "=" | "<>" | "<" | "<=" | ">" | ">=" | "~" | "!~")
            {
                Some(("?column?".to_string(), SqlType::Bool))
            } else if expression.kind == AExprKind::AexprOp as i32
                && matches!(token, "+" | "-" | "*")
            {
                expression
                    .lexpr
                    .as_deref()
                    .and_then(|left| expression_metadata(left, catalog, scopes))
                    .map(|(_, ty)| ("?column?".to_string(), ty))
            } else {
                None
            }
        }
        NodeEnum::BoolExpr(_) | NodeEnum::NullTest(_) | NodeEnum::BooleanTest(_) => {
            Some(("?column?".to_string(), SqlType::Bool))
        }
        NodeEnum::CollateClause(collate) => collate
            .arg
            .as_deref()
            .and_then(|arg| expression_metadata(arg, catalog, scopes)),
        NodeEnum::AIndirection(indirection) => {
            let source = indirection.arg.as_deref()?;
            catalog_array_element_type(source, catalog, scopes)
                .ok()
                .map(|ty| ("?column?".to_string(), ty))
        }
        NodeEnum::CaseExpr(case) => {
            let mut ty = None;
            for result in case
                .args
                .iter()
                .filter_map(|arm| match node_enum(arm).ok()? {
                    NodeEnum::CaseWhen(arm) => arm.result.as_deref(),
                    _ => None,
                })
                .chain(case.defresult.as_deref())
            {
                let Some((_, candidate)) = expression_metadata(result, catalog, scopes) else {
                    continue;
                };
                if ty.is_some_and(|ty| ty != candidate) {
                    return None;
                }
                ty = Some(candidate);
            }
            ty.map(|ty| ("case".to_string(), ty))
        }
        NodeEnum::SubLink(link) => {
            if matches!(
                link.sub_link_type,
                value if value == SubLinkType::ExistsSublink as i32
                    || value == SubLinkType::AnySublink as i32
                    || value == SubLinkType::AllSublink as i32
            ) {
                return Some(("exists".to_string(), SqlType::Bool));
            }
            if link.sub_link_type == SubLinkType::ArraySublink as i32 {
                return None;
            }
            if link.sub_link_type != SubLinkType::ExprSublink as i32 {
                return None;
            }
            let NodeEnum::SelectStmt(select) = node_enum(link.subselect.as_deref()?).ok()? else {
                return None;
            };
            let output = validate_select(select, catalog, scopes).ok()?;
            let [column] = output.as_slice() else {
                return None;
            };
            column.ty.map(|ty| (column.name.clone(), ty))
        }
        _ => None,
    }
}

impl Engine {
    pub(super) fn execute_empty_catalog_select_if_applicable(
        &self,
        stmt: &SelectStmt,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        if !select_tree_has_empty_catalog_source(stmt, &catalog) {
            return Ok(None);
        }
        let columns = validate_select(stmt, &catalog, &[])?;
        if !select_tree_is_proven_empty_catalog(stmt, &catalog) {
            return Ok(None);
        }
        let columns = columns
            .iter()
            .map(|column| {
                column
                    .ty
                    .map(|ty| (column.name.as_str(), ty))
                    .ok_or_else(|| {
                        sql_pg_error(format!(
                            "cannot derive exact metadata for empty catalog column {:?}",
                            column.name
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = catalog_relation_table("pg_catalog", "__empty_catalog_result", &columns);
        let select = Select {
            table: output.name.clone(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::All,
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        self.execute_transient_rows_via_general(&select, output, Vec::new(), boundary)
            .map(Some)
    }
}
