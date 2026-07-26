//! GPU-catalog projection presentation and typed empty-result orchestration.
//!
//! Relational filtering, joining, sorting, and projection remain device operations. This
//! leaf only maps PostgreSQL catalog display metadata around those device results.

use super::*;

mod pg16;
pub(super) use pg16::catalog_relation_device_sizes;
use pg16::{
    catalog_attstattarget_case_is_exact, catalog_join_relation_size_source,
    catalog_reloptions_array_source_is_exact, catalog_relpersistence_case_is_exact,
};

#[derive(Clone)]
pub(super) struct CatalogJoinPresentation {
    output_name: String,
}

pub(super) struct CatalogJoinProjectionPlan {
    pub(super) projection: Vec<JoinProjItem>,
    pub(super) aliases: Vec<Option<String>>,
    pub(super) order_by: Vec<(JoinColRef, bool)>,
    pub(super) order_by_nulls_first: Vec<Option<bool>>,
    pub(super) presentation: Vec<CatalogJoinPresentation>,
}

pub(super) struct CatalogSingleProjectionPlan {
    pub(super) select: Select,
    pub(super) qualifier: String,
    pub(super) presentation: Vec<CatalogJoinPresentation>,
}

pub(super) fn scalar_aggregate_alias_presentation(
    stmt: &SelectStmt,
    select: &Select,
) -> Result<Option<Vec<CatalogJoinPresentation>>, ExecuteError> {
    if !matches!(
        select.projection,
        SelectProjection::CountAll
            | SelectProjection::CountDistinct { .. }
            | SelectProjection::Sum { .. }
            | SelectProjection::Avg { .. }
            | SelectProjection::Min { .. }
            | SelectProjection::Max { .. }
    ) {
        return Ok(None);
    }
    let [target] = stmt.target_list.as_slice() else {
        return Ok(None);
    };
    let NodeEnum::ResTarget(target) = node_enum(target)? else {
        return Err(sql_pg_error(
            "malformed scalar aggregate target".to_string(),
        ));
    };
    Ok((!target.name.is_empty()).then(|| {
        vec![CatalogJoinPresentation {
            output_name: target.name.clone(),
        }]
    }))
}

pub(super) fn catalog_join_projection_plan(
    stmt: &SelectStmt,
) -> Result<CatalogJoinProjectionPlan, ExecuteError> {
    let mut projection = Vec::with_capacity(stmt.target_list.len());
    let mut aliases = Vec::with_capacity(stmt.target_list.len());
    let mut presentation = Vec::with_capacity(stmt.target_list.len());
    for target in &stmt.target_list {
        let NodeEnum::ResTarget(target) = node_enum(target)? else {
            return Err(sql_pg_error("malformed catalog join target".to_string()));
        };
        let value = target
            .val
            .as_deref()
            .ok_or_else(|| sql_pg_error("catalog join target has no value".to_string()))?;
        let source = match node_enum(value)? {
            NodeEnum::ColumnRef(_) => parse_join_col_ref(value)?,
            NodeEnum::CaseExpr(case) => catalog_case_projection_source(case)?,
            NodeEnum::FuncCall(function)
                if catalog_function_name(function)? == "pg_get_userbyid" =>
            {
                let [arg] = function.args.as_slice() else {
                    return Err(sql_pg_error(
                        "pg_get_userbyid requires one catalog column".to_string(),
                    ));
                };
                let mut source = parse_join_col_ref(arg)?;
                if !matches!(
                    source.column.as_str(),
                    "relowner" | "nspowner" | "proowner" | "defaclrole"
                ) {
                    return Err(sql_pg_error(
                        "pg_get_userbyid presentation requires a modeled catalog owner column"
                            .to_string(),
                    ));
                }
                source.column = GPU_CATALOG_OWNER_NAME.to_string();
                source
            }
            NodeEnum::FuncCall(function)
                if catalog_function_name(function)? == "pg_get_function_result" =>
            {
                let [arg] = function.args.as_slice() else {
                    return Err(sql_pg_error(
                        "pg_get_function_result requires one catalog OID column".to_string(),
                    ));
                };
                let mut source = parse_join_col_ref(arg)?;
                if source.column != "oid" {
                    return Err(sql_pg_error(
                        "pg_get_function_result presentation requires pg_proc.oid".to_string(),
                    ));
                }
                source.column = GPU_CATALOG_FUNCTION_RESULT.to_string();
                source
            }
            NodeEnum::FuncCall(function)
                if catalog_function_name(function)? == "pg_get_function_arguments" =>
            {
                let [arg] = function.args.as_slice() else {
                    return Err(sql_pg_error(
                        "pg_get_function_arguments requires one catalog OID column".to_string(),
                    ));
                };
                let mut source = parse_join_col_ref(arg)?;
                if source.column != "oid" {
                    return Err(sql_pg_error(
                        "pg_get_function_arguments presentation requires pg_proc.oid".to_string(),
                    ));
                }
                source.column = GPU_CATALOG_EMPTY_TEXT.to_string();
                source
            }
            NodeEnum::FuncCall(function) if catalog_function_name(function)? == "obj_description" => {
                catalog_obj_description_source(function)?
            }
            NodeEnum::FuncCall(function) if catalog_function_name(function)? == "format_type" => {
                let [source, typmod] = function.args.as_slice() else {
                    return Err(sql_pg_error(
                        "format_type requires an OID column and a typmod column".to_string(),
                    ));
                };
                let mut source = parse_join_col_ref(source)?;
                let typmod = parse_join_col_ref(typmod)?;
                let modeled_pair = matches!(
                    (source.column.as_str(), typmod.column.as_str()),
                    ("typbasetype", "typtypmod") | ("atttypid", "atttypmod")
                );
                if !modeled_pair || source.qualifier != typmod.qualifier {
                    return Err(sql_pg_error(
                        "format_type presentation requires a modeled OID and typmod pair"
                            .to_string(),
                    ));
                }
                source.column = GPU_CATALOG_FORMATTED_TYPE.to_string();
                source
            }
            NodeEnum::FuncCall(function) if catalog_function_name(function)? == "pg_get_expr" => {
                let [expression, relation, rest @ ..] = function.args.as_slice() else {
                    return Err(sql_pg_error(
                        "pg_get_expr requires an expression and relation OID".to_string(),
                    ));
                };
                if rest.len() > 1
                    || rest
                        .first()
                        .is_some_and(|pretty| !catalog_bool_constant_is_exact(pretty, true))
                {
                    return Err(sql_pg_error(
                        "pg_get_expr accepts only an optional true pretty-print argument"
                            .to_string(),
                    ));
                }
                let source = parse_join_col_ref(expression)?;
                let relation = parse_join_col_ref(relation)?;
                if source.column != "adbin"
                    || relation.column != "adrelid"
                    || source.qualifier != relation.qualifier
                {
                    return Err(sql_pg_error(
                        "pg_get_expr catalog presentation requires pg_attrdef adbin/adrelid"
                            .to_string(),
                    ));
                }
                source
            }
            NodeEnum::SubLink(sublink) => {
                catalog_join_sublink_projection_source(sublink).ok_or_else(|| {
                    sql_pg_error(
                        "catalog join scalar subquery has no modeled presentation column"
                            .to_string(),
                    )
                })?
            }
            NodeEnum::FuncCall(function)
                if catalog_function_name(function)? == "array_to_string" =>
            {
                catalog_join_array_to_string_source(function).ok_or_else(|| {
                    sql_pg_error(
                        "catalog array_to_string has no modeled presentation column".to_string(),
                    )
                })?
            }
            NodeEnum::FuncCall(function) if catalog_function_name(function)? == "pg_size_pretty" => {
                catalog_join_relation_size_source(function)?
            }
            NodeEnum::AConst(constant) => {
                let value = catalog_scalar_constant(constant)?;
                let mut source = projection
                    .iter()
                    .find_map(|item| match item {
                        JoinProjItem::Column(column) => Some(column.clone()),
                        JoinProjItem::Star(_) => None,
                    })
                    .ok_or_else(|| {
                        sql_pg_error(
                            "a constant catalog projection requires an earlier source column"
                                .to_string(),
                        )
                    })?;
                source.column = match value {
                    SqlValue::Bool(false) => GPU_CATALOG_FALSE.to_string(),
                    SqlValue::Text(value) if value.is_empty() => GPU_CATALOG_EMPTY_TEXT.to_string(),
                    _ => {
                        return Err(sql_pg_error(
                            "catalog join constant has no device presentation column".to_string(),
                        ))
                    }
                };
                source
            }
            _ => {
                return Err(sql_pg_error(
                    "catalog join presentation supports columns and modeled GPU catalog display expressions"
                        .to_string(),
                ))
            }
        };
        let default_name = match node_enum(value)? {
            NodeEnum::FuncCall(function) => catalog_function_name(function)?,
            NodeEnum::CaseExpr(_) => "case".to_string(),
            NodeEnum::AConst(_) | NodeEnum::SubLink(_) => "?column?".to_string(),
            _ => source.column.clone(),
        };
        let output_name = if target.name.is_empty() {
            default_name
        } else {
            target.name.clone()
        };
        projection.push(JoinProjItem::Column(source));
        aliases.push(Some(output_name.clone()));
        presentation.push(CatalogJoinPresentation { output_name });
    }
    if projection.is_empty() {
        return Err(sql_pg_error(
            "catalog join must project at least one value".to_string(),
        ));
    }

    let mut order_by = Vec::with_capacity(stmt.sort_clause.len());
    let mut order_by_nulls_first = Vec::with_capacity(stmt.sort_clause.len());
    for item in &stmt.sort_clause {
        let NodeEnum::SortBy(sort) = node_enum(item)? else {
            return Err(sql_pg_error("malformed catalog ORDER BY".to_string()));
        };
        let node = sort
            .node
            .as_deref()
            .ok_or_else(|| sql_pg_error("catalog ORDER BY key has no value".to_string()))?;
        let key = match node_enum(node)? {
            NodeEnum::AConst(constant) => {
                let Some(a_const::Val::Ival(position)) = &constant.val else {
                    return Err(sql_pg_error(
                        "catalog ORDER BY position must be an integer".to_string(),
                    ));
                };
                let index = usize::try_from(position.ival)
                    .ok()
                    .and_then(|position| position.checked_sub(1))
                    .filter(|index| *index < projection.len())
                    .ok_or_else(|| {
                        sql_pg_error("catalog ORDER BY position is out of range".to_string())
                    })?;
                let JoinProjItem::Column(column) = &projection[index] else {
                    unreachable!("catalog presentation projects source columns only")
                };
                column.clone()
            }
            NodeEnum::ColumnRef(_) => {
                let requested = parse_join_col_ref(node)?;
                if requested.qualifier.is_none() {
                    if let Some(index) = presentation
                        .iter()
                        .position(|item| item.output_name == requested.column)
                    {
                        let JoinProjItem::Column(column) = &projection[index] else {
                            unreachable!("catalog presentation projects source columns only")
                        };
                        column.clone()
                    } else {
                        requested
                    }
                } else {
                    requested
                }
            }
            _ => {
                return Err(sql_pg_error(
                    "catalog ORDER BY supports a source column, output alias, or position"
                        .to_string(),
                ))
            }
        };
        order_by.push((key, sort.sortby_dir == SortByDir::SortbyDesc as i32));
        order_by_nulls_first.push(
            if sort.sortby_nulls == SortByNulls::SortbyNullsFirst as i32 {
                Some(true)
            } else if sort.sortby_nulls == SortByNulls::SortbyNullsLast as i32 {
                Some(false)
            } else {
                None
            },
        );
    }
    Ok(CatalogJoinProjectionPlan {
        projection,
        aliases,
        order_by,
        order_by_nulls_first,
        presentation,
    })
}

fn catalog_case_projection_source(
    case: &pg_query::protobuf::CaseExpr,
) -> Result<JoinColRef, ExecuteError> {
    if catalog_function_internal_name_case_is_exact(case) {
        return Ok(JoinColRef {
            qualifier: Some("p".to_string()),
            column: GPU_CATALOG_FUNCTION_INTERNAL_NAME.to_string(),
        });
    }
    let CatalogCaseBinding {
        mut source,
        arms,
        otherwise,
        reject_unmatched,
    } = parse_catalog_case(case)?;
    match source.column.as_str() {
        "relkind"
            if !reject_unmatched && otherwise.is_none() && catalog_relkind_case_is_exact(&arms) =>
        {
            source.column = GPU_CATALOG_RELKIND_DISPLAY.to_string();
            Ok(source)
        }
        "relkind"
            if !reject_unmatched
                && otherwise.is_none()
                && catalog_relation_acl_kind_case_is_exact(&arms) =>
        {
            source.column = GPU_CATALOG_RELKIND_ACL_DISPLAY.to_string();
            Ok(source)
        }
        "relpersistence"
            if !reject_unmatched
                && otherwise.is_none()
                && catalog_relpersistence_case_is_exact(&arms) =>
        {
            source.column = GPU_CATALOG_RELPERSISTENCE_DISPLAY.to_string();
            Ok(source)
        }
        "reloftype"
            if reject_unmatched
                && otherwise.is_none()
                && arms == [(SqlValue::Int4(0), SqlValue::Text(String::new()))] =>
        {
            source.column = GPU_CATALOG_RELTYPE_DISPLAY.to_string();
            Ok(source)
        }
        "typnotnull"
            if !reject_unmatched
                && otherwise.is_none()
                && arms == [(SqlValue::Bool(true), SqlValue::Text("not null".to_string()))] =>
        {
            source.column = GPU_CATALOG_NULLABLE_DISPLAY.to_string();
            Ok(source)
        }
        "prokind"
            if !reject_unmatched
                && otherwise == Some(SqlValue::Text("func".to_string()))
                && arms
                    == [
                        (
                            SqlValue::Text("a".to_string()),
                            SqlValue::Text("agg".to_string()),
                        ),
                        (
                            SqlValue::Text("w".to_string()),
                            SqlValue::Text("window".to_string()),
                        ),
                        (
                            SqlValue::Text("p".to_string()),
                            SqlValue::Text("proc".to_string()),
                        ),
                    ] =>
        {
            source.column = GPU_CATALOG_FUNCTION_KIND.to_string();
            Ok(source)
        }
        "provolatile" if catalog_function_volatility_case_is_exact(&arms, &otherwise) => {
            source.column = GPU_CATALOG_FUNCTION_VOLATILITY.to_string();
            Ok(source)
        }
        "proparallel" if catalog_function_parallel_case_is_exact(&arms, &otherwise) => {
            source.column = GPU_CATALOG_FUNCTION_PARALLEL.to_string();
            Ok(source)
        }
        "prosecdef"
            if arms == [(SqlValue::Bool(true), SqlValue::Text("definer".to_string()))]
                && otherwise == Some(SqlValue::Text("invoker".to_string())) =>
        {
            source.column = GPU_CATALOG_FUNCTION_SECURITY.to_string();
            Ok(source)
        }
        "defaclobjtype" if catalog_default_acl_kind_case_is_exact(&arms, &otherwise) => {
            source.column = GPU_CATALOG_DEFAULT_ACL_TYPE.to_string();
            Ok(source)
        }
        _ => Err(sql_pg_error(
            "catalog CASE has no exact device presentation column".to_string(),
        )),
    }
}

fn catalog_relkind_case_is_exact(arms: &[(SqlValue, SqlValue)]) -> bool {
    const EXPECTED: [(&str, &str); 9] = [
        ("r", "table"),
        ("v", "view"),
        ("m", "materialized view"),
        ("i", "index"),
        ("S", "sequence"),
        ("t", "TOAST table"),
        ("f", "foreign table"),
        ("p", "partitioned table"),
        ("I", "partitioned index"),
    ];
    arms.len() == EXPECTED.len()
        && arms
            .iter()
            .zip(EXPECTED)
            .all(|((input, output), expected)| {
                input == &SqlValue::Text(expected.0.to_string())
                    && output == &SqlValue::Text(expected.1.to_string())
            })
}

fn catalog_relation_acl_kind_case_is_exact(arms: &[(SqlValue, SqlValue)]) -> bool {
    const EXPECTED: [(&str, &str); 6] = [
        ("r", "table"),
        ("v", "view"),
        ("m", "materialized view"),
        ("S", "sequence"),
        ("f", "foreign table"),
        ("p", "partitioned table"),
    ];
    arms.len() == EXPECTED.len()
        && arms
            .iter()
            .zip(EXPECTED)
            .all(|((input, output), expected)| {
                input == &SqlValue::Text(expected.0.to_string())
                    && output == &SqlValue::Text(expected.1.to_string())
            })
}

fn catalog_function_volatility_case_is_exact(
    arms: &[(SqlValue, SqlValue)],
    otherwise: &Option<SqlValue>,
) -> bool {
    otherwise.is_none()
        && arms
            == [
                (
                    SqlValue::Text("i".to_string()),
                    SqlValue::Text("immutable".to_string()),
                ),
                (
                    SqlValue::Text("s".to_string()),
                    SqlValue::Text("stable".to_string()),
                ),
                (
                    SqlValue::Text("v".to_string()),
                    SqlValue::Text("volatile".to_string()),
                ),
            ]
}

fn catalog_function_parallel_case_is_exact(
    arms: &[(SqlValue, SqlValue)],
    otherwise: &Option<SqlValue>,
) -> bool {
    otherwise.is_none()
        && arms
            == [
                (
                    SqlValue::Text("r".to_string()),
                    SqlValue::Text("restricted".to_string()),
                ),
                (
                    SqlValue::Text("s".to_string()),
                    SqlValue::Text("safe".to_string()),
                ),
                (
                    SqlValue::Text("u".to_string()),
                    SqlValue::Text("unsafe".to_string()),
                ),
            ]
}

fn catalog_default_acl_kind_case_is_exact(
    arms: &[(SqlValue, SqlValue)],
    otherwise: &Option<SqlValue>,
) -> bool {
    const EXPECTED: [(&str, &str); 5] = [
        ("r", "table"),
        ("S", "sequence"),
        ("f", "function"),
        ("T", "type"),
        ("n", "schema"),
    ];
    otherwise.is_none()
        && arms.len() == EXPECTED.len()
        && arms
            .iter()
            .zip(EXPECTED)
            .all(|((input, output), expected)| {
                input == &SqlValue::Text(expected.0.to_string())
                    && output == &SqlValue::Text(expected.1.to_string())
            })
}

fn catalog_function_internal_name_case_is_exact(case: &pg_query::protobuf::CaseExpr) -> bool {
    if case.arg.is_some() || case.args.len() != 1 || case.defresult.is_some() {
        return false;
    }
    let Ok(NodeEnum::CaseWhen(arm)) = node_enum(&case.args[0]) else {
        return false;
    };
    let Some(condition) = arm.expr.as_deref() else {
        return false;
    };
    let Ok(NodeEnum::AExpr(expression)) = node_enum(condition) else {
        return false;
    };
    if expression.kind != AExprKind::AexprIn as i32
        || expression
            .lexpr
            .as_deref()
            .is_none_or(|left| !catalog_column_is_exact(left, "l", "lanname"))
    {
        return false;
    }
    let Some(right) = expression.rexpr.as_deref() else {
        return false;
    };
    let Ok(NodeEnum::List(values)) = node_enum(right) else {
        return false;
    };
    let expected = ["internal", "c"];
    if values.items.len() != expected.len()
        || !values.items.iter().zip(expected).all(|(value, expected)| {
            matches!(
                node_enum(value),
                Ok(NodeEnum::AConst(constant))
                    if matches!(&constant.val, Some(a_const::Val::Sval(value)) if value.sval == expected)
            )
        })
    {
        return false;
    }
    arm.result
        .as_deref()
        .is_some_and(|result| catalog_column_is_exact(result, "p", "prosrc"))
}

pub(super) fn catalog_single_projection_plan(
    stmt: &SelectStmt,
) -> Result<CatalogSingleProjectionPlan, ExecuteError> {
    if !stmt.group_clause.is_empty()
        || stmt.having_clause.is_some()
        || !stmt.distinct_clause.is_empty()
        || stmt.with_clause.is_some()
    {
        return Err(sql_pg_error(
            "catalog single-relation presentation does not support grouped, distinct, or CTE input"
                .to_string(),
        ));
    }
    let [from] = stmt.from_clause.as_slice() else {
        return Err(sql_pg_error(
            "catalog presentation requires one base relation".to_string(),
        ));
    };
    let NodeEnum::RangeVar(range) = node_enum(from)? else {
        return Err(sql_pg_error(
            "catalog presentation requires one base relation".to_string(),
        ));
    };
    let table = catalog_range_relation_key(range)?;
    let qualifier = range
        .alias
        .as_ref()
        .map(|alias| alias.aliasname.clone())
        .unwrap_or_else(|| range.relname.clone());
    let mut physical_columns = Vec::with_capacity(stmt.target_list.len());
    let mut presentation = Vec::with_capacity(stmt.target_list.len());
    for target in &stmt.target_list {
        let NodeEnum::ResTarget(target) = node_enum(target)? else {
            return Err(sql_pg_error("malformed catalog SELECT target".to_string()));
        };
        let value = target
            .val
            .as_deref()
            .ok_or_else(|| sql_pg_error("catalog SELECT target has no value".to_string()))?;
        let (source, default_name) = match node_enum(value)? {
            NodeEnum::ColumnRef(column) => {
                let source = resolve_column_name(column, &qualifier)?.to_string();
                (source.clone(), source)
            }
            NodeEnum::FuncCall(function) if catalog_function_name(function)? == "format_type" => {
                let [source, typmod] = function.args.as_slice() else {
                    return Err(sql_pg_error(
                        "format_type requires an OID column and a typmod column".to_string(),
                    ));
                };
                let NodeEnum::ColumnRef(column) = node_enum(source)? else {
                    return Err(sql_pg_error(
                        "format_type requires an OID column as its first argument".to_string(),
                    ));
                };
                if resolve_column_name(column, &qualifier)? != "atttypid" {
                    return Err(sql_pg_error(
                        "format_type presentation requires atttypid".to_string(),
                    ));
                }
                let NodeEnum::ColumnRef(column) = node_enum(typmod)? else {
                    return Err(sql_pg_error(
                        "format_type requires a typmod column as its second argument".to_string(),
                    ));
                };
                if resolve_column_name(column, &qualifier)? != "atttypmod" {
                    return Err(sql_pg_error(
                        "format_type presentation requires atttypmod".to_string(),
                    ));
                }
                (
                    GPU_CATALOG_FORMATTED_TYPE.to_string(),
                    "format_type".to_string(),
                )
            }
            NodeEnum::FuncCall(function)
                if catalog_function_name(function)? == "col_description" =>
            {
                pg16::catalog_single_col_description_source(
                    function,
                    &table,
                    &qualifier,
                )?
            }
            NodeEnum::FuncCall(function)
                if catalog_function_name(function)? == "pg_get_userbyid" =>
            {
                let [source] = function.args.as_slice() else {
                    return Err(sql_pg_error(
                        "pg_get_userbyid requires one catalog owner column".to_string(),
                    ));
                };
                let NodeEnum::ColumnRef(source) = node_enum(source)? else {
                    return Err(sql_pg_error(
                        "pg_get_userbyid requires a catalog owner column".to_string(),
                    ));
                };
                if !matches!(
                    resolve_column_name(source, &qualifier)?,
                    "relowner" | "nspowner" | "proowner" | "defaclrole" | "spcowner"
                ) {
                    return Err(sql_pg_error(
                        "pg_get_userbyid requires a modeled catalog owner column".to_string(),
                    ));
                }
                (
                    GPU_CATALOG_OWNER_NAME.to_string(),
                    "pg_get_userbyid".to_string(),
                )
            }
            NodeEnum::FuncCall(function) if catalog_function_name(function)? == "array_to_string" => {
                let source = catalog_single_array_to_string_source(function, &qualifier)
                    .ok_or_else(|| {
                        sql_pg_error(
                            "catalog array_to_string has no modeled presentation column"
                                .to_string(),
                        )
                    })?;
                (source, "array_to_string".to_string())
            }
            NodeEnum::FuncCall(function) if catalog_function_name(function)? == "obj_description" => {
                let source = catalog_obj_description_source(function)?;
                if source.qualifier.as_deref() != Some(qualifier.as_str()) {
                    return Err(sql_pg_error(
                        "obj_description source does not belong to the catalog relation"
                            .to_string(),
                    ));
                }
                (GPU_CATALOG_DESCRIPTION.to_string(), "obj_description".to_string())
            }
            NodeEnum::FuncCall(function)
                if catalog_function_name(function)? == "shobj_description" =>
            {
                catalog_single_shobj_description_source(function, &table, &qualifier)?
            }
            NodeEnum::FuncCall(function)
                if catalog_function_name(function)? == "pg_tablespace_location" =>
            {
                catalog_single_tablespace_oid_function_source(
                    function,
                    &table,
                    &qualifier,
                    GPU_CATALOG_TABLESPACE_LOCATION,
                )?
            }
            NodeEnum::FuncCall(function) if catalog_function_name(function)? == "pg_size_pretty" => {
                catalog_single_tablespace_size_source(function, &table, &qualifier)?
            }
            NodeEnum::FuncCall(function)
                if catalog_function_name(function)? == "pg_get_function_result" =>
            {
                let [source] = function.args.as_slice() else {
                    return Err(sql_pg_error(
                        "pg_get_function_result requires one catalog OID column".to_string(),
                    ));
                };
                let NodeEnum::ColumnRef(source) = node_enum(source)? else {
                    return Err(sql_pg_error(
                        "pg_get_function_result requires a catalog OID column".to_string(),
                    ));
                };
                if resolve_column_name(source, &qualifier)? != "oid" {
                    return Err(sql_pg_error(
                        "pg_get_function_result presentation requires pg_proc.oid".to_string(),
                    ));
                }
                (
                    GPU_CATALOG_FUNCTION_RESULT.to_string(),
                    "pg_get_function_result".to_string(),
                )
            }
            NodeEnum::FuncCall(function) if catalog_function_name(function)? == "acldefault" => {
                let [kind, owner] = function.args.as_slice() else {
                    return Err(sql_pg_error(
                        "acldefault requires an object kind and owner column".to_string(),
                    ));
                };
                let NodeEnum::AConst(kind) = node_enum(kind)? else {
                    return Err(sql_pg_error(
                        "acldefault object kind must be a text literal".to_string(),
                    ));
                };
                let Some(a_const::Val::Sval(kind)) = &kind.val else {
                    return Err(sql_pg_error(
                        "acldefault object kind must be a text literal".to_string(),
                    ));
                };
                let NodeEnum::ColumnRef(owner) = node_enum(owner)? else {
                    return Err(sql_pg_error(
                        "acldefault owner must be a catalog column".to_string(),
                    ));
                };
                let owner = resolve_column_name(owner, &qualifier)?;
                let supported = (table.ends_with("pg_namespace")
                    && kind.sval == "n"
                    && owner == "nspowner")
                    || (table.ends_with("pg_tablespace")
                        && kind.sval == "t"
                        && owner == "spcowner");
                if !supported {
                    return Err(sql_pg_error(
                        "catalog acldefault requires a modeled object kind and owner column"
                            .to_string(),
                    ));
                }
                (
                    GPU_CATALOG_ACL_DEFAULT.to_string(),
                    "acldefault".to_string(),
                )
            }
            NodeEnum::FuncCall(function)
                if catalog_function_name(function)? == "pg_get_constraintdef" =>
            {
                let source = match function.args.as_slice() {
                    [source] => source,
                    [source, pretty] => {
                        let NodeEnum::AConst(pretty) = node_enum(pretty)? else {
                            return Err(sql_pg_error(
                                "pg_get_constraintdef pretty-print argument must be true"
                                    .to_string(),
                            ));
                        };
                        if !matches!(pretty.val, Some(a_const::Val::Boolval(ref value)) if value.boolval)
                        {
                            return Err(sql_pg_error(
                                "pg_get_constraintdef pretty-print argument must be true"
                                    .to_string(),
                            ));
                        }
                        source
                    }
                    _ => {
                        return Err(sql_pg_error(
                            "pg_get_constraintdef requires one OID and optional true argument"
                                .to_string(),
                        ))
                    }
                };
                let NodeEnum::ColumnRef(source) = node_enum(source)? else {
                    return Err(sql_pg_error(
                        "pg_get_constraintdef requires the constraint OID column".to_string(),
                    ));
                };
                if resolve_column_name(source, &qualifier)? != "oid" {
                    return Err(sql_pg_error(
                        "pg_get_constraintdef requires the constraint OID column".to_string(),
                    ));
                }
                (
                    GPU_CATALOG_CONSTRAINT_DEF.to_string(),
                    "pg_get_constraintdef".to_string(),
                )
            }
            NodeEnum::SubLink(sublink) => {
                let source = catalog_sublink_guard(sublink, &qualifier).ok_or_else(|| {
                    sql_pg_error(
                        "catalog scalar subquery has no modeled empty-result guard".to_string(),
                    )
                })?;
                (source, "?column?".to_string())
            }
            NodeEnum::AConst(constant) => {
                let constant = catalog_scalar_constant(constant)?;
                let source = match constant {
                    SqlValue::Bool(false) => GPU_CATALOG_FALSE.to_string(),
                    SqlValue::Text(value) if value.is_empty() => {
                        GPU_CATALOG_EMPTY_TEXT.to_string()
                    }
                    _ => {
                        return Err(sql_pg_error(
                            "catalog constant has no device presentation column".to_string(),
                        ))
                    }
                };
                (source, "?column?".to_string())
            }
            NodeEnum::AExpr(expression)
                if catalog_current_user_comparison_is_exact(expression, &qualifier) =>
            {
                (
                    GPU_CATALOG_CURRENT_USER_MATCH.to_string(),
                    "?column?".to_string(),
                )
            }
            NodeEnum::CaseExpr(case)
                if catalog_attstattarget_case_is_exact(case, &qualifier) =>
            {
                (
                    GPU_CATALOG_NULL_INT4.to_string(),
                    "case".to_string(),
                )
            }
            _ => {
                return Err(sql_pg_error(
                    "catalog projection supports columns, format_type, scalar metadata subqueries, and constants"
                        .to_string(),
                ))
            }
        };
        physical_columns.push(source);
        presentation.push(CatalogJoinPresentation {
            output_name: if target.name.is_empty() {
                default_name
            } else {
                target.name.clone()
            },
        });
    }
    let order_by = catalog_single_order_by(
        &stmt.sort_clause,
        &qualifier,
        &physical_columns,
        &presentation,
    )?;
    Ok(CatalogSingleProjectionPlan {
        select: Select {
            table,
            public_only: range.schemaname == "public",
            distinct: false,
            projection: SelectProjection::Columns(physical_columns),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by,
            limit: parse_limit(&stmt.limit_count)?,
            offset: parse_limit(&stmt.limit_offset)?,
        },
        qualifier,
        presentation,
    })
}

fn catalog_single_order_by(
    sort_clause: &[Node],
    qualifier: &str,
    physical_columns: &[String],
    presentation: &[CatalogJoinPresentation],
) -> Result<Vec<SelectOrder>, ExecuteError> {
    let mut order_by = Vec::with_capacity(sort_clause.len());
    for item in sort_clause {
        let NodeEnum::SortBy(sort) = node_enum(item)? else {
            return Err(sql_pg_error("malformed catalog ORDER BY".to_string()));
        };
        let node = sort
            .node
            .as_deref()
            .ok_or_else(|| sql_pg_error("catalog ORDER BY key has no value".to_string()))?;
        let column = match node_enum(node)? {
            NodeEnum::AConst(constant) => {
                let Some(a_const::Val::Ival(position)) = &constant.val else {
                    return Err(sql_pg_error(
                        "catalog ORDER BY position must be an integer".to_string(),
                    ));
                };
                usize::try_from(position.ival)
                    .ok()
                    .and_then(|position| position.checked_sub(1))
                    .and_then(|index| physical_columns.get(index))
                    .cloned()
                    .ok_or_else(|| {
                        sql_pg_error("catalog ORDER BY position is out of range".to_string())
                    })?
            }
            NodeEnum::ColumnRef(reference) => {
                let requested = resolve_column_name(reference, qualifier)?;
                presentation
                    .iter()
                    .position(|item| item.output_name == requested)
                    .and_then(|index| physical_columns.get(index))
                    .cloned()
                    .unwrap_or_else(|| requested.to_string())
            }
            _ => {
                return Err(sql_pg_error(
                    "catalog ORDER BY supports a source column, output alias, or position"
                        .to_string(),
                ))
            }
        };
        order_by.push(SelectOrder {
            column,
            descending: sort.sortby_dir == SortByDir::SortbyDesc as i32,
        });
    }
    Ok(order_by)
}

fn catalog_sublink_guard(sublink: &pg_query::protobuf::SubLink, qualifier: &str) -> Option<String> {
    if sublink.sub_link_type != SubLinkType::ExprSublink as i32
        || sublink.testexpr.is_some()
        || !sublink.oper_name.is_empty()
    {
        return None;
    }
    let NodeEnum::SelectStmt(select) = node_enum(sublink.subselect.as_deref()?).ok()? else {
        return None;
    };
    if catalog_default_sublink_is_exact(select, qualifier) {
        return Some(GPU_CATALOG_DEFAULT_EXPR.to_string());
    }
    if catalog_collation_sublink_is_exact(select, qualifier) {
        return Some(GPU_CATALOG_COLLATION_NAME.to_string());
    }
    None
}

fn catalog_scalar_subselect_is_plain(select: &SelectStmt) -> bool {
    select.op == SetOperation::SetopNone as i32
        && !select.all
        && select.larg.is_none()
        && select.rarg.is_none()
        && select.distinct_clause.is_empty()
        && select.into_clause.is_none()
        && select.group_clause.is_empty()
        && !select.group_distinct
        && select.having_clause.is_none()
        && select.window_clause.is_empty()
        && select.values_lists.is_empty()
        && select.sort_clause.is_empty()
        && select.limit_offset.is_none()
        && select.limit_count.is_none()
        && select.locking_clause.is_empty()
        && select.with_clause.is_none()
}

fn catalog_range_is_exact(node: &Node, relation: &str, alias: &str) -> bool {
    let Ok(NodeEnum::RangeVar(range)) = node_enum(node) else {
        return false;
    };
    range.catalogname.is_empty()
        && range.schemaname == "pg_catalog"
        && range.relname == relation
        && range
            .alias
            .as_ref()
            .is_some_and(|actual| actual.aliasname == alias && actual.colnames.is_empty())
}

fn catalog_column_is_exact(node: &Node, qualifier: &str, column: &str) -> bool {
    let Ok(NodeEnum::ColumnRef(reference)) = node_enum(node) else {
        return false;
    };
    let [table, name] = reference.fields.as_slice() else {
        return false;
    };
    matches!(
        (table.node.as_ref(), name.node.as_ref()),
        (Some(NodeEnum::String(table)), Some(NodeEnum::String(name)))
            if table.sval == qualifier && name.sval == column
    )
}

fn catalog_bool_constant_is_exact(node: &Node, expected: bool) -> bool {
    matches!(
        node.node.as_ref(),
        Some(NodeEnum::AConst(constant))
            if matches!(&constant.val, Some(a_const::Val::Boolval(value)) if value.boolval == expected)
    )
}

fn catalog_target_value(select: &SelectStmt) -> Option<&Node> {
    let [target] = select.target_list.as_slice() else {
        return None;
    };
    let NodeEnum::ResTarget(target) = node_enum(target).ok()? else {
        return None;
    };
    if !target.name.is_empty() || !target.indirection.is_empty() {
        return None;
    }
    target.val.as_deref()
}

fn collect_catalog_and_terms<'a>(node: &'a Node, terms: &mut Vec<&'a Node>) {
    if let Some(NodeEnum::BoolExpr(expression)) = node.node.as_ref() {
        if expression.boolop == BoolExprType::AndExpr as i32 {
            for arg in &expression.args {
                collect_catalog_and_terms(arg, terms);
            }
            return;
        }
    }
    terms.push(node);
}

fn catalog_column_comparison_is_exact(
    node: &Node,
    token: &str,
    left: (&str, &str),
    right: (&str, &str),
) -> bool {
    let Ok(NodeEnum::AExpr(expression)) = node_enum(node) else {
        return false;
    };
    expression.kind == AExprKind::AexprOp as i32
        && aexpr_op_token(expression).ok() == Some(token)
        && expression
            .lexpr
            .as_deref()
            .is_some_and(|node| catalog_column_is_exact(node, left.0, left.1))
        && expression
            .rexpr
            .as_deref()
            .is_some_and(|node| catalog_column_is_exact(node, right.0, right.1))
}

fn catalog_current_user_comparison_is_exact(
    expression: &pg_query::protobuf::AExpr,
    qualifier: &str,
) -> bool {
    expression.kind == AExprKind::AexprOp as i32
        && aexpr_op_token(expression).ok() == Some("=")
        && expression.lexpr.as_deref().is_some_and(|node| {
            matches!(
                node_enum(node),
                Ok(NodeEnum::ColumnRef(column))
                    if resolve_column_name(column, qualifier).ok() == Some("rolname")
            )
        })
        && matches!(
            expression.rexpr.as_deref().and_then(|node| node.node.as_ref()),
            Some(NodeEnum::SqlvalueFunction(function))
                if function.op == pg_query::protobuf::SqlValueFunctionOp::SvfopCurrentUser as i32
                    && function.xpr.is_none()
        )
}

fn catalog_default_sublink_is_exact(select: &SelectStmt, outer: &str) -> bool {
    if !catalog_scalar_subselect_is_plain(select)
        || select.from_clause.len() != 1
        || !catalog_range_is_exact(&select.from_clause[0], "pg_attrdef", "d")
    {
        return false;
    }
    let Some(target) = catalog_target_value(select) else {
        return false;
    };
    let Ok(NodeEnum::FuncCall(function)) = node_enum(target) else {
        return false;
    };
    if catalog_function_name(function).ok().as_deref() != Some("pg_get_expr")
        || function.args.len() != 3
        || !catalog_column_is_exact(&function.args[0], "d", "adbin")
        || !catalog_column_is_exact(&function.args[1], "d", "adrelid")
        || !catalog_bool_constant_is_exact(&function.args[2], true)
    {
        return false;
    }
    let Some(predicate) = select.where_clause.as_deref() else {
        return false;
    };
    let mut terms = Vec::new();
    collect_catalog_and_terms(predicate, &mut terms);
    terms.len() == 3
        && terms.iter().any(|term| {
            catalog_column_comparison_is_exact(term, "=", ("d", "adrelid"), (outer, "attrelid"))
        })
        && terms.iter().any(|term| {
            catalog_column_comparison_is_exact(term, "=", ("d", "adnum"), (outer, "attnum"))
        })
        && terms
            .iter()
            .any(|term| catalog_column_is_exact(term, outer, "atthasdef"))
}

fn catalog_collation_sublink_is_exact(select: &SelectStmt, outer: &str) -> bool {
    if !catalog_scalar_subselect_is_plain(select)
        || select.from_clause.len() != 2
        || !catalog_range_is_exact(&select.from_clause[0], "pg_collation", "c")
        || !catalog_range_is_exact(&select.from_clause[1], "pg_type", "t")
        || !catalog_target_value(select)
            .is_some_and(|target| catalog_column_is_exact(target, "c", "collname"))
    {
        return false;
    }
    let Some(predicate) = select.where_clause.as_deref() else {
        return false;
    };
    let mut terms = Vec::new();
    collect_catalog_and_terms(predicate, &mut terms);
    terms.len() == 3
        && terms.iter().any(|term| {
            catalog_column_comparison_is_exact(term, "=", ("c", "oid"), (outer, "attcollation"))
        })
        && terms.iter().any(|term| {
            catalog_column_comparison_is_exact(term, "=", ("t", "oid"), (outer, "atttypid"))
        })
        && terms.iter().any(|term| {
            catalog_column_comparison_is_exact(
                term,
                "<>",
                (outer, "attcollation"),
                ("t", "typcollation"),
            )
        })
}

fn catalog_join_sublink_projection_source(
    sublink: &pg_query::protobuf::SubLink,
) -> Option<JoinColRef> {
    if sublink.sub_link_type != SubLinkType::ExprSublink as i32
        || sublink.testexpr.is_some()
        || !sublink.oper_name.is_empty()
    {
        return None;
    }
    let NodeEnum::SelectStmt(select) = node_enum(sublink.subselect.as_deref()?).ok()? else {
        return None;
    };
    catalog_domain_collation_sublink_is_exact(select, "t").then_some(JoinColRef {
        qualifier: Some("t".to_string()),
        column: GPU_CATALOG_COLLATION_NAME.to_string(),
    })
}

fn catalog_domain_collation_sublink_is_exact(select: &SelectStmt, outer: &str) -> bool {
    if !catalog_scalar_subselect_is_plain(select)
        || select.from_clause.len() != 2
        || !catalog_range_is_exact(&select.from_clause[0], "pg_collation", "c")
        || !catalog_range_is_exact(&select.from_clause[1], "pg_type", "bt")
        || !catalog_target_value(select)
            .is_some_and(|target| catalog_column_is_exact(target, "c", "collname"))
    {
        return false;
    }
    let Some(predicate) = select.where_clause.as_deref() else {
        return false;
    };
    let mut terms = Vec::new();
    collect_catalog_and_terms(predicate, &mut terms);
    terms.len() == 3
        && terms.iter().any(|term| {
            catalog_column_comparison_is_exact(term, "=", ("c", "oid"), (outer, "typcollation"))
        })
        && terms.iter().any(|term| {
            catalog_column_comparison_is_exact(term, "=", ("bt", "oid"), (outer, "typbasetype"))
        })
        && terms.iter().any(|term| {
            catalog_column_comparison_is_exact(
                term,
                "<>",
                (outer, "typcollation"),
                ("bt", "typcollation"),
            )
        })
}

fn catalog_join_array_to_string_source(
    function: &pg_query::protobuf::FuncCall,
) -> Option<JoinColRef> {
    let [source, separator] = function.args.as_slice() else {
        return None;
    };
    let NodeEnum::AConst(separator) = node_enum(separator).ok()? else {
        return None;
    };
    let Some(a_const::Val::Sval(separator)) = &separator.val else {
        return None;
    };
    if let Ok(NodeEnum::ColumnRef(_)) = node_enum(source) {
        let mut source = parse_join_col_ref(source).ok()?;
        if matches!(
            source.column.as_str(),
            "typacl" | "nspacl" | "relacl" | "proacl" | "defaclacl" | "spcacl"
        ) && separator.sval == "\n"
        {
            source.column = GPU_CATALOG_ACL_DISPLAY.to_string();
            return Some(source);
        }
        return None;
    }
    if separator.sval == ", " && catalog_reloptions_array_source_is_exact(source) {
        return Some(JoinColRef {
            qualifier: Some("c".to_string()),
            column: GPU_CATALOG_EMPTY_TEXT.to_string(),
        });
    }
    let Ok(NodeEnum::SubLink(sublink)) = node_enum(source) else {
        return None;
    };
    let NodeEnum::SelectStmt(select) = node_enum(sublink.subselect.as_deref()?).ok()? else {
        return None;
    };
    if separator.sval == " " && catalog_domain_check_sublink_is_exact(sublink, select, "t") {
        return Some(JoinColRef {
            qualifier: Some("t".to_string()),
            column: GPU_CATALOG_DOMAIN_CHECK.to_string(),
        });
    }
    catalog_relation_acl_sublink_source(sublink, select)
}

fn catalog_single_array_to_string_source(
    function: &pg_query::protobuf::FuncCall,
    qualifier: &str,
) -> Option<String> {
    let [source, separator] = function.args.as_slice() else {
        return None;
    };
    let NodeEnum::ColumnRef(source) = node_enum(source).ok()? else {
        return None;
    };
    let NodeEnum::AConst(separator) = node_enum(separator).ok()? else {
        return None;
    };
    let Some(a_const::Val::Sval(separator)) = &separator.val else {
        return None;
    };
    let source = resolve_column_name(source, qualifier).ok()?;
    if source == "spcoptions" && separator.sval == ", " {
        return Some("spcoptions".to_string());
    }
    (separator.sval == "\n"
        && matches!(
            source,
            "typacl" | "nspacl" | "relacl" | "proacl" | "defaclacl" | "spcacl"
        ))
    .then(|| GPU_CATALOG_ACL_DISPLAY.to_string())
}

fn catalog_obj_description_source(
    function: &pg_query::protobuf::FuncCall,
) -> Result<JoinColRef, ExecuteError> {
    let [source, class] = function.args.as_slice() else {
        return Err(sql_pg_error(
            "obj_description requires an OID column and catalog class".to_string(),
        ));
    };
    let mut source = parse_join_col_ref(source)?;
    let NodeEnum::AConst(class) = node_enum(class)? else {
        return Err(sql_pg_error(
            "obj_description catalog class must be text".to_string(),
        ));
    };
    let Some(a_const::Val::Sval(class)) = &class.val else {
        return Err(sql_pg_error(
            "obj_description catalog class must be text".to_string(),
        ));
    };
    if source.column != "oid"
        || !matches!(class.sval.as_str(), "pg_class" | "pg_namespace" | "pg_proc")
    {
        return Err(sql_pg_error(
            "obj_description requires a modeled catalog OID".to_string(),
        ));
    }
    source.column = GPU_CATALOG_DESCRIPTION.to_string();
    Ok(source)
}

fn catalog_single_shobj_description_source(
    function: &pg_query::protobuf::FuncCall,
    table: &str,
    qualifier: &str,
) -> Result<(String, String), ExecuteError> {
    let [source, class] = function.args.as_slice() else {
        return Err(sql_pg_error(
            "shobj_description requires an OID column and shared catalog class".to_string(),
        ));
    };
    let NodeEnum::ColumnRef(source) = node_enum(source)? else {
        return Err(sql_pg_error(
            "shobj_description requires a catalog OID column".to_string(),
        ));
    };
    let NodeEnum::AConst(class) = node_enum(class)? else {
        return Err(sql_pg_error(
            "shobj_description shared catalog class must be text".to_string(),
        ));
    };
    let Some(a_const::Val::Sval(class)) = &class.val else {
        return Err(sql_pg_error(
            "shobj_description shared catalog class must be text".to_string(),
        ));
    };
    let modeled = resolve_column_name(source, qualifier)? == "oid"
        && ((table.ends_with("pg_roles") && class.sval == "pg_authid")
            || (table.ends_with("pg_tablespace") && class.sval == "pg_tablespace"));
    if !modeled {
        return Err(sql_pg_error(
            "shobj_description requires a modeled shared catalog OID".to_string(),
        ));
    }
    Ok((
        GPU_CATALOG_DESCRIPTION.to_string(),
        "shobj_description".to_string(),
    ))
}

fn catalog_single_tablespace_oid_function_source(
    function: &pg_query::protobuf::FuncCall,
    table: &str,
    qualifier: &str,
    physical_column: &str,
) -> Result<(String, String), ExecuteError> {
    let [source] = function.args.as_slice() else {
        return Err(sql_pg_error(
            "tablespace catalog function requires one OID column".to_string(),
        ));
    };
    let NodeEnum::ColumnRef(source) = node_enum(source)? else {
        return Err(sql_pg_error(
            "tablespace catalog function requires an OID column".to_string(),
        ));
    };
    if !table.ends_with("pg_tablespace") || resolve_column_name(source, qualifier)? != "oid" {
        return Err(sql_pg_error(
            "tablespace catalog function requires pg_tablespace.oid".to_string(),
        ));
    }
    Ok((
        physical_column.to_string(),
        "pg_tablespace_location".to_string(),
    ))
}

fn catalog_single_tablespace_size_source(
    function: &pg_query::protobuf::FuncCall,
    table: &str,
    qualifier: &str,
) -> Result<(String, String), ExecuteError> {
    let [inner] = function.args.as_slice() else {
        return Err(sql_pg_error(
            "pg_size_pretty tablespace presentation requires one size expression".to_string(),
        ));
    };
    let NodeEnum::FuncCall(inner) = node_enum(inner)? else {
        return Err(sql_pg_error(
            "pg_size_pretty tablespace presentation requires pg_tablespace_size".to_string(),
        ));
    };
    if catalog_function_name(inner)? != "pg_tablespace_size" {
        return Err(sql_pg_error(
            "pg_size_pretty tablespace presentation requires pg_tablespace_size".to_string(),
        ));
    }
    let (source, _) = catalog_single_tablespace_oid_function_source(
        inner,
        table,
        qualifier,
        GPU_CATALOG_TABLESPACE_SIZE,
    )?;
    Ok((source, "pg_size_pretty".to_string()))
}

fn catalog_relation_acl_sublink_source(
    sublink: &pg_query::protobuf::SubLink,
    select: &SelectStmt,
) -> Option<JoinColRef> {
    if sublink.sub_link_type != SubLinkType::ArraySublink as i32
        || sublink.testexpr.is_some()
        || !sublink.oper_name.is_empty()
        || !catalog_scalar_subselect_is_plain(select)
    {
        return None;
    }
    let program = NodeEnum::SelectStmt(Box::new(select.clone()))
        .deparse()
        .ok()
        .and_then(|sql| canonicalize_sql_for_exact_match(&sql).ok())?;
    let column = match program.as_str() {
        "select (attname || ':\n  ') || pg_catalog.array_to_string(attacl, '\n  ') from pg_catalog.pg_attribute a where attrelid = c.oid and not attisdropped and attacl is not null" => GPU_CATALOG_COLUMN_PRIVILEGES,
        "select ((((polname || case when not polpermissive then ' (RESTRICTIVE)' else '' end) || case when polcmd <> '*' then (' (' || polcmd::pg_catalog.text) || '):' else ':' end) || case when polqual is not null then '\n  (u): ' || pg_catalog.pg_get_expr(polqual, polrelid) else '' end) || case when polwithcheck is not null then '\n  (c): ' || pg_catalog.pg_get_expr(polwithcheck, polrelid) else '' end) || case when polroles <> '{0}' then '\n  to: ' || pg_catalog.array_to_string(array(select rolname from pg_catalog.pg_roles where oid = any(polroles) order by 1), ', ') else '' end from pg_catalog.pg_policy pol where polrelid = c.oid" => GPU_CATALOG_POLICY_DISPLAY,
        _ => return None,
    };
    Some(JoinColRef {
        qualifier: Some("c".to_string()),
        column: column.to_string(),
    })
}

fn catalog_domain_check_sublink_is_exact(
    sublink: &pg_query::protobuf::SubLink,
    select: &SelectStmt,
    outer: &str,
) -> bool {
    if sublink.sub_link_type != SubLinkType::ArraySublink as i32
        || !catalog_scalar_subselect_is_plain(select)
        || select.from_clause.len() != 1
        || !catalog_range_is_exact(&select.from_clause[0], "pg_constraint", "r")
    {
        return false;
    }
    let Some(target) = catalog_target_value(select) else {
        return false;
    };
    let Ok(NodeEnum::FuncCall(function)) = node_enum(target) else {
        return false;
    };
    if catalog_function_name(function).ok().as_deref() != Some("pg_get_constraintdef")
        || function.args.len() != 2
        || !catalog_column_is_exact(&function.args[0], "r", "oid")
        || !catalog_bool_constant_is_exact(&function.args[1], true)
    {
        return false;
    }
    select.where_clause.as_deref().is_some_and(|predicate| {
        catalog_column_comparison_is_exact(predicate, "=", (outer, "oid"), ("r", "contypid"))
    })
}

pub(super) fn catalog_function_name(
    function: &pg_query::protobuf::FuncCall,
) -> Result<String, ExecuteError> {
    let parts = function
        .funcname
        .iter()
        .map(|name| match node_enum(name)? {
            NodeEnum::String(name) => Ok(name.sval.to_ascii_lowercase()),
            _ => Err(sql_pg_error(
                "catalog function name is malformed".to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let name = match parts.as_slice() {
        [name] => Ok(name.clone()),
        [schema, name] if schema == "pg_catalog" => Ok(name.clone()),
        _ => Err(sql_pg_error(format!(
            "catalog function {} is not an unqualified or pg_catalog builtin",
            parts.join(".")
        ))),
    }?;
    let aggregate = matches!(
        name.as_str(),
        "avg" | "bool_and" | "bool_or" | "count" | "max" | "min" | "string_agg" | "sum"
    );
    if !aggregate
        && (function.agg_distinct
            || function.agg_filter.is_some()
            || function.agg_within_group
            || !function.agg_order.is_empty()
            || function.agg_star
            || function.over.is_some()
            || function.func_variadic)
    {
        return Err(sql_pg_error(format!(
            "scalar catalog function {name} does not accept aggregate or window modifiers"
        )));
    }
    Ok(name)
}

struct CatalogCaseBinding {
    source: JoinColRef,
    arms: Vec<(SqlValue, SqlValue)>,
    otherwise: Option<SqlValue>,
    reject_unmatched: bool,
}

fn parse_catalog_case(
    case: &pg_query::protobuf::CaseExpr,
) -> Result<CatalogCaseBinding, ExecuteError> {
    let mut source = case.arg.as_deref().map(parse_join_col_ref).transpose()?;
    let mut arms = Vec::with_capacity(case.args.len());
    for arm in &case.args {
        let NodeEnum::CaseWhen(arm) = node_enum(arm)? else {
            return Err(sql_pg_error("catalog CASE arm is malformed".to_string()));
        };
        let condition = arm
            .expr
            .as_deref()
            .ok_or_else(|| sql_pg_error("catalog CASE arm has no condition".to_string()))?;
        let value = match node_enum(condition)? {
            NodeEnum::AConst(constant) => catalog_scalar_constant(constant)?,
            NodeEnum::ColumnRef(_) => {
                let candidate_source = parse_join_col_ref(condition)?;
                if let Some(expected) = &source {
                    if expected.qualifier != candidate_source.qualifier
                        || expected.column != candidate_source.column
                    {
                        return Err(sql_pg_error(
                            "catalog CASE arms must reference one source column".to_string(),
                        ));
                    }
                } else {
                    source = Some(candidate_source);
                }
                SqlValue::Bool(true)
            }
            NodeEnum::AExpr(compare) => {
                let lhs = compare.lexpr.as_deref().ok_or_else(|| {
                    sql_pg_error("catalog CASE comparison has no lhs".to_string())
                })?;
                let rhs = compare.rexpr.as_deref().ok_or_else(|| {
                    sql_pg_error("catalog CASE comparison has no rhs".to_string())
                })?;
                let (candidate_source, constant) =
                    match (node_enum(lhs)?, node_enum(rhs)?) {
                        (NodeEnum::ColumnRef(_), NodeEnum::AConst(constant)) => {
                            (parse_join_col_ref(lhs)?, constant)
                        }
                        (NodeEnum::CaseTestExpr(_), NodeEnum::AConst(constant)) => {
                            let source = source.clone().ok_or_else(|| {
                                sql_pg_error("simple catalog CASE lost its source".to_string())
                            })?;
                            (source, constant)
                        }
                        (NodeEnum::AConst(constant), NodeEnum::ColumnRef(_)) => {
                            (parse_join_col_ref(rhs)?, constant)
                        }
                        _ => return Err(sql_pg_error(
                            "catalog CASE comparison requires one source column and one constant"
                                .to_string(),
                        )),
                    };
                if let Some(expected) = &source {
                    if expected.qualifier != candidate_source.qualifier
                        || expected.column != candidate_source.column
                    {
                        return Err(sql_pg_error(
                            "catalog CASE arms must reference one source column".to_string(),
                        ));
                    }
                } else {
                    source = Some(candidate_source);
                }
                if aexpr_op_token(compare)? != "=" {
                    return Err(sql_pg_error(
                        "catalog CASE comparison must use equality".to_string(),
                    ));
                }
                catalog_scalar_constant(constant)?
            }
            _ => {
                return Err(sql_pg_error(
                    "catalog CASE comparison requires a text constant".to_string(),
                ))
            }
        };
        let result = arm
            .result
            .as_deref()
            .ok_or_else(|| sql_pg_error("catalog CASE arm has no result".to_string()))?;
        let NodeEnum::AConst(result) = node_enum(result)? else {
            return Err(sql_pg_error(
                "catalog CASE result requires a scalar constant".to_string(),
            ));
        };
        arms.push((value, catalog_scalar_constant(result)?));
    }
    let (otherwise, reject_unmatched) = match case.defresult.as_deref() {
        None => (None, false),
        Some(result) => match node_enum(result)? {
            NodeEnum::AConst(result) => (Some(catalog_scalar_constant(result)?), false),
            // psql uses a display cast in the ELSE arm of a CASE whose modeled catalog source
            // always matches the constant arm (for example pg_class.reloftype = 0). Preserve
            // that supported case, but reject instead of fabricating NULL if the invariant changes.
            _ => (None, true),
        },
    };
    Ok(CatalogCaseBinding {
        source: source
            .ok_or_else(|| sql_pg_error("catalog CASE has no source column".to_string()))?,
        arms,
        otherwise,
        reject_unmatched,
    })
}

fn catalog_scalar_constant(
    constant: &pg_query::protobuf::AConst,
) -> Result<SqlValue, ExecuteError> {
    match &constant.val {
        Some(a_const::Val::Sval(value)) => Ok(SqlValue::Text(value.sval.clone())),
        Some(a_const::Val::Ival(value)) => Ok(SqlValue::Int4(value.ival)),
        Some(a_const::Val::Boolval(value)) => Ok(SqlValue::Bool(value.boolval)),
        _ => Err(sql_pg_error(
            "catalog presentation requires a text, int4, or bool constant".to_string(),
        )),
    }
}

pub(super) fn apply_catalog_projection_metadata(
    mut result: RelationalSelectResult,
    presentation: &[CatalogJoinPresentation],
) -> Result<RelationalSelectResult, ExecuteError> {
    if result.columns.len() != presentation.len() {
        return Err(sql_pg_error(
            "catalog presentation width differs from GPU projection".to_string(),
        ));
    }
    let mut columns = result.columns.as_ref().clone();
    for (index, (column, item)) in columns.iter_mut().zip(presentation).enumerate() {
        column.name.clone_from(&item.output_name);
        column.attnum = (index + 1) as i16;
    }
    // Result values are already final typed values projected from the transient device relation.
    // Only wire-visible names/attribute numbers are metadata; no host expression evaluation occurs.
    result.columns = Arc::new(columns);
    Ok(result)
}
pub(super) fn select_tree_uses_synthesized_catalog(
    stmt: &SelectStmt,
    catalog: &CatalogSnapshot,
) -> bool {
    let mut relations = Vec::new();
    collect_select_tree_relations(stmt, &mut relations);
    relations.into_iter().any(|relation| {
        !public_relation_name_exists(catalog, &relation)
            && synthesize_catalog_relation(&relation, catalog).is_some()
    })
}

pub(super) fn collect_select_tree_relations(stmt: &SelectStmt, relations: &mut Vec<String>) {
    for from in &stmt.from_clause {
        collect_from_relations(from, relations);
    }
    if let Some(left) = stmt.larg.as_deref() {
        collect_select_tree_relations(left, relations);
    }
    if let Some(right) = stmt.rarg.as_deref() {
        collect_select_tree_relations(right, relations);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_catalog_lookup_keeps_the_unaliased_relation_qualifier() {
        let stmt = parse_single_select(
            "SELECT pg_catalog.format_type(pg_attribute.atttypid, pg_attribute.atttypmod) \
             FROM pg_catalog.pg_attribute \
             WHERE pg_attribute.attrelid = 'qualifier_probe'::regclass \
             ORDER BY pg_attribute.attnum",
        )
        .unwrap();
        let plan = catalog_single_projection_plan(&stmt).unwrap();
        assert_eq!(plan.select.table, "pg_catalog.pg_attribute");
        assert_eq!(plan.qualifier, "pg_attribute");
    }

    #[test]
    fn pg16_verbose_relation_presentation_accepts_only_the_frozen_gpu_columns() {
        let stmt = parse_single_select(
            "SELECT c.relname, \
             CASE c.relpersistence \
               WHEN 'p' THEN 'permanent' \
               WHEN 't' THEN 'temporary' \
               WHEN 'u' THEN 'unlogged' \
             END AS persistence, \
             pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) AS size \
             FROM pg_catalog.pg_class c \
             LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
        )
        .unwrap();
        let plan = catalog_join_projection_plan(&stmt).unwrap();
        assert!(matches!(
            &plan.projection[1],
            JoinProjItem::Column(column)
                if column.column == GPU_CATALOG_RELPERSISTENCE_DISPLAY
        ));
        assert!(matches!(
            &plan.projection[2],
            JoinProjItem::Column(column) if column.column == GPU_CATALOG_RELATION_SIZE
        ));

        for sql in [
            "SELECT CASE c.relpersistence \
               WHEN 'p' THEN 'persistent' \
               WHEN 't' THEN 'temporary' \
               WHEN 'u' THEN 'unlogged' \
             END \
             FROM pg_catalog.pg_class c \
             LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
            "SELECT pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(n.oid)) \
             FROM pg_catalog.pg_class c \
             LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
        ] {
            let stmt = parse_single_select(sql).unwrap();
            assert!(
                catalog_join_projection_plan(&stmt).is_err(),
                "altered PostgreSQL presentation must fail closed"
            );
        }
    }

    #[test]
    fn pg16_describe_reloptions_accepts_only_the_frozen_gpu_empty_display() {
        let sql = "SELECT pg_catalog.array_to_string(\
            c.reloptions || ARRAY(\
              SELECT 'toast.' || x FROM pg_catalog.unnest(tc.reloptions) x\
            ), ', ') \
            FROM pg_catalog.pg_class c \
            LEFT JOIN pg_catalog.pg_class tc ON c.reltoastrelid = tc.oid";
        let stmt = parse_single_select(sql).unwrap();
        let plan = catalog_join_projection_plan(&stmt).unwrap();
        assert!(matches!(
            &plan.projection[0],
            JoinProjItem::Column(column) if column.column == GPU_CATALOG_EMPTY_TEXT
        ));

        let altered = parse_single_select(&sql.replace("'toast.'", "'toastx.'")).unwrap();
        assert!(
            catalog_join_projection_plan(&altered).is_err(),
            "altered reloptions presentation must fail closed"
        );
    }

    #[test]
    fn pg16_verbose_attribute_presentation_accepts_only_the_frozen_gpu_columns() {
        let stmt = parse_single_select(
            "SELECT a.attname, a.attstorage, a.attcompression, \
             CASE WHEN a.attstattarget = -1 THEN NULL ELSE a.attstattarget END \
               AS attstattarget, \
             pg_catalog.col_description(a.attrelid, a.attnum) \
             FROM pg_catalog.pg_attribute a",
        )
        .unwrap();
        let plan = catalog_single_projection_plan(&stmt).unwrap();
        let SelectProjection::Columns(columns) = &plan.select.projection else {
            panic!("attribute presentation must project modeled columns");
        };
        assert_eq!(columns[1], "attstorage");
        assert_eq!(columns[2], "attcompression");
        assert_eq!(columns[3], GPU_CATALOG_NULL_INT4);
        assert_eq!(columns[4], GPU_CATALOG_DESCRIPTION);

        for sql in [
            "SELECT CASE WHEN a.attstattarget = 0 THEN NULL \
               ELSE a.attstattarget END \
             FROM pg_catalog.pg_attribute a",
            "SELECT pg_catalog.col_description(a.attrelid, a.atttypid) \
             FROM pg_catalog.pg_attribute a",
        ] {
            let stmt = parse_single_select(sql).unwrap();
            assert!(
                catalog_single_projection_plan(&stmt).is_err(),
                "altered PostgreSQL attribute presentation must fail closed"
            );
        }
    }

    #[test]
    fn relation_obj_description_requires_the_matching_catalog_class() {
        let stmt = parse_single_select(
            "SELECT pg_catalog.obj_description(c.oid, 'pg_class') \
             FROM pg_catalog.pg_class c \
             LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
        )
        .unwrap();
        let plan = catalog_join_projection_plan(&stmt).unwrap();
        assert!(matches!(
            &plan.projection[0],
            JoinProjItem::Column(column) if column.column == GPU_CATALOG_DESCRIPTION
        ));

        let altered = parse_single_select(
            "SELECT pg_catalog.obj_description(c.oid, 'pg_type') \
             FROM pg_catalog.pg_class c \
             LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
        )
        .unwrap();
        assert!(
            catalog_join_projection_plan(&altered).is_err(),
            "a mismatched description class must fail closed"
        );
    }
}
