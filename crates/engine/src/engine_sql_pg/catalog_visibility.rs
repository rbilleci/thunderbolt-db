//! GPU predicate lowering for the modeled PostgreSQL catalog visibility functions.

use super::select_lowering::{node_enum, resolve_column_name, sql_pg_error};
use super::*;

pub(super) fn map_catalog_predicate_function(
    function: &pg_query::protobuf::FuncCall,
    table: &RelationalTable,
    qualifier: &str,
    catalog: &CatalogSnapshot,
) -> Result<ResidentExpr, ExecuteError> {
    if !matches!(table.schema.as_str(), "pg_catalog" | "information_schema") {
        return Err(sql_pg_error(
            "unsupported expression node for the general GPU executor".to_string(),
        ));
    }
    let name = super::catalog_presentation::catalog_function_name(function)?;
    if name == "current_database" {
        if function.args.is_empty() {
            // The endpoint currently exposes one fixed database identity. Lower that immutable
            // session/catalog fact to a typed literal; the surrounding comparison still compiles
            // and executes as a device text predicate.
            return Ok(ResidentExpr::TextLiteral("postgres".to_string()));
        }
        return Err(sql_pg_error(
            "function current_database requires no arguments".to_string(),
        ));
    }
    let expected_relation = match name.as_str() {
        "pg_table_is_visible" => Some("pg_class"),
        "pg_type_is_visible" => Some("pg_type"),
        "pg_function_is_visible" => Some("pg_proc"),
        _ => None,
    };
    let Some(expected_relation) = expected_relation else {
        return Err(sql_pg_error(format!(
            "function {name} is not supported in a GPU predicate"
        )));
    };
    if function.args.len() != 1 || table.schema != "pg_catalog" || table.name != expected_relation {
        return Err(visibility_shape_error(&name, expected_relation));
    }
    let NodeEnum::ColumnRef(column_ref) = node_enum(&function.args[0])? else {
        return Err(sql_pg_error(format!(
            "function {name} requires a catalog oid column reference"
        )));
    };
    let exposed_name = resolve_column_name(column_ref, qualifier)?;
    let index = relational_column_index(table, exposed_name)?;
    // RangeVar column aliases change only the exposed name. Catalog `attnum` remains the immutable
    // source-column identity, so the first/oid column stays provable after `AS t(x, ...)` while a
    // renamed namespace/name column still fails closed.
    if table.columns[index].attnum != 1 || table.columns[index].ty != SqlType::Int4 {
        return Err(visibility_shape_error(&name, expected_relation));
    }

    let hidden_oids = match name.as_str() {
        "pg_table_is_visible" => shadowed_public_relation_oids(catalog),
        "pg_type_is_visible" => shadowed_public_domain_oids(catalog),
        // The modeled pg_proc relation is currently empty. Retain the shape-checked expression so
        // the path remains GPU-only if an empty transient relation reaches this mapper.
        "pg_function_is_visible" => Vec::new(),
        _ => unreachable!("visibility function matched above"),
    };
    oid_exclusion_predicate(index, hidden_oids)
}

fn visibility_shape_error(name: &str, relation: &str) -> ExecuteError {
    sql_pg_error(format!(
        "function {name} requires the oid column of pg_catalog.{relation}"
    ))
}

fn shadowed_public_relation_oids(catalog: &CatalogSnapshot) -> Vec<u32> {
    let modeled_name = |name: &str| MODELED_PG_CATALOG_RELATION_NAMES.contains(&name);
    catalog
        .relational_catalog
        .values()
        .filter(|relation| modeled_name(&relation.name))
        .map(|relation| relation.oid)
        .chain(
            catalog
                .relational_views
                .values()
                .filter(|relation| modeled_name(&relation.name))
                .map(|relation| relation.oid),
        )
        .chain(
            catalog
                .relational_materialized_views
                .values()
                .filter(|relation| modeled_name(&relation.name))
                .map(|relation| relation.oid),
        )
        .chain(
            catalog
                .relational_sequences
                .values()
                .filter(|relation| modeled_name(&relation.name))
                .map(|relation| relation.oid),
        )
        .collect()
}

fn shadowed_public_domain_oids(catalog: &CatalogSnapshot) -> Vec<u32> {
    catalog
        .relational_domains
        .values()
        .filter(|domain| {
            gpu_db_sql::SUPPORTED_SQL_TYPES
                .iter()
                .any(|ty| ty.catalog_name() == domain.name)
        })
        .map(|domain| domain.oid)
        .collect()
}

/// Compile search-path shadowing as device OID comparisons. The host supplies immutable catalog
/// constants during planning; row membership/filtering stays entirely in the GPU predicate VM.
fn oid_exclusion_predicate(
    column: usize,
    hidden_oids: Vec<u32>,
) -> Result<ResidentExpr, ExecuteError> {
    let value = ResidentExpr::Column(column);
    let mut visible = binary_expr(ResidentBinaryOp::Eq, value.clone(), value);
    for oid in hidden_oids {
        let oid = i32::try_from(oid)
            .map_err(|_| sql_pg_error("catalog OID exceeds the modeled int4 range".to_string()))?;
        visible = binary_expr(
            ResidentBinaryOp::And,
            visible,
            binary_expr(
                ResidentBinaryOp::Ne,
                ResidentExpr::Column(column),
                ResidentExpr::Int4Literal(oid),
            ),
        );
    }
    Ok(visible)
}

fn binary_expr(op: ResidentBinaryOp, lhs: ResidentExpr, rhs: ResidentExpr) -> ResidentExpr {
    ResidentExpr::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}
