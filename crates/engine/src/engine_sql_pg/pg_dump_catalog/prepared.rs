//! Typed GPU plans for PostgreSQL 16 prepared catalog programs.

use super::*;

pub(crate) fn pg16_prepared_catalog_program_table(
    program: &PreparedCatalogProgram,
) -> RelationalTable {
    let (name, columns): (&str, &[(&str, SqlType)]) = match program {
        PreparedCatalogProgram::Pg16DomainConstraints { .. } => (
            "__pg16_dump_domain_constraints",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("conname", SqlType::Text),
                ("consrc", SqlType::Text),
                ("convalidated", SqlType::Bool),
            ],
        ),
        PreparedCatalogProgram::Pg16DomainDefinition { .. } => (
            "__pg16_dump_domain_definition",
            &[
                ("typnotnull", SqlType::Bool),
                ("typdefn", SqlType::Text),
                ("typdefaultbin", SqlType::Text),
                ("typdefault", SqlType::Text),
                ("typcollation", SqlType::Int4),
            ],
        ),
        PreparedCatalogProgram::Pg16FunctionDefinition { .. } => (
            "__pg16_dump_function_definition",
            &[
                ("proretset", SqlType::Bool),
                ("prosrc", SqlType::Text),
                ("probin", SqlType::Text),
                ("provolatile", SqlType::Text),
                ("proisstrict", SqlType::Bool),
                ("prosecdef", SqlType::Bool),
                ("lanname", SqlType::Text),
                ("proconfig", SqlType::Text),
                ("procost", SqlType::Text),
                ("prorows", SqlType::Text),
                ("funcargs", SqlType::Text),
                ("funciargs", SqlType::Text),
                ("funcresult", SqlType::Text),
                ("proleakproof", SqlType::Bool),
                ("protrftypes", SqlType::Text),
                ("proparallel", SqlType::Text),
                ("prokind", SqlType::Text),
                ("prosupport", SqlType::Text),
                ("prosqlbody", SqlType::Text),
            ],
        ),
        PreparedCatalogProgram::Pg16MaterializedViewDependencies => (
            "__pg16_dump_materialized_view_dependencies",
            &[
                ("classid", SqlType::Int4),
                ("objid", SqlType::Int4),
                ("refobjid", SqlType::Int4),
            ],
        ),
    };
    catalog_relation_table("pg_catalog", name, columns)
}

impl Engine {
    pub(super) fn execute_pg16_prepared_catalog_program_gpu(
        &self,
        program: &PreparedCatalogProgram,
        catalog: &CatalogSnapshot,
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        match program {
            PreparedCatalogProgram::Pg16DomainConstraints { type_oid } => {
                bound_catalog_oid(type_oid)?;
                // RelationalDomain has no constraint member, so this source is authoritatively empty
                // for every parameter. No host predicate decides its cardinality.
                self.execute_pg_dump_transient_relation(
                    pg16_prepared_catalog_program_table(program),
                    Vec::new(),
                    &["conname"],
                    boundary,
                )
            }
            PreparedCatalogProgram::Pg16DomainDefinition { type_oid } => {
                let requested = bound_catalog_oid(type_oid)?;
                let (table, rows) = pg16_domain_definition_source(catalog);
                let predicate = Some(int4_comparison(
                    &table,
                    "__oid",
                    ResidentBinaryOp::Eq,
                    requested.map_or(-1, |oid| oid as i32),
                )?);
                self.execute_pg_dump_gpu_select(
                    table,
                    rows,
                    SelectProjection::Columns(
                        [
                            "typnotnull",
                            "typdefn",
                            "typdefaultbin",
                            "typdefault",
                            "typcollation",
                        ]
                        .into_iter()
                        .map(str::to_string)
                        .collect(),
                    ),
                    predicate,
                    &[],
                    boundary,
                )
            }
            PreparedCatalogProgram::Pg16FunctionDefinition { function_oid } => {
                let requested = bound_catalog_oid(function_oid)?;
                let (functions, function_rows) = pg16_function_definition_source(catalog);
                let languages = catalog_relation_table(
                    "pg_catalog",
                    "pg_language",
                    &[("oid", SqlType::Int4), ("lanname", SqlType::Text)],
                );
                let language_rows =
                    vec![vec![SqlValue::Int4(14), SqlValue::Text("sql".to_string())]];
                let predicate = Some(int4_comparison(
                    &functions,
                    "__oid",
                    ResidentBinaryOp::Eq,
                    requested.map_or(-1, |oid| oid as i32),
                )?);
                let projection = [
                    "proretset",
                    "prosrc",
                    "probin",
                    "provolatile",
                    "proisstrict",
                    "prosecdef",
                    "proconfig",
                    "procost",
                    "prorows",
                    "funcargs",
                    "funciargs",
                    "funcresult",
                    "proleakproof",
                    "protrftypes",
                    "proparallel",
                    "prokind",
                    "prosupport",
                    "prosqlbody",
                ];
                let mut projected = projection
                    .into_iter()
                    .take(6)
                    .map(|column| ("p", column, column))
                    .collect::<Vec<_>>();
                projected.push(("l", "lanname", "lanname"));
                projected.extend(
                    projection
                        .into_iter()
                        .skip(6)
                        .map(|column| ("p", column, column)),
                );
                let plan = join_plan(
                    &[(&functions, "p"), (&languages, "l")],
                    vec![join_step("p", "__prolang", "l", "oid", false)],
                    projected,
                    Vec::new(),
                );
                self.execute_pg_dump_gpu_join(
                    &plan,
                    vec![functions, languages],
                    vec![function_rows, language_rows],
                    vec![predicate, None],
                    boundary,
                )
            }
            PreparedCatalogProgram::Pg16MaterializedViewDependencies => {
                // CREATE MATERIALIZED VIEW rejects a source view/materialized view in
                // preflight_create_materialized_view. Assert the generation invariant before using
                // the resulting authoritative-empty recursive dependency relation.
                let invalid = catalog.relational_materialized_views.values().any(|view| {
                    catalog.relational_views.contains_key(&view.query.table)
                        || catalog
                            .relational_materialized_views
                            .contains_key(&view.query.table)
                });
                if invalid {
                    return Err(sql_pg_error(
                        "catalog contains a materialized-view dependency forbidden by DDL preflight"
                            .to_string(),
                    ));
                }
                self.execute_pg_dump_transient_relation(
                    pg16_prepared_catalog_program_table(program),
                    Vec::new(),
                    &[],
                    boundary,
                )
            }
        }
    }
}

fn pg16_domain_definition_source(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let mut table =
        pg16_prepared_catalog_program_table(&PreparedCatalogProgram::Pg16DomainDefinition {
            type_oid: SqlValue::Null,
        });
    append_internal_column(&mut table, "__oid", SqlType::Int4);
    let rows = catalog
        .relational_domains
        .values()
        .map(|domain| {
            vec![
                SqlValue::Bool(false),
                SqlValue::Text(information_schema_data_type(domain.base_type).to_string()),
                SqlValue::Null,
                SqlValue::Null,
                SqlValue::Int4(0),
                SqlValue::Int4(domain.oid as i32),
            ]
        })
        .collect();
    (table, rows)
}

fn pg16_function_definition_source(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let mut table =
        pg16_prepared_catalog_program_table(&PreparedCatalogProgram::Pg16FunctionDefinition {
            function_oid: SqlValue::Null,
        });
    // The result's lanname is projected from the joined pg_language source, not this encoded slot.
    append_internal_column(&mut table, "__oid", SqlType::Int4);
    append_internal_column(&mut table, "__prolang", SqlType::Int4);
    let rows = catalog
        .relational_functions
        .values()
        .map(|function| {
            vec![
                SqlValue::Bool(false),
                SqlValue::Text(function.body.clone()),
                SqlValue::Null,
                SqlValue::Text("v".to_string()),
                SqlValue::Bool(false),
                SqlValue::Bool(false),
                SqlValue::Text("sql".to_string()),
                SqlValue::Null,
                SqlValue::Text("100".to_string()),
                SqlValue::Text("0".to_string()),
                SqlValue::Text(String::new()),
                SqlValue::Text(String::new()),
                SqlValue::Text(information_schema_data_type(function.return_type).to_string()),
                SqlValue::Bool(false),
                SqlValue::Null,
                SqlValue::Text("u".to_string()),
                SqlValue::Text("f".to_string()),
                SqlValue::Text("-".to_string()),
                SqlValue::Null,
                SqlValue::Int4(function.oid as i32),
                SqlValue::Int4(14),
            ]
        })
        .collect();
    (table, rows)
}

fn append_internal_column(table: &mut RelationalTable, name: &str, ty: SqlType) {
    table.columns.push(RelationalColumn {
        id: 0,
        table_oid: table.oid,
        attnum: (table.columns.len() + 1) as i16,
        name: name.to_string(),
        ty,
        domain: None,
        default: None,
        type_oid: ty.postgres_oid(),
        type_size: ty.type_size(),
    });
}

fn bound_catalog_oid(value: &SqlValue) -> Result<Option<u32>, ExecuteError> {
    match value {
        SqlValue::Int4(value) if *value >= 0 => Ok(Some(*value as u32)),
        SqlValue::Null => Ok(None),
        SqlValue::Parameter { .. } => Err(sql_pg_error(
            "prepared catalog program reached execution with an unbound parameter".to_string(),
        )),
        _ => Err(sql_pg_error(
            "prepared catalog program requires an oid-compatible int4 parameter".to_string(),
        )),
    }
}
