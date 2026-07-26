//! PostgreSQL 16 role listings with the frozen bootstrap-first GPU order.

use super::*;

const ROLE_LIST: &str = "select r.rolname, r.rolsuper, r.rolinherit, r.rolcreaterole, \
    r.rolcreatedb, r.rolcanlogin, r.rolconnlimit, r.rolvaliduntil , r.rolreplication , \
    r.rolbypassrls from pg_catalog.pg_roles r where r.rolname !~ '^pg_' order by 1";
const ROLE_LIST_VERBOSE: &str = "select r.rolname, r.rolsuper, r.rolinherit, r.rolcreaterole, \
    r.rolcreatedb, r.rolcanlogin, r.rolconnlimit, r.rolvaliduntil , \
    pg_catalog.shobj_description(r.oid, 'pg_authid') as description , r.rolreplication , \
    r.rolbypassrls from pg_catalog.pg_roles r where r.rolname !~ '^pg_' order by 1";
const ROLE_OIDS: &str = "select oid, rolname from pg_catalog.pg_roles order by 1";

impl Engine {
    pub(super) fn execute_psql_roles_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        if !matches!(
            canonical.as_str(),
            ROLE_LIST | ROLE_LIST_VERBOSE | ROLE_OIDS
        ) {
            return Ok(None);
        }
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        if canonical == ROLE_OIDS {
            let table = catalog_relation_table(
                "pg_catalog",
                "__psql_role_oids",
                &[
                    ("__bootstrap", SqlType::Int4),
                    ("oid", SqlType::Int4),
                    ("rolname", SqlType::Text),
                ],
            );
            let rows = role_entries(&catalog)
                .into_iter()
                .map(|entry| {
                    vec![
                        SqlValue::Int4(entry.priority),
                        SqlValue::Int4(entry.oid as i32),
                        SqlValue::Text(entry.name),
                    ]
                })
                .collect();
            return self
                .execute_pg_dump_gpu_select(
                    table,
                    rows,
                    SelectProjection::Columns(vec!["oid".to_string(), "rolname".to_string()]),
                    None,
                    &["__bootstrap", "rolname"],
                    boundary,
                )
                .map(Some);
        }

        let verbose = canonical == ROLE_LIST_VERBOSE;
        let mut columns = vec![
            ("__bootstrap", SqlType::Int4),
            ("rolname", SqlType::Text),
            ("rolsuper", SqlType::Bool),
            ("rolinherit", SqlType::Bool),
            ("rolcreaterole", SqlType::Bool),
            ("rolcreatedb", SqlType::Bool),
            ("rolcanlogin", SqlType::Bool),
            ("rolconnlimit", SqlType::Int4),
            ("rolvaliduntil", SqlType::Text),
        ];
        if verbose {
            columns.push(("description", SqlType::Text));
        }
        columns.extend([
            ("rolreplication", SqlType::Bool),
            ("rolbypassrls", SqlType::Bool),
        ]);
        let table = catalog_relation_table("pg_catalog", "__psql_roles", &columns);
        let rows = role_entries(&catalog)
            .into_iter()
            .map(|entry| {
                let bootstrap = entry.priority == 0;
                let mut row = vec![
                    SqlValue::Int4(entry.priority),
                    SqlValue::Text(entry.name.clone()),
                    SqlValue::Bool(bootstrap),
                    SqlValue::Bool(true),
                    SqlValue::Bool(bootstrap),
                    SqlValue::Bool(bootstrap),
                    SqlValue::Bool(entry.login),
                    SqlValue::Int4(-1),
                    SqlValue::Null,
                ];
                if verbose {
                    row.push(
                        catalog
                            .relational_comments
                            .get(&RelationalCommentTarget::Role {
                                role: entry.name.clone(),
                            })
                            .cloned()
                            .map_or(SqlValue::Null, SqlValue::Text),
                    );
                }
                row.extend([SqlValue::Bool(bootstrap), SqlValue::Bool(bootstrap)]);
                row
            })
            .collect();
        let projection = columns
            .iter()
            .skip(1)
            .map(|(name, _)| (*name).to_string())
            .collect();
        self.execute_pg_dump_gpu_select(
            table,
            rows,
            SelectProjection::Columns(projection),
            None,
            &["__bootstrap", "rolname"],
            boundary,
        )
        .map(Some)
    }
}

struct RoleEntry {
    priority: i32,
    oid: u32,
    name: String,
    login: bool,
}

fn role_entries(catalog: &CatalogSnapshot) -> Vec<RoleEntry> {
    let mut entries = vec![RoleEntry {
        priority: 0,
        oid: 10,
        name: "postgres".to_string(),
        login: true,
    }];
    entries.extend(catalog.relational_roles.values().map(|role| RoleEntry {
        priority: 1,
        oid: role.oid,
        name: role.name.clone(),
        login: role.login,
    }));
    entries
}
