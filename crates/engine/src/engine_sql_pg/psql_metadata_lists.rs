//! Bounded PostgreSQL 16 psql schema/type list programs on transient GPU relations.

use super::*;

impl Engine {
    pub(super) fn execute_psql_metadata_lists_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        if is_pg16_schema_list(&canonical) {
            return self.execute_pg16_schema_list().map(Some);
        }
        if is_pg16_direct_schema_acl(&canonical) {
            return self.execute_pg16_direct_schema_acl().map(Some);
        }
        if is_pg16_domain_list(&canonical) {
            return self.execute_pg16_domain_list(&canonical).map(Some);
        }
        let Some(type_filter) = pg16_type_list_filter(&canonical) else {
            return Ok(None);
        };
        self.execute_pg16_type_list(&canonical, type_filter)
            .map(Some)
    }

    fn execute_pg16_schema_list(&self) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let table = catalog_relation_table(
            "pg_catalog",
            "__psql_schema_list",
            &[("Name", SqlType::Text), ("Owner", SqlType::Text)],
        );
        let rows = if catalog.relational_public_schema_exists {
            vec![vec![
                SqlValue::Text("public".to_string()),
                SqlValue::Text("postgres".to_string()),
            ]]
        } else {
            Vec::new()
        };
        self.execute_pg_dump_gpu_select(
            table,
            rows,
            SelectProjection::Columns(vec!["Name".to_string(), "Owner".to_string()]),
            None,
            &["Name"],
            boundary,
        )
    }

    fn execute_pg16_direct_schema_acl(&self) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let table = catalog_relation_table(
            "pg_catalog",
            "__psql_direct_schema_acl",
            &[("nspname", SqlType::Text), ("nspacl", SqlType::Text)],
        );
        let rows = if catalog.relational_public_schema_exists {
            vec![vec![
                SqlValue::Text("public".to_string()),
                schema_acl_display(&catalog.relational_schema_acl),
            ]]
        } else {
            Vec::new()
        };
        self.execute_pg_dump_transient_relation(table, rows, &["nspname"], boundary)
    }

    fn execute_pg16_type_list(
        &self,
        canonical: &str,
        type_filter: Option<&str>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let verbose = canonical.contains("as \"Internal name\"");
        let columns = if verbose {
            vec![
                ("__internal_key", SqlType::Text),
                ("Schema", SqlType::Text),
                ("Name", SqlType::Text),
                ("Internal name", SqlType::Text),
                ("Size", SqlType::Text),
                ("Elements", SqlType::Text),
                ("Owner", SqlType::Text),
                ("Access privileges", SqlType::Text),
                ("Description", SqlType::Text),
            ]
        } else {
            vec![
                ("__internal_key", SqlType::Text),
                ("Schema", SqlType::Text),
                ("Name", SqlType::Text),
                ("Description", SqlType::Text),
            ]
        };
        let table = catalog_relation_table("pg_catalog", "__psql_type_list", &columns);
        let rows = gpu_db_sql::SUPPORTED_SQL_TYPES
            .into_iter()
            .map(|ty| {
                let internal = ty.catalog_name();
                let display = catalog_type_name(ty.postgres_oid() as i32, &catalog)
                    .expect("every supported SQL type has a PostgreSQL display name");
                if verbose {
                    vec![
                        SqlValue::Text(internal.to_string()),
                        SqlValue::Text("pg_catalog".to_string()),
                        SqlValue::Text(display.to_string()),
                        SqlValue::Text(internal.to_string()),
                        legacy_type_size(internal),
                        SqlValue::Null,
                        SqlValue::Text("postgres".to_string()),
                        SqlValue::Null,
                        SqlValue::Null,
                    ]
                } else {
                    vec![
                        SqlValue::Text(internal.to_string()),
                        SqlValue::Text("pg_catalog".to_string()),
                        SqlValue::Text(display.to_string()),
                        SqlValue::Null,
                    ]
                }
            })
            .collect();
        let predicate = type_filter.map(|name| ResidentExpr::Binary {
            op: ResidentBinaryOp::Eq,
            lhs: Box::new(ResidentExpr::Column(0)),
            rhs: Box::new(ResidentExpr::TextLiteral(name.to_string())),
        });
        let projection = if verbose {
            [
                "Schema",
                "Name",
                "Internal name",
                "Size",
                "Elements",
                "Owner",
                "Access privileges",
                "Description",
            ]
            .into_iter()
            .map(str::to_string)
            .collect()
        } else {
            ["Schema", "Name", "Description"]
                .into_iter()
                .map(str::to_string)
                .collect()
        };
        self.execute_pg_dump_gpu_select(
            table,
            rows,
            SelectProjection::Columns(projection),
            predicate,
            &["Schema", "Name"],
            boundary,
        )
    }

    fn execute_pg16_domain_list(
        &self,
        canonical: &str,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let verbose = canonical.contains("as \"Access privileges\"");
        let names = if verbose {
            vec![
                "Schema",
                "Name",
                "Type",
                "Collation",
                "Nullable",
                "Default",
                "Check",
                "Access privileges",
                "Description",
            ]
        } else {
            vec![
                "Schema",
                "Name",
                "Type",
                "Collation",
                "Nullable",
                "Default",
                "Check",
            ]
        };
        let table = catalog_relation_table(
            "pg_catalog",
            "__psql_domain_list",
            &names
                .iter()
                .map(|name| (*name, SqlType::Text))
                .collect::<Vec<_>>(),
        );
        let rows = catalog
            .relational_domains
            .values()
            .map(|domain| {
                let mut row = vec![
                    SqlValue::Text(domain.schema.clone()),
                    SqlValue::Text(domain.name.clone()),
                    SqlValue::Text(
                        catalog_type_name(domain.base_type.postgres_oid() as i32, &catalog)
                            .expect("domain base type is modeled")
                            .to_string(),
                    ),
                    SqlValue::Null,
                    SqlValue::Null,
                    SqlValue::Null,
                    SqlValue::Null,
                ];
                if verbose {
                    row.extend([
                        SqlValue::Null,
                        catalog
                            .relational_comments
                            .get(&RelationalCommentTarget::Domain {
                                domain: domain.name.clone(),
                            })
                            .cloned()
                            .map_or(SqlValue::Null, SqlValue::Text),
                    ]);
                }
                row
            })
            .collect();
        self.execute_pg_dump_transient_relation(table, rows, &["Schema", "Name"], boundary)
    }
}

fn is_pg16_schema_list(canonical: &str) -> bool {
    canonical
        == "select n.nspname as \"Name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"Owner\" \
            from pg_catalog.pg_namespace n where n.nspname !~ '^pg_' and n.nspname <> \
            'information_schema' order by 1"
}

fn is_pg16_direct_schema_acl(canonical: &str) -> bool {
    canonical
        == "select n.nspname, n.nspacl from pg_catalog.pg_namespace n where n.nspname = 'public' order by n.nspname"
}

fn schema_acl_display(acl: &BTreeMap<String, BTreeSet<SchemaPrivilege>>) -> SqlValue {
    let entries = acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let mut letters = String::new();
            if privileges.contains(&SchemaPrivilege::Usage) {
                letters.push('U');
            }
            if privileges.contains(&SchemaPrivilege::Create) {
                letters.push('C');
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!("{grantee}={letters}/postgres"))
        })
        .collect::<Vec<_>>();
    if entries.is_empty() {
        SqlValue::Null
    } else {
        SqlValue::Text(entries.join("\n"))
    }
}

fn is_pg16_domain_list(canonical: &str) -> bool {
    canonical.contains("from pg_catalog.pg_type t")
        && canonical.contains("t.typtype = 'd'")
        && canonical.contains("as \"Schema\"")
        && canonical.contains("as \"Name\"")
        && canonical.contains("as \"Type\"")
        && canonical.ends_with("order by 1, 2")
}

/// `Some(None)` recognizes the wildcard program; `Some(Some(name))` recognizes an exact
/// PostgreSQL internal type name.
fn pg16_type_list_filter(canonical: &str) -> Option<Option<&str>> {
    if !canonical.starts_with(
        "select n.nspname as \"Schema\", pg_catalog.format_type(t.oid, null) as \"Name\"",
    ) || !canonical.contains("from pg_catalog.pg_type t left join pg_catalog.pg_namespace n")
        || !canonical.contains(
            "n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default",
        )
        || !canonical.ends_with("order by 1, 2")
    {
        return None;
    }
    for name in [
        "bool",
        "int2",
        "int4",
        "int8",
        "numeric",
        "text",
        "date",
        "timestamp",
        "uuid",
    ] {
        if canonical.contains(&format!("'^{name}$'"))
            || canonical.contains(&format!("'^({name})$'"))
        {
            return Some(Some(name));
        }
    }
    canonical
        .contains("'^(.+)$'")
        .then_some(None)
        .or_else(|| canonical.contains("'^(.*)$'").then_some(None))
        .or_else(|| {
            (!canonical.contains("t.typname operator(pg_catalog.~)")
                && !canonical
                    .contains("pg_catalog.format_type(t.oid, null) operator(pg_catalog.~)"))
            .then_some(None)
        })
}

fn legacy_type_size(internal: &str) -> SqlValue {
    match internal {
        "int4" | "date" => SqlValue::Text("4".to_string()),
        "numeric" | "text" => SqlValue::Text("var".to_string()),
        _ => SqlValue::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_list_recognition_is_exact() {
        let sql = r#"
            SELECT n.nspname AS "Name",
              pg_catalog.pg_get_userbyid(n.nspowner) AS "Owner"
            FROM pg_catalog.pg_namespace n
            WHERE n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
            ORDER BY 1
        "#;
        assert!(is_pg16_schema_list(
            &canonicalize_sql_for_exact_match(sql).unwrap()
        ));
    }
}
