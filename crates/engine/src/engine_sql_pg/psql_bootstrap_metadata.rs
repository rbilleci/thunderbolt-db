//! PostgreSQL 16 psql lists for fixed bootstrap metadata.

use super::*;

impl Engine {
    pub(super) fn execute_psql_bootstrap_metadata_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (table, rows, order_by) = if canonical.contains("from pg_catalog.pg_extension e")
            && canonical.contains("e.extname as \"Name\"")
            && canonical.contains("e.extversion as \"Version\"")
        {
            let description = catalog
                .relational_comments
                .get(&RelationalCommentTarget::Extension {
                    extension: "plpgsql".to_string(),
                })
                .cloned()
                .unwrap_or_else(|| "PL/pgSQL procedural language".to_string());
            (
                text_table(
                    "__psql_extensions",
                    &["Name", "Version", "Schema", "Description"],
                ),
                vec![vec![
                    SqlValue::Text("plpgsql".to_string()),
                    SqlValue::Text("1.0".to_string()),
                    SqlValue::Text("pg_catalog".to_string()),
                    SqlValue::Text(description),
                ]],
                vec!["Name"],
            )
        } else if canonical.contains("from pg_catalog.pg_language l")
            && canonical.contains("l.lanname as \"Name\"")
        {
            (
                catalog_relation_table(
                    "pg_catalog",
                    "__psql_languages",
                    &[
                        ("Name", SqlType::Text),
                        ("Owner", SqlType::Text),
                        ("Trusted", SqlType::Bool),
                        ("Description", SqlType::Text),
                    ],
                ),
                vec![vec![
                    SqlValue::Text("plpgsql".to_string()),
                    SqlValue::Text("postgres".to_string()),
                    SqlValue::Bool(true),
                    SqlValue::Text("PL/pgSQL procedural language".to_string()),
                ]],
                vec!["Name"],
            )
        } else if canonical.contains("from pg_catalog.pg_am")
            && canonical.contains("as \"Name\"")
            && canonical.contains("as \"Type\"")
        {
            (
                text_table("__psql_access_methods", &["Name", "Type"]),
                vec![vec![
                    SqlValue::Text("heap".to_string()),
                    SqlValue::Text("Table".to_string()),
                ]],
                vec!["Name"],
            )
        } else {
            return Ok(None);
        };
        self.execute_pg_dump_transient_relation(table, rows, &order_by, boundary)
            .map(Some)
    }
}

fn text_table(name: &str, columns: &[&str]) -> RelationalTable {
    catalog_relation_table(
        "pg_catalog",
        name,
        &columns
            .iter()
            .map(|column| (*column, SqlType::Text))
            .collect::<Vec<_>>(),
    )
}
