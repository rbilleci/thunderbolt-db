//! PostgreSQL 16 `psql \dRp` publication list on the GPU catalog path.

use super::*;

impl Engine {
    pub(super) fn execute_psql_publications_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        if !is_pg16_publication_list(&canonical) {
            return Ok(None);
        }
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let table = catalog_relation_table(
            "pg_catalog",
            "__psql_publication_list",
            &[
                ("Name", SqlType::Text),
                ("Owner", SqlType::Text),
                ("All tables", SqlType::Bool),
                ("Inserts", SqlType::Bool),
                ("Updates", SqlType::Bool),
                ("Deletes", SqlType::Bool),
                ("Truncates", SqlType::Bool),
                ("Via root", SqlType::Bool),
            ],
        );
        let rows = catalog
            .relational_publications
            .values()
            .map(|publication| {
                vec![
                    SqlValue::Text(publication.name.clone()),
                    SqlValue::Text("postgres".to_string()),
                    SqlValue::Bool(publication.all_tables),
                    SqlValue::Bool(true),
                    SqlValue::Bool(true),
                    SqlValue::Bool(true),
                    SqlValue::Bool(true),
                    SqlValue::Bool(false),
                ]
            })
            .collect();
        self.execute_pg_dump_transient_relation(table, rows, &["Name"], boundary)
            .map(Some)
    }
}

fn is_pg16_publication_list(canonical: &str) -> bool {
    canonical.starts_with("select pubname as \"Name\"")
        && canonical.contains("pg_catalog.pg_get_userbyid(pubowner) as \"Owner\"")
        && canonical.contains("puballtables as \"All tables\"")
        && canonical.contains("pubviaroot as \"Via root\"")
        && canonical.contains("from pg_catalog.pg_publication")
        && canonical.ends_with("order by 1")
}
