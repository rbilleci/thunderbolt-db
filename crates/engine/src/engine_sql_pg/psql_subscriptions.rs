//! PostgreSQL 16 `psql \dRs` presentation over the GPU catalog path.

use super::*;

impl Engine {
    pub(super) fn execute_psql_subscriptions_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        if !is_pg16_psql_subscription_list(&canonical) {
            return Ok(None);
        }

        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let table = catalog_relation_table(
            "pg_catalog",
            "__psql_subscription_list",
            &[
                ("Name", SqlType::Text),
                ("Owner", SqlType::Text),
                ("Enabled", SqlType::Bool),
                ("Publication", SqlType::Text),
            ],
        );
        let rows = catalog
            .relational_subscriptions
            .values()
            .map(|subscription| {
                vec![
                    SqlValue::Text(subscription.name.clone()),
                    SqlValue::Text("postgres".to_string()),
                    SqlValue::Bool(subscription.enabled),
                    SqlValue::Text(subscription.publications.join(", ")),
                ]
            })
            .collect();
        self.execute_pg_dump_gpu_select(
            table,
            rows,
            SelectProjection::Columns(
                ["Name", "Owner", "Enabled", "Publication"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            None,
            &["Name"],
            boundary,
        )
        .map(Some)
    }
}

fn is_pg16_psql_subscription_list(canonical: &str) -> bool {
    canonical.starts_with("select subname as \"Name\"")
        && canonical.contains("pg_catalog.pg_get_userbyid(subowner) as \"Owner\"")
        && canonical.contains("subenabled as \"Enabled\"")
        && canonical.contains("subpublications as \"Publication\"")
        && canonical.contains("from pg_catalog.pg_subscription")
        && canonical.contains("where subdbid = (select oid from pg_catalog.pg_database")
        && canonical.contains("where datname = pg_catalog.current_database())")
        && canonical.ends_with("order by 1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscription_list_recognition_is_bounded_to_pg16_program() {
        let sql = r#"
            SELECT subname AS "Name"
            , pg_catalog.pg_get_userbyid(subowner) AS "Owner"
            , subenabled AS "Enabled"
            , subpublications AS "Publication"
            FROM pg_catalog.pg_subscription
            WHERE subdbid = (SELECT oid
                             FROM pg_catalog.pg_database
                             WHERE datname = pg_catalog.current_database())ORDER BY 1
        "#;
        assert!(is_pg16_psql_subscription_list(
            &canonicalize_sql_for_exact_match(sql).unwrap()
        ));
        assert!(!is_pg16_psql_subscription_list(
            "select subname from pg_catalog.pg_subscription"
        ));
    }
}
