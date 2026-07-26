//! Authoritatively empty PostgreSQL 16 psql catalog families.
//!
//! These exact list programs have no modeled object kind to scan. Recognition supplies the
//! protocol descriptor while the result remains a typed, GPU-targeted empty catalog relation.

use super::*;

impl Engine {
    pub(super) fn execute_psql_empty_metadata_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let columns: Option<&[&str]> = if canonical.contains("from pg_catalog.pg_proc p")
            && canonical.contains("p.prokind")
            && catalog.relational_functions.is_empty()
        {
            if canonical.contains("p.prokind = 'a'") {
                Some(&[
                    "Schema",
                    "Name",
                    "Result data type",
                    "Argument data types",
                    "Description",
                ])
            } else if canonical.contains("as \"Result data type\"")
                && canonical.contains("as \"Argument data types\"")
            {
                Some(&[
                    "Schema",
                    "Name",
                    "Result data type",
                    "Argument data types",
                    "Type",
                ])
            } else {
                None
            }
        } else if canonical.contains("from pg_catalog.pg_conversion c") {
            Some(&["Schema", "Name", "Source", "Destination", "Default?"])
        } else if canonical.contains("from pg_catalog.pg_operator o") {
            Some(&[
                "Schema",
                "Name",
                "Left arg type",
                "Right arg type",
                "Result type",
                "Description",
            ])
        } else if canonical.contains("from pg_catalog.pg_type t")
            && canonical.contains("t.typtype = 'd'")
            && catalog.relational_domains.is_empty()
        {
            if canonical.contains("as \"Access privileges\"") {
                Some(&[
                    "Schema",
                    "Name",
                    "Type",
                    "Collation",
                    "Nullable",
                    "Default",
                    "Check",
                    "Access privileges",
                    "Description",
                ])
            } else {
                Some(&[
                    "Schema",
                    "Name",
                    "Type",
                    "Collation",
                    "Nullable",
                    "Default",
                    "Check",
                ])
            }
        } else if canonical.starts_with("select n.nspname as \"Schema\", c.collname as \"Name\"")
            && canonical.contains("from pg_catalog.pg_collation c")
        {
            Some(&[
                "Schema",
                "Name",
                "Provider",
                "Collate",
                "Ctype",
                "ICU Locale",
                "ICU Rules",
                "Deterministic?",
            ])
        } else if canonical.contains("from pg_catalog.pg_cast c") {
            Some(&["Source type", "Target type", "Function", "Implicit?"])
        } else if canonical.contains("from pg_catalog.pg_default_acl d")
            && catalog.relational_default_table_acl.is_empty()
        {
            Some(&["Owner", "Schema", "Type", "Access privileges"])
        } else {
            None
        };
        let Some(columns) = columns else {
            return Ok(None);
        };
        let table = catalog_relation_table(
            "pg_catalog",
            "__psql_empty_metadata",
            &columns
                .iter()
                .map(|column| (*column, SqlType::Text))
                .collect::<Vec<_>>(),
        );
        self.execute_pg_dump_transient_relation(table, Vec::new(), &[], boundary)
            .map(Some)
    }
}
