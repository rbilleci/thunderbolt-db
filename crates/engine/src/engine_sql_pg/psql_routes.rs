//! Exact PostgreSQL 16 catalog-client route dispatch.
//!
//! Each leaf owns one bounded query family. This coordinator preserves their precedence without
//! adding a second relational executor: every returned relation is synthesized for the existing
//! GPU execution/result boundary.

use super::*;
use pg_query::protobuf::SelectStmt;

impl Engine {
    pub(super) fn execute_psql_catalog_route_if_applicable(
        &self,
        sql: &str,
        stmt: &SelectStmt,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        if let Some(result) = self.execute_pg_dump_catalog_route_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_object_descriptions_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_relation_detail_route_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_information_schema_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_metadata_lists_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_databases_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_shared_comments_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_bootstrap_metadata_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_roles_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_functions_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_publications_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_empty_metadata_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_subscriptions_if_applicable(sql)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.execute_psql_sequence_catalog_if_applicable(sql)? {
            return Ok(Some(result));
        }
        self.execute_psql_describe_route_if_applicable(stmt)
    }
}
