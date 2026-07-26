//! PostgreSQL 16 dump object-description metadata.

use super::*;

const PG_CLASS_CLASS_OID: i32 = 1259;
const PG_CONSTRAINT_CLASS_OID: i32 = 2606;
const PG_NAMESPACE_CLASS_OID: i32 = 2615;
const PG_PROC_CLASS_OID: i32 = 1255;
const PG_PUBLICATION_CLASS_OID: i32 = 6104;
const PG_SUBSCRIPTION_CLASS_OID: i32 = 6100;
const PG_TYPE_CLASS_OID: i32 = 1247;
const PG_EXTENSION_CLASS_OID: i32 = 3079;

pub(super) fn is_pg16_dump_description_program(canonical: &str) -> bool {
    canonical
        == "select description, classoid, objoid, objsubid from pg_catalog.pg_description order by classoid, objoid, objsubid"
}

impl Engine {
    pub(super) fn execute_pg16_dump_descriptions(
        &self,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (table, rows) = pg16_dump_description_relation(&catalog);
        self.execute_pg_dump_transient_relation(
            table,
            rows,
            &["classoid", "objoid", "objsubid"],
            boundary,
        )
    }
}

fn pg16_dump_description_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__pg16_dump_descriptions",
        &[
            ("description", SqlType::Text),
            ("classoid", SqlType::Int4),
            ("objoid", SqlType::Int4),
            ("objsubid", SqlType::Int4),
        ],
    );
    let rows = catalog
        .relational_comments
        .iter()
        .filter_map(|(target, description)| {
            description_catalog_id(catalog, target).map(|(classoid, objoid, objsubid)| {
                vec![
                    SqlValue::Text(description.clone()),
                    SqlValue::Int4(classoid),
                    SqlValue::Int4(objoid as i32),
                    SqlValue::Int4(objsubid),
                ]
            })
        })
        .collect();
    (table, rows)
}

fn description_catalog_id(
    catalog: &CatalogSnapshot,
    target: &RelationalCommentTarget,
) -> Option<(i32, u32, i32)> {
    match target {
        RelationalCommentTarget::Schema { schema }
            if schema == "public" && catalog.relational_public_schema_exists =>
        {
            Some((PG_NAMESPACE_CLASS_OID, PG_PUBLIC_NAMESPACE_OID as u32, 0))
        }
        RelationalCommentTarget::Table { table } => catalog
            .relational_catalog
            .get(table)
            .map(|relation| (PG_CLASS_CLASS_OID, relation.oid, 0)),
        RelationalCommentTarget::Column { table, attnum } => catalog
            .relational_catalog
            .get(table)
            .filter(|relation| {
                relation
                    .columns
                    .iter()
                    .any(|column| column.attnum == *attnum)
            })
            .map(|relation| (PG_CLASS_CLASS_OID, relation.oid, i32::from(*attnum))),
        RelationalCommentTarget::Index { index } => {
            catalog_index_oid(catalog, index).map(|oid| (PG_CLASS_CLASS_OID, oid, 0))
        }
        RelationalCommentTarget::View { view } => catalog
            .relational_views
            .get(view)
            .map(|relation| (PG_CLASS_CLASS_OID, relation.oid, 0)),
        RelationalCommentTarget::MaterializedView { materialized_view } => catalog
            .relational_materialized_views
            .get(materialized_view)
            .map(|relation| (PG_CLASS_CLASS_OID, relation.oid, 0)),
        RelationalCommentTarget::Function { function } => catalog
            .relational_functions
            .get(function)
            .map(|function| (PG_PROC_CLASS_OID, function.oid, 0)),
        RelationalCommentTarget::Sequence { sequence } => catalog
            .relational_sequences
            .get(sequence)
            .map(|sequence| (PG_CLASS_CLASS_OID, sequence.oid, 0)),
        RelationalCommentTarget::Domain { domain } => catalog
            .relational_domains
            .get(domain)
            .map(|domain| (PG_TYPE_CLASS_OID, domain.oid, 0)),
        RelationalCommentTarget::Publication { publication } => catalog
            .relational_publications
            .get(publication)
            .map(|publication| (PG_PUBLICATION_CLASS_OID, publication.oid, 0)),
        RelationalCommentTarget::Subscription { subscription } => catalog
            .relational_subscriptions
            .get(subscription)
            .map(|subscription| (PG_SUBSCRIPTION_CLASS_OID, subscription.oid, 0)),
        RelationalCommentTarget::Constraint { table, constraint } => {
            catalog_constraint_oid(catalog, table, constraint)
                .map(|oid| (PG_CONSTRAINT_CLASS_OID, oid, 0))
        }
        RelationalCommentTarget::Extension { extension } if extension == "plpgsql" => {
            Some((PG_EXTENSION_CLASS_OID, 13_500, 0))
        }
        RelationalCommentTarget::Database { .. }
        | RelationalCommentTarget::Role { .. }
        | RelationalCommentTarget::Tablespace { .. }
        | RelationalCommentTarget::Extension { .. }
        | RelationalCommentTarget::Schema { .. } => None,
    }
}
