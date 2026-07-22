//! PostgreSQL 16 dump publication metadata.

use super::*;

#[derive(Clone, Copy)]
pub(super) enum DumpReplicationProgram {
    Publication,
    Relation,
    Namespace,
}

pub(super) fn pg16_dump_replication_program(canonical: &str) -> Option<DumpReplicationProgram> {
    if canonical
        == "select p.tableoid, p.oid, p.pubname, p.pubowner, p.puballtables, p.pubinsert, p.pubupdate, p.pubdelete, p.pubtruncate, p.pubviaroot from pg_publication p"
    {
        Some(DumpReplicationProgram::Publication)
    } else if canonical
        == "select tableoid, oid, prpubid, prrelid, pg_catalog.pg_get_expr(prqual, prrelid) as prrelqual, (case when pr.prattrs is not null then (select array_agg(attname) from pg_catalog.generate_series(0, pg_catalog.array_upper(pr.prattrs::pg_catalog.int2[], 1)) s, pg_catalog.pg_attribute where attrelid = pr.prrelid and attnum = prattrs[s]) else null end) prattrs from pg_catalog.pg_publication_rel pr"
    {
        Some(DumpReplicationProgram::Relation)
    } else if canonical
        == "select tableoid, oid, pnpubid, pnnspid from pg_catalog.pg_publication_namespace"
    {
        Some(DumpReplicationProgram::Namespace)
    } else {
        None
    }
}

impl Engine {
    pub(super) fn execute_pg16_dump_replication_metadata(
        &self,
        kind: DumpReplicationProgram,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (table, rows) = pg16_dump_replication_relation(&catalog, kind);
        self.execute_pg_dump_transient_relation(table, rows, &[], boundary)
    }
}

fn pg16_dump_replication_relation(
    catalog: &CatalogSnapshot,
    kind: DumpReplicationProgram,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    match kind {
        DumpReplicationProgram::Publication => {
            let table = catalog_relation_table(
                "pg_catalog",
                "__pg16_dump_publications",
                &[
                    ("tableoid", SqlType::Int4),
                    ("oid", SqlType::Int4),
                    ("pubname", SqlType::Text),
                    ("pubowner", SqlType::Int4),
                    ("puballtables", SqlType::Bool),
                    ("pubinsert", SqlType::Bool),
                    ("pubupdate", SqlType::Bool),
                    ("pubdelete", SqlType::Bool),
                    ("pubtruncate", SqlType::Bool),
                    ("pubviaroot", SqlType::Bool),
                ],
            );
            let rows = catalog
                .relational_publications
                .values()
                .map(|publication| {
                    vec![
                        SqlValue::Int4(6104),
                        SqlValue::Int4(publication.oid as i32),
                        SqlValue::Text(publication.name.clone()),
                        SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
                        SqlValue::Bool(publication.all_tables),
                        SqlValue::Bool(true),
                        SqlValue::Bool(true),
                        SqlValue::Bool(true),
                        SqlValue::Bool(true),
                        SqlValue::Bool(false),
                    ]
                })
                .collect();
            (table, rows)
        }
        DumpReplicationProgram::Relation => {
            let table = catalog_relation_table(
                "pg_catalog",
                "__pg16_dump_publication_relations",
                &[
                    ("tableoid", SqlType::Int4),
                    ("oid", SqlType::Int4),
                    ("prpubid", SqlType::Int4),
                    ("prrelid", SqlType::Int4),
                    ("prrelqual", SqlType::Text),
                    ("prattrs", SqlType::Text),
                ],
            );
            let mut rows = Vec::new();
            for publication in catalog.relational_publications.values() {
                if publication.all_tables {
                    continue;
                }
                for relation_name in &publication.tables {
                    let Some(relation) = catalog.relational_catalog.get(relation_name) else {
                        continue;
                    };
                    rows.push(vec![
                        SqlValue::Int4(6106),
                        SqlValue::Int4(
                            format!("{}{}", publication.oid, relation.oid)
                                .parse::<i32>()
                                .unwrap_or(0),
                        ),
                        SqlValue::Int4(publication.oid as i32),
                        SqlValue::Int4(relation.oid as i32),
                        SqlValue::Null,
                        SqlValue::Null,
                    ]);
                }
            }
            (table, rows)
        }
        DumpReplicationProgram::Namespace => {
            let table = catalog_relation_table(
                "pg_catalog",
                "__pg16_dump_publication_namespaces",
                &[
                    ("tableoid", SqlType::Int4),
                    ("oid", SqlType::Int4),
                    ("pnpubid", SqlType::Int4),
                    ("pnnspid", SqlType::Int4),
                ],
            );
            let rows = catalog
                .relational_publications
                .values()
                .filter(|publication| publication.all_tables)
                .map(|publication| {
                    vec![
                        SqlValue::Int4(6237),
                        SqlValue::Int4((publication.oid + 100_000) as i32),
                        SqlValue::Int4(publication.oid as i32),
                        SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
                    ]
                })
                .collect();
            (table, rows)
        }
    }
}
