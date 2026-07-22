//! PostgreSQL 16 dump index and key-constraint metadata.

use super::*;

const INDEX_PROGRAM_PREFIX: &str = "select t.tableoid, t.oid, i.indrelid, t.relname as indexname, pg_catalog.pg_get_indexdef(i.indexrelid) as indexdef, i.indkey, i.indisclustered, c.contype, c.conname, c.condeferrable, c.condeferred, c.tableoid as contableoid, c.oid as conoid, pg_catalog.pg_get_constraintdef(c.oid, false) as condef, (select spcname from pg_catalog.pg_tablespace s where s.oid = t.reltablespace) as tablespace, t.reloptions as indreloptions, i.indisreplident, inh.inhparent as parentidx, i.indnkeyatts as indnkeyatts, i.indnatts as indnatts, (select pg_catalog.array_agg(attnum order by attnum) from pg_catalog.pg_attribute where attrelid = i.indexrelid and attstattarget >= 0) as indstatcols, (select pg_catalog.array_agg(attstattarget order by attnum) from pg_catalog.pg_attribute where attrelid = i.indexrelid and attstattarget >= 0) as indstatvals, i.indnullsnotdistinct from unnest('{";
const INDEX_PROGRAM_SUFFIX: &str = "}'::pg_catalog.oid[]) as src(tbloid) join pg_catalog.pg_index i on (src.tbloid = i.indrelid) join pg_catalog.pg_class t on (t.oid = i.indexrelid) join pg_catalog.pg_class t2 on (t2.oid = i.indrelid) left join pg_catalog.pg_constraint c on (i.indrelid = c.conrelid and i.indexrelid = c.conindid and c.contype in ('p','u','x')) left join pg_catalog.pg_inherits inh on (inh.inhrelid = indexrelid) where (i.indisvalid or t2.relkind = 'p') and i.indisready order by i.indrelid, indexname";

const FOREIGN_KEY_PROGRAM_PREFIX: &str = "select c.tableoid, c.oid, conrelid, conname, confrelid, conindid, pg_catalog.pg_get_constraintdef(c.oid) as condef from unnest('{";
const FOREIGN_KEY_PROGRAM_SUFFIX: &str = "}'::pg_catalog.oid[]) as src(tbloid) join pg_catalog.pg_constraint c on (src.tbloid = c.conrelid) where contype = 'f' and conparentid = 0 order by conrelid, conname";

pub(super) fn pg16_dump_index_metadata_program_oids(canonical: &str) -> Option<Vec<u32>> {
    pg16_dump_oid_array_program_values(canonical, INDEX_PROGRAM_PREFIX, INDEX_PROGRAM_SUFFIX)
}

pub(super) fn pg16_dump_foreign_key_metadata_program_oids(canonical: &str) -> Option<Vec<u32>> {
    pg16_dump_oid_array_program_values(
        canonical,
        FOREIGN_KEY_PROGRAM_PREFIX,
        FOREIGN_KEY_PROGRAM_SUFFIX,
    )
}

impl Engine {
    pub(super) fn execute_pg16_dump_index_metadata(
        &self,
        relation_oids: &[u32],
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (requested, requested_rows) = oid_source_relation(relation_oids);
        let (indexes, index_rows) = pg16_dump_index_source_relation(&catalog);
        let (classes, class_rows) = pg16_dump_class_identity_relation(&catalog);
        let (constraints, constraint_rows) = pg16_dump_constraint_relation(&catalog);
        let inherits = catalog_relation_table(
            "pg_catalog",
            "pg_inherits",
            &[("inhrelid", SqlType::Int4), ("inhparent", SqlType::Int4)],
        );

        let predicate = and(
            bool_comparison(&indexes, "indisvalid", true)?,
            bool_comparison(&indexes, "indisready", true)?,
        );
        let constraint_step = JoinStep {
            conjuncts: vec![
                (join_column("i", "indexrelid"), join_column("c", "conindid")),
                (
                    join_column("i", "__constraint_kind_code"),
                    join_column("c", "__contype_code"),
                ),
            ],
            natural: false,
            coalesce: Vec::new(),
            outer_left: true,
            outer_right: false,
        };
        let projection = vec![
            ("t", "tableoid", "tableoid"),
            ("t", "oid", "oid"),
            ("i", "indrelid", "indrelid"),
            ("t", "relname", "indexname"),
            ("i", "indexdef", "indexdef"),
            ("i", "indkey", "indkey"),
            ("i", "indisclustered", "indisclustered"),
            ("c", "contype", "contype"),
            ("c", "conname", "conname"),
            ("c", "condeferrable", "condeferrable"),
            ("c", "condeferred", "condeferred"),
            ("c", "tableoid", "contableoid"),
            ("c", "oid", "conoid"),
            ("c", "condef", "condef"),
            ("i", "tablespace", "tablespace"),
            ("i", "indreloptions", "indreloptions"),
            ("i", "indisreplident", "indisreplident"),
            ("inh", "inhparent", "parentidx"),
            ("i", "indnkeyatts", "indnkeyatts"),
            ("i", "indnatts", "indnatts"),
            ("i", "indstatcols", "indstatcols"),
            ("i", "indstatvals", "indstatvals"),
            ("i", "indnullsnotdistinct", "indnullsnotdistinct"),
        ];
        let plan = join_plan(
            &[
                (&requested, "src"),
                (&indexes, "i"),
                (&classes, "t"),
                (&classes, "t2"),
                (&constraints, "c"),
                (&inherits, "inh"),
            ],
            vec![
                join_step("src", "tbloid", "i", "indrelid", false),
                join_step("i", "indexrelid", "t", "oid", false),
                join_step("i", "indrelid", "t2", "oid", false),
                constraint_step,
                join_step("i", "indexrelid", "inh", "inhrelid", true),
            ],
            projection,
            vec![("i", "indrelid"), ("t", "relname")],
        );
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![
                requested,
                indexes,
                classes.clone(),
                classes,
                constraints,
                inherits,
            ],
            vec![
                requested_rows,
                index_rows,
                class_rows.clone(),
                class_rows,
                constraint_rows,
                Vec::new(),
            ],
            vec![None, Some(predicate), None, None, None, None],
            boundary,
        )
    }

    pub(super) fn execute_pg16_dump_foreign_key_metadata(
        &self,
        relation_oids: &[u32],
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (requested, requested_rows) = oid_source_relation(relation_oids);
        let (constraints, constraint_rows) = pg16_dump_constraint_relation(&catalog);
        let predicate = and(
            text_comparison(&constraints, "contype", ResidentBinaryOp::Eq, "f")?,
            int4_comparison(&constraints, "conparentid", ResidentBinaryOp::Eq, 0)?,
        );
        let projection = [
            "tableoid",
            "oid",
            "conrelid",
            "conname",
            "confrelid",
            "conindid",
            "condef",
        ]
        .into_iter()
        .map(|column| ("c", column, column))
        .collect();
        let plan = join_plan(
            &[(&requested, "src"), (&constraints, "c")],
            vec![join_step("src", "tbloid", "c", "conrelid", false)],
            projection,
            vec![("c", "conrelid"), ("c", "conname")],
        );
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![requested, constraints],
            vec![requested_rows, constraint_rows],
            vec![None, Some(predicate)],
            boundary,
        )
    }
}

fn pg16_dump_class_identity_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_class",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("relname", SqlType::Text),
            ("relkind", SqlType::Text),
        ],
    );
    let row = |oid: u32, name: &str, kind: &str| {
        vec![
            SqlValue::Int4(1259),
            SqlValue::Int4(oid as i32),
            SqlValue::Text(name.to_string()),
            SqlValue::Text(kind.to_string()),
        ]
    };
    let mut rows = catalog
        .relational_catalog
        .values()
        .map(|relation| row(relation.oid, &relation.name, "r"))
        .collect::<Vec<_>>();
    rows.extend(
        catalog
            .relational_views
            .values()
            .map(|relation| row(relation.oid, &relation.name, "v")),
    );
    rows.extend(
        catalog
            .relational_materialized_views
            .values()
            .map(|relation| row(relation.oid, &relation.name, "m")),
    );
    rows.extend(
        catalog
            .relational_sequences
            .values()
            .map(|relation| row(relation.oid, &relation.name, "S")),
    );
    rows.extend(
        catalog_index_entries(catalog)
            .into_iter()
            .map(|entry| row(entry.index_oid, &entry.index.name, "i")),
    );
    (table, rows)
}

fn pg16_dump_index_source_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_index",
        &[
            ("indexrelid", SqlType::Int4),
            ("indrelid", SqlType::Int4),
            ("indexdef", SqlType::Text),
            ("indkey", SqlType::Text),
            ("indisclustered", SqlType::Bool),
            ("tablespace", SqlType::Text),
            ("indreloptions", SqlType::Text),
            ("indisreplident", SqlType::Bool),
            ("indnkeyatts", SqlType::Int4),
            ("indnatts", SqlType::Int4),
            ("indstatcols", SqlType::Text),
            ("indstatvals", SqlType::Text),
            ("indnullsnotdistinct", SqlType::Bool),
            ("indisvalid", SqlType::Bool),
            ("indisready", SqlType::Bool),
            ("__constraint_kind_code", SqlType::Int4),
        ],
    );
    let rows = catalog_index_entries(catalog)
        .into_iter()
        .map(|entry| {
            let keys = entry.index.key_columns.join(", ");
            vec![
                SqlValue::Int4(entry.index_oid as i32),
                SqlValue::Int4(entry.table.oid as i32),
                SqlValue::Text(format!(
                    "CREATE {}INDEX {} ON public.{} USING btree ({keys})",
                    if entry.index.unique { "UNIQUE " } else { "" },
                    entry.index.name,
                    entry.table.name,
                )),
                SqlValue::Text(
                    entry
                        .attnums
                        .iter()
                        .map(i16::to_string)
                        .collect::<Vec<_>>()
                        .join(" "),
                ),
                SqlValue::Bool(false),
                SqlValue::Null,
                SqlValue::Null,
                SqlValue::Bool(false),
                SqlValue::Int4(entry.attnums.len() as i32),
                SqlValue::Int4(entry.attnums.len() as i32),
                SqlValue::Null,
                SqlValue::Null,
                SqlValue::Bool(false),
                SqlValue::Bool(true),
                SqlValue::Bool(true),
                SqlValue::Int4(if entry.index.primary_key {
                    1
                } else if entry.index.unique_constraint {
                    2
                } else {
                    0
                }),
            ]
        })
        .collect();
    (table, rows)
}

fn pg16_dump_constraint_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_constraint",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("conrelid", SqlType::Int4),
            ("conname", SqlType::Text),
            ("confrelid", SqlType::Int4),
            ("conindid", SqlType::Int4),
            ("contype", SqlType::Text),
            ("condeferrable", SqlType::Bool),
            ("condeferred", SqlType::Bool),
            ("conparentid", SqlType::Int4),
            ("condef", SqlType::Text),
            ("__contype_code", SqlType::Int4),
        ],
    );
    let indexes = catalog_index_entries(catalog);
    let mut rows = Vec::new();
    for entry in &indexes {
        if !(entry.index.primary_key || entry.index.unique_constraint) {
            continue;
        }
        let kind = if entry.index.primary_key { "p" } else { "u" };
        let label = if entry.index.primary_key {
            "PRIMARY KEY"
        } else {
            "UNIQUE"
        };
        rows.push(vec![
            SqlValue::Int4(2606),
            SqlValue::Int4((40_000 + entry.index_oid) as i32),
            SqlValue::Int4(entry.table.oid as i32),
            SqlValue::Text(entry.index.name.clone()),
            SqlValue::Int4(0),
            SqlValue::Int4(entry.index_oid as i32),
            SqlValue::Text(kind.to_string()),
            SqlValue::Bool(false),
            SqlValue::Bool(false),
            SqlValue::Int4(0),
            SqlValue::Text(format!("{label} ({})", entry.index.key_columns.join(", "))),
            SqlValue::Int4(if entry.index.primary_key { 1 } else { 2 }),
        ]);
    }
    for relation in catalog.relational_catalog.values() {
        for (position, foreign_key) in relation.foreign_keys.iter().enumerate() {
            let referenced_oid = catalog
                .relational_catalog
                .get(&foreign_key.referenced_table)
                .map_or(0, |referenced| referenced.oid);
            let referenced_index_oid = indexes
                .iter()
                .find(|entry| {
                    entry.table.name == foreign_key.referenced_table
                        && entry
                            .index
                            .key_columns
                            .first()
                            .is_some_and(|column| column == &foreign_key.referenced_column)
                        && (entry.index.primary_key || entry.index.unique_constraint)
                })
                .map_or(0, |entry| entry.index_oid);
            rows.push(vec![
                SqlValue::Int4(2606),
                SqlValue::Int4((80_000 + relation.oid + position as u32) as i32),
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Text(foreign_key.name.clone()),
                SqlValue::Int4(referenced_oid as i32),
                SqlValue::Int4(referenced_index_oid as i32),
                SqlValue::Text("f".to_string()),
                SqlValue::Bool(false),
                SqlValue::Bool(false),
                SqlValue::Int4(0),
                SqlValue::Text(format!(
                    "FOREIGN KEY ({}) REFERENCES {}({})",
                    foreign_key.column, foreign_key.referenced_table, foreign_key.referenced_column,
                )),
                SqlValue::Int4(4),
            ]);
        }
    }
    (table, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oid_array_matchers_return_values_and_reject_shape_changes() {
        let index = format!("{INDEX_PROGRAM_PREFIX}11,22{INDEX_PROGRAM_SUFFIX}");
        assert_eq!(
            pg16_dump_index_metadata_program_oids(&index),
            Some(vec![11, 22])
        );
        let foreign = format!("{FOREIGN_KEY_PROGRAM_PREFIX}33{FOREIGN_KEY_PROGRAM_SUFFIX}");
        assert_eq!(
            pg16_dump_foreign_key_metadata_program_oids(&foreign),
            Some(vec![33])
        );
        assert!(
            pg16_dump_index_metadata_program_oids(&index.replace("indisready", "indisvalid"))
                .is_none()
        );
    }
}
