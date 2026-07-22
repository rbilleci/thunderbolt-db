//! PostgreSQL 16 archive metadata for the current database.

use super::*;

pub(super) fn is_pg16_dump_database_metadata_program(canonical: &str) -> bool {
    canonical
        == "select tableoid, oid, datname, datdba, pg_encoding_to_char(encoding) as encoding, datcollate, datctype, datfrozenxid, datacl, acldefault('d', datdba) as acldefault, datistemplate, datconnlimit, datminmxid, datlocprovider, daticulocale, datcollversion, daticurules, (select spcname from pg_tablespace t where t.oid = dattablespace) as tablespace, shobj_description(oid, 'pg_database') as description from pg_database where datname = current_database()"
}

impl Engine {
    pub(super) fn execute_pg16_dump_database_metadata(
        &self,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (table, rows) = pg16_dump_database_metadata_relation(&catalog);
        let predicate = text_comparison(&table, "datname", ResidentBinaryOp::Eq, "postgres")?;
        self.execute_pg_dump_gpu_select(
            table,
            rows,
            SelectProjection::All,
            Some(predicate),
            &[],
            boundary,
        )
    }
}

fn pg16_dump_database_metadata_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__pg16_dump_database_metadata",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("datname", SqlType::Text),
            ("datdba", SqlType::Int4),
            ("encoding", SqlType::Text),
            ("datcollate", SqlType::Text),
            ("datctype", SqlType::Text),
            ("datfrozenxid", SqlType::Text),
            ("datacl", SqlType::Text),
            ("acldefault", SqlType::Text),
            ("datistemplate", SqlType::Bool),
            ("datconnlimit", SqlType::Int4),
            ("datminmxid", SqlType::Text),
            ("datlocprovider", SqlType::Text),
            ("daticulocale", SqlType::Text),
            ("datcollversion", SqlType::Text),
            ("daticurules", SqlType::Text),
            ("tablespace", SqlType::Text),
            ("description", SqlType::Text),
        ],
    );
    let mut databases = catalog
        .relational_databases
        .values()
        .map(|database| (database.name.as_str(), database.oid))
        .collect::<Vec<_>>();
    if !databases.iter().any(|(name, _)| *name == "postgres") {
        databases.push(("postgres", 5));
    }
    let rows = databases
        .into_iter()
        .map(|(name, oid)| {
            let description = catalog
                .relational_comments
                .get(&RelationalCommentTarget::Database {
                    database: name.to_string(),
                })
                .cloned()
                .map_or(SqlValue::Null, SqlValue::Text);
            vec![
                SqlValue::Int4(1262),
                SqlValue::Int4(oid as i32),
                SqlValue::Text(name.to_string()),
                SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
                SqlValue::Text("UTF8".to_string()),
                SqlValue::Text("C.UTF-8".to_string()),
                SqlValue::Text("C.UTF-8".to_string()),
                SqlValue::Text("0".to_string()),
                SqlValue::Null,
                SqlValue::Null,
                SqlValue::Bool(false),
                SqlValue::Int4(-1),
                SqlValue::Text("0".to_string()),
                SqlValue::Text("c".to_string()),
                SqlValue::Null,
                SqlValue::Null,
                SqlValue::Null,
                SqlValue::Text("pg_default".to_string()),
                description,
            ]
        })
        .collect();
    (table, rows)
}
