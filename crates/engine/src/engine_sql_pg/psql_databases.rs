//! PostgreSQL 16 database-list and direct database catalog programs on the GPU path.

use super::*;

const LIST_DATABASES: &str = "select d.datname as \"Name\", pg_catalog.pg_get_userbyid(d.datdba) \
    as \"Owner\", pg_catalog.pg_encoding_to_char(d.encoding) as \"Encoding\", case \
    d.datlocprovider when 'c' then 'libc' when 'i' then 'icu' end as \"Locale Provider\", \
    d.datcollate as \"Collate\", d.datctype as \"Ctype\", d.daticulocale as \"ICU Locale\", \
    d.daticurules as \"ICU Rules\", pg_catalog.array_to_string(d.datacl, e'\\n') as \"Access \
    privileges\" from pg_catalog.pg_database d order by 1";

impl Engine {
    pub(super) fn execute_psql_databases_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        if canonical == LIST_DATABASES || is_verbose_database_list(&canonical) {
            let verbose = is_verbose_database_list(&canonical);
            let names = if verbose {
                vec![
                    "Name",
                    "Owner",
                    "Encoding",
                    "Locale Provider",
                    "Collate",
                    "Ctype",
                    "ICU Locale",
                    "ICU Rules",
                    "Access privileges",
                    "Size",
                    "Tablespace",
                    "Description",
                ]
            } else {
                vec![
                    "Name",
                    "Owner",
                    "Encoding",
                    "Locale Provider",
                    "Collate",
                    "Ctype",
                    "ICU Locale",
                    "ICU Rules",
                    "Access privileges",
                ]
            };
            let table = text_table("__psql_database_list", &names);
            let rows = database_entries(&catalog)
                .into_iter()
                .map(|(oid, name, acl)| {
                    let mut row = vec![
                        SqlValue::Text(name.clone()),
                        SqlValue::Text("postgres".to_string()),
                        SqlValue::Text("UTF8".to_string()),
                        SqlValue::Text("libc".to_string()),
                        SqlValue::Text("C.UTF-8".to_string()),
                        SqlValue::Text("C.UTF-8".to_string()),
                        SqlValue::Null,
                        SqlValue::Null,
                        database_acl_display(acl),
                    ];
                    if verbose {
                        row.extend([
                            SqlValue::Text("0 bytes".to_string()),
                            SqlValue::Text("pg_default".to_string()),
                            database_comment(&catalog, oid, &name),
                        ]);
                    }
                    row
                })
                .collect();
            return self
                .execute_pg_dump_transient_relation(table, rows, &["Name"], boundary)
                .map(Some);
        }
        if canonical == "select oid, datname from pg_catalog.pg_database order by datname" {
            let table = catalog_relation_table(
                "pg_catalog",
                "__database_oids",
                &[("oid", SqlType::Int4), ("datname", SqlType::Text)],
            );
            let rows = database_entries(&catalog)
                .into_iter()
                .map(|(oid, name, _)| vec![SqlValue::Int4(oid as i32), SqlValue::Text(name)])
                .collect();
            return self
                .execute_pg_dump_transient_relation(table, rows, &["datname"], boundary)
                .map(Some);
        }
        if canonical
            == "select datname, pg_catalog.array_to_string(datacl, e'\\n') as acl from \
                pg_catalog.pg_database order by datname"
        {
            let table = text_table("__database_acls", &["datname", "acl"]);
            let rows = database_entries(&catalog)
                .into_iter()
                .map(|(_, name, acl)| vec![SqlValue::Text(name), database_acl_display(acl)])
                .collect();
            return self
                .execute_pg_dump_transient_relation(table, rows, &["datname"], boundary)
                .map(Some);
        }
        Ok(None)
    }
}

fn is_verbose_database_list(canonical: &str) -> bool {
    canonical.starts_with("select d.datname as \"Name\"")
        && canonical.contains("from pg_catalog.pg_database d join pg_catalog.pg_tablespace t")
        && canonical.contains("as \"Access privileges\"")
        && canonical.contains("as \"Size\"")
        && canonical.contains("t.spcname as \"Tablespace\"")
        && canonical.contains("as \"Description\"")
        && canonical.ends_with("order by 1")
}

type DatabaseEntry<'a> = (
    u32,
    String,
    &'a BTreeMap<String, BTreeSet<DatabasePrivilege>>,
);

fn database_entries(catalog: &CatalogSnapshot) -> Vec<DatabaseEntry<'_>> {
    static EMPTY: std::sync::OnceLock<BTreeMap<String, BTreeSet<DatabasePrivilege>>> =
        std::sync::OnceLock::new();
    let mut rows = vec![(5, "postgres".to_string(), EMPTY.get_or_init(BTreeMap::new))];
    rows.extend(
        catalog
            .relational_databases
            .values()
            .map(|database| (database.oid, database.name.clone(), &database.acl)),
    );
    rows
}

fn database_acl_display(acl: &BTreeMap<String, BTreeSet<DatabasePrivilege>>) -> SqlValue {
    let entries = acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            let mut letters = String::new();
            if privileges.contains(&DatabasePrivilege::Connect) {
                letters.push('c');
            }
            if privileges.contains(&DatabasePrivilege::Temporary) {
                letters.push('T');
            }
            (!letters.is_empty()).then(|| {
                let grantee = if grantee == "public" { "" } else { grantee };
                format!("{grantee}={letters}/postgres")
            })
        })
        .collect::<Vec<_>>();
    if entries.is_empty() {
        SqlValue::Null
    } else {
        SqlValue::Text(entries.join("\n"))
    }
}

fn database_comment(catalog: &CatalogSnapshot, oid: u32, name: &str) -> SqlValue {
    let exists = oid == 5 || catalog.relational_databases.contains_key(name);
    if !exists {
        return SqlValue::Null;
    }
    catalog
        .relational_comments
        .get(&RelationalCommentTarget::Database {
            database: name.to_string(),
        })
        .cloned()
        .map_or(SqlValue::Null, SqlValue::Text)
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
