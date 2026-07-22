//! GPU catalog rows for PostgreSQL compatibility views and cross-relation object identities.

use super::*;

const FIRST_USER_INDEX_OID: u32 = 20_000;
const PG_CLASS_CLASS_OID: i32 = 1259;
const PG_CONSTRAINT_CLASS_OID: i32 = 2606;
const PG_NAMESPACE_CLASS_OID: i32 = 2615;
const PG_PROC_CLASS_OID: i32 = 1255;
const PG_PUBLICATION_CLASS_OID: i32 = 6104;
const PG_SUBSCRIPTION_CLASS_OID: i32 = 6100;
const PG_TYPE_CLASS_OID: i32 = 1247;

pub(super) fn synthesize_pg_roles(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_roles",
        &[
            ("oid", SqlType::Int4),
            ("rolname", SqlType::Text),
            ("rolsuper", SqlType::Bool),
            ("rolinherit", SqlType::Bool),
            ("rolcreaterole", SqlType::Bool),
            ("rolcreatedb", SqlType::Bool),
            ("rolcanlogin", SqlType::Bool),
            ("rolconnlimit", SqlType::Int4),
            ("rolpassword", SqlType::Text),
            ("rolvaliduntil", SqlType::Text),
            ("rolreplication", SqlType::Bool),
            ("rolbypassrls", SqlType::Bool),
        ],
    );
    let role_row = |oid: i32, name: String, superuser: bool, login: bool| {
        vec![
            SqlValue::Int4(oid),
            SqlValue::Text(name),
            SqlValue::Bool(superuser),
            SqlValue::Bool(true),
            SqlValue::Bool(false),
            SqlValue::Bool(false),
            SqlValue::Bool(login),
            SqlValue::Int4(-1),
            SqlValue::Null,
            SqlValue::Null,
            SqlValue::Bool(false),
            SqlValue::Bool(false),
        ]
    };
    let mut rows = vec![role_row(
        PG_BOOTSTRAP_OWNER_OID,
        "postgres".to_string(),
        true,
        true,
    )];
    rows.extend(
        catalog
            .relational_roles
            .values()
            .map(|role| role_row(role.oid as i32, role.name.clone(), false, role.login)),
    );
    (table, rows)
}

pub(crate) fn synthesize_pg_tablespace(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_tablespace",
        &[
            ("oid", SqlType::Int4),
            ("spcname", SqlType::Text),
            ("spcowner", SqlType::Int4),
            ("spcacl", SqlType::Text),
            ("spcoptions", SqlType::Text),
        ],
    );
    let built_in = |oid: i32, name: &str| {
        vec![
            SqlValue::Int4(oid),
            SqlValue::Text(name.to_string()),
            SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
            SqlValue::Null,
            SqlValue::Null,
        ]
    };
    let mut rows = vec![built_in(1663, "pg_default"), built_in(1664, "pg_global")];
    rows.extend(catalog.relational_tablespaces.values().map(|tablespace| {
        vec![
            SqlValue::Int4(tablespace.oid as i32),
            SqlValue::Text(tablespace.name.clone()),
            SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
            tablespace_acl_array(&tablespace.acl),
            SqlValue::Null,
        ]
    }));
    (table, rows)
}

fn tablespace_acl_array(acl: &BTreeMap<String, BTreeSet<TablespacePrivilege>>) -> SqlValue {
    let mut entries = acl
        .iter()
        .filter(|(_, privileges)| privileges.contains(&TablespacePrivilege::Create))
        .map(|(grantee, _)| {
            let grantee = if grantee == "public" { "" } else { grantee };
            format!("{grantee}=C/postgres")
        })
        .collect::<Vec<_>>();
    if entries.is_empty() {
        SqlValue::Null
    } else {
        entries.insert(0, "postgres=C/postgres".to_string());
        SqlValue::Text(format!("{{{}}}", entries.join(",")))
    }
}

pub(crate) struct CatalogIndexEntry<'a> {
    pub(crate) table: &'a RelationalTable,
    pub(crate) index: &'a RelationalIndex,
    pub(crate) index_oid: u32,
    pub(crate) attnums: Vec<i16>,
}

pub(crate) fn catalog_index_entries(catalog: &CatalogSnapshot) -> Vec<CatalogIndexEntry<'_>> {
    let mut indexed = catalog
        .relational_catalog
        .values()
        .flat_map(|table| table.indexes.iter().map(move |index| (table, index)))
        .collect::<Vec<_>>();
    indexed.sort_by(|left, right| {
        left.0
            .oid
            .cmp(&right.0.oid)
            .then_with(|| left.1.name.cmp(&right.1.name))
    });
    indexed
        .into_iter()
        .enumerate()
        .filter_map(|(position, (table, index))| {
            let attnums = index
                .key_columns
                .iter()
                .map(|name| {
                    table
                        .columns
                        .iter()
                        .find(|column| column.name == *name)
                        .map(|column| column.attnum)
                })
                .collect::<Option<Vec<_>>>()?;
            Some(CatalogIndexEntry {
                table,
                index,
                index_oid: FIRST_USER_INDEX_OID + position as u32,
                attnums,
            })
        })
        .collect()
}

pub(crate) fn catalog_index_oid(catalog: &CatalogSnapshot, name: &str) -> Option<u32> {
    catalog_index_entries(catalog)
        .into_iter()
        .find(|entry| entry.index.name == name)
        .map(|entry| entry.index_oid)
}

pub(crate) fn catalog_constraint_oid(
    catalog: &CatalogSnapshot,
    table_name: &str,
    constraint_name: &str,
) -> Option<u32> {
    if let Some(entry) = catalog_index_entries(catalog).into_iter().find(|entry| {
        entry.table.name == table_name
            && entry.index.name == constraint_name
            && (entry.index.primary_key || entry.index.unique_constraint)
    }) {
        return Some(40_000 + entry.index_oid);
    }
    let table = catalog.relational_catalog.get(table_name)?;
    if let Some(position) = table
        .check_constraints
        .iter()
        .position(|constraint| constraint.name == constraint_name)
    {
        return Some(60_000 + table.oid + position as u32);
    }
    table
        .foreign_keys
        .iter()
        .position(|constraint| constraint.name == constraint_name)
        .map(|position| 80_000 + table.oid + position as u32)
}

pub(super) fn synthesize_pg_indexes(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_indexes",
        &[
            ("schemaname", SqlType::Text),
            ("tablename", SqlType::Text),
            ("indexname", SqlType::Text),
            ("tablespace", SqlType::Text),
            ("indexdef", SqlType::Text),
        ],
    );
    let rows = catalog_index_entries(catalog)
        .into_iter()
        .map(|entry| {
            let keys = entry.index.key_columns.join(", ");
            vec![
                SqlValue::Text("public".to_string()),
                SqlValue::Text(entry.table.name.clone()),
                SqlValue::Text(entry.index.name.clone()),
                SqlValue::Null,
                SqlValue::Text(format!(
                    "CREATE {}INDEX {} ON public.{} USING btree ({keys})",
                    if entry.index.unique { "UNIQUE " } else { "" },
                    entry.index.name,
                    entry.table.name,
                )),
            ]
        })
        .collect();
    (table, rows)
}

pub(super) fn synthesize_pg_views(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_views",
        &[
            ("schemaname", SqlType::Text),
            ("viewname", SqlType::Text),
            ("viewowner", SqlType::Text),
            ("definition", SqlType::Text),
        ],
    );
    let rows = catalog
        .relational_views
        .values()
        .map(|view| {
            vec![
                SqlValue::Text(view.schema.clone()),
                SqlValue::Text(view.name.clone()),
                SqlValue::Text("postgres".to_string()),
                SqlValue::Text(view.definition.clone()),
            ]
        })
        .collect();
    (table, rows)
}

pub(super) fn synthesize_pg_proc(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_proc",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("proname", SqlType::Text),
            ("pronamespace", SqlType::Int4),
            ("proowner", SqlType::Int4),
            ("prorettype", SqlType::Int4),
            ("prosrc", SqlType::Text),
            ("prolang", SqlType::Int4),
            ("prokind", SqlType::Text),
            ("provolatile", SqlType::Text),
            ("proparallel", SqlType::Text),
            ("prosecdef", SqlType::Bool),
            ("proacl", SqlType::Text),
        ],
    );
    let rows = catalog
        .relational_functions
        .values()
        .map(|function| {
            vec![
                SqlValue::Int4(PG_PROC_CLASS_OID),
                SqlValue::Int4(function.oid as i32),
                SqlValue::Text(function.name.clone()),
                SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
                SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
                SqlValue::Int4(function.return_type.postgres_oid() as i32),
                SqlValue::Text(function.body.clone()),
                SqlValue::Int4(14),
                SqlValue::Text("f".to_string()),
                SqlValue::Text("v".to_string()),
                SqlValue::Text("u".to_string()),
                SqlValue::Bool(false),
                function_acl_array(&function.acl),
            ]
        })
        .collect();
    (table, rows)
}

pub(super) fn synthesize_pg_language() -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_language",
        &[("oid", SqlType::Int4), ("lanname", SqlType::Text)],
    );
    (
        table,
        vec![vec![SqlValue::Int4(14), SqlValue::Text("sql".to_string())]],
    )
}

pub(super) fn synthesize_pg_default_acl(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_default_acl",
        &[
            ("oid", SqlType::Int4),
            ("defaclrole", SqlType::Int4),
            ("defaclnamespace", SqlType::Int4),
            ("defaclobjtype", SqlType::Text),
            ("defaclacl", SqlType::Text),
        ],
    );
    let rows = if catalog.relational_default_table_acl.is_empty() {
        Vec::new()
    } else {
        vec![vec![
            SqlValue::Int4(1),
            SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
            SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
            SqlValue::Text("r".to_string()),
            relation_default_acl_array(&catalog.relational_default_table_acl),
        ]]
    };
    (table, rows)
}

fn relation_acl_entries(acl: &BTreeMap<String, BTreeSet<TablePrivilege>>) -> Vec<String> {
    acl.iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let mut letters = String::new();
            for (privilege, letter) in [
                (TablePrivilege::Insert, 'a'),
                (TablePrivilege::Select, 'r'),
                (TablePrivilege::Update, 'w'),
                (TablePrivilege::Delete, 'd'),
            ] {
                if privileges.contains(&privilege) {
                    letters.push(letter);
                }
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!("{grantee}={letters}/postgres"))
        })
        .collect()
}

fn acl_entries_value(entries: Vec<String>) -> SqlValue {
    if entries.is_empty() {
        SqlValue::Null
    } else {
        SqlValue::Text(format!("{{{}}}", entries.join(",")))
    }
}

pub(super) fn relation_acl_array(
    acl: &BTreeMap<String, BTreeSet<TablePrivilege>>,
    relation_kind: &str,
) -> SqlValue {
    let mut entries = relation_acl_entries(acl);
    if !entries.is_empty() {
        entries.insert(
            0,
            if relation_kind == "S" {
                "postgres=rwU/postgres"
            } else {
                "postgres=arwdDxt/postgres"
            }
            .to_string(),
        );
    }
    acl_entries_value(entries)
}

fn relation_default_acl_array(acl: &BTreeMap<String, BTreeSet<TablePrivilege>>) -> SqlValue {
    acl_entries_value(relation_acl_entries(acl))
}

fn function_acl_array(acl: &BTreeMap<String, BTreeSet<FunctionPrivilege>>) -> SqlValue {
    let entries = acl
        .iter()
        .filter(|(_, privileges)| privileges.contains(&FunctionPrivilege::Execute))
        .map(|(grantee, _)| {
            let grantee = if grantee == "public" { "" } else { grantee };
            format!("{grantee}=X/postgres")
        })
        .collect::<Vec<_>>();
    if entries.is_empty() {
        SqlValue::Null
    } else {
        SqlValue::Text(format!("{{{}}}", entries.join(",")))
    }
}

pub(super) fn synthesize_pg_description(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_description",
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
        RelationalCommentTarget::Database { .. }
        | RelationalCommentTarget::Role { .. }
        | RelationalCommentTarget::Tablespace { .. }
        | RelationalCommentTarget::Extension { .. }
        | RelationalCommentTarget::Schema { .. } => None,
    }
}
