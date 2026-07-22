//! Synthesized GPU-catalog relation descriptors and upload rows.
//!
//! This module owns deterministic catalog encoding only. Consumers bind and execute these
//! relations through the resident/transient GPU relational paths.

use super::*;

mod compatibility;
pub(crate) use compatibility::{
    catalog_constraint_oid, catalog_index_entries, catalog_index_oid, synthesize_pg_tablespace,
};
use compatibility::{
    relation_acl_array, synthesize_pg_default_acl, synthesize_pg_description,
    synthesize_pg_indexes, synthesize_pg_language, synthesize_pg_proc, synthesize_pg_roles,
    synthesize_pg_views,
};

pub(crate) const GPU_CATALOG_RELKIND_DISPLAY: &str = "__gpu_relkind_display";
pub(crate) const GPU_CATALOG_OWNER_NAME: &str = "__gpu_owner_name";
pub(crate) const GPU_CATALOG_FALSE: &str = "__gpu_false";
pub(crate) const GPU_CATALOG_EMPTY_TEXT: &str = "__gpu_empty_text";
pub(crate) const GPU_CATALOG_RELTYPE_DISPLAY: &str = "__gpu_reltype_display";
pub(crate) const GPU_CATALOG_FORMATTED_TYPE: &str = "__gpu_formatted_type";
pub(crate) const GPU_CATALOG_DEFAULT_EXPR: &str = "__gpu_default_expr";
pub(crate) const GPU_CATALOG_COLLATION_NAME: &str = "__gpu_collation_name";
pub(crate) const GPU_CATALOG_CONSTRAINT_DEF: &str = "__gpu_constraint_def";
pub(crate) const GPU_CATALOG_ACL_DEFAULT: &str = "__gpu_acl_default";
pub(crate) const GPU_CATALOG_FUNCTION_RESULT: &str = "__gpu_function_result";
pub(crate) const GPU_CATALOG_NULLABLE_DISPLAY: &str = "__gpu_nullable_display";
pub(crate) const GPU_CATALOG_DOMAIN_CHECK: &str = "__gpu_domain_check";
pub(crate) const GPU_CATALOG_ACL_DISPLAY: &str = "__gpu_acl_display";
pub(crate) const GPU_CATALOG_DESCRIPTION: &str = "__gpu_description";
pub(crate) const GPU_CATALOG_RELKIND_ACL_DISPLAY: &str = "__gpu_relkind_acl_display";
pub(crate) const GPU_CATALOG_FUNCTION_KIND: &str = "__gpu_function_kind";
pub(crate) const GPU_CATALOG_FUNCTION_VOLATILITY: &str = "__gpu_function_volatility";
pub(crate) const GPU_CATALOG_FUNCTION_PARALLEL: &str = "__gpu_function_parallel";
pub(crate) const GPU_CATALOG_FUNCTION_SECURITY: &str = "__gpu_function_security";
pub(crate) const GPU_CATALOG_FUNCTION_INTERNAL_NAME: &str = "__gpu_function_internal_name";
pub(crate) const GPU_CATALOG_DEFAULT_ACL_TYPE: &str = "__gpu_default_acl_type";
pub(crate) const GPU_CATALOG_COLUMN_PRIVILEGES: &str = "__gpu_column_privileges";
pub(crate) const GPU_CATALOG_POLICY_DISPLAY: &str = "__gpu_policy_display";
pub(crate) const GPU_CATALOG_CURRENT_USER_MATCH: &str = "__gpu_current_user_match";
pub(crate) const GPU_CATALOG_TABLESPACE_LOCATION: &str = "__gpu_tablespace_location";
pub(crate) const GPU_CATALOG_TABLESPACE_SIZE: &str = "__gpu_tablespace_size";

fn sql_pg_error(message: String) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message))
}

// ============================================================================
// Phase-3 M2 — engine-native pg_catalog / information_schema (single-relation).
//
// A SELECT against a catalog relation (qualified `pg_catalog.pg_class`, or bare
// `pg_class` via the implicit pg_catalog search path) is answered by synthesizing the
// relation's rows from the pinned `CatalogSnapshot` and running them through the SAME
// bind -> filter -> project -> order/limit pipeline as a user table, so introspection is
// MVCC-consistent and reuses the relational core. Multi-relation JOIN catalog queries
// (psql's `\d` family) are NOT handled here — the executor has no joins yet (a later
// milestone); this is the single-relation data layer.
//
// Catalog columns whose PostgreSQL type the engine lacks (`oid`, `name`, `char`) are
// mapped to the nearest engine type (oid -> int4, name/char -> text) — the VALUES are
// faithful. NULL-bearing columns are omitted until the value model gains NULL (M3).
// ============================================================================

/// The OID PostgreSQL assigns the `public` schema — the one namespace every engine
/// relation lives in (the engine models only the `public` user schema).
pub(crate) const PG_PUBLIC_NAMESPACE_OID: i32 = 2200;
/// The `pg_catalog` system-schema OID (fixed in PostgreSQL).
pub(crate) const PG_CATALOG_NAMESPACE_OID: i32 = 11;
/// A representative `information_schema` namespace OID (its real OID varies per cluster).
pub(crate) const PG_INFORMATION_SCHEMA_NAMESPACE_OID: i32 = 13183;
/// The bootstrap superuser OID used as every relation/namespace owner.
pub(crate) const PG_BOOTSTRAP_OWNER_OID: i32 = 10;
/// The built-in heap table access method OID.
pub(crate) const PG_HEAP_AM_OID: i32 = 2;

/// Built-in relation names whose implicit `pg_catalog` lookup precedes the modeled public schema.
/// This includes every relation synthesized by this GPU catalog slice plus the binding-only
/// relations referenced by its supported psql shapes.
pub(crate) const MODELED_PG_CATALOG_RELATION_NAMES: &[&str] = &[
    "pg_am",
    "pg_attrdef",
    "pg_attribute",
    "pg_class",
    "pg_collation",
    "pg_constraint",
    "pg_extension",
    "pg_description",
    "pg_default_acl",
    "pg_inherits",
    "pg_indexes",
    "pg_language",
    "pg_namespace",
    "pg_policy",
    "pg_proc",
    "pg_publication",
    "pg_publication_namespace",
    "pg_publication_rel",
    "pg_publication_tables",
    "pg_roles",
    "pg_settings",
    "pg_statistic_ext",
    "pg_tablespace",
    "pg_trigger",
    "pg_type",
    "pg_views",
];

/// Build a transient in-memory [`RelationalTable`] describing a catalog relation's fixed
/// columns. Only `columns` (and a nominal `oid`) are consulted by the bind/finalize path.
pub(crate) fn catalog_relation_table(
    schema: &str,
    name: &str,
    columns: &[(&str, SqlType)],
) -> RelationalTable {
    let columns = columns
        .iter()
        .enumerate()
        .map(|(idx, (col_name, ty))| RelationalColumn {
            id: 0,
            table_oid: 0,
            attnum: (idx + 1) as i16,
            name: (*col_name).to_string(),
            ty: *ty,
            domain: None,
            default: None,
            type_oid: ty.postgres_oid(),
            type_size: ty.type_size(),
        })
        .collect();
    RelationalTable {
        schema: schema.to_string(),
        name: name.to_string(),
        oid: 0,
        columns,
        indexes: Vec::new(),
        check_constraints: Vec::new(),
        foreign_keys: Vec::new(),
        acl: BTreeMap::new(),
    }
}

/// The `information_schema.columns.data_type` spelling for an engine type (PostgreSQL uses
/// the SQL-standard names here, e.g. `integer`/`bigint`, not the `pg_type` names).
pub(crate) fn information_schema_data_type(ty: SqlType) -> &'static str {
    match ty {
        SqlType::Int2 => "smallint",
        SqlType::Int4 => "integer",
        SqlType::Int8 => "bigint",
        SqlType::Numeric { .. } => "numeric",
        SqlType::Bool => "boolean",
        SqlType::Text => "text",
        SqlType::Date => "date",
        SqlType::Timestamp => "timestamp",
        SqlType::Uuid => "uuid",
    }
}

/// `pg_catalog.pg_namespace` — one row per schema. The engine models the `public` user
/// schema (when present) plus the always-exposed `pg_catalog`/`information_schema`.
pub(crate) fn synthesize_pg_namespace(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_namespace",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("nspname", SqlType::Text),
            ("nspowner", SqlType::Int4),
            ("nspacl", SqlType::Text),
        ],
    );
    let owner = SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID);
    let mut rows = vec![
        vec![
            SqlValue::Int4(2615),
            SqlValue::Int4(PG_CATALOG_NAMESPACE_OID),
            SqlValue::Text("pg_catalog".to_string()),
            owner.clone(),
            SqlValue::Null,
        ],
        vec![
            SqlValue::Int4(2615),
            SqlValue::Int4(PG_INFORMATION_SCHEMA_NAMESPACE_OID),
            SqlValue::Text("information_schema".to_string()),
            owner.clone(),
            SqlValue::Null,
        ],
    ];
    if catalog.relational_public_schema_exists {
        rows.push(vec![
            SqlValue::Int4(2615),
            SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
            SqlValue::Text("public".to_string()),
            owner,
            schema_acl_array_value(catalog),
        ]);
    }
    (table, rows)
}

fn schema_acl_array_value(catalog: &CatalogSnapshot) -> SqlValue {
    if catalog.relational_schema_acl.is_empty() {
        return SqlValue::Null;
    }
    let mut entries = vec![
        "postgres=UC/postgres".to_string(),
        "=U/postgres".to_string(),
    ];
    entries.extend(
        catalog
            .relational_schema_acl
            .iter()
            .filter_map(|(grantee, privileges)| {
                if privileges.is_empty() {
                    return None;
                }
                let mut letters = String::new();
                if privileges.contains(&SchemaPrivilege::Usage) {
                    letters.push('U');
                }
                if privileges.contains(&SchemaPrivilege::Create) {
                    letters.push('C');
                }
                let grantee = if grantee == "public" { "" } else { grantee };
                Some(format!("{grantee}={letters}/postgres"))
            }),
    );
    SqlValue::Text(format!("{{{}}}", entries.join(",")))
}

/// `pg_catalog.pg_class` — one row per relation (table `r`, view `v`, materialized view
/// `m`, sequence `S`), synthesized from the catalog's relation maps.
pub(crate) fn synthesize_pg_class(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_class",
        &[
            ("oid", SqlType::Int4),
            ("relname", SqlType::Text),
            ("relnamespace", SqlType::Int4),
            ("relkind", SqlType::Text),
            ("relnatts", SqlType::Int4),
            ("relowner", SqlType::Int4),
            ("relhasindex", SqlType::Bool),
            // PG `char`: 'p' permanent / 'u' unlogged / 't' temp. All synthesized relations are permanent.
            ("relpersistence", SqlType::Text),
            ("relam", SqlType::Int4),
            ("relchecks", SqlType::Int4),
            ("relhasrules", SqlType::Bool),
            ("relhastriggers", SqlType::Bool),
            ("relrowsecurity", SqlType::Bool),
            ("relforcerowsecurity", SqlType::Bool),
            ("relispartition", SqlType::Bool),
            ("reltablespace", SqlType::Int4),
            ("reloftype", SqlType::Int4),
            ("relreplident", SqlType::Text),
            ("reltoastrelid", SqlType::Int4),
        ],
    );
    let namespace = SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID);
    let owner = SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID);
    let row = |oid: u32, name: &str, kind: &str, natts: usize, has_index: bool| {
        vec![
            SqlValue::Int4(oid as i32),
            SqlValue::Text(name.to_string()),
            namespace.clone(),
            SqlValue::Text(kind.to_string()),
            SqlValue::Int4(natts as i32),
            owner.clone(),
            SqlValue::Bool(has_index),
            SqlValue::Text("p".to_string()),
            SqlValue::Int4(PG_HEAP_AM_OID),
            SqlValue::Int4(0),
            SqlValue::Bool(false),
            SqlValue::Bool(false),
            SqlValue::Bool(false),
            SqlValue::Bool(false),
            SqlValue::Bool(false),
            SqlValue::Int4(0),
            SqlValue::Int4(0),
            SqlValue::Text("d".to_string()),
            SqlValue::Int4(0),
        ]
    };
    let mut rows = Vec::new();
    for t in catalog.relational_catalog.values() {
        rows.push(row(
            t.oid,
            &t.name,
            "r",
            t.columns.len(),
            !t.indexes.is_empty(),
        ));
    }
    for v in catalog.relational_views.values() {
        rows.push(row(v.oid, &v.name, "v", 0, false));
    }
    for mv in catalog.relational_materialized_views.values() {
        rows.push(row(mv.oid, &mv.name, "m", mv.columns.len(), false));
    }
    for s in catalog.relational_sequences.values() {
        rows.push(row(s.oid, &s.name, "S", 0, false));
    }
    for index in catalog_index_entries(catalog) {
        rows.push(row(
            index.index_oid,
            &index.index.name,
            "i",
            index.attnums.len(),
            false,
        ));
    }
    (table, rows)
}

/// `pg_catalog.pg_am` — the heap access method used by synthesized user relations.
pub(crate) fn synthesize_pg_am() -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_am",
        &[("oid", SqlType::Int4), ("amname", SqlType::Text)],
    );
    (
        table,
        vec![vec![
            SqlValue::Int4(PG_HEAP_AM_OID),
            SqlValue::Text("heap".to_string()),
        ]],
    )
}

/// `pg_catalog.pg_constraint` — table-owned key/check/foreign-key metadata.
pub(crate) fn synthesize_pg_constraint(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_constraint",
        &[
            ("oid", SqlType::Int4),
            ("conname", SqlType::Text),
            ("conrelid", SqlType::Int4),
            ("contype", SqlType::Text),
            ("conparentid", SqlType::Int4),
            ("confrelid", SqlType::Int4),
        ],
    );
    let mut rows = Vec::new();
    for relation in catalog.relational_catalog.values() {
        for index in &relation.indexes {
            if !(index.primary_key || index.unique_constraint) {
                continue;
            }
            let oid = catalog_constraint_oid(catalog, &relation.name, &index.name)
                .expect("catalog key constraint has a deterministic OID");
            rows.push(vec![
                SqlValue::Int4(oid as i32),
                SqlValue::Text(index.name.clone()),
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Text(if index.primary_key { "p" } else { "u" }.to_string()),
                SqlValue::Int4(0),
                SqlValue::Int4(0),
            ]);
        }
        for constraint in &relation.check_constraints {
            let oid = catalog_constraint_oid(catalog, &relation.name, &constraint.name)
                .expect("catalog check constraint has a deterministic OID");
            rows.push(vec![
                SqlValue::Int4(oid as i32),
                SqlValue::Text(constraint.name.clone()),
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Text("c".to_string()),
                SqlValue::Int4(0),
                SqlValue::Int4(0),
            ]);
        }
        for constraint in &relation.foreign_keys {
            let referenced_oid = catalog
                .relational_catalog
                .get(&constraint.referenced_table)
                .map_or(0, |table| table.oid as i32);
            let oid = catalog_constraint_oid(catalog, &relation.name, &constraint.name)
                .expect("catalog foreign key has a deterministic OID");
            rows.push(vec![
                SqlValue::Int4(oid as i32),
                SqlValue::Text(constraint.name.clone()),
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Text("f".to_string()),
                SqlValue::Int4(0),
                SqlValue::Int4(referenced_oid),
            ]);
        }
    }
    (table, rows)
}

fn synthesize_empty_catalog_table(
    name: &str,
    columns: &[(&str, SqlType)],
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    (
        catalog_relation_table("pg_catalog", name, columns),
        Vec::new(),
    )
}

/// `pg_catalog.pg_attribute` — one row per user column. `attrelid` links to pg_class.oid;
/// `atttypid`/`attlen` come from the column's resolved type. Nullability isn't modeled yet
/// (M3), so `attnotnull` is always false.
pub(crate) fn synthesize_pg_attribute(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_attribute",
        &[
            ("attrelid", SqlType::Int4),
            ("attname", SqlType::Text),
            ("atttypid", SqlType::Int4),
            ("attnum", SqlType::Int4),
            ("attlen", SqlType::Int4),
            ("attnotnull", SqlType::Bool),
            // -1 = no type modifier (PG's atttypmod for unparameterized types); numeric typmod is a
            // follow-up alongside format_type. attisdropped is always false (no column drops yet).
            ("atttypmod", SqlType::Int4),
            ("attisdropped", SqlType::Bool),
            ("atthasdef", SqlType::Bool),
            ("attcollation", SqlType::Int4),
            ("attidentity", SqlType::Text),
            ("attgenerated", SqlType::Text),
        ],
    );
    let attribute_row = |oid: u32, col: &RelationalColumn| {
        vec![
            SqlValue::Int4(oid as i32),
            SqlValue::Text(col.name.clone()),
            SqlValue::Int4(col.type_oid as i32),
            SqlValue::Int4(i32::from(col.attnum)),
            SqlValue::Int4(i32::from(col.type_size)),
            SqlValue::Bool(false),
            SqlValue::Int4(-1),
            SqlValue::Bool(false),
            SqlValue::Bool(col.default.is_some()),
            SqlValue::Int4(0),
            SqlValue::Text(String::new()),
            SqlValue::Text(String::new()),
        ]
    };
    let mut rows = Vec::new();
    for t in catalog.relational_catalog.values() {
        for col in &t.columns {
            rows.push(attribute_row(t.oid, col));
        }
    }
    for mv in catalog.relational_materialized_views.values() {
        for col in &mv.columns {
            rows.push(attribute_row(mv.oid, col));
        }
    }
    (table, rows)
}

/// `pg_catalog.pg_type` — the engine's fixed base types plus any user domains.
pub(crate) fn synthesize_pg_type(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_type",
        &[
            ("oid", SqlType::Int4),
            ("typname", SqlType::Text),
            ("typlen", SqlType::Int4),
            ("typtype", SqlType::Text),
            ("typnamespace", SqlType::Int4),
            ("typbasetype", SqlType::Int4),
        ],
    );
    let base = |ty: SqlType| {
        vec![
            SqlValue::Int4(ty.postgres_oid() as i32),
            SqlValue::Text(ty.catalog_name().to_string()),
            SqlValue::Int4(i32::from(ty.type_size())),
            SqlValue::Text("b".to_string()),
            SqlValue::Int4(PG_CATALOG_NAMESPACE_OID),
            SqlValue::Int4(0),
        ]
    };
    let mut rows = vec![
        base(SqlType::Int4),
        base(SqlType::Int8),
        base(SqlType::Numeric {
            precision: NUMERIC_DEFAULT_PRECISION,
            scale: 0,
        }),
        base(SqlType::Bool),
        base(SqlType::Text),
    ];
    for domain in catalog.relational_domains.values() {
        rows.push(vec![
            SqlValue::Int4(domain.oid as i32),
            SqlValue::Text(domain.name.clone()),
            SqlValue::Int4(i32::from(domain.base_type.type_size())),
            SqlValue::Text("d".to_string()),
            SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
            SqlValue::Int4(domain.base_type.postgres_oid() as i32),
        ]);
    }
    (table, rows)
}

/// `information_schema.tables` — one row per user table (`BASE TABLE`) and view (`VIEW`) in
/// `public`. (Materialized views are not part of `information_schema.tables` in PostgreSQL.)
pub(crate) fn synthesize_information_schema_tables(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "information_schema",
        "tables",
        &[
            ("table_catalog", SqlType::Text),
            ("table_schema", SqlType::Text),
            ("table_name", SqlType::Text),
            ("table_type", SqlType::Text),
        ],
    );
    let row = |name: &str, table_type: &str| {
        vec![
            SqlValue::Text("postgres".to_string()),
            SqlValue::Text("public".to_string()),
            SqlValue::Text(name.to_string()),
            SqlValue::Text(table_type.to_string()),
        ]
    };
    let mut rows = Vec::new();
    for t in catalog.relational_catalog.values() {
        rows.push(row(&t.name, "BASE TABLE"));
    }
    for v in catalog.relational_views.values() {
        rows.push(row(&v.name, "VIEW"));
    }
    (table, rows)
}

/// `information_schema.columns` — one row per user column. `data_type` uses the SQL-standard
/// spelling; `is_nullable` is always `YES` until the value model gains NULL (M3).
pub(crate) fn synthesize_information_schema_columns(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "information_schema",
        "columns",
        &[
            ("table_catalog", SqlType::Text),
            ("table_schema", SqlType::Text),
            ("table_name", SqlType::Text),
            ("column_name", SqlType::Text),
            ("ordinal_position", SqlType::Int4),
            ("data_type", SqlType::Text),
            ("is_nullable", SqlType::Text),
            ("udt_schema", SqlType::Text),
            ("udt_name", SqlType::Text),
        ],
    );
    let mut rows = Vec::new();
    for t in catalog.relational_catalog.values() {
        for col in &t.columns {
            let (data_type, udt_schema, udt_name) = match &col.domain {
                Some(domain) => (domain.as_str(), "public", domain.as_str()),
                None => (
                    information_schema_data_type(col.ty),
                    "pg_catalog",
                    col.ty.catalog_name(),
                ),
            };
            rows.push(vec![
                SqlValue::Text("postgres".to_string()),
                SqlValue::Text("public".to_string()),
                SqlValue::Text(t.name.clone()),
                SqlValue::Text(col.name.clone()),
                SqlValue::Int4(i32::from(col.attnum)),
                SqlValue::Text(data_type.to_string()),
                SqlValue::Text("YES".to_string()),
                SqlValue::Text(udt_schema.to_string()),
                SqlValue::Text(udt_name.to_string()),
            ]);
        }
    }
    (table, rows)
}

/// Whether the bare name is already owned by any relation in the supported public namespace.
///
/// Catalog synthesis is search-path fallback, so every public relation kind must shadow a bare
/// `pg_*` name even when a particular execution route cannot consume that kind. Such a route must
/// fail closed instead of silently substituting the synthesized system relation.
pub(crate) fn public_relation_name_exists(catalog: &CatalogSnapshot, name: &str) -> bool {
    catalog.relational_catalog.contains_key(name)
        || catalog
            .relational_catalog
            .values()
            .any(|table| table.indexes.iter().any(|index| index.name == name))
        || catalog.relational_views.contains_key(name)
        || catalog.relational_materialized_views.contains_key(name)
        || catalog.relational_sequences.contains_key(name)
}

/// Route a SELECT's FROM relation to a synthesized catalog relation, or `None` if it is
/// not a catalog relation (then it resolves as a user table). `pg_catalog` relations match
/// qualified (`pg_catalog.pg_class`) or bare (`pg_class`); `information_schema` is qualified.
pub(crate) fn synthesize_catalog_relation(
    name: &str,
    catalog: &CatalogSnapshot,
) -> Option<(RelationalTable, Vec<Vec<SqlValue>>)> {
    if let Some(relation) = name.strip_prefix("information_schema.") {
        return match relation {
            "tables" => Some(synthesize_information_schema_tables(catalog)),
            "columns" => Some(synthesize_information_schema_columns(catalog)),
            _ => None,
        };
    }
    let relation = name.strip_prefix("pg_catalog.").unwrap_or(name);
    match relation {
        "pg_namespace" => Some(synthesize_pg_namespace(catalog)),
        "pg_class" => Some(synthesize_pg_class(catalog)),
        "pg_attribute" => Some(synthesize_pg_attribute(catalog)),
        "pg_type" => Some(synthesize_pg_type(catalog)),
        "pg_am" => Some(synthesize_pg_am()),
        "pg_roles" => Some(synthesize_pg_roles(catalog)),
        "pg_tablespace" => Some(synthesize_pg_tablespace(catalog)),
        "pg_constraint" => Some(synthesize_pg_constraint(catalog)),
        "pg_description" => Some(synthesize_pg_description(catalog)),
        "pg_default_acl" => Some(synthesize_pg_default_acl(catalog)),
        "pg_indexes" => Some(synthesize_pg_indexes(catalog)),
        "pg_language" => Some(synthesize_pg_language()),
        "pg_proc" => Some(synthesize_pg_proc(catalog)),
        "pg_views" => Some(synthesize_pg_views(catalog)),
        "pg_publication" if catalog.relational_publications.is_empty() => {
            Some(synthesize_empty_catalog_table(
                "pg_publication",
                &[
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
            ))
        }
        "pg_publication_rel" if catalog.relational_publications.is_empty() => {
            Some(synthesize_empty_catalog_table(
                "pg_publication_rel",
                &[
                    ("oid", SqlType::Int4),
                    ("prpubid", SqlType::Int4),
                    ("prrelid", SqlType::Int4),
                    ("prattrs", SqlType::Text),
                    ("prqual", SqlType::Text),
                ],
            ))
        }
        "pg_publication_namespace" if catalog.relational_publications.is_empty() => {
            Some(synthesize_empty_catalog_table(
                "pg_publication_namespace",
                &[
                    ("oid", SqlType::Int4),
                    ("pnpubid", SqlType::Int4),
                    ("pnnspid", SqlType::Int4),
                ],
            ))
        }
        "pg_publication_tables" if catalog.relational_publications.is_empty() => {
            Some(synthesize_empty_catalog_table(
                "pg_publication_tables",
                &[
                    ("pubname", SqlType::Text),
                    ("schemaname", SqlType::Text),
                    ("tablename", SqlType::Text),
                    ("attnames", SqlType::Text),
                    ("rowfilter", SqlType::Text),
                ],
            ))
        }
        "pg_policy" => Some(synthesize_empty_catalog_table(
            "pg_policy",
            &[
                ("oid", SqlType::Int4),
                ("polname", SqlType::Text),
                ("polpermissive", SqlType::Bool),
                ("polroles", SqlType::Text),
                ("polqual", SqlType::Text),
                ("polrelid", SqlType::Int4),
                ("polwithcheck", SqlType::Text),
                ("polcmd", SqlType::Text),
            ],
        )),
        "pg_extension" => Some(synthesize_empty_catalog_table(
            "pg_extension",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("extname", SqlType::Text),
                ("extnamespace", SqlType::Int4),
                ("extrelocatable", SqlType::Bool),
                ("extversion", SqlType::Text),
                ("extconfig", SqlType::Text),
                ("extcondition", SqlType::Text),
            ],
        )),
        // PostgreSQL 17+ dump clients conditionally set
        // `restrict_nonsystem_relation_kind` only when that setting exists. The engine does not
        // expose the server setting, so this catalog source is authoritatively empty; the normal
        // empty-catalog binder still validates the projected `set_config` call and returns its
        // exact descriptor.
        "pg_settings" => Some(synthesize_empty_catalog_table(
            "pg_settings",
            &[("name", SqlType::Text)],
        )),
        "pg_trigger" => Some(synthesize_empty_catalog_table(
            "pg_trigger",
            &[
                ("oid", SqlType::Int4),
                ("tgname", SqlType::Text),
                ("tgrelid", SqlType::Int4),
                ("tgconstraint", SqlType::Int4),
                ("tgdeferrable", SqlType::Bool),
                ("tginitdeferred", SqlType::Bool),
                ("tgenabled", SqlType::Text),
            ],
        )),
        "pg_statistic_ext" => Some(synthesize_empty_catalog_table(
            "pg_statistic_ext",
            &[
                ("oid", SqlType::Int4),
                ("stxname", SqlType::Text),
                ("stxrelid", SqlType::Int4),
                ("stxnamespace", SqlType::Int4),
                ("stxkeys", SqlType::Text),
                ("stxkind", SqlType::Text),
                ("stxstattarget", SqlType::Int4),
            ],
        )),
        "pg_inherits" => Some(synthesize_empty_catalog_table(
            "pg_inherits",
            &[
                ("inhrelid", SqlType::Int4),
                ("inhparent", SqlType::Int4),
                ("inhseqno", SqlType::Int4),
                ("inhdetachpending", SqlType::Bool),
            ],
        )),
        _ => None,
    }
}

/// Add query-independent display values to a synthesized catalog upload only when the SQL lowering
/// needs them. They are ordinary typed columns in the transient device relation, so CASE/function/
/// scalar-subquery compatibility projects final values on the GPU. Normal catalog `*` descriptors
/// never contain these internal columns.
pub(crate) fn add_gpu_catalog_presentation_columns(
    table: &mut RelationalTable,
    rows: &mut [Vec<SqlValue>],
    catalog: &CatalogSnapshot,
) -> Result<(), ExecuteError> {
    match table.name.as_str() {
        "pg_roles" => {
            let rolname = catalog_column_position(table, "rolname")?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_CURRENT_USER_MATCH,
                SqlType::Bool,
                rows.iter()
                    .map(|row| {
                        SqlValue::Bool(matches!(
                            row.get(rolname),
                            Some(SqlValue::Text(name)) if name == "postgres"
                        ))
                    })
                    .collect(),
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_DESCRIPTION,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(rolname) {
                        Some(SqlValue::Text(name)) => catalog
                            .relational_comments
                            .get(&RelationalCommentTarget::Role { role: name.clone() })
                            .map_or(SqlValue::Null, |value| SqlValue::Text(value.clone())),
                        _ => SqlValue::Null,
                    })
                    .collect(),
            )?;
        }
        "pg_tablespace" => {
            let spcname = catalog_column_position(table, "spcname")?;
            let spcowner = catalog_column_position(table, "spcowner")?;
            let spcacl = catalog_column_position(table, "spcacl")?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_OWNER_NAME,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(spcowner) {
                        Some(SqlValue::Int4(oid)) => catalog_role_name(*oid, catalog)
                            .map(|name| SqlValue::Text(name.to_string()))
                            .ok_or_else(|| {
                                sql_pg_error(format!(
                                    "catalog presentation does not model owner OID {oid}"
                                ))
                            }),
                        other => Err(sql_pg_error(format!(
                            "catalog spcowner row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_TABLESPACE_LOCATION,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(spcname) {
                        Some(SqlValue::Text(name)) => {
                            catalog.relational_tablespaces.get(name).map_or_else(
                                || SqlValue::Text(String::new()),
                                |tablespace| SqlValue::Text(tablespace.location.clone()),
                            )
                        }
                        _ => SqlValue::Null,
                    })
                    .collect(),
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_ACL_DISPLAY,
                SqlType::Text,
                rows.iter()
                    .map(|row| catalog_acl_display(row.get(spcacl)))
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_ACL_DEFAULT,
                SqlType::Text,
                vec![SqlValue::Text("{postgres=C/postgres}".to_string()); rows.len()],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_TABLESPACE_SIZE,
                SqlType::Text,
                vec![SqlValue::Text("0 bytes".to_string()); rows.len()],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_DESCRIPTION,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(spcname) {
                        Some(SqlValue::Text(name)) => catalog
                            .relational_comments
                            .get(&RelationalCommentTarget::Tablespace {
                                tablespace: name.clone(),
                            })
                            .map_or(SqlValue::Null, |value| SqlValue::Text(value.clone())),
                        _ => SqlValue::Null,
                    })
                    .collect(),
            )?;
        }
        "pg_namespace" => {
            let nspname = catalog_column_position(table, "nspname")?;
            let nspowner = catalog_column_position(table, "nspowner")?;
            let nspacl = catalog_column_position(table, "nspacl")?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_ACL_DEFAULT,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(nspname) {
                        Some(SqlValue::Text(name)) if name == "public" => {
                            SqlValue::Text("{postgres=UC/postgres,=U/postgres}".to_string())
                        }
                        _ => SqlValue::Null,
                    })
                    .collect(),
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_OWNER_NAME,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(nspowner) {
                        Some(SqlValue::Int4(oid)) => catalog_role_name(*oid, catalog)
                            .map(|name| SqlValue::Text(name.to_string()))
                            .ok_or_else(|| {
                                sql_pg_error(format!(
                                    "catalog presentation does not model owner OID {oid}"
                                ))
                            }),
                        other => Err(sql_pg_error(format!(
                            "catalog nspowner row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_ACL_DISPLAY,
                SqlType::Text,
                rows.iter()
                    .map(|row| catalog_acl_display(row.get(nspacl)))
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_DESCRIPTION,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(nspname) {
                        Some(SqlValue::Text(name)) => catalog
                            .relational_comments
                            .get(&RelationalCommentTarget::Schema {
                                schema: name.clone(),
                            })
                            .map_or(SqlValue::Null, |value| SqlValue::Text(value.clone())),
                        _ => SqlValue::Null,
                    })
                    .collect(),
            )?;
        }
        "pg_class" => {
            let oid = catalog_column_position(table, "oid")?;
            let relkind = catalog_column_position(table, "relkind")?;
            let relowner = catalog_column_position(table, "relowner")?;
            let reloftype = catalog_column_position(table, "reloftype")?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_RELKIND_DISPLAY,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(relkind) {
                        Some(SqlValue::Text(kind)) => catalog_relkind_display(kind)
                            .map(|display| SqlValue::Text(display.to_string()))
                            .ok_or_else(|| {
                                sql_pg_error(format!(
                                    "catalog presentation does not model relkind {kind:?}"
                                ))
                            }),
                        other => Err(sql_pg_error(format!(
                            "catalog relkind row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_RELKIND_ACL_DISPLAY,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(relkind) {
                        Some(SqlValue::Text(kind)) => match kind.as_str() {
                            "r" => SqlValue::Text("table".to_string()),
                            "v" => SqlValue::Text("view".to_string()),
                            "m" => SqlValue::Text("materialized view".to_string()),
                            "S" => SqlValue::Text("sequence".to_string()),
                            "f" => SqlValue::Text("foreign table".to_string()),
                            "p" => SqlValue::Text("partitioned table".to_string()),
                            _ => SqlValue::Null,
                        },
                        _ => SqlValue::Null,
                    })
                    .collect(),
            )?;
            let relation_acls = rows
                .iter()
                .map(|row| match row.get(oid) {
                    Some(SqlValue::Int4(oid)) => catalog_relation_acl(*oid as u32, catalog),
                    _ => SqlValue::Null,
                })
                .collect::<Vec<_>>();
            append_catalog_column(table, rows, "relacl", SqlType::Text, relation_acls.clone())?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_ACL_DISPLAY,
                SqlType::Text,
                relation_acls
                    .iter()
                    .map(|value| catalog_acl_display(Some(value)))
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_COLUMN_PRIVILEGES,
                SqlType::Text,
                vec![SqlValue::Null; rows.len()],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_POLICY_DISPLAY,
                SqlType::Text,
                vec![SqlValue::Null; rows.len()],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_OWNER_NAME,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(relowner) {
                        Some(SqlValue::Int4(oid)) => catalog_role_name(*oid, catalog)
                            .map(|name| SqlValue::Text(name.to_string()))
                            .ok_or_else(|| {
                                sql_pg_error(format!(
                                    "catalog presentation does not model owner OID {oid}"
                                ))
                            }),
                        other => Err(sql_pg_error(format!(
                            "catalog relowner row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_FALSE,
                SqlType::Bool,
                vec![SqlValue::Bool(false); rows.len()],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_EMPTY_TEXT,
                SqlType::Text,
                vec![SqlValue::Text(String::new()); rows.len()],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_RELTYPE_DISPLAY,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(reloftype) {
                        Some(SqlValue::Int4(0)) => Ok(SqlValue::Text(String::new())),
                        Some(SqlValue::Int4(oid)) => catalog_type_name(*oid, catalog)
                            .map(|name| SqlValue::Text(name.to_string()))
                            .ok_or_else(|| {
                                sql_pg_error(format!(
                                    "catalog presentation does not model row-type OID {oid}"
                                ))
                            }),
                        other => Err(sql_pg_error(format!(
                            "catalog reloftype row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
        }
        "pg_attribute" => {
            let attrelid = catalog_column_position(table, "attrelid")?;
            let attnum = catalog_column_position(table, "attnum")?;
            let atttypid = catalog_column_position(table, "atttypid")?;
            let attcollation = catalog_column_position(table, "attcollation")?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_FORMATTED_TYPE,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(atttypid) {
                        Some(SqlValue::Int4(oid)) => catalog_type_name(*oid, catalog)
                            .map(|name| SqlValue::Text(name.to_string()))
                            .ok_or_else(|| {
                                sql_pg_error(format!(
                                    "format_type does not model catalog type OID {oid}"
                                ))
                            }),
                        other => Err(sql_pg_error(format!(
                            "catalog atttypid row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_DEFAULT_EXPR,
                SqlType::Text,
                rows.iter()
                    .map(|row| catalog_attribute_default_expr(row, attrelid, attnum, catalog))
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_COLLATION_NAME,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(attcollation) {
                        Some(SqlValue::Int4(0)) | Some(SqlValue::Null) => Ok(SqlValue::Null),
                        Some(SqlValue::Int4(oid)) => Err(sql_pg_error(format!(
                            "catalog presentation does not model collation OID {oid}"
                        ))),
                        other => Err(sql_pg_error(format!(
                            "catalog attcollation row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
        }
        "pg_proc" => {
            let oid = catalog_column_position(table, "oid")?;
            let proname = catalog_column_position(table, "proname")?;
            let proowner = catalog_column_position(table, "proowner")?;
            let prorettype = catalog_column_position(table, "prorettype")?;
            let proacl = catalog_column_position(table, "proacl")?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_FUNCTION_RESULT,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(prorettype) {
                        Some(SqlValue::Int4(oid)) => catalog_type_name(*oid, catalog)
                            .map(|name| SqlValue::Text(name.to_string()))
                            .ok_or_else(|| {
                                sql_pg_error(format!(
                                    "pg_get_function_result does not model type OID {oid}"
                                ))
                            }),
                        other => Err(sql_pg_error(format!(
                            "catalog prorettype row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_OWNER_NAME,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(proowner) {
                        Some(SqlValue::Int4(oid)) => catalog_role_name(*oid, catalog)
                            .map(|name| SqlValue::Text(name.to_string()))
                            .ok_or_else(|| {
                                sql_pg_error(format!(
                                    "catalog presentation does not model owner OID {oid}"
                                ))
                            }),
                        other => Err(sql_pg_error(format!(
                            "catalog proowner row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_EMPTY_TEXT,
                SqlType::Text,
                vec![SqlValue::Text(String::new()); rows.len()],
            )?;
            for (name, value) in [
                (GPU_CATALOG_FUNCTION_KIND, "func"),
                (GPU_CATALOG_FUNCTION_VOLATILITY, "volatile"),
                (GPU_CATALOG_FUNCTION_PARALLEL, "unsafe"),
                (GPU_CATALOG_FUNCTION_SECURITY, "invoker"),
            ] {
                append_catalog_column(
                    table,
                    rows,
                    name,
                    SqlType::Text,
                    vec![SqlValue::Text(value.to_string()); rows.len()],
                )?;
            }
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_FUNCTION_INTERNAL_NAME,
                SqlType::Text,
                vec![SqlValue::Null; rows.len()],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_ACL_DISPLAY,
                SqlType::Text,
                rows.iter()
                    .map(|row| catalog_acl_display(row.get(proacl)))
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_DESCRIPTION,
                SqlType::Text,
                rows.iter()
                    .map(|row| match (row.get(oid), row.get(proname)) {
                        (Some(SqlValue::Int4(_)), Some(SqlValue::Text(name))) => catalog
                            .relational_comments
                            .get(&RelationalCommentTarget::Function {
                                function: name.clone(),
                            })
                            .map_or(SqlValue::Null, |value| SqlValue::Text(value.clone())),
                        _ => SqlValue::Null,
                    })
                    .collect(),
            )?;
        }
        "pg_default_acl" => {
            let owner = catalog_column_position(table, "defaclrole")?;
            let kind = catalog_column_position(table, "defaclobjtype")?;
            let acl = catalog_column_position(table, "defaclacl")?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_OWNER_NAME,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(owner) {
                        Some(SqlValue::Int4(oid)) => catalog_role_name(*oid, catalog)
                            .map(|name| SqlValue::Text(name.to_string()))
                            .ok_or_else(|| {
                                sql_pg_error(format!(
                                    "catalog presentation does not model owner OID {oid}"
                                ))
                            }),
                        other => Err(sql_pg_error(format!(
                            "catalog defaclrole row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_DEFAULT_ACL_TYPE,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(kind) {
                        Some(SqlValue::Text(kind)) => match kind.as_str() {
                            "r" => SqlValue::Text("table".to_string()),
                            "S" => SqlValue::Text("sequence".to_string()),
                            "f" => SqlValue::Text("function".to_string()),
                            "T" => SqlValue::Text("type".to_string()),
                            "n" => SqlValue::Text("schema".to_string()),
                            _ => SqlValue::Null,
                        },
                        _ => SqlValue::Null,
                    })
                    .collect(),
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_ACL_DISPLAY,
                SqlType::Text,
                rows.iter()
                    .map(|row| catalog_acl_display(row.get(acl)))
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
        }
        "pg_type" => {
            let typbasetype = catalog_column_position(table, "typbasetype")?;
            let row_count = rows.len();
            append_catalog_column(
                table,
                rows,
                "tableoid",
                SqlType::Int4,
                vec![SqlValue::Int4(1247); row_count],
            )?;
            append_catalog_column(
                table,
                rows,
                "typtypmod",
                SqlType::Int4,
                vec![SqlValue::Int4(-1); row_count],
            )?;
            append_catalog_column(
                table,
                rows,
                "typcollation",
                SqlType::Int4,
                vec![SqlValue::Int4(0); row_count],
            )?;
            append_catalog_column(
                table,
                rows,
                "typnotnull",
                SqlType::Bool,
                vec![SqlValue::Bool(false); row_count],
            )?;
            append_catalog_column(
                table,
                rows,
                "typdefault",
                SqlType::Text,
                vec![SqlValue::Null; row_count],
            )?;
            append_catalog_column(
                table,
                rows,
                "typacl",
                SqlType::Text,
                vec![SqlValue::Null; row_count],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_FORMATTED_TYPE,
                SqlType::Text,
                rows.iter()
                    .map(|row| match row.get(typbasetype) {
                        Some(SqlValue::Int4(0)) => Ok(SqlValue::Null),
                        Some(SqlValue::Int4(oid)) => catalog_type_name(*oid, catalog)
                            .map(|name| SqlValue::Text(name.to_string()))
                            .ok_or_else(|| {
                                sql_pg_error(format!(
                                    "format_type does not model domain base type OID {oid}"
                                ))
                            }),
                        other => Err(sql_pg_error(format!(
                            "catalog typbasetype row is malformed: {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_COLLATION_NAME,
                SqlType::Text,
                vec![SqlValue::Null; row_count],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_NULLABLE_DISPLAY,
                SqlType::Text,
                vec![SqlValue::Null; row_count],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_DOMAIN_CHECK,
                SqlType::Text,
                vec![SqlValue::Null; row_count],
            )?;
            append_catalog_column(
                table,
                rows,
                GPU_CATALOG_ACL_DISPLAY,
                SqlType::Text,
                vec![SqlValue::Null; row_count],
            )?;
        }
        _ => {}
    }
    Ok(())
}

fn catalog_column_position(table: &RelationalTable, name: &str) -> Result<usize, ExecuteError> {
    table
        .columns
        .iter()
        .position(|column| column.name == name)
        .ok_or_else(|| sql_pg_error(format!("catalog relation is missing column {name:?}")))
}

fn append_catalog_column(
    table: &mut RelationalTable,
    rows: &mut [Vec<SqlValue>],
    name: &str,
    ty: SqlType,
    values: Vec<SqlValue>,
) -> Result<(), ExecuteError> {
    if rows.len() != values.len() {
        return Err(sql_pg_error(
            "catalog presentation column height is inconsistent".to_string(),
        ));
    }
    let attnum = i16::try_from(table.columns.len() + 1)
        .map_err(|_| sql_pg_error("catalog presentation is too wide".to_string()))?;
    table.columns.push(RelationalColumn {
        id: 0,
        table_oid: table.oid,
        attnum,
        name: name.to_string(),
        ty,
        domain: None,
        default: None,
        type_oid: ty.postgres_oid(),
        type_size: ty.type_size(),
    });
    for (row, value) in rows.iter_mut().zip(values) {
        row.push(value);
    }
    Ok(())
}

fn catalog_relkind_display(kind: &str) -> Option<&'static str> {
    match kind {
        "r" => Some("table"),
        "v" => Some("view"),
        "m" => Some("materialized view"),
        "i" => Some("index"),
        "S" => Some("sequence"),
        "t" => Some("TOAST table"),
        "f" => Some("foreign table"),
        "p" => Some("partitioned table"),
        "I" => Some("partitioned index"),
        _ => None,
    }
}

fn catalog_role_name(oid: i32, catalog: &CatalogSnapshot) -> Option<&str> {
    if oid == PG_BOOTSTRAP_OWNER_OID {
        return Some("postgres");
    }
    catalog
        .relational_roles
        .values()
        .find(|role| role.oid == oid as u32)
        .map(|role| role.name.as_str())
}

fn catalog_type_name(oid: i32, catalog: &CatalogSnapshot) -> Option<&str> {
    let builtin = match oid as u32 {
        16 => Some("boolean"),
        20 => Some("bigint"),
        21 => Some("smallint"),
        23 => Some("integer"),
        25 => Some("text"),
        1082 => Some("date"),
        1114 => Some("timestamp without time zone"),
        1700 => Some("numeric"),
        2950 => Some("uuid"),
        _ => None,
    };
    builtin.or_else(|| {
        catalog
            .relational_domains
            .values()
            .find(|domain| domain.oid == oid as u32)
            .map(|domain| domain.name.as_str())
    })
}

fn catalog_acl_display(value: Option<&SqlValue>) -> Result<SqlValue, ExecuteError> {
    match value {
        Some(SqlValue::Null) | None => Ok(SqlValue::Null),
        Some(SqlValue::Text(value)) => Ok(SqlValue::Text(
            value
                .strip_prefix('{')
                .and_then(|value| value.strip_suffix('}'))
                .unwrap_or(value)
                .split(',')
                .collect::<Vec<_>>()
                .join("\n"),
        )),
        other => Err(sql_pg_error(format!(
            "catalog ACL row is malformed: {other:?}"
        ))),
    }
}

fn catalog_relation_acl(oid: u32, catalog: &CatalogSnapshot) -> SqlValue {
    catalog
        .relational_catalog
        .values()
        .find(|relation| relation.oid == oid)
        .map(|relation| relation_acl_array(&relation.acl, "r"))
        .or_else(|| {
            catalog
                .relational_views
                .values()
                .find(|relation| relation.oid == oid)
                .map(|relation| relation_acl_array(&relation.acl, "v"))
        })
        .or_else(|| {
            catalog
                .relational_materialized_views
                .values()
                .find(|relation| relation.oid == oid)
                .map(|relation| relation_acl_array(&relation.acl, "m"))
        })
        .or_else(|| {
            catalog
                .relational_sequences
                .values()
                .find(|relation| relation.oid == oid)
                .map(|relation| relation_acl_array(&relation.acl, "S"))
        })
        .unwrap_or(SqlValue::Null)
}

fn catalog_attribute_default_expr(
    row: &[SqlValue],
    attrelid: usize,
    attnum: usize,
    catalog: &CatalogSnapshot,
) -> Result<SqlValue, ExecuteError> {
    let (Some(SqlValue::Int4(relation_oid)), Some(SqlValue::Int4(attribute_number))) =
        (row.get(attrelid), row.get(attnum))
    else {
        return Err(sql_pg_error(
            "catalog attribute identity row is malformed".to_string(),
        ));
    };
    let column = catalog
        .relational_catalog
        .values()
        .find(|table| table.oid == *relation_oid as u32)
        .and_then(|table| {
            table
                .columns
                .iter()
                .find(|column| i32::from(column.attnum) == *attribute_number)
        });
    let Some(column) = column else {
        return Ok(SqlValue::Null);
    };
    match &column.default {
        None => Ok(SqlValue::Null),
        Some(ColumnDefault::Literal(value)) => render_sql_value_literal(value)
            .map(SqlValue::Text)
            .map_err(ExecuteError::Engine),
        Some(ColumnDefault::SequenceNextVal { sequence, .. }) => Ok(SqlValue::Text(format!(
            "nextval('{}'::regclass)",
            sequence.replace('\'', "''")
        ))),
    }
}
