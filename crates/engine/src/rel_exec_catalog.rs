//! Synthesized GPU-catalog relation descriptors and upload rows.
//!
//! This module owns deterministic catalog encoding only. Consumers bind and execute these
//! relations through the resident/transient GPU relational paths.

use super::*;

pub(crate) const GPU_CATALOG_RELKIND_DISPLAY: &str = "__gpu_relkind_display";
pub(crate) const GPU_CATALOG_OWNER_NAME: &str = "__gpu_owner_name";
pub(crate) const GPU_CATALOG_FALSE: &str = "__gpu_false";
pub(crate) const GPU_CATALOG_EMPTY_TEXT: &str = "__gpu_empty_text";
pub(crate) const GPU_CATALOG_RELTYPE_DISPLAY: &str = "__gpu_reltype_display";
pub(crate) const GPU_CATALOG_FORMATTED_TYPE: &str = "__gpu_formatted_type";
pub(crate) const GPU_CATALOG_DEFAULT_EXPR: &str = "__gpu_default_expr";
pub(crate) const GPU_CATALOG_COLLATION_NAME: &str = "__gpu_collation_name";
pub(crate) const GPU_CATALOG_CONSTRAINT_DEF: &str = "__gpu_constraint_def";

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
    "pg_inherits",
    "pg_namespace",
    "pg_policy",
    "pg_proc",
    "pg_publication",
    "pg_publication_namespace",
    "pg_publication_rel",
    "pg_publication_tables",
    "pg_roles",
    "pg_statistic_ext",
    "pg_trigger",
    "pg_type",
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
            ("oid", SqlType::Int4),
            ("nspname", SqlType::Text),
            ("nspowner", SqlType::Int4),
        ],
    );
    let owner = SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID);
    let mut rows = vec![
        vec![
            SqlValue::Int4(PG_CATALOG_NAMESPACE_OID),
            SqlValue::Text("pg_catalog".to_string()),
            owner.clone(),
        ],
        vec![
            SqlValue::Int4(PG_INFORMATION_SCHEMA_NAMESPACE_OID),
            SqlValue::Text("information_schema".to_string()),
            owner.clone(),
        ],
    ];
    if catalog.relational_public_schema_exists {
        rows.push(vec![
            SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
            SqlValue::Text("public".to_string()),
            owner,
        ]);
    }
    (table, rows)
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
    let mut synthetic_oid = 50_000_i32;
    for relation in catalog.relational_catalog.values() {
        for index in &relation.indexes {
            if !(index.primary_key || index.unique_constraint) {
                continue;
            }
            rows.push(vec![
                SqlValue::Int4(synthetic_oid),
                SqlValue::Text(index.name.clone()),
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Text(if index.primary_key { "p" } else { "u" }.to_string()),
                SqlValue::Int4(0),
                SqlValue::Int4(0),
            ]);
            synthetic_oid += 1;
        }
        for constraint in &relation.check_constraints {
            rows.push(vec![
                SqlValue::Int4(synthetic_oid),
                SqlValue::Text(constraint.name.clone()),
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Text("c".to_string()),
                SqlValue::Int4(0),
                SqlValue::Int4(0),
            ]);
            synthetic_oid += 1;
        }
        for constraint in &relation.foreign_keys {
            let referenced_oid = catalog
                .relational_catalog
                .get(&constraint.referenced_table)
                .map_or(0, |table| table.oid as i32);
            rows.push(vec![
                SqlValue::Int4(synthetic_oid),
                SqlValue::Text(constraint.name.clone()),
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Text("f".to_string()),
                SqlValue::Int4(0),
                SqlValue::Int4(referenced_oid),
            ]);
            synthetic_oid += 1;
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
        ],
    );
    let base = |ty: SqlType| {
        vec![
            SqlValue::Int4(ty.postgres_oid() as i32),
            SqlValue::Text(ty.catalog_name().to_string()),
            SqlValue::Int4(i32::from(ty.type_size())),
            SqlValue::Text("b".to_string()),
            SqlValue::Int4(PG_CATALOG_NAMESPACE_OID),
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
        ],
    );
    let mut rows = Vec::new();
    for t in catalog.relational_catalog.values() {
        for col in &t.columns {
            rows.push(vec![
                SqlValue::Text("postgres".to_string()),
                SqlValue::Text("public".to_string()),
                SqlValue::Text(t.name.clone()),
                SqlValue::Text(col.name.clone()),
                SqlValue::Int4(i32::from(col.attnum)),
                SqlValue::Text(information_schema_data_type(col.ty).to_string()),
                SqlValue::Text("YES".to_string()),
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
        "pg_constraint" => Some(synthesize_pg_constraint(catalog)),
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
        "pg_class" => {
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
