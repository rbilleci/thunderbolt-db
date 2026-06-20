//! Relational execution helpers — pure free functions extracted from `lib.rs`
//! (P0 §9.6 decomposition, behavior-preserving). Value coercion, catalog
//! synthesis, relational row codec, SELECT/DML binding, and aggregate
//! validation. No `Engine` state, no locks: all inputs are explicit.

use super::*;

pub(crate) fn sql_value_matches_type(value: &SqlValue, ty: SqlType) -> bool {
    matches!(
        (value, ty),
        (SqlValue::Int4(_), SqlType::Int4)
            | (SqlValue::Int8(_), SqlType::Int8)
            | (SqlValue::Numeric(_), SqlType::Numeric { .. })
            | (SqlValue::Bool(_), SqlType::Bool)
            | (SqlValue::Text(_), SqlType::Text)
            | (SqlValue::Date(_), SqlType::Date)
            | (SqlValue::Timestamp(_), SqlType::Timestamp)
            | (SqlValue::Uuid(_), SqlType::Uuid)
            | (SqlValue::Int2(_), SqlType::Int2)
    )
}

/// Widen an INSERT/UPDATE value to `column_ty` along the lossless integer→numeric tower,
/// so a bare-int literal populates a `numeric`/`int8` column (`INSERT INTO acct (bal)
/// VALUES (100)`) the way PostgreSQL's implicit assignment cast does. Only lossless
/// widenings are applied; a narrowing/rounding assignment cast (numeric→int, int8→int4)
/// is NOT — those still fail the type check in `coerce_insert_value`, as before. Numeric
/// widening lands at scale 0; the caller then rescales to the column's declared scale.
pub(crate) fn widen_value_to_column_type(value: SqlValue, column_ty: SqlType) -> SqlValue {
    match (value, column_ty) {
        (SqlValue::Int4(v), SqlType::Int8) => SqlValue::Int8(i64::from(v)),
        (SqlValue::Int4(v), SqlType::Numeric { .. }) => {
            SqlValue::Numeric(Decimal128::new(i128::from(v), 0))
        }
        (SqlValue::Int8(v), SqlType::Numeric { .. }) => {
            SqlValue::Numeric(Decimal128::new(i128::from(v), 0))
        }
        (other, _) => other,
    }
}

/// Validate `value` against the column type and coerce it into storable form. The value is
/// first widened along the integer→numeric tower (`widen_value_to_column_type`), so a
/// bare-int literal lands in a `numeric`/`int8` column. For a NUMERIC column this then
/// rescales the value to the column's declared `scale` (round-half-up) and enforces the
/// `precision` budget, raising a PostgreSQL-style `numeric field overflow` when the
/// rescaled mantissa exceeds `10^precision` or leaves i128 range. Other types pass through
/// unchanged after the type check.
pub(crate) fn coerce_insert_value(
    value: SqlValue,
    ty: SqlType,
    column_name: &str,
) -> Result<SqlValue, EngineError> {
    // A smallint column accepts an integer literal (parsed as int4) narrowed to int2, range-checked
    // (PG raises "smallint out of range" on overflow). An already-typed Int2 passes the check below.
    if let (SqlValue::Int4(v), SqlType::Int2) = (&value, ty) {
        return i16::try_from(*v)
            .map(SqlValue::Int2)
            .map_err(|_| EngineError::ApplyFailed("smallint out of range".to_string()));
    }

    // A string literal assigned to a date/timestamp column is parsed as that type (PG coerces an
    // unknown-type literal to the column type). An already-typed value passes through the check below.
    if let SqlValue::Text(text) = &value {
        match ty {
            SqlType::Date => {
                return gpu_db_sql::datetime::parse_date(text)
                    .map(SqlValue::Date)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "invalid input syntax for type date: \"{text}\""
                        ))
                    });
            }
            SqlType::Timestamp => {
                return gpu_db_sql::datetime::parse_timestamp(text)
                    .map(SqlValue::Timestamp)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "invalid input syntax for type timestamp: \"{text}\""
                        ))
                    });
            }
            SqlType::Uuid => {
                return gpu_db_sql::uuid::parse_uuid(text)
                    .map(SqlValue::Uuid)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "invalid input syntax for type uuid: \"{text}\""
                        ))
                    });
            }
            _ => {}
        }
    }
    let value = widen_value_to_column_type(value, ty);
    if !sql_value_matches_type(&value, ty) {
        return Err(EngineError::ApplyFailed(format!(
            "invalid value for column \"{column_name}\""
        )));
    }
    match (value, ty) {
        (SqlValue::Numeric(decimal), SqlType::Numeric { precision, scale }) => {
            let rescaled = decimal
                .rescale(scale)
                .map_err(|_| EngineError::ApplyFailed("numeric field overflow".to_string()))?;
            if numeric_exceeds_precision(rescaled.mantissa, precision) {
                return Err(EngineError::ApplyFailed(
                    "numeric field overflow".to_string(),
                ));
            }
            Ok(SqlValue::Numeric(rescaled))
        }
        (value, _) => Ok(value),
    }
}

/// Coerce a column DEFAULT to `ty` with the same lossless integer→numeric widening and
/// scale/precision handling as an INSERT value, so a cross-type default literal
/// (`bal NUMERIC DEFAULT 0`, `big BIGINT DEFAULT 5`) is accepted and stored at the
/// column's type and scale. A `nextval(...)` default is still restricted to `int4`
/// columns (the prior `column_default_matches_type` rule). The single source of truth for
/// "is this default valid for this column", used both to coerce-and-store (CREATE, ALTER
/// SET DEFAULT, ADD COLUMN) and to validate in the concurrent-DDL preflight.
pub(crate) fn coerce_column_default(
    default: ColumnDefault,
    ty: SqlType,
    column_name: &str,
) -> Result<ColumnDefault, EngineError> {
    match default {
        ColumnDefault::Literal(value) => Ok(ColumnDefault::Literal(coerce_insert_value(
            value,
            ty,
            column_name,
        )?)),
        ColumnDefault::SequenceNextVal { .. } => {
            if ty != SqlType::Int4 {
                return Err(EngineError::ApplyFailed(format!(
                    "invalid default for column \"{column_name}\""
                )));
            }
            Ok(default)
        }
    }
}

/// Whether `mantissa` needs more than `precision` significant decimal digits (the
/// PostgreSQL `numeric(p,s)` overflow condition once the value is at the column scale).
pub(crate) fn numeric_exceeds_precision(mantissa: i128, precision: u8) -> bool {
    let mut bound: i128 = 1;
    for _ in 0..precision {
        match bound.checked_mul(10) {
            Some(next) => bound = next,
            // 10^precision overflowed i128, so any in-range mantissa fits.
            None => return false,
        }
    }
    mantissa.unsigned_abs() >= bound.unsigned_abs()
}

/// Convert a decimal to an exact `i128` integer, or `None` if it carries a fractional
/// part. Used to coerce an integral numeric literal (`5.0`) to an integer column.
pub(crate) fn decimal_to_i128_exact(value: &Decimal128) -> Option<i128> {
    let canonical = value.canonical();
    (canonical.scale == 0).then_some(canonical.mantissa)
}

/// Coerce a WHERE-clause filter literal to `column_ty`, applying the implicit casts
/// PostgreSQL allows across the integer/numeric tower: a bare-int literal matches a
/// `numeric`/`int8` column (`WHERE bal = 5`, `WHERE big = 5`) and an integral numeric
/// literal matches an integer column (`WHERE id = 5.0`). This also fixes the equality
/// value-INDEX probe — the index keys on the column-typed encoding, so an un-coerced
/// `Int4(5)` would key `i:5` and miss a numeric column's `d:5:0` slot. A literal with no
/// implicit cast to the column type (or one out of the column's range) is returned
/// unchanged: it then compares unequal (correct — `5.5` matches no integer row) and the
/// index probe keys on the literal's own type and correctly finds nothing.
pub(crate) fn coerce_filter_literal(value: SqlValue, column_ty: SqlType) -> SqlValue {
    match (value, column_ty) {
        (SqlValue::Int4(v), SqlType::Numeric { .. }) => {
            SqlValue::Numeric(Decimal128::new(i128::from(v), 0))
        }
        (SqlValue::Int8(v), SqlType::Numeric { .. }) => {
            SqlValue::Numeric(Decimal128::new(i128::from(v), 0))
        }
        (SqlValue::Int4(v), SqlType::Int8) => SqlValue::Int8(i64::from(v)),
        (SqlValue::Int8(v), SqlType::Int4) => match i32::try_from(v) {
            Ok(narrowed) => SqlValue::Int4(narrowed),
            Err(_) => SqlValue::Int8(v),
        },
        (SqlValue::Numeric(d), SqlType::Int4) => match decimal_to_i128_exact(&d) {
            Some(i) => i32::try_from(i)
                .map(SqlValue::Int4)
                .unwrap_or(SqlValue::Numeric(d)),
            None => SqlValue::Numeric(d),
        },
        (SqlValue::Numeric(d), SqlType::Int8) => match decimal_to_i128_exact(&d) {
            Some(i) => i64::try_from(i)
                .map(SqlValue::Int8)
                .unwrap_or(SqlValue::Numeric(d)),
            None => SqlValue::Numeric(d),
        },
        (other, _) => other,
    }
}

pub(crate) fn add_column_default_supported(default: &ColumnDefault) -> bool {
    match default {
        ColumnDefault::Literal(_) => true,
        ColumnDefault::SequenceNextVal {
            create_if_missing, ..
        } => !create_if_missing,
    }
}

pub(crate) fn sequence_defaults(columns: &[ColumnDef]) -> impl Iterator<Item = &ColumnDefault> {
    columns.iter().filter_map(|column| column.default.as_ref())
}

pub(crate) fn relational_row_key(table: &str, row_id: u64) -> String {
    format!("rel/{table}/{row_id:020}")
}

pub(crate) fn relational_key_prefix(table: &str) -> String {
    format!("rel/{table}/")
}

pub(crate) fn relational_index_value(value: &SqlValue) -> String {
    match value {
        SqlValue::Int2(value) => format!("i2:{value}"),
        SqlValue::Int4(value) => format!("i:{value}"),
        SqlValue::Int8(value) => format!("n:{value}"),
        // The equality value-index keys on the CANONICAL decimal (trailing zeros stripped)
        // so a stored `1.0` and a `WHERE bal = 1.00` literal hash to the same slot
        // regardless of their declared scale (numeric equality is scale-insensitive).
        SqlValue::Numeric(value) => {
            let canonical = value.canonical();
            format!("d:{}:{}", canonical.mantissa, canonical.scale)
        }
        SqlValue::Bool(value) => format!("b:{}", if *value { 't' } else { 'f' }),
        SqlValue::Text(value) => format!("t:{value}"),
        SqlValue::Date(value) => format!("date:{value}"),
        SqlValue::Timestamp(value) => format!("ts:{value}"),
        SqlValue::Uuid(bytes) => format!("uuid:{}", gpu_db_sql::uuid::format_uuid(bytes)),
    }
}

/// The per-table value-index entries `rows` contribute, keyed by `(column, value)` (the owning
/// table is implied by the per-table [`SnapshotCell`]). Append-only: `apply_delta` merges these
/// into the table's `TableVersionData::value_index`.
pub(crate) fn relational_value_index_entries_for_rows(
    columns: &[RelationalColumn],
    rows: &[(String, Vec<SqlValue>)],
) -> BTreeMap<ColumnValueKey, Vec<String>> {
    let mut entries = BTreeMap::new();
    for (row_key, values) in rows {
        for (column, value) in columns.iter().zip(values.iter()) {
            entries
                .entry(ColumnValueKey {
                    column: column.name.clone(),
                    value: relational_index_value(value),
                })
                .or_insert_with(Vec::new)
                .push(row_key.clone());
        }
    }
    entries
}

pub(crate) fn render_relational_insert(insert: &Insert) -> Result<String, EngineError> {
    let mut sql = format!("INSERT INTO {}", insert.table);
    if !insert.columns.is_empty() {
        sql.push_str(" (");
        sql.push_str(&insert.columns.join(", "));
        sql.push(')');
    }
    sql.push_str(" VALUES ");
    let rendered_rows = insert
        .rows
        .iter()
        .map(|row| {
            let rendered_values = row
                .iter()
                .map(render_sql_value_literal)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("({})", rendered_values.join(", ")))
        })
        .collect::<Result<Vec<_>, EngineError>>()?;
    sql.push_str(&rendered_rows.join(", "));
    Ok(sql)
}

pub(crate) fn render_sql_value_literal(value: &SqlValue) -> Result<String, EngineError> {
    match value {
        SqlValue::Int2(value) => Ok(value.to_string()),
        SqlValue::Int4(value) => Ok(value.to_string()),
        SqlValue::Text(value) => Ok(format!("'{}'", value.replace('\'', "''"))),
        SqlValue::Int8(_)
        | SqlValue::Numeric(_)
        | SqlValue::Bool(_)
        | SqlValue::Date(_)
        | SqlValue::Timestamp(_)
        | SqlValue::Uuid(_) => {
            Err(EngineError::ApplyFailed(
                "COPY-to-engine ingestion supports int4/text rows only".to_string(),
            ))
        }
    }
}

pub(crate) fn relational_resident_value_bytes(value: &SqlValue) -> u64 {
    match value {
        // int2 rides the int4 device section widened to 4 bytes.
        SqlValue::Int2(_) => 4,
        SqlValue::Int4(_) => 4,
        SqlValue::Int8(_) => 8,
        // A NUMERIC is a fixed-width i128 mantissa + u8 scale.
        SqlValue::Numeric(_) => (std::mem::size_of::<i128>() + std::mem::size_of::<u8>()) as u64,
        SqlValue::Bool(_) => 1,
        SqlValue::Text(value) => value.len() as u64,
        SqlValue::Date(_) => 4,
        SqlValue::Timestamp(_) => 8,
        SqlValue::Uuid(_) => 16,
    }
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
        _ => None,
    }
}

pub(crate) fn encode_relational_row(values: &[SqlValue]) -> String {
    values
        .iter()
        .map(|value| match value {
            SqlValue::Int2(value) => format!("i2:{value}"),
            SqlValue::Int4(value) => format!("i:{value}"),
            SqlValue::Int8(value) => format!("n:{value}"),
            // Storage preserves the value's declared scale (`d:<mantissa>:<scale>`); the
            // value-index canonicalizes separately for scale-insensitive equality lookups.
            SqlValue::Numeric(value) => format!("d:{}:{}", value.mantissa, value.scale),
            SqlValue::Bool(value) => format!("b:{}", if *value { 't' } else { 'f' }),
            SqlValue::Text(value) => {
                format!("t:{}", value.replace('\\', "\\\\").replace('|', "\\|"))
            }
            SqlValue::Date(value) => format!("date:{value}"),
            SqlValue::Timestamp(value) => format!("ts:{value}"),
            SqlValue::Uuid(bytes) => format!("uuid:{}", gpu_db_sql::uuid::format_uuid(bytes)),
        })
        .collect::<Vec<_>>()
        .join("|")
}

pub(crate) fn decode_relational_row(
    input: &str,
    columns: &[RelationalColumn],
) -> Result<Vec<SqlValue>, ExecuteError> {
    let parts = split_escaped_row(input);
    if parts.len() != columns.len() {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "stored relational row does not match catalog shape".to_string(),
        )));
    }
    parts
        .into_iter()
        .zip(columns.iter())
        .map(|(part, column)| decode_relational_value(&part, column))
        .collect()
}

/// Decode one stored, escape-resolved cell into a [`SqlValue`] of the column's type.
/// The storage prefix vocabulary is `i:`int4 `n:`int8 `d:`numeric(`mantissa:scale`)
/// `b:`bool `t:`text — chosen to mirror [`relational_index_value`]'s key vocabulary.
pub(crate) fn decode_relational_value(
    part: &str,
    column: &RelationalColumn,
) -> Result<SqlValue, ExecuteError> {
    let wrong_type = || {
        ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "stored value for column \"{}\" has wrong type",
            column.name
        )))
    };
    match column.ty {
        SqlType::Int2 => {
            let value = part.strip_prefix("i2:").ok_or_else(wrong_type)?;
            value.parse::<i16>().map(SqlValue::Int2).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "stored SMALLINT value is invalid".to_string(),
                ))
            })
        }
        SqlType::Int4 => {
            let value = part.strip_prefix("i:").ok_or_else(wrong_type)?;
            value.parse::<i32>().map(SqlValue::Int4).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "stored INT value is invalid".to_string(),
                ))
            })
        }
        SqlType::Int8 => {
            let value = part.strip_prefix("n:").ok_or_else(wrong_type)?;
            value.parse::<i64>().map(SqlValue::Int8).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "stored BIGINT value is invalid".to_string(),
                ))
            })
        }
        SqlType::Numeric { .. } => {
            let body = part.strip_prefix("d:").ok_or_else(wrong_type)?;
            let (mantissa, scale) = body.split_once(':').ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "stored NUMERIC value is invalid".to_string(),
                ))
            })?;
            let mantissa = mantissa.parse::<i128>().map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "stored NUMERIC value is invalid".to_string(),
                ))
            })?;
            let scale = scale.parse::<u8>().map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "stored NUMERIC value is invalid".to_string(),
                ))
            })?;
            Ok(SqlValue::Numeric(Decimal128::new(mantissa, scale)))
        }
        SqlType::Bool => {
            let value = part.strip_prefix("b:").ok_or_else(wrong_type)?;
            match value {
                "t" => Ok(SqlValue::Bool(true)),
                "f" => Ok(SqlValue::Bool(false)),
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "stored BOOL value is invalid".to_string(),
                ))),
            }
        }
        SqlType::Text => {
            let value = part.strip_prefix("t:").ok_or_else(wrong_type)?;
            Ok(SqlValue::Text(value.to_string()))
        }
        SqlType::Date => {
            let value = part.strip_prefix("date:").ok_or_else(wrong_type)?;
            value.parse::<i32>().map(SqlValue::Date).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "stored DATE value is invalid".to_string(),
                ))
            })
        }
        SqlType::Timestamp => {
            let value = part.strip_prefix("ts:").ok_or_else(wrong_type)?;
            value.parse::<i64>().map(SqlValue::Timestamp).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "stored TIMESTAMP value is invalid".to_string(),
                ))
            })
        }
        SqlType::Uuid => {
            let value = part.strip_prefix("uuid:").ok_or_else(wrong_type)?;
            gpu_db_sql::uuid::parse_uuid(value)
                .map(SqlValue::Uuid)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "stored UUID value is invalid".to_string(),
                    ))
                })
        }
    }
}

pub(crate) fn split_escaped_row(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut escaped = false;
    for ch in input.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '|' {
            out.push(current);
            current = String::new();
        } else {
            current.push(ch);
        }
    }
    if escaped {
        current.push('\\');
    }
    out.push(current);
    out
}

pub(crate) fn compare_sql_values(left: &SqlValue, right: &SqlValue) -> Ordering {
    match (left, right) {
        // smallint widens to int4 for every comparison (PG's numeric tower); recurse with it widened
        // so the existing integer/numeric cross-type arms apply -- no per-pair int2 spread.
        (SqlValue::Int2(left), right) => {
            compare_sql_values(&SqlValue::Int4(i32::from(*left)), right)
        }
        (left, SqlValue::Int2(right)) => {
            compare_sql_values(left, &SqlValue::Int4(i32::from(*right)))
        }
        (SqlValue::Int4(left), SqlValue::Int4(right)) => left.cmp(right),
        (SqlValue::Int8(left), SqlValue::Int8(right)) => left.cmp(right),
        (SqlValue::Int4(left), SqlValue::Int8(right)) => i64::from(*left).cmp(right),
        (SqlValue::Int8(left), SqlValue::Int4(right)) => left.cmp(&i64::from(*right)),
        // Numeric vs numeric is scale-aligned (1.0 == 1.00). Integer-vs-numeric promotes the
        // integer to a scale-0 Decimal128 so `bal > 5` works across the int4/numeric boundary.
        (SqlValue::Numeric(left), SqlValue::Numeric(right)) => left.cmp(right),
        (SqlValue::Int4(left), SqlValue::Numeric(right)) => {
            Decimal128::new(i128::from(*left), 0).cmp(right)
        }
        (SqlValue::Int8(left), SqlValue::Numeric(right)) => {
            Decimal128::new(i128::from(*left), 0).cmp(right)
        }
        (SqlValue::Numeric(left), SqlValue::Int4(right)) => {
            left.cmp(&Decimal128::new(i128::from(*right), 0))
        }
        (SqlValue::Numeric(left), SqlValue::Int8(right)) => {
            left.cmp(&Decimal128::new(i128::from(*right), 0))
        }
        (SqlValue::Bool(left), SqlValue::Bool(right)) => left.cmp(right),
        (SqlValue::Text(left), SqlValue::Text(right)) => left.cmp(right),
        // Cross-family ordering follows the variant order int < numeric < bool < text. This
        // only surfaces for heterogeneous comparisons (e.g. sorting a mixed projection) — the
        // typed engine never compares a numeric to a bool in a real predicate.
        (SqlValue::Int4(_) | SqlValue::Int8(_), SqlValue::Bool(_) | SqlValue::Text(_)) => {
            Ordering::Less
        }
        (SqlValue::Numeric(_), SqlValue::Bool(_) | SqlValue::Text(_)) => Ordering::Less,
        (SqlValue::Bool(_), SqlValue::Int4(_) | SqlValue::Int8(_) | SqlValue::Numeric(_)) => {
            Ordering::Greater
        }
        (SqlValue::Bool(_), SqlValue::Text(_)) => Ordering::Less,
        (
            SqlValue::Text(_),
            SqlValue::Int4(_) | SqlValue::Int8(_) | SqlValue::Numeric(_) | SqlValue::Bool(_),
        ) => Ordering::Greater,
        // Date is its own tier (sorts last); same-type dates compare by day count. Cross-type
        // date comparisons are type errors the typed engine rejects upstream.
        (SqlValue::Date(left), SqlValue::Date(right)) => left.cmp(right),
        (
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_),
            SqlValue::Date(_),
        ) => Ordering::Less,
        (
            SqlValue::Date(_),
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_),
        ) => Ordering::Greater,
        // Timestamp is the last tier (sorts after date); same-type compares by microsecond count.
        (SqlValue::Timestamp(left), SqlValue::Timestamp(right)) => left.cmp(right),
        (
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_)
            | SqlValue::Date(_),
            SqlValue::Timestamp(_),
        ) => Ordering::Less,
        (
            SqlValue::Timestamp(_),
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_)
            | SqlValue::Date(_),
        ) => Ordering::Greater,
        // Uuid is the final tier; same-type compares byte-wise (PG's uuid order).
        (SqlValue::Uuid(left), SqlValue::Uuid(right)) => left.cmp(right),
        (
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_)
            | SqlValue::Date(_)
            | SqlValue::Timestamp(_),
            SqlValue::Uuid(_),
        ) => Ordering::Less,
        (
            SqlValue::Uuid(_),
            SqlValue::Int4(_)
            | SqlValue::Int8(_)
            | SqlValue::Numeric(_)
            | SqlValue::Bool(_)
            | SqlValue::Text(_)
            | SqlValue::Date(_)
            | SqlValue::Timestamp(_),
        ) => Ordering::Greater,
    }
}

pub(crate) fn select_filter_matches(left: &SqlValue, op: SelectFilterOp, right: &SqlValue) -> bool {
    match op {
        // Eq is scale/type-aware like the ordering ops, so `numeric = int` matches across
        // the numeric tower. Bound filter literals are pre-coerced to the column type, but
        // routing Eq through compare_sql_values keeps it correct for any direct caller too.
        SelectFilterOp::Eq => compare_sql_values(left, right).is_eq(),
        SelectFilterOp::Lt => compare_sql_values(left, right).is_lt(),
        SelectFilterOp::Lte => !compare_sql_values(left, right).is_gt(),
        SelectFilterOp::Gt => compare_sql_values(left, right).is_gt(),
        SelectFilterOp::Gte => !compare_sql_values(left, right).is_lt(),
        SelectFilterOp::LikePrefix => match (left, right) {
            (SqlValue::Text(left), SqlValue::Text(prefix)) => left.starts_with(prefix),
            _ => false,
        },
    }
}

pub(crate) fn sort_relational_keys(keyed_rows: &mut [(String, SqlValue)], descending: bool) {
    keyed_rows.sort_by(|(left_key, left_value), (right_key, right_value)| {
        let value_order = compare_sql_values(left_value, right_value);
        let order = if descending {
            value_order.reverse()
        } else {
            value_order
        };
        order.then_with(|| left_key.cmp(right_key))
    });
}

pub(crate) fn relational_column_index(
    table: &RelationalTable,
    name: &str,
) -> Result<usize, ExecuteError> {
    table
        .columns
        .iter()
        .position(|column| column.name == name)
        .ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "column \"{}\" does not exist",
                name
            )))
        })
}

pub(crate) fn bind_relational_select(
    table: &RelationalTable,
    select: &Select,
) -> Result<BoundRelationalSelect, ExecuteError> {
    let selected_indexes = match &select.projection {
        SelectProjection::All => (0..table.columns.len()).collect::<Vec<_>>(),
        SelectProjection::Columns(columns) => columns
            .iter()
            .map(|name| relational_column_index(table, name))
            .collect::<Result<Vec<_>, _>>()?,
        SelectProjection::CountAll => Vec::new(),
        SelectProjection::GroupedCount { column } => vec![relational_column_index(table, column)?],
        SelectProjection::Sum { .. } => Vec::new(),
        SelectProjection::GroupedSum { group_column, .. } => {
            vec![relational_column_index(table, group_column)?]
        }
        SelectProjection::Avg { .. } => Vec::new(),
        SelectProjection::GroupedAvg { group_column, .. } => {
            vec![relational_column_index(table, group_column)?]
        }
        SelectProjection::Min { .. } | SelectProjection::Max { .. } => Vec::new(),
        SelectProjection::GroupedMin { group_column, .. }
        | SelectProjection::GroupedMax { group_column, .. } => {
            vec![relational_column_index(table, group_column)?]
        }
        SelectProjection::GroupedAggregates { group_column, .. } => {
            vec![relational_column_index(table, group_column)?]
        }
    };
    let mut selected_columns = selected_indexes
        .iter()
        .map(|idx| table.columns[*idx].clone())
        .collect::<Vec<_>>();
    if matches!(
        select.projection,
        SelectProjection::CountAll
            | SelectProjection::GroupedCount { .. }
            | SelectProjection::Sum { .. }
            | SelectProjection::GroupedSum { .. }
            | SelectProjection::Avg { .. }
            | SelectProjection::GroupedAvg { .. }
            | SelectProjection::Min { .. }
            | SelectProjection::GroupedMin { .. }
            | SelectProjection::Max { .. }
            | SelectProjection::GroupedMax { .. }
    ) {
        let aggregate_name = match &select.projection {
            SelectProjection::CountAll | SelectProjection::GroupedCount { .. } => "count",
            SelectProjection::Sum { .. } | SelectProjection::GroupedSum { .. } => "sum",
            SelectProjection::Avg { .. } | SelectProjection::GroupedAvg { .. } => "avg",
            SelectProjection::Min { .. } | SelectProjection::GroupedMin { .. } => "min",
            SelectProjection::Max { .. } | SelectProjection::GroupedMax { .. } => "max",
            SelectProjection::All
            | SelectProjection::Columns(_)
            | SelectProjection::GroupedAggregates { .. } => unreachable!(),
        };
        let (aggregate_ty, aggregate_type_oid, aggregate_type_size) = match &select.projection {
            // AVG yields a fixed-point numeric at scale 16 (numeric OID 1700).
            SelectProjection::Avg { .. } | SelectProjection::GroupedAvg { .. } => (
                SqlType::Numeric {
                    precision: NUMERIC_DEFAULT_PRECISION,
                    scale: AVG_RESULT_SCALE,
                },
                1700,
                -1,
            ),
            // COUNT is int8 (OID 20) regardless of the counted column (Phase-3 widening).
            SelectProjection::CountAll | SelectProjection::GroupedCount { .. } => {
                (SqlType::Int8, 20, 8)
            }
            // SUM: PG SUM(int8) -> numeric (the bigint sum can exceed int8); SUM(int4) keeps the
            // pre-existing (Int4 ty, oid 20, size 8) declaration (value widened to int8). NB:
            // aggregate_source_column intentionally returns None for SUM, so look the source column
            // up directly here -- relying on it silently fell through to the int4 default for int8.
            SelectProjection::Sum { column }
            | SelectProjection::GroupedSum {
                sum_column: column, ..
            } => {
                let idx = relational_column_index(table, column)?;
                match table.columns[idx].ty {
                    // SUM(int8) is an integer sum -> numeric scale 0 (OID 1700).
                    SqlType::Int8 => (
                        SqlType::Numeric {
                            precision: NUMERIC_DEFAULT_PRECISION,
                            scale: 0,
                        },
                        1700,
                        -1,
                    ),
                    // SUM(numeric) -> numeric at the column scale (the mantissas share that scale).
                    SqlType::Numeric { precision, scale } => {
                        (SqlType::Numeric { precision, scale }, 1700, -1)
                    }
                    _ => (SqlType::Int4, 20, 8),
                }
            }
            // MIN/MAX inherit the source column's wire type (PG preserves the type).
            _ => match aggregate_source_column(table, select)? {
                Some(column) => (column.ty, column.type_oid, column.type_size),
                None => (SqlType::Int4, 20, 8),
            },
        };
        let aggregate_attnum = selected_columns.len() as i16 + 1;
        selected_columns.push(RelationalColumn {
            id: 0,
            table_oid: table.oid,
            attnum: aggregate_attnum,
            name: aggregate_name.to_string(),
            ty: aggregate_ty,
            domain: None,
            default: None,
            type_oid: aggregate_type_oid,
            type_size: aggregate_type_size,
        });
    }
    // The general grouped form projects the group column (already in selected_columns) plus one result
    // column per aggregate. Each aggregate's wire type follows PG: COUNT->int8, AVG->numeric@16,
    // SUM(int8/numeric)->numeric, SUM(int*)->int4(bigint oid), MIN/MAX-> the source column's type.
    if let SelectProjection::GroupedAggregates { aggregates, .. } = &select.projection {
        for aggregate in aggregates {
            let (name, ty, type_oid, type_size) = match aggregate.kind {
                GroupedAggKind::Count => ("count", SqlType::Int8, 20, 8),
                GroupedAggKind::Avg => (
                    "avg",
                    SqlType::Numeric {
                        precision: NUMERIC_DEFAULT_PRECISION,
                        scale: AVG_RESULT_SCALE,
                    },
                    1700,
                    -1,
                ),
                GroupedAggKind::Sum => {
                    let column = aggregate.value_column.as_ref().ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "grouped SUM requires a value column".to_string(),
                        ))
                    })?;
                    let idx = relational_column_index(table, column)?;
                    match table.columns[idx].ty {
                        SqlType::Int8 => (
                            "sum",
                            SqlType::Numeric {
                                precision: NUMERIC_DEFAULT_PRECISION,
                                scale: 0,
                            },
                            1700,
                            -1,
                        ),
                        SqlType::Numeric { precision, scale } => {
                            ("sum", SqlType::Numeric { precision, scale }, 1700, -1)
                        }
                        _ => ("sum", SqlType::Int4, 20, 8),
                    }
                }
                GroupedAggKind::Min | GroupedAggKind::Max => {
                    let name = if matches!(aggregate.kind, GroupedAggKind::Min) {
                        "min"
                    } else {
                        "max"
                    };
                    let column = aggregate.value_column.as_ref().ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "grouped MIN/MAX requires a value column".to_string(),
                        ))
                    })?;
                    let idx = relational_column_index(table, column)?;
                    let col = &table.columns[idx];
                    (name, col.ty, col.type_oid, col.type_size)
                }
            };
            let attnum = selected_columns.len() as i16 + 1;
            selected_columns.push(RelationalColumn {
                id: 0,
                table_oid: table.oid,
                attnum,
                name: name.to_string(),
                ty,
                domain: None,
                default: None,
                type_oid,
                type_size,
            });
        }
    }
    let raw_filter_groups = if select.filter_groups.is_empty() {
        let filter_refs = if select.filters.is_empty() {
            select.filter.iter().cloned().collect::<Vec<_>>()
        } else {
            select.filters.clone()
        };
        if filter_refs.is_empty() {
            Vec::new()
        } else {
            vec![filter_refs]
        }
    } else {
        select.filter_groups.clone()
    };
    let filter_groups = raw_filter_groups
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .map(|filter| {
                    relational_column_index(table, &filter.column).map(|idx| {
                        // PG implicitly casts the literal to the column type across the
                        // integer/numeric tower, so `WHERE bal = 5` matches a numeric
                        // column and the equality index probe keys on the right slot.
                        let value = coerce_filter_literal(filter.value, table.columns[idx].ty);
                        (idx, filter.op, value)
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let filters = filter_groups.first().cloned().unwrap_or_default();
    let filter = filters.first().cloned();
    let order = select
        .order_by
        .as_ref()
        .map(|order| {
            if select_is_aggregate_result_column(select, &order.column) {
                Ok((usize::MAX, order.descending))
            } else {
                relational_column_index(table, &order.column).map(|idx| (idx, order.descending))
            }
        })
        .transpose()?;
    if select.distinct {
        match &select.projection {
            SelectProjection::All => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "SELECT DISTINCT * is unsupported".to_string(),
                )));
            }
            SelectProjection::Columns(columns) => {
                if let Some(order) = &select.order_by {
                    if !columns.iter().any(|column| column == &order.column) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "SELECT DISTINCT ORDER BY must reference a selected column".to_string(),
                        )));
                    }
                }
            }
            SelectProjection::CountAll
            | SelectProjection::GroupedCount { .. }
            | SelectProjection::Sum { .. }
            | SelectProjection::GroupedSum { .. }
            | SelectProjection::Avg { .. }
            | SelectProjection::GroupedAvg { .. }
            | SelectProjection::Min { .. }
            | SelectProjection::GroupedMin { .. }
            | SelectProjection::Max { .. }
            | SelectProjection::GroupedMax { .. }
            | SelectProjection::GroupedAggregates { .. } => unreachable!(),
        }
    }
    let group_by_index = if let Some(group_by) = &select.group_by {
        Some(relational_column_index(table, group_by)?)
    } else {
        None
    };
    match (&select.projection, group_by_index) {
        (SelectProjection::CountAll, None) => {}
        (SelectProjection::CountAll, Some(_)) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "GROUP BY requires grouped COUNT(*) projection".to_string(),
            )));
        }
        (SelectProjection::GroupedCount { column }, Some(idx)) => {
            let projected_idx = relational_column_index(table, column)?;
            if projected_idx != idx {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "GROUP BY column must match grouped COUNT(*) projection".to_string(),
                )));
            }
            if let Some(order) = &select.order_by {
                if order.column != *column && !order.column.eq_ignore_ascii_case("count") {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "GROUP BY ORDER BY must reference grouped column or count".to_string(),
                    )));
                }
            }
        }
        (SelectProjection::GroupedCount { .. }, None) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "grouped COUNT(*) requires GROUP BY".to_string(),
            )));
        }
        (SelectProjection::Sum { column }, None) => {
            validate_sum_column(table, column)?;
            if let Some(order) = &select.order_by {
                if !order.column.eq_ignore_ascii_case("sum") {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "SUM ORDER BY only supports sum".to_string(),
                    )));
                }
            }
        }
        (SelectProjection::Sum { .. }, Some(_)) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "GROUP BY requires grouped SUM projection".to_string(),
            )));
        }
        (
            SelectProjection::GroupedSum {
                group_column,
                sum_column,
            },
            Some(idx),
        ) => {
            let projected_idx = relational_column_index(table, group_column)?;
            if projected_idx != idx {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "GROUP BY column must match grouped SUM projection".to_string(),
                )));
            }
            validate_sum_column(table, sum_column)?;
            if let Some(order) = &select.order_by {
                if order.column != *group_column && !order.column.eq_ignore_ascii_case("sum") {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "GROUP BY ORDER BY must reference grouped column or sum".to_string(),
                    )));
                }
            }
        }
        (SelectProjection::GroupedSum { .. }, None) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "grouped SUM requires GROUP BY".to_string(),
            )));
        }
        (SelectProjection::Avg { column }, None) => {
            validate_avg_column(table, column)?;
            if let Some(order) = &select.order_by {
                if !order.column.eq_ignore_ascii_case("avg") {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "AVG ORDER BY only supports avg".to_string(),
                    )));
                }
            }
        }
        (SelectProjection::Avg { .. }, Some(_)) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "GROUP BY requires grouped AVG projection".to_string(),
            )));
        }
        (
            SelectProjection::GroupedAvg {
                group_column,
                avg_column,
            },
            Some(idx),
        ) => {
            let projected_idx = relational_column_index(table, group_column)?;
            if projected_idx != idx {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "GROUP BY column must match grouped AVG projection".to_string(),
                )));
            }
            validate_avg_column(table, avg_column)?;
            if let Some(order) = &select.order_by {
                if order.column != *group_column && !order.column.eq_ignore_ascii_case("avg") {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "GROUP BY ORDER BY must reference grouped column or avg".to_string(),
                    )));
                }
            }
        }
        (SelectProjection::GroupedAvg { .. }, None) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "grouped AVG requires GROUP BY".to_string(),
            )));
        }
        (SelectProjection::Min { column } | SelectProjection::Max { column }, None) => {
            relational_column_index(table, column)?;
            let aggregate_name = if matches!(select.projection, SelectProjection::Min { .. }) {
                "min"
            } else {
                "max"
            };
            if let Some(order) = &select.order_by {
                if !order.column.eq_ignore_ascii_case(aggregate_name) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "MIN/MAX ORDER BY only supports the aggregate result".to_string(),
                    )));
                }
            }
        }
        (SelectProjection::Min { .. } | SelectProjection::Max { .. }, Some(_)) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "GROUP BY requires grouped MIN/MAX projection".to_string(),
            )));
        }
        (
            SelectProjection::GroupedMin {
                group_column,
                min_column,
            },
            Some(idx),
        ) => {
            validate_grouped_extreme(table, select, group_column, min_column, idx, "min")?;
        }
        (
            SelectProjection::GroupedMax {
                group_column,
                max_column,
            },
            Some(idx),
        ) => {
            validate_grouped_extreme(table, select, group_column, max_column, idx, "max")?;
        }
        (SelectProjection::GroupedMin { .. } | SelectProjection::GroupedMax { .. }, None) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "grouped MIN/MAX requires GROUP BY".to_string(),
            )));
        }
        (
            SelectProjection::GroupedAggregates {
                group_column,
                aggregates,
            },
            Some(idx),
        ) => {
            let projected_idx = relational_column_index(table, group_column)?;
            if projected_idx != idx {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "GROUP BY column must match the grouped projection".to_string(),
                )));
            }
            for aggregate in aggregates {
                match aggregate.kind {
                    GroupedAggKind::Count => {}
                    GroupedAggKind::Sum => {
                        let column = aggregate.value_column.as_ref().ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(
                                "grouped SUM requires a value column".to_string(),
                            ))
                        })?;
                        validate_sum_column(table, column)?;
                    }
                    GroupedAggKind::Avg => {
                        let column = aggregate.value_column.as_ref().ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(
                                "grouped AVG requires a value column".to_string(),
                            ))
                        })?;
                        validate_avg_column(table, column)?;
                    }
                    GroupedAggKind::Min | GroupedAggKind::Max => {
                        // The device value-type classification rejects unsupported MIN/MAX types; here
                        // just confirm the value column exists.
                        let column = aggregate.value_column.as_ref().ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(
                                "grouped MIN/MAX requires a value column".to_string(),
                            ))
                        })?;
                        relational_column_index(table, column)?;
                    }
                }
            }
        }
        (SelectProjection::GroupedAggregates { .. }, None) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "grouped aggregates require GROUP BY".to_string(),
            )));
        }
        (SelectProjection::All | SelectProjection::Columns(_), Some(_)) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "GROUP BY requires COUNT(*) projection".to_string(),
            )));
        }
        (SelectProjection::All | SelectProjection::Columns(_), None) => {}
    }

    Ok(BoundRelationalSelect {
        selected_columns,
        selected_indexes,
        group_by_index,
        filter,
        filters,
        filter_groups,
        order,
    })
}

pub(crate) type BoundDeleteFilter = (usize, SelectFilterOp, SqlValue);
pub(crate) type BoundDeleteFilterGroup = Vec<BoundDeleteFilter>;
pub(crate) type BoundUpdateAssignment = (usize, SqlValue);

pub(crate) fn bind_delete_filter_groups(
    table: &RelationalTable,
    delete: &Delete,
) -> Result<Vec<BoundDeleteFilterGroup>, ExecuteError> {
    let raw_filter_groups = if delete.filter_groups.is_empty() {
        vec![delete.filters.clone()]
    } else {
        delete.filter_groups.clone()
    };
    if raw_filter_groups.is_empty() || raw_filter_groups.iter().any(Vec::is_empty) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "DELETE requires WHERE filters".to_string(),
        )));
    }
    raw_filter_groups
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .map(|filter| {
                    let idx = relational_column_index(table, &filter.column)?;
                    // Coerce across the integer/numeric tower for parity with SELECT; a
                    // value with no implicit cast to the column type still errors loudly.
                    let value = coerce_filter_literal(filter.value, table.columns[idx].ty);
                    if !sql_value_matches_type(&value, table.columns[idx].ty) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "invalid value for column \"{}\"",
                            filter.column
                        ))));
                    }
                    Ok((idx, filter.op, value))
                })
                .collect()
        })
        .collect()
}

pub(crate) fn bind_update_assignments(
    table: &RelationalTable,
    update: &Update,
) -> Result<Vec<BoundUpdateAssignment>, ExecuteError> {
    let mut seen = BTreeSet::new();
    update
        .assignments
        .iter()
        .map(|assignment| {
            let idx = relational_column_index(table, &assignment.column)?;
            if !seen.insert(idx) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column \"{}\" assigned more than once",
                    assignment.column
                ))));
            }
            // Coerce + rescale to the column type (widen int→numeric/int8, rescale a
            // numeric to the declared scale, enforce precision) — parity with INSERT.
            let value = coerce_insert_value(
                assignment.value.clone(),
                table.columns[idx].ty,
                &assignment.column,
            )
            .map_err(ExecuteError::Engine)?;
            Ok((idx, value))
        })
        .collect()
}

pub(crate) fn relational_select_pushes_limit(select: &Select) -> bool {
    select.limit.is_some() && select.offset.is_none() && select.order_by.is_none()
}

pub(crate) fn relational_select_pushed_limit(
    select: &Select,
    ordered_access_path: bool,
) -> Option<usize> {
    if select.distinct || select_is_aggregate(select) {
        return None;
    }
    let can_push = select.order_by.is_none() || ordered_access_path;
    if !can_push {
        return None;
    }
    select
        .limit
        .map(|limit| limit.saturating_add(select.offset.unwrap_or(0)))
}

pub(crate) fn relational_select_limit_satisfied_by_access_path(
    select: &Select,
    access_path: &RelationalAccessPath,
) -> bool {
    if select.distinct || select_is_aggregate(select) {
        return false;
    }
    select.offset.is_none()
        && (relational_select_pushes_limit(select)
            || (select.limit.is_some()
                && matches!(access_path, RelationalAccessPath::OrderedKeyBatch { .. })))
}

pub(crate) fn relational_select_needs_host_sql_finalization(
    select: &Select,
    access_path: &RelationalAccessPath,
) -> bool {
    (select_has_relational_filters(select)
        && !matches!(
            access_path,
            RelationalAccessPath::EqualityIndex { .. }
                | RelationalAccessPath::FilteredKeyBatch { .. }
                | RelationalAccessPath::ConjunctiveFilteredKeyBatch { .. }
                | RelationalAccessPath::DisjunctiveFilteredKeyBatch { .. }
                | RelationalAccessPath::OrderedKeyBatch {
                    predicate_column: Some(_),
                    ..
                }
        ))
        || (select.order_by.is_some()
            && !matches!(access_path, RelationalAccessPath::OrderedKeyBatch { .. }))
        || (select.limit.is_some()
            && select.offset.is_none()
            && !select.distinct
            && !relational_select_limit_satisfied_by_access_path(select, access_path))
}

pub(crate) fn select_has_relational_filters(select: &Select) -> bool {
    select.filter.is_some() || !select.filters.is_empty() || !select.filter_groups.is_empty()
}

pub(crate) fn select_is_plain_view_scan(select: &Select) -> bool {
    !select.distinct
        && matches!(select.projection, SelectProjection::All)
        && select.group_by.is_none()
        && select.having_groups.is_empty()
        && !select_has_relational_filters(select)
        && select.order_by.is_none()
        && select.limit.is_none()
        && select.offset.is_none()
}

pub(crate) fn select_is_aggregate(select: &Select) -> bool {
    matches!(
        select.projection,
        SelectProjection::CountAll
            | SelectProjection::GroupedCount { .. }
            | SelectProjection::Sum { .. }
            | SelectProjection::GroupedSum { .. }
            | SelectProjection::Avg { .. }
            | SelectProjection::GroupedAvg { .. }
            | SelectProjection::Min { .. }
            | SelectProjection::GroupedMin { .. }
            | SelectProjection::Max { .. }
            | SelectProjection::GroupedMax { .. }
    )
}

pub(crate) fn select_is_aggregate_result_column(select: &Select, column: &str) -> bool {
    match select.projection {
        SelectProjection::CountAll | SelectProjection::GroupedCount { .. } => {
            column.eq_ignore_ascii_case("count")
        }
        SelectProjection::Sum { .. } | SelectProjection::GroupedSum { .. } => {
            column.eq_ignore_ascii_case("sum")
        }
        SelectProjection::Avg { .. } | SelectProjection::GroupedAvg { .. } => {
            column.eq_ignore_ascii_case("avg")
        }
        SelectProjection::Min { .. } | SelectProjection::GroupedMin { .. } => {
            column.eq_ignore_ascii_case("min")
        }
        SelectProjection::Max { .. } | SelectProjection::GroupedMax { .. } => {
            column.eq_ignore_ascii_case("max")
        }
        // GroupedAggregates is produced only on the Expr path, which does not route through this
        // CPU-path HAVING/ORDER-BY helper; never reached for it.
        SelectProjection::All
        | SelectProjection::Columns(_)
        | SelectProjection::GroupedAggregates { .. } => false,
    }
}

pub(crate) fn select_aggregate_result_column_name(select: &Select) -> Option<&'static str> {
    match select.projection {
        SelectProjection::CountAll | SelectProjection::GroupedCount { .. } => Some("count"),
        SelectProjection::Sum { .. } | SelectProjection::GroupedSum { .. } => Some("sum"),
        SelectProjection::Avg { .. } | SelectProjection::GroupedAvg { .. } => Some("avg"),
        SelectProjection::Min { .. } | SelectProjection::GroupedMin { .. } => Some("min"),
        SelectProjection::Max { .. } | SelectProjection::GroupedMax { .. } => Some("max"),
        // GroupedAggregates has N aggregates (no single result-column name); Expr-path only.
        SelectProjection::All
        | SelectProjection::Columns(_)
        | SelectProjection::GroupedAggregates { .. } => None,
    }
}

pub(crate) fn grouped_row_matches_having(
    select: &Select,
    group_column: &str,
    group_value: &SqlValue,
    aggregate_name: &'static str,
    aggregate_value: &SqlValue,
) -> Result<bool, ExecuteError> {
    if select.having_groups.is_empty() {
        return Ok(true);
    }

    for filters in &select.having_groups {
        let mut group_matches = true;
        for filter in filters {
            let value = if filter.column == group_column {
                group_value
            } else if filter.column.eq_ignore_ascii_case(aggregate_name) {
                aggregate_value
            } else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "HAVING must reference grouped column or aggregate result".to_string(),
                )));
            };
            if !select_filter_matches(value, filter.op, &filter.value) {
                group_matches = false;
                break;
            }
        }
        if group_matches {
            return Ok(true);
        }
    }

    Ok(false)
}

pub(crate) fn aggregate_source_column<'a>(
    table: &'a RelationalTable,
    select: &Select,
) -> Result<Option<&'a RelationalColumn>, ExecuteError> {
    let column_name = match &select.projection {
        SelectProjection::Min { column } | SelectProjection::Max { column } => Some(column),
        SelectProjection::GroupedMin { min_column, .. } => Some(min_column),
        SelectProjection::GroupedMax { max_column, .. } => Some(max_column),
        SelectProjection::CountAll
        | SelectProjection::GroupedCount { .. }
        | SelectProjection::Sum { .. }
        | SelectProjection::GroupedSum { .. }
        | SelectProjection::Avg { .. }
        | SelectProjection::GroupedAvg { .. }
        | SelectProjection::All
        | SelectProjection::Columns(_)
        | SelectProjection::GroupedAggregates { .. } => None,
    };
    let Some(column_name) = column_name else {
        return Ok(None);
    };
    let idx = relational_column_index(table, column_name)?;
    Ok(table.columns.get(idx))
}

pub(crate) fn validate_grouped_extreme(
    table: &RelationalTable,
    select: &Select,
    group_column: &str,
    value_column: &str,
    group_by_idx: usize,
    aggregate_name: &str,
) -> Result<(), ExecuteError> {
    let projected_idx = relational_column_index(table, group_column)?;
    if projected_idx != group_by_idx {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "GROUP BY column must match grouped MIN/MAX projection".to_string(),
        )));
    }
    relational_column_index(table, value_column)?;
    if let Some(order) = &select.order_by {
        if order.column != group_column && !order.column.eq_ignore_ascii_case(aggregate_name) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "GROUP BY ORDER BY must reference grouped column or min/max".to_string(),
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_sum_column(
    table: &RelationalTable,
    column: &str,
) -> Result<usize, ExecuteError> {
    validate_int4_aggregate_column(table, column, "SUM")
}

pub(crate) fn validate_avg_column(
    table: &RelationalTable,
    column: &str,
) -> Result<usize, ExecuteError> {
    validate_int4_aggregate_column(table, column, "AVG")
}

pub(crate) fn validate_int4_aggregate_column(
    table: &RelationalTable,
    column: &str,
    aggregate: &'static str,
) -> Result<usize, ExecuteError> {
    let idx = relational_column_index(table, column)?;
    // SUM/AVG accept int2 / int4 / int8 / numeric on the general GPU executor (int2/int4 share the
    // int4 read; int8 + numeric reduce to i128). The enumerated path (int4-only) rejects the wider
    // types later at execution -- so this is still a hard error there, just not at validation.
    if !matches!(
        table.columns[idx].ty,
        SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. }
    ) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            aggregate_int4_error_message(aggregate).to_string(),
        )));
    }
    Ok(idx)
}

pub(crate) fn aggregate_int4_error_message(aggregate: &str) -> &'static str {
    match aggregate {
        "AVG" => "AVG supports int2 / int4 / int8 / numeric columns",
        _ => "SUM supports int2 / int4 / int8 / numeric columns",
    }
}

pub(crate) fn int4_aggregate_value(
    value: &SqlValue,
    aggregate: &'static str,
) -> Result<i32, ExecuteError> {
    match value {
        SqlValue::Int4(value) => Ok(*value),
        SqlValue::Int2(_)
        | SqlValue::Int8(_)
        | SqlValue::Numeric(_)
        | SqlValue::Bool(_)
        | SqlValue::Text(_)
        | SqlValue::Date(_)
        | SqlValue::Timestamp(_)
        | SqlValue::Uuid(_) => {
            Err(ExecuteError::Engine(EngineError::ApplyFailed(
                aggregate_int4_error_message(aggregate).to_string(),
            )))
        }
    }
}

/// PostgreSQL's `numeric` AVG (division) targets ~16 SIGNIFICANT digits (NUMERIC_MIN_SIG_DIGITS), so
/// the RESULT SCALE is dynamic (keyed to the quotient magnitude), not fixed. We keep 16 as the
/// significant-digit base + the AVG result COLUMN's display-scale hint (PG reports AVG as unconstrained
/// `numeric`, so the column scale is cosmetic; the value carries its own scale). See
/// [`average_sql_value`].
pub(crate) const AVG_RESULT_SCALE: u8 = 16;

/// Decimal digit count of a non-negative i128 (`n >= 1` -> `>= 1`).
fn decimal_digit_count(mut n: i128) -> i32 {
    let mut digits = 1;
    while n >= 10 {
        n /= 10;
        digits += 1;
    }
    digits
}

/// `AVG = sum / count` as a `numeric`, matching PostgreSQL exactly: a DYNAMIC result scale (PG's
/// `select_div_scale`) plus the final digit ROUNDED half-away-from-zero. PG picks a scale giving ~16
/// significant digits: `rscale = max(0, 16 - 4*floor(dw/4))` where `dw = floor(log10(|sum/count|))` --
/// so a 1-4-digit integer part -> scale 16, 5-8 -> 12, 9-12 -> 8, ...; a sub-1 quotient -> 20, 24, ...
/// (The earlier fixed-scale-16 FLOOR diverged from PG on most fractional + large averages -- an AVG
/// audit P0, shared with the enumerated path.) The long division is interleaved digit-by-digit to
/// avoid an `abs_sum * 10^rscale` i128 overflow; `rscale` shrinks as the quotient grows, so the
/// mantissa stays well within i128. `count == 0` is the empty-aggregate sentinel (callers needing SQL
/// NULL guard it upstream).
pub(crate) fn average_sql_value(sum: i128, count: usize) -> SqlValue {
    // AVG(int/int8) is AVG over a scale-0 dividend; share the (PG-exact) numeric AVG path so the
    // division scale + rounding match PostgreSQL identically (the int path had the same scale bug).
    avg_numeric_sql_value(sum, count, 0)
}

/// PostgreSQL's `select_div_scale` (numeric.c) for `SUM / count`: the display scale of the quotient.
/// The dividend is `|sum|` at scale S (its numeric value is `|sum| / 10^S`); the divisor is `count`
/// (an integer, scale 0). PG estimates the quotient weight in base-10000 (DEC_DIGITS = 4) units:
/// `qweight = w1 - w2`, DECREMENTED by 1 when the dividend's leading base-10000 digit <= the
/// divisor's, then `rscale = clamp(16 - 4*qweight, max(S, 0), 255)`. The leading-digit decrement is
/// the subtle part a naive "quotient decimal weight" derivation got WRONG -- it diverged from PG on
/// e.g. `AVG(1.00, 1.00, 1.00)` = 1.0 (PG renders scale 20, not 16) and every zero-sum. Verified
/// against PostgreSQL across 340+ (scale, magnitude, sign, zero-sum, large-count) cases. (Clamped to
/// 255 because `Decimal128`'s scale is a u8; PG's 1000 cap only bites for sub-10^-60 quotients, which
/// `Decimal128` cannot represent anyway.)
fn pg_div_result_scale(abs_sum: i128, count: i128, dividend_scale: u8) -> i32 {
    let s = i32::from(dividend_scale);
    // Leading base-10000 digit of `abs_val / 10^total_scale` at NBASE weight `w`. The result is a
    // single NBASE digit in [0, 9999], so the exp<0 multiply cannot overflow (abs_val <= 9999 there).
    let nbase_lead = |abs_val: i128, total_scale: i32, w: i32| -> i128 {
        let exp = total_scale + 4 * w;
        if exp >= 0 {
            abs_val / 10i128.pow(exp as u32)
        } else {
            abs_val * 10i128.pow((-exp) as u32)
        }
    };
    let (w1, fd1) = if abs_sum == 0 {
        (0, 0)
    } else {
        let dwt1 = (decimal_digit_count(abs_sum) - 1) - s;
        let w1 = dwt1.div_euclid(4);
        (w1, nbase_lead(abs_sum, s, w1))
    };
    let dwt2 = decimal_digit_count(count) - 1;
    let w2 = dwt2.div_euclid(4);
    let fd2 = nbase_lead(count, 0, w2);
    let mut qweight = w1 - w2;
    if fd1 <= fd2 {
        qweight -= 1;
    }
    (16 - 4 * qweight).max(s).clamp(0, i32::from(u8::MAX))
}

/// `AVG(numeric)` = SUM / count where SUM is the i128 mantissa at `column_scale` S, i.e. the true
/// average is `sum_mantissa / (count * 10^S)`. PostgreSQL picks the display scale via
/// [`pg_div_result_scale`] and rounds half-away-from-zero. Long-divides `|sum| * 10^(rscale-S) /
/// count` (rscale >= S so the exponent is >= 0), interleaved to avoid an `abs_sum * 10^rscale`
/// overflow (the mantissa stays near 10^16..10^19 as the quotient grows).
pub(crate) fn avg_numeric_sql_value(sum_mantissa: i128, count: usize, column_scale: u8) -> SqlValue {
    if count == 0 {
        return SqlValue::Numeric(Decimal128::new(0, column_scale));
    }
    let count = count as i128;
    let s = i32::from(column_scale);
    let negative = sum_mantissa.is_negative();
    let abs_sum = sum_mantissa.abs();
    let rscale = pg_div_result_scale(abs_sum, count, column_scale);
    let p = rscale - s;
    let mut mantissa = abs_sum / count;
    let mut remainder = abs_sum % count;
    for _ in 0..p {
        remainder *= 10;
        mantissa = mantissa * 10 + remainder / count;
        remainder %= count;
    }
    if 2 * remainder >= count {
        mantissa += 1;
    }
    if negative {
        mantissa = -mantissa;
    }
    SqlValue::Numeric(Decimal128::new(mantissa, rscale as u8))
}

pub(crate) fn current_timestamp_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_micros()
        .try_into()
        .unwrap_or(u64::MAX)
}
