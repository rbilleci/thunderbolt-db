//! Relational execution helpers — pure free functions extracted from `lib.rs`
//! (P0 §9.6 decomposition, behavior-preserving). Value coercion, catalog
//! synthesis, relational row codec, SELECT/DML binding, and aggregate
//! validation. No `Engine` state, no locks: all inputs are explicit.

use super::*;

mod datetime;
mod row_codec;
// Preserve the former crate-private facade for codec-focused tests and future internal callers;
// some build modes do not consume the direct cell/split helpers.
pub(crate) use datetime::{coerce_datetime_text, validate_datetime_carrier};
#[allow(unused_imports)]
pub(crate) use row_codec::{
    append_relational_cell, decode_relational_row, decode_relational_value, encode_relational_row,
    split_escaped_row, RelationalCellRef,
};

pub(crate) fn sql_value_matches_type(value: &SqlValue, ty: SqlType) -> bool {
    matches!(
        (value, ty),
        // NULL is the typeless SQL null — valid for a column of ANY type (every column is nullable;
        // `NOT NULL` is not accepted by the parser). Coercion passes it through unchanged.
        (SqlValue::Null, _)
            | (SqlValue::Int4(_), SqlType::Int4)
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
            .map_err(|_| EngineError::NumericValueOutOfRange("smallint out of range".to_string()));
    }
    // SQL literal inference promotes an integer that exceeds int4 to `Int8`. Preserve the
    // existing no-narrowing type rule for in-range `Int8`, but distinguish an actual int4 range
    // overflow from a generic type mismatch so every relational INSERT path reaches 22003.
    if let (SqlValue::Int8(v), SqlType::Int4) = (&value, ty) {
        if i32::try_from(*v).is_err() {
            return Err(EngineError::NumericValueOutOfRange(
                "integer out of range".to_string(),
            ));
        }
    }

    // A string literal assigned to a date/timestamp column is parsed as that type (PG coerces an
    // unknown-type literal to the column type). An already-typed value passes through the check below.
    if let SqlValue::Text(text) = &value {
        match ty {
            SqlType::Date => {
                return coerce_datetime_text(text, ty);
            }
            SqlType::Timestamp => {
                return coerce_datetime_text(text, ty);
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
    validate_datetime_carrier(&value, ty)?;
    match (value, ty) {
        (SqlValue::Numeric(decimal), SqlType::Numeric { precision, scale }) => {
            let rescaled = decimal.rescale(scale).map_err(|_| {
                EngineError::NumericValueOutOfRange("numeric field overflow".to_string())
            })?;
            if numeric_exceeds_precision(rescaled.mantissa, precision) {
                return Err(EngineError::NumericValueOutOfRange(
                    "numeric field overflow".to_string(),
                ));
            }
            Ok(SqlValue::Numeric(rescaled))
        }
        (value, _) => Ok(value),
    }
}

/// Resolve a CHECK predicate literal and its concrete catalog RHS input/cast type.
///
/// This is intentionally NOT an INSERT assignment cast: unknown SQL strings are parsed at their
/// comparison target without NUMERIC typmod rounding, while known numeric-tower values use only
/// comparison coercions. The raw AST/WAL command remains the retry/recovery authority.
pub(crate) fn resolve_check_comparison_operand(
    value: SqlValue,
    provenance: CheckLiteralProvenance,
    column_ty: SqlType,
    op: SelectFilterOp,
    column_name: &str,
) -> Result<(SqlValue, SqlType), EngineError> {
    let invalid_value =
        || EngineError::ApplyFailed(format!("invalid value for column \"{column_name}\""));
    let provenance = normalize_check_literal_provenance(&value, provenance, &invalid_value)?;
    let literal = match provenance {
        CheckLiteralProvenance::Unknown => {
            resolve_unknown_check_literal(value, column_ty, column_name)?
        }
        CheckLiteralProvenance::Known(right_base) => {
            if !matches!(value, SqlValue::Null) && !literal_matches_type(&value, right_base) {
                return Err(invalid_value());
            }
            if !check_comparison_types_compatible(column_ty, right_base) {
                return Err(EngineError::UndefinedOperator(format!(
                    "{} {} {} for CHECK column \"{column_name}\"",
                    column_ty.catalog_name(),
                    check_operator_name(op),
                    right_base.catalog_name()
                )));
            }
            resolve_known_check_literal(value, column_ty, right_base)
        }
        CheckLiteralProvenance::LegacyAmbiguous => {
            unreachable!("legacy provenance normalized above")
        }
    };

    if op == SelectFilterOp::LikePrefix
        && !(column_ty == SqlType::Text && matches!(literal, SqlValue::Text(_)))
    {
        return Err(invalid_value());
    }
    Ok((
        literal.clone(),
        resolved_check_operand_type(provenance, column_ty, &literal),
    ))
}

fn resolved_check_operand_type(
    provenance: CheckLiteralProvenance,
    column_ty: SqlType,
    literal: &SqlValue,
) -> SqlType {
    match (provenance, literal) {
        // Unknown binds at the comparison target, including NULL.
        (CheckLiteralProvenance::Unknown, _) => column_ty,
        // Preserve a NUMERIC cast typmod: it controls deferred CHECK evaluation.
        (CheckLiteralProvenance::Known(ty @ SqlType::Numeric { .. }), SqlValue::Numeric(_)) => ty,
        // Same-family inputs retain their explicit/inferred type. Cross numeric inputs are
        // represented by the resolved operand type so the catalog cannot carry mismatched value
        // and metadata shapes.
        (CheckLiteralProvenance::Known(ty), SqlValue::Null) => ty,
        (CheckLiteralProvenance::Known(ty), value) if literal_matches_type(value, ty) => ty,
        (CheckLiteralProvenance::Known(_), value) => {
            natural_check_literal_type(value).unwrap_or(column_ty)
        }
        (CheckLiteralProvenance::LegacyAmbiguous, _) => {
            unreachable!("legacy provenance normalized")
        }
    }
}

/// Resolve a CHECK predicate literal where the caller needs only the comparison value. DDL
/// publication uses [`resolve_check_comparison_operand`] so catalog metadata shares the exact
/// same legacy normalization rule.
pub(crate) fn resolve_check_comparison_literal(
    value: SqlValue,
    provenance: CheckLiteralProvenance,
    column_ty: SqlType,
    op: SelectFilterOp,
    column_name: &str,
) -> Result<SqlValue, EngineError> {
    resolve_check_comparison_operand(value, provenance, column_ty, op, column_name)
        .map(|(value, _)| value)
}

fn normalize_check_literal_provenance(
    value: &SqlValue,
    provenance: CheckLiteralProvenance,
    invalid_value: &dyn Fn() -> EngineError,
) -> Result<CheckLiteralProvenance, EngineError> {
    match provenance {
        // Pre-provenance WAL cannot tell an uncast string from explicit `::text`. Keep the old
        // scalar shapes compatible: non-text values retain their natural input type, while Text
        // and NULL retain the historical target-directed/unknown handling. New parser output
        // never uses this variant.
        CheckLiteralProvenance::LegacyAmbiguous => match value {
            SqlValue::Text(_) | SqlValue::Null => Ok(CheckLiteralProvenance::Unknown),
            _ => natural_check_literal_type(value)
                .map(CheckLiteralProvenance::Known)
                .ok_or_else(invalid_value),
        },
        provenance => Ok(provenance),
    }
}

fn resolve_unknown_check_literal(
    value: SqlValue,
    column_ty: SqlType,
    column_name: &str,
) -> Result<SqlValue, EngineError> {
    // An uncast NULL is unknown and is accepted for every comparable target. The CHECK verdict
    // machinery later turns its comparison into UNKNOWN, which satisfies the constraint.
    if matches!(value, SqlValue::Null) {
        return Ok(value);
    }
    let SqlValue::Text(text) = value else {
        return Err(EngineError::ApplyFailed(format!(
            "invalid value for column \"{column_name}\""
        )));
    };
    let value = match column_ty {
        SqlType::Int2 | SqlType::Int4 | SqlType::Int8 => {
            let parsed = text.parse::<i128>().map_err(|_| {
                EngineError::InvalidTextRepresentation(format!(
                    "invalid input syntax for type {}: \"{text}\"",
                    column_ty.catalog_name()
                ))
            })?;
            match column_ty {
                SqlType::Int2 => i16::try_from(parsed).map(SqlValue::Int2).map_err(|_| {
                    EngineError::NumericValueOutOfRange("smallint out of range".to_string())
                })?,
                SqlType::Int4 => i32::try_from(parsed).map(SqlValue::Int4).map_err(|_| {
                    EngineError::NumericValueOutOfRange("integer out of range".to_string())
                })?,
                SqlType::Int8 => i64::try_from(parsed).map(SqlValue::Int8).map_err(|_| {
                    EngineError::NumericValueOutOfRange("bigint out of range".to_string())
                })?,
                _ => unreachable!(),
            }
        }
        SqlType::Numeric { .. } => {
            Decimal128::parse(&text)
                .map(SqlValue::Numeric)
                .ok_or_else(|| {
                    EngineError::InvalidTextRepresentation(format!(
                        "invalid input syntax for type numeric: \"{text}\""
                    ))
                })?
        }
        SqlType::Bool => gpu_db_sql::parse_bool_value(&text)
            .map(SqlValue::Bool)
            .ok_or_else(|| {
                EngineError::InvalidTextRepresentation(format!(
                    "invalid input syntax for type boolean: \"{text}\""
                ))
            })?,
        SqlType::Text => SqlValue::Text(text),
        SqlType::Date | SqlType::Timestamp => coerce_datetime_text(&text, column_ty)?,
        SqlType::Uuid => gpu_db_sql::uuid::parse_uuid(&text)
            .map(SqlValue::Uuid)
            .ok_or_else(|| {
                EngineError::InvalidTextRepresentation(format!(
                    "invalid input syntax for type uuid: \"{text}\""
                ))
            })?,
    };
    Ok(value)
}

/// Apply only the comparison matrix for a known literal. This deliberately has no dependency on
/// the generic filter coercer: CHECK DDL must preserve its source value unless the comparison
/// itself promotes the integer side to numeric. Operator-base binding is not catalog state in this
/// slice; the matrix is kept here so a later device lowering can persist it without changing DDL
/// semantics.
fn resolve_known_check_literal(value: SqlValue, left: SqlType, right: SqlType) -> SqlValue {
    match (left, right) {
        // Integer/integer comparisons retain each side's source type, including a literal that is
        // outside the column's assignment range (`smallint < 32768`).
        (left, right) if is_integer_type(left) && is_integer_type(right) => value,
        // The known literal is already NUMERIC; promote only the column comparison base, not the
        // literal's scale or mantissa.
        (left, SqlType::Numeric { .. }) if is_integer_type(left) => value,
        // A numeric column compared to a known integer is evaluated in the numeric family. Its
        // literal becomes an exact scale-0 decimal; NULL stays typeless.
        (SqlType::Numeric { .. }, right) if is_integer_type(right) => {
            integer_literal_as_numeric(value)
        }
        // Numeric/numeric comparisons retain the literal's natural scale.
        (SqlType::Numeric { .. }, SqlType::Numeric { .. }) => value,
        // The compatibility gate already ruled out all cross-family cases.
        _ => value,
    }
}

fn is_integer_type(ty: SqlType) -> bool {
    matches!(ty, SqlType::Int2 | SqlType::Int4 | SqlType::Int8)
}

fn integer_literal_as_numeric(value: SqlValue) -> SqlValue {
    match value {
        SqlValue::Int2(value) => SqlValue::Numeric(Decimal128::new(i128::from(value), 0)),
        SqlValue::Int4(value) => SqlValue::Numeric(Decimal128::new(i128::from(value), 0)),
        SqlValue::Int8(value) => SqlValue::Numeric(Decimal128::new(i128::from(value), 0)),
        SqlValue::Null => SqlValue::Null,
        _ => unreachable!("known integer literal type was validated before numeric promotion"),
    }
}

fn natural_check_literal_type(value: &SqlValue) -> Option<SqlType> {
    match value {
        SqlValue::Int2(_) => Some(SqlType::Int2),
        SqlValue::Int4(_) => Some(SqlType::Int4),
        SqlValue::Int8(_) => Some(SqlType::Int8),
        SqlValue::Numeric(value) => Some(SqlType::Numeric {
            precision: NUMERIC_DEFAULT_PRECISION,
            scale: value.scale,
        }),
        SqlValue::Bool(_) => Some(SqlType::Bool),
        SqlValue::Date(_) => Some(SqlType::Date),
        SqlValue::Timestamp(_) => Some(SqlType::Timestamp),
        SqlValue::Uuid(_) => Some(SqlType::Uuid),
        SqlValue::Null | SqlValue::Text(_) | SqlValue::Parameter { .. } => None,
    }
}

fn literal_matches_type(value: &SqlValue, ty: SqlType) -> bool {
    matches!(
        (value, ty),
        (SqlValue::Int2(_), SqlType::Int2)
            | (SqlValue::Int4(_), SqlType::Int4)
            | (SqlValue::Int8(_), SqlType::Int8)
            | (SqlValue::Numeric(_), SqlType::Numeric { .. })
            | (SqlValue::Bool(_), SqlType::Bool)
            | (SqlValue::Text(_), SqlType::Text)
            | (SqlValue::Date(_), SqlType::Date)
            | (SqlValue::Timestamp(_), SqlType::Timestamp)
            | (SqlValue::Uuid(_), SqlType::Uuid)
    )
}

fn check_comparison_types_compatible(left: SqlType, right: SqlType) -> bool {
    matches!(
        (left, right),
        (
            SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. },
            SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. }
        ) | (SqlType::Bool, SqlType::Bool)
            | (SqlType::Text, SqlType::Text)
            | (SqlType::Date, SqlType::Date)
            | (SqlType::Timestamp, SqlType::Timestamp)
            | (SqlType::Date, SqlType::Timestamp)
            | (SqlType::Timestamp, SqlType::Date)
            | (SqlType::Uuid, SqlType::Uuid)
    )
}

fn check_operator_name(op: SelectFilterOp) -> &'static str {
    match op {
        SelectFilterOp::Eq => "=",
        SelectFilterOp::Lt => "<",
        SelectFilterOp::Lte => "<=",
        SelectFilterOp::Gt => ">",
        SelectFilterOp::Gte => ">=",
        SelectFilterOp::LikePrefix => "LIKE",
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
    bind_to_column(default, ty, column_name)
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
        // TYPE-COVERAGE track 2 (audit cede8e70 SHOULD-FIX): the natural PG forms
        // `WHERE d = '2027-01-01'` and `WHERE s = 5` bind Text/Int4 literals against
        // Date/Int2 columns — without these arms the DML binder rejected them outright
        // ("invalid value for column"), leaving the device Date/Int2 needle path
        // unreachable via plain syntax. Mirrors `coerce_insert_value`: a parseable date
        // string coerces; an unparseable one stays Text (the binder's type check then
        // errors, as PG does on a bad date literal); an out-of-i16-range integer stays
        // Int4 (the binder errors — stricter than PG's promote-and-compare, consistent
        // with this engine's checked-arithmetic posture).
        (SqlValue::Text(s), SqlType::Date) => match gpu_db_sql::datetime::parse_date(&s) {
            Some(days) => SqlValue::Date(days),
            None => SqlValue::Text(s),
        },
        (SqlValue::Int4(v), SqlType::Int2) => match i16::try_from(v) {
            Ok(narrowed) => SqlValue::Int2(narrowed),
            Err(_) => SqlValue::Int4(v),
        },
        (other, _) => other,
    }
}

pub(crate) fn add_column_default_supported(default: &ColumnDefault) -> bool {
    match default {
        ColumnDefault::Literal(_) | ColumnDefault::DeferredScalar { .. } => true,
        ColumnDefault::SequenceNextVal {
            create_if_missing, ..
        } => !create_if_missing,
    }
}

pub(crate) fn relational_row_key(table: &str, row_id: u64) -> String {
    format!("rel/{table}/{row_id:020}")
}

pub(crate) fn relational_key_prefix(table: &str) -> String {
    format!("rel/{table}/")
}

/// The reserved, prefix-free storage/index token for SQL `NULL`. No typed value can
/// produce it (every typed encoding is type-prefixed, e.g. `t:`/`i:`), so it is an
/// unambiguous sentinel shared by the value-index key, the stored-row encoding, and the
/// decode path — kept in one place so those three can't drift apart.
pub(crate) const NULL_TOKEN: &str = "null";

pub(crate) fn relational_index_value(value: &SqlValue) -> String {
    match value {
        // NULL keys to the reserved prefix-free token. `WHERE col = NULL` is never TRUE in
        // SQL, so this key is not consulted by equality lookups; it exists for storage
        // symmetry and a future `IS NULL` index probe.
        SqlValue::Null => NULL_TOKEN.to_string(),
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
        SqlValue::Parameter { .. } => {
            unreachable!("value indexes never contain unbound prepared parameters")
        }
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

pub(crate) fn render_sql_value_literal(value: &SqlValue) -> Result<String, EngineError> {
    // Catalog/default display and psql-introspection helper. COPY retains typed programmatic cells
    // and no longer reconstructs INSERT text through this formatter.
    match value {
        SqlValue::Int2(value) => Ok(value.to_string()),
        SqlValue::Int4(value) => Ok(value.to_string()),
        SqlValue::Int8(value) => Ok(value.to_string()),
        SqlValue::Numeric(value) => Ok(value.to_decimal_string()),
        SqlValue::Bool(value) => Ok(if *value { "true" } else { "false" }.to_string()),
        SqlValue::Text(value) => Ok(format!("'{}'", value.replace('\'', "''"))),
        SqlValue::Date(value) => Ok(format!("'{}'", gpu_db_sql::datetime::format_date(*value))),
        SqlValue::Timestamp(value) => Ok(format!(
            "'{}'",
            gpu_db_sql::datetime::format_timestamp(*value)
        )),
        SqlValue::Uuid(value) => Ok(format!("'{}'", gpu_db_sql::uuid::format_uuid(value))),
        // M3 (doc 21): a NULL cell (e.g. a COPY `\N` field) renders as the SQL NULL keyword; the
        // re-parsed INSERT recognizes the unquoted `NULL` literal and stores a SqlValue::Null.
        SqlValue::Null => Ok("NULL".to_string()),
        SqlValue::Parameter { .. } => Err(EngineError::ApplyFailed(
            "COPY rendering received an unbound prepared parameter".to_string(),
        )),
    }
}

pub(crate) fn relational_resident_value_bytes(value: &SqlValue) -> u64 {
    match value {
        // NULL occupies a null-bitmap bit (slice 2), not typed payload bytes; this is only the
        // `resident_bytes` accounting metric, so a NULL cell contributes 0 typed bytes.
        SqlValue::Null => 0,
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
        SqlValue::Parameter { .. } => {
            unreachable!("resident rows never contain unbound prepared parameters")
        }
    }
}

pub(crate) fn compare_sql_values(left: &SqlValue, right: &SqlValue) -> Ordering {
    match (left, right) {
        (SqlValue::Parameter { .. }, _) | (_, SqlValue::Parameter { .. }) => {
            unreachable!("execution never receives an unbound prepared parameter")
        }
        // NULL sorts lowest in this INTERNAL total order (value-index/dedup only). SQL 3VL —
        // where a comparison to NULL is UNKNOWN, never an ordering — is enforced one level up in
        // `select_filter_matches` (which excludes any row whose operand is NULL); this arm only
        // keeps the comparator total so internal ordered structures never panic on a NULL cell.
        (SqlValue::Null, SqlValue::Null) => Ordering::Equal,
        (SqlValue::Null, _) => Ordering::Less,
        (_, SqlValue::Null) => Ordering::Greater,
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
    // SQL three-valued logic: any comparison with a NULL operand is UNKNOWN, and a WHERE
    // predicate that is UNKNOWN excludes the row (it is not TRUE). `IS NULL`/`IS NOT NULL` are
    // separate operators (not modeled here), so for every comparison op a NULL operand → false.
    if matches!(left, SqlValue::Null) || matches!(right, SqlValue::Null) {
        return false;
    }
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

/// Validate the value columns of grouped aggregates (SUM/AVG/MIN/MAX need an existing/typed column;
/// COUNT(*) needs none). Shared by the column-key and expression-key grouped-projection paths.
fn validate_grouped_aggregate_value_columns(
    table: &RelationalTable,
    aggregates: &[GroupedAggregate],
) -> Result<(), ExecuteError> {
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
                // The device value-type classification rejects unsupported MIN/MAX types; here just
                // confirm the value column exists.
                let column = aggregate.value_column.as_ref().ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "grouped MIN/MAX requires a value column".to_string(),
                    ))
                })?;
                relational_column_index(table, column)?;
            }
            GroupedAggKind::CountDistinct => {
                // COUNT(DISTINCT v) GPU-sorts the (group, value) tuple. A fixed-width value packs
                // into i64 keys: int2/int4/int8/date/timestamp -> one i64 (g, v); numeric/uuid ->
                // the raw 16-byte i128 split into two i64 limbs (g, v_hi, v_lo). A varlen TEXT value
                // routes through the hetero sort + the text-aware mark (byte-equality).
                let column = aggregate.value_column.as_ref().ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "grouped COUNT(DISTINCT) requires a value column".to_string(),
                    ))
                })?;
                let idx = relational_column_index(table, column)?;
                if !matches!(
                    table.columns[idx].ty,
                    SqlType::Int2
                        | SqlType::Int4
                        | SqlType::Int8
                        | SqlType::Date
                        | SqlType::Timestamp
                        | SqlType::Numeric { .. }
                        | SqlType::Uuid
                        | SqlType::Text
                ) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "grouped COUNT(DISTINCT) supports int2/int4/int8/date/timestamp/numeric/uuid/\
                         text value columns on the GPU path"
                            .to_string(),
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn bind_relational_select(
    table: &RelationalTable,
    select: &Select,
) -> Result<BoundRelationalSelect, ExecuteError> {
    let selected_indexes = match &select.projection {
        SelectProjection::All => (0..table.columns.len()).collect::<Vec<_>>(),
        SelectProjection::Columns(columns) => {
            let mut selected = Vec::new();
            for name in columns {
                if name == PROJECTION_WILDCARD_SENTINEL {
                    selected.extend(0..table.columns.len());
                } else {
                    selected.push(relational_column_index(table, name)?);
                }
            }
            selected
        }
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
        // A bare/scalar COUNT(DISTINCT v) selects no group column; the single int8 "count" aggregate
        // column is appended below, and it runs on the GPU as a one-group distinct count.
        SelectProjection::CountDistinct { .. } => Vec::new(),
        SelectProjection::GroupedMin { group_column, .. }
        | SelectProjection::GroupedMax { group_column, .. } => {
            vec![relational_column_index(table, group_column)?]
        }
        SelectProjection::GroupedAggregates { group_column, .. } => {
            // An expression GROUP BY uses the placeholder name "(expr)": the group key is a DERIVED
            // value, not a column, so bind column 0 as a placeholder (the executor overrides the result
            // schema's group-key column with the expression's int4/int8 result type).
            if group_column == "(expr)" {
                vec![0]
            } else {
                vec![relational_column_index(table, group_column)?]
            }
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
            | SelectProjection::CountDistinct { .. }
    ) {
        let aggregate_name = match &select.projection {
            // Scalar COUNT(DISTINCT v) projects a single int8 "count" column (PG names it "count").
            SelectProjection::CountAll
            | SelectProjection::GroupedCount { .. }
            | SelectProjection::CountDistinct { .. } => "count",
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
            // COUNT / COUNT(DISTINCT) are int8 (OID 20) regardless of the counted column.
            SelectProjection::CountAll
            | SelectProjection::GroupedCount { .. }
            | SelectProjection::CountDistinct { .. } => (SqlType::Int8, 20, 8),
            // SUM: PG SUM(int8) -> numeric (the bigint sum can exceed int8); SUM(int2/int4) ->
            // bigint. The declared SqlType must match the materialized SqlValue as well as the OID:
            // materialized views and transient GPU relations consume this type to lay out payload bytes. NB:
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
                    _ => (SqlType::Int8, 20, 8),
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
    // SUM(int8/numeric)->numeric, SUM(int2/int4)->int8, MIN/MAX-> the source column's type.
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
                        _ => ("sum", SqlType::Int8, 20, 8),
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
                // COUNT(DISTINCT v) is an int8 count (PG names it "count"), like COUNT.
                GroupedAggKind::CountDistinct => ("count", SqlType::Int8, 20, 8),
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
        .first()
        // An ORDER BY EXPRESSION (`a+b`) carries an empty placeholder column and has no pushed CPU
        // order -- the general GPU executor evaluates + sorts it. Skip the column lookup here (else it
        // resolves "" -> "column \"\" does not exist").
        .filter(|order| !order.column.is_empty())
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
                if let Some(order) = select.order_by.first() {
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
            | SelectProjection::CountDistinct { .. }
            | SelectProjection::GroupedAggregates { .. } => unreachable!(),
        }
    }
    let group_by_index = if let Some(group_by) = &select.group_by {
        // An expression GROUP BY ("(expr)" placeholder) has no source column index -- the key is a
        // derived value the executor materializes; skip the column-key projection validation below.
        if group_by == "(expr)" {
            None
        } else {
            Some(relational_column_index(table, group_by)?)
        }
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
        // Scalar COUNT(DISTINCT v) (no GROUP BY): runs on the GPU as a single-group distinct count
        // (sort/mark/SUM over (g=0, v)). Validate the value column exists + is a supported type. A
        // GROUP BY folds COUNT(DISTINCT) into GroupedAggregates at parse time, so the bare
        // CountDistinct projection should never carry a group_by_index.
        (SelectProjection::CountDistinct { column }, None) => {
            let idx = relational_column_index(table, column)?;
            if !matches!(
                table.columns[idx].ty,
                SqlType::Int2
                    | SqlType::Int4
                    | SqlType::Int8
                    | SqlType::Date
                    | SqlType::Timestamp
                    | SqlType::Numeric { .. }
                    | SqlType::Uuid
                    | SqlType::Text
            ) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "scalar COUNT(DISTINCT) supports int2/int4/int8/date/timestamp/numeric/uuid/\
                     text value columns on the GPU path"
                        .to_string(),
                )));
            }
        }
        (SelectProjection::CountDistinct { .. }, Some(_)) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "COUNT(DISTINCT) with GROUP BY uses the grouped aggregates projection".to_string(),
            )));
        }
        (SelectProjection::GroupedCount { column }, Some(idx)) => {
            let projected_idx = relational_column_index(table, column)?;
            if projected_idx != idx {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "GROUP BY column must match grouped COUNT(*) projection".to_string(),
                )));
            }
            if let Some(order) = select.order_by.first() {
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
            if let Some(order) = select.order_by.first() {
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
            if let Some(order) = select.order_by.first() {
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
            if let Some(order) = select.order_by.first() {
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
            if let Some(order) = select.order_by.first() {
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
            if let Some(order) = select.order_by.first() {
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
            validate_grouped_aggregate_value_columns(table, aggregates)?;
        }
        (SelectProjection::GroupedAggregates { aggregates, .. }, None) => {
            // An EXPRESSION GROUP BY ("(expr)" placeholder) has a derived group key with no source
            // column index, so it legitimately reaches here with None (group_by_index was set None
            // above for it). Validate the aggregate value columns + pass; the result group key is the
            // derived value (result column-0). A genuinely absent GROUP BY is still the error.
            if select.group_by.as_deref() == Some("(expr)") {
                validate_grouped_aggregate_value_columns(table, aggregates)?;
            } else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "grouped aggregates require GROUP BY".to_string(),
                )));
            }
        }
        (SelectProjection::All | SelectProjection::Columns(_), Some(_)) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "GROUP BY requires COUNT(*) projection".to_string(),
            )));
        }
        (SelectProjection::All | SelectProjection::Columns(_), None) => {}
    }

    // Bind HAVING before selecting a physical route. Otherwise a semantically-invalid typed SELECT
    // can be declined by an enumerated route, attempted by the broader Expr bridge, and finally be
    // reported as a residency miss instead of the SQL error that belongs to the statement. The
    // predicate itself still runs on the GPU; this only resolves its result-column names against
    // the already-bound SELECT schema.
    if !select.having_groups.is_empty() {
        let group_column = select.group_by.as_deref().ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "HAVING requires GROUP BY".to_string(),
            ))
        })?;
        for filter in select.having_groups.iter().flatten() {
            if !filter.column.eq_ignore_ascii_case(group_column)
                && !select_is_aggregate_result_column(select, &filter.column)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "HAVING must reference grouped column or aggregate result".to_string(),
                )));
            }
        }
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
#[derive(Debug, Clone)]
pub(crate) struct BoundUpdateAssignment {
    pub(crate) column_idx: usize,
    pub(crate) value: BoundUpdateValue,
}

#[derive(Debug, Clone)]
pub(crate) enum BoundUpdateValue {
    Literal(SqlValue),
    AddSameColumn(SqlValue),
}

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
                    let ty = table.columns[idx].ty;
                    // Coerce across the integer/numeric tower for parity with SELECT.
                    let value = coerce_filter_literal(filter.value.clone(), ty);
                    if sql_value_matches_type(&value, ty) {
                        return Ok((idx, filter.op, value));
                    }
                    // COMPOUND KEYS (wider types): a TEXT literal against a Uuid/Date/Timestamp column
                    // (or a numeric needing rescale) needs the INSERT-side coercion (Text -> Uuid via
                    // parse_uuid, etc.) so the WHERE value MATCHES the stored typed value (else the
                    // predicate recheck compares Text vs Uuid -> 0 rows). A value with no cast still
                    // errors loudly.
                    let value = coerce_insert_value(filter.value, ty, &filter.column)
                        .map_err(ExecuteError::Engine)?;
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
            .map_err(|error| {
                if error
                    .to_string()
                    .starts_with("apply failed: invalid value for column")
                {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "column type mismatch".to_string(),
                    ))
                } else {
                    ExecuteError::Engine(error)
                }
            })?;
            let value = match &assignment.source_column {
                None => BoundUpdateValue::Literal(value),
                Some(source) if source == &assignment.column => {
                    if !matches!(table.columns[idx].ty, SqlType::Int4 | SqlType::Int8) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "checked same-column addition is supported only for int4/int8 column \"{}\"",
                            assignment.column
                        ))));
                    }
                    BoundUpdateValue::AddSameColumn(value)
                }
                Some(_) => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "UPDATE expression for \"{}\" must read the same column",
                        assignment.column
                    ))));
                }
            };
            Ok(BoundUpdateAssignment {
                column_idx: idx,
                value,
            })
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn relational_select_pushes_limit(select: &Select) -> bool {
    select.limit.is_some() && select.offset.is_none() && select.order_by.is_empty()
}

pub(crate) fn relational_select_pushed_limit(
    select: &Select,
    ordered_access_path: bool,
) -> Option<usize> {
    if select.distinct || select_is_aggregate(select) {
        return None;
    }
    let can_push = select.order_by.is_empty() || ordered_access_path;
    if !can_push {
        return None;
    }
    select
        .limit
        .map(|limit| limit.saturating_add(select.offset.unwrap_or(0)))
}

#[cfg(test)]
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

pub(crate) fn select_has_relational_filters(select: &Select) -> bool {
    select.filter.is_some() || !select.filters.is_empty() || !select.filter_groups.is_empty()
}

pub(crate) fn select_is_plain_view_scan(select: &Select) -> bool {
    !select.distinct
        && matches!(select.projection, SelectProjection::All)
        && select.group_by.is_none()
        && select.having_groups.is_empty()
        && !select_has_relational_filters(select)
        && select.order_by.is_empty()
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
            | SelectProjection::CountDistinct { .. }
            | SelectProjection::GroupedAggregates { .. }
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
        // The Expr path's grouped projection: ORDER BY may reference any aggregate by its result
        // column name (count/sum/avg/min/max); the executor resolves it against the result columns,
        // so the binding only needs to recognize the name and skip the table-column lookup.
        SelectProjection::GroupedAggregates { ref aggregates, .. } => {
            aggregates.iter().any(|agg| {
                let name = match agg.kind {
                    GroupedAggKind::Count | GroupedAggKind::CountDistinct => "count",
                    GroupedAggKind::Sum => "sum",
                    GroupedAggKind::Avg => "avg",
                    GroupedAggKind::Min => "min",
                    GroupedAggKind::Max => "max",
                };
                column.eq_ignore_ascii_case(name)
            })
        }
        // A bare/scalar COUNT(DISTINCT v) projects a "count" result column (rejected at binding, but
        // recognize the name for completeness).
        SelectProjection::CountDistinct { .. } => column.eq_ignore_ascii_case("count"),
        SelectProjection::All | SelectProjection::Columns(_) => false,
    }
}

#[cfg(test)]
pub(crate) fn select_aggregate_result_column_name(select: &Select) -> Option<&'static str> {
    match select.projection {
        SelectProjection::CountAll | SelectProjection::GroupedCount { .. } => Some("count"),
        SelectProjection::Sum { .. } | SelectProjection::GroupedSum { .. } => Some("sum"),
        SelectProjection::Avg { .. } | SelectProjection::GroupedAvg { .. } => Some("avg"),
        SelectProjection::Min { .. } | SelectProjection::GroupedMin { .. } => Some("min"),
        SelectProjection::Max { .. } | SelectProjection::GroupedMax { .. } => Some("max"),
        // A bare/scalar COUNT(DISTINCT v) projects a single "count" column (rejected at binding).
        SelectProjection::CountDistinct { .. } => Some("count"),
        // GroupedAggregates has N aggregates (no single result-column name); Expr-path only.
        SelectProjection::All
        | SelectProjection::Columns(_)
        | SelectProjection::GroupedAggregates { .. } => None,
    }
}

#[cfg(test)]
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
        | SelectProjection::CountDistinct { .. }
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
    if let Some(order) = select.order_by.first() {
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

#[cfg(test)]
pub(crate) fn int4_aggregate_value(
    value: &SqlValue,
    aggregate: &'static str,
) -> Result<i32, ExecuteError> {
    match value {
        SqlValue::Int4(value) => Ok(*value),
        SqlValue::Null
        | SqlValue::Int2(_)
        | SqlValue::Int8(_)
        | SqlValue::Numeric(_)
        | SqlValue::Bool(_)
        | SqlValue::Text(_)
        | SqlValue::Date(_)
        | SqlValue::Timestamp(_)
        | SqlValue::Uuid(_)
        | SqlValue::Parameter { .. } => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            aggregate_int4_error_message(aggregate).to_string(),
        ))),
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
pub(crate) fn avg_numeric_sql_value(
    sum_mantissa: i128,
    count: usize,
    column_scale: u8,
) -> SqlValue {
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
