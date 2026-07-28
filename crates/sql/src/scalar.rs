//! SQL scalar type/value contracts and literal parsing.

use super::{find_matching_paren, Decimal128, ParseError};

/// A storable column type. `Numeric` carries the PostgreSQL `numeric(p,s)` typmod
/// (`precision`/`scale`); every variant's payload is `Copy`, so `SqlType` stays
/// `Copy` exactly like the original `Int4`/`Text`-only enum (the ~60 `== SqlType::Int4`
/// guards and by-value passes are unaffected by the widening).
#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord,
)]
pub enum SqlType {
    /// PostgreSQL `smallint` (int2) — stored widened to i32 in the int4 device section (so it reuses
    /// the int4 compare path); arithmetic (int16-bounds overflow) is a follow-on.
    Int2,
    Int4,
    Int8,
    /// PostgreSQL `numeric(precision, scale)` — stored as a fixed-point [`Decimal128`].
    Numeric {
        precision: u8,
        scale: u8,
    },
    Bool,
    Text,
    /// PostgreSQL `date` — stored as i32 DAYS since 2000-01-01 (so it reuses the int4 device path).
    Date,
    /// PostgreSQL `timestamp` (without time zone) — stored as i64 MICROSECONDS since 2000-01-01
    /// 00:00:00 (so it reuses the int8 device path).
    Timestamp,
    /// PostgreSQL `uuid` — 16 raw bytes, compared byte-wise (unsigned). Reuses the i128 (16-byte)
    /// residency section, but with a byte-wise compare kernel rather than the signed i128 one.
    Uuid,
}

/// The default `numeric` typmod when a `NUMERIC`/`DECIMAL` column omits `(p,s)`.
/// PostgreSQL treats unconstrained `numeric` specially; we pin a wide fixed typmod
/// (38 significant digits — the i128 mantissa ceiling) so values round-trip without
/// a bignum fallback (>38 digits is a documented, errored edge — a future milestone).
pub const NUMERIC_DEFAULT_PRECISION: u8 = 38;
pub const NUMERIC_DEFAULT_SCALE: u8 = 0;

pub const SUPPORTED_SQL_TYPES: [SqlType; 9] = [
    SqlType::Int2,
    SqlType::Int4,
    SqlType::Int8,
    SqlType::Numeric {
        precision: NUMERIC_DEFAULT_PRECISION,
        scale: NUMERIC_DEFAULT_SCALE,
    },
    SqlType::Bool,
    SqlType::Text,
    SqlType::Date,
    SqlType::Timestamp,
    SqlType::Uuid,
];

impl SqlType {
    pub const fn postgres_oid(self) -> u32 {
        match self {
            Self::Int2 => 21,
            Self::Int4 => 23,
            Self::Int8 => 20,
            Self::Numeric { .. } => 1700,
            Self::Bool => 16,
            Self::Text => 25,
            Self::Date => 1082,
            Self::Timestamp => 1114,
            Self::Uuid => 2950,
        }
    }

    pub const fn type_size(self) -> i16 {
        match self {
            Self::Int2 => 2,
            Self::Int4 => 4,
            Self::Int8 => 8,
            Self::Numeric { .. } => -1,
            Self::Bool => 1,
            Self::Text => -1,
            Self::Date => 4,
            Self::Timestamp => 8,
            Self::Uuid => 16,
        }
    }

    pub const fn catalog_name(self) -> &'static str {
        match self {
            Self::Int2 => "int2",
            Self::Int4 => "int4",
            Self::Int8 => "int8",
            Self::Numeric { .. } => "numeric",
            Self::Bool => "bool",
            Self::Text => "text",
            Self::Date => "date",
            Self::Timestamp => "timestamp",
            Self::Uuid => "uuid",
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SqlValue {
    /// SQL `NULL` — the typeless absence of a value. Placed first so the *derived*
    /// `Ord`/`Eq` (used only for internal value-index keys and dedup, never for SQL
    /// truth) gives a deterministic total order with `Null` sorting lowest. SQL
    /// semantics (where `NULL = NULL` is UNKNOWN and a comparison to NULL is never
    /// TRUE) route through `compare_sql_values`/`select_filter_matches`, never the
    /// derived traits. See `docs/architecture/21-null-representation-and-three-valued-logic.md`.
    Null,
    Int4(i32),
    Int8(i64),
    /// A fixed-point decimal (`numeric`). Equality and ordering are scale-aligned via
    /// [`Decimal128`], so `1.0` and `1.00` compare equal — which is why a value-index
    /// equality lookup on a NUMERIC column must key on a canonical scale.
    Numeric(Decimal128),
    Bool(bool),
    Text(String),
    /// A `date` as i32 DAYS since 2000-01-01 (PostgreSQL's date epoch). Ordering is the natural
    /// integer ordering of the day count.
    Date(i32),
    /// A `timestamp` as i64 MICROSECONDS since 2000-01-01 00:00:00. Ordering is the natural integer
    /// ordering of the microsecond count.
    Timestamp(i64),
    /// A `uuid` as its 16 raw bytes. Ordering is the byte-wise (unsigned) order of `[u8; 16]`, which
    /// is how PostgreSQL compares uuids.
    Uuid([u8; 16]),
    /// A `smallint` (int2) as i16. Widens to int4 for comparison (PG's numeric tower).
    Int2(i16),
    /// A typed-AST parameter slot owned only by [`crate::PreparedCommand`]. Ordinary
    /// [`crate::ParsedCommand`] construction rejects commands that still contain this variant,
    /// and binding replaces every occurrence before the command can cross engine admission.
    #[serde(skip)]
    Parameter {
        /// PostgreSQL's one-based `$n` index.
        index: usize,
        /// An explicit SQL cast attached to the placeholder, when present.
        cast: Option<SqlType>,
    },
}

/// Parse a PostgreSQL boolean *value* literal (for a `bool` column / `::bool` cast).
/// Accepts the canonical wire forms plus the spelled-out / single-letter aliases
/// PostgreSQL recognizes, case-insensitively. Distinct from [`parse_bool_literal`],
/// which is the stricter `true`/`t`/`false`/`f`-only parser for option arguments.
pub fn parse_bool_value(input: &str) -> Option<bool> {
    let trimmed = input.trim();
    if trimmed.eq_ignore_ascii_case("t")
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("yes")
        || trimmed.eq_ignore_ascii_case("on")
        || trimmed == "1"
    {
        Some(true)
    } else if trimmed.eq_ignore_ascii_case("f")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
        || trimmed.eq_ignore_ascii_case("off")
        || trimmed == "0"
    {
        Some(false)
    } else {
        None
    }
}

pub(super) fn parse_supported_sql_type_name(input: &str) -> Option<SqlType> {
    let ty = if input
        .get(.."pg_catalog.".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("pg_catalog."))
    {
        &input["pg_catalog.".len()..]
    } else {
        input
    };
    // Split off an optional `(...)` typmod (only NUMERIC/DECIMAL accept one).
    let (base, typmod) = match ty.find('(') {
        Some(open) => {
            let close = find_matching_paren(ty, open)?;
            if !ty[close + 1..].trim().is_empty() {
                return None;
            }
            (ty[..open].trim(), Some(ty[open + 1..close].trim()))
        }
        None => (ty.trim(), None),
    };
    if base.eq_ignore_ascii_case("INT")
        || base.eq_ignore_ascii_case("INT4")
        || base.eq_ignore_ascii_case("INTEGER")
    {
        typmod.is_none().then_some(SqlType::Int4)
    } else if base.eq_ignore_ascii_case("INT8") || base.eq_ignore_ascii_case("BIGINT") {
        typmod.is_none().then_some(SqlType::Int8)
    } else if base.eq_ignore_ascii_case("INT2") || base.eq_ignore_ascii_case("SMALLINT") {
        typmod.is_none().then_some(SqlType::Int2)
    } else if base.eq_ignore_ascii_case("NUMERIC") || base.eq_ignore_ascii_case("DECIMAL") {
        parse_numeric_typmod(typmod)
    } else if base.eq_ignore_ascii_case("BOOL") || base.eq_ignore_ascii_case("BOOLEAN") {
        typmod.is_none().then_some(SqlType::Bool)
    } else if base.eq_ignore_ascii_case("TEXT") {
        typmod.is_none().then_some(SqlType::Text)
    } else if base.eq_ignore_ascii_case("DATE") {
        typmod.is_none().then_some(SqlType::Date)
    } else if base.eq_ignore_ascii_case("TIMESTAMP") {
        // `timestamp` (without time zone); a fractional-second typmod is a follow-on.
        typmod.is_none().then_some(SqlType::Timestamp)
    } else if base.eq_ignore_ascii_case("UUID") {
        typmod.is_none().then_some(SqlType::Uuid)
    } else {
        None
    }
}

/// Parse a `numeric` typmod body (`"12,2"`, `"10"`, or absent) into a
/// `SqlType::Numeric { precision, scale }`. An absent typmod yields the
/// unconstrained default; precision must be 1..=38 (the i128 ceiling) and scale
/// 0..=precision, mirroring PostgreSQL's `numeric(p,s)` constraints.
fn parse_numeric_typmod(typmod: Option<&str>) -> Option<SqlType> {
    let Some(body) = typmod else {
        return Some(SqlType::Numeric {
            precision: NUMERIC_DEFAULT_PRECISION,
            scale: NUMERIC_DEFAULT_SCALE,
        });
    };
    let mut parts = body.split(',');
    let precision: u8 = parts.next()?.trim().parse().ok()?;
    let scale: u8 = match parts.next() {
        Some(scale) => scale.trim().parse().ok()?,
        None => 0,
    };
    if parts.next().is_some()
        || precision == 0
        || precision > NUMERIC_DEFAULT_PRECISION
        || scale > precision
    {
        return None;
    }
    Some(SqlType::Numeric { precision, scale })
}

/// The source of a scalar's SQL input type.  Typed DEFAULT parsing needs the distinction between
/// an uncast `unknown` string and an explicit `::text`; the latter is not assignment-coercible to
/// an arbitrary column just because its rendered characters happen to parse at that target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScalarInputTypeProvenance {
    Unknown,
    Inferred(SqlType),
    Explicit(SqlType),
}

impl ScalarInputTypeProvenance {
    pub(super) const fn concrete_type(self) -> Option<SqlType> {
        match self {
            Self::Unknown => None,
            Self::Inferred(ty) | Self::Explicit(ty) => Some(ty),
        }
    }
}

pub(super) fn parse_sql_value(input: &str) -> Result<SqlValue, ParseError> {
    parse_sql_value_with_input_provenance(input).map(|(value, _)| value)
}

/// Compatibility projection for CHECK parsing, which only needs to know whether the scalar is
/// concrete. Typed DEFAULT parsing calls the provenance-preserving entry point below.
pub(super) fn parse_sql_value_with_input_type(
    input: &str,
) -> Result<(SqlValue, Option<SqlType>), ParseError> {
    parse_sql_value_with_input_provenance(input)
        .map(|(value, provenance)| (value, provenance.concrete_type()))
}

/// Parse a scalar while preserving whether its concrete type was inferred or explicitly cast.
pub(super) fn parse_sql_value_with_input_provenance(
    input: &str,
) -> Result<(SqlValue, ScalarInputTypeProvenance), ParseError> {
    let (s, cast) = split_supported_sql_value_cast(input.trim())?;
    if let Some(digits) = s.strip_prefix('$') {
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(ParseError::InvalidParameterReference);
        }
        let index = digits
            .parse::<usize>()
            .map_err(|_| ParseError::InvalidParameterReference)?;
        if index == 0 {
            return Err(ParseError::InvalidParameterReference);
        }
        let provenance = cast.map_or(
            ScalarInputTypeProvenance::Unknown,
            ScalarInputTypeProvenance::Explicit,
        );
        return Ok((SqlValue::Parameter { index, cast }, provenance));
    }
    // NULL remains typeless even with an explicit cast; the cast supplies only its eventual SQL
    // type. This also keeps a bound `$n::type` NULL canonical source parseable for recovery.
    if s.eq_ignore_ascii_case("NULL") {
        let provenance = cast.map_or(
            ScalarInputTypeProvenance::Unknown,
            ScalarInputTypeProvenance::Explicit,
        );
        return Ok((SqlValue::Null, provenance));
    }
    if s.starts_with('\'') {
        if !s.ends_with('\'') || s.len() < 2 {
            return Err(ParseError::InvalidRelationalSql);
        }
        let inner = &s[1..s.len() - 1];
        let value = inner.replace("''", "'");
        return match cast {
            None => Ok((SqlValue::Text(value), ScalarInputTypeProvenance::Unknown)),
            Some(SqlType::Text) => Ok((
                SqlValue::Text(value),
                ScalarInputTypeProvenance::Explicit(SqlType::Text),
            )),
            Some(ty) => parse_typed_value_from_str(&value, ty)
                .map(|value| (value, ScalarInputTypeProvenance::Explicit(ty))),
        };
    }
    // Unquoted literal. A cast pins the target type; otherwise we infer it (a bare
    // integer stays Int4 as before — widening only kicks in for the new shapes:
    // `TRUE`/`FALSE` → Bool, a value with a decimal point → Numeric, and an integer
    // that overflows i32 → Int8).
    match cast {
        Some(ty) => parse_typed_value_from_str(s, ty)
            .map(|value| (value, ScalarInputTypeProvenance::Explicit(ty))),
        None => {
            let value = parse_inferred_unquoted_literal(s)?;
            let ty = match &value {
                SqlValue::Int2(_) => SqlType::Int2,
                SqlValue::Int4(_) => SqlType::Int4,
                SqlValue::Int8(_) => SqlType::Int8,
                SqlValue::Numeric(value) => SqlType::Numeric {
                    precision: NUMERIC_DEFAULT_PRECISION,
                    scale: value.scale,
                },
                SqlValue::Bool(_) => SqlType::Bool,
                // The early NULL branch and quoted branch own these; keep this parser fail-closed
                // if a new inferred variant is introduced without an input-type rule.
                SqlValue::Null
                | SqlValue::Text(_)
                | SqlValue::Date(_)
                | SqlValue::Timestamp(_)
                | SqlValue::Uuid(_)
                | SqlValue::Parameter { .. } => return Err(ParseError::InvalidRelationalSql),
            };
            Ok((value, ScalarInputTypeProvenance::Inferred(ty)))
        }
    }
}

/// DEFAULT-specific scalar parsing.  Unlike ordinary scalar expressions, a numeric
/// DEFAULT must retain its natural mantissa/scale until the engine evaluates its
/// source cast and then the assignment target.  In particular,
/// `999.5::numeric(3,0)` is stored as `9995@scale=1`, not as the already-rounded
/// `1000@scale=0` carrier.
pub(super) fn parse_sql_default_value_with_input_provenance(
    input: &str,
) -> Result<(SqlValue, ScalarInputTypeProvenance), ParseError> {
    let (source, cast) = split_supported_sql_value_cast(input.trim())?;
    let Some(numeric @ SqlType::Numeric { .. }) = cast else {
        return parse_sql_value_with_input_provenance(input);
    };
    if source.eq_ignore_ascii_case("NULL") || source.starts_with('$') {
        return parse_sql_value_with_input_provenance(input);
    }
    let text = if source.starts_with('\'') {
        if !source.ends_with('\'') || source.len() < 2 {
            return Err(ParseError::InvalidRelationalSql);
        }
        source[1..source.len() - 1].replace("''", "'")
    } else {
        source.to_string()
    };
    parse_default_numeric_literal(&text, numeric).map(|value| {
        (
            SqlValue::Numeric(value),
            ScalarInputTypeProvenance::Explicit(numeric),
        )
    })
}

/// Parse a NUMERIC input without applying its typmod.  The input is still syntactically
/// validated here; precision/range checks are intentionally the later evaluator's job.
pub fn parse_default_numeric_literal(text: &str, ty: SqlType) -> Result<Decimal128, ParseError> {
    let SqlType::Numeric { .. } = ty else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Decimal128::parse(text).ok_or_else(|| {
        if numeric_text_is_well_formed(text) {
            numeric_range_error(ty, text)
        } else {
            invalid_text_error(ty, text)
        }
    })
}

/// Parse a bounded scalar projection and retain the type that PostgreSQL exposes in RowDescription.
/// Unlike storage literals, a projected NULL must have a concrete output type; PostgreSQL resolves
/// an otherwise-unknown top-level NULL to text, while an explicit cast owns the type.
pub(super) fn parse_typed_sql_literal(input: &str) -> Result<(SqlType, SqlValue), ParseError> {
    let (value, cast) = parse_sql_value_with_input_type(input)?;
    let ty = match (cast, &value) {
        (Some(ty), _) => ty,
        (None, SqlValue::Null | SqlValue::Text(_)) => SqlType::Text,
        (None, SqlValue::Int2(_)) => SqlType::Int2,
        (None, SqlValue::Int4(_)) => SqlType::Int4,
        (None, SqlValue::Int8(_)) => SqlType::Int8,
        (None, SqlValue::Numeric(value)) => SqlType::Numeric {
            precision: NUMERIC_DEFAULT_PRECISION,
            scale: value.scale,
        },
        (None, SqlValue::Bool(_)) => SqlType::Bool,
        (None, SqlValue::Date(_)) => SqlType::Date,
        (None, SqlValue::Timestamp(_)) => SqlType::Timestamp,
        (None, SqlValue::Uuid(_)) => SqlType::Uuid,
        (None, SqlValue::Parameter { cast: Some(ty), .. }) => *ty,
        (None, SqlValue::Parameter { cast: None, .. }) => {
            return Err(ParseError::InvalidRelationalSql)
        }
    };
    Ok((ty, value))
}

/// Parse `text` into a specific [`SqlType`] (used for an explicit `::type` cast and
/// for a quoted literal carrying a cast). The numeric arm rounds to the column scale.
pub fn parse_typed_value_from_str(text: &str, ty: SqlType) -> Result<SqlValue, ParseError> {
    match ty {
        SqlType::Int2 => text
            .parse::<i16>()
            .map(SqlValue::Int2)
            .map_err(|error| integer_parse_error(SqlType::Int2, text, error.kind())),
        SqlType::Int4 => text
            .parse::<i32>()
            .map(SqlValue::Int4)
            .map_err(|error| integer_parse_error(SqlType::Int4, text, error.kind())),
        SqlType::Int8 => text
            .parse::<i64>()
            .map(SqlValue::Int8)
            .map_err(|error| integer_parse_error(SqlType::Int8, text, error.kind())),
        // Do not apply `numeric(p,s)` precision here. A CHECK's explicit scalar cast is
        // cataloged successfully in PostgreSQL and its overflow is raised when the CHECK is
        // evaluated (INSERT / ALTER ADD CHECK validation). The caller retains the target type in
        // the catalog-resolved input type, which is the deferred typmod authority.
        SqlType::Numeric { scale, .. } => {
            parse_numeric_at_scale(text, scale).map(SqlValue::Numeric)
        }
        SqlType::Bool => parse_bool_value(text)
            .map(SqlValue::Bool)
            .ok_or_else(|| invalid_text_error(ty, text)),
        SqlType::Text => Ok(SqlValue::Text(text.to_string())),
        SqlType::Date => crate::datetime::parse_date_detailed(text)
            .map(SqlValue::Date)
            .map_err(|error| datetime_parse_error(text, error)),
        SqlType::Timestamp => crate::datetime::parse_timestamp_detailed(text)
            .map(SqlValue::Timestamp)
            .map_err(|error| datetime_parse_error(text, error)),
        SqlType::Uuid => crate::uuid::parse_uuid(text)
            .map(SqlValue::Uuid)
            .ok_or_else(|| invalid_text_error(ty, text)),
    }
}

fn datetime_parse_error(text: &str, error: crate::datetime::DatetimeParseError) -> ParseError {
    match error {
        crate::datetime::DatetimeParseError::MalformedFormat => ParseError::InvalidDatetimeFormat {
            input: text.to_string(),
        },
        crate::datetime::DatetimeParseError::FieldOverflow => ParseError::DatetimeFieldOverflow {
            input: text.to_string(),
        },
    }
}

fn integer_parse_error(ty: SqlType, input: &str, kind: &std::num::IntErrorKind) -> ParseError {
    match kind {
        std::num::IntErrorKind::PosOverflow | std::num::IntErrorKind::NegOverflow => {
            numeric_range_error(ty, input)
        }
        _ => invalid_text_error(ty, input),
    }
}

fn parse_numeric_at_scale(text: &str, scale: u8) -> Result<Decimal128, ParseError> {
    let parsed = Decimal128::parse(text).ok_or_else(|| {
        if numeric_text_is_well_formed(text) {
            numeric_range_error(
                SqlType::Numeric {
                    precision: NUMERIC_DEFAULT_PRECISION,
                    scale,
                },
                text,
            )
        } else {
            invalid_text_error(
                SqlType::Numeric {
                    precision: NUMERIC_DEFAULT_PRECISION,
                    scale,
                },
                text,
            )
        }
    })?;
    parsed.rescale(scale).map_err(|_| {
        numeric_range_error(
            SqlType::Numeric {
                precision: NUMERIC_DEFAULT_PRECISION,
                scale,
            },
            text,
        )
    })
}

fn numeric_text_is_well_formed(input: &str) -> bool {
    let input = input.trim();
    let input = input
        .strip_prefix('-')
        .or_else(|| input.strip_prefix('+'))
        .unwrap_or(input);
    if input.is_empty() {
        return false;
    }
    let (whole, fraction) = input.split_once('.').unwrap_or((input, ""));
    !(whole.is_empty() && fraction.is_empty())
        && whole.bytes().all(|byte| byte.is_ascii_digit())
        && fraction.bytes().all(|byte| byte.is_ascii_digit())
}

fn invalid_text_error(ty: SqlType, input: &str) -> ParseError {
    ParseError::InvalidTextRepresentation {
        ty: ty.catalog_name(),
        input: input.to_string(),
    }
}

fn numeric_range_error(ty: SqlType, input: &str) -> ParseError {
    ParseError::NumericValueOutOfRange {
        ty: ty.catalog_name(),
        input: input.to_string(),
    }
}

/// Infer a [`SqlValue`] from an unquoted, uncast literal. Preserves the pre-existing
/// rule that a bare integer is `Int4`; widens only to the genuinely new shapes.
fn parse_inferred_unquoted_literal(s: &str) -> Result<SqlValue, ParseError> {
    // The unquoted `NULL` keyword is the typeless SQL null (a quoted `'NULL'` is the text value,
    // handled on the quoted path). Valid for any column type; coercion passes it through unchanged.
    if s.eq_ignore_ascii_case("NULL") {
        return Ok(SqlValue::Null);
    }
    if s.eq_ignore_ascii_case("TRUE") {
        return Ok(SqlValue::Bool(true));
    }
    if s.eq_ignore_ascii_case("FALSE") {
        return Ok(SqlValue::Bool(false));
    }
    if let Ok(value) = s.parse::<i32>() {
        return Ok(SqlValue::Int4(value));
    }
    // A decimal point means NUMERIC; carry the literal's natural scale (its fractional
    // digit count) so `1.00` keeps scale 2 until the engine rescales to the column.
    if s.contains('.') {
        if let Some(decimal) = Decimal128::parse(s) {
            return Ok(SqlValue::Numeric(decimal));
        }
        return Err(ParseError::InvalidRelationalSql);
    }
    // An integer too wide for i32 widens to Int8 (rather than the old hard error).
    s.parse::<i64>()
        .map(SqlValue::Int8)
        .map_err(|_| ParseError::InvalidRelationalSql)
}

fn split_supported_sql_value_cast(input: &str) -> Result<(&str, Option<SqlType>), ParseError> {
    let Some(pos) = find_cast_operator_outside_quotes(input) else {
        return Ok((input, None));
    };
    let value = input[..pos].trim();
    let ty = input[pos + 2..].trim();
    if value.is_empty() || ty.is_empty() || find_cast_operator_outside_quotes(ty).is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let ty = if ty
        .get(.."pg_catalog.".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("pg_catalog."))
    {
        &ty["pg_catalog.".len()..]
    } else {
        ty
    };
    let cast = parse_supported_sql_type_name(ty).ok_or(ParseError::InvalidRelationalSql)?;
    Ok((value, Some(cast)))
}

fn find_cast_operator_outside_quotes(input: &str) -> Option<usize> {
    let bytes = input.as_bytes();
    let mut in_quote = false;
    let mut idx = 0;
    while idx + 1 < bytes.len() {
        match bytes[idx] {
            b'\'' => {
                if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                    idx += 1;
                } else {
                    in_quote = !in_quote;
                }
            }
            b':' if !in_quote && bytes[idx + 1] == b':' => return Some(idx),
            _ => {}
        }
        idx += 1;
    }
    None
}

#[cfg(test)]
mod decimal_tests {
    use super::*;

    #[test]
    fn parses_supported_type_names_including_typmod() {
        assert_eq!(parse_supported_sql_type_name("INT"), Some(SqlType::Int4));
        assert_eq!(
            parse_supported_sql_type_name("integer"),
            Some(SqlType::Int4)
        );
        assert_eq!(parse_supported_sql_type_name("BIGINT"), Some(SqlType::Int8));
        assert_eq!(parse_supported_sql_type_name("int8"), Some(SqlType::Int8));
        assert_eq!(parse_supported_sql_type_name("text"), Some(SqlType::Text));
        assert_eq!(parse_supported_sql_type_name("BOOL"), Some(SqlType::Bool));
        assert_eq!(
            parse_supported_sql_type_name("boolean"),
            Some(SqlType::Bool)
        );
        assert_eq!(
            parse_supported_sql_type_name("NUMERIC(12,2)"),
            Some(SqlType::Numeric {
                precision: 12,
                scale: 2
            })
        );
        assert_eq!(
            parse_supported_sql_type_name("numeric(12, 2)"),
            Some(SqlType::Numeric {
                precision: 12,
                scale: 2
            })
        );
        assert_eq!(
            parse_supported_sql_type_name("DECIMAL(10)"),
            Some(SqlType::Numeric {
                precision: 10,
                scale: 0
            })
        );
        assert_eq!(
            parse_supported_sql_type_name("NUMERIC"),
            Some(SqlType::Numeric {
                precision: NUMERIC_DEFAULT_PRECISION,
                scale: NUMERIC_DEFAULT_SCALE
            })
        );
        // A typmod on a non-numeric type, or an out-of-range numeric typmod, is rejected.
        assert_eq!(parse_supported_sql_type_name("INT(4)"), None);
        assert_eq!(parse_supported_sql_type_name("NUMERIC(0,0)"), None);
        assert_eq!(parse_supported_sql_type_name("NUMERIC(2,5)"), None);
        assert_eq!(parse_supported_sql_type_name("NUMERIC(99)"), None);
    }

    #[test]
    fn parses_typed_literals_and_casts() {
        // Bare literal inference: integer stays Int4, decimal becomes Numeric, TRUE/FALSE bool.
        assert_eq!(parse_sql_value("5").unwrap(), SqlValue::Int4(5));
        assert_eq!(
            parse_sql_value("1.50").unwrap(),
            SqlValue::Numeric(Decimal128::new(150, 2))
        );
        assert_eq!(parse_sql_value("TRUE").unwrap(), SqlValue::Bool(true));
        assert_eq!(parse_sql_value("false").unwrap(), SqlValue::Bool(false));
        // An integer beyond i32 widens to Int8.
        assert_eq!(
            parse_sql_value("9000000000").unwrap(),
            SqlValue::Int8(9_000_000_000)
        );
        // Casts pin the type (numeric cast rounds to the typmod scale).
        assert_eq!(
            parse_sql_value("'1.005'::numeric(12,2)").unwrap(),
            SqlValue::Numeric(Decimal128::new(101, 2))
        );
        assert_eq!(parse_sql_value("'t'::bool").unwrap(), SqlValue::Bool(true));
        assert_eq!(parse_sql_value("'42'::int8").unwrap(), SqlValue::Int8(42));
    }

    #[test]
    fn explicit_scalar_casts_keep_typed_error_categories_and_defer_numeric_precision() {
        for input in [
            "'not-a-bool'::bool",
            "'not-a-uuid'::uuid",
            "'x'::int2",
            "'x'::int4",
            "'x'::numeric",
        ] {
            assert!(matches!(
                parse_sql_value(input),
                Err(ParseError::InvalidTextRepresentation { .. })
            ));
        }
        for input in ["'not-a-date'::date", "'not-a-timestamp'::timestamp"] {
            assert!(matches!(
                parse_sql_value(input),
                Err(ParseError::InvalidDatetimeFormat { .. })
            ));
        }
        for input in [
            "'2024-02-30'::date",
            "'0000-01-01'::date",
            "'2024-01-01 25:00:00'::timestamp",
            "'294277-12-31 00:00:00'::timestamp",
        ] {
            assert!(matches!(
                parse_sql_value(input),
                Err(ParseError::DatetimeFieldOverflow { .. })
            ));
        }
        for input in [
            "'32768'::int2",
            "'2147483648'::int4",
            "'999999999999999999999999999999999999999'::numeric(38,0)",
        ] {
            assert!(matches!(
                parse_sql_value(input),
                Err(ParseError::NumericValueOutOfRange { .. })
            ));
        }

        // PostgreSQL accepts this scalar cast in CREATE CHECK and defers its typmod failure
        // until the published expression is evaluated. Keep both the rounded value and cast type.
        assert_eq!(
            parse_sql_value_with_input_type("999.5::numeric(3,0)").unwrap(),
            (
                SqlValue::Numeric(Decimal128::new(1_000, 0)),
                Some(SqlType::Numeric {
                    precision: 3,
                    scale: 0,
                })
            )
        );
    }
}
