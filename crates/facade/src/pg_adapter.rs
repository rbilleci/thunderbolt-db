//! PostgreSQL wire adapter mappings over the neutral façade types.
//!
//! This module is the **only** place PostgreSQL-specific wire concepts —
//! type OIDs, `SQLSTATE` codes, and the `INSERT 0 N` / `SELECT N` completion-tag
//! grammar — live. A MySQL or HTTP adapter would provide its own equivalent
//! module; the engine and the façade stay free of all of it.

use crate::{CommandTag, DbValue, ErrorCategory, LogicalType, QueryOutcome};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PgValueCodecError {
    #[error("unsupported PostgreSQL type OID {0}")]
    UnsupportedOid(u32),
    #[error("PostgreSQL parameter decoding is not implemented for {0:?}")]
    UnsupportedParameterType(LogicalType),
    #[error("unsupported PostgreSQL format code {0}")]
    UnsupportedFormat(i16),
    #[error("invalid {logical_type:?} parameter in PostgreSQL format {format}")]
    InvalidValue {
        logical_type: LogicalType,
        format: i16,
    },
    #[error("binary result encoding is not implemented for {0:?}")]
    UnsupportedBinaryResult(LogicalType),
}

/// PostgreSQL type OID for a neutral logical type.
pub fn logical_type_oid(ty: LogicalType) -> u32 {
    match ty {
        LogicalType::Int2 => 21,
        LogicalType::Int4 => 23,
        LogicalType::Int8 => 20,
        LogicalType::Numeric => 1700,
        LogicalType::Bool => 16,
        LogicalType::Text => 25,
        LogicalType::Date => 1082,
        LogicalType::Timestamp => 1114,
        LogicalType::Uuid => 2950,
    }
}

/// Neutral logical type for a PostgreSQL type OID. This mapping stays in the protocol adapter;
/// engine and SQL layers never receive OIDs.
pub fn logical_type_from_oid(oid: u32) -> Result<LogicalType, PgValueCodecError> {
    match oid {
        21 => Ok(LogicalType::Int2),
        23 => Ok(LogicalType::Int4),
        20 => Ok(LogicalType::Int8),
        1700 => Ok(LogicalType::Numeric),
        16 => Ok(LogicalType::Bool),
        25 => Ok(LogicalType::Text),
        1082 => Ok(LogicalType::Date),
        1114 => Ok(LogicalType::Timestamp),
        2950 => Ok(LogicalType::Uuid),
        _ => Err(PgValueCodecError::UnsupportedOid(oid)),
    }
}

/// Decode one PostgreSQL Bind parameter for the canonical BENCH integer/UUID vocabulary.
/// A missing value is SQL NULL; text (format 0) and exact-width network-order binary (format 1)
/// are both accepted. Broader codecs remain explicit rather than silently guessing.
pub fn decode_parameter(
    oid: u32,
    format: i16,
    value: Option<&[u8]>,
) -> Result<DbValue, PgValueCodecError> {
    let logical_type = logical_type_from_oid(oid)?;
    if !matches!(format, 0 | 1) {
        return Err(PgValueCodecError::UnsupportedFormat(format));
    }
    // NULL has no payload to decode. Once its declared OID and format code are valid, every
    // supported logical type can bind it even when that type's non-NULL codec is not implemented.
    let Some(value) = value else {
        return Ok(DbValue::Null);
    };
    if !matches!(
        logical_type,
        LogicalType::Int2 | LogicalType::Int4 | LogicalType::Int8 | LogicalType::Uuid
    ) {
        return Err(PgValueCodecError::UnsupportedParameterType(logical_type));
    }
    match format {
        0 => decode_text_parameter(logical_type, value),
        1 => decode_binary_parameter(logical_type, value),
        _ => Err(PgValueCodecError::UnsupportedFormat(format)),
    }
}

fn decode_text_parameter(
    logical_type: LogicalType,
    value: &[u8],
) -> Result<DbValue, PgValueCodecError> {
    let text = std::str::from_utf8(value).map_err(|_| PgValueCodecError::InvalidValue {
        logical_type,
        format: 0,
    })?;
    let invalid = || PgValueCodecError::InvalidValue {
        logical_type,
        format: 0,
    };
    match logical_type {
        LogicalType::Int2 => parse_pg_integer(text, i16::MIN.into(), i16::MAX.into())
            .map(|value| DbValue::Int2(value as i16))
            .ok_or_else(invalid),
        LogicalType::Int4 => parse_pg_integer(text, i32::MIN.into(), i32::MAX.into())
            .map(|value| DbValue::Int4(value as i32))
            .ok_or_else(invalid),
        LogicalType::Int8 => parse_pg_integer(text, i64::MIN, i64::MAX)
            .map(DbValue::Int8)
            .ok_or_else(invalid),
        LogicalType::Uuid => parse_pg_uuid(text).map(DbValue::Uuid).ok_or_else(invalid),
        _ => unreachable!("decode_parameter restricts the canonical vocabulary"),
    }
}

/// Match PostgreSQL's `pg_strtoint*` text input: ASCII leading/trailing whitespace, an optional
/// sign, decimal/`0x`/`0o`/`0b` forms, and single underscore separators. Accumulate against the
/// target type's signed bound so the minimum value remains representable without wrapping.
fn parse_pg_integer(input: &str, min: i64, max: i64) -> Option<i64> {
    let input = input.trim_matches(|ch: char| ch.is_ascii_whitespace());
    let (negative, unsigned) = match input.as_bytes().first() {
        Some(b'-') => (true, &input[1..]),
        Some(b'+') => (false, &input[1..]),
        _ => (false, input),
    };
    let (radix, digits, prefix_underscore) = if unsigned
        .get(..2)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("0x"))
    {
        (16_u32, &unsigned[2..], true)
    } else if unsigned
        .get(..2)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("0o"))
    {
        (8, &unsigned[2..], true)
    } else if unsigned
        .get(..2)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("0b"))
    {
        (2, &unsigned[2..], true)
    } else {
        (10, unsigned, false)
    };
    if digits.is_empty() {
        return None;
    }
    let bytes = digits.as_bytes();
    let magnitude_limit = if negative {
        min.checked_abs()
            .map_or(i64::MAX as u128 + 1, |v| v as u128)
    } else {
        max as u128
    };
    let mut magnitude = 0_u128;
    let mut saw_digit = false;
    for (index, byte) in bytes.iter().copied().enumerate() {
        if byte == b'_' {
            let valid_before = index > 0 && ascii_digit_value(bytes[index - 1], radix).is_some();
            let valid_after = bytes
                .get(index + 1)
                .is_some_and(|next| ascii_digit_value(*next, radix).is_some());
            if !(valid_after && (valid_before || (prefix_underscore && index == 0))) {
                return None;
            }
            continue;
        }
        let digit = ascii_digit_value(byte, radix)?;
        saw_digit = true;
        magnitude = magnitude
            .checked_mul(u128::from(radix))?
            .checked_add(u128::from(digit))?;
        if magnitude > magnitude_limit {
            return None;
        }
    }
    if !saw_digit {
        return None;
    }
    if negative {
        if magnitude == i64::MAX as u128 + 1 {
            Some(i64::MIN)
        } else {
            Some(-(magnitude as i64))
        }
    } else {
        Some(magnitude as i64)
    }
}

fn ascii_digit_value(byte: u8, radix: u32) -> Option<u32> {
    let value = match byte {
        b'0'..=b'9' => u32::from(byte - b'0'),
        b'a'..=b'f' => u32::from(byte - b'a') + 10,
        b'A'..=b'F' => u32::from(byte - b'A') + 10,
        _ => return None,
    };
    (value < radix).then_some(value)
}

/// Match PostgreSQL `uuid_in`: 32 hexadecimal digits, optional braces, and an optional hyphen
/// after any four-digit group. Whitespace is data here and therefore rejected.
fn parse_pg_uuid(input: &str) -> Option<[u8; 16]> {
    let (body, braced) = input
        .strip_prefix('{')
        .map_or((input, false), |body| (body, true));
    let mut output = [0_u8; 16];
    let mut at = 0_usize;
    let bytes = body.as_bytes();
    for (index, slot) in output.iter_mut().enumerate() {
        let hi = ascii_digit_value(*bytes.get(at)?, 16)?;
        let lo = ascii_digit_value(*bytes.get(at + 1)?, 16)?;
        *slot = ((hi << 4) | lo) as u8;
        at += 2;
        if index % 2 == 1 && index < 15 && bytes.get(at) == Some(&b'-') {
            at += 1;
        }
    }
    if braced {
        if bytes.get(at) != Some(&b'}') {
            return None;
        }
        at += 1;
    }
    (at == bytes.len()).then_some(output)
}

fn decode_binary_parameter(
    logical_type: LogicalType,
    value: &[u8],
) -> Result<DbValue, PgValueCodecError> {
    let invalid = || PgValueCodecError::InvalidValue {
        logical_type,
        format: 1,
    };
    match logical_type {
        LogicalType::Int2 => value
            .try_into()
            .map(i16::from_be_bytes)
            .map(DbValue::Int2)
            .map_err(|_| invalid()),
        LogicalType::Int4 => value
            .try_into()
            .map(i32::from_be_bytes)
            .map(DbValue::Int4)
            .map_err(|_| invalid()),
        LogicalType::Int8 => value
            .try_into()
            .map(i64::from_be_bytes)
            .map(DbValue::Int8)
            .map_err(|_| invalid()),
        LogicalType::Uuid => value.try_into().map(DbValue::Uuid).map_err(|_| invalid()),
        _ => unreachable!("decode_parameter restricts the canonical vocabulary"),
    }
}

/// Encode one neutral result value in PostgreSQL text (0) or binary (1) format. SQL NULL is
/// represented out-of-band as `None`. PostgreSQL's binary `text` representation is the raw UTF-8
/// payload, so it shares the neutral string bytes without a second wire-specific value path.
pub fn encode_result_value(
    value: &DbValue,
    format: i16,
) -> Result<Option<Vec<u8>>, PgValueCodecError> {
    if !matches!(format, 0 | 1) {
        return Err(PgValueCodecError::UnsupportedFormat(format));
    }
    if matches!(value, DbValue::Null) {
        return Ok(None);
    }
    match format {
        0 => Ok(Some(db_value_text(value).into_bytes())),
        1 => match value {
            DbValue::Int2(value) => Ok(Some(value.to_be_bytes().to_vec())),
            DbValue::Int4(value) => Ok(Some(value.to_be_bytes().to_vec())),
            DbValue::Int8(value) => Ok(Some(value.to_be_bytes().to_vec())),
            DbValue::Text(value) => Ok(Some(value.as_bytes().to_vec())),
            DbValue::Uuid(value) => Ok(Some(value.to_vec())),
            other => Err(PgValueCodecError::UnsupportedBinaryResult(
                logical_type_for_value(other),
            )),
        },
        _ => Err(PgValueCodecError::UnsupportedFormat(format)),
    }
}

fn logical_type_for_value(value: &DbValue) -> LogicalType {
    match value {
        DbValue::Null => unreachable!("NULL returns before type lookup"),
        DbValue::Int2(_) => LogicalType::Int2,
        DbValue::Int4(_) => LogicalType::Int4,
        DbValue::Int8(_) => LogicalType::Int8,
        DbValue::Numeric(_) => LogicalType::Numeric,
        DbValue::Bool(_) => LogicalType::Bool,
        DbValue::Text(_) => LogicalType::Text,
        DbValue::Date(_) => LogicalType::Date,
        DbValue::Timestamp(_) => LogicalType::Timestamp,
        DbValue::Uuid(_) => LogicalType::Uuid,
    }
}

/// PostgreSQL wire type size (negative for variable-length) for a logical type.
pub fn logical_type_size(ty: LogicalType) -> i16 {
    match ty {
        LogicalType::Int2 => 2,
        LogicalType::Int4 => 4,
        LogicalType::Int8 => 8,
        LogicalType::Numeric => -1,
        LogicalType::Bool => 1,
        LogicalType::Text => -1,
        LogicalType::Date => 4,
        LogicalType::Timestamp => 8,
        LogicalType::Uuid => 16,
    }
}

/// Text-format wire encoding of a *non-null* neutral value. A `Numeric` renders via
/// its fixed-point decimal string (with the decimal point); a `Bool` renders as the
/// PostgreSQL `t`/`f` text form. NULL is represented out-of-band on the wire (a `-1`
/// `DataRow` field length), so callers at the wire boundary must use
/// [`db_value_text_opt`], which returns `None` for [`DbValue::Null`]; this function's
/// `Null` arm renders `NULL` only as a defensive textual fallback for direct callers.
pub fn db_value_text(value: &DbValue) -> String {
    match value {
        DbValue::Null => "NULL".to_string(),
        DbValue::Int2(value) => value.to_string(),
        DbValue::Int4(value) => value.to_string(),
        DbValue::Int8(value) => value.to_string(),
        DbValue::Numeric(value) => value.to_decimal_string(),
        DbValue::Bool(value) => if *value { "t" } else { "f" }.to_string(),
        DbValue::Text(value) => value.clone(),
        DbValue::Date(value) => gpu_db_sql::datetime::format_date(*value),
        DbValue::Timestamp(value) => gpu_db_sql::datetime::format_timestamp(*value),
        DbValue::Uuid(value) => gpu_db_sql::uuid::format_uuid(value),
    }
}

/// Wire-boundary encoding of a neutral value: [`DbValue::Null`] maps to `None` (the
/// PostgreSQL `DataRow` `-1` field length), every other value to `Some(text)`. This is
/// the encoder the engine→wire path uses so that a NULL cell is emitted as a NULL field
/// rather than as the text `"NULL"`.
pub fn db_value_text_opt(value: &DbValue) -> Option<String> {
    match value {
        DbValue::Null => None,
        other => Some(db_value_text(other)),
    }
}

/// `SQLSTATE` for a neutral error category.
pub fn error_sqlstate(category: ErrorCategory) -> &'static str {
    match category {
        ErrorCategory::Syntax => "42601",
        ErrorCategory::Unsupported => "0A000",
        ErrorCategory::UndefinedRelation => "42P01",
        ErrorCategory::UndefinedColumn => "42703",
        ErrorCategory::IndeterminateDatatype => "42P18",
        ErrorCategory::DatatypeMismatch => "42804",
        ErrorCategory::InvalidRequest => "08P01",
        ErrorCategory::ResourceExhausted => "53000",
        ErrorCategory::InFailedTransaction => "25P02",
        ErrorCategory::UniqueViolation => "23505",
        ErrorCategory::Engine => "XX000",
        ErrorCategory::Internal => "XX000",
        // Class 40 — Transaction Rollback; 40001 serialization_failure is the retryable code
        // PostgreSQL clients already retry on (write-half MVCC, Stage 4).
        ErrorCategory::Serialization => "40001",
    }
}

/// The PostgreSQL `CommandComplete` tag for a neutral outcome.
pub fn command_complete_tag(outcome: &QueryOutcome) -> String {
    match outcome {
        // Empty statements get an EmptyQueryResponse, not a CommandComplete, so an
        // adapter should special-case `QueryOutcome::Empty` before calling this;
        // the empty string here is a defensive fallback.
        QueryOutcome::Empty => String::new(),
        QueryOutcome::Rows { rows, .. } => format!("SELECT {}", rows.len()),
        QueryOutcome::Returning {
            tag, rows_affected, ..
        } => match tag {
            CommandTag::Insert => format!("INSERT 0 {rows_affected}"),
            CommandTag::Update => format!("UPDATE {rows_affected}"),
            CommandTag::Delete => format!("DELETE {rows_affected}"),
            _ => format!("{} {rows_affected}", command_tag_label(tag)),
        },
        QueryOutcome::Command { tag, rows_affected } => match tag {
            CommandTag::Begin => "BEGIN".to_string(),
            CommandTag::Commit => "COMMIT".to_string(),
            CommandTag::Rollback => "ROLLBACK".to_string(),
            CommandTag::CreateTable => "CREATE TABLE".to_string(),
            CommandTag::CreateIndex => "CREATE INDEX".to_string(),
            CommandTag::Insert => format!("INSERT 0 {}", rows_affected.unwrap_or(0)),
            CommandTag::Update => format!("UPDATE {}", rows_affected.unwrap_or(0)),
            CommandTag::Delete => format!("DELETE {}", rows_affected.unwrap_or(0)),
            CommandTag::Other(label) => label.clone(),
        },
    }
}

fn command_tag_label(tag: &CommandTag) -> &str {
    match tag {
        CommandTag::Begin => "BEGIN",
        CommandTag::Commit => "COMMIT",
        CommandTag::Rollback => "ROLLBACK",
        CommandTag::CreateTable => "CREATE TABLE",
        CommandTag::CreateIndex => "CREATE INDEX",
        CommandTag::Insert => "INSERT",
        CommandTag::Update => "UPDATE",
        CommandTag::Delete => "DELETE",
        CommandTag::Other(label) => label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_neutral_types_to_postgres_oids() {
        assert_eq!(logical_type_oid(LogicalType::Int4), 23);
        assert_eq!(logical_type_oid(LogicalType::Int8), 20);
        assert_eq!(logical_type_oid(LogicalType::Numeric), 1700);
        assert_eq!(logical_type_oid(LogicalType::Text), 25);
        assert_eq!(logical_type_from_oid(21), Ok(LogicalType::Int2));
        assert_eq!(logical_type_from_oid(2950), Ok(LogicalType::Uuid));
        assert_eq!(
            logical_type_from_oid(0),
            Err(PgValueCodecError::UnsupportedOid(0))
        );
    }

    #[test]
    fn canonical_parameter_codecs_cover_text_binary_null_and_fail_loudly() {
        let uuid = gpu_db_sql::uuid::parse_uuid("550e8400-e29b-41d4-a716-446655440000").unwrap();
        for (oid, value, text, binary) in [
            (
                21,
                DbValue::Int2(-7),
                b"-7".as_slice(),
                (-7_i16).to_be_bytes().to_vec(),
            ),
            (
                23,
                DbValue::Int4(42),
                b"42".as_slice(),
                42_i32.to_be_bytes().to_vec(),
            ),
            (
                20,
                DbValue::Int8(i64::MIN + 9),
                b"-9223372036854775799".as_slice(),
                (i64::MIN + 9).to_be_bytes().to_vec(),
            ),
            (
                2950,
                DbValue::Uuid(uuid),
                b"550e8400-e29b-41d4-a716-446655440000".as_slice(),
                uuid.to_vec(),
            ),
        ] {
            assert_eq!(decode_parameter(oid, 0, Some(text)).unwrap(), value);
            assert_eq!(decode_parameter(oid, 1, Some(&binary)).unwrap(), value);
            assert_eq!(encode_result_value(&value, 0).unwrap(), Some(text.to_vec()));
            assert_eq!(encode_result_value(&value, 1).unwrap(), Some(binary));
            assert_eq!(decode_parameter(oid, 0, None).unwrap(), DbValue::Null);
        }
        assert!(matches!(
            decode_parameter(23, 1, Some(&[0, 1, 2])),
            Err(PgValueCodecError::InvalidValue { .. })
        ));
        assert!(matches!(
            decode_parameter(2950, 0, Some(b"not-a-uuid")),
            Err(PgValueCodecError::InvalidValue { .. })
        ));
        assert_eq!(
            decode_parameter(25, 0, Some(b"text")),
            Err(PgValueCodecError::UnsupportedParameterType(
                LogicalType::Text
            ))
        );
        for oid in [16, 25, 1082, 1114, 1700] {
            assert_eq!(decode_parameter(oid, 0, None).unwrap(), DbValue::Null);
            assert_eq!(decode_parameter(oid, 1, None).unwrap(), DbValue::Null);
        }
        assert_eq!(
            encode_result_value(&DbValue::Text("Ada".to_string()), 1).unwrap(),
            Some(b"Ada".to_vec())
        );
        assert_eq!(encode_result_value(&DbValue::Null, 1).unwrap(), None);
        assert_eq!(
            decode_parameter(23, 2, None),
            Err(PgValueCodecError::UnsupportedFormat(2))
        );
        assert_eq!(
            encode_result_value(&DbValue::Null, 2),
            Err(PgValueCodecError::UnsupportedFormat(2))
        );

        assert_eq!(
            decode_parameter(23, 0, Some(b" \t-0x7f_ff\n")).unwrap(),
            DbValue::Int4(-32_767)
        );
        assert_eq!(
            decode_parameter(20, 0, Some(b"0b_1010_0001")).unwrap(),
            DbValue::Int8(161)
        );
        assert_eq!(
            decode_parameter(21, 0, Some(b"0o_7_777")).unwrap(),
            DbValue::Int2(4095)
        );
        for invalid in [b"_1".as_slice(), b"1__0", b"0x_", b"32768"] {
            assert!(matches!(
                decode_parameter(21, 0, Some(invalid)),
                Err(PgValueCodecError::InvalidValue { .. })
            ));
        }

        assert_eq!(
            decode_parameter(2950, 0, Some(b"{a0ee-bc99-9c0b-4ef8-bb6d-6bb9-bd38-0a11}")).unwrap(),
            DbValue::Uuid([
                0xa0, 0xee, 0xbc, 0x99, 0x9c, 0x0b, 0x4e, 0xf8, 0xbb, 0x6d, 0x6b, 0xb9, 0xbd, 0x38,
                0x0a, 0x11,
            ])
        );
        assert!(matches!(
            decode_parameter(2950, 0, Some(b" 550e8400-e29b-41d4-a716-446655440000")),
            Err(PgValueCodecError::InvalidValue { .. })
        ));
    }

    #[test]
    fn maps_neutral_error_categories_to_sqlstate() {
        assert_eq!(error_sqlstate(ErrorCategory::Syntax), "42601");
        assert_eq!(error_sqlstate(ErrorCategory::Unsupported), "0A000");
        assert_eq!(error_sqlstate(ErrorCategory::UniqueViolation), "23505");
        assert_eq!(error_sqlstate(ErrorCategory::Engine), "XX000");
    }

    #[test]
    fn formats_select_completion_tag_from_row_count() {
        let outcome = QueryOutcome::Rows {
            columns: vec![],
            rows: vec![vec![DbValue::Int4(1)], vec![DbValue::Int4(2)]],
        };
        assert_eq!(command_complete_tag(&outcome), "SELECT 2");
    }
}
