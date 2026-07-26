//! PostgreSQL wire adapter mappings over the neutral façade types.
//!
//! This module is the **only** place PostgreSQL-specific wire concepts —
//! type OIDs, `SQLSTATE` codes, and the `INSERT 0 N` / `SELECT N` completion-tag
//! grammar — live. A MySQL or HTTP adapter would provide its own equivalent
//! module; the engine and the façade stay free of all of it.

use crate::{CommandTag, DbValue, ErrorCategory, LogicalType, QueryOutcome};
use gpu_db_sql::Decimal128;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PgValueCodecError {
    #[error("unsupported PostgreSQL type OID {0}")]
    UnsupportedOid(u32),
    #[error("unsupported PostgreSQL format code {0}")]
    UnsupportedFormat(i16),
    #[error("invalid {logical_type:?} parameter in PostgreSQL format {format}")]
    InvalidValue {
        logical_type: LogicalType,
        format: i16,
    },
    #[error("{logical_type:?} parameter is outside the supported numeric range")]
    NumericValueOutOfRange {
        logical_type: LogicalType,
        format: i16,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericCodecFailure {
    Invalid,
    OutOfRange,
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
        // JDBC and some pool clients declare String Bind parameters as VARCHAR rather than
        // TEXT. Both feed the one neutral text domain; result metadata remains canonical TEXT.
        25 | 1043 => Ok(LogicalType::Text),
        1082 => Ok(LogicalType::Date),
        1114 => Ok(LogicalType::Timestamp),
        2950 => Ok(LogicalType::Uuid),
        _ => Err(PgValueCodecError::UnsupportedOid(oid)),
    }
}

/// Decode one PostgreSQL Bind parameter for every logical type the facade currently exposes.
/// A missing value is SQL NULL; text (format 0) and PostgreSQL's network-order binary (format 1)
/// are both accepted. The codec is deliberately finite: numeric specials and values outside the
/// fixed [`Decimal128`] representation reject rather than being rounded or silently approximated.
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
    // supported logical type can bind it.
    let Some(value) = value else {
        return Ok(DbValue::Null);
    };
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
            .map_err(|failure| numeric_codec_error(logical_type, 0, failure)),
        LogicalType::Int4 => parse_pg_integer(text, i32::MIN.into(), i32::MAX.into())
            .map(|value| DbValue::Int4(value as i32))
            .map_err(|failure| numeric_codec_error(logical_type, 0, failure)),
        LogicalType::Int8 => parse_pg_integer(text, i64::MIN, i64::MAX)
            .map(DbValue::Int8)
            .map_err(|failure| numeric_codec_error(logical_type, 0, failure)),
        LogicalType::Numeric => parse_pg_numeric(text)
            .map(DbValue::Numeric)
            .map_err(|failure| numeric_codec_error(logical_type, 0, failure)),
        LogicalType::Bool => parse_pg_bool(text).map(DbValue::Bool).ok_or_else(invalid),
        LogicalType::Text => {
            if text.contains('\0') {
                Err(invalid())
            } else {
                Ok(DbValue::Text(text.to_string()))
            }
        }
        LogicalType::Uuid => parse_pg_uuid(text).map(DbValue::Uuid).ok_or_else(invalid),
        LogicalType::Date => gpu_db_sql::datetime::parse_date(text)
            .map(DbValue::Date)
            .ok_or_else(invalid),
        LogicalType::Timestamp => gpu_db_sql::datetime::parse_timestamp(text)
            .map(DbValue::Timestamp)
            .ok_or_else(invalid),
    }
}

/// PostgreSQL boolean input accepts case-insensitive unique prefixes of true/false, yes/no, and
/// on/off, plus the one-byte `1`/`0` spellings. `o` is deliberately ambiguous and therefore not
/// accepted.
fn parse_pg_bool(input: &str) -> Option<bool> {
    let input = input.trim_matches(|ch: char| ch.is_ascii_whitespace());
    let lower = input.to_ascii_lowercase();
    match lower.as_str() {
        "1" | "t" | "tr" | "tru" | "true" | "y" | "ye" | "yes" | "on" => Some(true),
        "0" | "f" | "fa" | "fal" | "fals" | "false" | "n" | "no" | "of" | "off" => Some(false),
        _ => None,
    }
}

/// Parse PostgreSQL 16's finite NUMERIC input vocabulary into the bounded Decimal128 domain.
/// `numeric_in` selects `0x`/`0o`/`0b` after an optional sign, and both its decimal and
/// non-decimal paths accept digit separators. We retain those grammar rules while rejecting
/// PostgreSQL's NaN/infinity specials because the neutral value domain is finite.
fn parse_pg_numeric(input: &str) -> Result<Decimal128, NumericCodecFailure> {
    let input = input.trim_matches(|ch: char| ch.is_ascii_whitespace());
    let (negative, body) = match input.as_bytes().first() {
        Some(b'-') => (true, &input[1..]),
        Some(b'+') => (false, &input[1..]),
        _ => (false, input),
    };
    let bytes = body.as_bytes();
    if bytes.is_empty() {
        return Err(NumericCodecFailure::Invalid);
    }
    let radix = match bytes.get(..2) {
        Some(b"0x") | Some(b"0X") => Some(16),
        Some(b"0o") | Some(b"0O") => Some(8),
        Some(b"0b") | Some(b"0B") => Some(2),
        _ => None,
    };
    match radix {
        Some(radix) => parse_pg_non_decimal_numeric(&bytes[2..], negative, radix),
        None => parse_pg_decimal_numeric(bytes, negative),
    }
}

fn parse_pg_non_decimal_numeric(
    digits: &[u8],
    negative: bool,
    radix: u32,
) -> Result<Decimal128, NumericCodecFailure> {
    let mut magnitude = 0_u128;
    let mut out_of_range = false;
    let mut saw_digit = false;
    for (index, byte) in digits.iter().copied().enumerate() {
        if byte == b'_' {
            // PostgreSQL's non-decimal reader permits one separator immediately after the
            // base prefix, but every underscore must lead to a valid base digit.
            if !digits
                .get(index + 1)
                .is_some_and(|next| ascii_digit_value(*next, radix).is_some())
            {
                return Err(NumericCodecFailure::Invalid);
            }
            continue;
        }
        let digit = ascii_digit_value(byte, radix).ok_or(NumericCodecFailure::Invalid)?;
        saw_digit = true;
        if !out_of_range {
            match magnitude
                .checked_mul(u128::from(radix))
                .and_then(|value| value.checked_add(u128::from(digit)))
            {
                Some(next) => magnitude = next,
                None => out_of_range = true,
            }
        }
    }
    if !saw_digit {
        return Err(NumericCodecFailure::Invalid);
    }
    if out_of_range {
        return Err(NumericCodecFailure::OutOfRange);
    }
    decimal_from_magnitude(magnitude, negative, 0)
}

fn parse_pg_decimal_numeric(
    bytes: &[u8],
    negative: bool,
) -> Result<Decimal128, NumericCodecFailure> {
    let mut at = 0;
    let mut decimal_point = false;
    if bytes.first() == Some(&b'.') {
        decimal_point = true;
        at += 1;
        if !bytes.get(at).is_some_and(u8::is_ascii_digit) {
            return Err(NumericCodecFailure::Invalid);
        }
    }

    let mut magnitude = 0_u128;
    let mut magnitude_out_of_range = false;
    let mut fractional_digits = 0_i64;
    let mut saw_digit = false;
    while let Some(byte) = bytes.get(at).copied() {
        if byte.is_ascii_digit() {
            saw_digit = true;
            if decimal_point {
                fractional_digits = fractional_digits.saturating_add(1);
            }
            if !magnitude_out_of_range {
                match magnitude
                    .checked_mul(10)
                    .and_then(|value| value.checked_add(u128::from(byte - b'0')))
                {
                    Some(next) => magnitude = next,
                    None => magnitude_out_of_range = true,
                }
            }
            at += 1;
        } else if byte == b'.' {
            if decimal_point || bytes.get(at + 1) == Some(&b'_') {
                return Err(NumericCodecFailure::Invalid);
            }
            decimal_point = true;
            at += 1;
        } else if byte == b'_' {
            let valid_before = at > 0 && bytes[at - 1].is_ascii_digit();
            let valid_after = bytes.get(at + 1).is_some_and(u8::is_ascii_digit);
            if !(valid_before && valid_after) {
                return Err(NumericCodecFailure::Invalid);
            }
            at += 1;
        } else {
            break;
        }
    }
    if !saw_digit {
        return Err(NumericCodecFailure::Invalid);
    }
    let exponent = if matches!(bytes.get(at), Some(b'e' | b'E')) {
        parse_pg_numeric_exponent(&bytes[at + 1..])?
    } else if at == bytes.len() {
        0
    } else {
        return Err(NumericCodecFailure::Invalid);
    };
    if magnitude_out_of_range {
        return Err(NumericCodecFailure::OutOfRange);
    }
    let scale = fractional_digits
        .checked_sub(exponent)
        .ok_or(NumericCodecFailure::OutOfRange)?;
    if scale < 0 {
        let shift = scale.unsigned_abs();
        if magnitude == 0 {
            return Ok(Decimal128::new(0, 0));
        }
        // `Decimal128` cannot hold more than 38 decimal places of positive shifting. Avoid a
        // client-controlled loop while retaining the exact checked multiplication for the edge.
        if shift > 38 {
            return Err(NumericCodecFailure::OutOfRange);
        }
        for _ in 0..shift {
            magnitude = magnitude
                .checked_mul(10)
                .ok_or(NumericCodecFailure::OutOfRange)?;
        }
        return decimal_from_magnitude(magnitude, negative, 0);
    }
    decimal_from_magnitude(
        magnitude,
        negative,
        u8::try_from(scale).map_err(|_| NumericCodecFailure::OutOfRange)?,
    )
}

fn parse_pg_numeric_exponent(input: &[u8]) -> Result<i64, NumericCodecFailure> {
    let (negative, digits) = match input.first() {
        Some(b'-') => (true, &input[1..]),
        Some(b'+') => (false, &input[1..]),
        _ => (false, input),
    };
    if !digits.first().is_some_and(u8::is_ascii_digit) {
        return Err(NumericCodecFailure::Invalid);
    }
    // numeric.c bounds its intermediate exponent to PG_INT32_MAX / 2 before applying the
    // decimal scale adjustment. Continue validating after that boundary so a malformed suffix
    // remains 22P02 rather than being mislabeled as a representational overflow.
    const PG_NUMERIC_EXPONENT_MAX: i64 = i32::MAX as i64 / 2;
    let mut magnitude = 0_i64;
    let mut out_of_range = false;
    for (index, byte) in digits.iter().copied().enumerate() {
        if byte.is_ascii_digit() {
            if !out_of_range {
                match magnitude
                    .checked_mul(10)
                    .and_then(|value| value.checked_add(i64::from(byte - b'0')))
                {
                    Some(next) if next <= PG_NUMERIC_EXPONENT_MAX => magnitude = next,
                    Some(_) | None => out_of_range = true,
                }
            }
        } else if byte == b'_' {
            let valid_before = index > 0 && digits[index - 1].is_ascii_digit();
            let valid_after = digits.get(index + 1).is_some_and(u8::is_ascii_digit);
            if !(valid_before && valid_after) {
                return Err(NumericCodecFailure::Invalid);
            }
        } else {
            return Err(NumericCodecFailure::Invalid);
        }
    }
    if out_of_range {
        return Err(NumericCodecFailure::OutOfRange);
    }
    Ok(if negative { -magnitude } else { magnitude })
}

fn decimal_from_magnitude(
    magnitude: u128,
    negative: bool,
    scale: u8,
) -> Result<Decimal128, NumericCodecFailure> {
    let mantissa = if negative {
        match i128::try_from(magnitude) {
            Ok(value) => -value,
            Err(_) if magnitude == i128::MAX as u128 + 1 => i128::MIN,
            Err(_) => return Err(NumericCodecFailure::OutOfRange),
        }
    } else {
        i128::try_from(magnitude).map_err(|_| NumericCodecFailure::OutOfRange)?
    };
    Ok(Decimal128::new(mantissa, scale))
}

/// Match PostgreSQL's `pg_strtoint*` text input: ASCII leading/trailing whitespace, an optional
/// sign, decimal/`0x`/`0o`/`0b` forms, and single underscore separators. Accumulate against the
/// target type's signed bound so the minimum value remains representable without wrapping.
///
/// A fully well-formed value outside that bound is a numeric range error, not invalid input
/// syntax. Keep scanning after the first out-of-range digit so malformed suffixes do not get
/// mislabeled as an overflow, without ever accumulating an unbounded client value.
fn parse_pg_integer(input: &str, min: i64, max: i64) -> Result<i64, NumericCodecFailure> {
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
        return Err(NumericCodecFailure::Invalid);
    }
    let bytes = digits.as_bytes();
    let magnitude_limit = if negative {
        min.checked_abs()
            .map_or(i64::MAX as u128 + 1, |v| v as u128)
    } else {
        max as u128
    };
    let mut magnitude = 0_u128;
    let mut out_of_range = false;
    let mut saw_digit = false;
    for (index, byte) in bytes.iter().copied().enumerate() {
        if byte == b'_' {
            let valid_before = index > 0 && ascii_digit_value(bytes[index - 1], radix).is_some();
            let valid_after = bytes
                .get(index + 1)
                .is_some_and(|next| ascii_digit_value(*next, radix).is_some());
            if !(valid_after && (valid_before || (prefix_underscore && index == 0))) {
                return Err(NumericCodecFailure::Invalid);
            }
            continue;
        }
        let digit = ascii_digit_value(byte, radix).ok_or(NumericCodecFailure::Invalid)?;
        saw_digit = true;
        if !out_of_range {
            match magnitude
                .checked_mul(u128::from(radix))
                .and_then(|value| value.checked_add(u128::from(digit)))
            {
                Some(next) if next <= magnitude_limit => magnitude = next,
                Some(_) | None => out_of_range = true,
            }
        }
    }
    if !saw_digit {
        return Err(NumericCodecFailure::Invalid);
    }
    if out_of_range {
        return Err(NumericCodecFailure::OutOfRange);
    }
    if negative {
        if magnitude == i64::MAX as u128 + 1 {
            Ok(i64::MIN)
        } else {
            Ok(-(magnitude as i64))
        }
    } else {
        Ok(magnitude as i64)
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
        LogicalType::Numeric => decode_binary_numeric(value)
            .map(DbValue::Numeric)
            .map_err(|failure| numeric_codec_error(logical_type, 1, failure)),
        LogicalType::Bool => value
            .try_into()
            .map(|bytes: &[u8; 1]| DbValue::Bool(bytes[0] != 0))
            .map_err(|_| invalid()),
        LogicalType::Text => std::str::from_utf8(value)
            .map_err(|_| invalid())
            .and_then(|value| {
                if value.contains('\0') {
                    Err(invalid())
                } else {
                    Ok(DbValue::Text(value.to_string()))
                }
            }),
        LogicalType::Uuid => value.try_into().map(DbValue::Uuid).map_err(|_| invalid()),
        LogicalType::Date => value
            .try_into()
            .map(i32::from_be_bytes)
            .map(DbValue::Date)
            .map_err(|_| invalid()),
        LogicalType::Timestamp => value
            .try_into()
            .map(i64::from_be_bytes)
            .map(DbValue::Timestamp)
            .map_err(|_| invalid()),
    }
}

const NUMERIC_POS: u16 = 0x0000;
const NUMERIC_NEG: u16 = 0x4000;
const NUMERIC_DSCALE_MASK: u16 = 0x3fff;

/// Decode PostgreSQL's finite base-10000 numeric wire format. Values outside Decimal128's i128
/// mantissa/u8 scale envelope are errors even when PostgreSQL itself could store them.
fn decode_binary_numeric(value: &[u8]) -> Result<Decimal128, NumericCodecFailure> {
    if value.len() < 8 {
        return Err(NumericCodecFailure::Invalid);
    }
    let ndigits = usize::from(u16::from_be_bytes(value[0..2].try_into().expect("header")));
    let wire_weight = i16::from_be_bytes(value[2..4].try_into().expect("header"));
    let sign = u16::from_be_bytes(value[4..6].try_into().expect("header"));
    let wire_dscale = u16::from_be_bytes(value[6..8].try_into().expect("header"));
    if !matches!(sign, NUMERIC_POS | NUMERIC_NEG)
        || value.len()
            != 8usize
                .checked_add(ndigits.checked_mul(2).ok_or(NumericCodecFailure::Invalid)?)
                .ok_or(NumericCodecFailure::Invalid)?
    {
        return Err(NumericCodecFailure::Invalid);
    }
    // PostgreSQL 16's numeric_recv rejects the two reserved high dscale bits before considering
    // the actual display-scale envelope. Do not turn a malformed header into our finite-range
    // error merely because its raw u16 happens to exceed Decimal128's u8 scale.
    if wire_dscale & !NUMERIC_DSCALE_MASK != 0 {
        return Err(NumericCodecFailure::Invalid);
    }
    let dscale = usize::from(wire_dscale);
    if dscale > usize::from(u8::MAX) {
        return Err(NumericCodecFailure::OutOfRange);
    }
    let digit_at = |index: usize| -> u16 {
        let offset = 8 + index * 2;
        u16::from_be_bytes(value[offset..offset + 2].try_into().expect("numeric digit"))
    };
    let mut first_nonzero = None;
    for index in 0..ndigits {
        let digit = digit_at(index);
        if digit > 9_999 {
            return Err(NumericCodecFailure::Invalid);
        }
        if first_nonzero.is_none() && digit != 0 {
            first_nonzero = Some(index);
        }
    }
    let Some(first_nonzero) = first_nonzero else {
        // numeric_recv normalizes signed zero through strip_var. Decimal128 has no negative-zero
        // representation, so preserve the display scale and normalize either finite sign.
        return Ok(Decimal128::new(0, dscale as u8));
    };

    // normalize_var/strip_var discard legal leading zero groups. Keep the arithmetic weight wide
    // enough for an i16 wire weight plus every declared group before judging Decimal128 range.
    let normalized_weight = i32::from(wire_weight)
        .checked_sub(i32::try_from(first_nonzero).expect("u16 numeric digit count fits i32"))
        .expect("numeric normalized weight fits i32");
    let scale = i32::try_from(dscale).expect("Decimal128 scale fits i32");
    let mut magnitude = 0_u128;
    for index in first_nonzero..ndigits {
        let digit = u128::from(digit_at(index));
        if digit == 0 {
            continue;
        }
        let exponent = normalized_weight
            .checked_sub(
                i32::try_from(index - first_nonzero).expect("u16 numeric digit distance fits i32"),
            )
            .expect("numeric normalized exponent fits i32");
        let decimal_power = scale
            .checked_add(
                exponent
                    .checked_mul(4)
                    .expect("numeric exponent arithmetic fits i32"),
            )
            .expect("numeric decimal power fits i32");
        let contribution = if decimal_power >= 0 {
            let power = u32::try_from(decimal_power).expect("nonnegative decimal power");
            if power > 38 {
                return Err(NumericCodecFailure::OutOfRange);
            }
            digit
                .checked_mul(ten_to_u128(power))
                .ok_or(NumericCodecFailure::OutOfRange)?
        } else if decimal_power >= -3 {
            // numeric_recv truncates digits hidden beyond dscale. A partially visible final
            // base-10000 group contributes only its leading decimal digits.
            digit / ten_to_u128(decimal_power.unsigned_abs())
        } else {
            continue;
        };
        magnitude = magnitude
            .checked_add(contribution)
            .ok_or(NumericCodecFailure::OutOfRange)?;
    }
    decimal_from_magnitude(magnitude, sign == NUMERIC_NEG, dscale as u8)
}

fn ten_to_u128(power: u32) -> u128 {
    let mut value = 1_u128;
    for _ in 0..power {
        value *= 10;
    }
    value
}

fn numeric_codec_error(
    logical_type: LogicalType,
    format: i16,
    failure: NumericCodecFailure,
) -> PgValueCodecError {
    match failure {
        NumericCodecFailure::Invalid => PgValueCodecError::InvalidValue {
            logical_type,
            format,
        },
        NumericCodecFailure::OutOfRange => PgValueCodecError::NumericValueOutOfRange {
            logical_type,
            format,
        },
    }
}

fn encode_binary_numeric(value: Decimal128) -> Vec<u8> {
    let dscale = value.scale;
    if value.mantissa == 0 {
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&0_i16.to_be_bytes());
        out.extend_from_slice(&0_i16.to_be_bytes());
        out.extend_from_slice(&NUMERIC_POS.to_be_bytes());
        out.extend_from_slice(&u16::from(dscale).to_be_bytes());
        return out;
    }
    let rendered = value.to_decimal_string();
    let (negative, rendered) = rendered
        .strip_prefix('-')
        .map_or((false, rendered.as_str()), |body| (true, body));
    let (whole, fractional) = rendered.split_once('.').unwrap_or((rendered, ""));
    let whole = whole.trim_start_matches('0');
    let mut groups = Vec::new();
    let whole_groups = if whole.is_empty() {
        0
    } else {
        let first = whole.len() % 4;
        let first = if first == 0 { 4 } else { first };
        groups.push(whole[..first].parse::<u16>().expect("decimal group"));
        for chunk in whole.as_bytes()[first..].chunks_exact(4) {
            groups.push(
                std::str::from_utf8(chunk)
                    .expect("ASCII decimal group")
                    .parse::<u16>()
                    .expect("decimal group"),
            );
        }
        groups.len()
    };
    let mut fractional = fractional.to_string();
    while fractional.len() % 4 != 0 {
        fractional.push('0');
    }
    for chunk in fractional.as_bytes().chunks_exact(4) {
        groups.push(
            std::str::from_utf8(chunk)
                .expect("ASCII decimal group")
                .parse::<u16>()
                .expect("decimal group"),
        );
    }
    let mut weight = i16::try_from(whole_groups as i32 - 1).expect("Decimal128 numeric weight");
    while groups.first() == Some(&0) {
        groups.remove(0);
        weight = weight.checked_sub(1).expect("Decimal128 numeric weight");
    }
    while groups.last() == Some(&0) {
        groups.pop();
    }
    let mut out = Vec::with_capacity(8 + groups.len() * 2);
    out.extend_from_slice(
        &i16::try_from(groups.len())
            .expect("Decimal128 digit count")
            .to_be_bytes(),
    );
    out.extend_from_slice(&weight.to_be_bytes());
    out.extend_from_slice(&(if negative { NUMERIC_NEG } else { NUMERIC_POS }).to_be_bytes());
    out.extend_from_slice(&u16::from(dscale).to_be_bytes());
    for group in groups {
        out.extend_from_slice(&(group as i16).to_be_bytes());
    }
    out
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
            DbValue::Numeric(value) => Ok(Some(encode_binary_numeric(*value))),
            DbValue::Bool(value) => Ok(Some(vec![u8::from(*value)])),
            DbValue::Text(value) => Ok(Some(value.as_bytes().to_vec())),
            DbValue::Date(value) => Ok(Some(value.to_be_bytes().to_vec())),
            DbValue::Timestamp(value) => Ok(Some(value.to_be_bytes().to_vec())),
            DbValue::Uuid(value) => Ok(Some(value.to_vec())),
            DbValue::Null => unreachable!("NULL returns before binary encoding"),
        },
        _ => Err(PgValueCodecError::UnsupportedFormat(format)),
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

/// PostgreSQL's numeric typmod encoding, derived from neutral precision/scale metadata. Other
/// logical types return `-1` (no type modifier).
pub fn logical_type_typmod(ty: LogicalType, numeric_typmod: Option<(u8, u8)>) -> i32 {
    match (ty, numeric_typmod) {
        (LogicalType::Numeric, Some((precision, scale))) => {
            4 + ((i32::from(precision) << 16) | i32::from(scale))
        }
        _ => -1,
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
        ErrorCategory::UndefinedObject => "42704",
        ErrorCategory::PermissionDenied => "42501",
        ErrorCategory::DuplicateColumn => "42701",
        ErrorCategory::IndeterminateDatatype => "42P18",
        ErrorCategory::DatatypeMismatch => "42804",
        ErrorCategory::InvalidRequest => "08P01",
        ErrorCategory::ResourceExhausted => "53000",
        ErrorCategory::Cancelled => "57014",
        ErrorCategory::InFailedTransaction => "25P02",
        ErrorCategory::UniqueViolation => "23505",
        ErrorCategory::NotNullViolation => "23502",
        ErrorCategory::ForeignKeyViolation => "23503",
        ErrorCategory::CheckViolation => "23514",
        ErrorCategory::NumericValueOutOfRange => "22003",
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
            CommandTag::Truncate => "TRUNCATE TABLE".to_string(),
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
            CommandTag::Truncate => "TRUNCATE TABLE".to_string(),
            CommandTag::Copy => format!("COPY {}", rows_affected.unwrap_or(0)),
            CommandTag::Other(label) => label.clone(),
        },
        QueryOutcome::CopyIn { .. } => String::new(),
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
        CommandTag::Truncate => "TRUNCATE TABLE",
        CommandTag::Copy => "COPY",
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
        assert_eq!(logical_type_from_oid(1043), Ok(LogicalType::Text));
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
        for format in [0, 1] {
            assert_eq!(decode_parameter(25, format, None).unwrap(), DbValue::Null);
            assert_eq!(
                decode_parameter(25, format, Some("Grüße".as_bytes())).unwrap(),
                DbValue::Text("Grüße".to_string())
            );
            assert!(matches!(
                decode_parameter(25, format, Some(&[0xff])),
                Err(PgValueCodecError::InvalidValue {
                    logical_type: LogicalType::Text,
                    format: failed_format,
                }) if failed_format == format
            ));
            assert!(matches!(
                decode_parameter(25, format, Some(b"nul\0byte")),
                Err(PgValueCodecError::InvalidValue {
                    logical_type: LogicalType::Text,
                    format: failed_format,
                }) if failed_format == format
            ));
        }
        for oid in [16, 1082, 1114, 1700] {
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
        for invalid in [b"_1".as_slice(), b"1__0", b"0x_"] {
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
    fn integer_text_syntax_and_range_errors_remain_distinct() {
        for (oid, malformed) in [(21, b"12z".as_slice()), (23, b"0x7__fff"), (20, b"0b102")] {
            assert!(matches!(
                decode_parameter(oid, 0, Some(malformed)),
                Err(PgValueCodecError::InvalidValue {
                    logical_type,
                    format: 0,
                }) if logical_type_from_oid(oid) == Ok(logical_type)
            ));
        }
        for (oid, out_of_range) in [
            (21, b"0x8_000".as_slice()),
            (23, b"0o2_000_000_0000"),
            (20, b"0x8000_0000_0000_0000"),
        ] {
            assert!(matches!(
                decode_parameter(oid, 0, Some(out_of_range)),
                Err(PgValueCodecError::NumericValueOutOfRange {
                    logical_type,
                    format: 0,
                }) if logical_type_from_oid(oid) == Ok(logical_type)
            ));
        }
    }

    #[test]
    fn maps_neutral_error_categories_to_sqlstate() {
        assert_eq!(error_sqlstate(ErrorCategory::Syntax), "42601");
        assert_eq!(error_sqlstate(ErrorCategory::Unsupported), "0A000");
        assert_eq!(error_sqlstate(ErrorCategory::UndefinedObject), "42704");
        assert_eq!(error_sqlstate(ErrorCategory::PermissionDenied), "42501");
        assert_eq!(error_sqlstate(ErrorCategory::DuplicateColumn), "42701");
        assert_eq!(error_sqlstate(ErrorCategory::Cancelled), "57014");
        assert_eq!(error_sqlstate(ErrorCategory::UniqueViolation), "23505");
        assert_eq!(error_sqlstate(ErrorCategory::NotNullViolation), "23502");
        assert_eq!(error_sqlstate(ErrorCategory::ForeignKeyViolation), "23503");
        assert_eq!(error_sqlstate(ErrorCategory::CheckViolation), "23514");
        assert_eq!(
            error_sqlstate(ErrorCategory::NumericValueOutOfRange),
            "22003"
        );
        assert_eq!(error_sqlstate(ErrorCategory::Engine), "XX000");
    }

    #[test]
    fn all_logical_types_have_postgres_text_and_binary_golden_vectors() {
        let uuid = gpu_db_sql::uuid::parse_uuid("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let numeric = Decimal128::new(123_456_700, 4);
        let vectors = [
            (
                21,
                DbValue::Int2(-7),
                b"-7".to_vec(),
                (-7_i16).to_be_bytes().to_vec(),
            ),
            (
                23,
                DbValue::Int4(42),
                b"42".to_vec(),
                42_i32.to_be_bytes().to_vec(),
            ),
            (
                20,
                DbValue::Int8(-9),
                b"-9".to_vec(),
                (-9_i64).to_be_bytes().to_vec(),
            ),
            (
                1700,
                DbValue::Numeric(numeric),
                b"12345.6700".to_vec(),
                vec![
                    0, 3, 0, 1, 0, 0, 0, 4, // ndigits, weight, sign, dscale
                    0, 1, 0x09, 0x29, 0x1a, 0x2c,
                ],
            ),
            (16, DbValue::Bool(true), b"t".to_vec(), vec![1]),
            (
                25,
                DbValue::Text("Grüße".to_string()),
                "Grüße".as_bytes().to_vec(),
                "Grüße".as_bytes().to_vec(),
            ),
            (
                1082,
                DbValue::Date(-1),
                b"1999-12-31".to_vec(),
                (-1_i32).to_be_bytes().to_vec(),
            ),
            (
                1114,
                DbValue::Timestamp(1_234_567),
                b"2000-01-01 00:00:01.234567".to_vec(),
                1_234_567_i64.to_be_bytes().to_vec(),
            ),
            (
                2950,
                DbValue::Uuid(uuid),
                b"550e8400-e29b-41d4-a716-446655440000".to_vec(),
                uuid.to_vec(),
            ),
        ];
        for (oid, value, text, binary) in vectors {
            assert_eq!(encode_result_value(&value, 0).unwrap(), Some(text.clone()));
            assert_eq!(
                encode_result_value(&value, 1).unwrap(),
                Some(binary.clone())
            );
            assert_eq!(decode_parameter(oid, 0, Some(&text)).unwrap(), value);
            assert_eq!(decode_parameter(oid, 1, Some(&binary)).unwrap(), value);
            assert_eq!(decode_parameter(oid, 0, None).unwrap(), DbValue::Null);
            assert_eq!(decode_parameter(oid, 1, None).unwrap(), DbValue::Null);
        }
    }

    #[test]
    fn numeric_codec_preserves_scale_and_rejects_malformed_special_and_out_of_range_forms() {
        let tiny = Decimal128::new(-1, 8);
        let tiny_binary = vec![0, 1, 0xff, 0xfe, 0x40, 0, 0, 8, 0, 1];
        assert_eq!(
            encode_result_value(&DbValue::Numeric(tiny), 1).unwrap(),
            Some(tiny_binary.clone())
        );
        assert_eq!(
            decode_parameter(1700, 1, Some(&tiny_binary)).unwrap(),
            DbValue::Numeric(tiny)
        );
        assert_eq!(
            decode_parameter(1700, 0, Some(b" +1.20e-2 ")).unwrap(),
            DbValue::Numeric(Decimal128::new(120, 4))
        );
        assert_eq!(
            decode_parameter(1700, 1, Some(&[0, 0, 0, 0, 0, 0, 0, 3])).unwrap(),
            DbValue::Numeric(Decimal128::new(0, 3))
        );
        assert_eq!(
            decode_parameter(1700, 1, Some(&[0, 0, 0, 0, 0x40, 0, 0, 3])).unwrap(),
            DbValue::Numeric(Decimal128::new(0, 3)),
            "PG numeric receive normalizes a finite negative zero"
        );
        assert_eq!(
            decode_parameter(1700, 1, Some(&[0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 0x09, 0x29])).unwrap(),
            DbValue::Numeric(Decimal128::new(123, 2)),
            "PG16 numeric_recv truncates physical digits hidden by dscale"
        );
        for value in [b"NaN".as_slice(), b"Infinity", b"1e", b"1..0"] {
            assert!(matches!(
                decode_parameter(1700, 0, Some(value)),
                Err(PgValueCodecError::InvalidValue { format: 0, .. })
            ));
        }
        for value in [
            b"170141183460469231731687303715884105728".as_slice(),
            b"1e256",
        ] {
            assert!(matches!(
                decode_parameter(1700, 0, Some(value)),
                Err(PgValueCodecError::NumericValueOutOfRange { format: 0, .. })
            ));
        }
        let large_leading_zero = format!("{}1", "0".repeat(32 * 1024));
        assert_eq!(
            decode_parameter(1700, 0, Some(large_leading_zero.as_bytes())).unwrap(),
            DbValue::Numeric(Decimal128::new(1, 0)),
            "leading-zero input stays bounded without a second coefficient allocation"
        );
        for value in [
            vec![0, 1, 0, 0, 0xc0, 0, 0, 0, 0, 1], // NaN is not a finite Decimal128
            vec![0, 1, 0, 0, 0, 0, 0, 0, 0x27, 0x10], // base-10000 digit 10000
            vec![0, 1, 0, 0, 0, 0, 0, 0],          // declared digit missing
        ] {
            assert!(matches!(
                decode_parameter(1700, 1, Some(&value)),
                Err(PgValueCodecError::InvalidValue { format: 1, .. })
            ));
        }
        for value in [
            vec![0, 0, 0, 0, 0, 0, 1, 0], // dscale 256 cannot fit Decimal128's u8 scale
            vec![0, 1, 0, 10, 0, 0, 0, 0, 0, 1], // 10^40 is outside i128
        ] {
            assert!(matches!(
                decode_parameter(1700, 1, Some(&value)),
                Err(PgValueCodecError::NumericValueOutOfRange { format: 1, .. })
            ));
        }
    }

    #[test]
    fn numeric_codec_matches_pg16_finite_text_and_binary_edges() {
        for (text, expected) in [
            (b"0x2a".as_slice(), Decimal128::new(42, 0)),
            (b"-0o7_7".as_slice(), Decimal128::new(-63, 0)),
            (b"+0b1_0_1".as_slice(), Decimal128::new(5, 0)),
            // `numeric_in` also permits the first non-decimal separator immediately after its
            // radix prefix, as long as it precedes a valid digit.
            (b"0x_2a".as_slice(), Decimal128::new(42, 0)),
            (b"1_500.25_00".as_slice(), Decimal128::new(15_002_500, 4)),
            (b"1e1_0".as_slice(), Decimal128::new(10_000_000_000, 0)),
        ] {
            assert_eq!(
                decode_parameter(1700, 0, Some(text)).unwrap(),
                DbValue::Numeric(expected),
                "PG16 finite numeric text {text:?}"
            );
        }
        for text in [
            b"0x".as_slice(),
            b"0x__2",
            b"0b102",
            b"1__500",
            b"1._0",
            b"1e_10",
            b"1e1__0",
        ] {
            assert!(matches!(
                decode_parameter(1700, 0, Some(text)),
                Err(PgValueCodecError::InvalidValue { format: 0, .. })
            ));
        }
        assert!(matches!(
            decode_parameter(1700, 0, Some(b"0x8000_0000_0000_0000_0000_0000_0000_0000")),
            Err(PgValueCodecError::NumericValueOutOfRange { format: 0, .. })
        ));
        assert_eq!(
            decode_parameter(1700, 0, Some(b"-0x8000_0000_0000_0000_0000_0000_0000_0000")).unwrap(),
            DbValue::Numeric(Decimal128::new(i128::MIN, 0))
        );

        // A legal leading zero group must be stripped before the Decimal128 finite-envelope
        // check: the effective wire value is 10^36, not an out-of-range 10^40.
        assert_eq!(
            decode_parameter(1700, 1, Some(&[0, 2, 0, 10, 0, 0, 0, 0, 0, 0, 0, 1]),).unwrap(),
            DbValue::Numeric(Decimal128::new(10_i128.pow(36), 0))
        );
        for value in [Decimal128::new(i128::MIN, 0), Decimal128::new(i128::MIN, 1)] {
            let wire = encode_result_value(&DbValue::Numeric(value), 1)
                .unwrap()
                .unwrap();
            assert_eq!(
                decode_parameter(1700, 1, Some(&wire)).unwrap(),
                DbValue::Numeric(value),
                "NUMERIC binary round-trips the negative Decimal128 boundary at scale {}",
                value.scale
            );
        }
        assert!(matches!(
            decode_parameter(1700, 1, Some(&[0, 0, 0, 0, 0, 0, 0x40, 0])),
            Err(PgValueCodecError::InvalidValue { format: 1, .. })
        ));
        assert!(matches!(
            decode_parameter(1700, 1, Some(&[0, 0, 0, 0, 0, 0, 1, 0])),
            Err(PgValueCodecError::NumericValueOutOfRange { format: 1, .. })
        ));
    }

    #[test]
    fn boolean_text_prefixes_binary_nonzero_and_numeric_typmod_are_exact() {
        for text in [b"t".as_slice(), b"TRUE", b"ye", b"ON", b"1"] {
            assert_eq!(
                decode_parameter(16, 0, Some(text)).unwrap(),
                DbValue::Bool(true)
            );
        }
        for text in [b"f".as_slice(), b"FALSE", b"no", b"of", b"0"] {
            assert_eq!(
                decode_parameter(16, 0, Some(text)).unwrap(),
                DbValue::Bool(false)
            );
        }
        assert!(matches!(
            decode_parameter(16, 0, Some(b"o")),
            Err(PgValueCodecError::InvalidValue { .. })
        ));
        assert_eq!(
            decode_parameter(16, 1, Some(&[0])).unwrap(),
            DbValue::Bool(false)
        );
        assert_eq!(
            decode_parameter(16, 1, Some(&[2])).unwrap(),
            DbValue::Bool(true)
        );
        assert!(matches!(
            decode_parameter(16, 1, Some(&[])),
            Err(PgValueCodecError::InvalidValue { .. })
        ));
        assert_eq!(
            logical_type_typmod(LogicalType::Numeric, Some((12, 4))),
            786_440
        );
        assert_eq!(logical_type_typmod(LogicalType::Int4, None), -1);
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
