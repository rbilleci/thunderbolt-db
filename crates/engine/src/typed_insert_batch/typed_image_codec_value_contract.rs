//! Shared typed-vector value and allocation contract for the inert v2 image facade.
//!
//! Keeping this leaf separate makes value representation, SQL carrier bounds, and exact decoded
//! owner accounting one authority for both retained responses and final-table images.

use super::typed_image_codec::image_error;
use super::*;

pub(super) fn decoded_column_owned_bytes(
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
) -> Result<u64, EngineError> {
    let validity: u64 = match validity {
        TypedInsertColumnValidity::AllValid => 0,
        TypedInsertColumnValidity::Bitmap(words) => u64::try_from(words.len())
            .map_err(|_| image_error("validity allocation addressability"))?
            .checked_mul(4)
            .ok_or_else(|| image_error("validity allocation overflow"))?,
    };
    let values = match values {
        TypedInsertColumnValues::I32(values) => u64::try_from(values.len())
            .map_err(|_| image_error("i32 allocation addressability"))?
            .checked_mul(4),
        TypedInsertColumnValues::I64(values) => u64::try_from(values.len())
            .map_err(|_| image_error("i64 allocation addressability"))?
            .checked_mul(8),
        TypedInsertColumnValues::I128(values) => u64::try_from(values.len())
            .map_err(|_| image_error("numeric allocation addressability"))?
            .checked_mul(16),
        TypedInsertColumnValues::Bytes16(values) => u64::try_from(values.len())
            .map_err(|_| image_error("uuid allocation addressability"))?
            .checked_mul(16),
        TypedInsertColumnValues::BoolBits(words) => u64::try_from(words.len())
            .map_err(|_| image_error("bool allocation addressability"))?
            .checked_mul(4),
        TypedInsertColumnValues::Text { offsets, bytes } => u64::try_from(offsets.len())
            .map_err(|_| image_error("text offset allocation addressability"))?
            .checked_mul(8)
            .and_then(|value| value.checked_add(u64::try_from(bytes.len()).ok()?)),
    }
    .ok_or_else(|| image_error("decoded value allocation overflow"))?;
    validity
        .checked_add(values)
        .ok_or_else(|| image_error("decoded column allocation overflow"))
}

pub(super) fn decoded_column_allocation_slots(
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
) -> Option<u64> {
    let validity: u64 = match validity {
        TypedInsertColumnValidity::AllValid => 0,
        TypedInsertColumnValidity::Bitmap(words) if words.is_empty() => 0,
        TypedInsertColumnValidity::Bitmap(_) => 1,
    };
    let values = match values {
        TypedInsertColumnValues::I32(values) => u64::from(!values.is_empty()),
        TypedInsertColumnValues::I64(values) => u64::from(!values.is_empty()),
        TypedInsertColumnValues::I128(values) => u64::from(!values.is_empty()),
        TypedInsertColumnValues::Bytes16(values) => u64::from(!values.is_empty()),
        TypedInsertColumnValues::BoolBits(words) => u64::from(!words.is_empty()),
        TypedInsertColumnValues::Text { offsets, bytes } => {
            u64::from(!offsets.is_empty()).checked_add(u64::from(!bytes.is_empty()))?
        }
    };
    validity.checked_add(values)
}

pub(super) fn i32_values_are_valid(values: &[i32], ty: SqlType) -> bool {
    values.iter().copied().all(|value| {
        (ty != SqlType::Int2 || i16::try_from(value).is_ok())
            && (ty != SqlType::Date || gpu_db_sql::datetime::validate_date_carrier(value).is_ok())
    })
}

pub(super) fn i64_values_are_valid(values: &[i64], ty: SqlType) -> bool {
    ty != SqlType::Timestamp
        || values
            .iter()
            .copied()
            .all(|value| gpu_db_sql::datetime::validate_timestamp_carrier(value).is_ok())
}

pub(super) fn numeric_values_are_valid(values: &[i128], ty: SqlType) -> bool {
    let SqlType::Numeric { precision, .. } = ty else {
        return false;
    };
    values
        .iter()
        .copied()
        .all(|value| !crate::numeric_exceeds_precision(value, precision))
}

pub(super) fn text_shape_is_exact(offsets: &[u64], bytes: &[u8], rows: usize) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    offsets.len() == rows.saturating_add(1)
        && offsets.first() == Some(&0)
        && offsets.windows(2).all(|pair| {
            pair[0] <= pair[1]
                && usize::try_from(pair[1])
                    .ok()
                    .is_some_and(|offset| text.is_char_boundary(offset))
        })
        && offsets.last().and_then(|end| usize::try_from(*end).ok()) == Some(bytes.len())
}

pub(super) fn sql_type_bytes(ty: SqlType) -> Result<[u8; 4], EngineError> {
    let (tag, precision, scale) = match ty {
        SqlType::Int2 => (1, 0, 0),
        SqlType::Int4 => (2, 0, 0),
        SqlType::Int8 => (3, 0, 0),
        SqlType::Numeric { precision, scale } => (4, precision, scale),
        SqlType::Bool => (5, 0, 0),
        SqlType::Text => (6, 0, 0),
        SqlType::Date => (7, 0, 0),
        SqlType::Timestamp => (8, 0, 0),
        SqlType::Uuid => (9, 0, 0),
    };
    if tag == 4 && (precision == 0 || precision > 38 || scale > precision) {
        return Err(image_error(
            "numeric typmod is outside the canonical domain",
        ));
    }
    Ok([tag, precision, scale, 0])
}

pub(super) fn sql_type_from_bytes(raw: &[u8]) -> Result<SqlType, EngineError> {
    let [tag, precision, scale, reserved]: [u8; 4] = raw
        .try_into()
        .map_err(|_| image_error("SQL type bytes are truncated"))?;
    match (tag, precision, scale, reserved) {
        (1, 0, 0, 0) => Ok(SqlType::Int2),
        (2, 0, 0, 0) => Ok(SqlType::Int4),
        (3, 0, 0, 0) => Ok(SqlType::Int8),
        (4, precision @ 1..=38, scale, 0) if scale <= precision => {
            Ok(SqlType::Numeric { precision, scale })
        }
        (5, 0, 0, 0) => Ok(SqlType::Bool),
        (6, 0, 0, 0) => Ok(SqlType::Text),
        (7, 0, 0, 0) => Ok(SqlType::Date),
        (8, 0, 0, 0) => Ok(SqlType::Timestamp),
        (9, 0, 0, 0) => Ok(SqlType::Uuid),
        _ => Err(image_error("SQL type tag/typmod is noncanonical")),
    }
}
