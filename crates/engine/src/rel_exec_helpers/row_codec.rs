use super::NULL_TOKEN;
use crate::{ExecuteError, RelationalColumn};
use gpu_db_sql::{Decimal128, SqlType, SqlValue};
use gpu_db_types::EngineError;

pub(crate) fn encode_relational_row(values: &[SqlValue]) -> String {
    values
        .iter()
        .map(|value| match value {
            // Stored type-independently as the reserved prefix-free token; decodes back to
            // `SqlValue::Null` regardless of column type (see `decode_relational_value`).
            SqlValue::Null => NULL_TOKEN.to_string(),
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
            SqlValue::Parameter { .. } => {
                unreachable!("stored rows never contain unbound prepared parameters")
            }
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
/// `b:`bool `t:`text — chosen to mirror [`super::relational_index_value`]'s key vocabulary.
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
    // A stored NULL is the reserved prefix-free token (see `encode_relational_row`),
    // type-independent: it decodes to `SqlValue::Null` for any column type. Real text never
    // collides — text is stored with a `t:` prefix, so a bare `null` can only be a stored NULL.
    if part == NULL_TOKEN {
        return Ok(SqlValue::Null);
    }
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
