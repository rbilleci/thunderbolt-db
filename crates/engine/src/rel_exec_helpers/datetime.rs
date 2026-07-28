//! Date/timestamp text coercion shared by ordinary DML and CHECK binding.
//!
//! Keeping the parser-to-engine error mapping beside the relational coercion boundary prevents a
//! CHECK-only SQLSTATE distinction: every text value assigned to a temporal column preserves
//! malformed-input (`22007`) versus numeric-field/range (`22008`) semantics.

use super::*;
use gpu_db_sql::datetime::{parse_date_detailed, parse_timestamp_detailed, DatetimeParseError};

pub(crate) fn coerce_datetime_text(text: &str, ty: SqlType) -> Result<SqlValue, EngineError> {
    match ty {
        SqlType::Date => parse_date_detailed(text)
            .map(SqlValue::Date)
            .map_err(|error| datetime_parse_engine_error(text, ty, error)),
        SqlType::Timestamp => parse_timestamp_detailed(text)
            .map(SqlValue::Timestamp)
            .map_err(|error| datetime_parse_engine_error(text, ty, error)),
        _ => unreachable!("temporal coercion requires a date or timestamp target"),
    }
}

/// Reject an already-typed temporal carrier outside PostgreSQL's finite domain. Binary Bind and
/// in-process prepared routes can construct `SqlValue::Date`/`Timestamp` directly, so this must
/// run independently of text parsing before an INSERT plan reaches device residency.
pub(crate) fn validate_datetime_carrier(value: &SqlValue, ty: SqlType) -> Result<(), EngineError> {
    let invalid = |type_name: &str| {
        EngineError::DatetimeFieldOverflow(format!("{type_name} field value out of range"))
    };
    match (value, ty) {
        (SqlValue::Date(days), SqlType::Date) => gpu_db_sql::datetime::validate_date_carrier(*days)
            .map(|_| ())
            .map_err(|_| invalid("date")),
        (SqlValue::Timestamp(micros), SqlType::Timestamp) => {
            gpu_db_sql::datetime::validate_timestamp_carrier(*micros)
                .map(|_| ())
                .map_err(|_| invalid("timestamp"))
        }
        _ => Ok(()),
    }
}

fn datetime_parse_engine_error(text: &str, ty: SqlType, error: DatetimeParseError) -> EngineError {
    let message = format!(
        "{}: \"{text}\"",
        match ty {
            SqlType::Date => "invalid input syntax for type date",
            SqlType::Timestamp => "invalid input syntax for type timestamp",
            _ => unreachable!("temporal coercion requires a date or timestamp target"),
        }
    );
    match error {
        DatetimeParseError::MalformedFormat => EngineError::InvalidDatetimeFormat(message),
        DatetimeParseError::FieldOverflow => EngineError::DatetimeFieldOverflow(message),
    }
}
