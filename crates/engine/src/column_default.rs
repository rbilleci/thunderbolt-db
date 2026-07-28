//! Durable scalar column-default binding, evaluation, and catalog rendering.
//!
//! Defaults are SQL control-plane expressions.  They deliberately resolve before a
//! typed INSERT/WAL/device plan, while sequence defaults stay an explicit per-row
//! stateful special case in the sequence owner.

use super::*;

const MICROS_PER_DAY: i64 = 86_400_000_000;

#[cfg(test)]
fn scalar_default_evaluation_counts() -> &'static std::sync::Mutex<BTreeMap<String, usize>> {
    static COUNTS: std::sync::OnceLock<std::sync::Mutex<BTreeMap<String, usize>>> =
        std::sync::OnceLock::new();
    COUNTS.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()))
}

/// A narrowly scoped deterministic test seam for scalar-default lowering.  Counts are keyed by
/// column name so independent parallel tests do not affect a caller's assertion.
#[cfg(test)]
pub(crate) fn reset_scalar_default_evaluation_count(column_name: &str) {
    scalar_default_evaluation_counts()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(column_name.to_string(), 0);
}

#[cfg(test)]
pub(crate) fn scalar_default_evaluation_count(column_name: &str) -> usize {
    *scalar_default_evaluation_counts()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(column_name)
        .unwrap_or(&0)
}

#[cfg(test)]
fn record_scalar_default_evaluation(column_name: &str) {
    let mut counts = scalar_default_evaluation_counts()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *counts.entry(column_name.to_string()).or_default() += 1;
}

/// Bind a parsed/programmatic default to one column without evaluating its scalar value.
///
/// `Unknown` is a transient ALTER-parser form.  Catalog publication never retains it:
/// it is parsed at the target here, but target numeric typmod/range validation stays
/// deferred to [`evaluate_scalar`].
pub(crate) fn bind_to_column(
    default: ColumnDefault,
    target: SqlType,
    column_name: &str,
) -> Result<ColumnDefault, EngineError> {
    match default {
        ColumnDefault::Literal(value) => Ok(ColumnDefault::Literal(coerce_insert_value(
            value,
            target,
            column_name,
        )?)),
        ColumnDefault::DeferredScalar { value, input } => {
            let (value, input) = match input {
                DefaultInputType::Unknown => (
                    parse_unknown_at_target(value, target, column_name)?,
                    DefaultInputType::TargetTyped,
                ),
                DefaultInputType::Inferred(source) | DefaultInputType::Explicit(source) => {
                    if !assignment_supported(source, target) {
                        return Err(invalid_default_assignment(column_name));
                    }
                    (value, input)
                }
                DefaultInputType::TargetTyped => (value, input),
            };
            Ok(ColumnDefault::DeferredScalar { value, input })
        }
        ColumnDefault::SequenceNextVal { .. } if target == SqlType::Int4 => Ok(default),
        ColumnDefault::SequenceNextVal { .. } => Err(invalid_default_assignment(column_name)),
    }
}

/// Evaluate a scalar default once at its defined SQL error boundary.
///
/// Source casts and source numeric typmods run before assignment conversion; target
/// coercion/typmod then runs last.  This is intentionally state-free so TypedInsertBatch
/// can compute one broadcast value before WAL and the DDL owner can preflight ADD COLUMN.
pub(crate) fn evaluate_scalar(
    default: &ColumnDefault,
    target: SqlType,
    column_name: &str,
) -> Result<SqlValue, EngineError> {
    #[cfg(test)]
    record_scalar_default_evaluation(column_name);

    match default {
        // Historical/programmatic compatibility: legacy literals have always represented
        // already-bound catalog values, so do not reinterpret their source type here.
        ColumnDefault::Literal(value) => Ok(value.clone()),
        ColumnDefault::DeferredScalar { value, input } => {
            let value = match input {
                DefaultInputType::TargetTyped => value.clone(),
                DefaultInputType::Inferred(source) | DefaultInputType::Explicit(source) => {
                    let source_value = coerce_insert_value(value.clone(), *source, column_name)?;
                    assignment_convert(source_value, *source, target, column_name)?
                }
                DefaultInputType::Unknown => {
                    return Err(EngineError::ApplyFailed(format!(
                        "column default for \"{column_name}\" was published without target binding"
                    )));
                }
            };
            coerce_insert_value(value, target, column_name)
        }
        ColumnDefault::SequenceNextVal { .. } => Err(EngineError::ApplyFailed(
            "stateful sequence default requires the sequence owner".to_string(),
        )),
    }
}

/// Stable pg_catalog/pg_dump rendering for every durable default form.
pub(crate) fn render_expression(default: &ColumnDefault) -> Result<String, EngineError> {
    match default {
        ColumnDefault::Literal(value) => pg16_column_default_expression(value),
        ColumnDefault::DeferredScalar { value, input } => match input {
            DefaultInputType::Explicit(ty) => Ok(format!(
                "{}::{}",
                render_sql_value_literal(value)?,
                type_sql(*ty)
            )),
            DefaultInputType::Unknown
            | DefaultInputType::TargetTyped
            | DefaultInputType::Inferred(_) => pg16_column_default_expression(value),
        },
        ColumnDefault::SequenceNextVal { sequence, .. } => Ok(format!(
            "nextval('{}'::regclass)",
            sequence.replace('\'', "''")
        )),
    }
}

pub(crate) const fn is_sequence(default: &ColumnDefault) -> bool {
    matches!(default, ColumnDefault::SequenceNextVal { .. })
}

fn parse_unknown_at_target(
    value: SqlValue,
    target: SqlType,
    column_name: &str,
) -> Result<SqlValue, EngineError> {
    match value {
        SqlValue::Null => Ok(SqlValue::Null),
        SqlValue::Text(text) => (match target {
            SqlType::Numeric { .. } => {
                gpu_db_sql::parse_default_numeric_literal(&text, target).map(SqlValue::Numeric)
            }
            _ => gpu_db_sql::parse_typed_value_from_str(&text, target),
        })
        .map_err(|error| default_target_parse_error(error, column_name)),
        SqlValue::Parameter { .. } => Err(EngineError::ApplyFailed(
            "column default retains an unbound parameter".to_string(),
        )),
        _ => Err(EngineError::ApplyFailed(format!(
            "invalid unknown column default for \"{column_name}\""
        ))),
    }
}

/// Preserve scalar input diagnostics while binding an ALTER DEFAULT's unknown literal at the
/// target type.  Other parser failures would violate this binder's invariant, so keep them as
/// generic engine errors rather than inventing a protocol-facing category.
fn default_target_parse_error(error: ParseError, column_name: &str) -> EngineError {
    match error {
        error @ ParseError::InvalidTextRepresentation { .. } => {
            EngineError::InvalidTextRepresentation(error.to_string())
        }
        error @ ParseError::InvalidDatetimeFormat { .. } => {
            EngineError::InvalidDatetimeFormat(error.to_string())
        }
        error @ ParseError::DatetimeFieldOverflow { .. } => {
            EngineError::DatetimeFieldOverflow(error.to_string())
        }
        error @ ParseError::NumericValueOutOfRange { .. } => {
            EngineError::NumericValueOutOfRange(error.to_string())
        }
        error => EngineError::ApplyFailed(format!(
            "invalid target-directed default for column \"{column_name}\": {error}"
        )),
    }
}

fn assignment_supported(source: SqlType, target: SqlType) -> bool {
    source == target
        || matches!(
            (source, target),
            (
                SqlType::Int2 | SqlType::Int4 | SqlType::Int8,
                SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. }
            ) | (
                SqlType::Numeric { .. },
                SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. }
            ) | (SqlType::Date, SqlType::Timestamp)
                | (SqlType::Timestamp, SqlType::Date)
                | (_, SqlType::Text)
        )
}

fn assignment_convert(
    value: SqlValue,
    source: SqlType,
    target: SqlType,
    column_name: &str,
) -> Result<SqlValue, EngineError> {
    if matches!(value, SqlValue::Null) || source == target {
        return Ok(value);
    }
    if target == SqlType::Text {
        return scalar_to_text(value, column_name);
    }
    match (source, target, value) {
        (
            SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. },
            SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. },
            value,
        ) => convert_numeric_family(value, target),
        (SqlType::Date, SqlType::Timestamp, SqlValue::Date(days)) => i64::from(days)
            .checked_mul(MICROS_PER_DAY)
            .map(SqlValue::Timestamp)
            .ok_or_else(|| {
                EngineError::NumericValueOutOfRange("timestamp out of range".to_string())
            }),
        (SqlType::Timestamp, SqlType::Date, SqlValue::Timestamp(micros)) => {
            let days = micros.div_euclid(MICROS_PER_DAY);
            i32::try_from(days)
                .map(SqlValue::Date)
                .map_err(|_| EngineError::NumericValueOutOfRange("date out of range".to_string()))
        }
        _ => Err(invalid_default_assignment(column_name)),
    }
}

fn convert_numeric_family(value: SqlValue, target: SqlType) -> Result<SqlValue, EngineError> {
    let number = match value {
        SqlValue::Int2(value) => Decimal128::new(i128::from(value), 0),
        SqlValue::Int4(value) => Decimal128::new(i128::from(value), 0),
        SqlValue::Int8(value) => Decimal128::new(i128::from(value), 0),
        SqlValue::Numeric(value) => value,
        _ => unreachable!("numeric source type and value were source-validated"),
    };
    match target {
        SqlType::Numeric { .. } => Ok(SqlValue::Numeric(number)),
        SqlType::Int2 => number
            .rescale(0)
            .map_err(|_| numeric_range("smallint"))
            .and_then(|value| {
                i16::try_from(value.mantissa)
                    .map(SqlValue::Int2)
                    .map_err(|_| numeric_range("smallint"))
            }),
        SqlType::Int4 => number
            .rescale(0)
            .map_err(|_| numeric_range("integer"))
            .and_then(|value| {
                i32::try_from(value.mantissa)
                    .map(SqlValue::Int4)
                    .map_err(|_| numeric_range("integer"))
            }),
        SqlType::Int8 => number
            .rescale(0)
            .map_err(|_| numeric_range("bigint"))
            .and_then(|value| {
                i64::try_from(value.mantissa)
                    .map(SqlValue::Int8)
                    .map_err(|_| numeric_range("bigint"))
            }),
        _ => unreachable!("numeric assignment target was checked"),
    }
}

fn scalar_to_text(value: SqlValue, column_name: &str) -> Result<SqlValue, EngineError> {
    Ok(SqlValue::Text(match value {
        SqlValue::Int2(value) => value.to_string(),
        SqlValue::Int4(value) => value.to_string(),
        SqlValue::Int8(value) => value.to_string(),
        SqlValue::Numeric(value) => value.to_decimal_string(),
        SqlValue::Bool(value) => value.to_string(),
        SqlValue::Text(value) => value,
        SqlValue::Date(value) => gpu_db_sql::datetime::format_date(value),
        SqlValue::Timestamp(value) => gpu_db_sql::datetime::format_timestamp(value),
        SqlValue::Uuid(value) => gpu_db_sql::uuid::format_uuid(&value),
        SqlValue::Null => return Ok(SqlValue::Null),
        SqlValue::Parameter { .. } => return Err(invalid_default_assignment(column_name)),
    }))
}

fn numeric_range(type_name: &str) -> EngineError {
    EngineError::NumericValueOutOfRange(format!("{type_name} out of range"))
}

fn invalid_default_assignment(column_name: &str) -> EngineError {
    EngineError::DatatypeMismatch(format!(
        "column \"{column_name}\" has an incompatible default"
    ))
}

fn type_sql(ty: SqlType) -> String {
    match ty {
        SqlType::Int2 => "smallint".to_string(),
        SqlType::Int4 => "integer".to_string(),
        SqlType::Int8 => "bigint".to_string(),
        SqlType::Numeric { precision, scale } => format!("numeric({precision},{scale})"),
        SqlType::Bool => "boolean".to_string(),
        SqlType::Text => "text".to_string(),
        SqlType::Date => "date".to_string(),
        SqlType::Timestamp => "timestamp".to_string(),
        SqlType::Uuid => "uuid".to_string(),
    }
}
