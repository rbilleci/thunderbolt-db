//! PostgreSQL wire adapter mappings over the neutral façade types.
//!
//! This module is the **only** place PostgreSQL-specific wire concepts —
//! type OIDs, `SQLSTATE` codes, and the `INSERT 0 N` / `SELECT N` completion-tag
//! grammar — live. A MySQL or HTTP adapter would provide its own equivalent
//! module; the engine and the façade stay free of all of it.

use crate::{CommandTag, DbValue, ErrorCategory, LogicalType, QueryOutcome};

/// PostgreSQL type OID for a neutral logical type.
pub fn logical_type_oid(ty: LogicalType) -> u32 {
    match ty {
        LogicalType::Int4 => 23,
        LogicalType::Int8 => 20,
        LogicalType::Numeric => 1700,
        LogicalType::Bool => 16,
        LogicalType::Text => 25,
        LogicalType::Date => 1082,
    }
}

/// PostgreSQL wire type size (negative for variable-length) for a logical type.
pub fn logical_type_size(ty: LogicalType) -> i16 {
    match ty {
        LogicalType::Int4 => 4,
        LogicalType::Int8 => 8,
        LogicalType::Numeric => -1,
        LogicalType::Bool => 1,
        LogicalType::Text => -1,
        LogicalType::Date => 4,
    }
}

/// Text-format wire encoding of a neutral value. A `Numeric` renders via its
/// fixed-point decimal string (with the decimal point); a `Bool` renders as the
/// PostgreSQL `t`/`f` text form.
pub fn db_value_text(value: &DbValue) -> String {
    match value {
        DbValue::Int4(value) => value.to_string(),
        DbValue::Int8(value) => value.to_string(),
        DbValue::Numeric(value) => value.to_decimal_string(),
        DbValue::Bool(value) => if *value { "t" } else { "f" }.to_string(),
        DbValue::Text(value) => value.clone(),
        DbValue::Date(value) => gpu_db_sql::datetime::format_date(*value),
    }
}

/// `SQLSTATE` for a neutral error category.
pub fn error_sqlstate(category: ErrorCategory) -> &'static str {
    match category {
        ErrorCategory::Syntax => "42601",
        ErrorCategory::Unsupported => "0A000",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_neutral_types_to_postgres_oids() {
        assert_eq!(logical_type_oid(LogicalType::Int4), 23);
        assert_eq!(logical_type_oid(LogicalType::Int8), 20);
        assert_eq!(logical_type_oid(LogicalType::Numeric), 1700);
        assert_eq!(logical_type_oid(LogicalType::Text), 25);
    }

    #[test]
    fn maps_neutral_error_categories_to_sqlstate() {
        assert_eq!(error_sqlstate(ErrorCategory::Syntax), "42601");
        assert_eq!(error_sqlstate(ErrorCategory::Unsupported), "0A000");
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
