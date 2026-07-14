//! State-free operand/type recognition and literal normalization for resident predicates.

use crate::engine_expr_ir::ResidentExpr;
use crate::relational_model::RelationalTable;
use crate::ExecuteError;
use gpu_db_sql::{Decimal128, SqlType};
use gpu_db_types::EngineError;

pub(super) fn is_int4_literal(expr: &ResidentExpr) -> bool {
    matches!(expr, ResidentExpr::Int4Literal(_))
}

/// The column index if `expr` is a `Column` of int8 (`SqlType::Int8`) type, else `None` (the type
/// matrix, doc 19).
pub(super) fn int8_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Int8) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether any `Column` referenced anywhere in `expr` is int8 — used to reject an int8 shape the
/// simple-comparison path does not yet support (int8 arithmetic, mixed int4/int8) rather than
/// silently routing it to the int4 path.
pub(super) fn expr_mentions_int8(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Int8)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::Int8Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
        // IS NULL is a type-neutral validity test (it lowers to a bitmap mask), not a typed value
        // operand -- it never drives element-type routing.
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_int8(lhs, table) || expr_mentions_int8(rhs, table)
        }
    }
}

/// Whether any `Column` referenced anywhere in `expr` is int4 — used to reject a mixed int4/int8
/// expression. An `Int4Literal` is NOT an int4 column (it is coercible to int8), so it does not count.
pub(super) fn expr_mentions_int4_column(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Int4)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::Int8Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_int4_column(lhs, table) || expr_mentions_int4_column(rhs, table)
        }
    }
}

/// The column index if `expr` is a `Column` of numeric (`SqlType::Numeric`) type, else `None`.
pub(super) fn numeric_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if matches!(
                table.columns.get(*idx).map(|column| column.ty),
                Some(SqlType::Numeric { .. })
            ) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether `expr` mentions numeric anywhere — a numeric `Column` or a `NumericLiteral`. Marks a
/// predicate as numeric (the type matrix, doc 19); an `Int4Literal` is integer until it is coerced
/// against a numeric column.
pub(super) fn expr_mentions_numeric(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => matches!(
            table.columns.get(*idx).map(|column| column.ty),
            Some(SqlType::Numeric { .. })
        ),
        ResidentExpr::NumericLiteral(_) => true,
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::Int8Literal(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_numeric(lhs, table) || expr_mentions_numeric(rhs, table)
        }
    }
}

/// The column index if `expr` is a `Column` of text (`SqlType::Text`) type, else `None`.
pub(super) fn text_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Text) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// The literal bytes if `expr` is a `TextLiteral`, else `None`.
pub(super) fn text_literal_value(expr: &ResidentExpr) -> Option<&str> {
    match expr {
        ResidentExpr::TextLiteral(value) => Some(value.as_str()),
        _ => None,
    }
}

/// Whether `expr` mentions text anywhere -- a text `Column` or a `TextLiteral`. Marks a predicate as
/// text (the type matrix, doc 19).
pub(super) fn expr_mentions_text(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Text)
        }
        ResidentExpr::TextLiteral(_) => true,
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::Int8Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_text(lhs, table) || expr_mentions_text(rhs, table)
        }
    }
}

/// Whether `expr` mentions a BOOL column anywhere (a bool literal alone does not -- it only matters
/// paired with a bool column). Marks a predicate as touching the i32 bool-mask path.
pub(super) fn expr_mentions_bool_column(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Bool)
        }
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_bool_column(lhs, table) || expr_mentions_bool_column(rhs, table)
        }
        _ => false,
    }
}

/// The column index if `expr` is a `Column` of `date` type, else `None`.
pub(super) fn date_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Date) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether `expr` mentions a date COLUMN anywhere. A predicate is "date" when a date column is
/// involved; a bare string literal is the date VALUE, resolved against the column at lowering.
pub(super) fn expr_mentions_date(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Date)
        }
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_date(lhs, table) || expr_mentions_date(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::Int8Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
    }
}

/// The i32 day count of a date literal: a `TextLiteral` parsed as ISO `YYYY-MM-DD` (PG coerces an
/// unknown-type string literal to the column type). Anything else (an integer literal, a non-date
/// column) is a type error -- a date column compares only to a date literal or another date column.
pub(super) fn date_literal_days(expr: &ResidentExpr) -> Result<i32, ExecuteError> {
    match expr {
        ResidentExpr::TextLiteral(text) => {
            gpu_db_sql::datetime::parse_date(text).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "invalid input syntax for type date: \"{text}\""
                )))
            })
        }
        // NB (ADR-006 date compound): deliberately NOT accepting a raw-days `Int4Literal` here — the
        // READ path lowers a plain integer literal to `Int4Literal`, and `date = 5` must stay a HARD
        // ERROR (PG semantics), never silently treat 5 as days. The DML builder instead emits a
        // `TextLiteral(format_date(days))` (the canonical renderer; parse_date round-trips), exactly
        // like the uuid builder's `format_uuid` needle.
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a date column compares only to a date literal or another date column".to_string(),
        ))),
    }
}

/// The column index if `expr` is a `Column` of `timestamp` type, else `None`.
pub(super) fn timestamp_column_index(
    expr: &ResidentExpr,
    table: &RelationalTable,
) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Timestamp) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether `expr` mentions a timestamp COLUMN anywhere (the predicate is "timestamp" when a timestamp
/// column is involved; a bare string literal is the timestamp VALUE, resolved at lowering).
pub(super) fn expr_mentions_timestamp(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Timestamp)
        }
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_timestamp(lhs, table) || expr_mentions_timestamp(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::Int8Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
    }
}

/// The i64 microsecond count of a timestamp literal: a `TextLiteral` parsed as `YYYY-MM-DD HH:MM:SS`.
/// Anything else is a type error -- a timestamp column compares only to a timestamp literal or column.
pub(super) fn timestamp_literal_micros(expr: &ResidentExpr) -> Result<i64, ExecuteError> {
    match expr {
        ResidentExpr::TextLiteral(text) => {
            gpu_db_sql::datetime::parse_timestamp(text).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "invalid input syntax for type timestamp: \"{text}\""
                )))
            })
        }
        // ADR-006 (wider-type range DML): the DML predicate builder supplies an already-bound timestamp as
        // its raw i64 microseconds via `Int8Literal` (there is no host string to re-parse), so accept it
        // directly — the same i64 micros the text path produces after `parse_timestamp`.
        ResidentExpr::Int8Literal(micros) => Ok(*micros),
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a timestamp column compares only to a timestamp literal or another timestamp column"
                .to_string(),
        ))),
    }
}

/// The column index if `expr` is a `Column` of `uuid` type, else `None`.
pub(super) fn uuid_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Uuid) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether `expr` mentions a uuid COLUMN anywhere (the predicate is "uuid" when a uuid column is
/// involved; a bare string literal is the uuid VALUE, resolved at lowering).
pub(super) fn expr_mentions_uuid(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Uuid)
        }
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::Int8Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
    }
}

/// The 16 bytes of a uuid literal: a `TextLiteral` parsed as a uuid. Anything else is a type error --
/// a uuid column compares only to a uuid literal or another uuid column.
pub(super) fn uuid_literal_bytes(expr: &ResidentExpr) -> Result<[u8; 16], ExecuteError> {
    match expr {
        ResidentExpr::TextLiteral(text) => gpu_db_sql::uuid::parse_uuid(text).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "invalid input syntax for type uuid: \"{text}\""
            )))
        }),
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a uuid column compares only to a uuid literal or another uuid column".to_string(),
        ))),
    }
}

/// The column index if `expr` is a `Column` of `smallint` (int2) type, else `None`.
pub(super) fn int2_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
    match expr {
        ResidentExpr::Column(idx)
            if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Int2) =>
        {
            Some(*idx)
        }
        _ => None,
    }
}

/// Whether `expr` mentions a smallint COLUMN anywhere.
pub(super) fn expr_mentions_int2(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    match expr {
        ResidentExpr::Column(idx) => {
            table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Int2)
        }
        ResidentExpr::IsNull { .. } => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_int2(lhs, table) || expr_mentions_int2(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::Int8Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
    }
}

/// The i32 scalar a smallint compares against: an `Int4Literal` (PG promotes both sides to int4, so
/// an out-of-int16 literal is a valid comparison that simply matches no rows -- NOT a range error).
/// A non-integer literal is a type error -- a smallint compares only to an integer literal or column.
pub(super) fn int2_literal_value(expr: &ResidentExpr) -> Result<i32, ExecuteError> {
    match expr {
        ResidentExpr::Int4Literal(value) => Ok(*value),
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a smallint column compares only to an integer literal or another smallint column"
                .to_string(),
        ))),
    }
}

/// Compile a SQL `LIKE` pattern to the device kernel's u32 token array: one token per output position,
/// `(op << 8) | literal_byte`, op 0 = literal byte, 1 = any-one (`_`), 2 = any-run (`%`). The default
/// `\` escape is resolved here (`\%` / `\_` / `\\` -> a literal byte); a lone trailing `\` is an error
/// (PG: "LIKE pattern must not end with escape character"). A multi-byte UTF-8 literal character
/// becomes one literal token per byte (matched byte-by-byte against the same bytes in the text).
pub(super) fn compile_like_pattern(pattern: &str) -> Result<Vec<u32>, ExecuteError> {
    let bytes = pattern.as_bytes();
    let mut tokens = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "LIKE pattern must not end with escape character".to_string(),
                    )));
                }
                tokens.push(u32::from(bytes[i])); // op 0 (literal), the escaped byte
            }
            b'%' => tokens.push(2u32 << 8),
            b'_' => tokens.push(1u32 << 8),
            other => tokens.push(u32::from(other)),
        }
        i += 1;
    }
    Ok(tokens)
}

/// The numeric value of a literal operand: a `NumericLiteral` as-is, or an `Int4Literal` coerced to
/// `Decimal128` (scale 0) — PG's integer->numeric coercion. `None` for a non-literal.
pub(super) fn numeric_literal_value(expr: &ResidentExpr) -> Option<Decimal128> {
    match expr {
        ResidentExpr::NumericLiteral(value) => Some(*value),
        ResidentExpr::Int4Literal(value) => Some(Decimal128::new(i128::from(*value), 0)),
        _ => None,
    }
}

/// The declared scale of a numeric column, or `None` if the column is not numeric.
pub(super) fn column_numeric_scale(table: &RelationalTable, column_idx: usize) -> Option<u8> {
    match table.columns.get(column_idx).map(|column| column.ty) {
        Some(SqlType::Numeric { scale, .. }) => Some(scale),
        _ => None,
    }
}
