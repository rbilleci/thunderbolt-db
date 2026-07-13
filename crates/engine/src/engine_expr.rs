//! General GPU executor — the `Expr` IR and its device interpreter (Charter rule 2;
//! `docs/architecture/17-general-gpu-executor.md`). This is the GENERAL path that replaces the
//! enumerated `execute_relational_*_with_resident_device_memory_probe` shape methods: a query is an
//! expression tree, lowered to a pipeline of device primitives, NOT matched against a fixed catalog
//! of shapes.
//!
//! `ResidentExpr` is the general scalar IR for the supported resident SQL types. Lowering compiles
//! arithmetic, comparison, boolean, NULL, and text predicates into typed device VM/operators; coverage
//! grows by expression node and type, never by adding another whole-query shape method.
//!
//! The IR + op-code maps + `execute_resident_expr_select_with_binding` are now the production path the
//! SQL->Expr binding (`engine_sql_pg`) routes into; the GPU parity tests exercise the same lowering
//! via programmatic `ResidentExpr`s. The 2-arg `execute_resident_expr_select` convenience wrapper (run
//! a select with a programmatic predicate, binding internally) has no production caller yet — a
//! prepared-route / facade caller is the likely one — so it carries a targeted `allow(dead_code)`.

use super::*;

/// One shard-to-unified TEXT offset rebase operation: destination offsets/base row/blob base,
/// source allocation/offsets/blob/length, and row count.
type TextRebaseOp = (u64, u32, u64, u64, u64, u64, u64, u32);

pub(crate) use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};

pub(crate) use crate::engine_join_ir::{
    JoinColRef, JoinPlan, JoinProjItem, JoinRelationRef, JoinStep,
};

mod normalization;
use normalization::{having_op_to_resident, having_value_to_resident_literal};
pub(crate) use normalization::{
    grouped_projection_to_aggregates, like_pattern_for_literal_prefix,
    resident_predicate_from_bound_filters,
};

mod shard_pruning;
use shard_pruning::{mandatory_int4_equalities, shard_zone_map_excludes};
pub(crate) use shard_pruning::shard_point_lookup_int4_eq;

mod grouped_values;
use grouped_values::{composite_group_count_reps, narrow_ordered_value};

mod execution_source;
pub(crate) use execution_source::{
    ResidentExecSource, ResidentVisibility, ShardedUnifiedExecSource,
};

/// Sentinel carried in a join's per-relation index vectors meaning "no row -> emit NULL for this
/// relation's columns" -- a LEFT OUTER join's NULL pad for an unmatched left row (M3 -- doc 21). A real
/// absolute row index can never be `u32::MAX` (residency row counts are far smaller), so it is unambiguous.
const JOIN_NULL_ROW: u32 = u32::MAX;

/// The device memory backing a join relation: a RESIDENT user table's published `Arc` (shared), or a
/// SYNTHESIZED catalog relation's freshly-uploaded TRANSIENT payload (owned for the query). `.mem()`
/// yields the `&CudaResidentDeviceMemory` the GPU pre-filter / key-projection / hash-join kernels run on.
pub(crate) enum JoinDeviceMemory {
    Resident(std::sync::Arc<gpu_db_execution::CudaResidentDeviceMemory>),
    Transient(gpu_db_execution::CudaResidentDeviceMemory),
}

struct JoinNullPadMask {
    mask: Option<gpu_db_execution::CudaPredicateMaskI32>,
    _source: gpu_db_execution::CudaResidentDeviceMemory,
    _allocation: gpu_db_execution::CudaExternalAllocationReservation,
}

pub(crate) type JoinExecSide = (
    RelationalResidencyEntry,
    JoinDeviceMemory,
    usize,
    Option<ResidentVisibility>,
);

impl JoinDeviceMemory {
    pub(crate) fn mem(&self) -> &gpu_db_execution::CudaResidentDeviceMemory {
        match self {
            JoinDeviceMemory::Resident(memory) => memory,
            JoinDeviceMemory::Transient(memory) => memory,
        }
    }
}

pub(crate) use crate::engine_result_sort::gpu_sort_permutation;


/// Device op-code for an arithmetic binary op (matches `expression_i*.ptx`: 0=add, 1=sub, 2=mul), or
/// `None` if `op` is not arithmetic.
fn arith_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::Add => Some(0),
        ResidentBinaryOp::Sub => Some(1),
        ResidentBinaryOp::Mul => Some(2),
        _ => None,
    }
}

/// Device comparison code for a comparison binary op (matches `expression_i*.ptx`: 0=eq, 1=lt, 2=le,
/// 3=gt, 4=ge), or `None` if `op` is not a kernel-supported comparison (`Ne` has no primitive yet).
fn compare_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::Eq => Some(0),
        ResidentBinaryOp::Lt => Some(1),
        ResidentBinaryOp::Le => Some(2),
        ResidentBinaryOp::Gt => Some(3),
        ResidentBinaryOp::Ge => Some(4),
        _ => None,
    }
}

/// Flip a comparison code for a swapped operand order: `k <cmp> value` == `value <flip(cmp)> k`.
/// eq stays eq; lt<->gt; le<->ge.
fn flip_comparison_code(code: u32) -> u32 {
    match code {
        1 => 3,
        2 => 4,
        3 => 1,
        4 => 2,
        other => other,
    }
}

fn is_int4_literal(expr: &ResidentExpr) -> bool {
    matches!(expr, ResidentExpr::Int4Literal(_))
}

/// The column index if `expr` is a `Column` of int8 (`SqlType::Int8`) type, else `None` (the type
/// matrix, doc 19).
fn int8_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
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
fn expr_mentions_int8(expr: &ResidentExpr, table: &RelationalTable) -> bool {
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
fn expr_mentions_int4_column(expr: &ResidentExpr, table: &RelationalTable) -> bool {
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
fn numeric_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
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
fn expr_mentions_numeric(expr: &ResidentExpr, table: &RelationalTable) -> bool {
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
fn text_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
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
fn text_literal_value(expr: &ResidentExpr) -> Option<&str> {
    match expr {
        ResidentExpr::TextLiteral(value) => Some(value.as_str()),
        _ => None,
    }
}

/// Whether `expr` mentions text anywhere -- a text `Column` or a `TextLiteral`. Marks a predicate as
/// text (the type matrix, doc 19).
fn expr_mentions_text(expr: &ResidentExpr, table: &RelationalTable) -> bool {
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
fn expr_mentions_bool_column(expr: &ResidentExpr, table: &RelationalTable) -> bool {
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
fn date_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
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
fn expr_mentions_date(expr: &ResidentExpr, table: &RelationalTable) -> bool {
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
fn date_literal_days(expr: &ResidentExpr) -> Result<i32, ExecuteError> {
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
fn timestamp_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
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
fn expr_mentions_timestamp(expr: &ResidentExpr, table: &RelationalTable) -> bool {
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
fn timestamp_literal_micros(expr: &ResidentExpr) -> Result<i64, ExecuteError> {
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
fn uuid_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
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
fn expr_mentions_uuid(expr: &ResidentExpr, table: &RelationalTable) -> bool {
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
fn uuid_literal_bytes(expr: &ResidentExpr) -> Result<[u8; 16], ExecuteError> {
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
fn int2_column_index(expr: &ResidentExpr, table: &RelationalTable) -> Option<usize> {
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
fn expr_mentions_int2(expr: &ResidentExpr, table: &RelationalTable) -> bool {
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
fn int2_literal_value(expr: &ResidentExpr) -> Result<i32, ExecuteError> {
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
fn compile_like_pattern(pattern: &str) -> Result<Vec<u32>, ExecuteError> {
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
fn numeric_literal_value(expr: &ResidentExpr) -> Option<Decimal128> {
    match expr {
        ResidentExpr::NumericLiteral(value) => Some(*value),
        ResidentExpr::Int4Literal(value) => Some(Decimal128::new(i128::from(*value), 0)),
        _ => None,
    }
}

/// The declared scale of a numeric column, or `None` if the column is not numeric.
fn column_numeric_scale(table: &RelationalTable, column_idx: usize) -> Option<u8> {
    match table.columns.get(column_idx).map(|column| column.ty) {
        Some(SqlType::Numeric { scale, .. }) => Some(scale),
        _ => None,
    }
}

/// Rescale a numeric literal to a column's scale and return the comparable i128 mantissa. The literal
/// rescales UP exactly (its scale <= the column scale). A literal with MORE fractional digits than the
/// column is rejected: rescaling it down would round, and PG compares numerics exactly — rounding
/// would yield wrong rows. Cross-scale comparison (rescaling the column on the GPU) is a follow-on.
fn rescale_numeric_literal(literal: Decimal128, column_scale: u8) -> Result<i128, ExecuteError> {
    // Canonicalize first (strip trailing zeros) so `10.500` is treated as `10.5` — PG ignores trailing
    // zeros (10.500 == 10.50), so it must NOT be rejected just for a wider written scale.
    let literal = literal.canonical();
    if literal.scale > column_scale {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "numeric literal has more fractional digits than the column scale (exact cross-scale \
             comparison is a follow-on)"
                .to_string(),
        )));
    }
    literal
        .rescale(column_scale)
        .map(|rescaled| rescaled.mantissa)
        .map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric literal out of range after rescaling to the column scale".to_string(),
            ))
        })
}

/// 10^exp as i32 — the multiplier that rescales a numeric mantissa UP by `exp` decimal places (a
/// scale gap). Errors if it does not fit i32 (exp > 9), since the in-VM scalar is i32-bounded; a wider
/// cross-scale gap is a follow-on.
fn pow10_i32(exp: u8) -> Result<i32, ExecuteError> {
    let mut acc: i32 = 1;
    for _ in 0..exp {
        acc = acc.checked_mul(10).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric cross-scale comparison spans more than 9 fractional digits (a follow-on)"
                    .to_string(),
            ))
        })?;
    }
    Ok(acc)
}

/// Push a step rescaling the top-of-stack i128 buffer from scale `from` UP to `to` (mantissa *
/// 10^(to-from), a checked multiply), or nothing if already equal. `to >= from` (the caller passes the
/// common = max scale). Used to bring the operands of a cross-scale numeric comparison to one scale.
fn push_numeric_rescale(from: u8, to: u8, program: &mut Vec<ExprStep>) -> Result<(), ExecuteError> {
    if to > from {
        program.push(ExprStep::ScalarBinary {
            op: 2, // multiply by 10^(to-from)
            scalar: pow10_i32(to - from)?,
            scalar_on_left: false,
        });
    }
    Ok(())
}

/// The result scale a numeric arithmetic expression compiles to, computed WITHOUT emitting steps. The
/// cross-scale add/sub/compare lowering needs each side's scale up front (to size the common = max
/// scale) before the operands are pushed, since the rescale step acts on the top of the stack. Columns
/// at their catalog scale; literals at their canonical scale; `+`/`-` -> max(operand scales); `*` ->
/// the sum. Mirrors `compile_numeric_arith`'s scale arithmetic exactly.
fn numeric_arith_scale(expr: &ResidentExpr, table: &RelationalTable) -> Result<u8, ExecuteError> {
    match expr {
        ResidentExpr::Column(col) => column_numeric_scale(table, *col).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric arithmetic operand is not a numeric column".to_string(),
            ))
        }),
        ResidentExpr::NumericLiteral(decimal) => Ok(decimal.canonical().scale),
        ResidentExpr::Int4Literal(_) => Ok(0),
        // An `Int8Literal` is only produced for an int8 COMPARISON leaf (`int8col <op> Int8Literal`), never
        // as a numeric-arithmetic operand — a clean error rather than a silent mis-scale.
        ResidentExpr::Int8Literal(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "an int8/text/bool literal is not a numeric arithmetic operand".to_string(),
        ))),
        // IS NULL is a predicate leaf, never an arithmetic operand (the parser never produces it here).
        ResidentExpr::IsNull { .. } => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "IS NULL is not a numeric arithmetic operand".to_string(),
        ))),
        ResidentExpr::Binary { op, lhs, rhs } => {
            let lhs_scale = numeric_arith_scale(lhs, table)?;
            let rhs_scale = numeric_arith_scale(rhs, table)?;
            match arith_op_code(*op) {
                Some(2) => lhs_scale.checked_add(rhs_scale).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "numeric multiply result scale exceeds the supported range".to_string(),
                    ))
                }),
                Some(0 | 1) => Ok(lhs_scale.max(rhs_scale)),
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports numeric +, -, * arithmetic only".to_string(),
                ))),
            }
        }
    }
}

/// Compile a numeric ARITHMETIC subtree to i128 VM steps, RETURNING the result's scale (computed
/// bottom-up). A `Column` loads from its device offset (scale = its catalog scale). `+`/`-` bring both
/// operands to the common = max scale (rescaling the coarser UP) and keep it; `*` ADDS the operand
/// scales (PG numeric multiply). A literal operand folds into a `ScalarBinary`: for `+`/`-` its
/// mantissa rescales to the common scale; for `*` its CANONICAL mantissa is the multiplier and its
/// canonical scale adds to the result. Every scalar must fit i32 (the `ExprStep` bound; larger literals
/// are a follow-on). Mirrors `compile_arith_program`.
fn compile_numeric_arith(
    expr: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<u8, ExecuteError> {
    let scalar_i32 = |mantissa: i128| -> Result<i32, ExecuteError> {
        i32::try_from(mantissa).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric arithmetic literal is too large for the fast path yet".to_string(),
            ))
        })
    };
    let scale_overflow = || {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "numeric multiply result scale exceeds the supported range".to_string(),
        ))
    };
    let literal_only = || {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "the general GPU executor does not constant-fold literal-only numeric arithmetic"
                .to_string(),
        ))
    };
    match expr {
        ResidentExpr::Column(col) => {
            let scale = column_numeric_scale(table, *col).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "numeric arithmetic operand is not a numeric column".to_string(),
                ))
            })?;
            let byte_offset = resident_device_numeric_column_offset(snapshot, table, *col)?;
            program.push(ExprStep::LoadColumn { byte_offset });
            Ok(scale)
        }
        // IS NULL is a predicate leaf, never an arithmetic operand (the parser never produces it here).
        ResidentExpr::IsNull { .. } => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "IS NULL is not a numeric arithmetic operand".to_string(),
        ))),
        ResidentExpr::Binary { op, lhs, rhs } => {
            let op_code = arith_op_code(*op).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports numeric +, -, * arithmetic only".to_string(),
                ))
            })?;
            let lhs_lit = numeric_literal_value(lhs);
            let rhs_lit = numeric_literal_value(rhs);
            if op_code == 2 {
                // MULTIPLY: the result scale is the SUM of the operand scales. A literal multiplier
                // contributes its canonical mantissa + canonical scale (so `* 1.5` adds scale 1).
                return match (lhs_lit, rhs_lit) {
                    (None, Some(literal)) | (Some(literal), None) => {
                        let value = if lhs_lit.is_none() {
                            lhs.as_ref()
                        } else {
                            rhs.as_ref()
                        };
                        let value_scale = compile_numeric_arith(value, table, snapshot, program)?;
                        let canonical = literal.canonical();
                        program.push(ExprStep::ScalarBinary {
                            op: 2,
                            scalar: scalar_i32(canonical.mantissa)?,
                            scalar_on_left: false, // multiply is commutative
                        });
                        value_scale
                            .checked_add(canonical.scale)
                            .ok_or_else(scale_overflow)
                    }
                    (None, None) => {
                        let lhs_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
                        let rhs_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
                        program.push(ExprStep::BufferBinary { op: 2 });
                        lhs_scale.checked_add(rhs_scale).ok_or_else(scale_overflow)
                    }
                    (Some(_), Some(_)) => Err(literal_only()),
                };
            }
            // ADD / SUB: bring both operands to the common = max scale (rescale the COARSER side UP),
            // then add/sub; the result keeps the common scale.
            match (lhs_lit, rhs_lit) {
                (None, Some(literal)) => {
                    let value_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
                    let common = value_scale.max(literal.canonical().scale);
                    push_numeric_rescale(value_scale, common, program)?;
                    let mantissa = rescale_numeric_literal(literal, common)?;
                    program.push(ExprStep::ScalarBinary {
                        op: op_code,
                        scalar: scalar_i32(mantissa)?,
                        scalar_on_left: false,
                    });
                    Ok(common)
                }
                (Some(literal), None) => {
                    let value_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
                    let common = value_scale.max(literal.canonical().scale);
                    push_numeric_rescale(value_scale, common, program)?;
                    let mantissa = rescale_numeric_literal(literal, common)?;
                    program.push(ExprStep::ScalarBinary {
                        op: op_code,
                        scalar: scalar_i32(mantissa)?,
                        scalar_on_left: true,
                    });
                    Ok(common)
                }
                (None, None) => {
                    // The common scale must be known BEFORE compiling either side (the rescale step
                    // operates on the top of the stack), so predict each side's scale statically.
                    let common =
                        numeric_arith_scale(lhs, table)?.max(numeric_arith_scale(rhs, table)?);
                    let lhs_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
                    push_numeric_rescale(lhs_scale, common, program)?;
                    let rhs_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
                    push_numeric_rescale(rhs_scale, common, program)?;
                    program.push(ExprStep::BufferBinary { op: op_code });
                    Ok(common)
                }
                (Some(_), Some(_)) => Err(literal_only()),
            }
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::Int8Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a bare literal cannot be a numeric arithmetic value".to_string(),
        ))),
    }
}

/// Compile a single numeric COMPARISON `lhs <cmp> rhs` to i128 VM steps that leave one MASK on the
/// stack (the buffer path: compile each side via [`compile_numeric_arith`], rescale to the common =
/// max scale, then a `CompareScalar`/`CompareBuffers` mask step). Each side may be a column, an
/// arithmetic subtree, or a literal (folded into the scalar). This is the shared comparison core for
/// both the single-predicate arithmetic arm and the numeric AND/OR mask program.
fn compile_numeric_compare(
    lhs: &ResidentExpr,
    rhs: &ResidentExpr,
    cmp: u32,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    // The numeric program runs at elem I128, so the comparison literal is a FULL i128 mantissa via
    // CompareScalarI128 -- NOT the i32 `CompareScalar` scalar (which capped a high-scale literal, e.g. a
    // scale-20 AVG result rescaled, at i32 and clean-errored a valid AND/OR HAVING/WHERE leaf).
    match (numeric_literal_value(lhs), numeric_literal_value(rhs)) {
        (Some(literal), None) => {
            // literal <cmp> arith: bring both to the common = max scale.
            let arith_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
            let common = arith_scale.max(literal.canonical().scale);
            push_numeric_rescale(arith_scale, common, program)?;
            program.push(ExprStep::CompareScalarI128 {
                cmp,
                scalar: rescale_numeric_literal(literal, common)?,
                scalar_on_left: true,
            });
            // M3 (doc 21) 3VL: AND the leaf mask with the arith operand columns' validity (a NULL operand
            // is UNKNOWN ⇒ excluded). No-op when no operand column has a validity bitmap (non-null path
            // stays byte-identical). Correct for WHERE under AND/OR (no NOT) — same rule as the int4 leaf.
            push_leaf_validity_and(&[rhs], table, snapshot, program)
        }
        (None, Some(literal)) => {
            let arith_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
            let common = arith_scale.max(literal.canonical().scale);
            push_numeric_rescale(arith_scale, common, program)?;
            program.push(ExprStep::CompareScalarI128 {
                cmp,
                scalar: rescale_numeric_literal(literal, common)?,
                scalar_on_left: false,
            });
            push_leaf_validity_and(&[lhs], table, snapshot, program)
        }
        (None, None) => {
            // arith <cmp> arith: bring both sides to the common = max scale.
            let common = numeric_arith_scale(lhs, table)?.max(numeric_arith_scale(rhs, table)?);
            let lhs_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
            push_numeric_rescale(lhs_scale, common, program)?;
            let rhs_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
            push_numeric_rescale(rhs_scale, common, program)?;
            program.push(ExprStep::CompareBuffers { cmp });
            push_leaf_validity_and(&[lhs, rhs], table, snapshot, program)
        }
        (Some(_), Some(_)) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "the general GPU executor does not evaluate literal-only numeric comparisons"
                .to_string(),
        ))),
    }
}

/// Compile a numeric boolean predicate (comparisons combined by `AND`/`OR`) into a mask program for
/// the i128 VM, mirroring [`compile_predicate_program`] (the int path) but scale-aware: `AND`/`OR`
/// compile both operand predicates then a `MaskBinary`; a comparison leaf goes through
/// [`compile_numeric_compare`]. The VM compacts the final mask to surviving row indices.
fn compile_numeric_predicate_program(
    predicate: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    let ResidentExpr::Binary { op, lhs, rhs } = predicate else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident numeric predicate must be a comparison or AND/OR combination".to_string(),
        )));
    };
    if let Some(bool_op) = boolean_op_code(*op) {
        compile_numeric_predicate_program(lhs, table, snapshot, program)?;
        compile_numeric_predicate_program(rhs, table, snapshot, program)?;
        program.push(ExprStep::MaskBinary { op: bool_op });
        return Ok(());
    }
    let Some(cmp) = predicate_compare_code(*op) else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident numeric predicate node must be a comparison (eq/ne/lt/le/gt/ge) or AND/OR"
                .to_string(),
        )));
    };
    compile_numeric_compare(lhs, rhs, cmp, table, snapshot, program)
}

/// Compile an int4 arithmetic value expression into postfix [`ExprStep`] bytecode for the device
/// VM. Recurses: a `Column` loads to a buffer; a `Binary{arith}` emits its operands then a buffer x
/// buffer op, or folds an immediate literal operand into a buffer x scalar op (preserving operand
/// side). Rejects non-arithmetic ops and literal-only subtrees (host constant-folding is a later
/// step) — the predicate's top-level comparison is handled by the caller.
fn compile_arith_program(
    expr: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    match expr {
        ResidentExpr::Column(col) => {
            // Resolve int4 OR int8 offset by the column's catalog type; a program is mono-typed (the
            // engine rejects mixed int4/int8), and the VM is run with the matching element type.
            let byte_offset = resident_device_int_column_offset(snapshot, table, *col)?;
            program.push(ExprStep::LoadColumn { byte_offset });
            Ok(())
        }
        // IS NULL is a predicate leaf, never an arithmetic operand (the parser never produces it here).
        ResidentExpr::IsNull { .. } => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "IS NULL is not an arithmetic operand".to_string(),
        ))),
        ResidentExpr::Binary { op, lhs, rhs } => {
            let op_code = arith_op_code(*op).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident Expr arithmetic position requires an arithmetic op (add/sub/mul)"
                        .to_string(),
                ))
            })?;
            match (lhs.as_ref(), rhs.as_ref()) {
                (value, ResidentExpr::Int4Literal(scalar)) if !is_int4_literal(value) => {
                    compile_arith_program(value, table, snapshot, program)?;
                    program.push(ExprStep::ScalarBinary {
                        op: op_code,
                        scalar: *scalar,
                        scalar_on_left: false,
                    });
                    Ok(())
                }
                (ResidentExpr::Int4Literal(scalar), value) if !is_int4_literal(value) => {
                    compile_arith_program(value, table, snapshot, program)?;
                    program.push(ExprStep::ScalarBinary {
                        op: op_code,
                        scalar: *scalar,
                        scalar_on_left: true,
                    });
                    Ok(())
                }
                (lhs_expr, rhs_expr)
                    if !is_int4_literal(lhs_expr) && !is_int4_literal(rhs_expr) =>
                {
                    compile_arith_program(lhs_expr, table, snapshot, program)?;
                    compile_arith_program(rhs_expr, table, snapshot, program)?;
                    program.push(ExprStep::BufferBinary { op: op_code });
                    Ok(())
                }
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident Expr interpreter does not yet constant-fold literal-only arithmetic \
                     subtrees"
                        .to_string(),
                ))),
            }
        }
        ResidentExpr::Int4Literal(_)
        | ResidentExpr::Int8Literal(_)
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident Expr arithmetic value cannot be a bare literal (constant-folding pending)"
                .to_string(),
        ))),
    }
}

/// Device comparison code including `Ne` (matches the mask kernels: 0=eq, 1=lt, 2=le, 3=gt, 4=ge,
/// 5=ne), used by the mask-based predicate VM. (`compare_op_code` omits `Ne` because the fused
/// scalar/buffer compact kernels only cover 0-4.)
fn predicate_compare_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::Eq => Some(0),
        ResidentBinaryOp::Lt => Some(1),
        ResidentBinaryOp::Le => Some(2),
        ResidentBinaryOp::Gt => Some(3),
        ResidentBinaryOp::Ge => Some(4),
        ResidentBinaryOp::Ne => Some(5),
        _ => None,
    }
}

/// Device boolean-combinator code for the mask `MaskBinary` step (0=and, 1=or), or `None`.
fn boolean_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::And => Some(0),
        ResidentBinaryOp::Or => Some(1),
        _ => None,
    }
}

/// Collect the (distinct, first-seen order) column indices a predicate-leaf operand references. Literals
/// and `IS NULL` contribute no value-operand column (an `IS NULL` leaf is its own validity test, never an
/// arithmetic operand).
fn collect_expr_columns(expr: &ResidentExpr, out: &mut Vec<usize>) {
    match expr {
        ResidentExpr::Column(idx) => {
            if !out.contains(idx) {
                out.push(*idx);
            }
        }
        ResidentExpr::Binary { lhs, rhs, .. } => {
            collect_expr_columns(lhs, out);
            collect_expr_columns(rhs, out);
        }
        _ => {}
    }
}

/// 3VL (M3 — doc 21): AND a comparison-leaf mask (on top of the VM stack) with column `col`'s NULL
/// VALIDITY mask, so a NULL operand makes the leaf UNKNOWN ⇒ mask 0 ⇒ the row is not selected. A column
/// with NO validity bitmap (it holds no NULLs) is all-valid, so this is a no-op (skipped) and the
/// non-null predicate program stays byte-identical. Pushes `BoolMask(validity, negate=false)` then
/// `MaskBinary(AND)` (the validity bit is 1 when the row is present). Correct for WHERE under AND/OR
/// because UNKNOWN and FALSE both exclude the row; this predicate path carries no logical NOT (which
/// would distinguish them — `<>`/`!=` is a value comparison, itself UNKNOWN on a NULL operand).
fn push_column_validity_and(
    col: usize,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    if let Some(bitmap_byte_offset) = resident_device_null_column_offset(snapshot, table, col)? {
        program.push(ExprStep::BoolMask {
            bitmap_byte_offset,
            negate: false,
        });
        program.push(ExprStep::MaskBinary { op: 0 }); // 0 = AND
    }
    Ok(())
}

/// Apply [`push_column_validity_and`] for every nullable column referenced across `operands` (a
/// comparison leaf's arithmetic operand expressions), so the leaf's mask excludes any row in which ANY
/// operand column is NULL (an arithmetic result is NULL if any input is NULL).
fn push_leaf_validity_and(
    operands: &[&ResidentExpr],
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    let mut cols = Vec::new();
    for operand in operands {
        collect_expr_columns(operand, &mut cols);
    }
    for col in cols {
        push_column_validity_and(col, table, snapshot, program)?;
    }
    Ok(())
}

/// True if any VALUE-operand column the predicate references has a NULL validity bitmap (holds ≥1 NULL).
/// `IS NULL` operands are excluded (an `IS NULL` test is always defined — it needs no 3VL routing).
fn predicate_references_nullable_column(
    predicate: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
) -> Result<bool, ExecuteError> {
    let mut cols = Vec::new();
    collect_expr_columns(predicate, &mut cols);
    for col in cols {
        if resident_device_null_column_offset(snapshot, table, col)?.is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The general mask-VM element type that lowers `predicate` with NULL 3VL, or `None` if the VM cannot
/// lower it (a clean-error follow-up). The VM is MONO-TYPED — one element width for the whole program —
/// and the NULL validity AND (`push_column_validity_and`) is a type-independent 1-bit mask, so the only
/// constraint is that every value-operand column shares a VM width class. `I32` = all columns int4 / text
/// / bool (text+bool are i32-compatible mask leaves). `I64` = all columns int8 (BIGINT): the i64 buffer
/// VM the non-null int8 AND/OR path already uses (`run_expr_predicate_filter(elem=I64)`), where the
/// validity BoolMask composes as an i32 mask. A MIXED-width predicate (int4+int8, or int8+text/bool), or a
/// nullable numeric/uuid/date/timestamp/int2 column, returns `None` (those compare kernels are not yet
/// validity-aware), so it must NOT route here.
fn predicate_vm_elem_type(
    predicate: &ResidentExpr,
    table: &RelationalTable,
) -> Option<ResidentElemType> {
    let mut cols = Vec::new();
    collect_expr_columns(predicate, &mut cols);
    let ty = |col: usize| table.columns.get(col).map(|column| column.ty);
    // int2 (smallint) is stored WIDENED to i32 in the int4 section, so it lowers on the I32 VM exactly
    // like int4 (its literal is an i32 that fits CompareScalar). compile_arith_program resolves its offset
    // via resident_device_int_column_offset (Int2 -> the int4 section).
    if cols.iter().all(|&col| {
        matches!(
            ty(col),
            Some(SqlType::Int4 | SqlType::Int2 | SqlType::Text | SqlType::Bool)
        )
    }) {
        Some(ResidentElemType::I32)
    } else if cols
        .iter()
        .all(|&col| matches!(ty(col), Some(SqlType::Int8)))
    {
        Some(ResidentElemType::I64)
    } else {
        // NB: Timestamp is deliberately NOT folded into the I64 case here — the NULLABLE-column read
        // branch keys off this helper and must keep routing a nullable timestamp to its dedicated 3VL
        // temporal peephole (whose literal is a parsed micros, not an Int8Literal the generic VM emits).
        // The non-null timestamp AND/OR path checks i64-section columns LOCALLY (see
        // `try_lower_timestamp_predicate`) so it does not disturb that routing.
        None
    }
}

/// ADR-006 (MIXED-WIDTH groups): TRUE iff every i64-section (Int8/Timestamp) column in `expr`
/// appears ONLY as a BARE `Column` compared against a scalar literal — the shapes the WIDTH-SAFE
/// I64 scalar leaves serve (`LoadColumnI64` loads 8 bytes REGARDLESS of the program elem and
/// `CompareScalarI64` immediately consumes that buffer — the SV3b mixed-width VM contract). Any
/// other i64 usage (an arithmetic subtree, col-vs-col — where `LoadColumn`/`CompareBuffers` obey
/// the PROGRAM width and would 4-byte-mis-read in an I32 program) fails the check. `IsNull` is
/// width-free (it reads the validity bitmap, never the value section).
fn i64_section_leaves_scalar_only(expr: &ResidentExpr, table: &RelationalTable) -> bool {
    let mentions_i64 = |e: &ResidentExpr| {
        let mut cols = Vec::new();
        collect_expr_columns(e, &mut cols);
        cols.iter().any(|&col| {
            matches!(
                table.columns.get(col).map(|column| column.ty),
                Some(SqlType::Int8 | SqlType::Timestamp)
            )
        })
    };
    match expr {
        ResidentExpr::Binary { op, lhs, rhs } if boolean_op_code(*op).is_some() => {
            i64_section_leaves_scalar_only(lhs, table) && i64_section_leaves_scalar_only(rhs, table)
        }
        ResidentExpr::Binary { lhs, rhs, .. } => {
            let scalar_side_ok = |col_side: &ResidentExpr, lit_side: &ResidentExpr| {
                let ResidentExpr::Column(col) = col_side else {
                    return false; // an arith subtree mentioning i64 -> width-unsafe
                };
                match table.columns.get(*col).map(|column| column.ty) {
                    // The int8 scalar arms accept an i64 or a widened i32 literal.
                    Some(SqlType::Int8) => matches!(
                        lit_side,
                        ResidentExpr::Int8Literal(_) | ResidentExpr::Int4Literal(_)
                    ),
                    // The timestamp leaf accepts raw micros (DML) or a parsed text bound (reads).
                    Some(SqlType::Timestamp) => matches!(
                        lit_side,
                        ResidentExpr::Int8Literal(_) | ResidentExpr::TextLiteral(_)
                    ),
                    _ => false,
                }
            };
            match (mentions_i64(lhs), mentions_i64(rhs)) {
                (false, false) => true,
                (true, true) => false, // i64 col-vs-col / both-sides usage
                (true, false) => scalar_side_ok(lhs, rhs),
                (false, true) => scalar_side_ok(rhs, lhs),
            }
        }
        ResidentExpr::IsNull { .. } => true,
        // A bare i64 Column used AS the predicate (invalid SQL for non-bool) is width-unsafe.
        ResidentExpr::Column(_) => !mentions_i64(expr),
        _ => true, // literals mention no column
    }
}

/// ADR-006 (MIXED-WIDTH groups): the LOCAL I32 gate for a predicate `predicate_vm_elem_type`
/// declines — `Some(I32)` iff every column is I32-mask-servable ({Int4, Int2, Text, Bool, Date,
/// Uuid} — 4-byte loads or mask-only leaves) or an i64-section column used ONLY in width-safe
/// scalar leaves (`i64_section_leaves_scalar_only`). ALL-i64-section predicates return `None`
/// so the proven I64 paths (int8 general VM / timestamp AND-OR / nullable I64 gate) keep serving
/// them — this gate exists for genuinely MIXED groups (e.g. `big > 5 AND name = 'x'`), which
/// previously hard-errored everywhere. Deliberately NOT a widening of the SHARED
/// `predicate_vm_elem_type` (the timestamp-diversion lesson: widening the shared helper reroutes
/// working read paths); every caller opts in LOCALLY.
fn mixed_width_i32_elem(
    predicate: &ResidentExpr,
    table: &RelationalTable,
) -> Option<ResidentElemType> {
    let mut cols = Vec::new();
    collect_expr_columns(predicate, &mut cols);
    if cols.is_empty() {
        return None;
    }
    let ty = |col: usize| table.columns.get(col).map(|column| column.ty);
    let all_servable = cols.iter().all(|&col| {
        matches!(
            ty(col),
            Some(
                SqlType::Int4
                    | SqlType::Int2
                    | SqlType::Text
                    | SqlType::Bool
                    | SqlType::Date
                    | SqlType::Uuid
                    | SqlType::Int8
                    | SqlType::Timestamp
            )
        )
    });
    let all_i64_section = cols
        .iter()
        .all(|&col| matches!(ty(col), Some(SqlType::Int8 | SqlType::Timestamp)));
    if all_servable && !all_i64_section && i64_section_leaves_scalar_only(predicate, table) {
        Some(ResidentElemType::I32)
    } else {
        None
    }
}

/// Compile a boolean predicate expression into postfix [`ExprStep`] bytecode that leaves one MASK on
/// the VM stack. Recurses: `AND`/`OR` compile both operand predicates then a `MaskBinary`; a
/// comparison compiles its arithmetic operand(s) (via [`compile_arith_program`]) then a
/// `CompareScalar`/`CompareBuffers` mask step. The predicate VM compacts the final mask to indices.
fn compile_predicate_program(
    expr: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
    needles: &mut Vec<Vec<u8>>,
) -> Result<(), ExecuteError> {
    // A bare BOOL column used as a predicate leaf (`WHERE flag AND ...`) -> its bitmap as a mask.
    if let ResidentExpr::Column(idx) = expr {
        if table.columns.get(*idx).map(|column| column.ty) == Some(SqlType::Bool) {
            let offset = resident_device_bool_column_offset(snapshot, table, *idx)?;
            program.push(ExprStep::BoolMask {
                bitmap_byte_offset: offset,
                negate: false,
            });
            return Ok(());
        }
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a bare column predicate leaf must be a bool column (argument of WHERE must be boolean)"
                .to_string(),
        )));
    }
    // `col IS NULL` / `col IS NOT NULL` (M3 -- doc 21): the column's NULL validity bitmap as a mask,
    // reusing the bool bitmap->mask kernel. The bitmap bit is 1 when the row is VALID (present), so
    // `IS NOT NULL` reads it as-is and `IS NULL` complements it (`negate = !is_not_null`). A column with
    // no validity bitmap holds no NULLs -> every row valid -> a constant mask (all-1 for IS NOT NULL,
    // all-0 for IS NULL), needing no kernel.
    if let ResidentExpr::IsNull { col, is_not_null } = expr {
        match resident_device_null_column_offset(snapshot, table, *col)? {
            Some(bitmap_byte_offset) => program.push(ExprStep::BoolMask {
                bitmap_byte_offset,
                negate: !is_not_null,
            }),
            None => program.push(ExprStep::ConstMask {
                value: *is_not_null,
            }),
        }
        return Ok(());
    }
    let ResidentExpr::Binary { op, lhs, rhs } = expr else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident predicate must be a comparison or boolean (AND/OR) combination".to_string(),
        )));
    };
    if let Some(bool_op) = boolean_op_code(*op) {
        compile_predicate_program(lhs, table, snapshot, program, needles)?;
        compile_predicate_program(rhs, table, snapshot, program, needles)?;
        program.push(ExprStep::MaskBinary { op: bool_op });
        return Ok(());
    }
    // UUID comparison leaf -> a `UuidCmpMask` step (ADR-006) — checked BEFORE text because a uuid
    // literal is a `TextLiteral` (which would otherwise route this leaf to the text compiler and
    // error); `expr_mentions_uuid` keys on a uuid COLUMN, which no text leaf has. Covers uuid IN
    // (OR of `=`), uuid ranges, and mixed uuid+int4/text WHEREs.
    if expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table) {
        return compile_uuid_leaf(*op, lhs, rhs, table, snapshot, program, needles);
    }
    // TIMESTAMP scalar-comparison leaf (ADR-006): a BARE timestamp Column against a TextLiteral (the
    // read path's bound, parsed to micros) or a raw-micros Int8Literal (the DML builder's bound) —
    // checked BEFORE text for the same reason as uuid (the TextLiteral would mis-route to the text
    // compiler and error). Emits `LoadColumnI64` (an 8-byte load REGARDLESS of the program's elem — the
    // SV3b mixed-width step, so this leaf is width-safe in an I32 or I64 program) + `CompareScalarI64`
    // + the per-leaf validity AND. Makes compound timestamp WHEREs (nullable or not) run on the GPU.
    // Timestamp col-vs-col inside AND/OR and arith subtrees stay follow-ons (clean error → decline).
    {
        let ts_scalar = match (
            timestamp_column_index(lhs, table),
            timestamp_column_index(rhs, table),
        ) {
            (Some(col), None)
                if matches!(
                    rhs.as_ref(),
                    ResidentExpr::TextLiteral(_) | ResidentExpr::Int8Literal(_)
                ) =>
            {
                Some((col, timestamp_literal_micros(rhs)?, false))
            }
            (None, Some(col))
                if matches!(
                    lhs.as_ref(),
                    ResidentExpr::TextLiteral(_) | ResidentExpr::Int8Literal(_)
                ) =>
            {
                Some((col, timestamp_literal_micros(lhs)?, true))
            }
            _ => None,
        };
        if let Some((col, micros, scalar_on_left)) = ts_scalar {
            let Some(cmp) = predicate_compare_code(*op) else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "a timestamp predicate leaf must be a comparison (eq/ne/lt/le/gt/ge)"
                        .to_string(),
                )));
            };
            let byte_offset = resident_device_int8_column_offset(snapshot, table, col)?;
            program.push(ExprStep::LoadColumnI64 { byte_offset });
            program.push(ExprStep::CompareScalarI64 {
                cmp,
                scalar: micros,
                scalar_on_left,
            });
            // 3VL: a NULL timestamp operand (placeholder micros 0) is UNKNOWN ⇒ excluded.
            return push_column_validity_and(col, table, snapshot, program);
        }
    }
    // DATE scalar-comparison leaf (ADR-006): a BARE date Column against a TextLiteral (the read path's
    // bound, parsed to days) or a raw-days Int4Literal (the DML builder's bound) — checked BEFORE text
    // for the same mis-route reason. A date is i32 DAYS in the int4 section, so this emits a plain
    // `LoadColumn` + `CompareScalar` — 4-byte, VALID ONLY IN AN I32 PROGRAM. Width discipline: every
    // path that compiles date leaves runs at I32 (the non-null And/Or/Ne branch and the nullable local
    // gate's I32 set); the I64 int8 general path REJECTS mixed int8/date, and the I64/I128 nullable
    // arms exclude Date — so a date leaf can never enter a non-I32 program.
    {
        // TextLiteral ONLY (both the read path's bound and the DML builder's `format_date` bound) —
        // an `Int4Literal` against a date column must stay a hard error (`date = 5`, PG semantics),
        // so it deliberately falls through to the generic arm (whose offset resolver rejects Date).
        let date_scalar = match (date_column_index(lhs, table), date_column_index(rhs, table)) {
            (Some(col), None) if matches!(rhs.as_ref(), ResidentExpr::TextLiteral(_)) => {
                Some((col, date_literal_days(rhs)?, false))
            }
            (None, Some(col)) if matches!(lhs.as_ref(), ResidentExpr::TextLiteral(_)) => {
                Some((col, date_literal_days(lhs)?, true))
            }
            _ => None,
        };
        if let Some((col, days, scalar_on_left)) = date_scalar {
            let Some(cmp) = predicate_compare_code(*op) else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "a date predicate leaf must be a comparison (eq/ne/lt/le/gt/ge)".to_string(),
                )));
            };
            let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
            program.push(ExprStep::LoadColumn { byte_offset });
            program.push(ExprStep::CompareScalar {
                cmp,
                scalar: days,
                scalar_on_left,
            });
            // 3VL: a NULL date operand (placeholder days 0) is UNKNOWN ⇒ excluded.
            return push_column_validity_and(col, table, snapshot, program);
        }
    }
    // TEXT comparison leaf -> a mask step the VM combines with AND/OR (so text IN / multi-text WHERE /
    // text RANGES / a nullable-text or compound LIKE run on the GPU). LIKE -> TextLikeMask; `=`/`<>` ->
    // TextEqMask; `<`/`<=`/`>`/`>=` -> TextCmpMask (ADR-006). int4 leaves fall through to the arith VM.
    if expr_mentions_text(lhs, table) || expr_mentions_text(rhs, table) {
        if matches!(op, ResidentBinaryOp::Like) {
            return compile_text_like_leaf(lhs, rhs, table, snapshot, program, needles);
        }
        return compile_text_eq_leaf(*op, lhs, rhs, table, snapshot, program, needles);
    }
    // BOOL comparison leaf (`boolcol =/<> true|false`, either order) -> a BoolMask step. `NOT flag` is
    // mapped to `flag = false` upstream, so this also covers it. int4 leaves fall through below.
    let is_bool_col = |e: &ResidentExpr| {
        matches!(e, ResidentExpr::Column(i)
            if table.columns.get(*i).map(|column| column.ty) == Some(SqlType::Bool))
    };
    if is_bool_col(lhs) || is_bool_col(rhs) {
        return compile_bool_leaf(*op, lhs, rhs, table, snapshot, program);
    }
    let Some(cmp) = predicate_compare_code(*op) else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident predicate node must be a comparison (eq/ne/lt/le/gt/ge) or AND/OR"
                .to_string(),
        )));
    };
    // ADR-006 (MIXED-WIDTH groups): a BARE Int8 column against an int literal is a WIDTH-SAFE
    // scalar leaf — `LoadColumnI64` (8-byte load regardless of the program elem, the SV3b step)
    // + `CompareScalarI64` (immediately consumes that buffer) — so `big > 5` composes inside an
    // I32 program (mixed int8+int4/text/bool/date/uuid AND-OR). ARM-STEAL equivalence in an I64
    // program (the pure-int8 path): `LoadColumn`@I64 is the SAME `gpu_db_resident_i64_load_column`
    // kernel, `CompareScalar`@I64 widens its i32 scalar via `i64::from` exactly like the widening
    // below, and `push_leaf_validity_and(&[Column])` reduces to `push_column_validity_and(col)`.
    // (Timestamp scalar leaves are handled by the dedicated leaf above; int8 ARITH subtrees fall
    // through to `compile_arith_program`, which stays program-width — I64-only paths.)
    let int8_scalar_column = |e: &ResidentExpr| match e {
        ResidentExpr::Column(col)
            if table.columns.get(*col).map(|column| column.ty) == Some(SqlType::Int8) =>
        {
            Some(*col)
        }
        _ => None,
    };
    let int_literal_i64 = |e: &ResidentExpr| match e {
        ResidentExpr::Int8Literal(v) => Some(*v),
        ResidentExpr::Int4Literal(v) => Some(i64::from(*v)),
        _ => None,
    };
    let int8_scalar_leaf = match (
        int8_scalar_column(lhs).zip(int_literal_i64(rhs)),
        int8_scalar_column(rhs).zip(int_literal_i64(lhs)),
    ) {
        (Some((col, scalar)), _) => Some((col, scalar, false)),
        (_, Some((col, scalar))) => Some((col, scalar, true)),
        _ => None,
    };
    if let Some((col, scalar, scalar_on_left)) = int8_scalar_leaf {
        let byte_offset = resident_device_int8_column_offset(snapshot, table, col)?;
        program.push(ExprStep::LoadColumnI64 { byte_offset });
        program.push(ExprStep::CompareScalarI64 {
            cmp,
            scalar,
            scalar_on_left,
        });
        // 3VL: a NULL int8 operand (placeholder 0) is UNKNOWN ⇒ excluded.
        return push_column_validity_and(col, table, snapshot, program);
    }
    match (lhs.as_ref(), rhs.as_ref()) {
        (value, ResidentExpr::Int4Literal(scalar)) if !is_int4_literal(value) => {
            compile_arith_program(value, table, snapshot, program)?;
            program.push(ExprStep::CompareScalar {
                cmp,
                scalar: *scalar,
                scalar_on_left: false,
            });
            push_leaf_validity_and(&[value], table, snapshot, program)
        }
        (ResidentExpr::Int4Literal(scalar), value) if !is_int4_literal(value) => {
            compile_arith_program(value, table, snapshot, program)?;
            program.push(ExprStep::CompareScalar {
                cmp,
                scalar: *scalar,
                scalar_on_left: true,
            });
            push_leaf_validity_and(&[value], table, snapshot, program)
        }
        // ADR-006 (wider-type range DML): a full-width i64 literal against an int8/timestamp column ->
        // `CompareScalarI64` (the whole program runs at `I64` element width, so `LoadColumn` reads 8 bytes
        // and this compares the s64 scalar). `value` is the int8 column (or arith subtree); a literal-vs-
        // literal falls to `compile_arith_program`'s bare-literal error, exactly like the Int4Literal arms.
        (value, ResidentExpr::Int8Literal(scalar)) if !is_int4_literal(value) => {
            compile_arith_program(value, table, snapshot, program)?;
            program.push(ExprStep::CompareScalarI64 {
                cmp,
                scalar: *scalar,
                scalar_on_left: false,
            });
            push_leaf_validity_and(&[value], table, snapshot, program)
        }
        (ResidentExpr::Int8Literal(scalar), value) if !is_int4_literal(value) => {
            compile_arith_program(value, table, snapshot, program)?;
            program.push(ExprStep::CompareScalarI64 {
                cmp,
                scalar: *scalar,
                scalar_on_left: true,
            });
            push_leaf_validity_and(&[value], table, snapshot, program)
        }
        (lhs_expr, rhs_expr) if !is_int4_literal(lhs_expr) && !is_int4_literal(rhs_expr) => {
            compile_arith_program(lhs_expr, table, snapshot, program)?;
            compile_arith_program(rhs_expr, table, snapshot, program)?;
            program.push(ExprStep::CompareBuffers { cmp });
            push_leaf_validity_and(&[lhs_expr, rhs_expr], table, snapshot, program)
        }
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident predicate comparison cannot be literal-vs-literal (constant-folding pending)"
                .to_string(),
        ))),
    }
}

/// Compile a TEXT comparison leaf into a mask VM step + record its needle bytes in `needles` (indexed
/// by `needle_idx`). `=` -> `TextEqMask` (negate false), `<>` -> `TextEqMask` (negate true);
/// `<`/`<=`/`>`/`>=` (ADR-006) -> `TextCmpMask` (the lexicographic byte-compare kernel, matching the
/// host `str::cmp`; a column-on-RIGHT `'lit' < col` sets `scalar_on_left` — the kernel negates the
/// ordinal). Mirrors the single-comparison text fast paths but as masks the VM can AND/OR — text
/// ranges (`name >= 'a' AND name < 'm'`), text IN, and mixed text+int4 WHEREs. Text column-vs-column
/// (ADR-006) -> `TextCmpColumnsMask` (the per-row two-column byte-compare kernel; both validity
/// masks AND'd — either operand NULL is UNKNOWN).
fn compile_text_eq_leaf(
    op: ResidentBinaryOp,
    lhs: &ResidentExpr,
    rhs: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
    needles: &mut Vec<Vec<u8>>,
) -> Result<(), ExecuteError> {
    // (col, literal, column_on_left). Equality is symmetric; the inequality kernel needs the side.
    let (col, literal, column_on_left) =
        match (text_column_index(lhs, table), text_column_index(rhs, table)) {
            (Some(col), None) if text_literal_value(rhs).is_some() => {
                (col, text_literal_value(rhs).expect("checked"), true)
            }
            (None, Some(col)) if text_literal_value(lhs).is_some() => {
                (col, text_literal_value(lhs).expect("checked"), false)
            }
            (Some(a), Some(b)) => {
                // COL-VS-COL (ADR-006): the per-row two-column lexicographic byte-compare kernel
                // as a mask step (`ta <cmp> tb`, positional — no side flip). All six ops share
                // one cmp-code kernel. 3VL: EITHER operand NULL is UNKNOWN — AND both columns'
                // validity masks (a NULL row's EMPTY placeholder span would otherwise compare as
                // "" and mis-order).
                let Some(cmp) = predicate_compare_code(op) else {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "a text col-vs-col leaf must be a comparison (eq/ne/lt/le/gt/ge)"
                            .to_string(),
                    )));
                };
                let a_layout = resident_device_text_column_layout(snapshot, table, a)?;
                let b_layout = resident_device_text_column_layout(snapshot, table, b)?;
                program.push(ExprStep::TextCmpColumnsMask {
                    a_offsets_byte_offset: a_layout.offsets_byte_offset,
                    a_bytes_byte_offset: a_layout.bytes_byte_offset,
                    a_bytes_len: a_layout.bytes_len,
                    b_offsets_byte_offset: b_layout.offsets_byte_offset,
                    b_bytes_byte_offset: b_layout.bytes_byte_offset,
                    b_bytes_len: b_layout.bytes_len,
                    cmp,
                });
                push_column_validity_and(a, table, snapshot, program)?;
                return push_column_validity_and(b, table, snapshot, program);
            }
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "text comparison must be a column against a literal".to_string(),
                )));
            }
        };
    let layout = resident_device_text_column_layout(snapshot, table, col)?;
    let needle_idx = needles.len() as u32;
    needles.push(literal.as_bytes().to_vec());
    match op {
        ResidentBinaryOp::Eq | ResidentBinaryOp::Ne => {
            program.push(ExprStep::TextEqMask {
                offsets_byte_offset: layout.offsets_byte_offset,
                bytes_byte_offset: layout.bytes_byte_offset,
                bytes_len: layout.bytes_len,
                needle_idx,
                negate: matches!(op, ResidentBinaryOp::Ne),
            });
        }
        ResidentBinaryOp::Lt
        | ResidentBinaryOp::Le
        | ResidentBinaryOp::Gt
        | ResidentBinaryOp::Ge => {
            let cmp = predicate_compare_code(op).expect("lt/le/gt/ge have compare codes");
            program.push(ExprStep::TextCmpMask {
                offsets_byte_offset: layout.offsets_byte_offset,
                bytes_byte_offset: layout.bytes_byte_offset,
                bytes_len: layout.bytes_len,
                needle_idx,
                // The kernel evaluates `scalar <cmp> textcol[i]` when scalar_on_left — i.e. when the
                // COLUMN is on the RIGHT of the original comparison.
                scalar_on_left: !column_on_left,
                cmp,
            });
        }
        _ => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports text = <> < <= > >= and LIKE only".to_string(),
            )));
        }
    }
    // 3VL: a NULL text operand makes any comparison UNKNOWN ⇒ the row is not selected (its placeholder
    // is an empty span, which would otherwise mis-match `= ''` / mis-pass `<> 'x'` / mis-order `< 'x'`).
    push_column_validity_and(col, table, snapshot, program)
}

/// Compile a UUID comparison leaf (`uuidcol <cmp> 'uuid-literal'`, either operand order) into a
/// `UuidCmpMask` VM step + record the literal's 16 bytes in `needles` (ADR-006). Mirrors the
/// single-comparison uuid fast path (`try_lower_uuid_predicate`) but as a mask the VM can AND/OR —
/// uuid IN (an OR of `=`), uuid ranges, mixed uuid+int4/text WHEREs. All six comparison ops lower
/// (uuid is byte-comparable, PG's uuid order). uuid column-vs-column (ADR-006) -> the per-row b128
/// columns kernel as a `UuidCmpColumnsMask` step, both validity masks AND'd.
fn compile_uuid_leaf(
    op: ResidentBinaryOp,
    lhs: &ResidentExpr,
    rhs: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
    needles: &mut Vec<Vec<u8>>,
) -> Result<(), ExecuteError> {
    let Some(cmp) = predicate_compare_code(op) else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a uuid predicate leaf must be a comparison (eq/ne/lt/le/gt/ge)".to_string(),
        )));
    };
    let (col, needle, column_on_left) =
        match (uuid_column_index(lhs, table), uuid_column_index(rhs, table)) {
            (Some(col), None) => (col, uuid_literal_bytes(rhs)?, true),
            (None, Some(col)) => (col, uuid_literal_bytes(lhs)?, false),
            (Some(a), Some(b)) => {
                // COL-VS-COL inside AND/OR (ADR-006): the same per-row b128 memcmp kernel the
                // standalone col-vs-col path launches, composed as a mask step. Column order is
                // positional (`ua <cmp> ub`), so no side flip. 3VL: EITHER operand NULL makes the
                // comparison UNKNOWN — AND both columns' validity masks.
                let a_byte_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_byte_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                program.push(ExprStep::UuidCmpColumnsMask {
                    a_byte_offset,
                    b_byte_offset,
                    cmp,
                });
                push_column_validity_and(a, table, snapshot, program)?;
                return push_column_validity_and(b, table, snapshot, program);
            }
            (None, None) => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "a uuid predicate must involve a uuid column".to_string(),
                )));
            }
        };
    let byte_offset = resident_device_numeric_column_offset(snapshot, table, col)?;
    let needle_idx = needles.len() as u32;
    needles.push(needle.to_vec());
    program.push(ExprStep::UuidCmpMask {
        byte_offset,
        needle_idx,
        // The kernel evaluates `needle <cmp> uuid[i]` when scalar_on_left — the COLUMN on the RIGHT.
        scalar_on_left: !column_on_left,
        cmp,
    });
    // 3VL: a NULL uuid operand makes any comparison UNKNOWN ⇒ the row is not selected (its 16-zero-byte
    // placeholder would otherwise mis-match `= '00000000-...'` / mis-order `< x`).
    push_column_validity_and(col, table, snapshot, program)
}

/// Compile a TEXT `LIKE` leaf (`textcol LIKE 'pattern'`) into a `TextLikeMask` VM step + record the
/// COMPILED pattern tokens (LE-serialized) in `needles` (indexed by `pattern_idx`). The mask-VM sibling of
/// the standalone `try_lower_text_predicate` LIKE path: the standalone path serves a non-null single LIKE
/// (it returns indices directly), but a NULLABLE-text or compound LIKE routes through the mask VM, which
/// had no LIKE step (only `Eq`/`Ne`). `LIKE` is NOT symmetric — the column is the lhs, the pattern literal
/// the rhs (PG: `x ~~ y`). The tokens are the host-compiled u32 array (`compile_like_pattern`, escapes
/// resolved), serialized to bytes for the generic `text_needles` channel; the kernel reads `ntok = len/4`.
fn compile_text_like_leaf(
    lhs: &ResidentExpr,
    rhs: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
    needles: &mut Vec<Vec<u8>>,
) -> Result<(), ExecuteError> {
    let (col, pattern) = match (text_column_index(lhs, table), text_literal_value(rhs)) {
        (Some(col), Some(pattern)) => (col, pattern),
        _ => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "LIKE must be a text column LIKE a literal pattern".to_string(),
            )));
        }
    };
    let tokens = compile_like_pattern(pattern)?;
    let layout = resident_device_text_column_layout(snapshot, table, col)?;
    let pattern_idx = needles.len() as u32;
    needles.push(tokens.iter().flat_map(|t| t.to_le_bytes()).collect());
    program.push(ExprStep::TextLikeMask {
        offsets_byte_offset: layout.offsets_byte_offset,
        bytes_byte_offset: layout.bytes_byte_offset,
        bytes_len: layout.bytes_len,
        pattern_idx,
    });
    // 3VL: a NULL text operand makes `LIKE` UNKNOWN ⇒ the row is excluded. A NULL's placeholder is an
    // empty span (start==end), which already fails any non-empty pattern; but `LIKE '%'` matches the empty
    // span, so the validity AND is load-bearing (PG: `NULL LIKE '%'` is UNKNOWN, not TRUE).
    push_column_validity_and(col, table, snapshot, program)
}

/// Compile a BOOL comparison leaf (`boolcol =/<> true|false`, either operand order) into a `BoolMask` VM
/// step the mask VM combines with AND/OR. `= true` -> the set bits (negate false); `= false` (and the
/// upstream `NOT flag` -> `flag = false`) -> the clear bits (negate true); `<>` is the complement.
/// Mirrors the single-comparison bool fast path but as a mask. Ordering ops are rejected (never mis-run).
fn compile_bool_leaf(
    op: ResidentBinaryOp,
    lhs: &ResidentExpr,
    rhs: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    let is_bool_col =
        |idx: usize| table.columns.get(idx).map(|column| column.ty) == Some(SqlType::Bool);
    let (col, literal, column_on_left) = match (lhs, rhs) {
        (ResidentExpr::Column(col), ResidentExpr::BoolLiteral(b)) if is_bool_col(*col) => {
            (*col, *b, true)
        }
        (ResidentExpr::BoolLiteral(b), ResidentExpr::Column(col)) if is_bool_col(*col) => {
            (*col, *b, false)
        }
        _ => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a bool comparison must be a bool column against a true/false literal".to_string(),
            )));
        }
    };
    // ADR-006 (bool inequalities): PG orders `false < true`, so every `<`/`<=`/`>`/`>=` against a
    // TWO-VALUED literal CONSTANT-FOLDS to an equality mask or a constant — no new kernel:
    //   col <  true  ⇔ col = false        col <  false ⇔ FALSE (nothing sorts below false)
    //   col <= false ⇔ col = false        col <= true  ⇔ TRUE-for-KNOWN (3VL: NULL is UNKNOWN)
    //   col >  false ⇔ col = true         col >  true  ⇔ FALSE
    //   col >= true  ⇔ col = true         col >= false ⇔ TRUE-for-KNOWN
    // A literal-on-left comparison is the column-on-left one with the op FLIPPED (`true > col` ⇔
    // `col < true`). The host recheck (`compare_sql_values` Bool = `bool::cmp`) orders identically.
    let effective = if column_on_left {
        op
    } else {
        match op {
            ResidentBinaryOp::Lt => ResidentBinaryOp::Gt,
            ResidentBinaryOp::Le => ResidentBinaryOp::Ge,
            ResidentBinaryOp::Gt => ResidentBinaryOp::Lt,
            ResidentBinaryOp::Ge => ResidentBinaryOp::Le,
            other => other,
        }
    };
    // `Some(needle)` selects rows where col == needle (a BoolMask); `None` is a constant verdict.
    let (needle, const_true_for_known) = match effective {
        ResidentBinaryOp::Eq => (Some(literal), false),
        ResidentBinaryOp::Ne => (Some(!literal), false),
        ResidentBinaryOp::Lt if literal => (Some(false), false),
        ResidentBinaryOp::Le if !literal => (Some(false), false),
        ResidentBinaryOp::Gt if !literal => (Some(true), false),
        ResidentBinaryOp::Ge if literal => (Some(true), false),
        // col < false / col > true: no bool sorts there — constant FALSE (NULL rows are UNKNOWN
        // ⇒ excluded too, so no validity needed).
        ResidentBinaryOp::Lt | ResidentBinaryOp::Gt => (None, false),
        // col <= true / col >= false: TRUE for every KNOWN bool — constant TRUE masked by
        // validity (a NULL operand is UNKNOWN ⇒ excluded; the 3VL net below is load-bearing).
        ResidentBinaryOp::Le | ResidentBinaryOp::Ge => (None, true),
        _ => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only comparisons against a bool literal"
                    .to_string(),
            )));
        }
    };
    let Some(needle) = needle else {
        program.push(ExprStep::ConstMask {
            value: const_true_for_known,
        });
        if const_true_for_known {
            // 3VL: without this AND, a NULL row would ride the constant-TRUE mask.
            return push_column_validity_and(col, table, snapshot, program);
        }
        return Ok(());
    };
    let offset = resident_device_bool_column_offset(snapshot, table, col)?;
    program.push(ExprStep::BoolMask {
        bitmap_byte_offset: offset,
        negate: !needle,
    });
    // 3VL: a NULL bool operand makes the comparison UNKNOWN ⇒ not selected. The value-bitmap bit
    // of a NULL row is the 0 placeholder, so `= false` / `<> true` / `< true` would otherwise
    // wrongly select it; AND with the validity mask excludes it.
    push_column_validity_and(col, table, snapshot, program)
}

impl Engine {
    /// General GPU executor entry (Charter rule 2): run `SELECT <int4 columns> FROM <table>` filtered
    /// by a general predicate [`ResidentExpr`], evaluating the predicate on the GPU via the device
    /// interpreter and materializing the surviving rows by gathering the projected columns. The
    /// predicate is supplied as an expression tree (the unit of execution is an expression, not a
    /// recognized shape); the `Select` carries the table + projection + MVCC binding. NOT routed
    /// through `resident_route_query_shape` — this is the general path, parallel to the (frozen)
    /// enumerated probe dispatch.
    // Forward API: run a select with a PROGRAMMATIC predicate (binds the catalog internally). The GPU
    // parity tests use it; the production caller is the SQL->Expr entry, which binds once itself and
    // calls `execute_resident_expr_select_with_binding` directly, so this wrapper has no non-test
    // caller yet.
    #[allow(dead_code)]
    pub(crate) fn execute_resident_expr_select(
        &self,
        select: &Select,
        predicate: &ResidentExpr,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        self.execute_resident_expr_select_with_binding(
            select,
            &table,
            None,
            bound,
            copin_s,
            Some(predicate),
            None, // SV3b visibility: single-buffer path is never a versioned shard
            &[],
            &[],
            None,
            &[],
        )
    }

    /// `&Select`->general-executor BRIDGE (S8 grouped int4 aggregates; S10a non-grouped projections).
    /// Runs a `Select` through the SAME on-device general executor as the SQL->Expr path, retiring the
    /// legacy resident-probe methods. For a GROUPED select it normalizes the legacy 1-aggregate
    /// projection and runs the grouped path (the int4-group + int4-value shapes the route classifier
    /// accepts); for a NON-grouped select (`group_by == None`) `grouped_projection_to_aggregates` is a
    /// no-op and the group-key list is EMPTY, so the binding executor runs the plain-projection path
    /// (WHERE VM + GPU sort + LIMIT/OFFSET window) -- this is how S10a routes the `int4_ordered_projection`
    /// shape, replacing its probe. The grouped probes' `!gpu_ordered` branches did a HOST sort / HAVING /
    /// LIMIT (a charter violation -- relational finalization on the host). The general executor does ORDER BY / HAVING / LIMIT
    /// ON-DEVICE (S2/S3/S4), so this is behavior-preserving (enumerated == general proven 0/24
    /// differential; bridge == general re-verified before the probe methods were deleted; the general
    /// grouped ORDER BY now appends a group-key tie-break, matching the legacy group-ASC tie order).
    ///
    /// Unlike `execute_resident_expr_select_sql`, this sources its inputs from the engine `&Select`
    /// (the hand-rolled parse) rather than the libpg_query parse tree, so it covers the text entry AND
    /// the `&Select` callers with no raw SQL -- CTAS and view/matview -- uniformly. The WHERE predicate
    /// is rebuilt from the bound's resolved filters (a `ResidentExpr` DNF -- the third predicate path),
    /// then the bound filters are CLEARED so the executor filters SOLELY via the predicate, exactly as
    /// the SQL->Expr path does (whose bound carries no filters -- the WHERE rides the predicate). The
    /// grouped ORDER BY keys are RESULT columns (the group column or an aggregate), resolved by the
    /// executor against the projection, so every `order_by_exprs` entry is `None` (a plain key, not an
    /// expression); NULLS FIRST/LAST is `None` (PG default) since the hand-rolled `SelectOrder` carries
    /// no explicit override -- byte-identical to the probe path it replaces.
    pub(crate) fn execute_resident_grouped_via_general(
        &self,
        select: &Select,
        // `None` = look up the table's whole-table single resident store by name (the original
        // text/CTAS/view callers). `Some(src)` INJECTS an already-built source (S10c slice 2b: the
        // unified multi-shard buffer recompacted by `execute_resident_sharded_via_general`), so
        // the same on-device grouped/distinct/ordered path serves the sharded shapes.
        src: Option<&ResidentExecSource>,
        // R-ver PART 2: the SV3b/SV6 visibility for a VERSIONED unified `src` — threaded into
        // `indices` so GROUP BY / DISTINCT / ORDER BY group/sort/dedup over VISIBLE rows only
        // (tombstoned + too-new versions never reach the keys). MUST be `None` for a `src: None`
        // caller (with_binding builds + resolves the unified source's visibility itself — the
        // debug_assert there enforces it).
        visibility: Option<ResidentVisibility>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Normalize the legacy 1-aggregate grouped projection (GroupedCount / GroupedSum / GroupedAvg /
        // GroupedMin / GroupedMax produced by the hand-rolled parser) to the general GroupedAggregates
        // form the Expr executor consumes -- and bind against THAT, so the binding (selected columns,
        // result schema, ORDER-BY-result-column resolution) is byte-identical to the SQL->Expr path,
        // which always produces GroupedAggregates. The WHERE / GROUP BY / ORDER BY / HAVING / LIMIT are
        // carried over unchanged in the clone.
        let mut select_owned = select.clone();
        if let Some(projection) = grouped_projection_to_aggregates(&select.projection) {
            select_owned.projection = projection;
        }
        let select = &select_owned;
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        self.execute_resident_grouped_via_general_with_binding(
            select, &table, bound, copin_s, src, visibility,
        )
    }

    pub(crate) fn execute_resident_grouped_via_general_with_binding(
        &self,
        select: &Select,
        table: &RelationalTable,
        mut bound: BoundRelationalSelect,
        copin_s: Index,
        src: Option<&ResidentExecSource>,
        visibility: Option<ResidentVisibility>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Rebuild the WHERE predicate from the bound filters BEFORE clearing them; then clear so the
        // executor's filter (and the access-path planner it feeds) sees an empty-filter bound, matching
        // the SQL->Expr path exactly. The predicate ResidentExpr is now the sole filter.
        let predicate = resident_predicate_from_bound_filters(&bound)?;
        bound.filter = None;
        bound.filters.clear();
        bound.filter_groups.clear();
        // Single bare-column GROUP BY (the only grouped shape the route accepts); the executor packs a
        // composite key when len()==2, but the grouped routes are single-key, so this is one column.
        let group_key_columns: Vec<String> = match &select.group_by {
            Some(column) => vec![column.clone()],
            None => Vec::new(),
        };
        // Grouped ORDER BY keys are result columns (None = plain key); no expression GROUP BY here.
        let order_by_exprs: Vec<Option<ResidentExpr>> = vec![None; select.order_by.len()];
        let order_by_nulls_first: Vec<Option<bool>> = vec![None; select.order_by.len()];
        self.execute_resident_expr_select_with_binding(
            select,
            table,
            src,
            bound,
            copin_s,
            predicate.as_ref(),
            // R-ver PART 2: forward the versioned unified src's visibility so the grouped/ordered
            // survivors are the VISIBLE rows (was hard-`None`, which forced the caller refusal).
            visibility,
            &order_by_exprs,
            &order_by_nulls_first,
            None,
            &group_key_columns,
        )
    }

    /// SV4 (GPU-native DELETE -- LOCATE phase): find the resident `(shard_id, LOCAL slot)` positions of
    /// every row matching `predicate` (an int4-equality point-lookup shape), via zone-map-pruned per-shard
    /// `lower_resident_predicate`. PER-SHARD (NOT the recompacted unified buffer of the read path), so the
    /// returned slots are LOCAL to each shard's own device buffer -- exactly what the per-shard,
    /// local-slot-indexed `deleted_by` region needs. Returns `None` if the table is not shard-resident / a
    /// shard is invalid or missing device memory / the predicate cannot lower on a shard (caller falls back to
    /// the O(table) invalidate + re-admit). Visibility = `None`: locate addresses PHYSICAL positions (a DELETE
    /// stamps a row by WHERE IT SITS, independent of read-time visibility; the raw buffer's rows are present),
    /// and only reads that already-committed shard buffer. WIRED by SV4b (the DELETE commit path routes
    /// through `try_tombstone_resident_delete_commit`).
    pub(crate) fn locate_resident_delete_slots(
        &self,
        table: &RelationalTable,
        predicate: &ResidentExpr,
    ) -> Option<Vec<(u32, Vec<u32>)>> {
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        // Zone-map prune inputs: the mandatory (top-level-AND) int4 equalities the predicate requires.
        let mut constraints: Vec<(usize, i32)> = Vec::new();
        mandatory_int4_equalities(predicate, &mut constraints);
        let column_names: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut out: Vec<(u32, Vec<u32>)> = Vec::new();
        for shard in table_shards.iter() {
            // S-d3 zone-map prune: skip a shard whose min/max provably excludes every mandatory needle (it
            // cannot hold a matching row). Same soundness as the sharded read: prune ONLY on a stat-carrying
            // shard that provably excludes; a no-stat shard is always kept.
            if !constraints.is_empty()
                && constraints.iter().any(|(col, needle)| {
                    shard_zone_map_excludes(
                        &column_names,
                        &shard.resident_device_int4_column_stats,
                        *col,
                        *needle,
                    )
                })
            {
                continue;
            }
            // Per-shard identity/validity precheck (mirror `execute_resident_sharded_via_general::source_for`).
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            // D4: the buffer rides the loaded descriptor (generation-consistent by construction).
            let device_memory = shard.device_memory.clone()?;
            // The per-shard descriptor is capacity-strided + row_count-sized to THIS shard's buffer
            // (`resident_snapshot_for_shard`), so the predicate reads the shard's int4 columns at the right
            // offsets and returns slots LOCAL to `[0, row_count)`.
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            let slots = self
                .lower_resident_predicate(
                    predicate,
                    table,
                    &descriptor,
                    &device_memory,
                    shard.row_count as u64,
                    None,
                )
                .ok()?;
            if !slots.is_empty() {
                out.push((shard.shard_id, slots));
            }
        }
        Some(out)
    }

    /// CPU-ENGINE RETIREMENT (ADR-006): the GENERATION-CONSISTENT twin of [`Self::locate_resident_delete_slots`]
    /// for the elided-DML predicate RESOLVE (prepare_delete / prepare_update, which run OFF the commit lock).
    /// For every resident slot matching `predicate`, returns a [`ShardPkHit`] whose device buffer + all three
    /// version regions (`deleted_by` / `created_by` / `row_id`) + descriptor are CAPTURED FROM THE SAME
    /// `shards.load()` snapshot the slot was computed against, gated by the W0 `shard_write_locate_cell_live`
    /// liveness check. This closes the concurrent TOCTOU the slot-only variant would expose to a lock-free
    /// caller: a reordering re-admit (VACUUM / SV3a recompaction) between two independent `shards.load()`s
    /// could otherwise apply one generation's slots to another generation's compacted buffer = a wrong row.
    /// Mirrors `locate_resident_pk_via_shard_index_detailed`'s single-snapshot discipline. `None` = decline
    /// (the caller rehydrates): not shard-resident, a shard invalid / memory-pressured / catalog-mismatched /
    /// superseded (W0) / missing its device memory, or a predicate that could not lower on a shard.
    pub(crate) fn locate_resident_delete_slots_detailed(
        &self,
        table: &RelationalTable,
        predicate: &ResidentExpr,
    ) -> Option<Vec<crate::engine_retained_read::ShardPkHit>> {
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        if table_shards.is_empty() {
            return None;
        }
        let mut constraints: Vec<(usize, i32)> = Vec::new();
        mandatory_int4_equalities(predicate, &mut constraints);
        let column_names: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
        let runtime_snapshot = self.router.runtime().snapshot();
        let mut out: Vec<crate::engine_retained_read::ShardPkHit> = Vec::new();
        for shard in table_shards.iter() {
            // S-d3 zone-map prune (same soundness as the slot-only variant: prune ONLY a stat-carrying shard
            // that provably excludes every mandatory needle; a no-stat shard is always kept).
            if !constraints.is_empty()
                && constraints.iter().any(|(col, needle)| {
                    shard_zone_map_excludes(
                        &column_names,
                        &shard.resident_device_int4_column_stats,
                        *col,
                        *needle,
                    )
                })
            {
                continue;
            }
            if shard.schema != table.schema || shard.table != table.name {
                return None;
            }
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            if !shard.is_valid(memory_pressure_active) {
                return None;
            }
            let device_memory = shard.device_memory.clone()?;
            // W0: the descriptor flags don't see concurrent invalidations — require the authoritative cell to
            // still publish THIS buffer, else decline (the located slot would address a superseded generation).
            if !self.shard_write_locate_cell_live(&table.name, shard.shard_id, &device_memory) {
                return None;
            }
            let descriptor = self.resident_snapshot_for_shard(shard, table);
            let slots = self
                .lower_resident_predicate(
                    predicate,
                    table,
                    &descriptor,
                    &device_memory,
                    shard.row_count as u64,
                    None,
                )
                .ok()?;
            // Every hit captures the SAME generation's buffer + regions + descriptor as its slot.
            for slot in slots {
                out.push(crate::engine_retained_read::ShardPkHit {
                    shard_id: shard.shard_id,
                    slot,
                    descriptor: descriptor.clone(),
                    device_memory: device_memory.clone(),
                    deleted_by: shard.deleted_by_region.clone(),
                    created_by: shard.created_by_region.clone(),
                    row_id: shard.row_id_region.clone(),
                });
            }
        }
        Some(out)
    }

    /// SV4 (GPU-native DELETE): LOCATE the resident slots matching `predicate` + stamp `deleted_by =
    /// commit_seq` on them IN PLACE (O(rows touched)), instead of the O(table) invalidate + re-admit. Returns
    /// `Some(n)` = n slots tombstoned (n may be 0: the predicate matched no resident row -- still a success,
    /// nothing to re-admit); `None` = the caller must fall back to invalidate + re-admit (not shard-resident /
    /// locate could not run / a tombstone write failed). A PARTIAL stamp before a `None` is harmless: the
    /// fallback re-admit rebuilds every shard all-live from the host store (which already applied the DELETE)
    /// AND SV4-prereq-#1 releases any partial `deleted_by` region. MUST run under the commit lock so the
    /// per-shard `deleted_by` get-or-allocate is atomic (SV2 prereq #2). WIRED by SV4b.
    pub(crate) fn try_tombstone_resident_delete(
        &self,
        table: &RelationalTable,
        predicate: &ResidentExpr,
        commit_seq: Index,
    ) -> Option<usize> {
        let located = self.locate_resident_delete_slots(table, predicate)?;
        // Ledger #16 (SI-fix audit follow-up) — the DEFENSIVE consumer gate: the locate is
        // visibility-blind (physical int4-image match), so a caller-supplied STALE old image can
        // physically match an ALREADY-DEAD slot; re-stamping it would leave the truly-current
        // version live (the fixed SV6 double-read class). Read each located slot's deleted_by
        // FIRST and treat any already-tombstoned slot as NO MATCH (drop it) — the caller's
        // exact-count gate then mismatches and falls to the always-correct re-admit. One i64
        // read per located slot on an O(rows-touched) path; a false "already dead" can only
        // trigger a correct rebuild, never a wrong result.
        let shards = self.read_state.residency.shards.load();
        let table_shards = shards.get(&table.name)?;
        let mut total = 0usize;
        for (shard_id, slots) in &located {
            let live_slots: Vec<u32> = match table_shards
                .iter()
                .find(|shard| shard.shard_id == *shard_id)
                .and_then(|shard| shard.deleted_by_region.clone())
            {
                None => slots.clone(), // no region = delete-free shard: every located slot is live
                Some(region) => slots
                    .iter()
                    .copied()
                    .filter(|slot| {
                        region
                            .read_resident_i32_column(u64::from(*slot) * 8, 2)
                            .ok()
                            .map(|halves| {
                                let deleted =
                                    (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32);
                                // The delete-free fill (0x7F...) reads far above any real seq.
                                deleted > commit_seq
                            })
                            // A device-read failure keeps the slot: the stamp attempt below
                            // fails loudly -> None -> re-admit (never a silent drop).
                            .unwrap_or(true)
                    })
                    .collect(),
            };
            if live_slots.is_empty() {
                continue;
            }
            if !self.tombstone_resident_shard_slots(&table.name, *shard_id, &live_slots, commit_seq)
            {
                return None;
            }
            total = total.saturating_add(live_slots.len());
        }
        Some(total)
    }

    /// COMPOUND KEYS (wider types, Stage 2b): locate + tombstone the deleted `row`'s resident slot via the
    /// compound FINGERPRINT index, for tables the int4-column predicate can't uniquely locate (an i64 key
    /// column is not in the predicate). Folds the row's key tuple to its fingerprint, probes the compound
    /// index, and — because the probe is fingerprint-based (a collision could point at a DIFFERENT tuple) —
    /// TUPLE-VERIFIES each hit by materializing the slot on-device and comparing the full key columns. Only
    /// the verified, snapshot-LIVE slot is tombstoned. Returns `Some(1)` on the unique match, else `None`
    /// (ambiguous / device decline / can't-materialize) -> the caller re-admits (always correct). `ord` is
    /// `index`'s position in `table.indexes`.
    fn try_tombstone_resident_delete_via_fingerprint(
        &self,
        table: &RelationalTable,
        index: &crate::relational_model::RelationalIndex,
        ord: usize,
        row: &[SqlValue],
        commit_seq: Index,
    ) -> Option<usize> {
        let fingerprint =
            crate::engine_residency::compound_index_row_fingerprint(table, index, row)?;
        let key_id = crate::engine_residency::index_probe_key_id(table, index, ord)?;
        let key_positions = crate::engine_residency::index_key_column_positions(table, index)?;
        let hits = self.locate_resident_pk_via_shard_index_detailed(table, key_id, fingerprint)?;
        // TUPLE-VERIFY every fingerprint hit: materialize the slot at the commit boundary (created_by <=
        // seq && deleted_by > seq = snapshot-live, so an already-tombstoned slot yields None and is
        // dropped — the SI-fix already-dead discipline), then confirm the full key tuple matches. A device
        // decline (materialize None) re-admits.
        let mut matched: Vec<(u32, u32)> = Vec::new();
        for hit in &hits {
            let materialized = self.materialize_resident_row_via_hit(table, hit, commit_seq);
            let mrow = match materialized {
                Some(Some(mrow)) => mrow,
                Some(None) => continue, // not snapshot-live (already dead / future): not our slot
                None => return None, // can't materialize (device err / wider value column) -> re-admit
            };
            if key_positions.iter().all(|&p| mrow.get(p) == row.get(p)) {
                matched.push((hit.shard_id, hit.slot));
            }
        }
        // EXACT-1: a unique key identifies exactly one live slot. Anything else (0 = the resolved row
        // moved/vanished; >1 = a fingerprint collision that both tuple-matched, impossible for a unique
        // key but guarded) declines to the re-admit.
        if matched.len() != 1 {
            return None;
        }
        let (shard_id, slot) = matched[0];
        if !self.tombstone_resident_shard_slots(&table.name, shard_id, &[slot], commit_seq) {
            return None;
        }
        Some(1)
    }

    /// COMPOUND KEYS (wider types): the first compound unique index whose slot the int4-column predicate
    /// CANNOT uniquely locate — i.e. it has a key column outside the i32 section (an i64 key). Such a
    /// table's DELETE/UPDATE must locate via the fingerprint index (`try_tombstone_resident_delete_via_
    /// fingerprint`); an all-i32-section table keeps the proven int4-predicate locate. Returns `(ord, index)`.
    fn compound_index_needing_fingerprint_locate(
        table: &RelationalTable,
    ) -> Option<(usize, &crate::relational_model::RelationalIndex)> {
        table.indexes.iter().enumerate().find(|(_, index)| {
            index.unique
                && crate::engine_residency::index_is_compound(index)
                && index.key_columns.iter().any(|name| {
                    table
                        .columns
                        .iter()
                        .find(|c| &c.name == name)
                        .is_some_and(|c| {
                            !matches!(c.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2)
                        })
                })
        })
    }

    /// SV4b (commit path): for a single-entry DELETE commit, LOCATE + tombstone the deleted rows' resident
    /// slots IN PLACE instead of the O(table) invalidate + re-admit. Builds an int4-equality predicate that
    /// matches the deleted row's resident int4 columns and stamps the located slots. Returns `true` (the
    /// caller SKIPS re-admit) ONLY when the located+tombstoned count EXACTLY equals the deleted-row count;
    /// ANY ambiguity or unsupported shape returns `false` -> the caller invalidates + re-admits (rebuild
    /// all-live from the host store = always correct, so a false here is only a missed optimization, never a
    /// wrong result). Conservative FIRST-SLICE scope: exactly one deleted row, all int4 columns non-NULL
    /// plain `Int4`. `cat` is the catalog the commit already holds (NO latch re-entry). Runs under
    /// commit_mutex + catalog latch, so the per-shard `deleted_by` get-or-allocate is atomic (SV2 prereq #2).
    /// (Audit note: `cat` is the WORKING catalog while `prepare_delete` decoded the row against the published
    /// snapshot; for a single-entry non-DDL commit under the held latch these are the same shape, and any
    /// mismatch is caught by the `row.len() != table.columns.len()` guard below -> fallback.)
    pub(crate) fn try_tombstone_resident_delete_commit(
        &self,
        cat: &DdlCatalogState,
        table_name: &str,
        deleted_rows: &[Vec<SqlValue>],
        commit_seq: Index,
    ) -> bool {
        // RETIREMENT A4b: MULTI-ROW — per-row locate + tombstone, each gated EXACT count == 1. Any
        // ambiguity on ANY row (dup int4 values across the statement's rows, a locate miss, an
        // int4-identical already-tombstoned slot, NULL/non-int4) returns false -> the caller
        // invalidates + re-admits, which SUPERSEDES any tombstones already stamped this commit
        // (they are pre-publish; the re-admit rebuilds all-live and releases the regions — the
        // same partial-failure argument SV5 documented for tombstone-without-append).
        if deleted_rows.is_empty() {
            // ADR-006: a ZERO-ROW DELETE (WHERE matched nothing) is a data NO-OP — nothing to tombstone,
            // the elided table is byte-unchanged. Report it HANDLED (`true`) so the commit path keeps the
            // table ELIDED instead of treating the no-op as unhandled and REHYDRATING (de-eliding) it — a
            // pure de-elide trigger on the common `DELETE ... WHERE <no match>` OLTP shape (confirmed via
            // backtrace: apply_and_publish_committed_inner's `!handled && elided -> rehydrate` arm).
            // `deleted_rows` is the APPLIED removed set (resolved at apply), so empty == genuinely zero
            // matches, never a resolution failure. Elision-ENTER is separately gated on a non-empty applied
            // set in the caller, so this no-op never drives a table INTO elision.
            return true;
        }
        let Some(table) = cat.relational_catalog.get(table_name) else {
            return false;
        };
        // COMPOUND KEYS (wider types): a compound key with an i64 column can't be located by the
        // int4-column predicate -> use the fingerprint index + tuple-verify. All-i32-section tables keep
        // the proven int4-predicate locate.
        let fp_index = Self::compound_index_needing_fingerprint_locate(table);
        for row in deleted_rows {
            if row.len() != table.columns.len() {
                return false;
            }
            if let Some((ord, index)) = fp_index {
                if !matches!(
                    self.try_tombstone_resident_delete_via_fingerprint(
                        table, index, ord, row, commit_seq
                    ),
                    Some(1)
                ) {
                    return false;
                }
                continue;
            }
            let Some(predicate) = Self::resident_int4_row_predicate(table, row) else {
                return false;
            };
            // EXACT-1 per row. NOTE: a slot tombstoned by an EARLIER row of this same statement
            // may still be visible to this locate (stamped at commit_seq, read below it) — that
            // can only happen when two deleted rows are int4-identical, and then the FIRST row's
            // locate already saw count 2 and bailed. The per-row gate is the wrong-results net.
            if !matches!(
                self.try_tombstone_resident_delete(table, &predicate, commit_seq),
                Some(1)
            ) {
                return false;
            }
        }
        // VACUUM #5: every stamped tombstone is a DEAD SLOT until a rebuild — feed the churn
        // signal the auto-trigger reads (serialized path; the counter resets on any re-admit).
        self.add_tombstone_churn(table_name, deleted_rows.len() as u64);
        true
    }

    /// The AND-of-int4-equalities predicate locating exactly one physical row image: `(col_i =
    /// v_i)` over the table's plain-`Int4` columns. `None` when any int4 column holds NULL or a
    /// non-`Int4` value, or the table has zero int4 columns (cannot safely locate) -> re-admit.
    fn resident_int4_row_predicate(
        table: &RelationalTable,
        row: &[SqlValue],
    ) -> Option<ResidentExpr> {
        let mut predicate: Option<ResidentExpr> = None;
        for (idx, column) in table.columns.iter().enumerate() {
            if column.ty != SqlType::Int4 {
                continue;
            }
            let value = match &row[idx] {
                SqlValue::Int4(v) => *v,
                _ => return None,
            };
            let eq = ResidentExpr::Binary {
                op: ResidentBinaryOp::Eq,
                lhs: Box::new(ResidentExpr::Column(idx)),
                rhs: Box::new(ResidentExpr::Int4Literal(value)),
            };
            predicate = Some(match predicate {
                None => eq,
                Some(prev) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(prev),
                    rhs: Box::new(eq),
                },
            });
        }
        predicate
    }

    /// SV5 (commit path): for a single-entry UPDATE commit, TOMBSTONE the OLD version's resident slot + APPEND
    /// the NEW image to the open shard IN PLACE, instead of the O(table) invalidate + re-admit. ORDER matters:
    /// tombstone-OLD FIRST so `locate` runs on the buffer BEFORE the new row is appended -- an UPDATE that
    /// leaves the int4 columns UNCHANGED still locates EXACTLY the old row (count 1) rather than matching both
    /// the old + the just-appended new slot. Returns `true` (caller SKIPS re-admit) only if BOTH steps
    /// succeed; ANY failure (multi-row / NULL / non-int4 / dup-ambiguous / non-resident / no append headroom)
    /// returns `false` -> the caller invalidates + re-admits, which rebuilds the table all-live from the host
    /// store (old hidden + new present, the version rewrite already applied) = always correct. So a partial
    /// tombstone-without-append (append failed after the tombstone) is harmless -- the re-admit supersedes it
    /// and SV4-prereq-#1 releases the partial region. Runs under commit_mutex + catalog latch.
    ///
    /// **SI (SV6 — the audit-flagged P2 flip-gate, FIXED):** the append + `row_count` bump happen BEFORE
    /// `publish_committed_seq`, and lock-free reads bind `read_txn_id = committed_seq()` then load `shards`
    /// separately -- so a concurrent reader that observes `committed_seq = C-1` can observe the shards with
    /// the appended row already present. The appended NEW version is therefore STAMPED
    /// `created_by = commit_seq` (a per-shard on-demand i64 region mirroring `deleted_by`, written while the
    /// slots are still invisible headroom — see the append path's ORDER comment), and EVERY sharded read
    /// path ANDs the device-side `created_by <= read_txn_id` lower bound (scan/VM conjunct, 3b route
    /// per-hit gate, batched gather gate; the un-gated GPU dense kernel DECLINES stamped shards). So the
    /// C-1 reader sees the key EXACTLY ONCE (the OLD version: `deleted_by = C > C-1` visible, new hidden)
    /// and a reader at C sees exactly the NEW one. Gated by the
    /// `sv6_created_by_gate_reader_at_prior_snapshot_never_sees_updated_key_twice` torn-window differential
    /// and the concurrent hammer test. (DELETE/SV4b needs no lower bound -- no new row; a plain INSERT
    /// append stays unstamped/born-visible, the milder as-if-later read of a decided commit.)
    pub(crate) fn try_update_resident_commit(
        &self,
        cat: &DdlCatalogState,
        table_name: &str,
        old_rows: &[Vec<SqlValue>],
        new_rows: &[Vec<SqlValue>],
        commit_seq: Index,
        row_ids: Option<&[u64]>,
    ) -> bool {
        // ADR-006: a ZERO-ROW UPDATE (WHERE matched nothing) is a data NO-OP — `old_rows`/`new_rows` are
        // both empty (parallel), nothing to tombstone or append, the elided table is byte-unchanged.
        // Report it HANDLED (`true`) so the commit path keeps the table ELIDED instead of REHYDRATING it
        // (the `DELETE/UPDATE ... WHERE <no match>` de-elide trigger). Elision-ENTER is separately gated on
        // a non-empty applied set in the caller, so this no-op never drives a table INTO elision.
        if old_rows.is_empty() {
            return new_rows.is_empty();
        }
        // RETIREMENT A4b: MULTI-ROW — old/new/row_ids must be parallel and identity-complete;
        // ALL tombstones land before ANY append (the locate must run on the pre-append buffer).
        if old_rows.len() != new_rows.len() || row_ids.is_none_or(|ids| ids.len() != new_rows.len())
        {
            return false;
        }
        // 1. Tombstone every OLD version's slot (locates run on the buffer BEFORE the appends).
        if !self.try_tombstone_resident_delete_commit(cat, table_name, old_rows, commit_seq) {
            return false;
        }
        // 2. Append the NEW image to the open shard, stamped `created_by = commit_seq` (SV6 — the P2
        //    flip-gate fix): the append + row_count bump land BEFORE `publish_committed_seq`, so a
        //    concurrent reader bound to `committed_seq = C-1` can observe the appended slots; the stamp +
        //    the read path's `created_by <= read_txn_id` device conjunct hide the new version from that
        //    reader (it sees exactly the OLD version, still live at its snapshot). If this fails AFTER the
        //    tombstone, the caller's re-admit rebuilds all-live from the host store (which already applied
        //    the version rewrite), superseding.
        if !self.try_append_resident_int4_open_shard(
            table_name,
            new_rows,
            crate::engine_residency::AppendCreatedBy::UpdateNewVersion(commit_seq),
            row_ids,
        ) {
            return false;
        }
        true
    }

    /// SLICE B: build the SHARDED UNIFIED EXEC SOURCE — recompact a shard-resident table's shards into ONE
    /// unified device buffer (8-byte header + int4 columns in catalog order + SV3b/SV6 version columns +
    /// M3 null-validity bitmaps) and return it as an injectable [`ResidentExecSource`] plus its MVCC
    /// [`ResidentVisibility`], so ANY general-executor caller runs the SAME on-device execution over
    /// sharded tables: the sharded shape BRIDGE below AND the SQL->Expr PG path (which serves `IS NULL` /
    /// `IS NOT NULL` and every other general shape — previously those ERRORED on a sharded-only table
    /// because the PG path only knew the single-buffer store). `predicate` drives the S-d3 zone-map prune
    /// (mandatory int4 equalities only; `None` / non-equality predicates gather every shard — sound, the
    /// device predicate filters). The unified SoA is laid out exactly as a whole-table single store, so
    /// the single-store offset helpers address it byte-identically; per-shard identity/validity prechecks
    /// mirror the retired probes. Extracted VERBATIM from `execute_resident_sharded_via_general` (which
    /// now calls it) — the recompaction logic exists ONCE.
    pub(crate) fn build_sharded_unified_exec_source(
        &self,
        table: &RelationalTable,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
    ) -> Result<ShardedUnifiedExecSource, ExecuteError> {
        // Load the table's resident shards in published order (sorted by (row_start, shard_id)).
        // Error text mirrors the retired probes.
        let mut shards = self
            .read_state
            .residency
            .shards
            .load()
            .get(&table.name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident shards",
                    table.name
                )))
            })?;
        if shards.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no resident shards",
                table.name
            ))));
        }
        // S-d3 zone-map pruning: for a point-lookup shape (`col = needle`, ANDed at top level) drop every
        // shard whose min/max zone map for that column excludes the needle — it cannot hold a matching row,
        // so the recompaction never gathers it. This turns the sharded read from O(num_shards) toward O(1).
        // SOUNDNESS: prune ONLY on a shard that carries a zone-map stat for the column AND provably excludes
        // the needle; a shard with no stat (e.g. a benchmark install) or an unresolved column is always kept.
        // The constraint's `col` is a FULL-CATALOG column index (from `ResidentExpr::Column`, built via
        // `relational_column_index`), so we resolve it to the column's real NAME through `table.columns` and
        // match the zone-map stat by name — the SAME catalog->int4-ordinal translation the resident read
        // offsets perform (`resident_device_int4_column_offset`). Indexing the int4-ordinal-compacted stat
        // list by the raw catalog index would read the WRONG column on a mixed-type table (non-int4 column
        // before the filter column) and could wrongly prune a matching shard. If pruning would drop EVERY
        // shard (needle in no range), keep the first shard so the recompaction machinery stays well-formed
        // and the device predicate returns the correct empty set.
        if let Some(pred) = predicate {
            let mut constraints: Vec<(usize, i32)> = Vec::new();
            mandatory_int4_equalities(pred, &mut constraints);
            if !constraints.is_empty() {
                // Catalog column names in catalog order — the constraints' `col` indexes into THIS.
                let column_names: Vec<&str> =
                    table.columns.iter().map(|c| c.name.as_str()).collect();
                let kept: Vec<RelationalResidentShard> = shards
                    .iter()
                    .filter(|shard| {
                        // Keep the shard unless SOME mandatory equality's zone map provably excludes it.
                        !constraints.iter().any(|(col, needle)| {
                            shard_zone_map_excludes(
                                &column_names,
                                &shard.resident_device_int4_column_stats,
                                *col,
                                *needle,
                            )
                        })
                    })
                    .cloned()
                    .collect();
                shards = if kept.is_empty() {
                    vec![shards[0].clone()]
                } else {
                    kept
                };
            }
        }
        self.read_state
            .residency
            .sharded_shards_gathered
            .fetch_add(shards.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let gpu_id = shards[0].gpu_id;
        let runtime_snapshot = self.router.runtime().snapshot();

        // Validate each shard + resolve its pinned device memory (the per-shard identity + validity
        // prechecks mirror the probe's at engine_resident_probe.rs ~873), or return the error a probe did.
        let source_for =
            |shard: &RelationalResidentShard| -> Result<ResidentExecSource, ExecuteError> {
                if shard.schema != table.schema || shard.table != table.name {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident shard no longer matches catalog table identity".to_string(),
                    )));
                }
                let memory_pressure_active = runtime_snapshot
                    .memory_pressured_gpu_ids
                    .contains(&shard.gpu_id);
                if !shard.is_valid(memory_pressure_active) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident shard {} is invalid",
                        shard.shard_id
                    ))));
                }
                // D4 (ADR-013 pre2): the buffer rides the loaded descriptor — the SAME generation
                // as the metadata by construction (no second map load to pair a stale descriptor
                // with a republished buffer).
                let device_memory = shard.device_memory.clone().ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident shard {} has no retained device memory",
                        shard.shard_id
                    )))
                })?;
                Ok(ResidentExecSource {
                    descriptor: Arc::new(self.resident_snapshot_for_shard(shard, table)),
                    device_memory,
                    row_count: shard.row_count as u64,
                })
            };

        // FLIP slice — ZERO-COPY single-shard fast path (measured: the per-read recompaction DtoD copy
        // costs ~440-460us at 524k rows vs the single-buffer's 19us COUNT — ledger #4). When the
        // zone-map prune leaves EXACTLY ONE shard and that shard is VERSION-FREE (no deleted_by /
        // created_by region — a version region lives in a SEPARATE device buffer, so it cannot be
        // addressed by an absolute offset inside the shard's own buffer), serve the shard's OWN buffer
        // directly: `resident_snapshot_for_shard` already describes it (capacity-strided, its own null
        // bitmaps), and every consumer reads through the descriptor's offset helpers. No DtoD, no
        // allocation. A versioned or multi-shard survivor set takes the recompaction below unchanged.
        if shards.len() == 1 {
            let shard = &shards[0];
            // D4: the version-free check reads the SAME loaded descriptor the buffer came from — a
            // concurrent re-admit purging the side maps can no longer fake version-freeness for a
            // reader still holding the old (tombstone-bearing) generation (the resurrection race).
            // D3 hwm gate: a created_by-only shard whose stamps are ALL <= the reader's boundary
            // (s >= max_created_by) is EFFECTIVELY VERSION-FREE for this reader — every row is
            // born-visible at s, so serving the raw buffer is exact. Only a reader pinned inside
            // an append window (s < hwm) falls through to the gated recompaction.
            let version_free = shard.deleted_by_region.is_none()
                && (shard.created_by_region.is_none() || copin_s >= shard.max_created_by);
            if version_free {
                let src = source_for(shard)?;
                return Ok(ShardedUnifiedExecSource {
                    src,
                    visibility: None,
                    gpu_id,
                });
            }
        }

        // The unified int4-only buffer lays the table's int4 columns out in catalog order, each contiguous
        // over `total_row_count` rows after the 8-byte row-count header — exactly the whole-table single
        // store the offset helpers expect. The recompaction indexes source slices POSITIONALLY by int4
        // ordinal and the unified descriptor labels the buffer with this list, so every shard MUST
        // carry the SAME `resident_device_int4_columns` (same names, same order) as shard 0 — otherwise
        // a slice would land in the wrong column's slot (silent wrong data) or read past a too-short source
        // (the DtoD primitive bounds-checks only the destination). The sharded benchmark install runs no
        // layout validation, so we enforce uniformity per shard in the gather loop below (audit F1).
        let int4_columns = shards[0].resident_device_int4_columns.clone();
        let num_int4_cols = int4_columns.len();
        // TYPE-COVERAGE track 2 slice 2: the i64 SECTION (Int8/Timestamp) recompacts exactly like
        // the i32 sections — same builder layout (header + all i32 sections + all i64 sections,
        // capacity-strided, NO padding: the offset helper + the 2x-u32 load discipline own the
        // 4-mod-8 case), same per-shard uniformity guard, same positional-ordinal DtoD plan.
        let int8_columns = shards[0].resident_device_int8_columns.clone();
        let num_int8_cols = int8_columns.len();
        // TYPE-COVERAGE #14 (numeric): the b128 (Numeric/Uuid) 16-byte section, recompacted after
        // the i64 sections into the unified buffer (else a numeric column read on a MULTI-SHARD
        // table would miss its bytes — a correctness hole, not an optimization).
        let numeric_columns = shards[0].resident_device_numeric_columns.clone();
        let num_numeric_cols = numeric_columns.len();

        // Gather each shard's (device_ptr, row_count) by running the SAME identity/validity precheck the
        // probes did (via `source_for`), accumulate `total_row_count`, and build the device-to-device copy
        // plan: for each int4 column ordinal `c` and each shard `p`, copy `p`'s slice of column `c`
        // (`8 + c*p.row_count*4`, len `p.row_count*4`) into the unified slot
        // (`8 + c*total_row_count*4 + rows_before_p*4`). Empty shards contribute a zero-length segment
        // (skipped by the primitive). The host never touches the column bytes.
        let mut shard_ptrs: Vec<(u64, usize, usize)> = Vec::with_capacity(shards.len());
        let mut total_row_count = 0_usize;
        for shard in &shards {
            let src = source_for(shard)?;
            // Uniformity guard (audit F1): the positional ordinal gather + the unified descriptor both
            // assume every shard's int4 layout equals shard 0's. Reject a mismatch with a clean
            // error rather than recompact a slice into the wrong column (or read past a short source).
            if shard.resident_device_int4_columns != int4_columns {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident shard {} int4 column layout {:?} does not match shard 0 layout {:?}",
                    shard.shard_id, shard.resident_device_int4_columns, int4_columns
                ))));
            }
            if shard.resident_device_int8_columns != int8_columns {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident shard {} int8 column layout {:?} does not match shard 0 layout {:?}",
                    shard.shard_id, shard.resident_device_int8_columns, int8_columns
                ))));
            }
            if shard.resident_device_numeric_columns != numeric_columns {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident shard {} b128 column layout {:?} does not match shard 0 layout {:?}",
                    shard.shard_id, shard.resident_device_numeric_columns, numeric_columns
                ))));
            }
            shard_ptrs.push((
                src.device_memory.device_ptr(),
                shard.row_count,
                shard.capacity,
            ));
            total_row_count = total_row_count.saturating_add(shard.row_count);
        }

        let mut segments: Vec<gpu_db_execution::RecompactSegment> =
            Vec::with_capacity(num_int4_cols * shards.len());
        for ordinal in 0..num_int4_cols {
            let mut rows_before = 0_u64;
            for (device_ptr, row_count, capacity) in &shard_ptrs {
                let row_count = *row_count as u64;
                let capacity = *capacity as u64;
                let byte_len = row_count.saturating_mul(4);
                segments.push(gpu_db_execution::RecompactSegment {
                    src_device_ptr: *device_ptr,
                    // S-d2: a column's live rows sit at its CAPACITY-strided start in the (possibly padded)
                    // shard (`8 + ordinal*capacity*4`); copy only the `row_count` live rows into the dense
                    // unified buffer. capacity == row_count for a dense/sealed shard (unchanged there).
                    src_byte_offset: 8 + (ordinal as u64) * capacity * 4,
                    dst_byte_offset: 8
                        + (ordinal as u64) * (total_row_count as u64) * 4
                        + rows_before * 4,
                    byte_len,
                });
                rows_before = rows_before.saturating_add(row_count);
            }
        }
        let int4_bytes = 8 + (total_row_count as u64) * 4 * (num_int4_cols as u64);
        // The i64 sections sit immediately after the i32 sections in BOTH the shard payloads and
        // the unified buffer (the shared offset-helper formula; per-shard stride = capacity,
        // unified stride = total_row_count).
        for ordinal in 0..num_int8_cols {
            let mut rows_before = 0_u64;
            for (device_ptr, row_count, capacity) in &shard_ptrs {
                let row_count = *row_count as u64;
                let capacity = *capacity as u64;
                segments.push(gpu_db_execution::RecompactSegment {
                    src_device_ptr: *device_ptr,
                    src_byte_offset: 8
                        + (num_int4_cols as u64) * capacity * 4
                        + (ordinal as u64) * capacity * 8,
                    dst_byte_offset: int4_bytes
                        + (ordinal as u64) * (total_row_count as u64) * 8
                        + rows_before * 8,
                    byte_len: row_count.saturating_mul(8),
                });
                rows_before = rows_before.saturating_add(row_count);
            }
        }
        let int8_end_bytes = int4_bytes + (total_row_count as u64) * 8 * (num_int8_cols as u64);
        // TYPE-COVERAGE #14 (numeric): the b128 sections sit immediately after the i64 sections in
        // BOTH the shard payloads (per-shard stride = capacity*16) and the unified buffer (stride =
        // total_row_count*16). The per-shard section base = 8 + num_int4*cap*4 + num_int8*cap*8.
        for ordinal in 0..num_numeric_cols {
            let mut rows_before = 0_u64;
            for (device_ptr, row_count, capacity) in &shard_ptrs {
                let row_count = *row_count as u64;
                let capacity = *capacity as u64;
                segments.push(gpu_db_execution::RecompactSegment {
                    src_device_ptr: *device_ptr,
                    src_byte_offset: 8
                        + (num_int4_cols as u64) * capacity * 4
                        + (num_int8_cols as u64) * capacity * 8
                        + (ordinal as u64) * capacity * 16,
                    dst_byte_offset: int8_end_bytes
                        + (ordinal as u64) * (total_row_count as u64) * 16
                        + rows_before * 16,
                    byte_len: row_count.saturating_mul(16),
                });
                rows_before = rows_before.saturating_add(row_count);
            }
        }
        let fixed_section_bytes =
            int8_end_bytes + (total_row_count as u64) * 16 * (num_numeric_cols as u64);

        // SV3b/SV6 MVCC visibility gather: if ANY surviving shard is VERSIONED (carries an on-demand
        // `deleted_by` and/or `created_by` region), append the corresponding co-resident DENSE i64 column(s)
        // to the unified buffer right after the int4 columns. The predicate then ANDs
        // `deleted_by > read_txn_id` (hide tombstoned rows) and/or `created_by <= read_txn_id` (hide
        // versions appended by a commit newer than the read snapshot — the SV5 double-read flip-gate). The
        // int4 descriptor/offsets are untouched (the executor reads the version columns by ABSOLUTE byte
        // offset via LoadColumnI64). The un-versioned majority allocates zero extra bytes and takes the
        // `None` visibility path -- byte-identical to the pre-SV3b read.
        let mut fills: Vec<gpu_db_execution::RecompactFill> = Vec::new();
        let mut deleted_by_offset: Option<u64> = None;
        let mut created_by_offset: Option<u64> = None;
        let mut allocated_bytes = fixed_section_bytes;
        // The two version-metadata regions (`deleted_by` upper bound, SV3b; `created_by` lower bound, SV6)
        // are gathered INDEPENDENTLY — an UPDATE tombstones the old version in one shard and appends the
        // stamped new version into the OPEN shard, so a shard can carry either region without the other.
        // Each region mirrors the same fill+segment pattern: FILL the whole unified column with the
        // "visible" sentinel (covers un-versioned shards' rows + any tail), then one DtoD segment per
        // region-bearing shard overwrites its own live rows. `rows_before` walks shards in the SAME
        // published order the int4 gather used, so the version rows line up 1:1 with the int4 rows.
        // D4: regions come from the SAME loaded descriptors as the buffers/metadata — one snapshot.
        type RegionOf = fn(&RelationalResidentShard) -> Option<&Arc<CudaResidentDeviceMemory>>;
        // D3 hwm gate: the created_by AXIS is needed only when SOME surviving shard carries a
        // stamp ABOVE the reader's boundary (s < hwm — a reader pinned inside an append window).
        // When every stamp is <= s the conjunct `created_by <= s` is identically true — skipping
        // the axis is exact, keeps `visibility: None` for insert-only tables at the newest
        // boundary, and thereby keeps the reshaping (DISTINCT/GROUP BY/ORDER BY/JOIN) shapes
        // served (their guards fire on `visibility.is_some()`). The deleted_by axis has no such
        // shortcut (a tombstone hides rows at ANY later boundary).
        let created_axis_needed = shards
            .iter()
            .any(|shard| shard.created_by_region.is_some() && copin_s < shard.max_created_by);
        let region_axes: [(RegionOf, u8, &mut Option<u64>, bool); 2] = [
            (
                |shard| shard.deleted_by_region.as_ref(),
                crate::engine_residency::DELETED_BY_LIVE_FILL_BYTE,
                &mut deleted_by_offset,
                true,
            ),
            (
                |shard| shard.created_by_region.as_ref(),
                crate::engine_residency::CREATED_BY_VISIBLE_FILL_BYTE,
                &mut created_by_offset,
                created_axis_needed,
            ),
        ];
        for (region_of, fill_byte, offset_out, axis_needed) in region_axes {
            let has_region = shards.iter().any(|shard| region_of(shard).is_some());
            if !has_region || !axis_needed {
                continue;
            }
            let region_offset = allocated_bytes;
            let region_bytes = (total_row_count as u64) * 8;
            fills.push(gpu_db_execution::RecompactFill {
                byte_offset: region_offset,
                len: region_bytes,
                fill_byte,
            });
            let mut rows_before = 0_u64;
            for shard in &shards {
                let row_count = shard.row_count as u64;
                if let Some(region) = region_of(shard) {
                    segments.push(gpu_db_execution::RecompactSegment {
                        src_device_ptr: region.device_ptr(),
                        src_byte_offset: 0,
                        dst_byte_offset: region_offset + rows_before * 8,
                        byte_len: row_count * 8,
                    });
                }
                rows_before = rows_before.saturating_add(row_count);
            }
            *offset_out = Some(region_offset);
            allocated_bytes += region_bytes;
        }
        let visibility: Option<ResidentVisibility> =
            if deleted_by_offset.is_some() || created_by_offset.is_some() {
                Some(ResidentVisibility {
                    read_txn_id: copin_s as i64,
                    deleted_by_offset,
                    created_by_offset,
                })
            } else {
                None
            };

        // M3-for-shards: recompact each column's NULL VALIDITY BITMAP into the unified buffer. A column gets a
        // unified bitmap iff SOME surviving shard carries one; the region is 1 bit/row (u32 words, LSB-first,
        // 1 = valid / 0 = NULL), placed 4-aligned after the int4 (+ version) sections. The executor reads it
        // by ABSOLUTE offset (`resident_device_null_column_offset`) and materializes `SqlValue::Null` for a
        // 0 bit, so the sharded scan stops reading a NULL-stored-0 placeholder as `0`. ADR-006 (NULL coverage):
        // a null-bearing table is now MULTI-shard (a NULL insert rolls a dense shard), so the region is
        // PRE-FILLED 0xFF for shards that elide an all-valid bitmap; each null-bearing shard's bits are
        // explicitly set/cleared with the ALIGNMENT-FREE per-bit gather kernel — the same
        // strategy as the bool bitmaps, since a shard's `rows_before` is generally not 32-row aligned and a
        // byte-copy would land its bits in the wrong destination word. No null-bearing shard -> zero extra
        // bytes, byte-identical read.
        let mut unified_null_columns: Vec<crate::relational_model::ResidentDeviceNullBitmapLayout> =
            Vec::new();
        let sidecar_u32 = |value: u64, field: &str| {
            u32::try_from(value).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "sharded resident sidecar {field} exceeds the CUDA u32 geometry"
                )))
            })
        };
        let sidecar_add = |left: u64, right: u64, field: &str| {
            left.checked_add(right).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "sharded resident sidecar {field} overflow"
                )))
            })
        };
        // (dst_bitmap_offset, dst_base_row, src_device_ptr, src_bitmap_offset, count) per (column, shard).
        let mut null_gather_ops: Vec<(u64, u32, u64, u64, u32)> = Vec::new();
        {
            // Union of null-bearing column names across surviving shards, in catalog order (deterministic).
            let null_col_names: Vec<String> = table
                .columns
                .iter()
                .filter(|column| {
                    shards.iter().any(|s| {
                        s.resident_device_null_columns
                            .iter()
                            .any(|n| n.name == column.name)
                    })
                })
                .map(|column| column.name.clone())
                .collect();
            let words_per_col = (total_row_count as u64).div_ceil(32);
            let col_region_bytes = words_per_col * 4;
            for name in &null_col_names {
                let col_offset = allocated_bytes;
                // Born all-valid (0xFF): a row/shard without a bitmap for this column is non-NULL.
                fills.push(gpu_db_execution::RecompactFill {
                    byte_offset: col_offset,
                    len: col_region_bytes,
                    fill_byte: 0xFF,
                });
                let mut rows_before = 0_u64;
                for (shard, (device_ptr, row_count, _capacity)) in
                    shards.iter().zip(shard_ptrs.iter())
                {
                    let row_count = *row_count as u64;
                    if let Some(layout) = shard
                        .resident_device_null_columns
                        .iter()
                        .find(|n| &n.name == name)
                    {
                        // Alignment-free per-bit repack (like bool): the kernel places source local row `l`
                        // at unified row `rows_before + l`, writing both validity states — no 32-row-alignment
                        // requirement on `rows_before`, so a null-bearing shard at any offset is correct.
                        if row_count > 0 {
                            let dst_base = sidecar_u32(rows_before, "NULL destination base")?;
                            let count = sidecar_u32(row_count, "NULL row count")?;
                            null_gather_ops.push((
                                col_offset,
                                dst_base,
                                *device_ptr,
                                layout.bitmap_byte_offset,
                                count,
                            ));
                        }
                    }
                    rows_before = sidecar_add(rows_before, row_count, "NULL row base")?;
                }
                unified_null_columns.push(
                    crate::relational_model::ResidentDeviceNullBitmapLayout {
                        name: name.clone(),
                        bitmap_byte_offset: col_offset,
                    },
                );
                allocated_bytes = allocated_bytes.saturating_add(col_region_bytes);
            }
        }
        // TYPE-COVERAGE #14 (bool): recompact each bool column's 1-bit/row bitmap into the unified buffer.
        // Unlike the NULL bitmaps (sparse) or the fixed-width sections (byte-copyable at any boundary), a
        // bool bitmap CANNOT be byte-concatenated across shards: shards seal at ARBITRARY row counts (the
        // first dense admit seals at exactly its row count, e.g. 100), so a shard's `rows_before` is
        // generally not 32-row aligned and its bits would land in the wrong destination word. So the region
        // is defensively PRE-ZEROED here (RecompactFill 0x00); after DtoD recompaction each shard's bits are
        // repacked into place by a per-bit gather KERNEL (`gather_bool_bitmap_from_shard`) that reads source
        // bit `l` and atomically writes that state at `rows_before + l` — alignment-free. The executor reads the
        // region by ABSOLUTE offset (`resident_device_bool_column_offset`), like the NULL bitmaps.
        let mut unified_bool_columns: Vec<crate::relational_model::ResidentDeviceBoolColumnLayout> =
            Vec::new();
        // (dst_bitmap_offset, dst_base_row, src_device_ptr, src_bitmap_offset, count) per (column, shard).
        let mut bool_gather_ops: Vec<(u64, u32, u64, u64, u32)> = Vec::new();
        {
            let words_per_col = (total_row_count as u64).div_ceil(32);
            let col_region_bytes = words_per_col * 4;
            let bool_names: Vec<String> = shards[0]
                .resident_device_bool_columns
                .iter()
                .map(|b| b.name.clone())
                .collect();
            for name in &bool_names {
                let col_offset = allocated_bytes;
                fills.push(gpu_db_execution::RecompactFill {
                    byte_offset: col_offset,
                    len: col_region_bytes,
                    fill_byte: 0x00,
                });
                let mut rows_before = 0_u64;
                for (shard, (device_ptr, row_count, _capacity)) in
                    shards.iter().zip(shard_ptrs.iter())
                {
                    let row_count = *row_count as u64;
                    let Some(layout) = shard
                        .resident_device_bool_columns
                        .iter()
                        .find(|b| &b.name == name)
                    else {
                        // A shard missing a bool column its siblings carry => layout skew => decline.
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "sharded bool recompaction: shard {} lacks bool column \"{name}\"",
                            shard.shard_id
                        ))));
                    };
                    if row_count > 0 {
                        let dst_base = sidecar_u32(rows_before, "bool destination base")?;
                        let count = sidecar_u32(row_count, "bool row count")?;
                        bool_gather_ops.push((
                            col_offset,
                            dst_base,
                            *device_ptr,
                            layout.bitmap_byte_offset,
                            count,
                        ));
                    }
                    rows_before = sidecar_add(rows_before, row_count, "bool row base")?;
                }
                unified_bool_columns.push(
                    crate::relational_model::ResidentDeviceBoolColumnLayout {
                        name: name.clone(),
                        bitmap_byte_offset: col_offset,
                    },
                );
                allocated_bytes = allocated_bytes.saturating_add(col_region_bytes);
            }
        }
        // TYPE-COVERAGE #14 (text): recompact each TEXT column into the unified buffer. Text can't
        // byte-concatenate directly across shards: each shard's offsets are RELATIVE to its own bytes
        // blob. So the unified layout is, per column, an 8-aligned offsets section (total+1 i64) then the
        // concatenated bytes blob; the blobs byte-copy at a running blob_base (RecompactSegment) and a
        // per-element REBASE kernel adds each shard's blob_base to its offsets. Executor reads via the text
        // offset helper (offsets_byte_offset + bytes_byte_offset), same layout as the single buffer.
        let mut unified_text_columns: Vec<crate::relational_model::ResidentDeviceTextColumnLayout> =
            Vec::new();
        // (dst_offsets_byte_offset, dst_base_row, blob_base, src_ptr, src_offsets_byte_offset,
        //  src_bytes_byte_offset, src_blob_len, count).
        let mut text_rebase_ops: Vec<TextRebaseOp> = Vec::new();
        {
            let text_names: Vec<String> = shards[0]
                .resident_device_text_columns
                .iter()
                .map(|t| t.name.clone())
                .collect();
            for name in &text_names {
                while !allocated_bytes.is_multiple_of(8) {
                    allocated_bytes += 1;
                }
                let offsets_byte_offset = allocated_bytes;
                let offsets_bytes = (total_row_count as u64 + 1) * 8;
                allocated_bytes = allocated_bytes.saturating_add(offsets_bytes);
                let bytes_byte_offset = allocated_bytes;
                // Every offset entry IS written by the rebase (the shard ranges tile [0..total]); the
                // fill is a defensive pre-zero (a skipped/empty shard leaves no garbage gap).
                fills.push(gpu_db_execution::RecompactFill {
                    byte_offset: offsets_byte_offset,
                    len: offsets_bytes,
                    fill_byte: 0,
                });
                let mut rows_before = 0_u64;
                let mut blob_base = 0_u64;
                for (shard, (device_ptr, row_count, _capacity)) in
                    shards.iter().zip(shard_ptrs.iter())
                {
                    let row_count = *row_count as u64;
                    let Some(layout) = shard
                        .resident_device_text_columns
                        .iter()
                        .find(|t| &t.name == name)
                    else {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "sharded text recompaction: shard {} lacks text column \"{name}\"",
                            shard.shard_id
                        ))));
                    };
                    if row_count > 0 {
                        if layout.bytes_len > 0 {
                            segments.push(gpu_db_execution::RecompactSegment {
                                src_device_ptr: *device_ptr,
                                src_byte_offset: layout.bytes_byte_offset,
                                dst_byte_offset: bytes_byte_offset + blob_base,
                                byte_len: layout.bytes_len,
                            });
                        }
                        // Rebase this shard's (row_count+1) offsets into [rows_before .. +row_count].
                        let dst_base = sidecar_u32(rows_before, "text destination base")?;
                        let offset_count = sidecar_add(row_count, 1, "text offset count")?;
                        let count = sidecar_u32(offset_count, "text offset count")?;
                        text_rebase_ops.push((
                            offsets_byte_offset,
                            dst_base,
                            blob_base,
                            *device_ptr,
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            layout.bytes_len,
                            count,
                        ));
                    }
                    rows_before = sidecar_add(rows_before, row_count, "text row base")?;
                    blob_base = sidecar_add(blob_base, layout.bytes_len, "text blob base")?;
                }
                allocated_bytes = allocated_bytes.saturating_add(blob_base);
                unified_text_columns.push(
                    crate::relational_model::ResidentDeviceTextColumnLayout {
                        name: name.clone(),
                        offsets_byte_offset,
                        bytes_byte_offset,
                        bytes_len: blob_base,
                    },
                );
            }
        }
        let header = (total_row_count as u64).to_le_bytes();

        // Recompact ON-DEVICE into one unified buffer, then build the whole-table descriptor + injected
        // source the executor runs over ONCE.
        let runtime = self.cuda_driver_probe_runtime();
        let unified_mem = runtime
            .retain_device_memory_recompacted(gpu_id, allocated_bytes, &header, &fills, &segments)
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "sharded resident recompaction into a unified device buffer failed: {err}"
                )))
            })?;
        let sidecar_source = |device_ptr: u64, byte_offset: u64| {
            shards
                .iter()
                .find_map(|shard| {
                    shard.device_memory.as_ref().and_then(|memory| {
                        (memory.device_ptr() == device_ptr).then(|| CudaSidecarSource {
                            memory: Arc::clone(memory),
                            byte_offset,
                        })
                    })
                })
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "sidecar source generation is no longer owned".to_string(),
                    ))
                })
        };
        // TYPE-COVERAGE #14 (bool): repack each shard's bool bits into the unified regions with
        // the alignment-free per-bit gather kernel (DtoD, no HtoD). Runs after the DtoD recompaction so the
        // unified buffer + shard buffers are both live; a failure declines the whole sharded read.
        for (dst_off, dst_base, src_ptr, src_off, count) in &bool_gather_ops {
            let source = sidecar_source(*src_ptr, *src_off)?;
            unified_mem
                .gather_bool_bitmap_from_shard(*dst_off, *dst_base, &source, *count)
                .map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sharded bool bitmap gather kernel failed: {err}"
                    )))
                })?;
        }
        // ADR-006 (NULL coverage): repack each null-bearing shard's validity bits into the unified regions
        // with the alignment-free per-bit gather kernel (DtoD). The 0xFF fill covers shards that elide an
        // all-valid bitmap. A failure declines the read.
        for (dst_off, dst_base, src_ptr, src_off, count) in &null_gather_ops {
            let source = sidecar_source(*src_ptr, *src_off)?;
            unified_mem
                .gather_null_bitmap_from_shard(*dst_off, *dst_base, &source, *count)
                .map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sharded null bitmap gather kernel failed: {err}"
                    )))
                })?;
        }
        // TYPE-COVERAGE #14 (text): rebase each shard's offsets into the unified offsets section (DtoD,
        // after the blob byte-copies above). A failure declines the whole sharded read.
        for (
            dst_off,
            dst_base,
            blob_base,
            src_ptr,
            src_off,
            src_bytes_off,
            src_blob_len,
            count,
        ) in &text_rebase_ops
        {
            let source = sidecar_source(*src_ptr, *src_off)?;
            let source = CudaTextOffsetSource {
                memory: source.memory,
                offsets_byte_offset: source.byte_offset,
                bytes_byte_offset: *src_bytes_off,
                bytes_len: *src_blob_len,
            };
            unified_mem
                .rebase_text_offsets_from_shard(
                    *dst_off, *dst_base, *blob_base, &source, *count,
                )
                .map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sharded text offset rebase kernel failed: {err}"
                    )))
                })?;
        }
        let proof = unified_mem.metadata().clone();
        let snapshot = self.resident_snapshot_for_unified(
            table,
            crate::engine_residency::UnifiedResidentSnapshotParts {
                total_row_count,
                gpu_id,
                resident_bytes: allocated_bytes,
                proof,
                int4_columns,
                int8_columns,
                numeric_columns,
                bool_columns: unified_bool_columns,
                text_columns: unified_text_columns,
                null_columns: unified_null_columns,
            },
        );
        Ok(ShardedUnifiedExecSource {
            src: ResidentExecSource {
                descriptor: Arc::new(snapshot),
                device_memory: Arc::new(unified_mem),
                row_count: total_row_count as u64,
            },
            visibility,
            gpu_id,
        })
    }

    /// S10c slice 2a: `&Select`->general BRIDGE for the MULTI-PARTITION resident shapes (single-GPU,
    /// int4-only). RECOMPACTS the table's shard buffers into ONE unified int4-only
    /// `CudaResidentDeviceMemory` via `build_sharded_unified_exec_source` (device-to-device copies — the
    /// host stays fully out; only the 8-byte row-count header crosses HtoD), then runs the SAME on-device
    /// general resident-Expr executor ONCE over the unified buffer.
    ///
    /// All-empty handling: the general SUM/MIN/MAX/AVG HARD-ERROR on an empty filtered set (NULL-on-empty
    /// is an unfinished M3 feature for the general path). So we first run a `COUNT(*)` over the unified
    /// buffer; if it is 0 AND the projection is an aggregate, we return the PG-correct empty value WITHOUT
    /// the hard error: SUM/AVG/MIN/MAX -> `SqlValue::Null` (an aggregate of no rows is NULL), COUNT(*) ->
    /// `Int8(0)`. (Was the legacy empty-text/zero sentinel; PG-correctness wins -- `sql-spec-over-cpu-parity`.)
    /// Otherwise the executor runs ONCE over the unified buffer with the real
    /// projection and its result is returned directly (it handles COUNT/SUM/MIN/MAX/AVG/projection
    /// on-device). Text columns are DEFERRED in this slice (the unified buffer is int4-only).
    pub(crate) fn execute_resident_sharded_via_general(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, mut bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        // Sub-slice 3b: detect a single top-level int4 `Eq` POINT-lookup shape from the (still-populated)
        // bound BEFORE the filters are cleared below — the precondition for the cross-shard PK-index route.
        // Mirrors the retained-read (lpb) route's shape gate: exactly one equality group of one, `Eq`, an
        // int4 filter column + an int4 needle. `None` = not a point lookup -> the scan path runs unchanged.
        let point_lookup_eq: Option<(usize, i32)> = shard_point_lookup_int4_eq(&bound, &table);
        // Rebuild the WHERE predicate from the bound filters, then clear them so the executor filters
        // SOLELY via the predicate (the SQL->Expr contract), exactly as the grouped bridge does.
        let predicate = resident_predicate_from_bound_filters(&bound)?;
        bound.filter = None;
        bound.filters.clear();
        bound.filter_groups.clear();
        // Sub-slice 3b: try the CROSS-SHARD PK-INDEX point-lookup route (flag-gated, DEFAULT OFF). On a hit
        // it uses the cached hash+bloom `locate` to jump straight to the (shard, slot) and gather ONLY that
        // row (a few tiny DtoH reads), skipping the zone-map scan + recompaction below; on ANY shape or
        // soundness guard it returns None and we fall through to the scan (byte-identical). Placed before the
        // zone-map prune so it also wins when zone maps DEGRADE under UPDATE key-scatter (membership pruning,
        // scalability-ledger #4/#8) — locate finds the exact shard even when [min,max] can't exclude any.
        // M3-for-shards: the point-index route gathers RAW i32 slots (no validity bitmap), so it would read a
        // NULL-stored-0 as 0 while the scan below is now NULL-aware -> SKIP it for a null-bearing table so the
        // NULL-aware scan serves it. (ADR-006: a null-bearing table can now be MULTI-shard — a NULL insert
        // rolls a dense shard — so this may forgo the many-shard point-index win for such tables; correctness
        // first.) The `..._null_blind_matches_scan` differential is the tripwire that this decline keeps route == scan.
        // (The null check runs on its own lightweight shards load; the route re-validates internally against
        // its own generation-consistent capture, and the unified gather below is NULL-aware regardless.)
        if let Some((filter_idx, needle)) = point_lookup_eq {
            let any_shard_has_nulls = self
                .read_state
                .residency
                .shards
                .load()
                .get(&table.name)
                .is_some_and(|shards| {
                    shards
                        .iter()
                        .any(|s| !s.resident_device_null_columns.is_empty())
                });
            if !any_shard_has_nulls {
                if let Some(result) = self.try_shard_index_point_route(
                    select, &table, &bound, filter_idx, needle, copin_s,
                ) {
                    return Ok(result);
                }
            }
        }
        // FLIP slice — METADATA COUNT fast path (measured: the unpredicated sharded COUNT(*) paid the
        // full recompaction DtoD, p50 ~430-530us at 524k rows vs single-buffer's 19us). An unpredicated
        // COUNT(*) over ALL-version-free shards is exactly `sum(shard.row_count)` — descriptor metadata
        // the route planner already reads (control plane; no row data touched, no kernel, no copy). Any
        // version region (a tombstone could hide rows / a stamp could hide appended versions) or any
        // predicate falls through to the device path unchanged.
        if matches!(select.projection, SelectProjection::CountAll) && predicate.is_none() {
            let shards_guard = self.read_state.residency.shards.load();
            if let Some(shards) = shards_guard.get(&table.name) {
                // D4: read the version-freeness from the SAME loaded descriptors being summed —
                // the metadata COUNT can no longer pair an old shard list with freshly-purged maps.
                // D3 hwm gate: created_by-only shards whose stamps are all <= the reader's boundary
                // count every row (effectively version-free); a reader pinned inside an append
                // window (copin_s < hwm) falls through to the gated device path.
                // W0c (audit B1): ALSO require every shard VALID — this executor-side load can be
                // NEWER than the accepted route plan's (a concurrent commit flags + publishes in
                // between), and a flagged shard's row_count excludes the host-installed rows the
                // reader's pinned boundary includes. An invalid shard falls through to the gated
                // device path, whose source_for declines and the statement re-serves from the CPU.
                let runtime_snapshot = self.router.runtime().snapshot();
                let version_free = shards.iter().all(|shard| {
                    shard.is_valid(
                        runtime_snapshot
                            .memory_pressured_gpu_ids
                            .contains(&shard.gpu_id),
                    ) && shard.deleted_by_region.is_none()
                        && (shard.created_by_region.is_none() || copin_s >= shard.max_created_by)
                });
                if version_free && !shards.is_empty() {
                    let total: i64 = shards.iter().map(|s| s.row_count as i64).sum();
                    let gpu_id = shards[0].gpu_id;
                    drop(shards_guard);
                    let (_query, access_path) =
                        self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
                    return Ok(RelationalSelectResult {
                        columns: Arc::new(bound.selected_columns.clone()),
                        rows: vec![vec![SqlValue::Int8(total)]].into(),
                        planned_target: DeviceTarget::Gpu(gpu_id),
                        executed_target: DeviceTarget::Gpu(gpu_id),
                        fallback_reason: None,
                        access_path: Arc::new(access_path),
                    });
                }
            }
        }
        // SLICE B: the shard load + zone-map prune + int4/version/null-bitmap recompaction live in
        // `build_sharded_unified_exec_source`, SHARED with the SQL->Expr PG path so IS NULL and every
        // other general-executor shape run over sharded tables through the SAME on-device execution.
        let unified =
            self.build_sharded_unified_exec_source(&table, predicate.as_ref(), copin_s)?;
        let gpu_id = unified.gpu_id;
        let visibility = unified.visibility;
        let unified_src = unified.src;

        // Run one (already-bound) select against an injected source via the general executor. `vis` is the
        // SV3b/SV6 MVCC visibility descriptor for the unified buffer (`Some` when any surviving shard is
        // versioned, else `None`) -- forwarded so the on-device predicate ANDs the visibility bound(s)
        // (`deleted_by > read_txn_id`, `created_by <= read_txn_id`) and hides invisible versions. BOTH the
        // COUNT precheck and the real run pass the SAME `vis` so the count and the projection agree on
        // which rows are visible.
        let run = |select_ref: &Select,
                   bound_for_select: BoundRelationalSelect,
                   src: &ResidentExecSource,
                   vis: Option<ResidentVisibility>|
         -> Result<RelationalSelectResult, ExecuteError> {
            self.execute_resident_expr_select_with_binding(
                select_ref,
                &table,
                Some(src),
                bound_for_select,
                copin_s,
                predicate.as_ref(),
                vis,
                &[],
                &[],
                None,
                &[],
            )
        };

        // The access path is shape/table-level metadata (it does not depend on the shard data), so
        // compute it once from the (filter-cleared) bound exactly as a probe did (engine_resident_probe.rs
        // ~846). bound was filter-cleared above; the sharded shapes carry no ORDER BY / LIMIT.
        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;

        let finalize = |rows: Vec<Vec<SqlValue>>| -> RelationalSelectResult {
            RelationalSelectResult {
                columns: Arc::new(bound.selected_columns.clone()),
                rows: rows.into(),
                planned_target: DeviceTarget::Gpu(gpu_id),
                executed_target: DeviceTarget::Gpu(gpu_id),
                fallback_reason: None,
                access_path: Arc::new(access_path.clone()),
            }
        };

        // S10c slice 2b: DISTINCT / GROUP BY / ORDER-BY-projection are CORRECT over the unified buffer (it
        // holds the WHOLE table), so route each to the grouped/distinct sub-bridge with the unified source
        // injected. These sub-bridges re-bind + re-derive the WHERE predicate INTERNALLY from `select`, so
        // they need only `select` + `Some(&unified_src)` (the outer filter-cleared `bound` is unused by
        // them). Order matters: DISTINCT carries no `group_by` but synthesizes one internally, so it must be
        // checked first; a grouped select may ALSO carry ORDER BY and must take the grouped path. The plain
        // scalar/projection shapes fall through to the COUNT-precheck + single run below (unchanged).
        //
        // R-ver PART 2: the reshaping sub-bridges are visibility-correct — they route back into
        // `execute_resident_expr_select_with_binding`, whose survivor `indices` fold in the SV3b/SV6
        // visibility BEFORE group/sort/dedup (verified: every grouped/distinct/ordered kernel reads
        // only `indices`/`indices_u64`). So thread the versioned buffer's `visibility` through to
        // them instead of refusing. (`visibility` is `None` for a version-free buffer -> byte-identical.)
        if select.distinct {
            return self.execute_resident_distinct_via_general(
                select,
                Some(&unified_src),
                visibility,
            );
        }
        if select.group_by.is_some() {
            return self.execute_resident_grouped_via_general(
                select,
                Some(&unified_src),
                visibility,
            );
        }
        if !select.order_by.is_empty() {
            return self.execute_resident_grouped_via_general(
                select,
                Some(&unified_src),
                visibility,
            );
        }

        // All-empty handling (pins byte-identicality with slice 1): the general SUM/MIN/MAX/AVG hard-error
        // on an empty filtered set, so first run a COUNT(*) over the unified buffer; if it is 0 AND the
        // projection is an aggregate, return the SAME placeholder slice 1 did.
        let count_select = {
            let mut s = select.clone();
            s.projection = SelectProjection::CountAll;
            s
        };
        let count_bound = bind_relational_select(&table, &count_select)?;
        let matched = {
            let result = run(&count_select, count_bound, &unified_src, visibility)?;
            match result.rows.iter().next().and_then(|row| row.first()) {
                Some(SqlValue::Int8(n)) => *n,
                other => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sharded resident COUNT(*) precheck returned an unexpected value: {other:?}"
                    ))));
                }
            }
        };
        if matched == 0 {
            // PG: SUM/AVG/MIN/MAX over zero rows is NULL (never 0 or an empty-text sentinel); only
            // COUNT(*) is 0. The old non-NULL placeholders were legacy CPU-engine parity (interim debt,
            // ADR-006); PG-correctness wins (see the `sql-spec-over-cpu-parity` working agreement).
            let placeholder = match &select.projection {
                SelectProjection::Min { .. }
                | SelectProjection::Max { .. }
                | SelectProjection::Avg { .. }
                | SelectProjection::Sum { .. } => Some(SqlValue::Null),
                SelectProjection::CountAll => Some(SqlValue::Int8(0)),
                _ => None,
            };
            if let Some(cell) = placeholder {
                return Ok(finalize(vec![vec![cell]]));
            }
        }

        // Run the executor ONCE over the unified buffer with the real projection — it handles
        // COUNT / SUM / MIN / MAX / AVG / projection on-device — and return its result directly.
        run(select, bound.clone(), &unified_src, visibility)
    }

    /// Sub-slice 3b: the CROSS-SHARD PK-INDEX point-lookup route for the sharded read path. For a
    /// shard-resident int4 UNIQUE-key equality POINT lookup with a PLAIN int4-column projection, use the
    /// cached hash+bloom `locate` to jump straight to the located `(shard, slot)` and gather ONLY that row
    /// (a few tiny DtoH reads) instead of scanning + recompacting the whole (zone-map-pruned) shard — the
    /// per-shard scan is ~half of a point-lookup's latency at the 4M production shard size (MEASURED: point
    /// p50 131k=343us -> 4M=709us at avg-gathered 1.00, so the rise is the one gathered shard's scan).
    ///
    /// Returns `Some(result)` — BYTE-IDENTICAL to the scan (`columns` / `access_path` / target computed the
    /// SAME way as the scan's `finalize`) — when it served the read, or `None` to FALL BACK to the existing
    /// scan + recompaction (the caller runs it unchanged) on ANY shape or soundness guard: never a wrong
    /// result. Guards (each -> None -> scan): the flag is OFF; the projection is not a plain `All`/`Columns`
    /// of int4-only columns, or carries DISTINCT / GROUP BY / ORDER BY / LIMIT / OFFSET / HAVING; `locate`
    /// declines (duplicate / oversize / invalid shard); more than one hit (a cross-shard duplicate -> the
    /// scan applies its multi-row + ordering); or the located shard is gone / invalid / raced (`slot >=
    /// row_count`).
    ///
    /// NULLs (M3-for-shards): the sharded SCAN is now NULL-AWARE (its recompaction rebuilds each column's
    /// validity bitmap into the unified buffer + labels the unified descriptor), but this route gathers RAW i32
    /// slots with no validity channel. So the CALLER SKIPS this route entirely for a null-bearing table (any
    /// surviving shard with a non-empty `resident_device_null_columns`) -> the NULL-aware scan serves it. null-
    /// bearing is single-shard by construction, so the skip never costs the many-shard route. The
    /// `deleted_by[slot] > read_txn_id` visibility gate mirrors the scan's SV3b filter — a tombstoned row
    /// materializes ZERO rows. (`cross_shard_pk_index_route_declines_on_null_bearing` is the tripwire that this
    /// decline keeps route == the NULL-aware scan.)
    fn try_shard_index_point_route(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: &BoundRelationalSelect,
        filter_idx: usize,
        needle: i32,
        copin_s: Index,
    ) -> Option<RelationalSelectResult> {
        if !self.shard_index_probe_enabled() {
            return None;
        }
        // Plain int4-column projection ONLY: `All`/`Columns` (both resolve to `selected_indexes`), no
        // aggregate / DISTINCT / GROUP BY / ORDER BY / LIMIT / OFFSET / HAVING — anything else is the scan's.
        if !matches!(
            select.projection,
            SelectProjection::All | SelectProjection::Columns(_)
        ) || select.distinct
            || select.group_by.is_some()
            || !select.order_by.is_empty()
            || select.limit.is_some()
            || select.offset.is_some()
            || !select.having_groups.is_empty()
        {
            return None;
        }
        if bound.selected_indexes.is_empty() {
            return None;
        }
        for &idx in &bound.selected_indexes {
            if table.columns.get(idx).map(|c| c.ty) != Some(SqlType::Int4) {
                return None;
            }
        }
        // Locate the (shard_id, slot) via the cached hash+bloom index. `None` = the index declined (a
        // duplicate / oversize / invalidated shard) -> scan. `Some(hits)`: 0 hits = key absent everywhere
        // (0 rows), 1 hit = the row, >1 = a cross-shard duplicate -> scan (the scan returns every match).
        let hits = self.locate_resident_pk_via_shard_index_detailed(table, filter_idx, needle)?;
        if hits.len() > 1 {
            return None;
        }
        // gpu_id is a result LABEL (shape metadata, not shard data), so a lightweight lock-free load is fine
        // — the DATA read below uses the generation-consistent handles captured inside the hit, which is what
        // closes the concurrent TOCTOU (a stale gpu_id label on the same single GPU is harmless).
        let gpu_id = self
            .read_state
            .residency
            .shards
            .load()
            .get(&table.name)?
            .first()?
            .gpu_id;
        // `access_path` + `columns` are shape/table-level metadata (independent of the shard data), computed
        // EXACTLY as the scan's `finalize` does (from the filter-cleared `bound`) so the result is
        // byte-identical to the scan's. The transient read pin is dropped immediately (RAII).
        let (_query, access_path) = self
            .relational_select_mvcc_query_pinned(select, table, bound, copin_s)
            .ok()?;

        let rows: Vec<Vec<SqlValue>> = if let Some(hit) = hits.first() {
            // GENERATION-CONSISTENT materialization (audit fix). The slot, the capacity-stride
            // (`hit.descriptor`), the int4 buffer (`hit.device_memory`, PINNED by the Arc), and the
            // `deleted_by` region were ALL captured in the SAME `shards.load()` snapshot inside
            // `locate...detailed`. Reading the slot out of `hit`'s pinned buffer (NOT a fresh
            // `shard_device_memory.get`) closes the concurrent TOCTOU: reads are lock-free and straddle
            // commits, so a concurrent DELETE re-admit can republish a shard_id's buffer with COMPACTED /
            // reordered slots; resolving the slot against one generation and re-`get`-ing the buffer (a
            // second independent `ArcSwap` load) could read the slot out of a DIFFERENT generation -> a wrong
            // row. Holding the exact buffer the slot indexes into makes that impossible. (locate already ran
            // the identity / is_valid / memory-pressure prechecks before capturing the hit.)
            let slot = hit.slot as u64;
            if hit.slot as usize >= hit.descriptor.row_count {
                return None; // defensive: slot past the captured live region
            }
            // NULL handling: the sharded read path is uniformly NULL-BLIND (NULL stored as 0). Its
            // recompaction is int4-only with NO validity-bitmap segment, and BOTH descriptors it builds --
            // `resident_snapshot_for_shard` AND the scan's `resident_snapshot_for_unified` -- carry
            // `resident_device_null_columns: Vec::new()`, so the scan reads the raw i32 (a NULL reads back as
            // 0, and `col = 0` MATCHES a NULL-stored-0 row). This raw-i32 slot gather is therefore
            // BYTE-IDENTICAL to the scan on NULLs BY CONSTRUCTION (the `..._null_blind_matches_scan`
            // differential proves it; it is the tripwire when M3 recompacts bitmaps through the sharded path).
            //
            // SV3b visibility gate: a VERSIONED shard's `deleted_by[slot]` (dense i64 at byte `slot*8`, NO
            // header, little-endian) HIDES the row when `deleted_by <= read_txn_id`; an un-versioned shard (no
            // region) is all-live. `copin_s as i64` is the read snapshot, exactly the scan's `vis`. The region
            // is the SAME-generation handle captured in the hit.
            let read_i64_at_slot = |region: &Arc<CudaResidentDeviceMemory>| -> Option<i64> {
                let halves = region.read_resident_i32_column(slot * 8, 2).ok()?;
                Some(
                    ((*halves.first()? as u32 as u64) | ((*halves.get(1)? as u32 as u64) << 32))
                        as i64,
                )
            };
            let visible = match &hit.deleted_by {
                Some(region) => read_i64_at_slot(region)? > copin_s as i64,
                None => true,
            }
            // SV6 lower bound: a `created_by`-versioned shard's `created_by[slot]` HIDES the row when it
            // exceeds the read snapshot (an UPDATE-appended version whose commit this reader must not see —
            // the double-read gate), mirroring the scan's `created_by <= read_txn_id` conjunct. An
            // un-stamped shard (no region) is born-visible.
            && match &hit.created_by {
                Some(region) => read_i64_at_slot(region)? <= copin_s as i64,
                None => true,
            };
            if visible {
                // Materialize the projected row: one tiny DtoH per projected int4 column at its
                // capacity-strided slot byte (`col_base + slot*4`) in the PINNED buffer. Column order =
                // `selected_indexes` = the scan's projection order, so the row is byte-identical to the scan's.
                let mut row = Vec::with_capacity(bound.selected_indexes.len());
                for &idx in &bound.selected_indexes {
                    let col_base =
                        resident_device_int4_column_offset(&hit.descriptor, table, idx).ok()?;
                    let v = hit
                        .device_memory
                        .read_resident_i32_column(col_base + slot * 4, 1)
                        .ok()?;
                    row.push(SqlValue::Int4(*v.first()?));
                }
                vec![row]
            } else {
                Vec::new()
            }
        } else {
            // All shards Miss -> the key is absent -> zero rows (the scan returns the same empty set).
            Vec::new()
        };

        self.read_state
            .residency
            .shard_index_route_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns.clone()),
            rows: rows.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(access_path),
        })
    }

    /// S10b: `&Select`->general BRIDGE for a single-column `SELECT DISTINCT` (the int4_[filtered_]distinct
    /// route shapes). DISTINCT over column `a` is exactly `GROUP BY a` with no aggregate; the most-audited
    /// on-device grouped path (S8) is built around >=1 aggregate, so synthesize a `COUNT(*)` grouped select,
    /// run it through `execute_resident_grouped_via_general` (one row per distinct key, ON THE DEVICE -- the
    /// retired probe deduped on a HOST `BTreeSet`), then DROP the trailing COUNT column. WHERE / ORDER BY a /
    /// LIMIT / OFFSET ride the grouped select unchanged. **PG-correct on NULLs** (a behavior change vs the
    /// retired probe, like the S10a ordered projection): GROUP BY groups a NULL key into one group (M3
    /// NULL-key slot) -> DISTINCT yields ONE `SqlValue::Null` row, whereas the NULL-blind probe read the int4
    /// column directly and surfaced a NULL as a phantom `Int4(0)`. For the FILTERED shape a NULL fails the
    /// range predicate (3VL) so it is excluded either way. **No-ORDER-BY order:** the probe returned
    /// first-seen order; the grouped path returns the deterministic default order (key ASC) -- same SET, a
    /// PG-unspecified sequence made deterministic (like the S8 tie-break).
    pub(crate) fn execute_resident_distinct_via_general(
        &self,
        select: &Select,
        // `None` = look up the table's whole-table single resident store by name (the original callers).
        // `Some(src)` forwards an injected source to the grouped bridge it synthesizes (S10c slice 2b:
        // the unified multi-shard buffer), so DISTINCT runs on-device over the whole table.
        src: Option<&ResidentExecSource>,
        // R-ver PART 2: the versioned unified src's visibility, forwarded to the grouped bridge so
        // the distinct SET is over VISIBLE rows only. `None` for a `src: None` caller.
        visibility: Option<ResidentVisibility>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let SelectProjection::Columns(columns) = &select.projection else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident DISTINCT bridge requires a column projection".to_string(),
            )));
        };
        let [column] = columns.as_slice() else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident DISTINCT bridge currently supports a single projected column".to_string(),
            )));
        };
        // SELECT DISTINCT a [WHERE ..] [ORDER BY a] [LIMIT ..] == SELECT a, COUNT(*) FROM t [WHERE ..]
        // GROUP BY a [ORDER BY a] [LIMIT ..], with the COUNT column dropped from the result.
        let mut grouped = select.clone();
        grouped.distinct = false;
        grouped.group_by = Some(column.clone());
        grouped.projection = SelectProjection::GroupedAggregates {
            group_column: column.clone(),
            aggregates: vec![GroupedAggregate {
                kind: GroupedAggKind::Count,
                value_column: None,
            }],
        };
        let mut result = self.execute_resident_grouped_via_general(&grouped, src, visibility)?;
        // Drop the trailing COUNT(*) column -> the bare distinct keys (column 0 is the group key).
        // `columns` is now `Arc`-shared; `make_mut` gives an owned `&mut Vec` (no clone — this freshly
        // produced result holds the only reference).
        Arc::make_mut(&mut result.columns).truncate(1);
        // Drop the trailing COUNT(*) value from every row too — `rows` is a flat RowBlock, so reshape it to
        // a single column (the bare distinct group key, column 0).
        let keys: Vec<SqlValue> = result.rows.iter().map(|row| row[0].clone()).collect();
        result.rows = RowBlock::flat(keys, 1);
        Ok(result)
    }

    /// CPU-ENGINE RETIREMENT (ADR-006): the `&Select`->general-executor DISPATCH for the DECLINED-shape
    /// read fallback (`execute_relational_select_instrumented`, when the specialized resident route did
    /// not recognize the shape). Routes by shape to the correct `src: None` sub-bridge — a `SELECT
    /// DISTINCT col` to the DISTINCT bridge (which synthesizes `GROUP BY col` and dedups on-device;
    /// `with_binding` does NOT dedup a bare distinct projection itself), and everything else (GROUP BY /
    /// single-key ORDER BY / plain projection / scalar aggregate) to the grouped bridge, whose non-grouped
    /// path runs the plain projection / aggregate. `src: None` lets `with_binding` resolve the payload —
    /// the whole-table buffer OR the TYPE-COMPLETE unified shard source (THE FLIP), so wider-type
    /// (int8 / numeric / uuid / bool / text) shapes run on-device, unlike the int4-only
    /// `execute_resident_sharded_via_general`. Mirrors that method's shape dispatch (distinct-first, then
    /// grouped for GROUP BY / ORDER BY). Errors (never mis-answers) on a shape it cannot express; the
    /// caller then serves it from the CPU pinned path.
    pub(crate) fn execute_resident_select_via_general(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        if select.distinct {
            self.execute_resident_distinct_via_general(select, None, None)
        } else {
            self.execute_resident_grouped_via_general(select, None, None)
        }
    }

    /// Benchmark helper (tests only): run GROUP BY on a resident table with the SINGLE-LEVEL or
    /// TWO-LEVEL kernel selected explicitly, over a full-table scan. Returns the result rows so the
    /// caller can confirm both kernels agree; the caller times repeated calls. Not on the query path.
    #[cfg(test)]
    pub(crate) fn group_by_i32_bench(
        &self,
        table_name: &str,
        key_col: &str,
        value_col: &str,
        two_level: bool,
    ) -> Result<Vec<gpu_db_execution::GroupByI32Row>, ExecuteError> {
        let table = self.relational_catalog_table(table_name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!("no table {table_name}")))
        })?;
        let snapshot = self
            .relational_residency_snapshot_ref(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no resident snapshot".to_string()))
            })?;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no device memory".to_string()))
            })?;
        let key_idx = relational_column_index(&table, key_col)?;
        let val_idx = relational_column_index(&table, value_col)?;
        let key_off = resident_device_int4_column_offset(&snapshot, &table, key_idx)?;
        let val_off = resident_device_int4_column_offset(&snapshot, &table, val_idx)?;
        let n = u32::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed("rows exceed u32".to_string()))
        })?;
        let indices: Vec<u32> = (0..n).collect();
        device_memory
            .group_by_i32_count_sum_bench(
                gpu_db_execution::CudaGroupByInput::resident_i32(
                    key_off,
                    val_off,
                    u64::from(n),
                ),
                &indices,
                two_level,
            )
            .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))
    }

    /// Benchmark helper (tests only): time JUST the GROUP BY kernel (CUDA events, min of `runs`),
    /// isolating it from the alloc/H2D/D2H/host-compact overhead. Returns the min kernel milliseconds.
    #[cfg(test)]
    // Benchmark-only facade: explicit dimensions keep call sites auditable without creating a
    // product/runtime parameter object for a test-only kernel probe.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn group_by_i32_bench_kernel_ms(
        &self,
        table_name: &str,
        key_col: &str,
        value_col: &str,
        two_level: bool,
        runs: u32,
        rows_limit: usize,
        // Query-aware aggregate-selection mask (`grouped_agg_mask`): `COUNT` times the pruned path,
        // `ALL` the full-compute path, isolating the per-row atomic savings.
        agg_mask: u32,
    ) -> Result<f32, ExecuteError> {
        let table = self.relational_catalog_table(table_name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!("no table {table_name}")))
        })?;
        let snapshot = self
            .relational_residency_snapshot_ref(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no resident snapshot".to_string()))
            })?;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no device memory".to_string()))
            })?;
        let key_idx = relational_column_index(&table, key_col)?;
        let val_idx = relational_column_index(&table, value_col)?;
        let key_off = resident_device_int4_column_offset(&snapshot, &table, key_idx)?;
        let val_off = resident_device_int4_column_offset(&snapshot, &table, val_idx)?;
        let n = u32::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed("rows exceed u32".to_string()))
        })?;
        // rows_limit == 0 means the whole table; otherwise group only the first `rows_limit` rows (to
        // sweep data size at fixed cardinality from one residency).
        let m = if rows_limit == 0 {
            n
        } else {
            (rows_limit as u32).min(n)
        };
        let indices: Vec<u32> = (0..m).collect();
        device_memory
            .group_by_i32_count_sum_kernel_timed(
                gpu_db_execution::CudaGroupByInput::resident_i32(
                    key_off,
                    val_off,
                    u64::from(n),
                ),
                &indices,
                two_level,
                runs,
                agg_mask,
            )
            .map(|(_, ms)| ms)
            .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))
    }

    /// As [`Engine::execute_resident_expr_select`] but over an ALREADY-BOUND table/projection: the
    /// caller bound the catalog once and resolved the predicate's `Column` indices against this SAME
    /// `table`. The SQL->Expr entry (`engine_sql_pg`) routes through here so the predicate's column
    /// indices, the projection, and the residency snapshot all derive from one catalog generation — a
    /// concurrent shape-changing DDL cannot split the column resolution from the execution.
    /// Resolve one join relation to its (residency entry, device memory, row count): a RESIDENT user
    /// table uses its published snapshot + retained device memory; a SYNTHESIZED catalog relation
    /// (`rows = Some`, M5 J5) is uploaded as a TRANSIENT device payload + descriptor that lives only for
    /// this query. Either way the join runs the SAME GPU pre-filter + key-projection + hash-join kernels
    /// over the result -- no CPU relational join (charter).
    pub(crate) fn resolve_join_side(
        &self,
        name: &str,
        table: &RelationalTable,
        rows: Option<Vec<Vec<SqlValue>>>,
        copin_s: Index,
    ) -> Result<JoinExecSide, ExecuteError> {
        match rows {
            None => {
                // THE FLIP: a SHARD-resident relation (no single-buffer entry) joins via its UNIFIED
                // exec source — the same GPU pre-filter/key-projection/hash-join kernels run over the
                // recompacted (or zero-copy single-shard) buffer. A VERSIONED sharded relation
                // clean-errors: the join kernels do not thread the visibility conjuncts (never a
                // tombstone leak). No host rows ride the entry — the join is GPU-only (charter).
                if self.relational_residency_entry(name).is_none()
                    && self
                        .read_state
                        .residency
                        .shards
                        .load()
                        .get(name)
                        .is_some_and(|shards| !shards.is_empty())
                {
                    let unified = self.build_sharded_unified_exec_source(table, None, copin_s)?;
                    let row_count = unified.src.descriptor.row_count;
                    let entry = RelationalResidencyEntry::new(unified.src.descriptor);
                    return Ok((
                        entry,
                        JoinDeviceMemory::Resident(unified.src.device_memory),
                        row_count,
                        unified.visibility,
                    ));
                }
                let entry = self.relational_residency_entry(name).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{name}\" has no resident snapshot (the join path is GPU-only)"
                    )))
                })?;
                if !entry.descriptor.is_valid() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{name}\" resident snapshot is invalid"
                    ))));
                }
                let row_count = entry.descriptor.row_count;
                let memory = self
                    .read_state
                    .residency
                    .device_memory
                    .get(name)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "relation \"{name}\" has no retained resident device memory"
                        )))
                    })?;
                Ok((entry, JoinDeviceMemory::Resident(memory), row_count, None))
            }
            Some(rows) => {
                let _transient_scope =
                    gpu_db_execution::Probe::scope("join_transient_side_upload");
                let (snapshot, memory) = self.build_transient_relation_residency(table, &rows)?;
                let row_count = rows.len();
                let entry = RelationalResidencyEntry::new(std::sync::Arc::new(snapshot));
                Ok((entry, JoinDeviceMemory::Transient(memory), row_count, None))
            }
        }
    }

    /// Compile a per-relation WHERE predicate over an OUTER join's synthetic all-NULL pad into a
    /// retained one-row DEVICE mask (S6 -- doc 22). The filter kernel consumes `mask[0]` directly;
    /// no host boolean verdict or semantic re-upload exists.
    /// Build a 1-row TRANSIENT relation whose every column is NULL and run the predicate through the SAME
    /// GPU WHERE-3VL mask VM as the real rows (`lower_resident_predicate`); the pad survives iff row 0
    /// survives. On the all-NULL row a comparison/arithmetic/bare-column leaf is AND'd with the (all-zero)
    /// validity mask -> UNKNOWN -> excluded; `col IS NULL` reads the 0 validity bit -> TRUE (the anti-join);
    /// AND/OR fold via the VM's Kleene masks. So the host no longer evaluates 3VL -- the GPU does. A
    /// predicate shape the VM cannot lower over a nullable column clean-errors there (the engine-wide WHERE
    /// 3VL limitation), exactly as the real-row survivor pass would for the same shape.
    fn predicate_mask_on_null_pad(
        &self,
        predicate: &ResidentExpr,
        table: &RelationalTable,
    ) -> Result<JoinNullPadMask, ExecuteError> {
        // One all-NULL row has no text bytes. Per column, 64 bytes strictly dominates the widest
        // 16-byte value/offset slot, its validity word, bool storage, and every <=8-byte section
        // alignment pad; the fixed 64-byte header allowance dominates the resident header/final pad.
        // Reserve this conservative raw-allocation bound BEFORE the transient builder allocates.
        let pad_source_bound = 64_usize.saturating_add(table.columns.len().saturating_mul(64));
        let allocation = gpu_db_execution::CudaAllocationScope::reserve_external(pad_source_bound)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let null_row = vec![vec![SqlValue::Null; table.columns.len()]];
        let (snapshot, memory) = self.build_transient_relation_residency(table, &null_row)?;
        let mask = self.resident_predicate_device_mask(
            Some(predicate),
            table,
            &snapshot,
            &memory,
            1,
            None,
        )?;
        Ok(JoinNullPadMask {
            mask,
            _source: memory,
            _allocation: allocation,
        })
    }

    /// Execute a left-deep equi-join chain on resident payloads. Predicates, MVCC masks, N:N/OUTER
    /// membership, carried coordinates, ordering, and intermediate projection remain device-resident;
    /// only the final requested result columns are framed on the host. Streaming callers may retain an
    /// opaque D2D-materialized run instead, so source chunks can be evicted without a D2H intermediate.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_resident_expr_inner_join(
        &self,
        plan: &JoinPlan,
        tables: Vec<RelationalTable>,
        rows: Vec<Option<Vec<Vec<SqlValue>>>>,
        predicates: Vec<Option<ResidentExpr>>,
        copin_s: Index,
        side_override: Option<Vec<JoinExecSide>>,
        coordinate_out: Option<&mut Option<gpu_db_execution::CudaJoinCoordinatesU32>>,
        force_outer_where: bool,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_resident_expr_join_with_device_run(
            plan,
            tables,
            rows,
            predicates,
            copin_s,
            side_override,
            coordinate_out,
            None,
            force_outer_where,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_resident_expr_join_with_device_run(
        &self,
        plan: &JoinPlan,
        tables: Vec<RelationalTable>,
        rows: Vec<Option<Vec<Vec<SqlValue>>>>,
        predicates: Vec<Option<ResidentExpr>>,
        copin_s: Index,
        side_override: Option<Vec<JoinExecSide>>,
        coordinate_out: Option<&mut Option<gpu_db_execution::CudaJoinCoordinatesU32>>,
        materialized_out: Option<&mut Option<gpu_db_execution::CudaMaterializedRelation>>,
        force_outer_where: bool,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_resident_expr_join_with_device_overrides(
            plan,
            tables,
            rows,
            predicates,
            copin_s,
            side_override,
            None,
            coordinate_out,
            materialized_out,
            force_outer_where,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_resident_expr_join_with_device_ranges(
        &self,
        plan: &JoinPlan,
        tables: Vec<RelationalTable>,
        rows: Vec<Option<Vec<Vec<SqlValue>>>>,
        predicates: Vec<Option<ResidentExpr>>,
        copin_s: Index,
        side_override: Option<Vec<JoinExecSide>>,
        row_range_override: Vec<(u32, u32)>,
        coordinate_out: Option<&mut Option<gpu_db_execution::CudaJoinCoordinatesU32>>,
        materialized_out: Option<&mut Option<gpu_db_execution::CudaMaterializedRelation>>,
        force_outer_where: bool,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_resident_expr_join_with_device_overrides(
            plan,
            tables,
            rows,
            predicates,
            copin_s,
            side_override,
            Some(row_range_override),
            coordinate_out,
            materialized_out,
            force_outer_where,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_resident_expr_join_with_device_overrides(
        &self,
        plan: &JoinPlan,
        tables: Vec<RelationalTable>,
        rows: Vec<Option<Vec<Vec<SqlValue>>>>,
        predicates: Vec<Option<ResidentExpr>>,
        // SC5 rider (ADR-013 adjunct): the STATEMENT'S bound boundary — the same `s` the caller
        // bound the catalog at, so every relation's device state resolves at ONE snapshot
        // (previously each sharded side re-read `committed_seq()`, seeding cross-relation skew).
        copin_s: Index,
        side_override: Option<Vec<JoinExecSide>>,
        row_range_override: Option<Vec<(u32, u32)>>,
        coordinate_out: Option<&mut Option<gpu_db_execution::CudaJoinCoordinatesU32>>,
        materialized_out: Option<&mut Option<gpu_db_execution::CudaMaterializedRelation>>,
        force_outer_where: bool,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let n_rel = plan.relations.len();
        if row_range_override
            .as_ref()
            .is_some_and(|ranges| ranges.len() != n_rel)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "join row-range override count does not match the plan".to_string(),
            )));
        }
        // OUTER joins (LEFT/RIGHT/FULL, M3 -- doc 21). N-way (multi-step) OUTER is supported: a prior step's
        // NULL pad carries a `JOIN_NULL_ROW` sentinel whose validity is gathered as 0, so the hash-join
        // kernel skips it as a NULL key (matches nothing; a LEFT step re-pads it), and the final gather
        // emits SqlValue::Null for it.
        //
        // A WHERE on an OUTER join is NOT filter-commutative: pushing a per-side predicate down before the
        // join would drop rows the outer join must NULL-pad, and a predicate on the padded side would
        // change which rows are padded. So for an outer join the per-side predicates are NOT pushed down
        // (the join sees ALL rows); instead each predicate's GPU-computed survivor set becomes a POST-join
        // membership filter applied to the result below: a tuple survives only if, for every predicated
        // relation, its carried row is REAL (a JOIN_NULL_ROW pad means that relation's columns are NULL ->
        // the predicate is UNKNOWN -> drop) AND that row passed the predicate on-device. This matches PG
        // (a WHERE on the inner side of a LEFT join effectively makes it inner on that condition).
        let outer_where = (force_outer_where
            || plan.steps.iter().any(|s| s.outer_left || s.outer_right))
            && predicates.iter().any(Option::is_some);
        // Resolve a JOIN column reference to (relation index, column index) against relations[0..=upto]:
        // a qualifier must name exactly one of them; an unqualified column must be in exactly one (PG's
        // "ambiguous" / "does not exist"). `upto` bounds an ON operand to the relations joined so far.
        let resolve = |c: &JoinColRef, upto: usize| -> Result<(usize, usize), ExecuteError> {
            match &c.qualifier {
                Some(q) => {
                    for (i, rel) in plan.relations.iter().enumerate().take(upto + 1) {
                        if rel.alias == *q {
                            return Ok((i, relational_column_index(&tables[i], &c.column)?));
                        }
                    }
                    Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "missing FROM-clause entry for table \"{q}\""
                    ))))
                }
                None => {
                    let mut found: Option<(usize, usize)> = None;
                    for (i, table) in tables.iter().take(upto + 1).enumerate() {
                        if let Ok(ci) = relational_column_index(table, &c.column) {
                            if found.is_some() {
                                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                    format!("column reference \"{}\" is ambiguous", c.column),
                                )));
                            }
                            found = Some((i, ci));
                        }
                    }
                    found.ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "column \"{}\" does not exist",
                            c.column
                        )))
                    })
                }
            }
        };
        // A single-conjunct key may be any int (int2/int4/date in the i32 section, int8/timestamp in the
        // i64 section -- both project to i64, so a mixed int4=int8 equi-join compares correctly). A
        // COMPOSITE (2-conjunct) key packs two members into one i64 (member0 in the high 32 bits, member1
        // in the low), so each member must be <=32 bits (int2/int4/date); an int8/timestamp composite
        // member (or >2 conjuncts) overflows 64 bits -> a follow-up. numeric/uuid/text keys are J4b/c.
        let int_key = |t: SqlType| {
            matches!(
                t,
                SqlType::Int4 | SqlType::Int2 | SqlType::Date | SqlType::Int8 | SqlType::Timestamp
            )
        };
        let narrow_key = |t: SqlType| matches!(t, SqlType::Int4 | SqlType::Int2 | SqlType::Date);
        // Pre-resolve each step's ON conjuncts + validate the key types, and resolve the SELECT
        // projection, BEFORE touching residency -- so a malformed query (unknown column, ambiguous ref,
        // non-int key) fails fast with a query error, not a "not resident" one. `step_keys[k]` = the
        // per-conjunct (accumulated relation, its key column, the new relation's key column); the newly
        // joined relation is `k+1`. Conjuncts may reference DIFFERENT accumulated relations.
        let mut step_keys: Vec<Vec<(usize, usize, usize)>> = Vec::with_capacity(plan.steps.len());
        // Per step's key kind: TEXT or NUMERIC/UUID (b128, both -> the GPU text/byte hash join) vs INT
        // (the i64 path). `step_is_text` = TEXT bytes; `step_is_b128` = a 16-byte numeric/uuid value.
        let mut step_is_text: Vec<bool> = Vec::with_capacity(plan.steps.len());
        let mut step_is_b128: Vec<bool> = Vec::with_capacity(plan.steps.len());
        // USING/NATURAL join columns, coalesced (emitted ONCE in `*`, resolvable unqualified). 2-relation
        // only (build_join_plan rejects multi-way USING/NATURAL), so this collects a single step's set.
        let mut coalesce_cols: Vec<String> = Vec::new();
        for (k, step) in plan.steps.iter().enumerate() {
            let new_rel = k + 1;
            let mut conj_keys: Vec<(usize, usize, usize)> = Vec::new();
            let step_coalesce: Vec<String> = if step.natural {
                // NATURAL (2-relation, so new_rel == 1 and acc == 0): join on the relations' COMMON column
                // names -> conjuncts (rel0.c = rel1.c) + the coalesce set.
                let mut common: Vec<String> = Vec::new();
                for (c0, col) in tables[0].columns.iter().enumerate() {
                    if let Ok(c1) = relational_column_index(&tables[new_rel], &col.name) {
                        conj_keys.push((0, c0, c1));
                        common.push(col.name.clone());
                    }
                }
                if common.is_empty() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "NATURAL JOIN has no common column name between the two relations"
                            .to_string(),
                    )));
                }
                common
            } else {
                for (on_a, on_b) in &step.conjuncts {
                    // Each conjunct must equate the newly joined relation (`new_rel`) to an already-joined
                    // one (<new_rel).
                    let ra = resolve(on_a, new_rel)?;
                    let rb = resolve(on_b, new_rel)?;
                    let ((acc_rel, acc_col), new_col) = if rb.0 == new_rel && ra.0 < new_rel {
                        ((ra.0, ra.1), rb.1)
                    } else if ra.0 == new_rel && rb.0 < new_rel {
                        ((rb.0, rb.1), ra.1)
                    } else {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "each JOIN's ON conjunct must equate the newly joined relation to an \
                             already-joined one (a.k = b.k)"
                                .to_string(),
                        )));
                    };
                    conj_keys.push((acc_rel, acc_col, new_col));
                }
                step.coalesce.clone()
            };
            // A step joins on 1 or 2 columns (ON/comma guarantee >=1; NATURAL errored above on 0). It always
            // reaches `pack_keys`, which handles only 1-2 members; reject an empty step (defensive) and >2
            // columns (a composite key wider than 64 bits -- e.g. NATURAL/USING over 3+ columns).
            if conj_keys.is_empty() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "a join step requires at least one equality condition between the relations"
                        .to_string(),
                )));
            }
            if conj_keys.len() > 2 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "a join on more than 2 equality columns (a composite key wider than 64 bits) is a \
                     follow-up"
                        .to_string(),
                )));
            }
            coalesce_cols.extend(step_coalesce);
            // Determine the key kind + validate types. A single-conjunct key may be TEXT (-> the GPU text
            // hash join: FNV + byte-verify), NUMERIC/UUID (-> the SAME kernel over the 16-byte canonical
            // value), or INT (-> the i64 hash join). A 2-conjunct composite is INT-only (i64-packed).
            let non_int =
                |t: SqlType| matches!(t, SqlType::Text | SqlType::Numeric { .. } | SqlType::Uuid);
            let has_non_int = conj_keys.iter().any(|&(acc_rel, acc_col, new_col)| {
                non_int(tables[acc_rel].columns[acc_col].ty)
                    || non_int(tables[new_rel].columns[new_col].ty)
            });
            let mut is_text = false;
            let mut is_b128 = false;
            if has_non_int {
                // text / numeric / uuid: a single conjunct, the SAME key type on both sides (numeric also
                // the same scale, so the i128 mantissa compares value-for-value).
                let (acc_rel, acc_col, new_col) = conj_keys[0];
                let acc_ty = tables[acc_rel].columns[acc_col].ty;
                let new_ty = tables[new_rel].columns[new_col].ty;
                let same_type = conj_keys.len() == 1
                    && match (acc_ty, new_ty) {
                        (SqlType::Text, SqlType::Text) => {
                            is_text = true;
                            true
                        }
                        (SqlType::Uuid, SqlType::Uuid) => {
                            is_b128 = true;
                            true
                        }
                        (
                            SqlType::Numeric { scale: s_acc, .. },
                            SqlType::Numeric { scale: s_new, .. },
                        ) if s_acc == s_new => {
                            is_b128 = true;
                            true
                        }
                        _ => false,
                    };
                if !same_type {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "a text/numeric/uuid join key must be a single `a.k = b.k` with the SAME type \
                         on BOTH sides (numeric: the same scale); composite / mixed / different-scale \
                         keys with these types are a follow-up"
                            .to_string(),
                    )));
                }
            } else {
                // A composite member must be <=32 bits so two pack into one i64; a single key may be int8.
                let composite = conj_keys.len() == 2;
                for &(acc_rel, acc_col, new_col) in &conj_keys {
                    let ok = |t: SqlType| if composite { narrow_key(t) } else { int_key(t) };
                    if !ok(tables[acc_rel].columns[acc_col].ty)
                        || !ok(tables[new_rel].columns[new_col].ty)
                    {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "a join key must be an integer column on both sides (int2/int4/int8/date/\
                             timestamp for a single key; int2/int4/date for each member of a 2-column \
                             composite key)"
                                .to_string(),
                        )));
                    }
                }
            }
            step_keys.push(conj_keys);
            step_is_text.push(is_text);
            step_is_b128.push(is_b128);
        }
        // Resolve the SELECT list to a flat (relation index, column index) list, expanding `*` and
        // `alias.*`. With USING/NATURAL, a join column is COALESCED: it appears ONCE in bare `*` (PG order:
        // the join columns first, then the left relation's other columns, then the right's), and an
        // UNQUALIFIED reference to it resolves to the left copy (not ambiguous). `alias.*` is unchanged
        // (a relation's own columns). `coalesce_cols` is empty for ON/comma joins -> the prior behavior.
        let is_coalesced = |name: &str| coalesce_cols.iter().any(|c| c == name);
        let mut proj: Vec<(usize, usize)> = Vec::new();
        for item in &plan.projection {
            match item {
                JoinProjItem::Column(c) if c.qualifier.is_none() && is_coalesced(&c.column) => {
                    // The coalesced join column lives in relation 0 (the left side of the 2-relation join).
                    proj.push((0, relational_column_index(&tables[0], &c.column)?));
                }
                JoinProjItem::Column(c) => proj.push(resolve(c, n_rel - 1)?),
                JoinProjItem::Star(None) if !coalesce_cols.is_empty() => {
                    // USING/NATURAL (2-relation): coalesced columns first (from rel0), then each relation's
                    // remaining columns left-to-right (the right copy of a coalesced column is skipped).
                    for name in &coalesce_cols {
                        proj.push((0, relational_column_index(&tables[0], name)?));
                    }
                    for (ri, table) in tables.iter().enumerate() {
                        proj.extend(
                            table
                                .columns
                                .iter()
                                .enumerate()
                                .filter(|(_, col)| !is_coalesced(&col.name))
                                .map(|(ci, _)| (ri, ci)),
                        );
                    }
                }
                JoinProjItem::Star(None) => {
                    for (ri, table) in tables.iter().enumerate() {
                        proj.extend((0..table.columns.len()).map(|ci| (ri, ci)));
                    }
                }
                JoinProjItem::Star(Some(alias)) => {
                    let ri = plan
                        .relations
                        .iter()
                        .position(|r| r.alias == *alias)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "missing FROM-clause entry for table \"{alias}\""
                            )))
                        })?;
                    proj.extend((0..tables[ri].columns.len()).map(|ci| (ri, ci)));
                }
            }
        }
        // Resolve every relation's device payload, at the one bound catalog generation: a RESIDENT user
        // table's published memory, or a SYNTHESIZED catalog relation's transient upload.
        let sides: Vec<JoinExecSide> = match side_override {
            Some(sides) => {
                if sides.len() != n_rel {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "join side override count does not match the plan".to_string(),
                    )));
                }
                sides
            }
            None => {
                let mut sides = Vec::with_capacity(n_rel);
                for (row_opt, (relation, table)) in
                    rows.into_iter().zip(plan.relations.iter().zip(&tables))
                {
                    sides.push(self.resolve_join_side(
                        &relation.table,
                        table,
                        row_opt,
                        copin_s,
                    )?);
                }
                sides
            }
        };
        // The `JOIN_NULL_ROW` sentinel (LEFT-pad) must be distinguishable from every real absolute row
        // index -- it is, because residency never holds anywhere near u32::MAX rows. Make it explicit.
        debug_assert!(
            sides
                .iter()
                .all(|s| s.0.descriptor.row_count < JOIN_NULL_ROW as usize),
            "a join relation has too many rows to distinguish the LEFT-join NULL-pad sentinel"
        );
        let gpu_id = sides[0].0.descriptor.gpu_id;
        let projection_aliases = self.join_projection_output_aliases(plan, &tables)?;
        let mut resolved_order = Vec::with_capacity(plan.order_by.len());
        for (order_idx, (column, descending)) in plan.order_by.iter().enumerate() {
            let alias_matches = if column.qualifier.is_none() {
                projection_aliases
                    .iter()
                    .enumerate()
                    .filter_map(|(index, alias)| {
                        (alias.as_deref() == Some(&column.column)).then_some(index)
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let resolved = if let [index] = alias_matches.as_slice() {
                proj[*index]
            } else if alias_matches.len() > 1 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "ORDER BY \"{}\" is ambiguous",
                    column.column
                ))));
            } else if column.qualifier.is_none() && is_coalesced(&column.column) {
                (0, relational_column_index(&tables[0], &column.column)?)
            } else {
                resolve(column, n_rel - 1)?
            };
            resolved_order.push((
                resolved.0,
                resolved.1,
                *descending,
                plan.order_by_nulls_first
                    .get(order_idx)
                    .copied()
                    .flatten(),
            ));
        }
        self.execute_resident_device_coordinate_join(
            plan,
            &tables,
            &sides,
            &step_keys,
            &step_is_text,
            &step_is_b128,
            &proj,
            &predicates,
            outer_where,
            row_range_override.as_deref(),
            &resolved_order,
            gpu_id,
            coordinate_out,
            materialized_out,
        )
    }

    pub(crate) fn resident_predicate_device_mask(
        &self,
        predicate: Option<&ResidentExpr>,
        table: &RelationalTable,
        descriptor: &RelationalResidencySnapshot,
        memory: &gpu_db_execution::CudaResidentDeviceMemory,
        row_count: u32,
        visibility: Option<ResidentVisibility>,
    ) -> Result<Option<gpu_db_execution::CudaPredicateMaskI32>, ExecuteError> {
        use gpu_db_execution::ResidentElemType;
        if row_count == 0 || (predicate.is_none() && visibility.is_none()) {
            return Ok(None);
        }
        let mut program = Vec::new();
        let mut needles = Vec::new();
        let elem = if let Some(predicate) = predicate {
            if expr_mentions_numeric(predicate, table) {
                compile_numeric_predicate_program(predicate, table, descriptor, &mut program)?;
                ResidentElemType::I128
            } else {
                compile_predicate_program(
                    predicate,
                    table,
                    descriptor,
                    &mut program,
                    &mut needles,
                )?;
                predicate_vm_elem_type(predicate, table)
                    .or_else(|| mixed_width_i32_elem(predicate, table))
                    .unwrap_or(ResidentElemType::I32)
            }
        } else {
            ResidentElemType::I64
        };
        if let Some(visibility) = visibility {
            visibility.push_conjuncts(&mut program, predicate.is_some());
        }
        #[cfg(feature = "probe-timing")]
        eprintln!(
            "[probe] predicate_mask table={} rows={} elem={elem:?} steps={program:?} needles={needles:?}",
            table.name, row_count
        );
        memory
            .run_expr_predicate_mask_with_text(&program, &needles, row_count, elem)
            .map(Some)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    pub(crate) fn resident_join_identity_coordinates(
        &self,
        table: &RelationalTable,
        side: &JoinExecSide,
        row_range: Option<(u32, u32)>,
    ) -> Result<gpu_db_execution::CudaJoinCoordinatesU32, ExecuteError> {
        let mut mask = self.resident_predicate_device_mask(
            None,
            table,
            &side.0.descriptor,
            side.1.mem(),
            side.2 as u32,
            side.3,
        )?;
        if let Some((start, end)) = row_range {
            let range = side
                .1
                .mem()
                .row_range_mask_u32(side.2 as u32, start, end)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            mask = match mask {
                Some(mask) => Some(
                    side.1
                        .mem()
                        .and_predicate_masks(&mask, &range)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?,
                ),
                None => Some(range),
            };
        }
        side.1
            .mem()
            .identity_join_coordinates(side.2 as u32, mask.as_ref())
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_resident_join_coordinate_matches(
        &self,
        plan: &JoinPlan,
        tables: &[RelationalTable],
        sides: &[JoinExecSide],
        step_index: usize,
        accumulated: &gpu_db_execution::CudaJoinCoordinatesU32,
        left_matches: &gpu_db_execution::CudaMatchBitmapU32,
        right_matches: Option<&gpu_db_execution::CudaMatchBitmapU32>,
        right_range: Option<(u32, u32)>,
    ) -> Result<gpu_db_execution::CudaJoinCoordinatesU32, ExecuteError> {
        use gpu_db_execution::CudaJoinPayloadKey;

        let right_relation = step_index + 1;
        if right_relation >= sides.len() || accumulated.relation_count() != right_relation as u32 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "incremental join step arity does not match its accumulated coordinates".to_string(),
            )));
        }
        let resolve = |column: &JoinColRef| -> Result<(usize, usize), ExecuteError> {
            if let Some(qualifier) = &column.qualifier {
                let relation = plan
                    .relations
                    .iter()
                    .take(right_relation + 1)
                    .position(|relation| relation.alias == *qualifier)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "missing FROM-clause entry for table \"{qualifier}\""
                        )))
                    })?;
                return Ok((
                    relation,
                    relational_column_index(&tables[relation], &column.column)?,
                ));
            }
            let matches = tables
                .iter()
                .take(right_relation + 1)
                .enumerate()
                .filter_map(|(relation, table)| {
                    relational_column_index(table, &column.column)
                        .ok()
                        .map(|column| (relation, column))
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [value] => Ok(*value),
                [] => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column \"{}\" does not exist",
                    column.column
                )))),
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column reference \"{}\" is ambiguous",
                    column.column
                )))),
            }
        };
        let descriptor = |relation: usize, column: usize| -> Result<CudaJoinPayloadKey<'_>, ExecuteError> {
            let entry = &sides[relation].0;
            let payload = sides[relation].1.mem();
            let validity = resident_device_null_column_offset(
                &entry.descriptor,
                &tables[relation],
                column,
            )?;
            Ok(match tables[relation].columns[column].ty {
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?;
                    CudaJoinPayloadKey {
                        payload,
                        byte_offset: layout.offsets_byte_offset,
                        validity_bitmap_offset: validity,
                        width: 255,
                        text_bytes_byte_offset: Some(layout.bytes_byte_offset),
                        text_bytes_len: layout.bytes_len,
                    }
                }
                SqlType::Numeric { .. } | SqlType::Uuid => CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_numeric_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset: validity,
                    width: 16,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                },
                SqlType::Int8 | SqlType::Timestamp => CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_int8_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset: validity,
                    width: 8,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                },
                SqlType::Int2 | SqlType::Int4 | SqlType::Date => CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_int4_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset: validity,
                    width: 4,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                },
                SqlType::Bool => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "bool equi-join keys are not supported".to_string(),
                    )))
                }
            })
        };
        let step = &plan.steps[step_index];
        if step.natural || step.conjuncts.is_empty() || step.conjuncts.len() > 2 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "incremental streaming OUTER joins require one or two explicit equality keys"
                    .to_string(),
            )));
        }
        let mut left_keys = Vec::with_capacity(step.conjuncts.len());
        let mut right_keys = Vec::with_capacity(step.conjuncts.len());
        let mut left_key_relations = Vec::with_capacity(step.conjuncts.len());
        for (left, right) in &step.conjuncts {
            let a = resolve(left)?;
            let b = resolve(right)?;
            let (acc, new) = if a.0 == right_relation && b.0 < right_relation {
                (b, a)
            } else if b.0 == right_relation && a.0 < right_relation {
                (a, b)
            } else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "each incremental join key must connect the new relation to the accumulated side"
                        .to_string(),
                )));
            };
            left_key_relations.push(acc.0 as u32);
            left_keys.push(descriptor(acc.0, acc.1)?);
            right_keys.push(descriptor(new.0, new.1)?);
        }
        let mut right_mask = self.resident_predicate_device_mask(
            None,
            &tables[right_relation],
            &sides[right_relation].0.descriptor,
            sides[right_relation].1.mem(),
            sides[right_relation].2 as u32,
            sides[right_relation].3,
        )?;
        if let Some((start, end)) = right_range {
            let range = sides[right_relation]
                .1
                .mem()
                .row_range_mask_u32(sides[right_relation].2 as u32, start, end)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            right_mask = match right_mask {
                Some(mask) => Some(
                    sides[right_relation]
                        .1
                        .mem()
                        .and_predicate_masks(&mask, &range)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?,
                ),
                None => Some(range),
            };
        }
        sides[0]
            .1
            .mem()
            .join_fixed_payload_coordinate_matches(
                Some(accumulated),
                0,
                &left_key_relations,
                &left_keys,
                sides[right_relation].2 as u32,
                &right_keys,
                None,
                right_mask.as_ref(),
                left_matches,
                right_matches,
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    pub(crate) fn materialize_join_projection_coordinates(
        &self,
        plan: &JoinPlan,
        tables: &[RelationalTable],
        sides: &[JoinExecSide],
        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32,
    ) -> Result<
        (
            Arc<Vec<RelationalColumn>>,
            gpu_db_execution::CudaMaterializedRelation,
        ),
        ExecuteError,
    > {
        let resolve = |column: &JoinColRef| -> Result<(usize, usize), ExecuteError> {
            if let Some(qualifier) = &column.qualifier {
                let relation = plan
                    .relations
                    .iter()
                    .position(|relation| relation.alias == *qualifier)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "missing FROM-clause entry for table \"{qualifier}\""
                        )))
                    })?;
                return Ok((
                    relation,
                    relational_column_index(&tables[relation], &column.column)?,
                ));
            }
            let matches = tables
                .iter()
                .enumerate()
                .filter_map(|(relation, table)| {
                    relational_column_index(table, &column.column)
                        .ok()
                        .map(|column| (relation, column))
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [resolved] => Ok(*resolved),
                [] => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column \"{}\" does not exist",
                    column.column
                )))),
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column reference \"{}\" is ambiguous",
                    column.column
                )))),
            }
        };
        let coalesced = plan
            .steps
            .iter()
            .flat_map(|step| step.coalesce.iter().cloned())
            .collect::<Vec<_>>();
        let mut projection = Vec::new();
        for item in &plan.projection {
            match item {
                JoinProjItem::Column(column)
                    if column.qualifier.is_none() && coalesced.contains(&column.column) =>
                {
                    projection.push((
                        0,
                        relational_column_index(&tables[0], &column.column)?,
                    ));
                }
                JoinProjItem::Column(column) => projection.push(resolve(column)?),
                JoinProjItem::Star(None) if !coalesced.is_empty() => {
                    for name in &coalesced {
                        projection.push((0, relational_column_index(&tables[0], name)?));
                    }
                    for (relation, table) in tables.iter().enumerate() {
                        projection.extend(
                            table
                                .columns
                                .iter()
                                .enumerate()
                                .filter(|(_, column)| !coalesced.contains(&column.name))
                                .map(|(column, _)| (relation, column)),
                        );
                    }
                }
                JoinProjItem::Star(None) => {
                    for (relation, table) in tables.iter().enumerate() {
                        projection.extend(
                            table
                                .columns
                                .iter()
                                .enumerate()
                                .map(|(column, _)| (relation, column)),
                        );
                    }
                }
                JoinProjItem::Star(Some(alias)) => {
                    let relation = plan
                        .relations
                        .iter()
                        .position(|relation| relation.alias == *alias)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "missing FROM-clause entry for table \"{alias}\""
                            )))
                        })?;
                    projection.extend(
                        tables[relation]
                            .columns
                            .iter()
                            .enumerate()
                            .map(|(column, _)| (relation, column)),
                    );
                }
            }
        }
        let output_aliases = self.join_projection_output_aliases(plan, tables)?;
        let mut columns = Vec::with_capacity(projection.len());
        let mut specs = Vec::with_capacity(projection.len());
        for (output_index, &(relation, column)) in projection.iter().enumerate() {
            let mut output = tables[relation].columns[column].clone();
            output.attnum = (output_index + 1) as i16;
            if let Some(alias) = &output_aliases[output_index] {
                output.name.clone_from(alias);
            }
            columns.push(output);
            let entry = &sides[relation].0;
            let payload = sides[relation].1.mem();
            let validity = resident_device_null_column_offset(
                &entry.descriptor,
                &tables[relation],
                column,
            )?;
            specs.push(match tables[relation].columns[column].ty {
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?;
                    gpu_db_execution::CudaMaterializeJoinColumn::Text {
                        relation: relation as u32,
                        payload,
                        offsets_byte_offset: layout.offsets_byte_offset,
                        bytes_byte_offset: layout.bytes_byte_offset,
                        bytes_len: layout.bytes_len,
                        validity_bitmap_offset: validity,
                    }
                }
                SqlType::Bool => gpu_db_execution::CudaMaterializeJoinColumn::Bool {
                    relation: relation as u32,
                    payload,
                    bitmap_byte_offset: resident_device_bool_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset: validity,
                },
                ty => {
                    let (byte_offset, width) = match ty {
                        SqlType::Int8 | SqlType::Timestamp => (
                            resident_device_int8_column_offset(
                                &entry.descriptor,
                                &tables[relation],
                                column,
                            )?,
                            8,
                        ),
                        SqlType::Numeric { .. } | SqlType::Uuid => (
                            resident_device_numeric_column_offset(
                                &entry.descriptor,
                                &tables[relation],
                                column,
                            )?,
                            16,
                        ),
                        SqlType::Int2 | SqlType::Int4 | SqlType::Date => (
                            resident_device_int4_column_offset(
                                &entry.descriptor,
                                &tables[relation],
                                column,
                            )?,
                            4,
                        ),
                        SqlType::Text | SqlType::Bool => unreachable!(),
                    };
                    gpu_db_execution::CudaMaterializeJoinColumn::Fixed {
                        relation: relation as u32,
                        payload,
                        byte_offset,
                        validity_bitmap_offset: validity,
                        width,
                    }
                }
            });
        }
        let run = sides[0]
            .1
            .mem()
            .materialize_join_coordinates(coordinates, &specs)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        Ok((Arc::new(columns), run))
    }

    pub(crate) fn join_projection_sources(
        &self,
        plan: &JoinPlan,
        tables: &[RelationalTable],
    ) -> Result<Vec<(usize, usize)>, ExecuteError> {
        let resolve = |column: &JoinColRef| -> Result<(usize, usize), ExecuteError> {
            if let Some(qualifier) = &column.qualifier {
                let relation = plan
                    .relations
                    .iter()
                    .position(|relation| relation.alias == *qualifier)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "missing FROM-clause entry for table \"{qualifier}\""
                        )))
                    })?;
                return Ok((
                    relation,
                    relational_column_index(&tables[relation], &column.column)?,
                ));
            }
            let matches = tables
                .iter()
                .enumerate()
                .filter_map(|(relation, table)| {
                    relational_column_index(table, &column.column)
                        .ok()
                        .map(|column| (relation, column))
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [resolved] => Ok(*resolved),
                [] => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column \"{}\" does not exist",
                    column.column
                )))),
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "column reference \"{}\" is ambiguous",
                    column.column
                )))),
            }
        };
        let coalesced = plan
            .steps
            .iter()
            .flat_map(|step| step.coalesce.iter().cloned())
            .collect::<Vec<_>>();
        let mut projection = Vec::new();
        for item in &plan.projection {
            match item {
                JoinProjItem::Column(column)
                    if column.qualifier.is_none() && coalesced.contains(&column.column) =>
                {
                    projection.push((
                        0,
                        relational_column_index(&tables[0], &column.column)?,
                    ));
                }
                JoinProjItem::Column(column) => projection.push(resolve(column)?),
                JoinProjItem::Star(None) if !coalesced.is_empty() => {
                    for name in &coalesced {
                        projection.push((0, relational_column_index(&tables[0], name)?));
                    }
                    for (relation, table) in tables.iter().enumerate() {
                        projection.extend(
                            table
                                .columns
                                .iter()
                                .enumerate()
                                .filter(|(_, column)| !coalesced.contains(&column.name))
                                .map(|(column, _)| (relation, column)),
                        );
                    }
                }
                JoinProjItem::Star(None) => {
                    for (relation, table) in tables.iter().enumerate() {
                        projection.extend(
                            table
                                .columns
                                .iter()
                                .enumerate()
                                .map(|(column, _)| (relation, column)),
                        );
                    }
                }
                JoinProjItem::Star(Some(alias)) => {
                    let relation = plan
                        .relations
                        .iter()
                        .position(|relation| relation.alias == *alias)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "missing FROM-clause entry for table \"{alias}\""
                            )))
                        })?;
                    projection.extend(
                        tables[relation]
                            .columns
                            .iter()
                            .enumerate()
                            .map(|(column, _)| (relation, column)),
                    );
                }
            }
        }
        Ok(projection)
    }

    pub(crate) fn join_projection_output_aliases(
        &self,
        plan: &JoinPlan,
        tables: &[RelationalTable],
    ) -> Result<Vec<Option<String>>, ExecuteError> {
        if plan.projection.len() != plan.projection_aliases.len() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "join projection alias metadata is misaligned".to_string(),
            )));
        }
        let coalesced = plan
            .steps
            .iter()
            .flat_map(|step| step.coalesce.iter())
            .collect::<Vec<_>>();
        let mut aliases = Vec::new();
        for (item, alias) in plan.projection.iter().zip(&plan.projection_aliases) {
            match item {
                JoinProjItem::Column(_) => aliases.push(alias.clone()),
                JoinProjItem::Star(None) if !coalesced.is_empty() => {
                    aliases.extend((0..coalesced.len()).map(|_| None));
                    aliases.extend(
                        tables
                            .iter()
                            .flat_map(|table| &table.columns)
                            .filter(|column| !coalesced.iter().any(|name| ***name == column.name))
                            .map(|_| None),
                    );
                }
                JoinProjItem::Star(None) => aliases.extend(
                    tables
                        .iter()
                        .flat_map(|table| &table.columns)
                        .map(|_| None),
                ),
                JoinProjItem::Star(Some(qualifier)) => {
                    let relation = plan
                        .relations
                        .iter()
                        .position(|relation| relation.alias == *qualifier)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "missing FROM-clause entry for table \"{qualifier}\""
                            )))
                        })?;
                    aliases.extend((0..tables[relation].columns.len()).map(|_| None));
                }
            }
        }
        Ok(aliases)
    }

    pub(crate) fn project_join_column_values(
        &self,
        table: &RelationalTable,
        side: &JoinExecSide,
        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32,
        relation: u32,
        column: usize,
    ) -> Result<Vec<SqlValue>, ExecuteError> {
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let payload = side.1.mem();
        let validity = resident_device_null_column_offset(&side.0.descriptor, table, column)?;
        let ty = table.columns[column].ty;
        Ok(match ty {
            SqlType::Text => {
                let layout = resident_device_text_column_layout(&side.0.descriptor, table, column)?;
                payload
                    .project_text_from_join_coordinates(
                        coordinates,
                        relation,
                        payload,
                        layout.offsets_byte_offset,
                        layout.bytes_byte_offset,
                        layout.bytes_len,
                        validity,
                    )
                    .map_err(map_err)?
                    .into_iter()
                    .map(|value| value.map_or(SqlValue::Null, SqlValue::Text))
                    .collect()
            }
            SqlType::Bool => payload
                .project_bool_from_join_coordinates(
                    coordinates,
                    relation,
                    payload,
                    resident_device_bool_column_offset(&side.0.descriptor, table, column)?,
                    validity,
                )
                .map_err(map_err)?
                .into_iter()
                .map(|value| value.map_or(SqlValue::Null, SqlValue::Bool))
                .collect(),
            _ => {
                let (byte_offset, width) = match ty {
                    SqlType::Int8 | SqlType::Timestamp => (
                        resident_device_int8_column_offset(&side.0.descriptor, table, column)?,
                        8_u8,
                    ),
                    SqlType::Numeric { .. } | SqlType::Uuid => (
                        resident_device_numeric_column_offset(&side.0.descriptor, table, column)?,
                        16_u8,
                    ),
                    SqlType::Int2 | SqlType::Int4 | SqlType::Date => (
                        resident_device_int4_column_offset(&side.0.descriptor, table, column)?,
                        4_u8,
                    ),
                    SqlType::Text | SqlType::Bool => unreachable!(),
                };
                let (raw, valid) = payload
                    .project_fixed_from_join_coordinates(
                        coordinates,
                        relation,
                        payload,
                        byte_offset,
                        validity,
                        width,
                    )
                    .map_err(map_err)?;
                raw.chunks_exact(width as usize)
                    .zip(valid)
                    .map(|(bytes, valid)| {
                        if !valid {
                            return SqlValue::Null;
                        }
                        match ty {
                            SqlType::Int2 => SqlValue::Int2(
                                i32::from_le_bytes(bytes.try_into().expect("int2 width")) as i16,
                            ),
                            SqlType::Int4 => SqlValue::Int4(i32::from_le_bytes(
                                bytes.try_into().expect("int4 width"),
                            )),
                            SqlType::Date => SqlValue::Date(i32::from_le_bytes(
                                bytes.try_into().expect("date width"),
                            )),
                            SqlType::Int8 => SqlValue::Int8(i64::from_le_bytes(
                                bytes.try_into().expect("int8 width"),
                            )),
                            SqlType::Timestamp => SqlValue::Timestamp(i64::from_le_bytes(
                                bytes.try_into().expect("timestamp width"),
                            )),
                            SqlType::Numeric { scale, .. } => SqlValue::Numeric(
                                gpu_db_sql::Decimal128::new(
                                    i128::from_le_bytes(bytes.try_into().expect("numeric width")),
                                    scale,
                                ),
                            ),
                            SqlType::Uuid => {
                                SqlValue::Uuid(bytes.try_into().expect("uuid width"))
                            }
                            SqlType::Text | SqlType::Bool => unreachable!(),
                        }
                    })
                    .collect()
            }
        })
    }

    pub(crate) fn filter_outer_join_projection_coordinates(
        &self,
        tables: &[RelationalTable],
        sides: &[JoinExecSide],
        predicates: &[Option<ResidentExpr>],
        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32,
    ) -> Result<gpu_db_execution::CudaJoinCoordinatesU32, ExecuteError> {
        if tables.len() != sides.len() || predicates.len() != sides.len() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "outer-coordinate filter arity does not match the join".to_string(),
            )));
        }
        let mut masks = Vec::with_capacity(sides.len());
        let mut pad_masks = Vec::with_capacity(sides.len());
        for relation in 0..sides.len() {
            masks.push(self.resident_predicate_device_mask(
                predicates[relation].as_ref(),
                &tables[relation],
                &sides[relation].0.descriptor,
                sides[relation].1.mem(),
                sides[relation].2 as u32,
                sides[relation].3,
            )?);
            pad_masks.push(
                predicates[relation]
                    .as_ref()
                    .map(|predicate| self.predicate_mask_on_null_pad(predicate, &tables[relation]))
                    .transpose()?,
            );
        }
        sides[0]
            .1
            .mem()
            .filter_join_coordinates(
                coordinates,
                &masks.iter().map(Option::as_ref).collect::<Vec<_>>(),
                &pad_masks
                    .iter()
                    .map(|guard| guard.as_ref().and_then(|guard| guard.mask.as_ref()))
                    .collect::<Vec<_>>(),
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    pub(crate) fn filter_join_visibility_coordinates(
        &self,
        tables: &[RelationalTable],
        sides: &[JoinExecSide],
        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32,
    ) -> Result<gpu_db_execution::CudaJoinCoordinatesU32, ExecuteError> {
        let mut masks = Vec::with_capacity(sides.len());
        for relation in 0..sides.len() {
            masks.push(self.resident_predicate_device_mask(
                None,
                &tables[relation],
                &sides[relation].0.descriptor,
                sides[relation].1.mem(),
                sides[relation].2 as u32,
                sides[relation].3,
            )?);
        }
        sides[0]
            .1
            .mem()
            .filter_join_coordinates(
                coordinates,
                &masks.iter().map(Option::as_ref).collect::<Vec<_>>(),
                &vec![None; sides.len()],
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_resident_device_coordinate_join(
        &self,
        plan: &JoinPlan,
        tables: &[RelationalTable],
        sides: &[JoinExecSide],
        step_keys: &[Vec<(usize, usize, usize)>],
        _step_is_text: &[bool],
        _step_is_b128: &[bool],
        projection: &[(usize, usize)],
        predicates: &[Option<ResidentExpr>],
        outer_where: bool,
        row_ranges: Option<&[(u32, u32)]>,
        resolved_order: &[(usize, usize, bool, Option<bool>)],
        gpu_id: u16,
        coordinate_out: Option<&mut Option<gpu_db_execution::CudaJoinCoordinatesU32>>,
        materialized_out: Option<&mut Option<gpu_db_execution::CudaMaterializedRelation>>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        use gpu_db_execution::{CudaJoinOrderKey, CudaJoinPayloadKey, CudaPredicateMaskI32};
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        for (_, _, row_count, _) in sides {
            if *row_count > u32::MAX as usize {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "join row count exceeds the device u32 coordinate range".to_string(),
                )));
            }
        }

        let compile_mask = |relation: usize,
                            predicate: Option<&ResidentExpr>,
                            visibility: Option<ResidentVisibility>|
         -> Result<Option<CudaPredicateMaskI32>, ExecuteError> {
            self.resident_predicate_device_mask(
                predicate,
                &tables[relation],
                &sides[relation].0.descriptor,
                sides[relation].1.mem(),
                sides[relation].2 as u32,
                visibility,
            )
        };

        // Input eligibility always includes MVCC visibility. INNER joins may push their per-relation
        // WHERE into the input mask; OUTER joins retain WHERE as a post-coordinate mask. Streaming block
        // subsets are scheduler metadata and become one additional device mask.
        let mut input_masks: Vec<Option<CudaPredicateMaskI32>> = Vec::with_capacity(sides.len());
        let mut post_masks: Vec<Option<CudaPredicateMaskI32>> = Vec::with_capacity(sides.len());
        let mut pad_masks: Vec<Option<JoinNullPadMask>> =
            (0..sides.len()).map(|_| None).collect();
        for relation in 0..sides.len() {
            let _mask_scope = gpu_db_execution::Probe::scope("join_input_mask");
            let pushed = (!outer_where)
                .then_some(predicates[relation].as_ref())
                .flatten();
            let mut input = compile_mask(relation, pushed, sides[relation].3)?;
            if let Some(ranges) = row_ranges {
                let (start, end) = ranges[relation];
                let range = sides[relation]
                    .1
                    .mem()
                    .row_range_mask_u32(sides[relation].2 as u32, start, end)
                    .map_err(map_err)?;
                input = match input {
                    Some(mask) => Some(
                        sides[relation]
                            .1
                            .mem()
                            .and_predicate_masks(&mask, &range)
                            .map_err(map_err)?,
                    ),
                    None => Some(range),
                };
            }
            input_masks.push(input);
            if outer_where && predicates[relation].is_some() {
                post_masks.push(compile_mask(
                    relation,
                    predicates[relation].as_ref(),
                    None,
                )?);
                pad_masks[relation] = Some(self.predicate_mask_on_null_pad(
                    predicates[relation].as_ref().expect("checked"),
                    &tables[relation],
                )?);
            } else {
                post_masks.push(None);
            }
        }

        let key_descriptor = |relation: usize,
                              column: usize|
         -> Result<CudaJoinPayloadKey<'_>, ExecuteError> {
            let entry = &sides[relation].0;
            let payload = sides[relation].1.mem();
            let validity_bitmap_offset = resident_device_null_column_offset(
                &entry.descriptor,
                &tables[relation],
                column,
            )?;
            match tables[relation].columns[column].ty {
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?;
                    Ok(CudaJoinPayloadKey {
                        payload,
                        byte_offset: layout.offsets_byte_offset,
                        validity_bitmap_offset,
                        width: 255,
                        text_bytes_byte_offset: Some(layout.bytes_byte_offset),
                        text_bytes_len: layout.bytes_len,
                    })
                }
                SqlType::Numeric { .. } | SqlType::Uuid => Ok(CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_numeric_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset,
                    width: 16,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                }),
                SqlType::Int8 | SqlType::Timestamp => Ok(CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_int8_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset,
                    width: 8,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                }),
                SqlType::Int4 | SqlType::Int2 | SqlType::Date => Ok(CudaJoinPayloadKey {
                    payload,
                    byte_offset: resident_device_int4_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?,
                    validity_bitmap_offset,
                    width: 4,
                    text_bytes_byte_offset: None,
                    text_bytes_len: 0,
                }),
                SqlType::Bool => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "bool equi-join keys are not represented as fixed payload keys".to_string(),
                ))),
            }
        };

        let context = sides[0].1.mem();
        let mut coordinates: Option<gpu_db_execution::CudaJoinCoordinatesU32> = None;
        for (step_index, conjuncts) in step_keys.iter().enumerate() {
            let _coordinate_scope =
                gpu_db_execution::Probe::scope("join_fixed_payload_coordinates");
            let right_relation = step_index + 1;
            let left_keys = conjuncts
                .iter()
                .map(|&(relation, column, _)| key_descriptor(relation, column))
                .collect::<Result<Vec<_>, _>>()?;
            let right_keys = conjuncts
                .iter()
                .map(|&(_, _, column)| key_descriptor(right_relation, column))
                .collect::<Result<Vec<_>, _>>()?;
            let left_key_relations = conjuncts
                .iter()
                .map(|&(relation, _, _)| relation as u32)
                .collect::<Vec<_>>();
            coordinates = Some(
                context
                    .join_fixed_payload_coordinates(
                        coordinates.as_ref(),
                        if step_index == 0 {
                            sides[0].2 as u32
                        } else {
                            0
                        },
                        &left_key_relations,
                        &left_keys,
                        sides[right_relation].2 as u32,
                        &right_keys,
                        if step_index == 0 {
                            input_masks[0].as_ref()
                        } else {
                            None
                        },
                        input_masks[right_relation].as_ref(),
                        plan.steps[step_index].outer_left,
                        plan.steps[step_index].outer_right,
                    )
                    .map_err(map_err)?,
            );
        }
        let mut coordinates = coordinates.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "a join plan requires at least one join step".to_string(),
            ))
        })?;

        // Streaming match state consumes an ON-coordinate copy before OUTER-WHERE, ORDER BY, or LIMIT.
        if let Some(out) = coordinate_out {
            *out = Some(
                context
                    .window_join_coordinates(&coordinates, 0, None)
                    .map_err(map_err)?,
            );
        }
        if outer_where {
            coordinates = context
                .filter_join_coordinates(
                    &coordinates,
                    &post_masks.iter().map(Option::as_ref).collect::<Vec<_>>(),
                    &pad_masks
                        .iter()
                        .map(|guard| guard.as_ref().and_then(|guard| guard.mask.as_ref()))
                        .collect::<Vec<_>>(),
                )
                .map_err(map_err)?;
        }

        if !resolved_order.is_empty() {
            let order = resolved_order
                .iter()
                .map(|&(relation, column, descending, nulls_first)| {
                    Ok(CudaJoinOrderKey {
                        relation: relation as u32,
                        key: key_descriptor(relation, column)?,
                        descending,
                        nulls_first: nulls_first.unwrap_or(descending),
                        lexicographic_16: tables[relation].columns[column].ty == SqlType::Uuid,
                    })
                })
                .collect::<Result<Vec<_>, ExecuteError>>()?;
            coordinates = context
                .sort_join_coordinates(&coordinates, &order)
                .map_err(map_err)?;
        }
        if plan.offset.is_some() || plan.limit.is_some() {
            let offset = u32::try_from(plan.offset.unwrap_or(0)).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "join OFFSET exceeds the device coordinate range".to_string(),
                ))
            })?;
            let limit = plan
                .limit
                .map(u32::try_from)
                .transpose()
                .map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "join LIMIT exceeds the device coordinate range".to_string(),
                    ))
                })?;
            coordinates = context
                .window_join_coordinates(&coordinates, offset, limit)
                .map_err(map_err)?;
        }

        let output_aliases = self.join_projection_output_aliases(plan, tables)?;
        let mut columns = Vec::with_capacity(projection.len());
        for (index, &(relation, column)) in projection.iter().enumerate() {
            let mut output = tables[relation].columns[column].clone();
            output.attnum = (index + 1) as i16;
            if let Some(alias) = &output_aliases[index] {
                output.name.clone_from(alias);
            }
            columns.push(output);
        }
        if let Some(out) = materialized_out {
            let mut specs = Vec::with_capacity(projection.len());
            for &(relation, column) in projection {
                let entry = &sides[relation].0;
                let payload = sides[relation].1.mem();
                let validity = resident_device_null_column_offset(
                    &entry.descriptor,
                    &tables[relation],
                    column,
                )?;
                specs.push(match tables[relation].columns[column].ty {
                    SqlType::Text => {
                        let layout = resident_device_text_column_layout(
                            &entry.descriptor,
                            &tables[relation],
                            column,
                        )?;
                        gpu_db_execution::CudaMaterializeJoinColumn::Text {
                            relation: relation as u32,
                            payload,
                            offsets_byte_offset: layout.offsets_byte_offset,
                            bytes_byte_offset: layout.bytes_byte_offset,
                            bytes_len: layout.bytes_len,
                            validity_bitmap_offset: validity,
                        }
                    }
                    SqlType::Bool => gpu_db_execution::CudaMaterializeJoinColumn::Bool {
                        relation: relation as u32,
                        payload,
                        bitmap_byte_offset: resident_device_bool_column_offset(
                            &entry.descriptor,
                            &tables[relation],
                            column,
                        )?,
                        validity_bitmap_offset: validity,
                    },
                    ty => {
                        let (byte_offset, width) = match ty {
                            SqlType::Int8 | SqlType::Timestamp => (
                                resident_device_int8_column_offset(
                                    &entry.descriptor,
                                    &tables[relation],
                                    column,
                                )?,
                                8,
                            ),
                            SqlType::Numeric { .. } | SqlType::Uuid => (
                                resident_device_numeric_column_offset(
                                    &entry.descriptor,
                                    &tables[relation],
                                    column,
                                )?,
                                16,
                            ),
                            SqlType::Int2 | SqlType::Int4 | SqlType::Date => (
                                resident_device_int4_column_offset(
                                    &entry.descriptor,
                                    &tables[relation],
                                    column,
                                )?,
                                4,
                            ),
                            SqlType::Text | SqlType::Bool => unreachable!(),
                        };
                        gpu_db_execution::CudaMaterializeJoinColumn::Fixed {
                            relation: relation as u32,
                            payload,
                            byte_offset,
                            validity_bitmap_offset: validity,
                            width,
                        }
                    }
                });
            }
            *out = Some(
                context
                    .materialize_join_coordinates(&coordinates, &specs)
                    .map_err(map_err)?,
            );
            return Ok(RelationalSelectResult {
                columns: Arc::new(columns),
                rows: Vec::<Vec<SqlValue>>::new().into(),
                planned_target: DeviceTarget::Gpu(gpu_id),
                executed_target: DeviceTarget::Gpu(gpu_id),
                fallback_reason: None,
                access_path: Arc::new(RelationalAccessPath::FullTableScan),
            });
        }

        let mut projected_values: Vec<Vec<SqlValue>> = Vec::with_capacity(projection.len());
        for &(relation, column) in projection {
            let _projection_scope = gpu_db_execution::Probe::scope("join_projection_column");
            if sides[relation].2 == 0 {
                projected_values.push(vec![
                    SqlValue::Null;
                    coordinates.row_count() as usize
                ]);
                continue;
            }
            let entry = &sides[relation].0;
            let payload = sides[relation].1.mem();
            let validity = resident_device_null_column_offset(
                &entry.descriptor,
                &tables[relation],
                column,
            )?;
            let values = match tables[relation].columns[column].ty {
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?;
                    context
                        .project_text_from_join_coordinates(
                            &coordinates,
                            relation as u32,
                            payload,
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            layout.bytes_len,
                            validity,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(|value| value.map_or(SqlValue::Null, SqlValue::Text))
                        .collect()
                }
                SqlType::Bool => {
                    let offset = resident_device_bool_column_offset(
                        &entry.descriptor,
                        &tables[relation],
                        column,
                    )?;
                    context
                        .project_bool_from_join_coordinates(
                            &coordinates,
                            relation as u32,
                            payload,
                            offset,
                            validity,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(|value| value.map_or(SqlValue::Null, SqlValue::Bool))
                        .collect()
                }
                ty => {
                    let (offset, width) = match ty {
                        SqlType::Int8 | SqlType::Timestamp => (
                            resident_device_int8_column_offset(
                                &entry.descriptor,
                                &tables[relation],
                                column,
                            )?,
                            8_u8,
                        ),
                        SqlType::Numeric { .. } | SqlType::Uuid => (
                            resident_device_numeric_column_offset(
                                &entry.descriptor,
                                &tables[relation],
                                column,
                            )?,
                            16_u8,
                        ),
                        SqlType::Int4 | SqlType::Int2 | SqlType::Date => (
                            resident_device_int4_column_offset(
                                &entry.descriptor,
                                &tables[relation],
                                column,
                            )?,
                            4_u8,
                        ),
                        SqlType::Text | SqlType::Bool => unreachable!(),
                    };
                    let (raw, valid) = context
                        .project_fixed_from_join_coordinates(
                            &coordinates,
                            relation as u32,
                            payload,
                            offset,
                            validity,
                            width,
                        )
                        .map_err(map_err)?;
                    raw.chunks_exact(width as usize)
                        .zip(valid)
                        .map(|(bytes, valid)| {
                            if !valid {
                                return SqlValue::Null;
                            }
                            match ty {
                                SqlType::Int4 => SqlValue::Int4(i32::from_le_bytes(
                                    bytes.try_into().expect("4-byte int4"),
                                )),
                                SqlType::Int2 => SqlValue::Int2(i32::from_le_bytes(
                                    bytes.try_into().expect("4-byte int2"),
                                ) as i16),
                                SqlType::Date => SqlValue::Date(i32::from_le_bytes(
                                    bytes.try_into().expect("4-byte date"),
                                )),
                                SqlType::Int8 => SqlValue::Int8(i64::from_le_bytes(
                                    bytes.try_into().expect("8-byte int8"),
                                )),
                                SqlType::Timestamp => SqlValue::Timestamp(i64::from_le_bytes(
                                    bytes.try_into().expect("8-byte timestamp"),
                                )),
                                SqlType::Numeric { scale, .. } => SqlValue::Numeric(
                                    Decimal128::new(
                                        i128::from_le_bytes(
                                            bytes.try_into().expect("16-byte numeric"),
                                        ),
                                        scale,
                                    ),
                                ),
                                SqlType::Uuid => SqlValue::Uuid(
                                    bytes.try_into().expect("16-byte uuid"),
                                ),
                                SqlType::Text | SqlType::Bool => unreachable!(),
                            }
                        })
                        .collect()
                }
            };
            projected_values.push(values);
        }
        let result_rows = (0..coordinates.row_count() as usize)
            .map(|row| projected_values.iter().map(|column| column[row].clone()).collect())
            .collect::<Vec<Vec<SqlValue>>>();
        Ok(RelationalSelectResult {
            columns: Arc::new(columns),
            rows: result_rows.into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        })
    }

    #[allow(clippy::too_many_arguments)] // group_key_expr is threaded alongside the predicate/binding
    pub(crate) fn execute_resident_expr_select_with_binding(
        &self,
        select: &Select,
        table: &RelationalTable,
        // `None` = look up the table payload (descriptor + device buffer + row count) by name in the
        // SINGLE resident store (the whole-table buffer), as before. `Some(src)` INJECTS them so the
        // same executor serves one shard slice (S10c). The identity/validity guards run for both.
        src: Option<&ResidentExecSource>,
        bound: BoundRelationalSelect,
        copin_s: Index,
        // `None` = no WHERE clause: a full-table scan (every row survives).
        predicate: Option<&ResidentExpr>,
        // SV3b/SV6 (MVCC visibility): `Some` = this (unified) buffer carries co-resident i64 version
        // column(s); AND `deleted_by > read_txn_id` (hide tombstoned rows) and/or `created_by <=
        // read_txn_id` (hide too-new appended versions) onto the survivors. `None` = a version-free buffer
        // (the common case) -> no visibility mask, byte-identical. Only the sharded read passes `Some`.
        visibility: Option<ResidentVisibility>,
        // Parallel to `select.order_by`: `Some(expr)` = a SORT EXPRESSION key (`ORDER BY a+b`),
        // evaluated on-device into an i64 key column; `None` = a plain column key. Empty = no ORDER BY.
        order_by_exprs: &[Option<ResidentExpr>],
        // Parallel to `select.order_by`: the explicit NULLS FIRST/LAST override per key (`Some(true)` =
        // NULLS FIRST, `Some(false)` = NULLS LAST, `None` = PG default = NULLS LAST under ASC / FIRST under
        // DESC). Honored ON-DEVICE in the GPU sort comparator via the per-key nulls_first bitmask.
        order_by_nulls_first: &[Option<bool>],
        // `Some(expr)` = GROUP BY an EXPRESSION (`GROUP BY a+b`): materialized on-device into a derived
        // int key column the grouped kernel groups by (via key_base_override). `None` = a plain column
        // GROUP BY (the matrix path) or no GROUP BY.
        group_key_expr: Option<&ResidentExpr>,
        // The GROUP BY key columns. len()==2 = a COMPOSITE key (`GROUP BY a, b`): the two fixed-width
        // int columns are packed on-device into one i64 key (high|low) grouped via key_base_override,
        // and the result key unpacks back to the two columns. len()<=1 = the single-key path (unchanged).
        group_key_columns: &[String],
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Scalar aggregates (operator axis): COUNT(*) -> the surviving-row count; SUM(col) -> a GPU
        // reduction over the filtered column. They have no projected columns to materialize, so they
        // skip the projected-column checks and are computed from the filtered indices below.
        let is_aggregate = matches!(
            select.projection,
            SelectProjection::CountAll
                | SelectProjection::Sum { .. }
                | SelectProjection::Min { .. }
                | SelectProjection::Max { .. }
                | SelectProjection::Avg { .. }
                | SelectProjection::CountDistinct { .. }
        );
        // Grouped aggregates (GROUP BY) emit one row per group from a GPU hash aggregation; they are
        // handled separately below (not via the scalar-aggregate or the plain-projection paths).
        let is_grouped = matches!(
            select.projection,
            SelectProjection::GroupedCount { .. }
                | SelectProjection::GroupedSum { .. }
                | SelectProjection::GroupedAvg { .. }
                | SelectProjection::GroupedMin { .. }
                | SelectProjection::GroupedMax { .. }
                | SelectProjection::GroupedAggregates { .. }
        );
        if !is_aggregate && !is_grouped {
            if bound.selected_indexes.is_empty() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident Expr select requires at least one projected column".to_string(),
                )));
            }
            for &col in &bound.selected_indexes {
                let ty = table.columns[col].ty;
                if ty != SqlType::Int4
                    && ty != SqlType::Int8
                    && !matches!(ty, SqlType::Numeric { .. })
                    && ty != SqlType::Date
                    && ty != SqlType::Timestamp
                    && ty != SqlType::Uuid
                    && ty != SqlType::Int2
                    && ty != SqlType::Bool
                    && ty != SqlType::Text
                {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident Expr select currently materializes int4 / int8 / numeric / date / \
                         timestamp / uuid / int2 / bool / text projection columns only"
                            .to_string(),
                    )));
                }
            }
        }

        // We discard `_query` and keep only `access_path` (metadata). Computing it for an ORDER BY /
        // LIMIT select would run a full CPU ordered table sort (relational_ordered_table_keys) whose
        // result we throw away -- a charter violation (the GPU does the sort here) + 2x work. The CPU
        // sort keys off `bound.order` (set at bind from order_by.first()), which the planner below reads
        // -- NOT `ap_select.order_by` -- so clearing ap_select alone is INEFFECTIVE. Clear bound.order:
        // that alone stops the CPU sort. bound.order is read nowhere else on this path (the GPU/grouped
        // sort uses select.order_by + order_by_exprs), so this is safe. The ap_select clear keeps the
        // synthesized path unordered/unlimited; the GPU sort + host OFFSET/LIMIT own ordering+windowing.
        let mut bound = bound;
        bound.order = None;
        let mut ap_select = select.clone();
        ap_select.order_by.clear();
        ap_select.limit = None;
        ap_select.offset = None;
        // The access-path planner resolves the projection's group COLUMN; an expression GROUP BY has
        // none (the placeholder "(expr)" is not a column), so synthesize a scan path for it. The query
        // is discarded metadata anyway; the grouped GPU aggregation/sort owns execution.
        let access_path = if group_key_expr.is_some() {
            RelationalAccessPath::FullTableScan
        } else {
            let (_query, access_path) =
                self.relational_select_mvcc_query_pinned(&ap_select, table, &bound, copin_s)?;
            access_path
        };
        // The table payload the kernels read: the published device buffer + the catalog/GPU descriptor
        // (its `row_count` + `resident_device_*` section vectors define every column byte-offset) + the
        // row count. `None` looks these up by name in the SINGLE resident store -- byte-identical to the
        // pre-S10c path: one atomic load of the descriptor, then the device-memory cell. `Some(src)`
        // INJECTS them (S10c: one shard slice). The identity + validity guards run for BOTH, so a
        // descriptor that drifted from the catalog or got invalidated is rejected either way. No
        // `host_rows` read on this path: a text GROUP BY key result is materialized ON-DEVICE.
        // THE FLIP: a SHARD-resident table (no single-buffer entry) resolves to the UNIFIED exec source
        // HERE — one resolution point for every `src: None` caller (the SQL->Expr PG path, the
        // parity-test wrapper, the CTAS/view bridges), so the general executor serves sharded tables
        // uniformly (zero-copy at one surviving shard; recompaction otherwise). The unified source
        // carries its own SV3b/SV6 visibility, which OVERRIDES the caller's `None`. R-ver PART 2: a
        // VERSIONED table with a reshaping clause (DISTINCT / GROUP BY / ORDER BY / HAVING) is now
        // SERVED — the visibility resolved below flows into the survivor `indices` (folded in BEFORE
        // group/sort/dedup), so no tombstoned/too-new row reaches a key. Single-buffer tables and
        // `src: Some` callers are byte-identical.
        let sharded_unified: Option<ShardedUnifiedExecSource> = if src.is_none()
            && self.relational_residency_entry(&table.name).is_none()
            && self
                .read_state
                .residency
                .shards
                .load()
                .get(&table.name)
                .is_some_and(|shards| !shards.is_empty())
        {
            Some(self.build_sharded_unified_exec_source(table, predicate, copin_s)?)
        } else {
            None
        };
        debug_assert!(
            sharded_unified.is_none() || visibility.is_none(),
            "a caller passed src: None + visibility: Some for a sharded table — the unified source's \
             visibility would silently replace it (audit P3 hardening; thread it via src instead)"
        );
        let visibility = sharded_unified
            .as_ref()
            .map_or(visibility, |unified| unified.visibility);
        let src = src.or(sharded_unified.as_ref().map(|unified| &unified.src));
        let (snapshot, device_memory, row_count) = match src {
            Some(src) => {
                let snapshot = src.descriptor.clone();
                if snapshot.schema != table.schema || snapshot.table != table.name {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot no longer matches catalog table identity".to_string(),
                    )));
                }
                if !snapshot.is_valid() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" resident snapshot is invalid",
                        table.name
                    ))));
                }
                (snapshot, src.device_memory.clone(), src.row_count)
            }
            None => {
                let residency_entry =
                    self.relational_residency_entry(&table.name)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "relation \"{}\" has no resident snapshot",
                                table.name
                            )))
                        })?;
                let snapshot = residency_entry.descriptor.clone();
                if snapshot.schema != table.schema || snapshot.table != table.name {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot no longer matches catalog table identity".to_string(),
                    )));
                }
                if !snapshot.is_valid() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" resident snapshot is invalid",
                        table.name
                    ))));
                }
                let device_memory = self
                    .read_state
                    .residency
                    .device_memory
                    .get(&table.name)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "relation \"{}\" has no retained resident device memory",
                            table.name
                        )))
                    })?;
                let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot row count exceeds retained device-memory proof range"
                            .to_string(),
                    ))
                })?;
                (snapshot, device_memory, row_count)
            }
        };

        // PG aggregate semantics skip NULL inputs: SUM/AVG/MIN/MAX(col) aggregate the NON-NULL values
        // (all-NULL -> SQL NULL via the empty-set guard below) and COUNT(DISTINCT col) counts distinct
        // NON-NULL values. The scalar reductions below are RAW payload reduces (a NULL row's placeholder
        // is 0 — it would poison MIN toward 0 and inflate AVG's divisor), so when the aggregate column
        // carries a validity bitmap, AND `col IS NOT NULL` into the predicate: the survivors then
        // exclude NULL inputs BEFORE the reduce, AVG's divisor (`indices.len()`) is exactly the non-NULL
        // count, and the existing IsNull lowering + 3VL VM (+ the SV3b/SV6 visibility conjuncts, when
        // versioned) do all the work. COUNT(*) is deliberately NOT augmented (it counts NULL rows). A
        // bitmap-free column (the no-NULLs majority — bitmaps are data-driven) adds no conjunct and the
        // full-scan `(0..n)` fast arm below is untouched: byte-identical to before.
        //
        // LEDGERED trade (audit F1): the conjunct is UNCONDITIONAL. On the no-fallback text path, a
        // nullable NON-int4 aggregate column (or an int4 aggregate whose WHERE makes the augmented
        // predicate mixed-width) is no longer VM-lowerable -> CLEAN ERROR where the pre-fix code
        // returned a NULL-BLIND value (wrong for MIN/MAX/AVG, coincidentally right for SUM). A clean
        // error strictly dominates a silent wrong result; do NOT "fix" this by dropping the conjunct
        // on un-lowerable shapes — route those to the (now NULL-correct) host finalizer instead when
        // the coverage gap matters (type-coverage ledger).
        let aggregate_validity_conjunct: Option<ResidentExpr> = match &select.projection {
            SelectProjection::Sum { column }
            | SelectProjection::Avg { column }
            | SelectProjection::Min { column }
            | SelectProjection::Max { column }
            | SelectProjection::CountDistinct { column } => {
                let col_idx = relational_column_index(table, column)?;
                resident_device_null_column_offset(&snapshot, table, col_idx)?.map(|_| {
                    ResidentExpr::IsNull {
                        col: col_idx,
                        is_not_null: true,
                    }
                })
            }
            _ => None,
        };
        let augmented_predicate: Option<ResidentExpr> =
            aggregate_validity_conjunct.map(|validity| match predicate {
                Some(pred) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(pred.clone()),
                    rhs: Box::new(validity),
                },
                None => validity,
            });
        let predicate = augmented_predicate.as_ref().or(predicate);

        // Evaluate the predicate on the GPU -> surviving row indices (ascending). With no WHERE clause
        // every row survives, so the indices are the full 0..row_count scan (the aggregate + projection
        // paths below are index-driven and need no other change).
        let indices = {
            // probe-timing (VM lever): the WHOLE predicate eval (compare kernel + compact). `compact`
            // (timed separately in compact_mask_*) vs this total localizes where the predicate cost lives.
            let _pred_scope = gpu_db_execution::Probe::scope("predicate_total");
            match predicate {
                Some(predicate) => self.lower_resident_predicate(
                    predicate,
                    table,
                    &snapshot,
                    &device_memory,
                    row_count,
                    visibility,
                )?,
                None => match visibility {
                    // SV3b/SV6: no WHERE but a versioned buffer -> the survivors are exactly the VISIBLE
                    // rows. Run a visibility-only VM program (`deleted_by > read_txn_id` and/or
                    // `created_by <= read_txn_id`) at elem=I64 -- the same mixed-width interpreter, with no
                    // i32 WHERE mask to AND against (the first conjunct IS the mask).
                    Some(vis) => {
                        let mut program: Vec<gpu_db_execution::ExprStep> = Vec::new();
                        vis.push_conjuncts(&mut program, false);
                        device_memory
                            .run_expr_predicate_filter_with_text(
                                &program,
                                &[],
                                row_count,
                                gpu_db_execution::ResidentElemType::I64,
                            )
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?
                    }
                    None => {
                        let n = u32::try_from(row_count).map_err(|_| {
                            ExecuteError::Engine(EngineError::ApplyFailed(
                                "full-table scan row count exceeds the u32 row-index range"
                                    .to_string(),
                            ))
                        })?;
                        (0..n).collect()
                    }
                },
            }
        };
        let indices_u64: Vec<u64> = indices.iter().map(|&i| u64::from(i)).collect();

        // COUNT(DISTINCT v) over a (group key, value) tuple: GPU-sort (g, value) ASC + mark the first
        // row of each distinct tuple -> per-group SUM of the new-distinct flags = the per-group distinct
        // count. Shared by the grouped branch (g = the real group key) AND the scalar form below (g = a
        // constant 0 -> ONE group whose count = the total distinct). A FIXED-WIDTH value packs into an
        // i64 multikey matrix (int = k2 (g, v); numeric/uuid = k3 (g, v_hi, v_lo)); a varlen TEXT value
        // routes through the hetero sort + the text-aware mark (reads the value text on-device). The
        // sort/mark/SUM run ENTIRELY on the GPU; the host marshals only the control-plane g / index
        // arrays. `g_vals`/`idx_u64` are parallel, length n (the surviving positions). Empty -> caller
        // guards n == 0.
        let count_distinct_groups = |value_idx: usize,
                                     g_vals: &[i64],
                                     idx_u64: &[u64]|
         -> Result<
            Vec<gpu_db_execution::GroupByI32Row>,
            ExecuteError,
        > {
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            let n = idx_u64.len();
            let value_ty = table.columns[value_idx].ty;
            let (g_sorted, new_distinct) = match value_ty {
                SqlType::Int2
                | SqlType::Int4
                | SqlType::Int8
                | SqlType::Date
                | SqlType::Timestamp
                | SqlType::Numeric { .. }
                | SqlType::Uuid => {
                    let (matrix, k) = if matches!(value_ty, SqlType::Numeric { .. } | SqlType::Uuid)
                    {
                        let off =
                            resident_device_numeric_column_offset(&snapshot, table, value_idx)?;
                        let v128 = device_memory
                            .project_i128_rows_from_payload(off, idx_u64)
                            .map_err(map_err)?;
                        let mut m = Vec::with_capacity(n * 3);
                        for i in 0..n {
                            m.push(g_vals[i]);
                            m.push((v128[i] >> 64) as i64);
                            m.push((v128[i] as u64) as i64);
                        }
                        (m, 3usize)
                    } else {
                        let v_vals: Vec<i64> = match value_ty {
                            SqlType::Int8 | SqlType::Timestamp => device_memory
                                .project_i64_rows_from_payload(
                                    resident_device_int8_column_offset(
                                        &snapshot, table, value_idx,
                                    )?,
                                    idx_u64,
                                )
                                .map_err(map_err)?,
                            _ => device_memory
                                .project_i32_rows_from_payload(
                                    resident_device_int4_column_offset(
                                        &snapshot, table, value_idx,
                                    )?,
                                    idx_u64,
                                )
                                .map_err(map_err)?
                                .into_iter()
                                .map(i64::from)
                                .collect(),
                        };
                        let mut m = Vec::with_capacity(n * 2);
                        for i in 0..n {
                            m.push(g_vals[i]);
                            m.push(v_vals[i]);
                        }
                        (m, 2usize)
                    };
                    let perm = device_memory
                        .bitonic_sort_multikey(&matrix, n, k, 0)
                        .map_err(map_err)?;
                    device_memory
                        .mark_new_distinct_device(&matrix, &perm, n as u64, k)
                        .map_err(map_err)?
                }
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(&snapshot, table, value_idx)?;
                    let key_plan: Vec<u32> = vec![0, 0x4000_0000_u32];
                    let perm = device_memory
                        .bitonic_sort_hetero(
                            idx_u64,
                            g_vals,
                            1,
                            &[(layout.offsets_byte_offset, layout.bytes_byte_offset)],
                            &[],
                            &key_plan,
                            0,
                            // COUNT(DISTINCT) (g, text_v) reps: both keys are non-NULL by construction, so
                            // there is no NULL bitmap and the nulls_first mask is irrelevant.
                            &[],
                            0,
                        )
                        .map_err(map_err)?;
                    device_memory
                        .mark_new_distinct_text_device(
                            &perm,
                            idx_u64,
                            g_vals,
                            gpu_db_execution::CudaGroupTextSource {
                                offsets_byte_offset: layout.offsets_byte_offset,
                                bytes_byte_offset: layout.bytes_byte_offset,
                                bytes_len: layout.bytes_len,
                                row_count,
                            },
                            n as u64,
                        )
                        .map_err(map_err)?
                }
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "COUNT(DISTINCT) value column must be int/numeric/uuid/text on the \
                         GPU path"
                            .to_string(),
                    )));
                }
            };
            // SUM(new_distinct) grouped by g_sorted = the per-group distinct count. indices = 0..n (the
            // sorted positions); the kernel reads g_sorted / new_distinct (both i64) via the
            // key/value_base_override. The leases live across this synced call.
            let scan_indices: Vec<u32> = (0..n as u32).collect();
            let cd_groups = device_memory
                .group_by_i32_count_sum_minmax_from_payload(
                    gpu_db_execution::CudaGroupByInput {
                        key: gpu_db_execution::CudaGroupKeySource::Fixed(
                            gpu_db_execution::CudaGroupFixedSource::Derived {
                                buffer: g_sorted.group_view(),
                                width: 8,
                                row_count: n as u64,
                            },
                        ),
                        value: gpu_db_execution::CudaGroupValueSource::Fixed(
                            gpu_db_execution::CudaGroupFixedSource::Derived {
                                buffer: new_distinct.group_view(),
                                width: 8,
                                row_count: n as u64,
                            },
                        ),
                        key_validity_bitmap_offset: None,
                        value_validity_bitmap_offset: None,
                    },
                    &scan_indices,
                    // COUNT(DISTINCT) SUM-of-new-distinct pass: the result builder reads this pass's
                    // `.sum`, so it must compute SUM (and COUNT). ALL is correct + behavior-preserving.
                    gpu_db_execution::grouped_agg_mask::ALL,
                )
                .map_err(map_err)?;
            drop((g_sorted, new_distinct));
            Ok(cd_groups)
        };

        // Grouped aggregate (GROUP BY <int4 key>): GPU hash aggregation over the filtered rows -> one
        // row per distinct key. COUNT/SUM/AVG share the count+sum kernel; grouped MIN/MAX is a
        // follow-on. PG does not order GROUP BY without ORDER BY; sort by key for determinism.
        if is_grouped {
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            let SelectProjection::GroupedAggregates {
                group_column: group_name,
                aggregates,
            } = &select.projection
            else {
                unreachable!("is_grouped gates on GroupedAggregates on the Expr path");
            };
            // QUERY-AWARE AGGREGATE PRUNING (this slice): the DIRECT (hash) GROUP BY kernel computes
            // count+sum+min+max for one value column, but the executor reads only the field(s) the query
            // needs. Build a query-wide mask (`grouped_agg_mask`: 1=COUNT 2=SUM 4=MIN 8=MAX) = the OR of
            // every aggregate's field set, and pass it to EVERY direct pass (each computes a superset of
            // what its own column needs, and the executor reads only computed fields, so one shared mask
            // is correct). Bit derivation per aggregate kind:
            //   Count            -> COUNT
            //   Sum / Avg        -> SUM   (Avg also divides by count)
            //   Min              -> MIN
            //   Max              -> MAX
            //   CountDistinct    -> {} here (it runs a SEPARATE sort/mark/SUM pass; the direct kernel
            //                       reads NOTHING for it -- its passes always run their own ALL mask)
            // COUNT is FORCED ON whenever ANY value aggregate (Sum/Avg/Min/Max) is present, because the
            // result builder reads `g.count` for EVERY value pass to detect an all-NULL group (count==0 ->
            // SQL NULL); the count slot inits to 0, so a masked-out COUNT would read 0 and wrongly NULL
            // every group. (Avg already requires COUNT for its divisor.)
            //
            // COMPLETENESS vs HAVING / aggregate-ORDER-BY (the correctness-critical claim): this mask covers
            // them WITHOUT an ALL-fallback, and here is the proof. HAVING and ORDER BY on a grouped result
            // do NOT read the kernel's count/sum/min/max directly -- they consume the already-materialized
            // result `rows` (HAVING builds a transient device relation from `rows`; ORDER BY GPU-sorts result
            // COLUMNS). Both resolve their referenced column via `result_column_name` to a NAME, which the
            // executor's `col_index` maps against `bound.selected_columns` (the SELECT result columns). A
            // HAVING/ORDER-BY reference to an aggregate that is NOT a SELECT result column is a hard error
            // ("references unknown column"), so EVERY aggregate HAVING/ORDER-BY can reach is necessarily
            // already in the SELECT list -> already in `aggregates` -> already in this mask. (The grouped SQL
            // binder also builds `aggregates` only from the SELECT target list; it adds no HAVING/ORDER-BY
            // aggregate, which is exactly why such an unlisted reference errors rather than introducing a new
            // kernel field read.) Hence the OR-over-`aggregates` mask is the COMPLETE set the executor reads.
            let agg_mask: u32 = {
                use gpu_db_execution::grouped_agg_mask as gm;
                let mut m = 0u32;
                let mut any_value_agg = false;
                for a in aggregates.iter() {
                    match a.kind {
                        GroupedAggKind::Count => m |= gm::COUNT,
                        GroupedAggKind::Sum | GroupedAggKind::Avg => {
                            m |= gm::SUM;
                            any_value_agg = true;
                        }
                        GroupedAggKind::Min => {
                            m |= gm::MIN;
                            any_value_agg = true;
                        }
                        GroupedAggKind::Max => {
                            m |= gm::MAX;
                            any_value_agg = true;
                        }
                        // COUNT(DISTINCT) reads nothing from the direct kernel (separate pass).
                        GroupedAggKind::CountDistinct => {}
                    }
                }
                if any_value_agg {
                    m |= gm::COUNT; // the all-NULL-group check reads g.count on every value pass
                }
                m
            };
            // Resolve each aggregate's value column to an index (None for COUNT(*)), and the distinct
            // value columns in first-seen order -- one grouping pass per distinct value column, since
            // the kernel yields count+sum+min+max for one value column.
            let agg_value_indices: Vec<Option<usize>> = aggregates
                .iter()
                .map(|a| {
                    a.value_column
                        .as_deref()
                        .map(|c| relational_column_index(table, c))
                        .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?;
            // The DIRECT (hash) passes -- one per distinct value column of a COUNT/SUM/AVG/MIN/MAX
            // aggregate. A COUNT(DISTINCT v) column is NOT added here: it runs a separate sort-based
            // pass (added after the direct passes), unless another DIRECT aggregate also needs it.
            let mut value_indices: Vec<usize> = Vec::new();
            for (aggregate, value_idx) in aggregates.iter().zip(&agg_value_indices) {
                if aggregate.kind == GroupedAggKind::CountDistinct {
                    continue;
                }
                if let Some(value_idx) = value_idx {
                    if !value_indices.contains(value_idx) {
                        value_indices.push(*value_idx);
                    }
                }
            }
            // M3 (doc 21) GROUP BY 3VL: a NULL group KEY forms its OWN group (NULLs group together,
            // distinct from real keys). The kernel routes a NULL key (of ANY single-column type) to a
            // dedicated reserved slot. Supported for a SINGLE COLUMN key of int2/4/8/date/timestamp / text /
            // numeric / uuid. A nullable COMPOSITE / EXPRESSION key is still a clean-error follow-up: its
            // per-member NULL semantics ((NULL,5) ≠ (NULL,6) ≠ (1,5)) need NULL encoded into the key, not
            // one reserved slot. A NULL-free key has no bitmap, so it runs unchanged either way.
            let single_key_col = if group_key_expr.is_none() && group_key_columns.len() < 2 {
                relational_column_index(table, group_name).ok()
            } else {
                None
            };
            let key_is_single_nullable_column = single_key_col.is_some_and(|idx| {
                matches!(
                    table.columns.get(idx).map(|column| column.ty),
                    Some(
                        SqlType::Int2
                            | SqlType::Int4
                            | SqlType::Int8
                            | SqlType::Date
                            | SqlType::Timestamp
                            | SqlType::Text
                            | SqlType::Numeric { .. }
                            | SqlType::Uuid
                    )
                )
            });
            // M3 (doc 21): a COMPOSITE GROUP BY key (`GROUP BY a, b`) with a nullable member groups via the
            // general wide-key path with PER-MEMBER NULL validity (gpu_db_build_wide_key writes a trailing
            // validity word; the claim hashes + memcmps it, so (NULL,5) != (0,5) != (NULL,6) are distinct).
            // A nullable TEXT member is a clean-error follow-up (text NULL in the wide key needs the text
            // descriptor's verify to be NULL-aware). Detected here so the routing + clean-error below agree.
            let composite_key_has_null = if group_key_columns.len() >= 2 {
                let mut any = false;
                for name in group_key_columns {
                    let idx = relational_column_index(table, name)?;
                    if resident_device_null_column_offset(&snapshot, table, idx)?.is_some() {
                        if matches!(table.columns[idx].ty, SqlType::Text) {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "GROUP BY a composite key with a nullable TEXT member is a follow-up on \
                                 the GPU (M3 3VL; a nullable fixed-width member is supported)"
                                    .to_string(),
                            )));
                        }
                        any = true;
                    }
                }
                any
            } else {
                false
            };
            // M3 (doc 21): an EXPRESSION group key (`a + b`) is NULL exactly when an operand is NULL. If it
            // references EXACTLY ONE nullable operand column, the expression's NULL group IS that column's
            // NULL rows, so reuse the single-column NULL-key reserved slot (pass_key_null_off = that
            // column's validity bitmap). The kernel routes by validity BEFORE reading the derived key, so a
            // NULL row's garbage derived value is never used, and the NULL group renders SqlValue::Null via
            // the existing key_is_null path -- ZERO kernel change, int4 or int8 expression alike. Two+
            // nullable operands need a DERIVED validity (the AND of the operand bitmaps): a clean-error
            // follow-up.
            let expr_key_single_null_off: Option<u64> = if let Some(expr) = group_key_expr {
                let mut cols = Vec::new();
                collect_expr_columns(expr, &mut cols);
                cols.sort_unstable();
                cols.dedup();
                let mut nullable_offs = Vec::new();
                for c in cols {
                    if let Some(off) = resident_device_null_column_offset(&snapshot, table, c)? {
                        nullable_offs.push(off);
                    }
                }
                // exactly one nullable operand -> its bitmap is the expression's validity; 0 = no NULLs
                // (unchanged); >1 -> None here, caught by the clean-error loop below.
                (nullable_offs.len() == 1).then(|| nullable_offs[0])
            } else {
                None
            };
            if !key_is_single_nullable_column
                && expr_key_single_null_off.is_none()
                && !composite_key_has_null
            {
                let mut group_by_key_check: Vec<usize> = Vec::new();
                if let Some(expr) = group_key_expr {
                    collect_expr_columns(expr, &mut group_by_key_check);
                } else if !group_key_columns.is_empty() {
                    for name in group_key_columns {
                        group_by_key_check.push(relational_column_index(table, name)?);
                    }
                } else if let Ok(idx) = relational_column_index(table, group_name) {
                    group_by_key_check.push(idx);
                }
                for col in group_by_key_check {
                    if resident_device_null_column_offset(&snapshot, table, col)?.is_some() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "GROUP BY over an EXPRESSION key with >1 nullable operand (or a bool key) is \
                             not yet supported on the GPU (M3 3VL follow-up: it needs a DERIVED NULL \
                             encoding; a single int/date/timestamp/text/numeric/uuid column key, a \
                             composite of fixed-width members, or an expression over exactly one nullable \
                             operand, is supported)"
                                .to_string(),
                        )));
                    }
                }
            }
            // M3 (doc 21): a nullable NUMERIC value runs the numeric TWO-PASS min/max kernel — pass 1
            // records each NON-NULL row's claimed slot into a POOLED `row_slots` scratch and skips NULL
            // rows (the value-skip gate at do_agg), pass 2 (gpu_db_group_by_numeric_minmax_lo) folds the
            // i128 low limb. BOTH passes now read the value validity bitmap and skip NULL rows, so a NULL
            // row's stale pooled slot is never folded (pass 2 gained `value_null_off`, matching pass 1).
            // All-NULL group → count 0 → SqlValue::Null at finalization (shared with int/int8). So a
            // nullable numeric aggregate value is supported, like int2/4/8/date/timestamp/uuid/text.
            // GROUP BY <expression> (`a+b`): the key is a DERIVED int buffer materialized on-device
            // below (grouped via key_base_override), NOT a column -- so group_idx is only a placeholder
            // for the COUNT(*) pass (which reads no value), and key_ty is the expression result type.
            // COMPOSITE GROUP BY (`GROUP BY a, b`): the on-device pack foundation
            // (gpu_db_pack_two_int4_cols / pack_two_int4_cols_device) is landed, but the N-column result
            // wiring -- build_grouped_projection accepting 2 leading group targets, bound.selected_columns
            // carrying both columns, and the result-row UNPACK of the packed i64 key back into the two
            // columns -- is a coupled change to the most-audited grouped path that belongs in its own
            // auditable slice. Until then, reject cleanly (NOT a silent single-key grouping by the first
            // column, which `select.group_by` carries).
            // COMPOSITE GROUP BY (`GROUP BY a, b[, ...]`). TWO fast paths for a 2-member key, packed
            // on-device into ONE derived key (key_base_override) + UNPACKED in the result:
            //  - both fixed-width int (int2/int4/int8/date/timestamp): one i64 `(c0<<32)|c1`
            //    (gpu_db_pack_two_int4_cols) or, when a member is int8/timestamp (> 64 bits combined),
            //    one i128 `c0:c1` (gpu_db_pack_two_cols_i128 -> the b128 key path);
            //  - one text + one fixed-int: the text-key b128 claim with the fixed member folded in.
            // The GENERAL path (composite_is_widekey) handles every OTHER composite -- >2 columns, a
            // numeric/uuid member, a tuple > 128 bits, AND any text member beyond the 2-member (fixed,
            // text) case (two-text, text in a >2 key) -- via a fixed-width WIDE KEY buffer
            // (gpu_db_build_wide_key, for the fixed members) PLUS a text-member descriptor; the claim
            // hashes + verifies both. The result reads each member from the representative row.
            let is_text = |t: SqlType| matches!(t, SqlType::Text);
            let is_fixed_int = |t: SqlType| {
                matches!(
                    t,
                    SqlType::Int4
                        | SqlType::Int2
                        | SqlType::Date
                        | SqlType::Int8
                        | SqlType::Timestamp
                )
            };
            let is_widekey_member = |t: SqlType| {
                is_fixed_int(t)
                    || matches!(t, SqlType::Numeric { .. } | SqlType::Uuid | SqlType::Bool)
            };
            let composite_members: Option<Vec<(usize, SqlType)>> = if group_key_columns.len() >= 2 {
                let mut m = Vec::with_capacity(group_key_columns.len());
                for name in group_key_columns {
                    let idx = relational_column_index(table, name)?;
                    m.push((idx, table.columns[idx].ty));
                }
                Some(m)
            } else {
                None
            };
            // 2-member fast path: both fixed-int, OR exactly one text + one fixed-int. A NULLABLE composite
            // takes the general WIDE-KEY path instead (composite_cols = None) -- the i64/i128 pack has no
            // room for a per-member validity bit, but the wide key carries a validity word.
            let composite_cols: Option<(usize, SqlType, usize, SqlType)> = match &composite_members
            {
                Some(m) if m.len() == 2 && !composite_key_has_null => {
                    let (c0, t0) = m[0];
                    let (c1, t1) = m[1];
                    let two_fixed = is_fixed_int(t0) && is_fixed_int(t1);
                    let fixed_text =
                        (is_text(t0) && is_fixed_int(t1)) || (is_fixed_int(t0) && is_text(t1));
                    if two_fixed || fixed_text {
                        Some((c0, t0, c1, t1))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            // General WIDE KEY path: an N>=2 composite NOT taken by the 2-member fast path. Every member
            // is fixed-width (int/numeric/uuid) OR text -- the fixed members concatenate into the
            // comp_w wide-key buffer; the text members ride a descriptor the claim folds into the hash +
            // byte-verifies (so two-text and text-in-a->2 keys are covered). bool members are a follow-up.
            let widekey_cols: Option<Vec<(usize, SqlType)>> = match &composite_members {
                Some(m) if composite_cols.is_none() => {
                    if m.iter().all(|&(_, t)| is_widekey_member(t) || is_text(t)) {
                        Some(m.clone())
                    } else {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "composite GROUP BY supports fixed-width members \
                             (int2/int4/int8/date/timestamp/numeric/uuid) and text members, any \
                             count, on the Expr path (a bool member is a follow-up)"
                                .to_string(),
                        )));
                    }
                }
                _ => None,
            };
            let composite_is_widekey = widekey_cols.is_some();
            let is_composite_key = composite_cols.is_some();
            // A composite with exactly one TEXT member groups via the text-key b128 claim with the
            // OTHER (fixed-width) member folded into the hash/verify (key_base_override). Otherwise both
            // members are fixed: i128-packed iff a member is int8/timestamp (combined width > 64 bits),
            // else i64-packed. Drives key_is_text / key_is_i128 / key_is_int8 / the pack / the result.
            let composite_is_text = composite_cols.is_some_and(|(_, t0, _, t1)| {
                matches!(t0, SqlType::Text) != matches!(t1, SqlType::Text)
            });
            let composite_is_i128 = !composite_is_text
                && composite_cols.is_some_and(|(_, t0, _, t1)| {
                    matches!(t0, SqlType::Int8 | SqlType::Timestamp)
                        || matches!(t1, SqlType::Int8 | SqlType::Timestamp)
                });
            // Composite-text + wide-key use a (rep_idx, hash) claim -> the per-pass group order is
            // rep-row (race) based, so the multi-pass alignment by a per-group key is a follow-up;
            // restrict both to a SINGLE aggregate here.
            if (composite_is_text || composite_is_widekey) && aggregates.len() > 1 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "a (fixed, text) or general all-fixed wide-key composite GROUP BY supports a \
                     single aggregate on the Expr path (multiple aggregates are a follow-up)"
                        .to_string(),
                )));
            }
            let is_expr_key = group_key_expr.is_some();
            let key_expr_is_int8 = match group_key_expr {
                Some(expr) => expr_mentions_int8(expr, table),
                None => false,
            };
            // The arith VM is mono-typed (one element width per program): a MIXED int4/int8 expression
            // key would load an int4 column with the i64 kernel (wrong stride -> reads off the section ->
            // garbage). Reject it (honest error, not a wrong answer) -- pure int4 or pure int8 work;
            // widening int4->i64 in the VM is a follow-up. Covers the COUNT(DISTINCT) reduction too (it
            // reuses this expr key buffer).
            if let Some(expr) = group_key_expr {
                if key_expr_is_int8 && expr_mentions_int4_column(expr, table) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "GROUP BY a mixed int4/int8 arithmetic expression is not supported (the \
                         on-device arith program is mono-typed); cast the operands to one width"
                            .to_string(),
                    )));
                }
            }
            let group_idx = if is_composite_key {
                // The packed key is grouped via key_base_override; group_idx is only the COUNT(*) pass's
                // value placeholder (reads no value), so col0's index is a valid placeholder.
                composite_cols.unwrap().0
            } else if composite_is_widekey {
                // Wide-key: grouped via the wide-key buffer; group_idx is just the COUNT(*) placeholder.
                widekey_cols.as_ref().unwrap()[0].0
            } else if is_expr_key {
                0
            } else {
                relational_column_index(table, group_name)?
            };
            let key_ty = if is_composite_key || composite_is_widekey {
                // composite: the derived key (packed i64/i128, or the wide-key b128 slot) -- the result
                // reconstructs the member columns separately, so Int8 is a neutral placeholder that does
                // NOT trigger the numeric/uuid/text single-key paths.
                SqlType::Int8
            } else if is_expr_key {
                if key_expr_is_int8 {
                    SqlType::Int8
                } else {
                    SqlType::Int4
                }
            } else {
                table.columns[group_idx].ty
            };
            // GROUP BY key: int2/int4/date ride the int4 (4-byte) section; int8/timestamp ride the int8
            // (8-byte) section, which forces the single-level kernel (only it reads 64-bit keys + routes
            // the i64::MIN key, which collides with EMPTY, to its dedicated slot). An expression key is
            // int4/int8 by its result width.
            let key_is_int8 = if composite_is_widekey {
                // The wide-key path reads its key via comp_w (NOT the int8/i128/text single-key paths).
                false
            } else if is_composite_key {
                // A two-fixed composite packs into one derived key: the i64 pack reads via the int8 key
                // path; the i128 pack via the i128 path; a (fixed, text) composite via the text path
                // (key_is_int8 = false for both i128 and text).
                !composite_is_i128 && !composite_is_text
            } else if is_expr_key {
                key_expr_is_int8
            } else {
                match key_ty {
                    SqlType::Int4 | SqlType::Int2 | SqlType::Date => false,
                    SqlType::Int8 | SqlType::Timestamp => true,
                    SqlType::Numeric { .. } | SqlType::Uuid => false,
                    SqlType::Text => false,
                    // A bool key is materialized bool->int4 (0/1) into a derived buffer + grouped via
                    // key_base_override on the int4 path -- avoids a bool GROUP BY kernel + its hazard.
                    SqlType::Bool => false,
                    // EXHAUSTIVE (the prior bool lesson): a NEW SqlType is a COMPILE error here, forcing
                    // its GROUP-BY-key handling to be considered rather than silently mis-grouped.
                }
            };
            // numeric / uuid GROUP BY keys are 128-bit -- claimed via atom.cas.b128 into slot_keys_i128
            // (the single-level kernel). A WIDER COMPOSITE key (int8/timestamp member) is also packed into
            // an i128 and uses the same b128 claim. An expression key is never i128/text. key_scale
            // carries the numeric column scale onto the result key.
            let key_is_i128 = composite_is_i128
                || (!is_expr_key && matches!(key_ty, SqlType::Numeric { .. } | SqlType::Uuid));
            // A TEXT key is varlen: the kernel hashes the bytes, claims a b128 (rep_row_idx, hash) in
            // slot_keys_i128 with a full-text verify-on-lost-CAS, and the result key is read host-side
            // from the representative row. key_offsets_off/key_bytes_off locate the Arrow varlen column.
            // A (fixed, text) composite ALSO uses the text-key b128 claim, with the fixed member folded
            // into the hash/verify via key_base_override (set below).
            let key_is_text =
                composite_is_text || (!is_expr_key && matches!(key_ty, SqlType::Text));
            // A bool key is materialized bool->int4 (0/1) into a derived buffer + grouped via
            // key_base_override (like an expression key); int4-width, never i128/text.
            let key_is_bool = !is_expr_key && matches!(key_ty, SqlType::Bool);
            let (key_offsets_off, key_bytes_off, key_bytes_len) = if composite_is_text {
                // The text member's varlen column (the other member rides key_base_override).
                let (c0, t0, c1, _) = composite_cols.unwrap();
                let text_col = if matches!(t0, SqlType::Text) { c0 } else { c1 };
                let layout = resident_device_text_column_layout(&snapshot, table, text_col)?;
                (
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    layout.bytes_len,
                )
            } else if key_is_text {
                let layout = resident_device_text_column_layout(&snapshot, table, group_idx)?;
                (
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    layout.bytes_len,
                )
            } else {
                (0, 0, 0)
            };
            let key_scale: u8 = match key_ty {
                SqlType::Numeric { scale, .. } => scale,
                _ => 0,
            };
            // The key BYTE offset, unused when key_base_override is set (the kernel then reads the
            // override base + idx*stride, not resident_base + key_offset).
            let key_offset = if is_expr_key
                || is_composite_key
                || composite_is_widekey
                || key_is_text
                || key_is_bool
            {
                0
            } else if key_is_i128 {
                resident_device_numeric_column_offset(&snapshot, table, group_idx)?
            } else if key_is_int8 {
                resident_device_int8_column_offset(&snapshot, table, group_idx)?
            } else {
                resident_device_int4_column_offset(&snapshot, table, group_idx)?
            };
            // The wide-key width (bytes/row) of the FIXED members only: each int member 8 bytes (i64),
            // each numeric/uuid 16; text members contribute 0 here (they ride the text descriptor, not
            // the fixed buffer). 0 when not a wide-key composite OR a pure all-text composite. Drives
            // comp_w on the GROUP BY launch + the build descriptors.
            let widekey_w: u64 = widekey_cols.as_ref().map_or(0, |m| {
                let fixed: u64 = m
                    .iter()
                    .map(|&(_, t)| match t {
                        SqlType::Numeric { .. } | SqlType::Uuid => 16,
                        SqlType::Text => 0,
                        _ => 8,
                    })
                    .sum();
                // M3 (doc 21): a nullable composite reserves a trailing 8-byte validity word (bit per fixed
                // member). gpu_db_build_wide_key writes it at widekey_w-8; the claim memcmps all widekey_w.
                fixed + if composite_key_has_null { 8 } else { 0 }
            });
            // GROUP BY <expression>: materialize the expr ONCE over all rows into a resident device key
            // buffer (checked overflow -> PG error) + hold its typed view alive across EVERY pass.
            let _derived_key_buf;
            if let Some(expr) = group_key_expr {
                if row_count == 0 {
                    // Empty input: the on-device arith materialize rejects n=0, and there are no rows to
                    // group anyway. Skip it and group 0 rows -> 0 groups (PG returns no rows), matching
                    // the plain-column path's empty-table behavior.
                    _derived_key_buf = None;
                } else {
                    let mut program = Vec::new();
                    compile_arith_program(expr, table, &snapshot, &mut program)?;
                    let elem = if key_expr_is_int8 {
                        ResidentElemType::I64
                    } else {
                        ResidentElemType::I32
                    };
                    let buf = device_memory
                        .arith_value_column_device(&program, row_count, elem)
                        .map_err(|e| {
                            ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                        })?;
                    _derived_key_buf = Some(buf);
                }
            } else if key_is_bool {
                if row_count == 0 {
                    _derived_key_buf = None;
                } else {
                    // GROUP BY a bool column: materialize bool->int4 (0/1) into a typed derived buffer.
                    // No bool GROUP BY kernel -> no hazard.
                    let offset = resident_device_bool_column_offset(&snapshot, table, group_idx)?;
                    let buf = device_memory
                        .bool_to_int4_column_device(offset, row_count)
                        .map_err(|e| {
                            ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                        })?;
                    _derived_key_buf = Some(buf);
                }
            } else if is_composite_key {
                if row_count == 0 {
                    _derived_key_buf = None;
                } else {
                    // GROUP BY a, b: pack the two members into one derived key; the result UNPACKS it.
                    // Both-int4 -> one i64 ((col0<<32)|col1);
                    // a wider member (int8/timestamp) -> one i128 (col0 high 64, col1 low 64) read by
                    // the b128 claim. Each member's width selects its section offset + the pack arg.
                    let (c0, t0, c1, t1) = composite_cols.unwrap();
                    let col_off = |idx: usize, ty: SqlType| -> Result<u64, ExecuteError> {
                        if matches!(ty, SqlType::Int8 | SqlType::Timestamp) {
                            resident_device_int8_column_offset(&snapshot, table, idx)
                        } else {
                            resident_device_int4_column_offset(&snapshot, table, idx)
                        }
                    };
                    let width = |ty: SqlType| -> u64 {
                        if matches!(ty, SqlType::Int8 | SqlType::Timestamp) {
                            8
                        } else {
                            4
                        }
                    };
                    let buf = if composite_is_text {
                        // (fixed, text): widen the FIXED member to an i64 derived buffer; the text-key
                        // claim reads the text column directly + folds this buffer in (hash + verify).
                        let (fc, ft) = if matches!(t0, SqlType::Text) {
                            (c1, t1)
                        } else {
                            (c0, t0)
                        };
                        device_memory
                            .widen_col_to_i64_device(col_off(fc, ft)?, width(ft), row_count)
                            .map_err(|e| {
                                ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                            })?
                    } else if composite_is_i128 {
                        let off0 = col_off(c0, t0)?;
                        let off1 = col_off(c1, t1)?;
                        device_memory
                            .pack_two_cols_i128_device(off0, width(t0), off1, width(t1), row_count)
                            .map_err(|e| {
                                ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                            })?
                    } else {
                        let off0 = col_off(c0, t0)?;
                        let off1 = col_off(c1, t1)?;
                        device_memory
                            .pack_two_int4_cols_device(off0, off1, row_count)
                            .map_err(|e| {
                                ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                            })?
                    };
                    _derived_key_buf = Some(buf);
                }
            } else if composite_is_widekey {
                if row_count == 0 || widekey_w == 0 {
                    // Empty input, OR a pure all-text composite (no fixed members) -> no fixed wide-key
                    // buffer; the text members are grouped via the text descriptor.
                    _derived_key_buf = None;
                } else {
                    // General composite: build the widekey_w-byte/row FIXED wide key (each fixed member's
                    // canonical bytes concatenated -- int 8B, numeric/uuid 16B; TEXT members are skipped
                    // here, handled by the text descriptor) -> grouped via the (rep_idx, hash) b128 claim
                    // (comp_w = widekey_w). The result reads each member from the rep row.
                    let members = widekey_cols.as_ref().expect("composite_is_widekey");
                    let mut descriptors: Vec<gpu_db_execution::CudaWideKeyDescriptor<'_>> =
                        Vec::with_capacity(members.len());
                    // M3 (doc 21): per-FIXED-member NULL validity offsets, lockstep with `descriptors`
                    // (text members are skipped in both). Empty unless the composite is nullable.
                    let mut validity_descs: Vec<gpu_db_execution::CudaWideKeyValidity> = Vec::new();
                    let mut dst_off: u64 = 0;
                    for &(idx, ty) in members {
                        let (source, w) = match ty {
                            SqlType::Numeric { .. } | SqlType::Uuid => (
                                gpu_db_execution::CudaWideKeySource::ResidentI128 {
                                    byte_offset: resident_device_numeric_column_offset(
                                        &snapshot, table, idx,
                                    )?,
                                },
                                16u64,
                            ),
                            SqlType::Int8 | SqlType::Timestamp => (
                                gpu_db_execution::CudaWideKeySource::ResidentI64 {
                                    byte_offset: resident_device_int8_column_offset(
                                        &snapshot, table, idx,
                                    )?,
                                },
                                8u64,
                            ),
                            // A bool member (1-byte resident) is widened 0/1 -> i64 by build kind 3.
                            SqlType::Bool => (
                                gpu_db_execution::CudaWideKeySource::ResidentBool {
                                    bitmap_byte_offset: resident_device_bool_column_offset(
                                        &snapshot, table, idx,
                                    )?,
                                },
                                8u64,
                            ),
                            // Text members are not in the fixed buffer (they ride the text descriptor).
                            SqlType::Text => continue,
                            _ => (
                                gpu_db_execution::CudaWideKeySource::ResidentI32 {
                                    byte_offset: resident_device_int4_column_offset(
                                        &snapshot, table, idx,
                                    )?,
                                },
                                8u64,
                            ),
                        };
                        descriptors.push(gpu_db_execution::CudaWideKeyDescriptor {
                            source,
                            destination_byte_offset: dst_off,
                        });
                        if composite_key_has_null {
                            validity_descs.push(
                                match resident_device_null_column_offset(&snapshot, table, idx)? {
                                    Some(byte_offset) => {
                                        gpu_db_execution::CudaWideKeyValidity::Bitmap {
                                            byte_offset,
                                        }
                                    }
                                    None => gpu_db_execution::CudaWideKeyValidity::NonNullable,
                                },
                            );
                        }
                        dst_off += w;
                    }
                    let buf = device_memory
                        .build_wide_key_device(&descriptors, widekey_w, row_count, &validity_descs)
                        .map_err(|e| {
                            ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                        })?;
                    _derived_key_buf = Some(buf);
                }
            } else {
                _derived_key_buf = None;
            }
            // General composite TEXT members: each text member's (offsets_off, bytes_off, bytes_len) ->
            // a small device descriptor the wide-key claim bounds, hashes, and verifies (row vs rep),
            // in DECLARED-relative order. Built once + held alive across the pass. No text members
            // (all-fixed wide key) -> n_text = 0, byte-identical to the prior wide-key path.
            let _widekey_text_desc = match &widekey_cols {
                Some(members) if row_count > 0 => {
                    let mut sources = Vec::new();
                    for &(idx, ty) in members {
                        if matches!(ty, SqlType::Text) {
                            let layout = resident_device_text_column_layout(&snapshot, table, idx)?;
                            sources.push(gpu_db_execution::CudaGroupTextSource {
                                offsets_byte_offset: layout.offsets_byte_offset,
                                bytes_byte_offset: layout.bytes_byte_offset,
                                bytes_len: layout.bytes_len,
                                row_count,
                            });
                        }
                    }
                    if sources.is_empty() {
                        None
                    } else {
                        Some(
                            device_memory
                                .upload_group_text_descriptors(&sources)
                                .map_err(map_err)?,
                        )
                    }
                }
                _ => None,
            };
            // One grouping PASS per distinct value column. The kernel yields count+sum+min+max for one
            // value column; each aggregate projects from its column's pass. Every pass groups the SAME
            // key column over the SAME rows, so the i-th group of every pass is the same key (single
            // level is forced when there are >=2 passes so the passes share one compaction order) ->
            // the result merges the passes by group index.
            struct Pass {
                value_idx: usize,
                groups: Vec<gpu_db_execution::GroupByI32Row>,
                value_ty: SqlType,
                value_scale: u8,
                value_is_int8: bool,
                value_is_numeric: bool,
                value_is_uuid: bool,
                value_is_text: bool,
                // A COUNT(DISTINCT v) pass (sort -> mark -> SUM), where `groups[i].sum` is the per-group
                // distinct count. Distinguished from a DIRECT pass on the same `value_idx` so the result
                // builder reads the right one (a column can have both SUM(v) and COUNT(DISTINCT v)).
                is_count_distinct: bool,
            }
            // The result group-key column for an expression GROUP BY is the DERIVED value (no source
            // column); the binding placeholdered it as column 0, so set its name + type to the
            // expression's int4/int8 result so the result schema is correct.
            if is_expr_key {
                if let Some(col) = bound.selected_columns.first_mut() {
                    "?column?".clone_into(&mut col.name);
                    col.ty = key_ty;
                }
            }
            if let Some((_, _, c1, _)) = composite_cols {
                // The binding produced [a, agg...] (GroupedAggregates carries one group_column); the
                // composite result row is [a, b, agg...]. Insert b's full column metadata (from the
                // table -- correct name/type/oid) after a, then renumber the result attnums 1..N.
                let b_col = table.columns[c1].clone();
                bound.selected_columns.insert(1, b_col);
                for (i, col) in bound.selected_columns.iter_mut().enumerate() {
                    col.attnum = (i + 1) as i16;
                }
            } else if let Some(members) = &widekey_cols {
                // Wide-key: the binding produced [member0, agg...]; insert members[1..]'s column metadata
                // after member0 (the result row is [member0..N-1, agg...]), then renumber the attnums.
                for (i, &(idx, _)) in members.iter().enumerate().skip(1) {
                    bound.selected_columns.insert(i, table.columns[idx].clone());
                }
                for (i, col) in bound.selected_columns.iter_mut().enumerate() {
                    col.attnum = (i + 1) as i16;
                }
            }
            // An expression key is read via key_base_override, which only the single-level kernel honors.
            // M3 (doc 21): a nullable VALUE column also forces single-level — only that kernel has the
            // NULL-value skip. (A column with no NULLs has no bitmap, so this never triggers for it.)
            let any_value_nullable = value_indices.iter().try_fold(false, |acc, &vidx| {
                Ok::<bool, ExecuteError>(
                    acc || resident_device_null_column_offset(&snapshot, table, vidx)?.is_some(),
                )
            })?;
            // M3 (doc 21): the group key's NULL validity offset, for a SINGLE-COLUMN key (int2/4/8/date/
            // timestamp/text/numeric/uuid) — the kernel's NULL-key check (now hoisted before the type
            // dispatch) routes a NULL key of any of these to the reserved slot, forming its own group.
            // `None` (no bitmap / a composite/expression key) leaves grouping unchanged. Forces
            // single-level (only that kernel honors the route).
            let pass_key_null_off = if key_is_single_nullable_column {
                resident_device_null_column_offset(&snapshot, table, group_idx)?
            } else {
                // An expression key over exactly one nullable operand reuses that operand's validity bitmap
                // (see expr_key_single_null_off): the kernel routes the expression's NULL rows to the NULL
                // reserved slot, forming their own group, rendered SqlValue::Null.
                expr_key_single_null_off
            };
            let force_single = value_indices.len() > 1
                || is_expr_key
                || any_value_nullable
                || pass_key_null_off.is_some();
            // M3 (doc 21): COUNT(DISTINCT v) over a NULLABLE group key is a clean-error follow-up. The
            // COUNT(DISTINCT) sub-passes (composite_group_count_reps + the step-2 GROUP BY /
            // count_distinct_groups) do NOT route the NULL key to the reserved slot, so they merge NULL-key
            // rows into the placeholder group -> fewer groups than the reference (direct) pass, whose null
            // group IS routed -> the by-index pass merge mis-aligns / panics. Reject cleanly rather than
            // panic or mis-answer. (This guard also covers the pre-existing nullable-INT-key case.) A
            // nullable COMPOSITE key has the same hole: composite_group_count_reps builds its step-1 dedup
            // wide key with NO validity (it would merge (NULL,5) with (0,5)) while step-2 re-groups with the
            // validity-bearing wide key -> a misaligned / dropped group. So clean-error it too (the
            // non-DISTINCT nullable-composite aggregates above are fully supported).
            if (pass_key_null_off.is_some() || composite_key_has_null)
                && aggregates
                    .iter()
                    .any(|a| a.kind == GroupedAggKind::CountDistinct)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "GROUP BY a nullable key with COUNT(DISTINCT) is not yet supported on the GPU \
                     (M3 3VL follow-up: the COUNT(DISTINCT) sub-pass does not yet route the NULL key to \
                     its own group)"
                        .to_string(),
                )));
            }
            // M3 (doc 21): COUNT(DISTINCT v) over a NULLABLE VALUE column. PG counts distinct NON-NULL
            // values, but the sort-based reps pass groups by (g, v) WITHOUT value validity -- it would fold
            // NULL v into a (deterministic-or-stale) value and count it as a distinct value (an over-count),
            // and an all-NULL-v group would vanish from the pass (misaligning the by-index merge). The value
            // column is NOT in `value_indices` (it skips COUNT(DISTINCT)) so `any_value_nullable` misses it;
            // check it here. Clean-error rather than silently mis-count (excluding NULL v + keeping the
            // all-NULL-v group at 0 is the follow-up).
            for (aggregate, value_idx) in aggregates.iter().zip(&agg_value_indices) {
                if aggregate.kind == GroupedAggKind::CountDistinct {
                    if let Some(idx) = value_idx {
                        if resident_device_null_column_offset(&snapshot, table, *idx)?.is_some() {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "COUNT(DISTINCT) over a nullable column is not yet supported on the GPU \
                                 (M3 3VL follow-up: excluding NULL values from the distinct count)"
                                    .to_string(),
                            )));
                        }
                    }
                }
            }
            let run_pass = |value_idx_opt: Option<usize>,
                            has_minmax: bool|
             -> Result<Pass, ExecuteError> {
                // A None value column is the COUNT(*)-only pass: group over the key, read .count.
                let value_idx = value_idx_opt.unwrap_or(group_idx);
                let value_ty = table.columns[value_idx].ty;
                // M3 (doc 21) 3VL: a value pass over a NULLABLE column skips NULL values ON-DEVICE so
                // its count/sum/min/max are over only the non-NULL rows (COUNT(*) passes None and
                // counts every row). `None` = no bitmap (no NULLs) ⇒ byte-identical. Only the
                // single-level kernel honors it, so a nullable value forces single-level (below).
                let pass_value_null_off = if value_idx_opt.is_some() {
                    resident_device_null_column_offset(&snapshot, table, value_idx)?
                } else {
                    None
                };
                let value_scale: u8 = match value_ty {
                    SqlType::Numeric { scale, .. } => scale,
                    _ => 0,
                };
                // Classify by the value TYPE (the device read width). MIN/MAX accept every ordered
                // type; SUM/AVG over an unsupported type is rejected at bind. COUNT reads no value.
                let (value_is_int8, value_is_numeric, value_is_uuid, value_is_text) =
                    if value_idx_opt.is_none() {
                        (false, false, false, false)
                    } else {
                        match value_ty {
                            SqlType::Int4 | SqlType::Int2 | SqlType::Date => {
                                (false, false, false, false)
                            }
                            SqlType::Int8 | SqlType::Timestamp => (true, false, false, false),
                            SqlType::Numeric { .. } => (false, true, false, false),
                            SqlType::Uuid => (false, false, true, false),
                            SqlType::Text => (false, false, false, true),
                            // bool value (MIN/MAX): materialized bool->int4, read via
                            // value_base_override on the int4 value path.
                            SqlType::Bool => (false, false, false, false),
                        }
                    };
                let value_is_bool = value_idx_opt.is_some() && matches!(value_ty, SqlType::Bool);
                let value_offset = if value_idx_opt.is_none() || value_is_text || value_is_bool {
                    // bool: value_offset is unused (value_base_override is set); key_offset is a safe
                    // placeholder (avoids resolving an int4 offset on a bitmap bool column).
                    key_offset
                } else if value_is_numeric || value_is_uuid {
                    resident_device_numeric_column_offset(&snapshot, table, value_idx)?
                } else if value_is_int8 {
                    resident_device_int8_column_offset(&snapshot, table, value_idx)?
                } else {
                    resident_device_int4_column_offset(&snapshot, table, value_idx)?
                };
                // MIN/MAX over a bool VALUE: materialize bool->int4 (0/1) into a derived buffer + read
                // it via value_base_override. Held alive across the kernel call (closure-local lease).
                let _derived_value_buf = if value_is_bool && row_count > 0 {
                    let offset = resident_device_bool_column_offset(&snapshot, table, value_idx)?;
                    let buf = device_memory
                        .bool_to_int4_column_device(offset, row_count)
                        .map_err(map_err)?;
                    Some(buf)
                } else {
                    None
                };
                let (value_offsets_off, value_bytes_off, value_bytes_len) = if value_is_text {
                    let layout = resident_device_text_column_layout(&snapshot, table, value_idx)?;
                    (
                        layout.offsets_byte_offset,
                        layout.bytes_byte_offset,
                        layout.bytes_len,
                    )
                } else {
                    (0, 0, 0)
                };
                let use_single_level = has_minmax
                    || value_is_int8
                    || value_is_numeric
                    || value_is_uuid
                    || value_is_text
                    || value_is_bool
                    || key_is_int8
                    || key_is_i128
                    || key_is_text
                    || key_is_bool
                    || composite_is_widekey
                    || force_single;
                let groups = if use_single_level {
                    let key = if composite_is_widekey {
                        gpu_db_execution::CudaGroupKeySource::Composite {
                            fixed: _derived_key_buf.as_ref().map(|buf| {
                                gpu_db_execution::CudaGroupWideSource {
                                    buffer: buf.group_view(),
                                    row_width: widekey_w,
                                    row_count,
                                }
                            }),
                            text: _widekey_text_desc.as_ref().map(|buf| buf.descriptors()),
                            row_count,
                        }
                    } else if key_is_text {
                        gpu_db_execution::CudaGroupKeySource::Text {
                            text: gpu_db_execution::CudaGroupTextSource {
                                offsets_byte_offset: key_offsets_off,
                                bytes_byte_offset: key_bytes_off,
                                bytes_len: key_bytes_len,
                                row_count,
                            },
                            fixed_component: if composite_is_text {
                                Some(gpu_db_execution::CudaGroupFixedSource::Derived {
                                    buffer: _derived_key_buf
                                        .as_ref()
                                        .expect("fixed text component")
                                        .group_view(),
                                    width: 8,
                                    row_count,
                                })
                            } else {
                                None
                            },
                        }
                    } else {
                        let width = if key_is_i128 {
                            16
                        } else if key_is_int8 {
                            8
                        } else {
                            4
                        };
                        let source = if let Some(buf) = &_derived_key_buf {
                            gpu_db_execution::CudaGroupFixedSource::Derived {
                                buffer: buf.group_view(),
                                width,
                                row_count,
                            }
                        } else {
                            gpu_db_execution::CudaGroupFixedSource::Resident {
                                byte_offset: key_offset,
                                width,
                                row_count,
                            }
                        };
                        gpu_db_execution::CudaGroupKeySource::Fixed(source)
                    };
                    let value = if value_idx_opt.is_none() {
                        gpu_db_execution::CudaGroupValueSource::Unused { row_count }
                    } else if value_is_text {
                        gpu_db_execution::CudaGroupValueSource::Text(
                            gpu_db_execution::CudaGroupTextSource {
                                offsets_byte_offset: value_offsets_off,
                                bytes_byte_offset: value_bytes_off,
                                bytes_len: value_bytes_len,
                                row_count,
                            },
                        )
                    } else {
                        let width = if value_is_numeric || value_is_uuid {
                            16
                        } else if value_is_int8 {
                            8
                        } else {
                            4
                        };
                        let source = if let Some(buf) = &_derived_value_buf {
                            gpu_db_execution::CudaGroupFixedSource::Derived {
                                buffer: buf.group_view(),
                                width,
                                row_count,
                            }
                        } else {
                            gpu_db_execution::CudaGroupFixedSource::Resident {
                                byte_offset: value_offset,
                                width,
                                row_count,
                            }
                        };
                        if value_is_numeric {
                            gpu_db_execution::CudaGroupValueSource::Numeric(source)
                        } else if value_is_uuid {
                            gpu_db_execution::CudaGroupValueSource::Uuid(source)
                        } else {
                            gpu_db_execution::CudaGroupValueSource::Fixed(source)
                        }
                    };
                    device_memory.group_by_i32_count_sum_minmax_from_payload(
                        gpu_db_execution::CudaGroupByInput {
                            key,
                            value,
                            key_validity_bitmap_offset: pass_key_null_off,
                            value_validity_bitmap_offset: pass_value_null_off,
                        },
                        &indices,
                        // The query-wide pruning mask: this pass computes a superset of what its column
                        // needs; the executor reads only the masked-in field(s) it requested.
                        if value_idx_opt.is_none() {
                            gpu_db_execution::grouped_agg_mask::COUNT
                        } else {
                            agg_mask
                        },
                    )
                } else {
                    device_memory.group_by_i32_count_sum_from_payload(
                        gpu_db_execution::CudaGroupByInput::resident_i32(
                            key_offset,
                            value_offset,
                            row_count,
                        ),
                        &indices,
                        agg_mask,
                    )
                }
                .map_err(map_err)?;
                Ok(Pass {
                    value_idx,
                    groups,
                    value_ty,
                    value_scale,
                    value_is_int8,
                    value_is_numeric,
                    value_is_uuid,
                    value_is_text,
                    is_count_distinct: false,
                })
            };
            let mut passes: Vec<Pass> = if value_indices.is_empty() {
                // COUNT(*) only: a single pass over the key column.
                vec![run_pass(None, false)?]
            } else {
                let mut passes = Vec::with_capacity(value_indices.len() + 1);
                // M3 (doc 21): a nullable value pass's `count` is the NON-NULL count, but COUNT(*) needs
                // the TOTAL (the merge reads COUNT(*) from passes[0].count). So when a value is nullable
                // AND the query has a COUNT(*), prepend a dedicated total-count pass (value_null_off=None,
                // every row counted) as the reference. All passes claim a slot for every row, so they
                // share one compaction order; the value passes only skip NULL from their count/sum/min/max.
                // (Without a COUNT(*), passes[0] = the first value pass already carries the full key set.)
                let has_count_star = aggregates.iter().any(|a| a.kind == GroupedAggKind::Count);
                if any_value_nullable && has_count_star {
                    passes.push(run_pass(None, false)?);
                }
                for &value_idx in &value_indices {
                    let has_minmax = aggregates.iter().zip(&agg_value_indices).any(|(a, vi)| {
                        *vi == Some(value_idx)
                            && matches!(a.kind, GroupedAggKind::Min | GroupedAggKind::Max)
                    });
                    passes.push(run_pass(Some(value_idx), has_minmax)?);
                }
                passes
            };
            // COUNT(DISTINCT v) passes: a separate SORT-based pass per such aggregate (the direct hash
            // pass cannot dedup). Materialize (group_key, v) as the i64 tuple matrix over the surviving
            // rows, GPU-sort by (g, v), mark first-seen (g, v) tuples, then SUM the new-distinct flags
            // grouped by g via key/value_base_override -> the per-group distinct count. Scoped to a
            // plain-column int group key (expression/bool/composite/text/numeric/uuid keys are follow-
            // ups); the value column is int2/4/8/date/timestamp (validated at bind). A pure map + the
            // GPU sort + the AUDITED int4 GROUP BY kernel (fully-drained launches) -> no new hazard.
            if aggregates
                .iter()
                .any(|a| a.kind == GroupedAggKind::CountDistinct)
            {
                let idx_u64: Vec<u64> = indices.iter().map(|&i| u64::from(i)).collect();
                let n = idx_u64.len();
                if is_composite_key
                    || composite_is_widekey
                    || key_is_text
                    || key_is_i128
                    || key_is_bool
                    || is_expr_key
                {
                    // Non-plain-integer/composite group-key route: reduce COUNT(DISTINCT v) per g to
                    // counting DISTINCT (g, v)
                    // pairs per g. (1) GROUP BY (g..., v) -> one representative row per distinct (g, v)
                    // [the general composite path]; (2) GROUP BY g over those reps, COUNT(*) -> the
                    // distinct-v count per g [reusing the MAIN g config so the groups carry the real g
                    // key and align with the reference pass via materialize_key]. All on the GPU.
                    // An EXPRESSION group key has NO group COLUMNS -- it rides a DERIVED buffer
                    // (key_base_override), fed to step 1 as the wide key's derived member (build kind
                    // 4/5) and reused as the step-2 key config; column group keys pass derived = None.
                    let g_members: Vec<(usize, SqlType)> = if is_expr_key {
                        Vec::new()
                    } else if let Some(m) = &widekey_cols {
                        m.clone()
                    } else if let Some((c0, t0, c1, t1)) = composite_cols {
                        vec![(c0, t0), (c1, t1)]
                    } else {
                        vec![(group_idx, key_ty)]
                    };
                    for (aggregate, value_idx) in aggregates.iter().zip(&agg_value_indices) {
                        if aggregate.kind != GroupedAggKind::CountDistinct {
                            continue;
                        }
                        let value_idx = value_idx.expect("COUNT(DISTINCT) has a value column");
                        let groups = if n == 0 {
                            Vec::new()
                        } else {
                            // (1) the distinct (g, v) representative rows.
                            let mut gv_members = g_members.clone();
                            gv_members.push((value_idx, table.columns[value_idx].ty));
                            let reps = composite_group_count_reps(
                                &snapshot,
                                table,
                                &device_memory,
                                &gv_members,
                                &indices,
                                row_count,
                                if is_expr_key {
                                    Some((
                                        _derived_key_buf
                                            .as_ref()
                                            .expect("expression group key buffer")
                                            .group_view(),
                                        key_expr_is_int8,
                                    ))
                                } else {
                                    None
                                },
                            )?;
                            // (2) GROUP BY g over the reps (reusing the main g config) COUNT(*).
                            let mut g2 = if reps.is_empty() {
                                Vec::new()
                            } else {
                                device_memory
                                    .group_by_i32_count_sum_minmax_from_payload(
                                        gpu_db_execution::CudaGroupByInput {
                                            key: if composite_is_widekey {
                                                gpu_db_execution::CudaGroupKeySource::Composite {
                                                    fixed: _derived_key_buf.as_ref().map(|buf| gpu_db_execution::CudaGroupWideSource {
                                                        buffer: buf.group_view(), row_width: widekey_w, row_count,
                                                    }),
                                                    text: _widekey_text_desc.as_ref().map(|buf| buf.descriptors()),
                                                    row_count,
                                                }
                                            } else if key_is_text {
                                                gpu_db_execution::CudaGroupKeySource::Text {
                                                    text: gpu_db_execution::CudaGroupTextSource {
                                                        offsets_byte_offset: key_offsets_off,
                                                        bytes_byte_offset: key_bytes_off,
                                                        bytes_len: key_bytes_len,
                                                        row_count,
                                                    },
                                                    fixed_component: if composite_is_text {
                                                        Some(gpu_db_execution::CudaGroupFixedSource::Derived {
                                                            buffer: _derived_key_buf.as_ref().expect("fixed text component").group_view(),
                                                            width: 8,
                                                            row_count,
                                                        })
                                                    } else { None },
                                                }
                                            } else {
                                                let width = if key_is_i128 { 16 } else if key_is_int8 { 8 } else { 4 };
                                                gpu_db_execution::CudaGroupKeySource::Fixed(
                                                    if let Some(buf) = &_derived_key_buf {
                                                        gpu_db_execution::CudaGroupFixedSource::Derived {
                                                            buffer: buf.group_view(), width, row_count,
                                                        }
                                                    } else {
                                                        gpu_db_execution::CudaGroupFixedSource::Resident {
                                                            byte_offset: key_offset, width, row_count,
                                                        }
                                                    },
                                                )
                                            },
                                            value: gpu_db_execution::CudaGroupValueSource::Unused { row_count },
                                            key_validity_bitmap_offset: None,
                                            value_validity_bitmap_offset: None,
                                        },
                                        &reps,
                                        // Step-2 GROUP BY g over reps, COUNT(*): the result builder reads
                                        // `.count` (-> `.sum`). ALL is correct + behavior-preserving.
                                        gpu_db_execution::grouped_agg_mask::COUNT,
                                    )
                                    .map_err(map_err)?
                            };
                            // The CountDistinct pass carries the per-group distinct count in `.sum`
                            // (the result builder reads groups[i].sum for such a pass).
                            for grp in &mut g2 {
                                grp.sum = grp.count as i64;
                            }
                            g2
                        };
                        passes.push(Pass {
                            value_idx,
                            groups,
                            value_ty: SqlType::Int8,
                            value_scale: 0,
                            value_is_int8: false,
                            value_is_numeric: false,
                            value_is_uuid: false,
                            value_is_text: false,
                            is_count_distinct: true,
                        });
                    }
                } else {
                    // Plain int group key: the direct sort -> mark -> per-group SUM pipeline.
                    let materialize_i64_col =
                        |col_idx: usize, idx_u64: &[u64]| -> Result<Vec<i64>, ExecuteError> {
                            match table.columns[col_idx].ty {
                                SqlType::Int4 | SqlType::Int2 | SqlType::Date => Ok(device_memory
                                    .project_i32_rows_from_payload(
                                        resident_device_int4_column_offset(
                                            &snapshot, table, col_idx,
                                        )?,
                                        idx_u64,
                                    )
                                    .map_err(map_err)?
                                    .into_iter()
                                    .map(i64::from)
                                    .collect()),
                                SqlType::Int8 | SqlType::Timestamp => device_memory
                                    .project_i64_rows_from_payload(
                                        resident_device_int8_column_offset(
                                            &snapshot, table, col_idx,
                                        )?,
                                        idx_u64,
                                    )
                                    .map_err(map_err),
                                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                    "COUNT(DISTINCT) group key / value must be an i64-representable \
                                     int column"
                                        .to_string(),
                                ))),
                            }
                        };
                    // The group key column values (sorted-tuple key 0), materialized once per pass.
                    let g_vals = if n > 0 {
                        Some(materialize_i64_col(group_idx, &idx_u64)?)
                    } else {
                        None
                    };
                    for (aggregate, value_idx) in aggregates.iter().zip(&agg_value_indices) {
                        if aggregate.kind != GroupedAggKind::CountDistinct {
                            continue;
                        }
                        let value_idx = value_idx.expect("COUNT(DISTINCT) has a value column");
                        // GPU sort -> mark -> per-group SUM over (group_key, value); see
                        // `count_distinct_groups`. The group key is the real plain-int group column.
                        let groups = if n == 0 {
                            Vec::new()
                        } else {
                            let g_vals = g_vals.as_ref().expect("g_vals materialized for n > 0");
                            count_distinct_groups(value_idx, g_vals, &idx_u64)?
                        };
                        passes.push(Pass {
                            value_idx,
                            groups,
                            value_ty: SqlType::Int8,
                            value_scale: 0,
                            value_is_int8: false,
                            value_is_numeric: false,
                            value_is_uuid: false,
                            value_is_text: false,
                            is_count_distinct: true,
                        });
                    }
                }
            }
            // A TEXT key/value's string -- and a wide-key composite's member values -- live host-side: the
            // kernel stored each group's representative ABSOLUTE row index; read it from the same
            // residency_entry generation as the GPU result.
            // GROUP BY result key/value/member materialization is now fully ON-DEVICE (key_text_map S2.2a,
            // text_value_minmax S2.2b-i, member_cell S2.2b-ii) -- no host_rows clone for the grouped path.
            // S2.2a: a PLAIN text group KEY, materialized ON-DEVICE. One rep_idx -> key-string map gathered
            // from the resident payload (project_text_rows_from_payload) over EVERY pass's group rep indices
            // (materialize_key runs per-pass in #30), instead of reading host_rows. NULL-key groups are
            // excluded (they render SqlValue::Null directly, never via a rep row).
            let key_text_map: Option<std::collections::HashMap<usize, String>> =
                if key_is_text && !is_composite_key && !composite_is_widekey {
                    let mut reps: Vec<u64> = Vec::new();
                    for pass in &passes {
                        for g in &pass.groups {
                            if !g.key_is_null {
                                reps.push(g.key_i128 as u64);
                            }
                        }
                    }
                    reps.sort_unstable();
                    reps.dedup();
                    let layout = resident_device_text_column_layout(&snapshot, table, group_idx)?;
                    let texts = device_memory
                        .project_text_rows_from_payload(
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            layout.bytes_len,
                            row_count,
                            &reps,
                        )
                        .map_err(map_err)?;
                    Some(reps.iter().map(|&r| r as usize).zip(texts).collect())
                } else {
                    None
                };
            // S2.2b-ii: composite (fixed,text) / wide-key MEMBER values, materialized ON-DEVICE into a
            // sparse (rep_row, col) -> SqlValue map -- replacing the text_host_rows host_rows clone reads.
            // Members may be ANY type, so gather each column by type (mirrors the S1 projection) with NULL
            // validity (a nullable member is SqlValue::Null), at every group's key rep index across ALL
            // passes (covers materialize_key in #30 AND the result builder, incl. NULL-key groups whose
            // members the result builder reads from the rep row).
            let materialize_col_at =
                |col: usize, reps: &[u64]| -> Result<Vec<SqlValue>, ExecuteError> {
                    if reps.is_empty() {
                        return Ok(Vec::new());
                    }
                    let validity: Option<Vec<bool>> =
                        match resident_device_null_column_offset(&snapshot, table, col)? {
                            Some(off) => Some(
                                device_memory
                                    .project_bool_rows_from_payload(off, reps)
                                    .map_err(map_err)?,
                            ),
                            None => None,
                        };
                    let base: Vec<SqlValue> = match table.columns[col].ty {
                        SqlType::Int8 => device_memory
                            .project_i64_rows_from_payload(
                                resident_device_int8_column_offset(&snapshot, table, col)?,
                                reps,
                            )
                            .map_err(map_err)?
                            .into_iter()
                            .map(SqlValue::Int8)
                            .collect(),
                        SqlType::Timestamp => device_memory
                            .project_i64_rows_from_payload(
                                resident_device_int8_column_offset(&snapshot, table, col)?,
                                reps,
                            )
                            .map_err(map_err)?
                            .into_iter()
                            .map(SqlValue::Timestamp)
                            .collect(),
                        SqlType::Numeric { scale, .. } => device_memory
                            .project_i128_rows_from_payload(
                                resident_device_numeric_column_offset(&snapshot, table, col)?,
                                reps,
                            )
                            .map_err(map_err)?
                            .into_iter()
                            .map(|v| SqlValue::Numeric(Decimal128::new(v, scale)))
                            .collect(),
                        SqlType::Uuid => device_memory
                            .project_i128_rows_from_payload(
                                resident_device_numeric_column_offset(&snapshot, table, col)?,
                                reps,
                            )
                            .map_err(map_err)?
                            .into_iter()
                            .map(|v| SqlValue::Uuid(v.to_le_bytes()))
                            .collect(),
                        SqlType::Date => device_memory
                            .project_i32_rows_from_payload(
                                resident_device_int4_column_offset(&snapshot, table, col)?,
                                reps,
                            )
                            .map_err(map_err)?
                            .into_iter()
                            .map(SqlValue::Date)
                            .collect(),
                        SqlType::Int2 => device_memory
                            .project_i32_rows_from_payload(
                                resident_device_int4_column_offset(&snapshot, table, col)?,
                                reps,
                            )
                            .map_err(map_err)?
                            .into_iter()
                            .map(|v| SqlValue::Int2(v as i16))
                            .collect(),
                        SqlType::Bool => device_memory
                            .project_bool_rows_from_payload(
                                resident_device_bool_column_offset(&snapshot, table, col)?,
                                reps,
                            )
                            .map_err(map_err)?
                            .into_iter()
                            .map(SqlValue::Bool)
                            .collect(),
                        SqlType::Text => {
                            let layout = resident_device_text_column_layout(&snapshot, table, col)?;
                            device_memory
                                .project_text_rows_from_payload(
                                    layout.offsets_byte_offset,
                                    layout.bytes_byte_offset,
                                    layout.bytes_len,
                                    row_count,
                                    reps,
                                )
                                .map_err(map_err)?
                                .into_iter()
                                .map(SqlValue::Text)
                                .collect()
                        }
                        SqlType::Int4 => device_memory
                            .project_i32_rows_from_payload(
                                resident_device_int4_column_offset(&snapshot, table, col)?,
                                reps,
                            )
                            .map_err(map_err)?
                            .into_iter()
                            .map(SqlValue::Int4)
                            .collect(),
                    };
                    Ok(base
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| match &validity {
                            Some(val) if !val[i] => SqlValue::Null,
                            _ => v,
                        })
                        .collect())
                };
            let member_cell: std::collections::HashMap<(usize, usize), SqlValue> = {
                let member_cols: Vec<usize> = if let Some(m) = &widekey_cols {
                    m.iter().map(|&(c, _)| c).collect()
                } else if composite_is_text {
                    let (c0, _, c1, _) =
                        composite_cols.expect("composite_is_text implies composite_cols");
                    vec![c0, c1]
                } else {
                    Vec::new()
                };
                let mut cells = std::collections::HashMap::new();
                if !member_cols.is_empty() {
                    let mut reps: Vec<u64> = passes
                        .iter()
                        .flat_map(|p| p.groups.iter().map(|g| g.key_i128 as u64))
                        .collect();
                    reps.sort_unstable();
                    reps.dedup();
                    for &col in &member_cols {
                        let vals = materialize_col_at(col, &reps)?;
                        for (r, v) in reps.iter().zip(vals) {
                            cells.insert((*r as usize, col), v);
                        }
                    }
                }
                cells
            };
            // The kernel's hash-slot / compaction order is RACE-dependent: the cas.b64 linear-probe
            // resolves bucket ownership differently per launch, so two passes do NOT share a group
            // order (proven: an int8 i64::MIN-key query misaligned only on the 6th launch). Re-sort
            // every pass by the MATERIALIZED group key so pass[k][i] is the same group across all
            // passes (and the merged output ends up key-ordered). A TEXT key must sort by the
            // materialized string -- there key_i128 is a per-pass representative row index, not the key.
            let materialize_key = |gk: &gpu_db_execution::GroupByI32Row| -> SqlValue {
                // M3 (doc 21): the NULL-KEY group (every NULL key grouped together) renders its key as SQL
                // NULL — for BOTH the pass-alignment sort (NULLs sort consistently) and the result row.
                if gk.key_is_null {
                    return SqlValue::Null;
                }
                if composite_is_widekey {
                    // Wide-key: the b128 slot's lo = the representative row index. Return the FIRST
                    // member's value for the (single-pass) alignment sort; the result rows re-sort by the
                    // FULL member tuple, so this lone value need only be consistent + non-panicking.
                    let rep_idx = gk.key_i128 as u64 as usize;
                    let m0 = widekey_cols.as_ref().expect("composite_is_widekey")[0].0;
                    member_cell[&(rep_idx, m0)].clone()
                } else if is_composite_key {
                    // The result row UNPACKS the key into the two columns; here we only need a
                    // CONSISTENT representation for pass-alignment + ordering. i64 pack -> Int8(key);
                    // i128 pack -> Numeric wrapping the packed i128; (fixed, text) -> the rep row's text
                    // value (composite-text is single-aggregate, so a one-pass sort suffices and the
                    // result rows re-sort by the full tuple). (Composite sets key_is_i128/key_is_text for
                    // the wider/text cases, so this MUST precede those arms below.)
                    if composite_is_text {
                        let rep_idx = gk.key_i128 as u64 as usize;
                        let (cc0, ct0, cc1, _) = composite_cols.unwrap();
                        let text_col = if matches!(ct0, SqlType::Text) {
                            cc0
                        } else {
                            cc1
                        };
                        member_cell[&(rep_idx, text_col)].clone()
                    } else if composite_is_i128 {
                        SqlValue::Numeric(Decimal128::new(gk.key_i128, 0))
                    } else {
                        SqlValue::Int8(gk.key)
                    }
                } else if key_is_text {
                    // S2.2a: plain text key, looked up from the ON-DEVICE-gathered map (no host_rows read).
                    let rep_idx = gk.key_i128 as u64 as usize;
                    SqlValue::Text(
                        key_text_map
                            .as_ref()
                            .expect("key_text_map is Some for a plain text key")[&rep_idx]
                            .clone(),
                    )
                } else if key_is_i128 {
                    match key_ty {
                        SqlType::Numeric { .. } => {
                            SqlValue::Numeric(Decimal128::new(gk.key_i128, key_scale))
                        }
                        SqlType::Uuid => SqlValue::Uuid(gk.key_i128.to_le_bytes()),
                        _ => unreachable!("key_is_i128 is only numeric/uuid"),
                    }
                } else {
                    narrow_ordered_value(key_ty, gk.key, 0, 0)
                }
            };
            // The FULL group-key tuple for a group (all key columns/members, in declared order) -- the
            // device-materialized values (key_text_map / member_cell / the packed-int unpack / the typed
            // struct). Used to align passes ON-DEVICE below.
            let full_key = |gk: &gpu_db_execution::GroupByI32Row| -> Vec<SqlValue> {
                if let Some(members) = &widekey_cols {
                    let rep_idx = gk.key_i128 as u64 as usize;
                    members
                        .iter()
                        .map(|&(idx, _)| member_cell[&(rep_idx, idx)].clone())
                        .collect()
                } else if let Some((c0, t0, c1, t1)) = composite_cols {
                    if composite_is_text {
                        let rep_idx = gk.key_i128 as u64 as usize;
                        vec![
                            member_cell[&(rep_idx, c0)].clone(),
                            member_cell[&(rep_idx, c1)].clone(),
                        ]
                    } else {
                        let (col0, col1) = if composite_is_i128 {
                            let k = gk.key_i128;
                            ((k >> 64) as i64, k as u64 as i64)
                        } else {
                            let col0 = ((((gk.key as u64) >> 32) as u32) as i32) as i64;
                            let col1 = (((gk.key as u64) as u32) as i32) as i64;
                            (col0, col1)
                        };
                        vec![
                            narrow_ordered_value(t0, col0, 0, 0),
                            narrow_ordered_value(t1, col1, 0, 0),
                        ]
                    }
                } else {
                    vec![materialize_key(gk)]
                }
            };
            // Pass-alignment (was charter debt #30, NOW ON-DEVICE -- S2.3): each aggregate's hash-agg pass
            // compacts groups in a RACE-dependent order, so MULTIPLE passes don't share a group order; the
            // merge below indexes pass[k][i] expecting the same group. Give every pass ONE shared order by
            // sorting each by the FULL group key ON THE GPU (gpu_sort_permutation). The full key is UNIQUE
            // per group, so this is a TOTAL order -> all passes align by index with no host sort and no
            // sort-stability dependence. #30 needs only CONSISTENT alignment (the FINAL result order is
            // the gpu_sort_permutation window below), so any one deterministic order works; ASC / NULL-first
            // is fine.
            // A single pass needs no alignment (its groups are internally consistent), so skip it there.
            if passes.len() > 1 {
                let key_types: Vec<SqlType> = if let Some(members) = &widekey_cols {
                    members.iter().map(|&(_, t)| t).collect()
                } else if let Some((_, t0, _, t1)) = composite_cols {
                    vec![t0, t1]
                } else {
                    vec![key_ty]
                };
                let key_order: Vec<(usize, bool)> =
                    (0..key_types.len()).map(|c| (c, false)).collect();
                let key_nulls: Vec<Option<bool>> = vec![Some(true); key_types.len()];
                for pass in passes.iter_mut() {
                    let key_rows: Vec<Vec<SqlValue>> =
                        pass.groups.iter().map(&full_key).collect();
                    let perm = gpu_sort_permutation(
                        &key_rows,
                        &key_order,
                        &key_nulls,
                        &key_types,
                        &device_memory,
                    )?;
                    pass.groups = perm
                        .iter()
                        .map(|&p| pass.groups[p as usize])
                        .collect();
                }
            }
            // S2.2b-i: MIN/MAX over a TEXT value, materialized ON-DEVICE. For each text value pass, gather
            // the result string at every group's g.min / g.max ROW INDEX (project_text_rows_from_payload),
            // not from host_rows. A count==0 (all-NULL) group's min/max is unused (its result is NULL), so
            // it gets a 0 placeholder rep. Passes are #30-aligned here, so [i] matches reference.groups[i].
            let text_value_minmax: std::collections::HashMap<usize, (Vec<String>, Vec<String>)> = {
                let mut m = std::collections::HashMap::new();
                for pass in &passes {
                    if pass.value_is_text && !pass.is_count_distinct {
                        let layout =
                            resident_device_text_column_layout(&snapshot, table, pass.value_idx)?;
                        let gather = |reps: &[u64]| -> Result<Vec<String>, ExecuteError> {
                            device_memory
                                .project_text_rows_from_payload(
                                    layout.offsets_byte_offset,
                                    layout.bytes_byte_offset,
                                    layout.bytes_len,
                                    row_count,
                                    reps,
                                )
                                .map_err(map_err)
                        };
                        let min_reps: Vec<u64> = pass
                            .groups
                            .iter()
                            .map(|g| if g.count == 0 { 0 } else { g.min as u64 })
                            .collect();
                        let max_reps: Vec<u64> = pass
                            .groups
                            .iter()
                            .map(|g| if g.count == 0 { 0 } else { g.max as u64 })
                            .collect();
                        let min_txt = gather(&min_reps)?;
                        let max_txt = gather(&max_reps)?;
                        m.insert(pass.value_idx, (min_txt, max_txt));
                    }
                }
                m
            };
            // Merge the (now key-aligned) passes by group index into N+1 columns [key, agg_1, .., agg_N].
            // COUNT reads the group's row count from the reference pass; each other aggregate projects
            // from its value column's pass with the per-type narrowing (SUM/AVG widen, MIN/MAX narrow).
            let reference = &passes[0];
            let mut rows: Vec<Vec<SqlValue>> = Vec::with_capacity(reference.groups.len());
            for i in 0..reference.groups.len() {
                let gk = &reference.groups[i];
                let n_group_cols = if is_composite_key {
                    2
                } else if let Some(m) = &widekey_cols {
                    m.len()
                } else {
                    1
                };
                let mut row: Vec<SqlValue> = Vec::with_capacity(aggregates.len() + n_group_cols);
                if let Some(members) = &widekey_cols {
                    // Wide-key: the b128 slot's lo = the representative row index -- read EACH member from
                    // the rep row, materialized ON-DEVICE into member_cell, in DECLARED order.
                    let rep_idx = gk.key_i128 as u64 as usize;
                    for &(idx, _) in members {
                        row.push(member_cell[&(rep_idx, idx)].clone());
                    }
                } else if let Some((c0, t0, c1, t1)) = composite_cols {
                    if composite_is_text {
                        // (fixed, text): the b128 slot holds the rep row index -- read BOTH members from
                        // the rep row, materialized ON-DEVICE into member_cell, in DECLARED order.
                        let rep_idx = gk.key_i128 as u64 as usize;
                        row.push(member_cell[&(rep_idx, c0)].clone());
                        row.push(member_cell[&(rep_idx, c1)].clone());
                    } else {
                        // Two-fixed composite: UNPACK the packed key into the two columns, narrowed to
                        // their real SqlType. i64 pack -> col0 = high 32 bits, col1 = low 32 (`as u32 as
                        // i32` round-trips negatives). i128 pack -> col0 = high 64, col1 = low 64.
                        let (col0, col1) = if composite_is_i128 {
                            let k = gk.key_i128;
                            ((k >> 64) as i64, k as u64 as i64)
                        } else {
                            let col0 = ((((gk.key as u64) >> 32) as u32) as i32) as i64;
                            let col1 = (((gk.key as u64) as u32) as i32) as i64;
                            (col0, col1)
                        };
                        row.push(narrow_ordered_value(t0, col0, 0, 0));
                        row.push(narrow_ordered_value(t1, col1, 0, 0));
                    }
                } else {
                    // The GROUP BY key narrows back to its own type (same closure used for the sort above).
                    row.push(materialize_key(gk));
                }
                for (aggregate, value_idx_opt) in aggregates.iter().zip(&agg_value_indices) {
                    let value = match aggregate.kind {
                        GroupedAggKind::Count => SqlValue::Int8(reference.groups[i].count as i64),
                        GroupedAggKind::CountDistinct => {
                            let value_idx =
                                value_idx_opt.expect("COUNT(DISTINCT) has a value column");
                            // The CountDistinct (sort -> mark -> SUM) pass for THIS column; its
                            // groups[i].sum = the per-group distinct count (a small non-negative i64).
                            let pass = passes
                                .iter()
                                .find(|p| p.value_idx == value_idx && p.is_count_distinct)
                                .expect("a COUNT(DISTINCT) pass exists for its value column");
                            SqlValue::Int8(pass.groups[i].sum)
                        }
                        _ => {
                            let value_idx =
                                value_idx_opt.expect("non-count aggregate has a value column");
                            // A DIRECT pass (not the CountDistinct pass) -- a column may carry both.
                            let pass = passes
                                .iter()
                                .find(|p| p.value_idx == value_idx && !p.is_count_distinct)
                                .expect("a pass exists for every value column");
                            let g = &pass.groups[i];
                            // M3 (doc 21): a value pass's `count` is the NON-NULL count (the kernel skips
                            // NULL values), so count == 0 means EVERY value in this group is NULL -> the
                            // aggregate over zero non-NULL rows is SQL NULL (PG: SUM/AVG/MIN/MAX of no
                            // rows). A non-nullable value never yields count 0 for an existing group, so
                            // this is a no-op for the non-NULL path. COUNT(*) reads the total-count pass.
                            if g.count == 0 {
                                SqlValue::Null
                            } else {
                                // For int8/numeric the SUM is the i128 (sum_hi:sum); else sign-extend.
                                let sum_i128 = if pass.value_is_int8 || pass.value_is_numeric {
                                    (i128::from(g.sum_hi) << 64) | i128::from(g.sum as u64)
                                } else {
                                    i128::from(g.sum)
                                };
                                match aggregate.kind {
                                    GroupedAggKind::Sum if pass.value_is_numeric => {
                                        SqlValue::Numeric(Decimal128::new(
                                            sum_i128,
                                            pass.value_scale,
                                        ))
                                    }
                                    GroupedAggKind::Sum if pass.value_is_int8 => {
                                        SqlValue::Numeric(Decimal128::new(sum_i128, 0))
                                    }
                                    GroupedAggKind::Sum => SqlValue::Int8(g.sum),
                                    GroupedAggKind::Avg if pass.value_is_numeric => {
                                        avg_numeric_sql_value(
                                            sum_i128,
                                            g.count as usize,
                                            pass.value_scale,
                                        )
                                    }
                                    GroupedAggKind::Avg => {
                                        average_sql_value(sum_i128, g.count as usize)
                                    }
                                    GroupedAggKind::Min if pass.value_is_uuid => {
                                        SqlValue::Uuid(g.min_uuid)
                                    }
                                    GroupedAggKind::Max if pass.value_is_uuid => {
                                        SqlValue::Uuid(g.max_uuid)
                                    }
                                    // S2.2b-i: device-gathered MIN/MAX text (text_value_minmax), aligned with
                                    // reference.groups by index i. count==0 was handled above (returns NULL).
                                    GroupedAggKind::Min if pass.value_is_text => SqlValue::Text(
                                        text_value_minmax[&pass.value_idx].0[i].clone(),
                                    ),
                                    GroupedAggKind::Max if pass.value_is_text => SqlValue::Text(
                                        text_value_minmax[&pass.value_idx].1[i].clone(),
                                    ),
                                    GroupedAggKind::Min => narrow_ordered_value(
                                        pass.value_ty,
                                        g.min,
                                        g.min_hi,
                                        pass.value_scale,
                                    ),
                                    GroupedAggKind::Max => narrow_ordered_value(
                                        pass.value_ty,
                                        g.max,
                                        g.max_hi,
                                        pass.value_scale,
                                    ),
                                    GroupedAggKind::Count => unreachable!("count handled above"),
                                    GroupedAggKind::CountDistinct => {
                                        unreachable!("count distinct handled above")
                                    }
                                }
                            }
                        }
                    };
                    row.push(value);
                }
                rows.push(row);
            }
            // The deterministic default order is applied ON THE GPU below (the gpu_sort_permutation window),
            // not by a host sort -- so the merged rows stay in raw group order here. HAVING (S3: a transient
            // device relation + the predicate VM) and LIMIT/OFFSET (S4: a control-plane window of the device
            // sort permutation) are both ON-DEVICE now; ORDER BY and the default order are GPU sorts. Each
            // clause maps its referenced result column name to an index. All are empty/None for a bare GROUP
            // BY, so only the default GPU order runs there.
            let col_index = |name: &str| -> Result<usize, ExecuteError> {
                let mut hits = bound
                    .selected_columns
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.name.eq_ignore_ascii_case(name));
                let first = hits.next().map(|(i, _)| i).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "GROUP BY ORDER BY / HAVING references unknown column \"{name}\""
                    )))
                })?;
                // PG: a name shared by two aggregates (e.g. SUM(v), SUM(w) -> both "sum") is ambiguous;
                // error rather than silently bind the first.
                if hits.next().is_some() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "GROUP BY ORDER BY / HAVING column reference \"{name}\" is ambiguous"
                    ))));
                }
                Ok(first)
            };
            if !select.having_groups.is_empty() && !rows.is_empty() {
                // HAVING on the GPU (charter: no host relational filter). Build a TRANSIENT device relation
                // from the grouped result and evaluate the HAVING DNF via the SAME device predicate VM as
                // WHERE. The VM is SINGLE-WIDTH per program, so PROMOTE every integer-family column (and
                // value) to ONE comparable width: if the HAVING
                // touches a NUMERIC column/constant the whole predicate is NUMERIC (i128) -> promote integers
                // to Numeric(scale 0); else it is INT (i64) -> promote integers to Int8. This makes int-key +
                // int8-COUNT, and numeric-SUM + int-COUNT, a single width the VM can lower.
                let n_cols = bound.selected_columns.len();
                let numeric_mode = select.having_groups.iter().flatten().try_fold(
                    false,
                    |acc, f| -> Result<bool, ExecuteError> {
                        let idx = col_index(&f.column)?;
                        Ok(acc
                            || matches!(f.value, SqlValue::Numeric(_))
                            || matches!(bound.selected_columns[idx].ty, SqlType::Numeric { .. })
                            || rows.iter().any(|r| matches!(r[idx], SqlValue::Numeric(_))))
                    },
                )?;
                let promote_int: Vec<bool> = (0..n_cols)
                    .map(|c| {
                        rows.iter().any(|r| {
                            matches!(
                                r[c],
                                SqlValue::Int2(_)
                                    | SqlValue::Int4(_)
                                    | SqlValue::Int8(_)
                                    | SqlValue::Date(_)
                                    | SqlValue::Timestamp(_)
                            )
                        })
                    })
                    .collect();
                // Per result column, the transient column's target. An integer-family column is promoted
                // (to Numeric scale 0 in numeric_mode, else Int8). A GENUINE numeric column is normalized to
                // the MAX scale across its values: AVG yields per-GROUP scales (PG division), but
                // build_relational_device_payload stores only mantissas at ONE column scale, so every value
                // must share it -- rescaling UP to the max is exact (no rounding). `col_scale[c] = Some(s)`
                // marks a numeric-target column at scale `s`.
                let col_scale: Vec<Option<u8>> = (0..n_cols)
                    .map(|c| {
                        if promote_int[c] {
                            numeric_mode.then_some(0u8)
                        } else {
                            rows.iter()
                                .filter_map(|r| match &r[c] {
                                    SqlValue::Numeric(d) => Some(d.scale),
                                    _ => None,
                                })
                                .max()
                        }
                    })
                    .collect();
                let having_columns: Vec<RelationalColumn> = bound
                    .selected_columns
                    .iter()
                    .enumerate()
                    .map(|(c, col)| {
                        let mut col = col.clone();
                        if let Some(scale) = col_scale[c] {
                            col.ty = SqlType::Numeric {
                                precision: 38,
                                scale,
                            };
                        } else if promote_int[c] {
                            col.ty = SqlType::Int8;
                        }
                        col
                    })
                    .collect();
                let having_rows: Vec<Vec<SqlValue>> = rows
                    .iter()
                    .map(|r| -> Result<Vec<SqlValue>, ExecuteError> {
                        r.iter()
                            .enumerate()
                            .map(|(c, v)| -> Result<SqlValue, ExecuteError> {
                                let int_as_i64 = match v {
                                    SqlValue::Int2(x) => Some(i64::from(*x)),
                                    SqlValue::Int4(x) | SqlValue::Date(x) => Some(i64::from(*x)),
                                    SqlValue::Int8(x) | SqlValue::Timestamp(x) => Some(*x),
                                    _ => None,
                                };
                                Ok(match (col_scale[c], promote_int[c]) {
                                    // int-family promoted to Numeric (scale 0; int_as_i64 is exact).
                                    (Some(scale), true) => match int_as_i64 {
                                        Some(x) => {
                                            SqlValue::Numeric(Decimal128::new(i128::from(x), scale))
                                        }
                                        None => v.clone(),
                                    },
                                    // genuine numeric -> rescale UP to the column's max scale (exact).
                                    (Some(scale), false) => match v {
                                        SqlValue::Numeric(d) => {
                                            SqlValue::Numeric(d.rescale(scale).map_err(|_| {
                                                ExecuteError::Engine(EngineError::ApplyFailed(
                                                    "HAVING numeric value overflowed normalizing scale"
                                                        .to_string(),
                                                ))
                                            })?)
                                        }
                                        _ => v.clone(),
                                    },
                                    // int-family promoted to Int8 (no numeric in the predicate).
                                    (None, true) => match int_as_i64 {
                                        Some(x) => SqlValue::Int8(x),
                                        None => v.clone(),
                                    },
                                    // text/bool/uuid (or an all-NULL numeric column) -- unchanged.
                                    (None, false) => v.clone(),
                                })
                            })
                            .collect()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // DNF -> ResidentExpr: each filter -> `Column(idx) <op> literal`; AND within a group, OR
                // across groups. col_index still validates each referenced name (unknown / ambiguous).
                let mut dnf: Option<ResidentExpr> = None;
                for group in &select.having_groups {
                    let mut conj: Option<ResidentExpr> = None;
                    for f in group {
                        let leaf = ResidentExpr::Binary {
                            op: having_op_to_resident(f.op),
                            lhs: Box::new(ResidentExpr::Column(col_index(&f.column)?)),
                            rhs: Box::new(having_value_to_resident_literal(
                                &f.value,
                                numeric_mode,
                            )?),
                        };
                        conj = Some(match conj {
                            None => leaf,
                            Some(prev) => ResidentExpr::Binary {
                                op: ResidentBinaryOp::And,
                                lhs: Box::new(prev),
                                rhs: Box::new(leaf),
                            },
                        });
                    }
                    if let Some(c) = conj {
                        dnf = Some(match dnf {
                            None => c,
                            Some(prev) => ResidentExpr::Binary {
                                op: ResidentBinaryOp::Or,
                                lhs: Box::new(prev),
                                rhs: Box::new(c),
                            },
                        });
                    }
                }
                if let Some(predicate) = dnf {
                    let having_table = RelationalTable {
                        schema: table.schema.clone(),
                        name: table.name.clone(),
                        oid: table.oid,
                        columns: having_columns,
                        indexes: Vec::new(),
                        check_constraints: Vec::new(),
                        foreign_keys: Vec::new(),
                        acl: std::collections::BTreeMap::new(),
                    };
                    let (h_snapshot, h_memory) =
                        self.build_transient_relation_residency(&having_table, &having_rows)?;
                    let survivors = self.lower_resident_predicate(
                        &predicate,
                        &having_table,
                        &h_snapshot,
                        &h_memory,
                        having_rows.len() as u64,
                        None,
                    )?;
                    let kept: Vec<Vec<SqlValue>> = survivors
                        .iter()
                        .map(|&i| rows[i as usize].clone())
                        .collect();
                    rows = kept;
                }
            }
            // The grouped result is ordered ON THE GPU (charter: every relational sort is a GPU sort, no
            // host-side finalization) -- both the explicit ORDER BY and, in its absence, the deterministic
            // DEFAULT order. INT/bool keys feed an i64 matrix by group position; TEXT/NUMERIC/UUID keys feed
            // a resident-like payload built (build_relational_device_payload) from just those result
            // columns, which the hetero comparator reads on-device. Single- and multi-key share the path.
            // OFFSET/LIMIT is then a control-plane WINDOW of the device-produced permutation -- never a host
            // relational drain/truncate on result data. gpu_sort_permutation returns the sort index vector
            // (identity for <=1 row / empty order); we slice it to [OFFSET, OFFSET+LIMIT) and gather ONLY
            // that window from the materialized group rows. With no LIMIT the window is the full range, so
            // this is byte-identical to applying the same permutation to the materialized rows.
            if rows.len() > 1 || select.offset.is_some() || select.limit.is_some() {
                let col_types: Vec<SqlType> = bound.selected_columns.iter().map(|c| c.ty).collect();
                // The GROUP-KEY result columns (emitted first by the merge): result columns 0..n_group_cols.
                let n_group_cols = if is_composite_key {
                    2
                } else if let Some(members) = &widekey_cols {
                    members.len()
                } else {
                    1
                };
                let (order, nulls_first): (Vec<(usize, bool)>, Vec<Option<bool>>) =
                    if select.order_by.is_empty() {
                        // DEFAULT order: by the full group-key tuple, ASC, with the NULL group FIRST --
                        // matching the prior host key order (PG leaves a bare GROUP BY unordered; our
                        // stable choice).
                        (
                            (0..n_group_cols).map(|c| (c, false)).collect(),
                            vec![Some(true); n_group_cols],
                        )
                    } else {
                        // Explicit ORDER BY: resolve each key to its result-column index; explicit NULLS
                        // FIRST/LAST honored ON-DEVICE (order_by_nulls_first threaded through; every key
                        // type, incl. int, reads its NULL validity bitmap in the comparator).
                        let mut order: Vec<(usize, bool)> = select
                            .order_by
                            .iter()
                            .map(|o| Ok::<_, ExecuteError>((col_index(&o.column)?, o.descending)))
                            .collect::<Result<_, _>>()?;
                        let mut nulls_first = order_by_nulls_first.to_vec();
                        // Append the GROUP-KEY columns (ASC, NULL group first) as a deterministic TIE-BREAK
                        // so rows that tie on the explicit ORDER BY key(s) -- e.g. two groups with equal
                        // SUM under `ORDER BY sum DESC` -- order by group key. This matches the legacy
                        // probe/host group-ASC tie-break (a documented cross-path contract) and keeps the
                        // result fully deterministic. The grouped result is at most one row per group, so
                        // the extra (cheap) sort key is negligible. A group column already named as an
                        // explicit key is skipped (it would be a redundant, no-op secondary key).
                        for c in 0..n_group_cols {
                            if !order.iter().any(|(idx, _)| *idx == c) {
                                order.push((c, false));
                                nulls_first.push(Some(true));
                            }
                        }
                        (order, nulls_first)
                    };
                let perm =
                    gpu_sort_permutation(&rows, &order, &nulls_first, &col_types, &device_memory)?;
                let start = select.offset.unwrap_or(0).min(perm.len());
                let end = select
                    .limit
                    .map_or(perm.len(), |l| start.saturating_add(l).min(perm.len()));
                let windowed: Vec<Vec<SqlValue>> = perm[start..end]
                    .iter()
                    .map(|&p| rows[p as usize].clone())
                    .collect();
                rows = windowed;
            }
            return Ok(RelationalSelectResult {
                columns: Arc::new(bound.selected_columns),
                rows: rows.into(),
                planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
                executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
                fallback_reason: None,
                access_path: Arc::new(access_path),
            });
        }

        // Scalar aggregate? Compute it from the filtered indices and return a single row.
        if is_aggregate {
            // PG: an aggregate over ZERO surviving rows -- SUM/AVG/MIN/MAX are SQL NULL (COUNT(*) and
            // COUNT(DISTINCT) are 0, handled in their arms below). NULL support is present now, so this
            // resolves the former "the engine cannot represent NULL yet (M3)" hard-error. The result
            // schema (bound.selected_columns) carries the aggregate's column type, so the NULL is typed.
            if indices.is_empty()
                && matches!(
                    select.projection,
                    SelectProjection::Sum { .. }
                        | SelectProjection::Avg { .. }
                        | SelectProjection::Min { .. }
                        | SelectProjection::Max { .. }
                )
            {
                return Ok(RelationalSelectResult {
                    columns: Arc::new(bound.selected_columns),
                    rows: (vec![vec![SqlValue::Null]]).into(),
                    planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
                    executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
                    fallback_reason: None,
                    access_path: Arc::new(access_path),
                });
            }
            let value = match &select.projection {
                // COUNT(*): the surviving row count IS the result (PG returns bigint). The GPU filter
                // + compaction already produced the count; no per-row materialization.
                SelectProjection::CountAll => SqlValue::Int8(indices.len() as i64),
                // SUM(int4): a GPU reduction over the filtered column (gather col[indices] + reduce);
                // PG returns bigint. (An empty filtered set => SQL NULL is handled above the match.)
                SelectProjection::Sum { column } => {
                    let col_idx = relational_column_index(table, column)?;
                    let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    };
                    // PG: SUM(int4) -> bigint; SUM(int8) -> numeric (the sum can exceed i64, so it
                    // reduces into i128 via the two-atomic carry kernel).
                    match table.columns[col_idx].ty {
                        SqlType::Int4 => {
                            let byte_offset =
                                resident_device_int4_column_offset(&snapshot, table, col_idx)?;
                            let sum = device_memory
                                .sum_i32_at_indices_from_payload(byte_offset, &indices)
                                .map_err(map_err)?;
                            SqlValue::Int8(sum)
                        }
                        SqlType::Int8 => {
                            let byte_offset =
                                resident_device_int8_column_offset(&snapshot, table, col_idx)?;
                            let sum = device_memory
                                .sum_i64_at_indices_i128_from_payload(byte_offset, &indices)
                                .map_err(map_err)?;
                            SqlValue::Numeric(Decimal128::new(sum, 0))
                        }
                        // SUM(numeric) -> numeric at the column scale: sum the i128 mantissas (the
                        // partials kernel with CHECKED i128 overflow -> numeric field overflow).
                        SqlType::Numeric { scale, .. } => {
                            let byte_offset =
                                resident_device_numeric_column_offset(&snapshot, table, col_idx)?;
                            let mantissa = device_memory
                                .sum_i128_at_indices_from_payload(byte_offset, &indices)
                                .map_err(map_err)?;
                            SqlValue::Numeric(Decimal128::new(mantissa, scale))
                        }
                        _ => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "SUM supports int4 / int8 / numeric columns on the Expr path"
                                    .to_string(),
                            )));
                        }
                    }
                }
                // MIN/MAX(int4): a GPU reduction; PG MIN/MAX preserve the column type (int4 -> int4).
                // (An empty filtered set => SQL NULL is handled above the match.)
                SelectProjection::Min { column } | SelectProjection::Max { column } => {
                    let col_idx = relational_column_index(table, column)?;
                    let is_max = matches!(select.projection, SelectProjection::Max { .. });
                    let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    };
                    // MIN/MAX preserve the column type (PG): int4 -> int4 (i32 reduce), int8 -> int8
                    // (i64 reduce). Other types are follow-ons.
                    match table.columns[col_idx].ty {
                        SqlType::Int4 => {
                            let byte_offset =
                                resident_device_int4_column_offset(&snapshot, table, col_idx)?;
                            let value = if is_max {
                                device_memory.max_i32_at_indices_from_payload(byte_offset, &indices)
                            } else {
                                device_memory.min_i32_at_indices_from_payload(byte_offset, &indices)
                            }
                            .map_err(map_err)?;
                            SqlValue::Int4(value)
                        }
                        SqlType::Int8 => {
                            let byte_offset =
                                resident_device_int8_column_offset(&snapshot, table, col_idx)?;
                            let value = if is_max {
                                device_memory.max_i64_at_indices_from_payload(byte_offset, &indices)
                            } else {
                                device_memory.min_i64_at_indices_from_payload(byte_offset, &indices)
                            }
                            .map_err(map_err)?;
                            SqlValue::Int8(value)
                        }
                        // MIN/MAX(numeric) -> numeric: reduce the i128 mantissas (no native 128-bit
                        // atomic, so a partials + host-combine reduction); the result carries the
                        // column scale.
                        SqlType::Numeric { scale, .. } => {
                            let byte_offset =
                                resident_device_numeric_column_offset(&snapshot, table, col_idx)?;
                            let mantissa = if is_max {
                                device_memory
                                    .max_i128_at_indices_from_payload(byte_offset, &indices)
                            } else {
                                device_memory
                                    .min_i128_at_indices_from_payload(byte_offset, &indices)
                            }
                            .map_err(map_err)?;
                            SqlValue::Numeric(Decimal128::new(mantissa, scale))
                        }
                        _ => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "MIN / MAX support int4 / int8 / numeric columns on the Expr path"
                                    .to_string(),
                            )));
                        }
                    }
                }
                // AVG(int4) = the GPU SUM / the count, as numeric (PG). The reduction is on the GPU;
                // the final scalar divide reuses `average_sql_value` (scale-16, matching the enumerated
                // path). (An empty filtered set => SQL NULL is handled above the match.)
                SelectProjection::Avg { column } => {
                    let col_idx = relational_column_index(table, column)?;
                    let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    };
                    // AVG -> numeric = the GPU sum / the count. int4 reduces to i64 (widened), int8 to
                    // i128 (the two-atomic carry); both divide via average_sql_value (integer sum).
                    // numeric sums the i128 mantissas and divides via avg_numeric_sql_value, which
                    // carries the column scale into PG's division-scale derivation.
                    match table.columns[col_idx].ty {
                        SqlType::Int4 => {
                            let byte_offset =
                                resident_device_int4_column_offset(&snapshot, table, col_idx)?;
                            let sum = i128::from(
                                device_memory
                                    .sum_i32_at_indices_from_payload(byte_offset, &indices)
                                    .map_err(map_err)?,
                            );
                            average_sql_value(sum, indices.len())
                        }
                        SqlType::Int8 => {
                            let byte_offset =
                                resident_device_int8_column_offset(&snapshot, table, col_idx)?;
                            let sum = device_memory
                                .sum_i64_at_indices_i128_from_payload(byte_offset, &indices)
                                .map_err(map_err)?;
                            average_sql_value(sum, indices.len())
                        }
                        SqlType::Numeric { scale, .. } => {
                            let byte_offset =
                                resident_device_numeric_column_offset(&snapshot, table, col_idx)?;
                            let mantissa = device_memory
                                .sum_i128_at_indices_from_payload(byte_offset, &indices)
                                .map_err(map_err)?;
                            avg_numeric_sql_value(mantissa, indices.len(), scale)
                        }
                        _ => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "AVG supports int4 / int8 / numeric columns on the Expr path"
                                    .to_string(),
                            )));
                        }
                    }
                }
                // Scalar COUNT(DISTINCT v) (no GROUP BY) = ONE group: sort/mark/SUM over (g=0, v) via
                // `count_distinct_groups` with a constant group key, so the single group's count is the
                // total distinct. PG: COUNT(DISTINCT) over zero rows is 0 (NOT NULL), so an empty
                // filtered set returns 0 (the lone exception to the SUM/AVG empty-set NULL hard-error).
                SelectProjection::CountDistinct { column } => {
                    let value_idx = relational_column_index(table, column)?;
                    // M3 (doc 21): COUNT(DISTINCT v) counts distinct NON-NULL values (PG). A nullable
                    // v is handled by the aggregate validity conjunct ANDed into the predicate above
                    // (`v IS NOT NULL`), so the surviving `indices` here contain no NULL values — the
                    // raw sort/mark/SUM pass over them is PG-exact. (The GROUPED COUNT(DISTINCT)
                    // paths still guard nullable values with a clean error.)
                    if indices_u64.is_empty() {
                        SqlValue::Int8(0)
                    } else {
                        let g_vals = vec![0i64; indices_u64.len()];
                        let groups = count_distinct_groups(value_idx, &g_vals, &indices_u64)?;
                        // The constant key yields exactly one group; its SUM = the total distinct.
                        SqlValue::Int8(groups.first().map_or(0, |g| g.sum))
                    }
                }
                _ => unreachable!(
                    "is_aggregate gates on CountAll | Sum | Min | Max | Avg | CountDistinct"
                ),
            };
            return Ok(RelationalSelectResult {
                columns: Arc::new(bound.selected_columns),
                rows: (vec![vec![value]]).into(),
                planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
                executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
                fallback_reason: None,
                access_path: Arc::new(access_path),
            });
        }

        // Non-grouped ORDER BY: reorder the surviving indices by the order key(s) on the GPU (bitonic
        // sort) BEFORE gathering, so the projected rows come out sorted -- a charter-native GPU sort,
        // not a host/CPU sort. The routing only sends sorts whose keys are ALL i64-sortable int columns
        // (int2/int4/int8/date/timestamp) to this path -- one key OR several (`ORDER BY a ASC, b DESC`).
        // A single TEXT-key ORDER BY takes the varlen GPU sort path (the byte-wise comparator); all
        // other routed keys are i64-sortable ints and go through the key-matrix path below.
        // Classify the ORDER BY keys: k==1 text -> the varlen text sort; k>1 with ANY text key -> the
        // heterogeneous mixed (int+text) sort; all-int -> the i64 key matrix.
        // Materialize an INT-bearing ORDER BY key's i64 values for `indices`. A sort EXPRESSION
        // (`ORDER BY a+b`) is evaluated on-device into an i64 column (checked int4 overflow -> PG error,
        // never CPU); a plain int column is projected. Both sort arms below use this; an expression key
        // always counts as an int key. `indices` is a param (the else arm below moves `indices_u64`).
        let materialize_int_key_column =
            |ki: usize, indices: &[u64]| -> Result<Vec<i64>, ExecuteError> {
                let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                    ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                };
                if let Some(Some(expr)) = order_by_exprs.get(ki) {
                    let mut program = Vec::new();
                    compile_arith_program(expr, table, &snapshot, &mut program)?;
                    let idx32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
                    // int8 operands -> the i64 arith VM + i64 value width; else int4/I32. Reading an int8
                    // expr as I32 would stride a BIGINT column by 4 bytes -> silently garbage sort keys.
                    let elem = if expr_mentions_int8(expr, table) {
                        ResidentElemType::I64
                    } else {
                        ResidentElemType::I32
                    };
                    // M3 (doc 21): a NULLABLE int4 expression blends the i64::MAX default-end sentinel
                    // ON-DEVICE where any operand is NULL (a validity mask VM run + the on-device blend). int8
                    // nullable + explicit NULLS FIRST/LAST are clean-errored at the routing loop below; a
                    // non-nullable expression keeps the plain (no-validity) path.
                    if elem == ResidentElemType::I32
                        && predicate_references_nullable_column(expr, table, &snapshot)?
                    {
                        let mut validity_program = vec![ExprStep::ConstMask { value: true }];
                        push_leaf_validity_and(&[expr], table, &snapshot, &mut validity_program)?;
                        return device_memory
                            .arith_value_column_at_indices_nullable(
                                &program,
                                &validity_program,
                                row_count,
                                &idx32,
                            )
                            .map_err(map_err);
                    }
                    return device_memory
                        .arith_value_column_at_indices(&program, row_count, &idx32, elem)
                        .map_err(map_err);
                }
                let order = &select.order_by[ki];
                let order_idx = relational_column_index(table, &order.column)?;
                let keys: Vec<i64> = match table.columns[order_idx].ty {
                    SqlType::Int4 | SqlType::Int2 | SqlType::Date => device_memory
                        .project_i32_rows_from_payload(
                            resident_device_int4_column_offset(&snapshot, table, order_idx)?,
                            indices,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                    SqlType::Int8 | SqlType::Timestamp => device_memory
                        .project_i64_rows_from_payload(
                            resident_device_int8_column_offset(&snapshot, table, order_idx)?,
                            indices,
                        )
                        .map_err(map_err)?,
                    _ => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "ORDER BY key is not an i64-sortable int column".to_string(),
                        )))
                    }
                };
                // M3 (doc 21): NULL placement is done ON-DEVICE by the sort comparator (it reads the per-key
                // validity bitmap and orders NULL keys to the PG-default end), so the key VALUES here are raw —
                // a NULL row's placeholder value is never compared. (A nullable key always routes to the
                // validity-aware hetero comparator below; this raw matrix only feeds that path or the non-null
                // pure-int path.) No host-side NULL overwrite.
                Ok(keys)
            };
        // A sort EXPRESSION is int-valued. has_text_key (TEXT only) gates the single-text fast path;
        // has_hetero_key (TEXT/NUMERIC/UUID -- the keys that can't live in the i64 matrix) routes to the
        // heterogeneous comparator.
        let mut has_text_key = false;
        let mut has_hetero_key = false;
        // M3 (doc 21): per-key NULL validity bitmap byte offset (sentinel u64::MAX = the key holds no NULL).
        // The hetero sort comparator reads this ON-DEVICE and orders NULL keys to the PG-default end (last
        // ASC / first DESC) — GPU-native, no host shard / sentinel. A nullable key of ANY type routes
        // to that comparator (below). A nullable sort EXPRESSION stays a clean-error follow-up (an
        // expression has no single column validity bitmap; a derived one is a follow-up).
        let mut key_null_offs: Vec<u64> = vec![u64::MAX; select.order_by.len()];
        for (ki, order) in select.order_by.iter().enumerate() {
            if let Some(Some(expr)) = order_by_exprs.get(ki) {
                // A NULLABLE int4 expression is supported via the on-device value-sentinel
                // (materialize_int_key_column): a NULL result becomes the i64::MAX default-end sentinel,
                // so key_null_offs stays u64::MAX (no validity bitmap -- the NULL is value-encoded). Two
                // cases still clean-error: an int8 expression (i64::MAX could collide with a real result)
                // and an explicit NULLS FIRST/LAST (the value-sentinel only realizes the PG default).
                if predicate_references_nullable_column(expr, table, &snapshot)? {
                    if expr_mentions_int8(expr, table) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "ORDER BY a nullable int8 expression is a follow-up (the i64::MAX NULL \
                             sentinel could collide with a real bigint result)"
                                .to_string(),
                        )));
                    }
                    if order_by_nulls_first.get(ki).copied().flatten().is_some() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "explicit NULLS FIRST/LAST on a nullable ORDER BY expression is a follow-up \
                             (the default NULL placement is supported)"
                                .to_string(),
                        )));
                    }
                }
                continue;
            }
            let order_idx = relational_column_index(table, &order.column)?;
            match table.columns[order_idx].ty {
                SqlType::Text => {
                    has_text_key = true;
                    has_hetero_key = true;
                }
                SqlType::Numeric { .. } | SqlType::Uuid => has_hetero_key = true,
                _ => {}
            }
            if let Some(off) = resident_device_null_column_offset(&snapshot, table, order_idx)? {
                key_null_offs[ki] = off;
            }
        }
        // A nullable key (any type) must route to the validity-aware hetero comparator. A non-null single
        // text key keeps the dedicated text fast path; non-null int-only keys keep the int matrix/radix path.
        let any_nullable_key = key_null_offs.iter().any(|&o| o != u64::MAX);
        let single_text_key = select.order_by.len() == 1 && has_text_key && !any_nullable_key;
        let use_hetero = has_hetero_key || any_nullable_key;
        let indices_u64 = if select.order_by.is_empty() {
            indices_u64
        } else if single_text_key {
            // The varlen text key can't live in the i64 key matrix, so the kernel sorts the surviving
            // rows by reading each row's bytes from the resident text column via the indices indirection
            // (lexicographic, unsigned bytes, a prefix sorts smaller). Charter-native GPU sort.
            let order = &select.order_by[0];
            let order_idx = relational_column_index(table, &order.column)?;
            let layout = resident_device_text_column_layout(&snapshot, table, order_idx)?;
            let perm = device_memory
                .bitonic_sort_text(
                    &indices_u64,
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    order.descending,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            perm.iter().map(|&p| indices_u64[p as usize]).collect()
        } else if use_hetero {
            // A key tuple with a TEXT/NUMERIC/UUID key (`ORDER BY name /*text*/, age /*int*/`, or a
            // single numeric/uuid key): the heterogeneous GPU comparator dispatches each key to the s64
            // compare (int, from a row-major by-position matrix), the byte compare (text, in place), or
            // the 16-byte compare (numeric = signed-hi/unsigned-lo i128; uuid = big-endian unsigned),
            // reading numeric/uuid in place from the resident column. Charter-native GPU sort.
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            let n = indices_u64.len();
            let k = select.order_by.len();
            if k > 64 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "ORDER BY supports at most 64 sort keys on the GPU sort path".to_string(),
                )));
            }
            // Walk the keys in order: each int key takes the next int-matrix column, each text key the
            // next text-column slot; build key_plan (bit31=is_text, low bits=slot) + desc_mask.
            let mut num_int = 0usize;
            let mut text_cols: Vec<(u64, u64)> = Vec::new();
            let mut b128_cols: Vec<u64> = Vec::new();
            let mut key_plan: Vec<u32> = Vec::with_capacity(k);
            let mut desc_mask: u64 = 0;
            // Per-key effective NULLS FIRST bit: the explicit override, else the key's DESC (PG default).
            // The comparator reads it to place NULLs ON-DEVICE, decoupled from the value-compare direction.
            let mut nulls_first_mask: u64 = 0;
            let mut int_key_slots: Vec<(usize, usize)> = Vec::new();
            for (ki, order) in select.order_by.iter().enumerate() {
                if order.descending {
                    desc_mask |= 1u64 << ki;
                }
                if order_by_nulls_first
                    .get(ki)
                    .copied()
                    .flatten()
                    .unwrap_or(order.descending)
                {
                    nulls_first_mask |= 1u64 << ki;
                }
                // A sort EXPRESSION is an int-valued key -> the next int slot (materialized below).
                if order_by_exprs.get(ki).is_some_and(|e| e.is_some()) {
                    let int_slot = num_int;
                    num_int += 1;
                    key_plan.push(int_slot as u32);
                    int_key_slots.push((ki, int_slot));
                    continue;
                }
                let order_idx = relational_column_index(table, &order.column)?;
                match table.columns[order_idx].ty {
                    SqlType::Text => {
                        let layout =
                            resident_device_text_column_layout(&snapshot, table, order_idx)?;
                        let text_slot = text_cols.len() as u32;
                        text_cols.push((layout.offsets_byte_offset, layout.bytes_byte_offset));
                        // kind 1 (key_plan bits 30-31 = 01) = text.
                        key_plan.push(0x4000_0000_u32 | text_slot);
                    }
                    SqlType::Numeric { .. } => {
                        let b128_slot = b128_cols.len() as u32;
                        b128_cols.push(resident_device_numeric_column_offset(
                            &snapshot, table, order_idx,
                        )?);
                        // kind 2 (bits 30-31 = 10) = numeric (signed-hi/unsigned-lo i128).
                        key_plan.push(0x8000_0000_u32 | b128_slot);
                    }
                    SqlType::Uuid => {
                        let b128_slot = b128_cols.len() as u32;
                        b128_cols.push(resident_device_numeric_column_offset(
                            &snapshot, table, order_idx,
                        )?);
                        // kind 3 (bits 30-31 = 11) = uuid (big-endian unsigned).
                        key_plan.push(0xC000_0000_u32 | b128_slot);
                    }
                    SqlType::Int4
                    | SqlType::Int2
                    | SqlType::Date
                    | SqlType::Int8
                    | SqlType::Timestamp => {
                        let int_slot = num_int;
                        num_int += 1;
                        key_plan.push(int_slot as u32);
                        int_key_slots.push((ki, int_slot));
                    }
                    _ => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "mixed-key ORDER BY on the Expr path supports int2 / int4 / int8 / date / \
                             timestamp / text / numeric / uuid columns (the GPU sort); other types \
                             are a follow-on"
                                .to_string(),
                        )));
                    }
                }
            }
            // Materialize the int keys into a row-major n*num_int matrix by position (matching the
            // multikey kernel's layout; the text keys read in place via the indices indirection).
            let mut int_keys = vec![0i64; n * num_int];
            for (ki, int_slot) in int_key_slots {
                let col_keys = materialize_int_key_column(ki, &indices_u64)?;
                for (i, v) in col_keys.into_iter().enumerate() {
                    int_keys[i * num_int + int_slot] = v;
                }
            }
            let perm = device_memory
                .bitonic_sort_hetero(
                    &indices_u64,
                    &int_keys,
                    num_int,
                    &text_cols,
                    &b128_cols,
                    &key_plan,
                    desc_mask,
                    &key_null_offs,
                    nulls_first_mask,
                )
                .map_err(map_err)?;
            perm.iter().map(|&p| indices_u64[p as usize]).collect()
        } else {
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            let n = indices_u64.len();
            let k = select.order_by.len();
            if k > 64 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "ORDER BY supports at most 64 sort keys on the GPU sort path".to_string(),
                )));
            }
            // Materialize every ORDER BY key column's i64 value for the surviving rows into a row-major
            // key matrix (`keys[row * K + key]`, key 0 most significant) and the per-key direction mask
            // (bit k set => key k is DESC). int2/int4/date sign-extend through i32; int8/timestamp are
            // the full i64. The GPU multi-key comparator breaks ties on key 0 by key 1, then key 2, ...
            let mut key_matrix = vec![0i64; n * k];
            let mut desc_mask: u64 = 0;
            for (kk, order) in select.order_by.iter().enumerate() {
                if order.descending {
                    desc_mask |= 1u64 << kk;
                }
                // Each key is a plain int column OR a sort expression (`a+b`); the helper materializes
                // the i64 value column either way (expression -> on-device eval with checked overflow).
                let col_keys = materialize_int_key_column(kk, &indices_u64)?;
                for (i, v) in col_keys.into_iter().enumerate() {
                    key_matrix[i * k + kk] = v;
                }
            }
            // Single-key dispatches by size -- bitonic for small n, radix (O(n)) for large n -- via
            // order_by_sort_i64; multi-key uses the row-major bitonic comparator.
            let perm = if k == 1 {
                device_memory
                    .order_by_sort_i64(&key_matrix, (desc_mask & 1) != 0)
                    .map_err(map_err)?
            } else {
                device_memory
                    .bitonic_sort_multikey(&key_matrix, n, k, desc_mask)
                    .map_err(map_err)?
            };
            perm.iter().map(|&p| indices_u64[p as usize]).collect()
        };
        // LIMIT/OFFSET as control-plane WINDOWING of the device-ordered index vector: slice the surviving
        // indices to the [OFFSET, OFFSET+LIMIT) window BEFORE the column gather, so only the kept rows are
        // materialized from the device (we never gather rows that would then be dropped -- the real win).
        // SQL clause order: ORDER BY (the GPU sort above) -> OFFSET -> LIMIT. Slicing an index vector is
        // control-plane; the relational ordering+windowing decision rode the device sort.
        let indices_u64 = if select.offset.is_some() || select.limit.is_some() {
            let start = select.offset.unwrap_or(0).min(indices_u64.len());
            let end = select.limit.map_or(indices_u64.len(), |l| {
                start.saturating_add(l).min(indices_u64.len())
            });
            indices_u64[start..end].to_vec()
        } else {
            indices_u64
        };
        // Materialize: gather each projected column at the surviving row indices on the GPU, by type
        // (int4 -> i32 gather, int8 -> i64 gather; the type matrix, doc 19).
        enum ProjectedColumn {
            Int4(Vec<i32>),
            Int8(Vec<i64>),
            Numeric(Vec<i128>, u8),
            Date(Vec<i32>),
            Timestamp(Vec<i64>),
            // Stored as the i128 the i128 projector returns; the raw 16 uuid bytes are its LE form.
            Uuid(Vec<i128>),
            // Stored as the widened i32s the int4 projector returns; narrowed back to i16 per row.
            Int2(Vec<i32>),
            // Gathered straight from the 1-bit-per-row bool bitmap.
            Bool(Vec<bool>),
            // The text VALUE is gathered ON-DEVICE from the resident payload's offsets+bytes sections
            // (project_text_rows_from_payload) at the GPU-sorted surviving indices -- no host_rows read
            // (the host is control plane only). A NULL/empty cell returns ""; the validity override below
            // restores SqlValue::Null for a NULL row.
            Text(Vec<String>),
        }
        let mut projected_columns: Vec<ProjectedColumn> =
            Vec::with_capacity(bound.selected_indexes.len());
        for &col in &bound.selected_indexes {
            let column = match table.columns[col].ty {
                SqlType::Int8 => {
                    let byte_offset = resident_device_int8_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i64_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Int8(values)
                }
                SqlType::Numeric { scale, .. } => {
                    let byte_offset = resident_device_numeric_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i128_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Numeric(values, scale)
                }
                SqlType::Date => {
                    // Date rides the i32 section; project it as i32 then tag it as a date.
                    let byte_offset = resident_device_int4_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i32_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Date(values)
                }
                SqlType::Int2 => {
                    // Smallint rides the i32 section widened; project as i32, narrow per row below.
                    let byte_offset = resident_device_int4_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i32_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Int2(values)
                }
                SqlType::Bool => {
                    // Bool is a 1-bit-per-row bitmap; gather the selected rows' bits.
                    let byte_offset = resident_device_bool_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_bool_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Bool(values)
                }
                SqlType::Timestamp => {
                    // Timestamp rides the i64 section; project it as i64 then tag it as a timestamp.
                    let byte_offset = resident_device_int8_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i64_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Timestamp(values)
                }
                SqlType::Uuid => {
                    // Uuid rides the i128 section; project as i128 (its LE bytes are the raw uuid).
                    let byte_offset = resident_device_numeric_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i128_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Uuid(values)
                }
                SqlType::Text => {
                    // Gather the text VALUE ON-DEVICE from the resident payload's offsets+bytes sections at
                    // the GPU-sorted surviving indices -- no host_rows read (the host is control plane
                    // only; the GPU did the filter + sort AND now materializes the strings). A NULL/empty
                    // cell returns ""; the projected_validity override below restores SqlValue::Null.
                    let layout = resident_device_text_column_layout(&snapshot, table, col)?;
                    let values = device_memory
                        .project_text_rows_from_payload(
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            layout.bytes_len,
                            row_count,
                            &indices_u64,
                        )
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Text(values)
                }
                _ => {
                    let byte_offset = resident_device_int4_column_offset(&snapshot, table, col)?;
                    let values = device_memory
                        .project_i32_rows_from_payload(byte_offset, &indices_u64)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    ProjectedColumn::Int4(values)
                }
            };
            projected_columns.push(column);
        }
        // M3 (doc 21): per projected column, the NULL validity of each surviving row (1 = present). A
        // nullable column's device value is a 0/empty PLACEHOLDER for a NULL row, so the result must emit
        // SqlValue::Null there. The validity bitmap is 1-bit-per-row like a bool column, so the bool
        // projector gathers it at the surviving indices. `None` = the column holds no NULLs (all valid),
        // so non-nullable projections are unchanged. (Text's device gather returns "" for a NULL cell, so
        // this override is what restores its SqlValue::Null.)
        let projected_validity: Vec<Option<Vec<bool>>> = bound
            .selected_indexes
            .iter()
            .map(|&col| -> Result<Option<Vec<bool>>, ExecuteError> {
                match resident_device_null_column_offset(&snapshot, table, col)? {
                    Some(off) => Ok(Some(
                        device_memory
                            .project_bool_rows_from_payload(off, &indices_u64)
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?,
                    )),
                    None => Ok(None),
                }
            })
            .collect::<Result<_, _>>()?;
        // Build the result rows FLAT, directly into the RowBlock (DECISIONS read-path lever): the dominant
        // general-route cost was the per-row `Vec<SqlValue>` boxing PLUS the `rows.into()` RowBlock
        // conversion (a 2nd O(n) pass) -- together ~75% of the per-row time (measured 38+39 of 102 ns/row).
        // Emit ONE row-major `Vec<SqlValue>` (no inner Vec, no conversion). Byte-identical: same values in
        // the same row-major order `RowBlock::from(Vec<Vec<_>>)` produced.
        let ncols = projected_columns.len();
        let mut flat: Vec<SqlValue> = Vec::with_capacity(indices_u64.len() * ncols);
        for row in 0..indices_u64.len() {
            for (c, column) in projected_columns.iter().enumerate() {
                // A NULL row (validity bit 0) projects as SQL NULL regardless of its placeholder.
                if projected_validity[c]
                    .as_ref()
                    .is_some_and(|validity| !validity[row])
                {
                    flat.push(SqlValue::Null);
                    continue;
                }
                flat.push(match column {
                    ProjectedColumn::Int4(values) => SqlValue::Int4(values[row]),
                    ProjectedColumn::Int8(values) => SqlValue::Int8(values[row]),
                    ProjectedColumn::Numeric(values, scale) => {
                        SqlValue::Numeric(Decimal128::new(values[row], *scale))
                    }
                    ProjectedColumn::Date(values) => SqlValue::Date(values[row]),
                    ProjectedColumn::Timestamp(values) => SqlValue::Timestamp(values[row]),
                    ProjectedColumn::Uuid(values) => SqlValue::Uuid(values[row].to_le_bytes()),
                    // The stored i32 is a widened i16, so the narrowing is exact.
                    ProjectedColumn::Int2(values) => SqlValue::Int2(values[row] as i16),
                    ProjectedColumn::Bool(values) => SqlValue::Bool(values[row]),
                    // Device-gathered String; a NULL row was already handled by the validity override
                    // above, so a bare value here is a real (possibly empty) string.
                    ProjectedColumn::Text(values) => SqlValue::Text(values[row].clone()),
                });
            }
        }
        // OFFSET/LIMIT was already applied as a control-plane window of `indices_u64` above (before the
        // gather), so `flat` is the final windowed result -- no host drain/truncate on result data.
        Ok(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns),
            rows: RowBlock::flat(flat, ncols),
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path: Arc::new(access_path),
        })
    }

    /// Try to lower a SIMPLE int8 comparison to surviving row indices via the i64 compare kernels
    /// (the type matrix, doc 19). Supported shapes: `int8col <cmp> int4literal` (the int4 literal is
    /// widened to int8 — PG's int4->int8 coercion), `int4literal <cmp> int8col` (operand order
    /// preserved via `scalar_on_left`), and `int8col <cmp> int8col`.
    ///
    /// Returns `Ok(None)` for a pure-int4 predicate (the caller falls through to the int4 path) and
    /// `Err` when int8 appears in a shape not yet supported (int8 arithmetic, or a mixed int4/int8
    /// expression) — never a silent fall-through that would mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_int8_predicate(
        &self,
        predicate: &ResidentExpr,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        let lhs_int8 = int8_column_index(lhs, table);
        let rhs_int8 = int8_column_index(rhs, table);
        // Surface the device error verbatim (e.g. int64 overflow -> "bigint out of range"), matching
        // the int4 path's mapping.
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };

        // The simple int8 comparison PEEPHOLE (resident-column compare kernels) — only for an actual
        // comparison op. `predicate_compare_code` is None for AND/OR, so boolean combinators skip this
        // and fall through to the general int8 path below (NOT the int4 path — that was the silent
        // mis-answer the audit caught: int8 AND/OR read with the i32 4-byte stride).
        if let Some(cmp) = predicate_compare_code(compare) {
            match (lhs, rhs) {
                (ResidentExpr::Int4Literal(scalar), _) if rhs_int8.is_some() => {
                    let offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        rhs_int8.expect("checked"),
                    )?;
                    return device_memory
                        .expr_i64_compare_scalar_filter(
                            offset,
                            i64::from(*scalar),
                            true,
                            cmp,
                            row_count,
                        )
                        .map(Some)
                        .map_err(map_err);
                }
                (_, ResidentExpr::Int4Literal(scalar)) if lhs_int8.is_some() => {
                    let offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        lhs_int8.expect("checked"),
                    )?;
                    return device_memory
                        .expr_i64_compare_scalar_filter(
                            offset,
                            i64::from(*scalar),
                            false,
                            cmp,
                            row_count,
                        )
                        .map(Some)
                        .map_err(map_err);
                }
                _ if lhs_int8.is_some() && rhs_int8.is_some() => {
                    let a_offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        lhs_int8.expect("checked"),
                    )?;
                    let b_offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        rhs_int8.expect("checked"),
                    )?;
                    return device_memory
                        .expr_i64_compare_columns_filter(a_offset, b_offset, cmp, row_count)
                        .map(Some)
                        .map_err(map_err);
                }
                _ => {}
            }
        }

        // General int8 path: int8 ARITHMETIC / AND-OR / deeper predicates lower via the i64 buffer VM
        // (int4 LITERALS are widened to i64). A pure-int4 predicate returns None (the int4 paths handle
        // it); a MIXED int4/int8 expression is a hard error — never a silent mis-answer.
        let mentions_int8 = expr_mentions_int8(lhs, table) || expr_mentions_int8(rhs, table);
        if !mentions_int8 {
            return Ok(None);
        }
        // ADR-006 (MIXED-WIDTH groups): a mixed int8+{int4,int2,text,bool,date,uuid} predicate
        // now runs on the I32 mask VM when every int8 leaf is a WIDTH-SAFE scalar comparison (the
        // `LoadColumnI64`+`CompareScalarI64` arm — see `mixed_width_i32_elem`): fall through
        // (`Ok(None)`) to the general And/Or I32 branch. Otherwise: a mix bearing a 4-BYTE value
        // leaf (int4/int2/date — `LoadColumn` follows the program width and would 8-byte mis-read
        // in this I64 program; int2 was the audit's MEDIUM: it used to slip past this block into
        // the I64 compile) or a text/bool mask leaf keeps the HARD ERROR — never a silent
        // mis-answer at one element width. A UUID-ONLY mix with non-scalar int8 (col-vs-col /
        // arith) deliberately falls THROUGH to the I64 compile below — the uuid leaf is a
        // width-agnostic mask (`UuidCmpMask` reads b128 bytes, not elem-strided) and that path
        // pre-dates this slice (audit LOW: rejecting it here regressed a working shape).
        // int8-only predicates stay on this i64 VM below.
        let mentions_width_bound = expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int2(lhs, table)
            || expr_mentions_int2(rhs, table)
            || expr_mentions_text(lhs, table)
            || expr_mentions_text(rhs, table)
            || expr_mentions_bool_column(lhs, table)
            || expr_mentions_bool_column(rhs, table)
            || expr_mentions_date(lhs, table)
            || expr_mentions_date(rhs, table);
        if mentions_width_bound || expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table)
        {
            if mixed_width_i32_elem(predicate, table).is_some() {
                return Ok(None);
            }
            if mentions_width_bound {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports a mixed int8 predicate only when every \
                     int8 leaf is a bare-column scalar comparison"
                        .to_string(),
                )));
            }
            // uuid-only mix with non-scalar int8: the pre-slice I64 path serves it correctly.
        }
        let mut program = Vec::new();
        let mut needles: Vec<Vec<u8>> = Vec::new();
        compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
        device_memory
            .run_expr_predicate_filter(&program, row_count, ResidentElemType::I64)
            .map(Some)
            .map_err(map_err)
    }

    /// Cross-scale `numcol <cmp> literal` (or flipped): the literal is FINER than the column, so load
    /// the column, rescale it UP to the literal's scale (mantissa * 10^k, the buffer VM path), and
    /// compare to the literal's mantissa. `scalar_on_left` is the literal's side of the comparison.
    #[allow(clippy::too_many_arguments)]
    fn numeric_cross_scale_scalar(
        &self,
        column_offset: u64,
        column_scale: u8,
        literal: Decimal128,
        scalar_on_left: bool,
        cmp: u32,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        let needle = i32::try_from(literal.mantissa).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric comparison literal is too large for the fast path yet".to_string(),
            ))
        })?;
        let mut program = vec![ExprStep::LoadColumn {
            byte_offset: column_offset,
        }];
        push_numeric_rescale(column_scale, literal.scale, &mut program)?;
        program.push(ExprStep::CompareScalar {
            cmp,
            scalar: needle,
            scalar_on_left,
        });
        device_memory
            .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    /// Try to lower a SIMPLE date comparison (`datecol <cmp> 'YYYY-MM-DD'`, either operand order, or
    /// `datecol <cmp> datecol`) to surviving row indices via the i32 compare path (the type matrix,
    /// doc 19): a `date` is i32 days since 2000-01-01, so it reuses the int4 residency section + the
    /// I32 VM. The string literal is coerced to a day count (`parse_date`) at lowering, like PG. Returns
    /// None for a non-date predicate. Date `AND`/`OR` / arithmetic, and a date compared to a non-date
    /// value, are hard errors -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_date_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_date(lhs, table) || expr_mentions_date(rhs, table)) {
            return Ok(None);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            // AND/OR (ADR-006 date compound): fall through — the And/Or branch of
            // `lower_resident_predicate` runs `compile_predicate_program` at I32, whose DATE leaf
            // (LoadColumn 4-byte + CompareScalar days + validity) serves date ranges. A shape the
            // leaf can't express errors there and the caller declines — never a silent mis-answer.
            return Ok(None);
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let run = |program: &[ExprStep]| {
            device_memory
                .run_expr_predicate_filter(program, row_count, ResidentElemType::I32)
                .map(Some)
                .map_err(map_err)
        };
        match (date_column_index(lhs, table), date_column_index(rhs, table)) {
            (Some(col), None) => {
                let days = date_literal_days(rhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar: days,
                        scalar_on_left: false,
                    },
                ])
            }
            (None, Some(col)) => {
                let days = date_literal_days(lhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar: days,
                        scalar_on_left: true,
                    },
                ])
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_int4_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_int4_column_offset(snapshot, table, b)?;
                run(&[
                    ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    },
                    ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    },
                    ExprStep::CompareBuffers { cmp },
                ])
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a date predicate must involve a date column".to_string(),
            ))),
        }
    }

    /// Try to lower a SIMPLE timestamp comparison (`ts_col <cmp> 'YYYY-MM-DD HH:MM:SS'`, either order,
    /// or `ts_col <cmp> ts_col`) to surviving row indices via the i64 compare KERNELS (the type matrix,
    /// doc 19): a `timestamp` is i64 microseconds, so it reuses the int8 residency section + the i64
    /// compare kernels. (The i64 microsecond literal exceeds the i32 `ExprStep` scalar, so this uses
    /// `expr_i64_compare_scalar_filter` directly, not the VM's i32-scalar step.) Returns None for a
    /// non-timestamp predicate. Timestamp `AND`/`OR` / arithmetic, and a timestamp compared to a
    /// non-timestamp value, are hard errors -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn try_lower_timestamp_predicate(
        &self,
        predicate: &ResidentExpr,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_timestamp(lhs, table) || expr_mentions_timestamp(rhs, table)) {
            return Ok(None);
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let Some(cmp) = predicate_compare_code(compare) else {
            // AND/OR (ADR-006 multi-bound timestamp DML): a timestamp is i64 micros in the i64
            // section, so a TIMESTAMP-ONLY (i64-section) AND/OR lowers via the i64 buffer VM — the
            // SAME path int8 AND/OR uses (`compile_predicate_program` emits `CompareScalarI64` for
            // each `Int8Literal` bound the DML builder produced; `resident_device_int_column_offset`
            // resolves the timestamp column to the i64 section). Gate LOCALLY on every column being an
            // i64 section (Int8/Timestamp) — NOT via `predicate_vm_elem_type`, which the nullable-read
            // branch keys off (folding Timestamp in there would divert nullable-timestamp 3VL reads).
            // A MIXED timestamp/other-type predicate fails this check → clean error, never a silent
            // mis-answer at one element width. (`compile_predicate_program`'s TIMESTAMP leaf accepts
            // both an `Int8Literal` (raw micros — the DML builder) AND a `TextLiteral` bound (parsed
            // via `timestamp_literal_micros` — the read path); an unparseable literal errors cleanly,
            // so it can never mis-read a text needle against the i64 column.)
            let mut cols = Vec::new();
            collect_expr_columns(predicate, &mut cols);
            let all_i64_section = !cols.is_empty()
                && cols.iter().all(|&col| {
                    matches!(
                        table.columns.get(col).map(|column| column.ty),
                        Some(SqlType::Int8 | SqlType::Timestamp)
                    )
                });
            if all_i64_section {
                let mut program = Vec::new();
                let mut needles: Vec<Vec<u8>> = Vec::new();
                compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
                return device_memory
                    .run_expr_predicate_filter(&program, row_count, ResidentElemType::I64)
                    .map(Some)
                    .map_err(map_err);
            }
            // ADR-006 (MIXED-WIDTH groups): a mixed ts+{int4,text,bool,date,uuid} AND/OR whose
            // i64-section leaves are all WIDTH-SAFE scalar comparisons (the timestamp leaf emits
            // `LoadColumnI64` regardless of the program elem) falls through to the general And/Or
            // I32 branch instead of erroring.
            if mixed_width_i32_elem(predicate, table).is_some() {
                return Ok(None);
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports timestamp AND/OR only over a timestamp-only \
                 (i64) predicate or a mixed group of width-safe scalar leaves; timestamp \
                 arithmetic is a follow-on"
                    .to_string(),
            )));
        };
        match (
            timestamp_column_index(lhs, table),
            timestamp_column_index(rhs, table),
        ) {
            (Some(col), None) => {
                let micros = timestamp_literal_micros(rhs)?;
                let offset = resident_device_int8_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_i64_compare_scalar_filter(offset, micros, false, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (None, Some(col)) => {
                let micros = timestamp_literal_micros(lhs)?;
                let offset = resident_device_int8_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_i64_compare_scalar_filter(offset, micros, true, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_int8_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_int8_column_offset(snapshot, table, b)?;
                device_memory
                    .expr_i64_compare_columns_filter(a_offset, b_offset, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a timestamp predicate must involve a timestamp column".to_string(),
            ))),
        }
    }

    /// M3 (doc 21) WHERE 3VL for a SIMPLE comparison over a nullable DATE or TIMESTAMP column (scalar,
    /// either operand order, or col-vs-col). The generic mask VM (`predicate_vm_elem_type`) cannot lower
    /// these: a date/timestamp literal is not an `Int4Literal`, and a timestamp's i64 microseconds exceed
    /// the VM's i32 `CompareScalar` scalar. So build the VM program directly — DATE: I32 + `CompareScalar`
    /// over the i32 days; TIMESTAMP: I64 + `CompareScalarI64` over the i64 micros; col-vs-col:
    /// `CompareBuffers` — then append the validity-AND (`push_column_validity_and` = `BoolMask` +
    /// `MaskBinary` AND) for each nullable operand, so a NULL operand is UNKNOWN ⇒ the row is excluded
    /// (correct at the top level / under no-NOT). Returns `None` for a non-date/timestamp predicate or a
    /// shape the peephole doesn't support (AND/OR / arithmetic), so the caller clean-errors. NO kernel
    /// change: the i64 compare-scalar / compare-buffers / bool-to-mask / mask-binary kernels already exist.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_nullable_temporal_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        // Only a simple comparison (eq/ne/lt/le/gt/ge); AND/OR yield None -> caller clean-errors.
        let Some(cmp) = predicate_compare_code(compare) else {
            return Ok(None);
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // DATE: i32 days in the int4 section -> the I32 VM (CompareScalar fits the i32 days literal).
        if expr_mentions_date(lhs, table) || expr_mentions_date(rhs, table) {
            let mut program = Vec::new();
            match (date_column_index(lhs, table), date_column_index(rhs, table)) {
                (Some(col), None) => {
                    let days = date_literal_days(rhs)?;
                    let off = resident_device_int4_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalar {
                        cmp,
                        scalar: days,
                        scalar_on_left: false,
                    });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (None, Some(col)) => {
                    let days = date_literal_days(lhs)?;
                    let off = resident_device_int4_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalar {
                        cmp,
                        scalar: days,
                        scalar_on_left: true,
                    });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (Some(a), Some(b)) => {
                    let a_off = resident_device_int4_column_offset(snapshot, table, a)?;
                    let b_off = resident_device_int4_column_offset(snapshot, table, b)?;
                    program.push(ExprStep::LoadColumn { byte_offset: a_off });
                    program.push(ExprStep::LoadColumn { byte_offset: b_off });
                    program.push(ExprStep::CompareBuffers { cmp });
                    push_column_validity_and(a, table, snapshot, &mut program)?;
                    push_column_validity_and(b, table, snapshot, &mut program)?;
                }
                (None, None) => return Ok(None),
            }
            return device_memory
                .run_expr_predicate_filter(&program, row_count, ResidentElemType::I32)
                .map(Some)
                .map_err(map_err);
        }
        // TIMESTAMP: i64 micros in the int8 section -> the I64 VM with CompareScalarI64 (the micros
        // literal exceeds the i32 CompareScalar scalar).
        if expr_mentions_timestamp(lhs, table) || expr_mentions_timestamp(rhs, table) {
            let mut program = Vec::new();
            match (
                timestamp_column_index(lhs, table),
                timestamp_column_index(rhs, table),
            ) {
                (Some(col), None) => {
                    let micros = timestamp_literal_micros(rhs)?;
                    let off = resident_device_int8_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalarI64 {
                        cmp,
                        scalar: micros,
                        scalar_on_left: false,
                    });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (None, Some(col)) => {
                    let micros = timestamp_literal_micros(lhs)?;
                    let off = resident_device_int8_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalarI64 {
                        cmp,
                        scalar: micros,
                        scalar_on_left: true,
                    });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (Some(a), Some(b)) => {
                    let a_off = resident_device_int8_column_offset(snapshot, table, a)?;
                    let b_off = resident_device_int8_column_offset(snapshot, table, b)?;
                    program.push(ExprStep::LoadColumn { byte_offset: a_off });
                    program.push(ExprStep::LoadColumn { byte_offset: b_off });
                    program.push(ExprStep::CompareBuffers { cmp });
                    push_column_validity_and(a, table, snapshot, &mut program)?;
                    push_column_validity_and(b, table, snapshot, &mut program)?;
                }
                (None, None) => return Ok(None),
            }
            return device_memory
                .run_expr_predicate_filter(&program, row_count, ResidentElemType::I64)
                .map(Some)
                .map_err(map_err);
        }
        Ok(None)
    }

    /// M3 (doc 21) WHERE 3VL over a nullable NUMERIC column. The generic mask VM can't lower numeric (a
    /// numeric literal is not an Int4Literal; the i128 mantissa exceeds the i32 CompareScalar). Two routes,
    /// both on the I128 VM, each AND'ing the operand columns' validity (a NULL operand is UNKNOWN, excluded
    /// — correct for WHERE under AND/OR since there is no NOT). (a) A SIMPLE same-or-coarser-scale scalar /
    /// same-scale col-vs-col uses a direct `CompareScalarI128` / `CompareBuffers` program (handles a LARGE
    /// mantissa the i32-needle path can't). (b) AND/OR, numeric arithmetic, a FINER cross-scale literal, or
    /// a different-scale col-vs-col uses the shared `compile_numeric_predicate_program` /
    /// `compile_numeric_compare` VM path, now validity-aware (`push_leaf_validity_and` per leaf; same
    /// i32-needle limit as the non-null path). Returns `None` only for a non-numeric predicate; a mixed
    /// numeric/integer predicate is a clean error. NO kernel change.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_nullable_numeric_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_numeric(lhs, table) || expr_mentions_numeric(rhs, table)) {
            return Ok(None);
        }
        // Mixed numeric/integer is a clean error (compile_numeric_arith expects numeric operands) — mirror
        // the non-null path's guard so the message is clear rather than an opaque compile failure.
        if expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int8(lhs, table)
            || expr_mentions_int8(rhs, table)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed numeric/integer expressions yet"
                    .to_string(),
            )));
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let run = |program: &[ExprStep]| {
            device_memory
                .run_expr_predicate_filter(program, row_count, ResidentElemType::I128)
                .map(Some)
                .map_err(map_err)
        };
        // numeric AND/OR -> the validity-aware compile path (each comparison leaf via compile_numeric_compare,
        // which now AND's the operand validity; MaskBinary combines).
        if let Some(bool_op) = boolean_op_code(compare) {
            let mut program = Vec::new();
            compile_numeric_predicate_program(lhs, table, snapshot, &mut program)?;
            compile_numeric_predicate_program(rhs, table, snapshot, &mut program)?;
            program.push(ExprStep::MaskBinary { op: bool_op });
            return run(&program);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            return Ok(None);
        };
        // SIMPLE same-or-coarser-scale scalar / same-scale col-vs-col -> the CompareScalarI128 fast path
        // (handles a large mantissa). `None` here means "not this fast shape" -> fall to compile_numeric_compare.
        let scalar_program = |col: usize,
                              literal: Decimal128,
                              scalar_on_left: bool|
         -> Result<Option<Vec<ExprStep>>, ExecuteError> {
            let col_scale = column_numeric_scale(table, col).expect("numeric column has a scale");
            let literal = literal.canonical();
            if literal.scale > col_scale {
                return Ok(None); // finer cross-scale literal -> the compile_numeric_compare fallback
            }
            let mantissa = rescale_numeric_literal(literal, col_scale)?;
            let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
            let mut program = vec![
                ExprStep::LoadColumn {
                    byte_offset: offset,
                },
                ExprStep::CompareScalarI128 {
                    cmp,
                    scalar: mantissa,
                    scalar_on_left,
                },
            ];
            push_column_validity_and(col, table, snapshot, &mut program)?;
            Ok(Some(program))
        };
        let fast: Option<Vec<ExprStep>> = match (
            numeric_column_index(lhs, table),
            numeric_column_index(rhs, table),
        ) {
            (Some(col), None) if numeric_literal_value(rhs).is_some() => {
                scalar_program(col, numeric_literal_value(rhs).expect("checked"), false)?
            }
            (None, Some(col)) if numeric_literal_value(lhs).is_some() => {
                scalar_program(col, numeric_literal_value(lhs).expect("checked"), true)?
            }
            (Some(a), Some(b))
                if column_numeric_scale(table, a) == column_numeric_scale(table, b) =>
            {
                let a_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                let mut program = vec![
                    ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    },
                    ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    },
                    ExprStep::CompareBuffers { cmp },
                ];
                push_column_validity_and(a, table, snapshot, &mut program)?;
                push_column_validity_and(b, table, snapshot, &mut program)?;
                Some(program)
            }
            _ => None,
        };
        if let Some(program) = fast {
            return run(&program);
        }
        // Fallback: a FINER cross-scale literal / different-scale col-vs-col / numeric arithmetic -> the
        // validity-aware shared compile path (compile_numeric_compare appends push_leaf_validity_and).
        let mut program = Vec::new();
        compile_numeric_compare(lhs, rhs, cmp, table, snapshot, &mut program)?;
        run(&program)
    }

    /// Try to lower a SIMPLE uuid comparison (`id <cmp> 'uuid-literal'`, either operand order, or
    /// `id <cmp> id2`) to surviving row indices via the byte-wise uuid compare kernel (the type matrix,
    /// doc 19): a `uuid` is 16 raw bytes in the i128 section; PG compares uuids by an unsigned
    /// big-endian memcmp. The string literal is parsed to 16 bytes at lowering. Returns None for a
    /// non-uuid predicate. Uuid `AND`/`OR`, and a uuid compared to a non-uuid value, are hard errors.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_uuid_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table)) {
            return Ok(None);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            // AND/OR (ADR-006): fall through — the And/Or branch of `lower_resident_predicate` runs
            // `compile_predicate_program`, whose uuid leaf compiles to a `UuidCmpMask` mask-VM step
            // (uuid IN / uuid ranges / mixed uuid+int4/text). Never a silent mis-answer: a shape the
            // leaf compiler can't express errors there and the caller declines.
            return Ok(None);
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // M3 (doc 21): a NULLABLE uuid operand's validity bitmap offset (None when the column holds no
        // NULL); the launcher AND's it with the compare mask so a NULL operand is UNKNOWN ⇒ excluded. A
        // non-nullable column contributes nothing (the no-NULL path stays byte-identical).
        let validity = |col: usize| -> Result<Vec<u64>, ExecuteError> {
            Ok(resident_device_null_column_offset(snapshot, table, col)?
                .into_iter()
                .collect())
        };
        match (uuid_column_index(lhs, table), uuid_column_index(rhs, table)) {
            (Some(col), None) => {
                let needle = uuid_literal_bytes(rhs)?;
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_uuid_compare_scalar_filter(
                        offset,
                        &needle,
                        false,
                        cmp,
                        row_count,
                        &validity(col)?,
                    )
                    .map(Some)
                    .map_err(map_err)
            }
            (None, Some(col)) => {
                let needle = uuid_literal_bytes(lhs)?;
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_uuid_compare_scalar_filter(
                        offset,
                        &needle,
                        true,
                        cmp,
                        row_count,
                        &validity(col)?,
                    )
                    .map(Some)
                    .map_err(map_err)
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                let mut validity_offsets = validity(a)?;
                validity_offsets.extend(validity(b)?);
                device_memory
                    .expr_uuid_compare_columns_filter(
                        a_offset,
                        b_offset,
                        cmp,
                        row_count,
                        &validity_offsets,
                    )
                    .map(Some)
                    .map_err(map_err)
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a uuid predicate must involve a uuid column".to_string(),
            ))),
        }
    }

    /// Try to lower a SIMPLE smallint comparison (`sz <cmp> 5`, either operand order, or `sz <cmp>
    /// sz2`) to surviving row indices via the i32 compare path (the type matrix, doc 19): a `smallint`
    /// is stored widened to i32 in the int4 section, so it REUSES the int4 compare VM with the literal
    /// taken as an i32 scalar. Returns None for a non-smallint predicate. Smallint `AND`/`OR` /
    /// arithmetic (int16-bounds overflow is a follow-on), and a smallint compared to a non-integer
    /// value, are hard errors -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_int2_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_int2(lhs, table) || expr_mentions_int2(rhs, table)) {
            return Ok(None);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            // ADR-006 (MIXED-WIDTH groups): int2 AND/OR falls THROUGH to the general And/Or I32
            // branch — an int2 leaf is a 4-byte `LoadColumn` + `CompareScalar` in an I32 program
            // (the int4 section; `predicate_vm_elem_type` has always classed Int2 as I32), so
            // all-int2/int4 groups and int8+int2 mixes (the audit's MEDIUM: this error used to
            // block the int2 arm of the mixed fall-through) compile there. Non-boolean
            // non-comparison shapes keep the hard error.
            if boolean_op_code(compare).is_some() {
                return Ok(None);
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only simple smallint comparisons (smallint \
                 arithmetic is a follow-on)"
                    .to_string(),
            )));
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let run = |program: &[ExprStep]| {
            device_memory
                .run_expr_predicate_filter(program, row_count, ResidentElemType::I32)
                .map(Some)
                .map_err(map_err)
        };
        match (int2_column_index(lhs, table), int2_column_index(rhs, table)) {
            (Some(col), None) => {
                let scalar = int2_literal_value(rhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar,
                        scalar_on_left: false,
                    },
                ])
            }
            (None, Some(col)) => {
                let scalar = int2_literal_value(lhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar,
                        scalar_on_left: true,
                    },
                ])
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_int4_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_int4_column_offset(snapshot, table, b)?;
                run(&[
                    ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    },
                    ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    },
                    ExprStep::CompareBuffers { cmp },
                ])
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a smallint predicate must involve a smallint column".to_string(),
            ))),
        }
    }

    /// Try to lower a SIMPLE text comparison (`textcol = 'lit'` / `textcol <> 'lit'`, either operand
    /// order) to surviving row indices via the byte-wise text-equality kernel (the type matrix, doc
    /// 19). Equality is byte identity -- PG deterministic-collation semantics. Returns None for a
    /// non-text predicate (the other type paths handle it). Inequalities, LIKE, and col-vs-col
    /// (ADR-006) lower via their kernels below; text `AND`/`OR` falls through to the mask VM; text
    /// mixed with another type in ONE comparison is a hard error -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_text_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_text(lhs, table) || expr_mentions_text(rhs, table)) {
            return Ok(None);
        }
        // AND/OR is not a single comparison: fall through (Ok(None)) so the mask VM compiles each leaf
        // (text -> TextEqMask, int4 -> arith+compare) and combines them. A text+int4 AND/OR is VALID
        // even though mixing the two types within ONE comparison (the guard below) is not. Mixed
        // text+int8/numeric AND/OR is rejected later (the int8/numeric paths) -- the i32 mask VM only.
        if matches!(compare, ResidentBinaryOp::And | ResidentBinaryOp::Or) {
            return Ok(None);
        }
        if expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int8(lhs, table)
            || expr_mentions_int8(rhs, table)
            || expr_mentions_numeric(lhs, table)
            || expr_mentions_numeric(rhs, table)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed text/non-text expressions"
                    .to_string(),
            )));
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // COL-VS-COL single comparison (ADR-006): BOTH sides are text columns — run the per-row
        // two-column compare kernel via a one-leaf mask program (`compile_text_eq_leaf`'s
        // col-vs-col arm, which ANDs BOTH validity masks — either operand NULL is UNKNOWN). All
        // six ops; `a LIKE b` (col-vs-col LIKE) errors cleanly inside the leaf. The AND/OR shapes
        // already composed via the fall-through above.
        if text_column_index(lhs, table).is_some() && text_column_index(rhs, table).is_some() {
            let mut program = Vec::new();
            let mut needles: Vec<Vec<u8>> = Vec::new();
            compile_text_eq_leaf(
                compare,
                lhs,
                rhs,
                table,
                snapshot,
                &mut program,
                &mut needles,
            )?;
            return device_memory
                .run_expr_predicate_filter_with_text(
                    &program,
                    &needles,
                    row_count,
                    ResidentElemType::I32,
                )
                .map(Some)
                .map_err(map_err);
        }
        // textcol LIKE 'pattern' (the pattern is on the right; LIKE is NOT symmetric). The host
        // compiles the pattern (resolving `\` escapes) to the kernel's u32 token array.
        if matches!(compare, ResidentBinaryOp::Like) {
            let (col, pattern) = match (text_column_index(lhs, table), text_literal_value(rhs)) {
                (Some(col), Some(pattern)) => (col, pattern),
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "LIKE must be a text column LIKE a literal pattern".to_string(),
                    )));
                }
            };
            let tokens = compile_like_pattern(pattern)?;
            let layout = resident_device_text_column_layout(snapshot, table, col)?;
            return device_memory
                .expr_text_like_scalar_filter(
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    layout.bytes_len,
                    &tokens,
                    row_count,
                )
                .map(Some)
                .map_err(map_err);
        }
        // textcol <lt/le/gt/ge> 'literal' (ADR-006): LEXICOGRAPHIC unsigned byte compare on-device
        // (memcmp of the common prefix; shorter sorts first) — byte-identical to the host
        // `compare_sql_values` Text order (Rust `str::cmp`) that the DML recheck uses, so device ==
        // recheck. A column-on-RIGHT (`'lit' < col`) flips `scalar_on_left`; a nullable text column
        // AND's its validity mask (a NULL operand is UNKNOWN ⇒ excluded).
        if matches!(
            compare,
            ResidentBinaryOp::Lt
                | ResidentBinaryOp::Le
                | ResidentBinaryOp::Gt
                | ResidentBinaryOp::Ge
        ) {
            let (col, needle, scalar_on_left) =
                match (text_column_index(lhs, table), text_column_index(rhs, table)) {
                    (Some(col), None) if text_literal_value(rhs).is_some() => {
                        (col, text_literal_value(rhs).expect("checked"), false)
                    }
                    (None, Some(col)) if text_literal_value(lhs).is_some() => {
                        (col, text_literal_value(lhs).expect("checked"), true)
                    }
                    _ => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "text comparison must be a column against a literal".to_string(),
                        )));
                    }
                };
            let cmp = predicate_compare_code(compare).expect("lt/le/gt/ge have compare codes");
            let layout = resident_device_text_column_layout(snapshot, table, col)?;
            let validity: Vec<u64> = resident_device_null_column_offset(snapshot, table, col)?
                .into_iter()
                .collect();
            return device_memory
                .expr_text_compare_scalar_filter(
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    layout.bytes_len,
                    needle.as_bytes(),
                    scalar_on_left,
                    cmp,
                    row_count,
                    &validity,
                )
                .map(Some)
                .map_err(map_err);
        }
        let negate = match compare {
            ResidentBinaryOp::Eq => false,
            ResidentBinaryOp::Ne => true,
            ResidentBinaryOp::Lt
            | ResidentBinaryOp::Le
            | ResidentBinaryOp::Gt
            | ResidentBinaryOp::Ge => {
                // Handled by the inequality block above; unreachable here.
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "text inequality reached the eq/ne arm unexpectedly".to_string(),
                )));
            }
            // AND/OR returned early (above) -> the mask VM. Any other op here is unsupported.
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports text = and <> only (LIKE is a follow-on)"
                        .to_string(),
                )));
            }
        };
        // textcol <eq/ne> 'literal' -- equality is symmetric, so operand order does not matter.
        let (col, literal) = match (text_column_index(lhs, table), text_column_index(rhs, table)) {
            (Some(col), None) if text_literal_value(rhs).is_some() => {
                (col, text_literal_value(rhs).expect("checked"))
            }
            (None, Some(col)) if text_literal_value(lhs).is_some() => {
                (col, text_literal_value(lhs).expect("checked"))
            }
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "text comparison must be a column against a literal".to_string(),
                )));
            }
        };
        let layout = resident_device_text_column_layout(snapshot, table, col)?;
        device_memory
            .expr_text_eq_scalar_filter(
                layout.offsets_byte_offset,
                layout.bytes_byte_offset,
                layout.bytes_len,
                literal.as_bytes(),
                negate,
                row_count,
            )
            .map(Some)
            .map_err(map_err)
    }

    /// Try to lower a SIMPLE numeric comparison to surviving row indices via the i128 compare kernels
    /// (the type matrix, doc 19). Supported: `numcol <cmp> literal` / `literal <cmp> numcol` and
    /// `numcol <cmp> numcol`, at ANY scale -- same-scale (and coarser-or-equal literal) use the
    /// resident compare peephole; a SCALE MISMATCH (a finer literal, or two columns of different scale)
    /// loads to i128 buffers, rescales the coarser side UP to the common = max scale (mantissa * 10^k,
    /// k <= 9; a wider gap or a rescale overflow errors), then compares. Numeric ARITHMETIC comparisons
    /// and `AND`/`OR` of numeric comparisons also lower here (each comparison -> a mask via the i128
    /// VM, MaskBinary combines). Returns None for a non-numeric predicate (the int4/int8 paths handle
    /// it). A numeric value mixed with an int4/int8 COLUMN is a hard error -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_numeric_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_numeric(lhs, table) || expr_mentions_numeric(rhs, table)) {
            return Ok(None);
        }
        if expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int8(lhs, table)
            || expr_mentions_int8(rhs, table)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed numeric/integer expressions yet"
                    .to_string(),
            )));
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // numeric AND/OR: compile each side's comparison(s) to a MASK (the buffer path), combine with
        // MaskBinary, and compact -- the same i128 mask VM the single-comparison arithmetic path uses.
        if let Some(bool_op) = boolean_op_code(compare) {
            let mut program = Vec::new();
            compile_numeric_predicate_program(lhs, table, snapshot, &mut program)?;
            compile_numeric_predicate_program(rhs, table, snapshot, &mut program)?;
            program.push(ExprStep::MaskBinary { op: bool_op });
            return device_memory
                .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
                .map(Some)
                .map_err(map_err);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only numeric comparisons and AND/OR".to_string(),
            )));
        };
        match (
            numeric_column_index(lhs, table),
            numeric_column_index(rhs, table),
        ) {
            (Some(col), None) if numeric_literal_value(rhs).is_some() => {
                let col_scale =
                    column_numeric_scale(table, col).expect("numeric column has a scale");
                let literal = numeric_literal_value(rhs).expect("checked").canonical();
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                if literal.scale <= col_scale {
                    // literal coarser-or-equal: rescale it UP to the column scale; resident peephole.
                    let mantissa = rescale_numeric_literal(literal, col_scale)?;
                    device_memory
                        .expr_i128_compare_scalar_filter(offset, mantissa, false, cmp, row_count)
                        .map(Some)
                        .map_err(map_err)
                } else {
                    // CROSS-SCALE: the literal is FINER than the column -> rescale the column UP to the
                    // literal's scale (buffer VM path) and compare to the literal's mantissa.
                    self.numeric_cross_scale_scalar(
                        offset,
                        col_scale,
                        literal,
                        false,
                        cmp,
                        device_memory,
                        row_count,
                    )
                    .map(Some)
                }
            }
            (None, Some(col)) if numeric_literal_value(lhs).is_some() => {
                let col_scale =
                    column_numeric_scale(table, col).expect("numeric column has a scale");
                let literal = numeric_literal_value(lhs).expect("checked").canonical();
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                if literal.scale <= col_scale {
                    let mantissa = rescale_numeric_literal(literal, col_scale)?;
                    device_memory
                        .expr_i128_compare_scalar_filter(offset, mantissa, true, cmp, row_count)
                        .map(Some)
                        .map_err(map_err)
                } else {
                    self.numeric_cross_scale_scalar(
                        offset,
                        col_scale,
                        literal,
                        true,
                        cmp,
                        device_memory,
                        row_count,
                    )
                    .map(Some)
                }
            }
            (Some(a), Some(b)) => {
                let a_scale = column_numeric_scale(table, a).expect("numeric column has a scale");
                let b_scale = column_numeric_scale(table, b).expect("numeric column has a scale");
                let a_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                if a_scale == b_scale {
                    device_memory
                        .expr_i128_compare_columns_filter(a_offset, b_offset, cmp, row_count)
                        .map(Some)
                        .map_err(map_err)
                } else {
                    // CROSS-SCALE: load both, rescale the coarser column UP to the common (max) scale,
                    // then compare buffers (the buffer VM path).
                    let common = a_scale.max(b_scale);
                    let mut program = vec![ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    }];
                    push_numeric_rescale(a_scale, common, &mut program)?;
                    program.push(ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    });
                    push_numeric_rescale(b_scale, common, &mut program)?;
                    program.push(ExprStep::CompareBuffers { cmp });
                    device_memory
                        .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
                        .map(Some)
                        .map_err(map_err)
                }
            }
            _ => {
                // Numeric ARITHMETIC comparison (`price * tax > 100`, `100 < price - fee`, ...):
                // compile both sides to a mask via the shared comparison core (each side's scale is
                // computed bottom-up and the coarser brought up to the common scale), then run the VM.
                let mut program = Vec::new();
                compile_numeric_compare(lhs, rhs, cmp, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
                    .map(Some)
                    .map_err(map_err)
            }
        }
    }

    /// Lower a predicate [`ResidentExpr`] to the surviving row indices, evaluated on the GPU.
    ///
    /// Coverage (grows by extending this method, per the design): the prototype shape
    /// `Compare(arith(Column a, Column b), Int4Literal k)` lowers to the composed buffer->buffer
    /// facade `expr_filter_two_col_compare_from_payload` (typed column loads and `a <arith> b` into
    /// an intermediate device buffer, then ordered compare-to-row-indices). Anything else is rejected with a
    /// pointer to the device bytecode VM that generalizes it — NOT by falling back to a shape method
    /// or to the CPU.
    /// Lower a bare bool-column predicate (`WHERE flag`) to surviving row indices (the type matrix,
    /// doc 19): the bool column's 1-bit-per-row bitmap expands directly to the row mask. A bare
    /// non-bool column is invalid SQL (PG: "argument of WHERE must be type boolean") -> hard error.
    fn lower_bool_column_predicate(
        &self,
        col: usize,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        let column = table.columns.get(col).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr predicate column is outside the catalog table".to_string(),
            ))
        })?;
        if column.ty != SqlType::Bool {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a bare column predicate is valid only for a bool column (argument of WHERE must be \
                 type boolean)"
                    .to_string(),
            )));
        }
        let offset = resident_device_bool_column_offset(snapshot, table, col)?;
        device_memory
            .expr_bool_to_mask_filter(offset, false, row_count)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    /// Lower a standalone `col IS NULL` / `IS NOT NULL` to surviving row indices (M3 -- doc 21). The
    /// column's NULL validity bitmap (1 = valid/present) feeds the same bitmap->mask kernel as a bool
    /// column: `IS NOT NULL` reads it as-is, `IS NULL` complements it (`negate = !is_not_null`). A column
    /// with NO validity bitmap holds no NULLs, so every row is valid -- IS NOT NULL is all rows, IS NULL
    /// none -- returned directly without a kernel.
    fn lower_is_null_predicate(
        &self,
        col: usize,
        is_not_null: bool,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        match resident_device_null_column_offset(snapshot, table, col)? {
            Some(offset) => device_memory
                .expr_bool_to_mask_filter(offset, !is_not_null, row_count)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))),
            None if is_not_null => {
                let n = u32::try_from(row_count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident row count exceeds u32".to_string(),
                    ))
                })?;
                Ok((0..n).collect())
            }
            None => Ok(Vec::new()),
        }
    }

    /// Try to lower `bool_col = true` / `bool_col = false` (and `<>`, either operand order) to
    /// surviving row indices via the bitmap->mask kernel (the type matrix, doc 19): `= true` selects
    /// the set bits, `= false` (and `NOT flag`, which the mapper rewrites to `= false`) the clear bits
    /// (negate). Returns None if not a `bool_col <op> bool_literal` shape. Ordering ops (`< > ...`) and
    /// bool in AND/OR are hard errors (follow-ons) so the executor never mis-answers.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_bool_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        let is_bool_col =
            |idx: usize| table.columns.get(idx).map(|column| column.ty) == Some(SqlType::Bool);
        let (col, literal, column_on_left) = match (lhs, rhs) {
            (ResidentExpr::Column(col), ResidentExpr::BoolLiteral(b)) if is_bool_col(*col) => {
                (*col, *b, true)
            }
            (ResidentExpr::BoolLiteral(b), ResidentExpr::Column(col)) if is_bool_col(*col) => {
                (*col, *b, false)
            }
            _ => return Ok(None),
        };
        // ADR-006 (bool inequalities): the same constant-fold as `compile_bool_leaf` — PG orders
        // `false < true`, so `<`/`<=`/`>`/`>=` against a two-valued literal reduces to an
        // equality mask or a constant verdict. This peephole serves NON-NULL columns only (a
        // nullable-bool predicate routes through the VM block above), so constant-TRUE is ALL
        // rows — there is no UNKNOWN to exclude.
        let effective = if column_on_left {
            compare
        } else {
            match compare {
                ResidentBinaryOp::Lt => ResidentBinaryOp::Gt,
                ResidentBinaryOp::Le => ResidentBinaryOp::Ge,
                ResidentBinaryOp::Gt => ResidentBinaryOp::Lt,
                ResidentBinaryOp::Ge => ResidentBinaryOp::Le,
                other => other,
            }
        };
        let needle = match effective {
            ResidentBinaryOp::Eq => literal,
            ResidentBinaryOp::Ne => !literal,
            ResidentBinaryOp::Lt if literal => false,
            ResidentBinaryOp::Le if !literal => false,
            ResidentBinaryOp::Gt if !literal => true,
            ResidentBinaryOp::Ge if literal => true,
            ResidentBinaryOp::Lt | ResidentBinaryOp::Gt => {
                return Ok(Some(Vec::new())); // col < false / col > true: nothing qualifies
            }
            ResidentBinaryOp::Le | ResidentBinaryOp::Ge => {
                // col <= true / col >= false over a NON-NULL column: every row qualifies.
                let n = u32::try_from(row_count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "bool predicate row count exceeds u32".to_string(),
                    ))
                })?;
                return Ok(Some((0..n).collect()));
            }
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports only comparisons against a bool literal"
                        .to_string(),
                )));
            }
        };
        // `col == true` -> set bits (negate=false); `col == false` -> clear bits (negate=true).
        let offset = resident_device_bool_column_offset(snapshot, table, col)?;
        device_memory
            .expr_bool_to_mask_filter(offset, !needle, row_count)
            .map(Some)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    pub(crate) fn lower_resident_predicate(
        &self,
        predicate: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
        // SV3b/SV6 (MVCC visibility): `Some` on a VERSIONED shard — AND the on-device visibility mask
        // (`deleted_by > read_txn_id`, and/or `created_by <= read_txn_id`) onto the WHERE survivors.
        // `None` (version-free/cold shard, the majority) keeps the peephole fast paths below byte-identically.
        visibility: Option<ResidentVisibility>,
    ) -> Result<Vec<u32>, ExecuteError> {
        // SV3b: a versioned shard forces the mask VM (bypassing the typed peephole kernels) so the WHERE is a
        // program we can AND the i64 visibility mask onto — via the mixed-width VM (int4 WHERE at elem=I32 +
        // an i64 `deleted_by`/`created_by` compare). Only versioned shards reach here, so the peephole fast
        // paths below stay the common-case default.
        if let Some(vis) = visibility {
            // A predicate the VM can't lower at one elem width (mixed non-int8 widths / an unsupported shape)
            // can't compose the i64 visibility conjunct, so this HARD-ERRORS rather than risk leaking
            // tombstoned rows. NB: this is NOT the CPU fallback (that fires only on residency invalidation);
            // it surfaces as a query error. Inert until DELETE tombstoning is wired (SV4) -- at which point
            // extending visibility to these shapes (or routing them to the CPU-pinned path) is the follow-up.
            // ADR-006 (MIXED-WIDTH groups): a mixed int8/ts + i32-servable WHERE with width-safe
            // scalar i64 leaves composes with the (i64) visibility conjuncts at I32 — the same
            // LoadColumnI64-self-consumed contract the conjuncts themselves use. LOCAL fallback,
            // not a widening of the shared `predicate_vm_elem_type`.
            let elem = predicate_vm_elem_type(predicate, table)
                .or_else(|| mixed_width_i32_elem(predicate, table))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident visibility filter: this WHERE predicate is not VM-lowerable (mixed width / \
                         unsupported shape) so the MVCC visibility conjunct cannot be composed"
                            .to_string(),
                    ))
                })?;
            let mut program: Vec<gpu_db_execution::ExprStep> = Vec::new();
            let mut needles: Vec<Vec<u8>> = Vec::new();
            compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
            // AND the visibility conjunct(s) onto the WHERE mask. Each `CompareScalarI64` immediately
            // consumes its own `LoadColumnI64` buffer (the mixed-width VM caller contract).
            vis.push_conjuncts(&mut program, true);
            return device_memory
                .run_expr_predicate_filter_with_text(&program, &needles, row_count, elem)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())));
        }
        // bool-predicate (type matrix, doc 19): a bare `WHERE flag` is a bool COLUMN used directly as
        // a predicate -- a bool column is a 1-bit-per-row bitmap, so it expands straight to the row
        // mask (true rows). A bare NON-bool column is invalid SQL (PG: "argument of WHERE must be type
        // boolean"), so it hard-errors rather than mis-answering.
        if let ResidentExpr::Column(col) = predicate {
            return self.lower_bool_column_predicate(
                *col,
                table,
                snapshot,
                device_memory,
                row_count,
            );
        }

        // A standalone `col IS NULL` / `IS NOT NULL` (M3 -- doc 19/21): its NULL validity bitmap straight
        // to the row mask via the SAME bitmap->mask kernel as a bool column, pointed at the validity
        // bitmap. (Inside AND/OR it instead rides the general mask VM via `compile_predicate_program`.)
        if let ResidentExpr::IsNull { col, is_not_null } = predicate {
            return self.lower_is_null_predicate(
                *col,
                *is_not_null,
                table,
                snapshot,
                device_memory,
                row_count,
            );
        }

        // M3 (doc 21) WHERE 3VL: a predicate over a NULLABLE column routes through the general mask VM
        // (`compile_predicate_program`), which AND's each comparison leaf with the column's validity mask
        // so a NULL operand evaluates to UNKNOWN (the row is excluded). The typed peephole kernels below
        // read the column's placeholder bytes (0 / empty) and would mis-select NULL rows. The VM lowers an
        // int4/text/bool predicate (elem I32) or an int8 predicate (elem I64, the same i64 buffer VM the
        // non-null int8 AND/OR path uses — the validity AND is a type-independent i32 mask); a nullable
        // numeric/uuid/date/timestamp/int2 or MIXED-width predicate is a clean-error follow-up (no silent
        // mis-answer). A no-NULL table never enters here, so existing (non-nullable) plans keep their
        // peephole fast paths byte-identically.
        if predicate_references_nullable_column(predicate, table, snapshot)? {
            if let Some(elem) = predicate_vm_elem_type(predicate, table) {
                let mut program = Vec::new();
                let mut needles: Vec<Vec<u8>> = Vec::new();
                compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
                return device_memory
                    .run_expr_predicate_filter_with_text(&program, &needles, row_count, elem)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    });
            }
            // A nullable DATE/TIMESTAMP/NUMERIC simple comparison: the generic VM can't lower the literal
            // (temporal/numeric), so build the program directly (date I32 / timestamp+numeric I128/I64)
            // with the validity-AND appended.
            if let ResidentExpr::Binary { op, lhs, rhs } = predicate {
                if let Some(indices) = self.try_lower_nullable_temporal_predicate(
                    *op,
                    lhs,
                    rhs,
                    table,
                    snapshot,
                    device_memory,
                    row_count,
                )? {
                    return Ok(indices);
                }
                if let Some(indices) = self.try_lower_nullable_numeric_predicate(
                    *op,
                    lhs,
                    rhs,
                    table,
                    snapshot,
                    device_memory,
                    row_count,
                )? {
                    return Ok(indices);
                }
                // A nullable uuid SIMPLE comparison: the uuid memcmp peephole resolves the operand's
                // validity bitmap and AND's it with the compare mask (uuid has no VM step). Runs BEFORE
                // the uuid AND/OR block below so col-vs-col and every simple shape keep this proven path.
                if expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table) {
                    if let Some(indices) = self.try_lower_uuid_predicate(
                        *op,
                        lhs,
                        rhs,
                        table,
                        snapshot,
                        device_memory,
                        row_count,
                    )? {
                        return Ok(indices);
                    }
                }
                // NULLABLE uuid / timestamp AND/OR (ADR-006): uuid + timestamp are not in
                // `predicate_vm_elem_type`'s sets (deliberately — widening the SHARED helper diverts
                // working paths; the timestamp lesson), so a nullable uuid range/IN or a nullable
                // timestamp range missed the VM block above and used to hit the final error. Gate
                // LOCALLY by column set: {Uuid, Int4, Int2, Text, Bool} runs at I32 (uuid/text/bool
                // leaves are mask-only, int4/int2 load 4 bytes); {Int8, Timestamp} runs at I64 (a
                // timestamp is i64 micros in the int8 section; the `Int8Literal` leaf arm loads 8
                // bytes + CompareScalarI64). ZERO DIVERSION: all-non-uuid I32 combos and all-Int8
                // took the elem-Some block above, so these gates fire only with a uuid / timestamp
                // column present — shapes that previously ERRORED. Each leaf appends its own
                // validity-AND (3VL), exactly how the elem-Some VM block serves nullable int4/text.
                if matches!(op, ResidentBinaryOp::And | ResidentBinaryOp::Or) {
                    let mut cols = Vec::new();
                    collect_expr_columns(predicate, &mut cols);
                    let ty = |col: usize| table.columns.get(col).map(|column| column.ty);
                    let local_elem = if cols.is_empty() {
                        None
                    } else if cols.iter().all(|&col| {
                        // Date joins the I32 set (ADR-006 date compound): a date is i32 days in the
                        // int4 section, and its VM leaf emits a 4-byte LoadColumn — I32-only.
                        matches!(
                            ty(col),
                            Some(
                                SqlType::Uuid
                                    | SqlType::Int4
                                    | SqlType::Int2
                                    | SqlType::Text
                                    | SqlType::Bool
                                    | SqlType::Date
                            )
                        )
                    }) {
                        Some(ResidentElemType::I32)
                    } else if cols
                        .iter()
                        .all(|&col| matches!(ty(col), Some(SqlType::Int8 | SqlType::Timestamp)))
                    {
                        Some(ResidentElemType::I64)
                    } else {
                        // ADR-006 (MIXED-WIDTH groups): a nullable mixed int8/ts + i32-servable
                        // group runs at I32 when every i64 leaf is a width-safe scalar comparison.
                        // ZERO DIVERSION: the two branches above already served every all-I32-set
                        // and all-i64-section combination — this fires only for mixed sets, which
                        // previously fell to the final error. Each leaf appends its own
                        // validity-AND (3VL), including the int8 scalar arm.
                        mixed_width_i32_elem(predicate, table)
                    };
                    if let Some(elem) = local_elem {
                        let mut program = Vec::new();
                        let mut needles: Vec<Vec<u8>> = Vec::new();
                        compile_predicate_program(
                            predicate,
                            table,
                            snapshot,
                            &mut program,
                            &mut needles,
                        )?;
                        return device_memory
                            .run_expr_predicate_filter_with_text(
                                &program, &needles, row_count, elem,
                            )
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            });
                    }
                }
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "WHERE over a nullable column of this type/shape is not yet supported on the GPU \
                 (M3 3VL follow-up: e.g. a cross-scale or arithmetic numeric, a compound or \
                 mixed-type temporal/numeric/uuid predicate)"
                    .to_string(),
            )));
        }

        let ResidentExpr::Binary {
            op: compare,
            lhs,
            rhs,
        } = predicate
        else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr predicate must be a top-level comparison or AND/OR".to_string(),
            )));
        };

        // bool path (type matrix, doc 19): `flag = true` / `flag = false` / `<>` (and `NOT flag`,
        // which the mapper rewrites to `flag = false`) expand the bool bitmap to the mask via negate.
        // None for a non-bool-literal predicate; checked first since a BoolLiteral has no other home.
        if let Some(indices) = self.try_lower_bool_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // int8 path (type matrix, doc 19): a SIMPLE int8 comparison (`int8col <cmp> literal` /
        // `literal <cmp> int8col` / `int8col <cmp> int8col`) evaluates via the i64 compare kernels,
        // before the int4 paths below. `try_lower_int8_comparison` returns None for a pure-int4
        // predicate (fall through to the int4 path) and errors for an int8 shape not yet supported
        // (int8 arithmetic / mixed int4-int8) so the executor never silently mis-answers.
        if let Some(indices) = self.try_lower_int8_predicate(
            predicate,
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // numeric path (type matrix, doc 19): a SIMPLE numeric comparison (`numcol <cmp> literal` /
        // `literal <cmp> numcol` / equal-scale `numcol <cmp> numcol`) evaluates via the i128 compare
        // kernels. Returns None for a non-numeric predicate (fall through to int4); errors on a numeric
        // shape not yet supported (numeric arithmetic / AND-OR / mixed) so it never mis-answers.
        if let Some(indices) = self.try_lower_numeric_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // date path (type matrix, doc 19): a SIMPLE date comparison (`hire_date <cmp> '2024-01-15'`)
        // evaluates via the i32 compare path (a date is i32 days). BEFORE the text path, because the
        // date literal is a string literal the text path would otherwise grab. None for a non-date
        // predicate; errors on date AND/OR / arithmetic / mixed.
        if let Some(indices) = self.try_lower_date_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // timestamp path (type matrix, doc 19): a SIMPLE timestamp comparison (i64 microseconds ->
        // the int8 compare kernels). BEFORE the text path (the literal is a string). None for a
        // non-timestamp predicate; errors on timestamp AND/OR / arithmetic / mixed.
        if let Some(indices) = self.try_lower_timestamp_predicate(
            predicate,
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // uuid path (type matrix, doc 19): a SIMPLE uuid comparison via the byte-wise compare kernel.
        // BEFORE the text path (the literal is a string). None for a non-uuid predicate; errors on
        // uuid AND/OR / mixed.
        if let Some(indices) = self.try_lower_uuid_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // int2 path (type matrix, doc 19): a SIMPLE smallint comparison via the i32 compare VM
        // (a smallint is stored widened to i32). None for a non-smallint predicate; errors on
        // smallint AND/OR / arithmetic / a non-integer comparand.
        if let Some(indices) = self.try_lower_int2_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // text path (type matrix, doc 19): a SIMPLE text comparison (`textcol = 'lit'` / `<>`)
        // evaluates via the byte-wise text-equality kernel. Returns None for a non-text predicate
        // (fall through to int4); errors on a text shape not yet supported (LIKE / inequalities /
        // AND-OR / mixed) so it never silently mis-answers.
        if let Some(indices) = self.try_lower_text_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // Boolean combinators (AND/OR) and `Ne` lower to the general mask-based predicate VM (each
        // comparison -> a mask, MaskBinary combines, the terminal compacts). The specialized 2-col /
        // arith / col-vs-col lowerings below handle the single eq/lt/le/gt/ge comparisons.
        if matches!(
            compare,
            ResidentBinaryOp::And | ResidentBinaryOp::Or | ResidentBinaryOp::Ne
        ) {
            let mut program = Vec::new();
            let mut needles: Vec<Vec<u8>> = Vec::new();
            compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
            return device_memory
                .run_expr_predicate_filter_with_text(
                    &program,
                    &needles,
                    row_count,
                    ResidentElemType::I32,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())));
        }

        let Some(comparison) = compare_op_code(*compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr predicate requires a comparison op (eq/lt/le/gt/ge/ne) or AND/OR"
                    .to_string(),
            )));
        };

        // Specialized facade for the exact `Compare(arith(Column, Column), Int4Literal)` shape. The
        // execution crate constructs the same typed postfix VM program used by general expressions,
        // then emits ascending indices through ordered compaction; this keeps one stable call surface
        // without a raw-offset shape kernel.
        if let (
            ResidentExpr::Binary {
                op: arith,
                lhs: a,
                rhs: b,
            },
            ResidentExpr::Int4Literal(needle),
        ) = (lhs.as_ref(), rhs.as_ref())
        {
            if let (Some(op_code), ResidentExpr::Column(col_a), ResidentExpr::Column(col_b)) =
                (arith_op_code(*arith), a.as_ref(), b.as_ref())
            {
                let a_offset = resident_device_int4_column_offset(snapshot, table, *col_a)?;
                let b_offset = resident_device_int4_column_offset(snapshot, table, *col_b)?;
                return device_memory
                    .expr_filter_two_col_compare_from_payload(
                        a_offset, b_offset, op_code, row_count, *needle, comparison,
                    )
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    });
            }
        }

        // PEEPHOLE fast-path (the `category = 0` shape): the exact predicate `int4col <cmp> literal`
        // (and the flipped `literal <cmp> int4col`, comparison flipped) lowers to the ORDERED parallel
        // compaction that emits surviving ROW INDICES ascending with NO host sort — replacing the
        // general `run_expr_arith_filter` path, which now uses the same ordered compactor. The
        // ordered compaction guarantees ascending output BY CONSTRUCTION (the contiguous block
        // partition + ordered intra-block prefix sum), which is exactly the contract the gather +
        // assembly depend on, so the result is byte-identical to the prior path. Only a PLAIN resident
        // int4 column qualifies: nullable columns / date / int2 / int8 / other types are routed by the
        // earlier type paths and never reach here, but the `SqlType::Int4` guard makes that explicit so
        // a future reordering can't silently feed a date/int2 column (same i32 section) into this path.
        let plain_int4_column = |expr: &ResidentExpr| -> Option<usize> {
            if let ResidentExpr::Column(col) = expr {
                if table.columns.get(*col).map(|column| column.ty) == Some(SqlType::Int4) {
                    return Some(*col);
                }
            }
            None
        };
        let ordered_indices = match (lhs.as_ref(), rhs.as_ref()) {
            (col_expr, ResidentExpr::Int4Literal(needle)) => {
                plain_int4_column(col_expr).map(|col| (col, *needle, comparison))
            }
            (ResidentExpr::Int4Literal(needle), col_expr) => plain_int4_column(col_expr)
                .map(|col| (col, *needle, flip_comparison_code(comparison))),
            _ => None,
        };
        if let Some((col, needle, cmp_code)) = ordered_indices {
            let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
            return device_memory
                .compare_indices_ordered_from_payload(byte_offset, row_count, needle, cmp_code)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())));
        }

        // GENERAL path: compile the arithmetic side(s) to bytecode and run the device VM.
        //   - `arith_tree <cmp> literal`        -> arith VM (one value buffer vs scalar)
        //   - `literal <cmp> arith_tree`        -> arith VM, comparison flipped
        //   - `arith_tree <cmp> arith_tree`     -> compile both, col-vs-col / expr-vs-expr VM
        // (AND/OR over masks is the next slice.)
        match (lhs.as_ref(), rhs.as_ref()) {
            (value, ResidentExpr::Int4Literal(needle)) => {
                let mut program = Vec::new();
                compile_arith_program(value, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_arith_filter(&program, row_count, comparison, *needle)
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
            }
            (ResidentExpr::Int4Literal(needle), value) => {
                let mut program = Vec::new();
                compile_arith_program(value, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_arith_filter(
                        &program,
                        row_count,
                        flip_comparison_code(comparison),
                        *needle,
                    )
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
            }
            (lhs_expr, rhs_expr) => {
                let mut program = Vec::new();
                compile_arith_program(lhs_expr, table, snapshot, &mut program)?;
                compile_arith_program(rhs_expr, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_compare_buffers_filter(&program, row_count, comparison)
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
            }
        }
    }
}
