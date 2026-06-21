//! General GPU executor — the `Expr` IR and its device interpreter (Charter rule 2;
//! `docs/architecture/17-general-gpu-executor.md`). This is the GENERAL path that replaces the
//! enumerated `execute_relational_*_with_resident_device_memory_probe` shape methods: a query is an
//! expression tree, lowered to a pipeline of device primitives, NOT matched against a fixed catalog
//! of shapes.
//!
//! `ResidentExpr` is the general scalar IR (it can represent any int4 arithmetic/comparison/boolean
//! tree). The interpreter's *coverage* grows node-by-node; today `lower_resident_predicate` lowers
//! the prototype shape `Compare(arith(Column, Column), Int4Literal)` to the composed buffer->buffer
//! primitive (`expr_filter_two_col_compare_from_payload`) and materializes the projected int4
//! columns by gathering the surviving rows. Fuller trees (deeper arithmetic, AND/OR over masks,
//! column-vs-column compares) land as the device bytecode VM in §2.3 of the design doc — by
//! extending this interpreter, never by adding a new shape method.
//!
//! The IR + op-code maps + `execute_resident_expr_select_with_binding` are now the production path the
//! SQL->Expr binding (`engine_sql_pg`) routes into; the GPU parity tests exercise the same lowering
//! via programmatic `ResidentExpr`s. The 2-arg `execute_resident_expr_select` convenience wrapper (run
//! a select with a programmatic predicate, binding internally) has no production caller yet — a
//! prepared-route / facade caller is the likely one — so it carries a targeted `allow(dead_code)`.

use super::*;

/// A binary operator in the resident expression IR (arithmetic, comparison, or boolean). The
/// interpreter pattern-matches these; callers (tests now, the parser/planner later) construct them.
/// Not all variants have interpreter coverage yet (the device bytecode VM, design §2.3, adds
/// AND/OR/Ne/col-vs-col).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentBinaryOp {
    Add,
    Sub,
    Mul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    /// SQL `LIKE`: the lhs is a text column, the rhs a [`ResidentExpr::TextLiteral`] pattern (`%` =
    /// any run, `_` = any one character). Byte-wise / UTF-8-char-aware (the type matrix, doc 19).
    Like,
}

/// The general scalar-expression IR the GPU interpreter evaluates. Column fields are table column
/// indices; the interpreter resolves device byte-offsets. Grows by node (literals of other types,
/// casts, unary ops, `LIKE`, ...), never by enumerating whole query shapes.
#[derive(Debug, Clone)]
pub(crate) enum ResidentExpr {
    Column(usize),
    Int4Literal(i32),
    /// A numeric (DECIMAL) literal as its [`Decimal128`] (mantissa + scale). Compared by rescaling to
    /// the target column's scale at lowering time (the type matrix, doc 19). An integer literal
    /// compared to a numeric column arrives as `Int4Literal` and is coerced to numeric on that path.
    NumericLiteral(Decimal128),
    /// A text (`text`/`varchar`) literal -- the comparison/LIKE value for a text column. Compared
    /// byte-wise (deterministic-collation equality is byte-identity; the type matrix, doc 19).
    TextLiteral(String),
    /// A boolean literal (`true`/`false`) -- the comparison value for `flag = true` / `flag = false`
    /// (the type matrix, doc 19). A bool column is a bitmap, so the comparison lowers to the
    /// bitmap->mask kernel with the appropriate `negate`.
    BoolLiteral(bool),
    Binary {
        op: ResidentBinaryOp,
        lhs: Box<ResidentExpr>,
        rhs: Box<ResidentExpr>,
    },
}

/// A column reference inside a JOIN (ON condition or projection): a bare column (`qualifier: None`) or
/// a qualified `alias.column` (`qualifier: Some(alias)`). Resolved to a specific relation + column
/// index in the join executor (against the two bound tables).
#[derive(Debug, Clone)]
pub(crate) struct JoinColRef {
    pub qualifier: Option<String>,
    pub column: String,
}

/// A 2-relation INNER equi-join parsed from the libpg_query FROM clause (M5). The probe/build choice is
/// made in the executor (build on the smaller relation). `on_a`/`on_b` are the two ON operands (one
/// resolves to the left relation, one to the right); `projection` is the SELECT column list (each a
/// qualified/unqualified column from either relation). A separate path from the single-table `Select`,
/// so the single-relation executor + the legacy parser stay untouched.
#[derive(Debug, Clone)]
pub(crate) struct JoinPlan {
    pub left_table: String,
    pub left_alias: String,
    pub right_table: String,
    pub right_alias: String,
    pub on_a: JoinColRef,
    pub on_b: JoinColRef,
    pub projection: Vec<JoinColRef>,
}

/// Narrow a grouped MIN/MAX or GROUP BY key, which the GPU kernel computes as an i64 (or, for
/// numeric, an i128 split into `lo`/`hi`), back to the column's own `SqlType`. int2/int4/date ride the
/// 4-byte read; int8/timestamp the 8-byte read; numeric reconstructs `hi:lo` at `scale`. `hi`/`scale`
/// are ignored for the non-numeric types.
fn narrow_ordered_value(ty: SqlType, lo: i64, hi: i64, scale: u8) -> SqlValue {
    match ty {
        SqlType::Numeric { .. } => {
            SqlValue::Numeric(Decimal128::new((i128::from(hi) << 64) | i128::from(lo as u64), scale))
        }
        SqlType::Int8 => SqlValue::Int8(lo),
        SqlType::Timestamp => SqlValue::Timestamp(lo),
        SqlType::Int2 => SqlValue::Int2(lo as i16),
        SqlType::Date => SqlValue::Date(lo as i32),
        // bool key / bool MIN/MAX value: the derived int4 column is 0/1 -> Bool.
        SqlType::Bool => SqlValue::Bool(lo != 0),
        _ => SqlValue::Int4(lo as i32),
    }
}

/// Device op-code for an arithmetic binary op (matches `expr_proto.ptx`: 0=add, 1=sub, 2=mul), or
/// `None` if `op` is not arithmetic.
fn arith_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::Add => Some(0),
        ResidentBinaryOp::Sub => Some(1),
        ResidentBinaryOp::Mul => Some(2),
        _ => None,
    }
}

/// Device comparison code for a comparison binary op (matches `expr_proto.ptx`: 0=eq, 1=lt, 2=le,
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
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
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
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
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
        | ResidentExpr::TextLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
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
        | ResidentExpr::NumericLiteral(_)
        | ResidentExpr::BoolLiteral(_) => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_text(lhs, table) || expr_mentions_text(rhs, table)
        }
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
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_date(lhs, table) || expr_mentions_date(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
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
        ResidentExpr::TextLiteral(text) => gpu_db_sql::datetime::parse_date(text).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "invalid input syntax for type date: \"{text}\""
            )))
        }),
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
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_timestamp(lhs, table) || expr_mentions_timestamp(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
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
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
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
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_int2(lhs, table) || expr_mentions_int2(rhs, table)
        }
        ResidentExpr::Int4Literal(_)
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
        ResidentExpr::TextLiteral(_) | ResidentExpr::BoolLiteral(_) => {
            Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a text/bool literal is not a numeric arithmetic operand".to_string(),
            )))
        }
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
                        let value = if lhs_lit.is_none() { lhs.as_ref() } else { rhs.as_ref() };
                        let value_scale = compile_numeric_arith(value, table, snapshot, program)?;
                        let canonical = literal.canonical();
                        program.push(ExprStep::ScalarBinary {
                            op: 2,
                            scalar: scalar_i32(canonical.mantissa)?,
                            scalar_on_left: false, // multiply is commutative
                        });
                        value_scale.checked_add(canonical.scale).ok_or_else(scale_overflow)
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
    let needle = |literal: Decimal128, scale: u8| -> Result<i32, ExecuteError> {
        let mantissa = rescale_numeric_literal(literal, scale)?;
        i32::try_from(mantissa).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric comparison literal is too large for the fast path yet".to_string(),
            ))
        })
    };
    match (numeric_literal_value(lhs), numeric_literal_value(rhs)) {
        (Some(literal), None) => {
            // literal <cmp> arith: bring both to the common = max scale.
            let arith_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
            let common = arith_scale.max(literal.canonical().scale);
            push_numeric_rescale(arith_scale, common, program)?;
            program.push(ExprStep::CompareScalar {
                cmp,
                scalar: needle(literal, common)?,
                scalar_on_left: true,
            });
            Ok(())
        }
        (None, Some(literal)) => {
            let arith_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
            let common = arith_scale.max(literal.canonical().scale);
            push_numeric_rescale(arith_scale, common, program)?;
            program.push(ExprStep::CompareScalar {
                cmp,
                scalar: needle(literal, common)?,
                scalar_on_left: false,
            });
            Ok(())
        }
        (None, None) => {
            // arith <cmp> arith: bring both sides to the common = max scale.
            let common = numeric_arith_scale(lhs, table)?.max(numeric_arith_scale(rhs, table)?);
            let lhs_scale = compile_numeric_arith(lhs, table, snapshot, program)?;
            push_numeric_rescale(lhs_scale, common, program)?;
            let rhs_scale = compile_numeric_arith(rhs, table, snapshot, program)?;
            push_numeric_rescale(rhs_scale, common, program)?;
            program.push(ExprStep::CompareBuffers { cmp });
            Ok(())
        }
        (Some(_), Some(_)) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "the general GPU executor does not evaluate literal-only numeric comparisons".to_string(),
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

/// Compile a boolean predicate expression into postfix [`ExprStep`] bytecode that leaves one MASK on
/// the VM stack. Recurses: `AND`/`OR` compile both operand predicates then a `MaskBinary`; a
/// comparison compiles its arithmetic operand(s) (via [`compile_arith_program`]) then a
/// `CompareScalar`/`CompareBuffers` mask step. The predicate VM compacts the final mask to indices.
fn compile_predicate_program(
    expr: &ResidentExpr,
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
    let ResidentExpr::Binary { op, lhs, rhs } = expr else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident predicate must be a comparison or boolean (AND/OR) combination".to_string(),
        )));
    };
    if let Some(bool_op) = boolean_op_code(*op) {
        compile_predicate_program(lhs, table, snapshot, program)?;
        compile_predicate_program(rhs, table, snapshot, program)?;
        program.push(ExprStep::MaskBinary { op: bool_op });
        return Ok(());
    }
    let Some(cmp) = predicate_compare_code(*op) else {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident predicate node must be a comparison (eq/ne/lt/le/gt/ge) or AND/OR".to_string(),
        )));
    };
    match (lhs.as_ref(), rhs.as_ref()) {
        (value, ResidentExpr::Int4Literal(scalar)) if !is_int4_literal(value) => {
            compile_arith_program(value, table, snapshot, program)?;
            program.push(ExprStep::CompareScalar {
                cmp,
                scalar: *scalar,
                scalar_on_left: false,
            });
            Ok(())
        }
        (ResidentExpr::Int4Literal(scalar), value) if !is_int4_literal(value) => {
            compile_arith_program(value, table, snapshot, program)?;
            program.push(ExprStep::CompareScalar {
                cmp,
                scalar: *scalar,
                scalar_on_left: true,
            });
            Ok(())
        }
        (lhs_expr, rhs_expr) if !is_int4_literal(lhs_expr) && !is_int4_literal(rhs_expr) => {
            compile_arith_program(lhs_expr, table, snapshot, program)?;
            compile_arith_program(rhs_expr, table, snapshot, program)?;
            program.push(ExprStep::CompareBuffers { cmp });
            Ok(())
        }
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident predicate comparison cannot be literal-vs-literal (constant-folding pending)"
                .to_string(),
        ))),
    }
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
            bound,
            copin_s,
            Some(predicate),
            &[],
            None,
            &[],
        )
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
            .group_by_i32_count_sum_bench(key_off, val_off, &indices, two_level)
            .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))
    }

    /// Benchmark helper (tests only): time JUST the GROUP BY kernel (CUDA events, min of `runs`),
    /// isolating it from the alloc/H2D/D2H/host-compact overhead. Returns the min kernel milliseconds.
    #[cfg(test)]
    pub(crate) fn group_by_i32_bench_kernel_ms(
        &self,
        table_name: &str,
        key_col: &str,
        value_col: &str,
        two_level: bool,
        runs: u32,
        rows_limit: usize,
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
            .group_by_i32_count_sum_kernel_timed(key_off, val_off, &indices, two_level, runs)
            .map(|(_, ms)| ms)
            .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))
    }

    /// As [`Engine::execute_resident_expr_select`] but over an ALREADY-BOUND table/projection: the
    /// caller bound the catalog once and resolved the predicate's `Column` indices against this SAME
    /// `table`. The SQL->Expr entry (`engine_sql_pg`) routes through here so the predicate's column
    /// indices, the projection, and the residency snapshot all derive from one catalog generation — a
    /// concurrent shape-changing DDL cannot split the column resolution from the execution.
    /// Execute a 2-relation INNER equi-join (M5) on the GPU. Binds both relations at ONE catalog
    /// generation, projects the two int-key columns, runs the GPU hash join (build on whichever side
    /// has a UNIQUE key; N:N is a follow-up), and marshals the matched rows' projected columns from
    /// host_rows. The join MATCH is on the GPU (the hash build+probe); only the matched index pairs +
    /// the key columns cross to the host -- the same control-plane gather the single-table text
    /// projection uses. No CPU relational join (charter).
    pub(crate) fn execute_resident_expr_inner_join(
        &self,
        plan: &JoinPlan,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // Bind BOTH relations at one catalog generation (co-pinned, like the single-table bind).
        let s = self.committed_seq();
        let catalog = self.read_state.catalog_as_of(s);
        let get_table = |name: &str| -> Result<RelationalTable, ExecuteError> {
            catalog
                .relational_catalog
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{name}\" does not exist"
                    )))
                })
        };
        let left_table = get_table(&plan.left_table)?;
        let right_table = get_table(&plan.right_table)?;
        #[derive(Clone, Copy, PartialEq)]
        enum Rel {
            Left,
            Right,
        }
        // Resolve a JOIN column reference to (relation, column index) against the two bound tables: a
        // qualifier must name exactly one relation; an unqualified column must be in exactly one (PG's
        // "ambiguous" / "does not exist").
        let resolve = |c: &JoinColRef| -> Result<(Rel, usize), ExecuteError> {
            match &c.qualifier {
                Some(q) if *q == plan.left_alias => {
                    Ok((Rel::Left, relational_column_index(&left_table, &c.column)?))
                }
                Some(q) if *q == plan.right_alias => {
                    Ok((Rel::Right, relational_column_index(&right_table, &c.column)?))
                }
                Some(q) => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "missing FROM-clause entry for table \"{q}\""
                )))),
                None => {
                    let l = relational_column_index(&left_table, &c.column).ok();
                    let r = relational_column_index(&right_table, &c.column).ok();
                    match (l, r) {
                        (Some(_), Some(_)) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            format!("column reference \"{}\" is ambiguous", c.column),
                        ))),
                        (Some(i), None) => Ok((Rel::Left, i)),
                        (None, Some(i)) => Ok((Rel::Right, i)),
                        (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "column \"{}\" does not exist", c.column
                        )))),
                    }
                }
            }
        };
        // ON: one operand resolves to each relation -> the per-relation key column indices.
        let a = resolve(&plan.on_a)?;
        let b = resolve(&plan.on_b)?;
        let (left_key_idx, right_key_idx) = match (a.0, b.0) {
            (Rel::Left, Rel::Right) => (a.1, b.1),
            (Rel::Right, Rel::Left) => (b.1, a.1),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the join ON must equate a column from EACH relation (a.k = b.k)".to_string(),
                )))
            }
        };
        // Slice scope: int4-section keys (int2/int4/date) on BOTH sides. int8/numeric/uuid/text keys
        // are a follow-up (J4).
        let int4_key = |t: SqlType| matches!(t, SqlType::Int4 | SqlType::Int2 | SqlType::Date);
        if !int4_key(left_table.columns[left_key_idx].ty)
            || !int4_key(right_table.columns[right_key_idx].ty)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a join key must be an int2/int4/date column on both sides on the join path yet \
                 (int8/numeric/uuid/text keys are a follow-up)"
                    .to_string(),
            )));
        }
        let proj: Vec<(Rel, usize)> = plan
            .projection
            .iter()
            .map(&resolve)
            .collect::<Result<_, _>>()?;
        // Pin both residency entries + their device memory (no CPU join fallback -- charter).
        let load = |name: &str| -> Result<(RelationalResidencyEntry, usize), ExecuteError> {
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
            Ok((entry, row_count))
        };
        let (left_entry, left_n) = load(&plan.left_table)?;
        let (right_entry, right_n) = load(&plan.right_table)?;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&plan.left_table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    plan.left_table
                )))
            })?;
        let right_dm = self
            .read_state
            .residency
            .device_memory
            .get(&plan.right_table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    plan.right_table
                )))
            })?;
        let gpu_id = left_entry.descriptor.gpu_id;
        // Project each relation's int key column to host i64 (int4 section, sign-extended).
        let key_i64 = |entry: &RelationalResidencyEntry,
                       table: &RelationalTable,
                       dm: &gpu_db_execution::CudaResidentDeviceMemory,
                       col_idx: usize,
                       n: usize|
         -> Result<Vec<i64>, ExecuteError> {
            if n == 0 {
                return Ok(Vec::new());
            }
            let off = resident_device_int4_column_offset(&entry.descriptor, table, col_idx)?;
            let idxs: Vec<u64> = (0..n as u64).collect();
            Ok(dm
                .project_i32_rows_from_payload(off, &idxs)
                .map_err(map_err)?
                .into_iter()
                .map(i64::from)
                .collect())
        };
        let left_keys = key_i64(&left_entry, &left_table, &device_memory, left_key_idx, left_n)?;
        let right_keys = key_i64(&right_entry, &right_table, &right_dm, right_key_idx, right_n)?;
        // GPU hash join: build on the SMALLER side; if its key is non-unique, retry building on the
        // OTHER side (covers 1:1 + 1:N regardless of size). Both non-unique => N:N (a follow-up).
        use gpu_db_execution::HashJoinOutcome;
        let smaller_is_left = left_n <= right_n;
        let (first_build, first_probe) = if smaller_is_left {
            (&left_keys, &right_keys)
        } else {
            (&right_keys, &left_keys)
        };
        let (left_is_build, build_idxs, probe_idxs) = match device_memory
            .hash_join_inner_i64(first_build, first_probe)
            .map_err(map_err)?
        {
            HashJoinOutcome::Pairs {
                build_idxs,
                probe_idxs,
            } => (smaller_is_left, build_idxs, probe_idxs),
            HashJoinOutcome::DuplicateBuildKey => {
                match device_memory
                    .hash_join_inner_i64(first_probe, first_build)
                    .map_err(map_err)?
                {
                    HashJoinOutcome::Pairs {
                        build_idxs,
                        probe_idxs,
                    } => (!smaller_is_left, build_idxs, probe_idxs),
                    HashJoinOutcome::DuplicateBuildKey => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "N:N join (both sides have duplicate join keys) is a follow-up; one \
                             side's join key must be unique"
                                .to_string(),
                        )))
                    }
                }
            }
        };
        // Map the (build_idx, probe_idx) pairs back to (left_idx, right_idx) per pair.
        let (left_match, right_match): (&[u32], &[u32]) = if left_is_build {
            (&build_idxs, &probe_idxs)
        } else {
            (&probe_idxs, &build_idxs)
        };
        // Marshal the matched rows: each projected column read from its relation's host_rows at the
        // matched absolute index (control-plane gather; the join itself ran on the GPU).
        let n_pairs = left_match.len();
        let mut rows: Vec<Vec<SqlValue>> = Vec::with_capacity(n_pairs);
        for i in 0..n_pairs {
            let mut row = Vec::with_capacity(proj.len());
            for &(rel, col_idx) in &proj {
                let cell = match rel {
                    Rel::Left => left_entry.host_rows[left_match[i] as usize][col_idx].clone(),
                    Rel::Right => right_entry.host_rows[right_match[i] as usize][col_idx].clone(),
                };
                row.push(cell);
            }
            rows.push(row);
        }
        // Result schema: each projected column's RelationalColumn from its owning table, attnums 1..N.
        let mut columns = Vec::with_capacity(proj.len());
        for (i, &(rel, col_idx)) in proj.iter().enumerate() {
            let src = match rel {
                Rel::Left => &left_table,
                Rel::Right => &right_table,
            };
            let mut col = src.columns[col_idx].clone();
            col.attnum = (i + 1) as i16;
            columns.push(col);
        }
        Ok(RelationalSelectResult {
            columns,
            rows,
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: RelationalAccessPath::FullTableScan,
        })
    }

    #[allow(clippy::too_many_arguments)] // group_key_expr is threaded alongside the predicate/binding
    pub(crate) fn execute_resident_expr_select_with_binding(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: BoundRelationalSelect,
        copin_s: Index,
        // `None` = no WHERE clause: a full-table scan (every row survives).
        predicate: Option<&ResidentExpr>,
        // Parallel to `select.order_by`: `Some(expr)` = a SORT EXPRESSION key (`ORDER BY a+b`),
        // evaluated on-device into an i64 key column; `None` = a plain column key. Empty = no ORDER BY.
        order_by_exprs: &[Option<ResidentExpr>],
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
        // Load the WHOLE residency entry (descriptor + host rows) from ONE atomic load() so a text
        // GROUP BY key -- whose result string is read from host_rows[representative_row] -- sees host
        // rows from the SAME generation as the descriptor / device memory the GPU grouped over.
        let residency_entry = self
            .relational_residency_entry(&table.name)
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
                "resident snapshot row count exceeds retained device-memory proof range".to_string(),
            ))
        })?;

        // Evaluate the predicate on the GPU -> surviving row indices (ascending). With no WHERE clause
        // every row survives, so the indices are the full 0..row_count scan (the aggregate + projection
        // paths below are index-driven and need no other change).
        let indices = match predicate {
            Some(predicate) => self.lower_resident_predicate(
                predicate,
                table,
                &snapshot,
                &device_memory,
                row_count,
            )?,
            None => {
                let n = u32::try_from(row_count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "full-table scan row count exceeds the u32 row-index range".to_string(),
                    ))
                })?;
                (0..n).collect()
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
         -> Result<Vec<gpu_db_execution::GroupByI32Row>, ExecuteError> {
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
                    let (matrix, k) =
                        if matches!(value_ty, SqlType::Numeric { .. } | SqlType::Uuid) {
                            let off = resident_device_numeric_column_offset(
                                &snapshot, table, value_idx,
                            )?;
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
                    let layout =
                        resident_device_text_column_layout(&snapshot, table, value_idx)?;
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
                        )
                        .map_err(map_err)?;
                    device_memory
                        .mark_new_distinct_text_device(
                            &perm,
                            idx_u64,
                            g_vals,
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
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
                    0,
                    0,
                    &scan_indices,
                    true,  // value_is_int8 (new_distinct is i64)
                    true,  // key_is_int8 (g_sorted is i64)
                    false, // value not numeric
                    false, // value not uuid
                    false, // key not i128
                    false, // key not text
                    0,
                    0,
                    false, // value not text
                    0,
                    0,
                    g_sorted.device_ptr(),
                    new_distinct.device_ptr(),
                    0, // comp_w (not a wide-key composite)
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
            // The GENERAL path (composite_is_widekey) handles every OTHER all-fixed composite -- >2
            // columns, a numeric/uuid member, or a tuple > 128 bits -- via a fixed-width WIDE KEY buffer
            // (gpu_db_build_wide_key) grouped by a (rep_idx, hash) b128 claim (memcmp the wide bytes);
            // the result reads each member from the representative row. A text member beyond the
            // 2-member (fixed, text) case (two-text, text in a >2 key) is a follow-up.
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
            let is_widekey_member =
                |t: SqlType| is_fixed_int(t) || matches!(t, SqlType::Numeric { .. } | SqlType::Uuid);
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
            // 2-member fast path: both fixed-int, OR exactly one text + one fixed-int.
            let composite_cols: Option<(usize, SqlType, usize, SqlType)> = match &composite_members {
                Some(m) if m.len() == 2 => {
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
            // General all-fixed WIDE KEY path: an N>=2 composite NOT taken by the 2-member fast path,
            // with EVERY member fixed-width (int/numeric/uuid -- no text).
            let widekey_cols: Option<Vec<(usize, SqlType)>> = match &composite_members {
                Some(m) if composite_cols.is_none() => {
                    if m.iter().all(|&(_, t)| is_widekey_member(t)) {
                        Some(m.clone())
                    } else {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "composite GROUP BY supports all-fixed-width members \
                             (int2/int4/int8/date/timestamp/numeric/uuid, any count) or a 2-member \
                             (fixed, text) key on the Expr path (a text member in a two-text / >2 \
                             composite is a follow-up)"
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
            let composite_is_text = composite_cols
                .is_some_and(|(_, t0, _, t1)| matches!(t0, SqlType::Text) != matches!(t1, SqlType::Text));
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
            let key_is_text = composite_is_text || (!is_expr_key && matches!(key_ty, SqlType::Text));
            // A bool key is materialized bool->int4 (0/1) into a derived buffer + grouped via
            // key_base_override (like an expression key); int4-width, never i128/text.
            let key_is_bool = !is_expr_key && matches!(key_ty, SqlType::Bool);
            let (key_offsets_off, key_bytes_off) = if composite_is_text {
                // The text member's varlen column (the other member rides key_base_override).
                let (c0, t0, c1, _) = composite_cols.unwrap();
                let text_col = if matches!(t0, SqlType::Text) { c0 } else { c1 };
                let layout = resident_device_text_column_layout(&snapshot, table, text_col)?;
                (layout.offsets_byte_offset, layout.bytes_byte_offset)
            } else if key_is_text {
                let layout = resident_device_text_column_layout(&snapshot, table, group_idx)?;
                (layout.offsets_byte_offset, layout.bytes_byte_offset)
            } else {
                (0, 0)
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
            // The wide-key width (bytes/row): each int member 8 bytes (i64), each numeric/uuid 16. 0 when
            // not a wide-key composite. Drives comp_w on the GROUP BY launch + the build descriptors.
            let widekey_w: u64 = widekey_cols.as_ref().map_or(0, |m| {
                m.iter()
                    .map(|&(_, t)| {
                        if matches!(t, SqlType::Numeric { .. } | SqlType::Uuid) {
                            16
                        } else {
                            8
                        }
                    })
                    .sum()
            });
            // GROUP BY <expression>: materialize the expr ONCE over all rows into a resident device key
            // buffer (checked overflow -> PG error) + hold it alive across EVERY pass; key_base_override
            // points the single-level kernel at it (it reads override + idx*stride, idx = the row id).
            // A column key passes 0 -> the byte-identical column path.
            let _derived_key_buf;
            let key_base_override: u64 = if let Some(expr) = group_key_expr {
                if row_count == 0 {
                    // Empty input: the on-device arith materialize rejects n=0, and there are no rows to
                    // group anyway. Skip it and group 0 rows -> 0 groups (PG returns no rows), matching
                    // the plain-column path's empty-table behavior. The override is never read (no rows).
                    _derived_key_buf = None;
                    0
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
                        .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))?;
                    let ptr = buf.device_ptr();
                    _derived_key_buf = Some(buf);
                    ptr
                }
            } else if key_is_bool {
                if row_count == 0 {
                    _derived_key_buf = None;
                    0
                } else {
                    // GROUP BY a bool column: materialize bool->int4 (0/1) into a derived buffer + group
                    // via key_base_override (the audited int4 path). No bool GROUP BY kernel -> no hazard.
                    let offset = resident_device_bool_column_offset(&snapshot, table, group_idx)?;
                    let buf = device_memory
                        .bool_to_int4_column_device(offset, row_count)
                        .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))?;
                    let ptr = buf.device_ptr();
                    _derived_key_buf = Some(buf);
                    ptr
                }
            } else if is_composite_key {
                if row_count == 0 {
                    _derived_key_buf = None;
                    0
                } else {
                    // GROUP BY a, b: pack the two members into one derived key grouped via
                    // key_base_override; the result UNPACKS it. Both-int4 -> one i64 ((col0<<32)|col1);
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
                    let ptr = buf.device_ptr();
                    _derived_key_buf = Some(buf);
                    ptr
                }
            } else if composite_is_widekey {
                if row_count == 0 {
                    _derived_key_buf = None;
                    0
                } else {
                    // General all-fixed composite: build the W-byte/row wide key (each member's canonical
                    // bytes concatenated -- int 8B, numeric/uuid 16B) -> grouped via the (rep_idx, hash)
                    // b128 claim (comp_w = widekey_w). The result reads each member from the rep row.
                    let members = widekey_cols.as_ref().expect("composite_is_widekey");
                    let mut descriptors: Vec<(u64, u64, u64)> = Vec::with_capacity(members.len());
                    let mut dst_off: u64 = 0;
                    for &(idx, ty) in members {
                        let (kind, src_off, w) = match ty {
                            SqlType::Numeric { .. } | SqlType::Uuid => (
                                2u64,
                                resident_device_numeric_column_offset(&snapshot, table, idx)?,
                                16u64,
                            ),
                            SqlType::Int8 | SqlType::Timestamp => (
                                1u64,
                                resident_device_int8_column_offset(&snapshot, table, idx)?,
                                8u64,
                            ),
                            _ => (
                                0u64,
                                resident_device_int4_column_offset(&snapshot, table, idx)?,
                                8u64,
                            ),
                        };
                        descriptors.push((kind, src_off, dst_off));
                        dst_off += w;
                    }
                    let buf = device_memory
                        .build_wide_key_device(&descriptors, widekey_w, row_count)
                        .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))?;
                    let ptr = buf.device_ptr();
                    _derived_key_buf = Some(buf);
                    ptr
                }
            } else {
                _derived_key_buf = None;
                0
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
            let force_single = value_indices.len() > 1 || is_expr_key;
            let run_pass =
                |value_idx_opt: Option<usize>, has_minmax: bool| -> Result<Pass, ExecuteError> {
                    // A None value column is the COUNT(*)-only pass: group over the key, read .count.
                    let value_idx = value_idx_opt.unwrap_or(group_idx);
                    let value_ty = table.columns[value_idx].ty;
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
                    let value_is_bool =
                        value_idx_opt.is_some() && matches!(value_ty, SqlType::Bool);
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
                    let _derived_value_buf;
                    let value_base_override: u64 = if value_is_bool && row_count > 0 {
                        let offset = resident_device_bool_column_offset(&snapshot, table, value_idx)?;
                        let buf = device_memory
                            .bool_to_int4_column_device(offset, row_count)
                            .map_err(map_err)?;
                        let ptr = buf.device_ptr();
                        _derived_value_buf = Some(buf);
                        ptr
                    } else {
                        _derived_value_buf = None;
                        0
                    };
                    let (value_offsets_off, value_bytes_off) = if value_is_text {
                        let layout =
                            resident_device_text_column_layout(&snapshot, table, value_idx)?;
                        (layout.offsets_byte_offset, layout.bytes_byte_offset)
                    } else {
                        (0, 0)
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
                        device_memory.group_by_i32_count_sum_minmax_from_payload(
                            key_offset,
                            value_offset,
                            &indices,
                            value_is_int8,
                            key_is_int8,
                            value_is_numeric,
                            value_is_uuid,
                            key_is_i128,
                            key_is_text,
                            key_offsets_off,
                            key_bytes_off,
                            value_is_text,
                            value_offsets_off,
                            value_bytes_off,
                            key_base_override,
                            value_base_override,
                            widekey_w,
                        )
                    } else {
                        device_memory.group_by_i32_count_sum_from_payload(
                            key_offset,
                            value_offset,
                            &indices,
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
                let mut passes = Vec::with_capacity(value_indices.len());
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
                if is_expr_key
                    || key_is_bool
                    || is_composite_key
                    || composite_is_widekey
                    || key_is_text
                    || key_is_i128
                {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "COUNT(DISTINCT v) supports a plain int group key on the GPU path \
                         (expression/bool/composite/text/numeric/uuid group keys are follow-ups)"
                            .to_string(),
                    )));
                }
                let materialize_i64_col =
                    |col_idx: usize, idx_u64: &[u64]| -> Result<Vec<i64>, ExecuteError> {
                        match table.columns[col_idx].ty {
                            SqlType::Int4 | SqlType::Int2 | SqlType::Date => Ok(device_memory
                                .project_i32_rows_from_payload(
                                    resident_device_int4_column_offset(&snapshot, table, col_idx)?,
                                    idx_u64,
                                )
                                .map_err(map_err)?
                                .into_iter()
                                .map(i64::from)
                                .collect()),
                            SqlType::Int8 | SqlType::Timestamp => device_memory
                                .project_i64_rows_from_payload(
                                    resident_device_int8_column_offset(&snapshot, table, col_idx)?,
                                    idx_u64,
                                )
                                .map_err(map_err),
                            _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "COUNT(DISTINCT) group key / value must be an i64-representable int \
                                 column"
                                    .to_string(),
                            ))),
                        }
                    };
                let idx_u64: Vec<u64> = indices.iter().map(|&i| u64::from(i)).collect();
                let n = idx_u64.len();
                // The group key column values (sorted-tuple key 0), materialized once for every pass.
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
            // A TEXT key/value's string -- and a wide-key composite's member values -- live host-side: the
            // kernel stored each group's representative ABSOLUTE row index; read it from the same
            // residency_entry generation as the GPU result.
            let any_text_value = passes.iter().any(|p| p.value_is_text);
            let text_host_rows = if key_is_text || any_text_value || composite_is_widekey {
                Some(residency_entry.host_rows.clone())
            } else {
                None
            };
            // The kernel's hash-slot / compaction order is RACE-dependent: the cas.b64 linear-probe
            // resolves bucket ownership differently per launch, so two passes do NOT share a group
            // order (proven: an int8 i64::MIN-key query misaligned only on the 6th launch). Re-sort
            // every pass by the MATERIALIZED group key so pass[k][i] is the same group across all
            // passes (and the merged output ends up key-ordered). A TEXT key must sort by the
            // materialized string -- there key_i128 is a per-pass representative row index, not the key.
            let materialize_key = |gk: &gpu_db_execution::GroupByI32Row| -> SqlValue {
                if composite_is_widekey {
                    // Wide-key: the b128 slot's lo = the representative row index. Return the FIRST
                    // member's value for the (single-pass) alignment sort; the result rows re-sort by the
                    // FULL member tuple, so this lone value need only be consistent + non-panicking.
                    let rep_idx = gk.key_i128 as u64 as usize;
                    let m0 = widekey_cols.as_ref().expect("composite_is_widekey")[0].0;
                    text_host_rows
                        .as_ref()
                        .expect("text_host_rows is Some when composite_is_widekey")[rep_idx][m0]
                        .clone()
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
                        let text_col = if matches!(ct0, SqlType::Text) { cc0 } else { cc1 };
                        text_host_rows
                            .as_ref()
                            .expect("text_host_rows is Some when composite_is_text")[rep_idx]
                            [text_col]
                            .clone()
                    } else if composite_is_i128 {
                        SqlValue::Numeric(Decimal128::new(gk.key_i128, 0))
                    } else {
                        SqlValue::Int8(gk.key)
                    }
                } else if key_is_text {
                    let rep_idx = gk.key_i128 as u64 as usize;
                    text_host_rows
                        .as_ref()
                        .expect("text_host_rows is Some when key_is_text")[rep_idx][group_idx]
                        .clone()
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
            let key_cmp = |a: &SqlValue, b: &SqlValue| -> std::cmp::Ordering {
                let i64_key = |v: &SqlValue| match v {
                    SqlValue::Int4(k) | SqlValue::Date(k) => i64::from(*k),
                    SqlValue::Int2(k) => i64::from(*k),
                    SqlValue::Int8(k) | SqlValue::Timestamp(k) => *k,
                    SqlValue::Bool(b) => i64::from(*b), // false(0) < true(1)
                    _ => i64::MIN,
                };
                match (a, b) {
                    (SqlValue::Numeric(x), SqlValue::Numeric(y)) => x.cmp(y),
                    (SqlValue::Uuid(x), SqlValue::Uuid(y)) => x.cmp(y),
                    (SqlValue::Text(x), SqlValue::Text(y)) => x.cmp(y),
                    _ => i64_key(a).cmp(&i64_key(b)),
                }
            };
            for pass in &mut passes {
                pass.groups
                    .sort_by(|a, b| key_cmp(&materialize_key(a), &materialize_key(b)));
            }
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
                let mut row: Vec<SqlValue> =
                    Vec::with_capacity(aggregates.len() + n_group_cols);
                if let Some(members) = &widekey_cols {
                    // Wide-key: the b128 slot's lo = the representative row index -- read EACH member from
                    // the rep row (host_rows), in DECLARED order. host_rows holds the typed SqlValue.
                    let rep_idx = gk.key_i128 as u64 as usize;
                    let host = text_host_rows
                        .as_ref()
                        .expect("text_host_rows is Some when composite_is_widekey");
                    for &(idx, _) in members {
                        row.push(host[rep_idx][idx].clone());
                    }
                } else if let Some((c0, t0, c1, t1)) = composite_cols {
                    if composite_is_text {
                        // (fixed, text): the b128 slot holds the rep row index -- read BOTH members from
                        // the rep row (host_rows), in DECLARED order, narrowed to their real types.
                        let rep_idx = gk.key_i128 as u64 as usize;
                        let host = text_host_rows
                            .as_ref()
                            .expect("text_host_rows is Some when composite_is_text");
                        row.push(host[rep_idx][c0].clone());
                        row.push(host[rep_idx][c1].clone());
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
                            // For int8/numeric the SUM is the i128 (sum_hi:sum); else sign-extend.
                            let sum_i128 = if pass.value_is_int8 || pass.value_is_numeric {
                                (i128::from(g.sum_hi) << 64) | i128::from(g.sum as u64)
                            } else {
                                i128::from(g.sum)
                            };
                            match aggregate.kind {
                                GroupedAggKind::Sum if pass.value_is_numeric => {
                                    SqlValue::Numeric(Decimal128::new(sum_i128, pass.value_scale))
                                }
                                GroupedAggKind::Sum if pass.value_is_int8 => {
                                    SqlValue::Numeric(Decimal128::new(sum_i128, 0))
                                }
                                GroupedAggKind::Sum => SqlValue::Int8(g.sum),
                                GroupedAggKind::Avg if pass.value_is_numeric => avg_numeric_sql_value(
                                    sum_i128,
                                    g.count as usize,
                                    pass.value_scale,
                                ),
                                GroupedAggKind::Avg => {
                                    average_sql_value(sum_i128, g.count as usize)
                                }
                                GroupedAggKind::Min if pass.value_is_uuid => {
                                    SqlValue::Uuid(g.min_uuid)
                                }
                                GroupedAggKind::Max if pass.value_is_uuid => {
                                    SqlValue::Uuid(g.max_uuid)
                                }
                                GroupedAggKind::Min if pass.value_is_text => text_host_rows
                                    .as_ref()
                                    .expect("text_host_rows is Some when value_is_text")
                                    [g.min as u64 as usize][pass.value_idx]
                                    .clone(),
                                GroupedAggKind::Max if pass.value_is_text => text_host_rows
                                    .as_ref()
                                    .expect("text_host_rows is Some when value_is_text")
                                    [g.max as u64 as usize][pass.value_idx]
                                    .clone(),
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
                    };
                    row.push(value);
                }
                rows.push(row);
            }
            // Deterministic default order. A single-key result is already key-sorted (the pass sort
            // above); a COMPOSITE result orders by the FULL tuple (both group columns) -- necessary for
            // a (fixed, text) composite whose per-pass order is rep-row (race) based, and natural for
            // packed composites too.
            if is_composite_key {
                rows.sort_by(|a, b| key_cmp(&a[0], &b[0]).then_with(|| key_cmp(&a[1], &b[1])));
            } else if let Some(members) = &widekey_cols {
                // Wide-key: order by the FULL member tuple (the per-pass order is rep-row/race based).
                let ncols = members.len();
                rows.sort_by(|a, b| {
                    for c in 0..ncols {
                        let ord = key_cmp(&a[c], &b[c]);
                        if ord != std::cmp::Ordering::Equal {
                            return ord;
                        }
                    }
                    std::cmp::Ordering::Equal
                });
            } else {
                rows.sort_by(|a, b| key_cmp(&a[0], &b[0]));
            }
            // Apply the grouped query's HAVING (filter), ORDER BY (re-sort), and LIMIT/OFFSET (slice)
            // HOST-SIDE over the materialized group rows, mapping each clause's referenced result column
            // name to its index. All four are empty/None for a bare GROUP BY, so this is a no-op there.
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
            if !select.having_groups.is_empty() {
                // Pre-resolve each filter's result-column index (fail fast on an unknown name); then
                // keep a row iff ANY OR-group's ANDed comparisons all hold.
                let resolved = select
                    .having_groups
                    .iter()
                    .map(|group| {
                        group
                            .iter()
                            .map(|f| col_index(&f.column).map(|idx| (idx, f)))
                            .collect::<Result<Vec<_>, ExecuteError>>()
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()?;
                rows.retain(|row| {
                    resolved.iter().any(|group| {
                        group
                            .iter()
                            .all(|(idx, f)| select_filter_matches(&row[*idx], f.op, &f.value))
                    })
                });
            }
            // The grouped result is sorted ON THE GPU (charter: every relational sort is a GPU sort,
            // regardless of result size -- no host-side finalization). INT keys feed an i64 matrix by
            // group position; TEXT/NUMERIC/UUID keys feed a resident-like payload built (via
            // build_relational_device_payload) from just those result columns, which the hetero
            // comparator reads on-device. Single- and multi-key are the same path (k=1 is K=1).
            if !select.order_by.is_empty() && rows.len() > 1 {
                let n = rows.len();
                // (result-column index, kind 0=int / 1=text / 2=numeric / 3=uuid) per ORDER BY key.
                let mut classified: Vec<(usize, u8)> = Vec::with_capacity(select.order_by.len());
                for order in &select.order_by {
                    let idx = col_index(&order.column)?;
                    let kind = match bound.selected_columns[idx].ty {
                        SqlType::Int4
                        | SqlType::Int8
                        | SqlType::Int2
                        | SqlType::Date
                        | SqlType::Timestamp => 0u8,
                        SqlType::Text => 1,
                        SqlType::Numeric { .. } => 2,
                        SqlType::Uuid => 3,
                        other => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "ORDER BY on a grouped {other:?} result is not yet supported"
                            ))));
                        }
                    };
                    classified.push((idx, kind));
                }
                let num_int = classified.iter().filter(|&&(_, k)| k == 0).count();
                // INT key matrix, row-major by group position (matches `indices` = 0..n order).
                let mut int_keys: Vec<i64> = Vec::with_capacity(n * num_int);
                for row in &rows {
                    for &(idx, kind) in &classified {
                        if kind == 0 {
                            int_keys.push(match row[idx] {
                                SqlValue::Int4(v) | SqlValue::Date(v) => i64::from(v),
                                SqlValue::Int2(v) => i64::from(v),
                                SqlValue::Int8(v) | SqlValue::Timestamp(v) => v,
                                _ => {
                                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                        "grouped ORDER BY int key encountered a non-int value"
                                            .to_string(),
                                    )));
                                }
                            });
                        }
                    }
                }
                let mut desc_mask = 0u64;
                for (ki, order) in select.order_by.iter().enumerate() {
                    if order.descending {
                        desc_mask |= 1u64 << ki;
                    }
                }
                let non_int: Vec<(usize, u8)> =
                    classified.iter().copied().filter(|&(_, k)| k != 0).collect();
                let map_sort_err = |e: gpu_db_execution::CudaRuntimeProbeError| {
                    ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
                };
                let perm: Vec<u32> = if non_int.is_empty() {
                    device_memory
                        .bitonic_sort_multikey(&int_keys, n, num_int, desc_mask)
                        .map_err(map_sort_err)?
                } else {
                    // A resident-like payload over ONLY the non-int key columns; the helper returns the
                    // text (offsets,bytes) + numeric/uuid section offsets we feed the hetero comparator.
                    let names: Vec<String> =
                        (0..non_int.len()).map(|i| format!("__gsk{i}")).collect();
                    let types: Vec<SqlType> = non_int
                        .iter()
                        .map(|&(idx, _)| bound.selected_columns[idx].ty)
                        .collect();
                    let payload_rows: Vec<Vec<SqlValue>> = rows
                        .iter()
                        .map(|r| non_int.iter().map(|&(idx, _)| r[idx].clone()).collect())
                        .collect();
                    let (payload, text_layouts, _bool, _int4, b128_layouts) =
                        crate::engine_residency::build_relational_device_payload(
                            &names,
                            &types,
                            &payload_rows,
                        )?;
                    // Walk ORDER BY order: int -> next matrix slot; text/numeric/uuid -> the next
                    // section in its type group (helper lays them out in passed-column order per group).
                    let mut int_slot = 0u32;
                    let mut text_idx = 0usize;
                    let mut b128_idx = 0usize;
                    let mut text_cols: Vec<(u64, u64)> = Vec::new();
                    let mut b128_cols: Vec<u64> = Vec::new();
                    let mut key_plan: Vec<u32> = Vec::with_capacity(select.order_by.len());
                    for &(_, kind) in &classified {
                        match kind {
                            0 => {
                                key_plan.push(int_slot);
                                int_slot += 1;
                            }
                            1 => {
                                let tl = &text_layouts[text_idx];
                                key_plan.push(0x4000_0000_u32 | text_cols.len() as u32);
                                text_cols.push((tl.offsets_byte_offset, tl.bytes_byte_offset));
                                text_idx += 1;
                            }
                            k => {
                                let off = b128_layouts[b128_idx].1;
                                let tag = if k == 2 { 0x8000_0000_u32 } else { 0xC000_0000_u32 };
                                key_plan.push(tag | b128_cols.len() as u32);
                                b128_cols.push(off);
                                b128_idx += 1;
                            }
                        }
                    }
                    let indices: Vec<u64> = (0..n as u64).collect();
                    device_memory
                        .bitonic_sort_hetero_on_payload(
                            &payload, &indices, &int_keys, num_int, &text_cols, &b128_cols,
                            &key_plan, desc_mask,
                        )
                        .map_err(map_sort_err)?
                };
                let reordered: Vec<Vec<SqlValue>> =
                    perm.iter().map(|&p| rows[p as usize].clone()).collect();
                rows = reordered;
            }
            if select.offset.is_some() || select.limit.is_some() {
                let start = select.offset.unwrap_or(0).min(rows.len());
                rows.drain(..start);
                if let Some(limit) = select.limit {
                    rows.truncate(limit);
                }
            }
            return Ok(RelationalSelectResult {
                columns: bound.selected_columns,
                rows,
                planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
                executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
                fallback_reason: None,
                access_path,
            });
        }

        // Scalar aggregate? Compute it from the filtered indices and return a single row.
        if is_aggregate {
            let value = match &select.projection {
                // COUNT(*): the surviving row count IS the result (PG returns bigint). The GPU filter
                // + compaction already produced the count; no per-row materialization.
                SelectProjection::CountAll => SqlValue::Int8(indices.len() as i64),
                // SUM(int4): a GPU reduction over the filtered column (gather col[indices] + reduce);
                // PG returns bigint. An EMPTY filtered set is SQL NULL, which the engine cannot
                // represent until M3 -- so it hard-errors rather than returning a wrong 0.
                SelectProjection::Sum { column } => {
                    let col_idx = relational_column_index(table, column)?;
                    if indices.is_empty() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "SUM over an empty set is NULL, which the engine cannot represent yet \
                             (NULL support is M3)"
                                .to_string(),
                        )));
                    }
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
                // Empty set is NULL -> hard error (M3), like SUM.
                SelectProjection::Min { column } | SelectProjection::Max { column } => {
                    let col_idx = relational_column_index(table, column)?;
                    if indices.is_empty() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "MIN / MAX over an empty set is NULL, which the engine cannot represent \
                             yet (NULL support is M3)"
                                .to_string(),
                        )));
                    }
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
                                device_memory
                                    .max_i32_at_indices_from_payload(byte_offset, &indices)
                            } else {
                                device_memory
                                    .min_i32_at_indices_from_payload(byte_offset, &indices)
                            }
                            .map_err(map_err)?;
                            SqlValue::Int4(value)
                        }
                        SqlType::Int8 => {
                            let byte_offset =
                                resident_device_int8_column_offset(&snapshot, table, col_idx)?;
                            let value = if is_max {
                                device_memory
                                    .max_i64_at_indices_from_payload(byte_offset, &indices)
                            } else {
                                device_memory
                                    .min_i64_at_indices_from_payload(byte_offset, &indices)
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
                // path). Empty set is NULL -> hard error (M3), like the others.
                SelectProjection::Avg { column } => {
                    let col_idx = relational_column_index(table, column)?;
                    if indices.is_empty() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "AVG over an empty set is NULL, which the engine cannot represent yet \
                             (NULL support is M3)"
                                .to_string(),
                        )));
                    }
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
                columns: bound.selected_columns,
                rows: vec![vec![value]],
                planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
                executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
                fallback_reason: None,
                access_path,
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
        let materialize_int_key_column = |ki: usize, indices: &[u64]| -> Result<Vec<i64>, ExecuteError> {
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
                return device_memory
                    .arith_value_column_at_indices(&program, row_count, &idx32, elem)
                    .map_err(map_err);
            }
            let order = &select.order_by[ki];
            let order_idx = relational_column_index(table, &order.column)?;
            match table.columns[order_idx].ty {
                SqlType::Int4 | SqlType::Int2 | SqlType::Date => Ok(device_memory
                    .project_i32_rows_from_payload(
                        resident_device_int4_column_offset(&snapshot, table, order_idx)?,
                        indices,
                    )
                    .map_err(map_err)?
                    .into_iter()
                    .map(i64::from)
                    .collect()),
                SqlType::Int8 | SqlType::Timestamp => device_memory
                    .project_i64_rows_from_payload(
                        resident_device_int8_column_offset(&snapshot, table, order_idx)?,
                        indices,
                    )
                    .map_err(map_err),
                _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "ORDER BY key is not an i64-sortable int column".to_string(),
                ))),
            }
        };
        // A sort EXPRESSION is int-valued. has_text_key (TEXT only) gates the single-text fast path;
        // has_hetero_key (TEXT/NUMERIC/UUID -- the keys that can't live in the i64 matrix) routes to the
        // heterogeneous comparator.
        let mut has_text_key = false;
        let mut has_hetero_key = false;
        for (ki, order) in select.order_by.iter().enumerate() {
            if order_by_exprs.get(ki).is_some_and(|e| e.is_some()) {
                continue;
            }
            match table.columns[relational_column_index(table, &order.column)?].ty {
                SqlType::Text => {
                    has_text_key = true;
                    has_hetero_key = true;
                }
                SqlType::Numeric { .. } | SqlType::Uuid => has_hetero_key = true,
                _ => {}
            }
        }
        let single_text_key = select.order_by.len() == 1 && has_text_key;
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
        } else if has_hetero_key {
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
            let mut int_key_slots: Vec<(usize, usize)> = Vec::new();
            for (ki, order) in select.order_by.iter().enumerate() {
                if order.descending {
                    desc_mask |= 1u64 << ki;
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
            // The text VALUE is materialized host-side (the SqlValue::Text from host_rows), like the
            // GROUP BY text result -- the GPU did the filter + the sort; this gathers the result strings.
            Text(Vec<SqlValue>),
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
                    // The text VALUE is materialized host-side from the residency_entry's host rows (the
                    // SAME generation as the GPU residency, like the GROUP BY text result), gathered at
                    // the GPU-sorted surviving indices. The GPU did the hot path (filter + sort).
                    let host_rows = &residency_entry.host_rows;
                    let values: Vec<SqlValue> = indices_u64
                        .iter()
                        .map(|&row| host_rows[row as usize][col].clone())
                        .collect();
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
        let mut rows: Vec<Vec<SqlValue>> = (0..indices_u64.len())
            .map(|row| {
                projected_columns
                    .iter()
                    .map(|column| match column {
                        ProjectedColumn::Int4(values) => SqlValue::Int4(values[row]),
                        ProjectedColumn::Int8(values) => SqlValue::Int8(values[row]),
                        ProjectedColumn::Numeric(values, scale) => {
                            SqlValue::Numeric(Decimal128::new(values[row], *scale))
                        }
                        ProjectedColumn::Date(values) => SqlValue::Date(values[row]),
                        ProjectedColumn::Timestamp(values) => SqlValue::Timestamp(values[row]),
                        ProjectedColumn::Uuid(values) => {
                            SqlValue::Uuid(values[row].to_le_bytes())
                        }
                        // The stored i32 is a widened i16, so the narrowing is exact.
                        ProjectedColumn::Int2(values) => SqlValue::Int2(values[row] as i16),
                        ProjectedColumn::Bool(values) => SqlValue::Bool(values[row]),
                        // Already a SqlValue::Text from host_rows; clone it through.
                        ProjectedColumn::Text(values) => values[row].clone(),
                    })
                    .collect()
            })
            .collect();
        // OFFSET then LIMIT, applied after the GPU sort (SQL clause order: ORDER BY -> OFFSET -> LIMIT).
        if select.offset.is_some() || select.limit.is_some() {
            let start = select.offset.unwrap_or(0).min(rows.len());
            rows.drain(..start);
            if let Some(limit) = select.limit {
                rows.truncate(limit);
            }
        }

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
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
        if expr_mentions_int4_column(lhs, table) || expr_mentions_int4_column(rhs, table) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed int4/int8 expressions yet"
                    .to_string(),
            )));
        }
        let mut program = Vec::new();
        compile_predicate_program(predicate, table, snapshot, &mut program)?;
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
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only simple date comparisons (date AND/OR and \
                 arithmetic are follow-ons)"
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
    fn try_lower_timestamp_predicate(
        &self,
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
        let Some(cmp) = predicate_compare_code(compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only simple timestamp comparisons (timestamp \
                 AND/OR and arithmetic are follow-ons)"
                    .to_string(),
            )));
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
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
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only simple uuid comparisons (uuid AND/OR is a \
                 follow-on)"
                    .to_string(),
            )));
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        match (uuid_column_index(lhs, table), uuid_column_index(rhs, table)) {
            (Some(col), None) => {
                let needle = uuid_literal_bytes(rhs)?;
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_uuid_compare_scalar_filter(offset, &needle, false, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (None, Some(col)) => {
                let needle = uuid_literal_bytes(lhs)?;
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_uuid_compare_scalar_filter(offset, &needle, true, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                device_memory
                    .expr_uuid_compare_columns_filter(a_offset, b_offset, cmp, row_count)
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
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only simple smallint comparisons (smallint \
                 AND/OR and arithmetic are follow-ons)"
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
    /// non-text predicate (the other type paths handle it). Text INEQUALITIES (need collation sort
    /// keys), `LIKE`, text `AND`/`OR`, text column-vs-column, and text mixed with another type are hard
    /// errors -- never a silent mis-answer.
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
                    &tokens,
                    row_count,
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
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "text inequalities need collation sort keys (a follow-on); only = and <> run \
                     on the GPU"
                        .to_string(),
                )));
            }
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports text = and <> only (LIKE and text AND/OR \
                     are follow-ons)"
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
            (Some(_), Some(_)) => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "text column-vs-column comparison is a follow-on".to_string(),
                )));
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
                let col_scale = column_numeric_scale(table, col).expect("numeric column has a scale");
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
                let col_scale = column_numeric_scale(table, col).expect("numeric column has a scale");
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
                        offset, col_scale, literal, true, cmp, device_memory, row_count,
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
    /// primitive `expr_filter_two_col_compare_from_payload` (elementwise `a <arith> b` into an
    /// intermediate device buffer, then compare-to-row-indices). Anything else is rejected with a
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
        let is_bool_col = |idx: usize| {
            table.columns.get(idx).map(|column| column.ty) == Some(SqlType::Bool)
        };
        let (col, literal) = match (lhs, rhs) {
            (ResidentExpr::Column(col), ResidentExpr::BoolLiteral(b)) if is_bool_col(*col) => {
                (*col, *b)
            }
            (ResidentExpr::BoolLiteral(b), ResidentExpr::Column(col)) if is_bool_col(*col) => {
                (*col, *b)
            }
            _ => return Ok(None),
        };
        // `flag = true` -> set bits (negate=false); `flag = false` -> clear bits (negate=true).
        // `<>` is the complement.
        let negate = match compare {
            ResidentBinaryOp::Eq => !literal,
            ResidentBinaryOp::Ne => literal,
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports only = / <> against a bool literal".to_string(),
                )));
            }
        };
        let offset = resident_device_bool_column_offset(snapshot, table, col)?;
        device_memory
            .expr_bool_to_mask_filter(offset, negate, row_count)
            .map(Some)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    fn lower_resident_predicate(
        &self,
        predicate: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        // bool-predicate (type matrix, doc 19): a bare `WHERE flag` is a bool COLUMN used directly as
        // a predicate -- a bool column is a 1-bit-per-row bitmap, so it expands straight to the row
        // mask (true rows). A bare NON-bool column is invalid SQL (PG: "argument of WHERE must be type
        // boolean"), so it hard-errors rather than mis-answering.
        if let ResidentExpr::Column(col) = predicate {
            return self.lower_bool_column_predicate(*col, table, snapshot, device_memory, row_count);
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
        if let Some(indices) =
            self.try_lower_bool_predicate(*compare, lhs, rhs, table, snapshot, device_memory, row_count)?
        {
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
        // comparison -> a mask, MaskBinary combines, the terminal compacts). The fused 2-col / arith
        // / col-vs-col fast paths below handle the single eq/lt/le/gt/ge comparisons.
        if matches!(
            compare,
            ResidentBinaryOp::And | ResidentBinaryOp::Or | ResidentBinaryOp::Ne
        ) {
            let mut program = Vec::new();
            compile_predicate_program(predicate, table, snapshot, &mut program)?;
            return device_memory
                .run_expr_predicate_filter(&program, row_count, ResidentElemType::I32)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())));
        }

        let Some(comparison) = compare_op_code(*compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr predicate requires a comparison op (eq/lt/le/gt/ge/ne) or AND/OR"
                    .to_string(),
            )));
        };

        // PEEPHOLE fast-path (design: tuned fused kernels live UNDER the general executor): the exact
        // shape `Compare(arith(Column, Column), Int4Literal)` lowers to the single fused 2-col kernel
        // instead of the 3-launch VM program. Behavior-identical to the VM; just fewer launches.
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
