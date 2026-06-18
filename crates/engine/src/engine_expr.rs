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
    // Constructed by the SQL->Expr text mapper, landing in the next commit (the lowering).
    #[allow(dead_code)]
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
    Binary {
        op: ResidentBinaryOp,
        lhs: Box<ResidentExpr>,
        rhs: Box<ResidentExpr>,
    },
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
        | ResidentExpr::TextLiteral(_) => false,
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
        | ResidentExpr::TextLiteral(_) => false,
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
        ResidentExpr::Int4Literal(_) | ResidentExpr::TextLiteral(_) => false,
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
        ResidentExpr::Int4Literal(_) | ResidentExpr::NumericLiteral(_) => false,
        ResidentExpr::Binary { lhs, rhs, .. } => {
            expr_mentions_text(lhs, table) || expr_mentions_text(rhs, table)
        }
    }
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
        ResidentExpr::TextLiteral(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "a text literal is not a numeric arithmetic operand".to_string(),
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
        | ResidentExpr::TextLiteral(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
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
        | ResidentExpr::TextLiteral(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
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
        self.execute_resident_expr_select_with_binding(select, &table, bound, copin_s, predicate)
    }

    /// As [`Engine::execute_resident_expr_select`] but over an ALREADY-BOUND table/projection: the
    /// caller bound the catalog once and resolved the predicate's `Column` indices against this SAME
    /// `table`. The SQL->Expr entry (`engine_sql_pg`) routes through here so the predicate's column
    /// indices, the projection, and the residency snapshot all derive from one catalog generation — a
    /// concurrent shape-changing DDL cannot split the column resolution from the execution.
    pub(crate) fn execute_resident_expr_select_with_binding(
        &self,
        select: &Select,
        table: &RelationalTable,
        bound: BoundRelationalSelect,
        copin_s: Index,
        predicate: &ResidentExpr,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        if bound.selected_indexes.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr select requires at least one projected column".to_string(),
            )));
        }
        for &col in &bound.selected_indexes {
            let ty = table.columns[col].ty;
            if ty != SqlType::Int4 && ty != SqlType::Int8 && !matches!(ty, SqlType::Numeric { .. }) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident Expr select currently materializes int4 / int8 / numeric projection \
                     columns only"
                        .to_string(),
                )));
            }
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot_ref(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident snapshot",
                    table.name
                )))
            })?;
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

        // Evaluate the predicate on the GPU -> surviving row indices (ascending).
        let indices = self.lower_resident_predicate(
            predicate,
            table,
            &snapshot,
            &device_memory,
            row_count,
        )?;
        let indices_u64: Vec<u64> = indices.iter().map(|&i| u64::from(i)).collect();

        // Materialize: gather each projected column at the surviving row indices on the GPU, by type
        // (int4 -> i32 gather, int8 -> i64 gather; the type matrix, doc 19).
        enum ProjectedColumn {
            Int4(Vec<i32>),
            Int8(Vec<i64>),
            Numeric(Vec<i128>, u8),
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
        let rows: Vec<Vec<SqlValue>> = (0..indices_u64.len())
            .map(|row| {
                projected_columns
                    .iter()
                    .map(|column| match column {
                        ProjectedColumn::Int4(values) => SqlValue::Int4(values[row]),
                        ProjectedColumn::Int8(values) => SqlValue::Int8(values[row]),
                        ProjectedColumn::Numeric(values, scale) => {
                            SqlValue::Numeric(Decimal128::new(values[row], *scale))
                        }
                    })
                    .collect()
            })
            .collect();

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
            .map_err(|err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            })
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
    fn lower_resident_predicate(
        &self,
        predicate: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
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
