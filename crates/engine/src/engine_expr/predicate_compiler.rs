//! Typed postfix predicate compilation for resident GPU execution.

use super::predicate_operands::{
    column_numeric_scale, compile_like_pattern, date_column_index, date_literal_days,
    expr_mentions_text, expr_mentions_uuid, is_int4_literal, numeric_literal_value,
    text_column_index, text_literal_value, timestamp_column_index, timestamp_literal_micros,
    uuid_column_index, uuid_literal_bytes,
};
use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};
use crate::relational_model::{
    resident_device_bool_column_offset, resident_device_int4_column_offset,
    resident_device_int8_column_offset, resident_device_int_column_offset,
    resident_device_null_column_offset, resident_device_numeric_column_offset,
    resident_device_text_column_layout, RelationalResidencySnapshot, RelationalTable,
};
use crate::ExecuteError;
use gpu_db_execution::{ExprStep, ResidentElemType};
use gpu_db_sql::{Decimal128, SqlType};
use gpu_db_types::EngineError;

/// Device op-code for an arithmetic binary op (matches `expression_i*.ptx`: 0=add, 1=sub, 2=mul), or
/// `None` if `op` is not arithmetic.
pub(super) fn arith_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::Add => Some(0),
        ResidentBinaryOp::Sub => Some(1),
        ResidentBinaryOp::Mul => Some(2),
        _ => None,
    }
}

/// Device comparison code for a comparison binary op (matches `expression_i*.ptx`: 0=eq, 1=lt, 2=le,
/// 3=gt, 4=ge), or `None` if `op` is not a kernel-supported comparison (`Ne` has no primitive yet).
pub(super) fn compare_op_code(op: ResidentBinaryOp) -> Option<u32> {
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
pub(super) fn flip_comparison_code(code: u32) -> u32 {
    match code {
        1 => 3,
        2 => 4,
        3 => 1,
        4 => 2,
        other => other,
    }
}

/// Rescale a numeric literal to a column's scale and return the comparable i128 mantissa. The literal
/// rescales UP exactly (its scale <= the column scale). A literal with MORE fractional digits than the
/// column is rejected: rescaling it down would round, and PG compares numerics exactly — rounding
/// would yield wrong rows. Cross-scale comparison (rescaling the column on the GPU) is a follow-on.
pub(super) fn rescale_numeric_literal(
    literal: Decimal128,
    column_scale: u8,
) -> Result<i128, ExecuteError> {
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
pub(super) fn push_numeric_rescale(
    from: u8,
    to: u8,
    program: &mut Vec<ExprStep>,
) -> Result<(), ExecuteError> {
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
pub(super) fn compile_numeric_compare(
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
pub(super) fn compile_numeric_predicate_program(
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
pub(super) fn compile_arith_program(
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
pub(super) fn predicate_compare_code(op: ResidentBinaryOp) -> Option<u32> {
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
pub(super) fn boolean_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::And => Some(0),
        ResidentBinaryOp::Or => Some(1),
        _ => None,
    }
}

/// Collect the (distinct, first-seen order) column indices a predicate-leaf operand references. Literals
/// and `IS NULL` contribute no value-operand column (an `IS NULL` leaf is its own validity test, never an
/// arithmetic operand).
pub(super) fn collect_expr_columns(expr: &ResidentExpr, out: &mut Vec<usize>) {
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
pub(super) fn push_column_validity_and(
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
pub(super) fn push_leaf_validity_and(
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
pub(super) fn predicate_references_nullable_column(
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
pub(super) fn predicate_vm_elem_type(
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
pub(super) fn mixed_width_i32_elem(
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
pub(super) fn compile_predicate_program(
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
pub(super) fn compile_text_eq_leaf(
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
