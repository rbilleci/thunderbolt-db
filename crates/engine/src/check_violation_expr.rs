//! Catalog-resolved, row-local CHECK lowering to one device *violation* expression.
//!
//! CHECK succeeds for TRUE and UNKNOWN, so a NULL input never appears in this mask.  The
//! compiler normalizes cross-numeric comparisons on the stored column lattice before emitting a
//! `ResidentExpr`; callers never build a passing expression and complement it later.

use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};
use crate::relational_model::{RelationalCheckConstraint, RelationalTable};
use crate::{Decimal128, SelectFilterOp, SqlType, SqlValue};

const DAY_US: i128 = 86_400_000_000;

/// Compile one catalog-published CHECK as the rows which violate it. `None` is a fail-closed
/// lowering decline; DDL already rejected type-incompatible predicates, but recovery/programmatic
/// catalog input must not be treated as supported merely because it has a comparison operator.
pub(crate) fn compile_check_violation(
    table: &RelationalTable,
    constraint: &RelationalCheckConstraint,
    column_idx: usize,
) -> Option<ResidentExpr> {
    let column = table.columns.get(column_idx)?;
    if column.name != constraint.column {
        return None;
    }
    if !resolved_operand_is_admissible(column.ty, constraint) {
        return None;
    }
    if matches!(constraint.value, SqlValue::Null) {
        return Some(all_zero(column_idx));
    }
    match column.ty {
        SqlType::Int2 => compile_integer_violation(
            column_idx,
            constraint.op,
            &constraint.value,
            i128::from(i16::MIN),
            i128::from(i16::MAX),
            |value| i32::try_from(value).ok().map(ResidentExpr::Int4Literal),
        ),
        SqlType::Int4 => compile_integer_violation(
            column_idx,
            constraint.op,
            &constraint.value,
            i128::from(i32::MIN),
            i128::from(i32::MAX),
            |value| i32::try_from(value).ok().map(ResidentExpr::Int4Literal),
        ),
        SqlType::Int8 => compile_integer_violation(
            column_idx,
            constraint.op,
            &constraint.value,
            i128::from(i64::MIN),
            i128::from(i64::MAX),
            |value| i64::try_from(value).ok().map(ResidentExpr::Int8Literal),
        ),
        SqlType::Numeric { precision, scale } => compile_numeric_violation(
            column_idx,
            constraint.op,
            &constraint.value,
            precision,
            scale,
        ),
        SqlType::Bool => same_type_violation(
            column_idx,
            constraint.op,
            &constraint.value,
            |value| match value {
                SqlValue::Bool(value) => Some(ResidentExpr::BoolLiteral(*value)),
                _ => None,
            },
        ),
        SqlType::Text => same_type_violation(
            column_idx,
            constraint.op,
            &constraint.value,
            |value| match value {
                SqlValue::Text(value) => Some(ResidentExpr::TextLiteral(value.clone())),
                _ => None,
            },
        ),
        SqlType::Date => match &constraint.value {
            SqlValue::Date(value) => {
                same_type_violation(column_idx, constraint.op, &constraint.value, |_| {
                    Some(ResidentExpr::TextLiteral(
                        gpu_db_sql::datetime::format_date(*value),
                    ))
                })
            }
            SqlValue::Timestamp(value) => {
                compile_date_timestamp_violation(column_idx, constraint.op, *value)
            }
            _ => None,
        },
        SqlType::Timestamp => match &constraint.value {
            SqlValue::Timestamp(value) => {
                same_type_violation(column_idx, constraint.op, &constraint.value, |_| {
                    Some(ResidentExpr::Int8Literal(*value))
                })
            }
            SqlValue::Date(value) => {
                compile_timestamp_date_violation(column_idx, constraint.op, *value)
            }
            _ => None,
        },
        SqlType::Uuid => same_type_violation(
            column_idx,
            constraint.op,
            &constraint.value,
            |value| match value {
                SqlValue::Uuid(value) => Some(ResidentExpr::TextLiteral(
                    gpu_db_sql::uuid::format_uuid(value),
                )),
                _ => None,
            },
        ),
    }
}

/// PostgreSQL permits a CREATE-time `numeric(p,s)` scalar cast whose rounded result exceeds the
/// typmod. The cast failure is deferred until the published CHECK is evaluated. The catalog keeps
/// the original source type so this stays exact across typed, transient, and recovery routes.
pub(crate) fn validate_check_literal_at_evaluation(
    constraint: &RelationalCheckConstraint,
) -> Result<(), crate::EngineError> {
    let SqlType::Numeric { precision, .. } = constraint.resolved_input_type else {
        return Ok(());
    };
    if matches!(constraint.value, SqlValue::Null) {
        return Ok(());
    }
    let SqlValue::Numeric(value) = constraint.value else {
        return Err(crate::EngineError::ApplyFailed(format!(
            "numeric CHECK literal metadata disagrees with constraint \"{}\"",
            constraint.name
        )));
    };
    if numeric_exceeds_precision(value.mantissa, precision) {
        return Err(crate::EngineError::NumericValueOutOfRange(
            "numeric field overflow".to_string(),
        ));
    }
    Ok(())
}

/// CHECK uses SQL's TRUE-or-UNKNOWN rule rather than WHERE's TRUE-only rule. Keep temporal
/// coercion here, beside the device lattice lowering, so ADD CHECK never falls through the
/// engine's heterogeneous sort ordering.
pub(crate) fn check_comparison_satisfies(
    left: &SqlValue,
    check_op: SelectFilterOp,
    right: &SqlValue,
) -> bool {
    if matches!(left, SqlValue::Null) || matches!(right, SqlValue::Null) {
        return true;
    }
    let temporal_order = match (left, right) {
        (SqlValue::Date(days), SqlValue::Timestamp(timestamp)) => {
            Some((i128::from(*days) * DAY_US).cmp(&i128::from(*timestamp)))
        }
        (SqlValue::Timestamp(timestamp), SqlValue::Date(days)) => {
            Some(i128::from(*timestamp).cmp(&(i128::from(*days) * DAY_US)))
        }
        _ => None,
    };
    let Some(order) = temporal_order else {
        return crate::select_filter_matches(left, check_op, right);
    };
    match check_op {
        SelectFilterOp::Eq => order.is_eq(),
        SelectFilterOp::Lt => order.is_lt(),
        SelectFilterOp::Lte => !order.is_gt(),
        SelectFilterOp::Gt => order.is_gt(),
        SelectFilterOp::Gte => !order.is_lt(),
        SelectFilterOp::LikePrefix => false,
    }
}

fn numeric_exceeds_precision(mantissa: i128, precision: u8) -> bool {
    let mut bound = 1_i128;
    for _ in 0..precision {
        let Some(next) = bound.checked_mul(10) else {
            return false;
        };
        bound = next;
    }
    mantissa.unsigned_abs() >= bound.unsigned_abs()
}

fn compile_integer_violation(
    column_idx: usize,
    check_op: SelectFilterOp,
    value: &SqlValue,
    min: i128,
    max: i128,
    literal: impl Fn(i128) -> Option<ResidentExpr>,
) -> Option<ResidentExpr> {
    let (floor, ceil) = integer_floor_ceil(value)?;
    let (violation_op, threshold) = violation_threshold(check_op, floor, ceil)?;
    lattice_comparison(column_idx, violation_op, threshold, min, max, literal)
}

fn compile_numeric_violation(
    column_idx: usize,
    check_op: SelectFilterOp,
    value: &SqlValue,
    precision: u8,
    scale: u8,
) -> Option<ResidentExpr> {
    let decimal = numeric_value(value)?;
    let (floor, ceil) = decimal_floor_ceil_at_scale(decimal, scale);
    let max = pow10(precision)?.checked_sub(1)?;
    let min = max.checked_neg()?;
    let (violation_op, threshold) = violation_threshold(check_op, floor, ceil)?;
    lattice_comparison(column_idx, violation_op, threshold, min, max, |mantissa| {
        Some(ResidentExpr::NumericLiteral(Decimal128::new(
            mantissa, scale,
        )))
    })
}

fn compile_timestamp_date_violation(
    column_idx: usize,
    check_op: SelectFilterOp,
    date_days: i32,
) -> Option<ResidentExpr> {
    let threshold = i128::from(date_days) * DAY_US;
    let (floor, ceil) = exact(threshold);
    let (violation_op, threshold) = violation_threshold(check_op, floor, ceil)?;
    lattice_comparison(
        column_idx,
        violation_op,
        threshold,
        i128::from(i64::MIN),
        i128::from(i64::MAX),
        |value| i64::try_from(value).ok().map(ResidentExpr::Int8Literal),
    )
}

fn compile_date_timestamp_violation(
    column_idx: usize,
    check_op: SelectFilterOp,
    timestamp_us: i64,
) -> Option<ResidentExpr> {
    let timestamp_us = i128::from(timestamp_us);
    let days = timestamp_us.div_euclid(DAY_US);
    let floor = LatticeValue::Value(days);
    let ceil = if timestamp_us.rem_euclid(DAY_US) == 0 {
        floor
    } else {
        LatticeValue::Value(days.checked_add(1)?)
    };
    let (violation_op, threshold) = violation_threshold(check_op, floor, ceil)?;
    lattice_comparison(
        column_idx,
        violation_op,
        threshold,
        i128::from(i32::MIN),
        i128::from(i32::MAX),
        |value| {
            i32::try_from(value)
                .ok()
                .map(gpu_db_sql::datetime::format_date)
                .map(ResidentExpr::TextLiteral)
        },
    )
}

fn resolved_operand_is_admissible(
    column_ty: SqlType,
    constraint: &RelationalCheckConstraint,
) -> bool {
    check_operand_types_compatible(column_ty, constraint.resolved_input_type)
        && (matches!(constraint.value, SqlValue::Null)
            || value_matches_type(&constraint.value, constraint.resolved_input_type))
}

fn value_matches_type(value: &SqlValue, ty: SqlType) -> bool {
    matches!(
        (value, ty),
        (SqlValue::Int2(_), SqlType::Int2)
            | (SqlValue::Int4(_), SqlType::Int4)
            | (SqlValue::Int8(_), SqlType::Int8)
            | (SqlValue::Numeric(_), SqlType::Numeric { .. })
            | (SqlValue::Bool(_), SqlType::Bool)
            | (SqlValue::Text(_), SqlType::Text)
            | (SqlValue::Date(_), SqlType::Date)
            | (SqlValue::Timestamp(_), SqlType::Timestamp)
            | (SqlValue::Uuid(_), SqlType::Uuid)
    )
}

fn check_operand_types_compatible(left: SqlType, right: SqlType) -> bool {
    matches!(
        (left, right),
        (
            SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. },
            SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. }
        ) | (SqlType::Bool, SqlType::Bool)
            | (SqlType::Text, SqlType::Text)
            | (SqlType::Date, SqlType::Date | SqlType::Timestamp)
            | (SqlType::Timestamp, SqlType::Date | SqlType::Timestamp)
            | (SqlType::Uuid, SqlType::Uuid)
    )
}

/// Map `column CHECK_OP literal` to the exact violation comparison on the column's integer
/// lattice. `Eq` with a non-lattice literal is false for every non-NULL row, so its violation is
/// the validity mask. The strict directions use floor/ceil rather than assignment rounding.
fn violation_threshold(
    check_op: SelectFilterOp,
    floor: LatticeValue,
    ceil: LatticeValue,
) -> Option<(ResidentBinaryOp, LatticeValue)> {
    match check_op {
        SelectFilterOp::Eq if floor != ceil => {
            Some((ResidentBinaryOp::Ne, LatticeValue::NonLattice))
        }
        SelectFilterOp::Eq => Some((ResidentBinaryOp::Ne, floor)),
        SelectFilterOp::Lt => Some((ResidentBinaryOp::Ge, ceil)),
        SelectFilterOp::Lte => Some((ResidentBinaryOp::Gt, floor)),
        SelectFilterOp::Gt => Some((ResidentBinaryOp::Le, floor)),
        SelectFilterOp::Gte => Some((ResidentBinaryOp::Lt, ceil)),
        SelectFilterOp::LikePrefix => None,
    }
}

/// Emit a comparison, or fold it to the validity mask/all-zero mask when its threshold lies
/// outside the physical domain. A folded true means "true for each valid input", never a mask of
/// all rows: NULL still satisfies CHECK.
fn lattice_comparison(
    column_idx: usize,
    op: ResidentBinaryOp,
    threshold: LatticeValue,
    min: i128,
    max: i128,
    literal: impl Fn(i128) -> Option<ResidentExpr>,
) -> Option<ResidentExpr> {
    if threshold == LatticeValue::NonLattice {
        return Some(valid_rows(column_idx));
    }
    let folded = match (op, threshold) {
        (ResidentBinaryOp::Ne, LatticeValue::Below | LatticeValue::Above) => Some(true),
        (ResidentBinaryOp::Ge, LatticeValue::Below)
        | (ResidentBinaryOp::Gt, LatticeValue::Below) => Some(true),
        (ResidentBinaryOp::Le, LatticeValue::Above)
        | (ResidentBinaryOp::Lt, LatticeValue::Above) => Some(true),
        (ResidentBinaryOp::Ge, LatticeValue::Above)
        | (ResidentBinaryOp::Gt, LatticeValue::Above) => Some(false),
        (ResidentBinaryOp::Le, LatticeValue::Below)
        | (ResidentBinaryOp::Lt, LatticeValue::Below) => Some(false),
        (_, LatticeValue::Value(value)) => match op {
            ResidentBinaryOp::Ne if value < min || value > max => Some(true),
            ResidentBinaryOp::Ge if value <= min => Some(true),
            ResidentBinaryOp::Ge if value > max => Some(false),
            ResidentBinaryOp::Gt if value < min => Some(true),
            ResidentBinaryOp::Gt if value >= max => Some(false),
            ResidentBinaryOp::Le if value < min => Some(false),
            ResidentBinaryOp::Le if value >= max => Some(true),
            ResidentBinaryOp::Lt if value <= min => Some(false),
            ResidentBinaryOp::Lt if value > max => Some(true),
            _ => None,
        },
        _ => return None,
    };
    if let Some(value) = folded {
        return Some(if value {
            valid_rows(column_idx)
        } else {
            all_zero(column_idx)
        });
    }
    let LatticeValue::Value(value) = threshold else {
        return None;
    };
    Some(ResidentExpr::Binary {
        op,
        lhs: Box::new(ResidentExpr::Column(column_idx)),
        rhs: Box::new(literal(value)?),
    })
}

fn same_type_violation(
    column_idx: usize,
    check_op: SelectFilterOp,
    value: &SqlValue,
    literal: impl Fn(&SqlValue) -> Option<ResidentExpr>,
) -> Option<ResidentExpr> {
    Some(ResidentExpr::Binary {
        op: complement(check_op)?,
        lhs: Box::new(ResidentExpr::Column(column_idx)),
        rhs: Box::new(literal(value)?),
    })
}

fn complement(op: SelectFilterOp) -> Option<ResidentBinaryOp> {
    Some(match op {
        SelectFilterOp::Eq => ResidentBinaryOp::Ne,
        SelectFilterOp::Lt => ResidentBinaryOp::Ge,
        SelectFilterOp::Lte => ResidentBinaryOp::Gt,
        SelectFilterOp::Gt => ResidentBinaryOp::Le,
        SelectFilterOp::Gte => ResidentBinaryOp::Lt,
        SelectFilterOp::LikePrefix => return None,
    })
}

fn valid_rows(column_idx: usize) -> ResidentExpr {
    ResidentExpr::IsNull {
        col: column_idx,
        is_not_null: true,
    }
}

fn all_zero(column_idx: usize) -> ResidentExpr {
    ResidentExpr::Binary {
        op: ResidentBinaryOp::And,
        lhs: Box::new(valid_rows(column_idx)),
        rhs: Box::new(ResidentExpr::IsNull {
            col: column_idx,
            is_not_null: false,
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LatticeValue {
    Below,
    Value(i128),
    Above,
    /// A value between adjacent lattice points, used only for `=`. No stored value can equal it.
    NonLattice,
}

fn integer_floor_ceil(value: &SqlValue) -> Option<(LatticeValue, LatticeValue)> {
    match value {
        SqlValue::Int2(value) => Some(exact(i128::from(*value))),
        SqlValue::Int4(value) => Some(exact(i128::from(*value))),
        SqlValue::Int8(value) => Some(exact(i128::from(*value))),
        SqlValue::Numeric(value) => Some(decimal_floor_ceil_at_scale(*value, 0)),
        _ => None,
    }
}

fn numeric_value(value: &SqlValue) -> Option<Decimal128> {
    match value {
        SqlValue::Int2(value) => Some(Decimal128::new(i128::from(*value), 0)),
        SqlValue::Int4(value) => Some(Decimal128::new(i128::from(*value), 0)),
        SqlValue::Int8(value) => Some(Decimal128::new(i128::from(*value), 0)),
        SqlValue::Numeric(value) => Some(*value),
        _ => None,
    }
}

fn exact(value: i128) -> (LatticeValue, LatticeValue) {
    (LatticeValue::Value(value), LatticeValue::Value(value))
}

/// Exact floor/ceil of `decimal * 10^target_scale`, without assignment rounding. The overflow
/// sentinels are sufficient because every caller immediately folds against a finite physical
/// lattice range.
fn decimal_floor_ceil_at_scale(
    decimal: Decimal128,
    target_scale: u8,
) -> (LatticeValue, LatticeValue) {
    if target_scale >= decimal.scale {
        let shift = target_scale - decimal.scale;
        let Some(factor) = pow10(shift) else {
            return overflow_pair(decimal.mantissa);
        };
        return match decimal.mantissa.checked_mul(factor) {
            Some(value) => exact(value),
            None => overflow_pair(decimal.mantissa),
        };
    }
    let shift = decimal.scale - target_scale;
    let Some(factor) = pow10(shift) else {
        return match decimal.mantissa.cmp(&0) {
            std::cmp::Ordering::Less => (LatticeValue::Value(-1), LatticeValue::Value(0)),
            std::cmp::Ordering::Equal => exact(0),
            std::cmp::Ordering::Greater => (LatticeValue::Value(0), LatticeValue::Value(1)),
        };
    };
    let quotient = decimal.mantissa / factor;
    let remainder = decimal.mantissa % factor;
    if remainder == 0 {
        exact(quotient)
    } else if decimal.mantissa < 0 {
        (
            LatticeValue::Value(quotient - 1),
            LatticeValue::Value(quotient),
        )
    } else {
        (
            LatticeValue::Value(quotient),
            LatticeValue::Value(quotient + 1),
        )
    }
}

fn overflow_pair(mantissa: i128) -> (LatticeValue, LatticeValue) {
    if mantissa < 0 {
        (LatticeValue::Below, LatticeValue::Below)
    } else if mantissa > 0 {
        (LatticeValue::Above, LatticeValue::Above)
    } else {
        exact(0)
    }
}

fn pow10(scale: u8) -> Option<i128> {
    let mut value = 1_i128;
    for _ in 0..scale {
        value = value.checked_mul(10)?;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(ty: SqlType) -> RelationalTable {
        RelationalTable {
            schema: "public".to_string(),
            name: "checks".to_string(),
            oid: 1,
            columns: vec![crate::RelationalColumn {
                id: 1,
                table_oid: 1,
                attnum: 1,
                name: "value".to_string(),
                ty,
                domain: None,
                default: None,
                type_oid: ty.postgres_oid(),
                type_size: ty.type_size(),
            }],
            indexes: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
            acl: Default::default(),
        }
    }

    fn check(op: SelectFilterOp, value: SqlValue) -> RelationalCheckConstraint {
        let resolved_input_type = match &value {
            SqlValue::Int2(_) => SqlType::Int2,
            SqlValue::Int4(_) => SqlType::Int4,
            SqlValue::Int8(_) => SqlType::Int8,
            SqlValue::Numeric(value) => SqlType::Numeric {
                precision: crate::NUMERIC_DEFAULT_PRECISION,
                scale: value.scale,
            },
            SqlValue::Bool(_) => SqlType::Bool,
            SqlValue::Text(_) => SqlType::Text,
            SqlValue::Date(_) => SqlType::Date,
            SqlValue::Timestamp(_) => SqlType::Timestamp,
            SqlValue::Uuid(_) => SqlType::Uuid,
            SqlValue::Null => SqlType::Int4,
            SqlValue::Parameter { .. } => unreachable!("check compiler receives bound values"),
        };
        RelationalCheckConstraint {
            name: "value_check".to_string(),
            column: "value".to_string(),
            op,
            value,
            resolved_input_type,
            identity_version: crate::CheckOperandIdentityVersion::LegacyV1,
        }
    }

    fn binary(expr: ResidentExpr) -> (ResidentBinaryOp, ResidentExpr) {
        let ResidentExpr::Binary { op, lhs: _, rhs } = expr else {
            panic!("expected comparison expression");
        };
        (op, *rhs)
    }

    #[test]
    fn integer_lattice_rewrites_every_operator_with_fractional_positive_and_negative_values() {
        let table = table(SqlType::Int4);
        let cases = [
            (SelectFilterOp::Eq, Decimal128::new(12, 1), None),
            (
                SelectFilterOp::Lt,
                Decimal128::new(12, 1),
                Some((ResidentBinaryOp::Ge, 2)),
            ),
            (
                SelectFilterOp::Lte,
                Decimal128::new(12, 1),
                Some((ResidentBinaryOp::Gt, 1)),
            ),
            (
                SelectFilterOp::Gt,
                Decimal128::new(-12, 1),
                Some((ResidentBinaryOp::Le, -2)),
            ),
            (
                SelectFilterOp::Gte,
                Decimal128::new(-12, 1),
                Some((ResidentBinaryOp::Lt, -1)),
            ),
        ];
        for (op, value, expected) in cases {
            let expr =
                compile_check_violation(&table, &check(op, SqlValue::Numeric(value)), 0).unwrap();
            match expected {
                Some((op, literal)) => {
                    let (actual_op, rhs) = binary(expr);
                    assert_eq!(actual_op, op);
                    assert!(matches!(rhs, ResidentExpr::Int4Literal(value) if value == literal));
                }
                None => assert!(matches!(
                    expr,
                    ResidentExpr::IsNull {
                        is_not_null: true,
                        ..
                    }
                )),
            }
        }
    }

    #[test]
    fn integer_min_max_and_wider_constants_fold_to_validity_or_zero_without_narrowing() {
        let table = table(SqlType::Int2);
        for (op, value, valid) in [
            (SelectFilterOp::Lt, SqlValue::Int4(-32_768), true),
            (SelectFilterOp::Lt, SqlValue::Int4(32_768), false),
            (SelectFilterOp::Gt, SqlValue::Int4(-32_769), false),
            (SelectFilterOp::Gt, SqlValue::Int4(32_767), true),
            (SelectFilterOp::Eq, SqlValue::Int4(32_768), true),
        ] {
            let expr = compile_check_violation(&table, &check(op, value), 0).unwrap();
            if valid {
                assert!(matches!(
                    expr,
                    ResidentExpr::IsNull {
                        is_not_null: true,
                        ..
                    }
                ));
            } else {
                assert!(matches!(
                    expr,
                    ResidentExpr::Binary {
                        op: ResidentBinaryOp::And,
                        ..
                    }
                ));
            }
        }
    }

    #[test]
    fn numeric_thresholds_use_the_stored_scale_without_rounding() {
        let table = table(SqlType::Numeric {
            precision: 8,
            scale: 2,
        });
        let cases = [
            (
                SelectFilterOp::Lt,
                Decimal128::new(12_345, 3),
                ResidentBinaryOp::Ge,
                1_235,
            ),
            (
                SelectFilterOp::Lte,
                Decimal128::new(12_345, 3),
                ResidentBinaryOp::Gt,
                1_234,
            ),
            (
                SelectFilterOp::Gt,
                Decimal128::new(-12_345, 3),
                ResidentBinaryOp::Le,
                -1_235,
            ),
            (
                SelectFilterOp::Gte,
                Decimal128::new(-12_345, 3),
                ResidentBinaryOp::Lt,
                -1_234,
            ),
        ];
        for (op, value, expected_op, mantissa) in cases {
            let expr =
                compile_check_violation(&table, &check(op, SqlValue::Numeric(value)), 0).unwrap();
            let (op, rhs) = binary(expr);
            assert_eq!(op, expected_op);
            assert!(
                matches!(rhs, ResidentExpr::NumericLiteral(value) if value == Decimal128::new(mantissa, 2))
            );
        }
        let equality = compile_check_violation(
            &table,
            &check(
                SelectFilterOp::Eq,
                SqlValue::Numeric(Decimal128::new(12_345, 3)),
            ),
            0,
        )
        .unwrap();
        assert!(matches!(
            equality,
            ResidentExpr::IsNull {
                is_not_null: true,
                ..
            }
        ));
    }

    #[test]
    fn null_and_same_type_leaves_preserve_device_operators() {
        let int_table = table(SqlType::Int4);
        let null =
            compile_check_violation(&int_table, &check(SelectFilterOp::Eq, SqlValue::Null), 0)
                .unwrap();
        assert!(matches!(
            null,
            ResidentExpr::Binary {
                op: ResidentBinaryOp::And,
                ..
            }
        ));

        let bool_expr = compile_check_violation(
            &table(SqlType::Bool),
            &check(SelectFilterOp::Lte, SqlValue::Bool(false)),
            0,
        )
        .unwrap();
        let (op, rhs) = binary(bool_expr);
        assert_eq!(op, ResidentBinaryOp::Gt);
        assert!(matches!(rhs, ResidentExpr::BoolLiteral(false)));

        let text_expr = compile_check_violation(
            &table(SqlType::Text),
            &check(SelectFilterOp::Eq, SqlValue::Text("x".to_string())),
            0,
        )
        .unwrap();
        let (op, rhs) = binary(text_expr);
        assert_eq!(op, ResidentBinaryOp::Ne);
        assert!(matches!(rhs, ResidentExpr::TextLiteral(value) if value == "x"));
    }

    #[test]
    fn direct_check_compiler_covers_the_typed_insert_boundary_batch_sizes() {
        let table = table(SqlType::Int8);
        for rows in [31_usize, 32, 33] {
            let expr =
                compile_check_violation(&table, &check(SelectFilterOp::Gt, SqlValue::Int4(7)), 0)
                    .unwrap();
            let (op, rhs) = binary(expr);
            assert_eq!(op, ResidentBinaryOp::Le, "rows={rows}");
            assert!(matches!(rhs, ResidentExpr::Int8Literal(7)), "rows={rows}");
        }
    }

    #[test]
    fn timestamp_column_date_operand_uses_exact_midnight_lattice_thresholds() {
        let table = table(SqlType::Timestamp);
        for (days, threshold) in [(-1_i32, -DAY_US), (0, 0), (1, DAY_US)] {
            for (check_op, violation_op) in [
                (SelectFilterOp::Eq, ResidentBinaryOp::Ne),
                (SelectFilterOp::Lt, ResidentBinaryOp::Ge),
                (SelectFilterOp::Lte, ResidentBinaryOp::Gt),
                (SelectFilterOp::Gt, ResidentBinaryOp::Le),
                (SelectFilterOp::Gte, ResidentBinaryOp::Lt),
            ] {
                let expr =
                    compile_check_violation(&table, &check(check_op, SqlValue::Date(days)), 0)
                        .unwrap();
                let (actual_op, rhs) = binary(expr);
                assert_eq!(actual_op, violation_op, "days={days} op={check_op:?}");
                assert!(
                    matches!(rhs, ResidentExpr::Int8Literal(value) if value == threshold as i64)
                );
            }
        }
    }

    #[test]
    fn date_column_timestamp_operand_uses_floor_ceil_for_midnight_and_pre_epoch_edges() {
        let table = table(SqlType::Date);
        for (timestamp, expected) in [
            (
                0_i64,
                [
                    Some((ResidentBinaryOp::Ne, 0)),
                    Some((ResidentBinaryOp::Ge, 0)),
                    Some((ResidentBinaryOp::Gt, 0)),
                    Some((ResidentBinaryOp::Le, 0)),
                    Some((ResidentBinaryOp::Lt, 0)),
                ],
            ),
            (
                1,
                [
                    None,
                    Some((ResidentBinaryOp::Ge, 1)),
                    Some((ResidentBinaryOp::Gt, 0)),
                    Some((ResidentBinaryOp::Le, 0)),
                    Some((ResidentBinaryOp::Lt, 1)),
                ],
            ),
            (
                -1,
                [
                    None,
                    Some((ResidentBinaryOp::Ge, 0)),
                    Some((ResidentBinaryOp::Gt, -1)),
                    Some((ResidentBinaryOp::Le, -1)),
                    Some((ResidentBinaryOp::Lt, 0)),
                ],
            ),
        ] {
            for (check_op, expected) in [
                SelectFilterOp::Eq,
                SelectFilterOp::Lt,
                SelectFilterOp::Lte,
                SelectFilterOp::Gt,
                SelectFilterOp::Gte,
            ]
            .into_iter()
            .zip(expected)
            {
                let expr = compile_check_violation(
                    &table,
                    &check(check_op, SqlValue::Timestamp(timestamp)),
                    0,
                )
                .unwrap();
                match expected {
                    Some((expected_op, expected_value)) => {
                        let (actual_op, rhs) = binary(expr);
                        assert_eq!(
                            actual_op, expected_op,
                            "timestamp={timestamp} op={check_op:?}"
                        );
                        assert!(
                            matches!(rhs, ResidentExpr::TextLiteral(value) if value == gpu_db_sql::datetime::format_date(expected_value))
                        );
                    }
                    None => assert!(matches!(
                        expr,
                        ResidentExpr::IsNull {
                            is_not_null: true,
                            ..
                        }
                    )),
                }
            }
        }
    }

    #[test]
    fn timestamp_date_thresholds_outside_i64_fold_without_literal_narrowing() {
        let table = table(SqlType::Timestamp);
        for (op, date, violates_every_valid_row) in [
            (SelectFilterOp::Lt, i32::MIN, true),
            (SelectFilterOp::Gt, i32::MIN, false),
            (SelectFilterOp::Lt, i32::MAX, false),
            (SelectFilterOp::Gt, i32::MAX, true),
        ] {
            let expr = compile_check_violation(&table, &check(op, SqlValue::Date(date)), 0)
                .expect("out-of-i64 temporal threshold must fold");
            if violates_every_valid_row {
                assert!(matches!(
                    expr,
                    ResidentExpr::IsNull {
                        is_not_null: true,
                        ..
                    }
                ));
            } else {
                assert!(matches!(
                    expr,
                    ResidentExpr::Binary {
                        op: ResidentBinaryOp::And,
                        ..
                    }
                ));
            }
        }
    }

    #[test]
    fn temporal_check_host_helper_preserves_null_unknown_and_exact_cross_type_ordering() {
        assert!(check_comparison_satisfies(
            &SqlValue::Null,
            SelectFilterOp::Gt,
            &SqlValue::Timestamp(-1),
        ));
        assert!(check_comparison_satisfies(
            &SqlValue::Date(0),
            SelectFilterOp::Lt,
            &SqlValue::Timestamp(1),
        ));
        assert!(!check_comparison_satisfies(
            &SqlValue::Date(0),
            SelectFilterOp::Eq,
            &SqlValue::Timestamp(-1),
        ));
        assert!(check_comparison_satisfies(
            &SqlValue::Timestamp(-1),
            SelectFilterOp::Lt,
            &SqlValue::Date(0),
        ));
    }

    #[test]
    fn resolved_operand_metadata_sabotage_fails_closed_before_device_lowering() {
        let timestamp_table = table(SqlType::Timestamp);
        let mut mismatched_value = check(SelectFilterOp::Eq, SqlValue::Timestamp(0));
        mismatched_value.resolved_input_type = SqlType::Date;
        assert!(compile_check_violation(&timestamp_table, &mismatched_value, 0).is_none());

        let mut incompatible_null = check(SelectFilterOp::Eq, SqlValue::Null);
        incompatible_null.resolved_input_type = SqlType::Bool;
        assert!(compile_check_violation(&timestamp_table, &incompatible_null, 0).is_none());

        let mut compatible_null = check(SelectFilterOp::Eq, SqlValue::Null);
        compatible_null.resolved_input_type = SqlType::Date;
        assert!(matches!(
            compile_check_violation(&timestamp_table, &compatible_null, 0),
            Some(ResidentExpr::Binary {
                op: ResidentBinaryOp::And,
                ..
            })
        ));
    }
}
