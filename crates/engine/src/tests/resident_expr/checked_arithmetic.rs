use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};
use crate::{parse_command, Command, Engine, ExecuteError, RelationalSelectResult};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;

// ---- Checked int4 arithmetic (Postgres `integer out of range`, Charter rule 2) ----

/// `Column(0)` — the single int4 column the checked-arithmetic tests project + filter on.
fn column0() -> ResidentExpr {
    ResidentExpr::Column(0)
}

/// `lhs * rhs`.
fn mul(lhs: ResidentExpr, rhs: ResidentExpr) -> ResidentExpr {
    ResidentExpr::Binary {
        op: ResidentBinaryOp::Mul,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

/// `lhs + rhs`.
fn add(lhs: ResidentExpr, rhs: ResidentExpr) -> ResidentExpr {
    ResidentExpr::Binary {
        op: ResidentBinaryOp::Add,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

/// `lhs - rhs`.
fn sub(lhs: ResidentExpr, rhs: ResidentExpr) -> ResidentExpr {
    ResidentExpr::Binary {
        op: ResidentBinaryOp::Sub,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

/// `value > 0` — a predicate whose only failure mode here is the arithmetic in `value` overflowing.
fn gt_zero(value: ResidentExpr) -> ResidentExpr {
    ResidentExpr::Binary {
        op: ResidentBinaryOp::Gt,
        lhs: Box::new(value),
        rhs: Box::new(ResidentExpr::Int4Literal(0)),
    }
}

/// Build a single-int4-column table `<table>(a) = values`, push a GPU residency snapshot, and run
/// `SELECT a FROM <table> WHERE <predicate>` through the general GPU executor. Returns `None` when no
/// GPU is present (the snapshot has no device-memory proof) so the test skips cleanly off-GPU. The
/// result is owned host data, so it outlives the local engine.
fn eval_single_col_predicate(
    table: &str,
    values: &[i32],
    predicate: &ResidentExpr,
) -> Option<Result<RelationalSelectResult, ExecuteError>> {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, &format!("CREATE TABLE {table} (a INT)"))
        .unwrap();
    let tuples = values
        .iter()
        .map(|v| format!("({v})"))
        .collect::<Vec<_>>()
        .join(", ");
    e.execute_text(2, &format!("INSERT INTO {table} (a) VALUES {tuples}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot(table).unwrap();
    snapshot.device_memory_proof.as_ref()?;
    let Command::Select(select) = parse_command(&format!("SELECT a FROM {table}")).unwrap() else {
        unreachable!()
    };
    Some(e.execute_resident_expr_select(&select, predicate))
}

/// Assert a select result is the `integer out of range` error (not rows). Matches on the error rather
/// than `expect_err` so it does not require `RelationalSelectResult: Debug`.
fn assert_integer_out_of_range(
    result: Result<RelationalSelectResult, ExecuteError>,
    context: &str,
) {
    match result {
        Ok(_) => panic!("{context}: must raise integer out of range, not return rows"),
        Err(err) => assert!(
            err.to_string().contains("integer out of range"),
            "{context}: expected `integer out of range`, got: {err}"
        ),
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_checked_arithmetic_raises_integer_out_of_range_across_arith_kernels() {
    // PG raises `integer out of range` on int4 overflow; the on-device VM must too (checked
    // arithmetic, GPU-native, NO CPU fallback) — it must NEVER wrap and mis-answer. Construction
    // oracle (the exact int32 boundary, not a CPU re-implementation): the smallest positive `a`
    // making each op exceed i32::MAX = 2147483647 is
    //   a*a      -> 46341 (46341^2 = 2147488281; 46340^2 = 2147395600 is in range)
    //   a*100000 -> 21475 (2147500000; 21474*100000 = 2147400000 is in range)
    //   a*a*a    -> 1291  (1291^3 = 2151685171; 1290^3 = 2146689000 is in range)
    // The predicates cover the specialized two-column facade plus scalar-fold and buffer x buffer
    // VM programs. STRUCT-001GE routes the first through the same checked buffer-binary primitive;
    // every lowering must surface overflow. A safe row (a=2) is present too, proving the kernel
    // scans the whole payload and one overflowing row still aborts the query (PG per-row evaluation).

    // a*a -> specialized two-column facade over the typed VM.
    let a_sq = gt_zero(mul(column0(), column0()));
    if let Some(result) = eval_single_col_predicate("ovf_sq", &[2, 46341], &a_sq) {
        assert_integer_out_of_range(result, "a*a with a=46341");
    }

    // a*100000 -> scalar-fold VM kernel (literal operand folded, no constant buffer).
    let a_scaled = gt_zero(mul(column0(), ResidentExpr::Int4Literal(100_000)));
    if let Some(result) = eval_single_col_predicate("ovf_scaled", &[2, 21475], &a_scaled) {
        assert_integer_out_of_range(result, "a*100000 with a=21475");
    }

    // a*a*a -> buffer x buffer VM kernel (the outer multiply overflows; inner a*a is in range).
    let a_cube = gt_zero(mul(mul(column0(), column0()), column0()));
    if let Some(result) = eval_single_col_predicate("ovf_cube", &[2, 1291], &a_cube) {
        assert_integer_out_of_range(result, "a*a*a with a=1291");
    }

    // The widen+range-check is shared by all three ops, so also exercise ADD and SUB overflow (not
    // just MUL): the smallest-magnitude operands past the bound on each side.
    //   a + 2000000000 -> 2000000000 + 2000000000 = 4e9 > i32::MAX (the safe row a=2 stays in range).
    let a_added = gt_zero(add(column0(), ResidentExpr::Int4Literal(2_000_000_000)));
    if let Some(result) = eval_single_col_predicate("ovf_add", &[2, 2_000_000_000], &a_added) {
        assert_integer_out_of_range(result, "a + 2000000000 with a=2000000000");
    }
    //   a - 2000000000 -> -2e9 - 2e9 = -4e9 < i32::MIN (the safe row a=-2 -> -2000000002 is in range).
    let a_subbed = gt_zero(sub(column0(), ResidentExpr::Int4Literal(2_000_000_000)));
    if let Some(result) = eval_single_col_predicate("ovf_sub", &[-2, -2_000_000_000], &a_subbed) {
        assert_integer_out_of_range(result, "a - 2000000000 with a=-2000000000");
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_checked_arithmetic_admits_the_largest_in_range_values() {
    // Companion to the overflow test, pinning the OTHER side of each boundary: the largest in-range
    // operand must NOT error and must return the right rows — proving the 64-bit range check is exact
    // at the boundary (no false positive) and the wrapped store value is unused on the in-range path.
    // Closed-form: each value's square / scaled / cube is > 0, so `... > 0` selects every row, in
    // ascending row order.
    let a_sq = gt_zero(mul(column0(), column0()));
    if let Some(result) = eval_single_col_predicate("ok_sq", &[3, 46340], &a_sq) {
        let rows = result.expect("a*a at the boundary 46340 is in range, must not error");
        assert_eq!(
            rows.rows,
            vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(46340)]],
            "46340^2 is in range -> both rows returned"
        );
        assert_eq!(rows.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(rows.fallback_reason, None);
    }

    let a_scaled = gt_zero(mul(column0(), ResidentExpr::Int4Literal(100_000)));
    if let Some(result) = eval_single_col_predicate("ok_scaled", &[3, 21474], &a_scaled) {
        let rows = result.expect("a*100000 at the boundary 21474 is in range, must not error");
        assert_eq!(
            rows.rows,
            vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(21474)]],
            "21474*100000 is in range -> both rows returned"
        );
    }

    let a_cube = gt_zero(mul(mul(column0(), column0()), column0()));
    if let Some(result) = eval_single_col_predicate("ok_cube", &[3, 1290], &a_cube) {
        let rows = result.expect("a*a*a at the boundary 1290 is in range, must not error");
        assert_eq!(
            rows.rows,
            vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(1290)]],
            "1290^3 is in range -> both rows returned"
        );
    }
}
