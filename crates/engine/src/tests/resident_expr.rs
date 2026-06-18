//! Engine-level tests for the general GPU executor (`engine_expr`, Charter rule 2,
//! docs/architecture/17-general-gpu-executor.md): a `SELECT` filtered by a general expression tree
//! evaluated on the GPU via the device interpreter, with rows materialized on-device. Distinct from
//! the (frozen) enumerated `resident_probe` shape methods — here the unit of execution is an
//! expression, not a recognized shape.

use super::*;

use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_select_evaluates_arithmetic_predicate_and_materializes_rows() {
    // The general path runs `SELECT a FROM t WHERE (a + b) > K` — an ARITHMETIC predicate no
    // enumerated shape method can express — fully on the GPU (interpreter lowers the Expr to the
    // composed buffer->buffer primitive, then gathers the projected column at the surviving rows).
    // GPU-NATIVE oracle = CLOSED FORM (project rule, not a CPU re-implementation): with a[i]=b[i]=i,
    // a+b = 2*i is MONOTONE, so {i : 2*i > K} is the contiguous range [K/2+1, N); the projected a is
    // a[i]=i, so the result rows are exactly those indices. The range is distinct from "only a"
    // ({i:i>K} = [K+1,N)), so a passing assert proves the kernel evaluated the Add-then-Gt tree.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();

    const N: i32 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (a, b) VALUES {values}"))
        .unwrap();

    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // Projection only (the WHERE is supplied as the Expr, since the hand-rolled parser cannot yet
    // produce arithmetic predicates — that binding is the next step). Columns: a=0, b=1.
    let Command::Select(select) = parse_command("SELECT a FROM t").unwrap() else {
        unreachable!()
    };
    const K: i32 = 400;
    let predicate = ResidentExpr::Binary {
        op: ResidentBinaryOp::Gt,
        lhs: Box::new(ResidentExpr::Binary {
            op: ResidentBinaryOp::Add,
            lhs: Box::new(ResidentExpr::Column(0)),
            rhs: Box::new(ResidentExpr::Column(1)),
        }),
        rhs: Box::new(ResidentExpr::Int4Literal(K)),
    };

    let result = e
        .execute_resident_expr_select(&select, &predicate)
        .expect("general Expr select on GPU");

    let start = K / 2 + 1; // K even => 2i>K <=> i>=K/2+1 ; = 201
    let expected: Vec<Vec<SqlValue>> = (start..N).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(result.columns.len(), 1);
    assert_eq!(result.columns[0].name, "a");
    assert_eq!(
        result.rows, expected,
        "GPU (a+b)>{K} must materialize a-values for rows i in [{start}, {N}) — proves the \
         interpreter evaluated the arithmetic tree and gathered the projection on-device"
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    // Non-vacuity: 'only column a' (a>K) would start at K+1=401, a strictly smaller set.
    assert_ne!(
        result.rows.len(),
        (N - (K + 1)) as usize,
        "result must differ from a>K — else the interpreter ignored column b"
    );

    // The general path REJECTS a tree it cannot yet lower (a bare column is not a predicate); it does
    // not silently fall back to the CPU or to a shape method (Charter rules 1 + 2).
    let not_a_predicate = ResidentExpr::Column(0);
    assert!(
        e.execute_resident_expr_select(&select, &not_a_predicate)
            .is_err(),
        "unsupported Expr must be a hard error, not a silent fallback"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_select_evaluates_deep_arithmetic_tree_via_vm() {
    // A DEEPER arithmetic tree than the 2-col fast-path — `WHERE (a + b) * 2 - 5 > K` — routes
    // through the engine's Expr compiler -> device bytecode VM (not the peephole), evaluated and
    // materialized on the GPU. Closed-form oracle: a[i]=b[i]=i => value = 4*i - 5 (monotone), so
    // 4i-5 > K <=> i >= (K+5+3)/4 ; with K=395, 4i > 400 <=> i >= 101. Projected a[i]=i => the
    // result rows are exactly those indices.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();

    const N: i32 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (a, b) VALUES {values}"))
        .unwrap();

    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let Command::Select(select) = parse_command("SELECT a FROM t").unwrap() else {
        unreachable!()
    };
    // (a + b) * 2 - 5  : Binary(Sub, Binary(Mul, Binary(Add, a, b), 2), 5)
    const K: i32 = 395;
    let value_expr = ResidentExpr::Binary {
        op: ResidentBinaryOp::Sub,
        lhs: Box::new(ResidentExpr::Binary {
            op: ResidentBinaryOp::Mul,
            lhs: Box::new(ResidentExpr::Binary {
                op: ResidentBinaryOp::Add,
                lhs: Box::new(ResidentExpr::Column(0)),
                rhs: Box::new(ResidentExpr::Column(1)),
            }),
            rhs: Box::new(ResidentExpr::Int4Literal(2)),
        }),
        rhs: Box::new(ResidentExpr::Int4Literal(5)),
    };
    let predicate = ResidentExpr::Binary {
        op: ResidentBinaryOp::Gt,
        lhs: Box::new(value_expr),
        rhs: Box::new(ResidentExpr::Int4Literal(K)),
    };

    let result = e
        .execute_resident_expr_select(&select, &predicate)
        .expect("deep arithmetic tree via VM on GPU");

    let start = 101; // 4i - 5 > 395 <=> i >= 101
    let expected: Vec<Vec<SqlValue>> = (start..N).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        result.rows, expected,
        "GPU (a+b)*2-5 > {K} must materialize a-values for rows i in [{start}, {N})"
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));

    // Literal on the LEFT flips the comparison: `K < (a+b)*2 - 5` is the same predicate.
    let predicate_flipped = ResidentExpr::Binary {
        op: ResidentBinaryOp::Lt,
        lhs: Box::new(ResidentExpr::Int4Literal(K)),
        rhs: Box::new(ResidentExpr::Binary {
            op: ResidentBinaryOp::Sub,
            lhs: Box::new(ResidentExpr::Binary {
                op: ResidentBinaryOp::Mul,
                lhs: Box::new(ResidentExpr::Binary {
                    op: ResidentBinaryOp::Add,
                    lhs: Box::new(ResidentExpr::Column(0)),
                    rhs: Box::new(ResidentExpr::Column(1)),
                }),
                rhs: Box::new(ResidentExpr::Int4Literal(2)),
            }),
            rhs: Box::new(ResidentExpr::Int4Literal(5)),
        }),
    };
    let flipped = e
        .execute_resident_expr_select(&select, &predicate_flipped)
        .expect("flipped literal-on-left predicate via VM");
    assert_eq!(
        flipped.rows, expected,
        "K < (a+b)*2-5 must equal (a+b)*2-5 > K (comparison flipped)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_select_evaluates_column_vs_column_predicates() {
    // Column-vs-column / expr-vs-expr predicates (the comparison RHS is an expression, not a literal)
    // through the engine's col-vs-col VM. Closed-form oracle: a[i]=i, b[i]=N-1-i (strictly decreasing,
    // never ties a). `a < b` <=> 2i < N-1 ; `a*2 > b` <=> 3i > N-1. Projected a[i]=i => result rows
    // are the matching indices.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();

    const N: i32 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", N - 1 - i));
    }
    e.execute_text(2, &format!("INSERT INTO t (a, b) VALUES {values}"))
        .unwrap();

    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let Command::Select(select) = parse_command("SELECT a FROM t").unwrap() else {
        unreachable!()
    };

    // a < b  : i < N-1-i <=> 2i < N-1 <=> i in [0, 300) for N=600.
    let a_lt_b = ResidentExpr::Binary {
        op: ResidentBinaryOp::Lt,
        lhs: Box::new(ResidentExpr::Column(0)),
        rhs: Box::new(ResidentExpr::Column(1)),
    };
    let lt = e
        .execute_resident_expr_select(&select, &a_lt_b)
        .expect("a < b col-vs-col on GPU");
    let lt_expected: Vec<Vec<SqlValue>> = (0..300).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(lt.rows, lt_expected, "a < b <=> i in [0, 300)");

    // a*2 > b : 2i > N-1-i <=> 3i > N-1 <=> i >= 200 for N=600.
    let a2_gt_b = ResidentExpr::Binary {
        op: ResidentBinaryOp::Gt,
        lhs: Box::new(ResidentExpr::Binary {
            op: ResidentBinaryOp::Mul,
            lhs: Box::new(ResidentExpr::Column(0)),
            rhs: Box::new(ResidentExpr::Int4Literal(2)),
        }),
        rhs: Box::new(ResidentExpr::Column(1)),
    };
    let gt = e
        .execute_resident_expr_select(&select, &a2_gt_b)
        .expect("a*2 > b expr-vs-col on GPU");
    let gt_expected: Vec<Vec<SqlValue>> = (200..N).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(gt.rows, gt_expected, "a*2 > b <=> i >= 200");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_select_evaluates_boolean_and_or_ne_predicates() {
    // Boolean predicates (AND / OR / Ne) through the engine's mask-based predicate VM, evaluated and
    // materialized on the GPU. Closed-form oracle over a[i]=i: matching sets are explicit index
    // ranges.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT)").unwrap();

    const N: i32 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (a) VALUES {values}"))
        .unwrap();

    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) = parse_command("SELECT a FROM t").unwrap() else {
        unreachable!()
    };
    let cmp = |op: ResidentBinaryOp, k: i32| ResidentExpr::Binary {
        op,
        lhs: Box::new(ResidentExpr::Column(0)),
        rhs: Box::new(ResidentExpr::Int4Literal(k)),
    };

    // a > 200 AND a < 400  ->  i in [201, 400).
    let and_pred = ResidentExpr::Binary {
        op: ResidentBinaryOp::And,
        lhs: Box::new(cmp(ResidentBinaryOp::Gt, 200)),
        rhs: Box::new(cmp(ResidentBinaryOp::Lt, 400)),
    };
    let got_and = e
        .execute_resident_expr_select(&select, &and_pred)
        .expect("a>200 AND a<400 on GPU");
    let and_expected: Vec<Vec<SqlValue>> = (201..400).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(got_and.rows, and_expected, "a>200 AND a<400 <=> [201, 400)");

    // a < 100 OR a > 500  ->  [0, 100) U [501, 600).
    let or_pred = ResidentExpr::Binary {
        op: ResidentBinaryOp::Or,
        lhs: Box::new(cmp(ResidentBinaryOp::Lt, 100)),
        rhs: Box::new(cmp(ResidentBinaryOp::Gt, 500)),
    };
    let got_or = e
        .execute_resident_expr_select(&select, &or_pred)
        .expect("a<100 OR a>500 on GPU");
    let mut or_expected: Vec<Vec<SqlValue>> = (0..100).map(|i| vec![SqlValue::Int4(i)]).collect();
    or_expected.extend((501..N).map(|i| vec![SqlValue::Int4(i)]));
    assert_eq!(got_or.rows, or_expected, "a<100 OR a>500 <=> [0,100) U [501,600)");

    // a != 300  ->  everything except 300.
    let ne_pred = cmp(ResidentBinaryOp::Ne, 300);
    let got_ne = e
        .execute_resident_expr_select(&select, &ne_pred)
        .expect("a != 300 on GPU");
    let mut ne_expected: Vec<Vec<SqlValue>> = (0..300).map(|i| vec![SqlValue::Int4(i)]).collect();
    ne_expected.extend((301..N).map(|i| vec![SqlValue::Int4(i)]));
    assert_eq!(got_ne.rows, ne_expected, "a != 300 <=> all rows but 300");
}

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
    let mut e = Engine::new_local();
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
fn assert_integer_out_of_range(result: Result<RelationalSelectResult, ExecuteError>, context: &str) {
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
    // Each predicate routes to a DIFFERENT arithmetic kernel — the 2-col peephole
    // (gpu_db_resident_i32_binary_elementwise), the scalar-fold VM (gpu_db_buffer_i32_binary_scalar),
    // and the buffer x buffer VM (gpu_db_buffer_i32_binary) — so an overflowing row in any of the
    // three must surface the error. A safe row (a=2) is present too, proving the kernel scans the
    // whole payload and the single overflowing row still aborts the query (PG per-row evaluation).

    // a*a -> 2-col peephole elementwise kernel.
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

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int8_predicates() {
    // int8 (BIGINT) end to end on the general GPU executor from SQL text (the type matrix, doc 19):
    // scalar comparison, column-vs-column with values ABOVE i32::MAX (proving genuine 64-bit), the Ne
    // operator, and both int8 + int4 projection. Plus the unsupported-int8-shape hard errors.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, big BIGINT, big2 BIGINT, small BIGINT)")
        .unwrap();

    const N: i64 = 600;
    const BASE: i64 = 4_000_000_000; // > i32::MAX (2_147_483_647)
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {}, {}, {i})", BASE + i, BASE + (N - 1 - i)));
    }
    e.execute_text(2, &format!("INSERT INTO t (a, big, big2, small) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // (1) int8 scalar comparison + int8 projection: small[i]=i, small > 400 => [401, 600).
    let scalar = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE small > 400")
        .expect("int8 scalar comparison on GPU");
    let scalar_expected: Vec<Vec<SqlValue>> = (401..N).map(|i| vec![SqlValue::Int8(i)]).collect();
    assert_eq!(
        scalar.rows, scalar_expected,
        "small > 400 => small=i for i in [401, 600)"
    );
    assert_eq!(scalar.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(scalar.fallback_reason, None);

    // (2) int8 column-vs-column + 64-bit + i64 projection: big[i]=BASE+i, big2[i]=BASE+(N-1-i);
    // big < big2 <=> i < N-1-i <=> [0, 300). Projected big = BASE+i (values above i32::MAX).
    let cols = e
        .execute_resident_expr_select_sql("SELECT big FROM t WHERE big < big2")
        .expect("int8 col-vs-col on GPU");
    let cols_expected: Vec<Vec<SqlValue>> =
        (0..300).map(|i| vec![SqlValue::Int8(BASE + i)]).collect();
    assert_eq!(
        cols.rows, cols_expected,
        "big < big2 => big=BASE+i for i in [0, 300) (64-bit values)"
    );

    // (3) int8 Ne predicate + int4 projection (the projection type is independent of the predicate
    // type): small <> 300 => every row but i=300, projecting the int4 column a.
    let ne = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE small <> 300")
        .expect("int8 Ne on GPU");
    let mut ne_expected: Vec<Vec<SqlValue>> =
        (0..300).map(|i| vec![SqlValue::Int4(i)]).collect();
    ne_expected.extend((301..N as i32).map(|i| vec![SqlValue::Int4(i)]));
    assert_eq!(
        ne.rows, ne_expected,
        "small <> 300 => all rows but i=300, projecting int4 a"
    );

    // (4) A MIXED int4/int8 comparison is a HARD error (never a silent mis-answer). int8 arithmetic
    // is now supported — see gpu_execute_resident_expr_select_sql_runs_int8_arithmetic.
    assert!(
        e.execute_resident_expr_select_sql("SELECT big FROM t WHERE a < big")
            .is_err(),
        "mixed int4/int8 comparison -> hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int8_arithmetic() {
    // int8 (BIGINT) ARITHMETIC on the general GPU executor (the type matrix, doc 19): the i64 buffer
    // VM evaluates int8 arith trees (add/sub/mul, col-vs-col + scalar) with values ABOVE i32::MAX.
    // Closed-form oracle: a[i]=BASE+i, b[i]=BASE, c[i]=2*BASE+300, small[i]=i.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT, small BIGINT)")
        .unwrap();

    const N: i64 = 600;
    const BASE: i64 = 4_000_000_000; // > i32::MAX
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({}, {BASE}, {}, {i})", BASE + i, 2 * BASE + 300));
    }
    e.execute_text(2, &format!("INSERT INTO t (a, b, c, small) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // col-vs-col ADD: a + b > c <=> (2*BASE+i) > (2*BASE+300) <=> i > 300 => [301, 600). 64-bit, no
    // overflow (2*BASE ~ 8e9 < i64::MAX). Projects a = BASE+i.
    let added = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE a + b > c")
        .expect("int8 a+b>c on GPU");
    let added_expected: Vec<Vec<SqlValue>> =
        (301..N).map(|i| vec![SqlValue::Int8(BASE + i)]).collect();
    assert_eq!(added.rows, added_expected, "a+b>c => a=BASE+i for i in [301, 600)");
    assert_eq!(added.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(added.fallback_reason, None);

    // SUB then scalar compare: a - b > 200 <=> i > 200 => [201, 600). Projects small = i.
    let subbed = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE a - b > 200")
        .expect("int8 a-b>200 on GPU");
    let subbed_expected: Vec<Vec<SqlValue>> = (201..N).map(|i| vec![SqlValue::Int8(i)]).collect();
    assert_eq!(subbed.rows, subbed_expected, "a-b>200 => small=i for i in [201, 600)");

    // scalar MUL: small * 2 > 800 <=> i > 400 => [401, 600). Projects small = i.
    let scaled = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE small * 2 > 800")
        .expect("int8 small*2>800 on GPU");
    let scaled_expected: Vec<Vec<SqlValue>> = (401..N).map(|i| vec![SqlValue::Int8(i)]).collect();
    assert_eq!(scaled.rows, scaled_expected, "small*2>800 => small=i for i in [401, 600)");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_raises_int8_integer_out_of_range_on_overflow() {
    // int8 arithmetic overflow raises Postgres `integer out of range` (the i64 buffer VM's checked mul
    // via mul.hi vs the sign-extension of mul.lo). Construction oracle (the int64 boundary):
    // 3037000500^2 = 9223372037000250000 > i64::MAX (9223372036854775807); 3037000499^2 =
    // 9223372030926249001 is in range.
    let overflow = run_int8_square_gt_zero(3_037_000_500);
    if let Some(result) = overflow {
        match result {
            Ok(_) => panic!("a*a over a=3037000500 overflows int64 -> must raise bigint out of range"),
            Err(err) => assert!(
                err.to_string().contains("bigint out of range"),
                "int8 overflow must be PG's `bigint out of range` (not `integer`), got: {err}"
            ),
        }
    }

    // The largest in-range square does NOT error and returns the rows (a*a > 0 for both rows).
    if let Some(result) = run_int8_square_gt_zero(3_037_000_499) {
        let rows = result.expect("a*a at the int64 boundary 3037000499 is in range, must not error");
        assert_eq!(
            rows.rows,
            vec![vec![SqlValue::Int8(2)], vec![SqlValue::Int8(3_037_000_499)]],
            "a*a>0 in range -> both rows"
        );
    }
}

/// Build a single-BIGINT-column table `t(a) = [2, boundary]`, push a GPU snapshot, and run
/// `SELECT a FROM t WHERE a * a > 0` on the general executor. Returns `None` off-GPU.
fn run_int8_square_gt_zero(
    boundary: i64,
) -> Option<Result<RelationalSelectResult, ExecuteError>> {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a BIGINT)").unwrap();
    e.execute_text(2, &format!("INSERT INTO t (a) VALUES (2), ({boundary})"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    snapshot.device_memory_proof.as_ref()?;
    Some(e.execute_resident_expr_select_sql("SELECT a FROM t WHERE a * a > 0"))
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int8_boolean_predicates() {
    // REGRESSION (the audit's P0): int8 AND/OR predicates must run on the i64 VM, NOT silently route
    // to the i32 VM (which read int8 columns at the wrong 4-byte stride -> garbage rows). The same
    // routing gap also bypassed the mixed-int4/int8 guard, so a mixed AND must still hard-error.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, small BIGINT, big BIGINT, big2 BIGINT)")
        .unwrap();

    const N: i64 = 600;
    const BASE: i64 = 4_000_000_000; // > i32::MAX
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {i}, {}, {})", BASE + i, BASE + (N - 1 - i)));
    }
    e.execute_text(2, &format!("INSERT INTO t (a, small, big, big2) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // int8 AND (i32-range literals over an int8 column): small > 200 AND small < 400 => [201, 400).
    // The buggy i32-VM route read `small` at a 4-byte stride and returned a garbled set.
    let and_rows = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE small > 200 AND small < 400")
        .expect("int8 AND on GPU");
    let and_expected: Vec<Vec<SqlValue>> = (201..400).map(|i| vec![SqlValue::Int8(i)]).collect();
    assert_eq!(and_rows.rows, and_expected, "small>200 AND small<400 => [201, 400)");
    assert_eq!(and_rows.executed_target, DeviceTarget::Gpu(0));

    // int8 OR: small < 100 OR small > 500 => [0,100) U [501,600).
    let or_rows = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE small < 100 OR small > 500")
        .expect("int8 OR on GPU");
    let mut or_expected: Vec<Vec<SqlValue>> = (0..100).map(|i| vec![SqlValue::Int8(i)]).collect();
    or_expected.extend((501..N).map(|i| vec![SqlValue::Int8(i)]));
    assert_eq!(or_rows.rows, or_expected, "small<100 OR small>500 => [0,100) U [501,600)");

    // 64-bit AND over two int8 columns (values above i32::MAX): big > 100 AND big2 > 100 => all rows.
    // An i32-stride read of big/big2 would NOT yield all rows, so this pins the genuine 64-bit read.
    let big_and = e
        .execute_resident_expr_select_sql("SELECT big FROM t WHERE big > 100 AND big2 > 100")
        .expect("int8 64-bit AND on GPU");
    let big_and_expected: Vec<Vec<SqlValue>> =
        (0..N).map(|i| vec![SqlValue::Int8(BASE + i)]).collect();
    assert_eq!(big_and.rows, big_and_expected, "big>100 AND big2>100 => all rows (64-bit)");

    // MIXED int4/int8 inside AND must hard-error (the routing gap bypassed the mixed-type guard).
    assert!(
        e.execute_resident_expr_select_sql("SELECT a FROM t WHERE big > 5 AND a < 3")
            .is_err(),
        "mixed int4/int8 AND -> hard error, not a wrong answer"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_comparisons() {
    // numeric (NUMERIC / i128) comparison + projection end-to-end from SQL (the type matrix, doc 19).
    // price[i] = i.50 (NUMERIC(10,2)), cost[i] = (N-1-i).50. Closed-form oracles; the i128 signedness
    // is proven separately in the execution-crate primitive test.
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE t (price NUMERIC(10,2), cost NUMERIC(10,2), label INT)",
    )
    .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        // price = i.50, cost = (N-1-i).50, label = i
        values.push_str(&format!("({i}.50, {}.50, {i})", N - 1 - i));
    }
    e.execute_text(2, &format!("INSERT INTO t (price, cost, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let num = |whole: i64| SqlValue::Numeric(Decimal128::new(i128::from(whole) * 100 + 50, 2)); // whole.50

    // literal at the column scale: price > 10.50 => i+0.50 > 10.50 => i > 10 => [11, N). Project price.
    let gt = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price > 10.50")
        .expect("price > 10.50 on GPU");
    let gt_expected: Vec<Vec<SqlValue>> = (11..N).map(|i| vec![num(i)]).collect();
    assert_eq!(gt.rows, gt_expected, "price > 10.50 => i.50 for i in [11, 600)");
    assert_eq!(gt.executed_target, DeviceTarget::Gpu(0));

    // integer literal coerced to numeric: price < 5 => i+0.50 < 5 => i <= 4 => [0, 5).
    let lt_int = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price < 5")
        .expect("price < 5 on GPU");
    let lt_int_expected: Vec<Vec<SqlValue>> = (0..5).map(|i| vec![num(i)]).collect();
    assert_eq!(lt_int.rows, lt_int_expected, "price < 5 (int coerced) => [0, 5)");

    // lower-scale literal rescales UP exactly: price > 10.5 (scale 1) == price > 10.50 => [11, N).
    let gt_low = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price > 10.5")
        .expect("price > 10.5 on GPU");
    assert_eq!(gt_low.rows, gt_expected, "price > 10.5 (scale 1) == price > 10.50");

    // trailing zeros are insignificant: price > 10.500 (written scale 3) == price > 10.50 => [11, N).
    let gt_trailing = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price > 10.500")
        .expect("price > 10.500 on GPU");
    assert_eq!(
        gt_trailing.rows, gt_expected,
        "price > 10.500 (trailing zeros) == price > 10.50"
    );

    // col-vs-col, equal scale: price < cost => i+0.50 < (N-1-i)+0.50 => 2i < N-1 => [0, 300).
    let cols = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price < cost")
        .expect("price < cost on GPU");
    let cols_expected: Vec<Vec<SqlValue>> = (0..300).map(|i| vec![num(i)]).collect();
    assert_eq!(cols.rows, cols_expected, "price < cost => [0, 300)");

    // a literal FINER than the column now rescales the column UP (cross-scale): price > 10.555
    // (scale 3) <=> i+0.50 > 10.555 <=> i >= 11 => [11, N).
    let finer = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price > 10.555")
        .expect("price > 10.555 (cross-scale) on GPU");
    let finer_expected: Vec<Vec<SqlValue>> = (11..N).map(|i| vec![num(i)]).collect();
    assert_eq!(
        finer.rows, finer_expected,
        "price > 10.555 (finer literal, cross-scale) => [11, N)"
    );

    // REJECTION — a mixed numeric vs an int4 column is a hard error, never wrong rows.
    assert!(
        e.execute_resident_expr_select_sql("SELECT price FROM t WHERE price > label")
            .is_err(),
        "mixed numeric/int4 column => hard error"
    );
    // numeric AND/OR now lowers via the i128 mask VM: price > 1.00 AND price < 100.00 (price = i.50)
    // <=> 0 < i < 100 => [1, 100). Project price.
    let and = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price > 1.00 AND price < 100.00")
        .expect("price>1.00 AND price<100.00 on GPU");
    let and_expected: Vec<Vec<SqlValue>> = (1..100).map(|i| vec![num(i)]).collect();
    assert_eq!(
        and.rows, and_expected,
        "price>1.00 AND price<100.00 => price for i in [1, 100)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_arithmetic() {
    // numeric (i128) CHECKED add/sub arithmetic end-to-end from SQL (the type matrix, doc 19).
    // price[i]=i.50, cost[i]=i.25 (NUMERIC(10,2)), label[i]=i. Closed-form; mul + mixed are rejected.
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE t (price NUMERIC(10,2), cost NUMERIC(10,2), label INT)",
    )
    .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}.50, {i}.25, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (price, cost, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // col+col add then scalar compare: price+cost = 2i+0.75; > 100 => 200i+75 > 10000 => i >= 50.
    let added = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price + cost > 100")
        .expect("price+cost>100 on GPU");
    let added_expected: Vec<Vec<SqlValue>> = (50..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(added.rows, added_expected, "price+cost>100 => label in [50, 600)");
    assert_eq!(added.executed_target, DeviceTarget::Gpu(0));

    // scalar sub then compare: price-5 = (i-5)+0.50; > 100 => 100i-450 > 10000 => i >= 105.
    let subbed = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price - 5 > 100")
        .expect("price-5>100 on GPU");
    let subbed_expected: Vec<Vec<SqlValue>> =
        (105..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(subbed.rows, subbed_expected, "price-5>100 => label in [105, 600)");

    // buffer-vs-buffer (arith on the left, column on the right): price+cost > price <=> cost > 0 => all.
    let cmp_buffers = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price + cost > price")
        .expect("price+cost>price on GPU");
    let all_expected: Vec<Vec<SqlValue>> = (0..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(cmp_buffers.rows, all_expected, "price+cost>price <=> cost>0 => all rows");

    // REJECTION — a mixed numeric + int4 column in arithmetic is a hard error, never wrong rows.
    // (integer-literal multiply is now supported — see ..._runs_numeric_multiply.)
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE price + label > 100")
            .is_err(),
        "mixed numeric + int4 column in arithmetic => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_multiply() {
    // numeric (i128) CHECKED multiply by an INTEGER literal end-to-end from SQL (the type matrix,
    // doc 19): price*2 = 2i+1.00 (mantissa 200i+100). Fractional + column*column multipliers (which
    // change the result scale) are rejected.
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE t (price NUMERIC(10,2), cost NUMERIC(10,2), label INT)",
    )
    .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}.50, {i}.25, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (price, cost, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // price * 2 > 100 : (200i+100) > 10000 => 200i > 9900 => i >= 50 => [50, N).
    let mul = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * 2 > 100")
        .expect("price*2>100 on GPU");
    let mul_expected: Vec<Vec<SqlValue>> = (50..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(mul.rows, mul_expected, "price*2>100 => label in [50, 600)");
    assert_eq!(mul.executed_target, DeviceTarget::Gpu(0));

    // commutative: 2 * price > 100 is the same set.
    let mul_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE 2 * price > 100")
        .expect("2*price>100 on GPU");
    assert_eq!(mul_left.rows, mul_expected, "2*price>100 == price*2>100 (commutative)");

    // fractional-literal multiply: price*1.5 (result scale 2+1=3): (1500i+750) > 100000 => i >= 67.
    let frac = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * 1.5 > 100")
        .expect("price*1.5>100 on GPU");
    let frac_expected: Vec<Vec<SqlValue>> = (67..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(frac.rows, frac_expected, "price*1.5>100 => label in [67, 600)");

    // column*column multiply: price*cost (result scale 2+2=4): (100i+50)(100i+25) > 1000000 => i >= 10.
    let cols = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * cost > 100")
        .expect("price*cost>100 on GPU");
    let cols_expected: Vec<Vec<SqlValue>> = (10..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(cols.rows, cols_expected, "price*cost>100 => label in [10, 600)");

    // cross-scale arith-vs-arith: price*cost (scale 4) > price (scale 2): rescale price up, then
    // (100i+25) > 100 <=> i >= 1 => [1, N).
    let cross = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * cost > price")
        .expect("price*cost>price (cross-scale) on GPU");
    let cross_expected: Vec<Vec<SqlValue>> = (1..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(cross.rows, cross_expected, "price*cost > price (cross-scale) => [1, 600)");

    // REJECTION — a mixed numeric * int4 column is a hard error, never wrong rows.
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE price * label > 100")
            .is_err(),
        "mixed numeric * int4 column => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_cross_scale_numeric_comparisons() {
    // CROSS-SCALE numeric comparison (the type matrix, doc 19): operands of different scales are
    // rescaled UP to the common (max) scale on-device (mantissa * 10^k) before comparing.
    // p2 = i.50 (NUMERIC(10,2)), p4 = (2i).0000 (NUMERIC(10,4)), label = i.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (p2 NUMERIC(10,2), p4 NUMERIC(10,4), label INT)")
        .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}.50, {}.0000, {i})", 2 * i));
    }
    e.execute_text(2, &format!("INSERT INTO t (p2, p4, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // cross-scale column-vs-column: p2 > p4 <=> i+0.50 > 2i <=> i < 0.5 => only row 0.
    let cols = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 > p4")
        .expect("p2>p4 on GPU");
    assert_eq!(
        cols.rows,
        vec![vec![SqlValue::Int4(0)]],
        "p2>p4 (cross-scale col-vs-col) => row 0 only"
    );
    assert_eq!(cols.executed_target, DeviceTarget::Gpu(0));

    // cross-scale column-vs-FINER-literal: p2 > 1.555 <=> i+0.50 > 1.555 <=> i >= 2 => [2, N).
    let lit = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 > 1.555")
        .expect("p2>1.555 on GPU");
    let lit_expected: Vec<Vec<SqlValue>> = (2..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(lit.rows, lit_expected, "p2>1.555 (finer literal) => [2, 600)");

    // literal on the left: 1.555 < p2 is the same set.
    let lit_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE 1.555 < p2")
        .expect("1.555<p2 on GPU");
    assert_eq!(lit_left.rows, lit_expected, "1.555<p2 == p2>1.555");

    // same-scale still uses the resident peephole (regression): p2 > 10.50 => [11, N).
    let same = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 > 10.50")
        .expect("p2>10.50 on GPU");
    let same_expected: Vec<Vec<SqlValue>> = (11..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(same.rows, same_expected, "p2>10.50 (same scale) => [11, 600)");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_cross_scale_numeric_add_sub() {
    // CROSS-SCALE numeric ADD/SUB (the type matrix, doc 19): operands of different scales are rescaled
    // UP to the common (max) scale before the buffer add/sub. p2 = i.50 (NUMERIC(10,2)), p4 = (2i).2500
    // (NUMERIC(10,4)), label = i.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (p2 NUMERIC(10,2), p4 NUMERIC(10,4), label INT)")
        .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}.50, {}.2500, {i})", 2 * i));
    }
    e.execute_text(2, &format!("INSERT INTO t (p2, p4, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // cross-scale ADD (col+col): p2 + p4 = 3i + 0.75 (scale 4); > 100 => 3i > 99.25 => i >= 34.
    let add = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 + p4 > 100")
        .expect("p2+p4>100 on GPU");
    let add_expected: Vec<Vec<SqlValue>> = (34..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(add.rows, add_expected, "p2+p4>100 (cross-scale add) => [34, 600)");
    assert_eq!(add.executed_target, DeviceTarget::Gpu(0));

    // cross-scale SUB (col-col): p4 - p2 = i - 0.25 (scale 4); > 100 => i >= 101.
    let sub = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p4 - p2 > 100")
        .expect("p4-p2>100 on GPU");
    let sub_expected: Vec<Vec<SqlValue>> = (101..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(sub.rows, sub_expected, "p4-p2>100 (cross-scale sub) => [101, 600)");

    // cross-scale add with a FINER literal: p2 + 0.0001 = i.5001 (scale 4); > 50.5 => i >= 50.
    let lit = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 + 0.0001 > 50.5")
        .expect("p2+0.0001>50.5 on GPU");
    let lit_expected: Vec<Vec<SqlValue>> = (50..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(lit.rows, lit_expected, "p2 + 0.0001 (finer literal) > 50.5 => [50, 600)");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_and_or() {
    // numeric AND/OR (the type matrix, doc 19): each comparison -> a mask via the i128 VM, MaskBinary
    // combines, terminal compact. price = i.50 (NUMERIC(10,2)), cost = i.2500 (NUMERIC(10,4)).
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE t (price NUMERIC(10,2), cost NUMERIC(10,4), label INT)",
    )
    .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}.50, {i}.2500, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (price, cost, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // AND: price > 10.50 AND price < 100.50 <=> 10 < i < 100 => [11, 100).
    let and = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE price > 10.50 AND price < 100.50",
        )
        .expect("AND on GPU");
    let and_expected: Vec<Vec<SqlValue>> =
        (11i64..100).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(and.rows, and_expected, "price>10.50 AND price<100.50 => [11, 100)");
    assert_eq!(and.executed_target, DeviceTarget::Gpu(0));

    // OR: price < 5.50 OR price > 595.50 => [0,5) U [596, N).
    let or = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE price < 5.50 OR price > 595.50",
        )
        .expect("OR on GPU");
    let or_expected: Vec<Vec<SqlValue>> = (0..5)
        .chain(596..N)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();
    assert_eq!(or.rows, or_expected, "price<5.50 OR price>595.50 => [0,5) U [596,600)");

    // CROSS-SCALE AND (price scale 2, cost scale 4 -- each comparison rescales independently):
    // price > 10.50 AND cost > 50.2500 <=> i>10 AND i>50 => [51, N).
    let cross = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE price > 10.50 AND cost > 50.2500",
        )
        .expect("cross-scale AND on GPU");
    let cross_expected: Vec<Vec<SqlValue>> = (51..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        cross.rows, cross_expected,
        "price>10.50 AND cost>50.2500 (cross-scale) => [51, 600)"
    );

    // NESTED: (price > 10.50 AND price < 100.50) OR price > 595.50 => [11,100) U [596, N).
    let nested = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE (price > 10.50 AND price < 100.50) OR price > 595.50",
        )
        .expect("nested AND/OR on GPU");
    let nested_expected: Vec<Vec<SqlValue>> = (11..100)
        .chain(596..N)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();
    assert_eq!(
        nested.rows, nested_expected,
        "(price>10.50 AND price<100.50) OR price>595.50 => [11,100) U [596,600)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_text_equality() {
    // Text equality on the general GPU executor (the type matrix, doc 19): byte-wise = / <>. The 7-row
    // (ODD) count places the text offsets section at a 4-mod-8 byte offset (8 header + 7*4 int4 = 36),
    // exercising the 2x 4-byte offset loads end to end through the real residency builder.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (name TEXT, label INT)").unwrap();
    let names = ["alice", "bob", "alice", "carol", "bob", "alice", "dave"];
    let mut values = String::new();
    for (i, n) in names.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("('{n}', {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (name, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // = 'alice' -> rows 0,2,5
    let eq = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name = 'alice'")
        .expect("name = 'alice' on GPU");
    assert_eq!(
        eq.rows,
        vec![
            vec![SqlValue::Int4(0)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(5)]
        ],
        "name = 'alice' => rows 0,2,5"
    );
    assert_eq!(eq.executed_target, DeviceTarget::Gpu(0));

    // <> 'alice' -> rows 1,3,4,6
    let ne = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name <> 'alice'")
        .expect("name <> 'alice' on GPU");
    assert_eq!(
        ne.rows,
        vec![
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(3)],
            vec![SqlValue::Int4(4)],
            vec![SqlValue::Int4(6)]
        ],
        "name <> 'alice' => rows 1,3,4,6"
    );

    // literal on the LEFT (equality is symmetric): 'bob' = name -> rows 1,4
    let lit_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE 'bob' = name")
        .expect("'bob' = name on GPU");
    assert_eq!(
        lit_left.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(4)]],
        "'bob' = name => rows 1,4"
    );

    // no match
    let none = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name = 'zzz'")
        .expect("name = 'zzz' on GPU");
    assert!(none.rows.is_empty(), "name = 'zzz' matches nothing");

    // REJECTIONS -- hard errors, never wrong rows:
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE name < 'bob'")
            .is_err(),
        "text inequality => collation-sort-key follow-on"
    );
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE 'a%'")
            .is_err(),
        "text LIKE => Slice B follow-on"
    );
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE name = label")
            .is_err(),
        "mixed text/int => hard error"
    );
}
