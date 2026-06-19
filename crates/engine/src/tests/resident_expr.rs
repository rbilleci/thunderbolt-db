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

    // REJECTIONS -- hard errors, never wrong rows (LIKE is now supported; see the text_like test):
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE name < 'bob'")
            .is_err(),
        "text inequality => collation-sort-key follow-on"
    );
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE name = label")
            .is_err(),
        "mixed text/int => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_text_like() {
    // Text LIKE on the general GPU executor (the type matrix, doc 19): general %/_ backtracking match.
    // 7 rows (ODD) -> text offsets at a 4-mod-8 byte offset. Includes the `\_` escape vs a bare `_`.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (name TEXT, label INT)").unwrap();
    // "a_b" stores a literal underscore; "axb" distinguishes the `_` wildcard from the `\_` escape.
    let names = ["alice", "alicia", "bob", "alfred", "carol", "a_b", "axb"];
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

    let labels = |idx: &[i32]| -> Vec<Vec<SqlValue>> {
        idx.iter().map(|i| vec![SqlValue::Int4(*i)]).collect()
    };

    // prefix: 'al%' -> alice, alicia, alfred
    let prefix = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE 'al%'")
        .expect("LIKE 'al%' on GPU");
    assert_eq!(prefix.rows, labels(&[0, 1, 3]), "LIKE 'al%' => alice/alicia/alfred");
    assert_eq!(prefix.executed_target, DeviceTarget::Gpu(0));

    // contains: '%i%' -> alice, alicia
    let contains = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE '%i%'")
        .expect("LIKE '%i%' on GPU");
    assert_eq!(contains.rows, labels(&[0, 1]), "LIKE '%i%' => alice/alicia");

    // single-char wildcard: 'a_b' -> a_b AND axb (the `_` matches any one char)
    let wild = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE 'a_b'")
        .expect("LIKE 'a_b' on GPU");
    assert_eq!(wild.rows, labels(&[5, 6]), "LIKE 'a_b' => a_b AND axb");

    // ESCAPED underscore: 'a\\_b' -> only the literal "a_b" (NOT axb)
    let escaped = e
        .execute_resident_expr_select_sql(r"SELECT label FROM t WHERE name LIKE 'a\_b'")
        .expect("LIKE 'a\\_b' on GPU");
    assert_eq!(escaped.rows, labels(&[5]), "LIKE 'a\\_b' => only the literal a_b");

    // exact (no wildcards) behaves like equality
    let exact = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE 'bob'")
        .expect("LIKE 'bob' on GPU");
    assert_eq!(exact.rows, labels(&[2]), "LIKE 'bob' => bob");

    // '%' matches every row
    let all = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE '%'")
        .expect("LIKE '%' on GPU");
    assert_eq!(all.rows, labels(&[0, 1, 2, 3, 4, 5, 6]), "LIKE '%' => all rows");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_date_comparisons() {
    // Date comparison on the general GPU executor (the type matrix, doc 19): a `date` is i32 days
    // since 2000-01-01, reusing the int4 residency section + the I32 VM. hire_date[i] = 2024-01-(i+1),
    // label = i. The string literal is coerced to a day count at lowering (like PG).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (hire_date DATE, label INT)")
        .unwrap();
    const N: i64 = 30; // 2024-01-01 .. 2024-01-30
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("('2024-01-{:02}', {i})", i + 1));
    }
    e.execute_text(2, &format!("INSERT INTO t (hire_date, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let labels = |range: std::ops::Range<i64>| -> Vec<Vec<SqlValue>> {
        range.map(|i| vec![SqlValue::Int4(i as i32)]).collect()
    };

    // = '2024-01-15' -> the day i+1 == 15 -> row 14
    let eq = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE hire_date = '2024-01-15'")
        .expect("hire_date = date on GPU");
    assert_eq!(eq.rows, vec![vec![SqlValue::Int4(14)]], "= '2024-01-15' => row 14");
    assert_eq!(eq.executed_target, DeviceTarget::Gpu(0));

    // > '2024-01-15' -> [15, 30)
    let gt = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE hire_date > '2024-01-15'")
        .expect("hire_date > date on GPU");
    assert_eq!(gt.rows, labels(15..N), "> '2024-01-15' => [15, 30)");

    // < '2024-01-10' -> [0, 9)
    let lt = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE hire_date < '2024-01-10'")
        .expect("hire_date < date on GPU");
    assert_eq!(lt.rows, labels(0..9), "< '2024-01-10' => [0, 9)");

    // literal on the LEFT: '2024-01-15' < hire_date -> [15, 30)
    let lit_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE '2024-01-15' < hire_date")
        .expect("date < hire_date on GPU");
    assert_eq!(lit_left.rows, labels(15..N), "'2024-01-15' < hire_date => [15, 30)");

    // projecting the DATE column yields SqlValue::Date(days)
    let proj = e
        .execute_resident_expr_select_sql("SELECT hire_date FROM t WHERE hire_date = '2024-01-15'")
        .expect("project date on GPU");
    let days = gpu_db_sql::datetime::parse_date("2024-01-15").expect("valid date");
    assert_eq!(
        proj.rows,
        vec![vec![SqlValue::Date(days)]],
        "projecting hire_date returns the date value"
    );

    // REJECTIONS -- hard errors, never wrong rows:
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE hire_date = 5")
            .is_err(),
        "date compared to an integer => hard error"
    );
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE hire_date = 'not-a-date'")
            .is_err(),
        "invalid date literal => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_timestamp_comparisons() {
    // Timestamp comparison on the general GPU executor (the type matrix, doc 19): a `timestamp` is
    // i64 microseconds since 2000-01-01, reusing the int8 section + the i64 compare kernels (the i64
    // micro literal exceeds the i32 VM scalar, so it uses expr_i64_compare_scalar_filter directly).
    // event_at[i] = 2024-01-15 i:00:00, created_at = constant 2024-01-15 12:00:00, label = i.
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE t (event_at TIMESTAMP, created_at TIMESTAMP, label INT)",
    )
    .unwrap();
    const N: i64 = 24; // hours 00:00:00 .. 23:00:00
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!(
            "('2024-01-15 {i:02}:00:00', '2024-01-15 12:00:00', {i})"
        ));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (event_at, created_at, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let labels = |range: std::ops::Range<i64>| -> Vec<Vec<SqlValue>> {
        range.map(|i| vec![SqlValue::Int4(i as i32)]).collect()
    };

    // = '2024-01-15 10:00:00' -> hour 10 -> row 10
    let eq = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE event_at = '2024-01-15 10:00:00'",
        )
        .expect("event_at = ts on GPU");
    assert_eq!(eq.rows, vec![vec![SqlValue::Int4(10)]], "= 10:00:00 => row 10");
    assert_eq!(eq.executed_target, DeviceTarget::Gpu(0));

    // > '2024-01-15 10:00:00' -> [11, 24)
    let gt = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE event_at > '2024-01-15 10:00:00'",
        )
        .expect("event_at > ts on GPU");
    assert_eq!(gt.rows, labels(11..N), "> 10:00:00 => [11, 24)");

    // fractional / sub-hour literal: < '2024-01-15 05:30:00' -> hours 0..5 -> [0, 6)
    let lt = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE event_at < '2024-01-15 05:30:00'",
        )
        .expect("event_at < ts on GPU");
    assert_eq!(lt.rows, labels(0..6), "< 05:30:00 => [0, 6)");

    // literal on the LEFT
    let lit_left = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE '2024-01-15 10:00:00' < event_at",
        )
        .expect("ts < event_at on GPU");
    assert_eq!(lit_left.rows, labels(11..N), "10:00:00 < event_at => [11, 24)");

    // col-vs-col: event_at > created_at (noon) -> hours > 12 -> [13, 24)
    let col_col = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE event_at > created_at")
        .expect("event_at > created_at on GPU");
    assert_eq!(col_col.rows, labels(13..N), "event_at > created_at (noon) => [13, 24)");

    // projecting the TIMESTAMP column yields SqlValue::Timestamp(micros)
    let proj = e
        .execute_resident_expr_select_sql(
            "SELECT event_at FROM t WHERE event_at = '2024-01-15 10:00:00'",
        )
        .expect("project timestamp on GPU");
    let micros = gpu_db_sql::datetime::parse_timestamp("2024-01-15 10:00:00").expect("valid ts");
    assert_eq!(
        proj.rows,
        vec![vec![SqlValue::Timestamp(micros)]],
        "projecting event_at returns the timestamp value"
    );

    // REJECTIONS -- hard errors, never wrong rows:
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE event_at = 5")
            .is_err(),
        "timestamp compared to an integer => hard error"
    );
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE event_at = 'not-a-ts'")
            .is_err(),
        "invalid timestamp literal => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_uuid_comparisons() {
    // UUID comparison on the general GPU executor (the type matrix, doc 19): a `uuid` is 16 raw bytes
    // in the i128 (16-byte) section, compared by an unsigned big-endian memcmp kernel (PG's uuid
    // order). id[i] = ...{i:02x} (last byte = i, so byte-wise ascending), peer = constant ...0a,
    // label = i.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (id UUID, peer UUID, label INT)")
        .unwrap();
    const N: i64 = 20;
    let uuid_for = |i: i64| format!("00000000-0000-0000-0000-0000000000{i:02x}");
    let peer = uuid_for(10);
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("('{}', '{peer}', {i})", uuid_for(i)));
    }
    e.execute_text(2, &format!("INSERT INTO t (id, peer, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let labels = |range: std::ops::Range<i64>| -> Vec<Vec<SqlValue>> {
        range.map(|i| vec![SqlValue::Int4(i as i32)]).collect()
    };

    // = '...0a' -> last byte 10 -> row 10
    let eq = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT label FROM t WHERE id = '{}'",
            uuid_for(10)
        ))
        .expect("id = uuid on GPU");
    assert_eq!(eq.rows, vec![vec![SqlValue::Int4(10)]], "= ...0a => row 10");
    assert_eq!(eq.executed_target, DeviceTarget::Gpu(0));

    // > '...0a' -> [11, 20)
    let gt = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT label FROM t WHERE id > '{}'",
            uuid_for(10)
        ))
        .expect("id > uuid on GPU");
    assert_eq!(gt.rows, labels(11..N), "> ...0a => [11, 20)");

    // < '...05' -> [0, 5)
    let lt = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT label FROM t WHERE id < '{}'",
            uuid_for(5)
        ))
        .expect("id < uuid on GPU");
    assert_eq!(lt.rows, labels(0..5), "< ...05 => [0, 5)");

    // literal on the LEFT
    let lit_left = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT label FROM t WHERE '{}' < id",
            uuid_for(10)
        ))
        .expect("uuid < id on GPU");
    assert_eq!(lit_left.rows, labels(11..N), "...0a < id => [11, 20)");

    // col-vs-col: id > peer (constant ...0a) -> [11, 20)
    let col_col = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE id > peer")
        .expect("id > peer on GPU");
    assert_eq!(col_col.rows, labels(11..N), "id > peer (...0a) => [11, 20)");

    // <> excludes only the equal row
    let ne = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT label FROM t WHERE id <> '{}'",
            uuid_for(10)
        ))
        .expect("id <> uuid on GPU");
    let mut expected_ne = labels(0..10);
    expected_ne.extend(labels(11..N));
    assert_eq!(ne.rows, expected_ne, "<> ...0a => all but row 10");

    // projecting the UUID column yields SqlValue::Uuid(bytes)
    let proj = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT id FROM t WHERE id = '{}'",
            uuid_for(10)
        ))
        .expect("project uuid on GPU");
    let bytes = gpu_db_sql::uuid::parse_uuid(&uuid_for(10)).expect("valid uuid");
    assert_eq!(
        proj.rows,
        vec![vec![SqlValue::Uuid(bytes)]],
        "projecting id returns the uuid value"
    );

    // REJECTIONS -- hard errors, never wrong rows:
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE id = 5")
            .is_err(),
        "uuid compared to an integer => hard error"
    );
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE id = 'not-a-uuid'")
            .is_err(),
        "invalid uuid literal => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int2_comparisons() {
    // smallint comparison on the general GPU executor (the type matrix, doc 19): a `smallint` is
    // stored WIDENED to i32 in the int4 section, so it reuses the i32 compare VM. sz[i] = i - 10
    // (so -10..9, exercising negatives + sign extension), peer = 0, label = i.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (sz SMALLINT, peer SMALLINT, label INT)")
        .unwrap();
    const N: i64 = 20;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({}, 0, {i})", i - 10));
    }
    e.execute_text(2, &format!("INSERT INTO t (sz, peer, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let labels = |range: std::ops::Range<i64>| -> Vec<Vec<SqlValue>> {
        range.map(|i| vec![SqlValue::Int4(i as i32)]).collect()
    };

    // = 0 -> i-10 = 0 -> row 10
    let eq = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE sz = 0")
        .expect("sz = 0 on GPU");
    assert_eq!(eq.rows, vec![vec![SqlValue::Int4(10)]], "= 0 => row 10");
    assert_eq!(eq.executed_target, DeviceTarget::Gpu(0));

    // > 0 -> [11, 20)
    let gt = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE sz > 0")
        .expect("sz > 0 on GPU");
    assert_eq!(gt.rows, labels(11..N), "> 0 => [11, 20)");

    // < -5 -> i < 5 -> [0, 5) (NEGATIVE literal + values -> sign extension is correct)
    let lt = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE sz < -5")
        .expect("sz < -5 on GPU");
    assert_eq!(lt.rows, labels(0..5), "< -5 => [0, 5)");

    // literal on the LEFT
    let lit_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE 0 < sz")
        .expect("0 < sz on GPU");
    assert_eq!(lit_left.rows, labels(11..N), "0 < sz => [11, 20)");

    // col-vs-col: sz > peer (constant 0) -> [11, 20)
    let col_col = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE sz > peer")
        .expect("sz > peer on GPU");
    assert_eq!(col_col.rows, labels(11..N), "sz > peer (0) => [11, 20)");

    // An out-of-int16 literal is a VALID comparison (PG widens both to int4): sz < 30000 -> all rows.
    let wide = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE sz < 30000")
        .expect("sz < 30000 on GPU");
    assert_eq!(wide.rows, labels(0..N), "< 30000 (> i16::MAX) => all rows");

    // projecting the SMALLINT column yields SqlValue::Int2(i16)
    let proj = e
        .execute_resident_expr_select_sql("SELECT sz FROM t WHERE sz = 0")
        .expect("project smallint on GPU");
    assert_eq!(
        proj.rows,
        vec![vec![SqlValue::Int2(0)]],
        "projecting sz returns the smallint value"
    );

    // REJECTIONS -- hard errors, never wrong rows:
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE sz = 'x'")
            .is_err(),
        "smallint compared to a text literal => hard error"
    );
    // INSERT out of the int16 range is rejected ("smallint out of range").
    assert!(
        e.execute_text(3, "INSERT INTO t (sz, peer, label) VALUES (40000, 0, 99)")
            .is_err(),
        "INSERT 40000 into smallint => out of range error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_bool_predicate() {
    // bool-predicate on the general GPU executor (the type matrix, doc 19): a bool column is a
    // 1-bit-per-row BITMAP, so `WHERE flag` expands the bitmap straight to the row mask -- no compare.
    // flag[i] = (i even), label = i.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (flag BOOL, label INT)").unwrap();
    const N: i64 = 20;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        let flag = if i % 2 == 0 { "true" } else { "false" };
        values.push_str(&format!("({flag}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (flag, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // WHERE flag -> the rows where flag is true (the even labels)
    let r = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE flag")
        .expect("WHERE flag on GPU");
    let expected: Vec<Vec<SqlValue>> = (0..N)
        .filter(|i| i % 2 == 0)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();
    assert_eq!(r.rows, expected, "WHERE flag => the true (even) rows");
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));

    let even: Vec<Vec<SqlValue>> = (0..N)
        .filter(|i| i % 2 == 0)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();
    let odd: Vec<Vec<SqlValue>> = (0..N)
        .filter(|i| i % 2 != 0)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();

    // flag = true / = false / <> (bool literal compares lower to the bitmap->mask kernel via negate).
    for (sql, expected, label) in [
        ("SELECT label FROM t WHERE flag = true", &even, "= true => even"),
        ("SELECT label FROM t WHERE flag = false", &odd, "= false => odd"),
        ("SELECT label FROM t WHERE true = flag", &even, "true = flag => even"),
        ("SELECT label FROM t WHERE flag <> true", &odd, "<> true => odd"),
        ("SELECT label FROM t WHERE flag <> false", &even, "<> false => even"),
        // NOT flag === flag = false (the mapper rewrites it).
        ("SELECT label FROM t WHERE NOT flag", &odd, "NOT flag => odd"),
    ] {
        let got = e.execute_resident_expr_select_sql(sql).expect(label);
        assert_eq!(&got.rows, expected, "{label}");
        assert_eq!(got.executed_target, DeviceTarget::Gpu(0), "{label} on GPU");
    }

    // PROJECT the bool column (gathered straight from the bitmap): flag for rows 0..4 = T,F,T,F.
    let proj = e
        .execute_resident_expr_select_sql("SELECT flag FROM t WHERE label < 4")
        .expect("project bool on GPU");
    assert_eq!(
        proj.rows,
        vec![
            vec![SqlValue::Bool(true)],
            vec![SqlValue::Bool(false)],
            vec![SqlValue::Bool(true)],
            vec![SqlValue::Bool(false)],
        ],
        "SELECT flag => the bitmap bits gathered as bools"
    );

    // A bare NON-bool column predicate is invalid SQL -> hard error (never wrong rows).
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE label")
            .is_err(),
        "WHERE <int column> => argument-of-WHERE-must-be-boolean error"
    );
    // NOT over a non-bool column is also invalid.
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE NOT label")
            .is_err(),
        "NOT <int column> => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_count_star() {
    // First operator-axis aggregate: COUNT(*) WHERE <pred> on the general GPU executor. The count is
    // the GPU filter's surviving-row count (the compaction result); PG returns bigint. a[i] = i,
    // flag[i] = (i % 3 == 0).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, flag BOOL)").unwrap();
    const N: i64 = 50;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        let flag = if i % 3 == 0 { "true" } else { "false" };
        values.push_str(&format!("({i}, {flag})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (a, flag) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // COUNT(*) WHERE a > 10 -> a in [11, 50) = 39 rows.
    let r = e
        .execute_resident_expr_select_sql("SELECT COUNT(*) FROM t WHERE a > 10")
        .expect("count on GPU");
    assert_eq!(r.rows, vec![vec![SqlValue::Int8(39)]], "COUNT(*) WHERE a > 10 => 39");
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));

    // COUNT(*) WHERE flag -- a popcount over the bool bitmap. i%3==0 in [0,50) = 17 rows.
    let flag_count = (0..N).filter(|i| i % 3 == 0).count() as i64;
    let r2 = e
        .execute_resident_expr_select_sql("SELECT COUNT(*) FROM t WHERE flag")
        .expect("count flag on GPU");
    assert_eq!(r2.rows, vec![vec![SqlValue::Int8(flag_count)]], "COUNT(*) WHERE flag");

    // COUNT(*) of an empty result -> 0 (not an error / not NULL).
    let r3 = e
        .execute_resident_expr_select_sql("SELECT COUNT(*) FROM t WHERE a > 1000")
        .expect("count empty on GPU");
    assert_eq!(r3.rows, vec![vec![SqlValue::Int8(0)]], "COUNT(*) empty => 0");

    // count(*) is case-insensitive.
    let r4 = e
        .execute_resident_expr_select_sql("SELECT count(*) FROM t WHERE a >= 0")
        .expect("lowercase count on GPU");
    assert_eq!(r4.rows, vec![vec![SqlValue::Int8(N)]], "count(*) WHERE a >= 0 => all");

    // SUM(int4) over a filtered set -- a GPU reduction over the gathered column; PG returns bigint.
    let sum_of = |keep: &dyn Fn(i64) -> bool| -> i64 { (0..N).filter(|&i| keep(i)).sum() };
    let s1 = e
        .execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE a > 10")
        .expect("sum on GPU");
    assert_eq!(s1.rows, vec![vec![SqlValue::Int8(sum_of(&|a| a > 10))]], "SUM(a) WHERE a > 10");
    assert_eq!(s1.executed_target, DeviceTarget::Gpu(0));
    let s2 = e
        .execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE flag")
        .expect("sum where flag on GPU");
    assert_eq!(s2.rows, vec![vec![SqlValue::Int8(sum_of(&|a| a % 3 == 0))]], "SUM(a) WHERE flag");
    let s3 = e
        .execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE a >= 0")
        .expect("sum all on GPU");
    assert_eq!(s3.rows, vec![vec![SqlValue::Int8(sum_of(&|_| true))]], "SUM(a) WHERE a >= 0 => total");
    // SUM over an EMPTY set is NULL (M3) -> hard error, NOT a wrong 0.
    assert!(
        e.execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE a > 1000")
            .is_err(),
        "SUM over empty => NULL/M3 hard error (not 0)"
    );

    // MIN / MAX(int4) over a filtered set -- GPU reductions; PG preserves the type (int4 -> int4).
    let min_of = |keep: &dyn Fn(i64) -> bool| (0..N).filter(|&i| keep(i)).min().unwrap() as i32;
    let max_of = |keep: &dyn Fn(i64) -> bool| (0..N).filter(|&i| keep(i)).max().unwrap() as i32;
    let mn = e
        .execute_resident_expr_select_sql("SELECT MIN(a) FROM t WHERE a > 10")
        .expect("min on GPU");
    assert_eq!(mn.rows, vec![vec![SqlValue::Int4(min_of(&|a| a > 10))]], "MIN(a) WHERE a > 10 => 11");
    let mx = e
        .execute_resident_expr_select_sql("SELECT MAX(a) FROM t WHERE flag")
        .expect("max where flag on GPU");
    assert_eq!(mx.rows, vec![vec![SqlValue::Int4(max_of(&|a| a % 3 == 0))]], "MAX(a) WHERE flag");
    let mx2 = e
        .execute_resident_expr_select_sql("SELECT MAX(a) FROM t WHERE a >= 0")
        .expect("max all on GPU");
    assert_eq!(mx2.rows, vec![vec![SqlValue::Int4((N - 1) as i32)]], "MAX(a) WHERE a >= 0 => N-1");
    let mn2 = e
        .execute_resident_expr_select_sql("SELECT MIN(a) FROM t WHERE a >= 0")
        .expect("min all on GPU");
    assert_eq!(mn2.rows, vec![vec![SqlValue::Int4(0)]], "MIN(a) WHERE a >= 0 => 0");
    // MIN/MAX over an EMPTY set is NULL (M3) -> hard error.
    assert!(
        e.execute_resident_expr_select_sql("SELECT MIN(a) FROM t WHERE a > 1000")
            .is_err()
            && e.execute_resident_expr_select_sql("SELECT MAX(a) FROM t WHERE a > 1000")
                .is_err(),
        "MIN/MAX over empty => NULL/M3 hard error"
    );

    // The remaining aggregates are follow-ons -> hard error (clear message, never a wrong/blank
    // answer). FILTER / OVER live INSIDE the FuncCall: they must reject, not silently drop (audit P0).
    for sql in [
        "SELECT COUNT(a) FROM t WHERE a > 0",
        "SELECT COUNT(*) FILTER (WHERE a > 90) FROM t WHERE a >= 0",
        "SELECT COUNT(*) OVER () FROM t WHERE a >= 0",
        "SELECT COUNT(*) OVER (ORDER BY a) FROM t WHERE a >= 0",
        "SELECT COUNT(*) AS c FROM t WHERE a > 0",
        "SELECT SUM(*) FROM t WHERE a > 0",
        "SELECT MIN(*) FROM t WHERE a > 0",
        "SELECT AVG(*) FROM t WHERE a > 0",
    ] {
        assert!(
            e.execute_resident_expr_select_sql(sql).is_err(),
            "{sql} => aggregate follow-on error (no silent FILTER/OVER drop)"
        );
    }

    // AVG(int4) = the GPU sum / the count, as numeric (PG, scale 16). The reduction is on the GPU;
    // the scalar divide reuses average_sql_value (so this checks the pipeline picks the right
    // sum + count). a > 10 -> 1170/39 = 30 exactly; a >= 0 -> 1225/50 = 24.5.
    let avg_expected = |keep: &dyn Fn(i64) -> bool| -> Vec<Vec<SqlValue>> {
        let sum: i128 = (0..N).filter(|&i| keep(i)).map(i128::from).sum();
        let count = (0..N).filter(|&i| keep(i)).count();
        vec![vec![average_sql_value(sum, count)]]
    };
    let a1 = e
        .execute_resident_expr_select_sql("SELECT AVG(a) FROM t WHERE a > 10")
        .expect("avg on GPU");
    assert_eq!(a1.rows, avg_expected(&|a| a > 10), "AVG(a) WHERE a > 10 => 30");
    assert_eq!(a1.executed_target, DeviceTarget::Gpu(0));
    // 30 exactly, at scale 16.
    assert_eq!(
        a1.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(30 * 10_i128.pow(16), 16))]],
        "AVG = 30.0000000000000000"
    );
    let a2 = e
        .execute_resident_expr_select_sql("SELECT AVG(a) FROM t WHERE a >= 0")
        .expect("avg all on GPU");
    assert_eq!(a2.rows, avg_expected(&|_| true), "AVG(a) all => 24.5");
    let a3 = e
        .execute_resident_expr_select_sql("SELECT AVG(a) FROM t WHERE flag")
        .expect("avg where flag on GPU");
    assert_eq!(a3.rows, avg_expected(&|a| a % 3 == 0), "AVG(a) WHERE flag");
    // AVG over an EMPTY set is NULL (M3) -> hard error.
    assert!(
        e.execute_resident_expr_select_sql("SELECT AVG(a) FROM t WHERE a > 1000")
            .is_err(),
        "AVG over empty => NULL/M3 hard error"
    );
}

#[test]
fn average_sql_value_matches_postgres_dynamic_scale_and_rounding() {
    // AVG = sum/count must match PostgreSQL's numeric division EXACTLY: a dynamic result scale
    // (PG select_div_scale, ~16 significant digits) + round half-away-from-zero. The hardcoded
    // expectations are PostgreSQL 18's text output (NOT computed via average_sql_value -- the prior
    // self-referential tests could not catch the fixed-scale-16 / truncation P0 the AVG audit found).
    let avg_str = |sum: i128, count: usize| -> String {
        match average_sql_value(sum, count) {
            SqlValue::Numeric(d) => d.to_decimal_string(),
            other => panic!("AVG must be numeric, got {other:?}"),
        }
    };
    // 1-4 integer digits -> scale 16.
    assert_eq!(avg_str(3, 1), "3.0000000000000000", "exact integer, scale 16");
    assert_eq!(avg_str(7, 2), "3.5000000000000000", "3.5, scale 16");
    // sub-1 quotient -> scale 20; repeating, ROUNDS the last digit up.
    assert_eq!(avg_str(2, 3), "0.66666666666666666667", "2/3 rounds, scale 20");
    assert_eq!(avg_str(1, 2), "0.50000000000000000000", "1/2, scale 20");
    // 5-digit integer part -> scale 12 (even exact integers carry the dynamic scale).
    assert_eq!(avg_str(234_000, 4), "58500.000000000000", "58500, scale 12");
    // 11-digit integer part -> scale 8.
    assert_eq!(
        avg_str(100_000_000_000, 7),
        "14285714285.71428571",
        "5-digit-group scale 8"
    );
    // Negative: magnitude + sign both correct (round away from zero).
    assert_eq!(avg_str(-2, 3), "-0.66666666666666666667", "negative rounds away");
    // PG select_div_scale decrements the quotient weight when the dividend's leading base-10000 digit
    // <= the divisor's -- so an exact 1.0 from sum==count renders at scale 20, NOT 16. The naive
    // "quotient decimal weight" formula shipped scale 16 here; these lock the fix (verified vs PG 18).
    assert_eq!(avg_str(3, 3), "1.00000000000000000000", "sum==count -> scale 20 (firstdigit decr)");
    assert_eq!(avg_str(5, 5), "1.00000000000000000000", "leading-digit-equal -> scale 20");
    assert_eq!(avg_str(9, 3), "3.0000000000000000", "fd1(9) > fd2(3) -> no decr, scale 16");
    // Zero sum (e.g. AVG over cancelling rows): PG renders 0 at scale max(S, 20), not 16.
    assert_eq!(avg_str(0, 2), "0.00000000000000000000", "zero sum -> scale 20");
    assert_eq!(avg_str(0, 5), "0.00000000000000000000", "zero sum, count 5 -> scale 20");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int8_aggregates() {
    // int8 aggregates: MIN/MAX -> int8, SUM/AVG -> numeric (a sum of int8 can exceed i64, so SUM
    // reduces to i128 via the two-atomic carry kernel). Values span > i32::MAX, negatives, and a
    // subset (rows 0,1) whose SUM EXCEEDS i64::MAX. label = row index (the int4 filter column).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (b BIGINT, label INT)").unwrap();
    let vals: [i64; 6] = [
        9_000_000_000_000_000_000,
        8_000_000_000_000_000_000,
        0,
        5_000_000_000,
        -7_000_000_000,
        -9_000_000_000_000_000_000,
    ];
    let mut values = String::new();
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({v}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (b, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let keep_all = |_: usize| true;
    let keep_lt2 = |i: usize| i < 2;
    let min_of = |k: &dyn Fn(usize) -> bool| vals.iter().enumerate().filter(|(i, _)| k(*i)).map(|(_, v)| *v).min().unwrap();
    let max_of = |k: &dyn Fn(usize) -> bool| vals.iter().enumerate().filter(|(i, _)| k(*i)).map(|(_, v)| *v).max().unwrap();
    let sum_of = |k: &dyn Fn(usize) -> bool| vals.iter().enumerate().filter(|(i, _)| k(*i)).map(|(_, v)| i128::from(*v)).sum::<i128>();

    // MIN/MAX(int8) -> int8.
    let mn = e.execute_resident_expr_select_sql("SELECT MIN(b) FROM t WHERE label >= 0").expect("min");
    assert_eq!(mn.rows, vec![vec![SqlValue::Int8(min_of(&keep_all))]], "MIN(b) all");
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));
    let mx = e.execute_resident_expr_select_sql("SELECT MAX(b) FROM t WHERE label < 2").expect("max subset");
    assert_eq!(mx.rows, vec![vec![SqlValue::Int8(max_of(&keep_lt2))]], "MAX(b) subset => 9e18");

    // SUM(int8) -> numeric (scale 0). The subset {9e18, 8e18} sums to 17e18 -- EXCEEDS i64::MAX, so
    // the i128 two-atomic carry must be correct.
    let s_sub = e.execute_resident_expr_select_sql("SELECT SUM(b) FROM t WHERE label < 2").expect("sum subset");
    assert_eq!(
        s_sub.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(sum_of(&keep_lt2), 0))]],
        "SUM subset => 17e18 (> i64::MAX, i128 carry)"
    );
    assert!(sum_of(&keep_lt2) > i128::from(i64::MAX), "test really exceeds i64");
    // The result COLUMN descriptor must match the numeric value (the int8-agg audit caught SUM(int8)
    // declaring Int4/oid 23 -- a wire-decode mismatch). PG SUM(int8) -> numeric (oid 1700).
    assert!(
        matches!(s_sub.columns[0].ty, SqlType::Numeric { .. }),
        "SUM(int8) result column must be numeric, got {:?}",
        s_sub.columns[0].ty
    );
    assert_eq!(s_sub.columns[0].type_oid, 1700, "SUM(int8) wire oid = numeric 1700");
    let s_all = e.execute_resident_expr_select_sql("SELECT SUM(b) FROM t WHERE label >= 0").expect("sum all");
    assert_eq!(
        s_all.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(sum_of(&keep_all), 0))]],
        "SUM all (incl. negatives)"
    );

    // AVG(int8) -> numeric. 17e18 / 2 = 8.5e18 (scale 0 at this magnitude).
    let av = e.execute_resident_expr_select_sql("SELECT AVG(b) FROM t WHERE label < 2").expect("avg subset");
    assert_eq!(
        av.rows,
        vec![vec![average_sql_value(sum_of(&keep_lt2), 2)]],
        "AVG subset"
    );
    match &av.rows[0][0] {
        SqlValue::Numeric(d) => assert_eq!(d.to_decimal_string(), "8500000000000000000", "AVG = 8.5e18"),
        other => panic!("AVG must be numeric, got {other:?}"),
    }

    // Empty SUM/MIN/AVG -> hard error (M3).
    for sql in [
        "SELECT MIN(b) FROM t WHERE label > 1000",
        "SELECT SUM(b) FROM t WHERE label > 1000",
        "SELECT AVG(b) FROM t WHERE label > 1000",
    ] {
        assert!(e.execute_resident_expr_select_sql(sql).is_err(), "{sql} => empty NULL/M3 error");
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_minmax() {
    // MIN/MAX(numeric) over a filtered set -> numeric (PG preserves the type). Reduces the i128
    // mantissas via the partials + host-combine reduction. label = row index (int4 filter col).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (p NUMERIC(10,2), label INT)").unwrap();
    let prices = ["12.50", "-3.75", "100.00", "0.01", "-99.99", "42.42"];
    let mut values = String::new();
    for (i, p) in prices.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({p}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (p, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // mantissas at scale 2.
    let num = |m: i128| SqlValue::Numeric(Decimal128::new(m, 2));

    // All rows: MIN = -99.99, MAX = 100.00.
    let mn = e.execute_resident_expr_select_sql("SELECT MIN(p) FROM t WHERE label >= 0").expect("min num");
    assert_eq!(mn.rows, vec![vec![num(-9999)]], "MIN(p) all => -99.99");
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));
    let mx = e.execute_resident_expr_select_sql("SELECT MAX(p) FROM t WHERE label >= 0").expect("max num");
    assert_eq!(mx.rows, vec![vec![num(10000)]], "MAX(p) all => 100.00");

    // Subset (label < 3 -> 12.50, -3.75, 100.00): MIN = -3.75, MAX = 100.00.
    let mn2 = e.execute_resident_expr_select_sql("SELECT MIN(p) FROM t WHERE label < 3").expect("min subset");
    assert_eq!(mn2.rows, vec![vec![num(-375)]], "MIN subset => -3.75");
    let mx2 = e.execute_resident_expr_select_sql("SELECT MAX(p) FROM t WHERE label < 3").expect("max subset");
    assert_eq!(mx2.rows, vec![vec![num(10000)]], "MAX subset => 100.00");

    // Empty -> hard error (M3).
    assert!(
        e.execute_resident_expr_select_sql("SELECT MAX(p) FROM t WHERE label > 1000").is_err(),
        "MAX(numeric) over empty => NULL/M3 hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_sum_avg() {
    // SUM(numeric) -> numeric at the column scale (i128 mantissa sum); AVG(numeric) -> numeric at PG's
    // division scale. Expected values derived from PG's numeric semantics (SUM keeps scale 2; AVG of a
    // weight-0 quotient over a scale-2 dividend has rscale max(2, 16) = 16).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (p NUMERIC(10,2), label INT)").unwrap();
    let prices = ["12.50", "-3.75", "100.00", "0.01", "-99.99", "42.42"];
    let mut values = String::new();
    for (i, p) in prices.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({p}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (p, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // SUM all = 12.50 - 3.75 + 100.00 + 0.01 - 99.99 + 42.42 = 51.19 (mantissa 5119, scale 2).
    let s_all = e.execute_resident_expr_select_sql("SELECT SUM(p) FROM t WHERE label >= 0").expect("sum all");
    assert_eq!(s_all.rows, vec![vec![SqlValue::Numeric(Decimal128::new(5119, 2))]], "SUM all = 51.19");
    assert_eq!(s_all.executed_target, DeviceTarget::Gpu(0));
    assert!(matches!(s_all.columns[0].ty, SqlType::Numeric { .. }), "SUM(numeric) col numeric");
    assert_eq!(s_all.columns[0].type_oid, 1700, "SUM(numeric) oid 1700");
    // SUM subset (12.50, -3.75) = 8.75.
    let s_sub = e.execute_resident_expr_select_sql("SELECT SUM(p) FROM t WHERE label < 2").expect("sum subset");
    assert_eq!(s_sub.rows, vec![vec![SqlValue::Numeric(Decimal128::new(875, 2))]], "SUM subset = 8.75");

    // AVG all = 51.19 / 6 = 8.5316666... -> scale 16, round half-away.
    let a_all = e.execute_resident_expr_select_sql("SELECT AVG(p) FROM t WHERE label >= 0").expect("avg all");
    match &a_all.rows[0][0] {
        SqlValue::Numeric(d) => assert_eq!(d.to_decimal_string(), "8.5316666666666667", "AVG all"),
        other => panic!("AVG numeric, got {other:?}"),
    }
    assert!(matches!(a_all.columns[0].ty, SqlType::Numeric { .. }), "AVG(numeric) col numeric");
    // AVG subset = 8.75 / 2 = 4.375 -> scale 16.
    let a_sub = e.execute_resident_expr_select_sql("SELECT AVG(p) FROM t WHERE label < 2").expect("avg subset");
    match &a_sub.rows[0][0] {
        SqlValue::Numeric(d) => assert_eq!(d.to_decimal_string(), "4.3750000000000000", "AVG subset"),
        other => panic!("AVG numeric, got {other:?}"),
    }

    // Empty SUM/AVG -> hard error (M3).
    for sql in [
        "SELECT SUM(p) FROM t WHERE label > 1000",
        "SELECT AVG(p) FROM t WHERE label > 1000",
    ] {
        assert!(e.execute_resident_expr_select_sql(sql).is_err(), "{sql} => empty error");
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_numeric_sum_overflow_errors() {
    // SUM(numeric) is CHECKED: a mantissa sum exceeding i128 is PG `numeric field overflow`, NEVER a
    // silent wrap. Each mantissa is 9e18 * 10^19 = 9e37 (column NUMERIC(38,19), integer part 9e18 fits
    // the legacy parser's i64 literal range); two sum to 1.8e38 > i128::MAX (~1.7e38).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE big (v NUMERIC(38,19), label INT)").unwrap();
    let big = "9000000000000000000"; // 9e18, fits i64
    e.execute_text(2, &format!("INSERT INTO big (v, label) VALUES ({big}, 0), ({big}, 1)"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("big").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let mantissa = 9_000_000_000_000_000_000_i128 * 10_i128.pow(19); // 9e37
    assert!(
        mantissa.checked_add(mantissa).is_none(),
        "sanity: 2 * 9e37 must overflow i128 (else the test does not exercise overflow)"
    );
    let err = e
        .execute_resident_expr_select_sql("SELECT SUM(v) FROM big WHERE label >= 0")
        .expect_err("SUM overflow must error");
    assert!(
        err.to_string().contains("numeric field overflow"),
        "expected numeric field overflow, got: {err}"
    );
    // A single value (no overflow) still sums correctly to its mantissa at scale 19.
    let ok = e
        .execute_resident_expr_select_sql("SELECT SUM(v) FROM big WHERE label < 1")
        .expect("single value sums");
    assert_eq!(
        ok.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(mantissa, 19))]],
        "SUM of one 9e37-mantissa value"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_full_table_no_where() {
    // No WHERE clause = a full-table scan (indices 0..row_count): aggregates reduce over every row and
    // projection materializes every row, all on the general GPU executor.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a, b) VALUES (5,10),(3,20),(8,30),(1,40),(9,50)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let a = [5_i32, 3, 8, 1, 9];

    // COUNT(*) over the whole table.
    let c = e.execute_resident_expr_select_sql("SELECT COUNT(*) FROM t").expect("count *");
    assert_eq!(c.rows, vec![vec![SqlValue::Int8(a.len() as i64)]], "COUNT(*) no WHERE = 5");
    assert_eq!(c.executed_target, DeviceTarget::Gpu(0));

    // SUM/MIN/MAX over the whole column.
    let s = e.execute_resident_expr_select_sql("SELECT SUM(a) FROM t").expect("sum");
    assert_eq!(s.rows, vec![vec![SqlValue::Int8(a.iter().map(|&v| i64::from(v)).sum())]], "SUM(a)=26");
    let mn = e.execute_resident_expr_select_sql("SELECT MIN(a) FROM t").expect("min");
    assert_eq!(mn.rows, vec![vec![SqlValue::Int4(*a.iter().min().unwrap())]], "MIN(a)=1");
    let mx = e.execute_resident_expr_select_sql("SELECT MAX(a) FROM t").expect("max");
    assert_eq!(mx.rows, vec![vec![SqlValue::Int4(*a.iter().max().unwrap())]], "MAX(a)=9");

    // AVG(a) = 26/5 = 5.2 -> numeric scale 16 (fd1=26 > fd2=5, no leading-digit decrement).
    let av = e.execute_resident_expr_select_sql("SELECT AVG(a) FROM t").expect("avg");
    match &av.rows[0][0] {
        SqlValue::Numeric(d) => assert_eq!(d.to_decimal_string(), "5.2000000000000000", "AVG no WHERE"),
        other => panic!("AVG numeric, got {other:?}"),
    }

    // Full-table projection: every row, in residency (insertion) order.
    let p = e.execute_resident_expr_select_sql("SELECT a FROM t").expect("project a");
    let got: Vec<i32> = p
        .rows
        .iter()
        .map(|r| match r[0] {
            SqlValue::Int4(v) => v,
            ref other => panic!("expected int4, got {other:?}"),
        })
        .collect();
    assert_eq!(got, a.to_vec(), "SELECT a FROM t projects all rows in order");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_group_by() {
    // GROUP BY an int4 key on the general GPU executor (hash aggregation): COUNT/SUM/AVG per group,
    // results sorted by key for determinism. Groups: g=1 -> v{10,20,30}, g=2 -> v{5,15}, g=3 -> v{100}.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (g,v) VALUES (1,10),(2,5),(1,20),(3,100),(2,15),(1,30)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;
    let i8 = SqlValue::Int8;

    // GROUP BY g, COUNT(*): g=1->3, g=2->2, g=3->1.
    let c = e.execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g").expect("grouped count");
    assert_eq!(
        c.rows,
        vec![vec![i4(1), i8(3)], vec![i4(2), i8(2)], vec![i4(3), i8(1)]],
        "GROUP BY count"
    );
    assert_eq!(c.executed_target, DeviceTarget::Gpu(0));

    // GROUP BY g, SUM(v): g=1->60, g=2->20, g=3->100.
    let s = e.execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g").expect("grouped sum");
    assert_eq!(
        s.rows,
        vec![vec![i4(1), i8(60)], vec![i4(2), i8(20)], vec![i4(3), i8(100)]],
        "GROUP BY sum"
    );

    // GROUP BY g, AVG(v): 60/3=20, 20/2=10, 100/1=100 -> numeric scale 16 (PG select_div_scale).
    let a = e.execute_resident_expr_select_sql("SELECT g, AVG(v) FROM t GROUP BY g").expect("grouped avg");
    let avg_strs: Vec<String> = a
        .rows
        .iter()
        .map(|r| match &r[1] {
            SqlValue::Numeric(d) => d.to_decimal_string(),
            other => panic!("AVG must be numeric, got {other:?}"),
        })
        .collect();
    assert_eq!(a.rows.iter().map(|r| r[0].clone()).collect::<Vec<_>>(), vec![i4(1), i4(2), i4(3)], "AVG keys");
    assert_eq!(
        avg_strs,
        vec!["20.0000000000000000", "10.0000000000000000", "100.0000000000000000"],
        "GROUP BY avg"
    );

    // GROUP BY with a WHERE: group only the surviving rows. v > 10 -> (1,20),(3,100),(2,15),(1,30):
    // g=1 -> 2 (20,30), g=2 -> 1 (15), g=3 -> 1 (100).
    let w = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t WHERE v > 10 GROUP BY g")
        .expect("grouped count + where");
    assert_eq!(
        w.rows,
        vec![vec![i4(1), i8(2)], vec![i4(2), i8(1)], vec![i4(3), i8(1)]],
        "GROUP BY count + WHERE"
    );

    // The result schema is [group key, aggregate]: 2 columns named g + the aggregate.
    assert_eq!(c.columns.len(), 2, "grouped result has key + aggregate columns");
    assert_eq!(c.columns[0].name, "g", "first result column is the group key");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_grouped_min_max() {
    // GROUP BY with per-group MIN/MAX on the general GPU executor (single-level kernel; signed s64
    // atom.min/max). Expected values are CONSTRUCTED from the inserted rows (a GPU-native oracle, not
    // a CPU re-fold): g=1 -> v{10,30,20}; g=2 -> v{5,15,-7}; g=3 -> v{100}. A NEGATIVE value exercises
    // the signed min.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES (1,10),(2,5),(1,30),(3,100),(2,15),(1,20),(2,-7)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;

    // g=1 -> min 10; g=2 -> min -7 (signed); g=3 -> min 100.
    let mn = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t GROUP BY g")
        .expect("grouped min");
    assert_eq!(
        mn.rows,
        vec![vec![i4(1), i4(10)], vec![i4(2), i4(-7)], vec![i4(3), i4(100)]],
        "GROUP BY min"
    );
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));

    // g=1 -> max 30; g=2 -> max 15; g=3 -> max 100.
    let mx = e
        .execute_resident_expr_select_sql("SELECT g, MAX(v) FROM t GROUP BY g")
        .expect("grouped max");
    assert_eq!(
        mx.rows,
        vec![vec![i4(1), i4(30)], vec![i4(2), i4(15)], vec![i4(3), i4(100)]],
        "GROUP BY max"
    );

    // MIN with a WHERE: the predicate-filtered indices feed the same kernel. v > 0 drops (2,-7), so
    // g=2's min becomes 5.
    let mw = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t WHERE v > 0 GROUP BY g")
        .expect("grouped min + where");
    assert_eq!(
        mw.rows,
        vec![vec![i4(1), i4(10)], vec![i4(2), i4(5)], vec![i4(3), i4(100)]],
        "GROUP BY min + WHERE"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_min_max_over_int8_value() {
    // GROUP BY an int4 key, MIN/MAX of an int8 (BIGINT) value -> exercises the 8-byte-stride
    // 2x4-byte value read + values WAY beyond the i32 range. Constructed oracle:
    //   g=1 -> {1e10, -5e9, 3e10}      (min -5e9, max 3e10)
    //   g=2 -> {i64::MAX, -9e18}        (min -9e18, max i64::MAX)
    // The i64::MAX value also probes that the min-identity (i64::MAX) collision is benign (the slot
    // is occupied, so the real value is read even when it equals the fill identity).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v BIGINT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES \
         (1,10000000000),(2,9223372036854775807),(1,-5000000000),(1,30000000000),(2,-9000000000000000000)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;
    let i8 = SqlValue::Int8;

    let mn = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t GROUP BY g")
        .expect("int8 grouped min");
    assert_eq!(
        mn.rows,
        vec![
            vec![i4(1), i8(-5_000_000_000)],
            vec![i4(2), i8(-9_000_000_000_000_000_000)],
        ],
        "int8 GROUP BY min"
    );
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));

    let mx = e
        .execute_resident_expr_select_sql("SELECT g, MAX(v) FROM t GROUP BY g")
        .expect("int8 grouped max");
    assert_eq!(
        mx.rows,
        vec![vec![i4(1), i8(30_000_000_000)], vec![i4(2), i8(i64::MAX)]],
        "int8 GROUP BY max"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_sum_avg_over_int8_value() {
    // GROUP BY int4 key, SUM/AVG of an int8 (BIGINT) value. The per-group SUM is accumulated as i128
    // (the two-atomic carry, now per hash slot), so it can EXCEED i64 in both directions. PG:
    // SUM(bigint) -> numeric (scale 0). Constructed oracle:
    //   g=1 -> {5e18, 5e18}     sum  1.0e19  (> i64::MAX)
    //   g=2 -> {-6e18, -6e18}   sum -1.2e19  (< i64::MIN)
    //   g=3 -> {100, 200, 300}  sum  600     (fits i64)
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v BIGINT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES \
         (1,5000000000000000000),(1,5000000000000000000),\
         (2,-6000000000000000000),(2,-6000000000000000000),\
         (3,100),(3,200),(3,300)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;

    // SUM(int8) -> numeric (scale 0), exceeding i64 in both directions via the i128 carry.
    let s = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g")
        .expect("int8 grouped sum");
    let sum_strs: Vec<(SqlValue, String)> = s
        .rows
        .iter()
        .map(|r| {
            (
                r[0].clone(),
                match &r[1] {
                    SqlValue::Numeric(d) => d.to_decimal_string(),
                    other => panic!("SUM(int8) must be numeric, got {other:?}"),
                },
            )
        })
        .collect();
    assert_eq!(
        sum_strs,
        vec![
            (i4(1), "10000000000000000000".to_string()),
            (i4(2), "-12000000000000000000".to_string()),
            (i4(3), "600".to_string()),
        ],
        "int8 GROUP BY sum (i128 carry)"
    );
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));

    // AVG(int8) -> numeric, computed from the i128 sum. Expected = the engine's own average_sql_value
    // over the constructed i128 sum / count, so the assertion tracks PG's div-scale exactly without
    // hardcoding a scale that varies with magnitude.
    let avg = crate::rel_exec_helpers::average_sql_value;
    let a = e
        .execute_resident_expr_select_sql("SELECT g, AVG(v) FROM t GROUP BY g")
        .expect("int8 grouped avg");
    assert_eq!(
        a.rows,
        vec![
            vec![i4(1), avg(10_000_000_000_000_000_000_i128, 2)],
            vec![i4(2), avg(-12_000_000_000_000_000_000_i128, 2)],
            vec![i4(3), avg(600_i128, 3)],
        ],
        "int8 GROUP BY avg"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_int8_sum_survives_many_low_limb_wraps() {
    // Permanent regression guard for the LOCK-FREE per-slot i128 carry. One group of N rows all =
    // i64::MAX makes the slot's low limb wrap ~N/2 times under concurrent atomicAdds; a dropped or
    // double-counted carry shows up as the high limb (sum_hi) off by the wrap count. A second group
    // of N x i64::MIN stresses the negative path. Oracle = N * value as i128 (computed, not hardcoded).
    const N: usize = 1000;
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v BIGINT)").unwrap();
    let pos = vec!["(1,9223372036854775807)"; N].join(",");
    let neg = vec!["(2,-9223372036854775808)"; N].join(",");
    e.execute_text(2, &format!("INSERT INTO t (g,v) VALUES {pos},{neg}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let s = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g")
        .expect("many-wraps sum");
    let strs: Vec<String> = s
        .rows
        .iter()
        .map(|r| match &r[1] {
            SqlValue::Numeric(d) => d.to_decimal_string(),
            other => panic!("SUM(int8) must be numeric, got {other:?}"),
        })
        .collect();
    assert_eq!(
        strs,
        vec![
            (i128::from(N as i64) * i128::from(i64::MAX)).to_string(),
            (i128::from(N as i64) * i128::from(i64::MIN)).to_string(),
        ],
        "int8 SUM under many concurrent low-limb wraps"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_by_int8_key() {
    // GROUP BY an int8 (BIGINT) key -> the single-level kernel reads 64-bit keys. Covers keys beyond
    // the i32 range AND the i64::MIN key, which collides with the EMPTY sentinel and so is routed to
    // its dedicated slot (the crux). Constructed oracle:
    //   g=1e10     -> v{10,20,30}  (count 3, sum 60, min 10)
    //   g=-8e9     -> v{5,15}      (count 2, sum 20, min 5)
    //   g=i64::MIN -> v{100,200}   (count 2, sum 300, min 100)  [dedicated-slot edge]
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g BIGINT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES \
         (10000000000,10),(10000000000,20),(10000000000,30),\
         (-8000000000,5),(-8000000000,15),\
         (-9223372036854775808,100),(-9223372036854775808,200)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i8v = SqlValue::Int8;
    let i4 = SqlValue::Int4;

    // Keys sorted ascending: i64::MIN < -8e9 < 1e10. Result keys are Int8.
    let c = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g")
        .expect("int8-key count");
    assert_eq!(
        c.rows,
        vec![
            vec![i8v(i64::MIN), i8v(2)],
            vec![i8v(-8_000_000_000), i8v(2)],
            vec![i8v(10_000_000_000), i8v(3)],
        ],
        "GROUP BY int8 key, COUNT (incl. i64::MIN dedicated slot)"
    );
    assert_eq!(c.executed_target, DeviceTarget::Gpu(0));

    let s = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g")
        .expect("int8-key sum");
    assert_eq!(
        s.rows,
        vec![
            vec![i8v(i64::MIN), i8v(300)],
            vec![i8v(-8_000_000_000), i8v(20)],
            vec![i8v(10_000_000_000), i8v(60)],
        ],
        "GROUP BY int8 key, SUM(int4)"
    );

    let m = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t GROUP BY g")
        .expect("int8-key min");
    assert_eq!(
        m.rows,
        vec![
            vec![i8v(i64::MIN), i4(100)],
            vec![i8v(-8_000_000_000), i4(5)],
            vec![i8v(10_000_000_000), i4(10)],
        ],
        "GROUP BY int8 key, MIN(int4)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_by_int8_key_i64min_heavy_contention_and_misaligned() {
    // Permanent regression guard for the two scariest int8-key failure modes (the prior audit
    // verified both via probes, since reverted):
    //  1. SENTINEL CONTENTION: N rows all keyed i64::MIN hammer the single dedicated slot
    //     concurrently -- the atomicAdds must serialize (count == N), and a lost sentinel route
    //     would drop the group entirely.
    //  2. MISALIGNED int8 KEY column: `(b INT, g BIGINT)` with an ODD row count puts the int8 key
    //     section at offset 4-mod-8, so the kernel's 2x4-byte key read is exercised (a single ld.u64
    //     would fault). We assert the offset is genuinely 4-mod-8 before trusting the result.
    const N: usize = 500;
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (b INT, g BIGINT)").unwrap();
    // N i64::MIN-key rows + 1 normal-key row => N+1 (odd) rows => one int4 col (b) * odd rows is odd
    // => the int8 `g` section lands at 4-mod-8.
    let sentinel = vec!["(7,-9223372036854775808)"; N].join(",");
    e.execute_text(2, &format!("INSERT INTO t (b,g) VALUES {sentinel},(7,42)"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // Confirm we actually exercised a 4-mod-8 int8 key column (else the misalignment axis is untested).
    let table = e.relational_catalog_table("t").unwrap();
    let g_idx = table.columns.iter().position(|c| c.name == "g").unwrap();
    let snap = e.relational_residency_snapshot_ref("t").unwrap();
    let g_off =
        crate::relational_model::resident_device_int8_column_offset(&snap, &table, g_idx).unwrap();
    assert_eq!(g_off % 8, 4, "int8 key column must be 4-mod-8 to exercise the misaligned read");

    let c = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g")
        .expect("sentinel-contention count");
    assert_eq!(
        c.rows,
        vec![
            vec![SqlValue::Int8(i64::MIN), SqlValue::Int8(N as i64)],
            vec![SqlValue::Int8(42), SqlValue::Int8(1)],
        ],
        "i64::MIN key under heavy contention (count == N), misaligned 4-mod-8 key column"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_sum_avg_over_numeric_value() {
    // GROUP BY an int4 key, SUM/AVG over a NUMERIC(20,2) value -> the single-level kernel reads the
    // i128 mantissa (16-byte stride) and accumulates i128 per slot; the result carries the column
    // scale (2). Constructed oracle:
    //   g=1 -> {10.50, 20.25, -3.75}  sum 27.00
    //   g=2 -> {100.00, -50.50}        sum 49.50
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(20,2))").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES (1,10.50),(1,20.25),(1,-3.75),(2,100.00),(2,-50.50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;

    let s = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g")
        .expect("numeric grouped sum");
    let sum_strs: Vec<(SqlValue, String)> = s
        .rows
        .iter()
        .map(|r| {
            (
                r[0].clone(),
                match &r[1] {
                    SqlValue::Numeric(d) => d.to_decimal_string(),
                    other => panic!("SUM(numeric) must be numeric, got {other:?}"),
                },
            )
        })
        .collect();
    assert_eq!(
        sum_strs,
        vec![
            (i4(1), "27.00".to_string()),
            (i4(2), "49.50".to_string()),
        ],
        "numeric GROUP BY sum (scale preserved)"
    );
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));

    // AVG(numeric): expected = the engine's own avg_numeric_sql_value over the per-group i128 sum
    // mantissa / count / column scale, so the assertion tracks PG's numeric div-scale exactly.
    let avg = crate::rel_exec_helpers::avg_numeric_sql_value;
    let a = e
        .execute_resident_expr_select_sql("SELECT g, AVG(v) FROM t GROUP BY g")
        .expect("numeric grouped avg");
    assert_eq!(
        a.rows,
        vec![vec![i4(1), avg(2700, 3, 2)], vec![i4(2), avg(4950, 2, 2)]],
        "numeric GROUP BY avg"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_numeric_sum_overflow_errors() {
    // A per-group numeric SUM that exceeds the i128 mantissa range must ERROR (PG numeric field
    // overflow), never silently wrap. Two values ~9e37 in one group -> ~1.8e38 > i128::MAX (~1.7e38);
    // the kernel's on-device per-add overflow check sets the flag and the host surfaces it.
    let mut e = Engine::new_local();
    // NUMERIC(38,19) value 9e18 -> mantissa 9e18 * 10^19 = 9e37 (the literal 9e18 fits the parser's
    // i64 range; the scale lifts the mantissa to 9e37). Two in one group sum to 1.8e38 > i128::MAX.
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(38,19))").unwrap();
    let big = "9000000000000000000"; // 9e18
    e.execute_text(2, &format!("INSERT INTO t (g,v) VALUES (1,{big}),(1,{big})"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let mantissa = 9_000_000_000_000_000_000_i128 * 10_i128.pow(19); // 9e37
    assert!(
        mantissa.checked_add(mantissa).is_none(),
        "sanity: 2 * 9e37 must overflow i128"
    );
    let r = e.execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g");
    assert!(r.is_err(), "numeric SUM overflow must error, got {r:?}");
    let msg = format!("{:?}", r.unwrap_err()).to_lowercase();
    assert!(
        msg.contains("overflow"),
        "expected a numeric-overflow error, got: {msg}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_numeric_sum_large_high_limb_no_overflow() {
    // Coverage for the numeric i128 carry's HIGH limb on a NON-overflowing sum -- the gap between the
    // fractional test (mantissas fit i64, so val_hi == 0) and the overflow test (errors before a
    // result). NUMERIC(38,19) value 5.0 has mantissa 5*10^19 > i64::MAX, so val_hi != 0; the per-group
    // sums stay within i128. A wrong high-limb carry would corrupt the result by multiples of 2^64.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(38,19))").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES (1,5.0),(1,5.0),(1,5.0),(2,-5.0),(2,-5.0)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let unit = 10_i128.pow(19); // mantissa units per 1.0 at scale 19
    assert!(
        5 * unit > i128::from(i64::MAX),
        "the value mantissa must exceed i64 to genuinely exercise the i128 high limb"
    );
    let s = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g")
        .expect("numeric sum large");
    let strs: Vec<(SqlValue, String)> = s
        .rows
        .iter()
        .map(|r| {
            (
                r[0].clone(),
                match &r[1] {
                    SqlValue::Numeric(d) => d.to_decimal_string(),
                    other => panic!("SUM(numeric) must be numeric, got {other:?}"),
                },
            )
        })
        .collect();
    assert_eq!(
        strs,
        vec![
            (SqlValue::Int4(1), Decimal128::new(15 * unit, 19).to_decimal_string()), // 3 * 5.0
            (SqlValue::Int4(2), Decimal128::new(-10 * unit, 19).to_decimal_string()), // 2 * -5.0
        ],
        "numeric SUM with a non-zero i128 high limb (no overflow)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_numeric_min_max() {
    // GROUP BY an int4 key, MIN/MAX over a NUMERIC(10,2) value (small mantissas, high limb 0). Exercises
    // the per-slot i128 spin-lock compare-and-update + signed ordering (a negative is the min), and a
    // duplicate max. Constructed oracle:
    //   g=1 -> {10.50, -3.25, 7.00}        min -3.25   max 10.50
    //   g=2 -> {100.00, -50.50, 100.00}    min -50.50  max 100.00 (duplicate max)
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(10,2))").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES (1,10.50),(1,-3.25),(1,7.00),(2,100.00),(2,-50.50),(2,100.00)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;
    let numeric_strs = |rows: &[Vec<SqlValue>]| -> Vec<(SqlValue, String)> {
        rows.iter()
            .map(|r| {
                (
                    r[0].clone(),
                    match &r[1] {
                        SqlValue::Numeric(d) => d.to_decimal_string(),
                        other => panic!("MIN/MAX(numeric) must be numeric, got {other:?}"),
                    },
                )
            })
            .collect()
    };

    let mn = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t GROUP BY g")
        .expect("numeric min");
    assert_eq!(
        numeric_strs(&mn.rows),
        vec![(i4(1), "-3.25".to_string()), (i4(2), "-50.50".to_string())],
        "numeric MIN"
    );
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));

    let mx = e
        .execute_resident_expr_select_sql("SELECT g, MAX(v) FROM t GROUP BY g")
        .expect("numeric max");
    assert_eq!(
        numeric_strs(&mx.rows),
        vec![(i4(1), "10.50".to_string()), (i4(2), "100.00".to_string())],
        "numeric MAX"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_numeric_min_max_large_high_limb() {
    // numeric MIN/MAX where the i128 mantissa EXCEEDS i64 (val_hi != 0), so the locked compare must
    // order on the signed high limb then the unsigned low limb. NUMERIC(38,19): mantissa = v*10^19, so
    // 5.0 -> 5e19 (val_hi=2), 1.0 -> 1e19 (val_hi=0 but low-limb bit63 set), negatives -> val_hi<0.
    //   g=1 -> {5.0, 2.5, -5.0, 1.0}  min -5.0  max 5.0
    //   g=2 -> {-2.0, -8.0, -1.0}     min -8.0  max -1.0  (ordering among negatives)
    //   g=3 -> {0.0}                  min  0.0  max 0.0   (single row -> identity overwritten)
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(38,19))").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES \
         (1,5.0),(1,2.5),(1,-5.0),(1,1.0),(2,-2.0),(2,-8.0),(2,-1.0),(3,0.0)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let unit = 10_i128.pow(19); // mantissa units per 1.0 at scale 19
    assert!(
        5 * unit > i128::from(i64::MAX),
        "the value mantissa must exceed i64 to genuinely exercise the i128 high limb in the compare"
    );
    let i4 = SqlValue::Int4;
    let dec = |m: i128| Decimal128::new(m, 19).to_decimal_string();
    let numeric_strs = |rows: &[Vec<SqlValue>]| -> Vec<(SqlValue, String)> {
        rows.iter()
            .map(|r| {
                (
                    r[0].clone(),
                    match &r[1] {
                        SqlValue::Numeric(d) => d.to_decimal_string(),
                        other => panic!("MIN/MAX(numeric) must be numeric, got {other:?}"),
                    },
                )
            })
            .collect()
    };

    let mn = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t GROUP BY g")
        .expect("numeric min hi");
    assert_eq!(
        numeric_strs(&mn.rows),
        vec![
            (i4(1), dec(-5 * unit)),
            (i4(2), dec(-8 * unit)),
            (i4(3), dec(0)),
        ],
        "numeric MIN with non-zero high limb"
    );

    let mx = e
        .execute_resident_expr_select_sql("SELECT g, MAX(v) FROM t GROUP BY g")
        .expect("numeric max hi");
    assert_eq!(
        numeric_strs(&mx.rows),
        vec![
            (i4(1), dec(5 * unit)),
            (i4(2), dec(-unit)),
            (i4(3), dec(0)),
        ],
        "numeric MAX with non-zero high limb"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_numeric_min_max_same_high_limb_tie() {
    // Pass 2's REASON TO EXIST: when several values in a group share the same i128 HIGH limb, the
    // min/max is decided by the LOW limb (unsigned). The other numeric MIN/MAX tests all use distinct
    // high limbs (pass 2 trivial) -- this exercises the tie + the decoy guard (a higher-high-limb
    // value must NOT corrupt the low-limb min). NUMERIC(38,19): 4.000...00NN all share high limb 2
    // (4e19 / 2^64 ~ 2.17); 13.0 has high limb 7. g=2 ties on a NEGATIVE high limb.
    let unit = 10_i128.pow(19);
    // sanity: the ties genuinely share a high limb (else this doesn't test pass 2).
    assert_eq!((4 * unit + 10) >> 64, (4 * unit + 200) >> 64, "positive ties must share high limb");
    assert_eq!((-(4 * unit + 10)) >> 64, (-(4 * unit + 200)) >> 64, "negative ties must share high limb");
    assert_ne!((4 * unit + 10) >> 64, (13 * unit) >> 64, "the decoy must have a DIFFERENT high limb");

    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(38,19))").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES \
         (1,4.0000000000000000010),(1,4.0000000000000000200),(1,4.0000000000000000050),(1,13.0),\
         (2,-4.0000000000000000010),(2,-4.0000000000000000200),(2,-4.0000000000000000050)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // Construction oracle: per-group min/max of the i128 mantissas.
    let g1 = [4 * unit + 10, 4 * unit + 200, 4 * unit + 50, 13 * unit];
    let g2 = [-(4 * unit + 10), -(4 * unit + 200), -(4 * unit + 50)];
    let ds = |m: i128| Decimal128::new(m, 19).to_decimal_string();
    let got = |sql: &str| -> Vec<String> {
        e.execute_resident_expr_select_sql(sql)
            .expect("tie query")
            .rows
            .iter()
            .map(|r| match &r[1] {
                SqlValue::Numeric(d) => d.to_decimal_string(),
                other => panic!("numeric MIN/MAX must be numeric, got {other:?}"),
            })
            .collect()
    };
    assert_eq!(
        got("SELECT g, MIN(v) FROM t GROUP BY g"),
        vec![ds(*g1.iter().min().unwrap()), ds(*g2.iter().min().unwrap())],
        "MIN decided by the unsigned low limb among high-limb ties (decoy must not leak)"
    );
    assert_eq!(
        got("SELECT g, MAX(v) FROM t GROUP BY g"),
        vec![ds(*g1.iter().max().unwrap()), ds(*g2.iter().max().unwrap())],
        "MAX decided by the unsigned low limb among high-limb ties"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_group_by_two_level_at_scale() {
    // The two-level shared-mem GROUP BY at scale: LOW cardinality (many rows per group, exercising the
    // block-local aggregation + cross-block merge) and HIGH cardinality (thousands of distinct keys
    // across many blocks). Both assert against host oracles -- a wrong merge would surface here.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE lo (g INT, v INT)").unwrap();
    let n = 10_000usize;
    let ngroups = 7usize;
    let mut counts = vec![0i64; ngroups];
    let mut sums = vec![0i64; ngroups];
    let mut values = String::with_capacity(n * 8);
    for i in 0..n {
        let g = i % ngroups;
        let v = (i % 13) as i64;
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({g},{v})"));
        counts[g] += 1;
        sums[g] += v;
    }
    e.execute_text(2, &format!("INSERT INTO lo (g,v) VALUES {values}")).unwrap();
    let snapshot = e.populate_relational_residency_snapshot("lo").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let c = e.execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM lo GROUP BY g").expect("lo count");
    let exp_c: Vec<Vec<SqlValue>> = (0..ngroups)
        .map(|g| vec![SqlValue::Int4(g as i32), SqlValue::Int8(counts[g])])
        .collect();
    assert_eq!(c.rows, exp_c, "low-card COUNT at scale (two-level merge)");
    let s = e.execute_resident_expr_select_sql("SELECT g, SUM(v) FROM lo GROUP BY g").expect("lo sum");
    let exp_s: Vec<Vec<SqlValue>> = (0..ngroups)
        .map(|g| vec![SqlValue::Int4(g as i32), SqlValue::Int8(sums[g])])
        .collect();
    assert_eq!(s.rows, exp_s, "low-card SUM at scale");

    // HIGH cardinality: every key distinct -> one group per row, merged across many blocks.
    let mut e2 = Engine::new_local();
    e2.execute_text(1, "CREATE TABLE hi (g INT, v INT)").unwrap();
    let h = 3000usize;
    let mut hv = String::with_capacity(h * 10);
    for i in 0..h {
        if i > 0 {
            hv.push(',');
        }
        hv.push_str(&format!("({},{})", i as i64, (i * 2) as i64));
    }
    e2.execute_text(2, &format!("INSERT INTO hi (g,v) VALUES {hv}")).unwrap();
    if e2.populate_relational_residency_snapshot("hi").unwrap().device_memory_proof.is_none() {
        return;
    }
    let hc = e2.execute_resident_expr_select_sql("SELECT g, SUM(v) FROM hi GROUP BY g").expect("hi sum");
    assert_eq!(hc.rows.len(), h, "high-card: one group per distinct key");
    // Each key i -> single row, sum = 2*i; sorted by key.
    let exp_hi: Vec<Vec<SqlValue>> = (0..h)
        .map(|i| vec![SqlValue::Int4(i as i32), SqlValue::Int8((i * 2) as i64)])
        .collect();
    assert_eq!(hc.rows, exp_hi, "high-card SUM per distinct key");
}

#[test]
#[ignore = "GPU benchmark (run with --nocapture): two-level vs single-level GROUP BY"]
fn gpu_group_by_two_level_vs_single_level_bench() {
    use std::time::Instant;
    let mut e = Engine::new_local();
    // One table, four key columns of different cardinality over the same rows -> one residency, four
    // GROUP BY cardinalities. g4/g64/g4k cycle; gall is all-distinct (high cardinality).
    e.execute_text(1, "CREATE TABLE t (g4 INT, g64 INT, g4k INT, gall INT, v INT)").unwrap();
    let n = 200_000usize;
    let chunk = 20_000usize;
    let mut txid = 2u64;
    let mut i = 0;
    while i < n {
        let end = (i + chunk).min(n);
        let mut vals = String::with_capacity(chunk * 28);
        for j in i..end {
            if j > i {
                vals.push(',');
            }
            vals.push_str(&format!("({},{},{},{},{})", j % 4, j % 64, j % 4096, j, j % 100));
        }
        e.execute_text(txid, &format!("INSERT INTO t (g4,g64,g4k,gall,v) VALUES {vals}")).unwrap();
        txid += 1;
        i = end;
    }
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        eprintln!("no GPU residency; skipping benchmark");
        return;
    }

    let _ = Instant::now(); // (full-call latency is alloc/D2H-bound; we time the kernel via events)
    eprintln!(
        "\n=== GROUP BY g, SUM(v): two-level shared-mem vs single-level global-atomic ===\n\
         {n} rows; KERNEL-only time (CUDA events, min of 200 launches), the alloc/H2D/D2H/compact\n\
         overhead excluded; speedup = single-level / two-level kernel time"
    );
    eprintln!("{:>8}  {:>13}  {:>13}  {:>9}", "groups", "single ms", "two-lvl ms", "speedup");
    for key in ["g4", "g64", "g4k", "gall"] {
        // Correctness: both kernels must agree on (key, count, sum) before we trust the timings.
        // (min/max intentionally differ: the single-level kernel computes them, the two-level does
        // not -- so compare the COUNT/SUM aggregates both kernels produce, not the whole row.)
        let mut a = e.group_by_i32_bench("t", key, "v", false).unwrap();
        let mut b = e.group_by_i32_bench("t", key, "v", true).unwrap();
        a.sort_by_key(|r| r.key);
        b.sort_by_key(|r| r.key);
        let proj = |rows: &[gpu_db_execution::GroupByI32Row]| {
            rows.iter().map(|r| (r.key, r.count, r.sum)).collect::<Vec<_>>()
        };
        assert_eq!(proj(&a), proj(&b), "single-level and two-level disagree for {key}");
        let single = e.group_by_i32_bench_kernel_ms("t", key, "v", false, 200, 0).unwrap();
        let two = e.group_by_i32_bench_kernel_ms("t", key, "v", true, 200, 0).unwrap();
        eprintln!("{:>8}  {:>13.4}  {:>13.4}  {:>8.2}x", a.len(), single, two, single / two);
    }

    // SCALE AXIS: fixed LOW cardinality (g4 = 4 groups), growing row count. This is the axis that
    // answers "does it scale" -- the two-level win should GROW with rows (more single-level global
    // contention to avoid), unlike the cardinality table above (fixed rows, varying group count).
    eprintln!(
        "\n=== SCALE: GROUP BY g4 (4 groups fixed), growing rows (kernel-only, min of 200) ==="
    );
    eprintln!("{:>9}  {:>13}  {:>13}  {:>9}", "rows", "single ms", "two-lvl ms", "speedup");
    for rows in [25_000usize, 50_000, 100_000, 200_000] {
        let single = e.group_by_i32_bench_kernel_ms("t", "g4", "v", false, 200, rows).unwrap();
        let two = e.group_by_i32_bench_kernel_ms("t", "g4", "v", true, 200, rows).unwrap();
        eprintln!("{:>9}  {:>13.4}  {:>13.4}  {:>8.2}x", rows, single, two, single / two);
    }
    eprintln!();
}
