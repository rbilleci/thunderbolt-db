use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};
use crate::{parse_command, Command, Engine};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;

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
    let mut e = Engine::new_local_test_engine();
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
fn gpu_resident_expr_simple_int4_predicate_uses_ordered_index_route() {
    // The `int4col <cmp> literal` peephole (`compare_indices_ordered_from_payload`): the simple single-
    // column-vs-literal shape lowers to the ORDERED parallel compaction that emits surviving ROW INDICES
    // ascending with NO host sort (the general arithmetic route now shares that compactor). This
    // gate is the NON-VACUOUS proof that the new route is on the path AND ascending-correct: it uses a
    // MULTI-BLOCK payload (1000 rows >> the 256-row chunk, so matches span many blocks and exercise the
    // cross-block exclusive scan + intra-block prefix-sum scatter) with INTERLEAVED matches (a[i]=i%7),
    // and asserts the EXACT ascending projected `id` vector. A broken cross-block scatter, an off-by-one
    // in the index store, or a missing eq/ne fold would reorder or drop indices and fail this equality.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (id INT, a INT)").unwrap();

    const N: i32 = 1000;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        // a = i % 7 -> the rows matching `a = K` are {i : i % 7 == K}: interleaved across the WHOLE
        // range and spread over many 256-row blocks (the cross-block ordering is load-bearing).
        values.push_str(&format!("({i}, {})", i % 7));
    }
    e.execute_text(2, &format!("INSERT INTO t (id, a) VALUES {values}"))
        .unwrap();

    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) = parse_command("SELECT id FROM t").unwrap() else {
        unreachable!()
    };
    let col_a = ResidentExpr::Column(1);
    let lit = |k: i32| Box::new(ResidentExpr::Int4Literal(k));
    let col = || Box::new(col_a.clone());

    // (1) EQUALITY (code 0, the shape this lever targets): a = 3 -> ids {i : i%7 == 3}, ascending.
    let eq = ResidentExpr::Binary {
        op: ResidentBinaryOp::Eq,
        lhs: col(),
        rhs: lit(3),
    };
    let got_eq = e
        .execute_resident_expr_select(&select, &eq)
        .expect("a = 3 simple int4 predicate on GPU");
    let eq_expected: Vec<Vec<SqlValue>> = (0..N)
        .filter(|i| i % 7 == 3)
        .map(|i| vec![SqlValue::Int4(i)])
        .collect();
    assert!(
        eq_expected.len() > 32,
        "must be multi-warp/multi-block to exercise cross-block ordering"
    );
    assert_eq!(
        got_eq.rows, eq_expected,
        "a = 3 -> ascending ids with i%7==3"
    );
    assert_eq!(got_eq.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(got_eq.fallback_reason, None);

    // (2) RANGE (code 1): a < 3 -> ids {i : i%7 < 3}, ascending.
    let lt = ResidentExpr::Binary {
        op: ResidentBinaryOp::Lt,
        lhs: col(),
        rhs: lit(3),
    };
    let got_lt = e
        .execute_resident_expr_select(&select, &lt)
        .expect("a < 3 simple int4 predicate on GPU");
    let lt_expected: Vec<Vec<SqlValue>> = (0..N)
        .filter(|i| i % 7 < 3)
        .map(|i| vec![SqlValue::Int4(i)])
        .collect();
    assert_eq!(
        got_lt.rows, lt_expected,
        "a < 3 -> ascending ids with i%7<3"
    );

    // (3) FLIPPED operand order (literal <cmp> column): 3 > a == a < 3 (comparison flipped). Must equal
    // the a < 3 result exactly (proves the flip path maps the code correctly).
    let flipped = ResidentExpr::Binary {
        op: ResidentBinaryOp::Gt,
        lhs: lit(3),
        rhs: col(),
    };
    let got_flipped = e
        .execute_resident_expr_select(&select, &flipped)
        .expect("3 > a flipped simple int4 predicate on GPU");
    assert_eq!(
        got_flipped.rows, lt_expected,
        "3 > a must equal a < 3 (flipped comparison code)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_select_evaluates_deep_arithmetic_tree_via_vm() {
    // A DEEPER arithmetic tree than the specialized two-column facade — `WHERE (a + b) * 2 - 5 > K` — routes
    // through the engine's Expr compiler -> device bytecode VM (not the peephole), evaluated and
    // materialized on the GPU. Closed-form oracle: a[i]=b[i]=i => value = 4*i - 5 (monotone), so
    // 4i-5 > K <=> i >= (K+5+3)/4 ; with K=395, 4i > 400 <=> i >= 101. Projected a[i]=i => the
    // result rows are exactly those indices.
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    assert_eq!(
        got_or.rows, or_expected,
        "a<100 OR a>500 <=> [0,100) U [501,600)"
    );

    // a != 300  ->  everything except 300.
    let ne_pred = cmp(ResidentBinaryOp::Ne, 300);
    let got_ne = e
        .execute_resident_expr_select(&select, &ne_pred)
        .expect("a != 300 on GPU");
    let mut ne_expected: Vec<Vec<SqlValue>> = (0..300).map(|i| vec![SqlValue::Int4(i)]).collect();
    ne_expected.extend((301..N).map(|i| vec![SqlValue::Int4(i)]));
    assert_eq!(got_ne.rows, ne_expected, "a != 300 <=> all rows but 300");
}
