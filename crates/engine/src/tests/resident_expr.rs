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
    let mut e = Engine::new_local_cpu_oracle();
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
    // ascending with NO host sort (replacing `run_expr_arith_filter`'s atomic-append + host-sort). This
    // gate is the NON-VACUOUS proof that the new route is on the path AND ascending-correct: it uses a
    // MULTI-BLOCK payload (1000 rows >> the 256-row chunk, so matches span many blocks and exercise the
    // cross-block exclusive scan + intra-block prefix-sum scatter) with INTERLEAVED matches (a[i]=i%7),
    // and asserts the EXACT ascending projected `id` vector. A broken cross-block scatter, an off-by-one
    // in the index store, or a missing eq/ne fold would reorder or drop indices and fail this equality.
    let mut e = Engine::new_local_cpu_oracle();
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
fn gpu_resident_expr_where_excludes_null_operands_and_projection_carries_null() {
    // M3 (doc 21) Slice D + C: a WHERE comparison over a NULLABLE column evaluates to UNKNOWN for a NULL
    // operand ON THE GPU (the leaf mask is AND'd with the column's validity bitmap) -> the row is NOT
    // selected; and a projected nullable column carries SqlValue::Null through the gather. Column a is
    // nullable (NULL at i%4==0), b is the constant 100 (non-null). a[i]=i for the non-null rows.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    const N: i32 = 12;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        if i % 4 == 0 {
            values.push_str("(NULL, 100)");
        } else {
            values.push_str(&format!("({i}, 100)"));
        }
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
    let is_null = |i: i32| i % 4 == 0;

    // (1) WHERE a < 5: only the NON-NULL a in {1,2,3} qualify. The NULL rows (placeholder 0) would pass
    //     0 < 5 WITHOUT the validity AND — so this asserts the 3VL exclusion is load-bearing.
    let a_lt_5 = ResidentExpr::Binary {
        op: ResidentBinaryOp::Lt,
        lhs: Box::new(ResidentExpr::Column(0)),
        rhs: Box::new(ResidentExpr::Int4Literal(5)),
    };
    let r = e.execute_resident_expr_select(&select, &a_lt_5).unwrap();
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(3)]
        ],
        "WHERE a < 5 must exclude NULL rows (3VL), not fold their placeholder 0"
    );
    assert_eq!(r.fallback_reason, None);
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));

    // (2) WHERE b >= 0: b is non-null and constant 100, so EVERY row qualifies; projecting a then carries
    //     SqlValue::Null from the device validity payload. Tests projection-carries-NULL.
    let b_ge_0 = ResidentExpr::Binary {
        op: ResidentBinaryOp::Ge,
        lhs: Box::new(ResidentExpr::Column(1)),
        rhs: Box::new(ResidentExpr::Int4Literal(0)),
    };
    let r = e.execute_resident_expr_select(&select, &b_ge_0).unwrap();
    let expected_all: Vec<Vec<SqlValue>> = (0..N)
        .map(|i| {
            vec![if is_null(i) {
                SqlValue::Null
            } else {
                SqlValue::Int4(i)
            }]
        })
        .collect();
    assert_eq!(
        r.rows, expected_all,
        "projecting a nullable column must carry SqlValue::Null for the NULL rows"
    );

    // (3) col-vs-col `a < b` (b = 100): every NON-NULL a < 100 qualifies; the NULL rows are excluded even
    //     though their placeholder 0 < 100. Result a-values = the non-null i in row order.
    let a_lt_b = ResidentExpr::Binary {
        op: ResidentBinaryOp::Lt,
        lhs: Box::new(ResidentExpr::Column(0)),
        rhs: Box::new(ResidentExpr::Column(1)),
    };
    let r = e.execute_resident_expr_select(&select, &a_lt_b).unwrap();
    let expected_non_null: Vec<Vec<SqlValue>> = (0..N)
        .filter(|&i| !is_null(i))
        .map(|i| vec![SqlValue::Int4(i)])
        .collect();
    assert_eq!(
        r.rows, expected_non_null,
        "col-vs-col `a < b` must exclude NULL-a rows (3VL), not fold the placeholder 0"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_projection_carries_null_past_word_boundary() {
    // Gather-fan-out audit P3 hardening: the per-column NULL-validity bitmap is materialized by the bool
    // gather kernel (word = idx>>5, bit = idx&31, as a u8). A wrong intra-word shift (idx&7) would misread
    // bits whose position is >= 8 within a 32-bit word, AND any index past the first word (>= 32). N=100
    // with NULLs at i%7==0 scatters NULLs across validity words 0..3 at bit positions incl. 10/14/17/20/21/
    // 24/27/28/31 (e.g. idx 42->word1 bit10, 70->word2 bit6, 91->word2 bit27) -> a DIRECT check that the
    // projected nullable column carries SqlValue::Null at the right high indices through this gather.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    const N: i32 = 100;
    let is_null = |i: i32| i % 7 == 0;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        if is_null(i) {
            values.push_str("(NULL, 100)");
        } else {
            values.push_str(&format!("({i}, 100)"));
        }
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
    // WHERE b >= 0: every row qualifies, so projecting `a` materializes ALL 100 rows in order and the
    // validity gather alone decides NULL vs value at each index (incl. past word boundaries).
    let b_ge_0 = ResidentExpr::Binary {
        op: ResidentBinaryOp::Ge,
        lhs: Box::new(ResidentExpr::Column(1)),
        rhs: Box::new(ResidentExpr::Int4Literal(0)),
    };
    let r = e.execute_resident_expr_select(&select, &b_ge_0).unwrap();
    let expected: Vec<Vec<SqlValue>> = (0..N)
        .map(|i| {
            vec![if is_null(i) {
                SqlValue::Null
            } else {
                SqlValue::Int4(i)
            }]
        })
        .collect();
    assert_eq!(
        r.rows, expected,
        "nullable projection must carry SqlValue::Null at the right indices past validity word \
         boundaries (the bool/validity gather's idx&31 shift)"
    );
    assert_eq!(r.fallback_reason, None);
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_where_3vl_over_nullable_bigint() {
    // M3 (doc 21) Track A.3: a WHERE over a NULLABLE int8 (BIGINT) column evaluates to UNKNOWN for a NULL
    // operand ON THE GPU and excludes the row — routed to the i64 mask VM (elem I64), the same VM the
    // non-null int8 AND/OR path uses, with each comparison leaf AND'd with the column's validity bitmap.
    // No kernel change: the validity AND is a type-independent i32 BoolMask. v is nullable; w is non-null.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tb (id INT, v BIGINT, w BIGINT)")
        .unwrap();
    // v = [100, NULL, 300, NULL, 5000000000, 250]; w = 1000 (non-null). v=5e9 exceeds i32 -> proves the
    // i64 read is not truncated to i32. The NULL placeholder is 0, which passes `0 < 300` WITHOUT the
    // validity AND, so the NULL exclusions below are load-bearing.
    e.execute_text(
        2,
        "INSERT INTO tb (id,v,w) VALUES \
         (1,100,1000),(2,NULL,1000),(3,300,1000),(4,NULL,1000),(5,5000000000,1000),(6,250,1000)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tb").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // (1) simple scalar `v < 300`: non-NULL v in {100, 250} -> ids 1, 6 (index order). NULL rows excluded;
    //     5e9 not < 300.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tb WHERE v < 300")
        .expect("WHERE over a nullable bigint runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(6)]],
        "WHERE v < 300 must exclude NULL rows (3VL), not fold the placeholder 0"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));

    // (2) compound AND `v > 50 AND v < 400`: v in {100, 300, 250} -> ids 1, 3, 6. The i64 VM combines two
    //     i64 comparison leaves with AND, each validity-masked. NULLs excluded.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tb WHERE v > 50 AND v < 400")
        .expect("compound AND over a nullable bigint runs on the GPU");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(3)],
            vec![SqlValue::Int4(6)]
        ],
        "compound AND over a nullable bigint must exclude NULL rows"
    );

    // (3) col-vs-col `v < w` (w = 1000): non-NULL v < 1000 -> ids 1, 3, 6 (5e9 not < 1000). NULLs excluded.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tb WHERE v < w")
        .expect("col-vs-col over a nullable bigint runs on the GPU");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(3)],
            vec![SqlValue::Int4(6)]
        ],
        "col-vs-col v < w must exclude NULL-v rows (3VL)"
    );

    // (4) large-value `v > 1000000000`: only id 5 (5e9). Proves the i64 storage/read is exact (a truncated
    //     i32 read of 5000000000 = 705032704 would mis-answer), and the NULL placeholder 0 is excluded.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tb WHERE v > 1000000000")
        .expect("large-value bigint compare runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(5)]],
        "WHERE v > 1e9 must match only the 5e9 row (i64 read, not truncated), NULLs excluded"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_where_nullable_mixed_type_clean_errors() {
    // M3 (doc 21): WHERE 3VL now covers EVERY nullable SCALAR type (int2/int4/int8/text/bool/date/
    // timestamp/numeric/uuid), so the remaining clean-errors are MIXED-type predicates the mono-typed VM
    // can't lower — never a silent mis-answer.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tm (a INT, big BIGINT, n NUMERIC(10,2))")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO tm (a,big,n) VALUES (1,10,1.50),(2,NULL,NULL),(3,30,3.50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tm").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // Mixed numeric + int4 predicate (nullable n) -> clean error (mixed numeric/integer).
    let err = e
        .execute_resident_expr_select_sql("SELECT a FROM tm WHERE n < 3.00 AND a < 5")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("mixed numeric/integer") || err.contains("not yet supported"),
        "mixed numeric/int nullable WHERE must clean-error, got: {err}"
    );
    // Mixed int4 + NULLABLE int8 now RUNS at I32 (ADR-006 mixed-width groups: the nullable
    // branch's third local gate; the int8 leaf carries its own validity AND). `big > 5 AND a < 3`
    // -> a=1 (big 10); a=2 is NULL-big (excluded by 3VL); a=3 fails a<3.
    let r = e
        .execute_resident_expr_select_sql("SELECT a FROM tm WHERE big > 5 AND a < 3")
        .expect("mixed int4 + nullable int8 WHERE runs on the GPU (mixed-width group)");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)]],
        "big>5 AND a<3 => a=1 only (NULL-big row excluded by 3VL)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_where_3vl_over_nullable_date() {
    // M3 (doc 21): a WHERE over a nullable DATE column evaluates to UNKNOWN for a NULL operand on the GPU
    // and excludes the row. A date is i32 days, so it routes to the I32 mask VM (CompareScalar over the
    // days literal) with the column's validity AND'd in. The NULL placeholder is day 0 (< any real date),
    // so it would pass `d < '2024-01-20'` WITHOUT the validity AND -> the exclusions below are load-bearing.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE td (id INT, d DATE)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO td (id,d) VALUES \
         (1,'2024-01-05'),(2,NULL),(3,'2024-01-15'),(4,NULL),(5,'2024-01-25')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("td").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // d < '2024-01-20' -> id 1 (05), 3 (15). NULLs excluded.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM td WHERE d < '2024-01-20'")
        .expect("WHERE over a nullable date runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(3)]],
        "WHERE d < date must exclude NULL rows (3VL), not fold the placeholder day 0"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    // literal on the LEFT: '2024-01-10' < d -> id 3 (15), 5 (25). NULLs excluded.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM td WHERE '2024-01-10' < d")
        .expect("date-literal-on-left over a nullable date runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(5)]],
        "WHERE date < d must exclude NULL rows (3VL)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_where_3vl_over_nullable_timestamp() {
    // M3 (doc 21): a WHERE over a nullable TIMESTAMP column excludes NULL rows on the GPU. A timestamp is
    // i64 microseconds whose literal exceeds the VM's i32 CompareScalar, so it routes to the I64 VM via
    // the new CompareScalarI64 step (scalar) or CompareBuffers (col-vs-col), with the validity AND'd in.
    // The NULL placeholder is micros 0 (< any 2024 timestamp), so the exclusions are load-bearing.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tts (id INT, ts TIMESTAMP, ts2 TIMESTAMP)")
        .unwrap();
    // ts nullable; ts2 = noon (non-null) for the col-vs-col case.
    e.execute_text(
        2,
        "INSERT INTO tts (id,ts,ts2) VALUES \
         (1,'2024-01-15 09:00:00','2024-01-15 12:00:00'),\
         (2,NULL,'2024-01-15 12:00:00'),\
         (3,'2024-01-15 15:00:00','2024-01-15 12:00:00'),\
         (4,NULL,'2024-01-15 12:00:00'),\
         (5,'2024-01-15 11:00:00','2024-01-15 12:00:00')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tts").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // scalar `ts < '2024-01-15 12:00:00'` -> ts before noon: id 1 (09:00), 5 (11:00). NULLs excluded.
    // Exercises CompareScalarI64 (the i64 micros literal).
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tts WHERE ts < '2024-01-15 12:00:00'")
        .expect("WHERE over a nullable timestamp runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(5)]],
        "WHERE ts < timestamp must exclude NULL rows (3VL), not fold the placeholder micros 0"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    // col-vs-col `ts < ts2` (ts2 = noon): same survivors id 1, 5. Exercises CompareBuffers (I64) + validity.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tts WHERE ts < ts2")
        .expect("col-vs-col over a nullable timestamp runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(5)]],
        "col-vs-col ts < ts2 must exclude NULL-ts rows (3VL)"
    );
    // A COMPOUND nullable timestamp predicate now RUNS ON THE GPU (ADR-006: the nullable branch's
    // local I64 gate + the VM's timestamp scalar leaf, which parses the TextLiteral bound to micros
    // via `LoadColumnI64` + `CompareScalarI64` + the per-leaf validity AND). `ts < noon AND
    // ts2 > 06:00` -> ts before noon (ids 1, 5); the NULL-ts rows (2, 4) are excluded by 3VL even
    // though their placeholder micros 0 satisfies `< noon` — the validity AND is load-bearing.
    let r = e
        .execute_resident_expr_select_sql(
            "SELECT id FROM tts WHERE ts < '2024-01-15 12:00:00' AND ts2 > '2024-01-15 06:00:00'",
        )
        .expect("compound nullable timestamp WHERE runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(5)]],
        "compound nullable ts WHERE: ids 1,5 (before noon); NULL rows excluded (3VL, placeholder \
         micros 0 would otherwise match)"
    );
    // ts-vs-ts COL-VS-COL inside an AND (audit LOW adopted — the newly-reachable shape): lowers via
    // the general CompareBuffers arm at I64 (two 8-byte loads) + per-leaf validity. `ts < ts2 AND
    // ts2 > 06:00` -> the same ids 1,5; NULL-ts rows excluded even though placeholder 0 < noon.
    let r = e
        .execute_resident_expr_select_sql(
            "SELECT id FROM tts WHERE ts < ts2 AND ts2 > '2024-01-15 06:00:00'",
        )
        .expect("ts-vs-ts col-vs-col inside AND runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(5)]],
        "col-vs-col ts < ts2 inside AND: ids 1,5; NULL rows excluded (3VL)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_mixed_width_where_3vl() {
    // ADR-006 (MIXED-WIDTH groups): a WHERE mixing an INT8 scalar leaf with i32-servable leaves
    // (text/bool/int4/timestamp) runs ON THE GPU at I32 — the int8 leaf loads 8 bytes via the
    // width-safe `LoadColumnI64` arm (`mixed_width_i32_elem`). Used to hard-error ("mixed
    // int8/text") -> CPU. big values straddle i32::MAX so a 4-byte mis-read can't fake the
    // answer; the NULL-big rows pin 3VL with a PLACEHOLDER-SPANNING bound (placeholder 0
    // satisfies `big >= 0` — only the validity AND excludes them; reads have NO recheck net).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE tmw (id INT, big BIGINT, name TEXT, f BOOL)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO tmw (id,big,name,f) VALUES \
         (1,4294967300,'hit',true),\
         (2,NULL,'hit',true),\
         (3,50,'hit',false),\
         (4,NULL,'miss',true),\
         (5,4294967301,'miss',true),\
         (6,4294967302,'hit',false)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tmw").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // NULLABLE mixed int8+text (the local third-branch gate): `big > i32::MAX AND name = 'hit'`
    // -> ids 1, 6 (id 3 fails the range at 50, id 5 fails the eq, NULLs 2/4 excluded by 3VL).
    let r = e
        .execute_resident_expr_select_sql(
            "SELECT id FROM tmw WHERE big > 2147483647 AND name = 'hit'",
        )
        .expect("mixed int8+text WHERE runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(6)]],
        "mixed int8+text AND: full-width int8 compare + text eq; NULLs excluded"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    // PLACEHOLDER-SPANNING 3VL pin: `big >= 0 AND name = 'hit'` — the NULL placeholder (0)
    // satisfies the bound, so ONLY the per-leaf validity AND keeps ids 2 out.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tmw WHERE big >= 0 AND name = 'hit'")
        .expect("placeholder-spanning mixed WHERE runs on the GPU");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(3)],
            vec![SqlValue::Int4(6)]
        ],
        "placeholder-spanning mixed AND: NULL-big row 2 must be excluded by validity, not value"
    );
    // MIXED bool+int8 (`f = true AND big > i32::MAX` -> ids 1, 5): the bool leaf is a mask step,
    // the int8 leaf an 8-byte scalar compare, composed in one I32 program.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tmw WHERE f = true AND big > 2147483647")
        .expect("mixed bool+int8 WHERE runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(5)]],
        "mixed bool+int8 AND: ids 1,5"
    );
    // NON-NULL mixed int8+text (the int8-general-path fall-through, no nullable branch): a
    // separate all-non-null table.
    e.execute_text(
        3,
        "CREATE TABLE tmw2 (id INT, big BIGINT, name TEXT, small SMALLINT)",
    )
    .unwrap();
    e.execute_text(
        4,
        "INSERT INTO tmw2 (id,big,name,small) VALUES \
         (1,4294967300,'a',7),(2,10,'a',7),(3,4294967301,'b',9)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tmw2").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let r = e
        .execute_resident_expr_select_sql(
            "SELECT id FROM tmw2 WHERE big > 2147483647 AND name = 'a'",
        )
        .expect("non-null mixed int8+text WHERE runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)]],
        "non-null mixed int8+text AND: id 1 only (2 fails the range, 3 fails the eq)"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    // BOOL INEQUALITIES over the NULLABLE-table VM path (ADR-006: PG `false < true`; each shape
    // constant-folds — `f < true` ⇔ `f = false`; `f <= true` ⇔ constant-TRUE-for-KNOWN). tmw has
    // f=[true,true,false,true,true] and NULL big rows but NON-null f — pin the fold shapes here,
    // then the 3VL trap on a nullable-bool table below.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tmw WHERE f < true AND big >= 0")
        .expect("bool inequality inside AND runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(6)]],
        "f < true ⇔ f = false -> ids 3,6 (NULL-big rows excluded by the int8 leaf's 3VL)"
    );
    // NULLABLE-BOOL 3VL trap: `nb <= true` folds to a CONSTANT-TRUE mask that never reads the
    // value bitmap — ONLY the validity AND can exclude the NULL row (UNKNOWN, PG 3VL).
    e.execute_text(5, "CREATE TABLE tbn (id INT, nb BOOL)")
        .unwrap();
    e.execute_text(
        6,
        "INSERT INTO tbn (id,nb) VALUES (1,true),(2,NULL),(3,false)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tbn").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tbn WHERE nb <= true AND id > 0")
        .expect("nullable-bool <= true runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(3)]],
        "nb <= true is TRUE for known bools only — the NULL row must be excluded by validity"
    );
    // The empty-set fold: `nb > true` matches nothing (not even NULL).
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tbn WHERE nb > true AND id > 0")
        .expect("nullable-bool > true runs on the GPU");
    assert_eq!(
        r.rows,
        Vec::<Vec<SqlValue>>::new(),
        "nb > true is constant FALSE"
    );
    // Literal-on-left flips the op: `true > nb` ⇔ `nb < true` ⇔ `nb = false`.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tbn WHERE true > nb AND id > 0")
        .expect("literal-on-left bool inequality runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(3)]],
        "true > nb ⇔ nb = false -> id 3 (NULL excluded)"
    );
    // Audit polarity-gap pins: the Ge const-true shape, and the remaining two mask shapes.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tbn WHERE nb >= false AND id > 0")
        .expect("nb >= false runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(3)]],
        "nb >= false is TRUE for known bools only (the Ge const-true fold; NULL excluded)"
    );
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tbn WHERE nb <= false AND id > 0")
        .expect("nb <= false runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(3)]],
        "nb <= false ⇔ nb = false"
    );
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tbn WHERE nb >= true AND id > 0")
        .expect("nb >= true runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)]],
        "nb >= true ⇔ nb = true"
    );
    // The PEEPHOLE const-true ALL-indices arm (non-null single comparison, no AND): tmw's `f` is
    // a NON-null bool (the nullable-branch gate keys on the REFERENCED columns only), so this is
    // the direct pin of try_lower_bool_predicate's (0..n) return.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tmw WHERE f <= true")
        .expect("non-null bare f <= true runs on the GPU (peephole const-true)");
    assert_eq!(
        r.rows,
        (1..=6).map(|i| vec![SqlValue::Int4(i)]).collect::<Vec<_>>(),
        "f <= true over a NON-null bool = every row (the peephole all-indices arm)"
    );

    // NON-NULL mixed int8+int2 (the audit's MEDIUM: this shape used to SLIP PAST the mixed
    // block — int2 was unchecked — into the I64 compile, where the 4-byte int2 slot was read at
    // an 8-byte stride = silent wrong rows). Now it routes to I32 like the rest: id 1 only
    // (id 2 fails the big range, id 3 fails small=7).
    let r = e
        .execute_resident_expr_select_sql(
            "SELECT id FROM tmw2 WHERE big > 2147483647 AND small = 7",
        )
        .expect("non-null mixed int8+int2 WHERE runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)]],
        "non-null mixed int8+int2 AND: id 1 only (an 8-byte int2 mis-read could not produce this)"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_col_vs_col_text_uuid() {
    // ADR-006 (col-vs-col, the LAST predicate edge): `text_a <cmp> text_b` runs per-row on the
    // NEW two-column lexicographic byte-compare kernel (singles + inside AND/OR), and
    // `uuid_a <cmp> uuid_b` composes inside AND/OR via the existing b128 columns kernel as a
    // mask step. Rows are ADVERSARIAL for the text kernel (the hand-PTX doctrine): byte order
    // ('B' < 'b'), shorter-prefix-first ('ab' < 'b'), length tiebreak ('ab' > 'a'), equal,
    // empty-vs-nonempty. 3VL: a NULL operand's placeholder (empty span / 16 zero bytes) sorts
    // below everything — only the BOTH-validity AND keeps those rows out.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE tcc (id INT, a TEXT, b TEXT, u1 UUID, u2 UUID)",
    )
    .unwrap();
    let u = |n: u32| format!("00000000-0000-0000-0000-{n:012x}");
    let z2 = u(2);
    e.execute_text(
        2,
        &format!(
            "INSERT INTO tcc (id,a,b,u1,u2) VALUES \
             (1,'B','b','{0}','{z2}'),\
             (2,'ab','b','{z2}','{z2}'),\
             (3,'ab','a','{z2}','{0}'),\
             (4,'same','same','{0}','{0}'),\
             (5,'','x','{0}','{z2}'),\
             (6,NULL,'x',NULL,'{z2}'),\
             (7,'q',NULL,'{0}',NULL),\
             (8,'zé','za','{z2}','{z2}'),\
             (9,'abc','abd','{z2}','{z2}')",
            u(1)
        ),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tcc").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // SINGLE text col-vs-col `a < b`: 1 ('B'<'b' byte order), 2 ('ab'<'b' first byte), 5 (''<'x'
    // shorter-first). 3 ('ab'>'a'), 4 (equal) fail; 6/7 are NULL-operand rows — the placeholder
    // empty span would sort below everything, only validity excludes them.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tcc WHERE a < b")
        .expect("text col-vs-col single runs on the GPU");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(5)],
            vec![SqlValue::Int4(9)],
        ],
        "a < b: byte order + shorter-first + multi-char prefix ('abc'<'abd'); row 8 ('z\u{e9}' vs \
         'za') must NOT match - the 0xC3 lead byte compares UNSIGNED above 'a' (a signed compare \
         would leak it); NULL rows excluded by BOTH-validity 3VL"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    // SINGLE `a = b`: row 4 only (empty-vs-'x' and NULLs excluded).
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tcc WHERE a = b")
        .expect("text col-vs-col equality runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(4)]],
        "a = b: the equal row only"
    );
    // Text col-vs-col INSIDE AND (the mask-VM composition): `a >= b AND id < 7` -> 3, 4
    // (row 7's NULL b excluded by validity, not by the id bound — id 7 fails both).
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tcc WHERE a >= b AND id < 90")
        .expect("text col-vs-col inside AND runs on the GPU");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(3)],
            vec![SqlValue::Int4(4)],
            vec![SqlValue::Int4(8)]
        ],
        "a >= b AND id: length tiebreak ('ab' > 'a') + equal + UNSIGNED high-bit ('z\u{e9}' >= \
         'za'); NULL-b row 7 excluded (3VL)"
    );
    // UUID col-vs-col INSIDE AND: `u1 < u2 AND id < 90` -> 1, 5 (byte-wise b128; row 2 equal,
    // row 3 u1>u2, row 4 equal; row 6's NULL u1 = 16 ZERO bytes < u2 — validity excludes it).
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tcc WHERE u1 < u2 AND id < 90")
        .expect("uuid col-vs-col inside AND runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(5)]],
        "u1 < u2 in AND: NULL-u1 row 6 (zero-byte placeholder < u2) excluded by validity"
    );
    // UUID equality in AND: `u1 = u2 AND id < 90` -> 2, 4 (row 7's NULL u2 excluded).
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tcc WHERE u1 = u2 AND id < 90")
        .expect("uuid col-vs-col equality inside AND runs on the GPU");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(4)],
            vec![SqlValue::Int4(8)],
            vec![SqlValue::Int4(9)],
        ],
        "u1 = u2 in AND: rows 2,4,8,9; NULL rows excluded (3VL)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_where_3vl_over_nullable_numeric() {
    // M3 (doc 21): a WHERE over a nullable NUMERIC column excludes NULL rows on the GPU. A numeric is an
    // i128 mantissa whose value exceeds the VM's i32 CompareScalar, so a same-or-coarser-scale comparison
    // routes to the I128 VM via the new CompareScalarI128 step (scalar) or CompareBuffers (col-vs-col),
    // with the validity AND'd in. The NULL placeholder is mantissa 0 (= 0.00, < 100.00), so the
    // exclusions are load-bearing.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE tn2 (id INT, amt NUMERIC(10,2), amt2 NUMERIC(10,2))",
    )
    .unwrap();
    // amt nullable = [10.50, NULL, 30.25, NULL, 250.75]; amt2 = 100.00 (non-null) for col-vs-col.
    e.execute_text(
        2,
        "INSERT INTO tn2 (id,amt,amt2) VALUES \
         (1,10.50,100.00),(2,NULL,100.00),(3,30.25,100.00),(4,NULL,100.00),(5,250.75,100.00)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tn2").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // scalar `amt < 100.00` -> 10.50, 30.25 -> id 1, 3. NULLs excluded (CompareScalarI128).
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tn2 WHERE amt < 100.00")
        .expect("WHERE over a nullable numeric runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(3)]],
        "WHERE amt < numeric must exclude NULL rows (3VL), not fold the placeholder 0.00"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    // coarser INTEGER literal `amt < 100` (scale 0 <= column scale 2, rescaled up) -> same id 1, 3.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tn2 WHERE amt < 100")
        .expect("nullable numeric vs an integer literal runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(3)]],
        "amt < 100 (int literal coerced to numeric) excludes NULLs"
    );
    // literal on the LEFT `100 < amt` -> amt > 100 -> id 5 (250.75). NULLs excluded.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tn2 WHERE 100 < amt")
        .expect("numeric-literal-on-left over a nullable numeric runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(5)]],
        "100 < amt must exclude NULL rows (3VL)"
    );
    // col-vs-col `amt < amt2` (amt2 = 100.00, same scale) -> id 1, 3. CompareBuffers (I128) + validity.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tn2 WHERE amt < amt2")
        .expect("col-vs-col over a nullable numeric runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(3)]],
        "col-vs-col amt < amt2 must exclude NULL-amt rows (3VL)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_where_3vl_over_nullable_uuid() {
    // M3 (doc 21): a WHERE over a nullable UUID column excludes NULL rows on the GPU. UUID compares by an
    // unsigned big-endian memcmp (a dedicated kernel, NOT a VM step), so the compare mask is AND'd with
    // the column's validity mask in the launcher (compact_mask_with_validity). The NULL placeholder is 16
    // zero bytes (= uuid ...00), so `u = ...00` and `u < ...0a` would WRONGLY include NULL rows without
    // the validity AND -> the assertions are load-bearing.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tu (id INT, u UUID, u2 UUID)")
        .unwrap();
    let uuid_for = |i: i64| format!("00000000-0000-0000-0000-0000000000{i:02x}");
    let peer = uuid_for(10);
    // u nullable = [..05, NULL, ..0f, NULL, ..14]; u2 = ..0a (non-null) for col-vs-col.
    e.execute_text(
        2,
        &format!(
            "INSERT INTO tu (id,u,u2) VALUES \
             (1,'{}','{peer}'),(2,NULL,'{peer}'),(3,'{}','{peer}'),(4,NULL,'{peer}'),(5,'{}','{peer}')",
            uuid_for(5),
            uuid_for(15),
            uuid_for(20),
        ),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tu").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // `u < ..0a` -> u in {..05} -> id 1. NULLs (placeholder ..00 < ..0a) excluded.
    let r = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT id FROM tu WHERE u < '{}'",
            uuid_for(10)
        ))
        .expect("WHERE over a nullable uuid runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)]],
        "WHERE u < uuid must exclude NULL rows (3VL), not fold the placeholder ..00"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    // `u = ..00`: no real u is ..00, and the NULL placeholder IS ..00 -> WITHOUT the validity AND the
    // NULL rows would match. With it, the result is EMPTY. The decisive equality test.
    let r = e
        .execute_resident_expr_select_sql(&format!("SELECT id FROM tu WHERE u = '{}'", uuid_for(0)))
        .expect("uuid equality over a nullable uuid runs on the GPU");
    assert!(
        r.rows.is_empty(),
        "u = ..00 must be EMPTY: no real u is ..00 and NULL rows (placeholder ..00) are excluded, got {:?}",
        r.rows
    );
    // `u > ..0a` -> u in {..0f, ..14} -> id 3, 5. NULLs excluded.
    let r = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT id FROM tu WHERE u > '{}'",
            uuid_for(10)
        ))
        .expect("uuid > over a nullable uuid runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(5)]],
        "WHERE u > uuid must exclude NULL rows (3VL)"
    );
    // col-vs-col `u < u2` (u2 = ..0a): u < 10 -> id 1. NULLs excluded (validity AND on u only).
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tu WHERE u < u2")
        .expect("col-vs-col over a nullable uuid runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)]],
        "col-vs-col u < u2 must exclude NULL-u rows (3VL)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_where_3vl_over_nullable_int2() {
    // M3 (doc 21): a WHERE over a nullable SMALLINT (int2) column excludes NULL rows on the GPU. int2 is
    // stored widened to i32 in the int4 section, so it routes on the I32 mask VM exactly like int4 (incl.
    // AND/OR). The NULL placeholder is 0 (< 20), so the exclusions are load-bearing.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE ti (id INT, s SMALLINT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO ti (id,s) VALUES (1,5),(2,NULL),(3,15),(4,NULL),(5,25)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("ti").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // s < 20 -> {5, 15} -> id 1, 3. NULLs (placeholder 0 < 20) excluded.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM ti WHERE s < 20")
        .expect("WHERE over a nullable smallint runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(3)]],
        "WHERE s < 20 must exclude NULL rows (3VL), not fold the placeholder 0"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    // compound AND `s > 10 AND s < 30` -> {15, 25} -> id 3, 5. NULLs excluded.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM ti WHERE s > 10 AND s < 30")
        .expect("compound AND over a nullable smallint runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(5)]],
        "compound AND over a nullable smallint must exclude NULL rows"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_where_3vl_over_nullable_numeric_compound_and_cross_scale() {
    // M3 (doc 21): a nullable NUMERIC WHERE also runs for AND/OR and a FINER cross-scale literal — these
    // route through the validity-aware compile_numeric_compare VM path (push_leaf_validity_and per leaf).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tnc (id INT, amt NUMERIC(10,2))")
        .unwrap();
    // amt nullable = [10.50, NULL, 30.25, NULL, 250.75].
    e.execute_text(
        2,
        "INSERT INTO tnc (id,amt) VALUES (1,10.50),(2,NULL),(3,30.25),(4,NULL),(5,250.75)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tnc").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // AND `amt > 20.00 AND amt < 100.00` -> {30.25} -> id 3. NULLs excluded (each leaf validity-masked).
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tnc WHERE amt > 20.00 AND amt < 100.00")
        .expect("AND over a nullable numeric runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(3)]],
        "AND over a nullable numeric must exclude NULL rows"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    // OR `amt < 20.00 OR amt > 200.00` -> {10.50, 250.75} -> id 1, 5. NULLs excluded (NULL OR NULL = NULL).
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tnc WHERE amt < 20.00 OR amt > 200.00")
        .expect("OR over a nullable numeric runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(5)]],
        "OR over a nullable numeric must exclude rows where both leaves are UNKNOWN"
    );
    // FINER cross-scale literal `amt < 30.255` (scale 3 > column scale 2): the column is rescaled UP to
    // scale 3 -> {10.50, 30.25} -> id 1, 3. NULLs excluded.
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM tnc WHERE amt < 30.255")
        .expect("cross-scale literal over a nullable numeric runs on the GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(3)]],
        "cross-scale amt < 30.255 must exclude NULL rows (3VL)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_order_by_places_nulls_per_pg_default() {
    // M3 (doc 21) Slice E: ORDER BY a NULLABLE int column places NULLs at PG's DEFAULT end ON THE GPU
    // sort — last under ASC, first under DESC (the i64::MAX sentinel realizes both). Without it the sort
    // would error on the NULL value. a is nullable; b = 100 (non-null) so `WHERE b >= 0` keeps every row.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    // a = [3, NULL, 1, NULL, 2]
    e.execute_text(
        2,
        "INSERT INTO t (a, b) VALUES (3, 100), (NULL, 100), (1, 100), (NULL, 100), (2, 100)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let all_rows = ResidentExpr::Binary {
        op: ResidentBinaryOp::Ge,
        lhs: Box::new(ResidentExpr::Column(1)),
        rhs: Box::new(ResidentExpr::Int4Literal(0)),
    };

    // ASC: non-NULL ascending then NULLs last.
    let Command::Select(asc) = parse_command("SELECT a FROM t ORDER BY a").unwrap() else {
        unreachable!()
    };
    let r = e.execute_resident_expr_select(&asc, &all_rows).unwrap();
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(3)],
            vec![SqlValue::Null],
            vec![SqlValue::Null],
        ],
        "ORDER BY a (ASC) must place NULLs last (PG default)"
    );

    // DESC: NULLs first then non-NULL descending.
    let Command::Select(desc) = parse_command("SELECT a FROM t ORDER BY a DESC").unwrap() else {
        unreachable!()
    };
    let r = e.execute_resident_expr_select(&desc, &all_rows).unwrap();
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null],
            vec![SqlValue::Null],
            vec![SqlValue::Int4(3)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(1)],
        ],
        "ORDER BY a DESC must place NULLs first (PG default)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_order_by_multikey_nullable_placement_on_device() {
    // M3 (doc 21): a MULTI-key ORDER BY with NULLs in BOTH a nullable int key and a nullable text key,
    // placed entirely ON-DEVICE — the hetero sort comparator reads each key's validity bitmap per row and
    // orders NULL as greatest (PG default: last ASC / first DESC), per key. No host partition / overwrite.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tmk (na INT, t TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO tmk (na,t) VALUES (1,'b'),(NULL,'a'),(1,NULL),(NULL,NULL),(2,'a')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tmk").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // ORDER BY na, t (ASC, ASC): na groups 1,2,NULL (NULL last); within na, t ASC with NULL last.
    let r = e
        .execute_resident_expr_select_sql("SELECT na, t FROM tmk ORDER BY na, t")
        .expect("multi-key nullable ORDER BY runs on the GPU");
    let txt = |s: &str| SqlValue::Text(s.to_string());
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(1), txt("b")],
            vec![SqlValue::Int4(1), SqlValue::Null],
            vec![SqlValue::Int4(2), txt("a")],
            vec![SqlValue::Null, txt("a")],
            vec![SqlValue::Null, SqlValue::Null],
        ],
        "multi-key (nullable int, nullable text) places NULLs per key at PG default, on-device"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_order_by_nullable_text_numeric_uuid_keys_place_nulls() {
    // M3 (doc 21): ORDER BY a SOLE nullable TEXT / NUMERIC / UUID key places NULLs at PG's default end
    // (last ASC, first DESC) — the on-device hetero sort comparator reads the key's validity bitmap and
    // orders the rest, then NULLs are placed. Without it the hetero comparator would mis-place NULLs (it
    // reads the placeholder, not the validity bitmap).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tonull (k TEXT, n NUMERIC(10,2), u UUID)")
        .unwrap();
    let uuid_for = |i: i64| format!("00000000-0000-0000-0000-0000000000{i:02x}");
    e.execute_text(
        2,
        &format!(
            "INSERT INTO tonull (k,n,u) VALUES \
             ('b',2.50,'{}'),(NULL,NULL,NULL),('a',1.50,'{}'),(NULL,NULL,NULL),('c',3.50,'{}')",
            uuid_for(2),
            uuid_for(1),
            uuid_for(3),
        ),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tonull").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let uuid = |i: i64| SqlValue::Uuid(gpu_db_sql::uuid::parse_uuid(&uuid_for(i)).expect("uuid"));

    // TEXT ASC: 'a','b','c' then NULLs last.
    let r = e
        .execute_resident_expr_select_sql("SELECT k FROM tonull ORDER BY k")
        .expect("ORDER BY nullable text ASC");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Text("a".to_string())],
            vec![SqlValue::Text("b".to_string())],
            vec![SqlValue::Text("c".to_string())],
            vec![SqlValue::Null],
            vec![SqlValue::Null],
        ],
        "ORDER BY nullable text ASC: non-NULL ascending then NULLs last"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    // TEXT DESC: NULLs first then 'c','b','a'.
    let r = e
        .execute_resident_expr_select_sql("SELECT k FROM tonull ORDER BY k DESC")
        .expect("ORDER BY nullable text DESC");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null],
            vec![SqlValue::Null],
            vec![SqlValue::Text("c".to_string())],
            vec![SqlValue::Text("b".to_string())],
            vec![SqlValue::Text("a".to_string())],
        ],
        "ORDER BY nullable text DESC: NULLs first then non-NULL descending"
    );

    // NUMERIC ASC: 1.50, 2.50, 3.50 then NULLs last.
    let r = e
        .execute_resident_expr_select_sql("SELECT n FROM tonull ORDER BY n")
        .expect("ORDER BY nullable numeric ASC");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Numeric(Decimal128::new(150, 2))],
            vec![SqlValue::Numeric(Decimal128::new(250, 2))],
            vec![SqlValue::Numeric(Decimal128::new(350, 2))],
            vec![SqlValue::Null],
            vec![SqlValue::Null],
        ],
        "ORDER BY nullable numeric ASC: non-NULL ascending then NULLs last"
    );

    // UUID DESC: NULLs first then ..03, ..02, ..01.
    let r = e
        .execute_resident_expr_select_sql("SELECT u FROM tonull ORDER BY u DESC")
        .expect("ORDER BY nullable uuid DESC");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null],
            vec![SqlValue::Null],
            vec![uuid(3)],
            vec![uuid(2)],
            vec![uuid(1)],
        ],
        "ORDER BY nullable uuid DESC: NULLs first then non-NULL descending"
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
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE t (a INT, big BIGINT, big2 BIGINT, small BIGINT)",
    )
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
    e.execute_text(
        2,
        &format!("INSERT INTO t (a, big, big2, small) VALUES {values}"),
    )
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
    let mut ne_expected: Vec<Vec<SqlValue>> = (0..300).map(|i| vec![SqlValue::Int4(i)]).collect();
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT, small BIGINT)",
    )
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
    e.execute_text(
        2,
        &format!("INSERT INTO t (a, b, c, small) VALUES {values}"),
    )
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
    assert_eq!(
        added.rows, added_expected,
        "a+b>c => a=BASE+i for i in [301, 600)"
    );
    assert_eq!(added.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(added.fallback_reason, None);

    // SUB then scalar compare: a - b > 200 <=> i > 200 => [201, 600). Projects small = i.
    let subbed = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE a - b > 200")
        .expect("int8 a-b>200 on GPU");
    let subbed_expected: Vec<Vec<SqlValue>> = (201..N).map(|i| vec![SqlValue::Int8(i)]).collect();
    assert_eq!(
        subbed.rows, subbed_expected,
        "a-b>200 => small=i for i in [201, 600)"
    );

    // scalar MUL: small * 2 > 800 <=> i > 400 => [401, 600). Projects small = i.
    let scaled = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE small * 2 > 800")
        .expect("int8 small*2>800 on GPU");
    let scaled_expected: Vec<Vec<SqlValue>> = (401..N).map(|i| vec![SqlValue::Int8(i)]).collect();
    assert_eq!(
        scaled.rows, scaled_expected,
        "small*2>800 => small=i for i in [401, 600)"
    );
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
            Ok(_) => {
                panic!("a*a over a=3037000500 overflows int64 -> must raise bigint out of range")
            }
            Err(err) => assert!(
                err.to_string().contains("bigint out of range"),
                "int8 overflow must be PG's `bigint out of range` (not `integer`), got: {err}"
            ),
        }
    }

    // The largest in-range square does NOT error and returns the rows (a*a > 0 for both rows).
    if let Some(result) = run_int8_square_gt_zero(3_037_000_499) {
        let rows =
            result.expect("a*a at the int64 boundary 3037000499 is in range, must not error");
        assert_eq!(
            rows.rows,
            vec![vec![SqlValue::Int8(2)], vec![SqlValue::Int8(3_037_000_499)]],
            "a*a>0 in range -> both rows"
        );
    }
}

/// Build a single-BIGINT-column table `t(a) = [2, boundary]`, push a GPU snapshot, and run
/// `SELECT a FROM t WHERE a * a > 0` on the general executor. Returns `None` off-GPU.
fn run_int8_square_gt_zero(boundary: i64) -> Option<Result<RelationalSelectResult, ExecuteError>> {
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE t (a INT, small BIGINT, big BIGINT, big2 BIGINT)",
    )
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
    e.execute_text(
        2,
        &format!("INSERT INTO t (a, small, big, big2) VALUES {values}"),
    )
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
    assert_eq!(
        and_rows.rows, and_expected,
        "small>200 AND small<400 => [201, 400)"
    );
    assert_eq!(and_rows.executed_target, DeviceTarget::Gpu(0));

    // int8 OR: small < 100 OR small > 500 => [0,100) U [501,600).
    let or_rows = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE small < 100 OR small > 500")
        .expect("int8 OR on GPU");
    let mut or_expected: Vec<Vec<SqlValue>> = (0..100).map(|i| vec![SqlValue::Int8(i)]).collect();
    or_expected.extend((501..N).map(|i| vec![SqlValue::Int8(i)]));
    assert_eq!(
        or_rows.rows, or_expected,
        "small<100 OR small>500 => [0,100) U [501,600)"
    );

    // 64-bit AND over two int8 columns (values above i32::MAX): big > 100 AND big2 > 100 => all rows.
    // An i32-stride read of big/big2 would NOT yield all rows, so this pins the genuine 64-bit read.
    let big_and = e
        .execute_resident_expr_select_sql("SELECT big FROM t WHERE big > 100 AND big2 > 100")
        .expect("int8 64-bit AND on GPU");
    let big_and_expected: Vec<Vec<SqlValue>> =
        (0..N).map(|i| vec![SqlValue::Int8(BASE + i)]).collect();
    assert_eq!(
        big_and.rows, big_and_expected,
        "big>100 AND big2>100 => all rows (64-bit)"
    );

    // MIXED int4/int8 inside AND now RUNS at I32 (ADR-006 mixed-width groups: the int8 leaf is a
    // width-safe `LoadColumnI64` scalar compare) — it used to hard-error. `big` is all > i32::MAX
    // (and > 5), so the pin is row-exactness: a 4-byte mis-read could not return exactly a<3.
    let mixed = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE big > 5 AND a < 3")
        .expect("mixed int4/int8 AND runs on the GPU (mixed-width group)");
    let mixed_expected: Vec<Vec<SqlValue>> = (0..3).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        mixed.rows, mixed_expected,
        "big>5 AND a<3 => a in {{0,1,2}} (all bigs exceed 5)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_comparisons() {
    // numeric (NUMERIC / i128) comparison + projection end-to-end from SQL (the type matrix, doc 19).
    // price[i] = i.50 (NUMERIC(10,2)), cost[i] = (N-1-i).50. Closed-form oracles; the i128 signedness
    // is proven separately in the execution-crate primitive test.
    let mut e = Engine::new_local_cpu_oracle();
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
    e.execute_text(
        2,
        &format!("INSERT INTO t (price, cost, label) VALUES {values}"),
    )
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
    assert_eq!(
        gt.rows, gt_expected,
        "price > 10.50 => i.50 for i in [11, 600)"
    );
    assert_eq!(gt.executed_target, DeviceTarget::Gpu(0));

    // integer literal coerced to numeric: price < 5 => i+0.50 < 5 => i <= 4 => [0, 5).
    let lt_int = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price < 5")
        .expect("price < 5 on GPU");
    let lt_int_expected: Vec<Vec<SqlValue>> = (0..5).map(|i| vec![num(i)]).collect();
    assert_eq!(
        lt_int.rows, lt_int_expected,
        "price < 5 (int coerced) => [0, 5)"
    );

    // lower-scale literal rescales UP exactly: price > 10.5 (scale 1) == price > 10.50 => [11, N).
    let gt_low = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price > 10.5")
        .expect("price > 10.5 on GPU");
    assert_eq!(
        gt_low.rows, gt_expected,
        "price > 10.5 (scale 1) == price > 10.50"
    );

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
        .execute_resident_expr_select_sql(
            "SELECT price FROM t WHERE price > 1.00 AND price < 100.00",
        )
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
    let mut e = Engine::new_local_cpu_oracle();
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
    e.execute_text(
        2,
        &format!("INSERT INTO t (price, cost, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // col+col add then scalar compare: price+cost = 2i+0.75; > 100 => 200i+75 > 10000 => i >= 50.
    let added = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price + cost > 100")
        .expect("price+cost>100 on GPU");
    let added_expected: Vec<Vec<SqlValue>> =
        (50..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        added.rows, added_expected,
        "price+cost>100 => label in [50, 600)"
    );
    assert_eq!(added.executed_target, DeviceTarget::Gpu(0));

    // scalar sub then compare: price-5 = (i-5)+0.50; > 100 => 100i-450 > 10000 => i >= 105.
    let subbed = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price - 5 > 100")
        .expect("price-5>100 on GPU");
    let subbed_expected: Vec<Vec<SqlValue>> =
        (105..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        subbed.rows, subbed_expected,
        "price-5>100 => label in [105, 600)"
    );

    // buffer-vs-buffer (arith on the left, column on the right): price+cost > price <=> cost > 0 => all.
    let cmp_buffers = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price + cost > price")
        .expect("price+cost>price on GPU");
    let all_expected: Vec<Vec<SqlValue>> = (0..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        cmp_buffers.rows, all_expected,
        "price+cost>price <=> cost>0 => all rows"
    );

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
    let mut e = Engine::new_local_cpu_oracle();
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
    e.execute_text(
        2,
        &format!("INSERT INTO t (price, cost, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // price * 2 > 100 : (200i+100) > 10000 => 200i > 9900 => i >= 50 => [50, N).
    let mul = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * 2 > 100")
        .expect("price*2>100 on GPU");
    let mul_expected: Vec<Vec<SqlValue>> =
        (50..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(mul.rows, mul_expected, "price*2>100 => label in [50, 600)");
    assert_eq!(mul.executed_target, DeviceTarget::Gpu(0));

    // commutative: 2 * price > 100 is the same set.
    let mul_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE 2 * price > 100")
        .expect("2*price>100 on GPU");
    assert_eq!(
        mul_left.rows, mul_expected,
        "2*price>100 == price*2>100 (commutative)"
    );

    // fractional-literal multiply: price*1.5 (result scale 2+1=3): (1500i+750) > 100000 => i >= 67.
    let frac = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * 1.5 > 100")
        .expect("price*1.5>100 on GPU");
    let frac_expected: Vec<Vec<SqlValue>> =
        (67..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        frac.rows, frac_expected,
        "price*1.5>100 => label in [67, 600)"
    );

    // column*column multiply: price*cost (result scale 2+2=4): (100i+50)(100i+25) > 1000000 => i >= 10.
    let cols = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * cost > 100")
        .expect("price*cost>100 on GPU");
    let cols_expected: Vec<Vec<SqlValue>> =
        (10..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        cols.rows, cols_expected,
        "price*cost>100 => label in [10, 600)"
    );

    // cross-scale arith-vs-arith: price*cost (scale 4) > price (scale 2): rescale price up, then
    // (100i+25) > 100 <=> i >= 1 => [1, N).
    let cross = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * cost > price")
        .expect("price*cost>price (cross-scale) on GPU");
    let cross_expected: Vec<Vec<SqlValue>> =
        (1..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        cross.rows, cross_expected,
        "price*cost > price (cross-scale) => [1, 600)"
    );

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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE t (p2 NUMERIC(10,2), p4 NUMERIC(10,4), label INT)",
    )
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
    assert_eq!(
        lit.rows, lit_expected,
        "p2>1.555 (finer literal) => [2, 600)"
    );

    // literal on the left: 1.555 < p2 is the same set.
    let lit_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE 1.555 < p2")
        .expect("1.555<p2 on GPU");
    assert_eq!(lit_left.rows, lit_expected, "1.555<p2 == p2>1.555");

    // same-scale still uses the resident peephole (regression): p2 > 10.50 => [11, N).
    let same = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 > 10.50")
        .expect("p2>10.50 on GPU");
    let same_expected: Vec<Vec<SqlValue>> =
        (11..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        same.rows, same_expected,
        "p2>10.50 (same scale) => [11, 600)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_cross_scale_numeric_add_sub() {
    // CROSS-SCALE numeric ADD/SUB (the type matrix, doc 19): operands of different scales are rescaled
    // UP to the common (max) scale before the buffer add/sub. p2 = i.50 (NUMERIC(10,2)), p4 = (2i).2500
    // (NUMERIC(10,4)), label = i.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE t (p2 NUMERIC(10,2), p4 NUMERIC(10,4), label INT)",
    )
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
    let add_expected: Vec<Vec<SqlValue>> =
        (34..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        add.rows, add_expected,
        "p2+p4>100 (cross-scale add) => [34, 600)"
    );
    assert_eq!(add.executed_target, DeviceTarget::Gpu(0));

    // cross-scale SUB (col-col): p4 - p2 = i - 0.25 (scale 4); > 100 => i >= 101.
    let sub = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p4 - p2 > 100")
        .expect("p4-p2>100 on GPU");
    let sub_expected: Vec<Vec<SqlValue>> =
        (101..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        sub.rows, sub_expected,
        "p4-p2>100 (cross-scale sub) => [101, 600)"
    );

    // cross-scale add with a FINER literal: p2 + 0.0001 = i.5001 (scale 4); > 50.5 => i >= 50.
    let lit = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 + 0.0001 > 50.5")
        .expect("p2+0.0001>50.5 on GPU");
    let lit_expected: Vec<Vec<SqlValue>> =
        (50..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        lit.rows, lit_expected,
        "p2 + 0.0001 (finer literal) > 50.5 => [50, 600)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_and_or() {
    // numeric AND/OR (the type matrix, doc 19): each comparison -> a mask via the i128 VM, MaskBinary
    // combines, terminal compact. price = i.50 (NUMERIC(10,2)), cost = i.2500 (NUMERIC(10,4)).
    let mut e = Engine::new_local_cpu_oracle();
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
    e.execute_text(
        2,
        &format!("INSERT INTO t (price, cost, label) VALUES {values}"),
    )
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
    let and_expected: Vec<Vec<SqlValue>> = (11i64..100)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();
    assert_eq!(
        and.rows, and_expected,
        "price>10.50 AND price<100.50 => [11, 100)"
    );
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
    assert_eq!(
        or.rows, or_expected,
        "price<5.50 OR price>595.50 => [0,5) U [596,600)"
    );

    // CROSS-SCALE AND (price scale 2, cost scale 4 -- each comparison rescales independently):
    // price > 10.50 AND cost > 50.2500 <=> i>10 AND i>50 => [51, N).
    let cross = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE price > 10.50 AND cost > 50.2500",
        )
        .expect("cross-scale AND on GPU");
    let cross_expected: Vec<Vec<SqlValue>> =
        (51..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
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
    // Text equality on the general GPU executor (the type matrix, doc 19): byte-wise = / <>. The
    // residency builder pads the text offsets section to its required 8-byte alignment even after an
    // odd-sized int4 section.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (name TEXT, label INT)")
        .unwrap();
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

    // Text INEQUALITY now runs on-device (ADR-006: the lexicographic byte-compare kernel). `name < 'bob'`
    // = the three 'alice' rows (a < b byte-wise) -> labels 0,2,5, matching the host `str::cmp` order.
    let lt = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name < 'bob'")
        .expect("name < 'bob' on GPU");
    assert_eq!(
        lt.rows,
        vec![
            vec![SqlValue::Int4(0)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(5)]
        ],
        "name < 'bob' => the 'alice' rows 0,2,5 (byte-wise lexicographic)"
    );

    // REJECTIONS -- hard errors, never wrong rows:
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
    // The offsets section remains 8-byte aligned after the odd-sized int4 section. Includes the `\_`
    // escape vs a bare `_`.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (name TEXT, label INT)")
        .unwrap();
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
    assert_eq!(
        prefix.rows,
        labels(&[0, 1, 3]),
        "LIKE 'al%' => alice/alicia/alfred"
    );
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
    assert_eq!(
        escaped.rows,
        labels(&[5]),
        "LIKE 'a\\_b' => only the literal a_b"
    );

    // exact (no wildcards) behaves like equality
    let exact = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE 'bob'")
        .expect("LIKE 'bob' on GPU");
    assert_eq!(exact.rows, labels(&[2]), "LIKE 'bob' => bob");

    // '%' matches every row
    let all = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE '%'")
        .expect("LIKE '%' on GPU");
    assert_eq!(
        all.rows,
        labels(&[0, 1, 2, 3, 4, 5, 6]),
        "LIKE '%' => all rows"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_date_comparisons() {
    // Date comparison on the general GPU executor (the type matrix, doc 19): a `date` is i32 days
    // since 2000-01-01, reusing the int4 residency section + the I32 VM. hire_date[i] = 2024-01-(i+1),
    // label = i. The string literal is coerced to a day count at lowering (like PG).
    let mut e = Engine::new_local_cpu_oracle();
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
    e.execute_text(
        2,
        &format!("INSERT INTO t (hire_date, label) VALUES {values}"),
    )
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
    assert_eq!(
        eq.rows,
        vec![vec![SqlValue::Int4(14)]],
        "= '2024-01-15' => row 14"
    );
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
    assert_eq!(
        lit_left.rows,
        labels(15..N),
        "'2024-01-15' < hire_date => [15, 30)"
    );

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
    let mut e = Engine::new_local_cpu_oracle();
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
    assert_eq!(
        eq.rows,
        vec![vec![SqlValue::Int4(10)]],
        "= 10:00:00 => row 10"
    );
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
    assert_eq!(
        lit_left.rows,
        labels(11..N),
        "10:00:00 < event_at => [11, 24)"
    );

    // col-vs-col: event_at > created_at (noon) -> hours > 12 -> [13, 24)
    let col_col = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE event_at > created_at")
        .expect("event_at > created_at on GPU");
    assert_eq!(
        col_col.rows,
        labels(13..N),
        "event_at > created_at (noon) => [13, 24)"
    );

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
    let mut e = Engine::new_local_cpu_oracle();
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
    e.execute_text(
        2,
        &format!("INSERT INTO t (id, peer, label) VALUES {values}"),
    )
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
    let mut e = Engine::new_local_cpu_oracle();
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
    e.execute_text(
        2,
        &format!("INSERT INTO t (sz, peer, label) VALUES {values}"),
    )
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (flag BOOL, label INT)")
        .unwrap();
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
        (
            "SELECT label FROM t WHERE flag = true",
            &even,
            "= true => even",
        ),
        (
            "SELECT label FROM t WHERE flag = false",
            &odd,
            "= false => odd",
        ),
        (
            "SELECT label FROM t WHERE true = flag",
            &even,
            "true = flag => even",
        ),
        (
            "SELECT label FROM t WHERE flag <> true",
            &odd,
            "<> true => odd",
        ),
        (
            "SELECT label FROM t WHERE flag <> false",
            &even,
            "<> false => even",
        ),
        // NOT flag === flag = false (the mapper rewrites it).
        (
            "SELECT label FROM t WHERE NOT flag",
            &odd,
            "NOT flag => odd",
        ),
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, flag BOOL)")
        .unwrap();
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
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int8(39)]],
        "COUNT(*) WHERE a > 10 => 39"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));

    // COUNT(*) WHERE flag -- a popcount over the bool bitmap. i%3==0 in [0,50) = 17 rows.
    let flag_count = (0..N).filter(|i| i % 3 == 0).count() as i64;
    let r2 = e
        .execute_resident_expr_select_sql("SELECT COUNT(*) FROM t WHERE flag")
        .expect("count flag on GPU");
    assert_eq!(
        r2.rows,
        vec![vec![SqlValue::Int8(flag_count)]],
        "COUNT(*) WHERE flag"
    );

    // COUNT(*) of an empty result -> 0 (not an error / not NULL).
    let r3 = e
        .execute_resident_expr_select_sql("SELECT COUNT(*) FROM t WHERE a > 1000")
        .expect("count empty on GPU");
    assert_eq!(
        r3.rows,
        vec![vec![SqlValue::Int8(0)]],
        "COUNT(*) empty => 0"
    );

    // count(*) is case-insensitive.
    let r4 = e
        .execute_resident_expr_select_sql("SELECT count(*) FROM t WHERE a >= 0")
        .expect("lowercase count on GPU");
    assert_eq!(
        r4.rows,
        vec![vec![SqlValue::Int8(N)]],
        "count(*) WHERE a >= 0 => all"
    );

    // SUM(int4) over a filtered set -- a GPU reduction over the gathered column; PG returns bigint.
    let sum_of = |keep: &dyn Fn(i64) -> bool| -> i64 { (0..N).filter(|&i| keep(i)).sum() };
    let s1 = e
        .execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE a > 10")
        .expect("sum on GPU");
    assert_eq!(
        s1.rows,
        vec![vec![SqlValue::Int8(sum_of(&|a| a > 10))]],
        "SUM(a) WHERE a > 10"
    );
    assert_eq!(s1.executed_target, DeviceTarget::Gpu(0));
    let s2 = e
        .execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE flag")
        .expect("sum where flag on GPU");
    assert_eq!(
        s2.rows,
        vec![vec![SqlValue::Int8(sum_of(&|a| a % 3 == 0))]],
        "SUM(a) WHERE flag"
    );
    let s3 = e
        .execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE a >= 0")
        .expect("sum all on GPU");
    assert_eq!(
        s3.rows,
        vec![vec![SqlValue::Int8(sum_of(&|_| true))]],
        "SUM(a) WHERE a >= 0 => total"
    );
    // SUM over an EMPTY set is SQL NULL (PG) -- one row, NULL (was the M3 hard error).
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE a > 1000")
            .expect("SUM empty on GPU")
            .rows,
        vec![vec![SqlValue::Null]],
        "SUM over empty => SQL NULL"
    );

    // MIN / MAX(int4) over a filtered set -- GPU reductions; PG preserves the type (int4 -> int4).
    let min_of = |keep: &dyn Fn(i64) -> bool| (0..N).filter(|&i| keep(i)).min().unwrap() as i32;
    let max_of = |keep: &dyn Fn(i64) -> bool| (0..N).filter(|&i| keep(i)).max().unwrap() as i32;
    let mn = e
        .execute_resident_expr_select_sql("SELECT MIN(a) FROM t WHERE a > 10")
        .expect("min on GPU");
    assert_eq!(
        mn.rows,
        vec![vec![SqlValue::Int4(min_of(&|a| a > 10))]],
        "MIN(a) WHERE a > 10 => 11"
    );
    let mx = e
        .execute_resident_expr_select_sql("SELECT MAX(a) FROM t WHERE flag")
        .expect("max where flag on GPU");
    assert_eq!(
        mx.rows,
        vec![vec![SqlValue::Int4(max_of(&|a| a % 3 == 0))]],
        "MAX(a) WHERE flag"
    );
    let mx2 = e
        .execute_resident_expr_select_sql("SELECT MAX(a) FROM t WHERE a >= 0")
        .expect("max all on GPU");
    assert_eq!(
        mx2.rows,
        vec![vec![SqlValue::Int4((N - 1) as i32)]],
        "MAX(a) WHERE a >= 0 => N-1"
    );
    let mn2 = e
        .execute_resident_expr_select_sql("SELECT MIN(a) FROM t WHERE a >= 0")
        .expect("min all on GPU");
    assert_eq!(
        mn2.rows,
        vec![vec![SqlValue::Int4(0)]],
        "MIN(a) WHERE a >= 0 => 0"
    );
    // MIN/MAX over an EMPTY set is SQL NULL (PG).
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT MIN(a) FROM t WHERE a > 1000")
            .expect("MIN empty on GPU")
            .rows,
        vec![vec![SqlValue::Null]],
        "MIN over empty => SQL NULL"
    );
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT MAX(a) FROM t WHERE a > 1000")
            .expect("MAX empty on GPU")
            .rows,
        vec![vec![SqlValue::Null]],
        "MAX over empty => SQL NULL"
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
    assert_eq!(
        a1.rows,
        avg_expected(&|a| a > 10),
        "AVG(a) WHERE a > 10 => 30"
    );
    assert_eq!(a1.executed_target, DeviceTarget::Gpu(0));
    // 30 exactly, at scale 16.
    assert_eq!(
        a1.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(
            30 * 10_i128.pow(16),
            16
        ))]],
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
    // AVG over an EMPTY set is SQL NULL (PG).
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT AVG(a) FROM t WHERE a > 1000")
            .expect("AVG empty on GPU")
            .rows,
        vec![vec![SqlValue::Null]],
        "AVG over empty => SQL NULL"
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
    assert_eq!(
        avg_str(3, 1),
        "3.0000000000000000",
        "exact integer, scale 16"
    );
    assert_eq!(avg_str(7, 2), "3.5000000000000000", "3.5, scale 16");
    // sub-1 quotient -> scale 20; repeating, ROUNDS the last digit up.
    assert_eq!(
        avg_str(2, 3),
        "0.66666666666666666667",
        "2/3 rounds, scale 20"
    );
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
    assert_eq!(
        avg_str(-2, 3),
        "-0.66666666666666666667",
        "negative rounds away"
    );
    // PG select_div_scale decrements the quotient weight when the dividend's leading base-10000 digit
    // <= the divisor's -- so an exact 1.0 from sum==count renders at scale 20, NOT 16. The naive
    // "quotient decimal weight" formula shipped scale 16 here; these lock the fix (verified vs PG 18).
    assert_eq!(
        avg_str(3, 3),
        "1.00000000000000000000",
        "sum==count -> scale 20 (firstdigit decr)"
    );
    assert_eq!(
        avg_str(5, 5),
        "1.00000000000000000000",
        "leading-digit-equal -> scale 20"
    );
    assert_eq!(
        avg_str(9, 3),
        "3.0000000000000000",
        "fd1(9) > fd2(3) -> no decr, scale 16"
    );
    // Zero sum (e.g. AVG over cancelling rows): PG renders 0 at scale max(S, 20), not 16.
    assert_eq!(
        avg_str(0, 2),
        "0.00000000000000000000",
        "zero sum -> scale 20"
    );
    assert_eq!(
        avg_str(0, 5),
        "0.00000000000000000000",
        "zero sum, count 5 -> scale 20"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int8_aggregates() {
    // int8 aggregates: MIN/MAX -> int8, SUM/AVG -> numeric (a sum of int8 can exceed i64, so SUM
    // reduces to i128 via the two-atomic carry kernel). Values span > i32::MAX, negatives, and a
    // subset (rows 0,1) whose SUM EXCEEDS i64::MAX. label = row index (the int4 filter column).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (b BIGINT, label INT)")
        .unwrap();
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
    // Closed-form oracles (S9): vals = [9e18, 8e18, 0, 5e9, -7e9, -9e18] (label = row index, the int4
    // filter col). keep_all = all rows; keep_lt2 = rows 0,1 (label < 2) = {9e18, 8e18}. Explicit
    // constants, not a host `vals.iter()...min()/max()/sum()` re-implementation of the aggregate.
    let min_all: i64 = -9_000_000_000_000_000_000; // MIN over all rows
    let max_lt2: i64 = 9_000_000_000_000_000_000; // MAX of {9e18, 8e18}
    let sum_lt2: i128 = 17_000_000_000_000_000_000; // 9e18 + 8e18 (> i64::MAX -> i128 carry)
    let sum_all: i128 = 7_999_999_998_000_000_000; // 9e18 + 8e18 + 0 + 5e9 - 7e9 - 9e18

    // MIN/MAX(int8) -> int8.
    let mn = e
        .execute_resident_expr_select_sql("SELECT MIN(b) FROM t WHERE label >= 0")
        .expect("min");
    assert_eq!(mn.rows, vec![vec![SqlValue::Int8(min_all)]], "MIN(b) all");
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));
    let mx = e
        .execute_resident_expr_select_sql("SELECT MAX(b) FROM t WHERE label < 2")
        .expect("max subset");
    assert_eq!(
        mx.rows,
        vec![vec![SqlValue::Int8(max_lt2)]],
        "MAX(b) subset => 9e18"
    );

    // SUM(int8) -> numeric (scale 0). The subset {9e18, 8e18} sums to 17e18 -- EXCEEDS i64::MAX, so
    // the i128 two-atomic carry must be correct.
    let s_sub = e
        .execute_resident_expr_select_sql("SELECT SUM(b) FROM t WHERE label < 2")
        .expect("sum subset");
    assert_eq!(
        s_sub.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(sum_lt2, 0))]],
        "SUM subset => 17e18 (> i64::MAX, i128 carry)"
    );
    assert!(sum_lt2 > i128::from(i64::MAX), "test really exceeds i64");
    // The result COLUMN descriptor must match the numeric value (the int8-agg audit caught SUM(int8)
    // declaring Int4/oid 23 -- a wire-decode mismatch). PG SUM(int8) -> numeric (oid 1700).
    assert!(
        matches!(s_sub.columns[0].ty, SqlType::Numeric { .. }),
        "SUM(int8) result column must be numeric, got {:?}",
        s_sub.columns[0].ty
    );
    assert_eq!(
        s_sub.columns[0].type_oid, 1700,
        "SUM(int8) wire oid = numeric 1700"
    );
    let s_all = e
        .execute_resident_expr_select_sql("SELECT SUM(b) FROM t WHERE label >= 0")
        .expect("sum all");
    assert_eq!(
        s_all.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(sum_all, 0))]],
        "SUM all (incl. negatives)"
    );

    // AVG(int8) -> numeric. 17e18 / 2 = 8.5e18 (scale 0 at this magnitude).
    let av = e
        .execute_resident_expr_select_sql("SELECT AVG(b) FROM t WHERE label < 2")
        .expect("avg subset");
    assert_eq!(
        av.rows,
        vec![vec![average_sql_value(sum_lt2, 2)]],
        "AVG subset"
    );
    match &av.rows[0][0] {
        SqlValue::Numeric(d) => {
            assert_eq!(d.to_decimal_string(), "8500000000000000000", "AVG = 8.5e18")
        }
        other => panic!("AVG must be numeric, got {other:?}"),
    }

    // Empty SUM/MIN/AVG -> SQL NULL (PG).
    for sql in [
        "SELECT MIN(b) FROM t WHERE label > 1000",
        "SELECT SUM(b) FROM t WHERE label > 1000",
        "SELECT AVG(b) FROM t WHERE label > 1000",
    ] {
        assert_eq!(
            e.execute_resident_expr_select_sql(sql).expect(sql).rows,
            vec![vec![SqlValue::Null]],
            "{sql} => SQL NULL"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_minmax() {
    // MIN/MAX(numeric) over a filtered set -> numeric (PG preserves the type). Reduces the i128
    // mantissas via the partials + host-combine reduction. label = row index (int4 filter col).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (p NUMERIC(10,2), label INT)")
        .unwrap();
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
    let mn = e
        .execute_resident_expr_select_sql("SELECT MIN(p) FROM t WHERE label >= 0")
        .expect("min num");
    assert_eq!(mn.rows, vec![vec![num(-9999)]], "MIN(p) all => -99.99");
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));
    let mx = e
        .execute_resident_expr_select_sql("SELECT MAX(p) FROM t WHERE label >= 0")
        .expect("max num");
    assert_eq!(mx.rows, vec![vec![num(10000)]], "MAX(p) all => 100.00");

    // Subset (label < 3 -> 12.50, -3.75, 100.00): MIN = -3.75, MAX = 100.00.
    let mn2 = e
        .execute_resident_expr_select_sql("SELECT MIN(p) FROM t WHERE label < 3")
        .expect("min subset");
    assert_eq!(mn2.rows, vec![vec![num(-375)]], "MIN subset => -3.75");
    let mx2 = e
        .execute_resident_expr_select_sql("SELECT MAX(p) FROM t WHERE label < 3")
        .expect("max subset");
    assert_eq!(mx2.rows, vec![vec![num(10000)]], "MAX subset => 100.00");

    // Empty -> SQL NULL (PG).
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT MAX(p) FROM t WHERE label > 1000")
            .expect("MAX(numeric) empty on GPU")
            .rows,
        vec![vec![SqlValue::Null]],
        "MAX(numeric) over empty => SQL NULL"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_sum_avg() {
    // SUM(numeric) -> numeric at the column scale (i128 mantissa sum); AVG(numeric) -> numeric at PG's
    // division scale. Expected values derived from PG's numeric semantics (SUM keeps scale 2; AVG of a
    // weight-0 quotient over a scale-2 dividend has rscale max(2, 16) = 16).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (p NUMERIC(10,2), label INT)")
        .unwrap();
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
    let s_all = e
        .execute_resident_expr_select_sql("SELECT SUM(p) FROM t WHERE label >= 0")
        .expect("sum all");
    assert_eq!(
        s_all.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(5119, 2))]],
        "SUM all = 51.19"
    );
    assert_eq!(s_all.executed_target, DeviceTarget::Gpu(0));
    assert!(
        matches!(s_all.columns[0].ty, SqlType::Numeric { .. }),
        "SUM(numeric) col numeric"
    );
    assert_eq!(s_all.columns[0].type_oid, 1700, "SUM(numeric) oid 1700");
    // SUM subset (12.50, -3.75) = 8.75.
    let s_sub = e
        .execute_resident_expr_select_sql("SELECT SUM(p) FROM t WHERE label < 2")
        .expect("sum subset");
    assert_eq!(
        s_sub.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(875, 2))]],
        "SUM subset = 8.75"
    );

    // AVG all = 51.19 / 6 = 8.5316666... -> scale 16, round half-away.
    let a_all = e
        .execute_resident_expr_select_sql("SELECT AVG(p) FROM t WHERE label >= 0")
        .expect("avg all");
    match &a_all.rows[0][0] {
        SqlValue::Numeric(d) => assert_eq!(d.to_decimal_string(), "8.5316666666666667", "AVG all"),
        other => panic!("AVG numeric, got {other:?}"),
    }
    assert!(
        matches!(a_all.columns[0].ty, SqlType::Numeric { .. }),
        "AVG(numeric) col numeric"
    );
    // AVG subset = 8.75 / 2 = 4.375 -> scale 16.
    let a_sub = e
        .execute_resident_expr_select_sql("SELECT AVG(p) FROM t WHERE label < 2")
        .expect("avg subset");
    match &a_sub.rows[0][0] {
        SqlValue::Numeric(d) => {
            assert_eq!(d.to_decimal_string(), "4.3750000000000000", "AVG subset")
        }
        other => panic!("AVG numeric, got {other:?}"),
    }

    // Empty SUM/AVG -> SQL NULL (PG).
    for sql in [
        "SELECT SUM(p) FROM t WHERE label > 1000",
        "SELECT AVG(p) FROM t WHERE label > 1000",
    ] {
        assert_eq!(
            e.execute_resident_expr_select_sql(sql).expect(sql).rows,
            vec![vec![SqlValue::Null]],
            "{sql} => SQL NULL"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_numeric_sum_overflow_errors() {
    // SUM(numeric) is CHECKED: a mantissa sum exceeding i128 is PG `numeric field overflow`, NEVER a
    // silent wrap. Each mantissa is 9e18 * 10^19 = 9e37 (column NUMERIC(38,19), integer part 9e18 fits
    // the legacy parser's i64 literal range); two sum to 1.8e38 > i128::MAX (~1.7e38).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE big (v NUMERIC(38,19), label INT)")
        .unwrap();
    let big = "9000000000000000000"; // 9e18, fits i64
    e.execute_text(
        2,
        &format!("INSERT INTO big (v, label) VALUES ({big}, 0), ({big}, 1)"),
    )
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a, b) VALUES (5,10),(3,20),(8,30),(1,40),(9,50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let a = [5_i32, 3, 8, 1, 9];

    // COUNT(*) over the whole table.
    let c = e
        .execute_resident_expr_select_sql("SELECT COUNT(*) FROM t")
        .expect("count *");
    assert_eq!(
        c.rows,
        vec![vec![SqlValue::Int8(a.len() as i64)]],
        "COUNT(*) no WHERE = 5"
    );
    assert_eq!(c.executed_target, DeviceTarget::Gpu(0));

    // SUM/MIN/MAX over the whole column.
    // Closed-form oracles (a = [5,3,8,1,9]): SUM=26, MIN=1, MAX=9 -- explicit constants, not a host
    // .iter() re-implementation of the aggregate (GPU-native-oracle charter, S9).
    let s = e
        .execute_resident_expr_select_sql("SELECT SUM(a) FROM t")
        .expect("sum");
    assert_eq!(s.rows, vec![vec![SqlValue::Int8(26)]], "SUM(a)=26");
    let mn = e
        .execute_resident_expr_select_sql("SELECT MIN(a) FROM t")
        .expect("min");
    assert_eq!(mn.rows, vec![vec![SqlValue::Int4(1)]], "MIN(a)=1");
    let mx = e
        .execute_resident_expr_select_sql("SELECT MAX(a) FROM t")
        .expect("max");
    assert_eq!(mx.rows, vec![vec![SqlValue::Int4(9)]], "MAX(a)=9");

    // AVG(a) = 26/5 = 5.2 -> numeric scale 16 (fd1=26 > fd2=5, no leading-digit decrement).
    let av = e
        .execute_resident_expr_select_sql("SELECT AVG(a) FROM t")
        .expect("avg");
    match &av.rows[0][0] {
        SqlValue::Numeric(d) => {
            assert_eq!(d.to_decimal_string(), "5.2000000000000000", "AVG no WHERE")
        }
        other => panic!("AVG numeric, got {other:?}"),
    }

    // Full-table projection: every row, in residency (insertion) order.
    let p = e
        .execute_resident_expr_select_sql("SELECT a FROM t")
        .expect("project a");
    let got: Vec<i32> = p
        .rows
        .iter()
        .map(|r| match r[0] {
            SqlValue::Int4(v) => v,
            ref other => panic!("expected int4, got {other:?}"),
        })
        .collect();
    assert_eq!(
        got,
        a.to_vec(),
        "SELECT a FROM t projects all rows in order"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_skips_null_values_and_groups_null_keys() {
    // M3 (doc 21): GROUP BY full 3VL on the GPU — a NULL aggregate VALUE is SKIPPED from SUM/AVG/MIN/MAX
    // (the single-level kernel's value-skip) while COUNT(*) still counts the row (a dedicated total-count
    // pass), an all-NULL group's aggregate is SQL NULL, AND a NULL group KEY forms its OWN group (the
    // kernel's reserved NULL-key slot) rendered SqlValue::Null. A NULL-free nullable column is unchanged.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT, h INT)")
        .unwrap();
    // g has a NULL (row 2); v has a NULL (rows 4 and 6); h has none.
    // h=7 -> v{10, 20}; h=8 -> v{30, NULL}; h=9 -> v{NULL} (an all-NULL group).
    e.execute_text(
        2,
        "INSERT INTO t (g,v,h) VALUES (1,10,7),(NULL,20,7),(2,30,8),(1,NULL,8),(2,NULL,9)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // GROUP BY a NULL-bearing KEY: the NULL keys form their OWN group (rendered SqlValue::Null), which
    // sorts first. g: 1,NULL,2,1,2 -> g=1 count 2, g=2 count 2, g=NULL count 1.
    let gc = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g")
        .expect("GROUP BY a nullable key forms a NULL group");
    assert_eq!(
        gc.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(1)],
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(2)],
        ],
        "NULL keys form their own group (COUNT(*) counts them); it sorts first"
    );

    // SUM over a nullable VALUE: NULLs skipped. h=7 -> 30, h=8 -> 30 (NULL skipped), h=9 -> NULL (all-NULL).
    let s = e
        .execute_resident_expr_select_sql("SELECT h, SUM(v) FROM t GROUP BY h")
        .expect("SUM over a nullable value runs");
    assert_eq!(
        s.rows,
        vec![
            vec![SqlValue::Int4(7), SqlValue::Int8(30)],
            vec![SqlValue::Int4(8), SqlValue::Int8(30)],
            vec![SqlValue::Int4(9), SqlValue::Null],
        ],
        "SUM skips NULL values; an all-NULL group is NULL"
    );
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));

    // COUNT(*) + SUM together: COUNT(*) counts EVERY row (incl. NULL-v), SUM skips NULLs. h=8 -> (2, 30);
    // h=9 -> (1, NULL). Exercises the dedicated total-count pass alongside the value-skip pass.
    let cs = e
        .execute_resident_expr_select_sql("SELECT h, COUNT(*), SUM(v) FROM t GROUP BY h")
        .expect("COUNT(*) + SUM over a nullable value");
    assert_eq!(
        cs.rows,
        vec![
            vec![SqlValue::Int4(7), SqlValue::Int8(2), SqlValue::Int8(30)],
            vec![SqlValue::Int4(8), SqlValue::Int8(2), SqlValue::Int8(30)],
            vec![SqlValue::Int4(9), SqlValue::Int8(1), SqlValue::Null],
        ],
        "COUNT(*) counts NULL-valued rows; SUM skips them"
    );

    // MIN / MAX / AVG over the nullable value: NULLs skipped; all-NULL group -> NULL.
    let mn = e
        .execute_resident_expr_select_sql("SELECT h, MIN(v) FROM t GROUP BY h")
        .expect("MIN over a nullable value");
    assert_eq!(
        mn.rows,
        vec![
            vec![SqlValue::Int4(7), SqlValue::Int4(10)],
            vec![SqlValue::Int4(8), SqlValue::Int4(30)],
            vec![SqlValue::Int4(9), SqlValue::Null],
        ],
        "MIN skips NULLs (not the 0 placeholder); all-NULL group is NULL"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_null_key_group_with_null_values() {
    // M3 (doc 21): the NULL-KEY group + the value-skip + the total-count pass interact correctly. Every
    // NULL-key row groups together (distinct from real key 0); COUNT(*) counts ALL of them (incl. a NULL-
    // value one); SUM skips the NULL value AMONG the null-key rows. k: 1,NULL,2,NULL,1,NULL.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tk (k INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO tk (k,v) VALUES (1,10),(NULL,20),(2,30),(NULL,40),(1,50),(NULL,NULL)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tk").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // k=1 -> v{10,50}: count 2, sum 60. k=2 -> v{30}: count 1, sum 30.
    // k=NULL -> rows (NULL,20),(NULL,40),(NULL,NULL): count 3 (ALL), sum 60 (the NULL value skipped).
    let r = e
        .execute_resident_expr_select_sql("SELECT k, COUNT(*), SUM(v) FROM tk GROUP BY k")
        .expect("GROUP BY a nullable key with nullable values");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(3), SqlValue::Int8(60)],
            vec![SqlValue::Int4(1), SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![SqlValue::Int4(2), SqlValue::Int8(1), SqlValue::Int8(30)],
        ],
        "NULL keys group together; COUNT(*) counts all (incl. the NULL-value row); SUM skips the NULL"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_null_key_group_with_all_null_values() {
    // M3 (doc 21) regression: a NULL-key group whose aggregate values are ALL NULL. The value pass's
    // reserved null slot then has count 0 — but the group MUST still appear (COUNT(*) counts the rows;
    // SUM is NULL). The reserved slot is emitted on its CLAIMED MARKER (slot_keys != EMPTY), not count,
    // so it appears CONSISTENTLY in every pass and the by-index merge stays aligned (else: panic / the
    // null row silently vanishes).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tp (k INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO tp (k,v) VALUES (1,10),(NULL,NULL),(2,30),(NULL,NULL),(1,50),(NULL,NULL)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tp").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // k=NULL -> 3 rows, all v NULL: COUNT(*) 3, SUM NULL. The multi-pass query is the panic case.
    let r = e
        .execute_resident_expr_select_sql("SELECT k, COUNT(*), SUM(v) FROM tp GROUP BY k")
        .expect("COUNT(*)+SUM with an all-NULL-value NULL-key group");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(3), SqlValue::Null],
            vec![SqlValue::Int4(1), SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![SqlValue::Int4(2), SqlValue::Int8(1), SqlValue::Int8(30)],
        ],
        "the all-NULL-value NULL-key group still appears (COUNT(*)=3, SUM=NULL), aligned across passes"
    );
    // SUM only (no COUNT*): the null group must NOT silently vanish.
    let s = e
        .execute_resident_expr_select_sql("SELECT k, SUM(v) FROM tp GROUP BY k")
        .expect("SUM with an all-NULL-value NULL-key group");
    assert_eq!(
        s.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Null],
            vec![SqlValue::Int4(1), SqlValue::Int8(60)],
            vec![SqlValue::Int4(2), SqlValue::Int8(30)],
        ],
        "the all-NULL-value NULL-key group still appears with SUM NULL (not dropped)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_skips_null_int8_values() {
    // M3 (doc 21): the value-skip is type-agnostic (it gates the accumulate before the per-type sum), so
    // a nullable BIGINT value also skips NULLs on the GPU (the i64 value / i128-carry sum path). MIN(v)
    // returns int8; an all-NULL group is NULL.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t8 (g INT, v BIGINT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t8 (g,v) VALUES (1,100),(1,NULL),(2,9999999999),(3,NULL)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t8").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // g=1 -> {100, NULL} -> MIN 100; g=2 -> {9999999999} -> MIN 9999999999; g=3 -> {NULL} -> NULL.
    let mn = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t8 GROUP BY g")
        .expect("MIN over a nullable bigint");
    assert_eq!(
        mn.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(100)],
            vec![SqlValue::Int4(2), SqlValue::Int8(9999999999)],
            vec![SqlValue::Int4(3), SqlValue::Null],
        ],
        "MIN(bigint) skips NULLs; all-NULL group is NULL"
    );
    // COUNT(*) counts every row (incl. the NULL-v rows).
    let c = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t8 GROUP BY g")
        .expect("COUNT(*) over a nullable bigint table");
    assert_eq!(
        c.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(1)],
            vec![SqlValue::Int4(3), SqlValue::Int8(1)],
        ],
        "COUNT(*) counts NULL-valued rows"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_key_with_count_distinct_clean_errors() {
    // M3 (doc 21): COUNT(DISTINCT v) over a NULLABLE group key is a clean-error follow-up. The
    // COUNT(DISTINCT) sub-passes don't route the NULL key to the reserved slot, so they'd merge NULL-key
    // rows into the placeholder group -> fewer groups than the reference pass -> by-index merge panic.
    // Reject cleanly rather than panic / mis-answer. (Pre-existing for int keys; this guard fixes that
    // too.) A NON-nullable key with COUNT(DISTINCT) is unaffected.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tcd (g INT, v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO tcd (g,v) VALUES (1,10),(NULL,20),(0,30),(NULL,20),(1,10),(0,40)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tcd").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let err = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM tcd GROUP BY g")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("nullable key with COUNT(DISTINCT)") || err.contains("not yet supported"),
        "COUNT(DISTINCT) over a nullable key must clean-error (not panic), got: {err}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_text_key_forms_null_group() {
    // M3 (doc 21): GROUP BY a nullable TEXT key — a NULL key forms its OWN group (rendered SqlValue::Null,
    // sorts first), distinct from real keys, via the kernel's hoisted NULL-key check routing to the
    // reserved slot BEFORE the text claim. A NULL text key is NOT folded into the empty-string group.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tgt (k TEXT, v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO tgt (k,v) VALUES ('a',10),(NULL,20),('b',30),(NULL,40),('a',50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tgt").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // k: 'a'->{10,50}=2/60, 'b'->{30}=1/30, NULL->{20,40}=2/60. NULL sorts first.
    let r = e
        .execute_resident_expr_select_sql("SELECT k, COUNT(*), SUM(v) FROM tgt GROUP BY k")
        .expect("GROUP BY a nullable text key runs on the GPU");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![SqlValue::Text("a".to_string()), SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![SqlValue::Text("b".to_string()), SqlValue::Int8(1), SqlValue::Int8(30)],
        ],
        "NULL text key forms its own group (sorts first), not folded into a real/empty-string group"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_numeric_key_forms_null_group() {
    // M3 (doc 21): GROUP BY a nullable NUMERIC key — a NULL key forms its own group (the i128 claim path
    // now sees only non-NULL keys; NULLs route to the reserved slot). A NULL is NOT folded into 0.00.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tgn (k NUMERIC(10,2), v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO tgn (k,v) VALUES (1.50,10),(NULL,20),(2.50,30),(NULL,40),(1.50,50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tgn").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // k: 1.50->2/60, 2.50->1/30, NULL->2/60. NULL sorts first.
    let r = e
        .execute_resident_expr_select_sql("SELECT k, COUNT(*), SUM(v) FROM tgn GROUP BY k")
        .expect("GROUP BY a nullable numeric key runs on the GPU");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![
                SqlValue::Numeric(Decimal128::new(150, 2)),
                SqlValue::Int8(2),
                SqlValue::Int8(60)
            ],
            vec![
                SqlValue::Numeric(Decimal128::new(250, 2)),
                SqlValue::Int8(1),
                SqlValue::Int8(30)
            ],
        ],
        "NULL numeric key forms its own group (sorts first), not folded into 0.00"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_uuid_key_forms_null_group() {
    // M3 (doc 21): GROUP BY a nullable UUID key — a NULL key forms its own group (the i128/b128 claim sees
    // only non-NULL keys). A NULL is NOT folded into the all-zero uuid.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tgu (k UUID, v INT)")
        .unwrap();
    let uuid_for = |i: i64| format!("00000000-0000-0000-0000-0000000000{i:02x}");
    e.execute_text(
        2,
        &format!(
            "INSERT INTO tgu (k,v) VALUES ('{}',10),(NULL,20),('{}',30),(NULL,40),('{}',50)",
            uuid_for(5),
            uuid_for(15),
            uuid_for(5),
        ),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tgu").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // k: uuid5->2/60, uuid15->1/30, NULL->2/60. NULL sorts first, then uuid5, uuid15 (byte order).
    let r = e
        .execute_resident_expr_select_sql("SELECT k, COUNT(*), SUM(v) FROM tgu GROUP BY k")
        .expect("GROUP BY a nullable uuid key runs on the GPU");
    let uuid =
        |i: i64| SqlValue::Uuid(gpu_db_sql::uuid::parse_uuid(&uuid_for(i)).expect("valid uuid"));
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![uuid(5), SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![uuid(15), SqlValue::Int8(1), SqlValue::Int8(30)],
        ],
        "NULL uuid key forms its own group (sorts first), not folded into the all-zero uuid"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_numeric_value_skips_nulls() {
    // M3 (doc 21): GROUP BY over a nullable NUMERIC value now runs full 3VL on the GPU. The numeric
    // MIN/MAX is a TWO-PASS kernel: pass 1 finalizes the i128 HIGH limb + records each NON-NULL row's
    // claimed slot into a pooled row_slots scratch; pass 2 (gpu_db_group_by_numeric_minmax_lo) resolves
    // the LOW limb. BOTH passes now read the value validity bitmap and skip NULL rows — so a NULL row's
    // STALE pooled row_slots slot is never folded (the prior 700/OOB hazard). SUM/MIN/MAX skip NULLs;
    // COUNT(*) counts every row; an all-NULL group's aggregate is SQL NULL.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE tn (g INT, v NUMERIC(10,2), w NUMERIC(10,2))",
    )
    .unwrap();
    // g=1 -> v{10.50, 30.25, NULL}: real MIN/MAX distinction (10.50 vs 30.25) + a NULL skip, SUM 40.75.
    // g=2 -> v{5.00, NULL}: one non-NULL + a NULL skip. g=3 -> v{NULL}: an all-NULL group -> NULL. w: none.
    e.execute_text(
        2,
        "INSERT INTO tn (g,v,w) VALUES \
         (1,10.50,1.00),(1,30.25,2.00),(1,NULL,3.00),(2,5.00,4.00),(2,NULL,5.00),(3,NULL,6.00)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tn").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // Dirty the pooled row_slots buffer FIRST with a non-NULL numeric GROUP BY of a DIFFERENT shape, so a
    // stale-slot read in pass 2 (the bug this slice fixes) would surface as a wrong answer below.
    let _ = e
        .execute_resident_expr_select_sql("SELECT g, MIN(w), MAX(w) FROM tn GROUP BY g")
        .expect("non-nullable numeric MIN/MAX runs (dirties row_slots)");

    // SUM(v): NULLs skipped. g=1 -> 40.75 (10.50+30.25), g=2 -> 5.00, g=3 -> NULL (all-NULL).
    let s = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM tn GROUP BY g")
        .expect("SUM over a nullable numeric value runs");
    assert_eq!(
        s.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Numeric(Decimal128::new(4075, 2))
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Numeric(Decimal128::new(500, 2))
            ],
            vec![SqlValue::Int4(3), SqlValue::Null],
        ],
        "SUM(numeric) skips NULLs; an all-NULL group is NULL"
    );
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));

    // MIN(v) + MAX(v): the TWO-PASS path. g=1 -> MIN 10.50 / MAX 30.25 (NULL skipped, not folded as 0);
    // g=2 -> 5.00 / 5.00; g=3 -> NULL / NULL.
    let mm = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v), MAX(v) FROM tn GROUP BY g")
        .expect("MIN/MAX over a nullable numeric value runs");
    assert_eq!(
        mm.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Numeric(Decimal128::new(1050, 2)),
                SqlValue::Numeric(Decimal128::new(3025, 2)),
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Numeric(Decimal128::new(500, 2)),
                SqlValue::Numeric(Decimal128::new(500, 2)),
            ],
            vec![SqlValue::Int4(3), SqlValue::Null, SqlValue::Null],
        ],
        "MIN/MAX(numeric) skip NULLs (never the 0 placeholder); all-NULL group is NULL"
    );

    // COUNT(*) + MIN(v): COUNT(*) counts EVERY row (incl. NULL-v), MIN skips NULLs. g=1 -> (3, 10.50);
    // g=3 -> (1, NULL). Exercises the total-count pass alongside the two-pass numeric MIN.
    let cm = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*), MIN(v) FROM tn GROUP BY g")
        .expect("COUNT(*) + MIN over a nullable numeric value");
    assert_eq!(
        cm.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Int8(3),
                SqlValue::Numeric(Decimal128::new(1050, 2))
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int8(2),
                SqlValue::Numeric(Decimal128::new(500, 2))
            ],
            vec![SqlValue::Int4(3), SqlValue::Int8(1), SqlValue::Null],
        ],
        "COUNT(*) counts NULL-valued numeric rows; MIN skips them"
    );

    // A NON-nullable numeric value (w has no NULLs) is unaffected — no validity bitmap, byte-identical path.
    let ok = e
        .execute_resident_expr_select_sql("SELECT g, SUM(w) FROM tn GROUP BY g")
        .expect("SUM over a NON-nullable numeric value still runs");
    assert_eq!(
        ok.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Numeric(Decimal128::new(600, 2))
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Numeric(Decimal128::new(900, 2))
            ],
            vec![
                SqlValue::Int4(3),
                SqlValue::Numeric(Decimal128::new(600, 2))
            ],
        ],
        "non-nullable numeric SUM is unchanged"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_group_by() {
    // GROUP BY an int4 key on the general GPU executor (hash aggregation): COUNT/SUM/AVG per group,
    // results sorted by key for determinism. Groups: g=1 -> v{10,20,30}, g=2 -> v{5,15}, g=3 -> v{100}.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES (1,10),(2,5),(1,20),(3,100),(2,15),(1,30)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;
    let i8 = SqlValue::Int8;

    // GROUP BY g, COUNT(*): g=1->3, g=2->2, g=3->1.
    let c = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g")
        .expect("grouped count");
    assert_eq!(
        c.rows,
        vec![vec![i4(1), i8(3)], vec![i4(2), i8(2)], vec![i4(3), i8(1)]],
        "GROUP BY count"
    );
    assert_eq!(c.executed_target, DeviceTarget::Gpu(0));

    // GROUP BY g, SUM(v): g=1->60, g=2->20, g=3->100.
    let s = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g")
        .expect("grouped sum");
    assert_eq!(
        s.rows,
        vec![
            vec![i4(1), i8(60)],
            vec![i4(2), i8(20)],
            vec![i4(3), i8(100)]
        ],
        "GROUP BY sum"
    );

    // GROUP BY g, AVG(v): 60/3=20, 20/2=10, 100/1=100 -> numeric scale 16 (PG select_div_scale).
    let a = e
        .execute_resident_expr_select_sql("SELECT g, AVG(v) FROM t GROUP BY g")
        .expect("grouped avg");
    let avg_strs: Vec<String> = a
        .rows
        .iter()
        .map(|r| match &r[1] {
            SqlValue::Numeric(d) => d.to_decimal_string(),
            other => panic!("AVG must be numeric, got {other:?}"),
        })
        .collect();
    assert_eq!(
        a.rows.iter().map(|r| r[0].clone()).collect::<Vec<_>>(),
        vec![i4(1), i4(2), i4(3)],
        "AVG keys"
    );
    assert_eq!(
        avg_strs,
        vec![
            "20.0000000000000000",
            "10.0000000000000000",
            "100.0000000000000000"
        ],
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
    assert_eq!(
        c.columns.len(),
        2,
        "grouped result has key + aggregate columns"
    );
    assert_eq!(
        c.columns[0].name, "g",
        "first result column is the group key"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_expression() {
    // GROUP BY a+b: the expression is materialized ON-DEVICE into a derived int key column the kernel
    // groups by (key_base_override); the result group VALUE is the distinct a+b (not raw a/b), and the
    // SELECT projection of the same expression reads it. Result is key-sorted.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c INT)")
        .unwrap();
    // a+b: (1,2)=3,(2,1)=3,(5,5)=10,(4,4)=8,(3,0)=3,(6,4)=10. groups 3{c:10,20,30}/8{c:40}/10{c:100,200}.
    e.execute_text(
        2,
        "INSERT INTO t (a,b,c) VALUES (1,2,10),(2,1,20),(5,5,100),(4,4,40),(3,0,30),(6,4,200)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;
    let i8 = SqlValue::Int8;
    let g = e
        .execute_resident_expr_select_sql("SELECT a+b, COUNT(*), SUM(c) FROM t GROUP BY a+b")
        .expect("GROUP BY a+b");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![i4(3), i8(3), i8(60)],
            vec![i4(8), i8(1), i8(40)],
            vec![i4(10), i8(2), i8(300)],
        ],
        "GROUP BY a+b -> derived group key + per-group count/sum"
    );
    // GROUP BY a*2 -> 6 distinct doubled keys, key-sorted.
    let m = e
        .execute_resident_expr_select_sql("SELECT a*2, COUNT(*) FROM t GROUP BY a*2")
        .expect("GROUP BY a*2");
    assert_eq!(
        m.rows.iter().map(|r| r[0].clone()).collect::<Vec<_>>(),
        vec![i4(2), i4(4), i4(6), i4(8), i4(10), i4(12)],
        "GROUP BY a*2 -> 6 doubled keys"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_expression_minmax() {
    // MIN/MAX over a value column with an EXPRESSION group key -- orthogonal mechanisms (group by the
    // derived a+b, MIN/MAX over the real column c). Closes the audit-flagged coverage gap.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c INT)")
        .unwrap();
    // a+b groups: (1,2)->3{c=10,30}, (5,3)->8{c=40}, (5,5)->10{c=100}.
    e.execute_text(
        2,
        "INSERT INTO t (a,b,c) VALUES (1,2,10),(2,1,30),(5,3,40),(5,5,100)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a+b, MIN(c), MAX(c) FROM t GROUP BY a+b")
        .expect("GROUP BY a+b with MIN/MAX(c)");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![SqlValue::Int4(3), SqlValue::Int4(10), SqlValue::Int4(30)],
            vec![SqlValue::Int4(8), SqlValue::Int4(40), SqlValue::Int4(40)],
            vec![SqlValue::Int4(10), SqlValue::Int4(100), SqlValue::Int4(100)],
        ],
        "MIN/MAX(c) per derived a+b group"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_expression_empty_table() {
    // GROUP BY <expr> on an EMPTY table -> 0 groups (PG returns no rows), matching the plain-column
    // path. Guards the audit P1: the on-device arith materialize rejects n=0, so the grouped branch
    // now skips it for 0 rows instead of erroring.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a+b, COUNT(*) FROM t GROUP BY a+b")
        .expect("empty-table GROUP BY a+b -> 0 rows, not an error");
    assert!(
        g.rows.is_empty(),
        "0 groups on empty input, got: {:?}",
        g.rows
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_expression_overflow_is_pg_error() {
    // GROUP BY a+b where a+b overflows int4 -> a clean PG "integer out of range" (checked on-device,
    // no wrap, no CPU), inherited from the arith VM -- same as the WHERE/ORDER BY expression paths.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a,b) VALUES (2147483647, 1), (1, 1)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let err = e
        .execute_resident_expr_select_sql("SELECT a+b, COUNT(*) FROM t GROUP BY a+b")
        .expect_err("a+b overflows int4 -> error");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("out of range") || msg.contains("overflow"),
        "checked overflow -> PG error, got: {err:?}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_grouped_min_max() {
    // GROUP BY with per-group MIN/MAX on the general GPU executor (single-level kernel; signed s64
    // atom.min/max). Expected values are CONSTRUCTED from the inserted rows (a GPU-native oracle, not
    // a CPU re-fold): g=1 -> v{10,30,20}; g=2 -> v{5,15,-7}; g=3 -> v{100}. A NEGATIVE value exercises
    // the signed min.
    let mut e = Engine::new_local_cpu_oracle();
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
        vec![
            vec![i4(1), i4(10)],
            vec![i4(2), i4(-7)],
            vec![i4(3), i4(100)]
        ],
        "GROUP BY min"
    );
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));

    // g=1 -> max 30; g=2 -> max 15; g=3 -> max 100.
    let mx = e
        .execute_resident_expr_select_sql("SELECT g, MAX(v) FROM t GROUP BY g")
        .expect("grouped max");
    assert_eq!(
        mx.rows,
        vec![
            vec![i4(1), i4(30)],
            vec![i4(2), i4(15)],
            vec![i4(3), i4(100)]
        ],
        "GROUP BY max"
    );

    // MIN with a WHERE: the predicate-filtered indices feed the same kernel. v > 0 drops (2,-7), so
    // g=2's min becomes 5.
    let mw = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t WHERE v > 0 GROUP BY g")
        .expect("grouped min + where");
    assert_eq!(
        mw.rows,
        vec![
            vec![i4(1), i4(10)],
            vec![i4(2), i4(5)],
            vec![i4(3), i4(100)]
        ],
        "GROUP BY min + WHERE"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_min_max_over_uuid_value() {
    // GROUP BY an int4 key, MIN/MAX of a UUID value on the general GPU executor. uuid MIN/MAX runs the
    // LOCK-FREE b128 CAS-loop kernel (native 128-bit atomic, sm_90+), comparing the two UNSIGNED 64-bit
    // halves = PG's unsigned big-endian memcmp order. Constructed oracle: each group's min/max is the
    // byte-wise (memcmp) min/max of its uuids -- and SqlValue::Uuid bytes ARE in memcmp order, so the
    // lexicographic min/max of the parsed byte arrays is the definitional answer (not a CPU re-fold of
    // the kernel's logic).
    //
    //   g=1  HIGH limb (bytes 0-7) all tie -> the LOW limb (bytes 8-15) decides, and 0x80.. must
    //        compare UNSIGNED (a signed low-limb compare would mis-rank it as negative):
    //          ..-0100-..   byte 8 = 0x01   low limb 0x0100000000000000
    //          ..-8000-..   byte 8 = 0x80   low limb 0x8000000000000000   <- MAX
    //          ..-..00ff    byte 15 = 0xff  low limb 0x00000000000000ff   <- MIN
    //   g=2  the HIGH limb (bytes 0-7) decides; 0xff.. must compare UNSIGNED and DOMINATE a huge low
    //        limb (the min has the LARGEST possible low limb but the smallest high limb):
    //          00000000-..-ffff-ffffffffffff   uhi 0, ulo max              <- MIN
    //          01000000-..                      uhi 0x0100000000000000
    //          ff000000-..                      uhi 0xff00000000000000      <- MAX
    //   g=3  single row -> min == max == the value.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, u UUID)").unwrap();
    let rows: &[(i32, &str)] = &[
        (1, "00000000-0000-0000-0100-000000000000"),
        (1, "00000000-0000-0000-8000-000000000000"),
        (1, "00000000-0000-0000-0000-0000000000ff"),
        (2, "00000000-0000-0000-ffff-ffffffffffff"),
        (2, "01000000-0000-0000-0000-000000000000"),
        (2, "ff000000-0000-0000-0000-000000000000"),
        (3, "12345678-9abc-def0-1234-56789abcdef0"),
    ];
    let values = rows
        .iter()
        .map(|(g, u)| format!("({g}, '{u}')"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, u) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;
    let uuid = |s: &str| SqlValue::Uuid(gpu_db_sql::uuid::parse_uuid(s).expect("valid uuid"));

    // Construction oracle: byte-wise (memcmp) min/max per group, built by sorting the parsed bytes.
    let mut expected_min: Vec<Vec<SqlValue>> = Vec::new();
    let mut expected_max: Vec<Vec<SqlValue>> = Vec::new();
    for g in [1i32, 2, 3] {
        let mut bytes: Vec<[u8; 16]> = rows
            .iter()
            .filter(|(rg, _)| *rg == g)
            .map(|(_, u)| gpu_db_sql::uuid::parse_uuid(u).expect("valid uuid"))
            .collect();
        bytes.sort();
        expected_min.push(vec![i4(g), SqlValue::Uuid(*bytes.first().unwrap())]);
        expected_max.push(vec![i4(g), SqlValue::Uuid(*bytes.last().unwrap())]);
    }

    let mn = e
        .execute_resident_expr_select_sql("SELECT g, MIN(u) FROM t GROUP BY g")
        .expect("uuid grouped min");
    assert_eq!(mn.rows, expected_min, "GROUP BY uuid MIN");
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));

    let mx = e
        .execute_resident_expr_select_sql("SELECT g, MAX(u) FROM t GROUP BY g")
        .expect("uuid grouped max");
    assert_eq!(mx.rows, expected_max, "GROUP BY uuid MAX");

    // Spell the discriminating cases out so a regression names itself.
    // g=1: LOW-limb decides; the 0x80.. value is the MAX only under an UNSIGNED compare.
    assert_eq!(
        mn.rows[0],
        vec![i4(1), uuid("00000000-0000-0000-0000-0000000000ff")],
        "g=1 MIN -- late byte (low limb)"
    );
    assert_eq!(
        mx.rows[0],
        vec![i4(1), uuid("00000000-0000-0000-8000-000000000000")],
        "g=1 MAX -- low-limb high bit set, must be unsigned"
    );
    // g=2: HIGH-limb decides and dominates the low limb; ff.. is the MAX only under an UNSIGNED compare.
    assert_eq!(
        mn.rows[1],
        vec![i4(2), uuid("00000000-0000-0000-ffff-ffffffffffff")],
        "g=2 MIN -- smallest high limb despite a huge low limb"
    );
    assert_eq!(
        mx.rows[1],
        vec![i4(2), uuid("ff000000-0000-0000-0000-000000000000")],
        "g=2 MAX -- early byte (high limb), must be unsigned"
    );

    // SUM/AVG over a uuid have no meaning -> hard error, never a wrong row.
    assert!(
        e.execute_resident_expr_select_sql("SELECT g, SUM(u) FROM t GROUP BY g")
            .is_err(),
        "SUM(uuid) => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_by_numeric_key() {
    // GROUP BY a NUMERIC (i128) key -- the 128-bit key is claimed via the native atom.cas.b128 into
    // slot_keys_i128 (single-level kernel). Covers a NEGATIVE key + a key whose mantissa exceeds 2^64
    // (non-zero HIGH limb), and reconstructs the mantissa @ the column scale. EMPTY128 = i128::MIN is
    // outside the +/-10^38 numeric range, so no real numeric key ever collides with the sentinel.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g NUMERIC(30,4), v INT)")
        .unwrap();
    let rows: &[(&str, i32)] = &[
        ("12.5000", 10),
        ("12.5000", 5),
        ("-7.2500", 100),
        ("-7.2500", 3),
        ("1234567890123456789.0000", 1), // mantissa 1.23e22 > 2^64 -> exercises the high limb
        ("1234567890123456789.0000", 2),
    ];
    let values = rows
        .iter()
        .map(|(g, v)| format!("({g}, {v})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let num = |m: i128| SqlValue::Numeric(Decimal128::new(m, 4));
    let big: i128 = 1_234_567_890_123_456_789_i128 * 10_000;

    // Output sorts by numeric VALUE: -7.25 < 12.5 < 1.23e18.
    let count = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g")
        .expect("numeric-key COUNT");
    assert_eq!(count.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        count.rows,
        vec![
            vec![num(-72_500), SqlValue::Int8(2)],
            vec![num(125_000), SqlValue::Int8(2)],
            vec![num(big), SqlValue::Int8(2)],
        ],
        "GROUP BY numeric key, COUNT"
    );

    let sum = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g")
        .expect("numeric-key SUM");
    assert_eq!(
        sum.rows,
        vec![
            vec![num(-72_500), SqlValue::Int8(103)],
            vec![num(125_000), SqlValue::Int8(15)],
            vec![num(big), SqlValue::Int8(3)],
        ],
        "GROUP BY numeric key, SUM(int4)->bigint"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_bool_key() {
    // GROUP BY a BOOL column -- 2 groups (false<true) via the bool->int4 materialize + key_base_override
    // (the audited int4 path; NO bool GROUP BY kernel -> no concurrency hazard).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (flag BOOL, i INT)")
        .unwrap();
    // false: i={10,30} (count 2, sum 40); true: i={20,40,50} (count 3, sum 110).
    e.execute_text(
        2,
        "INSERT INTO t (flag, i) VALUES (false,10),(true,20),(false,30),(true,40),(true,50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT flag, COUNT(*), SUM(i) FROM t GROUP BY flag")
        .expect("GROUP BY bool key");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![SqlValue::Bool(false), SqlValue::Int8(2), SqlValue::Int8(40)],
            vec![SqlValue::Bool(true), SqlValue::Int8(3), SqlValue::Int8(110)],
        ],
        "GROUP BY bool -> false then true, count + sum"
    );
    let m = e
        .execute_resident_expr_select_sql("SELECT flag, MIN(i) FROM t GROUP BY flag")
        .expect("bool key + MIN(int)");
    assert_eq!(
        m.rows,
        vec![
            vec![SqlValue::Bool(false), SqlValue::Int4(10)],
            vec![SqlValue::Bool(true), SqlValue::Int4(20)],
        ],
        "bool key + MIN(int4)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_bool_minmax_value() {
    // MIN/MAX over a BOOL VALUE (int key): group all-false -> min=max=false; all-true -> true; mixed ->
    // min=false, max=true. Via bool->int4 materialize + value_base_override.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (k INT, flag BOOL)")
        .unwrap();
    // k=1: {false,false}; k=2: {true,true}; k=3: {false,true}.
    e.execute_text(
        2,
        "INSERT INTO t (k, flag) VALUES (1,false),(1,false),(2,true),(2,true),(3,false),(3,true)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT k, MIN(flag), MAX(flag) FROM t GROUP BY k")
        .expect("MIN/MAX over bool value");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Bool(false),
                SqlValue::Bool(false)
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Bool(true),
                SqlValue::Bool(true)
            ],
            vec![
                SqlValue::Int4(3),
                SqlValue::Bool(false),
                SqlValue::Bool(true)
            ],
        ],
        "MIN/MAX(bool) per int group"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_two_columns() {
    // Composite GROUP BY a, b: two int4 columns packed on-device into one i64 key `(a<<32)|b`, grouped,
    // then the result UNPACKS it back into a, b. Distinct (a,b) tuples; same-a-diff-b are distinct;
    // identical (a,b) MERGE. Positive values -> the packed-key order equals (a,b) order.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c INT)")
        .unwrap();
    // (a,b): (1,1)x2 c={10,20}, (1,2)x1 c={5}, (2,1)x1 c={7}.
    e.execute_text(
        2,
        "INSERT INTO t (a,b,c) VALUES (1,1,10),(1,1,20),(1,2,5),(2,1,7)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*), SUM(c) FROM t GROUP BY a, b")
        .expect("composite GROUP BY a, b");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Int4(1),
                SqlValue::Int8(2),
                SqlValue::Int8(30)
            ],
            vec![
                SqlValue::Int4(1),
                SqlValue::Int4(2),
                SqlValue::Int8(1),
                SqlValue::Int8(5)
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int4(1),
                SqlValue::Int8(1),
                SqlValue::Int8(7)
            ],
        ],
        "composite (a,b) groups + count + sum, unpacked"
    );
    // Guard the result-SCHEMA patch (insert b's column + renumber attnums): a rows-only assertion lets
    // a dropped `insert(1, b_col)` slip past (the audit's Fault B). The columns must be [a, b, ...] with
    // the right group-column names/types.
    assert_eq!(
        g.columns.len(),
        4,
        "composite result columns: a, b, count, sum"
    );
    assert_eq!(g.columns[0].name, "a");
    assert_eq!(g.columns[0].ty, SqlType::Int4);
    assert_eq!(g.columns[1].name, "b");
    assert_eq!(g.columns[1].ty, SqlType::Int4);
    assert_eq!(g.columns[2].ty, SqlType::Int8);
    assert_eq!(g.columns[3].ty, SqlType::Int8);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_negatives_ordered() {
    // Negative members round-trip through the `as u32 / as i32` pack/unpack; ORDER BY a, b gives the
    // true (a,b) order (the host group order is by the packed key, signed).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    // (a,b): (-1,5)x2, (2,-3)x1, (-1,4)x1. ORDER BY a,b: (-1,4),(-1,5),(2,-3).
    e.execute_text(2, "INSERT INTO t (a,b) VALUES (-1,5),(-1,5),(2,-3),(-1,4)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql(
            "SELECT a, b, COUNT(*) FROM t GROUP BY a, b ORDER BY a, b",
        )
        .expect("composite GROUP BY with negatives + ORDER BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![SqlValue::Int4(-1), SqlValue::Int4(4), SqlValue::Int8(1)],
            vec![SqlValue::Int4(-1), SqlValue::Int4(5), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int4(-3), SqlValue::Int8(1)],
        ],
        "negative composite keys round-trip, ORDER BY a,b true order"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_int8_member_bare() {
    // Composite GROUP BY a, b where a is BIGINT (so combined width > 64 bits) -> the i128 pack
    // (col0 high 64, col1 low 64) + the b128 claim, UNPACKED back to (a:int8, b:int4). Bare GROUP BY:
    // default order is by a (distinct here). a holds a value beyond the int4 range.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, b INT)")
        .unwrap();
    // (a,b): (100,1)x2, (200,2)x1, (9000000000,3)x1 -> 3 groups, distinct a.
    e.execute_text(
        2,
        "INSERT INTO t (a,b) VALUES (100,1),(100,1),(200,2),(9000000000,3)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*) FROM t GROUP BY a, b")
        .expect("composite int8+int4 GROUP BY (i128 pack)");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![SqlValue::Int8(100), SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int8(200), SqlValue::Int4(2), SqlValue::Int8(1)],
            vec![
                SqlValue::Int8(9000000000),
                SqlValue::Int4(3),
                SqlValue::Int8(1)
            ],
        ],
        "int8+int4 composite unpacks to (int8, int4); value beyond int4 range survives"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_single_bigint_key_i64_min_dedicated_slot() {
    // The slot table's EMPTY sentinel is i64::MIN, so a GROUP BY key == i64::MIN cannot live in the
    // hash table -- the kernel routes it to the DEDICATED slot (idx = nslots). The result-path GPU
    // stream-compaction must emit that dedicated slot (presence + correct stats). This is the BARE
    // single-BIGINT i64 path, NOT the composite/i128 path that `*_two_int8_min_edge` exercises.
    // (Audit follow-up to the stream-compaction commit -- closes the i64::MIN-bare-key coverage gap.)
    use std::collections::BTreeMap;
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, v INT)")
        .unwrap();
    e.execute_text(
        2,
        &format!(
            "INSERT INTO t (a,v) VALUES ({min},10),({min},20),(100,1),(200,2),(200,3)",
            min = i64::MIN
        ),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, COUNT(*), SUM(v) FROM t GROUP BY a")
        .expect("single bigint GROUP BY incl the i64::MIN dedicated slot");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    let got: BTreeMap<i64, (i64, i64)> = g
        .rows
        .iter()
        .map(|r| {
            let (SqlValue::Int8(k), SqlValue::Int8(c), SqlValue::Int8(s)) = (&r[0], &r[1], &r[2])
            else {
                panic!("expected (Int8 key, Int8 count, Int8 sum), got {r:?}");
            };
            (*k, (*c, *s))
        })
        .collect();
    assert_eq!(
        got.len(),
        3,
        "exactly 3 groups (incl the i64::MIN dedicated slot)"
    );
    assert_eq!(
        got.get(&i64::MIN),
        Some(&(2, 30)),
        "i64::MIN key (dedicated slot) => count 2, sum 30"
    );
    assert_eq!(got.get(&100), Some(&(1, 1)), "100 => count 1, sum 1");
    assert_eq!(got.get(&200), Some(&(2, 5)), "200 => count 2, sum 5");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_int8_and_int4_ordered() {
    // Composite GROUP BY a, b (a BIGINT, b INT) with a NEGATIVE wide member + a duplicate group +
    // SUM(c); ORDER BY a, b gives the true (a,b) order over the unpacked columns.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, b INT, c INT)")
        .unwrap();
    // (a,b,c): (9e9,1,10),(9e9,1,20),(9e9,2,5),(-5,1,7).
    e.execute_text(
        2,
        "INSERT INTO t (a,b,c) VALUES \
         (9000000000,1,10),(9000000000,1,20),(9000000000,2,5),(-5,1,7)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql(
            "SELECT a, b, COUNT(*), SUM(c) FROM t GROUP BY a, b ORDER BY a, b",
        )
        .expect("composite int8+int4 GROUP BY with negatives + ORDER BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Int8(-5),
                SqlValue::Int4(1),
                SqlValue::Int8(1),
                SqlValue::Int8(7),
            ],
            vec![
                SqlValue::Int8(9000000000),
                SqlValue::Int4(1),
                SqlValue::Int8(2),
                SqlValue::Int8(30),
            ],
            vec![
                SqlValue::Int8(9000000000),
                SqlValue::Int4(2),
                SqlValue::Int8(1),
                SqlValue::Int8(5),
            ],
        ],
        "negative + wide composite keys round-trip, ORDER BY a,b true order"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_two_int8_min_edge() {
    // Composite GROUP BY a, b where BOTH are BIGINT, INCLUDING the (i64::MIN, 0) tuple whose i128 pack
    // == i128::MIN == EMPTY128 -- it must route to the b128 claim's DEDICATED slot, not vanish.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, b BIGINT, c INT)")
        .unwrap();
    // (i64::MIN, 0)x2 [the EMPTY128 edge], (5e9, 6e9)x1.
    e.execute_text(
        2,
        "INSERT INTO t (a,b,c) VALUES \
         (-9223372036854775808,0,1),(-9223372036854775808,0,2),(5000000000,6000000000,3)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql(
            "SELECT a, b, COUNT(*), SUM(c) FROM t GROUP BY a, b ORDER BY a, b",
        )
        .expect("composite two-int8 GROUP BY incl. the i64::MIN/EMPTY128 edge");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Int8(i64::MIN),
                SqlValue::Int8(0),
                SqlValue::Int8(2),
                SqlValue::Int8(3),
            ],
            vec![
                SqlValue::Int8(5000000000),
                SqlValue::Int8(6000000000),
                SqlValue::Int8(1),
                SqlValue::Int8(3),
            ],
        ],
        "the (i64::MIN, 0) composite routes to the dedicated slot (not lost to EMPTY128)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_int_and_text() {
    // Composite GROUP BY a, b where a is INT and b is TEXT -> the text-key b128 claim with the fixed
    // member (a) folded into the hash + verify (key_base_override). CRITICAL: the SAME text "x" appears
    // under a=1 AND a=2 -> they MUST be distinct groups (the fixed member splits them). Single agg.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b TEXT)").unwrap();
    // (a,b): (1,"x")x2, (1,"y")x1, (2,"x")x1.
    e.execute_text(
        2,
        "INSERT INTO t (a,b) VALUES (1,'x'),(1,'x'),(1,'y'),(2,'x')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*) FROM t GROUP BY a, b")
        .expect("composite (int, text) GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("x".into()),
                SqlValue::Int8(2)
            ],
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("y".into()),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("x".into()),
                SqlValue::Int8(1)
            ],
        ],
        "same text under different fixed members are distinct groups (default order by a,b)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_text_first_with_sum() {
    // Composite GROUP BY name, k where name is TEXT (the FIRST member) and k is INT, with SUM(c) (single
    // aggregate). Verifies declared member ORDER in the result (text, int) + a non-COUNT aggregate.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (name TEXT, k INT, c INT)")
        .unwrap();
    // ("apple",1,10),("apple",1,20),("apple",2,5),("banana",1,7).
    e.execute_text(
        2,
        "INSERT INTO t (name,k,c) VALUES ('apple',1,10),('apple',1,20),('apple',2,5),('banana',1,7)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT name, k, SUM(c) FROM t GROUP BY name, k")
        .expect("composite (text, int) GROUP BY with SUM");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Text("apple".into()),
                SqlValue::Int4(1),
                SqlValue::Int8(30)
            ],
            vec![
                SqlValue::Text("apple".into()),
                SqlValue::Int4(2),
                SqlValue::Int8(5)
            ],
            vec![
                SqlValue::Text("banana".into()),
                SqlValue::Int4(1),
                SqlValue::Int8(7)
            ],
        ],
        "text-first composite, SUM per (name,k), default order by name,k"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_int8_and_text() {
    // Composite GROUP BY a, b where a is BIGINT (width-8 widen) + b is TEXT, with a value beyond the
    // int4 range. Exercises the width-8 fixed-member widen folded into the text-key hash/verify.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, b TEXT)")
        .unwrap();
    // (9e9,"x")x2, (9e9,"y")x1, (5,"x")x1.
    e.execute_text(
        2,
        "INSERT INTO t (a,b) VALUES (9000000000,'x'),(9000000000,'x'),(9000000000,'y'),(5,'x')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*) FROM t GROUP BY a, b")
        .expect("composite (int8, text) GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Int8(5),
                SqlValue::Text("x".into()),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Int8(9000000000),
                SqlValue::Text("x".into()),
                SqlValue::Int8(2)
            ],
            vec![
                SqlValue::Int8(9000000000),
                SqlValue::Text("y".into()),
                SqlValue::Int8(1)
            ],
        ],
        "int8 fixed member (width-8 widen) + text, value beyond int4 range"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_fixed_text_multi_aggregate_rejected() {
    // A (fixed, text) composite supports a SINGLE aggregate; multiple aggregates need per-pass alignment
    // (a follow-up) -> clean reject (at execution, on the GPU path).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b TEXT, c INT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO t (a,b,c) VALUES (1,'x',10)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let err = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*), SUM(c) FROM t GROUP BY a, b")
        .expect_err("multi-aggregate (fixed, text) composite rejected");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("single aggregate") || msg.contains("follow-up"),
        "clean reject, got: {err:?}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_three_int_columns() {
    // >2 columns (all fixed-width int) -> the general WIDE-KEY path (gpu_db_build_wide_key + the
    // (rep_idx, hash) b128 claim with a memcmp verify). Distinct (a,b,c) tuples by construction.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c INT)")
        .unwrap();
    // (1,1,1)x2, (1,1,2)x1, (1,2,1)x1, (2,1,1)x1.
    e.execute_text(
        2,
        "INSERT INTO t (a,b,c) VALUES (1,1,1),(1,1,1),(1,1,2),(1,2,1),(2,1,1)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, b, c, COUNT(*) FROM t GROUP BY a, b, c")
        .expect("3-column wide-key GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Int4(1),
                SqlValue::Int4(1),
                SqlValue::Int8(2)
            ],
            vec![
                SqlValue::Int4(1),
                SqlValue::Int4(1),
                SqlValue::Int4(2),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Int4(1),
                SqlValue::Int4(2),
                SqlValue::Int4(1),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int4(1),
                SqlValue::Int4(1),
                SqlValue::Int8(1)
            ],
        ],
        "distinct (a,b,c) tuples, default order by the full tuple"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_two_text() {
    // Composite GROUP BY a, b where BOTH members are TEXT -> the general wide-key path with NO fixed
    // members (comp_w = 0) + TWO text descriptors (n_text = 2): the claim folds + byte-verifies each
    // text member. The SAME first text under different second texts must be distinct groups.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a TEXT, b TEXT)")
        .unwrap();
    // (x,p)x2, (x,q)x1, (y,p)x1.
    e.execute_text(
        2,
        "INSERT INTO t (a,b) VALUES ('x','p'),('x','p'),('x','q'),('y','p')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*) FROM t GROUP BY a, b")
        .expect("two-text composite GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Text("x".into()),
                SqlValue::Text("p".into()),
                SqlValue::Int8(2)
            ],
            vec![
                SqlValue::Text("x".into()),
                SqlValue::Text("q".into()),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Text("y".into()),
                SqlValue::Text("p".into()),
                SqlValue::Int8(1)
            ],
        ],
        "two text members, distinct (a,b) groups, default order by (a,b)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_two_text_concat_ambiguity() {
    // CRITICAL adversarial case: ('ab','c') and ('a','bc') must be DISTINCT groups even though a naive
    // concatenated hash of the member bytes ("abc") collides -- the PER-MEMBER byte-verify distinguishes
    // them (member 0 "ab" != "a"). ('a','c') is a third distinct group sharing member 0 with ('a','bc').
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a TEXT, b TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a,b) VALUES ('ab','c'),('a','bc'),('a','c')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*) FROM t GROUP BY a, b")
        .expect("two-text concat-ambiguity GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Text("a".into()),
                SqlValue::Text("bc".into()),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Text("a".into()),
                SqlValue::Text("c".into()),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Text("ab".into()),
                SqlValue::Text("c".into()),
                SqlValue::Int8(1)
            ],
        ],
        "concatenation-ambiguous member splits stay distinct (the per-member verify, not the hash)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_two_text_empty_member() {
    // Empty-string text members: a zero-length member (offsets[i]==offsets[i+1]) hashes to nothing and
    // verifies as a 0-byte compare. ('','x'), ('x',''), and ('','') are three distinct groups.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a TEXT, b TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a,b) VALUES ('','x'),('','x'),('x',''),('','')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*) FROM t GROUP BY a, b")
        .expect("two-text empty-member GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Text("".into()),
                SqlValue::Text("".into()),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Text("".into()),
                SqlValue::Text("x".into()),
                SqlValue::Int8(2)
            ],
            vec![
                SqlValue::Text("x".into()),
                SqlValue::Text("".into()),
                SqlValue::Int8(1)
            ],
        ],
        "empty-string text members group correctly (order by (a,b): '' < 'x')"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_int_text_int() {
    // A TEXT member in a >2-column composite: (int, text, int) -> the general wide-key path with TWO
    // fixed members (comp_w = 16) + ONE text member (n_text = 1). The SAME text under different fixed
    // members are distinct groups -- exercises BOTH the fixed memcmp AND the text byte-verify legs.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b TEXT, c INT)")
        .unwrap();
    // (1,x,1)x2, (1,x,2)x1, (1,y,1)x1, (2,x,1)x1.
    e.execute_text(
        2,
        "INSERT INTO t (a,b,c) VALUES (1,'x',1),(1,'x',1),(1,'x',2),(1,'y',1),(2,'x',1)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, b, c, COUNT(*) FROM t GROUP BY a, b, c")
        .expect("(int, text, int) composite GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("x".into()),
                SqlValue::Int4(1),
                SqlValue::Int8(2)
            ],
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("x".into()),
                SqlValue::Int4(2),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("y".into()),
                SqlValue::Int4(1),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("x".into()),
                SqlValue::Int4(1),
                SqlValue::Int8(1)
            ],
        ],
        "text member among fixed members, distinct (a,b,c), order by the full tuple"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_two_text_and_int_with_sum() {
    // Mixed (text, text, int) composite with a non-COUNT aggregate (SUM) -> comp_w = 8 (the int) +
    // n_text = 2. Verifies declared member ORDER (text, text, int) in the result + the value aggregate.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a TEXT, b TEXT, k INT, v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a,b,k,v) VALUES ('x','p',1,10),('x','p',1,20),('x','q',1,5),('y','p',2,7)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, b, k, SUM(v) FROM t GROUP BY a, b, k")
        .expect("(text, text, int) composite GROUP BY with SUM");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Text("x".into()),
                SqlValue::Text("p".into()),
                SqlValue::Int4(1),
                SqlValue::Int8(30)
            ],
            vec![
                SqlValue::Text("x".into()),
                SqlValue::Text("q".into()),
                SqlValue::Int4(1),
                SqlValue::Int8(5)
            ],
            vec![
                SqlValue::Text("y".into()),
                SqlValue::Text("p".into()),
                SqlValue::Int4(2),
                SqlValue::Int8(7)
            ],
        ],
        "(text, text, int) composite, SUM per group, default order by (a,b,k)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_bool_member() {
    // A BOOL composite member (1-byte resident, widened 0/1 -> i64 by build kind 3) -> the wide-key
    // path. (bool, int): (true,1)x2,(true,2)x1,(false,1)x1. Default order by (flag,k): false(0)<true(1).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (flag BOOL, k INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (flag, k) VALUES (true,1),(true,1),(true,2),(false,1)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT flag, k, COUNT(*) FROM t GROUP BY flag, k")
        .expect("(bool, int) composite GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![SqlValue::Bool(false), SqlValue::Int4(1), SqlValue::Int8(1)],
            vec![SqlValue::Bool(true), SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Bool(true), SqlValue::Int4(2), SqlValue::Int8(1)],
        ],
        "bool member groups by 0/1, distinct (flag,k), order by (flag,k)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_bool_member_word_boundary() {
    // >32 rows so the bool BITMAP spans TWO LE u32 words -> the (i/32)*4 word-index math in wk_bool is
    // exercised ACROSS the word boundary (the prior gap: 4-row tests stay in word 0). flag = row >= 20
    // (the true group crosses row 32); k = row % 2. 4 groups x 10 rows each.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (flag BOOL, k INT)")
        .unwrap();
    let values = (0..40)
        .map(|r| format!("({}, {})", if r >= 20 { "true" } else { "false" }, r % 2))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (flag, k) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT flag, k, COUNT(*) FROM t GROUP BY flag, k")
        .expect("(bool, int) composite GROUP BY across a bitmap word boundary");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![SqlValue::Bool(false), SqlValue::Int4(0), SqlValue::Int8(10)],
            vec![SqlValue::Bool(false), SqlValue::Int4(1), SqlValue::Int8(10)],
            vec![SqlValue::Bool(true), SqlValue::Int4(0), SqlValue::Int8(10)],
            vec![SqlValue::Bool(true), SqlValue::Int4(1), SqlValue::Int8(10)],
        ],
        "bool bitmap read is correct across the 32-row word boundary"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_bool_and_text_member() {
    // A BOOL fixed member + a TEXT member -> the general wide-key (comp_w=8) + text descriptor (n_text=1)
    // path. (true,a)x2,(false,a)x1,(true,b)x1. Order by (flag,name): false<true.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (flag BOOL, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (flag, name) VALUES (true,'a'),(true,'a'),(false,'a'),(true,'b')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT flag, name, COUNT(*) FROM t GROUP BY flag, name")
        .expect("(bool, text) composite GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Bool(false),
                SqlValue::Text("a".into()),
                SqlValue::Int8(1)
            ],
            vec![
                SqlValue::Bool(true),
                SqlValue::Text("a".into()),
                SqlValue::Int8(2)
            ],
            vec![
                SqlValue::Bool(true),
                SqlValue::Text("b".into()),
                SqlValue::Int8(1)
            ],
        ],
        "bool fixed member + text member, distinct (flag,name), order by (flag,name)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_numeric_member() {
    // A composite with a NUMERIC member (can't pack into <=128 bits with another) -> the wide-key path
    // (16 bytes for the numeric + 8 for the int). SUM(c) (single aggregate). Construction oracle.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (n NUMERIC(10,2), k INT, c INT)")
        .unwrap();
    // (1.50,1,10),(1.50,1,20),(1.50,2,5),(2.50,1,7).
    e.execute_text(
        2,
        "INSERT INTO t (n,k,c) VALUES (1.50,1,10),(1.50,1,20),(1.50,2,5),(2.50,1,7)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT n, k, SUM(c) FROM t GROUP BY n, k")
        .expect("composite (numeric, int) wide-key GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    let num = |m: i128| SqlValue::Numeric(Decimal128::new(m, 2));
    assert_eq!(
        g.rows,
        vec![
            vec![num(150), SqlValue::Int4(1), SqlValue::Int8(30)],
            vec![num(150), SqlValue::Int4(2), SqlValue::Int8(5)],
            vec![num(250), SqlValue::Int4(1), SqlValue::Int8(7)],
        ],
        "(numeric, int) composite, SUM per group, default order by (n,k)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_int8_and_numeric_member() {
    // A composite of an INT8 member + a NUMERIC member -> the wide-key path with BOTH a wk_int8 (8-byte)
    // and a wk_i128 (16-byte) leg in gpu_db_build_wide_key. The int8 value is beyond the int4 range, so
    // a truncated (4-byte) int8 write would mis-group it -> this exercises the wk_int8 build leg.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, n NUMERIC(10,2))")
        .unwrap();
    // (9e9,1.50)x2, (9e9,2.50)x1, (5,1.50)x1.
    e.execute_text(
        2,
        "INSERT INTO t (a,n) VALUES (9000000000,1.50),(9000000000,1.50),(9000000000,2.50),(5,1.50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a, n, COUNT(*) FROM t GROUP BY a, n")
        .expect("composite (int8, numeric) wide-key GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    let num = |m: i128| SqlValue::Numeric(Decimal128::new(m, 2));
    assert_eq!(
        g.rows,
        vec![
            vec![SqlValue::Int8(5), num(150), SqlValue::Int8(1)],
            vec![SqlValue::Int8(9000000000), num(150), SqlValue::Int8(2)],
            vec![SqlValue::Int8(9000000000), num(250), SqlValue::Int8(1)],
        ],
        "int8 (8-byte) + numeric (16-byte) wide-key legs; int8 beyond int4 range survives"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_uuid_member() {
    // A composite with a UUID member -> the wide-key path (16 bytes for the uuid + 8 for the int).
    let a = "11111111-1111-1111-1111-111111111111";
    let b = "22222222-2222-2222-2222-222222222222";
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (id UUID, k INT)")
        .unwrap();
    // (a,1)x2, (a,2)x1, (b,1)x1.
    e.execute_text(
        2,
        &format!("INSERT INTO t (id,k) VALUES ('{a}',1),('{a}',1),('{a}',2),('{b}',1)"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT id, k, COUNT(*) FROM t GROUP BY id, k")
        .expect("composite (uuid, int) wide-key GROUP BY");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    let uid = |s: &str| SqlValue::Uuid(gpu_db_sql::uuid::parse_uuid(s).expect("valid uuid"));
    assert_eq!(
        g.rows,
        vec![
            vec![uid(a), SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![uid(a), SqlValue::Int4(2), SqlValue::Int8(1)],
            vec![uid(b), SqlValue::Int4(1), SqlValue::Int8(1)],
        ],
        "(uuid, int) composite, default order by (id,k)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_composite_widekey_multi_aggregate_rejected() {
    // A wide-key composite supports a SINGLE aggregate (multi-pass alignment is a follow-up) -> reject.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c INT, d INT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO t (a,b,c,d) VALUES (1,1,1,10)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let err = e
        .execute_resident_expr_select_sql(
            "SELECT a, b, c, COUNT(*), SUM(d) FROM t GROUP BY a, b, c",
        )
        .expect_err("multi-aggregate wide-key composite rejected");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("single aggregate") || msg.contains("follow-up"),
        "clean reject, got: {err:?}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_by_text_key() {
    // GROUP BY a TEXT (varlen) key -- the kernel FNV-1a-hashes the bytes, claims a b128
    // (representative_row_idx, hash) in slot_keys_i128 via atom.cas.b128 with a full-text
    // VERIFY-ON-LOST-CAS, so same-text rows COLLAPSE into one group and hash collisions never merge.
    // Covers duplicates (apple x3), an EMPTY string, different lengths, and a SHARED PREFIX (app vs
    // apple) to exercise the length-check + byte-compare in the verify. Result key is read host-side
    // from the representative row.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g TEXT, v INT)").unwrap();
    let rows: &[(&str, i32)] = &[
        ("apple", 10),
        ("apple", 20),
        ("apple", 30),
        ("banana", 5),
        ("banana", 15),
        ("cherry", 100),
        ("", 7),
        ("app", 1),
    ];
    let values = rows
        .iter()
        .map(|(g, v)| format!("('{g}', {v})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let txt = |s: &str| SqlValue::Text(s.to_string());

    // Output sorts lexicographically: "" < "app" < "apple" < "banana" < "cherry".
    let count = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g")
        .expect("text-key COUNT");
    assert_eq!(count.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        count.rows,
        vec![
            vec![txt(""), SqlValue::Int8(1)],
            vec![txt("app"), SqlValue::Int8(1)],
            vec![txt("apple"), SqlValue::Int8(3)], // x3 collapsed into ONE group
            vec![txt("banana"), SqlValue::Int8(2)],
            vec![txt("cherry"), SqlValue::Int8(1)],
        ],
        "GROUP BY text key COUNT -- same-text rows collapse; app/apple stay separate"
    );

    let sum = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g")
        .expect("text-key SUM");
    assert_eq!(
        sum.rows,
        vec![
            vec![txt(""), SqlValue::Int8(7)],
            vec![txt("app"), SqlValue::Int8(1)],
            vec![txt("apple"), SqlValue::Int8(60)],
            vec![txt("banana"), SqlValue::Int8(20)],
            vec![txt("cherry"), SqlValue::Int8(100)],
        ],
        "GROUP BY text key, SUM(int4)->bigint"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_min_max_over_text_value() {
    // Grouped MIN/MAX over a TEXT VALUE -- the kernel keeps each group's min/max value text's ROW INDEX
    // in slot_min/slot_max (EMPTY = u64::MAX) via a lock-free CAS loop with a LEXICOGRAPHIC byte compare
    // (first differing byte unsigned; a strict prefix is smaller). Exercises a shared PREFIX (app<apple),
    // an EMPTY string (the MIN of its group), different lengths, a last-byte-only difference, and a
    // single-row group (MIN == MAX). Result text is read host-side from the winning row.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, s TEXT)").unwrap();
    let rows: &[(i32, &str)] = &[
        (1, "apple"),
        (1, "app"),     // prefix of "apple" -> "app" < "apple"
        (1, "apricot"), // "apple" < "apricot"
        (2, "z"),
        (2, ""), // empty string is the MIN of group 2
        (2, "a"),
        (3, "xy1"),
        (3, "xy2"), // last-byte-only difference
        (3, "xy0"),
        (4, "solo"), // single row: MIN == MAX
    ];
    let values = rows
        .iter()
        .map(|(g, s)| format!("({g}, '{s}')"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, s) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let txt = |s: &str| SqlValue::Text(s.to_string());

    let min = e
        .execute_resident_expr_select_sql("SELECT g, MIN(s) FROM t GROUP BY g")
        .expect("text-value MIN");
    assert_eq!(min.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        min.rows,
        vec![
            vec![SqlValue::Int4(1), txt("app")], // app < apple < apricot
            vec![SqlValue::Int4(2), txt("")],    // empty string is smallest
            vec![SqlValue::Int4(3), txt("xy0")],
            vec![SqlValue::Int4(4), txt("solo")],
        ],
        "grouped MIN(text): prefix app<apple, empty string is the min"
    );

    let max = e
        .execute_resident_expr_select_sql("SELECT g, MAX(s) FROM t GROUP BY g")
        .expect("text-value MAX");
    assert_eq!(
        max.rows,
        vec![
            vec![SqlValue::Int4(1), txt("apricot")],
            vec![SqlValue::Int4(2), txt("z")],
            vec![SqlValue::Int4(3), txt("xy2")],
            vec![SqlValue::Int4(4), txt("solo")],
        ],
        "grouped MAX(text)"
    );

    // SUM over a text value must hard-error (text is MIN/MAX/COUNT only).
    assert!(
        e.execute_resident_expr_select_sql("SELECT g, SUM(s) FROM t GROUP BY g")
            .is_err(),
        "SUM(text) must error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_text_value_after_text_key_offset_alignment() {
    // Regression: a text VALUE placed after a text KEY whose bytes are NOT a 4-multiple lands the
    // value-text offsets at a non-4-aligned device offset. The kernels read each 8-byte offset entry as
    // 2x `ld.global.u32` (4-byte alignment required), so the unaligned section faulted CUDA 716 (and
    // pinned the GPU ~20s) until engine_residency aligned every varlen offsets section to 8 bytes. The
    // key bytes here sum to 11 ('app'x2 + 'be'x2 + 'c' -- a non-4-multiple), which previously misaligned
    // the value offsets. GROUP BY a text key with MIN/MAX over a text value must now run cleanly.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE w (k TEXT, v TEXT)")
        .unwrap();
    let rows: &[(&str, &str)] = &[
        ("app", "banana"),
        ("app", "apple"),
        ("be", "cherry"),
        ("be", "date"),
        ("c", "fig"),
    ];
    let values = rows
        .iter()
        .map(|(k, v)| format!("('{k}', '{v}')"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO w (k, v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("w").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let txt = |s: &str| SqlValue::Text(s.to_string());

    let min = e
        .execute_resident_expr_select_sql("SELECT k, MIN(v) FROM w GROUP BY k")
        .expect("text key + text value MIN must not fault on a non-4-multiple key-bytes layout");
    assert_eq!(min.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        min.rows,
        vec![
            vec![txt("app"), txt("apple")],
            vec![txt("be"), txt("cherry")],
            vec![txt("c"), txt("fig")],
        ],
        "GROUP BY text key, MIN(text value) -- value offsets must be 8-aligned"
    );

    let max = e
        .execute_resident_expr_select_sql("SELECT k, MAX(v) FROM w GROUP BY k")
        .expect("text key + text value MAX");
    assert_eq!(
        max.rows,
        vec![
            vec![txt("app"), txt("banana")],
            vec![txt("be"), txt("date")],
            vec![txt("c"), txt("fig")],
        ],
        "GROUP BY text key, MAX(text value)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_multiple_aggregates_same_value_column() {
    // SELECT g, COUNT(*), SUM(v), AVG(v), MIN(v), MAX(v) FROM t GROUP BY g -- FIVE aggregates over ONE
    // value column, projected from a SINGLE kernel pass (count+sum+min+max are computed together).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    let rows: &[(i32, i32)] = &[(1, 10), (1, 20), (1, 30), (2, 5), (2, 15), (3, 100)];
    let values = rows
        .iter()
        .map(|(g, v)| format!("({g}, {v})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*), SUM(v), AVG(v), MIN(v), MAX(v) FROM t GROUP BY g",
        )
        .expect("multiple aggregates over one value column");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    // [g, COUNT->int8, SUM(int4)->int8, AVG->numeric@16, MIN->int4, MAX->int4]
    assert_eq!(
        res.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Int8(3),
                SqlValue::Int8(60),
                average_sql_value(60, 3),
                SqlValue::Int4(10),
                SqlValue::Int4(30),
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int8(2),
                SqlValue::Int8(20),
                average_sql_value(20, 2),
                SqlValue::Int4(5),
                SqlValue::Int4(15),
            ],
            vec![
                SqlValue::Int4(3),
                SqlValue::Int8(1),
                SqlValue::Int8(100),
                average_sql_value(100, 1),
                SqlValue::Int4(100),
                SqlValue::Int4(100),
            ],
        ],
        "COUNT/SUM/AVG/MIN/MAX over one value column in one pass"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_basic() {
    // SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g -- distinct counts KNOWN BY CONSTRUCTION:
    //   g=1: v in {10,10,20} -> 2 distinct (< count 3, has a duplicate)
    //   g=2: v in {5,15,25}  -> 3 distinct (== count 3, all distinct)
    //   g=3: v in {7}        -> 1 distinct (single value)
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    let rows: &[(i32, i32)] = &[(1, 10), (1, 10), (1, 20), (2, 5), (2, 15), (2, 25), (3, 7)];
    let values = rows
        .iter()
        .map(|(g, v)| format!("({g}, {v})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(DISTINCT v) grouped");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(3)],
            vec![SqlValue::Int4(3), SqlValue::Int8(1)],
        ],
        "per-group distinct count (duplicate / all-distinct / single)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_combined_with_count_star() {
    // SELECT g, COUNT(*), COUNT(DISTINCT v) FROM t GROUP BY g -- the multi-aggregate merge folds a
    // direct COUNT(*) pass and the sort-based COUNT(DISTINCT) pass by group key. count >= distinct.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    let rows: &[(i32, i32)] = &[(1, 10), (1, 10), (1, 20), (2, 5), (2, 15), (2, 25), (3, 7)];
    let values = rows
        .iter()
        .map(|(g, v)| format!("({g}, {v})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*), COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(*) + COUNT(DISTINCT v) grouped");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(3), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(3), SqlValue::Int8(3)],
            vec![SqlValue::Int4(3), SqlValue::Int8(1), SqlValue::Int8(1)],
        ],
        "COUNT(*) and COUNT(DISTINCT v) merged by group"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_with_sum_same_column() {
    // SELECT g, SUM(v), COUNT(DISTINCT v) FROM t GROUP BY g -- a DIRECT (SUM) pass AND a CountDistinct
    // pass over the SAME value column; the result builder must read the right pass for each.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    let rows: &[(i32, i32)] = &[(1, 10), (1, 10), (1, 20), (2, 5), (2, 15), (2, 25), (3, 7)];
    let values = rows
        .iter()
        .map(|(g, v)| format!("({g}, {v})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v), COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("SUM(v) + COUNT(DISTINCT v) over one column");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    // SUM by construction: g=1 -> 40, g=2 -> 45, g=3 -> 7. Distinct: 2 / 3 / 1.
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(40), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(45), SqlValue::Int8(3)],
            vec![SqlValue::Int4(3), SqlValue::Int8(7), SqlValue::Int8(1)],
        ],
        "SUM and COUNT(DISTINCT) over the same column read distinct passes"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_int8_negative_and_large() {
    // COUNT(DISTINCT v) over a BIGINT column spanning negatives + a value beyond int4 range.
    //   g=1: v in {-5, -5, 9000000000} -> 2 distinct
    //   g=2: v in {0}                  -> 1 distinct
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v BIGINT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g, v) VALUES (1, -5),(1, -5),(1, 9000000000),(2, 0)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(DISTINCT bigint) grouped");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(1)],
        ],
        "distinct count over int8 with negatives + beyond-int4 magnitude"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_numeric_value() {
    // COUNT(DISTINCT v) over a NUMERIC value -- the 16-byte i128 mantissa packs into the (g, v_hi,
    // v_lo) k=3 multikey sort. Distinct counts KNOWN BY CONSTRUCTION; g=4 proves SCALE NORMALIZATION
    // (8.4 and 8.40 rescale to the same column-scale mantissa 840 -> ONE distinct, PG-correct).
    //   g=1: {1.50, 1.50, 2.50} -> 2 distinct (a duplicate)
    //   g=2: {3.00, 4.00, 5.00} -> 3 distinct (all distinct)
    //   g=3: {7.25}             -> 1 distinct (single)
    //   g=4: {8.40, 8.4}        -> 1 distinct (equal numerics, different display scale)
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(10,2))")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g, v) VALUES \
         (1, 1.50),(1, 1.50),(1, 2.50),(2, 3.00),(2, 4.00),(2, 5.00),(3, 7.25),(4, 8.40),(4, 8.4)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(DISTINCT numeric) grouped");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(3)],
            vec![SqlValue::Int4(3), SqlValue::Int8(1)],
            vec![SqlValue::Int4(4), SqlValue::Int8(1)],
        ],
        "distinct numeric count, with display-scale-normalized equality"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_uuid_value() {
    // COUNT(DISTINCT v) over a UUID value -- 16 raw bytes packed into the (g, v_hi, v_lo) k=3 sort;
    // distinctness is byte-identity. Distinct counts KNOWN BY CONSTRUCTION (a, b, c, d, e are five
    // distinct uuids):
    //   g=1: {a, a, b} -> 2 distinct (a duplicated)
    //   g=2: {c}       -> 1 distinct
    //   g=3: {d, e, d} -> 2 distinct (d repeated non-adjacently before the sort)
    let a = "11111111-1111-1111-1111-111111111111";
    let b = "22222222-2222-2222-2222-222222222222";
    let c = "33333333-3333-3333-3333-333333333333";
    let d = "44444444-4444-4444-4444-444444444444";
    let f = "55555555-5555-5555-5555-555555555555";
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v UUID)").unwrap();
    e.execute_text(
        2,
        &format!(
            "INSERT INTO t (g, v) VALUES \
             (1, '{a}'),(1, '{a}'),(1, '{b}'),(2, '{c}'),(3, '{d}'),(3, '{f}'),(3, '{d}')"
        ),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(DISTINCT uuid) grouped");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(1)],
            vec![SqlValue::Int4(3), SqlValue::Int8(2)],
        ],
        "distinct uuid count by byte-identity"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_numeric_combined_with_count_star() {
    // SELECT g, COUNT(*), COUNT(DISTINCT v) over a NUMERIC value -- a direct COUNT(*) pass folded
    // with the k=3 sort-based COUNT(DISTINCT) pass. The multi-aggregate merge re-sorts each pass by
    // the MATERIALIZED group key, so the count and the distinct count align per group. count >= distinct.
    //   g=1: {1.50, 1.50, 2.50} -> count 3, distinct 2
    //   g=2: {3.00, 4.00, 5.00} -> count 3, distinct 3
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(10,2))")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g, v) VALUES (1, 1.50),(1, 1.50),(1, 2.50),(2, 3.00),(2, 4.00),(2, 5.00)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*), COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(*) + COUNT(DISTINCT numeric) grouped");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(3), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(3), SqlValue::Int8(3)],
        ],
        "COUNT(*) and COUNT(DISTINCT numeric) merged by group"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_text_value() {
    // COUNT(DISTINCT v) over a TEXT value -- varlen, so the (g, text_v) tuple GPU-sorts via the hetero
    // sort and the text-aware mark compares the value bytes. 7 rows (ODD -> the text offsets section is
    // 4-mod-8 after the single int4 column, exercising the 8-align pad). Distinct counts KNOWN BY
    // CONSTRUCTION; g=2 includes length-differing prefixes (the empty-string case in
    // `..._combined_with_count_star` is the robust guard for the byte-length check):
    //   g=1: {"apple", "apple", "banana"} -> 2 distinct (a duplicate)
    //   g=2: {"x", "xy", "xyz"}           -> 3 distinct (each a prefix of the next; lengths differ)
    //   g=3: {"hello"}                     -> 1 distinct (single)
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v TEXT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g, v) VALUES \
         (1, 'apple'),(1, 'apple'),(1, 'banana'),(2, 'x'),(2, 'xy'),(2, 'xyz'),(3, 'hello')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(DISTINCT text) grouped");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(3)],
            vec![SqlValue::Int4(3), SqlValue::Int8(1)],
        ],
        "distinct text count (duplicate / length-differing prefixes / single)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_text_combined_with_count_star() {
    // SELECT g, COUNT(*), COUNT(DISTINCT v) over a TEXT value -- a direct COUNT(*) pass folded with the
    // hetero-sort text COUNT(DISTINCT) pass, merged by the MATERIALIZED group key. g=1 includes the
    // EMPTY STRING (a valid distinct value, length 0 -> the byte loop runs zero iterations).
    //   g=1: {"", "", "z"}   -> count 3, distinct 2 (empty duplicated)
    //   g=2: {"foo", "bar"}  -> count 2, distinct 2
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v TEXT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g, v) VALUES (1, ''),(1, ''),(1, 'z'),(2, 'foo'),(2, 'bar')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*), COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(*) + COUNT(DISTINCT text) grouped");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(3), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(2), SqlValue::Int8(2)],
        ],
        "COUNT(*) and COUNT(DISTINCT text) merged by group, incl. the empty string"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_text_shared_value_across_groups() {
    // The SAME text value ("same") appears in three groups -- so in (g, text_v) sorted order the rows
    // (1,"same"),(1,"same"),(2,"same"),(3,"same") are ADJACENT with IDENTICAL text but changing g.
    // This makes the text mark's GROUP-KEY comparison load-bearing: if it ignored g and compared only
    // the text, g=2 and g=3 would collapse into g=1's run (distinct 0/1 instead of 1/1). Construction:
    //   g=1: {"same", "same"} -> 1 distinct
    //   g=2: {"same"}         -> 1 distinct (text equals g=1's, but a new group)
    //   g=3: {"same", "zzz"}  -> 2 distinct (shared "same" + a distinct value)
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v TEXT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g, v) VALUES (1, 'same'),(1, 'same'),(2, 'same'),(3, 'same'),(3, 'zzz')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(DISTINCT text) with a value shared across groups");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(1)],
            vec![SqlValue::Int4(2), SqlValue::Int8(1)],
            vec![SqlValue::Int4(3), SqlValue::Int8(2)],
        ],
        "the group-key compare splits identical text across group boundaries"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_scalar_count_distinct_int() {
    // Scalar COUNT(DISTINCT v) with NO GROUP BY -> one group (g=0). KNOWN BY CONSTRUCTION: v in
    // {10,10,20,20,20,30,30} -> 3 distinct values across the whole table.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (v) VALUES (10),(10),(20),(20),(20),(30),(30)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT COUNT(DISTINCT v) FROM t")
        .expect("scalar COUNT(DISTINCT int)");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(res.columns.len(), 1);
    assert!(res.columns[0].name.eq_ignore_ascii_case("count"));
    assert_eq!(
        res.rows,
        vec![vec![SqlValue::Int8(3)]],
        "total distinct values across the table"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_scalar_count_distinct_text_numeric_and_filtered() {
    // Scalar COUNT(DISTINCT) over a TEXT value, a NUMERIC value, and an int value WITH a WHERE filter
    // (so the surviving indices are not the full scan), plus an empty-result case (PG -> 0, not NULL).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (k INT, v INT, s TEXT, n NUMERIC(10,2))")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (k, v, s, n) VALUES \
         (1, 5, 'a', 1.50),(1, 5, 'a', 1.50),(1, 7, 'b', 2.50),(2, 9, 'a', 1.50),(2, 9, 'c', 3.00)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // TEXT distinct over the whole table: {"a","a","b","a","c"} -> 3 distinct.
    let text = e
        .execute_resident_expr_select_sql("SELECT COUNT(DISTINCT s) FROM t")
        .expect("scalar COUNT(DISTINCT text)");
    assert_eq!(text.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(text.rows, vec![vec![SqlValue::Int8(3)]], "distinct text");
    // NUMERIC distinct: {1.50,1.50,2.50,1.50,3.00} -> 3 distinct.
    let num = e
        .execute_resident_expr_select_sql("SELECT COUNT(DISTINCT n) FROM t")
        .expect("scalar COUNT(DISTINCT numeric)");
    assert_eq!(num.rows, vec![vec![SqlValue::Int8(3)]], "distinct numeric");
    // WHERE k = 1 -> v in {5,5,7} -> 2 distinct (the filter narrows the surviving rows).
    let filtered = e
        .execute_resident_expr_select_sql("SELECT COUNT(DISTINCT v) FROM t WHERE k = 1")
        .expect("scalar COUNT(DISTINCT int) with WHERE");
    assert_eq!(
        filtered.rows,
        vec![vec![SqlValue::Int8(2)]],
        "distinct over the filtered survivors"
    );
    // WHERE matches nothing -> COUNT(DISTINCT) is 0 (not NULL).
    let empty = e
        .execute_resident_expr_select_sql("SELECT COUNT(DISTINCT v) FROM t WHERE k = 99")
        .expect("scalar COUNT(DISTINCT) over empty");
    assert_eq!(
        empty.rows,
        vec![vec![SqlValue::Int8(0)]],
        "distinct over an empty set is 0"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_text_group_key() {
    // COUNT(DISTINCT v) over a TEXT group key (the GROUP-BY-(g,v) reduction: distinct (cat, uid) pairs
    // per cat). cat=a: uid in {1,1,2} -> 2 distinct; cat=b: {5,5} -> 1 distinct.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (cat TEXT, uid INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (cat, uid) VALUES ('a',1),('a',1),('a',2),('b',5),('b',5)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT cat, COUNT(DISTINCT uid) FROM t GROUP BY cat")
        .expect("COUNT(DISTINCT) over a text group key");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Text("a".into()), SqlValue::Int8(2)],
            vec![SqlValue::Text("b".into()), SqlValue::Int8(1)],
        ],
        "distinct uid per text category"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_text_group_and_text_value() {
    // COUNT(DISTINCT v) where BOTH the group key AND the value are TEXT -> the (g, v) reduction's step 1
    // is a two-text composite. cat=a: tag in {x,x,y} -> 2; cat=b: {z} -> 1.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (cat TEXT, tag TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (cat, tag) VALUES ('a','x'),('a','x'),('a','y'),('b','z')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT cat, COUNT(DISTINCT tag) FROM t GROUP BY cat")
        .expect("COUNT(DISTINCT text) over a text group key");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Text("a".into()), SqlValue::Int8(2)],
            vec![SqlValue::Text("b".into()), SqlValue::Int8(1)],
        ],
        "distinct text tag per text category"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_text_group_combined_with_count_star() {
    // A TEXT group key (not composite) supports COUNT(*) (direct pass) + COUNT(DISTINCT) (reduction)
    // merged by the group key. cat=a: count 3, distinct{1,2}=2; cat=b: count 1, distinct{5}=1.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (cat TEXT, uid INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (cat, uid) VALUES ('a',1),('a',1),('a',2),('b',5)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT cat, COUNT(*), COUNT(DISTINCT uid) FROM t GROUP BY cat",
        )
        .expect("text group key COUNT(*) + COUNT(DISTINCT)");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![
                SqlValue::Text("a".into()),
                SqlValue::Int8(3),
                SqlValue::Int8(2)
            ],
            vec![
                SqlValue::Text("b".into()),
                SqlValue::Int8(1),
                SqlValue::Int8(1)
            ],
        ],
        "count >= distinct, aligned by the text group key"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_numeric_group_key() {
    // COUNT(DISTINCT v) over a NUMERIC group key (i128 key; the (g,v) reduction's step 1 is a
    // (numeric, int) wide-key). g=1.50: v in {5,5,7} -> 2; g=2.50: {9} -> 1.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g NUMERIC(10,2), v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g, v) VALUES (1.50,5),(1.50,5),(1.50,7),(2.50,9)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(DISTINCT) over a numeric group key");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let num = |m: i128| SqlValue::Numeric(Decimal128::new(m, 2));
    assert_eq!(
        res.rows,
        vec![
            vec![num(150), SqlValue::Int8(2)],
            vec![num(250), SqlValue::Int8(1)],
        ],
        "distinct v per numeric group"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_composite_group_key() {
    // COUNT(DISTINCT v) over a COMPOSITE (int, int) group key (single aggregate). step 1 = (a,b,v)
    // wide-key; step 2 = (a,b) i64-pack over the reps. (1,1): v{5,5,7}->2; (1,2): {9}->1; (2,1): {9}->1.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a,b,v) VALUES (1,1,5),(1,1,5),(1,1,7),(1,2,9),(2,1,9)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(DISTINCT v) FROM t GROUP BY a, b")
        .expect("COUNT(DISTINCT) over a composite group key");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(1), SqlValue::Int4(2), SqlValue::Int8(1)],
            vec![SqlValue::Int4(2), SqlValue::Int4(1), SqlValue::Int8(1)],
        ],
        "distinct v per (a,b) composite group"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_text_group_key_empty() {
    // A WHERE that drops every row -> no groups (the reduction handles empty survivors / empty reps).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (cat TEXT, uid INT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO t (cat, uid) VALUES ('a',1),('b',2)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT cat, COUNT(DISTINCT uid) FROM t WHERE uid > 100 GROUP BY cat",
        )
        .expect("COUNT(DISTINCT) text group key, empty survivors");
    assert!(res.rows.is_empty(), "no surviving rows -> no groups");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_numeric_value_text_group_shared() {
    // COUNT(DISTINCT numeric_value) over a TEXT group key, with a value SHARED across groups: 1.50
    // appears under cat=a AND cat=b -> it counts once PER group. a: {1.50,2.50}=2; b: {1.50}=1.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (cat TEXT, n NUMERIC(10,2))")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (cat, n) VALUES ('a',1.50),('a',1.50),('a',2.50),('b',1.50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT cat, COUNT(DISTINCT n) FROM t GROUP BY cat")
        .expect("COUNT(DISTINCT numeric) over a text group key");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Text("a".into()), SqlValue::Int8(2)],
            vec![SqlValue::Text("b".into()), SqlValue::Int8(1)],
        ],
        "distinct numeric value per text group; a shared value is counted once per group"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_expr_group_key() {
    // COUNT(DISTINCT v) over an EXPRESSION group key (a+b, int4) -> the (g,v) reduction with the expr's
    // DERIVED buffer as the wide-key's kind-4 (i32) member; step 2 reuses the expr key_base_override.
    // v=5 and v=9 are SHARED across groups (so distinct-per-group != global distinct -> the derived
    // member is load-bearing). a+b=2: v{5,5,7,9}->3; a+b=3: {9,5}->2; a+b=0: {3}->1. Order: 0,2,3.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a,b,v) VALUES (1,1,5),(1,1,5),(1,1,7),(1,1,9),(3,0,9),(3,0,5),(0,0,3)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT a + b, COUNT(DISTINCT v) FROM t GROUP BY a + b")
        .expect("COUNT(DISTINCT v) over an expression group key");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int4(0), SqlValue::Int8(1)],
            vec![SqlValue::Int4(2), SqlValue::Int8(3)],
            vec![SqlValue::Int4(3), SqlValue::Int8(2)],
        ],
        "distinct v per (a+b) expression group (v shared across groups)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_expr_group_key_int8() {
    // COUNT(DISTINCT v) over a PURE INT8 EXPRESSION group key (a+c, both BIGINT) -> the derived buffer
    // is i64, so the wide key uses kind 5 (i64 derived) and step 2 reuses the int8 expr config. The
    // expr value is beyond the int4 range; v=5 and v=9 are SHARED across the two groups (derived member
    // load-bearing). a+c=10000000000: v{5,5,7,9}->3; a+c=5: {9,5}->2.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, c BIGINT, v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a,c,v) VALUES (10000000000,0,5),(10000000000,0,5),(10000000000,0,7),\
         (10000000000,0,9),(5,0,9),(5,0,5)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT a + c, COUNT(DISTINCT v) FROM t GROUP BY a + c")
        .expect("COUNT(DISTINCT v) over a pure int8 expression group key");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Int8(5), SqlValue::Int8(2)],
            vec![SqlValue::Int8(10000000000), SqlValue::Int8(3)],
        ],
        "distinct v per (a+c) int8 expression group, value beyond int4 range, v shared across groups"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_mixed_int_width_expr_key_rejected() {
    // GROUP BY a MIXED int4/int8 arithmetic expression (a BIGINT + b INT) is rejected -- the arith VM is
    // mono-typed, so a mixed expr would load the int4 column at the wrong stride (garbage). An honest
    // error, not a wrong answer (pre-existing latent bug; surfaced + guarded). Covers the plain GROUP BY
    // (no CD) AND the COUNT(DISTINCT) reduction (which reuses this expr key buffer).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, b INT, v INT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO t (a,b,v) VALUES (10000000000,1,5),(5,2,9)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    for sql in [
        "SELECT a + b, COUNT(*) FROM t GROUP BY a + b",
        "SELECT a + b, COUNT(DISTINCT v) FROM t GROUP BY a + b",
    ] {
        let err = e
            .execute_resident_expr_select_sql(sql)
            .expect_err("mixed int4/int8 expression GROUP BY rejected");
        assert!(
            format!("{err:?}")
                .to_lowercase()
                .contains("mixed int4/int8"),
            "clean reject for {sql}, got: {err:?}"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_distinct_bool_group_key() {
    // COUNT(DISTINCT v) over a BOOL group key: the (bool, v) reduction (step 1 uses build kind 3 for the
    // bool member; step 2 reuses the bool->int4 key buffer). flag=true: v{1,1,2}->2; flag=false: {5}->1.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (flag BOOL, v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (flag, v) VALUES (true,1),(true,1),(true,2),(false,5)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT flag, COUNT(DISTINCT v) FROM t GROUP BY flag")
        .expect("COUNT(DISTINCT) over a bool group key");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![SqlValue::Bool(false), SqlValue::Int8(1)],
            vec![SqlValue::Bool(true), SqlValue::Int8(2)],
        ],
        "distinct v per bool group (order by flag: false < true)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_multiple_aggregates_different_value_columns() {
    // SELECT g, SUM(v), MIN(w), MAX(w) FROM t GROUP BY g -- aggregates over TWO different value columns
    // (v int4, w int8) -> two grouping passes (single-level forced) merged by group index.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT, w BIGINT)")
        .unwrap();
    let rows: &[(i32, i32, i64)] = &[
        (1, 10, 100),
        (1, 20, 50),
        (1, 30, 200),
        (2, 5, 1000),
        (2, 15, 999),
    ];
    let values = rows
        .iter()
        .map(|(g, v, w)| format!("({g}, {v}, {w})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, v, w) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v), MIN(w), MAX(w) FROM t GROUP BY g")
        .expect("aggregates over two different value columns");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    // [g, SUM(v int4)->int8, MIN(w int8)->int8, MAX(w int8)->int8]
    assert_eq!(
        res.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Int8(60),
                SqlValue::Int8(50),
                SqlValue::Int8(200),
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int8(20),
                SqlValue::Int8(999),
                SqlValue::Int8(1000),
            ],
        ],
        "two-pass merge: SUM(v) + MIN(w)/MAX(w) over distinct value columns"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_count_with_text_value_min() {
    // SELECT g, COUNT(*), MIN(s) FROM t GROUP BY g -- COUNT alongside a TEXT-value MIN (the text-value
    // pass yields both the group count and the lexicographic-min winner's row index).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, s TEXT)").unwrap();
    let rows: &[(i32, &str)] = &[
        (1, "banana"),
        (1, "apple"),
        (1, "cherry"),
        (2, "zebra"),
        (2, "ant"),
    ];
    let values = rows
        .iter()
        .map(|(g, s)| format!("({g}, '{s}')"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, s) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*), MIN(s) FROM t GROUP BY g")
        .expect("COUNT(*) + MIN(text)");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Int8(3),
                SqlValue::Text("apple".to_string()),
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int8(2),
                SqlValue::Text("ant".to_string()),
            ],
        ],
        "COUNT(*) + MIN(text value) in one grouped query"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_text_key_multiple_value_columns() {
    // SELECT k, SUM(v), MIN(w) FROM t GROUP BY k -- a TEXT key with TWO value columns (two passes).
    // The cross-pass merge is by group INDEX; for a text key each pass's representative row index can
    // differ (parallel claim race), but the slot assignment (hence compaction order) is deterministic
    // for the same texts, so the i-th group of each pass is the same key. This pins that invariant.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (k TEXT, v INT, w BIGINT)")
        .unwrap();
    let rows: &[(&str, i32, i64)] = &[
        ("apple", 10, 100),
        ("apple", 20, 50),
        ("banana", 5, 999),
        ("cherry", 7, 7),
        ("apple", 1, 200),
    ];
    let values = rows
        .iter()
        .map(|(k, v, w)| format!("('{k}', {v}, {w})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (k, v, w) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT k, SUM(v), MIN(w) FROM t GROUP BY k")
        .expect("text key + two value-column aggregates");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            vec![
                SqlValue::Text("apple".to_string()),
                SqlValue::Int8(31),
                SqlValue::Int8(50),
            ],
            vec![
                SqlValue::Text("banana".to_string()),
                SqlValue::Int8(5),
                SqlValue::Int8(999),
            ],
            vec![
                SqlValue::Text("cherry".to_string()),
                SqlValue::Int8(7),
                SqlValue::Int8(7),
            ],
        ],
        "text key + SUM(v)/MIN(w) merged across two passes by group index"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_multi_aggregate_cross_pass_merge_alignment() {
    // Regression: with >=2 distinct value columns the executor runs one grouping pass PER COLUMN, and
    // the kernel's cas.b64 linear-probe slot order is RACE-dependent across launches -- so passes must
    // be aligned by the MATERIALIZED group key (a sort), NOT by slot index. With many groups (hash
    // collisions guaranteed) a slot-index merge silently misattributes aggregates. The bug surfaced in
    // audit only on the 6th launch of an int8 key, so run the query many times to defeat the race.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT, w BIGINT)")
        .unwrap();
    let n: i32 = 130;
    let mut tuples = Vec::new();
    for g in 1..=n {
        tuples.push(format!("({g}, {g}, {})", 10000 - g));
        tuples.push(format!("({g}, {}, {})", g + 1000, 20000 + g));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (g, v, w) VALUES {}", tuples.join(",")),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // Host oracle per group g: COUNT=2, SUM(v)=2g+1000, MIN(w)=10000-g, MAX(w)=20000+g; sorted by g.
    let expected: Vec<Vec<SqlValue>> = (1..=n)
        .map(|g| {
            vec![
                SqlValue::Int4(g),
                SqlValue::Int8(2),
                SqlValue::Int8((2 * g + 1000) as i64),
                SqlValue::Int8((10000 - g) as i64),
                SqlValue::Int8((20000 + g) as i64),
            ]
        })
        .collect();
    // v (int4) + w (int8) = TWO distinct value columns -> two passes; repeat to defeat the race.
    for trial in 0..25 {
        let res = e
            .execute_resident_expr_select_sql(
                "SELECT g, COUNT(*), SUM(v), MIN(w), MAX(w) FROM t GROUP BY g",
            )
            .expect("multi-aggregate cross-pass query");
        assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(
            res.rows, expected,
            "cross-pass merge misaligned on trial {trial}"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_multi_aggregate_text_key_merge_alignment() {
    // The same race, but a TEXT key: the per-pass merge must sort by the materialized STRING (a text
    // group's key_i128 is a per-pass representative row index, which differs across passes), so the
    // index-merge would misalign without the key-sort. Many groups + repeated launches.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (k TEXT, v INT, w BIGINT)")
        .unwrap();
    let n: i32 = 110;
    let mut tuples = Vec::new();
    for i in 0..n {
        tuples.push(format!("('grp_{i:04}', {i}, {})", 50000 - i));
        tuples.push(format!("('grp_{i:04}', {}, {})", i + 2000, 60000 + i));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (k, v, w) VALUES {}", tuples.join(",")),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // grp_0000..grp_0109 sort lexicographically == numerically; group i: SUM(v)=2i+2000,
    // MIN(w)=50000-i, MAX(w)=60000+i.
    let expected: Vec<Vec<SqlValue>> = (0..n)
        .map(|i| {
            vec![
                SqlValue::Text(format!("grp_{i:04}")),
                SqlValue::Int8((2 * i + 2000) as i64),
                SqlValue::Int8((50000 - i) as i64),
                SqlValue::Int8((60000 + i) as i64),
            ]
        })
        .collect();
    for trial in 0..25 {
        let res = e
            .execute_resident_expr_select_sql("SELECT k, SUM(v), MIN(w), MAX(w) FROM t GROUP BY k")
            .expect("text-key multi-aggregate cross-pass query");
        assert_eq!(
            res.rows, expected,
            "text-key cross-pass merge misaligned on trial {trial}"
        );
    }
}

// group counts for `t` below: g1=3, g2=1, g3=2, g4=4, g5=1.
const GROUPED_CLAUSE_ROWS: &str =
    "(1,10),(1,20),(1,30),(2,5),(3,7),(3,8),(4,1),(4,2),(4,3),(4,4),(5,99)";

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_order_by_and_limit() {
    // ORDER BY (the key DESC, and an AGGREGATE DESC) + LIMIT/OFFSET windowed ON-DEVICE (a slice of the
    // gpu_sort_permutation index vector, gathering only the kept window) on the Expr path.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        &format!("INSERT INTO t (g, v) VALUES {GROUPED_CLAUSE_ROWS}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let r = |g: i32, c: i64| vec![SqlValue::Int4(g), SqlValue::Int8(c)];

    let a = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g DESC")
        .unwrap();
    assert_eq!(a.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        a.rows,
        vec![r(5, 1), r(4, 4), r(3, 2), r(2, 1), r(1, 3)],
        "ORDER BY the group key DESC"
    );

    let b = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY COUNT(*) DESC LIMIT 3",
        )
        .unwrap();
    assert_eq!(
        b.rows,
        vec![r(4, 4), r(1, 3), r(3, 2)],
        "ORDER BY COUNT(*) DESC LIMIT 3 (top 3 by count)"
    );

    let d = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g LIMIT 2 OFFSET 1")
        .unwrap();
    assert_eq!(
        d.rows,
        vec![r(2, 1), r(3, 2)],
        "LIMIT 2 OFFSET 1 over the default key order (skip g1, take g2,g3)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_limit_offset_window_edges() {
    // S4: OFFSET/LIMIT on the grouped path is now a control-plane WINDOW of the on-device sort
    // permutation (gpu_sort_permutation), gathering only the kept window -- no host drain/truncate.
    // These edge cases pin the windowing math against the prior drain/truncate: OFFSET past the end,
    // LIMIT 0, and OFFSET+LIMIT running past the end (clamped), plus a no-LIMIT default-order sanity.
    // group counts (GROUPED_CLAUSE_ROWS): g1=3, g2=1, g3=2, g4=4, g5=1 -> default key order is g ASC.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        &format!("INSERT INTO t (g, v) VALUES {GROUPED_CLAUSE_ROWS}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let r = |g: i32, c: i64| vec![SqlValue::Int4(g), SqlValue::Int8(c)];

    // OFFSET past the end -> empty (start clamps to len; nothing gathered).
    let beyond = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g OFFSET 10",
        )
        .unwrap();
    assert_eq!(beyond.executed_target, DeviceTarget::Gpu(0));
    assert!(beyond.rows.is_empty(), "OFFSET past the end -> no rows");

    // LIMIT 0 -> empty (window end == start).
    let zero = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g LIMIT 0")
        .unwrap();
    assert!(zero.rows.is_empty(), "LIMIT 0 -> no rows");

    // OFFSET 3 + LIMIT 100 running past the end -> clamped to the remaining tail [g4, g5].
    let tail = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g LIMIT 100 OFFSET 3",
        )
        .unwrap();
    assert_eq!(
        tail.rows,
        vec![r(4, 4), r(5, 1)],
        "LIMIT past the end clamps to the tail"
    );

    // No LIMIT, default order: the window is the full range -> identical to the prior reorder.
    let full = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g")
        .unwrap();
    assert_eq!(
        full.rows,
        vec![r(1, 3), r(2, 1), r(3, 2), r(4, 4), r(5, 1)],
        "no LIMIT -> full default-order result unchanged"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_having_and_combined() {
    // HAVING filters groups by an aggregate (or key) predicate; combined HAVING + ORDER BY + LIMIT.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        &format!("INSERT INTO t (g, v) VALUES {GROUPED_CLAUSE_ROWS}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let r = |g: i32, c: i64| vec![SqlValue::Int4(g), SqlValue::Int8(c)];

    let c = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g HAVING COUNT(*) > 2",
        )
        .unwrap();
    assert_eq!(c.rows, vec![r(1, 3), r(4, 4)], "HAVING COUNT(*) > 2");

    let comb = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g HAVING COUNT(*) >= 2 ORDER BY COUNT(*) DESC LIMIT 2",
        )
        .unwrap();
    assert_eq!(
        comb.rows,
        vec![r(4, 4), r(1, 3)],
        "HAVING >= 2 then ORDER BY COUNT(*) DESC then LIMIT 2"
    );

    let k = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g HAVING g >= 4")
        .unwrap();
    assert_eq!(
        k.rows,
        vec![r(4, 4), r(5, 1)],
        "HAVING on the group key column"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_having_sum_and_dnf_runs_on_gpu() {
    // Regression coverage (audit-found): a HAVING over an int8 SUM(int4) result and a DNF mixing an int4
    // group key with an int8 COUNT must RUN on the GPU, not error.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g, v) VALUES \
         (1,10),(1,20),(1,30), (2,15), (3,5),(3,10), (4,20),(4,30),(4,40),(4,9), (5,99)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // groups (default order by g): g1 count3 sum60, g2 count1 sum15, g3 count2 sum15, g4 count4 sum99,
    // g5 count1 sum99.
    let rv = |g: i32, x: i64| vec![SqlValue::Int4(g), SqlValue::Int8(x)];

    // HAVING over a SUM(int4) result (the P0 the first attempt regressed) -> g1, g4, g5.
    let a = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g HAVING SUM(v) > 15")
        .unwrap();
    assert_eq!(
        a.rows,
        vec![rv(1, 60), rv(4, 99), rv(5, 99)],
        "HAVING SUM(int4) > 15"
    );

    // DNF mixing an int4 key AND an int8 COUNT (the mixed-width case) -> g3, g4.
    let b = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g HAVING g >= 2 AND COUNT(*) > 1",
        )
        .unwrap();
    assert_eq!(
        b.rows,
        vec![rv(3, 2), rv(4, 4)],
        "HAVING int4-key AND int8-count"
    );

    // DNF mixing an int4 key OR an int8 COUNT -> g1, g4.
    let c = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g HAVING g = 1 OR COUNT(*) >= 3",
        )
        .unwrap();
    assert_eq!(
        c.rows,
        vec![rv(1, 3), rv(4, 4)],
        "HAVING int4-key OR int8-count"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_having_numeric_int_mixed_dnf_runs_on_gpu() {
    // Regression coverage (2nd audit): a HAVING DNF mixing a NUMERIC aggregate with an integer COUNT must
    // RUN on the GPU -- the integers are promoted to Numeric so the predicate is a single i128 width --
    // rather than clean-error.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (k INT, n NUMERIC(10,2))")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (k, n) VALUES (1,2.00),(1,2.00), (2,10.00), (3,1.00),(3,1.00),(3,1.00)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // groups (default order by k): k1 sum 4.00 count 2; k2 sum 10.00 count 1; k3 sum 3.00 count 3.
    let row = |k: i32, sum_cents: i128, c: i64| {
        vec![
            SqlValue::Int4(k),
            SqlValue::Numeric(Decimal128::new(sum_cents, 2)),
            SqlValue::Int8(c),
        ]
    };
    // numeric SUM AND integer COUNT -> k1 only.
    let a = e
        .execute_resident_expr_select_sql(
            "SELECT k, SUM(n), COUNT(*) FROM t GROUP BY k HAVING SUM(n) > 3.00 AND COUNT(*) >= 2",
        )
        .unwrap();
    assert_eq!(a.rows, vec![row(1, 400, 2)], "numeric SUM AND int COUNT");
    // numeric SUM OR integer COUNT -> k2, k3.
    let b = e
        .execute_resident_expr_select_sql(
            "SELECT k, SUM(n), COUNT(*) FROM t GROUP BY k HAVING SUM(n) > 8.00 OR COUNT(*) >= 3",
        )
        .unwrap();
    assert_eq!(
        b.rows,
        vec![row(2, 1000, 1), row(3, 300, 3)],
        "numeric SUM OR int COUNT"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_having_avg_heterogeneous_scale_on_gpu() {
    // Regression coverage (3rd audit, a SILENT WRONG ANSWER): AVG yields per-GROUP Numeric scales (PG
    // division). The HAVING transient must normalize each value to the column's (max) scale, else a
    // low-AVG group's mantissa is misread at a smaller scale as a huge number and wrongly KEPT.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (g, v) VALUES (1,3), (2,10), (3,1)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // AVG: g1=3.0, g2=10.0, g3=1.0, each at PG's per-group division scale. Assert the surviving KEYS.
    let keys = |sql: &str| -> Vec<SqlValue> {
        e.execute_resident_expr_select_sql(sql)
            .unwrap()
            .rows
            .iter()
            .map(|r| r[0].clone())
            .collect()
    };
    assert_eq!(
        keys("SELECT g, AVG(v) FROM t GROUP BY g HAVING AVG(v) > 2.00"),
        vec![SqlValue::Int4(1), SqlValue::Int4(2)],
        "HAVING AVG(v) > 2.00 must DROP g3 (1.0), not misread its scale as huge"
    );
    assert_eq!(
        keys("SELECT g, AVG(v) FROM t GROUP BY g HAVING AVG(v) < 5"),
        vec![SqlValue::Int4(1), SqlValue::Int4(3)],
        "HAVING AVG(v) < 5 keeps g1(3.0), g3(1.0)"
    );
    // 4th-audit case: a HIGH-SCALE numeric (AVG, scale ~20) leaf inside an AND/OR DNF must use the i128
    // comparison, not the i32 `CompareScalar` fast path (whose rescaled literal overflowed i32).
    assert_eq!(
        keys("SELECT g, AVG(v), COUNT(*) FROM t GROUP BY g HAVING AVG(v) > 2.00 AND COUNT(*) >= 1"),
        vec![SqlValue::Int4(1), SqlValue::Int4(2)],
        "HAVING high-scale AVG AND int COUNT in a DNF"
    );
    assert_eq!(
        keys("SELECT g, AVG(v), COUNT(*) FROM t GROUP BY g HAVING AVG(v) > 5.00 OR COUNT(*) >= 99"),
        vec![SqlValue::Int4(2)],
        "HAVING high-scale AVG OR int COUNT in a DNF"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_duplicate_aggregate_name_is_ambiguous() {
    // Two same-function aggregates share a result-column name ("sum"); referencing it in ORDER BY or
    // HAVING is ambiguous (PG: "column reference ... is ambiguous") -> error, not silent first-match.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT, w INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g, v, w) VALUES (1, 3, 100), (1, 4, 200), (2, 50, 1)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    assert!(
        e.execute_resident_expr_select_sql(
            "SELECT g, SUM(v), SUM(w) FROM t GROUP BY g ORDER BY sum"
        )
        .is_err(),
        "ORDER BY an ambiguous aggregate name must error"
    );
    assert!(
        e.execute_resident_expr_select_sql(
            "SELECT g, SUM(v), SUM(w) FROM t GROUP BY g HAVING sum > 5"
        )
        .is_err(),
        "HAVING an ambiguous aggregate name must error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_expression() {
    // ORDER BY an EXPRESSION (`a+b`, `a*2`) on the general GPU path: the device Expr interpreter
    // evaluates it into an i64 key column feeding the GPU bitonic sort -- single key, multi-key
    // (expr + column), expr + a text key (hetero), WHERE, LIMIT. executed_target==Gpu throughout.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c INT, name TEXT)")
        .unwrap();
    let rows: &[(i32, i32, i32, &str)] = &[
        (5, 1, 1, "bob"),  // a+b=6
        (2, 9, 2, "amy"),  // a+b=11
        (8, 0, 3, "cara"), // a+b=8
        (1, 1, 4, "dan"),  // a+b=2
        (3, 5, 5, "amy"),  // a+b=8 (ties c=3 on the sum; "amy" ties c=2 on the name)
        (4, 3, 6, "bob"),  // a+b=7 ("bob" ties c=1 on the name)
    ];
    let values = rows
        .iter()
        .map(|(a, b, c, n)| format!("({a}, {b}, {c}, '{n}')"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (a, b, c, name) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = |res: &RelationalSelectResult, col: usize| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[col] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };

    // (a) ORDER BY a+b ASC -- the sum ties (c=3,c=5 both 8; bitonic is unstable), so assert the SUM
    // sequence (computed from the projected a,b) is monotonic, not the exact rows.
    let s = e
        .execute_relational_select_text("SELECT a, b FROM t ORDER BY a + b")
        .unwrap();
    assert_eq!(
        s.executed_target,
        DeviceTarget::Gpu(0),
        "ORDER BY a+b on GPU"
    );
    let sums: Vec<i32> = i4(&s, 0)
        .iter()
        .zip(i4(&s, 1))
        .map(|(a, b)| a + b)
        .collect();
    assert_eq!(sums, vec![2, 6, 7, 8, 8, 11], "ORDER BY a+b ASC monotonic");

    // (b) ORDER BY a*2 DESC -- monotonic in a, no ties -> exact (c identifies rows).
    let s = e
        .execute_relational_select_text("SELECT c FROM t ORDER BY a * 2 DESC")
        .unwrap();
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(i4(&s, 0), vec![3, 1, 6, 5, 2, 4], "ORDER BY a*2 DESC");

    // (c) multi-key expr-primary: ORDER BY a+b, c -- the (sum,c) tuple is distinct -> deterministic.
    let s = e
        .execute_relational_select_text("SELECT c FROM t ORDER BY a + b, c")
        .unwrap();
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(i4(&s, 0), vec![4, 1, 6, 3, 5, 2], "ORDER BY a+b, c");

    // (d) hetero (text + expr): ORDER BY name, a+b -- (name,sum) distinct -> deterministic.
    let s = e
        .execute_relational_select_text("SELECT c FROM t ORDER BY name, a + b")
        .unwrap();
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        i4(&s, 0),
        vec![5, 2, 1, 6, 3, 4],
        "ORDER BY name, a+b (hetero)"
    );

    // (e) WHERE + expr ORDER BY + LIMIT. a>2: c1(sum6),c3(sum8),c5(sum8),c6(sum7) -> sorted 6,7,8,8;
    // LIMIT 2 -> c1, c6.
    let s = e
        .execute_relational_select_text("SELECT c FROM t WHERE a > 2 ORDER BY a + b LIMIT 2")
        .unwrap();
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(i4(&s, 0), vec![1, 6], "WHERE a>2 ORDER BY a+b LIMIT 2");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_int8_expression_sorts_at_i64_width() {
    // ORDER BY a BIGINT expression must read the arith value buffer at i64 width. Reading it as i32
    // (the pre-fix bug) would stride the 8-byte BIGINT column by 4 bytes -> garbage keys. A value
    // beyond i32::MAX also exercises the i64 range (an i32 read could not even represent it).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, b BIGINT, id INT)")
        .unwrap();
    // a+b: id1->15, id2->2, id3->5000000001 (> i32::MAX), id4->7. asc by a+b: 2,7,15,5e9 -> ids 2,4,1,3.
    e.execute_text(
        2,
        "INSERT INTO t (a, b, id) VALUES (10,5,1),(1,1,2),(5000000000,1,3),(3,4,4)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let s = e
        .execute_relational_select_text("SELECT id FROM t ORDER BY a + b")
        .expect("int8 expression ORDER BY runs on the GPU");
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        s.rows,
        vec![
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(4)],
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(3)],
        ],
        "BIGINT a+b sorted at i64 width"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_numeric_b128_width() {
    // ORDER BY a NUMERIC (i128) column on the GPU: the 16-byte comparator (signed HIGH limb, unsigned
    // LOW limb). Mixed-sign values exercise both limbs -- the signed hi distinguishes sign (negatives
    // hi=-1 below positives hi=0); the unsigned lo decides within a sign (two's-complement low bits).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (v NUMERIC(10,2), label INT)")
        .unwrap();
    // labels = the sorted rank (inserted shuffled): -20 < -10 < 0 < 5 < 10 < 20. 6 rows -> npot 8.
    e.execute_text(
        2,
        "INSERT INTO t (v, label) VALUES \
         (20.00, 5), (-10.00, 1), (5.00, 3), (-20.00, 0), (10.00, 4), (0.00, 2)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let asc = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY v")
        .expect("numeric ORDER BY on the GPU");
    assert_eq!(asc.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        asc.rows,
        (0..6).map(|i| vec![SqlValue::Int4(i)]).collect::<Vec<_>>(),
        "numeric ASC (signed hi, unsigned lo)"
    );
    let desc = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY v DESC")
        .expect("numeric DESC on the GPU");
    assert_eq!(
        desc.rows,
        (0..6)
            .rev()
            .map(|i| vec![SqlValue::Int4(i)])
            .collect::<Vec<_>>(),
        "numeric DESC"
    );
    // WHERE v>0 -> ranks 3,4,5 ; LIMIT 2 -> 3,4.
    let win = e
        .execute_relational_select_text("SELECT label FROM t WHERE v > 0 ORDER BY v LIMIT 2")
        .expect("numeric WHERE+LIMIT on the GPU");
    assert_eq!(
        win.rows,
        vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(4)]],
        "v>0 asc limit2 -> ranks 3,4"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_b128_secondary_key_dispatch() {
    // A b128 (numeric) key as a tie-broken SECONDARY: the int primary `grp` ties, so the numeric `v`
    // decides. Guards the 2-bit key_plan dispatch for a b128 key that is NOT the primary -- a
    // `kind >> 31` (instead of >> 30) bug would misdispatch the secondary numeric onto the text leg
    // and misorder within each group. (The other b128 multi-key test uses a b128 PRIMARY.)
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (grp INT, v NUMERIC(10,2), label INT)")
        .unwrap();
    // ORDER BY grp ASC, v ASC: grp=1 {v=10,20,30 -> 0,1,2}, grp=2 {v=5,15,25 -> 3,4,5}. label = rank.
    e.execute_text(
        2,
        "INSERT INTO t (grp, v, label) VALUES \
         (1, 30.00, 2), (1, 10.00, 0), (1, 20.00, 1), (2, 15.00, 4), (2, 5.00, 3), (2, 25.00, 5)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let s = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY grp ASC, v ASC")
        .expect("int-primary + numeric-secondary ORDER BY on the GPU");
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        s.rows,
        (0..6).map(|i| vec![SqlValue::Int4(i)]).collect::<Vec<_>>(),
        "numeric as a tie-broken SECONDARY key dispatches correctly"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_uuid_big_endian_unsigned() {
    // ORDER BY a UUID column: 16 raw bytes, UNSIGNED BIG-ENDIAN (byte 0 most significant). Bytes >= 0x80
    // sort ABOVE 0x7f (unsigned). byte-15 breaks a byte-0 tie. 7 rows -> npot 8 exercises padding.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (id UUID, label INT)")
        .unwrap();
    let uuid_b0 = |b: u32| format!("{b:02x}000000-0000-0000-0000-000000000000");
    // sorted order: 00/00, 00/ff, 10, 40, 7f, 80, ff -> labels = rank, inserted shuffled.
    let rows: [(String, i32); 7] = [
        (uuid_b0(0xff), 6),
        (uuid_b0(0x00), 0),
        (uuid_b0(0x80), 5),
        (uuid_b0(0x10), 2),
        ("00000000-0000-0000-0000-0000000000ff".to_string(), 1),
        (uuid_b0(0x7f), 4),
        (uuid_b0(0x40), 3),
    ];
    let mut values = String::new();
    for (i, (uuid, label)) in rows.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("('{uuid}', {label})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (id, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let asc = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY id")
        .expect("uuid ORDER BY on the GPU");
    assert_eq!(asc.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        asc.rows,
        (0..7).map(|i| vec![SqlValue::Int4(i)]).collect::<Vec<_>>(),
        "uuid ASC big-endian unsigned"
    );
    let desc = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY id DESC")
        .expect("uuid DESC on the GPU");
    assert_eq!(
        desc.rows,
        (0..7)
            .rev()
            .map(|i| vec![SqlValue::Int4(i)])
            .collect::<Vec<_>>(),
        "uuid DESC"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_mixed_b128_keys() {
    // Multi-key ORDER BY with a b128 key as the tie-broken primary, on the heterogeneous comparator.
    let mut e = Engine::new_local_cpu_oracle();
    // (a) numeric DESC, int ASC: a numeric tie is broken by the int key.
    e.execute_text(1, "CREATE TABLE t (v NUMERIC(10,2), tb INT, label INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (v, tb, label) VALUES (10.00, 2, 2), (10.00, 1, 1), (20.00, 5, 0)",
    )
    .unwrap();
    let s1 = e.populate_relational_residency_snapshot("t").unwrap();
    if s1.device_memory_proof.is_none() {
        return;
    }
    let r = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY v DESC, tb ASC")
        .expect("numeric+int hetero sort");
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(0)],
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(2)]
        ],
        "v=20 first; the v=10 tie broken by tb ASC"
    );
    // (b) uuid ASC, text ASC: a uuid tie is broken by the text key.
    e.execute_text(3, "CREATE TABLE u (id UUID, name TEXT, label INT)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO u (id, name, label) VALUES \
         ('00000000-0000-0000-0000-000000000001', 'bob', 1), \
         ('00000000-0000-0000-0000-000000000001', 'amy', 0), \
         ('00000000-0000-0000-0000-000000000002', 'zoe', 2)",
    )
    .unwrap();
    let s2 = e.populate_relational_residency_snapshot("u").unwrap();
    if s2.device_memory_proof.is_none() {
        return;
    }
    let r2 = e
        .execute_relational_select_text("SELECT label FROM u ORDER BY id ASC, name ASC")
        .expect("uuid+text hetero sort");
    assert_eq!(r2.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        r2.rows,
        vec![
            vec![SqlValue::Int4(0)],
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(2)]
        ],
        "uuid tie (..01) broken by name ASC (amy<bob), then ..02"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_expression_overflow_is_pg_error() {
    // ORDER BY a+b where a+b overflows int4 -> a clean PG "integer out of range" error (checked
    // arithmetic on-device), NOT a wrapped value, NOT a CPU re-execution.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a, b) VALUES (2147483647, 1), (1, 1)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let err = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a + b")
        .expect_err("a+b overflow must surface as a PG error, not wrap or CPU-fallback");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("out of range") || msg.contains("overflow"),
        "expected integer out of range, got: {err:?}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_via_gpu_sort() {
    // A non-grouped ORDER BY over an int column runs on the GENERAL GPU Expr executor + the GPU bitonic
    // sort (NOT the enumerated ordered-projection shape, NOT the CPU path). executed_target==Gpu proves
    // it took the general GPU path through the routing gate (`execute_relational_select_text`).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c BIGINT)")
        .unwrap();
    let rows: &[(i32, i32, i64)] = &[
        (5, 50, 500),
        (2, 20, 200),
        (8, 80, 800),
        (1, 10, 100),
        (9, 90, 900),
        (3, 30, 300),
    ];
    let values = rows
        .iter()
        .map(|(a, b, c)| format!("({a}, {b}, {c})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (a, b, c) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let int4_col = |res: &RelationalSelectResult, col: usize| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[col] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };

    // (a) ORDER BY a ASC -- and confirm it took the general GPU path.
    let asc = e
        .execute_relational_select_text("SELECT a, b FROM t ORDER BY a")
        .unwrap();
    assert_eq!(
        asc.executed_target,
        DeviceTarget::Gpu(0),
        "non-grouped ORDER BY must run on the general GPU path"
    );
    assert_eq!(int4_col(&asc, 0), vec![1, 2, 3, 5, 8, 9], "ORDER BY a ASC");

    // (b) ORDER BY a DESC.
    let desc = e
        .execute_relational_select_text("SELECT a, b FROM t ORDER BY a DESC")
        .unwrap();
    assert_eq!(desc.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int4_col(&desc, 0),
        vec![9, 8, 5, 3, 2, 1],
        "ORDER BY a DESC"
    );

    // (c) WHERE b > 25 ORDER BY a -> a in {3,5,8,9} (their b are 30/50/80/90).
    let filtered = e
        .execute_relational_select_text("SELECT a FROM t WHERE b > 25 ORDER BY a")
        .unwrap();
    assert_eq!(filtered.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(int4_col(&filtered, 0), vec![3, 5, 8, 9], "WHERE + ORDER BY");

    // (d) ORDER BY c DESC (an int8 key).
    let by_c = e
        .execute_relational_select_text("SELECT a, c FROM t ORDER BY c DESC")
        .unwrap();
    assert_eq!(by_c.executed_target, DeviceTarget::Gpu(0));
    let c_col: Vec<i64> = by_c
        .rows
        .iter()
        .map(|r| match r[1] {
            SqlValue::Int8(v) => v,
            ref other => panic!("expected Int8, got {other:?}"),
        })
        .collect();
    assert_eq!(c_col, vec![900, 800, 500, 300, 200, 100], "ORDER BY c DESC");

    // (e) ORDER BY a LIMIT 3 OFFSET 1 -> [2, 3, 5].
    let limited = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a LIMIT 3 OFFSET 1")
        .unwrap();
    assert_eq!(limited.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int4_col(&limited, 0),
        vec![2, 3, 5],
        "ORDER BY a LIMIT 3 OFFSET 1"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_select_limit_offset_window_edges() {
    // S4: OFFSET/LIMIT on the projection path now slices `indices_u64` (the device-ordered index vector)
    // BEFORE the column gather -- only the kept window is materialized from the device, no host
    // drain/truncate. These edge cases pin the windowing math against the prior drain/truncate: OFFSET
    // past the end, LIMIT 0, OFFSET+LIMIT past the end (clamped), and a DESC window. The ORDER BY makes
    // every window deterministic.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a) VALUES (5),(2),(8),(1),(9),(3)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let col = |res: &RelationalSelectResult| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };
    // Sorted ascending the rows are [1,2,3,5,8,9].

    // OFFSET past the end -> empty (start clamps to len).
    let beyond = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a OFFSET 10")
        .unwrap();
    assert_eq!(beyond.executed_target, DeviceTarget::Gpu(0));
    assert!(beyond.rows.is_empty(), "OFFSET past the end -> no rows");

    // LIMIT 0 -> empty.
    let zero = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a LIMIT 0")
        .unwrap();
    assert!(zero.rows.is_empty(), "LIMIT 0 -> no rows");

    // OFFSET 4 + LIMIT 100 past the end -> clamped to the tail [8,9].
    let tail = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a LIMIT 100 OFFSET 4")
        .unwrap();
    assert_eq!(
        col(&tail),
        vec![8, 9],
        "LIMIT past the end clamps to the tail"
    );

    // OFFSET only (no LIMIT) -> drop the first four, keep [8,9].
    let off = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a OFFSET 4")
        .unwrap();
    assert_eq!(col(&off), vec![8, 9], "OFFSET only keeps the tail");

    // DESC + LIMIT 2 OFFSET 1 -> from [9,8,5,3,2,1] skip 1, take 2 -> [8,5].
    let desc = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a DESC LIMIT 2 OFFSET 1")
        .unwrap();
    assert_eq!(col(&desc), vec![8, 5], "DESC LIMIT 2 OFFSET 1 window");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_500_rows() {
    // 500 rows (not a power of two -> padding) shuffled via a coprime stride (a permutation of 0..500),
    // sorted on the GPU. Exercises the bitonic sort at scale on the projection path.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE big (a INT)").unwrap();
    let vals = (0..500i32)
        .map(|i| format!("({})", (i * 137 + 11).rem_euclid(500)))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO big (a) VALUES {vals}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("big").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_relational_select_text("SELECT a FROM big ORDER BY a")
        .unwrap();
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let a_col: Vec<i32> = res
        .rows
        .iter()
        .map(|r| match r[0] {
            SqlValue::Int4(v) => v,
            ref other => panic!("expected Int4, got {other:?}"),
        })
        .collect();
    assert_eq!(
        a_col,
        (0..500).collect::<Vec<i32>>(),
        "500-row GPU ORDER BY a"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_radix_above_crossover() {
    // 11_000 rows (> the 10_000 adaptive crossover) -> the single-int-key ORDER BY takes the GPU RADIX
    // arm (engine_expr order_by_sort_i64), end to end. A coprime-stride (137, gcd(137,11000)=1)
    // permutation of 0..11000 must sort back to 0..11000 (ASC) / its reverse (DESC), on the GPU.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 11_000;
    let vals = (0..N)
        .map(|i| format!("({})", (i * 137 + 11).rem_euclid(N)))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO big (a) VALUES {vals}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("big").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let col = |sql: &str| -> Vec<i32> {
        let res = e.execute_relational_select_text(sql).unwrap();
        assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
        res.rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };
    assert_eq!(
        col("SELECT a FROM big ORDER BY a"),
        (0..N).collect::<Vec<i32>>(),
        "11k-row GPU radix ORDER BY a"
    );
    assert_eq!(
        col("SELECT a FROM big ORDER BY a DESC"),
        (0..N).rev().collect::<Vec<i32>>(),
        "11k-row GPU radix ORDER BY a DESC"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_text() {
    // A non-grouped ORDER BY over a TEXT column sorts on the GENERAL GPU Expr executor via the byte-wise
    // text bitonic comparator (lexicographic, UNSIGNED bytes, a prefix sorts smaller) -- NOT a CPU sort.
    // executed_target==Gpu proves the general GPU path. Covers prefixes, the empty string, duplicates, DESC.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (s TEXT, id INT)")
        .unwrap();
    // Deliberate corner cases: prefixes ('a' < 'ab'), empty string (sorts first), duplicates, mixed length.
    let rows: &[(&str, i32)] = &[
        ("banana", 1),
        ("apple", 2),
        ("ab", 3),
        ("a", 4),
        ("", 5),
        ("apple", 6),
        ("ab", 7),
        ("cherry", 8),
    ];
    let values = rows
        .iter()
        .map(|(s, id)| format!("('{s}', {id})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (s, id) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let text_col = |res: &RelationalSelectResult| -> Vec<String> {
        res.rows
            .iter()
            .map(|r| match &r[0] {
                SqlValue::Text(v) => v.to_string(),
                other => panic!("expected Text, got {other:?}"),
            })
            .collect()
    };
    // Closed-form oracle (S9): the 8 rows in unsigned-byte (lexicographic) order -- empty string first,
    // a < ab (a prefix sorts smaller), duplicates kept -- stated explicitly, not via a host Rust sort of
    // the input (which would re-implement the ORDER BY comparator on the CPU).
    let oracle: Vec<&str> = vec!["", "a", "ab", "ab", "apple", "apple", "banana", "cherry"];

    // (a) ORDER BY s ASC -- and confirm it took the general GPU path.
    let asc = e
        .execute_relational_select_text("SELECT s FROM t ORDER BY s")
        .unwrap();
    assert_eq!(
        asc.executed_target,
        DeviceTarget::Gpu(0),
        "text ORDER BY must run on the general GPU path"
    );
    assert_eq!(
        text_col(&asc),
        oracle,
        "ORDER BY s ASC (prefixes, empty, dups)"
    );

    // (b) ORDER BY s DESC -- the reverse key order (ties are identical strings, so order among them is moot).
    let desc = e
        .execute_relational_select_text("SELECT s FROM t ORDER BY s DESC")
        .unwrap();
    assert_eq!(desc.executed_target, DeviceTarget::Gpu(0));
    let mut oracle_desc = oracle.clone();
    oracle_desc.reverse();
    assert_eq!(text_col(&desc), oracle_desc, "ORDER BY s DESC");

    // (c) WHERE id > 4 ORDER BY s -> id in {5,6,7,8} = {"", "apple", "ab", "cherry"} sorted.
    let filtered = e
        .execute_relational_select_text("SELECT s FROM t WHERE id > 4 ORDER BY s")
        .unwrap();
    assert_eq!(filtered.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        text_col(&filtered),
        vec!["", "ab", "apple", "cherry"],
        "WHERE + text ORDER BY"
    );

    // (d) ORDER BY s LIMIT 3 -> the first three: ["", "a", "ab"].
    let limited = e
        .execute_relational_select_text("SELECT s FROM t ORDER BY s LIMIT 3")
        .unwrap();
    assert_eq!(limited.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        text_col(&limited),
        vec!["", "a", "ab"],
        "text ORDER BY LIMIT 3"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_text_300_rows() {
    // 300 rows (not a power of two -> bitonic padding), distinct zero-padded strings shuffled via a
    // coprime stride (a permutation of 0..300), GPU-sorted by the text comparator back to order.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE big (s TEXT)").unwrap();
    let vals = (0..300usize)
        .map(|i| format!("('{:04}')", (i * 137 + 11) % 300))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO big (s) VALUES {vals}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("big").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_relational_select_text("SELECT s FROM big ORDER BY s")
        .unwrap();
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let got: Vec<String> = res
        .rows
        .iter()
        .map(|r| match &r[0] {
            SqlValue::Text(v) => v.to_string(),
            other => panic!("expected Text, got {other:?}"),
        })
        .collect();
    let expected: Vec<String> = (0..300).map(|i| format!("{i:04}")).collect();
    assert_eq!(got, expected, "300-row GPU text ORDER BY");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_multikey() {
    // MULTI-KEY ORDER BY (`ORDER BY a ASC, b DESC, ...`) sorts on the GPU via the multi-key bitonic
    // comparator on the general Expr executor: each key, in significance order with its own direction,
    // breaks ties for the next. executed_target==Gpu proves the general GPU path (routing gate + GPU
    // multi-key sort), NOT the CPU/enumerated path. Rows are engineered so EVERY key is the real
    // tie-breaker -- drop any key and the expected order changes.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t2 (a INT, b INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t2 (a, b) VALUES (1, 10), (1, 30), (1, 20), (2, 50), (2, 40)",
    )
    .unwrap();
    e.execute_text(3, "CREATE TABLE t3 (a INT, b INT, c INT)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO t3 (a, b, c) VALUES (1, 5, 100), (1, 5, 50), (1, 8, 10), (2, 3, 7), (2, 3, 9)",
    )
    .unwrap();
    e.execute_text(5, "CREATE TABLE td (d DATE, x INT)")
        .unwrap();
    e.execute_text(
        6,
        "INSERT INTO td (d, x) VALUES ('2024-01-02', 5), ('2024-01-01', 9), \
         ('2024-01-01', 3), ('2024-01-02', 7)",
    )
    .unwrap();
    let s2 = e.populate_relational_residency_snapshot("t2").unwrap();
    e.populate_relational_residency_snapshot("t3").unwrap();
    e.populate_relational_residency_snapshot("td").unwrap();
    if s2.device_memory_proof.is_none() {
        return;
    }
    let int4_col = |res: &RelationalSelectResult, col: usize| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[col] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };

    // (a) two keys: a ASC, b DESC -- a-ties (1,1,1 / 2,2) broken by b DESCENDING.
    let ab = e
        .execute_relational_select_text("SELECT a, b FROM t2 ORDER BY a ASC, b DESC")
        .unwrap();
    assert_eq!(
        ab.executed_target,
        DeviceTarget::Gpu(0),
        "multi-key ORDER BY must run on the general GPU path"
    );
    assert_eq!(int4_col(&ab, 0), vec![1, 1, 1, 2, 2], "key a ASC");
    assert_eq!(
        int4_col(&ab, 1),
        vec![30, 20, 10, 50, 40],
        "key b DESC breaks a-ties"
    );

    // (b) three keys: a ASC, b DESC, c ASC -- a-ties broken by b, then (a,b)-ties broken by c. The two
    // (1,5,*) rows tie on a AND b and are ordered ONLY by c ASC (50 before 100).
    let abc = e
        .execute_relational_select_text("SELECT a, b, c FROM t3 ORDER BY a ASC, b DESC, c ASC")
        .unwrap();
    assert_eq!(abc.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(int4_col(&abc, 0), vec![1, 1, 1, 2, 2], "key a ASC");
    assert_eq!(int4_col(&abc, 1), vec![8, 5, 5, 3, 3], "key b DESC");
    assert_eq!(
        int4_col(&abc, 2),
        vec![10, 50, 100, 7, 9],
        "key c ASC breaks (a,b)-ties"
    );

    // (c) DATE + int: d ASC (primary), x DESC. The x order [9,3,7,5] proves d is the PRIMARY key --
    // sorting by x DESC alone would give [9,7,5,3]. (A mixed key-type multi-key sort.)
    let dx = e
        .execute_relational_select_text("SELECT x FROM td ORDER BY d ASC, x DESC")
        .unwrap();
    assert_eq!(dx.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int4_col(&dx, 0),
        vec![9, 3, 7, 5],
        "date primary ASC, int secondary DESC"
    );

    // (d) single-key down the SAME path is unchanged (K=1) -- regression guard.
    let single = e
        .execute_relational_select_text("SELECT a FROM t2 ORDER BY a ASC")
        .unwrap();
    assert_eq!(single.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int4_col(&single, 0),
        vec![1, 1, 1, 2, 2],
        "single key a ASC unchanged"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_mixed_int_text() {
    // MIXED int+text multi-key ORDER BY (`ORDER BY name /*text*/, age /*int*/, id /*int*/`) sorts on the
    // GPU via the HETEROGENEOUS comparator -- each key dispatched to the s64 compare (int) or the byte
    // compare (text). executed_target==Gpu proves the general GPU path. Rows are engineered so EVERY key
    // is the real tie-breaker. Completes the canonical `ORDER BY last_name, age, id`.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (name TEXT, age INT, id INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (name, age, id) VALUES \
         ('bob', 30, 1), ('alice', 25, 2), ('alice', 25, 3), ('alice', 40, 4), \
         ('bob', 30, 5), ('bob', 20, 6)",
    )
    .unwrap();
    e.execute_text(3, "CREATE TABLE names (last TEXT, first TEXT, id INT)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO names (last, first, id) VALUES \
         ('smith', 'bob', 1), ('jones', 'amy', 2), ('smith', 'amy', 3), ('smith', 'al', 4)",
    )
    .unwrap();
    let snap = e.populate_relational_residency_snapshot("people").unwrap();
    e.populate_relational_residency_snapshot("names").unwrap();
    if snap.device_memory_proof.is_none() {
        return;
    }
    let id_col = |res: &RelationalSelectResult| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };

    // (a) the canonical 3-key: name ASC (text primary), age DESC (int), id ASC (int). alice<bob; within
    // a name, age DESC; within name+age (the two alice/25 rows), id ASC. Each key a real tie-breaker.
    let abc = e
        .execute_relational_select_text("SELECT id FROM people ORDER BY name ASC, age DESC, id ASC")
        .unwrap();
    assert_eq!(
        abc.executed_target,
        DeviceTarget::Gpu(0),
        "mixed int+text ORDER BY must run on the general GPU path"
    );
    assert_eq!(
        id_col(&abc),
        vec![4, 2, 3, 1, 5, 6],
        "name ASC / age DESC / id ASC"
    );

    // (b) int primary, text secondary: age ASC, name ASC, id ASC. ages group; name (degenerate tie
    // within each age here) then id ASC. Proves age is primary (not name).
    let ba = e
        .execute_relational_select_text("SELECT id FROM people ORDER BY age ASC, name ASC, id ASC")
        .unwrap();
    assert_eq!(ba.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        id_col(&ba),
        vec![6, 2, 3, 1, 5, 4],
        "age ASC / name ASC / id ASC"
    );

    // (c) DESC on the TEXT key: name DESC, id ASC. bob before alice; within a name, id ASC. (The text
    // key must sort DESC via the comparator direction, not a key sentinel.)
    let nd = e
        .execute_relational_select_text("SELECT id FROM people ORDER BY name DESC, id ASC")
        .unwrap();
    assert_eq!(nd.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(id_col(&nd), vec![1, 5, 6, 2, 3, 4], "name DESC / id ASC");

    // (d) TWO text keys: last ASC, first ASC. jones<smith; within smith, first ASC ('al'<'amy'<'bob').
    // Two text slots in the key_plan, both dispatched to the byte compare.
    let lf = e
        .execute_relational_select_text("SELECT id FROM names ORDER BY last ASC, first ASC")
        .unwrap();
    assert_eq!(lf.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        id_col(&lf),
        vec![2, 4, 3, 1],
        "last ASC / first ASC (two text keys)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_by_uuid_key() {
    // GROUP BY a UUID (i128) key via atom.cas.b128; output sorts by canonical/memcmp byte order. Uses
    // early-byte AND late-byte differences (exercises the sort + the full 128-bit key equality), and
    // INCLUDES the uuid whose LE i128 == EMPTY128 (i128::MIN) -> the DEDICATED slot path for i128 keys.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE u (id UUID, v INT)")
        .unwrap();
    let rows: &[(&str, i32)] = &[
        ("00000000-0000-0000-0000-000000000001", 10),
        ("00000000-0000-0000-0000-000000000001", 20),
        ("00000000-0000-0000-0000-000000000080", 9), // LE i128 == i128::MIN -> dedicated slot
        ("00000000-0000-0000-0000-0000000000ff", 7),
        ("ff000000-0000-0000-0000-000000000000", 5), // early-byte difference
    ];
    let values = rows
        .iter()
        .map(|(id, v)| format!("('{id}', {v})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO u (id, v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("u").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let uuid = |s: &str| SqlValue::Uuid(gpu_db_sql::uuid::parse_uuid(s).expect("valid uuid"));
    // memcmp order: ..0001 < ..0080 < ..00ff < ff00..
    let count = e
        .execute_resident_expr_select_sql("SELECT id, COUNT(*) FROM u GROUP BY id")
        .expect("uuid-key COUNT");
    assert_eq!(count.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        count.rows,
        vec![
            vec![uuid("00000000-0000-0000-0000-000000000001"), SqlValue::Int8(2)],
            vec![uuid("00000000-0000-0000-0000-000000000080"), SqlValue::Int8(1)],
            vec![uuid("00000000-0000-0000-0000-0000000000ff"), SqlValue::Int8(1)],
            vec![uuid("ff000000-0000-0000-0000-000000000000"), SqlValue::Int8(1)],
        ],
        "GROUP BY uuid key, COUNT, sorted by memcmp (incl. the i128::MIN-valued dedicated-slot uuid)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_uuid_key_and_uuid_value_min() {
    // Compose both b128 paths in one query: GROUP BY a uuid KEY (atom.cas.b128 claim) while taking MIN
    // of a uuid VALUE (the b128 CAS loop).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE w (k UUID, val UUID)")
        .unwrap();
    let rows: &[(&str, &str)] = &[
        (
            "aaaaaaaa-0000-0000-0000-000000000000",
            "00000000-0000-0000-0000-000000000005",
        ),
        (
            "aaaaaaaa-0000-0000-0000-000000000000",
            "00000000-0000-0000-0000-000000000002",
        ),
        (
            "bbbbbbbb-0000-0000-0000-000000000000",
            "ffffffff-ffff-ffff-ffff-ffffffffffff",
        ),
    ];
    let values = rows
        .iter()
        .map(|(k, val)| format!("('{k}', '{val}')"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO w (k, val) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("w").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let uuid = |s: &str| SqlValue::Uuid(gpu_db_sql::uuid::parse_uuid(s).expect("valid uuid"));
    let r = e
        .execute_resident_expr_select_sql("SELECT k, MIN(val) FROM w GROUP BY k")
        .expect("uuid key + uuid value MIN");
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        r.rows,
        vec![
            vec![
                uuid("aaaaaaaa-0000-0000-0000-000000000000"),
                uuid("00000000-0000-0000-0000-000000000002"),
            ],
            vec![
                uuid("bbbbbbbb-0000-0000-0000-000000000000"),
                uuid("ffffffff-ffff-ffff-ffff-ffffffffffff"),
            ],
        ],
        "GROUP BY uuid key, MIN(uuid value)"
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v BIGINT)")
        .unwrap();
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v BIGINT)")
        .unwrap();
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v BIGINT)")
        .unwrap();
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g BIGINT, v INT)")
        .unwrap();
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (b INT, g BIGINT)")
        .unwrap();
    // i64-SECTION FLIP pin: this guard exercises the SINGLE-BUFFER misaligned int8 key
    // (the kill-switch configuration since the 2026-07-03 flip).
    e.set_shard_int8_section_enabled(false);
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
    assert_eq!(
        g_off % 8,
        4,
        "int8 key column must be 4-mod-8 to exercise the misaligned read"
    );

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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(20,2))")
        .unwrap();
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
        vec![(i4(1), "27.00".to_string()), (i4(2), "49.50".to_string()),],
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
    let mut e = Engine::new_local_cpu_oracle();
    // NUMERIC(38,19) value 9e18 -> mantissa 9e18 * 10^19 = 9e37 (the literal 9e18 fits the parser's
    // i64 range; the scale lifts the mantissa to 9e37). Two in one group sum to 1.8e38 > i128::MAX.
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(38,19))")
        .unwrap();
    let big = "9000000000000000000"; // 9e18
    e.execute_text(
        2,
        &format!("INSERT INTO t (g,v) VALUES (1,{big}),(1,{big})"),
    )
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(38,19))")
        .unwrap();
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
            (
                SqlValue::Int4(1),
                Decimal128::new(15 * unit, 19).to_decimal_string()
            ), // 3 * 5.0
            (
                SqlValue::Int4(2),
                Decimal128::new(-10 * unit, 19).to_decimal_string()
            ), // 2 * -5.0
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(10,2))")
        .unwrap();
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
    let numeric_strs = |rows: &RowBlock| -> Vec<(SqlValue, String)> {
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
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(38,19))")
        .unwrap();
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
    let numeric_strs = |rows: &RowBlock| -> Vec<(SqlValue, String)> {
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
        vec![(i4(1), dec(5 * unit)), (i4(2), dec(-unit)), (i4(3), dec(0)),],
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
    assert_eq!(
        (4 * unit + 10) >> 64,
        (4 * unit + 200) >> 64,
        "positive ties must share high limb"
    );
    assert_eq!(
        (-(4 * unit + 10)) >> 64,
        (-(4 * unit + 200)) >> 64,
        "negative ties must share high limb"
    );
    assert_ne!(
        (4 * unit + 10) >> 64,
        (13 * unit) >> 64,
        "the decoy must have a DIFFERENT high limb"
    );

    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v NUMERIC(38,19))")
        .unwrap();
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
    // Closed-form oracle: the explicit per-group tie-break winners (the i128 mantissa min/max),
    // stated as constants rather than a host .iter().min()/.max() re-implementation (S9). g1's MIN is
    // the smallest mantissa among the high-limb ties (4e19+10); MAX is the distinct decoy 13e19. g2 is
    // all-negative, so MIN is the most negative (-(4e19+200)) and MAX the least negative (-(4e19+10)).
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
        vec![ds(4 * unit + 10), ds(-(4 * unit + 200))],
        "MIN decided by the unsigned low limb among high-limb ties (decoy must not leak)"
    );
    assert_eq!(
        got("SELECT g, MAX(v) FROM t GROUP BY g"),
        vec![ds(13 * unit), ds(-(4 * unit + 10))],
        "MAX decided by the unsigned low limb among high-limb ties"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_reuse_types_int2_date_timestamp() {
    // Int2 + Date ride the int4 (4-byte) read; Timestamp rides the int8 (8-byte) read -- executor
    // type-recognition only, no kernel change. Verify GROUP BY keys + MIN/MAX narrow back to the right
    // SqlType, and int2 SUM (PG SUM(int2) -> int8).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE t (g INT, d DATE, ts TIMESTAMP, s SMALLINT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,d,ts,s) VALUES \
         (1, '2024-01-10', '2024-01-10 08:00:00', 5), \
         (1, '2024-03-20', '2024-02-01 12:00:00', 15), \
         (2, '2023-12-01', '2023-12-01 00:00:00', -7)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;

    // int2 MIN/MAX -> Int2; SUM -> Int8 (PG widening).
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT g, MIN(s) FROM t GROUP BY g")
            .unwrap()
            .rows,
        vec![
            vec![i4(1), SqlValue::Int2(5)],
            vec![i4(2), SqlValue::Int2(-7)]
        ],
        "MIN(int2) -> Int2"
    );
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT g, MAX(s) FROM t GROUP BY g")
            .unwrap()
            .rows,
        vec![
            vec![i4(1), SqlValue::Int2(15)],
            vec![i4(2), SqlValue::Int2(-7)]
        ],
        "MAX(int2) -> Int2"
    );
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT g, SUM(s) FROM t GROUP BY g")
            .unwrap()
            .rows,
        vec![
            vec![i4(1), SqlValue::Int8(20)],
            vec![i4(2), SqlValue::Int8(-7)]
        ],
        "SUM(int2) -> Int8"
    );

    // date/timestamp MIN/MAX -> the right variant, correctly ordered (g=1 has two rows).
    let md = e
        .execute_resident_expr_select_sql("SELECT g, MIN(d) FROM t GROUP BY g")
        .unwrap();
    let xd = e
        .execute_resident_expr_select_sql("SELECT g, MAX(d) FROM t GROUP BY g")
        .unwrap();
    match (&md.rows[0][1], &xd.rows[0][1]) {
        (SqlValue::Date(a), SqlValue::Date(b)) => assert!(a < b, "g=1 MIN(date) < MAX(date)"),
        o => panic!("MIN/MAX(date) must be Date, got {o:?}"),
    }
    let mt = e
        .execute_resident_expr_select_sql("SELECT g, MIN(ts) FROM t GROUP BY g")
        .unwrap();
    let xt = e
        .execute_resident_expr_select_sql("SELECT g, MAX(ts) FROM t GROUP BY g")
        .unwrap();
    match (&mt.rows[0][1], &xt.rows[0][1]) {
        (SqlValue::Timestamp(a), SqlValue::Timestamp(b)) => {
            assert!(a < b, "g=1 MIN(ts) < MAX(ts)")
        }
        o => panic!("MIN/MAX(timestamp) must be Timestamp, got {o:?}"),
    }

    // GROUP BY a DATE key -> Date key variant.
    let cd = e
        .execute_resident_expr_select_sql("SELECT d, COUNT(*) FROM t GROUP BY d")
        .unwrap();
    assert_eq!(cd.rows.len(), 3, "3 distinct dates");
    assert!(
        cd.rows.iter().all(|r| matches!(r[0], SqlValue::Date(_))),
        "GROUP BY date yields Date keys"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_group_by_two_level_at_scale() {
    // The two-level shared-mem GROUP BY at scale: LOW cardinality (many rows per group, exercising the
    // block-local aggregation + cross-block merge) and HIGH cardinality (thousands of distinct keys
    // across many blocks). Both assert against host oracles -- a wrong merge would surface here.
    let mut e = Engine::new_local_cpu_oracle();
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
    e.execute_text(2, &format!("INSERT INTO lo (g,v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("lo").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let c = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM lo GROUP BY g")
        .expect("lo count");
    let exp_c: Vec<Vec<SqlValue>> = (0..ngroups)
        .map(|g| vec![SqlValue::Int4(g as i32), SqlValue::Int8(counts[g])])
        .collect();
    assert_eq!(c.rows, exp_c, "low-card COUNT at scale (two-level merge)");
    let s = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM lo GROUP BY g")
        .expect("lo sum");
    let exp_s: Vec<Vec<SqlValue>> = (0..ngroups)
        .map(|g| vec![SqlValue::Int4(g as i32), SqlValue::Int8(sums[g])])
        .collect();
    assert_eq!(s.rows, exp_s, "low-card SUM at scale");

    // HIGH cardinality: every key distinct -> one group per row, merged across many blocks.
    let mut e2 = Engine::new_local_cpu_oracle();
    e2.execute_text(1, "CREATE TABLE hi (g INT, v INT)")
        .unwrap();
    let h = 3000usize;
    let mut hv = String::with_capacity(h * 10);
    for i in 0..h {
        if i > 0 {
            hv.push(',');
        }
        hv.push_str(&format!("({},{})", i as i64, (i * 2) as i64));
    }
    e2.execute_text(2, &format!("INSERT INTO hi (g,v) VALUES {hv}"))
        .unwrap();
    if e2
        .populate_relational_residency_snapshot("hi")
        .unwrap()
        .device_memory_proof
        .is_none()
    {
        return;
    }
    let hc = e2
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM hi GROUP BY g")
        .expect("hi sum");
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
    let mut e = Engine::new_local_cpu_oracle();
    // THE FLIP: this test exercises the SINGLE-BUFFER layer (a supported, settable configuration;
    // sharded is the default) — pin the layout under test.
    e.set_shard_residency_enabled(false);
    // One table, four key columns of different cardinality over the same rows -> one residency, four
    // GROUP BY cardinalities. g4/g64/g4k cycle; gall is all-distinct (high cardinality).
    e.execute_text(
        1,
        "CREATE TABLE t (g4 INT, g64 INT, g4k INT, gall INT, v INT)",
    )
    .unwrap();
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
            vals.push_str(&format!(
                "({},{},{},{},{})",
                j % 4,
                j % 64,
                j % 4096,
                j,
                j % 100
            ));
        }
        e.execute_text(
            txid,
            &format!("INSERT INTO t (g4,g64,g4k,gall,v) VALUES {vals}"),
        )
        .unwrap();
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
    eprintln!(
        "{:>8}  {:>13}  {:>13}  {:>9}",
        "groups", "single ms", "two-lvl ms", "speedup"
    );
    for key in ["g4", "g64", "g4k", "gall"] {
        // Correctness: both kernels must agree on (key, count, sum) before we trust the timings.
        // (min/max intentionally differ: the single-level kernel computes them, the two-level does
        // not -- so compare the COUNT/SUM aggregates both kernels produce, not the whole row.)
        let mut a = e.group_by_i32_bench("t", key, "v", false).unwrap();
        let mut b = e.group_by_i32_bench("t", key, "v", true).unwrap();
        a.sort_by_key(|r| r.key);
        b.sort_by_key(|r| r.key);
        let proj = |rows: &[gpu_db_execution::GroupByI32Row]| {
            rows.iter()
                .map(|r| (r.key, r.count, r.sum))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            proj(&a),
            proj(&b),
            "single-level and two-level disagree for {key}"
        );
        let all = gpu_db_execution::grouped_agg_mask::ALL;
        let single = e
            .group_by_i32_bench_kernel_ms("t", key, "v", false, 200, 0, all)
            .unwrap();
        let two = e
            .group_by_i32_bench_kernel_ms("t", key, "v", true, 200, 0, all)
            .unwrap();
        eprintln!(
            "{:>8}  {:>13.4}  {:>13.4}  {:>8.2}x",
            a.len(),
            single,
            two,
            single / two
        );
    }

    // QUERY-AWARE AGGREGATE PRUNING (this slice): COUNT-only mask vs ALL mask, on BOTH the single-level
    // (count+sum+min+max) and two-level (count+sum) kernels, across cardinalities. A COUNT-only mask skips
    // the SUM (+ MIN/MAX, single-level) per-row atomics -> expect COUNT-only <= ALL, most at mid/high card
    // where the avoided global atomics contend.
    eprintln!(
        "\n=== AGG-PRUNE: COUNT-only mask vs ALL mask (kernel-only ms, min of 200) ===\n\
         single-level computes COUNT+SUM+MIN+MAX; two-level computes COUNT+SUM. speedup = ALL / COUNT-only"
    );
    let count = gpu_db_execution::grouped_agg_mask::COUNT;
    let all = gpu_db_execution::grouped_agg_mask::ALL;
    eprintln!(
        "{:>8}  {:>12}  {:>12}  {:>8}   {:>12}  {:>12}  {:>8}",
        "groups", "1lvl ALL", "1lvl CNT", "spd", "2lvl ALL", "2lvl CNT", "spd"
    );
    for key in ["g4", "g64", "g4k", "gall"] {
        let groups = e.group_by_i32_bench("t", key, "v", false).unwrap().len();
        let s_all = e
            .group_by_i32_bench_kernel_ms("t", key, "v", false, 200, 0, all)
            .unwrap();
        let s_cnt = e
            .group_by_i32_bench_kernel_ms("t", key, "v", false, 200, 0, count)
            .unwrap();
        let t_all = e
            .group_by_i32_bench_kernel_ms("t", key, "v", true, 200, 0, all)
            .unwrap();
        let t_cnt = e
            .group_by_i32_bench_kernel_ms("t", key, "v", true, 200, 0, count)
            .unwrap();
        eprintln!(
            "{:>8}  {:>12.4}  {:>12.4}  {:>7.2}x   {:>12.4}  {:>12.4}  {:>7.2}x",
            groups,
            s_all,
            s_cnt,
            s_all / s_cnt,
            t_all,
            t_cnt,
            t_all / t_cnt
        );
    }

    // SCALE AXIS: fixed LOW cardinality (g4 = 4 groups), growing row count. This is the axis that
    // answers "does it scale" -- the two-level win should GROW with rows (more single-level global
    // contention to avoid), unlike the cardinality table above (fixed rows, varying group count).
    eprintln!(
        "\n=== SCALE: GROUP BY g4 (4 groups fixed), growing rows (kernel-only, min of 200) ==="
    );
    eprintln!(
        "{:>9}  {:>13}  {:>13}  {:>9}",
        "rows", "single ms", "two-lvl ms", "speedup"
    );
    for rows in [25_000usize, 50_000, 100_000, 200_000] {
        let all = gpu_db_execution::grouped_agg_mask::ALL;
        let single = e
            .group_by_i32_bench_kernel_ms("t", "g4", "v", false, 200, rows, all)
            .unwrap();
        let two = e
            .group_by_i32_bench_kernel_ms("t", "g4", "v", true, 200, rows, all)
            .unwrap();
        eprintln!(
            "{:>9}  {:>13.4}  {:>13.4}  {:>8.2}x",
            rows,
            single,
            two,
            single / two
        );
    }
    eprintln!();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_select_evaluates_is_null_and_is_not_null_via_validity_bitmap() {
    // `WHERE v IS NULL` / `IS NOT NULL` runs ON THE GPU (M3 -- doc 21): the column's NULL validity
    // bitmap (slice 2a) feeds the SAME bitmap->mask kernel as a bool column, pointed at the validity
    // bitmap. GPU-native oracle = CONSTRUCTION (we know which rows are NULL by the insert rule). Projects
    // `id` (which has no NULLs) for the surviving rows.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();

    const N: i32 = 300;
    let is_null = |i: i32| i % 3 == 0; // v is NULL when i % 3 == 0, else i
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        if is_null(i) {
            values.push_str(&format!("({i}, NULL)"));
        } else {
            values.push_str(&format!("({i}, {i})"));
        }
    }
    e.execute_text(2, &format!("INSERT INTO t (id, v) VALUES {values}"))
        .unwrap();

    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let Command::Select(select) = parse_command("SELECT id FROM t").unwrap() else {
        unreachable!()
    };

    // IS NULL: surviving ids are exactly those where v is NULL.
    let is_null_pred = ResidentExpr::IsNull {
        col: 1,
        is_not_null: false,
    };
    let res_null = e
        .execute_resident_expr_select(&select, &is_null_pred)
        .expect("IS NULL on GPU");
    let expected_null: Vec<Vec<SqlValue>> = (0..N)
        .filter(|&i| is_null(i))
        .map(|i| vec![SqlValue::Int4(i)])
        .collect();
    assert_eq!(
        res_null.rows, expected_null,
        "WHERE v IS NULL must return exactly the NULL-v rows' ids, evaluated on the GPU bitmap"
    );
    assert_eq!(res_null.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(res_null.fallback_reason, None);

    // IS NOT NULL: the complement.
    let not_null_pred = ResidentExpr::IsNull {
        col: 1,
        is_not_null: true,
    };
    let res_not_null = e
        .execute_resident_expr_select(&select, &not_null_pred)
        .expect("IS NOT NULL on GPU");
    let expected_not_null: Vec<Vec<SqlValue>> = (0..N)
        .filter(|&i| !is_null(i))
        .map(|i| vec![SqlValue::Int4(i)])
        .collect();
    assert_eq!(
        res_not_null.rows, expected_not_null,
        "WHERE v IS NOT NULL must return exactly the non-NULL rows' ids"
    );
    assert_eq!(res_not_null.executed_target, DeviceTarget::Gpu(0));

    // Non-vacuity: the two results partition all rows, both non-empty, and disjoint.
    assert_eq!(res_null.rows.len() + res_not_null.rows.len(), N as usize);
    assert!(!res_null.rows.is_empty() && !res_not_null.rows.is_empty());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_is_null_on_a_column_with_no_nulls_uses_a_constant_mask() {
    // The all-valid case (M3 -- doc 21): a column with NO NULLs has no validity bitmap, so inside AND/OR
    // `IS NULL`/`IS NOT NULL` lowers to a CONSTANT mask in the predicate VM (a device memset, no kernel) --
    // IS NOT NULL is all-1, IS NULL all-0. Combined with `id < K` to prove the constant is real (not just
    // "all rows" / "no rows" by accident).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE u (id INT, w INT)").unwrap();
    const N: i32 = 200;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {i})")); // w is never NULL
    }
    e.execute_text(2, &format!("INSERT INTO u (id, w) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("u").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // w has no NULLs -> no validity bitmap -> exercises the ConstMask path (not BoolMask).
    assert!(snapshot.resident_device_null_columns.is_empty());

    let Command::Select(select) = parse_command("SELECT id FROM u").unwrap() else {
        unreachable!()
    };
    const K: i32 = 50;
    let lt_k = || ResidentExpr::Binary {
        op: ResidentBinaryOp::Lt,
        lhs: Box::new(ResidentExpr::Column(0)),
        rhs: Box::new(ResidentExpr::Int4Literal(K)),
    };

    // `w IS NOT NULL AND id < K`: ConstMask{true} (all valid) AND (id<K) -> id in [0, K). If ConstMask
    // were wrongly all-0, this would be empty.
    let pred_not_null = ResidentExpr::Binary {
        op: ResidentBinaryOp::And,
        lhs: Box::new(ResidentExpr::IsNull {
            col: 1,
            is_not_null: true,
        }),
        rhs: Box::new(lt_k()),
    };
    let res = e
        .execute_resident_expr_select(&select, &pred_not_null)
        .expect("ConstMask(true) AND on GPU");
    let expected: Vec<Vec<SqlValue>> = (0..K).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        res.rows, expected,
        "w IS NOT NULL (all valid) AND id<K must be id in [0,K)"
    );
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));

    // `w IS NULL AND id < K`: ConstMask{false} AND (...) -> empty. If ConstMask were wrongly all-1, this
    // would be id<K (non-empty).
    let pred_null = ResidentExpr::Binary {
        op: ResidentBinaryOp::And,
        lhs: Box::new(ResidentExpr::IsNull {
            col: 1,
            is_not_null: false,
        }),
        rhs: Box::new(lt_k()),
    };
    let res_null = e
        .execute_resident_expr_select(&select, &pred_null)
        .expect("ConstMask(false) AND on GPU");
    assert!(
        res_null.rows.is_empty(),
        "w IS NULL on a no-NULL column matches nothing"
    );
    assert_eq!(res_null.executed_target, DeviceTarget::Gpu(0));

    // STANDALONE (not in AND/OR) over the no-NULL column exercises `lower_is_null_predicate`'s no-bitmap
    // arm: IS NOT NULL returns all rows, IS NULL none -- directly, without a kernel.
    let res_all = e
        .execute_resident_expr_select(
            &select,
            &ResidentExpr::IsNull {
                col: 1,
                is_not_null: true,
            },
        )
        .expect("standalone IS NOT NULL no-bitmap");
    let all_ids: Vec<Vec<SqlValue>> = (0..N).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        res_all.rows, all_ids,
        "standalone w IS NOT NULL over a no-NULL column = all rows"
    );
    let res_none = e
        .execute_resident_expr_select(
            &select,
            &ResidentExpr::IsNull {
                col: 1,
                is_not_null: false,
            },
        )
        .expect("standalone IS NULL no-bitmap");
    assert!(
        res_none.rows.is_empty(),
        "standalone w IS NULL over a no-NULL column = empty"
    );
}

// ===========================================================================
// S4 AUDIT (audit-237f3e34): adversarial LIMIT/OFFSET windowing tests.
// These probe the corners the shipped tests miss. SAFE TO DELETE after audit.
// ===========================================================================

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s4_grouped_single_group_default_limit() {
    // RISK: the GROUP BY guard changed `rows.len() > 1` -> `... || offset.is_some() || limit.is_some()`.
    // With EXACTLY ONE group + a LIMIT, the OLD code SKIPPED the sort entirely (rows.len() <= 1) and ran
    // drain/truncate on the 1 row. The NEW code now ENTERS the block and calls gpu_sort_permutation on a
    // 1-row payload (which must return identity, not error). Verify the single group still comes back.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (g, v) VALUES (7,1),(7,2),(7,3)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let r = |g: i32, c: i64| vec![SqlValue::Int4(g), SqlValue::Int8(c)];

    // default order (no ORDER BY) + LIMIT 1 over a single group.
    let lim = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g LIMIT 1")
        .expect("single-group default-order LIMIT 1 must run");
    assert_eq!(
        lim.rows,
        vec![r(7, 3)],
        "single group, default order, LIMIT 1"
    );

    // default order + OFFSET 1 over a single group -> empty.
    let off = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g OFFSET 1")
        .expect("single-group OFFSET 1 must run");
    assert!(off.rows.is_empty(), "OFFSET 1 over 1 group -> empty");

    // explicit ORDER BY + LIMIT 1 over a single group (forces gpu_sort_permutation on 1 row).
    let ord = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g DESC LIMIT 1",
        )
        .expect("single-group ORDER BY LIMIT 1 must run");
    assert_eq!(
        ord.rows,
        vec![r(7, 3)],
        "single group, ORDER BY DESC, LIMIT 1"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s4_grouped_single_text_group_order_limit() {
    // RISK (claim #2): a single TEXT group + ORDER BY + LIMIT 1 now builds the hetero payload over a
    // 1-row result and calls gpu_sort_permutation. gpu_sort_permutation short-circuits rows.len()<=1 to
    // identity BEFORE building any payload, so this must NOT error and must return the one group.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g TEXT, v INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (g, v) VALUES ('apple',10),('apple',20)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let want = vec![vec![SqlValue::Text("apple".to_string()), SqlValue::Int8(2)]];

    let lim = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g LIMIT 1")
        .expect("single text group default LIMIT 1");
    assert_eq!(lim.rows, want, "single text group, default order, LIMIT 1");

    let ord = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g LIMIT 5")
        .expect("single text group ORDER BY LIMIT 5");
    assert_eq!(ord.rows, want, "single text group, ORDER BY, LIMIT > 1");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s4_grouped_composite_single_group_limit() {
    // RISK (claim #2): a single COMPOSITE-key group + LIMIT now enters the windowing block. The default
    // branch computes n_group_cols=2 for the composite key; gpu_sort_permutation identity-short-circuits
    // at 1 row so the 2-col order is never evaluated. Verify the group survives.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, v INT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO t (a,b,v) VALUES (1,2,10),(1,2,20),(1,2,30)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let want = vec![vec![
        SqlValue::Int4(1),
        SqlValue::Int4(2),
        SqlValue::Int8(3),
    ]];
    let lim = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*) FROM t GROUP BY a, b LIMIT 1")
        .expect("single composite group LIMIT 1");
    assert_eq!(
        lim.rows, want,
        "single composite group, default order, LIMIT 1"
    );
    let off0 = e
        .execute_resident_expr_select_sql(
            "SELECT a, b, COUNT(*) FROM t GROUP BY a, b OFFSET 0 LIMIT 1",
        )
        .expect("OFFSET 0 LIMIT 1");
    assert_eq!(off0.rows, want, "OFFSET 0 LIMIT 1 over 1 composite group");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s4_grouped_having_empties_then_limit() {
    // RISK (claim #4): HAVING filters EVERYTHING -> rows is empty, but LIMIT is present so the windowing
    // block is entered with an EMPTY perm. perm[start..end] must be the empty slice (no panic), result
    // empty. Then a HAVING that leaves exactly ONE group + LIMIT (the 1-row windowing path post-HAVING).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        &format!("INSERT INTO t (g, v) VALUES {GROUPED_CLAUSE_ROWS}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // No group has COUNT(*) > 100 -> HAVING empties the result; LIMIT present -> empty perm window.
    let empty = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g HAVING COUNT(*) > 100 ORDER BY g LIMIT 3",
        )
        .expect("HAVING-empty + LIMIT must not panic");
    assert!(
        empty.rows.is_empty(),
        "HAVING removed all groups -> empty, no panic"
    );

    // HAVING leaves exactly ONE group (g4 has COUNT 4) -> single-row windowing post-HAVING.
    let one = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g HAVING COUNT(*) > 3 ORDER BY g LIMIT 5",
        )
        .expect("HAVING-one + LIMIT");
    assert_eq!(
        one.rows,
        vec![vec![SqlValue::Int4(4), SqlValue::Int8(4)]],
        "HAVING leaves 1 group; window keeps it"
    );

    // HAVING-empty + OFFSET only (no LIMIT) -> empty, no panic.
    let empty_off = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g HAVING COUNT(*) > 100 OFFSET 2",
        )
        .expect("HAVING-empty + OFFSET must not panic");
    assert!(empty_off.rows.is_empty(), "HAVING-empty + OFFSET -> empty");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s4_grouped_multi_aggregate_order_limit_offset() {
    // RISK (claim #6, #30 alignment): a MULTI-aggregate GROUP BY (SUM + MIN + MAX) where the multi-pass
    // alignment built `rows`, then ORDER BY an AGGREGATE + LIMIT + OFFSET windows the permutation. Verify
    // the windowed rows are exactly the right groups with the right (cross-pass-aligned) aggregate values.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        &format!("INSERT INTO t (g, v) VALUES {GROUPED_CLAUSE_ROWS}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // Per group: g1 sum=60 min=10 max=30; g2 sum=5 min=5 max=5; g3 sum=15 min=7 max=8;
    //            g4 sum=10 min=1 max=4; g5 sum=99 min=99 max=99.
    // ORDER BY SUM(v) DESC -> g5(99), g1(60), g3(15), g4(10), g2(5). OFFSET 1 LIMIT 2 -> g1, g3.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT g, SUM(v), MIN(v), MAX(v) FROM t GROUP BY g ORDER BY SUM(v) DESC LIMIT 2 OFFSET 1",
        )
        .expect("multi-agg ORDER BY agg LIMIT OFFSET");
    let row = |g: i32, s: i64, mn: i32, mx: i32| {
        vec![
            SqlValue::Int4(g),
            SqlValue::Int8(s),
            SqlValue::Int4(mn),
            SqlValue::Int4(mx),
        ]
    };
    assert_eq!(
        res.rows,
        vec![row(1, 60, 10, 30), row(3, 15, 7, 8)],
        "multi-agg window must keep cross-pass-aligned g1,g3"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s4_resident_limit_no_order_index_order_preserved() {
    // RISK (claim #8): a LIMIT WITHOUT ORDER BY on the resident-projection path. The new code windows the
    // UNSORTED indices_u64 (compaction output, ascending row index) BEFORE the gather. The OLD code
    // gathered all then drained/truncated. Both must yield the SAME rows in the SAME order. With values
    // chosen so the stored row order != value order, this distinguishes "windowed survivor indices" from
    // any accidental sort. Insert order = stored order on a fresh table.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a) VALUES (50),(20),(80),(10),(90),(30)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let col = |res: &RelationalSelectResult| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref o => panic!("expected Int4 got {o:?}"),
            })
            .collect()
    };
    // No ORDER BY: stored order is insert order [50,20,80,10,90,30].
    // LIMIT 3 -> first three in stored order [50,20,80].
    let lim3 = e
        .execute_relational_select_text("SELECT a FROM t LIMIT 3")
        .expect("LIMIT 3 no ORDER BY");
    assert_eq!(
        col(&lim3),
        vec![50, 20, 80],
        "LIMIT 3 keeps the first 3 in stored order"
    );
    // OFFSET 2 LIMIT 2 -> [80,10].
    let win = e
        .execute_relational_select_text("SELECT a FROM t LIMIT 2 OFFSET 2")
        .expect("LIMIT 2 OFFSET 2 no ORDER BY");
    assert_eq!(col(&win), vec![80, 10], "OFFSET 2 LIMIT 2 in stored order");
    // OFFSET only.
    let off = e
        .execute_relational_select_text("SELECT a FROM t OFFSET 4")
        .expect("OFFSET 4 no ORDER BY");
    assert_eq!(col(&off), vec![90, 30], "OFFSET 4 keeps stored tail");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s4_resident_with_where_limit_window() {
    // RISK: LIMIT windowing interacts with a WHERE filter (indices_u64 is the SURVIVOR set). Window must
    // slice survivors, not raw rows. WHERE a > 25 over [50,20,80,10,90,30] -> survivors [50,80,90,30]
    // (stored order). LIMIT 2 OFFSET 1 -> [80,90].
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a) VALUES (50),(20),(80),(10),(90),(30)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let col = |res: &RelationalSelectResult| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref o => panic!("expected Int4 got {o:?}"),
            })
            .collect()
    };
    let win = e
        .execute_relational_select_text("SELECT a FROM t WHERE a > 25 LIMIT 2 OFFSET 1")
        .expect("WHERE + LIMIT window");
    assert_eq!(
        col(&win),
        vec![80, 90],
        "WHERE survivors windowed, not raw rows"
    );
    // OFFSET past the survivor count -> empty (4 survivors, OFFSET 4).
    let beyond = e
        .execute_relational_select_text("SELECT a FROM t WHERE a > 25 OFFSET 4")
        .expect("WHERE + OFFSET past survivors");
    assert!(beyond.rows.is_empty(), "OFFSET == survivor count -> empty");
    // WHERE matches nothing + LIMIT -> empty, no panic.
    let none = e
        .execute_relational_select_text("SELECT a FROM t WHERE a > 1000 LIMIT 5")
        .expect("WHERE-empty + LIMIT must not panic");
    assert!(none.rows.is_empty(), "WHERE-empty + LIMIT -> empty");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s4_resident_limit_zero_and_huge() {
    // RISK (claim #1): LIMIT 0 -> empty; a HUGE LIMIT (well past len) -> the whole (windowed) set; OFFSET
    // exactly == len -> empty. saturating_add must keep a huge LIMIT from overflowing start+l.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a) VALUES (5),(2),(8),(1)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let col = |res: &RelationalSelectResult| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref o => panic!("expected Int4 got {o:?}"),
            })
            .collect()
    };
    let zero = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a LIMIT 0")
        .expect("LIMIT 0");
    assert!(zero.rows.is_empty(), "LIMIT 0 -> empty");
    // huge LIMIT well within usize but past len -> the whole sorted set.
    let huge = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a LIMIT 1000000000")
        .expect("huge LIMIT");
    assert_eq!(col(&huge), vec![1, 2, 5, 8], "huge LIMIT -> whole set");
    // OFFSET == len -> empty.
    let at_end = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a OFFSET 4")
        .expect("OFFSET == len");
    assert!(at_end.rows.is_empty(), "OFFSET == len -> empty");
    // OFFSET huge + LIMIT huge -> empty (saturating_add must not overflow-panic; start clamps to len).
    let huge_off = e
        .execute_relational_select_text(
            "SELECT a FROM t ORDER BY a LIMIT 999999999 OFFSET 999999999",
        )
        .expect("huge OFFSET + huge LIMIT must not panic");
    assert!(huge_off.rows.is_empty(), "huge OFFSET -> empty");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s4_grouped_nulls_override_with_limit_window() {
    // RISK (claim #3): the explicit-ORDER-BY branch now threads `order_by_nulls_first.to_vec()` into
    // gpu_sort_permutation instead of passing the slice into gpu_sort_result_rows. Identical contents must
    // produce identical placement. A nullable INT group KEY forms a NULL group; with explicit NULLS LAST
    // the NULL group must sort LAST (overriding the ASC default of FIRST), then a LIMIT window must keep
    // the right groups. This is a MULTI-group result so the real sort runs (not the 1-row identity).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE tg (k INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO tg (k,v) VALUES (10,1),(NULL,2),(20,3),(NULL,4),(10,5),(30,6)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tg").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // groups: NULL->2, 10->2, 20->1, 30->1.
    let r = |k: SqlValue, c: i64| vec![k, SqlValue::Int8(c)];
    let n = SqlValue::Null;
    let i = SqlValue::Int4;

    // ASC NULLS LAST: 10,20,30,NULL. LIMIT 2 OFFSET 1 -> [20, 30].
    let last = e
        .execute_resident_expr_select_sql(
            "SELECT k, COUNT(*) FROM tg GROUP BY k ORDER BY k ASC NULLS LAST LIMIT 2 OFFSET 1",
        )
        .expect("grouped ASC NULLS LAST + window");
    assert_eq!(
        last.rows,
        vec![r(i(20), 1), r(i(30), 1)],
        "ASC NULLS LAST window keeps the middle two"
    );

    // ASC NULLS FIRST (default): NULL,10,20,30. LIMIT 2 -> [NULL, 10].
    let first = e
        .execute_resident_expr_select_sql(
            "SELECT k, COUNT(*) FROM tg GROUP BY k ORDER BY k ASC NULLS FIRST LIMIT 2",
        )
        .expect("grouped ASC NULLS FIRST + window");
    assert_eq!(
        first.rows,
        vec![r(n.clone(), 2), r(i(10), 2)],
        "ASC NULLS FIRST window keeps the NULL group then 10"
    );

    // DESC NULLS LAST: 30,20,10,NULL. LIMIT 2 OFFSET 2 -> [10, NULL].
    let desc_last = e
        .execute_resident_expr_select_sql(
            "SELECT k, COUNT(*) FROM tg GROUP BY k ORDER BY k DESC NULLS LAST LIMIT 2 OFFSET 2",
        )
        .expect("grouped DESC NULLS LAST + window");
    assert_eq!(
        desc_last.rows,
        vec![r(i(10), 2), r(n, 2)],
        "DESC NULLS LAST window keeps the tail [10, NULL]"
    );
}

#[test]
fn audit_s4_windowing_math_equals_drain_truncate() {
    // NON-VACUITY + EXHAUSTIVE EQUIVALENCE (no GPU): the NEW windowing formula must equal the OLD
    // drain/truncate for EVERY (len, offset, limit). This is a pure-math model of the production code at
    // engine_expr.rs:5209-5217 and :4657-4663. Fault-inject the formula here (not in production) to prove
    // this test is non-vacuous: e.g. `end = start + l` (no `.min(len)`) would diverge on overflow/clamp
    // cases below, and `start = offset` (no `.min(len)`) would panic-slice.
    fn windowed(len: usize, offset: Option<usize>, limit: Option<usize>) -> (usize, usize) {
        let start = offset.unwrap_or(0).min(len);
        let end = limit.map_or(len, |l| start.saturating_add(l).min(len));
        (start, end) // keep [start, end)
    }
    // The OLD semantics, applied to a vector of `len` elements; returns the kept index RANGE [lo, hi).
    fn drain_truncate(len: usize, offset: Option<usize>, limit: Option<usize>) -> (usize, usize) {
        let start = offset.unwrap_or(0).min(len); // drain(..start)
        let remaining = len - start;
        let kept = match limit {
            Some(l) => l.min(remaining), // truncate(l)
            None => remaining,
        };
        (start, start + kept)
    }
    let lens = [0usize, 1, 2, 3, 5, 10];
    let vals: [Option<usize>; 6] = [
        None,
        Some(0),
        Some(1),
        Some(3),
        Some(usize::MAX), // overflow probe for start+limit
        Some(usize::MAX - 1),
    ];
    for &len in &lens {
        for &offset in &vals {
            for &limit in &vals {
                let w = windowed(len, offset, limit);
                let d = drain_truncate(len, offset, limit);
                assert_eq!(
                    w, d,
                    "windowing != drain/truncate at len={len} offset={offset:?} limit={limit:?}"
                );
                // sanity: the window is a valid slice of [0, len].
                assert!(
                    w.0 <= w.1 && w.1 <= len,
                    "invalid window {w:?} for len={len}"
                );
            }
        }
    }
}

/// S8 fixture: a resident `g (k INT, v INT)` with negatives, a negative group key, and a non-integer
/// AVG (k=2 -> 3.5). Returns `None` (test skips) if the box has no GPU. Groups:
///   k=-1 {100, 0}   cnt2 sum100 avg50    min0   max100
///   k=1  {10,-5,2}  cnt3 sum7   avg2.33  min-5  max10
///   k=2  {4, 3}     cnt2 sum7   avg3.5   min3   max4
///   k=3  {-10}      cnt1 sum-10 avg-10   min-10 max-10
fn s8_resident_g() -> Option<Engine> {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE g (k INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO g (k, v) VALUES (1, 10), (1, -5), (1, 2), (2, 4), (2, 3), (3, -10), (-1, 100), (-1, 0)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("g").unwrap();
    snapshot.device_memory_proof.is_some().then_some(e)
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_s8_bridge_matches_general_grouped_differential() {
    // S8 PROOF: the `&Select`->general BRIDGE (`execute_resident_grouped_via_general`) must be
    // byte-identical to the SQL->Expr general path (`execute_resident_expr_select_sql`) over the grouped
    // int4 matrix. The bridge rebuilds the WHERE predicate from the bound's resolved filters (a THIRD
    // predicate-construction path, vs the route classifier and `map_predicate_node`); this differential
    // proves the reconstruction matches the general path BEFORE the legacy resident-probe grouped
    // methods are deleted. The oracle is the general path, itself already proven == the enumerated route
    // (0/24, doc 22 S8). Filters are single non-equality int4 comparisons (the only filtered shape the
    // route accepts) over all four ops (>=, >, <, <=); the `v >= 0` filter drops a whole group (k=3) so
    // the predicate is load-bearing.
    let Some(e) = s8_resident_g() else { return };

    let aggregates = [
        ("COUNT(*)", "count"),
        ("SUM(v)", "sum"),
        ("AVG(v)", "avg"),
        ("MIN(v)", "min"),
        ("MAX(v)", "max"),
    ];
    let mut shapes: Vec<String> = Vec::new();
    for (agg, name) in aggregates {
        shapes.push(format!("SELECT k, {agg} FROM g GROUP BY k"));
        shapes.push(format!("SELECT k, {agg} FROM g GROUP BY k ORDER BY k"));
        shapes.push(format!("SELECT k, {agg} FROM g GROUP BY k ORDER BY k DESC"));
        shapes.push(format!("SELECT k, {agg} FROM g GROUP BY k ORDER BY {name}"));
        shapes.push(format!(
            "SELECT k, {agg} FROM g GROUP BY k ORDER BY {name} DESC"
        ));
        shapes.push(format!(
            "SELECT k, {agg} FROM g GROUP BY k HAVING {name} >= 3 ORDER BY k"
        ));
        shapes.push(format!(
            "SELECT k, {agg} FROM g GROUP BY k HAVING {name} > 100000 ORDER BY k"
        ));
        shapes.push(format!(
            "SELECT k, {agg} FROM g GROUP BY k ORDER BY k LIMIT 2"
        ));
        shapes.push(format!(
            "SELECT k, {agg} FROM g GROUP BY k ORDER BY k LIMIT 0"
        ));
        shapes.push(format!(
            "SELECT k, {agg} FROM g WHERE v >= 0 GROUP BY k ORDER BY k"
        ));
        shapes.push(format!(
            "SELECT k, {agg} FROM g WHERE v > 0 GROUP BY k ORDER BY k DESC"
        ));
        shapes.push(format!(
            "SELECT k, {agg} FROM g WHERE v < 50 GROUP BY k ORDER BY {name} DESC LIMIT 2"
        ));
        shapes.push(format!(
            "SELECT k, {agg} FROM g WHERE v <= 10 GROUP BY k HAVING {name} >= 0 ORDER BY k"
        ));
    }

    let mut divergences = 0usize;
    for sql in &shapes {
        let Command::Select(select) =
            parse_command(sql).unwrap_or_else(|_| panic!("hand-rolled parse failed: {sql}"))
        else {
            panic!("not a SELECT: {sql}");
        };
        let bridge = e
            .execute_resident_grouped_via_general(&select, None, None)
            .unwrap_or_else(|err| panic!("bridge failed for {sql}: {err:?}"));
        let general = e
            .execute_resident_expr_select_sql(sql)
            .unwrap_or_else(|err| panic!("general failed for {sql}: {err:?}"));
        if bridge.columns != general.columns || bridge.rows != general.rows {
            divergences += 1;
            eprintln!(
                "DIVERGENCE for {sql}\n  bridge.cols={:?}\n  gen.cols   ={:?}\n  bridge.rows={:?}\n  gen.rows   ={:?}",
                bridge.columns, general.columns, bridge.rows, general.rows
            );
        }
        // Both paths must run ON THE GPU (no CPU fallback) for the comparison to be real.
        assert_eq!(
            bridge.executed_target,
            DeviceTarget::Gpu(0),
            "bridge fell off the GPU: {sql}"
        );
        assert_eq!(
            general.executed_target,
            DeviceTarget::Gpu(0),
            "general fell off the GPU: {sql}"
        );
    }
    assert_eq!(
        divergences,
        0,
        "{divergences} bridge-vs-general divergences across {} grouped shapes",
        shapes.len()
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s8_filtered_grouped_having_order_limit() {
    // S8 regression (production routing): a FILTERED grouped aggregate with HAVING + ORDER BY + LIMIT
    // reaches the bridge through the live route dispatch (`execute_relational_select`), runs ON THE GPU,
    // and finalizes sort/HAVING/LIMIT on-device (the legacy probe did this on the HOST). Non-vacuous:
    // the hard-coded rows pin the filtered sums, the HAVING/ORDER/LIMIT window, and SUM(int4)->Int8.
    let Some(e) = s8_resident_g() else { return };
    let Command::Select(select) = parse_command(
        "SELECT k, SUM(v) FROM g WHERE v >= 0 GROUP BY k HAVING sum >= 7 ORDER BY sum DESC LIMIT 2",
    )
    .unwrap() else {
        unreachable!()
    };
    // v>=0 drops (1,-5) and (3,-10): k=1 sum12, k=2 sum7, k=-1 sum100 (k=3 disappears). HAVING sum>=7
    // keeps all three; ORDER BY sum DESC -> 100,12,7; LIMIT 2 -> [k=-1 100, k=1 12].
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(-1), SqlValue::Int8(100)],
            vec![SqlValue::Int4(1), SqlValue::Int8(12)],
        ]
    );
    // The bridge produces the same result called directly as through the dispatch.
    assert_eq!(
        e.execute_resident_grouped_via_general(&select, None, None)
            .unwrap()
            .rows,
        result.rows.into_boxed()
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s8_avg_non_integer_result() {
    // S8 regression: a grouped AVG with a NON-INTEGER result (k=2 -> 3.5) through the live dispatch.
    // AVG yields numeric at AVG_RESULT_SCALE (16). Pins the exact fixed-point AVG so a scale/repr
    // regression is caught (the differential's general oracle could drift; these are absolute).
    let Some(e) = s8_resident_g() else { return };
    let Command::Select(select) =
        parse_command("SELECT k, AVG(v) FROM g GROUP BY k ORDER BY k").unwrap()
    else {
        unreachable!()
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.rows.len(), 4, "four groups: -1, 1, 2, 3");
    // ORDER BY k asc -> rows[0]=k-1 (avg 50.0), rows[2]=k2 (avg 3.5). Both exact at scale 16.
    assert_eq!(
        result.rows[0],
        vec![
            SqlValue::Int4(-1),
            SqlValue::Numeric(Decimal128::parse("50.0000000000000000").unwrap()),
        ]
    );
    assert_eq!(
        result.rows[2],
        vec![
            SqlValue::Int4(2),
            SqlValue::Numeric(Decimal128::parse("3.5000000000000000").unwrap()),
        ]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s8_where_and_or_dnf_matches_general() {
    // S8 DNF-builder coverage: the bridge's `resident_predicate_from_bound_filters` builds an AND-group
    // (from `bound.filters`) and an OR-of-groups (from `bound.filter_groups`). These multi-leaf filters
    // do NOT route to the bridge in production (the grouped route accepts a single leaf only), so call
    // the bridge DIRECTLY and diff against the general path, which builds the same predicate via
    // `map_predicate_node`. Hard-coded expected rows make it non-vacuous (a dropped DNF group would
    // change them). COUNT(*) -> Int8.
    let Some(e) = s8_resident_g() else { return };

    // AND: v>0 AND k<3 keeps (1,10),(1,2),(2,4),(2,3),(-1,100) -> k=-1:1, k=1:2, k=2:2.
    let and_sql = "SELECT k, COUNT(*) FROM g WHERE v > 0 AND k < 3 GROUP BY k ORDER BY k";
    // OR: v>50 OR v<0 keeps (1,-5),(3,-10),(-1,100) -> k=-1:1, k=1:1, k=3:1.
    let or_sql = "SELECT k, COUNT(*) FROM g WHERE v > 50 OR v < 0 GROUP BY k ORDER BY k";

    let expected = [
        (
            and_sql,
            vec![
                vec![SqlValue::Int4(-1), SqlValue::Int8(1)],
                vec![SqlValue::Int4(1), SqlValue::Int8(2)],
                vec![SqlValue::Int4(2), SqlValue::Int8(2)],
            ],
        ),
        (
            or_sql,
            vec![
                vec![SqlValue::Int4(-1), SqlValue::Int8(1)],
                vec![SqlValue::Int4(1), SqlValue::Int8(1)],
                vec![SqlValue::Int4(3), SqlValue::Int8(1)],
            ],
        ),
    ];
    for (sql, want) in expected {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let bridge = e
            .execute_resident_grouped_via_general(&select, None, None)
            .unwrap();
        let general = e.execute_resident_expr_select_sql(sql).unwrap();
        assert_eq!(bridge.columns, general.columns, "{sql}");
        assert_eq!(bridge.rows, general.rows, "{sql}");
        assert_eq!(bridge.rows, want, "{sql}");
        assert_eq!(bridge.executed_target, DeviceTarget::Gpu(0), "{sql}");
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s8_grouped_order_by_aggregate_tie_break() {
    // S8 regression (adopted from the independent audit -- closes the gap that the differential's
    // sabotage exposed: the other s8 tests have no TIES on the ORDER BY aggregate at a LIMIT boundary,
    // so disabling the group-key tie-break left them all green). The general grouped ORDER BY appends
    // the group key ASC as a deterministic tie-break, matching the legacy probe/host group-ASC order.
    // Without it, the order among groups that tie on the aggregate is implementation-defined and a LIMIT
    // would pick a DIFFERENT group (the auditor measured [10,3] vs [20,3] for `ORDER BY count DESC
    // LIMIT 1`). These asserts pin the group-ASC tie order, so they FAIL if the tie-break regresses.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE g3 (k INT, v INT)").unwrap();
    // k=-5: v=-3,-3   -> cnt2 sum-6  ;  k=10: v=10,10,10 -> cnt3 sum30
    // k=20: v=5,10,15 -> cnt3 sum30  ;  k=30: v=10       -> cnt1 sum10  ;  k=40: v=2,8 -> cnt2 sum10
    // Deliberate ties: k=10 & k=20 both cnt3/sum30; k=30 & k=40 both sum10; k=-5 & k=40 both cnt2.
    e.execute_text(
        2,
        "INSERT INTO g3 (k, v) VALUES (10,10),(10,10),(10,10),(20,5),(20,10),(20,15),(30,10),(40,2),(40,8),(-5,-3),(-5,-3)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("g3").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let run = |sql: &str| -> Vec<Vec<SqlValue>> {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let result = e.execute_relational_select(&select).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0), "{sql}");
        result.rows.into_boxed()
    };

    // COUNT ties: k=10 & k=20 both 3 -> ORDER BY count DESC breaks by group ASC -> k=10 first.
    assert_eq!(
        run("SELECT k, COUNT(*) FROM g3 GROUP BY k ORDER BY count DESC LIMIT 1"),
        vec![vec![SqlValue::Int4(10), SqlValue::Int8(3)]]
    );
    assert_eq!(
        run("SELECT k, COUNT(*) FROM g3 GROUP BY k ORDER BY count DESC LIMIT 2"),
        vec![
            vec![SqlValue::Int4(10), SqlValue::Int8(3)],
            vec![SqlValue::Int4(20), SqlValue::Int8(3)],
        ]
    );
    // SUM ties: k=10 & k=20 both 30 -> ORDER BY sum DESC LIMIT 1 -> k=10 (group ASC). SUM(int4)->Int8.
    assert_eq!(
        run("SELECT k, SUM(v) FROM g3 GROUP BY k ORDER BY sum DESC LIMIT 1"),
        vec![vec![SqlValue::Int4(10), SqlValue::Int8(30)]]
    );
    // SUM ascending ties: -6(k-5), then 10(k=30 & k=40) -> group ASC -> k=30 before k=40. LIMIT 2.
    assert_eq!(
        run("SELECT k, SUM(v) FROM g3 GROUP BY k ORDER BY sum ASC LIMIT 2"),
        vec![
            vec![SqlValue::Int4(-5), SqlValue::Int8(-6)],
            vec![SqlValue::Int4(30), SqlValue::Int8(10)],
        ]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s8_grouped_materialized_view_via_bridge() {
    // S8 regression (adopted from the independent audit -- the CTAS/view deliverable, the whole reason a
    // `&Select`->general BRIDGE was built instead of routing only the text entry). A grouped
    // MATERIALIZED VIEW ... WITH DATA runs its grouped SELECT through `execute_relational_select(&Select)`
    // -> the bridge AT CREATE TIME (no raw SQL text), so this proves a grouped view/CTAS materializes
    // correctly on the GPU. The auditor confirmed these rows are byte-identical to the parent (probe).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE base (k INT, v INT)")
        .unwrap();
    // k=1 {10,20} sum30 cnt2 ; k=2 {5,5,5} sum15 cnt3 ; k=3 {-7,100} sum93 cnt2 (v>=5 drops -7 -> cnt1).
    e.execute_text(
        2,
        "INSERT INTO base (k, v) VALUES (1,10),(1,20),(2,5),(2,5),(3,-7),(3,100),(2,5)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("base").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let readback = |e: &Engine, sql: &str| -> Vec<Vec<SqlValue>> {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        e.execute_relational_select(&select)
            .unwrap()
            .rows
            .into_boxed()
    };

    // Grouped matview: the SELECT runs through the bridge at create time; readback returns stored rows.
    e.execute_text(
        10,
        "CREATE MATERIALIZED VIEW mg AS SELECT k, SUM(v) FROM base GROUP BY k ORDER BY k WITH DATA",
    )
    .unwrap();
    let mg = e.relational_catalog_materialized_view("mg").unwrap();
    assert_eq!(mg.columns[0].ty, SqlType::Int4);
    assert_eq!(mg.columns[1].ty, SqlType::Int8);
    assert_eq!(mg.columns[1].type_oid, 20);
    assert_eq!(
        readback(&e, "SELECT * FROM mg"),
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(30)],
            vec![SqlValue::Int4(2), SqlValue::Int8(15)],
            vec![SqlValue::Int4(3), SqlValue::Int8(93)],
        ]
    );
    // Filtered grouped matview (the int4_filtered_grouped_aggregate route through the bridge).
    e.execute_text(
        20,
        "CREATE MATERIALIZED VIEW mf AS SELECT k, COUNT(*) FROM base WHERE v >= 5 GROUP BY k ORDER BY k WITH DATA",
    )
    .unwrap();
    assert_eq!(
        readback(&e, "SELECT * FROM mf"),
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(3)],
            vec![SqlValue::Int4(3), SqlValue::Int8(1)],
        ]
    );
    // Grouped matview with HAVING + ORDER BY aggregate DESC -- the on-device finalization via the bridge.
    e.execute_text(
        30,
        "CREATE MATERIALIZED VIEW mh AS SELECT k, SUM(v) FROM base GROUP BY k HAVING sum >= 15 ORDER BY sum DESC WITH DATA",
    )
    .unwrap();
    assert_eq!(
        readback(&e, "SELECT * FROM mh"),
        vec![
            vec![SqlValue::Int4(3), SqlValue::Int8(93)],
            vec![SqlValue::Int4(1), SqlValue::Int8(30)],
            vec![SqlValue::Int4(2), SqlValue::Int8(15)],
        ]
    );
    // REFRESH re-runs the grouped SELECT through the bridge; the stored rows are unchanged.
    e.execute_text(40, "REFRESH MATERIALIZED VIEW mg").unwrap();
    assert_eq!(
        readback(&e, "SELECT * FROM mg"),
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(30)],
            vec![SqlValue::Int4(2), SqlValue::Int8(15)],
            vec![SqlValue::Int4(3), SqlValue::Int8(93)],
        ]
    );
}
