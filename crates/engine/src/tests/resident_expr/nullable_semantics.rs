use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};
use crate::{parse_command, Command, Engine};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{Decimal128, SqlValue};

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_where_excludes_null_operands_and_projection_carries_null() {
    // M3 (doc 21) Slice D + C: a WHERE comparison over a NULLABLE column evaluates to UNKNOWN for a NULL
    // operand ON THE GPU (the leaf mask is AND'd with the column's validity bitmap) -> the row is NOT
    // selected; and a projected nullable column carries SqlValue::Null through the gather. Column a is
    // nullable (NULL at i%4==0), b is the constant 100 (non-null). a[i]=i for the non-null rows.
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
