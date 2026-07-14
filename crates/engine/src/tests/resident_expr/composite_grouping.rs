use crate::Engine;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{Decimal128, SqlType, SqlValue};

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
