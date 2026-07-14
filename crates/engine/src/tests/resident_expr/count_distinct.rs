use crate::Engine;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{Decimal128, SqlValue};

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
