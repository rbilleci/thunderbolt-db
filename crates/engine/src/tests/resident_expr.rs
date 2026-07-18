//! Engine-level tests for the general GPU executor (`engine_expr`, Charter rule 2,
//! docs/architecture/17-general-gpu-executor.md): a `SELECT` filtered by a general expression tree
//! evaluated on the GPU via the device interpreter, with rows materialized on-device. Distinct from
//! the (frozen) enumerated `resident_probe` shape methods — here the unit of execution is an
//! expression, not a recognized shape.

use super::*;

use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};

mod checked_arithmetic;
mod composite_grouping;
mod count_distinct;
mod grouped_multi_aggregate;
mod nongrouped_ordering;
mod nullable_grouping;
mod nullable_semantics;
mod programmatic_predicates;
mod scalar_aggregates;
mod single_key_grouping;
mod sql_scalar_predicates;

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_full_table_no_where() {
    // No WHERE clause = a full-table scan (indices 0..row_count): aggregates reduce over every row and
    // projection materializes every row, all on the general GPU executor.
    let mut e = Engine::new_local_test_engine();
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
fn gpu_grouped_by_text_key() {
    // GROUP BY a TEXT (varlen) key -- the kernel FNV-1a-hashes the bytes, claims a b128
    // (representative_row_idx, hash) in slot_keys_i128 via atom.cas.b128 with a full-text
    // VERIFY-ON-LOST-CAS, so same-text rows COLLAPSE into one group and hash collisions never merge.
    // Covers duplicates (apple x3), an EMPTY string, different lengths, and a SHARED PREFIX (app vs
    // apple) to exercise the length-check + byte-compare in the verify. Result key is read host-side
    // from the representative row.
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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

// group counts for `t` below: g1=3, g2=1, g3=2, g4=4, g5=1.
const GROUPED_CLAUSE_ROWS: &str =
    "(1,10),(1,20),(1,30),(2,5),(3,7),(3,8),(4,1),(4,2),(4,3),(4,4),(5,99)";

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_order_by_and_limit() {
    // ORDER BY (the key DESC, and an AGGREGATE DESC) + LIMIT/OFFSET windowed ON-DEVICE (a slice of the
    // gpu_sort_permutation index vector, gathering only the kept window) on the Expr path.
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
fn gpu_grouped_by_uuid_key() {
    // GROUP BY a UUID (i128) key via atom.cas.b128; output sorts by canonical/memcmp byte order. Uses
    // early-byte AND late-byte differences (exercises the sort + the full 128-bit key equality), and
    // INCLUDES the uuid whose LE i128 == EMPTY128 (i128::MIN) -> the DEDICATED slot path for i128 keys.
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
fn gpu_grouped_sum_avg_over_numeric_value() {
    // GROUP BY an int4 key, SUM/AVG over a NUMERIC(20,2) value -> the single-level kernel reads the
    // i128 mantissa (16-byte stride) and accumulates i128 per slot; the result carries the column
    // scale (2). Constructed oracle:
    //   g=1 -> {10.50, 20.25, -3.75}  sum 27.00
    //   g=2 -> {100.00, -50.50}        sum 49.50
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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

    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e2 = Engine::new_local_test_engine();
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
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_expr_select_evaluates_is_null_and_is_not_null_via_validity_bitmap() {
    // `WHERE v IS NULL` / `IS NOT NULL` runs ON THE GPU (M3 -- doc 21): the column's NULL validity
    // bitmap (slice 2a) feeds the SAME bitmap->mask kernel as a bool column, pointed at the validity
    // bitmap. GPU-native oracle = CONSTRUCTION (we know which rows are NULL by the insert rule). Projects
    // `id` (which has no NULLs) for the surviving rows.
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    // RISK (claim #3): the explicit-ORDER-BY branch threads `order_by_nulls_first.to_vec()` into the
    // `gpu_sort_permutation` setup. The owned copy must preserve identical placement. A nullable INT group
    // KEY forms a NULL group; with explicit NULLS LAST
    // the NULL group must sort LAST (overriding the ASC default of FIRST), then a LIMIT window must keep
    // the right groups. This is a MULTI-group result so the real sort runs (not the 1-row identity).
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
    let mut e = Engine::new_local_test_engine();
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
