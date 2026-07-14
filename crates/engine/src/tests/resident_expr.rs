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
    // RISK (claim #3): the explicit-ORDER-BY branch threads `order_by_nulls_first.to_vec()` into the
    // `gpu_sort_permutation` setup. The owned copy must preserve identical placement. A nullable INT group
    // KEY forms a NULL group; with explicit NULLS LAST
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
