use crate::{average_sql_value, Engine};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;

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
