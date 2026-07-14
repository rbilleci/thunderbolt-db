use super::{gpu_available, select};
use crate::{Engine, RelationalSelectResult};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;

/// Sort grouped result rows by their key cell (grouped output order is arbitrary per SQL without
/// ORDER BY, so the differentials compare SORTED row sets).
fn sorted_rows(result: &RelationalSelectResult) -> Vec<Vec<SqlValue>> {
    let mut rows = result.rows.clone().into_boxed();
    rows.sort_by(|a, b| crate::rel_exec_helpers::compare_sql_values(&a[0], &b[0]));
    rows
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_grouped_over_budget_two_level_merge() {
    // S-E.3: GROUP BY over an over-budget table streams TWO-LEVEL — per-chunk device grouped partials,
    // concat, one final device merge (COUNT folds as SUM(count), SUM as SUM(sum), MIN/MAX as the
    // extreme). Groups SPAN chunks (g = i % 7 over 1500 rows, many chunks), so a broken merge
    // double-counts or drops cross-chunk groups.
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (g INT, a INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({}, {i})", i % 7));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (g, a) VALUES {values}"))
        .unwrap();
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    // Closed-form per-group expectations over i in [0,N) with g = i % 7.
    let group_rows = |agg: &dyn Fn(i64) -> SqlValue| -> Vec<Vec<SqlValue>> {
        (0..7)
            .map(|g| vec![SqlValue::Int4(g as i32), agg(g)])
            .collect()
    };
    let count_of = |g: i64| (i64::from(N) + 6 - g) / 7; // ceil((N-g)/7)
    let sum_of = |g: i64| {
        let n = count_of(g);
        n * g + 7 * n * (n - 1) / 2 // sum of g, g+7, g+14, ...
    };
    let max_of = |g: i64| g + 7 * (count_of(g) - 1);

    let count = e
        .execute_relational_select(&select("SELECT g, COUNT(*) FROM big GROUP BY g"))
        .unwrap();
    assert_eq!(
        sorted_rows(&count),
        group_rows(&|g| SqlValue::Int8(count_of(g))),
        "grouped COUNT across chunks"
    );
    assert_eq!(count.executed_target, DeviceTarget::Gpu(0));
    assert!(e.streaming_fold_hits() >= 1, "grouped fold fired");
    assert!(
        e.streaming_fold_chunks() > 1,
        "multi-chunk grouped fold, got {}",
        e.streaming_fold_chunks()
    );
    assert!(
        e.streaming_fold_peak_chunk_bytes() <= budget,
        "peak bounded"
    );

    let sum = e
        .execute_relational_select(&select("SELECT g, SUM(a) FROM big GROUP BY g"))
        .unwrap();
    assert_eq!(
        sorted_rows(&sum),
        group_rows(&|g| SqlValue::Int8(sum_of(g))),
        "grouped SUM across chunks (partials merged, Int8-narrowed)"
    );

    let min = e
        .execute_relational_select(&select("SELECT g, MIN(a) FROM big GROUP BY g"))
        .unwrap();
    assert_eq!(
        sorted_rows(&min),
        group_rows(&|g| SqlValue::Int4(g as i32)),
        "grouped MIN across chunks"
    );
    let max = e
        .execute_relational_select(&select("SELECT g, MAX(a) FROM big GROUP BY g"))
        .unwrap();
    assert_eq!(
        sorted_rows(&max),
        group_rows(&|g| SqlValue::Int4(max_of(g) as i32)),
        "grouped MAX across chunks"
    );

    // WHERE + GROUP BY: the predicate filters on-device per chunk BEFORE grouping.
    let filtered = e
        .execute_relational_select(&select(
            "SELECT g, COUNT(*) FROM big WHERE a >= 700 GROUP BY g",
        ))
        .unwrap();
    let filtered_count_of = |g: i64| (700..i64::from(N)).filter(|i| i % 7 == g).count() as i64;
    assert_eq!(
        sorted_rows(&filtered),
        group_rows(&|g| SqlValue::Int8(filtered_count_of(g))),
        "device-filtered grouped COUNT"
    );

    // Differential: the CPU pinned path (budget cleared) agrees on the sorted row set.
    e.clear_relational_residency_budget_bytes(0);
    let cpu = e
        .execute_relational_select(&select("SELECT g, SUM(a) FROM big GROUP BY g"))
        .unwrap();
    assert_eq!(
        sorted_rows(&cpu),
        sorted_rows(&sum),
        "GPU streaming == CPU oracle"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_distinct_over_budget_set_union() {
    // S-E.3 DISTINCT: per-chunk device distinct keys, concat, final device re-distinct = SET UNION.
    // Keys repeat across chunks (i % 13), so a broken union duplicates or drops values.
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (v INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({})", i % 13));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (v) VALUES {values}"))
        .unwrap();
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    let result = e
        .execute_relational_select(&select("SELECT DISTINCT v FROM big"))
        .unwrap();
    let expected: Vec<Vec<SqlValue>> = (0..13).map(|v| vec![SqlValue::Int4(v)]).collect();
    assert_eq!(
        sorted_rows(&result),
        expected,
        "distinct set union across chunks"
    );
    assert_eq!(result.columns.len(), 1, "count column dropped");
    assert!(e.streaming_fold_hits() >= 1, "distinct fold fired");
    assert!(
        e.streaming_fold_chunks() > 1,
        "multi-chunk distinct, got {}",
        e.streaming_fold_chunks()
    );

    // Differential vs the CPU pinned path.
    e.clear_relational_residency_budget_bytes(0);
    let cpu = e
        .execute_relational_select(&select("SELECT DISTINCT v FROM big"))
        .unwrap();
    assert_eq!(sorted_rows(&cpu), expected, "CPU oracle distinct set");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_grouped_compaction_and_over_cardinality_defer() {
    // S-E.3 compaction + the honest over-cardinality defer. (1) 100 true groups (2.0KB of partials at
    // the 6c-0(c) sizes: 4B key + 16B Numeric count, JUST under the 2KB chunk target) over ~3 chunks:
    // after chunk 2 the accumulator holds ~200 partial rows (4KB, OVER target) so it MUST compact
    // mid-scan via the device merge back to 100 — and still produce exact counts. (2) an all-unique
    // key (1500 groups, 30KB of true partials) cannot compact below the target -> the fold DEFERS to
    // the CPU path (correct rows, no hit counted).
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE mid (g INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({})", i % 100));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO mid (g) VALUES {values}"))
        .unwrap();
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    let result = e
        .execute_relational_select(&select("SELECT g, COUNT(*) FROM mid GROUP BY g"))
        .unwrap();
    let expected: Vec<Vec<SqlValue>> = (0..100)
        .map(|g| {
            let count = (0..N).filter(|i| i % 100 == g).count() as i64;
            vec![SqlValue::Int4(g), SqlValue::Int8(count)]
        })
        .collect();
    assert_eq!(
        sorted_rows(&result),
        expected,
        "100-group counts exact THROUGH the mid-scan compaction"
    );
    let hits_after_compaction = e.streaming_fold_hits();
    assert!(hits_after_compaction >= 1, "compacted grouped fold fired");

    // (2) Over-cardinality: an all-unique key can never compact under the budget -> honest defer.
    seq += 1;
    e.execute_text(seq, "CREATE TABLE uniq (g INT)").unwrap();
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO uniq (g) VALUES {values}"))
        .unwrap();
    let deferred = e
        .execute_relational_select(&select("SELECT g, COUNT(*) FROM uniq GROUP BY g"))
        .unwrap();
    assert_eq!(
        deferred.rows.len(),
        N as usize,
        "over-cardinality grouped result served correctly (by the CPU defer)"
    );
    assert_eq!(
        e.streaming_fold_hits(),
        hits_after_compaction,
        "the over-cardinality query must DEFER (no streaming hit)"
    );
    // Audit Finding 1 regression gate: the merge must never be the thing that busts the budget — the
    // over-cardinality accumulator (1500 x 12B partials from 4B rows) DEFERS WITHOUT uploading, so the
    // peak transient device residency stays bounded even though the partials outgrew the source chunk.
    assert!(
        e.streaming_fold_peak_chunk_bytes() <= budget,
        "peak transient residency ({}) must stay <= budget ({budget}) through the over-cardinality defer",
        e.streaming_fold_peak_chunk_bytes()
    );
}
