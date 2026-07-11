//! STRATA S-E.1 — out-of-core streaming scalar reductions (ADR-012 / PLAN S-E).
//!
//! A table whose bytes exceed the configured per-GPU residency budget has no all-resident representation,
//! so today its aggregate would de-elide to the CPU host engine (the ADR-006 charter violation). The
//! streaming fold serves `COUNT(*)`/`SUM`/`MIN`/`MAX` OUT-OF-CORE: the visible rows are chunked to the
//! budget, each chunk uploaded + reduced ON THE DEVICE, partials combined host-side (control plane). The
//! non-vacuity proof is the fired counter + `streaming_fold_chunks > 1` (a genuine multi-chunk fold) +
//! `streaming_fold_peak_chunk_bytes <= budget` (only one chunk ever resident). The differential is the
//! SAME engine's CPU-pinned answer with the budget cleared.

use super::*;

/// Probe for a usable GPU by populating a throwaway table and checking the retained device proof, then
/// drop it (so it leaves no residency / budget footprint). Returns false to SKIP a GPU test off-box.
fn gpu_available(e: &mut Engine, seq: &mut u64) -> bool {
    *seq += 1;
    e.execute_text(*seq, "CREATE TABLE __se_probe (x INT)")
        .unwrap();
    *seq += 1;
    e.execute_text(*seq, "INSERT INTO __se_probe VALUES (1)")
        .unwrap();
    let snapshot = e
        .populate_relational_residency_snapshot("__se_probe")
        .unwrap();
    let available = snapshot.device_memory_proof.is_some();
    *seq += 1;
    e.execute_text(*seq, "DROP TABLE __se_probe").unwrap();
    available
}

fn select(sql: &str) -> Select {
    let Command::Select(select) = parse_command(sql).unwrap() else {
        panic!("not a SELECT: {sql}");
    };
    select
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_reduction_over_budget_stays_on_device_out_of_core() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }

    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();

    // A budget FAR below the table's ~12 KB of int4 payload -> the fold must chunk (peak residency stays
    // at one chunk, ~budget/2, never the whole table).
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    // COUNT(*) -> N, served by the streaming fold (multi-chunk, bounded residency).
    let count = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
        .unwrap();
    assert_eq!(
        count.rows,
        vec![vec![SqlValue::Int8(i64::from(N))]],
        "COUNT(*) over the over-budget table"
    );
    assert_eq!(count.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(count.fallback_reason, None);
    assert!(
        e.streaming_fold_hits() >= 1,
        "the streaming fold must have fired (not the CPU host engine)"
    );
    assert!(
        e.streaming_fold_chunks() > 1,
        "the fold must be MULTI-CHUNK over an over-budget table (genuine out-of-core), got {}",
        e.streaming_fold_chunks()
    );
    let peak = e.streaming_fold_peak_chunk_bytes();
    assert!(
        peak <= budget && peak > 0,
        "peak single-chunk device residency ({peak}) must stay within the budget ({budget}) even though \
         the whole table dwarfs it"
    );

    // SUM(a) = 0+1+..+(N-1); the executor returns Int8 for SUM(int4), combined across chunks.
    let expected_sum: i64 = (0..i64::from(N)).sum();
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    assert_eq!(sum.rows, vec![vec![SqlValue::Int8(expected_sum)]], "SUM(a)");

    // MIN / MAX preserve the int4 type and fold via the extreme.
    assert_eq!(
        e.execute_relational_select(&select("SELECT MIN(a) FROM big"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(0)]],
        "MIN(a)"
    );
    assert_eq!(
        e.execute_relational_select(&select("SELECT MAX(a) FROM big"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(N - 1)]],
        "MAX(a)"
    );

    // The WHERE predicate runs ON THE DEVICE per chunk (host stages bytes, never filters): a >= 1000
    // survives for a in [1000, N).
    let filtered = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM big WHERE a >= 1000"))
        .unwrap();
    assert_eq!(
        filtered.rows,
        vec![vec![SqlValue::Int8(i64::from(N) - 1000)]],
        "device-filtered COUNT(*)"
    );

    // Differential: the SAME engine on the CPU-pinned path (budget cleared) must produce IDENTICAL rows.
    e.clear_relational_residency_budget_bytes(0);
    let hits_before_cpu = e.streaming_fold_hits();
    let cpu_sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    assert_eq!(
        cpu_sum.rows, sum.rows,
        "GPU streaming SUM must equal the CPU oracle SUM"
    );
    assert_eq!(
        e.streaming_fold_hits(),
        hits_before_cpu,
        "with the budget cleared the streaming fold must NOT fire (CPU path)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_reduction_bigint_sum_combines_as_numeric() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE amounts (v BIGINT)")
        .unwrap();
    const N: i64 = 800;
    // Values large enough that the running total exceeds i64 conceptually is unnecessary here — the point
    // is that SUM(int8) returns NUMERIC per chunk, exercising the scale-aligned Decimal128 checked_add
    // COMBINE across chunks.
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({})", 1_000_000 + i));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO amounts (v) VALUES {values}"))
        .unwrap();
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    let expected: i128 = (0..N).map(|i| 1_000_000_i128 + i as i128).sum();
    let sum = e
        .execute_relational_select(&select("SELECT SUM(v) FROM amounts"))
        .unwrap();
    assert_eq!(
        sum.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(expected, 0))]],
        "SUM(bigint) -> numeric, combined via Decimal128 across chunks"
    );
    assert!(e.streaming_fold_hits() >= 1, "streaming fold fired");
    assert!(
        e.streaming_fold_chunks() > 1,
        "multi-chunk numeric SUM combine, got {}",
        e.streaming_fold_chunks()
    );

    // Streaming SUM(bigint) STRICTLY EXTENDS coverage: the CPU host reduction cannot serve it (the host
    // `int4_aggregate_value` extracts an i32 and hard-errors on an int8 column), so with the budget
    // cleared the same query is a clean error. The streaming device fold is the ONLY engine that answers
    // it — proving the charter win is real, not a re-route of an already-supported host shape.
    e.clear_relational_residency_budget_bytes(0);
    assert!(
        e.execute_relational_select(&select("SELECT SUM(v) FROM amounts"))
            .is_err(),
        "the CPU host path cannot compute SUM(bigint); the streaming fold is the only path that can"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_reduction_null_heavy_table_stays_bounded() {
    // Audit Finding 1 regression gate: a NULL cell still occupies its fixed device slot, so a null-heavy
    // over-budget table must STILL chunk. Before the fix (chunk sizing by logical value bytes, 0 for
    // NULL) the whole table accumulated into one chunk and the out-of-core bound broke. Also exercises
    // MIN/MAX skipping NULL across chunks (the min of the sparse non-null values, not NULL).
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE sparse (a INT)").unwrap();
    const NULLS: i32 = 1500;
    let mut values = String::new();
    for i in 0..NULLS {
        if i > 0 {
            values.push(',');
        }
        values.push_str("(NULL)");
    }
    // Three sparse non-null values: MIN = 2, MAX = 9, SUM = 18.
    values.push_str(",(7),(2),(9)");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO sparse (a) VALUES {values}"))
        .unwrap();
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    let total = i64::from(NULLS) + 3;
    assert_eq!(
        e.execute_relational_select(&select("SELECT COUNT(*) FROM sparse"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int8(total)]],
        "COUNT(*) counts NULL rows"
    );
    assert!(
        e.streaming_fold_chunks() > 1,
        "a null-heavy over-budget table must still chunk (NULL slots counted), got {}",
        e.streaming_fold_chunks()
    );
    assert!(
        e.streaming_fold_peak_chunk_bytes() <= budget,
        "peak residency stays within the budget for null-heavy data"
    );
    // MIN/MAX skip the NULLs (PG); SUM over the sparse non-nulls.
    assert_eq!(
        e.execute_relational_select(&select("SELECT MIN(a) FROM sparse"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(2)]],
        "MIN skips NULL"
    );
    assert_eq!(
        e.execute_relational_select(&select("SELECT MAX(a) FROM sparse"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(9)]],
        "MAX skips NULL"
    );
    assert_eq!(
        e.execute_relational_select(&select("SELECT SUM(a) FROM sparse"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int8(18)]],
        "SUM over the sparse non-nulls"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_reduction_empty_table_pg_semantics() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE hollow (a INT)").unwrap();
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    // PG: over ZERO rows COUNT(*) is 0 but SUM/MIN/MAX are NULL (never 0). The fold runs one empty chunk.
    assert_eq!(
        e.execute_relational_select(&select("SELECT COUNT(*) FROM hollow"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int8(0)]],
        "COUNT(*) over empty = 0"
    );
    assert_eq!(
        e.execute_relational_select(&select("SELECT SUM(a) FROM hollow"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Null]],
        "SUM over empty = NULL"
    );
    assert_eq!(
        e.execute_relational_select(&select("SELECT MIN(a) FROM hollow"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Null]],
        "MIN over empty = NULL"
    );
    assert_eq!(
        e.execute_relational_select(&select("SELECT MAX(a) FROM hollow"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Null]],
        "MAX over empty = NULL"
    );
    assert!(
        e.streaming_fold_hits() >= 1,
        "streaming fold fired for the empty table"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_projection_over_budget_filters_on_device() {
    // S-E.2: a filtered PROJECTION over an over-budget table streams — each chunk's WHERE + column gather
    // run on the device, survivors CONCAT across chunks (scan order == the CPU pinned path's seq order).
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    // Filtered single-column projection spanning many chunks.
    let result = e
        .execute_relational_select(&select("SELECT a FROM big WHERE a >= 1000"))
        .unwrap();
    let expected: Vec<Vec<SqlValue>> = (1000..N).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        result.rows.clone().into_boxed(),
        expected,
        "device-filtered projection"
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert!(e.streaming_fold_hits() >= 1, "the projection fold fired");
    assert!(
        e.streaming_fold_chunks() > 1,
        "multi-chunk projection over the over-budget table, got {}",
        e.streaming_fold_chunks()
    );
    assert!(
        e.streaming_fold_peak_chunk_bytes() <= budget,
        "peak residency bounded"
    );

    // Multi-column `SELECT *` point filter.
    let star = e
        .execute_relational_select(&select("SELECT * FROM big WHERE a = 7"))
        .unwrap();
    assert_eq!(
        star.rows.clone().into_boxed(),
        vec![vec![SqlValue::Int4(7), SqlValue::Int4(14)]],
        "SELECT * survivor row"
    );

    // Differential: the CPU pinned path (budget cleared) returns the identical rows in the same order.
    e.clear_relational_residency_budget_bytes(0);
    let cpu = e
        .execute_relational_select(&select("SELECT a FROM big WHERE a >= 1000"))
        .unwrap();
    assert_eq!(
        cpu.rows.clone().into_boxed(),
        expected,
        "GPU streaming projection == CPU oracle"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_projection_limit_offset_windows_and_early_exits() {
    // S-E.2: LIMIT/OFFSET window the concatenated survivor stream across chunks; a satisfied LIMIT stops
    // the scan EARLY (chunks-run proves the tail was never staged). LIMIT without ORDER BY is any-N-rows
    // per SQL; this engine's scan order is the deterministic seq order, matching the CPU pinned path.
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
        .unwrap();
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    // EARLY EXIT: LIMIT 5 must fill from the FIRST chunk — exactly one chunk runs.
    let chunks_before = e.streaming_fold_chunks();
    let limited = e
        .execute_relational_select(&select("SELECT a FROM big LIMIT 5"))
        .unwrap();
    let expected_first5: Vec<Vec<SqlValue>> = (0..5).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        limited.rows.clone().into_boxed(),
        expected_first5,
        "LIMIT 5 rows"
    );
    assert_eq!(
        e.streaming_fold_chunks() - chunks_before,
        1,
        "a satisfied LIMIT must stop the scan after ONE chunk (early exit)"
    );

    // Cross-chunk OFFSET+LIMIT windowing: survivors a >= 10 in scan order are 10..1500; skip 500 ->
    // start at 510; take 700 -> [510, 1210). The window spans several chunks.
    let windowed = e
        .execute_relational_select(&select(
            "SELECT a FROM big WHERE a >= 10 LIMIT 700 OFFSET 500",
        ))
        .unwrap();
    let expected_window: Vec<Vec<SqlValue>> =
        (510..1210).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        windowed.rows.clone().into_boxed(),
        expected_window,
        "cross-chunk OFFSET+LIMIT window"
    );

    // 6c-0 coverage (audit LOW): OFFSET WITHOUT LIMIT — unbounded take, device-sliced [offset, len).
    let offset_only = e
        .execute_relational_select(&select("SELECT a FROM big OFFSET 1495"))
        .unwrap();
    let expected_tail: Vec<Vec<SqlValue>> = (1495..N).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        offset_only.rows.clone().into_boxed(),
        expected_tail,
        "OFFSET without LIMIT"
    );
    // 6c-0 coverage (audit LOW): LIMIT 0 — zero chunks, empty result, no error.
    let zero = e
        .execute_relational_select(&select("SELECT a FROM big LIMIT 0"))
        .unwrap();
    assert!(zero.rows.is_empty(), "LIMIT 0 is the empty result");

    // Differential vs the CPU pinned path for the same windowed query.
    e.clear_relational_residency_budget_bytes(0);
    let cpu = e
        .execute_relational_select(&select(
            "SELECT a FROM big WHERE a >= 10 LIMIT 700 OFFSET 500",
        ))
        .unwrap();
    assert_eq!(
        cpu.rows.clone().into_boxed(),
        expected_window,
        "GPU streaming window == CPU oracle window"
    );
}

/// Sort grouped result rows by their key cell (grouped output order is arbitrary per SQL without
/// ORDER BY, so the differentials compare SORTED row sets).
/// Disable chunk-class ENTRY for a store-driven gate test (6c-1/6c-3/P2 machinery), restoring
/// on drop (the GPU suite is --test-threads=1, so set/restore is race-free).
struct ClassEntryDisabled;
impl ClassEntryDisabled {
    fn new() -> Self {
        crate::engine_streaming_exec::CHUNK_CLASS_ENTRY_ENABLED_TEST
            .store(false, std::sync::atomic::Ordering::Relaxed);
        ClassEntryDisabled
    }
}
impl Drop for ClassEntryDisabled {
    fn drop(&mut self) {
        crate::engine_streaming_exec::CHUNK_CLASS_ENTRY_ENABLED_TEST
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

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
    let mut e = Engine::new_local();
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
    let mut e = Engine::new_local();
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
    let mut e = Engine::new_local();
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

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_ordered_top_n_across_chunks() {
    // S-E.4: ORDER BY + LIMIT streams as a device top-N fold — each chunk's device-sorted local
    // top-(offset+limit) run concats, compaction re-sorts + re-windows on the device, and ONE final
    // device sort + the real window produces the answer. The global top-N spans chunks (ascending
    // values inserted in scan order, so the DESC winners live in the LAST chunk — a first-chunk-only
    // fold would answer wrongly).
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
        .unwrap();
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    // ASC top-5 = the first chunk's head; DESC top-5 = the LAST chunk's tail (cross-chunk winner).
    let asc = e
        .execute_relational_select(&select("SELECT a FROM big ORDER BY a LIMIT 5"))
        .unwrap();
    let expected_asc: Vec<Vec<SqlValue>> = (0..5).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(asc.rows.clone().into_boxed(), expected_asc, "ASC top-5");
    assert_eq!(asc.executed_target, DeviceTarget::Gpu(0));
    assert!(e.streaming_fold_hits() >= 1, "ordered fold fired");
    assert!(
        e.streaming_fold_chunks() > 1,
        "multi-chunk ordered fold, got {}",
        e.streaming_fold_chunks()
    );
    assert!(
        e.streaming_fold_peak_chunk_bytes() <= budget,
        "peak bounded"
    );

    let desc = e
        .execute_relational_select(&select("SELECT a FROM big ORDER BY a DESC LIMIT 5"))
        .unwrap();
    let expected_desc: Vec<Vec<SqlValue>> =
        (0..5).map(|i| vec![SqlValue::Int4(N - 1 - i)]).collect();
    assert_eq!(
        desc.rows.clone().into_boxed(),
        expected_desc,
        "DESC top-5 from the last chunk"
    );

    // OFFSET windows on the device in the final pass.
    let offset = e
        .execute_relational_select(&select("SELECT a FROM big ORDER BY a LIMIT 5 OFFSET 7"))
        .unwrap();
    let expected_offset: Vec<Vec<SqlValue>> = (7..12).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        offset.rows.clone().into_boxed(),
        expected_offset,
        "OFFSET+LIMIT window"
    );

    // WHERE + ORDER BY + LIMIT: predicate on-device per chunk, then the ordered window.
    let filtered = e
        .execute_relational_select(&select(
            "SELECT a FROM big WHERE a >= 700 ORDER BY a LIMIT 3",
        ))
        .unwrap();
    let expected_filtered: Vec<Vec<SqlValue>> =
        (700..703).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        filtered.rows.clone().into_boxed(),
        expected_filtered,
        "filtered ordered window"
    );

    // COMPACTION: LIMIT 400 -> each chunk contributes up to 400 run rows (1600B), the accumulator
    // crosses the 2KB target after chunk 2 and must device-compact — the result stays exact.
    let compacted = e
        .execute_relational_select(&select("SELECT a FROM big ORDER BY a LIMIT 400"))
        .unwrap();
    let expected_top400: Vec<Vec<SqlValue>> = (0..400).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        compacted.rows.clone().into_boxed(),
        expected_top400,
        "top-400 exact THROUGH the mid-scan compaction"
    );

    // Differential vs the CPU pinned path.
    e.clear_relational_residency_budget_bytes(0);
    let cpu = e
        .execute_relational_select(&select("SELECT a FROM big ORDER BY a DESC LIMIT 5"))
        .unwrap();
    assert_eq!(
        cpu.rows.clone().into_boxed(),
        expected_desc,
        "CPU oracle DESC top-5"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_ordered_unbounded_fits_or_defers() {
    // S-E.4 unbounded ORDER BY: a selective WHERE whose survivor set fits the budget streams (per-chunk
    // plain filter/project, ONE final device sort); a survivor set that outgrows the budget DEFERS
    // honestly to the CPU path (its final device sort could not fit) — correct rows, no hit.
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
        .unwrap();
    let budget = 4096u64;
    e.set_relational_residency_budget_bytes(0, budget);

    // Selective: 100 survivors (400B) fit -> streamed, device-sorted DESC.
    let sorted = e
        .execute_relational_select(&select("SELECT a FROM big WHERE a >= 1400 ORDER BY a DESC"))
        .unwrap();
    let expected: Vec<Vec<SqlValue>> = (0..100).map(|i| vec![SqlValue::Int4(N - 1 - i)]).collect();
    assert_eq!(
        sorted.rows.clone().into_boxed(),
        expected,
        "unbounded ordered survivors"
    );
    let hits_after = e.streaming_fold_hits();
    assert!(hits_after >= 1, "unbounded ordered fold fired");

    // Non-selective: 1500 survivors (6000B) outgrow the 4096B budget -> DEFER (no hit), CPU serves it.
    let deferred = e
        .execute_relational_select(&select("SELECT a FROM big ORDER BY a"))
        .unwrap();
    assert_eq!(
        deferred.rows.len(),
        N as usize,
        "deferred full ordered scan served by CPU"
    );
    assert_eq!(
        deferred.rows.clone().into_boxed()[0],
        vec![SqlValue::Int4(0)],
        "CPU order correct"
    );
    assert_eq!(
        e.streaming_fold_hits(),
        hits_after,
        "the over-budget unbounded ORDER BY must DEFER (no streaming hit)"
    );
    assert!(
        e.streaming_fold_peak_chunk_bytes() <= budget,
        "peak stays bounded through the defer (no over-budget upload)"
    );
}

/// No-regression + non-vacuity WITHOUT a GPU: with NO budget configured, the streaming fold must NOT fire
/// and the CPU host path serves the aggregate identically. Runs in the normal (non-ignored) suite so the
/// default byte-identical behavior is gated everywhere.
#[test]
fn streaming_reduction_absent_without_budget_uses_host_path() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a) VALUES (1), (2), (3), (4)")
        .unwrap();

    // No budget set -> the streaming gate returns None -> the CPU pinned path serves it.
    let count = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM t"))
        .unwrap();
    assert_eq!(count.rows, vec![vec![SqlValue::Int8(4)]]);
    assert_eq!(
        e.streaming_fold_hits(),
        0,
        "with no residency budget the streaming fold must never fire"
    );

    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM t"))
        .unwrap();
    assert_eq!(sum.rows, vec![vec![SqlValue::Int8(10)]]);
    assert_eq!(e.streaming_fold_hits(), 0, "SUM stays on the host path too");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_cold_tier_replay_probe() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 100_000;
    for batch in 0..10 {
        let mut values = String::new();
        for j in 0..(N / 10) {
            let i = batch * (N / 10) + j;
            if j > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, {})", i * 2));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 65536);
    // warm-up
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
        .unwrap();
    for run in 0..3 {
        let t = std::time::Instant::now();
        let c = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap();
        let t1 = t.elapsed().as_micros();
        let t = std::time::Instant::now();
        let s = e
            .execute_relational_select(&select("SELECT SUM(a) FROM big"))
            .unwrap();
        let t2 = t.elapsed().as_micros();
        assert_eq!(
            c.rows.clone().into_boxed(),
            vec![vec![SqlValue::Int8(100_000)]]
        );
        assert_eq!(
            s.rows.clone().into_boxed(),
            vec![vec![SqlValue::Int8(4_999_950_000i64)]]
        );
        eprintln!("COLDPROBE run={run} count_us={t1} sum_us={t2}");
        assert!(
            e.streaming_cold_hits() >= 1,
            "cold tier served the repeat reads"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_cold_tier_invalidates_on_write() {
    // S-E.6 THE correctness gate: the cold tier serves BYTE-REPLAYS of a prior build, so a WRITE must
    // invalidate it (the tuple-store generation Arc changes on every COW publish) — a stale hit would
    // serve pre-write data to post-write readers. Build -> hit -> INSERT -> fresh result -> hit again
    // -> DELETE -> fresh result.
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);

    // Build (scan + capture), then a hit (byte replay), identical results.
    let count = |e: &Engine| {
        e.execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap()
            .rows
            .clone()
            .into_boxed()
    };
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    let builds_after_first = e.streaming_cold_builds();
    assert!(
        builds_after_first >= 1,
        "the first streaming read installs the cold tier"
    );
    let hits_before = e.streaming_cold_hits();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    assert!(
        e.streaming_cold_hits() > hits_before,
        "the repeat read must SERVE FROM the cold tier"
    );
    // A different fold shape hits the SAME cache (chunks are fold-agnostic).
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    let expected_sum: i64 = (0..i64::from(N)).sum();
    assert_eq!(
        sum.rows.clone().into_boxed(),
        vec![vec![SqlValue::Int8(expected_sum)]]
    );

    // INSERT -> the generation Arc changes -> MISS -> fresh scan sees N+1 (a stale hit would say N).
    seq += 1;
    e.execute_text(seq, "INSERT INTO big (a) VALUES (100000)")
        .unwrap();
    assert_eq!(
        count(&e),
        vec![vec![SqlValue::Int8(i64::from(N) + 1)]],
        "a write must invalidate the cold tier (stale replay would return the OLD count)"
    );
    // 6c-1: the post-write read PATCHES the entry (O(delta)) instead of rebuilding it.
    assert_eq!(
        e.streaming_cold_builds(),
        builds_after_first,
        "the post-write read PATCHES — no fresh build"
    );
    assert!(
        e.streaming_cold_patches() >= 1,
        "the write was served by a PATCH"
    );
    // The rebuilt cache serves hits again...
    let hits_before = e.streaming_cold_hits();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N) + 1)]]);
    assert!(
        e.streaming_cold_hits() > hits_before,
        "rebuilt cache hits again"
    );
    // ...and a DELETE invalidates again.
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a = 100000")
        .unwrap();
    assert_eq!(
        count(&e),
        vec![vec![SqlValue::Int8(i64::from(N))]],
        "a DELETE must invalidate the cold tier"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_cold_tier_spills_and_replays_from_disk() {
    // S-E.6b: over the spill threshold the cold tier lives in an UNLINKED temp file, not host RAM.
    // Force a tiny threshold (test-only override) so the capture spills, then prove: the spill
    // counter fired; replays (aggregates AND the exact-order projection — chunk offsets must map
    // back byte-exactly) match the scan build; a write invalidates; a rebuilt spill serves again.
    crate::engine_streaming_exec::STREAMING_COLD_SPILL_THRESHOLD_TEST
        .store(1024, std::sync::atomic::Ordering::Relaxed);
    let result = std::panic::catch_unwind(|| {
        let mut e = Engine::new_local();
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return;
        }
        seq += 1;
        e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
        const N: i32 = 1500;
        let mut values = String::new();
        for i in 0..N {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i})"));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
            .unwrap();
        e.set_relational_residency_budget_bytes(0, 4096);

        // Build: 6000B of payloads > the 1KB forced threshold -> the capture SPILLS.
        let count = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap();
        assert_eq!(
            count.rows.clone().into_boxed(),
            vec![vec![SqlValue::Int8(i64::from(N))]]
        );
        assert!(
            e.streaming_cold_spills() >= 1,
            "the capture must have SPILLED (threshold forced to 1KB)"
        );
        // Aggregate replay from disk.
        let hits_before = e.streaming_cold_hits();
        let sum = e
            .execute_relational_select(&select("SELECT SUM(a) FROM big"))
            .unwrap();
        let expected_sum: i64 = (0..i64::from(N)).sum();
        assert_eq!(
            sum.rows.clone().into_boxed(),
            vec![vec![SqlValue::Int8(expected_sum)]]
        );
        assert!(
            e.streaming_cold_hits() > hits_before,
            "spilled replay served the SUM"
        );
        // EXACT-ORDER projection replay: chunk offsets must round-trip byte-exactly (a swapped or
        // misaligned positional read would reorder or corrupt rows).
        let rows = e
            .execute_relational_select(&select("SELECT a FROM big WHERE a >= 1000"))
            .unwrap();
        let expected: Vec<Vec<SqlValue>> = (1000..N).map(|i| vec![SqlValue::Int4(i)]).collect();
        assert_eq!(
            rows.rows.clone().into_boxed(),
            expected,
            "spilled projection byte-exact"
        );

        // A write invalidates the spilled entry (generation change), and the rebuild re-spills.
        let spills_before = e.streaming_cold_spills();
        seq += 1;
        e.execute_text(seq, "INSERT INTO big (a) VALUES (100000)")
            .unwrap();
        let count = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap();
        assert_eq!(
            count.rows.clone().into_boxed(),
            vec![vec![SqlValue::Int8(i64::from(N) + 1)]],
            "post-write count fresh (stale spilled replay would say N)"
        );
        assert!(
            e.streaming_cold_spills() > spills_before,
            "the rebuild re-spilled"
        );
    });
    crate::engine_streaming_exec::STREAMING_COLD_SPILL_THRESHOLD_TEST
        .store(0, std::sync::atomic::Ordering::Relaxed);
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_grouped_bigint_sum_repro() {
    // COVERAGE gate: grouped SUM(bigint) over a streamed table = Numeric{38,0} partials in the
    // merge with a SUM-only aggregate mask — the exact shape of the masked-pass2 phantom-group
    // kernel bug (execution/lib.rs pass2_fn gate). NOTE (audit, sabotage-verified): this standalone
    // test does NOT deterministically reproduce the phantom (its small row_slots lease lands in a
    // pool bucket the phase-1 contamination does not dirty) — the AUTHORITATIVE regression gate is
    // the suite ORDER `gpu_streaming_cold_tier_spills_and_replays_from_disk` then
    // `gpu_streaming_distinct_over_budget_set_union`, which failed deterministically when the pass2
    // gate was removed. This test pins the SHAPE (numeric partials + SUM mask reach the merge).
    crate::engine_streaming_exec::STREAMING_COLD_SPILL_THRESHOLD_TEST
        .store(1024, std::sync::atomic::Ordering::Relaxed);
    let outcome = std::panic::catch_unwind(|| {
        let mut e = Engine::new_local();
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return;
        }
        // Phase 1: pool contamination — a spilled cold build + replays (mirrors the spill test).
        seq += 1;
        e.execute_text(seq, "CREATE TABLE contam (a INT)").unwrap();
        let mut values = String::new();
        for i in 0..1500 {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i})"));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO contam (a) VALUES {values}"))
            .unwrap();
        e.set_relational_residency_budget_bytes(0, 4096);
        let _ = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM contam"))
            .unwrap();
        let _ = e
            .execute_relational_select(&select("SELECT SUM(a) FROM contam"))
            .unwrap();
        let _ = e
            .execute_relational_select(&select("SELECT a FROM contam WHERE a >= 1000"))
            .unwrap();

        // Phase 2: grouped SUM(bigint) over a streamed table -> numeric partials in the merge.
        seq += 1;
        e.execute_text(seq, "CREATE TABLE gb (g INT, v BIGINT)")
            .unwrap();
        let mut values = String::new();
        for i in 0..1500i64 {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({}, {})", i % 13, 1000 + i));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO gb (g, v) VALUES {values}"))
            .unwrap();
        let result = e
            .execute_relational_select(&select("SELECT g, SUM(v) FROM gb GROUP BY g"))
            .unwrap();
        let mut rows = result.rows.clone().into_boxed();
        rows.sort_by(|a, b| crate::rel_exec_helpers::compare_sql_values(&a[0], &b[0]));
        let expected: Vec<Vec<SqlValue>> = (0..13i64)
            .map(|g| {
                let sum: i128 = (0..1500i64)
                    .filter(|i| i % 13 == g)
                    .map(|i| (1000 + i) as i128)
                    .sum();
                vec![
                    SqlValue::Int4(g as i32),
                    SqlValue::Numeric(Decimal128::new(sum, 0)),
                ]
            })
            .collect();
        assert_eq!(rows.len(), 13, "13 groups, no duplicates: got {rows:?}");
        assert_eq!(rows, expected, "grouped SUM(bigint) exact");
        assert!(e.streaming_fold_hits() >= 1, "streamed (not CPU)");
    });
    crate::engine_streaming_exec::STREAMING_COLD_SPILL_THRESHOLD_TEST
        .store(0, std::sync::atomic::Ordering::Relaxed);
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_cold_tier_patches_deltas_chunk_granular() {
    // These gates exercise the STORE-DRIVEN patch/stamp machinery (live for non-class tables);
    // without this the table class-enters mid-test and the semantics legitimately change.
    let _class_off = ClassEntryDisabled::new();
    // 6c-1: a write PATCHES the cold entry at CHUNK granularity instead of discarding it — the
    // O(delta) maintenance win. INSERT = a pure tail append (zero dirty chunks rebuilt); a one-row
    // DELETE rebuilds EXACTLY ONE dirty chunk (of several); every aggregate stays exact through
    // the patches. Composes: COW chain identity (imbl diff), effective-range tiling, the rollover
    // tail, the S-E.6a settled-boundary install.
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);

    let count = |e: &Engine| {
        e.execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap()
            .rows
            .clone()
            .into_boxed()
    };
    // Build (~3 chunks of 512 rows).
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    assert_eq!(
        e.streaming_cold_patches(),
        0,
        "first read is a build, not a patch"
    );

    // INSERT -> tail-append patch: zero dirty chunks rebuilt.
    seq += 1;
    e.execute_text(seq, "INSERT INTO big (a) VALUES (100000)")
        .unwrap();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N) + 1)]]);
    assert_eq!(e.streaming_cold_patches(), 1, "the write PATCHED the entry");
    assert_eq!(
        e.streaming_cold_chunks_rebuilt(),
        0,
        "an INSERT is a pure TAIL append — no existing chunk rebuilds"
    );

    // One-row DELETE inside the FIRST chunk -> P2: a SIDECAR STAMP, zero rebuilds (the chunk's
    // bytes stay; the tombstone masks the row in-kernel at replay).
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a = 3").unwrap();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    assert_eq!(e.streaming_cold_patches(), 2);
    assert_eq!(
        e.streaming_cold_chunks_rebuilt(),
        0,
        "a pure one-row DELETE STAMPS its chunk's sidecar — nothing rebuilds (P2)"
    );
    assert_eq!(
        e.streaming_cold_stamps(),
        1,
        "exactly the deleted row is tombstone-stamped"
    );

    // Aggregate exactness through the patched chunks (SUM over the survivors + the tail row).
    let expected_sum: i64 = (0..i64::from(N)).sum::<i64>() - 3 + 100000;
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    assert_eq!(
        sum.rows.clone().into_boxed(),
        vec![vec![SqlValue::Int8(expected_sum)]]
    );

    // The patched entry serves plain hits again (no further patches).
    let patches = e.streaming_cold_patches();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    assert_eq!(
        e.streaming_cold_patches(),
        patches,
        "clean hit after the patch"
    );

    // F6 (audit): the EMPTY-table sentinel -> INSERT patch path (the (1,0) sentinel chunk's hi=0
    // routes every new id to the tail; no panic, exact results).
    seq += 1;
    e.execute_text(seq, "CREATE TABLE hollow (a INT)").unwrap();
    let hollow_count = |e: &Engine| {
        e.execute_relational_select(&select("SELECT COUNT(*) FROM hollow"))
            .unwrap()
            .rows
            .clone()
            .into_boxed()
    };
    assert_eq!(
        hollow_count(&e),
        vec![vec![SqlValue::Int8(0)]],
        "empty build"
    );
    seq += 1;
    e.execute_text(seq, "INSERT INTO hollow (a) VALUES (1), (2)")
        .unwrap();
    assert_eq!(
        hollow_count(&e),
        vec![vec![SqlValue::Int8(2)]],
        "insert-into-empty patches (sentinel -> tail) without a panic"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_cold_tier_eager_commit_maintenance() {
    // These gates exercise the STORE-DRIVEN patch/stamp machinery (live for non-class tables);
    // without this the table class-enters mid-test and the semantics legitimately change.
    let _class_off = ClassEntryDisabled::new();
    // 6c-3: a COMMIT eagerly patches the table's cold entry (best-effort, under the held commit
    // mutex, O(delta)) — the patch counter moves AT COMMIT TIME, before any read; the next read is
    // a CLEAN HIT (no read-time patch). Reads never pay the maintenance.
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);

    // First read builds the entry.
    let count = |e: &Engine| {
        e.execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap()
            .rows
            .clone()
            .into_boxed()
    };
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    assert_eq!(e.streaming_cold_patches(), 0);

    // THE COMMIT ITSELF patches — no read in between.
    seq += 1;
    e.execute_text(seq, "INSERT INTO big (a) VALUES (100000)")
        .unwrap();
    assert_eq!(
        e.streaming_cold_patches(),
        1,
        "the COMMIT must have eagerly patched the cold entry (before any read)"
    );

    // The next read is a CLEAN HIT: correct result, no read-time patch.
    let hits_before = e.streaming_cold_hits();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N) + 1)]]);
    assert_eq!(
        e.streaming_cold_patches(),
        1,
        "no read-time patch — the read was a clean hit"
    );
    assert!(e.streaming_cold_hits() > hits_before);

    // A DELETE commit patches eagerly too — P2: a SIDECAR STAMP at commit, zero rebuilds.
    let rebuilt_before = e.streaming_cold_chunks_rebuilt();
    let stamps_before = e.streaming_cold_stamps();
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a = 3").unwrap();
    assert_eq!(
        e.streaming_cold_patches(),
        2,
        "the DELETE commit patched eagerly"
    );
    assert_eq!(
        e.streaming_cold_chunks_rebuilt(),
        rebuilt_before,
        "the eager DELETE patch STAMPS — no chunk rebuild at commit (P2)"
    );
    assert_eq!(
        e.streaming_cold_stamps(),
        stamps_before + 1,
        "the commit stamped exactly the deleted row"
    );
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
}

// ===================== P1 (sealed-shards-primary): the DURABLE cold checkpoint =====================

/// Pure encode/decode round-trip of the chunk descriptor (no GPU): every persisted field survives;
/// the transient bookkeeping (memory proof, refresh cost, invalidation) restores to its defaults.
#[test]
fn cold_checkpoint_descriptor_round_trips() {
    let descriptor = RelationalResidencySnapshot {
        gpu_id: 3,
        schema: "public".into(),
        table: "t".into(),
        generation: 41,
        row_count: 12,
        capacity: 12,
        column_count: 4,
        resident_bytes: 4096,
        resident_device_int4_columns: vec!["a".into(), "b".into()],
        resident_device_int4_column_stats: vec![
            crate::relational_model::ResidentDeviceInt4ColumnStats {
                name: "a".into(),
                min: -7,
                max: 900,
            },
        ],
        resident_device_int8_columns: vec!["big".into()],
        resident_device_numeric_columns: vec!["price".into()],
        resident_device_bool_columns: vec![
            crate::relational_model::ResidentDeviceBoolColumnLayout {
                name: "flag".into(),
                bitmap_byte_offset: 128,
            },
        ],
        resident_device_text_columns: vec![
            crate::relational_model::ResidentDeviceTextColumnLayout {
                name: "name".into(),
                offsets_byte_offset: 256,
                bytes_byte_offset: 304,
                bytes_len: 77,
            },
        ],
        resident_device_null_columns: vec![
            crate::relational_model::ResidentDeviceNullBitmapLayout {
                name: "b".into(),
                bitmap_byte_offset: 512,
            },
        ],
        valid_through_index: 99,
        invalidated_by_txn_id: None,
        invalidated_at_index: None,
        invalidated_by_memory_pressure: false,
        memory_pressure_active: false,
        last_refresh_cost: None,
        admission_budget_bytes: None,
        resident_bytes_after_admission: 4096,
        evicted_tables_on_admission: Vec::new(),
        device_memory_proof: None,
    };
    let mut w = crate::engine_streaming_exec::ColdCkptWriter {
        inner: Vec::<u8>::new(),
        hash: crate::engine_streaming_exec::FNV_OFFSET,
    };
    crate::engine_streaming_exec::encode_cold_descriptor(&mut w, &descriptor).unwrap();
    let mut r = crate::engine_streaming_exec::ColdCkptReader {
        inner: std::io::Cursor::new(w.inner),
    };
    let decoded = crate::engine_streaming_exec::decode_cold_descriptor(&mut r).unwrap();
    assert_eq!(decoded, descriptor);
}

/// Shared P1 harness: a lanes-mode durable database whose table `t` (plain 2-col int4, NO PK —
/// keeps the shape elision-ineligible so the streaming scan's store premise holds) has
/// `serial_rows` rows from the serial (pre-activation) phase and 24 fabricated lane-commit rows.
/// Returns (wal base path, expected row count, next fabricated row id base, next lane seq).
fn p1_lanes_streaming_fixture(
    tag: &str,
    serial_rows: i32,
) -> Option<(std::path::PathBuf, i64, u64)> {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-cold-ckpt-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("db.wal");
    let row_base = {
        let mut e = Engine::new_local();
        e.commit_state_mut().wal = WalBuffer::with_durable_segment(&base);
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return None; // off-box: skip
        }
        seq += 1;
        e.execute_text(seq, "CREATE TABLE t (a INT, b INT)")
            .unwrap();
        let mut values = String::new();
        for i in 0..serial_rows {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, {})", i * 2));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO t (a, b) VALUES {values}"))
            .unwrap();
        e.read_state.mvcc.current_row_id()
    };
    // Fabricated 2-lane history: 24 one-row binary INSERT commits (the recovery-suite pattern —
    // no live post-activation writes are needed, so the test never trips the classic-write guard).
    let tiny = 16 << 10;
    {
        let set = gpu_db_wal::FuaWalLaneSet::create(&base, 2, 2, tiny).expect("create lanes");
        for seq in 0..24u64 {
            let values = vec![SqlValue::Int4(10_000 + seq as i32), SqlValue::Int4(0)];
            let payload = crate::wal_binary::try_encode_binary_insert(
                "t",
                &[(row_base + seq, values.as_slice())],
            )
            .expect("binary encode");
            set.append(
                (seq % 2) as usize,
                seq,
                &[gpu_db_wal::WalRecord {
                    txn_id: 500 + seq,
                    payload: payload.into(),
                }],
            )
            .expect("append");
        }
        set.wait_durable(24).expect("durable");
    }
    Some((base, i64::from(serial_rows) + 24, row_base + 24))
}

fn p1_count(e: &Engine) -> i64 {
    let q = select("SELECT COUNT(*) FROM t");
    match e.execute_relational_select(&q).unwrap().rows.row(0)[0] {
        SqlValue::Int8(n) => n,
        ref other => panic!("COUNT returned {other:?}"),
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_restores_streaming_cold_across_reopen() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("restore", 300) else {
        return;
    };
    let budget = 512u64;
    let cut = {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        e.set_relational_residency_budget_bytes(0, budget);
        assert_eq!(p1_count(&e), expected);
        assert!(
            e.streaming_fold_hits() >= 1 && e.streaming_cold_builds() >= 1,
            "premise: the read streamed and captured the cold tier (fold {}, builds {})",
            e.streaming_fold_hits(),
            e.streaming_cold_builds()
        );
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert_eq!(cut, 24);
        assert!(
            e.streaming_cold_checkpointed() >= 1,
            "the lanes checkpoint must persist the quiesced cold tier"
        );
        cut
    };
    let artifact = base.with_file_name(format!(
        "{}.cold-checkpoint.{cut}",
        base.file_name().unwrap().to_string_lossy()
    ));
    assert!(
        artifact.exists(),
        "artifact {} must exist",
        artifact.display()
    );

    // REOPEN: the seam install restores the cold tier; the first streaming read is a byte REPLAY
    // (a HIT with zero fresh scan-builds), and the answer matches.
    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen after checkpoint");
    assert!(
        e.streaming_cold_restored() >= 1,
        "reopen must restore the cold tier from the checkpoint artifact"
    );
    e.set_relational_residency_budget_bytes(0, budget);
    assert_eq!(p1_count(&e), expected);
    assert!(
        e.streaming_cold_hits() >= 1,
        "the restored entry must serve the first streaming read"
    );
    assert_eq!(
        e.streaming_cold_builds(),
        0,
        "no fresh scan-build: the restore IS the build"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_patches_forward_post_checkpoint_wal_suffix() {
    let Some((base, expected, next_row)) = p1_lanes_streaming_fixture("suffix", 300) else {
        return;
    };
    let budget = 512u64;
    {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        e.set_relational_residency_budget_bytes(0, budget);
        assert_eq!(p1_count(&e), expected);
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert_eq!(cut, 24);
        assert!(e.streaming_cold_checkpointed() >= 1);
    }
    // Post-checkpoint WAL SUFFIX: 6 more fabricated lane commits ABOVE the checkpoint cut.
    let tiny = 16 << 10;
    {
        let set = gpu_db_wal::FuaWalLaneSet::reopen_from(&base, 2, 2, tiny, 24)
            .expect("wal-level reopen from baseline");
        for seq in 24..30u64 {
            let values = vec![SqlValue::Int4(20_000 + seq as i32), SqlValue::Int4(1)];
            let payload = crate::wal_binary::try_encode_binary_insert(
                "t",
                &[(next_row + (seq - 24), values.as_slice())],
            )
            .expect("binary encode");
            set.append(
                (seq % 2) as usize,
                seq,
                &[gpu_db_wal::WalRecord {
                    txn_id: 900 + seq,
                    payload: payload.into(),
                }],
            )
            .expect("append");
        }
        set.wait_durable(30).expect("durable");
    }
    // REOPEN: restore at the seam, then the 6-record suffix replays THROUGH the restored entry —
    // the 6c-3 commit hooks patch it forward (the WAL suffix IS the delta stream). The first
    // streaming read replays patched bytes and sees the suffix rows.
    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen after suffix");
    assert!(
        e.streaming_cold_restored() >= 1,
        "the seam restore must land before the suffix replays"
    );
    assert!(
        e.streaming_cold_patches() >= 1,
        "suffix replay must patch the restored entry via the commit hooks"
    );
    e.set_relational_residency_budget_bytes(0, budget);
    assert_eq!(
        p1_count(&e),
        expected + 6,
        "the suffix rows must be visible"
    );
    assert!(e.streaming_cold_hits() >= 1);
    assert_eq!(
        e.streaming_cold_builds(),
        0,
        "restore + patches carried the entry — no fresh scan-build"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_corrupt_artifact_is_skipped_never_wrong() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("corrupt", 300) else {
        return;
    };
    let budget = 512u64;
    let cut = {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        e.set_relational_residency_budget_bytes(0, budget);
        assert_eq!(p1_count(&e), expected);
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert!(e.streaming_cold_checkpointed() >= 1);
        cut
    };
    // Flip one byte in the artifact BODY (past the magic): the FNV trailer must reject it.
    let artifact = base.with_file_name(format!(
        "{}.cold-checkpoint.{cut}",
        base.file_name().unwrap().to_string_lossy()
    ));
    let mut bytes = std::fs::read(&artifact).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x40;
    std::fs::write(&artifact, &bytes).unwrap();

    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen with corrupt artifact");
    assert_eq!(
        e.streaming_cold_restored(),
        0,
        "a checksum-failed artifact must restore NOTHING"
    );
    e.set_relational_residency_budget_bytes(0, budget);
    assert_eq!(
        p1_count(&e),
        expected,
        "the read rebuilds from the store — never wrong"
    );
    assert!(
        e.streaming_cold_builds() >= 1,
        "the skipped restore leaves the first read to scan + capture"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_boundary_mismatch_is_skipped() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("boundary", 300) else {
        return;
    };
    let budget = 512u64;
    let cut = {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        e.set_relational_residency_budget_bytes(0, budget);
        assert_eq!(p1_count(&e), expected);
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert!(e.streaming_cold_checkpointed() >= 1);
        cut
    };
    // Tamper the artifact's BOUNDARY field (u64 right after the magic) and RECOMPUTE the FNV
    // trailer — a checksum-valid artifact whose boundary does not match the replay seam. The
    // strict-equality guard must skip it (installing would replay bytes from the WRONG commit
    // index — the one guard corruption cannot exercise).
    let artifact = base.with_file_name(format!(
        "{}.cold-checkpoint.{cut}",
        base.file_name().unwrap().to_string_lossy()
    ));
    let mut bytes = std::fs::read(&artifact).unwrap();
    let magic_len = b"GPUDBCOLDCKPT1\n".len();
    let boundary = u64::from_le_bytes(bytes[magic_len..magic_len + 8].try_into().unwrap());
    bytes[magic_len..magic_len + 8].copy_from_slice(&(boundary + 1).to_le_bytes());
    let body_len = bytes.len() - 8;
    let mut hash = crate::engine_streaming_exec::FNV_OFFSET;
    for b in &bytes[..body_len] {
        hash = (hash ^ u64::from(*b)).wrapping_mul(0x100000001b3);
    }
    bytes[body_len..].copy_from_slice(&hash.to_le_bytes());
    std::fs::write(&artifact, &bytes).unwrap();

    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen with tampered boundary");
    assert_eq!(
        e.streaming_cold_restored(),
        0,
        "a boundary-mismatched artifact must restore NOTHING (checksum alone cannot catch it)"
    );
    e.set_relational_residency_budget_bytes(0, budget);
    assert_eq!(
        p1_count(&e),
        expected,
        "the read rebuilds from the store — never wrong"
    );
    assert!(e.streaming_cold_builds() >= 1);
}

/// AUDIT HIGH regression (the boundary convention): the LIVE lane pump publishes committed_seq
/// as the EXCLUSIVE frontier (`visible_global_cut = base_seq + cut`), while the recovery seam's
/// replay publishes the INCLUSIVE last record index (`base_seq + cut - 1`) — one less. The
/// pre-fix code stamped the artifact with the live watermark verbatim, so every artifact captured
/// from a pump-published engine carried a boundary ONE HIGH and the restore silently never fired
/// in production (the four sibling tests replay-derive their watermark on BOTH sides, so they
/// cannot see it). This test emulates the pump's convention exactly — it re-publishes the
/// watermark at the frontier before checkpointing — and requires the restore to land anyway.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_restores_under_lane_pump_frontier_watermark() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("frontier", 300) else {
        return;
    };
    let budget = 512u64;
    {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        e.set_relational_residency_budget_bytes(0, budget);
        assert_eq!(p1_count(&e), expected);
        assert!(e.streaming_cold_builds() >= 1);
        // Emulate the pump: publish the EXCLUSIVE frontier (base_seq + cut), the value
        // engine_dml_concurrent's settle path publishes after a quiesced wave. The visible set
        // is unchanged (no stamp exists at the frontier).
        let lanes = e.intent_lanes.as_ref().expect("lanes installed");
        let frontier = lanes.base_seq.load(std::sync::atomic::Ordering::Acquire) + 24;
        e.publish_committed_seq(frontier);
        assert_eq!(
            e.committed_seq(),
            frontier,
            "premise: frontier-convention watermark"
        );
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert_eq!(cut, 24);
        assert!(
            e.streaming_cold_checkpointed() >= 1,
            "the frontier watermark must be ACCEPTED as the quiescence proof"
        );
    }
    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen after frontier checkpoint");
    assert!(
        e.streaming_cold_restored() >= 1,
        "the artifact must carry the SEAM boundary (inclusive last index), not the live \
         frontier — a frontier-stamped artifact never restores"
    );
    e.set_relational_residency_budget_bytes(0, budget);
    assert_eq!(p1_count(&e), expected);
    assert!(e.streaming_cold_hits() >= 1);
    assert_eq!(e.streaming_cold_builds(), 0);
}

// ============ P3 (sealed-shards-primary): the DML WHERE-locate as a streaming fold ============

/// Range-WHERE DELETE on a NON-ADMITTED (over-budget) table: the value index cannot bound it (no
/// Eq leaf) and the device arm has no shards — previously the pure-host seq_scan+filter loop. The
/// locate must now run ON-DEVICE via the streaming fold (counter-gated) and produce exactly the
/// host arm's result (differential: a twin engine with no budget runs the identical statement
/// through the host loop).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_dml_locate_range_delete_on_device() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    let mut twin = Engine::new_local(); // host-arm oracle (no budget -> host seq_scan locate)
    let mut twin_seq = 0u64;

    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    for (engine, s) in [(&mut e, &mut seq), (&mut twin, &mut twin_seq)] {
        *s += 1;
        engine
            .execute_text(*s, "CREATE TABLE big (a INT, b INT)")
            .unwrap();
        *s += 1;
        engine
            .execute_text(*s, &format!("INSERT INTO big (a, b) VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 4096);

    assert_eq!(e.dml_streaming_resolve_hits(), 0);
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a > 1200")
        .unwrap();
    assert_eq!(
        e.dml_streaming_resolve_hits(),
        1,
        "the range DELETE locate must resolve via the streaming fold"
    );
    twin_seq += 1;
    twin.execute_text(twin_seq, "DELETE FROM big WHERE a > 1200")
        .unwrap();

    // Differential: identical surviving rows (read both through the same CPU-pinned path).
    e.clear_relational_residency_budget_bytes(0);
    let q = select("SELECT a, b FROM big ORDER BY a");
    let got = e
        .execute_relational_select(&q)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    let want = twin
        .execute_relational_select(&q)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(got.len(), 1201, "rows 0..=1200 survive");
    assert_eq!(got, want, "device locate == host locate");
}

/// The UPDATE twin: range-WHERE assignments through the streaming locate, differential-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_dml_locate_range_update_on_device() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    let mut twin = Engine::new_local();
    let mut twin_seq = 0u64;

    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    for (engine, s) in [(&mut e, &mut seq), (&mut twin, &mut twin_seq)] {
        *s += 1;
        engine
            .execute_text(*s, "CREATE TABLE big (a INT, b INT)")
            .unwrap();
        *s += 1;
        engine
            .execute_text(*s, &format!("INSERT INTO big (a, b) VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 4096);

    seq += 1;
    e.execute_text(seq, "UPDATE big SET b = -5 WHERE a >= 1400")
        .unwrap();
    assert_eq!(
        e.dml_streaming_resolve_hits(),
        1,
        "the range UPDATE locate must resolve via the streaming fold"
    );
    twin_seq += 1;
    twin.execute_text(twin_seq, "UPDATE big SET b = -5 WHERE a >= 1400")
        .unwrap();

    e.clear_relational_residency_budget_bytes(0);
    let q = select("SELECT a, b FROM big ORDER BY a");
    let got = e
        .execute_relational_select(&q)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    let want = twin
        .execute_relational_select(&q)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(got.len(), N as usize);
    assert_eq!(got, want, "device locate == host locate");
    assert_eq!(
        got.iter().filter(|r| r[1] == SqlValue::Int4(-5)).count(),
        100,
        "rows 1400..=1499 updated"
    );
}

/// A zero-match range DELETE is a VALID streaming resolve (Some(vec![]) — the conditional 0-row
/// delete), not a decline: the counter fires and nothing changes.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_dml_locate_zero_matches_is_a_resolve() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);

    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a > 999999")
        .unwrap();
    assert_eq!(
        e.dml_streaming_resolve_hits(),
        1,
        "0-match locate still resolves on-device"
    );

    let q = select("SELECT COUNT(*) FROM big");
    assert_eq!(
        e.execute_relational_select(&q)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(i64::from(N))]],
        "nothing deleted"
    );
}

/// Without a configured budget the locate DECLINES (no streaming) and the host arm serves —
/// byte-identical default behavior (the activation-gate contract).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_dml_locate_declines_without_budget() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO big (a, b) VALUES (1, 2), (5, 6), (9, 10)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a > 4").unwrap();
    assert_eq!(e.dml_streaming_resolve_hits(), 0, "no budget -> host arm");
    let q = select("SELECT COUNT(*) FROM big");
    assert_eq!(
        e.execute_relational_select(&q)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(1)]]
    );
}

/// AUDIT M1 (P3): the streaming locate is the FIRST consumer of the DML predicate lowering with
/// NO host recheck (the sibling device arms recheck; the read folds set the no-recheck
/// precedent) — so the non-int4 type matrix must be differential-gated on THIS path. One
/// NULL-bearing mixed-type table; per statement: the budgeted engine must resolve via the
/// streaming fold (counter-gated) and its final state must equal a host-arm twin's exactly.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_dml_locate_type_matrix_differential() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    let mut twin = Engine::new_local();
    let mut twin_seq = 0u64;

    const N: i32 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        // Every 7th text NULL; every 5th numeric NULL — the 3VL exclusion must match the host's.
        let t = if i % 7 == 0 {
            "NULL".to_string()
        } else {
            format!("'txt{:04}'", (i * 37) % 1000)
        };
        let n = if i % 5 == 0 {
            "NULL".to_string()
        } else {
            format!("{}.{:02}", i % 90, i % 100)
        };
        let d = format!("'2024-{:02}-{:02}'", 1 + (i % 12), 1 + (i % 28));
        let flag = if i % 2 == 0 { "true" } else { "false" };
        let big = i64::from(i) * 1_000_000_007;
        values.push_str(&format!("({i}, {t}, {d}, {n}, {flag}, {big})"));
    }
    for (engine, s) in [(&mut e, &mut seq), (&mut twin, &mut twin_seq)] {
        *s += 1;
        engine
            .execute_text(
                *s,
                "CREATE TABLE mix (a INT, t TEXT, d DATE, n NUMERIC(10,2), flag BOOL, big BIGINT)",
            )
            .unwrap();
        *s += 1;
        engine
            .execute_text(*s, &format!("INSERT INTO mix VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 4096);

    let statements = [
        // TEXT ordering (byte-lexicographic device kernel vs host compare) over a NULL-bearing col.
        "DELETE FROM mix WHERE t > 'txt0800'",
        // DATE ordering (canonical-text lowering -> days round-trip).
        "DELETE FROM mix WHERE d < '2024-03-15'",
        // NUMERIC scale (rescale-to-column-scale peephole) range.
        "DELETE FROM mix WHERE n >= 44.10",
        // BIGINT (i64 section) range + OR-of-AND groups incl. a bool leaf.
        "DELETE FROM mix WHERE big > 400000000000 OR flag = false AND a < 100",
        // UPDATE through the same locate: text range target, int assignment.
        "UPDATE mix SET a = -1 WHERE t < 'txt0200'",
    ];
    let q = select("SELECT a, t, d, n, flag, big FROM mix ORDER BY big");
    for (i, statement) in statements.iter().enumerate() {
        let hits_before = e.dml_streaming_resolve_hits();
        seq += 1;
        e.execute_text(seq, statement).unwrap();
        assert_eq!(
            e.dml_streaming_resolve_hits(),
            hits_before + 1,
            "statement {i} ({statement}) must resolve via the streaming fold"
        );
        twin_seq += 1;
        twin.execute_text(twin_seq, statement).unwrap();

        e.clear_relational_residency_budget_bytes(0);
        let got = e
            .execute_relational_select(&q)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>();
        let want = twin
            .execute_relational_select(&q)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(
            got, want,
            "statement {i} ({statement}): device locate != host locate"
        );
        e.set_relational_residency_budget_bytes(0, 4096);
    }
}

// ========== P2 (sealed-shards-primary): SV2 tombstone sidecars for cold chunks ==========

/// Multi-row deletes across SEVERAL chunks stamp sidecars (zero rebuilds) and every read shape —
/// aggregate AND row-level projection — masks the tombstoned rows in-kernel, matching a
/// host-path twin exactly.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_sidecar_stamps_mask_rows_across_chunks() {
    // These gates exercise the STORE-DRIVEN patch/stamp machinery (live for non-class tables);
    // without this the table class-enters mid-test and the semantics legitimately change.
    let _class_off = ClassEntryDisabled::new();
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);

    // Build the cold tier (multi-chunk).
    let q_count = select("SELECT COUNT(*) FROM big");
    assert_eq!(
        e.execute_relational_select(&q_count)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(i64::from(N))]]
    );
    assert!(e.streaming_cold_builds() >= 1);

    // Rows 100, 700 and 1400 live in DIFFERENT chunks (ids ascending; ~12 chunks of ~128 rows).
    // Each Eq-DELETE resolves via the value index (host arm) and must EAGERLY STAMP at commit.
    for a in [100, 700, 1400] {
        seq += 1;
        e.execute_text(seq, &format!("DELETE FROM big WHERE a = {a}"))
            .unwrap();
    }
    assert_eq!(e.streaming_cold_stamps(), 3, "three rows stamped");
    assert_eq!(
        e.streaming_cold_chunks_rebuilt(),
        0,
        "no rebuild for pure deletes"
    );

    // Aggregate through stamped chunks.
    assert_eq!(
        e.execute_relational_select(&q_count)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(i64::from(N) - 3)]]
    );
    // VALUE-SENSITIVE reads through the stamped chunks (COUNT alone cannot catch a mask on the
    // WRONG slot): SUM must reflect exactly WHICH rows are masked (closed form), and a bounded
    // window projection around a deleted row must return exactly the surviving neighbors (small
    // survivor set — no honest-defer; an unbounded ORDER BY here would defer to the CPU and
    // test nothing).
    let expected_sum: i64 = (0..i64::from(N)).sum::<i64>() - 100 - 700 - 1400;
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    assert_eq!(
        sum.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(expected_sum)]],
        "SUM through stamped chunks must miss exactly the deleted rows"
    );
    let q_window = select("SELECT a FROM big WHERE a >= 98 AND a <= 102");
    let got = e
        .execute_relational_select(&q_window)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(
        got,
        vec![
            vec![SqlValue::Int4(98)],
            vec![SqlValue::Int4(99)],
            vec![SqlValue::Int4(101)],
            vec![SqlValue::Int4(102)],
        ],
        "the window around the deleted row must skip EXACTLY it"
    );
}

/// Mixed workload over a stamped entry: a DELETE stamps, an INSERT tail-patches with the stamped
/// chunk REUSED (its sidecar preserved), an UPDATE (same-id version chain) falls to the REBUILD
/// arm — every step correct, and a stamped entry DECLINES the P1 cold checkpoint (the v1
/// artifact has no sidecar sections; benign skip, ledgered as P2b).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_sidecar_mixed_workload_and_v2_artifact_roundtrip() {
    // A STORE-DRIVEN-era gate (P2 stamps + the P2b artifact round-trip against a replayed twin
    // whose id space must match): the class would shift ids via skipped installs mid-test.
    let _class_off = ClassEntryDisabled::new();
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 900;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    let q_count = select("SELECT COUNT(*) FROM big");
    let count = |e: &Engine| -> i64 {
        match e.execute_relational_select(&q_count).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        }
    };
    assert_eq!(count(&e), i64::from(N));

    // DELETE -> stamp.
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a = 10").unwrap();
    assert_eq!(e.streaming_cold_stamps(), 1);
    assert_eq!(count(&e), i64::from(N) - 1);

    // INSERT -> pure tail patch; the STAMPED chunk is REUSED (stamps preserved, still masked).
    let rebuilt_before = e.streaming_cold_chunks_rebuilt();
    seq += 1;
    e.execute_text(seq, "INSERT INTO big (a, b) VALUES (100000, 1)")
        .unwrap();
    assert_eq!(
        e.streaming_cold_chunks_rebuilt(),
        rebuilt_before,
        "INSERT stays a pure tail append beside a stamped chunk"
    );
    assert_eq!(
        count(&e),
        i64::from(N),
        "tail row visible AND the stamp still masks"
    );

    // UPDATE (same-id version chain change) -> the classifier must refuse the stamp downgrade;
    // the rebuild arm serves it. Correctness is the assert; the arm split is the counter.
    seq += 1;
    e.execute_text(seq, "UPDATE big SET b = -7 WHERE a = 20")
        .unwrap();
    assert_eq!(count(&e), i64::from(N), "update preserves cardinality");
    let q_probe = select("SELECT b FROM big WHERE a = 20");
    assert_eq!(
        e.execute_relational_select(&q_probe)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int4(-7)]],
        "the updated value must be visible through the streaming read"
    );

    // P2b: a sidecar-bearing entry now QUALIFIES for the v2 artifact — and the sidecar
    // ROUND-TRIPS: a twin engine at the same commit boundary restores the entry and the
    // stamped rows STAY MASKED (no store consultation, no rebuild).
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a = 30").unwrap();
    assert!(
        e.streaming_cold_stamps() >= 2,
        "premise: the entry carries a sidecar"
    );
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-p2b-ckpt-roundtrip-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("db.wal");
    let boundary = e.committed_seq();
    let written = e
        .write_streaming_cold_checkpoint(&base, 1, boundary, boundary + 1)
        .expect("capture runs");
    assert_eq!(
        written, 1,
        "the v2 artifact must carry the sidecar-bearing entry"
    );

    // The twin replays the identical statement history (same commit boundary), restores the
    // artifact directly, and its FIRST streaming read replays the stamped bytes.
    let mut twin = Engine::new_local();
    let mut twin_seq = 0u64;
    if !gpu_available(&mut twin, &mut twin_seq) {
        return;
    }
    twin_seq += 1;
    twin.execute_text(twin_seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    twin_seq += 1;
    twin.execute_text(twin_seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    for statement in [
        "DELETE FROM big WHERE a = 10",
        "INSERT INTO big (a, b) VALUES (100000, 1)",
        "UPDATE big SET b = -7 WHERE a = 20",
        "DELETE FROM big WHERE a = 30",
    ] {
        twin_seq += 1;
        twin.execute_text(twin_seq, statement).unwrap();
    }
    assert_eq!(
        twin.committed_seq(),
        boundary,
        "premise: the twin reached the artifact boundary"
    );
    let restored = twin.restore_streaming_cold_checkpoint(&base, 1);
    assert_eq!(
        restored, 1,
        "the twin must restore the sidecar-bearing entry"
    );
    twin.set_relational_residency_budget_bytes(0, 4096);
    assert_eq!(
        count(&twin),
        i64::from(N) - 1,
        "restored stamps still mask (a=10, a=30 gone; tail row present)"
    );
    assert_eq!(
        twin.streaming_cold_builds(),
        0,
        "the restore IS the build — no scan"
    );
    assert!(twin.streaming_cold_hits() >= 1);

    // AUDIT LOW (adopted): a POST-RESTORE delete pins the persisted `payload_copin_s` — the
    // stamp's slot is the id's rank among ids visible at the PAYLOAD boundary; a restore that
    // defaulted the boundary to the seam would exclude the already-stamped rows from the rank,
    // shift the slot, and mask the WRONG row. COUNT is slot-blind; the closed-form SUM bites.
    twin_seq += 1;
    twin.execute_text(twin_seq, "DELETE FROM big WHERE a = 40")
        .unwrap();
    assert!(
        twin.streaming_cold_stamps() >= 1,
        "the post-restore delete must STAMP"
    );
    assert_eq!(count(&twin), i64::from(N) - 2);
    let expected_sum: i64 = (0..i64::from(N)).sum::<i64>() - 10 - 30 - 40 + 100000;
    let sum = twin
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    assert_eq!(
        sum.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(expected_sum)]],
        "the post-restore stamp must mask EXACTLY a=40 (payload-boundary rank)"
    );
}

/// THE CHANGE-LOG REGRESSION (found by P2's stamp counter, but 6c-1-era): `imbl::OrdMap::diff`
/// MISSED a real change — three sequential single-row deletes, each against a freshly pinned
/// generation (the cold-tier entry's exact usage), and the THIRD delete vanished from the diff
/// while the two generations' chains provably differed (deleted_by None vs Some). A missed delta
/// = a patched cold entry silently serving a deleted row. `changed_tuple_ids` therefore reads
/// the store's WRITE-SIDE CHANGE LOG (exact by construction) and structural diffing is BANNED
/// for correctness-bearing deltas. This is the minimal CPU repro, pinned forever.
#[test]
fn cow_change_log_reports_every_pinned_generation_delta() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    e.execute_text(2, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    let g0 = e.read_state.mvcc.table_rows("big").generation_payload();
    e.execute_text(3, "DELETE FROM big WHERE a = 100").unwrap();
    let g1 = e.read_state.mvcc.table_rows("big").generation_payload();
    assert_eq!(g0.rows.changed_tuple_ids(&g1.rows), vec![101]);
    drop(g0);
    e.execute_text(4, "DELETE FROM big WHERE a = 700").unwrap();
    let g2 = e.read_state.mvcc.table_rows("big").generation_payload();
    assert_eq!(g1.rows.changed_tuple_ids(&g2.rows), vec![701]);
    drop(g1);
    e.execute_text(5, "DELETE FROM big WHERE a = 1400").unwrap();
    let g3 = e.read_state.mvcc.table_rows("big").generation_payload();
    // The chains provably differ...
    assert_eq!(g2.rows.chain(1401).map(|c| c[0].deleted_by), Some(None));
    assert_eq!(g3.rows.chain(1401).map(|c| c[0].deleted_by), Some(Some(5)));
    // ...and the delta MUST say so (the imbl structural diff returned [] here).
    assert_eq!(
        g2.rows.changed_tuple_ids(&g3.rows),
        vec![1401],
        "the pinned-generation delta must report the third delete"
    );
}

// ========== P4-1 (chunk-authoritative tables): the REVERSE GATHER ==========

/// The round-trip differential that gates the host columnar decoder: a mixed-type NULL-bearing
/// table streams into cold chunks; the reverse gather must reproduce EXACTLY the store's visible
/// rows (order included — chunk order is scan order), across every section type (int4/date/int2
/// i32, int8/timestamp i64, numeric/uuid b128, bool bitmaps, text blobs, NULL validity bitmaps),
/// and honor the P2 sidecar with the kernel's semantics after a stamped delete.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_reverse_gather_round_trips_all_types_and_sidecars() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(
        seq,
        "CREATE TABLE mix (a INT, s SMALLINT, big BIGINT, d DATE, ts TIMESTAMP, \
         n NUMERIC(10,2), flag BOOL, t TEXT, u UUID)",
    )
    .unwrap();
    const N: i32 = 400;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        // Audit LOW: a NULL in EVERY nullable section type (i32/i64/b128/bool/text/uuid paths).
        let t = if i % 7 == 0 {
            "NULL".into()
        } else {
            format!("'txt{:04}'", i)
        };
        let n = if i % 5 == 0 {
            "NULL".into()
        } else {
            format!("{}.{:02}", i, i % 100)
        };
        let big = if i % 11 == 0 {
            "NULL".into()
        } else {
            format!("{}", i64::from(i) * 1_000_000_007)
        };
        let s16 = if i % 13 == 0 {
            "NULL".into()
        } else {
            format!("{}", i % 300 - 150)
        };
        let flag = if i % 17 == 0 {
            "NULL".into()
        } else if i % 2 == 0 {
            "true".into()
        } else {
            "false".to_string()
        };
        let u = if i % 19 == 0 {
            "NULL".into()
        } else {
            format!("'00000000-0000-0000-0000-{:012x}'", i)
        };
        values.push_str(&format!(
            "({i}, {s16}, {big}, '2024-{:02}-{:02}', '2024-01-01 00:{:02}:{:02}', {n}, {flag}, {t}, {u})",
            1 + (i % 12),
            1 + (i % 28),
            i % 60,
            (i * 7) % 60,
        ));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO mix VALUES {values}"))
        .unwrap();

    // The ORACLE: the store's visible rows BEFORE streaming (host path, scan order).
    let q = select("SELECT a, s, big, d, ts, n, flag, t, u FROM mix");
    let oracle = e
        .execute_relational_select(&q)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(oracle.len(), N as usize);

    // Stream -> cold chunks; reverse-gather at the current boundary.
    e.set_relational_residency_budget_bytes(0, 4096);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM mix"))
        .unwrap();
    assert!(
        e.streaming_cold_builds() >= 1,
        "premise: a cold entry exists"
    );
    let gathered = e
        .reverse_gather_streamed_rows("mix", e.committed_seq())
        .expect("cold entry present")
        .expect("decode succeeds");
    assert_eq!(
        gathered, oracle,
        "the reverse gather must reproduce the store rows exactly"
    );

    // A stamped DELETE: the gather at the current boundary must exclude EXACTLY that row.
    seq += 1;
    e.execute_text(seq, "DELETE FROM mix WHERE a = 42").unwrap();
    assert!(
        e.streaming_cold_stamps() >= 1,
        "premise: the delete STAMPED"
    );
    let gathered = e
        .reverse_gather_streamed_rows("mix", e.committed_seq())
        .expect("cold entry present")
        .expect("decode succeeds");
    let want: Vec<Vec<SqlValue>> = oracle
        .iter()
        .filter(|row| row[0] != SqlValue::Int4(42))
        .cloned()
        .collect();
    assert_eq!(
        gathered, want,
        "the sidecar mask must apply kernel-identically on the host"
    );
}

// ========== P4-2a (chunk-authoritative tables): chunk-native locate + locate-driven stamp ==========

/// THE LOCATE DIFFERENTIAL: the chunk-native locate (device predicate over the chunks themselves,
/// slots back) must select EXACTLY the rows the store-driven P3 locate selects for the same
/// predicate — compared by ROW VALUES (slots translate to rows through the P4-1 decoder: on an
/// unstamped entry, decoded[slot] IS the slot's row).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_native_locate_matches_store_locate() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    // Build the cold entry (multi-chunk).
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
        .unwrap();
    assert!(e.streaming_cold_builds() >= 1);

    let table = e
        .catalog_snapshot()
        .relational_catalog
        .get("big")
        .cloned()
        .unwrap();
    // A range + OR shape (value-index-unbindable): a > 1200 OR b < 100.
    let filter_groups: Vec<Vec<(usize, SelectFilterOp, SqlValue)>> = vec![
        vec![(0, SelectFilterOp::Gt, SqlValue::Int4(1200))],
        vec![(1, SelectFilterOp::Lt, SqlValue::Int4(100))],
    ];
    let predicate =
        crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(&table, &filter_groups)
            .expect("lowerable");
    let rtx = e.committed_seq();
    let located = e
        .locate_streaming_cold_slots(&table, &predicate, rtx)
        .expect("locate serves");
    // Translate (chunk, slot) -> row values through the P4-1 decoder (unstamped: slot-addressed).
    let entry_rows: Vec<Vec<Vec<SqlValue>>> = {
        let map = e.read_state.residency.streaming_cold_chunks.load();
        let entry = map.get("big").unwrap();
        entry
            .chunks
            .iter()
            .map(|c| crate::engine_streaming_exec::decode_cold_chunk_rows(&table, c, rtx).unwrap())
            .collect()
    };
    let mut got: Vec<Vec<SqlValue>> = located
        .iter()
        .flat_map(|(chunk_idx, slots)| {
            slots
                .iter()
                .map(|slot| entry_rows[*chunk_idx][*slot as usize].clone())
        })
        .collect();
    // The store-driven P3 locate on the SAME pinned view.
    let visibility = StorageVisibility { read_txn_id: rtx };
    let table_rows = e.read_state.mvcc.table_rows("big");
    let mut want: Vec<Vec<SqlValue>> = e
        .try_streaming_dml_locate(&table, &filter_groups, visibility, &table_rows)
        .expect("store locate serves")
        .into_iter()
        .map(|(_, _, row)| row)
        .collect();
    got.sort();
    want.sort();
    assert_eq!(got.len(), 349, "1201..=1499 (299) + b<100 => a<50 (50)");
    assert_eq!(
        got, want,
        "chunk-native locate == store-driven locate (row values)"
    );
}

/// THE STAMP ISOLATION GATE: locate coordinates on the chunks, stamp them at the current
/// boundary, and every chunk-served view (streaming COUNT/SUM, the reverse gather) must exclude
/// exactly those rows — WITHOUT any store write (the entry's generation is untouched; this is the
/// store-free write primitive in isolation). Already-stamped slots must NOT re-locate (the
/// sidecar visibility composes into the locate).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_locate_driven_stamp_masks_rows_without_store() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
        .unwrap();

    let table = e
        .catalog_snapshot()
        .relational_catalog
        .get("big")
        .cloned()
        .unwrap();
    let filter_groups: Vec<Vec<(usize, SelectFilterOp, SqlValue)>> =
        vec![vec![(0, SelectFilterOp::Gte, SqlValue::Int4(1490))]];
    let predicate =
        crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(&table, &filter_groups)
            .expect("lowerable");
    let rtx = e.committed_seq();
    let located = e
        .locate_streaming_cold_slots(&table, &predicate, rtx)
        .expect("locate serves");
    let located_count: usize = located.iter().map(|(_, s)| s.len()).sum();
    assert_eq!(located_count, 10, "a in 1490..=1499");

    assert!(
        e.stamp_streaming_cold_slots("big", &located, rtx, false),
        "the stamp must install"
    );
    assert_eq!(e.streaming_cold_stamps(), 10);

    // Chunk-served views exclude the stamped rows; the STORE was never written.
    let q_count = select("SELECT COUNT(*) FROM big");
    assert_eq!(
        e.execute_relational_select(&q_count)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(i64::from(N) - 10)]],
        "streaming COUNT masks the stamped rows"
    );
    let expected_sum: i64 = (0..i64::from(N) - 10).sum();
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    assert_eq!(
        sum.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(expected_sum)]],
        "streaming SUM masks EXACTLY the stamped rows"
    );
    let gathered = e
        .reverse_gather_streamed_rows("big", e.committed_seq())
        .unwrap()
        .unwrap();
    assert_eq!(
        gathered.len(),
        (N - 10) as usize,
        "the reverse gather agrees"
    );

    // Idempotence of visibility: re-locating the same predicate finds NOTHING (the sidecar mask
    // composes into the locate — stamped slots are invisible to it).
    let relocated = e
        .locate_streaming_cold_slots(&table, &predicate, rtx)
        .expect("locate serves");
    assert!(
        relocated.is_empty(),
        "already-stamped slots must not re-locate (got {relocated:?})"
    );
}

// ========== P4-2b-i (S-E.P4): the CHUNK-AUTHORITATIVE class — enter, freeze, stream, exit ==========

/// THE CLASS LIFECYCLE GATE: an over-budget, elision-INeligible (text-bearing), keyless FK-free
/// table ENTERS the class at a commit; subsequent INSERTs skip the host store (FROZEN — proven by
/// the store's version count) while the streamed reads see every row (the tail appends are the
/// materialization); an unstreamable read DE-AUTHORITIZES (the post-freeze delta replays into the
/// store) and the host path serves exactly the full data. Differential twin throughout.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_enters_freezes_streams_and_deauths() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    let mut twin = Engine::new_local();
    let mut twin_seq = 0u64;

    const N: i32 = 1200;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    for (engine, s) in [(&mut e, &mut seq), (&mut twin, &mut twin_seq)] {
        *s += 1;
        engine
            .execute_text(*s, "CREATE TABLE facts (a INT, t TEXT)")
            .unwrap();
        *s += 1;
        engine
            .execute_text(*s, &format!("INSERT INTO facts (a, t) VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 8192);
    let q_count = select("SELECT COUNT(*) FROM facts");
    let count = |e: &Engine| -> i64 {
        match e.execute_relational_select(&q_count).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        }
    };
    // Build the cold entry, then the ENTER commit (the eager patch makes the entry fresh at it).
    assert_eq!(count(&e), i64::from(N));
    assert_eq!(e.chunk_class_entries(), 0);
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (100000, 'enter')")
        .unwrap();
    twin_seq += 1;
    twin.execute_text(
        twin_seq,
        "INSERT INTO facts (a, t) VALUES (100000, 'enter')",
    )
    .unwrap();
    assert_eq!(
        e.chunk_class_entries(),
        1,
        "the table must ENTER the class at this commit"
    );

    // RECLAIMED (P4): class entry DELETED the host chains — the store-deletion payoff; the
    // chunks are the representation. The count below is 0 and stays 0 through every class write.
    assert!(
        e.chunk_class_reclaimed_rows() > 0,
        "entry must reclaim the host rows"
    );
    let frozen_versions = e
        .read_state
        .mvcc
        .table_rows("facts")
        .store()
        .all_versions()
        .len();
    assert_eq!(frozen_versions, 0, "the class table's host chains are GONE");
    for k in 0..5 {
        seq += 1;
        e.execute_text(
            seq,
            &format!(
                "INSERT INTO facts (a, t) VALUES ({}, 'tail{k}')",
                200000 + k
            ),
        )
        .unwrap();
        twin_seq += 1;
        twin.execute_text(
            twin_seq,
            &format!(
                "INSERT INTO facts (a, t) VALUES ({}, 'tail{k}')",
                200000 + k
            ),
        )
        .unwrap();
    }
    assert_eq!(
        e.chunk_class_skipped_installs(),
        5,
        "five commits skipped the host install"
    );
    assert_eq!(
        e.read_state
            .mvcc
            .table_rows("facts")
            .store()
            .all_versions()
            .len(),
        frozen_versions,
        "the store is FROZEN at the class boundary"
    );
    assert_eq!(
        count(&e),
        i64::from(N) + 6,
        "the streamed read sees every tail row"
    );
    assert_eq!(e.chunk_class_deauths(), 0, "no exit yet");

    // A value-sensitive streamed read through the tails (SUM over a).
    let expected_sum: i64 =
        (0..i64::from(N)).sum::<i64>() + 100000 + (0..5).map(|k| 200000 + k).sum::<i64>();
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM facts"))
        .unwrap();
    assert_eq!(
        sum.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(expected_sum)]]
    );

    // DE-AUTH: clear the budget — streaming deactivates, the CPU-pinned guard replays the
    // post-freeze delta into the store, the class exits, and the host path serves EVERYTHING.
    e.clear_relational_residency_budget_bytes(0);
    let q_rows = select("SELECT a, t FROM facts ORDER BY a");
    let got = e
        .execute_relational_select(&q_rows)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(
        e.chunk_class_deauths(),
        1,
        "the unstreamable read exited the class LOUDLY"
    );
    let want = twin
        .execute_relational_select(&q_rows)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(got.len(), (N + 6) as usize);
    assert_eq!(
        got, want,
        "post-de-auth host reads == the never-classed twin"
    );
    assert!(
        e.read_state
            .mvcc
            .table_rows("facts")
            .store()
            .all_versions()
            .len()
            > frozen_versions,
        "the delta replayed into the store"
    );

    // Post-exit writes are plain store writes again.
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a = 100000")
        .unwrap();
    twin_seq += 1;
    twin.execute_text(twin_seq, "DELETE FROM facts WHERE a = 100000")
        .unwrap();
    let got = e
        .execute_relational_select(&q_rows)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    let want = twin
        .execute_relational_select(&q_rows)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(got, want);
}

/// P4-2b-ii — CLASS DML STAYS CLASSED: DELETE stamps the chunk-native coordinates (no de-auth,
/// no store touch); UPDATE stamps the old versions and tail-appends the new images; every
/// streamed read reflects them exactly (closed-form SUM); an UNLOWERABLE predicate still falls
/// back to the loud de-auth exit.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_dml_stamps_without_deauth() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE facts (a INT, t TEXT)")
        .unwrap();
    const N: i32 = 1200;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO facts (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let q_count = select("SELECT COUNT(*) FROM facts");
    let count = |e: &Engine| -> i64 {
        match e.execute_relational_select(&q_count).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        }
    };
    let sum_a = |e: &Engine| -> i64 {
        match e
            .execute_relational_select(&select("SELECT SUM(a) FROM facts"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("sum: {other:?}"),
        }
    };
    let _ = count(&e);
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);

    // CLASS DELETE: a range WHERE resolves from the chunks (spanning the base AND the tail-
    // absorbed enter row: a=100000 also matches), stamps, and STAYS CLASSED.
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a >= 1195")
        .unwrap();
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "the class DELETE must NOT de-auth"
    );
    assert_eq!(
        e.streaming_cold_stamps(),
        6,
        "rows 1195..=1199 AND a=100000 stamped"
    );
    assert_eq!(count(&e), i64::from(N) - 5);
    let expected_sum: i64 = (0..1195i64).sum::<i64>();
    assert_eq!(
        sum_a(&e),
        expected_sum,
        "SUM reflects EXACTLY the stamped rows"
    );

    // CLASS UPDATE: stamp-old + tail-append-new, still classed.
    let stamps_before = e.streaming_cold_stamps();
    seq += 1;
    e.execute_text(seq, "UPDATE facts SET a = -7 WHERE a = 1000")
        .unwrap();
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "the class UPDATE must NOT de-auth"
    );
    assert!(
        e.streaming_cold_stamps() > stamps_before,
        "the old version stamped"
    );
    assert_eq!(count(&e), i64::from(N) - 5, "cardinality preserved");
    assert_eq!(
        sum_a(&e),
        expected_sum - 1000 - 7,
        "the new image replaced the old in every streamed read"
    );

    // The class survives further INSERTs after DML.
    let skipped_before = e.chunk_class_skipped_installs();
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (500000, 'post')")
        .unwrap();
    assert!(
        e.chunk_class_skipped_installs() > skipped_before,
        "still classed"
    );
    assert_eq!(sum_a(&e), expected_sum - 1000 - 7 + 500000);

    // The H2 DDL SWEEP exit: any non-DML statement de-authoritizes every class table BEFORE its
    // preflight reads the store — the replayed store must be MVCC-whole (tails inserted at their
    // born boundaries, every post-freeze stamp applied as a tombstone).
    seq += 1;
    e.execute_text(seq, "CREATE TABLE zzz (x INT)").unwrap();
    assert_eq!(e.chunk_class_deauths(), 1, "the DDL sweep exits the class");
    assert_eq!(
        count(&e),
        i64::from(N) - 4,
        "the de-authed store serves the exact post-DML state (stamps replayed, tails present)"
    );
    assert_eq!(
        sum_a(&e),
        expected_sum - 1000 - 7 + 500000,
        "value-exact after the exit"
    );
}

/// P4-3 — THE BORN GATE: a reader boundary below a tail chunk's born commit must not see its
/// rows (the reverse gather and the chunk-native locate are the directly-drivable surfaces; the
/// fold replay shares the same skip). Sidecar stamps above the boundary keep rows visible —
/// exact per-reader MVCC over the chunks.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_born_gate_serves_old_boundaries() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE facts (a INT, t TEXT)")
        .unwrap();
    const N: i32 = 900;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO facts (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM facts"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);
    let freeze = e.table_chunk_authoritative("facts").expect("classed");

    // Two post-freeze commits: a tail INSERT, then a class DELETE (a sidecar stamp).
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (200000, 'tail')")
        .unwrap();
    let after_insert = e.committed_seq();
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a = 5")
        .unwrap();
    let after_delete = e.committed_seq();

    // The reverse gather AT THE FREEZE: no tails, no post-freeze stamps applied (a=5 visible).
    let at_freeze = e
        .reverse_gather_streamed_rows("facts", freeze)
        .unwrap()
        .unwrap();
    assert_eq!(
        at_freeze.len(),
        (N + 1) as usize,
        "the freeze boundary sees base + enter only"
    );
    assert!(
        at_freeze.iter().any(|r| r[0] == SqlValue::Int4(5)),
        "the pre-delete boundary still sees a=5"
    );

    // At the post-insert boundary: the tail row appears; a=5 still visible (its stamp is later).
    let mid = e
        .reverse_gather_streamed_rows("facts", after_insert)
        .unwrap()
        .unwrap();
    assert_eq!(mid.len(), (N + 2) as usize);
    assert!(mid.iter().any(|r| r[0] == SqlValue::Int4(200000)));
    assert!(mid.iter().any(|r| r[0] == SqlValue::Int4(5)));

    // At the current boundary: the stamp masks a=5.
    let now = e
        .reverse_gather_streamed_rows("facts", after_delete)
        .unwrap()
        .unwrap();
    assert_eq!(now.len(), (N + 1) as usize);
    assert!(!now.iter().any(|r| r[0] == SqlValue::Int4(5)));

    // The chunk-native LOCATE born gate: a predicate matching ONLY the tail row finds it at the
    // current boundary and NOTHING at the freeze boundary.
    let table = e
        .catalog_snapshot()
        .relational_catalog
        .get("facts")
        .cloned()
        .unwrap();
    let groups: Vec<Vec<(usize, SelectFilterOp, SqlValue)>> =
        vec![vec![(0, SelectFilterOp::Eq, SqlValue::Int4(200000))]];
    let predicate =
        crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(&table, &groups).unwrap();
    let now_hits = e
        .locate_streaming_cold_slots(&table, &predicate, after_delete)
        .expect("locate serves");
    assert_eq!(now_hits.iter().map(|(_, s)| s.len()).sum::<usize>(), 1);
    let frozen_hits = e
        .locate_streaming_cold_slots(&table, &predicate, freeze)
        .expect("locate serves");
    assert!(
        frozen_hits.is_empty(),
        "a tail row is invisible to the freeze boundary"
    );
}

/// P4 COMPACTION: a class DELETE that kills most of a chunk triggers the in-install survivor
/// rebuild — the sidecar and dead slots are physically deleted, and every read stays
/// value-exact through the compacted chunk (closed-form SUM + the de-auth exit differential).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_compaction_deletes_dead_slots() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    // Audit note adopted: a NULLABLE b128 column rides the compaction round-trip (the survivor
    // gather + re-encode must preserve NULL validity and numeric mantissas, not just int/text).
    e.execute_text(seq, "CREATE TABLE facts (a INT, t TEXT, n NUMERIC(10,2))")
        .unwrap();
    const N: i32 = 900;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        let n = if i % 5 == 0 {
            "NULL".to_string()
        } else {
            format!("{i}.25")
        };
        values.push_str(&format!("({i}, 'txt{:04}', {n})", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO facts (a, t, n) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let q_count = select("SELECT COUNT(*) FROM facts");
    let count = |e: &Engine| -> i64 {
        match e.execute_relational_select(&q_count).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        }
    };
    let sum_a = |e: &Engine| -> i64 {
        match e
            .execute_relational_select(&select("SELECT SUM(a) FROM facts"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("sum: {other:?}"),
        }
    };
    let _ = count(&e);
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO facts (a, t, n) VALUES (100000, 'enter', 7.75)",
    )
    .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);

    // Kill MOST of the first chunk's rows (a < 200 spans it): the stamp install must COMPACT.
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a < 200")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 0, "stays classed");
    assert!(
        e.chunk_class_compactions() >= 1,
        "the heavily-stamped chunk must compact in the same install"
    );
    assert!(
        e.chunk_class_compacted_slots() >= 150,
        "the dead slots are physically deleted (got {})",
        e.chunk_class_compacted_slots()
    );

    // Value-exact through the compacted chunk.
    assert_eq!(count(&e), i64::from(N) - 200 + 1);
    let expected_sum: i64 = (200..i64::from(N)).sum::<i64>() + 100000;
    assert_eq!(
        sum_a(&e),
        expected_sum,
        "SUM through the compacted chunk is exact"
    );
    // The nullable numeric column survived compaction value-exactly: SUM(n) over survivors
    // (i.25 for i in 200..900 where i % 5 != 0) + the enter row's 7.75.
    let mantissa_sum: i128 = (200..i128::from(N))
        .filter(|i| i % 5 != 0)
        .map(|i| i * 100 + 25)
        .sum::<i128>()
        + 775;
    let sum_n = e
        .execute_relational_select(&select("SELECT SUM(n) FROM facts"))
        .unwrap();
    assert_eq!(
        sum_n.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Numeric(gpu_db_sql::Decimal128::new(
            mantissa_sum,
            2
        ))]],
        "the NULL-bearing numeric column round-tripped compaction exactly"
    );

    // Post-compaction DML + the exit both stay correct (coordinates re-slotted: the NEXT delete
    // locates against the fresh epoch).
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a = 500")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 0);
    assert_eq!(sum_a(&e), expected_sum - 500);
    seq += 1;
    e.execute_text(seq, "CREATE TABLE zzz2 (x INT)").unwrap(); // the DDL-sweep exit
    assert_eq!(e.chunk_class_deauths(), 1);
    assert_eq!(
        sum_a(&e),
        expected_sum - 500,
        "the de-authed store is value-exact"
    );
}

/// P5-0 — THE DEVICE SLOT RECHECK differential: for every slot of a staged mixed-type
/// NULL-bearing chunk, the single-slot device materialization must equal the P4-1 host
/// decoder's row exactly, and the sidecar/born masks must agree (a stamped slot returns
/// Some(None) at-or-above its stamp and the live row below it).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_device_slot_recheck_matches_host_decoder() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(
        seq,
        "CREATE TABLE mix (a INT, s SMALLINT, big BIGINT, d DATE, ts TIMESTAMP, \
         n NUMERIC(10,2), flag BOOL, t TEXT, u UUID)",
    )
    .unwrap();
    const N: i32 = 300;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        let t = if i % 7 == 0 {
            "NULL".into()
        } else {
            format!("'txt{:04}'", i)
        };
        let n = if i % 5 == 0 {
            "NULL".into()
        } else {
            format!("{}.{:02}", i, i % 100)
        };
        let big = if i % 11 == 0 {
            "NULL".into()
        } else {
            format!("{}", i64::from(i) * 999_983)
        };
        let flag = if i % 17 == 0 {
            "NULL".into()
        } else if i % 2 == 0 {
            "true".into()
        } else {
            "false".to_string()
        };
        let u = if i % 19 == 0 {
            "NULL".into()
        } else {
            format!("'00000000-0000-0000-0000-{:012x}'", i)
        };
        values.push_str(&format!(
            "({i}, {}, {big}, '2024-{:02}-{:02}', '2024-01-01 00:{:02}:{:02}', {n}, {flag}, {t}, {u})",
            if i % 13 == 0 { "NULL".to_string() } else { format!("{}", i % 300 - 150) },
            1 + (i % 12),
            1 + (i % 28),
            i % 60,
            (i * 7) % 60,
        ));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO mix VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM mix"))
        .unwrap();
    // Stamp one row so the mask path is exercised (class or store-driven — either stamps).
    seq += 1;
    e.execute_text(seq, "DELETE FROM mix WHERE a = 42").unwrap();
    let rtx = e.committed_seq();

    let table = e
        .catalog_snapshot()
        .relational_catalog
        .get("mix")
        .cloned()
        .unwrap();
    let map = e.read_state.residency.streaming_cold_chunks.load();
    let entry = map.get("mix").expect("cold entry");
    let mut checked = 0usize;
    let mut masked = 0usize;
    for chunk in &entry.chunks {
        // Stage once per chunk; recheck every slot against the host decoder.
        let (staged, _vis) = e
            .stage_cold_chunk(chunk, rtx)
            .expect("stage")
            .ready()
            .expect("ready");
        let host_unmasked =
            crate::engine_streaming_exec::decode_cold_chunk_rows(&table, chunk, 0).unwrap();
        let host_masked =
            crate::engine_streaming_exec::decode_cold_chunk_rows(&table, chunk, rtx).unwrap();
        let mut masked_iter = host_masked.iter();
        for (slot, expected) in host_unmasked.iter().enumerate() {
            let got = e
                .materialize_cold_chunk_slot(&table, chunk, &staged, slot, rtx)
                .expect("no decline");
            // Determine liveness from the host sidecar semantics: the unmasked row is always
            // present; the masked stream skips dead slots.
            match &got {
                Some(row) => {
                    assert_eq!(row, expected, "slot {slot} value mismatch");
                    assert_eq!(
                        Some(row),
                        masked_iter.next(),
                        "masked-stream alignment at slot {slot}"
                    );
                }
                None => {
                    masked += 1;
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, N as usize, "every slot rechecked");
    assert_eq!(masked, 1, "exactly the stamped row masks");
}

/// P5-1 — THE CHUNK KEY-INDEX CACHE: build per-chunk device hash indexes over a key column,
/// probe present/absent needles in ONE multi-chunk launch, recheck each hit's value via the
/// P5-0 device slot read, and verify the all-visible-index + recheck-mask contract (a stamped
/// row still HITS the index; the recheck masks it — the P5-2 uniqueness semantics).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_key_index_builds_probes_and_rechecks() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE facts (a INT, t TEXT)")
        .unwrap();
    const N: i32 = 900;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO facts (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM facts"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);
    // A stamped delete: the index is ALL-VISIBLE, so the probe must still hit and the P5-0
    // recheck must mask.
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a = 7")
        .unwrap();
    let rtx = e.committed_seq();

    let table = e
        .catalog_snapshot()
        .relational_catalog
        .get("facts")
        .cloned()
        .unwrap();
    let entry = e
        .read_state
        .residency
        .streaming_cold_chunks
        .load()
        .get("facts")
        .cloned()
        .unwrap();
    let indexes = e
        .ensure_chunk_key_indexes(&table, &entry, &[0], 0)
        .expect("indexes build");
    assert!(!indexes.is_empty());
    assert!(
        e.read_state
            .residency
            .chunk_key_index_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0,
        "the retained buffers are accounted"
    );

    // Needles: three present (5, 500, 100000 — the tail row), the stamped one (7), one absent.
    let needles: Vec<i32> = vec![5, 500, 100000, 7, 424242];
    let hits = e
        .probe_chunk_key_indexes(&indexes, &needles)
        .expect("probe");
    assert_eq!(hits.len(), 5);
    // Present keys: exactly one live hit each whose recheck yields the key value.
    for (n, expect_a) in [(0usize, 5i32), (1, 500), (2, 100000)] {
        let mut live = 0;
        for (pos, slot) in &hits[n] {
            let chunk = &entry.chunks[*pos];
            let (staged, _) = e.stage_cold_chunk(chunk, rtx).unwrap().ready().unwrap();
            if let Some(row) = e
                .materialize_cold_chunk_slot(&table, chunk, &staged, *slot as usize, rtx)
                .expect("no decline")
            {
                assert_eq!(
                    row[0],
                    SqlValue::Int4(expect_a),
                    "hit rechecks to the needle"
                );
                live += 1;
            }
        }
        assert_eq!(live, 1, "needle {n}: exactly one live hit");
    }
    // The STAMPED key: the index hits, the recheck masks — no live hit (the P5-2 not-a-conflict).
    assert!(
        !hits[3].is_empty(),
        "the all-visible index still hits the stamped key"
    );
    let mut live = 0;
    for (pos, slot) in &hits[3] {
        let chunk = &entry.chunks[*pos];
        let (staged, _) = e.stage_cold_chunk(chunk, rtx).unwrap().ready().unwrap();
        if e.materialize_cold_chunk_slot(&table, chunk, &staged, *slot as usize, rtx)
            .expect("no decline")
            .is_some()
        {
            live += 1;
        }
    }
    assert_eq!(live, 0, "the stamped hit is masked at the recheck");
    // The absent key: no hits at all.
    assert!(hits[4].is_empty(), "an absent key misses every chunk index");

    // THE FOLD PATH (audit HIGH regression: per-column-parallel blob_offsets — a single int8 key
    // folds on-device; the needle is the host fingerprint via the shared helper): build indexes
    // over a BIGINT column and probe present/absent keys through fingerprints.
    seq += 1;
    e.execute_text(seq, "CREATE TABLE keyed8 (k BIGINT, v INT)")
        .unwrap();
    let mut v8 = String::new();
    for i in 0..600i64 {
        if i > 0 {
            v8.push(',');
        }
        v8.push_str(&format!("({}, {})", i * 1_000_000_007, i));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO keyed8 (k, v) VALUES {v8}"))
        .unwrap();
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM keyed8"))
        .unwrap();
    let table8 = e
        .catalog_snapshot()
        .relational_catalog
        .get("keyed8")
        .cloned()
        .unwrap();
    let entry8 = e
        .read_state
        .residency
        .streaming_cold_chunks
        .load()
        .get("keyed8")
        .cloned()
        .expect("keyed8 cold entry");
    let indexes8 = e
        .ensure_chunk_key_indexes(&table8, &entry8, &[0], 0)
        .expect("int8-key indexes build (the fold path)");
    let present = Engine::chunk_key_needle(
        &table8,
        &[0],
        &[SqlValue::Int8(5 * 1_000_000_007), SqlValue::Int4(5)],
    )
    .expect("needle");
    let absent = Engine::chunk_key_needle(
        &table8,
        &[0],
        &[SqlValue::Int8(999_999_999_999), SqlValue::Int4(0)],
    )
    .expect("needle");
    let hits8 = e
        .probe_chunk_key_indexes(&indexes8, &[present, absent])
        .expect("probe");
    let rtx8 = e.committed_seq();
    let mut live = 0;
    for (pos, slot) in &hits8[0] {
        let chunk = &entry8.chunks[*pos];
        let (staged, _) = e.stage_cold_chunk(chunk, rtx8).unwrap().ready().unwrap();
        if let Some(row) = e
            .materialize_cold_chunk_slot(&table8, chunk, &staged, *slot as usize, rtx8)
            .expect("no decline")
        {
            if row[0] == SqlValue::Int8(5 * 1_000_000_007) {
                live += 1;
            }
        }
    }
    assert_eq!(live, 1, "the folded int8 key locates its exact row");
    // The absent fingerprint may collide (32-bit) — every hit must FAIL the recheck.
    for (pos, slot) in &hits8[1] {
        let chunk = &entry8.chunks[*pos];
        let (staged, _) = e.stage_cold_chunk(chunk, rtx8).unwrap().ready().unwrap();
        if let Some(row) = e
            .materialize_cold_chunk_slot(&table8, chunk, &staged, *slot as usize, rtx8)
            .expect("no decline")
        {
            assert_ne!(
                row[0],
                SqlValue::Int8(999_999_999_999),
                "collision resolved by recheck"
            );
        }
    }
}

/// P5-2 — THE KEYED-CLASS LIFT (INSERT): a PK'd over-budget table ENTERS the class (eligibility
/// no longer refuses unique indexes); its host rows are RECLAIMED; INSERT uniqueness is then
/// validated ON-DEVICE (per-chunk key-index probe + P5-0 slot recheck at the statement
/// snapshot): a genuine dup rejects WITHOUT de-auth, an in-batch dup rejects host-exact, a
/// tombstoned key re-inserts (a masked hit is NOT a conflict), and fresh keys append as tails.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_lift_insert_uniqueness() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE ku (a INT PRIMARY KEY, t TEXT)")
        .unwrap();
    const N: i32 = 1200;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO ku (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let q_count = select("SELECT COUNT(*) FROM ku");
    let count = |e: &Engine| -> i64 {
        match e.execute_relational_select(&q_count).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        }
    };
    let _ = count(&e);
    seq += 1;
    e.execute_text(seq, "INSERT INTO ku (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(
        e.chunk_class_entries(),
        1,
        "the KEYED table must ENTER the class (the P5-2 lift)"
    );
    assert!(
        e.chunk_class_reclaimed_rows() > 0,
        "entry reclaims the host rows"
    );

    // A genuine duplicate vs a BASE chunk: rejected on-device, class INTACT.
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO ku (a, t) VALUES (500, 'dup')")
        .expect_err("dup key 500 must reject");
    assert!(
        format!("{err:?}").contains("duplicate key value"),
        "unique violation, got {err:?}"
    );
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "a dup rejection must NOT de-auth"
    );
    assert!(
        e.chunk_class_unique_probe_conflicts() >= 1,
        "the conflict came from the device probe's recheck"
    );

    // A duplicate vs a TAIL chunk (the enter row): the tail's index builds lazily and probes.
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO ku (a, t) VALUES (100000, 'dup-tail')")
        .expect_err("dup key 100000 must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0);

    // An IN-BATCH duplicate: host-exact structural check inside the class preflight.
    seq += 1;
    let err = e
        .execute_text(
            seq,
            "INSERT INTO ku (a, t) VALUES (777001, 'x'), (777001, 'y')",
        )
        .expect_err("in-batch dup must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0);
    assert_eq!(count(&e), i64::from(N) + 1, "no rejected row ever landed");

    // Fresh keys append as tails; the probe VALIDATED (non-vacuity) and the class held.
    let probes_before = e.chunk_class_unique_probes();
    let skipped_before = e.chunk_class_skipped_installs();
    seq += 1;
    e.execute_text(seq, "INSERT INTO ku (a, t) VALUES (600000, 'fresh')")
        .unwrap();
    assert!(
        e.chunk_class_unique_probes() > probes_before,
        "the accept path went through the device probe"
    );
    assert!(
        e.chunk_class_skipped_installs() > skipped_before,
        "still classed"
    );
    assert_eq!(count(&e), i64::from(N) + 2);

    // Tombstone-then-reinsert: the probe HITS the dead slot, the recheck masks it at the
    // statement snapshot — NOT a conflict.
    seq += 1;
    e.execute_text(seq, "DELETE FROM ku WHERE a = 500").unwrap();
    assert_eq!(e.chunk_class_deauths(), 0, "class DELETE stays classed");
    seq += 1;
    e.execute_text(seq, "INSERT INTO ku (a, t) VALUES (500, 'reborn')")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 0, "a masked hit is NOT a conflict");
    assert_eq!(count(&e), i64::from(N) + 2);
}

/// P5-2 — C1 SELF-EXCLUSION: an UPDATE's own located coordinates are SELF, not conflicts (the
/// old versions are live at probe time — stamps land in the commit hook). Key-preserving
/// multi-row updates pass; a key change INTO an existing key rejects; a key change to a fresh
/// key frees the old key for re-insert.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_update_self_exclusion() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE kv (a INT PRIMARY KEY, t TEXT)")
        .unwrap();
    const N: i32 = 1000;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kv (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kv"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kv (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);

    // Key-preserving multi-row UPDATE: every new image's key HITS its own located slot — all
    // self-excluded, zero conflicts, class intact.
    let conflicts_before = e.chunk_class_unique_probe_conflicts();
    seq += 1;
    e.execute_text(seq, "UPDATE kv SET t = 'self' WHERE a < 50")
        .unwrap();
    assert_eq!(
        e.chunk_class_unique_probe_conflicts(),
        conflicts_before,
        "C1: self-hits are NOT conflicts"
    );
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "key-preserving UPDATE stays classed"
    );

    // A key change INTO an existing key: a genuine conflict (the hit is NOT self).
    seq += 1;
    let err = e
        .execute_text(seq, "UPDATE kv SET a = 43 WHERE a = 44")
        .expect_err("44 -> 43 collides with the live 43");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0);

    // A key change to a FRESH key: passes; the old key is then free for re-insert and the new
    // key is taken.
    seq += 1;
    e.execute_text(seq, "UPDATE kv SET a = 999999 WHERE a = 45")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kv (a, t) VALUES (45, 'reused')")
        .unwrap();
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO kv (a, t) VALUES (999999, 'taken')")
        .expect_err("the moved-to key is live");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0, "the whole arc stayed classed");
}

/// P5-2 — FOLD-PATH NEEDLE PARITY + FINGERPRINT COLLISION: a COMPOUND key (two int4 columns)
/// builds its chunk indexes over device-folded fingerprints and probes with the HOST-derived
/// twin (`chunk_key_needle`) — a derivation mismatch is a silent all-miss (dup accepted), so the
/// dup rejection here IS the parity proof. Then the adversarial case: two DISTINCT keys with
/// COLLIDING 32-bit fingerprints — the colliding insert must be ACCEPTED (the full-tuple
/// recheck distinguishes), the true dup still rejects.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_compound_fold_parity_and_collision() {
    // Host-side birthday search for a fingerprint collision on (5, j): two j values whose key
    // word-vectors [w(5), w(j)] fold to the SAME fingerprint.
    let word = |v: i32| {
        crate::engine_residency::sql_value_key_words(SqlType::Int4, &SqlValue::Int4(v)).unwrap()
    };
    let fp = |a: i32, b: i32| {
        let mut words = word(a);
        words.extend(word(b));
        crate::engine_residency::compound_key_fingerprint(&words)
    };
    // CONSTRUCT the collision (a birthday search cannot find one here: the fingerprint's final
    // per-word round is a BIJECTION of the last word, so fp(a1,b1) == fp(a2,b2) reduces to
    // h1(a1) ^ h1(a2) == b1 ^ b2 — vanishingly rare over a small grid). Bucket first-word
    // states by their TOP 12 BITS; two same-bucket states differ by x < 2^20, and b2 = b1 ^ x
    // completes the pair. `step` replicates the fingerprint's per-word round FOR THE SEARCH
    // ONLY — the REAL `compound_key_fingerprint` verifies the constructed pair below (drift in
    // the round fails that assert loudly, never a silent mis-gate).
    let step = |h: u32, w: i32| -> u32 {
        let h = (h ^ (w as u32)).wrapping_mul(0x0100_0193);
        h.rotate_left(13).wrapping_add(0x9E37_79B1)
    };
    let mut buckets: std::collections::HashMap<u32, (i32, u32)> = std::collections::HashMap::new();
    let mut found: Option<((i32, i32), (i32, i32))> = None;
    // Outside the filler key space (filler a < 1000, b = 3i < 3000; skip the enter row's
    // a = 100000; b values sit at 2^20 +- x, far above every filler b).
    let mut a: i32 = 10_000;
    while found.is_none() {
        assert!(a < 2_000_000, "no same-bucket first-word pair found");
        if a != 100_000 {
            let h1 = step(0x811C_9DC5, word(a)[0]);
            if let Some((a_prev, h_prev)) = buckets.insert(h1 >> 20, (a, h1)) {
                let x = (h_prev ^ h1) as i32;
                let b1 = 1_i32 << 20;
                found = Some(((a_prev, b1), (a, b1 ^ x)));
            }
        }
        a += 1;
    }
    let ((a1, b1), (a2, b2)) = found.unwrap();
    assert_eq!(
        fp(a1, b1),
        fp(a2, b2),
        "the constructed pair must collide under the REAL fingerprint"
    );
    assert!((a1, b1) != (a2, b2), "distinct tuples");

    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(
        seq,
        "CREATE TABLE kc (a INT, b INT, t TEXT, PRIMARY KEY (a, b))",
    )
    .unwrap();
    const N: i32 = 1000;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {}, 'txt{:04}')", i * 3, i % 500));
    }
    // The first collision twin rides the base data.
    values.push_str(&format!(",({a1}, {b1}, 'twin1')"));
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kc (a, b, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kc"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kc (a, b, t) VALUES (100000, 0, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1, "compound-keyed table enters");

    // Parity: a true compound dup (10, 30) rejects via the folded probe.
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO kc (a, b, t) VALUES (10, 30, 'dup')")
        .expect_err("compound dup must reject (needle parity)");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0);

    // Same first column, different second: NOT a dup.
    seq += 1;
    e.execute_text(seq, "INSERT INTO kc (a, b, t) VALUES (10, 31, 'ok')")
        .unwrap();

    // The COLLIDING key: same fingerprint as (5, j1), different tuple — the recheck must ACCEPT.
    let conflicts_before = e.chunk_class_unique_probe_conflicts();
    seq += 1;
    e.execute_text(
        seq,
        &format!("INSERT INTO kc (a, b, t) VALUES ({a2}, {b2}, 'twin2')"),
    )
    .unwrap_or_else(|err| panic!("fingerprint collision must NOT reject a distinct key: {err:?}"));
    assert_eq!(
        e.chunk_class_unique_probe_conflicts(),
        conflicts_before,
        "no conflict was recorded for the collision"
    );
    // And the true dup of the twin still rejects.
    seq += 1;
    let err = e
        .execute_text(
            seq,
            &format!("INSERT INTO kc (a, b, t) VALUES ({a1}, {b1}, 'dup-twin')"),
        )
        .expect_err("the twin's true dup must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0, "the whole arc stayed classed");
}

/// P5-2 — NULL KEY DECLINE: host unique semantics are STRUCTURAL (NULL == NULL conflicts), but
/// the chunk fold reads raw payload bytes under the null bitmap — a NULL key can be neither
/// built nor probed faithfully, so the class preflight DECLINES to host (de-auth). The first
/// NULL insert succeeds on the rebuilt store; the second rejects host-side (structural dup).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_null_unique_declines_to_host() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE kn (a INT, u INT UNIQUE, t TEXT)")
        .unwrap();
    const N: i32 = 1000;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {}, 'txt{:04}')", i + 50_000, i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kn (a, u, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kn"))
        .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO kn (a, u, t) VALUES (100000, 99000, 'enter')",
    )
    .unwrap();
    assert_eq!(e.chunk_class_entries(), 1, "UNIQUE-column table enters");

    // Sanity: the device probe is live for non-NULL keys.
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO kn (a, u, t) VALUES (1, 50001, 'dup')")
        .expect_err("dup u must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0);

    // A NULL key: DECLINE -> de-auth -> the host validates against the rebuilt store (a single
    // NULL passes).
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO kn (a, u, t) VALUES (2000000, NULL, 'null1')",
    )
    .unwrap();
    assert_eq!(
        e.chunk_class_deauths(),
        1,
        "the NULL key must decline the class (host semantics are structural)"
    );
    // The SECOND NULL: the host's structural check (NULL == NULL) rejects — the exact semantics
    // the device probe cannot reproduce, proving the decline was the right call.
    seq += 1;
    let err = e
        .execute_text(
            seq,
            "INSERT INTO kn (a, u, t) VALUES (2000001, NULL, 'null2')",
        )
        .expect_err("the second NULL is a structural dup");
    assert!(format!("{err:?}").contains("duplicate key value"));
}

/// P5-2 — THE C2 REPLAY DIFFERENTIAL: every keyed-class verdict must MATCH what recovery's
/// host-path replay would decide — a probe false-accept is a WAL-durable duplicate the replay
/// then REJECTS, i.e. an UNREPLAYABLE acked commit (an RPO violation, strictly worse than a
/// wrong answer). Drive the full keyed history through the class (accepts + rejects +
/// tombstone-reinsert + a key-moving update), crash WITHOUT a checkpoint, reopen: recovery must
/// succeed and the replayed state must be value-identical.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_replay_differential() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-p52-replay-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("db.wal");
    const N: i32 = 1000;
    let (count_live, sum_live) = {
        let mut e = Engine::new_local();
        e.commit_state_mut().wal = WalBuffer::with_durable_segment(&base);
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return;
        }
        seq += 1;
        e.execute_text(seq, "CREATE TABLE kr (a INT PRIMARY KEY, t TEXT)")
            .unwrap();
        let mut values = String::new();
        for i in 0..N {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO kr (a, t) VALUES {values}"))
            .unwrap();
        e.set_relational_residency_budget_bytes(0, 8192);
        let _ = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM kr"))
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (100000, 'enter')")
            .unwrap();
        assert_eq!(e.chunk_class_entries(), 1, "premise: classed");

        // The adversarial history: device-accepted commits interleaved with device-rejected
        // statements (the rejects must NOT be in the WAL), a tombstone re-insert, and a
        // key-moving update.
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (600000, 'fresh')")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (500, 'dup')")
            .expect_err("dup rejected live");
        seq += 1;
        e.execute_text(seq, "DELETE FROM kr WHERE a = 500").unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (500, 'reborn')")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "UPDATE kr SET a = 999999 WHERE a = 45")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (45, 'reused')")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (999999, 'taken')")
            .expect_err("moved-to key rejected live");
        assert_eq!(
            e.chunk_class_deauths(),
            0,
            "the whole history stayed classed"
        );

        let count = match e
            .execute_relational_select(&select("SELECT COUNT(*) FROM kr"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        };
        let sum = match e
            .execute_relational_select(&select("SELECT SUM(a) FROM kr"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("sum: {other:?}"),
        };
        (count, sum)
        // DROP = the crash: no checkpoint, the WAL is the only truth.
    };

    // Recovery replays the acked history through the HOST path — it must accept every acked
    // commit (C2) and land value-identical.
    let e = Engine::open_durable_wal_segment(&base).expect("recovery must replay cleanly (C2)");
    let count = match e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kr"))
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("count: {other:?}"),
    };
    let sum = match e
        .execute_relational_select(&select("SELECT SUM(a) FROM kr"))
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("sum: {other:?}"),
    };
    assert_eq!(count, count_live, "replayed cardinality differs (C2)");
    assert_eq!(sum, sum_live, "replayed values differ (C2)");
    // Spot checks on the interesting keys.
    for (key, expect) in [(500, 1i64), (45, 1), (999999, 1), (600000, 1), (44, 1)] {
        let q = select(&format!("SELECT COUNT(*) FROM kr WHERE a = {key}"));
        let got = match e.execute_relational_select(&q).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("spot: {other:?}"),
        };
        assert_eq!(got, expect, "key {key}");
    }
}

/// P5-2 — THE TRANSACTION PATH: an explicit-txn INSERT's ONLY unique guard is the preflight
/// (the commit-time de-auth runs AFTER it), so the class probe must reject a dup at statement
/// time inside BEGIN/COMMIT, and accept fresh keys.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_txn_insert_dup_rejected() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE kt (a INT PRIMARY KEY, t TEXT)")
        .unwrap();
    const N: i32 = 1000;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kt (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kt"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kt (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);

    // The dup rejects INSIDE the transaction (the preflight probe). A txn's statements all
    // carry the BEGIN's seq — the txn id.
    seq += 1;
    e.execute_text(seq, "BEGIN").unwrap();
    let err = e
        .execute_text(seq, "INSERT INTO kt (a, t) VALUES (500, 'dup')")
        .expect_err("txn dup must reject at preflight");
    assert!(format!("{err:?}").contains("duplicate key value"));
    e.execute_text(seq, "ROLLBACK").unwrap();

    // A fresh key commits through the txn path.
    seq += 1;
    e.execute_text(seq, "BEGIN").unwrap();
    e.execute_text(seq, "INSERT INTO kt (a, t) VALUES (700000, 'fresh')")
        .unwrap();
    e.execute_text(seq, "COMMIT").unwrap();
    let q = select("SELECT COUNT(*) FROM kt WHERE a = 700000");
    let got = match e.execute_relational_select(&q).unwrap().rows.row(0)[0] {
        SqlValue::Int8(n) => n,
        ref other => panic!("count: {other:?}"),
    };
    assert_eq!(got, 1, "the txn insert landed");
}

/// P5-2 (audit MEDIUM) — TEXT UNIQUE KEY through the class probe: a text key folds to ONE
/// FNV-1a word over its UTF-8 bytes via the fold kernel's TEXT SENTINEL branch (widths[k]==0,
/// blob span read), and the host needle derives the SAME word — a divergence is a silent
/// all-miss dup accept, so the dup rejection is the parity proof. The tombstone re-insert and
/// the C2 replay differential ride the same history.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_text_key_probe_and_replay() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-p52-text-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("db.wal");
    const N: i32 = 1000;
    let (count_live, sum_live) = {
        let mut e = Engine::new_local();
        e.commit_state_mut().wal = WalBuffer::with_durable_segment(&base);
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return;
        }
        seq += 1;
        e.execute_text(seq, "CREATE TABLE kx (k TEXT PRIMARY KEY, v INT)")
            .unwrap();
        let mut values = String::new();
        for i in 0..N {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("('key-{i:05}', {i})"));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO kx (k, v) VALUES {values}"))
            .unwrap();
        e.set_relational_residency_budget_bytes(0, 8192);
        let _ = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM kx"))
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kx (k, v) VALUES ('enter', -1)")
            .unwrap();
        assert_eq!(e.chunk_class_entries(), 1, "the TEXT-keyed table enters");

        // Parity: the true text dup rejects via the folded probe (base chunk + tail chunk).
        seq += 1;
        let err = e
            .execute_text(seq, "INSERT INTO kx (k, v) VALUES ('key-00500', 0)")
            .expect_err("text dup must reject (fold/needle parity)");
        assert!(format!("{err:?}").contains("duplicate key value"));
        seq += 1;
        let err = e
            .execute_text(seq, "INSERT INTO kx (k, v) VALUES ('enter', 0)")
            .expect_err("tail text dup must reject");
        assert!(format!("{err:?}").contains("duplicate key value"));
        assert_eq!(e.chunk_class_deauths(), 0, "rejections stay classed");

        // Near-miss shapes: prefix / suffix / case variants are DISTINCT keys and must accept.
        seq += 1;
        e.execute_text(
            seq,
            "INSERT INTO kx (k, v) VALUES ('key-0050', 1), ('key-005000', 2), ('KEY-00500', 3)",
        )
        .unwrap();

        // Tombstone re-insert: masked hit is not a conflict.
        seq += 1;
        e.execute_text(seq, "DELETE FROM kx WHERE k = 'key-00007'")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kx (k, v) VALUES ('key-00007', 700)")
            .unwrap();
        assert_eq!(
            e.chunk_class_deauths(),
            0,
            "the whole history stayed classed"
        );

        let count = match e
            .execute_relational_select(&select("SELECT COUNT(*) FROM kx"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        };
        let sum = match e
            .execute_relational_select(&select("SELECT SUM(v) FROM kx"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("sum: {other:?}"),
        };
        (count, sum)
        // DROP = the crash.
    };

    // C2: the acked text-key history must replay cleanly through the host path.
    let e = Engine::open_durable_wal_segment(&base).expect("recovery must replay cleanly (C2)");
    let count = match e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kx"))
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("count: {other:?}"),
    };
    let sum = match e
        .execute_relational_select(&select("SELECT SUM(v) FROM kx"))
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("sum: {other:?}"),
    };
    assert_eq!(count, count_live, "replayed cardinality differs (C2)");
    assert_eq!(sum, sum_live, "replayed values differ (C2)");
}

/// P5-3 — BY-KEY DML LOCATE: an Eq-on-unique-key WHERE resolves through the chunk key-index
/// probe (one device locate + slot rechecks) instead of the full fold scan, with matches
/// materialized from the rechecked slots — the reverse-gather decoder stays off the point-DML
/// hot path. Residual non-key predicates re-filter host-side; misses and dead keys yield 0-row
/// DML; range WHERE keeps the fold. The differential twin: the same logical history driven
/// through range predicates (the fold path) must land the identical state.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_dml_key_locate() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    for table in ["kp", "kf"] {
        seq += 1;
        e.execute_text(
            seq,
            &format!("CREATE TABLE {table} (a INT PRIMARY KEY, v INT)"),
        )
        .unwrap();
        const N: i32 = 1000;
        let mut values = String::new();
        for i in 0..N {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, {})", i * 10));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO {table} (a, v) VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 8192);
    for table in ["kp", "kf"] {
        let _ = e
            .execute_relational_select(&select(&format!("SELECT COUNT(*) FROM {table}")))
            .unwrap();
        seq += 1;
        e.execute_text(
            seq,
            &format!("INSERT INTO {table} (a, v) VALUES (100000, -1)"),
        )
        .unwrap();
    }
    assert_eq!(e.chunk_class_entries(), 2, "both twins classed");

    // THE PROBE TWIN (kp): point DML by key. THE FOLD TWIN (kf): the same logical ops through
    // range predicates (`a >= k AND a <= k` is 2 non-Eq filters -> the fold locate).
    let key_locates_0 = e.chunk_class_dml_key_locates();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kp WHERE a = 500").unwrap();
    assert!(
        e.chunk_class_dml_key_locates() > key_locates_0,
        "the point DELETE rode the key probe"
    );
    seq += 1;
    e.execute_text(seq, "DELETE FROM kf WHERE a >= 500 AND a <= 500")
        .unwrap();

    seq += 1;
    e.execute_text(seq, "UPDATE kp SET v = 12345 WHERE a = 700")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "UPDATE kf SET v = 12345 WHERE a >= 700 AND a <= 700")
        .unwrap();

    // Residual predicate: key matches, non-key predicate does NOT -> 0-row DML.
    let key_locates_1 = e.chunk_class_dml_key_locates();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kp WHERE a = 701 AND v = -999")
        .unwrap();
    assert!(
        e.chunk_class_dml_key_locates() > key_locates_1,
        "the residual-predicate DELETE still rode the probe"
    );
    seq += 1;
    e.execute_text(
        seq,
        "DELETE FROM kf WHERE a >= 701 AND a <= 701 AND v = -999",
    )
    .unwrap();

    // A missing key and a DEAD key: 0-row DML, no error, still classed.
    seq += 1;
    e.execute_text(seq, "DELETE FROM kp WHERE a = 987654")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kp WHERE a = 500").unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kf WHERE a >= 987654 AND a <= 987654")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kf WHERE a >= 500 AND a <= 500")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 0, "every shape stayed classed");

    // THE DIFFERENTIAL: identical final state on both twins.
    for q in [
        "SELECT COUNT(*) FROM {T}",
        "SELECT SUM(v) FROM {T}",
        "SELECT SUM(a) FROM {T}",
        "SELECT COUNT(*) FROM {T} WHERE v = 12345",
    ] {
        let probe = e
            .execute_relational_select(&select(&q.replace("{T}", "kp")))
            .unwrap()
            .rows
            .row(0)
            .to_vec();
        let fold = e
            .execute_relational_select(&select(&q.replace("{T}", "kf")))
            .unwrap()
            .rows
            .row(0)
            .to_vec();
        assert_eq!(probe, fold, "probe/fold divergence on {q}");
    }
    // And the closed form: 1001 rows - 1 deleted; v updated on one row.
    let q = select("SELECT COUNT(*) FROM kp");
    match e.execute_relational_select(&q).unwrap().rows.row(0)[0] {
        SqlValue::Int8(n) => assert_eq!(n, 1000),
        ref other => panic!("count: {other:?}"),
    }
}

/// P5-3 (audit LOW) — COMPOUND-KEY DML through the probe: the by-key locate on a (a, b) PK
/// rides the FOLDED fingerprint needle (not the raw-i32 fast path) — a needle/build divergence
/// here is a silently MISSED DML match (lost delete/update), so the probe twin (Eq on both key
/// columns) differentials against the fold twin (range predicates) over the identical history.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_compound_dml_key_locate() {
    let mut e = Engine::new_local();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    for table in ["cp", "cf"] {
        seq += 1;
        e.execute_text(
            seq,
            &format!("CREATE TABLE {table} (a INT, b INT, v INT, PRIMARY KEY (a, b))"),
        )
        .unwrap();
        const N: i32 = 1000;
        let mut values = String::new();
        for i in 0..N {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, {}, {})", i * 3, i * 10));
        }
        seq += 1;
        e.execute_text(
            seq,
            &format!("INSERT INTO {table} (a, b, v) VALUES {values}"),
        )
        .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 8192);
    for table in ["cp", "cf"] {
        let _ = e
            .execute_relational_select(&select(&format!("SELECT COUNT(*) FROM {table}")))
            .unwrap();
        seq += 1;
        e.execute_text(
            seq,
            &format!("INSERT INTO {table} (a, b, v) VALUES (100000, 0, -1)"),
        )
        .unwrap();
    }
    assert_eq!(e.chunk_class_entries(), 2, "both compound twins classed");

    // Point DELETE + UPDATE by the FULL compound key (probe twin) vs range (fold twin).
    let key_locates_0 = e.chunk_class_dml_key_locates();
    seq += 1;
    e.execute_text(seq, "DELETE FROM cp WHERE a = 500 AND b = 1500")
        .unwrap();
    assert!(
        e.chunk_class_dml_key_locates() > key_locates_0,
        "the compound point DELETE rode the folded-needle probe"
    );
    seq += 1;
    e.execute_text(
        seq,
        "DELETE FROM cf WHERE a >= 500 AND a <= 500 AND b >= 1500 AND b <= 1500",
    )
    .unwrap();
    seq += 1;
    e.execute_text(seq, "UPDATE cp SET v = 777 WHERE a = 700 AND b = 2100")
        .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "UPDATE cf SET v = 777 WHERE a >= 700 AND a <= 700 AND b >= 2100 AND b <= 2100",
    )
    .unwrap();
    // A partial-key Eq (only `a`) does NOT cover the compound index -> the fold serves it.
    let key_locates_1 = e.chunk_class_dml_key_locates();
    seq += 1;
    e.execute_text(seq, "DELETE FROM cp WHERE a = 600").unwrap();
    assert_eq!(
        e.chunk_class_dml_key_locates(),
        key_locates_1,
        "a partial key must NOT ride the probe"
    );
    seq += 1;
    e.execute_text(seq, "DELETE FROM cf WHERE a >= 600 AND a <= 600")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 0, "every shape stayed classed");

    for q in [
        "SELECT COUNT(*) FROM {T}",
        "SELECT SUM(v) FROM {T}",
        "SELECT SUM(b) FROM {T}",
        "SELECT COUNT(*) FROM {T} WHERE v = 777",
    ] {
        let probe = e
            .execute_relational_select(&select(&q.replace("{T}", "cp")))
            .unwrap()
            .rows
            .row(0)
            .to_vec();
        let fold = e
            .execute_relational_select(&select(&q.replace("{T}", "cf")))
            .unwrap()
            .rows
            .row(0)
            .to_vec();
        assert_eq!(probe, fold, "compound probe/fold divergence on {q}");
    }
}
