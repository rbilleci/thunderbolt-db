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

    // FROZEN: the store's version count stops moving; the chunks carry the tails.
    let frozen_versions = e
        .read_state
        .mvcc
        .table_rows("facts")
        .store()
        .all_versions()
        .len();
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
