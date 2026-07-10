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
    e.execute_text(*seq, "CREATE TABLE __se_probe (x INT)").unwrap();
    *seq += 1;
    e.execute_text(*seq, "INSERT INTO __se_probe VALUES (1)").unwrap();
    let snapshot = e.populate_relational_residency_snapshot("__se_probe").unwrap();
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
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)").unwrap();
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
    e.execute_text(seq, "CREATE TABLE amounts (v BIGINT)").unwrap();
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
    assert!(e.streaming_fold_hits() >= 1, "streaming fold fired for the empty table");
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
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)").unwrap();
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
    assert_eq!(result.rows.clone().into_boxed(), expected, "device-filtered projection");
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
    assert_eq!(limited.rows.clone().into_boxed(), expected_first5, "LIMIT 5 rows");
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
