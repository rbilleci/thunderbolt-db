use super::{gpu_available, select};
use crate::Engine;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_projection_over_budget_filters_on_device() {
    // S-E.2: a filtered PROJECTION over an over-budget table streams — each chunk's WHERE + column gather
    // run on the device, survivors CONCAT across chunks (scan order == the CPU pinned path's seq order).
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
