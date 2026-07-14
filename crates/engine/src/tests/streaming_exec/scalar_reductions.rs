//! STRATA S-E.1 — out-of-core streaming scalar reductions (ADR-012 / PLAN S-E).
//!
//! A table whose bytes exceed the configured per-GPU residency budget has no all-resident representation,
//! so today its aggregate would de-elide to the CPU host engine (the ADR-006 charter violation). The
//! streaming fold serves `COUNT(*)`/`SUM`/`MIN`/`MAX` OUT-OF-CORE: the visible rows are chunked to the
//! budget, each chunk uploaded + reduced ON THE DEVICE, and partials combined by one final device pass. The
//! non-vacuity proof is the fired counter + `streaming_fold_chunks > 1` (a genuine multi-chunk fold) +
//! `streaming_fold_peak_chunk_bytes <= budget` (the largest single descriptor stays bounded). S-E.5
//! lookahead overlaps at most two chunks, each targeted at budget/2. The differential is the SAME engine's
//! CPU-pinned answer with the budget cleared.

use super::{gpu_available, select};
use crate::Engine;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{Decimal128, SqlValue};

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_reduction_over_budget_stays_on_device_out_of_core() {
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
#[ignore = "requires at least two local NVIDIA GPUs"]
fn gpu_streaming_scalar_partial_combine_executes_chunks_on_multiple_gpus() {
    let Ok(runtime) = gpu_db_execution::CudaDriverRuntime::probe() else {
        return;
    };
    if runtime.snapshot().device_count < 2 {
        return;
    }

    let mut e = Engine::new_local_cpu_oracle();
    e.set_auto_admit_on_commit(false);
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE multi_gpu_fold (a INT)")
        .unwrap();
    const N: i32 = 2_000;
    let values = (0..N)
        .map(|value| format!("({value})"))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(
        seq,
        &format!("INSERT INTO multi_gpu_fold (a) VALUES {values}"),
    )
    .unwrap();
    let budget = 4_096;
    e.set_relational_residency_budget_bytes(0, budget);
    e.set_relational_residency_budget_bytes(1, budget);

    let secondary_before = e.streaming_fold_secondary_gpu_chunks();
    let result = e
        .execute_relational_select(&select("SELECT SUM(a) FROM multi_gpu_fold"))
        .unwrap();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int8((0..i64::from(N)).sum())]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert!(
        e.streaming_fold_secondary_gpu_chunks() > secondary_before,
        "at least one completed partial must come from a secondary GPU"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_reduction_bigint_sum_combines_as_numeric() {
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
