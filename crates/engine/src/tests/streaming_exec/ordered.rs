use super::{gpu_available, select};
use crate::Engine;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_ordered_top_n_across_chunks() {
    // S-E.4: ORDER BY + LIMIT streams as a device top-N fold — each chunk's device-sorted local
    // top-(offset+limit) run concats, compaction re-sorts + re-windows on the device, and ONE final
    // device sort + the real window produces the answer. The global top-N spans chunks (ascending
    // values inserted in scan order, so the DESC winners live in the LAST chunk — a first-chunk-only
    // fold would answer wrongly).
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
