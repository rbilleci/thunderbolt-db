use super::{gpu_available, select};
use crate::Engine;
use gpu_db_sql::{SelectFilterOp, SqlValue};

// ========== P4-2a (chunk-authoritative tables): chunk-native locate + locate-driven stamp ==========

/// THE LOCATE DIFFERENTIAL: the chunk-native locate (device predicate over the chunks themselves,
/// slots back) must select EXACTLY the rows the store-driven P3 locate selects for the same
/// predicate — compared by ROW VALUES (slots translate to rows through the P4-1 decoder: on an
/// unstamped entry, decoded[slot] IS the slot's row).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_native_locate_matches_store_locate() {
    let mut e = Engine::new_local_test_engine();
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
    // Closed-form construction oracle; no retired host relational locate participates.
    let mut want: Vec<Vec<SqlValue>> = (0..N)
        .filter(|i| *i > 1200 || *i * 2 < 100)
        .map(|i| vec![SqlValue::Int4(i), SqlValue::Int4(i * 2)])
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
    let mut e = Engine::new_local_test_engine();
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
