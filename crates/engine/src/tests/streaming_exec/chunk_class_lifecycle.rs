use super::{gpu_available, select};
use crate::Engine;
use gpu_db_sql::{SelectFilterOp, SqlValue};

// ========== P4-2b-i (S-E.P4): the CHUNK-AUTHORITATIVE class — enter, freeze, stream, decline ==========

/// THE CLASS LIFECYCLE GATE: an over-budget, elision-INeligible (text-bearing), keyless FK-free
/// table ENTERS the class at a commit; subsequent INSERTs skip the host store (FROZEN — proven by
/// the store's version count) while the streamed reads see every row (the tail appends are the
/// materialization); disabling streaming makes an unsupported read fail loudly without deauthorizing
/// or reconstructing a host result. Closed-form COUNT and SUM assertions own row semantics.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_enters_freezes_streams_and_fails_loudly_without_read_deauth() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    const N: i32 = 1200;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE facts (a INT, t TEXT)")
        .unwrap();
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
    // A complete cold capture enters chunk authority immediately.
    assert_eq!(count(&e), i64::from(N));
    assert_eq!(e.chunk_class_entries(), 1);
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(
        e.chunk_class_entries(),
        1,
        "the table remains in the class after its first tail append"
    );

    // RECLAIMED (P4): class entry DELETED the host chains — the store-deletion payoff; the
    // chunks are the representation. The count below is 0 and stays 0 through every class write.
    let frozen_versions = e
        .read_state
        .mvcc
        .table_rows("facts")
        .store()
        .all_versions()
        .len();
    assert_eq!(frozen_versions, 0, "the class table's host chains are GONE");
    let class_commits_before = e.chunk_class_device_commits();
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
    }
    assert_eq!(
        e.chunk_class_device_commits() - class_commits_before,
        5,
        "five commits published directly through chunk authority"
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

    // Clearing the budget disables streaming. The unsupported read must fail loudly without
    // deauthorizing the class or recreating host tuple chains.
    e.clear_relational_residency_budget_bytes(0);
    let q_rows = select("SELECT a, t FROM facts ORDER BY a");
    let fallback_before = e.metrics().snapshot().fallback_total;
    let error = e.execute_relational_select(&q_rows).unwrap_err();
    crate::tests::common::assert_gpu_relational_execution_required(
        &e,
        error,
        "facts",
        fallback_before,
    );
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "a read decline must not cross the RETIRE-002 repair boundary"
    );
    assert_eq!(
        e.read_state
            .mvcc
            .table_rows("facts")
            .store()
            .all_versions()
            .len(),
        frozen_versions,
        "a read decline must not recreate host tuple chains"
    );
}

/// P4-2b-ii — CLASS DML STAYS CLASSED: DELETE stamps the chunk-native coordinates (no de-auth,
/// no store touch); UPDATE stamps the old versions and tail-appends the new images; every
/// streamed read reflects them exactly (closed-form SUM); an UNLOWERABLE predicate still falls
/// back to the loud de-auth exit.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_dml_stamps_without_deauth() {
    let mut e = Engine::new_local_test_engine();
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
    let skipped_before = e.chunk_class_device_commits();
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (500000, 'post')")
        .unwrap();
    assert!(
        e.chunk_class_device_commits() > skipped_before,
        "still classed"
    );
    assert_eq!(sum_a(&e), expected_sum - 1000 - 7 + 500000);

    // The H2 repair-DDL exit: a representation-changing statement de-authoritizes every class
    // table BEFORE its preflight reads the store — the replayed store must be MVCC-whole (tails
    // inserted at their born boundaries, every post-freeze stamp applied as a tombstone).
    e.clear_relational_residency_budget_bytes(0);
    seq += 1;
    e.execute_text(seq, "ALTER TABLE facts ADD COLUMN z INT DEFAULT 0")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 1, "the repair DDL exits the class");
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
    let mut e = Engine::new_local_test_engine();
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
        N as usize,
        "the freeze boundary sees the base snapshot only"
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
    let mut e = Engine::new_local_test_engine();
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
    e.maybe_compact_chunk_class("facts");
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
    e.clear_relational_residency_budget_bytes(0);
    seq += 1;
    e.execute_text(seq, "ALTER TABLE facts ADD COLUMN z INT DEFAULT 0")
        .unwrap(); // the repair-DDL exit
    assert_eq!(e.chunk_class_deauths(), 1);
    assert_eq!(
        sum_a(&e),
        expected_sum - 500,
        "the de-authed store is value-exact"
    );
}

/// R3-004: COPY and a multi-entry group commit append directly to the chunk-authoritative
/// generation. Neither path may reverse-gather/deauthorize to make the host tuple store current.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_copy_and_multi_entry_batch_stay_device_authoritative() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE copy_batch (id INT, v INT)")
        .unwrap();
    let values = (0..900)
        .map(|id| format!("({id}, {})", id * 10))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(
        seq,
        &format!("INSERT INTO copy_batch (id, v) VALUES {values}"),
    )
    .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    e.transition_device_table_to_streaming_repair_above("copy_batch", 1)
        .unwrap();
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM copy_batch"))
        .unwrap();
    assert!(e.table_chunk_authoritative("copy_batch").is_some());

    let copy = gpu_db_sql::CopyFromStdin {
        table: "copy_batch".to_string(),
        columns: Some(vec!["id".to_string(), "v".to_string()]),
        options: gpu_db_sql::CopyOptions::TEXT,
    };
    e.execute_relational_copy_rows(
        10_000,
        &copy,
        vec![
            vec![SqlValue::Int4(100_000), SqlValue::Int4(1)],
            vec![SqlValue::Int4(100_001), SqlValue::Int4(2)],
        ],
    )
    .unwrap();
    assert!(
        e.table_chunk_authoritative("copy_batch").is_some(),
        "COPY must preserve chunk authority"
    );

    let payload = |sql: &str| -> std::sync::Arc<[u8]> { std::sync::Arc::from(sql.as_bytes()) };
    e.commit_mutation_batch(&[
        (10_001, payload("INSERT INTO copy_batch VALUES (100002, 3)")),
        (10_002, payload("INSERT INTO copy_batch VALUES (100003, 4)")),
        (10_003, payload("INSERT INTO copy_batch VALUES (100004, 5)")),
    ])
    .map_err(|failure| failure.error)
    .expect("multi-entry group commit");
    assert!(
        e.table_chunk_authoritative("copy_batch").is_some(),
        "multi-entry DML must preserve chunk authority"
    );
    assert_eq!(e.chunk_class_deauths(), 0);

    let count = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM copy_batch"))
        .unwrap();
    assert_eq!(
        count
            .rows
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(905)]]
    );
    let sum = e
        .execute_relational_select(&select("SELECT SUM(v) FROM copy_batch"))
        .unwrap();
    assert_eq!(
        sum.rows.iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(
            (0..900i64).map(|id| id * 10).sum::<i64>() + 15
        )]]
    );
}
