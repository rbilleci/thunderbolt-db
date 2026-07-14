use super::{gpu_available, select};
use crate::Engine;
use gpu_db_sql::{SelectFilterOp, SqlValue};

// ========== P4-2b-i (S-E.P4): the CHUNK-AUTHORITATIVE class — enter, freeze, stream, exit ==========

/// THE CLASS LIFECYCLE GATE: an over-budget, elision-INeligible (text-bearing), keyless FK-free
/// table ENTERS the class at a commit; subsequent INSERTs skip the host store (FROZEN — proven by
/// the store's version count) while the streamed reads see every row (the tail appends are the
/// materialization); an unstreamable read DE-AUTHORITIZES (the post-freeze delta replays into the
/// store) and the host path serves exactly the full data. Differential twin throughout.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_enters_freezes_streams_and_deauths() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    let mut twin = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
