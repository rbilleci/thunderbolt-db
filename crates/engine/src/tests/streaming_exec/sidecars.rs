use super::{gpu_available, select, ClassEntryDisabled};
use crate::Engine;
use gpu_db_sql::SqlValue;

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
    let mut e = Engine::new_local_cpu_oracle();
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
        .write_streaming_cold_checkpoint(&base, 1, boundary)
        .expect("capture runs");
    assert_eq!(
        written, 1,
        "the v2 artifact must carry the sidecar-bearing entry"
    );

    // The twin replays the identical statement history (same commit boundary), restores the
    // artifact directly, and its FIRST streaming read replays the stamped bytes.
    let mut twin = Engine::new_local_cpu_oracle();
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
    let e = Engine::new_local_cpu_oracle();
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
