use crate::Engine;

fn engine_with_commits(n: u64) -> Engine {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE t (id INT)")
        .expect("create table");
    for i in 0..n {
        engine
            .execute_text(i + 2, &format!("INSERT INTO t (id) VALUES ({i})"))
            .expect("insert");
    }
    engine
}

#[test]
fn running_max_equals_full_scan_of_map() {
    let engine = engine_with_commits(64);
    let commit = engine.commit_state();
    let scan_max = commit
        .wal_commit_timestamps_micros
        .values()
        .copied()
        .max()
        .unwrap_or(0);
    assert!(commit.wal_commit_timestamps_micros.len() >= 64);
    assert_ne!(scan_max, 0);
    assert_eq!(commit.max_commit_timestamp_micros, scan_max);
}

#[test]
fn assigned_timestamps_are_strictly_monotonic() {
    let engine = engine_with_commits(32);
    let commit = engine.commit_state();
    let mut by_txn: Vec<(gpu_db_types::TxnId, u64)> = commit
        .wal_commit_timestamps_micros
        .iter()
        .map(|(txn_id, stamp)| (*txn_id, *stamp))
        .collect();
    by_txn.sort_unstable_by_key(|(txn_id, _)| *txn_id);
    let stamps: Vec<u64> = by_txn.into_iter().map(|(_, stamp)| stamp).collect();
    assert!(stamps.len() >= 32);
    for pair in stamps.windows(2) {
        assert!(pair[1] > pair[0]);
    }
}

#[test]
fn fresh_engine_next_timestamp_is_wall_clock() {
    let engine = Engine::new_local();
    {
        let commit = engine.commit_state();
        assert_eq!(commit.max_commit_timestamp_micros, 0);
        assert!(commit.wal_commit_timestamps_micros.is_empty());
    }
    assert!(engine.next_commit_timestamp_micros() > 1);
}
