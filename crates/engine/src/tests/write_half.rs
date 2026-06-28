use super::*;

#[test]
fn recent_commits_ledger_conflict_record_and_prune() {
    // Unit-level proof of the SI conflict-detection + GC-boundary logic the concurrent commit
    // path relies on (write-half MVCC, Stage 4).
    let mut ledger = RecentCommitsLedger::default();
    let row = |id: u64| RowWriteKey {
        table: "t".to_string(),
        row_key: format!("rel/t/{id:020}"),
    };
    let slot = |v: &str| UniqueIndexSlotKey {
        table: "t".to_string(),
        column: "u".to_string(),
        value: v.to_string(),
    };

    // Txn at read_snapshot 5 commits at seq 6, writing row 1 and unique slot "a".
    let mut ws = WriteSet::default();
    ws.rows.push(row(1));
    ws.unique_slots.push(slot("a"));
    assert!(
        !ledger.conflicts(&ws, 5),
        "empty ledger: no conflict against any snapshot"
    );
    ledger.record(&ws, 6);
    assert_eq!(ledger.len(), 2);

    // A txn that snapshotted at 5 and writes the SAME row conflicts (row written at 6 > 5).
    let mut overlap = WriteSet::default();
    overlap.rows.push(row(1));
    assert!(
        ledger.conflicts(&overlap, 5),
        "row written after the snapshot must conflict (first-committer-wins)"
    );
    // A txn that snapshotted at 6 (saw the commit) does NOT conflict.
    assert!(
        !ledger.conflicts(&overlap, 6),
        "equality on the snapshot is a commit the txn already saw — no conflict"
    );
    // The unique-slot conflict dimension behaves the same.
    let mut slot_overlap = WriteSet::default();
    slot_overlap.unique_slots.push(slot("a"));
    assert!(ledger.conflicts(&slot_overlap, 5));
    assert!(!ledger.conflicts(&slot_overlap, 6));
    // A disjoint write (different row + slot) never conflicts.
    let mut disjoint = WriteSet::default();
    disjoint.rows.push(row(2));
    disjoint.unique_slots.push(slot("b"));
    assert!(!ledger.conflicts(&disjoint, 0));

    // Pruning below the oldest active snapshot drops entries no active txn can still win against.
    ledger.record(&disjoint, 10); // now seqs 6 and 10 are recorded
    assert_eq!(ledger.len(), 4);
    ledger.prune_below(6); // oldest active snapshot is 7 → prune <= 6
    assert_eq!(
        ledger.len(),
        2,
        "entries at seq 6 pruned, seq-10 entries kept"
    );
    assert!(
        !ledger.conflicts(&overlap, 5),
        "a pruned-out commit no longer reported (its snapshot floor moved past it)"
    );
    assert!(
        ledger.conflicts(&disjoint, 9),
        "the seq-10 entry still conflicts a snapshot below it"
    );
}

#[test]
fn active_snapshots_track_oldest_boundary() {
    // The oldest-active read-snapshot boundary (the GC/ledger-prune floor) under registration +
    // deregistration (write-half MVCC, Stage 4).
    let mut active = ActiveSnapshots::default();
    assert_eq!(active.oldest(), None);
    active.register(10);
    active.register(7);
    active.register(7);
    active.register(12);
    assert_eq!(active.oldest(), Some(7));
    active.deregister(7); // one of the two at 7 remains
    assert_eq!(active.oldest(), Some(7));
    active.deregister(7);
    assert_eq!(
        active.oldest(),
        Some(10),
        "both 7s gone → next oldest is 10"
    );
    active.deregister(10);
    active.deregister(12);
    assert_eq!(active.oldest(), None, "no in-flight snapshots");
}

#[test]
fn concurrent_dml_classification_routes_sequence_inserts_to_serialized_path() {
    // `is_concurrent_dml` gates which statements take the off-lock concurrent path vs the
    // serialized catalog-latch path (write-half MVCC, Stage 4).
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE plain (id INT, v INT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE serial_t (id SERIAL, v TEXT)")
        .unwrap();

    // Plain INSERT/UPDATE/DELETE on a base table → concurrent.
    assert!(e.is_concurrent_dml("INSERT INTO plain (id, v) VALUES (1, 2)"));
    assert!(e.is_concurrent_dml("UPDATE plain SET v = 3 WHERE id = 1"));
    assert!(e.is_concurrent_dml("DELETE FROM plain WHERE id = 1"));
    // An INSERT that consumes a nextval default → serialized (sequence mutation needs &mut self).
    assert!(!e.is_concurrent_dml("INSERT INTO serial_t (v) VALUES ('x')"));
    // UPDATE/DELETE never touch sequences → still concurrent even on the SERIAL table.
    assert!(e.is_concurrent_dml("UPDATE serial_t SET v = 'y' WHERE id = 1"));
    // DDL, KV, unknown tables, transaction control, SELECT, parse errors → not concurrent DML.
    assert!(!e.is_concurrent_dml("CREATE TABLE z (a INT)"));
    assert!(!e.is_concurrent_dml("SET k=v"));
    assert!(!e.is_concurrent_dml("INSERT INTO missing (id) VALUES (1)"));
    assert!(!e.is_concurrent_dml("BEGIN"));
    assert!(!e.is_concurrent_dml("SELECT * FROM plain"));
    assert!(!e.is_concurrent_dml("not valid sql ;;;"));
}

#[test]
fn execute_dml_concurrent_matches_the_serialized_path_single_threaded() {
    // Run the SAME mixed workload through the concurrent path (`execute_dml_concurrent`) and the
    // serialized path (`execute_text`), single-threaded, and assert identical visible state —
    // the concurrent path is behavior-preserving (the prepare→commit split + re-resolve at
    // commit_seq is byte-identical to the serialized apply).
    let read_ids = |e: &Engine| -> Vec<(i64, i64)> {
        let Command::Select(select) = parse_command("SELECT id, v FROM t ORDER BY id").unwrap()
        else {
            unreachable!()
        };
        e.execute_relational_select(&select)
            .unwrap()
            .rows
            .iter()
            .map(|row| match (&row[0], &row[1]) {
                (SqlValue::Int4(id), SqlValue::Int4(v)) => (*id as i64, *v as i64),
                other => panic!("unexpected row {other:?}"),
            })
            .collect()
    };

    let concurrent = Engine::new_local();
    concurrent
        .execute_text(1, "CREATE TABLE t (id INT, v INT)")
        .unwrap();
    concurrent
        .execute_dml_concurrent(2, "INSERT INTO t (id, v) VALUES (1, 10), (2, 20), (3, 30)")
        .unwrap();
    concurrent
        .execute_dml_concurrent(3, "UPDATE t SET v = 99 WHERE id = 2")
        .unwrap();
    concurrent
        .execute_dml_concurrent(4, "DELETE FROM t WHERE id = 1")
        .unwrap();

    let serialized = Engine::new_local();
    serialized
        .execute_text(1, "CREATE TABLE t (id INT, v INT)")
        .unwrap();
    serialized
        .execute_text(2, "INSERT INTO t (id, v) VALUES (1, 10), (2, 20), (3, 30)")
        .unwrap();
    serialized
        .execute_text(3, "UPDATE t SET v = 99 WHERE id = 2")
        .unwrap();
    serialized
        .execute_text(4, "DELETE FROM t WHERE id = 1")
        .unwrap();

    assert_eq!(read_ids(&concurrent), read_ids(&serialized));
    assert_eq!(read_ids(&concurrent), vec![(2, 99), (3, 30)]);
    // Both reached the same visibility boundary (3 commits after the CREATE).
    assert_eq!(concurrent.committed_seq(), serialized.committed_seq());
}

#[test]
fn relational_index_access_path_survives_wal_recovery() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Linus')",
    )
    .unwrap();

    let durable = e.durable_wal_records().to_vec();
    let recovered = Engine::recover_from_durable_wal(&durable).unwrap();
    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE name = 'Linus' ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = recovered.execute_relational_select(&select).unwrap();

    assert_eq!(
        *result.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("name".to_string()),
            predicate_op: Some(SelectFilterOp::Eq),
            order_column: "id".to_string(),
            descending: false,
            matched_keys: 2,
        }
    );
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(3)]]
    );
}

// ---- Stage 0 (write-half MVCC): commit-seq oracle / stamp == read-boundary unification ----

/// Replay-determinism: a WAL replay must reproduce **byte-identical** MVCC version stamps
/// (`created_by`/`deleted_by`) and identical query results versus the live-applied state.
///
/// The façade `txn_id`s used below are deliberately sparse and out of step with commit order
/// (100, 250, 9_999, 3, 77_000, ...) to prove the version stamp is derived from the commit
/// `Index` (log order), NOT from the recorded façade transaction id. If the stamp still tracked
/// the façade txn_id, the live `created_by`/`deleted_by` values would be these arbitrary numbers
/// while a replay (which re-proposes in log order) would assign 1,2,3,... — and the byte-for-byte
/// version comparison below would fail.
#[test]
fn stage0_wal_replay_reproduces_byte_identical_version_stamps() {
    let live = Engine::new_local();
    // Mix of DDL + DML, including UPDATE and DELETE so both `created_by` and `deleted_by`
    // are exercised. Sparse, non-monotonic-relative-to-commit txn_ids on purpose.
    live.execute_text(100, "CREATE TABLE acct (id INT, bal INT)")
        .unwrap();
    live.execute_text(250, "INSERT INTO acct (id, bal) VALUES (1, 10), (2, 20)")
        .unwrap();
    live.execute_text(9_999, "INSERT INTO acct (id, bal) VALUES (3, 30)")
        .unwrap();
    live.execute_text(3, "UPDATE acct SET bal = 25 WHERE id = 2")
        .unwrap();
    live.execute_text(77_000, "DELETE FROM acct WHERE id = 1")
        .unwrap();
    live.execute_text(42, "INSERT INTO acct (id, bal) VALUES (4, 40)")
        .unwrap();

    // Capture the full version set (ALL versions, visible or not) including stamps.
    let live_versions = live.read_state.mvcc.all_versions();
    let live_visible_up_to = live.visible_up_to();

    // The live stamps must be the commit `Index` sequence (1..=6 for our six commits), NOT the
    // sparse façade txn_ids — proving the decoupling at the source.
    let mut live_created: Vec<TxnId> = live_versions.iter().map(|v| v.created_by).collect();
    live_created.sort_unstable();
    live_created.dedup();
    assert!(
        live_created.iter().all(|&c| (1..=6).contains(&c)),
        "created_by stamps must be commit-Index values (1..=6), got {live_created:?}"
    );
    assert!(
        !live_created.contains(&100) && !live_created.contains(&250),
        "stamps must NOT be the façade txn_ids; got {live_created:?}"
    );
    // The DELETE of id=1 (the 5th commit) must stamp deleted_by = 5.
    let deleted_id1 = live_versions
        .iter()
        .find(|v| v.deleted_by.is_some())
        .expect("the deleted row's version must carry a deleted_by stamp");
    assert_eq!(
        deleted_id1.deleted_by,
        Some(5),
        "deleted_by must be the commit Index of the DELETE statement"
    );

    // Replay from the durable WAL into a fresh engine.
    let recovered = Engine::recover_from_durable_wal(&live.durable_wal_records()).unwrap();
    let recovered_versions = recovered.read_state.mvcc.all_versions();

    // The crux: byte-identical version chains, stamps and all.
    assert_eq!(
        recovered_versions, live_versions,
        "WAL replay must reproduce byte-identical MVCC version stamps"
    );
    assert_eq!(
        recovered.visible_up_to(),
        live_visible_up_to,
        "replay must reproduce the same read boundary"
    );

    // And identical query results.
    let Command::Select(select) =
        parse_command("SELECT id, bal FROM acct ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let live_rows = live.execute_relational_select(&select).unwrap().rows;
    let recovered_rows = recovered.execute_relational_select(&select).unwrap().rows;
    assert_eq!(recovered_rows, live_rows);
    assert_eq!(
        live_rows,
        vec![
            vec![SqlValue::Int4(2), SqlValue::Int4(25)],
            vec![SqlValue::Int4(3), SqlValue::Int4(30)],
            vec![SqlValue::Int4(4), SqlValue::Int4(40)],
        ],
        "id=1 deleted; id=2 updated; id=3,4 present"
    );
}

/// The read boundary and the version stamp are the SAME monotonic commit sequence: a row
/// committed at commit-seq `N` is visible iff the read snapshot's boundary `>= N`, and a delete
/// at commit-seq `M` hides it iff the boundary `>= M`. We assert directly against the storage
/// visibility predicate using explicit boundaries (`read_txn_id`), which is exactly the unit
/// reads thread through `visible_up_to`.
#[test]
fn stage0_read_boundary_equals_stamp_sequence() {
    let e = Engine::new_local();
    // commit 1: CREATE TABLE (no row versions)
    e.execute_text(500, "CREATE TABLE t (id INT)").unwrap();
    // commit 2: INSERT id=1  -> row version stamped created_by = 2
    e.execute_text(501, "INSERT INTO t (id) VALUES (1)")
        .unwrap();

    let row = e
        .read_state
        .mvcc
        .all_versions()
        .into_iter()
        .find(|v| v.deleted_by.is_none())
        .expect("inserted row version");
    let n = row.created_by; // the commit-seq at which the row was created
    assert_eq!(n, 2, "row created at commit Index 2");

    // Helper: how many row versions of table `t` are visible at a given boundary.
    let table_prefix = relational_key_prefix("t");
    let visible_at = |engine: &Engine, boundary: TxnId| -> usize {
        let table_rows = engine.read_state.mvcc.table_rows("t");
        let mut cursor = table_rows
            .store()
            .seq_scan_open(StorageVisibility {
                read_txn_id: boundary,
            })
            .unwrap();
        let mut count = 0;
        while let Some(tuple) = cursor.next() {
            if tuple.key.starts_with(&table_prefix) {
                count += 1;
            }
        }
        count
    };

    // Visible iff boundary >= N. (read_txn_id == 0 is the "invalid"/empty snapshot.)
    assert_eq!(visible_at(&e, n - 1), 0, "not visible below the create seq");
    assert_eq!(visible_at(&e, n), 1, "visible exactly at the create seq");
    assert_eq!(visible_at(&e, n + 100), 1, "visible above the create seq");

    // commit 3: DELETE id=1 -> the version's deleted_by stamped = 3
    e.execute_text(502, "DELETE FROM t WHERE id = 1").unwrap();
    let deleted = e
        .read_state
        .mvcc
        .all_versions()
        .into_iter()
        .find(|v| v.created_by == n)
        .expect("the original row version still present in the chain");
    let m = deleted
        .deleted_by
        .expect("row now carries a deleted_by stamp");
    assert_eq!(m, 3, "delete committed at commit Index 3");
    assert!(m > n, "delete seq strictly after create seq");

    // Between create and delete (n <= boundary < m): still visible.
    assert_eq!(visible_at(&e, n), 1, "visible at create seq, before delete");
    assert_eq!(
        visible_at(&e, m - 1),
        1,
        "still visible just below the delete seq"
    );
    // At/after the delete seq: hidden.
    assert_eq!(visible_at(&e, m), 0, "hidden exactly at the delete seq");
    assert_eq!(visible_at(&e, m + 100), 0, "hidden above the delete seq");

    // And the engine's own live boundary (visible_up_to) agrees: after the delete the row is gone.
    assert!(
        e.visible_up_to() >= m,
        "live read boundary advanced past the delete seq"
    );
    let Command::Select(select) = parse_command("SELECT id FROM t").unwrap() else {
        panic!("expected SELECT plan");
    };
    assert!(
        e.execute_relational_select(&select)
            .unwrap()
            .rows
            .is_empty(),
        "row hidden at the live boundary after delete"
    );
}
