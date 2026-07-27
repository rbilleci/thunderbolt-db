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
fn table_root_and_rewrite_fence_lifecycle_is_catalog_and_epoch_bounded() {
    let mut ledger = RecentCommitsLedger::default();
    ledger.tables.insert("before".to_string(), 7);
    ledger.tables.insert("dropped".to_string(), 8);
    let prior = BTreeMap::from([("before".to_string(), 41), ("dropped".to_string(), 42)]);
    let current = BTreeMap::from([("after".to_string(), 41), ("dropped".to_string(), 99)]);
    ledger.reconcile_table_roots(&prior, &current);
    assert_eq!(ledger.table_root_index("after"), 7);
    assert_eq!(ledger.table_root_index("before"), 0);
    assert_eq!(
        ledger.table_root_index("dropped"),
        0,
        "drop/recreate under one name must not inherit the old OID's root"
    );

    let read_state = ReadState::new();
    read_state.publish_table_rewrite_fences([41, 42], 10);
    read_state.prune_table_rewrite_fences(&BTreeSet::from([41]), 9);
    assert_eq!(read_state.table_rewrite_fences.load().len(), 2);
    read_state.prune_table_rewrite_fences(&BTreeSet::from([41]), 10);
    let fences = read_state.table_rewrite_fences.load();
    assert_eq!(fences.get(&41), Some(&10));
    assert!(!fences.contains_key(&42));
}

/// E2.2(a) — the integer-keyed unique-slot conflict dimension. The intent fast path claims a slot
/// as `(packed_id, i32)` with NO String allocation; the ledger must detect first-committer-wins on
/// it EXACTLY like the String slot, and — crucially — a classic-path write (which claims BOTH the
/// String slot AND the same integer slot) must conflict with an intent-path write to that slot and
/// vice versa (cross-path interop through the shared integer map).
#[test]
fn recent_commits_ledger_integer_slot_conflict_and_cross_path() {
    let mut ledger = RecentCommitsLedger::default();
    let slot_id = 0xdead_0000_0000_0007_u64; // (table_oid<<32)|column_id, opaque here

    // An intent write: ONLY the integer slot (no String slot).
    let mut intent = WriteSet::default();
    intent.unique_slots_i32.push((slot_id, 42));
    assert!(!ledger.conflicts(&intent, 5), "empty ledger: no conflict");
    ledger.record(&intent, 6);
    assert_eq!(ledger.len(), 1, "only the integer slot recorded");

    // A second intent to the SAME pk value, snapshotted before the commit at 6, conflicts.
    let mut same_pk = WriteSet::default();
    same_pk.unique_slots_i32.push((slot_id, 42));
    assert!(
        ledger.conflicts(&same_pk, 5),
        "same integer slot written after the snapshot must conflict"
    );
    assert!(
        !ledger.conflicts(&same_pk, 6),
        "equality on the snapshot is a commit already seen — no conflict"
    );
    // A different pk value on the same slot id never conflicts.
    let mut other_pk = WriteSet::default();
    other_pk.unique_slots_i32.push((slot_id, 43));
    assert!(!ledger.conflicts(&other_pk, 0));

    // CROSS-PATH: a classic-path write claims the String slot AND the mirror integer slot.
    let mut classic = WriteSet::default();
    classic.unique_slots.push(UniqueIndexSlotKey {
        table: "t".to_string(),
        column: "id".to_string(),
        value: "42".to_string(),
    });
    classic.unique_slots_i32.push((slot_id, 42));
    ledger.record(&classic, 8);
    // An intent txn snapshotted at 7 now sees the classic write at 8 via the integer map.
    assert!(
        ledger.conflicts(&same_pk, 7),
        "intent must see the classic write to the same slot (cross-path, integer map)"
    );
    // And a classic re-attempt sees prior writes via either dimension.
    assert!(ledger.conflicts(&classic, 7));

    // Prune drops the integer slots too (bounded by the active-snapshot window).
    ledger.prune_below(8);
    assert_eq!(ledger.len(), 0, "all slots at/below the floor pruned");
    assert!(!ledger.conflicts(&same_pk, 0));
}

#[test]
fn active_snapshots_track_oldest_boundary() {
    // The oldest-active read-snapshot boundary (the GC/ledger-prune floor) folds statement-local
    // guards and keyed transaction-lifetime holds into the same commit-sequence multiset.
    let mut active = ActiveSnapshots::default();
    assert_eq!(active.oldest(), None);
    active.register(10);
    active.register(7);
    active.register(7);
    active.register(12);
    active.register_transaction(
        41,
        Arc::new(TransactionSnapshot {
            characteristics: TransactionCharacteristics::REPEATABLE_READ_WRITE,
            boundary: 6,
            next_row_id: 1,
            catalog: Arc::new(CatalogSnapshot::default()),
            table_versions: BTreeMap::new(),
            resident_snapshots: Arc::new(BTreeMap::new()),
            resident_shards: Arc::new(BTreeMap::new()),
            device_authoritative_tables: Arc::new(BTreeSet::new()),
            chunk_authoritative_tables: Arc::new(BTreeMap::new()),
            delta: Arc::new(std::sync::Mutex::new({
                let resident_shards = Arc::new(BTreeMap::new());
                let streaming_cold_chunks = Arc::new(BTreeMap::new());
                TransactionDeltaState {
                    generation: 0,
                    resident_shards: Arc::clone(&resident_shards),
                    resident_shards_authority: resident_shards,
                    streaming_cold_chunks: Arc::clone(&streaming_cold_chunks),
                    streaming_cold_chunks_authority: streaming_cold_chunks,
                    operations: Vec::new(),
                    write_set: WriteSet::default(),
                    next_row_id: 1,
                    sequence_state: BTreeMap::new(),
                    sequence_state_by_oid: BTreeMap::new(),
                    sequence_value_references: Vec::new(),
                    catalog_base: None,
                    catalog_overlay: None,
                    private_gpu_bytes_by_gpu: BTreeMap::new(),
                    commit_gpu_bytes_by_gpu: BTreeMap::new(),
                }
            })),
            table_access: Arc::new(TableAccessRegistry::default()).lease(),
            rewrite_fenced_tables: Arc::new(std::sync::Mutex::new(BTreeSet::new())),
            statement_lock: Arc::new(std::sync::Mutex::new(())),
            program_owned: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            data_snapshot_acquired: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            base_streaming_cold_chunks: Arc::new(BTreeMap::new()),
            private_gpu_account: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            _resident_index_resources: Vec::new(),
            _resident_gpu_charge: Arc::new(TransactionRetainedGpuCharge::empty(Arc::new(
                std::sync::Mutex::new(BTreeMap::new()),
            ))),
        }),
    );
    assert_eq!(active.transaction_snapshot(41), Some(6));
    assert_eq!(active.oldest(), Some(6));
    assert_eq!(
        active
            .deregister_transaction(41)
            .map(|snapshot| snapshot.boundary),
        Some(6)
    );
    assert_eq!(active.transaction_snapshot(41), None);
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
    let e = Engine::new_local_test_engine();
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

    let concurrent = Engine::new_local_test_engine();
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

    let serialized = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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

    assert_recovered_relational_access_path(
        &result,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("name".to_string()),
            predicate_op: Some(SelectFilterOp::Eq),
            order_column: "id".to_string(),
            descending: false,
            matched_keys: 2,
        },
    );
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(3)]]
    );
}

// ---- Stage 0 (write-half MVCC): commit-seq oracle / stamp == read-boundary unification ----

/// Replay-determinism after R3-004: live apply and WAL replay publish identical device-visible
/// state while neither path installs normal relational versions into the retired host store.
#[test]
fn stage0_wal_replay_reproduces_device_state_without_host_versions() {
    let live = Engine::new_local_test_engine();
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

    let live_visible_up_to = live.visible_up_to();
    assert!(live.read_state.mvcc.all_versions().is_empty());

    // Replay from the durable WAL into a fresh engine.
    let recovered = Engine::recover_from_durable_wal(&live.durable_wal_records()).unwrap();
    assert!(recovered.read_state.mvcc.all_versions().is_empty());
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

/// The published read boundary and device version regions advance on the same commit sequence;
/// no host tuple chain participates in visibility.
#[test]
fn stage0_read_boundary_tracks_device_version_publication() {
    let e = Engine::new_local_test_engine();
    // commit 1: CREATE TABLE (no row versions)
    e.execute_text(500, "CREATE TABLE t (id INT)").unwrap();
    // commit 2: INSERT id=1  -> row version stamped created_by = 2
    e.execute_text(501, "INSERT INTO t (id) VALUES (1)")
        .unwrap();

    assert_eq!(e.visible_up_to(), 2);
    assert!(e.read_state.mvcc.all_versions().is_empty());
    let Command::Select(select) = parse_command("SELECT id FROM t").unwrap() else {
        panic!("expected SELECT plan");
    };
    assert_eq!(
        e.execute_relational_select(&select).unwrap().rows,
        vec![vec![SqlValue::Int4(1)]]
    );
    {
        let shards = e.read_residency_shards();
        assert_eq!(
            shards["t"].iter().map(|shard| shard.max_created_by).max(),
            Some(2)
        );
    }

    // commit 3: DELETE id=1 -> the version's deleted_by stamped = 3
    e.execute_text(502, "DELETE FROM t WHERE id = 1").unwrap();
    assert_eq!(e.visible_up_to(), 3);
    assert!(e.read_state.mvcc.all_versions().is_empty());
    assert!(e.read_residency_shards()["t"]
        .iter()
        .any(|shard| shard.deleted_by_region.is_some()));
    assert!(
        e.execute_relational_select(&select)
            .unwrap()
            .rows
            .is_empty(),
        "row hidden at the live boundary after delete"
    );
}

#[test]
fn concurrent_commits_share_group_fsyncs_and_recover_durably() {
    // D3b (group commit): concurrent committers append + apply under the commit_mutex but fsync
    // through the shared group-flush protocol. Under real concurrency the durable WAL's flush
    // groups may combine many commits into one fsync (never more fsyncs than commits); every
    // acknowledged commit must be visible AND must survive a restart from the segment.
    const WRITERS: usize = 8;
    const COMMITS_PER_WRITER: usize = 25;

    let path = test_wal_path("group-commit-concurrent");
    let e = Engine::with_durable_wal_segment(&path);
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();

    let engine = std::sync::Arc::new(e);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(WRITERS));
    let handles: Vec<_> = (0..WRITERS)
        .map(|w| {
            let engine = std::sync::Arc::clone(&engine);
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..COMMITS_PER_WRITER {
                    let id = (w * COMMITS_PER_WRITER + i) as u64;
                    let txn_id = 2 + id;
                    engine
                        .execute_dml_concurrent(
                            txn_id,
                            &format!("INSERT INTO t (id, v) VALUES ({id}, {id})"),
                        )
                        .unwrap();
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let total = WRITERS * COMMITS_PER_WRITER;
    let Command::Select(select) = parse_command("SELECT id FROM t ORDER BY id").unwrap() else {
        panic!("expected SELECT plan");
    };
    assert_eq!(
        engine
            .execute_relational_select(&select)
            .unwrap()
            .rows
            .len(),
        total,
        "every acknowledged concurrent commit is visible"
    );

    // Group accounting: all records durable; a group NEVER exceeds one fsync per commit, and the
    // whole history is flushed (no unflushed tail left behind by the protocol).
    let stats = engine.wal_group_commit_stats();
    assert_eq!(stats.durable_records, (total + 1) as u64);
    assert!(stats.flush_groups <= stats.durable_records);
    assert_eq!(engine.wal_unflushed_count(), 0);
    eprintln!(
        "group commit: {} commits in {} fsync groups (mean group size {:.2}, max {})",
        stats.durable_records,
        stats.flush_groups,
        stats.mean_group_size(),
        stats.max_group_size
    );

    // Restart-replay: every acknowledged commit is in the durable segment.
    drop(engine);
    let recovered = Engine::open_durable_wal_segment(&path).unwrap();
    assert_eq!(
        recovered
            .execute_relational_select(&select)
            .unwrap()
            .rows
            .len(),
        total,
        "every acknowledged concurrent commit survives restart recovery"
    );
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn group_fsync_failure_wedges_the_concurrent_commit_path_without_exposing_the_delta() {
    // D3b failure semantics: the group fsync runs AFTER the committer's delta is applied, so a
    // flush failure cannot roll back into a clean per-statement abort — instead the flusher
    // panics (poisoning the commit_mutex, the engine's wedge-don't-serve-torn-state policy) and
    // the sticky group failure makes later concurrent commits error out. Crucially the failed
    // commit's delta must NEVER become visible (committed_seq was not published), and a restart
    // recovers exactly the durable prefix (the un-fsynced record was never acknowledged).
    let path = test_wal_path("group-commit-wedge");
    let mut e = Engine::with_durable_wal_segment(&path);
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
    e.simulate_next_wal_flush_failure();

    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        e.execute_dml_concurrent(2, "INSERT INTO t (id, v) VALUES (7, 70)")
    }));
    assert!(
        panicked.is_err(),
        "the group flusher must wedge (panic) on a post-apply fsync failure"
    );
    assert!(
        e.is_commit_path_poisoned(),
        "the wedge poisons the commit_mutex so the façade refuses further service"
    );

    // The engine-wide fail-stop refuses reads after the applied-but-unpublished delta wedges the
    // commit path. Visibility is proved from the durable prefix after restart below; serving a
    // live snapshot here would risk exposing torn state.
    let Command::Select(select) = parse_command("SELECT id FROM t").unwrap() else {
        panic!("expected SELECT plan");
    };
    assert!(
        matches!(
            e.execute_relational_select(&select),
            Err(ExecuteError::Engine(EngineError::Durability(_)))
        ),
        "a wedged engine must fail reads closed until restart recovery"
    );

    // The sticky group failure turns later concurrent commits into errors, not panics.
    let later = e.execute_dml_concurrent(3, "INSERT INTO t (id, v) VALUES (8, 80)");
    assert!(
        matches!(later, Err(ExecuteError::Engine(EngineError::Durability(_)))),
        "later concurrent commits fail closed while wedged, got {later:?}"
    );

    // Restart-replay recovers exactly the durable prefix: the CREATE, neither INSERT.
    drop(e);
    let recovered = Engine::open_durable_wal_segment(&path).unwrap();
    assert_eq!(recovered.wal_flushed_count(), 1);
    assert!(
        recovered
            .execute_relational_select(&select)
            .unwrap()
            .rows
            .is_empty(),
        "un-fsynced commits are absent after restart recovery"
    );
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
}

/// R3-004 — mandatory device DML resolution across point, stale-version, OR, duplicate-value,
/// range, and delete/reinsert shapes. Every scenario pins its terminal semantics directly.
#[test]
fn device_dml_resolve_covers_point_or_range_and_version_churn() {
    let Command::Update(negative_range) =
        parse_command("UPDATE t SET v = 0 WHERE v > -100").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        negative_range.filters[0].value,
        SqlValue::Int4(-100),
        "the DML device predicate must retain the signed literal"
    );
    let scenarios: Vec<Vec<&str>> = vec![
        // 1. Point DELETE + point UPDATE on distinct values.
        vec![
            "DELETE FROM t WHERE id = 3",
            "UPDATE t SET v = 999 WHERE id = 5",
        ],
        // 2. STALE-ENTRY (DELETE): move id 7 -> 70, then DELETE by the OLD value — it must delete
        //    NOTHING (the index still carries the (id,7)->key entry; the fetched current version is
        //    id=70 and the predicate RECHECK excludes it). The scenario ENDS here so a wrongly
        //    deleted row diverges the final state (a follow-up delete-by-70 would mask it).
        vec![
            "UPDATE t SET id = 70 WHERE id = 7",
            "DELETE FROM t WHERE id = 7",
        ],
        // 2b. STALE-ENTRY (UPDATE): same window, an UPDATE by the OLD value must update nothing.
        vec![
            "UPDATE t SET id = 70 WHERE id = 7",
            "UPDATE t SET v = 111 WHERE id = 7",
        ],
        // 2c. The NEW value resolves through its own (appended) entry.
        vec![
            "UPDATE t SET id = 70 WHERE id = 7",
            "DELETE FROM t WHERE id = 70",
        ],
        // 3. OR groups + duplicate matches (v carries duplicates by construction).
        vec![
            "DELETE FROM t WHERE id = 1 OR id = 4",
            "UPDATE t SET v = -1 WHERE v = 20",
        ],
        // 4. Range-only predicate -> ineligible -> the scan arm under the flag (still correct).
        vec![
            "DELETE FROM t WHERE id < 3",
            "UPDATE t SET v = 123 WHERE id > 8",
        ],
        // 5. DELETE then re-insert the same value, then UPDATE by it (key/entry reuse).
        vec![
            "DELETE FROM t WHERE id = 6",
            "INSERT INTO t (id, v) VALUES (6, 606)",
            "UPDATE t SET v = 707 WHERE id = 6",
        ],
    ];
    for (i, statements) in scenarios.iter().enumerate() {
        let build = || -> Vec<Vec<SqlValue>> {
            let e = Engine::new_local();
            e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
            // v = (id % 5) * 10 -> deliberate duplicates in v.
            let values: Vec<String> = (0..10_i64)
                .map(|k| format!("({k},{})", (k % 5) * 10))
                .collect();
            e.execute_text(
                2,
                &format!("INSERT INTO t (id, v) VALUES {}", values.join(",")),
            )
            .unwrap();
            for (j, sql) in statements.iter().enumerate() {
                e.execute_text(10 + j as u64, sql).unwrap();
            }
            e.execute_relational_select_text("SELECT id, v FROM t ORDER BY id")
                .unwrap()
                .rows
                .into_boxed()
        };
        let rows = build();
        let expected_len = [9, 10, 10, 9, 8, 7, 10][i];
        assert_eq!(rows.len(), expected_len, "scenario {i}: terminal row count");
        match i {
            0 => {
                assert!(!rows
                    .iter()
                    .any(|row| row.first() == Some(&SqlValue::Int4(3))));
                assert!(rows.iter().any(|row| {
                    row.first() == Some(&SqlValue::Int4(5))
                        && row.get(1) == Some(&SqlValue::Int4(999))
                }));
            }
            1 | 2 => assert!(rows.iter().any(|row| {
                row.first() == Some(&SqlValue::Int4(70)) && row.get(1) == Some(&SqlValue::Int4(20))
            })),
            3 => assert!(!rows
                .iter()
                .any(|row| row.first() == Some(&SqlValue::Int4(70)))),
            4 => assert_eq!(
                rows.iter()
                    .filter(|row| row.get(1) == Some(&SqlValue::Int4(-1)))
                    .count(),
                2
            ),
            5 => assert!(rows.iter().any(|row| {
                row.first() == Some(&SqlValue::Int4(9)) && row.get(1) == Some(&SqlValue::Int4(123))
            })),
            6 => assert!(rows.iter().any(|row| {
                row.first() == Some(&SqlValue::Int4(6)) && row.get(1) == Some(&SqlValue::Int4(707))
            })),
            _ => unreachable!(),
        }
    }
}

/// R3-004 — constrained and range DML remain device-native and preserve unique enforcement.
#[test]
fn device_dml_resolve_enforces_unique_during_range_updates() {
    let build = || -> (Vec<Vec<SqlValue>>, String) {
        let e = Engine::new_local();
        e.execute_text(1, "CREATE TABLE p (id INT UNIQUE, v INT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO p (id, v) VALUES (1,10),(2,20),(3,30)")
            .unwrap();
        e.execute_text(3, "DELETE FROM p WHERE id = 2").unwrap();
        e.execute_text(4, "UPDATE p SET v = 99 WHERE id = 3")
            .unwrap();
        // A unique violation must still fire through the (scan-backed) validator.
        let err = e
            .execute_text(5, "UPDATE p SET id = 1 WHERE id = 3")
            .unwrap_err()
            .to_string();
        // Range-only predicate: index-ineligible -> the scan arm serves (fallback coverage).
        e.execute_text(6, "UPDATE p SET v = 7 WHERE v > -100")
            .unwrap();
        let rows = e
            .execute_relational_select_text("SELECT id, v FROM p ORDER BY id")
            .unwrap()
            .rows
            .into_boxed();
        (rows, err)
    };
    let (rows, err) = build();
    assert!(
        err.contains("duplicate key"),
        "expected unique violation: {err}"
    );
    assert_eq!(
        rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(7)],
            vec![SqlValue::Int4(3), SqlValue::Int4(7)],
        ]
    );
}

#[test]
fn durable_unique_replay_rebuilds_device_constraint_generation() {
    let path = test_wal_path("device-unique-replay");
    let engine = Engine::with_durable_wal_segment(&path);
    engine
        .execute_text(1, "CREATE TABLE t (id INT UNIQUE, v INT)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO t (id, v) VALUES (1, 10)")
        .unwrap();
    engine
        .execute_text(3, "INSERT INTO t (id, v) VALUES (2, 20)")
        .unwrap();
    drop(engine);

    let recovered = Engine::open_durable_wal_segment(&path).unwrap();
    let rows = recovered
        .execute_relational_select_text("SELECT id, v FROM t ORDER BY id")
        .unwrap()
        .rows
        .into_boxed();
    assert_eq!(
        rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(10)],
            vec![SqlValue::Int4(2), SqlValue::Int4(20)],
        ]
    );
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn commit_wave_mixed_fast_and_slow_items_stay_correct_and_recover() {
    // Ledger #6 (deterministic commit wave): concurrent commits are sequenced in waves; plain
    // INSERTs into constraint-free tables take the batched fast-run install while unique-indexed
    // inserts and UPDATEs are per-item SLOW items that flush the pending run first. Mixed
    // concurrent traffic across both classes must produce exactly the per-key expected state,
    // acknowledge only real commits, and replay identically from the durable segment.
    const WRITERS: usize = 8;
    const OPS_PER_WRITER: usize = 30;

    let path = test_wal_path("wave-mixed");
    let e = Engine::with_durable_wal_segment(&path);
    e.execute_text(1, "CREATE TABLE plain (id INT, v INT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE uniq (id INT, v INT)")
        .unwrap();
    e.execute_text(3, "CREATE UNIQUE INDEX uniq_id ON uniq (id)")
        .unwrap();

    let engine = std::sync::Arc::new(e);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(WRITERS));
    let handles: Vec<_> = (0..WRITERS)
        .map(|w| {
            let engine = std::sync::Arc::clone(&engine);
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..OPS_PER_WRITER {
                    let id = (w * OPS_PER_WRITER + i) as u64;
                    let txn = 10 + id * 3;
                    // Fast-run candidate: plain INSERT.
                    engine
                        .execute_dml_concurrent(
                            txn,
                            &format!("INSERT INTO plain (id, v) VALUES ({id}, 0)"),
                        )
                        .unwrap();
                    // Slow item: unique-indexed INSERT (disjoint ids per writer => must succeed).
                    engine
                        .execute_dml_concurrent(
                            txn + 1,
                            &format!("INSERT INTO uniq (id, v) VALUES ({id}, {id})"),
                        )
                        .unwrap();
                    // Slow item: UPDATE the plain row just inserted (flushes the fast run first,
                    // so its predicate resolution must see the earlier same-thread insert).
                    engine
                        .execute_dml_concurrent(
                            txn + 2,
                            &format!("UPDATE plain SET v = 1 WHERE id = {id}"),
                        )
                        .unwrap();
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let total = (WRITERS * OPS_PER_WRITER) as i64;
    let count = |e: &Engine, sql: &str| -> usize {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            panic!("expected SELECT");
        };
        e.execute_relational_select(&select).unwrap().rows.len()
    };
    assert_eq!(count(&engine, "SELECT id FROM plain") as i64, total);
    assert_eq!(count(&engine, "SELECT id FROM uniq") as i64, total);
    assert_eq!(
        count(&engine, "SELECT id FROM plain WHERE v = 1") as i64,
        total,
        "every UPDATE saw its own thread's earlier fast-run insert"
    );
    // A duplicate unique insert through the wave still fails with the REAL constraint error.
    let dup = engine.execute_dml_concurrent(9_000_000, "INSERT INTO uniq (id, v) VALUES (0, 0)");
    assert!(
        matches!(&dup, Err(ExecuteError::Engine(EngineError::UniqueViolation(msg))) if msg.contains("duplicate key"))
            || matches!(&dup, Err(ExecuteError::Serialization(msg)) if msg.contains("duplicate key")),
        "expected a duplicate-key failure, got {dup:?}"
    );

    // Restart-replay: the durable segment reproduces the exact same state.
    drop(engine);
    let recovered = Engine::open_durable_wal_segment(&path).unwrap();
    assert_eq!(count(&recovered, "SELECT id FROM plain") as i64, total);
    assert_eq!(count(&recovered, "SELECT id FROM uniq") as i64, total);
    assert_eq!(
        count(&recovered, "SELECT id FROM plain WHERE v = 1") as i64,
        total
    );
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
}

/// R3-004 — device validators cover:
/// unique (self-value-keeping update must NOT self-conflict; cross-row duplicate must; NULLs ARE
/// duplicates in this engine), CHECK on the new image, OUTBOUND FK (update to a missing/present
/// parent), INBOUND FK RESTRICT (delete/update-away the last provider vs a surviving provider vs
/// re-provided by the update itself), and the SELF-REFERENCING-FK fallback. Identical error
/// strings and identical final states. SABOTAGE-VERIFIED: dropping the touched-keys exclusion
/// false-conflicts the self-value update; skipping the surviving-provider probe false-fires the
/// inbound FK; skipping the inbound section misses the last-provider violation.
#[test]
fn device_dml_validators_cover_unique_check_and_foreign_keys() {
    type ValidatorOutcome = (Result<(), String>, Vec<Vec<SqlValue>>, Vec<Vec<SqlValue>>);
    let run = |setup: &[&str], stmt: &str| -> ValidatorOutcome {
        let e = Engine::new_local();
        for (i, sql) in setup.iter().enumerate() {
            e.execute_text(1 + i as u64, sql).unwrap();
        }
        let out = e
            .execute_text(500, stmt)
            .map(|_| ())
            .map_err(|err| err.to_string());
        let parent = e
            .execute_relational_select_text("SELECT id, v FROM p ORDER BY id")
            .map(|r| r.rows.into_boxed())
            .unwrap_or_default();
        let child = e
            .execute_relational_select_text("SELECT id, pid FROM c ORDER BY id")
            .map(|r| r.rows.into_boxed())
            .unwrap_or_default();
        (out, parent, child)
    };
    let scenarios: Vec<(Vec<&str>, &str, bool)> = vec![
        // 1. unique: self-value-keeping update (must NOT self-conflict).
        (
            vec![
                "CREATE TABLE p (id INT UNIQUE, v INT)",
                "INSERT INTO p (id, v) VALUES (1,10),(2,20),(3,30)",
            ],
            "UPDATE p SET v = 99 WHERE id = 2",
            false,
        ),
        // 2. unique: cross-row duplicate (must error identically).
        (
            vec![
                "CREATE TABLE p (id INT UNIQUE, v INT)",
                "INSERT INTO p (id, v) VALUES (1,10),(2,20),(3,30)",
            ],
            "UPDATE p SET id = 1 WHERE id = 3",
            true,
        ),
        // 3. CHECK on the new image (violation) — and a passing variant.
        (
            vec![
                "CREATE TABLE p (id INT, v INT, CONSTRAINT p_v_positive CHECK (v > 0))",
                "INSERT INTO p (id, v) VALUES (1,10),(2,20)",
            ],
            "UPDATE p SET v = -5 WHERE id = 2",
            true,
        ),
        (
            vec![
                "CREATE TABLE p (id INT, v INT, CONSTRAINT p_v_positive CHECK (v > 0))",
                "INSERT INTO p (id, v) VALUES (1,10),(2,20)",
            ],
            "UPDATE p SET v = 5 WHERE id = 2",
            false,
        ),
        // 4. OUTBOUND FK: update the child's FK to a missing parent (error) / present parent (ok).
        (
            vec![
                "CREATE TABLE p (id INT UNIQUE, v INT)",
                "INSERT INTO p (id, v) VALUES (1,10),(2,20)",
                "CREATE TABLE c (id INT, pid INT)",
                "ALTER TABLE ONLY c ADD CONSTRAINT c_pid_fk FOREIGN KEY (pid) REFERENCES p(id)",
                "INSERT INTO c (id, pid) VALUES (100,1)",
            ],
            "UPDATE c SET pid = 9 WHERE id = 100",
            true,
        ),
        (
            vec![
                "CREATE TABLE p (id INT UNIQUE, v INT)",
                "INSERT INTO p (id, v) VALUES (1,10),(2,20)",
                "CREATE TABLE c (id INT, pid INT)",
                "ALTER TABLE ONLY c ADD CONSTRAINT c_pid_fk FOREIGN KEY (pid) REFERENCES p(id)",
                "INSERT INTO c (id, pid) VALUES (100,1)",
            ],
            "UPDATE c SET pid = 2 WHERE id = 100",
            false,
        ),
        // 5. INBOUND FK RESTRICT: delete the LAST provider of a referenced value (error).
        (
            vec![
                "CREATE TABLE p (id INT UNIQUE, v INT)",
                "INSERT INTO p (id, v) VALUES (1,10),(2,20)",
                "CREATE TABLE c (id INT, pid INT)",
                "ALTER TABLE ONLY c ADD CONSTRAINT c_pid_fk FOREIGN KEY (pid) REFERENCES p(id)",
                "INSERT INTO c (id, pid) VALUES (100,1)",
            ],
            "DELETE FROM p WHERE id = 1",
            true,
        ),
        // 6. INBOUND FK: delete an UNREFERENCED provider (ok).
        (
            vec![
                "CREATE TABLE p (id INT UNIQUE, v INT)",
                "INSERT INTO p (id, v) VALUES (1,10),(2,20)",
                "CREATE TABLE c (id INT, pid INT)",
                "ALTER TABLE ONLY c ADD CONSTRAINT c_pid_fk FOREIGN KEY (pid) REFERENCES p(id)",
                "INSERT INTO c (id, pid) VALUES (100,1)",
            ],
            "DELETE FROM p WHERE id = 2",
            false,
        ),
        // 7. INBOUND FK: update-away the referenced value but RE-PROVIDE it in the new image (ok).
        (
            vec![
                "CREATE TABLE p (id INT UNIQUE, v INT)",
                "INSERT INTO p (id, v) VALUES (1,10),(2,20)",
                "CREATE TABLE c (id INT, pid INT)",
                "ALTER TABLE ONLY c ADD CONSTRAINT c_pid_fk FOREIGN KEY (pid) REFERENCES p(id)",
                "INSERT INTO c (id, pid) VALUES (100,1)",
            ],
            "UPDATE p SET v = 111 WHERE id = 1",
            false,
        ),
        // 8. INBOUND FK: update-away the LAST provider's key (error).
        (
            vec![
                "CREATE TABLE p (id INT UNIQUE, v INT)",
                "INSERT INTO p (id, v) VALUES (1,10),(2,20)",
                "CREATE TABLE c (id INT, pid INT)",
                "ALTER TABLE ONLY c ADD CONSTRAINT c_pid_fk FOREIGN KEY (pid) REFERENCES p(id)",
                "INSERT INTO c (id, pid) VALUES (100,1)",
            ],
            "UPDATE p SET id = 5 WHERE id = 1",
            true,
        ),
    ];
    for (i, (setup, stmt, expect_err)) in scenarios.iter().enumerate() {
        let (out, parent, child) = run(setup, stmt);
        assert_eq!(out.is_err(), *expect_err, "scenario {i}: {out:?}");
        assert!(
            !parent.is_empty() || !child.is_empty(),
            "scenario {i}: setup remains visible"
        );
    }
}

/// Ledger #18 audit fix (the mid-flight constraint-adding DDL race): an INSERT whose OFF-LOCK
/// prepare ran BEFORE an `ADD CHECK` (or `ADD UNIQUE`) committed must be FULLY re-validated at
/// its wave re-resolve — the ledger-covered skip's coverage proof only holds while the catalog
/// generation matches (the DDL records nothing in the conflict ledger and the item's write_set
/// lacks slots for a constraint that did not exist at S). Deterministic via the instrumented
/// hook: the writer parks between prepare and enqueue while the DDL commits. Sabotage-verified:
/// granting the skip on a stamp mismatch lets the violating row COMMIT silently.
#[test]
fn wave_insert_prepared_before_add_check_is_revalidated() {
    let e = std::sync::Arc::new(Engine::new_local_test_engine());
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (id, v) VALUES (1, 10)")
        .unwrap();
    let (prepared_tx, prepared_rx) = std::sync::mpsc::channel::<()>();
    let (ddl_done_tx, ddl_done_rx) = std::sync::mpsc::channel::<()>();
    let writer = {
        let e = std::sync::Arc::clone(&e);
        std::thread::spawn(move || {
            // v = -5 violates the CHECK that commits while this item is parked post-prepare.
            e.execute_dml_concurrent_instrumented(
                100,
                "INSERT INTO t (id, v) VALUES (2, -5)",
                move || {
                    prepared_tx.send(()).unwrap();
                    ddl_done_rx
                        .recv_timeout(std::time::Duration::from_secs(30))
                        .expect("the DDL must commit while the writer is parked");
                },
            )
        })
    };
    prepared_rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the writer must reach the post-prepare hook");
    e.execute_text(
        3,
        "ALTER TABLE ONLY t ADD CONSTRAINT t_v_floor CHECK (v > 0)",
    )
    .unwrap();
    ddl_done_tx.send(()).unwrap();
    let outcome = writer.join().unwrap();
    assert!(
        outcome.is_err(),
        "the post-DDL wave re-resolve must reject the violating row (got {outcome:?})"
    );
    // The violating row must not exist; the seed row must.
    let rows = e
        .execute_relational_select_text("SELECT id, v FROM t")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1, "only the seed row survives");
}

/// One sticky engine gate owns every public service surface and releases classic work already
/// queued when a post-durable invariant failure wedges the process.
#[test]
fn central_commit_wedge_drains_classic_queue_and_rejects_reads_writes_and_drivers() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "SET live=value").unwrap();
    e.execute_text(2, "CREATE TABLE surface_gate (id INT)")
        .unwrap();
    let Command::Select(surface_select) = parse_command("SELECT id FROM surface_gate").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let ready_submission = e
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[])
        .unwrap();
    let ready_batched_submission = e
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[])
        .unwrap();
    let ready_detached_submission = e
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[])
        .unwrap();
    let text = "INSERT INTO t VALUES (1)";
    let item = e.make_covered_insert_wave_item(
        2,
        parse_command(text).unwrap(),
        crate::engine_dml_concurrent::CanonicalRequest::from_text(&e, text),
        WriteSet::default(),
        e.committed_seq(),
        e.catalog_snapshot().commit_seq,
        None,
        None,
    );
    let outcome = e.submit_commit_wave_item(item).unwrap();

    e.wedge_commit_path();
    let queued = outcome
        .take_if_done()
        .expect("wedging must settle every queued classic outcome")
        .unwrap_err();
    assert!(queued.to_string().contains("restart recovery"));
    assert!(
        !e.drive_commit_wave(),
        "a wedged driver must perform no work"
    );
    assert!(e
        .commit_mutation(3, Arc::from(&b"SET later=value"[..]))
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(e
        .execute_read_text("GET live")
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert_eq!(e.get("live"), None, "the public KV read fail-stops too");

    let hook_called = std::sync::atomic::AtomicBool::new(false);
    let select_error = e
        .execute_relational_select_instrumented(&surface_select, || {
            hook_called.store(true, AtomicOrdering::Release);
        })
        .unwrap_err();
    assert!(select_error.to_string().contains("restart recovery"));
    assert!(
        !hook_called.load(AtomicOrdering::Acquire),
        "the instrumented read must fail before binding or execution"
    );
    assert!(e
        .execute_relational_function(&SelectFunction {
            name: "missing".to_string(),
        })
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    let mvcc_query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 2 },
        filter: None,
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };
    assert!(e
        .evaluate_mvcc_query_specification(&mvcc_query)
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(e
        .execute_mvcc_query_with_cuda_driver_probe(&mvcc_query)
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(e
        .prepare_relational_retained_read_job(&surface_select)
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    let submit_error =
        match e.submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[]) {
            Err(error) => error,
            Ok(_) => panic!("a wedged engine accepted a retained-read submission"),
        };
    assert!(submit_error.to_string().contains("restart recovery"));
    assert!(e
        .complete_relational_retained_read_submission(ready_submission)
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(e
        .complete_relational_retained_read_submission_batched(ready_batched_submission)
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(ready_detached_submission
        .complete_detached()
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(e
        .submit_sharded_point_lookups_batched(&surface_select, &[1])
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(e
        .relational_retained_snapshot_handle("surface_gate")
        .is_none());
    assert!(e
        .execute_resident_plan(&surface_select)
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(e
        .execute_resident_expr_select_sql("SELECT id FROM surface_gate")
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(e
        .execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
            &[]
        )
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(e
        .relational_copy_columns("surface_gate")
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
    assert!(e
        .execute_relational_copy_rows(
            4,
            &CopyFromStdin {
                table: "surface_gate".to_string(),
                columns: Some(vec!["id".to_string()]),
                options: gpu_db_sql::CopyOptions::TEXT,
            },
            Vec::new(),
        )
        .unwrap_err()
        .to_string()
        .contains("restart recovery"));
}

#[test]
fn instrumented_autocommit_dml_rejects_an_active_transaction_identity() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE accounts (id INT)").unwrap();
    e.execute_text(90, "BEGIN").unwrap();
    let wal_before = e.durable_wal_records().len();
    let hook_called = std::sync::atomic::AtomicBool::new(false);
    let error = e
        .execute_dml_concurrent_instrumented(90, "INSERT INTO accounts (id) VALUES (1)", || {
            hook_called.store(true, AtomicOrdering::Release)
        })
        .unwrap_err();
    assert!(error.to_string().contains("autocommit-only"), "{error}");
    assert!(!hook_called.load(AtomicOrdering::Acquire));
    assert_eq!(e.durable_wal_records().len(), wal_before);
    assert!(e.transaction_snapshot_handle(90).is_some());
    e.execute_text(90, "ROLLBACK").unwrap();
}

#[test]
fn retained_completion_rejects_cross_engine_and_mid_completion_wedges() {
    let origin = Engine::new_local_test_engine();
    let other = Engine::new_local_test_engine();
    let foreign = origin
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[])
        .unwrap();
    let error = other
        .complete_relational_retained_read_submission(foreign)
        .unwrap_err();
    assert!(error.to_string().contains("different engine"), "{error}");
    let foreign_batched = origin
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[])
        .unwrap();
    let error = other
        .complete_relational_retained_read_submission_batched(foreign_batched)
        .unwrap_err();
    assert!(error.to_string().contains("different engine"), "{error}");

    let engine = Arc::new(Engine::new_local_test_engine());
    let submission = engine
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[])
        .unwrap();
    let reached = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    engine.set_retained_completion_post_hook(Arc::clone(&reached), Arc::clone(&resume));
    let completion = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || engine.complete_relational_retained_read_submission(submission))
    };
    reached.wait();
    engine.wedge_commit_path();
    resume.wait();
    let error = completion.join().unwrap().unwrap_err();
    assert!(error.to_string().contains("restart recovery"), "{error}");

    let engine = Arc::new(Engine::new_local_test_engine());
    let submission = engine
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[])
        .unwrap();
    let reached = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    engine.set_retained_completion_post_hook(Arc::clone(&reached), Arc::clone(&resume));
    let completion = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            engine.complete_relational_retained_read_submission_batched(submission)
        })
    };
    reached.wait();
    engine.wedge_commit_path();
    resume.wait();
    let error = completion.join().unwrap().unwrap_err();
    assert!(error.to_string().contains("restart recovery"), "{error}");
}
