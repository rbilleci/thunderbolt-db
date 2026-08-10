use crate::{tests::test_wal_path, CoveredInsertRoute, Engine, ExecuteError};
use gpu_db_sql::SqlValue;
use std::sync::atomic::{AtomicU64, Ordering};

/// Commit one covered-insert intent through the optimized submit/poll surface.
fn commit_intent_via_submit(
    engine: &Engine,
    txn_ids: &AtomicU64,
    route: &CoveredInsertRoute,
    params: &[i32],
) -> Result<u64, ExecuteError> {
    let mut ticket = engine
        .submit_covered_insert_intent(txn_ids.fetch_add(1, Ordering::Relaxed), route, params)
        .expect("submit intent");
    let mut spins = 0u64;
    loop {
        engine.drive_commit_wave();
        if let Some(result) = engine.poll_intent(&mut ticket) {
            return result;
        }
        spins += 1;
        assert!(spins < 10_000_000, "intent never settled");
    }
}

fn select_all_rows(engine: &Engine) -> Vec<Vec<SqlValue>> {
    // ASYNC-COMMIT read-back contract (GPU_DB_SYNCHRONOUS_COMMIT=off arm):
    // acked rows become READABLE once the visible cut covers them (<= ~one
    // fence). The ORDER BY path refuses a half-visible versioned shard
    // (SV3b/SV6 wiring), so pump until the window closes instead of failing.
    let mut spins = 0u32;
    loop {
        engine.drive_commit_wave();
        match engine.execute_relational_select_text("SELECT id, v FROM t ORDER BY id") {
            Ok(result) => return result.rows.into_boxed(),
            Err(err) => {
                spins += 1;
                assert!(
                    spins < 10_000_000,
                    "visibility never caught up for ORDER BY read-back: {err}"
                );
            }
        }
    }
}

/// U1: unordered visibility read + host-side sort. Lane DELETEs version the shard
/// (deleted_by region), and the ORDER BY / GROUP BY / DISTINCT paths REFUSE versioned
/// sharded tables (the pre-existing SV3b/SV6 wiring gap — reachable now that deletes
/// exist on lane tables; VACUUM's dense rebuild un-versions). The unordered scan carries
/// the full mask-VM visibility filter, so it is the read surface for delete tests.
fn select_rows_unordered_sorted(engine: &Engine) -> Vec<Vec<SqlValue>> {
    let mut spins = 0u32;
    loop {
        engine.drive_commit_wave();
        match engine.execute_relational_select_text("SELECT id, v FROM t") {
            Ok(result) => {
                let mut rows = result.rows.into_boxed();
                rows.sort();
                return rows;
            }
            Err(err) => {
                spins += 1;
                assert!(
                    spins < 2_000_000,
                    "unordered read-back never settled: {err}"
                );
            }
        }
    }
}

/// Resolve one covered key through the same GPU visible-locate result the lane apply consumes.
/// The returned row id is ADR-014's stable entity identity, not a physical version coordinate.
fn visible_entity_id(engine: &Engine, key: i32) -> u64 {
    let table = engine.relational_catalog_table("t").unwrap();
    let snapshot = engine.committed_seq();
    let located = engine
        .wave_batch_visible_locate(&table, 0, &[key], &[snapshot])
        .expect("GPU visible-locate must answer for the covered key");
    assert_eq!(located.counts, vec![1], "covered key must resolve uniquely");
    let entity_id = located.row_ids[0];
    assert_ne!(
        entity_id,
        u64::MAX,
        "covered mutation lineage must carry a stable entity identity"
    );
    entity_id
}

/// The full E2.1 arc on real hardware: warm a PK'd int4 table into elision,
/// prepare the covered route, drive concurrent intents through the fast path
/// (wave-batched device PK validation + device open-shard apply + W5a binary
/// WAL records + canonical durability), verify duplicate-key semantics,
/// then CRASH (drop) and reopen from the canonical log — the replayed store must be
/// row-identical to the pre-crash store.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_intent_fast_path_recovers_fua_log_with_row_parity() {
    const RETRY_TXN: u64 = 9_000_001;
    const RETRY_ROW: [i32; 2] = [2_000_000, 77];
    let path = test_wal_path("intent-fua");
    let prior_durability = std::env::var("GPU_DB_WAL_DURABILITY").ok();
    std::env::set_var("GPU_DB_WAL_DURABILITY", "fua");
    let mut engine = Engine::with_durable_wal_segment(&path);
    match &prior_durability {
        Some(value) => std::env::set_var("GPU_DB_WAL_DURABILITY", value),
        None => std::env::remove_var("GPU_DB_WAL_DURABILITY"),
    }
    assert!(engine.wal_is_durable());
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    // GPU gate: this suite runs on driverless boxes too — skip without device
    // memory (the elision warm-up below can never succeed there).
    let txn_ids = AtomicU64::new(2);
    engine
        .execute_dml_concurrent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            "INSERT INTO t VALUES (1000000, 0)",
        )
        .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("t")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // Warm into elision (lazy elide-entry on the first wave-batched append).
    let mut warmed = false;
    for i in 0..10_000_i32 {
        engine
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &format!("INSERT INTO t VALUES ({}, 0)", 1_000_001 + i),
            )
            .unwrap();
        if engine.table_device_authoritative("t") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "table never entered elision on a GPU box");

    // The covered route + concurrent intents through the fast path. The preparation-lane optimizer
    // and classic/general writers share canonical transaction authority.
    // Effective mode (env OR the E2.5c-3 default): lanes >= 2 routes intents through the
    // lane pipeline, so this test must pick the matching write surface.
    let lanes_mode = crate::engine_intent_lanes::intent_lane_count() >= 2;
    let route = engine.prepare_covered_insert_route("t").unwrap();
    assert_eq!(route.table(), "t");
    assert_eq!(route.column_count(), 2);
    std::thread::scope(|scope| {
        for w in 0..4_i32 {
            let engine = &engine;
            let route = &route;
            let txn_ids = &txn_ids;
            scope.spawn(move || {
                for i in 0..100_i32 {
                    let id = w * 1_000 + i;
                    if lanes_mode {
                        commit_intent_via_submit(engine, txn_ids, route, &[id, id + 7]).unwrap();
                    } else {
                        engine
                            .execute_covered_insert_intent(
                                txn_ids.fetch_add(1, Ordering::Relaxed),
                                route,
                                &[id, id + 7],
                            )
                            .unwrap();
                    }
                }
            });
        }
        // Adversarial mixed traffic: the classic wave path interleaves with optimized intent
        // lanes. Both must draw from one replicator/WAL order and remain row-identical on replay.
        let engine = &engine;
        let txn_ids = &txn_ids;
        scope.spawn(move || {
            for i in 0..100_i32 {
                engine
                    .execute_dml_concurrent(
                        txn_ids.fetch_add(1, Ordering::Relaxed),
                        &format!("INSERT INTO t VALUES ({}, {})", 500_000 + i, i),
                    )
                    .expect("mixed classic writer");
            }
        });
    });

    if lanes_mode {
        let mut first = engine
            .submit_covered_insert_intent(RETRY_TXN, &route, &RETRY_ROW)
            .expect("stable-id first submission");
        loop {
            engine.drive_commit_wave();
            if let Some(result) = engine.poll_intent(&mut first) {
                assert_eq!(result.unwrap(), 1);
                break;
            }
        }
        let durable_records = engine.durable_wal_records().len();
        let mut retry = engine
            .submit_covered_insert_intent(RETRY_TXN, &route, &RETRY_ROW)
            .expect("same-id same-digest retry");
        assert_eq!(engine.poll_intent(&mut retry).unwrap().unwrap(), 1);
        assert_eq!(
            engine.durable_wal_records().len(),
            durable_records,
            "terminal retry must not append another canonical record"
        );
        let cross_path_exact = engine
            .execute_dml_concurrent_with_result(RETRY_TXN, "INSERT INTO t VALUES (2000000, 77)")
            .expect("classic retry of the same logical lane request must resolve terminal status");
        assert_eq!(cross_path_exact.rows_affected, 1);
        assert_eq!(engine.durable_wal_records().len(), durable_records);
        let cross_path_wal_len = engine.durable_wal_records().len();
        let cross_path_mismatch = engine
            .execute_dml_concurrent(RETRY_TXN, "INSERT INTO t VALUES (2000001, 77)")
            .expect_err("classic traffic must honor the lane's canonical transaction identity");
        assert!(
            cross_path_mismatch
                .to_string()
                .contains("claimed by a different request"),
            "{cross_path_mismatch}"
        );
        assert_eq!(
            engine.durable_wal_records().len(),
            cross_path_wal_len,
            "a cross-path stable-id mismatch must not append a second canonical record"
        );
        let mismatch =
            match engine.submit_covered_insert_intent(RETRY_TXN, &route, &[RETRY_ROW[0] + 1, 77]) {
                Err(error) => error,
                Ok(_) => panic!("same stable id with a different request must fail"),
            };
        assert!(
            mismatch
                .to_string()
                .contains("claimed by a different request"),
            "{mismatch}"
        );
    }

    // SQL semantics: a duplicate PK through the intent path raises the same
    // 23505 the classic path raises (wave-batched device locate verdict), and
    // commits nothing.
    let err = if lanes_mode {
        commit_intent_via_submit(&engine, &txn_ids, &route, &[7, 99]).unwrap_err()
    } else {
        engine
            .execute_covered_insert_intent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &route,
                &[7, 99],
            )
            .unwrap_err()
    };
    assert!(
        err.to_string()
            .contains("duplicate key value violates unique index"),
        "{err}"
    );

    // Param arity is validated against the route.
    let err = engine
        .execute_covered_insert_intent(txn_ids.fetch_add(1, Ordering::Relaxed), &route, &[5])
        .unwrap_err();
    assert!(err.to_string().contains("expects 2 params"), "{err}");

    let before = select_all_rows(&engine);
    assert_eq!(before.len(), 500 + warm_count(&before));
    // E2.5c 2M+ push (b) non-vacuity: with the fused apply flag on, the merged applies must
    // have run through the FUSED device pass (row parity alone cannot prove which kernel
    // produced the state).
    assert!(
        engine.fused_apply_hits() > 0,
        "fused apply is always on but no fused pass ever ran"
    );
    assert!(engine.wal_unflushed_count() == 0);
    assert_eq!(
        gpu_db_wal::discover_lane_count(&path).unwrap(),
        None,
        "fresh canonical optimized traffic must never create retired .lane-* WAL files"
    );
    drop(engine); // crash

    // Canonical reopen replays one WAL and continues appending. In lanes mode this test
    // additionally proves the reopened engine accepts new intent commits (the
    // route's elision re-entry arm re-admits the recovered table) and that a SECOND reopen
    // replays the post-reopen commits too.
    // Disk-authoritative FUA reopen: replay the frame log (binary row-op
    // records decode+install; no SQL re-parse for covered inserts) and verify
    // the store is row-identical.
    let mut recovered = Engine::open_durable_wal_segment(&path).unwrap();
    assert_eq!(gpu_db_wal::discover_lane_count(&path).unwrap(), None);
    let after = select_all_rows(&recovered);
    assert_eq!(before, after, "replayed store must be row-identical");
    // The recovered engine serves the same table from a valid residency
    // snapshot (the device store is reconstructible from the log alone).
    let snapshot = recovered
        .populate_relational_residency_snapshot("t")
        .expect("populate residency after recovery");
    assert!(snapshot.is_valid());
    if lanes_mode {
        // The preparation-lane optimizer does not change the product contract: a classic write
        // continues in the same canonical sequence/WAL after reopen.
        recovered
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                "INSERT INTO t VALUES (900000, 0)",
            )
            .expect("classic write after optimized intent traffic must remain supported");
        // CONTINUE APPENDING: re-arm the runtime flags, re-prepare the route (the E2.5c-1
        // elision re-entry admits the recovered table with real device backing), and commit
        // fresh intents through the reopened lane set.
        recovered.set_auto_admit_on_commit(true);
        recovered.set_device_write_locate_wave_batch_enabled(true);
        let route = recovered
            .prepare_covered_insert_route("t")
            .expect("route re-prepares after reopen (elision re-entry)");
        let wal_before_retry = recovered.durable_wal_records().len();
        let mut recovered_retry = recovered
            .submit_covered_insert_intent(RETRY_TXN, &route, &RETRY_ROW)
            .expect("recovered same-id retry");
        assert_eq!(
            recovered
                .poll_intent(&mut recovered_retry)
                .unwrap()
                .unwrap(),
            1
        );
        assert_eq!(
            recovered.durable_wal_records().len(),
            wal_before_retry,
            "recovered terminal retry must not append another canonical record"
        );
        for i in 0..50_i32 {
            commit_intent_via_submit(&recovered, &txn_ids, &route, &[10_000 + i, i])
                .expect("post-reopen intent commits");
        }
        // Duplicate of a PRE-CRASH committed PK still raises 23505 through the reopened
        // validate path (the recovered device index sees the replayed rows).
        let err = commit_intent_via_submit(&recovered, &txn_ids, &route, &[7, 1])
            .expect_err("pre-crash PK must still conflict after reopen");
        assert!(
            err.to_string()
                .contains("duplicate key value violates unique index"),
            "{err}"
        );
        let mid = select_all_rows(&recovered);
        assert_eq!(
            mid.len(),
            before.len() + 51,
            "one classic plus 50 optimized post-reopen commits visible"
        );
        drop(recovered);
        // SECOND reopen: post-reopen optimized commits replay above the first history.
        let recovered_again = Engine::open_durable_wal_segment(&path).unwrap();
        let after_again = select_all_rows(&recovered_again);
        assert_eq!(
            mid, after_again,
            "second reopen must be row-identical including post-reopen lane commits"
        );
        drop(recovered_again);
    } else {
        drop(recovered);
    }

    // Cleanup: serial sidecar + FUA frame segments + lane segments.
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
    if let (Some(parent), Some(stem)) = (path.parent(), path.file_name()) {
        let fua_prefix = format!("{}.fua.", stem.to_string_lossy());
        let lane_prefix = format!("{}.lane-", stem.to_string_lossy());
        if let Ok(entries) = std::fs::read_dir(parent) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&fua_prefix) || name.starts_with(&lane_prefix) {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }
}

/// E2.2 — warm a PK'd int4 table into elision and return the prepared covered route (shared setup
/// for the driver-multiplexed submit/poll semantics test). Returns `None` on a driverless box.
fn warm_intent_route(engine: &mut Engine, txn_ids: &AtomicU64) -> Option<CoveredInsertRoute> {
    if engine.intent_lanes.is_none() {
        engine.attach_test_intent_lanes(test_wal_path("warm-intent-lanes").into_path_buf(), 4);
    }
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine
        .execute_dml_concurrent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            "INSERT INTO t VALUES (1000000, 0)",
        )
        .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("t")
        .expect("populate residency");
    snapshot.device_memory_proof.as_ref()?;
    for i in 0..10_000_i32 {
        engine
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &format!("INSERT INTO t VALUES ({}, 0)", 1_000_001 + i),
            )
            .unwrap();
        if engine.table_device_authoritative("t") {
            return Some(engine.prepare_covered_insert_route("t").unwrap());
        }
    }
    panic!("table never entered elision on a GPU box");
}

/// PRODUCT-001: both deferred covered-write representations own their table lease. A root reset
/// must conflict while a LaneIntent or classic CommitWaveItem is queued, then succeed immediately
/// after that exact item reaches its terminal outcome.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_deferred_intent_items_retain_table_guard_until_terminal() {
    let mut engine = Engine::new_local_test_engine();
    let txn_ids = AtomicU64::new(100);
    let Some(route) = warm_intent_route(&mut engine, &txn_ids) else {
        return;
    };
    let oid = engine.relational_catalog_table("t").unwrap().oid;

    let mut lane_ticket = engine
        .submit_covered_insert_intent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            &route,
            &[2_000_001, 11],
        )
        .unwrap();
    assert!(
        engine.poll_intent(&mut lane_ticket).is_none(),
        "lane fixture must still be queued"
    );
    let lane_reset = engine.table_access.lease();
    assert!(matches!(
        lane_reset.acquire_exclusive([oid]),
        Err(ExecuteError::Serialization(_))
    ));
    loop {
        engine.drive_commit_wave();
        if let Some(result) = engine.poll_intent(&mut lane_ticket) {
            assert_eq!(result.unwrap(), 1);
            break;
        }
    }
    lane_reset.acquire_exclusive([oid]).unwrap();
    drop(lane_reset);

    // Detaching the now-empty lane strategy selects the classic async CommitWaveItem arm without
    // changing the resident route or canonical coordinator.
    engine.intent_lanes = None;
    let mut wave_ticket = engine
        .submit_covered_insert_intent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            &route,
            &[2_000_002, 12],
        )
        .unwrap();
    assert!(
        engine.poll_intent(&mut wave_ticket).is_none(),
        "classic fixture must still be queued"
    );
    let wave_reset = engine.table_access.lease();
    assert!(matches!(
        wave_reset.acquire_exclusive([oid]),
        Err(ExecuteError::Serialization(_))
    ));
    loop {
        engine.drive_commit_wave();
        if let Some(result) = engine.poll_intent(&mut wave_ticket) {
            assert_eq!(result.unwrap(), 1);
            break;
        }
    }
    wave_reset.acquire_exclusive([oid]).unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_terminal_intent_retries_precede_reset_guards_and_roots_recover() {
    fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
        gpu_db_sql::ParsedCommand::parse(sql).unwrap()
    }

    fn settle(engine: &Engine, mut ticket: crate::IntentTicket) -> u64 {
        loop {
            engine.drive_commit_wave();
            if let Some(result) = engine.poll_intent(&mut ticket) {
                return result.unwrap();
            }
        }
    }

    fn terminal(engine: &Engine, mut ticket: crate::IntentTicket) -> u64 {
        engine
            .poll_intent(&mut ticket)
            .expect("terminal retry must resolve without entering a new wave")
            .unwrap()
    }

    let mut engine = Engine::new_local_test_engine();
    let warm_ids = AtomicU64::new(100);
    let Some(insert_route) = warm_intent_route(&mut engine, &warm_ids) else {
        return;
    };

    const INSERT_TXN: u64 = 900_001;
    const UPDATE_TXN: u64 = 900_002;
    const DELETE_TXN: u64 = 900_003;
    let insert_values = [2_000_001, 11];
    assert_eq!(
        settle(
            &engine,
            engine
                .submit_covered_insert_intent(INSERT_TXN, &insert_route, &insert_values)
                .unwrap(),
        ),
        1
    );
    assert_eq!(
        engine.test_table_root_index("t"),
        engine.committed_seq(),
        "lane INSERT must advance the canonical table root"
    );
    engine.submit_transaction(910_001, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(910_001, parsed("TRUNCATE t"))
        .unwrap();
    let wal_before_retry = engine.durable_wal_records().len();
    assert_eq!(
        terminal(
            &engine,
            engine
                .submit_covered_insert_intent(INSERT_TXN, &insert_route, &insert_values)
                .unwrap(),
        ),
        1
    );
    engine
        .execute_covered_insert_intent(INSERT_TXN, &insert_route, &insert_values)
        .expect("blocking exact intent retry has no fresh table access");
    let insert_mismatch = engine
        .submit_covered_insert_intent(INSERT_TXN, &insert_route, &[2_000_002, 11])
        .map(|_| ())
        .unwrap_err();
    assert!(
        insert_mismatch.to_string().contains("different request"),
        "identity mismatch must win over reset contention: {insert_mismatch}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before_retry);
    engine
        .submit_transaction(910_001, parsed("COMMIT"))
        .unwrap();

    let insert_route = engine.prepare_covered_insert_route("t").unwrap();
    assert_eq!(
        settle(
            &engine,
            engine
                .submit_covered_insert_intent(900_010, &insert_route, &[3_000_001, 20])
                .unwrap(),
        ),
        1
    );
    let update_route = engine.prepare_covered_update_route("t").unwrap();
    let update_values = [3_000_001, 21];
    assert_eq!(
        settle(
            &engine,
            engine
                .submit_covered_update_intent(UPDATE_TXN, &update_route, &update_values)
                .unwrap(),
        ),
        1
    );
    assert_eq!(
        engine.test_table_root_index("t"),
        engine.committed_seq(),
        "nonempty lane UPDATE must advance the canonical table root"
    );
    engine.submit_transaction(910_002, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(910_002, parsed("TRUNCATE t"))
        .unwrap();
    assert_eq!(
        terminal(
            &engine,
            engine
                .submit_covered_update_intent(UPDATE_TXN, &update_route, &update_values)
                .unwrap(),
        ),
        1
    );
    let update_mismatch = engine
        .submit_covered_update_intent(UPDATE_TXN, &update_route, &[3_000_001, 22])
        .map(|_| ())
        .unwrap_err();
    assert!(update_mismatch.to_string().contains("different request"));
    engine
        .submit_transaction(910_002, parsed("COMMIT"))
        .unwrap();

    let insert_route = engine.prepare_covered_insert_route("t").unwrap();
    assert_eq!(
        settle(
            &engine,
            engine
                .submit_covered_insert_intent(900_020, &insert_route, &[4_000_001, 30])
                .unwrap(),
        ),
        1
    );
    let delete_route = engine.prepare_covered_delete_route("t").unwrap();
    assert_eq!(
        settle(
            &engine,
            engine
                .submit_covered_delete_intent(DELETE_TXN, &delete_route, 4_000_001)
                .unwrap(),
        ),
        1
    );
    assert_eq!(
        engine.test_table_root_index("t"),
        engine.committed_seq(),
        "lane DELETE must advance the canonical table root"
    );
    engine.submit_transaction(910_003, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(910_003, parsed("TRUNCATE t"))
        .unwrap();
    assert_eq!(
        terminal(
            &engine,
            engine
                .submit_covered_delete_intent(DELETE_TXN, &delete_route, 4_000_001)
                .unwrap(),
        ),
        1
    );
    let delete_mismatch = engine
        .submit_covered_delete_intent(DELETE_TXN, &delete_route, 4_000_002)
        .map(|_| ())
        .unwrap_err();
    assert!(delete_mismatch.to_string().contains("different request"));
    engine
        .submit_transaction(910_003, parsed("COMMIT"))
        .unwrap();

    let delete_route = engine.prepare_covered_delete_route("t").unwrap();
    assert_eq!(
        settle(
            &engine,
            engine
                .submit_covered_delete_intent(900_030, &delete_route, 9_999_999)
                .unwrap(),
        ),
        0
    );
    assert_eq!(
        engine.test_table_root_index("t"),
        engine.committed_seq(),
        "zero-row DELETE replay still carries a table mutation footprint"
    );
    engine.submit_transaction(910_004, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(910_004, parsed("TRUNCATE t"))
        .unwrap();
    engine
        .submit_transaction(910_004, parsed("COMMIT"))
        .unwrap();

    let update_route = engine.prepare_covered_update_route("t").unwrap();
    let root_before_zero_update = engine.test_table_root_index("t");
    assert_eq!(
        settle(
            &engine,
            engine
                .submit_covered_update_intent(900_040, &update_route, &[8_888_888, 40])
                .unwrap(),
        ),
        0
    );
    assert_eq!(
        engine.test_table_root_index("t"),
        root_before_zero_update,
        "zero-row UPDATE replay carries no AppliedRowMutation and must not move the live root"
    );
    engine.submit_transaction(910_005, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(910_005, parsed("TRUNCATE t"))
        .unwrap();
    engine
        .submit_transaction(910_005, parsed("COMMIT"))
        .unwrap();

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(
        recovered
            .execute_relational_select_text("SELECT id FROM t")
            .unwrap()
            .rows
            .is_empty(),
        "every lane mutation/reset root pair must replay to the final empty generation"
    );
}

/// Merely enabling the physical intent optimizer must not change the product write contract:
/// classic/general mutations continue through the same canonical sequence and WAL.
#[test]
fn configured_intent_lanes_do_not_reject_classic_writers() {
    let mut engine = Engine::new_local_test_engine();
    engine.attach_test_intent_lanes(
        test_wal_path("lane-classic-shared-authority").into_path_buf(),
        4,
    );
    let before_next = engine.commit_state().repl.peek_next_index();
    let before_wal = engine.wal_buffered_count();
    let token = engine
        .commit_mutation_at(77, std::sync::Arc::from(&b"SET overlap=value"[..]), 1)
        .expect("configured lanes must not reject a canonical classic write");
    assert_eq!(token.index, before_next);
    assert_eq!(
        engine.commit_state().repl.peek_next_index(),
        before_next + 1
    );
    assert_eq!(engine.wal_buffered_count(), before_wal + 1);
}

/// The engine-wide fail-stop drains lane ingress, not only the classic wave queue. This is a pure
/// host test: no lane validation/apply is driven and therefore no CUDA device is required.
#[test]
fn central_commit_wedge_drains_queued_lane_intents() {
    let path = test_wal_path("central-wedge-lane-drain");
    let engine = Engine::with_durable_wal_segment(&path);
    let Some(lanes) = engine.intent_lanes.as_ref().map(std::sync::Arc::clone) else {
        let _ = std::fs::remove_file(&path);
        return;
    };
    let outcome = crate::engine_dml_concurrent::new_pending_outcome();
    lanes.outstanding.fetch_add(1, Ordering::Relaxed);
    lanes.queues[0]
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push_back(crate::engine_dml_concurrent::LaneIntent {
            op: crate::engine_dml_concurrent::LaneOpKind::Delete,
            txn_id: 7,
            slot: (0, 7),
            read_snapshot: 0,
            prepared_catalog_seq: 0,
            filter_idx: 0,
            row_id_offset: 0,
            table: std::sync::Arc::from("t"),
            table_oid: 1,
            template: std::sync::Arc::from(&b""[..]),
            values: Vec::new(),
            outcome: std::sync::Arc::clone(&outcome),
            table_access: None,
            request_digest: [0; 32],
            transaction_claims: None,
            outstanding: Some(std::sync::Arc::clone(&lanes.outstanding)),
            rows_affected: 1,
            rows_affected_cell: Some(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0))),
        });

    engine.wedge_commit_path();
    let error = outcome
        .take_if_done()
        .expect("queued lane intent must settle on central wedge")
        .unwrap_err();
    assert!(error.to_string().contains("restart recovery"));
    assert_eq!(lanes.outstanding.load(Ordering::Relaxed), 0);
    assert!(!engine.drive_commit_wave());

    drop(engine);
    let _ = gpu_db_wal::remove_stale_lane_files(&path);
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
}

/// E2.2(c) + (a) — the driver-multiplexed submit/poll API and bounded arbitration
/// semantics. Submits a batch of distinct-PK intents WITHOUT per-commit blocking, drains them via
/// the single-writer pump, and reaps every ticket. Then proves first-committer-wins on a SAME-WAVE
/// duplicate PK (the integer conflict slot: exactly one commits, the other is a retryable conflict)
/// and a committed-dup 23505 through the async surface.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_intent_submit_poll_driver_and_conflict_semantics() {
    let mut engine = Engine::new_local_test_engine();
    let txn_ids = AtomicU64::new(2);
    let Some(route) = warm_intent_route(&mut engine, &txn_ids) else {
        return; // driverless box
    };

    // Driver-multiplexed batch: submit 300 distinct-PK intents (non-blocking), then pump + reap.
    let mut tickets: Vec<_> = (0..300_i32)
        .map(|i| {
            engine
                .submit_covered_insert_intent(
                    txn_ids.fetch_add(1, Ordering::Relaxed),
                    &route,
                    &[i, i + 1],
                )
                .expect("submit intent")
        })
        .collect();
    let mut reaped = 0usize;
    let mut spins = 0u32;
    while reaped < tickets.len() {
        engine.drive_commit_wave();
        for ticket in tickets.iter_mut() {
            if let Some(result) = engine.poll_intent(ticket) {
                result.expect("distinct-PK intent commits");
                reaped += 1;
            }
        }
        spins += 1;
        assert!(spins < 100_000, "driver failed to drain the intent batch");
    }
    let count = engine
        .execute_relational_select_text("SELECT COUNT(*) FROM t WHERE id < 1000000")
        .unwrap();
    assert!(
        format!("{:?}", count.rows.row(0).first()).contains("(300)"),
        "all 300 distinct-PK intents visible: {:?}",
        count.rows.row(0).first()
    );

    // SAME-WAVE duplicate PK: submit two intents on the same fresh PK before any pump. Exactly one
    // wins; the other is a first-committer-wins conflict caught by same-wave slot arbitration.
    let mut a = engine
        .submit_covered_insert_intent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            &route,
            &[500_000, 1],
        )
        .unwrap();
    let mut b = engine
        .submit_covered_insert_intent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            &route,
            &[500_000, 2],
        )
        .unwrap();
    let (mut ra, mut rb) = (None, None);
    let mut spins = 0u32;
    while ra.is_none() || rb.is_none() {
        engine.drive_commit_wave();
        if ra.is_none() {
            ra = engine.poll_intent(&mut a);
        }
        if rb.is_none() {
            rb = engine.poll_intent(&mut b);
        }
        spins += 1;
        assert!(spins < 100_000, "same-PW duplicate never resolved");
    }
    let wins = [ra.as_ref().unwrap(), rb.as_ref().unwrap()]
        .iter()
        .filter(|r| r.is_ok())
        .count();
    assert_eq!(
        wins, 1,
        "exactly one of two same-PK intents commits (same-wave first-committer-wins): {ra:?} / {rb:?}"
    );

    // Committed-dup through the async surface: a fresh intent on the now-committed PK 500000 is
    // rejected with 23505 (wave-batched device locate verdict) once the winner is durable.
    let mut dup = engine
        .submit_covered_insert_intent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            &route,
            &[500_000, 3],
        )
        .unwrap();
    let mut result = None;
    let mut spins = 0u32;
    while result.is_none() {
        engine.drive_commit_wave();
        result = engine.poll_intent(&mut dup);
        spins += 1;
        assert!(spins < 100_000, "committed-dup never resolved");
    }
    let err = result.unwrap().unwrap_err();
    assert!(
        err.to_string()
            .contains("duplicate key value violates unique index"),
        "committed-dup must raise 23505: {err}"
    );

    // SAME-ADMISSION transaction identity: before any pump, one request owns the shared
    // admission-to-WAL reservation. Neither an exact concurrent retry nor a mismatched request
    // may become a second queued transaction.
    let pending_txn = txn_ids.fetch_add(1, Ordering::Relaxed);
    let mut pending = engine
        .submit_covered_insert_intent(pending_txn, &route, &[550_000, 1])
        .expect("first transaction-id claimant");
    let exact_pending =
        match engine.submit_covered_insert_intent(pending_txn, &route, &[550_000, 1]) {
            Err(error) => error,
            Ok(_) => panic!("an exact concurrent retry must not queue a second transaction"),
        };
    assert!(exact_pending
        .to_string()
        .contains("is pending in canonical mutation admission"));
    let mismatch_pending =
        match engine.submit_covered_insert_intent(pending_txn, &route, &[550_001, 1]) {
            Err(error) => error,
            Ok(_) => panic!("a mismatched concurrent retry must not queue a second transaction"),
        };
    assert!(mismatch_pending.to_string().contains("different request"));
    let wal_before_pending = engine.wal_buffered_count();
    loop {
        engine.drive_commit_wave();
        if let Some(result) = engine.poll_intent(&mut pending) {
            assert_eq!(result.unwrap(), 1);
            break;
        }
    }
    assert_eq!(
        engine.wal_buffered_count(),
        wal_before_pending + 1,
        "one admitted transaction identity produces exactly one canonical WAL record"
    );

    // Sustained optimized traffic cannot starve a classic async ticket. A driver call advances
    // one lane and then services the classic queue; the classic result must resolve in a bounded
    // number of calls while optimized work is still outstanding.
    let mut backlog: Vec<_> = (0..2048_i32)
        .map(|i| {
            engine
                .submit_covered_insert_intent(
                    txn_ids.fetch_add(1, Ordering::Relaxed),
                    &route,
                    &[600_000 + i, i],
                )
                .expect("submit optimized backlog")
        })
        .collect();
    let classic_text = "INSERT INTO t VALUES (700000, 9)";
    let classic = engine.make_test_commit_wave_item(
        txn_ids.fetch_add(1, Ordering::Relaxed),
        gpu_db_sql::parse_command(classic_text).unwrap(),
        crate::engine_dml_concurrent::CanonicalRequest::from_text(&engine, classic_text),
        crate::write_path::WriteSet::default(),
        engine.committed_seq(),
    );
    let classic_outcome = engine.submit_test_commit_wave_item(classic).unwrap();
    let mut classic_result = None;
    for _ in 0..4 {
        engine.drive_commit_wave();
        if let Some(result) = classic_outcome.take_if_done() {
            classic_result = Some(result);
            break;
        }
    }
    assert_eq!(classic_result.unwrap().unwrap(), 1);
    assert!(
        engine
            .intent_lanes
            .as_ref()
            .unwrap()
            .outstanding
            .load(Ordering::Acquire)
            > 0,
        "classic completion must be demonstrated while optimized backlog remains"
    );
    let mut reaped = 0usize;
    while reaped < backlog.len() {
        engine.drive_commit_wave();
        for ticket in backlog.iter_mut() {
            if let Some(result) = engine.poll_intent(ticket) {
                result.unwrap();
                reaped += 1;
            }
        }
    }
}

/// Rows the warm-up phase inserted (ids >= 1_000_000).
fn warm_count(rows: &[Vec<SqlValue>]) -> usize {
    rows.iter()
        .filter(|row| matches!(row[0], SqlValue::Int4(id) if id >= 1_000_000))
        .count()
}

/// ADR-014 RPO-0 compatibility: `SynchronousCommit::Off` is accepted but remains strict. Mixed
/// settings and the engine default must still recover every acknowledged row.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_synchronous_commit_off_remains_durable_and_recovers() {
    let path = test_wal_path("intent-async-commit");
    let mut engine = Engine::with_durable_wal_segment(&path);
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let txn_ids = AtomicU64::new(2);
    engine
        .execute_dml_concurrent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            "INSERT INTO t VALUES (1000000, 0)",
        )
        .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("t")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // driverless box
    }
    let mut warmed = false;
    for i in 0..10_000_i32 {
        engine
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &format!("INSERT INTO t VALUES ({}, 0)", 1_000_001 + i),
            )
            .unwrap();
        if engine.table_device_authoritative("t") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "table never entered elision on a GPU box");
    let route = engine.prepare_covered_insert_route("t").unwrap();

    // MIXED SETTINGS: both compatibility values retain the same strict acknowledgement gate.
    let mut tickets: Vec<_> = (0..200_i32)
        .map(|i| {
            let mode = if i % 2 == 0 {
                crate::SynchronousCommit::Off
            } else {
                crate::SynchronousCommit::On
            };
            engine
                .submit_covered_insert_intent_with_commit(
                    txn_ids.fetch_add(1, Ordering::Relaxed),
                    &route,
                    &[i, i + 1],
                    mode,
                )
                .expect("submit intent")
        })
        .collect();
    let mut reaped = 0usize;
    let mut spins = 0u32;
    while reaped < tickets.len() {
        engine.drive_commit_wave();
        for ticket in tickets.iter_mut() {
            if let Some(result) = engine.poll_intent(ticket) {
                result.expect("distinct-PK intent commits under both settings");
                reaped += 1;
            }
        }
        spins += 1;
        assert!(spins < 10_000_000, "mixed-setting batch never drained");
    }

    // The default compatibility setter accepts Off, while plain submits remain strict.
    engine.set_synchronous_commit_default(crate::SynchronousCommit::Off);
    let mut tickets: Vec<_> = (1000..1100_i32)
        .map(|i| {
            engine
                .submit_covered_insert_intent(
                    txn_ids.fetch_add(1, Ordering::Relaxed),
                    &route,
                    &[i, i + 1],
                )
                .expect("submit intent after compatibility setting")
        })
        .collect();
    let mut reaped = 0usize;
    let mut spins = 0u32;
    while reaped < tickets.len() {
        engine.drive_commit_wave();
        for ticket in tickets.iter_mut() {
            if let Some(result) = engine.poll_intent(ticket) {
                result.expect("default intent commits strictly");
                reaped += 1;
            }
        }
        spins += 1;
        assert!(spins < 10_000_000, "default batch never drained");
    }

    // REOPEN: every acknowledged row recovers because both settings are strict.
    drop(engine);
    let recovered = Engine::open_durable_wal_segment(&path).expect("reopen recovers");
    let count = recovered
        .execute_relational_select_text("SELECT COUNT(*) FROM t WHERE id < 2000")
        .unwrap();
    assert!(
        format!("{:?}", count.rows.row(0).first()).contains("(300)"),
        "all 300 acked rows (200 mixed + 100 async-default) must survive reopen: {:?}",
        count.rows.row(0).first()
    );
}

/// U1: poll one covered-DELETE intent to completion, driving the pipeline. `Ok(n)` = rows
/// affected (0 or 1 by the covered shape).
fn commit_delete_via_submit(
    engine: &Engine,
    txn_ids: &AtomicU64,
    route: &crate::CoveredDeleteRoute,
    pk: i32,
) -> Result<u64, ExecuteError> {
    let mut ticket =
        engine.submit_covered_delete_intent(txn_ids.fetch_add(1, Ordering::Relaxed), route, pk)?;
    let mut spins = 0u64;
    loop {
        engine.drive_commit_wave();
        if let Some(result) = engine.poll_intent(&mut ticket) {
            return result;
        }
        spins += 1;
        assert!(spins < 10_000_000, "delete intent never settled");
    }
}

/// U1 END-TO-END: covered lane DELETE intents — 1-row and 0-row outcomes, dead-twin reinsert
/// (the visibility-aware rebuild), same-key races (delete/delete and delete/insert in the
/// un-settled window), rows-affected reporting, and the FIRED counters for the device
/// visible-locate + in-place tombstone paths (non-vacuity: output parity alone cannot prove
/// the device arms ran).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_lane_delete_intents_end_to_end() {
    let path = test_wal_path("lane-delete-e2e");
    let prior_durability = std::env::var("GPU_DB_WAL_DURABILITY").ok();
    std::env::set_var("GPU_DB_WAL_DURABILITY", "fua");
    let mut engine = Engine::with_durable_wal_segment(&path);
    match &prior_durability {
        Some(value) => std::env::set_var("GPU_DB_WAL_DURABILITY", value),
        None => std::env::remove_var("GPU_DB_WAL_DURABILITY"),
    }
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    // v1 exclusion probe target: a second unique i32 index makes the DELETE route refuse.
    engine
        .execute_text(2, "CREATE TABLE t2 (a INT PRIMARY KEY, b INT UNIQUE)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let txn_ids = AtomicU64::new(10);
    engine
        .execute_dml_concurrent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            "INSERT INTO t VALUES (1000000, 0)",
        )
        .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("t")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // driverless box
    }
    let mut warmed = false;
    for i in 0..10_000_i32 {
        engine
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &format!("INSERT INTO t VALUES ({}, 0)", 1_000_001 + i),
            )
            .unwrap();
        if engine.table_device_authoritative("t") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "table never entered elision on a GPU box");

    let insert_route = engine.prepare_covered_insert_route("t").unwrap();
    let delete_route = engine.prepare_covered_delete_route("t").unwrap();
    assert_eq!(delete_route.table(), "t");

    let lanes_mode = crate::engine_intent_lanes::intent_lane_count() >= 2;
    if !lanes_mode {
        // Serial arm: delete intents are lanes-only — the refusal is loud, not a fallback.
        let err = engine
            .submit_covered_delete_intent(txn_ids.fetch_add(1, Ordering::Relaxed), &delete_route, 1)
            .map(|_| ())
            .expect_err("delete intents must refuse without lanes");
        assert!(err.to_string().contains("intent-lanes"), "{err}");
        return;
    }

    // Activate lanes + seed rows 0..200; every insert reports rows_affected == 1.
    for id in 0..200_i32 {
        let rows =
            commit_intent_via_submit(&engine, &txn_ids, &insert_route, &[id, id + 7]).unwrap();
        assert_eq!(rows, 1, "covered insert reports exactly one row");
    }
    let locates_before = engine
        .read_state
        .residency
        .device_visible_locate_hits
        .load(Ordering::Relaxed);
    let tombstones_before = engine
        .read_state
        .residency
        .lane_tombstone_applies
        .load(Ordering::Relaxed);
    // 1-row delete, then the SAME key again (dead twin -> 0 rows), then a never-existed key.
    assert_eq!(
        commit_delete_via_submit(&engine, &txn_ids, &delete_route, 50).unwrap(),
        1,
        "deleting a visible row affects exactly one row"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "an in-place lane tombstone must not de-elide the table (fallback fired?)"
    );
    assert_eq!(
        commit_delete_via_submit(&engine, &txn_ids, &delete_route, 50).unwrap(),
        0,
        "re-deleting a dead key affects zero rows (pre-claim filter)"
    );
    assert_eq!(
        commit_delete_via_submit(&engine, &txn_ids, &delete_route, 987_654).unwrap(),
        0,
        "deleting a never-existing key affects zero rows"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "0-row deletes must not de-elide"
    );
    assert_eq!(
        commit_intent_via_submit(&engine, &txn_ids, &insert_route, &[987_654, 123]).unwrap(),
        1,
        "a zero-row delete must not leave a stale conflict slot that poisons a same-key insert"
    );

    // Visibility: id=50 is gone; total row count dropped by exactly one.
    let rows = select_rows_unordered_sorted(&engine);
    assert!(
        !rows
            .iter()
            .any(|row| row.first() == Some(&SqlValue::Int4(50))),
        "deleted key must not be visible"
    );
    // The GPU SELECT over a delete-versioned elided shard applies deleted_by visibility and keeps
    // relational authority on device. The covered routes are re-prepared against the current
    // generation before the next mutation.
    let insert_route = engine.prepare_covered_insert_route("t").unwrap();
    let _delete_route = engine.prepare_covered_delete_route("t").unwrap();

    // DEAD-TWIN REINSERT (the visibility-aware rebuild's reason to exist): reinserting the
    // deleted key must succeed and be visible EXACTLY once — and later deletes still locate.
    assert_eq!(
        commit_intent_via_submit(&engine, &txn_ids, &insert_route, &[50, 999]).unwrap(),
        1,
        "reinserting a deleted key succeeds"
    );
    let rows = select_rows_unordered_sorted(&engine);
    let insert_route = engine.prepare_covered_insert_route("t").unwrap();
    let delete_route = engine.prepare_covered_delete_route("t").unwrap();
    let _ = &insert_route;
    let fifty: Vec<_> = rows
        .iter()
        .filter(|row| row.first() == Some(&SqlValue::Int4(50)))
        .collect();
    assert_eq!(fifty.len(), 1, "reinserted key visible exactly once");
    assert_eq!(
        fifty[0].get(1),
        Some(&SqlValue::Int4(999)),
        "new image wins"
    );
    assert_eq!(
        commit_delete_via_submit(&engine, &txn_ids, &delete_route, 50).unwrap(),
        1,
        "the reinserted row is locatable + deletable (index survived the dead twin)"
    );

    // SAME-KEY RACES in the un-settled window: two deletes of one key — exactly one wins;
    // the loser is 0-row or a retryable serialization error, never a second deletion.
    let mut d1 = engine
        .submit_covered_delete_intent(txn_ids.fetch_add(1, Ordering::Relaxed), &delete_route, 60)
        .unwrap();
    let mut d2 = engine
        .submit_covered_delete_intent(txn_ids.fetch_add(1, Ordering::Relaxed), &delete_route, 60)
        .unwrap();
    let (mut r1, mut r2) = (None, None);
    let mut spins = 0u64;
    while r1.is_none() || r2.is_none() {
        engine.drive_commit_wave();
        if r1.is_none() {
            r1 = engine.poll_intent(&mut d1);
        }
        if r2.is_none() {
            r2 = engine.poll_intent(&mut d2);
        }
        spins += 1;
        assert!(spins < 10_000_000, "racing deletes never settled");
    }
    let outcomes = [r1.unwrap(), r2.unwrap()];
    let winners = outcomes.iter().filter(|o| matches!(o, Ok(1))).count();
    assert_eq!(winners, 1, "exactly one racing delete wins: {outcomes:?}");
    assert!(
        outcomes.iter().all(|o| match o {
            Ok(0) | Ok(1) => true,
            Err(err) =>
                err.to_string().contains("conflict") || err.to_string().contains("intra-wave"),
            _ => false,
        }),
        "loser is 0-row or retryable: {outcomes:?}"
    );

    // v1 exclusion: a second unique i32 index refuses the DELETE route at prepare.
    let err = engine.prepare_covered_delete_route("t2").unwrap_err();
    assert!(
        err.to_string().contains("ONLY unique i32 index")
            || err.to_string().contains("not covered"),
        "{err}"
    );

    // FIRED counters: the device visible-locate ran and tombstones were stamped in place
    // (a silent fallback would pass every assertion above while abandoning the design).
    assert!(
        engine
            .read_state
            .residency
            .device_visible_locate_hits
            .load(Ordering::Relaxed)
            > locates_before,
        "the coalesced device visible-locate never fired"
    );
    assert!(
        engine
            .read_state
            .residency
            .lane_tombstone_applies
            .load(Ordering::Relaxed)
            >= tombstones_before + 3,
        "in-place tombstone stamps never fired"
    );
}

/// U1 RECOVERY: mixed insert/delete lane history replays to a row-identical store — the W5b
/// by-key records re-resolve deterministically (winner insert replays before its delete;
/// dead-twin reinsert lands as the visible image), and the reopened engine keeps serving.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_lane_delete_recovery_replays_row_identical() {
    let path = test_wal_path("lane-delete-recovery");
    let prior_durability = std::env::var("GPU_DB_WAL_DURABILITY").ok();
    std::env::set_var("GPU_DB_WAL_DURABILITY", "fua");
    let mut engine = Engine::with_durable_wal_segment(&path);
    match &prior_durability {
        Some(value) => std::env::set_var("GPU_DB_WAL_DURABILITY", value),
        None => std::env::remove_var("GPU_DB_WAL_DURABILITY"),
    }
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    let txn_ids = AtomicU64::new(10);
    engine
        .execute_dml_concurrent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            "INSERT INTO t VALUES (1000000, 0)",
        )
        .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("t")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let mut warmed = false;
    for i in 0..10_000_i32 {
        engine
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &format!("INSERT INTO t VALUES ({}, 0)", 1_000_001 + i),
            )
            .unwrap();
        if engine.table_device_authoritative("t") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "table never entered elision on a GPU box");
    if crate::engine_intent_lanes::intent_lane_count() < 2 {
        return; // lanes-only surface
    }

    let insert_route = engine.prepare_covered_insert_route("t").unwrap();
    let delete_route = engine.prepare_covered_delete_route("t").unwrap();
    for id in 0..100_i32 {
        assert_eq!(
            commit_intent_via_submit(&engine, &txn_ids, &insert_route, &[id, id]).unwrap(),
            1
        );
    }
    // Delete the evens below 40; reinsert two of them with new images; delete a missing key
    // (0-row: must leave NO record — replay parity proves it).
    for id in (0..40_i32).step_by(2) {
        assert_eq!(
            commit_delete_via_submit(&engine, &txn_ids, &delete_route, id).unwrap(),
            1
        );
    }
    assert_eq!(
        commit_delete_via_submit(&engine, &txn_ids, &delete_route, 555_555).unwrap(),
        0
    );
    for id in [2_i32, 4] {
        assert_eq!(
            commit_intent_via_submit(&engine, &txn_ids, &insert_route, &[id, id + 5000]).unwrap(),
            1
        );
    }
    let before = select_rows_unordered_sorted(&engine);
    drop(engine); // crash

    let recovered = Engine::open_durable_wal_segment(&path).unwrap();
    recovered.set_auto_admit_on_commit(true);
    recovered.set_device_write_locate_wave_batch_enabled(true);
    let after = select_rows_unordered_sorted(&recovered);
    assert_eq!(before, after, "replayed store must be row-identical");
    // The recovered engine still deletes through the intent surface (route re-prepare
    // exercises the elision re-entry arm on a recovered lanes engine).
    let txn_ids = AtomicU64::new(1_000_000);
    let delete_route = recovered.prepare_covered_delete_route("t").unwrap();
    assert_eq!(
        commit_delete_via_submit(&recovered, &txn_ids, &delete_route, 1).unwrap(),
        1,
        "post-recovery delete locates and tombstones"
    );
    assert_eq!(
        commit_delete_via_submit(&recovered, &txn_ids, &delete_route, 1).unwrap(),
        0,
        "post-recovery dead key reports zero rows"
    );
}

/// U2: poll one covered-UPDATE intent to completion, driving the pipeline. `new_values` is the
/// FULL new row (all columns); `Ok(n)` = rows affected (0 or 1 by the covered shape).
fn commit_update_via_submit(
    engine: &Engine,
    txn_ids: &AtomicU64,
    route: &crate::CoveredUpdateRoute,
    new_values: &[i32],
) -> Result<u64, ExecuteError> {
    let mut ticket = engine.submit_covered_update_intent(
        txn_ids.fetch_add(1, Ordering::Relaxed),
        route,
        new_values,
    )?;
    let mut spins = 0u64;
    loop {
        engine.drive_commit_wave();
        if let Some(result) = engine.poll_intent(&mut ticket) {
            return result;
        }
        spins += 1;
        assert!(spins < 10_000_000, "update intent never settled");
    }
}

/// U2 END-TO-END: covered lane UPDATE intents — 1-row full-row replace (new image visible, old
/// hidden), 0-row update (missing key: the CONDITIONAL append fires nothing), chained updates,
/// update-then-delete (the delete locates the new version through the dropped-then-rebuilt index),
/// rows-affected reporting, and the FIRED counters for the device visible-locate + in-place
/// tombstone (a silent fallback would pass output parity while abandoning the design).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_lane_update_intents_end_to_end() {
    let path = test_wal_path("lane-update-e2e");
    let prior_durability = std::env::var("GPU_DB_WAL_DURABILITY").ok();
    std::env::set_var("GPU_DB_WAL_DURABILITY", "fua");
    let mut engine = Engine::with_durable_wal_segment(&path);
    match &prior_durability {
        Some(value) => std::env::set_var("GPU_DB_WAL_DURABILITY", value),
        None => std::env::remove_var("GPU_DB_WAL_DURABILITY"),
    }
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    // v1 exclusion probe target: a second unique i32 index makes the UPDATE route refuse.
    engine
        .execute_text(2, "CREATE TABLE t2 (a INT PRIMARY KEY, b INT UNIQUE)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let txn_ids = AtomicU64::new(10);
    engine
        .execute_dml_concurrent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            "INSERT INTO t VALUES (1000000, 0)",
        )
        .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("t")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // driverless box
    }
    let mut warmed = false;
    for i in 0..10_000_i32 {
        engine
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &format!("INSERT INTO t VALUES ({}, 0)", 1_000_001 + i),
            )
            .unwrap();
        if engine.table_device_authoritative("t") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "table never entered elision on a GPU box");

    let insert_route = engine.prepare_covered_insert_route("t").unwrap();
    let mut update_route = engine.prepare_covered_update_route("t").unwrap();
    assert_eq!(update_route.table(), "t");

    let lanes_mode = crate::engine_intent_lanes::intent_lane_count() >= 2;
    if !lanes_mode {
        // Serial arm: update intents are lanes-only — the refusal is loud, not a fallback.
        let err = engine
            .submit_covered_update_intent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &update_route,
                &[1, 1],
            )
            .map(|_| ())
            .expect_err("update intents must refuse without lanes");
        assert!(err.to_string().contains("intent-lanes"), "{err}");
        return;
    }

    // Seed rows 0..200 as (id, id + 7); every insert reports rows_affected == 1.
    for id in 0..200_i32 {
        let rows =
            commit_intent_via_submit(&engine, &txn_ids, &insert_route, &[id, id + 7]).unwrap();
        assert_eq!(rows, 1, "covered insert reports exactly one row");
    }
    let locates_before = engine
        .read_state
        .residency
        .device_visible_locate_hits
        .load(Ordering::Relaxed);
    let tombstones_before = engine
        .read_state
        .residency
        .lane_tombstone_applies
        .load(Ordering::Relaxed);
    let entity_50 = visible_entity_id(&engine, 50);

    // The GPU read-back over an update/delete-versioned elided shard applies deleted_by visibility
    // and keeps relational authority on device. Refresh the covered route for the current
    // generation after each read, exactly as the delete e2e does.

    // 1-ROW UPDATE (full-row replace): key 50 (id=50, v=57) -> (id=50, v=9999).
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[50, 9999]).unwrap(),
        1,
        "updating a visible row affects exactly one row"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "an in-place lane update must not de-elide the table (fallback fired?)"
    );
    assert_eq!(
        visible_entity_id(&engine, 50),
        entity_50,
        "a covered UPDATE must preserve stable entity identity"
    );
    let rows = select_rows_unordered_sorted(&engine); // GPU visibility read-back
    let fifty: Vec<_> = rows
        .iter()
        .filter(|row| row.first() == Some(&SqlValue::Int4(50)))
        .collect();
    assert_eq!(
        fifty.len(),
        1,
        "the updated key is visible EXACTLY once (old twin hidden)"
    );
    assert_eq!(
        fifty[0].get(1),
        Some(&SqlValue::Int4(9999)),
        "new image wins"
    );
    let rows_before_zero = rows.len();

    // 0-ROW UPDATE (missing key): the CONDITIONAL append must fire NOTHING.
    update_route = engine.prepare_covered_update_route("t").unwrap(); // refresh the prepared generation
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[987_654, 1]).unwrap(),
        0,
        "updating a never-existing key affects zero rows"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "0-row updates must not de-elide"
    );
    assert_eq!(
        select_rows_unordered_sorted(&engine).len(),
        rows_before_zero,
        "a 0-row update must not append a phantom row"
    );

    // CHAINED UPDATE: key 50 again -> (50, 8888); the new image wins, still exactly once.
    update_route = engine.prepare_covered_update_route("t").unwrap(); // refresh the prepared generation
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[50, 8888]).unwrap(),
        1,
        "chained update of the same key affects one row"
    );
    assert_eq!(
        visible_entity_id(&engine, 50),
        entity_50,
        "chained covered UPDATEs must retain the original entity identity"
    );
    let rows = select_rows_unordered_sorted(&engine); // GPU visibility read-back
    let fifty: Vec<_> = rows
        .iter()
        .filter(|row| row.first() == Some(&SqlValue::Int4(50)))
        .collect();
    assert_eq!(
        fifty.len(),
        1,
        "chained update leaves the key visible exactly once"
    );
    assert_eq!(
        fifty[0].get(1),
        Some(&SqlValue::Int4(8888)),
        "latest image wins"
    );

    // UPDATE-THEN-DELETE: update key 60, then delete it — the delete must locate the NEW version
    // through the version-aware pk index after the update appended a second physical version.
    update_route = engine.prepare_covered_update_route("t").unwrap(); // refresh the prepared generation
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[60, 4242]).unwrap(),
        1,
        "update the target of the delete"
    );
    let delete_route = engine.prepare_covered_delete_route("t").unwrap();
    assert_eq!(
        commit_delete_via_submit(&engine, &txn_ids, &delete_route, 60).unwrap(),
        1,
        "the updated row is locatable + deletable (index survived the version twin)"
    );
    assert!(
        !select_rows_unordered_sorted(&engine)
            .iter()
            .any(|row| row.first() == Some(&SqlValue::Int4(60))),
        "the updated-then-deleted key must not be visible"
    );

    // v1 exclusion: a second unique i32 index refuses the UPDATE route at prepare.
    let err = engine.prepare_covered_update_route("t2").unwrap_err();
    assert!(
        err.to_string().contains("ONLY unique i32 index")
            || err.to_string().contains("not covered"),
        "{err}"
    );

    // FIRED counters: the device visible-locate ran and tombstones were stamped in place for the
    // update-olds (a silent fallback would pass every assertion above while abandoning the design).
    assert!(
        engine
            .read_state
            .residency
            .device_visible_locate_hits
            .load(Ordering::Relaxed)
            > locates_before,
        "the coalesced device visible-locate never fired for updates"
    );
    assert!(
        engine
            .read_state
            .residency
            .lane_tombstone_applies
            .load(Ordering::Relaxed)
            >= tombstones_before + 3,
        "in-place tombstone stamps never fired for update-olds"
    );
}

/// U2 RECOVERY (THE ALLOCATOR HIGH-WATER GATE): a mixed insert/update/delete history — including a
/// 0-ROW update and CHAINED updates — replays to a row-identical store AND a preserved row-id
/// allocator high-water. Non-vacuity: the 0-row update's `new_row_id` is claimed + WAL-durable but
/// installs nothing, so replay MUST advance the allocator by 1 for it too, or the recovered
/// high-water lands 1 short of the live one — which the `live_hwm == recovered_hwm` assertion below
/// catches (row-value parity alone does NOT, since every version installs at its EXPLICIT id).
/// (Deliberately NOT asserting `next_row_id == record.new_row_id` per record: the pump claims the
/// row-id and seq blocks as SEPARATE lock-free fetch_adds across concurrent lanes, so per-record
/// positional equality is a FALSE invariant — see the replay arm's note in engine_commit.rs.)
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_lane_update_recovery_replays_row_identical() {
    let path = test_wal_path("lane-update-recovery");
    let prior_durability = std::env::var("GPU_DB_WAL_DURABILITY").ok();
    std::env::set_var("GPU_DB_WAL_DURABILITY", "fua");
    let mut engine = Engine::with_durable_wal_segment(&path);
    match &prior_durability {
        Some(value) => std::env::set_var("GPU_DB_WAL_DURABILITY", value),
        None => std::env::remove_var("GPU_DB_WAL_DURABILITY"),
    }
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    let txn_ids = AtomicU64::new(10);
    engine
        .execute_dml_concurrent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            "INSERT INTO t VALUES (1000000, 0)",
        )
        .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("t")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let mut warmed = false;
    for i in 0..10_000_i32 {
        engine
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &format!("INSERT INTO t VALUES ({}, 0)", 1_000_001 + i),
            )
            .unwrap();
        if engine.table_device_authoritative("t") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "table never entered elision on a GPU box");
    if crate::engine_intent_lanes::intent_lane_count() < 2 {
        return; // lanes-only surface
    }

    let insert_route = engine.prepare_covered_insert_route("t").unwrap();
    let mut update_route = engine.prepare_covered_update_route("t").unwrap();
    for id in 0..100_i32 {
        assert_eq!(
            commit_intent_via_submit(&engine, &txn_ids, &insert_route, &[id, id]).unwrap(),
            1
        );
    }
    // Update the evens below 40 to (id, id + 5000) — one row each. Each update's version append
    // versions the shard and can de-elide it (the F3/U4 rebuild-decline cliff), so re-prepare
    // (re-enter elision) after each update for the NEXT one. The WAL records this writes are
    // UNCHANGED by the host-side rehydration, so replay parity is unaffected.
    for id in (0..40_i32).step_by(2) {
        assert_eq!(
            commit_update_via_submit(&engine, &txn_ids, &update_route, &[id, id + 5000]).unwrap(),
            1
        );
        update_route = engine.prepare_covered_update_route("t").unwrap();
    }
    // 0-ROW update of a missing key: a durable record that installs nothing but BURNS the row id.
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[555_555, 1]).unwrap(),
        0
    );
    update_route = engine.prepare_covered_update_route("t").unwrap();
    // CHAINED update AFTER the 0-row burn (its new_row_id only aligns at replay if the 0-row
    // update advanced the allocator): key 2 -> (2, 22222).
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[2, 22222]).unwrap(),
        1
    );
    update_route = engine.prepare_covered_update_route("t").unwrap();
    // UPDATE-THEN-DELETE across the crash boundary: update key 10, then delete it.
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[10, 99999]).unwrap(),
        1
    );
    let delete_route = engine.prepare_covered_delete_route("t").unwrap();
    assert_eq!(
        commit_delete_via_submit(&engine, &txn_ids, &delete_route, 10).unwrap(),
        1
    );
    let before = select_rows_unordered_sorted(&engine);
    // Capture the LIVE allocator high-water: recovery must reconstruct it exactly (every
    // row-consuming record — insert row, 1-row update, AND 0-row update — advances by 1).
    let live_hwm = engine.read_state.mvcc.current_row_id();
    drop(engine); // crash

    let recovered = Engine::open_durable_wal_segment(&path).unwrap();
    recovered.set_auto_admit_on_commit(true);
    recovered.set_device_write_locate_wave_batch_enabled(true);
    let after = select_rows_unordered_sorted(&recovered);
    assert_eq!(before, after, "replayed store must be row-identical");
    // THE 0-ROW-ADVANCE GATE (non-vacuous): the recovered high-water must EQUAL the live one. A
    // dropped 0-row-update advance lands it 1 short, and a post-recovery insert would then reuse a
    // live id. Row-value parity above cannot see this (explicit-id installs); this can.
    assert_eq!(
        recovered.read_state.mvcc.current_row_id(),
        live_hwm,
        "recovery must reconstruct the row-id allocator high-water exactly (0-row updates included)"
    );
    // The recovered engine still updates through the intent surface (route re-prepare exercises
    // the elision re-entry arm on a recovered lanes engine).
    let txn_ids = AtomicU64::new(1_000_000);
    let update_route = recovered.prepare_covered_update_route("t").unwrap();
    assert_eq!(
        commit_update_via_submit(&recovered, &txn_ids, &update_route, &[1, 4321]).unwrap(),
        1,
        "post-recovery update locates and rewrites"
    );
    assert_eq!(
        commit_update_via_submit(&recovered, &txn_ids, &update_route, &[555_555, 7]).unwrap(),
        0,
        "post-recovery dead key reports zero rows"
    );
}

/// F3/U4 (THE VERSION-AWARE-INDEX GATE): many CONSECUTIVE covered updates on distinct keys, with
/// the update route prepared ONCE and NEVER re-prepared, must all succeed and the table must stay
/// ELIDED throughout. Before F3/U4 the physical version-twin append dropped the pk-index cache, the next
/// update's locate rebuild declined on the above-boundary twin, the apply rehydrated + de-elided,
/// and this route DRIFTED (submit returns Err) within a handful of updates. With the dup-tolerant
/// index the twin is placed at the next probe slot, the visible-locate resolves it, and the index
/// stays valid — so the route never drifts and pk-index rebuilds stay BOUNDED (geometric load
/// cadence, not one-per-update). Sabotage: reverting the insert-kernel flip fails this promptly.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_lane_update_sustained_stays_elided() {
    let path = test_wal_path("lane-update-sustained");
    let prior_durability = std::env::var("GPU_DB_WAL_DURABILITY").ok();
    std::env::set_var("GPU_DB_WAL_DURABILITY", "fua");
    let mut engine = Engine::with_durable_wal_segment(&path);
    match &prior_durability {
        Some(value) => std::env::set_var("GPU_DB_WAL_DURABILITY", value),
        None => std::env::remove_var("GPU_DB_WAL_DURABILITY"),
    }
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    let txn_ids = AtomicU64::new(10);
    engine
        .execute_dml_concurrent(
            txn_ids.fetch_add(1, Ordering::Relaxed),
            "INSERT INTO t VALUES (1000000, 0)",
        )
        .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("t")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let mut warmed = false;
    for i in 0..10_000_i32 {
        engine
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &format!("INSERT INTO t VALUES ({}, 0)", 1_000_001 + i),
            )
            .unwrap();
        if engine.table_device_authoritative("t") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "table never entered elision on a GPU box");
    if crate::engine_intent_lanes::intent_lane_count() < 2 {
        return; // lanes-only surface
    }

    let insert_route = engine.prepare_covered_insert_route("t").unwrap();
    for id in 0..200_i32 {
        assert_eq!(
            commit_intent_via_submit(&engine, &txn_ids, &insert_route, &[id, id]).unwrap(),
            1
        );
    }
    // Prepare the update route ONCE — the whole point is that we NEVER re-prepare it.
    let update_route = engine.prepare_covered_update_route("t").unwrap();
    let rebuilds_before = engine.pk_index_rebuilds_diag();
    // 150 consecutive updates on distinct committed keys, no re-prepare, no reads in between.
    for id in 0..150_i32 {
        assert_eq!(
            commit_update_via_submit(&engine, &txn_ids, &update_route, &[id, id + 100_000])
                .unwrap(),
            1,
            "update {id} must apply on the still-elided device path (route must not drift)"
        );
        assert!(
            engine.table_device_authoritative("t"),
            "the dup-tolerant index must keep the table elided across update {id} (no version-twin \
             de-elision)"
        );
    }
    // Rebuilds must stay BOUNDED (geometric load cadence), not ~one-per-update.
    let rebuilds = engine.pk_index_rebuilds_diag() - rebuilds_before;
    assert!(
        rebuilds < 30,
        "pk-index rebuilds must stay bounded under sustained updates, saw {rebuilds}"
    );
}
