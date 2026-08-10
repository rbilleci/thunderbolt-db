/// SV4b (GPU-native incremental DELETE, commit WIRING): a single-row SQL DELETE on a
/// shard-resident table LOCATES + tombstones the row's slot IN PLACE. NON-VACUITY: the deleted_by
/// region EXISTING after the DELETE proves the tombstone route ran. A MULTI-ROW DELETE uses the same
/// identity-checked device maintenance path.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv4b_sql_delete_tombstones_in_place_with_exact_visibility() {
    let load = |e: &Engine| {
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64); // 200 rows -> shards 64,64,64,8
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..200_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!(
                    "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                    i * 10
                ),
            )
            .unwrap();
        }
    };
    let count = |e: &Engine| match e
        .execute_relational_select_text("SELECT COUNT(*) FROM accounts")
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("unexpected COUNT shape {other:?}"),
    };
    let present = |e: &Engine, id: i64| {
        !e.execute_relational_select_text(&format!("SELECT id FROM accounts WHERE id = {id}"))
            .unwrap()
            .rows
            .is_empty()
    };

    let e = Engine::new_local();
    load(&e);
    assert!(
        !table_has_any_deleted_by_cell(&e, "accounts"),
        "delete-free: no region"
    );
    assert_eq!(count(&e), 200);

    e.execute_text(202, "DELETE FROM accounts WHERE id = 130")
        .unwrap();
    // NON-VACUITY: the tombstone path ran (region allocated).
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "single-row DELETE routed through the in-place tombstone (region allocated)"
    );
    assert!(
        !present(&e, 130),
        "id=130 deleted -> hidden on the GPU route"
    );
    assert!(
        present(&e, 129) && present(&e, 131),
        "same-shard neighbors still visible"
    );
    assert!(present(&e, 5), "a row in a different shard untouched");
    assert_eq!(count(&e), 199, "COUNT drops by exactly one");

    // RETIREMENT A4b: a MULTI-ROW DELETE (2 rows) is now INCREMENTAL (per-row exact-1
    // locate+tombstone) — the region stays live with both slots stamped.
    e.execute_text(203, "DELETE FROM accounts WHERE id = 50 OR id = 51")
        .unwrap();
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "multi-row DELETE must stay incremental (region live, A4b)"
    );
    assert!(
        !present(&e, 50) && !present(&e, 51),
        "multi-row DELETE removed both rows"
    );
    assert!(
        !present(&e, 130),
        "the earlier single-row delete stays deleted"
    );
    assert_eq!(count(&e), 197, "COUNT after 3 total deletes");
}

/// R3-003 generation ownership: BEGIN pins the old shard descriptor and boundary. A later UPDATE
/// appends a new physical version to the same allocation; transaction SELECT still executes against
/// its retained boundary while an autocommit SELECT sees new data.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_select_retains_old_gpu_boundary_across_in_place_update() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
        .unwrap();
    e.execute_text(90, "BEGIN ISOLATION LEVEL REPEATABLE READ")
        .unwrap();

    let select = match parse_command("SELECT balance FROM accounts WHERE id = 1").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let first = e
        .execute_relational_select_in_transaction(90, &select)
        .unwrap();
    assert_eq!(first.rows.row(0)[0], SqlValue::Int4(100));

    let captured = e.transaction_snapshot_handle(90).unwrap();
    let old_memory = captured.resident_shards["accounts"][0]
        .device_memory
        .as_ref()
        .unwrap()
        .clone();
    drop(captured);
    e.execute_text(3, "UPDATE accounts SET balance = 200 WHERE id = 1")
        .unwrap();
    let current_memory = e.read_residency_shards()["accounts"][0]
        .device_memory
        .as_ref()
        .unwrap()
        .clone();
    assert!(
        Arc::ptr_eq(&old_memory, &current_memory),
        "normal UPDATE must maintain the authoritative allocation in place"
    );

    let old = e
        .execute_relational_select_in_transaction(90, &select)
        .unwrap();
    assert_eq!(old.rows.row(0)[0], SqlValue::Int4(100));
    let current = e.execute_relational_select(&select).unwrap();
    assert_eq!(current.rows.row(0)[0], SqlValue::Int4(200));
    e.execute_text(90, "ROLLBACK").unwrap();
}

/// R3-003 transaction DML prepare: after a current-generation in-place update, the old transaction's
/// point predicate must resolve from its retained boundary. The residual balance predicate
/// distinguishes the generations; staging succeeds against the retained bytes, then COMMIT rejects
/// the changed stable identity/row image from the current device generation.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_dml_prepare_uses_retained_gpu_generation_for_conflict_verdict() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT, marker INT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance, marker) VALUES (1, 100, NULL)",
    )
    .unwrap();
    e.execute_text(90, "BEGIN ISOLATION LEVEL REPEATABLE READ")
        .unwrap();

    let pin = match parse_command("SELECT balance FROM accounts WHERE id = 1").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let first = e
        .execute_relational_select_in_transaction(90, &pin)
        .unwrap();
    assert_eq!(first.rows.row(0)[0], SqlValue::Int4(100));

    let captured = e.transaction_snapshot_handle(90).unwrap();
    let old_memory = captured.resident_shards["accounts"][0]
        .device_memory
        .as_ref()
        .unwrap()
        .clone();
    e.execute_dml_concurrent(3, "UPDATE accounts SET balance = 200 WHERE id = 1")
        .unwrap();
    let current_memory = e.read_residency_shards()["accounts"][0]
        .device_memory
        .as_ref()
        .unwrap()
        .clone();
    assert!(
        Arc::ptr_eq(&old_memory, &current_memory),
        "normal DML keeps one device allocation and separates versions by stamps"
    );

    let device_hits_before = e.dml_device_resolve_hits();
    e.execute_dml_concurrent(
        90,
        "UPDATE accounts SET balance = 300 WHERE id = 1 AND balance = 100",
    )
    .unwrap();
    assert!(
        e.dml_device_resolve_hits() > device_hits_before,
        "non-vacuity: retained-generation predicate resolution must complete on the GPU"
    );

    let select = match parse_command("SELECT balance, marker FROM accounts WHERE id = 1").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert_eq!(
        e.execute_relational_select_in_transaction(90, &select)
            .unwrap()
            .rows
            .row(0)[0],
        SqlValue::Int4(300)
    );
    assert_eq!(
        e.execute_relational_select_in_transaction(90, &select)
            .unwrap()
            .rows
            .row(0)[1],
        SqlValue::Null
    );
    let wal_before = e.durable_wal_records().len();
    let err = e.execute_text(90, "COMMIT").unwrap_err();
    assert!(
        matches!(&err, ExecuteError::Serialization(message) if message.contains("device write-write conflict")),
        "expected current-generation device write conflict, got {err:?}"
    );
    assert_eq!(e.durable_wal_records().len(), wal_before);
    assert_eq!(
        e.execute_relational_select(&select).unwrap().rows.row(0)[0],
        SqlValue::Int4(200)
    );
    assert_eq!(
        e.execute_relational_select(&select).unwrap().rows.row(0)[1],
        SqlValue::Null
    );
    e.execute_text(90, "ROLLBACK").unwrap();
}

/// R3-003 transaction delta: UPDATE/INSERT/DELETE publish only a private device generation.
/// Transaction SELECT and later DML observe it, autocommit readers do not, and rollback releases it
/// without ever changing the global generation. NULL proves the private payload/bitmap route.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_private_gpu_delta_provides_read_your_writes_and_rollback() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_shard_size_target(64);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT, marker INT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance, marker) VALUES (1, 100, NULL)",
    )
    .unwrap();
    e.execute_text(90, "BEGIN").unwrap();

    e.execute_dml_concurrent(90, "UPDATE accounts SET balance = 200 WHERE id = 1")
        .unwrap();
    e.execute_dml_concurrent(
        90,
        "INSERT INTO accounts (id, balance, marker) VALUES (2, 300, NULL)",
    )
    .unwrap();

    let row = match parse_command("SELECT balance, marker FROM accounts WHERE id = 1").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let inserted = match parse_command("SELECT balance, marker FROM accounts WHERE id = 2").unwrap()
    {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let private = e
        .execute_relational_select_in_transaction(90, &row)
        .unwrap();
    assert_eq!(private.rows.row(0), &[SqlValue::Int4(200), SqlValue::Null]);
    let private_insert = e
        .execute_relational_select_in_transaction(90, &inserted)
        .unwrap();
    assert_eq!(
        private_insert.rows.row(0),
        &[SqlValue::Int4(300), SqlValue::Null]
    );
    assert_eq!(
        e.execute_relational_select(&row).unwrap().rows.row(0),
        &[SqlValue::Int4(100), SqlValue::Null]
    );
    assert!(e
        .execute_relational_select(&inserted)
        .unwrap()
        .rows
        .is_empty());

    e.execute_dml_concurrent(90, "DELETE FROM accounts WHERE id = 1")
        .unwrap();
    assert!(e
        .execute_relational_select_in_transaction(90, &row)
        .unwrap()
        .rows
        .is_empty());
    e.execute_text(90, "ROLLBACK").unwrap();

    assert_eq!(
        e.execute_relational_select(&row).unwrap().rows.row(0),
        &[SqlValue::Int4(100), SqlValue::Null]
    );
    assert!(e
        .execute_relational_select(&inserted)
        .unwrap()
        .rows
        .is_empty());
}

/// Private CUDA allocations reserve the shared residency budget before allocation, release a
/// failed statement immediately, charge the exact current private Arc graph (not every historical
/// COW), and release the retained charge at transaction teardown.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_private_gpu_budget_is_preallocated_exact_and_lifetime_scoped() {
    let mut e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance) VALUES (1, 100), (2, 200)",
    )
    .unwrap();
    let table = e.relational_catalog_table("accounts").unwrap();
    assert_eq!(
        e.locate_resident_pk_via_shard_index_detailed(&table, 0, 1)
            .unwrap()
            .len(),
        1,
        "prime the device identity index before imposing the exact budget"
    );
    let global_bytes = e.relational_resident_bytes_for_gpu(0);
    assert!(global_bytes > 0, "test requires retained device allocation");
    let (expected_first_charge, replacement_cow_bytes) = {
        let table = e.relational_catalog_table("accounts").unwrap();
        let names = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let types = table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect::<Vec<_>>();
        let rows = vec![vec![SqlValue::Int4(1), SqlValue::Int4(101)]];
        let (payload, ..) =
            crate::engine_residency::build_relational_device_payload(&names, &types, &rows)
                .unwrap();
        let tombstone_bytes = e
            .read_residency_shards()
            .get("accounts")
            .unwrap()
            .iter()
            .map(|shard| shard.capacity as u64 * 8)
            .max()
            .unwrap();
        (
            tombstone_bytes + payload.len() as u64 + 8, // one private stable-identity cell
            tombstone_bytes,
        )
    };
    e.set_relational_residency_budget_bytes(0, global_bytes);
    e.execute_text(90, "BEGIN").unwrap();
    let error = e
        .execute_dml_concurrent(90, "UPDATE accounts SET balance = 101 WHERE id = 1")
        .unwrap_err();
    assert!(error.to_string().contains("exceeds residency budget"));
    assert!(
        e.transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty(),
        "a pre-allocation decline must leave no private charge"
    );

    e.set_relational_residency_budget_bytes(0, global_bytes + expected_first_charge);
    e.execute_dml_concurrent(90, "UPDATE accounts SET balance = 101 WHERE id = 1")
        .unwrap();
    let snapshot = e.transaction_snapshot_handle(90).unwrap();
    let first_charge = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .private_gpu_bytes_by_gpu[&0];
    assert_eq!(first_charge, expected_first_charge);
    assert_eq!(
        e.transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())[&0],
        first_charge
    );

    // Both statements COW the same base-shard tombstone sidecar. The second retains a replacement
    // sidecar of the same size plus the already-retained UPDATE payload; it must not accumulate the
    // now-unreachable first sidecar charge. Its preflight must nevertheless admit the temporary
    // old+new COW peak until the replacement graph is published.
    let _replacement_cow_bytes = replacement_cow_bytes;
    e.set_relational_residency_budget_bytes(0, u64::MAX);
    e.execute_dml_concurrent(90, "DELETE FROM accounts WHERE id = 2")
        .unwrap();
    let second_charge = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .private_gpu_bytes_by_gpu[&0];
    assert_eq!(second_charge, first_charge);
    drop(snapshot);
    e.execute_text(90, "ROLLBACK").unwrap();
    assert!(
        e.transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty(),
        "terminal transaction release must drop the final private Arc charge"
    );
}

/// A transaction SELECT owns its private generation for the whole statement. A later statement
/// in the same transaction cannot replace that generation (and retire its exact GPU charge) until
/// the reader releases it, so global admission never observes live private VRAM as uncharged.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_private_reader_pin_serializes_replacement_and_budget_accounting() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(1, "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)")
        .unwrap();
    engine
        .execute_text(
            2,
            "INSERT INTO accounts (id, balance) VALUES (1, 100), (2, 200)",
        )
        .unwrap();
    // Admission limits are covered by the preceding exact-budget test. This race isolates charge
    // lifetime/serialization and must allow the temporary old+new replacement peak.
    engine.set_relational_residency_budget_bytes(0, u64::MAX);
    engine.execute_text(90, "BEGIN").unwrap();
    engine
        .execute_dml_concurrent(90, "UPDATE accounts SET balance = 101 WHERE id = 1")
        .unwrap();
    let snapshot = engine.transaction_snapshot_handle(90).unwrap();
    let first_charge = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .private_gpu_bytes_by_gpu[&0];
    drop(snapshot);

    let Command::Select(select) =
        parse_command("SELECT id, balance FROM accounts ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let engine = Arc::new(engine);
    let reached = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    let reader = {
        let engine = Arc::clone(&engine);
        let reached = Arc::clone(&reached);
        let resume = Arc::clone(&resume);
        std::thread::spawn(move || {
            engine.execute_relational_select_in_transaction_instrumented(90, &select, || {
                reached.wait();
                resume.wait();
            })
        })
    };
    reached.wait();

    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let replacement = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            let result = engine.execute_dml_concurrent(90, "DELETE FROM accounts WHERE id = 2");
            done_tx.send(()).unwrap();
            result
        })
    };
    assert!(
        done_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err(),
        "same-transaction DML replaced a private generation while its reader was pinned"
    );
    assert_eq!(
        engine
            .transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())[&0],
        first_charge,
        "the pinned reader's exact private GPU charge must remain published"
    );

    resume.wait();
    assert_eq!(reader.join().unwrap().unwrap().rows.len(), 2);
    replacement.join().unwrap().unwrap();
    let snapshot = engine.transaction_snapshot_handle(90).unwrap();
    let replacement_charge = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .private_gpu_bytes_by_gpu[&0];
    assert_eq!(replacement_charge, first_charge);
    drop(snapshot);
    engine.execute_text(90, "ROLLBACK").unwrap();
    assert!(engine
        .transaction_private_gpu_bytes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_empty());
}

/// The legacy batching entry is transaction-aware: active DML stages into the same private GPU
/// generation as the canonical path, and terminal control publishes that delta rather than
/// discarding it through the empty transaction-context helper.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn enqueue_active_transaction_routes_private_dml_and_commit_atomically() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(1, "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO accounts (id, balance) VALUES (0, 0)")
        .unwrap();
    let residency = engine
        .populate_relational_residency_snapshot("accounts")
        .expect("admit empty transaction base generation");
    if residency.device_memory_proof.is_none() {
        return;
    }
    let now = Instant::now();
    engine.enqueue_set_text(90, "BEGIN", now).unwrap();
    engine
        .enqueue_set_text(
            90,
            "INSERT INTO accounts (id, balance) VALUES (1, 100)",
            now,
        )
        .unwrap();
    let Command::Select(select) =
        parse_command("SELECT id, balance FROM accounts WHERE id = 1").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    assert!(engine
        .execute_relational_select(&select)
        .unwrap()
        .rows
        .is_empty());
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(90, &select)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(100)]]
    );
    engine.enqueue_set_text(90, "COMMIT", now).unwrap();
    assert_eq!(
        engine.execute_relational_select(&select).unwrap().rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(100)]]
    );
    assert!(engine.transaction_snapshot_handle(90).is_none());
}

/// Pending GPU submissions retain their originating service token through both Engine completion
/// forms. A wedge published after the device drain but before return must suppress the result.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn pending_retained_completions_recheck_origin_wedge_after_device_drain() {
    fn pending_submission() -> (Arc<Engine>, RelationalRetainedReadSubmission) {
        let mut engine = Engine::new_local();
        engine.set_auto_admit_on_commit(true);
        engine
            .execute_text(1, "CREATE TABLE events (id INT PRIMARY KEY, value INT)")
            .unwrap();
        engine
            .execute_text(2, "INSERT INTO events (id, value) VALUES (1, 10), (2, 20)")
            .unwrap();
        install_test_single_buffer_residency(&mut engine, "events");
        let engine = Arc::new(engine);
        let Command::Select(select) = parse_command("SELECT id FROM events WHERE id = 1").unwrap()
        else {
            panic!("expected SELECT plan");
        };
        let job = engine
            .prepare_relational_retained_read_job(&select)
            .unwrap();
        let submission = engine
            .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[job])
            .unwrap();
        assert!(
            submission.is_pending(),
            "test requires a deferred GPU result"
        );
        (engine, submission)
    }

    let (engine, submission) = pending_submission();
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

    let (engine, submission) = pending_submission();
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

/// PRODUCT-001: the deferred submission, rather than only its submit call, owns the source-table
/// guard through GPU completion or caller cancellation/drop.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn pending_retained_submission_blocks_reset_until_completion_or_drop() {
    fn pending_submission() -> (Arc<Engine>, u32, RelationalRetainedReadSubmission) {
        let mut engine = Engine::new_local();
        engine.set_auto_admit_on_commit(true);
        engine
            .execute_text(1, "CREATE TABLE retained_guard (id INT PRIMARY KEY, value INT)")
            .unwrap();
        engine
            .execute_text(
                2,
                "INSERT INTO retained_guard (id, value) VALUES (1, 10), (2, 20)",
            )
            .unwrap();
        install_test_single_buffer_residency(&mut engine, "retained_guard");
        let oid = engine.relational_catalog_table("retained_guard").unwrap().oid;
        let engine = Arc::new(engine);
        let Command::Select(select) =
            parse_command("SELECT id FROM retained_guard WHERE id = 1").unwrap()
        else {
            panic!("expected SELECT plan");
        };
        let job = engine
            .prepare_relational_retained_read_job(&select)
            .unwrap();
        let submission = engine
            .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[job])
            .unwrap();
        assert!(submission.is_pending(), "fixture must own deferred GPU work");
        (engine, oid, submission)
    }

    let (engine, oid, submission) = pending_submission();
    let reset = engine.table_access.lease();
    assert!(matches!(
        reset.acquire_exclusive([oid]),
        Err(ExecuteError::Serialization(_))
    ));
    let results = engine
        .complete_relational_retained_read_submission(submission)
        .unwrap();
    assert_eq!(results[0].rows, vec![vec![SqlValue::Int4(1)]]);
    reset.acquire_exclusive([oid]).unwrap();

    let (engine, oid, submission) = pending_submission();
    let reset = engine.table_access.lease();
    assert!(matches!(
        reset.acquire_exclusive([oid]),
        Err(ExecuteError::Serialization(_))
    ));
    drop(submission);
    reset.acquire_exclusive([oid]).unwrap();
}

/// R3-003 atomic publication: several private GPU statements (including an update of a
/// transaction-local insert, a delete, and NULL payloads) become visible at one commit index and
/// one resolved WAL record. Recovery must reproduce the same final identity state.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_private_gpu_delta_commits_one_atomic_recoverable_record() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_shard_size_target(64);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT, marker INT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance, marker) VALUES (1, 100, NULL), (3, 400, 7)",
    )
    .unwrap();
    let base_memory = e.read_residency_shards()["accounts"][0]
        .device_memory
        .as_ref()
        .unwrap()
        .clone();
    e.execute_text(90, "BEGIN").unwrap();
    let wal_before = e.durable_wal_records().len();

    e.execute_dml_concurrent(90, "UPDATE accounts SET balance = 200 WHERE id = 1")
        .unwrap();
    e.execute_dml_concurrent(
        90,
        "INSERT INTO accounts (id, balance, marker) VALUES (2, 300, NULL)",
    )
    .unwrap();
    e.execute_dml_concurrent(90, "UPDATE accounts SET balance = 350 WHERE id = 2")
        .unwrap();
    e.execute_dml_concurrent(90, "DELETE FROM accounts WHERE id = 3")
        .unwrap();

    let select =
        match parse_command("SELECT id, balance, marker FROM accounts ORDER BY id").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
    assert_eq!(
        e.execute_relational_select(&select).unwrap().rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(100), SqlValue::Null],
            vec![SqlValue::Int4(3), SqlValue::Int4(400), SqlValue::Int4(7)],
        ],
        "autocommit visibility remains unchanged before COMMIT"
    );
    assert_eq!(
        e.execute_relational_select_in_transaction(90, &select)
            .unwrap()
            .rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(200), SqlValue::Null],
            vec![SqlValue::Int4(2), SqlValue::Int4(350), SqlValue::Null],
        ],
        "transaction generation includes all prior private statements"
    );

    let before_commit_seq = e.visible_up_to();
    e.execute_text(90, "COMMIT").unwrap();
    assert!(e.transaction_snapshot_handle(90).is_none());
    assert_eq!(e.visible_up_to(), before_commit_seq + 1);
    assert!(
        e.read_residency_shards()["accounts"]
            .iter()
            .filter_map(|shard| shard.device_memory.as_ref())
            .any(|memory| Arc::ptr_eq(memory, &base_memory)),
        "non-vacuity: atomic COMMIT retained the base GPU allocation instead of invalidating and re-admitting the table"
    );
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "non-vacuity: update/delete commit stamps remained on the resident generation"
    );
    let durable = e.durable_wal_records();
    assert_eq!(durable.len(), wal_before + 1);
    assert!(matches!(
        decode_binary_record(&durable.last().unwrap().payload).unwrap(),
        BinaryWalRecord::Transaction(record) if record.mutations.len() == 3
    ));
    let expected = vec![
        vec![SqlValue::Int4(1), SqlValue::Int4(200), SqlValue::Null],
        vec![SqlValue::Int4(2), SqlValue::Int4(350), SqlValue::Null],
    ];
    assert_eq!(e.execute_relational_select(&select).unwrap().rows, expected);

    let recovered = Engine::recover_from_durable_wal(&durable).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        expected,
        "replay of the one resolved transaction record is byte-equivalent"
    );
}

/// A failure after the one transaction WAL record is durable is not an ordinary abort: the engine
/// wedges, retains the transaction context, and refuses rollback/service until restart replay.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_post_durable_apply_failure_is_sticky_fail_stop() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
        .unwrap();
    e.execute_text(90, "BEGIN").unwrap();
    e.execute_dml_concurrent(90, "UPDATE accounts SET balance = 200 WHERE id = 1")
        .unwrap();

    e.fail_next_transaction_post_durable_apply();
    let err = e.execute_text(90, "COMMIT").unwrap_err();
    assert!(
        err.to_string().contains("restart recovery required"),
        "{err}"
    );
    assert!(e.is_commit_path_poisoned());
    assert!(e.transaction_snapshot_handle(90).is_some());
    let rollback = e.execute_text(90, "ROLLBACK").unwrap_err();
    assert!(
        rollback.to_string().contains("restart recovery required"),
        "{rollback}"
    );
    let select = match parse_command("SELECT id, balance FROM accounts ORDER BY id").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let read = e.execute_relational_select(&select).unwrap_err();
    assert!(
        read.to_string().contains("restart recovery required"),
        "direct engine reads must fail-stop too: {read}"
    );
    let later = e
        .execute_dml_concurrent(91, "INSERT INTO accounts (id, balance) VALUES (2, 300)")
        .unwrap_err();
    assert!(
        later.to_string().contains("restart recovery required"),
        "direct concurrent writes must fail-stop too: {later}"
    );

    let durable = e.durable_wal_records();
    let recovered = Engine::recover_from_durable_wal(&durable).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(200)]],
        "restart applies the durable-but-unacknowledged transaction exactly once"
    );
}

/// PRODUCT-001 ordinary published-sequence defaults: values publish independently before each
/// private row statement, survive user rollback, and are referenced exactly by the later atomic
/// user envelope. Transaction-private CREATE/RESTART coverage remains in the lifecycle suite.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_published_sequence_defaults_advance_independently_and_recover_atomically() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE SEQUENCE txn_ids").unwrap();
    e.execute_text(
        2,
        "CREATE TABLE serial_accounts (id INT DEFAULT nextval('txn_ids'::regclass), balance INT)",
    )
    .unwrap();
    // Seed a resident generation without consuming the sequence; first private nextval remains 1.
    e.execute_text(
        3,
        "INSERT INTO serial_accounts (id, balance) VALUES (99, 990)",
    )
    .unwrap();
    e.execute_text(90, "BEGIN").unwrap();
    let wal_before = e.durable_wal_records().len();
    e.execute_dml_concurrent(
        90,
        "INSERT INTO serial_accounts (balance) VALUES (10), (20)",
    )
    .unwrap();
    e.execute_dml_concurrent(90, "INSERT INTO serial_accounts (balance) VALUES (30)")
        .unwrap();

    let select = match parse_command("SELECT id, balance FROM serial_accounts ORDER BY id").unwrap()
    {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert_eq!(
        e.execute_relational_select(&select).unwrap().rows,
        vec![vec![SqlValue::Int4(99), SqlValue::Int4(990)]],
        "private nextval rows remain globally invisible"
    );
    assert_eq!(
        e.execute_relational_select_in_transaction(90, &select)
            .unwrap()
            .rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(10)],
            vec![SqlValue::Int4(2), SqlValue::Int4(20)],
            vec![SqlValue::Int4(3), SqlValue::Int4(30)],
            vec![SqlValue::Int4(99), SqlValue::Int4(990)],
        ]
    );
    let sequence = e.relational_catalog_sequence("txn_ids").unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (3, true));
    assert_eq!(
        e.durable_wal_records().len(),
        wal_before + 3,
        "each ordinary default publishes before its private row"
    );

    e.execute_text(90, "COMMIT").unwrap();
    let sequence = e.relational_catalog_sequence("txn_ids").unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (3, true));
    let durable = e.durable_wal_records();
    assert_eq!(durable.len(), wal_before + 4);
    assert!(matches!(
        decode_binary_record(&durable.last().unwrap().payload).unwrap(),
        BinaryWalRecord::Transaction(record)
            if !record.sequence_advances.contains_key("txn_ids")
                && record.sequence_value_references.len() == 3
                && record.mutations.len() == 3
    ));

    let recovered = Engine::recover_from_durable_wal(&durable).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        e.execute_relational_select(&select).unwrap().rows
    );
    let sequence = recovered.relational_catalog_sequence("txn_ids").unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (3, true));

    // Caller-assigned compatibility ids leave space for engine-owned transition identities.
    e.execute_text(100, "BEGIN").unwrap();
    e.execute_dml_concurrent(100, "INSERT INTO serial_accounts (balance) VALUES (40)")
        .unwrap();
    e.execute_text(100, "ROLLBACK").unwrap();
    let sequence = e.relational_catalog_sequence("txn_ids").unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (4, true));
    e.execute_text(110, "INSERT INTO serial_accounts (balance) VALUES (40)")
        .unwrap();
    let id_five = match parse_command("SELECT balance FROM serial_accounts WHERE id = 5").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert_eq!(
        e.execute_relational_select(&id_five).unwrap().rows.row(0)[0],
        SqlValue::Int4(40),
        "rollback must not rewind the independently published sequence value"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_atomic_commit_rechecks_conflicts_after_last_staged_statement() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT, marker INT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance, marker) VALUES (1, 100, NULL)",
    )
    .unwrap();
    e.execute_text(90, "BEGIN").unwrap();
    e.execute_dml_concurrent(90, "UPDATE accounts SET balance = 200 WHERE id = 1")
        .unwrap();
    let wal_before_winner = e.durable_wal_records().len();
    e.execute_dml_concurrent(3, "UPDATE accounts SET balance = 100 WHERE id = 1")
        .unwrap();
    let wal_after_winner = e.durable_wal_records().len();
    assert_eq!(wal_after_winner, wal_before_winner + 1);
    let current_table = e.relational_catalog_table("accounts").unwrap();
    for shard in e.read_residency_shards()["accounts"].iter() {
        let memory = shard.device_memory.as_ref().unwrap();
        assert!(
            e.shard_write_locate_cell_live("accounts", shard.shard_id, memory),
            "shard {} descriptor and write cell must name the same allocation",
            shard.shard_id
        );
        assert!(
            memory
                .row_range_indices_u32(shard.row_count as u32, 0, shard.row_count as u32)
                .is_ok(),
            "shard {} must support the predicate-free device slot range",
            shard.shard_id
        );
    }
    assert!(
        e.locate_resident_all_slots_detailed(&current_table)
            .is_some(),
        "the current device generation must support a physical identity scan before COMMIT"
    );
    let staged = match parse_command("SELECT balance FROM accounts WHERE id = 1").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert_eq!(
        e.execute_relational_select_in_transaction(90, &staged)
            .unwrap()
            .rows
            .row(0),
        &[SqlValue::Int4(200)],
        "READ COMMITTED statement refresh must rebase and retain the transaction's own write"
    );
    let err = e.execute_text(90, "COMMIT").unwrap_err();
    assert!(
        matches!(&err, ExecuteError::Serialization(message) if message.contains("write-write conflict")),
        "losing transaction must abort before WAL append: {err:?}"
    );
    assert!(
        matches!(&err, ExecuteError::Serialization(message) if message.contains("device write-write conflict")),
        "non-vacuity: COMMIT must return the device-side identity/version verdict: {err:?}"
    );
    assert_eq!(e.durable_wal_records().len(), wal_after_winner);
    assert!(e.transaction_snapshot_handle(90).is_some());
    let select = match parse_command("SELECT balance, marker FROM accounts WHERE id = 1").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert_eq!(
        e.execute_relational_select(&select).unwrap().rows.row(0),
        &[SqlValue::Int4(100), SqlValue::Null]
    );
    assert_eq!(
        e.execute_relational_select_in_transaction(90, &select)
            .unwrap()
            .rows
            .row(0),
        &[SqlValue::Int4(200), SqlValue::Null]
    );
    e.execute_text(90, "ROLLBACK").unwrap();
    assert!(
        !table_has_any_created_by_cell(&e, "accounts"),
        "terminal transaction release must GC a conflict sidecar whose high-water is now globally visible"
    );
}

/// R3-003 cross-table stamps: FK preparation consumes the private device generations, while
/// COMMIT rechecks untouched providers/children against the current device generations.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_foreign_keys_use_private_and_current_device_generations() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE parents (id INT PRIMARY KEY)")
        .unwrap();
    e.execute_text(
        2,
        "CREATE TABLE children (id INT PRIMARY KEY, parent_id INT)",
    )
    .unwrap();
    e.execute_text(
        3,
        "ALTER TABLE ONLY children ADD CONSTRAINT children_parent_fk FOREIGN KEY (parent_id) REFERENCES parents(id)",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO parents (id) VALUES (1), (2)")
        .unwrap();
    e.execute_text(5, "INSERT INTO children (id, parent_id) VALUES (10, 1)")
        .unwrap();

    e.execute_text(90, "BEGIN").unwrap();
    e.execute_dml_concurrent(90, "INSERT INTO parents (id) VALUES (3)")
        .unwrap();
    e.execute_dml_concurrent(90, "INSERT INTO children (id, parent_id) VALUES (30, 3)")
        .unwrap();
    e.execute_dml_concurrent(90, "DELETE FROM children WHERE id = 10")
        .unwrap();
    e.execute_dml_concurrent(90, "DELETE FROM parents WHERE id = 1")
        .unwrap();
    e.execute_text(90, "COMMIT").unwrap();
    let joined = match parse_command("SELECT id, parent_id FROM children ORDER BY id").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert_eq!(
        e.execute_relational_select(&joined).unwrap().rows,
        vec![vec![SqlValue::Int4(30), SqlValue::Int4(3)]]
    );

    // Provider ABA after the child statement: READ COMMITTED advances on a later statement and
    // current state once again contains the provider. The exact FK dependency floor must still
    // reject the delete+reinsert history before another WAL record is appended.
    e.execute_text(91, "BEGIN").unwrap();
    e.execute_dml_concurrent(91, "INSERT INTO children (id, parent_id) VALUES (40, 2)")
        .unwrap();
    e.execute_dml_concurrent(10, "DELETE FROM parents WHERE id = 2")
        .unwrap();
    e.execute_dml_concurrent(13, "INSERT INTO parents (id) VALUES (2)")
        .unwrap();
    let refresh = match parse_command("SELECT id FROM children WHERE id = 40").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert_eq!(
        e.execute_relational_select_in_transaction(91, &refresh)
            .unwrap()
            .rows
            .row(0)[0],
        SqlValue::Int4(40)
    );
    let wal_after_provider_delete = e.durable_wal_records().len();
    let outbound = e.execute_text(91, "COMMIT").unwrap_err();
    assert!(
        matches!(&outbound, ExecuteError::Serialization(message) if message.contains("foreign-key dependency relation \"parents\" changed")),
        "expected exact-floor provider-history conflict, got {outbound:?}"
    );
    assert_eq!(e.durable_wal_records().len(), wal_after_provider_delete);
    e.execute_text(91, "ROLLBACK").unwrap();

    // Child ABA after the parent-delete statement: current state is empty again, so only the
    // retained dependency floor can prove the intervening insert+delete race.
    e.execute_dml_concurrent(11, "INSERT INTO parents (id) VALUES (4)")
        .unwrap();
    e.execute_text(92, "BEGIN").unwrap();
    e.execute_dml_concurrent(92, "DELETE FROM parents WHERE id = 4")
        .unwrap();
    e.execute_dml_concurrent(12, "INSERT INTO children (id, parent_id) VALUES (50, 4)")
        .unwrap();
    e.execute_dml_concurrent(14, "DELETE FROM children WHERE id = 50")
        .unwrap();
    let refresh = match parse_command("SELECT id FROM parents WHERE id = 4").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert!(
        e.execute_relational_select_in_transaction(92, &refresh)
            .unwrap()
            .rows
            .is_empty(),
        "the transaction must continue to read its private parent delete after RC rebase"
    );
    let wal_after_child_insert = e.durable_wal_records().len();
    let inbound = e.execute_text(92, "COMMIT").unwrap_err();
    assert!(
        matches!(&inbound, ExecuteError::Serialization(message) if message.contains("foreign-key dependency relation \"children\" changed")),
        "expected exact-floor child-history conflict, got {inbound:?}"
    );
    assert_eq!(e.durable_wal_records().len(), wal_after_child_insert);
    e.execute_text(92, "ROLLBACK").unwrap();
}

/// The classic sequencer has applied the parent DELETE but has not yet handed its durability tail
/// to the deque. An explicit child transaction must treat that interval as unsettled: it waits for
/// the registered applied tail, then rejects against the published parent deletion.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_fk_commit_waits_across_classic_wave_tail_handoff() {
    let mut e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE parents (id INT PRIMARY KEY)")
        .unwrap();
    e.execute_text(
        2,
        "CREATE TABLE children (id INT PRIMARY KEY, parent_id INT)",
    )
    .unwrap();
    e.execute_text(
        3,
        "ALTER TABLE ONLY children ADD CONSTRAINT children_parent_fk FOREIGN KEY (parent_id) REFERENCES parents(id)",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO parents (id) VALUES (1), (2)")
        .unwrap();
    e.execute_text(5, "INSERT INTO children (id, parent_id) VALUES (0, 2)")
        .unwrap();
    let parents_residency = e
        .populate_relational_residency_snapshot("parents")
        .expect("refresh parent transaction base generation");
    let children_residency = e
        .populate_relational_residency_snapshot("children")
        .expect("admit child transaction base generation");
    if parents_residency.device_memory_proof.is_none()
        || children_residency.device_memory_proof.is_none()
    {
        return;
    }
    let e = Arc::new(e);
    e.execute_text(90, "BEGIN").unwrap();
    e.execute_dml_concurrent(90, "INSERT INTO children (id, parent_id) VALUES (10, 1)")
        .unwrap();

    let reached = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    e.set_wave_tail_handoff_hook(Arc::clone(&reached), Arc::clone(&resume));
    let delete = {
        let e = Arc::clone(&e);
        std::thread::spawn(move || {
            e.execute_dml_concurrent(100, "DELETE FROM parents WHERE id = 1")
        })
    };
    reached.wait();

    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let commit = {
        let e = Arc::clone(&e);
        std::thread::spawn(move || {
            let result = e.execute_text(90, "COMMIT");
            done_tx.send(()).unwrap();
            result
        })
    };
    assert!(
        done_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err(),
        "explicit COMMIT crossed an applied classic wave before its tail was handed off"
    );
    resume.wait();
    delete.join().unwrap().unwrap();
    let result = commit.join().unwrap();
    assert!(
        matches!(&result, Err(ExecuteError::Serialization(message))
            if message.contains("foreign-key dependency relation \"parents\" changed")),
        "the settled parent delete must reject the child commit: {result:?}"
    );
    e.execute_text(90, "ROLLBACK").unwrap();
}

/// A classic tail can fail after an explicit COMMIT's optimistic poison check. The full-drain
/// result and the under-lock sticky recheck are both load-bearing: the transaction must stop before
/// claiming final identities or appending its WAL record once that concurrent tail wedges service.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_commit_fails_closed_when_waited_classic_tail_wedges() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(1, "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
        .unwrap();
    engine.execute_text(90, "BEGIN").unwrap();
    engine
        .execute_dml_concurrent(90, "UPDATE accounts SET balance = 101 WHERE id = 1")
        .unwrap();
    let durable_before = engine.durable_wal_records().len();
    let visible_before = engine.committed_seq();
    engine.simulate_next_wal_flush_failure();

    let engine = Arc::new(engine);
    let reached = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    let failure_published = Arc::new(std::sync::Barrier::new(2));
    let finish_failure = Arc::new(std::sync::Barrier::new(2));
    engine.set_wave_tail_handoff_hook(Arc::clone(&reached), Arc::clone(&resume));
    engine.set_wave_tail_failure_publish_hook(
        Arc::clone(&failure_published),
        Arc::clone(&finish_failure),
    );
    let classic = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                engine.execute_dml_concurrent(
                    100,
                    "INSERT INTO accounts (id, balance) VALUES (2, 200)",
                )
            }))
        })
    };
    reached.wait();

    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let transaction = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            let result = engine.execute_text(90, "COMMIT");
            done_tx.send(()).unwrap();
            result
        })
    };
    assert!(
        done_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err(),
        "explicit COMMIT crossed the classic tail before its durability verdict"
    );
    resume.wait();
    failure_published.wait();
    assert!(
        done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .is_ok(),
        "explicit COMMIT did not fail after the wedge was published and before the tail advertised finished"
    );
    finish_failure.wait();
    assert!(
        classic.join().unwrap().is_err(),
        "the injected classic-tail fsync failure must take the wedge path"
    );
    let error = transaction.join().unwrap().unwrap_err();
    assert!(error.to_string().contains("restart recovery"), "{error}");
    assert!(engine.is_commit_path_poisoned());
    assert_eq!(engine.committed_seq(), visible_before);
    assert_eq!(
        engine.durable_wal_records().len(),
        durable_before,
        "neither the failed classic wave nor the explicit transaction may extend the durable prefix"
    );
    assert!(
        engine.transaction_snapshot_handle(90).is_some(),
        "a pre-WAL fail-stop must leave the explicit transaction uncommitted"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn repeatable_read_first_data_statement_is_lazy_and_uses_current_generation() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    e.execute_text(90, "BEGIN ISOLATION LEVEL REPEATABLE READ")
        .unwrap();
    e.execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1, 200)")
        .unwrap();

    let select = match parse_command("SELECT balance FROM accounts WHERE id = 1").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let first = e
        .execute_relational_select_in_transaction(90, &select)
        .unwrap();
    assert_eq!(first.rows.row(0)[0], SqlValue::Int4(200));
    let current = e.execute_relational_select(&select).unwrap();
    assert_eq!(current.rows.row(0)[0], SqlValue::Int4(200));
    e.execute_text(90, "ROLLBACK").unwrap();
}

/// SV5 (GPU-native incremental UPDATE, commit WIRING): a
/// single-row SQL UPDATE on a shard-resident table TOMBSTONES the old version's slot + APPENDS the new
/// image IN PLACE. NON-VACUITY: the deleted_by region EXISTING after the UPDATE proves the
/// tombstone-old route ran; the read returns the NEW value; COUNT is unchanged (old hidden + new
/// visible); the OLD value is hidden; and a MULTI-ROW UPDATE uses the same device maintenance path.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv5_sql_update_tombstones_old_appends_new_with_exact_visibility() {
    let load = |e: &Engine| {
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..200_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!(
                    "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                    i * 10
                ),
            )
            .unwrap();
        }
    };
    let count = |e: &Engine| match e
        .execute_relational_select_text("SELECT COUNT(*) FROM accounts")
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("unexpected COUNT shape {other:?}"),
    };
    let balance_of = |e: &Engine, id: i64| -> Option<i32> {
        let rows = e
            .execute_relational_select_text(&format!(
                "SELECT id, balance FROM accounts WHERE id = {id}"
            ))
            .unwrap()
            .rows;
        if rows.is_empty() {
            return None;
        }
        match &rows.row(0)[1] {
            SqlValue::Int4(b) => Some(*b),
            other => panic!("unexpected row shape {other:?}"),
        }
    };
    let old_balance_visible = |e: &Engine| {
        // The OLD (id=130, balance=1300) image must be HIDDEN: a lookup by the old balance finds nothing.
        !e.execute_relational_select_text("SELECT id FROM accounts WHERE balance = 1300")
            .unwrap()
            .rows
            .is_empty()
    };

    let e = Engine::new_local();
    load(&e);
    assert!(
        !table_has_any_deleted_by_cell(&e, "accounts"),
        "no region pre-update"
    );
    assert_eq!(count(&e), 200);
    assert_eq!(balance_of(&e, 130), Some(1300), "pre-update balance");

    e.execute_text(202, "UPDATE accounts SET balance = 9999 WHERE id = 130")
        .unwrap();
    // NON-VACUITY: the tombstone-old path ran (region allocated).
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "single-row UPDATE routed through tombstone-old + append-new (region allocated)"
    );
    assert_eq!(
        balance_of(&e, 130),
        Some(9999),
        "id=130 reads the NEW balance (appended version)"
    );
    assert!(
        !old_balance_visible(&e),
        "the OLD (id=130,balance=1300) image is hidden"
    );
    assert_eq!(
        balance_of(&e, 131),
        Some(1310),
        "same-shard neighbor untouched"
    );
    assert_eq!(
        balance_of(&e, 5),
        Some(50),
        "a row in a different shard untouched"
    );
    assert_eq!(count(&e), 200, "COUNT unchanged (old hidden + new visible)");

    // An int4-UNCHANGED update (same-value: id=5 already has balance 5*10=50) still routes: tombstone-OLD
    // FIRST locates the old slot on the buffer BEFORE the identical-int4 new row is appended (count 1), so
    // it tombstones the OLD slot, not the new. Exercises the order-sensitivity the value-changing case can't.
    e.execute_text(203, "UPDATE accounts SET balance = 50 WHERE id = 5")
        .unwrap();
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "same-value UPDATE still routes through tombstone-old + append-new"
    );
    assert_eq!(
        balance_of(&e, 5),
        Some(50),
        "id=5 still reads 50 (old hidden, new appended, same value)"
    );
    assert_eq!(
        count(&e),
        200,
        "COUNT unchanged after the int4-unchanged update"
    );

    // RETIREMENT A4b: a MULTI-ROW UPDATE (2 rows) is now INCREMENTAL (tombstones + one
    // batched identity-stamped append) — the region stays LIVE, no re-admit.
    e.execute_text(
        204,
        "UPDATE accounts SET balance = 0 WHERE id = 10 OR id = 11",
    )
    .unwrap();
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "multi-row UPDATE must stay incremental (region live, A4b)"
    );
    assert_eq!(balance_of(&e, 10), Some(0));
    assert_eq!(balance_of(&e, 11), Some(0));
    assert_eq!(
        balance_of(&e, 130),
        Some(9999),
        "single-row update persists across later device maintenance"
    );
    assert_eq!(count(&e), 200);
}

/// SV6 (`created_by` SI flip-gate) — the DOUBLE-READ differential, deterministic torn-window form.
/// The SV5 incremental UPDATE appends the new version + bumps `row_count` BEFORE the publication join,
/// and a lock-free reader binds `read_txn_id = committed_seq()` THEN loads shards — so a reader that
/// observes `committed_seq = C-1` while the shards ALREADY carry the appended row is the torn window the
/// SV5 audit flagged (P2). This test constructs that window EXACTLY: it applies the incremental UPDATE at
/// `commit_seq = C0+1` directly (the same call the commit path makes) WITHOUT publishing, then reads.
/// SNAPSHOT-CORRECT (the `created_by` gate): the C-1 reader sees the key EXACTLY ONCE, with the OLD image
/// (old visible: `deleted_by = C0+1 > C0`; new hidden: `created_by = C0+1 > C0`); COUNT is unchanged.
/// THE PRE-FIX BUG: the key TWICE (old + new — a state that never existed). After publish, a reader at C
/// sees exactly the NEW image (old hidden: `deleted_by = C0+1 <= C0+1`; new visible: `created_by <= C0+1`).
/// SABOTAGE-VERIFIED: skip the `created_by` stamp on append (or drop the VM conjunct) and this FAILS.
/// Derive the REAL row identity for a unique int4 key via the device locate + A1 region —
/// what the production commit arm surfaces from the installs' keys (A4b made identity
/// MANDATORY on the incremental update path, so direct `try_update_resident_commit` callers
/// must pass the true ids).
fn device_row_id_for(e: &Engine, table_name: &str, key: i32) -> u64 {
    let table = e.relational_catalog_table(table_name).unwrap();
    let hits = e
        .locate_resident_pk_via_shard_index_detailed(&table, 0, key)
        .expect("locate must answer for a unique resident key");
    let hit = hits.first().expect("at least one slot");
    let region = hit.row_id.as_ref().expect("identity region present");
    let halves = region
        .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
        .unwrap();
    (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32)
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv6_created_by_gate_reader_at_prior_snapshot_never_sees_updated_key_twice() {
    // Run the torn-window differential over BOTH stamp branches: 200 rows -> the open shard has
    // headroom, the append stamps IN PLACE; 256 rows (= 64*4 under the E2.5b-2 first-capacity
    // FLOOR clamped by the target: EVERY shard is born at the 64-row target, so 4 exactly-full
    // shards) -> the open shard is FULL, the append ROLLS OVER a new stamped shard (whose
    // created_by region must install before the shard publishes). The branch actually taken is
    // PROVEN structurally below (shard-count delta), so neither variant can go vacuous if the
    // admit shape changes. (Recalibrated from 258: the pre-floor 1-row admit built a capacity-2
    // shard 0; the floor commit 6c3fe683 made shard 0 birth at the target too.)
    for total_rows in [200_i64, 256_i64] {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..total_rows {
            e.execute_text(
                (i as u64) + 2,
                &format!(
                    "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                    i * 10
                ),
            )
            .unwrap();
        }
        let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;
        let c0 = e.committed_seq();
        assert_eq!(
            sel("SELECT id, balance FROM accounts WHERE id = 130").len(),
            1,
            "pre-update: one row"
        );
        let shard_count = |e: &Engine| {
            e.read_state
                .residency
                .shards
                .load()
                .get("accounts")
                .map_or(0, |s| s.len())
        };
        let shards_before = shard_count(&e);

        // Apply the incremental UPDATE (tombstone-old + append-new) at commit_seq C0+1 WITHOUT
        // publishing — exactly the state a concurrent reader can observe between the residency
        // maintenance and the publication join inside a real commit.
        {
            let id_130 = device_row_id_for(&e, "accounts", 130);
            let guard = e.ddl_catalog();
            let ok = e.try_update_resident_commit(
                &guard,
                "accounts",
                &[vec![SqlValue::Int4(130), SqlValue::Int4(1300)]],
                &[vec![SqlValue::Int4(130), SqlValue::Int4(9999)]],
                c0 + 1,
                Some(&[id_130]),
            );
            assert!(
                ok,
                "the incremental tombstone-old + append-new route must fire at {total_rows} rows \
                 (else this test is vacuous)"
            );
        }
        // NON-VACUITY (route proof): the append STAMPED a created_by region (fallback re-admit / an
        // unstamped append would leave none — and the reads below would then double-count).
        assert!(
            table_has_any_created_by_cell(&e, "accounts"),
            "the UPDATE append must have stamped a created_by region at {total_rows} rows"
        );
        // NON-VACUITY (branch proof): 200 rows must exercise the IN-PLACE stamp (same shard set);
        // 256 rows must exercise the ROLLOVER stamp (a new shard appeared). If the admit shape ever
        // changes these row counts, this assert flags the variant instead of silently going vacuous.
        if total_rows == 200 {
            assert_eq!(
                shard_count(&e),
                shards_before,
                "200 rows: the in-place branch must serve"
            );
        } else {
            assert_eq!(
                shard_count(&e),
                shards_before + 1,
                "{total_rows} rows: the ROLLOVER branch must serve (open shard full)"
            );
        }

        // The C-1 reader (committed_seq is still C0): EXACTLY ONE row, the OLD image.
        let rows = sel("SELECT id, balance FROM accounts WHERE id = 130");
        assert_eq!(
            rows.len(),
            1,
            "SI at {total_rows} rows: a reader at committed_seq C-1 must see the updated key EXACTLY \
             ONCE (2 = the SV5 P2 double-read: old visible via deleted_by > C-1 AND new visible with \
             no created_by gate)"
        );
        assert_eq!(
            rows.row(0),
            &[SqlValue::Int4(130), SqlValue::Int4(1300)],
            "the C-1 snapshot reads the OLD image (the appended new version is not yet visible)"
        );
        assert_eq!(
            sel("SELECT COUNT(*) FROM accounts").row(0),
            &[SqlValue::Int8(total_rows)],
            "COUNT at C-1 is snapshot-correct (no phantom appended row)"
        );

        // Publish the commit: a reader at C sees exactly the NEW image, once.
        e.publish_ready_index(c0 + 1).unwrap();
        let rows = sel("SELECT id, balance FROM accounts WHERE id = 130");
        assert_eq!(rows.len(), 1, "post-publish: exactly one row");
        assert_eq!(
            rows.row(0),
            &[SqlValue::Int4(130), SqlValue::Int4(9999)],
            "a reader at C sees the NEW image (old hidden by deleted_by, new admitted by created_by)"
        );
        assert_eq!(
            sel("SELECT COUNT(*) FROM accounts").row(0),
            &[SqlValue::Int8(total_rows)]
        );
    }
}

/// SV6 — the CONCURRENT-reader form of the double-read differential: a reader thread hammers the point
/// lookup while the writer commits real single-row SQL UPDATEs through mandatory device maintenance
/// ON. SI invariant under EVERY interleaving: the key appears EXACTLY ONCE per read (never 2 = the SV5
/// double-read; never 0 = a lost row). Crosses open-shard append headroom AND rollover (shard target 64,
/// ~300 appended versions), so both created_by stamp branches are exercised under load.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv6_concurrent_reader_never_sees_updated_key_twice_under_update_load() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    // A5 FLIP: this hammer runs ELIDED BY DEFAULT — it is the regression gate for the
    // (fixed) elided-churn SI bug: a rehydrating decline used to leave the fallback on a
    // STALE view -> stale old image -> the tombstone stamped an already-dead slot -> the
    // current version leaked (double-read) or the update silently no-oped (lost update,
    // caught by the end-state assert below). Fix = re-pin the view at every
    // post-rehydration fallback.
    e.set_shard_size_target(64);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    for i in 0..200_i64 {
        e.execute_text(
            (i as u64) + 2,
            &format!(
                "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                i * 10
            ),
        )
        .unwrap();
    }
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        let reader = s.spawn(|| {
            let mut reads = 0_u64;
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                let rows = e
                    .execute_relational_select_text(
                        "SELECT id, balance FROM accounts WHERE id = 130",
                    )
                    .unwrap()
                    .rows;
                assert_eq!(
                    rows.len(),
                    1,
                    "SI under concurrency: id=130 must appear EXACTLY ONCE per read (2 = the SV5 \
                     double-read window; 0 = a lost row)"
                );
                assert_eq!(rows.row(0)[0], SqlValue::Int4(130));
                reads += 1;
            }
            reads
        });
        for t in 0..300_u64 {
            e.execute_text(
                300 + t,
                &format!(
                    "UPDATE accounts SET balance = {} WHERE id = 130",
                    100_000 + t
                ),
            )
            .unwrap();
        }
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let reads = reader
            .join()
            .expect("reader thread must not panic (SI violation = panic)");
        assert!(reads > 0, "the reader must have raced at least one read");
    });
    // Quiescent end-state: the last committed value, exactly once.
    let rows = e
        .execute_relational_select_text("SELECT id, balance FROM accounts WHERE id = 130")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0), &[SqlValue::Int4(130), SqlValue::Int4(100_299)]);
}

/// SV6 lifecycle (mirrors SV4-prereq-#1 for `created_by`): the on-demand `created_by` region is
/// RELEASED at every site the buffer it annotates is retired — a re-admit (here: a multi-row UPDATE
/// falling back to invalidate + rebuild-all-live) must not leave a stale stamp region that would
/// wrongly HIDE rebuilt rows from older-snapshot readers, and DROP TABLE must erase the cell keys
/// entirely (no per-table host-cell leak). Sabotage: remove the `shard_created_by_memory` cleanup at
/// either site and the corresponding assert FAILS.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv6_created_by_region_released_on_readmit_and_drop() {
    let load = |e: &Engine| {
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..200_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!(
                    "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                    i * 10
                ),
            )
            .unwrap();
        }
        e.execute_text(202, "UPDATE accounts SET balance = 9999 WHERE id = 130")
            .unwrap();
        assert!(
            table_has_any_created_by_cell(e, "accounts"),
            "precondition: the incremental UPDATE stamped a live created_by region"
        );
    };

    // Explicit VACUUM crosses the RETIRE-002 repair boundary and replaces the allocation. The
    // region must go with the buffer it annotated.
    let e = Engine::new_local();
    load(&e);
    e.vacuum_table("accounts").unwrap();
    assert!(
        !table_has_any_created_by_cell(&e, "accounts"),
        "re-admit must release the stale created_by region (wrong-results + leak guard)"
    );
    assert!(
        e.read_residency_shards()["accounts"]
            .iter()
            .all(|shard| shard.created_by_region.is_some() || shard.max_created_by == 0),
        "an all-visible re-admission must clear the creation high-water with its sidecar"
    );
    // Reads after the re-admit are the plain all-live scan (no phantom hiding).
    let rows = e
        .execute_relational_select_text("SELECT id, balance FROM accounts WHERE id = 130")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0), &[SqlValue::Int4(130), SqlValue::Int4(9999)]);

    // DROP gate (fresh engine, live stamped region): the cell KEYS must be erased (not just
    // tombstoned to `None`) — invalidate alone would leak a dangling key per dropped table.
    let d = Engine::new_local();
    load(&d);
    d.execute_text(203, "DROP TABLE accounts").unwrap();
    assert!(
        !table_has_any_created_by_key(&d, "accounts"),
        "DROP TABLE must erase the created_by cell entries (no leaked per-table keys / device memory)"
    );
}

/// SV6 — the created_by gate on the INDEX ROUTES (3b single-flight per-hit gate plus the batched
/// device gather and dense emit). The double-read shape can't reach the routes (a duplicated key
/// declines them to the scan), but a KEY-MOVING incremental UPDATE (`id 130 -> 999` at unpublished
/// `C0+1`) leaves the NEW key as a SINGLE stamped hit: a C-1 reader looking up 999 must get ZERO rows
/// (999 does not exist at its snapshot) while 130 still reads the OLD image — on the 3b route AND the
/// batched path, whose GPU kernel applies the created_by visibility boundary directly. Post-publish,
/// 999 is visible and 130 is gone. NON-VACUITY: `shard_index_route_hits` /
/// `sharded_point_batch_hits` prove the routes (not the scan) served. Sabotage: drop the per-hit
/// created_by check or the batched visibility AND — either makes 999 visible at C-1.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv6_created_by_gate_on_index_routes_hides_moved_key_from_older_snapshot() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_index_probe_enabled(true);
    e.set_shard_batched_point_read_enabled(true);
    e.set_shard_size_target(64);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    for i in 0..200_i64 {
        e.execute_text(
            (i as u64) + 2,
            &format!(
                "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                i * 10
            ),
        )
        .unwrap();
    }
    let c0 = e.committed_seq();
    // Move the key: UPDATE accounts SET id = 999 WHERE id = 130, applied at C0+1, UNPUBLISHED.
    {
        let id_130 = device_row_id_for(&e, "accounts", 130);
        let guard = e.ddl_catalog();
        let ok = e.try_update_resident_commit(
            &guard,
            "accounts",
            &[vec![SqlValue::Int4(130), SqlValue::Int4(1300)]],
            &[vec![SqlValue::Int4(999), SqlValue::Int4(1300)]],
            c0 + 1,
            Some(&[id_130]),
        );
        assert!(ok, "the incremental key-moving UPDATE must fire");
    }
    assert!(
        table_has_any_created_by_cell(&e, "accounts"),
        "stamp route proof"
    );
    let table = e.relational_catalog_table("accounts").unwrap();
    let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;

    // (a) 3b single-flight route: the NEW key is a single stamped hit -> the per-hit created_by gate
    // hides it (0 rows at C-1); the OLD key is a single tombstoned-at-C0+1 hit -> still visible.
    let route_hits_before = e.shard_index_route_hits();
    assert_eq!(
        sel("SELECT id, balance FROM accounts WHERE id = 999").len(),
        0,
        "3b route: the moved-to key must be HIDDEN from the C-1 reader (created_by gate)"
    );
    let rows = sel("SELECT id, balance FROM accounts WHERE id = 130");
    assert_eq!(rows.len(), 1, "3b route: the old key is still live at C-1");
    assert_eq!(rows.row(0), &[SqlValue::Int4(130), SqlValue::Int4(1300)]);
    assert!(
        e.shard_index_route_hits() > route_hits_before,
        "non-vacuity: the 3b index route (not the scan) served the C-1 point lookups"
    );

    // (b) Batched gather (the GPU dense kernel applies created/deleted visibility gates to the stamped shard):
    // needle 999 -> 0 rows; needle 130 -> the old image.
    let batch_hits_before = e.sharded_point_batch_hits();
    let batch = e
        .gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            0,
            &[0, 1],
            &[999, 130],
        )
        .expect("batched GPU route completed")
        .expect("the batched sharded gather must serve the visibility-gated device path");
    assert_eq!(batch.ncols, 2);
    assert_eq!(
        batch.needle_ranges[0].1, 0,
        "batched: the moved-to key must be HIDDEN from the C-1 reader (created_by gate)"
    );
    assert_eq!(
        batch.needle_ranges[1].1, 1,
        "batched: the old key is still live at C-1"
    );
    let start = batch.needle_ranges[1].0 as usize * 2;
    assert_eq!(&batch.values[start..start + 2], &[130, 1300]);
    assert!(
        e.sharded_point_batch_hits() > batch_hits_before,
        "non-vacuity: the batched path (not a fallback) served"
    );

    // (c) Publish -> a reader at C sees the move: 999 visible, 130 gone (both routes).
    e.publish_ready_index(c0 + 1).unwrap();
    let rows = sel("SELECT id, balance FROM accounts WHERE id = 999");
    assert_eq!(rows.len(), 1, "post-publish: the moved-to key is visible");
    assert_eq!(rows.row(0), &[SqlValue::Int4(999), SqlValue::Int4(1300)]);
    assert_eq!(
        sel("SELECT id FROM accounts WHERE id = 130").len(),
        0,
        "post-publish: the old key is gone"
    );
    let batch = e
        .gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            0,
            &[0, 1],
            &[999, 130],
        )
        .expect("batched GPU route completed")
        .expect("batched gather post-publish");
    assert_eq!(
        batch.needle_ranges[0].1, 1,
        "batched post-publish: 999 visible"
    );
    assert_eq!(
        batch.needle_ranges[1].1, 0,
        "batched post-publish: 130 hidden"
    );
}
