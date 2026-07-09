//! E2.1 — covered-INSERT intent fast path: route eligibility, duplicate-key
//! semantics through the wave-batched device validation, and crash-recovery
//! parity of the FUA WAL log (replay reproduces the committed store).

use super::*;

/// Route preparation is a SHAPE PROOF: a table that is not yet elided
/// (device-authoritative), or that has no unique index, must be refused with a
/// clear error instead of silently taking an unvalidated fast path.
#[test]
fn covered_insert_route_requires_covered_shape() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();

    // Missing table.
    let err = engine.prepare_covered_insert_route("nope").unwrap_err();
    assert!(err.to_string().contains("does not exist"), "{err}");

    // Binary WAL records disabled.
    let err = engine.prepare_covered_insert_route("t").unwrap_err();
    assert!(err.to_string().contains("binary WAL records"), "{err}");

    // Flags on, but the table is not elided (no GPU warm-up ran), so the
    // wave-batched device validation is unavailable.
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    let err = engine.prepare_covered_insert_route("t").unwrap_err();
    assert!(
        err.to_string()
            .contains("wave-batched device PK validation"),
        "{err}"
    );

    // A non-INT4 column refuses the route before eligibility is even probed.
    engine
        .execute_text(2, "CREATE TABLE wide (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    let err = engine.prepare_covered_insert_route("wide").unwrap_err();
    assert!(
        err.to_string().contains("every column must be INT4"),
        "{err}"
    );
}

/// Commit ONE covered-insert intent via the submit/poll surface — the lanes-mode write entry
/// (the blocking classic-shaped path is refused once the lanes activate).
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

/// The full E2.1 arc on real hardware: warm a PK'd int4 table into elision,
/// prepare the covered route, drive concurrent intents through the fast path
/// (wave-batched device PK validation + device open-shard apply + W5a binary
/// WAL records + FUA fence-pool durability), verify duplicate-key semantics,
/// then CRASH (drop) and reopen from the FUA log — the replayed store must be
/// row-identical to the pre-crash store.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_intent_fast_path_recovers_fua_log_with_row_parity() {
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
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);

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
        if engine.table_install_elided("t") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "table never entered elision on a GPU box");

    // The covered route + concurrent intents through the fast path. In LANES mode the
    // submit/poll surface is the write entry (the blocking classic-shaped path is refused
    // once the lanes activate); serial mode keeps the blocking path.
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
    });

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
    assert_eq!(before.len(), 400 + warm_count(&before));
    // E2.5c 2M+ push (b) non-vacuity: with the fused apply flag on, the merged applies must
    // have run through the FUSED device pass (row parity alone cannot prove which kernel
    // produced the state).
    assert!(
        engine.fused_apply_hits() > 0,
        "fused apply is always on but no fused pass ever ran"
    );
    assert!(engine.wal_unflushed_count() == 0);
    drop(engine); // crash

    // E2.5c-1: lanes-mode reopen REPLAYS serial-then-lanes and CONTINUES appending. In lanes
    // mode this test additionally proves the reopened engine accepts new intent commits (the
    // route's elision re-entry arm re-admits the recovered table) and that a SECOND reopen
    // replays the post-reopen commits too.
    // Disk-authoritative FUA reopen: replay the frame log (binary row-op
    // records decode+install; no SQL re-parse for covered inserts) and verify
    // the store is row-identical.
    let mut recovered = Engine::open_durable_wal_segment(&path).unwrap();
    let after = select_all_rows(&recovered);
    assert_eq!(before, after, "replayed store must be row-identical");
    // The recovered engine serves the same table from a valid residency
    // snapshot (the device store is reconstructible from the log alone).
    let snapshot = recovered
        .populate_relational_residency_snapshot("t")
        .expect("populate residency after recovery");
    assert!(snapshot.is_valid());
    if lanes_mode {
        // The v1 intent-only contract survives reopen: classic DML refused fail-loud.
        let err = recovered
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                "INSERT INTO t VALUES (900000, 0)",
            )
            .expect_err("classic write after lanes reopen must be refused");
        assert!(
            err.to_string().contains("intent lanes are ACTIVE"),
            "expected the intent-only refusal, got: {err}"
        );
        // CONTINUE APPENDING: re-arm the runtime flags, re-prepare the route (the E2.5c-1
        // elision re-entry admits the recovered table with real device backing), and commit
        // fresh intents through the reopened lane set.
        recovered.set_auto_admit_on_commit(true);
        recovered.set_host_install_elision_enabled(true);
        recovered.set_binary_wal_records_enabled(true);
        recovered.set_device_write_locate_enabled(true);
        recovered.set_device_write_locate_wave_batch_enabled(true);
        recovered.set_constrained_elision_enabled(true);
        let route = recovered
            .prepare_covered_insert_route("t")
            .expect("route re-prepares after reopen (elision re-entry)");
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
            before.len() + 50,
            "50 post-reopen commits visible"
        );
        drop(recovered);
        // SECOND reopen: the post-reopen lane commits replay above the first history.
        let recovered_again = Engine::open_durable_wal_segment(&path).unwrap();
        let after_again = select_all_rows(&recovered_again);
        assert_eq!(
            mid, after_again,
            "second reopen must be row-identical including post-reopen lane commits"
        );
        // E2.5c-2: LANES CHECKPOINT on the live recovered engine, then a THIRD reopen through
        // the checkpoint path (sidecar commit + checkpoint-then-suffix replay) with row parity.
        let baseline = recovered_again
            .checkpoint_intent_lanes()
            .expect("lanes checkpoint on the recovered engine");
        drop(recovered_again);
        let recovered_from_checkpoint = Engine::open_durable_wal_segment_auto(&path).unwrap();
        assert_eq!(
            mid,
            select_all_rows(&recovered_from_checkpoint),
            "checkpointed reopen must be row-identical"
        );
        drop(recovered_from_checkpoint);
        let _ = std::fs::remove_file(gpu_db_wal::lanes_checkpoint_sidecar_path(&path));
        let _ = std::fs::remove_file(gpu_db_wal::lanes_checkpoint_segment_path(&path, baseline));
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
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
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
        if engine.table_install_elided("t") {
            return Some(engine.prepare_covered_insert_route("t").unwrap());
        }
    }
    panic!("table never entered elision on a GPU box");
}

/// E2.2(c) + (a) — the driver-multiplexed submit/poll API and the integer-ledger conflict
/// semantics. Submits a batch of distinct-PK intents WITHOUT per-commit blocking, drains them via
/// the single-writer pump, and reaps every ticket. Then proves first-committer-wins on a SAME-WAVE
/// duplicate PK (the integer conflict slot: exactly one commits, the other is a retryable conflict)
/// and a committed-dup 23505 through the async surface.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_intent_submit_poll_driver_and_conflict_semantics() {
    let mut engine = Engine::new_local();
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
    // wins; the other is a first-committer-wins conflict caught by the INTEGER unique-slot ledger.
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
        "exactly one of two same-PK intents commits (integer-ledger first-committer-wins): {ra:?} / {rb:?}"
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
}

/// E2.4a VARIANT 1 — SAME-PK single-winner ACROSS SHARD WORKERS. A LARGE homogeneous-intent wave
/// (>= `SHARD_MIN_WAVE`) containing a duplicate PK takes the sharded sequencer when
/// `GPU_DB_INTENT_SEQUENCER_SHARDS>1`; the duplicate must still resolve to exactly one winner (the
/// per-shard private dedup set — same PK hashes to the same worker), and every distinct PK commits.
/// Robust to either mode: under the default (shards=1, serial) the same invariant holds via the
/// shared integer ledger. Run the sharded arm with `GPU_DB_INTENT_SEQUENCER_SHARDS=4`.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_intent_sharded_wave_same_pk_single_winner() {
    let mut engine = Engine::new_local();
    let txn_ids = AtomicU64::new(2);
    let Some(route) = warm_intent_route(&mut engine, &txn_ids) else {
        return; // driverless box
    };

    // Build ONE wave (submit everything before pumping): 200 distinct PKs plus a duplicate of the
    // first, all in [600000, 600200). 201 items >= SHARD_MIN_WAVE forces the sharded fan-out.
    const BASE: i32 = 600_000;
    const DISTINCT: i32 = 200;
    let mut tickets: Vec<_> = (0..DISTINCT)
        .map(|i| {
            engine
                .submit_covered_insert_intent(
                    txn_ids.fetch_add(1, Ordering::Relaxed),
                    &route,
                    &[BASE + i, i],
                )
                .expect("submit distinct")
        })
        .collect();
    // The duplicate: same PK as position 0, later wave position → it must be the loser.
    tickets.push(
        engine
            .submit_covered_insert_intent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &route,
                &[BASE, 999],
            )
            .expect("submit dup"),
    );

    let mut results: Vec<Option<Result<u64, ExecuteError>>> =
        (0..tickets.len()).map(|_| None).collect();
    let mut reaped = 0usize;
    let mut spins = 0u32;
    while reaped < tickets.len() {
        engine.drive_commit_wave();
        for (ticket, slot) in tickets.iter_mut().zip(results.iter_mut()) {
            if slot.is_none() {
                if let Some(result) = engine.poll_intent(ticket) {
                    *slot = Some(result);
                    reaped += 1;
                }
            }
        }
        spins += 1;
        assert!(spins < 1_000_000, "sharded wave failed to drain");
    }

    let oks = results
        .iter()
        .filter(|r| r.as_ref().unwrap().is_ok())
        .count();
    let errs: Vec<String> = results
        .iter()
        .filter_map(|r| r.as_ref().unwrap().as_ref().err().map(|e| e.to_string()))
        .collect();
    assert_eq!(
        oks,
        DISTINCT as usize,
        "every distinct PK commits; exactly the duplicate loses ({} errs: {:?})",
        errs.len(),
        errs
    );
    assert_eq!(errs.len(), 1, "exactly one loser (the duplicate): {errs:?}");
    assert!(
        errs[0].contains("conflict") || errs[0].contains("duplicate key"),
        "duplicate loses as a first-committer-wins conflict: {}",
        errs[0]
    );

    let count = engine
        .execute_relational_select_text("SELECT COUNT(*) FROM t WHERE id >= 600000 AND id < 700000")
        .unwrap();
    assert!(
        format!("{:?}", count.rows.row(0).first()).contains(&format!("({DISTINCT})")),
        "exactly {DISTINCT} distinct rows visible: {:?}",
        count.rows.row(0).first()
    );
}

/// Rows the warm-up phase inserted (ids >= 1_000_000).
fn warm_count(rows: &[Vec<SqlValue>]) -> usize {
    rows.iter()
        .filter(|row| matches!(row[0], SqlValue::Int4(id) if id >= 1_000_000))
        .count()
}

/// PG-MODEL ASYNC COMMIT (`SynchronousCommit::Off`, per statement): async
/// intents ack at the APPLIED cut while their WAL frames fence behind the
/// ack; a clean drain + reopen recovers EVERY acked row (the loss window
/// exists only under power failure, by contract). Mixed sync/async waves
/// settle both tiers; the engine default flips via
/// `set_synchronous_commit_default` and per-statement mode overrides it.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_async_commit_acks_early_and_recovers_clean_drain() {
    let path = test_wal_path("intent-async-commit");
    let mut engine = Engine::with_durable_wal_segment(&path);
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);

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
        if engine.table_install_elided("t") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "table never entered elision on a GPU box");
    let route = engine.prepare_covered_insert_route("t").unwrap();

    // MIXED WAVES: alternate strict and async intents; every one must ack Ok.
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
                result.expect("distinct-PK intent commits in both modes");
                reaped += 1;
            }
        }
        spins += 1;
        assert!(spins < 10_000_000, "mixed sync/async batch never drained");
    }

    // ENGINE-DEFAULT flip: plain submits now run async; they must still ack.
    engine.set_synchronous_commit_default(crate::SynchronousCommit::Off);
    let mut tickets: Vec<_> = (1000..1100_i32)
        .map(|i| {
            engine
                .submit_covered_insert_intent(
                    txn_ids.fetch_add(1, Ordering::Relaxed),
                    &route,
                    &[i, i + 1],
                )
                .expect("submit intent (async default)")
        })
        .collect();
    let mut reaped = 0usize;
    let mut spins = 0u32;
    while reaped < tickets.len() {
        engine.drive_commit_wave();
        for ticket in tickets.iter_mut() {
            if let Some(result) = engine.poll_intent(ticket) {
                result.expect("async-default intent commits");
                reaped += 1;
            }
        }
        spins += 1;
        assert!(spins < 10_000_000, "async-default batch never drained");
    }

    // CLEAN DRAIN + REOPEN: every acked row (async included) recovers — the
    // async loss window exists only under power failure, never a clean stop.
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
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);

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
        if engine.table_install_elided("t") {
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
        engine.table_install_elided("t"),
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
        engine.table_install_elided("t"),
        "0-row deletes must not de-elide"
    );

    // Visibility: id=50 is gone; total row count dropped by exactly one.
    let rows = select_rows_unordered_sorted(&engine);
    assert!(
        !rows
            .iter()
            .any(|row| row.first() == Some(&SqlValue::Int4(50))),
        "deleted key must not be visible"
    );
    // KNOWN CLIFF (U1 exit finding, open board): a SELECT over a delete-VERSIONED elided
    // shard falls into the rehydrating host arm and DE-ELIDES the table (the plain-scan
    // read path lacks the deleted_by visibility conjunct this shape needs). The covered
    // routes' re-prepare carries the elision RE-ENTRY arm — the production contract until
    // the read path is wired (Tier-3 mixed read+write gate work).
    let insert_route = engine.prepare_covered_insert_route("t").unwrap();
    let delete_route = engine.prepare_covered_delete_route("t").unwrap();

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
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
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
        if engine.table_install_elided("t") {
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

    let mut recovered = Engine::open_durable_wal_segment(&path).unwrap();
    recovered.set_auto_admit_on_commit(true);
    recovered.set_host_install_elision_enabled(true);
    recovered.set_binary_wal_records_enabled(true);
    recovered.set_device_write_locate_enabled(true);
    recovered.set_device_write_locate_wave_batch_enabled(true);
    recovered.set_constrained_elision_enabled(true);
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
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);

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
        if engine.table_install_elided("t") {
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

    // KNOWN CLIFF (U1/U2 exit finding, open board): a read-back over an update/delete-VERSIONED
    // elided shard falls into the rehydrating host arm and DE-ELIDES the table (the plain scan
    // lacks the deleted_by visibility conjunct). The covered route's re-prepare carries the elision
    // RE-ENTRY arm — so this test re-prepares the update route after EVERY de-eliding read, exactly
    // as the delete e2e does. (Un-versioning is VACUUM's job; the mixed read+write gate is Tier-3.)

    // 1-ROW UPDATE (full-row replace): key 50 (id=50, v=57) -> (id=50, v=9999).
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[50, 9999]).unwrap(),
        1,
        "updating a visible row affects exactly one row"
    );
    assert!(
        engine.table_install_elided("t"),
        "an in-place lane update must not de-elide the table (fallback fired?)"
    );
    let rows = select_rows_unordered_sorted(&engine); // DE-ELIDES the versioned shard
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
    update_route = engine.prepare_covered_update_route("t").unwrap(); // re-enter elision
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[987_654, 1]).unwrap(),
        0,
        "updating a never-existing key affects zero rows"
    );
    assert!(
        engine.table_install_elided("t"),
        "0-row updates must not de-elide"
    );
    assert_eq!(
        select_rows_unordered_sorted(&engine).len(),
        rows_before_zero,
        "a 0-row update must not append a phantom row"
    );

    // CHAINED UPDATE: key 50 again -> (50, 8888); the new image wins, still exactly once.
    update_route = engine.prepare_covered_update_route("t").unwrap(); // re-enter elision
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[50, 8888]).unwrap(),
        1,
        "chained update of the same key affects one row"
    );
    let rows = select_rows_unordered_sorted(&engine); // DE-ELIDES
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
    // through the pk index that the update's dead-twin append dropped then a locate rebuilt.
    update_route = engine.prepare_covered_update_route("t").unwrap(); // re-enter elision
    assert_eq!(
        commit_update_via_submit(&engine, &txn_ids, &update_route, &[60, 4242]).unwrap(),
        1,
        "update the target of the delete"
    );
    let delete_route = engine.prepare_covered_delete_route("t").unwrap();
    assert_eq!(
        commit_delete_via_submit(&engine, &txn_ids, &delete_route, 60).unwrap(),
        1,
        "the updated row is locatable + deletable (index survived the dead twin)"
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
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
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
        if engine.table_install_elided("t") {
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
    // Update the evens below 40 to (id, id + 5000) — one row each. Each update's dead-twin append
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
    recovered.set_host_install_elision_enabled(true);
    recovered.set_binary_wal_records_enabled(true);
    recovered.set_device_write_locate_enabled(true);
    recovered.set_device_write_locate_wave_batch_enabled(true);
    recovered.set_constrained_elision_enabled(true);
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
/// ELIDED throughout. Before F3/U4 the dead-twin append dropped the pk-index cache, the next
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
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
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
        if engine.table_install_elided("t") {
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
            engine.table_install_elided("t"),
            "the dup-tolerant index must keep the table elided across update {id} (no dead-twin \
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

/// COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): a compound PRIMARY KEY over i32-section columns is
/// DEVICE-NATIVE — the table elides, and the wave-batched device write-locate validates uniqueness
/// on the surrogate FINGERPRINT while the authoritative recheck compares the FULL tuple. This proves
/// the DEVICE path FIRES (the `device_write_locate_hits` counter advances — not a silent host
/// fallback) AND that uniqueness is exact (a repeated tuple is 23505; a tuple differing in ONE key
/// column is a distinct row). Compound keys take the CLASSIC covered path (`execute_dml_concurrent`),
/// not the fused intent lane. Self-guards on a driverless box.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_primary_key_elides_and_validates_uniqueness_on_device() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE ct (a INT, b INT, v INT, PRIMARY KEY (a, b))",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);

    let txn_ids = AtomicU64::new(2);
    macro_rules! insert {
        ($sql:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $sql)
        };
    }
    insert!("INSERT INTO ct VALUES (1000000, 0, 0)").unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("ct")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU on this box
    }

    // Warm into elision on the classic covered wave path (distinct tuples: unique `a` per row).
    let mut warmed = false;
    for i in 0..10_000_i32 {
        insert!(&format!(
            "INSERT INTO ct VALUES ({}, {}, 0)",
            2_000_000 + i,
            i
        ))
        .unwrap();
        if engine.table_install_elided("ct") {
            warmed = true;
            break;
        }
    }
    assert!(
        warmed,
        "compound-PK table never entered elision on a GPU box"
    );

    // The DEVICE write-locate must actually fire for the compound key (non-vacuity).
    let hits_before = engine.device_write_locate_hits();
    // A brand-new distinct tuple commits.
    insert!("INSERT INTO ct VALUES (5000000, 1, 10)").unwrap();
    // The EXACT tuple again -> 23505 (device fingerprint hit, tuple recheck confirms).
    let dup = insert!("INSERT INTO ct VALUES (5000000, 1, 99)")
        .unwrap_err()
        .to_string();
    assert!(
        dup.contains("duplicate key value violates unique index"),
        "duplicate compound tuple must raise 23505, got: {dup}"
    );
    // Same first key column, DIFFERENT second -> a DISTINCT tuple, must commit (not first-col-only).
    insert!("INSERT INTO ct VALUES (5000000, 2, 11)").unwrap();
    // Same second key column, DIFFERENT first -> also distinct, must commit.
    insert!("INSERT INTO ct VALUES (7000000, 1, 12)").unwrap();

    assert!(
        engine.device_write_locate_hits() > hits_before,
        "the compound-key wave validation must run on the DEVICE (write-locate counter advanced)"
    );
    // Uniqueness never de-elided the table (sustained device path).
    assert!(
        engine.table_install_elided("ct"),
        "compound-PK table must stay elided across the validated inserts"
    );

    // CHARTER (device-fold consistency): the compound index REBUILD folds the fingerprint ON THE
    // DEVICE (submit_compound_fold_fingerprints), while the probe needle is HOST-folded
    // (compound_key_fingerprint) -- they MUST byte-match. The (1000000, 0) tuple was inserted before
    // elision and has survived every geometric device-fold rebuild during warm-up, so it sits in the
    // index at a DEVICE-folded slot; a duplicate of it (host-folded needle) raising 23505 proves the
    // two folds agree (a constant/order divergence would miss -> no 23505 -> this assert fails).
    let dup_rebuilt = insert!("INSERT INTO ct VALUES (1000000, 0, 55)")
        .unwrap_err()
        .to_string();
    assert!(
        dup_rebuilt.contains("duplicate key value violates unique index"),
        "device-folded index fingerprint must match the host needle fold, got: {dup_rebuilt}"
    );

    // Read-your-writes over the elided compound table: exactly the committed distinct tuples exist.
    let Command::Select(count) =
        parse_command("SELECT COUNT(*) FROM ct WHERE a = 5000000").unwrap()
    else {
        unreachable!()
    };
    let rows = engine.execute_relational_select(&count).unwrap().rows;
    assert_eq!(
        rows,
        vec![vec![SqlValue::Int8(2)]],
        "exactly two visible rows share a=5000000 (b=1 and b=2)"
    );
}

/// COMPOUND KEYS (wider types, Stage 2a): a compound PRIMARY KEY over i64 (Int8/Timestamp) columns — and
/// a MIXED int4+int8 key — elides and enforces uniqueness ON THE DEVICE. Each key column folds its i32
/// WORD decomposition into the surrogate fingerprint (i64 -> [low32, high32], matching the section's LE
/// layout); the device fold kernel reads `widths[k]` words per column. Proves: elision, tuple uniqueness
/// (dup -> 23505; distinct OK), the DEVICE fold matches the host needle even across the HIGH word (a
/// value > 2^32 survives geometric rebuilds and its duplicate is caught), and DELETE by the i64 key stays
/// COMPOUND KEYS (wider types, Stage 2c): a compound PRIMARY KEY over a b128 (UUID) column — mixed with
/// int4 — elides + enforces uniqueness ON THE DEVICE (each b128 key column folds 4 i32 words = the LE
/// section bytes) and DELETE by the key stays device-native (materialize now reassembles b128 for the
/// tuple-verify). Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_b128_uuid_key_elides_and_validates_on_device() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE ut (a INT, u UUID, v INT, PRIMARY KEY (a, u))")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);

    let txn_ids = AtomicU64::new(2);
    let uuid = |n: u32| format!("00000000-0000-0000-0000-{:012x}", n);
    macro_rules! sql {
        ($s:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $s)
        };
    }
    // Pre-elision sentinel with a HIGH-word-nonzero uuid (exercises all 4 folded words on rebuild).
    sql!(&format!(
        "INSERT INTO ut VALUES (5, '{}', 0)",
        "ffffffff-0000-0000-0000-000000000001"
    ))
    .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("ut")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU
    }
    let mut warmed = false;
    for i in 0..10_000_u32 {
        sql!(&format!("INSERT INTO ut VALUES ({}, '{}', 0)", 1000 + i, uuid(i))).unwrap();
        if engine.table_install_elided("ut") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "uuid compound-PK table never entered elision on a GPU box");

    // Tuple uniqueness over the b128 key: a distinct uuid commits; the exact (a,u) tuple repeats -> 23505.
    sql!(&format!("INSERT INTO ut VALUES (5, '{}', 1)", uuid(7))).unwrap();
    let dup = sql!(&format!(
        "INSERT INTO ut VALUES (5, '{}', 9)",
        "ffffffff-0000-0000-0000-000000000001"
    ))
    .unwrap_err()
    .to_string();
    assert!(
        dup.contains("duplicate key value violates unique index"),
        "duplicate uuid compound tuple must raise 23505 (4-word fold agreement), got: {dup}"
    );
    assert!(engine.table_install_elided("ut"));
    engine.set_resident_delete_tombstone_enabled(true);

    // DELETE by the b128 (uuid) compound key stays DEVICE-NATIVE: the WHERE uuid literal is coerced
    // Text->Uuid (`bind_delete_filter_groups`), the fingerprint probe locates the slot, and materialize
    // reassembles the uuid for the tuple-verify. Assert elision-retention BEFORE any verifying read.
    let resolve_before = engine.dml_device_resolve_hits();
    sql!(&format!(
        "DELETE FROM ut WHERE a = 5 AND u = '{}'",
        "ffffffff-0000-0000-0000-000000000001"
    ))
    .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > resolve_before,
        "uuid compound DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_install_elided("ut"),
        "uuid compound DELETE must stay device-native, not de-elide"
    );
    // Correctness (may de-elide the versioned table): exactly the (5, uuid(7)) row remains for a=5.
    let Command::Select(count) =
        parse_command("SELECT COUNT(*) FROM ut WHERE a = 5").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&count).unwrap().rows,
        vec![vec![SqlValue::Int8(1)]],
        "deleted the (5, ffff...0001) row; (5, ...0007) remains"
    );
}

/// COMPOUND KEYS (wider types, Stage 2d): a compound PRIMARY KEY over a TEXT column — mixed with int4 —
/// elides and enforces uniqueness ON THE DEVICE. A text key column is variable-length, so it folds to ONE
/// word = the FNV-1a hash of its UTF-8 bytes; the device fold kernel's TEXT branch (`widths[k] == 0`) reads
/// the row's `[start,end)` blob span from the shard's text section and hashes it BYTE-IDENTICALLY to the
/// host `fnv1a_bytes` (so the device index rebuild and the host probe needle agree). A fingerprint collision
/// can only OVER-report a hit, which the full-tuple recheck — now materializing the resident text on-device
/// (`materialize_resident_row_via_hit` Text arm) — separates. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_text_key_elides_and_validates_on_device() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE tt (a INT, s TEXT, v INT, PRIMARY KEY (a, s))")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);

    let txn_ids = AtomicU64::new(2);
    macro_rules! sql {
        ($s:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $s)
        };
    }
    // Pre-elision sentinel: a distinctive text key that must survive the device fold on rebuild.
    sql!("INSERT INTO tt VALUES (5, 'alpha-KEY', 0)").unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("tt")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU
    }
    let mut warmed = false;
    for i in 0..10_000_u32 {
        sql!(&format!("INSERT INTO tt VALUES ({}, 'k{}', 0)", 1000 + i, i)).unwrap();
        if engine.table_install_elided("tt") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "text compound-PK table never entered elision on a GPU box");

    // The DEVICE write-locate must actually fire for the text compound key (non-vacuity).
    let hits_before = engine.device_write_locate_hits();
    // A brand-new distinct tuple commits.
    sql!("INSERT INTO tt VALUES (5, 'beta', 10)").unwrap();
    // The EXACT (a, s) tuple again -> 23505 (device fingerprint hit, on-device text tuple recheck confirms).
    let dup = sql!("INSERT INTO tt VALUES (5, 'beta', 99)")
        .unwrap_err()
        .to_string();
    assert!(
        dup.contains("duplicate key value violates unique index"),
        "duplicate text compound tuple must raise 23505 (text-hash fold agreement), got: {dup}"
    );
    // Re-insert the PRE-ELISION sentinel tuple -> 23505: proves the DEVICE rebuild fold of the resident
    // text blob byte-matches the HOST probe needle's `fnv1a_bytes`.
    let dup_sentinel = sql!("INSERT INTO tt VALUES (5, 'alpha-KEY', 7)")
        .unwrap_err()
        .to_string();
    assert!(
        dup_sentinel.contains("duplicate key value violates unique index"),
        "re-inserting the pre-elision text tuple must raise 23505 (device rebuild == host needle), got: {dup_sentinel}"
    );
    // Same first key column, DIFFERENT text -> a DISTINCT tuple, must commit (not first-col-only).
    sql!("INSERT INTO tt VALUES (5, 'gamma', 11)").unwrap();
    // Same text, DIFFERENT first column -> also distinct, must commit.
    sql!("INSERT INTO tt VALUES (9, 'beta', 12)").unwrap();

    assert!(
        engine.device_write_locate_hits() > hits_before,
        "the text compound-key wave validation must run on the DEVICE (write-locate counter advanced)"
    );
    // Uniqueness never de-elided the table (sustained device path).
    assert!(
        engine.table_install_elided("tt"),
        "text compound-PK table must stay elided across the validated inserts"
    );

    // Read-your-writes over the elided text-compound table: the distinct tuples for a=5 are exactly
    // {alpha-KEY, beta, gamma} (3 rows).
    let Command::Select(count) =
        parse_command("SELECT COUNT(*) FROM tt WHERE a = 5").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&count).unwrap().rows,
        vec![vec![SqlValue::Int8(3)]],
        "a=5 holds exactly the 3 distinct text tuples"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006): the DECLINED-shape general-executor read fallback. A wider-type
/// (int8 / numeric / text) SELECT shape the SPECIALIZED resident route does NOT recognize — a scalar
/// aggregate, a filtered projection, DISTINCT, GROUP BY, single-key ORDER BY, `SELECT *` over a text
/// table — used to DE-ELIDE the table and run on the CPU relational engine. It now routes to the GENERAL
/// GPU Expr executor instead: the read stays ON THE DEVICE (`general_read_fallback_hits` advances) and the
/// table STAYS ELIDED (no rehydrate), while the result matches the expected (spec) answer. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_general_read_fallback_serves_declined_wider_type_shapes_on_device() {
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, b INT8, g INT8, s TEXT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);

    let mut txn = 2u64;
    // A fixed, KNOWN dataset (id 1..=12; b=id*10; g=id%3; s="v{id}"). Insert the first row, then guard on
    // a usable GPU, then insert the rest — the table elides during ingest and holds the full 12 rows.
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, 10, 1, 'v1')")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap.map(|s| s.device_memory_proof.is_none()).unwrap_or(true) {
        return; // self-guard: no usable GPU
    }
    for id in 2..=12i64 {
        engine
            .execute_dml_concurrent(
                txn,
                &format!("INSERT INTO t VALUES ({id}, {}, {}, 'v{id}')", id * 10, id % 3),
            )
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_install_elided("t"),
        "the wider-type PK'd table must elide before the read fallback is exercised"
    );

    // Each shape: (SQL, expected rows). Set-valued shapes are compared order-insensitively.
    let sorted = |mut rows: Vec<Vec<SqlValue>>| {
        rows.sort_by_key(|r| format!("{r:?}"));
        rows
    };
    let run = |engine: &Engine, shape: &str| -> (bool, bool, Vec<Vec<SqlValue>>) {
        let before = engine.general_read_fallback_hits();
        let Command::Select(select) = parse_command(shape).unwrap() else {
            unreachable!()
        };
        let res = engine.execute_relational_select(&select).unwrap();
        let fired = engine.general_read_fallback_hits() > before;
        let elided_after = engine.table_install_elided("t");
        (fired, elided_after, res.rows.iter().map(|r| r.to_vec()).collect())
    };

    // int8 scalar aggregate: SUM(bigint) -> numeric (PG spec); sum(10..120 step 10) = 780.
    let cases: Vec<(&str, Vec<Vec<SqlValue>>)> = vec![
        (
            "SELECT SUM(b) FROM t",
            vec![vec![SqlValue::Numeric(gpu_db_sql::Decimal128::new(780, 0))]],
        ),
        // int8-filtered projection: b = 50 -> id 5.
        ("SELECT id FROM t WHERE b = 50", vec![vec![SqlValue::Int4(5)]]),
        // DISTINCT over an int8 column -> {0,1,2}.
        (
            "SELECT DISTINCT g FROM t",
            vec![
                vec![SqlValue::Int8(0)],
                vec![SqlValue::Int8(1)],
                vec![SqlValue::Int8(2)],
            ],
        ),
        // GROUP BY an int8 column -> each residue class has 4 members.
        (
            "SELECT g, COUNT(*) FROM t GROUP BY g",
            vec![
                vec![SqlValue::Int8(0), SqlValue::Int8(4)],
                vec![SqlValue::Int8(1), SqlValue::Int8(4)],
                vec![SqlValue::Int8(2), SqlValue::Int8(4)],
            ],
        ),
        // single-key ORDER BY on an int8 column + LIMIT -> the 3 smallest b (ids 1,2,3).
        (
            "SELECT id FROM t ORDER BY b LIMIT 3",
            vec![
                vec![SqlValue::Int4(1)],
                vec![SqlValue::Int4(2)],
                vec![SqlValue::Int4(3)],
            ],
        ),
        // SELECT * over a TEXT-bearing table (the resident route excludes text from SELECT *).
        (
            "SELECT * FROM t WHERE id = 7",
            vec![vec![
                SqlValue::Int4(7),
                SqlValue::Int8(70),
                SqlValue::Int8(1),
                SqlValue::Text("v7".to_string()),
            ]],
        ),
        // EMPTY-filtered scalar aggregate: SUM over zero rows is NULL (PG spec), served on-device
        // WITHOUT de-eliding (the general executor's empty-set guard returns NULL, not a hard error).
        (
            "SELECT SUM(b) FROM t WHERE b = 999999",
            vec![vec![SqlValue::Null]],
        ),
    ];

    for (shape, expected) in cases {
        let (fired, elided_after, rows) = run(&engine, shape);
        assert!(
            fired,
            "shape {shape:?} must be served by the GENERAL GPU executor (fallback counter must advance), \
             not de-elided to the CPU engine"
        );
        assert!(
            elided_after,
            "shape {shape:?} must keep the table ELIDED (the on-device read must not rehydrate)"
        );
        assert_eq!(
            sorted(rows),
            sorted(expected),
            "shape {shape:?} result must match the spec answer"
        );
    }
}

/// CPU-ENGINE RETIREMENT (ADR-006): a ZERO-MATCH DELETE / UPDATE (a `WHERE` that matches nothing) is a
/// data NO-OP and must NOT de-elide the table. Confirmed by backtrace that the trigger is the commit
/// path's `apply_and_publish_committed_inner` `!handled && elided -> rehydrate_elided_table` arm:
/// `try_tombstone_resident_delete_commit` / `try_update_resident_commit` returned `false` on an empty
/// applied set, which the commit path treated as "unhandled" and REHYDRATED (de-elided) the table — a pure
/// de-elide trigger on the common `DELETE/UPDATE ... WHERE <no match>` OLTP shape. The zero-match commit
/// now reports HANDLED, the table STAYS ELIDED, and the data is byte-unchanged. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_zero_match_dml_keeps_table_elided() {
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, b INT8, s TEXT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, 100, 'v1')")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap.map(|s| s.device_memory_proof.is_none()).unwrap_or(true) {
        return; // self-guard: no usable GPU
    }
    for id in 2..=6i64 {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, {}, 'v{id}')", id * 100))
            .unwrap();
        txn += 1;
    }
    assert!(engine.table_install_elided("t"), "table must elide first");

    let count = |engine: &Engine| -> i64 {
        let Command::Select(s) = parse_command("SELECT COUNT(*) FROM t").unwrap() else {
            unreachable!()
        };
        match engine.execute_relational_select(&s).unwrap().rows.iter().next().and_then(|r| r.first()) {
            Some(SqlValue::Int8(n)) => *n,
            other => panic!("unexpected COUNT: {other:?}"),
        }
    };
    assert_eq!(count(&engine), 6, "6 rows committed");

    // Zero-match POINT DELETE / UPDATE (WHERE pk = <absent>) -> data no-op, must STAY ELIDED. (A RANGE
    // zero-match de-elides in the PREPARE phase via a separate trigger — the non-point device resolve —
    // handled in a follow-up slice; this slice closes the point-lookup no-match trigger in the commit path.)
    for stmt in [
        "DELETE FROM t WHERE id = 99999",
        "UPDATE t SET b = 0 WHERE id = 99999",
    ] {
        engine.execute_dml_concurrent(txn, stmt).unwrap();
        txn += 1;
        assert!(engine.table_install_elided("t"), "zero-match {stmt:?} must NOT de-elide");
        assert_eq!(count(&engine), 6, "zero-match {stmt:?} changed no rows");
    }

    // Sanity: a MATCHING DELETE still works + stays elided (the fix didn't break the real path).
    engine.execute_dml_concurrent(txn, "DELETE FROM t WHERE id = 3").unwrap();
    assert!(engine.table_install_elided("t"), "a matching DELETE stays elided");
    assert_eq!(count(&engine), 5, "the matching DELETE removed exactly one row");
}

/// CPU-ENGINE RETIREMENT (ADR-006, NULL coverage): an INSERT carrying a NULL used to DE-ELIDE the table
/// (the in-place append has no validity-bitmap channel, so it declined -> re-admit). Now a null-carrying
/// batch rolls a DENSE shard whose validity bitmaps the payload builder constructs (like TEXT), so the
/// table STAYS ELIDED and reads NULL-correctly on-device. And the rehydrate GATHER
/// (`gather_resident_table_rows_from_device`) now materializes a null-bearing shard (reads the bitmap ->
/// SqlValue::Null) instead of declining + hard-erroring — so a DML that must rehydrate a null-bearing
/// elided table de-elides SAFELY (correct), never crashes. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_null_insert_keeps_table_elided_and_reads_correctly() {
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);

    let mut txn = 2u64;
    engine.execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, 10)").unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap.map(|s| s.device_memory_proof.is_none()).unwrap_or(true) {
        return; // self-guard: no usable GPU
    }
    for id in 2..=6i64 {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, {})", id * 10))
            .unwrap();
        txn += 1;
    }
    assert!(engine.table_install_elided("t"), "table must elide first");

    // INSERT a NULL value: must STAY ELIDED (was: de-elide).
    engine.execute_dml_concurrent(txn, "INSERT INTO t VALUES (7, NULL)").unwrap();
    txn += 1;
    assert!(
        engine.table_install_elided("t"),
        "a NULL insert must NOT de-elide the table (the rollover builds the validity bitmap)"
    );

    // Read the NULL back ON-DEVICE (the row set stayed elided). Use the text entry so `IS NULL` (which the
    // strict hand-rolled parser rejects) routes through the general executor.
    let read = |engine: &Engine, sql: &str| -> Vec<Vec<SqlValue>> {
        let mut rows: Vec<Vec<SqlValue>> = engine
            .execute_relational_select_text(sql)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect();
        rows.sort_by_key(|r| format!("{r:?}"));
        rows
    };
    assert_eq!(
        read(&engine, "SELECT id, v FROM t WHERE id = 7"),
        vec![vec![SqlValue::Int4(7), SqlValue::Null]],
        "the NULL reads back as NULL, not a phantom 0"
    );
    assert_eq!(
        read(&engine, "SELECT COUNT(*) FROM t"),
        vec![vec![SqlValue::Int8(7)]],
        "all 7 rows present (6 + the null row)"
    );
    // 3VL: IS NULL finds exactly the null row; a value filter excludes it.
    assert_eq!(
        read(&engine, "SELECT id FROM t WHERE v IS NULL"),
        vec![vec![SqlValue::Int4(7)]],
        "IS NULL finds exactly the null row"
    );
    assert!(engine.table_install_elided("t"), "reads must not de-elide the null-bearing table");

    // A DML that must REHYDRATE the null-bearing table (materialize declines a null shard) de-elides
    // SAFELY now that the gather materializes nulls — no "device-authoritative invariant broken" crash.
    engine.execute_dml_concurrent(txn, "DELETE FROM t WHERE id = 5").unwrap();
    assert_eq!(
        read(&engine, "SELECT COUNT(*) FROM t"),
        vec![vec![SqlValue::Int8(6)]],
        "the DELETE removed exactly one row (no crash on the null-bearing rehydrate)"
    );
    assert_eq!(
        read(&engine, "SELECT id, v FROM t WHERE id = 7"),
        vec![vec![SqlValue::Int4(7), SqlValue::Null]],
        "the null row survives the rehydrate with its NULL intact"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006): a RANGE / non-point DELETE / UPDATE on an ELIDED table now resolves on
/// the DEVICE via the predicate scan-locate (`try_resolve_dml_via_predicate_scan` -> the WHERE lowered to a
/// ResidentExpr, evaluated per shard by `lower_resident_predicate`, each matching slot materialized with
/// SV3b/SV6 visibility + the full WHERE rechecked) instead of REHYDRATING (de-eliding) in the prepare
/// phase. A zero-match range stays elided (no churn); a matching range deletes/updates the exact rows and
/// STAYS ELIDED, with the device resolve counter advancing (non-vacuity). Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_range_dml_resolves_on_device_without_deelide() {
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    let mut txn = 2u64;
    engine.execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, 10)").unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap.map(|s| s.device_memory_proof.is_none()).unwrap_or(true) {
        return; // self-guard: no usable GPU
    }
    for id in 2..=8i64 {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, {})", id * 10))
            .unwrap();
        txn += 1;
    }
    assert!(engine.table_install_elided("t"), "table must elide first");

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
            unreachable!()
        };
        let mut out: Vec<i64> = engine
            .execute_relational_select(&s)
            .unwrap()
            .rows
            .iter()
            .map(|r| match r.first() {
                Some(SqlValue::Int4(n)) => *n as i64,
                other => panic!("unexpected id: {other:?}"),
            })
            .collect();
        out.sort_unstable();
        out
    };
    assert_eq!(ids(&engine), (1..=8).collect::<Vec<_>>());

    // Zero-match RANGE DELETE -> device resolve, no rows, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine.execute_dml_concurrent(txn, "DELETE FROM t WHERE id > 100000").unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "zero-match range DELETE must RESOLVE on the device (counter advances)"
    );
    assert!(engine.table_install_elided("t"), "zero-match range DELETE must NOT de-elide");
    assert_eq!(ids(&engine), (1..=8).collect::<Vec<_>>(), "no rows deleted");

    // Matching RANGE DELETE (id > 6) -> deletes ids 7,8 ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine.execute_dml_concurrent(txn, "DELETE FROM t WHERE id > 6").unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "matching range DELETE must RESOLVE on the device"
    );
    assert!(engine.table_install_elided("t"), "matching range DELETE must NOT de-elide");
    assert_eq!(ids(&engine), (1..=6).collect::<Vec<_>>(), "ids 7,8 deleted exactly");

    // Matching RANGE UPDATE (id <= 2 SET v=0) -> updates ids 1,2 ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine.execute_dml_concurrent(txn, "UPDATE t SET v = 0 WHERE id <= 2").unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "matching range UPDATE must RESOLVE on the device"
    );
    assert!(engine.table_install_elided("t"), "matching range UPDATE must NOT de-elide");
    // The update kept the row set (still ids 1..=6) and set v=0 for ids 1,2.
    assert_eq!(ids(&engine), (1..=6).collect::<Vec<_>>(), "UPDATE changed no id set");
    let Command::Select(cnt) = parse_command("SELECT COUNT(*) FROM t WHERE v = 0").unwrap() else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&cnt).unwrap().rows.iter().next().and_then(|r| r.first()),
        Some(&SqlValue::Int8(2)),
        "exactly ids 1,2 now have v=0"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, wider-type range DML): an INT8 range DELETE/UPDATE on an ELIDED table
/// resolves ON THE DEVICE via `try_resolve_dml_via_predicate_scan` — the WHERE lowers to an int8
/// `ResidentExpr` (`Column(int8) <op> Int8Literal`, the new VM literal) evaluated at I64 width by
/// `CompareScalarI64`, so a LARGE i64 bound (> i32::MAX, e.g. a bigint/timestamp-scale value) that cannot
/// fit an Int4Literal resolves on-device instead of de-eliding. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_int8_range_dml_resolves_on_device() {
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, b INT8)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    // b = id * 2_000_000_000 -> ids 3..=8 have b > i32::MAX (2.1e9), so the bound cannot be an Int4Literal.
    let big = 2_000_000_000i64;
    let mut txn = 2u64;
    engine.execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES (1, {})", big)).unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap.map(|s| s.device_memory_proof.is_none()).unwrap_or(true) {
        return; // self-guard: no usable GPU
    }
    for id in 2..=8i64 {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, {})", id * big))
            .unwrap();
        txn += 1;
    }
    assert!(engine.table_install_elided("t"), "table must elide first");

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
            unreachable!()
        };
        let mut out: Vec<i64> = engine
            .execute_relational_select(&s)
            .unwrap()
            .rows
            .iter()
            .map(|r| match r.first() {
                Some(SqlValue::Int4(n)) => *n as i64,
                other => panic!("unexpected id: {other:?}"),
            })
            .collect();
        out.sort_unstable();
        out
    };
    assert_eq!(ids(&engine), (1..=8).collect::<Vec<_>>());

    // A LARGE-bound int8 range DELETE (b > 6e9 -> ids 4..=8) resolves ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, &format!("DELETE FROM t WHERE b > {}", 6 * big))
        .unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "int8 range DELETE with a >i32 bound must RESOLVE on the device"
    );
    assert!(engine.table_install_elided("t"), "int8 range DELETE must NOT de-elide");
    assert_eq!(ids(&engine), (1..=6).collect::<Vec<_>>(), "ids 7,8 (b=14e9,16e9) deleted exactly");

    // A LARGE-bound int8 range UPDATE (b <= 4e9 = 2*big -> ids 1,2; 4e9 > i32::MAX) resolves ON THE
    // DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, &format!("UPDATE t SET b = 0 WHERE b <= {}", 2 * big))
        .unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "int8 range UPDATE with a >i32 bound must RESOLVE on the device"
    );
    assert!(engine.table_install_elided("t"), "int8 range UPDATE must NOT de-elide");
    assert_eq!(ids(&engine), (1..=6).collect::<Vec<_>>(), "UPDATE changed no id set");
    let Command::Select(cnt) = parse_command("SELECT COUNT(*) FROM t WHERE b = 0").unwrap() else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&cnt).unwrap().rows.iter().next().and_then(|r| r.first()),
        Some(&SqlValue::Int8(2)),
        "exactly ids 1,2 (b=2e9,4e9) now have b=0"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, wider-type range DML): a single-comparison TIMESTAMP range DELETE (the
/// common data-purge shape, `WHERE ts < '<cutoff>'`) on an ELIDED table resolves ON THE DEVICE — the DML
/// predicate builder emits `Column(ts) <op> Int8Literal(micros)` (a timestamp is i64 micros in the i64
/// section) and the timestamp peephole, now accepting a raw-micros Int8Literal, evaluates it via the i64
/// compare kernel. Stays elided; deletes the exact rows. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_timestamp_range_delete_resolves_on_device() {
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, ts TIMESTAMP)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);

    // ids 1..=6 at ts = 2020-01..06-01. Purge everything strictly before 2020-04-01 -> ids 1,2,3.
    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, '2020-01-01 00:00:00')")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap.map(|s| s.device_memory_proof.is_none()).unwrap_or(true) {
        return; // self-guard: no usable GPU
    }
    for (id, month) in (2..=6i64).zip(2..=6) {
        engine
            .execute_dml_concurrent(
                txn,
                &format!("INSERT INTO t VALUES ({id}, '2020-0{month}-01 00:00:00')"),
            )
            .unwrap();
        txn += 1;
    }
    assert!(engine.table_install_elided("t"), "table must elide first");

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
            unreachable!()
        };
        let mut out: Vec<i64> = engine
            .execute_relational_select(&s)
            .unwrap()
            .rows
            .iter()
            .map(|r| match r.first() {
                Some(SqlValue::Int4(n)) => *n as i64,
                other => panic!("unexpected id: {other:?}"),
            })
            .collect();
        out.sort_unstable();
        out
    };
    assert_eq!(ids(&engine), (1..=6).collect::<Vec<_>>());

    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, "DELETE FROM t WHERE ts < '2020-04-01 00:00:00'")
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "timestamp range DELETE must RESOLVE on the device"
    );
    assert!(engine.table_install_elided("t"), "timestamp range DELETE must NOT de-elide");
    assert_eq!(ids(&engine), vec![4, 5, 6], "ids 1,2,3 (Jan-Mar) purged exactly");
}

/// CPU-ENGINE RETIREMENT (ADR-006, audit BLOCKER fix): the `handled=true`-on-empty change lets a data
/// no-op report HANDLED, but `handled` ALSO drives elision-ENTER — which must be gated on a non-empty
/// applied set (only a real append/tombstone confirms the device residency). Otherwise a zero-row op on a
/// NON-elided / non-resident table would ENTER elision and a later op would hard-error
/// "device-authoritative invariant broken". This exercises the SERIALIZED commit path (`execute_text`,
/// where elision-ENTER lives) with an EMPTY eligible never-resident table (per the auditor's reproducer):
/// the zero-row DELETE produces `Some(rows=[])` -> `try_tombstone` empty-return `true` -> handled; without
/// the `applied_changed_rows` guard it would ENTER elision on a table with no device backing. Driverless.
#[test]
fn zero_row_dml_must_not_enter_elision_on_nonelided_table() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_host_install_elision_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    // EMPTY, eligible, NON-elided, no device residency. MUST use execute_text (serialized) — the ENTER
    // block lives there; apply_delete => Some(rows=[]) => try_tombstone empty-return => handled=true.
    engine
        .execute_text(2, "DELETE FROM t WHERE id > 100000")
        .unwrap();
    assert!(
        !engine.table_install_elided("t"),
        "a zero-row DELETE on a non-elided/non-resident table must NOT enter elision"
    );
    engine
        .execute_text(3, "UPDATE t SET v = 0 WHERE id > 100000")
        .unwrap();
    assert!(
        !engine.table_install_elided("t"),
        "a zero-row UPDATE on a non-elided/non-resident table must NOT enter elision"
    );

    // A subsequent real INSERT still commits + reads back (without the guard the table would be elided with
    // no device backing, and this path would hit the rehydrate "invariant broken" hard error).
    engine.execute_text(4, "INSERT INTO t VALUES (1, 10)").unwrap();
    engine.execute_text(5, "INSERT INTO t VALUES (2, 20)").unwrap();
    let Command::Select(s) = parse_command("SELECT COUNT(*) FROM t").unwrap() else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&s).unwrap().rows.iter().next().and_then(|r| r.first()),
        Some(&SqlValue::Int8(2)),
        "both inserts visible (no lost rows / no hard error)"
    );
}

/// COMPOUND KEYS (wider types, Stage 2a): a compound PRIMARY KEY over i64 (Int8/Timestamp) columns — and
/// a MIXED int4+int8 key — elides and enforces uniqueness ON THE DEVICE. Each key column folds its i32
/// WORD decomposition into the surrogate fingerprint (i64 -> [low32, high32] LE, matching the section's LE
/// layout); the device fold kernel reads `widths[k]` words per column. Proves: elision, tuple uniqueness
/// (dup -> 23505; distinct OK), the DEVICE fold matches the host needle even across the HIGH word (a
/// value > 2^32 survives geometric rebuilds and its duplicate is caught), and DELETE by the i64 key stays
/// device-native. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_i64_and_mixed_key_elides_and_validates_on_device() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE ct (a INT8, b INT8, v INT, PRIMARY KEY (a, b))")
        .unwrap();
    engine
        .execute_text(2, "CREATE TABLE mt (a INT, b INT8, v INT, PRIMARY KEY (a, b))")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    let txn_ids = AtomicU64::new(3);
    macro_rules! sql {
        ($s:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $s)
        };
    }
    // Pre-elision sentinel whose first key column exceeds 2^32 (high word non-zero) — it survives the
    // geometric device-fold rebuilds during warm-up, so a later duplicate probes it via the device fold.
    sql!("INSERT INTO ct VALUES (5000000000, 1, 0)").unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("ct")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU
    }
    let mut warmed = false;
    for i in 0..10_000_i64 {
        sql!(&format!(
            "INSERT INTO ct VALUES ({}, {}, 0)",
            6_000_000_000_i64 + i,
            i
        ))
        .unwrap();
        if engine.table_install_elided("ct") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "i64 compound-PK table never entered elision on a GPU box");

    // Tuple uniqueness over i64 keys: distinct tuples commit; the exact tuple repeats -> 23505.
    sql!("INSERT INTO ct VALUES (5000000000, 2, 10)").unwrap(); // same a, different b -> OK
    let dup = sql!("INSERT INTO ct VALUES (5000000000, 1, 99)")
        .unwrap_err()
        .to_string();
    // DEVICE-FOLD consistency across the HIGH word: (5000000000 = 0x1_2A05F200, high word = 1) sat in a
    // device-folded rebuilt slot; the host-folded duplicate needle must match (a 1-word device fold would
    // miss the high 32 bits -> no 23505).
    assert!(
        dup.contains("duplicate key value violates unique index"),
        "duplicate i64 compound tuple must raise 23505 (2-word fold agreement), got: {dup}"
    );
    assert!(engine.table_install_elided("ct"));

    // DELETE + UPDATE by the i64 compound key stay DEVICE-NATIVE (Stage 2b: the SV4b in-place
    // tombstone-locate folds the i64 fingerprint + tuple-verifies the slot, so no int4-predicate
    // de-elide). Run BOTH write ops FIRST and assert elision-retention immediately: a verifying SELECT
    // on a VERSIONED wider-type table de-elides it (a read-path limitation — R-ver is int4-only —
    // orthogonal to these WRITE ops), so correctness is checked AFTER the elision asserts.
    sql!("INSERT INTO ct VALUES (5000000000, 3, 30)").unwrap();
    let resolve_before = engine.dml_device_resolve_hits();
    sql!("DELETE FROM ct WHERE a = 5000000000 AND b = 1").unwrap();
    assert!(
        engine.dml_device_resolve_hits() > resolve_before,
        "i64 compound DELETE must RESOLVE its target on the device"
    );
    assert!(
        engine.table_install_elided("ct"),
        "i64 compound DELETE must stay device-native (fingerprint tombstone-locate), not de-elide"
    );
    sql!("UPDATE ct SET v = 777 WHERE a = 5000000000 AND b = 2").unwrap();
    assert!(
        engine.table_install_elided("ct"),
        "i64 compound UPDATE must stay device-native (fingerprint tombstone-locate), not de-elide"
    );

    // Correctness (may de-elide the versioned table — checked AFTER the elision-retention asserts):
    // deleted only (5000000000,1) [b=2 and b=3 remain]; the UPDATE set v=777 on exactly (5000000000,2).
    let Command::Select(count) =
        parse_command("SELECT COUNT(*) FROM ct WHERE a = 5000000000").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&count).unwrap().rows,
        vec![vec![SqlValue::Int8(2)]],
        "deleted only (5000000000,1); b=2 and b=3 remain"
    );
    let Command::Select(sel) =
        parse_command("SELECT v FROM ct WHERE a = 5000000000 AND b = 2").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&sel).unwrap().rows,
        vec![vec![SqlValue::Int4(777)]],
        "i64 compound UPDATE sets v=777 on exactly the (5000000000,2) tuple"
    );

    // MIXED int4+int8 compound key: elide + enforce uniqueness.
    sql!("INSERT INTO mt VALUES (7, 8000000000, 0)").unwrap();
    if engine
        .populate_relational_residency_snapshot("mt")
        .expect("populate mt")
        .device_memory_proof
        .is_some()
    {
        let mut mt_warmed = false;
        for i in 0..10_000_i32 {
            sql!(&format!(
                "INSERT INTO mt VALUES ({}, {}, 0)",
                100 + i,
                9_000_000_000_i64 + i as i64
            ))
            .unwrap();
            if engine.table_install_elided("mt") {
                mt_warmed = true;
                break;
            }
        }
        assert!(mt_warmed, "mixed compound-PK table never entered elision");
        sql!("INSERT INTO mt VALUES (7, 8000000001, 1)").unwrap(); // distinct b -> OK
        let mdup = sql!("INSERT INTO mt VALUES (7, 8000000000, 2)")
            .unwrap_err()
            .to_string();
        assert!(
            mdup.contains("duplicate key value violates unique index"),
            "duplicate mixed compound tuple must raise 23505, got: {mdup}"
        );
        assert!(engine.table_install_elided("mt"));
    }
}

/// COMPOUND KEYS (operational cases): a DELETE / UPDATE BY a compound key resolves its target ON THE
/// DEVICE (the SQL resolve builds the surrogate fingerprint from the key columns' Eq predicates and
/// probes the compound index; the full `filter_groups` recheck restores tuple exactness), so the table
/// STAYS ELIDED instead of de-eliding to a host rehydrate. This proves: the device resolve FIRES (the
/// `dml_device_resolve_hits` counter advances), the table stays elided across the DELETE + UPDATE, and
/// the ops are TUPLE-EXACT (deleting `(a,b1)` leaves `(a,b2)` — not first-column-only). Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_delete_update_by_key_stays_device_native() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE ct (a INT, b INT, v INT, PRIMARY KEY (a, b))")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_dml_device_resolve_enabled(true);

    let txn_ids = AtomicU64::new(2);
    macro_rules! sql {
        ($s:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $s)
        };
    }
    sql!("INSERT INTO ct VALUES (1000000, 0, 0)").unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("ct")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU
    }
    let mut warmed = false;
    for i in 0..10_000_i32 {
        sql!(&format!("INSERT INTO ct VALUES ({}, {}, 0)", 2_000_000 + i, i)).unwrap();
        if engine.table_install_elided("ct") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "compound-PK table never entered elision on a GPU box");
    // Two rows sharing the first key column a=5 but differing in b.
    sql!("INSERT INTO ct VALUES (5, 1, 100)").unwrap();
    sql!("INSERT INTO ct VALUES (5, 2, 200)").unwrap();
    assert!(engine.table_install_elided("ct"));

    // DELETE by the FULL compound key -> device resolve; the table must STAY ELIDED (not rehydrate).
    let resolve_before = engine.dml_device_resolve_hits();
    sql!("DELETE FROM ct WHERE a = 5 AND b = 1").unwrap();
    assert!(
        engine.dml_device_resolve_hits() > resolve_before,
        "compound DELETE must resolve its target ON THE DEVICE (counter advanced)"
    );
    assert!(
        engine.table_install_elided("ct"),
        "compound DELETE must not de-elide the table"
    );

    // TUPLE EXACTNESS: (5,1) is gone; (5,2) survives (a DELETE keyed on the tuple, not column a).
    let count_a5 = |engine: &Engine| -> i64 {
        let Command::Select(sel) =
            parse_command("SELECT COUNT(*) FROM ct WHERE a = 5").unwrap()
        else {
            unreachable!()
        };
        match engine.execute_relational_select(&sel).unwrap().rows[0][0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("unexpected count value {other:?}"),
        }
    };
    assert_eq!(count_a5(&engine), 1, "exactly one a=5 row remains after deleting (5,1)");

    // UPDATE by the full compound key -> device resolve; stays elided; hits the right tuple.
    sql!("UPDATE ct SET v = 999 WHERE a = 5 AND b = 2").unwrap();
    assert!(
        engine.table_install_elided("ct"),
        "compound UPDATE must not de-elide the table"
    );
    let Command::Select(sel) =
        parse_command("SELECT v FROM ct WHERE a = 5 AND b = 2").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&sel).unwrap().rows,
        vec![vec![SqlValue::Int4(999)]],
        "compound UPDATE must set v=999 on exactly the (5,2) tuple"
    );
}

/// COMPOUND KEYS (audit regression): a compound index's per-shard device-index cache is keyed by
/// `FLAG | ordinal`. Dropping an EARLIER constraint shifts the ordinals of the following indexes, so
/// a SURVIVING cache entry could alias the shifted index (its probe would read an index built from
/// the WRONG key columns -> a missed duplicate / silent UNIQUE violation). This is closed because any
/// index-shape DDL is an "other DDL" in `residency_invalidation_scope` -> the GLOBAL residency
/// invalidation, which purges the PK device-index cache (`purge_shard_pk_index_for_table`) for every
/// table (see `index_probe_key_id`'s CACHE SAFETY note). This test drops the PK so the surviving
/// `(c,d)` UNIQUE shifts from ordinal 1 to 0 and proves it still raises 23505 on a duplicate `(c,d)`
/// tuple — i.e. the ordinal-shift invariant holds end-to-end. Self-guards on a driverless box.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_drop_constraint_shifts_ordinal_without_aliasing_the_device_index() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE ct (a INT, b INT, c INT, d INT, \
             CONSTRAINT ct_pk PRIMARY KEY (a, b), CONSTRAINT ct_cd UNIQUE (c, d))",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);

    let txn_ids = AtomicU64::new(2);
    macro_rules! insert {
        ($sql:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $sql)
        };
    }
    insert!("INSERT INTO ct VALUES (1000000, 0, 1000000, 0)").unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("ct")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU
    }
    // Warm into elision; both compound indexes build their device caches during the wave probes.
    let mut warmed = false;
    for i in 0..10_000_i32 {
        insert!(&format!(
            "INSERT INTO ct VALUES ({}, {}, {}, {})",
            2_000_000 + i,
            i,
            3_000_000 + i,
            i
        ))
        .unwrap();
        if engine.table_install_elided("ct") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "compound table never entered elision on a GPU box");
    // A sentinel row establishes the (c,d) = (100, 200) tuple.
    insert!("INSERT INTO ct VALUES (5, 5, 100, 200)").unwrap();

    // Drop the PRIMARY KEY (ordinal 0) -> the surviving `ct_cd` UNIQUE (c,d) shifts to ordinal 0.
    engine
        .execute_text(90_000, "ALTER TABLE ONLY public.ct DROP CONSTRAINT ct_pk")
        .unwrap();

    // The (c,d) uniqueness MUST still be enforced through its device index after the shift: a
    // DISTINCT (a,b) but DUPLICATE (c,d) tuple raises 23505. On the buggy (un-purged) build the
    // (c,d) probe aliased the stale (a,b) index at ordinal 0, missed the duplicate, and committed.
    let dup = insert!("INSERT INTO ct VALUES (6, 6, 100, 200)")
        .unwrap_err()
        .to_string();
    assert!(
        dup.contains("duplicate key value violates unique index"),
        "surviving compound UNIQUE must catch the duplicate after the ordinal shift, got: {dup}"
    );
    // And a genuinely new (c,d) tuple still commits (the index is live, not wedged-declining).
    insert!("INSERT INTO ct VALUES (7, 7, 101, 201)").unwrap();
}
