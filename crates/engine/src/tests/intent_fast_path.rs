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
    let mut ticket = engine
        .submit_covered_delete_intent(txn_ids.fetch_add(1, Ordering::Relaxed), route, pk)?;
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

    // Visibility: id=50 is gone; total row count dropped by exactly one.
    let rows = select_all_rows(&engine);
    assert!(
        !rows
            .iter()
            .any(|row| row.first() == Some(&SqlValue::Int4(50))),
        "deleted key must not be visible"
    );

    // DEAD-TWIN REINSERT (the visibility-aware rebuild's reason to exist): reinserting the
    // deleted key must succeed and be visible EXACTLY once — and later deletes still locate.
    assert_eq!(
        commit_intent_via_submit(&engine, &txn_ids, &insert_route, &[50, 999]).unwrap(),
        1,
        "reinserting a deleted key succeeds"
    );
    let rows = select_all_rows(&engine);
    let fifty: Vec<_> = rows
        .iter()
        .filter(|row| row.first() == Some(&SqlValue::Int4(50)))
        .collect();
    assert_eq!(fifty.len(), 1, "reinserted key visible exactly once");
    assert_eq!(fifty[0].get(1), Some(&SqlValue::Int4(999)), "new image wins");
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
    let winners = outcomes
        .iter()
        .filter(|o| matches!(o, Ok(1)))
        .count();
    assert_eq!(winners, 1, "exactly one racing delete wins: {outcomes:?}");
    assert!(
        outcomes.iter().all(|o| match o {
            Ok(0) | Ok(1) => true,
            Err(err) => err.to_string().contains("conflict")
                || err.to_string().contains("intra-wave"),
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
    let mut before = select_all_rows(&engine);
    before.sort();
    drop(engine); // crash

    let mut recovered = Engine::open_durable_wal_segment(&path).unwrap();
    let mut after = select_all_rows(&recovered);
    after.sort();
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
