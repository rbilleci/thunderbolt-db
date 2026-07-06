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

fn select_all_rows(engine: &Engine) -> Vec<Vec<SqlValue>> {
    engine
        .execute_relational_select_text("SELECT id, v FROM t ORDER BY id")
        .expect("select id, v")
        .rows
        .into_boxed()
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

    // The covered route + concurrent intents through the fast path.
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
                    engine
                        .execute_covered_insert_intent(
                            txn_ids.fetch_add(1, Ordering::Relaxed),
                            route,
                            &[id, id + 7],
                        )
                        .unwrap();
                }
            });
        }
    });

    // SQL semantics: a duplicate PK through the intent path raises the same
    // 23505 the classic path raises (wave-batched device locate verdict), and
    // commits nothing.
    let err = engine
        .execute_covered_insert_intent(txn_ids.fetch_add(1, Ordering::Relaxed), &route, &[7, 99])
        .unwrap_err();
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
    assert!(engine.wal_unflushed_count() == 0);
    drop(engine); // crash

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
    drop(recovered);

    // Cleanup: serial sidecar + FUA frame segments.
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
    if let (Some(parent), Some(stem)) = (path.parent(), path.file_name()) {
        let prefix = format!("{}.fua.", stem.to_string_lossy());
        if let Ok(entries) = std::fs::read_dir(parent) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with(&prefix) {
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

    let mut results: Vec<Option<Result<(), _>>> = (0..tickets.len()).map(|_| None).collect();
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
