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

/// Rows the warm-up phase inserted (ids >= 1_000_000).
fn warm_count(rows: &[Vec<SqlValue>]) -> usize {
    rows.iter()
        .filter(|row| matches!(row[0], SqlValue::Int4(id) if id >= 1_000_000))
        .count()
}
