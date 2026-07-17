/// M1 design B — WAVE-TIME batched validation differential: an elided PK'd table runs the
/// constraint+DML gauntlet with wave-batch ON (device_write_locate + wave_batch) vs the
/// host-probe oracle (both off). Every outcome (incl 23505 text) + read must match — the
/// deferred INSERT unique check now happens at wave time, batched. NON-VACUITY: the batched
/// locate FIRED (device_write_locate_hits > 0).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn wave_batch_validation_matches_host_oracle() {
    let run = |wave_batch: bool| {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        if wave_batch {
            e.set_device_write_locate_enabled(true);
            e.set_device_write_locate_wave_batch_enabled(true);
        }
        e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
            .unwrap();
        let mut seq = 2u64;
        for chunk in 0..2_i64 {
            let values: Vec<String> = (chunk * 100..(chunk + 1) * 100)
                .map(|k| format!("({k},{})", k * 10))
                .collect();
            e.execute_text(
                seq,
                &format!("INSERT INTO t (id, v) VALUES {}", values.join(",")),
            )
            .unwrap();
            seq += 1;
        }
        // A DDL mid-stream forces the catalog-drift full-validate path for a later insert.
        let ladder = [
            "INSERT INTO t (id, v) VALUES (500, 5000)", // new key -> pass (batched)
            "INSERT INTO t (id, v) VALUES (42, 1)",     // dup seeded key -> 23505
            "INSERT INTO t (id, v) VALUES (500, 9)",    // dup elided-era key -> 23505
            "UPDATE t SET v = 7 WHERE id = 130",        // A2 resolve (not a wave-batch insert)
            "DELETE FROM t WHERE id = 43",
            "INSERT INTO t (id, v) VALUES (43, 2)", // reuse deleted key -> pass
        ];
        let mut outcomes: Vec<Result<(), String>> = Vec::new();
        for sql in &ladder {
            outcomes.push(
                e.execute_text(seq, sql)
                    .map(|_| ())
                    .map_err(|err| err.to_string()),
            );
            seq += 1;
        }
        let mut rows = e
            .execute_relational_select_text("SELECT id, v FROM t")
            .unwrap()
            .rows
            .into_boxed();
        rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        (e, outcomes, rows)
    };
    let (on, on_out, on_rows) = run(true);
    let (off, off_out, off_rows) = run(false);
    assert_eq!(on_out, off_out, "wave-batch outcome ladder == host oracle");
    assert_eq!(on_rows, off_rows, "wave-batch reads == host oracle");
    assert!(
        on.device_write_locate_hits() > 0,
        "non-vacuity: the batched locate must have FIRED"
    );
    assert_eq!(off.device_write_locate_hits(), 0);
}

/// M1 design B — the CONCURRENT dup race through the WAVE-BATCH path: 8 writers contend for
/// the SAME 200 keys on an elided PK'd table with wave-batch validation ON. Exactly one
/// writer wins each key: same-wave dups fall to bounded wave-local slot arbitration, and
/// already-committed dups fall to the wave-batch device locate. No key double-inserts.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn wave_batch_concurrent_dup_race_single_winner() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64);
    e.set_device_write_locate_enabled(true);
    e.set_device_write_locate_wave_batch_enabled(true);
    e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    for (seq, chunk) in (2_u64..).zip(0..2_i64) {
        let values: Vec<String> = (chunk * 100..(chunk + 1) * 100)
            .map(|k| format!("({k},{})", k * 10))
            .collect();
        e.execute_text(
            seq,
            &format!("INSERT INTO t (id, v) VALUES {}", values.join(",")),
        )
        .unwrap();
    }
    // Enter elision.
    for t in 0..20_u64 {
        e.execute_dml_concurrent(
            100 + t,
            &format!("INSERT INTO t (id, v) VALUES ({}, 1)", 5_000 + t),
        )
        .unwrap();
    }
    assert!(e.table_install_elided("t"), "premise: elided");
    let wins: Vec<std::sync::atomic::AtomicU32> = (0..200)
        .map(|_| std::sync::atomic::AtomicU32::new(0))
        .collect();
    let txn = std::sync::atomic::AtomicU64::new(10_000);
    std::thread::scope(|scope| {
        for w in 0..8_u64 {
            let e = &e;
            let wins = &wins;
            let txn = &txn;
            scope.spawn(move || {
                for k in 0..200_i64 {
                    let key = 20_000 + ((k + w as i64 * 25) % 200);
                    loop {
                        let t = txn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        match e.execute_dml_concurrent(
                            t,
                            &format!("INSERT INTO t (id, v) VALUES ({key}, {w})"),
                        ) {
                            Ok(()) => {
                                wins[(key - 20_000) as usize]
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                break;
                            }
                            Err(err) => {
                                if err.to_string().contains("duplicate key") {
                                    break;
                                }
                                // serialization conflict -> retry
                            }
                        }
                    }
                }
            });
        }
    });
    for (k, c) in wins.iter().enumerate() {
        assert_eq!(
            c.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "key {k}: exactly one winner"
        );
    }
    let rows = e
        .execute_relational_select_text("SELECT id FROM t")
        .unwrap()
        .rows;
    assert_eq!(
        rows.len(),
        200 + 20 + 200,
        "seed + waved + one win per contended key"
    );
}

/// Ledger #18 — the DETERMINISTIC same-snapshot dup race: two writers INSERT the SAME PK
/// on an elided table, BARRIERED between snapshot+prepare and commit (the instrumented
/// hook), so BOTH pass the off-lock validation and device history plus wave-local arbitration is
/// the ONLY guard left — the under-lock re-resolve deliberately skips the redundant unique
/// pass on FK-free tables (`InsertPrepareValidation::ReResolveDeviceCovered`). Exactly one
/// must win; sabotaging the wave arbitration makes BOTH land and this test FAIL
/// (verified — the stochastic dup-race test above cannot certify this window).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn constrained_elision_same_snapshot_dup_insert_single_winner() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64);
    e.set_constrained_elision_enabled(true);
    e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    for (seq, chunk) in (2_u64..).zip(0..2_i64) {
        let values: Vec<String> = (chunk * 100..(chunk + 1) * 100)
            .map(|k| format!("({k},{})", k * 10))
            .collect();
        e.execute_text(
            seq,
            &format!("INSERT INTO t (id, v) VALUES {}", values.join(",")),
        )
        .unwrap();
    }
    // Enter elision via a handled wave append.
    for t in 0..20_u64 {
        e.execute_dml_concurrent(
            100 + t,
            &format!("INSERT INTO t (id, v) VALUES ({}, {t})", 5_000 + t),
        )
        .unwrap();
    }
    assert!(e.table_install_elided("t"), "premise: elided");
    for round in 0..20_u64 {
        let key = 7_000 + round;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let outcomes: Vec<Result<(), String>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..2_u64)
                .map(|w| {
                    let e = std::sync::Arc::clone(&e);
                    let barrier = std::sync::Arc::clone(&barrier);
                    s.spawn(move || {
                        e.execute_dml_concurrent_instrumented(
                            1_000 + round * 10 + w,
                            &format!("INSERT INTO t (id, v) VALUES ({key}, {w})"),
                            || {
                                // Both writers are PREPARED (same-snapshot validated) and
                                // not yet committed: the exact window only the ledger covers.
                                barrier.wait();
                            },
                        )
                        .map_err(|err| err.to_string())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let wins = outcomes.iter().filter(|o| o.is_ok()).count();
        assert_eq!(
            wins, 1,
            "round {round}: exactly ONE same-snapshot writer may win key {key} \
             (outcomes: {outcomes:?})"
        );
    }
    // Device truth: each contended key exactly once.
    let rows = e
        .execute_relational_select_text("SELECT id, v FROM t")
        .unwrap()
        .rows;
    assert_eq!(
        rows.len(),
        200 + 20 + 20,
        "seed + waved + one win per round"
    );
}

/// TYPE-COVERAGE track 1 — the CONCURRENT dup race on an ELIDED PK'd table: 8 writers all
/// try to INSERT the SAME key set through `execute_dml_concurrent`. Off-lock prepares may
/// all pass validation (the device probe at their snapshots sees no dup — flushes are
/// wave-tail-deferred), so the UNIQUE-SLOT conflict ledger is the LOAD-BEARING guard:
/// first-committer-wins per slot, later writers get a retryable serialization conflict or
/// the 23505 at re-resolve. End state: every key EXACTLY once, no silent double-append,
/// table still elided (INSERT-only), elisions advancing.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn constrained_elision_concurrent_dup_race_single_winner_per_key() {
    let e = Engine::new_local();
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64);
    e.set_host_install_elision_enabled(true);
    e.set_constrained_elision_enabled(true);
    e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    for (seq, chunk) in (2_u64..).zip(0..2_i64) {
        let values: Vec<String> = (chunk * 100..(chunk + 1) * 100)
            .map(|k| format!("({k},{})", k * 10))
            .collect();
        e.execute_text(
            seq,
            &format!("INSERT INTO t (id, v) VALUES {}", values.join(",")),
        )
        .unwrap();
    }
    let successes: Vec<std::sync::atomic::AtomicU32> = (0..200)
        .map(|_| std::sync::atomic::AtomicU32::new(0))
        .collect();
    let txn = std::sync::atomic::AtomicU64::new(1_000);
    std::thread::scope(|s| {
        for w in 0..8_u64 {
            let e = &e;
            let successes = &successes;
            let txn = &txn;
            s.spawn(move || {
                for k in 0..200_i64 {
                    // Every writer contends for EVERY key; retry serialization conflicts a
                    // few times so the race resolves to a definitive dup answer, never a
                    // silent skip. `w` staggers start points to vary interleavings.
                    let key = (k + w as i64 * 25) % 200;
                    let mut attempts = 0;
                    loop {
                        let t = txn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        match e.execute_dml_concurrent(
                            t,
                            &format!("INSERT INTO t (id, v) VALUES ({}, {w})", 10_000 + key),
                        ) {
                            Ok(()) => {
                                successes[key as usize]
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                break;
                            }
                            Err(err) => {
                                let text = err.to_string();
                                if text.contains("duplicate key") {
                                    break; // definitive: another writer owns the slot
                                }
                                attempts += 1;
                                if attempts > 50 {
                                    panic!("key {key}: unresolved after 50 retries: {text}");
                                }
                                // serialization conflict: retry with a fresh snapshot
                            }
                        }
                    }
                }
            });
        }
    });
    for (k, wins) in successes.iter().enumerate() {
        assert_eq!(
            wins.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "key {k}: exactly ONE writer may win the unique slot"
        );
    }
    assert!(
        e.table_install_elided("t"),
        "INSERT-only dup race must not de-elide the table"
    );
    assert!(e.host_install_elisions() > 0, "non-vacuity: elisions fired");
    // Device truth: every contended key exactly once (no silent double-append survived).
    let rows = e
        .execute_relational_select_text("SELECT id, v FROM t")
        .unwrap()
        .rows;
    assert_eq!(
        rows.len(),
        400,
        "200 seeded + 200 contended keys, each once"
    );
}

/// Wave-BATCHED appends (audit N-1): REAL multi-item waves — 8 writer threads pump
/// single-row INSERTs into `execute_dml_concurrent` concurrently, so the coalescer forms
/// multi-item waves and `flush_appends` aggregates rows per (table, flush) (the sequential
/// sibling test only ever forms 1-item waves). Two tables interleave (the BTreeMap grouping);
/// end state must hold every row exactly once, the tables stay ELIDED through the load
/// (steady state = zero rehydrations), and the elision counter proves the skips.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a4e_multi_writer_waves_batch_appends_and_stay_elided() {
    let e = Engine::new_local();
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64);
    e.set_host_install_elision_enabled(true);
    for (seq, name) in [(1_u64, "ta"), (2, "tb")] {
        e.execute_text(seq, &format!("CREATE TABLE {name} (id INT, v INT)"))
            .unwrap();
    }
    let mut seq = 3_u64;
    for name in ["ta", "tb"] {
        for chunk in 0..2_i64 {
            let values: Vec<String> = (chunk * 100..(chunk + 1) * 100)
                .map(|k| format!("({k},{})", k * 10))
                .collect();
            e.execute_text(
                seq,
                &format!("INSERT INTO {name} (id, v) VALUES {}", values.join(",")),
            )
            .unwrap();
            seq += 1;
        }
    }
    let elisions_before = e.host_install_elisions();
    std::thread::scope(|s| {
        for w in 0..8_u64 {
            let e = &e;
            s.spawn(move || {
                for i in 0..50_u64 {
                    let table = if w % 2 == 0 { "ta" } else { "tb" };
                    let id = 10_000 + w * 1_000 + i;
                    e.execute_dml_concurrent(
                        1_000 + w * 100 + i,
                        &format!("INSERT INTO {table} (id, v) VALUES ({id}, {i})"),
                    )
                    .unwrap();
                }
            });
        }
    });
    let elided_through_load = e.table_install_elided("ta") && e.table_install_elided("tb");
    for name in ["ta", "tb"] {
        let rows = e
            .execute_relational_select_text(&format!("SELECT id, v FROM {name}"))
            .unwrap()
            .rows;
        assert_eq!(rows.len(), 400, "{name}: 200 base + 4 writers x 50 waves");
        let mut ids: Vec<i32> = (0..rows.len())
            .map(|i| match &rows.row(i)[0] {
                SqlValue::Int4(v) => *v,
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 400, "{name}: no duplicates, no losses");
    }
    assert!(
        elided_through_load,
        "both tables must stay ELIDED through the multi-writer load (no rehydration thrash)"
    );
    assert!(
        e.host_install_elisions() - elisions_before >= 350,
        "non-vacuity: the waves must have SKIPPED installs (got {})",
        e.host_install_elisions() - elisions_before
    );
}
