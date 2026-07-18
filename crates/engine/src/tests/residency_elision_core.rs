/// RETIREMENT A2 — the mandatory device resolve ladder. Covers: point DELETE/UPDATE (the locate's unique-key
/// shape), DML after an SV5 UPDATE (the appended version's A1 identity must resolve the SAME
/// key), delete-by-tombstoned-value (the PHYSICAL locate hits the tombstoned slot; the keyed
/// fetch at visibility must yield no match), duplicate values, OR-group and range predicates,
/// and post-rollover appends. NON-VACUITY: `dml_device_resolve_hits` must ADVANCE for point shapes.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a2_device_resolve_serves_point_or_and_range_ladder() {
    let scenarios: Vec<Vec<String>> = vec![
        vec![
            "DELETE FROM t WHERE id = 40".into(),
            "UPDATE t SET v = 999 WHERE id = 40".into(),
        ],
        vec![
            "UPDATE t SET v = 777 WHERE id = 50".into(), // SV5 append: new version, same identity
            "DELETE FROM t WHERE id = 50".into(),        // resolve THROUGH the appended version
        ],
        vec![
            "DELETE FROM t WHERE id = 60".into(),
            "DELETE FROM t WHERE id = 60".into(), // second delete: tombstoned -> no match
        ],
        vec!["DELETE FROM t WHERE v = 100".into()], // v = (id%37)*10 -> duplicates -> locate declines
        vec!["UPDATE t SET v = -1 WHERE id = 70 OR id = 71".into()], // OR-group -> fallback
        vec!["DELETE FROM t WHERE id < 5".into()],  // range -> fallback
    ];
    let build = |scenario: usize, statements: &[String]| -> Vec<Vec<SqlValue>> {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
        let mut seq = 2u64;
        for chunk in 0..2_i64 {
            let values: Vec<String> = (chunk * 100..(chunk + 1) * 100)
                .map(|k| format!("({k},{})", (k % 37) * 10))
                .collect();
            e.execute_text(
                seq,
                &format!("INSERT INTO t (id, v) VALUES {}", values.join(",")),
            )
            .unwrap();
            seq += 1;
        }
        let hits_before = e.dml_device_resolve_hits();
        for sql in statements {
            e.execute_text(seq, sql).unwrap();
            seq += 1;
        }
        // Non-vacuity only for the SINGLE-Eq point scenarios (0-2); the dup/OR/range
        // scenarios (3-5) are DESIGNED to decline to the fallback chain.
        if scenario <= 2 {
            assert!(
                e.dml_device_resolve_hits() > hits_before,
                "scenario {scenario}: non-vacuity — the device resolve must have served a point statement"
            );
        }
        // Plain projection (no ORDER BY): a VERSIONED sharded table clean-errors on reshaping
        // clauses (the documented SV3b/SV6 guard); identical lineages give identical row order.
        e.execute_relational_select_text("SELECT id, v FROM t")
            .unwrap()
            .rows
            .into_boxed()
    };
    for (i, statements) in scenarios.iter().enumerate() {
        let rows = build(i, statements);
        let expected_len = [199, 199, 199, 194, 200, 195][i];
        assert_eq!(rows.len(), expected_len, "scenario {i}: terminal row count");
        match i {
            0 => assert!(!rows
                .iter()
                .any(|row| row.first() == Some(&SqlValue::Int4(40)))),
            1 => assert!(!rows
                .iter()
                .any(|row| row.first() == Some(&SqlValue::Int4(50)))),
            2 => assert!(!rows
                .iter()
                .any(|row| row.first() == Some(&SqlValue::Int4(60)))),
            3 => assert!(!rows
                .iter()
                .any(|row| row.get(1) == Some(&SqlValue::Int4(100)))),
            4 => assert_eq!(
                rows.iter()
                    .filter(|row| row.get(1) == Some(&SqlValue::Int4(-1)))
                    .count(),
                2
            ),
            5 => assert!(!rows.iter().any(|row| {
                row.first()
                    .is_some_and(|value| matches!(value, SqlValue::Int4(id) if *id < 5))
            })),
            _ => unreachable!(),
        }
    }
}

/// RETIREMENT A4e — the device-authority lifecycle: admission, steady-state INSERTs,
/// point DELETE/UPDATE, an OR-group UPDATE, DDL repair, and post-DDL DML all preserve the exact
/// relational result. DDL deliberately crosses the RETIRE-002 repair boundary; ordinary DML does not.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a4e_elision_lifecycle_matches_install_twin() {
    let run = || {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        // Constraint-FREE (audit B1: only such tables may elide — constraint validators
        // read the host store); the UNIQUE never-elides gate is asserted separately below.
        e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
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
        let statements = [
            "INSERT INTO t (id, v) VALUES (500, 5000)", // elided steady-state insert
            "INSERT INTO t (id, v) VALUES (501, 5010)",
            "UPDATE t SET v = 999 WHERE id = 130", // device resolve + materializer
            "DELETE FROM t WHERE id = 42",         // device resolve + materializer
            "INSERT INTO t (id, v) VALUES (130, 1)", // a dup id row (no constraint): both twins keep BOTH
            // ADVERSARIAL: DML on ELIDED-ERA rows — they exist ONLY on the device; a
            // stale-store fetch would silently no-op them.
            "UPDATE t SET v = 7 WHERE id = 500", // materializer resolves an elided-era row
            "DELETE FROM t WHERE id = 501",      // ... and deletes one
            // OR-group incl an elided-era row: the stale value-index MISSES id=500 -> the
            // shape early-exit must REHYDRATE first.
            "UPDATE t SET v = -5 WHERE id = 500 OR id = 11",
            // Audit B1-DDL vector: DDL on an ELIDED table must rehydrate FIRST (the
            // execute_text non-DML seam) — its validators read the host store.
            "ALTER TABLE ONLY t ADD CONSTRAINT t_v_floor CHECK (v > -1000)",
            "INSERT INTO t (id, v) VALUES (502, 5020)", // post-rehydration: normal installs
        ];
        let mut outcomes: Vec<Result<(), String>> = Vec::new();
        for sql in &statements {
            outcomes.push(
                e.execute_text(seq, sql)
                    .map(|_| ())
                    .map_err(|err| err.to_string()),
            );
            seq += 1;
        }
        let rows = e
            .execute_relational_select_text("SELECT id, v FROM t")
            .unwrap()
            .rows
            .into_boxed();
        (e, outcomes, rows)
    };
    let (on, outcomes, rows) = run();
    assert!(outcomes.iter().all(Result::is_ok), "the lifecycle must commit every valid step: {outcomes:?}");
    assert_eq!(rows.len(), 202, "closed-form row count after inserts and deletes");
    assert!(
        on.device_authoritative_commits() > 0,
        "non-vacuity: device-authoritative commits must have fired"
    );
    assert!(
        on.table_device_authoritative("t"),
        "ordinary DML after the DDL repair must restore device authority"
    );
    // Audit SF4 gate: DROP purges the elided flag — a recreated same-name table must INSTALL.
    on.execute_text(430, "CREATE TABLE d (id INT, v INT)")
        .unwrap();
    for i in 0..3_i64 {
        on.execute_text(
            431 + i as u64,
            &format!("INSERT INTO d (id, v) VALUES ({i}, {i})"),
        )
        .unwrap();
    }
    on.execute_text(440, "DROP TABLE d").unwrap();
    assert!(
        !on.table_device_authoritative("d"),
        "DROP must purge the elided flag (a recreated table is not device-authoritative)"
    );
}

/// RETIREMENT A4e — the CONCURRENT form: single-row INSERT waves (the SLO workload) on an
/// ELIDED table through `execute_dml_concurrent`, racing a hammering reader. The wave arm's
/// native hooks must ENTER elision after the first handled append and stay there through
/// rollovers (steady state = ZERO rehydrations — a de-elision would mean the incremental path
/// silently degraded); the closed-form result contains every submitted key.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a4e_concurrent_insert_waves_elide_and_match_twin() {
    let run = || {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
        // TWO serialized chunks: the SECOND re-admits the table SHARDED (target 64) — the
        // wave appends then take the rollover-capable shard path. A dense SINGLE-BUFFER
        // table's append always declines (no rollover), so such a table never ENTERS elision
        // (safety by construction) — and this test would be vacuous.
        for chunk in 0..2_i64 {
            let values: Vec<String> = (chunk * 100..(chunk + 1) * 100)
                .map(|k| format!("({k},{})", k * 10))
                .collect();
            e.execute_text(
                2 + chunk as u64,
                &format!("INSERT INTO t (id, v) VALUES {}", values.join(",")),
            )
            .unwrap();
        }
        // NO concurrent reader here: a reader hammering the wave path trips the PRE-EXISTING
        // ADR-013 D4 publication tear ("resident shard 0 has no retained device memory" —
        // shard metadata paired with a separate device-memory map get mid-rollover), with
        // elision ON *and* OFF — live evidence for the generation-atomic publication gate
        // (pre2), not an elision defect. This test pins the DATA PLANE.
        for t in 0..300_u64 {
            e.execute_dml_concurrent(
                100 + t,
                &format!("INSERT INTO t (id, v) VALUES ({}, {})", 1000 + t, t),
            )
            .unwrap();
        }
        // Sample BEFORE the read: the full-projection SELECT below is a HOST-path shape, so
        // the A4e read-side ladder legitimately rehydrates + de-elides to serve it.
        let elided_through_waves = e.table_device_authoritative("t");
        let mut rows = e
            .execute_relational_select_text("SELECT id, v FROM t")
            .unwrap()
            .rows
            .into_boxed();
        rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        (e, rows, elided_through_waves)
    };
    let (on, on_rows, on_elided_through_waves) = run();
    assert_eq!(on_rows.len(), 500, "200 + 300 waves");
    assert!(
        on.device_authoritative_commits() >= 250,
        "non-vacuity: the waves must have SKIPPED installs (got {})",
        on.device_authoritative_commits()
    );
    assert!(
        on_elided_through_waves,
        "steady state must STAY elided through ~5 rollovers (a de-elision = degradation)"
    );
}

/// TYPE-COVERAGE track 1 — constrained device authority runs a PK'd table through the full
/// constraint gauntlet. Every accepted/rejected outcome is checked directly:
///   - dup of a seeded key and of a device-authoritative-era key,
///   - within-batch dup VALUES,
///   - dup-by-UPDATE, self-key UPDATE (exclude_keys), delete-then-reinsert,
///   - PK NOT NULL (23502 before 23505),
///   - post-UPDATE churn probe: the SV5 append creates physical key twins and the probe ladder
///     must still resolve the exact visible device version.
///
/// NON-VACUITY: the table is still device-authoritative after the INSERT-only prefix and the
/// device validator answered probes.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn device_authority_pk_constraint_ladder_matches_closed_form() {
    let run = || {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
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
        // INSERT-only prefix: the elided steady state (checkpointed below, pre-churn).
        let insert_prefix = [
            "INSERT INTO t (id, v) VALUES (500, 5000)",
            "INSERT INTO t (id, v) VALUES (501, 5010)",
            "INSERT INTO t (id, v) VALUES (42, 1)", // dup of a SEEDED key -> 23505
            "INSERT INTO t (id, v) VALUES (500, 1)", // dup of an ELIDED-ERA key -> 23505 (B1)
            "INSERT INTO t (id, v) VALUES (600, 1), (600, 2)", // within-batch dup -> 23505
            "INSERT INTO t (id, v) VALUES (NULL, 1)", // PK NOT NULL -> 23502 (before unique)
        ];
        let mut outcomes: Vec<Result<(), String>> = Vec::new();
        for sql in &insert_prefix {
            outcomes.push(
                e.execute_text(seq, sql)
                    .map(|_| ())
                    .map_err(|err| err.to_string()),
            );
            seq += 1;
        }
        let elided_after_insert_prefix = e.table_device_authoritative("t");
        // Churn + post-churn probes exercise exact version-aware device resolution.
        let churn_ladder = [
            "UPDATE t SET v = 999 WHERE id = 130", // SV5 append dups the open shard's id col
            "INSERT INTO t (id, v) VALUES (130, 1)", // post-churn dup probe -> 23505
            "UPDATE t SET id = 42 WHERE id = 131", // dup-by-UPDATE -> 23505
            "UPDATE t SET id = 131 WHERE id = 131", // self-key UPDATE: excluded -> ok
            "DELETE FROM t WHERE id = 42",
            "INSERT INTO t (id, v) VALUES (42, 77)", // deleted key is reusable -> ok
        ];
        for sql in &churn_ladder {
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
        (e, outcomes, rows, elided_after_insert_prefix)
    };
    let (on, outcomes, rows, on_elided_mid) = run();
    let accepted = outcomes.iter().map(Result::is_ok).collect::<Vec<_>>();
    assert_eq!(
        accepted,
        vec![true, true, false, false, false, false, true, false, false, true, true, true],
        "constraint outcome ladder"
    );
    assert_eq!(rows.len(), 202, "closed-form final cardinality");
    assert!(
        on_elided_mid,
        "the PK'd table must be ELIDED through the INSERT-only prefix (the flag's purpose)"
    );
    assert!(
        on.device_authoritative_commits() > 0,
        "non-vacuity: device-authoritative commits fired on the PK'd table"
    );
    assert!(
        on.dml_device_validate_hits() > 0,
        "non-vacuity: the DEVICE validator must have answered probes"
    );
}

/// AUDIT f80f2350 FINDING A regression: a single-entry constraint DDL (`CREATE UNIQUE
/// INDEX`) whose apply-time row validator reads an ELIDED table must not self-deadlock on
/// the commit lock. The pre-fix wedge: the off-lock execute_text sweep de-elides, a racing
/// INSERT wave RE-ELIDES during the DDL's commit window, the under-lock
/// `visible_relational_rows` seam then called `rehydrate_elided_serialized` whose re-lock
/// branch blocked on the mutex this thread held — permanently wedging the commit path. The
/// fix flags the whole apply (and the whole wave) as an internal read so the seam takes its
/// direct branch. This hammers the interleaving and fails by TIMEOUT if any iteration
/// wedges.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn constrained_elision_ddl_race_does_not_wedge_the_commit_path() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64);
    // Defaults: elision ON (the A5 flip). The wedge repro does NOT need constrained
    // elision — the DDL's validator read on an A5-elided table is enough.
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
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
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let txn = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(10_000));
    // Writer pressure: keeps the table entering elision between the DDL's off-lock sweep
    // and its commit window (the race the wedge needs).
    let writers: Vec<_> = (0..4_u64)
        .map(|w| {
            let e = std::sync::Arc::clone(&e);
            let stop = std::sync::Arc::clone(&stop);
            let txn = std::sync::Arc::clone(&txn);
            std::thread::spawn(move || {
                let mut i = 0_u64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let t = txn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let id = 1_000_000 + w * 1_000_000 + i;
                    // Ignore result: unique-index windows can reject dup-free inserts only
                    // via serialization retries; correctness is asserted at the end.
                    let _ = e.execute_dml_concurrent(
                        t,
                        &format!("INSERT INTO t (id, v) VALUES ({id}, 1)"),
                    );
                    i += 1;
                }
            })
        })
        .collect();
    let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    {
        let e = std::sync::Arc::clone(&e);
        let txn = std::sync::Arc::clone(&txn);
        std::thread::spawn(move || {
            for round in 0..40_u64 {
                let t1 = txn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if let Err(err) = e.execute_text(t1, "CREATE UNIQUE INDEX t_id_uq ON t (id)") {
                    let _ = done_tx.send(Err(format!("round {round} create: {err}")));
                    return;
                }
                let t2 = txn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if let Err(err) = e.execute_text(t2, "DROP INDEX t_id_uq") {
                    let _ = done_tx.send(Err(format!("round {round} drop: {err}")));
                    return;
                }
            }
            let _ = done_tx.send(Ok(()));
        });
    }
    // The wedge detector: pre-fix, an iteration deadlocks and the DDL thread never reports.
    // DISTINGUISH deadlock from commit-mutex STARVATION (the std Mutex is unfair and the
    // writers re-acquire in a tight loop): stop the writers after the first window — a
    // starved DDL thread then finishes; a DEADLOCKED one stays stuck forever.
    let outcome = match done_rx.recv_timeout(std::time::Duration::from_secs(45)) {
        Ok(outcome) => Ok(outcome),
        Err(_) => {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            done_rx
                .recv_timeout(std::time::Duration::from_secs(60))
                .map_err(|_| ())
        }
    };
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    match outcome {
        Ok(Ok(())) => {}
        Ok(Err(err)) => panic!("DDL round failed: {err}"),
        Err(()) => panic!(
            "WEDGED: the DDL/wave race deadlocked the commit path (FINDING A regression) — \
             still stuck with all writers stopped"
        ),
    }
    for w in writers {
        w.join().unwrap();
    }
    // Post-race sanity: the table still reads consistently.
    let rows = e
        .execute_relational_select_text("SELECT id, v FROM t")
        .unwrap()
        .rows;
    assert!(rows.len() >= 200, "seed rows survive the race");
}

/// AUDIT f80f2350 FINDING B regression: an OFF-LOCK concurrent INSERT prepare whose
/// validator ladder REHYDRATES an elided PK'd table (device-probe decline via a dup-churned
/// open shard) must serialize the store mutation behind the commit lock — the pre-fix direct
/// call raced `with_table_mut`'s clone-mutate-publish against the sequencer (lost/torn
/// generation publish). Concurrent writer pressure keeps the sequencer busy while the
/// decline-bearing INSERTs prepare off-lock; end state must hold every row exactly once.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn constrained_elision_offlock_rehydrate_races_sequencer_safely() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64);
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
    // Elide via waves, then CHURN the open shard: the SV5 update-append duplicates key 130
    // in the id column -> the cached index entry DECLINES (monotone) -> subsequent unique
    // probes must rehydrate.
    for t in 0..50_u64 {
        e.execute_dml_concurrent(
            100 + t,
            &format!("INSERT INTO t (id, v) VALUES ({}, {t})", 5_000 + t),
        )
        .unwrap();
    }
    assert!(
        e.table_device_authoritative("t"),
        "premise: the PK'd table is elided before the churn"
    );
    e.execute_text(300, "UPDATE t SET v = 999 WHERE id = 130")
        .unwrap();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let txn = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(20_000));
    let writers: Vec<_> = (0..6_u64)
        .map(|w| {
            let e = std::sync::Arc::clone(&e);
            let stop = std::sync::Arc::clone(&stop);
            let txn = std::sync::Arc::clone(&txn);
            std::thread::spawn(move || {
                let mut i = 0_u64;
                let mut ok = 0_u64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) && ok < 200 {
                    let t = txn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let id = 2_000_000 + w * 1_000_000 + i;
                    if e.execute_dml_concurrent(
                        t,
                        &format!("INSERT INTO t (id, v) VALUES ({id}, 1)"),
                    )
                    .is_ok()
                    {
                        ok += 1;
                    }
                    i += 1;
                }
                ok
            })
        })
        .collect();
    let mut total_ok = 0_u64;
    for w in writers {
        total_ok += w.join().unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    // Every successful INSERT must be readable exactly once — a torn generation publish
    // (the pre-fix race) loses rows or duplicates value-index entries.
    let rows = e
        .execute_relational_select_text("SELECT id, v FROM t")
        .unwrap()
        .rows;
    let expected = 200 + 50 + total_ok as usize;
    assert_eq!(
        rows.len(),
        expected,
        "seed + waved + raced inserts, each exactly once"
    );
    let mut ids: Vec<i32> = (0..rows.len())
        .map(|i| match &rows.row(i)[0] {
            SqlValue::Int4(v) => *v,
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), expected, "no duplicate ids survived the race");
}
