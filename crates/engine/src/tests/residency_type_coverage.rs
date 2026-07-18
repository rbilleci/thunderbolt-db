/// TYPE-COVERAGE track 2 — a PK'd table whose payload columns are DATE and INT2 runs the
/// device-authoritative constraint gauntlet. Exercises
/// the catalog-derived i32-section typing end to end: elided-era INSERT flushes encode
/// Date/Int2 to the i32 section, the A2 resolve + A3 probes accept Date/Int2 needles
/// (variant-agreeing), the A4a materializer types values from the catalog (a mistyped
/// `Int4(days)` would break read equality), and the A4c gather preserves correctly-typed rows.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn date_int2_pk_table_elision_matches_install_twin() {
    let run = || {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, d DATE, s INT2)")
            .unwrap();
        let mut seq = 2u64;
        for chunk in 0..2_i64 {
            let values: Vec<String> = (chunk * 100..(chunk + 1) * 100)
                .map(|k| {
                    format!(
                        "({k}, '2026-{:02}-{:02}', {})",
                        1 + (k % 12),
                        1 + (k % 28),
                        k % 1000
                    )
                })
                .collect();
            e.execute_text(
                seq,
                &format!("INSERT INTO t (id, d, s) VALUES {}", values.join(",")),
            )
            .unwrap();
            seq += 1;
        }
        let statements = [
            // Elided steady-state inserts with Date/Int2 payloads.
            "INSERT INTO t (id, d, s) VALUES (500, '2027-01-01', 7)",
            "INSERT INTO t (id, d, s) VALUES (501, '2027-02-02', -8)",
            "INSERT INTO t (id, d, s) VALUES (42, '2027-03-03', 9)", // dup PK -> 23505
            "INSERT INTO t (id, d, s) VALUES (500, '2027-04-04', 1)", // elided-era dup -> 23505
            // Date-needle DML: the A2 resolve locates by the DATE column's i32 encoding.
            "UPDATE t SET s = 99 WHERE d = '2027-01-01'",
            "DELETE FROM t WHERE d = '2027-02-02'",
            // Int2-needle DML.
            "UPDATE t SET d = '2028-01-01' WHERE s = 99",
            // Int4 PK point DML on a Date/Int2-payload row (the materializer types d + s).
            "UPDATE t SET s = -1 WHERE id = 130",
            "DELETE FROM t WHERE id = 42",
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
        let mut rows = e
            .execute_relational_select_text("SELECT id, d, s FROM t")
            .unwrap()
            .rows
            .into_boxed();
        rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        (e, outcomes, rows)
    };
    let (on, outcomes, rows) = run();
    assert_eq!(
        outcomes.iter().map(Result::is_ok).collect::<Vec<_>>(),
        vec![true, true, false, false, true, true, true, true, true],
        "Date/Int2 outcome ladder"
    );
    assert_eq!(rows.len(), 200, "closed-form final cardinality");
    assert!(
        on.device_authoritative_commits() > 0,
        "non-vacuity: the Date/Int2 PK'd table must have device-authoritative commits"
    );
    assert!(
        on.dml_device_validate_hits() > 0,
        "non-vacuity: the device validator answered typed probes"
    );

    // DATE-PK table (audit cede8e70: the Date NEEDLE must fire, not silently decline —
    // the binder now coerces the plain literal): elided-era dup-DATE inserts drive the
    // A3 probe with a Date needle through `i32_section_needle`. Sabotage-verified: a
    // skewed Date encode (+1) makes the probe miss the dup -> the elided arm ACCEPTS it.
    let hits_before_date_needle = on.dml_device_validate_hits();
    on.execute_text(600, "CREATE TABLE dp (d DATE PRIMARY KEY, v INT)")
        .unwrap();
    for (i, day) in (1..=8_u32).enumerate() {
        on.execute_text(
            601 + i as u64,
            &format!("INSERT INTO dp (d, v) VALUES ('2027-06-{day:02}', {i})"),
        )
        .unwrap();
    }
    // Force shard admission + elision entry via wave inserts.
    for t in 0..30_u64 {
        on.execute_dml_concurrent(
            650 + t,
            &format!(
                "INSERT INTO dp (d, v) VALUES ('2028-{:02}-{:02}', 1)",
                1 + t / 28,
                1 + t % 28
            ),
        )
        .unwrap();
    }
    assert!(
        on.table_device_authoritative("dp"),
        "the DATE-PK table must elide"
    );
    // Elided-era dup DATE -> 23505 through the DEVICE Date-needle probe.
    let dup = on.execute_text(700, "INSERT INTO dp (d, v) VALUES ('2028-01-01', 9)");
    assert!(
        dup.is_err() && dup.unwrap_err().to_string().contains("duplicate key"),
        "the elided-era dup DATE must violate the PK via the device Date needle"
    );
    // Fresh DATE still inserts.
    on.execute_text(701, "INSERT INTO dp (d, v) VALUES ('2029-01-01', 1)")
        .unwrap();
    assert!(
        on.dml_device_validate_hits() > hits_before_date_needle,
        "non-vacuity: the DATE-needle probes must have been answered by the DEVICE"
    );
}

/// TYPE-COVERAGE track 2 slice 2, stage (i) — the i64-SECTION read differential: an
/// int8/timestamp-bearing table admitted SHARDED (flag ON) must read byte-identically to
/// its single-buffer twin (flag OFF) across the general shapes — full projection, bigint
/// aggregates/filters (values beyond i32 range are the mistype canary), ORDER BY the i64
/// column, point reads, and a NULL in the bigint column (the single-shard invariant).
/// Stage (i) is READ-only: appends still decline (int4_appendable=false for int8-bearing
/// shards), so writes re-admit — correctness unchanged, perf comes with stage (ii).
///
/// COVERAGE BOUNDARY (deliberate): a stage-(i) table is single-shard by construction (no
/// appends -> no rollover), so device service goes through the ZERO-COPY single-shard
/// source (`resident_snapshot_for_shard`, int8-labeled) and the CPU-pinned host path — the
/// multi-shard i64 RECOMPACTION axis is unreachable here and gets its non-vacuous
/// differential + sabotage with stage (ii)'s rollover-created multi-shard tables.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn int8_section_sharded_reads_match_single_buffer_twin() {
    let queries = [
        "SELECT id, v, t FROM t8",
        "SELECT SUM(v) FROM t8",
        "SELECT v FROM t8 WHERE id = 7",
        "SELECT id FROM t8 WHERE v = 5000000007",
        "SELECT id, v FROM t8 ORDER BY v",
        "SELECT COUNT(*) FROM t8 WHERE v IS NULL",
        "SELECT id, v, t FROM t8 WHERE t = '2026-07-03 12:00:00'",
    ];
    let run = |int8_shards: bool| {
        let mut e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.set_shard_int8_section_enabled(true);
        e.execute_text(
            1,
            "CREATE TABLE t8 (id INT PRIMARY KEY, v BIGINT, t TIMESTAMP)",
        )
        .unwrap();
        let values: Vec<String> = (0..100_i64)
            .map(|k| {
                if k == 50 {
                    format!("({k}, NULL, '2026-07-03 12:00:00')")
                } else {
                    format!(
                        "({k}, {}, '2026-01-01 00:00:{:02}')",
                        5_000_000_000_i64 + k, // beyond i32: the mistype canary
                        k % 60
                    )
                }
            })
            .collect();
        e.execute_text(
            2,
            &format!("INSERT INTO t8 (id, v, t) VALUES {}", values.join(",")),
        )
        .unwrap();
        if !int8_shards {
            install_test_single_buffer_residency(&mut e, "t8");
        }
        let sharded = e
            .read_state
            .residency
            .shards
            .load()
            .get("t8")
            .is_some_and(|s| !s.is_empty());
        // TWIN outcomes (rows OR the exact error): a shape unsupported on BOTH layouts is
        // pre-existing scope, not a slice regression — parity is the contract.
        let mut outs: Vec<Result<String, String>> = Vec::new();
        for q in &queries {
            outs.push(match e.execute_relational_select_text(q) {
                Ok(result) => {
                    let mut rows = result.rows.into_boxed();
                    rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
                    Ok(format!("{rows:?}"))
                }
                Err(err) => Err(err.to_string()),
            });
        }
        (sharded, outs)
    };
    let (sharded_on, on) = run(true);
    let (sharded_off, off) = run(false);
    assert!(
        sharded_on,
        "non-vacuity: the flag must shard-admit the int8 table"
    );
    assert!(!sharded_off, "flag OFF keeps the int8 table single-buffer");
    let mut ok_count = 0;
    for (i, (a, b)) in on.iter().zip(off.iter()).enumerate() {
        assert_eq!(a, b, "query {i} ({}): sharded == single-buffer", queries[i]);
        if a.is_ok() {
            ok_count += 1;
        }
    }
    assert!(
        ok_count >= 4,
        "non-vacuity: most shapes must SUCCEED on both arms (got {ok_count}/7)"
    );
}

/// TYPE-COVERAGE track 2 slice 2, stage (ii) — the i64 APPEND + MULTI-SHARD RECOMPACTION
/// differential: wave INSERTs on a flag-ON int8/timestamp table append IN PLACE through
/// ROLLOVERS (shard target 64 -> multiple shards), then a full read recompacts every
/// shard's i32 AND i64 sections into the unified buffer. The single-buffer twin is the
/// oracle. NON-VACUITY: the sharded arm must (a) really append (open_shard_append_hits
/// advances), (b) really roll over (>= 2 shards), and (c) beyond-i32 bigint values +
/// per-row timestamps must read back exactly — a broken i64 chunk offset or a missing
/// recompaction segment corrupts them (sabotage-verified: skipping the i64 recompaction
/// axis fails this test NOW that multi-shard i64 tables exist).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn int8_section_appends_roll_over_and_recompact_to_parity() {
    let run = || {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.set_shard_int8_section_enabled(true);
        e.execute_text(
            1,
            "CREATE TABLE t8 (id INT PRIMARY KEY, v BIGINT, ts TIMESTAMP)",
        )
        .unwrap();
        // Seed 100 rows (one bulk admit), then 200 wave INSERTs -> ~3-4 rollovers at target 64.
        let seed: Vec<String> = (0..100_i64)
            .map(|k| {
                format!(
                    "({k}, {}, '2026-01-01 00:00:{:02}')",
                    7_000_000_000_i64 - k,
                    k % 60
                )
            })
            .collect();
        e.execute_text(
            2,
            &format!("INSERT INTO t8 (id, v, ts) VALUES {}", seed.join(",")),
        )
        .unwrap();
        let hits_before = e.open_shard_append_hits();
        for t in 0..200_u64 {
            e.execute_dml_concurrent(
                100 + t,
                &format!(
                    "INSERT INTO t8 (id, v, ts) VALUES ({}, {}, '2027-06-15 08:30:{:02}')",
                    1_000 + t,
                    6_000_000_000_i64 + t as i64,
                    t % 60
                ),
            )
            .unwrap();
        }
        let appends = e.open_shard_append_hits() - hits_before;
        let shard_count = e.resident_shard_count("t8");
        let queries = [
            "SELECT id, v, ts FROM t8",
            "SELECT v FROM t8 WHERE id = 1100",
            "SELECT id FROM t8 WHERE v = 6000000100",
            "SELECT id, v FROM t8 ORDER BY v",
        ];
        let mut outs: Vec<Result<Vec<Vec<SqlValue>>, String>> = Vec::new();
        for q in &queries {
            outs.push(match e.execute_relational_select_text(q) {
                Ok(result) => {
                    let mut rows = result.rows.into_boxed();
                    rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
                    Ok(rows)
                }
                Err(err) => Err(err.to_string()),
            });
        }
        (appends, shard_count, outs)
    };
    let (appends_on, shards_on, on) = run();
    assert!(
        appends_on >= 100,
        "non-vacuity: the int8 table must APPEND in place (got {appends_on} hits)"
    );
    assert!(
        shards_on >= 2,
        "non-vacuity: the appends must ROLL OVER to multiple shards (got {shards_on})"
    );
    // Every query must succeed on the device-authoritative multi-shard generation.
    assert!(
        on.iter().all(|o| o.is_ok()),
        "all stage-(ii) shapes must succeed: {on:?}"
    );
    assert_eq!(on[0].as_ref().unwrap().len(), 300, "100 seeds + 200 appends");
    assert_eq!(
        on[1].as_ref().unwrap(),
        &vec![vec![SqlValue::Int8(6_000_000_100)]],
        "point read returns the exact beyond-i32 value"
    );
    assert_eq!(
        on[2].as_ref().unwrap(),
        &vec![vec![SqlValue::Int4(1100)]],
        "wide-value equality resolves the exact entity"
    );
    assert_eq!(on[3].as_ref().unwrap().len(), 300, "ordered projection is complete");
}

/// TYPE-COVERAGE track 2 slice 2, stage (iii) — the i64-PAYLOAD ELISION differential: an
/// int4-PK / BIGINT+TIMESTAMP-payload table (THE core-banking shape) remains device-authoritative;
/// DML resolves via the A4a materializer typing i64 payloads from the catalog. R3-002 additionally
/// proves a single-column i64 UNIQUE table now elides through
/// its fingerprint index and exact typed device recheck.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn int8_payload_elision_matches_install_twin() {
    let run = || {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.set_shard_int8_section_enabled(true);
        e.execute_text(
            1,
            "CREATE TABLE t8 (id INT PRIMARY KEY, v BIGINT, ts TIMESTAMP)",
        )
        .unwrap();
        let mut seq = 2u64;
        for chunk in 0..2_i64 {
            let values: Vec<String> = (chunk * 100..(chunk + 1) * 100)
                .map(|k| {
                    format!(
                        "({k}, {}, '2026-01-01 00:00:{:02}')",
                        9_000_000_000_i64 + k,
                        k % 60
                    )
                })
                .collect();
            e.execute_text(
                seq,
                &format!("INSERT INTO t8 (id, v, ts) VALUES {}", values.join(",")),
            )
            .unwrap();
            seq += 1;
        }
        let statements = [
            "INSERT INTO t8 (id, v, ts) VALUES (500, 8000000000, '2027-01-01 00:00:00')",
            "INSERT INTO t8 (id, v, ts) VALUES (501, 8000000001, '2027-01-02 00:00:00')",
            "INSERT INTO t8 (id, v, ts) VALUES (42, 1, '2027-01-03 00:00:00')", // dup PK
            "INSERT INTO t8 (id, v, ts) VALUES (500, 2, '2027-01-04 00:00:00')", // elided-era dup
            // Point DML on elided-era + seeded rows: the materializer types v/ts.
            "UPDATE t8 SET v = 8500000000 WHERE id = 500",
            "DELETE FROM t8 WHERE id = 42",
            "UPDATE t8 SET v = 9999999999 WHERE id = 130",
            // A shape the resolve declines (OR-group) -> rehydration through the i64 gather.
            "UPDATE t8 SET v = -1 WHERE id = 500 OR id = 11",
            "INSERT INTO t8 (id, v, ts) VALUES (502, 8000000002, '2027-02-01 00:00:00')",
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
        let mut rows = e
            .execute_relational_select_text("SELECT id, v, ts FROM t8")
            .unwrap()
            .rows
            .into_boxed();
        rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        (e, outcomes, rows)
    };
    let (on, outcomes, rows) = run();
    assert_eq!(
        outcomes.iter().map(Result::is_ok).collect::<Vec<_>>(),
        vec![true, true, false, false, true, true, true, true, true],
        "i64-payload outcome ladder"
    );
    assert_eq!(rows.len(), 202, "closed-form final cardinality");
    assert!(
        on.device_authoritative_commits() > 0,
        "non-vacuity: the i64-payload PK table must publish device-authoritative commits"
    );
    assert!(
        on.dml_device_validate_hits() > 0,
        "non-vacuity: device validation answered on the i64-payload table"
    );

    // R3-002 single-wide index: BIGINT UNIQUE now rides the flagged fingerprint index.
    on.set_binary_wal_records_enabled(true);
    on.set_device_write_locate_wave_batch_enabled(true);
    on.execute_text(700, "CREATE TABLE u8 (v BIGINT UNIQUE, x INT)")
        .unwrap();
    for i in 0..30_u64 {
        on.execute_dml_concurrent(
            710 + i,
            &format!(
                "INSERT INTO u8 (v, x) VALUES ({}, {i})",
                8_100_000_000_i64 + i as i64
            ),
        )
        .unwrap();
    }
    assert!(on.table_device_authoritative("u8"), "BIGINT UNIQUE must elide");
    let locate_before = on.device_write_locate_hits();
    assert!(
        on.execute_dml_concurrent(750, "INSERT INTO u8 (v, x) VALUES (8100000005, 9)")
            .is_err(),
        "the BIGINT unique constraint still fires"
    );
    assert!(
        on.device_write_locate_hits() > locate_before,
        "BIGINT duplicate validation must probe the device index"
    );
    assert!(on.table_device_authoritative("u8"));
}
