/// M1 (charter-pure device locate) — the DEVICE-vs-HOST-PROBE differential: an elided PK'd
/// table runs the constraint + DML gauntlet with the DEVICE write-locate ON vs OFF (the host
/// PK-hash-probe oracle). Every outcome (incl violation text) + final read must match — the
/// locate feeds BOTH the A2 resolve (point DML) and the A3 validators (dup checks). Includes
/// an UPDATE that appends a new version (SV5) so a key lands in TWO shards — the kernel's
/// multi-hit emission is exercised (the host path returns both hits too). NON-VACUITY: the
/// device arm's `device_write_locate_hits` advances (the kernel really fired).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn device_write_locate_matches_host_probe_twin() {
    let run = |device: bool| {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.set_device_write_locate_enabled(device);
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
        let ladder = [
            "INSERT INTO t (id, v) VALUES (500, 5000)",
            "INSERT INTO t (id, v) VALUES (42, 1)", // dup PK -> 23505 (A3 via locate)
            "UPDATE t SET v = 999 WHERE id = 130",  // A2 resolve + SV5 append (2-shard key)
            "UPDATE t SET v = 7 WHERE id = 130",    // now id=130 is in 2 shards -> multi-hit
            "DELETE FROM t WHERE id = 42",          // A2 resolve
            "INSERT INTO t (id, v) VALUES (42, 77)", // reuse the deleted key -> ok
            "UPDATE t SET v = -1 WHERE id = 500",   // elided-era row
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
    assert_eq!(
        on_out, off_out,
        "device locate outcome ladder == host-probe oracle"
    );
    assert_eq!(
        on_rows, off_rows,
        "device locate reads == host-probe oracle"
    );
    assert!(
        on.device_write_locate_hits() > 0,
        "non-vacuity: the DEVICE write-locate kernel must have FIRED (got {})",
        on.device_write_locate_hits()
    );
    assert_eq!(
        off.device_write_locate_hits(),
        0,
        "flag OFF never touches the device locate"
    );
}

/// F3/U4 (audit CRITICAL 1/2 regression): with the DUP-TOLERANT device index, an UPDATE whose
/// old+new versions land in the SAME shard (a recently-inserted-then-updated key) must NOT
/// break the single-key device locate. Before the write-locate kernel was made
/// advance-past-every-match, it emitted only the FIRST match per shard = the dead OLD twin, so:
/// (1) a point read `WHERE pk = k` returned EMPTY for a live updated key, and (2) an INSERT of k
/// bypassed uniqueness (saw only the dead twin -> "no dup"). Both must now resolve correctly
/// (device ON == host oracle). Sabotage: reverting the write-locate FOUND->advance flip fails
/// this (the point read goes empty / the reinsert is wrongly accepted).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn device_locate_same_shard_twin_point_read_and_reinsert() {
    let run = |device: bool| {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(256); // a handful of rows stays in ONE open shard
        e.set_device_write_locate_enabled(device);
        e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO t (id, v) VALUES (1,10),(2,20),(3,30)")
            .unwrap();
        // UPDATE id=2: the new version appends into the SAME open shard as the old -> a
        // same-shard twin (old at the lower row-index / earlier probe slot).
        e.execute_text(3, "UPDATE t SET v = 999 WHERE id = 2")
            .unwrap();
        // (1) point read of the live updated key.
        let point = e
            .execute_relational_select_text("SELECT id, v FROM t WHERE id = 2")
            .unwrap()
            .rows
            .into_boxed();
        // (2) reinsert the SAME live key -> must be rejected as a duplicate (23505).
        let reinsert = e
            .execute_text(4, "INSERT INTO t (id, v) VALUES (2, 7)")
            .map(|_| ())
            .map_err(|err| err.to_string());
        let hits = e.device_write_locate_hits();
        (point, reinsert, hits)
    };
    let (on_point, on_reinsert, on_hits) = run(true);
    let (off_point, off_reinsert, _off_hits) = run(false);
    assert_eq!(on_point, off_point, "point read device == host oracle");
    assert_eq!(
        on_reinsert, off_reinsert,
        "reinsert outcome device == host oracle"
    );
    // The live updated key resolves to its NEW value (not empty).
    assert_eq!(
        on_point.len(),
        1,
        "point read returns the live updated row (not empty): {on_point:?}"
    );
    assert_eq!(
        on_point[0].get(1),
        Some(&SqlValue::Int4(999)),
        "point read sees the UPDATED value"
    );
    // Reinsert of a live key is a duplicate-PK violation on both arms.
    assert!(
        on_reinsert.is_err(),
        "reinserting a live key must be rejected as a duplicate: {on_reinsert:?}"
    );
    assert!(on_hits > 0, "non-vacuity: the device locate fired");
}

/// R-ver (read version resolution): a plain unfiltered `SELECT <cols> FROM t` over a VERSIONED
/// ELIDED table must (1) stay ELIDED — no de-elide — and (2) hide the tombstoned row. Before
/// this slice the unfiltered projection had no resident-route shape, so it dropped to the
/// CPU-pinned host path which REHYDRATES + de-elides. Now it routes to the sharded unified
/// executor with the SV3b `deleted_by` conjunct threaded. Sabotage: removing the
/// `int4_projection_all` route classification de-elides (the read falls to the CPU seam).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn plain_scan_over_versioned_elided_table_stays_elided() {
    let e = Engine::new_local();
    e.set_auto_admit_on_commit(true);
    e.set_host_install_elision_enabled(true);
    e.set_constrained_elision_enabled(true);
    e.set_device_write_locate_enabled(true);
    e.set_device_write_locate_wave_batch_enabled(true);
    e.set_shard_size_target(64);
    e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    let mut seq = 2u64;
    for chunk in 0..2_i64 {
        let vals: Vec<String> = (chunk * 100..(chunk + 1) * 100)
            .map(|k| format!("({k},{})", k * 10))
            .collect();
        e.execute_text(
            seq,
            &format!("INSERT INTO t (id, v) VALUES {}", vals.join(",")),
        )
        .unwrap();
        seq += 1;
    }
    if !e.table_install_elided("t") {
        return; // driverless box / never elided -> nothing to prove
    }
    // DELETE a row -> the shard becomes VERSIONED (a deleted_by region). Stays elided (U1).
    e.execute_text(seq, "DELETE FROM t WHERE id = 50").unwrap();
    assert!(
        e.table_install_elided("t"),
        "an in-place tombstone must not de-elide"
    );
    // THE GATE: a plain unfiltered scan must NOT de-elide, and must hide the tombstoned row.
    let rows = e
        .execute_relational_select_text("SELECT id, v FROM t")
        .unwrap()
        .rows
        .into_boxed();
    assert!(
        e.table_install_elided("t"),
        "a plain scan over a versioned elided table must NOT de-elide (route it on-device)"
    );
    assert!(
        !rows.iter().any(|r| r.first() == Some(&SqlValue::Int4(50))),
        "the tombstoned row must be hidden by the SV3b deleted_by conjunct"
    );
    assert_eq!(rows.len(), 199, "exactly one of the 200 rows is tombstoned");
    // A second scan still does not de-elide (idempotent, not a one-shot).
    let _ = e
        .execute_relational_select_text("SELECT id, v FROM t")
        .unwrap();
    assert!(e.table_install_elided("t"), "repeat scan stays elided");
    // `SELECT *` (SelectProjection::All) over the all-int4 table ALSO stays elided + hides the
    // tombstone (the classifier's All arm routes it on-device too).
    let star = e
        .execute_relational_select_text("SELECT * FROM t")
        .unwrap()
        .rows
        .into_boxed();
    assert!(
        e.table_install_elided("t"),
        "SELECT * over a versioned elided table must NOT de-elide"
    );
    assert_eq!(star.len(), 199, "SELECT * hides the tombstoned row too");
    assert!(!star.iter().any(|r| r.first() == Some(&SqlValue::Int4(50))));
}
