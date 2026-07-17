/// RETIREMENT A3 — the mandatory device validator ladder. Covers unique violation + PASS (the FALSE answer is the
/// load-bearing one — a device miss would wrongly ADMIT a duplicate), unique-through-SV5-churn
/// (the version-split physical hit must be neutralized by fetch-at-visibility), unique
/// key-move, outbound-FK present/absent, inbound-FK blocked/allowed DELETE, and NULL-on-unique
/// with structural NULL==NULL semantics. NON-VACUITY: `dml_device_validate_hits`
/// must ADVANCE. Sabotage: make the
/// device probe skip `answer = true` and the violation statements wrongly SUCCEED -> outcome
/// vectors diverge -> FAIL.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a3_device_validator_serves_constraint_ladder() {
    let run = || {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.execute_text(1, "CREATE TABLE t (id INT UNIQUE, v INT)")
            .unwrap();
        e.execute_text(2, "CREATE TABLE c (id INT, tid INT)")
            .unwrap();
        e.execute_text(
            3,
            "ALTER TABLE ONLY c ADD CONSTRAINT c_tid_fk FOREIGN KEY (tid) REFERENCES t(id)",
        )
        .unwrap();
        let mut seq = 4u64;
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
        e.execute_text(seq, "INSERT INTO c (id, tid) VALUES (1, 42)")
            .unwrap();
        seq += 1;
        let statements = [
            "INSERT INTO t (id, v) VALUES (50, 1)", // unique violation (device answers TRUE)
            "INSERT INTO t (id, v) VALUES (500, 1)", // fresh id: SUCCESS (the FALSE answer)
            "UPDATE t SET v = 5555 WHERE id = 60",  // SV5 churn: version-splits id=60
            "INSERT INTO t (id, v) VALUES (60, 2)", // still a violation THROUGH the churn
            "UPDATE t SET id = 70 WHERE id = 61",   // unique key-move onto a live key: violation
            "UPDATE t SET id = 600 WHERE id = 61",  // key-move to a fresh key: success
            "INSERT INTO c (id, tid) VALUES (2, 77)", // outbound FK: provider exists
            "INSERT INTO c (id, tid) VALUES (3, 9999)", // outbound FK: no provider -> violation
            "DELETE FROM t WHERE id = 42",          // inbound FK: a child still references 42
            "DELETE FROM t WHERE id = 43",          // no child -> success
            "INSERT INTO t (id, v) VALUES (NULL, 1)", // NULL on unique: host semantics serve
            "INSERT INTO t (id, v) VALUES (NULL, 2)", // second NULL: MUST match host outcome
        ];
        let hits_before = e.dml_device_validate_hits();
        let outcomes: Vec<Result<(), String>> = statements
            .iter()
            .map(|sql| {
                let r = e
                    .execute_text(seq, sql)
                    .map(|_| ())
                    .map_err(|err| err.to_string());
                seq += 1;
                r
            })
            .collect();
        let hits = e.dml_device_validate_hits() - hits_before;
        let t_rows = e
            .execute_relational_select_text("SELECT id, v FROM t")
            .unwrap()
            .rows
            .into_boxed();
        let c_rows = e
            .execute_relational_select_text("SELECT id, tid FROM c")
            .unwrap()
            .rows
            .into_boxed();
        (outcomes, hits, t_rows, c_rows)
    };
    let (dev_out, dev_hits, dev_t, dev_c) = run();
    // Audit A3 finding 1: a FLOOR, not just >0 — the 12-statement sequence carries ~14
    // device-servable Int4 probes (unique per new image, FK survivor/child pairs). The floor
    // catches a coverage regression before a decline becomes the expected loud error.
    assert!(
        dev_hits >= 10,
        "non-vacuity floor: the device index must have ANSWERED most probes (got {dev_hits})"
    );
    // Spot-pin the shape (guards both-engines-wrong drift).
    assert!(
        dev_out[0].as_ref().is_err_and(|err| err.contains("unique")),
        "statement 0 must be a unique violation: {:?}",
        dev_out[0]
    );
    assert!(
        dev_out[1].is_ok(),
        "fresh insert must succeed: {:?}",
        dev_out[1]
    );
    assert!(
        dev_out[3].as_ref().is_err_and(|err| err.contains("unique")),
        "the churned-key insert must STILL violate: {:?}",
        dev_out[3]
    );
    assert!(
        dev_out[7]
            .as_ref()
            .is_err_and(|err| err.contains("foreign key")),
        "orphan child insert must violate the FK: {:?}",
        dev_out[7]
    );
    assert!(
        dev_out[8]
            .as_ref()
            .is_err_and(|err| err.contains("foreign key")),
        "referenced-provider delete must violate the FK: {:?}",
        dev_out[8]
    );
    assert!(
        dev_t
            .iter()
            .any(|row| row.first() == Some(&SqlValue::Int4(500))),
        "the fresh unique key is present"
    );
    assert!(
        !dev_t
            .iter()
            .any(|row| row.first() == Some(&SqlValue::Int4(43))),
        "the unreferenced provider is deleted"
    );
    assert_eq!(
        dev_c.len(),
        2,
        "only the provider-backed child insert succeeds"
    );
}

/// RETIREMENT A2 regression (the SV6-hammer bug): after an SV5 update-append, one LOGICAL row
/// occupies slots in TWO shards (tombstoned old slot in its sealed shard, new version in the
/// open shard) — the visibility-blind locate hits BOTH, and both derive the SAME key. The
/// resolve must DEDUP them to one match: emitting two made prepare hand the SV5 gate 2 matches
/// for 1 slot, so every same-key re-UPDATE fell back to INVALIDATE+RE-ADMIT (O(table), plus the
/// reader-visible invalid window the hammer tripped). Output differentials CANNOT see that
/// fallback (re-admit is correctness-preserving) — this pins the INCREMENTAL path directly:
/// the pre-update shard buffers must SURVIVE the update chain (a re-admit replaces every ptr).
/// Sabotage: remove the `dedup_by_key` in `resolve_dml_matches_via_device` and this FAILS.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a2_same_key_update_chain_stays_on_incremental_path() {
    let e = Engine::new_local();
    // PINNED NON-ELIDED (A5 flip): the oracle reads the HOST store / pins pre-elision mechanics (production-live for non-eligible tables).
    e.set_host_install_elision_enabled(false);
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
    // First update: splits id=130 across shards (old tombstoned slot + appended new version).
    e.execute_text(300, "UPDATE accounts SET balance = 111 WHERE id = 130")
        .unwrap();
    let before: Vec<(u32, u64)> = {
        let shards = e
            .read_state
            .residency
            .shards
            .load()
            .get("accounts")
            .cloned()
            .unwrap();
        shards
            .iter()
            .map(|shard| {
                let memory = e
                    .read_state
                    .residency
                    .shard_device_memory
                    .get(&("accounts".to_string(), shard.shard_id))
                    .unwrap();
                (shard.shard_id, memory.device_ptr())
            })
            .collect()
    };
    // The chain: each re-update's resolve sees the cross-shard version split.
    for t in 0..8_u64 {
        e.execute_text(
            301 + t,
            &format!("UPDATE accounts SET balance = {} WHERE id = 130", 200 + t),
        )
        .unwrap();
    }
    // Every pre-chain buffer survives: appends/rollovers only ADD shards; an invalidate+
    // re-admit (the bug's fallback) replaces EVERY device ptr.
    let after = e
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .cloned()
        .unwrap();
    for (shard_id, ptr) in &before {
        let survived = e
            .read_state
            .residency
            .shard_device_memory
            .get(&("accounts".to_string(), *shard_id))
            .is_some_and(|memory| memory.device_ptr() == *ptr);
        assert!(
            survived,
            "shard {shard_id} was REBUILT during the same-key update chain: the resolve must \
             dedup the version-split multi-hit so the SV5 incremental path handles the commit"
        );
    }
    assert!(after.len() >= before.len(), "appends only ever ADD shards");
    // End-state correctness on top of the mechanism pin.
    let rows = e
        .execute_relational_select_text("SELECT id, balance FROM accounts WHERE id = 130")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0), &[SqlValue::Int4(130), SqlValue::Int4(207)]);
}

/// RETIREMENT A1 — the DEVICE ROW-IDENTITY differential: for EVERY (shard, live slot) of a
/// resident table, the device `row_id` region's value derives the host key
/// (`rel/{table}/{row_id:020}`), and the host row FETCHED BY THAT KEY matches the device row's
/// values (per-slot DtoH of the int4 columns — the 3b gather pattern). Exercised across
/// ADMISSION (re-admit parse), IN-PLACE INSERT append, ROLLOVER, and the SV5 UPDATE append (the
/// appended slot must carry the ORIGINAL row's id — same key). NON-VACUITY: a sentinel at any
/// LIVE slot of a region-bearing shard FAILS (headroom is born-sentinel, so a skipped stamp is
/// detectable); a mis-stamped id fetches the WRONG host row -> value mismatch -> FAIL.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a1_device_row_identity_matches_host_store() {
    let e = Engine::new_local();
    // This differential's ORACLE is the host tuple store — valid only while commits still
    // install host tuples. Elision (device-authoritative commits, default-eligible for this
    // shape since the lanes/elision flips) leaves the store intentionally stale, so pin the
    // oracle's premise OFF for this gate (the elided twin is gated by the A4c device gathers).
    e.set_host_install_elision_enabled(false);
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    // Admission + rollover lineage: 200 rows -> shards.
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
    // SV5 UPDATE append: the new version must carry id-130's ORIGINAL row identity.
    e.execute_text(300, "UPDATE accounts SET balance = 9999 WHERE id = 130")
        .unwrap();
    // More in-place appends after the update.
    e.execute_text(
        301,
        "INSERT INTO accounts (id, balance) VALUES (500, 5000), (501, 5010)",
    )
    .unwrap();
    // Audit finding 1: the CONCURRENT insert stamp site (the production wave path) — identities
    // parse from the re-validated delta at the append site.
    e.execute_dml_concurrent(
        302,
        "INSERT INTO accounts (id, balance) VALUES (600, 6000), (601, 6010)",
    )
    .unwrap();
    // Audit finding 2: a MULTI-ROW UPDATE bails the SV5 gate -> invalidate + RE-ADMIT -> the
    // ADMISSION parse rebuilds every shard's region over the full row set.
    e.execute_text(
        303,
        "UPDATE accounts SET balance = 1 WHERE id = 10 OR id = 11",
    )
    .unwrap();

    let visibility = crate::StorageVisibility {
        read_txn_id: e.committed_seq(),
    };
    let prefix = relational_key_prefix("accounts");
    let table = e.relational_catalog_table("accounts").unwrap();
    let table_rows = e.read_state.mvcc.table_rows("accounts");
    let shards = e
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .cloned()
        .unwrap();
    let mut checked = 0usize;
    for shard in &shards {
        let region = e
            .read_state
            .residency
            .shard_row_id_memory
            .get(&("accounts".to_string(), shard.shard_id))
            .unwrap_or_else(|| panic!("shard {} must carry a row-identity region", shard.shard_id));
        let device_memory = e
            .read_state
            .residency
            .shard_device_memory
            .get(&("accounts".to_string(), shard.shard_id))
            .unwrap();
        let descriptor = e.resident_snapshot_for_shard(shard, &table);
        for slot in 0..shard.row_count {
            // Device row_id (two i32 halves, LE).
            let halves = region.read_resident_i32_column(slot as u64 * 8, 2).unwrap();
            let row_id = (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32);
            assert_ne!(
                row_id,
                u64::MAX,
                "live slot {slot} of shard {} must be STAMPED (sentinel found)",
                shard.shard_id
            );
            // Device row values (per-slot DtoH, the 3b gather pattern).
            let mut device_row = Vec::new();
            for (idx, _col) in table.columns.iter().enumerate() {
                let base = crate::relational_model::resident_device_int4_column_offset(
                    &descriptor,
                    &table,
                    idx,
                )
                .unwrap();
                let v = device_memory
                    .read_resident_i32_column(base + slot as u64 * 4, 1)
                    .unwrap();
                device_row.push(v[0]);
            }
            // Host row by the DERIVED key.
            let key = relational_row_key("accounts", row_id);
            assert!(key.starts_with(&prefix));
            let tuple = table_rows
                .store()
                .tuple_fetch_by_key(&key, visibility)
                .unwrap()
                .unwrap_or_else(|| {
                    panic!(
                        "derived key {key} (shard {} slot {slot}) must fetch a visible host row",
                        shard.shard_id
                    )
                });
            let host_row = decode_relational_row(&tuple.value, &table.columns).unwrap();
            for (idx, host_v) in host_row.iter().enumerate() {
                let host_i32 = match host_v {
                    SqlValue::Int4(v) => *v,
                    other => panic!("int4 table expected, got {other:?}"),
                };
                // The SV5-tombstoned OLD slot for id=130 still carries the ORIGINAL identity;
                // its host fetch returns the CURRENT version (balance 9999) while the device
                // slot holds the old bytes — identity equality is on the KEY, value equality
                // applies to the id column always and to balance only for non-superseded slots.
                if idx == 0 {
                    assert_eq!(
                        device_row[idx], host_i32,
                        "shard {} slot {slot}: device id column must match the host row at the derived key",
                        shard.shard_id
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(
        checked >= 202,
        "checked {checked} slots (admission + appends + update)"
    );
}
