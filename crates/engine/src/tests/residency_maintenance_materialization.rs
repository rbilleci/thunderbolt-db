/// VACUUM #5 — the INDEX-RESTORATION differential: same-key update churn leaves the key in
/// TWO physical slots (old tombstoned + new), so the per-shard PK index dup-DECLINES and the
/// 3b point route stops serving (`shard_index_route_hits` stalls; the scan serves, correct
/// but slower — and the monotone decline caches it). `vacuum_table` rebuilds DENSE ALL-LIVE
/// (new buffer ptr -> the cached decline clears BY DESIGN) — the route serves again, rows are
/// byte-identical, and the churn counter resets. Also pins the tombstone-churn accounting.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn vacuum_restores_pk_index_after_update_churn() {
    let e = Engine::new_local();
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
    // Churn: two same-key updates -> id=130 occupies THREE slots (two dead) across shards.
    e.execute_text(300, "UPDATE accounts SET balance = 111 WHERE id = 130")
        .unwrap();
    e.execute_text(301, "UPDATE accounts SET balance = 222 WHERE id = 130")
        .unwrap();
    assert!(
        e.tombstone_churn("accounts") >= 2,
        "the churn counter must track the tombstone stamps (got {})",
        e.tombstone_churn("accounts")
    );
    let point = |e: &Engine| {
        e.execute_relational_select_text("SELECT id, balance FROM accounts WHERE id = 130")
            .unwrap()
            .rows
    };
    // The dup-declined regime: the point route must NOT serve via the PK index.
    let hits_before = e.shard_index_route_hits();
    let rows = point(&e);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0), &[SqlValue::Int4(130), SqlValue::Int4(222)]);
    assert_eq!(
        e.shard_index_route_hits(),
        hits_before,
        "precondition: the churned key dup-declines the index route (scan serves)"
    );
    let all_rows_sorted = |e: &Engine| {
        let mut rows = e
            .execute_relational_select_text("SELECT id, balance FROM accounts")
            .unwrap()
            .rows
            .into_boxed();
        rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        rows
    };
    let all_before = all_rows_sorted(&e);

    e.vacuum_table("accounts").unwrap();

    // Post-vacuum: the SAME reads, now index-served; data byte-identical; churn reset.
    let hits_before = e.shard_index_route_hits();
    let rows = point(&e);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0), &[SqlValue::Int4(130), SqlValue::Int4(222)]);
    assert!(
        e.shard_index_route_hits() > hits_before,
        "vacuum must RESTORE index serving (the rebuilt generation is dup-free)"
    );
    let all_after = all_rows_sorted(&e);
    assert_eq!(all_after, all_before, "vacuum preserves every row");
    assert_eq!(e.tombstone_churn("accounts"), 0, "the churn signal resets");
}

/// VACUUM #5 — the AUTO-TRIGGER + ELIDED lifecycle: an ELIDED table churns past the (forced)
/// threshold; the NEXT handled tombstone commit vacuums inside its own commit (rehydrate ->
/// rebuild), data stays correct, and the table RE-ENTERS elision on the following insert.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn vacuum_auto_trigger_rebuilds_elided_table_and_reelides() {
    let e = Engine::new_local();
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64);
    e.set_host_install_elision_enabled(true);
    e.set_auto_vacuum_enabled(true);
    e.set_tombstone_churn_threshold_override(3);
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
    e.execute_text(seq, "INSERT INTO t (id, v) VALUES (500, 5000)")
        .unwrap();
    seq += 1;
    assert!(
        e.table_install_elided("t"),
        "precondition: elided before the churn"
    );
    // Three single-key updates = 3 tombstone stamps -> the third commit crosses the forced
    // threshold and auto-vacuums (rehydrate + rebuild) INSIDE its own commit.
    for (i, key) in [10_i64, 11, 12].iter().enumerate() {
        e.execute_text(
            seq + i as u64,
            &format!("UPDATE t SET v = -1 WHERE id = {key}"),
        )
        .unwrap();
    }
    seq += 3;
    assert_eq!(
        e.tombstone_churn("t"),
        0,
        "the auto-vacuum reset the churn signal"
    );
    // Data correct after the mid-commit rebuild (incl the elided-era insert id=500).
    let rows = e
        .execute_relational_select_text("SELECT id, v FROM t WHERE id = 500")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0), &[SqlValue::Int4(500), SqlValue::Int4(5000)]);
    for key in [10, 11, 12] {
        let rows = e
            .execute_relational_select_text(&format!("SELECT id, v FROM t WHERE id = {key}"))
            .unwrap()
            .rows;
        assert_eq!(rows.len(), 1, "id {key}");
        assert_eq!(rows.row(0), &[SqlValue::Int4(key), SqlValue::Int4(-1)]);
    }
    // The vacuum de-elided (rehydration is sticky); the next handled insert RE-ENTERS.
    e.execute_text(seq, "INSERT INTO t (id, v) VALUES (501, 5010)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO t (id, v) VALUES (502, 5020)")
        .unwrap();
    assert!(
        e.table_install_elided("t"),
        "the table must RE-ENTER elision after the vacuum (the normal entry path)"
    );
    assert_eq!(
        e.execute_relational_select_text("SELECT id, v FROM t")
            .unwrap()
            .rows
            .len(),
        203,
        "200 + 3 inserts, updates in place"
    );
}

/// RETIREMENT A4c — the DEVICE GATHER differential: `gather_resident_table_rows_from_device`
/// (the re-admit / de-elision rebuild source) == the host store's visible rows, (row_id, row)
/// for (row_id, row), across the full write lineage (admission, SV5 version-split update,
/// tombstoned DELETE, post-churn append, multi-row A4b update). The gather must SKIP
/// tombstoned/old-version slots and carry every identity; a NULL-bearing table must DECLINE.
/// Sabotage: invert the visibility filter and the tombstoned rows surface -> FAIL.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a4c_device_gather_matches_host_store() {
    let e = Engine::new_local();
    // Host-store oracle premise pinned (see a1_device_row_identity_matches_host_store).
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
    e.execute_text(300, "UPDATE accounts SET balance = 9999 WHERE id = 130")
        .unwrap();
    e.execute_text(301, "DELETE FROM accounts WHERE id = 42")
        .unwrap();
    e.execute_text(302, "INSERT INTO accounts (id, balance) VALUES (500, 5000)")
        .unwrap();
    e.execute_text(
        303,
        "UPDATE accounts SET balance = 1 WHERE id = 10 OR id = 11",
    )
    .unwrap();
    let now = e.committed_seq();
    let table = e.relational_catalog_table("accounts").unwrap();

    let mut got = e
        .gather_resident_table_rows_from_device(&table, now)
        .expect("the gather must ANSWER for a clean int4 lineage (else A4c is vacuous)");
    // Host oracle: the seq-scan at the same snapshot, (row_id from key, decoded row).
    let table_rows = e.read_state.mvcc.table_rows("accounts");
    let prefix = relational_key_prefix("accounts");
    let mut want: Vec<(u64, Vec<SqlValue>)> = Vec::new();
    let mut cursor = table_rows
        .store()
        .seq_scan_open(crate::StorageVisibility { read_txn_id: now })
        .unwrap();
    while let Some(tuple) = cursor.next() {
        if !tuple.key.starts_with(&prefix) {
            continue;
        }
        let row_id = parse_relational_row_id(&tuple.key, &prefix)
            .expect("every stored key parses (A1 invariant)");
        want.push((
            row_id,
            decode_relational_row(&tuple.value, &table.columns).unwrap(),
        ));
    }
    drop(cursor);
    got.sort_by_key(|(row_id, _)| *row_id);
    want.sort_by_key(|(row_id, _)| *row_id);
    assert_eq!(
        got.len(),
        200,
        "200 - 1 delete + 1 insert = 200 visible rows"
    );
    assert_eq!(
        got, want,
        "device gather == host store, identity for identity"
    );

    // NULL-bearing table: the gather is now NULL-AWARE (the ADR-006 alignment-free
    // NULL_BITMAP_GATHER work) — it must materialize SqlValue::Null exactly, NEVER decline
    // (the old decline expectation) and NEVER NULL-as-0.
    e.execute_text(400, "CREATE TABLE n (id INT, v INT)")
        .unwrap();
    e.execute_text(401, "INSERT INTO n (id, v) VALUES (1, NULL), (2, 20)")
        .unwrap();
    let n_table = e.relational_catalog_table("n").unwrap();
    let gathered = e
        .gather_resident_table_rows_from_device(&n_table, e.committed_seq())
        .expect("the NULL-aware gather serves a null-bearing table");
    let mut rows: Vec<Vec<SqlValue>> = gathered.into_iter().map(|(_, row)| row).collect();
    rows.sort_by(|a, b| compare_sql_values(&a[0], &b[0]));
    assert_eq!(
        rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Null],
            vec![SqlValue::Int4(2), SqlValue::Int4(20)],
        ],
        "the gather materializes NULL exactly (never 0, never a decline)"
    );
}

/// RETIREMENT A4b — MULTI-ROW incremental DML: multi-row UPDATE/DELETE commits are handled
/// IN PLACE (per-row exact-1 locate+tombstone, one batched identity-stamped append) instead of
/// the O(table) invalidate+re-admit. Twin-engine differential (incremental ON vs OFF=re-admit
/// oracle) over multi-row UPDATE, multi-row DELETE, and an unchanged-values UPDATE; MECHANISM
/// pin: every pre-statement device buffer SURVIVES on the incremental engine (a re-admit
/// replaces all ptrs — output equality alone cannot see the fallback, the A2 lesson); IDENTITY
/// pin: after the multi-row UPDATE each new version materializes (A4a) to exactly the host row
/// fetched by its DERIVED key. Sabotage: force `try_update_resident_commit` multi-row arm to
/// false and the ptr-survival assert FAILS.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a4b_multi_row_dml_stays_incremental_and_matches_oracle() {
    let load = |e: &Engine, incremental: bool| {
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.set_resident_delete_tombstone_enabled(incremental);
        e.set_resident_update_tombstone_enabled(incremental);
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
    let statements = [
        "UPDATE accounts SET balance = 7777 WHERE id = 10 OR id = 11",
        "DELETE FROM accounts WHERE id = 20 OR id = 21 OR id = 22",
        // Unchanged int4 values: the tombstone-first order must still locate EXACTLY the olds.
        "UPDATE accounts SET balance = 300 WHERE id = 30",
        "UPDATE accounts SET balance = 1234 WHERE id = 40 OR id = 41",
    ];
    let e = Engine::new_local();
    // PINNED NON-ELIDED (A5 flip): the oracle reads the HOST store / pins pre-elision mechanics (production-live for non-eligible tables).
    e.set_host_install_elision_enabled(false);
    load(&e, true);
    let o = Engine::new_local();
    // PINNED NON-ELIDED (A5 flip): the oracle reads the HOST store / pins pre-elision mechanics (production-live for non-eligible tables).
    o.set_host_install_elision_enabled(false);
    load(&o, false);
    let ptrs_before: Vec<(u32, u64)> = {
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
    let mut seq = 300_u64;
    for sql in &statements {
        e.execute_text(seq, sql).unwrap();
        o.execute_text(seq, sql).unwrap();
        seq += 1;
    }
    for (shard_id, ptr) in &ptrs_before {
        let survived = e
            .read_state
            .residency
            .shard_device_memory
            .get(&("accounts".to_string(), *shard_id))
            .is_some_and(|memory| memory.device_ptr() == *ptr);
        assert!(
            survived,
            "shard {shard_id} was REBUILT: multi-row DML must stay on the incremental path"
        );
    }
    let got = e
        .execute_relational_select_text("SELECT id, balance FROM accounts")
        .unwrap()
        .rows
        .into_boxed();
    let want = o
        .execute_relational_select_text("SELECT id, balance FROM accounts")
        .unwrap()
        .rows
        .into_boxed();
    // The oracle re-admits (all-live rebuild) so its row ORDER can differ; compare as multisets.
    let mut got_rows = got;
    let mut want_rows = want;
    got_rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    want_rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    assert_eq!(
        got_rows, want_rows,
        "incremental multi-row DML == re-admit oracle"
    );
    // IDENTITY pin (A4a composition): each updated key's visible version materializes from the
    // device to exactly the host row fetched by its derived key.
    let table = e.relational_catalog_table("accounts").unwrap();
    let table_rows = e.read_state.mvcc.table_rows("accounts");
    let now = e.committed_seq();
    for id in [10_i32, 11, 40, 41, 30] {
        let hits = e
            .locate_resident_pk_via_shard_index_detailed(&table, 0, id)
            .expect("locate must answer post-update");
        let visible: Vec<Vec<SqlValue>> = hits
            .iter()
            .filter_map(|hit| {
                e.materialize_resident_row_via_hit(&table, hit, now)
                    .unwrap()
            })
            .collect();
        assert_eq!(visible.len(), 1, "id {id}: exactly one visible version");
        let region = hits
            .iter()
            .find_map(|hit| hit.row_id.as_ref().map(|r| (hit, r)))
            .expect("identity region");
        let halves = region
            .1
            .read_resident_i32_column(u64::from(region.0.slot) * 8, 2)
            .unwrap();
        let row_id = (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32);
        let host = table_rows
            .store()
            .tuple_fetch_by_key(
                &relational_row_key("accounts", row_id),
                crate::StorageVisibility { read_txn_id: now },
            )
            .unwrap()
            .map(|tuple| decode_relational_row(&tuple.value, &table.columns).unwrap());
        assert_eq!(
            host.as_ref(),
            Some(&visible[0]),
            "id {id}: device == host by derived key"
        );
    }
}

/// RETIREMENT A4b — the CONCURRENT form: a reader hammers TWO keys while the writer commits
/// real multi-row `UPDATE ... WHERE id = 130 OR id = 131` statements through the incremental
/// path (per-statement: two tombstones + one batched created_by-stamped append). SI invariant
/// under EVERY interleaving: EACH key appears EXACTLY ONCE per read (2 = the un-stamped
/// double-read; 0 = a lost row). Same-key chains stay incremental here (unlike the A2 resolve)
/// because the locate predicate is the FULL row image and balances keep changing.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a4b_concurrent_reader_exactly_once_under_multi_row_update_load() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_resident_delete_tombstone_enabled(true);
    e.set_resident_update_tombstone_enabled(true);
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
                for key in [130, 131] {
                    let rows = e
                        .execute_relational_select_text(&format!(
                            "SELECT id, balance FROM accounts WHERE id = {key}"
                        ))
                        .unwrap()
                        .rows;
                    assert_eq!(
                        rows.len(),
                        1,
                        "SI: key {key} must appear EXACTLY ONCE under multi-row update load"
                    );
                }
                reads += 1;
            }
            reads
        });
        for t in 0..150_u64 {
            e.execute_text(
                300 + t,
                &format!(
                    "UPDATE accounts SET balance = {} WHERE id = 130 OR id = 131",
                    100_000 + t
                ),
            )
            .unwrap();
        }
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let reads = reader.join().expect("reader must not panic (SI violation)");
        assert!(reads > 0, "the reader must have raced at least one read");
    });
    let rows = e
        .execute_relational_select_text("SELECT id, balance FROM accounts WHERE id = 131")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0), &[SqlValue::Int4(131), SqlValue::Int4(100_149)]);
}

/// RETIREMENT A4a — the DEVICE MATERIALIZATION differential: for located hits across every
/// write lineage (admission, SV5 update version-split + re-update chain, DELETE tombstone,
/// post-churn append) and MULTIPLE time-travel snapshots (pre/at/post each commit),
/// `materialize_resident_row_via_hit` == the host `tuple_fetch_by_key` at the same
/// `read_txn_id`: visible rows carry IDENTICAL values, invisible slots answer `Some(None)`
/// exactly where the host fetch misses. This is the primitive that REPLACES the host fetch
/// when A4e elides installs — the visibility boundary (`created_by <= t < deleted_by`) is the
/// load-bearing edge, probed AT the exact commit seqs. A NULL-bearing shard must DECLINE
/// (`None`, the M3 raw-i32 discipline). Sabotage: relax `read_txn_id < deleted_by` to `<=`
/// and the at-boundary probes see tombstoned rows -> FAIL.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a4a_device_materialization_matches_host_fetch() {
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
    let t_admitted = e.committed_seq();
    e.execute_text(300, "UPDATE accounts SET balance = 9999 WHERE id = 130")
        .unwrap();
    let t_update = e.committed_seq();
    e.execute_text(301, "DELETE FROM accounts WHERE id = 42")
        .unwrap();
    let t_delete = e.committed_seq();
    e.execute_text(302, "INSERT INTO accounts (id, balance) VALUES (500, 5000)")
        .unwrap();
    // NOTE: no SECOND update of id=130 — that would duplicate the key WITHIN the open shard
    // and locate would (correctly) DECLINE it; the dup-decline lineage is pinned by the A2
    // same-key-chain test. Here every probed key stays locate-eligible so the materializer
    // itself is what's under test.
    let t_latest = e.committed_seq();

    let table = e.relational_catalog_table("accounts").unwrap();
    let table_rows = e.read_state.mvcc.table_rows("accounts");
    let snapshots = [
        t_admitted,
        t_update - 1,
        t_update,
        t_delete - 1,
        t_delete,
        t_latest,
    ];
    let mut visible_checked = 0_usize;
    let mut invisible_checked = 0_usize;
    // id=500 (an INSERT-appended slot) is probed ONLY at t_latest: insert-appended slots are
    // BORN-VISIBLE (no created_by stamp; concurrent readers gate them via the pinned
    // row_count) — the primitive's contract is read_txn >= the slot's insert commit, which is
    // what the serialized DML path always passes. The admission/update/delete lineages carry
    // stamps and are probed across ALL snapshots (the update/delete boundaries are the edge).
    for id in [130_i32, 42, 500, 7, 60] {
        let Some(hits) = e.locate_resident_pk_via_shard_index_detailed(&table, 0, id) else {
            panic!("locate must answer for id {id} (unique key, valid shards)");
        };
        for &txn in &snapshots {
            if id == 500 && txn < t_latest {
                continue; // outside the born-visible contract (see above)
            }
            let mut device_visible: Vec<Vec<SqlValue>> = Vec::new();
            for hit in &hits {
                // Host oracle for THIS slot: the derived key fetched at the same snapshot.
                let region = hit.row_id.as_ref().expect("identity region present");
                let halves = region
                    .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
                    .unwrap();
                let row_id = (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32);
                let key = relational_row_key("accounts", row_id);
                let host = table_rows
                    .store()
                    .tuple_fetch_by_key(&key, crate::StorageVisibility { read_txn_id: txn })
                    .unwrap()
                    .map(|tuple| decode_relational_row(&tuple.value, &table.columns).unwrap());
                let device = e
                    .materialize_resident_row_via_hit(&table, hit, txn)
                    .unwrap_or_else(|| {
                        panic!("materializer must not DECLINE a null-free shard (id {id})")
                    });
                // The host key resolves the LOGICAL row (its current version at txn); the
                // device hit is a PHYSICAL slot. A visible device slot must carry exactly the
                // host row; an invisible slot pairs with either a host miss (row dead at txn)
                // OR the row being visible via its OTHER version slot — so per-slot we assert
                // only the visible direction, and per-(id, txn) the visible SETS must match.
                match device {
                    Some(row) => {
                        assert_eq!(
                            Some(&row),
                            host.as_ref(),
                            "id {id} txn {txn} slot {}: device row == host fetch",
                            hit.slot
                        );
                        device_visible.push(row);
                        visible_checked += 1;
                    }
                    None => invisible_checked += 1,
                }
            }
            // Set-level: the host sees the id at txn ⟺ EXACTLY ONE device slot is visible.
            let host_row = table_rows
                .store()
                .tuple_fetch_by_key(
                    &relational_row_key(
                        "accounts",
                        // any hit's row_id resolves the same logical row for this unique id
                        {
                            let region = hits[0].row_id.as_ref().unwrap();
                            let halves = region
                                .read_resident_i32_column(u64::from(hits[0].slot) * 8, 2)
                                .unwrap();
                            (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32)
                        },
                    ),
                    crate::StorageVisibility { read_txn_id: txn },
                )
                .unwrap();
            assert_eq!(
                device_visible.len(),
                usize::from(host_row.is_some()),
                "id {id} txn {txn}: exactly one visible slot iff the host sees the row"
            );
        }
    }
    assert!(
        visible_checked >= 20,
        "non-vacuity: visible probes ({visible_checked})"
    );
    assert!(
        invisible_checked >= 4,
        "non-vacuity: INVISIBLE probes must exercise the boundary ({invisible_checked})"
    );

    // Date column (audit A4 F1, LIFTED by type-coverage track 2): Date/Int2 share the
    // device i32 section; the materializer now derives the SqlValue variant from the
    // CATALOG column type — the materialized row must carry `Date(days)` matching the host
    // fetch EXACTLY (the F1 mistype `Int4(days)` would fail this equality).
    e.execute_text(390, "CREATE TABLE dd (id INT, d DATE)")
        .unwrap();
    e.execute_text(
        391,
        "INSERT INTO dd (id, d) VALUES (1, '2026-07-02'), (2, '2026-07-01')",
    )
    .unwrap();
    let dd_table = e.relational_catalog_table("dd").unwrap();
    let dd_rows = e.read_state.mvcc.table_rows("dd");
    let mut date_typed_checked = false;
    if let Some(hits) = e.locate_resident_pk_via_shard_index_detailed(&dd_table, 0, 1) {
        for hit in &hits {
            let device_row = e
                .materialize_resident_row_via_hit(&dd_table, hit, e.committed_seq())
                .expect("an i32-section table must answer, not decline")
                .expect("id 1 is live");
            assert!(
                matches!(device_row[1], SqlValue::Date(_)),
                "the d column must materialize as Date, not Int4 (got {:?})",
                device_row[1]
            );
            let host_row = {
                let region = hit.row_id.as_ref().unwrap();
                let halves = region
                    .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
                    .unwrap();
                let row_id = (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32);
                let tuple = dd_rows
                    .store()
                    .tuple_fetch_by_key(
                        &relational_row_key("dd", row_id),
                        crate::StorageVisibility {
                            read_txn_id: e.committed_seq(),
                        },
                    )
                    .unwrap()
                    .expect("host row exists");
                decode_relational_row(&tuple.value, &dd_table.columns).unwrap()
            };
            assert_eq!(device_row, host_row, "typed device row == host fetch");
            date_typed_checked = true;
        }
    }
    assert!(
        date_typed_checked,
        "the Date-typing path must be exercised (locate answered)"
    );

    // NULL-bearing shard (ADR-006 nullable-column DML): the materializer no longer DECLINES —
    // it reads the per-column validity bitmap and reconstructs the row WITH `SqlValue::Null`
    // (never aliasing a stored NULL as 0). Row id=1 has v=NULL.
    e.execute_text(400, "CREATE TABLE n (id INT, v INT)")
        .unwrap();
    e.execute_text(401, "INSERT INTO n (id, v) VALUES (1, NULL), (2, 20)")
        .unwrap();
    let n_table = e.relational_catalog_table("n").unwrap();
    let mut null_materialization_checked = false;
    if let Some(hits) = e.locate_resident_pk_via_shard_index_detailed(&n_table, 0, 1) {
        for hit in &hits {
            if let Some(Some(device_row)) =
                e.materialize_resident_row_via_hit(&n_table, hit, e.committed_seq())
            {
                assert_eq!(
                    device_row,
                    vec![SqlValue::Int4(1), SqlValue::Null],
                    "null-aware materialization: id=1 reads v as SqlValue::Null (not 0)"
                );
                null_materialization_checked = true;
            }
        }
    }
    assert!(
        null_materialization_checked,
        "the NULL-aware materialization path must be exercised (was a blanket decline)"
    );
}
