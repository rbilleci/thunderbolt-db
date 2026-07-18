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
        e.table_device_authoritative("t"),
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
        e.table_device_authoritative("t"),
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
/// matches the closed-form visible `(row_id, row)` set across the full write lineage (admission,
/// SV5 version-split update,
/// tombstoned DELETE, post-churn append, multi-row A4b update). The gather must SKIP
/// tombstoned/old-version slots and carry every identity; a NULL-bearing table must DECLINE.
/// Sabotage: invert the visibility filter and the tombstoned rows surface -> FAIL.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a4c_device_gather_matches_closed_form_state() {
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
    let mut want = (0_i32..200)
        .filter(|id| *id != 42)
        .map(|id| {
            let balance = match id {
                10 | 11 => 1,
                130 => 9999,
                _ => id * 10,
            };
            ((id as u64) + 1, vec![SqlValue::Int4(id), SqlValue::Int4(balance)])
        })
        .collect::<Vec<_>>();
    want.push((201, vec![SqlValue::Int4(500), SqlValue::Int4(5000)]));
    got.sort_by_key(|(row_id, _)| *row_id);
    want.sort_by_key(|(row_id, _)| *row_id);
    assert_eq!(
        got.len(),
        200,
        "200 - 1 delete + 1 insert = 200 visible rows"
    );
    assert_eq!(
        got, want,
        "device gather == closed-form state, identity for identity"
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
/// the O(table) invalidate+re-admit. Closed-form differential over multi-row UPDATE, multi-row
/// DELETE, and an unchanged-values UPDATE; MECHANISM
/// pin: every pre-statement device buffer SURVIVES on the incremental engine (a re-admit
/// replaces all ptrs — output equality alone cannot see the fallback, the A2 lesson); IDENTITY
/// pin: after the multi-row UPDATE each new version materializes (A4a) to exactly the host row
/// fetched by its DERIVED key. Sabotage: force `try_update_resident_commit` multi-row arm to
/// false and the ptr-survival assert FAILS.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a4b_multi_row_dml_stays_incremental_and_matches_oracle() {
    let load = |e: &Engine| {
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
    };
    let statements = [
        "UPDATE accounts SET balance = 7777 WHERE id = 10 OR id = 11",
        "DELETE FROM accounts WHERE id = 20 OR id = 21 OR id = 22",
        // Unchanged int4 values: the tombstone-first order must still locate EXACTLY the olds.
        "UPDATE accounts SET balance = 300 WHERE id = 30",
        "UPDATE accounts SET balance = 1234 WHERE id = 40 OR id = 41",
    ];
    let e = Engine::new_local();
    load(&e);
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
    for (seq, sql) in (300_u64..).zip(&statements) {
        e.execute_text(seq, sql).unwrap();
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
    let mut got_rows = got;
    let mut want_rows = (0_i32..200)
        .filter(|id| !matches!(id, 20..=22))
        .map(|id| {
            let balance = match id {
                10 | 11 => 7777,
                30 => 300,
                40 | 41 => 1234,
                _ => id * 10,
            };
            vec![SqlValue::Int4(id), SqlValue::Int4(balance)]
        })
        .collect::<Vec<_>>();
    got_rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    want_rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    assert_eq!(
        got_rows, want_rows,
        "incremental multi-row DML == closed-form relational result"
    );
    // IDENTITY pin (A4a composition): each updated key has exactly one visible device version,
    // carries the original entity identity, and materializes to the closed-form row image.
    let table = e.relational_catalog_table("accounts").unwrap();
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
        assert_ne!(row_id, u64::MAX, "id {id}: device identity is populated");
        let expected_balance = match id {
            10 | 11 => 7777,
            30 => 300,
            40 | 41 => 1234,
            _ => unreachable!(),
        };
        assert_eq!(
            visible[0],
            vec![SqlValue::Int4(id), SqlValue::Int4(expected_balance)],
            "id {id}: device materialization matches the committed row image"
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
/// `materialize_resident_row_via_hit` matches the closed-form state at the same `read_txn_id`:
/// visible rows carry exact values and invisible slots answer `Some(None)`. The visibility
/// boundary (`created_by <= t < deleted_by`) is the
/// load-bearing edge, probed AT the exact commit seqs. A NULL-bearing shard must DECLINE
/// (`None`, the M3 raw-i32 discipline). Sabotage: relax `read_txn_id < deleted_by` to `<=`
/// and the at-boundary probes see tombstoned rows -> FAIL.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn a4a_device_materialization_matches_version_boundaries() {
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
                        device_visible.push(row);
                        visible_checked += 1;
                    }
                    None => invisible_checked += 1,
                }
            }
            let expected = match id {
                130 => Some(vec![
                    SqlValue::Int4(130),
                    SqlValue::Int4(if txn < t_update { 1300 } else { 9999 }),
                ]),
                42 if txn >= t_delete => None,
                42 => Some(vec![SqlValue::Int4(42), SqlValue::Int4(420)]),
                500 => Some(vec![SqlValue::Int4(500), SqlValue::Int4(5000)]),
                7 => Some(vec![SqlValue::Int4(7), SqlValue::Int4(70)]),
                60 => Some(vec![SqlValue::Int4(60), SqlValue::Int4(600)]),
                _ => unreachable!(),
            };
            assert_eq!(
                device_visible.len(),
                usize::from(expected.is_some()),
                "id {id} txn {txn}: exactly one slot is visible when the entity exists"
            );
            if let Some(expected) = expected {
                assert_eq!(device_visible[0], expected, "id {id} txn {txn}: exact row image");
            }
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
    // CATALOG column type — the materialized row must carry `Date(days)` exactly.
    e.execute_text(390, "CREATE TABLE dd (id INT, d DATE)")
        .unwrap();
    e.execute_text(
        391,
        "INSERT INTO dd (id, d) VALUES (1, '2026-07-02'), (2, '2026-07-01')",
    )
    .unwrap();
    let dd_table = e.relational_catalog_table("dd").unwrap();
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
            let days = gpu_db_sql::datetime::parse_date("2026-07-02").unwrap();
            assert_eq!(
                device_row,
                vec![SqlValue::Int4(1), SqlValue::Date(days)],
                "typed device row == closed-form date row"
            );
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
