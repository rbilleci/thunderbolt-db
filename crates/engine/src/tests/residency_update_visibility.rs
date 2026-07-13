/// SV4b (GPU-native incremental DELETE, commit WIRING): with `resident_delete_tombstone_enabled` ON, a
/// single-row SQL DELETE on a shard-resident table LOCATES + tombstones the row's slot IN PLACE (no
/// O(table) re-admit) and the GPU read == host MVCC. NON-VACUITY: the deleted_by region EXISTING after
/// the DELETE proves the tombstone route ran (a re-admit fallback rebuilds ALL-LIVE => NO region), while
/// the flag-OFF control gives the IDENTICAL result via re-admit (NO region). A MULTI-ROW DELETE falls
/// back to re-admit (region cleared) and is still correct -- the exact-count safety net.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv4b_sql_delete_tombstones_in_place_and_matches_host_mvcc() {
    let load = |e: &Engine| {
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64); // 200 rows -> shards 64,64,64,8
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
    let count = |e: &Engine| match e
        .execute_relational_select_text("SELECT COUNT(*) FROM accounts")
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("unexpected COUNT shape {other:?}"),
    };
    let present = |e: &Engine, id: i64| {
        !e.execute_relational_select_text(&format!("SELECT id FROM accounts WHERE id = {id}"))
            .unwrap()
            .rows
            .is_empty()
    };

    // --- flag ON: the single-row DELETE routes through the in-place tombstone ---
    let e = Engine::new_local();
    // PINNED NON-ELIDED (A5 flip): the oracle reads the HOST store / pins pre-elision mechanics (production-live for non-eligible tables).
    e.set_host_install_elision_enabled(false);
    e.set_resident_delete_tombstone_enabled(true);
    load(&e);
    assert!(
        !table_has_any_deleted_by_cell(&e, "accounts"),
        "delete-free: no region"
    );
    assert_eq!(count(&e), 200);

    e.execute_text(202, "DELETE FROM accounts WHERE id = 130")
        .unwrap();
    // NON-VACUITY: the tombstone path ran (region allocated). A re-admit fallback would leave NO region.
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "single-row DELETE routed through the in-place tombstone (region allocated)"
    );
    assert!(
        !present(&e, 130),
        "id=130 deleted -> hidden on the GPU route"
    );
    assert!(
        present(&e, 129) && present(&e, 131),
        "same-shard neighbors still visible"
    );
    assert!(present(&e, 5), "a row in a different shard untouched");
    assert_eq!(count(&e), 199, "COUNT drops by exactly one (== host MVCC)");

    // RETIREMENT A4b: a MULTI-ROW DELETE (2 rows) is now INCREMENTAL (per-row exact-1
    // locate+tombstone) — the region stays LIVE with both slots stamped, no re-admit.
    e.execute_text(203, "DELETE FROM accounts WHERE id = 50 OR id = 51")
        .unwrap();
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "multi-row DELETE must stay incremental (region live, A4b)"
    );
    assert!(
        !present(&e, 50) && !present(&e, 51),
        "multi-row DELETE removed both rows"
    );
    assert!(
        !present(&e, 130),
        "the earlier single-row delete stays deleted (host store)"
    );
    assert_eq!(count(&e), 197, "COUNT == host MVCC after 3 total deletes");

    // --- flag OFF control: the SAME single-row DELETE via re-admit -> identical result, NO region ---
    let c = Engine::new_local();
    // PINNED NON-ELIDED (A5 flip): the oracle reads the HOST store / pins pre-elision mechanics (production-live for non-eligible tables).
    c.set_host_install_elision_enabled(false);
    c.set_resident_delete_tombstone_enabled(false); // THE FLIP: the control pins the re-admit path
    load(&c);
    c.execute_text(202, "DELETE FROM accounts WHERE id = 130")
        .unwrap();
    assert!(
        !table_has_any_deleted_by_cell(&c, "accounts"),
        "flag OFF: DELETE re-admits (all-live) -> no region"
    );
    assert!(!present(&c, 130), "control: id=130 deleted");
    assert_eq!(
        count(&c),
        199,
        "control: COUNT 199 == the flag-ON result (byte-identical semantics)"
    );
}

/// SV5 (GPU-native incremental UPDATE, commit WIRING): with `resident_update_tombstone_enabled` ON, a
/// single-row SQL UPDATE on a shard-resident table TOMBSTONES the old version's slot + APPENDS the new
/// image IN PLACE (no O(table) re-admit) and the GPU read == host MVCC. NON-VACUITY: the deleted_by region
/// EXISTING after the UPDATE proves the tombstone-old route ran (re-admit fallback rebuilds ALL-LIVE => NO
/// region); the read returns the NEW value; COUNT is unchanged (old hidden + new visible); the OLD value is
/// hidden; a MULTI-ROW UPDATE falls back to re-admit (correct); flag-OFF control identical.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv5_sql_update_tombstones_old_appends_new_matches_host_mvcc() {
    let load = |e: &Engine| {
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
    };
    let count = |e: &Engine| match e
        .execute_relational_select_text("SELECT COUNT(*) FROM accounts")
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("unexpected COUNT shape {other:?}"),
    };
    let balance_of = |e: &Engine, id: i64| -> Option<i32> {
        let rows = e
            .execute_relational_select_text(&format!(
                "SELECT id, balance FROM accounts WHERE id = {id}"
            ))
            .unwrap()
            .rows;
        if rows.is_empty() {
            return None;
        }
        match &rows.row(0)[1] {
            SqlValue::Int4(b) => Some(*b),
            other => panic!("unexpected row shape {other:?}"),
        }
    };
    let old_balance_visible = |e: &Engine| {
        // The OLD (id=130, balance=1300) image must be HIDDEN: a lookup by the old balance finds nothing.
        !e.execute_relational_select_text("SELECT id FROM accounts WHERE balance = 1300")
            .unwrap()
            .rows
            .is_empty()
    };

    // --- flag ON: the single-row UPDATE routes through tombstone-old + append-new ---
    let e = Engine::new_local();
    // PINNED NON-ELIDED (A5 flip): the oracle reads the HOST store / pins pre-elision mechanics (production-live for non-eligible tables).
    e.set_host_install_elision_enabled(false);
    e.set_resident_update_tombstone_enabled(true);
    load(&e);
    assert!(
        !table_has_any_deleted_by_cell(&e, "accounts"),
        "no region pre-update"
    );
    assert_eq!(count(&e), 200);
    assert_eq!(balance_of(&e, 130), Some(1300), "pre-update balance");

    e.execute_text(202, "UPDATE accounts SET balance = 9999 WHERE id = 130")
        .unwrap();
    // NON-VACUITY: the tombstone-old path ran (region allocated). Re-admit fallback would leave NO region.
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "single-row UPDATE routed through tombstone-old + append-new (region allocated)"
    );
    assert_eq!(
        balance_of(&e, 130),
        Some(9999),
        "id=130 reads the NEW balance (appended version)"
    );
    assert!(
        !old_balance_visible(&e),
        "the OLD (id=130,balance=1300) image is hidden"
    );
    assert_eq!(
        balance_of(&e, 131),
        Some(1310),
        "same-shard neighbor untouched"
    );
    assert_eq!(
        balance_of(&e, 5),
        Some(50),
        "a row in a different shard untouched"
    );
    assert_eq!(
        count(&e),
        200,
        "COUNT unchanged (old hidden + new visible) == host MVCC"
    );

    // An int4-UNCHANGED update (same-value: id=5 already has balance 5*10=50) still routes: tombstone-OLD
    // FIRST locates the old slot on the buffer BEFORE the identical-int4 new row is appended (count 1), so
    // it tombstones the OLD slot, not the new. Exercises the order-sensitivity the value-changing case can't.
    e.execute_text(203, "UPDATE accounts SET balance = 50 WHERE id = 5")
        .unwrap();
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "same-value UPDATE still routes through tombstone-old + append-new"
    );
    assert_eq!(
        balance_of(&e, 5),
        Some(50),
        "id=5 still reads 50 (old hidden, new appended, same value)"
    );
    assert_eq!(
        count(&e),
        200,
        "COUNT unchanged after the int4-unchanged update"
    );

    // RETIREMENT A4b: a MULTI-ROW UPDATE (2 rows) is now INCREMENTAL (tombstones + one
    // batched identity-stamped append) — the region stays LIVE, no re-admit.
    e.execute_text(
        204,
        "UPDATE accounts SET balance = 0 WHERE id = 10 OR id = 11",
    )
    .unwrap();
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "multi-row UPDATE must stay incremental (region live, A4b)"
    );
    assert_eq!(balance_of(&e, 10), Some(0));
    assert_eq!(balance_of(&e, 11), Some(0));
    assert_eq!(
        balance_of(&e, 130),
        Some(9999),
        "single-row update persists across the re-admit"
    );
    assert_eq!(count(&e), 200);

    // --- flag OFF control: the SAME single-row UPDATE via re-admit -> identical result, NO region ---
    let c = Engine::new_local();
    // PINNED NON-ELIDED (A5 flip): the oracle reads the HOST store / pins pre-elision mechanics (production-live for non-eligible tables).
    c.set_host_install_elision_enabled(false);
    c.set_resident_update_tombstone_enabled(false); // THE FLIP: the control pins the re-admit path
    load(&c);
    c.execute_text(202, "UPDATE accounts SET balance = 9999 WHERE id = 130")
        .unwrap();
    assert!(
        !table_has_any_deleted_by_cell(&c, "accounts"),
        "flag OFF: UPDATE re-admits (all-live) -> no region"
    );
    assert_eq!(balance_of(&c, 130), Some(9999), "control: new balance");
    assert_eq!(
        count(&c),
        200,
        "control: COUNT 200 == the flag-ON result (byte-identical semantics)"
    );
}

/// SV6 (`created_by` SI flip-gate) — the DOUBLE-READ differential, deterministic torn-window form.
/// The SV5 incremental UPDATE appends the new version + bumps `row_count` BEFORE `publish_committed_seq`,
/// and a lock-free reader binds `read_txn_id = committed_seq()` THEN loads shards — so a reader that
/// observes `committed_seq = C-1` while the shards ALREADY carry the appended row is the torn window the
/// SV5 audit flagged (P2). This test constructs that window EXACTLY: it applies the incremental UPDATE at
/// `commit_seq = C0+1` directly (the same call the commit path makes) WITHOUT publishing, then reads.
/// SNAPSHOT-CORRECT (the `created_by` gate): the C-1 reader sees the key EXACTLY ONCE, with the OLD image
/// (old visible: `deleted_by = C0+1 > C0`; new hidden: `created_by = C0+1 > C0`); COUNT is unchanged.
/// THE PRE-FIX BUG: the key TWICE (old + new — a state that never existed). After publish, a reader at C
/// sees exactly the NEW image (old hidden: `deleted_by = C0+1 <= C0+1`; new visible: `created_by <= C0+1`).
/// SABOTAGE-VERIFIED: skip the `created_by` stamp on append (or drop the VM conjunct) and this FAILS.
/// Derive the REAL row identity for a unique int4 key via the device locate + A1 region —
/// what the production commit arm surfaces from the installs' keys (A4b made identity
/// MANDATORY on the incremental update path, so direct `try_update_resident_commit` callers
/// must pass the true ids).
fn device_row_id_for(e: &Engine, table_name: &str, key: i32) -> u64 {
    let table = e.relational_catalog_table(table_name).unwrap();
    let hits = e
        .locate_resident_pk_via_shard_index_detailed(&table, 0, key)
        .expect("locate must answer for a unique resident key");
    let hit = hits.first().expect("at least one slot");
    let region = hit.row_id.as_ref().expect("identity region present");
    let halves = region
        .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
        .unwrap();
    (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32)
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv6_created_by_gate_reader_at_prior_snapshot_never_sees_updated_key_twice() {
    // Run the torn-window differential over BOTH stamp branches: 200 rows -> the open shard has
    // headroom, the append stamps IN PLACE; 256 rows (= 64*4 under the E2.5b-2 first-capacity
    // FLOOR clamped by the target: EVERY shard is born at the 64-row target, so 4 exactly-full
    // shards) -> the open shard is FULL, the append ROLLS OVER a new stamped shard (whose
    // created_by region must install before the shard publishes). The branch actually taken is
    // PROVEN structurally below (shard-count delta), so neither variant can go vacuous if the
    // admit shape changes. (Recalibrated from 258: the pre-floor 1-row admit built a capacity-2
    // shard 0; the floor commit 6c3fe683 made shard 0 birth at the target too.)
    for total_rows in [200_i64, 256_i64] {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..total_rows {
            e.execute_text(
                (i as u64) + 2,
                &format!(
                    "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                    i * 10
                ),
            )
            .unwrap();
        }
        let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;
        let c0 = e.committed_seq();
        assert_eq!(
            sel("SELECT id, balance FROM accounts WHERE id = 130").len(),
            1,
            "pre-update: one row"
        );
        let shard_count = |e: &Engine| {
            e.read_state
                .residency
                .shards
                .load()
                .get("accounts")
                .map_or(0, |s| s.len())
        };
        let shards_before = shard_count(&e);

        // Apply the incremental UPDATE (tombstone-old + append-new) at commit_seq C0+1 WITHOUT
        // publishing — exactly the state a concurrent reader can observe between the residency
        // maintenance and `publish_committed_seq` inside a real commit.
        {
            let id_130 = device_row_id_for(&e, "accounts", 130);
            let guard = e.ddl_catalog();
            let ok = e.try_update_resident_commit(
                &guard,
                "accounts",
                &[vec![SqlValue::Int4(130), SqlValue::Int4(1300)]],
                &[vec![SqlValue::Int4(130), SqlValue::Int4(9999)]],
                c0 + 1,
                Some(&[id_130]),
            );
            assert!(
                ok,
                "the incremental tombstone-old + append-new route must fire at {total_rows} rows \
                 (else this test is vacuous)"
            );
        }
        // NON-VACUITY (route proof): the append STAMPED a created_by region (fallback re-admit / an
        // unstamped append would leave none — and the reads below would then double-count).
        assert!(
            table_has_any_created_by_cell(&e, "accounts"),
            "the UPDATE append must have stamped a created_by region at {total_rows} rows"
        );
        // NON-VACUITY (branch proof): 200 rows must exercise the IN-PLACE stamp (same shard set);
        // 256 rows must exercise the ROLLOVER stamp (a new shard appeared). If the admit shape ever
        // changes these row counts, this assert flags the variant instead of silently going vacuous.
        if total_rows == 200 {
            assert_eq!(
                shard_count(&e),
                shards_before,
                "200 rows: the in-place branch must serve"
            );
        } else {
            assert_eq!(
                shard_count(&e),
                shards_before + 1,
                "{total_rows} rows: the ROLLOVER branch must serve (open shard full)"
            );
        }

        // The C-1 reader (committed_seq is still C0): EXACTLY ONE row, the OLD image.
        let rows = sel("SELECT id, balance FROM accounts WHERE id = 130");
        assert_eq!(
            rows.len(),
            1,
            "SI at {total_rows} rows: a reader at committed_seq C-1 must see the updated key EXACTLY \
             ONCE (2 = the SV5 P2 double-read: old visible via deleted_by > C-1 AND new visible with \
             no created_by gate)"
        );
        assert_eq!(
            rows.row(0),
            &[SqlValue::Int4(130), SqlValue::Int4(1300)],
            "the C-1 snapshot reads the OLD image (the appended new version is not yet visible)"
        );
        assert_eq!(
            sel("SELECT COUNT(*) FROM accounts").row(0),
            &[SqlValue::Int8(total_rows)],
            "COUNT at C-1 is snapshot-correct (no phantom appended row)"
        );

        // Publish the commit: a reader at C sees exactly the NEW image, once.
        e.publish_committed_seq(c0 + 1);
        let rows = sel("SELECT id, balance FROM accounts WHERE id = 130");
        assert_eq!(rows.len(), 1, "post-publish: exactly one row");
        assert_eq!(
            rows.row(0),
            &[SqlValue::Int4(130), SqlValue::Int4(9999)],
            "a reader at C sees the NEW image (old hidden by deleted_by, new admitted by created_by)"
        );
        assert_eq!(
            sel("SELECT COUNT(*) FROM accounts").row(0),
            &[SqlValue::Int8(total_rows)]
        );
    }
}

/// SV6 — the CONCURRENT-reader form of the double-read differential: a reader thread hammers the point
/// lookup while the writer commits real single-row SQL UPDATEs with `resident_update_tombstone_enabled`
/// ON. SI invariant under EVERY interleaving: the key appears EXACTLY ONCE per read (never 2 = the SV5
/// double-read; never 0 = a lost row). Crosses open-shard append headroom AND rollover (shard target 64,
/// ~300 appended versions), so both created_by stamp branches are exercised under load.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv6_concurrent_reader_never_sees_updated_key_twice_under_update_load() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_resident_delete_tombstone_enabled(true);
    e.set_resident_update_tombstone_enabled(true);
    // A5 FLIP: this hammer runs ELIDED BY DEFAULT — it is the regression gate for the
    // (fixed) elided-churn SI bug: a rehydrating decline used to leave the fallback on a
    // STALE view -> stale old image -> the tombstone stamped an already-dead slot -> the
    // current version leaked (double-read) or the update silently no-oped (lost update,
    // caught by the end-state assert below). Fix = re-pin the view at every
    // post-rehydration fallback.
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
                let rows = e
                    .execute_relational_select_text(
                        "SELECT id, balance FROM accounts WHERE id = 130",
                    )
                    .unwrap()
                    .rows;
                assert_eq!(
                    rows.len(),
                    1,
                    "SI under concurrency: id=130 must appear EXACTLY ONCE per read (2 = the SV5 \
                     double-read window; 0 = a lost row)"
                );
                assert_eq!(rows.row(0)[0], SqlValue::Int4(130));
                reads += 1;
            }
            reads
        });
        for t in 0..300_u64 {
            e.execute_text(
                300 + t,
                &format!(
                    "UPDATE accounts SET balance = {} WHERE id = 130",
                    100_000 + t
                ),
            )
            .unwrap();
        }
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let reads = reader
            .join()
            .expect("reader thread must not panic (SI violation = panic)");
        assert!(reads > 0, "the reader must have raced at least one read");
    });
    // Quiescent end-state: the last committed value, exactly once.
    let rows = e
        .execute_relational_select_text("SELECT id, balance FROM accounts WHERE id = 130")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0), &[SqlValue::Int4(130), SqlValue::Int4(100_299)]);
}

/// SV6 lifecycle (mirrors `shard_deleted_by_region_released_on_warmup_readmit`): a WARMUP/REFRESH
/// re-admit reaches the SHARDED re-admit branch with NO preceding commit invalidate, so it must itself
/// erase stale `created_by` regions — else the fresh all-live shard 0 (reused shard_id) inherits the
/// stamp region and wrongly HIDES rebuilt rows from older-snapshot readers. NON-VACUITY: region proven
/// present, then KEY-absent after the refresh. Sabotage: remove the sharded-branch
/// `shard_created_by_memory.remove_table` and this FAILS.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv6_created_by_region_released_on_warmup_readmit() {
    let mut e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance) VALUES (1,10),(2,20),(3,30)",
    )
    .unwrap();
    e.populate_relational_residency_snapshot("accounts")
        .unwrap();
    let (shard_id, capacity, gpu_id) = {
        let shards = e.read_state.residency.shards.load();
        let shard = &shards.get("accounts").unwrap()[0];
        (shard.shard_id, shard.capacity, shard.gpu_id)
    };
    assert!(e.stamp_created_by_resident_shard_slots(
        "accounts",
        shard_id,
        1,
        capacity,
        gpu_id,
        &[777]
    ));
    assert!(
        table_has_any_created_by_cell(&e, "accounts"),
        "precondition: the stamp allocated a live created_by region"
    );
    // Warmup/refresh re-admit -- NO commit, so NO invalidate precedes it.
    e.populate_relational_residency_snapshot("accounts")
        .unwrap();
    assert!(
        !table_has_any_created_by_key(&e, "accounts"),
        "warmup re-admit (no preceding invalidate) must erase the stale created_by region"
    );
}

/// SV6 lifecycle (mirrors SV4-prereq-#1 for `created_by`): the on-demand `created_by` region is
/// RELEASED at every site the buffer it annotates is retired — a re-admit (here: a multi-row UPDATE
/// falling back to invalidate + rebuild-all-live) must not leave a stale stamp region that would
/// wrongly HIDE rebuilt rows from older-snapshot readers, and DROP TABLE must erase the cell keys
/// entirely (no per-table host-cell leak). Sabotage: remove the `shard_created_by_memory` cleanup at
/// either site and the corresponding assert FAILS.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv6_created_by_region_released_on_readmit_and_drop() {
    let load = |e: &Engine| {
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
        e.execute_text(202, "UPDATE accounts SET balance = 9999 WHERE id = 130")
            .unwrap();
        assert!(
            table_has_any_created_by_cell(e, "accounts"),
            "precondition: the incremental UPDATE stamped a live created_by region"
        );
    };

    // RE-ADMIT gate: an AMBIGUOUS UPDATE falls back to invalidate + re-admit (rebuild all-live)
    // -> the region MUST go with the buffer it annotated, or the rebuilt rows would read a
    // stale stamp. A4b made plain multi-row UPDATEs INCREMENTAL, so the fallback trigger here
    // is int4-IDENTICAL duplicate rows: the per-row locate sees count 2 and declines (the
    // exact-count wrong-results net), forcing the re-admit this gate pins.
    // (The re-admitted table becomes ONE dense shard with no headroom, so no later single-row
    // UPDATE can re-stamp it — hence the separate fresh engine for the DROP gate below.)
    let e = Engine::new_local();
    load(&e);
    e.execute_text(
        203,
        "INSERT INTO accounts (id, balance) VALUES (900, 5), (900, 5)",
    )
    .unwrap();
    e.execute_text(204, "UPDATE accounts SET balance = 0 WHERE id = 900")
        .unwrap();
    assert!(
        !table_has_any_created_by_cell(&e, "accounts"),
        "re-admit must release the stale created_by region (wrong-results + leak guard)"
    );
    // Reads after the re-admit are the plain all-live scan (no phantom hiding).
    let rows = e
        .execute_relational_select_text("SELECT id, balance FROM accounts WHERE id = 130")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0), &[SqlValue::Int4(130), SqlValue::Int4(9999)]);

    // DROP gate (fresh engine, live stamped region): the cell KEYS must be erased (not just
    // tombstoned to `None`) — invalidate alone would leak a dangling key per dropped table.
    let d = Engine::new_local();
    load(&d);
    d.execute_text(203, "DROP TABLE accounts").unwrap();
    assert!(
        !table_has_any_created_by_key(&d, "accounts"),
        "DROP TABLE must erase the created_by cell entries (no leaked per-table keys / device memory)"
    );
}

/// SV6 — the created_by gate on the INDEX ROUTES (3b single-flight per-hit gate + the batched gather
/// gate + the GPU dense-emit DECLINE). The double-read shape can't reach the routes (a duplicated key
/// declines them to the scan), but a KEY-MOVING incremental UPDATE (`id 130 -> 999` at unpublished
/// `C0+1`) leaves the NEW key as a SINGLE stamped hit: a C-1 reader looking up 999 must get ZERO rows
/// (999 does not exist at its snapshot) while 130 still reads the OLD image — on the 3b route AND the
/// batched path (whose GPU dense kernel is un-gated and MUST decline the stamped shard to the gated
/// host gather). Post-publish, 999 is visible and 130 is gone. NON-VACUITY: `shard_index_route_hits` /
/// `sharded_point_batch_hits` prove the routes (not the scan) served. Sabotage: drop the per-hit
/// created_by check, the batched AND, or the dense-kernel decline — each makes 999 visible at C-1.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sv6_created_by_gate_on_index_routes_hides_moved_key_from_older_snapshot() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_index_probe_enabled(true);
    e.set_shard_batched_point_read_enabled(true);
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
    let c0 = e.committed_seq();
    // Move the key: UPDATE accounts SET id = 999 WHERE id = 130, applied at C0+1, UNPUBLISHED.
    {
        let id_130 = device_row_id_for(&e, "accounts", 130);
        let guard = e.ddl_catalog();
        let ok = e.try_update_resident_commit(
            &guard,
            "accounts",
            &[vec![SqlValue::Int4(130), SqlValue::Int4(1300)]],
            &[vec![SqlValue::Int4(999), SqlValue::Int4(1300)]],
            c0 + 1,
            Some(&[id_130]),
        );
        assert!(ok, "the incremental key-moving UPDATE must fire");
    }
    assert!(
        table_has_any_created_by_cell(&e, "accounts"),
        "stamp route proof"
    );
    let table = e.relational_catalog_table("accounts").unwrap();
    let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;

    // (a) 3b single-flight route: the NEW key is a single stamped hit -> the per-hit created_by gate
    // hides it (0 rows at C-1); the OLD key is a single tombstoned-at-C0+1 hit -> still visible.
    let route_hits_before = e.shard_index_route_hits();
    assert_eq!(
        sel("SELECT id, balance FROM accounts WHERE id = 999").len(),
        0,
        "3b route: the moved-to key must be HIDDEN from the C-1 reader (created_by gate)"
    );
    let rows = sel("SELECT id, balance FROM accounts WHERE id = 130");
    assert_eq!(rows.len(), 1, "3b route: the old key is still live at C-1");
    assert_eq!(rows.row(0), &[SqlValue::Int4(130), SqlValue::Int4(1300)]);
    assert!(
        e.shard_index_route_hits() > route_hits_before,
        "non-vacuity: the 3b index route (not the scan) served the C-1 point lookups"
    );

    // (b) Batched gather (the GPU dense kernel MUST decline the stamped shard -> gated host path):
    // needle 999 -> 0 rows; needle 130 -> the old image.
    let batch_hits_before = e.sharded_point_batch_hits();
    let batch = e
        .gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            0,
            &[0, 1],
            &[999, 130],
        )
        .expect("the batched sharded gather must serve (gated host path)");
    assert_eq!(batch.ncols, 2);
    assert_eq!(
        batch.needle_ranges[0].1, 0,
        "batched: the moved-to key must be HIDDEN from the C-1 reader (created_by gate)"
    );
    assert_eq!(
        batch.needle_ranges[1].1, 1,
        "batched: the old key is still live at C-1"
    );
    let start = batch.needle_ranges[1].0 as usize * 2;
    assert_eq!(&batch.values[start..start + 2], &[130, 1300]);
    assert!(
        e.sharded_point_batch_hits() > batch_hits_before,
        "non-vacuity: the batched path (not a fallback) served"
    );

    // (c) Publish -> a reader at C sees the move: 999 visible, 130 gone (both routes).
    e.publish_committed_seq(c0 + 1);
    let rows = sel("SELECT id, balance FROM accounts WHERE id = 999");
    assert_eq!(rows.len(), 1, "post-publish: the moved-to key is visible");
    assert_eq!(rows.row(0), &[SqlValue::Int4(999), SqlValue::Int4(1300)]);
    assert_eq!(
        sel("SELECT id FROM accounts WHERE id = 130").len(),
        0,
        "post-publish: the old key is gone"
    );
    let batch = e
        .gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            0,
            &[0, 1],
            &[999, 130],
        )
        .expect("batched gather post-publish");
    assert_eq!(
        batch.needle_ranges[0].1, 1,
        "batched post-publish: 999 visible"
    );
    assert_eq!(
        batch.needle_ranges[1].1, 0,
        "batched post-publish: 130 hidden"
    );
}
