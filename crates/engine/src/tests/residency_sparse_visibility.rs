/// SV2 (sparse-versioning): a shard is born DELETE-FREE and carries NO `deleted_by` region — the HyPer
/// "un-versioned rows pay nothing" property. Across admission + in-place append + rollover, NO shard has a
/// tombstone region until a DELETE touches it (SV4). NON-VACUITY: the table is really multiple shards
/// (rollover) and EVERY one has no region (a regression that eagerly allocated would fail this), while the
/// reads are still correct (all 200 rows live).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shard_born_delete_free_carries_no_region() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64); // small -> admission + in-place append + rollover all exercised
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
    let shard_ids: Vec<(u32, usize)> = {
        let shards = e.read_state.residency.shards.load();
        let table_shards = shards.get("accounts").unwrap();
        assert!(
            table_shards.len() >= 3,
            "need rollover into multiple shards (got {})",
            table_shards.len()
        );
        table_shards
            .iter()
            .map(|s| (s.shard_id, s.row_count))
            .collect()
    };
    for (shard_id, row_count) in shard_ids {
        assert!(
            read_shard_deleted_by_region(&e, "accounts", shard_id, row_count).is_none(),
            "a delete-free shard {shard_id} must carry NO deleted_by region (zero version overhead)"
        );
    }
    assert_eq!(
        e.execute_relational_select_text("SELECT COUNT(*) FROM accounts")
            .unwrap()
            .rows
            .row(0),
        &[SqlValue::Int8(200)],
        "all rows live (no tombstones)"
    );
}

/// SV2 (incremental DELETE write): the tombstone primitive ALLOCATES the shard's `deleted_by` region on
/// its FIRST delete (delete-free shards pay zero) + stamps `deleted_by[slot] = commit_seq` there,
/// OUT-OF-LINE (row column bytes untouched — the read visibility filter is SV3, so at SV2 a tombstoned row
/// STILL reads). NON-VACUITY: no region before the first delete; only the targeted slots flip; the region
/// is REUSED (not re-allocated) on a second delete; a full scan still returns all 5 rows byte-intact.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shard_tombstone_allocates_region_and_stamps_out_of_line() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance) VALUES (1,10),(2,20),(3,30),(4,40),(5,50)",
    )
    .unwrap();
    let shard_id = e
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .unwrap()
        .iter()
        .find(|shard| shard.row_count > 0)
        .expect("accounts has a non-empty shard")
        .shard_id;

    // Before any delete: NO region (the zero-cost property).
    assert!(
        read_shard_deleted_by_region(&e, "accounts", shard_id, 5).is_none(),
        "a delete-free shard has no deleted_by region"
    );

    // First delete: allocates the region + stamps slots 1 and 3 with seq 777.
    assert!(
        e.tombstone_resident_shard_slots("accounts", shard_id, &[1, 3], 777),
        "first tombstone allocates the region + stamps"
    );
    let db = read_shard_deleted_by_region(&e, "accounts", shard_id, 5)
        .expect("region is allocated on the first delete");
    assert_eq!(
        db,
        vec![0x7F7F_7F7F_7F7F_7F7F, 777, 0x7F7F_7F7F_7F7F_7F7F, 777, 0x7F7F_7F7F_7F7F_7F7F],
        "only the targeted slots are stamped; other rows stay the live sentinel (0x7F7F.. = signed-safe)"
    );

    // Out-of-line: column bytes untouched -> (visibility unwired at SV2) a full scan still returns 5 rows.
    let rows = e
        .execute_relational_select_text("SELECT id, balance FROM accounts")
        .unwrap()
        .rows;
    assert_eq!(
        rows.len(),
        5,
        "columns intact: all rows still read (visibility is SV3)"
    );
    let mut seen: Vec<(i32, i32)> = (0..rows.len())
        .map(|i| match (&rows.row(i)[0], &rows.row(i)[1]) {
            (SqlValue::Int4(id), SqlValue::Int4(bal)) => (*id, *bal),
            other => panic!("unexpected row shape {other:?}"),
        })
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);

    // Second delete on the SAME shard REUSES the region (no re-alloc) and stamps another slot.
    assert!(e.tombstone_resident_shard_slots("accounts", shard_id, &[0], 888));
    assert_eq!(
        read_shard_deleted_by_region(&e, "accounts", shard_id, 5).unwrap(),
        vec![888, 777, 0x7F7F_7F7F_7F7F_7F7F, 777, 0x7F7F_7F7F_7F7F_7F7F],
        "the second delete reuses the region + preserves the earlier stamps"
    );

    // Bounds: an out-of-range slot is rejected.
    assert!(
        !e.tombstone_resident_shard_slots("accounts", shard_id, &[5], 777),
        "slot == row_count (headroom) must be rejected"
    );
}

/// SV4 prerequisite #1 (lifecycle/leak, audit-flagged): the on-demand `deleted_by` region a tombstone
/// allocates MUST be released whenever the resident buffer it annotates is invalidated (an invalidating
/// commit -> O(table) re-admit) or dropped -- otherwise a re-admit rebuilds the shard ALL-LIVE from the
/// host store yet inherits a STALE tombstone region (wrong-results: rows wrongly hidden), and DROP TABLE
/// leaks the tombstone device buffers. NON-VACUITY: the region is proven PRESENT first, then proven GONE
/// after each lifecycle event, with an end-to-end read confirming the buffer really is fresh + all-live.
/// Path B (DROP) is specific to the `apply_drop_table` `remove_table` erase (KEY absence). NOTE: Path A's
/// DELETE fires BOTH the commit invalidate mirror AND the auto-admit re-admit `remove_table`, so the
/// re-admit MASKS the invalidate mirror here -- `shard_deleted_by_region_released_by_invalidate_alone`
/// (auto-admit OFF) isolates the serialized invalidate mirror; the sharded re-admit + eviction-cleanup
/// mirrors have their own isolated gates.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shard_deleted_by_region_released_on_invalidate_and_drop() {
    // --- Path A: explicit repair + invalidation + admission releases the region ---
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance) VALUES (1,10),(2,20),(3,30)",
    )
    .unwrap();
    let shard_id = e
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .unwrap()
        .iter()
        .find(|shard| shard.row_count > 0)
        .expect("accounts has a non-empty shard")
        .shard_id;
    // The SV2 primitive allocates the region on this first tombstone.
    assert!(e.tombstone_resident_shard_slots("accounts", shard_id, &[1], 777));
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "precondition: the tombstone allocated a live deleted_by region"
    );
    repair_test_relational_host_copy(&e, "accounts");
    invalidate_test_relational_residency(&e, "accounts");
    e.populate_relational_residency_snapshot_shared("accounts")
        .unwrap();
    assert!(
        !table_has_any_deleted_by_cell(&e, "accounts"),
        "invalidate/re-admit must release the stale deleted_by region (leak + wrong-results guard)"
    );
    // End-to-end: the rebuilt buffer reads all three repaired rows with no stale hide.
    let rows = e
        .execute_relational_select_text("SELECT id FROM accounts")
        .unwrap()
        .rows;
    let mut ids: Vec<i32> = (0..rows.len())
        .map(|i| match &rows.row(i)[0] {
            SqlValue::Int4(id) => *id,
            other => panic!("unexpected row shape {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![1, 2, 3],
        "all repaired rows are live after the explicit rebuild"
    );

    // --- Path B: DROP TABLE releases the region ---
    e.execute_text(4, "INSERT INTO accounts (id, balance) VALUES (7,70)")
        .unwrap();
    let shard_id2 = e
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .unwrap()
        .iter()
        .find(|shard| shard.row_count > 0)
        .expect("accounts has a non-empty shard")
        .shard_id;
    assert!(e.tombstone_resident_shard_slots("accounts", shard_id2, &[0], 888));
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "precondition: the region is re-allocated on the post-re-admit shard"
    );
    e.execute_text(5, "DROP TABLE accounts").unwrap();
    // DROP fully ERASES the cell entries (invalidate alone would leave a dangling `None` key per dropped
    // table). Asserting KEY absence (not just Some absence) makes this specific to `remove_table`.
    assert!(
        !table_has_any_deleted_by_key(&e, "accounts"),
        "DROP TABLE must erase the deleted_by cell entries (no leaked per-table keys / device memory)"
    );
}

/// SV4 prereq #1 (round-2 audit P3 hardening): ISOLATE the serialized-commit `invalidate_table` mirror.
/// With AUTO-ADMIT OFF, a DELETE invalidates residency but triggers NO re-admit, so the region is released
/// SOLELY by the commit-path `invalidate_relational_residency_table` deleted_by mirror -- nothing masks it
/// (unlike `..._on_invalidate_and_drop`, where the re-admit's `remove_table` would hide a deleted mirror).
/// Sabotage: delete ONLY the serialized `shard_deleted_by_memory.invalidate_table` (engine_commit.rs) and
/// this FAILS. This is the exact production DELETE path SV4 will build on.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shard_deleted_by_region_released_by_invalidate_alone() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance) VALUES (1,10),(2,20),(3,30)",
    )
    .unwrap();
    let shard_id = e
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .unwrap()
        .iter()
        .find(|shard| shard.row_count > 0)
        .expect("accounts has a non-empty shard")
        .shard_id;
    assert!(e.tombstone_resident_shard_slots("accounts", shard_id, &[1], 777));
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "precondition: the tombstone allocated a live deleted_by region"
    );
    invalidate_test_relational_residency(&e, "accounts");
    assert!(
        !table_has_any_deleted_by_cell(&e, "accounts"),
        "the serialized-commit invalidate mirror must release the region even with no re-admit"
    );
}

/// SV4 prereq #1 (audit Finding 1): `RelationalResidentCache::remove_table` (the BUDGET-EVICTION cleanup)
/// must release the table's `deleted_by` regions. This is DEFENSIVE today: the eviction loop draws its
/// candidates ONLY from the single-buffer `snapshots` map (engine_residency.rs, the `candidates` filter),
/// and a region-bearing table is by construction SHARD-resident (removed from `snapshots`), so `remove_table`
/// is currently only ever called on region-free tables. But it is the exact call a future shard-eviction
/// will make, so we test the METHOD CONTRACT directly: a shard-resident, tombstoned table passed to
/// `remove_table` has its region ERASED. NON-VACUITY: region present, then KEY-absent. Sabotage: remove the
/// `shard_deleted_by_memory.remove_table` in `RelationalResidentCache::remove_table` and this FAILS.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_cache_remove_table_releases_deleted_by_region() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1,10),(2,20)")
        .unwrap();
    let shard_id = e
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .unwrap()
        .iter()
        .find(|shard| shard.row_count > 0)
        .expect("accounts has a non-empty shard")
        .shard_id;
    assert!(e.tombstone_resident_shard_slots("accounts", shard_id, &[0], 5));
    assert!(
        table_has_any_deleted_by_cell(&e, "accounts"),
        "precondition: the tombstone allocated a live deleted_by region"
    );
    // Invoke the cache eviction-cleanup method DIRECTLY -- the exact call the budget-eviction loop makes
    // (`cat.relational_resident_cache.remove_table(map_key, &residency, &route_telemetry)`).
    {
        let guard = e.ddl_catalog();
        guard.relational_resident_cache.remove_table(
            "accounts",
            &e.read_state.residency,
            &e.read_state.route_telemetry,
        );
    }
    assert!(
        !table_has_any_deleted_by_key(&e, "accounts"),
        "RelationalResidentCache::remove_table must erase the table's deleted_by regions (eviction cleanup)"
    );
}

/// SV3b (MVCC read visibility): once a shard is tombstoned (the SV2 primitive), the SHARDED read path
/// HIDES the tombstoned rows -- the on-device predicate ANDs `deleted_by > read_txn_id` over a co-resident
/// i64 `deleted_by` column gathered into the unified buffer (memset to the all-live sentinel, then the
/// versioned shard's live prefix DtoD-copied over). Gate: a point lookup for a tombstoned key returns
/// EMPTY; COUNT(*) drops by exactly the tombstone count; a LIVE neighbor in the SAME now-versioned shard
/// still reads (the live sentinel passes the SIGNED compare -- guards the fill-vs-stamp boundary); a point
/// lookup pruned to a DIFFERENT, un-tombstoned shard is byte-identical (no deleted_by region -> the `None`
/// visibility path). This is the host-MVCC visibility semantics enforced ENTIRELY on the GPU.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shard_visibility_filter_hides_tombstoned_rows() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64); // 200 rows -> shards 64,64,64,8; id=k sits in shard (k/64) at slot (k%64)
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
    let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;

    // Baseline (delete-free): every shard is un-versioned -> the read takes the `None` visibility path.
    assert_eq!(
        sel("SELECT id FROM accounts WHERE id = 0").len(),
        1,
        "id=0 present pre-delete"
    );
    assert_eq!(
        sel("SELECT COUNT(*) FROM accounts").row(0),
        &[SqlValue::Int8(200)]
    );

    // Tombstone id=0 (shard 0, slot 0) at a commit seq well below the read snapshot so
    // `deleted_by(=5) > read_txn_id` is FALSE and the row is hidden.
    let shard0 = e
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .unwrap()
        .iter()
        .find(|shard| shard.row_count > 0)
        .expect("accounts has a non-empty shard")
        .shard_id;
    assert!(
        e.tombstone_resident_shard_slots("accounts", shard0, &[0], 5),
        "tombstone id=0 at slot 0"
    );

    // (1) the tombstoned key is now INVISIBLE to a point lookup (gathers the versioned shard 0).
    assert_eq!(
        sel("SELECT id FROM accounts WHERE id = 0").len(),
        0,
        "tombstoned id=0 hidden by the on-device visibility filter"
    );

    // (2) COUNT(*) over ALL shards drops by exactly one (deleted_by built for the whole unified buffer;
    //     the un-versioned shards' rows are the all-live memset fill).
    assert_eq!(
        sel("SELECT COUNT(*) FROM accounts").row(0),
        &[SqlValue::Int8(199)],
        "COUNT reflects the single tombstone"
    );

    // (3) a LIVE neighbor in the SAME now-versioned shard still reads -- the live sentinel passes the
    //     visibility compare (guards the fill-vs-tombstone boundary + the signed-safe sentinel).
    let n1 = sel("SELECT id, balance FROM accounts WHERE id = 1");
    assert_eq!(
        n1.len(),
        1,
        "live neighbor id=1 in the versioned shard still visible"
    );
    assert_eq!(n1.row(0), &[SqlValue::Int4(1), SqlValue::Int4(10)]);

    // (4) a point lookup pruned to a DIFFERENT, un-tombstoned shard is unaffected (no region -> `None`).
    let far = sel("SELECT id, balance FROM accounts WHERE id = 137");
    assert_eq!(far.len(), 1, "un-tombstoned shard unaffected");
    assert_eq!(far.row(0), &[SqlValue::Int4(137), SqlValue::Int4(1370)]);
}

/// SV4 (GPU-native DELETE, locate+tombstone data-plane primitive): `try_tombstone_resident_delete`
/// LOCATES a row by an int4-equality predicate (zone-map-pruned, per-shard) and stamps its `deleted_by`
/// -- so the SV3b read HIDES exactly that row, with NO O(table) re-admit. This is the mechanism SV4b wires
/// into the DELETE commit. NON-VACUITY / correctness of LOCATE: a wrong slot would hide the WRONG row, so
/// the neighbor-still-visible + COUNT-drops-by-exactly-one + other-shard-untouched asserts fail unless
/// locate returns the EXACT slot. Multi-shard (size 64) exercises zone-map pruning + cross-shard locate.
/// Delete-free byte-identical is proven by the pre-delete COUNT + the untouched rows post-delete.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_delete_locate_and_tombstone_hides_exactly_the_matched_row() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64); // 200 rows -> shards 64,64,64,8; id=k in shard k/64 at slot k%64
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
    let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;
    // Pre-delete: delete-free reads are byte-identical (all 200 rows live, none hidden).
    assert_eq!(
        sel("SELECT COUNT(*) FROM accounts").row(0),
        &[SqlValue::Int8(200)]
    );
    assert_eq!(
        sel("SELECT id FROM accounts WHERE id = 130").len(),
        1,
        "id=130 present pre-delete"
    );

    // Build the point predicate `id = 130` (id is catalog column 0) and DELETE it via the GPU primitive.
    let table = e.relational_catalog_table("accounts").unwrap();
    let id_col = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let pred = crate::engine_expr::ResidentExpr::Binary {
        op: crate::engine_expr::ResidentBinaryOp::Eq,
        lhs: Box::new(crate::engine_expr::ResidentExpr::Column(id_col)),
        rhs: Box::new(crate::engine_expr::ResidentExpr::Int4Literal(130)),
    };
    // commit_seq 5 (well below the read snapshot) -> `deleted_by(=5) > read_txn_id` is FALSE -> hidden.
    let n = e
        .try_tombstone_resident_delete(&table, &pred, 5)
        .expect("resident, prunable point delete succeeds");
    assert_eq!(n, 1, "exactly one resident row matched id=130");

    // The matched row is now hidden; its neighbors + other shards are UNTOUCHED (proves the RIGHT slot).
    assert_eq!(
        sel("SELECT id FROM accounts WHERE id = 130").len(),
        0,
        "id=130 tombstoned -> hidden"
    );
    assert_eq!(
        sel("SELECT id FROM accounts WHERE id = 129").len(),
        1,
        "same-shard neighbor 129 still visible"
    );
    assert_eq!(
        sel("SELECT id FROM accounts WHERE id = 131").len(),
        1,
        "same-shard neighbor 131 still visible"
    );
    assert_eq!(
        sel("SELECT id FROM accounts WHERE id = 5").len(),
        1,
        "a row in a DIFFERENT shard untouched"
    );
    assert_eq!(
        sel("SELECT COUNT(*) FROM accounts").row(0),
        &[SqlValue::Int8(199)],
        "COUNT drops by exactly the one tombstoned row"
    );
}
