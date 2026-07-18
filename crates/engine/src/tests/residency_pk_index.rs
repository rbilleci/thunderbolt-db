/// CROSS-SHARD PK INDEX sub-slice 1: the per-shard hash-index locate returns the IDENTICAL physical
/// (shard, LOCAL slot) the scan-based locate finds -- present keys across shards, absent keys (empty),
/// NULL-as-0 (id 0), and it DECLINES (None -> scan fallback) on a duplicate key. ORACLE = the proven SV4a
/// scan-based `locate_resident_delete_slots` (an INDEPENDENT mechanism: hash-probe vs scan-predicate, so
/// agreement is strong). NON-VACUITY: a wrong slot / missed shard / wrong decline diverges from the oracle;
/// a cross-check confirms exactly one hit per unique key. Multi-shard (size 64) exercises per-shard build.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cross_shard_pk_index_locate_matches_scan_locate() {
    let e = Engine::new_local();
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
    let table = e.relational_catalog_table("accounts").unwrap();
    let id_col = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    // Oracle: SV4a scan-based locate for `id = k`, flattened + sorted to (shard, slot).
    let scan_locate = |k: i32| -> Vec<(u32, u32)> {
        let pred = crate::engine_expr::ResidentExpr::Binary {
            op: crate::engine_expr::ResidentBinaryOp::Eq,
            lhs: Box::new(crate::engine_expr::ResidentExpr::Column(id_col)),
            rhs: Box::new(crate::engine_expr::ResidentExpr::Int4Literal(k)),
        };
        let mut v: Vec<(u32, u32)> = e
            .locate_resident_delete_slots(&table, &pred)
            .unwrap()
            .into_iter()
            .flat_map(|(shard, slots)| slots.into_iter().map(move |s| (shard, s)))
            .collect();
        v.sort_unstable();
        v
    };
    // Present keys across multiple shards: index locate == scan locate, exactly one hit each.
    for k in [0_i32, 5, 63, 64, 130, 199] {
        let mut idx = e
            .locate_resident_pk_via_shard_index(&table, id_col, k)
            .expect("resident + unique -> Some");
        idx.sort_unstable();
        assert_eq!(
            idx,
            scan_locate(k),
            "index locate == scan locate for id={k}"
        );
        assert_eq!(
            idx.len(),
            1,
            "unique key id={k} -> exactly one (shard,slot) hit"
        );
    }
    // Absent key: both empty.
    let mut absent = e
        .locate_resident_pk_via_shard_index(&table, id_col, 999)
        .expect("resident -> Some(empty)");
    absent.sort_unstable();
    assert_eq!(absent, scan_locate(999));
    assert!(absent.is_empty(), "absent key -> no hit");

    // DUP-DECLINE: a table with a duplicate int4 key -> the hash build declines -> None (scan fallback),
    // because a hash holds one row/key but the scan returns EVERY match.
    let d = Engine::new_local();
    d.set_shard_residency_enabled(true);
    d.set_auto_admit_on_commit(true);
    d.execute_text(1, "CREATE TABLE dup (id INT, balance INT)")
        .unwrap();
    d.execute_text(
        2,
        "INSERT INTO dup (id, balance) VALUES (1,10),(1,20),(2,30)",
    )
    .unwrap();
    let dtable = d.relational_catalog_table("dup").unwrap();
    let did = crate::rel_exec_helpers::relational_column_index(&dtable, "id").unwrap();
    assert!(
        d.locate_resident_pk_via_shard_index(&dtable, did, 1)
            .is_none(),
        "duplicate key -> hash declines -> None (caller falls back to the scan)"
    );
}

/// CROSS-SHARD PK INDEX sub-slice 3 (CACHE): the per-shard index cache is populated on first locate, and
/// on a GENERATION CHANGE (an explicit post-DELETE vacuum -> new device ptrs + SHIFTED row slots) the stale
/// cached index is NOT served -- ptr-validation rebuilds, so locate still == the scan on the NEW buffer.
/// This is the load-bearing cache-correctness gate: deleting id=50 moves id=51 from slot 51 to slot 50 in
/// shard 0, so a stale index would return the WRONG slot. Sabotage: drop the ptr check (serve stale) and
/// the post-re-admit locate diverges from the scan.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cross_shard_pk_index_cache_rebuilds_on_generation_change() {
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
    let table = e.relational_catalog_table("accounts").unwrap();
    let id_col = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let scan_locate = |t: &crate::relational_model::RelationalTable, k: i32| -> Vec<(u32, u32)> {
        let pred = crate::engine_expr::ResidentExpr::Binary {
            op: crate::engine_expr::ResidentBinaryOp::Eq,
            lhs: Box::new(crate::engine_expr::ResidentExpr::Column(id_col)),
            rhs: Box::new(crate::engine_expr::ResidentExpr::Int4Literal(k)),
        };
        let mut v: Vec<(u32, u32)> = e
            .locate_resident_delete_slots(t, &pred)
            .unwrap()
            .into_iter()
            .flat_map(|(s, slots)| slots.into_iter().map(move |x| (s, x)))
            .collect();
        v.sort_unstable();
        v
    };

    // Populate the cache (first locate builds + caches the per-shard indexes).
    assert_eq!(
        e.locate_resident_pk_via_shard_index(&table, id_col, 51)
            .unwrap()
            .len(),
        1
    );
    assert!(
        !e.read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty(),
        "the per-shard PK index cache is populated after a locate"
    );

    // GENERATION CHANGE: normal DELETE tombstones in place, then explicit vacuum rebuilds the
    // authoritative device image with new ptrs and shifts shard-0 rows (id=50 removed -> id=51
    // moves from slot 51 to slot 50).
    e.execute_text(202, "DELETE FROM accounts WHERE id = 50")
        .unwrap();
    e.vacuum_table("accounts").unwrap();
    let table2 = e.relational_catalog_table("accounts").unwrap();

    // The stale cached index (old ptr) must NOT be served: ptr-validation rebuilds against the new buffer.
    let mut after = e
        .locate_resident_pk_via_shard_index(&table2, id_col, 51)
        .unwrap();
    after.sort_unstable();
    assert_eq!(
        after,
        scan_locate(&table2, 51),
        "cache rebuilt on generation change -> locate == scan on the NEW buffer (no stale slot)"
    );
    assert_eq!(after.len(), 1, "id=51 still present (only id=50 deleted)");
    assert!(
        e.locate_resident_pk_via_shard_index(&table2, id_col, 50)
            .unwrap()
            .is_empty(),
        "id=50 is deleted -> not located"
    );
}

/// CROSS-SHARD PK INDEX sub-slice 3 (CACHE, in-place APPEND): an in-place open-shard INSERT grows the
/// shard's row_count with the SAME device ptr, so ptr-ONLY validation would serve a stale index MISSING
/// the appended key. The `(ptr, row_count)` validation rebuilds -> the appended key is located == scan.
/// Sabotage: drop the row_count check -> the stale ptr-hit misses the appended key. Fresh table (no
/// re-admit) so the INSERT is a genuine in-place append (same ptr), isolating the row_count check.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cross_shard_pk_index_cache_rebuilds_on_in_place_append() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64); // 200 rows -> shards 64/64/64/8; the last (open) shard is appendable
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
    let table = e.relational_catalog_table("accounts").unwrap();
    let id_col = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let scan_slots = |t: &crate::relational_model::RelationalTable, k: i32| -> Vec<(u32, u32)> {
        let pred = crate::engine_expr::ResidentExpr::Binary {
            op: crate::engine_expr::ResidentBinaryOp::Eq,
            lhs: Box::new(crate::engine_expr::ResidentExpr::Column(id_col)),
            rhs: Box::new(crate::engine_expr::ResidentExpr::Int4Literal(k)),
        };
        let mut v: Vec<(u32, u32)> = e
            .locate_resident_delete_slots(t, &pred)
            .unwrap()
            .into_iter()
            .flat_map(|(s, slots)| slots.into_iter().map(move |x| (s, x)))
            .collect();
        v.sort_unstable();
        v
    };
    // Populate the OPEN shard's cache entry (id=195 lives in the last/open shard).
    assert_eq!(
        e.locate_resident_pk_via_shard_index(&table, id_col, 195)
            .unwrap()
            .len(),
        1
    );
    // In-place append (id=250 -> the open shard grows by one row, SAME ptr, +row_count).
    e.execute_text(202, "INSERT INTO accounts (id, balance) VALUES (250, 2500)")
        .unwrap();
    let table2 = e.relational_catalog_table("accounts").unwrap();
    let mut appended = e
        .locate_resident_pk_via_shard_index(&table2, id_col, 250)
        .unwrap();
    appended.sort_unstable();
    assert_eq!(
        appended,
        scan_slots(&table2, 250),
        "appended key located == scan -> cache rebuilt on the row_count change (not a stale ptr-hit miss)"
    );
    assert_eq!(appended.len(), 1, "appended id=250 is located");
}

/// CROSS-SHARD PK INDEX sub-slice 3b (cache LIFECYCLE CLEANUP): the device index cache is PURGED for a
/// table on the residency-change lifecycle events (an explicit vacuum rebuild, and DROP), so a
/// wired index route can't leak the pinned shard buffers of a no-longer-resident table. Sabotage: make
/// `purge_shard_pk_index_for_table` a no-op and the post-VACUUM / post-DROP "cache empty" asserts FAIL.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cross_shard_pk_index_cache_purged_on_lifecycle() {
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
    let id_col = {
        let table = e.relational_catalog_table("accounts").unwrap();
        crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap()
    };
    let entries = |t: &str| {
        e.read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .filter(|(cached, _, _)| cached == t)
            .count()
    };
    let locate = |k: i32| {
        let table = e.relational_catalog_table("accounts").unwrap();
        e.locate_resident_pk_via_shard_index(&table, id_col, k)
            .unwrap()
    };

    // Populate the cache.
    assert_eq!(locate(130).len(), 1);
    assert!(entries("accounts") > 0, "cache populated after a locate");

    // Normal DELETE remains device-native; explicit vacuum performs the rebuild and purges the cache.
    e.execute_text(202, "DELETE FROM accounts WHERE id = 5")
        .unwrap();
    e.vacuum_table("accounts").unwrap();
    assert_eq!(
        entries("accounts"),
        0,
        "vacuum rebuild purged the cache (no leaked pinned buffers)"
    );

    // Re-populate, then DROP TABLE purges via apply_drop_table.
    assert_eq!(locate(130).len(), 1);
    assert!(entries("accounts") > 0, "cache re-populated");
    e.execute_text(203, "DROP TABLE accounts").unwrap();
    assert_eq!(entries("accounts"), 0, "DROP TABLE purged the cache");
}

/// SUB-SLICE 3b ROUTE — the CROSS-SHARD PK-INDEX point-lookup route returns rows BYTE-IDENTICAL to the
/// scan across present / absent / multi-shard / projection-variants / duplicate-fallback / generation-
/// rebuild, AND actually FIRES (`shard_index_route_hits` advances — output equality alone can't prove
/// which path ran, since the route and the scan are identical by construction). Sabotage-verified:
/// (a) breaking the slot materialization byte (`col_base + slot*4` -> `col_base`) returns shard row 0's
/// values for every key (diverges from the scan); (b) forcing `locate` to a wrong shard makes the gather
/// miss the row (0 rows) vs the scan's 1 row.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cross_shard_pk_index_route_matches_scan() {
    let e = Engine::new_local();
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
    let rows = |sql: &str| -> Vec<Vec<SqlValue>> {
        e.execute_relational_select_text(sql)
            .unwrap()
            .rows
            .into_boxed()
    };

    // Present keys spanning all 4 shards + boundaries, absent keys, and single/multi-column int4
    // projections — the EXPLICIT-projection shapes that route through the sharded general path.
    let mut sqls: Vec<String> = Vec::new();
    for k in [
        0_i32, 1, 5, 63, 64, 65, 128, 130, 191, 192, 199, 200, 999, -1,
    ] {
        sqls.push(format!("SELECT id, balance FROM accounts WHERE id = {k}"));
    }
    sqls.push("SELECT balance FROM accounts WHERE id = 64".to_string());
    sqls.push("SELECT id FROM accounts WHERE id = 0".to_string());

    // OFF = scan oracle.
    e.set_shard_index_probe_enabled(false);
    let oracle: Vec<Vec<Vec<SqlValue>>> = sqls.iter().map(|s| rows(s)).collect();
    // ON = index route: byte-identical to the scan, and it must FIRE for every one of these int4
    // unique-key point lookups (present / absent / single- / multi-column all qualify).
    e.set_shard_index_probe_enabled(true);
    for (s, want) in sqls.iter().zip(&oracle) {
        let hb = e.shard_index_route_hits();
        assert_eq!(&rows(s), want, "index route == scan for `{s}`");
        assert_eq!(
            e.shard_index_route_hits() - hb,
            1,
            "index route FIRED for `{s}` (non-vacuity)"
        );
    }

    // `SELECT *` gets a different query_shape and routes through a different resident path (NOT the
    // sharded general path this route hooks), so it does NOT take the index route — but flipping the flag
    // ON must not change its result (safe fallback / OFF-path parity). (Optimizing `SELECT * WHERE pk=k`
    // through the index is a noted follow-up.)
    e.set_shard_index_probe_enabled(false);
    let want_star = rows("SELECT * FROM accounts WHERE id = 130");
    e.set_shard_index_probe_enabled(true);
    assert_eq!(
        rows("SELECT * FROM accounts WHERE id = 130"),
        want_star,
        "SELECT * unaffected by the flag"
    );

    // DUP-FALLBACK: a duplicate int4 key declines the hash -> the route falls back to the scan (no hit),
    // still byte-identical (the scan returns EVERY match, a hash holds one row/key).
    let d = Engine::new_local();
    d.set_shard_residency_enabled(true);
    d.set_auto_admit_on_commit(true);
    d.set_shard_index_probe_enabled(true);
    d.execute_text(1, "CREATE TABLE dup (id INT, balance INT)")
        .unwrap();
    d.execute_text(
        2,
        "INSERT INTO dup (id, balance) VALUES (1,10),(1,20),(2,30)",
    )
    .unwrap();
    let dhb = d.shard_index_route_hits();
    let got = d
        .execute_relational_select_text("SELECT id, balance FROM dup WHERE id = 1")
        .unwrap()
        .rows
        .into_boxed();
    assert_eq!(
        d.shard_index_route_hits(),
        dhb,
        "duplicate key -> route declines -> scan (no hit)"
    );
    d.set_shard_index_probe_enabled(false);
    let want = d
        .execute_relational_select_text("SELECT id, balance FROM dup WHERE id = 1")
        .unwrap()
        .rows
        .into_boxed();
    assert_eq!(got, want, "dup fallback == scan");
    assert_eq!(got.len(), 2, "both duplicate rows returned");

    // GENERATION-REBUILD: a DELETE (tombstone flag OFF) invalidates + re-admits (new device ptrs, shifted
    // slots); the route on the rebuilt table still == scan (ptr-validated cache rebuild, purged on re-admit).
    e.execute_text(300, "DELETE FROM accounts WHERE id = 50")
        .unwrap();
    e.set_shard_index_probe_enabled(false);
    let want51 = rows("SELECT id, balance FROM accounts WHERE id = 51");
    let want50 = rows("SELECT id, balance FROM accounts WHERE id = 50");
    e.set_shard_index_probe_enabled(true);
    assert_eq!(
        rows("SELECT id, balance FROM accounts WHERE id = 51"),
        want51,
        "post-re-admit route == scan (survivor)"
    );
    assert_eq!(
        rows("SELECT id, balance FROM accounts WHERE id = 50"),
        want50,
        "post-re-admit route == scan (deleted)"
    );
    assert!(want50.is_empty(), "id=50 deleted");
    assert_eq!(want51.len(), 1, "id=51 survives");
}

/// SUB-SLICE 3b ROUTE — the `deleted_by` VISIBILITY gate. With in-place DELETE tombstoning ON, a deleted
/// row stays physically resident with `deleted_by[slot] = commit`; the index route must read that region
/// and HIDE the row (0 rows) exactly as the scan's SV3b filter does, while a LIVE row in the SAME (now
/// versioned) shard is still returned. Sabotage: invert the gate (`deleted_by <= read_txn_id`) and the
/// tombstoned row LEAKS (1 row) where the scan returns 0.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cross_shard_pk_index_route_deleted_by_gate() {
    let t = Engine::new_local();
    t.set_shard_residency_enabled(true);
    t.set_auto_admit_on_commit(true);
    t.set_shard_size_target(64);
    t.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    for i in 0..200_i64 {
        t.execute_text(
            (i as u64) + 2,
            &format!(
                "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                i * 10
            ),
        )
        .unwrap();
    }
    // Tombstone id=130 IN PLACE (shard 2, slot 2) -> shard 2 becomes versioned (has a deleted_by region).
    t.execute_text(300, "DELETE FROM accounts WHERE id = 130")
        .unwrap();
    let rows = |on: bool, sql: &str| -> Vec<Vec<SqlValue>> {
        t.set_shard_index_probe_enabled(on);
        t.execute_relational_select_text(sql)
            .unwrap()
            .rows
            .into_boxed()
    };
    // Tombstoned row hidden by the gate == scan (both empty), and the route DID fire (versioned shard).
    let want_del = rows(false, "SELECT id, balance FROM accounts WHERE id = 130");
    let hb = t.shard_index_route_hits();
    let got_del = rows(true, "SELECT id, balance FROM accounts WHERE id = 130");
    assert!(
        t.shard_index_route_hits() > hb,
        "route fired on the versioned shard"
    );
    assert_eq!(got_del, want_del, "index route deleted_by gate == scan");
    assert!(
        got_del.is_empty(),
        "tombstoned id=130 hidden by the deleted_by gate"
    );
    // A LIVE neighbor in the SAME versioned shard is still returned == scan.
    let want_live = rows(false, "SELECT id, balance FROM accounts WHERE id = 131");
    let got_live = rows(true, "SELECT id, balance FROM accounts WHERE id = 131");
    assert_eq!(got_live, want_live, "live neighbor route == scan");
    assert_eq!(got_live.len(), 1, "live neighbor id=131 visible");
}

/// SUB-SLICE 3b ROUTE — after M3-for-shards, the point-index route DECLINES on a NULL-BEARING table (the
/// resolved tripwire). The sharded SCAN is now NULL-aware (its recompaction rebuilds the validity bitmap +
/// labels the unified descriptor), but the raw-i32 slot route has NO validity channel -> it would read a
/// NULL-stored-0 as 0 and DIVERGE. So `execute_resident_sharded_via_general` SKIPS the route whenever any
/// surviving shard carries a null bitmap, falling to the NULL-aware scan. This test proves the decline
/// holds: route-ON == route-OFF (both the scan) AND the route does NOT fire on a null-bearing table, while
/// the scan is genuinely NULL-aware (case (a) projects SQL NULL). null-bearing => single-shard, so the
/// decline never costs the many-shard route on NULL-free tables.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cross_shard_pk_index_route_declines_on_null_bearing() {
    let sel = |e: &Engine, on: bool, sql: &str| -> (Vec<Vec<SqlValue>>, u64) {
        e.set_shard_index_probe_enabled(on);
        let hb = e.shard_index_route_hits();
        let r = e
            .execute_relational_select_text(sql)
            .unwrap()
            .rows
            .into_boxed();
        (r, e.shard_index_route_hits() - hb)
    };

    // (a) a nullable-column table: the route DECLINES (fired 0) -> the NULL-aware scan serves it, and
    // route-ON == route-OFF (both the scan). The scan projects the NULL as SQL NULL (not raw 0).
    let n = Engine::new_local();
    n.set_shard_residency_enabled(true);
    n.set_auto_admit_on_commit(true);
    n.execute_text(1, "CREATE TABLE nn (id INT, balance INT)")
        .unwrap();
    n.execute_text(
        2,
        "INSERT INTO nn (id, balance) VALUES (1,10),(2,NULL),(3,30)",
    )
    .unwrap();
    let (want_bal, _) = sel(&n, false, "SELECT id, balance FROM nn WHERE id = 2");
    let (got_bal, fired_bal) = sel(&n, true, "SELECT id, balance FROM nn WHERE id = 2");
    assert_eq!(
        got_bal, want_bal,
        "null-bearing: route declined -> route-ON == route-OFF (scan)"
    );
    assert_eq!(
        fired_bal, 0,
        "route DECLINED on the null-bearing table (M3 scan serves it)"
    );
    assert_eq!(
        got_bal,
        vec![vec![SqlValue::Int4(2), SqlValue::Null]],
        "the NULL-aware scan projects SQL NULL (proves the decline is not vacuous)"
    );
    let (want_id, _) = sel(&n, false, "SELECT id FROM nn WHERE id = 2");
    let (got_id, fired_id) = sel(&n, true, "SELECT id FROM nn WHERE id = 2");
    assert_eq!(got_id, want_id, "route declined -> == scan");
    assert_eq!(
        fired_id, 0,
        "route DECLINED (the table carries a null bitmap)"
    );

    // (b) the NULL-KEY table (a NULL id stored as 0): the route DECLINES here too -> route-ON == route-OFF.
    let k = Engine::new_local();
    k.set_shard_residency_enabled(true);
    k.set_auto_admit_on_commit(true);
    k.execute_text(1, "CREATE TABLE kn (id INT, balance INT)")
        .unwrap();
    k.execute_text(2, "INSERT INTO kn (id, balance) VALUES (5,50),(7,70)")
        .unwrap();
    k.execute_text(3, "INSERT INTO kn (id, balance) VALUES (NULL, 99)")
        .unwrap();
    let (want0, _) = sel(&k, false, "SELECT id, balance FROM kn WHERE id = 0");
    let (got0, fired0) = sel(&k, true, "SELECT id, balance FROM kn WHERE id = 0");
    assert_eq!(
        got0, want0,
        "NULL-key table: route declined -> route-ON == route-OFF (scan)"
    );
    assert_eq!(
        fired0, 0,
        "route DECLINED on the null-bearing (NULL-id) table"
    );
}
