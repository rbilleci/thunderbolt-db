/// SLICE B: a mixed-type shard-resident table (text column) serves sortable projections on the
/// GPU and remains byte-identical to the explicitly repaired single-buffer read layout.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_mixed_type_sortable_projection_stays_gpu_native() {
    let load = |e: &Engine| {
        e.set_auto_admit_on_commit(true);
        e.execute_text(1, "CREATE TABLE mt (id INT, name TEXT)")
            .unwrap();
        e.execute_text(
            2,
            "INSERT INTO mt (id, name) VALUES (3,'c'),(1,'a'),(2,'b')",
        )
        .unwrap();
    };
    let mut o = Engine::new_local(); // explicit single-buffer read-layout oracle
    load(&o);
    install_test_single_buffer_residency(&mut o, "mt");
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    load(&e);
    assert!(
        e.read_state.residency.snapshots.load().get("mt").is_none()
            && e.read_state.residency.shards.load().get("mt").is_some(),
        "precondition: a mixed-type table is shard-authoritative"
    );
    for sql in [
        "SELECT id, name FROM mt ORDER BY id",
        "SELECT id, name FROM mt ORDER BY id DESC",
        "SELECT id, name FROM mt",
    ] {
        let want = o
            .execute_relational_select_text(sql)
            .unwrap()
            .rows
            .into_boxed();
        let got = e
            .execute_relational_select_text(sql)
            .unwrap_or_else(|err| panic!("mixed-type sharded must serve {sql}: {err}"))
            .rows
            .into_boxed();
        assert_eq!(got, want, "mixed-type sharded == oracle for: {sql}");
    }
}

/// SLICE B — the VERSIONED interplay: predicate NULL 3VL composes with the SV3b/SV6 visibility
/// conjuncts ON THE DEVICE (one mask-VM program: validity-bitmap 3VL leaf AND `deleted_by >
/// read_txn`). With incremental DELETE ON, tombstoning a non-NULL row of a null-bearing sharded
/// table hides EXACTLY that row from IS NULL / IS NOT NULL / equality / COUNT — no tombstone leak,
/// no NULL mis-match. NON-VACUITY: the deleted_by region existing proves the tombstone route ran
/// (fallback re-admit leaves none and would trivially pass).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_predicate_null_3vl_on_versioned_shard() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE nn (id INT, balance INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO nn (id, balance) VALUES (1,10),(2,NULL),(3,30),(4,NULL)",
    )
    .unwrap();
    // Tombstone the non-NULL row (3,30) IN PLACE (a NULL-bearing deleted row would decline to
    // re-admit and vacuously pass — hence the region-exists route proof below).
    e.execute_text(3, "DELETE FROM nn WHERE id = 3").unwrap();
    assert!(
        table_has_any_deleted_by_cell(&e, "nn"),
        "route proof: the incremental tombstone fired (re-admit would leave no region)"
    );
    let run = |sql: &str| -> Vec<Vec<SqlValue>> {
        e.execute_relational_select_text(sql)
            .unwrap()
            .rows
            .into_boxed()
    };
    assert_eq!(
        run("SELECT id FROM nn WHERE balance IS NULL"),
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(4)]],
        "IS NULL over a versioned shard: NULL rows visible, tombstoned row hidden"
    );
    assert_eq!(
        run("SELECT id FROM nn WHERE balance IS NOT NULL"),
        vec![vec![SqlValue::Int4(1)]],
        "IS NOT NULL: only the live non-NULL row (the tombstoned (3,30) is hidden)"
    );
    assert_eq!(
        run("SELECT id, balance FROM nn WHERE balance = 30"),
        Vec::<Vec<SqlValue>>::new(),
        "equality on the tombstoned row's value: hidden by the visibility conjunct"
    );
    assert_eq!(
        run("SELECT COUNT(*) FROM nn"),
        vec![vec![SqlValue::Int8(3)]],
        "COUNT drops by exactly the tombstoned row (visibility-only program over the unified buffer)"
    );
    // R-ver PART 2: the DISTINCT / GROUP BY / ORDER BY paths now THREAD the SV3b/SV6 visibility
    // conjunct (into the survivor `indices` before the sort), so a versioned sharded ORDER BY
    // returns the VISIBLE rows sorted — the tombstoned (3,30) is hidden, NULLs preserved.
    // (This assertion was deliberately flipped from the pre-PART-2 clean-error.)
    assert_eq!(
        run("SELECT id, balance FROM nn ORDER BY id DESC"),
        vec![
            vec![SqlValue::Int4(4), SqlValue::Null],
            vec![SqlValue::Int4(2), SqlValue::Null],
            vec![SqlValue::Int4(1), SqlValue::Int4(10)],
        ],
        "versioned sharded ORDER BY: id DESC over the VISIBLE rows (tombstoned id=3 hidden)"
    );
}

/// M3-for-shards: the sharded read path's PROJECTION is now NULL-AWARE — a NULL materializes as
/// `SqlValue::Null`, not the raw-0 placeholder the ledger flagged as SQL-WRONG. The sharded scan's
/// recompaction rebuilds each column's validity bitmap into the unified buffer + labels the unified
/// descriptor, so the general executor emits NULLs. Asserts the SQL-SPEC-CORRECT result directly (the
/// authoritative reference — [[sql-spec-over-cpu-parity]]) for a NULL in the PROJECTED column AND a NULL in
/// the KEY column. Sabotage: passing an empty `unified_null_columns` (or dropping the null-region
/// fills/segments) reverts to raw-0 -> the NULL cells read back as Int4(0), failing the assertions below.
/// (IS NULL / IS NOT NULL predicates are a separate sharded-router-eligibility concern, out of scope here.)
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_null_read_projects_sql_null() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE nn (id INT, balance INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO nn (id, balance) VALUES (1,10),(2,NULL),(3,30),(4,40)",
    )
    .unwrap();
    let run = |sql: &str| -> Vec<Vec<SqlValue>> {
        e.execute_relational_select_text(sql)
            .unwrap()
            .rows
            .into_boxed()
    };
    // NULL in the PROJECTED column materializes as SQL NULL (the documented bug was Int4(0)).
    assert_eq!(
        run("SELECT id, balance FROM nn WHERE id = 2"),
        vec![vec![SqlValue::Int4(2), SqlValue::Null]],
        "NULL balance projects as NULL"
    );
    // Non-null control: unchanged (byte-identical to the pre-M3 read).
    assert_eq!(
        run("SELECT id, balance FROM nn WHERE id = 3"),
        vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]],
        "non-null row unchanged"
    );
    assert_eq!(
        run("SELECT balance FROM nn WHERE id = 4"),
        vec![vec![SqlValue::Int4(40)]],
        "non-null single projection unchanged"
    );
}

/// STEP 1 (lpb-for-shards) — the BATCHED cross-shard point-lookup gather returns, per needle, rows
/// BYTE-IDENTICAL to the single-flight 3b route (which is itself == scan == host), across
/// present / absent / multi-shard / NULL-blind, and the batched path FIRES (`sharded_point_batch_hits`
/// advances). Sabotage: dropping the slot from the gather (`project_i32_rows_from_payload(col_base, [0;n])`)
/// returns row-0 values for every needle → diverges from the single-flight route.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_batch_matches_single_flight_route() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_index_probe_enabled(true); // the single-flight 3b route is the per-needle oracle
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
    let bal_col = crate::rel_exec_helpers::relational_column_index(&table, "balance").unwrap();

    // A batch of needles spanning all 4 shards + absent keys + boundaries, plus REPEATED values (50, 130
    // appear twice) — the per-needle oracle loop asserts BOTH positions of a repeated key materialize the
    // same row, guarding the per-`needle_index` `hit_shard_count` accounting against treating a repeated
    // needle value as a (spurious) cross-shard duplicate (audit coverage follow-up).
    let needles: Vec<i32> = vec![
        0, 1, 5, 63, 64, 65, 128, 130, 191, 199, 200, 999, -1, 50, 51, 50, 130,
    ];
    let hb = e.sharded_point_batch_hits();
    let gpu_hb = e.sharded_point_gpu_probe_hits();
    let bin_hb = e.sharded_point_binary_route_hits();
    let proj = e
        .gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            id_col,
            &[id_col, bal_col],
            &needles,
        )
        .expect("batched path served this shape");
    assert!(
        e.sharded_point_batch_hits() > hb,
        "batched path FIRED (non-vacuity)"
    );
    // Sub-slice 8 v3 (O(1) routing): these 4 shards are ascending-disjoint (ordered inserts), so the kernel
    // takes the BINARY-SEARCH path (each needle -> its one shard in O(log shards)). Prove it fired so the
    // byte-identical comparison below is validating the binary route (present/absent/boundary/dup/out-of-
    // range all covered). Sabotage (a wrong binary candidate) breaks the per-needle equality below.
    assert!(
        e.sharded_point_binary_route_hits() > bin_hb,
        "the O(1) BINARY-SEARCH route FIRED (ascending-disjoint shards)"
    );
    // Sub-slice 8: this delete-free table takes the FULLY-GPU dense-emit path (not the host-probe
    // fallback) — prove it fired, so the byte-identical comparison below is validating the GPU path.
    assert!(
        e.sharded_point_gpu_probe_hits() > gpu_hb,
        "the GPU-native dense-emit probe path FIRED (delete-free -> not the host fallback)"
    );
    assert_eq!(proj.ncols, 2, "id, balance");
    // Per-needle rows from the flat batched projection.
    let batched: Vec<Vec<Vec<i32>>> = (0..needles.len())
        .map(|i| {
            let (start, count) = proj.needle_ranges[i];
            (0..count as usize)
                .map(|r| {
                    let base = (start as usize + r) * proj.ncols;
                    proj.values[base..base + proj.ncols].to_vec()
                })
                .collect()
        })
        .collect();

    // Single-flight 3b route (== scan == host) as the per-needle oracle.
    for (i, &k) in needles.iter().enumerate() {
        let rows = e
            .execute_relational_select_text(&format!(
                "SELECT id, balance FROM accounts WHERE id = {k}"
            ))
            .unwrap()
            .rows
            .into_boxed();
        let want: Vec<Vec<i32>> = rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|v| match v {
                        SqlValue::Int4(x) => *x,
                        other => panic!("expected int4, got {other:?}"),
                    })
                    .collect()
            })
            .collect();
        assert_eq!(
            batched[i], want,
            "batched == single-flight route for id={k}"
        );
    }

    // DUP-FALLBACK: a duplicate int4 key -> the batched path declines (None) -> caller scans.
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
    let dt = d.relational_catalog_table("dup").unwrap();
    let did = crate::rel_exec_helpers::relational_column_index(&dt, "id").unwrap();
    let dbal = crate::rel_exec_helpers::relational_column_index(&dt, "balance").unwrap();
    assert!(
        d.gather_sharded_int4_point_lookups_batched(
            d.committed_seq(),
            &dt,
            did,
            &[did, dbal],
            &[1, 2]
        )
        .is_none(),
        "duplicate key -> batched path declines -> None (caller falls back to the scan)"
    );

    // CROSS-SHARD DUP (multi-shard kernel v2 correctness): the SAME key in TWO shards (each once, NO
    // within-shard dup so both per-shard indexes build) -> the multi-shard kernel must NOT return only the
    // first shard's row (the scan returns BOTH). It detects the 2nd shard hit -> the whole batch DECLINES
    // (None) -> the caller falls back to the scan. A kernel that breaks on the first hit returns Some(1).
    let x = Engine::new_local();
    x.set_shard_residency_enabled(true);
    x.set_auto_admit_on_commit(true);
    x.set_shard_index_probe_enabled(true);
    x.set_shard_size_target(64);
    x.execute_text(1, "CREATE TABLE xdup (id INT, balance INT)")
        .unwrap();
    for i in 0..200i64 {
        // id = i%100 -> id 5 at row 5 (shard 0) AND row 105 (shard 1): a CROSS-shard dup, unique per shard.
        x.execute_text(
            (i as u64) + 2,
            &format!(
                "INSERT INTO xdup (id, balance) VALUES ({}, {})",
                i % 100,
                i * 10
            ),
        )
        .unwrap();
    }
    let xt = x.relational_catalog_table("xdup").unwrap();
    let xid = crate::rel_exec_helpers::relational_column_index(&xt, "id").unwrap();
    let xbal = crate::rel_exec_helpers::relational_column_index(&xt, "balance").unwrap();
    assert!(
        x.gather_sharded_int4_point_lookups_batched(
            x.committed_seq(),
            &xt,
            xid,
            &[xid, xbal],
            &[5]
        )
        .is_none(),
        "cross-shard duplicate key -> batched declines -> None (scan returns BOTH rows)"
    );
    let xrows = x
        .execute_relational_select_text("SELECT id, balance FROM xdup WHERE id = 5")
        .unwrap()
        .rows
        .len();
    assert_eq!(
        xrows, 2,
        "cross-shard dup id=5 -> 2 rows (row 5 + row 105) via the scan"
    );
    // (NULL-bearing tables are covered by `sharded_point_batch_declines_on_null_bearing` — the batched
    // gather declines them post-M3, so this NULL-free differential no longer exercises a NULL sub-case.)
}

/// M3-for-shards: the RAW-i32 batched gather DECLINES iff a REFERENCED filter/projection column carries a
/// NULL bitmap. A NULL in an unreferenced column is safe (the kernel never reads it) and must not disable the
/// GPU route for `SELECT id WHERE id = ?`; selecting that nullable column still declines to the NULL-aware
/// per-query scan. A NULL filter key likewise declines. These controls pin both sides of the metadata gate.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_batch_declines_on_null_bearing() {
    // NULL-FREE control: the batched gather SERVES it (Some) -> the decline below is null-specific.
    let f = Engine::new_local();
    f.set_shard_residency_enabled(true);
    f.set_auto_admit_on_commit(true);
    f.set_shard_index_probe_enabled(true);
    f.execute_text(1, "CREATE TABLE nf (id INT, balance INT)")
        .unwrap();
    f.execute_text(2, "INSERT INTO nf (id, balance) VALUES (5,50),(7,70)")
        .unwrap();
    let tf = f.relational_catalog_table("nf").unwrap();
    let idf = crate::rel_exec_helpers::relational_column_index(&tf, "id").unwrap();
    let balf = crate::rel_exec_helpers::relational_column_index(&tf, "balance").unwrap();
    assert!(
        f.gather_sharded_int4_point_lookups_batched(
            f.committed_seq(),
            &tf,
            idf,
            &[idf, balf],
            &[5]
        )
        .is_some(),
        "NULL-free table: batched gather SERVES (the decline is null-specific, not always-None)"
    );

    // A NULL in an UNREFERENCED column must not disable the raw-i32 route: SELECT id reads only
    // the non-null key/projection column. Selecting the nullable column still declines because
    // the compact batched result has no validity channel yet.
    let u = Engine::new_local();
    u.set_shard_residency_enabled(true);
    u.set_auto_admit_on_commit(true);
    u.set_shard_index_probe_enabled(true);
    u.execute_text(1, "CREATE TABLE unref (id INT PRIMARY KEY, balance INT)")
        .unwrap();
    u.execute_text(2, "INSERT INTO unref (id, balance) VALUES (5,NULL),(7,70)")
        .unwrap();
    let tu = u.relational_catalog_table("unref").unwrap();
    let idu = crate::rel_exec_helpers::relational_column_index(&tu, "id").unwrap();
    let balu = crate::rel_exec_helpers::relational_column_index(&tu, "balance").unwrap();
    let gpu_before = u.sharded_point_gpu_probe_hits();
    let projected = u
        .gather_sharded_int4_point_lookups_batched(u.committed_seq(), &tu, idu, &[idu], &[5])
        .expect("NULL in an unreferenced column does not disable the GPU batched route");
    assert_eq!(projected.values, vec![5]);
    assert!(
        u.sharded_point_gpu_probe_hits() > gpu_before,
        "unreferenced-NULL control executed the GPU probe"
    );
    assert!(
        u.gather_sharded_int4_point_lookups_batched(
            u.committed_seq(),
            &tu,
            idu,
            &[idu, balu],
            &[5],
        )
        .is_none(),
        "referencing the nullable projection still declines to the NULL-aware path"
    );

    // NULL-BEARING (a NULL id -> a null bitmap on the filter column): the batched gather DECLINES (None).
    let k = Engine::new_local();
    k.set_shard_residency_enabled(true);
    k.set_auto_admit_on_commit(true);
    k.set_shard_index_probe_enabled(true);
    k.execute_text(1, "CREATE TABLE knz (id INT, balance INT)")
        .unwrap();
    k.execute_text(2, "INSERT INTO knz (id, balance) VALUES (5,50),(7,70)")
        .unwrap();
    k.execute_text(3, "INSERT INTO knz (id, balance) VALUES (NULL, 99)")
        .unwrap();
    let t = k.relational_catalog_table("knz").unwrap();
    let id = crate::rel_exec_helpers::relational_column_index(&t, "id").unwrap();
    let bal = crate::rel_exec_helpers::relational_column_index(&t, "balance").unwrap();
    assert!(
        k.gather_sharded_int4_point_lookups_batched(k.committed_seq(), &t, id, &[id, bal], &[5])
            .is_none(),
        "NULL filter key: batched gather DECLINES (-> the facade per-query fallback runs the NULL-aware scan)"
    );
}

/// SUB-SLICE 8 v3 (O(1) routing) — the BINARY-SEARCH route at DEPTH across many ascending-disjoint shards.
/// 256 ordered rows over shard_size 16 -> ~16 disjoint shards, so each needle routes to its one shard in
/// O(log shards) (binary-search depth ~4) instead of the O(shards) linear scan. Needles span EVERY shard
/// (present), the exact seal boundaries (15/16/.../240), and out-of-all-ranges keys (300/-5/1000 -> the
/// binary BKEEP0 fallback -> absent). Byte-identical to the scan proves the binary search lands on the RIGHT
/// shard at every depth. A NULL in a referenced key/projection still declines to the NULL-aware scan; a
/// nullable unreferenced column is permitted because the batched kernel never reads it (see
/// `sharded_point_batch_declines_on_null_bearing`).
/// Sabotage: a wrong binary candidate (or a broken bound) makes a present needle materialize the wrong row.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_batch_binary_route_deep_shards() {
    let k = Engine::new_local();
    k.set_shard_residency_enabled(true);
    k.set_auto_admit_on_commit(true);
    k.set_shard_index_probe_enabled(true);
    k.set_shard_size_target(16); // 256 rows -> ~16 ascending-disjoint shards
    k.execute_text(1, "CREATE TABLE acc (id INT, balance INT)")
        .unwrap();
    for i in 0..256_i64 {
        k.execute_text(
            (i as u64) + 2,
            &format!("INSERT INTO acc (id, balance) VALUES ({i}, {})", i * 10),
        )
        .unwrap();
    }
    let t = k.relational_catalog_table("acc").unwrap();
    assert!(
        k.resident_shard_count("acc") >= 8,
        "many disjoint shards -> deep binary search"
    );
    let id = crate::rel_exec_helpers::relational_column_index(&t, "id").unwrap();
    let bal = crate::rel_exec_helpers::relational_column_index(&t, "balance").unwrap();
    // present in various shards + seal boundaries + out-of-all-ranges (BKEEP0 -> absent).
    let needles: Vec<i32> = vec![
        0, 15, 16, 17, 31, 32, 100, 128, 200, 239, 240, 255, 300, -5, 1000,
    ];
    let bin_hb = k.sharded_point_binary_route_hits();
    let gpu_hb = k.sharded_point_gpu_probe_hits();
    let proj = k
        .gather_sharded_int4_point_lookups_batched(k.committed_seq(), &t, id, &[id, bal], &needles)
        .expect("batched served (delete-free)");
    assert!(
        k.sharded_point_gpu_probe_hits() > gpu_hb,
        "the GPU-native probe fired (not host fallback)"
    );
    assert!(
        k.sharded_point_binary_route_hits() > bin_hb,
        "the O(1) BINARY-SEARCH route fired (>= 8 disjoint shards)"
    );
    for (i, &needle) in needles.iter().enumerate() {
        let (start, count) = proj.needle_ranges[i];
        let got: Vec<Vec<i32>> = (0..count as usize)
            .map(|r| {
                let base = (start as usize + r) * proj.ncols;
                proj.values[base..base + proj.ncols].to_vec()
            })
            .collect();
        let want = k
            .execute_relational_select_text(&format!(
                "SELECT id, balance FROM acc WHERE id = {needle}"
            ))
            .unwrap()
            .rows
            .into_boxed();
        let want_i32: Vec<Vec<i32>> = want
            .iter()
            .map(|row| {
                row.iter()
                    .map(|v| match v {
                        SqlValue::Int4(x) => *x,
                        o => panic!("expected int4, got {o:?}"),
                    })
                    .collect()
            })
            .collect();
        assert_eq!(
            got, want_i32,
            "binary route id={needle} == scan (right shard at depth)"
        );
    }
}

/// Dense sharded point batches apply BOTH visibility bounds on-device. A deleted needle is hidden while live
/// neighbors survive, and an INSERT committed after an explicitly pinned boundary is hidden at the old
/// boundary but visible at the current one. The GPU counter must advance for both versioned cases: a host
/// visibility gather would make this test vacuous. Sabotage either compare and a dead/future row leaks.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_batch_deleted_by_gate() {
    let t = Engine::new_local();
    t.set_shard_residency_enabled(true);
    t.set_auto_admit_on_commit(true);
    t.set_shard_index_probe_enabled(true);
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
    t.execute_text(300, "DELETE FROM accounts WHERE id = 130")
        .unwrap();
    let table = t.relational_catalog_table("accounts").unwrap();
    let id_col = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let bal_col = crate::rel_exec_helpers::relational_column_index(&table, "balance").unwrap();

    let needles: Vec<i32> = vec![129, 130, 131];
    let gpu_hb = t.sharded_point_gpu_probe_hits();
    let proj = t
        .gather_sharded_int4_point_lookups_batched(
            t.committed_seq(),
            &table,
            id_col,
            &[id_col, bal_col],
            &needles,
        )
        .expect("batched served");
    assert!(
        t.sharded_point_gpu_probe_hits() > gpu_hb,
        "versioned shard stays on the fully-GPU dense path"
    );
    let count = |i: usize| proj.needle_ranges[i].1;
    assert_eq!(count(0), 1, "id=129 live -> 1 row");
    assert_eq!(
        count(1),
        0,
        "id=130 tombstoned -> hidden by the batched deleted_by gate"
    );
    assert_eq!(
        count(2),
        1,
        "id=131 live neighbor in the same versioned shard -> 1 row"
    );

    // == single-flight route (row counts).
    for &k in &needles {
        let want = t
            .execute_relational_select_text(&format!(
                "SELECT id, balance FROM accounts WHERE id = {k}"
            ))
            .unwrap()
            .rows
            .len();
        let idx = needles.iter().position(|&x| x == k).unwrap();
        assert_eq!(
            count(idx) as usize,
            want,
            "batched row count == single-flight for id={k}"
        );
    }

    // D3 lower bound: append a fresh row, then query it at the prior and current snapshots. Both calls fire
    // the dense route; only the snapshot boundary changes the on-device visibility verdict.
    let prior_snapshot = t.committed_seq();
    t.execute_text(301, "INSERT INTO accounts (id, balance) VALUES (201, 2010)")
        .unwrap();
    let table = t.relational_catalog_table("accounts").unwrap();
    let gpu_before_birth_checks = t.sharded_point_gpu_probe_hits();
    let old = t
        .gather_sharded_int4_point_lookups_batched(
            prior_snapshot,
            &table,
            id_col,
            &[id_col, bal_col],
            &[201],
        )
        .expect("prior-snapshot dense probe served");
    assert_eq!(old.needle_ranges[0].1, 0, "future-born row hidden");
    let current = t
        .gather_sharded_int4_point_lookups_batched(
            t.committed_seq(),
            &table,
            id_col,
            &[id_col, bal_col],
            &[201],
        )
        .expect("current-snapshot dense probe served");
    assert_eq!(current.needle_ranges[0].1, 1, "born row visible now");
    assert!(
        t.sharded_point_gpu_probe_hits() >= gpu_before_birth_checks + 2,
        "both snapshot-bound checks fired the fully-GPU dense probe"
    );
}
