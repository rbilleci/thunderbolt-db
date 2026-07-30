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
/// authoritative fixture-derived reference) for a NULL in the PROJECTED column AND a NULL in
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
/// BYTE-IDENTICAL to the single-flight 3b GPU route, across
/// present / absent / multi-shard / NULL-blind, and the batched path FIRES (`sharded_point_batch_hits`
/// advances). Sabotage: dropping the slot from the gather (`project_i32_rows_from_payload(col_base, [0;n])`)
/// returns row-0 values for every needle → diverges from the single-flight route.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_batch_matches_single_flight_route() {
    let mut e = Engine::new_local();
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
    let cache_hb = e.sharded_point_route_cache_hits();
    let proj = e
        .gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            id_col,
            &[id_col, bal_col],
            &needles,
        )
        .expect("batched GPU route completed")
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
    // Sub-slice 8: this delete-free table takes the fully GPU-native dense-emit path. Prove it
    // fired so the byte-identical comparison below validates the intended route.
    assert!(
        e.sharded_point_gpu_probe_hits() > gpu_hb,
        "the GPU-native dense-emit probe path FIRED"
    );
    let cached = e
        .gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            id_col,
            &[id_col, bal_col],
            &needles,
        )
        .expect("batched GPU route completed")
        .expect("the same generation and projection must reuse the prepared GPU route");
    assert!(
        e.sharded_point_route_cache_hits() > cache_hb,
        "the exact-generation prepared GPU route was reused"
    );
    assert_eq!(cached.ncols, proj.ncols);
    assert_eq!(cached.values, proj.values);
    assert_eq!(cached.needle_count(), proj.needle_count());
    for i in 0..needles.len() {
        assert_eq!(cached.needle_range(i), proj.needle_range(i));
    }
    let compat_select =
        match parse_command("SELECT id, balance FROM accounts WHERE id = 1").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
    let compat_template = e
        .prepare_relational_retained_read_template(&compat_select)
        .unwrap();
    let compat_submission = e
        .submit_relational_retained_template_point_lookups(&compat_template, &needles)
        .unwrap();
    let compat = e
        .complete_relational_retained_read_submission_batched(compat_submission)
        .unwrap();
    assert_eq!(
        compat.needle_ranges.len(),
        needles.len(),
        "retained-template public ABI exposes one explicit range per needle"
    );
    // An unrelated table publication rotates only that table's point-route token. The accounts route must
    // remain reusable; comparing the outer database-global shard-map Arc would miss here and restore O(shards).
    e.execute_text(10_000, "CREATE TABLE route_other (id INT, value INT)")
        .unwrap();
    e.execute_text(10_001, "INSERT INTO route_other (id, value) VALUES (1, 10)")
        .unwrap();
    let unrelated_hb = e.sharded_point_route_cache_hits();
    e.gather_sharded_int4_point_lookups_batched(
        e.committed_seq(),
        &table,
        id_col,
        &[id_col, bal_col],
        &needles,
    )
    .expect("batched GPU route completed after unrelated publication")
    .expect("accounts route remains eligible after unrelated publication");
    assert!(
        e.sharded_point_route_cache_hits() > unrelated_hb,
        "unrelated-table write preserves the exact accounts table-generation cache hit"
    );
    assert_eq!(proj.ncols, 2, "id, balance");
    // Per-needle rows from the flat batched projection.
    let batched: Vec<Vec<Vec<i32>>> = (0..needles.len())
        .map(|i| {
            let (start, count) = proj.needle_range(i);
            (0..count as usize)
                .map(|r| {
                    let base = (start as usize + r) * proj.ncols;
                    proj.values[base..base + proj.ncols].to_vec()
                })
                .collect()
        })
        .collect();

    // Single-flight 3b route as the per-needle GPU baseline.
    for (i, &k) in needles.iter().enumerate() {
        let result = e
            .execute_relational_select_text(&format!(
                "SELECT id, balance FROM accounts WHERE id = {k}"
            ))
            .unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
        let rows = result.rows.into_boxed();
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

    // Projection-shape churn retains at most one route for this table. Its exact pooled descriptor bucket
    // is included in residency accounting, and a no-headroom budget allows the query but refuses retention.
    for projection in [vec![id_col], vec![bal_col], vec![bal_col, id_col]] {
        e.gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            id_col,
            &projection,
            &[5],
        )
        .expect("projection-churn GPU route completed")
        .expect("projection-churn shape served");
        assert_eq!(
            usize::from(
                e.read_state
                    .residency
                    .sharded_point_route_for_table("accounts")
                    .is_some()
            ),
            1,
            "one deterministic prepared shape per table"
        );
    }
    let route_bytes = e
        .read_state
        .residency
        .sharded_point_route_descriptor_bytes_for_gpu(0);
    assert!(
        route_bytes > 0,
        "prepared descriptor allocation is non-vacuous"
    );
    let with_route = e.relational_resident_bytes_for_gpu(0);
    e.read_state.residency.clear_sharded_point_routes_for_test();
    let without_route = e.relational_resident_bytes_for_gpu(0);
    assert_eq!(
        with_route.saturating_sub(without_route),
        route_bytes,
        "residency accounting charges the exact live descriptor bucket"
    );
    e.set_relational_residency_budget_bytes(0, without_route);
    e.gather_sharded_int4_point_lookups_batched(
        e.committed_seq(),
        &table,
        id_col,
        &[id_col, bal_col],
        &[5],
    )
    .expect("no-headroom GPU route completed")
    .expect("no-headroom route still serves transiently");
    assert!(
        e.read_state.residency.sharded_point_route_count() == 0,
        "prepared descriptor is not retained beyond the hard residency budget"
    );

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
        .expect("batched GPU route completed")
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
        .expect("batched GPU route completed")
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
        .expect("batched GPU route completed")
        .is_some(),
        "NULL-free table: batched gather SERVES (the decline is null-specific, not always-None)"
    );
    let cache_before = f.sharded_point_route_cache_hits();
    assert!(
        f.gather_sharded_int4_point_lookups_batched(
            f.committed_seq(),
            &tf,
            idf,
            &[idf, balf],
            &[5]
        )
        .expect("cached NULL-free batched GPU route completed")
        .is_some(),
        "the exact NULL-free generation remains cache eligible"
    );
    assert!(
        f.sharded_point_route_cache_hits() > cache_before,
        "the NULL-free G0 route was warmed before same-table publication"
    );
    f.execute_text(3, "INSERT INTO nf (id, balance) VALUES (9,NULL)")
        .unwrap();
    let cache_after_warm = f.sharded_point_route_cache_hits();
    let nullable_tf = f.relational_catalog_table("nf").unwrap();
    assert!(
        f.gather_sharded_int4_point_lookups_batched(
            f.committed_seq(),
            &nullable_tf,
            idf,
            &[idf, balf],
            &[9],
        )
        .expect("same-table NULL generation eligibility completed")
        .is_none(),
        "NULL-bearing G1 cannot reuse the warmed NULL-free G0 route"
    );
    assert_eq!(
        f.sharded_point_route_cache_hits(),
        cache_after_warm,
        "same-table generation rotation rejects the stale cached route before its NULL-free proof"
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
        .expect("batched GPU route completed")
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
        .expect("batched GPU route completed")
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
            .expect("batched GPU route completed")
            .is_none(),
        "NULL filter key: batched gather DECLINES (-> the facade per-query fallback runs the NULL-aware scan)"
    );
}

/// PERF-001 audit regressions: a route prepared against G0 cannot republish *or submit* after G1 wins, and
/// injected CUDA failures remain typed errors rather than ordinary declines that a caller could retry elsewhere.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_route_publication_and_cuda_failures_fail_closed() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(16);
    e.execute_text(1, "CREATE TABLE route_race (id INT, balance INT)")
        .unwrap();
    for id in 0..64_i64 {
        e.execute_text(
            id as u64 + 2,
            &format!(
                "INSERT INTO route_race (id, balance) VALUES ({id}, {})",
                id * 10
            ),
        )
        .unwrap();
    }
    let table = e.relational_catalog_table("route_race").unwrap();
    let id = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let balance = crate::rel_exec_helpers::relational_column_index(&table, "balance").unwrap();
    let old_generation = std::sync::Arc::clone(
        &e.read_state.residency.shards.load()["route_race"][0].point_route_generation,
    );

    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    e.set_sharded_point_route_pre_publish_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    let reader_engine = std::sync::Arc::clone(&e);
    let reader_table = table.clone();
    let reader = std::thread::spawn(move || {
        reader_engine.gather_sharded_int4_point_lookups_batched(
            reader_engine.committed_seq(),
            &reader_table,
            id,
            &[id, balance],
            &[5],
        )
    });
    reached.wait();
    e.execute_text(
        10_000,
        "INSERT INTO route_race (id, balance) VALUES (1000, 10000)",
    )
    .unwrap();
    let current_generation = std::sync::Arc::clone(
        &e.read_state.residency.shards.load()["route_race"][0].point_route_generation,
    );
    assert!(
        !std::sync::Arc::ptr_eq(&old_generation, &current_generation),
        "same-table publication rotates the O(1) generation token"
    );
    resume.wait();
    assert!(
        reader
            .join()
            .expect("reader thread")
            .expect("old prepared GPU read completed")
            .is_none(),
        "a plan whose G0 authority retired before pre-publish must decline before submission"
    );
    assert!(
        e.read_state
            .residency
            .sharded_point_route_for_table("route_race")
            .is_none(),
        "G0 reader must not republish its retired plan after G1 publication"
    );

    // Establish one current-generation cached route, then prove all three fault phases return Err and do
    // not increment the successfully-served batch counter. Phase 1 uses a different shape to force prepare.
    e.gather_sharded_int4_point_lookups_batched(
        e.committed_seq(),
        &table,
        id,
        &[id, balance],
        &[5],
    )
    .expect("current route completed")
    .expect("current route served");
    for (phase, projection) in [
        (1_u8, vec![id]),
        (2, vec![id, balance]),
        (3, vec![id, balance]),
    ] {
        let served_before = e.sharded_point_batch_hits();
        e.force_next_sharded_point_cuda_failure(phase);
        let err = e
            .gather_sharded_int4_point_lookups_batched(
                e.committed_seq(),
                &table,
                id,
                &projection,
                &[5],
            )
            .expect_err("injected CUDA failure must remain a typed error");
        assert!(err.to_string().contains("GPU prepared shard point-route"));
        assert_eq!(
            e.sharded_point_batch_hits(),
            served_before,
            "failed CUDA phase cannot be counted or converted into a served/fallback batch"
        );
    }
}

/// An index builder paused after CUDA construction cannot republish or submit a retired table generation after
/// DROP's route/index purge. It must never re-enter the durable index map and pin the dropped payload outside
/// residency accounting.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_index_build_cannot_republish_after_drop() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE dropped_build (id INT PRIMARY KEY, value INT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO dropped_build (id, value) VALUES (1, 10), (2, 20)",
    )
    .unwrap();
    let table = e.relational_catalog_table("dropped_build").unwrap();
    let id = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let value = crate::rel_exec_helpers::relational_column_index(&table, "value").unwrap();
    e.read_state
        .residency
        .purge_shard_pk_index_for_table("dropped_build");

    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    e.set_shard_pk_index_pre_publish_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    let reader_engine = std::sync::Arc::clone(&e);
    let reader = std::thread::spawn(move || {
        reader_engine.gather_sharded_int4_point_lookups_batched(
            reader_engine.committed_seq(),
            &table,
            id,
            &[id, value],
            &[1],
        )
    });
    reached.wait();
    e.execute_text(3, "DROP TABLE dropped_build").unwrap();
    assert!(
        e.read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .all(|(table, _, _)| table != "dropped_build"),
        "DROP purged every durable index owner before the paused builder resumes"
    );
    resume.wait();
    assert!(
        reader.join().unwrap().unwrap().is_none(),
        "the retired captured attempt declines instead of publishing a route"
    );
    assert!(
        e.read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .all(|(table, _, _)| table != "dropped_build"),
        "the completed G0 build cannot republish after lifecycle retirement"
    );
}

/// The point-route builder owns a pre-publication GPU plan while DDL can remove the old slot and
/// recreate its name with a new OID. The stale builder must decline before submission and can
/// never cache through the replacement table's slot or epoch.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_route_drop_recreate_cannot_cross_the_catalog_slot_fence() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE point_slot_drop_race (id INT PRIMARY KEY, value INT)",
    )
    .unwrap();
    e.execute_text(2, "INSERT INTO point_slot_drop_race VALUES (1, 10)")
        .unwrap();
    let old_table = e.relational_catalog_table("point_slot_drop_race").unwrap();
    let id = crate::rel_exec_helpers::relational_column_index(&old_table, "id").unwrap();
    let value = crate::rel_exec_helpers::relational_column_index(&old_table, "value").unwrap();
    let old_epoch = e
        .read_state
        .residency
        .point_index_mutation_epoch_for_table(&e.read_state, &old_table)
        .expect("the old table owns its slot before the paused build");
    let old_slot = e
        .read_state
        .residency
        .table_point_slot("point_slot_drop_race", old_table.oid)
        .expect("retain the old table slot through DROP/recreate");
    let old_identity = std::sync::Arc::clone(&old_slot.slot_identity);

    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    e.set_sharded_point_route_pre_publish_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    let builder_engine = std::sync::Arc::clone(&e);
    let builder_table = old_table.clone();
    let builder = std::thread::spawn(move || {
        builder_engine.gather_sharded_int4_point_lookups_batched(
            builder_engine.committed_seq(),
            &builder_table,
            id,
            &[id, value],
            &[1],
        )
    });
    reached.wait();
    e.execute_text(3, "DROP TABLE point_slot_drop_race").unwrap();
    e.execute_text(
        4,
        "CREATE TABLE point_slot_drop_race (id INT PRIMARY KEY, value INT)",
    )
    .unwrap();
    let replacement_table = e.relational_catalog_table("point_slot_drop_race").unwrap();
    assert_ne!(replacement_table.oid, old_table.oid);
    assert!(
        e.read_state
            .residency
            .table_point_slot("point_slot_drop_race", old_table.oid)
            .is_none(),
        "the dropped OID is no longer reachable under the recreated name"
    );
    resume.wait();
    assert!(
        builder
            .join()
            .expect("paused old builder joins")
            .expect("stale builder completed")
            .is_none(),
        "the dropped builder declines before submitting its captured plan"
    );
    assert!(
        e.read_state
            .residency
            .sharded_point_route_for_table("point_slot_drop_race")
            .is_none(),
        "the old builder cannot cache its route through the recreated catalog entry"
    );
    assert!(old_slot.sharded_route.load().is_none());
    assert!(old_slot.compound_route.load().is_none());

    e.execute_text(5, "INSERT INTO point_slot_drop_race VALUES (2, 20)")
        .unwrap();
    let replacement_id =
        crate::rel_exec_helpers::relational_column_index(&replacement_table, "id").unwrap();
    let replacement_value =
        crate::rel_exec_helpers::relational_column_index(&replacement_table, "value").unwrap();
    let fresh = e
        .gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &replacement_table,
            replacement_id,
            &[replacement_id, replacement_value],
            &[2],
        )
        .expect("fresh recreated-table GPU route completes")
        .expect("fresh recreated-table GPU route is eligible");
    assert_eq!(fresh.values, vec![2, 20]);
    let replacement_slot = e
        .read_state
        .residency
        .table_point_slot("point_slot_drop_race", replacement_table.oid)
        .expect("fresh binding installs only the replacement slot");
    assert!(!std::sync::Arc::ptr_eq(&old_slot, &replacement_slot));
    assert!(!std::sync::Arc::ptr_eq(&old_epoch, &replacement_slot.index_epoch));
    assert!(!std::sync::Arc::ptr_eq(
        &old_identity,
        &replacement_slot.slot_identity
    ));
}

/// Direct descriptor invalidation is itself a table-generation publication. A reader paused after preparing
/// G0 cannot republish or submit that route after the invalidated G1 descriptor wins.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_route_invalidation_rotates_generation_before_purge() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE invalidation_race (id INT PRIMARY KEY, value INT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO invalidation_race (id, value) VALUES (1, 10), (2, 20)",
    )
    .unwrap();
    let table = e.relational_catalog_table("invalidation_race").unwrap();
    let id = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let value = crate::rel_exec_helpers::relational_column_index(&table, "value").unwrap();
    let old_generation = std::sync::Arc::clone(
        &e.read_state.residency.shards.load()["invalidation_race"][0].point_route_generation,
    );

    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    e.set_sharded_point_route_pre_publish_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    let reader_engine = std::sync::Arc::clone(&e);
    let reader = std::thread::spawn(move || {
        reader_engine.gather_sharded_int4_point_lookups_batched(
            reader_engine.committed_seq(),
            &table,
            id,
            &[id, value],
            &[1],
        )
    });
    reached.wait();
    e.flag_residency_descriptors_invalidated(
        "invalidation_race",
        9_999,
        e.committed_seq().saturating_add(1),
    );
    let invalid_generation = std::sync::Arc::clone(
        &e.read_state.residency.shards.load()["invalidation_race"][0].point_route_generation,
    );
    assert!(!std::sync::Arc::ptr_eq(
        &old_generation,
        &invalid_generation
    ));
    resume.wait();
    assert!(
        reader.join().unwrap().unwrap().is_none(),
        "a plan whose descriptor generation was invalidated before pre-publish declines before submission"
    );
    assert!(
        e.read_state
            .residency
            .sharded_point_route_for_table("invalidation_race")
            .is_none(),
        "G0 cannot republish over the invalidated G1 token"
    );
}

/// A same-OID DDL publication clears the previous route but keeps the relation name and column positions
/// plausible. A stale caller must therefore fail the complete-shape fence before GPU construction, and it
/// must not fast-hit the current route that a newer boundary later installs. The Arc retained before the cut
/// remains independently executable: this distinguishes safe pre-cut ownership from post-cut cache reuse.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_same_oid_ddl_fences_stale_boundary_before_build_or_launch() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE same_oid_point_fence (id INT PRIMARY KEY, value INT)",
    )
    .unwrap();
    e.execute_text(2, "INSERT INTO same_oid_point_fence VALUES (1, 10)")
        .unwrap();
    let old_boundary = e.committed_seq();
    let old_table = e.relational_catalog_table("same_oid_point_fence").unwrap();
    let id = crate::rel_exec_helpers::relational_column_index(&old_table, "id").unwrap();
    let value = crate::rel_exec_helpers::relational_column_index(&old_table, "value").unwrap();

    e.gather_sharded_int4_point_lookups_batched(
        old_boundary,
        &old_table,
        id,
        &[id, value],
        &[1],
    )
    .expect("pre-cut route completed")
    .expect("pre-cut route served");
    let pre_cut_route = e
        .read_state
        .residency
        .sharded_point_route_for_table("same_oid_point_fence")
        .expect("retain the warmed pre-cut route Arc");

    e.execute_text(
        3,
        "CREATE INDEX same_oid_point_fence_value ON same_oid_point_fence (value)",
    )
    .unwrap();
    let current_table = e.relational_catalog_table("same_oid_point_fence").unwrap();
    assert_eq!(current_table.oid, old_table.oid, "CREATE INDEX preserves table OID");
    assert!(
        e.read_state
            .residency
            .sharded_point_route_for_table("same_oid_point_fence")
            .is_none(),
        "same-OID catalog-shape publication clears the old slot route"
    );

    // The stale descriptor has the same name/OID and its slot was just cleared. Phase 1 is consumed only by
    // a GPU construction attempt, so the following current caller proves this stale call declined before it
    // built or launched anything.
    e.force_next_sharded_point_cuda_failure(1);
    assert!(
        e.gather_sharded_int4_point_lookups_batched(
            old_boundary,
            &old_table,
            id,
            &[id, value],
            &[1],
        )
        .expect("stale same-OID call completed")
        .is_none(),
        "a cleared slot plus stale same-OID table fails before route build"
    );
    let current_boundary = e.committed_seq();
    let injected = e
        .gather_sharded_int4_point_lookups_batched(
            current_boundary,
            &current_table,
            id,
            &[id, value],
            &[1],
        )
        .expect_err("the stale call must leave phase-1 failure for the current builder");
    assert!(
        injected
            .to_string()
            .contains("GPU prepared shard point-route construction"),
        "phase 1 proves the current call, not the stale call, reached construction"
    );

    // A retained G0 Arc owns all of its device resources and may finish after the cut. It is not a cache hit
    // through the G1 table slot, and therefore cannot make a post-DDL caller observe stale catalog shape.
    let (pre_cut_cols, _) = pre_cut_route
        .launch_resident
        .submit_prepared_multi_shard_i32_index_probe_dense(&pre_cut_route.plan, &[1], old_boundary)
        .expect("retained pre-cut plan submits independently")
        .complete_detached_columnar_compact()
        .expect("retained pre-cut plan completes independently");
    assert_eq!(pre_cut_cols.status(), &[1]);

    let current = e
        .gather_sharded_int4_point_lookups_batched(
            current_boundary,
            &current_table,
            id,
            &[id, value],
            &[1],
        )
        .expect("current same-OID route completed")
        .expect("current same-OID route served");
    assert_eq!(current.values, vec![1, 10]);
    e.gather_sharded_int4_point_lookups_batched(
        current_boundary,
        &current_table,
        id,
        &[id, value],
        &[1],
    )
    .expect("current retained route completed")
    .expect("current retained route served");
    let cache_hits_before_stale = e.sharded_point_route_cache_hits();

    // G1 has now rebuilt the same route key and generation for the same OID. The `latest_catalog` boundary
    // guard forces this old S caller through `ensure_table_point_slot`, whose full relation comparison rejects
    // the old index shape; it cannot fast-hit G1's route.
    assert!(
        e.gather_sharded_int4_point_lookups_batched(
            old_boundary,
            &old_table,
            id,
            &[id, value],
            &[1],
        )
        .expect("old-boundary call against G1 route completed")
        .is_none(),
        "old boundary cannot fast-hit a current same-key route after same-OID DDL"
    );
    assert_eq!(
        e.sharded_point_route_cache_hits(),
        cache_hits_before_stale,
        "stale boundary did not consume the current cached route"
    );
}

/// NULL eligibility and route identity must consume one shard-map publication. A NULL-bearing G1 published
/// after G0's eligibility check makes the G0 attempt decline at its global-token recheck; it can never combine
/// G0 metadata with G1 raw payload placeholders.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_null_gate_and_route_capture_share_one_generation() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE null_race (id INT PRIMARY KEY, value INT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO null_race (id, value) VALUES (1, 10)")
        .unwrap();
    let table = e.relational_catalog_table("null_race").unwrap();
    let id = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let value = crate::rel_exec_helpers::relational_column_index(&table, "value").unwrap();
    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    e.set_sharded_point_after_eligibility_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    let reader_engine = std::sync::Arc::clone(&e);
    let reader = std::thread::spawn(move || {
        reader_engine.gather_sharded_int4_point_lookups_batched(
            reader_engine.committed_seq(),
            &table,
            id,
            &[id, value],
            &[1],
        )
    });
    reached.wait();
    e.execute_text(3, "INSERT INTO null_race (id, value) VALUES (2, NULL)")
        .unwrap();
    resume.wait();
    assert!(
        reader.join().unwrap().unwrap().is_none(),
        "captured NULL-free G0 cannot be paired with NULL-bearing G1"
    );
    assert!(
        e.gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &e.relational_catalog_table("null_race").unwrap(),
            id,
            &[id, value],
            &[2],
        )
        .unwrap()
        .is_none(),
        "the current NULL-bearing generation remains ineligible"
    );
}

/// Durable descriptor publication joins the same hard-budget allocation transaction as admission and lazy
/// indexes. Holding that transaction after plan preparation must prevent route-cache publication.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_point_route_publication_waits_for_budget_transaction() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE route_budget (id INT PRIMARY KEY, value INT)",
    )
    .unwrap();
    e.execute_text(2, "INSERT INTO route_budget (id, value) VALUES (1, 10)")
        .unwrap();
    let table = e.relational_catalog_table("route_budget").unwrap();
    let id = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let value = crate::rel_exec_helpers::relational_column_index(&table, "value").unwrap();
    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    e.set_sharded_point_route_pre_publish_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let reader_engine = std::sync::Arc::clone(&e);
    std::thread::spawn(move || {
        let result = reader_engine.gather_sharded_int4_point_lookups_batched(
            reader_engine.committed_seq(),
            &table,
            id,
            &[id, value],
            &[1],
        );
        done_tx.send(result).unwrap();
    });
    reached.wait();
    let budget = e
        .read_state
        .residency
        .budget_allocation_lock
        .lock()
        .unwrap();
    resume.wait();
    assert!(
        done_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err(),
        "route publication must wait behind the active budget transaction"
    );
    assert_eq!(e.read_state.residency.sharded_point_route_count(), 0);
    drop(budget);
    assert_eq!(
        done_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .unwrap()
            .unwrap()
            .values,
        vec![1, 10]
    );
    assert_eq!(e.read_state.residency.sharded_point_route_count(), 1);
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
        .expect("batched GPU route completed")
        .expect("batched served (delete-free)");
    assert!(
        k.sharded_point_gpu_probe_hits() > gpu_hb,
        "the GPU-native probe fired"
    );
    assert!(
        k.sharded_point_binary_route_hits() > bin_hb,
        "the O(1) BINARY-SEARCH route fired (>= 8 disjoint shards)"
    );
    for (i, &needle) in needles.iter().enumerate() {
        let (start, count) = proj.needle_range(i);
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
    let before_delete = t.committed_seq();
    t.execute_text(300, "DELETE FROM accounts WHERE id = 130")
        .unwrap();
    let table = t.relational_catalog_table("accounts").unwrap();
    let id_col = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let bal_col = crate::rel_exec_helpers::relational_column_index(&table, "balance").unwrap();

    // Force the first point-route index build to occur after DELETE committed and build the CURRENT route
    // first. That narrower index legitimately omits id=130, but it must not cache-hit for the subsequently
    // resumed old-boundary reader.
    t.read_state
        .residency
        .purge_shard_pk_index_for_table("accounts");
    let current_deleted = t
        .gather_sharded_int4_point_lookups_batched(
            t.committed_seq(),
            &table,
            id_col,
            &[id_col, bal_col],
            &[130],
        )
        .expect("current-boundary GPU route completed")
        .expect("current-boundary deleted-row probe served");
    assert_eq!(
        current_deleted.needle_range(0).1,
        0,
        "the current route may omit the row deleted at its boundary"
    );

    let replaced_index = {
        let cache = t
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::sync::Arc::downgrade(
            cache
                .iter()
                .find(|((cached_table, _, _), entry)| {
                    cached_table == "accounts" && entry.device_index.is_some()
                })
                .and_then(|(_, entry)| entry.device_index.as_ref())
                .expect("current route retained a device index"),
        )
    };

    // The cache must reject that newer-boundary route and rebuild the same generation/shape at the reader's
    // exact older boundary. Pause after its plan is ready: index replacement must already have retired the
    // current route, so that route cannot invisibly pin the replaced allocation outside hard-budget accounting.
    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    t.set_sharded_point_route_pre_publish_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    let old_deleted = std::thread::scope(|scope| {
        let reader = scope.spawn(|| {
            t.gather_sharded_int4_point_lookups_batched(
                before_delete,
                &table,
                id_col,
                &[id_col, bal_col],
                &[130],
            )
        });
        reached.wait();
        assert!(
            t.read_state
                .residency
                .sharded_point_route_for_table("accounts")
                .is_none(),
            "index replacement retires the route before removing its accounted index owner"
        );
        assert!(
            replaced_index.upgrade().is_none(),
            "no globally retained route invisibly pins the replaced index"
        );
        resume.wait();
        reader.join().expect("old-boundary reader thread")
    })
    .expect("old-boundary GPU route completed")
    .expect("old-boundary deleted-row probe served");
    assert_eq!(
        old_deleted.values,
        vec![130, 1_300],
        "a row deleted after the pinned boundary remains index-addressable and visible"
    );

    // A losing preparer can pin an index between ensure() and route publication. Rebuild a narrow current
    // index, clear only its route, pause a current-boundary two-column plan, then let an older one-column shape
    // replace the index and publish. The paused plan must fail the under-lock index-identity check, decline
    // before submission, and leave the older, accounted route in the global cache.
    t.read_state
        .residency
        .purge_shard_pk_index_for_table("accounts");
    t.gather_sharded_int4_point_lookups_batched(
        t.committed_seq(),
        &table,
        id_col,
        &[id_col, bal_col],
        &[130],
    )
    .expect("rebuild current-boundary route")
    .expect("current-boundary route served");
    t.read_state.residency.clear_sharded_point_routes_for_test();
    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    t.set_sharded_point_route_pre_publish_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    std::thread::scope(|scope| {
        let current_reader = scope.spawn(|| {
            t.gather_sharded_int4_point_lookups_batched(
                t.committed_seq(),
                &table,
                id_col,
                &[id_col, bal_col],
                &[130],
            )
        });
        reached.wait();
        let older_shape = t
            .gather_sharded_int4_point_lookups_batched(
                before_delete,
                &table,
                id_col,
                &[id_col],
                &[130],
            )
            .expect("older competing route completed")
            .expect("older competing route served");
        assert_eq!(older_shape.values, vec![130]);
        resume.wait();
        assert!(
            current_reader
                .join()
                .expect("current-boundary reader thread")
                .expect("current-boundary losing plan completed")
                .is_none(),
            "a losing plan must not submit after its prepared index was replaced"
        );
    });
    assert!(
        t.read_state
            .residency
            .sharded_point_route_for_table("accounts")
            .is_some_and(|route| route.route_key.1.as_slice() == [id_col]),
        "the losing newer-boundary plan cannot replace the accounted older route"
    );

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
        .expect("batched GPU route completed")
        .expect("batched served");
    assert!(
        t.sharded_point_gpu_probe_hits() > gpu_hb,
        "versioned shard stays on the fully-GPU dense path"
    );
    let count = |i: usize| proj.needle_range(i).1;
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
        .expect("batched GPU route completed")
        .expect("prior-snapshot dense probe served");
    assert_eq!(old.needle_range(0).1, 0, "future-born row hidden");
    let current = t
        .gather_sharded_int4_point_lookups_batched(
            t.committed_seq(),
            &table,
            id_col,
            &[id_col, bal_col],
            &[201],
        )
        .expect("batched GPU route completed")
        .expect("current-snapshot dense probe served");
    assert_eq!(current.needle_range(0).1, 1, "born row visible now");
    assert!(
        t.sharded_point_gpu_probe_hits() >= gpu_before_birth_checks + 2,
        "both snapshot-bound checks fired the fully-GPU dense probe"
    );
}

/// CPU-visible ownership guard for the latency fast path. The ignored GPU race covers runtime behavior; this
/// guard keeps the critical source ordering explicit when that hardware gate is unavailable.
#[test]
fn sharded_point_cached_route_fast_path_keeps_shape_and_submission_fences() {
    let source = include_str!("../engine_retained_read/shard_point_lookup.rs");
    let function = &source[source
        .find("fn gather_sharded_int4_point_lookups_batched_gpu(")
        .expect("sharded point helper exists")..];
    let fast_path = function
        .find("let fast_cached_route")
        .expect("exact cached-route fast path exists");
    let full_shape_fence = function
        .find("ensure_table_point_slot(&self.read_state, table)")
        .expect("all non-fast paths use complete table equality");
    let prepare = function
        .find("let prepared_route")
        .expect("GPU plan preparation exists");
    assert!(
        fast_path < full_shape_fence && full_shape_fence < prepare,
        "the fast path is distinct, and every miss reaches full-shape ensure before GPU preparation"
    );
    let fast_source = &function[fast_path..full_shape_fence];
    for required in [
        "latest_catalog.commit_seq == read_boundary",
        "table_point_slot(&table.name, table.oid)",
        "table_point_slot_is_current",
        "entry.route_key == route_key",
        "Arc::ptr_eq(&entry.table_generation, &table_generation)",
        "entry.read_boundary <= read_boundary",
        "memory_pressured_gpu_ids",
    ] {
        assert!(
            fast_source.contains(required),
            "fast cached route retains required proof: {required}"
        );
    }
    let final_currentness = function
        .find("if !(generation_is_current")
        .expect("final under-lock currentness gate exists");
    let submit = function
        .find("submit_prepared_multi_shard_i32_index_probe_dense")
        .expect("GPU submission exists");
    let final_source = &function[final_currentness..submit];
    assert!(
        final_source.contains("slot_is_current")
            && final_source.contains("catalog_is_current")
            && final_source.contains("indexes_are_current")
            && final_source.contains("return Ok(None);"),
        "any final authority drift declines before the prepared plan can submit"
    );
}
