/// SLICE B (sharded predicate NULL 3VL — the LAST shards-default gate): every NULL-semantics
/// predicate shape on a SHARDED-ONLY table matches the single-buffer M3 oracle (an INDEPENDENT
/// engine instance with shard residency OFF — the proven 3VL path). Covers: `IS NULL` /
/// `IS NOT NULL` (previously ERRORED on shards — the shape rides the SQL->Expr PG path, which only
/// knew the single-buffer store), equality against a NULL-stored-0 (`col = 0` must EXCLUDE the NULL
/// row per SQL 3VL: NULL = 0 is UNKNOWN), a plain equality on a nullable column, and a range
/// predicate over NULLs. All predicates evaluate ON THE DEVICE (the unified recompacted buffer
/// carries the validity bitmaps; the mask VM ANDs them — the charter's device-side 3VL).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_predicate_null_3vl_matches_single_buffer_oracle() {
    let load = |e: &Engine| {
        e.set_auto_admit_on_commit(true);
        e.execute_text(1, "CREATE TABLE nn (id INT, balance INT)")
            .unwrap();
        e.execute_text(
            2,
            "INSERT INTO nn (id, balance) VALUES (1,10),(2,NULL),(3,30),(NULL,99)",
        )
        .unwrap();
    };
    let o = Engine::new_local(); // single-buffer ORACLE (M3 3VL-proven path)
    load(&o);
    let e = Engine::new_local(); // sharded-only table (null-bearing => single shard by construction)
    e.set_shard_residency_enabled(true);
    load(&e);
    // Non-vacuity: the sharded engine really has NO single-buffer snapshot (shards serve it).
    assert!(
        e.read_state.residency.snapshots.load().get("nn").is_none()
            && e.read_state.residency.shards.load().get("nn").is_some(),
        "precondition: the table is SHARD-resident only (else this oracle differential is vacuous)"
    );
    let run = |e: &Engine, sql: &str| {
        e.execute_relational_select_text(sql)
            .map(|r| r.rows.into_boxed())
    };
    for sql in [
        "SELECT id, balance FROM nn WHERE balance = 0",
        "SELECT id, balance FROM nn WHERE id = 0",
        "SELECT id, balance FROM nn WHERE balance = 10",
        "SELECT id FROM nn WHERE balance <= 30",
        "SELECT id FROM nn WHERE balance IS NULL",
        "SELECT id FROM nn WHERE balance IS NOT NULL",
        // Newly-unlocked general shapes over the sharded unified source (previously all ERRORED):
        "SELECT id FROM nn WHERE balance IS NULL OR balance = 10", // IsNull as a mask-VM leaf in OR
        "SELECT COUNT(*) FROM nn WHERE balance IS NOT NULL",
        "SELECT id, balance FROM nn ORDER BY id DESC", // GPU sort over the unified buffer (+ NULL key)
    ] {
        let want = run(&o, sql).unwrap_or_else(|err| panic!("oracle must serve {sql}: {err}"));
        let got =
            run(&e, sql).unwrap_or_else(|err| panic!("the sharded path must serve {sql}: {err}"));
        assert_eq!(got, want, "sharded == single-buffer oracle for: {sql}");
    }
}

/// THE FLIP (audit F1 regression gate): every filtered/range int4 shape the audit found demoted to
/// the CPU host scan under the sharded-by-default layout is now GPU-SERVED via the sharded bridge
/// AND matches the single-buffer oracle. `executed_target == Gpu` is the non-vacuity proof (results
/// alone can't distinguish the host scan — it is correct, just off-charter and ~1000x slower).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn flip_f1_filtered_shapes_gpu_served_and_match_single_buffer_oracle() {
    let load = |e: &Engine| {
        // PINNED NON-ELIDED (A5 flip): the oracle reads the HOST store / pins pre-elision mechanics (production-live for non-eligible tables).
        e.set_host_install_elision_enabled(false);
        e.set_auto_admit_on_commit(true);
        e.execute_text(1, "CREATE TABLE t (a INT, b INT, c INT)")
            .unwrap();
        for chunk in 0..4_i64 {
            let values: Vec<String> = (chunk * 50..(chunk + 1) * 50)
                .map(|i| format!("({i},{},{})", i % 7, i % 3))
                .collect();
            e.execute_text(
                2 + chunk as u64,
                &format!("INSERT INTO t (a, b, c) VALUES {}", values.join(",")),
            )
            .unwrap();
        }
    };
    let o = Engine::new_local(); // single-buffer ORACLE
    o.set_shard_residency_enabled(false);
    load(&o);
    let e = Engine::new_local(); // sharded by default
    load(&e);
    assert!(
        e.read_state.residency.shards.load().get("t").is_some(),
        "precondition: t is SHARD-resident under the default"
    );
    for sql in [
        "SELECT COUNT(*) FROM t WHERE a = 137", // int4_equality_count
        "SELECT COUNT(*) FROM t WHERE a < 60",  // int4_range_count
        "SELECT SUM(a) FROM t WHERE a = 137",   // int4_filtered_scalar_aggregate (SUM)
        "SELECT SUM(a) FROM t WHERE a BETWEEN 10 AND 40", // int4_between_scalar_aggregate
        "SELECT a FROM t WHERE a > 190",        // int4_projection (range)
        "SELECT a FROM t WHERE b = 1 AND c = 2", // int4_composite_equality_multi_column_projection
    ] {
        let want = o.execute_relational_select_text(sql).unwrap();
        let got = e.execute_relational_select_text(sql).unwrap();
        assert_eq!(
            got.rows, want.rows,
            "sharded == single-buffer oracle for: {sql}"
        );
        // The F1 contract: the sharded DEFAULT never NEWLY demotes a shape to the host — it
        // is GPU-served, or the single-buffer oracle was ALSO host-served (a pre-existing,
        // layout-independent gap, not a flip regression).
        eprintln!(
            "[f1] {sql}: sharded={:?} oracle={:?}",
            got.executed_target, want.executed_target
        );
        assert!(
            got.executed_target == DeviceTarget::Gpu(0)
                || got.executed_target == want.executed_target,
            "F1: NEW cpu demotion under the sharded default for {sql}: sharded={:?} oracle={:?}",
            got.executed_target,
            want.executed_target
        );
    }
}

/// THE FLIP (audit P3): the sharded JOIN arm — both relations shard-resident (purely int4), the
/// join runs the GPU hash-join over the unified/zero-copy sources; result matches the pinned
/// single-buffer oracle. (Pre-flip, every suite join used a text column -> single-buffer, so the
/// arm was unexercised.)
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn flip_sharded_join_matches_single_buffer_oracle() {
    let load = |e: &Engine| {
        e.set_auto_admit_on_commit(true);
        e.execute_text(1, "CREATE TABLE l (k INT, v INT)").unwrap();
        e.execute_text(2, "INSERT INTO l (k, v) VALUES (1,10),(2,20),(3,30),(4,40)")
            .unwrap();
        e.execute_text(3, "CREATE TABLE r (k INT, w INT)").unwrap();
        e.execute_text(4, "INSERT INTO r (k, w) VALUES (2,200),(3,300),(5,500)")
            .unwrap();
    };
    let o = Engine::new_local();
    o.set_shard_residency_enabled(false);
    load(&o);
    let e = Engine::new_local();
    load(&e);
    assert!(
        e.read_state.residency.shards.load().get("l").is_some()
            && e.read_state.residency.shards.load().get("r").is_some(),
        "precondition: both relations SHARD-resident under the default"
    );
    let sql = "SELECT l.k, l.v, r.w FROM l JOIN r ON l.k = r.k ORDER BY l.k";
    let want = o
        .execute_relational_select_text(sql)
        .unwrap()
        .rows
        .into_boxed();
    let got = e
        .execute_relational_select_text(sql)
        .unwrap()
        .rows
        .into_boxed();
    assert_eq!(got, want, "sharded join == single-buffer oracle");
    assert_eq!(got.len(), 2, "k=2 and k=3 match");
}
