use super::*;

#[test]
fn resident_snapshot_probe_reads_valid_snapshot_and_rejects_invalidated_state() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    let Command::Select(select) =
        parse_command("SELECT label FROM events WHERE id = 2 LIMIT 1").unwrap()
    else {
        panic!("expected SELECT");
    };
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert!(snapshot.is_valid());
    // Host rows now live in the separate `host_rows` half of the residency entry (Option C split).
    assert_eq!(e.relational_residency_entry("events").unwrap().host_rows.len(), 2);

    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_select_with_resident_snapshot_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();
    // Closed-form oracle (S9): `SELECT label FROM events WHERE id = 2 LIMIT 1` over {1:'alpha',2:'beta'}.
    assert_eq!(resident.rows, vec![vec![SqlValue::Text("beta".to_string())]]);
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);

    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    assert!(e
        .execute_relational_select_with_resident_snapshot_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));

    e.populate_relational_residency_snapshot("events").unwrap();
    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_select_with_resident_snapshot_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn resident_snapshot_probe_reads_aggregate_distinct_without_transfer() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (id INT, label TEXT, amount INT, category TEXT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount, category) VALUES (1, 'alpha', 10, 'odd'), (2, 'beta', 20, 'even'), (3, 'gamma', 30, 'odd')",
        )
        .unwrap();
    // Closed-form construction oracles (S9): data is {1:'alpha',odd,10},{2:'beta',even,20},
    // {3:'gamma',odd,30} -> odd={10,30} (count 2, sum 40), even={20} (count 1, sum 20).
    let queries: [(&str, Vec<Vec<SqlValue>>); 6] = [
        (
            "SELECT DISTINCT category FROM events ORDER BY category",
            vec![
                vec![SqlValue::Text("even".to_string())],
                vec![SqlValue::Text("odd".to_string())],
            ],
        ),
        (
            "SELECT category, COUNT(*) FROM events GROUP BY category ORDER BY count DESC",
            vec![
                vec![SqlValue::Text("odd".to_string()), SqlValue::Int8(2)],
                vec![SqlValue::Text("even".to_string()), SqlValue::Int8(1)],
            ],
        ),
        (
            "SELECT category, SUM(amount) FROM events GROUP BY category ORDER BY sum DESC",
            vec![
                vec![SqlValue::Text("odd".to_string()), SqlValue::Int8(40)],
                vec![SqlValue::Text("even".to_string()), SqlValue::Int8(20)],
            ],
        ),
        (
            "SELECT AVG(amount) FROM events WHERE category = 'odd'",
            vec![vec![crate::rel_exec_helpers::average_sql_value(40, 2)]],
        ),
        ("SELECT MIN(amount) FROM events", vec![vec![SqlValue::Int4(10)]]),
        ("SELECT MAX(amount) FROM events", vec![vec![SqlValue::Int4(30)]]),
    ];

    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert!(snapshot.is_valid());

    for (sql, expected) in queries {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            panic!("expected SELECT");
        };
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_select_with_resident_snapshot_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.rows, expected, "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0, "{sql}");
        assert_eq!(after.d2h_bytes_total - before.d2h_bytes_total, 0, "{sql}");
        assert_eq!(
            after.kernel_exec_samples - before.kernel_exec_samples,
            0,
            "{sql}"
        );
    }
}

#[test]
fn resident_snapshot_budget_evicts_oldest_table_before_admission() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE small_a (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO small_a (id, label) VALUES (1, 'a')")
        .unwrap();
    let small_a = e.populate_relational_residency_snapshot("small_a").unwrap();

    e.execute_text(3, "CREATE TABLE small_b (id INT, label TEXT)")
        .unwrap();
    e.execute_text(4, "INSERT INTO small_b (id, label) VALUES (2, 'b')")
        .unwrap();
    let small_b = e.populate_relational_residency_snapshot("small_b").unwrap();

    let budget_bytes = small_b.resident_bytes.saturating_mul(2);
    e.set_relational_residency_budget_bytes(0, budget_bytes);
    e.execute_text(5, "CREATE TABLE small_c (id INT, label TEXT)")
        .unwrap();
    e.execute_text(6, "INSERT INTO small_c (id, label) VALUES (3, 'c')")
        .unwrap();
    let small_c = e.populate_relational_residency_snapshot("small_c").unwrap();

    assert_eq!(small_c.admission_budget_bytes, Some(budget_bytes));
    assert_eq!(small_c.evicted_tables_on_admission, vec!["small_a"]);
    assert!(e.relational_residency_snapshot("small_a").is_none());
    assert!(e.relational_residency_snapshot("small_b").is_some());
    assert!(e.relational_residency_snapshot("small_c").is_some());
    assert_eq!(
        small_c.resident_bytes_after_admission,
        e.relational_resident_bytes_for_gpu(0)
    );
    assert_eq!(small_a.valid_through_index, 2);
    assert_eq!(small_b.valid_through_index, 4);
}

#[test]
fn resident_snapshot_budget_rejects_oversized_snapshot_without_mutation() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO events (id, label) VALUES (1, 'alpha')")
        .unwrap();
    let original = e.populate_relational_residency_snapshot("events").unwrap();

    e.set_relational_residency_budget_bytes(0, original.resident_bytes - 1);
    let err = e
        .populate_relational_residency_snapshot("events")
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeding GPU 0 residency budget"));

    let retained = e.relational_residency_snapshot("events").unwrap();
    assert_eq!(retained, original);
    let status = e.status_snapshot();
    let table = status.relational_residency.table("events").unwrap();
    assert_eq!(table.cache_state, "Valid");
    assert_eq!(table.last_decision_accepted, Some(false));
    assert_eq!(
        table.last_decision_reason.as_deref(),
        Some("resident snapshot exceeds GPU budget")
    );
    assert_eq!(
        table.last_decision_current_bytes_before,
        Some(original.resident_bytes)
    );
    assert_eq!(
        table.last_decision_current_bytes_after,
        Some(original.resident_bytes)
    );
    assert_eq!(
        e.relational_resident_bytes_for_gpu(0),
        original.resident_bytes
    );
}

#[test]
fn resident_snapshot_budget_keeps_wal_and_pressure_invalidation_semantics() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    let original = e.populate_relational_residency_snapshot("events").unwrap();
    let budget_bytes = original.resident_bytes + 128;
    e.set_relational_residency_budget_bytes(0, budget_bytes);

    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    let invalidated = e.relational_residency_snapshot("events").unwrap();
    assert_eq!(invalidated.invalidated_by_txn_id, Some(3));
    assert!(!invalidated.is_valid());

    let refreshed = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(refreshed.admission_budget_bytes, Some(budget_bytes));
    assert!(refreshed.is_valid());

    e.mark_gpu_memory_pressured(0);
    let pressured = e.relational_residency_snapshot("events").unwrap();
    assert!(pressured.invalidated_by_memory_pressure);
    assert!(pressured.memory_pressure_active);
    assert!(!pressured.is_valid());
}

#[test]
fn resident_snapshot_records_absent_device_memory_proof_when_cuda_unavailable() {
    let mut e = Engine::new_local();
    let _ = e
        .cached_cuda_probe_runtime
        .set(CudaDriverRuntime::unavailable());
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO events (id, label) VALUES (1, 'alpha')")
        .unwrap();

    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(snapshot.device_memory_proof, None);
    assert_eq!(e.read_state.residency.device_memory.len(), 0);

    let status = e.status_snapshot();
    assert_eq!(
        status
            .relational_residency
            .table("events")
            .unwrap()
            .device_memory_proof,
        None
    );

    let Command::Select(select) = parse_command("SELECT COUNT(*) FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_resident_plan(&select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(filtered_select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id = 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_resident_plan(&filtered_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(text_prefix_select) =
        parse_command("SELECT COUNT(*) FROM events WHERE label LIKE 'a%'").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(&text_prefix_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(membership_select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id IN (1, 2)").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_membership_count_with_resident_device_memory_probe(&membership_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(range_select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id >= 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_resident_plan(&range_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(sum_select) = parse_command("SELECT SUM(id) FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_resident_plan(&sum_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(avg_select) = parse_command("SELECT AVG(id) FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_resident_plan(&avg_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(filtered_avg_select) =
        parse_command("SELECT AVG(id) FROM events WHERE id >= 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_resident_plan(
            &filtered_avg_select,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    // S8: grouped aggregates run via the `&Select`->general bridge (the resident-probe grouped methods
    // were retired); with no retained device memory it errors cleanly, like the other resident paths.
    let Command::Select(grouped_sum_select) =
        parse_command("SELECT id, SUM(id) FROM events GROUP BY id").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_resident_grouped_via_general(&grouped_sum_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(projection_select) =
        parse_command("SELECT id FROM events WHERE id >= 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_projection_with_resident_device_memory_probe(&projection_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(ordered_projection_select) =
        parse_command("SELECT id FROM events WHERE id >= 1 ORDER BY id DESC LIMIT 1").unwrap()
    else {
        unreachable!()
    };
    // S10a: the ordered projection now runs via the `&Select`->general bridge (the legacy
    // resident-probe ordered method was retired); with no retained device memory it errors cleanly,
    // like the grouped bridge above.
    let err = e
        .execute_resident_grouped_via_general(&ordered_projection_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(distinct_projection_select) =
        parse_command("SELECT DISTINCT id FROM events ORDER BY id").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(
            &distinct_projection_select,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));
}

#[test]
fn gpu_resident_device_memory_projection_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 20), (3, 'gamma', 30)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) =
        parse_command("SELECT amount FROM events WHERE amount >= 20").unwrap()
    else {
        unreachable!()
    };
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    // Closed-form oracle (S9): `SELECT amount WHERE amount >= 20` over amount=[10,20,30] (row order).
    assert_eq!(
        resident.rows,
        vec![vec![SqlValue::Int4(20)], vec![SqlValue::Int4(30)]]
    );
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        2 * std::mem::size_of::<i32>() as u64 + std::mem::size_of::<u64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    e.execute_text(
        3,
        "INSERT INTO events (id, label, amount) VALUES (4, 'delta', 40)",
    )
    .unwrap();
    assert!(e
        .execute_relational_projection_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_sum_probe_parallel_reduction_preserves_scalar_telemetry() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, amount INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, amount) VALUES (1, -5), (2, 0), (3, 7), (4, -2)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let Command::Select(select) = parse_command("SELECT SUM(amount) FROM events").unwrap() else {
        unreachable!()
    };
    let before = e.metrics().snapshot();
    let resident = e
        .execute_resident_plan(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    // Closed-form oracle (S9): SUM(amount) over [-5,0,7,-2] = 0.
    assert_eq!(resident.rows, vec![vec![SqlValue::Int8(0)]]);
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        std::mem::size_of::<i64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    e.execute_text(3, "CREATE TABLE empty_events (id INT, amount INT)")
        .unwrap();
    let empty_snapshot = e
        .populate_relational_residency_snapshot("empty_events")
        .unwrap();
    if empty_snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(empty_select) =
        parse_command("SELECT SUM(amount) FROM empty_events").unwrap()
    else {
        unreachable!()
    };
    let before_empty = e.metrics().snapshot();
    let empty_resident = e
        .execute_resident_plan(&empty_select)
        .unwrap();
    let after_empty = e.metrics().snapshot();

    // Closed-form oracle (S9): the legacy resident SUM probe returns Int8(0) (the reduction identity)
    // over an empty table -- not SQL NULL (the general executor's PG-correct empty result); this probe
    // path is behavior-preserved here and retired in S10.
    assert_eq!(empty_resident.rows, vec![vec![SqlValue::Int8(0)]]);
    assert_eq!(
        after_empty.h2d_bytes_total - before_empty.h2d_bytes_total,
        0
    );
    assert_eq!(
        after_empty.d2h_bytes_total - before_empty.d2h_bytes_total,
        std::mem::size_of::<i64>() as u64
    );
    assert_eq!(
        after_empty.kernel_exec_samples - before_empty.kernel_exec_samples,
        1
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_s10a_ordered_projection_routes_through_bridge_on_dispatch() {
    // S10a keeper: a single-int4-column ordered projection reaches the `int4_ordered_projection` route
    // arm through the LIVE `&Select` dispatch and now executes via the general bridge ON THE GPU, with
    // closed-form rows. Survives the probe's deletion (calls the dispatch, not the probe). Non-vacuous:
    // the literals pin the filtered + sorted + windowed result.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE p (a INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO p (a) VALUES (5), (1), (5), (-3), (0), (5), (2), (-3), (10), (1)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("p").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let cases: &[(&str, Vec<Vec<SqlValue>>)] = &[
        // a>=0 -> [5,1,5,0,5,2,1] sorted asc [0,1,1,2,5,5,5]; LIMIT 3 -> [0,1,1].
        (
            "SELECT a FROM p WHERE a >= 0 ORDER BY a LIMIT 3",
            vec![vec![SqlValue::Int4(0)], vec![SqlValue::Int4(1)], vec![SqlValue::Int4(1)]],
        ),
        // a>0 -> [5,1,5,5,2,10,1] sorted desc [10,5,5,5,2,1,1]; LIMIT 4 -> [10,5,5,5].
        (
            "SELECT a FROM p WHERE a > 0 ORDER BY a DESC LIMIT 4",
            vec![
                vec![SqlValue::Int4(10)],
                vec![SqlValue::Int4(5)],
                vec![SqlValue::Int4(5)],
                vec![SqlValue::Int4(5)],
            ],
        ),
        // a>=100 -> empty.
        ("SELECT a FROM p WHERE a >= 100 ORDER BY a LIMIT 5", vec![]),
    ];
    for (sql, expected) in cases {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let result = e.execute_relational_select(&select).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0), "off GPU: {sql}");
        assert_eq!(result.fallback_reason, None, "fell back: {sql}");
        assert_eq!(&result.rows, expected, "rows mismatch: {sql}");
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_s10a_ordered_projection_drops_nulls_pg_correct() {
    // S10a NULL regression (audit-adopted): the deleted `int4_ordered_projection` probe was NULL-BLIND --
    // it read the int4 column directly, so a NULL (stored as placeholder 0 + a cleared validity bit)
    // surfaced as a PHANTOM `Int4(0)` result row. The general bridge drops NULL rows via the 3VL WHERE
    // (a NULL fails `a >= k` -> UNKNOWN -> excluded), matching PostgreSQL and the engine's own SQL->Expr
    // path. So routing this shape is a PG-CORRECTNESS FIX, NOT byte-identical -- this test pins the
    // corrected behavior (the divergent axis the 480-shape non-null differential did not cover) and
    // cross-checks the bridge against the SQL->Expr reference on NULL data.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE p2 (a INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO p2 (a) VALUES (5), (1), (NULL), (5), (-3), (0), (NULL), (5), (2), (-3), (10), (1), (NULL)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("p2").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // a>=0 keeps the NON-NULL non-negative values 5,1,5,0,5,2,10,1 -> sorted asc [0,1,1,2,5,5,5,10].
    // Exactly ONE 0 (the real row); the 3 NULLs are NOT phantom 0s (that was the probe bug).
    let cases: &[(&str, Vec<Vec<SqlValue>>)] = &[
        (
            "SELECT a FROM p2 WHERE a >= 0 ORDER BY a LIMIT 100",
            [0, 1, 1, 2, 5, 5, 5, 10]
                .iter()
                .map(|&v| vec![SqlValue::Int4(v)])
                .collect(),
        ),
        // Windowed (same order): OFFSET 2 LIMIT 3 -> [1,2,5].
        (
            "SELECT a FROM p2 WHERE a >= 0 ORDER BY a LIMIT 3 OFFSET 2",
            [1, 2, 5].iter().map(|&v| vec![SqlValue::Int4(v)]).collect(),
        ),
    ];
    for (sql, expected) in cases {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let dispatch = e.execute_relational_select(&select).unwrap();
        assert_eq!(dispatch.executed_target, DeviceTarget::Gpu(0), "off GPU: {sql}");
        assert_eq!(dispatch.fallback_reason, None, "fell back: {sql}");
        assert_eq!(
            &dispatch.rows, expected,
            "NULL not dropped (probe phantom-0 bug regressed?): {sql}"
        );
        // The bridge (live dispatch) must match the engine's SQL->Expr general path byte-for-byte.
        let general = e.execute_resident_expr_select_sql(sql).unwrap();
        assert_eq!(dispatch.rows, general.rows, "bridge != general on NULL data: {sql}");
        assert_eq!(dispatch.columns, general.columns, "bridge cols != general: {sql}");
    }
}

#[test]
fn gpu_resident_device_memory_distinct_projection_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, bucket INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, bucket) VALUES (1, 'alpha', 2), (2, 'beta', 1), (3, 'gamma', 2), (4, 'delta', 3), (5, 'epsilon', 1)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) =
        parse_command("SELECT DISTINCT bucket FROM events ORDER BY bucket DESC LIMIT 2 OFFSET 1")
            .unwrap()
    else {
        unreachable!()
    };
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    // Closed-form oracle (S9): DISTINCT bucket = {1,2,3} sorted DESC = [3,2,1]; LIMIT 2 OFFSET 1 -> [2,1].
    assert_eq!(
        resident.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(1)]]
    );
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    // Kernel-less projection D2Hs only the i32 column (no device `out_count` readback): the
    // unfiltered distinct path reads exactly the value bytes. The `+ size_of::<u64>()` count term
    // was dropped in 38451a28 (which updated the plain-projection test but missed this DISTINCT
    // one); the filtered-distinct sibling still reads the u64 match-count, so it keeps the term.
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        5 * std::mem::size_of::<i32>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    let Command::Select(unsupported_text) =
        parse_command("SELECT DISTINCT label FROM events ORDER BY label").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(&unsupported_text)
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 projection columns"));

    let Command::Select(unsupported_offset_without_order) =
        parse_command("SELECT DISTINCT bucket FROM events OFFSET 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(
            &unsupported_offset_without_order,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("OFFSET proof currently requires same-column ORDER BY and LIMIT"));

    let Command::Select(unsupported_filter) =
        parse_command("SELECT DISTINCT bucket FROM events WHERE bucket >= 2").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(
            &unsupported_filter,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("optional same-column ORDER BY, LIMIT, and OFFSET"));

    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_filtered_distinct_projection_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, bucket INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, bucket) VALUES (1, 'alpha', 1), (2, 'beta', 2), (3, 'gamma', 2), (4, 'delta', 3), (5, 'epsilon', 4), (6, 'zeta', 4)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) = parse_command(
            "SELECT DISTINCT bucket FROM events WHERE bucket >= 2 ORDER BY bucket DESC LIMIT 2 OFFSET 1",
        )
        .unwrap() else {
            unreachable!()
        };
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    // Closed-form oracle (S9): bucket>=2 DISTINCT = {2,3,4} sorted DESC = [4,3,2]; LIMIT 2 OFFSET 1 -> [3,2].
    assert_eq!(
        resident.rows,
        vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(2)]]
    );
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        5 * std::mem::size_of::<i32>() as u64 + std::mem::size_of::<u64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    let Command::Select(unsupported_cross_column) = parse_command(
        "SELECT DISTINCT bucket FROM events WHERE id >= 2 ORDER BY bucket DESC LIMIT 2",
    )
    .unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(
            &unsupported_cross_column,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires predicate and projection to use the same int4 column"));

    let Command::Select(unsupported_offset_without_order) =
        parse_command("SELECT DISTINCT bucket FROM events WHERE bucket >= 2 OFFSET 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(
            &unsupported_offset_without_order,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("OFFSET proof currently requires same-column ORDER BY and LIMIT"));

    let Command::Select(unsupported_equality) =
        parse_command("SELECT DISTINCT bucket FROM events WHERE bucket = 2").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(
            &unsupported_equality,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only non-equality int4 comparisons"));

    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(&select,)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn relational_select_grouped_having_filters_engine_aggregate_rows() {
    let e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (bucket INT, label TEXT, amount INT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (bucket, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (2, 'epsilon', 5), (3, 'zeta', 15)",
        )
        .unwrap();

    let Command::Select(select) = parse_command(
            "SELECT bucket, SUM(amount) FROM events GROUP BY bucket HAVING bucket >= 2 AND sum > 40 ORDER BY sum DESC",
        )
        .unwrap() else {
            unreachable!()
        };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(3), SqlValue::Int8(55)]]
    );

    let Command::Select(or_select) = parse_command(
            "SELECT bucket, MAX(amount) FROM events GROUP BY bucket HAVING bucket = 1 OR max >= 40 ORDER BY bucket",
        )
        .unwrap() else {
            unreachable!()
        };
    let result = e.execute_relational_select(&or_select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(20)],
            vec![SqlValue::Int4(3), SqlValue::Int4(40)]
        ]
    );

    let Command::Select(unsupported_having) =
        parse_command("SELECT bucket, COUNT(*) FROM events GROUP BY bucket HAVING amount > 10")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_select(&unsupported_having)
        .unwrap_err()
        .to_string();
    assert!(err.contains("HAVING must reference grouped column or aggregate result"));
}

#[test]
fn gpu_resident_device_memory_membership_count_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (4, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    for sql in [
        "SELECT COUNT(*) FROM events WHERE id IN (1, 3, 99)",
        "SELECT COUNT(*) FROM events WHERE id IN (1, 1, 3)",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_membership_count_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        // Closed-form oracle (S9): ids=[1,2,1,3,4]; IN (1,3,99) and IN (1,1,3) both match {1,1,3} = 3.
        assert_eq!(resident.rows, vec![vec![SqlValue::Int8(3)]], "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.fallback_reason, None);
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    }

    let Command::Select(text_membership) =
        parse_command("SELECT COUNT(*) FROM events WHERE label IN ('alpha', 'delta')").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_membership_count_with_resident_device_memory_probe(&text_membership)
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 membership literals"));

    let Command::Select(cross_column) =
        parse_command("SELECT COUNT(*) FROM events WHERE id = 1 OR amount = 40").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_membership_count_with_resident_device_memory_probe(&cross_column)
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires all membership values to target the same column"));

    let Command::Select(equality_only) =
        parse_command("SELECT COUNT(*) FROM events WHERE id = 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_membership_count_with_resident_device_memory_probe(&equality_only)
        .unwrap_err()
        .to_string();
    assert!(err.contains("one int4 IN membership predicate"));

    e.mark_gpu_memory_pressured(0);
    let Command::Select(select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id IN (1, 3)").unwrap()
    else {
        unreachable!()
    };
    assert!(e
        .execute_relational_membership_count_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_between_count_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (3, 'gamma', 20), (4, 'delta', 40), (5, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // Closed-form oracles (S9): id=[1,2,3,4,5]; BETWEEN 2 AND 4 matches {2,3,4} = 3; the inverted
    // BETWEEN 4 AND 2 is an empty range = 0 (still executed on-device, hence the kernel/d2h asserts).
    for (sql, expected) in [
        ("SELECT COUNT(*) FROM events WHERE id BETWEEN 2 AND 4", SqlValue::Int8(3)),
        ("SELECT COUNT(*) FROM events WHERE id BETWEEN 4 AND 2", SqlValue::Int8(0)),
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_between_count_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.rows, vec![vec![expected]], "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0, "{sql}");
        assert!(after.d2h_bytes_total > before.d2h_bytes_total, "{sql}");
        assert_eq!(
            after.kernel_exec_samples - before.kernel_exec_samples,
            2,
            "{sql}"
        );
    }

    let Command::Select(cross_column) =
        parse_command("SELECT COUNT(*) FROM events WHERE id >= 2 AND amount <= 40").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_between_count_with_resident_device_memory_probe(&cross_column)
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires both range bounds to target the same column"));

    let Command::Select(text_bounds) =
        parse_command("SELECT COUNT(*) FROM events WHERE label >= 'beta' AND label <= 'gamma'")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_between_count_with_resident_device_memory_probe(&text_bounds)
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 bounds"));

    let Command::Select(equality_only) =
        parse_command("SELECT COUNT(*) FROM events WHERE id = 2").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_between_count_with_resident_device_memory_probe(&equality_only)
        .unwrap_err()
        .to_string();
    assert!(err.contains("one int4 BETWEEN predicate"));

    e.mark_gpu_memory_pressured(0);
    let Command::Select(select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id BETWEEN 2 AND 4").unwrap()
    else {
        unreachable!()
    };
    assert!(e
        .execute_relational_between_count_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_filter_group_count_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (3, 'gamma', 20), (4, 'delta', 40), (5, 'epsilon', 5), (6, 'zeta', 60)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // Closed-form oracles (S9): rows (id,amount) = (1,10),(2,30),(3,20),(4,40),(5,5),(6,60).
    // id>=2 AND amount<=40 -> {2,3,4,5} = 4; (id=1&amt>=10)|(id=4&amt<=40) -> {1,4} = 2;
    // id<=2 OR amount>=50 -> {1,2,6} = 3.
    for (sql, expected) in [
        ("SELECT COUNT(*) FROM events WHERE id >= 2 AND amount <= 40", SqlValue::Int8(4)),
        (
            "SELECT COUNT(*) FROM events WHERE (id = 1 AND amount >= 10) OR (id = 4 AND amount <= 40)",
            SqlValue::Int8(2),
        ),
        ("SELECT COUNT(*) FROM events WHERE id <= 2 OR amount >= 50", SqlValue::Int8(3)),
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_filter_group_count_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.rows, vec![vec![expected]], "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0, "{sql}");
        assert!(after.d2h_bytes_total > before.d2h_bytes_total, "{sql}");
        assert!(
            after.kernel_exec_samples > before.kernel_exec_samples,
            "{sql}"
        );
    }

    let Command::Select(unsupported_text_like) =
        parse_command("SELECT COUNT(*) FROM events WHERE label LIKE 'a%'").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filter_group_count_with_resident_device_memory_probe(
            &unsupported_text_like,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 literal predicates"));

    let Command::Select(unsupported_ordered) =
        parse_command("SELECT COUNT(*) FROM events WHERE id >= 2 ORDER BY count").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filter_group_count_with_resident_device_memory_probe(
            &unsupported_ordered,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("SELECT COUNT(*) with int4 WHERE filter groups"));

    e.mark_gpu_memory_pressured(0);
    let Command::Select(select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id >= 2 AND amount <= 40").unwrap()
    else {
        unreachable!()
    };
    assert!(e
        .execute_relational_filter_group_count_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_text_prefix_count_probe_materializes_text_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'alpine', 30), (3, 'beta', 20), (4, 'alphabet', 40), (5, 'gamma', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(
        snapshot
            .resident_device_text_columns
            .iter()
            .map(|layout| layout.name.as_str())
            .collect::<Vec<_>>(),
        vec!["label"]
    );
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT COUNT(*) FROM events WHERE label LIKE 'alp%'").unwrap()
    else {
        unreachable!()
    };
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    // Closed-form oracle (S9): labels {alpha,alpine,beta,alphabet,gamma} LIKE 'alp%' = 3.
    assert_eq!(resident.rows, vec![vec![SqlValue::Int8(3)]]);
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert!(after.d2h_bytes_total > before.d2h_bytes_total);
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 0);

    let Command::Select(unsupported_int4) =
        parse_command("SELECT COUNT(*) FROM events WHERE id LIKE '1%'").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(&unsupported_int4)
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only text predicates"));

    let Command::Select(unsupported_ordered) =
        parse_command("SELECT COUNT(*) FROM events WHERE label LIKE 'alp%' ORDER BY count")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(
            &unsupported_ordered,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("one text prefix LIKE predicate"));

    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_scalar_aggregate_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (bucket INT, label TEXT, amount INT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (bucket, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (2, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // Closed-form oracles (S9): amount=[10,30,20,40,5] -> AVG=105/5, MIN=5, MAX=40.
    for (sql, expected) in [
        (
            "SELECT AVG(amount) FROM events",
            crate::rel_exec_helpers::average_sql_value(105, 5),
        ),
        ("SELECT MIN(amount) FROM events", SqlValue::Int4(5)),
        ("SELECT MAX(amount) FROM events", SqlValue::Int4(40)),
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let before = e.metrics().snapshot();
        let resident = e
            .execute_resident_plan(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.rows, vec![vec![expected]], "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.fallback_reason, None);
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
        assert!(
            after.d2h_bytes_total > before.d2h_bytes_total,
            "{sql} should read aggregate stats from device memory"
        );
        assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);
    }

    let Command::Select(unsupported) = parse_command("SELECT AVG(label) FROM events").unwrap()
    else {
        unreachable!()
    };
    assert!(e
        .execute_resident_plan(&unsupported)
        .unwrap_err()
        .to_string()
        .contains("AVG supports int2 / int4 / int8 / numeric columns"));

    e.mark_gpu_memory_pressured(0);
    let Command::Select(select) = parse_command("SELECT MAX(amount) FROM events").unwrap() else {
        unreachable!()
    };
    assert!(e
        .execute_resident_plan(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_scalar_aggregate_skips_null_values_and_all_null_is_null() {
    // M3 (doc 21) end-to-end: unfiltered SUM/AVG/MIN/MAX over a nullable int4 column skip NULL rows ON
    // THE GPU (the validity bitmap is read on-device by the self-grouped stats kernel), and an all-NULL
    // column yields SQL NULL for every aggregate (PG: aggregate of no rows is NULL). Expected values are
    // closed-form construction (the non-NULL sum/count/min/max computed by hand, AVG via the engine's
    // own PG-exact finalization) — NOT a CPU-operator oracle, per the GPU-native-oracle charter.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (amount INT, allnull INT)")
        .unwrap();
    // amount: 10, NULL, 30, NULL, 20, 5  -> non-NULL {10, 30, 20, 5}: count 4, sum 65, min 5, max 30.
    // allnull: every row NULL.
    e.execute_text(
        2,
        "INSERT INTO events (amount, allnull) VALUES (10, NULL), (NULL, NULL), (30, NULL), (NULL, NULL), (20, NULL), (5, NULL)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // Partially-NULL column: NULL rows are excluded from every aggregate.
    let cases: [(&str, SqlValue); 4] = [
        ("SELECT SUM(amount) FROM events", SqlValue::Int8(65)),
        (
            "SELECT AVG(amount) FROM events",
            crate::rel_exec_helpers::average_sql_value(65, 4),
        ),
        ("SELECT MIN(amount) FROM events", SqlValue::Int4(5)),
        ("SELECT MAX(amount) FROM events", SqlValue::Int4(30)),
    ];
    for (sql, expected) in cases {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let resident = e.execute_resident_plan(&select).unwrap();
        assert_eq!(resident.rows, vec![vec![expected]], "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
    }

    // All-NULL column: SUM/AVG/MIN/MAX are SQL NULL (no surviving non-NULL rows).
    for sql in [
        "SELECT SUM(allnull) FROM events",
        "SELECT AVG(allnull) FROM events",
        "SELECT MIN(allnull) FROM events",
        "SELECT MAX(allnull) FROM events",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let resident = e.execute_resident_plan(&select).unwrap();
        assert_eq!(
            resident.rows,
            vec![vec![SqlValue::Null]],
            "{sql} over an all-NULL column must be NULL"
        );
    }

    // A FILTERED aggregate over a nullable column now runs on-device (Slice A): the `<= 10` predicate is
    // one the 0 placeholders WOULD pass, so the kernel's NULL-skip (not the filter) is what excludes the
    // NULL rows — MIN(amount) over the non-NULL rows ≤ 10 is {10, 5} ⇒ 5, never the phantom 0.
    let Command::Select(filtered) =
        parse_command("SELECT MIN(amount) FROM events WHERE amount <= 10").unwrap()
    else {
        unreachable!()
    };
    let resident = e.execute_resident_plan(&filtered).unwrap();
    assert_eq!(
        resident.rows,
        vec![vec![SqlValue::Int4(5)]],
        "filtered MIN over a nullable column excludes the NULL placeholder 0"
    );
}

#[test]
fn gpu_resident_filtered_scalar_aggregate_over_nullable_column_skips_null() {
    // M3 Slice A: filtered (compare + BETWEEN) SUM/AVG/MIN/MAX over a NULLABLE int4 column skip NULL rows
    // ON THE GPU, and a no-surviving-row result is SQL NULL. Closed-form construction oracle (not CPU).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (amount INT)").unwrap();
    // amount: 10, NULL, 30, NULL, 20, 5  -> non-NULL {10, 30, 20, 5}.
    e.execute_text(
        2,
        "INSERT INTO events (amount) VALUES (10), (NULL), (30), (NULL), (20), (5)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let cases: [(&str, SqlValue); 8] = [
        // compare-nullable: non-NULL rows passing the filter, NULL placeholders excluded by the kernel.
        ("SELECT SUM(amount) FROM events WHERE amount >= 20", SqlValue::Int8(50)), // {30,20}
        ("SELECT MIN(amount) FROM events WHERE amount >= 20", SqlValue::Int4(20)),
        ("SELECT MAX(amount) FROM events WHERE amount <= 10", SqlValue::Int4(10)), // {10,5}
        (
            "SELECT AVG(amount) FROM events WHERE amount <= 10",
            crate::rel_exec_helpers::average_sql_value(15, 2),
        ),
        // no surviving non-NULL row ⇒ SQL NULL (PG: aggregate of no rows is NULL).
        ("SELECT SUM(amount) FROM events WHERE amount >= 100", SqlValue::Null),
        ("SELECT MAX(amount) FROM events WHERE amount >= 100", SqlValue::Null),
        // BETWEEN-nullable: the [0,10] lower bound would admit the 0 placeholder, but the kernel's
        // NULL-skip excludes it, so MIN is 5 (a real value), not 0.
        ("SELECT MIN(amount) FROM events WHERE amount BETWEEN 0 AND 10", SqlValue::Int4(5)),
        ("SELECT SUM(amount) FROM events WHERE amount BETWEEN 100 AND 200", SqlValue::Null),
    ];
    for (sql, expected) in cases {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let resident = e.execute_resident_plan(&select).unwrap();
        assert_eq!(resident.rows, vec![vec![expected]], "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
    }
}

#[test]
fn gpu_resident_device_memory_filtered_scalar_aggregate_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (bucket INT, label TEXT, amount INT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (bucket, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (2, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // Closed-form oracles (S9): amount=[10,30,20,40,5]; amount>=20 -> {20,30,40} (sum 90, count 3).
    for (sql, expected) in [
        ("SELECT SUM(amount) FROM events WHERE amount >= 20", SqlValue::Int8(90)),
        (
            "SELECT AVG(amount) FROM events WHERE amount >= 20",
            crate::rel_exec_helpers::average_sql_value(90, 3),
        ),
        ("SELECT MIN(amount) FROM events WHERE amount >= 20", SqlValue::Int4(20)),
        ("SELECT MAX(amount) FROM events WHERE amount >= 20", SqlValue::Int4(40)),
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let before = e.metrics().snapshot();
        let resident = e
            .execute_resident_plan(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.rows, vec![vec![expected]], "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.fallback_reason, None);
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
        assert_eq!(
            after.d2h_bytes_total - before.d2h_bytes_total,
            (std::mem::size_of::<u64>()
                + std::mem::size_of::<i64>()
                + (2 * std::mem::size_of::<i32>())
                + std::mem::size_of::<u64>()) as u64,
            "{sql}"
        );
        assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);
    }

    let Command::Select(unsupported_text) =
        parse_command("SELECT MAX(label) FROM events WHERE label >= 'beta'").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_resident_plan(
            &unsupported_text,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 comparison literals"));

    let Command::Select(unsupported_cross_column) =
        parse_command("SELECT SUM(amount) FROM events WHERE bucket >= 2").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_resident_plan(
            &unsupported_cross_column,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires the predicate column to match the aggregate column"));

    let Command::Select(empty_max) =
        parse_command("SELECT MAX(amount) FROM events WHERE amount >= 1000").unwrap()
    else {
        unreachable!()
    };
    let before = e.metrics().snapshot();
    let resident = e
        .execute_resident_plan(&empty_max)
        .unwrap();
    let after = e.metrics().snapshot();
    // Closed-form oracle (S9): MAX over no surviving rows (amount >= 1000) returns this legacy probe's
    // empty-result sentinel -- an empty-text value (a quirk of the resident scalar-aggregate probe; the
    // general executor returns SQL NULL, see the M3 tests). Behavior-preserved here; retired in S10.
    assert_eq!(resident.rows, vec![vec![SqlValue::Text(String::new())]]);
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        std::mem::size_of::<i64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 0);

    let Command::Select(select) =
        parse_command("SELECT SUM(amount) FROM events WHERE amount >= 20").unwrap()
    else {
        unreachable!()
    };
    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_resident_plan(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_between_scalar_aggregate_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (bucket INT, label TEXT, amount INT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (bucket, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (2, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // Closed-form oracles (S9): amount=[10,30,20,40,5]; BETWEEN 10 AND 30 -> {10,20,30} (sum 60,
    // count 3); the inverted BETWEEN 40 AND 10 is an empty range -> SQL NULL.
    for (sql, expected) in [
        ("SELECT SUM(amount) FROM events WHERE amount BETWEEN 10 AND 30", SqlValue::Int8(60)),
        (
            "SELECT AVG(amount) FROM events WHERE amount BETWEEN 10 AND 30",
            crate::rel_exec_helpers::average_sql_value(60, 3),
        ),
        ("SELECT MIN(amount) FROM events WHERE amount BETWEEN 10 AND 30", SqlValue::Int4(10)),
        ("SELECT MAX(amount) FROM events WHERE amount BETWEEN 10 AND 30", SqlValue::Int4(30)),
        // Empty (inverted) range: this legacy resident-probe path returns Int8(0) for an empty SUM
        // (the scalar reduction's identity), NOT SQL NULL -- the general executor returns NULL (M3
        // tests); behavior-preserving here, the probe path is retired in S10.
        ("SELECT SUM(amount) FROM events WHERE amount BETWEEN 40 AND 10", SqlValue::Int8(0)),
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let before = e.metrics().snapshot();
        let resident = e
            .execute_resident_plan(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.rows, vec![vec![expected]], "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0, "{sql}");
        if sql.contains("40 AND 10") {
            assert_eq!(after.d2h_bytes_total - before.d2h_bytes_total, 0, "{sql}");
            assert_eq!(
                after.kernel_exec_samples - before.kernel_exec_samples,
                0,
                "{sql}"
            );
        } else {
            assert_eq!(
                after.d2h_bytes_total - before.d2h_bytes_total,
                (std::mem::size_of::<u64>()
                    + std::mem::size_of::<i64>()
                    + (2 * std::mem::size_of::<i32>())
                    + std::mem::size_of::<u64>()) as u64,
                "{sql}"
            );
            assert_eq!(
                after.kernel_exec_samples - before.kernel_exec_samples,
                1,
                "{sql}"
            );
        }
    }

    let Command::Select(unsupported_text) =
        parse_command("SELECT MAX(label) FROM events WHERE label BETWEEN 'beta' AND 'gamma'")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_resident_plan(
            &unsupported_text,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 bounds"));

    let Command::Select(unsupported_cross_column) =
        parse_command("SELECT SUM(amount) FROM events WHERE bucket BETWEEN 1 AND 2").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_resident_plan(
            &unsupported_cross_column,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires the predicate column to match the aggregate column"));

    let Command::Select(equality_only) =
        parse_command("SELECT SUM(amount) FROM events WHERE amount = 20").unwrap()
    else {
        unreachable!()
    };
    // Under the unified plan compiler a single equality filter on the aggregate column compiles via
    // the comparison path (the resident stats kernels evaluate only non-equality int4 comparisons),
    // so the rejection is the comparison-shape diagnostic rather than the old BETWEEN-arity one.
    let err = e
        .execute_resident_plan(&equality_only)
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only non-equality int4 comparisons"));

    let Command::Select(select) =
        parse_command("SELECT SUM(amount) FROM events WHERE amount BETWEEN 10 AND 30").unwrap()
    else {
        unreachable!()
    };
    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_resident_plan(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn status_and_telemetry_surface_relational_residency_state() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE aux (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO aux (id, label) VALUES (1, 'aux')")
        .unwrap();

    let events = e.populate_relational_residency_snapshot("events").unwrap();
    if let Some(proof) = &events.device_memory_proof {
        assert!(proof.retained);
    }
    let aux = e.populate_relational_residency_snapshot("aux").unwrap();
    let budget_bytes = events.resident_bytes;
    e.set_relational_residency_budget_bytes(0, budget_bytes);
    let admitted = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(admitted.evicted_tables_on_admission, vec!["aux"]);

    let status = e.status_snapshot();
    assert_eq!(status.resident_table_count(), 1);
    assert_eq!(status.relational_residency.snapshot_count(), 1);
    assert_eq!(status.relational_residency.valid_snapshot_count(), 1);
    assert_eq!(
        status.relational_residency.total_resident_bytes(),
        events.resident_bytes
    );
    assert_eq!(
        status.relational_residency.budget_bytes_by_gpu.get(&0),
        Some(&budget_bytes)
    );
    assert_eq!(
        status.relational_residency.resident_bytes_by_gpu.get(&0),
        Some(&events.resident_bytes)
    );
    let table = status.relational_residency.table("events").unwrap();
    assert_eq!(table.schema, "public");
    assert_eq!(table.gpu_id, 0);
    assert_eq!(table.row_count, 2);
    assert_eq!(table.column_count, 2);
    assert_eq!(table.resident_bytes, events.resident_bytes);
    assert_eq!(table.admission_budget_bytes, Some(budget_bytes));
    assert_eq!(table.resident_bytes_after_admission, events.resident_bytes);
    assert_eq!(table.evicted_tables_on_admission, vec!["aux"]);
    assert_eq!(table.cache_state, "Valid");
    assert_eq!(table.last_decision_accepted, Some(true));
    assert_eq!(
        table.last_decision_reason.as_deref(),
        Some("admitted after deterministic eviction")
    );
    assert_eq!(
        table.last_decision_current_bytes_before,
        Some(aux.resident_bytes)
    );
    assert_eq!(
        table.last_decision_current_bytes_after,
        Some(events.resident_bytes)
    );
    assert!(table.valid);
    assert!(status.relational_residency.table("aux").is_none());
    status.validate().unwrap();

    e.execute_text(5, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    let invalidated = e.telemetry_snapshot();
    let invalidated_table = invalidated.relational_residency.table("events").unwrap();
    assert_eq!(invalidated.resident_table_count(), 1);
    assert_eq!(invalidated_table.cache_state, "Invalidated");
    assert!(!invalidated_table.valid);
    assert_eq!(invalidated_table.invalidated_by_txn_id, Some(5));
    assert_eq!(invalidated.relational_residency.invalid_snapshot_count(), 1);
    if let Some(proof) = &invalidated_table.device_memory_proof {
        assert!(!proof.retained);
    }

    e.mark_gpu_memory_pressured(0);
    let pressured = e.status_snapshot();
    let pressured_table = pressured.relational_residency.table("events").unwrap();
    assert_eq!(pressured_table.cache_state, "InvalidatedByMemoryPressure");
    assert!(pressured_table.memory_pressure_active);
    assert!(pressured_table.invalidated_by_memory_pressure);
    if let Some(proof) = &pressured_table.device_memory_proof {
        assert!(!proof.retained);
    }
    assert_eq!(
        pressured
            .relational_residency
            .memory_pressured_snapshot_count(),
        1
    );
    assert_eq!(
        e.relational_resident_bytes_for_gpu(0),
        events.resident_bytes
    );
    assert!(aux.resident_bytes > 0);
}
