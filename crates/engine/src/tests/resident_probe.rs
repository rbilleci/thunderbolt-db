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
    let cpu = e.execute_relational_select(&select).unwrap();

    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert!(snapshot.is_valid());
    assert_eq!(snapshot.resident_rows.len(), 2);

    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_select_with_resident_snapshot_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(resident.rows, cpu.rows);
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
    let queries = [
        "SELECT DISTINCT category FROM events ORDER BY category",
        "SELECT category, COUNT(*) FROM events GROUP BY category ORDER BY count DESC",
        "SELECT category, SUM(amount) FROM events GROUP BY category ORDER BY sum DESC",
        "SELECT AVG(amount) FROM events WHERE category = 'odd'",
        "SELECT MIN(amount) FROM events",
        "SELECT MAX(amount) FROM events",
    ];

    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert!(snapshot.is_valid());

    for sql in queries {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            panic!("expected SELECT");
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_select_with_resident_snapshot_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
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
        .execute_relational_count_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(filtered_select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id = 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_count_with_resident_device_memory_probe(&filtered_select)
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
        .execute_relational_range_count_with_resident_device_memory_probe(&range_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(sum_select) = parse_command("SELECT SUM(id) FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_sum_with_resident_device_memory_probe(&sum_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(avg_select) = parse_command("SELECT AVG(id) FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_scalar_aggregate_with_resident_device_memory_probe(&avg_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(filtered_avg_select) =
        parse_command("SELECT AVG(id) FROM events WHERE id >= 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(
            &filtered_avg_select,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(grouped_sum_select) =
        parse_command("SELECT id, SUM(id) FROM events GROUP BY id").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_grouped_sum_with_resident_device_memory_probe(&grouped_sum_select)
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
    let err = e
        .execute_relational_ordered_projection_with_resident_device_memory_probe(
            &ordered_projection_select,
        )
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
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
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
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_sum_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
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
    let empty_cpu = e.execute_relational_select(&empty_select).unwrap();
    let before_empty = e.metrics().snapshot();
    let empty_resident = e
        .execute_relational_sum_with_resident_device_memory_probe(&empty_select)
        .unwrap();
    let after_empty = e.metrics().snapshot();

    assert_eq!(empty_resident.columns, empty_cpu.columns);
    assert_eq!(empty_resident.rows, empty_cpu.rows);
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
fn gpu_resident_device_memory_ordered_projection_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (3, 'gamma', 20), (4, 'delta', 40)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) = parse_command(
        "SELECT amount FROM events WHERE amount >= 20 ORDER BY amount DESC LIMIT 2 OFFSET 1",
    )
    .unwrap() else {
        unreachable!()
    };
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_ordered_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(
        resident.rows,
        vec![vec![SqlValue::Int4(30)], vec![SqlValue::Int4(20)]]
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

    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_ordered_projection_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
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
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
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
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
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
fn gpu_resident_device_memory_grouped_aggregate_probe_materializes_int4_results() {
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
    let Command::Select(select) = parse_command(
        "SELECT bucket, SUM(amount) FROM events GROUP BY bucket ORDER BY sum DESC LIMIT 2",
    )
    .unwrap() else {
        unreachable!()
    };
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_grouped_sum_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(
        resident.rows,
        vec![
            vec![SqlValue::Int4(3), SqlValue::Int8(40)],
            vec![SqlValue::Int4(2), SqlValue::Int8(35)]
        ]
    );
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        3 * (std::mem::size_of::<i32>()
            + std::mem::size_of::<u64>()
            + std::mem::size_of::<i64>()
            + (2 * std::mem::size_of::<i32>())) as u64
            + std::mem::size_of::<u64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    for sql in [
            "SELECT bucket, COUNT(*) FROM events GROUP BY bucket ORDER BY count DESC LIMIT 2",
            "SELECT bucket, COUNT(*) FROM events GROUP BY bucket HAVING count >= 2 ORDER BY bucket",
            "SELECT bucket, AVG(amount) FROM events GROUP BY bucket ORDER BY avg DESC LIMIT 2",
            "SELECT bucket, MIN(amount) FROM events GROUP BY bucket ORDER BY min DESC LIMIT 2",
            "SELECT bucket, MAX(amount) FROM events GROUP BY bucket ORDER BY max DESC LIMIT 2",
            "SELECT bucket, MAX(amount) FROM events GROUP BY bucket HAVING bucket = 1 OR max >= 40 ORDER BY bucket",
        ] {
            let Command::Select(select) = parse_command(sql).unwrap() else {
                unreachable!()
            };
            let cpu = e.execute_relational_select(&select).unwrap();
            let resident = e
                .execute_relational_grouped_aggregate_with_resident_device_memory_probe(&select)
                .unwrap();
            assert_eq!(resident.columns, cpu.columns, "{sql}");
            assert_eq!(resident.rows, cpu.rows, "{sql}");
            assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
            assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
            assert_eq!(resident.fallback_reason, None);
        }

    let Command::Select(unsupported_having) =
        parse_command("SELECT bucket, SUM(amount) FROM events GROUP BY bucket HAVING amount > 10")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_grouped_aggregate_with_resident_device_memory_probe(&unsupported_having)
        .unwrap_err()
        .to_string();
    assert!(err.contains("HAVING must reference grouped column or aggregate result"));

    e.execute_text(
        3,
        "INSERT INTO events (bucket, label, amount) VALUES (4, 'zeta', 50)",
    )
    .unwrap();
    assert!(e
        .execute_relational_grouped_sum_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_filtered_grouped_aggregate_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
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
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    for sql in [
            "SELECT bucket, COUNT(*) FROM events WHERE amount >= 15 GROUP BY bucket ORDER BY count DESC LIMIT 2",
            "SELECT bucket, COUNT(*) FROM events WHERE amount >= 15 GROUP BY bucket HAVING count >= 2 ORDER BY bucket",
            "SELECT bucket, SUM(amount) FROM events WHERE amount >= 15 GROUP BY bucket ORDER BY sum DESC LIMIT 2",
            "SELECT bucket, SUM(amount) FROM events WHERE amount >= 15 GROUP BY bucket HAVING sum > 30 ORDER BY sum DESC",
            "SELECT bucket, AVG(amount) FROM events WHERE amount >= 15 GROUP BY bucket ORDER BY avg DESC LIMIT 2",
            "SELECT bucket, MIN(amount) FROM events WHERE amount >= 15 GROUP BY bucket ORDER BY min DESC LIMIT 2",
            "SELECT bucket, MAX(amount) FROM events WHERE amount >= 15 GROUP BY bucket ORDER BY max DESC LIMIT 2",
            "SELECT bucket, MAX(amount) FROM events WHERE amount >= 15 GROUP BY bucket HAVING bucket = 2 OR max >= 40 ORDER BY bucket",
        ] {
            let Command::Select(select) = parse_command(sql).unwrap() else {
                unreachable!()
            };
            let cpu = e.execute_relational_select(&select).unwrap();
            let before = e.metrics().snapshot();
            let resident = e
                .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(
                    &select,
                )
                .unwrap();
            let after = e.metrics().snapshot();

            assert_eq!(resident.columns, cpu.columns, "{sql}");
            assert_eq!(resident.rows, cpu.rows, "{sql}");
            assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
            assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
            assert_eq!(resident.fallback_reason, None);
            assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0, "{sql}");
            assert!(
                after.d2h_bytes_total > before.d2h_bytes_total,
                "{sql} should read filtered grouped stats from device memory"
            );
            assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);
        }

    let Command::Select(unsupported_equality) =
        parse_command("SELECT bucket, SUM(amount) FROM events WHERE amount = 20 GROUP BY bucket")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(
            &unsupported_equality,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only non-equality int4 comparisons"));

    let Command::Select(unsupported_text) = parse_command(
        "SELECT bucket, MAX(amount) FROM events WHERE label >= 'beta' GROUP BY bucket",
    )
    .unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(
            &unsupported_text,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 comparison literals"));

    let Command::Select(unsupported_having) = parse_command(
            "SELECT bucket, SUM(amount) FROM events WHERE amount >= 15 GROUP BY bucket HAVING amount > 10",
        )
        .unwrap() else {
            unreachable!()
        };
    let err = e
        .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(
            &unsupported_having,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("HAVING must reference grouped column or aggregate result"));

    let Command::Select(select) =
        parse_command("SELECT bucket, SUM(amount) FROM events WHERE amount >= 15 GROUP BY bucket")
            .unwrap()
    else {
        unreachable!()
    };
    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
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
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_membership_count_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
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

    for sql in [
        "SELECT COUNT(*) FROM events WHERE id BETWEEN 2 AND 4",
        "SELECT COUNT(*) FROM events WHERE id BETWEEN 4 AND 2",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_between_count_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
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

    for sql in [
        "SELECT COUNT(*) FROM events WHERE id >= 2 AND amount <= 40",
        "SELECT COUNT(*) FROM events WHERE (id = 1 AND amount >= 10) OR (id = 4 AND amount <= 40)",
        "SELECT COUNT(*) FROM events WHERE id <= 2 OR amount >= 50",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_filter_group_count_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
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
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
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

    for sql in [
        "SELECT AVG(amount) FROM events",
        "SELECT MIN(amount) FROM events",
        "SELECT MAX(amount) FROM events",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_scalar_aggregate_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
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
        .execute_relational_scalar_aggregate_with_resident_device_memory_probe(&unsupported)
        .unwrap_err()
        .to_string()
        .contains("AVG only supports int4 columns"));

    e.mark_gpu_memory_pressured(0);
    let Command::Select(select) = parse_command("SELECT MAX(amount) FROM events").unwrap() else {
        unreachable!()
    };
    assert!(e
        .execute_relational_scalar_aggregate_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
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

    for sql in [
        "SELECT SUM(amount) FROM events WHERE amount >= 20",
        "SELECT AVG(amount) FROM events WHERE amount >= 20",
        "SELECT MIN(amount) FROM events WHERE amount >= 20",
        "SELECT MAX(amount) FROM events WHERE amount >= 20",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
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
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(
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
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(
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
    let cpu = e.execute_relational_select(&empty_max).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(&empty_max)
        .unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
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
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(&select)
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

    for sql in [
        "SELECT SUM(amount) FROM events WHERE amount BETWEEN 10 AND 30",
        "SELECT AVG(amount) FROM events WHERE amount BETWEEN 10 AND 30",
        "SELECT MIN(amount) FROM events WHERE amount BETWEEN 10 AND 30",
        "SELECT MAX(amount) FROM events WHERE amount BETWEEN 10 AND 30",
        "SELECT SUM(amount) FROM events WHERE amount BETWEEN 40 AND 10",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
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
        .execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(
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
        .execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(
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
    let err = e
        .execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(
            &equality_only,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("one int4 BETWEEN predicate"));

    let Command::Select(select) =
        parse_command("SELECT SUM(amount) FROM events WHERE amount BETWEEN 10 AND 30").unwrap()
    else {
        unreachable!()
    };
    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(&select)
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
