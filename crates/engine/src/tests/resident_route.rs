use super::*;
mod sharded_lookup;
mod sharded_reductions;

#[test]
fn p8_resident_route_decisions_use_cache_state_and_default_fallbacks() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    forget_test_relational_residency(&e, "events");
    e.set_shard_residency_enabled(false);

    let Command::Select(count_select) = parse_command("SELECT COUNT(*) FROM events").unwrap()
    else {
        unreachable!()
    };
    let absent = e.plan_relational_resident_route(&count_select);
    assert!(!absent.accepted);
    assert_eq!(absent.reason, "relation has no resident snapshot");
    assert_eq!(absent.query_shape, "count_all");
    assert_eq!(absent.d2h_bytes_estimate, 0);
    assert_eq!(absent.last_execution_h2d_bytes, None);
    assert_eq!(absent.last_execution_d2h_bytes, None);
    assert_eq!(absent.last_execution_kernel_samples, None);
    assert_eq!(absent.last_execution_kernel_ms, None);
    assert_eq!(absent.last_execution_kernel_event_elapsed_us, None);
    assert_eq!(absent.last_execution_rows, None);

    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    let decision = e.plan_relational_resident_route(&count_select);
    assert_eq!(decision.table, "events");
    assert_eq!(decision.gpu_id, Some(0));
    assert_eq!(decision.query_shape, "count_all");
    assert_eq!(decision.cache_state, "Valid");
    assert!(decision.valid);
    assert_eq!(decision.estimated_rows, 2);
    assert_eq!(decision.resident_bytes, snapshot.resident_bytes);
    assert_eq!(decision.h2d_bytes_if_resident, 0);
    assert_eq!(decision.h2d_bytes_if_cold, snapshot.resident_bytes);
    assert_eq!(
        decision.d2h_bytes_estimate,
        std::mem::size_of::<u64>() as u64
    );
    assert_eq!(decision.d2h_rows_estimate, 1);
    if snapshot.device_memory_proof.is_some() {
        assert!(decision.accepted);
        assert!(decision.has_retained_device_memory);
        assert_eq!(decision.reason, "resident route accepted");
    } else {
        assert!(!decision.accepted);
        assert!(!decision.has_retained_device_memory);
        assert_eq!(
            decision.reason,
            "resident snapshot has no retained device memory"
        );
    }
    assert_eq!(
        e.status_snapshot()
            .relational_residency
            .latest_route_decision("events")
            .unwrap(),
        &decision
    );

    if decision.accepted {
        let normal = e.execute_relational_select(&count_select).unwrap();
        assert_eq!(normal.planned_target, DeviceTarget::Gpu(0));
        assert_eq!(normal.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(normal.fallback_reason, None);
    } else {
        let fallback_before = e.metrics().snapshot().fallback_total;
        let error = e.execute_relational_select(&count_select).unwrap_err();
        assert_gpu_relational_execution_required(&e, error, "events", fallback_before);
    }

    let Command::Select(unsupported) = parse_command("SELECT * FROM events").unwrap() else {
        unreachable!()
    };
    let unsupported = e.plan_relational_resident_route(&unsupported);
    assert!(!unsupported.accepted);
    assert_eq!(unsupported.query_shape, "unsupported_select");
    assert_eq!(unsupported.d2h_bytes_estimate, 0);
    assert_eq!(unsupported.last_execution_h2d_bytes, None);
    assert_eq!(unsupported.last_execution_d2h_bytes, None);
    assert_eq!(unsupported.last_execution_kernel_samples, None);
    assert_eq!(unsupported.last_execution_kernel_ms, None);
    assert_eq!(unsupported.last_execution_kernel_event_elapsed_us, None);
    assert_eq!(unsupported.last_execution_rows, None);
    assert_eq!(
        unsupported.reason,
        "resident routing has no retained-kernel proof for this SELECT shape"
    );

    invalidate_test_relational_residency(&e, "events");
    let invalidated = e.plan_relational_resident_route(&count_select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident snapshot is Invalidated");
}

#[test]
fn p8_default_resident_route_executes_accepted_shapes() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE events (id INT, bucket INT, amount INT, label TEXT)",
    )
    .unwrap();
    // Bucket 1 (10+20) and bucket 2 (30) both sum to 30 — a deliberate tie that the explicit
    // resident route and the default GPU route must match against the closed-form expected rows:
    // equal SUMs break by group ASC, so `... ORDER BY sum DESC LIMIT 1` is deterministically bucket
    // 1. The resident path resolves this in the per-group two-level GROUP BY kernel.
    e.execute_text(
            2,
            "INSERT INTO events (id, bucket, amount, label) VALUES (1, 1, 10, 'alpha'), (2, 1, 20, 'beta'), (3, 2, 30, 'alpine')",
        )
        .unwrap();
    install_test_single_buffer_residency(&mut e, "events");

    let Command::Select(count_select) = parse_command("SELECT COUNT(*) FROM events").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&count_select);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    assert_eq!(route.h2d_bytes_if_resident, 0);

    // Closed-form construction oracles (S9): the GPU resident route AND the default (also-GPU) path are
    // checked against explicit expected rows over events = (1,1,10,'alpha'),(2,1,20,'beta'),
    // (3,2,30,'alpine') -- no host-finalized cuda-driver-probe re-execution as the reference. The two
    // buckets tie at SUM(amount)=30, so `... ORDER BY sum DESC LIMIT 1` tie-breaks to bucket 1 by
    // group-key ASC (the S8 cross-path contract). AVG uses the engine's own `average_sql_value`
    // finalizer (closed-form sum/count), never a CPU average.
    let avg = crate::rel_exec_helpers::average_sql_value;
    let cases: [(&str, Vec<Vec<SqlValue>>); 26] = [
        ("SELECT COUNT(*) FROM events", vec![vec![SqlValue::Int8(3)]]),
        ("SELECT COUNT(*) FROM events WHERE id = 2", vec![vec![SqlValue::Int8(1)]]),
        ("SELECT COUNT(*) FROM events WHERE id > 1", vec![vec![SqlValue::Int8(2)]]),
        ("SELECT COUNT(*) FROM events WHERE label LIKE 'al%'", vec![vec![SqlValue::Int8(2)]]),
        ("SELECT COUNT(*) FROM events WHERE id = 1 OR id = 3", vec![vec![SqlValue::Int8(2)]]),
        ("SELECT SUM(id) FROM events", vec![vec![SqlValue::Int8(6)]]),
        ("SELECT AVG(id) FROM events", vec![vec![avg(6, 3)]]),
        ("SELECT MIN(id) FROM events", vec![vec![SqlValue::Int4(1)]]),
        ("SELECT MAX(id) FROM events", vec![vec![SqlValue::Int4(3)]]),
        ("SELECT SUM(id) FROM events WHERE id > 1", vec![vec![SqlValue::Int8(5)]]),
        ("SELECT AVG(id) FROM events WHERE id < 3", vec![vec![avg(3, 2)]]),
        ("SELECT MIN(id) FROM events WHERE id >= 2", vec![vec![SqlValue::Int4(2)]]),
        ("SELECT MAX(id) FROM events WHERE id <= 2", vec![vec![SqlValue::Int4(2)]]),
        ("SELECT SUM(id) FROM events WHERE id BETWEEN 1 AND 2", vec![vec![SqlValue::Int8(3)]]),
        ("SELECT AVG(id) FROM events WHERE id BETWEEN 2 AND 3", vec![vec![avg(5, 2)]]),
        ("SELECT MIN(id) FROM events WHERE id BETWEEN 1 AND 3", vec![vec![SqlValue::Int4(1)]]),
        ("SELECT MAX(id) FROM events WHERE id BETWEEN 1 AND 1", vec![vec![SqlValue::Int4(1)]]),
        ("SELECT id FROM events WHERE id > 1", vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(3)]]),
        (
            "SELECT id FROM events WHERE id >= 1 ORDER BY id DESC LIMIT 2 OFFSET 1",
            vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(1)]],
        ),
        (
            "SELECT DISTINCT bucket FROM events ORDER BY bucket DESC LIMIT 2 OFFSET 1",
            vec![vec![SqlValue::Int4(1)]],
        ),
        (
            "SELECT DISTINCT bucket FROM events WHERE bucket >= 2 ORDER BY bucket DESC LIMIT 2 OFFSET 1",
            Vec::new(),
        ),
        (
            "SELECT bucket, COUNT(*) FROM events GROUP BY bucket HAVING count >= 1 ORDER BY bucket",
            vec![
                vec![SqlValue::Int4(1), SqlValue::Int8(2)],
                vec![SqlValue::Int4(2), SqlValue::Int8(1)],
            ],
        ),
        (
            "SELECT bucket, SUM(amount) FROM events GROUP BY bucket HAVING sum > 20 ORDER BY sum DESC LIMIT 1",
            vec![vec![SqlValue::Int4(1), SqlValue::Int8(30)]],
        ),
        (
            "SELECT bucket, AVG(amount) FROM events GROUP BY bucket ORDER BY bucket",
            vec![
                vec![SqlValue::Int4(1), avg(30, 2)],
                vec![SqlValue::Int4(2), avg(30, 1)],
            ],
        ),
        (
            "SELECT bucket, MIN(amount) FROM events WHERE amount >= 20 GROUP BY bucket HAVING min >= 20 ORDER BY bucket",
            vec![
                vec![SqlValue::Int4(1), SqlValue::Int4(20)],
                vec![SqlValue::Int4(2), SqlValue::Int4(30)],
            ],
        ),
        (
            "SELECT bucket, MAX(amount) FROM events WHERE amount > 10 GROUP BY bucket HAVING bucket = 1 OR max >= 30 ORDER BY max DESC",
            vec![
                vec![SqlValue::Int4(2), SqlValue::Int4(30)],
                vec![SqlValue::Int4(1), SqlValue::Int4(20)],
            ],
        ),
    ];
    for (sql, expected_rows) in cases {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let resident = e
            .execute_relational_select_with_resident_route(&select)
            .unwrap_or_else(|err| panic!("{sql}: {err}"));
        let before_default_metrics = e.metrics().snapshot();
        let default = e
            .execute_relational_select(&select)
            .unwrap_or_else(|err| panic!("{sql}: {err}"));
        let after_default_metrics = e.metrics().snapshot();
        assert_eq!(resident.rows, expected_rows, "{sql}");
        assert_eq!(default.rows, expected_rows, "{sql}");
        // Columns: GPU-vs-GPU consistency between the resident route and the default (also-GPU)
        // path -- no host/CPU oracle (S9).
        assert_eq!(*resident.columns, *default.columns, "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
        assert_eq!(default.planned_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(default.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(default.fallback_reason, None, "{sql}");
        assert_eq!(
            e.status_snapshot()
                .relational_residency
                .latest_route_decision("events")
                .unwrap()
                .h2d_bytes_if_resident,
            0,
            "{sql}"
        );
        let route_decision = e
            .status_snapshot()
            .relational_residency
            .latest_route_decision("events")
            .unwrap()
            .clone();
        let expected_d2h_bytes = match route_decision.query_shape.as_str() {
            "count_all"
            | "int4_equality_count"
            | "int4_range_count"
            | "int4_filter_group_count" => std::mem::size_of::<u64>() as u64,
            "text_prefix_like_count" => {
                e.relational_residency_snapshot("events")
                    .unwrap()
                    .resident_bytes
            }
            "int4_scalar_aggregate"
                if matches!(select.projection, SelectProjection::Sum { .. }) =>
            {
                std::mem::size_of::<i64>() as u64
            }
            // Ungrouped scalar aggregate copies a grouped-stats struct (group key +
            // count + sum + min/max) + result length.
            "int4_scalar_aggregate" => {
                (std::mem::size_of::<i32>()
                    + std::mem::size_of::<u64>()
                    + std::mem::size_of::<i64>()
                    + (2 * std::mem::size_of::<i32>())
                    + std::mem::size_of::<u64>()) as u64
            }
            // Filtered scalar aggregate copies a scalar-stats struct (count + sum +
            // min/max, no group key) + result length — matches the actual D2H in
            // run_resident_scalar_aggregate (the unified plan->kernel driver's Int4Compare arm)
            // and resident_route_d2h_bytes_estimate.
            "int4_filtered_scalar_aggregate" => {
                (std::mem::size_of::<u64>()
                    + std::mem::size_of::<i64>()
                    + (2 * std::mem::size_of::<i32>())
                    + std::mem::size_of::<u64>()) as u64
            }
            "int4_between_scalar_aggregate" => {
                (std::mem::size_of::<u64>()
                    + std::mem::size_of::<i64>()
                    + (2 * std::mem::size_of::<i32>())
                    + std::mem::size_of::<u64>()) as u64
            }
            "int4_grouped_aggregate" | "int4_filtered_grouped_aggregate" => route_decision
                .d2h_rows_estimate
                .checked_mul(
                    std::mem::size_of::<i32>()
                        + std::mem::size_of::<u64>()
                        + std::mem::size_of::<i64>()
                        + (2 * std::mem::size_of::<i32>()),
                )
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(u64::MAX),
            "int4_projection" | "int4_ordered_projection" => route_decision
                .d2h_rows_estimate
                .checked_mul(std::mem::size_of::<i32>())
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(u64::MAX),
            "int4_distinct_projection" | "int4_filtered_distinct_projection" => e
                .relational_residency_snapshot("events")
                .unwrap()
                .row_count
                .checked_mul(std::mem::size_of::<i32>())
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(u64::MAX),
            other => panic!("unexpected resident route shape {other} for {sql}"),
        };
        assert_eq!(
            route_decision.d2h_bytes_estimate, expected_d2h_bytes,
            "{sql}"
        );
        assert_eq!(route_decision.last_execution_h2d_bytes, Some(0), "{sql}");
        assert_eq!(
            route_decision.last_execution_d2h_bytes,
            Some(
                after_default_metrics
                    .d2h_bytes_total
                    .saturating_sub(before_default_metrics.d2h_bytes_total)
            ),
            "{sql}"
        );
        assert_eq!(
            route_decision.last_execution_kernel_samples,
            Some(
                after_default_metrics
                    .kernel_exec_samples
                    .saturating_sub(before_default_metrics.kernel_exec_samples)
            ),
            "{sql}"
        );
        assert_eq!(
            route_decision.last_execution_kernel_ms,
            Some(
                after_default_metrics
                    .kernel_exec_total_ms
                    .saturating_sub(before_default_metrics.kernel_exec_total_ms)
            ),
            "{sql}"
        );
        // Kernel-event timing is recorded only by shapes that actually launch a GPU
        // kernel. Some resident routes are GPU-resident but run no kernel — e.g.
        // text-prefix count and distinct/projection paths finalize on the CPU after a
        // D2H copy — so they record no kernel-event time and add no timing sample.
        // Assert telemetry consistency by what the route observed rather than by
        // hardcoding per-shape: if a kernel timed this execution, the route decision
        // matches the engine's last kernel-event metric and exactly one timing sample
        // lands; otherwise neither moves. (On GPU-less hosts no shape runs a kernel,
        // so every iteration takes the else branch — keeping CI green.)
        let kernel_event_timing_delta = after_default_metrics
            .kernel_event_timing_samples
            .saturating_sub(before_default_metrics.kernel_event_timing_samples);
        if route_decision
            .last_execution_kernel_event_elapsed_us
            .is_some()
        {
            assert_eq!(
                route_decision.last_execution_kernel_event_elapsed_us,
                after_default_metrics.last_kernel_event_elapsed_us,
                "{sql}"
            );
            assert_eq!(kernel_event_timing_delta, 1, "{sql}");
        } else {
            assert_eq!(kernel_event_timing_delta, 0, "{sql}");
        }
        assert_eq!(
            route_decision.last_execution_rows,
            Some(default.rows.len()),
            "{sql}"
        );
        assert!(
            matches!(
                route_decision.query_shape.as_str(),
                "count_all"
                    | "int4_equality_count"
                    | "int4_range_count"
                    | "text_prefix_like_count"
                    | "int4_filter_group_count"
                    | "int4_scalar_aggregate"
                    | "int4_filtered_scalar_aggregate"
                    | "int4_between_scalar_aggregate"
                    | "int4_grouped_aggregate"
                    | "int4_filtered_grouped_aggregate"
                    | "int4_projection"
                    | "int4_ordered_projection"
                    | "int4_distinct_projection"
                    | "int4_filtered_distinct_projection"
            ),
            "{sql}"
        );
    }

    for sql in [
        "SELECT DISTINCT label FROM events ORDER BY label",
        "SELECT DISTINCT bucket FROM events WHERE id >= 2 ORDER BY bucket DESC LIMIT 2",
        "SELECT DISTINCT bucket FROM events WHERE bucket = 2",
        "SELECT DISTINCT bucket FROM events OFFSET 1",
        "SELECT id FROM events WHERE id >= 1 ORDER BY id DESC",
        "SELECT id FROM events WHERE id = 1 ORDER BY id DESC LIMIT 1",
        "SELECT id FROM events WHERE amount >= 10 ORDER BY id DESC LIMIT 1",
        "SELECT label FROM events WHERE label LIKE 'a%' ORDER BY label LIMIT 1",
    ] {
        let Command::Select(unsupported) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let unsupported = e.plan_relational_resident_route(&unsupported);
        assert!(!unsupported.accepted, "{sql}");
        assert_eq!(unsupported.query_shape, "unsupported_select", "{sql}");
        assert_eq!(
            unsupported.reason,
            "resident routing has no retained-kernel proof for this SELECT shape",
            "{sql}"
        );
    }

    let Command::Select(unsupported_group_filter) =
        parse_command("SELECT bucket, COUNT(*) FROM events WHERE amount = 20 GROUP BY bucket")
            .unwrap()
    else {
        unreachable!()
    };
    let unsupported = e.plan_relational_resident_route(&unsupported_group_filter);
    assert!(!unsupported.accepted);
    assert_eq!(unsupported.query_shape, "unsupported_select");
    assert_eq!(
        unsupported.reason,
        "resident routing has no retained-kernel proof for this SELECT shape"
    );
}

#[test]
fn p8_resident_route_batches_int4_equality_projection_literals() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE events (id INT, bucket INT, amount INT, label TEXT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, bucket, amount, label) VALUES (1, 1, 10, 'alpha'), (2, 1, 20, 'beta'), (3, 2, 30, 'alpine')",
        )
        .unwrap();
    install_test_single_buffer_residency(&mut e, "events");

    let selects = [
        "SELECT id, bucket, amount FROM events WHERE id = 1",
        "SELECT id, bucket, amount FROM events WHERE id = 3",
    ]
    .into_iter()
    .map(|sql| {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        select
    })
    .collect::<Vec<_>>();
    let route = e.plan_relational_resident_route(&selects[0]);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }

    let before = e.metrics().snapshot();
    let results = e
            .execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
                &selects,
            )
            .unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].rows,
        vec![vec![
            SqlValue::Int4(1),
            SqlValue::Int4(1),
            SqlValue::Int4(10)
        ]]
    );
    assert_eq!(
        results[1].rows,
        vec![vec![
            SqlValue::Int4(3),
            SqlValue::Int4(2),
            SqlValue::Int4(30)
        ]]
    );
    for result in &results {
        assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
    }
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_equality_multi_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_kernel_samples,
        Some(
            after
                .kernel_exec_samples
                .saturating_sub(before.kernel_exec_samples)
        )
    );
    assert_eq!(decision.last_execution_kernel_samples, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(2));

    let single_column_selects = [
        "SELECT id FROM events WHERE id = 1",
        "SELECT id FROM events WHERE id = 3",
    ]
    .into_iter()
    .map(|sql| {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        select
    })
    .collect::<Vec<_>>();
    let single_column_results = e
            .execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
                &single_column_selects,
            )
            .unwrap();
    assert_eq!(single_column_results[0].rows, vec![vec![SqlValue::Int4(1)]]);
    assert_eq!(single_column_results[1].rows, vec![vec![SqlValue::Int4(3)]]);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "int4_equality_projection");
    assert_eq!(decision.last_execution_kernel_samples, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(2));
    let single_column_read_jobs = single_column_selects
        .iter()
        .map(|select| e.prepare_relational_retained_read_job(select))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let single_column_submission = e
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(
            &single_column_read_jobs,
        )
        .unwrap();
    assert_eq!(
        single_column_submission.job_count,
        single_column_read_jobs.len()
    );
    assert!(single_column_submission.submit_wall_micros > 0);
    let single_column_read_job_results = e
        .complete_relational_retained_read_submission(single_column_submission)
        .unwrap();
    assert_eq!(single_column_read_job_results, single_column_results);

    let mixed_column_selects = [
        "SELECT id, label FROM events WHERE id = 1",
        "SELECT id, label FROM events WHERE id = 3",
    ]
    .into_iter()
    .map(|sql| {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        select
    })
    .collect::<Vec<_>>();
    let mixed_column_results = e
            .execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
                &mixed_column_selects,
            )
            .unwrap();
    assert_eq!(
        mixed_column_results[0].rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("alpha".into())]]
    );
    assert_eq!(
        mixed_column_results[1].rows,
        vec![vec![SqlValue::Int4(3), SqlValue::Text("alpine".into())]]
    );
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_equality_mixed_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_kernel_samples, Some(3));
    assert_eq!(decision.last_execution_matched_rows, Some(2));

    let read_jobs = mixed_column_selects
        .iter()
        .map(|select| e.prepare_relational_retained_read_job(select))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(read_jobs.len(), 2);
    assert_eq!(
        read_jobs[0].snapshot_generation,
        read_jobs[1].snapshot_generation
    );
    assert!(
        read_jobs[0]
            .route_id
            .starts_with("int4_equality_mixed_column_projection:public:events:id,label:id"),
        "{}",
        read_jobs[0].route_id
    );
    assert_eq!(
        read_jobs[0].params,
        vec![RelationalRetainedReadParam::Int4Eq {
            column: "id".to_string(),
            value: 1
        }]
    );
    let submission = e
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&read_jobs)
        .unwrap();
    assert_eq!(submission.route_id, read_jobs[0].route_id);
    assert_eq!(
        submission.snapshot_generation,
        read_jobs[0].snapshot_generation
    );
    assert_eq!(submission.job_count, read_jobs.len());
    assert!(submission.submit_wall_micros > 0);
    let read_job_results = e
        .complete_relational_retained_read_submission(submission)
        .unwrap();
    assert_eq!(read_job_results, mixed_column_results);

    invalidate_test_relational_residency(&e, "events");
    e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    let stale_job = e
        .execute_relational_retained_read_jobs_with_resident_device_memory_probe(&read_jobs)
        .unwrap_err();
    assert!(
        stale_job
            .to_string()
            .contains("snapshot generation mismatch"),
        "{stale_job}"
    );
}

#[test]
fn p8_resident_route_executes_same_column_equality_projection() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE events (id INT, amount INT, label TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, amount, label) VALUES (1, 10, 'alpha'), (2, 20, 'beta'), (2, 30, 'delta'), (3, 40, 'gamma')",
        )
        .unwrap();
    install_test_single_buffer_residency(&mut e, "events");

    let Command::Select(select) = parse_command("SELECT id FROM events WHERE id = 2").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert_eq!(route.query_shape, "int4_equality_projection");
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }

    let before = e.metrics().snapshot();
    let result = e
        .execute_relational_select(&select)
        .expect("same-column equality projection should use resident route");
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(2)]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "int4_equality_projection");
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_rows, Some(2));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );

    let Command::Select(multi_column) =
        parse_command("SELECT id, amount FROM events WHERE id = 2").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&multi_column);
    assert_eq!(route.query_shape, "int4_equality_multi_column_projection");
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    let before = e.metrics().snapshot();
    let result = e
        .execute_relational_select(&multi_column)
        .expect("multi-column equality projection should use resident route");
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(2), SqlValue::Int4(20)],
            vec![SqlValue::Int4(2), SqlValue::Int4(30)]
        ]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_equality_multi_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_rows, Some(2));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    // S10a: the probe-specific EXACT d2h byte count is dropped -- these projection shapes now execute via
    // the `&Select`->general bridge, whose path does not feed `observe_d2h_bytes` (like every other
    // bridge-routed shape: grouped/ordered/distinct also record 0 execution-d2h here, see the default-route
    // test). The self-consistent delta check above + the rows assertion remain; the d2h ESTIMATE (route
    // planning) is still asserted in `p8_default_resident_route_executes_accepted_shapes`.

    let Command::Select(composite) =
        parse_command("SELECT id, amount FROM events WHERE id = 2 AND amount = 30").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&composite);
    assert_eq!(
        route.query_shape,
        "int4_composite_equality_multi_column_projection"
    );
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    let before = e.metrics().snapshot();
    let result = e
        .execute_relational_select(&composite)
        .expect("composite equality projection should use resident route");
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2), SqlValue::Int4(30)]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_composite_equality_multi_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    // S10a: probe-specific d2h byte count dropped (bridge records 0 execution-d2h; see the multi-column note above).

    let Command::Select(mixed_composite) =
        parse_command("SELECT id, amount, label FROM events WHERE id = 3 AND amount = 40").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&mixed_composite);
    assert_eq!(route.query_shape, "int4_equality_mixed_column_projection");
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    let before = e.metrics().snapshot();
    let result = e
        .execute_relational_select(&mixed_composite)
        .expect("mixed int4/text equality projection should use resident route");
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![
            SqlValue::Int4(3),
            SqlValue::Int4(40),
            SqlValue::Text("gamma".to_string())
        ]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_equality_mixed_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    // S10a: probe-specific d2h byte count dropped (bridge records 0 execution-d2h; see the multi-column note above).

    // Single-predicate mixed int4+text projection. The dispatcher delegates this shape to the
    // fused batch path (`..._batch_inner`); the route-execution telemetry observation must be
    // recorded EXACTLY ONCE for the delegation (the dispatcher owns it; the delegated batch
    // path suppresses its own), not double-counted. Assert via the observation counter.
    let Command::Select(mixed_single) =
        parse_command("SELECT id, amount, label FROM events WHERE id = 3").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&mixed_single);
    assert_eq!(route.query_shape, "int4_equality_mixed_column_projection");
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    let observations_before = e.route_execution_observation_count();
    let before = e.metrics().snapshot();
    let result = e
        .execute_relational_select(&mixed_single)
        .expect("single-predicate mixed int4/text projection should use resident route");
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![
            SqlValue::Int4(3),
            SqlValue::Int4(40),
            SqlValue::Text("gamma".to_string())
        ]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    // The delegation records the route-execution observation exactly once (no double-count).
    assert_eq!(
        e.route_execution_observation_count()
            .saturating_sub(observations_before),
        1,
        "single-predicate mixed int4+text route must record its execution observation exactly once"
    );
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_equality_mixed_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_rows, Some(1));
    // The stored d2h-bytes observation reflects the (single) delegated batch execution.
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
}

#[test]
fn p8_opt_in_resident_route_rejects_before_execution() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    forget_test_relational_residency(&e, "events");
    e.set_shard_residency_enabled(false);

    let Command::Select(absent) = parse_command("SELECT COUNT(*) FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_select_with_resident_route(&absent)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("relation has no resident snapshot"));

    e.populate_relational_residency_snapshot("events").unwrap();
    let Command::Select(unsupported) = parse_command("SELECT * FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_select_with_resident_route(&unsupported)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("resident routing has no retained-kernel proof"));

    invalidate_test_relational_residency(&e, "events");
    let err = e
        .execute_relational_select_with_resident_route(&absent)
        .unwrap_err();
    assert!(err.to_string().contains("resident snapshot is Invalidated"));
    assert_eq!(
        e.telemetry_snapshot()
            .relational_residency
            .latest_route_decision("events")
            .unwrap()
            .cache_state,
        "Invalidated"
    );
    let fallback_before = e.metrics().snapshot().fallback_total;
    let error = e.execute_relational_select(&absent).unwrap_err();
    assert!(error
        .to_string()
        .contains("GPU execution is required for SELECT on relation \"events\""));
    assert_eq!(e.metrics().snapshot().fallback_total, fallback_before);
}

#[test]
fn p8_resident_route_decisions_reject_evicted_and_memory_pressured_snapshots() {
    let mut e = Engine::new_local_test_engine();
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
    forget_test_relational_residency(&e, "events");
    forget_test_relational_residency(&e, "aux");
    e.set_shard_residency_enabled(false);

    let aux = e.populate_relational_residency_snapshot("aux").unwrap();
    let aux_allocated = e.relational_resident_bytes_for_gpu(0);
    e.populate_relational_residency_snapshot("events").unwrap();
    let events_allocated = e
        .relational_resident_bytes_for_gpu(0)
        .saturating_sub(aux_allocated);
    e.set_relational_residency_budget_bytes(0, events_allocated);
    let admitted = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(admitted.evicted_tables_on_admission, vec!["aux"]);

    let Command::Select(aux_count) = parse_command("SELECT COUNT(*) FROM aux").unwrap() else {
        unreachable!()
    };
    let evicted = e.plan_relational_resident_route(&aux_count);
    assert!(!evicted.accepted);
    assert_eq!(evicted.reason, "relation has no resident snapshot");
    assert_eq!(evicted.h2d_bytes_if_cold, 0);
    assert!(aux.resident_bytes > 0);

    let Command::Select(events_count) = parse_command("SELECT COUNT(*) FROM events").unwrap()
    else {
        unreachable!()
    };
    e.mark_gpu_memory_pressured(0);
    let pressured = e.plan_relational_resident_route(&events_count);
    assert!(!pressured.accepted);
    assert_eq!(pressured.cache_state, "InvalidatedByMemoryPressure");
    assert_eq!(
        pressured.reason,
        "resident snapshot is InvalidatedByMemoryPressure"
    );
    assert!(!pressured.valid);
    assert_eq!(
        e.telemetry_snapshot()
            .relational_residency
            .latest_route_decision("events")
            .unwrap()
            .reason,
        pressured.reason
    );
}

#[test]
fn p8_resident_warmup_policy_warms_refreshes_and_reports_route_readiness() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    forget_test_relational_residency(&e, "events");
    e.set_shard_residency_enabled(false);

    let report = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(report.gpu_id, 0);
    assert_eq!(report.entries.len(), 1);
    let entry = &report.entries[0];
    assert_eq!(entry.table, "events");
    assert_eq!(entry.action, RelationalResidencyWarmupAction::Warmed);
    assert!(entry.resident_bytes > 0);
    let first_handle = e.relational_retained_snapshot_handle("events").unwrap();
    assert_eq!(first_handle.generation, 1);
    assert_eq!(first_handle.row_count, 3);
    assert!(first_handle.valid);
    assert_eq!(
        first_handle.resident_device_int4_columns,
        vec!["id".to_string()]
    );
    let route = entry.route_decision.as_ref().unwrap();
    assert_eq!(route.query_shape, "count_all");
    assert_eq!(route.snapshot_generation, Some(first_handle.generation));
    if route.has_retained_device_memory {
        assert!(route.accepted);
        let Command::Select(select) = parse_command("SELECT COUNT(*) FROM events").unwrap() else {
            unreachable!()
        };
        let result = e.execute_relational_select(&select).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
    } else {
        assert!(!route.accepted);
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
    }

    invalidate_test_relational_residency(&e, "events");
    assert_eq!(
        e.relational_residency_snapshot("events")
            .unwrap()
            .invalidated_by_txn_id,
        Some(3)
    );
    let invalidated_handle = e.relational_retained_snapshot_handle("events").unwrap();
    assert!(first_handle.valid);
    assert_eq!(
        first_handle.has_retained_device_memory,
        route.has_retained_device_memory
    );
    assert_eq!(invalidated_handle.generation, first_handle.generation);
    assert!(!invalidated_handle.valid);
    assert!(!invalidated_handle.has_retained_device_memory);
    let refreshed = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(
        refreshed.entries[0].action,
        RelationalResidencyWarmupAction::Refreshed
    );
    assert_eq!(
        e.relational_residency_snapshot("events")
            .unwrap()
            .invalidated_by_txn_id,
        None
    );
    let refreshed_handle = e.relational_retained_snapshot_handle("events").unwrap();
    assert_eq!(
        refreshed_handle.generation,
        invalidated_handle.generation + 1
    );
    assert_eq!(refreshed_handle.row_count, 3);
    assert!(refreshed_handle.valid);
    assert_eq!(
        e.status_snapshot()
            .relational_residency
            .latest_route_decision("events")
            .unwrap()
            .estimated_rows,
        3
    );
    assert_eq!(
        e.status_snapshot()
            .relational_residency
            .table("events")
            .unwrap()
            .snapshot_generation,
        refreshed_handle.generation
    );

    let gpu_override = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        gpu_id: Some(7),
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(gpu_override.gpu_id, 7);
    assert_eq!(e.relational_residency_snapshot("events").unwrap().gpu_id, 7);
}

#[test]
fn p8_resident_warmup_policy_applies_budget_and_skips_unsafe_inputs() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE aux (id INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "CREATE TABLE oversized (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    e.execute_text(5, "INSERT INTO aux (id, label) VALUES (1, 'aux')")
        .unwrap();
    e.execute_text(
        6,
        "INSERT INTO oversized (id, label) VALUES (1, 'this-row-is-too-large-for-the-test-budget')",
    )
    .unwrap();
    forget_test_relational_residency(&e, "events");
    forget_test_relational_residency(&e, "aux");
    forget_test_relational_residency(&e, "oversized");
    e.set_shard_residency_enabled(false);

    e.populate_relational_residency_snapshot("events").unwrap();
    let events_budget = e.relational_resident_bytes_for_gpu(0);
    let aux_size = e.populate_relational_residency_snapshot("aux").unwrap();
    e.clear_relational_residency_budget_bytes(0);
    let report = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec![
            "events".to_string(),
            "aux".to_string(),
            "missing".to_string(),
        ],
        budget_bytes: Some(events_budget),
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(report.budget_bytes, Some(events_budget));
    assert_eq!(report.entries.len(), 3);
    assert!(report.entries.iter().any(|entry| entry.table == "missing"
        && entry.action == RelationalResidencyWarmupAction::Skipped));
    let events = report
        .entries
        .iter()
        .find(|entry| entry.table == "events")
        .unwrap();
    assert!(matches!(
        events.action,
        RelationalResidencyWarmupAction::AlreadyResident | RelationalResidencyWarmupAction::Warmed
    ));
    let aux = report
        .entries
        .iter()
        .find(|entry| entry.table == "aux")
        .unwrap();
    assert!(matches!(
        aux.action,
        RelationalResidencyWarmupAction::Warmed
            | RelationalResidencyWarmupAction::Refreshed
            | RelationalResidencyWarmupAction::AlreadyResident
    ));
    assert!(aux_size.resident_bytes > 0);
    assert!(e.status_snapshot().relational_residency.snapshot_count() <= 1);

    let oversized = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["oversized".to_string()],
        budget_bytes: Some(1),
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(
        oversized.entries[0].action,
        RelationalResidencyWarmupAction::Error
    );
    assert!(oversized.entries[0].reason.contains("exceeding GPU 0"));
    assert!(e.relational_residency_snapshot("oversized").is_none());

    e.mark_gpu_memory_pressured(0);
    let pressured = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(
        pressured.entries[0].action,
        RelationalResidencyWarmupAction::Skipped
    );
    assert_eq!(pressured.entries[0].reason, "GPU 0 is memory pressured");
}

#[test]
fn device_authoritative_warmup_rejects_an_impossible_budget_without_applying_it() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE authoritative_budget (id INT, label TEXT)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO authoritative_budget VALUES (1, 'resident')")
        .unwrap();
    assert!(engine.table_device_authoritative("authoritative_budget"));
    assert_eq!(engine.relational_residency_budget_bytes(0), None);

    let report = engine.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["authoritative_budget".to_string()],
        budget_bytes: Some(1),
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });

    assert_eq!(report.budget_bytes, None);
    assert_eq!(
        report.entries[0].action,
        RelationalResidencyWarmupAction::Error
    );
    assert!(report.entries[0]
        .reason
        .contains("cannot replace device-authoritative relation"));
    assert_eq!(engine.relational_residency_budget_bytes(0), None);
    assert!(engine.table_device_authoritative("authoritative_budget"));
}

#[test]
fn p8_resident_maintenance_tick_summarizes_refresh_and_route_readiness() {
    let mut e = Engine::new_local_test_engine();
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
    e.execute_text(5, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    forget_test_relational_residency(&e, "events");
    forget_test_relational_residency(&e, "aux");
    e.set_shard_residency_enabled(false);

    e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    invalidate_test_relational_residency(&e, "events");
    assert_eq!(
        e.relational_residency_snapshot("events")
            .unwrap()
            .invalidated_by_txn_id,
        Some(5)
    );

    let report = e
        .maintain_relational_residency_with_policy(RelationalResidencyMaintenancePolicy::default());
    assert_eq!(report.gpu_id, 0);
    assert_eq!(report.entry_count, 2);
    assert_eq!(report.refreshed_count, 1);
    assert_eq!(report.warmed_count, 1);
    assert_eq!(report.skipped_count, 0);
    assert_eq!(report.error_count, 0);
    assert_eq!(
        report.route_ready_count + report.route_blocked_count,
        report.entry_count
    );
    assert_eq!(
        e.relational_residency_snapshot("events")
            .unwrap()
            .invalidated_by_txn_id,
        None
    );
    assert!(e.relational_residency_snapshot("aux").is_some());

    let Command::Select(select) = parse_command("SELECT COUNT(*) FROM events").unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    if route.accepted {
        let result = e.execute_relational_select(&select).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
        assert_eq!(result.rows, vec![vec![SqlValue::Int8(3)]]);
        assert!(report
            .route_ready_tables
            .iter()
            .any(|table| table == "events"));
    } else {
        assert!(report
            .route_blockers
            .iter()
            .any(|blocker| blocker.table == "events" && blocker.reason == route.reason));
    }
}

#[test]
fn p8_resident_maintenance_tick_reports_pressure_and_budget_blockers() {
    let mut pressured = Engine::new_local_test_engine();
    pressured
        .execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    pressured
        .execute_text(2, "INSERT INTO events (id, label) VALUES (1, 'alpha')")
        .unwrap();
    forget_test_relational_residency(&pressured, "events");
    pressured.set_shard_residency_enabled(false);
    pressured.mark_gpu_memory_pressured(0);
    let pressure_report = pressured
        .maintain_relational_residency_with_policy(RelationalResidencyMaintenancePolicy::default());
    assert_eq!(pressure_report.entry_count, 1);
    assert_eq!(pressure_report.skipped_count, 1);
    assert_eq!(pressure_report.route_ready_count, 0);
    assert_eq!(pressure_report.route_blocked_count, 1);
    assert_eq!(pressure_report.route_blockers[0].table, "events");
    assert_eq!(
        pressure_report.route_blockers[0].reason,
        "GPU 0 is memory pressured"
    );
    assert!(pressured.relational_residency_snapshot("events").is_none());

    let mut oversized = Engine::new_local_test_engine();
    oversized
        .execute_text(1, "CREATE TABLE oversized (id INT, label TEXT)")
        .unwrap();
    oversized
        .execute_text(
            2,
            "INSERT INTO oversized (id, label) VALUES (1, 'this-row-is-too-large')",
        )
        .unwrap();
    forget_test_relational_residency(&oversized, "oversized");
    oversized.set_shard_residency_enabled(false);
    let budget_report =
        oversized.maintain_relational_residency_with_policy(RelationalResidencyMaintenancePolicy {
            budget_bytes: Some(1),
            ..RelationalResidencyMaintenancePolicy::default()
        });
    assert_eq!(budget_report.entry_count, 1);
    assert_eq!(budget_report.error_count, 1);
    assert_eq!(budget_report.route_ready_count, 0);
    assert_eq!(budget_report.route_blocked_count, 1);
    assert!(budget_report.route_blockers[0]
        .reason
        .contains("exceeding GPU 0"));
    assert!(oversized
        .relational_residency_snapshot("oversized")
        .is_none());
}

#[test]
fn telemetry_snapshot_reflects_replication_lag_and_runtime_metrics() {
    let mut e = Engine::with_batching(8, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();

    let snapshot = e.telemetry_snapshot();

    assert_eq!(snapshot.role, Role::Leader);
    assert_eq!(snapshot.replication_lag.commit_index, 0);
    assert_eq!(snapshot.replication_lag.applied_index, 0);
    assert_eq!(snapshot.replication_lag.visible_index, 0);
    assert_eq!(snapshot.replication_lag.commit_apply_gap, 0);
    assert_eq!(snapshot.replication_lag.apply_visible_gap, 0);
    assert_eq!(snapshot.runtime_metrics.pending_batch_peak, 1);
    assert_eq!(snapshot.runtime_metrics.last_pending_batch_len, Some(1));
    assert_eq!(snapshot.runtime_metrics.commits_total, 0);
    assert_eq!(snapshot.snapshot_id, 0);
    assert_eq!(snapshot.wal_flushed_count, 0);
    assert_eq!(snapshot.wal_last_durable_txn_id, None);
    assert_eq!(snapshot.wal_buffered_count, 0);
    assert_eq!(snapshot.wal_unflushed_count, 0);
    assert_eq!(snapshot.pending_batch_len, 1);
    assert_eq!(snapshot.pending_batch_cap, 8);
    assert_eq!(snapshot.active_txn_count, 0);
    assert_eq!(snapshot.backlog_blocker_count, 1);
    assert!(snapshot.has_backlog_blockers());
    assert!(!snapshot.quiescent_for_failover);
    assert!(!snapshot.mutation_admission_saturated);
    assert!(snapshot.gpu_parity_fallbacks.is_empty());
}

#[test]
fn status_snapshot_answers_snapshot_and_replication_health_questions() {
    let mut e = Engine::new_local_test_engine();
    let token = e.commit_mutation(1, b"SET a=1".to_vec().into()).unwrap();
    let exported = e.export_snapshot_meta();

    let status = e.status_snapshot();

    assert_eq!(status.role, Role::Leader);
    assert_eq!(status.term, 1);
    assert_eq!(status.snapshot.snapshot_id, exported.snapshot_id);
    assert_eq!(status.snapshot.last_included_index, token.index);
    assert_eq!(status.snapshot.last_included_term, 1);
    assert_eq!(status.snapshot.visible_index, token.index);
    assert_eq!(status.served_snapshot_frontier(), token.index);
    assert_eq!(status.replication_lag.commit_index, token.index);
    assert_eq!(status.replication_lag.applied_index, token.index);
    assert_eq!(status.replication_distance(), 0);
    assert!(status.why_routed_to_fallback_labels().is_empty());
    assert_eq!(status.latest_fallback_reason(), None);
    assert_eq!(status.backlog_blocker_labels(), Vec::<&'static str>::new());
    status.validate().unwrap();
}

#[test]
fn status_snapshot_surfaces_active_fallback_reasons_and_rollups() {
    let mut e = Engine::new_local_test_engine();
    e.mark_gpu_unavailable(0);
    e.set_gpu_runtime_saturated(true);

    e.execute_text(1, "SET a=1").unwrap();

    let status = e.status_snapshot();

    assert_eq!(
        status.latest_fallback_reason(),
        Some(FallbackReason::GpuUnavailable)
    );
    assert_eq!(
        status.why_routed_to_fallback_labels(),
        vec!["gpu_unavailable", "gpu_queue_saturated"]
    );
    assert!(status.fallback.is_actively_degraded());
    assert!(status.fallback.has_gpu_parity_fallbacks());
    assert_eq!(status.fallback.gpu_parity_fallback_total(), 1);
    assert_eq!(
        status.fallback.active_reasons,
        vec![
            ActiveFallbackReason::GpuUnavailable { gpu_ids: vec![0] },
            ActiveFallbackReason::GpuQueueSaturated,
        ]
    );
    status.validate().unwrap();
}

// S10c slice 2a: a sharded int4 aggregate whose predicate matches ZERO rows across ALL
// shards. The recompacted unified buffer is run ONCE; the COUNT(*) precheck returns 0 and the
// SUM projection yields SQL NULL (PG: SUM over zero rows is NULL — the COUNT-precheck placeholder is
// now `SqlValue::Null`, not the legacy Int8(0) sentinel; COUNT(*) still returns 0). This avoids the
// general SUM's empty-set hard error while staying PG-correct.
#[test]
fn p8_sharded_resident_sum_all_empty_returns_null() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
    )
    .unwrap();

    let shard_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![1, 2, 3, 4],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![500, 501, 502, 503],
        ],
        [
            vec![5, 6, 7, 8],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![600, 601, 602, 603],
        ],
        [
            vec![9, 10, 11, 12],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![700, 701, 702, 703],
        ],
        [
            vec![13, 14, 15, 16],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![800, 801, 802, 803],
        ],
    ];
    let shards = shard_values
        .iter()
        .enumerate()
        .map(|(shard_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            BenchmarkRelationalResidencyOwnedShard {
                shard_id: shard_id as u32,
                row_start: shard_id * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec![
                    "ol_o_id".to_string(),
                    "ol_i_id".to_string(),
                    "ol_quantity".to_string(),
                    "ol_amount".to_string(),
                ],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected shard install error: {err}"
        );
        return;
    }

    // ol_o_id is never 99999 in any shard, so the filtered set is empty across all shards.
    let Command::Select(select) =
        parse_command("SELECT SUM(ol_amount) FROM order_line WHERE ol_o_id = 99999").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "sharded_int4_equality_sum");
    assert_eq!(route.shard_count, 4);

    let result = e.execute_relational_select(&select).unwrap();
    // SUM over zero matched rows is SQL NULL (PG), not the legacy Int8(0) placeholder.
    assert_eq!(result.rows, vec![vec![SqlValue::Null]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
}

// S10c slice 2a (audit F1): the recompaction indexes shard slices POSITIONALLY by int4 ordinal and
// the unified descriptor labels the buffer with shard 0's int4 list, so a shard whose
// `resident_device_int4_columns` disagrees with shard 0 (here the first two columns are swapped) must
// be REJECTED with a clean error rather than silently recompacting a column's bytes into the wrong slot
// (or reading past a too-short source). The sharded benchmark install runs no layout validation, so
// the bridge enforces uniformity itself; the route's referenced-column membership check does not catch it.
#[test]
fn p8_sharded_resident_rejects_nonuniform_int4_layout() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
    )
    .unwrap();

    let layouts = [
        vec![
            "ol_o_id".to_string(),
            "ol_i_id".to_string(),
            "ol_quantity".to_string(),
            "ol_amount".to_string(),
        ],
        vec![
            // shard 1: ol_o_id / ol_i_id SWAPPED vs shard 0 (same set, different order).
            "ol_i_id".to_string(),
            "ol_o_id".to_string(),
            "ol_quantity".to_string(),
            "ol_amount".to_string(),
        ],
    ];
    let shards = layouts
        .iter()
        .enumerate()
        .map(|(shard_id, int4_columns)| {
            let row_count = 4_usize;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for _ in 0..4 {
                for value in 0..row_count as i32 {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            BenchmarkRelationalResidencyOwnedShard {
                shard_id: shard_id as u32,
                row_start: shard_id * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: int4_columns.clone(),
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "order_line",
            gpu_id: 0,
            shards,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected shard install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT SUM(ol_amount) FROM order_line WHERE ol_o_id = 1").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");

    let err = e
        .execute_relational_select(&select)
        .expect_err("a non-uniform shard int4 layout must be rejected, not silently recompacted");
    assert!(
        err.to_string().contains("does not match shard 0 layout"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------------------------------
// S10c slice 2b: sharded DISTINCT / GROUP BY / ORDER-BY-projection.
//
// The sharded bridge recompacts the whole table into ONE unified int4 buffer, so DISTINCT / GROUP BY /
// ORDER BY are CORRECT over it (it sees every row). Each fixture below builds the SAME logical rows TWICE:
// once as N device shards (the SoA-bytes `build_shards` pattern) and once as ONE single-store
// snapshot (rows INSERTed via `execute_text`, then `populate_relational_residency_snapshot`). The
// single-store engine is the ORACLE: it runs the SAME bridge over a whole-table store. We run the SAME SQL
// through `execute_relational_select` on each engine and assert the columns AND rows are byte-identical, plus
// the sharded route's `accepted` / `query_shape` / `executed_target == Gpu(0)`.
//
// Logical rows for table `pt (k INT, v INT)` across THREE shards (the cross-shard cases are the
// whole point): k=42 appears in p0 (twice) AND p2 -> ONE grouped/distinct row; k=1 spans p0+p1; k=5 spans
// p1+p2. `ORDER BY k DESC LIMIT 4` over all rows -> [42,42,42,9] drawn from p0 AND p2 (spans shards).
//
//   p0: (42,10) (1,20) (42,30)
//   p1: (3,5)   (1,15) (5,25)
//   p2: (42,40) (5,50) (9,60)

#[cfg(test)]
fn s10c_2b_shard_values() -> [[Vec<i32>; 2]; 3] {
    [
        [vec![42, 1, 42], vec![10, 20, 30]],
        [vec![3, 1, 5], vec![5, 15, 25]],
        [vec![42, 5, 9], vec![40, 50, 60]],
    ]
}

#[cfg(test)]
fn s10c_2b_logical_rows() -> Vec<(i32, i32)> {
    // The same rows in published shard order ((row_start, shard_id) — here shard order), so the
    // single-store oracle holds the identical multiset.
    let mut rows = Vec::new();
    for [ks, vs] in s10c_2b_shard_values() {
        for (k, v) in ks.into_iter().zip(vs) {
            rows.push((k, v));
        }
    }
    rows
}

/// Install the three shards for `pt (k INT, v INT)` as SoA device bytes (8-byte row-count header, then
/// k contiguous, then v contiguous). Returns `None` (caller should `return`) if there is no local GPU/driver.
#[cfg(test)]
fn s10c_2b_sharded_engine() -> Option<Engine> {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE pt (k INT, v INT)").unwrap();
    let shards = s10c_2b_shard_values()
        .iter()
        .enumerate()
        .map(|(shard_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            BenchmarkRelationalResidencyOwnedShard {
                shard_id: shard_id as u32,
                row_start: shard_id * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec!["k".to_string(), "v".to_string()],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();
    match e.install_benchmark_relational_residency_owned_shards(
        BenchmarkRelationalResidencyOwnedShardInstall {
            table: "pt",
            gpu_id: 0,
            shards,
        },
    ) {
        Ok(()) => Some(e),
        Err(err) => {
            assert!(
                err.to_string().contains("CUDA"),
                "unexpected shard install error: {err}"
            );
            None
        }
    }
}

/// Install the same logical rows as ONE whole-table single-store snapshot (the ORACLE). Returns `None`
/// (caller should `return`) if there is no local GPU/driver (no device-memory proof).
#[cfg(test)]
fn s10c_2b_single_store_engine() -> Option<Engine> {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE pt (k INT, v INT)").unwrap();
    let values = s10c_2b_logical_rows()
        .into_iter()
        .map(|(k, v)| format!("({k}, {v})"))
        .collect::<Vec<_>>()
        .join(", ");
    e.execute_text(2, &format!("INSERT INTO pt (k, v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("pt").unwrap();
    snapshot.device_memory_proof.is_some().then_some(e)
}

/// Run `sql` on the sharded engine and on the single-store oracle, assert the sharded route is
/// accepted with `expected_shape` and executed on GPU(0), and assert the sharded result is byte-identical
/// to the oracle's.
#[cfg(test)]
fn s10c_2b_assert_sharded_matches_oracle(
    sharded: &Engine,
    oracle: &Engine,
    sql: &str,
    expected_shape: &str,
) {
    let Command::Select(select) = parse_command(sql).unwrap() else {
        unreachable!()
    };
    let route = sharded.plan_relational_resident_route(&select);
    assert!(route.accepted, "{sql}: route not accepted: {route:?}");
    assert_eq!(route.query_shape, expected_shape, "{sql}");
    assert_eq!(route.shard_count, 3, "{sql}");

    let part = sharded.execute_relational_select(&select).unwrap();
    assert_eq!(part.executed_target, DeviceTarget::Gpu(0), "{sql}");
    assert_eq!(part.fallback_reason, None, "{sql}");

    let want = oracle.execute_relational_select(&select).unwrap();
    assert_eq!(*part.columns, *want.columns, "{sql}: columns diverged");
    assert_eq!(part.rows, want.rows, "{sql}: rows diverged from oracle");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn s10c_2b_sharded_grouped_aggregate_matches_oracle() {
    let Some(sharded) = s10c_2b_sharded_engine() else {
        return;
    };
    let Some(oracle) = s10c_2b_single_store_engine() else {
        return;
    };
    // Cross-shard case (a): k=42 is present in p0 (twice) AND p2 -> ONE grouped row (COUNT 3, SUM 80).
    // GROUP BY COUNT and GROUP BY SUM, with ORDER BY to pin order against the oracle.
    s10c_2b_assert_sharded_matches_oracle(
        &sharded,
        &oracle,
        "SELECT k, COUNT(*) FROM pt GROUP BY k ORDER BY k",
        "sharded_int4_grouped_aggregate",
    );
    s10c_2b_assert_sharded_matches_oracle(
        &sharded,
        &oracle,
        "SELECT k, SUM(v) FROM pt GROUP BY k ORDER BY k",
        "sharded_int4_grouped_aggregate",
    );
    // Independently pin the k=42 collapse so the oracle equality can't pass on two matching-but-wrong sides.
    let Command::Select(select) =
        parse_command("SELECT k, COUNT(*) FROM pt GROUP BY k ORDER BY k").unwrap()
    else {
        unreachable!()
    };
    let rows = sharded.execute_relational_select(&select).unwrap().rows;
    assert_eq!(
        rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(3), SqlValue::Int8(1)],
            vec![SqlValue::Int4(5), SqlValue::Int8(2)],
            vec![SqlValue::Int4(9), SqlValue::Int8(1)],
            vec![SqlValue::Int4(42), SqlValue::Int8(3)],
        ],
        "k=42 (in p0 twice + p2) must collapse to ONE row with COUNT 3"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn s10c_2b_sharded_filtered_grouped_aggregate_matches_oracle() {
    let Some(sharded) = s10c_2b_sharded_engine() else {
        return;
    };
    let Some(oracle) = s10c_2b_single_store_engine() else {
        return;
    };
    // Filtered grouped: WHERE v >= 25 keeps (1,? no) -> (42,30),(5,25),(42,40),(5,50),(9,60). k=42 still
    // spans p0+p2 -> ONE row; k=5 still spans p1+p2 -> ONE row.
    s10c_2b_assert_sharded_matches_oracle(
        &sharded,
        &oracle,
        "SELECT k, COUNT(*) FROM pt WHERE v >= 25 GROUP BY k ORDER BY k",
        "sharded_int4_filtered_grouped_aggregate",
    );
    s10c_2b_assert_sharded_matches_oracle(
        &sharded,
        &oracle,
        "SELECT k, SUM(v) FROM pt WHERE v >= 25 GROUP BY k ORDER BY k",
        "sharded_int4_filtered_grouped_aggregate",
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn s10c_2b_sharded_distinct_projection_matches_oracle() {
    let Some(sharded) = s10c_2b_sharded_engine() else {
        return;
    };
    let Some(oracle) = s10c_2b_single_store_engine() else {
        return;
    };
    // Cross-shard case (c): DISTINCT k where k=42 (p0+p2), k=1 (p0+p1), k=5 (p1+p2) each yield ONE row.
    s10c_2b_assert_sharded_matches_oracle(
        &sharded,
        &oracle,
        "SELECT DISTINCT k FROM pt ORDER BY k",
        "sharded_int4_distinct_projection",
    );
    // Filtered DISTINCT: WHERE k >= 5 -> {5,9,42}; k=5 spans p1+p2, k=42 spans p0+p2 -> still one row each.
    s10c_2b_assert_sharded_matches_oracle(
        &sharded,
        &oracle,
        "SELECT DISTINCT k FROM pt WHERE k >= 5 ORDER BY k",
        "sharded_int4_filtered_distinct_projection",
    );
    // Independently pin the multi-shard DISTINCT key collapse.
    let Command::Select(select) = parse_command("SELECT DISTINCT k FROM pt ORDER BY k").unwrap()
    else {
        unreachable!()
    };
    let rows = sharded.execute_relational_select(&select).unwrap().rows;
    assert_eq!(
        rows,
        vec![
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(3)],
            vec![SqlValue::Int4(5)],
            vec![SqlValue::Int4(9)],
            vec![SqlValue::Int4(42)],
        ],
        "DISTINCT keys present in multiple shards must each yield ONE row"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn s10c_2b_sharded_ordered_projection_matches_oracle() {
    let Some(sharded) = s10c_2b_sharded_engine() else {
        return;
    };
    let Some(oracle) = s10c_2b_single_store_engine() else {
        return;
    };
    // Cross-shard case (b): a global ORDER BY k DESC LIMIT 4 whose top-4 rows [42,42,42,9] are drawn
    // from p0 (a 42) AND p2 (two 42s + the 9) — i.e. the top-N window spans MULTIPLE shards.
    s10c_2b_assert_sharded_matches_oracle(
        &sharded,
        &oracle,
        "SELECT k FROM pt WHERE k >= 1 ORDER BY k DESC LIMIT 4",
        "sharded_int4_ordered_projection",
    );
    let Command::Select(select) =
        parse_command("SELECT k FROM pt WHERE k >= 1 ORDER BY k DESC LIMIT 4").unwrap()
    else {
        unreachable!()
    };
    let rows = sharded.execute_relational_select(&select).unwrap().rows;
    assert_eq!(
        rows,
        vec![
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(9)],
        ],
        "the global top-4 (42,42,42,9) must span p0 and p2"
    );
}

/// STRATA S-B acceptance: with `auto_admit_on_commit` ON, a freshly CREATE+INSERTed table becomes
/// GPU-resident WITHOUT any explicit warm/populate call, so reads take the GPU-native resident route
/// and return correct results — including a NULL-bearing int4 column. R3-004 also requires a flag-OFF
/// engine to retain its device write generation; the flag now controls optional broader read-cache
/// admission, not whether committed relational data has device authority. Self-guards on a non-GPU box
/// (the resident route is not accepted when the device-memory upload cannot run).
#[test]
fn s_b_auto_admit_on_commit_makes_committed_table_gpu_resident() {
    let expected = vec![
        vec![SqlValue::Int4(1), SqlValue::Int4(10)],
        vec![SqlValue::Int4(2), SqlValue::Null],
        vec![SqlValue::Int4(3), SqlValue::Int4(30)],
    ];
    let parse = |s: &str| {
        let Command::Select(sel) = parse_command(s).unwrap() else {
            unreachable!()
        };
        sel
    };

    // Auto-admit ON, and NO explicit populate/warm call — residency must come purely from the commit.
    let e = Engine::new_local_test_engine();
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (id, v) VALUES (1, 10), (2, NULL), (3, 30)",
    )
    .unwrap();

    // Probe with a definitely-accepted shape; self-guard on a box without a usable GPU (the
    // device-memory upload returns None there, so the route is not accepted — the auto-admit code
    // path still ran).
    let probe = parse("SELECT id FROM t WHERE id = 2");
    let route = e.plan_relational_resident_route(&probe);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    let probe_res = e.execute_relational_select(&probe).unwrap();
    assert_eq!(
        probe_res.executed_target,
        DeviceTarget::Gpu(0),
        "auto-admitted table must read on the GPU resident route"
    );
    assert_eq!(probe_res.rows, vec![vec![SqlValue::Int4(2)]]);

    // NULL-bearing projection: rows must be correct (NULL at id=2) whichever route serves it.
    let proj = parse("SELECT id, v FROM t ORDER BY id");
    let gpu = e.execute_relational_select(&proj).unwrap();
    assert_eq!(
        gpu.rows, expected,
        "auto-admitted GPU read must be correct, incl. NULL"
    );

    // Flag OFF (default): DML still establishes the mandatory device generation.
    let mandatory = Engine::new_local_test_engine();
    mandatory
        .execute_text(1, "CREATE TABLE t (id INT, v INT)")
        .unwrap();
    mandatory
        .execute_text(
            2,
            "INSERT INTO t (id, v) VALUES (1, 10), (2, NULL), (3, 30)",
        )
        .unwrap();
    let mandatory_res = mandatory.execute_relational_select(&proj).unwrap();
    assert_eq!(mandatory_res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(mandatory_res.rows, expected);
}

/// STRATA S-B (production commit path): in production a plain INSERT is routed through the CONCURRENT
/// DML path (`execute_dml_concurrent`), which hits a DIFFERENT commit hook than `execute_text`. Confirm
/// auto-admit-on-commit fires there too — the table is GPU-resident after a concurrent-path INSERT,
/// with NULL data, no explicit warm. Self-guards on a non-GPU box.
#[test]
fn s_b_auto_admit_fires_on_the_concurrent_dml_commit_path() {
    let expected = vec![
        vec![SqlValue::Int4(1), SqlValue::Int4(10)],
        vec![SqlValue::Int4(2), SqlValue::Null],
        vec![SqlValue::Int4(3), SqlValue::Int4(30)],
    ];
    let e = Engine::new_local_test_engine();
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
    // Drive the INSERT through the concurrent path that production uses for a no-sequence-default
    // INSERT (NOT `execute_text`), so the `commit_dml_concurrent` hook is the one under test.
    e.execute_dml_concurrent(
        2,
        "INSERT INTO t (id, v) VALUES (1, 10), (2, NULL), (3, 30)",
    )
    .unwrap();

    // Probe residency with a definitely-accepted shape (proving the concurrent hook admitted the
    // table); self-guard on a non-GPU box.
    let Command::Select(probe) = parse_command("SELECT id FROM t WHERE id = 2").unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&probe);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    let probe_res = e.execute_relational_select(&probe).unwrap();
    assert_eq!(
        probe_res.executed_target,
        DeviceTarget::Gpu(0),
        "auto-admit must fire on the concurrent DML commit path (the production INSERT route)"
    );
    assert_eq!(probe_res.rows, vec![vec![SqlValue::Int4(2)]]);

    // NULL-bearing projection: rows must be correct (NULL at id=2) whichever route serves it.
    let Command::Select(proj) = parse_command("SELECT id, v FROM t ORDER BY id").unwrap() else {
        unreachable!()
    };
    assert_eq!(e.execute_relational_select(&proj).unwrap().rows, expected);
}

/// ADR-009 R1: the GPU index-probe point-lookup route (`index_probe_enabled` ON) must return
/// BYTE-IDENTICAL results to the full-scan route (OFF) — including NULL projections, absent needles,
/// repeated needles, the duplicate-key fallback (a non-unique column makes the index abandon so the
/// scan runs), and across a generation change (an INSERT re-admits, so the per-generation index cache
/// rebuilds). The differential IS the gate: same engine, same needles, only the flag flips. Skips
/// gracefully when there is no GPU residency route (CI without a GPU).
#[test]
fn r1_wave_index_probe_matches_scan_differential() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE accounts (id INT, bucket INT, balance INT, note TEXT)",
    )
    .unwrap();
    // `id` is a UNIQUE key (the index route fires); `bucket` is NON-unique (duplicate -> scan fallback).
    // NULL `balance` (id=20) + NULL `note` exercise NULL handling on projected + unprojected columns.
    // NULL `balance` (id=20) exercises a NULL in a PROJECTED column (handled identically by both routes).
    // The NULL-`id` row exercises NULL-as-0 KEY handling: a NULL int4 is materialized as 0, so BOTH the
    // scan (raw int4 compare, no validity-bitmap consult) and the index (built from the SAME device bytes)
    // match it at `id = 0`. The index does NOT skip it — that byte-identity at needle 0 is the audit's
    // P1 #1, closed by building the index from the resident device buffer rather than from host rows.
    e.execute_text(
        2,
        "INSERT INTO accounts (id, bucket, balance, note) VALUES \
         (10, 1, 100, 'a'), (20, 1, NULL, 'b'), (30, 2, 300, NULL), (40, 2, 400, 'd'), (NULL, 5, 500, 'e')",
    )
    .unwrap();
    install_test_single_buffer_residency(&mut e, "accounts");

    let select_cmd = |sql: &str| -> Select {
        match parse_command(sql).unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        }
    };
    let select_unique = select_cmd("SELECT id, balance FROM accounts WHERE id = 1");
    // One probe is enough — every query below shares this resident table.
    if !e.plan_relational_resident_route(&select_unique).accepted {
        return;
    }
    let select_dup = select_cmd("SELECT id, bucket FROM accounts WHERE bucket = 1");

    // Per-needle rows via the retained-template path (the route the index/scan swap lives on).
    let run = |e: &Engine, select: &Select, needles: &[i32]| -> Vec<RowBlock> {
        let template = e.prepare_relational_retained_read_template(select).unwrap();
        let submission = e
            .submit_relational_retained_template_point_lookups(&template, needles)
            .unwrap();
        e.complete_relational_retained_read_submission(submission)
            .unwrap()
            .iter()
            .map(|result| result.rows.clone())
            .collect()
    };
    let cached_index = |e: &Engine| -> (usize, bool) {
        let cache = e
            .read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = cache
            .get("accounts")
            .expect("wave index cached after a flag-on run");
        (entry.column_idx, entry.index_memory.is_some())
    };

    // (a) Unique-key differential WITH NULL: DISTINCT present needles (incl the NULL-balance row 20), one
    // absent needle (25), and needle 0 which hits the NULL-`id` row (NULL materialized as 0 — both routes
    // must agree). Needles are distinct by contract — the batcher's `dedup_needles` collapses identical
    // point lookups — so the thread-per-needle index and the thread-per-row scan are a bijection.
    let unique_needles = vec![10, 20, 30, 40, 25, 0];
    e.set_index_probe_enabled(false);
    let scan_unique = run(&e, &select_unique, &unique_needles);
    e.set_index_probe_enabled(true);
    let index_unique = run(&e, &select_unique, &unique_needles);
    assert_eq!(
        index_unique, scan_unique,
        "index-probe rows must equal scan rows for a unique key (incl NULL projection)"
    );
    // Non-vacuous: the flag-on run BUILT + USED a real index over the `id` filter column (col 0).
    assert_eq!(
        cached_index(&e),
        (0, true),
        "unique key -> a real index over col 0, not a scan-fallback marker"
    );
    assert_eq!(
        index_unique[0],
        vec![vec![SqlValue::Int4(10), SqlValue::Int4(100)]],
        "needle 10 -> its row (gather of a non-NULL balance)"
    );
    assert_eq!(
        index_unique[2],
        vec![vec![SqlValue::Int4(30), SqlValue::Int4(300)]],
        "needle 30 -> its row"
    );
    assert_eq!(
        index_unique[4],
        Vec::<Vec<SqlValue>>::new(),
        "needle 25 absent -> no rows"
    );
    // needle 0 hits the NULL-`id` row (NULL int4 == 0): the index returns it exactly as the scan does
    // (audit P1 #1). The differential `index_unique == scan_unique` already pins byte-identity; assert the
    // row is non-empty so the case is not vacuous (both routes genuinely match a NULL-as-0 key).
    assert_eq!(
        index_unique[5].len(),
        1,
        "needle 0 -> the NULL-id row, on both routes"
    );
    // id=20 (NULL balance) is covered by the differential above — this fast path returns the raw stored
    // int4 for a NULL projection (no bitmap mask), identically on both routes.

    // (b) Duplicate-key fallback: `bucket` is non-unique, so the index build abandons (None) and the
    // scan runs -- flag on must STILL match flag off (and return BOTH rows for bucket = 1).
    let dup_needles = vec![1, 2, 9];
    e.set_index_probe_enabled(false);
    let scan_dup = run(&e, &select_dup, &dup_needles);
    e.set_index_probe_enabled(true);
    let index_dup = run(&e, &select_dup, &dup_needles);
    assert_eq!(
        index_dup, scan_dup,
        "a non-unique key must fall back to the scan and stay identical"
    );
    assert_eq!(
        index_dup[0].len(),
        2,
        "bucket = 1 matches two rows (id 10 and 20)"
    );
    assert_eq!(
        cached_index(&e),
        (1, false),
        "duplicate key over col 1 -> a scan-fallback marker (no index)"
    );

    // (c) Generation change: an INSERT re-admits the table (new generation). The per-generation index
    // cache must rebuild against the new rows -- flag on still matches flag off over the larger table.
    forget_test_relational_residency(&e, "accounts");
    e.set_shard_residency_enabled(true);
    e.execute_text(
        3,
        "INSERT INTO accounts (id, bucket, balance, note) VALUES (50, 3, 500, 'e')",
    )
    .unwrap();
    install_test_single_buffer_residency(&mut e, "accounts");
    let gen_needles = vec![10, 50, 40, 999];
    e.set_index_probe_enabled(false);
    let scan_gen = run(&e, &select_unique, &gen_needles);
    e.set_index_probe_enabled(true);
    let index_gen = run(&e, &select_unique, &gen_needles);
    assert_eq!(
        index_gen, scan_gen,
        "after a generation change the rebuilt index must still match the scan"
    );
    assert_eq!(
        index_gen[1],
        vec![vec![SqlValue::Int4(50), SqlValue::Int4(500)]],
        "the newly-inserted row 50 is found by the rebuilt index"
    );
    assert_eq!(
        cached_index(&e),
        (0, true),
        "the index rebuilt over col 0 for the new generation"
    );
}

/// S-F/R-1: a lazy single-buffer device index is an optional optimization, not permission to
/// exceed the residency cap. With the budget pinned to the already-resident table, the index
/// allocation must decline, the retained GPU scan must still return the row, and accounting must
/// remain at or below the cap.
#[test]
fn wave_index_declines_at_residency_budget_without_losing_gpu_scan() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE capped_index (id INT, balance INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO capped_index VALUES (10, 100), (20, 200), (30, 300)",
    )
    .unwrap();
    let admitted = install_test_single_buffer_residency(&mut e, "capped_index");
    if admitted.device_memory_proof.is_none() {
        return;
    }
    let budget = e.relational_resident_bytes_for_gpu(0);
    e.set_relational_residency_budget_bytes(0, budget);
    e.set_index_probe_enabled(true);
    let select = match parse_command("SELECT id, balance FROM capped_index WHERE id = 1").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    let template = e
        .prepare_relational_retained_read_template(&select)
        .unwrap();
    let results = e
        .complete_relational_retained_read_submission(
            e.submit_relational_retained_template_point_lookups(&template, &[20])
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        results[0].rows,
        vec![vec![SqlValue::Int4(20), SqlValue::Int4(200)]]
    );
    let cache = e
        .read_state
        .residency
        .wave_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert!(
        cache
            .get("capped_index")
            .is_some_and(|entry| entry.index_memory.is_none()),
        "the optional index must cache a decline rather than retain over-budget memory"
    );
    drop(cache);
    assert!(e.relational_resident_bytes_for_gpu(0) <= budget);
}

// ADR-009 R1 (lifecycle + routing): the launch-per-batch (lpb) GPU hash-index probe returns BYTE-IDENTICAL
// rows to the full scan, across NULL data, NULL-as-0 keys, absent needles, non-unique fallback, and a
// generation rebuild. The `dense_index_probe_hits` counter is the non-vacuity signal that the index route
// (not a silent scan) actually served the unique-key batches -- output equality alone can't tell them apart,
// since they are byte-identical by design. GPU test (#[ignore]).
// ADR-009 Result-path: the BATCHED completion (one flat RelationalRetainedBatchResult + per-needle ranges)
// must produce BYTE-IDENTICAL per-needle rows to the per-needle `complete_relational_retained_read_submission`
// path (same ascending row_index order, same NULL-as-0 + absent handling). Two submits of the SAME needles
// (the index probe is deterministic) -> one completed per-needle, one batched -> compare slice-by-slice. GPU test.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn r2_batched_completion_matches_per_needle() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE accounts (id INT, bucket INT, balance INT, note TEXT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, bucket, balance, note) VALUES \
         (10, 1, 100, 'a'), (20, 1, 200, 'b'), (30, 2, 300, NULL), (40, 2, 400, 'd'), (50, 5, 500, 'e')",
    )
    .unwrap();
    e.populate_relational_residency_snapshot("accounts")
        .unwrap();
    let select = match parse_command("SELECT id, balance FROM accounts WHERE id = 1").unwrap() {
        Command::Select(s) => s,
        _ => unreachable!(),
    };
    if !e.plan_relational_resident_route(&select).accepted {
        return; // no GPU
    }
    // lpb INDEX route (unique key `id`): wave_engine on.
    e.set_index_probe_enabled(true);
    let template = e
        .prepare_relational_retained_read_template(&select)
        .unwrap();
    // Distinct needles including an absent one. NULL fidelity is covered by the dedicated nullable
    // sharded point-read tests; this raw-i32 batch contract intentionally exercises non-NULL rows.
    let needles = vec![10, 20, 30, 40, 25, 50];

    let per_needle = e
        .complete_relational_retained_read_submission(
            e.submit_relational_retained_template_point_lookups(&template, &needles)
                .unwrap(),
        )
        .unwrap();
    let batched = e
        .complete_relational_retained_read_submission_batched(
            e.submit_relational_retained_template_point_lookups(&template, &needles)
                .unwrap(),
        )
        .unwrap();

    assert_eq!(
        batched.needle_count(),
        per_needle.len(),
        "batched needle count == per-needle result count"
    );
    assert_eq!(
        batched.columns, per_needle[0].columns,
        "batched shares the per-needle projected schema"
    );
    for (i, result) in per_needle.iter().enumerate() {
        // batched is raw i32; map to SqlValue::Int4 to compare with the per-needle SqlValue rows.
        let batched_vals: Vec<SqlValue> = batched
            .needle_values(i)
            .iter()
            .map(|&v| SqlValue::Int4(v))
            .collect();
        let per_needle_vals: Vec<SqlValue> = result.rows.iter().flatten().cloned().collect();
        assert_eq!(
            batched_vals, per_needle_vals,
            "needle {i}: batched rows must be byte-identical to the per-needle rows"
        );
    }
    // Non-vacuity spot checks: needle 50 is present; 25 is absent.
    assert_eq!(
        batched.needle_values(5),
        &[50, 500],
        "needle 50 -> its row via the batched path"
    );
    assert!(
        batched.needle_values(4).is_empty(),
        "absent needle 25 -> no rows"
    );
}

// Multi-row-needle companion to r2_batched_completion_matches_per_needle (audit P2): a NON-unique predicate
// (bucket) so each needle matches SEVERAL rows with DIFFERING projected values — this exercises the
// load-bearing intra-needle (needle_index, row_index) sort that the single-row dataset above cannot
// (there, reversing intra-needle order is invisible). A reversed intra-needle sort diverges from the
// per-needle path HERE. GPU test.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn r2_batched_completion_matches_per_needle_multirow() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE accounts (id INT, bucket INT, balance INT, note TEXT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, bucket, balance, note) VALUES \
         (10, 1, 100, 'a'), (20, 1, 200, 'b'), (30, 2, 300, 'c'), (40, 2, 400, 'd'), (50, 3, 500, 'e')",
    )
    .unwrap();
    e.populate_relational_residency_snapshot("accounts")
        .unwrap();
    let select = match parse_command("SELECT id, balance FROM accounts WHERE bucket = 1").unwrap() {
        Command::Select(s) => s,
        _ => unreachable!(),
    };
    if !e.plan_relational_resident_route(&select).accepted {
        return; // no GPU
    }
    // Non-unique `bucket` -> the index declines and the SCAN serves; wave_engine on exercises that fallback.
    e.set_index_probe_enabled(true);
    let template = e
        .prepare_relational_retained_read_template(&select)
        .unwrap();
    // bucket 1 -> 2 rows, bucket 2 -> 2 rows, bucket 3 -> 1 row, bucket 9 -> absent.
    let needles = vec![1, 2, 3, 9];

    let per_needle = e
        .complete_relational_retained_read_submission(
            e.submit_relational_retained_template_point_lookups(&template, &needles)
                .unwrap(),
        )
        .unwrap();
    let batched = e
        .complete_relational_retained_read_submission_batched(
            e.submit_relational_retained_template_point_lookups(&template, &needles)
                .unwrap(),
        )
        .unwrap();

    assert_eq!(batched.needle_count(), per_needle.len());
    // Guard against a vacuous run: at least one needle must carry MULTIPLE rows, else the intra-needle sort
    // is still untested.
    assert!(
        (0..batched.needle_count()).any(|i| batched.needle_values(i).len() > batched.ncols()),
        "test is vacuous: no needle matched >1 row (predicate did not route as multi-row)"
    );
    for (i, result) in per_needle.iter().enumerate() {
        let batched_vals: Vec<SqlValue> = batched
            .needle_values(i)
            .iter()
            .map(|&v| SqlValue::Int4(v))
            .collect();
        let per_needle_vals: Vec<SqlValue> = result.rows.iter().flatten().cloned().collect();
        assert_eq!(
            batched_vals, per_needle_vals,
            "needle {i}: multi-row batched rows must be byte-identical to the per-needle rows"
        );
    }
    // Non-vacuity: bucket=1 carries BOTH rows in ascending row_index (id 10 then 20) — a reversed
    // intra-needle sort breaks exactly this.
    assert_eq!(
        batched.needle_values(0),
        &[10, 100, 20, 200],
        "bucket=1 -> (10,100),(20,200) in ascending row order"
    );
    assert!(
        batched.needle_values(3).is_empty(),
        "absent bucket 9 -> no rows"
    );
}

// audit P3 (batched scatter): a GPU fixture's 2-row emit order coincidentally equals ascending row_index, so
// a "no-sort" regression on multi-row needles slips past the GPU differentials. This PURE-CPU test proves the
// within-needle sort is NECESSARY by feeding assemble_batched_rows a multi-row needle in DESCENDING emit order:
// the output MUST be ascending row_index, which only holds if the sort runs (a no-sort regression yields emit
// order and fails here). Complements r2_materialized_payload_arm (same idea for the per-needle path).
#[test]
fn r2_batched_assembly_sorts_multirow_needle_by_row_index() {
    let col = |name: &str| RelationalColumn {
        id: 0,
        table_oid: 0,
        attnum: 1,
        name: name.to_string(),
        ty: SqlType::Int4,
        domain: None,
        default: None,
        type_oid: 23,
        type_size: 4,
    };
    let shared_cols = std::sync::Arc::new(vec![col("id"), col("v")]);
    let shared_access = std::sync::Arc::new(RelationalAccessPath::EqualityIndex {
        table: "t".to_string(),
        column: "id".to_string(),
        matched_keys: 1,
    });
    // needle 0: TWO rows, DESCENDING emit order (row_index 5 then 2); needle 1: one row; needle 2: ABSENT.
    let rows = vec![
        CudaI32BatchProjectionRow {
            needle_index: 0,
            row_index: 5,
            values: vec![10, 105],
        },
        CudaI32BatchProjectionRow {
            needle_index: 0,
            row_index: 2,
            values: vec![10, 102],
        },
        CudaI32BatchProjectionRow {
            needle_index: 1,
            row_index: 9,
            values: vec![20, 209],
        },
    ];
    let projected = CudaI32BatchProjectionColumns::from_rows(rows);
    let batched = Engine::assemble_batched_rows(&projected, 3, 2, shared_cols, shared_access, 3);
    assert_eq!(batched.needle_count(), 3);
    // needle 0 MUST ascend by row_index: row 2's values THEN row 5's. A no-sort regression -> [10,105,10,102].
    assert_eq!(
        batched.needle_values(0),
        &[10, 102, 10, 105],
        "multi-row needle must be sorted ascending by row_index (the within-needle sort is NECESSARY)"
    );
    assert_eq!(batched.needle_values(1), &[20, 209]);
    assert!(
        batched.needle_values(2).is_empty(),
        "absent needle -> no rows"
    );
}

// The UNIQUE fast-path of assemble_batched_rows (<=1 row/needle, the dominant point read): each row is
// placed DIRECTLY at its needle's offset. Prove it scatters by needle_index regardless of emit order +
// handles an absent needle (count 0) — a regression that kept emit order, or mis-placed the absent gap,
// fails here. Pure CPU (no GPU).
#[test]
fn r2_batched_assembly_unique_fastpath_scatters_by_needle() {
    let col = |name: &str| RelationalColumn {
        id: 0,
        table_oid: 0,
        attnum: 1,
        name: name.to_string(),
        ty: SqlType::Int4,
        domain: None,
        default: None,
        type_oid: 23,
        type_size: 4,
    };
    let shared_cols = std::sync::Arc::new(vec![col("id"), col("v")]);
    let shared_access = std::sync::Arc::new(RelationalAccessPath::EqualityIndex {
        table: "t".to_string(),
        column: "id".to_string(),
        matched_keys: 1,
    });
    // 4 needles, each <=1 row (unique -> fast-path). Emit order is ARBITRARY (needle 2, then 0, then 3);
    // needle 1 is ABSENT (count 0).
    let rows = vec![
        CudaI32BatchProjectionRow {
            needle_index: 2,
            row_index: 7,
            values: vec![22, 202],
        },
        CudaI32BatchProjectionRow {
            needle_index: 0,
            row_index: 3,
            values: vec![10, 100],
        },
        CudaI32BatchProjectionRow {
            needle_index: 3,
            row_index: 1,
            values: vec![33, 303],
        },
    ];
    let projected = CudaI32BatchProjectionColumns::from_rows(rows);
    let batched = Engine::assemble_batched_rows(&projected, 4, 2, shared_cols, shared_access, 3);
    assert_eq!(batched.needle_count(), 4);
    // Output MUST be needle-ordered (0, [1 empty], 2, 3), NOT emit order (2, 0, 3).
    assert_eq!(batched.needle_values(0), &[10, 100]);
    assert!(
        batched.needle_values(1).is_empty(),
        "absent needle 1 -> no rows"
    );
    assert_eq!(batched.needle_values(2), &[22, 202]);
    assert_eq!(batched.needle_values(3), &[33, 303]);
}

#[test]
fn r2_dense_all_present_preserves_public_range_contract_and_expansion() {
    let columns = std::sync::Arc::new(vec![RelationalColumn {
        id: 0,
        table_oid: 0,
        attnum: 1,
        name: "id".to_string(),
        ty: SqlType::Int4,
        domain: None,
        default: None,
        type_oid: 23,
        type_size: 4,
    }]);
    let access = std::sync::Arc::new(RelationalAccessPath::EqualityIndex {
        table: "t".to_string(),
        column: "id".to_string(),
        matched_keys: 3,
    });
    let projected = CudaI32BatchProjectionColumns {
        values: vec![10, 20, 30],
        needle_indices: Vec::new(),
        row_indices: Vec::new(),
        projection_count: 1,
        status: vec![1, 1, 1],
    };
    let _: &[u32] = &projected.status;
    let batched = Engine::assemble_batched_rows(&projected, 3, 1, columns, access, 0);
    assert_eq!(batched.needle_ranges, vec![(0, 1), (1, 1), (2, 1)]);
    assert_eq!(batched.needle_count(), 3);
    assert_eq!(batched.needle_values(0), &[10]);
    assert_eq!(batched.needle_values(1), &[20]);
    assert_eq!(batched.needle_values(2), &[30]);
    let expanded = Engine::expand_ready_batched_result(batched);
    assert_eq!(expanded.len(), 3);
    let rows = expanded
        .into_iter()
        .map(|result| result.rows.into_boxed())
        .collect::<Vec<_>>();
    assert_eq!(rows[0], vec![vec![SqlValue::Int4(10)]]);
    assert_eq!(rows[1], vec![vec![SqlValue::Int4(20)]]);
    assert_eq!(rows[2], vec![vec![SqlValue::Int4(30)]]);

    // The production batcher may carry the identity mapping compactly, but conversion at the established
    // public boundary must recreate the same one-range-per-needle ABI.
    let compact_columns = std::sync::Arc::new(vec![RelationalColumn {
        id: 0,
        table_oid: 0,
        attnum: 1,
        name: "id".to_string(),
        ty: SqlType::Int4,
        domain: None,
        default: None,
        type_oid: 23,
        type_size: 4,
    }]);
    let compact_access = std::sync::Arc::new(RelationalAccessPath::EqualityIndex {
        table: "t".to_string(),
        column: "id".to_string(),
        matched_keys: 3,
    });
    let compact = RelationalPointBatchResult::new(
        compact_columns,
        compact_access,
        0,
        vec![10, 20, 30],
        1,
        Vec::new(),
    );
    assert_eq!(compact.needle_count(), 3);
    assert_eq!(compact.needle_values(1), &[20]);
    let compat = compact.into_compat();
    assert_eq!(compat.needle_ranges, vec![(0, 1), (1, 1), (2, 1)]);
}

// DECISIONS "lpb read levers" #1: the DENSE-emit unique index probe must be BYTE-IDENTICAL to the atomic
// index kernel, across the edge cases the kernel must get right: all-match, none-match, absent needles (the
// GAP case — validates that absent slots are never read as present), and NULL-as-0. The same index route is
// taken either way (lpb index, wave_engine on / persistent off); only the kernel + host compaction differ.
// Non-vacuity: dense_index_probe_hits must increment (the dense kernel actually ran — no silent fallback to
// atomic/scan; output equality alone can't prove it since the two are byte-identical by design). Distinct
// needles only (the route dedups; the dense kernel handles duplicates per-slot by design but the contract
// prevents them reaching it, so they're not exercised here). GPU test.
