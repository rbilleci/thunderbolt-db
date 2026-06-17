use super::*;

#[test]
fn p8_resident_route_decisions_use_cache_state_and_default_fallbacks() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();

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

    let normal = e.execute_relational_select(&count_select).unwrap();
    assert_eq!(normal.planned_target, DeviceTarget::Gpu(0));
    if decision.accepted {
        assert_eq!(normal.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(normal.fallback_reason, None);
    } else {
        assert_eq!(
            normal.fallback_reason,
            Some(FallbackReason::GpuMvccReadParityGap)
        );
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

    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&count_select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident snapshot is Invalidated");
}

#[test]
fn p8_default_resident_route_executes_accepted_shapes() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (id INT, bucket INT, amount INT, label TEXT)",
    )
    .unwrap();
    // Bucket 1 (10+20) and bucket 2 (30) both sum to 30 — a deliberate tie that all three paths
    // (resident GPU probe, cuda-driver-probe, default host) must resolve IDENTICALLY: equal SUMs
    // break by group ASC, so `... ORDER BY sum DESC LIMIT 1` is deterministically bucket 1. The
    // resident path finalizes this in `launch_cuda_resident_i32_grouped_stats`; the host paths in
    // `finalize_relational_select` (direction applied to the aggregate, group-ASC tie-break kept).
    e.execute_text(
            2,
            "INSERT INTO events (id, bucket, amount, label) VALUES (1, 1, 10, 'alpha'), (2, 1, 20, 'beta'), (3, 2, 30, 'alpine')",
        )
        .unwrap();
    e.populate_relational_residency_snapshot("events").unwrap();

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

    for sql in [
            "SELECT COUNT(*) FROM events",
            "SELECT COUNT(*) FROM events WHERE id = 2",
            "SELECT COUNT(*) FROM events WHERE id > 1",
            "SELECT COUNT(*) FROM events WHERE label LIKE 'al%'",
            "SELECT COUNT(*) FROM events WHERE id = 1 OR id = 3",
            "SELECT SUM(id) FROM events",
            "SELECT AVG(id) FROM events",
            "SELECT MIN(id) FROM events",
            "SELECT MAX(id) FROM events",
            "SELECT SUM(id) FROM events WHERE id > 1",
            "SELECT AVG(id) FROM events WHERE id < 3",
            "SELECT MIN(id) FROM events WHERE id >= 2",
            "SELECT MAX(id) FROM events WHERE id <= 2",
            "SELECT SUM(id) FROM events WHERE id BETWEEN 1 AND 2",
            "SELECT AVG(id) FROM events WHERE id BETWEEN 2 AND 3",
            "SELECT MIN(id) FROM events WHERE id BETWEEN 1 AND 3",
            "SELECT MAX(id) FROM events WHERE id BETWEEN 1 AND 1",
            "SELECT id FROM events WHERE id > 1",
            "SELECT id FROM events WHERE id >= 1 ORDER BY id DESC LIMIT 2 OFFSET 1",
            "SELECT DISTINCT bucket FROM events ORDER BY bucket DESC LIMIT 2 OFFSET 1",
            "SELECT DISTINCT bucket FROM events WHERE bucket >= 2 ORDER BY bucket DESC LIMIT 2 OFFSET 1",
            "SELECT bucket, COUNT(*) FROM events GROUP BY bucket HAVING count >= 1 ORDER BY bucket",
            "SELECT bucket, SUM(amount) FROM events GROUP BY bucket HAVING sum > 20 ORDER BY sum DESC LIMIT 1",
            "SELECT bucket, AVG(amount) FROM events GROUP BY bucket ORDER BY bucket",
            "SELECT bucket, MIN(amount) FROM events WHERE amount >= 20 GROUP BY bucket HAVING min >= 20 ORDER BY bucket",
            "SELECT bucket, MAX(amount) FROM events WHERE amount > 10 GROUP BY bucket HAVING bucket = 1 OR max >= 30 ORDER BY max DESC",
        ] {
            let Command::Select(select) = parse_command(sql).unwrap() else {
                unreachable!()
            };
            let expected = e
                .execute_relational_select_with_cuda_driver_probe(&select)
                .unwrap();
            let resident = e
                .execute_relational_select_with_resident_route(&select)
                .unwrap_or_else(|err| panic!("{sql}: {err}"));
            let before_default_metrics = e.metrics().snapshot();
            let default = e
                .execute_relational_select(&select)
                .unwrap_or_else(|err| panic!("{sql}: {err}"));
            let after_default_metrics = e.metrics().snapshot();
            assert_eq!(resident.rows, expected.rows, "{sql}");
            assert_eq!(resident.columns, expected.columns, "{sql}");
            assert_eq!(default.rows, expected.rows, "{sql}");
            assert_eq!(default.columns, expected.columns, "{sql}");
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
                "count_all" | "int4_equality_count" | "int4_range_count" | "int4_filter_group_count" => {
                    std::mem::size_of::<u64>() as u64
                }
                "text_prefix_like_count" => e
                    .relational_residency_snapshot("events")
                    .unwrap()
                    .resident_bytes,
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
                // execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe
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
    let mut e = Engine::new_local();
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
    e.populate_relational_residency_snapshot("events").unwrap();

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
    assert_eq!(read_jobs[0].snapshot_generation, 1);
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

    e.execute_text(
        3,
        "INSERT INTO events (id, bucket, amount, label) VALUES (4, 2, 40, 'amber')",
    )
    .unwrap();
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
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, amount INT, label TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, amount, label) VALUES (1, 10, 'alpha'), (2, 20, 'beta'), (2, 30, 'delta'), (3, 40, 'gamma')",
        )
        .unwrap();
    e.populate_relational_residency_snapshot("events").unwrap();

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
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some((2 * 2 * std::mem::size_of::<i32>() + std::mem::size_of::<u32>()) as u64)
    );

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
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some((2 * std::mem::size_of::<i32>() + std::mem::size_of::<u32>()) as u64)
    );

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
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(
            (2 * std::mem::size_of::<i32>()
                + std::mem::size_of::<u64>()
                + 2 * std::mem::size_of::<u64>()
                + "gamma".len()
                + std::mem::size_of::<u64>()
                + std::mem::size_of::<u64>()) as u64
        )
    );

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
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();

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

    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
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
    let fallback = e.execute_relational_select(&absent).unwrap();
    assert_eq!(fallback.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(
        fallback.fallback_reason,
        Some(FallbackReason::GpuMvccReadParityGap)
    );
}

#[test]
fn p8_resident_route_decisions_reject_evicted_and_memory_pressured_snapshots() {
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

    let aux = e.populate_relational_residency_snapshot("aux").unwrap();
    let events = e.populate_relational_residency_snapshot("events").unwrap();
    e.set_relational_residency_budget_bytes(0, events.resident_bytes);
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
fn p8_partitioned_resident_count_reduces_valid_partitions_and_rejects_invalidated() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partitions = (0..4_u32)
        .map(|partition_id| {
            let row_count = 256_usize;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            BenchmarkRelationalResidencyOwnedPartition {
                partition_id,
                row_start: partition_id as usize * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: Vec::new(),
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) = parse_command("SELECT COUNT(*) FROM order_line").unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_count_all");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 1024);
    assert_eq!(route.resident_bytes, 32);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.h2d_bytes_if_cold, 32);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int8(1024)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_count_all");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(
        decision.last_execution_kernel_samples,
        Some(
            after
                .kernel_exec_samples
                .saturating_sub(before.kernel_exec_samples)
        )
    );
    assert_eq!(decision.last_execution_matched_rows, Some(4));

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (1, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_count_all");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");
}

#[test]
fn p8_partitioned_resident_key_lookup_merges_matches_and_rejects_invalidated() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [Vec<i32>; 4] = [
        vec![42, 1, 42, 2],
        vec![3, 4, 5, 6],
        vec![42, 7, 8, 42],
        vec![9, 10, 11, 12],
    ];
    let partitions = partition_values
        .iter()
        .enumerate()
        .map(|(partition_id, values)| {
            let row_count = values.len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for value in values {
                bytes.extend_from_slice(&(*value).to_le_bytes());
            }
            BenchmarkRelationalResidencyOwnedPartition {
                partition_id: partition_id as u32,
                row_start: partition_id * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec!["ol_o_id".to_string()],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT ol_o_id FROM order_line WHERE ol_o_id = 42").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_equality_projection");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
        ]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_equality_projection");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_matched_rows, Some(4));

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (42, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(
        invalidated.query_shape,
        "partitioned_int4_equality_projection"
    );
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");
}

#[test]
fn p8_partitioned_resident_multi_column_lookup_merges_projected_rows_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![42, 1, 42, 2],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![500, 501, 502, 503],
        ],
        [
            vec![3, 4, 5, 6],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![600, 601, 602, 603],
        ],
        [
            vec![42, 7, 8, 42],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![700, 701, 702, 703],
        ],
        [
            vec![9, 10, 11, 12],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![800, 801, 802, 803],
        ],
    ];
    let partitions = partition_values
        .iter()
        .enumerate()
        .map(|(partition_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            BenchmarkRelationalResidencyOwnedPartition {
                partition_id: partition_id as u32,
                row_start: partition_id * row_count + 1,
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

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) = parse_command(
        "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = 42",
    )
    .unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(
        route.query_shape,
        "partitioned_int4_equality_multi_column_projection"
    );
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert!(route.d2h_bytes_estimate > 0);

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(100),
                SqlValue::Int4(5),
                SqlValue::Int4(500)
            ],
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(102),
                SqlValue::Int4(7),
                SqlValue::Int4(502)
            ],
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(300),
                SqlValue::Int4(13),
                SqlValue::Int4(700)
            ],
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(303),
                SqlValue::Int4(16),
                SqlValue::Int4(703)
            ],
        ]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "partitioned_int4_equality_multi_column_projection"
    );
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_matched_rows, Some(4));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision.last_execution_selected_projection_micros.is_some());

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (42, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(
        invalidated.query_shape,
        "partitioned_int4_equality_multi_column_projection"
    );
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = partition_values
        .iter()
        .enumerate()
        .map(|(partition_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            BenchmarkRelationalResidencyOwnedPartition {
                partition_id: partition_id as u32,
                row_start: partition_id * row_count + 1,
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
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(
        missing_layout.query_shape,
        "partitioned_int4_equality_multi_column_projection"
    );
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn p8_partitioned_resident_multi_column_lookup_orders_more_than_one_warp_of_matches_per_partition()
{
    // Coverage-gap closer (engine side) for the resident row-index host-sort fix. The
    // partitioned multi-column route iterates partitions in order and, within each partition,
    // materializes one output row per entry of `match_i32_equal_row_indices_from_payload(..)` in
    // that vector's order via a strictly positional gather. The CPU/non-resident reference emits
    // a partition's matching rows in ASCENDING row order. The resident kernel, however, appends
    // matches in `atom.global.add` SCHEDULE order, which is ascending only while all matches in
    // a partition fit in ONE warp (<= 32). Every existing partitioned parity test stays under
    // that boundary (<= 2 matches per partition), so this is the first test that puts MORE THAN
    // ONE WARP of matches in a SINGLE partition.
    //
    // Why this is non-vacuous (would fail/flake WITHOUT the host sort in
    // `launch_cuda_resident_i32_equal_row_indices`): with > 32 interleaved matches in a
    // partition, the kernel's cross-warp append order is non-deterministic and is essentially
    // never ascending, so the positional gather would emit that partition's rows in a
    // non-deterministic, non-ascending order — diverging from the ascending reference asserted
    // below and breaking the partitioned ascending-merge. It passes only because the route now
    // sorts the [0, count) indices host-side. The loop re-runs the query so a sort-less route
    // surfaces a wrong ordering on at least one iteration.
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    const NEEDLE: i32 = 42;
    // Two partitions. Partition 0 carries a MULTI-WARP block of matches: 200 rows where the
    // even rows match the needle (100 matches >> 32, interleaved across many warps and several
    // 128-thread blocks); the projected columns are distinct per row so the asserted order is
    // load-bearing. Partition 1 is a small non-matching tail (exercises the cross-partition
    // merge after the multi-warp partition).
    let p0_rows: usize = 200;
    let p0_ol_o_id: Vec<i32> = (0..p0_rows as i32)
        .map(|row| if row % 2 == 0 { NEEDLE } else { row + 1000 })
        .collect();
    // Make the other three columns unique, monotonic functions of the row so a mis-ordered
    // gather is caught by the exact row comparison (not just by the key column).
    let p0_ol_i_id: Vec<i32> = (0..p0_rows as i32).map(|row| 10_000 + row).collect();
    let p0_ol_quantity: Vec<i32> = (0..p0_rows as i32).map(|row| 20_000 + row).collect();
    let p0_ol_amount: Vec<i32> = (0..p0_rows as i32).map(|row| 30_000 + row).collect();

    let p1_ol_o_id = vec![1, 2, 3, 4];
    let p1_ol_i_id = vec![401, 402, 403, 404];
    let p1_ol_quantity = vec![17, 18, 19, 20];
    let p1_ol_amount = vec![801, 802, 803, 804];

    let partition_columns: Vec<[Vec<i32>; 4]> = vec![
        [p0_ol_o_id, p0_ol_i_id, p0_ol_quantity, p0_ol_amount],
        [p1_ol_o_id, p1_ol_i_id, p1_ol_quantity, p1_ol_amount],
    ];

    // CPU reference: rows from each partition in ASCENDING row order, partitions in order.
    let mut expected_rows: Vec<Vec<SqlValue>> = Vec::new();
    for columns in &partition_columns {
        for (row, key) in columns[0].iter().enumerate() {
            if *key == NEEDLE {
                expected_rows.push(vec![
                    SqlValue::Int4(*key),
                    SqlValue::Int4(columns[1][row]),
                    SqlValue::Int4(columns[2][row]),
                    SqlValue::Int4(columns[3][row]),
                ]);
            }
        }
    }
    assert!(
        expected_rows.len() > 32,
        "test must match more than one warp of rows in a single partition"
    );

    let mut row_cursor = 1usize;
    let partitions = partition_columns
        .iter()
        .enumerate()
        .map(|(partition_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            let row_start = row_cursor;
            row_cursor += row_count;
            BenchmarkRelationalResidencyOwnedPartition {
                partition_id: partition_id as u32,
                row_start,
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

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) = parse_command(
        "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = 42",
    )
    .unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(
        route.query_shape,
        "partitioned_int4_equality_multi_column_projection"
    );
    assert_eq!(route.partition_count, 2);

    // Re-run so a non-deterministic (sort-less) cross-warp order is caught on some iteration.
    for iter in 0..25 {
        let result = e.execute_relational_select(&select).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
        assert_eq!(
            result.rows, expected_rows,
            "partitioned multi-warp resident rows were not in ascending reference order on \
                 iteration {iter} — the resident route's host sort over [0, count) is missing?"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn p8_batched_multi_column_projection_matches_per_query_for_more_than_one_warp_of_matches() {
    // Thread-3 Stage-4 ordered-parity gate (multi-column all-int4). The batched submit/complete
    // path scatters rows per needle in the `equal_any` kernel's `atom.global.add` SCHEDULE
    // order; the per-query path (`execute_relational_select` -> the multi-column probe) now sorts
    // its fused `equal_project` output ascending-by-row_index. Both must return the SAME rows in
    // the SAME (ascending) order for a MULTI-WARP match count (>32, where the atomic-append
    // order is non-deterministic), so the batched output is byte-identical to the per-query path.
    //
    // Non-vacuous: the projected `seq` column is a by-row SCRAMBLED hash, so the ascending-by-row
    // reference is NOT value-sorted and NOT the atomic-append order. WITHOUT the stable-order
    // sort the batched scatter would emit a non-deterministic permutation (caught by the exact
    // comparison and the 25× loop), and it would differ from the per-query path.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (k INT, seq INT)")
        .unwrap();
    const NEEDLE: i32 = 7;
    let rows: usize = 200; // even rows match => 100 matches >> 32 (multi-warp/multi-block).
    let row_value = |row: usize| -> i32 {
        let r = row as u64;
        let h = r.wrapping_mul(2_654_435_761) ^ (r << 13) ^ 0x9E37_79B9;
        (1 + (h % 1_000_000)) as i32
    };
    let mut values = String::new();
    for row in 0..rows {
        if row > 0 {
            values.push_str(", ");
        }
        let k = if row % 2 == 0 {
            NEEDLE
        } else {
            row as i32 + 1000
        };
        values.push_str(&format!("({k}, {})", row_value(row)));
    }
    e.execute_text(2, &format!("INSERT INTO t (k, seq) VALUES {values}"))
        .unwrap();
    e.populate_relational_residency_snapshot("t").unwrap();

    let Command::Select(select) = parse_command("SELECT k, seq FROM t WHERE k = 7").unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    assert_eq!(route.query_shape, "int4_equality_multi_column_projection");

    // Ascending-by-row reference (independent of either GPU path).
    let expected: Vec<Vec<SqlValue>> = (0..rows)
        .filter(|row| row % 2 == 0)
        .map(|row| vec![SqlValue::Int4(NEEDLE), SqlValue::Int4(row_value(row))])
        .collect();
    assert!(
        expected.len() > 32,
        "test must match more than one warp of rows"
    );
    // The projected by-row sequence must NOT already be value-sorted, else an atomic-append or
    // value-sort impl would pass vacuously.
    let proj_by_row: Vec<i32> = expected
        .iter()
        .map(|row| match row[1] {
            SqlValue::Int4(v) => v,
            _ => unreachable!(),
        })
        .collect();
    let mut proj_sorted = proj_by_row.clone();
    proj_sorted.sort_unstable();
    assert_ne!(
        proj_by_row, proj_sorted,
        "projected by-row sequence is accidentally value-sorted — pick a payload that isn't"
    );

    for iter in 0..25 {
        let per_query = e.execute_relational_select(&select).unwrap();
        assert_eq!(
            per_query.rows, expected,
            "per-query multi-column rows not ascending-by-row on iteration {iter}"
        );

        let job = e.prepare_relational_retained_read_job(&select).unwrap();
        let submission = e
            .submit_relational_retained_read_jobs_with_resident_device_memory_probe(
                std::slice::from_ref(&job),
            )
            .unwrap();
        let batched = e
            .complete_relational_retained_read_submission(submission)
            .unwrap();
        assert_eq!(batched.len(), 1);
        assert_eq!(
            batched[0].rows, expected,
            "batched multi-column rows not ascending-by-row on iteration {iter} \
                 — the stable-order sort in the result assembly is missing or ineffective?"
        );
        assert_eq!(
            batched[0].rows, per_query.rows,
            "batched multi-column output diverged from the per-query path on iteration {iter}"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn p8_batched_mixed_column_projection_matches_per_query_for_more_than_one_warp_of_matches() {
    // Thread-3 Stage-4 ordered-parity gate (mixed int4/text). The batched path for the mixed
    // shape falls through to `..._batch_inner` (the int4 `equal_any` fast path is int4-only), and
    // the per-query mixed path delegates to the SAME `batch_inner`; both now sort each needle's
    // slice ascending-by-row_index. They must return the SAME rows in the SAME order for a
    // MULTI-WARP match count.
    //
    // Non-vacuous: the projected `label` text is a by-row SCRAMBLED value, so the ascending-by-row
    // reference is neither value-sorted nor the atomic-append order. WITHOUT the sort the scatter
    // is a non-deterministic permutation (caught by the exact comparison + the 25× loop).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (k INT, label TEXT)")
        .unwrap();
    const NEEDLE: i32 = 7;
    let rows: usize = 200; // even rows match => 100 matches >> 32.
                           // Scrambled-by-row label so the ascending-by-row text order is not lexicographically sorted.
    let label_value = |row: usize| -> String {
        let r = row as u64;
        let h = r.wrapping_mul(2_654_435_761) ^ (r << 13) ^ 0x9E37_79B9;
        format!("L{:08}", h % 100_000_000)
    };
    let mut values = String::new();
    for row in 0..rows {
        if row > 0 {
            values.push_str(", ");
        }
        let k = if row % 2 == 0 {
            NEEDLE
        } else {
            row as i32 + 1000
        };
        values.push_str(&format!("({k}, '{}')", label_value(row)));
    }
    e.execute_text(2, &format!("INSERT INTO t (k, label) VALUES {values}"))
        .unwrap();
    e.populate_relational_residency_snapshot("t").unwrap();

    let Command::Select(select) = parse_command("SELECT k, label FROM t WHERE k = 7").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    assert_eq!(route.query_shape, "int4_equality_mixed_column_projection");

    let expected: Vec<Vec<SqlValue>> = (0..rows)
        .filter(|row| row % 2 == 0)
        .map(|row| vec![SqlValue::Int4(NEEDLE), SqlValue::Text(label_value(row))])
        .collect();
    assert!(
        expected.len() > 32,
        "test must match more than one warp of rows"
    );
    let label_by_row: Vec<String> = expected
        .iter()
        .map(|row| match &row[1] {
            SqlValue::Text(v) => v.clone(),
            _ => unreachable!(),
        })
        .collect();
    let mut label_sorted = label_by_row.clone();
    label_sorted.sort();
    assert_ne!(
        label_by_row, label_sorted,
        "projected by-row label sequence is accidentally sorted — pick a payload that isn't"
    );

    for iter in 0..25 {
        let per_query = e.execute_relational_select(&select).unwrap();
        assert_eq!(
            per_query.rows, expected,
            "per-query mixed rows not ascending-by-row on iteration {iter}"
        );

        let job = e.prepare_relational_retained_read_job(&select).unwrap();
        let submission = e
            .submit_relational_retained_read_jobs_with_resident_device_memory_probe(
                std::slice::from_ref(&job),
            )
            .unwrap();
        let batched = e
            .complete_relational_retained_read_submission(submission)
            .unwrap();
        assert_eq!(batched.len(), 1);
        assert_eq!(
            batched[0].rows, expected,
            "batched mixed rows not ascending-by-row on iteration {iter} \
                 — the stable-order sort in the result assembly is missing or ineffective?"
        );
        assert_eq!(
            batched[0].rows, per_query.rows,
            "batched mixed output diverged from the per-query path on iteration {iter}"
        );
    }
}

#[test]
fn p8_partitioned_resident_sum_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![42, 1, 42, 2],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![500, 501, 502, 503],
        ],
        [
            vec![3, 4, 5, 6],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![600, 601, 602, 603],
        ],
        [
            vec![42, 7, 8, 42],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![700, 701, 702, 703],
        ],
        [
            vec![9, 10, 11, 12],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![800, 801, 802, 803],
        ],
    ];
    let build_partitions = || {
        partition_values
            .iter()
            .enumerate()
            .map(|(partition_id, columns)| {
                let row_count = columns[0].len();
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
                for column in columns {
                    for value in column {
                        bytes.extend_from_slice(&(*value).to_le_bytes());
                    }
                }
                BenchmarkRelationalResidencyOwnedPartition {
                    partition_id: partition_id as u32,
                    row_start: partition_id * row_count + 1,
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
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: build_partitions(),
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT SUM(ol_amount) FROM order_line WHERE ol_o_id = 42").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_equality_sum");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int8(2405)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_equality_sum");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(4));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision.last_execution_selected_projection_micros.is_some());
    assert!(decision
        .last_execution_result_materialization_micros
        .is_some());

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (42, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_int4_equality_sum");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = build_partitions();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(missing_layout.query_shape, "partitioned_int4_equality_sum");
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_partitioned_resident_between_avg_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![10, 20, 30, 40],
        ],
        [
            vec![15, 25, 35, 45],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![15, 25, 35, 45],
        ],
        [
            vec![50, 60, 70, 80],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![50, 60, 70, 80],
        ],
        [
            vec![20, 21, 22, 23],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![20, 21, 22, 23],
        ],
    ];
    let build_partitions = || {
        partition_values
            .iter()
            .enumerate()
            .map(|(partition_id, columns)| {
                let row_count = columns[0].len();
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
                for column in columns {
                    for value in column {
                        bytes.extend_from_slice(&(*value).to_le_bytes());
                    }
                }
                BenchmarkRelationalResidencyOwnedPartition {
                    partition_id: partition_id as u32,
                    row_start: partition_id * row_count + 1,
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
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: build_partitions(),
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_o_id BETWEEN 20 AND 35")
            .unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_between_avg");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Numeric(
            Decimal128::parse("24.5000000000000000").unwrap()
        )]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_between_avg");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(8));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision.last_execution_selected_projection_micros.is_some());
    assert!(decision
        .last_execution_result_materialization_micros
        .is_some());

    let Command::Select(no_match_select) =
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_o_id BETWEEN 90 AND 99")
            .unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    // AVG over zero matched rows now yields the canonical-zero numeric sentinel.
    assert_eq!(
        no_match.rows,
        vec![vec![SqlValue::Numeric(Decimal128::ZERO)]]
    );

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_int4_between_avg");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = build_partitions();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(missing_layout.query_shape, "partitioned_int4_between_avg");
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_partitioned_resident_filtered_max_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![10, 25, 70, 15],
        ],
        [
            vec![50, 60, 70, 80],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![20, 30, 40, 10],
        ],
        [
            vec![90, 91, 92, 93],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![75, 55, 65, 80],
        ],
        [
            vec![94, 95, 96, 97],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![1, 2, 3, 4],
        ],
    ];
    let build_partitions = || {
        partition_values
            .iter()
            .enumerate()
            .map(|(partition_id, columns)| {
                let row_count = columns[0].len();
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
                for column in columns {
                    for value in column {
                        bytes.extend_from_slice(&(*value).to_le_bytes());
                    }
                }
                BenchmarkRelationalResidencyOwnedPartition {
                    partition_id: partition_id as u32,
                    row_start: partition_id * row_count + 1,
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
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: build_partitions(),
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= 50").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_filtered_max");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(80)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_filtered_max");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(5));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision
        .last_execution_result_materialization_micros
        .is_some());

    let Command::Select(no_match_select) =
        parse_command("SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= 100").unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    assert_eq!(no_match.rows, vec![vec![SqlValue::Text(String::new())]]);

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 99, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_int4_filtered_max");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = build_partitions();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(missing_layout.query_shape, "partitioned_int4_filtered_max");
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_partitioned_resident_filtered_min_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![90, 25, 70, 85],
        ],
        [
            vec![50, 60, 70, 80],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![20, 30, 40, 10],
        ],
        [
            vec![90, 91, 92, 93],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![75, 55, 65, 80],
        ],
        [
            vec![94, 95, 96, 97],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![91, 92, 93, 94],
        ],
    ];
    let build_partitions = || {
        partition_values
            .iter()
            .enumerate()
            .map(|(partition_id, columns)| {
                let row_count = columns[0].len();
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
                for column in columns {
                    for value in column {
                        bytes.extend_from_slice(&(*value).to_le_bytes());
                    }
                }
                BenchmarkRelationalResidencyOwnedPartition {
                    partition_id: partition_id as u32,
                    row_start: partition_id * row_count + 1,
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
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: build_partitions(),
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT MIN(ol_amount) FROM order_line WHERE ol_amount <= 60").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_filtered_min");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(10)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_filtered_min");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(6));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision
        .last_execution_result_materialization_micros
        .is_some());

    let Command::Select(no_match_select) =
        parse_command("SELECT MIN(ol_amount) FROM order_line WHERE ol_amount <= 0").unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    assert_eq!(no_match.rows, vec![vec![SqlValue::Text(String::new())]]);

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 5, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_int4_filtered_min");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = build_partitions();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(missing_layout.query_shape, "partitioned_int4_filtered_min");
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_partitioned_resident_filtered_avg_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![10, 25, 70, 15],
        ],
        [
            vec![50, 60, 70, 80],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![20, 30, 40, 10],
        ],
        [
            vec![90, 91, 92, 93],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![75, 55, 65, 80],
        ],
        [
            vec![94, 95, 96, 97],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![91, 92, 93, 94],
        ],
    ];
    let build_partitions = || {
        partition_values
            .iter()
            .enumerate()
            .map(|(partition_id, columns)| {
                let row_count = columns[0].len();
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
                for column in columns {
                    for value in column {
                        bytes.extend_from_slice(&(*value).to_le_bytes());
                    }
                }
                BenchmarkRelationalResidencyOwnedPartition {
                    partition_id: partition_id as u32,
                    row_start: partition_id * row_count + 1,
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
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: build_partitions(),
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_amount <= 60").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_filtered_avg");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Numeric(
            Decimal128::parse("25.6250000000000000").unwrap()
        )]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_filtered_avg");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(8));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision
        .last_execution_result_materialization_micros
        .is_some());

    let Command::Select(no_match_select) =
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_amount <= 0").unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    // AVG over zero matched rows now yields the canonical-zero numeric sentinel.
    assert_eq!(
        no_match.rows,
        vec![vec![SqlValue::Numeric(Decimal128::ZERO)]]
    );

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 5, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_int4_filtered_avg");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = build_partitions();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(missing_layout.query_shape, "partitioned_int4_filtered_avg");
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_resident_warmup_policy_warms_refreshes_and_reports_route_readiness() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();

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
    assert_eq!(first_handle.row_count, 2);
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

    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
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
    assert_eq!(refreshed_handle.generation, 2);
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
    let mut e = Engine::new_local();
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

    let events_size = e.populate_relational_residency_snapshot("events").unwrap();
    let aux_size = e.populate_relational_residency_snapshot("aux").unwrap();
    e.clear_relational_residency_budget_bytes(0);
    let report = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec![
            "events".to_string(),
            "aux".to_string(),
            "missing".to_string(),
        ],
        budget_bytes: Some(events_size.resident_bytes),
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(report.budget_bytes, Some(events_size.resident_bytes));
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
fn p8_resident_maintenance_tick_summarizes_refresh_and_route_readiness() {
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

    e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    e.execute_text(5, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
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
    let mut pressured = Engine::new_local();
    pressured
        .execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    pressured
        .execute_text(2, "INSERT INTO events (id, label) VALUES (1, 'alpha')")
        .unwrap();
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

    let mut oversized = Engine::new_local();
    oversized
        .execute_text(1, "CREATE TABLE oversized (id INT, label TEXT)")
        .unwrap();
    oversized
        .execute_text(
            2,
            "INSERT INTO oversized (id, label) VALUES (1, 'this-row-is-too-large')",
        )
        .unwrap();
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
    let mut e = Engine::new_local();
    let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
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
    let mut e = Engine::new_local();
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
