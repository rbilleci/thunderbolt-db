use super::*;

#[test]
fn planner_targets_mutations_to_gpu() {
    let e = Engine::new_local_cpu_oracle();
    let plan = e.plan_text("SET a=1").unwrap();

    assert_eq!(plan.nodes().len(), 1);
    assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(0));
}

#[test]
fn planner_targets_get_to_cpu_fallback_path() {
    let e = Engine::new_local_cpu_oracle();
    let plan = e.plan_text("GET a").unwrap();

    assert_eq!(plan.nodes().len(), 1);
    assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Cpu);
}

#[test]
fn planner_config_can_override_default_gpu_target() {
    let e = Engine::with_planner_config(PlannerConfig { default_gpu_id: 3 });
    let plan = e.plan_text("SET a=1").unwrap();

    assert_eq!(plan.nodes().len(), 1);
    assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(3));
}

#[test]
fn wal_before_visibility_holds() {
    let e = Engine::new_local_cpu_oracle();
    let t = e.commit_mutation(1, b"SET a=1".to_vec().into()).unwrap();
    assert!(e.wal_flushed_count() >= 1);
    assert!(e.visible_up_to() >= t.index);
    assert!(e.applied_len() >= 1);
}

#[test]
fn commit_indices_monotonic() {
    let e = Engine::new_local_cpu_oracle();
    let a = e.commit_mutation(1, b"SET a=1".to_vec().into()).unwrap();
    let b = e.commit_mutation(2, b"SET b=2".to_vec().into()).unwrap();
    assert!(b.index > a.index);
    assert!(e.visible_up_to() >= b.index);
}

#[test]
fn execute_set_updates_state_machine() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET balance=100").unwrap();
    assert_eq!(e.get("balance").as_deref(), Some("100"));
    assert_eq!(e.metrics().snapshot().commits_total, 1);
}

#[test]
fn execute_set_accepts_session_and_local_scope_aliases() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET SESSION balance=100").unwrap();
    e.execute_text(2, "SET LOCAL balance TO 101").unwrap();

    assert_eq!(e.get("balance").as_deref(), Some("101"));
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn execute_del_removes_existing_key() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET balance=100").unwrap();
    e.execute_text(2, "DEL balance").unwrap();

    assert_eq!(e.get("balance"), None);
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn execute_delete_alias_removes_existing_key() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET balance=100").unwrap();
    e.execute_text(2, "DELETE balance").unwrap();

    assert_eq!(e.get("balance"), None);
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn execute_read_text_get_returns_current_value_without_committing() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET balance=100").unwrap();

    let value = e.execute_read_text("GET balance").unwrap();
    assert_eq!(value.as_deref(), Some("100"));
    assert_eq!(e.metrics().snapshot().commits_total, 1);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 1);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, "100".len() as u64);
    assert_eq!(
        e.metrics().last_fallback_reason(),
        Some(FallbackReason::NotGpuEligible)
    );
}

#[test]
fn execute_read_text_get_missing_key_does_not_track_d2h_bytes() {
    let mut e = Engine::new_local_cpu_oracle();
    let value = e.execute_read_text("GET absent").unwrap();

    assert_eq!(value, None);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, 0);
}

#[test]
fn execute_read_text_rejects_non_read_commands() {
    let mut e = Engine::new_local_cpu_oracle();

    let begin_err = e.execute_read_text("BEGIN").unwrap_err();
    assert!(matches!(begin_err, ExecuteError::NonReadCommand("BEGIN")));

    let set_err = e.execute_read_text("SET balance=100").unwrap_err();
    assert!(matches!(set_err, ExecuteError::NonReadCommand("SET")));

    let reset_err = e.execute_read_text("RESET ALL").unwrap_err();
    assert!(matches!(
        reset_err,
        ExecuteError::NonReadCommand("RESET ALL")
    ));

    let discard_err = e.execute_read_text("DISCARD TEMP").unwrap_err();
    assert!(matches!(
        discard_err,
        ExecuteError::NonReadCommand("RESET ALL")
    ));

    let del_err = e.execute_read_text("DELETE balance").unwrap_err();
    assert!(matches!(
        del_err,
        ExecuteError::NonReadCommand("DEL/DELETE")
    ));

    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn execute_read_text_rejects_get_when_not_leader() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET balance=100").unwrap();
    e.become_follower(2);

    let err = e.execute_read_text("GET balance").unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, 0);
}

#[test]
fn execute_text_get_rejects_when_not_leader() {
    let mut e = Engine::new_local_cpu_oracle();
    e.become_follower(2);

    let err = e.execute_text(1, "GET balance").unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn execute_text_get_rejects_when_candidate() {
    let mut e = Engine::new_local_cpu_oracle();
    e.become_candidate(2);

    let err = e.execute_text(1, "GET balance").unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, 0);
}

#[test]
fn execute_text_get_tracks_d2h_bytes_for_hits_only() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET balance=100").unwrap();

    e.execute_text(2, "GET balance").unwrap();
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, "100".len() as u64);

    e.execute_text(3, "GET missing").unwrap();
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, "100".len() as u64);
}

#[test]
fn execute_read_text_rejects_get_when_candidate() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET balance=100").unwrap();
    e.become_candidate(2);

    let err = e.execute_read_text("GET balance").unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, 0);
}

#[test]
fn batching_flushes_on_count_and_updates_metric() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.enqueue_set_text(2, "SET b=2", t0).unwrap();
    assert_eq!(e.get("a").as_deref(), Some("1"));
    assert_eq!(e.get("b").as_deref(), Some("2"));
    assert_eq!(e.metrics().snapshot().batch_flush_count, 1);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 1);
    assert_eq!(
        e.metrics().last_batch_flush_reason(),
        Some(BatchFlushReason::Count)
    );
    assert_eq!(e.metrics().snapshot().batch_wait_samples, 2);
    assert_eq!(e.metrics().snapshot().batch_wait_total_ms, 0);
    assert_eq!(e.metrics().last_batch_wait_ms(), Some(0));
    assert_eq!(
        e.metrics().snapshot().h2d_bytes_total,
        "SET a=1".len() as u64 + "SET b=2".len() as u64
    );
    assert_eq!(e.metrics().snapshot().kernel_exec_samples, 2);
    assert_eq!(e.metrics().snapshot().kernel_exec_total_ms, 2);
    assert_eq!(e.metrics().last_kernel_exec_ms(), Some(1));
    assert_eq!(e.metrics().snapshot().kernel_occupancy_samples, 2);
    assert_eq!(
        e.metrics().snapshot().kernel_occupancy_total_permyriad,
        6400
    );
    assert_eq!(e.metrics().last_kernel_occupancy_permyriad(), Some(3200));
    assert_eq!(e.metrics().snapshot().pending_batch_peak, 2);
    assert_eq!(e.metrics().last_pending_batch_len(), Some(0));
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn batching_flushes_on_time() {
    let mut e = Engine::with_batching(10, Duration::from_millis(2));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=7", t0).unwrap();
    assert!(e.has_pending_batch());
    assert_eq!(e.pending_batch_len(), 1);
    e.tick_batching(t0 + Duration::from_millis(3)).unwrap();
    assert_eq!(e.get("a").as_deref(), Some("7"));
    assert!(!e.has_pending_batch());
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 1);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Time), 1);
    assert_eq!(e.metrics().snapshot().batch_wait_samples, 1);
    assert_eq!(e.metrics().snapshot().batch_wait_total_ms, 3);
    assert_eq!(e.metrics().last_batch_wait_ms(), Some(3));
}

#[test]
fn batching_kernel_occupancy_caps_at_full_utilization() {
    let mut e = Engine::with_batching(1, Duration::from_secs(999));
    let t0 = Instant::now();
    let payload = format!("SET a={}", "x".repeat(512));

    e.enqueue_set_text(1, &payload, t0).unwrap();

    assert_eq!(e.metrics().snapshot().kernel_occupancy_samples, 1);
    assert_eq!(e.metrics().last_kernel_occupancy_permyriad(), Some(10_000));
    assert_eq!(
        e.metrics().snapshot().kernel_occupancy_total_permyriad,
        10_000
    );
}

#[test]
fn admin_flush_tracks_reason() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=9", t0).unwrap();
    assert_eq!(e.pending_batch_len(), 1);
    e.flush_admin().unwrap();

    assert_eq!(e.get("a").as_deref(), Some("9"));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
}

#[test]
fn admin_flush_without_pending_queue_is_noop() {
    let e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();

    e.flush_admin().unwrap();

    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.pending_batch_oldest_age(t0), None);
    assert_eq!(e.pending_batch_time_until_deadline(t0), None);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 0);
    assert_eq!(e.metrics().last_batch_flush_reason(), None);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn batching_config_reflects_engine_settings() {
    let e = Engine::with_batching(7, Duration::from_millis(42));
    assert_eq!(e.batching_config(), (7, Duration::from_millis(42)));
}

#[test]
fn batching_and_planner_config_can_be_combined() {
    let e = Engine::with_batching_and_planner_config(
        3,
        Duration::from_millis(9),
        PlannerConfig { default_gpu_id: 5 },
    );

    assert_eq!(e.batching_config(), (3, Duration::from_millis(9)));
    let plan = e.plan_text("SET a=1").unwrap();
    assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(5));
}

#[test]
fn pending_batch_deadline_counts_down_and_clears_after_flush() {
    let mut e = Engine::with_batching(10, Duration::from_millis(10));
    let t0 = Instant::now();

    assert_eq!(e.pending_batch_time_until_deadline(t0), None);

    e.enqueue_set_text(1, "SET a=9", t0).unwrap();
    assert_eq!(
        e.pending_batch_time_until_deadline(t0 + Duration::from_millis(4)),
        Some(Duration::from_millis(6))
    );
    assert_eq!(
        e.pending_batch_time_until_deadline(t0 + Duration::from_millis(12)),
        Some(Duration::ZERO)
    );

    e.flush_admin().unwrap();
    assert_eq!(
        e.pending_batch_time_until_deadline(t0 + Duration::from_millis(13)),
        None
    );
}

#[test]
fn pending_batch_oldest_age_tracks_then_clears_after_flush() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=9", t0).unwrap();

    let age = e
        .pending_batch_oldest_age(t0 + Duration::from_millis(5))
        .expect("pending batch age should exist");
    assert!(age >= Duration::from_millis(5));

    e.flush_admin().unwrap();
    assert_eq!(
        e.pending_batch_oldest_age(t0 + Duration::from_millis(6)),
        None
    );
}

#[test]
fn batching_can_apply_set_then_del_in_order() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.enqueue_set_text(2, "DEL a", t0).unwrap();

    assert_eq!(e.get("a"), None);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 1);
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn deterministic_replay_matches_between_immediate_and_batched_mutation_paths() {
    let trace = [
        (1, "SET acct_a=10"),
        (2, "SET acct_b=25"),
        (3, "DEL acct_a"),
        (4, "SET acct_c=77"),
        (5, "DELETE acct_b"),
        (6, "SET acct_a=99"),
    ];

    let immediate = Engine::new_local_cpu_oracle();
    for (txn_id, cmd) in trace {
        immediate.execute_text(txn_id, cmd).unwrap();
    }

    let mut batched = Engine::with_batching(64, Duration::from_secs(999));
    let t0 = Instant::now();
    for (txn_id, cmd) in trace {
        batched.enqueue_set_text(txn_id, cmd, t0).unwrap();
    }
    batched.flush_admin().unwrap();

    assert_eq!(
        immediate.commit_state().sm.applied,
        batched.commit_state().sm.applied
    );
    assert_eq!(immediate.commit_state().sm.kv, batched.commit_state().sm.kv);
    assert_eq!(immediate.visible_up_to(), batched.visible_up_to());
    assert_eq!(
        immediate.visible_state_fingerprint(),
        batched.visible_state_fingerprint()
    );
    assert_eq!(immediate.wal_flushed_count(), trace.len());
    assert_eq!(batched.wal_flushed_count(), trace.len());
}

#[test]
fn visible_state_fingerprint_changes_with_visible_kv_state() {
    let e = Engine::new_local_cpu_oracle();
    let empty = e.visible_state_fingerprint();

    e.execute_text(1, "SET a=1").unwrap();
    let after_set = e.visible_state_fingerprint();
    assert_ne!(after_set, empty);

    e.execute_text(2, "DELETE a").unwrap();
    let after_delete = e.visible_state_fingerprint();
    assert_eq!(after_delete, empty);
}

#[test]
fn flush_command_drains_pending_batch() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=5", t0).unwrap();
    e.execute_text(2, "FLUSH").unwrap();

    assert_eq!(e.get("a").as_deref(), Some("5"));
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
}

#[test]
fn flush_aliases_drain_pending_batch() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=5", t0).unwrap();
    e.execute_text(2, "FLUSH WAL").unwrap();

    e.enqueue_set_text(3, "SET b=7", t0).unwrap();
    e.execute_text(4, "FLUSH LOG").unwrap();

    assert_eq!(e.get("a").as_deref(), Some("5"));
    assert_eq!(e.get("b").as_deref(), Some("7"));
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 2);
}

#[test]
fn wal_flush_failure_prevents_visibility_advance() {
    let mut e = Engine::new_local_cpu_oracle();
    e.simulate_next_wal_flush_failure();
    let res = e.commit_mutation(1, b"SET a=1".to_vec().into());
    assert!(matches!(res, Err(EngineError::Durability(_))));
    assert_eq!(e.visible_up_to(), 0);
}

#[test]
fn wal_flush_failure_does_not_leak_into_later_successful_commit() {
    let mut e = Engine::new_local_cpu_oracle();
    e.simulate_next_wal_flush_failure();
    let _ = e.commit_mutation(1, b"SET a=1".to_vec().into());

    e.commit_mutation(2, b"SET b=2".to_vec().into()).unwrap();

    assert_eq!(e.get("a"), None);
    assert_eq!(e.get("b").as_deref(), Some("2"));
    assert_eq!(e.applied_len(), 1);
}

#[test]
fn wal_flush_failure_discards_unflushed_record_from_buffer() {
    let mut e = Engine::new_local_cpu_oracle();
    e.simulate_next_wal_flush_failure();

    let err = e
        .commit_mutation(1, b"SET a=1".to_vec().into())
        .unwrap_err();

    assert!(matches!(err, EngineError::Durability(_)));
    assert_eq!(e.wal_flushed_count(), 0);
    assert_eq!(e.wal_buffered_count(), 0);
    assert_eq!(e.wal_unflushed_count(), 0);
}

#[test]
fn durable_wal_records_exclude_failed_commit_attempts() {
    let mut e = Engine::new_local_cpu_oracle();

    e.commit_mutation(1, b"SET a=1".to_vec().into()).unwrap();
    e.simulate_next_wal_flush_failure();
    let _ = e.commit_mutation(2, b"SET b=2".to_vec().into());

    let durable = e.durable_wal_records();
    assert_eq!(durable.len(), 1);
    assert_eq!(durable[0].txn_id, 1);
    let envelope = gpu_db_wal::decode_canonical_record_payload(&durable[0].payload)
        .unwrap()
        .expect("durable success uses canonical authority");
    let replay = Engine::decode_engine_operation(&envelope.fragments[0].body).unwrap();
    assert_eq!(
        Engine::decode_engine_command(&replay).unwrap(),
        Some(parse_command("SET a=1").unwrap())
    );
}

#[test]
fn follower_rejects_commit_without_visibility_or_wal_flush() {
    let mut e = Engine::new_local_cpu_oracle();
    e.become_follower(2);

    let err = e
        .commit_mutation(1, b"SET a=1".to_vec().into())
        .unwrap_err();

    assert!(matches!(err, EngineError::NotLeader));
    assert_eq!(e.visible_up_to(), 0);
    assert_eq!(e.wal_flushed_count(), 0);
    assert_eq!(e.applied_len(), 0);
}

#[test]
fn follower_rejects_batched_enqueue_without_mutating_queue_or_metrics() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    e.become_follower(2);

    let t0 = Instant::now();
    let err = e.enqueue_set_text(1, "SET a=1", t0).unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 0);
    assert_eq!(e.metrics().last_batch_flush_reason(), None);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn enqueue_get_rejects_when_not_leader_without_queue_side_effects() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    e.become_follower(2);

    let t0 = Instant::now();
    let err = e.enqueue_set_text(1, "GET a", t0).unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn enqueue_get_rejects_when_candidate_without_queue_side_effects() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    e.become_candidate(2);

    let t0 = Instant::now();
    let err = e.enqueue_set_text(1, "GET a", t0).unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn enqueue_get_tracks_d2h_bytes_for_hits_only() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET balance=100", t0).unwrap();
    e.flush_admin().unwrap();

    e.enqueue_set_text(2, "GET balance", t0).unwrap();
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, "100".len() as u64);

    e.enqueue_set_text(3, "GET missing", t0).unwrap();
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, "100".len() as u64);
}

#[test]
fn execute_text_mutation_falls_back_to_cpu_when_gpu_is_unavailable() {
    let mut e = Engine::new_local_cpu_oracle();
    e.mark_gpu_unavailable(0);

    e.execute_text(1, "SET balance=100").unwrap();

    assert_eq!(e.get("balance").as_deref(), Some("100"));
    assert_eq!(e.metrics().snapshot().commits_total, 1);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);

    let snapshot = e.telemetry_snapshot();
    assert_eq!(snapshot.gpu_parity_fallback_total(), 1);
    assert!(snapshot.has_gpu_parity_fallbacks());
    assert!(snapshot.has_gpu_runtime_pressure());
    assert_eq!(snapshot.blocked_gpu_ids(), vec![0]);
    assert_eq!(
        snapshot.gpu_parity_fallbacks.get(&GpuParityIssue {
            id: "GPU-120",
            owner: "runtime",
            milestone: "m0-bootstrap",
        }),
        Some(&1)
    );
}

#[test]
fn enqueue_mutation_falls_back_to_cpu_when_gpu_is_memory_pressured() {
    let mut e = Engine::with_batching(8, Duration::from_secs(60));
    let t0 = Instant::now();
    e.mark_gpu_memory_pressured(0);

    e.enqueue_set_text(1, "SET balance=100", t0).unwrap();

    assert_eq!(e.get("balance").as_deref(), Some("100"));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().commits_total, 1);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(
        e.metrics().fallback_for(FallbackReason::GpuMemoryPressure),
        1
    );
}

#[test]
fn enqueue_mutation_runtime_saturation_falls_back_before_queueing() {
    let mut e = Engine::with_batching(8, Duration::from_secs(60));
    let t0 = Instant::now();
    e.set_gpu_runtime_saturated(true);

    e.enqueue_set_text(1, "SET balance=100", t0).unwrap();

    assert_eq!(e.get("balance").as_deref(), Some("100"));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().commits_total, 1);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(
        e.metrics().fallback_for(FallbackReason::GpuQueueSaturated),
        1
    );

    let snapshot = e.telemetry_snapshot();
    assert!(snapshot.has_gpu_runtime_pressure());
    assert!(snapshot.blocked_gpu_ids().is_empty());
    assert!(snapshot.gpu_runtime.saturated);
}

#[test]
fn candidate_rejects_commit_and_batched_enqueue() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    e.become_candidate(2);

    let commit_err = e
        .commit_mutation(1, b"SET a=1".to_vec().into())
        .unwrap_err();
    assert!(matches!(commit_err, EngineError::NotLeader));

    let enqueue_err = e
        .enqueue_set_text(1, "SET a=1", Instant::now())
        .unwrap_err();
    assert!(matches!(
        enqueue_err,
        ExecuteError::Engine(EngineError::NotLeader)
    ));

    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
    assert_eq!(e.visible_up_to(), 0);
}

#[test]
fn failed_admin_flush_does_not_increment_flush_metrics_or_drop_pending_queue() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.become_follower(2);

    let err = e.flush_admin().unwrap_err();

    assert!(matches!(err, EngineError::NotLeader));
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 0);
    assert_eq!(e.metrics().last_batch_flush_reason(), None);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
    assert_eq!(e.pending_batch_len(), 1);
}

#[test]
fn batch_flush_is_one_wal_fsync_group_for_all_items() {
    // D3 (write-path assessment / ledger #7): a k-item batch commits as ONE WAL flush group —
    // k appended records made durable by a single fsync — observed via the durable WAL's
    // group-commit accounting. Previously each item drove its own commit_mutation => k fsyncs.
    let path = test_wal_path("batch-one-fsync");
    let mut e = Engine::with_batching(8, Duration::from_secs(60));
    e.commit_state_mut().wal = WalBuffer::with_durable_segment(&path);
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.enqueue_set_text(2, "SET b=2", t0).unwrap();
    e.enqueue_set_text(3, "SET c=3", t0).unwrap();
    e.flush_admin().unwrap();

    let stats = e.wal_group_commit_stats();
    assert_eq!(stats.flush_groups, 1, "one fsync for the whole batch");
    assert_eq!(stats.durable_records, 3);
    assert_eq!(stats.max_group_size, 3);
    assert_eq!(e.get("a").as_deref(), Some("1"));
    assert_eq!(e.get("b").as_deref(), Some("2"));
    assert_eq!(e.get("c").as_deref(), Some("3"));
    assert_eq!(e.metrics().snapshot().commits_total, 3);

    // The group's records are real durable WAL records: a restart replays all three.
    drop(e);
    let recovered = Engine::open_durable_wal_segment(&path).unwrap();
    assert_eq!(recovered.wal_flushed_count(), 3);
    assert_eq!(recovered.get("a").as_deref(), Some("1"));
    assert_eq!(recovered.get("c").as_deref(), Some("3"));
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn multi_entry_apply_uses_the_working_catalog_before_publication() {
    // DUR-002: grouped apply holds the catalog latch and publishes only after every entry. Every
    // nested existence/dependency lookup must therefore bind to the evolving working catalog, not
    // the still-published pre-batch generation. This deliberately uses the internal grouped commit
    // seam: public statement preflight cannot manufacture the dependency because entry 1 has not
    // published when entry 2 is submitted.
    let e = Engine::new_local_cpu_oracle();
    let batch = [
        (
            1,
            std::sync::Arc::from(
                b"CREATE TABLE working_batch (id INT PRIMARY KEY, value INT)".as_slice(),
            ),
        ),
        (
            2,
            std::sync::Arc::from(
                b"INSERT INTO working_batch (id, value) VALUES (1, 10)".as_slice(),
            ),
        ),
        (
            3,
            std::sync::Arc::from(b"CREATE SEQUENCE working_batch_seq".as_slice()),
        ),
        (
            4,
            std::sync::Arc::from(b"SELECT nextval('working_batch_seq'::regclass)".as_slice()),
        ),
    ];
    if let Err(failure) = e.commit_mutation_batch(&batch) {
        panic!("working-catalog batch failed: {}", failure.error);
    }

    assert_eq!(e.visible_up_to(), 4);
    let rows = e
        .execute_relational_select_text("SELECT value FROM working_batch WHERE id = 1")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0)[0], SqlValue::Int4(10));
    let sequence = e.relational_catalog_sequence("working_batch_seq").unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (1, true));

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered
        .relational_catalog_table("working_batch")
        .is_some());
    let rows = recovered
        .execute_relational_select_text("SELECT value FROM working_batch WHERE id = 1")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0)[0], SqlValue::Int4(10));
    let sequence = recovered
        .relational_catalog_sequence("working_batch_seq")
        .unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (1, true));

    let reset_and_rewrite = [
        (
            5,
            std::sync::Arc::from(b"TRUNCATE TABLE working_batch".as_slice()),
        ),
        (
            6,
            std::sync::Arc::from(
                b"INSERT INTO working_batch (id, value) VALUES (2, 20)".as_slice(),
            ),
        ),
        (
            7,
            std::sync::Arc::from(
                b"ALTER TABLE working_batch ADD COLUMN extra INT DEFAULT 7".as_slice(),
            ),
        ),
        (
            8,
            std::sync::Arc::from(
                b"INSERT INTO working_batch (id, value, extra) VALUES (3, 30, 8)".as_slice(),
            ),
        ),
    ];
    if let Err(failure) = e.commit_mutation_batch(&reset_and_rewrite) {
        panic!(
            "working-catalog reset/rewrite batch failed: {}",
            failure.error
        );
    }
    let rows = e
        .execute_relational_select_text("SELECT id, value, extra FROM working_batch ORDER BY id")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 2, "mixed reset/rewrite rows: {rows:?}");
    assert_eq!(
        rows.row(0),
        &[SqlValue::Int4(2), SqlValue::Int4(20), SqlValue::Int4(7)]
    );
    assert_eq!(
        rows.row(1),
        &[SqlValue::Int4(3), SqlValue::Int4(30), SqlValue::Int4(8)]
    );

    let dependencies = [
        (
            9,
            std::sync::Arc::from(b"CREATE ROLE working_batch_reader".as_slice()),
        ),
        (
            10,
            std::sync::Arc::from(
                b"GRANT SELECT ON TABLE working_batch TO working_batch_reader".as_slice(),
            ),
        ),
        (
            11,
            std::sync::Arc::from(
                b"REVOKE SELECT ON TABLE working_batch FROM working_batch_reader".as_slice(),
            ),
        ),
        (
            12,
            std::sync::Arc::from(b"DROP ROLE working_batch_reader".as_slice()),
        ),
    ];
    if let Err(failure) = e.commit_mutation_batch(&dependencies) {
        panic!("working-catalog dependency batch failed: {}", failure.error);
    }
    assert!(e.relational_role("working_batch_reader").is_none());

    let lifecycle = [
        (
            13,
            std::sync::Arc::from(b"CREATE TABLE transient_batch (id INT)".as_slice()),
        ),
        (
            14,
            std::sync::Arc::from(b"DROP TABLE transient_batch".as_slice()),
        ),
        (
            15,
            std::sync::Arc::from(b"CREATE TABLE transient_batch (name TEXT)".as_slice()),
        ),
    ];
    if let Err(failure) = e.commit_mutation_batch(&lifecycle) {
        panic!("create/drop/recreate batch failed: {}", failure.error);
    }
    let table = e.relational_catalog_table("transient_batch").unwrap();
    assert_eq!(table.columns.len(), 1);
    assert_eq!(table.columns[0].name, "name");

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_role("working_batch_reader").is_none());
    let table = recovered
        .relational_catalog_table("transient_batch")
        .unwrap();
    assert_eq!(table.columns.len(), 1);
    assert_eq!(table.columns[0].name, "name");
    let rows = recovered
        .execute_relational_select_text("SELECT id, value, extra FROM working_batch ORDER BY id")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows.row(0),
        &[SqlValue::Int4(2), SqlValue::Int4(20), SqlValue::Int4(7)]
    );
    assert_eq!(
        rows.row(1),
        &[SqlValue::Int4(3), SqlValue::Int4(30), SqlValue::Int4(8)]
    );
}

#[test]
fn batch_commit_timestamps_stay_strictly_monotonic_per_item() {
    // The batched commit inlines the next_commit_timestamp_micros formula per item; the
    // recorded commit timestamps must stay strictly monotonic in batch order (PITR ordering).
    let mut e = Engine::with_batching(8, Duration::from_secs(60));
    let t0 = Instant::now();
    e.enqueue_set_text(11, "SET a=1", t0).unwrap();
    e.enqueue_set_text(12, "SET b=2", t0).unwrap();
    e.enqueue_set_text(13, "SET c=3", t0).unwrap();
    e.flush_admin().unwrap();

    let timestamps = e.durable_wal_record_timestamps();
    assert_eq!(
        timestamps.iter().map(|t| t.txn_id).collect::<Vec<_>>(),
        vec![11, 12, 13]
    );
    assert!(
        timestamps
            .windows(2)
            .all(|w| w[0].timestamp_micros < w[1].timestamp_micros),
        "batch commit timestamps must be strictly monotonic: {timestamps:?}"
    );
}

#[test]
fn batch_flush_wal_failure_requeues_items_for_retry() {
    let mut e = Engine::with_batching(2, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.simulate_next_wal_flush_failure();
    let err = e
        .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
        .unwrap_err();

    assert!(matches!(
        err,
        ExecuteError::Engine(EngineError::Durability(_))
    ));
    assert_eq!(e.pending_batch_len(), 2);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().snapshot().batch_wait_samples, 0);
    assert_eq!(e.metrics().snapshot().pending_batch_peak, 2);
    assert_eq!(e.metrics().last_pending_batch_len(), Some(2));
    assert_eq!(e.metrics().snapshot().commits_total, 0);

    e.flush_admin().unwrap();
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.get("a").as_deref(), Some("1"));
    assert_eq!(e.get("b").as_deref(), Some("2"));
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn enqueue_rejects_new_mutation_when_retry_queue_is_saturated() {
    let mut e = Engine::with_batching(2, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.simulate_next_wal_flush_failure();
    let flush_err = e
        .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
        .unwrap_err();
    assert!(matches!(
        flush_err,
        ExecuteError::Engine(EngineError::Durability(_))
    ));
    assert_eq!(e.pending_batch_len(), 2);

    let saturated_err = e
        .enqueue_set_text(3, "SET c=3", t0 + Duration::from_millis(2))
        .unwrap_err();
    assert!(matches!(
        saturated_err,
        ExecuteError::Engine(EngineError::MutationQueueOverloaded { pending: 2, cap: 2 })
    ));
    assert_eq!(e.pending_batch_len(), 2);
    assert_eq!(
        e.metrics().fallback_for(FallbackReason::GpuQueueSaturated),
        1
    );
    assert_eq!(e.metrics().snapshot().commits_total, 0);

    let snapshot = e.telemetry_snapshot();
    assert!(snapshot.has_gpu_parity_fallbacks());
    assert_eq!(snapshot.gpu_parity_fallback_total(), 1);
    assert_eq!(
        snapshot.gpu_parity_fallbacks.get(&GpuParityIssue {
            id: "GPU-121",
            owner: "runtime",
            milestone: "m0-bootstrap",
        }),
        Some(&1)
    );
}

#[test]
fn failed_time_flush_does_not_drop_pending_queue() {
    let mut e = Engine::with_batching(10, Duration::from_millis(2));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.become_follower(2);

    let err = e.tick_batching(t0 + Duration::from_millis(3)).unwrap_err();

    assert!(matches!(err, EngineError::NotLeader));
    assert_eq!(e.pending_batch_len(), 1);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn follower_tick_without_pending_batch_is_noop() {
    let mut e = Engine::with_batching(10, Duration::from_millis(2));
    e.become_follower(2);

    e.tick_batching(Instant::now()).unwrap();

    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn pending_batch_can_be_flushed_after_follower_is_promoted_back_to_leader() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.become_follower(2);

    let tick_err = e.tick_batching(t0 + Duration::from_secs(1)).unwrap_err();
    assert!(matches!(tick_err, EngineError::NotLeader));
    assert_eq!(e.pending_batch_len(), 1);
    assert_eq!(e.get("a"), None);

    e.become_leader(3);
    e.flush_admin().unwrap();

    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.get("a").as_deref(), Some("1"));
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
    assert_eq!(e.metrics().snapshot().commits_total, 1);
}

#[test]
fn execute_text_non_mutations_count_as_not_gpu_eligible_fallbacks() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(1, "BEGIN").unwrap();
    e.execute_text(1, "COMMIT").unwrap();
    e.execute_text(2, "BEGIN").unwrap();
    e.execute_text(2, "ROLLBACK").unwrap();
    e.execute_text(3, "GET missing").unwrap();
    e.execute_text(4, "FLUSH").unwrap();
    e.execute_text(5, "RESET ALL").unwrap();
    e.execute_text(6, "DISCARD TEMP").unwrap();
    e.execute_text(
        7,
        "CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog",
    )
    .unwrap();

    assert_eq!(e.active_txn_count(), 0);
    assert_eq!(e.metrics().snapshot().fallback_total, 9);
    assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 9);
    assert_eq!(
        e.metrics().last_fallback_reason(),
        Some(FallbackReason::NotGpuEligible)
    );
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn execute_text_bounds_bootstrap_extension_create() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(1, "CREATE EXTENSION IF NOT EXISTS plpgsql")
        .unwrap();
    e.execute_text(
        2,
        "CREATE EXTENSION IF NOT EXISTS \"plpgsql\" WITH SCHEMA pg_catalog",
    )
    .unwrap();

    let duplicate = e.execute_text(3, "CREATE EXTENSION plpgsql").unwrap_err();
    assert!(matches!(
        duplicate,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "extension \"plpgsql\" already exists"
    ));

    let unsupported = e
        .execute_text(4, "CREATE EXTENSION IF NOT EXISTS hstore")
        .unwrap_err();
    assert!(matches!(
        unsupported,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "only the bootstrap plpgsql extension is supported"
    ));

    let wrong_schema = e
        .execute_text(
            5,
            "CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA public",
        )
        .unwrap_err();
    assert!(matches!(
        wrong_schema,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "plpgsql extension creation is only supported in pg_catalog"
    ));
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn execute_text_accepts_bootstrap_extension_if_exists_cleanup() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(1, "COMMENT ON EXTENSION plpgsql IS 'bootstrap extension'")
        .unwrap();
    e.execute_text(2, "DROP EXTENSION IF EXISTS plpgsql")
        .unwrap();
    assert_eq!(
        e.relational_extension_comment("plpgsql").as_deref(),
        Some("bootstrap extension")
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered.relational_extension_comment("plpgsql").as_deref(),
        Some("bootstrap extension")
    );

    let bootstrap_drop = e.execute_text(3, "DROP EXTENSION plpgsql").unwrap_err();
    assert!(matches!(
        bootstrap_drop,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "cannot drop bootstrap extension \"plpgsql\""
    ));

    let unsupported = e
        .execute_text(4, "DROP EXTENSION IF EXISTS hstore")
        .unwrap_err();
    assert!(matches!(
        unsupported,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "extension \"hstore\" does not exist"
    ));
    assert_eq!(e.metrics().snapshot().commits_total, 1);
}

#[test]
fn execute_text_records_bootstrap_extension_comment_and_replays_from_wal() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(1, "COMMENT ON EXTENSION plpgsql IS 'bootstrap extension'")
        .unwrap();
    assert_eq!(
        e.relational_extension_comment("plpgsql").as_deref(),
        Some("bootstrap extension")
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered.relational_extension_comment("plpgsql").as_deref(),
        Some("bootstrap extension")
    );

    e.execute_text(2, "COMMENT ON EXTENSION plpgsql IS NULL")
        .unwrap();
    assert_eq!(e.relational_extension_comment("plpgsql"), None);

    let missing = e
        .execute_text(3, "COMMENT ON EXTENSION hstore IS 'missing'")
        .unwrap_err();
    assert!(matches!(
        missing,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "extension \"hstore\" does not exist"
    ));
}

#[test]
fn execute_text_replays_bounded_role_metadata_and_acl_grantees() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(1, "CREATE ROLE app_reader WITH LOGIN")
        .unwrap();
    e.execute_text(2, "CREATE USER app_writer").unwrap();
    e.execute_text(3, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(4, "GRANT SELECT ON TABLE people TO app_reader")
        .unwrap();
    e.execute_text(
        5,
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO app_writer",
    )
    .unwrap();
    e.execute_text(6, "COMMENT ON ROLE app_reader IS 'read-only app'")
        .unwrap();

    assert!(e.relational_role("app_reader").unwrap().login);
    assert!(e.relational_role("app_writer").unwrap().login);
    assert_eq!(
        e.relational_role_comment("app_reader").as_deref(),
        Some("read-only app")
    );
    assert!(e
        .relational_catalog_table("people")
        .unwrap()
        .acl
        .contains_key("app_reader"));
    assert!(e
        .ddl_catalog()
        .relational_default_table_acl
        .contains_key("app_writer"));

    e.execute_text(7, "ALTER ROLE app_reader RENAME TO app_analyst")
        .unwrap();
    assert!(e.relational_role("app_reader").is_none());
    assert!(e.relational_role("app_analyst").unwrap().login);
    assert_eq!(
        e.relational_role_comment("app_analyst").as_deref(),
        Some("read-only app")
    );
    assert!(e
        .relational_catalog_table("people")
        .unwrap()
        .acl
        .contains_key("app_analyst"));
    assert!(!e
        .relational_catalog_table("people")
        .unwrap()
        .acl
        .contains_key("app_reader"));

    let dependent_drop = e.execute_text(8, "DROP ROLE app_analyst").unwrap_err();
    assert!(dependent_drop
        .to_string()
        .contains("dependent metadata exists"));

    e.execute_text(9, "REVOKE SELECT ON TABLE people FROM app_analyst")
        .unwrap();
    e.execute_text(10, "COMMENT ON ROLE app_analyst IS NULL")
        .unwrap();
    e.execute_text(11, "DROP ROLE app_analyst").unwrap();
    e.execute_text(12, "DROP USER IF EXISTS app_missing")
        .unwrap();

    assert!(e.relational_role("app_analyst").is_none());
    assert!(e.relational_role("app_writer").is_some());

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_role("app_analyst").is_none());
    assert!(recovered.relational_role("app_writer").unwrap().login);
    assert!(recovered
        .ddl_catalog()
        .relational_default_table_acl
        .contains_key("app_writer"));

    let missing_grantee = Engine::new_local_cpu_oracle()
        .execute_text(1, "GRANT SELECT ON TABLE people TO missing_role")
        .unwrap_err();
    assert!(missing_grantee
        .to_string()
        .contains("relation \"people\" does not exist"));

    let missing_role = Engine::new_local_cpu_oracle();
    missing_role
        .execute_text(1, "CREATE TABLE people (id INT)")
        .unwrap();
    assert!(missing_role
        .execute_text(2, "GRANT SELECT ON TABLE people TO missing_role")
        .unwrap_err()
        .to_string()
        .contains("role \"missing_role\" does not exist"));
    assert!(missing_role
        .execute_text(3, "DROP ROLE postgres")
        .unwrap_err()
        .to_string()
        .contains("cannot drop bootstrap role"));
    assert!(missing_role
        .execute_text(4, "CREATE ROLE app_password PASSWORD 'secret'")
        .is_err());
    assert!(missing_role
        .execute_text(5, "ALTER ROLE postgres RENAME TO root")
        .unwrap_err()
        .to_string()
        .contains("cannot rename bootstrap role"));
    assert!(missing_role
        .execute_text(6, "ALTER ROLE missing_role RENAME TO renamed_role")
        .unwrap_err()
        .to_string()
        .contains("role \"missing_role\" does not exist"));
}

#[test]
fn execute_text_replays_bounded_database_metadata() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(1, "CREATE DATABASE appdb").unwrap();
    e.execute_text(2, "COMMENT ON DATABASE appdb IS 'application database'")
        .unwrap();
    e.execute_text(3, "CREATE ROLE app_reader").unwrap();
    e.execute_text(
        4,
        "GRANT CONNECT, TEMPORARY ON DATABASE appdb TO app_reader",
    )
    .unwrap();

    let appdb = e.relational_database("appdb").unwrap();
    assert_eq!(appdb.name, "appdb");
    let oid = appdb.oid;
    assert_eq!(
        e.relational_database_acl("appdb")
            .unwrap()
            .get("app_reader")
            .unwrap(),
        &BTreeSet::from([DatabasePrivilege::Connect, DatabasePrivilege::Temporary])
    );
    assert_eq!(
        e.relational_database_comment("appdb").as_deref(),
        Some("application database")
    );

    e.execute_text(5, "ALTER ROLE app_reader RENAME TO app_analyst")
        .unwrap();
    assert!(e
        .relational_database_acl("appdb")
        .unwrap()
        .contains_key("app_analyst"));
    e.execute_text(6, "REVOKE TEMP ON DATABASE appdb FROM app_analyst")
        .unwrap();
    assert_eq!(
        e.relational_database_acl("appdb")
            .unwrap()
            .get("app_analyst")
            .unwrap(),
        &BTreeSet::from([DatabasePrivilege::Connect])
    );

    e.execute_text(7, "ALTER DATABASE appdb RENAME TO appdb_renamed")
        .unwrap();
    let renamed = e.relational_database("appdb_renamed").unwrap();
    assert_eq!(renamed.oid, oid);
    assert_eq!(renamed.name, "appdb_renamed");
    assert!(e
        .relational_database_acl("appdb_renamed")
        .unwrap()
        .contains_key("app_analyst"));
    assert_eq!(
        e.relational_database_comment("appdb_renamed").as_deref(),
        Some("application database")
    );
    assert_eq!(e.relational_database_comment("appdb"), None);

    let duplicate = e
        .execute_text(8, "CREATE DATABASE appdb_renamed")
        .unwrap_err();
    assert!(duplicate
        .to_string()
        .contains("database \"appdb_renamed\" already exists"));
    let duplicate_rename = e
        .execute_text(9, "CREATE DATABASE appdb")
        .and_then(|_| e.execute_text(10, "ALTER DATABASE appdb_renamed RENAME TO appdb"))
        .unwrap_err();
    assert!(duplicate_rename
        .to_string()
        .contains("database \"appdb\" already exists"));

    e.execute_text(11, "DROP DATABASE appdb_renamed").unwrap();
    assert!(e.relational_database("appdb_renamed").is_none());
    assert_eq!(e.relational_database_comment("appdb_renamed"), None);
    e.execute_text(12, "DROP DATABASE IF EXISTS missing_db")
        .unwrap();

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_database("appdb_renamed").is_none());
    assert_eq!(recovered.relational_database_comment("appdb_renamed"), None);

    let kept = Engine::new_local_cpu_oracle();
    kept.execute_text(1, "CREATE DATABASE appdb").unwrap();
    let recovered_kept = Engine::recover_from_durable_wal(&kept.durable_wal_records()).unwrap();
    assert!(recovered_kept.relational_database("appdb").is_some());

    assert!(Engine::new_local_cpu_oracle()
        .execute_text(1, "DROP DATABASE postgres")
        .unwrap_err()
        .to_string()
        .contains("cannot drop bootstrap database"));
    assert!(Engine::new_local_cpu_oracle()
        .execute_text(1, "ALTER DATABASE postgres RENAME TO appdb")
        .unwrap_err()
        .to_string()
        .contains("cannot rename bootstrap database"));
    assert!(Engine::new_local_cpu_oracle()
        .execute_text(1, "ALTER DATABASE missing_db RENAME TO appdb")
        .unwrap_err()
        .to_string()
        .contains("database \"missing_db\" does not exist"));
    assert!(Engine::new_local_cpu_oracle()
        .execute_text(1, "CREATE DATABASE templated TEMPLATE template1")
        .is_err());
    assert!(Engine::new_local_cpu_oracle()
        .execute_text(1, "GRANT CONNECT ON DATABASE missing_db TO PUBLIC")
        .unwrap_err()
        .to_string()
        .contains("database \"missing_db\" does not exist"));
}

#[test]
fn enqueue_non_mutations_count_as_not_gpu_eligible_fallbacks() {
    let mut e = Engine::with_batching(2, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "BEGIN", t0).unwrap();
    e.enqueue_set_text(1, "COMMIT", t0).unwrap();
    e.enqueue_set_text(2, "BEGIN", t0).unwrap();
    e.enqueue_set_text(2, "ROLLBACK", t0).unwrap();
    e.enqueue_set_text(3, "GET missing", t0).unwrap();
    e.enqueue_set_text(4, "FLUSH", t0).unwrap();
    e.enqueue_set_text(5, "RESET ALL", t0).unwrap();
    e.enqueue_set_text(6, "DISCARD TEMP", t0).unwrap();

    assert_eq!(e.active_txn_count(), 0);
    assert_eq!(e.metrics().snapshot().fallback_total, 8);
    assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 8);
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn commit_and_rollback_require_active_transaction_context() {
    let e = Engine::new_local_cpu_oracle();

    let commit_err = e.execute_text(10, "COMMIT").unwrap_err();
    assert!(matches!(
        commit_err,
        ExecuteError::Txn(TxnError::NotFound(10))
    ));

    let rollback_err = e.execute_text(11, "ROLLBACK").unwrap_err();
    assert!(matches!(
        rollback_err,
        ExecuteError::Txn(TxnError::NotFound(11))
    ));

    e.execute_text(12, "BEGIN").unwrap();
    let duplicate_begin_err = e.execute_text(12, "BEGIN").unwrap_err();
    assert!(matches!(
        duplicate_begin_err,
        ExecuteError::Txn(TxnError::AlreadyExists(12))
    ));

    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(e.active_txn_count(), 1);
}

#[test]
fn active_engine_transaction_rejects_unsupported_autocommit_commands() {
    let e = Engine::new_local();
    e.execute_text(41, "BEGIN").unwrap();

    let err = e
        .execute_text(41, "CREATE TABLE escaped_commit (id INT)")
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("not supported inside an active transaction"),
        "{err}"
    );
    assert!(
        e.relational_catalog_table("escaped_commit").is_none(),
        "unsupported transaction commands must not autocommit"
    );
    e.execute_text(41, "ROLLBACK").unwrap();
}

#[test]
fn enqueue_active_transaction_rejects_unsupported_autocommit_commands() {
    let mut e = Engine::new_local_cpu_oracle();
    let now = Instant::now();
    e.enqueue_set_text(41, "BEGIN", now).unwrap();

    let err = e
        .enqueue_set_text(41, "CREATE TABLE escaped_enqueue (id INT)", now)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("not supported inside an active transaction"),
        "{err}"
    );
    assert!(e.relational_catalog_table("escaped_enqueue").is_none());
    assert!(e.transaction_snapshot_handle(41).is_some());
    e.enqueue_set_text(41, "ROLLBACK", now).unwrap();
    assert!(e.transaction_snapshot_handle(41).is_none());
}

#[test]
fn explicit_transaction_snapshot_lives_from_begin_through_terminal_control() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(1, "SET acct:1=open").unwrap();
    assert_eq!(e.committed_seq(), 1);
    e.execute_text(90, "BEGIN").unwrap();
    {
        let active = e
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(active.transaction_snapshot(90), Some(1));
        assert_eq!(active.oldest(), Some(1));
    }

    // A later autocommit advances the visible boundary, but the transaction keeps exactly the
    // snapshot captured by BEGIN and therefore keeps the GC/ledger floor at commit sequence 1.
    e.execute_text(2, "SET acct:1=closed").unwrap();
    assert_eq!(e.committed_seq(), 2);
    {
        let active = e
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(active.transaction_snapshot(90), Some(1));
        assert_eq!(active.oldest(), Some(1));
    }
    let vacuum_err = e.checkpoint_vacuum_mvcc_versions(1).unwrap_err();
    assert!(
        vacuum_err
            .to_string()
            .contains("crosses active read snapshot 1"),
        "got: {vacuum_err}"
    );

    e.execute_text(90, "ROLLBACK").unwrap();
    let active = e
        .active_snapshots
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(active.transaction_snapshot(90), None);
    assert_eq!(active.oldest(), None);
}

#[test]
fn explicit_transaction_select_reads_captured_catalog_and_table_generation() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    e.execute_text(90, "BEGIN").unwrap();

    // The table had no data cell at BEGIN. A later first write publishes one, but the transaction's
    // generation bundle deliberately records the table as empty rather than loading that newer cell
    // on first touch.
    e.execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1, 200)")
        .unwrap();
    let select = match parse_command("SELECT balance FROM accounts WHERE id = 1").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let old = e
        .execute_relational_select_in_transaction(90, &select)
        .unwrap();
    assert!(
        old.rows.is_empty(),
        "a first touch must not load a post-BEGIN table generation"
    );

    let current = e.execute_relational_select(&select).unwrap();
    assert_eq!(current.rows.row(0)[0], SqlValue::Int4(200));
    e.execute_text(90, "ROLLBACK").unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn explicit_transaction_dml_prepare_and_conflict_check_use_begin_generation() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
        .unwrap();
    e.execute_text(90, "BEGIN").unwrap();

    // A later writer replaces the row after BEGIN and publishes a newer device identity/version
    // stamp.
    e.execute_dml_concurrent(3, "UPDATE accounts SET balance = 200 WHERE id = 1")
        .unwrap();

    // The retained generation still contains balance=100. Resolving this predicate against current
    // rows would find zero targets. Successful staging and private read-your-writes therefore prove
    // predicate preparation used BEGIN's generation; COMMIT must reject the stale device stamp.
    e.execute_dml_concurrent(
        90,
        "UPDATE accounts SET balance = 300 WHERE id = 1 AND balance = 100",
    )
    .unwrap();

    let select = match parse_command("SELECT balance FROM accounts WHERE id = 1").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let old = e
        .execute_relational_select_in_transaction(90, &select)
        .unwrap();
    assert_eq!(old.rows.row(0)[0], SqlValue::Int4(300));
    let current = e.execute_relational_select(&select).unwrap();
    assert_eq!(current.rows.row(0)[0], SqlValue::Int4(200));

    let err = e.execute_text(90, "COMMIT").unwrap_err();
    assert!(
        matches!(&err, ExecuteError::Serialization(message) if message.contains("device write-write conflict")),
        "expected current-generation device write conflict, got {err:?}"
    );
    e.execute_text(90, "ROLLBACK").unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn explicit_transaction_unique_validation_uses_current_device_generation() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE accounts (id INT PRIMARY KEY, tenant_key INT UNIQUE)",
    )
    .unwrap();
    e.execute_text(2, "INSERT INTO accounts (id, tenant_key) VALUES (1, 10)")
        .unwrap();
    e.execute_text(90, "BEGIN").unwrap();

    e.execute_dml_concurrent(3, "INSERT INTO accounts (id, tenant_key) VALUES (2, 20)")
        .unwrap();
    e.execute_dml_concurrent(90, "INSERT INTO accounts (id, tenant_key) VALUES (3, 20)")
        .unwrap();
    let err = e.execute_text(90, "COMMIT").unwrap_err();
    assert!(
        matches!(&err, ExecuteError::Serialization(message) if message.contains("device unique conflict")),
        "BEGIN-generation validation must not see the later row, while COMMIT must arbitrate the \
         final unique value against the current device generation; got {err:?}"
    );

    let old = match parse_command("SELECT id FROM accounts WHERE tenant_key = 20").unwrap() {
        Command::Select(select) => e
            .execute_relational_select_in_transaction(90, &select)
            .unwrap(),
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert_eq!(old.rows.row(0)[0], SqlValue::Int4(3));
    e.execute_text(90, "ROLLBACK").unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn explicit_transaction_unique_key_away_history_uses_device_stamps() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_device_write_locate_wave_batch_enabled(true);
    e.execute_text(
        1,
        "CREATE TABLE history_accounts (id INT PRIMARY KEY, tenant_key INT UNIQUE)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO history_accounts (id, tenant_key) VALUES (1, 10)",
    )
    .unwrap();
    for i in 0..512_i32 {
        e.execute_dml_concurrent(
            10_000 + i as u64,
            &format!(
                "INSERT INTO history_accounts (id, tenant_key) VALUES ({}, {})",
                1000 + i,
                1000 + i
            ),
        )
        .unwrap();
        if e.table_device_authoritative("history_accounts") {
            break;
        }
    }
    assert!(e.table_device_authoritative("history_accounts"));
    e.execute_text(190, "BEGIN").unwrap();
    e.execute_dml_concurrent(
        190,
        "INSERT INTO history_accounts (id, tenant_key) VALUES (3, 30)",
    )
    .unwrap();

    e.execute_dml_concurrent(
        191,
        "INSERT INTO history_accounts (id, tenant_key) VALUES (4, 30)",
    )
    .unwrap();
    e.execute_dml_concurrent(192, "DELETE FROM history_accounts WHERE id = 4")
        .unwrap();

    // Force the dense-rebuild path while the stale writer is still registered. VACUUM must keep
    // the churn/history generation intact; resetting the counter proves a rebuild actually ran.
    // Sabotage control: removing the oldest-active fence resets this to zero and the stale COMMIT
    // below loses the key-away stamp.
    let churn_before_vacuum = e.tombstone_churn("history_accounts");
    assert!(churn_before_vacuum > 0, "external DELETE created history");
    e.vacuum_table("history_accounts").unwrap();
    assert_eq!(
        e.tombstone_churn("history_accounts"),
        churn_before_vacuum,
        "an older active writer must defer current-only dense VACUUM"
    );

    // Ordinary admission is allowed to rebuild the current live image while an old writer is
    // active, but that rebuilt generation must carry a history floor. Force the exact
    // claim+release-loss counterexample: the rebuild drops row 4's deleted physical version, so
    // COMMIT must decline the now-incomplete device miss rather than interpret it as no conflict.
    let tables = std::iter::once("history_accounts".to_string()).collect();
    e.auto_admit_resident_tables(&tables);
    assert_eq!(
        e.tombstone_churn("history_accounts"),
        0,
        "non-vacuity: ordinary admission performed the current-only rebuild"
    );

    let err = e.execute_text(190, "COMMIT").unwrap_err();
    assert!(
        matches!(&err, ExecuteError::Serialization(message)
            if message.contains("device unique conflict")),
        "a current-only rebuild newer than BEGIN must make incomplete device history fail closed, got {err:?}"
    );
    e.execute_text(190, "ROLLBACK").unwrap();
    let rows = e
        .execute_relational_select_text(
            "SELECT id, tenant_key FROM history_accounts WHERE id >= 3 AND id <= 4 ORDER BY id",
        )
        .unwrap()
        .rows;
    assert!(rows.is_empty());
}

#[test]
fn and_chain_forms_reopen_transaction_context() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(21, "BEGIN").unwrap();
    e.execute_text(21, "COMMIT AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    {
        let active = e
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(active.transaction_snapshot(21), None);
        assert_eq!(active.transaction_snapshot(22), Some(0));
    }
    e.execute_text(22, "COMMIT").unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.execute_text(31, "BEGIN").unwrap();
    e.execute_text(31, "ROLLBACK AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(32, "ROLLBACK").unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn enqueue_non_mutation_chain_forms_reopen_transaction_context() {
    let mut e = Engine::new_local_cpu_oracle();
    let t0 = Instant::now();

    e.enqueue_set_text(41, "BEGIN", t0).unwrap();
    e.enqueue_set_text(41, "COMMIT AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(42, "COMMIT", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.enqueue_set_text(51, "BEGIN", t0).unwrap();
    e.enqueue_set_text(51, "ROLLBACK AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(52, "ROLLBACK", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn transaction_control_alias_chain_forms_reopen_transaction_context() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(61, "BEGIN").unwrap();
    e.execute_text(61, "END AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(62, "END").unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.execute_text(71, "BEGIN").unwrap();
    e.execute_text(71, "ABORT AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(72, "ABORT").unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.execute_text(75, "BEGIN").unwrap();
    e.execute_text(75, "END WORK AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(76, "COMMIT").unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.execute_text(77, "BEGIN").unwrap();
    e.execute_text(77, "ABORT WORK AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(78, "ROLLBACK").unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn start_alias_and_work_aliases_drive_transaction_state_transitions() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(73, "START TRANSACTION READ ONLY").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(73, "COMMIT WORK").unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.execute_text(74, "START WORK, READ WRITE, DEFERRABLE")
        .unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(74, "ROLLBACK TRANSACTION").unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn enqueue_transaction_control_alias_chain_forms_reopen_transaction_context() {
    let mut e = Engine::new_local_cpu_oracle();
    let t0 = Instant::now();

    e.enqueue_set_text(81, "BEGIN", t0).unwrap();
    e.enqueue_set_text(81, "END AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(82, "END", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.enqueue_set_text(91, "BEGIN", t0).unwrap();
    e.enqueue_set_text(91, "ABORT AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(92, "ABORT", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.enqueue_set_text(95, "BEGIN", t0).unwrap();
    e.enqueue_set_text(95, "END WORK AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(96, "COMMIT", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.enqueue_set_text(97, "BEGIN", t0).unwrap();
    e.enqueue_set_text(97, "ABORT WORK AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(98, "ROLLBACK", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn enqueue_start_alias_and_work_aliases_drive_transaction_state_transitions() {
    let mut e = Engine::new_local_cpu_oracle();
    let t0 = Instant::now();

    e.enqueue_set_text(93, "START TRANSACTION READ ONLY", t0)
        .unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(93, "COMMIT WORK", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.enqueue_set_text(94, "START WORK, READ WRITE, DEFERRABLE", t0)
        .unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(94, "ROLLBACK TRANSACTION", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn commit_and_chain_propagates_txn_id_exhaustion() {
    let e = Engine::new_local_cpu_oracle();

    e.execute_text(u64::MAX, "BEGIN").unwrap();
    let err = e.execute_text(u64::MAX, "COMMIT AND CHAIN").unwrap_err();

    assert!(matches!(err, ExecuteError::Txn(TxnError::IdExhausted)));
    assert_eq!(e.active_txn_count(), 0);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}
