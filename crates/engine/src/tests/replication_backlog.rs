use super::*;

#[test]
fn replication_watermarks_track_commit_apply_visibility_and_durability() {
    let e = Engine::new_local_test_engine();

    let before = e.replication_watermarks();
    assert_eq!(before.role, Role::Leader);
    assert_eq!(before.commit_index, 0);
    assert_eq!(before.applied_index, 0);
    assert_eq!(before.visible_index, 0);
    assert_eq!(before.commit_apply_gap, 0);
    assert_eq!(before.apply_visible_gap, 0);
    assert_eq!(before.snapshot_id, 0);
    assert_eq!(before.wal_flushed_count, 0);
    assert_eq!(before.wal_buffered_count, 0);
    assert_eq!(before.wal_unflushed_count, 0);
    assert_eq!(before.pending_batch_len, 0);
    assert_eq!(before.pending_batch_cap, 64);
    assert_eq!(before.pending_batch_remaining_capacity, 64);
    assert_eq!(before.pending_batch_utilization_permyriad, 0);
    assert_eq!(before.pending_batch_remaining_capacity_permyriad, 10_000);
    assert_eq!(before.pending_batch_oldest_age_ms, None);
    assert_eq!(before.pending_batch_time_until_deadline_ms, None);
    assert_eq!(before.active_txn_count, 0);
    assert_eq!(before.oldest_active_txn_id, None);
    assert_eq!(before.newest_active_txn_id, None);
    assert!(!before.has_wal_backlog);
    assert!(!before.has_pending_batch_backlog);
    assert!(!before.has_active_txn_backlog);
    assert!(!before.has_commit_apply_gap);
    assert!(!before.has_apply_visible_gap);
    assert!(!before.has_backlog_blockers);
    assert_eq!(before.backlog_blocker_count, 0);
    assert_eq!(before.backlog_blocker_mask, 0);
    assert!(!before.mutation_admission_saturated);
    assert!(before.quiescent_for_failover);
    assert!(!before.follower_promotion_ready);

    let token = e.commit_mutation(1, b"SET a=1".to_vec().into()).unwrap();
    let after = e.replication_watermarks();

    assert_eq!(after.role, Role::Leader);
    assert!(after.term >= before.term);
    assert_eq!(after.commit_index, token.index);
    assert_eq!(after.applied_index, token.index);
    assert_eq!(after.visible_index, token.index);
    assert_eq!(after.commit_apply_gap, 0);
    assert_eq!(after.apply_visible_gap, 0);
    assert_eq!(after.snapshot_id, 0);
    assert!(after.wal_flushed_count >= 1);
    assert_eq!(after.wal_buffered_count, e.wal_buffered_count());
    assert_eq!(after.wal_unflushed_count, e.wal_unflushed_count());
    assert_eq!(after.pending_batch_len, 0);
    assert_eq!(after.pending_batch_cap, 64);
    assert_eq!(after.pending_batch_remaining_capacity, 64);
    assert_eq!(after.pending_batch_utilization_permyriad, 0);
    assert_eq!(after.pending_batch_remaining_capacity_permyriad, 10_000);
    assert_eq!(after.pending_batch_oldest_age_ms, None);
    assert_eq!(after.pending_batch_time_until_deadline_ms, None);
    assert_eq!(after.active_txn_count, 0);
    assert_eq!(after.oldest_active_txn_id, None);
    assert_eq!(after.newest_active_txn_id, None);
    assert!(!after.has_wal_backlog);
    assert!(!after.has_pending_batch_backlog);
    assert!(!after.has_active_txn_backlog);
    assert!(!after.has_commit_apply_gap);
    assert!(!after.has_apply_visible_gap);
    assert!(!after.has_backlog_blockers);
    assert_eq!(after.backlog_blocker_count, 0);
    assert_eq!(after.backlog_blocker_mask, 0);
    assert!(!after.mutation_admission_saturated);
    assert!(after.quiescent_for_failover);
    assert!(!after.follower_promotion_ready);
}

#[test]
fn replication_watermarks_do_not_advance_on_rejected_follower_commit() {
    let mut e = Engine::new_local_test_engine();
    e.become_follower(2);

    let err = e
        .commit_mutation(1, b"SET a=1".to_vec().into())
        .unwrap_err();
    assert!(matches!(err, EngineError::NotLeader));

    let marks = e.replication_watermarks();
    assert_eq!(marks.role, Role::Follower);
    assert_eq!(marks.term, 2);
    assert_eq!(marks.commit_index, 0);
    assert_eq!(marks.applied_index, 0);
    assert_eq!(marks.visible_index, 0);
    assert_eq!(marks.commit_apply_gap, 0);
    assert_eq!(marks.apply_visible_gap, 0);
    assert_eq!(marks.wal_flushed_count, 0);
    assert_eq!(marks.wal_last_durable_txn_id, None);
    assert_eq!(marks.wal_buffered_count, 0);
    assert_eq!(marks.wal_unflushed_count, 0);
    assert_eq!(marks.pending_batch_len, 0);
    assert_eq!(marks.pending_batch_cap, 64);
    assert_eq!(marks.pending_batch_remaining_capacity, 64);
    assert_eq!(marks.pending_batch_utilization_permyriad, 0);
    assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 10_000);
    assert_eq!(marks.pending_batch_oldest_age_ms, None);
    assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
    assert_eq!(marks.active_txn_count, 0);
    assert!(!marks.has_wal_backlog);
    assert!(!marks.has_pending_batch_backlog);
    assert!(!marks.has_active_txn_backlog);
    assert!(!marks.has_commit_apply_gap);
    assert!(!marks.has_apply_visible_gap);
    assert!(!marks.has_backlog_blockers);
    assert_eq!(marks.backlog_blocker_mask, 0);
    assert!(!marks.mutation_admission_saturated);
    assert!(!marks.quiescent_for_failover);
    assert!(marks.follower_promotion_ready);
}

#[test]
fn replication_watermarks_include_buffered_wal_records() {
    let e = Engine::new_local_test_engine();

    e.commit_mutation(1, b"SET a=1".to_vec().into()).unwrap();
    e.commit_mutation(2, b"SET b=2".to_vec().into()).unwrap();

    let marks = e.replication_watermarks();
    assert_eq!(marks.wal_buffered_count, 2);
    assert_eq!(marks.wal_flushed_count, 2);
    assert_eq!(marks.wal_last_durable_txn_id, Some(2));
    assert_eq!(marks.wal_unflushed_count, 0);
    assert_eq!(marks.pending_batch_len, 0);
    assert_eq!(marks.pending_batch_cap, 64);
    assert_eq!(marks.pending_batch_remaining_capacity, 64);
    assert_eq!(marks.pending_batch_utilization_permyriad, 0);
    assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 10_000);
    assert_eq!(marks.pending_batch_oldest_age_ms, None);
    assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
    assert_eq!(marks.active_txn_count, 0);
    assert!(!marks.mutation_admission_saturated);
    assert!(marks.quiescent_for_failover);
}

#[test]
fn replication_watermarks_pending_batch_time_fields_clear_after_flush() {
    let mut e = Engine::with_batching(3, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    let before_flush = e.replication_watermarks();
    assert_eq!(before_flush.pending_batch_len, 1);
    assert_eq!(before_flush.pending_batch_cap, 3);
    assert_eq!(before_flush.pending_batch_remaining_capacity, 2);
    assert_eq!(before_flush.pending_batch_utilization_permyriad, 3_333);
    assert_eq!(
        before_flush.pending_batch_remaining_capacity_permyriad,
        6_667
    );
    assert!(before_flush.pending_batch_oldest_age_ms.is_some());
    assert!(before_flush.pending_batch_time_until_deadline_ms.is_some());

    e.flush_admin().unwrap();
    let after_flush = e.replication_watermarks();
    assert_eq!(after_flush.pending_batch_len, 0);
    assert_eq!(after_flush.pending_batch_cap, 3);
    assert_eq!(after_flush.pending_batch_remaining_capacity, 3);
    assert_eq!(after_flush.pending_batch_utilization_permyriad, 0);
    assert_eq!(
        after_flush.pending_batch_remaining_capacity_permyriad,
        10_000
    );
    assert_eq!(after_flush.pending_batch_oldest_age_ms, None);
    assert_eq!(after_flush.pending_batch_time_until_deadline_ms, None);
}

#[test]
fn replication_watermarks_include_pending_batch_depth() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();

    let marks = e.replication_watermarks();
    assert_eq!(marks.pending_batch_len, 1);
    assert_eq!(marks.pending_batch_cap, 2);
    assert_eq!(marks.pending_batch_remaining_capacity, 1);
    assert_eq!(marks.pending_batch_utilization_permyriad, 5_000);
    assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 5_000);
    assert!(marks.pending_batch_oldest_age_ms.is_some());
    assert!(marks.pending_batch_time_until_deadline_ms.is_some());
    assert!(marks.has_pending_batch_backlog);
    assert!(marks.has_backlog_blockers);
    assert_eq!(marks.backlog_blocker_count, 1);
    assert_eq!(
        marks.backlog_blocker_mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
    );
    assert!(!marks.has_wal_backlog);
    assert!(!marks.has_active_txn_backlog);
    assert!(!marks.has_commit_apply_gap);
    assert!(!marks.has_apply_visible_gap);
    assert_eq!(marks.wal_buffered_count, 0);
    assert_eq!(marks.wal_unflushed_count, 0);
    assert_eq!(marks.active_txn_count, 0);
    assert!(!marks.quiescent_for_failover);
}

#[test]
fn replication_watermarks_flag_mutation_admission_saturation() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
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

    let marks = e.replication_watermarks();
    assert_eq!(marks.pending_batch_len, 2);
    assert_eq!(marks.pending_batch_cap, 2);
    assert_eq!(marks.pending_batch_remaining_capacity, 0);
    assert_eq!(marks.pending_batch_utilization_permyriad, 10_000);
    assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 0);
    assert!(marks.has_pending_batch_backlog);
    assert!(!marks.has_wal_backlog);
    assert!(!marks.has_active_txn_backlog);
    assert!(marks.mutation_admission_saturated);
    assert!(!marks.quiescent_for_failover);
    assert!(!marks.follower_promotion_ready);
}

#[test]
fn replication_watermarks_follower_promotion_ready_requires_no_backlog() {
    let mut e = Engine::with_batching(8, Duration::from_secs(999));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.become_follower(3);

    let marks = e.replication_watermarks();
    assert_eq!(marks.role, Role::Follower);
    assert_eq!(marks.pending_batch_len, 1);
    assert!(!marks.quiescent_for_failover);
    assert!(!marks.follower_promotion_ready);
}

#[test]
fn replication_watermarks_follower_promotion_ready_requires_zero_active_txns() {
    let mut e = Engine::new_local_test_engine();
    e.become_follower(5);
    e.execute_text(9, "BEGIN").unwrap();

    let marks = e.replication_watermarks();
    assert_eq!(marks.role, Role::Follower);
    assert_eq!(marks.active_txn_count, 1);
    assert_eq!(marks.oldest_active_txn_id, Some(9));
    assert_eq!(marks.newest_active_txn_id, Some(9));
    assert!(!marks.follower_promotion_ready);
}

#[test]
fn replication_watermarks_include_active_transaction_count() {
    let e = Engine::new_local_test_engine();

    e.execute_text(42, "BEGIN").unwrap();

    let marks = e.replication_watermarks();
    assert_eq!(marks.active_txn_count, 1);
    assert_eq!(marks.oldest_active_txn_id, Some(42));
    assert_eq!(marks.newest_active_txn_id, Some(42));
    assert_eq!(marks.pending_batch_len, 0);
    assert_eq!(marks.pending_batch_oldest_age_ms, None);
    assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
    assert!(marks.has_active_txn_backlog);
    assert!(marks.has_backlog_blockers);
    assert_eq!(marks.backlog_blocker_count, 1);
    assert_eq!(
        marks.backlog_blocker_mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
    );
    assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN));
    assert!(!marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_WAL));
    assert!(!marks.has_pending_batch_backlog);
    assert!(!marks.has_wal_backlog);
    assert_eq!(marks.wal_buffered_count, 0);
    assert!(!marks.mutation_admission_saturated);
    assert!(!marks.quiescent_for_failover);
}

#[test]
fn backlog_blocker_enum_roundtrips_through_bits_and_labels() {
    for blocker in BacklogBlocker::ALL {
        assert!(blocker.bit().is_power_of_two());
        assert!(!blocker.as_str().is_empty());
        assert_eq!(BacklogBlocker::from_bit(blocker.bit()), Some(blocker));
        assert_eq!(BacklogBlocker::from_label(blocker.as_str()), Some(blocker));

        let mut marks = Engine::new_local_test_engine().replication_watermarks();
        marks.backlog_blocker_mask = blocker.bit();
        assert!(marks.has_blocker_kind(blocker));
        assert_eq!(marks.backlog_blockers().collect::<Vec<_>>(), vec![blocker]);
        assert_eq!(
            marks.backlog_blocker_labels().collect::<Vec<_>>(),
            vec![blocker.as_str()]
        );
        assert_eq!(
            marks.backlog_blocker_bits().collect::<Vec<_>>(),
            vec![blocker.bit()]
        );
        assert_eq!(
            ReplicationWatermarks::backlog_blockers_from_mask(marks.backlog_blocker_mask)
                .collect::<Vec<_>>(),
            vec![blocker]
        );
    }
}

#[test]
fn backlog_blocker_display_and_from_str_roundtrip() {
    for blocker in BacklogBlocker::ALL {
        let label = blocker.to_string();
        assert_eq!(label, blocker.as_str());
        assert_eq!(label.parse::<BacklogBlocker>(), Ok(blocker));
    }
}

#[test]
fn backlog_blocker_from_str_reports_unknown_label() {
    let err = "  not-a-real-blocker  "
        .parse::<BacklogBlocker>()
        .expect_err("unknown blocker labels should fail to parse");

    assert_eq!(err.label(), "not-a-real-blocker");
    assert_eq!(
        err.to_string(),
        "unknown backlog blocker label: not-a-real-blocker"
    );
}

#[test]
fn replication_watermarks_backlog_blockers_from_mask_ignores_unknown_bits() {
    let known_mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
        | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN;
    let unknown_mask = 1 << 7;

    assert_eq!(BacklogBlocker::from_bit(unknown_mask), None);
    assert_eq!(
        ReplicationWatermarks::backlog_blockers_from_mask(known_mask | unknown_mask)
            .collect::<Vec<_>>(),
        vec![BacklogBlocker::Wal, BacklogBlocker::ActiveTxn]
    );
}

#[test]
fn backlog_blocker_mask_helpers_strip_unknown_bits() {
    let unknown_mask = (1 << 5) | (1 << 7);
    let mixed_mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
        | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
        | unknown_mask;

    assert_eq!(
        ReplicationWatermarks::known_backlog_blocker_mask(),
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
    assert_eq!(
        ReplicationWatermarks::unknown_backlog_blocker_mask(mixed_mask),
        unknown_mask
    );
    assert_eq!(
        ReplicationWatermarks::sanitize_backlog_blocker_mask(mixed_mask),
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
    );
    assert_eq!(
        ReplicationWatermarks::backlog_blocker_count_from_mask(mixed_mask),
        2
    );
    assert!(ReplicationWatermarks::has_backlog_blockers_in_mask(
        mixed_mask
    ));
    assert!(!ReplicationWatermarks::has_backlog_blockers_in_mask(
        unknown_mask
    ));
}

#[test]
fn replication_watermarks_backlog_blocker_mask_from_labels_ignores_unknowns() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_labels([
        "pending_batch",
        "unknown",
        "active_txn",
        "pending_batch",
    ]);

    assert_eq!(BacklogBlocker::from_label("unknown"), None);
    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
    );
    assert_eq!(
        ReplicationWatermarks::backlog_blockers_from_mask(mask).collect::<Vec<_>>(),
        vec![BacklogBlocker::PendingBatch, BacklogBlocker::ActiveTxn]
    );
    assert_eq!(
        ReplicationWatermarks::backlog_blocker_labels_from_mask(mask).collect::<Vec<_>>(),
        vec!["pending_batch", "active_txn"]
    );
}

#[test]
fn backlog_blocker_label_decode_normalizes_case_spacing_and_hyphenation() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_labels([
        " WAL ",
        "pending-batch",
        "ACTIVE TXN",
        "commit-apply-gap",
        "apply visible gap",
        "pending.batch",
        "commit--apply  gap",
    ]);

    assert_eq!(
        BacklogBlocker::from_label("PENDING-BATCH"),
        Some(BacklogBlocker::PendingBatch)
    );
    assert_eq!(
        BacklogBlocker::from_label("apply visible gap"),
        Some(BacklogBlocker::ApplyVisibleGap)
    );
    assert_eq!(
        BacklogBlocker::from_label("pending.batch"),
        Some(BacklogBlocker::PendingBatch)
    );
    assert_eq!(
        BacklogBlocker::from_label("commit--apply  gap"),
        Some(BacklogBlocker::CommitApplyGap)
    );
    assert_eq!(
        BacklogBlocker::from_label("__wal__"),
        Some(BacklogBlocker::Wal)
    );
    assert_eq!(
        BacklogBlocker::from_label("___active.txn___"),
        Some(BacklogBlocker::ActiveTxn)
    );
    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_decodes_csv_like_streams() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "wal, pending-batch; ACTIVE TXN | unknown / apply visible gap : commit apply gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_ignores_empty_segments() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        " , ; | pending_batch || wal ,, ",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_WAL
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_multiline_streams() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "wal\n pending_batch\r\nACTIVE TXN\t| commit apply gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_jsonish_arrays() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "[\"wal\",\"active_txn\",\"apply visible gap\"]",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_braced_streams() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "{'wal';'active_txn';'apply visible gap'}",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_parenthesized_and_angle_bracket_streams() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "<(wal|active_txn|apply visible gap)>",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_single_quoted_arrays() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "['pending-batch','commit_apply_gap']",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_backtick_quoted_labels() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "`wal`,`active_txn`,`apply_visible_gap`",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_plus_delimiter() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "wal+active_txn+apply_visible_gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_windows_style_backslash_delimiter() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "wal\\active_txn\\apply_visible_gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_assignment_and_ampersand_delimiters() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "backlog_blockers=wal&active_txn&apply_visible_gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_percent_encoded_streams() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "wal%2Cpending-batch%7CACTIVE%20TXN%2Fapply_visible_gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_delimited_labels_from_mask_emits_canonical_order() {
    let mask = ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        | ReplicationWatermarks::BACKLOG_BLOCKER_WAL
        | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;

    assert_eq!(
        ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ","),
        "wal,active_txn,apply_visible_gap"
    );
    assert_eq!(
        ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, " | "),
        "wal | active_txn | apply_visible_gap"
    );
}

#[test]
fn backlog_blocker_delimited_mask_roundtrip_is_stable() {
    let mask = ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
        | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP;
    let labels = ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ";");

    assert_eq!(
        ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(&labels),
        mask
    );
}

#[test]
fn backlog_blocker_delimited_mask_roundtrip_is_stable_with_colon_delimiter() {
    let mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
        | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;
    let labels = ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ":");

    assert_eq!(
        ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(&labels),
        mask
    );
}

#[test]
fn replication_watermarks_aggregate_multiple_backlog_blockers() {
    let mut e = Engine::with_batching(8, Duration::from_secs(999));
    let t0 = Instant::now();

    e.enqueue_set_text(7, "SET a=1", t0).unwrap();
    e.execute_text(8, "BEGIN").unwrap();

    let marks = e.replication_watermarks();
    assert!(marks.has_pending_batch_backlog);
    assert!(marks.has_active_txn_backlog);
    assert!(marks.has_backlog_blockers);
    assert_eq!(marks.backlog_blocker_count, 2);
    assert_eq!(
        marks.backlog_blocker_mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
    );
    assert_eq!(
        marks.backlog_blocker_count,
        marks.backlog_blocker_mask.count_ones() as u8
    );
    assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH));
    assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN));
    assert!(!marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_WAL));
    assert!(!marks.has_backlog_blocker(1 << 7));
    assert_eq!(
        marks.backlog_blockers().collect::<Vec<_>>(),
        vec![BacklogBlocker::PendingBatch, BacklogBlocker::ActiveTxn]
    );
    assert_eq!(
        marks.backlog_blocker_labels().collect::<Vec<_>>(),
        vec!["pending_batch", "active_txn"]
    );
    assert_eq!(
        marks.backlog_blocker_bits().collect::<Vec<_>>(),
        vec![
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH,
            ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        ]
    );
    assert_eq!(marks.max_replication_gap(), 0);
    assert_eq!(marks.total_backlog_items(), 2);
    assert!(!marks.is_fully_caught_up());
    assert!(!marks.quiescent_for_failover);
    assert!(!marks.follower_promotion_ready);
}
