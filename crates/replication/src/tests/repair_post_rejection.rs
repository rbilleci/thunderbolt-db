#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_refresh_updates_identity_without_perturbing_gap(
) {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 5,
                index: 8,
                payload: vec![8].into(),
            },
        ],
        7,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 29,
    });
    r.append_entries_from_leader(
        6,
        7,
        5,
        vec![
            LogEntry {
                term: 6,
                index: 8,
                payload: vec![80].into(),
            },
            LogEntry {
                term: 6,
                index: 9,
                payload: vec![90].into(),
            },
            LogEntry {
                term: 6,
                index: 10,
                payload: vec![100].into(),
            },
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 13,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 53,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let err = r
        .append_entries_from_leader(
            7,
            99,
            6,
            vec![LogEntry {
                term: 7,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        7,
        9,
        6,
        vec![
            LogEntry {
                term: 7,
                index: 10,
                payload: vec![110].into(),
            },
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![111].into(),
            },
        ],
        10,
    )
    .unwrap();

    let repair_status = r.status_snapshot();
    assert!(repair_status.has_speculative_tail());
    assert_eq!(repair_status.live.term, 7);
    assert_eq!(repair_status.live.commit_index, 10);
    assert_eq!(repair_status.live.applied_index, 9);
    assert_eq!(repair_status.live.next_index, 12);
    assert_eq!(repair_status.live.uncommitted_entry_count, 1);
    assert_eq!(repair_status.durable.commit_index, 10);
    assert_eq!(repair_status.durable.applied_index, 9);
    assert_eq!(repair_status.durable.next_index, 11);
    assert_eq!(repair_status.durable.uncommitted_entry_count, 0);
    assert_eq!(repair_status.live.snapshot.snapshot_id, 47);
    assert_eq!(repair_status.durable.snapshot.snapshot_id, 47);
    let repair_gap = r.recovery_progress_gap();

    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 59,
    });

    let refreshed_repair_status = r.status_snapshot();
    assert!(refreshed_repair_status.has_speculative_tail());
    assert_eq!(refreshed_repair_status.recovery_gap, repair_gap);
    assert_eq!(refreshed_repair_status.live.term, repair_status.live.term);
    assert_eq!(
        refreshed_repair_status.live.commit_index,
        repair_status.live.commit_index
    );
    assert_eq!(
        refreshed_repair_status.live.applied_index,
        repair_status.live.applied_index
    );
    assert_eq!(
        refreshed_repair_status.live.next_index,
        repair_status.live.next_index
    );
    assert_eq!(
        refreshed_repair_status.live.uncommitted_entry_count,
        repair_status.live.uncommitted_entry_count
    );
    assert_eq!(
        refreshed_repair_status.durable.commit_index,
        repair_status.durable.commit_index
    );
    assert_eq!(
        refreshed_repair_status.durable.applied_index,
        repair_status.durable.applied_index
    );
    assert_eq!(
        refreshed_repair_status.durable.next_index,
        repair_status.durable.next_index
    );
    assert_eq!(
        refreshed_repair_status.durable.uncommitted_entry_count,
        repair_status.durable.uncommitted_entry_count
    );
    assert_eq!(refreshed_repair_status.live.snapshot.snapshot_id, 59);
    assert_eq!(refreshed_repair_status.durable.snapshot.snapshot_id, 59);

    let refreshed_repair_recovery = r.recovery_state();
    assert_eq!(refreshed_repair_recovery.term, 7);
    assert_eq!(refreshed_repair_recovery.snapshot.snapshot_id, 59);
    let resumed_during_refreshed_repair =
        RaftReplicator::resume_as_follower(3, refreshed_repair_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_refreshed_repair.status_snapshot().live,
        refreshed_repair_status.durable
    );
    assert_eq!(
        refreshed_repair_recovery.progress_as_follower().unwrap(),
        refreshed_repair_status.durable
    );

    r.append_entries_from_leader(7, 11, 7, Vec::new(), 11)
        .unwrap();

    let committed_status = r.status_snapshot();
    assert!(committed_status.is_restart_equivalent());
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);
    assert_eq!(committed_status.live.commit_index, 11);
    assert_eq!(committed_status.live.applied_index, 9);
    assert_eq!(committed_status.live.next_index, 12);
    assert_eq!(committed_status.live.uncommitted_entry_count, 0);

    let committed_recovery = r.recovery_state();
    assert_eq!(committed_recovery.term, 7);
    assert_eq!(committed_recovery.snapshot.snapshot_id, 59);
    let resumed_after_commit =
        RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_commit.status_snapshot().live,
        committed_status.durable
    );
    assert_eq!(
        committed_recovery.progress_as_follower().unwrap(),
        committed_status.durable
    );

    r.mark_applied(11);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live, applied_status.durable);
    assert_eq!(applied_status.live.term, 7);
    assert_eq!(applied_status.live.applied_index, 11);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 59);
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.term, 7);
    assert_eq!(applied_recovery.snapshot.snapshot_id, 59);
    let resumed_after_apply =
        RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_apply.status_snapshot().live,
        applied_status.live
    );
    assert_eq!(
        applied_recovery.progress_as_follower().unwrap(),
        applied_status.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 59);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_role_handoff_discards_only_fresh_tail(
) {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 5,
                index: 8,
                payload: vec![8].into(),
            },
        ],
        7,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 29,
    });
    r.append_entries_from_leader(
        6,
        7,
        5,
        vec![
            LogEntry {
                term: 6,
                index: 8,
                payload: vec![80].into(),
            },
            LogEntry {
                term: 6,
                index: 9,
                payload: vec![90].into(),
            },
            LogEntry {
                term: 6,
                index: 10,
                payload: vec![100].into(),
            },
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 13,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 53,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let err = r
        .append_entries_from_leader(
            7,
            99,
            6,
            vec![LogEntry {
                term: 7,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        7,
        9,
        6,
        vec![
            LogEntry {
                term: 7,
                index: 10,
                payload: vec![110].into(),
            },
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![111].into(),
            },
        ],
        10,
    )
    .unwrap();

    let before_role_change = r.status_snapshot();
    assert!(before_role_change.has_speculative_tail());
    assert_eq!(before_role_change.live.term, 7);
    assert_eq!(before_role_change.live.commit_index, 10);
    assert_eq!(before_role_change.live.applied_index, 9);
    assert_eq!(before_role_change.live.next_index, 12);
    assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
    assert_eq!(before_role_change.live.snapshot.snapshot_id, 47);
    assert_eq!(before_role_change.durable.commit_index, 10);
    assert_eq!(before_role_change.durable.applied_index, 9);
    assert_eq!(before_role_change.durable.next_index, 11);
    assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(before_role_change.durable.snapshot.snapshot_id, 47);

    r.become_candidate(8);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.term, 8);
    assert_eq!(after_role_change.live.commit_index, 10);
    assert_eq!(after_role_change.live.applied_index, 9);
    assert_eq!(after_role_change.live.next_index, 11);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 47);
    assert!(after_role_change.live.has_committed_entries_pending_apply);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 8);
    assert_eq!(after_role_change.durable.commit_index, 10);
    assert_eq!(after_role_change.durable.applied_index, 9);
    assert_eq!(after_role_change.durable.next_index, 11);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 47);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 8);
    assert_eq!(recovery.snapshot.snapshot_id, 47);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 47);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_collapses_cleanly(
) {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 5,
                index: 8,
                payload: vec![8].into(),
            },
        ],
        7,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 29,
    });
    r.append_entries_from_leader(
        6,
        7,
        5,
        vec![
            LogEntry {
                term: 6,
                index: 8,
                payload: vec![80].into(),
            },
            LogEntry {
                term: 6,
                index: 9,
                payload: vec![90].into(),
            },
            LogEntry {
                term: 6,
                index: 10,
                payload: vec![100].into(),
            },
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 13,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 53,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let err = r
        .append_entries_from_leader(
            7,
            99,
            6,
            vec![LogEntry {
                term: 7,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        7,
        9,
        6,
        vec![
            LogEntry {
                term: 7,
                index: 10,
                payload: vec![110].into(),
            },
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![111].into(),
            },
        ],
        10,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 59,
    });

    let refreshed_status = r.status_snapshot();
    assert!(refreshed_status.has_speculative_tail());
    assert_eq!(refreshed_status.live.term, 7);
    assert_eq!(refreshed_status.live.commit_index, 10);
    assert_eq!(refreshed_status.live.applied_index, 9);
    assert_eq!(refreshed_status.live.next_index, 12);
    assert_eq!(refreshed_status.live.uncommitted_entry_count, 1);
    assert_eq!(refreshed_status.live.snapshot.snapshot_id, 59);
    assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 59);

    let err = r
        .append_entries_from_leader(
            8,
            99,
            7,
            vec![LogEntry {
                term: 8,
                index: 100,
                payload: vec![120].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    let after_rejection = r.status_snapshot();
    assert!(after_rejection.is_restart_equivalent());
    assert_eq!(after_rejection.live.term, 8);
    assert_eq!(after_rejection.live.snapshot.snapshot_id, 59);
    assert_eq!(after_rejection.durable.snapshot.snapshot_id, 59);
    assert_eq!(after_rejection.live.commit_index, 10);
    assert_eq!(after_rejection.live.applied_index, 9);
    assert_eq!(after_rejection.live.next_index, 11);
    assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
    assert_eq!(after_rejection.durable.commit_index, 10);
    assert_eq!(after_rejection.durable.applied_index, 9);
    assert_eq!(after_rejection.durable.next_index, 11);
    assert_eq!(after_rejection.durable.uncommitted_entry_count, 0);
    assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
    assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 8);
    assert_eq!(recovery.snapshot.snapshot_id, 59);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_rejection.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_rejection.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 59);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_survives_later_repair(
) {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 5,
                index: 8,
                payload: vec![8].into(),
            },
        ],
        7,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 29,
    });
    r.append_entries_from_leader(
        6,
        7,
        5,
        vec![
            LogEntry {
                term: 6,
                index: 8,
                payload: vec![80].into(),
            },
            LogEntry {
                term: 6,
                index: 9,
                payload: vec![90].into(),
            },
            LogEntry {
                term: 6,
                index: 10,
                payload: vec![100].into(),
            },
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 13,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 53,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let err = r
        .append_entries_from_leader(
            7,
            99,
            6,
            vec![LogEntry {
                term: 7,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        7,
        9,
        6,
        vec![
            LogEntry {
                term: 7,
                index: 10,
                payload: vec![110].into(),
            },
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![111].into(),
            },
        ],
        10,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 59,
    });

    let err = r
        .append_entries_from_leader(
            8,
            99,
            7,
            vec![LogEntry {
                term: 8,
                index: 100,
                payload: vec![120].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    let after_rejection = r.status_snapshot();
    assert!(after_rejection.is_restart_equivalent());
    assert_eq!(after_rejection.live.term, 8);
    assert_eq!(after_rejection.live.snapshot.snapshot_id, 59);
    assert_eq!(after_rejection.durable.snapshot.snapshot_id, 59);
    assert_eq!(after_rejection.live.commit_index, 10);
    assert_eq!(after_rejection.live.applied_index, 9);
    assert_eq!(after_rejection.live.next_index, 11);
    assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
    assert_eq!(after_rejection.durable.commit_index, 10);
    assert_eq!(after_rejection.durable.applied_index, 9);
    assert_eq!(after_rejection.durable.next_index, 11);
    assert_eq!(after_rejection.durable.uncommitted_entry_count, 0);
    assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
    assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

    r.append_entries_from_leader(
        8,
        10,
        7,
        vec![
            LogEntry {
                term: 8,
                index: 11,
                payload: vec![121].into(),
            },
            LogEntry {
                term: 8,
                index: 12,
                payload: vec![122].into(),
            },
        ],
        11,
    )
    .unwrap();

    let after_repair = r.status_snapshot();
    assert!(after_repair.has_speculative_tail());
    assert_eq!(after_repair.live.term, 8);
    assert_eq!(after_repair.live.commit_index, 11);
    assert_eq!(after_repair.live.applied_index, 9);
    assert_eq!(after_repair.live.next_index, 13);
    assert_eq!(after_repair.live.uncommitted_entry_count, 1);
    assert_eq!(after_repair.live.snapshot.snapshot_id, 59);
    assert_eq!(after_repair.durable.commit_index, 11);
    assert_eq!(after_repair.durable.applied_index, 9);
    assert_eq!(after_repair.durable.next_index, 12);
    assert_eq!(after_repair.durable.uncommitted_entry_count, 0);
    assert_eq!(after_repair.durable.snapshot.snapshot_id, 59);
    assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
    assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);

    let repair_recovery = r.recovery_state();
    assert_eq!(repair_recovery.term, 8);
    assert_eq!(repair_recovery.snapshot.snapshot_id, 59);
    let resumed_during_repair =
        RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_repair.status_snapshot().live,
        after_repair.durable
    );
    assert_eq!(
        repair_recovery.progress_as_follower().unwrap(),
        after_repair.durable
    );

    r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
        .unwrap();

    let committed_status = r.status_snapshot();
    assert!(committed_status.is_restart_equivalent());
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.term, 8);
    assert_eq!(committed_status.live.commit_index, 12);
    assert_eq!(committed_status.live.applied_index, 9);
    assert_eq!(committed_status.live.next_index, 13);
    assert_eq!(committed_status.live.uncommitted_entry_count, 0);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);

    r.mark_applied(12);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live, applied_status.durable);
    assert_eq!(applied_status.live.term, 8);
    assert_eq!(applied_status.live.applied_index, 12);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 59);

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.term, 8);
    assert_eq!(applied_recovery.snapshot.snapshot_id, 59);
    let resumed_after_apply =
        RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_apply.status_snapshot().live,
        applied_status.live
    );
    assert_eq!(
        applied_recovery.progress_as_follower().unwrap(),
        applied_status.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 59);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_role_handoff_discards_only_fresh_tail(
) {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 5,
                index: 8,
                payload: vec![8].into(),
            },
        ],
        7,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 29,
    });
    r.append_entries_from_leader(
        6,
        7,
        5,
        vec![
            LogEntry {
                term: 6,
                index: 8,
                payload: vec![80].into(),
            },
            LogEntry {
                term: 6,
                index: 9,
                payload: vec![90].into(),
            },
            LogEntry {
                term: 6,
                index: 10,
                payload: vec![100].into(),
            },
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 13,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 53,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let err = r
        .append_entries_from_leader(
            7,
            99,
            6,
            vec![LogEntry {
                term: 7,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        7,
        9,
        6,
        vec![
            LogEntry {
                term: 7,
                index: 10,
                payload: vec![110].into(),
            },
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![111].into(),
            },
        ],
        10,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 59,
    });

    let err = r
        .append_entries_from_leader(
            8,
            99,
            7,
            vec![LogEntry {
                term: 8,
                index: 100,
                payload: vec![120].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        8,
        10,
        7,
        vec![
            LogEntry {
                term: 8,
                index: 11,
                payload: vec![121].into(),
            },
            LogEntry {
                term: 8,
                index: 12,
                payload: vec![122].into(),
            },
        ],
        11,
    )
    .unwrap();

    let before_role_change = r.status_snapshot();
    assert!(before_role_change.has_speculative_tail());
    assert_eq!(before_role_change.live.role, Role::Follower);
    assert_eq!(before_role_change.live.term, 8);
    assert_eq!(before_role_change.live.commit_index, 11);
    assert_eq!(before_role_change.live.applied_index, 9);
    assert_eq!(before_role_change.live.next_index, 13);
    assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
    assert_eq!(before_role_change.live.snapshot.snapshot_id, 59);
    assert_eq!(before_role_change.durable.role, Role::Follower);
    assert_eq!(before_role_change.durable.term, 8);
    assert_eq!(before_role_change.durable.commit_index, 11);
    assert_eq!(before_role_change.durable.applied_index, 9);
    assert_eq!(before_role_change.durable.next_index, 12);
    assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(before_role_change.durable.snapshot.snapshot_id, 59);
    assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
    assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

    r.become_candidate(9);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.term, 9);
    assert_eq!(after_role_change.live.commit_index, 11);
    assert_eq!(after_role_change.live.applied_index, 9);
    assert_eq!(after_role_change.live.next_index, 12);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 59);
    assert!(after_role_change.live.has_committed_entries_pending_apply);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 9);
    assert_eq!(after_role_change.durable.commit_index, 11);
    assert_eq!(after_role_change.durable.applied_index, 9);
    assert_eq!(after_role_change.durable.next_index, 12);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 59);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 9);
    assert_eq!(recovery.snapshot.snapshot_id, 59);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 59);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_retires_cleanly_through_commit_and_apply(
) {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 5,
                index: 8,
                payload: vec![8].into(),
            },
        ],
        7,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 29,
    });
    r.append_entries_from_leader(
        6,
        7,
        5,
        vec![
            LogEntry {
                term: 6,
                index: 8,
                payload: vec![80].into(),
            },
            LogEntry {
                term: 6,
                index: 9,
                payload: vec![90].into(),
            },
            LogEntry {
                term: 6,
                index: 10,
                payload: vec![100].into(),
            },
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 13,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 53,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let err = r
        .append_entries_from_leader(
            7,
            99,
            6,
            vec![LogEntry {
                term: 7,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        7,
        9,
        6,
        vec![
            LogEntry {
                term: 7,
                index: 10,
                payload: vec![110].into(),
            },
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![111].into(),
            },
        ],
        10,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 59,
    });

    let err = r
        .append_entries_from_leader(
            8,
            99,
            7,
            vec![LogEntry {
                term: 8,
                index: 100,
                payload: vec![120].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        8,
        10,
        7,
        vec![
            LogEntry {
                term: 8,
                index: 11,
                payload: vec![121].into(),
            },
            LogEntry {
                term: 8,
                index: 12,
                payload: vec![122].into(),
            },
        ],
        11,
    )
    .unwrap();

    let before_commit = r.status_snapshot();
    assert!(before_commit.has_speculative_tail());
    assert_eq!(before_commit.live.term, 8);
    assert_eq!(before_commit.live.commit_index, 11);
    assert_eq!(before_commit.live.applied_index, 9);
    assert_eq!(before_commit.live.next_index, 13);
    assert_eq!(before_commit.live.uncommitted_entry_count, 1);
    assert_eq!(before_commit.live.snapshot.snapshot_id, 59);
    assert_eq!(before_commit.durable.term, 8);
    assert_eq!(before_commit.durable.commit_index, 11);
    assert_eq!(before_commit.durable.applied_index, 9);
    assert_eq!(before_commit.durable.next_index, 12);
    assert_eq!(before_commit.durable.uncommitted_entry_count, 0);
    assert_eq!(before_commit.durable.snapshot.snapshot_id, 59);
    assert_eq!(before_commit.recovery_gap.next_index_gap, 1);
    assert_eq!(before_commit.recovery_gap.uncommitted_entry_gap, 1);

    let repair_recovery = r.recovery_state();
    assert_eq!(repair_recovery.term, 8);
    assert_eq!(repair_recovery.snapshot.snapshot_id, 59);
    let resumed_during_rerepair =
        RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_rerepair.status_snapshot().live,
        before_commit.durable
    );
    assert_eq!(
        repair_recovery.progress_as_follower().unwrap(),
        before_commit.durable
    );

    r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
        .unwrap();

    let committed_status = r.status_snapshot();
    assert!(committed_status.is_restart_equivalent());
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.term, 8);
    assert_eq!(committed_status.live.commit_index, 12);
    assert_eq!(committed_status.live.applied_index, 9);
    assert_eq!(committed_status.live.next_index, 13);
    assert_eq!(committed_status.live.uncommitted_entry_count, 0);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
    assert_eq!(committed_status.durable.term, 8);
    assert_eq!(committed_status.durable.commit_index, 12);
    assert_eq!(committed_status.durable.applied_index, 9);
    assert_eq!(committed_status.durable.next_index, 13);
    assert_eq!(committed_status.durable.uncommitted_entry_count, 0);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);

    r.mark_applied(12);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live, applied_status.durable);
    assert_eq!(applied_status.live.term, 8);
    assert_eq!(applied_status.live.commit_index, 12);
    assert_eq!(applied_status.live.applied_index, 12);
    assert_eq!(applied_status.live.next_index, 13);
    assert_eq!(applied_status.live.uncommitted_entry_count, 0);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 59);

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.term, 8);
    assert_eq!(applied_recovery.snapshot.snapshot_id, 59);
    let resumed_after_apply =
        RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_apply.status_snapshot().live,
        applied_status.live
    );
    assert_eq!(
        applied_recovery.progress_as_follower().unwrap(),
        applied_status.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 59);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_keeps_stale_installs_inert_through_commit_and_apply(
) {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 5,
                index: 8,
                payload: vec![8].into(),
            },
        ],
        7,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 29,
    });
    r.append_entries_from_leader(
        6,
        7,
        5,
        vec![
            LogEntry {
                term: 6,
                index: 8,
                payload: vec![80].into(),
            },
            LogEntry {
                term: 6,
                index: 9,
                payload: vec![90].into(),
            },
            LogEntry {
                term: 6,
                index: 10,
                payload: vec![100].into(),
            },
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 13,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 53,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let err = r
        .append_entries_from_leader(
            7,
            99,
            6,
            vec![LogEntry {
                term: 7,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        7,
        9,
        6,
        vec![
            LogEntry {
                term: 7,
                index: 10,
                payload: vec![110].into(),
            },
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![111].into(),
            },
        ],
        10,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 59,
    });

    let err = r
        .append_entries_from_leader(
            8,
            99,
            7,
            vec![LogEntry {
                term: 8,
                index: 100,
                payload: vec![120].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        8,
        10,
        7,
        vec![
            LogEntry {
                term: 8,
                index: 11,
                payload: vec![121].into(),
            },
            LogEntry {
                term: 8,
                index: 12,
                payload: vec![122].into(),
            },
        ],
        11,
    )
    .unwrap();

    let repair_status = r.status_snapshot();
    assert!(repair_status.has_speculative_tail());
    assert_eq!(repair_status.live.term, 8);
    assert_eq!(repair_status.live.snapshot.snapshot_id, 59);
    assert_eq!(repair_status.durable.snapshot.snapshot_id, 59);

    let repair_recovery = r.recovery_state();
    let repair_gap = r.recovery_progress_gap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 58,
    });
    assert_eq!(r.status_snapshot(), repair_status);
    assert_eq!(r.recovery_state(), repair_recovery);
    assert_eq!(r.recovery_progress_gap(), repair_gap);
    let resumed_during_rerepair_after_stale =
        RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_rerepair_after_stale.status_snapshot().live,
        repair_status.durable
    );
    assert_eq!(
        repair_recovery.progress_as_follower().unwrap(),
        repair_status.durable
    );

    r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
        .unwrap();

    let committed_status = r.status_snapshot();
    assert!(committed_status.is_restart_equivalent());
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);

    let committed_recovery = r.recovery_state();
    let committed_gap = r.recovery_progress_gap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 57,
    });
    assert_eq!(r.status_snapshot(), committed_status);
    assert_eq!(r.recovery_state(), committed_recovery);
    assert_eq!(r.recovery_progress_gap(), committed_gap);
    let resumed_after_commit_stale =
        RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_commit_stale.status_snapshot().live,
        committed_status.durable
    );
    assert_eq!(
        committed_recovery.progress_as_follower().unwrap(),
        committed_status.durable
    );

    r.mark_applied(12);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live, applied_status.durable);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 59);

    let applied_recovery = r.recovery_state();
    let applied_gap = r.recovery_progress_gap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 56,
    });
    assert_eq!(r.status_snapshot(), applied_status);
    assert_eq!(r.recovery_state(), applied_recovery);
    assert_eq!(r.recovery_progress_gap(), applied_gap);
    let resumed_after_apply_stale =
        RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_apply_stale.status_snapshot().live,
        applied_status.durable
    );
    assert_eq!(
        applied_recovery.progress_as_follower().unwrap(),
        applied_status.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 59);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_keeps_gap_shape_and_retires_on_new_identity(
) {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 5,
                index: 8,
                payload: vec![8].into(),
            },
        ],
        7,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 29,
    });
    r.append_entries_from_leader(
        6,
        7,
        5,
        vec![
            LogEntry {
                term: 6,
                index: 8,
                payload: vec![80].into(),
            },
            LogEntry {
                term: 6,
                index: 9,
                payload: vec![90].into(),
            },
            LogEntry {
                term: 6,
                index: 10,
                payload: vec![100].into(),
            },
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 13,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 53,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let err = r
        .append_entries_from_leader(
            7,
            99,
            6,
            vec![LogEntry {
                term: 7,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        7,
        9,
        6,
        vec![
            LogEntry {
                term: 7,
                index: 10,
                payload: vec![110].into(),
            },
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![111].into(),
            },
        ],
        10,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 59,
    });

    let err = r
        .append_entries_from_leader(
            8,
            99,
            7,
            vec![LogEntry {
                term: 8,
                index: 100,
                payload: vec![120].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        8,
        10,
        7,
        vec![
            LogEntry {
                term: 8,
                index: 11,
                payload: vec![121].into(),
            },
            LogEntry {
                term: 8,
                index: 12,
                payload: vec![122].into(),
            },
        ],
        11,
    )
    .unwrap();

    let before_refresh = r.status_snapshot();
    assert!(before_refresh.has_speculative_tail());
    assert_eq!(before_refresh.live.snapshot.snapshot_id, 59);
    assert_eq!(before_refresh.durable.snapshot.snapshot_id, 59);
    assert_eq!(before_refresh.recovery_gap.next_index_gap, 1);
    assert_eq!(before_refresh.recovery_gap.uncommitted_entry_gap, 1);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 61,
    });

    let after_refresh = r.status_snapshot();
    assert!(after_refresh.has_speculative_tail());
    assert_eq!(after_refresh.live.term, 8);
    assert_eq!(after_refresh.live.commit_index, 11);
    assert_eq!(after_refresh.live.applied_index, 9);
    assert_eq!(after_refresh.live.next_index, 13);
    assert_eq!(after_refresh.live.uncommitted_entry_count, 1);
    assert_eq!(after_refresh.live.snapshot.snapshot_id, 61);
    assert_eq!(after_refresh.durable.term, 8);
    assert_eq!(after_refresh.durable.commit_index, 11);
    assert_eq!(after_refresh.durable.applied_index, 9);
    assert_eq!(after_refresh.durable.next_index, 12);
    assert_eq!(after_refresh.durable.uncommitted_entry_count, 0);
    assert_eq!(after_refresh.durable.snapshot.snapshot_id, 61);
    assert_eq!(after_refresh.recovery_gap.next_index_gap, 1);
    assert_eq!(after_refresh.recovery_gap.uncommitted_entry_gap, 1);

    let refresh_recovery = r.recovery_state();
    assert_eq!(refresh_recovery.term, 8);
    assert_eq!(refresh_recovery.snapshot.snapshot_id, 61);
    let resumed_during_refresh_rerepair =
        RaftReplicator::resume_as_follower(3, refresh_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_refresh_rerepair.status_snapshot().live,
        after_refresh.durable
    );
    assert_eq!(
        refresh_recovery.progress_as_follower().unwrap(),
        after_refresh.durable
    );

    r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
        .unwrap();

    let committed_status = r.status_snapshot();
    assert!(committed_status.is_restart_equivalent());
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 61);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 61);

    r.mark_applied(12);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live, applied_status.durable);
    assert_eq!(applied_status.live.term, 8);
    assert_eq!(applied_status.live.applied_index, 12);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 61);

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.term, 8);
    assert_eq!(applied_recovery.snapshot.snapshot_id, 61);
    let resumed_after_apply =
        RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_apply.status_snapshot().live,
        applied_status.live
    );
    assert_eq!(
        applied_recovery.progress_as_follower().unwrap(),
        applied_status.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 61);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_role_handoff_discards_only_fresh_tail(
) {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 5,
                index: 8,
                payload: vec![8].into(),
            },
        ],
        7,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 29,
    });
    r.append_entries_from_leader(
        6,
        7,
        5,
        vec![
            LogEntry {
                term: 6,
                index: 8,
                payload: vec![80].into(),
            },
            LogEntry {
                term: 6,
                index: 9,
                payload: vec![90].into(),
            },
            LogEntry {
                term: 6,
                index: 10,
                payload: vec![100].into(),
            },
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 13,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 53,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let err = r
        .append_entries_from_leader(
            7,
            99,
            6,
            vec![LogEntry {
                term: 7,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        7,
        9,
        6,
        vec![
            LogEntry {
                term: 7,
                index: 10,
                payload: vec![110].into(),
            },
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![111].into(),
            },
        ],
        10,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 59,
    });

    let err = r
        .append_entries_from_leader(
            8,
            99,
            7,
            vec![LogEntry {
                term: 8,
                index: 100,
                payload: vec![120].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        8,
        10,
        7,
        vec![
            LogEntry {
                term: 8,
                index: 11,
                payload: vec![121].into(),
            },
            LogEntry {
                term: 8,
                index: 12,
                payload: vec![122].into(),
            },
        ],
        11,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 61,
    });

    let before_role_change = r.status_snapshot();
    assert!(before_role_change.has_speculative_tail());
    assert_eq!(before_role_change.live.role, Role::Follower);
    assert_eq!(before_role_change.live.term, 8);
    assert_eq!(before_role_change.live.commit_index, 11);
    assert_eq!(before_role_change.live.applied_index, 9);
    assert_eq!(before_role_change.live.next_index, 13);
    assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
    assert_eq!(before_role_change.live.snapshot.snapshot_id, 61);
    assert_eq!(before_role_change.durable.role, Role::Follower);
    assert_eq!(before_role_change.durable.term, 8);
    assert_eq!(before_role_change.durable.commit_index, 11);
    assert_eq!(before_role_change.durable.applied_index, 9);
    assert_eq!(before_role_change.durable.next_index, 12);
    assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(before_role_change.durable.snapshot.snapshot_id, 61);
    assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
    assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

    r.become_candidate(9);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.term, 9);
    assert_eq!(after_role_change.live.commit_index, 11);
    assert_eq!(after_role_change.live.applied_index, 9);
    assert_eq!(after_role_change.live.next_index, 12);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 61);
    assert!(after_role_change.live.has_committed_entries_pending_apply);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 9);
    assert_eq!(after_role_change.durable.commit_index, 11);
    assert_eq!(after_role_change.durable.applied_index, 9);
    assert_eq!(after_role_change.durable.next_index, 12);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 61);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 9);
    assert_eq!(recovery.snapshot.snapshot_id, 61);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 61);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_still_collapses_cleanly_on_newer_rejection(
) {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 5,
                index: 8,
                payload: vec![8].into(),
            },
        ],
        7,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 29,
    });
    r.append_entries_from_leader(
        6,
        7,
        5,
        vec![
            LogEntry {
                term: 6,
                index: 8,
                payload: vec![80].into(),
            },
            LogEntry {
                term: 6,
                index: 9,
                payload: vec![90].into(),
            },
            LogEntry {
                term: 6,
                index: 10,
                payload: vec![100].into(),
            },
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 13,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 53,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let err = r
        .append_entries_from_leader(
            7,
            99,
            6,
            vec![LogEntry {
                term: 7,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        7,
        9,
        6,
        vec![
            LogEntry {
                term: 7,
                index: 10,
                payload: vec![110].into(),
            },
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![111].into(),
            },
        ],
        10,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 59,
    });

    let err = r
        .append_entries_from_leader(
            8,
            99,
            7,
            vec![LogEntry {
                term: 8,
                index: 100,
                payload: vec![120].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        8,
        10,
        7,
        vec![
            LogEntry {
                term: 8,
                index: 11,
                payload: vec![121].into(),
            },
            LogEntry {
                term: 8,
                index: 12,
                payload: vec![122].into(),
            },
        ],
        11,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 61,
    });

    let before_rejection = r.status_snapshot();
    assert!(before_rejection.has_speculative_tail());
    assert_eq!(before_rejection.live.snapshot.snapshot_id, 61);
    assert_eq!(before_rejection.durable.snapshot.snapshot_id, 61);

    let err = r
        .append_entries_from_leader(
            9,
            99,
            8,
            vec![LogEntry {
                term: 9,
                index: 100,
                payload: vec![130].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    let after_rejection = r.status_snapshot();
    assert!(after_rejection.is_restart_equivalent());
    assert_eq!(after_rejection.live.role, Role::Follower);
    assert_eq!(after_rejection.live.term, 9);
    assert_eq!(after_rejection.live.commit_index, 11);
    assert_eq!(after_rejection.live.applied_index, 9);
    assert_eq!(after_rejection.live.next_index, 12);
    assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
    assert_eq!(after_rejection.live.snapshot.snapshot_id, 61);
    assert!(after_rejection.live.has_committed_entries_pending_apply);
    assert_eq!(after_rejection.durable, after_rejection.live);
    assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
    assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

    let rejection_recovery = r.recovery_state();
    assert_eq!(rejection_recovery.term, 9);
    assert_eq!(rejection_recovery.snapshot.snapshot_id, 61);
    let resumed_after_rejection =
        RaftReplicator::resume_as_follower(3, rejection_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_rejection.status_snapshot().live,
        after_rejection.live
    );
    assert_eq!(
        rejection_recovery.progress_as_follower().unwrap(),
        after_rejection.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 61);
}
