#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair(
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

    r.append_entries_from_leader(
        9,
        11,
        8,
        vec![
            LogEntry {
                term: 9,
                index: 12,
                payload: vec![131].into(),
            },
            LogEntry {
                term: 9,
                index: 13,
                payload: vec![132].into(),
            },
        ],
        12,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 67,
    });

    let err = r
        .append_entries_from_leader(
            10,
            99,
            9,
            vec![LogEntry {
                term: 10,
                index: 100,
                payload: vec![140].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        10,
        12,
        9,
        vec![
            LogEntry {
                term: 10,
                index: 13,
                payload: vec![141].into(),
            },
            LogEntry {
                term: 10,
                index: 14,
                payload: vec![142].into(),
            },
        ],
        13,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 71,
    });

    let err = r
        .append_entries_from_leader(
            11,
            99,
            10,
            vec![LogEntry {
                term: 11,
                index: 100,
                payload: vec![150].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    let after_rejection = r.status_snapshot();
    assert!(after_rejection.is_restart_equivalent());
    assert_eq!(after_rejection.live.snapshot.snapshot_id, 71);
    assert_eq!(after_rejection.live.term, 11);
    assert_eq!(after_rejection.live.commit_index, 13);
    assert_eq!(after_rejection.live.next_index, 14);

    r.append_entries_from_leader(
        11,
        13,
        10,
        vec![
            LogEntry {
                term: 11,
                index: 14,
                payload: vec![151].into(),
            },
            LogEntry {
                term: 11,
                index: 15,
                payload: vec![152].into(),
            },
        ],
        14,
    )
    .unwrap();

    let after_repair = r.status_snapshot();
    assert!(!after_repair.is_restart_equivalent());
    assert!(after_repair.has_speculative_tail());
    assert_eq!(after_repair.live.role, Role::Follower);
    assert_eq!(after_repair.live.term, 11);
    assert_eq!(after_repair.live.commit_index, 14);
    assert_eq!(after_repair.live.applied_index, 9);
    assert_eq!(after_repair.live.next_index, 16);
    assert_eq!(after_repair.live.uncommitted_entry_count, 1);
    assert_eq!(after_repair.live.snapshot.snapshot_id, 71);
    assert_eq!(after_repair.durable.snapshot.snapshot_id, 71);
    assert_eq!(after_repair.durable.commit_index, 14);
    assert_eq!(after_repair.durable.next_index, 15);
    assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
    assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);
    assert!(after_repair.live.has_committed_entries_pending_apply);

    let repair_recovery = r.recovery_state();
    assert_eq!(repair_recovery.term, 11);
    assert_eq!(repair_recovery.snapshot.snapshot_id, 71);
    assert_eq!(repair_recovery.commit_index(), 14);
    assert_eq!(repair_recovery.next_index(), 15);
    let resumed_after_repair =
        RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_repair.status_snapshot().live,
        after_repair.durable
    );
    assert_eq!(
        repair_recovery.progress_as_follower().unwrap(),
        after_repair.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 71);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_role_handoff_discards_only_fresh_tail(
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

    r.append_entries_from_leader(
        9,
        11,
        8,
        vec![
            LogEntry {
                term: 9,
                index: 12,
                payload: vec![131].into(),
            },
            LogEntry {
                term: 9,
                index: 13,
                payload: vec![132].into(),
            },
        ],
        12,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 67,
    });

    let err = r
        .append_entries_from_leader(
            10,
            99,
            9,
            vec![LogEntry {
                term: 10,
                index: 100,
                payload: vec![140].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        10,
        12,
        9,
        vec![
            LogEntry {
                term: 10,
                index: 13,
                payload: vec![141].into(),
            },
            LogEntry {
                term: 10,
                index: 14,
                payload: vec![142].into(),
            },
        ],
        13,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 71,
    });

    let err = r
        .append_entries_from_leader(
            11,
            99,
            10,
            vec![LogEntry {
                term: 11,
                index: 100,
                payload: vec![150].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        11,
        13,
        10,
        vec![
            LogEntry {
                term: 11,
                index: 14,
                payload: vec![151].into(),
            },
            LogEntry {
                term: 11,
                index: 15,
                payload: vec![152].into(),
            },
        ],
        14,
    )
    .unwrap();

    let before_role_change = r.status_snapshot();
    assert!(before_role_change.has_speculative_tail());
    assert_eq!(before_role_change.live.role, Role::Follower);
    assert_eq!(before_role_change.live.term, 11);
    assert_eq!(before_role_change.live.commit_index, 14);
    assert_eq!(before_role_change.live.applied_index, 9);
    assert_eq!(before_role_change.live.next_index, 16);
    assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
    assert_eq!(before_role_change.live.snapshot.snapshot_id, 71);
    assert_eq!(before_role_change.durable.role, Role::Follower);
    assert_eq!(before_role_change.durable.term, 11);
    assert_eq!(before_role_change.durable.commit_index, 14);
    assert_eq!(before_role_change.durable.applied_index, 9);
    assert_eq!(before_role_change.durable.next_index, 15);
    assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(before_role_change.durable.snapshot.snapshot_id, 71);
    assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
    assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

    r.become_candidate(12);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.term, 12);
    assert_eq!(after_role_change.live.commit_index, 14);
    assert_eq!(after_role_change.live.applied_index, 9);
    assert_eq!(after_role_change.live.next_index, 15);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 71);
    assert!(after_role_change.live.has_committed_entries_pending_apply);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 12);
    assert_eq!(after_role_change.durable.commit_index, 14);
    assert_eq!(after_role_change.durable.applied_index, 9);
    assert_eq!(after_role_change.durable.next_index, 15);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 71);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let role_change_recovery = r.recovery_state();
    assert_eq!(role_change_recovery.term, 12);
    assert_eq!(role_change_recovery.snapshot.snapshot_id, 71);
    let resumed_after_role_change =
        RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_role_change.status_snapshot().live,
        after_role_change.durable
    );
    assert_eq!(
        role_change_recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 71);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_and_retires_cleanly(
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

    r.append_entries_from_leader(
        9,
        11,
        8,
        vec![
            LogEntry {
                term: 9,
                index: 12,
                payload: vec![131].into(),
            },
            LogEntry {
                term: 9,
                index: 13,
                payload: vec![132].into(),
            },
        ],
        12,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 67,
    });

    let err = r
        .append_entries_from_leader(
            10,
            99,
            9,
            vec![LogEntry {
                term: 10,
                index: 100,
                payload: vec![140].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        10,
        12,
        9,
        vec![
            LogEntry {
                term: 10,
                index: 13,
                payload: vec![141].into(),
            },
            LogEntry {
                term: 10,
                index: 14,
                payload: vec![142].into(),
            },
        ],
        13,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 71,
    });

    let err = r
        .append_entries_from_leader(
            11,
            99,
            10,
            vec![LogEntry {
                term: 11,
                index: 100,
                payload: vec![150].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        11,
        13,
        10,
        vec![
            LogEntry {
                term: 11,
                index: 14,
                payload: vec![151].into(),
            },
            LogEntry {
                term: 11,
                index: 15,
                payload: vec![152].into(),
            },
        ],
        14,
    )
    .unwrap();

    let during_repair = r.status_snapshot();
    assert!(during_repair.has_speculative_tail());
    assert_eq!(during_repair.live.snapshot.snapshot_id, 71);
    assert_eq!(during_repair.durable.snapshot.snapshot_id, 71);
    assert_eq!(during_repair.recovery_gap.next_index_gap, 1);
    assert_eq!(during_repair.recovery_gap.uncommitted_entry_gap, 1);

    r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
        .unwrap();

    let after_commit = r.status_snapshot();
    assert!(after_commit.is_restart_equivalent());
    assert_eq!(after_commit.live.role, Role::Follower);
    assert_eq!(after_commit.live.term, 11);
    assert_eq!(after_commit.live.commit_index, 15);
    assert_eq!(after_commit.live.applied_index, 9);
    assert_eq!(after_commit.live.next_index, 16);
    assert_eq!(after_commit.live.uncommitted_entry_count, 0);
    assert_eq!(after_commit.live.snapshot.snapshot_id, 71);
    assert!(after_commit.live.has_committed_entries_pending_apply);
    assert_eq!(after_commit.durable, after_commit.live);
    assert_eq!(after_commit.recovery_gap.next_index_gap, 0);
    assert_eq!(after_commit.recovery_gap.uncommitted_entry_gap, 0);

    r.mark_applied(15);

    let after_apply = r.status_snapshot();
    assert!(after_apply.is_restart_equivalent());
    assert_eq!(after_apply.live.role, Role::Follower);
    assert_eq!(after_apply.live.term, 11);
    assert_eq!(after_apply.live.commit_index, 15);
    assert_eq!(after_apply.live.applied_index, 15);
    assert_eq!(after_apply.live.next_index, 16);
    assert_eq!(after_apply.live.uncommitted_entry_count, 0);
    assert_eq!(after_apply.live.snapshot.snapshot_id, 71);
    assert!(!after_apply.live.has_committed_entries_pending_apply);
    assert_eq!(after_apply.durable, after_apply.live);
    assert_eq!(after_apply.recovery_gap.next_index_gap, 0);
    assert_eq!(after_apply.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 11);
    assert_eq!(recovery.snapshot.snapshot_id, 71);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_apply.live);
    assert_eq!(recovery.progress_as_follower().unwrap(), after_apply.live);
    assert_eq!(r.snapshot_meta().snapshot_id, 71);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_keeps_stale_installs_inert_through_commit_and_apply(
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

    r.append_entries_from_leader(
        9,
        11,
        8,
        vec![
            LogEntry {
                term: 9,
                index: 12,
                payload: vec![131].into(),
            },
            LogEntry {
                term: 9,
                index: 13,
                payload: vec![132].into(),
            },
        ],
        12,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 67,
    });

    let err = r
        .append_entries_from_leader(
            10,
            99,
            9,
            vec![LogEntry {
                term: 10,
                index: 100,
                payload: vec![140].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        10,
        12,
        9,
        vec![
            LogEntry {
                term: 10,
                index: 13,
                payload: vec![141].into(),
            },
            LogEntry {
                term: 10,
                index: 14,
                payload: vec![142].into(),
            },
        ],
        13,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 71,
    });

    let err = r
        .append_entries_from_leader(
            11,
            99,
            10,
            vec![LogEntry {
                term: 11,
                index: 100,
                payload: vec![150].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        11,
        13,
        10,
        vec![
            LogEntry {
                term: 11,
                index: 14,
                payload: vec![151].into(),
            },
            LogEntry {
                term: 11,
                index: 15,
                payload: vec![152].into(),
            },
        ],
        14,
    )
    .unwrap();

    let baseline = r.status_snapshot();
    assert!(baseline.has_speculative_tail());
    assert_eq!(baseline.live.snapshot.snapshot_id, 71);
    assert_eq!(baseline.durable.snapshot.snapshot_id, 71);
    assert_eq!(baseline.recovery_gap.next_index_gap, 1);
    assert_eq!(baseline.recovery_gap.uncommitted_entry_gap, 1);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 999,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 1000,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 15,
        last_included_term: 5,
        snapshot_id: 1001,
    });

    let during_repair = r.status_snapshot();
    assert_eq!(during_repair, baseline);

    r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
        .unwrap();

    let committed_status = r.status_snapshot();
    assert!(committed_status.is_restart_equivalent());
    assert_eq!(committed_status.live.snapshot.snapshot_id, 71);
    assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
    assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 1002,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 1003,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 15,
        last_included_term: 5,
        snapshot_id: 1004,
    });

    let after_commit_stale = r.status_snapshot();
    assert_eq!(after_commit_stale, committed_status);

    r.mark_applied(15);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 71);
    assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
    assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 1005,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 1006,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 15,
        last_included_term: 5,
        snapshot_id: 1007,
    });

    let after_apply_stale = r.status_snapshot();
    assert_eq!(after_apply_stale, applied_status);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 11);
    assert_eq!(recovery.snapshot.snapshot_id, 71);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, applied_status.live);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        applied_status.live
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 71);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_keeps_gap_shape_and_retires_on_new_identity(
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

    r.append_entries_from_leader(
        9,
        11,
        8,
        vec![
            LogEntry {
                term: 9,
                index: 12,
                payload: vec![131].into(),
            },
            LogEntry {
                term: 9,
                index: 13,
                payload: vec![132].into(),
            },
        ],
        12,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 67,
    });

    let err = r
        .append_entries_from_leader(
            10,
            99,
            9,
            vec![LogEntry {
                term: 10,
                index: 100,
                payload: vec![140].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        10,
        12,
        9,
        vec![
            LogEntry {
                term: 10,
                index: 13,
                payload: vec![141].into(),
            },
            LogEntry {
                term: 10,
                index: 14,
                payload: vec![142].into(),
            },
        ],
        13,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 71,
    });

    let err = r
        .append_entries_from_leader(
            11,
            99,
            10,
            vec![LogEntry {
                term: 11,
                index: 100,
                payload: vec![150].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        11,
        13,
        10,
        vec![
            LogEntry {
                term: 11,
                index: 14,
                payload: vec![151].into(),
            },
            LogEntry {
                term: 11,
                index: 15,
                payload: vec![152].into(),
            },
        ],
        14,
    )
    .unwrap();

    let before_refresh = r.status_snapshot();
    assert!(before_refresh.has_speculative_tail());
    assert_eq!(before_refresh.live.snapshot.snapshot_id, 71);
    assert_eq!(before_refresh.durable.snapshot.snapshot_id, 71);
    assert_eq!(before_refresh.recovery_gap.next_index_gap, 1);
    assert_eq!(before_refresh.recovery_gap.uncommitted_entry_gap, 1);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 73,
    });

    let after_refresh = r.status_snapshot();
    assert!(after_refresh.has_speculative_tail());
    assert_eq!(after_refresh.live.term, 11);
    assert_eq!(after_refresh.live.commit_index, 14);
    assert_eq!(after_refresh.live.applied_index, 9);
    assert_eq!(after_refresh.live.next_index, 16);
    assert_eq!(after_refresh.live.uncommitted_entry_count, 1);
    assert_eq!(after_refresh.live.snapshot.snapshot_id, 73);
    assert_eq!(after_refresh.durable.term, 11);
    assert_eq!(after_refresh.durable.commit_index, 14);
    assert_eq!(after_refresh.durable.applied_index, 9);
    assert_eq!(after_refresh.durable.next_index, 15);
    assert_eq!(after_refresh.durable.uncommitted_entry_count, 0);
    assert_eq!(after_refresh.durable.snapshot.snapshot_id, 73);
    assert_eq!(after_refresh.recovery_gap.next_index_gap, 1);
    assert_eq!(after_refresh.recovery_gap.uncommitted_entry_gap, 1);

    let refreshed_recovery = r.recovery_state();
    assert_eq!(refreshed_recovery.term, 11);
    assert_eq!(refreshed_recovery.snapshot.snapshot_id, 73);
    let resumed_during_refreshed_repair =
        RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_refreshed_repair.status_snapshot().live,
        after_refresh.durable
    );
    assert_eq!(
        refreshed_recovery.progress_as_follower().unwrap(),
        after_refresh.durable
    );

    r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
        .unwrap();

    let committed_status = r.status_snapshot();
    assert!(committed_status.is_restart_equivalent());
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.term, 11);
    assert_eq!(committed_status.live.commit_index, 15);
    assert_eq!(committed_status.live.applied_index, 9);
    assert_eq!(committed_status.live.next_index, 16);
    assert_eq!(committed_status.live.uncommitted_entry_count, 0);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 73);
    assert_eq!(committed_status.durable, committed_status.live);
    assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
    assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

    let committed_recovery = r.recovery_state();
    assert_eq!(committed_recovery.term, 11);
    assert_eq!(committed_recovery.snapshot.snapshot_id, 73);
    let resumed_after_commit =
        RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_commit.status_snapshot().live,
        committed_status.live
    );
    assert_eq!(
        committed_recovery.progress_as_follower().unwrap(),
        committed_status.live
    );

    r.mark_applied(15);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.term, 11);
    assert_eq!(applied_status.live.commit_index, 15);
    assert_eq!(applied_status.live.applied_index, 15);
    assert_eq!(applied_status.live.next_index, 16);
    assert_eq!(applied_status.live.uncommitted_entry_count, 0);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 73);
    assert_eq!(applied_status.durable, applied_status.live);
    assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
    assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.term, 11);
    assert_eq!(applied_recovery.snapshot.snapshot_id, 73);
    let resumed_after_apply =
        RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_apply.status_snapshot().live,
        applied_status.live
    );
    assert_eq!(
        applied_recovery.progress_as_follower().unwrap(),
        applied_status.live
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 73);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_role_handoff_discards_only_fresh_tail(
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

    r.append_entries_from_leader(
        9,
        11,
        8,
        vec![
            LogEntry {
                term: 9,
                index: 12,
                payload: vec![131].into(),
            },
            LogEntry {
                term: 9,
                index: 13,
                payload: vec![132].into(),
            },
        ],
        12,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 67,
    });

    let err = r
        .append_entries_from_leader(
            10,
            99,
            9,
            vec![LogEntry {
                term: 10,
                index: 100,
                payload: vec![140].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        10,
        12,
        9,
        vec![
            LogEntry {
                term: 10,
                index: 13,
                payload: vec![141].into(),
            },
            LogEntry {
                term: 10,
                index: 14,
                payload: vec![142].into(),
            },
        ],
        13,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 71,
    });

    let err = r
        .append_entries_from_leader(
            11,
            99,
            10,
            vec![LogEntry {
                term: 11,
                index: 100,
                payload: vec![150].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        11,
        13,
        10,
        vec![
            LogEntry {
                term: 11,
                index: 14,
                payload: vec![151].into(),
            },
            LogEntry {
                term: 11,
                index: 15,
                payload: vec![152].into(),
            },
        ],
        14,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 73,
    });

    let before_role_change = r.status_snapshot();
    assert!(before_role_change.has_speculative_tail());
    assert_eq!(before_role_change.live.role, Role::Follower);
    assert_eq!(before_role_change.live.term, 11);
    assert_eq!(before_role_change.live.commit_index, 14);
    assert_eq!(before_role_change.live.applied_index, 9);
    assert_eq!(before_role_change.live.next_index, 16);
    assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
    assert_eq!(before_role_change.live.snapshot.snapshot_id, 73);
    assert_eq!(before_role_change.durable.role, Role::Follower);
    assert_eq!(before_role_change.durable.term, 11);
    assert_eq!(before_role_change.durable.commit_index, 14);
    assert_eq!(before_role_change.durable.applied_index, 9);
    assert_eq!(before_role_change.durable.next_index, 15);
    assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(before_role_change.durable.snapshot.snapshot_id, 73);
    assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
    assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

    r.become_candidate(12);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.term, 12);
    assert_eq!(after_role_change.live.commit_index, 14);
    assert_eq!(after_role_change.live.applied_index, 9);
    assert_eq!(after_role_change.live.next_index, 15);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 73);
    assert!(after_role_change.live.has_committed_entries_pending_apply);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 12);
    assert_eq!(after_role_change.durable.commit_index, 14);
    assert_eq!(after_role_change.durable.applied_index, 9);
    assert_eq!(after_role_change.durable.next_index, 15);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 73);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let role_change_recovery = r.recovery_state();
    assert_eq!(role_change_recovery.term, 12);
    assert_eq!(role_change_recovery.snapshot.snapshot_id, 73);
    let resumed_after_role_change =
        RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_role_change.status_snapshot().live,
        after_role_change.durable
    );
    assert_eq!(
        role_change_recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 73);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_and_retires_cleanly(
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

    r.append_entries_from_leader(
        9,
        11,
        8,
        vec![
            LogEntry {
                term: 9,
                index: 12,
                payload: vec![131].into(),
            },
            LogEntry {
                term: 9,
                index: 13,
                payload: vec![132].into(),
            },
        ],
        12,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 67,
    });

    let err = r
        .append_entries_from_leader(
            10,
            99,
            9,
            vec![LogEntry {
                term: 10,
                index: 100,
                payload: vec![140].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        10,
        12,
        9,
        vec![
            LogEntry {
                term: 10,
                index: 13,
                payload: vec![141].into(),
            },
            LogEntry {
                term: 10,
                index: 14,
                payload: vec![142].into(),
            },
        ],
        13,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 71,
    });

    let err = r
        .append_entries_from_leader(
            11,
            99,
            10,
            vec![LogEntry {
                term: 11,
                index: 100,
                payload: vec![150].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        11,
        13,
        10,
        vec![
            LogEntry {
                term: 11,
                index: 14,
                payload: vec![151].into(),
            },
            LogEntry {
                term: 11,
                index: 15,
                payload: vec![152].into(),
            },
        ],
        14,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 73,
    });

    let during_repair = r.status_snapshot();
    assert!(during_repair.has_speculative_tail());
    assert_eq!(during_repair.live.snapshot.snapshot_id, 73);
    assert_eq!(during_repair.durable.snapshot.snapshot_id, 73);
    assert_eq!(during_repair.recovery_gap.next_index_gap, 1);
    assert_eq!(during_repair.recovery_gap.uncommitted_entry_gap, 1);

    r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
        .unwrap();

    let after_commit = r.status_snapshot();
    assert!(after_commit.is_restart_equivalent());
    assert_eq!(after_commit.live.role, Role::Follower);
    assert_eq!(after_commit.live.term, 11);
    assert_eq!(after_commit.live.commit_index, 15);
    assert_eq!(after_commit.live.applied_index, 9);
    assert_eq!(after_commit.live.next_index, 16);
    assert_eq!(after_commit.live.uncommitted_entry_count, 0);
    assert_eq!(after_commit.live.snapshot.snapshot_id, 73);
    assert!(after_commit.live.has_committed_entries_pending_apply);
    assert_eq!(after_commit.durable, after_commit.live);
    assert_eq!(after_commit.recovery_gap.next_index_gap, 0);
    assert_eq!(after_commit.recovery_gap.uncommitted_entry_gap, 0);

    r.mark_applied(15);

    let after_apply = r.status_snapshot();
    assert!(after_apply.is_restart_equivalent());
    assert_eq!(after_apply.live.role, Role::Follower);
    assert_eq!(after_apply.live.term, 11);
    assert_eq!(after_apply.live.commit_index, 15);
    assert_eq!(after_apply.live.applied_index, 15);
    assert_eq!(after_apply.live.next_index, 16);
    assert_eq!(after_apply.live.uncommitted_entry_count, 0);
    assert_eq!(after_apply.live.snapshot.snapshot_id, 73);
    assert!(!after_apply.live.has_committed_entries_pending_apply);
    assert_eq!(after_apply.durable, after_apply.live);
    assert_eq!(after_apply.recovery_gap.next_index_gap, 0);
    assert_eq!(after_apply.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 11);
    assert_eq!(recovery.snapshot.snapshot_id, 73);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_apply.live);
    assert_eq!(recovery.progress_as_follower().unwrap(), after_apply.live);
    assert_eq!(r.snapshot_meta().snapshot_id, 73);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_keeps_stale_installs_inert_through_commit_and_apply(
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

    r.append_entries_from_leader(
        9,
        11,
        8,
        vec![
            LogEntry {
                term: 9,
                index: 12,
                payload: vec![131].into(),
            },
            LogEntry {
                term: 9,
                index: 13,
                payload: vec![132].into(),
            },
        ],
        12,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 67,
    });

    let err = r
        .append_entries_from_leader(
            10,
            99,
            9,
            vec![LogEntry {
                term: 10,
                index: 100,
                payload: vec![140].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        10,
        12,
        9,
        vec![
            LogEntry {
                term: 10,
                index: 13,
                payload: vec![141].into(),
            },
            LogEntry {
                term: 10,
                index: 14,
                payload: vec![142].into(),
            },
        ],
        13,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 71,
    });

    let err = r
        .append_entries_from_leader(
            11,
            99,
            10,
            vec![LogEntry {
                term: 11,
                index: 100,
                payload: vec![150].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        11,
        13,
        10,
        vec![
            LogEntry {
                term: 11,
                index: 14,
                payload: vec![151].into(),
            },
            LogEntry {
                term: 11,
                index: 15,
                payload: vec![152].into(),
            },
        ],
        14,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 73,
    });

    let baseline = r.status_snapshot();
    assert!(baseline.has_speculative_tail());
    assert_eq!(baseline.live.snapshot.snapshot_id, 73);
    assert_eq!(baseline.durable.snapshot.snapshot_id, 73);
    assert_eq!(baseline.recovery_gap.next_index_gap, 1);
    assert_eq!(baseline.recovery_gap.uncommitted_entry_gap, 1);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 999,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 1000,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 15,
        last_included_term: 5,
        snapshot_id: 1001,
    });

    let during_repair = r.status_snapshot();
    assert_eq!(during_repair, baseline);

    r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
        .unwrap();

    let committed_status = r.status_snapshot();
    assert!(committed_status.is_restart_equivalent());
    assert_eq!(committed_status.live.snapshot.snapshot_id, 73);
    assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
    assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 1002,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 1003,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 15,
        last_included_term: 5,
        snapshot_id: 1004,
    });

    let after_commit_stale = r.status_snapshot();
    assert_eq!(after_commit_stale, committed_status);

    r.mark_applied(15);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 73);
    assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
    assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 1005,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 1006,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 15,
        last_included_term: 5,
        snapshot_id: 1007,
    });

    let after_apply_stale = r.status_snapshot();
    assert_eq!(after_apply_stale, applied_status);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 11);
    assert_eq!(recovery.snapshot.snapshot_id, 73);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, applied_status.live);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        applied_status.live
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 73);
}
