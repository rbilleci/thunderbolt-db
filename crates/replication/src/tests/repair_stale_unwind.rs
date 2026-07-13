#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_keeps_stale_installs_inert_through_commit_and_apply(
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

    let repaired_status = r.status_snapshot();
    let repaired_recovery = r.recovery_state();
    let repaired_gap = r.recovery_progress_gap();
    assert!(repaired_status.has_speculative_tail());
    assert_eq!(repaired_status.live.term, 10);
    assert_eq!(repaired_status.live.snapshot.snapshot_id, 67);
    assert_eq!(repaired_status.durable.snapshot.snapshot_id, 67);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 991,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 990,
    });
    assert_eq!(r.status_snapshot(), repaired_status);
    assert_eq!(r.recovery_state(), repaired_recovery);
    assert_eq!(r.recovery_progress_gap(), repaired_gap);
    let resumed_during_repair_after_stale =
        RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_repair_after_stale.status_snapshot().live,
        repaired_status.durable
    );
    assert_eq!(
        repaired_recovery.progress_as_follower().unwrap(),
        repaired_status.durable
    );

    r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
        .unwrap();

    let committed_status = r.status_snapshot();
    let committed_recovery = r.recovery_state();
    let committed_gap = r.recovery_progress_gap();
    assert!(committed_status.is_restart_equivalent());
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.term, 10);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 67);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 67);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 989,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 988,
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

    r.mark_applied(14);

    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    let applied_gap = r.recovery_progress_gap();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live, applied_status.durable);
    assert_eq!(applied_status.live.term, 10);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 67);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 987,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 986,
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
    assert_eq!(r.snapshot_meta().snapshot_id, 67);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_survives_later_repair_keeps_stale_installs_inert_through_commit_and_apply(
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

    let repaired_status = r.status_snapshot();
    let repaired_recovery = r.recovery_state();
    let repaired_gap = r.recovery_progress_gap();
    assert!(repaired_status.has_speculative_tail());
    assert_eq!(repaired_status.live.term, 10);
    assert_eq!(repaired_status.live.snapshot.snapshot_id, 67);
    assert_eq!(repaired_status.durable.snapshot.snapshot_id, 67);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 991,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 990,
    });
    assert_eq!(r.status_snapshot(), repaired_status);
    assert_eq!(r.recovery_state(), repaired_recovery);
    assert_eq!(r.recovery_progress_gap(), repaired_gap);
    let resumed_during_repair_after_stale =
        RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_repair_after_stale.status_snapshot().live,
        repaired_status.durable
    );
    assert_eq!(
        repaired_recovery.progress_as_follower().unwrap(),
        repaired_status.durable
    );

    r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
        .unwrap();

    let committed_status = r.status_snapshot();
    let committed_recovery = r.recovery_state();
    let committed_gap = r.recovery_progress_gap();
    assert!(committed_status.is_restart_equivalent());
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.term, 10);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 67);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 67);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 989,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 988,
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

    r.mark_applied(14);

    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    let applied_gap = r.recovery_progress_gap();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live, applied_status.durable);
    assert_eq!(applied_status.live.term, 10);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 67);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 987,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 986,
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
    assert_eq!(r.snapshot_meta().snapshot_id, 67);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_survives_later_repair_and_retires_cleanly(
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

    let repaired_status = r.status_snapshot();
    assert!(repaired_status.has_speculative_tail());
    assert_eq!(repaired_status.live.term, 9);
    assert_eq!(repaired_status.live.commit_index, 12);
    assert_eq!(repaired_status.live.applied_index, 9);
    assert_eq!(repaired_status.live.next_index, 14);
    assert_eq!(repaired_status.live.uncommitted_entry_count, 1);
    assert_eq!(repaired_status.live.snapshot.snapshot_id, 61);
    assert_eq!(repaired_status.durable.term, 9);
    assert_eq!(repaired_status.durable.commit_index, 12);
    assert_eq!(repaired_status.durable.applied_index, 9);
    assert_eq!(repaired_status.durable.next_index, 13);
    assert_eq!(repaired_status.durable.uncommitted_entry_count, 0);
    assert_eq!(repaired_status.durable.snapshot.snapshot_id, 61);
    assert_eq!(repaired_status.recovery_gap.next_index_gap, 1);
    assert_eq!(repaired_status.recovery_gap.uncommitted_entry_gap, 1);

    r.append_entries_from_leader(9, 13, 9, Vec::new(), 13)
        .unwrap();

    let committed_status = r.status_snapshot();
    assert!(!committed_status.has_speculative_tail());
    assert_eq!(committed_status.live.term, 9);
    assert_eq!(committed_status.live.commit_index, 13);
    assert_eq!(committed_status.live.applied_index, 9);
    assert_eq!(committed_status.live.next_index, 14);
    assert_eq!(committed_status.live.uncommitted_entry_count, 0);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 61);
    assert_eq!(committed_status.durable, committed_status.live);
    assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
    assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

    let committed_recovery = r.recovery_state();
    assert_eq!(committed_recovery.term, 9);
    assert_eq!(committed_recovery.snapshot.snapshot_id, 61);
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

    r.mark_applied(13);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.term, 9);
    assert_eq!(applied_status.live.commit_index, 13);
    assert_eq!(applied_status.live.applied_index, 13);
    assert_eq!(applied_status.live.next_index, 14);
    assert_eq!(applied_status.live.uncommitted_entry_count, 0);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 61);
    assert_eq!(applied_status.durable, applied_status.live);
    assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
    assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.term, 9);
    assert_eq!(applied_recovery.snapshot.snapshot_id, 61);
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
    assert_eq!(r.snapshot_meta().snapshot_id, 61);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_survives_later_repair_keeps_stale_installs_inert_through_commit_and_apply(
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

    let repaired_status = r.status_snapshot();
    let repaired_recovery = r.recovery_state();
    let repaired_gap = r.recovery_progress_gap();
    assert!(repaired_status.has_speculative_tail());
    assert_eq!(repaired_status.live.term, 9);
    assert_eq!(repaired_status.live.snapshot.snapshot_id, 61);
    assert_eq!(repaired_status.durable.snapshot.snapshot_id, 61);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 999,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 998,
    });
    assert_eq!(r.status_snapshot(), repaired_status);
    assert_eq!(r.recovery_state(), repaired_recovery);
    assert_eq!(r.recovery_progress_gap(), repaired_gap);
    let resumed_during_repair_after_stale =
        RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_repair_after_stale.status_snapshot().live,
        repaired_status.durable
    );
    assert_eq!(
        repaired_recovery.progress_as_follower().unwrap(),
        repaired_status.durable
    );

    r.append_entries_from_leader(9, 13, 9, Vec::new(), 13)
        .unwrap();

    let committed_status = r.status_snapshot();
    let committed_recovery = r.recovery_state();
    let committed_gap = r.recovery_progress_gap();
    assert!(committed_status.is_restart_equivalent());
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.term, 9);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 61);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 61);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 996,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 995,
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

    r.mark_applied(13);

    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    let applied_gap = r.recovery_progress_gap();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live, applied_status.durable);
    assert_eq!(applied_status.live.term, 9);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 61);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 993,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 992,
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
    assert_eq!(r.snapshot_meta().snapshot_id, 61);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_keeps_stale_installs_inert_through_commit_and_apply(
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
    assert_eq!(repair_status.live.snapshot.snapshot_id, 47);
    assert_eq!(repair_status.durable.snapshot.snapshot_id, 47);

    let repair_recovery = r.recovery_state();
    let repair_gap = r.recovery_progress_gap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 999,
    });
    assert_eq!(r.status_snapshot(), repair_status);
    assert_eq!(r.recovery_state(), repair_recovery);
    assert_eq!(r.recovery_progress_gap(), repair_gap);
    let resumed_during_repair_after_stale =
        RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_repair_after_stale.status_snapshot().live,
        repair_status.durable
    );
    assert_eq!(
        repair_recovery.progress_as_follower().unwrap(),
        repair_status.durable
    );

    r.append_entries_from_leader(7, 11, 7, Vec::new(), 11)
        .unwrap();

    let commit_status = r.status_snapshot();
    assert!(commit_status.is_restart_equivalent());
    assert!(commit_status.live.has_committed_entries_pending_apply);
    assert_eq!(commit_status.live.snapshot.snapshot_id, 47);
    assert_eq!(commit_status.durable.snapshot.snapshot_id, 47);

    let commit_recovery = r.recovery_state();
    let commit_gap = r.recovery_progress_gap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 46,
    });
    assert_eq!(r.status_snapshot(), commit_status);
    assert_eq!(r.recovery_state(), commit_recovery);
    assert_eq!(r.recovery_progress_gap(), commit_gap);
    let resumed_after_commit_stale =
        RaftReplicator::resume_as_follower(3, commit_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_commit_stale.status_snapshot().live,
        commit_status.durable
    );
    assert_eq!(
        commit_recovery.progress_as_follower().unwrap(),
        commit_status.durable
    );

    r.mark_applied(11);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live, applied_status.durable);
    assert_eq!(applied_status.live.term, 7);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 47);

    let applied_recovery = r.recovery_state();
    let applied_gap = r.recovery_progress_gap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 45,
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
    assert_eq!(r.snapshot_meta().snapshot_id, 47);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_keeps_stale_installs_inert_through_commit_and_apply(
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

    let refreshed_status = r.status_snapshot();
    assert!(refreshed_status.has_speculative_tail());
    assert_eq!(refreshed_status.live.snapshot.snapshot_id, 47);
    assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 47);
    assert_eq!(refreshed_status.live.next_index, 11);
    assert_eq!(refreshed_status.durable.next_index, 10);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 999,
    });
    let after_stale_repair = r.status_snapshot();
    assert_eq!(after_stale_repair, refreshed_status);

    r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
        .unwrap();
    let committed_status = r.status_snapshot();
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 47);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 47);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 46,
    });
    let after_stale_commit = r.status_snapshot();
    assert_eq!(after_stale_commit, committed_status);

    r.mark_applied(10);
    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 47);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 47);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 45,
    });
    let after_stale_apply = r.status_snapshot();
    assert_eq!(after_stale_apply, applied_status);

    let recovery = r.recovery_state();
    assert_eq!(recovery.snapshot.snapshot_id, 47);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, applied_status.live);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        applied_status.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 47);
}
