#[test]
fn repair_phase_second_refresh_advanced_replacement_collapses_cleanly_on_newer_leader_rejection() {
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

    let replaced_status = r.status_snapshot();
    assert!(replaced_status.has_speculative_tail());
    assert_eq!(replaced_status.live.snapshot.snapshot_id, 53);
    assert_eq!(replaced_status.durable.snapshot.snapshot_id, 53);
    assert_eq!(replaced_status.live.commit_index, 9);
    assert_eq!(replaced_status.live.applied_index, 9);
    assert_eq!(replaced_status.live.next_index, 11);
    assert_eq!(replaced_status.live.uncommitted_entry_count, 1);
    assert_eq!(replaced_status.durable.commit_index, 9);
    assert_eq!(replaced_status.durable.applied_index, 9);
    assert_eq!(replaced_status.durable.next_index, 10);
    assert_eq!(replaced_status.durable.uncommitted_entry_count, 0);
    assert_eq!(replaced_status.recovery_gap.next_index_gap, 1);
    assert_eq!(replaced_status.recovery_gap.uncommitted_entry_gap, 1);

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

    let after_reject = r.status_snapshot();
    assert!(after_reject.is_restart_equivalent());
    assert_eq!(after_reject.live.term, 7);
    assert_eq!(after_reject.live.snapshot.snapshot_id, 53);
    assert_eq!(after_reject.durable.snapshot.snapshot_id, 53);
    assert_eq!(after_reject.live.commit_index, 9);
    assert_eq!(after_reject.live.applied_index, 9);
    assert_eq!(after_reject.live.next_index, 10);
    assert_eq!(after_reject.live.uncommitted_entry_count, 0);
    assert_eq!(after_reject.durable.commit_index, 9);
    assert_eq!(after_reject.durable.applied_index, 9);
    assert_eq!(after_reject.durable.next_index, 10);
    assert_eq!(after_reject.durable.uncommitted_entry_count, 0);
    assert_eq!(after_reject.recovery_gap.next_index_gap, 0);
    assert_eq!(after_reject.recovery_gap.uncommitted_entry_gap, 0);
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    let after_reject_recovery = r.recovery_state();
    assert_eq!(after_reject_recovery.term, 7);
    assert_eq!(after_reject_recovery.snapshot.snapshot_id, 53);
    let resumed_after_reject =
        RaftReplicator::resume_as_follower(3, after_reject_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_reject.status_snapshot().live,
        after_reject.durable
    );
    assert_eq!(
        after_reject_recovery.progress_as_follower().unwrap(),
        after_reject.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 53);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_keeps_stale_installs_inert_through_commit_and_apply(
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

    let repair_status = r.status_snapshot();
    let repair_recovery = r.recovery_state();
    let repair_gap = r.recovery_progress_gap();
    assert!(repair_status.has_speculative_tail());
    assert_eq!(repair_status.live.snapshot.snapshot_id, 53);
    assert_eq!(repair_status.durable.snapshot.snapshot_id, 53);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 97,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 98,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 10,
        last_included_term: 5,
        snapshot_id: 99,
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

    r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
        .unwrap();
    let commit_status = r.status_snapshot();
    let commit_recovery = r.recovery_state();
    let commit_gap = r.recovery_progress_gap();
    assert!(commit_status.live.has_committed_entries_pending_apply);
    assert!(!commit_status.has_speculative_tail());
    assert_eq!(commit_status.live.snapshot.snapshot_id, 53);
    assert_eq!(commit_status.durable.snapshot.snapshot_id, 53);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 107,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 108,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 10,
        last_included_term: 5,
        snapshot_id: 109,
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

    r.mark_applied(10);
    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    let applied_gap = r.recovery_progress_gap();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 53);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 53);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 117,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 118,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 10,
        last_included_term: 5,
        snapshot_id: 119,
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
    assert_eq!(r.snapshot_meta().snapshot_id, 53);
}

#[test]
fn repair_phase_advanced_snapshot_second_refresh_keeps_stale_installs_inert_through_commit_and_apply(
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

    let repair_status = r.status_snapshot();
    let repair_recovery = r.recovery_state();
    let repair_gap = r.recovery_progress_gap();
    assert!(repair_status.has_speculative_tail());
    assert_eq!(repair_status.live.snapshot.snapshot_id, 13);
    assert_eq!(repair_status.durable.snapshot.snapshot_id, 13);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 97,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 98,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 99,
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

    r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
        .unwrap();
    let commit_status = r.status_snapshot();
    let commit_recovery = r.recovery_state();
    let commit_gap = r.recovery_progress_gap();
    assert!(commit_status.live.has_committed_entries_pending_apply);
    assert!(!commit_status.has_speculative_tail());
    assert_eq!(commit_status.live.snapshot.snapshot_id, 13);
    assert_eq!(commit_status.durable.snapshot.snapshot_id, 13);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 107,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 108,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 109,
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

    r.mark_applied(9);
    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    let applied_gap = r.recovery_progress_gap();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 13);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 13);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 117,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 118,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 119,
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
    assert_eq!(r.snapshot_meta().snapshot_id, 13);
}

#[test]
fn repair_phase_advanced_snapshot_second_refresh_survives_role_change_tail_discard() {
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

    let before_role_change = r.status_snapshot();
    assert!(before_role_change.has_speculative_tail());
    assert_eq!(before_role_change.live.snapshot.snapshot_id, 13);
    assert_eq!(before_role_change.durable.snapshot.snapshot_id, 13);
    assert_eq!(before_role_change.live.commit_index, 8);
    assert_eq!(before_role_change.live.applied_index, 8);
    assert_eq!(before_role_change.live.next_index, 10);
    assert_eq!(before_role_change.durable.next_index, 9);

    r.become_candidate(7);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.term, 7);
    assert_eq!(after_role_change.live.commit_index, 8);
    assert_eq!(after_role_change.live.applied_index, 8);
    assert_eq!(after_role_change.live.next_index, 9);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 13);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 7);
    assert_eq!(after_role_change.durable.commit_index, 8);
    assert_eq!(after_role_change.durable.applied_index, 8);
    assert_eq!(after_role_change.durable.next_index, 9);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 13);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 7);
    assert_eq!(recovery.snapshot.snapshot_id, 13);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 13);
}

#[test]
fn repair_phase_advanced_snapshot_refresh_keeps_stale_installs_inert_through_commit_and_apply() {
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

    let repair_status = r.status_snapshot();
    let repair_recovery = r.recovery_state();
    let repair_gap = r.recovery_progress_gap();
    assert!(repair_status.has_speculative_tail());
    assert_eq!(repair_status.live.snapshot.snapshot_id, 17);
    assert_eq!(repair_status.durable.snapshot.snapshot_id, 17);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 97,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 98,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 99,
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

    r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
        .unwrap();
    let commit_status = r.status_snapshot();
    let commit_recovery = r.recovery_state();
    let commit_gap = r.recovery_progress_gap();
    assert!(commit_status.live.has_committed_entries_pending_apply);
    assert!(!commit_status.has_speculative_tail());
    assert_eq!(commit_status.live.snapshot.snapshot_id, 17);
    assert_eq!(commit_status.durable.snapshot.snapshot_id, 17);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 107,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 108,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 109,
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

    r.mark_applied(9);
    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    let applied_gap = r.recovery_progress_gap();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 17);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 17);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 117,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 118,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 119,
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
    assert_eq!(r.snapshot_meta().snapshot_id, 17);
}

#[test]
fn repair_phase_advanced_snapshot_stale_installs_remain_noops_during_repair_commit_and_apply() {
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
        ],
        8,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });

    let repair_status = r.status_snapshot();
    let repair_recovery = r.recovery_state();
    let repair_gap = r.recovery_progress_gap();
    assert!(repair_status.has_speculative_tail());
    assert_eq!(repair_status.live.snapshot.snapshot_id, 41);
    assert_eq!(repair_status.durable.snapshot.snapshot_id, 41);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 97,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 98,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 99,
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

    r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
        .unwrap();
    let commit_status = r.status_snapshot();
    let commit_recovery = r.recovery_state();
    let commit_gap = r.recovery_progress_gap();
    assert!(commit_status.live.has_committed_entries_pending_apply);
    assert!(!commit_status.has_speculative_tail());
    assert_eq!(commit_status.live.snapshot.snapshot_id, 41);
    assert_eq!(commit_status.durable.snapshot.snapshot_id, 41);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 107,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 108,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 109,
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

    r.mark_applied(9);
    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    let applied_gap = r.recovery_progress_gap();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 41);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 41);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 117,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 118,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 5,
        snapshot_id: 119,
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
    assert_eq!(r.snapshot_meta().snapshot_id, 41);
}
