#[test]
fn repair_phase_advanced_snapshot_replaces_durable_identity_and_preserves_fresh_suffix() {
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

    let repair_status = r.status_snapshot();
    assert!(repair_status.has_speculative_tail());
    assert_eq!(repair_status.live.snapshot.snapshot_id, 29);
    assert_eq!(repair_status.durable.snapshot.snapshot_id, 29);
    assert_eq!(repair_status.live.commit_index, 8);
    assert_eq!(repair_status.live.applied_index, 7);
    assert_eq!(repair_status.live.next_index, 10);
    assert_eq!(repair_status.durable.commit_index, 8);
    assert_eq!(repair_status.durable.applied_index, 7);
    assert_eq!(repair_status.durable.next_index, 9);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 41,
    });

    let advanced_status = r.status_snapshot();
    assert!(advanced_status.has_speculative_tail());
    assert_eq!(advanced_status.live.term, 6);
    assert_eq!(advanced_status.live.snapshot.snapshot_id, 41);
    assert_eq!(advanced_status.durable.snapshot.snapshot_id, 41);
    assert_eq!(advanced_status.live.commit_index, 8);
    assert_eq!(advanced_status.live.applied_index, 8);
    assert_eq!(advanced_status.live.next_index, 10);
    assert_eq!(advanced_status.live.uncommitted_entry_count, 1);
    assert_eq!(advanced_status.durable.commit_index, 8);
    assert_eq!(advanced_status.durable.applied_index, 8);
    assert_eq!(advanced_status.durable.next_index, 9);
    assert_eq!(advanced_status.durable.uncommitted_entry_count, 0);
    assert_eq!(advanced_status.recovery_gap.next_index_gap, 1);
    assert_eq!(advanced_status.recovery_gap.uncommitted_entry_gap, 1);

    let advanced_recovery = r.recovery_state();
    assert_eq!(advanced_recovery.snapshot.snapshot_id, 41);
    let resumed_during_advanced_repair =
        RaftReplicator::resume_as_follower(3, advanced_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_advanced_repair.status_snapshot().live,
        advanced_status.durable
    );
    assert_eq!(
        advanced_recovery.progress_as_follower().unwrap(),
        advanced_status.durable
    );

    r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
        .unwrap();
    let committed_status = r.status_snapshot();
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert!(!committed_status.has_speculative_tail());
    assert_eq!(committed_status.live.snapshot.snapshot_id, 41);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 41);

    r.mark_applied(9);
    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 41);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 41);
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
    assert_eq!(r.snapshot_meta().snapshot_id, 41);
}

#[test]
fn repair_phase_advanced_snapshot_same_frontier_refresh_updates_identity_without_perturbing_gap() {
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

    let advanced_status = r.status_snapshot();
    assert!(advanced_status.has_speculative_tail());
    let advanced_gap = advanced_status.recovery_gap.clone();
    assert_eq!(advanced_status.live.snapshot.snapshot_id, 41);
    assert_eq!(advanced_status.durable.snapshot.snapshot_id, 41);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 6,
        snapshot_id: 17,
    });

    let refreshed_status = r.status_snapshot();
    assert!(refreshed_status.has_speculative_tail());
    assert_eq!(refreshed_status.live.term, 6);
    assert_eq!(refreshed_status.live.snapshot.snapshot_id, 17);
    assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 17);
    assert_eq!(
        refreshed_status.live.commit_index,
        advanced_status.live.commit_index
    );
    assert_eq!(
        refreshed_status.live.applied_index,
        advanced_status.live.applied_index
    );
    assert_eq!(
        refreshed_status.live.next_index,
        advanced_status.live.next_index
    );
    assert_eq!(
        refreshed_status.live.uncommitted_entry_count,
        advanced_status.live.uncommitted_entry_count
    );
    assert_eq!(
        refreshed_status.durable.commit_index,
        advanced_status.durable.commit_index
    );
    assert_eq!(
        refreshed_status.durable.applied_index,
        advanced_status.durable.applied_index
    );
    assert_eq!(
        refreshed_status.durable.next_index,
        advanced_status.durable.next_index
    );
    assert_eq!(
        refreshed_status.durable.uncommitted_entry_count,
        advanced_status.durable.uncommitted_entry_count
    );
    assert_eq!(refreshed_status.recovery_gap, advanced_gap);

    let refreshed_recovery = r.recovery_state();
    assert_eq!(refreshed_recovery.snapshot.snapshot_id, 17);
    let resumed_during_refreshed_repair =
        RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_refreshed_repair.status_snapshot().live,
        refreshed_status.durable
    );
    assert_eq!(
        refreshed_recovery.progress_as_follower().unwrap(),
        refreshed_status.durable
    );

    r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
        .unwrap();
    let committed_status = r.status_snapshot();
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert!(!committed_status.has_speculative_tail());
    assert_eq!(committed_status.live.snapshot.snapshot_id, 17);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 17);

    r.mark_applied(9);
    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 17);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 17);
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
    assert_eq!(r.snapshot_meta().snapshot_id, 17);
}

#[test]
fn repair_phase_advanced_snapshot_refresh_survives_role_change_tail_discard() {
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

    let before_role_change = r.status_snapshot();
    assert!(before_role_change.has_speculative_tail());
    assert_eq!(before_role_change.live.snapshot.snapshot_id, 17);
    assert_eq!(before_role_change.durable.snapshot.snapshot_id, 17);
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
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 17);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 7);
    assert_eq!(after_role_change.durable.commit_index, 8);
    assert_eq!(after_role_change.durable.applied_index, 8);
    assert_eq!(after_role_change.durable.next_index, 9);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 17);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 7);
    assert_eq!(recovery.snapshot.snapshot_id, 17);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 17);
}

#[test]
fn repair_phase_advanced_snapshot_refresh_collapses_cleanly_on_newer_leader_rejection() {
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

    let refreshed_status = r.status_snapshot();
    assert!(refreshed_status.has_speculative_tail());
    assert_eq!(refreshed_status.live.snapshot.snapshot_id, 17);
    assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 17);
    assert_eq!(refreshed_status.live.commit_index, 8);
    assert_eq!(refreshed_status.live.applied_index, 8);
    assert_eq!(refreshed_status.live.next_index, 10);
    assert_eq!(refreshed_status.durable.commit_index, 8);
    assert_eq!(refreshed_status.durable.applied_index, 8);
    assert_eq!(refreshed_status.durable.next_index, 9);

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
    assert_eq!(after_reject.live.snapshot.snapshot_id, 17);
    assert_eq!(after_reject.durable.snapshot.snapshot_id, 17);
    assert_eq!(after_reject.live.commit_index, 8);
    assert_eq!(after_reject.live.applied_index, 8);
    assert_eq!(after_reject.live.next_index, 9);
    assert_eq!(after_reject.live.uncommitted_entry_count, 0);
    assert_eq!(after_reject.durable.commit_index, 8);
    assert_eq!(after_reject.durable.applied_index, 8);
    assert_eq!(after_reject.durable.next_index, 9);
    assert_eq!(after_reject.durable.uncommitted_entry_count, 0);
    assert_eq!(after_reject.recovery_gap.next_index_gap, 0);
    assert_eq!(after_reject.recovery_gap.uncommitted_entry_gap, 0);
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    let after_reject_recovery = r.recovery_state();
    assert_eq!(after_reject_recovery.term, 7);
    assert_eq!(after_reject_recovery.snapshot.snapshot_id, 17);
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
    assert_eq!(r.snapshot_meta().snapshot_id, 17);
}

#[test]
fn repair_phase_advanced_snapshot_second_refresh_still_collapses_cleanly_on_newer_leader_rejection()
{
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

    let refreshed_status = r.status_snapshot();
    assert!(refreshed_status.has_speculative_tail());
    assert_eq!(refreshed_status.live.snapshot.snapshot_id, 13);
    assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 13);
    assert_eq!(refreshed_status.live.commit_index, 8);
    assert_eq!(refreshed_status.live.applied_index, 8);
    assert_eq!(refreshed_status.live.next_index, 10);
    assert_eq!(refreshed_status.durable.commit_index, 8);
    assert_eq!(refreshed_status.durable.applied_index, 8);
    assert_eq!(refreshed_status.durable.next_index, 9);

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
    assert_eq!(after_reject.live.snapshot.snapshot_id, 13);
    assert_eq!(after_reject.durable.snapshot.snapshot_id, 13);
    assert_eq!(after_reject.live.commit_index, 8);
    assert_eq!(after_reject.live.applied_index, 8);
    assert_eq!(after_reject.live.next_index, 9);
    assert_eq!(after_reject.live.uncommitted_entry_count, 0);
    assert_eq!(after_reject.durable.commit_index, 8);
    assert_eq!(after_reject.durable.applied_index, 8);
    assert_eq!(after_reject.durable.next_index, 9);
    assert_eq!(after_reject.durable.uncommitted_entry_count, 0);
    assert_eq!(after_reject.recovery_gap.next_index_gap, 0);
    assert_eq!(after_reject.recovery_gap.uncommitted_entry_gap, 0);
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    let after_reject_recovery = r.recovery_state();
    assert_eq!(after_reject_recovery.term, 7);
    assert_eq!(after_reject_recovery.snapshot.snapshot_id, 13);
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
    assert_eq!(r.snapshot_meta().snapshot_id, 13);
}

#[test]
fn repair_phase_advanced_snapshot_role_change_discards_only_fresh_suffix() {
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

    let before_role_change = r.status_snapshot();
    assert!(before_role_change.has_speculative_tail());
    assert_eq!(before_role_change.live.snapshot.snapshot_id, 41);
    assert_eq!(before_role_change.durable.snapshot.snapshot_id, 41);
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
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 41);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 7);
    assert_eq!(after_role_change.durable.commit_index, 8);
    assert_eq!(after_role_change.durable.applied_index, 8);
    assert_eq!(after_role_change.durable.next_index, 9);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 41);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 7);
    assert_eq!(recovery.snapshot.snapshot_id, 41);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 41);
}

#[test]
fn repair_phase_advanced_snapshot_second_refresh_preserves_identity_through_commit_and_apply() {
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

    let refreshed_status = r.status_snapshot();
    assert!(refreshed_status.has_speculative_tail());
    assert_eq!(refreshed_status.live.snapshot.snapshot_id, 13);
    assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 13);
    assert_eq!(refreshed_status.live.commit_index, 8);
    assert_eq!(refreshed_status.live.applied_index, 8);
    assert_eq!(refreshed_status.live.next_index, 10);
    assert_eq!(refreshed_status.live.uncommitted_entry_count, 1);
    assert_eq!(refreshed_status.durable.commit_index, 8);
    assert_eq!(refreshed_status.durable.applied_index, 8);
    assert_eq!(refreshed_status.durable.next_index, 9);
    assert_eq!(refreshed_status.durable.uncommitted_entry_count, 0);
    assert_eq!(refreshed_status.recovery_gap.next_index_gap, 1);
    assert_eq!(refreshed_status.recovery_gap.uncommitted_entry_gap, 1);

    let refreshed_recovery = r.recovery_state();
    assert_eq!(refreshed_recovery.snapshot.snapshot_id, 13);
    let resumed_during_refreshed_repair =
        RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_refreshed_repair.status_snapshot().live,
        refreshed_status.durable
    );
    assert_eq!(
        refreshed_recovery.progress_as_follower().unwrap(),
        refreshed_status.durable
    );

    r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
        .unwrap();
    let committed_status = r.status_snapshot();
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert!(!committed_status.has_speculative_tail());
    assert_eq!(committed_status.live.snapshot.snapshot_id, 13);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 13);
    assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
    assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

    let committed_recovery = r.recovery_state();
    assert_eq!(committed_recovery.snapshot.snapshot_id, 13);
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

    r.mark_applied(9);
    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 13);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 13);
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.snapshot.snapshot_id, 13);
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
    assert_eq!(r.snapshot_meta().snapshot_id, 13);
}
