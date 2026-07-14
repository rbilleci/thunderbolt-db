#[test]
fn repair_phase_second_refresh_still_allows_later_advanced_snapshot_replacement() {
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

    let refreshed_status = r.status_snapshot();
    assert!(refreshed_status.has_speculative_tail());
    assert_eq!(refreshed_status.live.snapshot.snapshot_id, 13);
    assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 13);
    assert_eq!(refreshed_status.live.commit_index, 8);
    assert_eq!(refreshed_status.live.applied_index, 8);
    assert_eq!(refreshed_status.live.next_index, 11);
    assert_eq!(refreshed_status.live.uncommitted_entry_count, 2);
    assert_eq!(refreshed_status.durable.next_index, 9);
    assert_eq!(refreshed_status.recovery_gap.next_index_gap, 2);
    assert_eq!(refreshed_status.recovery_gap.uncommitted_entry_gap, 2);

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

    let replaced_recovery = r.recovery_state();
    assert_eq!(replaced_recovery.snapshot.snapshot_id, 53);
    let resumed_during_replaced_repair =
        RaftReplicator::resume_as_follower(3, replaced_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_replaced_repair.status_snapshot().live,
        replaced_status.durable
    );
    assert_eq!(
        replaced_recovery.progress_as_follower().unwrap(),
        replaced_status.durable
    );

    r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
        .unwrap();
    let committed_status = r.status_snapshot();
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert!(!committed_status.has_speculative_tail());
    assert_eq!(committed_status.live.snapshot.snapshot_id, 53);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 53);

    r.mark_applied(10);
    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 53);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 53);
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
    assert_eq!(r.snapshot_meta().snapshot_id, 53);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_survives_role_change_tail_discard() {
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

    r.become_candidate(7);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.term, 7);
    assert_eq!(after_role_change.live.commit_index, 9);
    assert_eq!(after_role_change.live.applied_index, 9);
    assert_eq!(after_role_change.live.next_index, 10);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 53);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 7);
    assert_eq!(after_role_change.durable.commit_index, 9);
    assert_eq!(after_role_change.durable.applied_index, 9);
    assert_eq!(after_role_change.durable.next_index, 10);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 53);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 7);
    assert_eq!(recovery.snapshot.snapshot_id, 53);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 53);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_updates_identity_without_perturbing_gap(
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

    let replaced_status = r.status_snapshot();
    assert!(replaced_status.has_speculative_tail());
    let replaced_gap = replaced_status.recovery_gap.clone();
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

    r.install_snapshot(SnapshotMeta {
        last_included_index: 9,
        last_included_term: 6,
        snapshot_id: 47,
    });

    let refreshed_status = r.status_snapshot();
    assert!(refreshed_status.has_speculative_tail());
    assert_eq!(refreshed_status.live.snapshot.snapshot_id, 47);
    assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 47);
    assert_eq!(refreshed_status.live.term, replaced_status.live.term);
    assert_eq!(
        refreshed_status.live.commit_index,
        replaced_status.live.commit_index
    );
    assert_eq!(
        refreshed_status.live.applied_index,
        replaced_status.live.applied_index
    );
    assert_eq!(
        refreshed_status.live.next_index,
        replaced_status.live.next_index
    );
    assert_eq!(
        refreshed_status.live.uncommitted_entry_count,
        replaced_status.live.uncommitted_entry_count
    );
    assert_eq!(
        refreshed_status.durable.commit_index,
        replaced_status.durable.commit_index
    );
    assert_eq!(
        refreshed_status.durable.applied_index,
        replaced_status.durable.applied_index
    );
    assert_eq!(
        refreshed_status.durable.next_index,
        replaced_status.durable.next_index
    );
    assert_eq!(
        refreshed_status.durable.uncommitted_entry_count,
        replaced_status.durable.uncommitted_entry_count
    );
    assert_eq!(refreshed_status.recovery_gap, replaced_gap);

    let refreshed_recovery = r.recovery_state();
    assert_eq!(refreshed_recovery.snapshot.snapshot_id, 47);
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

    r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
        .unwrap();
    let committed_status = r.status_snapshot();
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert!(!committed_status.has_speculative_tail());
    assert_eq!(committed_status.live.snapshot.snapshot_id, 47);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 47);

    r.mark_applied(10);
    let applied_status = r.status_snapshot();
    let applied_recovery = r.recovery_state();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 47);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 47);
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
    assert_eq!(r.snapshot_meta().snapshot_id, 47);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_survives_role_change_tail_discard() {
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
    assert_eq!(refreshed_status.live.commit_index, 9);
    assert_eq!(refreshed_status.live.applied_index, 9);
    assert_eq!(refreshed_status.live.next_index, 11);
    assert_eq!(refreshed_status.durable.next_index, 10);

    r.become_candidate(7);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.term, 7);
    assert_eq!(after_role_change.live.commit_index, 9);
    assert_eq!(after_role_change.live.applied_index, 9);
    assert_eq!(after_role_change.live.next_index, 10);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 47);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 7);
    assert_eq!(after_role_change.durable.commit_index, 9);
    assert_eq!(after_role_change.durable.applied_index, 9);
    assert_eq!(after_role_change.durable.next_index, 10);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 47);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 7);
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
fn repair_phase_second_refresh_advanced_replacement_refresh_preserves_identity_through_commit_and_apply(
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
    assert_eq!(refreshed_status.live.commit_index, 9);
    assert_eq!(refreshed_status.live.applied_index, 9);
    assert_eq!(refreshed_status.live.next_index, 11);
    assert_eq!(refreshed_status.live.uncommitted_entry_count, 1);
    assert_eq!(refreshed_status.durable.commit_index, 9);
    assert_eq!(refreshed_status.durable.applied_index, 9);
    assert_eq!(refreshed_status.durable.next_index, 10);
    assert_eq!(refreshed_status.durable.uncommitted_entry_count, 0);
    assert_eq!(refreshed_status.recovery_gap.next_index_gap, 1);
    assert_eq!(refreshed_status.recovery_gap.uncommitted_entry_gap, 1);

    let refreshed_recovery = r.recovery_state();
    assert_eq!(refreshed_recovery.snapshot.snapshot_id, 47);
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

    r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
        .unwrap();
    let committed_status = r.status_snapshot();
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert!(!committed_status.has_speculative_tail());
    assert_eq!(committed_status.live.snapshot.snapshot_id, 47);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 47);
    assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
    assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

    let committed_recovery = r.recovery_state();
    assert_eq!(committed_recovery.snapshot.snapshot_id, 47);
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

    r.mark_applied(10);
    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live.snapshot.snapshot_id, 47);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 47);
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.snapshot.snapshot_id, 47);
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
    assert_eq!(r.snapshot_meta().snapshot_id, 47);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_collapses_cleanly_on_newer_leader_rejection(
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

    r.become_follower(7);

    let after_rejection = r.status_snapshot();
    assert!(after_rejection.is_restart_equivalent());
    assert_eq!(after_rejection.live.role, Role::Follower);
    assert_eq!(after_rejection.live.term, 7);
    assert_eq!(after_rejection.live.commit_index, 9);
    assert_eq!(after_rejection.live.applied_index, 9);
    assert_eq!(after_rejection.live.next_index, 10);
    assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
    assert_eq!(after_rejection.live.snapshot.snapshot_id, 47);
    assert_eq!(after_rejection.durable.role, Role::Follower);
    assert_eq!(after_rejection.durable.term, 7);
    assert_eq!(after_rejection.durable.commit_index, 9);
    assert_eq!(after_rejection.durable.applied_index, 9);
    assert_eq!(after_rejection.durable.next_index, 10);
    assert_eq!(after_rejection.durable.uncommitted_entry_count, 0);
    assert_eq!(after_rejection.durable.snapshot.snapshot_id, 47);
    assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
    assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 7);
    assert_eq!(recovery.snapshot.snapshot_id, 47);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_rejection.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_rejection.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 47);
}

#[test]
fn repair_phase_second_refresh_advanced_replacement_refresh_survives_newer_leader_rejection_and_later_repair(
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

    let after_rejection = r.status_snapshot();
    assert!(after_rejection.is_restart_equivalent());
    assert_eq!(after_rejection.live.term, 7);
    assert_eq!(after_rejection.live.snapshot.snapshot_id, 47);
    assert_eq!(after_rejection.durable.snapshot.snapshot_id, 47);
    assert_eq!(after_rejection.live.commit_index, 9);
    assert_eq!(after_rejection.live.applied_index, 9);
    assert_eq!(after_rejection.live.next_index, 10);
    assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
    assert_eq!(after_rejection.durable.commit_index, 9);
    assert_eq!(after_rejection.durable.applied_index, 9);
    assert_eq!(after_rejection.durable.next_index, 10);
    assert_eq!(after_rejection.durable.uncommitted_entry_count, 0);
    assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
    assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

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

    let after_repair = r.status_snapshot();
    assert!(after_repair.has_speculative_tail());
    assert_eq!(after_repair.live.term, 7);
    assert_eq!(after_repair.live.commit_index, 10);
    assert_eq!(after_repair.live.applied_index, 9);
    assert_eq!(after_repair.live.next_index, 12);
    assert_eq!(after_repair.live.uncommitted_entry_count, 1);
    assert_eq!(after_repair.live.snapshot.snapshot_id, 47);
    assert_eq!(after_repair.durable.commit_index, 10);
    assert_eq!(after_repair.durable.applied_index, 9);
    assert_eq!(after_repair.durable.next_index, 11);
    assert_eq!(after_repair.durable.uncommitted_entry_count, 0);
    assert_eq!(after_repair.durable.snapshot.snapshot_id, 47);
    assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
    assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);

    let repair_recovery = r.recovery_state();
    assert_eq!(repair_recovery.term, 7);
    assert_eq!(repair_recovery.snapshot.snapshot_id, 47);
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

    r.append_entries_from_leader(7, 11, 7, Vec::new(), 11)
        .unwrap();

    let committed_status = r.status_snapshot();
    assert!(committed_status.is_restart_equivalent());
    assert!(committed_status.live.has_committed_entries_pending_apply);
    assert_eq!(committed_status.live.snapshot.snapshot_id, 47);
    assert_eq!(committed_status.durable.snapshot.snapshot_id, 47);
    assert_eq!(committed_status.live.commit_index, 11);
    assert_eq!(committed_status.live.applied_index, 9);
    assert_eq!(committed_status.live.next_index, 12);
    assert_eq!(committed_status.live.uncommitted_entry_count, 0);

    r.mark_applied(11);

    let applied_status = r.status_snapshot();
    assert!(applied_status.is_restart_equivalent());
    assert_eq!(applied_status.live, applied_status.durable);
    assert_eq!(applied_status.live.term, 7);
    assert_eq!(applied_status.live.applied_index, 11);
    assert_eq!(applied_status.live.snapshot.snapshot_id, 47);

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.term, 7);
    assert_eq!(applied_recovery.snapshot.snapshot_id, 47);
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
    assert_eq!(r.snapshot_meta().snapshot_id, 47);
}
