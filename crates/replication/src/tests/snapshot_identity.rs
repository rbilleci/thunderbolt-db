#[test]
fn raft_install_snapshot_with_same_index_wrong_term_is_a_progress_no_op() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 2,
    });
    let baseline = r.progress();

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 2,
        snapshot_id: 3,
    });

    assert_eq!(r.progress(), baseline);
    assert_eq!(r.snapshot_meta().snapshot_id, 2);

    r.append_entries_from_leader(
        4,
        5,
        4,
        vec![LogEntry {
            term: 4,
            index: 6,
            payload: vec![9].into(),
        }],
        6,
    )
    .unwrap();

    assert_eq!(r.commit_index(), 6);
    assert_eq!(r.next_index, 7);
}

#[test]
fn raft_install_snapshot_with_higher_index_lower_term_is_a_progress_no_op() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 2,
    });
    let baseline = r.progress();

    r.install_snapshot(SnapshotMeta {
        last_included_index: 6,
        last_included_term: 3,
        snapshot_id: 8,
    });

    assert_eq!(r.progress(), baseline);
    assert_eq!(r.snapshot_meta().snapshot_id, 2);
}

#[test]
fn raft_install_snapshot_gap_is_stable_for_same_frontier_wrong_term() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 2,
    });
    r.append_entries_from_leader(
        4,
        5,
        4,
        vec![LogEntry {
            term: 4,
            index: 6,
            payload: vec![9].into(),
        }],
        5,
    )
    .unwrap();

    let baseline = r.progress();
    let baseline_recovery = r.recovery_progress();
    let baseline_gap = r.recovery_progress_gap();
    assert_eq!(
        baseline_gap,
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 1,
            uncommitted_entry_gap: 1,
        }
    );

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 2,
        snapshot_id: 3,
    });

    assert_eq!(r.progress(), baseline);
    assert_eq!(r.recovery_progress(), baseline_recovery);
    assert_eq!(r.recovery_progress_gap(), baseline_gap);
    assert_eq!(r.snapshot_meta().snapshot_id, 2);
}

#[test]
fn raft_install_snapshot_gap_is_stable_for_higher_frontier_lower_term() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 2,
    });
    r.append_entries_from_leader(
        4,
        5,
        4,
        vec![LogEntry {
            term: 4,
            index: 6,
            payload: vec![9].into(),
        }],
        5,
    )
    .unwrap();

    let baseline = r.progress();
    let baseline_recovery = r.recovery_progress();
    let baseline_gap = r.recovery_progress_gap();

    r.install_snapshot(SnapshotMeta {
        last_included_index: 6,
        last_included_term: 3,
        snapshot_id: 8,
    });

    assert_eq!(r.progress(), baseline);
    assert_eq!(r.recovery_progress(), baseline_recovery);
    assert_eq!(r.recovery_progress_gap(), baseline_gap);
    assert_eq!(r.snapshot_meta().snapshot_id, 2);
}

#[test]
fn raft_install_snapshot_updates_snapshot_id_for_same_frontier_same_term() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 2,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 7,
    });

    assert_eq!(r.snapshot_meta().snapshot_id, 7);
    assert_eq!(r.progress().snapshot.snapshot_id, 7);
}

#[test]
fn raft_install_snapshot_advancing_frontier_replaces_snapshot_identity_exactly() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 4,
    });

    let snapshot = r.snapshot_meta();
    assert_eq!(snapshot.last_included_index, 8);
    assert_eq!(snapshot.last_included_term, 5);
    assert_eq!(snapshot.snapshot_id, 4);
    assert_eq!(r.progress().snapshot, snapshot);
    assert_eq!(r.status_snapshot().live.snapshot, snapshot);
    assert_eq!(r.status_snapshot().durable.snapshot, snapshot);
}

#[test]
fn raft_install_snapshot_advancing_frontier_preserves_compatible_suffix() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(5);

    let t1 = r.propose(vec![1].into()).unwrap();
    let t2 = r.propose(vec![2].into()).unwrap();
    let t3 = r.propose(vec![3].into()).unwrap();

    r.register_follower_ack(t1.index, 1);
    r.register_follower_ack(t2.index, 1);
    assert_eq!(r.commit_index(), t2.index);
    assert!(r.ack_counts.contains_key(&t3.index));

    r.install_snapshot(SnapshotMeta {
        last_included_index: t2.index,
        last_included_term: 5,
        snapshot_id: 29,
    });

    let progress = r.progress();
    assert_eq!(progress.snapshot.snapshot_id, 29);
    assert_eq!(progress.snapshot.last_included_index, t2.index);
    assert_eq!(progress.snapshot.last_included_term, 5);
    assert_eq!(progress.commit_index, t2.index);
    assert_eq!(progress.applied_index, t2.index);
    assert_eq!(progress.next_index, t3.index + 1);
    assert_eq!(progress.uncommitted_entry_count, 1);
    assert!(progress.has_uncommitted_entries);
    assert_eq!(r.entries.len(), 1);
    assert_eq!(r.entries[0].index, t3.index);
    assert!(r.ack_counts.contains_key(&t3.index));

    let status = r.status_snapshot();
    assert!(status.has_speculative_tail());
    assert_eq!(status.live.snapshot.snapshot_id, 29);
    assert_eq!(status.durable.snapshot.snapshot_id, 29);
    assert_eq!(status.recovery_gap.next_index_gap, 1);
    assert_eq!(status.recovery_gap.uncommitted_entry_gap, 1);
}

#[test]
fn same_frontier_same_term_snapshot_refresh_preserves_speculative_gap_semantics() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 2,
    });
    r.append_entries_from_leader(
        4,
        5,
        4,
        vec![LogEntry {
            term: 4,
            index: 6,
            payload: vec![9].into(),
        }],
        5,
    )
    .unwrap();

    let baseline = r.status_snapshot();
    assert!(baseline.has_speculative_tail());
    assert_eq!(baseline.live.snapshot.snapshot_id, 2);
    assert_eq!(baseline.durable.snapshot.snapshot_id, 2);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 7,
    });

    let refreshed = r.status_snapshot();
    assert!(refreshed.has_speculative_tail());
    assert_eq!(refreshed.recovery_gap, baseline.recovery_gap);
    assert_eq!(refreshed.live.commit_index, baseline.live.commit_index);
    assert_eq!(refreshed.live.applied_index, baseline.live.applied_index);
    assert_eq!(refreshed.live.next_index, baseline.live.next_index);
    assert_eq!(
        refreshed.live.uncommitted_entry_count,
        baseline.live.uncommitted_entry_count
    );
    assert_eq!(
        refreshed.durable.commit_index,
        baseline.durable.commit_index
    );
    assert_eq!(
        refreshed.durable.applied_index,
        baseline.durable.applied_index
    );
    assert_eq!(refreshed.durable.next_index, baseline.durable.next_index);
    assert_eq!(
        refreshed.durable.uncommitted_entry_count,
        baseline.durable.uncommitted_entry_count
    );
    assert_eq!(refreshed.live.snapshot.last_included_index, 5);
    assert_eq!(refreshed.live.snapshot.last_included_term, 4);
    assert_eq!(refreshed.durable.snapshot.last_included_index, 5);
    assert_eq!(refreshed.durable.snapshot.last_included_term, 4);
    assert_eq!(refreshed.live.snapshot.snapshot_id, 7);
    assert_eq!(refreshed.durable.snapshot.snapshot_id, 7);
}

#[test]
fn same_frontier_snapshot_refresh_survives_role_change_tail_discard() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 2,
    });
    r.append_entries_from_leader(
        4,
        5,
        4,
        vec![LogEntry {
            term: 4,
            index: 6,
            payload: vec![9].into(),
        }],
        5,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 7,
    });

    let refreshed = r.status_snapshot();
    assert!(refreshed.has_speculative_tail());
    assert_eq!(refreshed.live.snapshot.snapshot_id, 7);
    assert_eq!(refreshed.durable.snapshot.snapshot_id, 7);

    r.become_candidate(5);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.commit_index, 5);
    assert_eq!(after_role_change.live.applied_index, 5);
    assert_eq!(after_role_change.live.next_index, 6);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 7);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, after_role_change.live.term);
    assert_eq!(
        after_role_change.durable.commit_index,
        after_role_change.live.commit_index
    );
    assert_eq!(
        after_role_change.durable.applied_index,
        after_role_change.live.applied_index
    );
    assert_eq!(
        after_role_change.durable.next_index,
        after_role_change.live.next_index
    );
    assert_eq!(
        after_role_change.durable.snapshot,
        after_role_change.live.snapshot
    );
}

#[test]
fn same_frontier_snapshot_refresh_survives_newer_leader_rejection_and_catch_up() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 2,
    });
    r.append_entries_from_leader(
        4,
        5,
        4,
        vec![LogEntry {
            term: 4,
            index: 6,
            payload: vec![6].into(),
        }],
        5,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 7,
    });

    let refreshed = r.status_snapshot();
    assert!(refreshed.has_speculative_tail());
    assert_eq!(refreshed.live.snapshot.snapshot_id, 7);
    assert_eq!(refreshed.durable.snapshot.snapshot_id, 7);

    let err = r
        .append_entries_from_leader(
            5,
            99,
            5,
            vec![LogEntry {
                term: 5,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    let after_reject = r.status_snapshot();
    assert!(after_reject.is_restart_equivalent());
    assert_eq!(after_reject.live.term, 5);
    assert_eq!(after_reject.live.commit_index, 5);
    assert_eq!(after_reject.live.applied_index, 5);
    assert_eq!(after_reject.live.next_index, 6);
    assert_eq!(after_reject.live.snapshot.snapshot_id, 7);
    assert_eq!(after_reject.durable.snapshot.snapshot_id, 7);

    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![60].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![70].into(),
            },
        ],
        6,
    )
    .unwrap();

    let after_append = r.status_snapshot();
    assert!(after_append.has_speculative_tail());
    assert_eq!(after_append.live.term, 5);
    assert_eq!(after_append.live.commit_index, 6);
    assert_eq!(after_append.live.applied_index, 5);
    assert_eq!(after_append.live.next_index, 8);
    assert_eq!(after_append.live.snapshot.snapshot_id, 7);
    assert_eq!(after_append.durable.commit_index, 6);
    assert_eq!(after_append.durable.applied_index, 5);
    assert_eq!(after_append.durable.next_index, 7);
    assert_eq!(after_append.durable.snapshot.snapshot_id, 7);

    r.append_entries_from_leader(5, 7, 5, vec![], 7).unwrap();

    let after_commit = r.status_snapshot();
    assert!(after_commit.is_restart_equivalent());
    assert_eq!(after_commit.live.commit_index, 7);
    assert_eq!(after_commit.live.applied_index, 5);
    assert_eq!(after_commit.live.next_index, 8);
    assert_eq!(after_commit.live.committed_but_unapplied_count, 2);
    assert_eq!(after_commit.live.snapshot.snapshot_id, 7);
    assert_eq!(after_commit.durable.snapshot.snapshot_id, 7);

    r.mark_applied(7);

    let after_apply = r.status_snapshot();
    assert!(after_apply.is_restart_equivalent());
    assert_eq!(after_apply.live, after_apply.durable);
    assert_eq!(after_apply.live.applied_index, 7);
    assert_eq!(after_apply.live.snapshot.snapshot_id, 7);
}

#[test]
fn same_frontier_snapshot_refresh_keeps_recovery_bundle_aligned_through_newer_leader_handoff() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 2,
    });
    r.append_entries_from_leader(
        4,
        5,
        4,
        vec![LogEntry {
            term: 4,
            index: 6,
            payload: vec![6].into(),
        }],
        5,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 7,
    });

    let err = r
        .append_entries_from_leader(
            5,
            99,
            5,
            vec![LogEntry {
                term: 5,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![60].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![70].into(),
            },
        ],
        6,
    )
    .unwrap();

    let after_append = r.status_snapshot();
    assert!(after_append.has_speculative_tail());
    assert_eq!(after_append.durable.snapshot.snapshot_id, 7);

    let recovery = r.recovery_state();
    assert_eq!(recovery.snapshot.snapshot_id, 7);
    let recovery_progress = recovery.progress_as_follower().unwrap();
    assert_eq!(recovery_progress, after_append.durable);

    let resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();
    let resumed_status = resumed.status_snapshot();
    assert_eq!(resumed_status.live, after_append.durable);
    assert_eq!(resumed_status.durable, after_append.durable);
    assert!(resumed_status.is_restart_equivalent());

    r.append_entries_from_leader(5, 7, 5, vec![], 7).unwrap();
    let recovery = r.recovery_state();
    assert_eq!(recovery.snapshot.snapshot_id, 7);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(
        resumed.status_snapshot().live,
        recovery.progress_as_follower().unwrap()
    );
}

#[test]
fn same_frontier_snapshot_refresh_keeps_recovery_gap_explicit_through_newer_leader_handoff() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 2,
    });
    r.append_entries_from_leader(
        4,
        5,
        4,
        vec![LogEntry {
            term: 4,
            index: 6,
            payload: vec![6].into(),
        }],
        5,
    )
    .unwrap();
    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 7,
    });

    assert_eq!(
        r.recovery_progress_gap(),
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 1,
            uncommitted_entry_gap: 1,
        }
    );

    let err = r
        .append_entries_from_leader(
            5,
            99,
            5,
            vec![LogEntry {
                term: 5,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    r.append_entries_from_leader(
        5,
        5,
        4,
        vec![
            LogEntry {
                term: 5,
                index: 6,
                payload: vec![60].into(),
            },
            LogEntry {
                term: 5,
                index: 7,
                payload: vec![70].into(),
            },
        ],
        6,
    )
    .unwrap();
    assert_eq!(
        r.recovery_progress_gap(),
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 1,
            uncommitted_entry_gap: 1,
        }
    );

    r.append_entries_from_leader(5, 7, 5, vec![], 7).unwrap();
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    r.mark_applied(7);
    assert!(r.recovery_progress_gap().is_restart_equivalent());
    assert_eq!(r.recovery_state().snapshot.snapshot_id, 7);
}

#[test]
fn advanced_frontier_snapshot_discards_incompatible_speculative_tail_before_epoch_handoffs() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 4,
        snapshot_id: 11,
    });
    r.append_entries_from_leader(
        4,
        5,
        4,
        vec![
            LogEntry {
                term: 4,
                index: 6,
                payload: vec![6].into(),
            },
            LogEntry {
                term: 4,
                index: 7,
                payload: vec![7].into(),
            },
            LogEntry {
                term: 4,
                index: 8,
                payload: vec![8].into(),
            },
            LogEntry {
                term: 4,
                index: 9,
                payload: vec![9].into(),
            },
        ],
        5,
    )
    .unwrap();

    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 5,
        snapshot_id: 4,
    });

    let refreshed = r.status_snapshot();
    assert!(refreshed.is_restart_equivalent());
    assert_eq!(refreshed.live.snapshot.snapshot_id, 4);
    assert_eq!(refreshed.durable.snapshot.snapshot_id, 4);
    assert_eq!(refreshed.live.snapshot.last_included_index, 8);
    assert_eq!(refreshed.live.snapshot.last_included_term, 5);
    assert_eq!(refreshed.live.commit_index, 8);
    assert_eq!(refreshed.live.applied_index, 8);
    assert_eq!(refreshed.live.next_index, 9);
    assert_eq!(refreshed.recovery_gap.next_index_gap, 0);
    assert_eq!(refreshed.recovery_gap.uncommitted_entry_gap, 0);

    let err = r
        .append_entries_from_leader(
            6,
            99,
            6,
            vec![LogEntry {
                term: 6,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    let after_reject = r.status_snapshot();
    assert!(after_reject.is_restart_equivalent());
    assert_eq!(after_reject.live.term, 6);
    assert_eq!(after_reject.live.snapshot.snapshot_id, 4);
    assert_eq!(after_reject.durable.snapshot.snapshot_id, 4);
    assert_eq!(after_reject.live.commit_index, 8);
    assert_eq!(after_reject.live.applied_index, 8);
    assert_eq!(after_reject.live.next_index, 9);

    r.append_entries_from_leader(
        6,
        8,
        5,
        vec![
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
        9,
    )
    .unwrap();

    let after_append = r.status_snapshot();
    assert!(after_append.has_speculative_tail());
    assert_eq!(after_append.live.snapshot.snapshot_id, 4);
    assert_eq!(after_append.durable.snapshot.snapshot_id, 4);
    assert_eq!(after_append.live.commit_index, 9);
    assert_eq!(after_append.live.applied_index, 8);
    assert_eq!(after_append.live.next_index, 11);
    assert_eq!(after_append.durable.commit_index, 9);
    assert_eq!(after_append.durable.applied_index, 8);
    assert_eq!(after_append.durable.next_index, 10);

    let recovery = r.recovery_state();
    assert_eq!(recovery.snapshot.snapshot_id, 4);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_append.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_append.durable
    );

    r.append_entries_from_leader(6, 10, 6, vec![], 10).unwrap();
    assert!(r.recovery_progress_gap().is_restart_equivalent());
    assert_eq!(r.recovery_state().snapshot.snapshot_id, 4);

    r.mark_applied(10);
    let after_apply = r.status_snapshot();
    assert!(after_apply.is_restart_equivalent());
    assert_eq!(after_apply.live, after_apply.durable);
    assert_eq!(after_apply.live.snapshot.snapshot_id, 4);
}

#[test]
fn compatible_advanced_snapshot_suffix_is_still_discarded_on_newer_leader_rejection() {
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
    assert!(r.status_snapshot().has_speculative_tail());

    let err = r
        .append_entries_from_leader(
            6,
            99,
            6,
            vec![LogEntry {
                term: 6,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    let status = r.status_snapshot();
    assert!(status.is_restart_equivalent());
    assert_eq!(status.live.term, 6);
    assert_eq!(status.live.snapshot.snapshot_id, 29);
    assert_eq!(status.durable.snapshot.snapshot_id, 29);
    assert_eq!(status.live.commit_index, 7);
    assert_eq!(status.live.applied_index, 7);
    assert_eq!(status.live.next_index, 8);
}

#[test]
fn compatible_advanced_snapshot_suffix_keeps_exact_identity_across_resume_and_repair() {
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

    let with_compatible_suffix = r.status_snapshot();
    assert!(with_compatible_suffix.has_speculative_tail());
    assert_eq!(with_compatible_suffix.live.snapshot.snapshot_id, 29);
    assert_eq!(with_compatible_suffix.durable.snapshot.snapshot_id, 29);
    assert_eq!(with_compatible_suffix.live.commit_index, 7);
    assert_eq!(with_compatible_suffix.live.applied_index, 7);
    assert_eq!(with_compatible_suffix.live.next_index, 9);
    assert_eq!(with_compatible_suffix.durable.commit_index, 7);
    assert_eq!(with_compatible_suffix.durable.applied_index, 7);
    assert_eq!(with_compatible_suffix.durable.next_index, 8);

    let recovery = r.recovery_state();
    assert_eq!(recovery.snapshot.snapshot_id, 29);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    let resumed_status = resumed.status_snapshot();
    assert!(resumed_status.is_restart_equivalent());
    assert_eq!(resumed_status.live, with_compatible_suffix.durable);
    assert_eq!(resumed_status.durable, with_compatible_suffix.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        with_compatible_suffix.durable
    );

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

    let after_repair = r.status_snapshot();
    assert!(after_repair.has_speculative_tail());
    assert_eq!(after_repair.live.term, 6);
    assert_eq!(after_repair.live.snapshot.snapshot_id, 29);
    assert_eq!(after_repair.durable.snapshot.snapshot_id, 29);
    assert_eq!(after_repair.live.commit_index, 8);
    assert_eq!(after_repair.live.applied_index, 7);
    assert_eq!(after_repair.live.next_index, 10);
    assert_eq!(after_repair.durable.commit_index, 8);
    assert_eq!(after_repair.durable.applied_index, 7);
    assert_eq!(after_repair.durable.next_index, 9);
    assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
    assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);

    let recovery_after_repair = r.recovery_state();
    assert_eq!(recovery_after_repair.snapshot.snapshot_id, 29);
    let resumed_after_repair =
        RaftReplicator::resume_as_follower(3, recovery_after_repair.clone()).unwrap();
    assert_eq!(
        resumed_after_repair.status_snapshot().live,
        after_repair.durable
    );
    assert_eq!(
        recovery_after_repair.progress_as_follower().unwrap(),
        after_repair.durable
    );
}

#[test]
fn compatible_advanced_snapshot_suffix_survives_role_change_tail_discard() {
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

    let before_role_change = r.status_snapshot();
    assert!(before_role_change.has_speculative_tail());
    assert_eq!(before_role_change.live.snapshot.snapshot_id, 29);
    assert_eq!(before_role_change.durable.snapshot.snapshot_id, 29);

    r.become_candidate(6);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.term, 6);
    assert_eq!(after_role_change.live.commit_index, 7);
    assert_eq!(after_role_change.live.applied_index, 7);
    assert_eq!(after_role_change.live.next_index, 8);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 29);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 6);
    assert_eq!(after_role_change.durable.commit_index, 7);
    assert_eq!(after_role_change.durable.applied_index, 7);
    assert_eq!(after_role_change.durable.next_index, 8);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 29);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 6);
    assert_eq!(recovery.snapshot.snapshot_id, 29);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 29);
}

#[test]
fn compatible_advanced_snapshot_suffix_refresh_keeps_exact_identity_across_resume_repair_and_apply_completion(
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
    let baseline = r.status_snapshot();
    assert!(baseline.has_speculative_tail());
    assert_eq!(baseline.live.snapshot.snapshot_id, 29);
    assert_eq!(baseline.durable.snapshot.snapshot_id, 29);
    assert_eq!(baseline.recovery_gap.next_index_gap, 1);
    assert_eq!(baseline.recovery_gap.uncommitted_entry_gap, 1);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 31,
    });

    let refreshed = r.status_snapshot();
    assert!(refreshed.has_speculative_tail());
    assert_eq!(refreshed.recovery_gap, baseline.recovery_gap);
    assert_eq!(refreshed.live.commit_index, baseline.live.commit_index);
    assert_eq!(refreshed.live.applied_index, baseline.live.applied_index);
    assert_eq!(refreshed.live.next_index, baseline.live.next_index);
    assert_eq!(
        refreshed.durable.commit_index,
        baseline.durable.commit_index
    );
    assert_eq!(
        refreshed.durable.applied_index,
        baseline.durable.applied_index
    );
    assert_eq!(refreshed.durable.next_index, baseline.durable.next_index);
    assert_eq!(refreshed.live.snapshot.snapshot_id, 31);
    assert_eq!(refreshed.durable.snapshot.snapshot_id, 31);

    let refreshed_recovery = r.recovery_state();
    assert_eq!(refreshed_recovery.snapshot.snapshot_id, 31);
    let resumed = RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, refreshed.durable);
    assert_eq!(
        refreshed_recovery.progress_as_follower().unwrap(),
        refreshed.durable
    );

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

    let after_repair = r.status_snapshot();
    assert!(after_repair.has_speculative_tail());
    assert_eq!(after_repair.live.term, 6);
    assert_eq!(after_repair.live.snapshot.snapshot_id, 31);
    assert_eq!(after_repair.durable.snapshot.snapshot_id, 31);
    assert_eq!(after_repair.live.commit_index, 8);
    assert_eq!(after_repair.live.applied_index, 7);
    assert_eq!(after_repair.live.next_index, 10);
    assert_eq!(after_repair.durable.commit_index, 8);
    assert_eq!(after_repair.durable.applied_index, 7);
    assert_eq!(after_repair.durable.next_index, 9);
    assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
    assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);

    let repair_recovery = r.recovery_state();
    assert_eq!(repair_recovery.snapshot.snapshot_id, 31);
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

    r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
        .unwrap();
    let after_commit = r.status_snapshot();
    assert!(after_commit.live.has_committed_entries_pending_apply);
    assert!(!after_commit.has_speculative_tail());
    assert_eq!(after_commit.live.snapshot.snapshot_id, 31);
    assert_eq!(after_commit.durable.snapshot.snapshot_id, 31);
    assert_eq!(after_commit.live.commit_index, 9);
    assert_eq!(after_commit.live.applied_index, 7);
    assert_eq!(after_commit.live.next_index, 10);
    assert_eq!(after_commit.durable.commit_index, 9);
    assert_eq!(after_commit.durable.applied_index, 7);
    assert_eq!(after_commit.durable.next_index, 10);
    assert_eq!(after_commit.recovery_gap.next_index_gap, 0);
    assert_eq!(after_commit.recovery_gap.uncommitted_entry_gap, 0);

    let committed_recovery = r.recovery_state();
    assert_eq!(committed_recovery.snapshot.snapshot_id, 31);
    let resumed_after_commit =
        RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_commit.status_snapshot().live,
        after_commit.durable
    );
    assert_eq!(
        committed_recovery.progress_as_follower().unwrap(),
        after_commit.durable
    );

    r.mark_applied(9);
    let after_apply = r.status_snapshot();
    assert!(after_apply.is_restart_equivalent());
    assert_eq!(after_apply.live.snapshot.snapshot_id, 31);
    assert_eq!(after_apply.durable.snapshot.snapshot_id, 31);
    assert_eq!(after_apply.live.commit_index, 9);
    assert_eq!(after_apply.live.applied_index, 9);
    assert_eq!(after_apply.live.next_index, 10);
    assert_eq!(after_apply.durable, after_apply.live);
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.snapshot.snapshot_id, 31);
    let resumed_after_apply =
        RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
    assert_eq!(resumed_after_apply.status_snapshot().live, after_apply.live);
    assert_eq!(
        applied_recovery.progress_as_follower().unwrap(),
        after_apply.live
    );
}

#[test]
fn compatible_advanced_snapshot_suffix_refresh_ignores_stale_snapshots_without_perturbing_gap() {
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
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 31,
    });

    let baseline_status = r.status_snapshot();
    let baseline_recovery = r.recovery_state();
    let baseline_gap = r.recovery_progress_gap();
    assert!(baseline_status.has_speculative_tail());
    assert_eq!(baseline_status.live.snapshot.snapshot_id, 31);
    assert_eq!(baseline_status.durable.snapshot.snapshot_id, 31);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 4,
        snapshot_id: 97,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 4,
        snapshot_id: 101,
    });

    assert_eq!(r.status_snapshot(), baseline_status);
    assert_eq!(r.recovery_state(), baseline_recovery);
    assert_eq!(r.recovery_progress_gap(), baseline_gap);
    assert_eq!(r.snapshot_meta().snapshot_id, 31);
    assert_eq!(
        baseline_recovery.progress_as_follower().unwrap(),
        baseline_status.durable
    );
}

#[test]
fn compatible_advanced_snapshot_suffix_refresh_survives_newer_leader_rejection() {
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
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 31,
    });
    assert!(r.status_snapshot().has_speculative_tail());

    let err = r
        .append_entries_from_leader(
            6,
            99,
            6,
            vec![LogEntry {
                term: 6,
                index: 100,
                payload: vec![100].into(),
            }],
            100,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));

    let after_reject = r.status_snapshot();
    assert!(after_reject.is_restart_equivalent());
    assert_eq!(after_reject.live.term, 6);
    assert_eq!(after_reject.live.snapshot.snapshot_id, 31);
    assert_eq!(after_reject.durable.snapshot.snapshot_id, 31);
    assert_eq!(after_reject.live.commit_index, 7);
    assert_eq!(after_reject.live.applied_index, 7);
    assert_eq!(after_reject.live.next_index, 8);
    assert!(r.recovery_progress_gap().is_restart_equivalent());
    assert_eq!(r.recovery_state().snapshot.snapshot_id, 31);
}

#[test]
fn refreshed_compatible_suffix_repair_phase_snapshot_refresh_updates_durable_identity_without_perturbing_gap(
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
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 31,
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
    assert_eq!(repair_status.live.snapshot.snapshot_id, 31);
    assert_eq!(repair_status.durable.snapshot.snapshot_id, 31);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 37,
    });

    let refreshed_repair_status = r.status_snapshot();
    assert!(refreshed_repair_status.has_speculative_tail());
    assert_eq!(
        refreshed_repair_status.recovery_gap,
        repair_status.recovery_gap
    );
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
    assert_eq!(refreshed_repair_status.live.snapshot.snapshot_id, 37);
    assert_eq!(refreshed_repair_status.durable.snapshot.snapshot_id, 37);

    let refreshed_repair_recovery = r.recovery_state();
    assert_eq!(refreshed_repair_recovery.snapshot.snapshot_id, 37);
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

    r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
        .unwrap();
    let after_commit = r.status_snapshot();
    assert!(after_commit.live.has_committed_entries_pending_apply);
    assert!(!after_commit.has_speculative_tail());
    assert_eq!(after_commit.live.snapshot.snapshot_id, 37);
    assert_eq!(after_commit.durable.snapshot.snapshot_id, 37);
    assert_eq!(after_commit.recovery_gap.next_index_gap, 0);
    assert_eq!(after_commit.recovery_gap.uncommitted_entry_gap, 0);

    let committed_recovery = r.recovery_state();
    assert_eq!(committed_recovery.snapshot.snapshot_id, 37);
    let resumed_after_commit =
        RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
    assert_eq!(
        resumed_after_commit.status_snapshot().live,
        after_commit.durable
    );
    assert_eq!(
        committed_recovery.progress_as_follower().unwrap(),
        after_commit.durable
    );

    r.mark_applied(9);
    let after_apply = r.status_snapshot();
    assert!(after_apply.is_restart_equivalent());
    assert_eq!(after_apply.live.snapshot.snapshot_id, 37);
    assert_eq!(after_apply.durable.snapshot.snapshot_id, 37);
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    let applied_recovery = r.recovery_state();
    assert_eq!(applied_recovery.snapshot.snapshot_id, 37);
    let resumed_after_apply =
        RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
    assert_eq!(resumed_after_apply.status_snapshot().live, after_apply.live);
    assert_eq!(
        applied_recovery.progress_as_follower().unwrap(),
        after_apply.live
    );
}

#[test]
fn refreshed_compatible_suffix_repair_phase_second_refresh_still_collapses_cleanly_on_rejection() {
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
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 31,
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
    assert_eq!(repair_status.live.snapshot.snapshot_id, 31);
    assert_eq!(repair_status.durable.snapshot.snapshot_id, 31);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 37,
    });

    let refreshed_repair = r.status_snapshot();
    assert!(refreshed_repair.has_speculative_tail());
    assert_eq!(refreshed_repair.recovery_gap, repair_status.recovery_gap);
    assert_eq!(refreshed_repair.live.snapshot.snapshot_id, 37);
    assert_eq!(refreshed_repair.durable.snapshot.snapshot_id, 37);

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
    assert_eq!(after_reject.live.snapshot.snapshot_id, 37);
    assert_eq!(after_reject.durable.snapshot.snapshot_id, 37);
    assert_eq!(after_reject.live.commit_index, 8);
    assert_eq!(after_reject.live.applied_index, 7);
    assert_eq!(after_reject.live.next_index, 9);
    assert_eq!(after_reject.durable.commit_index, 8);
    assert_eq!(after_reject.durable.applied_index, 7);
    assert_eq!(after_reject.durable.next_index, 9);
    assert_eq!(after_reject.recovery_gap.next_index_gap, 0);
    assert_eq!(after_reject.recovery_gap.uncommitted_entry_gap, 0);
    assert!(after_reject.live.has_committed_entries_pending_apply);
    assert!(r.recovery_progress_gap().is_restart_equivalent());

    let after_reject_recovery = r.recovery_state();
    assert_eq!(after_reject_recovery.term, 7);
    assert_eq!(after_reject_recovery.snapshot.snapshot_id, 37);
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
}

#[test]
fn refreshed_compatible_suffix_stale_snapshots_remain_noops_during_repair_commit_and_apply() {
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
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 31,
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
    let repair_recovery = r.recovery_state();
    let repair_gap = r.recovery_progress_gap();
    assert!(repair_status.has_speculative_tail());

    r.install_snapshot(SnapshotMeta {
        last_included_index: 6,
        last_included_term: 5,
        snapshot_id: 97,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 4,
        snapshot_id: 98,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 4,
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

    r.install_snapshot(SnapshotMeta {
        last_included_index: 6,
        last_included_term: 5,
        snapshot_id: 107,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 4,
        snapshot_id: 108,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 4,
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

    r.install_snapshot(SnapshotMeta {
        last_included_index: 6,
        last_included_term: 5,
        snapshot_id: 117,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 4,
        snapshot_id: 118,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 4,
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
    assert_eq!(r.snapshot_meta().snapshot_id, 31);
}

#[test]
fn refreshed_compatible_suffix_second_repair_refresh_keeps_stale_snapshots_inert_through_commit_and_apply(
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
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 31,
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
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 37,
    });

    let refreshed_repair_status = r.status_snapshot();
    let refreshed_repair_recovery = r.recovery_state();
    let refreshed_repair_gap = r.recovery_progress_gap();
    assert!(refreshed_repair_status.has_speculative_tail());
    assert_eq!(refreshed_repair_status.live.snapshot.snapshot_id, 37);
    assert_eq!(refreshed_repair_status.durable.snapshot.snapshot_id, 37);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 6,
        last_included_term: 5,
        snapshot_id: 97,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 4,
        snapshot_id: 98,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 4,
        snapshot_id: 99,
    });

    assert_eq!(r.status_snapshot(), refreshed_repair_status);
    assert_eq!(r.recovery_state(), refreshed_repair_recovery);
    assert_eq!(r.recovery_progress_gap(), refreshed_repair_gap);
    let resumed_during_refreshed_repair_after_stale =
        RaftReplicator::resume_as_follower(3, refreshed_repair_recovery.clone()).unwrap();
    assert_eq!(
        resumed_during_refreshed_repair_after_stale
            .status_snapshot()
            .live,
        refreshed_repair_status.durable
    );
    assert_eq!(
        refreshed_repair_recovery.progress_as_follower().unwrap(),
        refreshed_repair_status.durable
    );

    r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
        .unwrap();
    let commit_status = r.status_snapshot();
    let commit_recovery = r.recovery_state();
    let commit_gap = r.recovery_progress_gap();
    assert!(commit_status.live.has_committed_entries_pending_apply);
    assert!(!commit_status.has_speculative_tail());
    assert_eq!(commit_status.live.snapshot.snapshot_id, 37);
    assert_eq!(commit_status.durable.snapshot.snapshot_id, 37);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 6,
        last_included_term: 5,
        snapshot_id: 107,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 4,
        snapshot_id: 108,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 4,
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
    assert_eq!(applied_status.live.snapshot.snapshot_id, 37);
    assert_eq!(applied_status.durable.snapshot.snapshot_id, 37);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 6,
        last_included_term: 5,
        snapshot_id: 117,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 4,
        snapshot_id: 118,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: 8,
        last_included_term: 4,
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
    assert_eq!(r.snapshot_meta().snapshot_id, 37);
}

#[test]
fn refreshed_compatible_suffix_second_repair_refresh_survives_role_change_tail_discard() {
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
    r.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 31,
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
        last_included_index: 7,
        last_included_term: 5,
        snapshot_id: 37,
    });

    let before_role_change = r.status_snapshot();
    assert!(before_role_change.has_speculative_tail());
    assert_eq!(before_role_change.live.snapshot.snapshot_id, 37);
    assert_eq!(before_role_change.durable.snapshot.snapshot_id, 37);

    r.become_candidate(7);

    let after_role_change = r.status_snapshot();
    assert!(after_role_change.is_restart_equivalent());
    assert_eq!(after_role_change.live.role, Role::Candidate);
    assert_eq!(after_role_change.live.term, 7);
    assert_eq!(after_role_change.live.commit_index, 8);
    assert_eq!(after_role_change.live.applied_index, 7);
    assert_eq!(after_role_change.live.next_index, 9);
    assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.live.snapshot.snapshot_id, 37);
    assert_eq!(after_role_change.durable.role, Role::Follower);
    assert_eq!(after_role_change.durable.term, 7);
    assert_eq!(after_role_change.durable.commit_index, 8);
    assert_eq!(after_role_change.durable.applied_index, 7);
    assert_eq!(after_role_change.durable.next_index, 9);
    assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
    assert_eq!(after_role_change.durable.snapshot.snapshot_id, 37);
    assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
    assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

    let recovery = r.recovery_state();
    assert_eq!(recovery.term, 7);
    assert_eq!(recovery.snapshot.snapshot_id, 37);
    let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
    assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
    assert_eq!(
        recovery.progress_as_follower().unwrap(),
        after_role_change.durable
    );
    assert_eq!(r.snapshot_meta().snapshot_id, 37);
}
