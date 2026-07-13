#[test]
fn replication_progress_reports_caught_up_only_when_apply_and_tail_are_clear() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(5);
    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);

    let before_apply = r.progress();
    assert_eq!(before_apply.apply_gap(), 1);
    assert!(!before_apply.is_caught_up());

    r.mark_applied(t1.index);
    let after_apply = r.progress();
    assert_eq!(after_apply.apply_gap(), 0);
    assert!(after_apply.is_caught_up());
}

#[test]
fn recovery_state_helpers_report_commit_apply_boundaries() {
    let state = RecoveryState {
        term: 4,
        snapshot: SnapshotMeta {
            last_included_index: 3,
            last_included_term: 4,
            snapshot_id: 7,
        },
        committed_entries: vec![LogEntry {
            term: 4,
            index: 4,
            payload: vec![1].into(),
        }],
        applied_index: 3,
    };

    assert_eq!(state.commit_index(), 4);
    assert_eq!(state.next_index(), 5);
    assert!(state.has_committed_entries_pending_apply());
    assert_eq!(state.committed_but_unapplied_count(), 1);
    assert_eq!(state.apply_gap(), 1);
    assert!(!state.is_caught_up());
    state.validate().unwrap();

    let progress = state.progress_as_follower().unwrap();
    assert_eq!(progress.role, Role::Follower);
    assert_eq!(progress.term, 4);
    assert_eq!(progress.commit_index, 4);
    assert_eq!(progress.applied_index, 3);
    assert_eq!(progress.next_index, 5);
    assert_eq!(progress.apply_gap(), 1);
    assert!(!progress.is_caught_up());
}

#[test]
fn recovery_state_reports_caught_up_when_apply_reaches_commit_boundary() {
    let state = RecoveryState {
        term: 4,
        snapshot: SnapshotMeta {
            last_included_index: 3,
            last_included_term: 4,
            snapshot_id: 7,
        },
        committed_entries: vec![LogEntry {
            term: 4,
            index: 4,
            payload: vec![1].into(),
        }],
        applied_index: 4,
    };

    assert_eq!(state.apply_gap(), 0);
    assert!(state.is_caught_up());
    assert!(state.progress_as_follower().unwrap().is_caught_up());
}

#[test]
fn snapshot_only_recovery_state_maps_to_caught_up_progress() {
    let state = RecoveryState {
        term: 6,
        snapshot: SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 11,
        },
        committed_entries: vec![],
        applied_index: 9,
    };

    assert_eq!(state.commit_index(), 9);
    assert_eq!(state.next_index(), 10);
    assert_eq!(state.committed_but_unapplied_count(), 0);
    assert_eq!(state.apply_gap(), 0);
    assert!(state.is_caught_up());

    let progress = state.progress_as_follower().unwrap();
    assert_eq!(progress.role, Role::Follower);
    assert_eq!(progress.term, 6);
    assert_eq!(progress.commit_index, 9);
    assert_eq!(progress.applied_index, 9);
    assert_eq!(progress.next_index, 10);
    assert_eq!(progress.uncommitted_entry_count, 0);
    assert!(progress.is_caught_up());
}

#[test]
fn recovery_state_validation_rejects_applied_index_past_commit_boundary() {
    let err = RecoveryState {
        term: 4,
        snapshot: SnapshotMeta {
            last_included_index: 3,
            last_included_term: 4,
            snapshot_id: 7,
        },
        committed_entries: vec![LogEntry {
            term: 4,
            index: 4,
            payload: vec![1].into(),
        }],
        applied_index: 5,
    }
    .validate()
    .unwrap_err();

    assert_eq!(
        err,
        RecoveryInvariantError::AppliedExceedsCommit {
            applied_index: 5,
            commit_index: 4,
        }
    );
}

#[test]
fn resumed_follower_progress_matches_recovery_projection() {
    let mut leader = RaftReplicator::new(3);
    leader.become_leader(4);
    let t1 = leader.propose(vec![1].into()).unwrap();
    let t2 = leader.propose(vec![2].into()).unwrap();
    leader.register_follower_ack(t1.index, 1);
    leader.register_follower_ack(t2.index, 1);
    leader.mark_applied(t1.index);

    let recovery = leader.recovery_state();
    let projected = recovery.progress_as_follower().unwrap();
    let resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();

    assert_eq!(resumed.progress(), projected);
}

#[test]
fn resumed_follower_from_snapshot_only_recovery_matches_projection() {
    let recovery = RecoveryState {
        term: 6,
        snapshot: SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 11,
        },
        committed_entries: vec![],
        applied_index: 9,
    };

    let projected = recovery.progress_as_follower().unwrap();
    let resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();

    assert_eq!(resumed.progress(), projected);
    assert!(resumed.progress().is_caught_up());
}

#[test]
fn recovery_progress_handles_snapshot_boundary_committed_tail() {
    let recovery = RecoveryState {
        term: 7,
        snapshot: SnapshotMeta {
            last_included_index: 10,
            last_included_term: 6,
            snapshot_id: 21,
        },
        committed_entries: vec![
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![11].into(),
            },
            LogEntry {
                term: 7,
                index: 12,
                payload: vec![12].into(),
            },
        ],
        applied_index: 10,
    };

    let progress = recovery.progress_as_follower().unwrap();

    assert_eq!(progress.commit_index, 12);
    assert_eq!(progress.applied_index, 10);
    assert_eq!(progress.next_index, 13);
    assert_eq!(progress.apply_gap(), 2);
    assert_eq!(progress.committed_but_unapplied_count, 2);
    assert!(progress.has_committed_entries_pending_apply);
    assert!(!progress.has_uncommitted_entries);
}

#[test]
fn resumed_follower_from_snapshot_boundary_tail_matches_projection() {
    let recovery = RecoveryState {
        term: 7,
        snapshot: SnapshotMeta {
            last_included_index: 10,
            last_included_term: 6,
            snapshot_id: 21,
        },
        committed_entries: vec![
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![11].into(),
            },
            LogEntry {
                term: 7,
                index: 12,
                payload: vec![12].into(),
            },
        ],
        applied_index: 10,
    };

    let projected = recovery.progress_as_follower().unwrap();
    let resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();

    assert_eq!(resumed.progress(), projected);
    assert_eq!(resumed.progress().apply_gap(), 2);
    let status = resumed.status_snapshot();
    assert_eq!(status.live, projected);
    assert_eq!(status.durable, projected);
    assert!(status.is_restart_equivalent());
}

#[test]
fn resumed_follower_progress_stays_stable_through_catch_up_and_snapshot_stress() {
    let recovery = RecoveryState {
        term: 7,
        snapshot: SnapshotMeta {
            last_included_index: 10,
            last_included_term: 6,
            snapshot_id: 21,
        },
        committed_entries: vec![
            LogEntry {
                term: 7,
                index: 11,
                payload: vec![11].into(),
            },
            LogEntry {
                term: 7,
                index: 12,
                payload: vec![12].into(),
            },
        ],
        applied_index: 10,
    };

    let mut resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();
    let baseline = resumed.progress();
    let baseline_recovery = resumed.recovery_progress();
    let baseline_status = resumed.status_snapshot();
    assert_eq!(baseline.commit_index, 12);
    assert_eq!(baseline.applied_index, 10);
    assert_eq!(baseline.next_index, 13);
    assert_eq!(baseline.apply_gap(), 2);
    assert!(!baseline.is_caught_up());
    assert_eq!(baseline_recovery, baseline);
    assert_eq!(baseline_status.live, baseline);
    assert_eq!(baseline_status.durable, baseline_recovery);
    assert!(baseline_status.is_restart_equivalent());
    assert!(resumed.recovery_progress_gap().is_restart_equivalent());

    let err = resumed
        .append_entries_from_leader(
            7,
            12,
            7,
            vec![LogEntry {
                term: 7,
                index: 14,
                payload: vec![14].into(),
            }],
            14,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(resumed.progress(), baseline);
    assert_eq!(resumed.recovery_progress(), baseline_recovery);
    assert_eq!(resumed.status_snapshot(), baseline_status);
    assert!(resumed.recovery_progress_gap().is_restart_equivalent());

    resumed
        .append_entries_from_leader(
            7,
            12,
            7,
            vec![LogEntry {
                term: 7,
                index: 13,
                payload: vec![13].into(),
            }],
            12,
        )
        .unwrap();
    let after_append = resumed.progress();
    let after_append_recovery = resumed.recovery_progress();
    assert_eq!(after_append.commit_index, 12);
    assert_eq!(after_append.applied_index, 10);
    assert_eq!(after_append.next_index, 14);
    assert_eq!(after_append.uncommitted_entry_count, 1);
    assert!(after_append.has_uncommitted_entries);
    assert_eq!(after_append.apply_gap(), 2);
    assert!(!after_append.is_caught_up());
    after_append.validate().unwrap();
    assert_eq!(after_append_recovery.commit_index, 12);
    assert_eq!(after_append_recovery.applied_index, 10);
    assert_eq!(after_append_recovery.next_index, 13);
    assert_eq!(after_append_recovery.uncommitted_entry_count, 0);
    assert!(!after_append_recovery.has_uncommitted_entries);
    assert_eq!(after_append_recovery.apply_gap(), 2);
    assert!(!after_append_recovery.is_caught_up());
    let after_append_status = resumed.status_snapshot();
    assert_eq!(after_append_status.live, after_append);
    assert_eq!(after_append_status.durable, after_append_recovery);
    assert!(after_append_status.has_speculative_tail());
    assert_eq!(
        resumed.recovery_progress_gap(),
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 1,
            uncommitted_entry_gap: 1,
        }
    );

    resumed
        .append_entries_from_leader(7, 13, 7, vec![], 13)
        .unwrap();
    let after_heartbeat = resumed.progress();
    let after_heartbeat_recovery = resumed.recovery_progress();
    assert_eq!(after_heartbeat.commit_index, 13);
    assert_eq!(after_heartbeat.applied_index, 10);
    assert_eq!(after_heartbeat.next_index, 14);
    assert_eq!(after_heartbeat.uncommitted_entry_count, 0);
    assert!(!after_heartbeat.has_uncommitted_entries);
    assert_eq!(after_heartbeat.apply_gap(), 3);
    assert!(!after_heartbeat.is_caught_up());
    after_heartbeat.validate().unwrap();
    assert_eq!(after_heartbeat_recovery, after_heartbeat);
    assert!(resumed.status_snapshot().is_restart_equivalent());
    assert!(resumed.recovery_progress_gap().is_restart_equivalent());

    resumed.install_snapshot(SnapshotMeta {
        last_included_index: 13,
        last_included_term: 7,
        snapshot_id: 22,
    });
    let after_snapshot = resumed.progress();
    let after_snapshot_recovery = resumed.recovery_progress();
    assert_eq!(after_snapshot.snapshot.snapshot_id, 22);
    assert_eq!(after_snapshot.snapshot.last_included_index, 13);
    assert_eq!(after_snapshot.commit_index, 13);
    assert_eq!(after_snapshot.applied_index, 13);
    assert_eq!(after_snapshot.next_index, 14);
    assert_eq!(after_snapshot.apply_gap(), 0);
    assert!(after_snapshot.is_caught_up());
    after_snapshot.validate().unwrap();
    assert_eq!(after_snapshot_recovery, after_snapshot);
    assert!(resumed.status_snapshot().is_restart_equivalent());
    assert!(resumed.recovery_progress_gap().is_restart_equivalent());

    let err = resumed
        .append_entries_from_leader(
            7,
            12,
            7,
            vec![LogEntry {
                term: 7,
                index: 13,
                payload: vec![13].into(),
            }],
            13,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(resumed.progress(), after_snapshot);
    assert_eq!(resumed.recovery_progress(), after_snapshot_recovery);
    assert_eq!(resumed.status_snapshot().durable, after_snapshot_recovery);
    assert!(resumed.recovery_progress_gap().is_restart_equivalent());
}

#[test]
fn replication_progress_validation_rejects_inconsistent_uncommitted_flag() {
    let err = ReplicationProgress {
        role: Role::Follower,
        term: 4,
        commit_index: 5,
        applied_index: 5,
        next_index: 6,
        snapshot: SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 1,
        },
        committed_but_unapplied_count: 0,
        has_committed_entries_pending_apply: false,
        uncommitted_entry_count: 1,
        has_uncommitted_entries: false,
    }
    .validate()
    .unwrap_err();

    assert_eq!(
        err,
        ReplicationProgressInvariantError::UncommittedFlagMismatch {
            has_uncommitted: false,
            uncommitted_entry_count: 1,
        }
    );
}

#[test]
fn replication_status_snapshot_validation_rejects_durable_state_ahead_of_live() {
    let live = ReplicationProgress {
        role: Role::Leader,
        term: 4,
        commit_index: 5,
        applied_index: 5,
        next_index: 6,
        snapshot: SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 10,
        },
        committed_but_unapplied_count: 0,
        has_committed_entries_pending_apply: false,
        uncommitted_entry_count: 0,
        has_uncommitted_entries: false,
    };
    let durable = ReplicationProgress {
        next_index: 7,
        ..live.clone()
    };

    let err = ReplicationStatusSnapshot::new(
        live,
        durable,
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 0,
            uncommitted_entry_gap: 0,
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        ReplicationStatusInvariantError::DurableAheadOfLive {
            field: "next_index",
            durable: 7,
            live: 6,
        }
    );
}

#[test]
fn replication_status_snapshot_validation_rejects_term_mismatch() {
    let live = ReplicationProgress {
        role: Role::Leader,
        term: 4,
        commit_index: 5,
        applied_index: 5,
        next_index: 6,
        snapshot: SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 10,
        },
        committed_but_unapplied_count: 0,
        has_committed_entries_pending_apply: false,
        uncommitted_entry_count: 0,
        has_uncommitted_entries: false,
    };
    let durable = ReplicationProgress {
        role: Role::Follower,
        term: 3,
        ..live.clone()
    };

    let err = ReplicationStatusSnapshot::new(
        live,
        durable,
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 0,
            uncommitted_entry_gap: 0,
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        ReplicationStatusInvariantError::TermMismatch {
            durable: 3,
            live: 4,
        }
    );
}

#[test]
fn replication_status_snapshot_validation_rejects_same_frontier_snapshot_identity_drift() {
    let live = ReplicationProgress {
        role: Role::Follower,
        term: 4,
        commit_index: 5,
        applied_index: 5,
        next_index: 6,
        snapshot: SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 10,
        },
        committed_but_unapplied_count: 0,
        has_committed_entries_pending_apply: false,
        uncommitted_entry_count: 0,
        has_uncommitted_entries: false,
    };
    let durable = ReplicationProgress {
        snapshot: SnapshotMeta {
            snapshot_id: 9,
            ..live.snapshot.clone()
        },
        ..live.clone()
    };

    let err = ReplicationStatusSnapshot::new(
        live,
        durable,
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 0,
            uncommitted_entry_gap: 0,
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        ReplicationStatusInvariantError::SnapshotIdentityDrift {
            last_included_index: 5,
            last_included_term: 4,
            durable_snapshot_id: 9,
            live_snapshot_id: 10,
        }
    );
}

#[test]
fn replication_status_snapshot_validation_rejects_recovery_gap_mismatch() {
    let live = ReplicationProgress {
        role: Role::Follower,
        term: 4,
        commit_index: 5,
        applied_index: 5,
        next_index: 7,
        snapshot: SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 10,
        },
        committed_but_unapplied_count: 0,
        has_committed_entries_pending_apply: false,
        uncommitted_entry_count: 1,
        has_uncommitted_entries: true,
    };
    let durable = ReplicationProgress {
        next_index: 6,
        uncommitted_entry_count: 0,
        has_uncommitted_entries: false,
        ..live.clone()
    };

    let err = ReplicationStatusSnapshot::new(
        live,
        durable,
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 0,
            uncommitted_entry_gap: 0,
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        ReplicationStatusInvariantError::RecoveryGapMismatch {
            expected: RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 1,
                uncommitted_entry_gap: 1,
            },
            actual: RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 0,
                uncommitted_entry_gap: 0,
            },
        }
    );
}

#[test]
fn raft_recovery_progress_projects_durable_follower_state_from_live_leader() {
    let mut leader = RaftReplicator::new(3);
    leader.become_leader(4);
    let t1 = leader.propose(vec![1].into()).unwrap();
    let _t2 = leader.propose(vec![2].into()).unwrap();
    leader.register_follower_ack(t1.index, 1);
    leader.mark_applied(t1.index);

    let live = leader.progress();
    assert_eq!(live.role, Role::Leader);
    assert_eq!(live.uncommitted_entry_count, 1);
    assert!(live.has_uncommitted_entries);

    let durable = leader.recovery_progress();
    assert_eq!(durable.role, Role::Follower);
    assert_eq!(durable.term, live.term);
    assert_eq!(durable.commit_index, t1.index);
    assert_eq!(durable.applied_index, t1.index);
    assert_eq!(durable.next_index, t1.index + 1);
    assert_eq!(durable.snapshot, live.snapshot);
    assert_eq!(durable.uncommitted_entry_count, 0);
    assert!(!durable.has_uncommitted_entries);
    assert!(durable.is_caught_up());

    let gap = leader.recovery_progress_gap();
    assert_eq!(
        gap,
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 1,
            uncommitted_entry_gap: 1,
        }
    );
    assert!(gap.has_gap());
    assert!(gap.has_speculative_tail());
    assert!(!gap.is_restart_equivalent());

    let status = leader.status_snapshot();
    assert_eq!(status.live, live);
    assert_eq!(status.durable, durable);
    assert_eq!(status.recovery_gap, gap);
    assert!(status.has_speculative_tail());
}
