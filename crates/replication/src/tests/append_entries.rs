#[test]
fn raft_follower_append_entries_truncates_conflicting_uncommitted_tail() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    assert_eq!(r.commit_index(), t1.index);

    let _t2_old = r.propose(vec![2].into()).unwrap();
    r.become_follower(2);

    r.append_entries_from_leader(
        2,
        t1.index,
        1,
        vec![LogEntry {
            term: 2,
            index: t1.index + 1,
            payload: vec![9].into(),
        }],
        t1.index + 1,
    )
    .unwrap();

    assert_eq!(r.commit_index(), t1.index + 1);
    assert_eq!(r.next_index, t1.index + 2);
    assert!(r
        .entries
        .iter()
        .any(|entry| entry.index == t1.index + 1 && entry.term == 2));
}

#[test]
fn raft_follower_append_entries_empty_heartbeat_can_advance_commit_index() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    assert_eq!(r.commit_index(), t1.index);

    r.become_follower(2);
    let t2_index = t1.index + 1;
    r.append_entries_from_leader(
        2,
        t1.index,
        1,
        vec![LogEntry {
            term: 2,
            index: t2_index,
            payload: vec![2].into(),
        }],
        t1.index,
    )
    .unwrap();
    assert_eq!(r.commit_index(), t1.index);

    r.append_entries_from_leader(2, t2_index, 2, vec![], t2_index)
        .unwrap();

    assert_eq!(r.commit_index(), t2_index);
    assert_eq!(r.next_index, t2_index + 1);
}

#[test]
fn raft_progress_snapshot_tracks_heartbeat_commit_advancement() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.become_follower(2);

    let t2_index = t1.index + 1;
    r.append_entries_from_leader(
        2,
        t1.index,
        1,
        vec![LogEntry {
            term: 2,
            index: t2_index,
            payload: vec![2].into(),
        }],
        t1.index,
    )
    .unwrap();

    let before_heartbeat = r.progress();
    assert_eq!(before_heartbeat.commit_index, t1.index);
    assert_eq!(before_heartbeat.applied_index, 0);
    assert_eq!(before_heartbeat.next_index, t2_index + 1);
    assert_eq!(before_heartbeat.uncommitted_entry_count, 1);
    assert!(before_heartbeat.has_uncommitted_entries);
    assert_eq!(before_heartbeat.apply_gap(), t1.index as usize);

    r.append_entries_from_leader(2, t2_index, 2, vec![], t2_index)
        .unwrap();

    let after_heartbeat = r.progress();
    assert_eq!(after_heartbeat.commit_index, t2_index);
    assert_eq!(after_heartbeat.applied_index, 0);
    assert_eq!(after_heartbeat.next_index, t2_index + 1);
    assert_eq!(after_heartbeat.uncommitted_entry_count, 0);
    assert!(!after_heartbeat.has_uncommitted_entries);
    assert_eq!(after_heartbeat.apply_gap(), t2_index as usize);
    assert!(!after_heartbeat.is_caught_up());
    after_heartbeat.validate().unwrap();
}

#[test]
fn raft_follower_append_entries_bumps_local_term_from_leader_term() {
    let mut r = RaftReplicator::new(3);
    r.become_candidate(3);

    r.append_entries_from_leader(4, 0, 0, vec![], 0).unwrap();

    assert_eq!(r.current_term(), 4);
    assert_eq!(r.role(), Role::Follower);
}

#[test]
fn raft_follower_append_entries_rejects_stale_leader_term() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    let err = r
        .append_entries_from_leader(4, 0, 0, vec![], 0)
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.current_term(), 5);
}

#[test]
fn raft_follower_append_entries_rejects_stale_term_without_state_mutation() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(5);
    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.become_follower(5);

    let role_before = r.role();
    let term_before = r.current_term();
    let commit_before = r.commit_index();
    let next_before = r.next_index;
    let entries_before = r.entries.clone();

    let err = r
        .append_entries_from_leader(
            4,
            t1.index,
            5,
            vec![LogEntry {
                term: 4,
                index: t1.index + 1,
                payload: vec![9].into(),
            }],
            t1.index + 1,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.role(), role_before);
    assert_eq!(r.current_term(), term_before);
    assert_eq!(r.commit_index(), commit_before);
    assert_eq!(r.next_index, next_before);
    assert_eq!(r.entries, entries_before);
}

#[test]
fn raft_progress_snapshot_is_stable_across_stale_append_rejection() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(5);
    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.become_follower(5);

    let before = r.progress();

    let err = r
        .append_entries_from_leader(
            4,
            t1.index,
            5,
            vec![LogEntry {
                term: 4,
                index: t1.index + 1,
                payload: vec![9].into(),
            }],
            t1.index + 1,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.progress(), before);
}

#[test]
fn raft_follower_append_entries_rejects_entries_with_term_ahead_of_leader() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    let err = r
        .append_entries_from_leader(
            5,
            0,
            0,
            vec![LogEntry {
                term: 6,
                index: 1,
                payload: vec![1].into(),
            }],
            1,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.current_term(), 5);
    assert_eq!(r.commit_index(), 0);
    assert_eq!(r.next_index, 1);
    assert!(r.entries.is_empty());
}

#[test]
fn raft_leader_rejects_follower_append_path_without_state_mutation() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(6);

    let committed = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(committed.index, 1);

    let role_before = r.role();
    let term_before = r.current_term();
    let commit_before = r.commit_index();
    let next_before = r.next_index;

    let err = r
        .append_entries_from_leader(
            7,
            committed.index,
            term_before,
            vec![LogEntry {
                term: 7,
                index: committed.index + 1,
                payload: vec![9].into(),
            }],
            committed.index + 1,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.role(), role_before);
    assert_eq!(r.current_term(), term_before);
    assert_eq!(r.commit_index(), commit_before);
    assert_eq!(r.next_index, next_before);
    assert!(r.entries.iter().all(|entry| entry.term != 7));
}

#[test]
fn raft_follower_append_entries_rejects_prev_term_mismatch() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);
    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.become_follower(2);

    let err = r
        .append_entries_from_leader(
            2,
            t1.index,
            999,
            vec![LogEntry {
                term: 2,
                index: t1.index + 1,
                payload: vec![2].into(),
            }],
            t1.index + 1,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.commit_index(), t1.index);
}

#[test]
fn raft_follower_append_entries_rejects_non_contiguous_batches() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);
    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.become_follower(2);

    let err = r
        .append_entries_from_leader(
            2,
            t1.index,
            1,
            vec![
                LogEntry {
                    term: 2,
                    index: t1.index + 1,
                    payload: vec![2].into(),
                },
                LogEntry {
                    term: 2,
                    index: t1.index + 3,
                    payload: vec![3].into(),
                },
            ],
            t1.index + 3,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.commit_index(), t1.index);
    assert_eq!(r.next_index, t1.index + 1);
    assert!(r.entries.iter().all(|entry| entry.index <= t1.index));
}

#[test]
fn raft_progress_snapshot_is_stable_across_non_contiguous_append_rejection() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);
    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.become_follower(2);

    let before = r.progress();

    let err = r
        .append_entries_from_leader(
            2,
            t1.index,
            1,
            vec![
                LogEntry {
                    term: 2,
                    index: t1.index + 1,
                    payload: vec![2].into(),
                },
                LogEntry {
                    term: 2,
                    index: t1.index + 3,
                    payload: vec![3].into(),
                },
            ],
            t1.index + 3,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.progress(), before);
}

#[test]
fn raft_follower_append_entries_rejects_first_entry_that_skips_prev_index() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);
    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.become_follower(2);

    let err = r
        .append_entries_from_leader(
            2,
            t1.index,
            1,
            vec![LogEntry {
                term: 2,
                index: t1.index + 2,
                payload: vec![2].into(),
            }],
            t1.index + 2,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.commit_index(), t1.index);
    assert_eq!(r.next_index, t1.index + 1);
}

#[test]
fn raft_follower_append_entries_does_not_overwrite_committed_entries() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);
    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.become_follower(2);

    let err = r
        .append_entries_from_leader(
            2,
            0,
            0,
            vec![LogEntry {
                term: 2,
                index: t1.index,
                payload: vec![9].into(),
            }],
            t1.index,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    let committed = r
        .entries
        .iter()
        .find(|entry| entry.index == t1.index)
        .unwrap();
    assert_eq!(committed.term, 1);
    assert_eq!(&committed.payload[..], &vec![1][..]);
}

#[test]
fn raft_follower_append_entries_rejects_payload_mismatch_for_same_index_and_term() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(3);

    r.append_entries_from_leader(
        3,
        0,
        0,
        vec![LogEntry {
            term: 3,
            index: 1,
            payload: vec![1].into(),
        }],
        1,
    )
    .unwrap();

    let commit_before = r.commit_index();
    let next_before = r.next_index;

    let err = r
        .append_entries_from_leader(
            3,
            0,
            0,
            vec![LogEntry {
                term: 3,
                index: 1,
                payload: vec![9].into(),
            }],
            1,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.commit_index(), commit_before);
    assert_eq!(r.next_index, next_before);
    let preserved = r.entries.iter().find(|entry| entry.index == 1).unwrap();
    assert_eq!(preserved.term, 3);
    assert_eq!(&preserved.payload[..], &vec![1][..]);
}

#[test]
fn raft_follower_append_entries_caps_commit_index_to_local_log_tail() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(2);

    r.append_entries_from_leader(
        2,
        0,
        0,
        vec![LogEntry {
            term: 2,
            index: 1,
            payload: vec![1].into(),
        }],
        99,
    )
    .unwrap();

    assert_eq!(r.commit_index(), 1);
    assert_eq!(r.next_index, 2);
}

#[test]
fn raft_follower_append_entries_missing_prev_index_is_rejected_without_state_mutation() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(5);

    let role_before = r.role();
    let term_before = r.current_term();
    let commit_before = r.commit_index();
    let next_before = r.next_index;

    let err = r
        .append_entries_from_leader(
            5,
            10,
            5,
            vec![LogEntry {
                term: 5,
                index: 11,
                payload: vec![7].into(),
            }],
            11,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.role(), role_before);
    assert_eq!(r.current_term(), term_before);
    assert_eq!(r.commit_index(), commit_before);
    assert_eq!(r.next_index, next_before);
    assert!(r.entries.is_empty());
}

#[test]
fn raft_candidate_append_rejection_still_updates_term_and_role_for_newer_leader() {
    let mut r = RaftReplicator::new(3);
    r.become_candidate(5);

    let err = r
        .append_entries_from_leader(
            6,
            10,
            5,
            vec![LogEntry {
                term: 6,
                index: 11,
                payload: vec![7].into(),
            }],
            11,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.role(), Role::Follower);
    assert_eq!(r.current_term(), 6);
    assert_eq!(r.commit_index(), 0);
    assert_eq!(r.next_index, 1);
    assert!(r.entries.is_empty());
}

#[test]
fn raft_newer_leader_rejection_discards_candidate_speculative_tail() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(4);

    let committed = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(committed.index, 1);
    let speculative = r.propose(vec![2].into()).unwrap();
    let speculative_status = r.status_snapshot();
    assert!(speculative_status.has_speculative_tail());
    assert_eq!(speculative_status.live.next_index, speculative.index + 1);
    assert_eq!(speculative_status.live.uncommitted_entry_count, 1);

    r.become_candidate(5);
    let err = r
        .append_entries_from_leader(
            6,
            committed.index + 9,
            6,
            vec![LogEntry {
                term: 6,
                index: committed.index + 10,
                payload: vec![9].into(),
            }],
            committed.index + 10,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    let after = r.status_snapshot();
    assert!(after.is_restart_equivalent());
    assert_eq!(after.live.role, Role::Follower);
    assert_eq!(after.live.term, 6);
    assert_eq!(after.live.commit_index, committed.index);
    assert_eq!(after.live.applied_index, 0);
    assert_eq!(after.live.next_index, committed.index + 1);
    assert_eq!(after.live.uncommitted_entry_count, 0);
    assert!(!after.live.has_uncommitted_entries);
    assert_eq!(after.live, after.durable);
    assert_eq!(r.entries.len(), 1);
    assert_eq!(r.entries[0].index, committed.index);
}

#[test]
fn raft_newer_leader_rejection_discards_follower_speculative_tail() {
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

    let baseline = r.status_snapshot();
    assert!(baseline.has_speculative_tail());
    assert_eq!(baseline.live.uncommitted_entry_count, 1);
    assert_eq!(baseline.durable.snapshot.snapshot_id, 2);

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
    let after = r.status_snapshot();
    assert!(after.is_restart_equivalent());
    assert_eq!(after.live.role, Role::Follower);
    assert_eq!(after.live.term, 5);
    assert_eq!(after.live.commit_index, 5);
    assert_eq!(after.live.applied_index, 5);
    assert_eq!(after.live.next_index, 6);
    assert_eq!(after.live.snapshot.snapshot_id, 2);
    assert_eq!(after.live.uncommitted_entry_count, 0);
    assert!(!after.live.has_uncommitted_entries);
    assert_eq!(after.live, after.durable);
    assert!(r.entries.is_empty());
}

#[test]
fn raft_newer_leader_acceptance_replaces_follower_speculative_tail_with_fresh_gap() {
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

    let before = r.status_snapshot();
    assert!(before.has_speculative_tail());
    assert_eq!(before.live.term, 4);
    assert_eq!(before.live.commit_index, 5);
    assert_eq!(before.live.next_index, 7);
    assert_eq!(before.live.uncommitted_entry_count, 1);
    assert_eq!(before.durable.snapshot.snapshot_id, 2);

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

    let after = r.status_snapshot();
    assert_eq!(after.live.role, Role::Follower);
    assert_eq!(after.live.term, 5);
    assert_eq!(after.live.commit_index, 6);
    assert_eq!(after.live.applied_index, 5);
    assert_eq!(after.live.next_index, 8);
    assert_eq!(after.live.uncommitted_entry_count, 1);
    assert!(after.live.has_committed_entries_pending_apply);
    assert_eq!(after.live.committed_but_unapplied_count, 1);
    assert_eq!(after.durable.term, 5);
    assert_eq!(after.durable.commit_index, 6);
    assert_eq!(after.durable.applied_index, 5);
    assert_eq!(after.durable.next_index, 7);
    assert_eq!(after.durable.uncommitted_entry_count, 0);
    assert_eq!(
        after.recovery_gap,
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 1,
            uncommitted_entry_gap: 1,
        }
    );
    assert!(after.has_speculative_tail());
    assert_eq!(after.live.snapshot.snapshot_id, 2);
    assert_eq!(after.durable.snapshot.snapshot_id, 2);
    assert_eq!(r.entries.len(), 2);
    assert_eq!(r.entries[0].index, 6);
    assert_eq!(r.entries[0].term, 5);
    assert_eq!(&r.entries[0].payload[..], &[60u8][..]);
    assert_eq!(r.entries[1].index, 7);
    assert_eq!(r.entries[1].term, 5);
    assert_eq!(&r.entries[1].payload[..], &[70u8][..]);
}

#[test]
fn raft_newer_leader_catch_up_promotes_fresh_tail_without_reintroducing_stale_gap() {
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

    let after_repair = r.status_snapshot();
    assert_eq!(after_repair.live.commit_index, 6);
    assert_eq!(after_repair.live.applied_index, 5);
    assert_eq!(after_repair.live.next_index, 8);
    assert_eq!(after_repair.live.uncommitted_entry_count, 1);
    assert_eq!(after_repair.durable.commit_index, 6);
    assert_eq!(after_repair.durable.next_index, 7);
    assert!(after_repair.has_speculative_tail());

    r.append_entries_from_leader(5, 7, 5, vec![], 7).unwrap();

    let after_commit = r.status_snapshot();
    assert_eq!(after_commit.live.commit_index, 7);
    assert_eq!(after_commit.live.applied_index, 5);
    assert_eq!(after_commit.live.next_index, 8);
    assert_eq!(after_commit.live.uncommitted_entry_count, 0);
    assert_eq!(after_commit.live.committed_but_unapplied_count, 2);
    assert_eq!(after_commit.durable.commit_index, 7);
    assert_eq!(after_commit.durable.applied_index, 5);
    assert_eq!(after_commit.durable.next_index, 8);
    assert_eq!(after_commit.durable.uncommitted_entry_count, 0);
    assert!(after_commit.is_restart_equivalent());
    assert!(!after_commit.has_speculative_tail());

    r.mark_applied(7);

    let after_apply = r.status_snapshot();
    assert!(after_apply.is_restart_equivalent());
    assert!(!after_apply.has_speculative_tail());
    assert_eq!(after_apply.live.commit_index, 7);
    assert_eq!(after_apply.live.applied_index, 7);
    assert_eq!(after_apply.live.next_index, 8);
    assert_eq!(after_apply.live.committed_but_unapplied_count, 0);
    assert_eq!(after_apply.live, after_apply.durable);
    assert_eq!(after_apply.live.snapshot.snapshot_id, 2);
}

#[test]
fn raft_follower_append_entries_accepts_prev_index_at_snapshot_boundary_after_apply_advances() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(3);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 2,
        snapshot_id: 1,
    });

    r.append_entries_from_leader(
        3,
        5,
        2,
        vec![LogEntry {
            term: 3,
            index: 6,
            payload: vec![6].into(),
        }],
        6,
    )
    .unwrap();
    r.mark_applied(6);

    r.append_entries_from_leader(
        3,
        5,
        2,
        vec![LogEntry {
            term: 3,
            index: 6,
            payload: vec![6].into(),
        }],
        6,
    )
    .unwrap();

    assert_eq!(r.commit_index(), 6);
    assert_eq!(r.next_index, 7);
}

#[test]
fn raft_progress_snapshot_remains_consistent_across_append_at_snapshot_boundary() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(3);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 2,
        snapshot_id: 1,
    });

    let before = r.progress();
    assert_eq!(before.commit_index, 5);
    assert_eq!(before.applied_index, 5);
    assert_eq!(before.next_index, 6);
    assert!(before.is_caught_up());

    r.append_entries_from_leader(
        3,
        5,
        2,
        vec![LogEntry {
            term: 3,
            index: 6,
            payload: vec![6].into(),
        }],
        6,
    )
    .unwrap();

    let after_append = r.progress();
    assert_eq!(after_append.commit_index, 6);
    assert_eq!(after_append.applied_index, 5);
    assert_eq!(after_append.next_index, 7);
    assert_eq!(after_append.apply_gap(), 1);
    assert!(!after_append.is_caught_up());

    r.mark_applied(6);
    let after_apply = r.progress();
    assert_eq!(after_apply.commit_index, 6);
    assert_eq!(after_apply.applied_index, 6);
    assert_eq!(after_apply.next_index, 7);
    assert_eq!(after_apply.apply_gap(), 0);
    assert!(after_apply.is_caught_up());

    r.append_entries_from_leader(
        3,
        5,
        2,
        vec![LogEntry {
            term: 3,
            index: 6,
            payload: vec![6].into(),
        }],
        6,
    )
    .unwrap();

    let after_repeat = r.progress();
    assert_eq!(after_repeat, after_apply);
}

#[test]
fn raft_follower_append_entries_rejects_snapshot_boundary_term_mismatch() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 3,
        snapshot_id: 1,
    });

    let err = r
        .append_entries_from_leader(
            4,
            5,
            2,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![6].into(),
            }],
            6,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.commit_index(), 5);
    assert_eq!(r.applied_index(), 5);
    assert_eq!(r.next_index, 6);
    assert!(r.entries.is_empty());
}

#[test]
fn raft_progress_snapshot_is_stable_across_snapshot_boundary_term_mismatch() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 3,
        snapshot_id: 1,
    });

    let before = r.progress();

    let err = r
        .append_entries_from_leader(
            4,
            5,
            2,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![6].into(),
            }],
            6,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.progress(), before);
}

#[test]
fn raft_follower_append_entries_rejects_prev_index_behind_snapshot_boundary() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 3,
        snapshot_id: 1,
    });

    let err = r
        .append_entries_from_leader(
            4,
            0,
            0,
            vec![LogEntry {
                term: 4,
                index: 1,
                payload: vec![1].into(),
            }],
            1,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.commit_index(), 5);
    assert_eq!(r.applied_index(), 5);
    assert_eq!(r.next_index, 6);
    assert!(r.entries.is_empty());
}

#[test]
fn raft_progress_snapshot_is_stable_across_prev_index_behind_snapshot_boundary() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);

    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 3,
        snapshot_id: 1,
    });

    let before = r.progress();

    let err = r
        .append_entries_from_leader(
            4,
            0,
            0,
            vec![LogEntry {
                term: 4,
                index: 1,
                payload: vec![1].into(),
            }],
            1,
        )
        .unwrap_err();

    assert!(matches!(err, EngineError::ProposalFailed(_)));
    assert_eq!(r.progress(), before);
}
