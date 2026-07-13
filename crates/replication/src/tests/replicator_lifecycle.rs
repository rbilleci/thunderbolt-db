#[test]
fn raft_resume_rejects_non_contiguous_recovery_entries() {
    let err = RaftReplicator::resume_as_follower(
        3,
        RecoveryState {
            term: 5,
            snapshot: SnapshotMeta {
                last_included_index: 3,
                last_included_term: 4,
                snapshot_id: 8,
            },
            committed_entries: vec![LogEntry {
                term: 5,
                index: 5,
                payload: vec![9].into(),
            }],
            applied_index: 3,
        },
    )
    .unwrap_err();

    assert!(matches!(err, EngineError::ApplyFailed(_)));
}

#[test]
fn raft_single_node_leader_commits_immediately() {
    let mut r = RaftReplicator::single_node_leader();
    let tok = r.propose(vec![7].into()).unwrap();
    assert_eq!(r.commit_index(), tok.index);
}

#[test]
fn raft_progress_snapshot_tracks_single_node_immediate_commit() {
    let mut r = RaftReplicator::single_node_leader();
    let tok = r.propose(vec![7].into()).unwrap();

    let progress = r.progress();
    assert_eq!(progress.role, Role::Leader);
    assert_eq!(progress.commit_index, tok.index);
    assert_eq!(progress.applied_index, 0);
    assert_eq!(progress.next_index, tok.index + 1);
    assert_eq!(progress.uncommitted_entry_count, 0);
    assert!(!progress.has_uncommitted_entries);
    assert_eq!(progress.committed_but_unapplied_count, tok.index as usize);
    assert!(progress.has_committed_entries_pending_apply);
    assert_eq!(progress.apply_gap(), tok.index as usize);
    assert!(!progress.is_caught_up());
    progress.validate().unwrap();
}

#[test]
fn raft_follower_acks_are_ignored_when_not_leader() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);
    let t1 = r.propose(vec![1].into()).unwrap();

    r.become_follower(2);
    r.register_follower_ack(t1.index, 1);

    assert_eq!(r.commit_index(), 0);
}

#[test]
fn raft_progress_snapshot_is_stable_when_acks_arrive_off_leader() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);
    let t1 = r.propose(vec![1].into()).unwrap();

    r.become_follower(2);
    let before = r.progress();
    r.register_follower_ack(t1.index, 1);
    let after = r.progress();

    assert_eq!(after, before);
}

#[test]
fn raft_rejects_ack_for_unknown_index() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);
    let _ = r.propose(vec![1].into()).unwrap();

    r.register_follower_ack(2, 1);

    assert_eq!(r.commit_index(), 0);
}

#[test]
fn raft_progress_snapshot_is_stable_when_ack_targets_unknown_index() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);
    let _ = r.propose(vec![1].into()).unwrap();

    let before = r.progress();
    r.register_follower_ack(2, 1);
    let after = r.progress();

    assert_eq!(after, before);
}

#[test]
fn raft_ignores_reserved_self_ack_follower_id() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 0);

    assert_eq!(
        r.commit_index(),
        0,
        "follower id 0 is reserved for the leader self-ack"
    );
}

#[test]
fn raft_progress_snapshot_is_stable_for_reserved_self_ack() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    let before = r.progress();
    r.register_follower_ack(t1.index, 0);
    let after = r.progress();

    assert_eq!(after, before);
}

#[test]
fn raft_duplicate_follower_ack_does_not_count_twice() {
    let mut r = RaftReplicator::new(5);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.register_follower_ack(t1.index, 2);
    assert_eq!(r.commit_index(), t1.index);

    let t2 = r.propose(vec![2].into()).unwrap();
    r.register_follower_ack(t2.index, 1);
    r.register_follower_ack(t2.index, 1);

    assert_eq!(
        r.commit_index(),
        t1.index,
        "same follower should not be able to satisfy quorum twice"
    );

    r.register_follower_ack(t2.index, 2);
    assert_eq!(r.commit_index(), t2.index);
}

#[test]
fn raft_progress_snapshot_is_stable_for_duplicate_follower_ack_until_quorum_changes() {
    let mut r = RaftReplicator::new(5);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.register_follower_ack(t1.index, 2);
    assert_eq!(r.commit_index(), t1.index);

    let t2 = r.propose(vec![2].into()).unwrap();
    r.register_follower_ack(t2.index, 1);
    let before_duplicate = r.progress();

    r.register_follower_ack(t2.index, 1);
    let after_duplicate = r.progress();

    assert_eq!(after_duplicate, before_duplicate);

    r.register_follower_ack(t2.index, 2);
    let after_quorum = r.progress();
    assert_ne!(after_quorum, after_duplicate);
    assert_eq!(after_quorum.commit_index, t2.index);
}

#[test]
fn raft_prunes_ack_tracking_for_committed_entries() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    let t2 = r.propose(vec![2].into()).unwrap();

    assert!(r.ack_counts.contains_key(&t1.index));
    assert!(r.ack_counts.contains_key(&t2.index));

    r.register_follower_ack(t1.index, 1);
    assert_eq!(r.commit_index(), t1.index);
    assert!(!r.ack_counts.contains_key(&t1.index));
    assert!(r.ack_counts.contains_key(&t2.index));

    r.register_follower_ack(t2.index, 1);
    assert_eq!(r.commit_index(), t2.index);
    assert!(r.ack_counts.is_empty());
}

#[test]
fn raft_progress_snapshot_is_stable_across_ack_tracking_pruning() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    let t2 = r.propose(vec![2].into()).unwrap();

    r.register_follower_ack(t1.index, 1);
    let after_first_commit = r.progress();
    assert_eq!(after_first_commit.commit_index, t1.index);
    assert_eq!(after_first_commit.uncommitted_entry_count, 1);

    r.register_follower_ack(t2.index, 1);
    let after_second_commit = r.progress();
    assert_eq!(after_second_commit.commit_index, t2.index);
    assert_eq!(after_second_commit.uncommitted_entry_count, 0);
    assert!(r.ack_counts.is_empty());
    after_second_commit.validate().unwrap();
}

#[test]
fn raft_role_change_discards_uncommitted_tail() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    assert_eq!(r.commit_index(), t1.index);

    let _t2_uncommitted = r.propose(vec![2].into()).unwrap();
    assert_eq!(r.commit_index(), t1.index);

    r.become_follower(2);
    r.become_leader(3);

    let t2_new_epoch = r.propose(vec![3].into()).unwrap();
    assert_eq!(t2_new_epoch.index, t1.index + 1);

    r.register_follower_ack(t2_new_epoch.index, 1);
    assert_eq!(r.commit_index(), t2_new_epoch.index);
}

#[test]
fn raft_progress_snapshot_discards_uncommitted_tail_across_role_change() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    let _t2_uncommitted = r.propose(vec![2].into()).unwrap();

    let before = r.progress();
    assert_eq!(before.commit_index, t1.index);
    assert_eq!(before.uncommitted_entry_count, 1);
    assert!(before.has_uncommitted_entries);
    assert!(!before.is_caught_up());
    let before_gap = r.recovery_progress_gap();
    assert_eq!(before_gap.next_index_gap, 1);
    assert_eq!(before_gap.uncommitted_entry_gap, 1);
    assert!(before_gap.has_speculative_tail());

    r.become_follower(2);
    let after_follower = r.progress();
    assert_eq!(after_follower.role, Role::Follower);
    assert_eq!(after_follower.commit_index, t1.index);
    assert_eq!(after_follower.next_index, t1.index + 1);
    assert_eq!(after_follower.uncommitted_entry_count, 0);
    assert!(!after_follower.has_uncommitted_entries);
    assert_eq!(
        r.recovery_progress_gap(),
        RecoveryProgressGap {
            commit_index_gap: 0,
            applied_index_gap: 0,
            next_index_gap: 0,
            uncommitted_entry_gap: 0,
        }
    );

    r.become_leader(3);
    let after_leader = r.progress();
    assert_eq!(after_leader.role, Role::Leader);
    assert_eq!(after_leader.commit_index, t1.index);
    assert_eq!(after_leader.next_index, t1.index + 1);
    assert_eq!(after_leader.uncommitted_entry_count, 0);
    assert!(!after_leader.has_uncommitted_entries);
    after_leader.validate().unwrap();
    assert!(r.recovery_progress_gap().is_restart_equivalent());
}

#[test]
fn raft_truncate_uncommitted_from_drops_tail_and_resets_next_index() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    assert_eq!(r.commit_index(), t1.index);

    let t2 = r.propose(vec![2].into()).unwrap();
    let t3 = r.propose(vec![3].into()).unwrap();
    assert!(r.ack_counts.contains_key(&t2.index));
    assert!(r.ack_counts.contains_key(&t3.index));

    r.truncate_uncommitted_from(t2.index);

    assert_eq!(r.commit_index(), t1.index);
    assert!(r.entries.iter().all(|entry| entry.index <= t1.index));
    assert!(r.ack_counts.is_empty());

    let replacement = r.propose(vec![9].into()).unwrap();
    assert_eq!(replacement.index, t1.index + 1);
}

#[test]
fn raft_progress_snapshot_tracks_uncommitted_tail_truncation() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    let t2 = r.propose(vec![2].into()).unwrap();
    let t3 = r.propose(vec![3].into()).unwrap();

    let before = r.progress();
    assert_eq!(before.commit_index, t1.index);
    assert_eq!(before.next_index, t3.index + 1);
    assert_eq!(before.uncommitted_entry_count, 2);
    assert!(before.has_uncommitted_entries);
    assert!(!before.is_caught_up());

    r.truncate_uncommitted_from(t2.index);

    let after = r.progress();
    assert_eq!(after.commit_index, t1.index);
    assert_eq!(after.applied_index, 0);
    assert_eq!(after.next_index, t1.index + 1);
    assert_eq!(after.uncommitted_entry_count, 0);
    assert!(!after.has_uncommitted_entries);
    assert_eq!(after.committed_but_unapplied_count, t1.index as usize);
    assert!(after.has_committed_entries_pending_apply);
    assert_eq!(after.apply_gap(), t1.index as usize);
    assert!(!after.is_caught_up());
    after.validate().unwrap();
}

#[test]
fn raft_truncate_uncommitted_from_ignores_committed_boundary() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    assert_eq!(r.commit_index(), t1.index);

    r.truncate_uncommitted_from(t1.index);

    assert_eq!(r.commit_index(), t1.index);
    assert_eq!(r.next_index, t1.index + 1);
}

#[test]
fn raft_progress_snapshot_is_stable_when_truncation_targets_committed_boundary() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);

    let before = r.progress();
    r.truncate_uncommitted_from(t1.index);
    let after = r.progress();

    assert_eq!(after, before);
}

#[test]
fn local_wait_committed_rejects_uncommitted_token() {
    let mut r = LocalReplicator::leader();
    let token = r.propose(vec![1].into()).unwrap();
    r.rollback_unapplied_from(token.index);

    let err = r
        .wait_committed(token, std::time::Duration::from_millis(1))
        .unwrap_err();
    assert!(matches!(err, EngineError::ProposalFailed(_)));
}

#[test]
fn raft_wait_committed_resolves_after_quorum_commit() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(2);

    let token = r.propose(vec![1].into()).unwrap();
    let pending = r.wait_committed(token, std::time::Duration::from_millis(1));
    assert!(matches!(pending, Err(EngineError::ProposalFailed(_))));

    r.register_follower_ack(token.index, 1);
    let committed = r
        .wait_committed(token, std::time::Duration::from_millis(1))
        .unwrap();
    assert_eq!(committed, token.index);
}

#[test]
fn raft_progress_snapshot_is_stable_across_wait_committed_polling() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(2);

    let token = r.propose(vec![1].into()).unwrap();
    let before_pending = r.progress();
    let pending = r.wait_committed(token, std::time::Duration::from_millis(1));
    assert!(matches!(pending, Err(EngineError::ProposalFailed(_))));
    let after_pending = r.progress();
    assert_eq!(after_pending, before_pending);

    r.register_follower_ack(token.index, 1);
    let before_resolved = r.progress();
    let committed = r
        .wait_committed(token, std::time::Duration::from_millis(1))
        .unwrap();
    assert_eq!(committed, token.index);
    let after_resolved = r.progress();
    assert_eq!(after_resolved, before_resolved);
}

#[test]
fn raft_install_snapshot_prunes_ack_tracking_and_uncompacted_entries() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(4);

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
        snapshot_id: 42,
    });

    assert_eq!(r.current_term(), 5);
    assert_eq!(r.commit_index(), t2.index);
    assert_eq!(r.applied_index(), t2.index);
    assert_eq!(r.snapshot_meta().snapshot_id, 42);
    assert!(!r.ack_counts.contains_key(&t1.index));
    assert!(!r.ack_counts.contains_key(&t2.index));
    assert!(!r.ack_counts.contains_key(&t3.index));
    assert!(r.entries.is_empty());
}

#[test]
fn raft_progress_snapshot_advances_consistently_after_snapshot_install() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(4);

    let t1 = r.propose(vec![1].into()).unwrap();
    let t2 = r.propose(vec![2].into()).unwrap();

    r.register_follower_ack(t1.index, 1);
    r.register_follower_ack(t2.index, 1);
    r.mark_applied(t1.index);

    let before = r.progress();
    assert_eq!(before.commit_index, t2.index);
    assert_eq!(before.applied_index, t1.index);
    assert_eq!(before.apply_gap(), 1);

    r.install_snapshot(SnapshotMeta {
        last_included_index: t2.index,
        last_included_term: 5,
        snapshot_id: 42,
    });

    let after = r.progress();
    assert_eq!(after.commit_index, t2.index);
    assert_eq!(after.applied_index, t2.index);
    assert_eq!(after.snapshot.snapshot_id, 42);
    assert_eq!(after.snapshot.last_included_index, t2.index);
    assert_eq!(after.apply_gap(), 0);
    assert!(after.is_caught_up());
    after.validate().unwrap();
}

#[test]
fn raft_install_snapshot_preserves_next_index_from_uncompacted_tail() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(3);

    let t1 = r.propose(vec![1].into()).unwrap();
    let _t2 = r.propose(vec![2].into()).unwrap();

    r.register_follower_ack(t1.index, 1);
    assert_eq!(r.commit_index(), t1.index);

    r.install_snapshot(SnapshotMeta {
        last_included_index: t1.index,
        last_included_term: 3,
        snapshot_id: 7,
    });

    let replacement = r.propose(vec![9].into()).unwrap();
    assert_eq!(replacement.index, 3);
}

#[test]
fn raft_progress_snapshot_preserves_uncommitted_tail_after_snapshot_install() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(3);

    let t1 = r.propose(vec![1].into()).unwrap();
    let t2 = r.propose(vec![2].into()).unwrap();

    r.register_follower_ack(t1.index, 1);
    assert_eq!(r.commit_index(), t1.index);

    r.install_snapshot(SnapshotMeta {
        last_included_index: t1.index,
        last_included_term: 3,
        snapshot_id: 7,
    });

    let progress = r.progress();
    assert_eq!(progress.commit_index, t1.index);
    assert_eq!(progress.applied_index, t1.index);
    assert_eq!(progress.next_index, t2.index + 1);
    assert_eq!(progress.uncommitted_entry_count, 1);
    assert!(progress.has_uncommitted_entries);
    assert!(!progress.is_caught_up());
    progress.validate().unwrap();
}

#[test]
fn raft_snapshot_meta_preserves_last_applied_term_across_term_bumps() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(2);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    r.mark_applied(t1.index);

    r.become_follower(7);
    let meta = r.snapshot_meta();

    assert_eq!(r.current_term(), 7);
    assert_eq!(meta.last_included_index, t1.index);
    assert_eq!(meta.last_included_term, 2);
}
