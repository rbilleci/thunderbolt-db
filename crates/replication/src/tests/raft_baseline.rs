#[test]
fn raft_replicator_rejects_proposal_when_not_leader() {
    let mut r = RaftReplicator::new(3);
    let err = r.propose(vec![1].into()).unwrap_err();
    assert!(matches!(err, EngineError::NotLeader));
}

#[test]
fn raft_candidate_rejects_proposal_and_drops_uncommitted_tail() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(1);

    let t1 = r.propose(vec![1].into()).unwrap();
    r.register_follower_ack(t1.index, 1);
    let _uncommitted = r.propose(vec![2].into()).unwrap();

    r.become_candidate(2);
    let err = r.propose(vec![3].into()).unwrap_err();
    assert!(matches!(err, EngineError::NotLeader));

    r.become_leader(3);
    let tok = r.propose(vec![4].into()).unwrap();
    assert_eq!(tok.index, t1.index + 1);
}

#[test]
fn raft_replicator_commits_after_quorum_acks() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(2);

    let t1 = r.propose(vec![1].into()).unwrap();
    assert_eq!(r.commit_index(), 0, "self-ack is not quorum for 3 voters");

    r.register_follower_ack(t1.index, 1);

    assert_eq!(r.commit_index(), t1.index);
    assert_eq!(r.quorum_size(), 2);
    assert_eq!(r.voter_count(), 3);
}

#[test]
fn raft_replicator_commit_index_advances_in_order() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(3);

    let t1 = r.propose(vec![10].into()).unwrap();
    let t2 = r.propose(vec![20].into()).unwrap();

    r.register_follower_ack(t2.index, 2);
    assert_eq!(r.commit_index(), 0, "cannot skip index 1");

    r.register_follower_ack(t1.index, 1);
    assert_eq!(r.commit_index(), t2.index);
}

#[test]
fn raft_progress_snapshot_tracks_quorum_ack_commit_promotion() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(3);

    let t1 = r.propose(vec![10].into()).unwrap();
    let t2 = r.propose(vec![20].into()).unwrap();

    let before = r.progress();
    assert_eq!(before.commit_index, 0);
    assert_eq!(before.applied_index, 0);
    assert_eq!(before.uncommitted_entry_count, 2);
    assert!(before.has_uncommitted_entries);
    assert_eq!(before.committed_but_unapplied_count, 0);
    assert!(!before.has_committed_entries_pending_apply);

    r.register_follower_ack(t2.index, 2);
    let still_blocked = r.progress();
    assert_eq!(still_blocked.commit_index, 0);
    assert_eq!(still_blocked.uncommitted_entry_count, 2);
    assert!(still_blocked.has_uncommitted_entries);

    r.register_follower_ack(t1.index, 1);
    let after = r.progress();
    assert_eq!(after.commit_index, t2.index);
    assert_eq!(after.applied_index, 0);
    assert_eq!(after.uncommitted_entry_count, 0);
    assert!(!after.has_uncommitted_entries);
    assert_eq!(after.committed_but_unapplied_count, t2.index as usize);
    assert!(after.has_committed_entries_pending_apply);
    assert_eq!(after.apply_gap(), t2.index as usize);
    assert!(!after.is_caught_up());
    after.validate().unwrap();
}

#[test]
fn raft_replicator_entry_state_helpers_track_committed_and_uncommitted_work() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(3);

    let t1 = r.propose(vec![10].into()).unwrap();
    let t2 = r.propose(vec![20].into()).unwrap();

    assert_eq!(r.retained_entry_count(), 2);
    assert!(r.has_uncommitted_entries());
    assert_eq!(r.uncommitted_entry_count(), 2);
    assert!(!r.has_committed_entries_pending_apply());
    assert_eq!(r.committed_but_unapplied_count(), 0);

    r.register_follower_ack(t1.index, 1);
    assert!(r.has_committed_entries_pending_apply());
    assert_eq!(r.committed_but_unapplied_count(), 1);
    assert!(r.has_uncommitted_entries());
    assert_eq!(r.uncommitted_entry_count(), 1);

    r.mark_applied(t1.index);
    assert!(!r.has_committed_entries_pending_apply());
    assert_eq!(r.committed_but_unapplied_count(), 0);

    r.register_follower_ack(t2.index, 1);
    assert!(r.has_committed_entries_pending_apply());
    assert_eq!(r.committed_but_unapplied_count(), 1);
    assert!(!r.has_uncommitted_entries());
    assert_eq!(r.uncommitted_entry_count(), 0);
}

#[test]
fn raft_follower_lagging_apply_delay_exposes_pending_apply_until_catch_up() {
    let mut leader = RaftReplicator::new(3);
    leader.become_leader(3);
    let t1 = leader.propose(vec![10].into()).unwrap();
    let t2 = leader.propose(vec![20].into()).unwrap();
    leader.register_follower_ack(t1.index, 1);
    leader.register_follower_ack(t2.index, 1);

    let mut follower = RaftReplicator::new(3);
    follower.become_follower(3);
    follower
        .append_entries_from_leader(
            3,
            0,
            0,
            vec![
                LogEntry {
                    term: 3,
                    index: t1.index,
                    payload: vec![10].into(),
                },
                LogEntry {
                    term: 3,
                    index: t2.index,
                    payload: vec![20].into(),
                },
            ],
            t2.index,
        )
        .unwrap();

    assert_eq!(follower.commit_index(), t2.index);
    assert_eq!(follower.applied_index(), 0);
    assert!(follower.has_committed_entries_pending_apply());
    assert_eq!(follower.committed_but_unapplied_count(), 2);

    follower.mark_applied(t1.index);
    assert_eq!(follower.committed_but_unapplied_count(), 1);
    assert!(follower.has_committed_entries_pending_apply());

    follower.mark_applied(t2.index);
    assert_eq!(follower.applied_index(), t2.index);
    assert!(!follower.has_committed_entries_pending_apply());
    assert_eq!(follower.committed_but_unapplied_count(), 0);
}

#[test]
fn raft_progress_snapshot_clamps_apply_frontier_to_commit_boundary() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(3);
    let t1 = r.propose(vec![10].into()).unwrap();
    let t2 = r.propose(vec![20].into()).unwrap();
    r.register_follower_ack(t1.index, 1);

    r.mark_applied(t2.index + 10);

    let progress = r.progress();
    assert_eq!(progress.commit_index, t1.index);
    assert_eq!(progress.applied_index, t1.index);
    assert_eq!(progress.uncommitted_entry_count, 1);
    assert!(progress.has_uncommitted_entries);
    assert_eq!(progress.apply_gap(), 0);
    assert!(!progress.is_caught_up());
    progress.validate().unwrap();
}

#[test]
fn raft_recovery_state_resumes_pending_apply_without_rewinding_commit() {
    let mut leader = RaftReplicator::new(3);
    leader.become_leader(4);
    let t1 = leader.propose(vec![1].into()).unwrap();
    let t2 = leader.propose(vec![2].into()).unwrap();
    leader.register_follower_ack(t1.index, 1);
    leader.register_follower_ack(t2.index, 1);
    leader.mark_applied(t1.index);
    let snapshot = leader.export_snapshot_meta();

    let resumed = RaftReplicator::resume_as_follower(3, leader.recovery_state()).unwrap();

    assert_eq!(resumed.role(), Role::Follower);
    assert_eq!(resumed.current_term(), 4);
    assert_eq!(resumed.commit_index(), t2.index);
    assert_eq!(resumed.applied_index(), t1.index);
    assert_eq!(resumed.snapshot_meta(), snapshot);
    assert!(resumed.has_committed_entries_pending_apply());
    assert_eq!(resumed.committed_but_unapplied_count(), 1);
    assert_eq!(resumed.next_index, t2.index + 1);
}

#[test]
fn raft_progress_snapshot_tracks_uncommitted_and_pending_apply_work() {
    let mut r = RaftReplicator::new(3);
    r.become_leader(3);

    let t1 = r.propose(vec![10].into()).unwrap();
    let _t2 = r.propose(vec![20].into()).unwrap();
    r.register_follower_ack(t1.index, 1);

    let progress = r.progress();

    assert_eq!(progress.role, Role::Leader);
    assert_eq!(progress.commit_index, t1.index);
    assert_eq!(progress.applied_index, 0);
    assert_eq!(progress.next_index, 3);
    assert_eq!(progress.committed_but_unapplied_count, 1);
    assert!(progress.has_committed_entries_pending_apply);
    assert_eq!(progress.uncommitted_entry_count, 1);
    assert!(progress.has_uncommitted_entries);
    assert_eq!(progress.apply_gap(), 1);
    assert!(!progress.is_caught_up());
    progress.validate().unwrap();
}

#[test]
fn raft_progress_snapshot_stays_consistent_across_snapshot_boundary_append() {
    let mut r = RaftReplicator::new(3);
    r.become_follower(4);
    r.install_snapshot(SnapshotMeta {
        last_included_index: 5,
        last_included_term: 3,
        snapshot_id: 1,
    });

    r.append_entries_from_leader(
        4,
        5,
        3,
        vec![LogEntry {
            term: 4,
            index: 6,
            payload: vec![6].into(),
        }],
        6,
    )
    .unwrap();

    let progress = r.progress();

    assert_eq!(progress.snapshot.last_included_index, 5);
    assert_eq!(progress.commit_index, 6);
    assert_eq!(progress.applied_index, 5);
    assert_eq!(progress.next_index, 7);
    assert_eq!(progress.committed_but_unapplied_count, 1);
    assert!(progress.has_committed_entries_pending_apply);
    assert_eq!(progress.uncommitted_entry_count, 0);
    assert!(!progress.has_uncommitted_entries);
    assert_eq!(progress.apply_gap(), 1);
    assert!(!progress.is_caught_up());
    progress.validate().unwrap();
}
