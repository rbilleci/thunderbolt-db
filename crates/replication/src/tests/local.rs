#[test]
fn commit_index_monotonic() {
    let mut r = LocalReplicator::leader();
    let a = r.propose(vec![1].into()).unwrap();
    let b = r.propose(vec![2].into()).unwrap();

    assert!(b.index > a.index);
    assert_eq!(r.commit_index(), b.index);
    assert!(r.applied_index() <= r.commit_index());
}

#[test]
fn follower_rejects_writes() {
    let mut r = LocalReplicator::leader();
    r.become_follower(2);
    let err = r.propose(vec![1].into()).unwrap_err();
    assert!(matches!(err, EngineError::NotLeader));
    assert_eq!(r.current_term(), 2);
}

#[test]
fn leader_accepts_after_promotion() {
    let mut r = LocalReplicator::leader();
    r.become_follower(2);
    r.become_leader(3);
    let tok = r.propose(vec![42].into()).unwrap();
    assert_eq!(tok.index, 1);
    assert_eq!(r.current_term(), 3);
}

#[test]
fn candidate_rejects_writes() {
    let mut r = LocalReplicator::leader();
    r.become_candidate(2);

    let err = r.propose(vec![1].into()).unwrap_err();

    assert!(matches!(err, EngineError::NotLeader));
    assert_eq!(r.current_term(), 2);
    assert_eq!(r.role(), Role::Candidate);
}

#[test]
fn rollback_unapplied_removes_tail_and_resets_indices() {
    let mut r = LocalReplicator::leader();
    let _ = r.propose(vec![1].into()).unwrap();
    let t2 = r.propose(vec![2].into()).unwrap();
    assert_eq!(r.commit_index(), t2.index);

    r.rollback_unapplied_from(t2.index);

    assert_eq!(r.commit_index(), 1);
    let t3 = r.propose(vec![3].into()).unwrap();
    assert_eq!(t3.index, 2);
}

#[test]
fn local_progress_snapshot_tracks_rollback_of_unapplied_tail() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();
    let t2 = r.propose(vec![2].into()).unwrap();
    r.mark_applied(t1.index);

    let before = r.progress();
    assert_eq!(before.commit_index, t2.index);
    assert_eq!(before.applied_index, t1.index);
    assert_eq!(before.next_index, t2.index + 1);
    assert_eq!(before.committed_but_unapplied_count, 1);
    assert_eq!(before.uncommitted_entry_count, 0);

    r.rollback_unapplied_from(t2.index);

    let after = r.progress();
    assert_eq!(after.commit_index, t1.index);
    assert_eq!(after.applied_index, t1.index);
    assert_eq!(after.next_index, t1.index + 1);
    assert_eq!(after.committed_but_unapplied_count, 0);
    assert!(!after.has_committed_entries_pending_apply);
    assert_eq!(after.uncommitted_entry_count, 0);
    assert!(!after.has_uncommitted_entries);
    assert_eq!(after.apply_gap(), 0);
    assert!(after.is_caught_up());
    after.validate().unwrap();
}

#[test]
fn snapshot_meta_tracks_applied_index() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();
    r.mark_applied(t1.index);

    let meta = r.export_snapshot_meta();

    assert_eq!(meta.last_included_index, t1.index);
    assert_eq!(meta.last_included_term, r.current_term());
    assert_eq!(meta.snapshot_id, 1);
}

#[test]
fn snapshot_meta_preserves_last_applied_term_across_term_bumps() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();
    r.mark_applied(t1.index);

    r.become_follower(5);
    let meta = r.snapshot_meta();

    assert_eq!(r.current_term(), 5);
    assert_eq!(meta.last_included_index, t1.index);
    assert_eq!(meta.last_included_term, 1);
}

/// Locks the contiguity assumption the O(1) `drain_committed_from`/`entry_at` rewrite relies on
/// (audit follow-up): after prefix compaction via `install_snapshot`, `entries[0].index != 1`, so
/// the relative position math `index - entries[0].index` is the ONLY thing that keeps drain/apply
/// selecting the right WAL entries. A future mutator that broke contiguity would silently corrupt
/// the applied set; this test would catch it.
#[test]
fn local_drain_and_entry_at_after_prefix_compaction() {
    let mut r = LocalReplicator::leader();
    for i in 1..=8u64 {
        assert_eq!(r.propose(vec![i as u8].into()).unwrap().index, i);
    }
    // Compact the prefix: apply through 5, snapshot, install -> retained entries become 6,7,8.
    r.mark_applied(5);
    let meta = r.export_snapshot_meta();
    assert_eq!(meta.last_included_index, 5);
    r.install_snapshot(meta);
    // Append more committed entries (indices 9, 10) on top of the compacted log.
    for i in 9..=10u64 {
        assert_eq!(r.propose(vec![i as u8].into()).unwrap().index, i);
    }
    // entries[0].index is now 6 (not 1). drain over EVERY start must equal the predicate it
    // replaced: `index > start && index <= commit_index`, over the retained set [6, commit_index].
    let commit_index = r.commit_index();
    assert_eq!(commit_index, 10);
    let retained_first = 6u64;
    for start in 0..=12u64 {
        let got: Vec<u64> = r.drain_committed_from(start).map(|e| e.index).collect();
        let expected: Vec<u64> = (retained_first..=commit_index)
            .filter(|&idx| idx > start && idx <= commit_index)
            .collect();
        assert_eq!(
            got, expected,
            "drain_committed_from({start}) diverged after prefix compaction"
        );
    }
    // entry_at maps via the relative position; compacted/out-of-range -> None (never a panic or
    // a wrong-position hit).
    assert_eq!(r.entry_at(6).map(|e| e.index), Some(6));
    assert_eq!(r.entry_at(10).map(|e| e.index), Some(10));
    assert_eq!(r.entry_at(5), None, "index 5 was prefix-compacted away");
    assert_eq!(r.entry_at(11), None, "index 11 is past the tail");
    assert_eq!(r.entry_at(0), None);
    // mark_applied past the compaction boundary reads the correct entry's term (no panic).
    r.mark_applied(9);
    assert_eq!(r.applied_index(), 9);
}

#[test]
fn install_snapshot_advances_log_watermarks() {
    let mut r = LocalReplicator::leader();
    let _ = r.propose(vec![1].into()).unwrap();
    let t2 = r.propose(vec![2].into()).unwrap();

    r.install_snapshot(SnapshotMeta {
        last_included_index: t2.index,
        last_included_term: 2,
        snapshot_id: 9,
    });

    assert_eq!(r.commit_index(), t2.index);
    assert_eq!(r.applied_index(), t2.index);
    assert_eq!(r.current_term(), 2);
    assert_eq!(r.snapshot_meta().snapshot_id, 9);

    let t3 = r.propose(vec![3].into()).unwrap();
    assert_eq!(t3.index, t2.index + 1);
}

#[test]
fn install_older_snapshot_is_a_progress_no_op() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();
    r.mark_applied(t1.index);
    let baseline = r.progress();

    r.install_snapshot(SnapshotMeta {
        last_included_index: t1.index.saturating_sub(1),
        last_included_term: 1,
        snapshot_id: baseline.snapshot.snapshot_id + 10,
    });

    assert_eq!(r.progress(), baseline);
    assert_eq!(r.snapshot_meta(), baseline.snapshot);
}

#[test]
fn local_install_snapshot_preserves_next_index_from_uncompacted_tail() {
    let mut r = LocalReplicator::leader();
    let _t1 = r.propose(vec![1].into()).unwrap();
    let t2 = r.propose(vec![2].into()).unwrap();

    r.install_snapshot(SnapshotMeta {
        last_included_index: t2.index - 1,
        last_included_term: 1,
        snapshot_id: 11,
    });

    let next = r.propose(vec![3].into()).unwrap();
    assert_eq!(next.index, t2.index + 1);
}

#[test]
fn local_install_snapshot_updates_snapshot_id_for_same_frontier_same_term() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();
    r.mark_applied(t1.index);

    r.install_snapshot(SnapshotMeta {
        last_included_index: t1.index,
        last_included_term: 1,
        snapshot_id: 11,
    });

    assert_eq!(r.snapshot_meta().snapshot_id, 11);
    assert_eq!(r.snapshot_meta().last_included_index, t1.index);
    assert_eq!(r.progress().snapshot.snapshot_id, 11);
}

#[test]
fn local_install_snapshot_advancing_frontier_replaces_snapshot_identity_exactly() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();
    r.mark_applied(t1.index);

    r.install_snapshot(SnapshotMeta {
        last_included_index: t1.index,
        last_included_term: 1,
        snapshot_id: 11,
    });
    r.install_snapshot(SnapshotMeta {
        last_included_index: t1.index + 1,
        last_included_term: 2,
        snapshot_id: 4,
    });

    let snapshot = r.snapshot_meta();
    assert_eq!(snapshot.last_included_index, t1.index + 1);
    assert_eq!(snapshot.last_included_term, 2);
    assert_eq!(snapshot.snapshot_id, 4);
    assert_eq!(r.progress().snapshot, snapshot);
}

#[test]
fn local_install_snapshot_with_higher_index_lower_term_is_a_progress_no_op() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();
    r.mark_applied(t1.index);
    r.install_snapshot(SnapshotMeta {
        last_included_index: t1.index + 1,
        last_included_term: 3,
        snapshot_id: 11,
    });
    let baseline = r.progress();

    r.install_snapshot(SnapshotMeta {
        last_included_index: t1.index + 2,
        last_included_term: 2,
        snapshot_id: 19,
    });

    assert_eq!(r.progress(), baseline);
    assert_eq!(r.snapshot_meta().snapshot_id, 11);
}

#[test]
fn mark_applied_does_not_exceed_commit_index() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();

    r.mark_applied(t1.index + 10);

    assert_eq!(r.applied_index(), t1.index);
}

#[test]
fn local_progress_snapshot_clamps_apply_frontier_to_commit_boundary() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();

    r.mark_applied(t1.index + 10);

    let progress = r.progress();
    assert_eq!(progress.commit_index, t1.index);
    assert_eq!(progress.applied_index, t1.index);
    assert_eq!(progress.apply_gap(), 0);
    assert!(progress.is_caught_up());
}

#[test]
fn local_replicator_pending_apply_helpers_track_committed_tail() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();
    let _t2 = r.propose(vec![2].into()).unwrap();

    assert_eq!(r.retained_entry_count(), 2);
    assert!(r.has_committed_entries_pending_apply());
    assert_eq!(r.committed_but_unapplied_count(), 2);

    r.mark_applied(t1.index);
    assert!(r.has_committed_entries_pending_apply());
    assert_eq!(r.committed_but_unapplied_count(), 1);

    r.mark_applied(r.commit_index());
    assert!(!r.has_committed_entries_pending_apply());
    assert_eq!(r.committed_but_unapplied_count(), 0);
}

#[test]
fn local_progress_snapshot_matches_commit_and_apply_state() {
    let mut r = LocalReplicator::leader();
    let t1 = r.propose(vec![1].into()).unwrap();
    let _t2 = r.propose(vec![2].into()).unwrap();
    r.mark_applied(t1.index);

    let progress = r.progress();

    assert_eq!(progress.role, Role::Leader);
    assert_eq!(progress.commit_index, 2);
    assert_eq!(progress.applied_index, 1);
    assert_eq!(progress.next_index, 3);
    assert_eq!(progress.committed_but_unapplied_count, 1);
    assert!(progress.has_committed_entries_pending_apply);
    assert_eq!(progress.uncommitted_entry_count, 0);
    assert!(!progress.has_uncommitted_entries);
    assert_eq!(progress.apply_gap(), 1);
    assert!(!progress.is_caught_up());
    progress.validate().unwrap();
}

#[test]
fn local_status_snapshot_is_always_restart_equivalent() {
    let mut local = LocalReplicator::leader();
    let token = local.propose(b"set a=1".to_vec().into()).unwrap();
    local.mark_applied(token.index);

    let status = local.status_snapshot();

    assert_eq!(status.live, local.progress());
    assert_eq!(status.durable, local.progress());
    assert!(status.is_restart_equivalent());
    assert!(!status.has_speculative_tail());
}
