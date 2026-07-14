use gpu_db_types::{Index, LogEntry, Role, SnapshotMeta, Term};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryState {
    pub term: Term,
    pub snapshot: SnapshotMeta,
    pub committed_entries: Vec<LogEntry>,
    pub applied_index: Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationProgress {
    pub role: Role,
    pub term: Term,
    pub commit_index: Index,
    pub applied_index: Index,
    pub next_index: Index,
    pub snapshot: SnapshotMeta,
    pub committed_but_unapplied_count: usize,
    pub has_committed_entries_pending_apply: bool,
    pub uncommitted_entry_count: usize,
    pub has_uncommitted_entries: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryProgressGap {
    pub commit_index_gap: usize,
    pub applied_index_gap: usize,
    pub next_index_gap: usize,
    pub uncommitted_entry_gap: usize,
}

impl RecoveryProgressGap {
    pub fn has_gap(&self) -> bool {
        self.commit_index_gap > 0
            || self.applied_index_gap > 0
            || self.next_index_gap > 0
            || self.uncommitted_entry_gap > 0
    }

    pub fn has_speculative_tail(&self) -> bool {
        self.next_index_gap > 0 || self.uncommitted_entry_gap > 0
    }

    pub fn is_restart_equivalent(&self) -> bool {
        !self.has_gap()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationStatusSnapshot {
    pub live: ReplicationProgress,
    pub durable: ReplicationProgress,
    pub recovery_gap: RecoveryProgressGap,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReplicationStatusInvariantError {
    #[error("live progress is invalid: {0}")]
    LiveProgress(#[from] ReplicationProgressInvariantError),
    #[error("durable progress is invalid: {0}")]
    DurableProgress(ReplicationProgressInvariantError),
    #[error(
        "durable progress cannot be ahead of live progress for {field}: durable={durable}, live={live}"
    )]
    DurableAheadOfLive {
        field: &'static str,
        durable: u64,
        live: u64,
    },
    #[error(
        "snapshot identity drift at frontier index={last_included_index} term={last_included_term}: durable snapshot_id={durable_snapshot_id}, live snapshot_id={live_snapshot_id}"
    )]
    SnapshotIdentityDrift {
        last_included_index: Index,
        last_included_term: Term,
        durable_snapshot_id: u64,
        live_snapshot_id: u64,
    },
    #[error("durable term {durable} does not match live term {live}")]
    TermMismatch { durable: Term, live: Term },
    #[error("recovery gap {actual:?} does not match live-vs-durable delta {expected:?}")]
    RecoveryGapMismatch {
        expected: RecoveryProgressGap,
        actual: RecoveryProgressGap,
    },
}

impl ReplicationStatusSnapshot {
    pub fn new(
        live: ReplicationProgress,
        durable: ReplicationProgress,
        recovery_gap: RecoveryProgressGap,
    ) -> Result<Self, ReplicationStatusInvariantError> {
        let snapshot = Self {
            live,
            durable,
            recovery_gap,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> Result<(), ReplicationStatusInvariantError> {
        self.live
            .validate()
            .map_err(ReplicationStatusInvariantError::LiveProgress)?;
        self.durable
            .validate()
            .map_err(ReplicationStatusInvariantError::DurableProgress)?;

        if self.durable.term != self.live.term {
            return Err(ReplicationStatusInvariantError::TermMismatch {
                durable: self.durable.term,
                live: self.live.term,
            });
        }

        for (field, durable, live) in [
            (
                "snapshot.last_included_index",
                self.durable.snapshot.last_included_index,
                self.live.snapshot.last_included_index,
            ),
            (
                "snapshot.last_included_term",
                self.durable.snapshot.last_included_term,
                self.live.snapshot.last_included_term,
            ),
            (
                "commit_index",
                self.durable.commit_index,
                self.live.commit_index,
            ),
            (
                "applied_index",
                self.durable.applied_index,
                self.live.applied_index,
            ),
            ("next_index", self.durable.next_index, self.live.next_index),
            (
                "uncommitted_entry_count",
                self.durable.uncommitted_entry_count as u64,
                self.live.uncommitted_entry_count as u64,
            ),
        ] {
            if durable > live {
                return Err(ReplicationStatusInvariantError::DurableAheadOfLive {
                    field,
                    durable,
                    live,
                });
            }
        }

        if self.durable.snapshot.last_included_index == self.live.snapshot.last_included_index
            && self.durable.snapshot.last_included_term == self.live.snapshot.last_included_term
            && self.durable.snapshot.snapshot_id != self.live.snapshot.snapshot_id
        {
            return Err(ReplicationStatusInvariantError::SnapshotIdentityDrift {
                last_included_index: self.live.snapshot.last_included_index,
                last_included_term: self.live.snapshot.last_included_term,
                durable_snapshot_id: self.durable.snapshot.snapshot_id,
                live_snapshot_id: self.live.snapshot.snapshot_id,
            });
        }

        let expected = Self::recovery_gap_between(&self.live, &self.durable);
        if self.recovery_gap != expected {
            return Err(ReplicationStatusInvariantError::RecoveryGapMismatch {
                expected,
                actual: self.recovery_gap.clone(),
            });
        }

        Ok(())
    }

    pub fn is_restart_equivalent(&self) -> bool {
        self.recovery_gap.is_restart_equivalent()
    }

    pub fn has_speculative_tail(&self) -> bool {
        self.recovery_gap.has_speculative_tail()
    }

    pub(super) fn recovery_gap_between(
        live: &ReplicationProgress,
        durable: &ReplicationProgress,
    ) -> RecoveryProgressGap {
        RecoveryProgressGap {
            commit_index_gap: live.commit_index.saturating_sub(durable.commit_index) as usize,
            applied_index_gap: live.applied_index.saturating_sub(durable.applied_index) as usize,
            next_index_gap: live.next_index.saturating_sub(durable.next_index) as usize,
            uncommitted_entry_gap: live
                .uncommitted_entry_count
                .saturating_sub(durable.uncommitted_entry_count),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReplicationProgressInvariantError {
    #[error("applied_index {applied_index} exceeds commit_index {commit_index}")]
    AppliedExceedsCommit {
        applied_index: Index,
        commit_index: Index,
    },
    #[error("next_index {next_index} is behind commit boundary {commit_index}")]
    NextIndexBehindCommit {
        next_index: Index,
        commit_index: Index,
    },
    #[error(
        "committed_but_unapplied_count {committed_but_unapplied_count} does not match commit/apply gap {expected}"
    )]
    PendingApplyCountMismatch {
        committed_but_unapplied_count: usize,
        expected: usize,
    },
    #[error(
        "has_committed_entries_pending_apply {has_pending} does not match commit/apply gap {expected}"
    )]
    PendingApplyFlagMismatch { has_pending: bool, expected: bool },
    #[error(
        "has_uncommitted_entries {has_uncommitted} does not match uncommitted_entry_count {uncommitted_entry_count}"
    )]
    UncommittedFlagMismatch {
        has_uncommitted: bool,
        uncommitted_entry_count: usize,
    },
    #[error("snapshot last_included_index {snapshot_index} exceeds applied_index {applied_index}")]
    SnapshotAheadOfApplied {
        snapshot_index: Index,
        applied_index: Index,
    },
}

impl ReplicationProgress {
    pub fn apply_gap(&self) -> usize {
        self.commit_index.saturating_sub(self.applied_index) as usize
    }

    pub fn is_caught_up(&self) -> bool {
        self.apply_gap() == 0 && !self.has_uncommitted_entries
    }

    pub fn validate(&self) -> Result<(), ReplicationProgressInvariantError> {
        if self.applied_index > self.commit_index {
            return Err(ReplicationProgressInvariantError::AppliedExceedsCommit {
                applied_index: self.applied_index,
                commit_index: self.commit_index,
            });
        }
        if self.next_index < self.commit_index + 1 {
            return Err(ReplicationProgressInvariantError::NextIndexBehindCommit {
                next_index: self.next_index,
                commit_index: self.commit_index,
            });
        }

        let expected_pending = self.apply_gap();
        if self.committed_but_unapplied_count != expected_pending {
            return Err(
                ReplicationProgressInvariantError::PendingApplyCountMismatch {
                    committed_but_unapplied_count: self.committed_but_unapplied_count,
                    expected: expected_pending,
                },
            );
        }

        let expected_has_pending = expected_pending > 0;
        if self.has_committed_entries_pending_apply != expected_has_pending {
            return Err(
                ReplicationProgressInvariantError::PendingApplyFlagMismatch {
                    has_pending: self.has_committed_entries_pending_apply,
                    expected: expected_has_pending,
                },
            );
        }

        let expected_has_uncommitted = self.uncommitted_entry_count > 0;
        if self.has_uncommitted_entries != expected_has_uncommitted {
            return Err(ReplicationProgressInvariantError::UncommittedFlagMismatch {
                has_uncommitted: self.has_uncommitted_entries,
                uncommitted_entry_count: self.uncommitted_entry_count,
            });
        }

        if self.snapshot.last_included_index > self.applied_index {
            return Err(ReplicationProgressInvariantError::SnapshotAheadOfApplied {
                snapshot_index: self.snapshot.last_included_index,
                applied_index: self.applied_index,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecoveryInvariantError {
    #[error("applied_index {applied_index} is behind snapshot boundary {snapshot_index}")]
    AppliedBehindSnapshot {
        applied_index: Index,
        snapshot_index: Index,
    },
    #[error("recovery entries must be contiguous from {expected_index} but saw {actual_index}")]
    NonContiguousEntries {
        expected_index: Index,
        actual_index: Index,
    },
    #[error("recovery entry term {entry_term} exceeds local term {local_term} at index {index}")]
    EntryTermExceedsLocal {
        entry_term: Term,
        local_term: Term,
        index: Index,
    },
    #[error("applied_index {applied_index} exceeds commit boundary {commit_index}")]
    AppliedExceedsCommit {
        applied_index: Index,
        commit_index: Index,
    },
}

impl RecoveryState {
    pub fn commit_index(&self) -> Index {
        self.committed_entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.snapshot.last_included_index)
    }

    pub fn next_index(&self) -> Index {
        self.commit_index() + 1
    }

    pub fn committed_but_unapplied_count(&self) -> usize {
        self.commit_index().saturating_sub(self.applied_index) as usize
    }

    pub fn has_committed_entries_pending_apply(&self) -> bool {
        self.applied_index < self.commit_index()
    }

    pub fn apply_gap(&self) -> usize {
        self.commit_index().saturating_sub(self.applied_index) as usize
    }

    pub fn is_caught_up(&self) -> bool {
        self.apply_gap() == 0
    }

    pub fn progress_as_follower(&self) -> Result<ReplicationProgress, RecoveryInvariantError> {
        self.validate()?;
        let progress = ReplicationProgress {
            role: Role::Follower,
            term: self.term.max(self.snapshot.last_included_term),
            commit_index: self.commit_index(),
            applied_index: self.applied_index,
            next_index: self.next_index(),
            snapshot: self.snapshot.clone(),
            committed_but_unapplied_count: self.committed_but_unapplied_count(),
            has_committed_entries_pending_apply: self.has_committed_entries_pending_apply(),
            uncommitted_entry_count: 0,
            has_uncommitted_entries: false,
        };
        progress.validate().expect(
            "validated recovery state should always map to valid follower replication progress",
        );
        Ok(progress)
    }

    pub fn validate(&self) -> Result<(), RecoveryInvariantError> {
        if self.applied_index < self.snapshot.last_included_index {
            return Err(RecoveryInvariantError::AppliedBehindSnapshot {
                applied_index: self.applied_index,
                snapshot_index: self.snapshot.last_included_index,
            });
        }

        for (expected_index, entry) in
            (self.snapshot.last_included_index + 1..).zip(self.committed_entries.iter())
        {
            if entry.index != expected_index {
                return Err(RecoveryInvariantError::NonContiguousEntries {
                    expected_index,
                    actual_index: entry.index,
                });
            }
            if entry.term > self.term {
                return Err(RecoveryInvariantError::EntryTermExceedsLocal {
                    entry_term: entry.term,
                    local_term: self.term,
                    index: entry.index,
                });
            }
        }

        let commit_index = self.commit_index();
        if self.applied_index > commit_index {
            return Err(RecoveryInvariantError::AppliedExceedsCommit {
                applied_index: self.applied_index,
                commit_index,
            });
        }

        Ok(())
    }
}
