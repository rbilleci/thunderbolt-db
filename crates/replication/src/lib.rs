use std::collections::{BTreeMap, BTreeSet};

use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term};

mod local;
mod operational;
mod progress;
mod rpc;
mod transport;

pub use local::LocalReplicator;
pub use operational::{
    OperationalClusterSmokeReport, OperationalDeploymentPreflightReport,
    OperationalElectionSmokeReport, OperationalPackageSmokeReport, OperationalTransportSmokeReport,
};
pub use progress::{
    RecoveryInvariantError, RecoveryProgressGap, RecoveryState, ReplicationProgress,
    ReplicationProgressInvariantError, ReplicationStatusInvariantError, ReplicationStatusSnapshot,
};
pub use rpc::{
    AppendEntriesRequest, AppendEntriesResponse, RequestVoteRequest, RequestVoteResponse,
};
pub use transport::{
    load_replication_mtls_client_config, load_replication_mtls_server_config,
    load_replication_tls_client_config_without_client_auth, send_append_entries_mtls_once,
    send_append_entries_once, serve_append_entries_mtls_once, serve_append_entries_once,
};

impl AppendEntriesRequest {
    pub fn apply_to(&self, follower: &mut RaftReplicator) -> AppendEntriesResponse {
        let result = follower.append_entries_from_leader(
            self.leader_term,
            self.prev_log_index,
            self.prev_log_term,
            self.entries.clone(),
            self.leader_commit,
        );
        AppendEntriesResponse {
            accepted: result.is_ok(),
            follower_term: follower.current_term(),
            follower_commit_index: follower.commit_index(),
            follower_applied_index: follower.applied_index(),
            error: result.err().map(|err| err.to_string()),
        }
    }
}

pub trait LogReplicator {
    fn propose(&mut self, payload: std::sync::Arc<[u8]>) -> Result<CommitToken, EngineError>;

    /// W1b — drop retained log entries that are both COMMITTED and APPLIED (never read again:
    /// `drain_committed_from` only yields entries past the applied index). Default no-op; the
    /// local single-node replicator compacts its in-memory Vec (the third unbounded per-commit
    /// structure alongside the WAL buffer and the timestamp map); Raft keeps its own log
    /// management (follower catch-up may still need applied entries).
    fn compact_applied_prefix(&mut self) {}
    fn wait_committed(
        &self,
        token: CommitToken,
        timeout: std::time::Duration,
    ) -> Result<Index, EngineError>;
    fn role(&self) -> Role;
    fn current_term(&self) -> Term;
    fn commit_index(&self) -> Index;
    fn applied_index(&self) -> Index;
    fn snapshot_meta(&self) -> SnapshotMeta;
}

pub trait ReplicatedStateMachine {
    fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError>;
}

#[derive(Debug)]
pub struct RaftReplicator {
    term: Term,
    next_index: Index,
    commit_index: Index,
    applied_index: Index,
    applied_term: Term,
    role: Role,
    entries: Vec<LogEntry>,
    snapshot_id: u64,
    compacted_index: Index,
    compacted_term: Term,
    voters: usize,
    quorum: usize,
    voted_for: Option<u64>,
    ack_counts: BTreeMap<Index, BTreeSet<u64>>,
}

impl RaftReplicator {
    pub fn new(voters: usize) -> Self {
        assert!(voters >= 1, "raft requires at least one voter");
        let quorum = (voters / 2) + 1;
        Self {
            term: 1,
            next_index: 1,
            commit_index: 0,
            applied_index: 0,
            applied_term: 0,
            role: Role::Follower,
            entries: Vec::new(),
            snapshot_id: 0,
            compacted_index: 0,
            compacted_term: 0,
            voters,
            quorum,
            voted_for: None,
            ack_counts: BTreeMap::new(),
        }
    }

    pub fn single_node_leader() -> Self {
        let mut s = Self::new(1);
        s.role = Role::Leader;
        s
    }

    pub fn resume_as_follower(voters: usize, recovery: RecoveryState) -> Result<Self, EngineError> {
        recovery
            .validate()
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;

        let commit_index = recovery.commit_index();
        let next_index = recovery.next_index();

        let mut replicator = Self::new(voters);
        replicator.term = recovery.term.max(recovery.snapshot.last_included_term);
        replicator.role = Role::Follower;
        replicator.commit_index = commit_index;
        replicator.applied_index = recovery.applied_index;
        replicator.applied_term = if recovery.applied_index == recovery.snapshot.last_included_index
        {
            recovery.snapshot.last_included_term
        } else {
            recovery
                .committed_entries
                .iter()
                .find(|entry| entry.index == recovery.applied_index)
                .map(|entry| entry.term)
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "missing applied entry {} in recovery state",
                        recovery.applied_index
                    ))
                })?
        };
        replicator.snapshot_id = recovery.snapshot.snapshot_id;
        replicator.compacted_index = recovery.snapshot.last_included_index;
        replicator.compacted_term = recovery.snapshot.last_included_term;
        replicator.entries = recovery.committed_entries;
        replicator.next_index = next_index;
        Ok(replicator)
    }

    pub fn become_follower(&mut self, term: Term) {
        let previous_term = self.term;
        self.term = self.term.max(term);
        self.role = Role::Follower;
        if self.term > previous_term {
            self.voted_for = None;
        }
        self.entries.retain(|e| e.index <= self.commit_index);
        self.next_index = self.commit_index + 1;
        self.ack_counts.clear();
    }

    pub fn become_leader(&mut self, term: Term) {
        let previous_term = self.term;
        self.term = self.term.max(term);
        self.role = Role::Leader;
        if self.term > previous_term {
            self.voted_for = None;
        }
        self.entries.retain(|e| e.index <= self.commit_index);
        self.next_index = self.commit_index + 1;
        self.ack_counts.clear();
    }

    pub fn become_candidate(&mut self, term: Term) {
        let previous_term = self.term;
        self.term = self.term.max(term);
        self.role = Role::Candidate;
        if self.term > previous_term {
            self.voted_for = None;
        }
        self.entries.retain(|e| e.index <= self.commit_index);
        self.next_index = self.commit_index + 1;
        self.ack_counts.clear();
    }

    pub fn start_candidate_election(&mut self, candidate_id: u64) -> RequestVoteRequest {
        assert!(candidate_id > 0, "candidate id must be non-zero");
        self.term += 1;
        self.role = Role::Candidate;
        self.voted_for = Some(candidate_id);
        self.ack_counts.clear();
        let (last_log_index, last_log_term) = self.last_log_position();
        RequestVoteRequest {
            candidate_term: self.term,
            candidate_id,
            last_log_index,
            last_log_term,
        }
    }

    pub fn request_vote_from_candidate(
        &mut self,
        request: &RequestVoteRequest,
    ) -> RequestVoteResponse {
        if request.candidate_term < self.term {
            return RequestVoteResponse {
                granted: false,
                voter_term: self.term,
                error: Some(format!(
                    "stale candidate term {} (local term {})",
                    request.candidate_term, self.term
                )),
            };
        }

        if request.candidate_term > self.term {
            self.term = request.candidate_term;
            self.role = Role::Follower;
            self.voted_for = None;
            self.ack_counts.clear();
        }

        let already_voted_elsewhere = self
            .voted_for
            .is_some_and(|voted_for| voted_for != request.candidate_id);
        if already_voted_elsewhere {
            return RequestVoteResponse {
                granted: false,
                voter_term: self.term,
                error: Some(format!(
                    "already voted for candidate {} in term {}",
                    self.voted_for.expect("checked is_some above"),
                    self.term
                )),
            };
        }

        if !self.candidate_log_is_up_to_date(request.last_log_index, request.last_log_term) {
            return RequestVoteResponse {
                granted: false,
                voter_term: self.term,
                error: Some(format!(
                    "candidate log is behind voter log at index {} term {}",
                    self.last_log_position().0,
                    self.last_log_position().1
                )),
            };
        }

        self.voted_for = Some(request.candidate_id);
        self.role = Role::Follower;
        RequestVoteResponse {
            granted: true,
            voter_term: self.term,
            error: None,
        }
    }

    pub fn voter_count(&self) -> usize {
        self.voters
    }

    pub fn quorum_size(&self) -> usize {
        self.quorum
    }

    pub fn drain_committed_from(&self, start_exclusive: Index) -> impl Iterator<Item = &LogEntry> {
        self.entries
            .iter()
            .filter(move |e| e.index > start_exclusive && e.index <= self.commit_index)
    }

    pub fn retained_entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn committed_but_unapplied_count(&self) -> usize {
        self.commit_index.saturating_sub(self.applied_index) as usize
    }

    pub fn has_committed_entries_pending_apply(&self) -> bool {
        self.commit_index > self.applied_index
    }

    pub fn uncommitted_entry_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.index > self.commit_index)
            .count()
    }

    pub fn has_uncommitted_entries(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.index > self.commit_index)
    }

    pub fn mark_applied(&mut self, idx: Index) {
        let bounded = idx.min(self.commit_index);
        if bounded <= self.applied_index {
            return;
        }

        self.applied_index = bounded;
        if let Some(entry) = self.entries.iter().find(|entry| entry.index == bounded) {
            self.applied_term = entry.term;
        }
    }

    pub fn truncate_uncommitted_from(&mut self, index_inclusive: Index) {
        if index_inclusive <= self.commit_index {
            return;
        }

        self.entries.retain(|e| e.index < index_inclusive);
        self.ack_counts.retain(|idx, _| *idx < index_inclusive);

        let tail_index = self
            .entries
            .last()
            .map(|e| e.index)
            .unwrap_or(self.commit_index);
        self.next_index = tail_index + 1;
    }

    pub fn register_follower_ack(&mut self, index: Index, follower_id: u64) {
        if self.role != Role::Leader || index == 0 || index >= self.next_index || follower_id == 0 {
            return;
        }

        let Some(acks) = self.ack_counts.get_mut(&index) else {
            return;
        };
        acks.insert(follower_id);

        while self.commit_index + 1 < self.next_index {
            let next = self.commit_index + 1;
            let Some(acks) = self.ack_counts.get(&next) else {
                break;
            };
            if acks.len() >= self.quorum {
                self.commit_index = next;
            } else {
                break;
            }
        }

        self.ack_counts.retain(|idx, _| *idx > self.commit_index);
    }

    pub fn export_snapshot_meta(&mut self) -> SnapshotMeta {
        self.snapshot_id += 1;
        self.snapshot_meta()
    }

    pub fn recovery_state(&self) -> RecoveryState {
        let snapshot = self.snapshot_meta();
        RecoveryState {
            term: self.term,
            snapshot: snapshot.clone(),
            committed_entries: self
                .entries
                .iter()
                .filter(|entry| {
                    entry.index > snapshot.last_included_index && entry.index <= self.commit_index
                })
                .cloned()
                .collect(),
            applied_index: self.applied_index,
        }
    }

    pub fn recovery_progress(&self) -> ReplicationProgress {
        self.recovery_state().progress_as_follower().expect(
            "live raft recovery state should always map to valid follower recovery progress",
        )
    }

    pub fn recovery_progress_gap(&self) -> RecoveryProgressGap {
        ReplicationStatusSnapshot::recovery_gap_between(&self.progress(), &self.recovery_progress())
    }

    pub fn status_snapshot(&self) -> ReplicationStatusSnapshot {
        ReplicationStatusSnapshot::new(
            self.progress(),
            self.recovery_progress(),
            self.recovery_progress_gap(),
        )
        .expect("raft replication status snapshot invariants should hold")
    }

    pub fn progress(&self) -> ReplicationProgress {
        let progress = ReplicationProgress {
            role: self.role,
            term: self.term,
            commit_index: self.commit_index,
            applied_index: self.applied_index,
            next_index: self.next_index,
            snapshot: self.snapshot_meta(),
            committed_but_unapplied_count: self.committed_but_unapplied_count(),
            has_committed_entries_pending_apply: self.has_committed_entries_pending_apply(),
            uncommitted_entry_count: self.uncommitted_entry_count(),
            has_uncommitted_entries: self.has_uncommitted_entries(),
        };
        progress
            .validate()
            .expect("raft replication progress invariants should hold");
        progress
    }

    pub fn append_entries_from_leader(
        &mut self,
        leader_term: Term,
        prev_log_index: Index,
        prev_log_term: Term,
        entries: Vec<LogEntry>,
        leader_commit: Index,
    ) -> Result<(), EngineError> {
        if self.role == Role::Leader {
            return Err(EngineError::ProposalFailed(
                "leader cannot accept follower append path".to_string(),
            ));
        }

        if leader_term < self.term {
            return Err(EngineError::ProposalFailed(format!(
                "stale leader term {} (local term {})",
                leader_term, self.term
            )));
        }

        if self.role != Role::Follower || leader_term > self.term {
            self.become_follower(leader_term);
        } else {
            self.term = leader_term;
            self.role = Role::Follower;
        }

        if prev_log_index < self.compacted_index {
            return Err(EngineError::ProposalFailed(format!(
                "prev_log_index={} is behind compacted boundary {}",
                prev_log_index, self.compacted_index
            )));
        }

        if prev_log_index > 0 {
            let Some(local_prev_term) = self.term_at(prev_log_index) else {
                return Err(EngineError::ProposalFailed(format!(
                    "missing prev_log_index={} for append",
                    prev_log_index
                )));
            };

            if local_prev_term != prev_log_term {
                return Err(EngineError::ProposalFailed(format!(
                    "prev_log_term mismatch at index {}: local={}, remote={}",
                    prev_log_index, local_prev_term, prev_log_term
                )));
            }
        }

        for (expected_index, entry) in (prev_log_index + 1..).zip(entries.iter()) {
            if entry.term > leader_term {
                return Err(EngineError::ProposalFailed(format!(
                    "entry term {} exceeds leader term {} at index {}",
                    entry.term, leader_term, entry.index
                )));
            }

            if entry.index != expected_index {
                return Err(EngineError::ProposalFailed(format!(
                    "append entries must be contiguous from {} but saw {}",
                    prev_log_index + 1,
                    entry.index
                )));
            }
        }

        for incoming in entries {
            if let Some(existing) = self.entries.iter().find(|e| e.index == incoming.index) {
                if existing.term == incoming.term {
                    if existing.payload != incoming.payload {
                        return Err(EngineError::ProposalFailed(format!(
                            "payload mismatch at index {} term {}",
                            incoming.index, incoming.term
                        )));
                    }
                    continue;
                }

                if incoming.index <= self.commit_index {
                    return Err(EngineError::ProposalFailed(format!(
                        "refusing to overwrite committed index {}",
                        incoming.index
                    )));
                }

                self.truncate_uncommitted_from(incoming.index);
            }

            if self
                .entries
                .iter()
                .all(|entry| entry.index != incoming.index)
            {
                self.entries.push(incoming);
            }
        }

        self.entries.sort_by_key(|entry| entry.index);

        let last_local_index = self
            .entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.commit_index);
        let target_commit = leader_commit.min(last_local_index);
        if target_commit > self.commit_index {
            self.commit_index = target_commit;
        }
        self.next_index = last_local_index + 1;

        Ok(())
    }

    pub fn install_snapshot(&mut self, meta: SnapshotMeta) {
        let current = self.snapshot_meta();
        let advances_frontier = meta.last_included_index > current.last_included_index;
        let same_frontier_same_term = meta.last_included_index == current.last_included_index
            && meta.last_included_term == current.last_included_term;
        let retains_existing_suffix = same_frontier_same_term
            || self.term_at(meta.last_included_index) == Some(meta.last_included_term);
        let regresses_term_on_advanced_frontier =
            advances_frontier && meta.last_included_term < current.last_included_term;
        if regresses_term_on_advanced_frontier || (!advances_frontier && !same_frontier_same_term) {
            return;
        }

        self.term = self.term.max(meta.last_included_term);
        self.commit_index = self.commit_index.max(meta.last_included_index);
        if meta.last_included_index > self.compacted_index {
            self.compacted_index = meta.last_included_index;
            self.compacted_term = meta.last_included_term;
        }
        if meta.last_included_index > self.applied_index {
            self.applied_index = meta.last_included_index;
            self.applied_term = meta.last_included_term;
        }
        self.snapshot_id = meta.snapshot_id;
        self.entries.retain(|entry| {
            entry.index > meta.last_included_index
                && (retains_existing_suffix || entry.index <= self.commit_index)
        });
        if retains_existing_suffix {
            self.ack_counts
                .retain(|idx, _| *idx > meta.last_included_index);
        } else {
            self.ack_counts.clear();
        }

        let tail_index = self
            .entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.commit_index);
        self.next_index = tail_index + 1;
    }

    fn term_at(&self, index: Index) -> Option<Term> {
        if index == 0 {
            return Some(0);
        }

        if index == self.compacted_index {
            return Some(self.compacted_term);
        }

        if index < self.compacted_index {
            return None;
        }

        if let Some(entry) = self.entries.iter().find(|entry| entry.index == index) {
            return Some(entry.term);
        }

        if index == self.applied_index {
            return Some(self.applied_term);
        }

        None
    }

    fn last_log_position(&self) -> (Index, Term) {
        if let Some(entry) = self.entries.last() {
            (entry.index, entry.term)
        } else {
            (self.compacted_index, self.compacted_term)
        }
    }

    fn candidate_log_is_up_to_date(
        &self,
        candidate_last_index: Index,
        candidate_last_term: Term,
    ) -> bool {
        let (last_index, last_term) = self.last_log_position();
        candidate_last_term > last_term
            || (candidate_last_term == last_term && candidate_last_index >= last_index)
    }
}

impl LogReplicator for RaftReplicator {
    fn propose(&mut self, payload: std::sync::Arc<[u8]>) -> Result<CommitToken, EngineError> {
        if self.role != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        let idx = self.next_index;
        self.next_index += 1;

        self.entries.push(LogEntry {
            term: self.term,
            index: idx,
            payload,
        });

        // Leader has an implicit self-ack represented by voter id 0.
        let mut acks = BTreeSet::new();
        acks.insert(0);
        self.ack_counts.insert(idx, acks);
        if self.quorum == 1 {
            self.commit_index = idx;
        }

        Ok(CommitToken { index: idx })
    }

    fn wait_committed(
        &self,
        token: CommitToken,
        _timeout: std::time::Duration,
    ) -> Result<Index, EngineError> {
        if self.commit_index >= token.index {
            Ok(token.index)
        } else {
            Err(EngineError::ProposalFailed(format!(
                "token {} is not committed yet (commit_index={})",
                token.index, self.commit_index
            )))
        }
    }

    fn role(&self) -> Role {
        self.role
    }

    fn current_term(&self) -> Term {
        self.term
    }

    fn commit_index(&self) -> Index {
        self.commit_index
    }

    fn applied_index(&self) -> Index {
        self.applied_index
    }

    fn snapshot_meta(&self) -> SnapshotMeta {
        SnapshotMeta {
            last_included_index: self.applied_index,
            last_included_term: self.applied_term,
            snapshot_id: self.snapshot_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    include!("tests/rpc.rs");
    include!("tests/transport.rs");

    #[derive(Default)]
    struct AppliedLog {
        values: Vec<String>,
    }

    impl ReplicatedStateMachine for AppliedLog {
        fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError> {
            let value = std::str::from_utf8(&entry.payload)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            self.values.push(value.to_string());
            Ok(())
        }
    }

    fn apply_committed_entries(
        node: &mut RaftReplicator,
        state: &mut AppliedLog,
    ) -> Result<(), EngineError> {
        let committed: Vec<LogEntry> = node
            .drain_committed_from(node.applied_index())
            .cloned()
            .collect();
        for entry in committed {
            state.apply(&entry)?;
            node.mark_applied(entry.index);
        }
        Ok(())
    }

    fn apply_append_request(
        follower: &mut RaftReplicator,
        leader_term: Term,
        prev_log_index: Index,
        prev_log_term: Term,
        entries: Vec<LogEntry>,
        leader_commit: Index,
    ) -> AppendEntriesResponse {
        AppendEntriesRequest {
            leader_term,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit,
        }
        .apply_to(follower)
    }

    include!("tests/operational.rs");

    #[test]
    fn operational_replication_systemd_unit_contract_matches_follower_service() {
        let unit = include_str!(
            "../../../systemd/replication-follower/gpu-db-replication-follower@.service"
        );
        let follower_a =
            include_str!("../../../systemd/replication-follower/replication-follower@2.env");
        let follower_b =
            include_str!("../../../systemd/replication-follower/replication-follower@3.env");

        assert!(unit.contains("EnvironmentFile=-/etc/gpu-db/replication-follower@%i.env"));
        assert!(unit.contains(
            "ExecStart=/usr/local/bin/operational_service_smoke --follower-service --id ${GPU_DB_REPLICATION_FOLLOWER_ID} --expected-requests ${GPU_DB_REPLICATION_EXPECTED_REQUESTS} --listen ${GPU_DB_REPLICATION_LISTEN_ADDR}"
        ));
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("NoNewPrivileges=true"));
        assert!(unit.contains("ProtectSystem=strict"));

        for env_file in [follower_a, follower_b] {
            assert!(env_file.contains("GPU_DB_REPLICATION_FOLLOWER_ID="));
            assert!(env_file.contains("GPU_DB_REPLICATION_EXPECTED_REQUESTS=4"));
            assert!(env_file.contains("GPU_DB_REPLICATION_LISTEN_ADDR=0.0.0.0:55432"));
        }
    }

    #[test]
    fn operational_replication_kubernetes_manifest_contract_matches_follower_service() {
        let manifest = include_str!("../../../k8s/replication-service/follower-services.yml");

        for follower_id in ["2", "3"] {
            assert!(manifest.contains(&format!("name: gpu-db-replication-follower-{follower_id}")));
            assert!(manifest.contains(&format!(
                "gpu-db.openclaw.dev/follower-id: \"{follower_id}\""
            )));
            assert!(manifest.contains(&format!(
                "- name: GPU_DB_REPLICATION_FOLLOWER_ID\n              value: \"{follower_id}\""
            )));
        }
        assert!(manifest.contains("kind: Deployment"));
        assert!(manifest.contains("kind: Service"));
        assert!(manifest.contains("image: gpu-db-replication-service:local"));
        assert!(manifest.contains("imagePullPolicy: IfNotPresent"));
        assert!(manifest.contains("- --follower-service"));
        assert!(manifest.contains("- --id"));
        assert!(manifest.contains("- $(GPU_DB_REPLICATION_FOLLOWER_ID)"));
        assert!(manifest.contains("- --expected-requests"));
        assert!(manifest.contains("- $(GPU_DB_REPLICATION_EXPECTED_REQUESTS)"));
        assert!(manifest.contains("- --listen"));
        assert!(manifest.contains("- $(GPU_DB_REPLICATION_LISTEN_ADDR)"));
        assert!(manifest
            .contains("- name: GPU_DB_REPLICATION_EXPECTED_REQUESTS\n              value: \"4\""));
        assert!(manifest.contains(
            "- name: GPU_DB_REPLICATION_LISTEN_ADDR\n              value: 0.0.0.0:55432"
        ));
        assert!(manifest.contains("containerPort: 55432"));
        assert!(manifest.contains("targetPort: append"));
    }

    #[test]
    fn request_vote_elects_up_to_date_candidate_and_rejects_stale_log() {
        let mut leader = RaftReplicator::new(3);
        let mut up_to_date = RaftReplicator::new(3);
        let mut stale = RaftReplicator::new(3);
        let mut stale_voter = RaftReplicator::new(3);
        leader.become_leader(1);
        let first = leader.propose(vec![1].into()).unwrap();
        let entry = LogEntry {
            term: leader.current_term(),
            index: first.index,
            payload: vec![1].into(),
        };
        assert!(
            AppendEntriesRequest {
                leader_term: leader.current_term(),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![entry],
                leader_commit: leader.commit_index(),
            }
            .apply_to(&mut up_to_date)
            .accepted
        );

        let request = up_to_date.start_candidate_election(7);
        let vote = leader.request_vote_from_candidate(&request);
        assert!(vote.granted);
        assert_eq!(vote.voter_term, request.candidate_term);

        assert!(
            AppendEntriesRequest {
                leader_term: 1,
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![LogEntry {
                    term: 1,
                    index: 1,
                    payload: vec![1].into(),
                }],
                leader_commit: 0,
            }
            .apply_to(&mut stale_voter)
            .accepted
        );
        let stale_request = stale.start_candidate_election(8);
        let stale_vote = stale_voter.request_vote_from_candidate(&stale_request);
        assert!(!stale_vote.granted);
        assert_eq!(
            stale_vote.error.as_deref(),
            Some("candidate log is behind voter log at index 1 term 1")
        );
    }

    include!("tests/local.rs");

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

    include!("tests/progress.rs");

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
    fn compatible_advanced_snapshot_suffix_refresh_ignores_stale_snapshots_without_perturbing_gap()
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
    fn refreshed_compatible_suffix_repair_phase_second_refresh_still_collapses_cleanly_on_rejection(
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
    fn repair_phase_advanced_snapshot_same_frontier_refresh_updates_identity_without_perturbing_gap(
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
    fn repair_phase_advanced_snapshot_second_refresh_still_collapses_cleanly_on_newer_leader_rejection(
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
    fn repair_phase_second_refresh_advanced_replacement_refresh_survives_role_change_tail_discard()
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

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_refresh_updates_identity_without_perturbing_gap(
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

        let repair_status = r.status_snapshot();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.term, 7);
        assert_eq!(repair_status.live.commit_index, 10);
        assert_eq!(repair_status.live.applied_index, 9);
        assert_eq!(repair_status.live.next_index, 12);
        assert_eq!(repair_status.live.uncommitted_entry_count, 1);
        assert_eq!(repair_status.durable.commit_index, 10);
        assert_eq!(repair_status.durable.applied_index, 9);
        assert_eq!(repair_status.durable.next_index, 11);
        assert_eq!(repair_status.durable.uncommitted_entry_count, 0);
        assert_eq!(repair_status.live.snapshot.snapshot_id, 47);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 47);
        let repair_gap = r.recovery_progress_gap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let refreshed_repair_status = r.status_snapshot();
        assert!(refreshed_repair_status.has_speculative_tail());
        assert_eq!(refreshed_repair_status.recovery_gap, repair_gap);
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
            refreshed_repair_status.live.uncommitted_entry_count,
            repair_status.live.uncommitted_entry_count
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
        assert_eq!(
            refreshed_repair_status.durable.uncommitted_entry_count,
            repair_status.durable.uncommitted_entry_count
        );
        assert_eq!(refreshed_repair_status.live.snapshot.snapshot_id, 59);
        assert_eq!(refreshed_repair_status.durable.snapshot.snapshot_id, 59);

        let refreshed_repair_recovery = r.recovery_state();
        assert_eq!(refreshed_repair_recovery.term, 7);
        assert_eq!(refreshed_repair_recovery.snapshot.snapshot_id, 59);
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

        r.append_entries_from_leader(7, 11, 7, Vec::new(), 11)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);
        assert_eq!(committed_status.live.commit_index, 11);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 12);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 7);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 59);
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

        r.mark_applied(11);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 7);
        assert_eq!(applied_status.live.applied_index, 11);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 59);
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 7);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 59);
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
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_role_handoff_discards_only_fresh_tail(
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

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.term, 7);
        assert_eq!(before_role_change.live.commit_index, 10);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 12);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 47);
        assert_eq!(before_role_change.durable.commit_index, 10);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 11);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 47);

        r.become_candidate(8);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 8);
        assert_eq!(after_role_change.live.commit_index, 10);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 11);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 47);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 8);
        assert_eq!(after_role_change.durable.commit_index, 10);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 11);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 47);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 8);
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
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_collapses_cleanly(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.term, 7);
        assert_eq!(refreshed_status.live.commit_index, 10);
        assert_eq!(refreshed_status.live.applied_index, 9);
        assert_eq!(refreshed_status.live.next_index, 12);
        assert_eq!(refreshed_status.live.uncommitted_entry_count, 1);
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 59);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 59);

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.term, 8);
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 59);
        assert_eq!(after_rejection.durable.snapshot.snapshot_id, 59);
        assert_eq!(after_rejection.live.commit_index, 10);
        assert_eq!(after_rejection.live.applied_index, 9);
        assert_eq!(after_rejection.live.next_index, 11);
        assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.durable.commit_index, 10);
        assert_eq!(after_rejection.durable.applied_index, 9);
        assert_eq!(after_rejection.durable.next_index, 11);
        assert_eq!(after_rejection.durable.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
        assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 8);
        assert_eq!(recovery.snapshot.snapshot_id, 59);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_rejection.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_rejection.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_survives_later_repair(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.term, 8);
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 59);
        assert_eq!(after_rejection.durable.snapshot.snapshot_id, 59);
        assert_eq!(after_rejection.live.commit_index, 10);
        assert_eq!(after_rejection.live.applied_index, 9);
        assert_eq!(after_rejection.live.next_index, 11);
        assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.durable.commit_index, 10);
        assert_eq!(after_rejection.durable.applied_index, 9);
        assert_eq!(after_rejection.durable.next_index, 11);
        assert_eq!(after_rejection.durable.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
        assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();

        let after_repair = r.status_snapshot();
        assert!(after_repair.has_speculative_tail());
        assert_eq!(after_repair.live.term, 8);
        assert_eq!(after_repair.live.commit_index, 11);
        assert_eq!(after_repair.live.applied_index, 9);
        assert_eq!(after_repair.live.next_index, 13);
        assert_eq!(after_repair.live.uncommitted_entry_count, 1);
        assert_eq!(after_repair.live.snapshot.snapshot_id, 59);
        assert_eq!(after_repair.durable.commit_index, 11);
        assert_eq!(after_repair.durable.applied_index, 9);
        assert_eq!(after_repair.durable.next_index, 12);
        assert_eq!(after_repair.durable.uncommitted_entry_count, 0);
        assert_eq!(after_repair.durable.snapshot.snapshot_id, 59);
        assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);

        let repair_recovery = r.recovery_state();
        assert_eq!(repair_recovery.term, 8);
        assert_eq!(repair_recovery.snapshot.snapshot_id, 59);
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

        r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 8);
        assert_eq!(committed_status.live.commit_index, 12);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 13);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);

        r.mark_applied(12);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 8);
        assert_eq!(applied_status.live.applied_index, 12);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 59);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 8);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 59);
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
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_role_handoff_discards_only_fresh_tail(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 8);
        assert_eq!(before_role_change.live.commit_index, 11);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 13);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 59);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 8);
        assert_eq!(before_role_change.durable.commit_index, 11);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 12);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 59);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(9);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 9);
        assert_eq!(after_role_change.live.commit_index, 11);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 12);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 59);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 9);
        assert_eq!(after_role_change.durable.commit_index, 11);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 12);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 59);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 9);
        assert_eq!(recovery.snapshot.snapshot_id, 59);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_retires_cleanly_through_commit_and_apply(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();

        let before_commit = r.status_snapshot();
        assert!(before_commit.has_speculative_tail());
        assert_eq!(before_commit.live.term, 8);
        assert_eq!(before_commit.live.commit_index, 11);
        assert_eq!(before_commit.live.applied_index, 9);
        assert_eq!(before_commit.live.next_index, 13);
        assert_eq!(before_commit.live.uncommitted_entry_count, 1);
        assert_eq!(before_commit.live.snapshot.snapshot_id, 59);
        assert_eq!(before_commit.durable.term, 8);
        assert_eq!(before_commit.durable.commit_index, 11);
        assert_eq!(before_commit.durable.applied_index, 9);
        assert_eq!(before_commit.durable.next_index, 12);
        assert_eq!(before_commit.durable.uncommitted_entry_count, 0);
        assert_eq!(before_commit.durable.snapshot.snapshot_id, 59);
        assert_eq!(before_commit.recovery_gap.next_index_gap, 1);
        assert_eq!(before_commit.recovery_gap.uncommitted_entry_gap, 1);

        let repair_recovery = r.recovery_state();
        assert_eq!(repair_recovery.term, 8);
        assert_eq!(repair_recovery.snapshot.snapshot_id, 59);
        let resumed_during_rerepair =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_rerepair.status_snapshot().live,
            before_commit.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            before_commit.durable
        );

        r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 8);
        assert_eq!(committed_status.live.commit_index, 12);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 13);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
        assert_eq!(committed_status.durable.term, 8);
        assert_eq!(committed_status.durable.commit_index, 12);
        assert_eq!(committed_status.durable.applied_index, 9);
        assert_eq!(committed_status.durable.next_index, 13);
        assert_eq!(committed_status.durable.uncommitted_entry_count, 0);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);

        r.mark_applied(12);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 8);
        assert_eq!(applied_status.live.commit_index, 12);
        assert_eq!(applied_status.live.applied_index, 12);
        assert_eq!(applied_status.live.next_index, 13);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 59);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 8);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 59);
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
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_keeps_stale_installs_inert_through_commit_and_apply(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();

        let repair_status = r.status_snapshot();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.term, 8);
        assert_eq!(repair_status.live.snapshot.snapshot_id, 59);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 59);

        let repair_recovery = r.recovery_state();
        let repair_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 58,
        });
        assert_eq!(r.status_snapshot(), repair_status);
        assert_eq!(r.recovery_state(), repair_recovery);
        assert_eq!(r.recovery_progress_gap(), repair_gap);
        let resumed_during_rerepair_after_stale =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_rerepair_after_stale.status_snapshot().live,
            repair_status.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            repair_status.durable
        );

        r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);

        let committed_recovery = r.recovery_state();
        let committed_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 57,
        });
        assert_eq!(r.status_snapshot(), committed_status);
        assert_eq!(r.recovery_state(), committed_recovery);
        assert_eq!(r.recovery_progress_gap(), committed_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(12);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 59);

        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 56,
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
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_keeps_gap_shape_and_retires_on_new_identity(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();

        let before_refresh = r.status_snapshot();
        assert!(before_refresh.has_speculative_tail());
        assert_eq!(before_refresh.live.snapshot.snapshot_id, 59);
        assert_eq!(before_refresh.durable.snapshot.snapshot_id, 59);
        assert_eq!(before_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(before_refresh.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let after_refresh = r.status_snapshot();
        assert!(after_refresh.has_speculative_tail());
        assert_eq!(after_refresh.live.term, 8);
        assert_eq!(after_refresh.live.commit_index, 11);
        assert_eq!(after_refresh.live.applied_index, 9);
        assert_eq!(after_refresh.live.next_index, 13);
        assert_eq!(after_refresh.live.uncommitted_entry_count, 1);
        assert_eq!(after_refresh.live.snapshot.snapshot_id, 61);
        assert_eq!(after_refresh.durable.term, 8);
        assert_eq!(after_refresh.durable.commit_index, 11);
        assert_eq!(after_refresh.durable.applied_index, 9);
        assert_eq!(after_refresh.durable.next_index, 12);
        assert_eq!(after_refresh.durable.uncommitted_entry_count, 0);
        assert_eq!(after_refresh.durable.snapshot.snapshot_id, 61);
        assert_eq!(after_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(after_refresh.recovery_gap.uncommitted_entry_gap, 1);

        let refresh_recovery = r.recovery_state();
        assert_eq!(refresh_recovery.term, 8);
        assert_eq!(refresh_recovery.snapshot.snapshot_id, 61);
        let resumed_during_refresh_rerepair =
            RaftReplicator::resume_as_follower(3, refresh_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refresh_rerepair.status_snapshot().live,
            after_refresh.durable
        );
        assert_eq!(
            refresh_recovery.progress_as_follower().unwrap(),
            after_refresh.durable
        );

        r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 61);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 61);

        r.mark_applied(12);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 8);
        assert_eq!(applied_status.live.applied_index, 12);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 61);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 8);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 61);
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
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_role_handoff_discards_only_fresh_tail(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 8);
        assert_eq!(before_role_change.live.commit_index, 11);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 13);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 61);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 8);
        assert_eq!(before_role_change.durable.commit_index, 11);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 12);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 61);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(9);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 9);
        assert_eq!(after_role_change.live.commit_index, 11);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 12);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 61);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 9);
        assert_eq!(after_role_change.durable.commit_index, 11);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 12);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 61);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 9);
        assert_eq!(recovery.snapshot.snapshot_id, 61);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_still_collapses_cleanly_on_newer_rejection(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let before_rejection = r.status_snapshot();
        assert!(before_rejection.has_speculative_tail());
        assert_eq!(before_rejection.live.snapshot.snapshot_id, 61);
        assert_eq!(before_rejection.durable.snapshot.snapshot_id, 61);

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.role, Role::Follower);
        assert_eq!(after_rejection.live.term, 9);
        assert_eq!(after_rejection.live.commit_index, 11);
        assert_eq!(after_rejection.live.applied_index, 9);
        assert_eq!(after_rejection.live.next_index, 12);
        assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 61);
        assert!(after_rejection.live.has_committed_entries_pending_apply);
        assert_eq!(after_rejection.durable, after_rejection.live);
        assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
        assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

        let rejection_recovery = r.recovery_state();
        assert_eq!(rejection_recovery.term, 9);
        assert_eq!(rejection_recovery.snapshot.snapshot_id, 61);
        let resumed_after_rejection =
            RaftReplicator::resume_as_follower(3, rejection_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_rejection.status_snapshot().live,
            after_rejection.live
        );
        assert_eq!(
            rejection_recovery.progress_as_follower().unwrap(),
            after_rejection.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_survives_later_repair(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let collapsed_status = r.status_snapshot();
        assert!(collapsed_status.is_restart_equivalent());
        assert_eq!(collapsed_status.live.term, 9);
        assert_eq!(collapsed_status.live.commit_index, 11);
        assert_eq!(collapsed_status.live.applied_index, 9);
        assert_eq!(collapsed_status.live.next_index, 12);
        assert_eq!(collapsed_status.live.uncommitted_entry_count, 0);
        assert_eq!(collapsed_status.live.snapshot.snapshot_id, 61);
        assert_eq!(collapsed_status.durable, collapsed_status.live);

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 9);
        assert_eq!(repaired_status.live.commit_index, 12);
        assert_eq!(repaired_status.live.applied_index, 9);
        assert_eq!(repaired_status.live.next_index, 14);
        assert_eq!(repaired_status.live.uncommitted_entry_count, 1);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 61);
        assert_eq!(repaired_status.durable.term, 9);
        assert_eq!(repaired_status.durable.commit_index, 12);
        assert_eq!(repaired_status.durable.applied_index, 9);
        assert_eq!(repaired_status.durable.next_index, 13);
        assert_eq!(repaired_status.durable.uncommitted_entry_count, 0);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 61);
        assert_eq!(repaired_status.recovery_gap.next_index_gap, 1);
        assert_eq!(repaired_status.recovery_gap.uncommitted_entry_gap, 1);

        let repaired_recovery = r.recovery_state();
        assert_eq!(repaired_recovery.term, 9);
        assert_eq!(repaired_recovery.snapshot.snapshot_id, 61);
        let resumed_during_repair =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_survives_later_repair_role_handoff_discards_only_fresh_tail(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 10);
        assert_eq!(before_role_change.live.commit_index, 13);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 15);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 61);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 10);
        assert_eq!(before_role_change.durable.commit_index, 13);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 14);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 61);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(11);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 11);
        assert_eq!(after_role_change.live.commit_index, 13);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 14);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 61);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 11);
        assert_eq!(after_role_change.durable.commit_index, 13);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 14);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 61);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let role_change_recovery = r.recovery_state();
        assert_eq!(role_change_recovery.term, 11);
        assert_eq!(role_change_recovery.snapshot.snapshot_id, 61);
        let resumed_after_role_change =
            RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_role_change.status_snapshot().live,
            after_role_change.durable
        );
        assert_eq!(
            role_change_recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_survives_later_repair_and_retires_cleanly(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let collapsed_status = r.status_snapshot();
        assert!(collapsed_status.is_restart_equivalent());
        assert_eq!(collapsed_status.live.term, 10);
        assert_eq!(collapsed_status.live.commit_index, 12);
        assert_eq!(collapsed_status.live.applied_index, 9);
        assert_eq!(collapsed_status.live.next_index, 13);
        assert_eq!(collapsed_status.live.uncommitted_entry_count, 0);
        assert_eq!(collapsed_status.live.snapshot.snapshot_id, 67);
        assert_eq!(collapsed_status.durable, collapsed_status.live);

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 10);
        assert_eq!(repaired_status.live.commit_index, 13);
        assert_eq!(repaired_status.live.applied_index, 9);
        assert_eq!(repaired_status.live.next_index, 15);
        assert_eq!(repaired_status.live.uncommitted_entry_count, 1);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.durable.term, 10);
        assert_eq!(repaired_status.durable.commit_index, 13);
        assert_eq!(repaired_status.durable.applied_index, 9);
        assert_eq!(repaired_status.durable.next_index, 14);
        assert_eq!(repaired_status.durable.uncommitted_entry_count, 0);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.recovery_gap.next_index_gap, 1);
        assert_eq!(repaired_status.recovery_gap.uncommitted_entry_gap, 1);

        let repaired_recovery = r.recovery_state();
        assert_eq!(repaired_recovery.term, 10);
        assert_eq!(repaired_recovery.snapshot.snapshot_id, 67);
        let resumed_during_repair =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(!committed_status.has_speculative_tail());
        assert_eq!(committed_status.live.term, 10);
        assert_eq!(committed_status.live.commit_index, 14);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 15);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 67);
        assert_eq!(committed_status.durable, committed_status.live);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 10);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 67);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.live
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.live
        );

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.term, 10);
        assert_eq!(applied_status.live.commit_index, 14);
        assert_eq!(applied_status.live.applied_index, 14);
        assert_eq!(applied_status.live.next_index, 15);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 67);
        assert_eq!(applied_status.durable, applied_status.live);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 10);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 67);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 67);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_role_handoff_discards_only_fresh_tail(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 10);
        assert_eq!(before_role_change.live.commit_index, 13);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 15);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 67);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 10);
        assert_eq!(before_role_change.durable.commit_index, 13);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 14);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 67);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(11);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 11);
        assert_eq!(after_role_change.live.commit_index, 13);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 14);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 67);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 11);
        assert_eq!(after_role_change.durable.commit_index, 13);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 14);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 67);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let role_change_recovery = r.recovery_state();
        assert_eq!(role_change_recovery.term, 11);
        assert_eq!(role_change_recovery.snapshot.snapshot_id, 67);
        let resumed_after_role_change =
            RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_role_change.status_snapshot().live,
            after_role_change.durable
        );
        assert_eq!(
            role_change_recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 67);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_retires_cleanly_through_commit_and_apply(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 10);
        assert_eq!(repaired_status.live.commit_index, 13);
        assert_eq!(repaired_status.live.applied_index, 9);
        assert_eq!(repaired_status.live.next_index, 15);
        assert_eq!(repaired_status.live.uncommitted_entry_count, 1);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.durable.term, 10);
        assert_eq!(repaired_status.durable.commit_index, 13);
        assert_eq!(repaired_status.durable.applied_index, 9);
        assert_eq!(repaired_status.durable.next_index, 14);
        assert_eq!(repaired_status.durable.uncommitted_entry_count, 0);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.recovery_gap.next_index_gap, 1);
        assert_eq!(repaired_status.recovery_gap.uncommitted_entry_gap, 1);

        let repaired_recovery = r.recovery_state();
        assert_eq!(repaired_recovery.term, 10);
        assert_eq!(repaired_recovery.snapshot.snapshot_id, 67);
        let resumed_during_repair =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 10);
        assert_eq!(committed_status.live.commit_index, 14);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 15);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 67);
        assert_eq!(committed_status.durable, committed_status.live);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 10);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 67);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.live
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.live
        );

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.term, 10);
        assert_eq!(applied_status.live.commit_index, 14);
        assert_eq!(applied_status.live.applied_index, 14);
        assert_eq!(applied_status.live.next_index, 15);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 67);
        assert_eq!(applied_status.durable, applied_status.live);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 10);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 67);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 67);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_keeps_gap_shape_and_retires_on_new_identity(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let before_refresh = r.status_snapshot();
        assert!(before_refresh.has_speculative_tail());
        assert_eq!(before_refresh.live.snapshot.snapshot_id, 67);
        assert_eq!(before_refresh.durable.snapshot.snapshot_id, 67);
        assert_eq!(before_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(before_refresh.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let after_refresh = r.status_snapshot();
        assert!(after_refresh.has_speculative_tail());
        assert_eq!(after_refresh.live.term, 10);
        assert_eq!(after_refresh.live.commit_index, 13);
        assert_eq!(after_refresh.live.applied_index, 9);
        assert_eq!(after_refresh.live.next_index, 15);
        assert_eq!(after_refresh.live.uncommitted_entry_count, 1);
        assert_eq!(after_refresh.live.snapshot.snapshot_id, 71);
        assert_eq!(after_refresh.durable.term, 10);
        assert_eq!(after_refresh.durable.commit_index, 13);
        assert_eq!(after_refresh.durable.applied_index, 9);
        assert_eq!(after_refresh.durable.next_index, 14);
        assert_eq!(after_refresh.durable.uncommitted_entry_count, 0);
        assert_eq!(after_refresh.durable.snapshot.snapshot_id, 71);
        assert_eq!(after_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(after_refresh.recovery_gap.uncommitted_entry_gap, 1);

        let refreshed_recovery = r.recovery_state();
        assert_eq!(refreshed_recovery.term, 10);
        assert_eq!(refreshed_recovery.snapshot.snapshot_id, 71);
        let resumed_during_refreshed_repair =
            RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair.status_snapshot().live,
            after_refresh.durable
        );
        assert_eq!(
            refreshed_recovery.progress_as_follower().unwrap(),
            after_refresh.durable
        );

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 10);
        assert_eq!(committed_status.live.commit_index, 14);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 15);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 71);
        assert_eq!(committed_status.durable, committed_status.live);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 10);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 71);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.live
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.live
        );

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.term, 10);
        assert_eq!(applied_status.live.commit_index, 14);
        assert_eq!(applied_status.live.applied_index, 14);
        assert_eq!(applied_status.live.next_index, 15);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 71);
        assert_eq!(applied_status.durable, applied_status.live);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 10);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 71);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_role_handoff_discards_only_fresh_tail(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 10);
        assert_eq!(before_role_change.live.commit_index, 13);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 15);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 71);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 10);
        assert_eq!(before_role_change.durable.commit_index, 13);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 14);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 71);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(11);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 11);
        assert_eq!(after_role_change.live.commit_index, 13);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 14);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 71);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 11);
        assert_eq!(after_role_change.durable.commit_index, 13);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 14);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 71);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let role_change_recovery = r.recovery_state();
        assert_eq!(role_change_recovery.term, 11);
        assert_eq!(role_change_recovery.snapshot.snapshot_id, 71);
        let resumed_after_role_change =
            RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_role_change.status_snapshot().live,
            after_role_change.durable
        );
        assert_eq!(
            role_change_recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_still_collapses_cleanly_on_newer_rejection(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let before_rejection = r.status_snapshot();
        assert!(before_rejection.has_speculative_tail());
        assert_eq!(before_rejection.live.snapshot.snapshot_id, 71);
        assert_eq!(before_rejection.durable.snapshot.snapshot_id, 71);

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.role, Role::Follower);
        assert_eq!(after_rejection.live.term, 11);
        assert_eq!(after_rejection.live.commit_index, 13);
        assert_eq!(after_rejection.live.applied_index, 9);
        assert_eq!(after_rejection.live.next_index, 14);
        assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 71);
        assert!(after_rejection.live.has_committed_entries_pending_apply);
        assert_eq!(after_rejection.durable, after_rejection.live);
        assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
        assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

        let rejection_recovery = r.recovery_state();
        assert_eq!(rejection_recovery.term, 11);
        assert_eq!(rejection_recovery.snapshot.snapshot_id, 71);
        let resumed_after_rejection =
            RaftReplicator::resume_as_follower(3, rejection_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_rejection.status_snapshot().live,
            after_rejection.live
        );
        assert_eq!(
            rejection_recovery.progress_as_follower().unwrap(),
            after_rejection.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_keeps_stale_installs_inert_through_commit_and_apply(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let baseline = r.status_snapshot();
        assert!(baseline.has_speculative_tail());
        assert_eq!(baseline.live.snapshot.snapshot_id, 71);
        assert_eq!(baseline.durable.snapshot.snapshot_id, 71);
        assert_eq!(baseline.recovery_gap.next_index_gap, 1);
        assert_eq!(baseline.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1000,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 12,
            last_included_term: 5,
            snapshot_id: 1001,
        });

        let during_repair = r.status_snapshot();
        assert_eq!(during_repair, baseline);

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 71);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1002,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1003,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 14,
            last_included_term: 5,
            snapshot_id: 1004,
        });

        let after_commit_stale = r.status_snapshot();
        assert_eq!(after_commit_stale, committed_status);

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 71);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1005,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1006,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 14,
            last_included_term: 5,
            snapshot_id: 1007,
        });

        let after_apply_stale = r.status_snapshot();
        assert_eq!(after_apply_stale, applied_status);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 10);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 71);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 71);
        assert_eq!(after_rejection.live.term, 11);
        assert_eq!(after_rejection.live.commit_index, 13);
        assert_eq!(after_rejection.live.next_index, 14);

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();

        let after_repair = r.status_snapshot();
        assert!(!after_repair.is_restart_equivalent());
        assert!(after_repair.has_speculative_tail());
        assert_eq!(after_repair.live.role, Role::Follower);
        assert_eq!(after_repair.live.term, 11);
        assert_eq!(after_repair.live.commit_index, 14);
        assert_eq!(after_repair.live.applied_index, 9);
        assert_eq!(after_repair.live.next_index, 16);
        assert_eq!(after_repair.live.uncommitted_entry_count, 1);
        assert_eq!(after_repair.live.snapshot.snapshot_id, 71);
        assert_eq!(after_repair.durable.snapshot.snapshot_id, 71);
        assert_eq!(after_repair.durable.commit_index, 14);
        assert_eq!(after_repair.durable.next_index, 15);
        assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);
        assert!(after_repair.live.has_committed_entries_pending_apply);

        let repair_recovery = r.recovery_state();
        assert_eq!(repair_recovery.term, 11);
        assert_eq!(repair_recovery.snapshot.snapshot_id, 71);
        assert_eq!(repair_recovery.commit_index(), 14);
        assert_eq!(repair_recovery.next_index(), 15);
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
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_role_handoff_discards_only_fresh_tail(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 11);
        assert_eq!(before_role_change.live.commit_index, 14);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 16);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 71);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 11);
        assert_eq!(before_role_change.durable.commit_index, 14);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 15);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 71);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(12);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 12);
        assert_eq!(after_role_change.live.commit_index, 14);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 15);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 71);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 12);
        assert_eq!(after_role_change.durable.commit_index, 14);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 15);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 71);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let role_change_recovery = r.recovery_state();
        assert_eq!(role_change_recovery.term, 12);
        assert_eq!(role_change_recovery.snapshot.snapshot_id, 71);
        let resumed_after_role_change =
            RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_role_change.status_snapshot().live,
            after_role_change.durable
        );
        assert_eq!(
            role_change_recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_and_retires_cleanly(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();

        let during_repair = r.status_snapshot();
        assert!(during_repair.has_speculative_tail());
        assert_eq!(during_repair.live.snapshot.snapshot_id, 71);
        assert_eq!(during_repair.durable.snapshot.snapshot_id, 71);
        assert_eq!(during_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(during_repair.recovery_gap.uncommitted_entry_gap, 1);

        r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
            .unwrap();

        let after_commit = r.status_snapshot();
        assert!(after_commit.is_restart_equivalent());
        assert_eq!(after_commit.live.role, Role::Follower);
        assert_eq!(after_commit.live.term, 11);
        assert_eq!(after_commit.live.commit_index, 15);
        assert_eq!(after_commit.live.applied_index, 9);
        assert_eq!(after_commit.live.next_index, 16);
        assert_eq!(after_commit.live.uncommitted_entry_count, 0);
        assert_eq!(after_commit.live.snapshot.snapshot_id, 71);
        assert!(after_commit.live.has_committed_entries_pending_apply);
        assert_eq!(after_commit.durable, after_commit.live);
        assert_eq!(after_commit.recovery_gap.next_index_gap, 0);
        assert_eq!(after_commit.recovery_gap.uncommitted_entry_gap, 0);

        r.mark_applied(15);

        let after_apply = r.status_snapshot();
        assert!(after_apply.is_restart_equivalent());
        assert_eq!(after_apply.live.role, Role::Follower);
        assert_eq!(after_apply.live.term, 11);
        assert_eq!(after_apply.live.commit_index, 15);
        assert_eq!(after_apply.live.applied_index, 15);
        assert_eq!(after_apply.live.next_index, 16);
        assert_eq!(after_apply.live.uncommitted_entry_count, 0);
        assert_eq!(after_apply.live.snapshot.snapshot_id, 71);
        assert!(!after_apply.live.has_committed_entries_pending_apply);
        assert_eq!(after_apply.durable, after_apply.live);
        assert_eq!(after_apply.recovery_gap.next_index_gap, 0);
        assert_eq!(after_apply.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 11);
        assert_eq!(recovery.snapshot.snapshot_id, 71);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_apply.live);
        assert_eq!(recovery.progress_as_follower().unwrap(), after_apply.live);
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_keeps_stale_installs_inert_through_commit_and_apply(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();

        let baseline = r.status_snapshot();
        assert!(baseline.has_speculative_tail());
        assert_eq!(baseline.live.snapshot.snapshot_id, 71);
        assert_eq!(baseline.durable.snapshot.snapshot_id, 71);
        assert_eq!(baseline.recovery_gap.next_index_gap, 1);
        assert_eq!(baseline.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1000,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1001,
        });

        let during_repair = r.status_snapshot();
        assert_eq!(during_repair, baseline);

        r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 71);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1002,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1003,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1004,
        });

        let after_commit_stale = r.status_snapshot();
        assert_eq!(after_commit_stale, committed_status);

        r.mark_applied(15);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 71);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1005,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1006,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1007,
        });

        let after_apply_stale = r.status_snapshot();
        assert_eq!(after_apply_stale, applied_status);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 11);
        assert_eq!(recovery.snapshot.snapshot_id, 71);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, applied_status.live);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_keeps_gap_shape_and_retires_on_new_identity(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();

        let before_refresh = r.status_snapshot();
        assert!(before_refresh.has_speculative_tail());
        assert_eq!(before_refresh.live.snapshot.snapshot_id, 71);
        assert_eq!(before_refresh.durable.snapshot.snapshot_id, 71);
        assert_eq!(before_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(before_refresh.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 73,
        });

        let after_refresh = r.status_snapshot();
        assert!(after_refresh.has_speculative_tail());
        assert_eq!(after_refresh.live.term, 11);
        assert_eq!(after_refresh.live.commit_index, 14);
        assert_eq!(after_refresh.live.applied_index, 9);
        assert_eq!(after_refresh.live.next_index, 16);
        assert_eq!(after_refresh.live.uncommitted_entry_count, 1);
        assert_eq!(after_refresh.live.snapshot.snapshot_id, 73);
        assert_eq!(after_refresh.durable.term, 11);
        assert_eq!(after_refresh.durable.commit_index, 14);
        assert_eq!(after_refresh.durable.applied_index, 9);
        assert_eq!(after_refresh.durable.next_index, 15);
        assert_eq!(after_refresh.durable.uncommitted_entry_count, 0);
        assert_eq!(after_refresh.durable.snapshot.snapshot_id, 73);
        assert_eq!(after_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(after_refresh.recovery_gap.uncommitted_entry_gap, 1);

        let refreshed_recovery = r.recovery_state();
        assert_eq!(refreshed_recovery.term, 11);
        assert_eq!(refreshed_recovery.snapshot.snapshot_id, 73);
        let resumed_during_refreshed_repair =
            RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair.status_snapshot().live,
            after_refresh.durable
        );
        assert_eq!(
            refreshed_recovery.progress_as_follower().unwrap(),
            after_refresh.durable
        );

        r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 11);
        assert_eq!(committed_status.live.commit_index, 15);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 16);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 73);
        assert_eq!(committed_status.durable, committed_status.live);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 11);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 73);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.live
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.live
        );

        r.mark_applied(15);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.term, 11);
        assert_eq!(applied_status.live.commit_index, 15);
        assert_eq!(applied_status.live.applied_index, 15);
        assert_eq!(applied_status.live.next_index, 16);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 73);
        assert_eq!(applied_status.durable, applied_status.live);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 11);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 73);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 73);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_role_handoff_discards_only_fresh_tail(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 73,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 11);
        assert_eq!(before_role_change.live.commit_index, 14);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 16);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 73);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 11);
        assert_eq!(before_role_change.durable.commit_index, 14);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 15);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 73);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(12);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 12);
        assert_eq!(after_role_change.live.commit_index, 14);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 15);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 73);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 12);
        assert_eq!(after_role_change.durable.commit_index, 14);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 15);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 73);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let role_change_recovery = r.recovery_state();
        assert_eq!(role_change_recovery.term, 12);
        assert_eq!(role_change_recovery.snapshot.snapshot_id, 73);
        let resumed_after_role_change =
            RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_role_change.status_snapshot().live,
            after_role_change.durable
        );
        assert_eq!(
            role_change_recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 73);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_and_retires_cleanly(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 73,
        });

        let during_repair = r.status_snapshot();
        assert!(during_repair.has_speculative_tail());
        assert_eq!(during_repair.live.snapshot.snapshot_id, 73);
        assert_eq!(during_repair.durable.snapshot.snapshot_id, 73);
        assert_eq!(during_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(during_repair.recovery_gap.uncommitted_entry_gap, 1);

        r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
            .unwrap();

        let after_commit = r.status_snapshot();
        assert!(after_commit.is_restart_equivalent());
        assert_eq!(after_commit.live.role, Role::Follower);
        assert_eq!(after_commit.live.term, 11);
        assert_eq!(after_commit.live.commit_index, 15);
        assert_eq!(after_commit.live.applied_index, 9);
        assert_eq!(after_commit.live.next_index, 16);
        assert_eq!(after_commit.live.uncommitted_entry_count, 0);
        assert_eq!(after_commit.live.snapshot.snapshot_id, 73);
        assert!(after_commit.live.has_committed_entries_pending_apply);
        assert_eq!(after_commit.durable, after_commit.live);
        assert_eq!(after_commit.recovery_gap.next_index_gap, 0);
        assert_eq!(after_commit.recovery_gap.uncommitted_entry_gap, 0);

        r.mark_applied(15);

        let after_apply = r.status_snapshot();
        assert!(after_apply.is_restart_equivalent());
        assert_eq!(after_apply.live.role, Role::Follower);
        assert_eq!(after_apply.live.term, 11);
        assert_eq!(after_apply.live.commit_index, 15);
        assert_eq!(after_apply.live.applied_index, 15);
        assert_eq!(after_apply.live.next_index, 16);
        assert_eq!(after_apply.live.uncommitted_entry_count, 0);
        assert_eq!(after_apply.live.snapshot.snapshot_id, 73);
        assert!(!after_apply.live.has_committed_entries_pending_apply);
        assert_eq!(after_apply.durable, after_apply.live);
        assert_eq!(after_apply.recovery_gap.next_index_gap, 0);
        assert_eq!(after_apply.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 11);
        assert_eq!(recovery.snapshot.snapshot_id, 73);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_apply.live);
        assert_eq!(recovery.progress_as_follower().unwrap(), after_apply.live);
        assert_eq!(r.snapshot_meta().snapshot_id, 73);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_keeps_stale_installs_inert_through_commit_and_apply(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 73,
        });

        let baseline = r.status_snapshot();
        assert!(baseline.has_speculative_tail());
        assert_eq!(baseline.live.snapshot.snapshot_id, 73);
        assert_eq!(baseline.durable.snapshot.snapshot_id, 73);
        assert_eq!(baseline.recovery_gap.next_index_gap, 1);
        assert_eq!(baseline.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1000,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1001,
        });

        let during_repair = r.status_snapshot();
        assert_eq!(during_repair, baseline);

        r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 73);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1002,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1003,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1004,
        });

        let after_commit_stale = r.status_snapshot();
        assert_eq!(after_commit_stale, committed_status);

        r.mark_applied(15);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 73);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1005,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1006,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1007,
        });

        let after_apply_stale = r.status_snapshot();
        assert_eq!(after_apply_stale, applied_status);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 11);
        assert_eq!(recovery.snapshot.snapshot_id, 73);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, applied_status.live);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 73);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_keeps_stale_installs_inert_through_commit_and_apply(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        let repaired_recovery = r.recovery_state();
        let repaired_gap = r.recovery_progress_gap();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 10);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 991,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 990,
        });
        assert_eq!(r.status_snapshot(), repaired_status);
        assert_eq!(r.recovery_state(), repaired_recovery);
        assert_eq!(r.recovery_progress_gap(), repaired_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        let committed_recovery = r.recovery_state();
        let committed_gap = r.recovery_progress_gap();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 10);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 67);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 989,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 988,
        });
        assert_eq!(r.status_snapshot(), committed_status);
        assert_eq!(r.recovery_state(), committed_recovery);
        assert_eq!(r.recovery_progress_gap(), committed_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 10);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 987,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 986,
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
        assert_eq!(r.snapshot_meta().snapshot_id, 67);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_survives_later_repair_keeps_stale_installs_inert_through_commit_and_apply(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        let repaired_recovery = r.recovery_state();
        let repaired_gap = r.recovery_progress_gap();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 10);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 991,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 990,
        });
        assert_eq!(r.status_snapshot(), repaired_status);
        assert_eq!(r.recovery_state(), repaired_recovery);
        assert_eq!(r.recovery_progress_gap(), repaired_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        let committed_recovery = r.recovery_state();
        let committed_gap = r.recovery_progress_gap();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 10);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 67);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 989,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 988,
        });
        assert_eq!(r.status_snapshot(), committed_status);
        assert_eq!(r.recovery_state(), committed_recovery);
        assert_eq!(r.recovery_progress_gap(), committed_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 10);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 987,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 986,
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
        assert_eq!(r.snapshot_meta().snapshot_id, 67);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_survives_later_repair_and_retires_cleanly(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 9);
        assert_eq!(repaired_status.live.commit_index, 12);
        assert_eq!(repaired_status.live.applied_index, 9);
        assert_eq!(repaired_status.live.next_index, 14);
        assert_eq!(repaired_status.live.uncommitted_entry_count, 1);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 61);
        assert_eq!(repaired_status.durable.term, 9);
        assert_eq!(repaired_status.durable.commit_index, 12);
        assert_eq!(repaired_status.durable.applied_index, 9);
        assert_eq!(repaired_status.durable.next_index, 13);
        assert_eq!(repaired_status.durable.uncommitted_entry_count, 0);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 61);
        assert_eq!(repaired_status.recovery_gap.next_index_gap, 1);
        assert_eq!(repaired_status.recovery_gap.uncommitted_entry_gap, 1);

        r.append_entries_from_leader(9, 13, 9, Vec::new(), 13)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(!committed_status.has_speculative_tail());
        assert_eq!(committed_status.live.term, 9);
        assert_eq!(committed_status.live.commit_index, 13);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 14);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 61);
        assert_eq!(committed_status.durable, committed_status.live);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 9);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 61);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.live
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.live
        );

        r.mark_applied(13);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.term, 9);
        assert_eq!(applied_status.live.commit_index, 13);
        assert_eq!(applied_status.live.applied_index, 13);
        assert_eq!(applied_status.live.next_index, 14);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 61);
        assert_eq!(applied_status.durable, applied_status.live);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 9);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 61);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_survives_later_repair_keeps_stale_installs_inert_through_commit_and_apply(
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
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        let repaired_recovery = r.recovery_state();
        let repaired_gap = r.recovery_progress_gap();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 9);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 61);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 61);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 998,
        });
        assert_eq!(r.status_snapshot(), repaired_status);
        assert_eq!(r.recovery_state(), repaired_recovery);
        assert_eq!(r.recovery_progress_gap(), repaired_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );

        r.append_entries_from_leader(9, 13, 9, Vec::new(), 13)
            .unwrap();

        let committed_status = r.status_snapshot();
        let committed_recovery = r.recovery_state();
        let committed_gap = r.recovery_progress_gap();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 9);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 61);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 61);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 996,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 995,
        });
        assert_eq!(r.status_snapshot(), committed_status);
        assert_eq!(r.recovery_state(), committed_recovery);
        assert_eq!(r.recovery_progress_gap(), committed_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(13);

        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 9);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 61);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 993,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 992,
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
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_keeps_stale_installs_inert_through_commit_and_apply(
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

        let repair_status = r.status_snapshot();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.term, 7);
        assert_eq!(repair_status.live.snapshot.snapshot_id, 47);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 47);

        let repair_recovery = r.recovery_state();
        let repair_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
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

        r.append_entries_from_leader(7, 11, 7, Vec::new(), 11)
            .unwrap();

        let commit_status = r.status_snapshot();
        assert!(commit_status.is_restart_equivalent());
        assert!(commit_status.live.has_committed_entries_pending_apply);
        assert_eq!(commit_status.live.snapshot.snapshot_id, 47);
        assert_eq!(commit_status.durable.snapshot.snapshot_id, 47);

        let commit_recovery = r.recovery_state();
        let commit_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 46,
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

        r.mark_applied(11);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 7);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 47);

        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 45,
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
        assert_eq!(r.snapshot_meta().snapshot_id, 47);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_keeps_stale_installs_inert_through_commit_and_apply(
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

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        let after_stale_repair = r.status_snapshot();
        assert_eq!(after_stale_repair, refreshed_status);

        r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
            .unwrap();
        let committed_status = r.status_snapshot();
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 47);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 46,
        });
        let after_stale_commit = r.status_snapshot();
        assert_eq!(after_stale_commit, committed_status);

        r.mark_applied(10);
        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 47);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 47);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 45,
        });
        let after_stale_apply = r.status_snapshot();
        assert_eq!(after_stale_apply, applied_status);

        let recovery = r.recovery_state();
        assert_eq!(recovery.snapshot.snapshot_id, 47);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, applied_status.live);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 47);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_collapses_cleanly_on_newer_leader_rejection(
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
    fn repair_phase_advanced_snapshot_refresh_keeps_stale_installs_inert_through_commit_and_apply()
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
}
