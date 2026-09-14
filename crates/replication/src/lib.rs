use std::collections::{BTreeMap, BTreeSet};

use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term};

mod local;
mod operational;
mod progress;
mod rpc;
mod transport;

pub use local::{LocalReplicator, LocalReplicatorProposalReservation};
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
        if self.next_index == u64::MAX {
            return Err(EngineError::ProposalFailed(
                "commit sequence space exhausted".to_string(),
            ));
        }

        let idx = self.next_index;
        self.next_index = self
            .next_index
            .checked_add(1)
            .expect("reserved maximum commit sequence was refused");

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

    #[test]
    fn raft_refuses_reserved_maximum_commit_sequence_without_mutation() {
        let mut repl = RaftReplicator::single_node_leader();
        repl.install_snapshot(SnapshotMeta {
            last_included_index: u64::MAX - 1,
            last_included_term: 1,
            snapshot_id: 1,
        });
        let before = repl.progress();
        let error = repl.propose(std::sync::Arc::from(&b"x"[..])).unwrap_err();
        assert!(error.to_string().contains("sequence space exhausted"));
        assert_eq!(repl.progress(), before);
    }

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
            assert!(manifest.contains(&format!("gpu-db-follower-id: \"{follower_id}\"")));
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

    include!("tests/raft_baseline.rs");

    include!("tests/progress.rs");

    include!("tests/replicator_lifecycle.rs");

    include!("tests/append_entries.rs");

    include!("tests/snapshot_identity.rs");

    include!("tests/repair_advanced_snapshot.rs");

    include!("tests/repair_second_refresh.rs");

    include!("tests/repair_post_rejection.rs");

    include!("tests/repair_refresh_rejection.rs");

    include!("tests/repair_deep_refresh.rs");

    include!("tests/repair_stale_unwind.rs");

    include!("tests/repair_advanced_cleanup.rs");
}
