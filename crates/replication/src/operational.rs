use gpu_db_types::{Index, Role, Term};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalClusterSmokeReport {
    pub promoted_leader_term: Term,
    pub promoted_leader_commit_index: Index,
    pub follower_commit_index: Index,
    pub follower_applied_index: Index,
    pub follower_caught_up: bool,
    pub follower_read_after_apply: Vec<String>,
    pub old_leader_rejected_after_failover: bool,
    pub promoted_node_role: Role,
}

impl OperationalClusterSmokeReport {
    pub fn readiness_passed(&self) -> bool {
        self.follower_caught_up
            && self.old_leader_rejected_after_failover
            && self.promoted_node_role == Role::Leader
            && self.follower_commit_index == self.promoted_leader_commit_index
            && self.follower_applied_index == self.follower_commit_index
    }

    pub fn to_operator_lines(&self) -> Vec<String> {
        vec![
            format!(
                "operational_replication_smoke={}",
                if self.readiness_passed() {
                    "passed"
                } else {
                    "failed"
                }
            ),
            format!(
                "promoted_leader_term={} promoted_leader_commit={} follower_commit={} follower_applied={} follower_caught_up={}",
                self.promoted_leader_term,
                self.promoted_leader_commit_index,
                self.follower_commit_index,
                self.follower_applied_index,
                self.follower_caught_up
            ),
            format!(
                "follower_read_after_apply={}",
                self.follower_read_after_apply.join(" | ")
            ),
            format!(
                "failover_admission_gate=old_leader_{} promoted_node_role={:?}",
                if self.old_leader_rejected_after_failover {
                    "not_leader"
                } else {
                    "accepted_write"
                },
                self.promoted_node_role
            ),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalTransportSmokeReport {
    pub transport_scope: &'static str,
    pub append_batches_sent: usize,
    pub heartbeat_batches_sent: usize,
    pub follower_acks_recorded: usize,
}

impl OperationalTransportSmokeReport {
    pub fn readiness_passed(&self) -> bool {
        !self.transport_scope.is_empty()
            && self.append_batches_sent > 0
            && self.heartbeat_batches_sent > 0
            && self.follower_acks_recorded > 0
    }

    pub fn to_operator_line(&self) -> String {
        format!(
            "deployment_transport={} append_batches_sent={} heartbeat_batches_sent={} follower_acks_recorded={}",
            self.transport_scope,
            self.append_batches_sent,
            self.heartbeat_batches_sent,
            self.follower_acks_recorded
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalElectionSmokeReport {
    pub election_scope: &'static str,
    pub candidate_id: u64,
    pub elected_term: Term,
    pub votes_granted: usize,
    pub quorum: usize,
    pub elected: bool,
}

impl OperationalElectionSmokeReport {
    pub fn readiness_passed(&self) -> bool {
        !self.election_scope.is_empty()
            && self.candidate_id > 0
            && self.elected
            && self.votes_granted >= self.quorum
    }

    pub fn to_operator_line(&self) -> String {
        format!(
            "deployment_election={} candidate_id={} elected_term={} votes_granted={} quorum={} elected={}",
            self.election_scope,
            self.candidate_id,
            self.elected_term,
            self.votes_granted,
            self.quorum,
            self.elected
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalPackageSmokeReport {
    pub package_scope: &'static str,
    pub entrypoint: &'static str,
    pub smoke_script: &'static str,
    pub packaged_script: &'static str,
    pub reproducible: bool,
}

impl OperationalPackageSmokeReport {
    pub fn readiness_passed(&self) -> bool {
        !self.package_scope.is_empty()
            && !self.entrypoint.is_empty()
            && !self.smoke_script.is_empty()
            && !self.packaged_script.is_empty()
            && self.reproducible
    }

    pub fn to_operator_line(&self) -> String {
        format!(
            "deployment_package={} entrypoint={} smoke_script={} packaged_script={} reproducible={}",
            self.package_scope,
            self.entrypoint,
            self.smoke_script,
            self.packaged_script,
            self.reproducible
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalDeploymentPreflightReport {
    pub smoke: OperationalClusterSmokeReport,
    pub transport: OperationalTransportSmokeReport,
    pub election: OperationalElectionSmokeReport,
    pub package: OperationalPackageSmokeReport,
    pub network_transport_implemented: bool,
    pub automatic_election_implemented: bool,
    pub packaged_deployment_implemented: bool,
}

impl OperationalDeploymentPreflightReport {
    pub fn readiness_passed(&self) -> bool {
        self.smoke.readiness_passed()
            && self.transport.readiness_passed()
            && self.election.readiness_passed()
            && self.package.readiness_passed()
    }

    pub fn to_operator_lines(&self) -> Vec<String> {
        let mut lines = self.smoke.to_operator_lines();
        lines.push(self.transport.to_operator_line());
        lines.push(self.election.to_operator_line());
        lines.push(self.package.to_operator_line());
        lines.extend([
            format!(
                "operational_deployment_preflight={}",
                if self.readiness_passed() {
                    "passed"
                } else {
                    "failed"
                }
            ),
            "deployment_scope=packaged_local_three_node_raft_smoke".to_string(),
            format!(
                "deployment_gap_network_transport={}",
                if self.network_transport_implemented {
                    "implemented"
                } else {
                    "missing"
                }
            ),
            format!(
                "deployment_gap_automatic_election={}",
                if self.automatic_election_implemented {
                    "implemented"
                } else {
                    "missing"
                }
            ),
            format!(
                "deployment_gap_packaged_deployment={}",
                if self.packaged_deployment_implemented {
                    "implemented"
                } else {
                    "missing"
                }
            ),
        ]);
        lines
    }
}
