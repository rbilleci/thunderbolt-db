use std::time::Duration;

use gpu_db_replication::{
    LogReplicator, OperationalClusterSmokeReport, OperationalDeploymentPreflightReport,
    OperationalTransportSmokeReport, RaftReplicator, ReplicatedStateMachine,
};
use gpu_db_types::{EngineError, Index, LogEntry, Role, Term};

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

fn apply_committed(node: &mut RaftReplicator, state: &mut AppliedLog) -> Result<(), EngineError> {
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

fn append_entries(
    follower: &mut RaftReplicator,
    leader_term: Term,
    prev_log_index: Index,
    prev_log_term: Term,
    entries: Vec<LogEntry>,
    leader_commit: Index,
) -> Result<(), EngineError> {
    follower.append_entries_from_leader(
        leader_term,
        prev_log_index,
        prev_log_term,
        entries,
        leader_commit,
    )
}

fn main() -> Result<(), EngineError> {
    let mut leader = RaftReplicator::new(3);
    let mut follower_a = RaftReplicator::new(3);
    let mut follower_b = RaftReplicator::new(3);
    let mut state_a = AppliedLog::default();
    let mut state_b = AppliedLog::default();
    let mut append_batches_sent = 0;
    let mut heartbeat_batches_sent = 0;
    let mut follower_acks_recorded = 0;

    leader.become_leader(1);
    assert_eq!(leader.role(), Role::Leader);
    assert!(matches!(
        follower_a.propose(b"blocked follower write".to_vec()),
        Err(EngineError::NotLeader)
    ));

    let first = leader.propose(b"create table t(id int)".to_vec())?;
    let second = leader.propose(b"insert into t values (1)".to_vec())?;
    let term_one = leader.current_term();
    let first_batch = vec![
        LogEntry {
            term: term_one,
            index: first.index,
            payload: b"create table t(id int)".to_vec(),
        },
        LogEntry {
            term: term_one,
            index: second.index,
            payload: b"insert into t values (1)".to_vec(),
        },
    ];

    append_batches_sent += 1;
    append_entries(&mut follower_a, term_one, 0, 0, first_batch.clone(), 0)?;
    leader.register_follower_ack(first.index, 1);
    follower_acks_recorded += 1;
    leader.register_follower_ack(second.index, 1);
    follower_acks_recorded += 1;
    leader.wait_committed(second, Duration::from_millis(1))?;

    heartbeat_batches_sent += 1;
    append_entries(
        &mut follower_a,
        term_one,
        second.index,
        term_one,
        vec![],
        leader.commit_index(),
    )?;
    apply_committed(&mut follower_a, &mut state_a)?;

    append_batches_sent += 1;
    append_entries(
        &mut follower_b,
        term_one,
        0,
        0,
        first_batch,
        leader.commit_index(),
    )?;
    apply_committed(&mut follower_b, &mut state_b)?;

    assert_eq!(state_a.values, state_b.values);
    assert_eq!(
        state_b.values.last().map(String::as_str),
        Some("insert into t values (1)")
    );
    assert!(follower_b.progress().is_caught_up());
    assert_eq!(follower_b.status_snapshot().live.role, Role::Follower);

    leader.become_follower(2);
    follower_a.become_leader(2);
    let old_leader_rejected_after_failover = matches!(
        leader.propose(b"blocked after failover".to_vec()),
        Err(EngineError::NotLeader)
    );
    assert!(old_leader_rejected_after_failover);

    let third = follower_a.propose(b"insert into t values (2)".to_vec())?;
    let term_two = follower_a.current_term();
    let failover_entry = vec![LogEntry {
        term: term_two,
        index: third.index,
        payload: b"insert into t values (2)".to_vec(),
    }];

    append_batches_sent += 1;
    append_entries(
        &mut follower_b,
        term_two,
        second.index,
        term_one,
        failover_entry,
        follower_a.commit_index(),
    )?;
    follower_a.register_follower_ack(third.index, 2);
    follower_acks_recorded += 1;
    follower_a.wait_committed(third, Duration::from_millis(1))?;
    heartbeat_batches_sent += 1;
    append_entries(
        &mut follower_b,
        term_two,
        third.index,
        term_two,
        vec![],
        follower_a.commit_index(),
    )?;
    apply_committed(&mut follower_b, &mut state_b)?;

    assert_eq!(
        state_b.values.last().map(String::as_str),
        Some("insert into t values (2)")
    );
    assert!(follower_b.progress().is_caught_up());
    assert_eq!(follower_a.status_snapshot().live.role, Role::Leader);

    let smoke = OperationalClusterSmokeReport {
        promoted_leader_term: follower_a.current_term(),
        promoted_leader_commit_index: follower_a.commit_index(),
        follower_commit_index: follower_b.commit_index(),
        follower_applied_index: follower_b.applied_index(),
        follower_caught_up: follower_b.progress().is_caught_up(),
        follower_read_after_apply: state_b.values,
        old_leader_rejected_after_failover,
        promoted_node_role: follower_a.role(),
    };
    let report = OperationalDeploymentPreflightReport {
        smoke,
        transport: OperationalTransportSmokeReport {
            transport_scope: "in_memory_append_entries",
            append_batches_sent,
            heartbeat_batches_sent,
            follower_acks_recorded,
        },
        network_transport_implemented: false,
        automatic_election_implemented: false,
        packaged_deployment_implemented: false,
    };
    assert!(report.readiness_passed());
    for line in report.to_operator_lines() {
        println!("{line}");
    }

    Ok(())
}
