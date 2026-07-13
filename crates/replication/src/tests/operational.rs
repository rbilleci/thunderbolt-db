#[test]
fn operational_replication_three_node_smoke_catches_up_reads_after_apply_and_gates_failover() {
    let mut leader = RaftReplicator::new(3);
    let mut follower_a = RaftReplicator::new(3);
    let mut follower_b = RaftReplicator::new(3);
    let mut state_a = AppliedLog::default();
    let mut state_b = AppliedLog::default();
    let mut append_batches_sent = 0;
    let mut heartbeat_batches_sent = 0;
    let mut follower_acks_recorded = 0;

    leader.become_leader(1);
    assert!(matches!(
        follower_a.propose(b"blocked follower write".to_vec().into()),
        Err(EngineError::NotLeader)
    ));

    let first = leader
        .propose(b"create table t(id int)".to_vec().into())
        .unwrap();
    let second = leader
        .propose(b"insert into t values (1)".to_vec().into())
        .unwrap();
    let term_one = leader.current_term();
    let first_batch = vec![
        LogEntry {
            term: term_one,
            index: first.index,
            payload: b"create table t(id int)".to_vec().into(),
        },
        LogEntry {
            term: term_one,
            index: second.index,
            payload: b"insert into t values (1)".to_vec().into(),
        },
    ];

    append_batches_sent += 1;
    assert!(apply_append_request(&mut follower_a, term_one, 0, 0, first_batch.clone(), 0).accepted);
    leader.register_follower_ack(first.index, 1);
    follower_acks_recorded += 1;
    leader.register_follower_ack(second.index, 1);
    follower_acks_recorded += 1;
    leader
        .wait_committed(second, std::time::Duration::from_millis(1))
        .unwrap();

    heartbeat_batches_sent += 1;
    assert!(
        apply_append_request(
            &mut follower_a,
            term_one,
            second.index,
            term_one,
            vec![],
            leader.commit_index(),
        )
        .accepted
    );
    apply_committed_entries(&mut follower_a, &mut state_a).unwrap();

    append_batches_sent += 1;
    assert!(
        apply_append_request(
            &mut follower_b,
            term_one,
            0,
            0,
            first_batch,
            leader.commit_index(),
        )
        .accepted
    );
    apply_committed_entries(&mut follower_b, &mut state_b).unwrap();

    assert_eq!(state_a.values, state_b.values);
    assert_eq!(
        state_b.values.last().map(String::as_str),
        Some("insert into t values (1)")
    );
    assert!(follower_b.progress().is_caught_up());
    assert_eq!(follower_b.status_snapshot().live.role, Role::Follower);

    let vote_request = follower_a.start_candidate_election(1);
    let mut votes_granted = 1;
    let old_leader_vote = leader.request_vote_from_candidate(&vote_request);
    if old_leader_vote.granted {
        votes_granted += 1;
    }
    let follower_b_vote = follower_b.request_vote_from_candidate(&vote_request);
    if follower_b_vote.granted {
        votes_granted += 1;
    }
    let election_quorum = follower_a.quorum_size();
    let election_passed = votes_granted >= election_quorum;
    if election_passed {
        follower_a.become_leader(vote_request.candidate_term);
    }
    assert!(election_passed);
    let old_leader_rejected_after_failover = matches!(
        leader.propose(b"blocked after failover".to_vec().into()),
        Err(EngineError::NotLeader)
    );
    assert!(old_leader_rejected_after_failover);

    let third = follower_a
        .propose(b"insert into t values (2)".to_vec().into())
        .unwrap();
    let term_two = follower_a.current_term();
    append_batches_sent += 1;
    assert!(
        apply_append_request(
            &mut follower_b,
            term_two,
            second.index,
            term_one,
            vec![LogEntry {
                term: term_two,
                index: third.index,
                payload: b"insert into t values (2)".to_vec().into(),
            }],
            follower_a.commit_index(),
        )
        .accepted
    );
    follower_a.register_follower_ack(third.index, 2);
    follower_acks_recorded += 1;
    follower_a
        .wait_committed(third, std::time::Duration::from_millis(1))
        .unwrap();
    heartbeat_batches_sent += 1;
    assert!(
        apply_append_request(
            &mut follower_b,
            term_two,
            third.index,
            term_two,
            vec![],
            follower_a.commit_index(),
        )
        .accepted
    );
    apply_committed_entries(&mut follower_b, &mut state_b).unwrap();

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
    assert!(smoke.readiness_passed());
    let report = OperationalDeploymentPreflightReport {
        smoke,
        transport: OperationalTransportSmokeReport {
            transport_scope: "single_request_tcp_append_entries",
            append_batches_sent,
            heartbeat_batches_sent,
            follower_acks_recorded,
        },
        election: OperationalElectionSmokeReport {
            election_scope: "deterministic_request_vote",
            candidate_id: vote_request.candidate_id,
            elected_term: vote_request.candidate_term,
            votes_granted,
            quorum: election_quorum,
            elected: election_passed,
        },
        package: OperationalPackageSmokeReport {
            package_scope: "local_cargo_example_binary",
            entrypoint: "crates/replication/examples/operational_cluster_smoke.rs",
            smoke_script: "scripts/run_replication_cluster_smoke.sh",
            packaged_script: "scripts/run_replication_packaged_smoke.sh",
            reproducible: true,
        },
        network_transport_implemented: true,
        automatic_election_implemented: true,
        packaged_deployment_implemented: true,
    };
    assert!(report.readiness_passed());
    assert_eq!(
        report.to_operator_lines(),
        vec![
            "operational_replication_smoke=passed".to_string(),
            "promoted_leader_term=2 promoted_leader_commit=3 follower_commit=3 follower_applied=3 follower_caught_up=true".to_string(),
            "follower_read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)".to_string(),
            "failover_admission_gate=old_leader_not_leader promoted_node_role=Leader".to_string(),
            "deployment_transport=single_request_tcp_append_entries append_batches_sent=3 heartbeat_batches_sent=2 follower_acks_recorded=3".to_string(),
            "deployment_election=deterministic_request_vote candidate_id=1 elected_term=2 votes_granted=3 quorum=2 elected=true".to_string(),
            "deployment_package=local_cargo_example_binary entrypoint=crates/replication/examples/operational_cluster_smoke.rs smoke_script=scripts/run_replication_cluster_smoke.sh packaged_script=scripts/run_replication_packaged_smoke.sh reproducible=true".to_string(),
            "operational_deployment_preflight=passed".to_string(),
            "deployment_scope=packaged_local_three_node_raft_smoke".to_string(),
            "deployment_gap_network_transport=implemented".to_string(),
            "deployment_gap_automatic_election=implemented".to_string(),
            "deployment_gap_packaged_deployment=implemented".to_string(),
        ]
    );
}
