use std::{
    env,
    error::Error,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener},
    process::{Child, ChildStdout, Command, Stdio},
    time::Duration,
};

use gpu_db_replication::{
    send_append_entries_once, serve_append_entries_once, AppendEntriesRequest, LogReplicator,
    RaftReplicator, ReplicatedStateMachine,
};
use gpu_db_types::{EngineError, LogEntry};

const FOLLOWER_REQUESTS: usize = 4;
const TIMEOUT: Duration = Duration::from_secs(2);

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

struct FollowerService {
    id: u64,
    addr: SocketAddr,
    child: Child,
    stdout: BufReader<ChildStdout>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("--follower-service") {
        return run_follower_service(&args[2..]);
    }
    run_parent()
}

fn run_parent() -> Result<(), Box<dyn Error>> {
    let mut leader = RaftReplicator::new(3);
    leader.become_leader(1);

    let first = leader.propose(b"create table t(id int)".to_vec())?;
    let second = leader.propose(b"insert into t values (1)".to_vec())?;
    let term = leader.current_term();
    let first_batch = vec![
        LogEntry {
            term,
            index: first.index,
            payload: b"create table t(id int)".to_vec(),
        },
        LogEntry {
            term,
            index: second.index,
            payload: b"insert into t values (1)".to_vec(),
        },
    ];

    let mut follower_a = spawn_follower_service(2)?;
    let mut follower_b = spawn_follower_service(3)?;
    let mut append_batches_sent = 0usize;
    let mut heartbeat_batches_sent = 0usize;
    let mut follower_acks_recorded = 0usize;

    for follower in [&follower_a, &follower_b] {
        send_checked(
            follower,
            AppendEntriesRequest {
                leader_term: term,
                prev_log_index: 0,
                prev_log_term: 0,
                entries: first_batch.clone(),
                leader_commit: 0,
            },
        )?;
        append_batches_sent += 1;
        leader.register_follower_ack(first.index, follower.id);
        leader.register_follower_ack(second.index, follower.id);
        follower_acks_recorded += 2;
    }
    leader.wait_committed(second, TIMEOUT)?;

    for follower in [&follower_a, &follower_b] {
        send_checked(
            follower,
            AppendEntriesRequest {
                leader_term: term,
                prev_log_index: second.index,
                prev_log_term: term,
                entries: vec![],
                leader_commit: leader.commit_index(),
            },
        )?;
        heartbeat_batches_sent += 1;
    }

    let third = leader.propose(b"insert into t values (2)".to_vec())?;
    for follower in [&follower_a, &follower_b] {
        send_checked(
            follower,
            AppendEntriesRequest {
                leader_term: term,
                prev_log_index: second.index,
                prev_log_term: term,
                entries: vec![LogEntry {
                    term,
                    index: third.index,
                    payload: b"insert into t values (2)".to_vec(),
                }],
                leader_commit: leader.commit_index(),
            },
        )?;
        append_batches_sent += 1;
        leader.register_follower_ack(third.index, follower.id);
        follower_acks_recorded += 1;
    }
    leader.wait_committed(third, TIMEOUT)?;

    for follower in [&follower_a, &follower_b] {
        send_checked(
            follower,
            AppendEntriesRequest {
                leader_term: term,
                prev_log_index: third.index,
                prev_log_term: term,
                entries: vec![],
                leader_commit: leader.commit_index(),
            },
        )?;
        heartbeat_batches_sent += 1;
    }

    let follower_a_output = finish_follower_service(&mut follower_a)?;
    let follower_b_output = finish_follower_service(&mut follower_b)?;
    let follower_lines = [follower_a_output, follower_b_output];
    for expected in [
        "service_follower id=2 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)",
        "service_follower id=3 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)",
    ] {
        if !follower_lines.iter().any(|line| line.contains(expected)) {
            return Err(format!("missing service follower evidence line: {expected}").into());
        }
    }

    println!("operational_replication_service_smoke=passed");
    println!("service_deployment_scope=parent_leader_two_long_running_follower_services");
    println!(
        "service_transport=tcp_append_entries follower_services=2 append_batches_sent={append_batches_sent} heartbeat_batches_sent={heartbeat_batches_sent} follower_acks_recorded={follower_acks_recorded} requests_per_service={FOLLOWER_REQUESTS}"
    );
    for line in follower_lines {
        print!("{line}");
    }
    println!("service_shutdown=controlled follower_services=2");
    println!("deployment_gap_long_running_service=implemented");
    println!("deployment_gap_container_deployment=missing");

    Ok(())
}

fn send_checked(
    follower: &FollowerService,
    request: AppendEntriesRequest,
) -> Result<(), Box<dyn Error>> {
    let response = send_append_entries_once(follower.addr, &request, TIMEOUT)?;
    if response.accepted {
        Ok(())
    } else {
        Err(format!(
            "follower {} rejected append entries: {:?}",
            follower.id, response.error
        )
        .into())
    }
}

fn spawn_follower_service(id: u64) -> Result<FollowerService, Box<dyn Error>> {
    let current = env::current_exe()?;
    let mut child = Command::new(current)
        .arg("--follower-service")
        .arg("--id")
        .arg(id.to_string())
        .arg("--expected-requests")
        .arg(FOLLOWER_REQUESTS.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or("follower service stdout was not captured")?;
    let mut stdout = BufReader::new(stdout);
    let mut ready = String::new();
    stdout.read_line(&mut ready)?;
    let prefix = format!("service_follower_ready id={id} addr=");
    let addr = ready
        .trim()
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("unexpected follower service ready line: {}", ready.trim()))?
        .parse::<SocketAddr>()?;
    Ok(FollowerService {
        id,
        addr,
        child,
        stdout,
    })
}

fn finish_follower_service(follower: &mut FollowerService) -> Result<String, Box<dyn Error>> {
    let mut output = String::new();
    follower.stdout.read_to_string(&mut output)?;
    let status = follower.child.wait()?;
    if !status.success() {
        return Err(format!("follower service {} exited with {status}", follower.id).into());
    }
    Ok(output)
}

fn run_follower_service(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut id = None;
    let mut expected_requests = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--id" => id = iter.next().and_then(|value| value.parse::<u64>().ok()),
            "--expected-requests" => {
                expected_requests = iter.next().and_then(|value| value.parse::<usize>().ok());
            }
            _ => return Err(format!("unknown follower service argument: {arg}").into()),
        }
    }
    let id = id.ok_or("missing --id")?;
    let expected_requests = expected_requests.ok_or("missing --expected-requests")?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    println!(
        "service_follower_ready id={id} addr={}",
        listener.local_addr()?
    );
    std::io::stdout().flush()?;

    let mut follower = RaftReplicator::new(3);
    let mut applied = AppliedLog::default();
    for _ in 0..expected_requests {
        let response = serve_append_entries_once(&listener, &mut follower, TIMEOUT)?;
        if !response.accepted {
            return Err(format!(
                "append rejected for follower service {id}: {:?}",
                response.error
            )
            .into());
        }
        apply_committed(&mut follower, &mut applied)?;
    }

    println!(
        "service_follower id={id} commit={} applied={} caught_up={} read_after_apply={}",
        follower.commit_index(),
        follower.applied_index(),
        follower.progress().is_caught_up(),
        applied.values.join(" | ")
    );
    Ok(())
}

fn apply_committed(
    node: &mut RaftReplicator,
    state: &mut impl ReplicatedStateMachine,
) -> Result<(), EngineError> {
    let committed = node
        .drain_committed_from(node.applied_index())
        .cloned()
        .collect::<Vec<_>>();
    for entry in committed {
        state.apply(&entry)?;
        node.mark_applied(entry.index);
    }
    Ok(())
}
