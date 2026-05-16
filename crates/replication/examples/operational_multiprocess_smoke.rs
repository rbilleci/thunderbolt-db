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

const FOLLOWER_REQUESTS: usize = 2;
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

struct FollowerProcess {
    id: u64,
    addr: SocketAddr,
    child: Child,
    stdout: BufReader<ChildStdout>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("--follower") {
        return run_follower(&args[2..]);
    }
    run_parent()
}

fn run_parent() -> Result<(), Box<dyn Error>> {
    let mut leader = RaftReplicator::new(3);
    leader.become_leader(1);

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

    let mut follower_a = spawn_follower(2)?;
    let mut follower_b = spawn_follower(3)?;
    let mut append_batches_sent = 0usize;
    let mut heartbeat_batches_sent = 0usize;
    let mut follower_acks_recorded = 0usize;

    for follower in [&follower_a, &follower_b] {
        let response = send_append_entries_once(
            follower.addr,
            &AppendEntriesRequest {
                leader_term: term_one,
                prev_log_index: 0,
                prev_log_term: 0,
                entries: first_batch.clone(),
                leader_commit: 0,
            },
            TIMEOUT,
        )?;
        if !response.accepted {
            return Err(format!(
                "follower {} rejected initial append: {:?}",
                follower.id, response.error
            )
            .into());
        }
        append_batches_sent += 1;
        leader.register_follower_ack(first.index, follower.id);
        leader.register_follower_ack(second.index, follower.id);
        follower_acks_recorded += 2;
    }

    leader.wait_committed(second, TIMEOUT)?;

    for follower in [&follower_a, &follower_b] {
        let response = send_append_entries_once(
            follower.addr,
            &AppendEntriesRequest {
                leader_term: term_one,
                prev_log_index: second.index,
                prev_log_term: term_one,
                entries: vec![],
                leader_commit: leader.commit_index(),
            },
            TIMEOUT,
        )?;
        if !response.accepted {
            return Err(format!(
                "follower {} rejected commit heartbeat: {:?}",
                follower.id, response.error
            )
            .into());
        }
        heartbeat_batches_sent += 1;
    }

    let follower_a_output = finish_follower(&mut follower_a)?;
    let follower_b_output = finish_follower(&mut follower_b)?;
    let follower_lines = [follower_a_output, follower_b_output];
    for expected in [
        "multiprocess_follower id=2 commit=2 applied=2 caught_up=true read_after_apply=create table t(id int) | insert into t values (1)",
        "multiprocess_follower id=3 commit=2 applied=2 caught_up=true read_after_apply=create table t(id int) | insert into t values (1)",
    ] {
        if !follower_lines.iter().any(|line| line.contains(expected)) {
            return Err(format!("missing follower evidence line: {expected}").into());
        }
    }

    println!("operational_replication_multiprocess_smoke=passed");
    println!("multiprocess_deployment_scope=parent_leader_two_follower_processes");
    println!(
        "multiprocess_transport=tcp_append_entries child_processes=2 append_batches_sent={append_batches_sent} heartbeat_batches_sent={heartbeat_batches_sent} follower_acks_recorded={follower_acks_recorded}"
    );
    for line in follower_lines {
        print!("{line}");
    }
    println!("deployment_gap_packaged_multiprocess_smoke=implemented");
    println!("deployment_gap_long_running_service=missing");
    println!("deployment_gap_container_deployment=missing");

    Ok(())
}

fn spawn_follower(id: u64) -> Result<FollowerProcess, Box<dyn Error>> {
    let current = env::current_exe()?;
    let mut child = Command::new(current)
        .arg("--follower")
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
        .ok_or("follower stdout was not captured")?;
    let mut stdout = BufReader::new(stdout);
    let mut ready = String::new();
    stdout.read_line(&mut ready)?;
    let prefix = format!("multiprocess_follower_ready id={id} addr=");
    let addr = ready
        .trim()
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("unexpected follower ready line: {}", ready.trim()))?
        .parse::<SocketAddr>()?;
    Ok(FollowerProcess {
        id,
        addr,
        child,
        stdout,
    })
}

fn finish_follower(follower: &mut FollowerProcess) -> Result<String, Box<dyn Error>> {
    let mut output = String::new();
    follower.stdout.read_to_string(&mut output)?;
    let status = follower.child.wait()?;
    if !status.success() {
        return Err(format!("follower {} exited with {status}", follower.id).into());
    }
    Ok(output)
}

fn run_follower(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut id = None;
    let mut expected_requests = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--id" => id = iter.next().and_then(|value| value.parse::<u64>().ok()),
            "--expected-requests" => {
                expected_requests = iter.next().and_then(|value| value.parse::<usize>().ok());
            }
            _ => return Err(format!("unknown follower argument: {arg}").into()),
        }
    }
    let id = id.ok_or("missing --id")?;
    let expected_requests = expected_requests.ok_or("missing --expected-requests")?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    println!(
        "multiprocess_follower_ready id={id} addr={}",
        listener.local_addr()?
    );
    std::io::stdout().flush()?;

    let mut follower = RaftReplicator::new(3);
    let mut applied = AppliedLog::default();
    for _ in 0..expected_requests {
        let response = serve_append_entries_once(&listener, &mut follower, TIMEOUT)?;
        if !response.accepted {
            return Err(format!("append rejected for follower {id}: {:?}", response.error).into());
        }
        apply_committed(&mut follower, &mut applied)?;
    }

    println!(
        "multiprocess_follower id={id} commit={} applied={} caught_up={} read_after_apply={}",
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
