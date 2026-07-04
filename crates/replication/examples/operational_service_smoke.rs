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
    child: Option<Child>,
    stdout: Option<BufReader<ChildStdout>>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("--follower-service") {
        return run_follower_service(&args[2..]);
    }
    if args.get(1).map(String::as_str) == Some("--supervised-restart") {
        return run_supervised_restart_parent();
    }
    if args.get(1).map(String::as_str) == Some("--container-supervised-restart") {
        return run_container_supervised_restart_parent(&args[2..]);
    }
    if args.get(1).map(String::as_str) == Some("--compose-supervised-restart") {
        return run_compose_supervised_restart_parent(&args[2..]);
    }
    run_parent(&args[1..])
}

fn run_parent(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut leader = RaftReplicator::new(3);
    leader.become_leader(1);

    let first = leader.propose(b"create table t(id int)".to_vec().into())?;
    let second = leader.propose(b"insert into t values (1)".to_vec().into())?;
    let term = leader.current_term();
    let first_batch = vec![
        LogEntry {
            term,
            index: first.index,
            payload: b"create table t(id int)".to_vec().into(),
        },
        LogEntry {
            term,
            index: second.index,
            payload: b"insert into t values (1)".to_vec().into(),
        },
    ];

    let mut followers = parse_external_followers(args)?;
    let external_followers = !followers.is_empty();
    if followers.is_empty() {
        followers.push(spawn_follower_service(2)?);
        followers.push(spawn_follower_service(3)?);
    }
    if followers.len() != 2 {
        return Err(format!("expected exactly two followers, got {}", followers.len()).into());
    }
    let mut append_batches_sent = 0usize;
    let mut heartbeat_batches_sent = 0usize;
    let mut follower_acks_recorded = 0usize;

    for follower in &followers {
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

    for follower in &followers {
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

    let third = leader.propose(b"insert into t values (2)".to_vec().into())?;
    for follower in &followers {
        send_checked(
            follower,
            AppendEntriesRequest {
                leader_term: term,
                prev_log_index: second.index,
                prev_log_term: term,
                entries: vec![LogEntry {
                    term,
                    index: third.index,
                    payload: b"insert into t values (2)".to_vec().into(),
                }],
                leader_commit: leader.commit_index(),
            },
        )?;
        append_batches_sent += 1;
        leader.register_follower_ack(third.index, follower.id);
        follower_acks_recorded += 1;
    }
    leader.wait_committed(third, TIMEOUT)?;

    for follower in &followers {
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

    let mut follower_lines = Vec::new();
    for follower in &mut followers {
        let output = finish_follower_service(follower)?;
        if !output.is_empty() {
            follower_lines.push(output);
        }
    }
    if !external_followers {
        for expected in [
            "service_follower id=2 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)",
            "service_follower id=3 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)",
        ] {
            if !follower_lines.iter().any(|line| line.contains(expected)) {
                return Err(format!("missing service follower evidence line: {expected}").into());
            }
        }
    }

    if external_followers {
        println!("operational_replication_container_smoke=host_parent_passed");
        println!("container_deployment_scope=host_leader_two_follower_service_containers");
    } else {
        println!("operational_replication_service_smoke=passed");
        println!("service_deployment_scope=parent_leader_two_long_running_follower_services");
    }
    println!(
        "service_transport=tcp_append_entries follower_services=2 append_batches_sent={append_batches_sent} heartbeat_batches_sent={heartbeat_batches_sent} follower_acks_recorded={follower_acks_recorded} requests_per_service={FOLLOWER_REQUESTS}"
    );
    for line in follower_lines {
        print!("{line}");
    }
    println!("service_shutdown=controlled follower_services=2");
    println!("deployment_gap_long_running_service=implemented");
    println!(
        "deployment_gap_container_deployment={}",
        if external_followers {
            "implemented"
        } else {
            "missing"
        }
    );

    Ok(())
}

fn run_supervised_restart_parent() -> Result<(), Box<dyn Error>> {
    let mut leader = RaftReplicator::new(3);
    leader.become_leader(1);

    let first = leader.propose(b"create table t(id int)".to_vec().into())?;
    let second = leader.propose(b"insert into t values (1)".to_vec().into())?;
    let term = leader.current_term();
    let first_batch = vec![
        LogEntry {
            term,
            index: first.index,
            payload: b"create table t(id int)".to_vec().into(),
        },
        LogEntry {
            term,
            index: second.index,
            payload: b"insert into t values (1)".to_vec().into(),
        },
    ];

    let mut restarting_follower = spawn_follower_service_with_requests(2, 2)?;
    let mut stable_follower = spawn_follower_service(3)?;
    let mut append_batches_sent = 0usize;
    let mut heartbeat_batches_sent = 0usize;
    let mut follower_acks_recorded = 0usize;

    for follower in [&restarting_follower, &stable_follower] {
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

    for follower in [&restarting_follower, &stable_follower] {
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

    let before_restart_output = finish_follower_service(&mut restarting_follower)?;
    if !before_restart_output.contains(
        "service_follower id=2 commit=2 applied=2 caught_up=true read_after_apply=create table t(id int) | insert into t values (1)",
    ) {
        return Err("restarting follower did not report pre-restart catch-up".into());
    }

    let third = leader.propose(b"insert into t values (2)".to_vec().into())?;
    send_checked(
        &stable_follower,
        AppendEntriesRequest {
            leader_term: term,
            prev_log_index: second.index,
            prev_log_term: term,
            entries: vec![LogEntry {
                term,
                index: third.index,
                payload: b"insert into t values (2)".to_vec().into(),
            }],
            leader_commit: leader.commit_index(),
        },
    )?;
    append_batches_sent += 1;
    leader.register_follower_ack(third.index, stable_follower.id);
    follower_acks_recorded += 1;

    let mut restarted_follower = spawn_follower_service_with_requests(2, 2)?;
    let replay_batch = vec![
        LogEntry {
            term,
            index: first.index,
            payload: b"create table t(id int)".to_vec().into(),
        },
        LogEntry {
            term,
            index: second.index,
            payload: b"insert into t values (1)".to_vec().into(),
        },
        LogEntry {
            term,
            index: third.index,
            payload: b"insert into t values (2)".to_vec().into(),
        },
    ];
    send_checked(
        &restarted_follower,
        AppendEntriesRequest {
            leader_term: term,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: replay_batch,
            leader_commit: leader.commit_index(),
        },
    )?;
    append_batches_sent += 1;
    leader.register_follower_ack(third.index, restarted_follower.id);
    follower_acks_recorded += 1;
    leader.wait_committed(third, TIMEOUT)?;

    for follower in [&restarted_follower, &stable_follower] {
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

    let restarted_output = finish_follower_service(&mut restarted_follower)?;
    let stable_output = finish_follower_service(&mut stable_follower)?;
    let expected_restarted = "service_follower id=2 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)";
    let expected_stable = "service_follower id=3 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)";
    if !restarted_output.contains(expected_restarted) {
        return Err(
            format!("missing restarted follower evidence line: {expected_restarted}").into(),
        );
    }
    if !stable_output.contains(expected_stable) {
        return Err(format!("missing stable follower evidence line: {expected_stable}").into());
    }

    println!("operational_replication_supervised_restart_smoke=passed");
    println!("supervised_restart_scope=parent_leader_restarts_one_follower_service");
    println!(
        "supervised_restart_transport=tcp_append_entries follower_services=2 restarted_follower=2 append_batches_sent={append_batches_sent} heartbeat_batches_sent={heartbeat_batches_sent} follower_acks_recorded={follower_acks_recorded}"
    );
    print!("{before_restart_output}");
    print!("{restarted_output}");
    print!("{stable_output}");
    println!("supervised_restart_replay=full_durable_prefix_after_restart");
    println!("service_shutdown=controlled follower_services=2");
    println!("deployment_gap_service_restart_supervision=implemented_bounded_local_smoke");
    println!("deployment_gap_production_supervision=missing");
    println!("deployment_gap_kubernetes_deployment=missing");

    Ok(())
}

fn run_container_supervised_restart_parent(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut restarting_follower = None;
    let mut stable_follower = None;
    let mut restart_container = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--restarting-follower" => {
                restarting_follower = Some(parse_external_follower(
                    iter.next().ok_or("missing --restarting-follower value")?,
                )?);
            }
            "--stable-follower" => {
                stable_follower = Some(parse_external_follower(
                    iter.next().ok_or("missing --stable-follower value")?,
                )?);
            }
            "--restart-container" => {
                restart_container = Some(iter.next().ok_or("missing --restart-container value")?);
            }
            _ => return Err(format!("unknown container restart argument: {arg}").into()),
        }
    }
    let mut restarting_follower = restarting_follower.ok_or("missing --restarting-follower")?;
    let stable_follower = stable_follower.ok_or("missing --stable-follower")?;
    let restart_container = restart_container.ok_or("missing --restart-container")?;
    if restarting_follower.id == stable_follower.id {
        return Err("restarting and stable followers must be distinct".into());
    }

    let mut leader = RaftReplicator::new(3);
    leader.become_leader(1);

    let first = leader.propose(b"create table t(id int)".to_vec().into())?;
    let second = leader.propose(b"insert into t values (1)".to_vec().into())?;
    let term = leader.current_term();
    let first_batch = vec![
        LogEntry {
            term,
            index: first.index,
            payload: b"create table t(id int)".to_vec().into(),
        },
        LogEntry {
            term,
            index: second.index,
            payload: b"insert into t values (1)".to_vec().into(),
        },
    ];
    let mut append_batches_sent = 0usize;
    let mut heartbeat_batches_sent = 0usize;
    let mut follower_acks_recorded = 0usize;

    for follower in [&restarting_follower, &stable_follower] {
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

    for follower in [&restarting_follower, &stable_follower] {
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

    let restart_status = Command::new("docker")
        .arg("restart")
        .arg(restart_container)
        .stdout(Stdio::null())
        .status()?;
    if !restart_status.success() {
        return Err(format!("docker restart {restart_container} failed: {restart_status}").into());
    }
    restarting_follower.addr = docker_published_addr(restart_container)?;

    let third = leader.propose(b"insert into t values (2)".to_vec().into())?;
    send_checked(
        &stable_follower,
        AppendEntriesRequest {
            leader_term: term,
            prev_log_index: second.index,
            prev_log_term: term,
            entries: vec![LogEntry {
                term,
                index: third.index,
                payload: b"insert into t values (2)".to_vec().into(),
            }],
            leader_commit: leader.commit_index(),
        },
    )?;
    append_batches_sent += 1;
    leader.register_follower_ack(third.index, stable_follower.id);
    follower_acks_recorded += 1;

    let replay_batch = vec![
        LogEntry {
            term,
            index: first.index,
            payload: b"create table t(id int)".to_vec().into(),
        },
        LogEntry {
            term,
            index: second.index,
            payload: b"insert into t values (1)".to_vec().into(),
        },
        LogEntry {
            term,
            index: third.index,
            payload: b"insert into t values (2)".to_vec().into(),
        },
    ];
    send_checked_with_retry(
        &restarting_follower,
        AppendEntriesRequest {
            leader_term: term,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: replay_batch,
            leader_commit: leader.commit_index(),
        },
    )?;
    append_batches_sent += 1;
    leader.register_follower_ack(third.index, restarting_follower.id);
    follower_acks_recorded += 1;
    leader.wait_committed(third, TIMEOUT)?;

    for follower in [&restarting_follower, &stable_follower] {
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

    println!("operational_replication_container_restart_smoke=host_parent_passed");
    println!("container_restart_scope=host_leader_restarts_one_follower_container");
    println!(
        "container_restart_transport=tcp_append_entries follower_containers=2 restarted_follower={} stable_follower={} append_batches_sent={append_batches_sent} heartbeat_batches_sent={heartbeat_batches_sent} follower_acks_recorded={follower_acks_recorded}",
        restarting_follower.id, stable_follower.id
    );
    println!("container_restart_replay=full_durable_prefix_after_restart");
    println!("deployment_gap_container_restart_supervision=implemented_bounded_local_smoke");
    println!("deployment_gap_production_supervision=missing");
    println!("deployment_gap_kubernetes_deployment=missing");

    Ok(())
}

fn run_compose_supervised_restart_parent(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut restarting_follower = None;
    let mut stable_follower = None;
    let mut compose_file = None;
    let mut compose_project = None;
    let mut restart_service = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--restarting-follower" => {
                restarting_follower = Some(parse_external_follower(
                    iter.next().ok_or("missing --restarting-follower value")?,
                )?);
            }
            "--stable-follower" => {
                stable_follower = Some(parse_external_follower(
                    iter.next().ok_or("missing --stable-follower value")?,
                )?);
            }
            "--compose-file" => {
                compose_file = Some(iter.next().ok_or("missing --compose-file value")?);
            }
            "--compose-project" => {
                compose_project = Some(iter.next().ok_or("missing --compose-project value")?);
            }
            "--restart-service" => {
                restart_service = Some(iter.next().ok_or("missing --restart-service value")?);
            }
            _ => return Err(format!("unknown compose restart argument: {arg}").into()),
        }
    }
    let mut restarting_follower = restarting_follower.ok_or("missing --restarting-follower")?;
    let stable_follower = stable_follower.ok_or("missing --stable-follower")?;
    let compose_file = compose_file.ok_or("missing --compose-file")?;
    let compose_project = compose_project.ok_or("missing --compose-project")?;
    let restart_service = restart_service.ok_or("missing --restart-service")?;
    if restarting_follower.id == stable_follower.id {
        return Err("restarting and stable followers must be distinct".into());
    }

    let mut leader = RaftReplicator::new(3);
    leader.become_leader(1);

    let first = leader.propose(b"create table t(id int)".to_vec().into())?;
    let second = leader.propose(b"insert into t values (1)".to_vec().into())?;
    let term = leader.current_term();
    let first_batch = vec![
        LogEntry {
            term,
            index: first.index,
            payload: b"create table t(id int)".to_vec().into(),
        },
        LogEntry {
            term,
            index: second.index,
            payload: b"insert into t values (1)".to_vec().into(),
        },
    ];
    let mut append_batches_sent = 0usize;
    let mut heartbeat_batches_sent = 0usize;
    let mut follower_acks_recorded = 0usize;

    for follower in [&restarting_follower, &stable_follower] {
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

    for follower in [&restarting_follower, &stable_follower] {
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

    let restart_status = Command::new("docker")
        .arg("compose")
        .arg("-f")
        .arg(compose_file)
        .arg("-p")
        .arg(compose_project)
        .arg("restart")
        .arg(restart_service)
        .stdout(Stdio::null())
        .status()?;
    if !restart_status.success() {
        return Err(
            format!("docker compose restart {restart_service} failed: {restart_status}").into(),
        );
    }
    restarting_follower.addr =
        compose_published_addr(compose_file, compose_project, restart_service)?;

    let third = leader.propose(b"insert into t values (2)".to_vec().into())?;
    send_checked(
        &stable_follower,
        AppendEntriesRequest {
            leader_term: term,
            prev_log_index: second.index,
            prev_log_term: term,
            entries: vec![LogEntry {
                term,
                index: third.index,
                payload: b"insert into t values (2)".to_vec().into(),
            }],
            leader_commit: leader.commit_index(),
        },
    )?;
    append_batches_sent += 1;
    leader.register_follower_ack(third.index, stable_follower.id);
    follower_acks_recorded += 1;

    let replay_batch = vec![
        LogEntry {
            term,
            index: first.index,
            payload: b"create table t(id int)".to_vec().into(),
        },
        LogEntry {
            term,
            index: second.index,
            payload: b"insert into t values (1)".to_vec().into(),
        },
        LogEntry {
            term,
            index: third.index,
            payload: b"insert into t values (2)".to_vec().into(),
        },
    ];
    send_checked_with_retry(
        &restarting_follower,
        AppendEntriesRequest {
            leader_term: term,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: replay_batch,
            leader_commit: leader.commit_index(),
        },
    )?;
    append_batches_sent += 1;
    leader.register_follower_ack(third.index, restarting_follower.id);
    follower_acks_recorded += 1;
    leader.wait_committed(third, TIMEOUT)?;

    for follower in [&restarting_follower, &stable_follower] {
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

    println!("operational_replication_compose_restart_smoke=host_parent_passed");
    println!("compose_restart_scope=host_leader_restarts_one_compose_follower_service");
    println!(
        "compose_restart_transport=tcp_append_entries follower_services=2 restarted_follower={} stable_follower={} restart_service={} append_batches_sent={append_batches_sent} heartbeat_batches_sent={heartbeat_batches_sent} follower_acks_recorded={follower_acks_recorded}",
        restarting_follower.id, stable_follower.id, restart_service
    );
    println!("compose_restart_replay=full_durable_prefix_after_restart");
    println!("deployment_gap_compose_restart_supervision=implemented_bounded_local_smoke");
    println!("deployment_gap_production_supervision=missing");
    println!("deployment_gap_kubernetes_deployment=missing");

    Ok(())
}

fn parse_external_follower(value: &str) -> Result<FollowerService, Box<dyn Error>> {
    let (id, addr) = value
        .split_once('=')
        .ok_or_else(|| format!("external follower must be id=addr, got {value}"))?;
    Ok(FollowerService {
        id: id.parse()?,
        addr: addr.parse()?,
        child: None,
        stdout: None,
    })
}

fn docker_published_addr(container: &str) -> Result<SocketAddr, Box<dyn Error>> {
    let output = Command::new("docker")
        .arg("port")
        .arg(container)
        .arg("55432/tcp")
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "docker port {container} 55432/tcp failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    let line = stdout
        .lines()
        .next()
        .ok_or_else(|| format!("docker port {container} returned no mapping"))?;
    let port = line
        .rsplit_once(':')
        .ok_or_else(|| format!("unexpected docker port mapping: {line}"))?
        .1;
    Ok(format!("127.0.0.1:{port}").parse()?)
}

fn compose_published_addr(
    compose_file: &str,
    project: &str,
    service: &str,
) -> Result<SocketAddr, Box<dyn Error>> {
    let output = Command::new("docker")
        .arg("compose")
        .arg("-f")
        .arg(compose_file)
        .arg("-p")
        .arg(project)
        .arg("port")
        .arg(service)
        .arg("55432")
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "docker compose port {service} 55432 failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    let line = stdout
        .lines()
        .next()
        .ok_or_else(|| format!("docker compose port {service} returned no mapping"))?;
    let port = line
        .rsplit_once(':')
        .ok_or_else(|| format!("unexpected docker compose port mapping: {line}"))?
        .1;
    Ok(format!("127.0.0.1:{port}").parse()?)
}

fn parse_external_followers(args: &[String]) -> Result<Vec<FollowerService>, Box<dyn Error>> {
    let mut followers = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--external-follower" => {
                let value = iter.next().ok_or("missing --external-follower value")?;
                followers.push(parse_external_follower(value)?);
            }
            _ => return Err(format!("unknown parent argument: {arg}").into()),
        }
    }
    Ok(followers)
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

fn send_checked_with_retry(
    follower: &FollowerService,
    request: AppendEntriesRequest,
) -> Result<(), Box<dyn Error>> {
    let mut last_error = None;
    for _ in 0..50 {
        match send_checked(follower, request.clone()) {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_error = Some(err.to_string());
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
    Err(format!(
        "follower {} did not accept append entries after restart: {}",
        follower.id,
        last_error.unwrap_or_else(|| "unknown error".to_string())
    )
    .into())
}

fn spawn_follower_service(id: u64) -> Result<FollowerService, Box<dyn Error>> {
    spawn_follower_service_with_requests(id, FOLLOWER_REQUESTS)
}

fn spawn_follower_service_with_requests(
    id: u64,
    expected_requests: usize,
) -> Result<FollowerService, Box<dyn Error>> {
    let current = env::current_exe()?;
    let mut child = Command::new(current)
        .arg("--follower-service")
        .arg("--id")
        .arg(id.to_string())
        .arg("--expected-requests")
        .arg(expected_requests.to_string())
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
        child: Some(child),
        stdout: Some(stdout),
    })
}

fn finish_follower_service(follower: &mut FollowerService) -> Result<String, Box<dyn Error>> {
    let Some(stdout) = follower.stdout.as_mut() else {
        return Ok(String::new());
    };
    let mut output = String::new();
    stdout.read_to_string(&mut output)?;
    if let Some(child) = follower.child.as_mut() {
        let status = child.wait()?;
        if !status.success() {
            return Err(format!("follower service {} exited with {status}", follower.id).into());
        }
    }
    Ok(output)
}

fn run_follower_service(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut id = None;
    let mut expected_requests = None;
    let mut listen = "127.0.0.1:0".to_string();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--id" => id = iter.next().and_then(|value| value.parse::<u64>().ok()),
            "--expected-requests" => {
                expected_requests = iter.next().and_then(|value| value.parse::<usize>().ok());
            }
            "--listen" => listen = iter.next().ok_or("missing --listen value")?.to_string(),
            _ => return Err(format!("unknown follower service argument: {arg}").into()),
        }
    }
    let id = id.ok_or("missing --id")?;
    let expected_requests = expected_requests.ok_or("missing --expected-requests")?;
    let listener = TcpListener::bind(&listen)?;
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
