use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Barrier;
use tokio::task::JoinSet;
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};

type DynError = Box<dyn Error + Send + Sync>;

#[derive(Clone)]
struct Args {
    url: String,
    sqls: Vec<String>,
    expecteds: Vec<String>,
    concurrency: usize,
    requests_per_client: usize,
    warmup_requests_per_client: usize,
    pipeline_depth: usize,
    run_dir: PathBuf,
}

#[derive(Clone)]
struct RequestResult {
    client: usize,
    request: usize,
    latency_us: u128,
    status: &'static str,
    actual: String,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), DynError> {
    let sql = required_env("GPU_DB_PERSISTENT_PGWIRE_SQL")?;
    let expected = required_env("GPU_DB_PERSISTENT_PGWIRE_EXPECTED")?;
    let sqls = env::var("GPU_DB_PERSISTENT_PGWIRE_SQL_LIST")
        .ok()
        .map(|value| split_list_env(&value))
        .unwrap_or_else(|| vec![sql.clone()]);
    let expecteds = env::var("GPU_DB_PERSISTENT_PGWIRE_EXPECTED_LIST")
        .ok()
        .map(|value| split_list_env(&value))
        .unwrap_or_else(|| vec![expected.clone()]);
    if sqls.len() != expecteds.len() {
        return Err("GPU_DB_PERSISTENT_PGWIRE_SQL_LIST and GPU_DB_PERSISTENT_PGWIRE_EXPECTED_LIST must have the same length".into());
    }
    if sqls.is_empty() {
        return Err("GPU_DB_PERSISTENT_PGWIRE_SQL_LIST must not be empty".into());
    }
    let args = Args {
        url: required_env("GPU_DB_PERSISTENT_PGWIRE_URL")?,
        sqls,
        expecteds,
        concurrency: parse_usize_env("GPU_DB_PERSISTENT_PGWIRE_CONCURRENCY", 1)?,
        requests_per_client: parse_usize_env("GPU_DB_PERSISTENT_PGWIRE_REQUESTS_PER_CLIENT", 1)?,
        warmup_requests_per_client: parse_usize_env(
            "GPU_DB_PERSISTENT_PGWIRE_WARMUP_REQUESTS_PER_CLIENT",
            0,
        )?,
        pipeline_depth: parse_usize_env("GPU_DB_PERSISTENT_PGWIRE_PIPELINE_DEPTH", 1)?,
        run_dir: PathBuf::from(required_env("GPU_DB_PERSISTENT_PGWIRE_RUN_DIR")?),
    };
    if args.concurrency == 0 {
        return Err("GPU_DB_PERSISTENT_PGWIRE_CONCURRENCY must be positive".into());
    }
    if args.requests_per_client == 0 {
        return Err("GPU_DB_PERSISTENT_PGWIRE_REQUESTS_PER_CLIENT must be positive".into());
    }
    if args.pipeline_depth == 0 {
        return Err("GPU_DB_PERSISTENT_PGWIRE_PIPELINE_DEPTH must be positive".into());
    }
    fs::create_dir_all(&args.run_dir)?;

    let mut clients = Vec::with_capacity(args.concurrency);
    for client_id in 1..=args.concurrency {
        clients.push(connect_client(&args.url, client_id).await?);
    }

    let barrier = Arc::new(Barrier::new(args.concurrency + 1));
    let mut tasks = Vec::with_capacity(args.concurrency);
    for (client_index, client) in clients.into_iter().enumerate() {
        let task_args = args.clone();
        let task_barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            run_client(client_index + 1, client, task_args, task_barrier).await
        }));
    }

    let wall_start = Instant::now();
    barrier.wait().await;
    let mut all_results = Vec::new();
    for task in tasks {
        let mut results = task.await??;
        all_results.append(&mut results);
    }
    let wall_us = wall_start.elapsed().as_micros();

    write_result_artifacts(&args, wall_us, &all_results)?;
    Ok(())
}

async fn connect_client(url: &str, client_id: usize) -> Result<Arc<Client>, DynError> {
    let (client, connection) = tokio_postgres::connect(url, NoTls).await?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            eprintln!("persistent_pgwire_client_{client_id}_connection_error={error}");
        }
    });
    Ok(Arc::new(client))
}

async fn run_client(
    client_id: usize,
    client: Arc<Client>,
    args: Args,
    barrier: Arc<Barrier>,
) -> Result<Vec<RequestResult>, DynError> {
    barrier.wait().await;
    for _ in 0..args.warmup_requests_per_client {
        let (sql, expected) = request_sql_expected(&args, client_id, 0);
        let messages = client.simple_query(sql).await?;
        let actual = simple_query_actual(&messages);
        if actual != expected {
            return Err(format!(
                "warmup request for client {client_id} returned {actual:?}, expected {expected:?}"
            )
            .into());
        }
    }
    if args.pipeline_depth == 1 {
        return run_client_serial(client_id, client, args).await;
    }
    run_client_pipelined(client_id, client, args).await
}

async fn run_client_serial(
    client_id: usize,
    client: Arc<Client>,
    args: Args,
) -> Result<Vec<RequestResult>, DynError> {
    let mut results = Vec::with_capacity(args.requests_per_client);
    for request_id in 1..=args.requests_per_client {
        let (sql, expected) = request_sql_expected(&args, client_id, request_id);
        let started = Instant::now();
        match client.simple_query(sql).await {
            Ok(messages) => {
                let actual = simple_query_actual(&messages);
                let status = if actual == expected {
                    "pass"
                } else {
                    "wrong_result"
                };
                results.push(RequestResult {
                    client: client_id,
                    request: request_id,
                    latency_us: started.elapsed().as_micros(),
                    status,
                    actual,
                });
            }
            Err(error) => {
                results.push(RequestResult {
                    client: client_id,
                    request: request_id,
                    latency_us: started.elapsed().as_micros(),
                    status: "error",
                    actual: error.to_string(),
                });
            }
        }
    }
    Ok(results)
}

async fn run_client_pipelined(
    client_id: usize,
    client: Arc<Client>,
    args: Args,
) -> Result<Vec<RequestResult>, DynError> {
    let mut results = Vec::with_capacity(args.requests_per_client);
    let mut next_request_id = 1;
    let mut in_flight = JoinSet::new();
    while next_request_id <= args.requests_per_client || !in_flight.is_empty() {
        while next_request_id <= args.requests_per_client && in_flight.len() < args.pipeline_depth {
            let request_id = next_request_id;
            next_request_id += 1;
            let (sql, expected) = request_sql_expected(&args, client_id, request_id);
            let sql = sql.to_string();
            let expected = expected.to_string();
            let request_client = client.clone();
            in_flight.spawn(async move {
                let started = Instant::now();
                match request_client.simple_query(&sql).await {
                    Ok(messages) => {
                        let actual = simple_query_actual(&messages);
                        let status = if actual == expected {
                            "pass"
                        } else {
                            "wrong_result"
                        };
                        RequestResult {
                            client: client_id,
                            request: request_id,
                            latency_us: started.elapsed().as_micros(),
                            status,
                            actual,
                        }
                    }
                    Err(error) => RequestResult {
                        client: client_id,
                        request: request_id,
                        latency_us: started.elapsed().as_micros(),
                        status: "error",
                        actual: error.to_string(),
                    },
                }
            });
        }
        let Some(result) = in_flight.join_next().await else {
            continue;
        };
        results.push(result?);
    }
    Ok(results)
}

fn request_sql_expected(args: &Args, client_id: usize, request_id: usize) -> (&str, &str) {
    let idx = (client_id + request_id).saturating_sub(1) % args.sqls.len();
    (&args.sqls[idx], &args.expecteds[idx])
}

fn split_list_env(value: &str) -> Vec<String> {
    value.split(";;").map(str::to_string).collect()
}

fn simple_query_actual(messages: &[SimpleQueryMessage]) -> String {
    let mut rows = Vec::new();
    for message in messages {
        if let SimpleQueryMessage::Row(row) = message {
            let mut values = Vec::with_capacity(row.len());
            for index in 0..row.len() {
                values.push(row.get(index).unwrap_or("").to_string());
            }
            rows.push(values.join("|"));
        }
    }
    rows.join("|")
}

fn write_result_artifacts(
    args: &Args,
    wall_us: u128,
    results: &[RequestResult],
) -> Result<(), DynError> {
    let mut latencies = results
        .iter()
        .map(|result| result.latency_us)
        .collect::<Vec<_>>();
    latencies.sort_unstable();
    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);
    let error_count = results
        .iter()
        .filter(|result| result.status != "pass")
        .count();
    let request_count = results.len();
    let throughput = if wall_us > 0 {
        request_count as f64 * 1_000_000.0 / wall_us as f64
    } else {
        0.0
    };

    let mut sorted = File::create(args.run_dir.join("latencies.sorted"))?;
    for latency in &latencies {
        writeln!(sorted, "{latency}")?;
    }

    for result in results {
        let path = args.run_dir.join(format!(
            "client-{}-request-{}.result",
            result.client, result.request
        ));
        let mut file = File::create(path)?;
        writeln!(
            file,
            "{},{},{}",
            result.latency_us,
            result.status,
            json_escape(&result.actual)
        )?;
    }

    let mut summary = File::create(args.run_dir.join("summary.env"))?;
    writeln!(summary, "request_count={request_count}")?;
    writeln!(summary, "p50_us={p50}")?;
    writeln!(summary, "p95_us={p95}")?;
    writeln!(summary, "p99_us={p99}")?;
    writeln!(summary, "wall_us={wall_us}")?;
    writeln!(summary, "throughput_qps={throughput:.6}")?;
    writeln!(summary, "error_count={error_count}")?;
    writeln!(
        summary,
        "correctness_status={}",
        if error_count == 0 { "pass" } else { "error" }
    )?;
    writeln!(summary, "client_driver=tokio-postgres/simple-query")?;
    writeln!(summary, "persistent_sessions={}", args.concurrency)?;
    writeln!(summary, "sql_schedule_len={}", args.sqls.len())?;
    writeln!(summary, "requests_per_client={}", args.requests_per_client)?;
    writeln!(
        summary,
        "warmup_requests_per_client={}",
        args.warmup_requests_per_client
    )?;
    writeln!(summary, "pipeline_depth={}", args.pipeline_depth)?;
    Ok(())
}

fn percentile(latencies: &[u128], q: f64) -> u128 {
    if latencies.is_empty() {
        return 0;
    }
    let mut index = (q * latencies.len() as f64).ceil() as usize;
    index = index.saturating_sub(1).min(latencies.len() - 1);
    latencies[index]
}

fn required_env(name: &str) -> Result<String, DynError> {
    env::var(name).map_err(|_| format!("{name} is required").into())
}

fn parse_usize_env(name: &str, default: usize) -> Result<usize, DynError> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(_) => Ok(default),
    }
}

fn json_escape(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped
}
