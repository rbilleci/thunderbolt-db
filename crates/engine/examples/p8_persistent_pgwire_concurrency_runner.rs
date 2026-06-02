use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Barrier;
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};

type DynError = Box<dyn Error + Send + Sync>;

#[derive(Clone)]
struct Args {
    url: String,
    sql: String,
    expected: String,
    concurrency: usize,
    requests_per_client: usize,
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
    let args = Args {
        url: required_env("GPU_DB_PERSISTENT_PGWIRE_URL")?,
        sql: required_env("GPU_DB_PERSISTENT_PGWIRE_SQL")?,
        expected: required_env("GPU_DB_PERSISTENT_PGWIRE_EXPECTED")?,
        concurrency: parse_usize_env("GPU_DB_PERSISTENT_PGWIRE_CONCURRENCY", 1)?,
        requests_per_client: parse_usize_env("GPU_DB_PERSISTENT_PGWIRE_REQUESTS_PER_CLIENT", 1)?,
        run_dir: PathBuf::from(required_env("GPU_DB_PERSISTENT_PGWIRE_RUN_DIR")?),
    };
    if args.concurrency == 0 {
        return Err("GPU_DB_PERSISTENT_PGWIRE_CONCURRENCY must be positive".into());
    }
    if args.requests_per_client == 0 {
        return Err("GPU_DB_PERSISTENT_PGWIRE_REQUESTS_PER_CLIENT must be positive".into());
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

async fn connect_client(url: &str, client_id: usize) -> Result<Client, DynError> {
    let (client, connection) = tokio_postgres::connect(url, NoTls).await?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            eprintln!("persistent_pgwire_client_{client_id}_connection_error={error}");
        }
    });
    Ok(client)
}

async fn run_client(
    client_id: usize,
    client: Client,
    args: Args,
    barrier: Arc<Barrier>,
) -> Result<Vec<RequestResult>, DynError> {
    let mut results = Vec::with_capacity(args.requests_per_client);
    barrier.wait().await;
    for request_id in 1..=args.requests_per_client {
        let started = Instant::now();
        match client.simple_query(&args.sql).await {
            Ok(messages) => {
                let actual = simple_query_actual(&messages);
                let status = if actual == args.expected {
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
    writeln!(summary, "requests_per_client={}", args.requests_per_client)?;
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
