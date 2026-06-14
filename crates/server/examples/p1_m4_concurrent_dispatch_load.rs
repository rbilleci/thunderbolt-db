//! P1-M4 — concurrent-dispatch load harness (minimal open-loop / p99.9).
//!
//! Drives the engine-backed pgwire server over real TCP with `tokio-postgres`, sweeping
//! the number of concurrent client connections and capturing throughput + tail latency
//! (p50/p95/p99/p99.9). It runs against either dispatch model so the P1-M4 win is an A/B:
//!
//!   GPU_DB_LOAD_MODE=concurrent  cargo run -p gpu_db_server --example p1_m4_concurrent_dispatch_load --release
//!   GPU_DB_LOAD_MODE=sequential  cargo run -p gpu_db_server --example p1_m4_concurrent_dispatch_load --release
//!
//! - **concurrent** (`serve`): thread-per-connection over one shared engine; reads run
//!   concurrently (read lock), writes serialize (write lock).
//! - **sequential** (`serve_sequential`): one connection at a time — the prior baseline.
//!   At >1 offered connection it serves them strictly serially; only the few accepted
//!   back-to-back within the 5 s connect window report results, so `served` is a small
//!   run-dependent number (1–3 observed), the rest fail the connect deadline, and true
//!   sustained throughput stays pinned at the single-connection rate. (The reported qps at
//!   conn>1 is an artifact: it sums those serially-run clients' requests over one window —
//!   read the c1 row as the real ceiling.)
//!
//! This is the *minimal* cut of the Phase-5 harness (closed-loop per client = offered
//! concurrency via N connections, not a true open-loop offered-rate; fixed duration; one
//! query shape). It exists so the P1-M4 dispatch change is validated at load, per the
//! plan's "measure at load before the heavy lifts". Env: GPU_DB_LOAD_CONNECTIONS
//! (default 1,8,64,256), GPU_DB_LOAD_DURATION_SECS (3), GPU_DB_LOAD_ROWS (1000),
//! GPU_DB_LOAD_MODE (concurrent).

use std::env;
use std::error::Error;
use std::net::TcpListener;
use std::time::{Duration, Instant};

use tokio_postgres::NoTls;

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let mode = env::var("GPU_DB_LOAD_MODE").unwrap_or_else(|_| "concurrent".to_string());
    let connections: Vec<usize> = env::var("GPU_DB_LOAD_CONNECTIONS")
        .unwrap_or_else(|_| "1,8,64,256".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let duration_secs: u64 = env_parse("GPU_DB_LOAD_DURATION_SECS", 3);
    let rows: usize = env_parse("GPU_DB_LOAD_ROWS", 1000);
    let query = "SELECT COUNT(*) FROM load_t";

    // Bind first (so we know the port), then run the chosen server on its own OS thread —
    // `serve`/`serve_sequential` are blocking.
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    let server_mode = mode.clone();
    std::thread::spawn(move || {
        let _ = if server_mode == "sequential" {
            gpu_db_server::serve_sequential(listener)
        } else {
            gpu_db_server::serve(listener)
        };
    });
    let conn_str = format!("host=127.0.0.1 port={port} user=postgres dbname=postgres");

    // Seed a small table on one connection.
    {
        let (client, connection) = tokio_postgres::connect(&conn_str, NoTls).await?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client.simple_query("CREATE TABLE load_t (id INT)").await?;
        let mut insert = String::from("INSERT INTO load_t (id) VALUES ");
        for id in 0..rows {
            if id > 0 {
                insert.push(',');
            }
            insert.push_str(&format!("({id})"));
        }
        client.simple_query(&insert).await?;
    }

    println!(
        "p1_m4_concurrent_dispatch_load: mode={mode} rows={rows} duration={duration_secs}s query=\"{query}\""
    );
    println!("| conn | served | requests | qps | p50 us | p95 us | p99 us | p99.9 us |");
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|");

    let mut json_cells: Vec<String> = Vec::new();
    for &c in &connections {
        let mut handles = Vec::new();
        for _ in 0..c {
            let conn_str = conn_str.clone();
            handles.push(tokio::spawn(async move {
                // Connect with a deadline: on the sequential server, connections beyond the
                // first never get accepted, so they time out here and count as not-served.
                let connected = tokio::time::timeout(
                    Duration::from_secs(5),
                    tokio_postgres::connect(&conn_str, NoTls),
                )
                .await;
                let (client, connection) = match connected {
                    Ok(Ok(pair)) => pair,
                    _ => return (false, Vec::<u64>::new()),
                };
                tokio::spawn(async move {
                    let _ = connection.await;
                });
                let mut latencies = Vec::new();
                let end = Instant::now() + Duration::from_secs(duration_secs);
                while Instant::now() < end {
                    let started = Instant::now();
                    match client.simple_query(query).await {
                        Ok(_) => latencies.push(started.elapsed().as_micros() as u64),
                        Err(_) => break,
                    }
                }
                (true, latencies)
            }));
        }

        let mut all_latencies = Vec::new();
        let mut served = 0_usize;
        for handle in handles {
            let (ok, latencies) = handle.await.unwrap_or((false, Vec::new()));
            if ok {
                served += 1;
            }
            all_latencies.extend(latencies);
        }
        all_latencies.sort_unstable();
        // Aggregate throughput: total requests over the per-client measurement window
        // (all served clients overlap for ~duration_secs).
        let qps = all_latencies.len() as f64 / duration_secs as f64;
        let (p50, p95, p99, p999) = (
            percentile(&all_latencies, 0.50),
            percentile(&all_latencies, 0.95),
            percentile(&all_latencies, 0.99),
            percentile(&all_latencies, 0.999),
        );
        println!(
            "| {c} | {served} | {} | {qps:.0} | {p50} | {p95} | {p99} | {p999} |",
            all_latencies.len()
        );
        json_cells.push(format!(
            "{{\"connections\":{c},\"served\":{served},\"requests\":{},\"qps\":{qps:.1},\"p50_us\":{p50},\"p95_us\":{p95},\"p99_us\":{p99},\"p999_us\":{p999}}}",
            all_latencies.len()
        ));
    }
    println!();
    println!(
        "json={{\"kind\":\"p1_m4_concurrent_dispatch_load\",\"mode\":\"{mode}\",\"rows\":{rows},\"duration_secs\":{duration_secs},\"cells\":[{}]}}",
        json_cells.join(",")
    );
    Ok(())
}
