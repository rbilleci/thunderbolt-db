//! GPU-retained multi-query benchmark — throughput + tail latency per query type, over the
//! wire, through the new server architecture (P1-M4 concurrent / P1-M5 async dispatch).
//!
//! This is the comparable-to-M0 benchmark: the old `phase0-m0-baseline` measured a query mix
//! (count / projection / multi-column / mixed int+text) with qps + p50/p95/p99 on the
//! GPU-retained path through the *old* owner-thread server. This runs the same shape of mix
//! on the GPU-retained path through the *new* façade server.
//!
//! Setup (on an owned `&mut Engine`, before serving): build `order_line` (int4 + text),
//! INSERT rows, `populate_relational_residency_snapshot` to make it GPU-resident, then a
//! **GPU-acceptance gate** — each candidate query is run through
//! `execute_relational_select_with_resident_route`, which returns `Err` if the resident route
//! would reject (CPU fallback); only queries it accepts are benchmarked, so every reported
//! number is genuinely the GPU-retained path. The warmed engine is then served via
//! `serve_async_with_engine` (async, default) or `serve_with_engine` (concurrent).
//!
//! Env: GPU_DB_BENCH_ROWS (50000), GPU_DB_BENCH_CONNECTIONS (1,8,64),
//! GPU_DB_BENCH_DURATION_SECS (3), GPU_DB_BENCH_MODE (async|concurrent), GPU_DB_BENCH_PERMITS
//! (256). Requires a local GPU; aborts cleanly if `order_line` does not become resident.

use std::env;
use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpu_db_engine::Engine;
use gpu_db_facade::SharedEngine;
use gpu_db_protocol::{parse_command, Command};
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

/// One (query, concurrency) cell: spawn `c` clients each looping `sql` for `duration`,
/// return (qps, p50, p95, p99, p999) in µs.
async fn bench_cell(
    conn_str: &str,
    sql: &str,
    c: usize,
    duration_secs: u64,
) -> (f64, u64, u64, u64, u64) {
    let mut handles = Vec::new();
    for _ in 0..c {
        let conn_str = conn_str.to_string();
        let sql = sql.to_string();
        handles.push(tokio::spawn(async move {
            let connected = tokio::time::timeout(
                Duration::from_secs(10),
                tokio_postgres::connect(&conn_str, NoTls),
            )
            .await;
            let (client, connection) = match connected {
                Ok(Ok(pair)) => pair,
                _ => return Vec::<u64>::new(),
            };
            tokio::spawn(async move {
                let _ = connection.await;
            });
            let mut latencies = Vec::new();
            let end = Instant::now() + Duration::from_secs(duration_secs);
            while Instant::now() < end {
                let started = Instant::now();
                match client.simple_query(&sql).await {
                    Ok(_) => latencies.push(started.elapsed().as_micros() as u64),
                    Err(_) => break,
                }
            }
            latencies
        }));
    }
    let mut all = Vec::new();
    for handle in handles {
        all.extend(handle.await.unwrap_or_default());
    }
    all.sort_unstable();
    let qps = all.len() as f64 / duration_secs as f64;
    (
        qps,
        percentile(&all, 0.50),
        percentile(&all, 0.95),
        percentile(&all, 0.99),
        percentile(&all, 0.999),
    )
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let rows: usize = env_parse("GPU_DB_BENCH_ROWS", 50_000);
    let connections: Vec<usize> = env::var("GPU_DB_BENCH_CONNECTIONS")
        .unwrap_or_else(|_| "1,8,64".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let duration_secs: u64 = env_parse("GPU_DB_BENCH_DURATION_SECS", 3);
    let mode = env::var("GPU_DB_BENCH_MODE").unwrap_or_else(|_| "async".to_string());
    let permits: usize = env_parse("GPU_DB_BENCH_PERMITS", 256);

    // A lookup key that matches an existing row (point-lookup shapes).
    let lookup = rows / 2;
    let candidates: Vec<(&str, String)> = vec![
        ("count_all", "SELECT COUNT(*) FROM order_line".to_string()),
        (
            "equality_count",
            "SELECT COUNT(*) FROM order_line WHERE ol_i_id = 3".to_string(),
        ),
        (
            "equality_projection",
            format!("SELECT ol_amount FROM order_line WHERE ol_o_id = {lookup}"),
        ),
        (
            "multi_col_projection",
            format!(
                "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = {lookup}"
            ),
        ),
        (
            "mixed_int_text",
            format!("SELECT ol_o_id, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = {lookup}"),
        ),
    ];

    // ---- Build + warm the engine to GPU residency (owned &mut, before serving). ----
    let mut engine = Engine::new_local();
    engine.execute_text(
        1,
        "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
    )?;
    let mut txn = 2_u64;
    let mut start = 0;
    while start < rows {
        let endi = (start + 2000).min(rows);
        let mut stmt = String::from(
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES ",
        );
        for j in start..endi {
            if j > start {
                stmt.push(',');
            }
            stmt.push_str(&format!(
                "({}, {}, {}, {}, 'dist-{}')",
                j,
                j % 8,
                j % 50,
                (j * 7) % 1000,
                j % 10
            ));
        }
        engine.execute_text(txn, &stmt)?;
        txn += 1;
        start = endi;
    }
    let snapshot = engine.populate_relational_residency_snapshot("order_line")?;
    if snapshot.device_memory_proof.is_none() {
        eprintln!(
            "order_line did NOT become GPU-resident (no local GPU?). This benchmark is \
             GPU-retained only; aborting."
        );
        return Ok(());
    }
    println!(
        "gpu_retained_query_mix: rows={rows} resident_on_gpu=true generation={} mode={mode} \
         duration={duration_secs}s permits={permits}",
        snapshot.generation
    );

    // ---- GPU-acceptance gate: keep only queries the resident route accepts. ----
    let mut gpu_queries: Vec<(&str, String)> = Vec::new();
    for (name, sql) in &candidates {
        let Command::Select(select) = parse_command(sql).map_err(|e| e.to_string())? else {
            eprintln!("  skip {name}: not a SELECT");
            continue;
        };
        match engine.execute_relational_select_with_resident_route(&select) {
            Ok(result) => {
                println!("  GPU-accepted: {name:<22} ({} row(s))", result.rows.len());
                gpu_queries.push((name, sql.clone()));
            }
            Err(err) => eprintln!("  CPU-fallback (excluded): {name} -> {err}"),
        }
    }
    if gpu_queries.is_empty() {
        return Err("no candidate query took the GPU resident route".into());
    }

    // ---- Serve the warmed (GPU-resident) engine. ----
    let shared = Arc::new(SharedEngine::from_engine(engine));
    let port;
    if mode == "concurrent" {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        port = listener.local_addr()?.port();
        let shared = Arc::clone(&shared);
        std::thread::spawn(move || {
            let _ = gpu_db_server::serve_with_engine(listener, shared);
        });
    } else {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        port = listener.local_addr()?.port();
        let shared = Arc::clone(&shared);
        tokio::spawn(async move {
            let _ = gpu_db_server::serve_async_with_engine(listener, shared, permits).await;
        });
    }
    let conn_str = format!("host=127.0.0.1 port={port} user=postgres dbname=postgres");

    // ---- Benchmark each GPU-retained query across the concurrency sweep. ----
    println!();
    println!("| query | conn | p50 us | p95 us | p99 us | p99.9 us | qps |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    let mut json_rows: Vec<String> = Vec::new();
    for (name, sql) in &gpu_queries {
        for &c in &connections {
            let (qps, p50, p95, p99, p999) = bench_cell(&conn_str, sql, c, duration_secs).await;
            println!("| {name} | {c} | {p50} | {p95} | {p99} | {p999} | {qps:.0} |");
            json_rows.push(format!(
                "{{\"query\":\"{name}\",\"conn\":{c},\"p50_us\":{p50},\"p95_us\":{p95},\"p99_us\":{p99},\"p999_us\":{p999},\"qps\":{qps:.1}}}"
            ));
        }
    }
    println!();
    println!(
        "json={{\"kind\":\"gpu_retained_query_mix\",\"rows\":{rows},\"mode\":\"{mode}\",\"duration_secs\":{duration_secs},\"cells\":[{}]}}",
        json_rows.join(",")
    );
    Ok(())
}
