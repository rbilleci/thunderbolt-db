use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

use gpu_db_engine::{Engine, ExecuteError, RelationalSelectResult, RelationalSqlGpuBridgeReport};
use gpu_db_protocol::{parse_command, Command, Select};

#[derive(Debug)]
struct WorkloadReport {
    name: &'static str,
    query_count: usize,
    cpu_elapsed: Duration,
    gpu_elapsed: Duration,
    cpu_latency_us: LatencySummary,
    gpu_latency_us: LatencySummary,
    result_rows: usize,
    correctness_validated: bool,
    bridge: RelationalSqlGpuBridgeReport,
    h2d_bytes_total: u64,
    d2h_bytes_total: u64,
    kernel_exec_samples: u64,
    kernel_exec_total_ms: u64,
}

#[derive(Debug, Clone, Copy)]
struct LatencySummary {
    p50_us: u128,
    p95_us: u128,
    max_us: u128,
}

fn main() -> Result<(), Box<dyn Error>> {
    let row_count = env::var("GPU_DB_BENCH_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1_000);
    let lookup_count = env::var("GPU_DB_BENCH_LOOKUPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
    let device_info = env::var("GPU_DB_BENCH_DEVICE_INFO")
        .unwrap_or_else(|_| "not reported by runner".to_string());

    let app_queries = app_lookup_queries(row_count, lookup_count)?;
    let app_batched_queries = vec![app_batched_lookup_query(row_count, lookup_count)?];
    let analytic_queries = vec![select("SELECT * FROM events")?];
    let range_queries = vec![select(
        "SELECT id, amount FROM events WHERE amount >= 900 ORDER BY amount DESC LIMIT 25",
    )?];
    let conjunctive_queries = vec![select(
        "SELECT id, amount FROM events WHERE amount >= 900 AND category = 'even' ORDER BY amount DESC LIMIT 25",
    )?];
    let disjunctive_queries = vec![select(
        "SELECT id, amount FROM events WHERE amount >= 990 OR category = 'odd' ORDER BY amount DESC LIMIT 25",
    )?];

    let app = run_workload("app_indexed_point_lookup", row_count, &app_queries)?;
    let app_batched = run_workload("app_batched_or_lookup", row_count, &app_batched_queries)?;
    let analytic = run_workload("analytic_full_table_scan", row_count, &analytic_queries)?;
    let range = run_workload("analytic_range_filter", row_count, &range_queries)?;
    let conjunctive = run_workload(
        "analytic_conjunctive_filter",
        row_count,
        &conjunctive_queries,
    )?;
    let disjunctive = run_workload(
        "analytic_disjunctive_filter",
        row_count,
        &disjunctive_queries,
    )?;

    println!("# P7 Relational Workload Benchmark");
    println!();
    println!("- dataset_rows: {row_count}");
    println!("- concurrency: 1");
    println!("- device_info: {device_info}");
    println!("- cuda_driver_probe_runtime_cache: per_engine");
    println!();
    print_workload(&app);
    println!();
    print_workload(&app_batched);
    println!();
    print_workload(&analytic);
    println!();
    print_workload(&range);
    println!();
    print_workload(&conjunctive);
    println!();
    print_workload(&disjunctive);
    println!();
    print_decision(
        &app,
        &app_batched,
        &analytic,
        &range,
        &conjunctive,
        &disjunctive,
    );

    Ok(())
}

fn print_workload(report: &WorkloadReport) {
    println!("## {}", report.name);
    println!("- queries: {}", report.query_count);
    println!("- result_rows: {}", report.result_rows);
    println!("- cpu_total_us: {}", report.cpu_elapsed.as_micros());
    println!("- gpu_probe_total_us: {}", report.gpu_elapsed.as_micros());
    println!(
        "- cpu_qps: {:.2}",
        qps(report.query_count, report.cpu_elapsed)
    );
    println!(
        "- gpu_probe_qps: {:.2}",
        qps(report.query_count, report.gpu_elapsed)
    );
    println!("- cpu_latency_p50_us: {}", report.cpu_latency_us.p50_us);
    println!("- cpu_latency_p95_us: {}", report.cpu_latency_us.p95_us);
    println!("- cpu_latency_max_us: {}", report.cpu_latency_us.max_us);
    println!(
        "- gpu_probe_latency_p50_us: {}",
        report.gpu_latency_us.p50_us
    );
    println!(
        "- gpu_probe_latency_p95_us: {}",
        report.gpu_latency_us.p95_us
    );
    println!(
        "- gpu_probe_latency_max_us: {}",
        report.gpu_latency_us.max_us
    );
    println!(
        "- gpu_probe_vs_cpu_total_ratio: {:.3}",
        elapsed_ratio(report.gpu_elapsed, report.cpu_elapsed)
    );
    println!(
        "- gpu_executed_rate_permyriad: {}",
        report.bridge.gpu_executed_permyriad
    );
    println!(
        "- cpu_fallback_rate_permyriad: {}",
        report.bridge.cpu_fallback_permyriad
    );
    println!("- h2d_bytes_total: {}", report.h2d_bytes_total);
    println!(
        "- h2d_bytes_per_query: {:.2}",
        bytes_per(report.h2d_bytes_total, report.query_count)
    );
    println!("- d2h_bytes_total: {}", report.d2h_bytes_total);
    println!(
        "- d2h_bytes_per_result_row: {:.2}",
        bytes_per(report.d2h_bytes_total, report.result_rows)
    );
    println!("- kernel_exec_samples: {}", report.kernel_exec_samples);
    println!("- kernel_exec_total_ms: {}", report.kernel_exec_total_ms);
    println!("- correctness_validated: {}", report.correctness_validated);
}

fn print_decision(
    app: &WorkloadReport,
    app_batched: &WorkloadReport,
    analytic: &WorkloadReport,
    range: &WorkloadReport,
    conjunctive: &WorkloadReport,
    disjunctive: &WorkloadReport,
) {
    if analytic.bridge.gpu_executed_count > 0 && analytic.gpu_elapsed < analytic.cpu_elapsed {
        println!(
            "decision: GPU probe is faster for the analytical scan in this run; keep prioritizing SQL predicate/order/projection pushdown so more relational shapes can use the same path."
        );
        return;
    }

    if app_batched.bridge.gpu_executed_count > 0 && app_batched.gpu_elapsed < app.gpu_elapsed {
        println!(
            "decision: batching lookup predicates into one supported OR query reduces GPU probe latency versus repeated point lookups, but analytical scans still do not beat CPU; prioritize batching plus transfer layout before making broad performance claims."
        );
        return;
    }

    if analytic.bridge.gpu_executed_count > 0 {
        println!(
            "decision: analytical scans reach GPU execution with SQL-level transfer and timing telemetry but do not yet beat the CPU baseline in this run; compare batched lookup latency against repeated point lookups, then prioritize transfer layout, batching, and driver-level timing refinement before making broad performance claims."
        );
        return;
    }

    if app.bridge.cpu_fallback_count > 0
        || app_batched.bridge.cpu_fallback_count > 0
        || analytic.bridge.cpu_fallback_count > 0
        || range.bridge.cpu_fallback_count > 0
        || conjunctive.bridge.cpu_fallback_count > 0
        || disjunctive.bridge.cpu_fallback_count > 0
    {
        println!(
            "decision: current relational workloads still fall back for important SQL shapes; prioritize SQL-to-GPU bridge expansion before claiming workload-level GPU advantage."
        );
        return;
    }

    println!(
        "decision: no GPU advantage was demonstrated; keep P7 claims limited to correctness and routing evidence until a measured workload beats CPU."
    );
}

fn run_workload(
    name: &'static str,
    row_count: usize,
    queries: &[Select],
) -> Result<WorkloadReport, Box<dyn Error>> {
    let mut cpu = seeded_engine(row_count)?;
    let mut gpu = seeded_engine(row_count)?;

    let cpu_start = Instant::now();
    let (cpu_results, cpu_latencies) =
        execute_timed(queries, |query| cpu.execute_relational_select(query))?;
    let cpu_elapsed = cpu_start.elapsed();

    let gpu_start = Instant::now();
    let (gpu_results, gpu_latencies) = execute_timed(queries, |query| {
        gpu.execute_relational_select_with_cuda_driver_probe(query)
    })?;
    let gpu_elapsed = gpu_start.elapsed();

    let correctness_validated = same_sql_results(&cpu_results, &gpu_results);
    if !correctness_validated {
        return Err(format!("{name} CPU/GPU probe results diverged").into());
    }

    let result_rows = gpu_results.iter().map(|result| result.rows.len()).sum();
    let bridge = RelationalSqlGpuBridgeReport::from_results(&gpu_results);
    let metrics = gpu.metrics().snapshot();

    Ok(WorkloadReport {
        name,
        query_count: queries.len(),
        cpu_elapsed,
        gpu_elapsed,
        cpu_latency_us: summarize_latencies(&cpu_latencies),
        gpu_latency_us: summarize_latencies(&gpu_latencies),
        result_rows,
        correctness_validated,
        bridge,
        h2d_bytes_total: metrics.h2d_bytes_total,
        d2h_bytes_total: metrics.d2h_bytes_total,
        kernel_exec_samples: metrics.kernel_exec_samples,
        kernel_exec_total_ms: metrics.kernel_exec_total_ms,
    })
}

fn execute_timed<F>(
    queries: &[Select],
    mut execute: F,
) -> Result<(Vec<RelationalSelectResult>, Vec<Duration>), Box<dyn Error>>
where
    F: FnMut(&Select) -> Result<RelationalSelectResult, ExecuteError>,
{
    let mut results = Vec::with_capacity(queries.len());
    let mut latencies = Vec::with_capacity(queries.len());
    for query in queries {
        let start = Instant::now();
        let result = execute(query)?;
        latencies.push(start.elapsed());
        results.push(result);
    }
    Ok((results, latencies))
}

fn qps(query_count: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        return 0.0;
    }
    query_count as f64 / elapsed.as_secs_f64()
}

fn elapsed_ratio(numerator: Duration, denominator: Duration) -> f64 {
    if denominator.is_zero() {
        return 0.0;
    }
    numerator.as_secs_f64() / denominator.as_secs_f64()
}

fn bytes_per(bytes: u64, count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    bytes as f64 / count as f64
}

fn summarize_latencies(latencies: &[Duration]) -> LatencySummary {
    if latencies.is_empty() {
        return LatencySummary {
            p50_us: 0,
            p95_us: 0,
            max_us: 0,
        };
    }
    let mut micros = latencies
        .iter()
        .map(Duration::as_micros)
        .collect::<Vec<_>>();
    micros.sort_unstable();
    LatencySummary {
        p50_us: percentile_nearest_rank(&micros, 50),
        p95_us: percentile_nearest_rank(&micros, 95),
        max_us: *micros.last().unwrap_or(&0),
    }
}

fn percentile_nearest_rank(sorted_values: &[u128], percentile: usize) -> u128 {
    if sorted_values.is_empty() {
        return 0;
    }
    let clamped = percentile.clamp(1, 100);
    let rank = (clamped * sorted_values.len()).div_ceil(100);
    sorted_values[rank.saturating_sub(1)]
}

fn seeded_engine(row_count: usize) -> Result<Engine, Box<dyn Error>> {
    let mut engine = Engine::new_local();
    engine.execute_text(
        1,
        "CREATE TABLE events (id INT, account TEXT, amount INT, category TEXT)",
    )?;
    for id in 1..=row_count {
        let account = format!("acct{}", id % 64);
        let category = if id % 2 == 0 { "even" } else { "odd" };
        let amount = (id % 10_000) as i32;
        let sql = format!(
            "INSERT INTO events (id, account, amount, category) VALUES ({id}, '{account}', {amount}, '{category}')"
        );
        engine.execute_text((id + 1) as u64, &sql)?;
    }
    Ok(engine)
}

fn app_lookup_queries(
    row_count: usize,
    lookup_count: usize,
) -> Result<Vec<Select>, Box<dyn Error>> {
    let mut queries = Vec::with_capacity(lookup_count);
    for i in 0..lookup_count {
        let id = (i * 37 % row_count) + 1;
        queries.push(select(&format!("SELECT * FROM events WHERE id = {id}"))?);
    }
    Ok(queries)
}

fn app_batched_lookup_query(
    row_count: usize,
    lookup_count: usize,
) -> Result<Select, Box<dyn Error>> {
    let mut predicates = Vec::with_capacity(lookup_count);
    for i in 0..lookup_count {
        let id = (i * 37 % row_count) + 1;
        predicates.push(format!("id = {id}"));
    }
    select(&format!(
        "SELECT * FROM events WHERE {} ORDER BY id ASC",
        predicates.join(" OR ")
    ))
}

fn select(sql: &str) -> Result<Select, Box<dyn Error>> {
    match parse_command(sql)? {
        Command::Select(select) => Ok(select),
        _ => Err(format!("not a SELECT: {sql}").into()),
    }
}

fn same_sql_results(left: &[RelationalSelectResult], right: &[RelationalSelectResult]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.columns == right.columns && left.rows == right.rows)
}
