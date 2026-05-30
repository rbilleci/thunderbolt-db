use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use gpu_db_engine::{Engine, RelationalSelectResult};
use gpu_db_protocol::{parse_command, Command, Select};

const VRAM_BYTES: u64 = 24 * 1024 * 1024 * 1024;
const TARGET_TIERS: &[(&str, u64)] = &[
    ("25pct", 6 * 1024 * 1024 * 1024),
    ("50pct", 12 * 1024 * 1024 * 1024),
    ("100pct", 24 * 1024 * 1024 * 1024),
    ("200pct", 48 * 1024 * 1024 * 1024),
];
const CONCURRENCY_TARGETS: &[usize] = &[1, 10, 100, 1000, 10_000];
const RETAINED_BYTES_PER_ORDER_LINE_ROW: u64 = 40;
const GENERATED_BYTES_PER_ORDER_LINE_ROW: u64 = 96;

#[derive(Debug)]
struct Args {
    mode: Mode,
    output_dir: PathBuf,
    rows: usize,
    concurrency: Vec<usize>,
    max_rows: usize,
    chunk_rows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Estimate,
    Run,
    StreamingSelfCheck,
}

#[derive(Debug)]
struct QueryCase {
    name: &'static str,
    sql: String,
}

#[derive(Debug)]
struct QueryMetrics {
    query: String,
    logical_requests: usize,
    result_rows: usize,
    p95_us: u128,
    p99_us: u128,
    throughput_qps: f64,
    h2d_bytes_total: u64,
    d2h_bytes_total: u64,
    kernel_samples: u64,
    kernel_ms: u64,
    cuda_event_samples: u64,
    cuda_event_elapsed_us: u64,
    resident_route_accepted: bool,
    resident_route_zero_h2d: bool,
    resident_bytes: u64,
    correctness_validated: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    fs::create_dir_all(&args.output_dir)?;
    match args.mode {
        Mode::Estimate => write_estimate(&args),
        Mode::Run => run_probe(&args),
        Mode::StreamingSelfCheck => run_streaming_self_check(&args),
    }
}

fn write_estimate(args: &Args) -> Result<(), Box<dyn Error>> {
    let mut report = String::new();
    report.push_str("# P8 CH-benCHmark Residency Tier Estimate\n\n");
    report.push_str("- workload_subset: supported CH-benCHmark-derived analytical subset over `order_line`, `stock`, `customer`, and `orders`\n");
    report.push_str("- supported_queries: COUNT, SUM, AVG, MIN, MAX over single-table filters using supported int4/text predicates\n");
    report.push_str("- non_claim: full CH-benCHmark, joins, transaction mix, and BenchBase compatibility are not claimed\n");
    report.push_str(&format!("- local_vram_bytes: {VRAM_BYTES}\n"));
    report.push_str(&format!("- requested_rows: {}\n", args.rows));
    report.push_str(&format!("- max_safe_run_rows: {}\n", args.max_rows));
    report.push_str("- tiers:\n");

    let mut raw = File::create(args.output_dir.join("estimate.jsonl"))?;
    for (name, target_bytes) in TARGET_TIERS {
        let estimated_rows = target_bytes.div_ceil(RETAINED_BYTES_PER_ORDER_LINE_ROW);
        let generated_bytes = estimated_rows.saturating_mul(GENERATED_BYTES_PER_ORDER_LINE_ROW);
        let retained_bytes = estimated_rows.saturating_mul(RETAINED_BYTES_PER_ORDER_LINE_ROW);
        let wal_bytes = generated_bytes / 2;
        let report_bytes = 2 * 1024 * 1024;
        let safe_for_scheduled_run = estimated_rows <= args.max_rows as u64;
        report.push_str(&format!(
            "  - name: {name}, retained_target_bytes: {target_bytes}, estimated_order_line_rows: {estimated_rows}, generated_table_bytes: {generated_bytes}, wal_log_bytes: {wal_bytes}, report_bytes: {report_bytes}, safe_for_scheduled_run: {safe_for_scheduled_run}\n"
        ));
        writeln!(
            raw,
            "{{\"kind\":\"estimate\",\"tier\":\"{}\",\"retained_target_bytes\":{},\"estimated_rows\":{},\"generated_table_bytes\":{},\"retained_column_bytes\":{},\"wal_log_bytes\":{},\"report_bytes\":{},\"safe_for_scheduled_run\":{}}}",
            name,
            target_bytes,
            estimated_rows,
            generated_bytes,
            retained_bytes,
            wal_bytes,
            report_bytes,
            safe_for_scheduled_run
        )?;
    }

    report.push_str("- concurrency_targets: [1, 10, 100, 1000, 10000]\n");
    report.push_str(
        "- cleanup_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`\n",
    );
    fs::write(args.output_dir.join("estimate.md"), report)?;
    Ok(())
}

fn run_streaming_self_check(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.rows == 0 {
        return Err("--rows must be greater than zero".into());
    }
    if args.chunk_rows == 0 {
        return Err("--chunk-rows must be greater than zero".into());
    }
    let artifact_dir = args.output_dir.join("streaming-order-line");
    if artifact_dir.exists() {
        fs::remove_dir_all(&artifact_dir)?;
    }
    fs::create_dir_all(&artifact_dir)?;

    let mut manifest = BufWriter::new(File::create(artifact_dir.join("manifest.jsonl"))?);
    let mut total_rows = 0usize;
    let mut total_bytes = 0u64;
    let mut chunk_index = 0usize;
    let mut next_row = 1usize;
    while next_row <= args.rows {
        let chunk_start = next_row;
        let chunk_end = args.rows.min(chunk_start + args.chunk_rows - 1);
        let chunk_path = artifact_dir.join(format!("order_line_chunk_{chunk_index:05}.csv"));
        let mut chunk = BufWriter::new(File::create(&chunk_path)?);
        let mut chunk_rows = 0usize;
        for id in chunk_start..=chunk_end {
            write_order_line_row(&mut chunk, id)?;
            chunk_rows += 1;
        }
        chunk.flush()?;
        let chunk_bytes = fs::metadata(&chunk_path)?.len();
        total_rows += chunk_rows;
        total_bytes = total_bytes.saturating_add(chunk_bytes);
        writeln!(
            manifest,
            "{{\"kind\":\"streaming_chunk\",\"chunk_index\":{},\"path\":\"{}\",\"first_row\":{},\"last_row\":{},\"rows\":{},\"bytes\":{}}}",
            chunk_index,
            json_escape(&chunk_path.display().to_string()),
            chunk_start,
            chunk_end,
            chunk_rows,
            chunk_bytes
        )?;
        chunk_index += 1;
        next_row = chunk_end + 1;
    }
    writeln!(
        manifest,
        "{{\"kind\":\"streaming_summary\",\"rows\":{},\"chunks\":{},\"bytes\":{},\"chunk_rows\":{},\"bounded_generator\":true,\"chunked_cache_install_available\":false,\"blocker\":\"missing_relational_resident_cache_chunked_install_api\"}}",
        total_rows,
        chunk_index,
        total_bytes,
        args.chunk_rows
    )?;
    manifest.flush()?;

    let mut report = String::new();
    report.push_str("# P8 CH-benCHmark Streaming Generator Self-Check\n\n");
    report.push_str("- scope: benchmark-only `order_line` artifact generation under `target/`\n");
    report.push_str(&format!("- rows: {}\n", total_rows));
    report.push_str(&format!("- chunk_rows: {}\n", args.chunk_rows));
    report.push_str(&format!("- chunks: {}\n", chunk_index));
    report.push_str(&format!("- generated_bytes: {}\n", total_bytes));
    report.push_str("- bounded_generator: pass\n");
    report.push_str("- chunked_resident_cache_install: blocked\n");
    report.push_str("- blocker: `missing_relational_resident_cache_chunked_install_api`\n\n");
    report.push_str("The self-check writes deterministic generated rows directly to chunk files and never seeds the MVCC engine. It proves the benchmark generator side can be bounded, but the current engine residency path still installs a snapshot through `RelationalResidencySnapshot { resident_rows: Vec<Vec<SqlValue>>, ... }` and a single retained device-memory payload.\n");
    fs::write(artifact_dir.join("self-check.md"), report)?;

    if total_rows != args.rows {
        return Err(format!(
            "streaming generator wrote {total_rows} rows, expected {}",
            args.rows
        )
        .into());
    }
    if chunk_index == 0 || total_bytes == 0 {
        return Err("streaming generator produced no chunks".into());
    }
    Ok(())
}

fn write_order_line_row<W: Write>(writer: &mut W, id: usize) -> Result<(), Box<dyn Error>> {
    let bucket = (id % 10) as i32;
    let item = (id % 100_000) + 1;
    let qty = ((id % 50) + 1) as i32;
    let amount = ((id * 17) % 100_000) as i32;
    let dist = if id.is_multiple_of(2) {
        "alpha"
    } else {
        "omega"
    };
    writeln!(writer, "{id},{item},{qty},{amount},{dist}{bucket}")?;
    Ok(())
}

fn run_probe(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.rows == 0 {
        return Err("--rows must be greater than zero".into());
    }
    if args.rows > args.max_rows {
        return Err(format!(
            "--rows={} exceeds scheduled-run guardrail max_rows={}",
            args.rows, args.max_rows
        )
        .into());
    }

    let run_id = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let mut cpu = seed_engine(args.rows)?;
    let mut gpu = seed_engine(args.rows)?;
    let snapshot = gpu.populate_relational_residency_snapshot("order_line")?;
    let queries = query_cases(args.rows)?;
    let mut raw = File::create(args.output_dir.join("metrics.jsonl"))?;
    let mut markdown = String::new();
    markdown.push_str("# P8 CH-benCHmark Residency Baseline Probe\n\n");
    markdown.push_str(&format!("- run_id: {run_id}\n"));
    markdown.push_str(&format!("- row_count: {}\n", args.rows));
    markdown.push_str("- resident_table: order_line\n");
    markdown.push_str(&format!("- resident_bytes: {}\n", snapshot.resident_bytes));
    markdown.push_str("- configured_production_tiers: 25pct/50pct/100pct/200pct of 24GiB VRAM\n");
    markdown.push_str("- attempted_tier: calibration\n");
    markdown.push_str(
        "- cleanup_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`\n\n",
    );

    for logical_requests in &args.concurrency {
        markdown.push_str(&format!("## logical_requests_{logical_requests}\n"));
        for case in &queries {
            let expected = cpu.execute_relational_select(&select(&case.sql)?)?;
            let metrics = run_query_case(
                &mut gpu,
                case,
                *logical_requests,
                &expected,
                snapshot.resident_bytes,
            )?;
            writeln!(
                raw,
                "{{\"kind\":\"metric\",\"query\":\"{}\",\"logical_requests\":{},\"result_rows\":{},\"p95_us\":{},\"p99_us\":{},\"throughput_qps\":{:.3},\"h2d_bytes_total\":{},\"d2h_bytes_total\":{},\"kernel_samples\":{},\"kernel_ms\":{},\"cuda_event_samples\":{},\"cuda_event_elapsed_us\":{},\"resident_route_accepted\":{},\"resident_route_zero_h2d\":{},\"resident_bytes\":{},\"correctness_validated\":{}}}",
                metrics.query,
                metrics.logical_requests,
                metrics.result_rows,
                metrics.p95_us,
                metrics.p99_us,
                metrics.throughput_qps,
                metrics.h2d_bytes_total,
                metrics.d2h_bytes_total,
                metrics.kernel_samples,
                metrics.kernel_ms,
                metrics.cuda_event_samples,
                metrics.cuda_event_elapsed_us,
                metrics.resident_route_accepted,
                metrics.resident_route_zero_h2d,
                metrics.resident_bytes,
                metrics.correctness_validated
            )?;
            markdown.push_str(&format!(
                "- {}: pass p95_us={} p99_us={} throughput_qps={:.2} h2d_bytes_total={} d2h_bytes_total={} cuda_event_samples={} resident_zero_h2d={} rows={}\n",
                metrics.query,
                metrics.p95_us,
                metrics.p99_us,
                metrics.throughput_qps,
                metrics.h2d_bytes_total,
                metrics.d2h_bytes_total,
                metrics.cuda_event_samples,
                metrics.resident_route_zero_h2d,
                metrics.result_rows
            ));
        }
        markdown.push('\n');
    }

    let count_select = select("SELECT COUNT(*) FROM order_line")?;
    gpu.mark_gpu_memory_pressured(0);
    let pressured = gpu.plan_relational_resident_route(&count_select);
    writeln!(
        raw,
        "{{\"kind\":\"over_residency_probe\",\"memory_pressure_route_accepted\":{},\"reason\":\"{}\",\"cache_state\":\"{}\"}}",
        pressured.accepted,
        json_escape(&pressured.reason),
        json_escape(&pressured.cache_state)
    )?;
    markdown.push_str("## over_residency_probe\n");
    markdown.push_str(&format!(
        "- memory_pressure_route: {} reason={} cache_state={}\n",
        if pressured.accepted { "fail" } else { "pass" },
        pressured.reason,
        pressured.cache_state
    ));
    markdown.push_str("\n## decision\n");
    markdown.push_str("The scheduled-worker baseline is a calibration tier because the 25% VRAM target estimates hundreds of millions of retained rows and is not defensible for one short cron slice. The harness preserves the 25/50/100/200% tier plan and shows the next bottleneck is a streaming/on-disk workload generator plus a longer operator-approved run window before attempting the 6GiB retained tier.\n");

    fs::write(args.output_dir.join("baseline.md"), markdown)?;
    Ok(())
}

fn run_query_case(
    gpu: &mut Engine,
    case: &QueryCase,
    logical_requests: usize,
    expected: &RelationalSelectResult,
    resident_bytes: u64,
) -> Result<QueryMetrics, Box<dyn Error>> {
    let select = select(&case.sql)?;
    let mut latencies = Vec::with_capacity(logical_requests);
    let before = gpu.metrics().snapshot();
    let started = Instant::now();
    let mut result_rows = 0usize;
    for _ in 0..logical_requests {
        let query_started = Instant::now();
        let result = gpu.execute_relational_select(&select)?;
        latencies.push(query_started.elapsed());
        if result.columns != expected.columns || result.rows != expected.rows {
            return Err(format!("{} CPU/resident results diverged", case.name).into());
        }
        result_rows = result.rows.len();
    }
    let elapsed = started.elapsed();
    let after = gpu.metrics().snapshot();
    let decision = gpu.plan_relational_resident_route(&select);
    Ok(QueryMetrics {
        query: case.name.to_string(),
        logical_requests,
        result_rows,
        p95_us: percentile(&latencies, 95),
        p99_us: percentile(&latencies, 99),
        throughput_qps: qps(logical_requests, elapsed),
        h2d_bytes_total: after.h2d_bytes_total.saturating_sub(before.h2d_bytes_total),
        d2h_bytes_total: after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total),
        kernel_samples: after
            .kernel_exec_samples
            .saturating_sub(before.kernel_exec_samples),
        kernel_ms: after
            .kernel_exec_total_ms
            .saturating_sub(before.kernel_exec_total_ms),
        cuda_event_samples: after
            .kernel_event_timing_samples
            .saturating_sub(before.kernel_event_timing_samples),
        cuda_event_elapsed_us: after
            .kernel_event_elapsed_total_us
            .saturating_sub(before.kernel_event_elapsed_total_us),
        resident_route_accepted: decision.accepted,
        resident_route_zero_h2d: decision.last_execution_h2d_bytes == Some(0)
            || decision.h2d_bytes_if_resident == 0,
        resident_bytes,
        correctness_validated: true,
    })
}

fn seed_engine(row_count: usize) -> Result<Engine, Box<dyn Error>> {
    let mut engine = Engine::new_local();
    engine.execute_text(
        1,
        "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
    )?;
    engine.execute_text(
        2,
        "CREATE TABLE stock (s_i_id INT, s_quantity INT, s_dist_01 TEXT)",
    )?;
    engine.execute_text(
        3,
        "CREATE TABLE customer (c_id INT, c_w_id INT, c_balance INT, c_last TEXT)",
    )?;
    engine.execute_text(
        4,
        "CREATE TABLE orders (o_id INT, o_c_id INT, o_entry_d INT, o_carrier_id INT)",
    )?;
    for id in 1..=row_count {
        let bucket = (id % 10) as i32;
        let item = (id % 100_000) + 1;
        let qty = ((id % 50) + 1) as i32;
        let amount = ((id * 17) % 100_000) as i32;
        let dist = if id.is_multiple_of(2) {
            "alpha"
        } else {
            "omega"
        };
        engine.execute_text(
            (id + 10) as u64,
            &format!(
                "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES ({id}, {item}, {qty}, {amount}, '{dist}{bucket}')"
            ),
        )?;
    }
    for id in 1..=row_count.min(1_000) {
        engine.execute_text(
            (row_count + id + 20) as u64,
            &format!(
                "INSERT INTO stock (s_i_id, s_quantity, s_dist_01) VALUES ({id}, {}, 'stock{}')",
                (id % 200) + 1,
                id % 10
            ),
        )?;
        engine.execute_text(
            (row_count + id + 30_000) as u64,
            &format!(
                "INSERT INTO customer (c_id, c_w_id, c_balance, c_last) VALUES ({id}, {}, {}, 'last{}')",
                (id % 16) + 1,
                (id % 10_000) as i32,
                id % 20
            ),
        )?;
        engine.execute_text(
            (row_count + id + 60_000) as u64,
            &format!(
                "INSERT INTO orders (o_id, o_c_id, o_entry_d, o_carrier_id) VALUES ({id}, {id}, {}, {})",
                20260530,
                (id % 10) + 1
            ),
        )?;
    }
    Ok(engine)
}

fn query_cases(row_count: usize) -> Result<Vec<QueryCase>, Box<dyn Error>> {
    let lower = (row_count / 4).max(1);
    Ok(vec![
        QueryCase {
            name: "order_line_count_all",
            sql: "SELECT COUNT(*) FROM order_line".to_string(),
        },
        QueryCase {
            name: "order_line_sum_amount",
            sql: "SELECT SUM(ol_amount) FROM order_line".to_string(),
        },
        QueryCase {
            name: "order_line_avg_quantity_between",
            sql: "SELECT AVG(ol_quantity) FROM order_line WHERE ol_quantity BETWEEN 10 AND 40"
                .to_string(),
        },
        QueryCase {
            name: "order_line_max_amount_filter",
            sql: format!("SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= {lower}"),
        },
    ])
}

fn select(sql: &str) -> Result<Select, Box<dyn Error>> {
    match parse_command(sql)? {
        Command::Select(select) => Ok(select),
        _ => Err(format!("not a SELECT: {sql}").into()),
    }
}

fn percentile(latencies: &[Duration], percentile: usize) -> u128 {
    if latencies.is_empty() {
        return 0;
    }
    let mut values = latencies
        .iter()
        .map(Duration::as_micros)
        .collect::<Vec<_>>();
    values.sort_unstable();
    let rank = (percentile.clamp(1, 100) * values.len()).div_ceil(100);
    values[rank.saturating_sub(1)]
}

fn qps(count: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs == 0.0 {
        return count as f64;
    }
    count as f64 / secs
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut mode = None;
    let mut output_dir = PathBuf::from("target/p8-ch-benchmark-residency");
    let mut rows = env::var("GPU_DB_CH_BENCH_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(512);
    let mut concurrency = env::var("GPU_DB_CH_BENCH_CONCURRENCY")
        .ok()
        .map(|value| parse_csv_usize(&value))
        .transpose()?
        .unwrap_or_else(|| vec![1, 10]);
    let mut chunk_rows = env::var("GPU_DB_CH_BENCH_CHUNK_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
    let max_rows = env::var("GPU_DB_CH_BENCH_MAX_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10_000);

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--estimate" => mode = Some(Mode::Estimate),
            "--run" => mode = Some(Mode::Run),
            "--streaming-self-check" => mode = Some(Mode::StreamingSelfCheck),
            "--output-dir" => {
                output_dir = PathBuf::from(args.next().ok_or("--output-dir needs a value")?);
            }
            "--rows" => {
                rows = args
                    .next()
                    .ok_or("--rows needs a value")?
                    .parse()
                    .map_err(|_| "--rows must be an integer")?;
            }
            "--concurrency" => {
                concurrency = parse_csv_usize(&args.next().ok_or("--concurrency needs a value")?)?;
            }
            "--chunk-rows" => {
                chunk_rows = args
                    .next()
                    .ok_or("--chunk-rows needs a value")?
                    .parse()
                    .map_err(|_| "--chunk-rows must be an integer")?;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    if concurrency.is_empty() {
        return Err("at least one logical concurrency target is required".into());
    }
    if !concurrency
        .iter()
        .all(|target| CONCURRENCY_TARGETS.contains(target))
    {
        return Err("concurrency must be selected from 1,10,100,1000,10000".into());
    }
    Ok(Args {
        mode: mode.unwrap_or(Mode::Estimate),
        output_dir,
        rows,
        concurrency,
        max_rows,
        chunk_rows,
    })
}

fn parse_csv_usize(value: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid integer: {part}").into())
        })
        .collect()
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
