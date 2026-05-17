use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

use gpu_db_engine::{Engine, RelationalSelectResult};
use gpu_db_protocol::{parse_command, Command, Select};

#[derive(Debug)]
struct ProbeReport {
    name: &'static str,
    elapsed: Duration,
    result_rows: usize,
    planned_target: String,
    executed_target: String,
    access_path: String,
    sql_fallback: bool,
    fallback_reason: String,
    h2d_bytes: u64,
    d2h_bytes: u64,
    kernel_exec_samples: u64,
    kernel_exec_total_ms: u64,
    correctness_validated: bool,
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
    if row_count == 0 {
        return Err("GPU_DB_BENCH_ROWS must be greater than zero".into());
    }
    if lookup_count == 0 {
        return Err("GPU_DB_BENCH_LOOKUPS must be greater than zero".into());
    }

    let query = app_batched_lookup_query(row_count, lookup_count)?;
    let mutation_query = select(&format!(
        "SELECT * FROM events WHERE id = {}",
        row_count + 1
    ))?;

    let mut cpu = seeded_engine(row_count)?;
    let mut gpu = seeded_engine(row_count)?;

    let cpu_result = cpu.execute_relational_select(&query)?;
    let cold_probe = timed_probe(&mut gpu, &query, &cpu_result, "cold_per_query_h2d_probe")?;
    let warm_runtime_probe = timed_probe(
        &mut gpu,
        &query,
        &cpu_result,
        "warm_runtime_per_query_h2d_probe",
    )?;

    let new_id = row_count + 1;
    let insert_sql = format!(
        "INSERT INTO events (id, account, amount, category) VALUES ({new_id}, 'acct_refresh', 7777, 'refresh')"
    );
    cpu.execute_text((row_count + 2) as u64, &insert_sql)?;
    gpu.execute_text((row_count + 2) as u64, &insert_sql)?;
    let mutation_cpu_result = cpu.execute_relational_select(&mutation_query)?;
    let mutation_probe = timed_probe(
        &mut gpu,
        &mutation_query,
        &mutation_cpu_result,
        "post_mutation_per_query_h2d_probe",
    )?;

    println!("# P7 GPU Residency Baseline");
    println!();
    println!("- dataset_rows: {row_count}");
    println!("- lookup_count: {lookup_count}");
    println!("- concurrency: 1");
    println!("- device_info: {device_info}");
    println!("- current_data_residency_model: per_query_h2d_probe");
    println!("- warm_resident_execution_supported: false");
    println!("- resident_bytes_current: 0");
    println!("- resident_refresh_supported: false");
    println!("- memory_pressure_fallback_supported: false");
    println!("- correctness_oracle: CPU relational engine");
    println!();
    print_probe(&cold_probe);
    println!();
    print_probe(&warm_runtime_probe);
    println!();
    print_probe(&mutation_probe);
    println!();
    println!(
        "decision: current P7 evidence measures cached CUDA runtime plus per-query H2D transfer, not GPU-resident table data. Do not claim warm-resident performance until the engine implements MVCC/WAL-safe resident invalidation or refresh, resident-byte accounting, memory-pressure fallback, and a mutation refresh-cost benchmark."
    );

    Ok(())
}

fn timed_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result = engine.execute_relational_select_with_cuda_driver_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} CPU/GPU probe results diverged").into());
    }
    Ok(ProbeReport {
        name,
        elapsed,
        result_rows: result.rows.len(),
        planned_target: format!("{:?}", result.planned_target),
        executed_target: format!("{:?}", result.executed_target),
        access_path: format!("{:?}", result.access_path),
        sql_fallback: result.fallback_reason.is_some(),
        fallback_reason: result
            .fallback_reason
            .as_ref()
            .map(|reason| format!("{reason:?}"))
            .unwrap_or_else(|| "None".to_string()),
        h2d_bytes: after.h2d_bytes_total - before.h2d_bytes_total,
        d2h_bytes: after.d2h_bytes_total - before.d2h_bytes_total,
        kernel_exec_samples: after.kernel_exec_samples - before.kernel_exec_samples,
        kernel_exec_total_ms: after.kernel_exec_total_ms - before.kernel_exec_total_ms,
        correctness_validated,
    })
}

fn print_probe(report: &ProbeReport) {
    println!("## {}", report.name);
    println!("- elapsed_us: {}", report.elapsed.as_micros());
    println!("- result_rows: {}", report.result_rows);
    println!("- planned_target: {}", report.planned_target);
    println!("- executed_target: {}", report.executed_target);
    println!("- access_path: {}", report.access_path);
    println!("- sql_fallback: {}", report.sql_fallback);
    println!("- fallback_reason: {}", report.fallback_reason);
    println!("- h2d_bytes: {}", report.h2d_bytes);
    println!("- d2h_bytes: {}", report.d2h_bytes);
    println!("- kernel_exec_samples: {}", report.kernel_exec_samples);
    println!("- kernel_exec_total_ms: {}", report.kernel_exec_total_ms);
    println!("- correctness_validated: {}", report.correctness_validated);
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
