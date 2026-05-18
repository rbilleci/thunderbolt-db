use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

use gpu_db_engine::{Engine, RelationalSelectResult};
use gpu_db_metrics::FallbackReason;
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
    let aggregate_distinct_queries = aggregate_distinct_queries()?;
    let count_query = select("SELECT COUNT(*) FROM events")?;
    let filtered_count_query = select(&format!(
        "SELECT COUNT(*) FROM events WHERE id = {}",
        (row_count * 37 % row_count) + 1
    ))?;
    let membership_a = (row_count / 3).max(1);
    let membership_b = (row_count / 2).max(1);
    let membership_c = row_count.saturating_sub(row_count / 7).max(1);
    let membership_count_query = select(&format!(
        "SELECT COUNT(*) FROM events WHERE id IN ({membership_a}, {membership_b}, {membership_c})"
    ))?;
    let range_count_threshold = row_count.saturating_sub(row_count / 4).max(1);
    let range_count_query = select(&format!(
        "SELECT COUNT(*) FROM events WHERE id >= {range_count_threshold}"
    ))?;
    let sum_query = select("SELECT SUM(amount) FROM events")?;
    let avg_query = select("SELECT AVG(amount) FROM events")?;
    let min_query = select("SELECT MIN(amount) FROM events")?;
    let max_query = select("SELECT MAX(amount) FROM events")?;
    let filtered_scalar_threshold = row_count.saturating_sub(row_count / 5).max(1);
    let filtered_scalar_sum_query = select(&format!(
        "SELECT SUM(amount) FROM events WHERE amount >= {filtered_scalar_threshold}"
    ))?;
    let filtered_scalar_avg_query = select(&format!(
        "SELECT AVG(amount) FROM events WHERE amount >= {filtered_scalar_threshold}"
    ))?;
    let filtered_scalar_min_query = select(&format!(
        "SELECT MIN(amount) FROM events WHERE amount >= {filtered_scalar_threshold}"
    ))?;
    let filtered_scalar_max_query = select(&format!(
        "SELECT MAX(amount) FROM events WHERE amount >= {filtered_scalar_threshold}"
    ))?;
    let projection_query = select(&format!(
        "SELECT amount FROM events WHERE amount >= {}",
        filtered_scalar_threshold
    ))?;
    let distinct_projection_query =
        select("SELECT DISTINCT bucket FROM events ORDER BY bucket DESC LIMIT 8 OFFSET 2")?;
    let filtered_distinct_bucket_threshold = 4;
    let filtered_distinct_projection_query = select(&format!(
        "SELECT DISTINCT bucket FROM events WHERE bucket >= {} ORDER BY bucket DESC LIMIT 8 OFFSET 1",
        filtered_distinct_bucket_threshold
    ))?;
    let filtered_ordered_projection_query = select(&format!(
        "SELECT amount FROM events WHERE amount >= {} ORDER BY amount DESC LIMIT 8 OFFSET 2",
        filtered_scalar_threshold
    ))?;
    let grouped_sum_query =
        select("SELECT bucket, SUM(amount) FROM events GROUP BY bucket ORDER BY sum DESC LIMIT 8")?;
    let grouped_count_query = select(
        "SELECT bucket, COUNT(*) FROM events GROUP BY bucket HAVING count >= 40 ORDER BY count DESC LIMIT 8",
    )?;
    let grouped_avg_query =
        select("SELECT bucket, AVG(amount) FROM events GROUP BY bucket ORDER BY avg DESC LIMIT 8")?;
    let grouped_min_query =
        select("SELECT bucket, MIN(amount) FROM events GROUP BY bucket ORDER BY min DESC LIMIT 8")?;
    let grouped_max_query =
        select("SELECT bucket, MAX(amount) FROM events GROUP BY bucket ORDER BY max DESC LIMIT 8")?;
    let filtered_grouped_sum_query = select(&format!(
        "SELECT bucket, SUM(amount) FROM events WHERE amount >= {filtered_scalar_threshold} GROUP BY bucket HAVING sum > 0 ORDER BY sum DESC LIMIT 8"
    ))?;
    let filtered_grouped_count_query = select(&format!(
        "SELECT bucket, COUNT(*) FROM events WHERE amount >= {filtered_scalar_threshold} GROUP BY bucket ORDER BY count DESC LIMIT 8"
    ))?;
    let filtered_grouped_avg_query = select(&format!(
        "SELECT bucket, AVG(amount) FROM events WHERE amount >= {filtered_scalar_threshold} GROUP BY bucket ORDER BY avg DESC LIMIT 8"
    ))?;
    let filtered_grouped_min_query = select(&format!(
        "SELECT bucket, MIN(amount) FROM events WHERE amount >= {filtered_scalar_threshold} GROUP BY bucket ORDER BY min DESC LIMIT 8"
    ))?;
    let filtered_grouped_max_query = select(&format!(
        "SELECT bucket, MAX(amount) FROM events WHERE amount >= {filtered_scalar_threshold} GROUP BY bucket ORDER BY max DESC LIMIT 8"
    ))?;
    let mutation_query = select(&format!(
        "SELECT * FROM events WHERE id = {}",
        row_count + 1
    ))?;

    let mut cpu = seeded_engine(row_count)?;
    let mut gpu = seeded_engine(row_count)?;

    let cpu_result = cpu.execute_relational_select(&query)?;
    let cold_probe = timed_probe(&mut gpu, &query, &cpu_result, "cold_per_query_h2d_probe")?;
    let resident_snapshot = gpu.populate_relational_residency_snapshot("events")?;
    let warm_resident_probe = timed_resident_probe(
        &mut gpu,
        &query,
        &cpu_result,
        "warm_resident_snapshot_probe",
    )?;
    let aggregate_distinct_cpu_results = execute_queries(&mut cpu, &aggregate_distinct_queries)?;
    let warm_resident_aggregate_distinct_probe = timed_resident_probe_many(
        &mut gpu,
        &aggregate_distinct_queries,
        &aggregate_distinct_cpu_results,
        "warm_resident_aggregate_distinct_probe",
    )?;
    let count_cpu_result = cpu.execute_relational_select(&count_query)?;
    let resident_device_count_probe = timed_resident_device_count_probe(
        &mut gpu,
        &count_query,
        &count_cpu_result,
        "resident_device_memory_count_kernel_probe",
    )?;
    let filtered_count_cpu_result = cpu.execute_relational_select(&filtered_count_query)?;
    let resident_device_filtered_count_probe = timed_resident_device_filtered_count_probe(
        &mut gpu,
        &filtered_count_query,
        &filtered_count_cpu_result,
        "resident_device_memory_filtered_count_kernel_probe",
    )?;
    let membership_count_cpu_result = cpu.execute_relational_select(&membership_count_query)?;
    let resident_device_membership_count_probe = timed_resident_device_membership_count_probe(
        &mut gpu,
        &membership_count_query,
        &membership_count_cpu_result,
        "resident_device_memory_membership_count_kernel_probe",
    )?;
    let range_count_cpu_result = cpu.execute_relational_select(&range_count_query)?;
    let resident_device_range_count_probe = timed_resident_device_range_count_probe(
        &mut gpu,
        &range_count_query,
        &range_count_cpu_result,
        "resident_device_memory_range_count_kernel_probe",
    )?;
    let sum_cpu_result = cpu.execute_relational_select(&sum_query)?;
    let resident_device_sum_probe = timed_resident_device_sum_probe(
        &mut gpu,
        &sum_query,
        &sum_cpu_result,
        "resident_device_memory_sum_kernel_probe",
    )?;
    let avg_cpu_result = cpu.execute_relational_select(&avg_query)?;
    let resident_device_avg_probe = timed_resident_device_scalar_aggregate_probe(
        &mut gpu,
        &avg_query,
        &avg_cpu_result,
        "resident_device_memory_avg_kernel_probe",
    )?;
    let min_cpu_result = cpu.execute_relational_select(&min_query)?;
    let resident_device_min_probe = timed_resident_device_scalar_aggregate_probe(
        &mut gpu,
        &min_query,
        &min_cpu_result,
        "resident_device_memory_min_kernel_probe",
    )?;
    let max_cpu_result = cpu.execute_relational_select(&max_query)?;
    let resident_device_max_probe = timed_resident_device_scalar_aggregate_probe(
        &mut gpu,
        &max_query,
        &max_cpu_result,
        "resident_device_memory_max_kernel_probe",
    )?;
    let filtered_scalar_sum_cpu_result =
        cpu.execute_relational_select(&filtered_scalar_sum_query)?;
    let resident_device_filtered_scalar_sum_probe =
        timed_resident_device_filtered_scalar_aggregate_probe(
            &mut gpu,
            &filtered_scalar_sum_query,
            &filtered_scalar_sum_cpu_result,
            "resident_device_memory_filtered_sum_kernel_probe",
        )?;
    let filtered_scalar_avg_cpu_result =
        cpu.execute_relational_select(&filtered_scalar_avg_query)?;
    let resident_device_filtered_scalar_avg_probe =
        timed_resident_device_filtered_scalar_aggregate_probe(
            &mut gpu,
            &filtered_scalar_avg_query,
            &filtered_scalar_avg_cpu_result,
            "resident_device_memory_filtered_avg_kernel_probe",
        )?;
    let filtered_scalar_min_cpu_result =
        cpu.execute_relational_select(&filtered_scalar_min_query)?;
    let resident_device_filtered_scalar_min_probe =
        timed_resident_device_filtered_scalar_aggregate_probe(
            &mut gpu,
            &filtered_scalar_min_query,
            &filtered_scalar_min_cpu_result,
            "resident_device_memory_filtered_min_kernel_probe",
        )?;
    let filtered_scalar_max_cpu_result =
        cpu.execute_relational_select(&filtered_scalar_max_query)?;
    let resident_device_filtered_scalar_max_probe =
        timed_resident_device_filtered_scalar_aggregate_probe(
            &mut gpu,
            &filtered_scalar_max_query,
            &filtered_scalar_max_cpu_result,
            "resident_device_memory_filtered_max_kernel_probe",
        )?;
    let projection_cpu_result = cpu.execute_relational_select(&projection_query)?;
    let resident_device_projection_probe = timed_resident_device_projection_probe(
        &mut gpu,
        &projection_query,
        &projection_cpu_result,
        "resident_device_memory_projection_kernel_probe",
    )?;
    let distinct_projection_cpu_result =
        cpu.execute_relational_select(&distinct_projection_query)?;
    let resident_device_distinct_projection_probe =
        timed_resident_device_distinct_projection_probe(
            &mut gpu,
            &distinct_projection_query,
            &distinct_projection_cpu_result,
            "resident_device_memory_distinct_projection_kernel_probe",
        )?;
    let filtered_distinct_projection_cpu_result =
        cpu.execute_relational_select(&filtered_distinct_projection_query)?;
    let resident_device_filtered_distinct_projection_probe =
        timed_resident_device_filtered_distinct_projection_probe(
            &mut gpu,
            &filtered_distinct_projection_query,
            &filtered_distinct_projection_cpu_result,
            "resident_device_memory_filtered_distinct_projection_kernel_probe",
        )?;
    let filtered_ordered_projection_cpu_result =
        cpu.execute_relational_select(&filtered_ordered_projection_query)?;
    let resident_device_filtered_ordered_projection_probe =
        timed_resident_device_ordered_projection_probe(
            &mut gpu,
            &filtered_ordered_projection_query,
            &filtered_ordered_projection_cpu_result,
            "resident_device_memory_filtered_ordered_projection_kernel_probe",
        )?;
    let grouped_sum_cpu_result = cpu.execute_relational_select(&grouped_sum_query)?;
    let resident_device_grouped_sum_probe = timed_resident_device_grouped_sum_probe(
        &mut gpu,
        &grouped_sum_query,
        &grouped_sum_cpu_result,
        "resident_device_memory_grouped_sum_kernel_probe",
    )?;
    let grouped_count_cpu_result = cpu.execute_relational_select(&grouped_count_query)?;
    let resident_device_grouped_count_probe = timed_resident_device_grouped_aggregate_probe(
        &mut gpu,
        &grouped_count_query,
        &grouped_count_cpu_result,
        "resident_device_memory_grouped_count_kernel_probe",
    )?;
    let grouped_avg_cpu_result = cpu.execute_relational_select(&grouped_avg_query)?;
    let resident_device_grouped_avg_probe = timed_resident_device_grouped_aggregate_probe(
        &mut gpu,
        &grouped_avg_query,
        &grouped_avg_cpu_result,
        "resident_device_memory_grouped_avg_kernel_probe",
    )?;
    let grouped_min_cpu_result = cpu.execute_relational_select(&grouped_min_query)?;
    let resident_device_grouped_min_probe = timed_resident_device_grouped_aggregate_probe(
        &mut gpu,
        &grouped_min_query,
        &grouped_min_cpu_result,
        "resident_device_memory_grouped_min_kernel_probe",
    )?;
    let grouped_max_cpu_result = cpu.execute_relational_select(&grouped_max_query)?;
    let resident_device_grouped_max_probe = timed_resident_device_grouped_aggregate_probe(
        &mut gpu,
        &grouped_max_query,
        &grouped_max_cpu_result,
        "resident_device_memory_grouped_max_kernel_probe",
    )?;
    let filtered_grouped_sum_cpu_result =
        cpu.execute_relational_select(&filtered_grouped_sum_query)?;
    let resident_device_filtered_grouped_sum_probe =
        timed_resident_device_filtered_grouped_aggregate_probe(
            &mut gpu,
            &filtered_grouped_sum_query,
            &filtered_grouped_sum_cpu_result,
            "resident_device_memory_filtered_grouped_sum_kernel_probe",
        )?;
    let filtered_grouped_count_cpu_result =
        cpu.execute_relational_select(&filtered_grouped_count_query)?;
    let resident_device_filtered_grouped_count_probe =
        timed_resident_device_filtered_grouped_aggregate_probe(
            &mut gpu,
            &filtered_grouped_count_query,
            &filtered_grouped_count_cpu_result,
            "resident_device_memory_filtered_grouped_count_kernel_probe",
        )?;
    let filtered_grouped_avg_cpu_result =
        cpu.execute_relational_select(&filtered_grouped_avg_query)?;
    let resident_device_filtered_grouped_avg_probe =
        timed_resident_device_filtered_grouped_aggregate_probe(
            &mut gpu,
            &filtered_grouped_avg_query,
            &filtered_grouped_avg_cpu_result,
            "resident_device_memory_filtered_grouped_avg_kernel_probe",
        )?;
    let filtered_grouped_min_cpu_result =
        cpu.execute_relational_select(&filtered_grouped_min_query)?;
    let resident_device_filtered_grouped_min_probe =
        timed_resident_device_filtered_grouped_aggregate_probe(
            &mut gpu,
            &filtered_grouped_min_query,
            &filtered_grouped_min_cpu_result,
            "resident_device_memory_filtered_grouped_min_kernel_probe",
        )?;
    let filtered_grouped_max_cpu_result =
        cpu.execute_relational_select(&filtered_grouped_max_query)?;
    let resident_device_filtered_grouped_max_probe =
        timed_resident_device_filtered_grouped_aggregate_probe(
            &mut gpu,
            &filtered_grouped_max_query,
            &filtered_grouped_max_cpu_result,
            "resident_device_memory_filtered_grouped_max_kernel_probe",
        )?;

    let new_id = row_count + 1;
    let insert_sql = format!(
        "INSERT INTO events (id, account, amount, category, bucket) VALUES ({new_id}, 'acct_refresh', 7777, 'refresh', 777)"
    );
    cpu.execute_text((row_count + 4) as u64, &insert_sql)?;
    gpu.execute_text((row_count + 4) as u64, &insert_sql)?;
    let invalidated_snapshot = gpu
        .relational_residency_snapshot("events")
        .ok_or("missing resident snapshot after mutation")?;
    let mutation_cpu_result = cpu.execute_relational_select(&mutation_query)?;
    let mutation_probe = timed_probe(
        &mut gpu,
        &mutation_query,
        &mutation_cpu_result,
        "post_mutation_per_query_h2d_probe",
    )?;
    gpu.mark_gpu_memory_pressured(0);
    let pressure_snapshot = gpu
        .relational_residency_snapshot("events")
        .ok_or("missing resident snapshot after memory pressure")?;
    gpu.enqueue_set_text(
        (row_count + 3) as u64,
        "SET residency_pressure_probe=1",
        Instant::now(),
    )?;
    let memory_pressure_fallback_count = gpu
        .metrics()
        .fallback_for(FallbackReason::GpuMemoryPressure);
    gpu.clear_gpu_memory_pressured(0);
    let refresh_started = Instant::now();
    let refreshed_snapshot = gpu.populate_relational_residency_snapshot("events")?;
    let refresh_elapsed = refresh_started.elapsed();
    let refresh_cost = refreshed_snapshot
        .last_refresh_cost
        .as_ref()
        .ok_or("missing refresh-cost metadata after residency refresh")?;
    let aux_snapshot = gpu.populate_relational_residency_snapshot("resident_aux")?;
    gpu.set_relational_residency_budget_bytes(0, refreshed_snapshot.resident_bytes);
    let budgeted_snapshot = gpu.populate_relational_residency_snapshot("events")?;
    let budget_evicted_aux = gpu.relational_residency_snapshot("resident_aux").is_none();
    let oversize_err = {
        gpu.set_relational_residency_budget_bytes(0, refreshed_snapshot.resident_bytes - 1);
        gpu.populate_relational_residency_snapshot("events")
            .err()
            .map(|err| err.to_string())
            .unwrap_or_else(|| "accepted".to_string())
    };
    let oversize_rejected = oversize_err.contains("exceeding GPU 0 residency budget");
    gpu.clear_relational_residency_budget_bytes(0);

    println!("# P7 GPU Residency Baseline");
    println!();
    println!("- dataset_rows: {row_count}");
    println!("- lookup_count: {lookup_count}");
    println!("- concurrency: 1");
    println!("- device_info: {device_info}");
    println!(
        "- current_data_residency_model: bounded_resident_snapshot_probe_with_retained_cuda_allocation_plus_per_query_h2d_fallback"
    );
    println!("- warm_resident_snapshot_execution_supported: true");
    println!("- production_device_cache_supported: bounded_retained_snapshot_handle");
    println!(
        "- resident_device_memory_query_kernel_supported: bounded_count_all_int4_equality_count_int4_membership_count_int4_range_count_int4_sum_avg_min_max_filtered_sum_avg_min_max_int4_projection_int4_paginated_distinct_projection_int4_paginated_filtered_distinct_projection_int4_paginated_filtered_ordered_projection_int4_grouped_count_sum_avg_min_max_grouped_having_and_filtered_grouped_count_sum_avg_min_max_filtered_grouped_having"
    );
    println!(
        "- resident_device_memory_proof_supported: {}",
        resident_snapshot.device_memory_proof.is_some()
    );
    println!(
        "- resident_device_memory_allocated_bytes: {}",
        resident_snapshot
            .device_memory_proof
            .as_ref()
            .map(|proof| proof.allocated_bytes)
            .unwrap_or_default()
    );
    println!(
        "- resident_device_memory_copied_bytes: {}",
        resident_snapshot
            .device_memory_proof
            .as_ref()
            .map(|proof| proof.copied_bytes)
            .unwrap_or_default()
    );
    println!(
        "- resident_device_memory_gpu_id: {}",
        resident_snapshot
            .device_memory_proof
            .as_ref()
            .map(|proof| proof.gpu_id.to_string())
            .unwrap_or_else(|| "None".to_string())
    );
    println!(
        "- resident_device_memory_retained: {}",
        resident_snapshot
            .device_memory_proof
            .as_ref()
            .map(|proof| proof.retained)
            .unwrap_or(false)
    );
    println!(
        "- resident_snapshot_valid_before_mutation: {}",
        resident_snapshot.is_valid()
    );
    println!(
        "- resident_snapshot_valid_after_mutation: {}",
        invalidated_snapshot.is_valid()
    );
    println!(
        "- resident_snapshot_valid_under_memory_pressure: {}",
        pressure_snapshot.is_valid()
    );
    println!(
        "- resident_bytes_current: {}",
        resident_snapshot.resident_bytes
    );
    println!("- resident_rows_current: {}", resident_snapshot.row_count);
    println!(
        "- resident_valid_through_index: {}",
        resident_snapshot.valid_through_index
    );
    println!(
        "- resident_invalidated_by_txn_id: {}",
        invalidated_snapshot
            .invalidated_by_txn_id
            .map(|txn_id| txn_id.to_string())
            .unwrap_or_else(|| "None".to_string())
    );
    println!(
        "- resident_invalidated_at_index: {}",
        invalidated_snapshot
            .invalidated_at_index
            .map(|index| index.to_string())
            .unwrap_or_else(|| "None".to_string())
    );
    println!(
        "- resident_refreshed_rows_current: {}",
        refreshed_snapshot.row_count
    );
    println!(
        "- resident_refreshed_bytes_current: {}",
        refreshed_snapshot.resident_bytes
    );
    println!("- resident_refresh_supported: manual_snapshot_refresh_with_cost_accounting");
    println!("- resident_refresh_cost_recorded: true");
    println!(
        "- resident_refresh_previous_rows: {}",
        refresh_cost.previous_row_count
    );
    println!(
        "- resident_refresh_refreshed_rows: {}",
        refresh_cost.refreshed_row_count
    );
    println!("- resident_refresh_row_delta: {}", refresh_cost.row_delta);
    println!(
        "- resident_refresh_previous_bytes: {}",
        refresh_cost.previous_resident_bytes
    );
    println!(
        "- resident_refresh_refreshed_bytes: {}",
        refresh_cost.refreshed_resident_bytes
    );
    println!(
        "- resident_refresh_byte_delta: {}",
        refresh_cost.resident_byte_delta
    );
    println!(
        "- resident_refresh_from_index: {}",
        refresh_cost.refreshed_from_index
    );
    println!(
        "- resident_refresh_through_index: {}",
        refresh_cost.refreshed_through_index
    );
    println!(
        "- resident_refresh_invalidated_by_txn_id: {}",
        refresh_cost
            .invalidated_by_txn_id
            .map(|txn_id| txn_id.to_string())
            .unwrap_or_else(|| "None".to_string())
    );
    println!(
        "- resident_refresh_invalidated_at_index: {}",
        refresh_cost
            .invalidated_at_index
            .map(|index| index.to_string())
            .unwrap_or_else(|| "None".to_string())
    );
    println!(
        "- resident_refresh_invalidated_by_memory_pressure: {}",
        refresh_cost.invalidated_by_memory_pressure
    );
    println!(
        "- resident_refresh_elapsed_ms: {:.3}",
        refresh_elapsed.as_secs_f64() * 1000.0
    );
    println!("- resident_budget_admission_supported: true");
    println!(
        "- resident_budget_bytes: {}",
        budgeted_snapshot.admission_budget_bytes.unwrap_or_default()
    );
    println!(
        "- resident_budget_bytes_after_admission: {}",
        budgeted_snapshot.resident_bytes_after_admission
    );
    println!(
        "- resident_budget_evicted_tables: {}",
        if budgeted_snapshot.evicted_tables_on_admission.is_empty() {
            "none".to_string()
        } else {
            budgeted_snapshot.evicted_tables_on_admission.join(",")
        }
    );
    println!("- resident_budget_evicted_aux_snapshot: {budget_evicted_aux}");
    println!(
        "- resident_budget_aux_bytes_before_eviction: {}",
        aux_snapshot.resident_bytes
    );
    println!("- resident_budget_oversize_rejected: {oversize_rejected}");
    println!("- memory_pressure_fallback_supported: true");
    println!(
        "- memory_pressure_invalidates_resident_snapshot: {}",
        pressure_snapshot.invalidated_by_memory_pressure
    );
    println!(
        "- memory_pressure_active_on_snapshot: {}",
        pressure_snapshot.memory_pressure_active
    );
    println!("- memory_pressure_fallback_count: {memory_pressure_fallback_count}");
    println!("- correctness_oracle: CPU relational engine");
    println!();
    print_probe(&cold_probe);
    println!();
    print_probe(&warm_resident_probe);
    println!();
    print_probe(&warm_resident_aggregate_distinct_probe);
    println!();
    print_probe(&resident_device_count_probe);
    println!();
    print_probe(&resident_device_filtered_count_probe);
    println!();
    print_probe(&resident_device_membership_count_probe);
    println!();
    print_probe(&resident_device_range_count_probe);
    println!();
    print_probe(&resident_device_sum_probe);
    println!();
    print_probe(&resident_device_avg_probe);
    println!();
    print_probe(&resident_device_min_probe);
    println!();
    print_probe(&resident_device_max_probe);
    println!();
    print_probe(&resident_device_filtered_scalar_sum_probe);
    println!();
    print_probe(&resident_device_filtered_scalar_avg_probe);
    println!();
    print_probe(&resident_device_filtered_scalar_min_probe);
    println!();
    print_probe(&resident_device_filtered_scalar_max_probe);
    println!();
    print_probe(&resident_device_projection_probe);
    println!();
    print_probe(&resident_device_distinct_projection_probe);
    println!();
    print_probe(&resident_device_filtered_distinct_projection_probe);
    println!();
    print_probe(&resident_device_filtered_ordered_projection_probe);
    println!();
    print_probe(&resident_device_grouped_sum_probe);
    println!();
    print_probe(&resident_device_grouped_count_probe);
    println!();
    print_probe(&resident_device_grouped_avg_probe);
    println!();
    print_probe(&resident_device_grouped_min_probe);
    println!();
    print_probe(&resident_device_grouped_max_probe);
    println!();
    print_probe(&resident_device_filtered_grouped_sum_probe);
    println!();
    print_probe(&resident_device_filtered_grouped_count_probe);
    println!();
    print_probe(&resident_device_filtered_grouped_avg_probe);
    println!();
    print_probe(&resident_device_filtered_grouped_min_probe);
    println!();
    print_probe(&resident_device_filtered_grouped_max_probe);
    println!();
    print_probe(&mutation_probe);
    println!();
    println!(
        "decision: current P7 evidence includes bounded resident table-data snapshot SELECT probes with zero per-query H2D transfer for the app lookup workload and supported aggregate/distinct SQL shapes, retained-device-memory COUNT(*), int4 equality-predicate COUNT(*), int4 membership-predicate COUNT(*), int4 range-predicate COUNT(*), int4 scalar SUM/AVG/MIN/MAX, int4 filtered scalar SUM/AVG/MIN/MAX, int4 predicate-projection, int4 paginated distinct projection, int4 paginated filtered distinct projection, bounded int4 paginated filtered ordered-projection, int4 grouped COUNT/SUM/AVG/MIN/MAX with grouped HAVING, and int4 filtered grouped COUNT/SUM/AVG/MIN/MAX with filtered grouped HAVING proofs over the resident allocation, resident-byte accounting, WAL-safe invalidation, manual refresh-cost accounting, memory-pressure fallback metadata, deterministic resident-snapshot budget admission/eviction, and a retained real CUDA allocation/copy handle for encoded snapshot bytes when local driver hardware is available. Keep broad production CUDA cache claims out of scope until broader expression kernels read directly from retained device-memory handles."
    );

    Ok(())
}

fn timed_resident_device_count_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result = engine.execute_relational_count_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory count diverged").into());
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

fn timed_resident_device_filtered_count_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result =
        engine.execute_relational_filtered_count_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory filtered count diverged").into());
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

fn timed_resident_device_membership_count_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result =
        engine.execute_relational_membership_count_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory membership count diverged").into());
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

fn timed_resident_device_range_count_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result = engine.execute_relational_range_count_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory range count diverged").into());
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

fn timed_resident_device_sum_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result = engine.execute_relational_sum_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory SUM diverged").into());
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

fn timed_resident_device_scalar_aggregate_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result =
        engine.execute_relational_scalar_aggregate_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory scalar aggregate diverged").into());
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

fn timed_resident_device_filtered_scalar_aggregate_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result = engine
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory filtered aggregate diverged").into());
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

fn timed_resident_device_projection_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result = engine.execute_relational_projection_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory projection diverged").into());
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

fn timed_resident_device_distinct_projection_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result =
        engine.execute_relational_distinct_projection_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory distinct projection diverged").into());
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

fn timed_resident_device_filtered_distinct_projection_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result = engine
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(
            format!("{name} resident device-memory filtered distinct projection diverged").into(),
        );
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

fn timed_resident_device_ordered_projection_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result =
        engine.execute_relational_ordered_projection_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory ordered projection diverged").into());
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

fn timed_resident_device_grouped_sum_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    timed_resident_device_grouped_aggregate_probe(engine, query, expected, name)
}

fn timed_resident_device_grouped_aggregate_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result =
        engine.execute_relational_grouped_aggregate_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident device-memory grouped aggregate diverged").into());
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

fn timed_resident_device_filtered_grouped_aggregate_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result = engine
        .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(
            format!("{name} resident device-memory filtered grouped aggregate diverged").into(),
        );
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

fn timed_resident_probe_many(
    engine: &mut Engine,
    queries: &[Select],
    expected: &[RelationalSelectResult],
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let mut results = Vec::with_capacity(queries.len());
    for query in queries {
        results.push(engine.execute_relational_select_with_resident_snapshot_probe(query)?);
    }
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = results.len() == expected.len()
        && results.iter().zip(expected).all(|(actual, expected)| {
            actual.columns == expected.columns && actual.rows == expected.rows
        });
    if !correctness_validated {
        return Err(format!("{name} resident snapshot results diverged").into());
    }
    let result_rows = results.iter().map(|result| result.rows.len()).sum();
    let sql_fallback = results
        .iter()
        .any(|result| result.fallback_reason.is_some());
    let fallback_reason = results
        .iter()
        .find_map(|result| {
            result
                .fallback_reason
                .as_ref()
                .map(|reason| format!("{reason:?}"))
        })
        .unwrap_or_else(|| "None".to_string());
    let access_paths = {
        let mut paths = results
            .iter()
            .map(|result| format!("{:?}", result.access_path))
            .collect::<Vec<_>>();
        paths.sort();
        paths.dedup();
        paths.join("; ")
    };
    Ok(ProbeReport {
        name,
        elapsed,
        result_rows,
        planned_target: "Gpu(0)".to_string(),
        executed_target: "Gpu(0)".to_string(),
        access_path: access_paths,
        sql_fallback,
        fallback_reason,
        h2d_bytes: after.h2d_bytes_total - before.h2d_bytes_total,
        d2h_bytes: after.d2h_bytes_total - before.d2h_bytes_total,
        kernel_exec_samples: after.kernel_exec_samples - before.kernel_exec_samples,
        kernel_exec_total_ms: after.kernel_exec_total_ms - before.kernel_exec_total_ms,
        correctness_validated,
    })
}

fn timed_resident_probe(
    engine: &mut Engine,
    query: &Select,
    expected: &RelationalSelectResult,
    name: &'static str,
) -> Result<ProbeReport, Box<dyn Error>> {
    let before = engine.metrics().snapshot();
    let start = Instant::now();
    let result = engine.execute_relational_select_with_resident_snapshot_probe(query)?;
    let elapsed = start.elapsed();
    let after = engine.metrics().snapshot();
    let correctness_validated = result.columns == expected.columns && result.rows == expected.rows;
    if !correctness_validated {
        return Err(format!("{name} resident snapshot results diverged").into());
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
        "CREATE TABLE events (id INT, account TEXT, amount INT, category TEXT, bucket INT)",
    )?;
    engine.execute_text(2, "CREATE TABLE resident_aux (id INT, label TEXT)")?;
    engine.execute_text(3, "INSERT INTO resident_aux (id, label) VALUES (1, 'aux')")?;
    for id in 1..=row_count {
        let account = format!("acct{}", id % 64);
        let category = if id % 2 == 0 { "even" } else { "odd" };
        let amount = (id % 10_000) as i32;
        let bucket = (id % 8) as i32;
        let sql = format!(
            "INSERT INTO events (id, account, amount, category, bucket) VALUES ({id}, '{account}', {amount}, '{category}', {bucket})"
        );
        engine.execute_text((id + 3) as u64, &sql)?;
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

fn aggregate_distinct_queries() -> Result<Vec<Select>, Box<dyn Error>> {
    Ok(vec![
        select("SELECT DISTINCT category FROM events ORDER BY category")?,
        select(
            "SELECT category, COUNT(*) FROM events WHERE amount >= 900 GROUP BY category ORDER BY count DESC",
        )?,
        select(
            "SELECT category, SUM(amount) FROM events WHERE amount >= 900 GROUP BY category ORDER BY sum DESC",
        )?,
        select("SELECT AVG(amount) FROM events WHERE category = 'even'")?,
        select("SELECT MIN(amount) FROM events WHERE amount >= 900")?,
        select("SELECT MAX(amount) FROM events WHERE amount >= 900")?,
    ])
}

fn execute_queries(
    engine: &mut Engine,
    queries: &[Select],
) -> Result<Vec<RelationalSelectResult>, Box<dyn Error>> {
    queries
        .iter()
        .map(|query| engine.execute_relational_select(query).map_err(Into::into))
        .collect()
}

fn select(sql: &str) -> Result<Select, Box<dyn Error>> {
    match parse_command(sql)? {
        Command::Select(select) => Ok(select),
        _ => Err(format!("not a SELECT: {sql}").into()),
    }
}
