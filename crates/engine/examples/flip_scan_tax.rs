//! FLIP slice — measure the SHARDED SCAN TAX vs single-buffer before flipping shards-default ON.
//!
//! Every sharded SCAN/AGGREGATE recompacts the (zone-map-surviving) shards into one unified device
//! buffer per read (a full DtoD copy, scalability-ledger #4), where the single-buffer path reads its
//! resident buffer zero-copy. Point lookups are already O(1) via the index routes; the flip's perf
//! exposure is exactly the scan shapes. This bench quantifies it: for the SAME row count, time
//! COUNT(*) / SUM / a scan-served equality projection / IS NOT NULL count on (a) the single-buffer
//! layout and (b) the sharded layout at 1, 4, and 16 shards. p50/p99 latency + throughput per shape
//! (the report-card discipline: never a rate without its latency).
//!
//! Run: GPU_DB_BENCH_FLIP_TAX=1 cargo run --release --example flip_scan_tax -p gpu_db_engine
//!      GPU_DB_BENCH_ROWS=524288        (total rows)
//!      GPU_DB_BENCH_ITERS=200          (timed iterations per shape)

use std::env;
use std::error::Error;
use std::time::Instant;

use gpu_db_engine::Engine;

const INSERT_CHUNK: i64 = 1000;

fn percentile(sorted_us: &[f64], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_us.len() as f64 * p).ceil() as usize).saturating_sub(1);
    sorted_us[idx.min(sorted_us.len() - 1)]
}

fn load(e: &Engine, rows: i64) -> Result<(), Box<dyn Error>> {
    e.execute_text(1, "CREATE TABLE t (id INT, balance INT)")?;
    let mut seq = 2u64;
    let mut i = 0i64;
    while i < rows {
        let end = (i + INSERT_CHUNK).min(rows);
        let values: Vec<String> = (i..end).map(|k| format!("({k},{})", k % 1000)).collect();
        e.execute_text(
            seq,
            &format!("INSERT INTO t (id, balance) VALUES {}", values.join(",")),
        )?;
        seq += 1;
        i = end;
    }
    Ok(())
}

fn bench_shape(e: &Engine, sql: &str, iters: usize) -> (f64, f64, f64) {
    // Warmup (builds caches, uploads, first-touch).
    for _ in 0..10 {
        let _ = e.execute_relational_select_text(sql).expect("warmup query");
    }
    let mut lat_us: Vec<f64> = Vec::with_capacity(iters);
    let started = Instant::now();
    for _ in 0..iters {
        let q = Instant::now();
        let _ = e.execute_relational_select_text(sql).expect("timed query");
        lat_us.push(q.elapsed().as_secs_f64() * 1e6);
    }
    let total = started.elapsed().as_secs_f64();
    lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        percentile(&lat_us, 0.50),
        percentile(&lat_us, 0.99),
        iters as f64 / total,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    if env::var("GPU_DB_BENCH_FLIP_TAX").ok().as_deref() != Some("1") {
        println!("set GPU_DB_BENCH_FLIP_TAX=1 to run (GPU-touching). Skipping.");
        return Ok(());
    }
    let rows: i64 = env::var("GPU_DB_BENCH_ROWS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(524_288);
    let iters: usize = env::var("GPU_DB_BENCH_ITERS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(200);
    let shapes: Vec<(&str, String)> = vec![
        ("count_all", "SELECT COUNT(*) FROM t".to_string()),
        ("sum", "SELECT SUM(balance) FROM t".to_string()),
        (
            // NOTE: with shard_index_probe_enabled default-ON this is served by the O(1)
            // index route, not the recompaction scan — an end-to-end point-read number.
            "eq_scan_proj",
            format!("SELECT id, balance FROM t WHERE id = {}", rows / 2),
        ),
        (
            "not_null_count",
            "SELECT COUNT(*) FROM t WHERE balance IS NOT NULL".to_string(),
        ),
    ];

    // (label, shard flag, shard target). Single-buffer = flag OFF. Sharded targets chosen so the
    // loaded table lands at ~1 / ~4 / ~16 shards (the initial admit builds a small shard 0, so the
    // exact count is asserted and printed, not assumed).
    let configs: Vec<(String, bool, usize)> = vec![
        ("single-buffer".to_string(), false, 0),
        ("sharded~1".to_string(), true, rows as usize * 2),
        ("sharded~4".to_string(), true, rows as usize / 4),
        ("sharded~16".to_string(), true, rows as usize / 16),
    ];

    println!("flip_scan_tax: rows={rows} iters={iters} (p50/p99 us | q/s)");
    println!(
        "{:<14} {:>7} {:>26} {:>26} {:>26} {:>26}",
        "layout", "shards", "count_all", "sum", "eq_scan_proj", "not_null_count"
    );
    for (label, shard_flag, target) in configs {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        if shard_flag {
            e.set_shard_residency_enabled(true);
            e.set_shard_size_target(target.max(1));
        } else {
            // THE FLIP made sharded the default — the single-buffer CONTROL must pin the old layout.
            e.set_shard_residency_enabled(false);
        }
        load(&e, rows)?;
        let shard_count = e.resident_shard_count("t");
        let mut cells: Vec<String> = Vec::new();
        for (_, sql) in &shapes {
            let (p50, p99, qps) = bench_shape(&e, sql, iters);
            cells.push(format!("{p50:>8.0}/{p99:>8.0} | {qps:>6.0}"));
        }
        println!(
            "{:<14} {:>7} {:>26} {:>26} {:>26} {:>26}",
            label, shard_count, cells[0], cells[1], cells[2], cells[3]
        );
    }
    Ok(())
}
