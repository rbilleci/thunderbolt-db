//! GPU-native batched cross-shard point lookup vs the single-flight SQL route.
//!
//! The production batched path prepares one generation-owned device descriptor/index plan, then launches one
//! multi-shard probe+visibility+gather+dense-emit kernel and performs one bounded result readback per batch. This
//! example sweeps batch size and shard geometry against one-lookup-per-SQL-call latency. Its configured-row cases
//! are NOT cache-regime evidence; the canonical report card alone owns in-L2/out-of-L2 labels.
//!
//! Run: GPU_DB_BENCH_SHARD_INDEX_AB=1 cargo run --release --example r3_shard_index_route_ab -p gpu_db_engine
//!      GPU_DB_BENCH_ROWS=1048576               (rows in the configured-row cases)
//!      GPU_DB_BENCH_BATCH_SIZES=1,8,32,256,4096,65536
//!      GPU_DB_BENCH_BATCHES=200                (timed batches per size)

use std::env;
use std::error::Error;
use std::time::Instant;

use gpu_db_engine::Engine;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;

const INSERT_CHUNK: i64 = 2000;

fn next_rand(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn percentile(sorted_us: &[f64], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_us.len() as f64 * p).ceil() as usize).saturating_sub(1);
    sorted_us[idx.min(sorted_us.len() - 1)]
}

fn load(rows: i64, shard_size: usize) -> Result<Engine, Box<dyn Error>> {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.set_shard_index_probe_enabled(true);
    engine.set_shard_size_target(shard_size);
    engine.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")?;
    let mut txn = 2u64;
    let mut id = 0i64;
    while id < rows {
        let mut vals = String::new();
        for _ in 0..INSERT_CHUNK {
            if id >= rows {
                break;
            }
            if !vals.is_empty() {
                vals.push(',');
            }
            vals.push_str(&format!("({}, {})", id, id * 10));
            id += 1;
        }
        engine.execute_text(
            txn,
            &format!("INSERT INTO accounts (id, balance) VALUES {vals}"),
        )?;
        txn += 1;
    }
    Ok(engine)
}

fn main() -> Result<(), Box<dyn Error>> {
    if env::var("GPU_DB_BENCH_SHARD_INDEX_AB").ok().as_deref() != Some("1") {
        println!("set GPU_DB_BENCH_SHARD_INDEX_AB=1 to run (GPU-touching). Skipping.");
        return Ok(());
    }
    let rows: i64 = env::var("GPU_DB_BENCH_ROWS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1_048_576);
    let batch_sizes: Vec<usize> = env::var("GPU_DB_BENCH_BATCH_SIZES")
        .unwrap_or_else(|_| "1,8,32,256,4096,65536".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let batches: usize = env::var("GPU_DB_BENCH_BATCHES")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(200);

    // The many-shard case forces small shards (ordered inserts -> ascending-disjoint ranges) so the O(1)
    // BINARY-SEARCH route fires (each needle -> its one shard in O(log shards)); GPU_DB_BENCH_SHARDS sets the
    // target shard count (default 64). The labels deliberately describe geometry, not cache residency.
    let many_shards: i64 = env::var("GPU_DB_BENCH_SHARDS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(64);
    let many_shard_size = (rows / many_shards.max(1)).max(1) as usize;
    let regimes: [(&str, i64, usize, bool); 3] = [
        ("small control (65536 rows, 1 shard)", 65_536, 65_536, false),
        (
            "configured rows, 1 shard",
            rows,
            rows.max(1) as usize,
            false,
        ),
        (
            "configured rows, many shards (binary route)",
            rows,
            many_shard_size,
            true,
        ),
    ];
    let proj = vec!["id".to_string(), "balance".to_string()];

    for (label, rrows, shard_size, require_binary) in regimes {
        let engine = load(rrows, shard_size)?;
        let shards = engine.resident_shard_count("accounts");
        println!("\n## {label} — accounts(id INT, balance INT), rows={rrows}, shards={shards}");
        println!("| mode | batch | p50 us | p99 us | lookups/s | us/lookup |");
        println!("|---|---|---|---|---|---|");

        // Warm up the per-shard index cache + JIT.
        for w in 0..8i64 {
            let k = (w * rrows / 8).clamp(0, rrows - 1) as i32;
            let _ = engine.bench_sharded_point_lookup_batch("accounts", "id", &proj, &[k])?;
            let _ = engine.execute_relational_select_text(&format!(
                "SELECT id, balance FROM accounts WHERE id = {k}"
            ))?;
        }

        // SINGLE-FLIGHT baseline (the 3b SQL route, one lookup per call).
        let mut rng = 0x9E3779B97F4A7C15u64 ^ rrows as u64;
        let sf_lookups = 500usize;
        let mut sf_us: Vec<f64> = Vec::with_capacity(sf_lookups);
        let sf_route_hits_before = engine.shard_index_route_hits();
        for _ in 0..sf_lookups {
            let k = (next_rand(&mut rng) % rrows as u64) as i32;
            let t = Instant::now();
            let result = engine.execute_relational_select_text(&format!(
                "SELECT id, balance FROM accounts WHERE id = {k}"
            ))?;
            sf_us.push(t.elapsed().as_secs_f64() * 1e6);
            assert!(
                matches!(result.executed_target, DeviceTarget::Gpu(_)),
                "the measured SQL 3b lookup must execute on the GPU"
            );
            assert_eq!(result.rows.len(), 1, "the measured key must be present");
            assert_eq!(
                result.rows.row(0),
                [SqlValue::Int4(k), SqlValue::Int4(k * 10)],
                "the measured SQL 3b result must contain the requested key and value"
            );
        }
        let sf_route_hit_delta = engine
            .shard_index_route_hits()
            .saturating_sub(sf_route_hits_before);
        assert_eq!(
            sf_route_hit_delta, sf_lookups as u64,
            "every measured single-flight call must execute exactly one SQL 3b shard-index route"
        );
        sf_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sf_total_s: f64 = sf_us.iter().sum::<f64>() / 1e6;
        println!(
            "| single-flight (SQL 3b) | 1 | {:.1} | {:.1} | {:.0} | {:.2} |",
            percentile(&sf_us, 0.50),
            percentile(&sf_us, 0.99),
            sf_lookups as f64 / sf_total_s,
            sf_total_s * 1e6 / sf_lookups as f64,
        );
        println!("single-flight measured-loop SQL 3b route-hit delta = {sf_route_hit_delta}");

        // BATCHED sweep. Snapshot counters after warmup/single-flight so labels are proven by measured-loop deltas.
        let batch_hits_before = engine.sharded_point_batch_hits();
        let gpu_probe_hits_before = engine.sharded_point_gpu_probe_hits();
        let binary_hits_before = engine.sharded_point_binary_route_hits();
        let timed_calls = batch_sizes
            .len()
            .checked_mul(batches)
            .expect("measured batch-call count fits usize");
        for &b in &batch_sizes {
            let mut lat_us: Vec<f64> = Vec::with_capacity(batches);
            let mut found_total: usize = 0;
            for _ in 0..batches {
                let needles: Vec<i32> = (0..b)
                    .map(|_| (next_rand(&mut rng) % rrows as u64) as i32)
                    .collect();
                let t = Instant::now();
                let found = engine
                    .bench_sharded_point_lookup_batch("accounts", "id", &proj, &needles)
                    .expect("batched GPU route completed")
                    .expect("batched path served");
                lat_us.push(t.elapsed().as_secs_f64() * 1e6);
                found_total += found;
            }
            // Sanity: random present keys -> every needle finds a row (distinctness not enforced, dups ok).
            assert_eq!(
                found_total,
                batches * b,
                "production batched route must materialize every present benchmark needle"
            );
            lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let total_s: f64 = lat_us.iter().sum::<f64>() / 1e6;
            let lookups = (batches * b) as f64;
            println!(
                "| batched | {b} | {:.1} | {:.1} | {:.0} | {:.3} |",
                percentile(&lat_us, 0.50),
                percentile(&lat_us, 0.99),
                lookups / total_s,
                total_s * 1e6 / lookups,
            );
        }
        let batch_hit_delta = engine
            .sharded_point_batch_hits()
            .saturating_sub(batch_hits_before);
        let gpu_probe_hit_delta = engine
            .sharded_point_gpu_probe_hits()
            .saturating_sub(gpu_probe_hits_before);
        let binary_hit_delta = engine
            .sharded_point_binary_route_hits()
            .saturating_sub(binary_hits_before);
        assert_eq!(
            batch_hit_delta, timed_calls as u64,
            "every measured batched call must be served exactly once by the batched route"
        );
        assert_eq!(
            gpu_probe_hit_delta, timed_calls as u64,
            "every measured batched call must execute exactly one GPU probe"
        );
        if require_binary {
            assert_eq!(
                binary_hit_delta, timed_calls as u64,
                "every measured many-shard call must execute exactly one labelled binary route"
            );
        } else {
            assert_eq!(
                binary_hit_delta, 0,
                "one-shard regimes must not be attributed to the binary route"
            );
        }
        println!(
            "measured-loop deltas: batched hits = {batch_hit_delta} | gpu-probe hits = \
             {gpu_probe_hit_delta} | BINARY-route hits = {binary_hit_delta}"
        );
    }

    println!(
        "\nREAD: batched lookups/s should CLIMB with batch size toward tens of millions/s (lpb-class) as the \
         per-needle SQL-parse + GPU-launch + DtoH is amortized over the batch, while single-flight stays ~44k/s. \
         The gate for the shards-default flip is batched throughput reaching the canonical lpb/wave point-read \
         card's production-compact order of magnitude (r2_wave_engine_ab)."
    );
    Ok(())
}
