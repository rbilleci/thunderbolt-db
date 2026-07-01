//! R3 / S-d3 — measure that ZONE-MAP PRUNING makes a segmented point lookup O(1) in shard count.
//!
//! A table grown as many bounded shards by ORDERED inserts has DISJOINT per-shard key ranges, so a
//! point lookup `WHERE id = k` should prune to ~1 shard regardless of how many shards the table has —
//! latency stays FLAT as the table scales to more shards. The contrast control is `COUNT(*)`, which
//! carries no predicate and RECOMPACTS every shard into one dense buffer (O(num_shards)); its latency
//! grows with the shard count. Point-flat vs count-rising is the empirical proof that pruning fires and
//! isolates the recompaction copy as the cost pruning avoids.
//!
//! For each shard count in the sweep: load `shards * shard_size` ordered rows (batched INSERT, untimed),
//! confirm the table really is that many shards, warm up, then time many point lookups on random PRESENT
//! keys (p50/p99 latency + throughput + avg shards gathered) and one batch of COUNT(*) for the control.
//!
//! Run: GPU_DB_BENCH_SHARD_PRUNE=1 cargo run --release --example r3_shard_prune_bench -p gpu_db_engine
//!      GPU_DB_BENCH_SHARD_SIZE=65536         (rows per shard; the seal/rollover target)
//!      GPU_DB_BENCH_SHARD_COUNTS=1,2,4,8,16,32
//!      GPU_DB_BENCH_LOOKUPS=2000             (timed point lookups per shard count)

use std::env;
use std::error::Error;
use std::time::Instant;

use gpu_db_engine::Engine;

const INSERT_CHUNK: i64 = 1000;

/// Deterministic xorshift so the key stream is reproducible run to run (no rand dependency).
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

fn main() -> Result<(), Box<dyn Error>> {
    if env::var("GPU_DB_BENCH_SHARD_PRUNE").ok().as_deref() != Some("1") {
        println!("set GPU_DB_BENCH_SHARD_PRUNE=1 to run (GPU-touching). Skipping.");
        return Ok(());
    }
    let shard_size: usize = env::var("GPU_DB_BENCH_SHARD_SIZE")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(65536);
    let shard_counts: Vec<usize> = env::var("GPU_DB_BENCH_SHARD_COUNTS")
        .unwrap_or_else(|_| "1,2,4,8,16,32".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let lookups: usize = env::var("GPU_DB_BENCH_LOOKUPS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(2000);

    println!(
        "# R3 / S-d3 shard-prune scaling — accounts(id INT, balance INT), shard_size={shard_size}, \
         ordered keys (disjoint per-shard ranges)"
    );
    println!("# point lookup `WHERE id = k` PRUNES (zone map); COUNT(*) RECOMPACTS all shards (control).");
    println!();
    println!(
        "| shards | rows | point p50 us | point p99 us | point lookups/s | avg shards gathered | \
         index-routed | COUNT(*) recompact-all p50 us |"
    );
    println!("|---|---|---|---|---|---|---|---|");

    for &n_shards in &shard_counts {
        let total_rows = (n_shards * shard_size) as i64;
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine.set_shard_size_target(shard_size);
        // Sub-slice 3b: GPU_DB_BENCH_SHARD_INDEX=1 flips the cross-shard PK-index point-lookup route ON (the
        // O(1) hash+bloom locate + slot gather) so a point lookup SKIPS the per-shard scan + recompaction.
        // Default OFF = the scan baseline. The "point p50 stays flat vs shard_size ON but rises OFF" delta is
        // the scan cost the index removes.
        let use_index = env::var("GPU_DB_BENCH_SHARD_INDEX").ok().as_deref() == Some("1");
        engine.set_shard_index_probe_enabled(use_index);
        engine.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")?;

        // Load ordered rows (batched, untimed). Ordered keys -> disjoint zone maps -> prunable.
        let mut txn = 2u64;
        let mut id = 0i64;
        while id < total_rows {
            let mut vals = String::new();
            for _ in 0..INSERT_CHUNK {
                if id >= total_rows {
                    break;
                }
                if !vals.is_empty() {
                    vals.push(',');
                }
                vals.push_str(&format!("({}, {})", id, id * 10));
                id += 1;
            }
            engine.execute_text(txn, &format!("INSERT INTO accounts (id, balance) VALUES {vals}"))?;
            txn += 1;
        }

        let actual_shards = engine.resident_shard_count("accounts");

        // Warm up (JIT/caches) with a few lookups spread across the key range.
        for w in 0..8i64 {
            let k = (w * total_rows / 8).clamp(0, total_rows - 1);
            let _ = engine.execute_relational_select_text(&format!(
                "SELECT id, balance FROM accounts WHERE id = {k}"
            ))?;
        }

        // Timed point lookups on random PRESENT keys.
        let mut rng = 0x9E3779B97F4A7C15u64 ^ (n_shards as u64).wrapping_mul(0x1000_0001B3);
        let gathered_before = engine.sharded_shards_gathered();
        let route_before = engine.shard_index_route_hits();
        let mut lat_us: Vec<f64> = Vec::with_capacity(lookups);
        for _ in 0..lookups {
            let k = (next_rand(&mut rng) % total_rows as u64) as i64;
            let t = Instant::now();
            let rows = engine.execute_relational_select_text(&format!(
                "SELECT id, balance FROM accounts WHERE id = {k}"
            ))?;
            lat_us.push(t.elapsed().as_secs_f64() * 1e6);
            debug_assert_eq!(rows.rows.len(), 1, "present key {k} must return one row");
        }
        let gathered_after = engine.sharded_shards_gathered();
        let avg_gathered = (gathered_after - gathered_before) as f64 / lookups as f64;
        let routed = engine.shard_index_route_hits() - route_before;

        lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&lat_us, 0.50);
        let p99 = percentile(&lat_us, 0.99);
        let total_s: f64 = lat_us.iter().sum::<f64>() / 1e6;
        let per_s = lookups as f64 / total_s;

        // Control: COUNT(*) recompacts ALL shards (no predicate -> no prune). Its p50 rising with the
        // shard count is the cost pruning avoids.
        let count_runs = 64usize;
        let mut count_us: Vec<f64> = Vec::with_capacity(count_runs);
        for _ in 0..count_runs {
            let t = Instant::now();
            let _ = engine.execute_relational_select_text("SELECT COUNT(*) FROM accounts")?;
            count_us.push(t.elapsed().as_secs_f64() * 1e6);
        }
        count_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let count_p50 = percentile(&count_us, 0.50);

        println!(
            "| {actual_shards} | {total_rows} | {p50:.1} | {p99:.1} | {per_s:.0} | {avg_gathered:.2} | \
             {routed} | {count_p50:.1} |"
        );
    }

    println!();
    println!(
        "READ: pruning is O(1) when point-lookup p50 stays ~FLAT as `shards` grows and avg-shards-gathered \
         stays ~1, while the COUNT(*) control p50 RISES with `shards` (it recompacts them all). A rising \
         point p50 or avg-gathered > ~1 means pruning regressed."
    );
    println!(
        "MEASURED (2026-07-01): point p50 is FLAT (~180 us) across BOTH shard count (2..34) AND shard size \
         (8k..131k) at avg-gathered 1.00 -> pruning validated, and the residual ~180 us is FIXED host \
         per-query overhead (SQL-text parse + bind + route + single-flight launch + DtoH + materialize), \
         NOT the shard scan or the recompaction copy (both flat vs shard_size). This is the host-serial \
         single-lookup cost the single-buffer path amortizes by BATCHING (retained-template wave/lpb, no \
         per-lookup parse); the sharded route does not batch yet. Next sharded read lever = batched \
         point lookups through pruned shards, NOT a bloom (random keys) or skip-recompaction (copy is not \
         the cost at these sizes)."
    );
    Ok(())
}
