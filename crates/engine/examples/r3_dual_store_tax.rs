//! R3 — quantify the DUAL-STORE TAX (the cost the "host = control plane, data plane on GPU" target
//! removes).
//!
//! Today the host MVCC store is the source of truth and the GPU-resident copy is a DERIVED snapshot.
//! `auto_admit_resident_tables` (fired on every commit when auto-admit is on) re-runs
//! `populate_relational_residency_snapshot_inner`, which seq-scans EVERY row of the mutated table,
//! re-decodes them, rebuilds the whole columnar device payload, and re-uploads it to the GPU. So a
//! single-row INSERT on a resident table re-materializes + re-uploads the ENTIRE table — O(table) per
//! commit, O(n^2) over a load.
//!
//! This measures that tax: per resident base size S, load S rows (auto-admit OFF, batched — so the base
//! load itself doesn't pay the tax), populate residency ONCE, enable auto-admit, then time a handful of
//! single-row INSERTs (each pays a full re-admit). Reports per-insert latency vs S, and vs a non-resident
//! control (the ~O(1) ~29 us/row path). GPU-touching: small sizes only.
//!
//! Run: GPU_DB_BENCH_DUAL_STORE=1 cargo run --release --example r3_dual_store_tax -p gpu_db_engine
//!      GPU_DB_BENCH_BASES=1000,4000,16000 GPU_DB_BENCH_TIMED_INSERTS=10 (defaults shown)

use std::env;
use std::error::Error;
use std::time::Instant;

use gpu_db_engine::Engine;

const INSERT_CHUNK: i64 = 1000; // rows per INSERT statement for the (untimed) base load

fn main() -> Result<(), Box<dyn Error>> {
    if env::var("GPU_DB_BENCH_DUAL_STORE").ok().as_deref() != Some("1") {
        println!("set GPU_DB_BENCH_DUAL_STORE=1 to run (GPU-touching). Skipping.");
        return Ok(());
    }
    let bases: Vec<i64> = env::var("GPU_DB_BENCH_BASES")
        .unwrap_or_else(|_| "1000,4000,16000".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let timed: usize = env::var("GPU_DB_BENCH_TIMED_INSERTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    println!("# R3 Dual-Store Tax (single-row INSERT on a GPU-resident table)");
    println!();
    println!("- table: accounts(id INT, balance INT)");
    println!("- timed_inserts_per_base: {timed}");
    println!();

    // Non-resident control: per-insert cost with NO residency (the O(1) ~29 us/row path).
    let control_us = measure_inserts(0, timed, false)?;
    println!("## control (no residency, auto-admit OFF)");
    println!("- per_insert_mean_us: {:.1}", control_us);
    println!();

    println!("## resident (auto-admit ON — each commit re-admits the whole table)");
    println!("| base_rows | initial_admit_ms | per_insert_mean_us | per_insert_max_us | tax_vs_control |");
    println!("|---|---|---|---|---|");
    for &base in &bases {
        let (admit_ms, per_insert_us, per_insert_max_us) = measure_resident(base, timed)?;
        let tax = if control_us > 0.0 {
            per_insert_us / control_us
        } else {
            0.0
        };
        println!(
            "| {base} | {admit_ms:.1} | {per_insert_us:.1} | {per_insert_max_us:.1} | {tax:.0}x |"
        );
    }
    println!();
    println!("Reading: per-insert cost grows ~linearly with base_rows -> each single-row commit");
    println!("re-uploads the whole table. This is the dual-store tax GPU-native incremental writes remove.");
    Ok(())
}

/// Load `base` rows (auto-admit OFF, batched — untimed), then time `timed` single-row INSERTs with
/// auto-admit ON so each commit triggers a full re-admit. Returns (initial_admit_ms, mean_us, max_us).
fn measure_resident(base: i64, timed: usize) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let mut engine = Engine::new_local();
    engine.set_auto_admit_on_commit(false);
    engine.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")?;
    let mut txn = 2u64;
    let mut id = 0i64;
    while id < base {
        let mut vals = String::new();
        for _ in 0..INSERT_CHUNK {
            if id >= base {
                break;
            }
            if !vals.is_empty() {
                vals.push(',');
            }
            vals.push_str(&format!("({}, {})", id, (id * 7) % 100_000));
            id += 1;
        }
        engine.execute_text(txn, &format!("INSERT INTO accounts (id, balance) VALUES {vals}"))?;
        txn += 1;
    }

    // One initial resident upload (this is the legitimate, one-time admit cost).
    let admit_start = Instant::now();
    engine.populate_relational_residency_snapshot("accounts")?;
    let admit_ms = admit_start.elapsed().as_secs_f64() * 1e3;

    // Now every commit re-admits the whole table.
    engine.set_auto_admit_on_commit(true);
    let mut samples_us = Vec::with_capacity(timed);
    for i in 0..timed {
        let id = base + i as i64;
        let sql = format!("INSERT INTO accounts (id, balance) VALUES ({id}, {id})");
        let start = Instant::now();
        engine.execute_text(txn, &sql)?;
        txn += 1;
        samples_us.push(start.elapsed().as_secs_f64() * 1e6);
    }
    let mean = samples_us.iter().sum::<f64>() / samples_us.len().max(1) as f64;
    let max = samples_us.iter().cloned().fold(0.0_f64, f64::max);
    Ok((admit_ms, mean, max))
}

/// `timed` single-row INSERTs with no residency and auto-admit OFF (the O(1) control). Mean us/insert.
fn measure_inserts(_base: i64, timed: usize, auto_admit: bool) -> Result<f64, Box<dyn Error>> {
    let engine = Engine::new_local();
    engine.set_auto_admit_on_commit(auto_admit);
    engine.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")?;
    let mut samples_us = Vec::with_capacity(timed);
    for i in 0..timed {
        let id = i as i64;
        let sql = format!("INSERT INTO accounts (id, balance) VALUES ({id}, {id})");
        let start = Instant::now();
        engine.execute_text((i as u64) + 2, &sql)?;
        samples_us.push(start.elapsed().as_secs_f64() * 1e6);
    }
    Ok(samples_us.iter().sum::<f64>() / samples_us.len().max(1) as f64)
}
