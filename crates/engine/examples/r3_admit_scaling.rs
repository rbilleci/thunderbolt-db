//! R3 — measure where the SINGLE-BUFFER residency BREAKS, to ground the segmented shard size.
//!
//! For each row count N: load N rows (batched INSERT, untimed), then time ONE admit
//! (`populate_relational_residency_snapshot` = the full seq-scan + columnar rebuild + HtoD upload) and
//! record the resident device bytes. Admit is O(table); today a re-admit fires on EVERY headroom overflow
//! (the capacity-doubling), so this admit IS both the per-overflow stall and the per-table VRAM footprint.
//! The single-buffer append path also hard-declines at `row_count >= 1<<29` (~536M). Extrapolate
//! admit-us/row + bytes/row to 536M and to billions to pick a seal/rollover SHARD SIZE where (a) sealing
//! one shard (= admit one shard) stays cheap and (b) VRAM per shard fits, while the shard COUNT for a
//! billion-row table stays manageable for the per-shard read combine.
//!
//! Run: GPU_DB_BENCH_ADMIT_SCALING=1 cargo run --release --example r3_admit_scaling -p gpu_db_engine
//!      GPU_DB_BENCH_ROWS=250000,1000000,4000000,16000000 (default)

use std::env;
use std::error::Error;
use std::time::Instant;

use gpu_db_engine::Engine;

const INSERT_CHUNK: i64 = 1000;

fn main() -> Result<(), Box<dyn Error>> {
    if env::var("GPU_DB_BENCH_ADMIT_SCALING").ok().as_deref() != Some("1") {
        println!("set GPU_DB_BENCH_ADMIT_SCALING=1 to run (GPU-touching). Skipping.");
        return Ok(());
    }
    let rows: Vec<i64> = env::var("GPU_DB_BENCH_ROWS")
        .unwrap_or_else(|_| "250000,1000000,4000000,16000000".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("# R3 admit scaling (single-buffer residency) — accounts(id INT, balance INT), 2 int4 cols");
    println!();
    println!("| rows | load_s | admit_ms | admit_us_per_row | resident_MB | bytes_per_row |");
    println!("|---|---|---|---|---|---|");
    for &n in &rows {
        let mut engine = Engine::new_local();
        engine.set_auto_admit_on_commit(false);
        engine.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")?;

        let load_start = Instant::now();
        let mut txn = 2u64;
        let mut id = 0i64;
        while id < n {
            let mut vals = String::new();
            for _ in 0..INSERT_CHUNK {
                if id >= n {
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
        let load_s = load_start.elapsed().as_secs_f64();

        let admit_start = Instant::now();
        let snap = engine.populate_relational_residency_snapshot("accounts")?;
        let admit_ms = admit_start.elapsed().as_secs_f64() * 1e3;
        let bytes = snap.resident_bytes as f64;

        println!(
            "| {n} | {load_s:.1} | {admit_ms:.1} | {:.3} | {:.1} | {:.1} |",
            admit_ms * 1e3 / n as f64,
            bytes / 1e6,
            bytes / n as f64,
        );
    }
    println!();
    println!(
        "Caps: the incremental-append path declines at row_count >= 1<<29 (~536M); capacity = \
         next_pow2(2*rows) <= 1<<31 (so a single 2-col table tops out ~8.6 GB at the cap)."
    );
    println!(
        "A single re-admit at the cap is ~admit_us_per_row * 536M; billions is unreachable in one buffer. \
         Pick a shard size where one admit (seal) is cheap (e.g. < ~10 ms) AND 1B/shard_size shards stay \
         tractable for the read combine."
    );
    Ok(())
}
