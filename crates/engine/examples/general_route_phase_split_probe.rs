//! Cross-kernel gate (docs/optimizations/cross-kernel-result-path-transfer.md, step #1): localize the REAL
//! cap of the GENERAL non-unique row-producing route (`engine_resident_probe.rs:1100-1645`, the pre-lpb host
//! path) on a HIGH-OUTPUT query, to decide whether the flat-result-path transfer is worth it END TO END.
//!
//! A CPU micro (`result_materialization_probe` high_output) already showed the boxed ASSEMBLY for the
//! high-output shape is ~21-27M rows/s and the flat transfer is only ~1.4x there (vs 66-186x for the
//! point-read shape). But that micro models ONLY the assembly. This probe runs the REAL route on the GPU and
//! phase-splits: total engine WALL vs GPU KERNEL (`last_execution_kernel_event_elapsed_us`) — so we see
//! whether the cap is the SCAN KERNEL / DtoH (kernel ~ wall) or the HOST result path (wall >> kernel, the
//! stage the flat transfer targets).
//!
//! Drive: `SELECT id, value FROM events WHERE category = X` on a resident table where `category` has `CATS`
//! distinct values -> each query matches ~ROWS/CATS rows (sweep CATS to vary output size). Routes
//! `int4_equality_projection` -> `execute_resident_grouped_via_general` -> the general path.
//!
//! Run (RTX box; never `--gpu-reset`, always under `timeout`):
//!   timeout 240 cargo run --release --example general_route_phase_split_probe -p gpu_db_engine

use std::env;
use std::error::Error;
use std::time::Instant;

use gpu_db_engine::Engine;
use gpu_db_sql::{parse_command, Command, Select};

fn parse_select(sql: &str) -> Select {
    match parse_command(sql).expect("parse") {
        Command::Select(sel) => sel,
        _ => panic!("not a SELECT: {sql}"),
    }
}

fn dist(mut v: Vec<u128>) -> (u128, u128, u128) {
    if v.is_empty() {
        return (0, 0, 0);
    }
    v.sort_unstable();
    let at = |q: f64| v[((v.len() as f64 * q) as usize).min(v.len() - 1)];
    (at(0.50), at(0.99), v[v.len() - 1])
}

fn build_resident_engine(rows: i64, cats: i64) -> Result<Engine, Box<dyn Error>> {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, category INT, value INT)")?;
    let mut txn = 2u64;
    let mut id = 0i64;
    while id < rows {
        let mut vals = String::new();
        for _ in 0..1000 {
            if id >= rows {
                break;
            }
            if !vals.is_empty() {
                vals.push(',');
            }
            // category = id % cats  -> each category matches ~rows/cats rows (non-unique, high output)
            vals.push_str(&format!("({}, {}, {})", id, id % cats, (id * 7) % 100_000));
            id += 1;
        }
        e.execute_text(txn, &format!("INSERT INTO events (id, category, value) VALUES {vals}"))?;
        txn += 1;
    }
    e.populate_relational_residency_snapshot("events")?;
    Ok(e)
}

fn main() -> Result<(), Box<dyn Error>> {
    let rows: i64 =
        env::var("GPU_DB_BENCH_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(262_144);
    let iters: usize =
        env::var("GPU_DB_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    let cats_sweep: Vec<i64> = env::var("GPU_DB_BENCH_CATS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 16, 256, 4096]);

    println!("# GENERAL non-unique route phase-split. rows={rows}, iters={iters}");
    println!("# WALL = engine end-to-end; KERNEL = GPU event elapsed; HOST = WALL-KERNEL (the result path).");
    println!("# If HOST >> KERNEL -> the flat transfer's target stage is the cap. If KERNEL ~ WALL -> it isn't.");
    println!(
        "  {:>8} {:>9} {:>11} {:>11} {:>11} {:>11} {:>12}",
        "cats", "out_rows", "wall_us p50", "kern_us p50", "host_us p50", "wall p99", "rows/s(wall)"
    );

    for cats in cats_sweep {
        let t_build = Instant::now();
        let e = build_resident_engine(rows, cats)?;
        eprintln!("[cats={cats}] build {} ms", t_build.elapsed().as_millis());
        let probe = parse_select("SELECT id, value FROM events WHERE category = 0");
        if !e.plan_relational_resident_route(&probe).accepted {
            println!("  {cats:>8}  (resident route not accepted — no GPU?)");
            return Ok(());
        }
        // warmup
        let t_warm = Instant::now();
        for k in 0..8 {
            let sel = parse_select(&format!("SELECT id, value FROM events WHERE category = {}", k % cats));
            let _ = e.execute_relational_select(&sel)?;
        }
        eprintln!("[cats={cats}] warmup(20) {} ms", t_warm.elapsed().as_millis());
        let mut wall = Vec::with_capacity(iters);
        let mut kern = Vec::with_capacity(iters);
        let mut host = Vec::with_capacity(iters);
        let mut out_rows = 0usize;
        for k in 0..iters {
            let sel = parse_select(&format!(
                "SELECT id, value FROM events WHERE category = {}",
                (k as i64) % cats
            ));
            let t = Instant::now();
            let r = e.execute_relational_select(&sel)?;
            let w = t.elapsed().as_micros();
            out_rows = r.rows.len();
            if k == 0 {
                eprintln!(
                    "[cats={cats}] planned={:?} executed={:?} fallback={:?}",
                    r.planned_target, r.executed_target, r.fallback_reason
                );
            }
            let status = e.plan_relational_resident_route(&sel);
            let ku = status.last_execution_kernel_event_elapsed_us.unwrap_or(0) as u128;
            wall.push(w);
            kern.push(ku);
            host.push(w.saturating_sub(ku));
        }
        let (w50, _w99x, _) = dist(wall.clone());
        let (k50, _, _) = dist(kern);
        let (h50, _, _) = dist(host);
        let (_, w99, _) = dist(wall);
        let rows_per_s = if w50 > 0 { (out_rows as f64) * 1.0e6 / (w50 as f64) } else { 0.0 };
        println!(
            "  {cats:>8} {out_rows:>9} {w50:>11} {k50:>11} {h50:>11} {w99:>11} {rows_per_s:>12.0}"
        );
    }
    Ok(())
}
