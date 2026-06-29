//! lpb host-machinery spike — localize lpb's per-batch cost (submit vs complete) on the REAL engine path,
//! to decide whether host-machinery optimization can shrink the lpb per-batch overhead.
//!
//! lpb's ~27us/batch is NOT raw launch (~7us) — it is per-batch ENGINE machinery. Reading
//! `submit_cuda_resident_i32_index_probe` + `complete_detached`: the lpb COMPLETE does TWO sync round-trips
//! (DtoH the match count to size result arrays -> sync, THEN DtoH the 3 result arrays -> sync), plus 5
//! per-batch device-buffer leases + event timing. So the suspected lever is COLLAPSING lpb's two round-trips
//! into one (buffers are ALREADY pooled, so pooling is not the lever). The DENSE kernel sidesteps the count
//! round-trip (one slot per needle, host-compacted), so the lpb-vs-lpb-dense split localizes that cost.
//!
//! This probe TIMES, on the real retained-template path, the SUBMIT phase vs the COMPLETE phase separately,
//! for lpb (atomic) and lpb-dense, so we see WHERE each spends its per-batch time. NO production change.
//! (Faithful: same engine path; non-vacuity via dense_index_probe_hits.)
//!
//! Run (RTX box; never `--gpu-reset`, always under `timeout`):
//!   timeout 240 cargo run --release --example lpb_phase_split_probe -p gpu_db_engine

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

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn build_resident_engine(rows: i64) -> Result<Engine, Box<dyn Error>> {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")?;
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
            vals.push_str(&format!("({}, {})", id, (id * 7) % 100_000));
            id += 1;
        }
        e.execute_text(txn, &format!("INSERT INTO accounts (id, balance) VALUES {vals}"))?;
        txn += 1;
    }
    e.populate_relational_residency_snapshot("accounts")?;
    Ok(e)
}

fn needles_for_batch(b: usize, batch: usize, step: u64, rows: u64) -> Vec<i32> {
    (0..batch)
        .map(|k| {
            let g = (b * batch + k) as u64;
            (g.wrapping_mul(step) % rows) as i32
        })
        .collect()
}

fn p50(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    if v.is_empty() {
        0
    } else {
        v[v.len() / 2]
    }
}

/// (p50, p99, p99.9, max) — for localizing the TAIL, not just the median.
fn dist(mut v: Vec<u128>) -> (u128, u128, u128, u128) {
    if v.is_empty() {
        return (0, 0, 0, 0);
    }
    v.sort_unstable();
    let at = |q: f64| v[((v.len() as f64 * q) as usize).min(v.len() - 1)];
    (at(0.50), at(0.99), at(0.999), v[v.len() - 1])
}

fn main() -> Result<(), Box<dyn Error>> {
    let rows: i64 = env::var("GPU_DB_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_048_576);
    let batches: usize = env::var("GPU_DB_BENCH_BATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000);
    let batch_sizes: Vec<usize> = env::var("GPU_DB_BENCH_BATCH")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![1, 32]);

    let rows_u = rows as u64;
    let mut step = 0x9E37_79B1u64 % rows_u.max(2);
    if step == 0 {
        step = 1;
    }
    while gcd(step, rows_u) != 1 {
        step = (step % rows_u) + 1;
    }

    let e = build_resident_engine(rows)?;
    let select = parse_select("SELECT id, balance FROM accounts WHERE id = 1");
    if !e.plan_relational_resident_route(&select).accepted {
        println!("rows={rows}: no GPU resident route — skipping (no GPU?)");
        return Ok(());
    }
    let template = e.prepare_relational_retained_read_template(&select)?;

    println!("# TAIL-LOCALIZATION — lpb vs lpb-dense SUBMIT/COMPLETE distribution on the BATCHED path. rows={rows}");
    println!("# Goal: find WHERE the lpb p99 tail lives (submit vs complete). us = microseconds.\n");
    println!(
        "  {:>5} {:>6}  {:>26}  {:>26}",
        "mode", "batch", "submit p50/p99/p99.9/max", "complete p50/p99/p99.9/max"
    );

    for &batch in &batch_sizes {
        for (label, wave_engine, dense) in [
            ("lpb", true, false),
            ("lpb-dense", true, true),
        ] {
            e.set_index_probe_enabled(wave_engine);
            e.set_dense_index_probe_enabled(dense);
            // warmup (lpb builds the index)
            for b in 0..30 {
                let n = needles_for_batch(b, batch, step, rows_u);
                let sub = e.submit_relational_retained_template_point_lookups(&template, &n)?;
                let _ = e.complete_relational_retained_read_submission_batched(sub)?;
            }
            let dense_before = e.dense_index_probe_hits();
            let mut submit_us = Vec::with_capacity(batches);
            let mut complete_us = Vec::with_capacity(batches);
            for b in 0..batches {
                let n = needles_for_batch(b, batch, step, rows_u);
                let t0 = Instant::now();
                let sub = e.submit_relational_retained_template_point_lookups(&template, &n)?;
                let s = t0.elapsed().as_micros();
                let t1 = Instant::now();
                let _ = e.complete_relational_retained_read_submission_batched(sub)?;
                let c = t1.elapsed().as_micros();
                submit_us.push(s);
                complete_us.push(c);
            }
            // non-vacuity: dense must have served every dense batch (the dense kernel actually ran, no
            // silent fallback).
            let dense_hits = e.dense_index_probe_hits() - dense_before;
            let dense_expect = if dense { batches as u64 } else { 0 };
            assert_eq!(
                dense_hits, dense_expect,
                "{label}: dense_index_probe_hits {dense_hits} != {dense_expect}"
            );
            e.set_dense_index_probe_enabled(false);
            let (s50, s99, s999, smax) = dist(submit_us);
            let (c50, c99, c999, cmax) = dist(complete_us);
            println!(
                "  {label:>5} {batch:>6}  {:>8}/{:>5}/{:>5}/{:>5}  {:>8}/{:>5}/{:>5}/{:>6}",
                s50, s99, s999, smax, c50, c99, c999, cmax
            );
        }
    }
    let _ = p50(Vec::new()); // (p50 retained for back-compat; dist() drives the tail report)
    println!("\n# If the lpb COMPLETE max/p99.9 >> its p50, the tail is in lpb's blocking cuStreamSynchronize");
    println!("# drain (shared-GPU contention) — localize drain-vs-sync next.");
    Ok(())
}
