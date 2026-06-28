//! R2.2c host-machinery spike — localize lpb's per-batch cost (submit vs complete) on the REAL engine
//! path, to decide whether host-machinery optimization can close the lpb->wave gap without the wave.
//!
//! The R2.2b-3 A/B + the graphs spike showed: lpb's ~27us/batch is NOT raw launch (~7us) — it is per-batch
//! ENGINE machinery. Reading `submit_cuda_resident_i32_index_probe` + `complete_detached`: the lpb COMPLETE
//! does TWO sync round-trips (DtoH the match count to size result arrays -> sync, THEN DtoH the 3 result
//! arrays -> sync), plus 5 per-batch device-buffer leases + event timing; the WAVE does ONE round-trip (its
//! per-slot status ring is host-mapped, so no count DtoH, then one bulk DtoH). So the suspected lever is
//! COLLAPSING lpb's two round-trips into one (buffers are ALREADY pooled, so pooling is not the lever).
//!
//! This probe TIMES, on the real retained-template path, the SUBMIT phase vs the COMPLETE phase separately,
//! for lpb (wave_engine_enabled) and wave (both flags), so we see WHERE each spends its per-batch time. NO
//! production change. (Faithful: same engine path the A/B drives; non-vacuity via wave_route_hits.)
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

    println!("# R2.2c host-machinery spike — lpb SUBMIT vs COMPLETE phase split (real engine path)");
    println!("# lpb COMPLETE does 2 sync round-trips (count, then results); wave does 1. rows={rows}\n");
    println!(
        "  {:>5}  {:>5}  {:>14}  {:>15}  {:>13}  {:>10}",
        "mode", "batch", "submit p50 us", "complete p50 us", "total p50 us", "route_hits"
    );

    for &batch in &batch_sizes {
        for (label, wave_engine, persistent) in
            [("lpb", true, false), ("wave", true, true)]
        {
            e.set_wave_engine_enabled(wave_engine);
            e.set_wave_persistent_engine_enabled(persistent);
            // warmup (first wave batch builds the engine; lpb builds the index)
            for b in 0..30 {
                let n = needles_for_batch(b, batch, step, rows_u);
                let sub = e.submit_relational_retained_template_point_lookups(&template, &n)?;
                let _ = e.complete_relational_retained_read_submission(sub)?;
            }
            let hits_before = e.wave_route_hits();
            let mut submit_us = Vec::with_capacity(batches);
            let mut complete_us = Vec::with_capacity(batches);
            let mut total_us = Vec::with_capacity(batches);
            for b in 0..batches {
                let n = needles_for_batch(b, batch, step, rows_u);
                let t0 = Instant::now();
                let sub = e.submit_relational_retained_template_point_lookups(&template, &n)?;
                let s = t0.elapsed().as_micros();
                let t1 = Instant::now();
                let _ = e.complete_relational_retained_read_submission(sub)?;
                let c = t1.elapsed().as_micros();
                submit_us.push(s);
                complete_us.push(c);
                total_us.push(s + c);
            }
            let hits = e.wave_route_hits() - hits_before;
            // non-vacuity: wave must have served every batch; lpb never hits the wave route.
            let expect = if persistent { batches as u64 } else { 0 };
            assert_eq!(hits, expect, "{label}: wave_route_hits {hits} != {expect}");
            println!(
                "  {label:>5}  {batch:>5}  {:>14}  {:>15}  {:>13}  {:>10}",
                p50(submit_us),
                p50(complete_us),
                p50(total_us),
                hits
            );
        }
    }
    println!("\n# If lpb COMPLETE p50 >> wave COMPLETE p50 (the 2nd round-trip), collapsing lpb's count+result");
    println!("# DtoH into ONE covering sync (over-fetch results to needles.len, trim by count) is the lever.");
    Ok(())
}
