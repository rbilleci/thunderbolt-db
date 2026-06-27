//! R1 wave-index A/B — the end-to-end O(rows)→O(1) measurement (ADR-009 R1).
//!
//! R1 wired the GPU hash-index probe into the resident int4 unique-key point-lookup route behind the
//! default-OFF `wave_engine_enabled` flag (`submit_resident_int4_equal_any_payload`). The standalone
//! probe (`crates/execution/examples/wave_index_probe.rs`) already showed the index is O(1) and the
//! scan is O(rows) at the data-plane level; this measures the SAME swap **through the engine** — the
//! real retained-read template path the production batcher drives — so the win is end-to-end, results
//! materialized to `Vec<SqlValue>` and all.
//!
//! For each table size we run identical batches of DISTINCT point lookups twice: flag OFF (full-scan
//! `equal_any`, O(rows) per batch) and flag ON (GPU hash-index probe, O(1) per needle). Same needles,
//! same template, same result materialization — the only difference is the device route. We assert the
//! two routes return byte-identical rows (a silent wrong/empty index can't masquerade as a win), then
//! report per-batch latency + aggregate ops/s + the ON/OFF speedup.
//!
//! **The proof is the curve, not a single number:** a full scan CANNOT be flat across table sizes, so
//! if ON throughput stays ~flat while OFF degrades ~linearly with row count, the index genuinely fired
//! and the per-batch cost dropped O(rows)→O(1). The batcher stays the production default until this
//! path wins; this benchmark is the instrument that decides that.
//!
//! Env: `GPU_DB_BENCH_ROWS` (comma-separated table sizes, default "262144,1048576,4194304"),
//! `GPU_DB_BENCH_BATCH` (distinct needles per submission, default 256 — one GPU thread per needle, no
//! needle-count cap; the index's 256 is an unrelated per-key linear-probe cap in the build),
//! `GPU_DB_BENCH_BATCHES` (measured batches per mode, default 1000),
//! `GPU_DB_BENCH_WARMUP` (untimed warmup batches per mode, default 5 — first ON batch builds the index).
//!
//! Run (RTX box; never `--gpu-reset`, always under `timeout`):
//!   timeout 600 cargo run --release --example r1_wave_index_ab -p gpu_db_engine
//!   GPU_DB_BENCH_ROWS=16777216 timeout 900 cargo run --release --example r1_wave_index_ab -p gpu_db_engine

use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

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

struct Lat {
    p50: u128,
    p99: u128,
    max: u128,
    ops_per_s: f64,
    per_lookup_us: f64,
}

/// Percentiles over per-batch micros; ops/s and per-lookup cost over the whole timed run.
fn summarize(mut batch_micros: Vec<u128>, total_lookups: usize, wall: Duration) -> Lat {
    batch_micros.sort_unstable();
    let n = batch_micros.len();
    let pct = |p: f64| -> u128 {
        if n == 0 {
            return 0;
        }
        let rank = ((n as f64 * p).ceil() as usize).clamp(1, n);
        batch_micros[rank - 1]
    };
    let secs = wall.as_secs_f64();
    Lat {
        p50: pct(0.50),
        p99: pct(0.99),
        max: *batch_micros.last().unwrap_or(&0),
        ops_per_s: if secs == 0.0 {
            0.0
        } else {
            total_lookups as f64 / secs
        },
        per_lookup_us: if total_lookups == 0 {
            0.0
        } else {
            wall.as_micros() as f64 / total_lookups as f64
        },
    }
}

fn print_lat(label: &str, l: &Lat) {
    println!(
        "  {label:<18} batch p50={:>7}us p99={:>7}us max={:>7}us | {:>12.0} lookups/s ({:>6.3} us/lookup)",
        l.p50, l.p99, l.max, l.ops_per_s, l.per_lookup_us
    );
}

/// Load `rows` accounts (id 0..rows-1, balance derived) and make the table GPU-resident.
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
        e.execute_text(
            txn,
            &format!("INSERT INTO accounts (id, balance) VALUES {vals}"),
        )?;
        txn += 1;
    }
    e.populate_relational_residency_snapshot("accounts")?;
    Ok(e)
}

/// Deterministic distinct needles for batch `b`: `(g*step) % rows` over `batch` consecutive `g`.
/// `step` coprime to `rows` ⇒ the consecutive g's map to distinct residues ⇒ distinct needles (the
/// index route requires distinct needles; the batcher's `dedup_needles` guarantees it in production).
/// Indexing by `b` (not a rolling cursor) makes OFF batch b and ON batch b issue the SAME lookups.
fn needles_for_batch(b: usize, batch: usize, step: u64, rows: u64) -> Vec<i32> {
    let mut v = Vec::with_capacity(batch);
    for k in 0..batch {
        let g = (b * batch + k) as u64;
        v.push((g.wrapping_mul(step) % rows) as i32);
    }
    v
}

fn measure(
    e: &Engine,
    template: &gpu_db_engine::RelationalRetainedReadTemplate,
    flag: bool,
    batch: usize,
    batches: usize,
    warmup: usize,
    step: u64,
    rows: u64,
) -> Result<Lat, Box<dyn Error>> {
    e.set_wave_engine_enabled(flag);
    // Warmup: the first ON batch builds + caches the index (DtoH key read + hash); also JIT/allocator.
    for b in 0..warmup {
        let needles = needles_for_batch(b, batch, step, rows);
        let sub = e.submit_relational_retained_template_point_lookups(template, &needles)?;
        let _ = e.complete_relational_retained_read_submission(sub)?;
    }
    let mut batch_micros = Vec::with_capacity(batches);
    let t = Instant::now();
    for b in 0..batches {
        let needles = needles_for_batch(b, batch, step, rows);
        let s0 = Instant::now();
        let sub = e.submit_relational_retained_template_point_lookups(template, &needles)?;
        let _ = e.complete_relational_retained_read_submission(sub)?;
        batch_micros.push(s0.elapsed().as_micros());
    }
    Ok(summarize(batch_micros, batches * batch, t.elapsed()))
}

fn parse_sizes(s: &str) -> Vec<i64> {
    s.split(',')
        .filter_map(|tok| tok.trim().parse::<i64>().ok())
        .filter(|&r| r > 0)
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let sizes = parse_sizes(
        &env::var("GPU_DB_BENCH_ROWS").unwrap_or_else(|_| "262144,1048576,4194304".to_string()),
    );
    let batch: usize = env::var("GPU_DB_BENCH_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let batches: usize = env::var("GPU_DB_BENCH_BATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let warmup: usize = env::var("GPU_DB_BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    println!(
        "# R1 wave-index A/B  sizes={sizes:?}  batch={batch}  batches={batches}  warmup={warmup}"
    );
    println!("# OFF = full-scan equal_any (O(rows)/batch);  ON = GPU hash-index probe (O(1)/needle)\n");

    // (rows, off_ops, on_ops) for the closing O(rows)→O(1) summary.
    let mut headline: Vec<(i64, f64, f64)> = Vec::new();

    for &rows in &sizes {
        let batch = batch.min(rows as usize); // need >=`batch` distinct needles
        let rows_u = rows as u64;
        // step coprime to rows so the needle walk is a permutation (distinct within a batch).
        let mut step = 0x9E37_79B1u64 % rows_u.max(2);
        if step == 0 {
            step = 1;
        }
        while gcd(step, rows_u) != 1 {
            step = (step % rows_u) + 1;
        }

        let e = build_resident_engine(rows)?;
        let select = parse_select("SELECT id, balance FROM accounts WHERE id = 1");

        // Gate: bail cleanly if there's no GPU resident route (e.g. CI without a GPU).
        if !e.plan_relational_resident_route(&select).accepted {
            println!("rows={rows}: no GPU resident route accepted — skipping");
            continue;
        }
        let exec_target = format!(
            "{:?}",
            e.execute_relational_select(&select)?.executed_target
        );
        let template = e.prepare_relational_retained_read_template(&select)?;

        // Correctness: OFF and ON must return byte-identical rows (catches a silent wrong/empty index
        // before any timing claim). Same needles, both routes.
        let probe = needles_for_batch(0, batch, step, rows_u);
        e.set_wave_engine_enabled(false);
        let off_rows: Vec<_> = e
            .complete_relational_retained_read_submission(
                e.submit_relational_retained_template_point_lookups(&template, &probe)?,
            )?
            .into_iter()
            .map(|r| r.rows)
            .collect();
        e.set_wave_engine_enabled(true);
        let on_rows: Vec<_> = e
            .complete_relational_retained_read_submission(
                e.submit_relational_retained_template_point_lookups(&template, &probe)?,
            )?
            .into_iter()
            .map(|r| r.rows)
            .collect();
        assert_eq!(
            on_rows, off_rows,
            "rows={rows}: index route (ON) must return byte-identical rows to the scan (OFF)"
        );

        let off = measure(&e, &template, false, batch, batches, warmup, step, rows_u)?;
        let on = measure(&e, &template, true, batch, batches, warmup, step, rows_u)?;
        let speedup = if off.ops_per_s > 0.0 {
            on.ops_per_s / off.ops_per_s
        } else {
            0.0
        };

        println!("## rows={rows}  (executed_target={exec_target}, batch={batch})");
        print_lat("OFF scan", &off);
        print_lat("ON  index", &on);
        println!("  speedup (ON/OFF lookups/s): {speedup:.2}x\n");
        headline.push((rows, off.ops_per_s, on.ops_per_s));
    }

    if headline.len() >= 2 {
        println!("## O(rows)→O(1) summary  (lookups/s vs table size)");
        println!(
            "  {:>12}  {:>14}  {:>14}  {:>10}",
            "rows", "OFF scan", "ON index", "speedup"
        );
        for (rows, off_ops, on_ops) in &headline {
            let sp = if *off_ops > 0.0 { on_ops / off_ops } else { 0.0 };
            println!("  {rows:>12}  {off_ops:>14.0}  {on_ops:>14.0}  {sp:>9.2}x");
        }
        let (r0, off0, on0) = headline[0];
        let (rn, offn, onn) = *headline.last().unwrap();
        let size_growth = rn as f64 / r0 as f64;
        let scan_falloff = if offn > 0.0 { off0 / offn } else { 0.0 };
        let index_falloff = if onn > 0.0 { on0 / onn } else { 0.0 };
        println!(
            "\n  table grew {size_growth:.0}x:  scan lookups/s fell {scan_falloff:.1}x (≈O(rows)),  index fell {index_falloff:.2}x (≈flat ⇒ O(1))"
        );
    }
    Ok(())
}
