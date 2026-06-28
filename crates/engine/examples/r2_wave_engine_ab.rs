//! R2.2b-3 wave-engine A/B — the SHIP-decision instrument (ADR-009 R2.2b).
//!
//! R2.2b-2 wired the PERSISTENT `WaveReadEngine` into the resident int4 unique-key point-lookup route
//! behind the default-OFF `wave_persistent_engine_enabled` flag (NESTED under `wave_engine_enabled`,
//! which stays the lpb default). This benchmark measures the SAME swap **through the engine** — the real
//! retained-read template path the production batcher drives — across THREE byte-identical routes:
//!   - `scan`      : both flags off                         -> full-scan equal_any  (O(rows)/batch)
//!   - `lpb-index` : wave_engine_enabled only               -> launch-per-batch GPU hash-index (O(1)/needle)
//!   - `wave`      : both flags                              -> persistent kernel    (no per-batch launch)
//!
//! WHAT THIS MEASURES (and what it does NOT). The wired wave route is SINGLE-FLIGHT: one `WaveReadEngine`
//! per (table,proj), submits serialized by its `Arc<Mutex<_>>`. In production, point reads flow through
//! the facade's SINGLE coalescer thread, so the production-relevant regime is ONE caller varying BATCH
//! size (a higher offered rate coalesces into bigger batches). That is the PRIMARY sweep below. We ALSO
//! run a concurrent section (N threads hammering the same shape through one shared `Engine`) — for the
//! wave that contends the per-engine Mutex (the single-flight ceiling), for scan/lpb the GPU pipelines
//! per-batch launches. The concurrent numbers expose the Mutex ceiling that motivates R2.2c (a
//! multi-producer lock-free-ring replacement of the coalescer), and are NOT the production path.
//!
//! NON-VACUITY (the R2.2 "loses 118x" retraction lesson — wrong regime + wrong baseline + crippled impl):
//!   (1) every route is asserted BYTE-IDENTICAL before any timing (a silent wrong/empty result can't win);
//!   (2) `Engine::wave_route_hits()` confirms the wave route ACTUALLY served every batch (not a silent
//!       fallback to lpb) — the benchmark ABORTS if the wave didn't serve, so "wave" numbers are real.
//!
//! Env: GPU_DB_BENCH_ROWS (single table size, default 1048576 — wave-vs-lpb is ~table-size-independent,
//! both O(1)/needle; R1's A/B already covered the O(rows) scan curve), GPU_DB_BENCH_BATCH (comma-separated
//! batch sizes to sweep, default "1,8,32,256,4096"), GPU_DB_BENCH_BATCHES (measured batches/mode, default
//! 2000), GPU_DB_BENCH_WARMUP (untimed warmup batches/mode, default 20 — the first wave batch builds the
//! engine), GPU_DB_BENCH_THREADS (concurrent section thread counts, default "1,2,4,8"; "" disables it).
//!
//! Run (RTX box; never `--gpu-reset`, always under `timeout`):
//!   timeout 900 cargo run --release --example r2_wave_engine_ab -p gpu_db_engine

use std::env;
use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpu_db_engine::{RowBlock, Engine, RelationalRetainedReadTemplate};
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

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Scan,
    Lpb,
    Wave,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Scan => "scan",
            Mode::Lpb => "lpb-index",
            Mode::Wave => "wave",
        }
    }
    /// Set the two nested flags for this route. `wave` needs BOTH; `lpb` needs the index flag only.
    fn configure(self, e: &Engine) {
        match self {
            Mode::Scan => {
                e.set_wave_engine_enabled(false);
                e.set_wave_persistent_engine_enabled(false);
            }
            Mode::Lpb => {
                e.set_wave_engine_enabled(true);
                e.set_wave_persistent_engine_enabled(false);
            }
            Mode::Wave => {
                e.set_wave_engine_enabled(true);
                e.set_wave_persistent_engine_enabled(true);
            }
        }
    }
}

struct Lat {
    p50: u128,
    p99: u128,
    p999: u128,
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
        p999: pct(0.999),
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
        "  {label:<10} p50={:>7}us p99={:>7}us p99.9={:>7}us max={:>7}us | {:>13.0} lookups/s ({:>7.3} us/lookup)",
        l.p50, l.p99, l.p999, l.max, l.ops_per_s, l.per_lookup_us
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

/// Deterministic distinct needles for batch `b`: `(g*step) % rows` over `batch` consecutive `g`, `step`
/// coprime to `rows` ⇒ distinct within a batch (the index/wave routes require distinct needles, which the
/// batcher's `dedup_needles` guarantees in production). Indexing by `b` makes every mode issue the SAME
/// lookups in batch `b`.
fn needles_for_batch(b: usize, batch: usize, step: u64, rows: u64) -> Vec<i32> {
    let mut v = Vec::with_capacity(batch);
    for k in 0..batch {
        let g = (b * batch + k) as u64;
        v.push((g.wrapping_mul(step) % rows) as i32);
    }
    v
}

#[allow(clippy::too_many_arguments)]
fn measure(
    e: &Engine,
    template: &RelationalRetainedReadTemplate,
    mode: Mode,
    batch: usize,
    batches: usize,
    warmup: usize,
    step: u64,
    rows: u64,
) -> Result<Lat, Box<dyn Error>> {
    mode.configure(e);
    // Warmup: the first wave batch builds + caches the engine; lpb builds the index; also JIT/allocator.
    for b in 0..warmup {
        let needles = needles_for_batch(b, batch, step, rows);
        let sub = e.submit_relational_retained_template_point_lookups(template, &needles)?;
        let _ = e.complete_relational_retained_read_submission(sub)?;
    }
    let hits_before = e.wave_route_hits();
    let mut batch_micros = Vec::with_capacity(batches);
    let t = Instant::now();
    for b in 0..batches {
        let needles = needles_for_batch(b, batch, step, rows);
        let s0 = Instant::now();
        let sub = e.submit_relational_retained_template_point_lookups(template, &needles)?;
        let _ = e.complete_relational_retained_read_submission(sub)?;
        batch_micros.push(s0.elapsed().as_micros());
    }
    let wall = t.elapsed();
    // NON-VACUITY: the wave route must have SERVED every timed batch (not silently fallen back to lpb);
    // scan/lpb must NEVER hit the wave route. Abort loudly otherwise — a misleading number is worse than none.
    let hits = e.wave_route_hits() - hits_before;
    let expect = if mode == Mode::Wave { batches as u64 } else { 0 };
    assert_eq!(
        hits, expect,
        "{} mode: wave_route_hits delta {hits} != expected {expect} (silent fallback / mis-route)",
        mode.label()
    );
    Ok(summarize(batch_micros, batches * batch, wall))
}

/// Like `measure` but completes via the BATCHED path (`complete_relational_retained_read_submission_batched`)
/// — one flat result + per-needle ranges instead of N per-needle structs (DECISIONS "Result-path
/// optimization"). Shows how close the per-needle-result-model change gets end-to-end to the GPU drain.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn measure_batched(
    e: &Engine,
    template: &RelationalRetainedReadTemplate,
    mode: Mode,
    batch: usize,
    batches: usize,
    warmup: usize,
    step: u64,
    rows: u64,
) -> Result<Lat, Box<dyn Error>> {
    measure_batched_dense(e, template, mode, batch, batches, warmup, step, rows, false)
}

#[allow(clippy::too_many_arguments)]
fn measure_batched_dense(
    e: &Engine,
    template: &RelationalRetainedReadTemplate,
    mode: Mode,
    batch: usize,
    batches: usize,
    warmup: usize,
    step: u64,
    rows: u64,
    dense: bool,
) -> Result<Lat, Box<dyn Error>> {
    mode.configure(e);
    e.set_dense_index_probe_enabled(dense);
    for b in 0..warmup {
        let needles = needles_for_batch(b, batch, step, rows);
        let sub = e.submit_relational_retained_template_point_lookups(template, &needles)?;
        let _ = e.complete_relational_retained_read_submission_batched(sub)?;
    }
    let mut batch_micros = Vec::with_capacity(batches);
    let t = Instant::now();
    for b in 0..batches {
        let needles = needles_for_batch(b, batch, step, rows);
        let s0 = Instant::now();
        let sub = e.submit_relational_retained_template_point_lookups(template, &needles)?;
        let _ = e.complete_relational_retained_read_submission_batched(sub)?;
        batch_micros.push(s0.elapsed().as_micros());
    }
    let lat = summarize(batch_micros, batches * batch, t.elapsed());
    e.set_dense_index_probe_enabled(false);
    Ok(lat)
}

/// Rows from one batch in `mode`, for the byte-identity gate.
fn rows_once(
    e: &Engine,
    template: &RelationalRetainedReadTemplate,
    mode: Mode,
    needles: &[i32],
) -> Result<Vec<RowBlock>, Box<dyn Error>> {
    mode.configure(e);
    Ok(e
        .complete_relational_retained_read_submission(
            e.submit_relational_retained_template_point_lookups(template, needles)?,
        )?
        .into_iter()
        .map(|r| r.rows)
        .collect())
}

/// Concurrent section: `threads` workers each run `per_thread` batches through ONE shared engine in `mode`.
/// For `wave` this contends the per-engine Mutex (the single-flight ceiling); for scan/lpb the GPU
/// pipelines the per-batch launches. Returns aggregate lookups/s. NOT the production coalescer path.
fn measure_concurrent(
    engine: &Arc<Engine>,
    select: &Select,
    mode: Mode,
    batch: usize,
    threads: usize,
    per_thread: usize,
    step: u64,
    rows: u64,
) -> Result<f64, Box<dyn Error>> {
    mode.configure(engine);
    // Warm the route (build the engine/index) before the timed concurrent run.
    {
        let template = engine.prepare_relational_retained_read_template(select)?;
        for b in 0..4 {
            let needles = needles_for_batch(b, batch, step, rows);
            let sub = engine.submit_relational_retained_template_point_lookups(&template, &needles)?;
            let _ = engine.complete_relational_retained_read_submission(sub)?;
        }
    }
    let barrier = Arc::new(std::sync::Barrier::new(threads));
    let t = Instant::now();
    let workers: Vec<_> = (0..threads)
        .map(|w| {
            let engine = Arc::clone(engine);
            let barrier = Arc::clone(&barrier);
            let select = select.clone();
            std::thread::spawn(move || -> Result<(), String> {
                let template = engine
                    .prepare_relational_retained_read_template(&select)
                    .map_err(|e| e.to_string())?;
                barrier.wait();
                for i in 0..per_thread {
                    // Disjoint needle streams per worker so they don't all hit one cached needle.
                    let needles = needles_for_batch(w * per_thread + i, batch, step, rows);
                    let sub = engine
                        .submit_relational_retained_template_point_lookups(&template, &needles)
                        .map_err(|e| e.to_string())?;
                    engine
                        .complete_relational_retained_read_submission(sub)
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            })
        })
        .collect();
    for w in workers {
        w.join().expect("worker panicked").map_err(|e| -> Box<dyn Error> { e.into() })?;
    }
    let secs = t.elapsed().as_secs_f64();
    let total = (threads * per_thread * batch) as f64;
    Ok(if secs == 0.0 { 0.0 } else { total / secs })
}

fn parse_csv_usize(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|tok| tok.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let rows: i64 = env::var("GPU_DB_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_048_576);
    let batch_sizes = {
        let v = parse_csv_usize(
            &env::var("GPU_DB_BENCH_BATCH").unwrap_or_else(|_| "1,8,32,256,4096".to_string()),
        );
        if v.is_empty() {
            vec![256]
        } else {
            v
        }
    };
    let batches: usize = env::var("GPU_DB_BENCH_BATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let warmup: usize = env::var("GPU_DB_BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let thread_counts =
        parse_csv_usize(&env::var("GPU_DB_BENCH_THREADS").unwrap_or_else(|_| "1,2,4,8".to_string()));

    println!(
        "# R2.2b-3 wave-engine A/B  rows={rows}  batches={batch_sizes:?}  measured_batches={batches}  warmup={warmup}"
    );
    println!("# scan=both flags off | lpb-index=wave_engine_enabled | wave=both (persistent kernel)");
    println!("# single-flight = the production single-coalescer path; concurrent = the per-engine-Mutex ceiling (R2.2c)\n");

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
        println!("rows={rows}: no GPU resident route accepted — skipping (no GPU?)");
        return Ok(());
    }
    let exec_target = format!("{:?}", e.execute_relational_select(&select)?.executed_target);
    let template = e.prepare_relational_retained_read_template(&select)?;
    println!("# executed_target={exec_target}\n");

    // (batch, scan_ops, lpb_ops, wave_ops) for the closing summary.
    let mut headline: Vec<(usize, f64, f64, f64)> = Vec::new();

    println!("## SINGLE-FLIGHT (one caller, varying batch) — the production single-coalescer regime");
    for &b in &batch_sizes {
        let batch = b.min(rows as usize);

        // Correctness gate: scan == lpb == wave, byte-identical, before any timing.
        let probe = needles_for_batch(0, batch, step, rows_u);
        let scan_rows = rows_once(&e, &template, Mode::Scan, &probe)?;
        let lpb_rows = rows_once(&e, &template, Mode::Lpb, &probe)?;
        let wave_rows = rows_once(&e, &template, Mode::Wave, &probe)?;
        assert_eq!(lpb_rows, scan_rows, "batch={batch}: lpb != scan");
        assert_eq!(wave_rows, scan_rows, "batch={batch}: wave != scan");

        let scan = measure(&e, &template, Mode::Scan, batch, batches, warmup, step, rows_u)?;
        let lpb = measure(&e, &template, Mode::Lpb, batch, batches, warmup, step, rows_u)?;
        let wave = measure(&e, &template, Mode::Wave, batch, batches, warmup, step, rows_u)?;
        let wave_b = measure_batched(&e, &template, Mode::Wave, batch, batches, warmup, step, rows_u)?;
        let lpb_b = measure_batched(&e, &template, Mode::Lpb, batch, batches, warmup, step, rows_u)?;
        let lpb_dense_b = measure_batched_dense(
            &e, &template, Mode::Lpb, batch, batches, warmup, step, rows_u, true,
        )?;
        let sp_lpb = if lpb.ops_per_s > 0.0 {
            wave.ops_per_s / lpb.ops_per_s
        } else {
            0.0
        };
        let sp_scan = if scan.ops_per_s > 0.0 {
            wave.ops_per_s / scan.ops_per_s
        } else {
            0.0
        };
        println!("\n### batch={batch}");
        print_lat(Mode::Scan.label(), &scan);
        print_lat(Mode::Lpb.label(), &lpb);
        print_lat(Mode::Wave.label(), &wave);
        print_lat("lpb-batched", &lpb_b);
        print_lat("lpb-DENSE-batched", &lpb_dense_b);
        print_lat("wave-batched", &wave_b);
        let sp_lpb_b = if lpb_b.ops_per_s > 0.0 {
            wave_b.ops_per_s / lpb_b.ops_per_s
        } else {
            0.0
        };
        println!("  wave/lpb: {sp_lpb:.2}x lookups/s    wave/scan: {sp_scan:.2}x");
        println!("  BATCHED wave/lpb: {sp_lpb_b:.2}x    (batched completion = one flat result, no per-needle structs)");
        headline.push((batch, scan.ops_per_s, lpb.ops_per_s, wave.ops_per_s));
    }

    println!("\n## SINGLE-FLIGHT summary  (lookups/s by batch)");
    println!(
        "  {:>8}  {:>13}  {:>13}  {:>13}  {:>10}  {:>10}",
        "batch", "scan", "lpb-index", "wave", "wave/lpb", "wave/scan"
    );
    for (b, s, l, w) in &headline {
        let spl = if *l > 0.0 { w / l } else { 0.0 };
        let sps = if *s > 0.0 { w / s } else { 0.0 };
        println!("  {b:>8}  {s:>13.0}  {l:>13.0}  {w:>13.0}  {spl:>9.2}x  {sps:>9.2}x");
    }

    // Concurrent section — exposes the per-engine-Mutex single-flight ceiling for the wave (NOT production).
    if !thread_counts.is_empty() {
        let conc_batch = 32usize.min(rows as usize); // a small OLTP batch; the regime the wave is meant for
        let per_thread = (batches / thread_counts.iter().max().copied().unwrap_or(1)).max(50);
        let engine = Arc::new(e);
        println!(
            "\n## CONCURRENT (N threads, one shared Engine, batch={conc_batch}, {per_thread} batches/thread)"
        );
        println!("# wave contends its per-engine Mutex (single-flight ceiling); scan/lpb pipeline launches. NOT the prod path.");
        println!(
            "  {:>8}  {:>13}  {:>13}  {:>13}  {:>10}",
            "threads", "scan", "lpb-index", "wave", "wave/lpb"
        );
        for &threads in &thread_counts {
            let s = measure_concurrent(
                &engine, &select, Mode::Scan, conc_batch, threads, per_thread, step, rows_u,
            )?;
            let l = measure_concurrent(
                &engine, &select, Mode::Lpb, conc_batch, threads, per_thread, step, rows_u,
            )?;
            let w = measure_concurrent(
                &engine, &select, Mode::Wave, conc_batch, threads, per_thread, step, rows_u,
            )?;
            let spl = if l > 0.0 { w / l } else { 0.0 };
            println!("  {threads:>8}  {s:>13.0}  {l:>13.0}  {w:>13.0}  {spl:>9.2}x");
        }
    }

    Ok(())
}
