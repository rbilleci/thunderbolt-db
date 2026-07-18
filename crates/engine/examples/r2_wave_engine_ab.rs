//! Production sharded point-read result-path instrument.
//!
//! This benchmark measures the authoritative-shard int4 unique-key point route through two genuinely distinct
//! result contracts over the same prepared multi-shard dense kernel:
//!   - `compat-public`: retained-template completion materializes the public one-range-per-needle result;
//!   - `production-compact`: the facade's production entry keeps the internal all-present identity mapping.
//!
//! WHAT THIS MEASURES (and what it does NOT). In production, point reads flow through the facade's SINGLE
//! coalescer thread, so the production-relevant regime is ONE caller varying BATCH size (a higher offered
//! rate coalesces into bigger batches). That is the PRIMARY sweep below. We ALSO run a concurrent section
//! (N threads hammering the same shape through one shared `Engine`) where the GPU pipelines per-batch
//! launches — NOT the production single-coalescer path.
//!
//! NON-VACUITY: every route is asserted BYTE-IDENTICAL before any timing (a silent wrong/empty result can't
//! win).
//!
//! Env: GPU_DB_BENCH_ROWS (single table size, default 1048576), GPU_DB_BENCH_BATCH
//! (comma-separated batch sizes to sweep, default "1,8,32,256,4096,16384,65536"),
//! GPU_DB_BENCH_BATCHES (measured batches/mode, default 2000), GPU_DB_BENCH_WARMUP (untimed warmup
//! batches/mode, default 20 — lpb builds the index), GPU_DB_BENCH_THREADS (concurrent section thread
//! counts, default "1,2,4,8"; "" disables it), GPU_DB_BENCH_INSERT_CHUNK (rows per authoritative
//! insert publication; also the fixed-row descriptor-count sweep control).
//!
//! Run (RTX box; never `--gpu-reset`, always under `timeout`):
//!   timeout 900 cargo run --release --example r2_wave_engine_ab -p gpu_db_engine

use std::env;
use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpu_db_engine::{Engine, RelationalRetainedReadTemplate, RowBlock};
use gpu_db_sql::{parse_command, Command, Select, SqlValue};

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
/// Prints the build (INSERT-loop + residency) wall time -- at large row counts the SQL build is the
/// dominant cost, so the report card's out-of-L2 sizing is driven by this number.
fn build_resident_engine(rows: i64) -> Result<Engine, Box<dyn Error>> {
    let t_build = Instant::now();
    // Rows per INSERT statement. At large row counts the SQL build dominates wall time; a bigger chunk
    // amortizes per-statement parse/txn overhead so the out-of-L2 table can be built inside the timeout.
    let chunk: i64 = env::var("GPU_DB_BENCH_INSERT_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&c| c > 0)
        .unwrap_or(1000);
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")?;
    let mut txn = 2u64;
    let mut id = 0i64;
    while id < rows {
        let mut vals = String::new();
        for _ in 0..chunk {
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
    let t_insert = t_build.elapsed();
    e.populate_relational_residency_snapshot("accounts")?;
    println!(
        "# build: {rows} rows across {} authoritative shards, loaded + made resident in {:.1}s (insert {:.1}s + residency {:.1}s)",
        e.resident_shard_count("accounts"),
        t_build.elapsed().as_secs_f64(),
        t_insert.as_secs_f64(),
        (t_build.elapsed() - t_insert).as_secs_f64(),
    );
    Ok(e)
}

/// Deterministic distinct needles for batch `b`: `(g*step) % rows` over `batch` consecutive `g`, `step`
/// coprime to `rows` ⇒ distinct within a batch (the index/wave routes require distinct needles, which the
/// batcher's `dedup_needles` guarantees in production). Indexing by `b` makes both result contracts issue
/// the same lookups in batch `b`.
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
    batch: usize,
    batches: usize,
    warmup: usize,
    step: u64,
    rows: u64,
) -> Result<Lat, Box<dyn Error>> {
    // Warm the prepared route, JIT, and allocator.
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
    let wall = t.elapsed();
    Ok(summarize(batch_micros, batches * batch, wall))
}

/// Production facade entry: one compact result whose all-present case retains an internal identity mapping.
#[allow(clippy::too_many_arguments)]
fn measure_batched(
    e: &Engine,
    select: &Select,
    batch: usize,
    batches: usize,
    warmup: usize,
    step: u64,
    rows: u64,
) -> Result<Lat, Box<dyn Error>> {
    for b in 0..warmup {
        let needles = needles_for_batch(b, batch, step, rows);
        let result = e
            .submit_sharded_point_lookups_batched_compact(select, &needles)?
            .ok_or("production compact point route declined during warmup")?;
        std::hint::black_box(result.needle_count());
    }
    let mut batch_micros = Vec::with_capacity(batches);
    let t = Instant::now();
    for b in 0..batches {
        let needles = needles_for_batch(b, batch, step, rows);
        let s0 = Instant::now();
        let result = e
            .submit_sharded_point_lookups_batched_compact(select, &needles)?
            .ok_or("production compact point route declined during measurement")?;
        std::hint::black_box(result.needle_count());
        batch_micros.push(s0.elapsed().as_micros());
    }
    Ok(summarize(batch_micros, batches * batch, t.elapsed()))
}

/// Public compatibility rows from one batch, for the exact-value gate.
fn rows_once(
    e: &Engine,
    template: &RelationalRetainedReadTemplate,
    needles: &[i32],
) -> Result<Vec<RowBlock>, Box<dyn Error>> {
    Ok(e.complete_relational_retained_read_submission(
        e.submit_relational_retained_template_point_lookups(template, needles)?,
    )?
    .into_iter()
    .map(|r| r.rows)
    .collect())
}

/// Concurrent section: `threads` workers each run `per_thread` compact batches through ONE shared engine.
/// The GPU pipelines the per-batch launches. Returns per-batch latency and aggregate throughput. NOT the
/// production coalescer path.
#[allow(clippy::too_many_arguments)] // Benchmark dimensions stay explicit at every measured call site.
fn measure_concurrent(
    engine: &Arc<Engine>,
    select: &Select,
    batch: usize,
    threads: usize,
    per_thread: usize,
    step: u64,
    rows: u64,
) -> Result<Lat, Box<dyn Error>> {
    // Warm the exact production compact route before the timed concurrent run.
    for b in 0..4 {
        let needles = needles_for_batch(b, batch, step, rows);
        let result = engine
            .submit_sharded_point_lookups_batched_compact(select, &needles)?
            .ok_or("production compact point route declined during concurrent warmup")?;
        std::hint::black_box(result.needle_count());
    }
    let barrier = Arc::new(std::sync::Barrier::new(threads));
    let t = Instant::now();
    let workers: Vec<_> = (0..threads)
        .map(|w| {
            let engine = Arc::clone(engine);
            let barrier = Arc::clone(&barrier);
            let select = select.clone();
            std::thread::spawn(move || -> Result<Vec<u128>, String> {
                let mut batch_micros = Vec::with_capacity(per_thread);
                barrier.wait();
                for i in 0..per_thread {
                    // Disjoint needle streams per worker so they don't all hit one cached needle.
                    let needles = needles_for_batch(w * per_thread + i, batch, step, rows);
                    let s0 = Instant::now();
                    let result = engine
                        .submit_sharded_point_lookups_batched_compact(&select, &needles)
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| {
                            "production compact point route declined during concurrent run"
                                .to_string()
                        })?;
                    std::hint::black_box(result.needle_count());
                    batch_micros.push(s0.elapsed().as_micros());
                }
                Ok(batch_micros)
            })
        })
        .collect();
    let mut batch_micros = Vec::with_capacity(threads * per_thread);
    for w in workers {
        batch_micros.extend(
            w.join()
                .expect("worker panicked")
                .map_err(|e| -> Box<dyn Error> { e.into() })?,
        );
    }
    Ok(summarize(
        batch_micros,
        threads * per_thread * batch,
        t.elapsed(),
    ))
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
            &env::var("GPU_DB_BENCH_BATCH")
                .unwrap_or_else(|_| "1,8,32,256,4096,16384,65536".to_string()),
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
    let thread_counts = parse_csv_usize(
        &env::var("GPU_DB_BENCH_THREADS").unwrap_or_else(|_| "1,2,4,8".to_string()),
    );

    println!(
        "# sharded point result paths  rows={rows}  batches={batch_sizes:?}  measured_batches={batches}  warmup={warmup}"
    );
    println!(
        "# compat-public = prepared dense route + established one-range-per-needle materialization"
    );
    println!(
        "# production-compact = facade production entry + private all-present identity mapping"
    );
    println!(
        "# one-caller sweep = production coalescer regime; concurrent = multi-producer ceiling\n"
    );

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
    let exec_target = format!(
        "{:?}",
        e.execute_relational_select(&select)?.executed_target
    );
    let template = e.prepare_relational_retained_read_template(&select)?;
    println!("# executed_target={exec_target}\n");

    // (batch, compatibility p50/throughput, compact-production p50/throughput) for the closing summary.
    let mut headline: Vec<(usize, u128, f64, u128, f64)> = Vec::new();

    println!("## ONE CALLER (varying batch) — the production single-coalescer regime");
    for &b in &batch_sizes {
        let batch = b.min(rows as usize);

        // Correctness gate: both result contracts are exact and non-vacuous before any timing.
        let probe = needles_for_batch(0, batch, step, rows_u);
        let compat_rows = rows_once(&e, &template, &probe)?;
        assert_eq!(
            compat_rows.len(),
            probe.len(),
            "one result block per needle"
        );
        for (row, &needle) in compat_rows.iter().zip(&probe) {
            assert_eq!(row.len(), 1, "every benchmark needle is present and unique");
            assert_eq!(
                row.row(0),
                [
                    SqlValue::Int4(needle),
                    SqlValue::Int4(((i64::from(needle) * 7) % 100_000) as i32),
                ],
                "per-needle compatibility result is non-vacuous"
            );
        }
        let compact = e
            .submit_sharded_point_lookups_batched_compact(&select, &probe)?
            .expect("production compact point route must serve the benchmark shape");
        assert_eq!(compact.needle_count(), probe.len());
        for (needle, &value) in probe.iter().enumerate() {
            assert_eq!(
                compact.needle_values(needle),
                [value, ((i64::from(value) * 7) % 100_000) as i32],
                "production compact result is byte-identical and non-vacuous"
            );
        }

        let compat = measure(&e, &template, batch, batches, warmup, step, rows_u)?;
        let compact = measure_batched(&e, &select, batch, batches, warmup, step, rows_u)?;
        let compact_speedup = if compat.ops_per_s > 0.0 {
            compact.ops_per_s / compat.ops_per_s
        } else {
            0.0
        };
        println!("\n### batch={batch}");
        print_lat("compat-public", &compat);
        print_lat("prod-compact", &compact);
        println!(
            "  compact/public comparison: p50 batch latency {}/{}us | throughput {:.0}/{:.0} \
             lookups/s ({compact_speedup:.2}x)",
            compact.p50, compat.p50, compact.ops_per_s, compat.ops_per_s
        );
        headline.push((
            batch,
            compat.p50,
            compat.ops_per_s,
            compact.p50,
            compact.ops_per_s,
        ));
    }

    println!("\n## ONE-CALLER summary  (p50 batch latency + lookups/s by batch)");
    println!(
        "  {:>8}  {:>14}  {:>17}  {:>14}  {:>17}  {:>10}",
        "batch",
        "compat p50 us",
        "compat lookups/s",
        "compact p50 us",
        "compact lookups/s",
        "compact/compat"
    );
    for (b, compat_p50, compat_ops, compact_p50, compact_ops) in &headline {
        let speedup = if *compat_ops > 0.0 {
            compact_ops / compat_ops
        } else {
            0.0
        };
        println!(
            "  {b:>8}  {compat_p50:>14}  {compat_ops:>17.0}  {compact_p50:>14}  \
             {compact_ops:>17.0}  {speedup:>9.2}x"
        );
    }

    // Concurrent section — N producers hammering one shared Engine (NOT the production single-coalescer path).
    if !thread_counts.is_empty() {
        let conc_batch = 32usize.min(rows as usize); // a small OLTP batch
        let per_thread = (batches / thread_counts.iter().max().copied().unwrap_or(1)).max(50);
        let engine = Arc::new(e);
        println!(
            "\n## CONCURRENT (N threads, one shared Engine, batch={conc_batch}, {per_thread} batches/thread)"
        );
        println!("# production compact batches pipeline launches across threads; not the single-coalescer path.");
        println!(
            "  {:>8}  {:>14}  {:>17}  {:>14}",
            "threads", "p50 batch us", "lookups/s", "us/lookup"
        );
        for &threads in &thread_counts {
            let compact = measure_concurrent(
                &engine, &select, conc_batch, threads, per_thread, step, rows_u,
            )?;
            println!(
                "  {threads:>8}  {:>14}  {:>17.0}  {:>14.3}",
                compact.p50, compact.ops_per_s, compact.per_lookup_us
            );
        }
    }

    Ok(())
}
