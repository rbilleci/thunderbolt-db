//! Mixed read+write scaling — the write-half Thread-4 Stage-4 REAL thesis: **a committing
//! writer never blocks readers.**
//!
//! The write-only A/B (`concurrent_write_scaling`) is the worst case for Stage 4: for tiny
//! autocommit writes the off-lock-prepare benefit is swamped by the concurrent path's extra
//! synchronization, so it underperforms a single global lock. Stage 4's value is elsewhere —
//! removing the `RwLock<Engine>` WRITE branch means a committing writer no longer excludes
//! readers. This benchmark measures exactly that, at the façade boundary, the SAME workload two
//! ways across a reader-concurrency sweep with a FIXED background writer pool committing the
//! whole time:
//!   - `serialized` (the OLD model): an external `RwLock` — readers `.read()`, writers
//!     `.write()` — so a committing writer EXCLUDES every reader, exactly as the removed engine
//!     write lock did.
//!   - `concurrent` (Stage 4): no external lock; reads run lock-free on a snapshot, writes take
//!     only the short commit lock.
//!
//! The only difference is the reader-vs-writer exclusion, so the READ p50/p99/qps gap is exactly
//! what Stage 4 bought. The thesis holds if concurrent read qps RISES with reader concurrency and
//! read p99 stays bounded, where the serialized model's reads stall behind the writers.
//!
//! In-memory WAL by default (isolates the lock model from fsync). Env: GPU_DB_MIX_READERS
//! (default 1,2,4,8,16,32,64), GPU_DB_MIX_WRITERS (default 4), GPU_DB_MIX_READS_PER (default 200),
//! GPU_DB_MIX_REPS (default 5), GPU_DB_MIX_ROWS (default 512).

use std::env;
use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, RwLock};
use std::thread;
use std::time::Instant;

use gpu_db_facade::{DbError, QueryOutcome, SharedEngine, SubmissionRequest};

fn submit_text(shared: &SharedEngine, sql: &str) -> Result<QueryOutcome, DbError> {
    let mut session = shared.open_session();
    shared
        .submit(&mut session, SubmissionRequest::Text(sql))
        .into_immediate()
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn median_u64(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    percentile(values, 0.5)
}

fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if values.is_empty() {
        0.0
    } else {
        values[values.len() / 2]
    }
}

fn run_ok(shared: &SharedEngine, sql: &str) {
    if let Err(err) = submit_text(shared, sql) {
        panic!("{sql:?} setup failed: {err:?}");
    }
}

/// One (readers, mode) cell on a FRESH engine: `writers` threads INSERT continuously while
/// `readers` threads each do `reads_per` point-reads of a seeded row. Returns
/// (read_qps, read_p50_us, read_p99_us, writes_committed).
fn run_cell(
    readers: usize,
    writers: usize,
    writes_per: usize,
    reads_per: usize,
    rows: usize,
    serialize: bool,
) -> (f64, u64, u64, usize) {
    let shared = Arc::new(SharedEngine::new());
    run_ok(&shared, "CREATE TABLE t (id INT)");
    let mut seed = String::from("INSERT INTO t (id) VALUES ");
    for id in 0..rows {
        if id > 0 {
            seed.push(',');
        }
        seed.push('(');
        seed.push_str(&id.to_string());
        seed.push(')');
    }
    run_ok(&shared, &seed);

    let lock = Arc::new(RwLock::new(())); // the OLD-model writer-excludes-reader lock
    let stop = Arc::new(AtomicBool::new(false));
    let writes = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(readers + writers + 1));

    // Background writer pool — commits disjoint INSERTs the whole time the readers run.
    let writer_handles: Vec<_> = (0..writers)
        .map(|w| {
            let shared = Arc::clone(&shared);
            let lock = Arc::clone(&lock);
            let stop = Arc::clone(&stop);
            let writes = Arc::clone(&writes);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                // Disjoint id space per writer (no unique index, but keep ids semantically distinct).
                let mut id = rows + 1 + w * 100_000_000;
                // Bounded budget so the table can't grow without limit (keeps reads fast and the
                // cell terminating); `stop` ends the load early once the readers are done.
                let mut budget = writes_per;
                barrier.wait();
                while budget > 0 && !stop.load(Ordering::Relaxed) {
                    let sql = format!("INSERT INTO t (id) VALUES ({id})");
                    if serialize {
                        let _guard = lock.write().unwrap();
                        let _ = submit_text(&shared, &sql);
                    } else {
                        let _ = submit_text(&shared, &sql);
                    }
                    id += 1;
                    budget -= 1;
                    writes.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // Reader sweep — each thread point-reads a seeded row `reads_per` times, recording latency.
    let reader_handles: Vec<_> = (0..readers)
        .map(|_| {
            let shared = Arc::clone(&shared);
            let lock = Arc::clone(&lock);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let mut latencies = Vec::with_capacity(reads_per);
                barrier.wait();
                for _ in 0..reads_per {
                    let started = Instant::now();
                    if serialize {
                        let _guard = lock.read().unwrap();
                        submit_text(&shared, "SELECT id FROM t WHERE id = 100").unwrap();
                    } else {
                        submit_text(&shared, "SELECT id FROM t WHERE id = 100").unwrap();
                    }
                    latencies.push(started.elapsed().as_micros() as u64);
                }
                latencies
            })
        })
        .collect();

    barrier.wait();
    let wall_started = Instant::now();
    let mut all_latencies = Vec::with_capacity(readers * reads_per);
    for handle in reader_handles {
        all_latencies.extend(handle.join().expect("reader thread panicked"));
    }
    let read_wall = wall_started.elapsed();
    stop.store(true, Ordering::Relaxed);
    for handle in writer_handles {
        handle.join().expect("writer thread panicked");
    }

    let total_reads = (readers * reads_per) as f64;
    let read_qps = if read_wall.as_secs_f64() > 0.0 {
        total_reads / read_wall.as_secs_f64()
    } else {
        0.0
    };
    all_latencies.sort_unstable();
    (
        read_qps,
        percentile(&all_latencies, 0.50),
        percentile(&all_latencies, 0.99),
        writes.load(Ordering::Relaxed),
    )
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn Error>> {
    let reader_targets: Vec<usize> = env::var("GPU_DB_MIX_READERS")
        .unwrap_or_else(|_| "1,2,4,8,16,32,64".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let writers = env_usize("GPU_DB_MIX_WRITERS", 4);
    let writes_per = env_usize("GPU_DB_MIX_WRITES_PER", 2000);
    let reads_per = env_usize("GPU_DB_MIX_READS_PER", 200);
    let reps = env_usize("GPU_DB_MIX_REPS", 5);
    let rows = env_usize("GPU_DB_MIX_ROWS", 512);

    println!(
        "concurrent_mixed_read_write_scaling: writers={writers} reads_per={reads_per} reps={reps} rows={rows}"
    );
    println!("READ p50/p99/qps under a sustained {writers}-writer commit load — serialized (writer excludes readers) vs concurrent (lock-free reads)");
    println!("query=SELECT id FROM t WHERE id = 100");
    println!();
    println!(
        "| readers | ser read p50 us | ser read p99 us | ser read qps | conc read p50 us | conc read p99 us | conc read qps | read qps speedup |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|");

    let mut json_rows: Vec<String> = Vec::new();
    for &r in &reader_targets {
        let mut sp50s = Vec::new();
        let mut sp99s = Vec::new();
        let mut sqpss = Vec::new();
        let mut cp50s = Vec::new();
        let mut cp99s = Vec::new();
        let mut cqpss = Vec::new();
        for _ in 0..reps {
            let (sq, sp50, sp99, _sw) = run_cell(r, writers, writes_per, reads_per, rows, true);
            let (cq, cp50, cp99, _cw) = run_cell(r, writers, writes_per, reads_per, rows, false);
            sp50s.push(sp50);
            sp99s.push(sp99);
            sqpss.push(sq);
            cp50s.push(cp50);
            cp99s.push(cp99);
            cqpss.push(cq);
        }
        let sp50 = median_u64(&mut sp50s);
        let sp99 = median_u64(&mut sp99s);
        let sq = median_f64(&mut sqpss);
        let cp50 = median_u64(&mut cp50s);
        let cp99 = median_u64(&mut cp99s);
        let cq = median_f64(&mut cqpss);
        let qps_speedup = if sq > 0.0 { cq / sq } else { 0.0 };
        println!(
            "| {r} | {sp50} | {sp99} | {sq:.0} | {cp50} | {cp99} | {cq:.0} | {qps_speedup:.2}x |"
        );
        json_rows.push(format!(
            "{{\"readers\":{r},\"writers\":{writers},\"serial_read_p50_us\":{sp50},\"serial_read_p99_us\":{sp99},\"serial_read_qps\":{sq:.3},\"concurrent_read_p50_us\":{cp50},\"concurrent_read_p99_us\":{cp99},\"concurrent_read_qps\":{cq:.3},\"read_qps_speedup\":{qps_speedup:.3}}}"
        ));
    }
    println!();
    println!(
        "json={{\"kind\":\"concurrent_mixed_read_write_scaling\",\"writers\":{writers},\"reads_per\":{reads_per},\"reps\":{reps},\"rows\":{rows},\"cells\":[{}]}}",
        json_rows.join(",")
    );
    Ok(())
}
