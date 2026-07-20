//! Write-half Thread-4 Stage-4 payoff — concurrent **write** scaling A/B.
//!
//! The mirror of `p1_m3_concurrent_read_scaling` for the write path. It measures the
//! Stage-4 concurrency flip directly at the façade boundary: the SAME workload, the SAME
//! `SharedEngine`, executed two ways across a concurrency sweep —
//! `serialized` (every write holds one global lock for its whole duration — the
//! single-writer model the engine ran on before Stage 4) vs `concurrent` (every write
//! goes through `SharedEngine::submit`, off-lock prepare + a short commit critical
//! section, no engine write lock). The only difference is the global lock, so the qps gap
//! is exactly the serialization Stage 4 removed.
//!
//! Workload: N writers each INSERT a DISJOINT block of unique ids into one table (no unique
//! index — insert prepare is then O(1), so the run stays fast and it isolates exactly the
//! BUG-1 hazard: the insert ROW-KEY conflict unit, with no unique-slot dimension in play).
//! Because the ids are disjoint, EVERY insert must commit: a non-zero abort count in
//! `concurrent` mode would mean the BUG-1 false-conflict (insert row-key wrongly used as a
//! conflict unit) regressed — so `aborts` is both a correctness gate and part of the result.
//! INSERT is the most common write, which is why it is the headline here. The concurrent win
//! is bounded by the short commit critical section (Amdahl on the commit lock): the point is
//! that concurrent throughput RISES with concurrency where serialized is flat, at zero false
//! aborts.
//!
//! Reads the lock-removal win, not the fsync floor: the default in-memory WAL keeps the
//! commit critical section short so the measured contrast is the off-lock-prepare overlap.
//! Durable-WAL group-commit amortization is a separate axis (Stage 1 / Stage 5).
//!
//! Env: GPU_DB_WSCALE_CONCURRENCY (default 1,2,4,8,16,32,64),
//! GPU_DB_WSCALE_OPS_PER_THREAD (default 100), GPU_DB_WSCALE_REPS (default 5).

use std::env;
use std::error::Error;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Instant;

use gpu_db_facade::{DbError, ErrorCategory, QueryOutcome, SharedEngine, SubmissionRequest};

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

/// One measured run of a (concurrency, mode) cell on a FRESH engine. Each of `threads`
/// writers commits `ops_per_thread` disjoint INSERTs. Returns
/// (commit_qps, p50_us, p95_us, p99_us, commits, aborts).
fn run_cell(
    threads: usize,
    ops_per_thread: usize,
    serialize: bool,
) -> (f64, u64, u64, u64, usize, usize) {
    // Fresh state per cell so reps/cells are independent and ids never collide.
    let shared = Arc::new(SharedEngine::new());
    run_ok(&shared, "CREATE TABLE t (id INT)");

    let serial_lock = Arc::new(Mutex::new(()));
    let barrier = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|w| {
            let shared = Arc::clone(&shared);
            let serial_lock = Arc::clone(&serial_lock);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let base = w * ops_per_thread;
                let mut latencies = Vec::with_capacity(ops_per_thread);
                let mut commits = 0usize;
                let mut aborts = 0usize;
                barrier.wait(); // release together so the write bodies actually overlap
                for k in 0..ops_per_thread {
                    let id = base + k; // disjoint across writers => every insert must commit
                    let sql = format!("INSERT INTO t (id) VALUES ({id})");
                    let started = Instant::now();
                    let res = if serialize {
                        let _guard = serial_lock.lock().unwrap();
                        submit_text(&shared, &sql)
                    } else {
                        submit_text(&shared, &sql)
                    };
                    latencies.push(started.elapsed().as_micros() as u64);
                    match res {
                        Ok(_) => commits += 1,
                        // A disjoint insert must never lose a conflict; a Serialization here
                        // would be the BUG-1 false-conflict regressing. Record, don't panic,
                        // so the result surfaces it loudly.
                        Err(err) if err.category == ErrorCategory::Serialization => aborts += 1,
                        Err(err) => panic!("unexpected non-retryable write error: {err:?}"),
                    }
                }
                (latencies, commits, aborts)
            })
        })
        .collect();

    barrier.wait();
    let wall_started = Instant::now();
    let mut all_latencies = Vec::with_capacity(threads * ops_per_thread);
    let mut commits = 0usize;
    let mut aborts = 0usize;
    for handle in handles {
        let (lat, c, a) = handle.join().expect("writer thread panicked");
        all_latencies.extend(lat);
        commits += c;
        aborts += a;
    }
    let wall = wall_started.elapsed();

    let qps = if wall.as_secs_f64() > 0.0 {
        commits as f64 / wall.as_secs_f64()
    } else {
        0.0
    };
    all_latencies.sort_unstable();
    (
        qps,
        percentile(&all_latencies, 0.50),
        percentile(&all_latencies, 0.95),
        percentile(&all_latencies, 0.99),
        commits,
        aborts,
    )
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn Error>> {
    let targets: Vec<usize> = env::var("GPU_DB_WSCALE_CONCURRENCY")
        .unwrap_or_else(|_| "1,2,4,8,16,32,64".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let ops_per_thread = env_usize("GPU_DB_WSCALE_OPS_PER_THREAD", 100);
    let reps = env_usize("GPU_DB_WSCALE_REPS", 5);

    println!(
        "concurrent_write_scaling: ops_per_thread={ops_per_thread} reps={reps}  \
         workload=disjoint INSERT into one un-indexed table (every insert must commit)"
    );
    println!("A/B: serialized (one global lock, whole write) vs concurrent (off-lock prepare + short commit lock)");
    println!();
    println!("| conc | serial p50 us | serial p99 us | serial qps | conc p50 us | conc p99 us | conc qps | qps speedup | conc aborts |");
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|");

    let mut json_rows: Vec<String> = Vec::new();
    let mut total_conc_aborts = 0usize;
    for &c in &targets {
        let mut serial_p50s = Vec::new();
        let mut serial_p99s = Vec::new();
        let mut serial_qpss = Vec::new();
        let mut conc_p50s = Vec::new();
        let mut conc_p99s = Vec::new();
        let mut conc_qpss = Vec::new();
        let mut conc_aborts = 0usize;
        for _ in 0..reps {
            let (sq, sp50, _sp95, sp99, _sc, _sa) = run_cell(c, ops_per_thread, true);
            let (cq, cp50, _cp95, cp99, _cc, ca) = run_cell(c, ops_per_thread, false);
            serial_p50s.push(sp50);
            serial_p99s.push(sp99);
            serial_qpss.push(sq);
            conc_p50s.push(cp50);
            conc_p99s.push(cp99);
            conc_qpss.push(cq);
            conc_aborts += ca;
        }
        total_conc_aborts += conc_aborts;
        let sp50 = median_u64(&mut serial_p50s);
        let sp99 = median_u64(&mut serial_p99s);
        let sq = median_f64(&mut serial_qpss);
        let cp50 = median_u64(&mut conc_p50s);
        let cp99 = median_u64(&mut conc_p99s);
        let cq = median_f64(&mut conc_qpss);
        let qps_speedup = if sq > 0.0 { cq / sq } else { 0.0 };
        println!(
            "| {c} | {sp50} | {sp99} | {sq:.0} | {cp50} | {cp99} | {cq:.0} | {qps_speedup:.2}x | {conc_aborts} |"
        );
        json_rows.push(format!(
            "{{\"concurrency\":{c},\"serial_p50_us\":{sp50},\"serial_p99_us\":{sp99},\"serial_qps\":{sq:.3},\"concurrent_p50_us\":{cp50},\"concurrent_p99_us\":{cp99},\"concurrent_qps\":{cq:.3},\"qps_speedup\":{qps_speedup:.3},\"concurrent_aborts\":{conc_aborts}}}"
        ));
    }
    println!();
    if total_conc_aborts != 0 {
        println!(
            "WARNING: {total_conc_aborts} disjoint INSERTs falsely aborted in concurrent mode — BUG-1 (insert false-conflict) has regressed."
        );
    } else {
        println!("correctness: 0 false aborts across all concurrent cells (disjoint INSERTs all committed).");
    }
    println!(
        "json={{\"kind\":\"concurrent_write_scaling\",\"ops_per_thread\":{ops_per_thread},\"reps\":{reps},\"concurrent_aborts_total\":{total_conc_aborts},\"cells\":[{}]}}",
        json_rows.join(",")
    );
    Ok(())
}
