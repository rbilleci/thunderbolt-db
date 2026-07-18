//! OLTP batched-read scaling — the decisive test of the GPU-OLTP bet.
//!
//! Historical per-operation measurements showed a single GPU point read costs a
//! FIXED ~72µs (kernel launch + context-set + stream-sync + host round-trip), independent of table
//! size — i.e. it is amortizable launch overhead, not compute. The bet (ARCHITECTURE §9) is that
//! BATCHING — coalescing many concurrent point lookups into one GPU submission (`PointLookupBatcher`,
//! the embryo of the persistent-kernel wave engine) — divides that fixed cost across the batch and
//! beats a CPU index lookup at scale.
//!
//! This harness drives `SELECT id FROM accounts WHERE id = ?` from `threads` concurrent threads
//! (closed-loop; offered concurrency = `threads`) over a resident table, three ways:
//!   - **host (non-resident):** lock-free snapshot read on the CPU — the baseline to beat.
//!   - **gpu per-query:** the unbatched resident route — pays the full ~72µs every call.
//!   - **gpu batched:** the coalescing batcher — the contender; effective per-lookup cost should
//!     fall toward 72µs / batch_size if amortization works.
//!
//! Reports p50/p99/p99.9 + aggregate throughput. The question: does `gpu batched` throughput (and
//! p99) beat `host`?
//!
//! Env: GPU_DB_BENCH_ROWS (20000), GPU_DB_BENCH_THREADS (128), GPU_DB_BENCH_OPS (per-thread, 2000),
//! GPU_DB_BENCH_BATCH (batcher max_items trigger, 128), GPU_DB_BENCH_WAIT_US (batcher linger, 500).

use std::env;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use gpu_db_engine::Engine;
use gpu_db_facade::{
    execute_on_shared_engine, execute_on_shared_engine_batched, BatchedDispatch,
    PointLookupBatcher, SharedEngine,
};

fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() as f64 * p).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn build_engine(rows: i64, warm: bool) -> Engine {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
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
        )
        .unwrap();
        txn += 1;
    }
    if warm {
        // make the table GPU-resident so the resident route (and the batcher's classify) engage.
        e.populate_relational_residency_snapshot("accounts")
            .unwrap();
    }
    e
}

fn run(
    label: &str,
    shared: &Arc<SharedEngine>,
    batcher: Option<&Arc<PointLookupBatcher>>,
    threads: usize,
    ops_per_thread: usize,
    rows: i64,
) {
    let barrier = Arc::new(Barrier::new(threads + 1));
    let mut handles = Vec::with_capacity(threads);
    for t in 0..threads {
        let shared = Arc::clone(shared);
        let batcher = batcher.cloned();
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let mut state: u64 = 0x9e37_79b9_7f4a_7c15 ^ (t as u64 + 1).wrapping_mul(2_654_435_761);
            let mut lat = Vec::with_capacity(ops_per_thread);
            barrier.wait();
            for _ in 0..ops_per_thread {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let id = (state % rows as u64) as i64;
                let sql = format!("SELECT id FROM accounts WHERE id = {id}");
                let s = Instant::now();
                match &batcher {
                    Some(b) => match execute_on_shared_engine_batched(&shared, b, &sql) {
                        BatchedDispatch::Immediate(r) => {
                            r.unwrap();
                        }
                        BatchedDispatch::Batched(rx) => {
                            rx.blocking_recv().unwrap().unwrap();
                        }
                    },
                    None => {
                        execute_on_shared_engine(&shared, &sql).unwrap();
                    }
                }
                lat.push(s.elapsed().as_micros() as u64);
            }
            lat
        }));
    }
    barrier.wait();
    let wall = Instant::now();
    let mut all = Vec::with_capacity(threads * ops_per_thread);
    for h in handles {
        all.extend(h.join().unwrap());
    }
    let wall = wall.elapsed();
    all.sort_unstable();
    let tput = if wall.is_zero() {
        0.0
    } else {
        all.len() as f64 / wall.as_secs_f64()
    };
    println!(
        "  {label:<20} p50={:>7}us  p99={:>8}us  p99.9={:>8}us  max={:>8}us  {:>12.0} ops/s",
        pct(&all, 0.50),
        pct(&all, 0.99),
        pct(&all, 0.999),
        all.last().copied().unwrap_or(0),
        tput
    );
}

fn main() {
    let rows: i64 = env::var("GPU_DB_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let threads: usize = env::var("GPU_DB_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let ops: usize = env::var("GPU_DB_BENCH_OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000);
    let max_items: usize = env::var("GPU_DB_BENCH_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let max_wait_us: u64 = env::var("GPU_DB_BENCH_WAIT_US")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);

    println!(
        "# OLTP batched-read scaling  rows={rows} threads={threads} ops/thread={ops} \
         batch_trigger={max_items} max_wait={max_wait_us}us  (total {} ops/mode)",
        threads * ops
    );

    let resident = Arc::new(SharedEngine::from_engine(build_engine(rows, true)));
    let batcher = Arc::new(PointLookupBatcher::with_triggers(
        Arc::clone(&resident),
        max_items,
        Duration::from_micros(max_wait_us),
    ));
    let host = Arc::new(SharedEngine::from_engine(build_engine(rows, false)));

    println!("## SELECT id FROM accounts WHERE id = ?  @ {threads} concurrent threads");
    run("host (non-resident)", &host, None, threads, ops, rows);
    run("gpu per-query", &resident, None, threads, ops, rows);
    run("gpu batched", &resident, Some(&batcher), threads, ops, rows);
}
