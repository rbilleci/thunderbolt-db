//! Phase-D CONCURRENT-COMMIT SLO benchmark (HANDOVER (D); the write-side analog of
//! `r2_wave_engine_ab`): measures sustained + burst commit TPS and per-commit latency
//! percentiles for the concurrent DML path (`execute_dml_concurrent`) against the charter SLO —
//! **>100k TPS sustained, ≥400k TPS burst, p50/p99/p99.9 < 0.5/1/5 ms**.
//!
//! Closed-loop: N writer threads each drive single-row INSERTs on a plain (constraint-free)
//! table — the ADR-009 homogeneous fast-path wave shape — for a fixed duration, timing every
//! commit end-to-end (parse → prepare → commit critical section → group-flush wait → publish).
//! Burst TPS is the best 100 ms completion window × 10. Two arms:
//!   - in-memory WAL (default): isolates the concurrency-control / commit-path ceiling (the
//!     ledger #6 target). This is the SLO-comparison arm.
//!   - durable WAL (GPU_DB_BENCH_DURABLE=1): honest fsync-bound numbers on this machine's disk
//!     (group commit amortizes; the disk's fsync rate is the floor for per-commit latency).
//!
//! CPU + disk only — no GPU residency, no CUDA initialization.
//!
//! Run:  cargo run --release -p gpu_db_engine --example oltp_commit_slo_benchmark
//! Env:  GPU_DB_BENCH_WRITERS (default "1,2,4,8,16,32"), GPU_DB_BENCH_SECONDS (default 3),
//!       GPU_DB_BENCH_DURABLE (default unset = in-memory), GPU_DB_BENCH_WAL_DIR (default
//!       target/wal-bench).

use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use gpu_db_engine::Engine;

// jemalloc: the write path allocates ~20x/commit (parse tree, delta strings, payload clones);
// glibc's malloc arenas measurably collapse under multi-writer contention (the repo's server
// binary already ships jemalloc for the same reason — this benchmark would otherwise measure
// the allocator, not the commit path).
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

struct WriterSample {
    latencies_nanos: Vec<u64>,
    completions_millis: Vec<u64>,
    prepare_nanos_total: u64,
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() as f64) * p).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn main() -> Result<(), Box<dyn Error>> {
    let writers_sweep: Vec<usize> = std::env::var("GPU_DB_BENCH_WRITERS")
        .unwrap_or_else(|_| "1,2,4,8,16,32".to_string())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let seconds: u64 = std::env::var("GPU_DB_BENCH_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let durable = std::env::var("GPU_DB_BENCH_DURABLE").is_ok_and(|v| v == "1");
    let wal_dir = std::env::var("GPU_DB_BENCH_WAL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("target/wal-bench"));
    if durable {
        std::fs::create_dir_all(&wal_dir)?;
    }

    println!("Phase-D concurrent-commit SLO benchmark (closed-loop, single-row INSERT waves)");
    println!(
        "  arm={}  duration/point={seconds}s  SLO: >100k sustained / >=400k burst / p50 p99 p99.9 < 0.5 1 5 ms",
        if durable { "DURABLE WAL (fsync-bound)" } else { "in-memory WAL (CC ceiling)" }
    );
    println!();
    println!(
        "  writers | sustained TPS | burst TPS |   p50 |   p99 | p99.9 |   max | fsyncs | mean grp"
    );
    println!(
        "  ------- | ------------- | --------- | ----- | ----- | ----- | ----- | ------ | --------"
    );

    for &writers in &writers_sweep {
        let segment = wal_dir.join(format!("slo-c{writers}.wal"));
        let _ = std::fs::remove_file(&segment);
        let e = if durable {
            Engine::with_durable_wal_segment(&segment)
        } else {
            Engine::new_local()
        };
        // TYPE-COVERAGE track 1 (constrained elision): GPU_DB_BENCH_PK=1 declares the PK — the
        // core-banking table shape. Today a unique-indexed table is elision-INELIGIBLE and its
        // INSERT prepare pays the O(table) candidate scan (prepare_insert), so this arm is the
        // baseline the constrained-elision slice must move.
        if std::env::var("GPU_DB_BENCH_DATE").is_ok_and(|v| v == "1") {
            // TYPE-COVERAGE track 2: the Date/Int2 PK'd shape — every i32-section type
            // elides + validates device-side (pair with GPU_DB_BENCH_ELIDE/CELIDE).
            e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, d DATE, v INT2)")?;
        } else if std::env::var("GPU_DB_BENCH_PK").is_ok_and(|v| v == "1") {
            e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")?;
        } else {
            e.execute_text(1, "CREATE TABLE t (id INT, v INT)")?;
        }
        // RETIREMENT A4e A/B: GPU_DB_BENCH_ADMIT=1 = the honest baseline (auto-admit ON, the
        // dual-store commit path); GPU_DB_BENCH_ELIDE=1 = the elision arm on top of it.
        if std::env::var("GPU_DB_BENCH_ADMIT").is_ok_and(|v| v == "1")
            || std::env::var("GPU_DB_BENCH_ELIDE").is_ok_and(|v| v == "1")
        {
            e.set_auto_admit_on_commit(true);
        }
        if std::env::var("GPU_DB_BENCH_ELIDE").is_ok_and(|v| v == "1") {
            e.set_host_install_elision_enabled(true);
        }
        // TYPE-COVERAGE track 1: GPU_DB_BENCH_CELIDE=1 = the constrained-elision arm (PK'd
        // tables become elision-eligible; pair with GPU_DB_BENCH_PK=1 + GPU_DB_BENCH_ELIDE=1).
        if std::env::var("GPU_DB_BENCH_CELIDE").is_ok_and(|v| v == "1") {
            e.set_constrained_elision_enabled(true);
        }

        let engine = Arc::new(e);
        let stop = Arc::new(AtomicBool::new(false));
        let txn_ids = Arc::new(AtomicU64::new(2));
        let barrier = Arc::new(Barrier::new(writers + 1));
        let handles: Vec<_> = (0..writers)
            .map(|w| {
                let engine = Arc::clone(&engine);
                let stop = Arc::clone(&stop);
                let txn_ids = Arc::clone(&txn_ids);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut latencies_nanos = Vec::with_capacity(1 << 20);
                    let mut completions_millis = Vec::with_capacity(1 << 20);
                    let mut prepare_nanos_total = 0u64;
                    barrier.wait();
                    let run_started = Instant::now();
                    let mut i = 0_u64;
                    while !stop.load(Ordering::Relaxed) {
                        let txn_id = txn_ids.fetch_add(1, Ordering::Relaxed);
                        let id = w as u64 * 10_000_000 + i; // stays within int4 for <=200 writers
                        let sql = if std::env::var("GPU_DB_BENCH_DATE").is_ok_and(|v| v == "1") {
                            format!("INSERT INTO t (id, d, v) VALUES ({id}, '2026-07-03', 1)")
                        } else {
                            format!("INSERT INTO t (id, v) VALUES ({id}, 1)")
                        };
                        let commit_started = Instant::now();
                        let prepared_nanos = std::cell::Cell::new(0u64);
                        engine
                            .execute_dml_concurrent_instrumented(txn_id, &sql, || {
                                prepared_nanos.set(commit_started.elapsed().as_nanos() as u64)
                            })
                            .unwrap();
                        prepare_nanos_total += prepared_nanos.get();
                        latencies_nanos.push(commit_started.elapsed().as_nanos() as u64);
                        completions_millis.push(run_started.elapsed().as_millis() as u64);
                        i += 1;
                    }
                    WriterSample {
                        latencies_nanos,
                        completions_millis,
                        prepare_nanos_total,
                    }
                })
            })
            .collect();

        barrier.wait();
        let started = Instant::now();
        std::thread::sleep(Duration::from_secs(seconds));
        stop.store(true, Ordering::Relaxed);
        let samples: Vec<WriterSample> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let elapsed = started.elapsed();

        let mut latencies: Vec<u64> = samples
            .iter()
            .flat_map(|s| s.latencies_nanos.iter().copied())
            .collect();
        latencies.sort_unstable();
        let total = latencies.len();

        // Burst = the best 100 ms completion window across the run, scaled to 1 s.
        let horizon_millis = elapsed.as_millis() as u64 + 1;
        let mut per_ms = vec![0_u64; horizon_millis as usize + 1];
        for s in &samples {
            for &m in &s.completions_millis {
                if (m as usize) < per_ms.len() {
                    per_ms[m as usize] += 1;
                }
            }
        }
        let burst = per_ms
            .windows(100)
            .map(|w| w.iter().sum::<u64>())
            .max()
            .unwrap_or(0)
            * 10;

        {
            let prep: u64 = samples.iter().map(|s| s.prepare_nanos_total).sum();
            eprintln!(
                "    [off-lock prepare mean: {:.1} us/commit]",
                prep as f64 / total.max(1) as f64 / 1e3
            );
            use std::sync::atomic::Ordering;
            let w = &gpu_db_engine::engine_dml_concurrent_wave_stats();
            let (waves, items, nanos) = (
                w[0].swap(0, Ordering::Relaxed),
                w[1].swap(0, Ordering::Relaxed),
                w[2].swap(0, Ordering::Relaxed),
            );
            eprintln!(
                "    [waves: {waves}  mean items/wave {:.1}  mean us/item {:.1}]",
                items as f64 / waves.max(1) as f64,
                nanos as f64 / items.max(1) as f64 / 1e3,
            );
            // Elision/validator engagement (constrained-elision A/B): steady state = elisions
            // GROWING, the table STILL elided at teardown, device validate answering.
            eprintln!(
                "    [elisions: {}  still-elided: {}  device-validate-hits: {}]",
                engine.host_install_elisions(),
                engine.table_install_elided("t"),
                engine.dml_device_validate_hits(),
            );
            let (wx, px, rb) = engine.pk_index_maintenance_stats();
            eprintln!("    [pk-index: writer-extends {wx}  prober-extends {px}  rebuilds {rb}]");
        }
        let stats = engine.wal_group_commit_stats();
        let fsyncs = stats.flush_groups;
        println!(
            "  {:>7} | {:>13.0} | {:>9} | {:>4.2}ms | {:>4.2}ms | {:>4.2}ms | {:>4.1}ms | {:>6} | {:>8.1}",
            writers,
            total as f64 / elapsed.as_secs_f64(),
            burst,
            percentile(&latencies, 0.50) as f64 / 1e6,
            percentile(&latencies, 0.99) as f64 / 1e6,
            percentile(&latencies, 0.999) as f64 / 1e6,
            latencies.last().copied().unwrap_or(0) as f64 / 1e6,
            fsyncs,
            stats.mean_group_size(),
        );

        drop(samples);
        drop(engine);
        if durable {
            let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&segment));
            let _ = std::fs::remove_file(&segment);
        }
    }
    Ok(())
}
