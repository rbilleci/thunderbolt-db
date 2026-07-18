//! Phase-D CONCURRENT-COMMIT SLO benchmark (HANDOVER (D); the write-side analog of
//! `r2_wave_engine_ab`): measures sustained + burst commit TPS and per-commit latency
//! percentiles for the concurrent DML path (`execute_dml_concurrent`). This is an isolated W1
//! INSERT characterization: its latency reference is **p50/p99/p99.9 < 0.8/1.5/5 ms**, while its
//! TPS is diagnostic only. The binding >100k sustained / ≥400k burst gate belongs to the
//! canonical mixed-system BENCH-001 workload, not this isolated single-operation benchmark.
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
//! The relational generation is device-authoritative; the durable arm additionally measures disk.
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
    let binary_wal = std::env::var("GPU_DB_BENCH_BINWAL").is_ok_and(|v| v == "1");
    let wal_dir = std::env::var("GPU_DB_BENCH_WAL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("target/wal-bench"));
    if durable {
        std::fs::create_dir_all(&wal_dir)?;
    }

    println!("Phase-D concurrent-commit SLO benchmark (closed-loop, single-row INSERT waves)");
    println!(
        "  arm={}  duration/point={seconds}s  W1 latency: p50 p99 p99.9 < 0.8 1.5 5 ms; TPS diagnostic only",
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
        // GPU_DB_BENCH_PK=1 declares the core-banking primary-key shape.
        if std::env::var("GPU_DB_BENCH_INT8").is_ok_and(|v| v == "1") {
            // TYPE-COVERAGE track 2 slice 2: the int4-keyed / i64-payload core-banking shape
            // (BIGINT balances). The i64 section is device-resident by default.
            e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v BIGINT)")?;
        } else if std::env::var("GPU_DB_BENCH_DATE").is_ok_and(|v| v == "1") {
            // TYPE-COVERAGE track 2: the Date/Int2 PK'd device-validation shape.
            e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, d DATE, v INT2)")?;
        } else if std::env::var("GPU_DB_BENCH_PK").is_ok_and(|v| v == "1") {
            e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")?;
        } else {
            e.execute_text(1, "CREATE TABLE t (id INT, v INT)")?;
        }
        if binary_wal {
            // W5a: covered inserts log resolved binary records (decode+install replay).
            e.set_binary_wal_records_enabled(true);
        }
        // i64 sections are DEFAULT ON since the 2026-07-03 flip; GPU_DB_BENCH_I64SHARDS=0 is
        // the kill-switch A/B arm (=1 remains accepted, now redundant).
        match std::env::var("GPU_DB_BENCH_I64SHARDS").as_deref() {
            Ok("0") => e.set_shard_int8_section_enabled(false),
            Ok("1") => e.set_shard_int8_section_enabled(true),
            _ => {}
        }
        // M1 design B: GPU_DB_BENCH_WAVEBATCH=1 = wave-time batched PK-unique validation.
        if std::env::var("GPU_DB_BENCH_WAVEBATCH").is_ok_and(|v| v == "1") {
            e.set_device_write_locate_wave_batch_enabled(true);
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
                    // lpb-for-writes probe: GPU_DB_BENCH_ROWS=N inserts N rows per commit (the
                    // write-path deep-batch analog of a 65536-needle read batch). rows/sec =
                    // commits/sec * N.
                    let rows_per: u64 = std::env::var("GPU_DB_BENCH_ROWS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1)
                        .max(1);
                    while !stop.load(Ordering::Relaxed) {
                        let txn_id = txn_ids.fetch_add(1, Ordering::Relaxed);
                        // Per-writer disjoint id ranges (no cross-writer collision); consecutive
                        // within a writer. Keep writers*128M + rows under i32 for the multi-row probe.
                        let base = w as u64 * 128_000_000 + i * rows_per;
                        let sql = if rows_per > 1 {
                            let vals: Vec<String> = (0..rows_per)
                                .map(|r| format!("({}, 1)", base + r))
                                .collect();
                            format!("INSERT INTO t (id, v) VALUES {}", vals.join(","))
                        } else {
                            // W4 open-loop sweeps: 4M per writer keeps ids within int4 up to
                            // ~536 writers (536 * 4M ~= i32::MAX); a 60s point at 10k/writer/s
                            // stays well under the 4M per-writer budget.
                            let id = w as u64 * 4_000_000 + i;
                            if std::env::var("GPU_DB_BENCH_INT8").is_ok_and(|v| v == "1") {
                                format!(
                                    "INSERT INTO t (id, v) VALUES ({id}, {})",
                                    5_000_000_000_i64 + id as i64
                                )
                            } else if std::env::var("GPU_DB_BENCH_DATE").is_ok_and(|v| v == "1") {
                                format!("INSERT INTO t (id, d, v) VALUES ({id}, '2026-07-03', 1)")
                            } else {
                                format!("INSERT INTO t (id, v) VALUES ({id}, 1)")
                            }
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
            // FUSE recon (GPU_DB_BENCH_DEVPHASE=1): per-wave device round-trip attribution.
            let d = &gpu_db_engine::engine_dml_concurrent_wave_device_stats();
            let (loc, app, idx) = (
                d[0].swap(0, Ordering::Relaxed),
                d[1].swap(0, Ordering::Relaxed),
                d[2].swap(0, Ordering::Relaxed),
            );
            if loc + app + idx > 0 {
                eprintln!(
                    "    [dev-phase us/wave: locate {:.1}  append {:.1}  index-insert {:.1}]",
                    loc as f64 / waves.max(1) as f64 / 1e3,
                    app as f64 / waves.max(1) as f64 / 1e3,
                    idx as f64 / waves.max(1) as f64 / 1e3,
                );
            }
            // HOST-phase recon (GPU_DB_BENCH_HOSTPHASE=1): the serial work under the commit_mutex.
            let h = &gpu_db_engine::engine_dml_concurrent_wave_host_stats();
            let hp: Vec<u64> = (0..7).map(|i| h[i].swap(0, Ordering::Relaxed)).collect();
            if hp.iter().sum::<u64>() > 0 {
                let per = |n: u64| n as f64 / items.max(1) as f64 / 1e3;
                eprintln!(
                    "    [host-phase us/item: validate {:.2} conflict {:.2} reresolve {:.2} \
                     sequence {:.2} ledger {:.2} apply {:.2} invalidate {:.2}]",
                    per(hp[0]),
                    per(hp[1]),
                    per(hp[2]),
                    per(hp[3]),
                    per(hp[4]),
                    per(hp[5]),
                    per(hp[6]),
                );
            }
            // Device-authority and validator engagement.
            eprintln!(
                "    [device-authoritative commits: {}  authoritative: {}  device-validate-hits: {}]",
                engine.device_authoritative_commits(),
                engine.table_device_authoritative("t"),
                engine.dml_device_validate_hits(),
            );
            eprintln!(
                "    [device-pk-locate hits: {}]",
                engine.device_write_locate_hits()
            );
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
