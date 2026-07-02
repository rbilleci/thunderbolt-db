//! Group-commit benchmark for the CONCURRENT DML path (write-path assessment D3b / ledger #7).
//!
//! N writer threads drive `execute_dml_concurrent` INSERTs against a crash-durable engine. With
//! the designated-flusher group commit, committers that arrive while an fsync is in flight append
//! + apply and then share the NEXT fsync — so fsyncs per commit drop toward 1/G (G = mean group
//! size) instead of the strict 1 of the old flush-inside-the-commit-mutex path. The 1-writer run
//! IS the old cost model (groups of 1, one fsync per commit): compare commits/sec at c=1 vs c=N
//! and the reported mean group size.
//!
//! CPU + disk only — no GPU residency, no CUDA initialization.
//!
//! Run:  cargo run --release -p gpu_db_engine --example wal_group_commit_benchmark
//! Env:  GPU_DB_BENCH_WRITERS (default "1,2,4,8"), GPU_DB_BENCH_COMMITS_PER_WRITER (default 200),
//!       GPU_DB_BENCH_WAL_DIR (default target/wal-bench — put it on a REAL disk, not tmpfs, to
//!       see fsync-bound behavior).

use std::error::Error;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use gpu_db_engine::Engine;

fn main() -> Result<(), Box<dyn Error>> {
    let writers_sweep: Vec<usize> = std::env::var("GPU_DB_BENCH_WRITERS")
        .unwrap_or_else(|_| "1,2,4,8".to_string())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let commits_per_writer: usize = std::env::var("GPU_DB_BENCH_COMMITS_PER_WRITER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let wal_dir = std::env::var("GPU_DB_BENCH_WAL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("target/wal-bench"));
    std::fs::create_dir_all(&wal_dir)?;

    println!("WAL group-commit benchmark (concurrent DML path, durable segment)");
    println!(
        "  commits/writer={commits_per_writer}  wal_dir={} (fsync latency depends on this disk)",
        wal_dir.display()
    );
    println!();
    println!("  writers |   commits |  elapsed |  commits/s | fsyncs | mean group | max group");
    println!("  ------- | --------- | -------- | ---------- | ------ | ---------- | ---------");

    for &writers in &writers_sweep {
        let segment = wal_dir.join(format!("group-bench-c{writers}.wal"));
        let _ = std::fs::remove_file(&segment);
        let e = Engine::with_durable_wal_segment(&segment);
        e.execute_text(1, "CREATE TABLE t (id INT, v INT)")?;

        let engine = Arc::new(e);
        let barrier = Arc::new(Barrier::new(writers + 1));
        let handles: Vec<_> = (0..writers)
            .map(|w| {
                let engine = Arc::clone(&engine);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for i in 0..commits_per_writer {
                        let id = (w * commits_per_writer + i) as u64;
                        engine
                            .execute_dml_concurrent(
                                2 + id,
                                &format!("INSERT INTO t (id, v) VALUES ({id}, {id})"),
                            )
                            .unwrap();
                    }
                })
            })
            .collect();
        barrier.wait();
        let started = Instant::now();
        for handle in handles {
            handle.join().unwrap();
        }
        let elapsed = started.elapsed();

        let total = writers * commits_per_writer;
        let stats = engine.wal_group_commit_stats();
        // Subtract the CREATE TABLE's serialized flush group from the reported numbers.
        let fsyncs = stats.flush_groups.saturating_sub(1);
        println!(
            "  {:>7} | {:>9} | {:>7.2}s | {:>10.0} | {:>6} | {:>10.2} | {:>9}",
            writers,
            total,
            elapsed.as_secs_f64(),
            total as f64 / elapsed.as_secs_f64(),
            fsyncs,
            total as f64 / fsyncs.max(1) as f64,
            stats.max_group_size,
        );

        drop(engine);
        let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&segment));
        let _ = std::fs::remove_file(&segment);
    }
    Ok(())
}
