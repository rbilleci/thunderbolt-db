//! WAL fsync / commit microbenchmark (Thread-4 write-half, Stage 1).
//!
//! Measures the durability cost the real-fsync commit path adds, and demonstrates that the
//! group-commit mechanism batches multiple WAL records into a single fsync.
//!
//! Three measurements:
//!   1. **In-memory baseline** — commits through an `Engine::new_local` (WAL not fsynced). The
//!      lower bound: everything *except* the durability fsync.
//!   2. **Durable, size-1 groups (Stage 1 today)** — commits through an `Engine::with_durable_wal_segment`.
//!      The writer is serialized, so every commit drives its own `flush_all` => one fsync per commit
//!      (group size 1). The delta vs. the baseline is the per-commit durability cost.
//!   3. **Group-commit amortization (the Stage-4 shape)** — drives the `WalBuffer` directly: N records
//!      appended, then ONE `flush_all` => a single fsync for N records (group size N). This is what a
//!      designated flusher will achieve once Stage 4 lets multiple committers enqueue before a flush.
//!      Reported as the records-per-fsync ratio and the amortized per-record fsync cost.
//!
//! Run:  cargo run -p gpu_db_engine --example wal_fsync_commit_benchmark
//! Env:  GPU_DB_WAL_BENCH_COMMITS (default 2000), GPU_DB_WAL_BENCH_GROUP (default 64)

use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

use gpu_db_engine::Engine;
use gpu_db_wal::{WalBuffer, WalRecord};

fn micros_per(total: Duration, n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        total.as_secs_f64() * 1e6 / n as f64
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let commits = env::var("GPU_DB_WAL_BENCH_COMMITS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2_000);
    let group = env::var("GPU_DB_WAL_BENCH_GROUP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1);

    let tmp_dir = env::temp_dir().join(format!("gpu-db-wal-bench-{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir)?;

    // --- 1. In-memory baseline: commit cost without the durability fsync. -----------------------
    let mut mem = Engine::new_local();
    mem.execute_text(1, "CREATE TABLE t (id INT)")?;
    let started = Instant::now();
    for i in 0..commits {
        mem.execute_text((i + 2) as u64, &format!("INSERT INTO t (id) VALUES ({i})"))?;
    }
    let mem_elapsed = started.elapsed();

    // --- 2. Durable, serialized writer: one fsync per commit (group size 1). --------------------
    let segment_path = tmp_dir.join("commit-path.wal");
    let mut durable = Engine::with_durable_wal_segment(&segment_path);
    durable.execute_text(1, "CREATE TABLE t (id INT)")?;
    let started = Instant::now();
    for i in 0..commits {
        durable.execute_text((i + 2) as u64, &format!("INSERT INTO t (id) VALUES ({i})"))?;
    }
    let durable_elapsed = started.elapsed();
    let stats = durable.wal_group_commit_stats();
    drop(durable);

    // --- 3. Group-commit amortization: N records per fsync (the Stage-4 shape). ------------------
    // Drive the WalBuffer directly so the only work measured is append + the batched fsync.
    let group_segment = tmp_dir.join("group-commit.wal");
    let groups = commits.div_ceil(group);
    let mut wal = WalBuffer::with_durable_segment(&group_segment);
    let started = Instant::now();
    let mut next = 0_u64;
    for _ in 0..groups {
        for _ in 0..group {
            wal.append(WalRecord {
                txn_id: next,
                payload: b"INSERT INTO t (id) VALUES (0)".to_vec(),
            });
            next += 1;
        }
        wal.flush_all()?; // ONE fsync for `group` records.
    }
    let group_elapsed = started.elapsed();
    let group_stats = wal.group_commit_stats();
    let group_records = next as usize;
    drop(wal);

    let _ = std::fs::remove_dir_all(&tmp_dir);

    println!("WAL fsync / commit microbenchmark (Stage 1)");
    println!("  commits={commits}  group_size={group}");
    println!();
    println!(
        "  1. in-memory baseline       : {:>9.2} us/commit  ({:?} total)",
        micros_per(mem_elapsed, commits),
        mem_elapsed
    );
    println!(
        "  2. durable, 1 fsync/commit  : {:>9.2} us/commit  ({:?} total)",
        micros_per(durable_elapsed, commits),
        durable_elapsed
    );
    println!(
        "       => per-commit durability cost (fsync): {:>9.2} us",
        micros_per(durable_elapsed, commits) - micros_per(mem_elapsed, commits)
    );
    println!(
        "       => group-commit stats: flush_groups={} durable_records={} mean_group_size={:.2}",
        stats.flush_groups,
        stats.durable_records,
        stats.mean_group_size()
    );
    assert_eq!(
        stats.mean_group_size(),
        1.0,
        "the serialized Stage-1 writer must produce size-1 groups"
    );
    println!();
    println!(
        "  3. group-commit ({group} recs/fsync): {:>9.2} us/record  ({:?} total, {} fsyncs)",
        micros_per(group_elapsed, group_records),
        group_elapsed,
        group_stats.flush_groups
    );
    println!(
        "       => mean_group_size={:.2}  (records per fsync — the Stage-4 amortization)",
        group_stats.mean_group_size()
    );
    println!(
        "       => fsync amortization vs size-1: {:.1}x fewer fsyncs for the same record count",
        group_stats.mean_group_size()
    );
    assert!(
        group_stats.mean_group_size() > 1.0 || group <= 1,
        "group commit must batch >1 record per fsync when group_size > 1"
    );

    Ok(())
}
