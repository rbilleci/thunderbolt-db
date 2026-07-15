//! W1b — RECOVERY-TIME probe (durability mandate report card): measures restart time
//! (`open_durable_wal_segment_auto`) as a function of committed history, before and after an
//! auto-checkpoint rotation. HONESTY NOTE: rotation bounds the LIVE segment (fast fsync path +
//! bounded torn-tail scan); total REPLAY work is still O(full history) until the
//! resolved-change-record format (W5) + state snapshots land — this probe prints both so the
//! bound and the residual are visible.
//!
//! Run: cargo run --release -p gpu_db_engine --example wal_recovery_time_probe
//! Env: GPU_DB_PROBE_ROWS (default 20000); GPU_DB_PROBE_BINARY=1 = W5a binary records
//! (covered concurrent inserts; replay = decode+install, no SQL parse)

use gpu_db_engine::Engine;
use std::time::Instant;

fn main() {
    let rows: u64 = std::env::var("GPU_DB_PROBE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let dir = std::env::temp_dir().join("gpu-db-recovery-probe");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wal.segment");
    let control = gpu_db_wal::wal_checkpoint_control_path(&path);
    let checkpoint = gpu_db_wal::wal_checkpoint_segment_path(&path);

    let binary = std::env::var("GPU_DB_PROBE_BINARY").is_ok_and(|v| v == "1");
    let live_bytes;
    {
        let e = Engine::with_durable_wal_segment(&path);
        e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
        if binary {
            e.set_binary_wal_records_enabled(true);
            for i in 0..rows {
                e.execute_dml_concurrent(2 + i, &format!("INSERT INTO t (id, v) VALUES ({i}, 1)"))
                    .unwrap();
            }
        } else {
            for i in 0..rows {
                e.execute_text(2 + i, &format!("INSERT INTO t (id, v) VALUES ({i}, 1)"))
                    .unwrap();
            }
        }
        live_bytes = e.wal_durable_segment_bytes();
    }
    let t = Instant::now();
    let e = Engine::open_durable_wal_segment_auto(&path).unwrap();
    let open_pre_rotation = t.elapsed();
    println!(
        "pre-rotation : {rows} rows | live segment {:.2}MB logical | open {:?}",
        live_bytes as f64 / 1e6,
        open_pre_rotation
    );

    // Rotate, then commit a small suffix and measure the post-rotation restart.
    e.checkpoint_and_truncate_durable_wal_if_larger_than(&control, &checkpoint, 1)
        .unwrap();
    for i in 0..64u64 {
        e.execute_text(
            1_000_000 + i,
            &format!("INSERT INTO t (id, v) VALUES ({}, 2)", rows + i),
        )
        .unwrap();
    }
    let live_bytes = e.wal_durable_segment_bytes();
    drop(e);
    let t = Instant::now();
    let e = Engine::open_durable_wal_segment_auto(&path).unwrap();
    let open_post_rotation = t.elapsed();
    println!(
        "post-rotation: {} rows | live segment {:.2}MB logical (bounded by rotation) | open {:?} (replay still O(history) until W5)",
        rows + 64,
        live_bytes as f64 / 1e6,
        open_post_rotation
    );
    drop(e);
    let _ = std::fs::remove_dir_all(&dir);
}
