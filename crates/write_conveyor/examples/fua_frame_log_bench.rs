//! Smoke benchmark for the variable-payload FUA frame log: one appender
//! pacing on the fence pool, engine-shaped frame sizes, durable cut, scan
//! recovery validation. Confirms the frame log keeps the fixed-record lane's
//! fence economics (~28K+ durable fences/s at qd=16 on the reference host).
//!
//! ```bash
//! CONVEYOR_FUA_QD=16 CONVEYOR_EVENTS=200000 \
//!   cargo run --release -p gpu_db_write_conveyor --example fua_frame_log_bench
//! ```

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use gpu_db_write_conveyor::{recover_frame_log_by_scan, FuaFrameLog, FuaFrameLogConfig};

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn Error>> {
    let frames: u64 = parse_env("CONVEYOR_EVENTS", 200_000_u64);
    let fence_qd: usize = parse_env("CONVEYOR_FUA_QD", 16_usize).max(1);
    // engine-shaped mix: mostly small binary row-op batches, some larger
    let payload_sizes = [448_usize, 448, 960, 448, 1984, 448, 448, 4032];
    let max_padded = 4096 + 512;
    let path = PathBuf::from(format!("target/fua-frame-log-{}.dat", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let setup_start = Instant::now();
    let log = unsafe {
        FuaFrameLog::create(FuaFrameLogConfig {
            path: path.clone(),
            segment_id: 1,
            capacity_bytes: frames as usize * max_padded,
        })?
    };
    let setup = setup_start.elapsed();
    let pool = log.spawn_fence_pool(fence_qd);
    let mut appender = log.appender();
    let payload = vec![0xA5_u8; 4096];

    let run_start = Instant::now();
    let mut seq = 0_u64;
    for frame in 0..frames {
        while log.free_fence_slots(fence_qd) == 0 {
            if log.fence_failed() {
                return Err("fence lane failed".into());
            }
            std::hint::spin_loop();
        }
        let size = payload_sizes[(frame % payload_sizes.len() as u64) as usize];
        let count = (size / 64) as u32;
        appender.publish_frame(&payload[..size], seq, count)?;
        seq += count as u64;
    }
    appender.finish();
    let fences = pool.join()?;
    let elapsed = run_start.elapsed();
    assert_eq!(log.durable_frames(), frames);
    assert_eq!(log.durable_seq(), seq);
    drop(log);

    let recover_start = Instant::now();
    let recovered = recover_frame_log_by_scan(&path)?;
    let recover = recover_start.elapsed();
    assert_eq!(recovered.len() as u64, frames);
    assert_eq!(
        recovered.last().map(|f| f.first_seq + f.seq_count as u64),
        Some(seq)
    );
    let _ = std::fs::remove_file(&path);

    let frames_per_sec = frames as f64 / elapsed.as_secs_f64();
    let records_per_sec = seq as f64 / elapsed.as_secs_f64();
    println!(
        "fua-frame-log  {:.3} K fences/s  {:.3} M records-equiv/s  elapsed={:.3}s setup={:.3}s fence-qd={fence_qd} frames={frames} fences={fences} recover={:.3}s",
        frames_per_sec / 1_000.0,
        records_per_sec / 1_000_000.0,
        elapsed.as_secs_f64(),
        setup.as_secs_f64(),
        recover.as_secs_f64(),
    );
    Ok(())
}
