//! Production mixed read/write non-vacuity gate.
//!
//! Readers use the real facade `PointLookupBatcher` over a sharded resident int4-PK table
//! while concurrent facade writers append fresh rows. The run fails unless every read was admitted
//! by the batcher with no per-query fallback, GPU multi-shard index-probe batches fired inside the exact
//! writer-active interval, every write published device authority, and resident append waves fired.
//! Visibility-sensitive append windows must remain on the dense probe; any host-gather batch fails the gate.
//! Its >100k read-QPS floor is a gate-local non-vacuity/capacity control, not the charter's aggregate
//! committed-TPS target; BENCH-001 alone owns the canonical mixed-system throughput decision.
//!
//! Env: `GPU_DB_MIX_GPU_READERS` (32), `GPU_DB_MIX_GPU_WRITERS` (4),
//! `GPU_DB_MIX_GPU_READS_PER` (500), `GPU_DB_MIX_GPU_WRITES_PER` (40),
//! `GPU_DB_MIX_GPU_ROWS` (300), `GPU_DB_MIX_GPU_BATCH` (64),
//! `GPU_DB_MIX_GPU_WAIT_US` (0), `GPU_DB_MIX_GPU_WARMUP` (64),
//! `GPU_DB_MIX_GPU_READER_WARMUP` (2).

use std::env;
use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use gpu_db_engine::Engine;
use gpu_db_facade::{
    execute_on_shared_engine, execute_on_shared_engine_batched, BatchedDispatch, DbValue,
    PointLookupBatcher, QueryOutcome, SharedEngine,
};

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn expect_id(outcome: QueryOutcome, expected: i32) -> Result<(), String> {
    match outcome {
        QueryOutcome::Rows { rows, .. } if rows == vec![vec![DbValue::Int4(expected)]] => Ok(()),
        other => Err(format!(
            "point read for id={expected} returned an unexpected outcome: {other:?}"
        )),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let readers = env_usize("GPU_DB_MIX_GPU_READERS", 32);
    let writers = env_usize("GPU_DB_MIX_GPU_WRITERS", 4);
    let reads_per = env_usize("GPU_DB_MIX_GPU_READS_PER", 500);
    let writes_per = env_usize("GPU_DB_MIX_GPU_WRITES_PER", 40);
    let rows = env_usize("GPU_DB_MIX_GPU_ROWS", 300);
    let batch = env_usize("GPU_DB_MIX_GPU_BATCH", 64);
    let wait_us = env_usize("GPU_DB_MIX_GPU_WAIT_US", 0) as u64;
    let warmup = env_usize("GPU_DB_MIX_GPU_WARMUP", 64);
    let reader_warmup = env_usize("GPU_DB_MIX_GPU_READER_WARMUP", 2);
    const WRITE_BASE: usize = 1_000_000;
    const WRITE_KEY_STRIDE: usize = 10_000;
    if readers == 0
        || writers == 0
        || reads_per == 0
        || writes_per == 0
        || warmup == 0
        || reader_warmup == 0
        || rows < 2
    {
        return Err(
            "readers, writers, reads_per, writes_per, and warmups must be non-zero (rows >= 2)"
                .into(),
        );
    }
    if writes_per > WRITE_KEY_STRIDE {
        return Err(format!(
            "writes_per={writes_per} exceeds the non-overlapping writer-key stride {WRITE_KEY_STRIDE}"
        )
        .into());
    }
    let max_write_id = (writers - 1)
        .checked_mul(WRITE_KEY_STRIDE)
        .and_then(|offset| WRITE_BASE.checked_add(offset))
        .and_then(|base| base.checked_add(writes_per - 1))
        .ok_or("writer key range overflow")?;
    if max_write_id > i32::MAX as usize / 3 {
        return Err(format!(
            "writer key range ends at {max_write_id}; id and id*3 must both fit INT"
        )
        .into());
    }

    let engine = Engine::new_local();
    // Pin the production sharded layer: unlike the single-buffer kill-switch, its append path can
    // retain NULL-bearing data and birth visibility while writers and readers overlap.
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(64);
    engine.set_shard_index_probe_enabled(true);
    engine.set_shard_batched_point_read_enabled(true);
    if !engine.auto_admit_on_commit_enabled() {
        return Err(
            "STRATA S-F regression: production auto-admission is not enabled by default".into(),
        );
    }
    let shared = Arc::new(SharedEngine::from_engine(engine));
    execute_on_shared_engine(
        &shared,
        "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)",
    )?;
    for id in 0..rows {
        let balance = if id == 1 {
            "NULL".to_owned()
        } else {
            (id as i64 * 7).to_string()
        };
        execute_on_shared_engine(
            &shared,
            &format!("INSERT INTO accounts (id, balance) VALUES ({id}, {balance})"),
        )?;
    }

    let resident_status = shared.gpu_native_activity_snapshot("accounts");
    if resident_status.valid_resident_tables == 0 && resident_status.resident_shards == 0 {
        println!("gpu_mixed_read_write_gate: SKIP (no resident GPU table)");
        return Ok(());
    }
    let batcher = Arc::new(PointLookupBatcher::with_triggers(
        Arc::clone(&shared),
        batch,
        Duration::from_micros(wait_us),
    ));
    // Cold module/index construction is a deployment warmup concern, not steady-state OLTP latency. Exercise the
    // exact production facade route before opening the measured writer window, and report the cost separately so
    // it cannot disappear from the artifact. Enqueue first, then drain, so this warms the coalesced path rather
    // than N single-flight calls.
    let warmup_started = Instant::now();
    let mut warmup_receivers = Vec::with_capacity(warmup);
    for turn in 0..warmup {
        let id = (turn % rows) as i32;
        let sql = format!("SELECT id FROM accounts WHERE id = {id}");
        match execute_on_shared_engine_batched(&shared, &batcher, &sql) {
            BatchedDispatch::Batched(receiver) => warmup_receivers.push((id, receiver)),
            BatchedDispatch::Immediate(_) => {
                return Err(
                    format!("warmup point read id={id} bypassed the production batcher").into(),
                )
            }
        }
    }
    for (id, receiver) in warmup_receivers {
        let outcome = receiver
            .blocking_recv()
            .map_err(|_| "point-lookup batcher stopped during warmup")?
            .map_err(|err| format!("warmup read id={id}: {err:?}"))?;
        expect_id(outcome, id)?;
    }
    let warmup_elapsed = warmup_started.elapsed();
    let before = shared.gpu_native_activity_snapshot("accounts");
    let batcher_before = batcher.activity_snapshot();

    let active_readers = Arc::new(AtomicUsize::new(readers));
    let started_readers = Arc::new(AtomicUsize::new(0));
    let active_writers = Arc::new(AtomicUsize::new(0));
    let writers_remaining = Arc::new(AtomicUsize::new(writers));
    let writers_done = Arc::new(AtomicBool::new(false));
    let committed_writes = Arc::new(AtomicUsize::new(0));
    let overlapping_writes = Arc::new(AtomicUsize::new(0));
    let overlapping_reads = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(readers + writers + 1));
    let writer_barrier = Arc::new(Barrier::new(writers));
    let gpu_before_writes = Arc::new(AtomicU64::new(0));
    let gpu_batches_during_writes = Arc::new(AtomicU64::new(0));

    let writer_handles: Vec<_> = (0..writers)
        .map(|writer| {
            let shared = Arc::clone(&shared);
            let active_readers = Arc::clone(&active_readers);
            let started_readers = Arc::clone(&started_readers);
            let active_writers = Arc::clone(&active_writers);
            let writers_remaining = Arc::clone(&writers_remaining);
            let writers_done = Arc::clone(&writers_done);
            let committed_writes = Arc::clone(&committed_writes);
            let overlapping_writes = Arc::clone(&overlapping_writes);
            let barrier = Arc::clone(&barrier);
            let writer_barrier = Arc::clone(&writer_barrier);
            let gpu_before_writes = Arc::clone(&gpu_before_writes);
            let gpu_batches_during_writes = Arc::clone(&gpu_batches_during_writes);
            thread::spawn(move || -> Result<(), String> {
                barrier.wait();
                while started_readers.load(Ordering::Acquire) < readers {
                    std::hint::spin_loop();
                }
                active_writers.fetch_add(1, Ordering::AcqRel);
                if writer == 0 {
                    gpu_before_writes.store(
                        shared
                            .gpu_native_activity_snapshot("accounts")
                            .sharded_gpu_probe_batches,
                        Ordering::Release,
                    );
                }
                writer_barrier.wait();
                let result = (|| {
                    for offset in 0..writes_per {
                        let id = WRITE_BASE + writer * WRITE_KEY_STRIDE + offset;
                        execute_on_shared_engine(
                            &shared,
                            &format!(
                                "INSERT INTO accounts (id, balance) VALUES ({id}, {})",
                                id * 3
                            ),
                        )
                        .map_err(|err| format!("writer {writer} id={id}: {err:?}"))?;
                        committed_writes.fetch_add(1, Ordering::Relaxed);
                        if active_readers.load(Ordering::Acquire) > 0 {
                            overlapping_writes.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Ok(())
                })();
                writer_barrier.wait();
                if writer == 0 {
                    let gpu_after = shared
                        .gpu_native_activity_snapshot("accounts")
                        .sharded_gpu_probe_batches;
                    gpu_batches_during_writes.store(
                        gpu_after.saturating_sub(gpu_before_writes.load(Ordering::Acquire)),
                        Ordering::Release,
                    );
                }
                writer_barrier.wait();
                active_writers.fetch_sub(1, Ordering::AcqRel);
                if writers_remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                    writers_done.store(true, Ordering::Release);
                }
                result
            })
        })
        .collect();

    let reader_handles: Vec<_> = (0..readers)
        .map(|reader| {
            let shared = Arc::clone(&shared);
            let batcher = Arc::clone(&batcher);
            let active_readers = Arc::clone(&active_readers);
            let started_readers = Arc::clone(&started_readers);
            let active_writers = Arc::clone(&active_writers);
            let writers_done = Arc::clone(&writers_done);
            let overlapping_reads = Arc::clone(&overlapping_reads);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || -> Result<(Vec<u64>, Vec<u64>), String> {
                let mut state =
                    0x9e37_79b9_7f4a_7c15_u64 ^ (reader as u64 + 1).wrapping_mul(2_654_435_761);
                let mut latencies = Vec::with_capacity(reads_per);
                let mut writer_active_latencies = Vec::new();
                barrier.wait();
                // Remove per-thread scheduler/channel cold start from the sustained-latency sample. The global
                // warmup above builds modules/indexes; this one proves every reader is already circulating through
                // the production batcher before writers open the measured overlap window.
                for warm_turn in 0..reader_warmup {
                    let id = ((reader + warm_turn) % rows) as i32;
                    let sql = format!("SELECT id FROM accounts WHERE id = {id}");
                    let outcome = match execute_on_shared_engine_batched(&shared, &batcher, &sql) {
                        BatchedDispatch::Batched(receiver) => receiver
                            .blocking_recv()
                            .expect("batcher stayed alive during reader warmup")
                            .expect("reader warmup succeeded"),
                        BatchedDispatch::Immediate(_) => {
                            panic!("reader warmup bypassed the production batcher")
                        }
                    };
                    expect_id(outcome, id).expect("reader warmup returned the expected row");
                }
                started_readers.fetch_add(1, Ordering::Release);
                let mut turns = 0usize;
                loop {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let id = (state % rows as u64) as i32;
                    let sql = format!("SELECT id FROM accounts WHERE id = {id}");
                    let started = Instant::now();
                    let result = match execute_on_shared_engine_batched(&shared, &batcher, &sql) {
                        BatchedDispatch::Batched(receiver) => receiver
                            .blocking_recv()
                            .map_err(|_| "point-lookup batcher stopped".to_owned())?
                            .map_err(|err| format!("batched read id={id}: {err:?}"))?,
                        BatchedDispatch::Immediate(_) => {
                            return Err(format!(
                                "resident point read id={id} bypassed the production batcher"
                            ));
                        }
                    };
                    expect_id(result, id)?;
                    let latency_us = started.elapsed().as_micros() as u64;
                    latencies.push(latency_us);
                    if active_writers.load(Ordering::Acquire) > 0 {
                        writer_active_latencies.push(latency_us);
                        overlapping_reads.fetch_add(1, Ordering::Relaxed);
                    }
                    turns += 1;
                    if turns >= reads_per && writers_done.load(Ordering::Acquire) {
                        break;
                    }
                }
                active_readers.fetch_sub(1, Ordering::Release);
                Ok((latencies, writer_active_latencies))
            })
        })
        .collect();

    barrier.wait();
    let started = Instant::now();
    for handle in writer_handles {
        handle.join().map_err(|_| "writer panicked")??;
    }
    let mut latencies = Vec::with_capacity(readers * reads_per);
    let mut writer_active_latencies = Vec::new();
    for handle in reader_handles {
        let (reader_latencies, reader_writer_active_latencies) =
            handle.join().map_err(|_| "reader panicked")??;
        latencies.extend(reader_latencies);
        writer_active_latencies.extend(reader_writer_active_latencies);
    }
    let elapsed = started.elapsed();
    latencies.sort_unstable();
    writer_active_latencies.sort_unstable();

    let after = shared.gpu_native_activity_snapshot("accounts");
    let gpu_batches = after
        .sharded_gpu_probe_batches
        .saturating_sub(before.sharded_gpu_probe_batches);
    let sharded_batches = after
        .sharded_point_batches
        .saturating_sub(before.sharded_point_batches);
    let binary_batches = after
        .sharded_binary_route_batches
        .saturating_sub(before.sharded_binary_route_batches);
    let appends = after
        .open_shard_append_commits
        .saturating_sub(before.open_shard_append_commits);
    let elisions = after
        .device_authoritative_commits
        .saturating_sub(before.device_authoritative_commits);
    let writes = committed_writes.load(Ordering::Relaxed);
    let overlaps = overlapping_writes.load(Ordering::Relaxed);
    let read_overlaps = overlapping_reads.load(Ordering::Relaxed);
    let gpu_write_overlap = gpu_batches_during_writes.load(Ordering::Acquire);
    let batcher_after = batcher.activity_snapshot();
    let sharded_batched_groups = batcher_after
        .sharded_batched_groups
        .saturating_sub(batcher_before.sharded_batched_groups);
    let sharded_fallback_groups = batcher_after
        .sharded_per_query_fallback_groups
        .saturating_sub(batcher_before.sharded_per_query_fallback_groups);
    if gpu_batches == 0
        || gpu_write_overlap == 0
        || gpu_batches != sharded_batches
        || sharded_batches != sharded_batched_groups
        || sharded_fallback_groups != 0
    {
        return Err(format!(
            "GPU/sharded route proof failed: gpu_batches={gpu_batches} \
             gpu_batches_during_writes={gpu_write_overlap} \
             sharded_batches={sharded_batches} batcher_groups={} fallbacks={} \
             binary_batches={binary_batches} appends={appends} elisions={elisions} \
             gpu_raw={}->{} binary_raw={}->{}",
            sharded_batched_groups,
            sharded_fallback_groups,
            before.sharded_gpu_probe_batches,
            after.sharded_gpu_probe_batches,
            before.sharded_binary_route_batches,
            after.sharded_binary_route_batches
        )
        .into());
    }
    let expected_writes = writers * writes_per;
    if writes != expected_writes
        || overlaps != writes
        || read_overlaps == 0
        || appends == 0
        || elisions != writes as u64
    {
        return Err(format!(
            "mixed overlap/write proof failed: expected_writes={expected_writes} writes={writes} \
             overlapping_writes={overlaps} overlapping_reads={read_overlaps} \
             appends={appends} elisions={elisions}"
        )
        .into());
    }

    let total_reads = latencies.len();
    let read_qps = total_reads as f64 / elapsed.as_secs_f64();
    let p50 = percentile(&latencies, 0.50);
    let p99 = percentile(&latencies, 0.99);
    let p999 = percentile(&latencies, 0.999);
    let writer_active_p99 = percentile(&writer_active_latencies, 0.99);
    let writer_active_p999 = percentile(&writer_active_latencies, 0.999);
    if read_qps <= 100_000.0 || p50 >= 500 || p99 >= 1_000 || p999 >= 5_000 {
        return Err(format!(
            "mixed GPU route gate failed: read_qps={read_qps:.0}/s (gate-local floor >100000; not system TPS) \
             p50={p50}us (target <500) p99={p99}us (target <1000) \
             p99.9={p999}us (target <5000); writer-active samples={} \
             p99={writer_active_p99}us p99.9={writer_active_p999}us",
            writer_active_latencies.len()
        )
        .into());
    }
    println!(
        "gpu_mixed_read_write_gate: PASS readers={readers} writers={writers} reads={total_reads} \
         writes={writes} overlapping_writes={overlaps} overlapping_reads={read_overlaps} \
         warmup_reads={warmup} reader_warmup_reads={} warmup_ms={:.3}",
        readers * reader_warmup,
        warmup_elapsed.as_secs_f64() * 1_000.0
    );
    println!(
        "read p50={p50}us p99={p99}us p99.9={p999}us throughput={read_qps:.0}/s \
         writer_active_p99={writer_active_p99}us writer_active_p99.9={writer_active_p999}us \
         sharded_gpu_batches={gpu_batches} gpu_batches_during_writes={gpu_write_overlap} \
         host_gather_batches={} fallback_groups={} device_appends={appends} \
         device_authoritative_commits={elisions}",
        sharded_batches - gpu_batches,
        sharded_fallback_groups
    );
    println!(
        "json={{\"kind\":\"gpu_mixed_read_write_gate\",\"readers\":{readers},\"writers\":{writers},\
         \"reads\":{total_reads},\"writes\":{writes},\"overlapping_writes\":{overlaps},\
         \"overlapping_reads\":{read_overlaps},\
         \"warmup_reads\":{warmup},\"reader_warmup_reads\":{},\"warmup_ms\":{:.3},\
         \"read_p50_us\":{p50},\"read_p99_us\":{p99},\"read_p999_us\":{p999},\
         \"writer_active_p99_us\":{writer_active_p99},\"writer_active_p999_us\":{writer_active_p999},\
         \"read_qps\":{read_qps:.3},\"sharded_gpu_batches\":{gpu_batches},\
         \"gpu_batches_during_writes\":{gpu_write_overlap},\
         \"host_gather_batches\":{},\
         \"sharded_fallback_groups\":{},\
         \"device_appends\":{appends},\
         \"device_authoritative_commits\":{elisions}}}",
        readers * reader_warmup,
        warmup_elapsed.as_secs_f64() * 1_000.0,
        sharded_batches - gpu_batches,
        sharded_fallback_groups
    );
    Ok(())
}
