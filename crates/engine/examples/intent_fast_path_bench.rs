//! E2.1 — covered-INSERT INTENT fast-path benchmark (durable, FUA WAL lane).
//!
//! Measures the GPU-native intent fast path (`prepare_covered_insert_route` +
//! `execute_covered_insert_intent`: no SQL parse, no `prepare_insert`; wave-batched
//! device PK validation + wave-batched device open-shard apply + W5a binary WAL
//! records + FUA fence-pool durability) against the classic text path
//! (`execute_dml_concurrent`) on the SAME table shape, flags, and durability
//! backend. Closed-loop: N writer threads, single-row covered INSERTs into
//! `t (id INT PRIMARY KEY, v INT)`, per-commit end-to-end latency (issue ->
//! durable + published ack).
//!
//! The durable arm is the primary number (durability-first mandate); every
//! latency line pairs with its throughput.
//!
//! Run:  GPU_DB_WAL_DURABILITY=fua cargo run --release -p gpu_db_engine --example intent_fast_path_bench
//! Env:  GPU_DB_BENCH_WRITERS (default "8,16,32,64,128"), GPU_DB_BENCH_SECONDS (default 3),
//!       GPU_DB_BENCH_ARM (classic|intent|both, default both),
//!       GPU_DB_BENCH_WAL_DIR (default target/wal-intent-bench),
//!       GPU_DB_BENCH_RECOVER=1 (post-run crash-recovery replay + row-count parity check),
//!       GPU_DB_BENCH_HOSTPHASE=1 / GPU_DB_BENCH_DEVPHASE=1 (per-stage attribution),
//!       GPU_DB_WAL_DURABILITY (defaulted to `fua` by this bench; set `serial` to A/B).

use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use gpu_db_engine::Engine;

// jemalloc, same rationale as oltp_commit_slo_benchmark: don't measure glibc arenas.
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

struct WriterSample {
    latencies_nanos: Vec<u64>,
    completions_millis: Vec<u64>,
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() as f64) * p).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Remove a durable segment and every `<segment>.fua.<id>` sibling.
fn remove_segment_files(segment: &std::path::Path) {
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(segment));
    let _ = std::fs::remove_file(segment);
    if let (Some(parent), Some(stem)) = (segment.parent(), segment.file_name()) {
        let prefix = format!("{}.fua.", stem.to_string_lossy());
        if let Ok(entries) = std::fs::read_dir(parent) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with(&prefix) {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }
}

/// Build the flagship engine: durable WAL at `segment`, covered-INSERT flags ON.
fn build_engine(segment: &std::path::Path) -> Result<Engine, Box<dyn Error>> {
    let e = Engine::with_durable_wal_segment(segment);
    e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")?;
    e.set_auto_admit_on_commit(true);
    e.set_host_install_elision_enabled(true);
    e.set_binary_wal_records_enabled(true);
    e.set_device_write_locate_enabled(true);
    e.set_device_write_locate_wave_batch_enabled(true);
    e.set_constrained_elision_enabled(true);
    Ok(e)
}

/// Warm the table into the elided (device-authoritative) state: classic inserts
/// until `table_install_elided` flips (admission + elide-entry are lazy).
/// Warm-up ids live at the top of the int4 range, disjoint from bench ids.
fn warm_up_elision(engine: &Engine, txn_ids: &AtomicU64) -> Result<usize, Box<dyn Error>> {
    const WARMUP_BASE: i64 = 2_100_000_000;
    for i in 0..20_000_i64 {
        let txn_id = txn_ids.fetch_add(1, Ordering::Relaxed);
        engine.execute_dml_concurrent(
            txn_id,
            &format!("INSERT INTO t VALUES ({}, 1)", WARMUP_BASE + i),
        )?;
        if engine.table_install_elided("t") {
            return Ok(i as usize + 1);
        }
    }
    Err("table never entered elision during warm-up (is a GPU available?)".into())
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Classic,
    Intent,
    /// E2.2(c) — the driver-multiplexed arm: each thread is an EVENT-LOOP DRIVER carrying a WINDOW
    /// of logical clients (in-flight tickets), submitting non-blocking, pumping the single-writer
    /// commit wave via `drive_commit_wave`, and reaping tickets with `poll_intent`. No per-commit
    /// thread park/wake — the thread-per-client wake storm the E2.1 bench measured is gone.
    Driver,
}

fn run_arm(
    arm: Arm,
    writers: usize,
    seconds: u64,
    wal_dir: &std::path::Path,
    recover: bool,
) -> Result<(), Box<dyn Error>> {
    let name = match arm {
        Arm::Classic => "classic",
        Arm::Intent => "intent",
        Arm::Driver => "driver",
    };
    // E2.2(c): per-driver in-flight window = logical clients carried per driver thread.
    let window: usize = std::env::var("GPU_DB_BENCH_WINDOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&w| w > 0)
        .unwrap_or(64);
    let segment = wal_dir.join(format!("intent-{name}-c{writers}.wal"));
    remove_segment_files(&segment);
    let engine = build_engine(&segment)?;
    let txn_ids = Arc::new(AtomicU64::new(2));
    let warmed = warm_up_elision(&engine, &txn_ids)?;
    let route = Arc::new(engine.prepare_covered_insert_route("t")?);
    let engine = Arc::new(engine);

    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(writers + 1));
    let handles: Vec<_> = (0..writers)
        .map(|w| {
            let engine = Arc::clone(&engine);
            let route = Arc::clone(&route);
            let stop = Arc::clone(&stop);
            let txn_ids = Arc::clone(&txn_ids);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || -> Result<WriterSample, String> {
                // Bounded pre-allocation: high-population sweeps would OOM on 2 x 8MB per writer.
                let mut latencies_nanos = Vec::with_capacity(4096);
                let mut completions_millis = Vec::with_capacity(4096);
                barrier.wait();
                let run_started = Instant::now();
                let mut i = 0_i64;
                // Disjoint per-writer ranges, all below the warm-up base (2.1e9)
                // at any population; 4M ids/writer up to 512 writers, shrinking
                // proportionally past that (a 3s closed-loop point stays far
                // under the per-writer budget either way).
                let stride = 4_000_000_i64.min(2_100_000_000 / writers as i64);
                if arm == Arm::Driver {
                    // E2.2(c): event-loop driver carrying `window` logical clients. Keep the window
                    // full of in-flight tickets, pump the wave, reap completions, record per-client
                    // end-to-end latency (submit -> durable+published poll), and refill.
                    let mut inflight: Vec<Option<(gpu_db_engine::IntentTicket, Instant)>> =
                        (0..window).map(|_| None).collect();
                    while !stop.load(Ordering::Relaxed) {
                        for slot in inflight.iter_mut() {
                            if slot.is_none() {
                                let txn_id = txn_ids.fetch_add(1, Ordering::Relaxed);
                                let id = (w as i64 * stride + i) as i32;
                                i += 1;
                                let submitted = Instant::now();
                                match engine.submit_covered_insert_intent(txn_id, &route, &[id, 1])
                                {
                                    Ok(ticket) => *slot = Some((ticket, submitted)),
                                    Err(err) => return Err(format!("driver {w} submit: {err}")),
                                }
                            }
                        }
                        engine.drive_commit_wave();
                        for slot in inflight.iter_mut() {
                            if let Some((ticket, submitted)) = slot {
                                if let Some(result) = engine.poll_intent(ticket) {
                                    result.map_err(|err| format!("driver {w}: {err}"))?;
                                    latencies_nanos.push(submitted.elapsed().as_nanos() as u64);
                                    completions_millis
                                        .push(run_started.elapsed().as_millis() as u64);
                                    *slot = None;
                                }
                            }
                        }
                    }
                    // Drain the window so no ticket outlives the run (releases GC boundaries).
                    // Record drained commits too so `total` matches the durable row count
                    // (recovery-parity honesty).
                    let mut pending = inflight.iter().filter(|s| s.is_some()).count();
                    while pending > 0 {
                        engine.drive_commit_wave();
                        for slot in inflight.iter_mut() {
                            if let Some((ticket, submitted)) = slot {
                                if let Some(result) = engine.poll_intent(ticket) {
                                    result.map_err(|err| format!("driver {w} drain: {err}"))?;
                                    latencies_nanos.push(submitted.elapsed().as_nanos() as u64);
                                    completions_millis
                                        .push(run_started.elapsed().as_millis() as u64);
                                    *slot = None;
                                    pending -= 1;
                                }
                            }
                        }
                    }
                    return Ok(WriterSample {
                        latencies_nanos,
                        completions_millis,
                    });
                }
                while !stop.load(Ordering::Relaxed) {
                    let txn_id = txn_ids.fetch_add(1, Ordering::Relaxed);
                    let id = (w as i64 * stride + i) as i32;
                    let commit_started = Instant::now();
                    let outcome = match arm {
                        Arm::Intent => {
                            engine.execute_covered_insert_intent(txn_id, &route, &[id, 1])
                        }
                        Arm::Classic => engine.execute_dml_concurrent(
                            txn_id,
                            &format!("INSERT INTO t VALUES ({id}, 1)"),
                        ),
                        Arm::Driver => unreachable!("driver arm handled above"),
                    };
                    outcome.map_err(|err| format!("writer {w} id {id}: {err}"))?;
                    latencies_nanos.push(commit_started.elapsed().as_nanos() as u64);
                    completions_millis.push(run_started.elapsed().as_millis() as u64);
                    i += 1;
                }
                Ok(WriterSample {
                    latencies_nanos,
                    completions_millis,
                })
            })
        })
        .collect();

    barrier.wait();
    let started = Instant::now();
    std::thread::sleep(Duration::from_secs(seconds));
    stop.store(true, Ordering::Relaxed);
    let samples: Vec<WriterSample> = handles
        .into_iter()
        .map(|h| h.join().expect("writer panicked"))
        .collect::<Result<_, _>>()?;
    let elapsed = started.elapsed();

    let mut latencies: Vec<u64> = samples
        .iter()
        .flat_map(|s| s.latencies_nanos.iter().copied())
        .collect();
    latencies.sort_unstable();
    let total = latencies.len();

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

    let stats = engine.wal_group_commit_stats();
    println!(
        "  {:>7} | {:>7} | {:>13.0} | {:>9} | {:>5.2}ms | {:>5.2}ms | {:>5.2}ms | {:>6.1}ms | {:>7} | {:>8.1}",
        name,
        writers,
        total as f64 / elapsed.as_secs_f64(),
        burst,
        percentile(&latencies, 0.50) as f64 / 1e6,
        percentile(&latencies, 0.90) as f64 / 1e6,
        percentile(&latencies, 0.99) as f64 / 1e6,
        latencies.last().copied().unwrap_or(0) as f64 / 1e6,
        stats.flush_groups,
        stats.mean_group_size(),
    );

    // Attribution recon (stderr, opt-in phase timing).
    {
        let w = &gpu_db_engine::engine_dml_concurrent_wave_stats();
        let (waves, items, nanos) = (
            w[0].swap(0, Ordering::Relaxed),
            w[1].swap(0, Ordering::Relaxed),
            w[2].swap(0, Ordering::Relaxed),
        );
        eprintln!(
            "    [{name} c{writers}: warmed {warmed}  waves {waves}  items/wave {:.1}  sequencer us/item {:.2}]",
            items as f64 / waves.max(1) as f64,
            nanos as f64 / items.max(1) as f64 / 1e3,
        );
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
    }

    let committed = total + warmed;
    drop(samples);
    drop(engine);

    if recover {
        // Crash-recovery parity: reopen from the durable segment (FUA frame logs
        // are detected and replayed) and count the surviving rows.
        let recover_started = Instant::now();
        let recovered = Engine::open_durable_wal_segment(&segment)?;
        let result = recovered.execute_relational_select_text("SELECT COUNT(*) FROM t")?;
        let count = format!("{:?}", result.rows.row(0).first());
        let ok = count.contains(&format!("({committed})"));
        eprintln!(
            "    [recovery: replayed {} commits in {:.2}s  row-count {}  {}]",
            committed,
            recover_started.elapsed().as_secs_f64(),
            count,
            if ok { "PARITY-OK" } else { "MISMATCH" },
        );
        if !ok {
            return Err(format!(
                "recovery parity failed for {name} c{writers}: expected {committed}, got {count}"
            )
            .into());
        }
        drop(recovered);
    }
    remove_segment_files(&segment);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Default to the FUA fence-pool WAL backend (the E1 durable lane); callers
    // may override (e.g. GPU_DB_WAL_DURABILITY=serial for the fdatasync A/B).
    if std::env::var("GPU_DB_WAL_DURABILITY").is_err() {
        std::env::set_var("GPU_DB_WAL_DURABILITY", "fua");
    }
    let writers_sweep: Vec<usize> = std::env::var("GPU_DB_BENCH_WRITERS")
        .unwrap_or_else(|_| "8,16,32,64,128".to_string())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let seconds: u64 = std::env::var("GPU_DB_BENCH_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let arm = std::env::var("GPU_DB_BENCH_ARM").unwrap_or_else(|_| "both".to_string());
    let recover = std::env::var("GPU_DB_BENCH_RECOVER").is_ok_and(|v| v == "1");
    let wal_dir = std::env::var("GPU_DB_BENCH_WAL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("target/wal-intent-bench"));
    std::fs::create_dir_all(&wal_dir)?;

    println!("E2.1 covered-INSERT intent fast-path benchmark (durable, closed-loop)");
    println!(
        "  wal-durability={}  duration/point={seconds}s  table=t(id INT PRIMARY KEY, v INT)",
        std::env::var("GPU_DB_WAL_DURABILITY").unwrap_or_default()
    );
    println!();
    println!(
        "      arm | writers | sustained TPS | burst TPS |    p50 |    p90 |    p99 |     max | fsyncs | mean grp"
    );
    println!(
        "  ------- | ------- | ------------- | --------- | ------ | ------ | ------ | ------- | ------ | --------"
    );

    for &writers in &writers_sweep {
        if arm == "classic" || arm == "both" {
            run_arm(Arm::Classic, writers, seconds, &wal_dir, recover)?;
        }
        if arm == "intent" || arm == "both" {
            run_arm(Arm::Intent, writers, seconds, &wal_dir, recover)?;
        }
        // E2.2(c): the driver-multiplexed arm (opt-in via ARM=driver|all; `writers` = driver
        // threads, each carrying GPU_DB_BENCH_WINDOW logical clients).
        if arm == "driver" || arm == "all" {
            run_arm(Arm::Driver, writers, seconds, &wal_dir, recover)?;
        }
    }
    Ok(())
}
