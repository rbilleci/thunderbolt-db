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
//!       GPU_DB_BENCH_OFFERED_TPS=N (Driver arm: evenly paced open-loop arrivals across drivers;
//!       reported latency starts at the scheduled arrival and therefore includes producer/queue slip),
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
    /// U1 mixed workload: covered DELETEs this writer completed (0 for insert-only / non-driver).
    deletes: u64,
    /// U2 mixed workload: covered UPDATEs this writer completed (0 for insert-only / non-driver).
    updates: u64,
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
    e.set_device_write_locate_wave_batch_enabled(true);
    e.set_constrained_elision_enabled(true);
    // GPU_DB_BENCH_SHARD_TARGET: pre-size the open shard (rows) to control
    // rollover frequency in-run (rollovers serialize under the apply path and,
    // in lanes mode, stall the global cut).
    if let Some(target) = std::env::var("GPU_DB_BENCH_SHARD_TARGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        e.set_shard_size_target(target);
    }
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
    // E2.3 — DEDICATED-PUMP mode (disruptor staging validation): with `GPU_DB_BENCH_PUMPS=N>0`
    // (Driver arm only), N threads do NOTHING but pump the commit wave (`drive_commit_wave`:
    // sequence + finish durability tails) while the `writers` driver threads ONLY submit + poll.
    // This decouples the single-writer sequencer from ingress/ack so the sequencer stays hot on
    // one core instead of the role bouncing across every driver (the measured per-item inflation
    // 0.89 -> 2.7us). `0` (default) keeps the self-pumping driver loop unchanged.
    // GPU_DB_BENCH_ASYNC_COMMIT=1: drive the pg-style ASYNC COMMIT mode
    // (ack at the applied cut; WAL fence pipelined behind the ack).
    let commit_mode = if std::env::var("GPU_DB_BENCH_ASYNC_COMMIT").as_deref() == Ok("1") {
        gpu_db_engine::SynchronousCommit::Off
    } else {
        gpu_db_engine::SynchronousCommit::On
    };
    let pumps: usize = std::env::var("GPU_DB_BENCH_PUMPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let offered_tps: u64 = std::env::var("GPU_DB_BENCH_OFFERED_TPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let self_pump = !(arm == Arm::Driver && pumps > 0);
    // U1 MIXED I/D WORKLOAD (Driver arm only, bench-only knob): GPU_DB_BENCH_MIX_DELETE=P
    // makes ~P% of the driver's ops covered DELETEs of an OLDER, fully-committed live key
    // (a trailing cursor lagging the insert cursor — every delete is a guaranteed 1-row hit,
    // so this measures the pure delete cost + shard versioning under load, not 0-row churn).
    // Clamped to <50 so the delete cursor can never overtake the insert cursor (starving
    // targets). 0 (default) = the insert-only champion path, byte-identical.
    let mix_delete: i64 = std::env::var("GPU_DB_BENCH_MIX_DELETE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
        .clamp(0, 49);
    // U2 MIXED I/U WORKLOAD (Driver arm only, bench-only knob): GPU_DB_BENCH_MIX_UPDATE=P makes
    // ~P% of the driver's ops covered full-row UPDATEs of an OLDER committed live key (a trailing
    // cursor, every update a guaranteed 1-row hit). This measures the update cost + the F3/U4
    // DEAD-TWIN churn: an update versions the shard and, under the in-flight window's active
    // readers, its dead twin sits above the GC boundary → the pk-index rebuild declines → the
    // locate declines → rehydrate → DE-ELIDE. So each writer re-prepares its update route on drift
    // (the elision RE-ENTRY arm); the reported throughput is dominated by that rehydrate churn
    // until the kernel index-entry-replacement fix (F3/U4) lands. 0 (default) = insert-only.
    let mix_update: i64 = std::env::var("GPU_DB_BENCH_MIX_UPDATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
        .clamp(0, 49);
    let segment = wal_dir.join(format!("intent-{name}-c{writers}.wal"));
    remove_segment_files(&segment);
    let engine = build_engine(&segment)?;
    let txn_ids = Arc::new(AtomicU64::new(2));
    let warmed = warm_up_elision(&engine, &txn_ids)?;
    let route = Arc::new(engine.prepare_covered_insert_route("t")?);
    // U1: the covered-DELETE route (prepared only when the mix is enabled — a driverless /
    // non-elided warm-up would have already returned above).
    let delete_route = if arm == Arm::Driver && mix_delete > 0 {
        Some(Arc::new(engine.prepare_covered_delete_route("t")?))
    } else {
        None
    };
    let engine = Arc::new(engine);

    let stop = Arc::new(AtomicBool::new(false));
    let pump_count = if self_pump { 0 } else { pumps };
    // +1 = the main thread; +1 more when the 1Hz stage-attribution sampler runs (TIMELINE=1).
    let timeline_enabled = std::env::var("GPU_DB_BENCH_TIMELINE").as_deref() == Ok("1");
    let barrier = Arc::new(Barrier::new(
        writers + pump_count + 1 + usize::from(timeline_enabled),
    ));
    // E2.3 — dedicated pump threads (see `pumps`): pure sequencer/tail-finisher cores.
    let pump_handles: Vec<_> = (0..pump_count)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let stop = Arc::clone(&stop);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                while !stop.load(Ordering::Relaxed) {
                    if !engine.drive_commit_wave() {
                        std::hint::spin_loop();
                    }
                }
                // Drain: keep pumping until the queue + tails are empty so no ticket is stranded.
                while engine.drive_commit_wave() {}
            })
        })
        .collect();
    let handles: Vec<_> = (0..writers)
        .map(|w| {
            let engine = Arc::clone(&engine);
            let route = Arc::clone(&route);
            let delete_route = delete_route.clone();
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
                // at any population: split the whole sub-warm-up id space evenly.
                // (A fixed 4M/writer cap overflowed into the neighbor's range on
                // long runs — 30s at 1.66M TPS is ~5M ids/driver — and the engine
                // correctly rejected the wraparound as duplicate keys.)
                let stride = 2_100_000_000 / writers as i64;
                if arm == Arm::Driver {
                    // E2.2(c): event-loop driver carrying `window` logical clients. Keep the window
                    // full of in-flight tickets, pump the wave, reap completions, record per-client
                    // end-to-end latency (submit -> durable+published poll), and refill.
                    // U1: the third tuple field is the op kind (true = DELETE) for the mix report.
                    // U1/U2: op code per in-flight slot — 0 = insert, 1 = delete, 2 = update.
                    let mut inflight: Vec<Option<(gpu_db_engine::IntentTicket, Instant, u8)>> =
                        (0..window).map(|_| None).collect();
                    // U1/U2 mixed workload: `d`/`u` (delete/update cursors) trail `i` (insert
                    // cursor) with a settle lag so their target is always a fully-committed live key.
                    let mut d = 0_i64;
                    let mut u = 0_i64;
                    let mutation_lag = if offered_tps > 0 {
                        (window as i64 * 4).max(64)
                    } else {
                        4096
                    };
                    let mut scheduled_arrivals = 0_u64;
                    let mut deletes = 0_u64;
                    let mut updates = 0_u64;
                    // U2: this writer's OWN update route (re-prepared on drift — see mix_update).
                    let mut update_route = if mix_update > 0 {
                        Some(
                            engine
                                .prepare_covered_update_route("t")
                                .map_err(|e| format!("driver {w} update route: {e}"))?,
                        )
                    } else {
                        None
                    };
                    while !stop.load(Ordering::Relaxed) {
                        for slot in inflight.iter_mut() {
                            if slot.is_none() {
                                let scheduled = if offered_tps > 0 {
                                    let global_arrival = scheduled_arrivals
                                        .saturating_mul(writers as u64)
                                        .saturating_add(w as u64);
                                    let due_ns = global_arrival
                                        .saturating_mul(1_000_000_000)
                                        .checked_div(offered_tps)
                                        .expect("offered_tps is nonzero");
                                    let due = run_started + Duration::from_nanos(due_ns);
                                    if Instant::now() < due {
                                        break;
                                    }
                                    scheduled_arrivals += 1;
                                    due
                                } else {
                                    Instant::now()
                                };
                                let txn_id = txn_ids.fetch_add(1, Ordering::Relaxed);
                                // Delete OR update an older live key iff its mix quota is under
                                // target AND a lagged committed target exists; else insert a fresh
                                // key. Delete takes priority when both quotas want a slot.
                                let delete_target = d.saturating_mul(2);
                                let update_target = u.saturating_mul(2).saturating_add(1);
                                let want_delete = mix_delete > 0
                                    && delete_target + mutation_lag < i
                                    && d * 100 < mix_delete * (i + d);
                                let want_update = !want_delete
                                    && mix_update > 0
                                    && update_target + mutation_lag < i
                                    && u * 100 < mix_update * (i + u);
                                let submitted = scheduled;
                                let (ticket, op) = if want_delete {
                                    let key = (w as i64 * stride + delete_target) as i32;
                                    d += 1;
                                    let route = delete_route
                                        .as_ref()
                                        .expect("delete route present when mix > 0");
                                    match engine.submit_covered_delete_intent_with_commit(
                                        txn_id,
                                        route,
                                        key,
                                        commit_mode,
                                    ) {
                                        Ok(t) => (t, 1u8),
                                        Err(err) => {
                                            return Err(format!("driver {w} delete submit: {err}"))
                                        }
                                    }
                                } else if want_update {
                                    let key = (w as i64 * stride + update_target) as i32;
                                    // Full-row replace (key, key+1). On DRIFT (the dead-twin
                                    // de-elision cliff) re-prepare the route (elision re-entry) and
                                    // SKIP this slot — the next iteration retries with the fresh
                                    // route. `u` advances only on a submitted update, so no target
                                    // is skipped. The borrow is scoped so the reassign is legal.
                                    let result = {
                                        let route = update_route
                                            .as_ref()
                                            .expect("update route present when mix_update > 0");
                                        engine.submit_covered_update_intent_with_commit(
                                            txn_id,
                                            route,
                                            &[key, key.wrapping_add(1)],
                                            commit_mode,
                                        )
                                    };
                                    match result {
                                        Ok(t) => {
                                            u += 1;
                                            (t, 2u8)
                                        }
                                        Err(_) => {
                                            if let Ok(fresh) =
                                                engine.prepare_covered_update_route("t")
                                            {
                                                update_route = Some(fresh);
                                            }
                                            continue;
                                        }
                                    }
                                } else {
                                    let id = (w as i64 * stride + i) as i32;
                                    i += 1;
                                    match engine.submit_covered_insert_intent_with_commit(
                                        txn_id,
                                        &route,
                                        &[id, 1],
                                        commit_mode,
                                    ) {
                                        Ok(t) => (t, 0u8),
                                        Err(err) => {
                                            return Err(format!("driver {w} submit: {err}"))
                                        }
                                    }
                                };
                                *slot = Some((ticket, submitted, op));
                            }
                        }
                        // With dedicated pumps the drivers ONLY submit + poll (the pump threads
                        // own sequencing + tail-finishing); otherwise self-pump as before.
                        if self_pump {
                            engine.drive_commit_wave();
                        }
                        for slot in inflight.iter_mut() {
                            if let Some((ticket, submitted, op)) = slot {
                                if let Some(result) = engine.poll_intent(ticket) {
                                    let rows =
                                        result.map_err(|err| format!("driver {w}: {err}"))?;
                                    match *op {
                                        1 => {
                                            deletes += 1;
                                            debug_assert_eq!(
                                                rows, 1,
                                                "trailing-cursor delete misses"
                                            );
                                        }
                                        2 => {
                                            updates += 1;
                                            debug_assert_eq!(
                                                rows, 1,
                                                "trailing-cursor update misses"
                                            );
                                        }
                                        _ => {}
                                    }
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
                            if let Some((ticket, submitted, op)) = slot {
                                if let Some(result) = engine.poll_intent(ticket) {
                                    let rows =
                                        result.map_err(|err| format!("driver {w} drain: {err}"))?;
                                    match *op {
                                        1 => deletes += 1,
                                        2 => updates += 1,
                                        _ => {}
                                    }
                                    let _ = rows;
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
                        deletes,
                        updates,
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
                    deletes: 0,
                    updates: 0,
                })
            })
        })
        .collect();

    // Per-second STAGE ATTRIBUTION sampler (2M+ push (a)): the timeline shows WHERE the
    // sustained/burst gap lives; this shows WHY — cumulative stage counters sampled at 1Hz,
    // printed as per-second deltas so a TPS dip lines up with the stage that inflated
    // (validate/publish/apply/host passes/fence latency) in the same second.
    let sampler = timeline_enabled.then(|| {
        let engine = Arc::clone(&engine);
        let stop = Arc::clone(&stop);
        let barrier = Arc::clone(&barrier);
        std::thread::spawn(move || {
            barrier.wait();
            let mut rows: Vec<StageSnap> = vec![stage_snap(&engine)];
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1000));
                rows.push(stage_snap(&engine));
            }
            rows
        })
    });
    barrier.wait();
    let started = Instant::now();
    std::thread::sleep(Duration::from_secs(seconds));
    stop.store(true, Ordering::Relaxed);
    let samples: Vec<WriterSample> = handles
        .into_iter()
        .map(|h| h.join().expect("writer panicked"))
        .collect::<Result<_, _>>()?;
    for pump in pump_handles {
        pump.join().expect("pump panicked");
    }
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
    // Per-second completion timeline (GPU_DB_BENCH_TIMELINE=1): the
    // sustained/burst gap lives in STALL seconds — this shows where.
    if timeline_enabled {
        let secs: Vec<u64> = per_ms
            .chunks(1000)
            .map(|chunk| chunk.iter().sum::<u64>())
            .collect();
        eprintln!(
            "    [timeline tps/s: {}]",
            secs.iter()
                .map(|s| format!("{:.2}M", *s as f64 / 1e6))
                .collect::<Vec<_>>()
                .join(" ")
        );
        // Per-second STAGE deltas (see the sampler above): one row per second, per-wave
        // microseconds for each pipeline stage + fence latency — the dip in tps/s above lines
        // up with the stage column that inflated in the same row.
        if let Some(sampler) = sampler {
            let rows = sampler.join().expect("stage sampler panicked");
            eprintln!(
                "    [stages/s:  sec |  waves |  items | us/wave: validate publish apply | \
                 drain conflict patch settle | fence us/frame | acklag us/wave]"
            );
            for (sec, pair) in rows.windows(2).enumerate() {
                let d = pair[1].delta(&pair[0]);
                let per_wave = |ns: u64| ns as f64 / 1000.0 / d.waves.max(1) as f64;
                eprintln!(
                    "    [stages/s: {:>4} | {:>6} | {:>6} | {:>8.0} {:>7.0} {:>5.0} | {:>5.0} \
                     {:>8.0} {:>5.0} {:>6.0} | {:>14.0} | {:>14.0}]",
                    sec,
                    d.waves,
                    d.items,
                    per_wave(d.validate_ns),
                    per_wave(d.publish_ns),
                    per_wave(d.apply_ns),
                    per_wave(d.drain_ns),
                    per_wave(d.conflict_ns),
                    per_wave(d.patch_ns),
                    per_wave(d.settle_ns),
                    d.fence_ns as f64 / 1000.0 / d.fence_frames.max(1) as f64,
                    d.acklag_ns as f64 / 1000.0 / d.settled_waves.max(1) as f64,
                );
            }
        }
    } else if let Some(sampler) = sampler {
        let _ = sampler.join();
    }

    let stats = engine.wal_group_commit_stats();
    println!(
        "  {:>7} | {:>7} | {:>13.0} | {:>9} | {:>5.2}ms | {:>5.2}ms | {:>5.2}ms | {:>5.2}ms | {:>6.1}ms | {:>7} | {:>8.1}",
        name,
        writers,
        total as f64 / elapsed.as_secs_f64(),
        burst,
        percentile(&latencies, 0.50) as f64 / 1e6,
        percentile(&latencies, 0.90) as f64 / 1e6,
        percentile(&latencies, 0.99) as f64 / 1e6,
        percentile(&latencies, 0.999) as f64 / 1e6,
        latencies.last().copied().unwrap_or(0) as f64 / 1e6,
        stats.flush_groups,
        stats.mean_group_size(),
    );

    // U1 MIXED I/D report: the delete share + the device delete-path counters (non-vacuity —
    // the visible-locate + in-place tombstone counts prove the device arms fired; pk-rebuilds
    // shows the visibility-aware rebuild stays bounded under delete-versioning churn).
    let total_deletes: u64 = samples.iter().map(|s| s.deletes).sum();
    let total_updates: u64 = samples.iter().map(|s| s.updates).sum();
    if total_deletes > 0 || total_updates > 0 {
        eprintln!(
            "    [mix I/U/D: ops {total}  deletes {total_deletes} ({:.1}%)  \
             updates {total_updates} ({:.1}%)  visible-locates {}  tombstone-applies {}  \
             pk-rebuilds {}]",
            total_deletes as f64 * 100.0 / total.max(1) as f64,
            total_updates as f64 * 100.0 / total.max(1) as f64,
            engine.device_visible_locate_hits(),
            engine.lane_tombstone_applies(),
            engine.pk_index_rebuilds_diag(),
        );
    }

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
        if let Some((lw, li, val, claim, enc, publ, apply, durable, applied, rb)) =
            engine.intent_lane_stats()
        {
            if lw > 0 {
                eprintln!(
                    "    [lanes: waves {lw}  items/wave {:.1}  us/wave: validate {:.1} claim {:.1} encode {:.1} publish {:.1} device-apply {:.1}  cuts: durable {durable} applied {applied}  pk-rebuilds {rb}]",
                    li as f64 / lw as f64,
                    val as f64 / lw as f64 / 1e3,
                    claim as f64 / lw as f64 / 1e3,
                    enc as f64 / lw as f64 / 1e3,
                    publ as f64 / lw as f64 / 1e3,
                    apply as f64 / lw as f64 / 1e3,
                );
                if let Some((active, outstanding, resizes, resize_ns)) =
                    engine.intent_lane_adaptive_stats()
                {
                    eprintln!(
                        "    [adaptive: active_lanes {active}  outstanding {outstanding}  resizes {resizes} ({:.1}ms total barrier)]",
                        resize_ns as f64 / 1e6,
                    );
                }
                if let Some((lag_ns, lag_waves)) = engine.intent_lane_acklag_stats() {
                    if lag_waves > 0 {
                        eprintln!(
                            "    [publish->settle: {:.1} us/wave over {lag_waves} settled waves]",
                            lag_ns as f64 / lag_waves as f64 / 1e3,
                        );
                    }
                }
                if let Some((fence_ns, frames)) = engine.intent_lane_fence_stats() {
                    if frames > 0 {
                        eprintln!(
                            "    [fence: {:.1} us/frame over {frames} frames]",
                            fence_ns as f64 / frames as f64 / 1e3,
                        );
                    }
                }
                if let Some((drain, conflict, patch, settle)) = engine.intent_lane_hostpass_stats()
                {
                    eprintln!(
                        "    [pump host us/wave: drain {:.1} conflict {:.1} patch {:.1} settle {:.1}]",
                        drain as f64 / lw as f64 / 1e3,
                        conflict as f64 / lw as f64 / 1e3,
                        patch as f64 / lw as f64 / 1e3,
                        settle as f64 / lw as f64 / 1e3,
                    );
                }
                if let Some((vbusy, vlaunch, abusy, alaunch)) = engine.intent_lane_leader_stats() {
                    eprintln!(
                        "    [leaders: validate busy {:.2}s over {vlaunch} launches  apply busy {:.2}s over {alaunch} launches]",
                        vbusy as f64 / 1e9,
                        abusy as f64 / 1e9,
                    );
                }
            }
        }
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

    #[cfg(feature = "probe-timing")]
    {
        let (payload_and_regions, indexes) =
            engine.probe_relational_resident_table_byte_components("t", 0);
        let frames = engine
            .intent_lane_fence_stats()
            .map_or(0, |(_, frames)| frames);
        let physical_wal_bytes = frames.saturating_mul(4096);
        let appended_versions = total
            .saturating_add(warmed)
            .saturating_sub(total_deletes as usize);
        eprintln!(
            "    [footprint: payload+regions {payload_and_regions}B  device-index {indexes}B  \
             appended-versions {appended_versions}  FUA-frames {frames}  physical-WAL {physical_wal_bytes}B ({:.1}B/op)]",
            physical_wal_bytes as f64 / total.max(1) as f64,
        );
    }

    // Final visible rows = warm-up + INSERTs - DELETEs. `total` also includes UPDATE and DELETE
    // operations, so remove updates once and deletes twice (once to exclude the non-insert op,
    // once for the row it removed). The former insert-only formula falsely reported recovery
    // mismatch for mixed workloads even when replay was exact.
    let committed = total
        .saturating_add(warmed)
        .saturating_sub(total_updates as usize)
        .saturating_sub((total_deletes as usize).saturating_mul(2));
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
        "  wal-durability={}  duration/point={seconds}s  offered-tps={}  table=t(id INT PRIMARY KEY, v INT)",
        std::env::var("GPU_DB_WAL_DURABILITY").unwrap_or_default(),
        std::env::var("GPU_DB_BENCH_OFFERED_TPS").unwrap_or_else(|_| "closed-loop".to_string()),
    );
    println!();
    println!(
        "      arm | writers | sustained TPS | burst TPS |    p50 |    p90 |    p99 |  p99.9 |     max | fsyncs | mean grp"
    );
    println!(
        "  ------- | ------- | ------------- | --------- | ------ | ------ | ------ | ------ | ------- | ------ | --------"
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

/// Cumulative stage counters at one sampling instant (per-second STAGE ATTRIBUTION — see the
/// TIMELINE sampler). Deltas between consecutive snapshots give the per-second view.
#[derive(Clone, Copy, Default)]
struct StageSnap {
    waves: u64,
    items: u64,
    validate_ns: u64,
    publish_ns: u64,
    apply_ns: u64,
    drain_ns: u64,
    conflict_ns: u64,
    patch_ns: u64,
    settle_ns: u64,
    fence_ns: u64,
    fence_frames: u64,
    acklag_ns: u64,
    settled_waves: u64,
}

impl StageSnap {
    /// Per-second delta (saturating: the fence counters live in the ACTIVE segment and reset
    /// on a roll — a negative delta clamps to 0 for that second instead of wrapping).
    fn delta(&self, prev: &Self) -> Self {
        Self {
            waves: self.waves.saturating_sub(prev.waves),
            items: self.items.saturating_sub(prev.items),
            validate_ns: self.validate_ns.saturating_sub(prev.validate_ns),
            publish_ns: self.publish_ns.saturating_sub(prev.publish_ns),
            apply_ns: self.apply_ns.saturating_sub(prev.apply_ns),
            drain_ns: self.drain_ns.saturating_sub(prev.drain_ns),
            conflict_ns: self.conflict_ns.saturating_sub(prev.conflict_ns),
            patch_ns: self.patch_ns.saturating_sub(prev.patch_ns),
            settle_ns: self.settle_ns.saturating_sub(prev.settle_ns),
            fence_ns: self.fence_ns.saturating_sub(prev.fence_ns),
            fence_frames: self.fence_frames.saturating_sub(prev.fence_frames),
            acklag_ns: self.acklag_ns.saturating_sub(prev.acklag_ns),
            settled_waves: self.settled_waves.saturating_sub(prev.settled_waves),
        }
    }
}

fn stage_snap(engine: &gpu_db_engine::Engine) -> StageSnap {
    let mut snap = StageSnap::default();
    if let Some(stats) = engine.intent_lane_stats() {
        snap.waves = stats.0;
        snap.items = stats.1;
        snap.validate_ns = stats.2;
        snap.publish_ns = stats.5;
        snap.apply_ns = stats.6;
    }
    if let Some((drain, conflict, patch, settle)) = engine.intent_lane_hostpass_stats() {
        snap.drain_ns = drain;
        snap.conflict_ns = conflict;
        snap.patch_ns = patch;
        snap.settle_ns = settle;
    }
    if let Some((ns, frames)) = engine.intent_lane_fence_stats() {
        snap.fence_ns = ns;
        snap.fence_frames = frames;
    }
    if let Some((ns, settled)) = engine.intent_lane_acklag_stats() {
        snap.acklag_ns = ns;
        snap.settled_waves = settled;
    }
    snap
}
