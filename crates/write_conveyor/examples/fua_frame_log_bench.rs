//! General FUA frame-log microbenchmark.
//!
//! The default preserves the original open-loop saturation probe. Set
//! `CONVEYOR_FUA_MODE=closed` to measure closed-loop logical groups: each group is split into
//! `N` physical frames, waits for its terminal contiguous durable cut, then starts the next
//! group. That mode sweeps fragment counts `1,2,4,8,16,32` over 29/30/31KiB opaque payloads
//! while holding the fence pool at the production-shaped 32 lanes by default. The output reports
//! the configured pool width separately from the achieved in-flight depth.
//!
//! ```bash
//! CONVEYOR_FUA_QD=16 CONVEYOR_EVENTS=200000 \
//!   cargo run --release -p gpu_db_write_conveyor --example fua_frame_log_bench
//!
//! CONVEYOR_FUA_MODE=closed CONVEYOR_FUA_GROUPS=512 \
//!   cargo run --release -p gpu_db_write_conveyor --example fua_frame_log_bench
//!
//! CONVEYOR_FUA_MODE=controller_trace CONVEYOR_FUA_CONTROLLER_TRACE_GROUPS=8000 \
//!   cargo run --release -p gpu_db_write_conveyor --example fua_frame_log_bench
//! ```

use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use gpu_db_write_conveyor::{
    fua_frame_padded_bytes, recover_frame_log_by_scan, FuaControllerDecision,
    FuaControllerEligibility, FuaFrameLog, FuaFrameLogAppender, FuaFrameLogConfig,
    FuaFrameLogTelemetry, FuaPhysicalController, FUA_CONTROLLER_FAST_NANOS,
    FUA_CONTROLLER_QD16_FRAGMENTS, FUA_CONTROLLER_SUSTAINED_GROUPS,
};

const DEFAULT_OPEN_PAYLOADS: [usize; 8] = [448, 448, 960, 448, 1984, 448, 448, 4032];
const DEFAULT_CLOSED_LOGICAL_PAYLOADS: [usize; 3] = [29 * 1024, 30 * 1024, 31 * 1024];
const DEFAULT_CLOSED_QDS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const CONTROLLER_LOGICAL_PAYLOAD_BYTES: usize = 30 * 1024;
const CONTROLLER_POOL_LANES: usize = 32;
const CONTROLLER_SLOW_NANOS: u64 = 1_100_000;
const CONTROLLER_GATE_MEAN_NANOS: u64 = 680_000;
const CONTROLLER_GATE_P99_NANOS: u64 = 1_100_000;
const CONTROLLER_GATE_MAX_OVER_P99_NANOS: u64 = 64;
const CONTROLLER_GATE_GROUPS: u64 = 8_000;
const CONTROLLER_GAPS_MICROS: [u64; 6] = [0, 500, 1_000, 2_000, 5_000, 10_000];

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_list(name: &str, defaults: &[usize]) -> Result<Vec<usize>, Box<dyn Error>> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(defaults.to_vec());
    };
    let values: Result<Vec<_>, _> = value
        .split(',')
        .map(str::trim)
        .map(str::parse::<usize>)
        .collect();
    let values = values?;
    if values.is_empty() || values.contains(&0) {
        return Err(format!(
            "{name} must be a non-empty comma-separated list of positive integers"
        )
        .into());
    }
    Ok(values)
}

fn bench_path(label: &str) -> PathBuf {
    PathBuf::from(format!(
        "target/fua-frame-log-{label}-{}.dat",
        std::process::id()
    ))
}

fn nanos(duration: Duration) -> u64 {
    duration.as_nanos().try_into().unwrap_or(u64::MAX)
}

fn percentile_nanos(samples: &mut [u64], numerator: usize, denominator: usize) -> u64 {
    assert!(!samples.is_empty());
    samples.sort_unstable();
    let index = samples
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[index]
}

fn percentile_or_zero(samples: &mut [u64], numerator: usize, denominator: usize) -> u64 {
    if samples.is_empty() {
        0
    } else {
        percentile_nanos(samples, numerator, denominator)
    }
}

fn rate_per_second_milli(units: u64, elapsed: Duration) -> u64 {
    let elapsed_ns = nanos(elapsed).max(1) as u128;
    (u128::from(units)
        .saturating_mul(1_000)
        .saturating_mul(1_000_000_000)
        / elapsed_ns)
        .try_into()
        .unwrap_or(u64::MAX)
}

fn padded_group_bytes(logical_payload_bytes: usize, fragments: usize) -> usize {
    let base = logical_payload_bytes / fragments;
    let remainder = logical_payload_bytes % fragments;
    (0..fragments)
        .map(|index| base + usize::from(index < remainder))
        .map(fua_frame_padded_bytes)
        .sum()
}

fn print_telemetry(telemetry: FuaFrameLogTelemetry) {
    print!(
        " telemetry_published_frames={} telemetry_fenced_frames={} telemetry_fence_failures={} \
         telemetry_payload_bytes={} telemetry_padded_bytes={} telemetry_stage_copy_nanos={} \
         telemetry_stage_copy_frames={} telemetry_publish_to_claim_nanos={} \
         telemetry_publish_to_claim_frames={} telemetry_claim_to_write_done_nanos={} \
         telemetry_claim_to_write_done_frames={} telemetry_write_done_to_cut_nanos={} \
         telemetry_write_done_to_cut_frames={} telemetry_cut_events={} \
         telemetry_cut_advanced_frames={} telemetry_cut_advance_max_frames={} \
         telemetry_in_flight_depth_max={} telemetry_depth_1={} telemetry_depth_2={} \
         telemetry_depth_3_to_4={} telemetry_depth_5_to_8={} telemetry_depth_9_to_16={} \
         telemetry_depth_17_to_32={} telemetry_depth_33_plus={}",
        telemetry.published_frames,
        telemetry.fenced_frames,
        telemetry.fence_failures,
        telemetry.payload_bytes,
        telemetry.padded_bytes,
        telemetry.stage_copy_nanos,
        telemetry.stage_copy_frames,
        telemetry.publish_to_claim_nanos,
        telemetry.publish_to_claim_frames,
        telemetry.claim_to_write_done_nanos,
        telemetry.claim_to_write_done_frames,
        telemetry.write_done_to_contiguous_cut_nanos,
        telemetry.write_done_to_contiguous_cut_frames,
        telemetry.contiguous_cut_events,
        telemetry.contiguous_cut_advanced_frames,
        telemetry.contiguous_cut_advance_max_frames,
        telemetry.in_flight_depth_max,
        telemetry.in_flight_depth_histogram[0],
        telemetry.in_flight_depth_histogram[1],
        telemetry.in_flight_depth_histogram[2],
        telemetry.in_flight_depth_histogram[3],
        telemetry.in_flight_depth_histogram[4],
        telemetry.in_flight_depth_histogram[5],
        telemetry.in_flight_depth_histogram[6],
    );
}

fn run_open_loop() -> Result<(), Box<dyn Error>> {
    let fence_qd = parse_env("CONVEYOR_FUA_QD", 16_usize).max(1);
    let frames = parse_env("CONVEYOR_EVENTS", 200_000_u64);
    let max_padded = 4096 + 512;
    let path = bench_path("open");
    let _ = std::fs::remove_file(&path);
    let setup_start = Instant::now();
    let log = unsafe {
        FuaFrameLog::create(FuaFrameLogConfig {
            path: path.clone(),
            segment_id: 1,
            capacity_bytes: frames as usize * max_padded,
        })?
    };
    let pool = log.spawn_fence_pool(fence_qd);
    let mut appender = log.appender();
    let setup = setup_start.elapsed();

    let start = Instant::now();
    let mut seq = 0_u64;
    for index in 0..frames {
        while log.free_fence_slots(fence_qd) == 0 {
            if log.fence_failed() {
                return Err("fence pool failed".into());
            }
            std::thread::yield_now();
        }
        let size = DEFAULT_OPEN_PAYLOADS[index as usize % DEFAULT_OPEN_PAYLOADS.len()];
        let payload = vec![(index as u8).wrapping_mul(17); size];
        appender.publish_frame(&payload, seq, 1)?;
        seq += 1;
    }
    appender.finish();
    let fences = pool.join()?;
    let elapsed = start.elapsed();
    if log.durable_seq() != seq {
        return Err("durable sequence did not reach open-loop payload".into());
    }
    let recovered = recover_frame_log_by_scan(&path)?;
    if recovered.len() != frames as usize {
        return Err("open-loop recovery did not return every frame".into());
    }
    let telemetry = log.telemetry();
    let frames_per_sec = frames as f64 / elapsed.as_secs_f64();
    let records_per_sec = seq as f64 / elapsed.as_secs_f64();
    println!(
        "fua-frame-log  {:.3} K fences/s  {:.3} M records-equiv/s  elapsed={:.3}s setup={:.3}s fence-qd={fence_qd} frames={frames} fences={fences} recover=exact",
        frames_per_sec / 1_000.0,
        records_per_sec / 1_000_000.0,
        elapsed.as_secs_f64(),
        setup.as_secs_f64(),
    );
    print!(
        "fua_frame_log_bench_status=complete mode=open_loop physical_qd={fence_qd} frames={frames} fences={fences} \
         elapsed_nanos={} frames_per_second_milli={} recovery=exact",
        nanos(elapsed),
        rate_per_second_milli(frames, elapsed),
    );
    print_telemetry(telemetry);
    println!();
    drop(log);
    let _ = std::fs::remove_file(&path);
    Ok(())
}

fn run_closed_loop_case(
    logical_payload_bytes: usize,
    fragments: usize,
    configured_fence_lanes: usize,
    groups: u64,
) -> Result<(), Box<dyn Error>> {
    let group_padded_bytes = padded_group_bytes(logical_payload_bytes, fragments);
    let capacity_bytes = usize::try_from(groups)
        .ok()
        .and_then(|groups| groups.checked_mul(group_padded_bytes))
        .ok_or("closed-loop capacity overflow")?;
    let path = bench_path(&format!(
        "closed-{logical_payload_bytes}-{fragments}-{configured_fence_lanes}"
    ));
    let _ = std::fs::remove_file(&path);
    let logical_payload: Vec<u8> = (0..logical_payload_bytes)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let mut expected = Vec::with_capacity(
        usize::try_from(groups)
            .ok()
            .and_then(|groups| groups.checked_mul(logical_payload_bytes))
            .ok_or("closed-loop expected payload overflow")?,
    );
    for _ in 0..groups {
        expected.extend_from_slice(&logical_payload);
    }
    let log = unsafe {
        FuaFrameLog::create(FuaFrameLogConfig {
            path: path.clone(),
            segment_id: 1,
            capacity_bytes,
        })?
    };
    let pool = log.spawn_fence_pool(configured_fence_lanes);
    let mut appender = log.appender();
    let mut group_ack_nanos = Vec::with_capacity(groups as usize);
    let mut direct_write_nanos = Vec::with_capacity(groups as usize * fragments);
    let mut next_seq = 0_u64;
    let started = Instant::now();

    for _ in 0..groups {
        let group_started = Instant::now();
        let base = logical_payload_bytes / fragments;
        let remainder = logical_payload_bytes % fragments;
        let mut offset = 0usize;
        let mut frame_ids = Vec::with_capacity(fragments);
        for fragment in 0..fragments {
            let bytes = base + usize::from(fragment < remainder);
            let frame =
                appender.publish_frame(&logical_payload[offset..offset + bytes], next_seq, 1)?;
            frame_ids.push(frame.frame_id);
            offset += bytes;
            next_seq += 1;
        }
        let terminal_frame = *frame_ids.last().expect("positive fragment count");
        while log.durable_frames() <= terminal_frame {
            if log.fence_failed() {
                return Err("fence pool failed during closed-loop group".into());
            }
            std::thread::yield_now();
        }
        group_ack_nanos.push(nanos(group_started.elapsed()));
        for frame_id in frame_ids {
            direct_write_nanos.push(
                log.frame_direct_write_service_nanos(frame_id)
                    .ok_or("closed-loop direct-write timestamp was unavailable")?,
            );
        }
    }
    let elapsed = started.elapsed();
    appender.finish();
    let fences = pool.join()?;
    if log.durable_seq() != next_seq {
        return Err("durable sequence did not reach closed-loop payload".into());
    }
    let recovered = recover_frame_log_by_scan(&path)?;
    let recovered_payload: Vec<u8> = recovered
        .into_iter()
        .flat_map(|frame| frame.payload)
        .collect();
    if recovered_payload != expected {
        return Err("closed-loop recovery bytes differ from published logical groups".into());
    }
    let telemetry = log.telemetry();
    let group_p50_nanos = percentile_nanos(&mut group_ack_nanos, 50, 100);
    let group_p99_nanos = percentile_nanos(&mut group_ack_nanos, 99, 100);
    let direct_write_p50_nanos = percentile_nanos(&mut direct_write_nanos, 50, 100);
    let direct_write_p99_nanos = percentile_nanos(&mut direct_write_nanos, 99, 100);
    let logical_bytes = groups.saturating_mul(logical_payload_bytes as u64);
    let actual_padded_bytes = telemetry.padded_bytes;
    let amplification_ppm = u128::from(actual_padded_bytes)
        .saturating_mul(1_000_000)
        .checked_div(u128::from(logical_bytes).max(1))
        .unwrap_or(u128::from(u64::MAX))
        .try_into()
        .unwrap_or(u64::MAX);
    print!(
        "fua_frame_log_bench_status=complete mode=closed_loop logical_payload_bytes={logical_payload_bytes} fragments={fragments} configured_fence_lanes={configured_fence_lanes} groups={groups} physical_frames={} fences={fences} group_ack_p50_nanos={group_p50_nanos} group_ack_p99_nanos={group_p99_nanos} frame_direct_write_p50_nanos={direct_write_p50_nanos} frame_direct_write_p99_nanos={direct_write_p99_nanos} elapsed_nanos={} groups_per_second_milli={} frames_per_second_milli={} logical_bytes={} actual_padded_bytes={} padded_to_logical_amplification_ppm={amplification_ppm} recovery=exact",
        telemetry.published_frames,
        nanos(elapsed),
        rate_per_second_milli(groups, elapsed),
        rate_per_second_milli(telemetry.published_frames, elapsed),
        logical_bytes,
        actual_padded_bytes,
    );
    print_telemetry(telemetry);
    println!();
    drop(log);
    let _ = std::fs::remove_file(&path);
    Ok(())
}

fn run_closed_loop() -> Result<(), Box<dyn Error>> {
    let groups = parse_env("CONVEYOR_FUA_GROUPS", 512_u64);
    if groups == 0 {
        return Err("CONVEYOR_FUA_GROUPS must be positive".into());
    }
    let payloads = parse_list(
        "CONVEYOR_FUA_LOGICAL_PAYLOAD_BYTES",
        &DEFAULT_CLOSED_LOGICAL_PAYLOADS,
    )?;
    let fragments = parse_list("CONVEYOR_FUA_QDS", &DEFAULT_CLOSED_QDS)?;
    let configured_fence_lanes = parse_env("CONVEYOR_FUA_POOL_LANES", 32_usize);
    if configured_fence_lanes == 0 {
        return Err("CONVEYOR_FUA_POOL_LANES must be positive".into());
    }
    for logical_payload_bytes in payloads {
        for fragment_count in &fragments {
            run_closed_loop_case(
                logical_payload_bytes,
                *fragment_count,
                configured_fence_lanes,
                groups,
            )?;
        }
    }
    Ok(())
}

/// Measurement-only controller experiment. It does not encode a production policy: it asks
/// whether one wider physical group can reset the device behavior seen by the next shallow group
/// without paying a permanently wider-frame byte cost.
fn run_controller_experiment() -> Result<(), Box<dyn Error>> {
    const LOGICAL_PAYLOAD_BYTES: usize = 30 * 1024;
    const POOL_LANES: usize = 32;
    const SHALLOW_FRAGMENTS: usize = 1;
    const PULSE_FRAGMENTS: usize = 16;
    let groups = parse_env("CONVEYOR_FUA_CONTROLLER_GROUPS", 512_u64);
    let threshold_nanos = parse_env("CONVEYOR_FUA_CONTROLLER_THRESHOLD_NANOS", 1_000_000_u64);
    let pulse_cooldown_groups = parse_env("CONVEYOR_FUA_CONTROLLER_COOLDOWN_GROUPS", 1_u64);
    if groups == 0 || threshold_nanos == 0 || pulse_cooldown_groups == 0 {
        return Err("controller groups, threshold, and cooldown must be positive".into());
    }
    let worst_case_group_bytes = padded_group_bytes(LOGICAL_PAYLOAD_BYTES, PULSE_FRAGMENTS);
    let capacity_bytes = usize::try_from(groups)
        .ok()
        .and_then(|groups| groups.checked_mul(worst_case_group_bytes))
        .ok_or("controller capacity overflow")?;
    let path = bench_path("controller");
    let _ = std::fs::remove_file(&path);
    let logical_payload: Vec<u8> = (0..LOGICAL_PAYLOAD_BYTES)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let mut expected = Vec::with_capacity(
        usize::try_from(groups)
            .ok()
            .and_then(|groups| groups.checked_mul(LOGICAL_PAYLOAD_BYTES))
            .ok_or("controller expected payload overflow")?,
    );
    for _ in 0..groups {
        expected.extend_from_slice(&logical_payload);
    }
    let log = unsafe {
        FuaFrameLog::create(FuaFrameLogConfig {
            path: path.clone(),
            segment_id: 1,
            capacity_bytes,
        })?
    };
    let pool = log.spawn_fence_pool(POOL_LANES);
    let mut appender = log.appender();
    let mut shallow_group_ack = Vec::new();
    let mut pulse_group_ack = Vec::new();
    let mut shallow_direct_write = Vec::new();
    let mut pulse_direct_write = Vec::new();
    let mut shallow_groups = 0_u64;
    let mut pulse_groups = 0_u64;
    let mut pulse_groups_remaining = 0_u64;
    let mut next_seq = 0_u64;
    let started = Instant::now();

    for _ in 0..groups {
        let fragments = if pulse_groups_remaining == 0 {
            SHALLOW_FRAGMENTS
        } else {
            PULSE_FRAGMENTS
        };
        let group_started = Instant::now();
        let base = LOGICAL_PAYLOAD_BYTES / fragments;
        let remainder = LOGICAL_PAYLOAD_BYTES % fragments;
        let mut offset = 0usize;
        let mut frame_ids = Vec::with_capacity(fragments);
        for fragment in 0..fragments {
            let bytes = base + usize::from(fragment < remainder);
            let frame =
                appender.publish_frame(&logical_payload[offset..offset + bytes], next_seq, 1)?;
            frame_ids.push(frame.frame_id);
            offset += bytes;
            next_seq += 1;
        }
        let terminal_frame = *frame_ids
            .last()
            .expect("positive controller fragment count");
        while log.durable_frames() <= terminal_frame {
            if log.fence_failed() {
                return Err("fence pool failed during controller experiment".into());
            }
            std::thread::yield_now();
        }
        let group_ack = nanos(group_started.elapsed());
        let mut observed_direct_max = 0_u64;
        for frame_id in frame_ids {
            let direct = log
                .frame_direct_write_service_nanos(frame_id)
                .ok_or("controller direct-write timestamp was unavailable")?;
            observed_direct_max = observed_direct_max.max(direct);
            if fragments == SHALLOW_FRAGMENTS {
                shallow_direct_write.push(direct);
            } else {
                pulse_direct_write.push(direct);
            }
        }
        if fragments == SHALLOW_FRAGMENTS {
            shallow_groups += 1;
            shallow_group_ack.push(group_ack);
            if group_ack.max(observed_direct_max) > threshold_nanos {
                pulse_groups_remaining = pulse_cooldown_groups;
            }
        } else {
            pulse_groups += 1;
            pulse_group_ack.push(group_ack);
            pulse_groups_remaining = pulse_groups_remaining.saturating_sub(1);
        }
    }
    let elapsed = started.elapsed();
    appender.finish();
    let fences = pool.join()?;
    if log.durable_seq() != next_seq {
        return Err("durable sequence did not reach controller payload".into());
    }
    let recovered_payload: Vec<u8> = recover_frame_log_by_scan(&path)?
        .into_iter()
        .flat_map(|frame| frame.payload)
        .collect();
    if recovered_payload != expected {
        return Err("controller recovery bytes differ from published logical groups".into());
    }
    let telemetry = log.telemetry();
    let logical_bytes = groups.saturating_mul(LOGICAL_PAYLOAD_BYTES as u64);
    let amplification_ppm = u128::from(telemetry.padded_bytes)
        .saturating_mul(1_000_000)
        .checked_div(u128::from(logical_bytes).max(1))
        .unwrap_or(u128::from(u64::MAX))
        .try_into()
        .unwrap_or(u64::MAX);
    print!(
        "fua_frame_log_bench_status=complete mode=controller logical_payload_bytes={LOGICAL_PAYLOAD_BYTES} configured_fence_lanes={POOL_LANES} shallow_fragments={SHALLOW_FRAGMENTS} pulse_fragments={PULSE_FRAGMENTS} controller_threshold_nanos={threshold_nanos} controller_pulse_cooldown_groups={pulse_cooldown_groups} groups={groups} shallow_groups={shallow_groups} pulse_groups={pulse_groups} fences={fences} shallow_group_ack_p50_nanos={} shallow_group_ack_p99_nanos={} pulse_group_ack_p50_nanos={} pulse_group_ack_p99_nanos={} shallow_direct_write_p50_nanos={} shallow_direct_write_p99_nanos={} pulse_direct_write_p50_nanos={} pulse_direct_write_p99_nanos={} elapsed_nanos={} groups_per_second_milli={} logical_bytes={} actual_padded_bytes={} padded_to_logical_amplification_ppm={amplification_ppm} recovery=exact",
        percentile_or_zero(&mut shallow_group_ack, 50, 100),
        percentile_or_zero(&mut shallow_group_ack, 99, 100),
        percentile_or_zero(&mut pulse_group_ack, 50, 100),
        percentile_or_zero(&mut pulse_group_ack, 99, 100),
        percentile_or_zero(&mut shallow_direct_write, 50, 100),
        percentile_or_zero(&mut shallow_direct_write, 99, 100),
        percentile_or_zero(&mut pulse_direct_write, 50, 100),
        percentile_or_zero(&mut pulse_direct_write, 99, 100),
        nanos(elapsed),
        rate_per_second_milli(groups, elapsed),
        logical_bytes,
        telemetry.padded_bytes,
    );
    print_telemetry(telemetry);
    println!();
    drop(log);
    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[derive(Default)]
struct ControllerTraceMetrics {
    production_ack_nanos: Vec<u64>,
    production_durability_nanos: Vec<u64>,
    production_qd16_durability_nanos: Vec<u64>,
    production_qd1_probe_durability_nanos: Vec<u64>,
    preflight_groups: u64,
    production_groups: u64,
    natural_qd8_groups: u64,
    natural_qd16_groups: u64,
    natural_qd32_groups: u64,
    qd1_probe_groups: u64,
    sustained_qd16_groups: u64,
    pending_probe_cover_groups: u64,
    fast_qd1: u64,
    gray_qd1: u64,
    slow_qd1: u64,
    qd16_to_qd1_pairs: u64,
    gap_mask: u8,
}

struct GroupMeasurement {
    ack_nanos: u64,
    max_direct_write_nanos: u64,
    terminal_direct_write_nanos: u64,
}

fn split_logical_payload(payload: &[u8], fragments: usize) -> Option<Vec<&[u8]>> {
    if fragments == 0 || payload.len() < fragments {
        return None;
    }
    let base = payload.len() / fragments;
    let remainder = payload.len() % fragments;
    let mut offset = 0usize;
    let mut chunks = Vec::with_capacity(fragments);
    for fragment in 0..fragments {
        let bytes = base + usize::from(fragment < remainder);
        chunks.push(&payload[offset..offset + bytes]);
        offset += bytes;
    }
    Some(chunks)
}

fn append_and_wait_group(
    log: &FuaFrameLog,
    appender: &mut FuaFrameLogAppender,
    logical_payload: &[u8],
    fragments: usize,
    next_seq: &mut u64,
) -> Result<GroupMeasurement, Box<dyn Error>> {
    let group_started = Instant::now();
    let chunks = split_logical_payload(logical_payload, fragments)
        .ok_or("controller trace physical chunks are not nonempty")?;
    let (first_frame, terminal_frame) = if fragments == 1 {
        let frame = appender.publish_frame(logical_payload, *next_seq, 1)?;
        (frame.frame_id, frame.frame_id)
    } else {
        let batch = appender.publish_batch(&chunks, *next_seq, 1)?;
        (batch.first_frame_id, batch.terminal_frame_id)
    };
    *next_seq = next_seq.saturating_add(1);
    while log.durable_frames() <= terminal_frame {
        if log.fence_failed() {
            return Err("fence pool failed during controller trace".into());
        }
        std::thread::yield_now();
    }
    let mut max_direct_write_nanos = 0_u64;
    for frame_id in first_frame..=terminal_frame {
        max_direct_write_nanos = max_direct_write_nanos.max(
            log.frame_direct_write_service_nanos(frame_id)
                .ok_or("controller trace direct-write timestamp was unavailable")?,
        );
    }
    Ok(GroupMeasurement {
        ack_nanos: nanos(group_started.elapsed()),
        max_direct_write_nanos,
        terminal_direct_write_nanos: log
            .frame_direct_write_service_nanos(terminal_frame)
            .ok_or("controller trace terminal direct-write timestamp was unavailable")?,
    })
}

fn verify_recovered_repeating_payload(
    path: &std::path::Path,
    logical_payload: &[u8],
    logical_groups: u64,
) -> Result<(), Box<dyn Error>> {
    let expected_bytes = usize::try_from(logical_groups)
        .ok()
        .and_then(|groups| groups.checked_mul(logical_payload.len()))
        .ok_or("controller trace expected byte count overflow")?;
    let mut observed_bytes = 0usize;
    for frame in recover_frame_log_by_scan(path)? {
        for byte in frame.payload {
            if observed_bytes >= expected_bytes
                || byte != logical_payload[observed_bytes % logical_payload.len()]
            {
                return Err(
                    "controller trace recovery bytes differ from published logical groups".into(),
                );
            }
            observed_bytes += 1;
        }
    }
    if observed_bytes != expected_bytes {
        return Err("controller trace recovery ended before the published logical groups".into());
    }
    Ok(())
}

fn next_trace_random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn trace_gap_micros(group_index: u64, random_state: &mut u64) -> (usize, u64) {
    let index = if group_index < CONTROLLER_GAPS_MICROS.len() as u64 {
        group_index as usize
    } else {
        (next_trace_random(random_state) as usize) % CONTROLLER_GAPS_MICROS.len()
    };
    (index, CONTROLLER_GAPS_MICROS[index])
}

struct TraceRun<'a> {
    log: &'a FuaFrameLog,
    appender: &'a mut FuaFrameLogAppender,
    logical_payload: &'a [u8],
    next_seq: &'a mut u64,
    controller: &'a mut FuaPhysicalController,
    metrics: &'a mut ControllerTraceMetrics,
}

impl TraceRun<'_> {
    fn run_group(
        &mut self,
        natural_fragments: usize,
        is_production_group: bool,
        gap_index: usize,
        gap_micros: u64,
    ) -> Result<(), Box<dyn Error>> {
        let fragment_count = FUA_CONTROLLER_QD16_FRAGMENTS
            .saturating_sub(natural_fragments.min(FUA_CONTROLLER_QD16_FRAGMENTS));
        let candidate_chunks = split_logical_payload(self.logical_payload, fragment_count);
        let candidate_padded = candidate_chunks
            .as_deref()
            .map(|chunks| {
                chunks
                    .iter()
                    .map(|chunk| fua_frame_padded_bytes(chunk.len()))
                    .sum()
            })
            .unwrap_or(usize::MAX);
        let one_segment = candidate_chunks
            .as_deref()
            .is_some_and(|chunks| self.appender.can_publish_batch(chunks).is_ok());
        let decision = self.controller.select(FuaControllerEligibility {
            pool_lanes: CONTROLLER_POOL_LANES,
            natural_depth: natural_fragments,
            free_slots: CONTROLLER_POOL_LANES.saturating_sub(natural_fragments),
            fragment_count,
            chunks_nonempty: candidate_chunks.is_some(),
            one_segment,
            single_frame_padded_bytes: fua_frame_padded_bytes(self.logical_payload.len()),
            fragmented_padded_bytes: candidate_padded,
        });
        let fragments = if natural_fragments >= FUA_CONTROLLER_QD16_FRAGMENTS {
            natural_fragments
        } else {
            decision.fragments()
        };
        let measurement = append_and_wait_group(
            self.log,
            self.appender,
            self.logical_payload,
            fragments,
            self.next_seq,
        )?;
        self.metrics.gap_mask |= 1_u8 << gap_index;
        let sample_token = self.controller.record_published(decision);
        if !is_production_group {
            match natural_fragments {
                8 => {
                    self.metrics.natural_qd8_groups =
                        self.metrics.natural_qd8_groups.saturating_add(1)
                }
                16 => {
                    self.metrics.natural_qd16_groups =
                        self.metrics.natural_qd16_groups.saturating_add(1)
                }
                32 => {
                    self.metrics.natural_qd32_groups =
                        self.metrics.natural_qd32_groups.saturating_add(1)
                }
                0 => {}
                _ => return Err("controller trace used an unsupported natural depth".into()),
            }
        }
        match decision {
            FuaControllerDecision::Unfragmented(_) => {}
            FuaControllerDecision::Qd1Sample { .. } => {
                self.metrics.qd1_probe_groups = self.metrics.qd1_probe_groups.saturating_add(1);
                let direct_nanos = measurement.terminal_direct_write_nanos;
                self.controller.observe_qd1_sample(
                    sample_token.expect("published QD1 sample has a controller token"),
                    direct_nanos,
                );
                if direct_nanos <= FUA_CONTROLLER_FAST_NANOS {
                    self.metrics.fast_qd1 = self.metrics.fast_qd1.saturating_add(1);
                } else if direct_nanos >= CONTROLLER_SLOW_NANOS {
                    self.metrics.slow_qd1 = self.metrics.slow_qd1.saturating_add(1);
                } else {
                    self.metrics.gray_qd1 = self.metrics.gray_qd1.saturating_add(1);
                }
            }
            FuaControllerDecision::SustainedEpoch { .. } => {
                self.metrics.sustained_qd16_groups =
                    self.metrics.sustained_qd16_groups.saturating_add(1);
            }
            FuaControllerDecision::PendingProbeCover { .. } => {
                self.metrics.pending_probe_cover_groups =
                    self.metrics.pending_probe_cover_groups.saturating_add(1);
            }
        }
        if is_production_group {
            self.metrics.production_groups = self.metrics.production_groups.saturating_add(1);
            self.metrics
                .production_ack_nanos
                .push(measurement.ack_nanos);
            self.metrics
                .production_durability_nanos
                .push(measurement.max_direct_write_nanos);
            match decision {
                FuaControllerDecision::Qd1Sample { .. } => self
                    .metrics
                    .production_qd1_probe_durability_nanos
                    .push(measurement.terminal_direct_write_nanos),
                FuaControllerDecision::SustainedEpoch { .. }
                | FuaControllerDecision::PendingProbeCover { .. } => self
                    .metrics
                    .production_qd16_durability_nanos
                    .push(measurement.max_direct_write_nanos),
                FuaControllerDecision::Unfragmented(_) => {
                    return Err("controller trace produced a natural production decision".into())
                }
            }
        } else {
            self.metrics.preflight_groups = self.metrics.preflight_groups.saturating_add(1);
        }
        if gap_micros != 0 {
            std::thread::sleep(Duration::from_micros(gap_micros));
        }
        Ok(())
    }
}

fn mean_nanos(samples: &[u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let sum: u128 = samples.iter().map(|sample| u128::from(*sample)).sum();
    (sum / samples.len() as u128).try_into().unwrap_or(u64::MAX)
}

/// Execute one fresh-process-friendly, reproducible controller trace. The preflight rotates
/// randomized blocks containing QD16->QD1, QD8, and QD32. The following 8,000-group production
/// cadence consists only of logical QD1 requests so its gate measures the proposed controller,
/// not deliberately injected negative-control traffic.
fn run_controller_trace() -> Result<(), Box<dyn Error>> {
    let production_groups = parse_env(
        "CONVEYOR_FUA_CONTROLLER_TRACE_GROUPS",
        CONTROLLER_GATE_GROUPS,
    );
    let preflight_cycles = parse_env("CONVEYOR_FUA_CONTROLLER_TRACE_PREFLIGHT_CYCLES", 4_u64);
    let trace_seed = parse_env(
        "CONVEYOR_FUA_CONTROLLER_TRACE_SEED",
        0x6a09_e667_f3bc_c909_u64,
    );
    let mut random_state = trace_seed;
    let enforce_gate = parse_env("CONVEYOR_FUA_CONTROLLER_TRACE_ENFORCE_GATE", 0_u8) != 0;
    if production_groups == 0 || preflight_cycles == 0 {
        return Err("controller trace group and preflight-cycle counts must be positive".into());
    }
    let one_frame_padded = padded_group_bytes(CONTROLLER_LOGICAL_PAYLOAD_BYTES, 1);
    let boost_padded = padded_group_bytes(
        CONTROLLER_LOGICAL_PAYLOAD_BYTES,
        FUA_CONTROLLER_QD16_FRAGMENTS,
    );
    if CONTROLLER_LOGICAL_PAYLOAD_BYTES == 0
        || CONTROLLER_POOL_LANES < FUA_CONTROLLER_QD16_FRAGMENTS
        || boost_padded > one_frame_padded.saturating_mul(2)
    {
        return Err("controller trace boost eligibility was unexpectedly false".into());
    }
    let preflight_groups = preflight_cycles
        .checked_mul(5)
        .ok_or("controller trace preflight count overflow")?;
    let total_groups = production_groups
        .checked_add(preflight_groups)
        .ok_or("controller trace total group count overflow")?;
    let natural_qd32_padded =
        padded_group_bytes(CONTROLLER_LOGICAL_PAYLOAD_BYTES, CONTROLLER_POOL_LANES);
    let preflight_qd32_extra_bytes = usize::try_from(preflight_cycles)
        .ok()
        .and_then(|cycles| cycles.checked_mul(natural_qd32_padded.saturating_sub(boost_padded)))
        .ok_or("controller trace preflight capacity overflow")?;
    let capacity_bytes = usize::try_from(total_groups)
        .ok()
        .and_then(|groups| groups.checked_mul(boost_padded))
        .and_then(|bytes| bytes.checked_add(preflight_qd32_extra_bytes))
        .ok_or("controller trace capacity overflow")?;
    let path = bench_path("controller-trace");
    let _ = std::fs::remove_file(&path);
    let logical_payload: Vec<u8> = (0..CONTROLLER_LOGICAL_PAYLOAD_BYTES)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let log = unsafe {
        FuaFrameLog::create(FuaFrameLogConfig {
            path: path.clone(),
            segment_id: 1,
            capacity_bytes,
        })?
    };
    let pool = log.spawn_fence_pool(CONTROLLER_POOL_LANES);
    let mut appender = log.appender();
    let mut controller = FuaPhysicalController::default();
    let mut metrics = ControllerTraceMetrics::default();
    let mut next_seq = 0_u64;
    let started = Instant::now();
    let mut trace_group_index = 0_u64;
    {
        let mut trace = TraceRun {
            log: &log,
            appender: &mut appender,
            logical_payload: &logical_payload,
            next_seq: &mut next_seq,
            controller: &mut controller,
            metrics: &mut metrics,
        };

        for _ in 0..preflight_cycles {
            let mut block_order = [0_u8, 1, 2, 3];
            for index in (1..block_order.len()).rev() {
                let swap_index = (next_trace_random(&mut random_state) as usize) % (index + 1);
                block_order.swap(index, swap_index);
            }
            for block in block_order {
                let depths: &[usize] = match block {
                    0 => {
                        trace.metrics.qd16_to_qd1_pairs =
                            trace.metrics.qd16_to_qd1_pairs.saturating_add(1);
                        &[16, 0]
                    }
                    1 => &[8],
                    2 => &[32],
                    3 => &[0],
                    _ => unreachable!("fixed controller trace block"),
                };
                for natural_fragments in depths {
                    let (gap_index, gap_micros) =
                        trace_gap_micros(trace_group_index, &mut random_state);
                    trace.run_group(*natural_fragments, false, gap_index, gap_micros)?;
                    trace_group_index = trace_group_index.saturating_add(1);
                }
            }
        }

        // Preflight traffic proves the negative controls. Reset before measuring the production
        // cadence so every one of its logical groups is governed by the same initial controller state.
        *trace.controller = FuaPhysicalController::default();
        for _ in 0..production_groups {
            let (gap_index, gap_micros) = trace_gap_micros(trace_group_index, &mut random_state);
            trace.run_group(0, true, gap_index, gap_micros)?;
            trace_group_index = trace_group_index.saturating_add(1);
        }
    }
    let elapsed = started.elapsed();
    appender.finish();
    let fences = pool.join()?;
    if log.durable_seq() != next_seq {
        return Err("durable sequence did not reach controller trace payload".into());
    }
    verify_recovered_repeating_payload(&path, &logical_payload, total_groups)?;
    if metrics.qd16_to_qd1_pairs != preflight_cycles
        || metrics.natural_qd8_groups != preflight_cycles
        || metrics.natural_qd16_groups != preflight_cycles
        || metrics.natural_qd32_groups != preflight_cycles
        || metrics.gap_mask != (1_u8 << CONTROLLER_GAPS_MICROS.len()) - 1
    {
        return Err("controller trace did not cover all required randomized controls".into());
    }
    let decision_reconciled_groups = metrics
        .production_qd16_durability_nanos
        .len()
        .saturating_add(metrics.production_qd1_probe_durability_nanos.len())
        as u64;
    if decision_reconciled_groups != metrics.production_groups {
        return Err("controller trace decisions did not reconcile to production groups".into());
    }
    let telemetry = log.telemetry();
    let logical_bytes = total_groups.saturating_mul(CONTROLLER_LOGICAL_PAYLOAD_BYTES as u64);
    let amplification_ppm = u128::from(telemetry.padded_bytes)
        .saturating_mul(1_000_000)
        .checked_div(u128::from(logical_bytes).max(1))
        .unwrap_or(u128::from(u64::MAX))
        .try_into()
        .unwrap_or(u64::MAX);
    let durability_mean_nanos = mean_nanos(&metrics.production_durability_nanos);
    let durability_p99_nanos = percentile_nanos(&mut metrics.production_durability_nanos, 99, 100);
    let durability_over_p99_count = metrics
        .production_durability_nanos
        .iter()
        .filter(|sample| **sample > CONTROLLER_GATE_P99_NANOS)
        .count() as u64;
    let qd16_p99_nanos = percentile_or_zero(&mut metrics.production_qd16_durability_nanos, 99, 100);
    let qd16_over_p99_count = metrics
        .production_qd16_durability_nanos
        .iter()
        .filter(|sample| **sample > CONTROLLER_GATE_P99_NANOS)
        .count() as u64;
    let qd1_probe_p99_nanos =
        percentile_or_zero(&mut metrics.production_qd1_probe_durability_nanos, 99, 100);
    let qd1_probe_over_p99_count = metrics
        .production_qd1_probe_durability_nanos
        .iter()
        .filter(|sample| **sample > CONTROLLER_GATE_P99_NANOS)
        .count() as u64;
    let ack_mean_nanos = mean_nanos(&metrics.production_ack_nanos);
    let ack_p99_nanos = percentile_nanos(&mut metrics.production_ack_nanos, 99, 100);
    let gate_applicable = metrics.production_groups == CONTROLLER_GATE_GROUPS;
    let gate_pass = gate_applicable
        && durability_mean_nanos <= CONTROLLER_GATE_MEAN_NANOS
        && durability_p99_nanos <= CONTROLLER_GATE_P99_NANOS
        && durability_over_p99_count <= CONTROLLER_GATE_MAX_OVER_P99_NANOS;
    let gate_result = if !gate_applicable {
        "not_applicable"
    } else if gate_pass {
        "pass"
    } else {
        "fail"
    };
    let controller_telemetry = controller.telemetry();
    if !controller.action_reconciliation_ok()
        || !controller.sample_reconciliation_ok()
        || !controller.is_quiescent()
    {
        return Err("controller trace ended with unreconciled controller state".into());
    }
    print!(
        "fua_frame_log_bench_status=complete mode=controller_trace seed={trace_seed} configured_fence_lanes={CONTROLLER_POOL_LANES} logical_payload_bytes={CONTROLLER_LOGICAL_PAYLOAD_BYTES} controller_initial_phase=sustained_slow controller_fast_nanos={FUA_CONTROLLER_FAST_NANOS} controller_slow_nanos={CONTROLLER_SLOW_NANOS} controller_qd16_fragments={FUA_CONTROLLER_QD16_FRAGMENTS} controller_sustained_qd16_groups={FUA_CONTROLLER_SUSTAINED_GROUPS} controller_one_segment=1 controller_nonempty=1 controller_qd16_eligible=1 controller_one_frame_padded_bytes={one_frame_padded} controller_qd16_padded_bytes={boost_padded} preflight_cycles={preflight_cycles} preflight_groups={} production_cadence_groups={} decision_reconciled_groups={decision_reconciled_groups} decision_reconciliation=exact qd16_to_qd1_pairs={} natural_qd8_groups={} natural_qd16_groups={} natural_qd32_groups={} controller_qd1_probe_count={} controller_qd16_groups={} controller_pending_probe_cover_groups={} controller_sustained_qd16_groups_total={} controller_fast_qd1={} controller_gray_qd1={} controller_slow_qd1={} controller_phase_final={} controller_verify_fast_streak={} controller_sustained_qd16_remaining={} controller_action_reconciliation=exact controller_sample_reconciliation=exact controller_gap_mask={} fences={fences} production_durability_mean_nanos={durability_mean_nanos} production_durability_p99_nanos={durability_p99_nanos} production_durability_over_1100us_count={durability_over_p99_count} production_qd16_p99_nanos={qd16_p99_nanos} production_qd16_over_1100us_count={qd16_over_p99_count} production_qd1_probe_p99_nanos={qd1_probe_p99_nanos} production_qd1_probe_over_1100us_count={qd1_probe_over_p99_count} production_ack_mean_nanos={ack_mean_nanos} production_ack_p99_nanos={ack_p99_nanos} gate_required_groups={CONTROLLER_GATE_GROUPS} gate_mean_limit_nanos={CONTROLLER_GATE_MEAN_NANOS} gate_p99_limit_nanos={CONTROLLER_GATE_P99_NANOS} gate_max_over_1100us_count={CONTROLLER_GATE_MAX_OVER_P99_NANOS} gate_result={gate_result} elapsed_nanos={} groups_per_second_milli={} logical_bytes={} actual_padded_bytes={} padded_to_logical_amplification_ppm={amplification_ppm} recovery=exact",
        metrics.preflight_groups,
        metrics.production_groups,
        metrics.qd16_to_qd1_pairs,
        metrics.natural_qd8_groups,
        metrics.natural_qd16_groups,
        metrics.natural_qd32_groups,
        metrics.production_qd1_probe_durability_nanos.len(),
        metrics.production_qd16_durability_nanos.len(),
        metrics.pending_probe_cover_groups,
        metrics.sustained_qd16_groups,
        metrics.fast_qd1,
        metrics.gray_qd1,
        metrics.slow_qd1,
        controller_telemetry.phase.as_str(),
        controller_telemetry.verify_fast_streak,
        controller_telemetry.sustained_qd16_remaining,
        metrics.gap_mask,
        nanos(elapsed),
        rate_per_second_milli(total_groups, elapsed),
        logical_bytes,
        telemetry.padded_bytes,
    );
    print_telemetry(telemetry);
    println!();
    drop(log);
    let _ = std::fs::remove_file(&path);
    if enforce_gate && !gate_pass {
        return Err(format!(
            "controller trace gate {gate_result}: mean={durability_mean_nanos}ns p99={durability_p99_nanos}ns"
        )
        .into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    match std::env::var("CONVEYOR_FUA_MODE")
        .unwrap_or_else(|_| "open".to_string())
        .as_str()
    {
        "open" => run_open_loop(),
        "closed" => run_closed_loop(),
        "controller" => run_controller_experiment(),
        "controller_trace" => run_controller_trace(),
        "both" => {
            run_open_loop()?;
            run_closed_loop()
        }
        mode => Err(format!(
            "CONVEYOR_FUA_MODE must be open, closed, controller, controller_trace, or both (got {mode:?})"
        )
        .into()),
    }
}
