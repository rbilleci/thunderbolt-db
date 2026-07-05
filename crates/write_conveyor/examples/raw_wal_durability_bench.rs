use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use gpu_db_write_conveyor::{
    recover_wal_segment_by_scan, stats_for_range, MappedWalSegment, WalDataSyncMode,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FenceKind {
    Data,
    Prefix,
}

#[derive(Clone, Copy, Debug)]
struct Percentiles {
    p50: Duration,
    p90: Duration,
    p99: Duration,
    max: Duration,
    avg: Duration,
}

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}

fn parse_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => default,
    }
}

fn parse_fence_kind() -> Result<FenceKind, Box<dyn Error>> {
    match std::env::var("CONVEYOR_RAW_FENCE_KIND")
        .unwrap_or_else(|_| "data".to_string())
        .as_str()
    {
        "data" | "data-frontier" => Ok(FenceKind::Data),
        "prefix" | "durable-prefix" | "control" => Ok(FenceKind::Prefix),
        value => {
            Err(format!("unsupported CONVEYOR_RAW_FENCE_KIND={value}; use data or prefix").into())
        }
    }
}

fn parse_sync_modes() -> Result<Vec<WalDataSyncMode>, Box<dyn Error>> {
    let value = std::env::var("CONVEYOR_RAW_SYNC_MODES")
        .or_else(|_| std::env::var("CONVEYOR_DURABLE_SYNC_MODE"))
        .unwrap_or_else(|_| "sync-write-data".to_string());
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(parse_sync_mode)
        .collect()
}

fn parse_sync_mode(value: &str) -> Result<WalDataSyncMode, Box<dyn Error>> {
    match value {
        "range-and-file-data" | "strict" => Ok(WalDataSyncMode::RangeAndFileData),
        "write-and-file-data" | "write-through" => Ok(WalDataSyncMode::WriteAndFileData),
        "prewrite-and-file-data" | "prewrite" => Ok(WalDataSyncMode::PrewriteAndFileData),
        "sync-write-data" | "rwf-dsync" => Ok(WalDataSyncMode::SyncWriteData),
        "file-data-only" | "fdatasync" => Ok(WalDataSyncMode::FileDataOnly),
        value => Err(format!(
            "unsupported sync mode {value}; use range-and-file-data, write-and-file-data, prewrite-and-file-data, sync-write-data, or file-data-only"
        )
        .into()),
    }
}

fn parse_group_list() -> Result<Vec<u64>, Box<dyn Error>> {
    let value = std::env::var("CONVEYOR_RAW_FENCE_BLOCKS").unwrap_or_else(|_| "1".to_string());
    let mut groups = Vec::new();
    for item in value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let group = item.parse::<u64>()?;
        if group == 0 {
            return Err("CONVEYOR_RAW_FENCE_BLOCKS entries must be greater than zero".into());
        }
        groups.push(group);
    }
    if groups.is_empty() {
        return Err("CONVEYOR_RAW_FENCE_BLOCKS must contain at least one value".into());
    }
    Ok(groups)
}

fn parse_raw_block_size() -> usize {
    std::env::var("CONVEYOR_RAW_WAL_BLOCK")
        .or_else(|_| std::env::var("CONVEYOR_CLIENT_WAL_BLOCK"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1)
}

fn sync_mode_label(mode: WalDataSyncMode) -> &'static str {
    match mode {
        WalDataSyncMode::RangeAndFileData => "range-and-file-data",
        WalDataSyncMode::WriteAndFileData => "write-and-file-data",
        WalDataSyncMode::PrewriteAndFileData => "prewrite-and-file-data",
        WalDataSyncMode::SyncWriteData => "sync-write-data",
        WalDataSyncMode::FileDataOnly => "file-data-only",
    }
}

fn sync_case_label(mode: Option<WalDataSyncMode>) -> &'static str {
    match mode {
        Some(mode) => sync_mode_label(mode),
        None => "n/a",
    }
}

fn sync_case_path_label(mode: Option<WalDataSyncMode>) -> &'static str {
    match mode {
        Some(mode) => sync_mode_label(mode),
        None => "prefix",
    }
}

fn fence_kind_label(kind: FenceKind) -> &'static str {
    match kind {
        FenceKind::Data => "data-frontier",
        FenceKind::Prefix => "durable-prefix",
    }
}

fn percentile(sorted: &[Duration], percentile: usize) -> Duration {
    debug_assert!(!sorted.is_empty());
    let index = (sorted.len() - 1) * percentile / 100;
    sorted[index]
}

fn distribution(samples: &[Duration]) -> Option<Percentiles> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let total_ns: u128 = sorted.iter().map(Duration::as_nanos).sum();
    Some(Percentiles {
        p50: percentile(&sorted, 50),
        p90: percentile(&sorted, 90),
        p99: percentile(&sorted, 99),
        max: *sorted.last().expect("non-empty sorted samples"),
        avg: Duration::from_nanos((total_ns / sorted.len() as u128) as u64),
    })
}

fn format_duration(duration: Duration) -> String {
    let ns = duration.as_nanos();
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.2}us", ns as f64 / 1_000.0)
    } else {
        format!("{:.3}ms", ns as f64 / 1_000_000.0)
    }
}

fn print_distribution(label: &str, samples: &[Duration]) {
    match distribution(samples) {
        Some(dist) => println!(
            "    {label:<22} avg={} p50={} p90={} p99={} max={} samples={}",
            format_duration(dist.avg),
            format_duration(dist.p50),
            format_duration(dist.p90),
            format_duration(dist.p99),
            format_duration(dist.max),
            samples.len()
        ),
        None => println!("    {label:<22} no samples"),
    }
}

fn case_path(
    base: &Path,
    mode: Option<WalDataSyncMode>,
    fence_blocks: u64,
    multi_case: bool,
) -> PathBuf {
    if !multi_case {
        return base.to_path_buf();
    }
    let mut file_name = base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("raw-wal-durability")
        .to_string();
    file_name.push('.');
    file_name.push_str(sync_case_path_label(mode));
    file_name.push('.');
    file_name.push_str(&fence_blocks.to_string());
    file_name.push_str(".dat");
    base.with_file_name(file_name)
}

fn remove_existing(path: &Path, overwrite: bool) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        if !overwrite {
            return Err(format!(
                "{} already exists; set CONVEYOR_OVERWRITE=1 to replace it",
                path.display()
            )
            .into());
        }
        std::fs::remove_file(path)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let events: u64 = parse_env("CONVEYOR_EVENTS", 500_000_u64);
    let block_size: usize = parse_raw_block_size();
    let groups = parse_group_list()?;
    let fence_kind = parse_fence_kind()?;
    let sync_modes = match fence_kind {
        FenceKind::Data => parse_sync_modes()?,
        FenceKind::Prefix => Vec::new(),
    };
    let validate = parse_bool("CONVEYOR_RAW_VALIDATE", true);
    let keep_file = parse_bool("CONVEYOR_KEEP_FILE", false);
    let (base_path, generated_path) = match std::env::var("CONVEYOR_FILE") {
        Ok(path) => (PathBuf::from(path), false),
        Err(_) => (
            PathBuf::from(format!(
                "target/raw-wal-durability-{}.dat",
                std::process::id()
            )),
            true,
        ),
    };
    let overwrite = generated_path || parse_bool("CONVEYOR_OVERWRITE", false);

    if events == 0 {
        return Err("CONVEYOR_EVENTS must be greater than zero".into());
    }
    let total_blocks = events.div_ceil(block_size as u64);
    if events > usize::MAX as u64 {
        return Err("raw WAL benchmark event count exceeds usize".into());
    }
    if total_blocks > usize::MAX as u64 {
        return Err("raw WAL benchmark block count exceeds usize".into());
    }

    println!("raw WAL durability benchmark");
    println!(
        "  events={events} block_size={block_size} blocks={total_blocks} fence_kind={}",
        fence_kind_label(fence_kind)
    );
    println!(
        "  sync_modes={} fence_blocks={} validate={validate}",
        match fence_kind {
            FenceKind::Data => sync_modes
                .iter()
                .map(|mode| sync_mode_label(*mode))
                .collect::<Vec<_>>()
                .join(","),
            FenceKind::Prefix => "n/a".to_string(),
        },
        groups
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );

    let sync_cases: Vec<Option<WalDataSyncMode>> = match fence_kind {
        FenceKind::Data => sync_modes.into_iter().map(Some).collect(),
        FenceKind::Prefix => vec![None],
    };
    let multi_case = sync_cases.len() > 1 || groups.len() > 1;
    for mode in sync_cases {
        for &fence_blocks in groups.iter() {
            run_case(
                events,
                total_blocks,
                block_size,
                fence_blocks,
                mode,
                fence_kind,
                validate,
                keep_file,
                overwrite,
                &case_path(&base_path, mode, fence_blocks, multi_case),
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    events: u64,
    total_blocks: u64,
    block_size: usize,
    fence_blocks: u64,
    sync_mode: Option<WalDataSyncMode>,
    fence_kind: FenceKind,
    validate: bool,
    keep_file: bool,
    overwrite: bool,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    remove_existing(path, overwrite)?;
    let segment = unsafe { MappedWalSegment::create(path, 0, events as usize, block_size)? };
    let mut publish_samples = Vec::with_capacity(total_blocks as usize);
    let mut fence_samples = Vec::with_capacity(total_blocks.div_ceil(fence_blocks) as usize);
    let mut group_samples = Vec::with_capacity(total_blocks.div_ceil(fence_blocks) as usize);
    let mut published_records = 0_u64;
    let mut published_blocks = 0_u64;

    let elapsed_start = Instant::now();
    while published_blocks < total_blocks {
        let group_start = Instant::now();
        let group_end_blocks = published_blocks
            .saturating_add(fence_blocks)
            .min(total_blocks);
        while published_blocks < group_end_blocks {
            let remaining_records = events - published_records;
            let count = remaining_records.min(block_size as u64) as usize;
            let publish_start = Instant::now();
            let block_id = segment
                .try_publish_block_position(published_records, count)?
                .ok_or("raw WAL benchmark published an empty block")?;
            if block_id != published_blocks {
                return Err(format!(
                    "raw WAL benchmark block id mismatch: got {block_id}, expected {published_blocks}"
                )
                .into());
            }
            publish_samples.push(publish_start.elapsed());
            published_records += count as u64;
            published_blocks += 1;
        }

        let fence_start = Instant::now();
        match fence_kind {
            FenceKind::Data => {
                let sync_mode =
                    sync_mode.ok_or("raw WAL data-frontier case requires a sync mode")?;
                segment.sync_published_data_frontier_with_mode(published_blocks, sync_mode)?;
            }
            FenceKind::Prefix => {
                segment.sync_published_prefix(published_blocks)?;
            }
        }
        fence_samples.push(fence_start.elapsed());
        group_samples.push(group_start.elapsed());
    }
    let elapsed = elapsed_start.elapsed();
    drop(segment);

    let recovery_start = Instant::now();
    let recovery_result = if validate {
        Some((recover_wal_segment_by_scan(path)?, recovery_start.elapsed()))
    } else {
        None
    };

    if let Some((recovered, _recovery_elapsed)) = &recovery_result {
        let expected = stats_for_range(0, events);
        if recovered.recovered_blocks != total_blocks
            || recovered.recovered_records != events
            || recovered.stats != expected
        {
            return Err(format!(
                "raw WAL recovery mismatch: recovered_blocks={} recovered_records={} stats={:?} expected_blocks={total_blocks} expected_records={events} expected_stats={:?}",
                recovered.recovered_blocks, recovered.recovered_records, recovered.stats, expected
            )
            .into());
        }
    }

    if !keep_file {
        std::fs::remove_file(path)?;
    }

    let records_per_sec = events as f64 / elapsed.as_secs_f64();
    let blocks_per_sec = total_blocks as f64 / elapsed.as_secs_f64();
    let fences_per_sec = fence_samples.len() as f64 / elapsed.as_secs_f64();
    println!(
        "  case sync_mode={} fence_blocks={} kind={} elapsed={:.3}s throughput={:.3} M rec/s {:.3} M blocks/s {:.3} K fences/s",
        sync_case_label(sync_mode),
        fence_blocks,
        fence_kind_label(fence_kind),
        elapsed.as_secs_f64(),
        records_per_sec / 1_000_000.0,
        blocks_per_sec / 1_000_000.0,
        fences_per_sec / 1_000.0,
    );
    print_distribution("publish/block", &publish_samples);
    print_distribution("fence/group", &fence_samples);
    print_distribution("publish+fence/group", &group_samples);
    match recovery_result {
        Some((_recovered, recovery_elapsed)) => {
            println!(
                "    recovery/scan          {}",
                format_duration(recovery_elapsed)
            );
        }
        None => println!("    recovery/scan          skipped"),
    }
    Ok(())
}
