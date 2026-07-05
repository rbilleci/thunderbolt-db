use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, OnceLock};
use std::time::{Duration, Instant};

use gpu_db_write_conveyor::{
    intent_for, recover_wal_segment, stats_for_range, ContiguousCompletionBarrier, DrainStats,
    MappedBlockJournal, MappedJournal, MappedWalSegment, MpscBlockRing, MpscSequencedRing,
    OpenShardAppendStore, PublishError, SpscCursorRing, StagedBlockConveyor, WriteIntent,
};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct CleanupPath<'a> {
    path: &'a Path,
    keep_file: bool,
}

#[derive(Clone, Copy)]
struct FileModeOptions<'a> {
    path: &'a Path,
    keep_file: bool,
    overwrite_file: bool,
    sync: bool,
}

#[derive(Clone, Copy)]
struct WalRunOptions<'a> {
    file: FileModeOptions<'a>,
    publish_mode: WalPublishMode,
    completion_mode: WalCompletionMode,
    latency_samples: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WalPublishMode {
    Generated,
    PayloadSlice,
}

impl WalPublishMode {
    fn label(self, sync: bool, completion: WalCompletionMode) -> &'static str {
        match (self, completion, sync) {
            (Self::Generated, WalCompletionMode::Direct, false) => "file-wal-workers",
            (Self::Generated, WalCompletionMode::Direct, true) => "file-wal-workers-sync",
            (Self::Generated, WalCompletionMode::Visible, false) => "file-wal-visible",
            (Self::Generated, WalCompletionMode::Visible, true) => "file-wal-visible-sync",
            (Self::Generated, WalCompletionMode::Staged, false) => "file-wal-staged",
            (Self::Generated, WalCompletionMode::Staged, true) => "file-wal-staged-sync",
            (Self::Generated, WalCompletionMode::Store, false) => "file-wal-store",
            (Self::Generated, WalCompletionMode::Store, true) => "file-wal-store-sync",
            (Self::PayloadSlice, WalCompletionMode::Direct, false) => "file-wal-payload",
            (Self::PayloadSlice, WalCompletionMode::Direct, true) => "file-wal-payload-sync",
            (Self::PayloadSlice, WalCompletionMode::Visible, false) => "file-wal-payload-visible",
            (Self::PayloadSlice, WalCompletionMode::Visible, true) => {
                "file-wal-payload-visible-sync"
            }
            (Self::PayloadSlice, WalCompletionMode::Staged, false) => "file-wal-payload-staged",
            (Self::PayloadSlice, WalCompletionMode::Staged, true) => "file-wal-payload-staged-sync",
            (Self::PayloadSlice, WalCompletionMode::Store, false) => "file-wal-payload-store",
            (Self::PayloadSlice, WalCompletionMode::Store, true) => "file-wal-payload-store-sync",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WalCompletionMode {
    Direct,
    Visible,
    Staged,
    Store,
}

impl WalCompletionMode {
    fn has_waiter(self) -> bool {
        self != Self::Direct
    }

    fn has_staged_logged_barrier(self) -> bool {
        matches!(self, Self::Staged | Self::Store)
    }
}

const LATENCY_EMPTY: u64 = u64::MAX;

struct LatencySlot {
    publish_start_ns: AtomicU64,
    logged_ns: AtomicU64,
    applied_ns: AtomicU64,
}

impl LatencySlot {
    fn new() -> Self {
        Self {
            publish_start_ns: AtomicU64::new(LATENCY_EMPTY),
            logged_ns: AtomicU64::new(LATENCY_EMPTY),
            applied_ns: AtomicU64::new(LATENCY_EMPTY),
        }
    }
}

struct LatencySampler {
    epoch: OnceLock<Instant>,
    requested: usize,
    stride: u64,
    slots: Vec<LatencySlot>,
}

struct LatencyReport {
    requested: usize,
    stride: u64,
    publish_to_logged: LatencyDistribution,
    logged_to_applied: Option<LatencyDistribution>,
    publish_to_applied: Option<LatencyDistribution>,
    publish_to_final_sync: Option<LatencyDistribution>,
}

struct LatencyDistribution {
    samples: usize,
    percentiles: Percentiles,
}

#[derive(Clone, Copy)]
struct Percentiles {
    p50_ns: u64,
    p90_ns: u64,
    p99_ns: u64,
    max_ns: u64,
}

impl LatencySampler {
    fn new(total_blocks: usize, requested: usize) -> Option<Self> {
        if requested == 0 || total_blocks == 0 {
            return None;
        }
        let requested = requested.min(total_blocks);
        let stride = (total_blocks as u64).div_ceil(requested as u64).max(1);
        let slots = (total_blocks as u64).div_ceil(stride) as usize;
        Some(Self {
            epoch: OnceLock::new(),
            requested,
            stride,
            slots: (0..slots).map(|_| LatencySlot::new()).collect(),
        })
    }

    fn set_epoch(&self, epoch: Instant) {
        self.epoch.set(epoch).expect("latency epoch set once");
    }

    fn now_ns(&self) -> u64 {
        self.epoch
            .get()
            .expect("latency epoch initialized")
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX - 1)) as u64
    }

    fn samples_block(&self, block_id: u64) -> bool {
        self.slot_index(block_id).is_some()
    }

    fn record_logged(&self, block_id: u64, publish_start_ns: u64, logged_ns: u64) {
        if let Some(index) = self.slot_index(block_id) {
            let slot = &self.slots[index];
            slot.publish_start_ns
                .store(publish_start_ns, Ordering::Relaxed);
            slot.logged_ns.store(logged_ns, Ordering::Relaxed);
        }
    }

    fn record_applied(&self, block_id: u64, applied_ns: u64) {
        if let Some(index) = self.slot_index(block_id) {
            self.slots[index]
                .applied_ns
                .store(applied_ns, Ordering::Relaxed);
        }
    }

    fn report(&self, durable_end_ns: Option<u64>) -> Option<LatencyReport> {
        let mut publish_to_logged = Vec::with_capacity(self.slots.len());
        let mut logged_to_applied = Vec::with_capacity(self.slots.len());
        let mut publish_to_applied = Vec::with_capacity(self.slots.len());
        let mut publish_to_final_sync =
            durable_end_ns.map(|_| Vec::with_capacity(self.slots.len()));

        for slot in &self.slots {
            let publish_start_ns = slot.publish_start_ns.load(Ordering::Relaxed);
            let logged_ns = slot.logged_ns.load(Ordering::Relaxed);
            let applied_ns = slot.applied_ns.load(Ordering::Relaxed);
            if publish_start_ns == LATENCY_EMPTY || logged_ns == LATENCY_EMPTY {
                continue;
            }

            if logged_ns >= publish_start_ns {
                publish_to_logged.push(logged_ns - publish_start_ns);
            }
            if applied_ns != LATENCY_EMPTY && applied_ns >= publish_start_ns {
                publish_to_applied.push(applied_ns - publish_start_ns);
                if applied_ns >= logged_ns {
                    logged_to_applied.push(applied_ns - logged_ns);
                }
            }
            if let (Some(durable_end_ns), Some(publish_to_final_sync)) =
                (durable_end_ns, publish_to_final_sync.as_mut())
            {
                if durable_end_ns >= publish_start_ns {
                    publish_to_final_sync.push(durable_end_ns - publish_start_ns);
                }
            }
        }

        Some(LatencyReport {
            requested: self.requested,
            stride: self.stride,
            publish_to_logged: latency_distribution(publish_to_logged)?,
            logged_to_applied: latency_distribution(logged_to_applied),
            publish_to_applied: latency_distribution(publish_to_applied),
            publish_to_final_sync: publish_to_final_sync.and_then(latency_distribution),
        })
    }

    fn slot_index(&self, block_id: u64) -> Option<usize> {
        if !block_id.is_multiple_of(self.stride) {
            return None;
        }
        let index = (block_id / self.stride) as usize;
        (index < self.slots.len()).then_some(index)
    }
}

impl LatencyReport {
    fn print(&self, completion_mode: WalCompletionMode) {
        let completion_label = match completion_mode {
            WalCompletionMode::Direct => "worker",
            WalCompletionMode::Visible => "visible",
            WalCompletionMode::Staged => "applied",
            WalCompletionMode::Store => "store-applied",
        };
        println!(
            "    latency-samples requested={} block-stride={}",
            self.requested, self.stride
        );
        self.publish_to_logged.print("publish->logged");
        if let Some(distribution) = &self.logged_to_applied {
            distribution.print(&format!("logged->{completion_label}"));
        }
        if let Some(distribution) = &self.publish_to_applied {
            distribution.print(&format!("publish->{completion_label}"));
        }
        if let Some(distribution) = &self.publish_to_final_sync {
            distribution.print("publish->final-sync");
        }
    }
}

impl LatencyDistribution {
    fn print(&self, label: &str) {
        println!("    {label:<21} {}", self.format());
    }

    fn format(&self) -> String {
        format!("samples={} {}", self.samples, self.percentiles.format())
    }
}

impl Percentiles {
    fn format(self) -> String {
        format!(
            "p50={} p90={} p99={} max={}",
            format_latency_ns(self.p50_ns),
            format_latency_ns(self.p90_ns),
            format_latency_ns(self.p99_ns),
            format_latency_ns(self.max_ns)
        )
    }
}

fn latency_distribution(mut values: Vec<u64>) -> Option<LatencyDistribution> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(LatencyDistribution {
        samples: values.len(),
        percentiles: Percentiles {
            p50_ns: percentile(&values, 50),
            p90_ns: percentile(&values, 90),
            p99_ns: percentile(&values, 99),
            max_ns: *values.last().expect("non-empty latency values"),
        },
    })
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    debug_assert!(!sorted.is_empty());
    let index = percentile
        .saturating_mul(sorted.len())
        .div_ceil(100)
        .saturating_sub(1);
    sorted[index.min(sorted.len() - 1)]
}

fn format_latency_ns(ns: u64) -> String {
    if ns < 10_000 {
        format!("{:.2}us", ns as f64 / 1_000.0)
    } else if ns < 10_000_000 {
        format!("{:.3}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.3}s", ns as f64 / 1_000_000_000.0)
    }
}

impl Drop for CleanupPath<'_> {
    fn drop(&mut self) {
        if !self.keep_file {
            let _ = std::fs::remove_file(self.path);
        }
    }
}

fn prepare_journal_path<'a>(
    path: &'a Path,
    keep_file: bool,
    remove_existing: bool,
) -> Result<CleanupPath<'a>, Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if remove_existing {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    } else if path.try_exists()? {
        return Err(format!(
            "{} already exists; set CONVEYOR_OVERWRITE=1 to replace it",
            path.display()
        )
        .into());
    }
    Ok(CleanupPath { path, keep_file })
}

fn assert_expected_stats(
    label: &str,
    total: DrainStats,
    events: u64,
) -> Result<(), Box<dyn Error>> {
    let expected = stats_for_range(0, events);
    if total != expected {
        return Err(
            format!("{label} checksum mismatch: got {total:?}, expected {expected:?}").into(),
        );
    }
    Ok(())
}

fn unwrap_journal_arc(
    journal: Arc<MappedBlockJournal>,
) -> Result<MappedBlockJournal, Box<dyn Error>> {
    Arc::try_unwrap(journal)
        .map_err(|_| std::io::Error::other("mapped block journal still shared at sync").into())
}

fn unwrap_wal_arc(segment: Arc<MappedWalSegment>) -> Result<MappedWalSegment, Box<dyn Error>> {
    Arc::try_unwrap(segment)
        .map_err(|_| std::io::Error::other("mapped WAL segment still shared at sync").into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let events: u64 = parse_env("CONVEYOR_EVENTS", 100_000_000_u64);
    let capacity: usize = parse_env("CONVEYOR_CAPACITY", 1_usize << 20);
    let batch: usize = parse_env("CONVEYOR_BATCH", 64_usize).max(1);
    let producers: usize = parse_env(
        "CONVEYOR_PRODUCERS",
        std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1),
    )
    .max(1);
    let workers: usize = parse_env("CONVEYOR_WORKERS", producers).max(1);
    let latency_samples: usize = parse_env("CONVEYOR_LATENCY_SAMPLES", 0_usize);
    let mode = std::env::var("CONVEYOR_MODE").unwrap_or_else(|_| "both".to_string());
    let (file_path, generated_file_path) = match std::env::var("CONVEYOR_FILE") {
        Ok(path) => (PathBuf::from(path), false),
        Err(_) => (
            PathBuf::from(format!(
                "target/write-conveyor-journal-{}.dat",
                std::process::id()
            )),
            true,
        ),
    };
    let keep_file = std::env::var("CONVEYOR_KEEP_FILE").is_ok_and(|v| v == "1");
    let overwrite_file =
        generated_file_path || std::env::var("CONVEYOR_OVERWRITE").is_ok_and(|v| v == "1");
    if events == 0 {
        return Err("CONVEYOR_EVENTS must be greater than zero".into());
    }
    if !capacity.is_power_of_two() || capacity <= 1 {
        return Err("CONVEYOR_CAPACITY must be a power of two greater than one".into());
    }
    if batch > capacity {
        return Err("CONVEYOR_BATCH must not exceed CONVEYOR_CAPACITY".into());
    }
    let block_size = batch.next_power_of_two();
    if !capacity.is_multiple_of(block_size) {
        return Err(
            "CONVEYOR_CAPACITY must be a multiple of CONVEYOR_BATCH.next_power_of_two()".into(),
        );
    }
    if capacity < block_size.saturating_mul(2) {
        return Err("CONVEYOR_CAPACITY must hold at least two CONVEYOR_BATCH-sized blocks".into());
    }

    println!("write-conveyor benchmark");
    println!(
        "  events={events} capacity={capacity} batch={batch} producers={producers} workers={workers}"
    );
    println!(
        "  payload={}B",
        std::mem::size_of::<gpu_db_write_conveyor::WriteIntent>()
    );
    println!("  ring validation=full consumer checksum; append validation=first/last");

    if mode == "both" || mode == "append" {
        run_append_only(events)?;
    }
    if mode == "both" || mode == "spsc" {
        run_spsc(events, capacity, batch)?;
    }
    if mode == "both" || mode == "mpsc" {
        run_mpsc(events, capacity, batch, producers)?;
    }
    if mode == "both" || mode == "block" {
        run_mpsc_block(events, capacity, batch, producers)?;
    }
    if mode == "file-mmap" || mode == "file-mmap-sync" {
        run_file_mmap(
            events,
            FileModeOptions {
                path: &file_path,
                keep_file,
                overwrite_file,
                sync: mode == "file-mmap-sync",
            },
        )?;
    }
    if mode == "file-block" || mode == "file-block-sync" {
        run_file_block(
            events,
            batch.next_power_of_two(),
            producers,
            FileModeOptions {
                path: &file_path,
                keep_file,
                overwrite_file,
                sync: mode == "file-block-sync",
            },
        )?;
    }
    if mode == "file-block-write" || mode == "file-block-write-sync" {
        run_file_block_write_only(
            events,
            batch.next_power_of_two(),
            producers,
            FileModeOptions {
                path: &file_path,
                keep_file,
                overwrite_file,
                sync: mode == "file-block-write-sync",
            },
        )?;
    }
    if mode == "file-block-workers" || mode == "file-block-workers-sync" {
        run_file_block_workers(
            events,
            batch.next_power_of_two(),
            producers,
            workers,
            FileModeOptions {
                path: &file_path,
                keep_file,
                overwrite_file,
                sync: mode == "file-block-workers-sync",
            },
        )?;
    }
    if mode == "file-wal-workers" || mode == "file-wal-workers-sync" {
        run_file_wal_workers(
            events,
            batch.next_power_of_two(),
            producers,
            workers,
            WalRunOptions {
                file: FileModeOptions {
                    path: &file_path,
                    keep_file,
                    overwrite_file,
                    sync: mode == "file-wal-workers-sync",
                },
                publish_mode: WalPublishMode::Generated,
                completion_mode: WalCompletionMode::Direct,
                latency_samples,
            },
        )?;
    }
    if mode == "file-wal-visible-workers" || mode == "file-wal-visible-workers-sync" {
        run_file_wal_workers(
            events,
            batch.next_power_of_two(),
            producers,
            workers,
            WalRunOptions {
                file: FileModeOptions {
                    path: &file_path,
                    keep_file,
                    overwrite_file,
                    sync: mode == "file-wal-visible-workers-sync",
                },
                publish_mode: WalPublishMode::Generated,
                completion_mode: WalCompletionMode::Visible,
                latency_samples,
            },
        )?;
    }
    if mode == "file-wal-staged-workers" || mode == "file-wal-staged-workers-sync" {
        run_file_wal_workers(
            events,
            batch.next_power_of_two(),
            producers,
            workers,
            WalRunOptions {
                file: FileModeOptions {
                    path: &file_path,
                    keep_file,
                    overwrite_file,
                    sync: mode == "file-wal-staged-workers-sync",
                },
                publish_mode: WalPublishMode::Generated,
                completion_mode: WalCompletionMode::Staged,
                latency_samples,
            },
        )?;
    }
    if mode == "file-wal-store-workers" || mode == "file-wal-store-workers-sync" {
        run_file_wal_workers(
            events,
            batch.next_power_of_two(),
            producers,
            workers,
            WalRunOptions {
                file: FileModeOptions {
                    path: &file_path,
                    keep_file,
                    overwrite_file,
                    sync: mode == "file-wal-store-workers-sync",
                },
                publish_mode: WalPublishMode::Generated,
                completion_mode: WalCompletionMode::Store,
                latency_samples,
            },
        )?;
    }
    if mode == "file-wal-payload-workers" || mode == "file-wal-payload-workers-sync" {
        run_file_wal_workers(
            events,
            batch.next_power_of_two(),
            producers,
            workers,
            WalRunOptions {
                file: FileModeOptions {
                    path: &file_path,
                    keep_file,
                    overwrite_file,
                    sync: mode == "file-wal-payload-workers-sync",
                },
                publish_mode: WalPublishMode::PayloadSlice,
                completion_mode: WalCompletionMode::Direct,
                latency_samples,
            },
        )?;
    }
    if mode == "file-wal-payload-visible-workers" || mode == "file-wal-payload-visible-workers-sync"
    {
        run_file_wal_workers(
            events,
            batch.next_power_of_two(),
            producers,
            workers,
            WalRunOptions {
                file: FileModeOptions {
                    path: &file_path,
                    keep_file,
                    overwrite_file,
                    sync: mode == "file-wal-payload-visible-workers-sync",
                },
                publish_mode: WalPublishMode::PayloadSlice,
                completion_mode: WalCompletionMode::Visible,
                latency_samples,
            },
        )?;
    }
    if mode == "file-wal-payload-staged-workers" || mode == "file-wal-payload-staged-workers-sync" {
        run_file_wal_workers(
            events,
            batch.next_power_of_two(),
            producers,
            workers,
            WalRunOptions {
                file: FileModeOptions {
                    path: &file_path,
                    keep_file,
                    overwrite_file,
                    sync: mode == "file-wal-payload-staged-workers-sync",
                },
                publish_mode: WalPublishMode::PayloadSlice,
                completion_mode: WalCompletionMode::Staged,
                latency_samples,
            },
        )?;
    }
    if mode == "file-wal-payload-store-workers" || mode == "file-wal-payload-store-workers-sync" {
        run_file_wal_workers(
            events,
            batch.next_power_of_two(),
            producers,
            workers,
            WalRunOptions {
                file: FileModeOptions {
                    path: &file_path,
                    keep_file,
                    overwrite_file,
                    sync: mode == "file-wal-payload-store-workers-sync",
                },
                publish_mode: WalPublishMode::PayloadSlice,
                completion_mode: WalCompletionMode::Store,
                latency_samples,
            },
        )?;
    }

    Ok(())
}

fn run_append_only(events: u64) -> Result<(), Box<dyn Error>> {
    let len: usize = events.try_into()?;
    let mut log: Box<[std::mem::MaybeUninit<WriteIntent>]> =
        std::iter::repeat_with(std::mem::MaybeUninit::uninit)
            .take(len)
            .collect();
    for slot in log.iter_mut() {
        slot.write(WriteIntent::default());
    }
    let started = Instant::now();
    for (i, slot) in log.iter_mut().enumerate() {
        slot.write(intent_for(i as u64));
    }
    let elapsed = started.elapsed();
    let mut total = DrainStats::default();
    if len != 0 {
        let first = unsafe { log[0].assume_init_read() };
        let last = unsafe { log[len - 1].assume_init_read() };
        total.count = events;
        total.checksum = first.txn_id ^ last.txn_id ^ last.key ^ last.value0;
    }
    std::hint::black_box(&log);
    report("append-only", events, elapsed, total);
    Ok(())
}

fn run_spsc(events: u64, capacity: usize, batch: usize) -> Result<(), Box<dyn Error>> {
    let ring = Arc::new(SpscCursorRing::with_capacity(capacity));
    let barrier = Arc::new(Barrier::new(3));
    let go = Arc::new(AtomicBool::new(false));
    let consumer_ring = Arc::clone(&ring);
    let consumer_barrier = Arc::clone(&barrier);
    let consumer_go = Arc::clone(&go);
    let consumer = std::thread::spawn(move || {
        let mut consumer = consumer_ring.consumer();
        let mut total = DrainStats::default();
        consumer_barrier.wait();
        while !consumer_go.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        while total.count < events {
            let drained = consumer.drain_available(4096);
            if drained.count == 0 {
                std::hint::spin_loop();
            } else {
                total.add(drained);
            }
        }
        total
    });
    let producer_ring = Arc::clone(&ring);
    let producer_barrier = Arc::clone(&barrier);
    let producer_go = Arc::clone(&go);
    let producer = std::thread::spawn(move || {
        let mut producer = producer_ring.producer();
        producer_barrier.wait();
        while !producer_go.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        let mut sent = 0_u64;
        while sent < events {
            let n = (events - sent).min(batch as u64) as usize;
            producer.publish_batch(sent, n);
            sent += n as u64;
        }
    });

    barrier.wait();
    let started = Instant::now();
    go.store(true, Ordering::Release);
    producer.join().unwrap();
    let total = consumer.join().unwrap();
    let elapsed = started.elapsed();
    assert_expected_stats("spsc-cursor", total, events)?;
    report("spsc-cursor", events, elapsed, total);
    Ok(())
}

fn run_mpsc(
    events: u64,
    capacity: usize,
    batch: usize,
    producers: usize,
) -> Result<(), Box<dyn Error>> {
    let ring = Arc::new(MpscSequencedRing::with_capacity(capacity));
    let barrier = Arc::new(Barrier::new(producers + 2));
    let go = Arc::new(AtomicBool::new(false));
    let consumer_ring = Arc::clone(&ring);
    let consumer_barrier = Arc::clone(&barrier);
    let consumer_go = Arc::clone(&go);
    let consumer = std::thread::spawn(move || {
        let mut consumer = consumer_ring.consumer();
        let mut total = DrainStats::default();
        consumer_barrier.wait();
        while !consumer_go.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        while total.count < events {
            let drained = consumer.drain_available(4096);
            if drained.count == 0 {
                std::hint::spin_loop();
            } else {
                total.add(drained);
            }
        }
        total
    });

    let base = events / producers as u64;
    let rem = events % producers as u64;
    let mut handles = Vec::with_capacity(producers);
    for producer_id in 0..producers {
        let ring = Arc::clone(&ring);
        let barrier = Arc::clone(&barrier);
        let go = Arc::clone(&go);
        let count = base + u64::from((producer_id as u64) < rem);
        let first = base * producer_id as u64 + rem.min(producer_id as u64);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            while !go.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            let mut sent = 0_u64;
            while sent < count {
                let n = (count - sent).min(batch as u64) as usize;
                ring.publish_batch(first + sent, n);
                sent += n as u64;
            }
        }));
    }

    barrier.wait();
    let started = Instant::now();
    go.store(true, Ordering::Release);
    for handle in handles {
        handle.join().unwrap();
    }
    let total = consumer.join().unwrap();
    let elapsed = started.elapsed();
    assert_expected_stats("mpsc-sequenced", total, events)?;
    report("mpsc-sequenced", events, elapsed, total);
    Ok(())
}

fn run_mpsc_block(
    events: u64,
    capacity: usize,
    batch: usize,
    producers: usize,
) -> Result<(), Box<dyn Error>> {
    let block_size = batch.next_power_of_two();
    let ring = Arc::new(MpscBlockRing::with_capacity(capacity, block_size));
    let barrier = Arc::new(Barrier::new(producers + 2));
    let go = Arc::new(AtomicBool::new(false));
    let consumer_ring = Arc::clone(&ring);
    let consumer_barrier = Arc::clone(&barrier);
    let consumer_go = Arc::clone(&go);
    let consumer = std::thread::spawn(move || {
        let mut consumer = consumer_ring.consumer();
        let mut total = DrainStats::default();
        consumer_barrier.wait();
        while !consumer_go.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        while total.count < events {
            let drained = consumer.drain_available(256);
            if drained.count == 0 {
                std::hint::spin_loop();
            } else {
                total.add(drained);
            }
        }
        total
    });

    let base = events / producers as u64;
    let rem = events % producers as u64;
    let mut handles = Vec::with_capacity(producers);
    for producer_id in 0..producers {
        let ring = Arc::clone(&ring);
        let barrier = Arc::clone(&barrier);
        let go = Arc::clone(&go);
        let count = base + u64::from((producer_id as u64) < rem);
        let first = base * producer_id as u64 + rem.min(producer_id as u64);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            while !go.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            let mut sent = 0_u64;
            while sent < count {
                let n = (count - sent).min(block_size as u64) as usize;
                ring.publish_block(first + sent, n);
                sent += n as u64;
            }
        }));
    }

    barrier.wait();
    let started = Instant::now();
    go.store(true, Ordering::Release);
    for handle in handles {
        handle.join().unwrap();
    }
    let total = consumer.join().unwrap();
    let elapsed = started.elapsed();
    assert_expected_stats("mpsc-block", total, events)?;
    report("mpsc-block", events, elapsed, total);
    Ok(())
}

fn run_file_mmap(events: u64, opts: FileModeOptions<'_>) -> Result<(), Box<dyn Error>> {
    let _cleanup = prepare_journal_path(opts.path, opts.keep_file, opts.overwrite_file)?;
    let len: usize = events.try_into()?;
    let setup_elapsed;
    let elapsed;
    let write_elapsed;
    let mut sync_elapsed = Duration::ZERO;
    let total;
    {
        let setup_started = Instant::now();
        let mut journal = unsafe { MappedJournal::create(opts.path, len)? };
        setup_elapsed = setup_started.elapsed();
        let started = Instant::now();
        for i in 0..len {
            journal.write(i, intent_for(i as u64));
        }
        write_elapsed = started.elapsed();
        if opts.sync {
            let sync_started = Instant::now();
            journal.sync_mapping()?;
            sync_elapsed = sync_started.elapsed();
        }
        elapsed = started.elapsed();
        let mut observed = DrainStats::default();
        for i in 0..len {
            observed.observe(journal.read(i));
        }
        total = observed;
    }
    assert_expected_stats("file-mmap", total, events)?;
    let extra = if opts.sync {
        format!(
            " setup={:.3}s write={:.3}s sync={:.3}s full-segment-sync",
            setup_elapsed.as_secs_f64(),
            write_elapsed.as_secs_f64(),
            sync_elapsed.as_secs_f64()
        )
    } else {
        format!(
            " setup={:.3}s write={:.3}s",
            setup_elapsed.as_secs_f64(),
            write_elapsed.as_secs_f64()
        )
    };
    report_extra(
        if opts.sync {
            "file-mmap-sync"
        } else {
            "file-mmap"
        },
        events,
        elapsed,
        total,
        &extra,
    );
    Ok(())
}

fn run_file_block(
    events: u64,
    block_size: usize,
    producers: usize,
    opts: FileModeOptions<'_>,
) -> Result<(), Box<dyn Error>> {
    let _cleanup = prepare_journal_path(opts.path, opts.keep_file, opts.overwrite_file)?;
    let base = events / producers as u64;
    let rem = events % producers as u64;
    let total_blocks: usize = (0..producers)
        .map(|producer_id| {
            let count = base + u64::from((producer_id as u64) < rem);
            (count as usize).div_ceil(block_size)
        })
        .sum();
    let mapped_records = total_blocks
        .checked_mul(block_size)
        .ok_or("mapped block journal size overflow")?;
    let setup_elapsed;
    let elapsed;
    let producer_elapsed;
    let drain_elapsed;
    let mut sync_elapsed = Duration::ZERO;
    let total;
    {
        let setup_started = Instant::now();
        let journal =
            Arc::new(unsafe { MappedBlockJournal::create(opts.path, mapped_records, block_size)? });
        setup_elapsed = setup_started.elapsed();
        let barrier = Arc::new(Barrier::new(producers + 2));
        let go = Arc::new(AtomicBool::new(false));
        let consumer_journal = Arc::clone(&journal);
        let consumer_barrier = Arc::clone(&barrier);
        let consumer_go = Arc::clone(&go);
        let consumer = std::thread::spawn(move || {
            let mut consumer = consumer_journal.consumer();
            let mut total = DrainStats::default();
            consumer_barrier.wait();
            while !consumer_go.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            while total.count < events {
                let drained = consumer.drain_available(256);
                if drained.count == 0 {
                    std::hint::spin_loop();
                } else {
                    total.add(drained);
                }
            }
            total
        });
        let mut handles = Vec::with_capacity(producers);
        for producer_id in 0..producers {
            let journal = Arc::clone(&journal);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let count = base + u64::from((producer_id as u64) < rem);
            let first = base * producer_id as u64 + rem.min(producer_id as u64);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                let mut sent = 0_u64;
                while sent < count {
                    let n = (count - sent).min(block_size as u64) as usize;
                    journal.try_publish_block(first + sent, n)?;
                    sent += n as u64;
                }
                Ok::<(), PublishError>(())
            }));
        }
        barrier.wait();
        let started = Instant::now();
        go.store(true, Ordering::Release);
        for handle in handles {
            handle.join().unwrap()?;
        }
        producer_elapsed = started.elapsed();
        total = consumer.join().unwrap();
        drain_elapsed = started.elapsed();
        if opts.sync {
            let sync_started = Instant::now();
            let mut journal = unwrap_journal_arc(journal)?;
            journal.sync_mapping()?;
            sync_elapsed = sync_started.elapsed();
        }
        elapsed = started.elapsed();
    }
    assert_expected_stats("file-block", total, events)?;
    let extra = if opts.sync {
        format!(
            " setup={:.3}s produce={:.3}s drain={:.3}s sync={:.3}s full-segment-sync",
            setup_elapsed.as_secs_f64(),
            producer_elapsed.as_secs_f64(),
            drain_elapsed.as_secs_f64(),
            sync_elapsed.as_secs_f64()
        )
    } else {
        format!(
            " setup={:.3}s produce={:.3}s drain={:.3}s",
            setup_elapsed.as_secs_f64(),
            producer_elapsed.as_secs_f64(),
            drain_elapsed.as_secs_f64()
        )
    };
    report_extra(
        if opts.sync {
            "file-block-sync"
        } else {
            "file-block"
        },
        events,
        elapsed,
        total,
        &extra,
    );
    Ok(())
}

fn run_file_block_write_only(
    events: u64,
    block_size: usize,
    producers: usize,
    opts: FileModeOptions<'_>,
) -> Result<(), Box<dyn Error>> {
    let _cleanup = prepare_journal_path(opts.path, opts.keep_file, opts.overwrite_file)?;
    let base = events / producers as u64;
    let rem = events % producers as u64;
    let total_blocks: usize = (0..producers)
        .map(|producer_id| {
            let count = base + u64::from((producer_id as u64) < rem);
            (count as usize).div_ceil(block_size)
        })
        .sum();
    let mapped_records = total_blocks
        .checked_mul(block_size)
        .ok_or("mapped block journal size overflow")?;
    let setup_elapsed;
    let elapsed;
    let producer_elapsed;
    let mut sync_elapsed = Duration::ZERO;
    {
        let setup_started = Instant::now();
        let journal =
            Arc::new(unsafe { MappedBlockJournal::create(opts.path, mapped_records, block_size)? });
        setup_elapsed = setup_started.elapsed();
        debug_assert_eq!(journal.block_capacity(), total_blocks as u64);
        let barrier = Arc::new(Barrier::new(producers + 1));
        let go = Arc::new(AtomicBool::new(false));
        let mut producer_handles = Vec::with_capacity(producers);
        for producer_id in 0..producers {
            let journal = Arc::clone(&journal);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let count = base + u64::from((producer_id as u64) < rem);
            let first = base * producer_id as u64 + rem.min(producer_id as u64);
            producer_handles.push(std::thread::spawn(move || {
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                let mut sent = 0_u64;
                while sent < count {
                    let n = (count - sent).min(block_size as u64) as usize;
                    journal.try_publish_block(first + sent, n)?;
                    sent += n as u64;
                }
                Ok::<(), PublishError>(())
            }));
        }

        barrier.wait();
        let started = Instant::now();
        go.store(true, Ordering::Release);
        for handle in producer_handles {
            handle.join().unwrap()?;
        }
        producer_elapsed = started.elapsed();
        if opts.sync {
            let sync_started = Instant::now();
            let mut journal = unwrap_journal_arc(journal)?;
            journal.sync_mapping()?;
            sync_elapsed = sync_started.elapsed();
        }
        elapsed = started.elapsed();
    }
    let total = DrainStats {
        count: events,
        checksum: 0,
    };
    let extra = if opts.sync {
        format!(
            " setup={:.3}s produce={:.3}s sync={:.3}s full-segment-sync validation=none",
            setup_elapsed.as_secs_f64(),
            producer_elapsed.as_secs_f64(),
            sync_elapsed.as_secs_f64()
        )
    } else {
        format!(
            " setup={:.3}s produce={:.3}s validation=none",
            setup_elapsed.as_secs_f64(),
            producer_elapsed.as_secs_f64()
        )
    };
    report_extra(
        if opts.sync {
            "file-block-write-sync"
        } else {
            "file-block-write"
        },
        events,
        elapsed,
        total,
        &extra,
    );
    Ok(())
}

fn run_file_block_workers(
    events: u64,
    block_size: usize,
    producers: usize,
    workers: usize,
    opts: FileModeOptions<'_>,
) -> Result<(), Box<dyn Error>> {
    let _cleanup = prepare_journal_path(opts.path, opts.keep_file, opts.overwrite_file)?;
    let base = events / producers as u64;
    let rem = events % producers as u64;
    let total_blocks: usize = (0..producers)
        .map(|producer_id| {
            let count = base + u64::from((producer_id as u64) < rem);
            (count as usize).div_ceil(block_size)
        })
        .sum();
    let mapped_records = total_blocks
        .checked_mul(block_size)
        .ok_or("mapped block journal size overflow")?;
    let setup_elapsed;
    let elapsed;
    let producer_elapsed;
    let worker_elapsed;
    let mut sync_elapsed = Duration::ZERO;
    let total;
    {
        let setup_started = Instant::now();
        let journal =
            Arc::new(unsafe { MappedBlockJournal::create(opts.path, mapped_records, block_size)? });
        setup_elapsed = setup_started.elapsed();
        debug_assert_eq!(journal.block_capacity(), total_blocks as u64);
        let barrier = Arc::new(Barrier::new(producers + workers + 1));
        let go = Arc::new(AtomicBool::new(false));
        let next_block = Arc::new(AtomicU64::new(0));
        let mut worker_handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let journal = Arc::clone(&journal);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let next_block = Arc::clone(&next_block);
            worker_handles.push(std::thread::spawn(move || {
                let mut total = DrainStats::default();
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                loop {
                    let block_id = next_block.fetch_add(1, Ordering::Relaxed);
                    if block_id >= total_blocks as u64 {
                        break;
                    }
                    loop {
                        if let Some(drained) = journal.read_published_block(block_id) {
                            total.add(drained);
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
                total
            }));
        }
        let mut producer_handles = Vec::with_capacity(producers);
        for producer_id in 0..producers {
            let journal = Arc::clone(&journal);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let count = base + u64::from((producer_id as u64) < rem);
            let first = base * producer_id as u64 + rem.min(producer_id as u64);
            producer_handles.push(std::thread::spawn(move || {
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                let mut sent = 0_u64;
                while sent < count {
                    let n = (count - sent).min(block_size as u64) as usize;
                    journal.try_publish_block(first + sent, n)?;
                    sent += n as u64;
                }
                Ok::<(), PublishError>(())
            }));
        }

        barrier.wait();
        let started = Instant::now();
        go.store(true, Ordering::Release);
        for handle in producer_handles {
            handle.join().unwrap()?;
        }
        producer_elapsed = started.elapsed();
        let mut combined = DrainStats::default();
        for handle in worker_handles {
            combined.add(handle.join().unwrap());
        }
        worker_elapsed = started.elapsed();
        total = combined;
        if opts.sync {
            let sync_started = Instant::now();
            let mut journal = unwrap_journal_arc(journal)?;
            journal.sync_mapping()?;
            sync_elapsed = sync_started.elapsed();
        }
        elapsed = started.elapsed();
    }
    assert_expected_stats("file-workers", total, events)?;
    let extra = if opts.sync {
        format!(
            " setup={:.3}s produce={:.3}s workers={:.3}s sync={:.3}s full-segment-sync",
            setup_elapsed.as_secs_f64(),
            producer_elapsed.as_secs_f64(),
            worker_elapsed.as_secs_f64(),
            sync_elapsed.as_secs_f64()
        )
    } else {
        format!(
            " setup={:.3}s produce={:.3}s workers={:.3}s",
            setup_elapsed.as_secs_f64(),
            producer_elapsed.as_secs_f64(),
            worker_elapsed.as_secs_f64()
        )
    };
    report_extra(
        if opts.sync {
            "file-workers-sync"
        } else {
            "file-workers"
        },
        events,
        elapsed,
        total,
        &extra,
    );
    Ok(())
}

fn run_file_wal_workers(
    events: u64,
    block_size: usize,
    producers: usize,
    workers: usize,
    opts: WalRunOptions<'_>,
) -> Result<(), Box<dyn Error>> {
    let file_opts = opts.file;
    let publish_mode = opts.publish_mode;
    let completion_mode = opts.completion_mode;
    let _cleanup = prepare_journal_path(
        file_opts.path,
        file_opts.keep_file,
        file_opts.overwrite_file,
    )?;
    let base = events / producers as u64;
    let rem = events % producers as u64;
    let total_blocks: usize = (0..producers)
        .map(|producer_id| {
            let count = base + u64::from((producer_id as u64) < rem);
            (count as usize).div_ceil(block_size)
        })
        .sum();
    let latency = LatencySampler::new(total_blocks, opts.latency_samples).map(Arc::new);
    let mapped_records = total_blocks
        .checked_mul(block_size)
        .ok_or("mapped WAL segment size overflow")?;
    let setup_elapsed;
    let elapsed;
    let producer_elapsed;
    let worker_elapsed;
    let visibility_elapsed;
    let mut sync_elapsed = Duration::ZERO;
    let mut recovery_elapsed = Duration::ZERO;
    let mut store_validation_elapsed = Duration::ZERO;
    let mut durable_end_ns = None;
    let total;
    {
        let setup_started = Instant::now();
        let segment = Arc::new(unsafe {
            MappedWalSegment::create(file_opts.path, 1, mapped_records, block_size)?
        });
        setup_elapsed = setup_started.elapsed();
        debug_assert_eq!(segment.block_capacity(), total_blocks as u64);
        let completion_waiters = usize::from(completion_mode.has_waiter());
        let barrier = Arc::new(Barrier::new(producers + workers + completion_waiters + 1));
        let go = Arc::new(AtomicBool::new(false));
        let next_block = Arc::new(AtomicU64::new(0));
        let completion = (completion_mode == WalCompletionMode::Visible)
            .then(|| Arc::new(ContiguousCompletionBarrier::with_capacity(total_blocks)));
        let staged = completion_mode
            .has_staged_logged_barrier()
            .then(|| Arc::new(StagedBlockConveyor::with_capacity(total_blocks)));
        let store = (completion_mode == WalCompletionMode::Store).then(|| {
            Arc::new(OpenShardAppendStore::with_capacity(
                total_blocks,
                block_size,
            ))
        });
        let visibility_handle = match completion_mode {
            WalCompletionMode::Direct => None,
            WalCompletionMode::Visible => {
                let completion = Arc::clone(completion.as_ref().expect("visible barrier"));
                let barrier = Arc::clone(&barrier);
                let go = Arc::clone(&go);
                Some(std::thread::spawn(move || {
                    barrier.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    completion.wait_for(total_blocks as u64).unwrap();
                }))
            }
            WalCompletionMode::Staged | WalCompletionMode::Store => {
                let staged = Arc::clone(staged.as_ref().expect("staged conveyor"));
                let barrier = Arc::clone(&barrier);
                let go = Arc::clone(&go);
                Some(std::thread::spawn(move || {
                    barrier.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    staged.wait_applied(total_blocks as u64).unwrap();
                }))
            }
        };
        let mut worker_handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let segment = Arc::clone(&segment);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let next_block = Arc::clone(&next_block);
            let completion = completion.as_ref().map(Arc::clone);
            let staged = staged.as_ref().map(Arc::clone);
            let store = store.as_ref().map(Arc::clone);
            let latency = latency.as_ref().map(Arc::clone);
            worker_handles.push(std::thread::spawn(move || {
                let mut total = DrainStats::default();
                let mut payload = if store.is_some() {
                    Vec::with_capacity(block_size)
                } else {
                    Vec::new()
                };
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                loop {
                    let block_id = next_block.fetch_add(1, Ordering::Relaxed);
                    if block_id >= total_blocks as u64 {
                        break;
                    }
                    if let Some(staged) = staged.as_ref() {
                        staged.wait_logged(block_id + 1).unwrap();
                    }
                    loop {
                        if let Some(store) = store.as_ref() {
                            if segment
                                .read_published_block_into(block_id, &mut payload)
                                .is_some()
                            {
                                total.add(store.apply_block(block_id, &payload).unwrap());
                                if let Some(staged) = staged.as_ref() {
                                    staged.mark_applied(block_id).unwrap();
                                }
                                if let Some(latency) = latency.as_ref() {
                                    if latency.samples_block(block_id) {
                                        latency.record_applied(block_id, latency.now_ns());
                                    }
                                }
                                break;
                            }
                        } else if let Some(drained) = segment.read_published_block(block_id) {
                            total.add(drained);
                            if let Some(completion) = completion.as_ref() {
                                completion.complete(block_id).unwrap();
                            }
                            if let Some(staged) = staged.as_ref() {
                                staged.mark_applied(block_id).unwrap();
                            }
                            if let Some(latency) = latency.as_ref() {
                                if latency.samples_block(block_id) {
                                    latency.record_applied(block_id, latency.now_ns());
                                }
                            }
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
                total
            }));
        }

        let mut producer_handles = Vec::with_capacity(producers);
        for producer_id in 0..producers {
            let segment = Arc::clone(&segment);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let staged = staged.as_ref().map(Arc::clone);
            let latency = latency.as_ref().map(Arc::clone);
            let count = base + u64::from((producer_id as u64) < rem);
            let first = base * producer_id as u64 + rem.min(producer_id as u64);
            producer_handles.push(std::thread::spawn(move || {
                let mut scratch = if publish_mode == WalPublishMode::PayloadSlice {
                    vec![WriteIntent::default(); block_size]
                } else {
                    Vec::new()
                };
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                let mut sent = 0_u64;
                while sent < count {
                    let n = (count - sent).min(block_size as u64) as usize;
                    let (block_id, publish_start_ns) = match publish_mode {
                        WalPublishMode::Generated => {
                            let publish_start_ns = latency.as_ref().map(|latency| latency.now_ns());
                            let block_id = segment
                                .try_publish_block_position(first + sent, n)?
                                .expect("non-empty benchmark block");
                            (block_id, publish_start_ns)
                        }
                        WalPublishMode::PayloadSlice => {
                            for (offset, slot) in scratch[..n].iter_mut().enumerate() {
                                *slot = intent_for(first + sent + offset as u64);
                            }
                            let publish_start_ns = latency.as_ref().map(|latency| latency.now_ns());
                            let block_id = segment
                                .try_publish_intents_position(&scratch[..n])?
                                .expect("non-empty benchmark block");
                            (block_id, publish_start_ns)
                        }
                    };
                    if let (Some(latency), Some(publish_start_ns)) =
                        (latency.as_ref(), publish_start_ns)
                    {
                        if latency.samples_block(block_id) {
                            latency.record_logged(block_id, publish_start_ns, latency.now_ns());
                        }
                    }
                    if let Some(staged) = staged.as_ref() {
                        staged.mark_logged(block_id).unwrap();
                    }
                    sent += n as u64;
                }
                Ok::<(), PublishError>(())
            }));
        }

        barrier.wait();
        let started = Instant::now();
        if let Some(latency) = latency.as_ref() {
            latency.set_epoch(started);
        }
        go.store(true, Ordering::Release);
        for handle in producer_handles {
            handle.join().unwrap()?;
        }
        producer_elapsed = started.elapsed();
        let mut combined = DrainStats::default();
        for handle in worker_handles {
            combined.add(handle.join().unwrap());
        }
        worker_elapsed = started.elapsed();
        if let Some(handle) = visibility_handle {
            handle.join().unwrap();
        }
        visibility_elapsed = started.elapsed();
        total = combined;
        if file_opts.sync {
            let sync_started = Instant::now();
            let mut segment = unwrap_wal_arc(segment)?;
            segment.sync_published_prefix(total_blocks as u64)?;
            sync_elapsed = sync_started.elapsed();
            durable_end_ns = latency.as_ref().map(|latency| latency.now_ns());
        }
        elapsed = started.elapsed();
        if let Some(store) = store.as_ref() {
            let validate_started = Instant::now();
            let observed = store
                .validate_applied_blocks(total_blocks as u64)
                .ok_or("store validation could not read applied prefix")?;
            let expected = stats_for_range(0, events);
            if observed != expected {
                return Err(format!(
                    "store validation mismatch: got {observed:?}, expected {expected:?}"
                )
                .into());
            }
            store_validation_elapsed = validate_started.elapsed();
        }
    }

    assert_expected_stats(
        publish_mode.label(file_opts.sync, completion_mode),
        total,
        events,
    )?;
    if file_opts.sync {
        let recovery_started = Instant::now();
        let recovered = recover_wal_segment(file_opts.path)?;
        recovery_elapsed = recovery_started.elapsed();
        let expected = stats_for_range(0, events);
        if recovered.recovered_blocks != total_blocks as u64
            || recovered.recovered_records != events
            || recovered.stats != expected
        {
            return Err(format!(
                "WAL recovery mismatch: recovered {recovered:?}, expected_blocks={total_blocks}, expected_stats={expected:?}"
            )
            .into());
        }
    }

    let wait_label = match completion_mode {
        WalCompletionMode::Direct => "",
        WalCompletionMode::Visible => "visible",
        WalCompletionMode::Staged | WalCompletionMode::Store => "applied",
    };
    let store_validation = if completion_mode == WalCompletionMode::Store {
        format!(
            " store-validate={:.3}s",
            store_validation_elapsed.as_secs_f64()
        )
    } else {
        String::new()
    };
    let extra = match (file_opts.sync, completion_mode.has_waiter()) {
        (true, true) => format!(
            " setup={:.3}s produce={:.3}s workers={:.3}s {wait_label}={:.3}s sync={:.3}s recover={:.3}s range-sync{store_validation}",
            setup_elapsed.as_secs_f64(),
            producer_elapsed.as_secs_f64(),
            worker_elapsed.as_secs_f64(),
            visibility_elapsed.as_secs_f64(),
            sync_elapsed.as_secs_f64(),
            recovery_elapsed.as_secs_f64()
        ),
        (true, false) => format!(
            " setup={:.3}s produce={:.3}s workers={:.3}s sync={:.3}s recover={:.3}s range-sync{store_validation}",
            setup_elapsed.as_secs_f64(),
            producer_elapsed.as_secs_f64(),
            worker_elapsed.as_secs_f64(),
            sync_elapsed.as_secs_f64(),
            recovery_elapsed.as_secs_f64()
        ),
        (false, true) => format!(
            " setup={:.3}s produce={:.3}s workers={:.3}s {wait_label}={:.3}s{store_validation}",
            setup_elapsed.as_secs_f64(),
            producer_elapsed.as_secs_f64(),
            worker_elapsed.as_secs_f64(),
            visibility_elapsed.as_secs_f64()
        ),
        (false, false) => format!(
            " setup={:.3}s produce={:.3}s workers={:.3}s{store_validation}",
            setup_elapsed.as_secs_f64(),
            producer_elapsed.as_secs_f64(),
            worker_elapsed.as_secs_f64()
        ),
    };
    report_extra(
        publish_mode.label(file_opts.sync, completion_mode),
        events,
        elapsed,
        total,
        &extra,
    );
    if let Some(report) = latency
        .as_ref()
        .and_then(|latency| latency.report(durable_end_ns))
    {
        report.print(completion_mode);
    }
    Ok(())
}

fn report(name: &str, events: u64, elapsed: Duration, total: DrainStats) {
    report_extra(name, events, elapsed, total, "");
}

fn report_extra(name: &str, events: u64, elapsed: Duration, total: DrainStats, extra: &str) {
    let secs = elapsed.as_secs_f64();
    let eps = events as f64 / secs;
    println!(
        "  {name:14} {:>12.3} M/s  {:>7.2} ns/write  elapsed={:.3}s checksum={:#016x}{extra}",
        eps / 1e6,
        secs * 1e9 / events as f64,
        secs,
        total.checksum
    );
}
