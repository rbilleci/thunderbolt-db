use std::cell::UnsafeCell;
use std::error::Error;
use std::ffi::OsString;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use std::time::{Duration, Instant};

use gpu_db_write_conveyor::{
    intent_for, recover_wal_manager_by_scan, recover_wal_segment, recover_wal_segment_by_scan,
    stats_for_range, ContiguousCompletionBarrier, DrainStats, MappedBlockJournal, MappedJournal,
    MappedWalSegment, MpscBlockRing, MpscSequencedRing, OpenShardAppendStore, PublishError,
    SpscCursorRing, StagedBlockConveyor, WalDataSyncMode, WalPublishedBlock, WalSegmentManager,
    WalSegmentManagerConfig, WriteIntent,
};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_durable_sync_mode() -> Result<WalDataSyncMode, Box<dyn Error>> {
    match std::env::var("CONVEYOR_DURABLE_SYNC_MODE")
        .unwrap_or_else(|_| "write-and-file-data".to_string())
        .as_str()
    {
        "range-and-file-data" | "strict" => Ok(WalDataSyncMode::RangeAndFileData),
        "write-and-file-data" | "pwrite-and-file-data" | "pwrite-fdatasync" => {
            Ok(WalDataSyncMode::WriteAndFileData)
        }
        "prewrite-and-file-data" | "async-write-and-file-data" | "prewrite-fdatasync" => {
            Ok(WalDataSyncMode::PrewriteAndFileData)
        }
        "sync-write-data" | "pwritev2-dsync" | "rwf-dsync" => parse_sync_write_mode(),
        "file-data-only" | "fdatasync" => Ok(WalDataSyncMode::FileDataOnly),
        value => Err(format!(
            "unsupported CONVEYOR_DURABLE_SYNC_MODE={value}; use range-and-file-data, write-and-file-data, prewrite-and-file-data, sync-write-data, or file-data-only"
        )
        .into()),
    }
}

#[cfg(target_os = "linux")]
fn parse_sync_write_mode() -> Result<WalDataSyncMode, Box<dyn Error>> {
    Ok(WalDataSyncMode::SyncWriteData)
}

#[cfg(not(target_os = "linux"))]
fn parse_sync_write_mode() -> Result<WalDataSyncMode, Box<dyn Error>> {
    Err("CONVEYOR_DURABLE_SYNC_MODE=sync-write-data requires Linux pwritev2(RWF_DSYNC)".into())
}

struct CleanupPath<'a> {
    path: &'a Path,
    keep_file: bool,
}

struct CleanupDir {
    path: PathBuf,
    keep: bool,
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

#[derive(Clone, Copy)]
struct ClientRunOptions<'a> {
    file: FileModeOptions<'a>,
    ack_mode: ClientAckMode,
    latency_samples: usize,
    wait_spins: u32,
    client_driver_threads: usize,
    client_driver_issue_budget: usize,
    durable_group_us: u64,
    durable_min_blocks: usize,
    durable_lanes: usize,
    durable_sync_mode: WalDataSyncMode,
    background_durable: bool,
    stage_timings: bool,
    append_group_us: u64,
    min_records_per_block: usize,
    append_ring_capacity: usize,
    manager_records_per_segment: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CoalescedWalBackend {
    Segment,
    Manager,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ClientAckMode {
    Logged,
    StoreApplied,
    DurableStoreApplied,
}

impl ClientAckMode {
    fn label(self) -> &'static str {
        match self {
            Self::Logged => "file-wal-client-logged",
            Self::StoreApplied => "file-wal-client-store-applied",
            Self::DurableStoreApplied => "file-wal-client-durable",
        }
    }

    fn needs_store(self) -> bool {
        matches!(self, Self::StoreApplied | Self::DurableStoreApplied)
    }

    fn needs_durable(self) -> bool {
        self == Self::DurableStoreApplied
    }
}

fn is_direct_client_mode(mode: &str) -> bool {
    matches!(
        mode,
        "file-wal-client-logged" | "file-wal-client-store-applied" | "file-wal-client-durable"
    )
}

fn is_coalesced_client_mode(mode: &str) -> bool {
    matches!(
        mode,
        "file-wal-client-coalesced-logged"
            | "file-wal-client-coalesced-store-applied"
            | "file-wal-client-coalesced-durable"
            | "file-wal-manager-coalesced-logged"
            | "file-wal-manager-coalesced-store-applied"
            | "file-wal-manager-coalesced-durable"
    )
}

fn is_client_latency_mode(mode: &str) -> bool {
    is_direct_client_mode(mode) || is_coalesced_client_mode(mode)
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

#[derive(Default)]
struct ClientLatencySamples {
    client_to_logged: Vec<u64>,
    logged_to_ack: Vec<u64>,
    store_to_ack: Vec<u64>,
    client_to_ack: Vec<u64>,
}

struct ClientLatencyReport {
    requested: usize,
    stride: u64,
    client_to_logged: Option<LatencyDistribution>,
    logged_to_ack: Option<LatencyDistribution>,
    store_to_ack: Option<LatencyDistribution>,
    client_to_ack: LatencyDistribution,
}

#[derive(Default)]
struct StageSamples {
    samples_ns: Mutex<Vec<u64>>,
}

#[derive(Default)]
struct CoalescedStageCounters {
    append_batch_wait: StageSamples,
    append_wal_publish: StageSamples,
    append_client_notify: StageSamples,
    store_wait_for_block: StageSamples,
    store_read_block: StageSamples,
    store_apply_block: StageSamples,
    store_complete_block: StageSamples,
}

const BLOCK_TIMING_EMPTY: u64 = u64::MAX;

struct BlockTimingSlot {
    logged_ns: AtomicU64,
    store_applied_ns: AtomicU64,
    durable_ns: AtomicU64,
}

struct BlockStageTimeline {
    epoch: OnceLock<Instant>,
    slots: Vec<BlockTimingSlot>,
}

struct BlockStageReport {
    blocks: usize,
    logged_samples: usize,
    store_before_logged: usize,
    durable_before_logged: usize,
    durable_before_store: usize,
    logged_to_store: Option<LatencyDistribution>,
    logged_to_durable: Option<LatencyDistribution>,
    store_to_durable: Option<LatencyDistribution>,
}

#[derive(Clone, Copy)]
struct Percentiles {
    p50_ns: u64,
    p90_ns: u64,
    p99_ns: u64,
    max_ns: u64,
}

#[derive(Clone, Copy)]
enum DurableFlushReason {
    Pressure,
    Deadline,
    Final,
}

#[derive(Default)]
struct DurableSyncCounters {
    sample_timings: bool,
    sync_calls: AtomicU64,
    sync_frontier_calls: AtomicU64,
    synced_blocks: AtomicU64,
    sync_ns: AtomicU64,
    max_sync_ns: AtomicU64,
    wait_ns: AtomicU64,
    max_wait_ns: AtomicU64,
    pressure_flushes: AtomicU64,
    deadline_flushes: AtomicU64,
    final_flushes: AtomicU64,
    wait_samples_ns: Mutex<Vec<u64>>,
    sync_samples_ns: Mutex<Vec<u64>>,
}

#[derive(Clone, Copy, Default)]
struct DurableSyncSnapshot {
    sync_calls: u64,
    sync_frontier_calls: u64,
    synced_blocks: u64,
    sync_ns: u64,
    max_sync_ns: u64,
    wait_ns: u64,
    max_wait_ns: u64,
    pressure_flushes: u64,
    deadline_flushes: u64,
    final_flushes: u64,
}

struct DurableSyncTimingReport {
    wait: Option<LatencyDistribution>,
    sync: Option<LatencyDistribution>,
}

#[derive(Clone, Copy, Default)]
struct ChronicleCutSnapshot {
    published: u64,
    store_applied: Option<u64>,
    requested_durable: Option<u64>,
    durable: Option<u64>,
}

#[derive(Default)]
struct WalPrewriteCounters {
    write_calls: AtomicU64,
    written_blocks: AtomicU64,
    write_ns: AtomicU64,
    max_write_ns: AtomicU64,
}

#[derive(Clone, Copy, Default)]
struct WalPrewriteSnapshot {
    write_calls: u64,
    written_blocks: u64,
    write_ns: u64,
    max_write_ns: u64,
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

impl ClientLatencySamples {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            client_to_logged: Vec::with_capacity(capacity),
            logged_to_ack: Vec::with_capacity(capacity),
            store_to_ack: Vec::with_capacity(capacity),
            client_to_ack: Vec::with_capacity(capacity),
        }
    }

    fn append(&mut self, mut other: Self) {
        self.client_to_logged.append(&mut other.client_to_logged);
        self.logged_to_ack.append(&mut other.logged_to_ack);
        self.store_to_ack.append(&mut other.store_to_ack);
        self.client_to_ack.append(&mut other.client_to_ack);
    }

    fn report(self, requested: usize, stride: u64) -> Option<ClientLatencyReport> {
        Some(ClientLatencyReport {
            requested,
            stride,
            client_to_logged: latency_distribution(self.client_to_logged),
            logged_to_ack: latency_distribution(self.logged_to_ack),
            store_to_ack: latency_distribution(self.store_to_ack),
            client_to_ack: latency_distribution(self.client_to_ack)?,
        })
    }
}

impl ClientLatencyReport {
    fn print(&self, ack_mode: ClientAckMode) {
        let ack_label = match ack_mode {
            ClientAckMode::Logged => "logged",
            ClientAckMode::StoreApplied => "store-applied",
            ClientAckMode::DurableStoreApplied => "data-fenced-wal+volatile-store-applied",
        };
        println!(
            "    client-latency requested={} sample-stride={}",
            self.requested, self.stride
        );
        if let Some(distribution) = &self.client_to_logged {
            distribution.print("client->logged");
        }
        if let Some(distribution) = &self.logged_to_ack {
            distribution.print(&format!("logged->{ack_label}"));
        }
        if let Some(distribution) = &self.store_to_ack {
            distribution.print("store-applied->wal-data-fenced");
        }
        if ack_mode != ClientAckMode::Logged {
            self.client_to_ack.print(&format!("client->{ack_label}"));
        }
    }
}

impl StageSamples {
    fn record(&self, duration: Duration) {
        self.samples_ns
            .lock()
            .expect("stage samples mutex poisoned")
            .push(duration_ns(duration));
    }

    fn distribution(&self) -> Option<LatencyDistribution> {
        latency_distribution(
            self.samples_ns
                .lock()
                .expect("stage samples mutex poisoned")
                .clone(),
        )
    }
}

impl CoalescedStageCounters {
    fn print(&self) {
        println!("    stage-timing");
        self.print_stage("append batch-wait", &self.append_batch_wait);
        self.print_stage("append wal-publish", &self.append_wal_publish);
        self.print_stage("append notify", &self.append_client_notify);
        self.print_stage("store wait-block", &self.store_wait_for_block);
        self.print_stage("store read-block", &self.store_read_block);
        self.print_stage("store apply-block", &self.store_apply_block);
        self.print_stage("store complete", &self.store_complete_block);
    }

    fn print_stage(&self, label: &str, samples: &StageSamples) {
        if let Some(distribution) = samples.distribution() {
            distribution.print(label);
        }
    }
}

impl BlockStageTimeline {
    fn with_capacity(blocks: usize) -> Self {
        Self {
            epoch: OnceLock::new(),
            slots: (0..blocks)
                .map(|_| BlockTimingSlot {
                    logged_ns: AtomicU64::new(BLOCK_TIMING_EMPTY),
                    store_applied_ns: AtomicU64::new(BLOCK_TIMING_EMPTY),
                    durable_ns: AtomicU64::new(BLOCK_TIMING_EMPTY),
                })
                .collect(),
        }
    }

    fn set_epoch(&self, epoch: Instant) {
        self.epoch.set(epoch).expect("block timing epoch set once");
    }

    fn now_ns(&self) -> u64 {
        self.epoch
            .get()
            .expect("block timing epoch initialized")
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX - 1)) as u64
    }

    fn record_logged_ns(&self, block_id: u64, logged_ns: u64) {
        if let Some(slot) = self.slots.get(block_id as usize) {
            slot.logged_ns.store(logged_ns, Ordering::Release);
        }
    }

    fn record_store_applied(&self, block_id: u64) {
        if let Some(slot) = self.slots.get(block_id as usize) {
            slot.store_applied_ns
                .store(self.now_ns(), Ordering::Release);
        }
    }

    fn record_durable(&self, block_id: u64) {
        if let Some(slot) = self.slots.get(block_id as usize) {
            slot.durable_ns.store(self.now_ns(), Ordering::Release);
        }
    }

    fn report(&self, blocks: u64) -> Option<BlockStageReport> {
        let limit = (blocks as usize).min(self.slots.len());
        let mut logged_to_store = Vec::with_capacity(limit);
        let mut logged_to_durable = Vec::with_capacity(limit);
        let mut store_to_durable = Vec::with_capacity(limit);
        let mut logged_samples = 0_usize;
        let mut store_before_logged = 0_usize;
        let mut durable_before_logged = 0_usize;
        let mut durable_before_store = 0_usize;
        for slot in self.slots.iter().take(limit) {
            let logged_ns = slot.logged_ns.load(Ordering::Acquire);
            if logged_ns == BLOCK_TIMING_EMPTY {
                continue;
            }
            logged_samples += 1;
            let store_applied_ns = slot.store_applied_ns.load(Ordering::Acquire);
            if store_applied_ns != BLOCK_TIMING_EMPTY && store_applied_ns >= logged_ns {
                logged_to_store.push(store_applied_ns - logged_ns);
            } else if store_applied_ns != BLOCK_TIMING_EMPTY {
                store_before_logged += 1;
            }
            let durable_ns = slot.durable_ns.load(Ordering::Acquire);
            if durable_ns != BLOCK_TIMING_EMPTY && durable_ns >= logged_ns {
                logged_to_durable.push(durable_ns - logged_ns);
                if store_applied_ns != BLOCK_TIMING_EMPTY && durable_ns >= store_applied_ns {
                    store_to_durable.push(durable_ns - store_applied_ns);
                } else if store_applied_ns != BLOCK_TIMING_EMPTY {
                    durable_before_store += 1;
                }
            } else if durable_ns != BLOCK_TIMING_EMPTY {
                durable_before_logged += 1;
            }
        }
        Some(BlockStageReport {
            blocks: limit,
            logged_samples,
            store_before_logged,
            durable_before_logged,
            durable_before_store,
            logged_to_store: latency_distribution(logged_to_store),
            logged_to_durable: latency_distribution(logged_to_durable),
            store_to_durable: latency_distribution(store_to_durable),
        })
    }
}

impl BlockStageReport {
    fn print(&self) {
        if self.logged_to_store.is_none()
            && self.logged_to_durable.is_none()
            && self.store_to_durable.is_none()
        {
            return;
        }
        println!(
            "    block-timeline blocks={} publish-samples={} store-before-publish={} durable-before-publish={} durable-before-store={}",
            self.blocks,
            self.logged_samples,
            self.store_before_logged,
            self.durable_before_logged,
            self.durable_before_store
        );
        if let Some(distribution) = &self.logged_to_store {
            distribution.print("block publish->store");
        }
        if let Some(distribution) = &self.logged_to_durable {
            distribution.print("block publish->durable");
        }
        if let Some(distribution) = &self.store_to_durable {
            distribution.print("store->durable slack");
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

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn durable_sync_mode_label(mode: WalDataSyncMode) -> &'static str {
    match mode {
        WalDataSyncMode::RangeAndFileData => "range-and-file-data",
        WalDataSyncMode::WriteAndFileData => "write-and-file-data",
        WalDataSyncMode::PrewriteAndFileData => "prewrite-and-file-data",
        WalDataSyncMode::SyncWriteData => "sync-write-data",
        WalDataSyncMode::FileDataOnly => "file-data-only",
    }
}

impl DurableSyncCounters {
    fn new(sample_timings: bool) -> Self {
        Self {
            sample_timings,
            ..Self::default()
        }
    }

    fn record_wait(&self, duration: Duration) {
        let ns = duration_ns(duration);
        self.wait_ns.fetch_add(ns, Ordering::Relaxed);
        self.max_wait_ns.fetch_max(ns, Ordering::Relaxed);
        if self.sample_timings {
            self.wait_samples_ns
                .lock()
                .expect("durable wait samples mutex poisoned")
                .push(ns);
        }
    }

    fn record_sync(
        &self,
        blocks: u64,
        frontier_calls: u64,
        duration: Duration,
        reason: DurableFlushReason,
    ) {
        let ns = duration_ns(duration);
        self.sync_calls.fetch_add(1, Ordering::Relaxed);
        self.sync_frontier_calls
            .fetch_add(frontier_calls, Ordering::Relaxed);
        self.synced_blocks.fetch_add(blocks, Ordering::Relaxed);
        self.sync_ns.fetch_add(ns, Ordering::Relaxed);
        self.max_sync_ns.fetch_max(ns, Ordering::Relaxed);
        if self.sample_timings {
            self.sync_samples_ns
                .lock()
                .expect("durable sync samples mutex poisoned")
                .push(ns);
        }
        match reason {
            DurableFlushReason::Pressure => {
                self.pressure_flushes.fetch_add(1, Ordering::Relaxed);
            }
            DurableFlushReason::Deadline => {
                self.deadline_flushes.fetch_add(1, Ordering::Relaxed);
            }
            DurableFlushReason::Final => {
                self.final_flushes.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn snapshot(&self) -> DurableSyncSnapshot {
        DurableSyncSnapshot {
            sync_calls: self.sync_calls.load(Ordering::Relaxed),
            sync_frontier_calls: self.sync_frontier_calls.load(Ordering::Relaxed),
            synced_blocks: self.synced_blocks.load(Ordering::Relaxed),
            sync_ns: self.sync_ns.load(Ordering::Relaxed),
            max_sync_ns: self.max_sync_ns.load(Ordering::Relaxed),
            wait_ns: self.wait_ns.load(Ordering::Relaxed),
            max_wait_ns: self.max_wait_ns.load(Ordering::Relaxed),
            pressure_flushes: self.pressure_flushes.load(Ordering::Relaxed),
            deadline_flushes: self.deadline_flushes.load(Ordering::Relaxed),
            final_flushes: self.final_flushes.load(Ordering::Relaxed),
        }
    }

    fn timing_report(&self) -> DurableSyncTimingReport {
        if !self.sample_timings {
            return DurableSyncTimingReport {
                wait: None,
                sync: None,
            };
        }
        DurableSyncTimingReport {
            wait: latency_distribution(
                self.wait_samples_ns
                    .lock()
                    .expect("durable wait samples mutex poisoned")
                    .clone(),
            ),
            sync: latency_distribution(
                self.sync_samples_ns
                    .lock()
                    .expect("durable sync samples mutex poisoned")
                    .clone(),
            ),
        }
    }
}

impl DurableSyncSnapshot {
    fn is_empty(self) -> bool {
        self.sync_calls == 0
    }

    fn blocks_per_sync(self) -> f64 {
        if self.sync_calls == 0 {
            0.0
        } else {
            self.synced_blocks as f64 / self.sync_calls as f64
        }
    }

    fn frontiers_per_sync(self) -> f64 {
        if self.sync_calls == 0 {
            0.0
        } else {
            self.sync_frontier_calls as f64 / self.sync_calls as f64
        }
    }

    fn avg_sync_ns(self) -> u64 {
        if self.sync_calls == 0 {
            0
        } else {
            self.sync_ns / self.sync_calls
        }
    }

    fn avg_wait_ns(self) -> u64 {
        if self.sync_calls == 0 {
            0
        } else {
            self.wait_ns / self.sync_calls
        }
    }

    fn format(self) -> String {
        if self.is_empty() {
            return " durable-syncs=0".to_string();
        }
        format!(
            " durable-syncs={} frontier-calls={} blocks/sync={:.1} frontiers/sync={:.1} sync-avg={} sync-max={} wait-avg={} wait-max={} flushes={}/{}/{}",
            self.sync_calls,
            self.sync_frontier_calls,
            self.blocks_per_sync(),
            self.frontiers_per_sync(),
            format_latency_ns(self.avg_sync_ns()),
            format_latency_ns(self.max_sync_ns),
            format_latency_ns(self.avg_wait_ns()),
            format_latency_ns(self.max_wait_ns),
            self.pressure_flushes,
            self.deadline_flushes,
            self.final_flushes,
        )
    }
}

impl DurableSyncTimingReport {
    fn print(&self) {
        if self.wait.is_none() && self.sync.is_none() {
            return;
        }
        println!("    durable-stage-timing");
        if let Some(distribution) = &self.wait {
            distribution.print("durable wait/group");
        }
        if let Some(distribution) = &self.sync {
            distribution.print("durable sync");
        }
    }
}

impl ChronicleCutSnapshot {
    fn store_lag(self) -> Option<u64> {
        self.store_applied
            .map(|store_applied| self.published.saturating_sub(store_applied))
    }

    fn durable_lag(self) -> Option<u64> {
        self.requested_durable
            .zip(self.durable)
            .map(|(requested, durable)| requested.saturating_sub(durable))
    }

    fn format_with_label(self, label: &str) -> String {
        format!(
            " {label} published={} store-applied={} requested-durable={} durable={} store-lag={} durable-lag={}",
            self.published,
            format_optional_u64(self.store_applied),
            format_optional_u64(self.requested_durable),
            format_optional_u64(self.durable),
            format_optional_u64(self.store_lag()),
            format_optional_u64(self.durable_lag()),
        )
    }
}

fn format_optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |value| value.to_string())
}

impl WalPrewriteCounters {
    fn record_write(&self, blocks: u64, duration: Duration) {
        if blocks == 0 {
            return;
        }
        let ns = duration_ns(duration);
        self.write_calls.fetch_add(1, Ordering::Relaxed);
        self.written_blocks.fetch_add(blocks, Ordering::Relaxed);
        self.write_ns.fetch_add(ns, Ordering::Relaxed);
        self.max_write_ns.fetch_max(ns, Ordering::Relaxed);
    }

    fn snapshot(&self) -> WalPrewriteSnapshot {
        WalPrewriteSnapshot {
            write_calls: self.write_calls.load(Ordering::Relaxed),
            written_blocks: self.written_blocks.load(Ordering::Relaxed),
            write_ns: self.write_ns.load(Ordering::Relaxed),
            max_write_ns: self.max_write_ns.load(Ordering::Relaxed),
        }
    }
}

impl WalPrewriteSnapshot {
    fn is_empty(self) -> bool {
        self.write_calls == 0
    }

    fn blocks_per_write(self) -> f64 {
        if self.write_calls == 0 {
            0.0
        } else {
            self.written_blocks as f64 / self.write_calls as f64
        }
    }

    fn avg_write_ns(self) -> u64 {
        if self.write_calls == 0 {
            0
        } else {
            self.write_ns / self.write_calls
        }
    }

    fn format(self) -> String {
        if self.is_empty() {
            return " prewrites=0".to_string();
        }
        format!(
            " prewrites={} blocks/prewrite={:.1} prewrite-avg={} prewrite-max={}",
            self.write_calls,
            self.blocks_per_write(),
            format_latency_ns(self.avg_write_ns()),
            format_latency_ns(self.max_write_ns),
        )
    }
}

struct FailureGuard {
    failed: Arc<AtomicBool>,
    armed: bool,
}

impl FailureGuard {
    fn new(failed: Arc<AtomicBool>) -> Self {
        Self {
            failed,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for FailureGuard {
    fn drop(&mut self) {
        if self.armed {
            self.failed.store(true, Ordering::Release);
        }
    }
}

#[repr(align(64))]
#[derive(Clone, Copy)]
struct ClientAppendEntry {
    intent: WriteIntent,
    final_for_client: bool,
}

#[repr(align(64))]
struct ClientAppendRingSlot {
    sequence: AtomicU64,
    logged_block_prefix: AtomicU64,
    entry: UnsafeCell<MaybeUninit<ClientAppendEntry>>,
}

unsafe impl Sync for ClientAppendRingSlot {}

impl ClientAppendRingSlot {
    fn new(sequence: u64) -> Self {
        Self {
            sequence: AtomicU64::new(sequence),
            logged_block_prefix: AtomicU64::new(0),
            entry: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    fn wait_free_and_publish(
        &self,
        sequence: u64,
        intent: WriteIntent,
        final_for_client: bool,
        failed: &AtomicBool,
    ) -> Result<(), String> {
        while self.sequence.load(Ordering::Acquire) != sequence {
            if failed.load(Ordering::Acquire) {
                return Err("coalesced client aborted while waiting for append slot".to_string());
            }
            std::thread::yield_now();
        }
        self.logged_block_prefix.store(0, Ordering::Relaxed);
        unsafe {
            (*self.entry.get()).write(ClientAppendEntry {
                intent,
                final_for_client,
            });
        }
        self.sequence.store(sequence + 1, Ordering::Release);
        Ok(())
    }

    fn try_read_published(&self, sequence: u64) -> Option<ClientAppendEntry> {
        (self.sequence.load(Ordering::Acquire) == sequence + 1)
            .then(|| unsafe { (*self.entry.get()).assume_init_read() })
    }

    fn publish_logged_block(&self, block_prefix: u64) {
        debug_assert_ne!(block_prefix, 0);
        self.logged_block_prefix
            .store(block_prefix, Ordering::Release);
    }

    fn wait_logged_block(&self, failed: &AtomicBool) -> Result<u64, String> {
        loop {
            let block_prefix = self.logged_block_prefix.load(Ordering::Acquire);
            if block_prefix != 0 {
                return Ok(block_prefix);
            }
            if failed.load(Ordering::Acquire) {
                return Err("WAL appender failed before logging request".to_string());
            }
            std::thread::yield_now();
        }
    }

    fn try_logged_block(&self) -> Option<u64> {
        let block_prefix = self.logged_block_prefix.load(Ordering::Acquire);
        (block_prefix != 0).then_some(block_prefix)
    }

    fn release(&self, sequence: u64, ring_capacity: u64) {
        self.logged_block_prefix.store(0, Ordering::Relaxed);
        self.sequence
            .store(sequence + ring_capacity, Ordering::Release);
    }
}

#[repr(align(64))]
struct ManagerPublishedBlockSlot {
    sequence: AtomicU64,
    published: AtomicBool,
    block: UnsafeCell<MaybeUninit<WalPublishedBlock>>,
}

unsafe impl Sync for ManagerPublishedBlockSlot {}

impl ManagerPublishedBlockSlot {
    fn new(sequence: u64) -> Self {
        Self {
            sequence: AtomicU64::new(sequence),
            published: AtomicBool::new(false),
            block: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    fn publish(&self, sequence: u64, block: WalPublishedBlock) {
        unsafe {
            (*self.block.get()).write(block);
        }
        self.published.store(true, Ordering::Relaxed);
        self.sequence.store(sequence + 1, Ordering::Release);
    }

    fn wait_published(
        &self,
        sequence: u64,
        failed: &AtomicBool,
    ) -> Result<WalPublishedBlock, String> {
        while self.sequence.load(Ordering::Acquire) != sequence + 1 {
            if failed.load(Ordering::Acquire) {
                return Err("manager WAL block publisher failed before apply".to_string());
            }
            std::thread::yield_now();
        }
        Ok(unsafe { (*self.block.get()).assume_init_ref().clone() })
    }

    fn try_published(&self, sequence: u64) -> Option<WalPublishedBlock> {
        (self.sequence.load(Ordering::Acquire) == sequence + 1)
            .then(|| unsafe { (*self.block.get()).assume_init_ref().clone() })
    }
}

impl Drop for ManagerPublishedBlockSlot {
    fn drop(&mut self) {
        if self.published.load(Ordering::Relaxed) {
            unsafe {
                self.block.get_mut().assume_init_drop();
            }
        }
    }
}

impl Drop for CleanupPath<'_> {
    fn drop(&mut self) {
        if !self.keep_file {
            let _ = std::fs::remove_file(self.path);
        }
    }
}

impl Drop for CleanupDir {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.path);
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

fn manager_dir_for_path(path: &Path) -> PathBuf {
    let mut file_name = path
        .file_name()
        .map(OsString::from)
        .unwrap_or_else(|| OsString::from("write-conveyor"));
    file_name.push(".manager");
    path.with_file_name(file_name)
}

fn manager_lane_dir(base_dir: &Path, lane_id: usize) -> PathBuf {
    base_dir.join(format!("lane-{lane_id:03}"))
}

fn lane_block_count(lane_id: usize, global_blocks: u64, lanes: usize) -> u64 {
    if global_blocks <= lane_id as u64 {
        0
    } else {
        (global_blocks - 1 - lane_id as u64) / lanes as u64 + 1
    }
}

fn lane_blocks_between(start_block: u64, target_prefix: u64, lane_stride: u64) -> u64 {
    if target_prefix <= start_block {
        0
    } else {
        (target_prefix - 1 - start_block) / lane_stride + 1
    }
}

fn striped_global_prefix_from_lane_counts(lane_blocks: &[u64]) -> u64 {
    let lanes = lane_blocks.len();
    let max_prefix = lane_blocks.iter().sum();
    let mut prefix = 0_u64;
    while prefix < max_prefix {
        let lane_id = prefix as usize % lanes;
        let required_lane_blocks = prefix / lanes as u64 + 1;
        if lane_blocks[lane_id] < required_lane_blocks {
            break;
        }
        prefix += 1;
    }
    prefix
}

fn prepare_manager_dir(
    base_path: &Path,
    keep: bool,
    remove_existing: bool,
) -> Result<CleanupDir, Box<dyn Error>> {
    let dir = manager_dir_for_path(base_path);
    if remove_existing {
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    } else if dir.try_exists()? {
        return Err(format!(
            "{} already exists; set CONVEYOR_OVERWRITE=1 to replace it",
            dir.display()
        )
        .into());
    }
    std::fs::create_dir_all(&dir)?;
    Ok(CleanupDir { path: dir, keep })
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

fn wait_for_completion_or_failure(
    completion: &ContiguousCompletionBarrier,
    target_prefix: u64,
    spin_before_yield: u32,
    failed: &AtomicBool,
) -> Result<(), String> {
    if target_prefix > completion.capacity() {
        return Err(format!(
            "completion target {target_prefix} exceeds capacity {}",
            completion.capacity()
        ));
    }
    let mut spins = 0_u32;
    while completion.completed_prefix() < target_prefix {
        if failed.load(Ordering::Acquire) {
            return Err("completion worker failed before acknowledging request".to_string());
        }
        if spins < spin_before_yield {
            spins += 1;
            std::hint::spin_loop();
        } else {
            spins = 0;
            std::thread::yield_now();
        }
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
    let mode = std::env::var("CONVEYOR_MODE").unwrap_or_else(|_| "both".to_string());
    let producers: usize = parse_env(
        "CONVEYOR_PRODUCERS",
        std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1),
    )
    .max(1);
    let default_workers = if is_client_latency_mode(&mode) {
        1
    } else {
        producers
    };
    let workers: usize = parse_env("CONVEYOR_WORKERS", default_workers).max(1);
    let latency_samples: usize = parse_env("CONVEYOR_LATENCY_SAMPLES", 0_usize);
    let clients: usize = parse_env("CONVEYOR_CLIENTS", producers).max(1);
    let client_default_wal_block = if is_coalesced_client_mode(&mode) {
        64_usize
    } else {
        1_usize
    };
    let client_wal_block_size = parse_env("CONVEYOR_CLIENT_WAL_BLOCK", client_default_wal_block)
        .max(1)
        .next_power_of_two();
    let client_latency_samples: usize =
        parse_env("CONVEYOR_CLIENT_LATENCY_SAMPLES", 1_000_000_usize);
    let client_wait_spins: u32 = parse_env("CONVEYOR_CLIENT_WAIT_SPINS", 0_u32);
    let client_driver_threads: usize = parse_env("CONVEYOR_CLIENT_DRIVER_THREADS", clients)
        .max(1)
        .min(clients);
    let client_driver_issue_budget: usize =
        parse_env("CONVEYOR_CLIENT_DRIVER_ISSUE_BUDGET", 1_usize).max(1);
    let durable_group_us: u64 = parse_env("CONVEYOR_DURABLE_GROUP_US", 25_u64);
    let durable_min_blocks: usize = parse_env("CONVEYOR_DURABLE_MIN_BLOCKS", 0_usize);
    let durable_lanes: usize = parse_env("CONVEYOR_DURABLE_LANES", 1_usize).max(1);
    let durable_sync_mode = parse_durable_sync_mode()?;
    let background_durable = parse_env("CONVEYOR_BACKGROUND_DURABLE", 0_u32) != 0;
    let stage_timings = parse_env("CONVEYOR_STAGE_TIMINGS", 0_u32) != 0;
    let append_group_us: u64 = parse_env("CONVEYOR_CLIENT_APPEND_GROUP_US", 25_u64);
    let min_records_per_block: usize = parse_env(
        "CONVEYOR_CLIENT_MIN_RECORDS_PER_BLOCK",
        (client_wal_block_size / 4).max(1),
    )
    .max(1)
    .min(client_wal_block_size);
    let append_ring_capacity: usize = parse_env("CONVEYOR_CLIENT_APPEND_RING", 1_usize << 16)
        .max(clients.next_power_of_two())
        .max(2)
        .next_power_of_two();
    let manager_records_per_segment: usize = parse_env(
        "CONVEYOR_WAL_MANAGER_RECORDS_PER_SEGMENT",
        client_wal_block_size
            .saturating_mul(4096)
            .max(client_wal_block_size),
    )
    .max(client_wal_block_size);
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
    if mode == "file-wal-client-logged"
        || mode == "file-wal-client-store-applied"
        || mode == "file-wal-client-durable"
    {
        run_file_wal_client_latency(
            events,
            client_wal_block_size,
            clients,
            workers,
            ClientRunOptions {
                file: FileModeOptions {
                    path: &file_path,
                    keep_file,
                    overwrite_file,
                    sync: false,
                },
                ack_mode: match mode.as_str() {
                    "file-wal-client-logged" => ClientAckMode::Logged,
                    "file-wal-client-store-applied" => ClientAckMode::StoreApplied,
                    "file-wal-client-durable" => ClientAckMode::DurableStoreApplied,
                    _ => unreachable!("client mode checked above"),
                },
                latency_samples: client_latency_samples,
                wait_spins: client_wait_spins,
                client_driver_threads,
                client_driver_issue_budget,
                durable_group_us,
                durable_min_blocks,
                durable_lanes,
                durable_sync_mode,
                background_durable,
                stage_timings,
                append_group_us,
                min_records_per_block,
                append_ring_capacity,
                manager_records_per_segment,
            },
        )?;
    }
    if mode == "file-wal-client-coalesced-logged"
        || mode == "file-wal-client-coalesced-store-applied"
        || mode == "file-wal-client-coalesced-durable"
        || mode == "file-wal-manager-coalesced-logged"
        || mode == "file-wal-manager-coalesced-store-applied"
        || mode == "file-wal-manager-coalesced-durable"
    {
        run_file_wal_client_coalesced_latency(
            events,
            client_wal_block_size,
            clients,
            workers,
            ClientRunOptions {
                file: FileModeOptions {
                    path: &file_path,
                    keep_file,
                    overwrite_file,
                    sync: false,
                },
                ack_mode: match mode.as_str() {
                    "file-wal-client-coalesced-logged" => ClientAckMode::Logged,
                    "file-wal-client-coalesced-store-applied" => ClientAckMode::StoreApplied,
                    "file-wal-client-coalesced-durable" => ClientAckMode::DurableStoreApplied,
                    "file-wal-manager-coalesced-logged" => ClientAckMode::Logged,
                    "file-wal-manager-coalesced-store-applied" => ClientAckMode::StoreApplied,
                    "file-wal-manager-coalesced-durable" => ClientAckMode::DurableStoreApplied,
                    _ => unreachable!("coalesced client mode checked above"),
                },
                latency_samples: client_latency_samples,
                wait_spins: client_wait_spins,
                client_driver_threads,
                client_driver_issue_budget,
                durable_group_us,
                durable_min_blocks,
                durable_lanes,
                durable_sync_mode,
                background_durable,
                stage_timings,
                append_group_us,
                min_records_per_block,
                append_ring_capacity,
                manager_records_per_segment,
            },
            if mode.starts_with("file-wal-manager-coalesced-") {
                CoalescedWalBackend::Manager
            } else {
                CoalescedWalBackend::Segment
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

fn run_file_wal_client_latency(
    events: u64,
    block_size: usize,
    clients: usize,
    workers: usize,
    opts: ClientRunOptions<'_>,
) -> Result<(), Box<dyn Error>> {
    let file_opts = opts.file;
    let ack_mode = opts.ack_mode;
    let _cleanup = prepare_journal_path(
        file_opts.path,
        file_opts.keep_file,
        file_opts.overwrite_file,
    )?;
    let total_blocks: usize = events.try_into()?;
    let mapped_records = total_blocks
        .checked_mul(block_size)
        .ok_or("mapped WAL segment size overflow")?;
    let requested_samples = opts.latency_samples.min(total_blocks).max(1);
    let sample_stride = (total_blocks as u64)
        .div_ceil(requested_samples as u64)
        .max(1);
    let base = events / clients as u64;
    let rem = events % clients as u64;
    let worker_count = if ack_mode.needs_store() {
        workers.max(1)
    } else {
        0
    };
    let durable_worker_count = usize::from(ack_mode.needs_durable());

    let setup_elapsed;
    let elapsed;
    let mut recovery_elapsed = Duration::ZERO;
    let store_validation_elapsed;
    let total;
    let report;
    {
        let setup_started = Instant::now();
        let segment = Arc::new(unsafe {
            MappedWalSegment::create(file_opts.path, 1, mapped_records, block_size)?
        });
        let completion = ack_mode
            .needs_store()
            .then(|| Arc::new(ContiguousCompletionBarrier::with_capacity(total_blocks)));
        let store = ack_mode.needs_store().then(|| {
            Arc::new(OpenShardAppendStore::with_capacity(
                total_blocks,
                block_size,
            ))
        });
        setup_elapsed = setup_started.elapsed();

        let barrier = Arc::new(Barrier::new(
            clients + worker_count + durable_worker_count + 1,
        ));
        let go = Arc::new(AtomicBool::new(false));
        let next_apply_block = Arc::new(AtomicU64::new(0));
        let requested_durable_prefix = Arc::new(AtomicU64::new(0));
        let durable_prefix = Arc::new(AtomicU64::new(0));
        let run_failed = Arc::new(AtomicBool::new(false));
        let durable_handle = if ack_mode.needs_durable() {
            let segment = Arc::clone(&segment);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let requested_durable_prefix = Arc::clone(&requested_durable_prefix);
            let durable_prefix = Arc::clone(&durable_prefix);
            let run_failed = Arc::clone(&run_failed);
            let durable_group_window = Duration::from_micros(opts.durable_group_us);
            let durable_sync_mode = opts.durable_sync_mode;
            Some(std::thread::spawn(move || -> std::io::Result<()> {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                while durable_prefix.load(Ordering::Acquire) < total_blocks as u64 {
                    if run_failed.load(Ordering::Acquire) {
                        return Err(std::io::Error::other(
                            "durable WAL worker aborted after peer failure",
                        ));
                    }
                    let current = durable_prefix.load(Ordering::Acquire);
                    let mut target = requested_durable_prefix.load(Ordering::Acquire);
                    if target > current {
                        if !durable_group_window.is_zero() {
                            let deadline = Instant::now() + durable_group_window;
                            while Instant::now() < deadline {
                                let observed = requested_durable_prefix.load(Ordering::Acquire);
                                if observed > target {
                                    target = observed;
                                }
                                if target >= total_blocks as u64 {
                                    break;
                                }
                                std::thread::yield_now();
                            }
                        }
                        let published = segment.contiguous_published_prefix(current, target)?;
                        if published > current {
                            if let Err(error) = segment.sync_published_data_frontier_with_mode(
                                published,
                                durable_sync_mode,
                            ) {
                                run_failed.store(true, Ordering::Release);
                                return Err(error);
                            }
                            durable_prefix.store(published, Ordering::Release);
                        } else {
                            std::thread::yield_now();
                        }
                    } else {
                        std::thread::yield_now();
                    }
                }
                failure_guard.disarm();
                Ok(())
            }))
        } else {
            None
        };
        let mut worker_handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let segment = Arc::clone(&segment);
            let completion = Arc::clone(completion.as_ref().expect("client completion barrier"));
            let store = Arc::clone(store.as_ref().expect("client store"));
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let next_apply_block = Arc::clone(&next_apply_block);
            let run_failed = Arc::clone(&run_failed);
            worker_handles.push(std::thread::spawn(move || {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                let mut total = DrainStats::default();
                let mut payload = Vec::with_capacity(block_size);
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                loop {
                    if run_failed.load(Ordering::Acquire) {
                        return Err("apply worker aborted after peer failure".to_string());
                    }
                    let block_id = next_apply_block.fetch_add(1, Ordering::Relaxed);
                    if block_id >= total_blocks as u64 {
                        break;
                    }
                    loop {
                        if run_failed.load(Ordering::Acquire) {
                            return Err("apply worker aborted after peer failure".to_string());
                        }
                        if segment
                            .read_published_block_into(block_id, &mut payload)
                            .is_some()
                        {
                            total.add(
                                store
                                    .apply_block(block_id, &payload)
                                    .map_err(|error| error.to_string())?,
                            );
                            completion
                                .complete(block_id)
                                .map_err(|error| error.to_string())?;
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
                failure_guard.disarm();
                Ok::<DrainStats, String>(total)
            }));
        }

        let mut client_handles = Vec::with_capacity(clients);
        for client_id in 0..clients {
            let segment = Arc::clone(&segment);
            let completion = completion.as_ref().map(Arc::clone);
            let requested_durable_prefix = Arc::clone(&requested_durable_prefix);
            let durable_prefix = Arc::clone(&durable_prefix);
            let run_failed = Arc::clone(&run_failed);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let count = base + u64::from((client_id as u64) < rem);
            let first = base * client_id as u64 + rem.min(client_id as u64);
            client_handles.push(std::thread::spawn(move || {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                let sample_capacity = (count.div_ceil(sample_stride) as usize).saturating_add(1);
                let mut latencies = ClientLatencySamples::with_capacity(sample_capacity);
                let mut total = DrainStats::default();
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                for offset in 0..count {
                    if run_failed.load(Ordering::Acquire) {
                        return Err("client aborted after peer failure".to_string());
                    }
                    let sequence = first + offset;
                    let sampled = sequence.is_multiple_of(sample_stride);
                    let intent = intent_for(sequence);
                    let started = sampled.then(Instant::now);
                    let block_id = segment
                        .try_publish_intents_position(std::slice::from_ref(&intent))
                        .map_err(|error| error.to_string())?
                        .expect("single-intent publish returns a block id");
                    let logged_at = sampled.then(Instant::now);
                    if ack_mode.needs_durable() {
                        requested_durable_prefix.fetch_max(block_id + 1, Ordering::Release);
                    }
                    total.observe(intent);
                    let mut store_applied_at = None;
                    if let Some(completion) = completion.as_ref() {
                        wait_for_completion_or_failure(
                            completion,
                            block_id + 1,
                            opts.wait_spins,
                            &run_failed,
                        )?;
                        store_applied_at = sampled.then(Instant::now);
                    }
                    if ack_mode.needs_durable() {
                        while durable_prefix.load(Ordering::Acquire) < block_id + 1 {
                            if run_failed.load(Ordering::Acquire) {
                                return Err(
                                    "durable WAL worker failed before acknowledging request"
                                        .to_string(),
                                );
                            }
                            std::thread::yield_now();
                        }
                    }
                    if let (Some(started), Some(logged_at)) = (started, logged_at) {
                        let acked_at = Instant::now();
                        latencies
                            .client_to_logged
                            .push(duration_ns(logged_at.duration_since(started)));
                        if completion.is_some() {
                            latencies
                                .logged_to_ack
                                .push(duration_ns(acked_at.duration_since(logged_at)));
                        }
                        if let Some(store_applied_at) = store_applied_at {
                            if ack_mode.needs_durable() {
                                latencies
                                    .store_to_ack
                                    .push(duration_ns(acked_at.duration_since(store_applied_at)));
                            }
                        }
                        latencies
                            .client_to_ack
                            .push(duration_ns(acked_at.duration_since(started)));
                    }
                }
                failure_guard.disarm();
                Ok::<(DrainStats, ClientLatencySamples), String>((total, latencies))
            }));
        }

        barrier.wait();
        let started = Instant::now();
        go.store(true, Ordering::Release);
        let mut client_total = DrainStats::default();
        let mut samples = ClientLatencySamples::default();
        let mut first_error = None::<String>;
        for handle in client_handles {
            match handle.join() {
                Ok(Ok((thread_total, thread_samples))) => {
                    client_total.add(thread_total);
                    samples.append(thread_samples);
                }
                Ok(Err(message)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(message);
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some("client thread panicked".to_string());
                    }
                }
            }
        }
        let mut worker_total = DrainStats::default();
        for handle in worker_handles {
            match handle.join() {
                Ok(Ok(thread_total)) => worker_total.add(thread_total),
                Ok(Err(message)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(message);
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some("apply worker thread panicked".to_string());
                    }
                }
            }
        }
        if let Some(handle) = durable_handle {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(error.to_string());
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some("durable WAL worker thread panicked".to_string());
                    }
                }
            }
        }
        if let Some(message) = first_error {
            return Err(std::io::Error::other(message).into());
        }
        elapsed = started.elapsed();
        total = if ack_mode.needs_store() {
            worker_total
        } else {
            client_total
        };
        report = samples
            .report(requested_samples, sample_stride)
            .ok_or("client latency sampling produced no samples")?;
        let validate_started = Instant::now();
        if let Some(store) = store.as_ref() {
            let observed = store
                .validate_applied_blocks(total_blocks as u64)
                .ok_or("client store validation could not read applied prefix")?;
            let expected = stats_for_range(0, events);
            if observed != expected {
                return Err(format!(
                    "client store validation mismatch: got {observed:?}, expected {expected:?}"
                )
                .into());
            }
        }
        store_validation_elapsed = validate_started.elapsed();
    }

    assert_expected_stats(ack_mode.label(), total, events)?;
    if ack_mode.needs_durable() {
        let recovery_started = Instant::now();
        let recovered = recover_wal_segment_by_scan(file_opts.path)?;
        recovery_elapsed = recovery_started.elapsed();
        let expected = stats_for_range(0, events);
        if recovered.recovered_blocks != total_blocks as u64
            || recovered.recovered_records != events
            || recovered.stats != expected
        {
            return Err(format!(
                "client durable recovery mismatch: recovered {recovered:?}, expected_blocks={total_blocks}, expected_stats={expected:?}"
            )
            .into());
        }
    }
    let store_validation = if ack_mode.needs_store() {
        format!(
            " store-validate={:.3}s",
            store_validation_elapsed.as_secs_f64()
        )
    } else {
        String::new()
    };
    let extra = format!(
        " setup={:.3}s clients={clients} workers={worker_count} wal-block={block_size} wait-spins={wait_spins} ack={ack}{store_validation}{recovery}",
        setup_elapsed.as_secs_f64(),
        wait_spins = opts.wait_spins,
        ack = match ack_mode {
            ClientAckMode::Logged => "non-durable-logged",
            ClientAckMode::StoreApplied => "non-durable-store-applied",
            ClientAckMode::DurableStoreApplied => "data-fenced-wal+volatile-store-applied",
        },
        recovery = if ack_mode.needs_durable() {
            format!(
                " durable-group={}us durable-sync-mode={} recover={:.3}s",
                opts.durable_group_us,
                durable_sync_mode_label(opts.durable_sync_mode),
                recovery_elapsed.as_secs_f64()
            )
        } else {
            String::new()
        }
    );
    report_extra(ack_mode.label(), events, elapsed, total, &extra);
    report.print(ack_mode);
    Ok(())
}

fn run_file_wal_client_coalesced_latency(
    events: u64,
    block_size: usize,
    clients: usize,
    workers: usize,
    opts: ClientRunOptions<'_>,
    wal_backend: CoalescedWalBackend,
) -> Result<(), Box<dyn Error>> {
    let file_opts = opts.file;
    let ack_mode = opts.ack_mode;
    let durable_enabled = ack_mode.needs_durable() || opts.background_durable;
    let requested_durable_lanes = opts.durable_lanes.max(1);
    let durable_lane_count = if wal_backend == CoalescedWalBackend::Manager && durable_enabled {
        requested_durable_lanes
    } else {
        1
    };
    if requested_durable_lanes > 1
        && !(wal_backend == CoalescedWalBackend::Manager && durable_enabled)
    {
        return Err(
            "CONVEYOR_DURABLE_LANES>1 is currently supported only for manager-backed coalesced WAL with durable or background-durable enabled"
                .into(),
        );
    }
    if durable_lane_count > 1 && opts.durable_sync_mode == WalDataSyncMode::PrewriteAndFileData {
        return Err("CONVEYOR_DURABLE_LANES>1 does not yet support prewrite-and-file-data".into());
    }
    let _file_cleanup;
    let _manager_cleanup;
    let manager_configs;
    match wal_backend {
        CoalescedWalBackend::Segment => {
            _file_cleanup = Some(prepare_journal_path(
                file_opts.path,
                file_opts.keep_file,
                file_opts.overwrite_file,
            )?);
            _manager_cleanup = None;
            manager_configs = None;
        }
        CoalescedWalBackend::Manager => {
            _file_cleanup = None;
            let cleanup = prepare_manager_dir(
                file_opts.path,
                file_opts.keep_file,
                file_opts.overwrite_file,
            )?;
            let configs = if durable_lane_count == 1 {
                vec![WalSegmentManagerConfig::new(
                    cleanup.path.clone(),
                    "events",
                    opts.manager_records_per_segment,
                    block_size,
                )]
            } else {
                (0..durable_lane_count)
                    .map(|lane_id| {
                        WalSegmentManagerConfig::new(
                            manager_lane_dir(&cleanup.path, lane_id),
                            "events",
                            opts.manager_records_per_segment,
                            block_size,
                        )
                    })
                    .collect()
            };
            manager_configs = Some(configs);
            _manager_cleanup = Some(cleanup);
        }
    }
    let total_events: usize = events.try_into()?;
    let min_records_per_block = opts.min_records_per_block.max(1).min(block_size);
    let effective_min_records_per_block = min_records_per_block.min(clients).max(1);
    let append_ring_capacity = opts
        .append_ring_capacity
        .max(clients.next_power_of_two())
        .max(2)
        .next_power_of_two();
    let append_ring_mask = append_ring_capacity as u64 - 1;
    let minimum_block_budget = total_events
        .div_ceil(effective_min_records_per_block)
        .max(1);
    let sparse_closed_loop_budget =
        opts.append_group_us == 0 || clients <= effective_min_records_per_block.saturating_mul(2);
    let max_blocks = if sparse_closed_loop_budget {
        total_events.max(1)
    } else {
        minimum_block_budget
            .saturating_add(append_ring_capacity)
            .min(total_events)
            .max(1)
    };
    let partial_block_slack = max_blocks.saturating_sub(minimum_block_budget);
    let mapped_records = max_blocks
        .checked_mul(block_size)
        .ok_or("coalesced mapped WAL segment size overflow")?;
    let requested_samples = opts.latency_samples.min(total_events).max(1);
    let sample_stride = (total_events as u64)
        .div_ceil(requested_samples as u64)
        .max(1);
    let base = events / clients as u64;
    let rem = events % clients as u64;
    let active_clients = clients.min(total_events);
    let worker_count = if ack_mode.needs_store() {
        workers.max(1)
    } else {
        0
    };
    let durable_worker_count = if durable_enabled {
        durable_lane_count
    } else {
        0
    };
    let prewrite_worker_count = usize::from(
        durable_enabled && opts.durable_sync_mode == WalDataSyncMode::PrewriteAndFileData,
    );
    let client_driver_threads = opts.client_driver_threads.max(1).min(clients);
    let multiplexed_clients = client_driver_threads < clients;

    let setup_elapsed;
    let elapsed;
    let visible_elapsed;
    let total_elapsed;
    let durable_catchup_elapsed;
    let mut recovery_elapsed = Duration::ZERO;
    let store_validation_elapsed;
    let total;
    let report;
    let actual_blocks;
    let visible_cut_snapshot;
    let final_cut_snapshot;
    let durable_sync_snapshot;
    let durable_sync_timing_report;
    let prewrite_snapshot;
    let stage_counters = opts
        .stage_timings
        .then(|| Arc::new(CoalescedStageCounters::default()));
    let block_timeline = opts
        .stage_timings
        .then(|| Arc::new(BlockStageTimeline::with_capacity(max_blocks)));
    {
        let setup_started = Instant::now();
        let segment = if wal_backend == CoalescedWalBackend::Segment {
            Some(Arc::new(unsafe {
                MappedWalSegment::create(file_opts.path, 1, mapped_records, block_size)?
            }))
        } else {
            None
        };
        let managers = if wal_backend == CoalescedWalBackend::Manager {
            Some(
                manager_configs
                    .as_ref()
                    .expect("manager configs for manager-backed coalesced WAL")
                    .iter()
                    .cloned()
                    .map(|config| unsafe { WalSegmentManager::create(config) })
                    .collect::<std::io::Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        let manager_blocks: Option<Arc<[ManagerPublishedBlockSlot]>> =
            if wal_backend == CoalescedWalBackend::Manager {
                Some(
                    (0..max_blocks)
                        .map(|sequence| ManagerPublishedBlockSlot::new(sequence as u64))
                        .collect::<Vec<_>>()
                        .into(),
                )
            } else {
                None
            };
        let queue: Arc<[ClientAppendRingSlot]> = (0..append_ring_capacity)
            .map(|sequence| ClientAppendRingSlot::new(sequence as u64))
            .collect::<Vec<_>>()
            .into();
        let completion = ack_mode
            .needs_store()
            .then(|| Arc::new(ContiguousCompletionBarrier::with_capacity(max_blocks)));
        let store = ack_mode
            .needs_store()
            .then(|| Arc::new(OpenShardAppendStore::with_capacity(max_blocks, block_size)));
        setup_elapsed = setup_started.elapsed();

        let barrier = Arc::new(Barrier::new(
            client_driver_threads + worker_count + durable_worker_count + prewrite_worker_count + 2,
        ));
        let go = Arc::new(AtomicBool::new(false));
        let run_failed = Arc::new(AtomicBool::new(false));
        let enqueue_tail = Arc::new(AtomicU64::new(0));
        let published_blocks = Arc::new(AtomicU64::new(0));
        let appender_done = Arc::new(AtomicBool::new(false));
        let next_apply_block = Arc::new(AtomicU64::new(0));
        let requested_durable_prefix = Arc::new(AtomicU64::new(0));
        let durable_prefix = Arc::new(AtomicU64::new(0));
        let durable_syncs =
            durable_enabled.then(|| Arc::new(DurableSyncCounters::new(opts.stage_timings)));
        let prewrite_counters =
            (prewrite_worker_count != 0).then(|| Arc::new(WalPrewriteCounters::default()));

        let appender_handle = {
            let segment = segment.as_ref().map(Arc::clone);
            let manager_blocks = manager_blocks.as_ref().map(Arc::clone);
            let mut managers = managers;
            let stage_counters = stage_counters.as_ref().map(Arc::clone);
            let block_timeline = block_timeline.as_ref().map(Arc::clone);
            let queue = Arc::clone(&queue);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let run_failed = Arc::clone(&run_failed);
            let published_blocks = Arc::clone(&published_blocks);
            let appender_done = Arc::clone(&appender_done);
            let requested_durable_prefix = Arc::clone(&requested_durable_prefix);
            let enqueue_tail = Arc::clone(&enqueue_tail);
            let append_group_window = Duration::from_micros(opts.append_group_us);
            std::thread::spawn(move || {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                let mut next_sequence = 0_u64;
                let mut next_block_id = 0_u64;
                let mut active_clients_remaining = active_clients;
                let mut batch = Vec::with_capacity(block_size);
                let mut positions = Vec::with_capacity(block_size);
                let mut total = DrainStats::default();
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                while next_sequence < total_events as u64 {
                    if run_failed.load(Ordering::Acquire) {
                        return Err("WAL appender aborted after peer failure".to_string());
                    }
                    batch.clear();
                    positions.clear();
                    let mut active_clients_after_batch = active_clients_remaining;
                    let mut first_ready_at = None::<Instant>;
                    while batch.len() < block_size && next_sequence < total_events as u64 {
                        if run_failed.load(Ordering::Acquire) {
                            return Err("WAL appender aborted after peer failure".to_string());
                        }
                        let slot = &queue[(next_sequence & append_ring_mask) as usize];
                        if let Some(entry) = slot.try_read_published(next_sequence) {
                            first_ready_at.get_or_insert_with(Instant::now);
                            total.observe(entry.intent);
                            batch.push(entry.intent);
                            positions.push(next_sequence);
                            if entry.final_for_client {
                                active_clients_after_batch =
                                    active_clients_after_batch.saturating_sub(1);
                            }
                            next_sequence += 1;
                            continue;
                        }
                        if batch.is_empty() {
                            std::thread::yield_now();
                            continue;
                        }
                        let no_claimed_gap = enqueue_tail.load(Ordering::Acquire) <= next_sequence;
                        let final_tail = next_sequence >= total_events as u64;
                        let aged = append_group_window.is_zero()
                            || first_ready_at
                                .is_some_and(|started| started.elapsed() >= append_group_window);
                        let target_records =
                            effective_min_records_per_block.min(active_clients_after_batch.max(1));
                        let has_minimum_block = batch.len() >= target_records;
                        if final_tail
                            || has_minimum_block
                            || (sparse_closed_loop_budget && aged && no_claimed_gap)
                        {
                            break;
                        }
                        std::thread::yield_now();
                    }
                    if batch.is_empty() {
                        continue;
                    }
                    if let Some(first_ready_at) = first_ready_at {
                        if let Some(stage_counters) = &stage_counters {
                            stage_counters
                                .append_batch_wait
                                .record(first_ready_at.elapsed());
                        }
                    }
                    let publish_started = stage_counters.as_ref().map(|_| Instant::now());
                    let publish_started_ns =
                        block_timeline.as_ref().map(|timeline| timeline.now_ns());
                    let block_prefix = match wal_backend {
                        CoalescedWalBackend::Segment => {
                            let block_id = segment
                                .as_ref()
                                .expect("segment-backed coalesced WAL")
                                .try_publish_intents_position(&batch)
                                .map_err(|error| error.to_string())?
                                .ok_or("coalesced WAL appender published an empty batch")?;
                            block_id + 1
                        }
                        CoalescedWalBackend::Manager => {
                            let lane_id = (next_block_id as usize) % durable_lane_count;
                            let block = managers
                                .as_mut()
                                .expect("manager-backed coalesced WAL")
                                .get_mut(lane_id)
                                .expect("durable WAL lane exists")
                                .publish_intents_handle(&batch)
                                .map_err(|error| error.to_string())?;
                            let global_block_id = next_block_id;
                            if global_block_id >= max_blocks as u64 {
                                return Err(format!(
                                    "manager coalesced WAL block {global_block_id} exceeds benchmark block budget {max_blocks}"
                                ));
                            }
                            if let Some(timeline) = &block_timeline {
                                timeline.record_logged_ns(global_block_id, timeline.now_ns());
                            }
                            manager_blocks.as_ref().expect("manager block directory")
                                [global_block_id as usize]
                                .publish(global_block_id, block);
                            global_block_id + 1
                        }
                    };
                    if let (Some(timeline), Some(logged_ns)) =
                        (block_timeline.as_ref(), publish_started_ns)
                    {
                        if wal_backend == CoalescedWalBackend::Segment {
                            timeline.record_logged_ns(block_prefix - 1, logged_ns);
                        }
                    }
                    if let (Some(stage_counters), Some(publish_started)) =
                        (&stage_counters, publish_started)
                    {
                        stage_counters
                            .append_wal_publish
                            .record(publish_started.elapsed());
                    }
                    next_block_id = block_prefix;
                    let notify_started = stage_counters.as_ref().map(|_| Instant::now());
                    for sequence in positions.iter().copied() {
                        queue[(sequence & append_ring_mask) as usize]
                            .publish_logged_block(block_prefix);
                    }
                    if let (Some(stage_counters), Some(notify_started)) =
                        (&stage_counters, notify_started)
                    {
                        stage_counters
                            .append_client_notify
                            .record(notify_started.elapsed());
                    }
                    active_clients_remaining = active_clients_after_batch;
                    published_blocks.store(block_prefix, Ordering::Release);
                    if durable_enabled {
                        requested_durable_prefix.fetch_max(block_prefix, Ordering::Release);
                    }
                }
                appender_done.store(true, Ordering::Release);
                failure_guard.disarm();
                Ok::<DrainStats, String>(total)
            })
        };

        let prewrite_handle = if prewrite_worker_count != 0 {
            let segment = segment.as_ref().map(Arc::clone);
            let manager_blocks = manager_blocks.as_ref().map(Arc::clone);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let run_failed = Arc::clone(&run_failed);
            let published_blocks = Arc::clone(&published_blocks);
            let appender_done = Arc::clone(&appender_done);
            let prewrite_counters =
                Arc::clone(prewrite_counters.as_ref().expect("prewrite counters"));
            Some(std::thread::spawn(move || -> std::io::Result<()> {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                let mut current = 0_u64;
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                loop {
                    if run_failed.load(Ordering::Acquire) {
                        return Err(std::io::Error::other(
                            "WAL prewrite worker aborted after peer failure",
                        ));
                    }
                    let final_blocks = published_blocks.load(Ordering::Acquire);
                    if appender_done.load(Ordering::Acquire) && current >= final_blocks {
                        break;
                    }
                    if final_blocks <= current {
                        std::thread::yield_now();
                        continue;
                    }
                    let published = match wal_backend {
                        CoalescedWalBackend::Segment => segment
                            .as_ref()
                            .expect("segment-backed prewrite coalesced WAL")
                            .contiguous_published_prefix(current, final_blocks)?,
                        CoalescedWalBackend::Manager => final_blocks,
                    };
                    if published <= current {
                        std::thread::yield_now();
                        continue;
                    }

                    let write_started = Instant::now();
                    let mut written_blocks = 0_u64;
                    match wal_backend {
                        CoalescedWalBackend::Segment => {
                            written_blocks = segment
                                .as_ref()
                                .expect("segment-backed prewrite coalesced WAL")
                                .write_published_data_frontier(published)?;
                        }
                        CoalescedWalBackend::Manager => {
                            let manager_blocks =
                                manager_blocks.as_ref().expect("manager block directory");
                            let mut pending_segment = None::<u64>;
                            let mut pending_write = None::<WalPublishedBlock>;
                            for block_id in current..published {
                                let block = manager_blocks[block_id as usize]
                                    .wait_published(block_id, &run_failed)
                                    .map_err(std::io::Error::other)?;
                                let position = block.position();
                                if pending_segment
                                    .is_some_and(|segment_id| segment_id != position.segment_id)
                                {
                                    if let Some(to_write) = pending_write.take() {
                                        written_blocks +=
                                            to_write.write_published_data_frontier()?;
                                    }
                                }
                                pending_segment = Some(position.segment_id);
                                pending_write = Some(block);
                            }
                            if let Some(to_write) = pending_write {
                                written_blocks += to_write.write_published_data_frontier()?;
                            }
                        }
                    }
                    prewrite_counters.record_write(written_blocks, write_started.elapsed());
                    current = published;
                }
                failure_guard.disarm();
                Ok(())
            }))
        } else {
            None
        };

        let durable_handles = if durable_enabled {
            if durable_lane_count == 1 {
                let segment = segment.as_ref().map(Arc::clone);
                let manager_blocks = manager_blocks.as_ref().map(Arc::clone);
                let barrier = Arc::clone(&barrier);
                let go = Arc::clone(&go);
                let run_failed = Arc::clone(&run_failed);
                let requested_durable_prefix = Arc::clone(&requested_durable_prefix);
                let durable_prefix = Arc::clone(&durable_prefix);
                let published_blocks = Arc::clone(&published_blocks);
                let appender_done = Arc::clone(&appender_done);
                let durable_syncs =
                    Arc::clone(durable_syncs.as_ref().expect("durable sync counters"));
                let block_timeline = block_timeline.as_ref().map(Arc::clone);
                let durable_group_window = Duration::from_micros(opts.durable_group_us);
                let configured_durable_min_blocks = opts.durable_min_blocks as u64;
                let auto_durable_min_blocks = configured_durable_min_blocks == 0;
                let durable_sync_mode = opts.durable_sync_mode;
                vec![std::thread::spawn(move || -> std::io::Result<()> {
                    let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                    let mut durable_min_blocks = if auto_durable_min_blocks {
                        1
                    } else {
                        configured_durable_min_blocks.max(1)
                    };
                    barrier.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    loop {
                        if run_failed.load(Ordering::Acquire) {
                            return Err(std::io::Error::other(
                                "durable WAL worker aborted after peer failure",
                            ));
                        }
                        let current = durable_prefix.load(Ordering::Acquire);
                        let final_blocks = published_blocks.load(Ordering::Acquire);
                        if appender_done.load(Ordering::Acquire) && current >= final_blocks {
                            break;
                        }
                        let mut target = requested_durable_prefix.load(Ordering::Acquire);
                        if target > current {
                            let wait_started = Instant::now();
                            let mut reason = DurableFlushReason::Pressure;
                            if target.saturating_sub(current) < durable_min_blocks {
                                let deadline = Instant::now() + durable_group_window;
                                reason = DurableFlushReason::Deadline;
                                loop {
                                    let observed = requested_durable_prefix.load(Ordering::Acquire);
                                    if observed > target {
                                        target = observed;
                                    }
                                    let final_blocks = published_blocks.load(Ordering::Acquire);
                                    if appender_done.load(Ordering::Acquire)
                                        && observed >= final_blocks
                                    {
                                        reason = DurableFlushReason::Final;
                                        break;
                                    }
                                    if target.saturating_sub(current) >= durable_min_blocks {
                                        reason = DurableFlushReason::Pressure;
                                        break;
                                    }
                                    if durable_group_window.is_zero() || Instant::now() >= deadline
                                    {
                                        break;
                                    }
                                    std::thread::yield_now();
                                }
                            }
                            let published = match wal_backend {
                                CoalescedWalBackend::Segment => segment
                                    .as_ref()
                                    .expect("segment-backed durable coalesced WAL")
                                    .contiguous_published_prefix(current, target)?,
                                CoalescedWalBackend::Manager => target,
                            };
                            if published > current {
                                let wait_elapsed = wait_started.elapsed();
                                let sync_started = Instant::now();
                                let mut sync_frontier_calls = 0_u64;
                                match wal_backend {
                                    CoalescedWalBackend::Segment => {
                                        if let Err(error) = segment
                                            .as_ref()
                                            .expect("segment-backed durable coalesced WAL")
                                            .sync_published_data_frontier_with_mode(
                                                published,
                                                durable_sync_mode,
                                            )
                                        {
                                            run_failed.store(true, Ordering::Release);
                                            return Err(error);
                                        }
                                        sync_frontier_calls = 1;
                                    }
                                    CoalescedWalBackend::Manager => {
                                        let manager_blocks = manager_blocks
                                            .as_ref()
                                            .expect("manager block directory");
                                        let mut pending_segment = None::<u64>;
                                        let mut pending_sync = None::<WalPublishedBlock>;
                                        for block_id in current..published {
                                            let block = manager_blocks[block_id as usize]
                                                .wait_published(block_id, &run_failed)
                                                .map_err(std::io::Error::other)?;
                                            let position = block.position();
                                            if pending_segment.is_some_and(|segment_id| {
                                                segment_id != position.segment_id
                                            }) {
                                                if let Some(to_sync) = pending_sync.take() {
                                                    sync_frontier_calls += 1;
                                                    if let Err(error) = to_sync
                                                        .sync_published_data_frontier_with_mode(
                                                            durable_sync_mode,
                                                        )
                                                    {
                                                        run_failed.store(true, Ordering::Release);
                                                        return Err(error);
                                                    }
                                                }
                                            }
                                            pending_segment = Some(position.segment_id);
                                            pending_sync = Some(block);
                                        }
                                        if let Some(to_sync) = pending_sync {
                                            sync_frontier_calls += 1;
                                            if let Err(error) = to_sync
                                                .sync_published_data_frontier_with_mode(
                                                    durable_sync_mode,
                                                )
                                            {
                                                run_failed.store(true, Ordering::Release);
                                                return Err(error);
                                            }
                                        }
                                    }
                                }
                                let sync_elapsed = sync_started.elapsed();
                                durable_syncs.record_wait(wait_elapsed);
                                durable_syncs.record_sync(
                                    published - current,
                                    sync_frontier_calls,
                                    sync_elapsed,
                                    reason,
                                );
                                if auto_durable_min_blocks {
                                    let sync_ns = duration_ns(sync_elapsed);
                                    durable_min_blocks = if sync_ns >= 100_000 {
                                        4
                                    } else if sync_ns >= 25_000 {
                                        2
                                    } else {
                                        1
                                    };
                                }
                                for block_id in current..published {
                                    if let Some(block_timeline) = &block_timeline {
                                        block_timeline.record_durable(block_id);
                                    }
                                }
                                durable_prefix.store(published, Ordering::Release);
                            } else {
                                std::thread::yield_now();
                            }
                        } else {
                            std::thread::yield_now();
                        }
                    }
                    failure_guard.disarm();
                    Ok(())
                })]
            } else {
                let durable_completion =
                    Arc::new(ContiguousCompletionBarrier::with_capacity(max_blocks));
                (0..durable_lane_count)
                    .map(|lane_id| {
                        let manager_blocks =
                            Arc::clone(manager_blocks.as_ref().expect("manager block directory"));
                        let durable_completion = Arc::clone(&durable_completion);
                        let barrier = Arc::clone(&barrier);
                        let go = Arc::clone(&go);
                        let run_failed = Arc::clone(&run_failed);
                        let requested_durable_prefix = Arc::clone(&requested_durable_prefix);
                        let durable_prefix = Arc::clone(&durable_prefix);
                        let published_blocks = Arc::clone(&published_blocks);
                        let appender_done = Arc::clone(&appender_done);
                        let durable_syncs =
                            Arc::clone(durable_syncs.as_ref().expect("durable sync counters"));
                        let block_timeline = block_timeline.as_ref().map(Arc::clone);
                        let durable_group_window = Duration::from_micros(opts.durable_group_us);
                        let configured_durable_min_blocks = opts.durable_min_blocks as u64;
                        let auto_durable_min_blocks = configured_durable_min_blocks == 0;
                        let durable_sync_mode = opts.durable_sync_mode;
                        std::thread::spawn(move || -> std::io::Result<()> {
                            let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                            let mut next_global_block = lane_id as u64;
                            let lane_stride = durable_lane_count as u64;
                            let mut lane_batch_blocks = if auto_durable_min_blocks {
                                1
                            } else {
                                configured_durable_min_blocks.max(1)
                            };
                            let mut durable_batch = Vec::new();
                            let mut sync_batch = Vec::new();
                            barrier.wait();
                            while !go.load(Ordering::Acquire) {
                                std::hint::spin_loop();
                            }
                            loop {
                                if run_failed.load(Ordering::Acquire) {
                                    return Err(std::io::Error::other(
                                        "striped durable WAL worker aborted after peer failure",
                                    ));
                                }
                                let final_blocks = published_blocks.load(Ordering::Acquire);
                                if appender_done.load(Ordering::Acquire)
                                    && next_global_block >= final_blocks
                                {
                                    break;
                                }
                                let mut target = requested_durable_prefix.load(Ordering::Acquire);
                                if target <= next_global_block {
                                    std::thread::yield_now();
                                    continue;
                                }

                                let wait_started = Instant::now();
                                let mut reason = DurableFlushReason::Pressure;
                                if lane_blocks_between(next_global_block, target, lane_stride)
                                    < lane_batch_blocks
                                {
                                    let deadline = Instant::now() + durable_group_window;
                                    reason = DurableFlushReason::Deadline;
                                    loop {
                                        let observed =
                                            requested_durable_prefix.load(Ordering::Acquire);
                                        if observed > target {
                                            target = observed;
                                        }
                                        let final_blocks = published_blocks.load(Ordering::Acquire);
                                        if appender_done.load(Ordering::Acquire)
                                            && observed >= final_blocks
                                        {
                                            reason = DurableFlushReason::Final;
                                            break;
                                        }
                                        if lane_blocks_between(
                                            next_global_block,
                                            target,
                                            lane_stride,
                                        ) >= lane_batch_blocks
                                        {
                                            reason = DurableFlushReason::Pressure;
                                            break;
                                        }
                                        if durable_group_window.is_zero()
                                            || Instant::now() >= deadline
                                        {
                                            break;
                                        }
                                        std::thread::yield_now();
                                    }
                                }
                                durable_batch.clear();
                                sync_batch.clear();
                                let mut block_id = next_global_block;
                                let mut pending_segment = None::<u64>;
                                let mut pending_sync = None::<WalPublishedBlock>;
                                while block_id < target
                                    && durable_batch.len() < lane_batch_blocks as usize
                                {
                                    let block = manager_blocks[block_id as usize]
                                        .wait_published(block_id, &run_failed)
                                        .map_err(std::io::Error::other)?;
                                    let position = block.position();
                                    if pending_segment
                                        .is_some_and(|segment_id| segment_id != position.segment_id)
                                    {
                                        if let Some(to_sync) = pending_sync.take() {
                                            sync_batch.push(to_sync);
                                        }
                                    }
                                    pending_segment = Some(position.segment_id);
                                    pending_sync = Some(block);
                                    durable_batch.push(block_id);
                                    block_id += lane_stride;
                                }
                                if let Some(to_sync) = pending_sync {
                                    sync_batch.push(to_sync);
                                }
                                if sync_batch.is_empty() {
                                    std::thread::yield_now();
                                    continue;
                                }
                                let wait_elapsed = wait_started.elapsed();
                                let sync_started = Instant::now();
                                for sync_block in &sync_batch {
                                    if let Err(error) = sync_block
                                        .sync_published_data_frontier_with_mode(durable_sync_mode)
                                    {
                                        run_failed.store(true, Ordering::Release);
                                        return Err(error);
                                    }
                                }
                                let sync_elapsed = sync_started.elapsed();
                                durable_syncs.record_wait(wait_elapsed);
                                durable_syncs.record_sync(
                                    durable_batch.len() as u64,
                                    sync_batch.len() as u64,
                                    sync_elapsed,
                                    reason,
                                );
                                if auto_durable_min_blocks {
                                    let sync_ns = duration_ns(sync_elapsed);
                                    lane_batch_blocks = if sync_ns >= 100_000 {
                                        4
                                    } else if sync_ns >= 25_000 {
                                        2
                                    } else {
                                        1
                                    };
                                }
                                for block_id in durable_batch.iter().copied() {
                                    if let Some(block_timeline) = &block_timeline {
                                        block_timeline.record_durable(block_id);
                                    }
                                    let prefix =
                                        durable_completion.complete(block_id).map_err(|error| {
                                            std::io::Error::other(error.to_string())
                                        })?;
                                    durable_prefix.fetch_max(prefix, Ordering::AcqRel);
                                }
                                next_global_block = block_id;
                            }
                            failure_guard.disarm();
                            Ok(())
                        })
                    })
                    .collect()
            }
        } else {
            Vec::new()
        };

        let mut worker_handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let segment = segment.as_ref().map(Arc::clone);
            let manager_blocks = manager_blocks.as_ref().map(Arc::clone);
            let completion = Arc::clone(completion.as_ref().expect("coalesced completion barrier"));
            let store = Arc::clone(store.as_ref().expect("coalesced client store"));
            let stage_counters = stage_counters.as_ref().map(Arc::clone);
            let block_timeline = block_timeline.as_ref().map(Arc::clone);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let run_failed = Arc::clone(&run_failed);
            let next_apply_block = Arc::clone(&next_apply_block);
            let published_blocks = Arc::clone(&published_blocks);
            let appender_done = Arc::clone(&appender_done);
            worker_handles.push(std::thread::spawn(move || {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                let mut total = DrainStats::default();
                let mut payload = Vec::with_capacity(block_size);
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                loop {
                    if run_failed.load(Ordering::Acquire) {
                        return Err("coalesced apply worker aborted after peer failure".to_string());
                    }
                    let block_id = next_apply_block.fetch_add(1, Ordering::Relaxed);
                    if block_id >= max_blocks as u64 {
                        break;
                    }
                    let wait_started = stage_counters.as_ref().map(|_| Instant::now());
                    loop {
                        if run_failed.load(Ordering::Acquire) {
                            return Err(
                                "coalesced apply worker aborted after peer failure".to_string()
                            );
                        }
                        if appender_done.load(Ordering::Acquire)
                            && block_id >= published_blocks.load(Ordering::Acquire)
                        {
                            failure_guard.disarm();
                            return Ok::<DrainStats, String>(total);
                        }
                        let read_started = stage_counters.as_ref().map(|_| Instant::now());
                        let read = match wal_backend {
                            CoalescedWalBackend::Segment => segment
                                .as_ref()
                                .expect("segment-backed coalesced WAL")
                                .read_published_block_into(block_id, &mut payload),
                            CoalescedWalBackend::Manager => {
                                if let Some(block) =
                                    manager_blocks.as_ref().expect("manager block directory")
                                        [block_id as usize]
                                        .try_published(block_id)
                                {
                                    block.read_published_block_into(&mut payload)
                                } else {
                                    None
                                }
                            }
                        };
                        if read.is_some() {
                            if let (Some(stage_counters), Some(wait_started)) =
                                (&stage_counters, wait_started)
                            {
                                stage_counters
                                    .store_wait_for_block
                                    .record(wait_started.elapsed());
                            }
                            if let (Some(stage_counters), Some(read_started)) =
                                (&stage_counters, read_started)
                            {
                                stage_counters
                                    .store_read_block
                                    .record(read_started.elapsed());
                            }
                            let apply_started = stage_counters.as_ref().map(|_| Instant::now());
                            let applied = store
                                .apply_block(block_id, &payload)
                                .map_err(|error| error.to_string())?;
                            if let (Some(stage_counters), Some(apply_started)) =
                                (&stage_counters, apply_started)
                            {
                                stage_counters
                                    .store_apply_block
                                    .record(apply_started.elapsed());
                            }
                            total.add(applied);
                            let complete_started = stage_counters.as_ref().map(|_| Instant::now());
                            completion
                                .complete(block_id)
                                .map_err(|error| error.to_string())?;
                            if let (Some(stage_counters), Some(complete_started)) =
                                (&stage_counters, complete_started)
                            {
                                stage_counters
                                    .store_complete_block
                                    .record(complete_started.elapsed());
                            }
                            if let Some(block_timeline) = &block_timeline {
                                block_timeline.record_store_applied(block_id);
                            }
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
                failure_guard.disarm();
                Ok::<DrainStats, String>(total)
            }));
        }

        let mut client_handles = Vec::with_capacity(client_driver_threads);
        if multiplexed_clients {
            #[derive(Clone, Copy)]
            struct AsyncOutstanding {
                append_sequence: u64,
                intent: WriteIntent,
                started: Option<Instant>,
                logged_at: Option<Instant>,
                block_prefix: u64,
                store_applied_at: Option<Instant>,
            }

            struct AsyncLogicalClient {
                first: u64,
                count: u64,
                next_offset: u64,
                ready_started: Option<Instant>,
                outstanding: Option<AsyncOutstanding>,
            }

            for driver_id in 0..client_driver_threads {
                let queue = Arc::clone(&queue);
                let completion = completion.as_ref().map(Arc::clone);
                let durable_prefix = Arc::clone(&durable_prefix);
                let enqueue_tail = Arc::clone(&enqueue_tail);
                let barrier = Arc::clone(&barrier);
                let go = Arc::clone(&go);
                let run_failed = Arc::clone(&run_failed);
                let driver_clients = clients / client_driver_threads
                    + usize::from(driver_id < clients % client_driver_threads);
                let first_client = (clients / client_driver_threads) * driver_id
                    + (clients % client_driver_threads).min(driver_id);
                let issue_budget = opts.client_driver_issue_budget;
                client_handles.push(std::thread::spawn(move || {
                    let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                    let mut logical_clients = Vec::with_capacity(driver_clients);
                    let mut total_assigned = 0_u64;
                    for client_id in first_client..first_client + driver_clients {
                        let count = base + u64::from((client_id as u64) < rem);
                        let first = base * client_id as u64 + rem.min(client_id as u64);
                        total_assigned += count;
                        logical_clients.push(AsyncLogicalClient {
                            first,
                            count,
                            next_offset: 0,
                            ready_started: None,
                            outstanding: None,
                        });
                    }
                    let sample_capacity = total_assigned
                        .div_ceil(sample_stride)
                        .try_into()
                        .unwrap_or(usize::MAX)
                        .saturating_add(driver_clients);
                    let mut latencies = ClientLatencySamples::with_capacity(sample_capacity);
                    let mut total = DrainStats::default();
                    let mut completed = 0_u64;
                    let mut next_issue_index = 0_usize;
                    barrier.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    let ready_at = Instant::now();
                    for client in logical_clients.iter_mut() {
                        if client.next_offset < client.count {
                            let client_seq = client.first + client.next_offset;
                            client.ready_started =
                                client_seq.is_multiple_of(sample_stride).then_some(ready_at);
                        }
                    }
                    while completed < total_assigned {
                        if run_failed.load(Ordering::Acquire) {
                            return Err("coalesced async client driver aborted after peer failure"
                                .to_string());
                        }
                        let mut progressed = false;

                        let mut issued = 0_usize;
                        for _ in 0..logical_clients.len() {
                            if issued >= issue_budget {
                                break;
                            }
                            let client_index = next_issue_index;
                            next_issue_index = (next_issue_index + 1) % logical_clients.len();
                            let client = &mut logical_clients[client_index];
                            if client.outstanding.is_some() || client.next_offset >= client.count {
                                continue;
                            }
                            let client_seq = client.first + client.next_offset;
                            let intent = intent_for(client_seq);
                            let started = client.ready_started.take();
                            let append_sequence = enqueue_tail.fetch_add(1, Ordering::Relaxed);
                            if append_sequence >= total_events as u64 {
                                return Err("coalesced async client enqueue exceeded event count"
                                    .to_string());
                            }
                            let slot = &queue[(append_sequence & append_ring_mask) as usize];
                            slot.wait_free_and_publish(
                                append_sequence,
                                intent,
                                client.next_offset + 1 == client.count,
                                &run_failed,
                            )?;
                            client.next_offset += 1;
                            client.outstanding = Some(AsyncOutstanding {
                                append_sequence,
                                intent,
                                started,
                                logged_at: None,
                                block_prefix: 0,
                                store_applied_at: None,
                            });
                            issued += 1;
                            progressed = true;
                        }

                        for client in logical_clients.iter_mut() {
                            let Some(outstanding) = client.outstanding.as_mut() else {
                                continue;
                            };
                            let slot =
                                &queue[(outstanding.append_sequence & append_ring_mask) as usize];
                            if outstanding.block_prefix == 0 {
                                let Some(block_prefix) = slot.try_logged_block() else {
                                    continue;
                                };
                                outstanding.block_prefix = block_prefix;
                                outstanding.logged_at = outstanding.started.map(|_| Instant::now());
                                total.observe(outstanding.intent);
                                progressed = true;
                            }
                            if let Some(completion) = completion.as_ref() {
                                if outstanding.store_applied_at.is_none() {
                                    if completion.completed_prefix() < outstanding.block_prefix {
                                        continue;
                                    }
                                    outstanding.store_applied_at =
                                        outstanding.started.map(|_| Instant::now());
                                    progressed = true;
                                }
                            }
                            if ack_mode.needs_durable()
                                && durable_prefix.load(Ordering::Acquire) < outstanding.block_prefix
                            {
                                continue;
                            }

                            let acked_at = Instant::now();
                            if let (Some(started), Some(logged_at)) =
                                (outstanding.started, outstanding.logged_at)
                            {
                                latencies
                                    .client_to_logged
                                    .push(duration_ns(logged_at.duration_since(started)));
                                if completion.is_some() {
                                    latencies
                                        .logged_to_ack
                                        .push(duration_ns(acked_at.duration_since(logged_at)));
                                }
                                if let Some(store_applied_at) = outstanding.store_applied_at {
                                    if ack_mode.needs_durable() {
                                        latencies.store_to_ack.push(duration_ns(
                                            acked_at.duration_since(store_applied_at),
                                        ));
                                    }
                                }
                                latencies
                                    .client_to_ack
                                    .push(duration_ns(acked_at.duration_since(started)));
                            }
                            slot.release(outstanding.append_sequence, append_ring_capacity as u64);
                            client.outstanding = None;
                            completed += 1;
                            if client.next_offset < client.count {
                                let client_seq = client.first + client.next_offset;
                                client.ready_started =
                                    client_seq.is_multiple_of(sample_stride).then_some(acked_at);
                            }
                            progressed = true;
                        }

                        if !progressed {
                            std::thread::yield_now();
                        }
                    }
                    failure_guard.disarm();
                    Ok::<(DrainStats, ClientLatencySamples), String>((total, latencies))
                }));
            }
        } else {
            for client_id in 0..clients {
                let queue = Arc::clone(&queue);
                let completion = completion.as_ref().map(Arc::clone);
                let durable_prefix = Arc::clone(&durable_prefix);
                let enqueue_tail = Arc::clone(&enqueue_tail);
                let barrier = Arc::clone(&barrier);
                let go = Arc::clone(&go);
                let run_failed = Arc::clone(&run_failed);
                let count = base + u64::from((client_id as u64) < rem);
                let first = base * client_id as u64 + rem.min(client_id as u64);
                client_handles.push(std::thread::spawn(move || {
                    let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                    let sample_capacity =
                        (count.div_ceil(sample_stride) as usize).saturating_add(1);
                    let mut latencies = ClientLatencySamples::with_capacity(sample_capacity);
                    let mut total = DrainStats::default();
                    barrier.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    for offset in 0..count {
                        if run_failed.load(Ordering::Acquire) {
                            return Err("coalesced client aborted after peer failure".to_string());
                        }
                        let client_seq = first + offset;
                        let sampled = client_seq.is_multiple_of(sample_stride);
                        let intent = intent_for(client_seq);
                        let started = sampled.then(Instant::now);
                        let append_sequence = enqueue_tail.fetch_add(1, Ordering::Relaxed);
                        if append_sequence >= total_events as u64 {
                            return Err("coalesced client enqueue exceeded event count".to_string());
                        }
                        let slot = &queue[(append_sequence & append_ring_mask) as usize];
                        slot.wait_free_and_publish(
                            append_sequence,
                            intent,
                            offset + 1 == count,
                            &run_failed,
                        )?;
                        let block_prefix = slot.wait_logged_block(&run_failed)?;
                        let logged_at = sampled.then(Instant::now);
                        total.observe(intent);
                        let mut store_applied_at = None;
                        if let Some(completion) = completion.as_ref() {
                            wait_for_completion_or_failure(
                                completion,
                                block_prefix,
                                opts.wait_spins,
                                &run_failed,
                            )?;
                            store_applied_at = sampled.then(Instant::now);
                        }
                        if ack_mode.needs_durable() {
                            while durable_prefix.load(Ordering::Acquire) < block_prefix {
                                if run_failed.load(Ordering::Acquire) {
                                    return Err(
                                        "coalesced durable WAL worker failed before acknowledging request"
                                            .to_string(),
                                    );
                                }
                                std::thread::yield_now();
                            }
                        }
                        slot.release(append_sequence, append_ring_capacity as u64);
                        if let (Some(started), Some(logged_at)) = (started, logged_at) {
                            let acked_at = Instant::now();
                            latencies
                                .client_to_logged
                                .push(duration_ns(logged_at.duration_since(started)));
                            if completion.is_some() {
                                latencies
                                    .logged_to_ack
                                    .push(duration_ns(acked_at.duration_since(logged_at)));
                            }
                            if let Some(store_applied_at) = store_applied_at {
                                if ack_mode.needs_durable() {
                                    latencies.store_to_ack.push(duration_ns(
                                        acked_at.duration_since(store_applied_at),
                                    ));
                                }
                            }
                            latencies
                                .client_to_ack
                                .push(duration_ns(acked_at.duration_since(started)));
                        }
                    }
                    failure_guard.disarm();
                    Ok::<(DrainStats, ClientLatencySamples), String>((total, latencies))
                }));
            }
        }

        barrier.wait();
        let started = Instant::now();
        if let Some(block_timeline) = &block_timeline {
            block_timeline.set_epoch(started);
        }
        go.store(true, Ordering::Release);
        let mut first_error = None::<String>;
        let mut client_total = DrainStats::default();
        let mut samples = ClientLatencySamples::default();
        for handle in client_handles {
            match handle.join() {
                Ok(Ok((thread_total, thread_samples))) => {
                    client_total.add(thread_total);
                    samples.append(thread_samples);
                }
                Ok(Err(message)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(message);
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some("coalesced client thread panicked".to_string());
                    }
                }
            }
        }

        let mut worker_total = DrainStats::default();
        for handle in worker_handles {
            match handle.join() {
                Ok(Ok(thread_total)) => worker_total.add(thread_total),
                Ok(Err(message)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(message);
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some("coalesced apply worker thread panicked".to_string());
                    }
                }
            }
        }

        let appender_total = match appender_handle.join() {
            Ok(Ok(thread_total)) => thread_total,
            Ok(Err(message)) => {
                run_failed.store(true, Ordering::Release);
                if first_error.is_none() {
                    first_error = Some(message);
                }
                DrainStats::default()
            }
            Err(_) => {
                run_failed.store(true, Ordering::Release);
                if first_error.is_none() {
                    first_error = Some("coalesced WAL appender thread panicked".to_string());
                }
                DrainStats::default()
            }
        };
        visible_elapsed = started.elapsed();
        visible_cut_snapshot = ChronicleCutSnapshot {
            published: published_blocks.load(Ordering::Acquire),
            store_applied: completion
                .as_ref()
                .map(|completion| completion.completed_prefix()),
            requested_durable: durable_enabled
                .then(|| requested_durable_prefix.load(Ordering::Acquire)),
            durable: durable_enabled.then(|| durable_prefix.load(Ordering::Acquire)),
        };

        if let Some(handle) = prewrite_handle {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(error.to_string());
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error =
                            Some("coalesced WAL prewrite worker thread panicked".to_string());
                    }
                }
            }
        }

        for handle in durable_handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(error.to_string());
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error =
                            Some("coalesced durable WAL worker thread panicked".to_string());
                    }
                }
            }
        }
        if let Some(message) = first_error {
            return Err(std::io::Error::other(message).into());
        }

        total_elapsed = started.elapsed();
        durable_catchup_elapsed = total_elapsed.saturating_sub(visible_elapsed);
        elapsed = total_elapsed;
        actual_blocks = published_blocks.load(Ordering::Acquire) as usize;
        final_cut_snapshot = ChronicleCutSnapshot {
            published: actual_blocks as u64,
            store_applied: completion
                .as_ref()
                .map(|completion| completion.completed_prefix()),
            requested_durable: durable_enabled
                .then(|| requested_durable_prefix.load(Ordering::Acquire)),
            durable: durable_enabled.then(|| durable_prefix.load(Ordering::Acquire)),
        };
        durable_sync_snapshot = durable_syncs
            .as_ref()
            .map(|counters| counters.snapshot())
            .unwrap_or_default();
        durable_sync_timing_report = durable_syncs
            .as_ref()
            .map(|counters| counters.timing_report());
        prewrite_snapshot = prewrite_counters
            .as_ref()
            .map(|counters| counters.snapshot())
            .unwrap_or_default();
        total = if ack_mode.needs_store() {
            worker_total
        } else {
            appender_total
        };
        if client_total != appender_total {
            return Err(format!(
                "coalesced client/appender stats mismatch: client={client_total:?}, appender={appender_total:?}"
            )
            .into());
        }
        report = samples
            .report(requested_samples, sample_stride)
            .ok_or("coalesced client latency sampling produced no samples")?;
        let validate_started = Instant::now();
        if let Some(store) = store.as_ref() {
            let observed = store
                .validate_applied_blocks(actual_blocks as u64)
                .ok_or("coalesced client store validation could not read applied prefix")?;
            let expected = stats_for_range(0, events);
            if observed != expected {
                return Err(format!(
                    "coalesced client store validation mismatch: got {observed:?}, expected {expected:?}"
                )
                .into());
            }
        }
        store_validation_elapsed = validate_started.elapsed();
    }

    assert_expected_stats(coalesced_client_label(ack_mode, wal_backend), total, events)?;
    if durable_enabled {
        let recovery_started = Instant::now();
        let expected = stats_for_range(0, events);
        match wal_backend {
            CoalescedWalBackend::Segment => {
                let recovered = recover_wal_segment_by_scan(file_opts.path)?;
                recovery_elapsed = recovery_started.elapsed();
                if recovered.recovered_blocks != actual_blocks as u64
                    || recovered.recovered_records != events
                    || recovered.stats != expected
                {
                    return Err(format!(
                        "coalesced client durable recovery mismatch: recovered {recovered:?}, expected_blocks={actual_blocks}, expected_stats={expected:?}"
                    )
                    .into());
                }
            }
            CoalescedWalBackend::Manager => {
                let configs = manager_configs
                    .as_ref()
                    .expect("manager configs for manager durable recovery");
                if durable_lane_count == 1 {
                    let recovered = recover_wal_manager_by_scan(&configs[0])?;
                    recovery_elapsed = recovery_started.elapsed();
                    let blocks_per_segment =
                        opts.manager_records_per_segment.div_ceil(block_size) as u64;
                    let recovered_blocks = OpenShardAppendStore::global_block_id(
                        recovered.durable_segment_id,
                        blocks_per_segment,
                        recovered.durable_blocks,
                    );
                    if recovered_blocks != actual_blocks as u64
                        || recovered.recovered_records != events
                        || recovered.stats != expected
                    {
                        return Err(format!(
                            "manager coalesced durable recovery mismatch: recovered {recovered:?}, recovered_blocks={recovered_blocks}, expected_blocks={actual_blocks}, expected_stats={expected:?}"
                        )
                        .into());
                    }
                } else {
                    let mut recovered_records = 0_u64;
                    let mut recovered_stats = DrainStats::default();
                    let mut recovered_lane_block_counts = Vec::with_capacity(configs.len());
                    for (lane_id, config) in configs.iter().enumerate() {
                        let recovered = recover_wal_manager_by_scan(config)?;
                        let blocks_per_segment =
                            opts.manager_records_per_segment.div_ceil(block_size) as u64;
                        let recovered_lane_block_count = OpenShardAppendStore::global_block_id(
                            recovered.durable_segment_id,
                            blocks_per_segment,
                            recovered.durable_blocks,
                        );
                        let expected_lane_blocks =
                            lane_block_count(lane_id, actual_blocks as u64, durable_lane_count);
                        if recovered_lane_block_count != expected_lane_blocks {
                            return Err(format!(
                                "striped manager durable recovery lane {lane_id} mismatch: recovered {recovered:?}, recovered_lane_blocks={recovered_lane_block_count}, expected_lane_blocks={expected_lane_blocks}"
                            )
                            .into());
                        }
                        recovered_lane_block_counts.push(recovered_lane_block_count);
                        recovered_records += recovered.recovered_records;
                        recovered_stats.add(recovered.stats);
                    }
                    let recovered_global_prefix =
                        striped_global_prefix_from_lane_counts(&recovered_lane_block_counts);
                    recovery_elapsed = recovery_started.elapsed();
                    if recovered_global_prefix != actual_blocks as u64
                        || recovered_records != events
                        || recovered_stats != expected
                    {
                        return Err(format!(
                            "striped manager durable recovery mismatch: recovered_global_prefix={recovered_global_prefix}, expected_blocks={actual_blocks}, recovered_records={recovered_records}, expected_records={events}, recovered_stats={recovered_stats:?}, expected_stats={expected:?}"
                        )
                        .into());
                    }
                }
            }
        }
    }

    let avg_block_records = events as f64 / actual_blocks.max(1) as f64;
    let store_validation = if ack_mode.needs_store() {
        format!(
            " store-validate={:.3}s",
            store_validation_elapsed.as_secs_f64()
        )
    } else {
        String::new()
    };
    let durable_min_blocks_label = if opts.durable_min_blocks == 0 {
        "auto".to_string()
    } else {
        opts.durable_min_blocks.to_string()
    };
    let prewrite_extra = if prewrite_worker_count != 0 {
        prewrite_snapshot.format()
    } else {
        String::new()
    };
    let block_budget_label = if sparse_closed_loop_budget {
        "sparse-safe"
    } else {
        "compact"
    };
    let backend_label = match wal_backend {
        CoalescedWalBackend::Segment => "segment",
        CoalescedWalBackend::Manager => "manager",
    };
    let manager_extra = if wal_backend == CoalescedWalBackend::Manager {
        format!(
            " manager-records/segment={}",
            opts.manager_records_per_segment
        )
    } else {
        String::new()
    };
    let durable_lane_extra = if wal_backend == CoalescedWalBackend::Manager && durable_enabled {
        format!(" durable-lanes={durable_lane_count}")
    } else {
        String::new()
    };
    let client_driver_label = if multiplexed_clients {
        format!(
            "async:{client_driver_threads}/issue={}",
            opts.client_driver_issue_budget
        )
    } else {
        "thread-per-client".to_string()
    };
    let cut_extra = if durable_enabled && !ack_mode.needs_durable() {
        format!(
            "{}{}",
            visible_cut_snapshot.format_with_label("chronicle-visible-cuts"),
            final_cut_snapshot.format_with_label("chronicle-final-cuts"),
        )
    } else {
        final_cut_snapshot.format_with_label("chronicle-cuts")
    };
    let background_durable_extra = if opts.background_durable && !ack_mode.needs_durable() {
        let visible_secs = visible_elapsed.as_secs_f64();
        let visible_mps = events as f64 / visible_secs / 1e6;
        let visible_ns_per_write = visible_secs * 1e9 / events as f64;
        format!(
            " background-durable=1 visible-elapsed={:.3}s visible-throughput={:.3}M/s visible-ns/write={:.2} total-with-durable={:.3}s durable-catchup={:.3}s",
            visible_secs,
            visible_mps,
            visible_ns_per_write,
            total_elapsed.as_secs_f64(),
            durable_catchup_elapsed.as_secs_f64(),
        )
    } else if opts.background_durable {
        " background-durable=redundant-with-durable-ack".to_string()
    } else {
        String::new()
    };
    let extra = format!(
        " setup={:.3}s clients={clients} client-drivers={client_driver_label} workers={worker_count} wal-backend={backend_label} wal-block={block_size}{manager_extra}{durable_lane_extra} max-blocks={max_blocks} block-budget={block_budget_label} min-budget={minimum_block_budget} partial-slack={partial_block_slack} blocks={actual_blocks} avg-block={avg_block_records:.1} append-ring={append_ring_capacity} append-group={}us min-block={effective_min_records_per_block} requested-min-block={min_records_per_block} wait-spins={wait_spins} ack={ack}{cuts}{background_durable_extra}{store_validation}{recovery}",
        setup_elapsed.as_secs_f64(),
        opts.append_group_us,
        cuts = cut_extra,
        wait_spins = opts.wait_spins,
        ack = match ack_mode {
            ClientAckMode::Logged => "coalesced-non-durable-logged",
            ClientAckMode::StoreApplied => "coalesced-non-durable-store-applied",
            ClientAckMode::DurableStoreApplied =>
                "coalesced-data-fenced-wal+volatile-store-applied",
        },
        recovery = if durable_enabled {
            format!(
                " durable-group={}us durable-min-blocks={} durable-sync-mode={}{}{} recover-final-durable={:.3}s",
                opts.durable_group_us,
                durable_min_blocks_label,
                durable_sync_mode_label(opts.durable_sync_mode),
                prewrite_extra,
                durable_sync_snapshot.format(),
                recovery_elapsed.as_secs_f64()
            )
        } else {
            String::new()
        }
    );
    report_extra(
        coalesced_client_label(ack_mode, wal_backend),
        events,
        elapsed,
        total,
        &extra,
    );
    report.print(ack_mode);
    if let Some(block_timeline) = &block_timeline {
        if let Some(report) = block_timeline.report(actual_blocks as u64) {
            report.print();
        }
    }
    if let Some(stage_counters) = &stage_counters {
        stage_counters.print();
    }
    if let Some(report) = &durable_sync_timing_report {
        report.print();
    }
    Ok(())
}

fn coalesced_client_label(
    ack_mode: ClientAckMode,
    wal_backend: CoalescedWalBackend,
) -> &'static str {
    match (wal_backend, ack_mode) {
        (CoalescedWalBackend::Segment, ClientAckMode::Logged) => "file-wal-client-coalesced-logged",
        (CoalescedWalBackend::Segment, ClientAckMode::StoreApplied) => {
            "file-wal-client-coalesced-store-applied"
        }
        (CoalescedWalBackend::Segment, ClientAckMode::DurableStoreApplied) => {
            "file-wal-client-coalesced-durable"
        }
        (CoalescedWalBackend::Manager, ClientAckMode::Logged) => {
            "file-wal-manager-coalesced-logged"
        }
        (CoalescedWalBackend::Manager, ClientAckMode::StoreApplied) => {
            "file-wal-manager-coalesced-store-applied"
        }
        (CoalescedWalBackend::Manager, ClientAckMode::DurableStoreApplied) => {
            "file-wal-manager-coalesced-durable"
        }
    }
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
            let segment = unwrap_wal_arc(segment)?;
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
