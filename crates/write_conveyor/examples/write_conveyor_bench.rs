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
        self.sync_ns.checked_div(self.sync_calls).unwrap_or(0)
    }

    fn avg_wait_ns(self) -> u64 {
        self.wait_ns.checked_div(self.sync_calls).unwrap_or(0)
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
        self.write_ns.checked_div(self.write_calls).unwrap_or(0)
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

mod write_conveyor_bench {
    pub(super) mod coalesced_client_latency;
    pub(super) mod direct_client_latency;
}

use write_conveyor_bench::{
    coalesced_client_latency::run_file_wal_client_coalesced_latency,
    direct_client_latency::run_file_wal_client_latency,
};

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
