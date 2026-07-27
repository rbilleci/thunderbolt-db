//! Variable-payload FUA frame log — the engine-facing durability backend.
//!
//! Same physics and scheduling contract as [`crate::FuaWalSegment`] (pre-written
//! recycled extents, anonymous staging, FUA write-through fence pool, contiguous
//! durable cut, epoch-stamped recycle safety), generalized from fixed 64B
//! `WriteIntent` blocks to OPAQUE VARIABLE-LENGTH payloads so the engine's
//! `WalRecord` encodings (SQL text, binary row-ops, DDL) fit one totally
//! ordered log. Frames are 512B-aligned: header (64B) + payload + zero pad.
//!
//! Charter note: this is control-plane host work by design — WAL/durability
//! I/O is one of the host's enumerated jobs. The data-plane stages behind the
//! durable cut (validate/apply/index/visibility) belong on the GPU.

use std::fs::OpenOptions;
use std::mem::size_of;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::fua_wal::{fua_open_direct, prewrite_extents, AlignedStaging, PaddedAtomicU64};
use crate::wal_segment::{bytes_of, invalid_data, read_struct_at, write_all_at};

const FRAME_LOG_MAGIC: u64 = 0x4655_4146_4c4f_4731; // "FUAFLOG1"
const FRAME_MAGIC: u64 = 0x4655_4146_524d_4831; // "FUAFRMH1"
const FRAME_LOG_VERSION: u32 = 1;
// 4KiB, not 512: XFS serializes O_DIRECT writes that are not aligned to the
// FILESYSTEM block (4KiB) on the exclusive inode lock, collapsing the fence
// pool to near-serial throughput (measured: 3.5K fences/s at qd=16 with 512B
// alignment vs ~28K at 4KiB). Sub-4KiB payloads pay pad, which engine group
// frames amortize.
const FRAME_ALIGN: usize = 4096;
pub(crate) const FRAME_LOG_HEADER_BYTES: usize = 4096;
const FRAME_HEADER_BYTES: usize = 64;
/// Fixed actual-depth buckets sampled immediately before each frame publication:
/// `1`, `2`, `3..=4`, `5..=8`, `9..=16`, `17..=32`, and `33+`.
///
/// This is deliberately a small aggregate, rather than a per-frame trace. It makes a physical
/// QD=1 run distinguishable from QD=16 without adding allocation, locking, or a timestamp vector
/// to the appender hot path.
pub const FUA_IN_FLIGHT_DEPTH_BUCKETS: usize = 7;

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
struct FrameLogFileHeader {
    magic: u64,
    version: u32,
    header_bytes: u32,
    segment_id: u64,
    capacity_bytes: u64,
    reserved: [u64; 4],
}

// Layout is PADDING-FREE by construction (four u64s then eight u32s = 64B
// exactly): the header CRC covers the raw bytes, and implicit padding would
// be nondeterministic across stack copies.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
struct FrameHeader {
    magic: u64,
    frame_id: u64,
    /// First logical sequence covered by this frame — opaque to the log,
    /// used by callers to map the durable cut back to commits.
    first_seq: u64,
    reserved0: u64,
    /// Recycle epoch = segment id low 32 bits (non-zero; see config docs).
    epoch: u32,
    payload_bytes: u32,
    seq_count: u32,
    payload_crc: u32,
    /// CRC over every prior header field; makes garbage headers unambiguous.
    header_crc: u32,
    reserved1: u32,
    reserved2: u32,
    reserved3: u32,
}

fn header_crc(header: &FrameHeader) -> u32 {
    let mut copy = *header;
    copy.header_crc = 0;
    crc32c::crc32c(bytes_of(&copy))
}

fn padded_frame_bytes(payload_bytes: usize) -> usize {
    (FRAME_HEADER_BYTES + payload_bytes).div_ceil(FRAME_ALIGN) * FRAME_ALIGN
}

/// Physical bytes occupied by one frame carrying `payload_bytes`, including its header and
/// alignment padding. WAL aggregates use this as the no-fragmentation baseline.
pub fn fua_frame_padded_bytes(payload_bytes: usize) -> usize {
    padded_frame_bytes(payload_bytes)
}

/// Configuration for a FUA frame log segment.
#[derive(Clone, Debug)]
pub struct FuaFrameLogConfig {
    pub path: PathBuf,
    /// Segment id; low 32 bits are the recycle epoch (must be non-zero, and
    /// successive lives of one physical file must not reuse a low-32 epoch —
    /// same contract as `FuaWalSegmentConfig::segment_id`).
    pub segment_id: u64,
    /// Data capacity in bytes (excluding the 4KiB file header); rounded up to
    /// the 512B frame alignment.
    pub capacity_bytes: usize,
}

impl FuaFrameLogConfig {
    fn validate(&self) -> std::io::Result<(usize, usize)> {
        if self.segment_id as u32 == 0 {
            return Err(invalid_data(
                "FUA frame log segment id low 32 bits must be non-zero (epoch 0 is reserved)",
            ));
        }
        if self.capacity_bytes == 0 {
            return Err(invalid_data("FUA frame log capacity must be non-zero"));
        }
        let capacity = self.capacity_bytes.div_ceil(FRAME_ALIGN) * FRAME_ALIGN;
        let file_bytes = FRAME_LOG_HEADER_BYTES
            .checked_add(capacity)
            .ok_or_else(|| invalid_data("FUA frame log size overflow"))?;
        Ok((capacity, file_bytes))
    }
}

/// One published frame's placement, as returned by the appender and consumed
/// by fence lanes and durable-cut waiters.
#[derive(Clone, Copy, Debug)]
pub struct FrameHandle {
    pub frame_id: u64,
    pub last_seq: u64,
}

/// CRC-covered physical metadata for a fragmented logical group. All-zero reserved header fields
/// mean a legacy single-frame payload; they remain wire-compatible and are never reinterpreted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameGroupMetadata {
    pub total_payload_bytes: u64,
    pub fragment_index: u32,
    pub fragment_count: u32,
    pub group_crc32c: u32,
}

/// Placement of one atomically published physical group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameBatchHandle {
    pub first_frame_id: u64,
    pub terminal_frame_id: u64,
    pub last_seq: u64,
}

fn frame_group_metadata(header: &FrameHeader) -> Option<FrameGroupMetadata> {
    let all_zero = header.reserved0 == 0
        && header.reserved1 == 0
        && header.reserved2 == 0
        && header.reserved3 == 0;
    (!all_zero).then_some(FrameGroupMetadata {
        total_payload_bytes: header.reserved0,
        fragment_index: header.reserved1,
        fragment_count: header.reserved2,
        group_crc32c: header.reserved3,
    })
}

/// Monotonic aggregate attribution for one live FUA frame-log segment.
///
/// Every duration is a saturated sum of successful frames in nanoseconds. The boundaries are
/// `published_frames` release -> successful fence claim -> successful `O_DIRECT|O_DSYNC` write
/// return -> contiguous durable-cut publication. The depth histogram is sampled once for every
/// published frame using the buckets documented by [`FUA_IN_FLIGHT_DEPTH_BUCKETS`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FuaFrameLogTelemetry {
    pub published_frames: u64,
    pub fenced_frames: u64,
    pub fence_failures: u64,
    pub payload_bytes: u64,
    pub padded_bytes: u64,
    pub stage_copy_nanos: u64,
    pub stage_copy_frames: u64,
    pub publish_to_claim_nanos: u64,
    pub publish_to_claim_frames: u64,
    pub claim_to_write_done_nanos: u64,
    pub claim_to_write_done_frames: u64,
    pub write_done_to_contiguous_cut_nanos: u64,
    pub write_done_to_contiguous_cut_frames: u64,
    pub contiguous_cut_events: u64,
    pub contiguous_cut_advanced_frames: u64,
    pub contiguous_cut_advance_max_frames: u64,
    pub in_flight_depth_max: u64,
    pub in_flight_depth_histogram: [u64; FUA_IN_FLIGHT_DEPTH_BUCKETS],
}

impl FuaFrameLogTelemetry {
    /// Saturating aggregate addition used by WAL segment-chain snapshots.
    pub fn saturating_add_assign(&mut self, other: Self) {
        self.published_frames = self.published_frames.saturating_add(other.published_frames);
        self.fenced_frames = self.fenced_frames.saturating_add(other.fenced_frames);
        self.fence_failures = self.fence_failures.saturating_add(other.fence_failures);
        self.payload_bytes = self.payload_bytes.saturating_add(other.payload_bytes);
        self.padded_bytes = self.padded_bytes.saturating_add(other.padded_bytes);
        self.stage_copy_nanos = self.stage_copy_nanos.saturating_add(other.stage_copy_nanos);
        self.stage_copy_frames = self
            .stage_copy_frames
            .saturating_add(other.stage_copy_frames);
        self.publish_to_claim_nanos = self
            .publish_to_claim_nanos
            .saturating_add(other.publish_to_claim_nanos);
        self.publish_to_claim_frames = self
            .publish_to_claim_frames
            .saturating_add(other.publish_to_claim_frames);
        self.claim_to_write_done_nanos = self
            .claim_to_write_done_nanos
            .saturating_add(other.claim_to_write_done_nanos);
        self.claim_to_write_done_frames = self
            .claim_to_write_done_frames
            .saturating_add(other.claim_to_write_done_frames);
        self.write_done_to_contiguous_cut_nanos = self
            .write_done_to_contiguous_cut_nanos
            .saturating_add(other.write_done_to_contiguous_cut_nanos);
        self.write_done_to_contiguous_cut_frames = self
            .write_done_to_contiguous_cut_frames
            .saturating_add(other.write_done_to_contiguous_cut_frames);
        self.contiguous_cut_events = self
            .contiguous_cut_events
            .saturating_add(other.contiguous_cut_events);
        self.contiguous_cut_advanced_frames = self
            .contiguous_cut_advanced_frames
            .saturating_add(other.contiguous_cut_advanced_frames);
        self.contiguous_cut_advance_max_frames = self
            .contiguous_cut_advance_max_frames
            .max(other.contiguous_cut_advance_max_frames);
        self.in_flight_depth_max = self.in_flight_depth_max.max(other.in_flight_depth_max);
        for (total, value) in self
            .in_flight_depth_histogram
            .iter_mut()
            .zip(other.in_flight_depth_histogram)
        {
            *total = total.saturating_add(value);
        }
    }
}

fn saturating_add(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}

struct FrameSlot {
    /// Generation guard for bounded-ring telemetry readers. A delayed reader verifies this exact
    /// id before and after loading timestamps so a reused slot yields `None`, never another
    /// frame's direct-write service time.
    frame_id: AtomicU64,
    /// Data-region offset of the frame (not including the file header).
    offset: AtomicU64,
    /// Padded on-disk length of the frame.
    padded_len: AtomicU64,
    /// `first_seq + seq_count` for the frame.
    end_seq: AtomicU64,
    fenced: AtomicBool,
}

/// A variable-payload recoverable log whose durability is a pipelined FUA
/// fence pool. Exactly one appender; fence lanes via [`Self::spawn_fence_pool`].
pub struct FuaFrameLog {
    file: std::fs::File,
    staging: AlignedStaging,
    segment_id: u64,
    epoch: u32,
    capacity: usize,
    // frame slots are a bounded ring: a slot is reused only after its frame
    // is durable AND the caller advanced past it, enforced by pacing
    slots: Vec<FrameSlot>,
    published_frames: PaddedAtomicU64,
    appender_taken: AtomicBool,
    fence_cursor: PaddedAtomicU64,
    durable_frontier: Mutex<u64>,
    durable_frames: PaddedAtomicU64,
    durable_seq: PaddedAtomicU64,
    fences_completed: PaddedAtomicU64,
    publishing_finished: AtomicBool,
    fence_failed: AtomicBool,
    /// PARK-WHEN-IDLE for fence lanes: idle lanes previously `yield_now`-spun
    /// waiting for frames — with many lane sets (N logs x 16 lanes) the
    /// spinning threads starve the whole host at low load (measured: 160
    /// threads -> 6x ack inflation). Lanes spin briefly, then park here; the
    /// appender wakes ONE parked lane per publish (any lane can serve any
    /// frame — claims are CAS'd after wake, never pre-assigned). The hot path
    /// (frames arriving within the spin window) never touches the mutex.
    park: Mutex<usize>,
    park_wake: std::sync::Condvar,
    /// Latency attribution (cheap always-on aggregates): per-frame
    /// publish->fence-done nanos summed + frame count, and the same for
    /// fence-done->frontier-advance. Splits the ack path's WAL share into
    /// DRIVE latency vs pipeline discovery.
    stat_fence_ns: AtomicU64,
    stat_fenced_frames: AtomicU64,
    /// Publish instants ring (nanos from `stat_base`), indexed like `slots`.
    publish_ns: Vec<AtomicU64>,
    /// Fence-claim instants, retained in the same bounded ring for benchmark-local service
    /// percentiles. Production telemetry consumes only the aggregate claim->write sum.
    claim_ns: Vec<AtomicU64>,
    /// Successful direct-write completion instants, indexed like `slots`. A slot is not reused
    /// until its previous frame is inside the durable prefix, so an advancing cut can safely
    /// charge every newly-covered frame's write->cut lag exactly once.
    write_done_ns: Vec<AtomicU64>,
    /// Timestamp of the most recently published contiguous durable cut. WAL waiters use this to
    /// attribute cut->observation without allocating one timestamp per waiter.
    durable_cut_ns: AtomicU64,
    stat_payload_bytes: AtomicU64,
    stat_padded_bytes: AtomicU64,
    stat_stage_copy_ns: AtomicU64,
    stat_stage_copy_frames: AtomicU64,
    stat_fence_failures: AtomicU64,
    stat_publish_to_claim_ns: AtomicU64,
    stat_publish_to_claim_frames: AtomicU64,
    stat_claim_to_write_done_ns: AtomicU64,
    stat_claim_to_write_done_frames: AtomicU64,
    stat_write_done_to_cut_ns: AtomicU64,
    stat_write_done_to_cut_frames: AtomicU64,
    stat_cut_events: AtomicU64,
    stat_cut_advanced_frames: AtomicU64,
    stat_cut_advance_max_frames: AtomicU64,
    stat_in_flight_depth_max: AtomicU64,
    stat_in_flight_depth_histogram: [AtomicU64; FUA_IN_FLIGHT_DEPTH_BUCKETS],
    stat_base: std::time::Instant,
}

const FRAME_SLOTS: usize = 4096; // bounds in-flight frames; pacing keeps use << this

impl FuaFrameLog {
    /// Create a new frame-log file with WRITTEN extents and a durable header.
    ///
    /// # Safety
    /// Caller must own the path exclusively for the log's lifetime.
    pub unsafe fn create(config: FuaFrameLogConfig) -> std::io::Result<Arc<Self>> {
        let (capacity, file_bytes) = config.validate()?;
        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&config.path)?;
            file.set_len(file_bytes as u64)?;
        }
        prewrite_extents(&config.path, file_bytes as u64)?;
        crate::sync_parent_dir(&config.path)?;
        // Safety: exclusive ownership passed through from the caller.
        unsafe { Self::open_prepared(config, capacity) }
    }

    /// Reuse an existing pre-written file under a NEW segment id (epoch bump);
    /// geometry must match. Same safety contract as `create`.
    ///
    /// # Safety
    /// Caller must own the path exclusively and have retired the previous life.
    pub unsafe fn recycle(config: FuaFrameLogConfig) -> std::io::Result<Arc<Self>> {
        let (capacity, file_bytes) = config.validate()?;
        let metadata = std::fs::metadata(&config.path)?;
        if metadata.len() != file_bytes as u64 {
            return Err(invalid_data(
                "FUA frame log recycle geometry mismatch: file length differs from config",
            ));
        }
        // Safety: exclusive ownership passed through from the caller.
        unsafe { Self::open_prepared(config, capacity) }
    }

    /// # Safety
    /// File exists with validated geometry; caller owns it exclusively.
    unsafe fn open_prepared(
        config: FuaFrameLogConfig,
        capacity: usize,
    ) -> std::io::Result<Arc<Self>> {
        let file = fua_open_direct(&config.path)?;
        let staging = AlignedStaging::zeroed(capacity)?;
        let log = Self {
            file,
            staging,
            segment_id: config.segment_id,
            epoch: config.segment_id as u32,
            capacity,
            slots: (0..FRAME_SLOTS)
                .map(|_| FrameSlot {
                    frame_id: AtomicU64::new(u64::MAX),
                    offset: AtomicU64::new(0),
                    padded_len: AtomicU64::new(0),
                    end_seq: AtomicU64::new(0),
                    fenced: AtomicBool::new(false),
                })
                .collect(),
            published_frames: PaddedAtomicU64::zero(),
            park: Mutex::new(0),
            park_wake: std::sync::Condvar::new(),
            stat_fence_ns: AtomicU64::new(0),
            stat_fenced_frames: AtomicU64::new(0),
            publish_ns: (0..FRAME_SLOTS).map(|_| AtomicU64::new(0)).collect(),
            claim_ns: (0..FRAME_SLOTS).map(|_| AtomicU64::new(0)).collect(),
            write_done_ns: (0..FRAME_SLOTS).map(|_| AtomicU64::new(0)).collect(),
            durable_cut_ns: AtomicU64::new(0),
            stat_payload_bytes: AtomicU64::new(0),
            stat_padded_bytes: AtomicU64::new(0),
            stat_stage_copy_ns: AtomicU64::new(0),
            stat_stage_copy_frames: AtomicU64::new(0),
            stat_fence_failures: AtomicU64::new(0),
            stat_publish_to_claim_ns: AtomicU64::new(0),
            stat_publish_to_claim_frames: AtomicU64::new(0),
            stat_claim_to_write_done_ns: AtomicU64::new(0),
            stat_claim_to_write_done_frames: AtomicU64::new(0),
            stat_write_done_to_cut_ns: AtomicU64::new(0),
            stat_write_done_to_cut_frames: AtomicU64::new(0),
            stat_cut_events: AtomicU64::new(0),
            stat_cut_advanced_frames: AtomicU64::new(0),
            stat_cut_advance_max_frames: AtomicU64::new(0),
            stat_in_flight_depth_max: AtomicU64::new(0),
            stat_in_flight_depth_histogram: std::array::from_fn(|_| AtomicU64::new(0)),
            stat_base: std::time::Instant::now(),
            appender_taken: AtomicBool::new(false),
            fence_cursor: PaddedAtomicU64::zero(),
            durable_frontier: Mutex::new(0),
            durable_frames: PaddedAtomicU64::zero(),
            durable_seq: PaddedAtomicU64::zero(),
            fences_completed: PaddedAtomicU64::zero(),
            publishing_finished: AtomicBool::new(false),
            fence_failed: AtomicBool::new(false),
        };
        log.write_file_header()?;
        Ok(Arc::new(log))
    }

    fn write_file_header(&self) -> std::io::Result<()> {
        let header = FrameLogFileHeader {
            magic: FRAME_LOG_MAGIC,
            version: FRAME_LOG_VERSION,
            header_bytes: FRAME_LOG_HEADER_BYTES as u32,
            segment_id: self.segment_id,
            capacity_bytes: self.capacity as u64,
            reserved: [0; 4],
        };
        let buffer = AlignedStaging::zeroed(FRAME_LOG_HEADER_BYTES)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes_of(&header).as_ptr(),
                buffer.ptr(),
                size_of::<FrameLogFileHeader>(),
            );
            write_all_at(
                &self.file,
                std::slice::from_raw_parts(buffer.ptr(), FRAME_LOG_HEADER_BYTES),
                0,
            )
        }
    }

    /// Claim the single appender handle. Panics if claimed twice.
    pub fn appender(self: &Arc<Self>) -> FuaFrameLogAppender {
        assert!(
            !self.appender_taken.swap(true, Ordering::AcqRel),
            "FUA frame log supports exactly one appender"
        );
        FuaFrameLogAppender {
            log: Arc::clone(self),
            next_frame: 0,
            next_offset: 0,
        }
    }

    /// Spawn fence lanes; lanes exit after [`FuaFrameLogAppender::finish`]
    /// once every published frame is durable.
    pub fn spawn_fence_pool(self: &Arc<Self>, lanes: usize) -> FuaFrameLogFencePool {
        let lanes = lanes.max(1);
        let handles = (0..lanes)
            .map(|_| {
                let log = Arc::clone(self);
                std::thread::spawn(move || log.fence_lane_loop())
            })
            .collect();
        FuaFrameLogFencePool { handles }
    }

    fn fence_lane_loop(&self) -> std::io::Result<u64> {
        // Spin briefly before parking: at high rates the next frame lands
        // within the window and the mutex is never touched.
        const SPINS_BEFORE_PARK: u32 = 2_000;
        let mut fenced = 0_u64;
        loop {
            // WAIT for an unclaimed published frame, then CAS-claim it. Claims
            // are taken only when work exists, so any parked lane can serve
            // any frame and a single `notify_one` per publish suffices (no
            // pre-assigned frame = no missed-wake hang, no thundering herd).
            let (frame, claim_ns) = loop {
                let claimed = self.fence_cursor.load_acquire();
                let published = self.published_frames.load_acquire();
                if published > claimed {
                    if self.fence_cursor.compare_exchange(claimed, claimed + 1) {
                        let claim_ns = self.stat_now_nanos();
                        self.claim_ns[(claimed as usize) % FRAME_SLOTS]
                            .store(claim_ns, Ordering::Relaxed);
                        let published_ns = self.publish_ns[(claimed as usize) % FRAME_SLOTS]
                            .load(Ordering::Relaxed);
                        saturating_add(
                            &self.stat_publish_to_claim_ns,
                            claim_ns.saturating_sub(published_ns),
                        );
                        saturating_add(&self.stat_publish_to_claim_frames, 1);
                        break (claimed, claim_ns);
                    }
                    continue; // lost the claim race; re-check immediately
                }
                if self.publishing_finished.load(Ordering::Acquire) {
                    return Ok(fenced);
                }
                let mut spins = 0_u32;
                let should_park = loop {
                    let published = self.published_frames.load_acquire();
                    if published > self.fence_cursor.load_acquire()
                        || self.publishing_finished.load(Ordering::Acquire)
                    {
                        break false;
                    }
                    spins += 1;
                    if spins >= SPINS_BEFORE_PARK {
                        break true;
                    }
                    std::thread::yield_now();
                };
                if should_park {
                    let mut parked = self.park.lock().unwrap_or_else(|p| p.into_inner());
                    // Re-check UNDER the mutex (the publisher notifies under
                    // it) so a publish between our check and the wait cannot
                    // be missed.
                    if self.published_frames.load_acquire() <= self.fence_cursor.load_acquire()
                        && !self.publishing_finished.load(Ordering::Acquire)
                    {
                        *parked += 1;
                        parked = self
                            .park_wake
                            .wait(parked)
                            .unwrap_or_else(|p| p.into_inner());
                        *parked = parked.saturating_sub(1);
                    }
                }
            };
            if let Err(error) = self.fence_frame(frame, claim_ns) {
                saturating_add(&self.stat_fence_failures, 1);
                self.fence_failed.store(true, Ordering::Release);
                // Wake everyone so sibling lanes observe the failure/finish
                // promptly instead of parking forever.
                self.park_wake.notify_all();
                return Err(error);
            }
            fenced += 1;
        }
    }

    /// Wake fence lanes after state they wait on changed. A batch exposes several independent
    /// fenceable frames at one release point, so wake up to that many parked lanes; `all` is for
    /// finish/failure draining.
    fn wake_fence_lanes(&self, frames: usize, all: bool) {
        // The mutex bounds the race with a parking lane (it re-checks under
        // the lock before waiting); an EMPTY critical section is enough.
        let parked = self.park.lock().unwrap_or_else(|p| p.into_inner());
        if *parked > 0 {
            if all {
                self.park_wake.notify_all();
            } else {
                for _ in 0..frames.min(*parked) {
                    self.park_wake.notify_one();
                }
            }
        }
    }

    fn fence_frame(&self, frame_id: u64, claim_ns: u64) -> std::io::Result<()> {
        let slot = &self.slots[(frame_id as usize) % FRAME_SLOTS];
        let offset = slot.offset.load(Ordering::Acquire) as usize;
        let padded_len = slot.padded_len.load(Ordering::Acquire) as usize;
        unsafe {
            write_all_at(
                &self.file,
                std::slice::from_raw_parts(self.staging.ptr().add(offset), padded_len),
                (FRAME_LOG_HEADER_BYTES + offset) as u64,
            )?;
        }
        let write_done_ns = self.stat_now_nanos();
        self.write_done_ns[(frame_id as usize) % FRAME_SLOTS]
            .store(write_done_ns, Ordering::Relaxed);
        saturating_add(
            &self.stat_claim_to_write_done_ns,
            write_done_ns.saturating_sub(claim_ns),
        );
        saturating_add(&self.stat_claim_to_write_done_frames, 1);
        slot.fenced.store(true, Ordering::Release);
        let published = self.publish_ns[(frame_id as usize) % FRAME_SLOTS].load(Ordering::Relaxed);
        saturating_add(&self.stat_fence_ns, write_done_ns.saturating_sub(published));
        saturating_add(&self.stat_fenced_frames, 1);
        self.fences_completed.fetch_add(1);
        let mut frontier = self
            .durable_frontier
            .lock()
            .map_err(|_| std::io::Error::other("FUA frame log frontier poisoned"))?;
        let mut advanced = *frontier;
        while advanced < self.published_frames.load_acquire() {
            let slot = &self.slots[(advanced as usize) % FRAME_SLOTS];
            if !slot.fenced.load(Ordering::Acquire) {
                break;
            }
            advanced += 1;
        }
        if advanced != *frontier {
            let previous = *frontier;
            let cut_ns = self.stat_now_nanos();
            for cut_frame in previous..advanced {
                let write_done_ns =
                    self.write_done_ns[(cut_frame as usize) % FRAME_SLOTS].load(Ordering::Relaxed);
                saturating_add(
                    &self.stat_write_done_to_cut_ns,
                    cut_ns.saturating_sub(write_done_ns),
                );
            }
            let advanced_frames = advanced.saturating_sub(previous);
            saturating_add(&self.stat_write_done_to_cut_frames, advanced_frames);
            saturating_add(&self.stat_cut_events, 1);
            saturating_add(&self.stat_cut_advanced_frames, advanced_frames);
            self.stat_cut_advance_max_frames
                .fetch_max(advanced_frames, Ordering::Relaxed);
            *frontier = advanced;
            // Publish the timestamp before the cut itself; an acquire load of durable_seq then
            // observes the cut time without a waiter-side event allocation.
            self.durable_cut_ns
                .store(cut_ns.saturating_add(1), Ordering::Relaxed);
            self.durable_frames.store_release(advanced);
            let end_seq = self.slots[((advanced - 1) as usize) % FRAME_SLOTS]
                .end_seq
                .load(Ordering::Relaxed);
            // Continuation fragments carry `seq_count=0` and therefore their group start as
            // `end_seq`; never let a durable cut regress (or acknowledge past) the preceding
            // terminal group's logical boundary. Only the terminal fragment advances it.
            let prior_end = self.durable_seq.load_relaxed();
            self.durable_seq.store_release(prior_end.max(end_seq));
        }
        Ok(())
    }

    /// True once any fence lane failed; pacing loops and waiters must abort.
    pub fn fence_failed(&self) -> bool {
        self.fence_failed.load(Ordering::Acquire)
    }

    /// Contiguous durable frame prefix.
    pub fn durable_frames(&self) -> u64 {
        self.durable_frames.load_acquire()
    }

    /// Logical sequences covered by the durable cut (seq < durable_seq is
    /// durable) — the engine's commit-visibility gate.
    pub fn durable_seq(&self) -> u64 {
        self.durable_seq.load_acquire()
    }

    /// Aggregate publish->fence-done latency: (total ns, fenced frames).
    pub fn fence_latency_stats(&self) -> (u64, u64) {
        (
            self.stat_fence_ns.load(Ordering::Relaxed),
            self.stat_fenced_frames.load(Ordering::Relaxed),
        )
    }

    /// Snapshot the permanent aggregate FUA attribution for this segment. Relaxed loads are
    /// intentional: this is observability, not a recovery or acknowledgement authority.
    pub fn telemetry(&self) -> FuaFrameLogTelemetry {
        FuaFrameLogTelemetry {
            published_frames: self.published_frames.load_acquire(),
            fenced_frames: self.stat_fenced_frames.load(Ordering::Relaxed),
            fence_failures: self.stat_fence_failures.load(Ordering::Relaxed),
            payload_bytes: self.stat_payload_bytes.load(Ordering::Relaxed),
            padded_bytes: self.stat_padded_bytes.load(Ordering::Relaxed),
            stage_copy_nanos: self.stat_stage_copy_ns.load(Ordering::Relaxed),
            stage_copy_frames: self.stat_stage_copy_frames.load(Ordering::Relaxed),
            publish_to_claim_nanos: self.stat_publish_to_claim_ns.load(Ordering::Relaxed),
            publish_to_claim_frames: self.stat_publish_to_claim_frames.load(Ordering::Relaxed),
            claim_to_write_done_nanos: self.stat_claim_to_write_done_ns.load(Ordering::Relaxed),
            claim_to_write_done_frames: self
                .stat_claim_to_write_done_frames
                .load(Ordering::Relaxed),
            write_done_to_contiguous_cut_nanos: self
                .stat_write_done_to_cut_ns
                .load(Ordering::Relaxed),
            write_done_to_contiguous_cut_frames: self
                .stat_write_done_to_cut_frames
                .load(Ordering::Relaxed),
            contiguous_cut_events: self.stat_cut_events.load(Ordering::Relaxed),
            contiguous_cut_advanced_frames: self.stat_cut_advanced_frames.load(Ordering::Relaxed),
            contiguous_cut_advance_max_frames: self
                .stat_cut_advance_max_frames
                .load(Ordering::Relaxed),
            in_flight_depth_max: self.stat_in_flight_depth_max.load(Ordering::Relaxed),
            in_flight_depth_histogram: std::array::from_fn(|index| {
                self.stat_in_flight_depth_histogram[index].load(Ordering::Relaxed)
            }),
        }
    }

    /// Return the elapsed time from the last durable-cut publication to this waiter observation.
    /// `None` means this segment has not yet published any durable cut.
    pub fn durable_cut_to_observe_nanos(&self) -> Option<u64> {
        let cut_ns = self.durable_cut_ns.load(Ordering::Relaxed);
        (cut_ns != 0).then(|| self.stat_now_nanos().saturating_sub(cut_ns - 1))
    }

    /// Successful direct-write service time for one recently completed frame. This bounded-ring
    /// inspection generation-checks the slot before and after timestamp loads; a delayed reader
    /// that races reuse gets `None` rather than a stale service sample.
    pub fn frame_direct_write_service_nanos(&self, frame_id: u64) -> Option<u64> {
        let index = (frame_id as usize) % FRAME_SLOTS;
        let slot = &self.slots[index];
        if slot.frame_id.load(Ordering::Acquire) != frame_id || !slot.fenced.load(Ordering::Acquire)
        {
            return None;
        }
        let claim_ns = self.claim_ns[index].load(Ordering::Relaxed);
        let write_done_ns = self.write_done_ns[index].load(Ordering::Relaxed);
        (slot.frame_id.load(Ordering::Acquire) == frame_id)
            .then(|| write_done_ns.saturating_sub(claim_ns))
    }

    pub fn published_frames(&self) -> u64 {
        self.published_frames.load_acquire()
    }

    /// Pacing gate: free lanes in a pool of `lanes`, measured against total
    /// completions (order-independent), same law as the fixed-record segment.
    pub fn free_fence_slots(&self, lanes: usize) -> usize {
        let in_flight = self
            .published_frames
            .load_relaxed()
            .saturating_sub(self.fences_completed.load_acquire());
        (lanes as u64).saturating_sub(in_flight) as usize
    }

    fn stat_now_nanos(&self) -> u64 {
        self.stat_base
            .elapsed()
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX)
    }

    fn record_in_flight_depth(&self, depth: u64) {
        let depth = depth.max(1);
        self.stat_in_flight_depth_max
            .fetch_max(depth, Ordering::Relaxed);
        let bucket = match depth {
            1 => 0,
            2 => 1,
            3..=4 => 2,
            5..=8 => 3,
            9..=16 => 4,
            17..=32 => 5,
            _ => 6,
        };
        saturating_add(&self.stat_in_flight_depth_histogram[bucket], 1);
    }
}

/// The single publishing handle for a [`FuaFrameLog`].
pub struct FuaFrameLogAppender {
    log: Arc<FuaFrameLog>,
    next_frame: u64,
    next_offset: usize,
}

impl FuaFrameLogAppender {
    /// Stage one frame (header + payload + zero pad, epoch stamped, CRCs) and
    /// publish it to the fence pool. `first_seq`/`seq_count` are the logical
    /// commit sequences the frame covers; the durable cut reports them back
    /// through [`FuaFrameLog::durable_seq`]. No file I/O on this path.
    ///
    /// Errors with `StorageFull` when the segment cannot hold the frame —
    /// callers roll to a fresh (recycled) segment. `WouldBlock` is returned
    /// if more than `FRAME_SLOTS` frames would be unfenced at once; pace on
    /// [`FuaFrameLog::free_fence_slots`] and this never happens.
    pub fn publish_frame(
        &mut self,
        payload: &[u8],
        first_seq: u64,
        seq_count: u32,
    ) -> std::io::Result<FrameHandle> {
        let batch = self.publish_internal(&[payload], first_seq, seq_count, None)?;
        Ok(FrameHandle {
            frame_id: batch.terminal_frame_id,
            last_seq: batch.last_seq,
        })
    }

    /// Atomically stage and expose a fragmented physical representation of one logical group.
    /// No fragment becomes visible to fence lanes unless all chunks, slots, and the complete
    /// padded extent preflight successfully. Every fragment has the same `first_seq`; only the
    /// terminal fragment carries `seq_count`, so no durable cut can acknowledge a partial group.
    pub fn publish_batch(
        &mut self,
        chunks: &[&[u8]],
        first_seq: u64,
        seq_count: u32,
    ) -> std::io::Result<FrameBatchHandle> {
        if chunks.len() < 2 {
            return Err(invalid_data(
                "FUA fragmented batch must contain at least two nonempty chunks",
            ));
        }
        let total_payload_bytes = chunks.iter().try_fold(0_u64, |total, chunk| {
            total
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| invalid_data("FUA fragmented batch payload length overflow"))
        })?;
        let group_crc32c = chunks
            .iter()
            .fold(0_u32, |crc, chunk| crc32c::crc32c_append(crc, chunk));
        self.publish_internal(
            chunks,
            first_seq,
            seq_count,
            Some(FrameGroupMetadata {
                total_payload_bytes,
                fragment_index: 0,
                fragment_count: u32::try_from(chunks.len())
                    .map_err(|_| invalid_data("FUA fragmented batch has too many chunks"))?,
                group_crc32c,
            }),
        )
    }

    /// Preflight a physical group without writing staging memory or publishing a frame. This is
    /// intentionally the same complete-extent/ring check used by [`Self::publish_batch`].
    pub fn can_publish_batch(&self, chunks: &[&[u8]]) -> std::io::Result<()> {
        self.preflight(chunks).map(|_| ())
    }

    fn preflight(&self, chunks: &[&[u8]]) -> std::io::Result<(usize, usize)> {
        if chunks.is_empty() {
            return Err(invalid_data("FUA batch must contain at least one chunk"));
        }
        let mut total_padded = 0usize;
        let mut total_payload = 0usize;
        for chunk in chunks {
            if chunk.is_empty() || chunk.len() > u32::MAX as usize {
                return Err(invalid_data("FUA frame payload must be 1..=u32::MAX bytes"));
            }
            total_padded = total_padded
                .checked_add(padded_frame_bytes(chunk.len()))
                .ok_or_else(|| invalid_data("FUA batch padded length overflow"))?;
            total_payload = total_payload
                .checked_add(chunk.len())
                .ok_or_else(|| invalid_data("FUA batch payload length overflow"))?;
        }
        if self
            .next_offset
            .checked_add(total_padded)
            .is_none_or(|end| end > self.log.capacity)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                "FUA frame log segment is full; roll before publishing the complete batch",
            ));
        }
        let terminal = self
            .next_frame
            .checked_add(chunks.len() as u64)
            .ok_or_else(|| invalid_data("FUA frame id overflow"))?;
        if terminal > self.log.durable_frames.load_acquire() + FRAME_SLOTS as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "FUA frame log slot ring lacks capacity for the complete batch",
            ));
        }
        Ok((total_payload, total_padded))
    }

    fn publish_internal(
        &mut self,
        chunks: &[&[u8]],
        first_seq: u64,
        seq_count: u32,
        group: Option<FrameGroupMetadata>,
    ) -> std::io::Result<FrameBatchHandle> {
        if seq_count == 0 {
            return Err(invalid_data("FUA frame group seq_count must be non-zero"));
        }
        let last_seq = first_seq
            .checked_add(u64::from(seq_count))
            .ok_or_else(|| invalid_data("FUA frame group sequence range overflow"))?;
        let (total_payload, total_padded) = self.preflight(chunks)?;
        let log = &*self.log;
        let stage_copy_started = log.stat_now_nanos();
        let first_frame_id = self.next_frame;
        let mut offset = self.next_offset;
        for (index, payload) in chunks.iter().enumerate() {
            let frame_id = first_frame_id + index as u64;
            let padded = padded_frame_bytes(payload.len());
            let terminal = index + 1 == chunks.len();
            let mut header = FrameHeader {
                magic: FRAME_MAGIC,
                frame_id,
                first_seq,
                reserved0: group.map_or(0, |metadata| metadata.total_payload_bytes),
                epoch: log.epoch,
                payload_bytes: payload.len() as u32,
                seq_count: u32::from(terminal).saturating_mul(seq_count),
                payload_crc: crc32c::crc32c(payload),
                header_crc: 0,
                reserved1: group.map_or(0, |_| index as u32),
                reserved2: group.map_or(0, |metadata| metadata.fragment_count),
                reserved3: group.map_or(0, |metadata| metadata.group_crc32c),
            };
            header.header_crc = header_crc(&header);
            unsafe {
                let base = log.staging.ptr().add(offset);
                std::ptr::copy_nonoverlapping(bytes_of(&header).as_ptr(), base, FRAME_HEADER_BYTES);
                std::ptr::copy_nonoverlapping(
                    payload.as_ptr(),
                    base.add(FRAME_HEADER_BYTES),
                    payload.len(),
                );
                let used = FRAME_HEADER_BYTES + payload.len();
                if padded > used {
                    std::ptr::write_bytes(base.add(used), 0, padded - used);
                }
            }
            let slot = &log.slots[(frame_id as usize) % FRAME_SLOTS];
            let publish_ns = log.stat_now_nanos();
            log.publish_ns[(frame_id as usize) % FRAME_SLOTS].store(publish_ns, Ordering::Relaxed);
            slot.fenced.store(false, Ordering::Relaxed);
            slot.frame_id.store(frame_id, Ordering::Release);
            slot.offset.store(offset as u64, Ordering::Relaxed);
            slot.padded_len.store(padded as u64, Ordering::Relaxed);
            slot.end_seq.store(
                if terminal { last_seq } else { first_seq },
                Ordering::Relaxed,
            );
            offset += padded;
        }
        self.next_offset = offset;
        self.next_frame = first_frame_id + chunks.len() as u64;
        let completed = log.fences_completed.load_acquire();
        for frame_id in first_frame_id..self.next_frame {
            log.record_in_flight_depth(frame_id.saturating_add(1).saturating_sub(completed));
        }
        saturating_add(&log.stat_payload_bytes, total_payload as u64);
        saturating_add(&log.stat_padded_bytes, total_padded as u64);
        saturating_add(
            &log.stat_stage_copy_ns,
            log.stat_now_nanos().saturating_sub(stage_copy_started),
        );
        saturating_add(&log.stat_stage_copy_frames, chunks.len() as u64);
        // This is the sole visibility/claim publication for the complete group. All slots and
        // staging bytes above happen-before this release store, so a fence lane cannot claim a
        // prefix of the group.
        log.published_frames.store_release(self.next_frame);
        log.wake_fence_lanes(chunks.len(), false);
        Ok(FrameBatchHandle {
            first_frame_id,
            terminal_frame_id: self.next_frame - 1,
            last_seq,
        })
    }

    /// Declare publishing finished so fence lanes can drain and exit.
    pub fn finish(self) {
        self.log.publishing_finished.store(true, Ordering::Release);
        self.log.wake_fence_lanes(0, true);
    }
}

/// Handle for a frame log's fence threads.
pub struct FuaFrameLogFencePool {
    handles: Vec<JoinHandle<std::io::Result<u64>>>,
}

impl FuaFrameLogFencePool {
    /// Wait for every published frame to become durable and lanes to exit
    /// (requires the appender's `finish`). Returns total fences issued.
    pub fn join(self) -> std::io::Result<u64> {
        let mut fences = 0_u64;
        for handle in self.handles {
            fences += handle
                .join()
                .map_err(|_| std::io::Error::other("FUA frame log fence lane panicked"))??;
        }
        Ok(fences)
    }
}

/// One recovered frame: logical placement plus its payload bytes.
#[derive(Clone, Debug)]
pub struct RecoveredFrame {
    pub frame_id: u64,
    pub first_seq: u64,
    pub seq_count: u32,
    pub payload: Vec<u8>,
    /// `Some` only for a complete, scan-validated fragmented group. The generic scanner retains
    /// physical frame visibility, but incomplete or malformed metadata groups contribute no
    /// frames to its result.
    pub group: Option<FrameGroupMetadata>,
}

/// Scan recovery: walk the frame chain from the data region start, stopping
/// at the first frame that fails magic/epoch/id/CRC validation. Returns the
/// contiguous valid prefix — the durable cut as recoverable from disk.
pub fn recover_frame_log_by_scan(
    path: impl AsRef<std::path::Path>,
) -> std::io::Result<Vec<RecoveredFrame>> {
    use std::io::Read;
    use std::io::Seek;
    let mut file = std::fs::File::open(path)?;
    let file_header: FrameLogFileHeader = read_struct_at(&mut file, 0)?;
    if file_header.magic != FRAME_LOG_MAGIC
        || file_header.version != FRAME_LOG_VERSION
        || file_header.header_bytes as usize != FRAME_LOG_HEADER_BYTES
        || file_header.capacity_bytes == 0
        || file_header.segment_id as u32 == 0
    {
        return Err(invalid_data("invalid FUA frame log header"));
    }
    let expected_epoch = file_header.segment_id as u32;
    let capacity = file_header.capacity_bytes;
    let mut frames = Vec::new();
    struct PendingGroup {
        metadata: FrameGroupMetadata,
        first_seq: u64,
        next_index: u32,
        payload_bytes: u64,
        group_crc32c: u32,
        frames: Vec<RecoveredFrame>,
    }
    let mut pending: Option<PendingGroup> = None;
    let mut offset = 0_u64;
    let mut expected_frame = 0_u64;
    while offset + FRAME_HEADER_BYTES as u64 <= capacity {
        let header: FrameHeader =
            read_struct_at(&mut file, FRAME_LOG_HEADER_BYTES as u64 + offset)?;
        if header.magic != FRAME_MAGIC
            || header.epoch != expected_epoch
            || header.frame_id != expected_frame
            || header.header_crc != header_crc(&header)
            || header.payload_bytes == 0
        {
            break;
        }
        let padded = padded_frame_bytes(header.payload_bytes as usize) as u64;
        if offset + padded > capacity {
            break;
        }
        let mut payload = vec![0_u8; header.payload_bytes as usize];
        file.seek(std::io::SeekFrom::Start(
            FRAME_LOG_HEADER_BYTES as u64 + offset + FRAME_HEADER_BYTES as u64,
        ))?;
        file.read_exact(&mut payload)?;
        if crc32c::crc32c(&payload) != header.payload_crc {
            break;
        }
        let recovered = RecoveredFrame {
            frame_id: header.frame_id,
            first_seq: header.first_seq,
            seq_count: header.seq_count,
            payload,
            group: frame_group_metadata(&header),
        };
        match recovered.group {
            None => {
                // A legacy frame cannot complete or follow an incomplete fragmented group; the
                // entire partial group is outside the recoverable prefix. A zero-count legacy
                // frame is never valid: only metadata-marked continuations use `seq_count=0`,
                // so accepting it would let a crafted old-format payload bypass that law.
                if pending.is_some() || recovered.seq_count == 0 {
                    break;
                }
                frames.push(recovered);
            }
            Some(metadata) => {
                if metadata.total_payload_bytes == 0
                    || metadata.fragment_count < 2
                    || metadata.fragment_index >= metadata.fragment_count
                {
                    break;
                }
                let terminal = metadata.fragment_index + 1 == metadata.fragment_count;
                if (terminal && recovered.seq_count == 0) || (!terminal && recovered.seq_count != 0)
                {
                    break;
                }
                if let Some(group) = pending.as_mut() {
                    if metadata.total_payload_bytes != group.metadata.total_payload_bytes
                        || metadata.fragment_count != group.metadata.fragment_count
                        || metadata.group_crc32c != group.metadata.group_crc32c
                        || metadata.fragment_index != group.next_index
                        || recovered.first_seq != group.first_seq
                    {
                        break;
                    }
                    group.next_index = group.next_index.saturating_add(1);
                    let Some(payload_bytes) = group
                        .payload_bytes
                        .checked_add(recovered.payload.len() as u64)
                    else {
                        break;
                    };
                    group.payload_bytes = payload_bytes;
                    group.group_crc32c =
                        crc32c::crc32c_append(group.group_crc32c, &recovered.payload);
                    group.frames.push(recovered);
                } else {
                    if metadata.fragment_index != 0 || recovered.seq_count != 0 {
                        break;
                    }
                    pending = Some(PendingGroup {
                        metadata,
                        first_seq: recovered.first_seq,
                        next_index: 1,
                        payload_bytes: recovered.payload.len() as u64,
                        group_crc32c: crc32c::crc32c(&recovered.payload),
                        frames: vec![recovered],
                    });
                }
                if terminal {
                    let complete = pending.take().expect("terminal metadata group is pending");
                    if complete.next_index != complete.metadata.fragment_count
                        || complete.payload_bytes != complete.metadata.total_payload_bytes
                        || complete.group_crc32c != complete.metadata.group_crc32c
                    {
                        break;
                    }
                    frames.extend(complete.frames);
                }
            }
        }
        offset += padded;
        expected_frame += 1;
    }
    Ok(frames)
}

/// Read a frame log's data capacity (bytes) from its file header — the geometry a reopened
/// lane continues with (disk-authoritative; a reopen must not silently change segment sizes
/// under an existing database because env defaults differ from the creating process's).
pub fn frame_log_capacity_bytes(path: impl AsRef<std::path::Path>) -> std::io::Result<u64> {
    let mut file = std::fs::File::open(path)?;
    let file_header: FrameLogFileHeader = read_struct_at(&mut file, 0)?;
    if file_header.magic != FRAME_LOG_MAGIC
        || file_header.version != FRAME_LOG_VERSION
        || file_header.header_bytes as usize != FRAME_LOG_HEADER_BYTES
        || file_header.capacity_bytes == 0
    {
        return Err(invalid_data("invalid FUA frame log header"));
    }
    Ok(file_header.capacity_bytes)
}

/// DURABLY invalidate every frame from `frame_id` on in a frame-log segment by zeroing the
/// first invalidated frame's aligned block (scan recovery validates frames in chain order and
/// stops at the first invalid header, so the whole suffix drops). Returns the number of valid
/// frames that were invalidated (0 when the chain ends before `frame_id` — nothing to do).
///
/// Safety of the CALLER's semantics, not this function's: the dropped frames' payloads are
/// destroyed. The lane-set orphan repair uses this on frames strictly ABOVE the cross-lane
/// contiguous durable cut — sequences that were never acknowledged (acks gate on cut
/// coverage), so discarding them loses nothing a client was told was durable.
pub fn invalidate_frame_log_suffix(
    path: impl AsRef<std::path::Path>,
    frame_id: u64,
) -> std::io::Result<u64> {
    let path = path.as_ref();
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let file_header: FrameLogFileHeader = read_struct_at(&mut file, 0)?;
    if file_header.magic != FRAME_LOG_MAGIC
        || file_header.version != FRAME_LOG_VERSION
        || file_header.header_bytes as usize != FRAME_LOG_HEADER_BYTES
        || file_header.capacity_bytes == 0
        || file_header.segment_id as u32 == 0
    {
        return Err(invalid_data("invalid FUA frame log header"));
    }
    let expected_epoch = file_header.segment_id as u32;
    let capacity = file_header.capacity_bytes;
    // Header-only walk of the valid frame chain (same validation as scan recovery) to find the
    // byte offset of `frame_id` and count the valid frames from there.
    let mut offset = 0_u64;
    let mut expected_frame = 0_u64;
    let mut invalidate_at: Option<u64> = None;
    let mut invalidated = 0_u64;
    while offset + FRAME_HEADER_BYTES as u64 <= capacity {
        let header: FrameHeader =
            read_struct_at(&mut file, FRAME_LOG_HEADER_BYTES as u64 + offset)?;
        if header.magic != FRAME_MAGIC
            || header.epoch != expected_epoch
            || header.frame_id != expected_frame
            || header.header_crc != header_crc(&header)
            || header.payload_bytes == 0
        {
            break;
        }
        let padded = padded_frame_bytes(header.payload_bytes as usize) as u64;
        if offset + padded > capacity {
            break;
        }
        if expected_frame == frame_id {
            invalidate_at = Some(offset);
        }
        if expected_frame >= frame_id {
            invalidated += 1;
        }
        offset += padded;
        expected_frame += 1;
    }
    let Some(invalidate_at) = invalidate_at else {
        return Ok(0); // chain ends before frame_id — nothing to invalidate
    };
    // Zero one aligned block at the first dropped frame: its header (and the chain behind it)
    // becomes unambiguous garbage to the scan.
    let zeros = vec![0_u8; FRAME_ALIGN];
    write_all_at(&file, &zeros, FRAME_LOG_HEADER_BYTES as u64 + invalidate_at)?;
    file.sync_all()?;
    Ok(invalidated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("fua-frame-log-tests");
        std::fs::create_dir_all(&dir).expect("create test dir");
        let path = dir.join(format!("{name}-{}.dat", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn config(path: &std::path::Path, segment_id: u64) -> FuaFrameLogConfig {
        FuaFrameLogConfig {
            path: path.to_path_buf(),
            segment_id,
            capacity_bytes: 1 << 20,
        }
    }

    fn payload(len: usize, fill: u8) -> Vec<u8> {
        (0..len).map(|i| fill.wrapping_add(i as u8)).collect()
    }

    #[test]
    fn variable_frames_roundtrip_and_recover() {
        let path = test_path("roundtrip");
        let log = unsafe { FuaFrameLog::create(config(&path, 9)).expect("create") };
        let pool = log.spawn_fence_pool(4);
        let mut appender = log.appender();
        let sizes = [1_usize, 63, 448, 449, 4096, 100_000, 7];
        let mut seq = 0_u64;
        for (index, size) in sizes.iter().enumerate() {
            let handle = appender
                .publish_frame(&payload(*size, index as u8), seq, 3)
                .expect("publish");
            assert_eq!(handle.frame_id, index as u64);
            seq += 3;
        }
        appender.finish();
        let fences = pool.join().expect("pool");
        assert_eq!(fences, sizes.len() as u64);
        assert_eq!(log.durable_frames(), sizes.len() as u64);
        assert_eq!(log.durable_seq(), seq);
        let telemetry = log.telemetry();
        assert_eq!(telemetry.published_frames, sizes.len() as u64);
        assert_eq!(telemetry.fenced_frames, sizes.len() as u64);
        assert_eq!(telemetry.fence_failures, 0);
        assert_eq!(telemetry.stage_copy_frames, sizes.len() as u64);
        assert_eq!(telemetry.publish_to_claim_frames, sizes.len() as u64);
        assert_eq!(telemetry.claim_to_write_done_frames, sizes.len() as u64);
        assert_eq!(
            telemetry.write_done_to_contiguous_cut_frames,
            sizes.len() as u64
        );
        assert_eq!(telemetry.contiguous_cut_advanced_frames, sizes.len() as u64);
        assert_eq!(
            telemetry.in_flight_depth_histogram.iter().sum::<u64>(),
            sizes.len() as u64
        );
        assert!(telemetry.in_flight_depth_max >= 1);
        assert_eq!(
            telemetry.payload_bytes,
            sizes.iter().map(|size| *size as u64).sum::<u64>()
        );
        assert_eq!(
            telemetry.padded_bytes,
            sizes
                .iter()
                .map(|size| fua_frame_padded_bytes(*size) as u64)
                .sum::<u64>()
        );
        drop(log);
        let frames = recover_frame_log_by_scan(&path).expect("scan");
        assert_eq!(frames.len(), sizes.len());
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(frame.frame_id, index as u64);
            assert_eq!(frame.first_seq, index as u64 * 3);
            assert_eq!(frame.seq_count, 3);
            assert_eq!(frame.payload, payload(sizes[index], index as u8));
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unfenced_tail_is_not_recovered_and_recycle_rejects_prior_epoch() {
        let path = test_path("tail-epoch");
        {
            let log = unsafe { FuaFrameLog::create(config(&path, 1)).expect("create") };
            let mut appender = log.appender();
            for index in 0..5_u64 {
                appender
                    .publish_frame(&payload(1000, index as u8), index * 10, 10)
                    .expect("publish");
            }
            // fence only 0..3 (out of order to exercise the cut)
            log.fence_frame(1, log.stat_now_nanos()).expect("fence");
            assert_eq!(log.durable_frames(), 0);
            log.fence_frame(0, log.stat_now_nanos()).expect("fence");
            assert_eq!(log.durable_frames(), 2);
            assert_eq!(log.durable_seq(), 20);
            log.fence_frame(2, log.stat_now_nanos()).expect("fence");
            assert_eq!(log.durable_frames(), 3);
            drop(appender);
        }
        let frames = recover_frame_log_by_scan(&path).expect("scan");
        assert_eq!(frames.len(), 3, "unfenced frames 3..5 must not recover");
        // recycle under a new epoch: only the new life's frames recover
        {
            let log = unsafe { FuaFrameLog::recycle(config(&path, 2)).expect("recycle") };
            let pool = log.spawn_fence_pool(2);
            let mut appender = log.appender();
            appender
                .publish_frame(&payload(64, 0xEE), 0, 1)
                .expect("publish");
            appender.finish();
            pool.join().expect("pool");
        }
        let frames = recover_frame_log_by_scan(&path).expect("scan second life");
        assert_eq!(frames.len(), 1, "previous-life frames must be rejected");
        assert_eq!(frames[0].payload, payload(64, 0xEE));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn invalidate_suffix_drops_frames_durably_and_reports_capacity() {
        let path = test_path("invalidate-suffix");
        {
            let log = unsafe { FuaFrameLog::create(config(&path, 7)).expect("create") };
            let pool = log.spawn_fence_pool(2);
            let mut appender = log.appender();
            for index in 0..6_u64 {
                appender
                    .publish_frame(&payload(700, index as u8), index * 4, 4)
                    .expect("publish");
            }
            appender.finish();
            pool.join().expect("pool");
        }
        assert_eq!(
            frame_log_capacity_bytes(&path).expect("capacity"),
            1 << 20,
            "header capacity must round-trip"
        );
        // Chain ends before the requested frame: nothing to invalidate.
        assert_eq!(
            invalidate_frame_log_suffix(&path, 6).expect("noop"),
            0,
            "no frame 6 exists"
        );
        assert_eq!(recover_frame_log_by_scan(&path).expect("scan").len(), 6);
        // Drop frames 4..6.
        assert_eq!(invalidate_frame_log_suffix(&path, 4).expect("drop"), 2);
        let frames = recover_frame_log_by_scan(&path).expect("scan after drop");
        assert_eq!(frames.len(), 4, "suffix from frame 4 must be gone");
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(frame.frame_id, index as u64);
            assert_eq!(frame.payload, payload(700, index as u8));
        }
        // Idempotent: re-invalidating the same suffix is a no-op.
        assert_eq!(invalidate_frame_log_suffix(&path, 4).expect("again"), 0);
        assert_eq!(recover_frame_log_by_scan(&path).expect("scan").len(), 4);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn config_validation_and_storage_full() {
        let path = test_path("validate");
        assert!(unsafe { FuaFrameLog::create(config(&path, 0)) }.is_err());
        assert!(unsafe { FuaFrameLog::create(config(&path, 1_u64 << 32)) }.is_err());
        let mut small = config(&path, 3);
        small.capacity_bytes = 2048;
        let log = unsafe { FuaFrameLog::create(small).expect("create") };
        let mut appender = log.appender();
        appender
            .publish_frame(&payload(1000, 1), 0, 1)
            .expect("publish fits");
        let full = appender.publish_frame(&payload(1000, 2), 1, 1);
        assert_eq!(
            full.expect_err("must be full").kind(),
            std::io::ErrorKind::StorageFull
        );
        drop(appender);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn batch_publish_is_atomic_and_terminal_owns_the_cut() {
        let path = test_path("batch-atomic-terminal");
        let log = unsafe { FuaFrameLog::create(config(&path, 11)).expect("create") };
        let mut appender = log.appender();
        let chunks = [payload(700, 1), payload(900, 2), payload(800, 3)];
        let references: Vec<_> = chunks.iter().map(Vec::as_slice).collect();
        let batch = appender
            .publish_batch(&references, 0, 5)
            .expect("batch publish");
        assert_eq!(log.published_frames(), 3);
        // A terminal write that lands first cannot advance a cut across unfenced continuations.
        log.fence_frame(batch.terminal_frame_id, log.stat_now_nanos())
            .expect("terminal fence");
        assert_eq!(log.durable_frames(), 0);
        assert_eq!(log.durable_seq(), 0);
        log.fence_frame(batch.first_frame_id, log.stat_now_nanos())
            .expect("first fence");
        assert_eq!(log.durable_frames(), 1);
        assert_eq!(log.durable_seq(), 0);
        log.fence_frame(batch.first_frame_id + 1, log.stat_now_nanos())
            .expect("middle fence");
        assert_eq!(log.durable_frames(), 3);
        assert_eq!(log.durable_seq(), 5);
        drop(appender);
        drop(log);
        let recovered = recover_frame_log_by_scan(&path).expect("scan");
        assert_eq!(recovered.len(), 3, "complete group retains physical frames");
        assert!(recovered.iter().all(|frame| frame.group.is_some()));
        assert_eq!(recovered[0].seq_count, 0);
        assert_eq!(recovered[1].seq_count, 0);
        assert_eq!(recovered[2].seq_count, 5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn incomplete_batch_recovery_keeps_only_prior_legacy_prefix() {
        let path = test_path("batch-torn-continuation");
        {
            let log = unsafe { FuaFrameLog::create(config(&path, 12)).expect("create") };
            let mut appender = log.appender();
            appender
                .publish_frame(&payload(128, 9), 0, 1)
                .expect("legacy prefix");
            log.fence_frame(0, log.stat_now_nanos())
                .expect("prefix fence");
            let chunks = [payload(700, 1), payload(700, 2), payload(700, 3)];
            let references: Vec<_> = chunks.iter().map(Vec::as_slice).collect();
            let batch = appender
                .publish_batch(&references, 1, 3)
                .expect("batch publish");
            // Persist only the first continuation then simulate a torn/unfenced tail.
            log.fence_frame(batch.first_frame_id, log.stat_now_nanos())
                .expect("continuation fence");
            drop(appender);
        }
        let recovered = recover_frame_log_by_scan(&path).expect("scan");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].seq_count, 1);
        assert!(recovered[0].group.is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn batch_preflight_storage_full_leaves_no_visible_prefix() {
        let path = test_path("batch-preflight-full");
        let mut small = config(&path, 13);
        small.capacity_bytes = 2 * FRAME_ALIGN;
        let log = unsafe { FuaFrameLog::create(small).expect("create") };
        let mut appender = log.appender();
        let chunks = [payload(100, 1), payload(100, 2), payload(100, 3)];
        let references: Vec<_> = chunks.iter().map(Vec::as_slice).collect();
        assert_eq!(
            appender
                .publish_batch(&references, 0, 1)
                .expect_err("batch cannot fit")
                .kind(),
            std::io::ErrorKind::StorageFull
        );
        assert_eq!(log.published_frames(), 0);
        appender
            .publish_frame(&payload(100, 7), 0, 1)
            .expect("cursor remains usable after preflight failure");
        assert_eq!(log.published_frames(), 1);
        let _ = std::fs::remove_file(&path);
    }
}
