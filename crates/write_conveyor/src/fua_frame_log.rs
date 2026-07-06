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

struct FrameSlot {
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
            let frame = loop {
                let claimed = self.fence_cursor.load_acquire();
                let published = self.published_frames.load_acquire();
                if published > claimed {
                    if self.fence_cursor.compare_exchange(claimed, claimed + 1) {
                        break claimed;
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
            if let Err(error) = self.fence_frame(frame) {
                self.fence_failed.store(true, Ordering::Release);
                // Wake everyone so sibling lanes observe the failure/finish
                // promptly instead of parking forever.
                self.park_wake.notify_all();
                return Err(error);
            }
            fenced += 1;
        }
    }

    /// Wake fence lanes after state they wait on changed (a publish or
    /// finish). One frame needs one lane; `finish`/failure wake all.
    fn wake_fence_lanes(&self, all: bool) {
        // The mutex bounds the race with a parking lane (it re-checks under
        // the lock before waiting); an EMPTY critical section is enough.
        let parked = self.park.lock().unwrap_or_else(|p| p.into_inner());
        if *parked > 0 {
            if all {
                self.park_wake.notify_all();
            } else {
                self.park_wake.notify_one();
            }
        }
    }

    fn fence_frame(&self, frame_id: u64) -> std::io::Result<()> {
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
        slot.fenced.store(true, Ordering::Release);
        let published = self.publish_ns[(frame_id as usize) % FRAME_SLOTS].load(Ordering::Relaxed);
        let now = self.stat_base.elapsed().as_nanos() as u64;
        self.stat_fence_ns
            .fetch_add(now.saturating_sub(published), Ordering::Relaxed);
        self.stat_fenced_frames.fetch_add(1, Ordering::Relaxed);
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
            *frontier = advanced;
            self.durable_frames.store_release(advanced);
            let end_seq = self.slots[((advanced - 1) as usize) % FRAME_SLOTS]
                .end_seq
                .load(Ordering::Relaxed);
            self.durable_seq.store_release(end_seq);
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
        let log = &*self.log;
        if payload.is_empty() || payload.len() > u32::MAX as usize {
            return Err(invalid_data("FUA frame payload must be 1..=u32::MAX bytes"));
        }
        let padded = padded_frame_bytes(payload.len());
        if self.next_offset + padded > log.capacity {
            return Err(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                "FUA frame log segment is full; roll to the next segment",
            ));
        }
        let frame_id = self.next_frame;
        if frame_id >= log.durable_frames.load_acquire() + FRAME_SLOTS as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "FUA frame log slot ring exhausted; pace on free_fence_slots",
            ));
        }
        let mut header = FrameHeader {
            magic: FRAME_MAGIC,
            frame_id,
            first_seq,
            reserved0: 0,
            epoch: log.epoch,
            payload_bytes: payload.len() as u32,
            seq_count,
            payload_crc: crc32c::crc32c(payload),
            header_crc: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
        };
        header.header_crc = header_crc(&header);
        unsafe {
            let base = log.staging.ptr().add(self.next_offset);
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
        log.publish_ns[(frame_id as usize) % FRAME_SLOTS]
            .store(log.stat_base.elapsed().as_nanos() as u64, Ordering::Relaxed);
        slot.fenced.store(false, Ordering::Relaxed);
        slot.offset
            .store(self.next_offset as u64, Ordering::Relaxed);
        slot.padded_len.store(padded as u64, Ordering::Relaxed);
        slot.end_seq
            .store(first_seq + seq_count as u64, Ordering::Relaxed);
        self.next_offset += padded;
        self.next_frame = frame_id + 1;
        log.published_frames.store_release(self.next_frame);
        log.wake_fence_lanes(false);
        Ok(FrameHandle {
            frame_id,
            last_seq: first_seq + seq_count as u64,
        })
    }

    /// Declare publishing finished so fence lanes can drain and exit.
    pub fn finish(self) {
        self.log.publishing_finished.store(true, Ordering::Release);
        self.log.wake_fence_lanes(true);
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
        frames.push(RecoveredFrame {
            frame_id: header.frame_id,
            first_seq: header.first_seq,
            seq_count: header.seq_count,
            payload,
        });
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
            log.fence_frame(1).expect("fence");
            assert_eq!(log.durable_frames(), 0);
            log.fence_frame(0).expect("fence");
            assert_eq!(log.durable_frames(), 2);
            assert_eq!(log.durable_seq(), 20);
            log.fence_frame(2).expect("fence");
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
}
