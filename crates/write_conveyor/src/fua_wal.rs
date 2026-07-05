//! FUA-pipelined durable WAL segment.
//!
//! Production form of the design proven by `fua_wal_client_bench` and the raw
//! FUA probes (docs/WRITE_CONVEYOR.md, "FUA-Pipelined Durable Lane"):
//!
//! - Segment files are created with WRITTEN extents (real zeros streamed +
//!   fsync at setup) and are RECYCLED, never fallocate-only. Fallocate's
//!   unwritten extents cost an XFS extent-conversion journal force on every
//!   durable fence (~3x fence latency); recycling amortizes prep to zero.
//! - Publishing writes anonymous aligned staging, not a file mapping, so
//!   durable fences never contend with dirty page-cache writeback.
//! - Durability is a pool of fence lanes writing whole WAL frames through one
//!   `O_DIRECT|O_DSYNC` descriptor (FUA write-through). FUA writes are
//!   independent NVMe commands: unlike fdatasync's full-cache FLUSH they
//!   pipeline, and the measured device does ~28K durable 4KiB fences/s at
//!   queue depth 16 (p50 ~0.7ms) vs ~1.3K serial flushes/s.
//! - The durable cut is the contiguous prefix of completed fences. Client
//!   visibility gates on the cut, never on an out-of-order fence.
//! - Recycle safety: block headers stamp the segment epoch (segment id low 32
//!   bits) and the file header sets `WAL_SEGMENT_FLAG_EPOCH_STAMPED`, so scan
//!   recovery rejects valid-looking frames left over from the file's previous
//!   life.
//!
//! Scheduling contract for the caller (both laws are load-bearing; violating
//! either collapses the closed loop back to serial fence latency):
//! - ADAPTIVE FRAME SIZING: spread pending records across the pool
//!   (`frame ~= pending / fence_lanes`, clamped) rather than packing frames
//!   maximally.
//! - FENCE-POOL PACING: publish a frame only when `free_fence_slots() > 0`,
//!   accumulating while all lanes are busy.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::{File, OpenOptions};
use std::mem::size_of;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use super::{sync_parent_dir, WriteIntent};
use crate::wal_segment::{
    block_crc32c, bytes_of, checked_block_stride, checked_segment_bytes, invalid_data,
    write_all_at, WalBlockHeader, WalBlockTrailer, WalSegmentFileHeader, WAL_BLOCK_COMMIT_MARKER,
    WAL_BLOCK_HEADER_MAGIC, WAL_BLOCK_TRAILER_MAGIC, WAL_SEGMENT_FLAG_EPOCH_STAMPED,
    WAL_SEGMENT_HEADER_BYTES, WAL_SEGMENT_MAGIC, WAL_SEGMENT_VERSION,
};

const O_DIRECT: i32 = 0o40000;
const O_DSYNC: i32 = 0o10000;
const FUA_ALIGN: usize = 512;

/// Configuration for a FUA WAL segment.
#[derive(Clone, Debug)]
pub struct FuaWalSegmentConfig {
    pub path: PathBuf,
    /// Segment id; its LOW 32 BITS are the recycle epoch stamped into every
    /// block header. Two constraints (validated / contractual):
    /// - the low 32 bits must be NON-ZERO — epoch 0 is the legacy
    ///   (pre-epoch-stamping) sentinel, and reusing it would let a recycled
    ///   file's previous-life frames pass scan recovery;
    /// - successive lives of one physical file must not reuse a low-32 epoch,
    ///   i.e. assign ids monotonically and recycle a given file fewer than
    ///   2^32 - 1 times (managers assigning sequential ids satisfy this).
    pub segment_id: u64,
    /// Record capacity; rounded up to whole blocks.
    pub records: usize,
    /// Records per WAL block. The block stride (header + payload + trailer)
    /// must be a multiple of 512 bytes for O_DIRECT: 62 records = 4096B
    /// frames, 126 = 8192B, and generally `records % 8 == 6`.
    pub block_size: usize,
}

impl FuaWalSegmentConfig {
    fn validate(&self) -> std::io::Result<(usize, usize, usize)> {
        if self.records == 0 || self.block_size == 0 {
            return Err(invalid_data("FUA WAL segment must hold records"));
        }
        if self.segment_id as u32 == 0 {
            return Err(invalid_data(
                "FUA WAL segment id low 32 bits must be non-zero: epoch 0 is the legacy sentinel and would let a recycled file's previous-life frames pass scan recovery",
            ));
        }
        if self.block_size > u32::MAX as usize / size_of::<WriteIntent>() {
            return Err(invalid_data("FUA WAL block payload bytes must fit on disk"));
        }
        let block_stride = checked_block_stride(self.block_size)?;
        if !block_stride.is_multiple_of(FUA_ALIGN) {
            return Err(invalid_data(
                "FUA WAL block stride must be 512B-aligned; use block sizes like 62 or 126",
            ));
        }
        let block_capacity = self.records.div_ceil(self.block_size);
        if block_capacity > u32::MAX as usize {
            return Err(invalid_data("FUA WAL block capacity must fit on disk"));
        }
        let bytes = checked_segment_bytes(block_capacity, block_stride)?;
        Ok((block_capacity, block_stride, bytes))
    }
}

struct AlignedStaging {
    ptr: *mut u8,
    layout: Layout,
}
unsafe impl Send for AlignedStaging {}
unsafe impl Sync for AlignedStaging {}
impl AlignedStaging {
    fn zeroed(len: usize) -> std::io::Result<Self> {
        let layout = Layout::from_size_align(len.max(1), 4096)
            .map_err(|_| invalid_data("FUA WAL staging layout"))?;
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(std::io::Error::other("FUA WAL staging allocation failed"));
        }
        Ok(Self { ptr, layout })
    }
}
impl Drop for AlignedStaging {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

#[repr(align(128))]
struct PaddedAtomicU64(AtomicU64);

/// Optional per-block pipeline timestamps (nanoseconds from an internal base
/// `Instant`), for latency attribution: publish -> fence-start -> fence-done
/// -> durable-cut. Enabled by [`FuaWalSegment::enable_stage_timings`]; when
/// disabled the hot path pays one `Option` check per stage. Written by the
/// single appender (publish) and fence lanes (rest); read after the run.
pub struct FuaStageTimings {
    base: std::time::Instant,
    publish_ns: Vec<AtomicU64>,
    fence_start_ns: Vec<AtomicU64>,
    fence_done_ns: Vec<AtomicU64>,
    cut_ns: Vec<AtomicU64>,
}

impl FuaStageTimings {
    fn new(block_capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            base: std::time::Instant::now(),
            publish_ns: (0..block_capacity).map(|_| AtomicU64::new(0)).collect(),
            fence_start_ns: (0..block_capacity).map(|_| AtomicU64::new(0)).collect(),
            fence_done_ns: (0..block_capacity).map(|_| AtomicU64::new(0)).collect(),
            cut_ns: (0..block_capacity).map(|_| AtomicU64::new(0)).collect(),
        })
    }

    fn now_ns(&self) -> u64 {
        self.base.elapsed().as_nanos() as u64
    }

    /// (publish->fence_start, fence_start->fence_done, fence_done->cut) in
    /// nanoseconds for every block with a complete timeline.
    pub fn block_stages(&self, blocks: u64) -> Vec<(u64, u64, u64)> {
        (0..blocks as usize)
            .filter_map(|block| {
                let publish = self.publish_ns[block].load(Ordering::Relaxed);
                let start = self.fence_start_ns[block].load(Ordering::Relaxed);
                let done = self.fence_done_ns[block].load(Ordering::Relaxed);
                let cut = self.cut_ns[block].load(Ordering::Relaxed);
                (publish != 0 && start != 0 && done != 0 && cut != 0).then(|| {
                    (
                        start.saturating_sub(publish),
                        done.saturating_sub(start),
                        cut.saturating_sub(done),
                    )
                })
            })
            .collect()
    }
}

/// A recoverable WAL segment whose durability is a pipelined FUA fence pool.
///
/// Exactly one appender publishes blocks (claim it with [`Self::appender`]);
/// any number of readers may consume published blocks; fence lanes are
/// spawned with [`Self::spawn_fence_pool`].
pub struct FuaWalSegment {
    file: File,
    staging: AlignedStaging,
    segment_id: u64,
    epoch: u32,
    block_size: usize,
    block_capacity: u64,
    block_stride: usize,
    // publish frontier (contiguous; single appender)
    published_blocks: PaddedAtomicU64,
    block_end_seq: Vec<AtomicU64>,
    appender_taken: AtomicBool,
    // durable cut
    fence_cursor: PaddedAtomicU64,
    fence_completed: Vec<AtomicBool>,
    durable_frontier: Mutex<u64>,
    durable_blocks: PaddedAtomicU64,
    durable_record_seq: PaddedAtomicU64,
    /// Total completed fences, ORDER-INDEPENDENT (unlike the contiguous
    /// durable cut). This is the pacing denominator: a lane that finished an
    /// out-of-order frame is free for new work even though the cut has not
    /// reached its block yet.
    fences_completed: PaddedAtomicU64,
    publishing_finished: AtomicBool,
    /// Set by a fence lane that hit an IO error. The durable cut can never
    /// advance past the failed frame, so producers/waiters spinning on the
    /// cut would otherwise hang; they must poll [`FuaWalSegment::fence_failed`]
    /// and abort. The underlying error is returned by [`FuaFencePool::join`].
    fence_failed: AtomicBool,
    stage_timings: Mutex<Option<Arc<FuaStageTimings>>>,
}

impl FuaWalSegment {
    /// Create a new segment file with WRITTEN extents and a durable header.
    ///
    /// Extent prewrite streams real zeros through a buffered descriptor and
    /// fsyncs once; this is the setup-time cost that keeps every hot-path
    /// fence off the XFS unwritten-extent conversion path. Prefer
    /// [`Self::recycle`] for steady-state segment rolling.
    ///
    /// # Safety
    ///
    /// The caller must ensure the created path is exclusively owned for the
    /// lifetime of the segment. No other process may write the file.
    pub unsafe fn create(config: FuaWalSegmentConfig) -> std::io::Result<Arc<Self>> {
        let (block_capacity, block_stride, bytes) = config.validate()?;
        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&config.path)?;
            file.set_len(bytes as u64)?;
        }
        prewrite_extents(&config.path, bytes as u64)?;
        sync_parent_dir(&config.path)?;
        // Safety: exclusive ownership passed through from the caller.
        unsafe { Self::open_prepared(config, block_capacity, block_stride, bytes) }
    }

    /// Reuse an existing, fully pre-written segment file under a NEW segment
    /// id. The file must have exactly the geometry of `config`. The header is
    /// rewritten through the FUA descriptor (which also retires the previous
    /// life's control records — they live inside the header page), and the new
    /// epoch stamp makes scan recovery reject the previous life's frames.
    ///
    /// # Safety
    ///
    /// The caller must ensure exclusive ownership of the file and that the
    /// previous segment using this file has been retired (checkpointed past or
    /// no longer needed for recovery).
    pub unsafe fn recycle(config: FuaWalSegmentConfig) -> std::io::Result<Arc<Self>> {
        let (block_capacity, block_stride, bytes) = config.validate()?;
        let metadata = std::fs::metadata(&config.path)?;
        if metadata.len() != bytes as u64 {
            return Err(invalid_data(
                "FUA WAL recycle geometry mismatch: segment file length differs from config",
            ));
        }
        // Safety: exclusive ownership passed through from the caller.
        unsafe { Self::open_prepared(config, block_capacity, block_stride, bytes) }
    }

    /// # Safety
    ///
    /// File at `config.path` exists with `bytes` length and written extents;
    /// caller guarantees exclusive ownership.
    unsafe fn open_prepared(
        config: FuaWalSegmentConfig,
        block_capacity: usize,
        block_stride: usize,
        bytes: usize,
    ) -> std::io::Result<Arc<Self>> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | O_DIRECT | O_DSYNC)
            .open(&config.path)?;
        let staging = AlignedStaging::zeroed(block_capacity * block_stride)?;
        let segment = Self {
            file,
            staging,
            segment_id: config.segment_id,
            epoch: config.segment_id as u32,
            block_size: config.block_size,
            block_capacity: block_capacity as u64,
            block_stride,
            published_blocks: PaddedAtomicU64(AtomicU64::new(0)),
            block_end_seq: (0..block_capacity).map(|_| AtomicU64::new(0)).collect(),
            appender_taken: AtomicBool::new(false),
            fence_cursor: PaddedAtomicU64(AtomicU64::new(0)),
            fence_completed: (0..block_capacity)
                .map(|_| AtomicBool::new(false))
                .collect(),
            durable_frontier: Mutex::new(0),
            durable_blocks: PaddedAtomicU64(AtomicU64::new(0)),
            durable_record_seq: PaddedAtomicU64(AtomicU64::new(0)),
            fences_completed: PaddedAtomicU64(AtomicU64::new(0)),
            publishing_finished: AtomicBool::new(false),
            fence_failed: AtomicBool::new(false),
            stage_timings: Mutex::new(None),
        };
        segment.write_file_header(bytes)?;
        Ok(Arc::new(segment))
    }

    /// Durable header write through the FUA descriptor: the header (and with
    /// it the retirement of any previous life's control records) is on stable
    /// media before the first frame can complete.
    fn write_file_header(&self, _bytes: usize) -> std::io::Result<()> {
        let header = WalSegmentFileHeader {
            magic: WAL_SEGMENT_MAGIC,
            version: WAL_SEGMENT_VERSION,
            header_bytes: WAL_SEGMENT_HEADER_BYTES as u32,
            segment_id: self.segment_id,
            block_size: self.block_size as u32,
            block_capacity: self.block_capacity as u32,
            record_size: size_of::<WriteIntent>() as u32,
            block_header_size: size_of::<WalBlockHeader>() as u32,
            block_trailer_size: size_of::<WalBlockTrailer>() as u32,
            flags: WAL_SEGMENT_FLAG_EPOCH_STAMPED,
            reserved: [0; 2],
        };
        let buffer = AlignedStaging::zeroed(WAL_SEGMENT_HEADER_BYTES)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes_of(&header).as_ptr(),
                buffer.ptr,
                size_of::<WalSegmentFileHeader>(),
            );
            write_all_at(
                &self.file,
                std::slice::from_raw_parts(buffer.ptr, WAL_SEGMENT_HEADER_BYTES),
                0,
            )
        }
    }

    /// Claim the single appender handle. Panics if claimed twice.
    pub fn appender(self: &Arc<Self>) -> FuaWalAppender {
        assert!(
            !self.appender_taken.swap(true, Ordering::AcqRel),
            "FUA WAL segment supports exactly one appender"
        );
        FuaWalAppender {
            timings: self.stage_timings(),
            segment: Arc::clone(self),
            next_block: 0,
            next_record_seq: 0,
        }
    }

    /// Spawn `lanes` fence threads. Lanes exit once publishing is finished
    /// (see [`FuaWalAppender::finish`]) and every published block is durable.
    pub fn spawn_fence_pool(self: &Arc<Self>, lanes: usize) -> FuaFencePool {
        let lanes = lanes.max(1);
        let handles = (0..lanes)
            .map(|_| {
                let segment = Arc::clone(self);
                std::thread::spawn(move || segment.fence_lane_loop())
            })
            .collect();
        FuaFencePool {
            segment: Arc::clone(self),
            handles,
        }
    }

    fn fence_lane_loop(&self) -> std::io::Result<u64> {
        // capture once: the per-op Mutex read is contended at fence rates
        let timings = self.stage_timings();
        let mut fenced = 0_u64;
        loop {
            let block = self.fence_cursor.0.fetch_add(1, Ordering::Relaxed);
            loop {
                if self.published_blocks.0.load(Ordering::Acquire) > block {
                    break;
                }
                if self.publishing_finished.load(Ordering::Acquire)
                    && self.published_blocks.0.load(Ordering::Acquire) <= block
                {
                    return Ok(fenced);
                }
                std::thread::yield_now();
            }
            if let Err(error) = self.fence_block_timed(block, &timings) {
                // Signal producers/waiters before surfacing the error via
                // join(): the durable cut is now permanently stalled at or
                // before this frame.
                self.fence_failed.store(true, Ordering::Release);
                return Err(error);
            }
            fenced += 1;
        }
    }

    /// FUA-write one published frame and advance the contiguous durable cut.
    #[cfg(test)]
    fn fence_block(&self, block_id: u64) -> std::io::Result<()> {
        self.fence_block_timed(block_id, &None)
    }

    /// FUA-write one published frame and advance the contiguous durable cut.
    fn fence_block_timed(
        &self,
        block_id: u64,
        timings: &Option<Arc<FuaStageTimings>>,
    ) -> std::io::Result<()> {
        debug_assert!(block_id < self.block_capacity);
        if let Some(timings) = &timings {
            timings.fence_start_ns[block_id as usize].store(timings.now_ns(), Ordering::Relaxed);
        }
        let offset = block_id as usize * self.block_stride;
        let file_offset = WAL_SEGMENT_HEADER_BYTES as u64 + block_id * self.block_stride as u64;
        unsafe {
            write_all_at(
                &self.file,
                std::slice::from_raw_parts(self.staging.ptr.add(offset), self.block_stride),
                file_offset,
            )?;
        }
        if let Some(timings) = &timings {
            timings.fence_done_ns[block_id as usize].store(timings.now_ns(), Ordering::Relaxed);
        }
        self.fence_completed[block_id as usize].store(true, Ordering::Release);
        self.fences_completed.0.fetch_add(1, Ordering::AcqRel);
        let mut frontier = self
            .durable_frontier
            .lock()
            .map_err(|_| std::io::Error::other("FUA WAL durable frontier poisoned"))?;
        let mut advanced = *frontier;
        while advanced < self.block_capacity
            && self.fence_completed[advanced as usize].load(Ordering::Acquire)
        {
            advanced += 1;
        }
        if advanced != *frontier {
            if let Some(timings) = &timings {
                let now = timings.now_ns();
                for covered in *frontier..advanced {
                    timings.cut_ns[covered as usize].store(now, Ordering::Relaxed);
                }
            }
            *frontier = advanced;
            self.durable_blocks.0.store(advanced, Ordering::Release);
            self.durable_record_seq.0.store(
                self.block_end_seq[(advanced - 1) as usize].load(Ordering::Relaxed),
                Ordering::Release,
            );
        }
        Ok(())
    }

    /// True once any fence lane has failed with an IO error. The durable cut
    /// will never advance past the failed frame; pacing loops and ack waiters
    /// must poll this and abort instead of spinning on the cut. The
    /// underlying error is returned by [`FuaFencePool::join`].
    pub fn fence_failed(&self) -> bool {
        self.fence_failed.load(Ordering::Acquire)
    }

    /// Enable per-block stage timestamps for latency attribution. Must be
    /// called BEFORE [`Self::spawn_fence_pool`] and [`Self::appender`]: both
    /// capture the handle once at construction to keep it off the hot path.
    pub fn enable_stage_timings(&self) -> Arc<FuaStageTimings> {
        let timings = FuaStageTimings::new(self.block_capacity as usize);
        *self
            .stage_timings
            .lock()
            .expect("stage timings lock poisoned") = Some(Arc::clone(&timings));
        timings
    }

    fn stage_timings(&self) -> Option<Arc<FuaStageTimings>> {
        self.stage_timings
            .lock()
            .expect("stage timings lock poisoned")
            .clone()
    }

    /// Contiguous durable block prefix.
    pub fn durable_blocks(&self) -> u64 {
        self.durable_blocks.0.load(Ordering::Acquire)
    }

    /// Records covered by the durable cut (record seqs `< durable_record_seq`
    /// are durable).
    pub fn durable_record_seq(&self) -> u64 {
        self.durable_record_seq.0.load(Ordering::Acquire)
    }

    /// Contiguous published block prefix.
    pub fn published_blocks(&self) -> u64 {
        self.published_blocks.0.load(Ordering::Acquire)
    }

    pub fn block_capacity(&self) -> u64 {
        self.block_capacity
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Free capacity in a fence pool of `lanes`: the appender's pacing gate.
    /// Publish a frame only when this is non-zero; accumulate otherwise.
    ///
    /// In-flight is measured against TOTAL completions, not the contiguous
    /// durable cut: fences complete out of order, and gating on the cut
    /// would idle lanes behind a straggler (measured: effective device queue
    /// depth sags below the lane count and fence latency rises).
    pub fn free_fence_slots(&self, lanes: usize) -> usize {
        let in_flight = self
            .published_blocks
            .0
            .load(Ordering::Relaxed)
            .saturating_sub(self.fences_completed.0.load(Ordering::Acquire));
        (lanes as u64).saturating_sub(in_flight) as usize
    }

    /// Copy a published block's records into `out`. Returns `None` if the
    /// block is not yet published.
    pub fn read_published_block_into(
        &self,
        block_id: u64,
        out: &mut Vec<WriteIntent>,
    ) -> Option<u64> {
        if self.published_blocks.0.load(Ordering::Acquire) <= block_id {
            return None;
        }
        let offset = block_id as usize * self.block_stride;
        unsafe {
            let header = self.staging.ptr.add(offset).cast::<WalBlockHeader>().read();
            let count = header.count as usize;
            let payload = self
                .staging
                .ptr
                .add(offset + size_of::<WalBlockHeader>())
                .cast::<WriteIntent>();
            out.clear();
            out.reserve(count);
            std::ptr::copy_nonoverlapping(payload, out.as_mut_ptr(), count);
            out.set_len(count);
            Some(header.first_client_seq)
        }
    }
}

fn prewrite_extents(path: &Path, bytes: u64) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = OpenOptions::new()
        .write(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let chunk = vec![0u8; 1 << 20];
    let mut remaining = bytes;
    while remaining > 0 {
        let step = remaining.min(chunk.len() as u64) as usize;
        file.write_all(&chunk[..step])?;
        remaining -= step as u64;
    }
    file.sync_all()
}

/// The single publishing handle for a [`FuaWalSegment`].
pub struct FuaWalAppender {
    segment: Arc<FuaWalSegment>,
    timings: Option<Arc<FuaStageTimings>>,
    next_block: u64,
    next_record_seq: u64,
}

impl FuaWalAppender {
    /// Stage one WAL frame (block header + records + CRC trailer, epoch
    /// stamped) and publish it to fence lanes and readers. `intents` must
    /// hold between 1 and `block_size` records. Returns the block id.
    ///
    /// This is a staging write plus one release store: it does NOT touch the
    /// file. Durability happens in the fence pool; gate visibility on
    /// [`FuaWalSegment::durable_record_seq`].
    pub fn publish_intents(&mut self, intents: &[WriteIntent]) -> std::io::Result<u64> {
        let segment = &*self.segment;
        let count = intents.len();
        if count == 0 || count > segment.block_size {
            return Err(invalid_data(
                "FUA WAL frame must hold 1..=block_size records",
            ));
        }
        let block_id = self.next_block;
        if block_id >= segment.block_capacity {
            return Err(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                "FUA WAL segment is full; roll to the next segment",
            ));
        }
        let first_client_seq = self.next_record_seq;
        let payload_bytes = std::mem::size_of_val(intents);
        let header = WalBlockHeader {
            magic: WAL_BLOCK_HEADER_MAGIC,
            block_id,
            first_client_seq,
            count: count as u32,
            block_size: segment.block_size as u32,
            payload_bytes: payload_bytes as u32,
            reserved0: segment.epoch,
            reserved: [0; 3],
        };
        let block_offset = block_id as usize * segment.block_stride;
        let header_len = size_of::<WalBlockHeader>();
        unsafe {
            let base = segment.staging.ptr.add(block_offset);
            std::ptr::copy_nonoverlapping(bytes_of(&header).as_ptr(), base, header_len);
            std::ptr::copy_nonoverlapping(
                intents.as_ptr().cast::<u8>(),
                base.add(header_len),
                payload_bytes,
            );
            // deterministic frames: zero the unused payload tail
            let unused = (segment.block_size - count) * size_of::<WriteIntent>();
            if unused > 0 {
                std::ptr::write_bytes(base.add(header_len + payload_bytes), 0, unused);
            }
            let checksum = block_crc32c(
                &header,
                std::slice::from_raw_parts(base.add(header_len), payload_bytes),
            );
            let trailer = WalBlockTrailer {
                magic: WAL_BLOCK_TRAILER_MAGIC,
                block_id,
                first_client_seq,
                count: count as u32,
                payload_bytes: payload_bytes as u32,
                checksum: checksum as u64,
                reserved: [0; 2],
                commit_marker: WAL_BLOCK_COMMIT_MARKER,
            };
            let trailer_offset = header_len + segment.block_size * size_of::<WriteIntent>();
            std::ptr::copy_nonoverlapping(
                bytes_of(&trailer).as_ptr(),
                base.add(trailer_offset),
                size_of::<WalBlockTrailer>(),
            );
        }
        self.next_record_seq = first_client_seq + count as u64;
        segment.block_end_seq[block_id as usize].store(self.next_record_seq, Ordering::Relaxed);
        self.next_block = block_id + 1;
        if let Some(timings) = &self.timings {
            timings.publish_ns[block_id as usize].store(timings.now_ns(), Ordering::Relaxed);
        }
        segment
            .published_blocks
            .0
            .store(self.next_block, Ordering::Release);
        Ok(block_id)
    }

    /// Total records published so far.
    pub fn published_record_seq(&self) -> u64 {
        self.next_record_seq
    }

    /// Declare publishing finished so fence lanes can drain and exit.
    pub fn finish(self) {
        self.segment
            .publishing_finished
            .store(true, Ordering::Release);
    }
}

/// Handle for the fence threads of one segment.
pub struct FuaFencePool {
    segment: Arc<FuaWalSegment>,
    handles: Vec<JoinHandle<std::io::Result<u64>>>,
}

impl FuaFencePool {
    /// Wait for every published block to become durable and all lanes to
    /// exit. Requires [`FuaWalAppender::finish`] to have been called (or the
    /// appender dropped via `finish`); otherwise lanes keep waiting for more
    /// frames. Returns the total number of fences issued.
    pub fn join(self) -> std::io::Result<u64> {
        let mut fences = 0_u64;
        for handle in self.handles {
            fences += handle
                .join()
                .map_err(|_| std::io::Error::other("FUA WAL fence lane panicked"))??;
        }
        debug_assert_eq!(
            self.segment.durable_blocks(),
            self.segment.published_blocks()
        );
        Ok(fences)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{intent_for, recover_wal_segment_by_scan, stats_for_range};

    fn test_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("fua-wal-tests");
        std::fs::create_dir_all(&dir).expect("create test dir");
        let path = dir.join(format!("{name}-{}.dat", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn intents(first: u64, count: usize) -> Vec<WriteIntent> {
        (0..count as u64).map(|i| intent_for(first + i)).collect()
    }

    fn config(path: &Path, segment_id: u64) -> FuaWalSegmentConfig {
        FuaWalSegmentConfig {
            path: path.to_path_buf(),
            segment_id,
            records: 62 * 32,
            block_size: 62,
        }
    }

    #[test]
    fn publish_fence_recover_roundtrip() {
        let path = test_path("roundtrip");
        let segment = unsafe { FuaWalSegment::create(config(&path, 7)).expect("create") };
        let pool = segment.spawn_fence_pool(4);
        let mut appender = segment.appender();
        // varied frame sizes, including partial frames
        let mut seq = 0_u64;
        for count in [62_usize, 1, 17, 62, 30, 5] {
            appender
                .publish_intents(&intents(seq, count))
                .expect("publish");
            seq += count as u64;
        }
        appender.finish();
        let fences = pool.join().expect("fence pool");
        assert_eq!(fences, 6);
        assert_eq!(segment.durable_blocks(), 6);
        assert_eq!(segment.durable_record_seq(), seq);
        // reader sees published payloads
        let mut out = Vec::new();
        assert_eq!(segment.read_published_block_into(2, &mut out), Some(63));
        assert_eq!(out.len(), 17);
        assert_eq!(out[0], intent_for(63));
        drop(segment);
        let recovery = recover_wal_segment_by_scan(&path).expect("scan");
        assert_eq!(recovery.segment_id, 7);
        assert_eq!(recovery.recovered_blocks, 6);
        assert_eq!(recovery.recovered_records, seq);
        assert_eq!(recovery.stats, stats_for_range(0, seq));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn recycle_rejects_previous_epoch_frames() {
        let path = test_path("recycle");
        // first life: 8 durable blocks under segment id 1
        {
            let segment = unsafe { FuaWalSegment::create(config(&path, 1)).expect("create") };
            let pool = segment.spawn_fence_pool(2);
            let mut appender = segment.appender();
            for block in 0..8_u64 {
                appender
                    .publish_intents(&intents(block * 62, 62))
                    .expect("publish");
            }
            appender.finish();
            pool.join().expect("fence pool");
        }
        let first_life = recover_wal_segment_by_scan(&path).expect("scan first life");
        assert_eq!(first_life.recovered_blocks, 8);
        // second life: recycled as segment id 2, only 3 blocks written
        {
            let segment = unsafe { FuaWalSegment::recycle(config(&path, 2)).expect("recycle") };
            let pool = segment.spawn_fence_pool(2);
            let mut appender = segment.appender();
            for block in 0..3_u64 {
                appender
                    .publish_intents(&intents(1_000_000 + block * 62, 62))
                    .expect("publish");
            }
            appender.finish();
            pool.join().expect("fence pool");
        }
        let recovery = recover_wal_segment_by_scan(&path).expect("scan second life");
        assert_eq!(recovery.segment_id, 2);
        // the previous life's blocks 3..8 are still valid-looking frames on
        // disk, but their epoch stamp is 1, so the scan MUST stop at 3
        assert_eq!(recovery.recovered_blocks, 3);
        assert_eq!(recovery.recovered_records, 3 * 62);
        assert_eq!(recovery.stats, stats_for_range(1_000_000, 3 * 62));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unfenced_tail_is_not_recovered() {
        let path = test_path("torn-tail");
        let segment = unsafe { FuaWalSegment::create(config(&path, 3)).expect("create") };
        let mut appender = segment.appender();
        for block in 0..5_u64 {
            appender
                .publish_intents(&intents(block * 62, 62))
                .expect("publish");
        }
        // fence only blocks 0..3 by hand; 3 and 4 stay staging-only
        segment.fence_block(0).expect("fence");
        segment.fence_block(1).expect("fence");
        segment.fence_block(2).expect("fence");
        assert_eq!(segment.durable_blocks(), 3);
        assert_eq!(segment.durable_record_seq(), 3 * 62);
        drop(appender);
        drop(segment);
        let recovery = recover_wal_segment_by_scan(&path).expect("scan");
        assert_eq!(recovery.recovered_blocks, 3);
        assert_eq!(recovery.recovered_records, 3 * 62);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn out_of_order_fences_gate_the_durable_cut() {
        let path = test_path("ooo-cut");
        let segment = unsafe { FuaWalSegment::create(config(&path, 4)).expect("create") };
        let mut appender = segment.appender();
        for block in 0..4_u64 {
            appender
                .publish_intents(&intents(block * 62, 62))
                .expect("publish");
        }
        // complete 2 and 3 first: cut must stay at 0 until 0 and 1 land
        segment.fence_block(2).expect("fence");
        segment.fence_block(3).expect("fence");
        assert_eq!(segment.durable_blocks(), 0);
        assert_eq!(segment.durable_record_seq(), 0);
        segment.fence_block(0).expect("fence");
        assert_eq!(segment.durable_blocks(), 1);
        segment.fence_block(1).expect("fence");
        assert_eq!(segment.durable_blocks(), 4);
        assert_eq!(segment.durable_record_seq(), 4 * 62);
        drop(appender);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn config_validation_rejects_unaligned_strides_and_geometry_mismatch() {
        let path = test_path("validate");
        let mut bad = config(&path, 1);
        bad.block_size = 64; // stride 4224, not 512B-aligned
        assert!(unsafe { FuaWalSegment::create(bad) }.is_err());
        // geometry mismatch on recycle
        let segment = unsafe { FuaWalSegment::create(config(&path, 1)).expect("create") };
        drop(segment);
        let mut smaller = config(&path, 2);
        smaller.records = 62 * 16;
        assert!(unsafe { FuaWalSegment::recycle(smaller) }.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn config_validation_rejects_epoch_zero_segment_ids() {
        // Audit finding: `segment_id as u32` is the recycle epoch and 0 is the
        // legacy sentinel. An id whose low 32 bits are 0 (0, 1<<32, ...) would
        // let a recycled file's previous-life frames pass scan recovery.
        let path = test_path("epoch-zero");
        assert!(unsafe { FuaWalSegment::create(config(&path, 0)) }.is_err());
        assert!(unsafe { FuaWalSegment::create(config(&path, 1_u64 << 32)) }.is_err());
        // a valid first life, then a low-32-zero recycle id must be rejected
        let segment = unsafe { FuaWalSegment::create(config(&path, 1)).expect("create") };
        drop(segment);
        assert!(unsafe { FuaWalSegment::recycle(config(&path, 2_u64 << 32)) }.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn appender_pacing_accessor_tracks_pool_occupancy() {
        let path = test_path("pacing");
        let segment = unsafe { FuaWalSegment::create(config(&path, 5)).expect("create") };
        let mut appender = segment.appender();
        assert_eq!(segment.free_fence_slots(4), 4);
        for block in 0..3_u64 {
            appender
                .publish_intents(&intents(block * 62, 62))
                .expect("publish");
        }
        assert_eq!(segment.free_fence_slots(4), 1);
        segment.fence_block(0).expect("fence");
        assert_eq!(segment.free_fence_slots(4), 2);
        drop(appender);
        let _ = std::fs::remove_file(&path);
    }
}
