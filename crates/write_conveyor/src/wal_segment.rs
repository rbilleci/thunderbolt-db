use std::fs::{File, OpenOptions};
use std::hint::spin_loop;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::{size_of, MaybeUninit};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::{
    intent_for, preallocate_file, sync_parent_dir, DrainStats, OpenShardAppendStore, PublishError,
    WriteIntent,
};

pub(crate) const WAL_SEGMENT_MAGIC: u64 = 0x5743_4f4e_5659_5347;
const WAL_CONTROL_MAGIC: u64 = 0x5743_4f4e_4354_524c;
const WAL_MANAGER_CONTROL_MAGIC: u64 = 0x5743_4f4e_4d43_5452;
pub(crate) const WAL_BLOCK_HEADER_MAGIC: u64 = 0x5743_4f4e_4248_4452;
pub(crate) const WAL_BLOCK_TRAILER_MAGIC: u64 = 0x5743_4f4e_4254_524c;
const WAL_CONTROL_COMMIT_MARKER: u64 = 0x4455_5241_424c_455f;
const WAL_MANAGER_CONTROL_COMMIT_MARKER: u64 = 0x4d47_5244_5552_4142;
pub(crate) const WAL_BLOCK_COMMIT_MARKER: u64 = 0x434f_4d4d_4954_4544;
pub(crate) const WAL_SEGMENT_VERSION: u32 = 1;
pub(crate) const WAL_SEGMENT_HEADER_BYTES: usize = 4096;
const WAL_CONTROL_SLOTS: usize = 2;
/// Segment flag: block headers stamp `reserved0` with the segment id's low 32
/// bits (the recycle epoch). Scan recovery then rejects valid-looking blocks
/// left over from a previous life of a RECYCLED segment file, which would
/// otherwise be indistinguishable from the current segment's blocks beyond the
/// new frontier.
pub(crate) const WAL_SEGMENT_FLAG_EPOCH_STAMPED: u32 = 1;

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WalSegmentFileHeader {
    pub(crate) magic: u64,
    pub(crate) version: u32,
    pub(crate) header_bytes: u32,
    pub(crate) segment_id: u64,
    pub(crate) block_size: u32,
    pub(crate) block_capacity: u32,
    pub(crate) record_size: u32,
    pub(crate) block_header_size: u32,
    pub(crate) block_trailer_size: u32,
    pub(crate) flags: u32,
    pub(crate) reserved: [u64; 2],
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
struct WalControlRecord {
    magic: u64,
    version: u32,
    slot: u32,
    segment_id: u64,
    durable_blocks: u64,
    generation: u64,
    checksum: u32,
    reserved0: u32,
    reserved: u64,
    commit_marker: u64,
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
struct WalManagerControlRecord {
    magic: u64,
    version: u32,
    slot: u32,
    first_segment_id: u64,
    durable_segment_id: u64,
    durable_blocks: u64,
    generation: u64,
    records_per_segment: u32,
    block_size: u32,
    checksum: u32,
    reserved0: u32,
    reserved: [u64; 7],
    commit_marker: u64,
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WalBlockHeader {
    pub(crate) magic: u64,
    pub(crate) block_id: u64,
    pub(crate) first_client_seq: u64,
    pub(crate) count: u32,
    pub(crate) block_size: u32,
    pub(crate) payload_bytes: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved: [u64; 3],
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WalBlockTrailer {
    pub(crate) magic: u64,
    pub(crate) block_id: u64,
    pub(crate) first_client_seq: u64,
    pub(crate) count: u32,
    pub(crate) payload_bytes: u32,
    pub(crate) checksum: u64,
    pub(crate) reserved: [u64; 2],
    pub(crate) commit_marker: u64,
}

#[repr(align(128))]
struct WalBlockState {
    sequence: AtomicU64,
    count: AtomicU64,
}

#[repr(align(128))]
struct PaddedAtomicU64 {
    value: AtomicU64,
}

impl PaddedAtomicU64 {
    fn new(value: u64) -> Self {
        Self {
            value: AtomicU64::new(value),
        }
    }

    fn fetch_add(&self, value: u64, ordering: Ordering) -> u64 {
        self.value.fetch_add(value, ordering)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalRecovery {
    pub segment_id: u64,
    pub block_size: u32,
    pub block_capacity: u32,
    pub recovered_blocks: u64,
    pub recovered_records: u64,
    pub stats: DrainStats,
}

pub struct MappedWalSegment {
    file: File,
    ptr: NonNull<u8>,
    bytes: usize,
    segment_id: u64,
    block_size: u64,
    block_capacity: u64,
    block_stride: usize,
    states: Box<[WalBlockState]>,
    tail_block: PaddedAtomicU64,
    durable_blocks: AtomicU64,
    data_frontier_blocks: AtomicU64,
    prewrite_data_frontier_blocks: AtomicU64,
    write_data_frontier_blocks: AtomicU64,
    file_data_frontier_blocks: AtomicU64,
    control_generation: AtomicU64,
    sync_in_progress: AtomicBool,
    consumer_taken: AtomicBool,
    write_scratch: Mutex<Vec<u8>>,
}

unsafe impl Send for MappedWalSegment {}
unsafe impl Sync for MappedWalSegment {}

#[derive(Clone, Debug)]
pub struct WalSegmentManagerConfig {
    pub dir: PathBuf,
    pub prefix: String,
    pub records_per_segment: usize,
    pub block_size: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalPosition {
    pub segment_id: u64,
    pub block_id: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WalManagerRecovery {
    pub first_segment_id: u64,
    pub durable_segment_id: u64,
    pub durable_blocks: u64,
    pub recovered_segments: u64,
    pub recovered_records: u64,
    pub stats: DrainStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WalDataSyncMode {
    #[default]
    RangeAndFileData,
    WriteAndFileData,
    PrewriteAndFileData,
    SyncWriteData,
    FileDataOnly,
}

pub struct WalSegmentManager {
    config: WalSegmentManagerConfig,
    control: File,
    current_segment: Arc<MappedWalSegment>,
    current_segment_id: u64,
    current_blocks: u64,
    sealed_segments: Vec<Arc<MappedWalSegment>>,
    first_segment_id: u64,
    durable_segment_id: u64,
    durable_blocks: u64,
    control_generation: u64,
}

#[derive(Clone)]
pub struct WalPublishedBlock {
    position: WalPosition,
    global_block_id: u64,
    segment: Arc<MappedWalSegment>,
}

impl WalSegmentManagerConfig {
    pub fn new(
        dir: impl Into<PathBuf>,
        prefix: impl Into<String>,
        records_per_segment: usize,
        block_size: usize,
    ) -> Self {
        Self {
            dir: dir.into(),
            prefix: prefix.into(),
            records_per_segment,
            block_size,
        }
    }

    fn segment_path(&self, segment_id: u64) -> PathBuf {
        self.dir
            .join(format!("{}-{:020}.wal", self.prefix, segment_id))
    }

    fn control_path(&self) -> PathBuf {
        self.dir.join(format!("{}.control", self.prefix))
    }

    fn blocks_per_segment(&self) -> u64 {
        self.records_per_segment.div_ceil(self.block_size) as u64
    }
}

impl WalPublishedBlock {
    pub fn position(&self) -> WalPosition {
        self.position
    }

    pub fn global_block_id(&self) -> u64 {
        self.global_block_id
    }

    pub fn read_published_block_into(&self, out: &mut Vec<WriteIntent>) -> Option<()> {
        self.segment
            .read_published_block_into(self.position.block_id, out)
    }

    pub fn sync_published_data_frontier(&self) -> std::io::Result<()> {
        self.segment
            .sync_published_data_frontier(self.position.block_id + 1)
    }

    pub fn sync_published_data_frontier_with_mode(
        &self,
        mode: WalDataSyncMode,
    ) -> std::io::Result<()> {
        self.segment
            .sync_published_data_frontier_with_mode(self.position.block_id + 1, mode)
    }

    pub fn write_published_data_frontier(&self) -> std::io::Result<u64> {
        self.segment
            .write_published_data_frontier(self.position.block_id + 1)
    }
}

impl WalSegmentManager {
    /// # Safety
    ///
    /// The caller must ensure the configured directory/prefix is exclusively
    /// owned by this manager while it is live. Existing segment/control paths
    /// are not overwritten.
    pub unsafe fn create(config: WalSegmentManagerConfig) -> std::io::Result<Self> {
        validate_manager_config(&config)?;
        std::fs::create_dir_all(&config.dir)?;
        let control_path = config.control_path();
        let control = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&control_path)?;
        control.set_len((WAL_CONTROL_SLOTS * size_of::<WalManagerControlRecord>()) as u64)?;
        control.sync_data()?;
        sync_parent_dir(&control_path)?;

        let current_segment = Arc::new(unsafe {
            MappedWalSegment::create(
                config.segment_path(0),
                0,
                config.records_per_segment,
                config.block_size,
            )?
        });
        Ok(Self {
            config,
            control,
            current_segment,
            current_segment_id: 0,
            current_blocks: 0,
            sealed_segments: Vec::new(),
            first_segment_id: 0,
            durable_segment_id: 0,
            durable_blocks: 0,
            control_generation: 0,
        })
    }

    pub fn publish_block(
        &mut self,
        first_client_seq: u64,
        count: usize,
    ) -> Result<WalPosition, Box<dyn std::error::Error>> {
        Ok(self
            .publish_block_handle(first_client_seq, count)?
            .position())
    }

    pub fn publish_block_handle(
        &mut self,
        first_client_seq: u64,
        count: usize,
    ) -> Result<WalPublishedBlock, Box<dyn std::error::Error>> {
        if count == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL manager publish count must be non-zero",
            )
            .into());
        }
        if count > self.config.block_size {
            return Err(PublishError::CountExceedsBlockSize {
                count,
                block_size: self.config.block_size,
            }
            .into());
        }
        if self.current_blocks == self.current_segment.block_capacity() {
            self.roll_segment()?;
        }
        let position = WalPosition {
            segment_id: self.current_segment_id,
            block_id: self.current_blocks,
        };
        self.current_segment
            .try_publish_block(first_client_seq, count)?;
        self.current_blocks += 1;
        Ok(WalPublishedBlock {
            position,
            global_block_id: OpenShardAppendStore::global_block_id(
                position.segment_id,
                self.config.blocks_per_segment(),
                position.block_id,
            ),
            segment: Arc::clone(&self.current_segment),
        })
    }

    pub fn publish_intents(
        &mut self,
        intents: &[WriteIntent],
    ) -> Result<WalPosition, Box<dyn std::error::Error>> {
        Ok(self.publish_intents_handle(intents)?.position())
    }

    pub fn publish_intents_handle(
        &mut self,
        intents: &[WriteIntent],
    ) -> Result<WalPublishedBlock, Box<dyn std::error::Error>> {
        if intents.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL manager publish payload must be non-empty",
            )
            .into());
        }
        if intents.len() > self.config.block_size {
            return Err(PublishError::CountExceedsBlockSize {
                count: intents.len(),
                block_size: self.config.block_size,
            }
            .into());
        }
        if self.current_blocks == self.current_segment.block_capacity() {
            self.roll_segment()?;
        }
        let position = WalPosition {
            segment_id: self.current_segment_id,
            block_id: self.current_blocks,
        };
        self.current_segment.try_publish_intents(intents)?;
        self.current_blocks += 1;
        Ok(WalPublishedBlock {
            position,
            global_block_id: OpenShardAppendStore::global_block_id(
                position.segment_id,
                self.config.blocks_per_segment(),
                position.block_id,
            ),
            segment: Arc::clone(&self.current_segment),
        })
    }

    pub fn sync_all_published(&mut self) -> std::io::Result<WalPosition> {
        for segment in self.sealed_segments.iter_mut() {
            if segment.durable_blocks() < segment.block_capacity() {
                segment.sync_published_prefix(segment.block_capacity())?;
            }
        }
        self.current_segment
            .sync_published_prefix(self.current_blocks)?;
        let position = WalPosition {
            segment_id: self.current_segment_id,
            block_id: self.current_blocks,
        };
        self.write_external_control(position)?;
        self.durable_segment_id = position.segment_id;
        self.durable_blocks = position.block_id;
        Ok(position)
    }

    pub fn remove_segments_before(&mut self, first_kept_segment_id: u64) -> std::io::Result<usize> {
        if first_kept_segment_id < self.first_segment_id {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot move WAL manager retained boundary backward",
            ));
        }
        if first_kept_segment_id == self.first_segment_id {
            return Ok(0);
        }
        if first_kept_segment_id > self.durable_segment_id {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot remove WAL segments beyond durable boundary",
            ));
        }
        let previous_first_segment_id = self.first_segment_id;
        self.write_external_control_for(
            WalPosition {
                segment_id: self.durable_segment_id,
                block_id: self.durable_blocks,
            },
            first_kept_segment_id,
        )?;
        self.first_segment_id = first_kept_segment_id;
        self.sealed_segments
            .retain(|segment| segment.segment_id >= first_kept_segment_id);
        let mut removed = 0;
        for segment_id in previous_first_segment_id..first_kept_segment_id {
            match std::fs::remove_file(self.config.segment_path(segment_id)) {
                Ok(()) => removed += 1,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => return Err(err),
            }
        }
        if removed != 0 {
            sync_parent_dir(&self.config.segment_path(previous_first_segment_id))?;
        }
        Ok(removed)
    }

    fn roll_segment(&mut self) -> std::io::Result<()> {
        let next_segment_id = self.current_segment_id + 1;
        let next = Arc::new(unsafe {
            MappedWalSegment::create(
                self.config.segment_path(next_segment_id),
                next_segment_id,
                self.config.records_per_segment,
                self.config.block_size,
            )?
        });
        let previous = std::mem::replace(&mut self.current_segment, next);
        self.sealed_segments.push(previous);
        self.current_segment_id = next_segment_id;
        self.current_blocks = 0;
        Ok(())
    }

    fn write_external_control(&mut self, position: WalPosition) -> std::io::Result<()> {
        self.write_external_control_for(position, self.first_segment_id)
    }

    fn write_external_control_for(
        &mut self,
        position: WalPosition,
        first_segment_id: u64,
    ) -> std::io::Result<()> {
        if first_segment_id > position.segment_id {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL manager retained boundary cannot pass durable segment",
            ));
        }
        if position.segment_id < self.durable_segment_id
            || (position.segment_id == self.durable_segment_id
                && position.block_id < self.durable_blocks)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL manager durable position cannot move backward",
            ));
        }
        let generation = self.control_generation + 1;
        let slot = generation as usize % WAL_CONTROL_SLOTS;
        let mut record = WalManagerControlRecord {
            magic: WAL_MANAGER_CONTROL_MAGIC,
            version: WAL_SEGMENT_VERSION,
            slot: slot as u32,
            first_segment_id,
            durable_segment_id: position.segment_id,
            durable_blocks: position.block_id,
            generation,
            records_per_segment: self.config.records_per_segment as u32,
            block_size: self.config.block_size as u32,
            checksum: 0,
            reserved0: 0,
            reserved: [0; 7],
            commit_marker: WAL_MANAGER_CONTROL_COMMIT_MARKER,
        };
        record.checksum = manager_control_crc32c(&record);
        self.control
            .seek(SeekFrom::Start(manager_control_offset(slot) as u64))?;
        self.control.write_all(bytes_of(&record))?;
        self.control.sync_data()?;
        self.control_generation = generation;
        Ok(())
    }
}

impl MappedWalSegment {
    /// # Safety
    ///
    /// The caller must ensure the created path is exclusively owned for the
    /// lifetime of the returned mapping. No other process or mapping may
    /// truncate, resize, or write the file while this segment is alive.
    pub unsafe fn create(
        path: impl AsRef<Path>,
        segment_id: u64,
        records: usize,
        block_size: usize,
    ) -> std::io::Result<Self> {
        assert!(
            records > 0,
            "mapped WAL segment must contain at least one record"
        );
        assert!(block_size > 0, "WAL block size must be non-zero");
        assert!(
            block_size <= u32::MAX as usize,
            "WAL block size must fit on disk"
        );
        assert!(
            block_size <= u32::MAX as usize / size_of::<WriteIntent>(),
            "WAL block payload bytes must fit on disk"
        );
        let path = path.as_ref();
        let block_capacity = records.div_ceil(block_size);
        assert!(
            block_capacity <= u32::MAX as usize,
            "WAL block capacity must fit on disk"
        );
        let mapped_records = block_capacity
            .checked_mul(block_size)
            .expect("mapped WAL segment record count overflow");
        let payload_bytes = mapped_records
            .checked_mul(size_of::<WriteIntent>())
            .expect("mapped WAL segment payload byte length overflow");
        let block_stride = checked_block_stride(block_size)?;
        let bytes = checked_segment_bytes(block_capacity, block_stride)?;
        debug_assert_eq!(
            payload_bytes,
            block_capacity * block_size * size_of::<WriteIntent>()
        );

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)?;
        file.set_len(bytes as u64)?;
        preallocate_file(&file, bytes)?;
        sync_parent_dir(path)?;

        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        let ptr = NonNull::new(raw.cast::<u8>()).expect("mmap returned null");
        let states = (0..block_capacity)
            .map(|sequence| WalBlockState {
                sequence: AtomicU64::new(sequence as u64),
                count: AtomicU64::new(0),
            })
            .collect();
        let segment = Self {
            file,
            ptr,
            bytes,
            segment_id,
            block_size: block_size as u64,
            block_capacity: block_capacity as u64,
            block_stride,
            states,
            tail_block: PaddedAtomicU64::new(0),
            durable_blocks: AtomicU64::new(0),
            data_frontier_blocks: AtomicU64::new(0),
            prewrite_data_frontier_blocks: AtomicU64::new(0),
            write_data_frontier_blocks: AtomicU64::new(0),
            file_data_frontier_blocks: AtomicU64::new(0),
            control_generation: AtomicU64::new(0),
            sync_in_progress: AtomicBool::new(false),
            consumer_taken: AtomicBool::new(false),
            write_scratch: Mutex::new(Vec::new()),
        };
        segment.write_file_header();
        Ok(segment)
    }

    pub fn try_publish_block(
        &self,
        first_client_seq: u64,
        count: usize,
    ) -> Result<(), PublishError> {
        let _ = self.try_publish_block_position(first_client_seq, count)?;
        Ok(())
    }

    pub fn try_publish_block_position(
        &self,
        first_client_seq: u64,
        count: usize,
    ) -> Result<Option<u64>, PublishError> {
        if count == 0 {
            return Ok(None);
        }
        let block_id = self.claim_block(count)?;
        let header = self.write_block_header(block_id, first_client_seq, count);
        for offset in 0..count as u64 {
            let intent = intent_for(first_client_seq.wrapping_add(offset));
            unsafe {
                self.payload_ptr(block_id)
                    .add(offset as usize)
                    .write(intent);
            }
        }
        self.commit_block(block_id, header);
        Ok(Some(block_id))
    }

    pub fn try_publish_intents(&self, intents: &[WriteIntent]) -> Result<(), PublishError> {
        let _ = self.try_publish_intents_position(intents)?;
        Ok(())
    }

    pub fn try_publish_intents_position(
        &self,
        intents: &[WriteIntent],
    ) -> Result<Option<u64>, PublishError> {
        if intents.is_empty() {
            return Ok(None);
        }
        let count = intents.len();
        let block_id = self.claim_block(count)?;
        let header = self.write_block_header(block_id, intents[0].client_seq, count);
        unsafe {
            std::ptr::copy_nonoverlapping(intents.as_ptr(), self.payload_ptr(block_id), count);
        }
        self.commit_block(block_id, header);
        Ok(Some(block_id))
    }

    pub fn read_published_block(&self, block_id: u64) -> Option<DrainStats> {
        if block_id >= self.block_capacity {
            return None;
        }
        let state = &self.states[block_id as usize];
        if state.sequence.load(Ordering::Acquire) != block_id + 1 {
            return None;
        }
        Some(self.read_block_after_acquire(block_id))
    }

    pub fn read_published_block_into(
        &self,
        block_id: u64,
        out: &mut Vec<WriteIntent>,
    ) -> Option<()> {
        if block_id >= self.block_capacity {
            return None;
        }
        let state = &self.states[block_id as usize];
        if state.sequence.load(Ordering::Acquire) != block_id + 1 {
            return None;
        }
        let count = state.count.load(Ordering::Relaxed) as usize;
        out.clear();
        out.reserve(count);
        unsafe {
            std::ptr::copy_nonoverlapping(self.payload_ptr(block_id), out.as_mut_ptr(), count);
            out.set_len(count);
        }
        Some(())
    }

    pub fn consumer(&self) -> MappedWalConsumer<'_> {
        assert!(
            !self.consumer_taken.swap(true, Ordering::AcqRel),
            "mapped WAL segment supports exactly one ordered consumer"
        );
        MappedWalConsumer {
            segment: self,
            next_block: 0,
        }
    }

    pub fn block_capacity(&self) -> u64 {
        self.block_capacity
    }

    pub fn durable_blocks(&self) -> u64 {
        self.durable_blocks.load(Ordering::Acquire)
    }

    pub fn contiguous_published_prefix(
        &self,
        start_blocks: u64,
        blocks: u64,
    ) -> std::io::Result<u64> {
        if start_blocks > blocks || blocks > self.block_capacity {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL published-prefix scan exceeds segment",
            ));
        }
        Ok(self.contiguous_published_prefix_unchecked(start_blocks, blocks))
    }

    pub fn sync_published_prefix(&self, blocks: u64) -> std::io::Result<()> {
        if blocks > self.block_capacity {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL sync prefix exceeds segment",
            ));
        }
        self.acquire_sync_slot();
        let result = self.sync_published_prefix_locked(blocks);
        self.sync_in_progress.store(false, Ordering::Release);
        result
    }

    pub fn sync_published_data_frontier(&self, blocks: u64) -> std::io::Result<()> {
        self.sync_published_data_frontier_with_mode(blocks, WalDataSyncMode::RangeAndFileData)
    }

    pub fn sync_published_data_frontier_with_mode(
        &self,
        blocks: u64,
        mode: WalDataSyncMode,
    ) -> std::io::Result<()> {
        if blocks > self.block_capacity {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL data sync frontier exceeds segment",
            ));
        }
        self.acquire_sync_slot();
        let result = self.sync_published_data_frontier_locked(blocks, mode);
        self.sync_in_progress.store(false, Ordering::Release);
        result
    }

    pub fn write_published_data_frontier(&self, blocks: u64) -> std::io::Result<u64> {
        if blocks > self.block_capacity {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL data write frontier exceeds segment",
            ));
        }
        self.write_published_data_frontier_locked(blocks)
    }

    fn sync_published_prefix_locked(&self, blocks: u64) -> std::io::Result<()> {
        let durable_blocks = self.durable_blocks.load(Ordering::Acquire);
        let control_generation = self.control_generation.load(Ordering::Acquire);
        if blocks <= durable_blocks && control_generation != 0 {
            return Ok(());
        }
        for block_id in durable_blocks..blocks {
            self.wait_for_published_block(block_id);
        }
        let start = if durable_blocks == 0 && control_generation == 0 {
            0
        } else {
            WAL_SEGMENT_HEADER_BYTES + durable_blocks as usize * self.block_stride
        };
        let end = WAL_SEGMENT_HEADER_BYTES + blocks as usize * self.block_stride;
        self.sync_range(start, end)?;
        self.file.sync_data()?;
        let next_generation = control_generation + 1;
        self.write_control_record(blocks, next_generation);
        self.sync_range(control_offset(0), control_offset(WAL_CONTROL_SLOTS))?;
        self.file.sync_data()?;
        self.durable_blocks.store(blocks, Ordering::Release);
        self.data_frontier_blocks
            .fetch_max(blocks, Ordering::AcqRel);
        self.prewrite_data_frontier_blocks
            .fetch_max(blocks, Ordering::AcqRel);
        self.write_data_frontier_blocks
            .fetch_max(blocks, Ordering::AcqRel);
        self.file_data_frontier_blocks
            .fetch_max(blocks, Ordering::AcqRel);
        self.control_generation
            .store(next_generation, Ordering::Release);
        Ok(())
    }

    fn sync_published_data_frontier_locked(
        &self,
        blocks: u64,
        mode: WalDataSyncMode,
    ) -> std::io::Result<()> {
        let start_blocks = match mode {
            WalDataSyncMode::RangeAndFileData => self.data_frontier_blocks.load(Ordering::Acquire),
            WalDataSyncMode::WriteAndFileData | WalDataSyncMode::PrewriteAndFileData => {
                self.write_data_frontier_blocks.load(Ordering::Acquire)
            }
            WalDataSyncMode::SyncWriteData => {
                self.file_data_frontier_blocks.load(Ordering::Acquire)
            }
            WalDataSyncMode::FileDataOnly => self.file_data_frontier_blocks.load(Ordering::Acquire),
        };
        if blocks <= start_blocks {
            return Ok(());
        }
        for block_id in start_blocks..blocks {
            self.wait_for_published_block(block_id);
        }
        if mode == WalDataSyncMode::RangeAndFileData {
            let start = if start_blocks == 0 {
                0
            } else {
                WAL_SEGMENT_HEADER_BYTES + start_blocks as usize * self.block_stride
            };
            let end = WAL_SEGMENT_HEADER_BYTES + blocks as usize * self.block_stride;
            self.sync_range(start, end)?;
        } else if matches!(
            mode,
            WalDataSyncMode::WriteAndFileData | WalDataSyncMode::PrewriteAndFileData
        ) {
            self.write_published_data_frontier_locked(blocks)?;
        } else if mode == WalDataSyncMode::SyncWriteData {
            self.sync_write_published_data_frontier_locked(blocks)?;
        }
        if mode != WalDataSyncMode::SyncWriteData {
            self.file.sync_data()?;
        }
        match mode {
            WalDataSyncMode::RangeAndFileData => {
                self.data_frontier_blocks.store(blocks, Ordering::Release);
                self.prewrite_data_frontier_blocks
                    .fetch_max(blocks, Ordering::AcqRel);
                self.write_data_frontier_blocks
                    .fetch_max(blocks, Ordering::AcqRel);
                self.file_data_frontier_blocks
                    .fetch_max(blocks, Ordering::AcqRel);
            }
            WalDataSyncMode::WriteAndFileData => {
                self.write_data_frontier_blocks
                    .store(blocks, Ordering::Release);
                self.file_data_frontier_blocks
                    .fetch_max(blocks, Ordering::AcqRel);
            }
            WalDataSyncMode::PrewriteAndFileData => {
                self.write_data_frontier_blocks
                    .store(blocks, Ordering::Release);
                self.file_data_frontier_blocks
                    .fetch_max(blocks, Ordering::AcqRel);
            }
            WalDataSyncMode::SyncWriteData => {
                self.prewrite_data_frontier_blocks
                    .fetch_max(blocks, Ordering::AcqRel);
                self.write_data_frontier_blocks
                    .fetch_max(blocks, Ordering::AcqRel);
                self.file_data_frontier_blocks
                    .store(blocks, Ordering::Release);
            }
            WalDataSyncMode::FileDataOnly => {
                self.file_data_frontier_blocks
                    .store(blocks, Ordering::Release);
            }
        }
        Ok(())
    }

    fn write_file_header(&self) {
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
            flags: 0,
            reserved: [0; 2],
        };
        unsafe {
            self.ptr
                .as_ptr()
                .cast::<WalSegmentFileHeader>()
                .write(header);
        }
    }

    fn write_control_record(&self, durable_blocks: u64, generation: u64) {
        let slot = generation as usize % WAL_CONTROL_SLOTS;
        let mut record = WalControlRecord {
            magic: WAL_CONTROL_MAGIC,
            version: WAL_SEGMENT_VERSION,
            slot: slot as u32,
            segment_id: self.segment_id,
            durable_blocks,
            generation,
            checksum: 0,
            reserved0: 0,
            reserved: 0,
            commit_marker: WAL_CONTROL_COMMIT_MARKER,
        };
        record.checksum = control_crc32c(&record);
        std::sync::atomic::compiler_fence(Ordering::Release);
        unsafe {
            self.ptr
                .as_ptr()
                .add(control_offset(slot))
                .cast::<WalControlRecord>()
                .write(record);
        }
    }

    fn claim_block(&self, count: usize) -> Result<u64, PublishError> {
        let block_size = self.block_size as usize;
        if count > block_size {
            return Err(PublishError::CountExceedsBlockSize { count, block_size });
        }
        let block_id = self.tail_block.fetch_add(1, Ordering::Relaxed);
        if block_id >= self.block_capacity {
            return Err(PublishError::CapacityExhausted {
                claimed_block: block_id,
                block_capacity: self.block_capacity,
            });
        }

        let state = &self.states[block_id as usize];
        while state.sequence.load(Ordering::Acquire) != block_id {
            spin_loop();
        }
        Ok(block_id)
    }

    fn acquire_sync_slot(&self) {
        let mut spins = 0;
        while self
            .sync_in_progress
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            if spins < 64 {
                spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }

    fn wait_for_published_block(&self, block_id: u64) {
        let state = &self.states[block_id as usize];
        let mut spins = 0;
        while state.sequence.load(Ordering::Acquire) != block_id + 1 {
            if spins < 256 {
                spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }

    fn contiguous_published_prefix_unchecked(&self, start_blocks: u64, blocks: u64) -> u64 {
        let mut prefix = start_blocks;
        while prefix < blocks {
            let state = &self.states[prefix as usize];
            if state.sequence.load(Ordering::Acquire) != prefix + 1 {
                break;
            }
            prefix += 1;
        }
        prefix
    }

    fn write_block_header(
        &self,
        block_id: u64,
        first_client_seq: u64,
        count: usize,
    ) -> WalBlockHeader {
        let payload_bytes = count * size_of::<WriteIntent>();
        let header = WalBlockHeader {
            magic: WAL_BLOCK_HEADER_MAGIC,
            block_id,
            first_client_seq,
            count: count as u32,
            block_size: self.block_size as u32,
            payload_bytes: payload_bytes as u32,
            reserved0: 0,
            reserved: [0; 3],
        };
        unsafe {
            self.block_header_ptr(block_id).write(header);
        }
        header
    }

    fn commit_block(&self, block_id: u64, header: WalBlockHeader) {
        let payload_bytes = header.payload_bytes as usize;
        let checksum = unsafe {
            block_crc32c(
                &header,
                std::slice::from_raw_parts(self.payload_ptr(block_id).cast::<u8>(), payload_bytes),
            )
        } as u64;
        let trailer = WalBlockTrailer {
            magic: WAL_BLOCK_TRAILER_MAGIC,
            block_id,
            first_client_seq: header.first_client_seq,
            count: header.count,
            payload_bytes: header.payload_bytes,
            checksum,
            reserved: [0; 2],
            commit_marker: WAL_BLOCK_COMMIT_MARKER,
        };
        std::sync::atomic::compiler_fence(Ordering::Release);
        unsafe {
            self.block_trailer_ptr(block_id).write(trailer);
        }
        let state = &self.states[block_id as usize];
        state.count.store(header.count as u64, Ordering::Relaxed);
        state
            .sequence
            .store(block_id.wrapping_add(1), Ordering::Release);
    }

    fn read_block_after_acquire(&self, block_id: u64) -> DrainStats {
        let count = self.states[block_id as usize].count.load(Ordering::Relaxed);
        let mut stats = DrainStats::default();
        for offset in 0..count {
            let intent = unsafe { self.payload_ptr(block_id).add(offset as usize).read() };
            stats.observe(intent);
        }
        stats
    }

    fn sync_range(&self, start: usize, end: usize) -> std::io::Result<()> {
        debug_assert!(start <= end);
        if start == end {
            return Ok(());
        }
        let page_size = page_size();
        let aligned_start = start / page_size * page_size;
        let aligned_end = end.div_ceil(page_size) * page_size;
        let aligned_end = aligned_end.min(self.bytes);
        let len = aligned_end - aligned_start;
        let rc = unsafe {
            libc::msync(
                self.ptr.as_ptr().add(aligned_start).cast(),
                len,
                libc::MS_SYNC,
            )
        };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    fn write_published_data_frontier_locked(&self, blocks: u64) -> std::io::Result<u64> {
        let mut scratch = self
            .write_scratch
            .lock()
            .map_err(|_| std::io::Error::other("WAL write scratch buffer poisoned"))?;
        let start_blocks = self.prewrite_data_frontier_blocks.load(Ordering::Acquire);
        if blocks <= start_blocks {
            return Ok(0);
        }
        for block_id in start_blocks..blocks {
            self.wait_for_published_block(block_id);
        }
        let start = if start_blocks == 0 {
            0
        } else {
            WAL_SEGMENT_HEADER_BYTES + start_blocks as usize * self.block_stride
        };
        let end = WAL_SEGMENT_HEADER_BYTES + blocks as usize * self.block_stride;
        if start > end || end > self.bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL write-through range exceeds segment",
            ));
        }
        scratch.clear();
        unsafe {
            scratch.extend_from_slice(std::slice::from_raw_parts(
                self.ptr.as_ptr().add(start),
                end - start,
            ));
        }
        write_all_at(&self.file, &scratch, start as u64)?;
        self.prewrite_data_frontier_blocks
            .fetch_max(blocks, Ordering::AcqRel);
        Ok(blocks - start_blocks)
    }

    fn sync_write_published_data_frontier_locked(&self, blocks: u64) -> std::io::Result<()> {
        let mut scratch = self
            .write_scratch
            .lock()
            .map_err(|_| std::io::Error::other("WAL write scratch buffer poisoned"))?;
        let start_blocks = self.file_data_frontier_blocks.load(Ordering::Acquire);
        if blocks <= start_blocks {
            return Ok(());
        }
        for block_id in start_blocks..blocks {
            self.wait_for_published_block(block_id);
        }
        let start = if start_blocks == 0 {
            0
        } else {
            WAL_SEGMENT_HEADER_BYTES + start_blocks as usize * self.block_stride
        };
        let end = WAL_SEGMENT_HEADER_BYTES + blocks as usize * self.block_stride;
        if start > end || end > self.bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL sync-write range exceeds segment",
            ));
        }
        scratch.clear();
        unsafe {
            scratch.extend_from_slice(std::slice::from_raw_parts(
                self.ptr.as_ptr().add(start),
                end - start,
            ));
        }
        write_all_at_dsync(&self.file, &scratch, start as u64)
    }

    unsafe fn block_header_ptr(&self, block_id: u64) -> *mut WalBlockHeader {
        self.ptr
            .as_ptr()
            .add(self.block_offset(block_id))
            .cast::<WalBlockHeader>()
    }

    unsafe fn payload_ptr(&self, block_id: u64) -> *mut WriteIntent {
        self.ptr
            .as_ptr()
            .add(self.block_offset(block_id) + size_of::<WalBlockHeader>())
            .cast::<WriteIntent>()
    }

    unsafe fn block_trailer_ptr(&self, block_id: u64) -> *mut WalBlockTrailer {
        self.ptr
            .as_ptr()
            .add(
                self.block_offset(block_id)
                    + size_of::<WalBlockHeader>()
                    + self.block_size as usize * size_of::<WriteIntent>(),
            )
            .cast::<WalBlockTrailer>()
    }

    fn block_offset(&self, block_id: u64) -> usize {
        WAL_SEGMENT_HEADER_BYTES + block_id as usize * self.block_stride
    }
}

impl Drop for MappedWalSegment {
    fn drop(&mut self) {
        let _ = unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.bytes) };
    }
}

pub struct MappedWalConsumer<'a> {
    segment: &'a MappedWalSegment,
    next_block: u64,
}

impl MappedWalConsumer<'_> {
    pub fn drain_available(&mut self, max_blocks: usize) -> DrainStats {
        let mut stats = DrainStats::default();
        for _ in 0..max_blocks {
            let block_id = self.next_block;
            if block_id >= self.segment.block_capacity {
                break;
            }
            let state = &self.segment.states[block_id as usize];
            if state.sequence.load(Ordering::Acquire) != block_id + 1 {
                break;
            }
            stats.add(self.segment.read_block_after_acquire(block_id));
            self.next_block += 1;
        }
        stats
    }
}

pub fn recover_wal_segment(path: impl AsRef<Path>) -> std::io::Result<WalRecovery> {
    recover_wal_segment_prefix(path, None)
}

pub fn recover_wal_segment_by_scan(path: impl AsRef<Path>) -> std::io::Result<WalRecovery> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let file_header: WalSegmentFileHeader = read_struct_at(&mut file, 0)?;
    validate_file_header(file_header)?;

    let block_size = file_header.block_size as usize;
    let block_capacity = file_header.block_capacity as usize;
    let stride = checked_block_stride(block_size)?;
    let expected_len = checked_segment_bytes(block_capacity, stride)? as u64;
    if file_len < expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "WAL segment shorter than declared layout",
        ));
    }

    let mut recovery = WalRecovery {
        segment_id: file_header.segment_id,
        block_size: file_header.block_size,
        block_capacity: file_header.block_capacity,
        recovered_blocks: 0,
        recovered_records: 0,
        stats: DrainStats::default(),
    };

    let expected_epoch = expected_block_epoch(&file_header);
    for block_id in 0..block_capacity as u64 {
        let base = WAL_SEGMENT_HEADER_BYTES as u64 + block_id * stride as u64;
        let header: WalBlockHeader = read_struct_at(&mut file, base)?;
        if !valid_header(&header, block_id, block_size, expected_epoch) {
            break;
        }

        let trailer_offset = base
            + size_of::<WalBlockHeader>() as u64
            + block_size as u64 * size_of::<WriteIntent>() as u64;
        let trailer: WalBlockTrailer = read_struct_at(&mut file, trailer_offset)?;
        if !valid_trailer(&header, &trailer) {
            break;
        }

        let payload = read_payload_at(
            &mut file,
            base + size_of::<WalBlockHeader>() as u64,
            header.count as usize,
        )?;
        let payload_bytes = unsafe {
            std::slice::from_raw_parts(
                payload.as_ptr().cast::<u8>(),
                payload.len() * size_of::<WriteIntent>(),
            )
        };
        if block_crc32c(&header, payload_bytes) as u64 != trailer.checksum {
            break;
        }

        let mut stats = DrainStats::default();
        for intent in payload.iter().copied() {
            stats.observe(intent);
        }
        recovery.recovered_blocks += 1;
        recovery.recovered_records += header.count as u64;
        recovery.stats.add(stats);
    }

    Ok(recovery)
}

fn recover_wal_segment_prefix(
    path: impl AsRef<Path>,
    max_blocks: Option<u64>,
) -> std::io::Result<WalRecovery> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let file_header: WalSegmentFileHeader = read_struct_at(&mut file, 0)?;
    validate_file_header(file_header)?;

    let block_size = file_header.block_size as usize;
    let block_capacity = file_header.block_capacity as usize;
    let stride = checked_block_stride(block_size)?;
    let expected_len = checked_segment_bytes(block_capacity, stride)? as u64;
    if file_len < expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "WAL segment shorter than declared layout",
        ));
    }
    let durable_blocks = recover_control_record(&mut file, file_header)?
        .map(|record| record.durable_blocks.min(file_header.block_capacity as u64))
        .unwrap_or(0);
    let recover_blocks = match max_blocks {
        Some(max_blocks) if durable_blocks < max_blocks => {
            return Err(invalid_data(
                "WAL segment durable prefix is behind manager control",
            ));
        }
        Some(max_blocks) => max_blocks,
        None => durable_blocks,
    };
    let mut recovery = WalRecovery {
        segment_id: file_header.segment_id,
        block_size: file_header.block_size,
        block_capacity: file_header.block_capacity,
        recovered_blocks: 0,
        recovered_records: 0,
        stats: DrainStats::default(),
    };

    let expected_epoch = expected_block_epoch(&file_header);
    for block_id in 0..recover_blocks {
        let base = WAL_SEGMENT_HEADER_BYTES as u64 + block_id * stride as u64;
        let header: WalBlockHeader = read_struct_at(&mut file, base)?;
        if !valid_header(&header, block_id, block_size, expected_epoch) {
            return Err(invalid_data(
                "invalid WAL block header inside durable prefix",
            ));
        }

        let trailer_offset = base
            + size_of::<WalBlockHeader>() as u64
            + block_size as u64 * size_of::<WriteIntent>() as u64;
        let trailer: WalBlockTrailer = read_struct_at(&mut file, trailer_offset)?;
        if !valid_trailer(&header, &trailer) {
            return Err(invalid_data(
                "invalid WAL block trailer inside durable prefix",
            ));
        }

        let payload = read_payload_at(
            &mut file,
            base + size_of::<WalBlockHeader>() as u64,
            header.count as usize,
        )?;
        let mut stats = DrainStats::default();
        for intent in payload.iter().copied() {
            stats.observe(intent);
        }
        let payload_bytes = unsafe {
            std::slice::from_raw_parts(
                payload.as_ptr().cast::<u8>(),
                payload.len() * size_of::<WriteIntent>(),
            )
        };
        if block_crc32c(&header, payload_bytes) as u64 != trailer.checksum {
            return Err(invalid_data(
                "WAL block checksum mismatch inside durable prefix",
            ));
        }

        recovery.recovered_blocks += 1;
        recovery.recovered_records += header.count as u64;
        recovery.stats.add(stats);
    }

    Ok(recovery)
}

pub fn recover_wal_manager(
    config: &WalSegmentManagerConfig,
) -> std::io::Result<WalManagerRecovery> {
    validate_manager_config(config)?;
    let mut control = File::open(config.control_path())?;
    let Some(control_record) = recover_manager_control_record(&mut control, config)? else {
        return Ok(WalManagerRecovery::default());
    };

    let mut recovered = WalManagerRecovery {
        first_segment_id: control_record.first_segment_id,
        durable_segment_id: control_record.durable_segment_id,
        durable_blocks: control_record.durable_blocks,
        recovered_segments: 0,
        recovered_records: 0,
        stats: DrainStats::default(),
    };

    let blocks_per_segment = config.blocks_per_segment();
    for segment_id in control_record.first_segment_id..=control_record.durable_segment_id {
        let expected_blocks = if segment_id == control_record.durable_segment_id {
            control_record.durable_blocks
        } else {
            blocks_per_segment
        };
        let segment_recovery =
            recover_wal_segment_prefix(config.segment_path(segment_id), Some(expected_blocks))?;
        if segment_recovery.segment_id != segment_id {
            return Err(invalid_data(
                "WAL manager segment recovery did not match durable control",
            ));
        }
        validate_manager_segment_layout(config, &segment_recovery)?;
        recovered.recovered_segments += 1;
        recovered.recovered_records += segment_recovery.recovered_records;
        recovered.stats.add(segment_recovery.stats);
    }

    Ok(recovered)
}

pub fn recover_wal_manager_by_scan(
    config: &WalSegmentManagerConfig,
) -> std::io::Result<WalManagerRecovery> {
    validate_manager_config(config)?;
    let control_record = match File::open(config.control_path()) {
        Ok(mut control) => recover_manager_control_record(&mut control, config)?,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => return Err(err),
    };
    let first_segment_id = control_record
        .map(|record| record.first_segment_id)
        .unwrap_or(0);
    let mut recovered = WalManagerRecovery {
        first_segment_id,
        durable_segment_id: first_segment_id,
        durable_blocks: 0,
        recovered_segments: 0,
        recovered_records: 0,
        stats: DrainStats::default(),
    };
    let blocks_per_segment = config.blocks_per_segment();
    let mut saw_segment = false;
    let mut saw_partial = false;
    let mut segment_id = first_segment_id;
    loop {
        let path = config.segment_path(segment_id);
        if !path.try_exists()? {
            break;
        }
        if saw_partial {
            return Err(invalid_data(
                "WAL manager scan recovery found segment after partial segment",
            ));
        }
        let segment_recovery = recover_wal_segment_by_scan(&path)?;
        if segment_recovery.segment_id != segment_id {
            return Err(invalid_data(
                "WAL manager scan recovery did not match segment id",
            ));
        }
        validate_manager_segment_layout(config, &segment_recovery)?;
        saw_segment = true;
        recovered.recovered_segments += 1;
        recovered.recovered_records += segment_recovery.recovered_records;
        recovered.stats.add(segment_recovery.stats);
        recovered.durable_segment_id = segment_id;
        recovered.durable_blocks = segment_recovery.recovered_blocks;
        if segment_recovery.recovered_blocks < blocks_per_segment {
            saw_partial = true;
        }
        segment_id += 1;
    }
    if !saw_segment {
        recovered.durable_segment_id = first_segment_id;
        recovered.durable_blocks = 0;
    }
    if let Some(record) = control_record {
        if recovered.durable_segment_id < record.durable_segment_id
            || (recovered.durable_segment_id == record.durable_segment_id
                && recovered.durable_blocks < record.durable_blocks)
        {
            return Err(invalid_data(
                "WAL manager scan recovery is behind durable control",
            ));
        }
    }
    Ok(recovered)
}

fn validate_manager_segment_layout(
    config: &WalSegmentManagerConfig,
    recovery: &WalRecovery,
) -> std::io::Result<()> {
    if recovery.block_size as usize != config.block_size
        || recovery.block_capacity as u64 != config.blocks_per_segment()
    {
        return Err(invalid_data(
            "WAL manager segment layout did not match config",
        ));
    }
    Ok(())
}

fn validate_file_header(header: WalSegmentFileHeader) -> std::io::Result<()> {
    if header.magic != WAL_SEGMENT_MAGIC
        || header.version != WAL_SEGMENT_VERSION
        || header.header_bytes as usize != WAL_SEGMENT_HEADER_BYTES
        || header.record_size as usize != size_of::<WriteIntent>()
        || header.block_header_size as usize != size_of::<WalBlockHeader>()
        || header.block_trailer_size as usize != size_of::<WalBlockTrailer>()
        || header.block_size == 0
        || header.block_capacity == 0
        || header.block_size as usize > u32::MAX as usize / size_of::<WriteIntent>()
        || header.flags & !WAL_SEGMENT_FLAG_EPOCH_STAMPED != 0
        || header.reserved != [0; 2]
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid WAL segment header",
        ));
    }
    Ok(())
}

/// Block-header epoch a scan must require for this segment file. Legacy
/// (non-epoch-stamped) segments wrote `reserved0 == 0`; epoch-stamped segments
/// write the segment id's low 32 bits so recycled files reject stale blocks.
fn expected_block_epoch(file_header: &WalSegmentFileHeader) -> u32 {
    if file_header.flags & WAL_SEGMENT_FLAG_EPOCH_STAMPED != 0 {
        file_header.segment_id as u32
    } else {
        0
    }
}

fn valid_header(
    header: &WalBlockHeader,
    block_id: u64,
    block_size: usize,
    expected_epoch: u32,
) -> bool {
    header.magic == WAL_BLOCK_HEADER_MAGIC
        && header.block_id == block_id
        && header.block_size as usize == block_size
        && header.count != 0
        && header.count as usize <= block_size
        && header.payload_bytes as usize == header.count as usize * size_of::<WriteIntent>()
        && header.reserved0 == expected_epoch
        && header.reserved == [0; 3]
}

fn valid_trailer(header: &WalBlockHeader, trailer: &WalBlockTrailer) -> bool {
    trailer.magic == WAL_BLOCK_TRAILER_MAGIC
        && trailer.block_id == header.block_id
        && trailer.first_client_seq == header.first_client_seq
        && trailer.count == header.count
        && trailer.payload_bytes == header.payload_bytes
        && trailer.reserved == [0; 2]
        && trailer.commit_marker == WAL_BLOCK_COMMIT_MARKER
}

fn recover_control_record(
    file: &mut File,
    file_header: WalSegmentFileHeader,
) -> std::io::Result<Option<WalControlRecord>> {
    let mut best: Option<WalControlRecord> = None;
    for slot in 0..WAL_CONTROL_SLOTS {
        let record: WalControlRecord = read_struct_at(file, control_offset(slot) as u64)?;
        if valid_control_record(&record, file_header, slot) {
            match best {
                Some(best_record) if best_record.generation >= record.generation => {}
                _ => best = Some(record),
            }
        }
    }
    Ok(best)
}

fn recover_manager_control_record(
    file: &mut File,
    config: &WalSegmentManagerConfig,
) -> std::io::Result<Option<WalManagerControlRecord>> {
    let mut best: Option<WalManagerControlRecord> = None;
    for slot in 0..WAL_CONTROL_SLOTS {
        let record: WalManagerControlRecord =
            read_struct_at(file, manager_control_offset(slot) as u64)?;
        if valid_manager_control_record(&record, config, slot) {
            match best {
                Some(best_record) if best_record.generation >= record.generation => {}
                _ => best = Some(record),
            }
        }
    }
    Ok(best)
}

fn valid_control_record(
    record: &WalControlRecord,
    file_header: WalSegmentFileHeader,
    slot: usize,
) -> bool {
    record.magic == WAL_CONTROL_MAGIC
        && record.version == WAL_SEGMENT_VERSION
        && record.slot as usize == slot
        && record.segment_id == file_header.segment_id
        && record.durable_blocks <= file_header.block_capacity as u64
        && record.generation != 0
        && record.reserved0 == 0
        && record.reserved == 0
        && record.commit_marker == WAL_CONTROL_COMMIT_MARKER
        && record.checksum == control_crc32c(record)
}

fn valid_manager_control_record(
    record: &WalManagerControlRecord,
    config: &WalSegmentManagerConfig,
    slot: usize,
) -> bool {
    record.magic == WAL_MANAGER_CONTROL_MAGIC
        && record.version == WAL_SEGMENT_VERSION
        && record.slot as usize == slot
        && record.generation != 0
        && record.first_segment_id <= record.durable_segment_id
        && record.records_per_segment as usize == config.records_per_segment
        && record.block_size as usize == config.block_size
        && record.durable_blocks <= config.blocks_per_segment()
        && record.reserved0 == 0
        && record.reserved == [0; 7]
        && record.commit_marker == WAL_MANAGER_CONTROL_COMMIT_MARKER
        && record.checksum == manager_control_crc32c(record)
}

fn validate_manager_config(config: &WalSegmentManagerConfig) -> std::io::Result<()> {
    if !valid_manager_prefix(&config.prefix) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "WAL manager prefix must be one plain path component",
        ));
    }
    if config.records_per_segment == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "WAL manager segment must contain records",
        ));
    }
    if config.block_size == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "WAL manager block size must be non-zero",
        ));
    }
    if config.records_per_segment > u32::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "WAL manager records per segment must fit on disk",
        ));
    }
    if config.block_size > u32::MAX as usize / size_of::<WriteIntent>() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "WAL manager block payload bytes must fit on disk",
        ));
    }
    Ok(())
}

fn valid_manager_prefix(prefix: &str) -> bool {
    let mut components = Path::new(prefix).components();
    matches!(components.next(), Some(Component::Normal(value)) if value == prefix)
        && components.next().is_none()
}

pub(crate) fn block_crc32c(header: &WalBlockHeader, payload: &[u8]) -> u32 {
    let crc = crc32c::crc32c_append(0, bytes_of(header));
    crc32c::crc32c_append(crc, payload)
}

fn control_crc32c(record: &WalControlRecord) -> u32 {
    let mut copy = *record;
    copy.checksum = 0;
    copy.commit_marker = 0;
    crc32c::crc32c(bytes_of(&copy))
}

fn manager_control_crc32c(record: &WalManagerControlRecord) -> u32 {
    let mut copy = *record;
    copy.checksum = 0;
    copy.commit_marker = 0;
    crc32c::crc32c(bytes_of(&copy))
}

pub(crate) fn checked_block_stride(block_size: usize) -> std::io::Result<usize> {
    size_of::<WalBlockHeader>()
        .checked_add(
            block_size
                .checked_mul(size_of::<WriteIntent>())
                .ok_or_else(|| invalid_data("WAL block payload size overflow"))?,
        )
        .and_then(|value| value.checked_add(size_of::<WalBlockTrailer>()))
        .ok_or_else(|| invalid_data("WAL block stride overflow"))
}

pub(crate) fn checked_segment_bytes(
    block_capacity: usize,
    block_stride: usize,
) -> std::io::Result<usize> {
    WAL_SEGMENT_HEADER_BYTES
        .checked_add(
            block_capacity
                .checked_mul(block_stride)
                .ok_or_else(|| invalid_data("WAL segment block bytes overflow"))?,
        )
        .ok_or_else(|| invalid_data("WAL segment bytes overflow"))
}

#[cfg(test)]
fn block_stride(block_size: usize) -> usize {
    checked_block_stride(block_size).expect("valid test WAL block stride")
}

fn control_offset(slot: usize) -> usize {
    size_of::<WalSegmentFileHeader>() + slot * size_of::<WalControlRecord>()
}

fn manager_control_offset(slot: usize) -> usize {
    slot * size_of::<WalManagerControlRecord>()
}

pub(crate) fn invalid_data(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

pub(crate) fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

fn page_size() -> usize {
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if value <= 0 {
        4096
    } else {
        value as usize
    }
}

pub(crate) fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> std::io::Result<()> {
    while !bytes.is_empty() {
        match file.write_at(bytes, offset) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to write WAL bytes",
                ));
            }
            Ok(written) => {
                bytes = &bytes[written..];
                offset += written as u64;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_all_at_dsync(file: &File, mut bytes: &[u8], mut offset: u64) -> std::io::Result<()> {
    while !bytes.is_empty() {
        let iov = libc::iovec {
            iov_base: bytes.as_ptr().cast_mut().cast(),
            iov_len: bytes.len(),
        };
        let written = unsafe {
            libc::pwritev2(
                file.as_raw_fd(),
                &iov,
                1,
                offset as libc::off_t,
                libc::RWF_DSYNC,
            )
        };
        if written == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "failed to sync-write WAL bytes",
            ));
        }
        if written > 0 {
            let written = written as usize;
            bytes = &bytes[written..];
            offset += written as u64;
            continue;
        }
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return Err(error);
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn write_all_at_dsync(_file: &File, _bytes: &[u8], _offset: u64) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "WAL sync-write-data mode requires Linux pwritev2(RWF_DSYNC)",
    ))
}

pub(crate) fn read_struct_at<T: Copy>(file: &mut File, offset: u64) -> std::io::Result<T> {
    file.seek(SeekFrom::Start(offset))?;
    let mut value = MaybeUninit::<T>::uninit();
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(value.as_mut_ptr().cast::<u8>(), size_of::<T>()) };
    file.read_exact(bytes)?;
    Ok(unsafe { value.assume_init() })
}

fn read_payload_at(
    file: &mut File,
    offset: u64,
    count: usize,
) -> std::io::Result<Vec<WriteIntent>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut payload = vec![WriteIntent::default(); count];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            payload.as_mut_ptr().cast::<u8>(),
            count * size_of::<WriteIntent>(),
        )
    };
    file.read_exact(bytes)?;
    Ok(payload)
}

#[cfg(test)]
mod tests;
