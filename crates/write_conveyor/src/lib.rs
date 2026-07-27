use std::cell::UnsafeCell;
#[cfg(unix)]
use std::fs::{File, OpenOptions};
use std::hint::spin_loop;
use std::mem::MaybeUninit;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(unix)]
use std::path::Path;
#[cfg(unix)]
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::{error::Error, fmt};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WriteIntent {
    pub route_id: u32,
    pub flags: u32,
    pub txn_id: u64,
    pub tenant_id: u64,
    pub key: u64,
    pub value0: u64,
    pub value1: u64,
    pub value2: u64,
    pub client_seq: u64,
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

    fn load(&self, ordering: Ordering) -> u64 {
        self.value.load(ordering)
    }

    fn store(&self, value: u64, ordering: Ordering) {
        self.value.store(value, ordering);
    }

    fn fetch_add(&self, value: u64, ordering: Ordering) -> u64 {
        self.value.fetch_add(value, ordering)
    }

    fn compare_exchange(
        &self,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        self.value.compare_exchange(current, new, success, failure)
    }
}

#[repr(align(64))]
struct CompletionFlag {
    sequence: AtomicU64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionError {
    OutOfRange {
        block_id: u64,
        capacity: u64,
    },
    AlreadyCompleted {
        block_id: u64,
        observed_sequence: u64,
    },
    NotLogged {
        block_id: u64,
        logged_prefix: u64,
    },
}

impl fmt::Display for CompletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { block_id, capacity } => {
                write!(
                    f,
                    "completion block id {block_id} is outside capacity {capacity}"
                )
            }
            Self::AlreadyCompleted {
                block_id,
                observed_sequence,
            } => {
                write!(
                    f,
                    "completion block id {block_id} was already marked with sequence {observed_sequence}"
                )
            }
            Self::NotLogged {
                block_id,
                logged_prefix,
            } => {
                write!(
                    f,
                    "completion block id {block_id} cannot be applied before logged prefix {logged_prefix}"
                )
            }
        }
    }
}

impl Error for CompletionError {}

pub struct ContiguousCompletionBarrier {
    capacity: u64,
    completed_prefix: PaddedAtomicU64,
    slots: Box<[CompletionFlag]>,
}

unsafe impl Send for ContiguousCompletionBarrier {}
unsafe impl Sync for ContiguousCompletionBarrier {}

impl ContiguousCompletionBarrier {
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(
            capacity < u64::MAX as usize,
            "completion barrier capacity must leave room for block_id + 1"
        );
        let slots = (0..capacity)
            .map(|_| CompletionFlag {
                sequence: AtomicU64::new(0),
            })
            .collect();
        Self {
            capacity: capacity as u64,
            completed_prefix: PaddedAtomicU64::new(0),
            slots,
        }
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn completed_prefix(&self) -> u64 {
        self.completed_prefix.load(Ordering::Acquire)
    }

    pub fn complete(&self, block_id: u64) -> Result<u64, CompletionError> {
        if block_id >= self.capacity {
            return Err(CompletionError::OutOfRange {
                block_id,
                capacity: self.capacity,
            });
        }
        let expected_sequence = block_id + 1;
        let slot = &self.slots[block_id as usize];
        match slot.sequence.compare_exchange(
            0,
            expected_sequence,
            Ordering::Release,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(observed_sequence) => {
                return Err(CompletionError::AlreadyCompleted {
                    block_id,
                    observed_sequence,
                });
            }
        }

        Ok(self.advance_contiguous())
    }

    pub fn wait_for(&self, target_prefix: u64) -> Result<(), CompletionError> {
        if target_prefix > self.capacity {
            return Err(CompletionError::OutOfRange {
                block_id: target_prefix,
                capacity: self.capacity,
            });
        }
        while self.completed_prefix.load(Ordering::Acquire) < target_prefix {
            if self.advance_contiguous() < target_prefix {
                spin_loop();
            }
        }
        Ok(())
    }

    pub fn wait_for_with_yield(
        &self,
        target_prefix: u64,
        spin_before_yield: u32,
    ) -> Result<(), CompletionError> {
        if target_prefix > self.capacity {
            return Err(CompletionError::OutOfRange {
                block_id: target_prefix,
                capacity: self.capacity,
            });
        }
        let mut spins = 0_u32;
        while self.completed_prefix.load(Ordering::Acquire) < target_prefix {
            if self.advance_contiguous() >= target_prefix {
                break;
            }
            if spins < spin_before_yield {
                spins += 1;
                spin_loop();
            } else {
                spins = 0;
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    fn advance_contiguous(&self) -> u64 {
        loop {
            let prefix = self.completed_prefix.load(Ordering::Acquire);
            let mut next = prefix;
            while next < self.capacity {
                let expected_sequence = next + 1;
                if self.slots[next as usize].sequence.load(Ordering::Acquire) != expected_sequence {
                    break;
                }
                next += 1;
            }
            if next == prefix {
                return prefix;
            }
            match self.completed_prefix.compare_exchange(
                prefix,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return next,
                Err(observed) if observed >= next => return observed,
                Err(_) => {}
            }
        }
    }
}

struct SpscSlot {
    payload: UnsafeCell<MaybeUninit<WriteIntent>>,
}

unsafe impl Sync for SpscSlot {}

pub struct SpscCursorRing {
    capacity: u64,
    mask: u64,
    slots: Box<[SpscSlot]>,
    published: PaddedAtomicU64,
    consumed: PaddedAtomicU64,
    producer_taken: AtomicBool,
    consumer_taken: AtomicBool,
}

unsafe impl Send for SpscCursorRing {}
unsafe impl Sync for SpscCursorRing {}

impl SpscCursorRing {
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(
            capacity.is_power_of_two(),
            "ring capacity must be a power of two"
        );
        assert!(capacity > 1, "ring capacity must be greater than one");
        let slots = (0..capacity)
            .map(|_| SpscSlot {
                payload: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect();
        let ring = Self {
            capacity: capacity as u64,
            mask: capacity as u64 - 1,
            slots,
            published: PaddedAtomicU64::new(0),
            consumed: PaddedAtomicU64::new(0),
            producer_taken: AtomicBool::new(false),
            consumer_taken: AtomicBool::new(false),
        };
        ring.pretouch_payloads();
        ring
    }

    pub fn producer(&self) -> SpscProducer<'_> {
        assert!(
            !self.producer_taken.swap(true, Ordering::AcqRel),
            "SPSC ring supports exactly one live producer"
        );
        SpscProducer {
            ring: self,
            next: 0,
        }
    }

    pub fn consumer(&self) -> SpscConsumer<'_> {
        assert!(
            !self.consumer_taken.swap(true, Ordering::AcqRel),
            "SPSC ring supports exactly one live consumer"
        );
        SpscConsumer {
            ring: self,
            next: 0,
        }
    }

    fn pretouch_payloads(&self) {
        for slot in self.slots.iter() {
            unsafe {
                (*slot.payload.get()).write(WriteIntent::default());
            }
        }
    }
}

pub struct SpscProducer<'a> {
    ring: &'a SpscCursorRing,
    next: u64,
}

impl SpscProducer<'_> {
    pub fn publish_batch(&mut self, first_client_seq: u64, count: usize) {
        if count == 0 {
            return;
        }
        assert!(
            count as u64 <= self.ring.capacity,
            "SPSC batch cannot exceed ring capacity"
        );
        let count = count as u64;
        while self
            .next
            .saturating_add(count)
            .saturating_sub(self.ring.consumed.load(Ordering::Acquire))
            > self.ring.capacity
        {
            spin_loop();
        }
        let start = self.next;
        for offset in 0..count {
            let client_seq = first_client_seq.wrapping_add(offset);
            let intent = intent_for(client_seq);
            let idx = ((start + offset) & self.ring.mask) as usize;
            unsafe {
                (*self.ring.slots[idx].payload.get()).write(intent);
            }
        }
        self.next += count;
        self.ring.published.store(self.next, Ordering::Release);
    }
}

pub struct SpscConsumer<'a> {
    ring: &'a SpscCursorRing,
    next: u64,
}

impl SpscConsumer<'_> {
    pub fn drain_available(&mut self, max: usize) -> DrainStats {
        let published = self.ring.published.load(Ordering::Acquire);
        let available = published.saturating_sub(self.next).min(max as u64);
        let mut stats = DrainStats::default();
        for offset in 0..available {
            let idx = ((self.next + offset) & self.ring.mask) as usize;
            let intent = unsafe { (*self.ring.slots[idx].payload.get()).assume_init_read() };
            stats.observe(intent);
        }
        if available != 0 {
            self.next += available;
            self.ring.consumed.store(self.next, Ordering::Release);
        }
        stats
    }
}

pub struct StagedBlockConveyor {
    logged: ContiguousCompletionBarrier,
    applied: ContiguousCompletionBarrier,
}

unsafe impl Send for StagedBlockConveyor {}
unsafe impl Sync for StagedBlockConveyor {}

impl StagedBlockConveyor {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            logged: ContiguousCompletionBarrier::with_capacity(capacity),
            applied: ContiguousCompletionBarrier::with_capacity(capacity),
        }
    }

    pub fn capacity(&self) -> u64 {
        self.logged.capacity()
    }

    pub fn logged_prefix(&self) -> u64 {
        self.logged.completed_prefix()
    }

    pub fn applied_prefix(&self) -> u64 {
        self.applied.completed_prefix()
    }

    pub fn mark_logged(&self, block_id: u64) -> Result<u64, CompletionError> {
        self.logged.complete(block_id)
    }

    pub fn wait_logged(&self, target_prefix: u64) -> Result<(), CompletionError> {
        self.logged.wait_for(target_prefix)
    }

    pub fn mark_applied(&self, block_id: u64) -> Result<u64, CompletionError> {
        if block_id >= self.capacity() {
            return Err(CompletionError::OutOfRange {
                block_id,
                capacity: self.capacity(),
            });
        }
        let logged_prefix = self.logged_prefix();
        if logged_prefix <= block_id {
            return Err(CompletionError::NotLogged {
                block_id,
                logged_prefix,
            });
        }
        self.applied.complete(block_id)
    }

    pub fn wait_applied(&self, target_prefix: u64) -> Result<(), CompletionError> {
        self.applied.wait_for(target_prefix)
    }
}

#[repr(align(64))]
struct ShardBlockState {
    state: AtomicU64,
    count: AtomicU64,
}

struct ColumnSlot<T> {
    value: UnsafeCell<MaybeUninit<T>>,
}

unsafe impl<T: Send> Sync for ColumnSlot<T> {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreApplyError {
    CountExceedsBlockSize { count: usize, block_size: usize },
    BlockOutOfRange { block_id: u64, block_capacity: u64 },
    DuplicateBlock { block_id: u64, observed_state: u64 },
}

impl fmt::Display for StoreApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountExceedsBlockSize { count, block_size } => {
                write!(
                    f,
                    "store apply count {count} exceeds block size {block_size}"
                )
            }
            Self::BlockOutOfRange {
                block_id,
                block_capacity,
            } => {
                write!(
                    f,
                    "store apply block id {block_id} is outside capacity {block_capacity}"
                )
            }
            Self::DuplicateBlock {
                block_id,
                observed_state,
            } => {
                write!(
                    f,
                    "store apply block id {block_id} was already claimed with state {observed_state}"
                )
            }
        }
    }
}

impl Error for StoreApplyError {}

pub struct OpenShardAppendStore {
    block_size: u64,
    block_capacity: u64,
    row_capacity: usize,
    states: Box<[ShardBlockState]>,
    row_id: Box<[ColumnSlot<u64>]>,
    created_seq: Box<[ColumnSlot<u64>]>,
    route_id: Box<[ColumnSlot<u32>]>,
    flags: Box<[ColumnSlot<u32>]>,
    txn_id: Box<[ColumnSlot<u64>]>,
    tenant_id: Box<[ColumnSlot<u64>]>,
    key: Box<[ColumnSlot<u64>]>,
    value0: Box<[ColumnSlot<u64>]>,
    value1: Box<[ColumnSlot<u64>]>,
    value2: Box<[ColumnSlot<u64>]>,
}

unsafe impl Send for OpenShardAppendStore {}
unsafe impl Sync for OpenShardAppendStore {}

impl OpenShardAppendStore {
    pub fn with_capacity(block_capacity: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "store block size must be non-zero");
        let row_capacity = block_capacity
            .checked_mul(block_size)
            .expect("store row capacity overflow");
        let states = (0..block_capacity)
            .map(|_| ShardBlockState {
                state: AtomicU64::new(0),
                count: AtomicU64::new(0),
            })
            .collect();
        let store = Self {
            block_size: block_size as u64,
            block_capacity: block_capacity as u64,
            row_capacity,
            states,
            row_id: column_slots(row_capacity),
            created_seq: column_slots(row_capacity),
            route_id: column_slots(row_capacity),
            flags: column_slots(row_capacity),
            txn_id: column_slots(row_capacity),
            tenant_id: column_slots(row_capacity),
            key: column_slots(row_capacity),
            value0: column_slots(row_capacity),
            value1: column_slots(row_capacity),
            value2: column_slots(row_capacity),
        };
        store.pretouch();
        store
    }

    pub fn row_capacity(&self) -> usize {
        self.row_capacity
    }

    pub fn block_capacity(&self) -> u64 {
        self.block_capacity
    }

    pub fn apply_block(
        &self,
        global_block_id: u64,
        intents: &[WriteIntent],
    ) -> Result<DrainStats, StoreApplyError> {
        if intents.len() > self.block_size as usize {
            return Err(StoreApplyError::CountExceedsBlockSize {
                count: intents.len(),
                block_size: self.block_size as usize,
            });
        }
        if global_block_id >= self.block_capacity {
            return Err(StoreApplyError::BlockOutOfRange {
                block_id: global_block_id,
                block_capacity: self.block_capacity,
            });
        }
        let state = &self.states[global_block_id as usize];
        match state
            .state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {}
            Err(observed_state) => {
                return Err(StoreApplyError::DuplicateBlock {
                    block_id: global_block_id,
                    observed_state,
                });
            }
        }

        let base = global_block_id as usize * self.block_size as usize;
        let mut stats = DrainStats::default();
        for (offset, intent) in intents.iter().copied().enumerate() {
            let row = base + offset;
            unsafe {
                write_column(&self.row_id, row, row as u64);
                write_column(&self.created_seq, row, intent.client_seq);
                write_column(&self.route_id, row, intent.route_id);
                write_column(&self.flags, row, intent.flags);
                write_column(&self.txn_id, row, intent.txn_id);
                write_column(&self.tenant_id, row, intent.tenant_id);
                write_column(&self.key, row, intent.key);
                write_column(&self.value0, row, intent.value0);
                write_column(&self.value1, row, intent.value1);
                write_column(&self.value2, row, intent.value2);
            }
            stats.observe(intent);
        }
        state.count.store(intents.len() as u64, Ordering::Relaxed);
        state.state.store(2, Ordering::Release);
        Ok(stats)
    }

    pub fn read_block_stats(&self, block_id: u64) -> Option<DrainStats> {
        if block_id >= self.block_capacity {
            return None;
        }
        let state = &self.states[block_id as usize];
        if state.state.load(Ordering::Acquire) != 2 {
            return None;
        }
        let count = state.count.load(Ordering::Relaxed) as usize;
        let base = block_id as usize * self.block_size as usize;
        let mut stats = DrainStats::default();
        for offset in 0..count {
            stats.observe(unsafe { self.read_intent(base + offset) });
        }
        Some(stats)
    }

    pub fn read_row_id(&self, row: usize) -> Option<u64> {
        (row < self.row_capacity).then(|| unsafe { read_column(&self.row_id, row) })
    }

    pub fn read_created_seq(&self, row: usize) -> Option<u64> {
        (row < self.row_capacity).then(|| unsafe { read_column(&self.created_seq, row) })
    }

    pub fn validate_applied_blocks(&self, blocks: u64) -> Option<DrainStats> {
        if blocks > self.block_capacity {
            return None;
        }
        let mut stats = DrainStats::default();
        for block_id in 0..blocks {
            let state = &self.states[block_id as usize];
            if state.state.load(Ordering::Acquire) != 2 {
                return None;
            }
            let count = state.count.load(Ordering::Relaxed) as usize;
            let base = block_id as usize * self.block_size as usize;
            for offset in 0..count {
                let row = base + offset;
                let row_id = unsafe { read_column(&self.row_id, row) };
                if row_id != row as u64 {
                    return None;
                }
                stats.observe(unsafe { self.read_intent(row) });
            }
        }
        Some(stats)
    }

    pub fn global_block_id(segment_id: u64, blocks_per_segment: u64, block_id: u64) -> u64 {
        segment_id
            .checked_mul(blocks_per_segment)
            .and_then(|base| base.checked_add(block_id))
            .expect("global store block id overflow")
    }

    fn pretouch(&self) {
        for row in 0..self.row_capacity {
            unsafe {
                write_column(&self.row_id, row, 0);
                write_column(&self.created_seq, row, 0);
                write_column(&self.route_id, row, 0);
                write_column(&self.flags, row, 0);
                write_column(&self.txn_id, row, 0);
                write_column(&self.tenant_id, row, 0);
                write_column(&self.key, row, 0);
                write_column(&self.value0, row, 0);
                write_column(&self.value1, row, 0);
                write_column(&self.value2, row, 0);
            }
        }
    }

    unsafe fn read_intent(&self, row: usize) -> WriteIntent {
        WriteIntent {
            route_id: read_column(&self.route_id, row),
            flags: read_column(&self.flags, row),
            txn_id: read_column(&self.txn_id, row),
            tenant_id: read_column(&self.tenant_id, row),
            key: read_column(&self.key, row),
            value0: read_column(&self.value0, row),
            value1: read_column(&self.value1, row),
            value2: read_column(&self.value2, row),
            client_seq: read_column(&self.created_seq, row),
        }
    }
}

fn column_slots<T>(len: usize) -> Box<[ColumnSlot<T>]> {
    (0..len)
        .map(|_| ColumnSlot {
            value: UnsafeCell::new(MaybeUninit::uninit()),
        })
        .collect()
}

unsafe fn write_column<T: Copy>(column: &[ColumnSlot<T>], row: usize, value: T) {
    (*column[row].value.get()).write(value);
}

unsafe fn read_column<T: Copy>(column: &[ColumnSlot<T>], row: usize) -> T {
    (*column[row].value.get()).assume_init_read()
}

#[repr(align(128))]
struct MpscSlot {
    sequence: AtomicU64,
    payload: UnsafeCell<MaybeUninit<WriteIntent>>,
}

unsafe impl Sync for MpscSlot {}

pub struct MpscSequencedRing {
    capacity: u64,
    mask: u64,
    slots: Box<[MpscSlot]>,
    tail: PaddedAtomicU64,
    consumer_taken: AtomicBool,
}

unsafe impl Send for MpscSequencedRing {}
unsafe impl Sync for MpscSequencedRing {}

impl MpscSequencedRing {
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(
            capacity.is_power_of_two(),
            "ring capacity must be a power of two"
        );
        assert!(capacity > 1, "ring capacity must be greater than one");
        let slots = (0..capacity)
            .map(|sequence| MpscSlot {
                sequence: AtomicU64::new(sequence as u64),
                payload: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect();
        let ring = Self {
            capacity: capacity as u64,
            mask: capacity as u64 - 1,
            slots,
            tail: PaddedAtomicU64::new(0),
            consumer_taken: AtomicBool::new(false),
        };
        ring.pretouch_payloads();
        ring
    }

    pub fn publish_batch(&self, first_client_seq: u64, count: usize) {
        if count == 0 {
            return;
        }
        let start = self.claim_batch(count);
        self.publish_claimed_batch(start, first_client_seq, count);
    }

    fn claim_batch(&self, count: usize) -> u64 {
        debug_assert!(count > 0);
        self.tail.fetch_add(count as u64, Ordering::Relaxed)
    }

    fn publish_claimed(&self, sequence: u64, intent: WriteIntent) {
        let slot = &self.slots[(sequence & self.mask) as usize];
        while slot.sequence.load(Ordering::Acquire) != sequence {
            spin_loop();
        }
        unsafe {
            (*slot.payload.get()).write(intent);
        }
        slot.sequence
            .store(sequence.wrapping_add(1), Ordering::Release);
    }

    fn publish_claimed_batch(&self, start: u64, first_client_seq: u64, count: usize) {
        for offset in 0..count as u64 {
            self.publish_claimed(
                start.wrapping_add(offset),
                intent_for(first_client_seq.wrapping_add(offset)),
            );
        }
    }

    pub fn consumer(&self) -> MpscConsumer<'_> {
        assert!(
            !self.consumer_taken.swap(true, Ordering::AcqRel),
            "MPSC ring supports exactly one live consumer"
        );
        MpscConsumer {
            ring: self,
            next: 0,
        }
    }

    fn pretouch_payloads(&self) {
        for slot in self.slots.iter() {
            unsafe {
                (*slot.payload.get()).write(WriteIntent::default());
            }
        }
    }
}

pub struct MpscConsumer<'a> {
    ring: &'a MpscSequencedRing,
    next: u64,
}

impl MpscConsumer<'_> {
    pub fn drain_available(&mut self, max: usize) -> DrainStats {
        let mut stats = DrainStats::default();
        for _ in 0..max {
            let sequence = self.next;
            let slot = &self.ring.slots[(sequence & self.ring.mask) as usize];
            if slot.sequence.load(Ordering::Acquire) != sequence + 1 {
                break;
            }
            let intent = unsafe { (*slot.payload.get()).assume_init_read() };
            stats.observe(intent);
            slot.sequence
                .store(sequence + self.ring.capacity, Ordering::Release);
            self.next += 1;
        }
        stats
    }
}

#[repr(align(128))]
struct BlockState {
    sequence: AtomicU64,
    count: AtomicU64,
}

pub struct MpscBlockRing {
    block_size: u64,
    block_capacity: u64,
    block_mask: u64,
    states: Box<[BlockState]>,
    payloads: Box<[SpscSlot]>,
    tail_block: PaddedAtomicU64,
    consumer_taken: AtomicBool,
}

unsafe impl Send for MpscBlockRing {}
unsafe impl Sync for MpscBlockRing {}

impl MpscBlockRing {
    pub fn with_capacity(capacity: usize, block_size: usize) -> Self {
        assert!(
            capacity.is_power_of_two(),
            "ring capacity must be a power of two"
        );
        assert!(
            block_size.is_power_of_two(),
            "block size must be a power of two"
        );
        assert!(block_size > 0, "block size must be non-zero");
        assert!(
            capacity >= block_size * 2,
            "capacity must hold at least two blocks"
        );
        assert_eq!(
            capacity % block_size,
            0,
            "capacity must be a multiple of block size"
        );
        let block_capacity = capacity / block_size;
        assert!(
            block_capacity.is_power_of_two(),
            "block capacity must be a power of two"
        );
        let states = (0..block_capacity)
            .map(|sequence| BlockState {
                sequence: AtomicU64::new(sequence as u64),
                count: AtomicU64::new(0),
            })
            .collect();
        let payloads = (0..capacity)
            .map(|_| SpscSlot {
                payload: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect();
        let ring = Self {
            block_size: block_size as u64,
            block_capacity: block_capacity as u64,
            block_mask: block_capacity as u64 - 1,
            states,
            payloads,
            tail_block: PaddedAtomicU64::new(0),
            consumer_taken: AtomicBool::new(false),
        };
        ring.pretouch_payloads();
        ring
    }

    pub fn publish_block(&self, first_client_seq: u64, count: usize) {
        if count == 0 {
            return;
        }
        assert!(
            count as u64 <= self.block_size,
            "block publish count cannot exceed block size"
        );
        let block = self.claim_block();
        self.publish_claimed_block(block, first_client_seq, count);
    }

    fn claim_block(&self) -> u64 {
        self.tail_block.fetch_add(1, Ordering::Relaxed)
    }

    fn publish_claimed_block(&self, block_id: u64, first_client_seq: u64, count: usize) {
        let block_idx = (block_id & self.block_mask) as usize;
        let state = &self.states[block_idx];
        while state.sequence.load(Ordering::Acquire) != block_id {
            spin_loop();
        }
        let base = block_idx as u64 * self.block_size;
        for offset in 0..count as u64 {
            let intent = intent_for(first_client_seq.wrapping_add(offset));
            let idx = (base + offset) as usize;
            unsafe {
                (*self.payloads[idx].payload.get()).write(intent);
            }
        }
        state.count.store(count as u64, Ordering::Relaxed);
        state
            .sequence
            .store(block_id.wrapping_add(1), Ordering::Release);
    }

    pub fn consumer(&self) -> MpscBlockConsumer<'_> {
        assert!(
            !self.consumer_taken.swap(true, Ordering::AcqRel),
            "MPSC block ring supports exactly one live consumer"
        );
        MpscBlockConsumer {
            ring: self,
            next_block: 0,
        }
    }

    fn pretouch_payloads(&self) {
        for slot in self.payloads.iter() {
            unsafe {
                (*slot.payload.get()).write(WriteIntent::default());
            }
        }
    }
}

pub struct MpscBlockConsumer<'a> {
    ring: &'a MpscBlockRing,
    next_block: u64,
}

impl MpscBlockConsumer<'_> {
    pub fn drain_available(&mut self, max_blocks: usize) -> DrainStats {
        let mut stats = DrainStats::default();
        for _ in 0..max_blocks {
            let block_id = self.next_block;
            let block_idx = (block_id & self.ring.block_mask) as usize;
            let state = &self.ring.states[block_idx];
            if state.sequence.load(Ordering::Acquire) != block_id + 1 {
                break;
            }
            let count = state.count.load(Ordering::Relaxed);
            let base = block_idx as u64 * self.ring.block_size;
            for offset in 0..count {
                let idx = (base + offset) as usize;
                let intent = unsafe { (*self.ring.payloads[idx].payload.get()).assume_init_read() };
                stats.observe(intent);
            }
            state
                .sequence
                .store(block_id + self.ring.block_capacity, Ordering::Release);
            self.next_block += 1;
        }
        stats
    }
}

#[cfg(unix)]
pub struct MappedJournal {
    file: File,
    ptr: NonNull<WriteIntent>,
    records: usize,
    bytes: usize,
}

#[cfg(unix)]
impl MappedJournal {
    /// # Safety
    ///
    /// The caller must ensure the created path is exclusively owned for the
    /// lifetime of the returned mapping. No other process or mapping may
    /// truncate, resize, or write the file while this journal is alive.
    pub unsafe fn create(path: impl AsRef<Path>, records: usize) -> std::io::Result<Self> {
        assert!(
            records > 0,
            "mapped journal must contain at least one record"
        );
        let path = path.as_ref();
        let bytes = records
            .checked_mul(std::mem::size_of::<WriteIntent>())
            .expect("mapped journal byte length overflow");
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
        let ptr = NonNull::new(raw.cast::<WriteIntent>()).expect("mmap returned null");
        let mut journal = Self {
            file,
            ptr,
            records,
            bytes,
        };
        journal.prefault();
        Ok(journal)
    }

    pub fn write(&mut self, index: usize, intent: WriteIntent) {
        assert!(index < self.records, "mapped journal write out of bounds");
        unsafe {
            self.ptr.as_ptr().add(index).write(intent);
        }
    }

    pub fn read(&self, index: usize) -> WriteIntent {
        assert!(index < self.records, "mapped journal read out of bounds");
        unsafe { self.ptr.as_ptr().add(index).read() }
    }

    pub fn sync_mapping(&mut self) -> std::io::Result<()> {
        let rc = unsafe { libc::msync(self.ptr.as_ptr().cast(), self.bytes, libc::MS_SYNC) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        self.file.sync_data()
    }

    fn prefault(&mut self) {
        for index in 0..self.records {
            self.write(index, WriteIntent::default());
        }
    }
}

#[cfg(unix)]
impl Drop for MappedJournal {
    fn drop(&mut self) {
        let _ = unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.bytes) };
    }
}

#[cfg(unix)]
pub struct MappedBlockJournal {
    file: File,
    ptr: NonNull<WriteIntent>,
    records: usize,
    bytes: usize,
    block_size: u64,
    block_capacity: u64,
    states: Box<[BlockState]>,
    tail_block: PaddedAtomicU64,
    consumer_taken: AtomicBool,
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublishError {
    CountExceedsBlockSize {
        count: usize,
        block_size: usize,
    },
    CapacityExhausted {
        claimed_block: u64,
        block_capacity: u64,
    },
}

#[cfg(unix)]
impl fmt::Display for PublishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountExceedsBlockSize { count, block_size } => {
                write!(
                    f,
                    "mapped block publish count {count} exceeds block size {block_size}"
                )
            }
            Self::CapacityExhausted {
                claimed_block,
                block_capacity,
            } => {
                write!(
                    f,
                    "mapped block journal capacity exhausted: claimed block {claimed_block}, capacity {block_capacity}"
                )
            }
        }
    }
}

#[cfg(unix)]
impl Error for PublishError {}

#[cfg(unix)]
unsafe impl Send for MappedBlockJournal {}
#[cfg(unix)]
unsafe impl Sync for MappedBlockJournal {}

#[cfg(unix)]
impl MappedBlockJournal {
    /// # Safety
    ///
    /// The caller must ensure the created path is exclusively owned for the
    /// lifetime of the returned mapping. No other process or mapping may
    /// truncate, resize, or write the file while this journal is alive. Final
    /// durability sync requires writer quiescence and therefore takes `&mut self`.
    pub unsafe fn create(
        path: impl AsRef<Path>,
        records: usize,
        block_size: usize,
    ) -> std::io::Result<Self> {
        assert!(
            records > 0,
            "mapped block journal must contain at least one record"
        );
        assert!(
            block_size.is_power_of_two(),
            "block size must be a power of two"
        );
        assert!(block_size > 0, "block size must be non-zero");
        let path = path.as_ref();
        let block_capacity = records.div_ceil(block_size);
        let mapped_records = block_capacity
            .checked_mul(block_size)
            .expect("mapped block journal record count overflow");
        let bytes = mapped_records
            .checked_mul(std::mem::size_of::<WriteIntent>())
            .expect("mapped block journal byte length overflow");
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
        let states = (0..block_capacity)
            .map(|sequence| BlockState {
                sequence: AtomicU64::new(sequence as u64),
                count: AtomicU64::new(0),
            })
            .collect();
        let ptr = NonNull::new(raw.cast::<WriteIntent>()).expect("mmap returned null");
        let mut journal = Self {
            file,
            ptr,
            records: mapped_records,
            bytes,
            block_size: block_size as u64,
            block_capacity: block_capacity as u64,
            states,
            tail_block: PaddedAtomicU64::new(0),
            consumer_taken: AtomicBool::new(false),
        };
        journal.prefault();
        Ok(journal)
    }

    pub fn publish_block(&self, first_client_seq: u64, count: usize) {
        self.try_publish_block(first_client_seq, count)
            .expect("mapped block publish failed");
    }

    pub fn try_publish_block(
        &self,
        first_client_seq: u64,
        count: usize,
    ) -> Result<(), PublishError> {
        if count == 0 {
            return Ok(());
        }
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
        let base = block_id * self.block_size;
        for offset in 0..count as u64 {
            let intent = intent_for(first_client_seq.wrapping_add(offset));
            unsafe {
                self.ptr
                    .as_ptr()
                    .add((base + offset) as usize)
                    .write(intent);
            }
        }
        state.count.store(count as u64, Ordering::Relaxed);
        state
            .sequence
            .store(block_id.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    pub fn consumer(&self) -> MappedBlockConsumer<'_> {
        assert!(
            !self.consumer_taken.swap(true, Ordering::AcqRel),
            "mapped block journal supports exactly one live consumer"
        );
        MappedBlockConsumer {
            journal: self,
            next_block: 0,
        }
    }

    pub fn block_capacity(&self) -> u64 {
        self.block_capacity
    }

    pub fn read_published_block(&self, block_id: u64) -> Option<DrainStats> {
        if block_id >= self.block_capacity {
            return None;
        }
        let state = &self.states[block_id as usize];
        if state.sequence.load(Ordering::Acquire) != block_id + 1 {
            return None;
        }
        Some(self.drain_block_after_acquire(block_id))
    }

    pub fn sync_mapping(&mut self) -> std::io::Result<()> {
        let rc = unsafe { libc::msync(self.ptr.as_ptr().cast(), self.bytes, libc::MS_SYNC) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        self.file.sync_data()
    }

    fn prefault(&mut self) {
        for index in 0..self.records {
            unsafe {
                self.ptr.as_ptr().add(index).write(WriteIntent::default());
            }
        }
    }

    fn drain_block_after_acquire(&self, block_id: u64) -> DrainStats {
        let count = self.states[block_id as usize].count.load(Ordering::Relaxed);
        let base = block_id * self.block_size;
        let mut stats = DrainStats::default();
        for offset in 0..count {
            let intent = unsafe { self.ptr.as_ptr().add((base + offset) as usize).read() };
            stats.observe(intent);
        }
        stats
    }
}

#[cfg(unix)]
impl Drop for MappedBlockJournal {
    fn drop(&mut self) {
        let _ = unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.bytes) };
    }
}

#[cfg(unix)]
pub struct MappedBlockConsumer<'a> {
    journal: &'a MappedBlockJournal,
    next_block: u64,
}

#[cfg(unix)]
impl MappedBlockConsumer<'_> {
    pub fn drain_available(&mut self, max_blocks: usize) -> DrainStats {
        let mut stats = DrainStats::default();
        for _ in 0..max_blocks {
            let block_id = self.next_block;
            if block_id >= self.journal.block_capacity {
                break;
            }
            let state = &self.journal.states[block_id as usize];
            if state.sequence.load(Ordering::Acquire) != block_id + 1 {
                break;
            }
            stats.add(self.journal.drain_block_after_acquire(block_id));
            self.next_block += 1;
        }
        stats
    }
}

#[cfg(unix)]
fn preallocate_file(file: &File, bytes: usize) -> std::io::Result<()> {
    let len: libc::off_t = bytes.try_into().map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "mapped journal byte length exceeds off_t",
        )
    })?;
    let rc = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, len) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::from_raw_os_error(rc))
    }
}

#[cfg(unix)]
fn sync_parent_dir(path: &Path) -> std::io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrainStats {
    pub count: u64,
    pub checksum: u64,
}

impl DrainStats {
    pub fn observe(&mut self, intent: WriteIntent) {
        self.count += 1;
        self.checksum ^= (intent.route_id as u64).wrapping_mul(0xD6E8_FD9D_50A9_20D5)
            ^ (intent.flags as u64).rotate_left(7)
            ^ intent.tenant_id.rotate_left(23)
            ^ intent
                .txn_id
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left((intent.client_seq & 63) as u32)
            ^ intent.key
            ^ intent.value0
            ^ intent.value1
            ^ intent.value2;
    }

    pub fn add(&mut self, other: DrainStats) {
        self.count += other.count;
        self.checksum ^= other.checksum;
    }
}

pub fn intent_for(client_seq: u64) -> WriteIntent {
    WriteIntent {
        route_id: 1,
        flags: 0,
        txn_id: client_seq.wrapping_add(1),
        tenant_id: client_seq >> 20,
        key: client_seq.wrapping_mul(2).wrapping_add(1),
        value0: client_seq ^ 0xA5A5_A5A5_A5A5_A5A5,
        value1: client_seq.rotate_left(17),
        value2: !client_seq,
        client_seq,
    }
}

pub fn stats_for_range(first_client_seq: u64, count: u64) -> DrainStats {
    let mut stats = DrainStats::default();
    for offset in 0..count {
        stats.observe(intent_for(first_client_seq.wrapping_add(offset)));
    }
    stats
}

#[cfg(unix)]
mod fua_controller;
#[cfg(unix)]
mod fua_frame_log;
#[cfg(unix)]
mod fua_wal;
#[cfg(unix)]
pub use fua_controller::{
    FuaControllerDecision, FuaControllerEligibility, FuaControllerPhase, FuaControllerReason,
    FuaControllerSampleKind, FuaControllerSampleToken, FuaControllerTelemetry,
    FuaPhysicalController, FUA_CONTROLLER_FAST_NANOS, FUA_CONTROLLER_QD16_FRAGMENTS,
    FUA_CONTROLLER_SUSTAINED_GROUPS,
};
#[cfg(unix)]
pub use fua_frame_log::{
    frame_log_capacity_bytes, fua_frame_padded_bytes, invalidate_frame_log_suffix,
    recover_frame_log_by_scan, FrameBatchHandle, FrameGroupMetadata, FrameHandle, FuaFrameLog,
    FuaFrameLogAppender, FuaFrameLogConfig, FuaFrameLogFencePool, FuaFrameLogTelemetry,
    RecoveredFrame, FUA_IN_FLIGHT_DEPTH_BUCKETS,
};
#[cfg(unix)]
mod wal_segment;
#[cfg(unix)]
pub use fua_wal::{
    FuaFencePool, FuaStageTimings, FuaWalAppender, FuaWalSegment, FuaWalSegmentConfig,
};
#[cfg(unix)]
pub use wal_segment::{
    recover_wal_manager, recover_wal_manager_by_scan, WalDataSyncMode, WalManagerRecovery,
    WalPosition, WalPublishedBlock, WalSegmentManager, WalSegmentManagerConfig,
};
#[cfg(unix)]
pub use wal_segment::{
    recover_wal_segment, recover_wal_segment_by_scan, MappedWalSegment, WalRecovery,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::path::PathBuf;
    use std::sync::{Arc, Barrier};

    fn expected(first: u64, count: u64) -> DrainStats {
        let mut stats = DrainStats::default();
        for seq in first..first + count {
            stats.observe(intent_for(seq));
        }
        stats
    }

    #[cfg(unix)]
    fn temp_journal_path(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "gpu-db-write-conveyor-{name}-{}-{nonce}.dat",
            std::process::id()
        ))
    }

    #[test]
    fn spsc_cursor_wraps_and_preserves_payloads() {
        let ring = SpscCursorRing::with_capacity(8);
        let mut producer = ring.producer();
        let mut consumer = ring.consumer();
        let mut total = DrainStats::default();

        producer.publish_batch(0, 6);
        total.add(consumer.drain_available(4));
        producer.publish_batch(6, 6);
        while total.count < 12 {
            let drained = consumer.drain_available(8);
            assert!(
                drained.count > 0,
                "expected progress while draining test ring"
            );
            total.add(drained);
        }

        assert_eq!(total, expected(0, 12));
    }

    #[test]
    fn mpsc_sequenced_ring_preserves_all_claimed_payloads() {
        let ring = Arc::new(MpscSequencedRing::with_capacity(64));
        let producers = 4;
        let per_producer = 128_u64;
        let barrier = Arc::new(Barrier::new(producers + 1));
        let mut handles = Vec::new();
        for producer_id in 0..producers {
            let ring = Arc::clone(&ring);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let first = producer_id as u64 * per_producer;
                let mut sent = 0;
                while sent < per_producer {
                    let n = (per_producer - sent).min(7) as usize;
                    ring.publish_batch(first + sent, n);
                    sent += n as u64;
                }
            }));
        }
        barrier.wait();

        let mut consumer = ring.consumer();
        let mut total = DrainStats::default();
        while total.count < producers as u64 * per_producer {
            let drained = consumer.drain_available(16);
            if drained.count == 0 {
                spin_loop();
            } else {
                total.add(drained);
            }
        }
        for handle in handles {
            handle.join().unwrap();
        }

        let mut want = DrainStats::default();
        for producer_id in 0..producers {
            want.add(expected(producer_id as u64 * per_producer, per_producer));
        }
        assert_eq!(total, want);
    }

    #[test]
    fn mpsc_block_ring_wraps_partial_blocks() {
        let ring = Arc::new(MpscBlockRing::with_capacity(64, 8));
        let producers = 3;
        let per_producer = 43_u64;
        let barrier = Arc::new(Barrier::new(producers + 1));
        let mut handles = Vec::new();
        for producer_id in 0..producers {
            let ring = Arc::clone(&ring);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let first = producer_id as u64 * per_producer;
                let mut sent = 0;
                while sent < per_producer {
                    let n = (per_producer - sent).min(8) as usize;
                    ring.publish_block(first + sent, n);
                    sent += n as u64;
                }
            }));
        }
        barrier.wait();

        let mut consumer = ring.consumer();
        let mut total = DrainStats::default();
        while total.count < producers as u64 * per_producer {
            let drained = consumer.drain_available(4);
            if drained.count == 0 {
                spin_loop();
            } else {
                total.add(drained);
            }
        }
        for handle in handles {
            handle.join().unwrap();
        }

        let mut want = DrainStats::default();
        for producer_id in 0..producers {
            want.add(expected(producer_id as u64 * per_producer, per_producer));
        }
        assert_eq!(total, want);
    }

    #[test]
    fn completion_barrier_advances_only_contiguous_prefix() {
        let barrier = ContiguousCompletionBarrier::with_capacity(6);
        assert_eq!(barrier.capacity(), 6);
        assert_eq!(barrier.completed_prefix(), 0);

        assert_eq!(barrier.complete(2).unwrap(), 0);
        assert_eq!(barrier.complete(1).unwrap(), 0);
        assert_eq!(barrier.completed_prefix(), 0);
        assert_eq!(barrier.complete(0).unwrap(), 3);
        assert_eq!(barrier.completed_prefix(), 3);
        assert_eq!(barrier.complete(4).unwrap(), 3);
        assert_eq!(barrier.complete(3).unwrap(), 5);
        barrier.wait_for(5).unwrap();
        assert_eq!(barrier.completed_prefix(), 5);
    }

    #[test]
    fn completion_barrier_rejects_duplicate_and_bounds() {
        let barrier = ContiguousCompletionBarrier::with_capacity(2);
        barrier.complete(0).unwrap();
        assert_eq!(
            barrier.complete(0).unwrap_err(),
            CompletionError::AlreadyCompleted {
                block_id: 0,
                observed_sequence: 1,
            }
        );
        assert_eq!(
            barrier.complete(2).unwrap_err(),
            CompletionError::OutOfRange {
                block_id: 2,
                capacity: 2,
            }
        );
        assert_eq!(
            barrier.wait_for(3).unwrap_err(),
            CompletionError::OutOfRange {
                block_id: 3,
                capacity: 2,
            }
        );
        assert_eq!(
            barrier.wait_for_with_yield(3, 0).unwrap_err(),
            CompletionError::OutOfRange {
                block_id: 3,
                capacity: 2,
            }
        );
    }

    #[test]
    fn completion_barrier_yield_wait_observes_prefix() {
        let barrier = Arc::new(ContiguousCompletionBarrier::with_capacity(4));
        let waiter = {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || barrier.wait_for_with_yield(4, 1).unwrap())
        };
        for block_id in 0..4 {
            barrier.complete(block_id).unwrap();
        }
        waiter.join().unwrap();
        assert_eq!(barrier.completed_prefix(), 4);
    }

    #[test]
    fn completion_barrier_handles_concurrent_out_of_order_workers() {
        let blocks = 256_u64;
        let workers = 8;
        let barrier = Arc::new(ContiguousCompletionBarrier::with_capacity(blocks as usize));
        let start = Arc::new(Barrier::new(workers + 1));
        let mut handles = Vec::with_capacity(workers);
        for worker_id in 0..workers {
            let barrier = Arc::clone(&barrier);
            let start = Arc::clone(&start);
            handles.push(std::thread::spawn(move || {
                start.wait();
                for block_id in (worker_id as u64..blocks).step_by(workers) {
                    barrier.complete(blocks - block_id - 1).unwrap();
                }
            }));
        }
        start.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(barrier.completed_prefix(), blocks);
    }

    #[test]
    fn staged_conveyor_separates_logged_and_applied_prefixes() {
        let conveyor = StagedBlockConveyor::with_capacity(4);
        assert_eq!(conveyor.capacity(), 4);
        assert_eq!(conveyor.logged_prefix(), 0);
        assert_eq!(conveyor.applied_prefix(), 0);

        conveyor.mark_logged(1).unwrap();
        assert_eq!(conveyor.logged_prefix(), 0);
        conveyor.mark_logged(0).unwrap();
        conveyor.wait_logged(2).unwrap();
        assert_eq!(conveyor.logged_prefix(), 2);
        assert_eq!(conveyor.applied_prefix(), 0);

        conveyor.mark_applied(1).unwrap();
        assert_eq!(conveyor.applied_prefix(), 0);
        conveyor.mark_applied(0).unwrap();
        conveyor.wait_applied(2).unwrap();
        assert_eq!(conveyor.applied_prefix(), 2);
    }

    #[test]
    fn staged_conveyor_rejects_apply_before_logged_prefix() {
        let conveyor = StagedBlockConveyor::with_capacity(3);
        assert_eq!(
            conveyor.mark_applied(0).unwrap_err(),
            CompletionError::NotLogged {
                block_id: 0,
                logged_prefix: 0,
            }
        );
        assert_eq!(conveyor.applied_prefix(), 0);

        conveyor.mark_logged(1).unwrap();
        assert_eq!(conveyor.logged_prefix(), 0);
        assert_eq!(
            conveyor.mark_applied(1).unwrap_err(),
            CompletionError::NotLogged {
                block_id: 1,
                logged_prefix: 0,
            }
        );

        conveyor.mark_logged(0).unwrap();
        assert_eq!(conveyor.logged_prefix(), 2);
        conveyor.mark_applied(0).unwrap();
        conveyor.mark_applied(1).unwrap();
        assert_eq!(conveyor.applied_prefix(), 2);
    }

    #[test]
    fn staged_conveyor_handles_concurrent_journal_and_apply() {
        let blocks = 256_u64;
        let conveyor = Arc::new(StagedBlockConveyor::with_capacity(blocks as usize));
        let start = Arc::new(Barrier::new(3));

        let journal = {
            let conveyor = Arc::clone(&conveyor);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                for block_id in (0..blocks).rev() {
                    conveyor.mark_logged(block_id).unwrap();
                }
            })
        };
        let apply = {
            let conveyor = Arc::clone(&conveyor);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                for block_id in (0..blocks).rev() {
                    conveyor.wait_logged(block_id + 1).unwrap();
                    conveyor.mark_applied(block_id).unwrap();
                }
            })
        };

        start.wait();
        journal.join().unwrap();
        apply.join().unwrap();
        assert_eq!(conveyor.logged_prefix(), blocks);
        assert_eq!(conveyor.applied_prefix(), blocks);
    }

    #[test]
    fn open_shard_store_applies_blocks_out_of_order() {
        let store = OpenShardAppendStore::with_capacity(3, 4);
        assert_eq!(store.block_capacity(), 3);
        assert_eq!(store.row_capacity(), 12);

        let block1: Vec<_> = (4..8).map(intent_for).collect();
        let block0: Vec<_> = (0..4).map(intent_for).collect();
        let block2: Vec<_> = (8..10).map(intent_for).collect();

        assert_eq!(store.apply_block(1, &block1).unwrap(), expected(4, 4));
        assert_eq!(store.read_block_stats(0), None);
        assert_eq!(store.apply_block(0, &block0).unwrap(), expected(0, 4));
        assert_eq!(store.apply_block(2, &block2).unwrap(), expected(8, 2));

        assert_eq!(store.read_block_stats(0).unwrap(), expected(0, 4));
        assert_eq!(store.read_block_stats(1).unwrap(), expected(4, 4));
        assert_eq!(store.read_block_stats(2).unwrap(), expected(8, 2));
        assert_eq!(store.validate_applied_blocks(3).unwrap(), expected(0, 10));
        assert_eq!(store.read_row_id(8), Some(8));
        assert_eq!(store.read_created_seq(8), Some(8));
        assert_eq!(store.read_row_id(12), None);
    }

    #[test]
    fn open_shard_store_rejects_duplicate_bounds_and_oversized_blocks() {
        let store = OpenShardAppendStore::with_capacity(1, 2);
        let block: Vec<_> = (0..2).map(intent_for).collect();
        store.apply_block(0, &block).unwrap();
        assert_eq!(
            store.apply_block(0, &block).unwrap_err(),
            StoreApplyError::DuplicateBlock {
                block_id: 0,
                observed_state: 2,
            }
        );
        assert_eq!(
            store.apply_block(1, &block).unwrap_err(),
            StoreApplyError::BlockOutOfRange {
                block_id: 1,
                block_capacity: 1,
            }
        );
        let oversized: Vec<_> = (0..3).map(intent_for).collect();
        assert_eq!(
            store.apply_block(0, &oversized).unwrap_err(),
            StoreApplyError::CountExceedsBlockSize {
                count: 3,
                block_size: 2,
            }
        );
    }

    #[test]
    fn open_shard_store_uses_global_block_ordinals_across_segments() {
        let blocks_per_segment = 2;
        let store = OpenShardAppendStore::with_capacity(4, 4);
        let seg0_block1 = OpenShardAppendStore::global_block_id(0, blocks_per_segment, 1);
        let seg1_block0 = OpenShardAppendStore::global_block_id(1, blocks_per_segment, 0);

        let block1: Vec<_> = (4..8).map(intent_for).collect();
        let block2: Vec<_> = (8..12).map(intent_for).collect();
        store.apply_block(seg0_block1, &block1).unwrap();
        store.apply_block(seg1_block0, &block2).unwrap();

        assert_eq!(store.read_block_stats(seg0_block1).unwrap(), expected(4, 4));
        assert_eq!(store.read_block_stats(seg1_block0).unwrap(), expected(8, 4));
        assert_eq!(store.read_row_id(4), Some(4));
        assert_eq!(store.read_row_id(8), Some(8));
        assert_eq!(store.read_created_seq(8), Some(8));
    }

    #[test]
    fn open_shard_store_validation_checks_row_ids() {
        let store = OpenShardAppendStore::with_capacity(1, 4);
        let block: Vec<_> = (0..4).map(intent_for).collect();
        store.apply_block(0, &block).unwrap();
        assert_eq!(store.validate_applied_blocks(1).unwrap(), expected(0, 4));

        unsafe {
            write_column(&store.row_id, 2, 99);
        }
        assert_eq!(store.validate_applied_blocks(1), None);
    }

    #[cfg(unix)]
    #[test]
    fn mapped_journal_round_trips_payloads() {
        let path = temp_journal_path("mapped-round-trip");
        let _ = std::fs::remove_file(&path);
        {
            let mut journal = unsafe { MappedJournal::create(&path, 16).unwrap() };
            journal.write(0, intent_for(10));
            journal.write(15, intent_for(25));

            assert_eq!(journal.read(0), intent_for(10));
            assert_eq!(journal.read(15), intent_for(25));
        }
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn mapped_block_journal_drains_published_blocks() {
        let path = temp_journal_path("mapped-block");
        let _ = std::fs::remove_file(&path);
        {
            let journal = unsafe { MappedBlockJournal::create(&path, 8, 4).unwrap() };
            journal.try_publish_block(0, 4).unwrap();
            journal.try_publish_block(4, 3).unwrap();

            let mut total = DrainStats::default();
            total.add(journal.read_published_block(1).unwrap());
            total.add(journal.read_published_block(0).unwrap());

            assert_eq!(total, expected(0, 7));
        }
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn mapped_block_journal_reports_publish_bounds() {
        let path = temp_journal_path("mapped-bounds");
        let _ = std::fs::remove_file(&path);
        {
            let journal = unsafe { MappedBlockJournal::create(&path, 8, 4).unwrap() };
            assert_eq!(
                journal.try_publish_block(0, 5),
                Err(PublishError::CountExceedsBlockSize {
                    count: 5,
                    block_size: 4,
                })
            );
            journal.try_publish_block(0, 4).unwrap();
            journal.try_publish_block(4, 4).unwrap();
            assert_eq!(
                journal.try_publish_block(8, 1),
                Err(PublishError::CapacityExhausted {
                    claimed_block: 2,
                    block_capacity: 2,
                })
            );
        }
        let _ = std::fs::remove_file(&path);
    }
}
